/* Matrix products of the exchange build on the device
   (source/Exact_Exchange_Device.c), run against a model of the device
   libraries: tests/fixtures/exx_device_sdk declares the calls and this file
   defines them. Device memory is host memory that the model keeps apart,
   counts and fills with NaN, every call can be made to fail, another
   process can take device memory between the query and the allocations, and
   the model notes a call that breaks its rules (a host pointer where a
   device pointer belongs, a copy past the end of an allocation, a handle
   used or given back on another device than its own).

   The test asks five things of the unit: a product on the device equals the
   product of the host, a product that the device does not take leaves every
   array and the device untouched, a device without room for a product gives
   all device memory back and stays in use, a device call that fails for
   another reason gives all device memory back, clears the error of the
   runtime and keeps the device out of the rest of the run, and a device
   that was asked for GEMMul8 sends its products through the bridge and
   counts the ones that the bridge did not compute in FP64.

   The model of the bridge (openmx_gemmul8Dgemm) computes the same product
   as the model of the BLAS. It counts its calls that ran in FP64, as the
   bridge of the main program does, and a case chooses which calls those
   are. */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "openmx_gpu_backend.h"
#include "openmx_gpu_runtime.h"
#include "openmx_gpu_blas.h"
#include "openmx_gpu_device.h"
#include "openmx_exact_exchange_device.h"

enum {
  CALL_GET_DEVICE, CALL_SET_DEVICE, CALL_CREATE, CALL_DESTROY, CALL_QUERY, CALL_MALLOC, CALL_FREE, CALL_COPY,
  CALL_DGEMM, CALL_SYNCHRONIZE, CALL_LAST_ERROR, CALL_BRIDGE, CALL_KINDS
};

enum { LIVE_SLOTS = 8 };

struct OpenMXTestBlasContext { int device; };

typedef struct {
  int usable;                 /* answer of gpu_rank_device_usable */
  int usable_calls;
  int current_device;
  size_t capacity_bytes;      /* device memory, free when nothing is allocated */
  int count[CALL_KINDS];
  size_t taken_after_query;   /* device memory another process takes right after the next query */
  int fail_kind,fail_at;      /* the fail_at-th call of fail_kind fails; fail_at 0: none */
  int fail_status;            /* status of that failure; 0: the one the call usually fails with */
  int failed_status;          /* status the failed call returned */
  int latched;                /* an error waits in cudaGetLastError */
  void *live[LIVE_SLOTS];
  size_t live_bytes[LIVE_SLOTS];
  int live_count;
  size_t live_total,peak_total;
  int handles;                /* handles alive */
  int bridge_native;          /* the bridge computes its products in FP64 */
  long long native_calls;     /* calls of the bridge that ran in FP64 */
  int wrong;                  /* calls that broke the rules of the model */
} Model;

static Model model;

static void check(int ok, const char *message)
{
  if (!ok) {
    fprintf(stderr,"exx device product: %s\n",message);
    exit(1);
  }
}

static void model_reset(void)
{
  check(model.live_count==0 && model.handles==0,"the model is reset with device memory or a handle alive");
  memset(&model,0,sizeof(model));
  model.usable = 1;
  model.capacity_bytes = (size_t)1<<20;
}

/* The status of this call of the kind: 0, or the one it is to fail with,
   which is `usual` unless the case asked for another. */
static int model_fails(int kind, int usual)
{
  model.count[kind]++;
  if (model.fail_at==0 || model.fail_kind!=kind || model.count[kind]!=model.fail_at) return 0;
  model.failed_status = model.fail_status!=0 ? model.fail_status : usual;
  return model.failed_status;
}

static int model_slot(const void *pointer)
{
  for (int i=0; i<LIVE_SLOTS; i++)
    if (model.live[i]!=NULL && model.live[i]==pointer) return i;
  return -1;
}

int gpu_rank_device_usable(void)
{
  model.usable_calls++;
  return model.usable;
}

cudaError_t OpenMX_GpuMemGetInfo(size_t *free_bytes, size_t *total_bytes)
{
  const int failed=model_fails(CALL_QUERY,cudaErrorUnknown);
  if (failed) {
    model.latched = 1;
    return failed;
  }
  *free_bytes = model.capacity_bytes-model.live_total;
  *total_bytes = model.capacity_bytes;
  /* the answer is already old: another process allocates */
  model.capacity_bytes -= model.taken_after_query;
  model.taken_after_query = 0;
  return cudaSuccess;
}

cudaError_t cudaGetDevice(int *device)
{
  const int failed=model_fails(CALL_GET_DEVICE,cudaErrorNoDevice);
  if (failed) {
    model.latched = 1;
    return failed;
  }
  *device = model.current_device;
  return cudaSuccess;
}

cudaError_t cudaSetDevice(int device)
{
  model.count[CALL_SET_DEVICE]++;
  model.current_device = device;
  return cudaSuccess;
}

cudaError_t cudaMalloc(void **pointer, size_t bytes)
{
  int failed=model_fails(CALL_MALLOC,cudaErrorUnknown);
  *pointer = NULL;
  if (!failed && model.capacity_bytes-model.live_total<bytes) failed = model.failed_status = cudaErrorMemoryAllocation;
  if (failed) {
    model.latched = 1;
    return failed;
  }
  int slot=-1;
  for (int i=0; i<LIVE_SLOTS && slot<0; i++)
    if (model.live[i]==NULL) slot = i;
  check(0<=slot && 0<bytes && bytes%sizeof(double)==0,"the model has no slot for an allocation of this size");
  double *memory=malloc(bytes);
  check(memory!=NULL,"the host of the model is out of memory");
  /* nothing on the device has a value before a copy or a product wrote it */
  for (size_t i=0; i<bytes/sizeof(double); i++) memory[i] = NAN;
  model.live[slot] = memory;
  model.live_bytes[slot] = bytes;
  model.live_count++;
  model.live_total += bytes;
  if (model.peak_total<model.live_total) model.peak_total = model.live_total;
  *pointer = memory;
  return cudaSuccess;
}

cudaError_t cudaFree(void *pointer)
{
  model.count[CALL_FREE]++;
  if (pointer==NULL) return cudaSuccess;
  const int slot=model_slot(pointer);
  if (slot<0) {
    model.wrong++;
    return cudaErrorInvalidValue;
  }
  model.live_total -= model.live_bytes[slot];
  model.live_count--;
  model.live[slot] = NULL;
  model.live_bytes[slot] = 0;
  free(pointer);
  return cudaSuccess;
}

cudaError_t cudaMemcpy(void *destination, const void *source, size_t bytes, enum cudaMemcpyKind kind)
{
  const int to_device=kind==cudaMemcpyHostToDevice;
  const int device_slot=model_slot(to_device ? destination : source);
  const int host_slot=model_slot(to_device ? source : destination);
  if ((kind!=cudaMemcpyHostToDevice && kind!=cudaMemcpyDeviceToHost) || device_slot<0 || 0<=host_slot
      || model.live_bytes[device_slot]<bytes) {
    model.count[CALL_COPY]++;
    model.wrong++;
    return cudaErrorInvalidValue;
  }
  const int failed=model_fails(CALL_COPY,cudaErrorUnknown);
  if (failed) {
    /* a transfer that breaks off has written a part of its target */
    memset(destination,0x5a,bytes/2);
    model.latched = 1;
    return failed;
  }
  memcpy(destination,source,bytes);
  return cudaSuccess;
}

cudaError_t cudaDeviceSynchronize(void)
{
  const int failed=model_fails(CALL_SYNCHRONIZE,cudaErrorUnknown);
  if (failed) {
    model.latched = 1;
    return failed;
  }
  return cudaSuccess;
}

cudaError_t cudaGetLastError(void)
{
  const cudaError_t pending=model.latched ? cudaErrorUnknown : cudaSuccess;
  model.count[CALL_LAST_ERROR]++;
  model.latched = 0;
  return pending;
}

cublasStatus_t cublasCreate(cublasHandle_t *handle)
{
  const int failed=model_fails(CALL_CREATE,CUBLAS_STATUS_NOT_INITIALIZED);
  *handle = NULL;
  if (failed) return failed;
  struct OpenMXTestBlasContext *context=malloc(sizeof(*context));
  check(context!=NULL,"the host of the model is out of memory");
  context->device = model.current_device;
  model.handles++;
  *handle = context;
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDestroy(cublasHandle_t handle)
{
  model.count[CALL_DESTROY]++;
  if (handle==NULL) {
    model.wrong++;
    return CUBLAS_STATUS_NOT_INITIALIZED;
  }
  /* a handle goes back on the device that made it */
  if (handle->device!=model.current_device) model.wrong++;
  model.handles--;
  free(handle);
  return CUBLAS_STATUS_SUCCESS;
}

/* element (i,j) of op(x), x column-major with rows per column */
static double element(const double *x, int rows, int transposed, int i, int j)
{
  return transposed ? x[(size_t)i*rows+j] : x[(size_t)j*rows+i];
}

/* c = alpha*op(a)*op(b) + beta*c with the plain triple loop; c is not read
   for beta = 0 */
static void reference_dgemm(int transposed_a, int transposed_b, int m, int n, int k, double alpha, const double *a,
                            int lda, const double *b, int ldb, double beta, double *c, int ldc)
{
  for (int j=0; j<n; j++) {
    for (int i=0; i<m; i++) {
      double sum=0.0;
      for (int l=0; l<k; l++) sum += element(a,lda,transposed_a,i,l)*element(b,ldb,transposed_b,l,j);
      c[(size_t)j*ldc+i] = alpha*sum+(beta==0.0 ? 0.0 : beta*c[(size_t)j*ldc+i]);
    }
  }
}

/* A product of the device BLAS (CALL_DGEMM) or of the bridge (CALL_BRIDGE):
   both take a handle of the current device and three matrices on it. */
static cublasStatus_t model_product(int kind, cublasHandle_t handle, cublasOperation_t transa,
                                    cublasOperation_t transb, int m, int n, int k, const double *alpha,
                                    const double *a, int lda, const double *b, int ldb, const double *beta,
                                    double *c, int ldc)
{
  const int rows_a=transa==CUBLAS_OP_N ? m : k,columns_a=transa==CUBLAS_OP_N ? k : m;
  const int rows_b=transb==CUBLAS_OP_N ? k : n,columns_b=transb==CUBLAS_OP_N ? n : k;
  const int slot_a=model_slot(a),slot_b=model_slot(b),slot_c=model_slot(c);
  if (handle==NULL || handle->device!=model.current_device || slot_a<0 || slot_b<0 || slot_c<0
      || (transa!=CUBLAS_OP_N && transa!=CUBLAS_OP_T) || (transb!=CUBLAS_OP_N && transb!=CUBLAS_OP_T)
      || lda<rows_a || ldb<rows_b || ldc<m
      || model.live_bytes[slot_a]<sizeof(double)*(size_t)lda*(size_t)columns_a
      || model.live_bytes[slot_b]<sizeof(double)*(size_t)ldb*(size_t)columns_b
      || model.live_bytes[slot_c]<sizeof(double)*(size_t)ldc*(size_t)n) {
    model.count[kind]++;
    model.wrong++;
    return CUBLAS_STATUS_INVALID_VALUE;
  }
  const int failed=model_fails(kind,CUBLAS_STATUS_EXECUTION_FAILED);
  if (failed) return failed;
  reference_dgemm(transa==CUBLAS_OP_T,transb==CUBLAS_OP_T,m,n,k,*alpha,a,lda,b,ldb,*beta,c,ldc);
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n,
                           int k, const double *alpha, const double *a, int lda, const double *b, int ldb,
                           const double *beta, double *c, int ldc)
{
  return model_product(CALL_DGEMM,handle,transa,transb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc);
}

/* The bridge of the main program, declared in source/Exact_Exchange_Device.c.
   A call that the bridge computes in FP64 is counted; a call that fails is
   not. */
cublasStatus_t openmx_gemmul8Dgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m,
                                   int n, int k, const double *alpha, const double *a, int lda, const double *b,
                                   int ldb, const double *beta, double *c, int ldc);
long long openmx_gemmul8NativeCalls(int is_complex);

cublasStatus_t openmx_gemmul8Dgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m,
                                   int n, int k, const double *alpha, const double *a, int lda, const double *b,
                                   int ldb, const double *beta, double *c, int ldc)
{
  const cublasStatus_t status=model_product(CALL_BRIDGE,handle,transa,transb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc);
  if (status==CUBLAS_STATUS_SUCCESS && model.bridge_native) model.native_calls++;
  return status;
}

long long openmx_gemmul8NativeCalls(int is_complex)
{
  /* the products of the exchange build are real */
  if (is_complex) model.wrong++;
  return model.native_calls;
}

/* ---- the cases -------------------------------------------------------- */

enum { GUARD = 3 };
static const double untouched=-731.0;

typedef struct {
  char transa,transb;
  int m,n,k,lda,ldb;
  double alpha;
  double *a,*b,*c,*expected;   /* c and expected have GUARD cells before and after */
} Product;

static double sample(unsigned *seed)
{
  *seed = *seed*1664525u+1013904223u;
  return (double)(*seed>>8)/(double)(1u<<24)-0.5;
}

static void product_make(Product *p, char transa, char transb, int m, int n, int k, unsigned seed)
{
  p->transa = transa;
  p->transb = transb;
  p->m = m;
  p->n = n;
  p->k = k;
  p->lda = transa=='N' ? m : k;
  p->ldb = transb=='N' ? k : n;
  p->alpha = -0.25;
  const size_t elements_a=(size_t)p->lda*(size_t)(transa=='N' ? k : m);
  const size_t elements_b=(size_t)p->ldb*(size_t)(transb=='N' ? n : k);
  const size_t elements_c=(size_t)m*(size_t)n+2*GUARD;
  p->a = malloc(sizeof(double)*elements_a);
  p->b = malloc(sizeof(double)*elements_b);
  p->c = malloc(sizeof(double)*elements_c);
  p->expected = malloc(sizeof(double)*elements_c);
  check(p->a!=NULL && p->b!=NULL && p->c!=NULL && p->expected!=NULL,"the test is out of memory");
  for (size_t i=0; i<elements_a; i++) p->a[i] = sample(&seed);
  for (size_t i=0; i<elements_b; i++) p->b[i] = sample(&seed);
  for (size_t i=0; i<elements_c; i++) p->c[i] = p->expected[i] = untouched;
  reference_dgemm(transa=='T',transb=='T',m,n,k,p->alpha,p->a,p->lda,p->b,p->ldb,0.0,p->expected+GUARD,m);
}

static void product_free(Product *p)
{
  free(p->a);
  free(p->b);
  free(p->c);
  free(p->expected);
}

static size_t product_bytes(const Product *p)
{
  return sizeof(double)*((size_t)p->m*(size_t)p->k+(size_t)p->k*(size_t)p->n+(size_t)p->m*(size_t)p->n);
}

static int product_run(OpenMXExactExchangeDevice *device, Product *p, double beta)
{
  return Exact_Exchange_Device_Dgemm(device,p->transa,p->transb,p->m,p->n,p->k,p->alpha,p->a,p->lda,p->b,p->ldb,
                                     beta,p->c+GUARD,p->m);
}

static int product_matches(const Product *p)
{
  return memcmp(p->c,p->expected,sizeof(double)*((size_t)p->m*(size_t)p->n+2*GUARD))==0;
}

static int product_untouched(const Product *p)
{
  for (size_t i=0; i<(size_t)p->m*(size_t)p->n+2*GUARD; i++)
    if (p->c[i]!=untouched) return 0;
  return 1;
}

static int model_silent(void)
{
  for (int kind=0; kind<CALL_KINDS; kind++)
    if (model.count[kind]!=0) return 0;
  return 1;
}

static void device_make(OpenMXExactExchangeDevice *device)
{
  Exact_Exchange_Device_Initialize(device);
  check(device->blas==NULL && !device->disabled && device->failure==OPENMX_EXACT_EXCHANGE_DEVICE_OK
        && 0.0<device->minimum_work && 0<device->reserve_bytes && device->device_products==0
        && device->declined_products==0 && device->largest_bytes==0 && !device->gemmul8
        && device->gemmul8_products==0,
        "a new device does not have the values of a new calculation");
  /* the matrices of the test are small */
  device->minimum_work = 1.0;
  device->reserve_bytes = 4096;
}

static void products_on_the_device(void)
{
  const char modes[4][2]={{'N','N'},{'T','N'},{'N','T'},{'T','T'}};
  OpenMXExactExchangeDevice device;
  size_t largest=0;
  model_reset();
  device_make(&device);
  for (int i=0; i<4; i++) {
    Product p;
    product_make(&p,modes[i][0],modes[i][1],5+i,3+2*i,4+i,17u+(unsigned)i);
    check(product_run(&device,&p,0.0)==1,"a product that fits was not taken by the device");
    check(product_matches(&p),"the product of the device differs from the product of the host");
    check(model.live_count==0 && model.wrong==0 && !model.latched,"a product left device memory or broke the model");
    check(model.count[CALL_MALLOC]==3*(i+1) && model.count[CALL_COPY]==3*(i+1) && model.count[CALL_DGEMM]==i+1,
          "a product did not make three allocations, three copies and one product");
    check(model.peak_total==(largest<product_bytes(&p) ? product_bytes(&p) : largest),
          "a product took more device memory than its three matrices");
    if (largest<product_bytes(&p)) largest = product_bytes(&p);
    product_free(&p);
  }
  check(model.count[CALL_CREATE]==1 && model.handles==1,"the handle of the BLAS was not kept between the products");
  check(device.device_products==4 && device.declined_products==0 && device.largest_bytes==largest
        && !device.disabled,"the counters of four products are wrong");
  check(model.count[CALL_BRIDGE]==0 && device.gemmul8_products==0,
        "a device that was not asked for GEMMul8 used the bridge");

  check(model.count[CALL_SET_DEVICE]==0,"a process that stays on its device changed the binding");

  /* another device: the old handle goes back first, on its own device */
  {
    Product p;
    model.current_device = 1;
    product_make(&p,'N','N',4,4,4,5u);
    check(product_run(&device,&p,0.0)==1 && product_matches(&p),"the product on the second device is wrong");
    check(model.count[CALL_CREATE]==2 && model.count[CALL_DESTROY]==1 && model.handles==1 && model.wrong==0
          && device.blas_device==1,"the handle did not follow the device");
    check(model.current_device==1 && model.count[CALL_SET_DEVICE]==2,
          "the binding of the process was not restored after the old handle went back");
    check(device.largest_bytes==largest,"a smaller product lowered the largest device memory of a product");
    product_free(&p);
  }

  /* the release finds the process on another device than the handle */
  model.current_device = 0;
  Exact_Exchange_Device_Release(&device);
  check(device.blas==NULL && model.handles==0 && model.count[CALL_DESTROY]==2,"the release kept the handle");
  check(model.wrong==0 && model.current_device==0 && model.count[CALL_SET_DEVICE]==4,
        "the release did not give the handle back on its device or did not restore the binding");
  {
    const Model before=model;
    Exact_Exchange_Device_Release(&device);
    check(memcmp(&before,&model,sizeof(model))==0 && device.device_products==5,
          "a second release touched the device or the counters");
  }

  /* the release finds the process on the device of the handle: the binding
     is left alone */
  {
    Product p;
    product_make(&p,'N','N',4,4,4,6u);
    check(product_run(&device,&p,0.0)==1 && model.handles==1,"the device did not take a product after a release");
    Exact_Exchange_Device_Release(&device);
    check(model.handles==0 && model.wrong==0 && model.count[CALL_SET_DEVICE]==4,
          "a release on the device of the handle changed the binding");
    product_free(&p);
  }

  /* the release cannot read the binding: the handle still goes back */
  {
    Product p;
    product_make(&p,'N','N',4,4,4,6u);
    check(product_run(&device,&p,0.0)==1 && model.handles==1,"the device did not take a product after a release");
    model.fail_kind = CALL_GET_DEVICE;
    model.fail_at = model.count[CALL_GET_DEVICE]+1;
    Exact_Exchange_Device_Release(&device);
    check(device.blas==NULL && model.handles==0 && model.wrong==0 && !model.latched,
          "a release that cannot read the binding kept the handle or an error");
    product_free(&p);
  }
}

static void products_left_to_the_host(void)
{
  OpenMXExactExchangeDevice device;
  Product p;
  model_reset();
  device_make(&device);
  product_make(&p,'N','T',6,5,4,3u);

  check(Exact_Exchange_Device_Dgemm(NULL,p.transa,p.transb,p.m,p.n,p.k,p.alpha,p.a,p.lda,p.b,p.ldb,0.0,p.c+GUARD,
                                    p.m)==0,"a product without a device was taken");
  check(product_run(&device,&p,0.5)==0,"a product with beta other than 0 was taken");
  device.minimum_work = 121.0;
  check(product_run(&device,&p,0.0)==0,"a product below the smallest size was taken");
  device.minimum_work = 120.0;
  {
    /* 'C' with the arrays of a transposed factor: only the letter is wrong */
    Product q;
    product_make(&q,'T','T',6,5,4,7u);
    check(Exact_Exchange_Device_Dgemm(&device,'C',q.transb,q.m,q.n,q.k,q.alpha,q.a,q.lda,q.b,q.ldb,0.0,q.c+GUARD,
                                      q.m)==0,"a conjugated first factor was taken");
    check(Exact_Exchange_Device_Dgemm(&device,q.transa,'C',q.m,q.n,q.k,q.alpha,q.a,q.lda,q.b,q.ldb,0.0,q.c+GUARD,
                                      q.m)==0,"a conjugated second factor was taken");
    check(product_untouched(&q),"a product with a conjugated factor wrote the result");
    product_free(&q);
  }
  /* no smallest size: an empty product still stays on the host */
  device.minimum_work = 0.0;
  check(Exact_Exchange_Device_Dgemm(&device,p.transa,p.transb,0,p.n,p.k,p.alpha,p.a,p.lda,p.b,p.ldb,0.0,p.c+GUARD,
                                    1)==0,"a product without rows was taken");
  check(Exact_Exchange_Device_Dgemm(&device,p.transa,p.transb,p.m,0,p.k,p.alpha,p.a,p.lda,p.b,1,0.0,p.c+GUARD,
                                    p.m)==0,"a product without columns was taken");
  check(Exact_Exchange_Device_Dgemm(&device,p.transa,p.transb,p.m,p.n,0,p.alpha,p.a,p.lda,p.b,p.ldb,0.0,p.c+GUARD,
                                    p.m)==0,"a product without inner index was taken");
  device.minimum_work = 120.0;
  {
    /* one more row than the product has, in one of the three arrays at a
       time; every array is large enough for the wider column */
    double *wide_a=calloc((size_t)(p.m+1)*(size_t)p.k,sizeof(double));
    double *wide_b=calloc((size_t)(p.n+1)*(size_t)p.k,sizeof(double));
    double *wide_c=calloc((size_t)(p.m+1)*(size_t)p.n,sizeof(double));
    check(wide_a!=NULL && wide_b!=NULL && wide_c!=NULL,"the test is out of memory");
    check(Exact_Exchange_Device_Dgemm(&device,p.transa,p.transb,p.m,p.n,p.k,p.alpha,wide_a,p.lda+1,p.b,p.ldb,0.0,
                                      p.c+GUARD,p.m)==0,"a first factor with padded columns was taken");
    check(Exact_Exchange_Device_Dgemm(&device,p.transa,p.transb,p.m,p.n,p.k,p.alpha,p.a,p.lda,wide_b,p.ldb+1,0.0,
                                      p.c+GUARD,p.m)==0,"a second factor with padded columns was taken");
    check(Exact_Exchange_Device_Dgemm(&device,p.transa,p.transb,p.m,p.n,p.k,p.alpha,p.a,p.lda,p.b,p.ldb,0.0,
                                      wide_c,p.m+1)==0,"a result with padded columns was taken");
    for (size_t i=0; i<(size_t)(p.m+1)*(size_t)p.n; i++)
      check(wide_c[i]==0.0,"a result with padded columns was written");
    free(wide_a);
    free(wide_b);
    free(wide_c);
  }
  check(model.usable_calls==0,"a product that the device cannot take asked for the device");
  model.usable = 0;
  check(product_run(&device,&p,0.0)==0 && model.usable_calls==1,"a process without a usable device took a product");
  model.usable = 1;
  device.disabled = 1;
  check(product_run(&device,&p,0.0)==0 && model.usable_calls==1,"a device that is switched off took a product");
  device.disabled = 0;
  check(model_silent() && product_untouched(&p) && device.device_products==0 && device.declined_products==0
        && device.blas==NULL,"a product that was not taken touched the device, the result or the counters");

  Exact_Exchange_Device_Release(&device);
  check(model_silent(),"the release of a device without a handle touched the device");
  product_free(&p);
}

/* What a product without room has to leave behind: nothing on the device,
   no error, an untouched result, one more declined product and a device
   that is still in use. */
static void declined(const OpenMXExactExchangeDevice *device, const Product *p, long long count, const char *message)
{
  check(device->declined_products==count && !device->disabled && device->failure==OPENMX_EXACT_EXCHANGE_DEVICE_OK
        && device->device_products==0 && device->largest_bytes==0,message);
  check(model.live_count==0 && model.wrong==0 && !model.latched && model.count[CALL_COPY]==0
        && model.count[CALL_DGEMM]==0 && product_untouched(p),message);
}

static void products_without_room(void)
{
  OpenMXExactExchangeDevice device;
  Product p;
  model_reset();
  device_make(&device);
  product_make(&p,'N','T',6,5,4,3u);
  const size_t bytes_a=sizeof(double)*(size_t)p.m*(size_t)p.k,bytes_b=sizeof(double)*(size_t)p.n*(size_t)p.k;
  const size_t room=product_bytes(&p)+device.reserve_bytes;

  /* the three matrices and the reserve have to fit; the device is asked
     for its memory and for nothing else */
  model.capacity_bytes = room-1;
  check(product_run(&device,&p,0.0)==0,"a product that does not fit was taken");
  declined(&device,&p,1,"a product that does not fit was not declined cleanly");
  check(model.count[CALL_QUERY]==1 && model.count[CALL_MALLOC]==0 && model.count[CALL_CREATE]==0
        && device.blas==NULL,"a product that does not fit reached the device");

  /* the setting of the fault injection (Exact_Exchange_Configure): no
     product fits, whatever the device has */
  {
    const size_t reserve=device.reserve_bytes;
    model.capacity_bytes = (size_t)1<<40;
    device.reserve_bytes = (size_t)-1;
    check(product_run(&device,&p,0.0)==0,"a product was taken although the reserve is all memory");
    declined(&device,&p,2,"a product was not declined cleanly with a reserve of all memory");
    check(model.count[CALL_MALLOC]==0 && model.count[CALL_CREATE]==0,"a reserve of all memory reached the device");
    device.reserve_bytes = reserve;
  }

  /* the handle cannot get its memory */
  model.capacity_bytes = room;
  model.fail_kind = CALL_CREATE;
  model.fail_at = 1;
  model.fail_status = CUBLAS_STATUS_ALLOC_FAILED;
  check(product_run(&device,&p,0.0)==0,"a product was taken without a handle");
  declined(&device,&p,3,"a handle without memory was not treated as a product without room");
  check(device.blas==NULL && model.handles==0 && model.count[CALL_MALLOC]==0,
        "a handle without memory left a handle or reached the allocations");

  /* another process takes memory between the query and the allocations: the
     first, the second or the third allocation runs out */
  {
    const size_t left[3]={bytes_a-1,bytes_a+bytes_b-1,product_bytes(&p)-1};
    for (int i=0; i<3; i++) {
      const int allocations=model.count[CALL_MALLOC];
      model.capacity_bytes = room;
      model.taken_after_query = room-left[i];
      check(product_run(&device,&p,0.0)==0,"a product was taken although the device ran out of memory");
      declined(&device,&p,4+i,"a device that ran out of memory was not treated as a product without room");
      check(model.count[CALL_MALLOC]==allocations+i+1 && model.failed_status==cudaErrorMemoryAllocation,
            "the device did not run out of memory at the allocation of the case");
    }
  }
  check(model.count[CALL_CREATE]==2 && model.handles==1,"the handle was not kept over the products without room");

  /* the memory is back: the device takes the same product */
  model.capacity_bytes = room;
  check(product_run(&device,&p,0.0)==1 && product_matches(&p) && device.device_products==1
        && device.declined_products==6 && !device.disabled,"a product that just fits was not taken");
  check(model.wrong==0 && model.live_count==0 && model.peak_total==product_bytes(&p),
        "the product that just fits broke the model");

  /* the BLAS cannot get memory of its own for a product */
  {
    Product q;
    product_make(&q,'T','N',6,5,4,9u);
    model.fail_kind = CALL_DGEMM;
    model.fail_at = model.count[CALL_DGEMM]+1;
    model.fail_status = CUBLAS_STATUS_ALLOC_FAILED;
    check(product_run(&device,&q,0.0)==0 && product_untouched(&q),"a product was taken without memory for the BLAS");
    check(device.declined_products==7 && device.device_products==1 && !device.disabled
          && device.failure==OPENMX_EXACT_EXCHANGE_DEVICE_OK && model.live_count==0 && model.wrong==0
          && !model.latched,"a BLAS without memory was not treated as a product without room");
    check(product_run(&device,&q,0.0)==1 && product_matches(&q) && device.device_products==2,
          "the device did not take the product after the BLAS had run out of memory");
    product_free(&q);
  }

  Exact_Exchange_Device_Release(&device);
  product_free(&p);
}

static void device_calls_that_fail(void)
{
  /* the call that fails, with a status that does not mean a lack of
     memory, the step the unit has to name, and whether the result array may
     have been written */
  const struct { int kind,at,failure,result_written; } sites[]={
    {CALL_QUERY,1,OPENMX_EXACT_EXCHANGE_DEVICE_MEMORY_QUERY,0},
    {CALL_GET_DEVICE,1,OPENMX_EXACT_EXCHANGE_DEVICE_HANDLE,0},
    {CALL_CREATE,1,OPENMX_EXACT_EXCHANGE_DEVICE_HANDLE,0},
    {CALL_MALLOC,1,OPENMX_EXACT_EXCHANGE_DEVICE_ALLOCATION,0},
    {CALL_MALLOC,2,OPENMX_EXACT_EXCHANGE_DEVICE_ALLOCATION,0},
    {CALL_MALLOC,3,OPENMX_EXACT_EXCHANGE_DEVICE_ALLOCATION,0},
    {CALL_COPY,1,OPENMX_EXACT_EXCHANGE_DEVICE_UPLOAD,0},
    {CALL_COPY,2,OPENMX_EXACT_EXCHANGE_DEVICE_UPLOAD,0},
    {CALL_DGEMM,1,OPENMX_EXACT_EXCHANGE_DEVICE_PRODUCT,0},
    {CALL_SYNCHRONIZE,1,OPENMX_EXACT_EXCHANGE_DEVICE_PRODUCT,0},
    {CALL_COPY,3,OPENMX_EXACT_EXCHANGE_DEVICE_DOWNLOAD,1}};
  for (size_t site=0; site<sizeof(sites)/sizeof(sites[0]); site++) {
    OpenMXExactExchangeDevice device;
    Product p;
    model_reset();
    device_make(&device);
    product_make(&p,'T','N',7,4,5,29u+(unsigned)site);
    model.fail_kind = sites[site].kind;
    model.fail_at = sites[site].at;

    check(product_run(&device,&p,0.0)==0,"a product was reported although a device call failed");
    check(device.disabled && device.failure==sites[site].failure && device.failure_status==model.failed_status
          && device.failure_status!=0,"the failed call is not recorded as it happened");
    check(model.live_count==0 && model.wrong==0,"a failed product left device memory or broke the model");
    check(!model.latched && 0<model.count[CALL_LAST_ERROR],"a failed product left the error of the runtime behind");
    check(device.device_products==0 && device.declined_products==0 && device.largest_bytes==0,
          "a failed product was counted");
    if (sites[site].failure==OPENMX_EXACT_EXCHANGE_DEVICE_MEMORY_QUERY
        || sites[site].failure==OPENMX_EXACT_EXCHANGE_DEVICE_HANDLE)
      check(device.blas==NULL && model.handles==0 && model.count[CALL_MALLOC]==0,
            "a product that failed before it had a handle made one or reached the allocations");
    if (!sites[site].result_written)
      check(product_untouched(&p),"a product that failed before the download wrote the result");

    /* the host computes this product and all later ones */
    reference_dgemm(p.transa=='T',p.transb=='T',p.m,p.n,p.k,p.alpha,p.a,p.lda,p.b,p.ldb,0.0,p.c+GUARD,p.m);
    check(product_matches(&p),"the host could not repeat a product that failed on the device");
    {
      const Model before=model;
      check(product_run(&device,&p,0.0)==0 && memcmp(&before,&model,sizeof(model))==0 && product_matches(&p),
            "a device that failed was asked again");
    }
    check(Exact_Exchange_Device_Failure_Name(device.failure)[0]!='\0',"a failure has no name");

    Exact_Exchange_Device_Release(&device);
    check(model.handles==0 && !model.latched && model.wrong==0,
          "the release after a failure kept a handle or an error, or broke the model");
    product_free(&p);
  }
  for (int failure=OPENMX_EXACT_EXCHANGE_DEVICE_OK; failure<=OPENMX_EXACT_EXCHANGE_DEVICE_DOWNLOAD; failure++)
    for (int other=failure+1; other<=OPENMX_EXACT_EXCHANGE_DEVICE_DOWNLOAD+1; other++)
      check(strcmp(Exact_Exchange_Device_Failure_Name(failure),Exact_Exchange_Device_Failure_Name(other))!=0,
            "two failures have the same name");
}

/* A device that was asked for GEMMul8: every product goes through the
   bridge with the same handle, matrices and transfers, and the unit counts
   the products that the bridge did not compute in FP64. */
static void products_through_the_bridge(void)
{
  const char modes[4][2]={{'N','N'},{'T','N'},{'N','T'},{'T','T'}};
  OpenMXExactExchangeDevice device;
  size_t largest=0;
  model_reset();
  device_make(&device);
  device.gemmul8 = 1;
  for (int i=0; i<4; i++) {
    Product p;
    product_make(&p,modes[i][0],modes[i][1],5+i,3+2*i,4+i,41u+(unsigned)i);
    check(product_run(&device,&p,0.0)==1 && product_matches(&p),"a product through the bridge is wrong");
    check(model.live_count==0 && model.wrong==0 && !model.latched,
          "a product through the bridge left device memory or broke the model");
    check(model.count[CALL_MALLOC]==3*(i+1) && model.count[CALL_COPY]==3*(i+1) && model.count[CALL_BRIDGE]==i+1
          && model.count[CALL_DGEMM]==0,"a product through the bridge did not make one call of the bridge");
    check(model.peak_total==(largest<product_bytes(&p) ? product_bytes(&p) : largest),
          "a product through the bridge took more device memory than its three matrices");
    if (largest<product_bytes(&p)) largest = product_bytes(&p);
    product_free(&p);
  }
  check(model.count[CALL_CREATE]==1 && model.handles==1,"the bridge did not get the handle that the unit keeps");
  check(device.device_products==4 && device.gemmul8_products==4 && device.declined_products==0 && !device.disabled,
        "the counters of four products through the bridge are wrong");

  /* the bridge has no room for its workspace, or the environment switched
     GEMMul8 off: it computes in FP64, and the unit counts the difference */
  {
    Product p;
    product_make(&p,'N','T',6,5,4,47u);
    model.bridge_native = 1;
    check(product_run(&device,&p,0.0)==1 && product_matches(&p),"a product that the bridge computed in FP64 is wrong");
    check(device.device_products==5 && device.gemmul8_products==4 && model.native_calls==1
          && model.count[CALL_BRIDGE]==5 && model.count[CALL_DGEMM]==0 && model.wrong==0,
          "a product that the bridge computed in FP64 was counted as one of GEMMul8");
    model.bridge_native = 0;
    check(product_run(&device,&p,0.0)==1 && product_matches(&p) && device.device_products==6
          && device.gemmul8_products==5,"the count of GEMMul8 did not go on after a product in FP64");
    product_free(&p);
  }

  /* the bridge cannot get memory for a product: the host computes it, and
     the device stays in use */
  {
    Product p;
    product_make(&p,'T','N',6,5,4,53u);
    model.fail_kind = CALL_BRIDGE;
    model.fail_at = model.count[CALL_BRIDGE]+1;
    model.fail_status = CUBLAS_STATUS_ALLOC_FAILED;
    check(product_run(&device,&p,0.0)==0 && product_untouched(&p),"a product was taken without memory for the bridge");
    check(device.declined_products==1 && device.device_products==6 && device.gemmul8_products==5 && !device.disabled
          && device.failure==OPENMX_EXACT_EXCHANGE_DEVICE_OK && model.live_count==0 && model.wrong==0
          && !model.latched,"a bridge without memory was not treated as a product without room");
    check(product_run(&device,&p,0.0)==1 && product_matches(&p) && device.device_products==7
          && device.gemmul8_products==6,"the device did not take the product after the bridge had run out of memory");
    product_free(&p);
  }

  /* the unit goes back to the BLAS when the request is taken away */
  {
    Product p;
    const int bridge_calls=model.count[CALL_BRIDGE];
    product_make(&p,'N','N',5,4,6,59u);
    device.gemmul8 = 0;
    check(product_run(&device,&p,0.0)==1 && product_matches(&p),"a product after the request for GEMMul8 is wrong");
    check(model.count[CALL_BRIDGE]==bridge_calls && model.count[CALL_DGEMM]==1 && device.device_products==8
          && device.gemmul8_products==6,"a device without the request for GEMMul8 used the bridge");
    device.gemmul8 = 1;
    product_free(&p);
  }

  /* the bridge fails for another reason than memory: the device takes no
     further product */
  {
    Product p;
    const int bridge_calls=model.count[CALL_BRIDGE];
    product_make(&p,'T','T',6,5,4,61u);
    model.fail_kind = CALL_BRIDGE;
    model.fail_at = bridge_calls+1;
    model.fail_status = 0;
    check(product_run(&device,&p,0.0)==0 && product_untouched(&p),"a product was reported although the bridge failed");
    check(device.disabled && device.failure==OPENMX_EXACT_EXCHANGE_DEVICE_PRODUCT
          && device.failure_status==model.failed_status && device.failure_status!=0,
          "the failed call of the bridge is not recorded as it happened");
    check(model.live_count==0 && model.wrong==0 && !model.latched && device.device_products==8
          && device.gemmul8_products==6,"a product that failed in the bridge left device memory or was counted");
    check(product_run(&device,&p,0.0)==0 && model.count[CALL_BRIDGE]==bridge_calls+1,
          "a device that failed in the bridge was asked again");
    product_free(&p);
  }

  Exact_Exchange_Device_Release(&device);
  check(model.handles==0 && model.wrong==0 && !model.latched,"the release after the bridge kept a handle or an error");
}

int main(void)
{
  products_on_the_device();
  products_left_to_the_host();
  products_without_room();
  device_calls_that_fail();
  products_through_the_bridge();
  model_reset();
  printf("exx device product: ok\n");
  return 0;
}

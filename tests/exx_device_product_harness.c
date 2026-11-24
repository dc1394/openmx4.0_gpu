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

   It asks the same of the table for whole calls of modules/exx
   (Exact_Exchange_Device_Table), and three things more: the products and
   the moves of a call give the bits of the host, the products run on the
   stream of the OpenACC queue and the handle gets its stream back at the
   end of the call, and a call counts as computed on the device only when
   its result came back. The unit is compiled with
   OPENMX_EXACT_EXCHANGE_DEVICE_HOST_MOVES (tests/run_exx_device_product.py):
   the moves run on the host, on the memory of the model. Here the test
   calls the table in the order of modules/exx (exx.h), and the model notes
   a copy or a release while a product on the queue may still run.

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
  CALL_DGEMM, CALL_SYNCHRONIZE, CALL_LAST_ERROR, CALL_BRIDGE, CALL_ACC_STREAM, CALL_ACC_WAIT, CALL_GET_STREAM,
  CALL_SET_STREAM, CALL_KINDS
};

enum { LIVE_SLOTS = 32 };

struct OpenMXTestBlasContext { int device; cudaStream_t stream; };
struct OpenMXTestStream { int queue; };

/* the stream of the queue of the whole calls, and another one that a caller
   may have given the handle */
static struct OpenMXTestStream queue_stream={OPENMX_EXACT_EXCHANGE_DEVICE_QUEUE},other_stream={7};

typedef struct {
  int usable;                 /* answer of gpu_rank_device_usable */
  int usable_calls;
  int current_device;
  size_t capacity_bytes;      /* device memory, free when nothing is allocated */
  int count[CALL_KINDS];
  size_t taken_after_query;   /* device memory another process takes right after the next query */
  int fail_kind,fail_at;      /* the fail_at-th call of fail_kind fails; fail_at 0: none */
  int fail_status;            /* status of that failure; 0: the one the call usually fails with */
  int also_kind,also_at;      /* a second call that fails, with its usual status */
  int failed_status;          /* status the failed call returned */
  int latched;                /* an error waits in cudaGetLastError */
  void *live[LIVE_SLOTS];
  size_t live_bytes[LIVE_SLOTS];
  int live_count;
  size_t live_total,peak_total;
  int handles;                /* handles alive */
  int bridge_native;          /* the bridge computes its products in FP64 */
  long long native_calls;     /* calls of the bridge that ran in FP64 */
  int queue_products;         /* products on the stream of the queue: they may run after the call returned */
  int default_products;       /* products on another stream */
  int pending;                /* a product on the queue that nothing waited for yet */
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
  const int first=model.fail_at!=0 && model.fail_kind==kind && model.count[kind]==model.fail_at;
  const int second=model.also_at!=0 && model.also_kind==kind && model.count[kind]==model.also_at;
  if (!first && !second) return 0;
  model.failed_status = first && model.fail_status!=0 ? model.fail_status : usual;
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
  check(0<=slot && 0<bytes,"the model has no slot for an allocation of this size");
  const size_t doubles=(bytes+sizeof(double)-1)/sizeof(double);
  double *memory=malloc(doubles*sizeof(double));
  check(memory!=NULL,"the host of the model is out of memory");
  /* nothing on the device has a value before a copy or a product wrote it */
  for (size_t i=0; i<doubles; i++) memory[i] = NAN;
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
  /* a product on the queue may still use the memory */
  if (model.pending) model.wrong++;
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
  /* the copy runs on another stream than a product on the queue */
  if (model.pending) model.wrong++;
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
  /* the device is idle: the work of every stream is done */
  model.pending = 0;
  return cudaSuccess;
}

/* OpenACC: the stream of a queue, and the wait for the work of a queue. The
   unit uses one queue only. */
void *acc_get_cuda_stream(int async)
{
  if (model_fails(CALL_ACC_STREAM,1)) return NULL;
  if (async!=OPENMX_EXACT_EXCHANGE_DEVICE_QUEUE) model.wrong++;
  return &queue_stream;
}

void acc_wait(int wait_arg)
{
  model.count[CALL_ACC_WAIT]++;
  if (wait_arg!=OPENMX_EXACT_EXCHANGE_DEVICE_QUEUE) model.wrong++;
  else model.pending = 0;
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
  context->stream = NULL;
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

cublasStatus_t cublasGetStream(cublasHandle_t handle, cudaStream_t *stream)
{
  const int failed=model_fails(CALL_GET_STREAM,CUBLAS_STATUS_NOT_INITIALIZED);
  if (failed) return failed;
  if (handle==NULL || handle->device!=model.current_device) {
    model.wrong++;
    return CUBLAS_STATUS_NOT_INITIALIZED;
  }
  *stream = handle->stream;
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSetStream(cublasHandle_t handle, cudaStream_t stream)
{
  const int failed=model_fails(CALL_SET_STREAM,CUBLAS_STATUS_NOT_INITIALIZED);
  if (failed) return failed;
  if (handle==NULL || handle->device!=model.current_device) {
    model.wrong++;
    return CUBLAS_STATUS_NOT_INITIALIZED;
  }
  handle->stream = stream;
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
  if (handle->stream==&queue_stream) {
    model.queue_products++;
    model.pending = 1;
  }
  else model.default_products++;
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
  check(model.queue_products==0 && model.default_products==4 && !model.pending,
        "a product of its own ran on the stream of the queue of the whole calls");

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
  for (int failure=OPENMX_EXACT_EXCHANGE_DEVICE_OK; failure<=OPENMX_EXACT_EXCHANGE_DEVICE_WAIT; failure++)
    for (int other=failure+1; other<=OPENMX_EXACT_EXCHANGE_DEVICE_WAIT+1; other++)
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


/* ---- whole calls of modules/exx ---------------------------------------- */

/* A call as modules/exx makes one, in small: two products and three moves
   on the device. The first product is p = a b, the second q = a^T a. The
   moves write the array t: they set two of its columns to zero from one
   element 0, gather elements of p with lists of rows and columns and with
   lengths, and add elements of p at rows of a list and every second
   column. The array t comes back, and q with it. */
enum { CALL_M = 5, CALL_N = 4, CALL_K = 3, CALL_TARGET_ROWS = 4, CALL_TARGET_COLUMNS = 7, CALL_BLOCKS = 10,
       CALL_UPLOADS = 8 };

typedef struct {
  double a[CALL_M*CALL_K],b[CALL_K*CALL_N];      /* the factors, column-major */
  double product[CALL_M*CALL_N];                 /* a b as the host computes it */
  double square[CALL_K*CALL_K],square_expected[CALL_K*CALL_K];  /* a^T a as it came back, and from the host */
  double target[CALL_TARGET_ROWS*CALL_TARGET_COLUMNS];           /* t before the call */
  double expected[CALL_TARGET_ROWS*CALL_TARGET_COLUMNS];         /* t after the moves on the host */
  double result[CALL_TARGET_ROWS*CALL_TARGET_COLUMNS];           /* t as it came back */
  long long gather_rows[3],gather_columns[4],add_rows[3];
  int lengths[3];
  size_t bytes;                                  /* of the blocks of the call */
} CallData;

static const size_t call_sizes[CALL_BLOCKS]={
  sizeof(double)*CALL_M*CALL_K,sizeof(double)*CALL_K*CALL_N,sizeof(double)*CALL_M*CALL_N,sizeof(double)*CALL_K*CALL_K,
  sizeof(double)*CALL_TARGET_ROWS*CALL_TARGET_COLUMNS,sizeof(double),3*sizeof(long long),4*sizeof(long long),
  3*sizeof(long long),3*sizeof(int)};

static void call_make(CallData *d, unsigned seed)
{
  const long long gather_rows[3]={2,0,4},gather_columns[4]={0,CALL_M,3*CALL_M,2*CALL_M};
  const long long add_rows[3]={CALL_TARGET_COLUMNS,2*CALL_TARGET_COLUMNS,3*CALL_TARGET_COLUMNS};
  const int lengths[3]={4,0,3};
  memset(d,0,sizeof(*d));
  for (int i=0; i<CALL_M*CALL_K; i++) d->a[i] = sample(&seed);
  for (int i=0; i<CALL_K*CALL_N; i++) d->b[i] = sample(&seed);
  for (int i=0; i<CALL_TARGET_ROWS*CALL_TARGET_COLUMNS; i++) d->target[i] = sample(&seed);
  memcpy(d->gather_rows,gather_rows,sizeof(gather_rows));
  memcpy(d->gather_columns,gather_columns,sizeof(gather_columns));
  memcpy(d->add_rows,add_rows,sizeof(add_rows));
  memcpy(d->lengths,lengths,sizeof(lengths));
  reference_dgemm(0,0,CALL_M,CALL_N,CALL_K,1.0,d->a,CALL_M,d->b,CALL_K,0.0,d->product,CALL_M);
  reference_dgemm(1,0,CALL_K,CALL_K,CALL_M,1.0,d->a,CALL_M,d->a,CALL_M,0.0,d->square_expected,CALL_K);
  /* the moves on the host, in their order, the product first and then the
     sum */
  memcpy(d->expected,d->target,sizeof(d->target));
  for (int r=0; r<CALL_TARGET_ROWS; r++)
    for (int c=0; c<2; c++) d->expected[r*CALL_TARGET_COLUMNS+5+c] = 1.0*0.0;
  for (int r=0; r<3; r++) {
    for (int c=0; c<d->lengths[r]; c++) {
      const double value=-0.5*d->product[d->gather_rows[r]+d->gather_columns[c]];
      d->expected[r*CALL_TARGET_COLUMNS+c] = value;
    }
  }
  for (int r=0; r<3; r++) {
    for (int c=0; c<3; c++) {
      const double value=0.75*d->product[r+c*CALL_M];
      d->expected[d->add_rows[r]+2*c] += value;
    }
  }
  for (int i=0; i<CALL_K*CALL_K; i++) d->square[i] = untouched;
  for (int i=0; i<CALL_TARGET_ROWS*CALL_TARGET_COLUMNS; i++) d->result[i] = untouched;
  d->bytes = 0;
  for (int i=0; i<CALL_BLOCKS; i++) d->bytes += call_sizes[i];
}

/* The call in the order of modules/exx (exx.h): accept; the allocations;
   the uploads; the products and the moves; finish; the downloads; the
   releases. After a function that failed, or an allocation that gave NULL,
   the call stops as modules/exx stops it: finish when work was given to the
   device after the last finish, then the releases. Returns 1 when the
   result came back. */
static int call_run(const OpenMXExxDevice *table, CallData *d, double multiplications)
{
  void *context=table->context;
  void *blocks[CALL_BLOCKS]={NULL};
  double *da=NULL,*db=NULL,*dp=NULL,*dq=NULL,*dt=NULL,*dz=NULL;
  long long *gather_rows=NULL,*gather_columns=NULL,*add_rows=NULL;
  int *lengths=NULL;
  int count=0,pending=0,done=0;
  const double nothing=0.0;
  if (table->accept(context,multiplications,d->bytes)!=1) return 0;
  for (; count<CALL_BLOCKS; count++) {
    blocks[count] = table->allocate(context,call_sizes[count]);
    if (blocks[count]==NULL) goto stop;
  }
  da = blocks[0];
  db = blocks[1];
  dp = blocks[2];
  dq = blocks[3];
  dt = blocks[4];
  dz = blocks[5];
  gather_rows = blocks[6];
  gather_columns = blocks[7];
  add_rows = blocks[8];
  lengths = blocks[9];
  if (table->upload(context,da,d->a,sizeof(d->a))!=0 || table->upload(context,db,d->b,sizeof(d->b))!=0
      || table->upload(context,dt,d->target,sizeof(d->target))!=0
      || table->upload(context,dz,&nothing,sizeof(nothing))!=0
      || table->upload(context,gather_rows,d->gather_rows,sizeof(d->gather_rows))!=0
      || table->upload(context,gather_columns,d->gather_columns,sizeof(d->gather_columns))!=0
      || table->upload(context,add_rows,d->add_rows,sizeof(d->add_rows))!=0
      || table->upload(context,lengths,d->lengths,sizeof(d->lengths))!=0)
    goto stop;
  pending = 1;
  if (table->dgemm(context,'N','N',CALL_M,CALL_N,CALL_K,1.0,da,CALL_M,db,CALL_K,0.0,dp,CALL_M)!=0) goto stop;
  if (table->dgemm(context,'T','N',CALL_K,CALL_K,CALL_M,1.0,da,CALL_M,da,CALL_M,0.0,dq,CALL_K)!=0) goto stop;
  {
    const OpenMXExxDeviceIndex rows_of_t={CALL_TARGET_COLUMNS,NULL,1,NULL},one_element={0,NULL,0,NULL};
    const OpenMXExxDeviceIndex listed={0,gather_rows,0,gather_columns},stepped={0,add_rows,2,NULL};
    const OpenMXExxDeviceIndex columns_of_p={1,NULL,CALL_M,NULL};
    if (table->move(context,CALL_TARGET_ROWS,2,NULL,1.0,0,dt+5,&rows_of_t,dz,&one_element)!=0) goto stop;
    if (table->move(context,3,4,lengths,-0.5,0,dt,&rows_of_t,dp,&listed)!=0) goto stop;
    if (table->move(context,3,3,NULL,0.75,1,dt,&stepped,dp,&columns_of_p)!=0) goto stop;
  }
  /* finish waits also when it fails: it is called once */
  pending = 0;
  if (table->finish(context)!=0) goto stop;
  if (table->download(context,d->square,dq,sizeof(d->square))!=0) goto stop;
  if (table->download(context,d->result,dt,sizeof(d->result))!=0) goto stop;
  done = 1;
stop:
  if (pending) (void)table->finish(context);
  for (int i=0; i<count; i++) table->release(context,blocks[i]);
  return done;
}

static int call_matches(const CallData *d)
{
  return memcmp(d->result,d->expected,sizeof(d->result))==0
      && memcmp(d->square,d->square_expected,sizeof(d->square))==0;
}

static cudaStream_t handle_stream(const OpenMXExactExchangeDevice *device)
{
  return device->blas!=NULL ? ((const struct OpenMXTestBlasContext*)device->blas)->stream : NULL;
}

/* A device for the calls of the cases: the first product of a call has 60
   multiplications and is large, the second has 45 and is not. */
static const OpenMXExxDevice *call_device(OpenMXExactExchangeDevice *device, OpenMXExxDevice *table)
{
  device_make(device);
  device->minimum_work = 50.0;
  check(device->device_calls==0 && device->declined_calls==0 && device->call_products==0
        && device->call_large_products==0 && device->call_gemmul8_products==0 && device->largest_call_bytes==0
        && !device->call_accepted,"a new device does not have the values of a new calculation for whole calls");
  memset(table,0x5a,sizeof(*table));
  const OpenMXExxDevice *made=Exact_Exchange_Device_Table(device,table);
  check(made==table && table->context==device && table->accept!=NULL && table->allocate!=NULL
        && table->release!=NULL && table->upload!=NULL && table->download!=NULL && table->dgemm!=NULL
        && table->move!=NULL && table->finish!=NULL,"the CUDA build offers no complete table for whole calls");
  return made;
}

/* What every call has to leave behind: no device memory, no error, no
   pending work, and the handle on its stream of before. */
static void call_clean(const OpenMXExactExchangeDevice *device, cudaStream_t stream, const char *message)
{
  check(model.live_count==0 && model.wrong==0 && !model.latched && !model.pending && !device->call_accepted
        && !device->call_stream_set,message);
  check(handle_stream(device)==stream,message);
}

static void calls_on_the_device(void)
{
  OpenMXExactExchangeDevice device;
  OpenMXExxDevice table;
  CallData d;
  model_reset();
  const OpenMXExxDevice *t=call_device(&device,&table);
  check(Exact_Exchange_Device_Table(NULL,&table)==NULL && Exact_Exchange_Device_Table(&device,NULL)==NULL,
        "a table was made without a device or without its storage");
  check(Exact_Exchange_Device_Call_End(NULL)==0,"the end of a call without a device reported a call");
  {
    const Model before=model;
    check(Exact_Exchange_Device_Call_End(&device)==0 && memcmp(&before,&model,sizeof(model))==0
          && device.device_calls==0 && device.declined_calls==0,"the end of a call that never began counted it");
  }

  call_make(&d,71u);
  check(call_run(t,&d,60.0)==1,"the device did not compute a call that fits");
  check(device.call_accepted && device.call_downloaded && device.call_stream_set
        && handle_stream(&device)==&queue_stream,"the products of the call did not run on the stream of the queue");
  check(call_matches(&d),"the products and the moves of a call differ from those of the host");
  check(Exact_Exchange_Device_Call_End(&device)==1,"the end of a call on the device did not report it");
  call_clean(&device,NULL,"a call on the device left something behind");
  check(device.device_calls==1 && device.declined_calls==0 && device.call_products==2
        && device.call_large_products==1 && device.call_gemmul8_products==0 && device.largest_call_bytes==d.bytes
        && !device.disabled,"the counters of a call on the device are wrong");
  check(device.device_products==0 && device.declined_products==0,"a call counted products of its own");
  check(model.queue_products==2 && model.default_products==0 && model.count[CALL_DGEMM]==2
        && model.count[CALL_BRIDGE]==0,"the products of a call did not go to the BLAS on the queue");
  check(model.count[CALL_MALLOC]==CALL_BLOCKS && model.count[CALL_FREE]==CALL_BLOCKS
        && model.count[CALL_COPY]==CALL_UPLOADS+2 && model.peak_total==d.bytes,
        "a call did not allocate, copy and release its blocks once each");
  check(model.count[CALL_SYNCHRONIZE]==CALL_UPLOADS+1 && model.count[CALL_ACC_WAIT]==1,
        "an upload or the finish of a call did not wait for the device");
  check(model.count[CALL_ACC_STREAM]==1 && model.count[CALL_GET_STREAM]==1 && model.count[CALL_SET_STREAM]==2
        && model.count[CALL_CREATE]==1 && model.handles==1,"the handle did not get the stream of the queue and back");

  /* a product of its own after the call: the default stream */
  {
    Product p;
    product_make(&p,'N','T',6,5,4,73u);
    check(product_run(&device,&p,0.0)==1 && product_matches(&p),"a product after a call is wrong");
    check(model.default_products==1 && model.queue_products==2 && !model.pending && device.device_products==1,
          "a product of its own after a call ran on the stream of the queue");
    product_free(&p);
  }

  /* a smaller call: the largest memory of a call stays; a handle that had
     another stream gets that one back. The call tells accept 8 bytes less
     than it allocates: the unit uses the number for the room and for the
     counter only. */
  {
    const size_t largest=device.largest_call_bytes;
    ((struct OpenMXTestBlasContext*)device.blas)->stream = &other_stream;
    call_make(&d,79u);
    d.bytes = largest-8;
    check(call_run(t,&d,60.0)==1 && call_matches(&d),"the second call is wrong");
    check(Exact_Exchange_Device_Call_End(&device)==1,"the end of the second call did not report it");
    call_clean(&device,&other_stream,"the second call did not give the handle its stream back");
    check(device.device_calls==2 && device.call_products==4 && device.call_large_products==2
          && device.largest_call_bytes==largest,"the counters of the second call are wrong");
    ((struct OpenMXTestBlasContext*)device.blas)->stream = NULL;
  }

  /* an accept without the end of its call: the next accept gives the handle
     its stream back first */
  {
    check(t->accept(t->context,60.0,d.bytes)==1 && handle_stream(&device)==&queue_stream,
          "the device did not take a call");
    check(t->accept(t->context,60.0,d.bytes)==1 && handle_stream(&device)==&queue_stream,
          "the device did not take a call after a call without its end");
    check(Exact_Exchange_Device_Call_End(&device)==0 && device.device_calls==2,
          "a call without a result was counted as one on the device");
    call_clean(&device,NULL,"a call without its end left the stream of the queue on the handle");
  }

  /* GEMMul8: the large product goes through the bridge, the other one to the
     BLAS; a product that the bridge computed in FP64 is not counted as one of
     GEMMul8 */
  device.gemmul8 = 1;
  for (int native=0; native<2; native++) {
    const int bridge=model.count[CALL_BRIDGE],blas=model.count[CALL_DGEMM];
    model.bridge_native = native;
    call_make(&d,83u+(unsigned)native);
    check(call_run(t,&d,60.0)==1 && call_matches(&d),"a call through the bridge is wrong");
    check(Exact_Exchange_Device_Call_End(&device)==1,"the end of a call through the bridge did not report it");
    call_clean(&device,NULL,"a call through the bridge left something behind");
    check(model.count[CALL_BRIDGE]==bridge+1 && model.count[CALL_DGEMM]==blas+1,
          "a call did not send its large product, and only that one, through the bridge");
    check(device.call_large_products==3+native && device.call_gemmul8_products==1,
          "the count of GEMMul8 of a call is wrong");
  }
  model.bridge_native = 0;
  check(device.device_calls==4 && device.call_products==8 && model.queue_products==8,
        "the counters of the calls through the bridge are wrong");

  Exact_Exchange_Device_Release(&device);
  check(model.handles==0 && model.wrong==0 && !model.latched,"the release after the calls kept a handle or an error");
}

/* Calls that the device does not take: the host computes them. */
static int result_untouched_call(const CallData *d)
{
  for (int i=0; i<CALL_TARGET_ROWS*CALL_TARGET_COLUMNS; i++)
    if (d->result[i]!=untouched) return 0;
  for (int i=0; i<CALL_K*CALL_K; i++)
    if (d->square[i]!=untouched) return 0;
  return 1;
}

static void calls_left_to_the_host(void)
{
  OpenMXExactExchangeDevice device;
  OpenMXExxDevice table;
  CallData d;
  model_reset();
  const OpenMXExxDevice *t=call_device(&device,&table);
  call_make(&d,89u);

  /* small, without a usable device, or switched off: nothing is asked and
     nothing is counted */
  check(call_run(t,&d,49.0)==0 && model.usable_calls==0,"a call below the smallest size was taken");
  model.usable = 0;
  check(call_run(t,&d,60.0)==0 && model.usable_calls==1,"a process without a usable device took a call");
  model.usable = 1;
  device.disabled = 1;
  check(call_run(t,&d,60.0)==0 && model.usable_calls==1,"a device that is switched off took a call");
  device.disabled = 0;
  check(Exact_Exchange_Device_Call_End(&device)==0,"the end of a call that was not taken reported it");
  for (int kind=0; kind<CALL_KINDS; kind++)
    check(model.count[kind]==0,"a call that the device did not take reached the device");
  check(device.device_calls==0 && device.declined_calls==0 && device.blas==NULL && !device.disabled,
        "a call that the device did not take was counted");

  /* no room: the arrays of the call and the reserve have to fit */
  model.capacity_bytes = d.bytes+device.reserve_bytes-1;
  check(call_run(t,&d,60.0)==0 && Exact_Exchange_Device_Call_End(&device)==0,"a call that does not fit was taken");
  check(device.declined_calls==1 && device.blas==NULL && model.count[CALL_QUERY]==1 && model.count[CALL_MALLOC]==0,
        "a call that does not fit was not declined at the query");
  /* the reserve of the fault injection: no call fits */
  model.capacity_bytes = (size_t)1<<40;
  device.reserve_bytes = (size_t)-1;
  check(call_run(t,&d,60.0)==0 && Exact_Exchange_Device_Call_End(&device)==0 && device.declined_calls==2
        && model.count[CALL_MALLOC]==0,"a call was taken although the reserve is all memory");
  device.reserve_bytes = 4096;
  model.capacity_bytes = (size_t)1<<20;
  /* the handle cannot get its memory */
  model.fail_kind = CALL_CREATE;
  model.fail_at = 1;
  model.fail_status = CUBLAS_STATUS_ALLOC_FAILED;
  check(call_run(t,&d,60.0)==0 && Exact_Exchange_Device_Call_End(&device)==0 && device.declined_calls==3
        && device.blas==NULL && model.count[CALL_ACC_STREAM]==0,"a handle without memory took a call");
  call_clean(&device,NULL,"a call without room for the handle left something behind");
  check(!device.disabled && device.failure==OPENMX_EXACT_EXCHANGE_DEVICE_OK,
        "a call without room switched the device off");
  model.fail_at = 0;
  model.fail_status = 0;
  check(result_untouched_call(&d),"a call that the device did not take wrote its result");
}

/* Calls that find no room after the device took them: another process takes
   the memory, or the BLAS or the bridge cannot get memory of its own. The
   host computes the call, and the device stays in use. */
static void calls_without_room(void)
{
  OpenMXExactExchangeDevice device;
  OpenMXExxDevice table;
  CallData d;
  model_reset();
  const OpenMXExxDevice *t=call_device(&device,&table);
  call_make(&d,97u);
  const size_t room=d.bytes+device.reserve_bytes;

  /* the first, the fifth or the last allocation runs out */
  {
    const int at[3]={0,4,CALL_BLOCKS-1};
    for (int i=0; i<3; i++) {
      size_t before=0;
      for (int b=0; b<at[i]; b++) before += call_sizes[b];
      const int allocations=model.count[CALL_MALLOC];
      model.capacity_bytes = room;
      model.taken_after_query = room-before-1;
      check(call_run(t,&d,60.0)==0,"a call was computed although the device ran out of memory");
      check(model.count[CALL_MALLOC]==allocations+at[i]+1 && model.failed_status==cudaErrorMemoryAllocation,
            "the device did not run out of memory at the allocation of the case");
      check(Exact_Exchange_Device_Call_End(&device)==0 && device.declined_calls==i+1 && device.device_calls==0,
            "a call without room was not counted as one left to the host");
      call_clean(&device,NULL,"a call without room left something behind");
      check(!device.disabled && device.failure==OPENMX_EXACT_EXCHANGE_DEVICE_OK && result_untouched_call(&d),
            "a call without room switched the device off or wrote its result");
    }
  }
  model.capacity_bytes = (size_t)1<<20;

  /* the BLAS, and then the bridge, cannot get memory for a product */
  for (int bridge=0; bridge<2; bridge++) {
    const long long declined=device.declined_calls;
    device.gemmul8 = bridge;
    model.fail_kind = bridge ? CALL_BRIDGE : CALL_DGEMM;
    model.fail_at = model.count[model.fail_kind]+1;
    model.fail_status = CUBLAS_STATUS_ALLOC_FAILED;
    check(call_run(t,&d,60.0)==0 && model.count[CALL_ACC_WAIT]==bridge+1,
          "a call without memory for a product was not stopped and finished");
    check(Exact_Exchange_Device_Call_End(&device)==0 && device.declined_calls==declined+1 && device.device_calls==0,
          "a product without memory was not counted as a call left to the host");
    call_clean(&device,NULL,"a call without memory for a product left something behind");
    check(!device.disabled && device.failure==OPENMX_EXACT_EXCHANGE_DEVICE_OK && result_untouched_call(&d),
          "a product without memory switched the device off or wrote the result");
    model.fail_at = 0;
    model.fail_status = 0;
  }
  device.gemmul8 = 0;

  /* the memory is back: the device takes the call */
  check(call_run(t,&d,60.0)==1 && call_matches(&d) && Exact_Exchange_Device_Call_End(&device)==1,
        "the device did not take a call after the calls without room");
  call_clean(&device,NULL,"the call after the calls without room left something behind");
  check(device.device_calls==1 && device.declined_calls==5 && device.call_products==2,
        "the counters after the calls without room are wrong");
  Exact_Exchange_Device_Release(&device);
}

/* A function of a call fails for another reason than room: the host
   computes the call, the device takes no further call or product, and the
   first failure is the one recorded. */
static void calls_that_fail(void)
{
  /* the call that fails, its number among the calls of its kind in the
     call, the step the unit has to name, and the status it has to record
     (0: the one the model failed with) */
  const struct { int kind,at,failure,status; } sites[]={
    {CALL_QUERY,1,OPENMX_EXACT_EXCHANGE_DEVICE_MEMORY_QUERY,0},
    {CALL_GET_DEVICE,1,OPENMX_EXACT_EXCHANGE_DEVICE_HANDLE,0},
    {CALL_CREATE,1,OPENMX_EXACT_EXCHANGE_DEVICE_HANDLE,0},
    {CALL_ACC_STREAM,1,OPENMX_EXACT_EXCHANGE_DEVICE_STREAM,-1},
    {CALL_GET_STREAM,1,OPENMX_EXACT_EXCHANGE_DEVICE_STREAM,0},
    {CALL_SET_STREAM,1,OPENMX_EXACT_EXCHANGE_DEVICE_STREAM,0},
    {CALL_MALLOC,1,OPENMX_EXACT_EXCHANGE_DEVICE_ALLOCATION,0},
    {CALL_MALLOC,CALL_BLOCKS,OPENMX_EXACT_EXCHANGE_DEVICE_ALLOCATION,0},
    {CALL_COPY,1,OPENMX_EXACT_EXCHANGE_DEVICE_UPLOAD,0},
    {CALL_COPY,CALL_UPLOADS,OPENMX_EXACT_EXCHANGE_DEVICE_UPLOAD,0},
    {CALL_SYNCHRONIZE,1,OPENMX_EXACT_EXCHANGE_DEVICE_UPLOAD,0},
    {CALL_DGEMM,1,OPENMX_EXACT_EXCHANGE_DEVICE_PRODUCT,0},
    {CALL_DGEMM,2,OPENMX_EXACT_EXCHANGE_DEVICE_PRODUCT,0},
    {CALL_SYNCHRONIZE,CALL_UPLOADS+1,OPENMX_EXACT_EXCHANGE_DEVICE_WAIT,0},
    {CALL_COPY,CALL_UPLOADS+1,OPENMX_EXACT_EXCHANGE_DEVICE_DOWNLOAD,0},
    {CALL_COPY,CALL_UPLOADS+2,OPENMX_EXACT_EXCHANGE_DEVICE_DOWNLOAD,0}};
  for (size_t site=0; site<sizeof(sites)/sizeof(sites[0]); site++) {
    OpenMXExactExchangeDevice device;
    OpenMXExxDevice table;
    CallData d;
    model_reset();
    const OpenMXExxDevice *t=call_device(&device,&table);
    call_make(&d,101u+(unsigned)site);
    model.fail_kind = sites[site].kind;
    model.fail_at = sites[site].at;

    check(call_run(t,&d,60.0)==0,"a call was computed although a function of the device failed");
    check(Exact_Exchange_Device_Call_End(&device)==0,"the end of a call that failed reported it");
    check(device.disabled && device.failure==sites[site].failure && device.failure_status!=0
          && device.failure_status==(sites[site].status!=0 ? sites[site].status : model.failed_status),
          "the failed call of the device is not recorded as it happened");
    check(0<model.count[CALL_LAST_ERROR],"a call that failed did not take the error of the runtime");
    call_clean(&device,NULL,"a call that failed left something behind");
    check(device.device_calls==0 && device.declined_calls==0 && device.call_products==0 && device.largest_call_bytes==0
          && device.device_products==0,"a call that failed was counted");

    /* the host computes this call and all later ones, and every product */
    {
      const Model before=model;
      check(call_run(t,&d,60.0)==0 && Exact_Exchange_Device_Call_End(&device)==0
            && memcmp(&before,&model,sizeof(model))==0,"a device that failed in a call was asked again");
      Product p;
      product_make(&p,'N','N',5,4,6,107u);
      check(product_run(&device,&p,0.0)==0 && memcmp(&before,&model,sizeof(model))==0 && product_untouched(&p),
            "a device that failed in a call took a product");
      product_free(&p);
    }
    Exact_Exchange_Device_Release(&device);
    check(model.handles==0 && !model.latched && model.wrong==0,
          "the release after a failed call kept a handle or an error, or broke the model");
  }

  /* the finish after a failed product fails as well: the first failure
     stays, and the error of the second one is taken off the latch */
  {
    OpenMXExactExchangeDevice device;
    OpenMXExxDevice table;
    CallData d;
    model_reset();
    const OpenMXExxDevice *t=call_device(&device,&table);
    call_make(&d,109u);
    model.fail_kind = CALL_DGEMM;
    model.fail_at = 2;
    model.also_kind = CALL_SYNCHRONIZE;
    model.also_at = CALL_UPLOADS+1;
    check(call_run(t,&d,60.0)==0,"a call was computed although its product failed");
    check(model.count[CALL_ACC_WAIT]==1 && model.count[CALL_SYNCHRONIZE]==CALL_UPLOADS+1,
          "the call that failed in a product was not finished once");
    check(device.disabled && device.failure==OPENMX_EXACT_EXCHANGE_DEVICE_PRODUCT
          && device.failure_status==CUBLAS_STATUS_EXECUTION_FAILED,"the failed product is not the failure recorded");
    check(Exact_Exchange_Device_Call_End(&device)==0,"the end of the failed call reported it");
    call_clean(&device,NULL,"the call with two failures left something behind");
    Exact_Exchange_Device_Release(&device);
  }

  /* the handle cannot get its stream back after the result came back: the
     call counts, and the handle goes back to the runtime */
  {
    OpenMXExactExchangeDevice device;
    OpenMXExxDevice table;
    CallData d;
    model_reset();
    const OpenMXExxDevice *t=call_device(&device,&table);
    call_make(&d,113u);
    model.fail_kind = CALL_SET_STREAM;
    model.fail_at = 2;
    check(call_run(t,&d,60.0)==1 && call_matches(&d),"the call before the failed stream is wrong");
    check(Exact_Exchange_Device_Call_End(&device)==1 && device.device_calls==1 && !device.disabled,
          "a call whose handle could not get its stream back was not counted");
    check(device.blas==NULL && model.handles==0 && model.live_count==0 && model.wrong==0 && !model.latched,
          "a handle that could not get its stream back was kept");
    {
      Product p;
      product_make(&p,'T','N',5,4,6,127u);
      check(product_run(&device,&p,0.0)==1 && product_matches(&p) && model.handles==1 && handle_stream(&device)==NULL,
            "the next product did not get a new handle on the default stream");
      product_free(&p);
    }
    Exact_Exchange_Device_Release(&device);
  }
}

int main(void)
{
  products_on_the_device();
  products_left_to_the_host();
  products_without_room();
  device_calls_that_fail();
  products_through_the_bridge();
  calls_on_the_device();
  calls_left_to_the_host();
  calls_without_room();
  calls_that_fail();
  model_reset();
  printf("exx device product: ok\n");
  return 0;
}

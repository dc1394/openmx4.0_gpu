/**********************************************************************
  Exact_Exchange_Device.c:

     Matrix products of the exchange build on the GPU of this process.

     See docs/EXX_IMPLEMENTATION_GUIDE.md, section 11, and
     openmx_exact_exchange_device.h for the interface. A standalone product
     queues uploads, the FP64 product (or GEMMul8 when requested), and the
     download on one stream. It waits for that stream before returning.
     Workspace that fits the calculation's retention budget is reused.

     A product that does not fit is left to the host and the device stays
     in use; so is one for which the device runs out of memory after the
     check, as it can when several processes share the device. Any other
     device call that fails switches the device off for the rest of the
     calculation: the host then computes every product, the failed one
     included.

     CUDA and HIP also offer modules/exx the device for whole calls
     (Exact_Exchange_Device_Table). modules/exx then keeps its arrays on
     the device for the call and asks for products and moves of elements
     there. Transfers, products and moves run in order on one stream per
     backend. The calculation owns the reusable allocations
     and returns them before other post-SCF stages need the memory.

     The functions take every input as an argument and read no global
     variable.
***********************************************************************/

#include <stddef.h>
#include <stdint.h>
#include "openmx_gpu_backend.h"
#include "openmx_gpu_runtime.h"
#include "openmx_gpu_blas.h"
#include "openmx_gpu_device.h"
#include "openmx_exact_exchange_device.h"
#if OPENMX_GPU_HIP
#include "openmx_exact_exchange_hip.h"
#endif

/* Whole calls need device moves. The CUDA model test runs its moves on the host. */
#if OPENMX_GPU_HIP || (OPENMX_GPU_CUDA && (defined(_OPENACC) || defined(OPENMX_EXACT_EXCHANGE_DEVICE_HOST_MOVES)))
#define OPENMX_EXACT_EXCHANGE_DEVICE_CALLS 1
#else
#define OPENMX_EXACT_EXCHANGE_DEVICE_CALLS 0
#endif

/* The GEMMul8 bridge of the device BLAS (gemmul8_bridge.cu). It computes a
   product with GEMMul8, or in FP64 when it has no room for its workspace or
   the environment switches GEMMul8 off, and counts the calls that ran in
   FP64. */
OpenMXBlasStatus openmx_gemmul8Dgemm(OpenMXBlasHandle handle, OpenMXBlasOperation transa, OpenMXBlasOperation transb,
                                     int m, int n, int k, const double *alpha, const double *A, int lda,
                                     const double *B, int ldb, const double *beta, double *C, int ldc);
long long openmx_gemmul8NativeCalls(int is_complex);

/* The state of the call in progress as a new call finds it. */
static void ExactExchange_Device_Call_Clear(OpenMXExactExchangeDevice *device)
{
  device->call_accepted = 0;
  device->call_without_room = 0;
  device->call_failed = 0;
  device->call_downloaded = 0;
  device->call_stream_set = 0;
  device->call_stream = NULL;
  device->call_cached = 0;
  device->call_next_buffer = 0;
  device->call_stream_before = NULL;
  device->call_bytes = 0;
  device->call_products_now = 0;
  device->call_large_now = 0;
  device->call_gemmul8_now = 0;
}

void Exact_Exchange_Device_Initialize(OpenMXExactExchangeDevice *device)
{
  *device = (OpenMXExactExchangeDevice){0};
  device->blas_device = -1;
  /* A standalone product has to amortize its two input transfers and the
     result transfer. The calculation may set a higher dispatch threshold. */
  device->minimum_work = 1.0e9;
  device->sharing_ranks = 1;
  device->reserve_bytes = OPENMX_EXACT_EXCHANGE_DEVICE_RESERVE;
}

const char *Exact_Exchange_Device_Failure_Name(int failure)
{
  switch (failure) {
  case OPENMX_EXACT_EXCHANGE_DEVICE_OK:           return "no failure";
  case OPENMX_EXACT_EXCHANGE_DEVICE_MEMORY_QUERY: return "the query of the free device memory";
  case OPENMX_EXACT_EXCHANGE_DEVICE_HANDLE:       return "the handle of the device BLAS";
  case OPENMX_EXACT_EXCHANGE_DEVICE_ALLOCATION:   return "an allocation on the device";
  case OPENMX_EXACT_EXCHANGE_DEVICE_UPLOAD:       return "the copy of a factor to the device";
  case OPENMX_EXACT_EXCHANGE_DEVICE_PRODUCT:      return "the product on the device";
  case OPENMX_EXACT_EXCHANGE_DEVICE_DOWNLOAD:     return "the copy of the result from the device";
  case OPENMX_EXACT_EXCHANGE_DEVICE_STREAM:       return "the stream of the device queue";
  case OPENMX_EXACT_EXCHANGE_DEVICE_WAIT:         return "the wait for the work on the device";
  default:                                        return "an unknown step";
  }
}

static void ExactExchange_Device_Free_Buffers(OpenMXExactExchangeBuffer *buffers, int count)
{
  for (int i=0; i<count; i++) {
    if (buffers[i].pointer!=NULL) (void)OpenMXGpuFree(buffers[i].pointer);
    buffers[i] = (OpenMXExactExchangeBuffer){0};
  }
}

static size_t ExactExchange_Device_Buffer_Bytes(const OpenMXExactExchangeBuffer *buffers, int count)
{
  size_t bytes=0;
  for (int i=0; i<count; i++) bytes += buffers[i].bytes;
  return bytes;
}

static OpenMXGpuError ExactExchange_Device_Reserve_Buffer(OpenMXExactExchangeBuffer *buffer, size_t bytes)
{
  if (bytes<=buffer->bytes) return OPENMX_GPU_SUCCESS;
  ExactExchange_Device_Free_Buffers(buffer,1);
  void *pointer=NULL;
  const OpenMXGpuError status=OpenMXGpuMalloc(&pointer,bytes);
  if (status==OPENMX_GPU_SUCCESS) *buffer = (OpenMXExactExchangeBuffer){pointer,bytes};
  return status;
}

void Exact_Exchange_Device_Release(OpenMXExactExchangeDevice *device)
{
  if (device->blas!=NULL) {
    /* A handle goes back on the device that made it. The process is bound
       to that device unless the binding changed since the last product. */
    int current=-1;
    const int moved=OpenMXGpuGetDevice(&current)==OPENMX_GPU_SUCCESS && current!=device->blas_device
                    && OpenMXGpuSetDevice(device->blas_device)==OPENMX_GPU_SUCCESS;
    ExactExchange_Device_Free_Buffers(device->product_buffers,3);
    ExactExchange_Device_Free_Buffers(device->call_buffers,OPENMX_EXACT_EXCHANGE_CALL_BUFFERS);
    (void)OpenMXBlasDestroy((OpenMXBlasHandle)device->blas);
    if (device->product_stream!=NULL) (void)OpenMXGpuStreamDestroy((OpenMXGpuStream)device->product_stream);
    if (moved) (void)OpenMXGpuSetDevice(current);
    /* a handle that outlived its device leaves an error behind */
    (void)OpenMXGpuGetLastError();
  }
  device->blas = NULL;
  device->product_stream = NULL;
  device->blas_device = -1;
}

/* The device has no room for this product: the host computes it, and the
   device stays in use. */
static int ExactExchange_Device_Decline(OpenMXExactExchangeDevice *device)
{
  ExactExchange_Device_Free_Buffers(device->product_buffers,3);
  (void)OpenMXGpuGetLastError();
  device->declined_products++;
  return 0;
}

/* A device call failed for another reason than room: the device takes no
   further product. */
static void ExactExchange_Device_Note_Failure(OpenMXExactExchangeDevice *device, int failure, int status)
{
  (void)OpenMXGpuGetLastError();
  if (!device->disabled) {
    device->disabled = 1;
    device->failure = failure;
    device->failure_status = status;
  }
}

static int ExactExchange_Device_Fail(OpenMXExactExchangeDevice *device, int failure, int status)
{
  ExactExchange_Device_Free_Buffers(device->product_buffers,3);
  ExactExchange_Device_Note_Failure(device,failure,status);
  return 0;
}

/* Buffers and handles are tied to the same physical device. */
static OpenMXGpuError ExactExchange_Device_Current(OpenMXExactExchangeDevice *device, int *current)
{
  const OpenMXGpuError status=OpenMXGpuGetDevice(current);
  if (status==OPENMX_GPU_SUCCESS && device->blas!=NULL && device->blas_device!=*current)
    Exact_Exchange_Device_Release(device);
  return status;
}

static OpenMXBlasStatus ExactExchange_Device_Handle(OpenMXExactExchangeDevice *device, int current)
{
  if (device->blas==NULL) {
    OpenMXBlasHandle created=NULL;
    const OpenMXBlasStatus made=OpenMXBlasCreate(&created);
    if (made!=OPENMX_BLAS_SUCCESS) return made;
    device->blas = (void*)created;
    device->blas_device = current;
  }
  return OPENMX_BLAS_SUCCESS;
}

/* Both the BLAS and copies use this stream. HIP also queues its whole-call
   moves here. Keep allocation failures recoverable on a shared GPU. */
enum { EXX_STREAM_FAILED = -1, EXX_STREAM_NO_ROOM, EXX_STREAM_READY };

static int ExactExchange_Device_Private_Stream(OpenMXExactExchangeDevice *device, int *status)
{
  if (device->product_stream!=NULL) return EXX_STREAM_READY;
  OpenMXGpuStream stream=NULL;
  const OpenMXGpuError created=OpenMXGpuStreamCreateWithFlags(&stream,OPENMX_GPU_STREAM_NONBLOCKING);
  if (created!=OPENMX_GPU_SUCCESS) {
    *status = (int)created;
    return created==OPENMX_GPU_OUT_OF_MEMORY ? EXX_STREAM_NO_ROOM : EXX_STREAM_FAILED;
  }
  const OpenMXBlasStatus streamed=OpenMXBlasSetStream((OpenMXBlasHandle)device->blas,stream);
  if (streamed!=OPENMX_BLAS_SUCCESS) {
    (void)OpenMXGpuStreamDestroy(stream);
    *status = (int)streamed;
    return streamed==OPENMX_BLAS_ALLOC_FAILED ? EXX_STREAM_NO_ROOM : EXX_STREAM_FAILED;
  }
  device->product_stream = (void*)stream;
  return EXX_STREAM_READY;
}

static int ExactExchange_Device_Fits(size_t requested, size_t cached, size_t free_bytes, size_t reserve)
{
  const size_t extra=requested>cached ? requested-cached : 0;
  return free_bytes>=reserve && extra<=free_bytes-reserve;
}

static void ExactExchange_Device_Stream_Back(OpenMXExactExchangeDevice *device);

int Exact_Exchange_Device_Dgemm(OpenMXExactExchangeDevice *device, char transa, char transb, int m, int n, int k,
                                double alpha, const double *a, int lda, const double *b, int ldb, double beta,
                                double *c, int ldc)
{
  if (device==NULL || device->disabled) return 0;
  if (m<1 || n<1 || k<1 || beta!=0.0) return 0;
  if ((transa!='N' && transa!='T') || (transb!='N' && transb!='T')) return 0;
  const double work = (double)m*n*k;
  if (work < device->minimum_work || work < device->dispatch_work) return 0;

  const int rows_a=transa=='N' ? m : k,columns_a=transa=='N' ? k : m;
  const int rows_b=transb=='N' ? k : n,columns_b=transb=='N' ? n : k;
  if (lda!=rows_a || ldb!=rows_b || ldc!=m) return 0;
  if (!gpu_rank_device_usable()) return 0;

  const size_t elements[3]={(size_t)rows_a*columns_a,(size_t)rows_b*columns_b,(size_t)m*n};
  size_t sizes[3],bytes=0;
  for (int i=0; i<3; i++) {
    if (elements[i]>(SIZE_MAX-bytes)/sizeof(double)) return 0;
    sizes[i] = elements[i]*sizeof(double);
    bytes += sizes[i];
  }

  /* The caller has drained the whole call before retrying the products. */
  ExactExchange_Device_Stream_Back(device);
  int current=-1;
  OpenMXGpuError done=ExactExchange_Device_Current(device,&current);
  if (done!=OPENMX_GPU_SUCCESS)
    return ExactExchange_Device_Fail(device,OPENMX_EXACT_EXCHANGE_DEVICE_HANDLE,(int)done);
  ExactExchange_Device_Free_Buffers(device->call_buffers,OPENMX_EXACT_EXCHANGE_CALL_BUFFERS);
  if (bytes>device->retain_bytes) ExactExchange_Device_Free_Buffers(device->product_buffers,3);

  /* Room first: a product that does not fit is an ordinary case. */
  size_t free_bytes=0,total_bytes=0;
  const OpenMXGpuError queried=OpenMX_GpuMemGetInfo(&free_bytes,&total_bytes);
  if (queried!=OPENMX_GPU_SUCCESS)
    return ExactExchange_Device_Fail(device,OPENMX_EXACT_EXCHANGE_DEVICE_MEMORY_QUERY,(int)queried);
  size_t reusable=0;
  for (int i=0; i<3; i++) {
    const size_t cached=device->product_buffers[i].bytes;
    reusable += cached<sizes[i] ? cached : sizes[i];
  }
  if (!ExactExchange_Device_Fits(bytes,reusable,free_bytes,device->reserve_bytes))
    return ExactExchange_Device_Decline(device);

  /* The memory can be gone by now: another process that shares the device
     may have taken it since the query. Running out of it at the handle, at
     an allocation or inside the product is the same ordinary case. */
  const OpenMXBlasStatus handled=ExactExchange_Device_Handle(device,current);
  if (handled==OPENMX_BLAS_ALLOC_FAILED) return ExactExchange_Device_Decline(device);
  if (handled!=OPENMX_BLAS_SUCCESS)
    return ExactExchange_Device_Fail(device,OPENMX_EXACT_EXCHANGE_DEVICE_HANDLE,(int)handled);

  int stream_status=0;
  const int prepared=ExactExchange_Device_Private_Stream(device,&stream_status);
  if (prepared==EXX_STREAM_NO_ROOM) return ExactExchange_Device_Decline(device);
  if (prepared!=EXX_STREAM_READY)
    return ExactExchange_Device_Fail(device,OPENMX_EXACT_EXCHANGE_DEVICE_STREAM,stream_status);
  OpenMXGpuStream stream=NULL;
  const OpenMXBlasStatus streamed=OpenMXBlasGetStream((OpenMXBlasHandle)device->blas,&stream);
  if (streamed!=OPENMX_BLAS_SUCCESS)
    return ExactExchange_Device_Fail(device,OPENMX_EXACT_EXCHANGE_DEVICE_STREAM,(int)streamed);

  OpenMXExactExchangeBuffer *buffers=device->product_buffers;
  for (int i=0; i<3 && done==OPENMX_GPU_SUCCESS; i++)
    done = ExactExchange_Device_Reserve_Buffer(&buffers[i],sizes[i]);
  if (done==OPENMX_GPU_OUT_OF_MEMORY) return ExactExchange_Device_Decline(device);
  if (done!=OPENMX_GPU_SUCCESS)
    return ExactExchange_Device_Fail(device,OPENMX_EXACT_EXCHANGE_DEVICE_ALLOCATION,(int)done);

  double *da=buffers[0].pointer,*db=buffers[1].pointer,*dc=buffers[2].pointer;
  done = OpenMXGpuMemcpyAsync(da,a,sizes[0],OPENMX_GPU_COPY_HOST_TO_DEVICE,stream);
  if (done==OPENMX_GPU_SUCCESS) done = OpenMXGpuMemcpyAsync(db,b,sizes[1],OPENMX_GPU_COPY_HOST_TO_DEVICE,stream);
  if (done!=OPENMX_GPU_SUCCESS) {
    (void)OpenMXGpuStreamSynchronize(stream);
    return ExactExchange_Device_Fail(device,OPENMX_EXACT_EXCHANGE_DEVICE_UPLOAD,(int)done);
  }

  const OpenMXBlasOperation operation_a=transa=='N' ? OPENMX_BLAS_OP_N : OPENMX_BLAS_OP_T;
  const OpenMXBlasOperation operation_b=transb=='N' ? OPENMX_BLAS_OP_N : OPENMX_BLAS_OP_T;
  /* The bridge says only through its count whether a call ran in FP64. */
  const long long native_before=device->gemmul8 ? openmx_gemmul8NativeCalls(0) : 0;
  const OpenMXBlasStatus multiplied=device->gemmul8
    ? openmx_gemmul8Dgemm((OpenMXBlasHandle)device->blas,operation_a,operation_b,m,n,k,&alpha,da,lda,db,ldb,&beta,
                          dc,ldc)
    : OpenMXBlasDgemm((OpenMXBlasHandle)device->blas,operation_a,operation_b,m,n,k,&alpha,da,lda,db,ldb,&beta,dc,
                      ldc);
  const int by_gemmul8=device->gemmul8 && openmx_gemmul8NativeCalls(0)==native_before;
  if (multiplied!=OPENMX_BLAS_SUCCESS) {
    (void)OpenMXGpuStreamSynchronize(stream);
    if (multiplied==OPENMX_BLAS_ALLOC_FAILED) return ExactExchange_Device_Decline(device);
    return ExactExchange_Device_Fail(device,OPENMX_EXACT_EXCHANGE_DEVICE_PRODUCT,(int)multiplied);
  }

  done = OpenMXGpuMemcpyAsync(c,dc,sizes[2],OPENMX_GPU_COPY_DEVICE_TO_HOST,stream);
  const OpenMXGpuError waited=OpenMXGpuStreamSynchronize(stream);
  if (done!=OPENMX_GPU_SUCCESS)
    return ExactExchange_Device_Fail(device,OPENMX_EXACT_EXCHANGE_DEVICE_DOWNLOAD,(int)done);
  if (waited!=OPENMX_GPU_SUCCESS)
    return ExactExchange_Device_Fail(device,OPENMX_EXACT_EXCHANGE_DEVICE_WAIT,(int)waited);

  if (device->retain_bytes<ExactExchange_Device_Buffer_Bytes(buffers,3))
    ExactExchange_Device_Free_Buffers(buffers,3);
  device->device_products++;
  if (by_gemmul8) device->gemmul8_products++;
  if (device->largest_bytes<bytes) device->largest_bytes = bytes;
  return 1;
}

/* ---- whole calls of modules/exx ----------------------------------------- */

/* Restore the handle's stream before individual products use it again.
   A handle that refuses is returned and created on the next use. */
static void ExactExchange_Device_Stream_Back(OpenMXExactExchangeDevice *device)
{
#if OPENMX_EXACT_EXCHANGE_DEVICE_CALLS
  if (device->call_stream_set && device->blas!=NULL
      && OpenMXBlasSetStream((OpenMXBlasHandle)device->blas,(OpenMXGpuStream)device->call_stream_before)
         !=OPENMX_BLAS_SUCCESS)
    Exact_Exchange_Device_Release(device);
#endif
  device->call_stream_set = 0;
  device->call_stream = NULL;
  device->call_stream_before = NULL;
}

int Exact_Exchange_Device_Call_End(OpenMXExactExchangeDevice *device)
{
  int on_device=0;
  if (device==NULL) return 0;
  ExactExchange_Device_Stream_Back(device);
  if (device->call_accepted && !device->call_failed) {
    if (device->call_without_room) device->declined_calls++;
    else if (device->call_downloaded) {
      device->device_calls++;
      device->call_products += device->call_products_now;
      device->call_large_products += device->call_large_now;
      device->call_gemmul8_products += device->call_gemmul8_now;
      if (device->largest_call_bytes<device->call_bytes) device->largest_call_bytes = device->call_bytes;
      on_device = 1;
    }
  }
  if (device->call_without_room || device->call_failed || device->retain_bytes <
      ExactExchange_Device_Buffer_Bytes(device->call_buffers,OPENMX_EXACT_EXCHANGE_CALL_BUFFERS))
    ExactExchange_Device_Free_Buffers(device->call_buffers,OPENMX_EXACT_EXCHANGE_CALL_BUFFERS);
  ExactExchange_Device_Call_Clear(device);
  return on_device;
}

#if OPENMX_EXACT_EXCHANGE_DEVICE_CALLS

/* A device call of a whole call failed: the device takes no further call or
   product, and the first failure is the one that is recorded. Returns the
   value of a function of the table that failed. */
static int ExactExchange_Device_Call_Fail(OpenMXExactExchangeDevice *device, int failure, int status)
{
  ExactExchange_Device_Note_Failure(device,failure,status);
  device->call_failed = 1;
  return 1;
}

/* multiplications is the sum of m*n*k over the products of the call: a call
   with fewer than minimum_work stays on the host. Then room for the arrays
   of the call, as for a product, and the handle of the BLAS, set to the
   stream of the queue on which the moves run. */
static int ExactExchange_Device_Accept(void *context, double multiplications, size_t bytes)
{
  OpenMXExactExchangeDevice *device=(OpenMXExactExchangeDevice*)context;
  ExactExchange_Device_Stream_Back(device);
  ExactExchange_Device_Call_Clear(device);
  if (device->disabled || multiplications<device->minimum_work) return 0;
  if (!gpu_rank_device_usable()) return 0;

  int current=-1;
  const OpenMXGpuError bound=ExactExchange_Device_Current(device,&current);
  if (bound!=OPENMX_GPU_SUCCESS) {
    (void)ExactExchange_Device_Call_Fail(device,OPENMX_EXACT_EXCHANGE_DEVICE_HANDLE,(int)bound);
    return 0;
  }
  ExactExchange_Device_Free_Buffers(device->product_buffers,3);
  device->call_cached = bytes!=0 && bytes<=device->retain_bytes;
  if (!device->call_cached)
    ExactExchange_Device_Free_Buffers(device->call_buffers,OPENMX_EXACT_EXCHANGE_CALL_BUFFERS);

  size_t free_bytes=0,total_bytes=0;
  const OpenMXGpuError queried=OpenMX_GpuMemGetInfo(&free_bytes,&total_bytes);
  if (queried!=OPENMX_GPU_SUCCESS) {
    (void)ExactExchange_Device_Call_Fail(device,OPENMX_EXACT_EXCHANGE_DEVICE_MEMORY_QUERY,(int)queried);
    return 0;
  }
  const size_t cached=ExactExchange_Device_Buffer_Bytes(device->call_buffers,OPENMX_EXACT_EXCHANGE_CALL_BUFFERS);
  if (!ExactExchange_Device_Fits(bytes,cached,free_bytes,device->reserve_bytes)) {
    ExactExchange_Device_Free_Buffers(device->call_buffers,OPENMX_EXACT_EXCHANGE_CALL_BUFFERS);
    device->declined_calls++;
    return 0;
  }

  const OpenMXBlasStatus handled=ExactExchange_Device_Handle(device,current);
  if (handled==OPENMX_BLAS_ALLOC_FAILED) {
    (void)OpenMXGpuGetLastError();
    device->declined_calls++;
    return 0;
  }
  if (handled!=OPENMX_BLAS_SUCCESS) {
    (void)ExactExchange_Device_Call_Fail(device,OPENMX_EXACT_EXCHANGE_DEVICE_HANDLE,(int)handled);
    return 0;
  }

#if OPENMX_GPU_CUDA
  void *stream=acc_get_cuda_stream(OPENMX_EXACT_EXCHANGE_DEVICE_QUEUE);
  if (stream==NULL) {
    (void)ExactExchange_Device_Call_Fail(device,OPENMX_EXACT_EXCHANGE_DEVICE_STREAM,-1);
    return 0;
  }
  OpenMXGpuStream before=NULL;
  OpenMXBlasStatus streamed=OpenMXBlasGetStream((OpenMXBlasHandle)device->blas,&before);
  if (streamed==OPENMX_BLAS_SUCCESS)
    streamed = OpenMXBlasSetStream((OpenMXBlasHandle)device->blas,(OpenMXGpuStream)stream);
  if (streamed!=OPENMX_BLAS_SUCCESS) {
    (void)ExactExchange_Device_Call_Fail(device,OPENMX_EXACT_EXCHANGE_DEVICE_STREAM,(int)streamed);
    return 0;
  }
  device->call_stream_set = 1;
  device->call_stream = stream;
  device->call_stream_before = (void*)before;
#else
  int stream_status=0;
  const int prepared=ExactExchange_Device_Private_Stream(device,&stream_status);
  if (prepared==EXX_STREAM_NO_ROOM) {
    (void)OpenMXGpuGetLastError();
    device->declined_calls++;
    return 0;
  }
  if (prepared!=EXX_STREAM_READY) {
    (void)ExactExchange_Device_Call_Fail(device,OPENMX_EXACT_EXCHANGE_DEVICE_STREAM,stream_status);
    return 0;
  }
  device->call_stream = device->product_stream;
#endif
  device->call_accepted = 1;
  device->call_bytes = bytes;
  return 1;
}

/* NULL without room, as when another process took the memory after the
   query, and after a failure. */
static void *ExactExchange_Device_Allocate(void *context, size_t bytes)
{
  OpenMXExactExchangeDevice *device=(OpenMXExactExchangeDevice*)context;
  void *pointer=NULL;
  OpenMXGpuError done;
  if (device->call_cached) {
    if (device->call_next_buffer>=OPENMX_EXACT_EXCHANGE_CALL_BUFFERS) {
      device->call_without_room = 1;
      return NULL;
    }
    OpenMXExactExchangeBuffer *buffer=&device->call_buffers[device->call_next_buffer++];
    done = ExactExchange_Device_Reserve_Buffer(buffer,bytes);
    pointer = buffer->pointer;
  }
  else done = OpenMXGpuMalloc(&pointer,bytes);
  if (done==OPENMX_GPU_SUCCESS) return pointer;
  if (done==OPENMX_GPU_OUT_OF_MEMORY) {
    (void)OpenMXGpuGetLastError();
    device->call_without_room = 1;
  }
  else (void)ExactExchange_Device_Call_Fail(device,OPENMX_EXACT_EXCHANGE_DEVICE_ALLOCATION,(int)done);
  return NULL;
}

static void ExactExchange_Device_Release_Block(void *context, void *pointer)
{
  const OpenMXExactExchangeDevice *device=(const OpenMXExactExchangeDevice*)context;
  if (!device->call_cached && OpenMXGpuFree(pointer)!=OPENMX_GPU_SUCCESS) (void)OpenMXGpuGetLastError();
}

/* The caller retains the input through finish. Put the copy ahead of its
   consumers on the same queue. */
static int ExactExchange_Device_Upload(void *context, void *target, const void *source, size_t bytes)
{
  OpenMXExactExchangeDevice *device=(OpenMXExactExchangeDevice*)context;
  const OpenMXGpuError done=OpenMXGpuMemcpyAsync(target,source,bytes,OPENMX_GPU_COPY_HOST_TO_DEVICE,
                                               (OpenMXGpuStream)device->call_stream);
  if (done!=OPENMX_GPU_SUCCESS)
    return ExactExchange_Device_Call_Fail(device,OPENMX_EXACT_EXCHANGE_DEVICE_UPLOAD,(int)done);
  return 0;
}

static int ExactExchange_Device_Download(void *context, void *target, const void *source, size_t bytes)
{
  OpenMXExactExchangeDevice *device=(OpenMXExactExchangeDevice*)context;
  const OpenMXGpuStream stream=(OpenMXGpuStream)device->call_stream;
  const OpenMXGpuError done=OpenMXGpuMemcpyAsync(target,source,bytes,OPENMX_GPU_COPY_DEVICE_TO_HOST,stream);
  const OpenMXGpuError waited=OpenMXGpuStreamSynchronize(stream);
  if (done!=OPENMX_GPU_SUCCESS)
    return ExactExchange_Device_Call_Fail(device,OPENMX_EXACT_EXCHANGE_DEVICE_DOWNLOAD,(int)done);
  if (waited!=OPENMX_GPU_SUCCESS)
    return ExactExchange_Device_Call_Fail(device,OPENMX_EXACT_EXCHANGE_DEVICE_DOWNLOAD,(int)waited);
  device->call_downloaded = 1;
  return 0;
}

/* A product on the stream of the queue; it may run after the return. The
   large ones go through the GEMMul8 bridge when the caller asked for it. */
static int ExactExchange_Device_Call_Dgemm(void *context, char transa, char transb, int m, int n, int k,
                                           double alpha, const double *a, int lda, const double *b, int ldb,
                                           double beta, double *c, int ldc)
{
  OpenMXExactExchangeDevice *device=(OpenMXExactExchangeDevice*)context;
  const OpenMXBlasOperation operation_a=transa=='N' ? OPENMX_BLAS_OP_N : OPENMX_BLAS_OP_T;
  const OpenMXBlasOperation operation_b=transb=='N' ? OPENMX_BLAS_OP_N : OPENMX_BLAS_OP_T;
  const int large=device->minimum_work<=(double)m*(double)n*(double)k;
  const int bridge=device->gemmul8 && large;
  const long long native_before=bridge ? openmx_gemmul8NativeCalls(0) : 0;
  const OpenMXBlasStatus multiplied=bridge
    ? openmx_gemmul8Dgemm((OpenMXBlasHandle)device->blas,operation_a,operation_b,m,n,k,&alpha,a,lda,b,ldb,&beta,c,
                          ldc)
    : OpenMXBlasDgemm((OpenMXBlasHandle)device->blas,operation_a,operation_b,m,n,k,&alpha,a,lda,b,ldb,&beta,c,ldc);
  if (multiplied==OPENMX_BLAS_ALLOC_FAILED) {
    (void)OpenMXGpuGetLastError();
    device->call_without_room = 1;
    return 1;
  }
  if (multiplied!=OPENMX_BLAS_SUCCESS)
    return ExactExchange_Device_Call_Fail(device,OPENMX_EXACT_EXCHANGE_DEVICE_PRODUCT,(int)multiplied);
  device->call_products_now++;
  if (large) device->call_large_now++;
  if (bridge && openmx_gemmul8NativeCalls(0)==native_before) device->call_gemmul8_now++;
  return 0;
}

/* The move of exx.h: target(r, c) = alpha source(r, c), or += with add, for
   r < rows and c < lengths[r] (columns without lengths). The library gives
   no two pairs (r, c) the same target, so the pairs run in parallel. The
   product comes first, then the sum: the unit is compiled without fused
   multiply-adds in device code (source/Makefile). */
static int ExactExchange_Device_Move(void *context, int rows, int columns, const int *lengths, double alpha, int add,
                                     double *target, const OpenMXExxDeviceIndex *target_index, const double *source,
                                     const OpenMXExxDeviceIndex *source_index)
{
#if OPENMX_GPU_HIP
  OpenMXExactExchangeDevice *device=(OpenMXExactExchangeDevice*)context;
  const OpenMXGpuStream stream=(OpenMXGpuStream)device->call_stream;
  const OpenMXGpuError moved=Exact_Exchange_HIP_Move(stream,rows,columns,lengths,alpha,add,
                                                     target,*target_index,source,*source_index);
  if (moved!=OPENMX_GPU_SUCCESS)
    return ExactExchange_Device_Call_Fail(device,OPENMX_EXACT_EXCHANGE_DEVICE_PRODUCT,(int)moved);
#else
  (void)context;
  const long long target_stride=target_index->stride,target_step=target_index->step;
  const long long *target_rows=target_index->rows,*target_columns=target_index->columns;
  const long long source_stride=source_index->stride,source_step=source_index->step;
  const long long *source_rows=source_index->rows,*source_columns=source_index->columns;
#ifdef _OPENACC
#pragma acc parallel loop collapse(2) async(OPENMX_EXACT_EXCHANGE_DEVICE_QUEUE) \
    deviceptr(target,source,lengths,target_rows,target_columns,source_rows,source_columns)
#endif
  for (int r=0; r<rows; r++) {
    for (int c=0; c<columns; c++) {
      if (lengths==NULL || c<lengths[r]) {
        const long long t=(target_rows!=NULL ? target_rows[r] : (long long)r*target_stride)
                         +(target_columns!=NULL ? target_columns[c] : (long long)c*target_step);
        const long long s=(source_rows!=NULL ? source_rows[r] : (long long)r*source_stride)
                         +(source_columns!=NULL ? source_columns[c] : (long long)c*source_step);
        const double value=alpha*source[s];
        if (add) target[t] += value;
        else target[t] = value;
      }
    }
  }
#endif
  return 0;
}

/* The moves of the queue and the products on its stream, and the errors of
   both. */
static int ExactExchange_Device_Finish(void *context)
{
  OpenMXExactExchangeDevice *device=(OpenMXExactExchangeDevice*)context;
  const OpenMXGpuError done=OpenMXGpuStreamSynchronize((OpenMXGpuStream)device->call_stream);
  if (done!=OPENMX_GPU_SUCCESS)
    return ExactExchange_Device_Call_Fail(device,OPENMX_EXACT_EXCHANGE_DEVICE_WAIT,(int)done);
  return 0;
}

#endif /* OPENMX_EXACT_EXCHANGE_DEVICE_CALLS */

const OpenMXExxDevice *Exact_Exchange_Device_Table(OpenMXExactExchangeDevice *device, OpenMXExxDevice *table)
{
#if OPENMX_EXACT_EXCHANGE_DEVICE_CALLS
  if (device==NULL || table==NULL) return NULL;
  table->context = device;
  table->accept = ExactExchange_Device_Accept;
  table->allocate = ExactExchange_Device_Allocate;
  table->release = ExactExchange_Device_Release_Block;
  table->upload = ExactExchange_Device_Upload;
  table->download = ExactExchange_Device_Download;
  table->dgemm = ExactExchange_Device_Call_Dgemm;
  table->move = ExactExchange_Device_Move;
  table->finish = ExactExchange_Device_Finish;
  return table;
#else
  (void)device;
  (void)table;
  return NULL;
#endif
}

/**********************************************************************
  Exact_Exchange_Device.c:

     Matrix products of the exchange build on the GPU of this process.

     See docs/EXX_IMPLEMENTATION_GUIDE.md, section 11, and
     openmx_exact_exchange_device.h for the interface. One product
     allocates its three matrices on the device, copies the two factors,
     calls the FP64 product of the device BLAS and copies the result
     back. Nothing stays on the device between two products except the
     handle of the BLAS.

     A product that does not fit is left to the host and the device stays
     in use; so is one for which the device runs out of memory after the
     check, as it can when several processes share the device. Any other
     device call that fails switches the device off for the rest of the
     calculation: the host then computes every product, the failed one
     included.

     The functions take every input as an argument and read no global
     variable.
***********************************************************************/

#include <stddef.h>
#include "openmx_gpu_backend.h"
#include "openmx_gpu_runtime.h"
#include "openmx_gpu_blas.h"
#include "openmx_gpu_device.h"
#include "openmx_exact_exchange_device.h"

void Exact_Exchange_Device_Initialize(OpenMXExactExchangeDevice *device)
{
  device->blas = NULL;
  device->blas_device = -1;
  device->disabled = 0;
  device->failure = OPENMX_EXACT_EXCHANGE_DEVICE_OK;
  device->failure_status = 0;
  /* The two limits are first values and not measured on a device yet
     (guide, section 14). 1e9 multiplications keep the products of one V
     neighbour on the host and send the two products per centre atom that
     run over all columns to the device. */
  device->minimum_work = 1.0e9;
  device->reserve_bytes = (size_t)512*1024*1024;
  device->device_products = 0;
  device->declined_products = 0;
  device->largest_bytes = 0;
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
  default:                                        return "an unknown step";
  }
}

void Exact_Exchange_Device_Release(OpenMXExactExchangeDevice *device)
{
  if (device->blas!=NULL) {
    /* A handle goes back on the device that made it. The process is bound
       to that device unless the binding changed since the last product. */
    int current=-1;
    const int moved=OpenMXGpuGetDevice(&current)==OPENMX_GPU_SUCCESS && current!=device->blas_device
                    && OpenMXGpuSetDevice(device->blas_device)==OPENMX_GPU_SUCCESS;
    (void)OpenMXBlasDestroy((OpenMXBlasHandle)device->blas);
    if (moved) (void)OpenMXGpuSetDevice(current);
    /* a handle that outlived its device leaves an error behind */
    (void)OpenMXGpuGetLastError();
  }
  device->blas = NULL;
  device->blas_device = -1;
}

/* The matrices of a product that the device does not finish go back, and
   the error of the runtime is taken off its latch so that it does not reach
   the next caller that asks for one. */
static void ExactExchange_Device_Abandon(double *da, double *db, double *dc)
{
  if (da!=NULL) (void)OpenMXGpuFree(da);
  if (db!=NULL) (void)OpenMXGpuFree(db);
  if (dc!=NULL) (void)OpenMXGpuFree(dc);
  (void)OpenMXGpuGetLastError();
}

/* The device has no room for this product: the host computes it, and the
   device stays in use. */
static int ExactExchange_Device_Decline(OpenMXExactExchangeDevice *device, double *da, double *db, double *dc)
{
  ExactExchange_Device_Abandon(da,db,dc);
  device->declined_products++;
  return 0;
}

/* A device call failed for another reason than room: the device takes no
   further product. */
static int ExactExchange_Device_Fail(OpenMXExactExchangeDevice *device, int failure, int status, double *da,
                                     double *db, double *dc)
{
  ExactExchange_Device_Abandon(da,db,dc);
  device->disabled = 1;
  device->failure = failure;
  device->failure_status = status;
  return 0;
}

/* The handle of the device this process is bound to. A handle of another
   device is given back first. Returns the status of the BLAS, with
   *runtime the status of the runtime when the binding could not be read. */
static OpenMXBlasStatus ExactExchange_Device_Handle(OpenMXExactExchangeDevice *device, OpenMXGpuError *runtime)
{
  int current=-1;
  *runtime = OpenMXGpuGetDevice(&current);
  if (*runtime!=OPENMX_GPU_SUCCESS) return OPENMX_BLAS_SUCCESS;
  if (device->blas!=NULL && device->blas_device!=current) Exact_Exchange_Device_Release(device);
  if (device->blas==NULL) {
    OpenMXBlasHandle created=NULL;
    const OpenMXBlasStatus made=OpenMXBlasCreate(&created);
    if (made!=OPENMX_BLAS_SUCCESS) return made;
    device->blas = (void*)created;
    device->blas_device = current;
  }
  return OPENMX_BLAS_SUCCESS;
}

int Exact_Exchange_Device_Dgemm(OpenMXExactExchangeDevice *device, char transa, char transb, int m, int n, int k,
                                double alpha, const double *a, int lda, const double *b, int ldb, double beta,
                                double *c, int ldc)
{
  if (device==NULL || device->disabled) return 0;
  if (m<1 || n<1 || k<1 || beta!=0.0) return 0;
  if ((transa!='N' && transa!='T') || (transb!='N' && transb!='T')) return 0;
  if ((double)m*(double)n*(double)k<device->minimum_work) return 0;

  const int rows_a=transa=='N' ? m : k,columns_a=transa=='N' ? k : m;
  const int rows_b=transb=='N' ? k : n,columns_b=transb=='N' ? n : k;
  if (lda!=rows_a || ldb!=rows_b || ldc!=m) return 0;
  if (!gpu_rank_device_usable()) return 0;

  const size_t bytes_a=sizeof(double)*(size_t)rows_a*(size_t)columns_a;
  const size_t bytes_b=sizeof(double)*(size_t)rows_b*(size_t)columns_b;
  const size_t bytes_c=sizeof(double)*(size_t)m*(size_t)n;
  const size_t bytes=bytes_a+bytes_b+bytes_c;

  /* Room first: a product that does not fit is an ordinary case. */
  size_t free_bytes=0,total_bytes=0;
  const OpenMXGpuError queried=OpenMX_GpuMemGetInfo(&free_bytes,&total_bytes);
  if (queried!=OPENMX_GPU_SUCCESS)
    return ExactExchange_Device_Fail(device,OPENMX_EXACT_EXCHANGE_DEVICE_MEMORY_QUERY,(int)queried,NULL,NULL,NULL);
  if (free_bytes<device->reserve_bytes || free_bytes-device->reserve_bytes<bytes)
    return ExactExchange_Device_Decline(device,NULL,NULL,NULL);

  /* The memory can be gone by now: another process that shares the device
     may have taken it since the query. Running out of it at the handle, at
     an allocation or inside the product is the same ordinary case. */
  OpenMXGpuError done=OPENMX_GPU_SUCCESS;
  const OpenMXBlasStatus handled=ExactExchange_Device_Handle(device,&done);
  if (done!=OPENMX_GPU_SUCCESS)
    return ExactExchange_Device_Fail(device,OPENMX_EXACT_EXCHANGE_DEVICE_HANDLE,(int)done,NULL,NULL,NULL);
  if (handled==OPENMX_BLAS_ALLOC_FAILED) return ExactExchange_Device_Decline(device,NULL,NULL,NULL);
  if (handled!=OPENMX_BLAS_SUCCESS)
    return ExactExchange_Device_Fail(device,OPENMX_EXACT_EXCHANGE_DEVICE_HANDLE,(int)handled,NULL,NULL,NULL);

  double *da=NULL,*db=NULL,*dc=NULL;
  done = OpenMXGpuMalloc((void**)&da,bytes_a);
  if (done==OPENMX_GPU_SUCCESS) done = OpenMXGpuMalloc((void**)&db,bytes_b);
  if (done==OPENMX_GPU_SUCCESS) done = OpenMXGpuMalloc((void**)&dc,bytes_c);
  if (done==OPENMX_GPU_OUT_OF_MEMORY) return ExactExchange_Device_Decline(device,da,db,dc);
  if (done!=OPENMX_GPU_SUCCESS)
    return ExactExchange_Device_Fail(device,OPENMX_EXACT_EXCHANGE_DEVICE_ALLOCATION,(int)done,da,db,dc);

  done = OpenMXGpuMemcpy(da,a,bytes_a,OPENMX_GPU_COPY_HOST_TO_DEVICE);
  if (done==OPENMX_GPU_SUCCESS) done = OpenMXGpuMemcpy(db,b,bytes_b,OPENMX_GPU_COPY_HOST_TO_DEVICE);
  if (done!=OPENMX_GPU_SUCCESS)
    return ExactExchange_Device_Fail(device,OPENMX_EXACT_EXCHANGE_DEVICE_UPLOAD,(int)done,da,db,dc);

  const OpenMXBlasOperation operation_a=transa=='N' ? OPENMX_BLAS_OP_N : OPENMX_BLAS_OP_T;
  const OpenMXBlasOperation operation_b=transb=='N' ? OPENMX_BLAS_OP_N : OPENMX_BLAS_OP_T;
  const OpenMXBlasStatus multiplied=OpenMXBlasDgemm((OpenMXBlasHandle)device->blas,operation_a,operation_b,m,n,k,
                                                    &alpha,da,lda,db,ldb,&beta,dc,ldc);
  if (multiplied==OPENMX_BLAS_ALLOC_FAILED) return ExactExchange_Device_Decline(device,da,db,dc);
  if (multiplied!=OPENMX_BLAS_SUCCESS)
    return ExactExchange_Device_Fail(device,OPENMX_EXACT_EXCHANGE_DEVICE_PRODUCT,(int)multiplied,da,db,dc);
  /* the product runs after the call returned: its errors arrive here */
  done = OpenMXGpuDeviceSynchronize();
  if (done!=OPENMX_GPU_SUCCESS)
    return ExactExchange_Device_Fail(device,OPENMX_EXACT_EXCHANGE_DEVICE_PRODUCT,(int)done,da,db,dc);

  done = OpenMXGpuMemcpy(c,dc,bytes_c,OPENMX_GPU_COPY_DEVICE_TO_HOST);
  if (done!=OPENMX_GPU_SUCCESS)
    return ExactExchange_Device_Fail(device,OPENMX_EXACT_EXCHANGE_DEVICE_DOWNLOAD,(int)done,da,db,dc);

  (void)OpenMXGpuFree(da);
  (void)OpenMXGpuFree(db);
  (void)OpenMXGpuFree(dc);
  device->device_products++;
  if (device->largest_bytes<bytes) device->largest_bytes = bytes;
  return 1;
}

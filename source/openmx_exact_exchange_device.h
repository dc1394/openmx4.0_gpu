#ifndef OPENMX_EXACT_EXCHANGE_DEVICE_H
#define OPENMX_EXACT_EXCHANGE_DEVICE_H

/* Matrix products of the exchange build on the GPU of this process
   (docs/EXX_IMPLEMENTATION_GUIDE.md, section 11).

   modules/exx asks for every product through its table of functions
   (section 7.2). The layer source/Exact_Exchange.c offers each product to
   Exact_Exchange_Device_Dgemm first. The device takes it when the process
   has a usable device, the product is large enough to pay for the two
   transfers, and the three matrices fit into the free device memory with
   the reserve below left over. In every other case the function returns 0
   and the host computes the product: the result does not depend on where a
   product ran beyond the rounding of the two BLAS libraries.

   A product without room is counted and the device stays in use; that
   includes a device that runs out of memory after the check, as it can
   when several processes share it. A device call that fails for another
   reason switches the device off: the host computes that product and every
   later one, and the caller reports the failure once.

   The arithmetic is FP64 (cuBLAS). GEMMul8 is not used here: section 1.2
   of the guide asks for the measured difference first. */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Why the device stopped taking products; 0 while it takes them. */
enum {
  OPENMX_EXACT_EXCHANGE_DEVICE_OK = 0,
  OPENMX_EXACT_EXCHANGE_DEVICE_MEMORY_QUERY,
  OPENMX_EXACT_EXCHANGE_DEVICE_HANDLE,
  OPENMX_EXACT_EXCHANGE_DEVICE_ALLOCATION,
  OPENMX_EXACT_EXCHANGE_DEVICE_UPLOAD,
  OPENMX_EXACT_EXCHANGE_DEVICE_PRODUCT,
  OPENMX_EXACT_EXCHANGE_DEVICE_DOWNLOAD
};

typedef struct {
  void *blas;                  /* handle of the BLAS on the device; NULL until the first product */
  int blas_device;             /* device the handle belongs to */
  int disabled;                /* a device call failed: the host computes every later product */
  int failure;                 /* OPENMX_EXACT_EXCHANGE_DEVICE_..., the call that failed */
  int failure_status;          /* status that call returned */
  double minimum_work;         /* products with fewer multiplications m*n*k stay on the host */
  size_t reserve_bytes;        /* device memory that a product leaves free */
  long long device_products;   /* products computed on the device */
  long long declined_products; /* large enough, but the device had no room for the matrices */
  size_t largest_bytes;        /* most device memory one product took */
} OpenMXExactExchangeDevice;

/* The values of a new calculation: nothing on the device, the default
   limits. Does not touch the device. */
void Exact_Exchange_Device_Initialize(OpenMXExactExchangeDevice *device);

/* c = alpha*op(a)*op(b), column-major, with the arguments of DGEMM.
   Returns 1 when c holds the product computed on the device. Returns 0
   when the host has to compute it; the content of c is then unspecified,
   which is why only products with beta = 0 are taken. Matrices with
   padding between the columns (lda, ldb or ldc above the number of rows)
   stay on the host: modules/exx has none. */
int Exact_Exchange_Device_Dgemm(OpenMXExactExchangeDevice *device, char transa, char transb, int m, int n, int k,
                                double alpha, const double *a, int lda, const double *b, int ldb, double beta,
                                double *c, int ldc);

/* Text for the value of failure, for a message. */
const char *Exact_Exchange_Device_Failure_Name(int failure);

/* Gives the handle back. The counters stay. */
void Exact_Exchange_Device_Release(OpenMXExactExchangeDevice *device);

#ifdef __cplusplus
}
#endif

#endif /* OPENMX_EXACT_EXCHANGE_DEVICE_H */

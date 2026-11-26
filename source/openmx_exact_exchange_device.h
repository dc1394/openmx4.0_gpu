#ifndef OPENMX_EXACT_EXCHANGE_DEVICE_H
#define OPENMX_EXACT_EXCHANGE_DEVICE_H

/* Matrix products of the exchange build on the GPU of this process
   (docs/EXX_IMPLEMENTATION_GUIDE.md, section 11).

   modules/exx asks for every product through its table of functions
   (section 7.2). The layer source/Exact_Exchange.c offers each product to
   Exact_Exchange_Device_Dgemm first. The device takes it when the process
   has a usable device, the product meets the dispatch threshold, and the
   three matrices fit into the reusable and free device memory with
   the reserve below left over. In every other case the function returns 0
   and the host computes the product: the result does not depend on where a
   product ran beyond the rounding of the two BLAS libraries.

   A product without room is counted and the device stays in use; that
   includes a device that runs out of memory after the check, as it can
   when several processes share it. A device call that fails for another
   reason switches the device off: the host computes that product and every
   later one, and the caller reports the failure once.

   The arithmetic is FP64 (cuBLAS) unless the caller sets gemmul8. The
   product then goes through the bridge openmx_gemmul8Dgemm, as the products
   of the eigensolvers do: GEMMul8 computes it, or FP64 when the bridge has
   no room for its workspace or the environment switches GEMMul8 off. The
   bridge keeps its workspace between two calls, bounded by its own cap.
   The application reuses its own buffers within a separate per-rank limit.

   Whole calls. The CUDA build also offers modules/exx a device for whole
   calls of openmx_exx_add_blocks and openmx_exx_add_gamma
   (Exact_Exchange_Device_Table, OpenMXExxDevice in exx.h): the device then
   computes the contributions of all centre atoms of the call, with the
   arrays in device memory for the call. It takes a call whose products
   have at least minimum_work multiplications in all and whose arrays fit
   into the free device memory with the reserve left over; the other calls
   go to the products above. A call without room is counted and the device
   stays in use; a device call that fails for another reason switches the
   device off for the calls and for the products, as above. Transfers,
   products and moves run in order on the OpenACC queue's stream.
   Products with at least minimum_work multiplications go through the
   GEMMul8 bridge when the caller sets gemmul8; the others are FP64, and a
   call on the device may have none of the first kind. */

#include <stddef.h>
#include "openmx_exx.h"

#ifdef __cplusplus
extern "C" {
#endif

/* The OpenACC queue of the whole calls. */
#define OPENMX_EXACT_EXCHANGE_DEVICE_QUEUE 1
#define OPENMX_EXACT_EXCHANGE_DEVICE_RESERVE ((size_t)64*1024*1024)

/* Why the device stopped taking products; 0 while it takes them. */
enum {
  OPENMX_EXACT_EXCHANGE_DEVICE_OK = 0,
  OPENMX_EXACT_EXCHANGE_DEVICE_MEMORY_QUERY,
  OPENMX_EXACT_EXCHANGE_DEVICE_HANDLE,
  OPENMX_EXACT_EXCHANGE_DEVICE_ALLOCATION,
  OPENMX_EXACT_EXCHANGE_DEVICE_UPLOAD,
  OPENMX_EXACT_EXCHANGE_DEVICE_PRODUCT,
  OPENMX_EXACT_EXCHANGE_DEVICE_DOWNLOAD,
  OPENMX_EXACT_EXCHANGE_DEVICE_STREAM,  /* the stream of the queue for the products of a call */
  OPENMX_EXACT_EXCHANGE_DEVICE_WAIT     /* the wait for the work of a call */
};

typedef struct {
  void *pointer;
  size_t bytes;
} OpenMXExactExchangeBuffer;

/* DeviceArrays in modules/exx uses fewer slots even for force weights. */
enum { OPENMX_EXACT_EXCHANGE_CALL_BUFFERS = 32 };

typedef struct {
  void *blas;                  /* handle of the BLAS on the device; NULL until the first product */
  void *product_stream;        /* nonblocking stream for individual products */
  int blas_device;             /* device the handle belongs to */
  int disabled;                /* a device call failed: the host computes every later product */
  int failure;                 /* OPENMX_EXACT_EXCHANGE_DEVICE_..., the call that failed */
  int failure_status;          /* status that call returned */
  double minimum_work;         /* products with fewer multiplications m*n*k stay on the host, calls with fewer in all */
  double dispatch_work;        /* minimum work of an individual product for GPU dispatch */
  int sharing_ranks;            /* MPI ranks assigned to the same physical device */
  size_t reserve_bytes;        /* device memory that a product leaves free */
  size_t retain_bytes;         /* limit for workspace kept between uses */
  OpenMXExactExchangeBuffer product_buffers[3];
  OpenMXExactExchangeBuffer call_buffers[OPENMX_EXACT_EXCHANGE_CALL_BUFFERS];
  int gemmul8;                 /* products go through the GEMMul8 bridge of the device BLAS */
  long long device_products;   /* products computed on the device */
  long long gemmul8_products;  /* those of them that GEMMul8 computed */
  long long declined_products; /* large enough, but the device had no room for the matrices */
  size_t largest_bytes;        /* most device memory one product took */
  /* whole calls of modules/exx */
  long long device_calls;          /* calls that the device computed */
  long long declined_calls;        /* calls without room on the device: the host computed them */
  long long call_products;         /* matrix products of the calls that the device computed */
  long long call_large_products;   /* those of them with at least minimum_work multiplications */
  long long call_gemmul8_products; /* those of the large ones that GEMMul8 computed */
  size_t largest_call_bytes;       /* most device memory that the arrays of one call took */
  /* the call in progress, from its accept to Exact_Exchange_Device_Call_End */
  int call_accepted;               /* the device took the call */
  int call_without_room;           /* an allocation or a product found no room */
  int call_failed;                 /* a device call failed */
  int call_downloaded;             /* the result came back */
  int call_stream_set;             /* the handle runs on the stream of the queue */
  void *call_stream;               /* stream of the OpenACC queue, borrowed */
  int call_cached;                 /* allocate/release borrow from the owned buffers */
  int call_next_buffer;
  void *call_stream_before;        /* the stream of the handle before */
  size_t call_bytes;               /* of its arrays */
  long long call_products_now,call_large_now,call_gemmul8_now;
} OpenMXExactExchangeDevice;

/* The values of a new calculation: nothing on the device, the default
   limits, FP64. Does not touch the device. */
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

/* Gives buffers, the private stream and the handle back. The counters stay. */
void Exact_Exchange_Device_Release(OpenMXExactExchangeDevice *device);

/* The device for whole calls: fills table, whose context is device, and
   returns it. Returns NULL where whole calls are not offered: the CPU
   build, and a CUDA build without OpenACC. The table is valid while device
   and table are. */
const OpenMXExxDevice *Exact_Exchange_Device_Table(OpenMXExactExchangeDevice *device, OpenMXExxDevice *table);

/* Ends a call of openmx_exx_add_blocks or openmx_exx_add_gamma that got the
   table: counts it and gives the handle its stream back. Returns 1 when the
   device computed the call: it took the call, nothing failed or ran out of
   room, and the result came back. Returns 0 otherwise, also for a call that
   did not ask the device. */
int Exact_Exchange_Device_Call_End(OpenMXExactExchangeDevice *device);

#ifdef __cplusplus
}
#endif

#endif /* OPENMX_EXACT_EXCHANGE_DEVICE_H */

#ifndef OPENMX_CORE_PROJECTOR_EXCHANGE_H
#define OPENMX_CORE_PROJECTOR_EXCHANGE_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct OpenMXProjectorExchange OpenMXProjectorExchange;
enum { OPENMX_EXCHANGE_FLOAT = 1, OPENMX_EXCHANGE_DOUBLE = 2 };
enum { OPENMX_EXCHANGE_SEND = 0, OPENMX_EXCHANGE_RECEIVE = 1 };
typedef struct {
  int *sent, *received, *send_sizes, *receive_sizes; /* [peers], initially zero */
} OpenMXProjectorExchangeBuffers;

/* No MPI/GPU initialization. Views expire at destruction. Buffers grow only
   when needed; send and receive storage are independent. Complete all uses of
   one direction before growing it. Zero counts return an addressable dummy.
   Failure leaves existing output arguments and prior storage unchanged. */
OpenMXProjectorExchange *openmx_core_projector_exchange_create(int peers, int scalar,
    OpenMXProjectorExchangeBuffers *buffers, char *error, size_t capacity);
int openmx_core_projector_exchange_buffer(OpenMXProjectorExchange *exchange,
    int direction, int elements, void **buffer, char *error, size_t capacity);
void openmx_core_projector_exchange_destroy(OpenMXProjectorExchange *exchange);
#ifdef __cplusplus
}
#endif
#endif

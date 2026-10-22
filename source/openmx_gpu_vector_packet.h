#ifndef OPENMX_GPU_VECTOR_PACKET_H
#define OPENMX_GPU_VECTOR_PACKET_H
#include "openmx_gpu_vector.h"
#include <stdint.h>

/* Internal native wire layout. Encoders zero header padding; decoders also
   accept older packets with arbitrary padding. Payloads contain values. */
typedef struct { uint64_t n; int mode,columns,pairs; } OpenMXGpuVectorPacket;

#ifdef __cplusplus
extern "C" {
#endif
/* Successful pack and allocated remote results transfer malloc-compatible
   storage to the collective, including ordinary failed remote results. */
int openmx_gpu_vector_pack(void *context,size_t first,size_t last,void **packet,size_t *bytes);
int openmx_gpu_vector_remote(void *context,const void *packet,size_t bytes,void **result,size_t *result_bytes);
#ifdef __cplusplus
}
#endif
#endif

#ifndef OPENMX_GPU_TASK_PACKET_H
#define OPENMX_GPU_TASK_PACKET_H
#include "openmx_gpu_task.h"
#include <stdint.h>

/* Internal wire header, shared with the C transport tests. Payload sections
   use the aligned layout in openmx_gpu_task_layout.h; no process pointers. */
typedef struct {
    uint64_t first,count,parameter_bytes;
    int inputs,outputs,scratch,scope_kind;
    struct { uint64_t bytes,item_bytes,element_bytes; } view[OPENMX_GPU_TASK_VIEWS];
} OpenMXGpuTaskPacket;

#ifdef __cplusplus
extern "C" {
#endif
/* Successful pack and allocated remote results transfer malloc-compatible
   storage to the collective. A remote GPU failure may still return a buffer;
   the collective frees it without committing. No task/input ownership moves. */
int openmx_gpu_task_pack(void *context,size_t first,size_t last,void **packet,size_t *bytes);
int openmx_gpu_task_remote(void *context,const void *packet,size_t bytes,void **result,size_t *result_bytes);
#ifdef __cplusplus
}
#endif
#endif

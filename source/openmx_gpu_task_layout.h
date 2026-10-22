#ifndef OPENMX_GPU_TASK_LAYOUT_H
#define OPENMX_GPU_TASK_LAYOUT_H
/* Internal wire layout and final publication shared by the C controller and
   the C++ worker. Both compile the same arithmetic and copy operations. */
#include "openmx_gpu_task.h"
#include <stdint.h>
#include <string.h>
/* All payload sections start on a 16-byte boundary, including float and
   integer views. Saturation is a declined capacity request, never a wrap. */
static inline size_t task_add(size_t a,size_t b)
{
    if (b>SIZE_MAX-15 || a>SIZE_MAX-((b+15)&~(size_t)15)) return SIZE_MAX;
    return a+((b+15)&~(size_t)15);
}
static inline size_t task_view_bytes(const OpenMXGpuTaskView *v,size_t lo,size_t hi)
{
    size_t n=hi-lo;
    if (v->offsets) {
        if (!v->element_bytes || n>SIZE_MAX/sizeof(size_t)-1 || v->offsets[hi]<v->offsets[lo]) return SIZE_MAX;
        size_t elements=v->offsets[hi]-v->offsets[lo];
        if (elements>SIZE_MAX/v->element_bytes) return SIZE_MAX;
        return task_add(task_add(0,(n+1)*sizeof(size_t)),elements*v->element_bytes);
    }
    if (!v->item_bytes) return v->bytes;
    return n>SIZE_MAX/v->item_bytes ? SIZE_MAX : n*v->item_bytes;
}
static inline size_t task_bytes(const OpenMXGpuTask *t,size_t lo,size_t hi,int start,int end)
{
    size_t bytes=0;
    for (int i=start;i<end;i++) bytes=task_add(bytes,task_view_bytes(t->view+i,lo,hi));
    return bytes;
}
static inline void task_copy_view(void *target,const OpenMXGpuTaskView *v,size_t lo,size_t hi)
{
    if (v->offsets) {
        size_t *ptr=(size_t*)target,base=v->offsets[lo];
        for (size_t i=0;i<=hi-lo;i++) ptr[i]=v->offsets[lo+i]-base;
        size_t nb=(v->offsets[hi]-base)*v->element_bytes;
        if (nb) memcpy(openmx_gpu_task_ragged_data(target,hi-lo),
            (const unsigned char*)v->data+base*v->element_bytes,nb);
    } else {
        size_t nb=task_view_bytes(v,lo,hi);
        if (nb) memcpy(target,(const unsigned char*)v->data+lo*v->item_bytes,nb);
    }
}
static inline void task_commit(void *opaque,size_t lo,size_t hi,const void *result,size_t bytes)
{
    OpenMXGpuTask *t=(OpenMXGpuTask*)opaque;size_t offset=0;(void)bytes;
    for (int i=t->inputs;i<t->inputs+t->outputs;i++) {
        size_t nb=task_view_bytes(t->view+i,lo,hi);
        const unsigned char *src=(const unsigned char*)result+offset;
        if (t->view[i].item_bytes) {
            if (nb) memcpy((unsigned char*)t->view[i].data+lo*t->view[i].item_bytes,src,nb);
        }
        else for (size_t k=0;k<nb/sizeof(double);k++)
            ((double*)t->view[i].data)[k]+=((const double*)src)[k];
        offset=task_add(offset,nb);
    }
}

#ifdef __cplusplus
extern "C" {
#endif
int openmx_gpu_task_local(OpenMXGpuTask *task,const char *label);
#ifdef __cplusplus
}
#endif
#endif

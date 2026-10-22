#include "openmx_gpu_backend.h"
#include "openmx_gpu_device.h"
#include "openmx_gpu_runtime.h"
#include "openmx_gpu_task.h"
#include "openmx_gpu_task_layout.h"
#include "openmx_gpu_workers.h"
#include "openmx_gpu_collective.h"


#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>


typedef struct {
    uint64_t first,count,parameter_bytes;
    int inputs,outputs,scratch,scope_kind;
    struct { uint64_t bytes,item_bytes,element_bytes; } view[OPENMX_GPU_TASK_VIEWS];
} TaskPacket;

static int task_pack(void *opaque,size_t lo,size_t hi,void **packet,size_t *bytes)
{
    const OpenMXGpuTask *t=opaque;size_t n=hi-lo;
    size_t offset=task_add(0,sizeof(TaskPacket));
    offset=task_add(offset,t->parameter_bytes);
    *bytes=task_add(offset,task_bytes(t,lo,hi,0,t->inputs));
    if (*bytes==SIZE_MAX) return 0;
    TaskPacket *p=calloc(*bytes?*bytes:1,1);if (!p) return 0;
    *packet=p;p->first=t->first+lo;p->count=n;p->parameter_bytes=t->parameter_bytes;
    p->inputs=t->inputs;p->outputs=t->outputs;p->scratch=t->scratch;p->scope_kind=t->scope_kind;
    for (int i=0;i<t->inputs+t->outputs+t->scratch;i++) {p->view[i].bytes=t->view[i].bytes;p->view[i].item_bytes=t->view[i].item_bytes;p->view[i].element_bytes=t->view[i].element_bytes;}
    if (t->parameter_bytes) memcpy((unsigned char*)p+task_add(0,sizeof(*p)),t->parameters,t->parameter_bytes);
    for (int i=0;i<t->inputs;i++) {
        size_t nb=task_view_bytes(t->view+i,lo,hi);
        if (nb) task_copy_view((unsigned char*)p+offset,t->view+i,lo,hi);
        offset=task_add(offset,nb);
    }
    return 1;
}
static int task_remote(void *context,const void *packet,size_t bytes,void **result,size_t *result_bytes)
{
    const OpenMXGpuTask *original=context;
    if (bytes<sizeof(TaskPacket)) return 0;
    const TaskPacket *p=packet;
    if (p->inputs<0 || p->outputs<1 || p->scratch<0 || p->outputs>OPENMX_GPU_TASK_VIEWS ||
        p->inputs>OPENMX_GPU_TASK_VIEWS-p->outputs ||
        p->scratch>OPENMX_GPU_TASK_VIEWS-p->outputs-p->inputs || p->count>SIZE_MAX ||
        p->first>SIZE_MAX-p->count || p->parameter_bytes>SIZE_MAX) return 0;
    OpenMXGpuTask t={0};t.first=p->first;t.count=p->count;t.parameter_bytes=p->parameter_bytes;
    t.inputs=p->inputs;t.outputs=p->outputs;t.scratch=p->scratch;t.kernel=original->kernel;t.scope_kind=p->scope_kind;
    size_t offset=task_add(task_add(0,sizeof(*p)),t.parameter_bytes);
    if (offset>bytes) return 0;
    t.parameters=(const unsigned char*)p+task_add(0,sizeof(*p));
    for (int i=0;i<t.inputs+t.outputs+t.scratch;i++) {
        if (p->view[i].bytes>SIZE_MAX || p->view[i].item_bytes>SIZE_MAX || p->view[i].element_bytes>SIZE_MAX) return 0;
        t.view[i].bytes=p->view[i].bytes;t.view[i].item_bytes=p->view[i].item_bytes;
        if (p->view[i].element_bytes) {
            if (i>=t.inputs || t.count>SIZE_MAX/sizeof(size_t)-1) return 0;
            size_t index_bytes=task_add(0,(t.count+1)*sizeof(size_t));
            if (index_bytes>bytes-offset) return 0;
            t.view[i].offsets=(const size_t*)((const unsigned char*)packet+offset);
            t.view[i].element_bytes=p->view[i].element_bytes;
            if (t.view[i].offsets[0]) return 0;
            for (size_t k=0;k<t.count;k++) if (t.view[i].offsets[k+1]<t.view[i].offsets[k]) return 0;
        }
        if (i<t.inputs) {
            size_t next=task_add(offset,task_view_bytes(t.view+i,0,t.count));
            if (next>bytes) return 0;
            t.view[i].data=t.view[i].offsets?openmx_gpu_task_ragged_data((unsigned char*)packet+offset,t.count):(unsigned char*)packet+offset;
            offset=next;
        }
        else if (i<t.inputs+t.outputs && !t.view[i].item_bytes && t.view[i].bytes%sizeof(double)) return 0;
    }
    if (offset!=bytes) return 0;
    *result_bytes=task_bytes(&t,0,t.count,t.inputs,t.inputs+t.outputs);
    if (*result_bytes==SIZE_MAX) return 0;
    *result=calloc(*result_bytes?*result_bytes:1,1);if (!*result) return 0;
    offset=0;
    for (int i=t.inputs;i<t.inputs+t.outputs;i++) {
        t.view[i].data=(unsigned char*)*result+offset;
        offset=task_add(offset,task_view_bytes(t.view+i,0,t.count));
    }
    return openmx_gpu_task_local(&t,"remote-range");
}

static size_t task_cpu_bytes(const OpenMXGpuTask *t,size_t lo,size_t hi)
{
    size_t nb=task_bytes(t,lo,hi,t->inputs+t->outputs,t->inputs+t->outputs+t->scratch);
    for (int i=0;i<t->inputs+t->outputs;i++)
        if (t->view[i].offsets || (i>=t->inputs && !t->view[i].item_bytes))
            nb=task_add(nb,task_view_bytes(t->view+i,lo,hi));
    return nb;
}
static void task_cpu(OpenMXGpuTask *t)
{
    const int output_end=t->inputs+t->outputs,end=output_end+t->scratch;
    /* Keep scratch and CSR packing bounded on the CPU too. A final CPU
       fallback must not duplicate an entire large device workspace. */
    for (size_t first=0;first<t->count;) {
        size_t n=t->count-first,nb;unsigned char *arena=NULL;
        while (n>1 && task_cpu_bytes(t,first,first+n)>67108864) n=(n+1)/2;
        for (;;) {
            nb=task_cpu_bytes(t,first,first+n);
            if (nb!=SIZE_MAX) arena=malloc(nb?nb:1);
            if (arena || n<=1) break;
            n=(n+1)/2;
        }
        if (!arena) {fprintf(stderr,"OpenMX: cannot allocate a single CPU range workspace.\n");abort();}
        void *p[OPENMX_GPU_TASK_VIEWS];size_t offset=0;
        for (int i=0;i<output_end;i++) {
            p[i]=t->view[i].data;
            if (t->view[i].offsets) {
                p[i]=arena+offset;task_copy_view(p[i],t->view+i,first,first+n);
                offset=task_add(offset,task_view_bytes(t->view+i,first,first+n));
            } else if (t->view[i].item_bytes) p[i]=(unsigned char*)p[i]+first*t->view[i].item_bytes;
            else if (i>=t->inputs) {p[i]=arena+offset;offset=task_add(offset,t->view[i].bytes);}
        }
        for (int i=output_end;i<end;i++) {p[i]=arena+offset;offset=task_add(offset,task_view_bytes(t->view+i,first,first+n));}
        t->kernel(t->first+first,n,t->parameters,p,0);
        for (int i=t->inputs;i<output_end;i++) if (!t->view[i].item_bytes)
            for (size_t k=0;k<t->view[i].bytes/sizeof(double);k++) ((double*)t->view[i].data)[k]+=((double*)p[i])[k];
        free(arena);first+=n;
    }
}

int openmx_gpu_task_execute(MPI_Comm comm,OpenMXGpuTask *t,int enabled,const char *label)
{
    const int views=t->inputs+t->outputs;
    /* Reduction destinations are private to this operation; callers add
       existing accumulated values after successful/CPU completion. */
    for (int i=t->inputs;i<views;i++) if (!t->view[i].item_bytes)
        memset(t->view[i].data,0,t->view[i].bytes);
    int borrow=enabled && openmx_gpu_workers_capacity_enabled();
    int usable=enabled && (gpu_rank_device_usable() || borrow);
    int done=!t->count || (usable && openmx_gpu_task_local(t,label));
    if (comm!=MPI_COMM_NULL) {
        static const OpenMXGpuCollectiveOps ops={task_pack,task_remote,task_commit};
        done=openmx_gpu_collective_retry(comm,&ops,t,t->count,done?1:borrow?0:-1,borrow && usable,label);
    }
    if (!done) task_cpu(t);
    return done;
}

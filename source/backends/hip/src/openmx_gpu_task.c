#include "openmx_gpu_task.h"
#include "openmx_gpu_workers.h"
#include "openmx_gpu_collective.h"
#include "set_hip_default_device_from_local_rank.h"
#include "hip_runtime_compat.h"
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

typedef struct {
    OpenMXGpuWorkers pool;
    OpenMXGpuTask *task;
    unsigned char *device[OPENMX_GPU_WORKERS_MAX];
    unsigned char *result[OPENMX_GPU_WORKERS_MAX];
} TaskWork;

/* All payload sections start on a 16-byte boundary, including float and
   integer views. Saturation is a declined capacity request, never a wrap. */
static size_t task_add(size_t a,size_t b)
{
    if (b>SIZE_MAX-15 || a>SIZE_MAX-((b+15)&~(size_t)15)) return SIZE_MAX;
    return a+((b+15)&~(size_t)15);
}
static size_t task_view_bytes(const OpenMXGpuTaskView *v,size_t lo,size_t hi)
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
static size_t task_bytes(const OpenMXGpuTask *t,size_t lo,size_t hi,int start,int end)
{
    size_t bytes=0;
    for (int i=start;i<end;i++) bytes=task_add(bytes,task_view_bytes(t->view+i,lo,hi));
    return bytes;
}
static void task_copy_view(void *target,const OpenMXGpuTaskView *v,size_t lo,size_t hi)
{
    if (v->offsets) {
        size_t *ptr=target,base=v->offsets[lo];
        for (size_t i=0;i<=hi-lo;i++) ptr[i]=v->offsets[lo+i]-base;
        size_t nb=(v->offsets[hi]-base)*v->element_bytes;
        if (nb) memcpy(openmx_gpu_task_ragged_data(target,hi-lo),
            (const unsigned char*)v->data+base*v->element_bytes,nb);
    } else {
        size_t nb=task_view_bytes(v,lo,hi);
        if (nb) memcpy(target,(const unsigned char*)v->data+lo*v->item_bytes,nb);
    }
}
static size_t task_cost(void *opaque,int worker,int count)
{
    const OpenMXGpuTask *t=opaque;
    size_t lo=openmx_gpu_worker_cut(t->count,worker,count),hi=openmx_gpu_worker_cut(t->count,worker+1,count);
    return task_bytes(t,lo,hi,0,t->inputs+t->outputs+t->scratch);
}
static int task_worker(void *opaque,int worker,int phase)
{
    TaskWork *w=opaque;const OpenMXGpuTask *t=w->task;
    size_t lo=openmx_gpu_worker_cut(t->count,worker,w->pool.count);
    size_t hi=openmx_gpu_worker_cut(t->count,worker+1,w->pool.count),n=hi-lo;
    const int outputs=t->inputs+t->outputs,views=outputs+t->scratch;
    if (phase==OPENMX_GPU_WORKER_PREPARE) {
        size_t bytes=task_bytes(t,lo,hi,0,views),results=task_bytes(t,lo,hi,t->inputs,outputs);
        if (bytes==SIZE_MAX || results==SIZE_MAX) return 0;
        w->result[worker]=malloc(results?results:1);
        if (!w->result[worker] || hipMalloc((void**)&w->device[worker],bytes?bytes:1)!=hipSuccess) return 0;
        size_t offset=0;
        for (int i=0;i<t->inputs;i++) {
            size_t nb=task_view_bytes(t->view+i,lo,hi);
            const OpenMXGpuTaskView *v=t->view+i;
            if (v->offsets) {
                size_t nindex=(n+1)*sizeof(size_t),base=v->offsets[lo];
                size_t *index=malloc(nindex);if (!index) return 0;
                for (size_t k=0;k<=n;k++) index[k]=v->offsets[lo+k]-base;
                int ok=hipMemcpy(w->device[worker]+offset,index,nindex,hipMemcpyHostToDevice)==hipSuccess;
                free(index);if (!ok) return 0;
                size_t data_bytes=(v->offsets[hi]-base)*v->element_bytes;
                if (data_bytes && hipMemcpy(w->device[worker]+offset+task_add(0,nindex),
                    (const unsigned char*)v->data+base*v->element_bytes,data_bytes,hipMemcpyHostToDevice)!=hipSuccess) return 0;
            } else if (nb && hipMemcpy(w->device[worker]+offset,
                (const unsigned char*)v->data+lo*v->item_bytes,nb,hipMemcpyHostToDevice)!=hipSuccess) return 0;
            offset=task_add(offset,nb);
        }
    }
    else if (phase==OPENMX_GPU_WORKER_EXECUTE) {
        void *p[OPENMX_GPU_TASK_VIEWS];size_t offset=0;
        for (int i=0;i<views;i++) {p[i]=w->device[worker]+offset;offset=task_add(offset,task_view_bytes(t->view+i,lo,hi));}
        t->kernel(t->first+lo,n,t->parameters,p,1);
    }
    else if (phase==OPENMX_GPU_WORKER_COMMIT) {
        size_t input_bytes=task_bytes(t,lo,hi,0,t->inputs),nb=task_bytes(t,lo,hi,t->inputs,outputs);
        return !nb || hipMemcpy(w->result[worker],w->device[worker]+input_bytes,nb,hipMemcpyDeviceToHost)==hipSuccess;
    }
    else if (phase==OPENMX_GPU_WORKER_RELEASE) {
        if (w->device[worker] && hipFree(w->device[worker])!=hipSuccess) return 0;
    }
    return 1;
}
static void task_commit(void *opaque,size_t lo,size_t hi,const void *result,size_t bytes)
{
    OpenMXGpuTask *t=opaque;size_t offset=0;(void)bytes;
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
static int task_local(OpenMXGpuTask *t,const char *label)
{
    if (!t->count) return 1;
    TaskWork w={0};w.task=t;
    OpenMXGpuWorkerRetry retry={task_cost,t,{0}};
    while (openmx_gpu_workers_capacity(&w.pool,t->count,t->scope_kind==2?2:1,openmx_gpu_worker_retry_cost,&retry)) {
        int ok=openmx_gpu_workers_run(&w.pool,task_worker,&w,label);
        if (ok) for (int i=0;i<w.pool.count;i++) {
            size_t lo=openmx_gpu_worker_cut(t->count,i,w.pool.count),hi=openmx_gpu_worker_cut(t->count,i+1,w.pool.count);
            task_commit(t,lo,hi,w.result[i],task_bytes(t,lo,hi,t->inputs,t->inputs+t->outputs));
        }
        for (int i=0;i<w.pool.count;i++) {free(w.result[i]);w.result[i]=NULL;w.device[i]=NULL;}
        if (ok) return 1;
        retry.rejected[w.pool.count]=1;
    }
    return 0;
}

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
    return task_local(&t,"remote-range");
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
    int done=!t->count || (usable && task_local(t,label));
    if (comm!=MPI_COMM_NULL) {
        static const OpenMXGpuCollectiveOps ops={task_pack,task_remote,task_commit};
        done=openmx_gpu_collective_retry(comm,&ops,t,t->count,done?1:borrow?0:-1,borrow && usable,label);
    }
    if (!done) task_cpu(t);
    return done;
}

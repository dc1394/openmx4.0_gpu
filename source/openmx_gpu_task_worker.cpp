#include "openmx_gpu_task_layout.h"
#include "openmx_gpu_workers.h"
#include "openmx_gpu_memory.hpp"
#include "openmx_host_buffer.hpp"
#include <array>
#include <cstdlib>

namespace {
struct TaskWork {
    OpenMXGpuWorkers pool{};
    OpenMXGpuTask *task=nullptr;
    std::array<openmx::gpu::DeviceBuffer<unsigned char>,OPENMX_GPU_WORKERS_MAX> device;
    std::array<openmx::host::Buffer<unsigned char>,OPENMX_GPU_WORKERS_MAX> result;
};
static size_t task_cost(void *opaque,int worker,int count)
{
    const auto *t=static_cast<const OpenMXGpuTask*>(opaque);
    size_t lo=openmx_gpu_worker_cut(t->count,worker,count),hi=openmx_gpu_worker_cut(t->count,worker+1,count);
    return task_bytes(t,lo,hi,0,t->inputs+t->outputs+t->scratch);
}
static int task_worker(void *opaque,int worker,int phase)
{
    auto *w=static_cast<TaskWork*>(opaque);const OpenMXGpuTask *t=w->task;
    size_t lo=openmx_gpu_worker_cut(t->count,worker,w->pool.count);
    size_t hi=openmx_gpu_worker_cut(t->count,worker+1,w->pool.count),n=hi-lo;
    const int outputs=t->inputs+t->outputs,views=outputs+t->scratch;
    if (phase==OPENMX_GPU_WORKER_PREPARE) {
        size_t bytes=task_bytes(t,lo,hi,0,views),results=task_bytes(t,lo,hi,t->inputs,outputs);
        if (bytes==SIZE_MAX || results==SIZE_MAX) return 0;
        w->result[worker].reset(static_cast<unsigned char*>(std::malloc(results?results:1)));
        if (!w->result[worker] || !w->device[worker].allocate(bytes?bytes:1)) return 0;
        auto *device=w->device[worker].get();
        size_t offset=0;
        for (int i=0;i<t->inputs;i++) {
            size_t nb=task_view_bytes(t->view+i,lo,hi);
            const OpenMXGpuTaskView *v=t->view+i;
            if (v->offsets) {
                size_t nindex=(n+1)*sizeof(size_t),base=v->offsets[lo];
                openmx::host::Buffer<size_t> indices(static_cast<size_t*>(std::malloc(nindex)));
                auto *index=indices.get();if (!index) return 0;
                for (size_t k=0;k<=n;k++) index[k]=v->offsets[lo+k]-base;
                int ok=OpenMXGpuMemcpy(device+offset,index,nindex,OPENMX_GPU_COPY_HOST_TO_DEVICE)==OPENMX_GPU_SUCCESS;
                indices.reset();if (!ok) return 0;
                size_t data_bytes=(v->offsets[hi]-base)*v->element_bytes;
                if (data_bytes && OpenMXGpuMemcpy(device+offset+task_add(0,nindex),
                    (const unsigned char*)v->data+base*v->element_bytes,data_bytes,OPENMX_GPU_COPY_HOST_TO_DEVICE)!=OPENMX_GPU_SUCCESS) return 0;
            } else if (nb && OpenMXGpuMemcpy(device+offset,
                (const unsigned char*)v->data+lo*v->item_bytes,nb,OPENMX_GPU_COPY_HOST_TO_DEVICE)!=OPENMX_GPU_SUCCESS) return 0;
            offset=task_add(offset,nb);
        }
    }
    else if (phase==OPENMX_GPU_WORKER_EXECUTE) {
        void *p[OPENMX_GPU_TASK_VIEWS];size_t offset=0;
        for (int i=0;i<views;i++) {p[i]=w->device[worker].get()+offset;offset=task_add(offset,task_view_bytes(t->view+i,lo,hi));}
        t->kernel(t->first+lo,n,t->parameters,p,1);
    }
    else if (phase==OPENMX_GPU_WORKER_COMMIT) {
        size_t input_bytes=task_bytes(t,lo,hi,0,t->inputs),nb=task_bytes(t,lo,hi,t->inputs,outputs);
        return !nb || OpenMXGpuMemcpy(w->result[worker].get(),w->device[worker].get()+input_bytes,nb,OPENMX_GPU_COPY_DEVICE_TO_HOST)==OPENMX_GPU_SUCCESS;
    }
    else if (phase==OPENMX_GPU_WORKER_RELEASE) {
        if (w->device[worker].try_reset()!=OPENMX_GPU_SUCCESS) return 0;
    }
    return 1;
}
} // namespace

extern "C" int openmx_gpu_task_local(OpenMXGpuTask *t,const char *label)
{
    if (!t->count) return 1;
    TaskWork w;w.task=t;
    OpenMXGpuWorkerRetry retry={task_cost,t,{0}};
    while (openmx_gpu_workers_capacity(&w.pool,t->count,t->scope_kind==2?2:1,openmx_gpu_worker_retry_cost,&retry)) {
        int ok=openmx_gpu_workers_run_prepared(&w.pool,task_worker,&w,label);
        if (ok) for (int i=0;i<w.pool.count;i++) {
            size_t lo=openmx_gpu_worker_cut(t->count,i,w.pool.count),hi=openmx_gpu_worker_cut(t->count,i+1,w.pool.count);
            task_commit(t,lo,hi,w.result[i].get(),task_bytes(t,lo,hi,t->inputs,t->inputs+t->outputs));
        }
        for (int i=0;i<w.pool.count;i++) {w.result[i].reset();w.device[i].reset();}
        if (ok) return 1;
        retry.rejected[w.pool.count]=1;
    }
    return 0;
}

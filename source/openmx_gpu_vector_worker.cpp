#include "openmx_gpu_backend.h"
#include "openmx_gpu_vector_worker.h"
#include "openmx_gpu_workers.h"
#include "openmx_gpu_memory.hpp"
#include "openmx_host_buffer.hpp"
#if OPENMX_GPU_HIP
#include <omp.h>
#else
#include <openacc.h>
#endif
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace {
struct VectorWork {
    OpenMXGpuWorkers workers{};
    OpenMXGpuVector job{};
    std::array<openmx::gpu::DeviceBuffer<double>,OPENMX_GPU_WORKERS_MAX> arena;
    std::array<openmx::host::Buffer<double>,OPENMX_GPU_WORKERS_MAX> partial;
};
static size_t vector_cost(void *opaque,int worker,int count)
{
    auto *j=static_cast<OpenMXGpuVector*>(opaque);
    size_t n=openmx_gpu_worker_cut(j->n,worker+1,count)-openmx_gpu_worker_cut(j->n,worker,count);
    size_t extra=j->mode ? (size_t)j->columns : (size_t)j->pairs*2;
    const size_t columns=(size_t)j->columns+2;
    if (j->columns<1 || extra>SIZE_MAX/sizeof(double) ||
        n>(SIZE_MAX/sizeof(double)-extra)/columns) return SIZE_MAX;
    return (columns*n+extra)*sizeof(double);
}

static int vector_worker(void *opaque,int worker,int phase)
{
    auto *w=static_cast<VectorWork*>(opaque);const OpenMXGpuVector *j=&w->job;
    const size_t first=openmx_gpu_worker_cut(j->n,worker,w->workers.count);
    const size_t last=openmx_gpu_worker_cut(j->n,worker+1,w->workers.count),n=last-first;
    const size_t out_count=j->mode ? n : (size_t)j->pairs;
    double *data=w->arena[worker].get(),*weights=data?data+(size_t)j->columns*n:NULL;
    double *out=weights?weights+(j->mode ? (size_t)j->columns : n):NULL;
    int *left=(!j->mode && weights)?(int*)(weights+n+j->pairs):NULL,*right=left?left+j->pairs:NULL;
    if (phase==OPENMX_GPU_WORKER_PREPARE) {
        const size_t bytes=vector_cost((void*)j,worker,w->workers.count);
        if (bytes==SIZE_MAX) return 0;
        w->partial[worker].reset(static_cast<double*>(std::malloc((out_count?out_count:1)*sizeof(double))));
        if (!w->partial[worker] || !w->arena[worker].allocate(bytes/sizeof(double))) return 0;
        data=w->arena[worker].get();weights=data+(size_t)j->columns*n;
        for (int col=0;col<j->columns;col++)
            if (OpenMXGpuMemcpy(data+(size_t)col*n,j->data+(size_t)col*j->ld+first,n*sizeof(double),OPENMX_GPU_COPY_HOST_TO_DEVICE)!=OPENMX_GPU_SUCCESS) return 0;
        if (OpenMXGpuMemcpy(weights,j->mode?j->coeff:j->weight+first,
                       (j->mode?(size_t)j->columns:n)*sizeof(double),OPENMX_GPU_COPY_HOST_TO_DEVICE)!=OPENMX_GPU_SUCCESS) return 0;
        if (!j->mode) {
            /* The indices share the arena; vector_cost includes their bytes. */
            left=(int*)(weights+n+j->pairs);right=left+j->pairs;
            if (OpenMXGpuMemcpy(left,j->left,(size_t)j->pairs*sizeof(int),OPENMX_GPU_COPY_HOST_TO_DEVICE)!=OPENMX_GPU_SUCCESS ||
                OpenMXGpuMemcpy(right,j->right,(size_t)j->pairs*sizeof(int),OPENMX_GPU_COPY_HOST_TO_DEVICE)!=OPENMX_GPU_SUCCESS) return 0;
        }
    }
    else if (phase==OPENMX_GPU_WORKER_EXECUTE) {
        if (j->mode) {
            const int columns=j->columns;
#if OPENMX_GPU_HIP
#pragma omp target teams distribute parallel for is_device_ptr(data,weights,out)
#else
#pragma acc parallel loop deviceptr(data,weights,out)
#endif
            for (size_t i=0;i<n;i++) {
                double sum=0;
#if !OPENMX_GPU_HIP
#pragma acc loop seq
#endif
                for (int col=0;col<columns;col++) sum+=weights[col]*data[(size_t)col*n+i];
                out[i]=sum;
            }
        }
        else {
            const int pairs=j->pairs;
#if OPENMX_GPU_HIP
            /* One team owns a pair; its threads cooperate on the long
               vector. Parallelizing only pairs leaves a handful of GPU
               lanes doing the entire FP64 contraction serially. */
#pragma omp target teams distribute thread_limit(256) is_device_ptr(data,weights,out,left,right)
#else
#pragma acc parallel loop gang deviceptr(data,weights,out,left,right)
#endif
            for (int p=0;p<pairs;p++) {
                double sum=0;
                const size_t a=(size_t)left[p]*n,b=(size_t)right[p]*n;
#if OPENMX_GPU_HIP
#pragma omp parallel for reduction(+:sum)
#else
#pragma acc loop vector reduction(+:sum)
#endif
                for (size_t i=0;i<n;i++) sum+=weights[i]*data[a+i]*data[b+i];
                out[p]=sum;
            }
        }
    }
    else if (phase==OPENMX_GPU_WORKER_COMMIT) {
        return OpenMXGpuMemcpy(w->partial[worker].get(),out,out_count*sizeof(double),OPENMX_GPU_COPY_DEVICE_TO_HOST)==OPENMX_GPU_SUCCESS;
    }
    else if (phase==OPENMX_GPU_WORKER_RELEASE) {
        if (w->arena[worker].try_reset()!=OPENMX_GPU_SUCCESS) return 0;
    }
    return 1;
}

} // namespace

extern "C" int openmx_gpu_vector_local(OpenMXGpuVector *job)
{
    if (!job->n) return 1;
    VectorWork w;w.job=*job;
    OpenMXGpuWorkerRetry retry={vector_cost,job,{0}};
    while (openmx_gpu_workers_capacity(&w.workers,job->n,1,openmx_gpu_worker_retry_cost,&retry)) {
        int ok=openmx_gpu_workers_run_prepared(&w.workers,vector_worker,&w,"vector-contraction");
        if (ok) {
            if (!job->mode) memset(job->output,0,(size_t)job->pairs*sizeof(double));
            for (int worker=0;worker<w.workers.count;worker++) {
                size_t lo=openmx_gpu_worker_cut(job->n,worker,w.workers.count);
                size_t hi=openmx_gpu_worker_cut(job->n,worker+1,w.workers.count);
                if (job->mode) memcpy(job->output+lo,w.partial[worker].get(),(hi-lo)*sizeof(double));
                else for (int p=0;p<job->pairs;p++) job->output[p]+=w.partial[worker].get()[p];
            }
        }
        for (int i=0;i<w.workers.count;i++) {w.partial[i].reset();w.arena[i].reset();}
        if (ok) return 1;
        retry.rejected[w.workers.count]=1;
    }
    return 0;
}


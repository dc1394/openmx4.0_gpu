#include "openmx_gpu_vector.h"
#include "openmx_gpu_workers.h"
#include "openmx_gpu_collective.h"
#include "set_hip_default_device_from_local_rank.h"
#include "hip_runtime_compat.h"
#include <omp.h>
#include <stdint.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    OpenMXGpuWorkers workers;
    OpenMXGpuVector job;
    double *arena[OPENMX_GPU_WORKERS_MAX],*partial[OPENMX_GPU_WORKERS_MAX];
    int *indices[OPENMX_GPU_WORKERS_MAX];
} VectorWork;

static size_t vector_cost(void *opaque,int worker,int count)
{
    OpenMXGpuVector *j=opaque;
    size_t n=openmx_gpu_worker_cut(j->n,worker+1,count)-openmx_gpu_worker_cut(j->n,worker,count);
    size_t extra=j->mode ? (size_t)j->columns : (size_t)j->pairs*2;
    const size_t columns=(size_t)j->columns+2;
    if (j->columns<1 || extra>SIZE_MAX/sizeof(double) ||
        n>(SIZE_MAX/sizeof(double)-extra)/columns) return SIZE_MAX;
    return (columns*n+extra)*sizeof(double);
}

static int vector_worker(void *opaque,int worker,int phase)
{
    VectorWork *w=opaque;const OpenMXGpuVector *j=&w->job;
    const size_t first=openmx_gpu_worker_cut(j->n,worker,w->workers.count);
    const size_t last=openmx_gpu_worker_cut(j->n,worker+1,w->workers.count),n=last-first;
    const size_t out_count=j->mode ? n : (size_t)j->pairs;
    double *data=w->arena[worker],*weights=data?data+(size_t)j->columns*n:NULL;
    double *out=weights?weights+(j->mode ? (size_t)j->columns : n):NULL;
    int *left=w->indices[worker],*right=left?left+j->pairs:NULL;
    if (phase==OPENMX_GPU_WORKER_PREPARE) {
        w->partial[worker]=malloc((out_count?out_count:1)*sizeof(double));
        if (!w->partial[worker] || hipMalloc((void**)&w->arena[worker],vector_cost((void*)j,worker,w->workers.count))!=hipSuccess) return 0;
        data=w->arena[worker];weights=data+(size_t)j->columns*n;
        for (int col=0;col<j->columns;col++)
            if (hipMemcpy(data+(size_t)col*n,j->data+(size_t)col*j->ld+first,n*sizeof(double),hipMemcpyHostToDevice)!=hipSuccess) return 0;
        if (hipMemcpy(weights,j->mode?j->coeff:j->weight+first,
                       (j->mode?(size_t)j->columns:n)*sizeof(double),hipMemcpyHostToDevice)!=hipSuccess) return 0;
        if (!j->mode) {
            /* The indices share the arena; vector_cost includes their bytes. */
            w->indices[worker]=(int*)(weights+n+j->pairs);
            if (hipMemcpy(w->indices[worker],j->left,(size_t)j->pairs*sizeof(int),hipMemcpyHostToDevice)!=hipSuccess ||
                hipMemcpy(w->indices[worker]+j->pairs,j->right,(size_t)j->pairs*sizeof(int),hipMemcpyHostToDevice)!=hipSuccess) return 0;
        }
    }
    else if (phase==OPENMX_GPU_WORKER_EXECUTE) {
        if (j->mode) {
            const int columns=j->columns;
#pragma omp target teams distribute parallel for is_device_ptr(data,weights,out)
            for (size_t i=0;i<n;i++) {
                double sum=0;

                for (int col=0;col<columns;col++) sum+=weights[col]*data[(size_t)col*n+i];
                out[i]=sum;
            }
        }
        else {
            const int pairs=j->pairs;
#pragma omp target teams distribute parallel for is_device_ptr(data,weights,out,left,right)
            for (int p=0;p<pairs;p++) {
                double sum=0;
                const size_t a=(size_t)left[p]*n,b=(size_t)right[p]*n;

                for (size_t i=0;i<n;i++) sum+=weights[i]*data[a+i]*data[b+i];
                out[p]=sum;
            }
        }
    }
    else if (phase==OPENMX_GPU_WORKER_COMMIT) {
        return hipMemcpy(w->partial[worker],out,out_count*sizeof(double),hipMemcpyDeviceToHost)==hipSuccess;
    }
    else if (phase==OPENMX_GPU_WORKER_RELEASE) {
        if (w->arena[worker] && hipFree(w->arena[worker])!=hipSuccess) return 0;
    }
    return 1;
}

static int vector_local(OpenMXGpuVector *job)
{
    if (!job->n) return 1;
    VectorWork w={0};w.job=*job;
    OpenMXGpuWorkerRetry retry={vector_cost,job,{0}};
    while (openmx_gpu_workers_capacity(&w.workers,job->n,1,openmx_gpu_worker_retry_cost,&retry)) {
        int ok=openmx_gpu_workers_run(&w.workers,vector_worker,&w,"vector-contraction");
        if (ok) {
            if (!job->mode) memset(job->output,0,(size_t)job->pairs*sizeof(double));
            for (int worker=0;worker<w.workers.count;worker++) {
                size_t lo=openmx_gpu_worker_cut(job->n,worker,w.workers.count);
                size_t hi=openmx_gpu_worker_cut(job->n,worker+1,w.workers.count);
                if (job->mode) memcpy(job->output+lo,w.partial[worker],(hi-lo)*sizeof(double));
                else for (int p=0;p<job->pairs;p++) job->output[p]+=w.partial[worker][p];
            }
        }
        for (int i=0;i<w.workers.count;i++) {free(w.partial[i]);w.partial[i]=NULL;w.arena[i]=NULL;w.indices[i]=NULL;}
        if (ok) return 1;
        retry.rejected[w.workers.count]=1;
    }
    return 0;
}

typedef struct { uint64_t n; int mode,columns,pairs; } VectorPacket;

static int vector_packet_size(size_t n,int mode,int columns,int pairs,size_t *bytes)
{
    if ((mode!=0 && mode!=1) || columns<1 || pairs<0) return 0;
    size_t indices=mode?0:(size_t)pairs*2*sizeof(int);
    size_t weights=mode?(size_t)columns:n;
    if (indices>SIZE_MAX-sizeof(VectorPacket) ||
        weights>(SIZE_MAX-sizeof(VectorPacket)-indices)/sizeof(double)) return 0;
    size_t available=(SIZE_MAX-sizeof(VectorPacket)-indices)/sizeof(double)-weights;
    if (n>available/(size_t)columns) return 0;
    *bytes=sizeof(VectorPacket)+((size_t)columns*n+weights)*sizeof(double)+indices;
    return 1;
}

static int vector_pack(void *opaque,size_t first,size_t last,void **packet,size_t *bytes)
{
    OpenMXGpuVector *j=opaque;const size_t n=last-first;
    if (last<first || last>j->n || !vector_packet_size(n,j->mode,j->columns,j->pairs,bytes)) return 0;
    VectorPacket *p=malloc(*bytes);if (!p) return 0;
    *packet=p;p->n=n;p->mode=j->mode;p->columns=j->columns;p->pairs=j->pairs;
    double *data=(double*)(p+1);
    for (int col=0;col<j->columns;col++) memcpy(data+(size_t)col*n,j->data+(size_t)col*j->ld+first,n*sizeof(double));
    data+=(size_t)j->columns*n;
    memcpy(data,j->mode?j->coeff:j->weight+first,(j->mode?(size_t)j->columns:n)*sizeof(double));
    if (!j->mode) {
        int *list=(int*)(data+n);
        memcpy(list,j->left,(size_t)j->pairs*sizeof(int));
        memcpy(list+j->pairs,j->right,(size_t)j->pairs*sizeof(int));
    }
    return 1;
}

static int vector_remote(void *context,const void *packet,size_t bytes,void **result,size_t *result_bytes)
{
    (void)context;
    if (bytes<sizeof(VectorPacket)) return 0;
    const VectorPacket *p=packet;
    size_t expected=0;
    if (p->n>SIZE_MAX || !vector_packet_size((size_t)p->n,p->mode,p->columns,p->pairs,&expected) || bytes!=expected) return 0;
    OpenMXGpuVector j={0};j.n=j.ld=(size_t)p->n;j.mode=p->mode;j.columns=p->columns;j.pairs=p->pairs;
    j.data=(const double*)(p+1);j.coeff=j.weight=j.data+(size_t)j.columns*j.n;
    if (!j.mode) {
        j.left=(const int*)(j.weight+j.n);j.right=j.left+j.pairs;
        for (int i=0;i<j.pairs;i++)
            if (j.left[i]<0 || j.left[i]>=j.columns || j.right[i]<0 || j.right[i]>=j.columns) return 0;
    }
    *result_bytes=(j.mode?j.n:(size_t)j.pairs)*sizeof(double);
    *result=calloc(*result_bytes?*result_bytes:1,1);if (!*result) return 0;
    j.output=*result;
    return vector_local(&j);
}

static void vector_commit(void *opaque,size_t first,size_t last,const void *result,size_t bytes)
{
    OpenMXGpuVector *j=opaque;(void)bytes;
    if (j->mode) memcpy(j->output+first,result,(last-first)*sizeof(double));
    else for (int p=0;p<j->pairs;p++) j->output[p]+=((const double*)result)[p];
}

int openmx_gpu_vector_execute(MPI_Comm comm,OpenMXGpuVector *j,int enabled,const char *label)
{
    const size_t outputs=j->mode?j->n:(size_t)j->pairs;
    memset(j->output,0,outputs*sizeof(double));
    int borrow=enabled && openmx_gpu_workers_capacity_enabled();
    int usable=enabled && (gpu_rank_device_usable() || borrow);
    int done=!j->n || (usable && vector_local(j));
    static const OpenMXGpuCollectiveOps ops={vector_pack,vector_remote,vector_commit};
    done=openmx_gpu_collective_retry(comm,&ops,j,j->n,done?1:borrow?0:-1,borrow && usable,label);
    if (!done) {
        if (j->mode) {
#pragma omp parallel for
            for (size_t i=0;i<j->n;i++) {
                double sum=0;
                for (int col=0;col<j->columns;col++) sum+=j->coeff[col]*j->data[(size_t)col*j->ld+i];
                j->output[i]=sum;
            }
        }
        else {
#pragma omp parallel for
            for (int p=0;p<j->pairs;p++) {
                double sum=0;
                const double *a=j->data+(size_t)j->left[p]*j->ld,*b=j->data+(size_t)j->right[p]*j->ld;
                for (size_t i=0;i<j->n;i++) sum+=j->weight[i]*a[i]*b[i];
                j->output[p]=sum;
            }
        }
    }
    return done;
}

/* Noncollective, bounded per-device workers for existing GPU operations. */
#include "openmx_gpu_workers.h"
#include "openmx_mgpu_runtime.h"
#include "set_cuda_default_device_from_local_rank.h"
#include <openacc.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern void acc_clear_freelists(void);

static int worker_rank=0, worker_ranks=1, mapped_count=0;
static int device_ready[OPENMX_GPU_WORKERS_MAX];
static _Thread_local int worker_device=-1;

int openmx_gpu_worker_device(void) { return worker_device; }

void openmx_gpu_workers_configure(int local_rank, int local_size, int mapped_devices)
{
    worker_rank=local_rank;
    worker_ranks=local_size>0 ? local_size : 1;
    mapped_count=mapped_devices;
}

static void worker_pool(OpenMXGpuWorkers *pool, size_t items, int kind)
{
    int owner=-1, permitted[OPENMX_GPU_WORKERS_MAX], count=0;
    memset(pool, 0, sizeof(*pool));
    pool->items=items;
    if (!items) return;
    if (cudaGetDevice(&owner)!=cudaSuccess || owner<0) return;
    if (worker_device>=0) { pool->devices[0]=owner; pool->count=1; return; }
    const char *control=getenv("OPENMX_GPU_WORKERS");
    if (!control || strcmp(control,"0"))
        count=kind==2 ? openmx_mgpu_scope_devices(permitted, OPENMX_GPU_WORKERS_MAX) :
                       openmx_mgpu_pool_devices(permitted, OPENMX_GPU_WORKERS_MAX);
    if (count<2) { pool->devices[0]=owner; pool->count=1; return; }
    if (kind) {
        for (int i=0;i<count;i++) pool->devices[pool->count++]=permitted[i];
    } else {
        pool->devices[pool->count++]=owner;
        for (int i=0;i<count;i++) {
            const int device=permitted[i];
            if (device==owner) continue;
            int assigned=0;
            for (int rank=0;rank<worker_ranks;rank++) {
                if (rank!=worker_rank && openmx_gpu_map_rank_to_device(rank,worker_ranks,mapped_count)==device) {
                    assigned=1; break;
                }
            }
            if (!assigned && (device%worker_ranks==worker_rank ||
                openmx_gpu_map_rank_to_device(worker_rank,worker_ranks,mapped_count)==device))
                pool->devices[pool->count++]=device;
        }
    }
    if (items<(size_t)pool->count) pool->count=(int)items;
}

void openmx_gpu_workers_local(OpenMXGpuWorkers *p,size_t n) { worker_pool(p,n,0); }
void openmx_gpu_workers_owner(OpenMXGpuWorkers *p,size_t n) { worker_pool(p,n,1); }
void openmx_gpu_workers_scope(OpenMXGpuWorkers *p,size_t n) { worker_pool(p,n,2); }

typedef struct {
    int cuda_device, acc_device, override, scope;
    acc_device_t acc_type;
} WorkerBinding;

static int bind_worker(int device,WorkerBinding *saved)
{
    saved->cuda_device=-1;
    (void)cudaGetDevice(&saved->cuda_device);
    saved->acc_type=acc_get_device_type();
    saved->acc_device=acc_get_device_num(saved->acc_type);
    saved->override=worker_device;
    saved->scope=openmx_mgpu_scope_enter(0);
    worker_device=device;
    if (cudaSetDevice(device)!=cudaSuccess) return 0;
    acc_set_device_num(device,acc_device_nvidia);
    return 1;
}

static void restore_worker(const WorkerBinding *saved)
{
    if (saved->acc_device>=0) acc_set_device_num(saved->acc_device,saved->acc_type);
    if (saved->cuda_device>=0) (void)cudaSetDevice(saved->cuda_device);
    worker_device=saved->override;
    openmx_mgpu_scope_leave(saved->scope);
}

/* Called under the process's shared GPU admission lock. In particular,
   device-module initialization and memory-query caches are warmed before
   callbacks run concurrently. No device reset can invalidate other caches. */
static int prepare_device(int device)
{
    if (device<0 || device>=OPENMX_GPU_WORKERS_MAX) return 0;
    if (device_ready[device]) return 1;
    /* Check CUDA capacity before OpenACC loads its module on a new device.
       acc_set_device_num may otherwise abort in a recoverable low-memory
       situation before the callback has a chance to decline. */
    int original=-1;
    size_t available=0,capacity=0;
    if (cudaGetDevice(&original)!=cudaSuccess) return 0;
    int enough=cudaSetDevice(device)==cudaSuccess && cudaFree(0)==cudaSuccess &&
        OpenMX_GpuMemGetInfo(&available,&capacity)==cudaSuccess &&
        available>=(size_t)256*1024*1024;
    if (cudaSetDevice(original)!=cudaSuccess) return 0;
    if (!enough) { (void)cudaGetLastError();return 0; }
    WorkerBinding saved;
    int ok=bind_worker(device,&saved);
    if (ok) {
        size_t free_bytes=0,total_bytes=0;
        ok=cudaFree(0)==cudaSuccess &&
           OpenMX_GpuMemGetInfo(&free_bytes,&total_bytes)==cudaSuccess &&
           free_bytes>=(size_t)256*1024*1024;
        if (ok) {
            void *touch=acc_malloc((size_t)1024*1024);
            ok=touch!=NULL;
            if (touch) acc_free(touch);
            if (ok) OpenMX_GpuPresizeKernelStack();
        }
    }
    if (!ok) (void)cudaGetLastError();
    restore_worker(&saved);
    if (ok) device_ready[device]=1;
    return ok;
}

int openmx_gpu_workers_run(const OpenMXGpuWorkers *pool,
    int (*callback)(void*,int,int),void *context,const char *label)
{
    int ok=1, started[OPENMX_GPU_WORKERS_MAX]={0};
    if (!pool || !callback || pool->count<0 || pool->count>OPENMX_GPU_WORKERS_MAX) return 0;
    if (!pool->count) return pool->items==0;
    /* Nested worker launch would deadlock the shared admission mutex. */
    if (worker_device>=0) return 0;
    if (cudaDeviceSynchronize()!=cudaSuccess) return 0;
    void *admission=openmx_mgpu_admit_devices(pool->devices,pool->count);
    if (!admission) return 0;
    for (int i=0;i<pool->count;i++)
        if (!prepare_device(pool->devices[i])) { ok=0; break; }
    for (int phase=OPENMX_GPU_WORKER_PREPARE;phase<=OPENMX_GPU_WORKER_RELEASE;phase++) {
        if (!ok && phase!=OPENMX_GPU_WORKER_RELEASE) continue;
        int success=1;
#pragma omp parallel for num_threads(pool->count) if(!omp_in_parallel()) reduction(&:success)
        for (int i=0;i<pool->count;i++) {
            if (phase==OPENMX_GPU_WORKER_RELEASE && !started[i]) continue;
            WorkerBinding saved;
            if (bind_worker(pool->devices[i],&saved)) {
                if (phase==OPENMX_GPU_WORKER_PREPARE) started[i]=1;
                if (!callback(context,i,phase)) success=0;
                acc_wait_all();
                if (cudaDeviceSynchronize()!=cudaSuccess) success=0;
            } else success=0;
            restore_worker(&saved);
        }
        if (!success && ok && phase>=OPENMX_GPU_WORKER_COMMIT) {
            fprintf(stderr,"<OpenMX_MGPU> %s failed during result publication or cleanup; refusing an accumulating CPU retry.\n",label?label:"worker");
            openmx_mgpu_release_devices(admission);
            exit(EXIT_FAILURE);
        }
        ok &= success;
    }
    openmx_mgpu_release_devices(admission);
    const char *verbose=getenv("OPENMX_MGPU_VERBOSE");
    if (verbose && atoi(verbose) && pool->count>1) {
        fprintf(stderr,"<OpenMX_MGPU> work name=%s workers=%d items=%zu completed=%d devices=",
            label?label:"unnamed",pool->count,pool->items,ok);
        for (int i=0;i<pool->count;i++) fprintf(stderr,"%s%d",i?",":"",pool->devices[i]);
        fputc('\n',stderr);
    }
    return ok;
}

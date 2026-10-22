#include "openmx_gpu_backend.h"
#include "openmx_env.h"
#include "openmx_gpu_device.h"
#include "openmx_gpu_runtime.h"
/* Noncollective, bounded per-device workers for existing GPU operations. */
#include "openmx_gpu_workers.h"
#include "openmx_gpu_error.h"
#include "openmx_gpu_architecture.h"
#include "openmx_mgpu_runtime.h"

#if !OPENMX_GPU_HIP
#include <openacc.h>
#endif
#include <omp.h>
#if OPENMX_GPU_HIP
#include "openmx_target.h"
#endif
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#if !OPENMX_GPU_HIP

extern void acc_clear_freelists(void);
#endif

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

/* =1 requests performance distribution even when one GPU has enough room.
   Capacity callers use openmx_gpu_workers_capacity with auto as default. */
static int workers_enabled(void)
{
    return openmx_env_flag_at(OPENMX_ENV_GPU_WORKERS, 0)==1;
}

int openmx_gpu_workers_capacity_enabled(void)
{
    if(!openmx_env_flag_at(OPENMX_ENV_GPU, 1))return 0;
    if(openmx_env_flag_at(OPENMX_ENV_GPU_WORKERS, -1)==0)return 0;
    int devices[OPENMX_GPU_WORKERS_MAX];
    return openmx_mgpu_pool_devices(devices,OPENMX_GPU_WORKERS_MAX)>0;
}

static size_t worker_cost(OpenMXGpuWorkerCost cost,OpenMXGpuWorkerDeviceCost device_cost,
    void *context,int worker,int count,int device)
{
    return device_cost ? device_cost(context,worker,count,device) : cost(context,worker,count);
}

static int workers_capacity(OpenMXGpuWorkers *pool,size_t items,int kind,
    OpenMXGpuWorkerCost cost,OpenMXGpuWorkerDeviceCost device_cost,void *context)
{
    int owner=-1,ids[OPENMX_GPU_WORKERS_MAX],available[OPENMX_GPU_WORKERS_MAX]={0};
    size_t room[OPENMX_GPU_WORKERS_MAX]={0};
    const size_t reserve=(size_t)256*1024*1024;
    memset(pool,0,sizeof(*pool)); pool->items=items;
    if (!items) return 1;
    if ((!cost && !device_cost) || OpenMXGpuPeekAtLastError()!=OPENMX_GPU_SUCCESS ||
        OpenMXGpuGetDevice(&owner)!=OPENMX_GPU_SUCCESS || owner<0) return 0;
    /* No other device is even queried when the ordinary owner fits. */
    size_t free_bytes=0,total=0;
    int owner_fits=OpenMX_GpuMemGetInfo(&free_bytes,&total)==OPENMX_GPU_SUCCESS &&
        free_bytes>reserve && worker_cost(cost,device_cost,context,0,1,owner)<=free_bytes-reserve;
    if (owner_fits && (!workers_enabled() || worker_device>=0)) {
        pool->devices[0]=owner; pool->count=1; return 1;
    }
    if (worker_device<0 && openmx_gpu_workers_capacity_enabled()) {
        int count=kind==2 ? openmx_mgpu_scope_devices(ids,OPENMX_GPU_WORKERS_MAX) :
                           openmx_mgpu_pool_devices(ids,OPENMX_GPU_WORKERS_MAX);
        if (count>OPENMX_GPU_WORKERS_MAX) count=OPENMX_GPU_WORKERS_MAX;
        for (int i=0;i<count;i++) {
            if (OpenMXGpuSetDevice(ids[i])==OPENMX_GPU_SUCCESS &&
                OpenMX_GpuMemGetInfo(&free_bytes,&total)==OPENMX_GPU_SUCCESS && free_bytes>reserve) {
                room[i]=free_bytes-reserve; available[i]=1;
            }
        }
        /* Exact scopes retain their device order. Other pools put the owner
           first; no GPU outside the permitted list is appended. */
        if (kind!=2) for (int i=0;i<count;i++) if (ids[i]==owner) {
            int d=ids[0],a=available[0]; size_t r=room[0];
            ids[0]=ids[i]; available[0]=available[i]; room[0]=room[i];
            ids[i]=d; available[i]=a; room[i]=r;
        }
        const int limit=items<(size_t)count?(int)items:count;
        const int forced=workers_enabled();
        for (int n=forced?limit:1;n>=1 && n<=limit;n+=forced?-1:1) {
            int fits=1,chosen[OPENMX_GPU_WORKERS_MAX],used[OPENMX_GPU_WORKERS_MAX]={0};
            int assigned[OPENMX_GPU_WORKERS_MAX]={0};
            size_t need[OPENMX_GPU_WORKERS_MAX];
            for (int i=0;i<n;i++)
                need[i]=worker_cost(cost,device_cost,context,i,n,kind==2?ids[i]:owner);
            /* Place the largest shard first. Exact dense scopes retain
               their order; other callers can skip a full local device. */
            for (int step=0;step<n;step++) {
                int slot=-1,device=-1;
                if (kind==2) slot=step;
                else for (int i=0;i<n;i++) if (!assigned[i] && (slot<0 || need[i]>need[slot])) slot=i;
                assigned[slot]=1;
                if (kind==2) {
                    if (available[slot] && need[slot]!=SIZE_MAX && need[slot]<=room[slot]) device=slot;
                } else {
                    for (int i=0;i<count;i++) if (!used[i] && available[i]) {
                        const size_t required=device_cost ?
                            device_cost(context,slot,n,ids[i]) : need[slot];
                        if (required!=SIZE_MAX && required<=room[i] &&
                            (device<0 || ids[i]==owner || (ids[device]!=owner && room[i]<room[device])))
                            device=i;
                    }
                }
                if (device<0) {fits=0;break;}
                used[device]=1;chosen[slot]=ids[device];
            }
            if (fits) {
                pool->count=n;
                memcpy(pool->devices,chosen,(size_t)n*sizeof(int));
                break;
            }
        }
    }
    if (OpenMXGpuSetDevice(owner)!=OPENMX_GPU_SUCCESS) { pool->count=0; return 0; }
    if (!pool->count && owner_fits) { pool->devices[0]=owner; pool->count=1; }
    const OpenMXGpuError pending=OpenMXGpuPeekAtLastError();
    if (pending!=OPENMX_GPU_SUCCESS) {
        if (pending==OPENMX_GPU_OUT_OF_MEMORY) (void)OpenMXGpuGetLastError();
        pool->count=0;
    }
    return pool->count>0;
}

int openmx_gpu_workers_capacity(OpenMXGpuWorkers *pool,size_t items,int kind,
    OpenMXGpuWorkerCost cost,void *context)
{
    return workers_capacity(pool,items,kind,cost,NULL,context);
}

int openmx_gpu_workers_capacity_device(OpenMXGpuWorkers *pool,size_t items,int kind,
    OpenMXGpuWorkerDeviceCost cost,void *context)
{
    return workers_capacity(pool,items,kind,NULL,cost,context);
}

static void worker_pool(OpenMXGpuWorkers *pool, size_t items, int kind)
{
    int owner=-1, permitted[OPENMX_GPU_WORKERS_MAX], count=0;
    memset(pool, 0, sizeof(*pool));
    pool->items=items;
    if (!items) return;
    if (OpenMXGpuGetDevice(&owner)!=OPENMX_GPU_SUCCESS || owner<0) return;
    if (worker_device>=0) { pool->devices[0]=owner; pool->count=1; return; }
    if (workers_enabled())
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
#if OPENMX_GPU_HIP
    int hip_device, omp_device, override, scope;
#else
    int cuda_device, acc_device, override, scope;
    acc_device_t acc_type;
#endif
} WorkerBinding;

static int bind_worker(int device,WorkerBinding *saved)
{
#if OPENMX_GPU_HIP
    saved->hip_device=-1;
    (void)OpenMXGpuGetDevice(&saved->hip_device);
    saved->omp_device=omp_get_default_device();
#else
    saved->cuda_device=-1;
    (void)OpenMXGpuGetDevice(&saved->cuda_device);
    saved->acc_type=acc_get_device_type();
    saved->acc_device=acc_get_device_num(saved->acc_type);
#endif
    saved->override=worker_device;
    saved->scope=openmx_mgpu_scope_enter(0);
    worker_device=device;
    if (OpenMXGpuSetDevice(device)!=OPENMX_GPU_SUCCESS) return 0;
#if OPENMX_GPU_HIP
    omp_set_default_device(device);
#else
    acc_set_device_num(device,acc_device_nvidia);
#endif
    return 1;
}

static void restore_worker(const WorkerBinding *saved)
{
#if OPENMX_GPU_HIP
    omp_set_default_device(saved->omp_device);
    if (saved->hip_device>=0) (void)OpenMXGpuSetDevice(saved->hip_device);
#else
    if (saved->acc_device>=0) acc_set_device_num(saved->acc_device,saved->acc_type);
    if (saved->cuda_device>=0) (void)OpenMXGpuSetDevice(saved->cuda_device);
#endif
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
    if (OpenMXGpuGetDevice(&original)!=OPENMX_GPU_SUCCESS) return 0;
    int enough=OpenMXGpuSetDevice(device)==OPENMX_GPU_SUCCESS && OpenMXGpuFree(0)==OPENMX_GPU_SUCCESS &&
        OpenMX_GpuMemGetInfo(&available,&capacity)==OPENMX_GPU_SUCCESS &&
        available>=(size_t)256*1024*1024;
    if (OpenMXGpuSetDevice(original)!=OPENMX_GPU_SUCCESS) return 0;
    if (!enough) {
        if (OpenMXGpuPeekAtLastError()==OPENMX_GPU_OUT_OF_MEMORY) (void)OpenMXGpuGetLastError();
        return 0;
    }
    WorkerBinding saved;
    int ok=bind_worker(device,&saved);
    if (ok) {
        size_t free_bytes=0,total_bytes=0;
        ok=OpenMXGpuFree(0)==OPENMX_GPU_SUCCESS &&
           OpenMX_GpuMemGetInfo(&free_bytes,&total_bytes)==OPENMX_GPU_SUCCESS &&
           free_bytes>=(size_t)256*1024*1024;
        if (ok) {
#if OPENMX_GPU_HIP
            void *touch=openmx_target_alloc((size_t)1024*1024);
#else
            void *touch=acc_malloc((size_t)1024*1024);
#endif
            ok=touch!=NULL;
#if OPENMX_GPU_HIP
            if (touch) openmx_target_free(touch);
#else
            if (touch) acc_free(touch);
            if (ok) OpenMX_GpuPresizeKernelStack();
#endif
        }
    }
    if (!ok && OpenMXGpuPeekAtLastError()==OPENMX_GPU_OUT_OF_MEMORY) (void)OpenMXGpuGetLastError();
    restore_worker(&saved);
    if (ok) device_ready[device]=1;
    return ok;
}

int openmx_gpu_device_capabilities(int device, OpenMXAcceleratorCapabilities *caps)
{
    if (!caps) return 0;
    *caps=(OpenMXAcceleratorCapabilities){
        (OpenMXAcceleratorBackend)OPENMX_GPU_BACKEND,OPENMX_FP64_UNKNOWN,OPENMX_MEMORY_UNKNOWN,0};
    if (OpenMXGpuPeekAtLastError()!=OPENMX_GPU_SUCCESS) return 0;
#if OPENMX_GPU_HIP
    hipDeviceProp_t prop;
    if (hipGetDeviceProperties(&prop,device)!=hipSuccess) return 0;
    caps->memory=prop.integrated ? OPENMX_MEMORY_SHARED : OPENMX_MEMORY_DISCRETE;
    caps->fp64=!prop.arch.hasDoubles ? OPENMX_FP64_UNAVAILABLE :
        openmx_amd_high_throughput_fp64(prop.gcnArchName) ?
            OPENMX_FP64_NATIVE_HIGH_THROUGHPUT : OPENMX_FP64_NATIVE;
#else
    int ratio=0;
    if(cudaDeviceGetAttribute(&ratio,cudaDevAttrSingleToDoublePrecisionPerfRatio,device)!=cudaSuccess)
        return 0;
    caps->fp32_to_fp64_ratio=ratio;
    caps->fp64=ratio>0 && ratio<8 ? OPENMX_FP64_NATIVE_HIGH_THROUGHPUT : OPENMX_FP64_NATIVE;
#endif
    return 1;
}

int openmx_gpu_device_weak_fp64(int device)
{
    OpenMXAcceleratorCapabilities caps;
    return openmx_gpu_device_capabilities(device,&caps) &&
        caps.fp64==OPENMX_FP64_NATIVE && (caps.backend==OPENMX_ACCELERATOR_HIP ||
        caps.fp32_to_fp64_ratio>=8);
}

int openmx_gpu_worker_can_overlap(int device)
{
    {
        const int control=openmx_env_flag_at(OPENMX_ENV_GPU_WORKER_OVERLAP, -1);
        if(0<=control)return control;
    }
    /* Only the CUDA implementation currently has a calibrated overlap
       default. Avoid a device query on every HIP task just to return false. */
    if(OPENMX_GPU_BACKEND!=OPENMX_BACKEND_CUDA)return 0;
    OpenMXAcceleratorCapabilities caps;
    return openmx_gpu_device_capabilities(device,&caps) &&
        openmx_accelerator_default_worker_overlap(caps);
}

static int run_workers(const OpenMXGpuWorkers *pool,
    int (*callback)(void*,int,int),void *context,const char *label,int prepared)
{
    int ok=1, started[OPENMX_GPU_WORKERS_MAX]={0};
    if (!pool || !callback || pool->count<0 || pool->count>OPENMX_GPU_WORKERS_MAX) return 0;
    if (!pool->count) return pool->items==0;
    /* Nested worker launch would deadlock the shared admission mutex. */
    if (worker_device>=0) return 0;
    if (OpenMXGpuPeekAtLastError()!=OPENMX_GPU_SUCCESS) return 0;
    if (gpu_rank_device_usable() && OpenMXGpuDeviceSynchronize()!=OPENMX_GPU_SUCCESS) return 0;
    void *admission=openmx_mgpu_admit_devices(pool->devices,pool->count);
    if (!admission) return 0;
    for (int i=0;i<pool->count;i++)
        if (!prepare_device(pool->devices[i])) { ok=0; break; }
    int overlap=prepared;
    for(int worker=0;overlap && worker<pool->count;worker++)
        overlap=openmx_gpu_worker_can_overlap(pool->devices[worker]);
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
#if OPENMX_GPU_HIP
                openmx_target_wait();
#else
                acc_wait_all();
#endif
                if (OpenMXGpuDeviceSynchronize()!=OPENMX_GPU_SUCCESS) success=0;
            } else success=0;
            restore_worker(&saved);
        }
        if (!success && ok && phase>=OPENMX_GPU_WORKER_COMMIT) {
            fprintf(stderr,"<OpenMX_MGPU> %s failed during result publication or cleanup; refusing an accumulating CPU retry.\n",label?label:"worker");
            if(admission)openmx_mgpu_release_devices(admission);
            openmx_gpu_fatal("GPU worker failed during result publication or cleanup",
                __FILE__, __LINE__);
        }
        ok &= success;
        if(ok && overlap && phase==OPENMX_GPU_WORKER_PREPARE){
            openmx_mgpu_release_devices(admission);
            admission=NULL;
        }
    }
    if(admission)openmx_mgpu_release_devices(admission);
    if (openmx_env_flag_at(OPENMX_ENV_MGPU_VERBOSE, 0) && (pool->count>1 || openmx_gpu_capacity_test_recovery())) {
        fprintf(stderr,"<OpenMX_MGPU> work name=%s workers=%d items=%zu completed=%d devices=",
            label?label:"unnamed",pool->count,pool->items,ok);
        for (int i=0;i<pool->count;i++) fprintf(stderr,"%s%d",i?",":"",pool->devices[i]);
        fputc('\n',stderr);
    }
    return ok;
}

int openmx_gpu_workers_run(const OpenMXGpuWorkers *pool,
    int (*callback)(void*,int,int),void *context,const char *label)
{
    return run_workers(pool,callback,context,label,0);
}

int openmx_gpu_workers_run_prepared(const OpenMXGpuWorkers *pool,
    int (*callback)(void*,int,int),void *context,const char *label)
{
    return run_workers(pool,callback,context,label,1);
}

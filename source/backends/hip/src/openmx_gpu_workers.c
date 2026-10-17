/* Noncollective, bounded per-device workers for existing GPU operations. */
#include "openmx_gpu_workers.h"
#include "openmx_mgpu_runtime.h"
#include "set_hip_default_device_from_local_rank.h"
#include <omp.h>
#include "openmx_target.h"
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

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
    const char *control=getenv("OPENMX_GPU_WORKERS");
    return control && (strcmp(control,"1")==0 || strcmp(control,"on")==0 ||
                       strcmp(control,"ON")==0);
}

int openmx_gpu_workers_capacity_enabled(void)
{
    const char *gpu=getenv("OPENMX_GPU");
    if(gpu && *gpu && atoi(gpu)==0)return 0;
    const char *v=getenv("OPENMX_GPU_WORKERS");
    if(v && (strcmp(v,"0")==0 || strcmp(v,"off")==0 || strcmp(v,"OFF")==0))return 0;
    int devices[OPENMX_GPU_WORKERS_MAX];
    return openmx_mgpu_pool_devices(devices,OPENMX_GPU_WORKERS_MAX)>0;
}

int openmx_gpu_workers_capacity(OpenMXGpuWorkers *pool,size_t items,int kind,
    OpenMXGpuWorkerCost cost,void *context)
{
    int owner=-1,ids[OPENMX_GPU_WORKERS_MAX],available[OPENMX_GPU_WORKERS_MAX]={0};
    size_t room[OPENMX_GPU_WORKERS_MAX]={0};
    const size_t reserve=(size_t)256*1024*1024;
    memset(pool,0,sizeof(*pool)); pool->items=items;
    if (!items) return 1;
    if (!cost || hipGetDevice(&owner)!=hipSuccess || owner<0) return 0;
    /* No other device is even queried when the ordinary owner fits. */
    size_t free_bytes=0,total=0;
    int owner_fits=OpenMX_GpuMemGetInfo(&free_bytes,&total)==hipSuccess &&
        free_bytes>reserve && cost(context,0,1)<=free_bytes-reserve;
    if (owner_fits && (!workers_enabled() || worker_device>=0)) {
        pool->devices[0]=owner; pool->count=1; return 1;
    }
    if (worker_device<0 && openmx_gpu_workers_capacity_enabled()) {
        int count=kind==2 ? openmx_mgpu_scope_devices(ids,OPENMX_GPU_WORKERS_MAX) :
                           openmx_mgpu_pool_devices(ids,OPENMX_GPU_WORKERS_MAX);
        if (count>OPENMX_GPU_WORKERS_MAX) count=OPENMX_GPU_WORKERS_MAX;
        for (int i=0;i<count;i++) {
            if (hipSetDevice(ids[i])==hipSuccess &&
                OpenMX_GpuMemGetInfo(&free_bytes,&total)==hipSuccess && free_bytes>reserve) {
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
            for (int i=0;i<n;i++) need[i]=cost(context,i,n);
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
                    for (int i=0;i<count;i++) if (!used[i] && available[i] &&
                        need[slot]!=SIZE_MAX && need[slot]<=room[i]) {
                        if (device<0 || ids[i]==owner || (ids[device]!=owner && room[i]<room[device])) device=i;
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
    if (hipSetDevice(owner)!=hipSuccess) { pool->count=0; return 0; }
    if (!pool->count && owner_fits) { pool->devices[0]=owner; pool->count=1; }
    (void)hipGetLastError();
    return pool->count>0;
}

static void worker_pool(OpenMXGpuWorkers *pool, size_t items, int kind)
{
    int owner=-1, permitted[OPENMX_GPU_WORKERS_MAX], count=0;
    memset(pool, 0, sizeof(*pool));
    pool->items=items;
    if (!items) return;
    if (hipGetDevice(&owner)!=hipSuccess || owner<0) return;
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
    int hip_device, omp_device, override, scope;
} WorkerBinding;

static int bind_worker(int device,WorkerBinding *saved)
{
    saved->hip_device=-1;
    (void)hipGetDevice(&saved->hip_device);
    saved->omp_device=omp_get_default_device();
    saved->override=worker_device;
    saved->scope=openmx_mgpu_scope_enter(0);
    worker_device=device;
    if (hipSetDevice(device)!=hipSuccess) return 0;
    omp_set_default_device(device);
    return 1;
}

static void restore_worker(const WorkerBinding *saved)
{
    omp_set_default_device(saved->omp_device);
    if (saved->hip_device>=0) (void)hipSetDevice(saved->hip_device);
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
    /* Check HIP capacity before OpenMP loads its module on a new device.
       omp_set_default_device may otherwise abort in a recoverable low-memory
       situation before the callback has a chance to decline. */
    int original=-1;
    size_t available=0,capacity=0;
    if (hipGetDevice(&original)!=hipSuccess) return 0;
    int enough=hipSetDevice(device)==hipSuccess && hipFree(0)==hipSuccess &&
        OpenMX_GpuMemGetInfo(&available,&capacity)==hipSuccess &&
        available>=(size_t)256*1024*1024;
    if (hipSetDevice(original)!=hipSuccess) return 0;
    if (!enough) { (void)hipGetLastError();return 0; }
    WorkerBinding saved;
    int ok=bind_worker(device,&saved);
    if (ok) {
        size_t free_bytes=0,total_bytes=0;
        ok=hipFree(0)==hipSuccess &&
           OpenMX_GpuMemGetInfo(&free_bytes,&total_bytes)==hipSuccess &&
           free_bytes>=(size_t)256*1024*1024;
        if (ok) {
            void *touch=openmx_target_alloc((size_t)1024*1024);
            ok=touch!=NULL;
            if (touch) openmx_target_free(touch);
        }
    }
    if (!ok) (void)hipGetLastError();
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
    if (gpu_rank_device_usable() && hipDeviceSynchronize()!=hipSuccess) return 0;
    (void)hipGetLastError();
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
                openmx_target_wait();
                if (hipDeviceSynchronize()!=hipSuccess) success=0;
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
    if (verbose && atoi(verbose) && (pool->count>1 || openmx_gpu_capacity_test_recovery())) {
        fprintf(stderr,"<OpenMX_MGPU> work name=%s workers=%d items=%zu completed=%d devices=",
            label?label:"unnamed",pool->count,pool->items,ok);
        for (int i=0;i<pool->count;i++) fprintf(stderr,"%s%d",i?",":"",pool->devices[i]);
        fputc('\n',stderr);
    }
    return ok;
}

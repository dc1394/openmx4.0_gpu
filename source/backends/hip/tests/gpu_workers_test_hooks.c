/* Standalone worker-kernel fixture. Production admission/scope planning is
   separately exercised by run_mgpu_runtime_cpu.py. */
#include <assert.h>
#include "hip_runtime_compat.h"
#include <pthread.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>
#include "openmx_gpu_workers.h"
static pthread_mutex_t gate=PTHREAD_MUTEX_INITIALIZER;
static _Thread_local int scope;
int probe_gpus=4,probe_first=0,probe_count=4;
/* Quadrature fixtures may run on one to four visible devices. Dense-group
   fixtures instead set their own exact non-prefix group configurations. */
void openmx_gpu_test_use_visible_pool(void){
    int count=0;
    assert(hipGetDeviceCount(&count)==hipSuccess && count>0);
    probe_gpus=count<4?count:4;probe_first=0;probe_count=probe_gpus;
}
int openmx_mgpu_scope_enter(int allowed){int old=scope;scope=allowed?scope+1:0;return old;}
void openmx_mgpu_scope_leave(int old){scope=old;}
int openmx_mgpu_pool_devices(int *devices,int capacity){
    assert(probe_gpus<=capacity);
    for(int i=0;i<probe_gpus;i++)devices[i]=i;
    return probe_gpus;
}
int openmx_mgpu_scope_devices(int *devices,int capacity){
    assert(probe_count<=capacity && probe_first+probe_count<=probe_gpus);
    for(int i=0;i<probe_count;i++)devices[i]=probe_first+i;
    return probe_count;
}
void *openmx_mgpu_admit_devices(const int *devices,int count){
    for(int i=0;i<count;i++)assert(devices[i]>=0 && devices[i]<probe_gpus);
    pthread_mutex_lock(&gate);return &gate;
}
void openmx_mgpu_release_devices(void *p){assert(p==&gate);pthread_mutex_unlock(&gate);}
int openmx_gpu_map_rank_to_device(int rank,int size,int count){return count>0?rank/((size+count-1)/count):-1;}
hipError_t OpenMX_GpuMemGetInfo(size_t *free_bytes,size_t *total_bytes){return hipMemGetInfo(free_bytes,total_bytes);}
void OpenMX_GpuPresizeKernelStack(void){}

typedef struct {
    OpenMXGpuWorkers pool;
    int fail_phase;
    unsigned char *private_output[4],published[4][16];
    int seen[4][4];
} WorkerFailureProbe;
static int worker_failure(void *opaque,int index,int phase){
    WorkerFailureProbe *w=opaque;
    int device=-1;
    assert(hipGetDevice(&device)==hipSuccess && device==w->pool.devices[index]);
    assert(omp_get_default_device()==device && openmx_gpu_worker_device()==device);
    ++w->seen[index][phase];
    if(phase==OPENMX_GPU_WORKER_PREPARE)
        assert(hipMalloc((void**)&w->private_output[index],16)==hipSuccess);
    else if(phase==OPENMX_GPU_WORKER_EXECUTE)
        assert(hipMemset(w->private_output[index],index+1,16)==hipSuccess);
    else if(phase==OPENMX_GPU_WORKER_COMMIT)
        assert(hipMemcpy(w->published[index],w->private_output[index],16,hipMemcpyDeviceToHost)==hipSuccess);
    else if(phase==OPENMX_GPU_WORKER_RELEASE)
        assert(hipFree(w->private_output[index])==hipSuccess);
    return !(phase==w->fail_phase && index==w->pool.count-1);
}
/* Compile the real phase executor with ROCm as well as the host mock's
   GCC checks: failed OpenMP reductions must never publish partial sums. */
void openmx_gpu_test_worker_failures(void){
    int dynamic=omp_get_dynamic(),original=-1;
    assert(hipGetDevice(&original)==hipSuccess);
    omp_set_default_device(original);
    for(int teams=0;teams<2;teams++) for(int fail=-1;fail<=1;fail++){
        WorkerFailureProbe w={0};w.fail_phase=fail;
        memset(w.published,0x5a,sizeof(w.published));
        omp_set_dynamic(teams);
        openmx_gpu_workers_owner(&w.pool,16);
        assert(w.pool.count==probe_gpus);
        int ok=openmx_gpu_workers_run(&w.pool,worker_failure,&w,"worker-failure-probe");
        assert(ok==(fail<0));
        for(int i=0;i<w.pool.count;i++){
            assert(w.seen[i][OPENMX_GPU_WORKER_PREPARE]==1 && w.seen[i][OPENMX_GPU_WORKER_RELEASE]==1);
            assert(w.seen[i][OPENMX_GPU_WORKER_EXECUTE]==(fail!=0));
            assert(w.seen[i][OPENMX_GPU_WORKER_COMMIT]==(fail<0));
            for(int j=0;j<16;j++)assert(w.published[i][j]==(fail<0?i+1:0x5a));
        }
        int current=-1;assert(hipGetDevice(&current)==hipSuccess && current==original);
        assert(omp_get_default_device()==original && openmx_gpu_worker_device()==-1);
    }
    omp_set_dynamic(dynamic);
    puts("PASS: actual HIP/OpenMP worker rollback, publication barriers and dynamic teams");
}


int gpu_rank_device_usable(void) {return 1;}

#ifndef OPENMX_GPU_WORKERS_H
#define OPENMX_GPU_WORKERS_H

#include "openmx_accelerator.h"

#include "openmx_gpu_backend.h"

#include <stddef.h>
#include <stdlib.h>
#include "openmx_env.h"

#ifdef __cplusplus
extern "C" {
#endif

#define OPENMX_GPU_WORKERS_MAX 64

typedef struct {
    int count;
    int devices[OPENMX_GPU_WORKERS_MAX];
    size_t items;
} OpenMXGpuWorkers;

enum {
    OPENMX_GPU_WORKER_PREPARE,
    OPENMX_GPU_WORKER_EXECUTE,
    OPENMX_GPU_WORKER_COMMIT,
    OPENMX_GPU_WORKER_RELEASE
};

/* Initialize after input has trimmed the Band GPU pool. No communication
   occurs in a worker. The rank map stays identical to the original binding. */
void openmx_gpu_workers_configure(int local_rank, int local_size, int mapped_devices);

/* Local work keeps each rank's established GPU and assigns otherwise idle
   devices to ranks cyclically. Owner services may use the entire allowed
   pool; k-dependent work must use the current dense solver's exact group. */
void openmx_gpu_workers_local(OpenMXGpuWorkers *workers, size_t items);
void openmx_gpu_workers_owner(OpenMXGpuWorkers *workers, size_t items);
void openmx_gpu_workers_scope(OpenMXGpuWorkers *workers, size_t items);

/* Capacity planning: 0=rank-local, 1=owner service, 2=exact dense scope.
   cost gives the complete additional bytes of worker i in a count-way cut.
   Auto tries the owner alone, then increasing node-local groups. count==0
   means no admitted GPU layout; the caller must try its collective tier or
   CPU path. Unlike the optional performance splitter, capacity recovery may
   borrow GPUs assigned to other local ranks under the UUID admission lock.
   OPENMX_GPU_WORKERS=0 disables borrowing; =1 prefers the largest group. */
typedef size_t (*OpenMXGpuWorkerCost)(void *context, int worker, int count);
/* Some inputs are resident only on their owning device. These callbacks
   receive the candidate physical ordinal so remote-copy storage is charged
   to the actual target, even for a one-worker group. */
typedef size_t (*OpenMXGpuWorkerDeviceCost)(void *context, int worker, int count, int device);
/* Capacity can change after planning (or module initialization). A caller
   with private results may reject a failed layout and try the remaining
   node-local group sizes before asking other nodes for help. */
typedef struct {
    OpenMXGpuWorkerCost cost;
    void *context;
    unsigned char rejected[OPENMX_GPU_WORKERS_MAX+1];
} OpenMXGpuWorkerRetry;
static inline size_t openmx_gpu_worker_retry_cost(void *opaque,int worker,int count)
{
    OpenMXGpuWorkerRetry *retry=(OpenMXGpuWorkerRetry*)opaque;
    if (count<1 || count>OPENMX_GPU_WORKERS_MAX || retry->rejected[count]) return (size_t)-1;
    return retry->cost(retry->context,worker,count);
}
int openmx_gpu_workers_capacity(OpenMXGpuWorkers *workers, size_t items, int kind,
    OpenMXGpuWorkerCost cost, void *context);
int openmx_gpu_workers_capacity_device(OpenMXGpuWorkers *workers, size_t items, int kind,
    OpenMXGpuWorkerDeviceCost cost, void *context);
int openmx_gpu_workers_capacity_enabled(void);

/* Fault injection for regression tests: skip only the legacy resident fast
   paths, then exercise the ordinary capacity planner with real GPU memory. */
#if OPENMX_GPU_HIP
static inline int openmx_gpu_workers_requested(void)
{
    const char *v=openmx_env_get("OPENMX_GPU_WORKERS");
    return v && ((v[0]=='1' && v[1]=='\0') || (((v[0]=='o' && v[1]=='n') || (v[0]=='O' && v[1]=='N')) && v[2]=='\0'));
}

/* Fault injection for regression tests: skip only the legacy resident fast
   paths, then exercise the ordinary capacity planner with real GPU memory. */
#endif
static inline int openmx_gpu_capacity_test_recovery(void)
{
    const char *value = openmx_env_get("OPENMX_GPU_CAPACITY_TEST_RECOVERY");
    return value && value[0]=='1' && value[1]=='\0';
}

/* Balanced integer cuts without overflowing items * index. */
static inline size_t openmx_gpu_worker_cut(size_t items, int index, int count)
{
    return count > 0 ? (items / (size_t)count) * (size_t)index +
        ((items % (size_t)count) * (size_t)index) / (size_t)count : 0;
}

/* Device-binding helpers honor this override inside callbacks. It is
   thread-local; -1 means the ordinary rank binding remains in effect. */
int openmx_gpu_worker_device(void);
/* Query actual capability independently of the backend's scheduling policy.
   Failure returns zero and leaves an unknown, non-authorizing descriptor. */
int openmx_gpu_device_capabilities(int device, OpenMXAcceleratorCapabilities *caps);

/* Each phase finishes on every device before the next phase starts. All
   input uploads and allocations belong in PREPARE. EXECUTE writes private
   results. COMMIT publishes only after all workers succeed. RELEASE runs
   for every started worker, including a partially prepared failed worker.
   Callbacks must not perform MPI or start another distributed GPU operation.
   CUDA/OpenACC binding and the caller's dense scope are restored per thread.
   The runtime uses explicit device-index loops, so dynamic OpenMP teams and
   OMP_NUM_THREADS=1 cannot omit work. Returns 0 on an uncommitted failure. */
int openmx_gpu_workers_run(const OpenMXGpuWorkers *workers,
    int (*callback)(void *context, int worker, int phase), void *context,
    const char *label);
/* Opt-in for callbacks with no device allocation or input upload after
   PREPARE: GeForce-class CUDA devices release admission before EXECUTE so
   MPS may overlap private batches. Datacenter GPUs and HIP retain admission
   by default. OPENMX_GPU_WORKER_OVERLAP=0/1 overrides the hardware policy.
   Failed PREPARE retains admission through rollback in either mode. */
int openmx_gpu_worker_can_overlap(int device);
/* Hardware query only; environment overrides do not change this result. */
int openmx_gpu_device_weak_fp64(int device);
int openmx_gpu_workers_run_prepared(const OpenMXGpuWorkers *workers,
    int (*callback)(void *context, int worker, int phase), void *context,
    const char *label);

#ifdef __cplusplus
}
#endif
#endif

#ifndef OPENMX_GPU_WORKERS_H
#define OPENMX_GPU_WORKERS_H

#include <stddef.h>

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

/* Balanced integer cuts without overflowing items * index. */
static inline size_t openmx_gpu_worker_cut(size_t items, int index, int count)
{
    return count > 0 ? (items / (size_t)count) * (size_t)index +
        ((items % (size_t)count) * (size_t)index) / (size_t)count : 0;
}

/* Device-binding helpers honor this override inside callbacks. It is
   thread-local; -1 means the ordinary rank binding remains in effect. */
int openmx_gpu_worker_device(void);

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

#ifdef __cplusplus
}
#endif
#endif

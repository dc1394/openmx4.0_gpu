#ifndef OPENMX_GPU_COLLECTIVE_H
#define OPENMX_GPU_COLLECTIVE_H
#include <stddef.h>
#include <mpi.h>

/* A collective capacity retry for host-backed, independently partitionable
   work. All ranks enter at the same phase boundary, including ranks whose
   original work already succeeded. No MPI occurs inside a GPU callback.

   pack/run allocate their returned buffers with malloc. run must keep its
   results private: commit is called only on the originating rank, after
   EVERY remote piece succeeded. Buffers must contain values, never process
   addresses. Empty work and failed allocations participate in all required
   communication. CPU retry is safe after a zero return.

   Call only after the owner-GPU and node-local-GPU attempts. One rank per
   node executes a shard, and may use that node's bounded local worker pool.
   GPU eligibility must already respect input and scheduler device limits. */
typedef struct {
    int (*pack)(void *context, size_t first, size_t last, void **packet, size_t *bytes);
    int (*run)(void *context, const void *packet, size_t bytes, void **result, size_t *result_bytes);
    void (*commit)(void *context, size_t first, size_t last, const void *result, size_t bytes);
} OpenMXGpuCollectiveOps;

int openmx_gpu_collective_retry(MPI_Comm comm, const OpenMXGpuCollectiveOps *ops,
    void *context, size_t items, int local_done, int can_help, const char *label);
#endif

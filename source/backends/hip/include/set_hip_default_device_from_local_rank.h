#ifndef _SET_HIP_DEFAULT_DEVICE_FROM_LOCAL_RANK_H_
#define _SET_HIP_DEFAULT_DEVICE_FROM_LOCAL_RANK_H_

#include <mpi.h>
#include <stddef.h>

int set_hip_default_device_from_local_rank();
int set_hip_default_device_from_local_rank_noncollective(void);

/* Shared rank-to-device map used by every site that binds a rank to a GPU.
   Contiguous blocks of local ranks share a device (block map) instead of the
   old round-robin (local_rank % device_count): the Band solver's k-point
   worlds are contiguous rank blocks whose owner ranks sit world-size apart,
   so a round-robin map can place every k-owner on device 0 and serialize
   their concurrent eigensolves.  The block map spreads the owners across
   the allowed devices while keeping the same number of ranks per device,
   and it must be the ONLY map in the binary: a rank's HIP and OpenMP device
   must stay identical from initialization to the last SCF step.  Set
   OPENMX_GPU_RANK_MAP=mod to restore the old map. */
int openmx_gpu_map_rank_to_device(int local_rank, int local_size, int device_count);

/* Node-local rank/size without communication (launcher-provided envs,
   falling back to MPI_COMM_WORLD, which is exact on single-node runs). */
int openmx_gpu_local_rank_noncollective(void);
int openmx_gpu_local_size_noncollective(void);

/* Ranks sharing one GPU (node-local size / visible devices), no
   communication.  The dense-GPU band eigensolvers fall back to the CPU
   (ELPA2) path when this exceeds OPENMX_BAND_GPU_MAX_DEVICE_RANKS, because a
   heavily shared device serializes their per-rank solves and contends on the
   large packed-matrix gather (a UCX shared-memory assertion was observed in
   that gather for the 40-atom noncollinear GGFF at 24 ranks on one MI300A),
   while the CPU path parallelizes across every rank's cores and is both
   faster and stable in that regime. */
int openmx_gpu_ranks_per_device_noncollective(void);

/* True when the GPU is shared by more ranks than the dense-GPU band
   eigensolvers use well (see above); the band switch-number functions then
   report an unreachable threshold so the CPU (ELPA2) path is taken, and the
   packed H/S gather that feeds the dense GPU path is not built. */
int openmx_band_gpu_dense_oversubscribed(void);

/* One-time, noncollective, per-rank answer to "may this rank touch the GPU
   at all?".  Creates the rank's HIP context through error-returning runtime
   calls, verifies a free-memory margin, and forces OpenMP target
   initialization.  Ranks that fail keep the OpenMP initial device and must
   take host code paths. */
int gpu_rank_device_usable(void);

/* The free and total memory of this rank's device: 0 (hipSuccess) or the
   HIP error.  The NVIDIA version caps the free memory with NVML because
   under WSL2/WDDM a process does not see the others' allocations; on Linux
   ROCm hipMemGetInfo is device-wide (other processes' allocations
   included), so this is the plain query, kept as the single entry point of
   the memory tests. */
int OpenMX_GpuMemGetInfo(size_t *free_bytes, size_t *total_bytes);

#endif // _SET_HIP_DEFAULT_DEVICE_FROM_LOCAL_RANK_H_

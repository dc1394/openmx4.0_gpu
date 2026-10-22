#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L   /* setenv */
#endif
/**********************************************************************
  openmx_env.c

  The environment settings of this build, read once.  See openmx_env.h.

  The registry below lists every name the sources read as a string literal
  (getenv and the readers' own helpers), the kinds are hints for the
  documentation, the descriptions come from doc/OpenMX40_GPU_implementation_en.html
  section 10.2 where it has an entry.  tools/audit_global_usage.py reports
  names read by the code that are missing here.
***********************************************************************/
#include "openmx_env.h"

#include <errno.h>
#include <stdlib.h>
#include <string.h>

extern char **environ;

/* --- registry: sorted by name (openmx_env_lookup uses bsearch) --------- */
static const OpenMXEnvEntry registry[] = {
    {"GEMMUL8_DISABLE", "flag", "gemmul8_bridge.cu", ""},
    {"GEMMUL8_DISABLE_D", "flag", "gemmul8_bridge.cu", ""},
    {"GEMMUL8_DISABLE_Z", "flag", "gemmul8_bridge.cu", ""},
    {"GEMMUL8_FASTMODE_D", "flag", "gemmul8_bridge.cu", ""},
    {"GEMMUL8_FASTMODE_Z", "flag", "gemmul8_bridge.cu", ""},
    {"GEMMUL8_MAX_WORKSPACE_MB", "mib", "gemmul8_bridge.cu", ""},
    {"GEMMUL8_MAX_WORKSPACE_PERCENT", "int", "Band_DFT_NonCol.c,gemmul8_bridge.cu", ""},
    {"GEMMUL8_MIN_FREE_AFTER_MB", "mib", "Divide_Conquer.c,gemmul8_bridge.cu", ""},
    {"GEMMUL8_NUM_MOD_D", "flag", "gemmul8_bridge.cu", ""},
    {"GEMMUL8_NUM_MOD_Z", "flag", "gemmul8_bridge.cu", ""},
    {"GEMMUL8_VERBOSE", "flag", "gemmul8_bridge.cu", ""},
    {"OPENMX_AFFINITY_DIR", "tool", "build/tools", "mi300a_rank_launcher.py: directory of the per-rank CPU affinity files"},
    {"OPENMX_BAND_CACHE_TRACE", "flag", "Band_DFT_Col.c", ""},
    {"OPENMX_BAND_COL_KCACHE_MB", "mib", "Band_DFT_Col.c", ""},
    {"OPENMX_BAND_COL_OVERLAP_CACHE_MB", "mib", "Band_DFT_Col.c", ""},
    {"OPENMX_BAND_GEMMUL8_TURN_RELEASE", "string", "Band_DFT_Col.c,Band_DFT_NonCol.c,Cluster_DFT_Col.c,Cluster_DFT_NonCol.c,Divide_Conquer_LNO.c", "default 1: Per-turn GEMMul8 workspace release (0 keeps them)"},
    {"OPENMX_BAND_GPU_DIAG", "string", "Band_DFT_Col.c,Band_DFT_NonCol.c", "default auto: 1 = force the dense GPU path ignoring the fits verdict; 0 = always ELPA/ScaLAPACK (§5.6)"},
    {"OPENMX_BAND_GPU_DM", "flag", "Band_DFT_Col.c", "default 1: 0 = CPU loop for the DM accumulation of the collinear band ScaLAPACK/ELPA fallback. By default the fallback keeps the DM on the GPU: the eigenvecto..."},
    {"OPENMX_BAND_GPU_MAX_CONCURRENT_K", "int", "Band_DFT_Col.c,Band_DFT_NonCol.c", ""},
    {"OPENMX_BAND_GPU_MAX_RANKS_PER_DEVICE", "int", "Band_DFT_Col.c,Band_DFT_NonCol.c", ""},
    {"OPENMX_BAND_GPU_PERSISTENT", "flag", "Band_DFT_Col.c", ""},
    {"OPENMX_BAND_GPU_PERSISTENT_MIN_FREE_MB", "mib", "Band_DFT_Col.c", ""},
    {"OPENMX_BAND_GPU_RANK_OVERHEAD_MB", "mib", "Band_DFT_Col.c,Band_DFT_NonCol.c", "default 256: Runtime overhead added to the per-rank requirement model"},
    {"OPENMX_BAND_GPU_RESERVE_MB", "mib", "Band_DFT_Col.c,Band_DFT_NonCol.c", "default max(total/10, 1536): Raises the reserve used by the concurrency planner and the fits verdicts"},
    {"OPENMX_BAND_GPU_SWITCH_NUM", "int", "Band_DFT_Col.c", "default 800: GPU/CPU crossover dimension of the collinear band dense path (paths without their own knob keep GPU_CPU_SWITCH_NUM=1000). Measured crossover on one..."},
    {"OPENMX_BAND_GPU_TURN_GROUP", "int", "Band_DFT_Col.c,Band_DFT_NonCol.c", ""},
    {"OPENMX_BAND_NONCOL_CACHE_TRACE", "flag", "Band_DFT_NonCol.c", ""},
    {"OPENMX_BAND_NONCOL_GPU_DM", "flag", "Band_DFT_NonCol.c", "default 1: 0 = CPU loop for the per-k DM accumulation of the noncollinear ScaLAPACK/ELPA-fallback multi-k loop (the k-dense GPU flow keeps its own device DM r..."},
    {"OPENMX_BAND_NONCOL_GPU_SWITCH_NUM", "int", "Band_DFT_NonCol.c", "default 800: Same for the noncollinear band dense path, compared against 2n (measured crossover 2n ≈ 250–400)"},
    {"OPENMX_BAND_NONCOL_INTERLEAVE_K", "flag", "Band_DFT_NonCol.c", ""},
    {"OPENMX_BAND_NONCOL_KCACHE_MB", "mib", "Band_DFT_NonCol.c", ""},
    {"OPENMX_BAND_NONCOL_OVERLAP_CACHE_MB", "mib", "Band_DFT_NonCol.c", ""},
    {"OPENMX_BAND_NONCOL_PACK_CACHE_MB", "mib", "Band_DFT_NonCol.c", ""},
    {"OPENMX_BAND_PROFILE", "flag", "Band_DFT_Col.c,Cluster_DFT_Col.c,Force.c,Set_Density_Grid.c,Set_Hamiltonian.c,Set_Nonlocal.c,Set_OLP_Kin.c,Set_Vpot.c,Total_Energy.c,truncation.c", "default 0: Performance counters; also gates the per-rank stage lines SETHPROF (Set_Hamiltonian), SETHOTFPROF / SETHOTFDENPROF (the on-the-fly batches: pack, e..."},
    {"OPENMX_BAND_REFINE_BASIS_MB", "mib", "Band_DFT_Col.c,Band_DFT_NonCol.c", ""},
    {"OPENMX_BAND_SYEVDX_NOVECTOR", "string", "Band_DFT_Col.c,Band_DFT_NonCol.c", "default 1: Solve the eigenvalues-only first pass with NOVECTOR (0 restores the old behavior)"},
    {"OPENMX_BUILD_JOBS", "tool", "build/tools", "CMake: parallelism of the native build"},
    {"OPENMX_CLUSTER_GEMMUL8_TURN_RELEASE", "string", "Cluster_DFT_Col.c,Cluster_DFT_NonCol.c", "default 1: Per-solve GEMMul8 workspace release (inherits the BAND setting when unset)"},
    {"OPENMX_CLUSTER_GPU_DIAG", "string", "Cluster_DFT_Col.c,Cluster_DFT_NonCol.c", "default auto: 1 = force the GPU even if the reservation fails (then aborts with a diagnostic); 0 = always ELPA (§5.6)"},
    {"OPENMX_CLUSTER_GPU_DIAG_RESERVE_MB", "mib", "Cluster_DFT_Col.c,Cluster_DFT_NonCol.c", "default 1024: Free VRAM required to remain after the reservation probe"},
    {"OPENMX_CLUSTER_GPU_DIAG_SERIAL", "flag", "Cluster_DFT_Col.c", "default 1: Serialized mode when both spin owners cannot fit (0 = disable, 2 = force serial)"},
    {"OPENMX_CLUSTER_GPU_DM", "flag", "Cluster_DFT_Col.c", "default 1: GPU accumulation of DM/EDM (0 = CPU loop)"},
    {"OPENMX_CLUSTER_GPU_DM_ALL", "flag", "Cluster_DFT_Col.c", "default 1: Device accumulation of the cluster density matrix on every rank (its slice of states, staged with a halving retry); 0 = only the rank that owns the..."},
    {"OPENMX_CLUSTER_GPU_DM_GEMM", "flag", "Cluster_DFT_NonCol.c", "default 1: Noncollinear cluster dense path: the density matrix as one product WHW over the occupied states (GEMMul8) plus a gather of the sparse entries, and..."},
    {"OPENMX_CLUSTER_GPU_RETAIN_SCRATCH", "flag", "Cluster_DFT_Col.c", ""},
    {"OPENMX_CLUSTER_GPU_SWITCH_NUM", "int", "Cluster_DFT_Col.c", "default 400: GPU/CPU crossover dimension of the collinear cluster dense path (the GPU won at every measured size down to n=240)"},
    {"OPENMX_CLUSTER_GS2_FALLBACK", "string", "openmx_cluster_capacity.c", "default auto: Capacity fallback of gpusolver in the cluster solvers when the dense solve does not fit on one GPU: MAGMA on the GPUs of rank 0's node first, then..."},
    {"OPENMX_CLUSTER_HOST_MEM_AVAILABLE_MB", "mib", "openmx_cluster_capacity.c", ""},
    {"OPENMX_CLUSTER_NONCOL_GPU_SWITCH_NUM", "int", "Cluster_DFT_NonCol.c", "default 400: Same for the noncollinear cluster dense path, compared against 2n (the GPU won down to 2n=480)"},
    {"OPENMX_CLUSTER_PROFILE", "flag", "Cluster_DFT_Col.c,Cluster_DFT_NonCol.c", ""},
    {"OPENMX_COPY_TO_WORK", "tool", "build/tools", "tools/copy_openmx.py"},
    {"OPENMX_CORE_LIBRARY", "tool", "build/tools", "tests and CMake: path of libopenmx_core.so to load"},
    {"OPENMX_CORE_VERBOSE", "tool", "build/tools", "C++ core: report once why a transform ran on the host (external Kokkos owner, device change, FP64)"},
    {"OPENMX_CUBLAS_MIN_FREE_AFTER_MB", "mib", "Divide_Conquer.c", "default —: Same for cuBLAS"},
    {"OPENMX_CUSOLVER_ALGORITHM", "string", "openmx_cusolver_compat.c", ""},
    {"OPENMX_CUSOLVER_CACHE_MB", "mib", "gpusolver_Syevdx.c", ""},
    {"OPENMX_CUSOLVER_EMULATION", "flag", "openmx_cusolver_compat.c", "default 1: Every cuSOLVER handle the code creates runs its FP64 eigensolves in the fixed-point FP64 emulation of cuSOLVER 12.2+ (CUSOLVER_FP64_EMULATED_FIXEDP..."},
    {"OPENMX_CUSOLVER_RETRY_TEST", "flag", "openmx_cusolver_compat.c", ""},
    {"OPENMX_CUSOLVER_TWO_STAGE_MIN_N", "int", "openmx_cusolver_compat.c", ""},
    {"OPENMX_CUSOLVER_VERBOSE", "flag", "Cluster_DFT_Col.c,openmx_cusolver_compat.c", ""},
    {"OPENMX_DCLNO_GPU_THRESHOLD", "int", "Divide_Conquer_LNO.c", "default 800: Minimum local dimension for the GPU (collinear)"},
    {"OPENMX_DCLNO_GPU_TURN_RELEASE", "string", "Divide_Conquer_LNO.c", "default 1: End-of-phase release of solver buffers + GEMMul8 (inherits the BAND setting when unset)"},
    {"OPENMX_DCLNO_NONCOL_GPU_THRESHOLD", "int", "Divide_Conquer_LNO.c", "default 400: Minimum spinor dimension for the GPU (noncollinear)"},
    {"OPENMX_DCLNO_PROFILE", "flag", "Divide_Conquer_LNO.c", ""},
    {"OPENMX_DC_GPU_S_CACHE", "flag", "Divide_Conquer.c", ""},
    {"OPENMX_DC_GPU_S_CACHE_RESERVE_MB", "mib", "Divide_Conquer.c", ""},
    {"OPENMX_DC_GPU_THRESHOLD", "int", "Divide_Conquer.c", "default 400: Minimum local cluster dimension NUM for the GPU (collinear)"},
    {"OPENMX_DC_NONCOL_GPU_THRESHOLD", "int", "Divide_Conquer.c", "default 300: Minimum spinor dimension 2·NUM for the GPU (noncollinear)"},
    {"OPENMX_DENSITY_GRID_COL_SIMD", "flag", "openmx_grid_assembly.c", ""},
    {"OPENMX_DENSITY_GRID_COL_SIMD_TRACE", "flag", "openmx_grid_assembly.c", ""},
    {"OPENMX_DENSITY_GRID_GPU", "flag", "Set_Density_Grid_GPU.c", ""},
    {"OPENMX_DENSITY_GRID_GPU_CACHE_MB", "mib", "Set_Density_Grid_GPU.c", ""},
    {"OPENMX_DENSITY_GRID_GPU_LOCAL", "string", "Set_Density_Grid_GPU.c", "default 1: Distributed quadrature mode: 0=off, 1=only when the shared tables are resident, 2=always"},
    {"OPENMX_DENSITY_GRID_GPU_LOCAL_CACHE_MB", "mib", "Set_Density_Grid_GPU.c", ""},
    {"OPENMX_DENSITY_GRID_GPU_PERSISTENT", "flag", "Set_Density_Grid_GPU.c", ""},
    {"OPENMX_DENSITY_GRID_GPU_RESERVE_MB", "mib", "Set_Density_Grid_GPU.c", ""},
    {"OPENMX_DENSITY_GRID_NC_SIMD", "flag", "openmx_grid_assembly.c", ""},
    {"OPENMX_DENSITY_GRID_NC_SIMD_TRACE", "flag", "openmx_grid_assembly.c", ""},
    {"OPENMX_EIGEN_FP32_TWO_STAGE", "flag", "eigen_refine_gpu.c", "default 1: The full FP32 solve of the refined eigensolver uses the two-stage cusolverDnXsyevd (CUSOLVER_ALG_2; cuSOLVER 12.3.4 and later; RTX 5080, n = 5616:..."},
    {"OPENMX_EIGEN_REFINE", "flag", "eigen_refine_gpu.c", "Cluster dense paths (collinear, per spin; noncollinear) and the band solvers (one k-point per rank: the estimates of all k-points are gathered befo..."},
    {"OPENMX_EIGEN_REFINE_ALL", "flag", "eigen_refine_gpu.c", ""},
    {"OPENMX_EIGEN_REFINE_CHECK", "flag", "eigen_refine_gpu.c", ""},
    {"OPENMX_EIGEN_REFINE_COMPLEMENT", "flag", "eigen_refine_gpu.c", ""},
    {"OPENMX_EIGEN_REFINE_STOP_GUARD", "int", "DFT.c", "default 100: An SCF stop on a refined FP32 step also needs the printed NormRD below this factor times scf.criterion; OpenMX's test on dUele alone fired on a coi..."},
    {"OPENMX_EIGEN_REFINE_UNBLOCKED", "flag", "eigen_refine_gpu.c", "default 1: The refinement products with the full GEMMul8 workspace when the device has room for it (about 1.7x faster than the blocked products at the 256 MiB..."},
    {"OPENMX_EIGEN_REFINE_UNTIL", "int", "eigen_refine_gpu.c", "default 0 (off): Printed NormRD below which the refined eigensolver hands the rest of the SCF cycle to the FP64 eigensolver. Off by default: an SCF that ends on a r..."},
    {"OPENMX_EIGEN_REFINE_WARM", "flag", "eigen_refine_gpu.c", "default 1: Warm start of the refined eigensolver: the vectors of the previous step's refined solve are refined against the new Hamiltonian instead of solving..."},
    {"OPENMX_EIGEN_REFINE_WARM_BAND", "string", "eigen_refine_gpu.c", ""},
    {"OPENMX_EIGEN_REFINE_WARM_DELTA_FLOOR", "int", "eigen_refine_gpu.c", ""},
    {"OPENMX_EIGEN_REFINE_WARM_PERIOD", "int", "eigen_refine_gpu.c", ""},
    {"OPENMX_FORCE3_GPU", "flag", "Force.c", ""},
    {"OPENMX_FORCE3_GPU_DORB", "flag", "Force.c", ""},
    {"OPENMX_FORCE4B_ARCHIVE_GPU", "flag", "Force.c", ""},
    {"OPENMX_FORCE4B_CASE1_STREAM", "flag", "Force.c", ""},
    {"OPENMX_FORCE4B_CASE1_STREAM_MB", "mib", "Force.c", ""},
    {"OPENMX_FORCE4B_CASE1_TURNS", "flag", "Force.c", ""},
    {"OPENMX_FORCE4B_CASE1_TURN_MAX_RANKS", "int", "Force.c", ""},
    {"OPENMX_FORCE4B_CASE2_STREAM", "flag", "Force.c", ""},
    {"OPENMX_FORCE4B_FUSED", "flag", "Force.c", ""},
    {"OPENMX_FORCE4B_GPU", "flag", "Force.c", ""},
    {"OPENMX_FORCEHNL_GPU", "flag", "openmx_force_hnl_gpu.h", "default 1: Force_HNL GPU batch"},
    {"OPENMX_FORCE_GPU", "flag", "Force.c", "default 1: Parent gate for all force GPU paths (collective)"},
    {"OPENMX_FORCE_PROFILE", "flag", "Force.c", "default 0: [Force profile] lines (per-stage max/avg seconds)"},
    {"OPENMX_FORCE_RELEASE_SCF_CACHES", "string", "Force.c", ""},
    {"OPENMX_GEMMUL8_A100_FORWARD_MIN_N", "int", "gemmul8_bridge.cu", ""},
    {"OPENMX_GEMMUL8_DISABLE", "flag", "gemmul8_bridge.cu", ""},
    {"OPENMX_GEMMUL8_DISABLE_D", "flag", "gemmul8_bridge.cu", ""},
    {"OPENMX_GEMMUL8_DISABLE_Z", "flag", "gemmul8_bridge.cu", ""},
    {"OPENMX_GEMMUL8_FASTMODE_D", "flag", "gemmul8_bridge.cu", ""},
    {"OPENMX_GEMMUL8_FASTMODE_Z", "flag", "gemmul8_bridge.cu", ""},
    {"OPENMX_GEMMUL8_FORWARD", "string", "gemmul8_bridge.cu", ""},
    {"OPENMX_GEMMUL8_FORWARD_FASTMODE", "int", "gemmul8_bridge.cu", ""},
    {"OPENMX_GEMMUL8_FORWARD_NUM_MOD", "int", "gemmul8_bridge.cu", ""},
    {"OPENMX_GEMMUL8_FORWARD_REUSE", "flag", "gemmul8_bridge.cu", ""},
    {"OPENMX_GEMMUL8_FORWARD_TIMING", "flag", "gemmul8_bridge.cu", ""},
    {"OPENMX_GEMMUL8_FORWARD_UNBLOCKED", "flag", "gemmul8_bridge.cu", ""},
    {"OPENMX_GEMMUL8_LOCAL_RANKS", "int", "gemmul8_bridge.cu", ""},
    {"OPENMX_GEMMUL8_MAX_WORKSPACE_MB", "mib", "gemmul8_bridge.cu", ""},
    {"OPENMX_GEMMUL8_MAX_WORKSPACE_PERCENT", "int", "Band_DFT_NonCol.c,gemmul8_bridge.cu", ""},
    {"OPENMX_GEMMUL8_MIN_FREE_AFTER_MB", "mib", "Divide_Conquer.c,gemmul8_bridge.cu", ""},
    {"OPENMX_GEMMUL8_NUM_MOD_D", "flag", "gemmul8_bridge.cu", ""},
    {"OPENMX_GEMMUL8_NUM_MOD_Z", "flag", "gemmul8_bridge.cu", ""},
    {"OPENMX_GEMMUL8_TIMING", "flag", "gemmul8_bridge.cu", ""},
    {"OPENMX_GEMMUL8_VERBOSE", "flag", "gemmul8_bridge.cu", ""},
    {"OPENMX_GEMM_SAMPLE_DIR", "string", "Cluster_DFT_Col.c,Cluster_DFT_NonCol.c", ""},
    {"OPENMX_GEMM_SAMPLE_ITERS", "int", "Cluster_DFT_Col.c,Cluster_DFT_NonCol.c", ""},
    {"OPENMX_GEMM_VERBOSE", "flag", "gemmul8_bridge.cu", ""},
    {"OPENMX_GPU", "flag", "Input_std.c,openmx_gpu_workers.c,openmx_mgpu_runtime.cu,set_cuda_default_device_from_local_rank.c", ""},
    {"OPENMX_GPUSOLVER_MIN_FREE_AFTER_MB", "mib", "Divide_Conquer.c", ""},
    {"OPENMX_GPUSOLVER_SERIAL_GPU_TURNS", "flag", "Band_DFT_Col.c,Band_DFT_NonCol.c", "default 1: GPU-turn serialization of the all_knum==1 path (0 = concurrent submission)"},
    {"OPENMX_GPU_ARCH", "tool", "build/tools", "CMake: GPU architecture of the build"},
    {"OPENMX_GPU_BACKEND", "tool", "build/tools", "native layer / CMake: CUDA or HIP (a build-time selection, not read at run time)"},
    {"OPENMX_GPU_CAPACITY_TEST_RECOVERY", "flag", "openmx_gpu_workers.h", ""},
    {"OPENMX_GPU_HEAP_REEXEC", "string", "openmx.c", ""},
    {"OPENMX_GPU_LOCK_DIR", "string", "openmx_mgpu_runtime.cu", ""},
    {"OPENMX_GPU_LOCK_TIMEOUT_MS", "string", "openmx_mgpu_runtime.cu", ""},
    {"OPENMX_GPU_MEMINFO_NVML", "flag", "set_cuda_default_device_from_local_rank.c", "default 1: Every device-memory budget of the GPU paths starts from OpenMX_GpuMemGetInfo(): cudaMemGetInfo() with the free figure capped at the memory that is..."},
    {"OPENMX_GPU_PROBE_RESERVE_MB", "mib", "set_cuda_default_device_from_local_rank.c", ""},
    {"OPENMX_GPU_RANK_MAP", "int", "set_cuda_default_device_from_local_rank.c", ""},
    {"OPENMX_GPU_STACK_BYTES", "int", "set_cuda_default_device_from_local_rank.c", "Per-thread kernel stack limit set at device initialization (cudaLimitStackSize, OpenMX_GpuPresizeKernelStack) when it exceeds the driver's default..."},
    {"OPENMX_GPU_VERBOSE", "flag", "gemmul8_bridge.cu", ""},
    {"OPENMX_GPU_WORKERS", "flag", "openmx_gpu_workers.c,openmx_gpu_workers.h", ""},
    {"OPENMX_GPU_WORKER_COPY", "flag", "openmx_gpu_views.c", ""},
    {"OPENMX_GPU_WORKER_OVERLAP", "flag", "openmx_gpu_workers.c", ""},
    {"OPENMX_GRID_FP32", "flag", "grid_precision.c", "default 0: Experimental: plain FP32 kernels while the previous step's NormRD is above OPENMX_GRID_FP32_UNTIL (1e-5), FP64 afterwards (OPENMX_GRID_FP32_RESTART..."},
    {"OPENMX_GRID_FP32_RESTART", "flag", "grid_precision.c", ""},
    {"OPENMX_GRID_FP32_UNTIL", "int", "grid_precision.c", ""},
    {"OPENMX_GRID_PRECISION", "string", "grid_precision.c", "default df / fp64: The arithmetic of the grid-integral kernels (the dVH+Vxc+VNA matrix elements and the density): df = double-float, FP32 arithmetic with error-free t..."},
    {"OPENMX_GS2_ELPA_GPU", "flag", "elpa_cosma_bridge.c", ""},
    {"OPENMX_GS2_ELPA_SOLVER", "string", "elpa_cosma_bridge.c", "default 2: ELPA solver used by gpusolver2: 1 = 1stage, anything else = 2stage (default)"},
    {"OPENMX_GS2_GEMM_GPU", "flag", "elpa_cosma_bridge.c", ""},
    {"OPENMX_GS2_GPU_FACTOR", "int", "elpa_cosma_bridge.c", ""},
    {"OPENMX_GS2_GPU_FIXED_MB", "mib", "elpa_cosma_bridge.c", ""},
    {"OPENMX_GS2_GPU_RESERVE_MB", "mib", "elpa_cosma_bridge.c", ""},
    {"OPENMX_GS2_GPU_TILE", "int", "elpa_cosma_bridge.c", ""},
    {"OPENMX_GS2_GPU_VERBOSE", "flag", "elpa_cosma_bridge.c", ""},
    {"OPENMX_GS2_HOST_RESERVE_MB", "mib", "elpa_cosma_bridge.c", "gpusolver2 bridge (explicit or as the capacity fallback): host memory that must stay free on every node after COSMA allocates the buffers it keeps..."},
    {"OPENMX_GS2_PROFILE", "flag", "elpa_cosma_bridge.c", ""},
    {"OPENMX_KRYLOV_GPU_DGEMM_MIN_FLOPS", "string", "Krylov.c", "default 5×10⁷: Minimum m·n·k for dispatching a GEMM to the GPU"},
    {"OPENMX_KRYLOV_GPU_EIGEN_MIN", "string", "Krylov.c", "default 300: Minimum projected dimension Msize3 for the device eigensolve (§7.6)"},
    {"OPENMX_KRYLOV_GPU_FUSED", "flag", "Krylov.c", "default 1: Fused per-atom pipeline (0 = per-call sequence)"},
    {"OPENMX_KRYLOV_GPU_KU_CACHE", "flag", "Krylov.c", ""},
    {"OPENMX_KRYLOV_GPU_KU_RESERVE_MB", "mib", "Krylov.c", ""},
    {"OPENMX_KRYLOV_GPU_PROFILE", "flag", "Krylov.c", "default 0: Fused-path breakdown (KRYFUSE lines)"},
    {"OPENMX_MAGMA_FAILURE_DUMP", "string", "openmx_magma_mgpu.cpp", ""},
    {"OPENMX_MAGMA_FP32", "flag", "openmx_magma_mgpu.cpp", ""},
    {"OPENMX_MAGMA_FP32_COMPLEX_MAX_N", "int", "openmx_magma_mgpu.cpp", ""},
    {"OPENMX_MAGMA_HOST_CACHE_MB", "mib", "openmx_magma_mgpu.cpp", ""},
    {"OPENMX_MAGMA_REFINE_RESERVE_MB", "mib", "openmx_mgpu_gemm.cu", ""},
    {"OPENMX_MAGMA_REFINE_RESIDENT_MB", "mib", "openmx_mgpu_gemm.cu", ""},
    {"OPENMX_MAGMA_REFINE_STEPS", "string", "openmx_magma_refine.h", ""},
    {"OPENMX_MAGMA_RR_GPU_MIN", "string", "openmx_magma_refine.h", ""},
    {"OPENMX_MAGMA_VALIDATE_FULL", "string", "openmx_magma_mgpu.cpp", ""},
    {"OPENMX_MAKE_OPTIONS", "tool", "build/tools", "CMake: native make variable assignments"},
    {"OPENMX_MGPU", "string", "openmx_mgpu_runtime.cu", ""},
    {"OPENMX_MGPU_BAND_SHARED", "flag", "openmx_mgpu_runtime.cu", ""},
    {"OPENMX_MGPU_GEMM", "flag", "openmx_mgpu_runtime.cu", ""},
    {"OPENMX_MGPU_GEMM_ASYNC_GATHER", "flag", "openmx_mgpu_gemm.cu", ""},
    {"OPENMX_MGPU_GEMM_BACKEND", "string", "openmx_mgpu_gemm.cu", ""},
    {"OPENMX_MGPU_GEMM_FORCE_HOST", "string", "openmx_mgpu_gemm.cu", ""},
    {"OPENMX_MGPU_GEMM_HOST_MB", "mib", "openmx_mgpu_gemm.cu", ""},
    {"OPENMX_MGPU_GEMM_HOST_RESULT_MB", "mib", "openmx_mgpu_gemm.cu", ""},
    {"OPENMX_MGPU_GEMM_MIN_N", "int", "openmx_mgpu_runtime.cu", ""},
    {"OPENMX_MGPU_GEMM_PAGEABLE", "string", "openmx_mgpu_gemm.cu", ""},
    {"OPENMX_MGPU_GEMM_RESERVE_MB", "mib", "openmx_mgpu_gemm.cu", ""},
    {"OPENMX_MGPU_GEMM_RESULT_HOST", "string", "openmx_mgpu_gemm.cu", ""},
    {"OPENMX_MGPU_GEMM_RESULT_MB", "mib", "openmx_mgpu_gemm.cu", ""},
    {"OPENMX_MGPU_GEMM_TRAVERSAL", "string", "openmx_mgpu_gemm.cu", ""},
    {"OPENMX_MGPU_GEMM_WORKSPACE_MB", "mib", "openmx_mgpu_gemm.cu", ""},
    {"OPENMX_MGPU_MEMORY", "string", "openmx_mgpu_runtime.cu", ""},
    {"OPENMX_MGPU_NUM_GPUS", "int", "openmx_mgpu_runtime.cu", ""},
    {"OPENMX_MGPU_SOLVER", "string", "openmx_mgpu_runtime.cu", ""},
    {"OPENMX_MGPU_SOLVER_MIN_N", "int", "openmx_mgpu_runtime.cu", ""},
    {"OPENMX_MGPU_TUNE_TRIALS", "string", "openmx_mgpu_runtime.cu", ""},
    {"OPENMX_MGPU_VERBOSE", "flag", "openmx_gpu_workers.c,openmx_mgpu_runtime.cu", ""},
    {"OPENMX_MIXH_GPU", "flag", "Mixing_H.c", "default auto: RMM-DIISH tier selection: unset=auto, 0=legacy, 1=force GPU tier, 2=force incremental CPU tier"},
    {"OPENMX_MIXH_GPU_RESERVE_MB", "mib", "Mixing_H.c", ""},
    {"OPENMX_MIXH_GPU_VERBOSE", "flag", "Mixing_H.c", ""},
    {"OPENMX_MIX_FAST", "string", "DIIS_Mixing_DM.c,DIIS_Mixing_Rhok.c,GR_Pulay_DM.c,Kerker_Mixing_Rhok.c,Mixing_V.c", "default 1: CPU fast paths of RMM-DIIS/DIISK/DIISV/GR-Pulay/Kerker (0 = legacy)"},
    {"OPENMX_OLPKIN_BESSEL_CACHE_MAX", "int", "Set_OLP_Kin.c", ""},
    {"OPENMX_OLPKIN_RADIAL_GPU", "flag", "Set_OLP_Kin.c", "default 0: GPU radial integrals in Set_OLP_Kin (single thread required; usually slower — transfer-bound)"},
    {"OPENMX_ORBS_EVAL_PRECISION", "string", "Set_Orbitals_Grid.c", "default df / fp64: The arithmetic of the on-the-fly orbital evaluation (the device code of Set_Orbitals_Grid that the Hamiltonian and density tiles use when a rank ha..."},
    {"OPENMX_ORBS_GRID_GPU", "flag", "Set_Orbitals_Grid.c", ""},
    {"OPENMX_ORBS_GRID_GPU_VERBOSE", "flag", "Set_Orbitals_Grid.c", ""},
    {"OPENMX_SETDENSITY_GPU", "flag", "Set_Density_Grid_GPU.c", ""},
    {"OPENMX_SETDEN_FP64_CUDA", "string", "set_hamiltonian_gpu.cu", ""},
    {"OPENMX_SETDEN_FP64_SPLIT_SPIN", "flag", "set_hamiltonian_gpu.cu", ""},
    {"OPENMX_SETDEN_ONTHEFLY_MB", "mib", "Set_Hamiltonian.c", "default 128: Batch workspace bound (MiB) for on-the-fly density. The physical budget subtracts a 512 MiB node reserve, divides among on-the-fly ranks, then subt..."},
    {"OPENMX_SETHAM_BASE_GPU", "flag", "Set_Hamiltonian.c", "default 0: GPU execution of the base combination of H (CPU is faster in practice)"},
    {"OPENMX_SETHAM_CUDA_KERNEL", "flag", "Set_Hamiltonian.c", "default 1: The dedicated CUDA kernel (0 = OpenACC kernel only; numerically identical)"},
    {"OPENMX_SETHAM_FP64_SPLIT", "string", "set_hamiltonian_gpu.cu", ""},
    {"OPENMX_SETHAM_FP64_TILED", "string", "set_hamiltonian_gpu.cu", ""},
    {"OPENMX_SETHAM_GPU_KOWNER_ONLY", "string", "DFT.c", "default 0: 1 restores the old k-owner-only GPU execution"},
    {"OPENMX_SETHAM_GPU_MAX_RANKS_PER_DEVICE", "int", "Set_Hamiltonian.c", "default uncapped: Explicit cap on ranks per wave"},
    {"OPENMX_SETHAM_GPU_RANK_OVERHEAD_MB", "mib", "Set_Hamiltonian.c", ""},
    {"OPENMX_SETHAM_GPU_RESERVE_MB", "mib", "Set_Hamiltonian.c", ""},
    {"OPENMX_SETHAM_GPU_RESIDENT", "flag", "Set_Hamiltonian.c", ""},
    {"OPENMX_SETHAM_GPU_RESIDENT_FLOOR_MB", "mib", "Set_Hamiltonian.c", ""},
    {"OPENMX_SETHAM_GPU_SERIAL_WAVES", "string", "Set_Hamiltonian.c", "default 0: 1 = run all waves on the GPU serially (default: wave 1 on GPU with remaining ranks on CPU concurrently) Under memory pressure on the A100 (whole-ra..."},
    {"OPENMX_SETHAM_GPU_TEST_NEED_FROM_CALL", "string", "Set_Hamiltonian.c", ""},
    {"OPENMX_SETHAM_GPU_TEST_NEED_MB", "mib", "Set_Hamiltonian.c", ""},
    {"OPENMX_SETHAM_MATEL_SPLIT", "string", "set_hamiltonian_gpu.cu", "default 4: Blocks per atom pair along the grid points in the df matrix-element kernel (each block integrates whole 32-point tiles of its share and adds its pa..."},
    {"OPENMX_SETHAM_MATRIX_GPU", "flag", "Set_Hamiltonian.c", "default 1: GPU execution of the dVH+Vxc+VNA grid integrals (0 = CPU)"},
    {"OPENMX_SETHAM_ONTHEFLY", "string", "Set_Hamiltonian.c", "default 1: A rank whose orbital tables do not fit on the device evaluates the orbitals of every batch of atom pairs on the device (the device code of Set_Orbi..."},
    {"OPENMX_SETHAM_ONTHEFLY_MARGIN_MB", "mib", "Set_Hamiltonian.c", ""},
    {"OPENMX_SETHAM_ONTHEFLY_MB", "mib", "Set_Hamiltonian.c", ""},
    {"OPENMX_SETHAM_ONTHEFLY_PLAN_CACHE", "flag", "Set_Hamiltonian.c", "default 1: The on-the-fly paths keep a plan per rank: the batches (which pairs, their offsets), the pair descriptors and the per-point index lists (Nc, and th..."},
    {"OPENMX_SETHAM_SIMD", "flag", "openmx_grid_assembly.c", ""},
    {"OPENMX_SETHAM_SIMD_TRACE", "flag", "openmx_grid_assembly.c", ""},
    {"OPENMX_SETHAM_STREAMING", "flag", "Set_Hamiltonian.c", "default 1: Streamed pair batches (256 MiB workspace per rank, OPENMX_SETHAM_STREAM_MB) when the device holds a single whole-rank allocation; 2 streams wheneve..."},
    {"OPENMX_SETHAM_STREAM_MB", "mib", "Set_Hamiltonian.c", ""},
    {"OPENMX_SETNL_BESSEL_CACHE_MAX", "int", "Set_Nonlocal.c", ""},
    {"OPENMX_SETNL_OPENACC", "string", "Set_Nonlocal.c", "default 1: The Set_Nonlocal GPU batch (0 = CPU path)"},
    {"OPENMX_SETPRO_GPU", "flag", "DFT.c,Set_ProExpn_VNA.c", "default 1: The three Set_ProExpn_VNA batches (0 = CPU)"},
    {"OPENMX_SETPRO_VERBOSE", "flag", "Set_ProExpn_VNA.c", "default 0: One stderr line per rank and device stage of Set_ProExpn_VNA: the admission figures (need, budget, free memory, pair batch, rank turns), the batch..."},
    {"OPENMX_SOURCE_ROOT", "tool", "build/tools", "HIP Makefile: repository root of the staged build"},
    {"OPENMX_TOTALENERGY_GPU", "flag", "Total_Energy.c,Total_Energy_GPU.c", "default 1: Master gate for the total-energy device kernels (including NC)"},
    {"OPENMX_UCELL_OLG_GPU", "flag", "truncation.c", ""},
    {"OPENMX_UCELL_OLG_GPU_VERBOSE", "flag", "truncation.c", ""},
    {"OPENMX_XC_GRID_GPU", "flag", "Set_XC_Grid.c", "default 1: The GGA-PBE exchange-correlation potential on the grid (Set_XC_Grid, called by Set_Vpot every SCF step and by Total_Energy / Force) is evaluated on..."},
    {"OPENMX_ZHEEVD_SELFTEST", "string", "openmx.c", ""},
};
#define REGISTRY_COUNT (sizeof registry / sizeof registry[0])

/* --- the snapshot --------------------------------------------------------- */
typedef struct {
    char *name;
    char *value;   /* NULL: unset at the snapshot */
} Entry;

static struct {
    Entry *table;             /* sorted by name once openmx_env_init() is done */
    size_t count, capacity;
    char **unknown;           /* OPENMX_* names of the environment that the registry lacks */
    size_t unknown_count, unknown_capacity;
    int initialized;
} state;

static char *dup_string(const char *s)
{
    size_t n;
    char *copy;

    if (s == NULL) return NULL;
    n = strlen(s) + 1;
    copy = (char *)malloc(n);
    if (copy != NULL) memcpy(copy, s, n);
    return copy;
}

static int entry_compare(const void *a, const void *b)
{
    return strcmp(((const Entry *)a)->name, ((const Entry *)b)->name);
}

static Entry *find_entry(const char *name)
{
    Entry key;

    if (state.count == 0) return NULL;
    key.name = (char *)name;
    key.value = NULL;
    return (Entry *)bsearch(&key, state.table, state.count, sizeof *state.table, entry_compare);
}

/* append (unsorted); the caller sorts when it is done */
static int append_entry(const char *name, const char *value)
{
    Entry *entry;

    if (state.count == state.capacity) {
        size_t capacity = state.capacity ? 2 * state.capacity : 256;
        Entry *grown = (Entry *)realloc(state.table, capacity * sizeof *grown);

        if (grown == NULL) return -1;
        state.table = grown;
        state.capacity = capacity;
    }
    entry = &state.table[state.count];
    entry->name = dup_string(name);
    entry->value = dup_string(value);
    if (entry->name == NULL || (value != NULL && entry->value == NULL)) {
        free(entry->name);
        free(entry->value);
        return -1;
    }
    state.count++;
    return 0;
}

static int remember_unknown(const char *name)
{
    char *copy;

    if (state.unknown_count == state.unknown_capacity) {
        size_t capacity = state.unknown_capacity ? 2 * state.unknown_capacity : 8;
        char **grown = (char **)realloc(state.unknown, capacity * sizeof *grown);

        if (grown == NULL) return -1;
        state.unknown = grown;
        state.unknown_capacity = capacity;
    }
    copy = dup_string(name);
    if (copy == NULL) return -1;
    state.unknown[state.unknown_count++] = copy;
    return 0;
}

static int registry_compare(const void *a, const void *b)
{
    return strcmp((const char *)a, ((const OpenMXEnvEntry *)b)->name);
}

const OpenMXEnvEntry *openmx_env_lookup(const char *name)
{
    if (name == NULL) return NULL;
    return (const OpenMXEnvEntry *)bsearch(name, registry, REGISTRY_COUNT, sizeof registry[0], registry_compare);
}

size_t openmx_env_registry(const OpenMXEnvEntry **entries)
{
    if (entries != NULL) *entries = registry;
    return REGISTRY_COUNT;
}

void openmx_env_init(void)
{
    size_t i;
    char **e;

    if (state.initialized) return;
    for (i = 0; i < REGISTRY_COUNT; i++) {
        (void)append_entry(registry[i].name, getenv(registry[i].name));
    }
    /* every OPENMX_* present in the environment: snapshotted too, and
       reported when the registry does not know it (a misspelt setting) */
    for (e = environ; e != NULL && *e != NULL; e++) {
        const char *equals = strchr(*e, '=');
        size_t n;
        char name[256];

        if (strncmp(*e, "OPENMX_", 7) != 0 || equals == NULL) continue;
        n = (size_t)(equals - *e);
        if (n == 0 || n >= sizeof name) continue;
        memcpy(name, *e, n);
        name[n] = '\0';
        if (openmx_env_lookup(name) != NULL) continue;
        (void)remember_unknown(name);
        (void)append_entry(name, equals + 1);
    }
    qsort(state.table, state.count, sizeof *state.table, entry_compare);
    state.initialized = 1;
}

int openmx_env_initialized(void)
{
    return state.initialized;
}

void openmx_env_reset(void)
{
    size_t i;

    for (i = 0; i < state.count; i++) {
        free(state.table[i].name);
        free(state.table[i].value);
    }
    free(state.table);
    for (i = 0; i < state.unknown_count; i++) free(state.unknown[i]);
    free(state.unknown);
    memset(&state, 0, sizeof state);
}

char *openmx_env_get(const char *name)
{
    Entry *entry;

    if (name == NULL) return NULL;
    if (!state.initialized) return getenv(name);
    entry = find_entry(name);
    return entry != NULL ? entry->value : getenv(name);
}

static int ascii_equal_ignoring_case(const char *a, const char *b)
{
    for (; *a != '\0' && *b != '\0'; a++, b++) {
        int ca = *a, cb = *b;

        if ('A' <= ca && ca <= 'Z') ca += 'a' - 'A';
        if ('A' <= cb && cb <= 'Z') cb += 'a' - 'A';
        if (ca != cb) return 0;
    }
    return *a == '\0' && *b == '\0';
}

int openmx_env_flag(const char *name, int fallback)
{
    const char *value = openmx_env_get(name);

    if (value == NULL || value[0] == '\0') return fallback;
    if (ascii_equal_ignoring_case(value, "0") || ascii_equal_ignoring_case(value, "off") ||
        ascii_equal_ignoring_case(value, "no") || ascii_equal_ignoring_case(value, "false")) return 0;
    return 1;
}

long openmx_env_long(const char *name, long fallback)
{
    const char *value = openmx_env_get(name);
    char *end;
    long parsed;

    if (value == NULL || value[0] == '\0') return fallback;
    errno = 0;
    parsed = strtol(value, &end, 10);
    if (end == value || *end != '\0' || errno == ERANGE) return fallback;
    return parsed;
}

double openmx_env_double(const char *name, double fallback)
{
    const char *value = openmx_env_get(name);
    char *end;
    double parsed;

    if (value == NULL || value[0] == '\0') return fallback;
    errno = 0;
    parsed = strtod(value, &end);
    if (end == value || *end != '\0' || errno == ERANGE) return fallback;
    return parsed;
}

unsigned long long openmx_env_mib(const char *name, unsigned long long fallback)
{
    const char *value = openmx_env_get(name);
    char *end;
    unsigned long long parsed;

    if (value == NULL || value[0] == '\0' || value[0] == '-') return fallback;
    errno = 0;
    parsed = strtoull(value, &end, 10);
    if (end == value || *end != '\0' || errno == ERANGE) return fallback;
    return parsed;
}

int openmx_env_set(const char *name, const char *value)
{
    Entry *entry;

    if (name == NULL || value == NULL) return -1;
    if (setenv(name, value, 1) != 0) return -1;
    if (!state.initialized) return 0;
    entry = find_entry(name);
    if (entry != NULL) {
        char *copy = dup_string(value);

        if (copy == NULL) return -1;
        free(entry->value);
        entry->value = copy;
        return 0;
    }
    if (append_entry(name, value) != 0) return -1;
    qsort(state.table, state.count, sizeof *state.table, entry_compare);
    return 0;
}

int openmx_env_set_if_unset(const char *name, const char *value)
{
    if (openmx_env_get(name) != NULL) return 0;
    return openmx_env_set(name, value);
}

size_t openmx_env_unknown(const char *const **names)
{
    if (names != NULL) *names = (const char *const *)state.unknown;
    return state.unknown_count;
}

void openmx_env_report(FILE *out, int unknown_only)
{
    size_t i;
    int printed = 0;

    if (out == NULL) return;
    if (unknown_only) {
        for (i = 0; i < state.unknown_count; i++) {
            fprintf(out, "<OpenMX> environment: %s is set, but no setting of this build has that name"
                         " (check the spelling; section 10.2 of doc/OpenMX40_GPU_implementation_en.html lists them).\n",
                    state.unknown[i]);
        }
        fflush(out);
        return;
    }
    for (i = 0; i < REGISTRY_COUNT; i++) {
        const char *value = openmx_env_get(registry[i].name);

        if (value == NULL) continue;
        if (!printed) {
            fprintf(out, "<OpenMX> environment settings in effect:\n");
            printed = 1;
        }
        fprintf(out, "  %s=%s\n", registry[i].name, value);
    }
    if (printed) fflush(out);
}

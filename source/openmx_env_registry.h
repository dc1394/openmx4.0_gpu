/**********************************************************************
  openmx_env_registry.h

  The registry of the environment settings this build reads: one X() line
  per name, sorted by name.  openmx_env.h turns the first column into the
  ids OPENMX_ENV_<id> (an EXT_ id keeps its case: ELPA's ELPA_DEFAULT_real_kernel
  is read as OPENMX_ENV_EXT_ELPA_DEFAULT_real_kernel), openmx_env.c into the table behind openmx_env_at()
  and friends.  tools/update_env_registry.py rewrites this file from the
  readers in source/ (adds names, derives the kind and the default from
  the accessors, keeps the descriptions); tools/env_registry_doc.py turns
  it, with doc/environment_variables.json, into the tables of the
  implementation document; tools/audit_global_usage.py fails its gate on
  a name read by the code that is missing here.

  Columns: X(id, name, kind, scope, default, readers, description)
    kind     flag | int | uint | mib | double | string | tool
             (what the readers parse; "tool" names are read by the build
             or the tools only and never by openmx itself)
    scope    global: one value for the run, taken from rank 0 after
                     MPI_Init (openmx_env_pack/unpack)
             local:  per process (launcher-provided ranks, host name,
                     re-exec markers, MPS paths); never synchronized
             tool:   see kind tool
    default  the readers' fallback when it is one literal ("auto" for -1,
             "unlimited" for INT_MAX); "" when it is computed or differs
             between readers (the document then describes it)
***********************************************************************/
#ifndef OPENMX_ENV_REGISTRY_H
#define OPENMX_ENV_REGISTRY_H

#define OPENMX_ENV_LIST(X) \
    X(EXT_COSMA_CPU_MAX_MEMORY, "COSMA_CPU_MAX_MEMORY", "mib", "global", "unset (COSMA's default)", "elpa_cosma_memory.cpp", "COSMA: host memory cap in MiB, read for the CPU fallback planner") \
    X(EXT_COSMA_GPU_MAX_TILE_K, "COSMA_GPU_MAX_TILE_K", "int", "global", "5000", "elpa_cosma_bridge.c", "COSMA: Tiled-MM tile size, read to size the GPU GEMM workspace") \
    X(EXT_COSMA_GPU_MAX_TILE_M, "COSMA_GPU_MAX_TILE_M", "int", "global", "5000", "elpa_cosma_bridge.c", "COSMA: Tiled-MM tile size, read to size the GPU GEMM workspace") \
    X(EXT_COSMA_GPU_MAX_TILE_N, "COSMA_GPU_MAX_TILE_N", "int", "global", "5000", "elpa_cosma_bridge.c", "COSMA: Tiled-MM tile size, read to size the GPU GEMM workspace") \
    X(EXT_COSMA_GPU_STREAMS, "COSMA_GPU_STREAMS", "int", "global", "2", "elpa_cosma_bridge.c", "COSMA: GPU streams, read to size the GPU GEMM workspace") \
    X(EXT_CUBLAS_MIN_FREE_AFTER_MB, "CUBLAS_MIN_FREE_AFTER_MB", "mib", "global", "", "Divide_Conquer.c,backends/hip/src/Divide_Conquer.c", "device memory to keep free after a cuBLAS workspace (OPENMX_CUBLAS_MIN_FREE_AFTER_MB wins)") \
    X(EXT_DC_GPU_MAX_RANKS_PER_DEVICE, "DC_GPU_MAX_RANKS_PER_DEVICE", "uint", "global", "", "backends/hip/src/Divide_Conquer.c", "Alias of OPENMX_DC_GPU_MAX_RANKS_PER_DEVICE.") \
    X(EXT_ELPA_DEFAULT_complex_kernel, "ELPA_DEFAULT_complex_kernel", "string", "global", "unset (OpenMX chooses)", "openmx_elpa_cpu.c", "") \
    X(EXT_ELPA_DEFAULT_real_kernel, "ELPA_DEFAULT_real_kernel", "string", "global", "unset (OpenMX chooses)", "openmx_elpa_cpu.c", "") \
    X(EXT_GEMMUL8_DISABLE, "GEMMUL8_DISABLE", "flag", "global", "", "backends/hip/src/gemmul8_bridge.cu,gemmul8_bridge.cu", "") \
    X(EXT_GEMMUL8_DISABLE_D, "GEMMUL8_DISABLE_D", "flag", "global", "", "backends/hip/src/gemmul8_bridge.cu,gemmul8_bridge.cu", "") \
    X(EXT_GEMMUL8_DISABLE_Z, "GEMMUL8_DISABLE_Z", "flag", "global", "", "backends/hip/src/gemmul8_bridge.cu,gemmul8_bridge.cu", "") \
    X(EXT_GEMMUL8_FASTMODE_D, "GEMMUL8_FASTMODE_D", "flag", "global", "0", "backends/hip/src/gemmul8_bridge.cu,gemmul8_bridge.cu", "") \
    X(EXT_GEMMUL8_FASTMODE_Z, "GEMMUL8_FASTMODE_Z", "flag", "global", "0", "backends/hip/src/gemmul8_bridge.cu,gemmul8_bridge.cu", "") \
    X(EXT_GEMMUL8_MAX_WORKSPACE_MB, "GEMMUL8_MAX_WORKSPACE_MB", "mib", "global", "256", "backends/hip/src/gemmul8_bridge.cu,gemmul8_bridge.cu", "") \
    X(EXT_GEMMUL8_MAX_WORKSPACE_PERCENT, "GEMMUL8_MAX_WORKSPACE_PERCENT", "uint", "global", "", "Band_DFT_NonCol.c,backends/hip/src/Band_DFT_NonCol.c,backends/hip/src/gemmul8_bridge.cu,gemmul8_bridge.cu", "") \
    X(EXT_GEMMUL8_MIN_FREE_AFTER_MB, "GEMMUL8_MIN_FREE_AFTER_MB", "mib", "global", "", "Divide_Conquer.c,backends/hip/src/Divide_Conquer.c,backends/hip/src/gemmul8_bridge.cu,gemmul8_bridge.cu", "") \
    X(EXT_GEMMUL8_NUM_MOD_D, "GEMMUL8_NUM_MOD_D", "uint", "global", "", "backends/hip/src/gemmul8_bridge.cu,gemmul8_bridge.cu", "") \
    X(EXT_GEMMUL8_NUM_MOD_Z, "GEMMUL8_NUM_MOD_Z", "uint", "global", "", "backends/hip/src/gemmul8_bridge.cu,gemmul8_bridge.cu", "") \
    X(EXT_GEMMUL8_SOLVE_MIN_FREE_AFTER_MB, "GEMMUL8_SOLVE_MIN_FREE_AFTER_MB", "mib", "global", "256", "backends/hip/src/gemmul8_bridge.cu", "Alias of OPENMX_GEMMUL8_SOLVE_MIN_FREE_AFTER_MB.") \
    X(EXT_GEMMUL8_VERBOSE, "GEMMUL8_VERBOSE", "flag", "global", "", "backends/hip/src/gemmul8_bridge.cu,gemmul8_bridge.cu", "") \
    X(EXT_GPUSOLVER_MIN_FREE_AFTER_MB, "GPUSOLVER_MIN_FREE_AFTER_MB", "mib", "global", "1536", "Divide_Conquer.c,backends/hip/src/Divide_Conquer.c", "device memory to keep free after a cuSOLVER workspace (OPENMX_GPUSOLVER_MIN_FREE_AFTER_MB wins)") \
    X(EXT_HOSTNAME, "HOSTNAME", "string", "local", "", "exx_log.c", "shell: host name for the EXX log") \
    X(EXT_LIBOMPTARGET_HEAP_SIZE, "LIBOMPTARGET_HEAP_SIZE", "string", "local", "", "openmx.c", "OpenMP offload runtime heap; openmx re-executes itself with it set when absent") \
    X(EXT_MAGMA_OZAKI2_ALGO, "MAGMA_OZAKI2_ALGO", "string", "global", "", "backends/hip/src/magma_openmx.cu", "MAGMA Ozaki-II algorithm; OpenMX supplies 1stage when unset.") \
    X(EXT_MAGMA_OZAKI2_LOCAL_RANKS, "MAGMA_OZAKI2_LOCAL_RANKS", "int", "global", "4", "backends/hip/src/Cluster_DFT_Col.c,backends/hip/src/magma_openmx.cu", "Ranks sharing the MAGMA Ozaki-II workspace budget; default 4.") \
    X(EXT_MAGMA_OZAKI2_MAX_WS_MB, "MAGMA_OZAKI2_MAX_WS_MB", "double", "global", "512", "backends/hip/src/Cluster_DFT_Col.c,backends/hip/src/magma_openmx.cu", "MAGMA Ozaki-II per-process workspace cap in MiB; default 512.") \
    X(EXT_MAGMA_OZAKI2_MAX_WS_PERCENT, "MAGMA_OZAKI2_MAX_WS_PERCENT", "double", "global", "auto", "backends/hip/src/Cluster_DFT_Col.c", "Additional MAGMA workspace cap as a percentage of device memory; divided by local ranks.") \
    X(EXT_MAGMA_OZAKI2_MIN_FREE_MB, "MAGMA_OZAKI2_MIN_FREE_MB", "mib", "global", "", "backends/hip/src/magma_openmx.cu", "MAGMA Ozaki-II free-memory guard in MiB; OpenMX supplies 1024 when unset.") \
    X(EXT_MPI_LOCALRANKID, "MPI_LOCALRANKID", "int", "local", "auto", "backends/hip/src/Divide_Conquer.c", "Launcher-provided local rank alias; remains local and is never copied from rank 0.") \
    X(EXT_MV2_COMM_WORLD_LOCAL_RANK, "MV2_COMM_WORLD_LOCAL_RANK", "int", "local", "", "backends/cpu/device.c,backends/hip/src/set_hip_default_device_from_local_rank.c,set_cuda_default_device_from_local_rank.c", "MVAPICH2: node-local rank") \
    X(EXT_MV2_COMM_WORLD_LOCAL_SIZE, "MV2_COMM_WORLD_LOCAL_SIZE", "int", "local", "", "backends/cpu/device.c,backends/hip/src/set_hip_default_device_from_local_rank.c,set_cuda_default_device_from_local_rank.c", "MVAPICH2: ranks on this node") \
    X(EXT_OMPI_COMM_WORLD_LOCAL_RANK, "OMPI_COMM_WORLD_LOCAL_RANK", "int", "local", "auto", "backends/cpu/device.c,backends/hip/src/Divide_Conquer.c,backends/hip/src/set_hip_default_device_from_local_rank.c,set_cuda_default_device_from_local_rank.c", "Open MPI: node-local rank") \
    X(EXT_OMPI_COMM_WORLD_LOCAL_SIZE, "OMPI_COMM_WORLD_LOCAL_SIZE", "int", "local", "0", "backends/cpu/device.c,backends/hip/src/Cluster_DFT_Col.c,backends/hip/src/gemmul8_bridge.cu,backends/hip/src/set_hip_default_device_from_local_rank.c,elpa_cosma_bridge.c,gemmul8_bridge.cu,set_cuda_default_device_from_local_rank.c", "Open MPI: ranks on this node") \
    X(EXT_OMPI_COMM_WORLD_RANK, "OMPI_COMM_WORLD_RANK", "string", "local", "", "backends/hip/src/gemmul8_bridge.cu,gemmul8_bridge.cu", "Open MPI: world rank (GEMMul8 statistics label)") \
    X(ABORT_AFTER_EIGEN, "OPENMX_ABORT_AFTER_EIGEN", "flag", "global", "0", "backends/hip/src/Band_DFT_Col.c,backends/hip/src/Band_DFT_NonCol.c", "Diagnostic: terminate after the first band eigensolve.") \
    X(ABORT_AFTER_EIGEN_SCF, "OPENMX_ABORT_AFTER_EIGEN_SCF", "int", "global", "0", "backends/hip/src/Band_DFT_Col.c", "Diagnostic: SCF iteration at which to stop after the band eigensolve; 0 disables.") \
    X(AFFINITY_DIR, "OPENMX_AFFINITY_DIR", "tool", "local", "", "build/tools", "mi300a_rank_launcher.py: directory of the per-rank CPU affinity files") \
    X(BAND_CACHE_TRACE, "OPENMX_BAND_CACHE_TRACE", "flag", "global", "0", "Band_DFT_Col.c,backends/hip/src/Band_DFT_Col.c,backends/hip/src/DFT.c,backends/hip/src/Set_Hamiltonian.c", "") \
    X(BAND_COL_INTERLEAVE_K, "OPENMX_BAND_COL_INTERLEAVE_K", "flag", "global", "", "backends/hip/src/Band_DFT_Col.c", "Override collinear k-point interleaving; retains the existing automatic layout by default.") \
    X(BAND_COL_KCACHE_MB, "OPENMX_BAND_COL_KCACHE_MB", "double", "global", "1024", "Band_DFT_Col.c,backends/hip/src/Band_DFT_Col.c", "") \
    X(BAND_COL_OVERLAP_CACHE_MB, "OPENMX_BAND_COL_OVERLAP_CACHE_MB", "double", "global", "512", "Band_DFT_Col.c,backends/hip/src/Band_DFT_Col.c", "") \
    X(BAND_GEMMUL8_BACKTRANSFORM, "OPENMX_BAND_GEMMUL8_BACKTRANSFORM", "flag", "global", "1", "backends/hip/src/Band_DFT_Col.c", "Use GEMMul8 for the collinear band eigenvector back-transform.") \
    X(BAND_GEMMUL8_TURN_RELEASE, "OPENMX_BAND_GEMMUL8_TURN_RELEASE", "flag", "global", "", "Band_DFT_Col.c,Band_DFT_NonCol.c,Cluster_DFT_Col.c,Cluster_DFT_NonCol.c,Divide_Conquer_LNO.c,backends/hip/src/Band_DFT_Col.c,backends/hip/src/Band_DFT_NonCol.c,backends/hip/src/Cluster_DFT_Col.c,backends/hip/src/Cluster_DFT_NonCol.c,backends/hip/src/Divide_Conquer_LNO.c", "default 1: Per-turn GEMMul8 workspace release (0 keeps them)") \
    X(BAND_GPU_CACHED_TURN_GROUP, "OPENMX_BAND_GPU_CACHED_TURN_GROUP", "int", "global", "", "backends/hip/src/Band_DFT_Col.c", "GPU turn-group size after SCF iteration 1; defaults to the normal turn group, clamped to 1..1024.") \
    X(BAND_GPU_DIAG, "OPENMX_BAND_GPU_DIAG", "flag", "global", "", "Band_DFT_Col.c,Band_DFT_NonCol.c,backends/hip/src/Band_DFT_Col.c,backends/hip/src/Band_DFT_NonCol.c", "default auto: 1 = force the dense GPU path ignoring the fits verdict; 0 = always ELPA/ScaLAPACK (§5.6)") \
    X(BAND_GPU_DIAG_RESERVE_MB, "OPENMX_BAND_GPU_DIAG_RESERVE_MB", "mib", "global", "", "backends/hip/src/Band_DFT_Col.c", "Additional band eigensolver memory guard in MiB.") \
    X(BAND_GPU_DM, "OPENMX_BAND_GPU_DM", "flag", "global", "0 (1 under OPENMX_HOST_SIMD=legacy)", "Band_DFT_Col.c,backends/hip/src/Band_DFT_Col.c", "default 0 (1 under OPENMX_HOST_SIMD=legacy): 1 = GPU DM accumulation of the collinear band ScaLAPACK/ELPA fallback (behind the device-or-host vote); 0 = host SIMD DM (BandCol_DM_Block)") \
    X(BAND_GPU_MAX_CONCURRENT_K, "OPENMX_BAND_GPU_MAX_CONCURRENT_K", "int", "global", "", "Band_DFT_Col.c,Band_DFT_NonCol.c,backends/hip/src/Band_DFT_Col.c,backends/hip/src/Band_DFT_NonCol.c", "") \
    X(BAND_GPU_MAX_DEVICE_RANKS, "OPENMX_BAND_GPU_MAX_DEVICE_RANKS", "int", "global", "16", "backends/hip/src/set_hip_default_device_from_local_rank.c", "Above this many ranks per device, prefer CPU dense band/Krylov paths; 0 disables the limit.") \
    X(BAND_GPU_MAX_RANKS_PER_DEVICE, "OPENMX_BAND_GPU_MAX_RANKS_PER_DEVICE", "int", "global", "", "Band_DFT_Col.c,Band_DFT_NonCol.c,backends/hip/src/Band_DFT_Col.c,backends/hip/src/Band_DFT_NonCol.c", "") \
    X(BAND_GPU_PERSISTENT, "OPENMX_BAND_GPU_PERSISTENT", "flag", "global", "", "Band_DFT_Col.c,backends/hip/src/Band_DFT_Col.c", "") \
    X(BAND_GPU_PERSISTENT_MIN_FREE_MB, "OPENMX_BAND_GPU_PERSISTENT_MIN_FREE_MB", "mib", "global", "2048", "Band_DFT_Col.c,backends/hip/src/Band_DFT_Col.c", "") \
    X(BAND_GPU_RANK_OVERHEAD_MB, "OPENMX_BAND_GPU_RANK_OVERHEAD_MB", "mib", "global", "256", "Band_DFT_Col.c,Band_DFT_NonCol.c", "default 256: Runtime overhead added to the per-rank requirement model") \
    X(BAND_GPU_RESERVE_MB, "OPENMX_BAND_GPU_RESERVE_MB", "mib", "global", "max(total/10, 1536)", "Band_DFT_Col.c,Band_DFT_NonCol.c,backends/hip/src/Band_DFT_Col.c,backends/hip/src/Band_DFT_NonCol.c", "default max(total/10, 1536): Raises the reserve used by the concurrency planner and the fits verdicts") \
    X(BAND_GPU_SWITCH_NUM, "OPENMX_BAND_GPU_SWITCH_NUM", "int", "global", "800", "Band_DFT_Col.c,backends/hip/src/Band_DFT_Col.c", "default 800: GPU/CPU crossover dimension of the collinear band dense path (paths without their own knob keep GPU_CPU_SWITCH_NUM=1000). Measured crossover on one...") \
    X(BAND_GPU_TURN_GROUP, "OPENMX_BAND_GPU_TURN_GROUP", "int", "global", "", "Band_DFT_Col.c,Band_DFT_NonCol.c,backends/hip/src/Band_DFT_Col.c,backends/hip/src/Band_DFT_NonCol.c", "") \
    X(BAND_NONCOL_CACHE_TRACE, "OPENMX_BAND_NONCOL_CACHE_TRACE", "flag", "global", "0", "Band_DFT_NonCol.c,backends/hip/src/Band_DFT_NonCol.c", "") \
    X(BAND_NONCOL_GPU_DM, "OPENMX_BAND_NONCOL_GPU_DM", "flag", "global", "1", "Band_DFT_NonCol.c,backends/hip/src/Band_DFT_NonCol.c", "default 1: 0 = CPU loop for the per-k DM accumulation of the noncollinear ScaLAPACK/ELPA-fallback multi-k loop (the k-dense GPU flow keeps its own device DM r...") \
    X(BAND_NONCOL_GPU_SWITCH_NUM, "OPENMX_BAND_NONCOL_GPU_SWITCH_NUM", "int", "global", "800", "Band_DFT_NonCol.c,backends/hip/src/Band_DFT_NonCol.c", "default 800: Same for the noncollinear band dense path, compared against 2n (measured crossover 2n ≈ 250–400)") \
    X(BAND_NONCOL_INTERLEAVE_K, "OPENMX_BAND_NONCOL_INTERLEAVE_K", "flag", "global", "auto", "Band_DFT_NonCol.c,backends/hip/src/Band_DFT_NonCol.c", "") \
    X(BAND_NONCOL_KCACHE_MB, "OPENMX_BAND_NONCOL_KCACHE_MB", "double", "global", "1024", "Band_DFT_NonCol.c,backends/hip/src/Band_DFT_NonCol.c", "") \
    X(BAND_NONCOL_OVERLAP_CACHE_MB, "OPENMX_BAND_NONCOL_OVERLAP_CACHE_MB", "double", "global", "512", "Band_DFT_NonCol.c,backends/hip/src/Band_DFT_NonCol.c", "") \
    X(BAND_NONCOL_PACKED_CACHE, "OPENMX_BAND_NONCOL_PACKED_CACHE", "flag", "global", "1", "backends/hip/src/Band_DFT_NonCol.c", "Reuse packed Hamiltonian/overlap data for noncollinear bands when available.") \
    X(BAND_NONCOL_PACK_CACHE_MB, "OPENMX_BAND_NONCOL_PACK_CACHE_MB", "double", "global", "128", "Band_DFT_NonCol.c", "") \
    X(BAND_PROFILE, "OPENMX_BAND_PROFILE", "flag", "global", "0", "Band_DFT_Col.c,Cluster_DFT_Col.c,Force.c,Set_Density_Grid.c,Set_Hamiltonian.c,Set_Nonlocal.c,Set_OLP_Kin.c,Set_Vpot.c,Total_Energy.c,backends/hip/src/Force.c,backends/hip/src/Set_Density_Grid.c,backends/hip/src/Set_Hamiltonian.c,backends/hip/src/Set_Nonlocal.c,truncation.c", "default 0: Performance counters; also gates the per-rank stage lines SETHPROF (Set_Hamiltonian), SETHOTFPROF / SETHOTFDENPROF (the on-the-fly batches: pack, e...") \
    X(BAND_REFINE_BASIS_MB, "OPENMX_BAND_REFINE_BASIS_MB", "double", "global", "1024", "Band_DFT_Col.c,Band_DFT_NonCol.c,backends/hip/src/Band_DFT_Col.c,backends/hip/src/Band_DFT_NonCol.c", "") \
    X(BAND_SYEVDX_NOVECTOR, "OPENMX_BAND_SYEVDX_NOVECTOR", "flag", "global", "1", "Band_DFT_Col.c,Band_DFT_NonCol.c", "default 1: Solve the eigenvalues-only first pass with NOVECTOR (0 restores the old behavior)") \
    X(BUILD_JOBS, "OPENMX_BUILD_JOBS", "tool", "tool", "", "build/tools", "CMake: parallelism of the native build") \
    X(CLUSTER_GEMMUL8_TURN_RELEASE, "OPENMX_CLUSTER_GEMMUL8_TURN_RELEASE", "flag", "global", "1", "Cluster_DFT_Col.c,Cluster_DFT_NonCol.c,backends/hip/src/Cluster_DFT_Col.c,backends/hip/src/Cluster_DFT_NonCol.c", "default 1: Per-solve GEMMul8 workspace release (inherits the BAND setting when unset)") \
    X(CLUSTER_GPU_DIAG, "OPENMX_CLUSTER_GPU_DIAG", "flag", "global", "", "Cluster_DFT_Col.c,Cluster_DFT_NonCol.c,backends/hip/src/Cluster_DFT_Col.c,backends/hip/src/Cluster_DFT_NonCol.c", "default auto: 1 = force the GPU even if the reservation fails (then aborts with a diagnostic); 0 = always ELPA (§5.6)") \
    X(CLUSTER_GPU_DIAG_RESERVE_MB, "OPENMX_CLUSTER_GPU_DIAG_RESERVE_MB", "double", "global", "", "Cluster_DFT_Col.c,Cluster_DFT_NonCol.c,backends/hip/src/Cluster_DFT_Col.c,backends/hip/src/Cluster_DFT_NonCol.c", "default 1024: Free VRAM required to remain after the reservation probe") \
    X(CLUSTER_GPU_DIAG_SERIAL, "OPENMX_CLUSTER_GPU_DIAG_SERIAL", "int", "global", "1", "Cluster_DFT_Col.c,backends/hip/src/Cluster_DFT_Col.c", "default 1: Serialized mode when both spin owners cannot fit (0 = disable, 2 = force serial)") \
    X(CLUSTER_GPU_DM, "OPENMX_CLUSTER_GPU_DM", "flag", "global", "1", "Cluster_DFT_Col.c,backends/hip/src/Cluster_DFT_Col.c", "default 1: GPU accumulation of DM/EDM (0 = CPU loop)") \
    X(CLUSTER_GPU_DM_ALL, "OPENMX_CLUSTER_GPU_DM_ALL", "flag", "global", "1", "Cluster_DFT_Col.c", "default 1: Device accumulation of the cluster density matrix on every rank (its slice of states, staged with a halving retry); 0 = only the rank that owns the...") \
    X(CLUSTER_GPU_DM_GEMM, "OPENMX_CLUSTER_GPU_DM_GEMM", "flag", "global", "", "Cluster_DFT_NonCol.c,backends/hip/src/Cluster_DFT_NonCol.c", "default 1: Noncollinear cluster dense path: the density matrix as one product WHW over the occupied states (GEMMul8) plus a gather of the sparse entries, and...") \
    X(CLUSTER_GPU_GEMM, "OPENMX_CLUSTER_GPU_GEMM", "flag", "global", "1", "backends/hip/src/Cluster_DFT_Col.c", "Enable cluster GPU GEMM products; the capacity check accounts for their workspace.") \
    X(CLUSTER_GPU_OVERLAP_CACHE_MB, "OPENMX_CLUSTER_GPU_OVERLAP_CACHE_MB", "double", "global", "2048", "backends/hip/src/Cluster_DFT_Col.c", "Per-spin host overlap-cache limit in MiB; also limited to 1/32 of available host memory.") \
    X(CLUSTER_GPU_RETAIN_SCRATCH, "OPENMX_CLUSTER_GPU_RETAIN_SCRATCH", "string", "global", "", "Cluster_DFT_Col.c,backends/hip/src/Cluster_DFT_Col.c", "") \
    X(CLUSTER_GPU_ROOT_DENSE, "OPENMX_CLUSTER_GPU_ROOT_DENSE", "flag", "global", "1", "backends/hip/src/Cluster_DFT_NonCol.c", "Enable the owning-rank dense path for noncollinear clusters.") \
    X(CLUSTER_GPU_SERIAL, "OPENMX_CLUSTER_GPU_SERIAL", "int", "global", "1", "backends/hip/src/Cluster_DFT_Col.c", "Cluster capacity tiers: 0 disables serialized solves, 1 allows them, 2 skips the concurrent tier.") \
    X(CLUSTER_GPU_SWITCH_NUM, "OPENMX_CLUSTER_GPU_SWITCH_NUM", "int", "global", "400", "Cluster_DFT_Col.c,backends/hip/src/Cluster_DFT_Col.c", "default 400: GPU/CPU crossover dimension of the collinear cluster dense path (the GPU won at every measured size down to n=240)") \
    X(CLUSTER_GS2_FALLBACK, "OPENMX_CLUSTER_GS2_FALLBACK", "string", "global", "auto", "openmx_cluster_capacity.c", "default auto: Capacity fallback of gpusolver in the cluster solvers when the dense solve does not fit on one GPU: MAGMA on the GPUs of rank 0's node first, then...") \
    X(CLUSTER_HOST_MEM_AVAILABLE_MB, "OPENMX_CLUSTER_HOST_MEM_AVAILABLE_MB", "mib", "global", "measured", "openmx_cluster_capacity.c", "") \
    X(CLUSTER_MEASURE_TIME, "OPENMX_CLUSTER_MEASURE_TIME", "flag", "global", "0", "backends/hip/src/Cluster_DFT_Col.c", "Print HIP cluster dense-solver stage timings.") \
    X(CLUSTER_NC_DEVICE_KEEP, "OPENMX_CLUSTER_NC_DEVICE_KEEP", "flag", "global", "", "backends/hip/src/Cluster_DFT_NonCol.c", "Keep the noncollinear cluster solve on the device across SCF steps; automatic default on MI300A.") \
    X(CLUSTER_NC_DEVICE_SOLVE, "OPENMX_CLUSTER_NC_DEVICE_SOLVE", "flag", "global", "1", "backends/hip/src/Cluster_DFT_NonCol.c", "Enable the noncollinear cluster device-resident dense solve.") \
    X(CLUSTER_NC_SETHAM_CACHE, "OPENMX_CLUSTER_NC_SETHAM_CACHE", "flag", "global", "1", "backends/hip/src/Cluster_DFT_NonCol.c", "Reuse the noncollinear cluster Set_Hamiltonian cache.") \
    X(CLUSTER_NONCOL_GPU_SWITCH_NUM, "OPENMX_CLUSTER_NONCOL_GPU_SWITCH_NUM", "int", "global", "400", "Cluster_DFT_NonCol.c,backends/hip/src/Cluster_DFT_NonCol.c", "default 400: Same for the noncollinear cluster dense path, compared against 2n (the GPU won down to 2n=480)") \
    X(CLUSTER_PROFILE, "OPENMX_CLUSTER_PROFILE", "flag", "global", "0", "Cluster_DFT_Col.c,Cluster_DFT_NonCol.c,backends/hip/src/Cluster_DFT_Col.c,backends/hip/src/Cluster_DFT_NonCol.c", "") \
    X(COPY_TO_WORK, "OPENMX_COPY_TO_WORK", "tool", "tool", "", "build/tools", "tools/copy_openmx.py") \
    X(CORE_LIBRARY, "OPENMX_CORE_LIBRARY", "tool", "tool", "", "build/tools", "tests and CMake: path of libopenmx_core.so to load") \
    X(CORE_VERBOSE, "OPENMX_CORE_VERBOSE", "tool", "tool", "", "build/tools", "C++ core: report once why a transform ran on the host (external Kokkos owner, device change, FP64)") \
    X(CUBLAS_MIN_FREE_AFTER_MB, "OPENMX_CUBLAS_MIN_FREE_AFTER_MB", "mib", "global", "", "Divide_Conquer.c,backends/hip/src/Divide_Conquer.c", "default —: Same for cuBLAS") \
    X(CUSOLVER_ALGORITHM, "OPENMX_CUSOLVER_ALGORITHM", "string", "global", "auto", "openmx_cusolver_compat.c", "") \
    X(CUSOLVER_CACHE_MB, "OPENMX_CUSOLVER_CACHE_MB", "mib", "global", "64", "backends/hip/src/magma_openmx.cu,gpusolver_Syevdx.c", "") \
    X(CUSOLVER_EMULATION, "OPENMX_CUSOLVER_EMULATION", "flag", "global", "1", "openmx_cusolver_compat.c", "default 1: Every cuSOLVER handle the code creates runs its FP64 eigensolves in the fixed-point FP64 emulation of cuSOLVER 12.2+ (CUSOLVER_FP64_EMULATED_FIXEDP...") \
    X(CUSOLVER_RETRY_TEST, "OPENMX_CUSOLVER_RETRY_TEST", "flag", "global", "0", "backends/hip/src/magma_openmx.cu,openmx_cusolver_compat.c", "") \
    X(CUSOLVER_TWO_STAGE_MIN_N, "OPENMX_CUSOLVER_TWO_STAGE_MIN_N", "string", "global", "unset (cuSOLVER's choice)", "openmx_cusolver_compat.c", "") \
    X(CUSOLVER_VERBOSE, "OPENMX_CUSOLVER_VERBOSE", "flag", "global", "0", "Cluster_DFT_Col.c,openmx_cusolver_compat.c", "") \
    X(DCLNO_GPU_OWNER_THRESHOLD, "OPENMX_DCLNO_GPU_OWNER_THRESHOLD", "int", "global", "2048", "backends/hip/src/Divide_Conquer_LNO.c", "Matrix-order threshold for a DC-LNO owning-rank GPU solve.") \
    X(DCLNO_GPU_PROFILE, "OPENMX_DCLNO_GPU_PROFILE", "flag", "global", "0", "backends/hip/src/Divide_Conquer_LNO.c", "Print DC-LNO GPU proxy timings.") \
    X(DCLNO_GPU_THRESHOLD, "OPENMX_DCLNO_GPU_THRESHOLD", "int", "global", "800", "Divide_Conquer_LNO.c,backends/hip/src/Divide_Conquer_LNO.c", "default 800: Minimum local dimension for the GPU (collinear)") \
    X(DCLNO_GPU_TURN_RELEASE, "OPENMX_DCLNO_GPU_TURN_RELEASE", "flag", "global", "1", "Divide_Conquer_LNO.c,backends/hip/src/Divide_Conquer_LNO.c", "default 1: End-of-phase release of solver buffers + GEMMul8 (inherits the BAND setting when unset)") \
    X(DCLNO_NONCOL_GPU_THRESHOLD, "OPENMX_DCLNO_NONCOL_GPU_THRESHOLD", "int", "global", "400", "Divide_Conquer_LNO.c,backends/hip/src/Divide_Conquer_LNO.c", "default 400: Minimum spinor dimension for the GPU (noncollinear)") \
    X(DCLNO_PROFILE, "OPENMX_DCLNO_PROFILE", "flag", "global", "0", "Divide_Conquer_LNO.c,backends/hip/src/Divide_Conquer_LNO.c", "") \
    X(DCLNO_RETRY_TEST, "OPENMX_DCLNO_RETRY_TEST", "flag", "global", "0", "backends/hip/src/Divide_Conquer_LNO.c", "Diagnostic: exercise the DC-LNO retry path.") \
    X(DC_GPU_MAX_RANKS_PER_DEVICE, "OPENMX_DC_GPU_MAX_RANKS_PER_DEVICE", "uint", "global", "", "backends/hip/src/Divide_Conquer.c", "Limit DC GPU owners per device; launcher/topology fallback applies when unset.") \
    X(DC_GPU_S_CACHE, "OPENMX_DC_GPU_S_CACHE", "uint", "global", "1", "Divide_Conquer.c,backends/hip/src/Divide_Conquer.c", "") \
    X(DC_GPU_S_CACHE_RESERVE_MB, "OPENMX_DC_GPU_S_CACHE_RESERVE_MB", "mib", "global", "", "Divide_Conquer.c,backends/hip/src/Divide_Conquer.c", "") \
    X(DC_GPU_THRESHOLD, "OPENMX_DC_GPU_THRESHOLD", "int", "global", "400", "Divide_Conquer.c,backends/hip/src/Divide_Conquer.c", "default 400: Minimum local cluster dimension NUM for the GPU (collinear)") \
    X(DC_NONCOL_GPU_THRESHOLD, "OPENMX_DC_NONCOL_GPU_THRESHOLD", "int", "global", "300", "Divide_Conquer.c,backends/hip/src/Divide_Conquer.c", "default 300: Minimum spinor dimension 2·NUM for the GPU (noncollinear)") \
    X(DENSITY_GRID_COL_SIMD, "OPENMX_DENSITY_GRID_COL_SIMD", "string", "global", "", "openmx_grid_assembly.c", "") \
    X(DENSITY_GRID_COL_SIMD_TRACE, "OPENMX_DENSITY_GRID_COL_SIMD_TRACE", "flag", "global", "", "openmx_grid_assembly.c", "") \
    X(DENSITY_GRID_GPU, "OPENMX_DENSITY_GRID_GPU", "flag", "global", "", "Set_Density_Grid_GPU.c,backends/hip/src/Set_Density_Grid_GPU.c", "") \
    X(DENSITY_GRID_GPU_CACHE_MB, "OPENMX_DENSITY_GRID_GPU_CACHE_MB", "mib", "global", "64", "Set_Density_Grid_GPU.c,backends/hip/src/Set_Density_Grid_GPU.c", "") \
    X(DENSITY_GRID_GPU_LOCAL, "OPENMX_DENSITY_GRID_GPU_LOCAL", "int", "global", "1", "Set_Density_Grid_GPU.c,backends/hip/src/Set_Density_Grid_GPU.c", "default 1: Distributed quadrature mode: 0=off, 1=only when the shared tables are resident, 2=always") \
    X(DENSITY_GRID_GPU_LOCAL_CACHE_MB, "OPENMX_DENSITY_GRID_GPU_LOCAL_CACHE_MB", "mib", "global", "32", "Set_Density_Grid_GPU.c,backends/hip/src/Set_Density_Grid_GPU.c", "") \
    X(DENSITY_GRID_GPU_PERSISTENT, "OPENMX_DENSITY_GRID_GPU_PERSISTENT", "flag", "global", "0", "Set_Density_Grid_GPU.c,backends/hip/src/Set_Density_Grid_GPU.c", "") \
    X(DENSITY_GRID_GPU_RESERVE_MB, "OPENMX_DENSITY_GRID_GPU_RESERVE_MB", "mib", "global", "256", "Set_Density_Grid_GPU.c,backends/hip/src/Set_Density_Grid_GPU.c", "") \
    X(DENSITY_GRID_NC_SIMD, "OPENMX_DENSITY_GRID_NC_SIMD", "string", "global", "", "openmx_grid_assembly.c", "") \
    X(DENSITY_GRID_NC_SIMD_TRACE, "OPENMX_DENSITY_GRID_NC_SIMD_TRACE", "flag", "global", "", "openmx_grid_assembly.c", "") \
    X(DENSITY_GRID_ONTHEFLY, "OPENMX_DENSITY_GRID_ONTHEFLY", "flag", "global", "1", "backends/hip/src/Set_Hamiltonian.c", "Enable density-grid evaluation from orbitals computed on demand.") \
    X(EH0_GPU_CONCURRENCY, "OPENMX_EH0_GPU_CONCURRENCY", "int", "global", "auto", "backends/hip/src/Total_Energy_GPU.c", "EH0 GPU ranks per turn; 0 bypasses turn scheduling, automatic selects from batch size.") \
    X(EIGEN_FP32_SOLVER, "OPENMX_EIGEN_FP32_SOLVER", "string", "global", "", "backends/hip/src/eigen_refine_gpu.cu", "FP32 refinement eigensolver backend; magma selects MAGMA.") \
    X(EIGEN_FP32_TWO_STAGE, "OPENMX_EIGEN_FP32_TWO_STAGE", "flag", "global", "1", "eigen_refine_gpu.c", "default 1: The full FP32 solve of the refined eigensolver uses the two-stage cusolverDnXsyevd (CUSOLVER_ALG_2; cuSOLVER 12.3.4 and later; RTX 5080, n = 5616:...") \
    X(EIGEN_REFINE, "OPENMX_EIGEN_REFINE", "string", "global", "", "backends/hip/src/eigen_refine_gpu.cu,eigen_refine_gpu.c", "Cluster dense paths (collinear, per spin; noncollinear) and the band solvers (one k-point per rank: the estimates of all k-points are gathered befo...") \
    X(EIGEN_REFINE_ALL, "OPENMX_EIGEN_REFINE_ALL", "flag", "global", "0", "backends/hip/src/eigen_refine_gpu.cu,eigen_refine_gpu.c", "") \
    X(EIGEN_REFINE_CHECK, "OPENMX_EIGEN_REFINE_CHECK", "flag", "global", "0", "backends/hip/src/eigen_refine_gpu.cu,eigen_refine_gpu.c", "") \
    X(EIGEN_REFINE_COMPLEMENT, "OPENMX_EIGEN_REFINE_COMPLEMENT", "flag", "global", "1", "backends/hip/src/eigen_refine_gpu.cu,eigen_refine_gpu.c", "") \
    X(EIGEN_REFINE_FIRST_ALL, "OPENMX_EIGEN_REFINE_FIRST_ALL", "flag", "global", "0", "backends/hip/src/Cluster_DFT_Col.c,backends/hip/src/Cluster_DFT_NonCol.c", "Refine every state in the first cluster SCF step.") \
    X(EIGEN_REFINE_MIN_N, "OPENMX_EIGEN_REFINE_MIN_N", "int", "global", "1000", "backends/hip/src/eigen_refine_gpu.cu", "Minimum matrix order for refined FP32 eigensolves.") \
    X(EIGEN_REFINE_STOP_GUARD, "OPENMX_EIGEN_REFINE_STOP_GUARD", "double", "global", "100", "DFT.c,backends/hip/src/DFT.c", "default 100: An SCF stop on a refined FP32 step also needs the printed NormRD below this factor times scf.criterion; OpenMX's test on dUele alone fired on a coi...") \
    X(EIGEN_REFINE_UNBLOCKED, "OPENMX_EIGEN_REFINE_UNBLOCKED", "flag", "global", "1", "backends/hip/src/eigen_refine_gpu.cu,eigen_refine_gpu.c", "default 1: The refinement products with the full GEMMul8 workspace when the device has room for it (about 1.7x faster than the blocked products at the 256 MiB...") \
    X(EIGEN_REFINE_UNTIL, "OPENMX_EIGEN_REFINE_UNTIL", "double", "global", "0 (off)", "backends/hip/src/eigen_refine_gpu.cu,eigen_refine_gpu.c", "default 0 (off): Printed NormRD below which the refined eigensolver hands the rest of the SCF cycle to the FP64 eigensolver. Off by default: an SCF that ends on a r...") \
    X(EIGEN_REFINE_WARM, "OPENMX_EIGEN_REFINE_WARM", "flag", "global", "1", "backends/hip/src/eigen_refine_gpu.cu,eigen_refine_gpu.c", "default 1: Warm start of the refined eigensolver: the vectors of the previous step's refined solve are refined against the new Hamiltonian instead of solving...") \
    X(EIGEN_REFINE_WARM_BAND, "OPENMX_EIGEN_REFINE_WARM_BAND", "flag", "global", "1", "backends/hip/src/eigen_refine_gpu.cu,eigen_refine_gpu.c", "") \
    X(EIGEN_REFINE_WARM_DELTA_FLOOR, "OPENMX_EIGEN_REFINE_WARM_DELTA_FLOOR", "flag", "global", "1", "backends/hip/src/eigen_refine_gpu.cu,eigen_refine_gpu.c", "") \
    X(EIGEN_REFINE_WARM_PERIOD, "OPENMX_EIGEN_REFINE_WARM_PERIOD", "int", "global", "0", "backends/hip/src/eigen_refine_gpu.cu,eigen_refine_gpu.c", "") \
    X(FORCE3_GPU, "OPENMX_FORCE3_GPU", "flag", "global", "1", "Force.c,backends/hip/src/Force.c", "") \
    X(FORCE3_GPU_DORB, "OPENMX_FORCE3_GPU_DORB", "flag", "global", "1", "Force.c,backends/hip/src/Force.c", "") \
    X(FORCE4B_ARCHIVE_GPU, "OPENMX_FORCE4B_ARCHIVE_GPU", "flag", "global", "auto", "Force.c", "") \
    X(FORCE4B_CASE1_STREAM, "OPENMX_FORCE4B_CASE1_STREAM", "flag", "global", "auto", "Force.c,backends/hip/src/Force.c", "") \
    X(FORCE4B_CASE1_STREAM_MB, "OPENMX_FORCE4B_CASE1_STREAM_MB", "mib", "global", "1024", "Force.c,backends/hip/src/Force.c", "") \
    X(FORCE4B_CASE1_TURNS, "OPENMX_FORCE4B_CASE1_TURNS", "flag", "global", "auto", "Force.c,backends/hip/src/Force.c", "") \
    X(FORCE4B_CASE1_TURN_MAX_RANKS, "OPENMX_FORCE4B_CASE1_TURN_MAX_RANKS", "int", "global", "uncapped", "Force.c,backends/hip/src/Force.c", "") \
    X(FORCE4B_CASE2_STREAM, "OPENMX_FORCE4B_CASE2_STREAM", "flag", "global", "auto", "Force.c,backends/hip/src/Force.c", "") \
    X(FORCE4B_FUSED, "OPENMX_FORCE4B_FUSED", "flag", "global", "1", "Force.c,backends/hip/src/Force.c", "") \
    X(FORCE4B_GPU, "OPENMX_FORCE4B_GPU", "flag", "global", "1", "Force.c,backends/hip/src/Force.c", "") \
    X(FORCE4B_ORDERN_AUTO, "OPENMX_FORCE4B_ORDERN_AUTO", "flag", "global", "0", "backends/hip/src/Force.c", "Opt in to automatic order-N Force4B policy.") \
    X(FORCE4B_WEAK_FP64_AUTO, "OPENMX_FORCE4B_WEAK_FP64_AUTO", "flag", "global", "0", "backends/hip/src/Force.c", "Opt in to automatic Force4B policy on GPUs with weak native FP64.") \
    X(FORCEHNL_GPU, "OPENMX_FORCEHNL_GPU", "flag", "global", "1", "openmx_force_hnl_gpu.h", "default 1: Force_HNL GPU batch") \
    X(FORCE_GPU, "OPENMX_FORCE_GPU", "flag", "global", "", "Force.c,backends/hip/src/Force.c", "default 1: Parent gate for all force GPU paths (collective)") \
    X(FORCE_PROFILE, "OPENMX_FORCE_PROFILE", "flag", "global", "0", "Force.c,backends/hip/src/Force.c,openmx_force_archive_lease.h", "default 0: [Force profile] lines (per-stage max/avg seconds)") \
    X(FORCE_RELEASE_SCF_CACHES, "OPENMX_FORCE_RELEASE_SCF_CACHES", "flag", "global", "1", "Force.c,backends/hip/src/Force.c", "") \
    X(GEMMUL8_A100_FORWARD_MIN_N, "OPENMX_GEMMUL8_A100_FORWARD_MIN_N", "uint", "global", "6144", "gemmul8_bridge.cu", "") \
    X(GEMMUL8_DISABLE, "OPENMX_GEMMUL8_DISABLE", "flag", "global", "", "backends/hip/src/gemmul8_bridge.cu,gemmul8_bridge.cu", "") \
    X(GEMMUL8_DISABLE_D, "OPENMX_GEMMUL8_DISABLE_D", "flag", "global", "", "backends/hip/src/gemmul8_bridge.cu,gemmul8_bridge.cu", "") \
    X(GEMMUL8_DISABLE_Z, "OPENMX_GEMMUL8_DISABLE_Z", "flag", "global", "", "backends/hip/src/gemmul8_bridge.cu,gemmul8_bridge.cu", "") \
    X(GEMMUL8_FASTMODE_D, "OPENMX_GEMMUL8_FASTMODE_D", "flag", "global", "0", "backends/hip/src/gemmul8_bridge.cu,gemmul8_bridge.cu", "") \
    X(GEMMUL8_FASTMODE_Z, "OPENMX_GEMMUL8_FASTMODE_Z", "flag", "global", "0", "backends/hip/src/gemmul8_bridge.cu,gemmul8_bridge.cu", "") \
    X(GEMMUL8_FORWARD, "OPENMX_GEMMUL8_FORWARD", "string", "global", "as the real products", "gemmul8_bridge.cu", "") \
    X(GEMMUL8_FORWARD_FASTMODE, "OPENMX_GEMMUL8_FORWARD_FASTMODE", "flag", "global", "0", "gemmul8_bridge.cu", "") \
    X(GEMMUL8_FORWARD_NUM_MOD, "OPENMX_GEMMUL8_FORWARD_NUM_MOD", "uint", "global", "15", "gemmul8_bridge.cu", "") \
    X(GEMMUL8_FORWARD_REUSE, "OPENMX_GEMMUL8_FORWARD_REUSE", "flag", "global", "0", "gemmul8_bridge.cu", "") \
    X(GEMMUL8_FORWARD_TIMING, "OPENMX_GEMMUL8_FORWARD_TIMING", "flag", "global", "0", "gemmul8_bridge.cu", "") \
    X(GEMMUL8_FORWARD_UNBLOCKED, "OPENMX_GEMMUL8_FORWARD_UNBLOCKED", "flag", "global", "1", "backends/hip/src/Cluster_DFT_Col.c,gemmul8_bridge.cu", "") \
    X(GEMMUL8_LOCAL_RANKS, "OPENMX_GEMMUL8_LOCAL_RANKS", "uint", "global", "from the launcher", "backends/hip/src/gemmul8_bridge.cu,gemmul8_bridge.cu", "") \
    X(GEMMUL8_MAX_WORKSPACE_MB, "OPENMX_GEMMUL8_MAX_WORKSPACE_MB", "mib", "global", "256", "backends/hip/src/gemmul8_bridge.cu,gemmul8_bridge.cu", "") \
    X(GEMMUL8_MAX_WORKSPACE_PERCENT, "OPENMX_GEMMUL8_MAX_WORKSPACE_PERCENT", "uint", "global", "", "Band_DFT_NonCol.c,backends/hip/src/Band_DFT_NonCol.c,backends/hip/src/gemmul8_bridge.cu,gemmul8_bridge.cu", "") \
    X(GEMMUL8_MIN_FREE_AFTER_MB, "OPENMX_GEMMUL8_MIN_FREE_AFTER_MB", "mib", "global", "", "Divide_Conquer.c,backends/hip/src/Divide_Conquer.c,backends/hip/src/gemmul8_bridge.cu,gemmul8_bridge.cu", "") \
    X(GEMMUL8_NUM_MOD_D, "OPENMX_GEMMUL8_NUM_MOD_D", "uint", "global", "", "backends/hip/src/gemmul8_bridge.cu,gemmul8_bridge.cu", "") \
    X(GEMMUL8_NUM_MOD_Z, "OPENMX_GEMMUL8_NUM_MOD_Z", "uint", "global", "", "backends/hip/src/gemmul8_bridge.cu,gemmul8_bridge.cu", "") \
    X(GEMMUL8_SOLVE_MIN_FREE_AFTER_MB, "OPENMX_GEMMUL8_SOLVE_MIN_FREE_AFTER_MB", "mib", "global", "256", "backends/hip/src/gemmul8_bridge.cu", "Free-memory guard in MiB after solve-specific GEMMul8 workspace allocation.") \
    X(GEMMUL8_TIMING, "OPENMX_GEMMUL8_TIMING", "flag", "global", "0", "backends/hip/src/gemmul8_bridge.cu,gemmul8_bridge.cu", "") \
    X(GEMMUL8_VERBOSE, "OPENMX_GEMMUL8_VERBOSE", "flag", "global", "", "backends/hip/src/gemmul8_bridge.cu,gemmul8_bridge.cu", "") \
    X(GEMM_SAMPLE_DIR, "OPENMX_GEMM_SAMPLE_DIR", "string", "global", "unset (off)", "Cluster_DFT_Col.c,Cluster_DFT_NonCol.c,backends/hip/src/Cluster_DFT_Col.c", "") \
    X(GEMM_SAMPLE_ITERS, "OPENMX_GEMM_SAMPLE_ITERS", "string", "global", "1", "Cluster_DFT_Col.c,Cluster_DFT_NonCol.c,backends/hip/src/Cluster_DFT_Col.c", "") \
    X(GEMM_VERBOSE, "OPENMX_GEMM_VERBOSE", "flag", "global", "", "backends/hip/src/gemmul8_bridge.cu,backends/hip/src/my_hipblasDgemm.c,backends/hip/src/my_hipblasZgemm.c,gemmul8_bridge.cu", "") \
    X(GPU, "OPENMX_GPU", "flag", "global", "1", "Input_std.c,backends/hip/include/openmx_gpu_arch_policy.h,backends/hip/src/Set_Density_Grid_GPU.c,backends/hip/src/Set_Hamiltonian.c,backends/hip/src/Set_Orbitals_Grid.c,backends/hip/src/Set_XC_Grid.c,backends/hip/src/Total_Energy_GPU.c,backends/hip/src/set_hip_default_device_from_local_rank.c,openmx_gpu_workers.c,openmx_mgpu_runtime.cu,set_cuda_default_device_from_local_rank.c", "") \
    X(GPUSOLVER_CACHE_MB, "OPENMX_GPUSOLVER_CACHE_MB", "mib", "global", "64", "backends/hip/src/magma_openmx.cu", "HIP solver scratch/retry-copy cache cap in MiB; overrides the CUSOLVER_CACHE_MB alias.") \
    X(GPUSOLVER_MIN_FREE_AFTER_MB, "OPENMX_GPUSOLVER_MIN_FREE_AFTER_MB", "mib", "global", "", "Divide_Conquer.c,backends/hip/src/Divide_Conquer.c", "") \
    X(GPUSOLVER_PARTIAL_EVD_MIN_N, "OPENMX_GPUSOLVER_PARTIAL_EVD_MIN_N", "int", "global", "0", "backends/hip/src/magma_openmx.cu", "Minimum order for partial HIP eigensolves; 0 keeps the default policy.") \
    X(GPUSOLVER_RETRY_TEST, "OPENMX_GPUSOLVER_RETRY_TEST", "flag", "global", "0", "backends/hip/src/magma_openmx.cu", "Diagnostic: force a second dense solve through the retry backend.") \
    X(GPUSOLVER_SERIAL_GPU_TURNS, "OPENMX_GPUSOLVER_SERIAL_GPU_TURNS", "flag", "global", "1", "Band_DFT_Col.c,Band_DFT_NonCol.c,backends/hip/src/Band_DFT_Col.c,backends/hip/src/Band_DFT_NonCol.c", "default 1: GPU-turn serialization of the all_knum==1 path (0 = concurrent submission)") \
    X(GPU_ARCH, "OPENMX_GPU_ARCH", "tool", "tool", "", "build/tools", "CMake: GPU architecture of the build") \
    X(GPU_ASSUME_APU, "OPENMX_GPU_ASSUME_APU", "flag", "global", "auto", "backends/hip/src/magma_openmx.cu", "Override APU detection; auto queries HIP device properties.") \
    X(GPU_BACKEND, "OPENMX_GPU_BACKEND", "tool", "tool", "", "build/tools", "native layer / CMake: CUDA or HIP (a build-time selection, not read at run time)") \
    X(GPU_CAPACITY_TEST_RECOVERY, "OPENMX_GPU_CAPACITY_TEST_RECOVERY", "flag", "global", "0", "openmx_gpu_workers.h", "") \
    X(GPU_EIGENSOLVER, "OPENMX_GPU_EIGENSOLVER", "string", "global", "", "backends/hip/src/magma_openmx.cu", "HIP dense solver: hipsolver, magma, or automatic device policy.") \
    X(GPU_EIGENSOLVER_MAGMA_MIN_N, "OPENMX_GPU_EIGENSOLVER_MAGMA_MIN_N", "int", "global", "0", "backends/hip/src/magma_openmx.cu", "In automatic mode, use rocSOLVER below this matrix order; explicit solver selection wins.") \
    X(GPU_EIGENSOLVER_PROFILE, "OPENMX_GPU_EIGENSOLVER_PROFILE", "flag", "global", "0", "backends/hip/src/magma_openmx.cu", "Print dense HIP eigensolver backend/stage timings.") \
    X(GPU_FP32_FP64_RATIO, "OPENMX_GPU_FP32_FP64_RATIO", "int", "global", "0", "backends/hip/src/eigen_refine_gpu.cu", "Override FP32/FP64 throughput ratio; nonpositive or auto keeps hardware detection.") \
    X(GPU_HEAP_REEXEC, "OPENMX_GPU_HEAP_REEXEC", "string", "local", "", "openmx.c", "") \
    X(GPU_LOCK_DIR, "OPENMX_GPU_LOCK_DIR", "string", "global", "/tmp", "openmx_mgpu_runtime.cu", "") \
    X(GPU_LOCK_TIMEOUT_MS, "OPENMX_GPU_LOCK_TIMEOUT_MS", "int", "global", "300000", "openmx_mgpu_runtime.cu", "") \
    X(GPU_MEMINFO_NVML, "OPENMX_GPU_MEMINFO_NVML", "flag", "global", "1", "set_cuda_default_device_from_local_rank.c", "default 1: Every device-memory budget of the GPU paths starts from OpenMX_GpuMemGetInfo(): cudaMemGetInfo() with the free figure capped at the memory that is...") \
    X(GPU_PROBE_RESERVE_MB, "OPENMX_GPU_PROBE_RESERVE_MB", "mib", "global", "256", "backends/hip/src/set_hip_default_device_from_local_rank.c,set_cuda_default_device_from_local_rank.c", "") \
    X(GPU_RANK_MAP, "OPENMX_GPU_RANK_MAP", "string", "global", "block", "backends/hip/src/set_hip_default_device_from_local_rank.c,set_cuda_default_device_from_local_rank.c", "") \
    X(GPU_STACK_BYTES, "OPENMX_GPU_STACK_BYTES", "int", "global", "", "set_cuda_default_device_from_local_rank.c", "Per-thread kernel stack limit set at device initialization (cudaLimitStackSize, OpenMX_GpuPresizeKernelStack) when it exceeds the driver's default...") \
    X(GPU_VERBOSE, "OPENMX_GPU_VERBOSE", "flag", "global", "0", "backends/hip/src/Band_DFT_NonCol.c,backends/hip/src/Cluster_DFT_Col.c,backends/hip/src/Cluster_DFT_NonCol.c,backends/hip/src/Divide_Conquer_LNO.c,backends/hip/src/Set_Hamiltonian.c,backends/hip/src/Total_Energy_GPU.c,backends/hip/src/gemmul8_bridge.cu,backends/hip/src/magma_openmx.cu,backends/hip/src/my_hipblasDgemm.c,backends/hip/src/my_hipblasZgemm.c,gemmul8_bridge.cu", "") \
    X(GPU_WORKERS, "OPENMX_GPU_WORKERS", "int", "global", "auto", "backends/hip/src/Total_Energy_GPU.c,openmx_gpu_workers.c,openmx_gpu_workers.h", "") \
    X(GPU_WORKER_COPY, "OPENMX_GPU_WORKER_COPY", "flag", "global", "0", "openmx_gpu_views.cpp", "") \
    X(GPU_WORKER_OVERLAP, "OPENMX_GPU_WORKER_OVERLAP", "flag", "global", "auto", "openmx_gpu_workers.c", "") \
    X(GRID_FP32, "OPENMX_GRID_FP32", "flag", "global", "0", "grid_precision.c", "default 0: Experimental: plain FP32 kernels while the previous step's NormRD is above OPENMX_GRID_FP32_UNTIL (1e-5), FP64 afterwards (OPENMX_GRID_FP32_RESTART...") \
    X(GRID_FP32_RESTART, "OPENMX_GRID_FP32_RESTART", "flag", "global", "1", "grid_precision.c", "") \
    X(GRID_FP32_UNTIL, "OPENMX_GRID_FP32_UNTIL", "double", "global", "1e-5", "grid_precision.c", "") \
    X(GRID_PRECISION, "OPENMX_GRID_PRECISION", "string", "global", "df / fp64", "backends/hip/src/set_hamiltonian_hip.cu,grid_precision.c", "default df / fp64: The arithmetic of the grid-integral kernels (the dVH+Vxc+VNA matrix elements and the density): df = double-float, FP32 arithmetic with error-free t...") \
    X(GS2_ELPA_GPU, "OPENMX_GS2_ELPA_GPU", "flag", "global", "", "elpa_cosma_bridge.c", "") \
    X(GS2_ELPA_SOLVER, "OPENMX_GS2_ELPA_SOLVER", "int", "global", "2", "elpa_cosma_bridge.c", "default 2: ELPA solver used by gpusolver2: 1 = 1stage, anything else = 2stage (default)") \
    X(GS2_GEMM_GPU, "OPENMX_GS2_GEMM_GPU", "flag", "global", "", "elpa_cosma_bridge.c", "") \
    X(GS2_GPU_FACTOR, "OPENMX_GS2_GPU_FACTOR", "int", "global", "3", "elpa_cosma_bridge.c", "") \
    X(GS2_GPU_FIXED_MB, "OPENMX_GS2_GPU_FIXED_MB", "mib", "global", "128", "elpa_cosma_bridge.c", "") \
    X(GS2_GPU_RESERVE_MB, "OPENMX_GS2_GPU_RESERVE_MB", "mib", "global", "512", "elpa_cosma_bridge.c", "") \
    X(GS2_GPU_TILE, "OPENMX_GS2_GPU_TILE", "int", "global", "1024", "elpa_cosma_bridge.c", "") \
    X(GS2_GPU_VERBOSE, "OPENMX_GS2_GPU_VERBOSE", "int", "global", "0", "elpa_cosma_bridge.c", "") \
    X(GS2_HOST_RESERVE_MB, "OPENMX_GS2_HOST_RESERVE_MB", "mib", "global", "", "elpa_cosma_bridge.c", "gpusolver2 bridge (explicit or as the capacity fallback): host memory that must stay free on every node after COSMA allocates the buffers it keeps...") \
    X(GS2_PROFILE, "OPENMX_GS2_PROFILE", "int", "global", "0", "elpa_cosma_bridge.c", "") \
    X(HOST_SIMD, "OPENMX_HOST_SIMD", "string", "global", "", "openmx_host_simd.c", "") \
    X(HOST_SIMD_MAX_PIECES, "OPENMX_HOST_SIMD_MAX_PIECES", "int", "global", "8", "openmx_host_simd.c", "") \
    X(HOST_SIMD_PREFER, "OPENMX_HOST_SIMD_PREFER", "string", "global", "", "openmx_host_simd.c", "") \
    X(KRYLOV_GEMMUL8, "OPENMX_KRYLOV_GEMMUL8", "flag", "global", "", "backends/hip/src/Krylov.c", "Enable Krylov GEMMul8 products, subject to the input-file GEMMul8 veto.") \
    X(KRYLOV_GPU, "OPENMX_KRYLOV_GPU", "flag", "global", "", "backends/hip/src/Krylov.c", "Enable Krylov GPU dense operations; automatic disables them on oversubscribed devices.") \
    X(KRYLOV_GPU_DGEMM_MIN_FLOPS, "OPENMX_KRYLOV_GPU_DGEMM_MIN_FLOPS", "double", "global", "5×10⁷", "Krylov.c,backends/hip/src/Krylov.c", "default 5×10⁷: Minimum m·n·k for dispatching a GEMM to the GPU") \
    X(KRYLOV_GPU_EIGEN_MIN, "OPENMX_KRYLOV_GPU_EIGEN_MIN", "int", "global", "300", "Krylov.c,backends/hip/src/Krylov.c", "default 300: Minimum projected dimension Msize3 for the device eigensolve (§7.6)") \
    X(KRYLOV_GPU_FUSED, "OPENMX_KRYLOV_GPU_FUSED", "flag", "global", "1", "Krylov.c,backends/hip/src/Krylov.c", "default 1: Fused per-atom pipeline (0 = per-call sequence)") \
    X(KRYLOV_GPU_KU_CACHE, "OPENMX_KRYLOV_GPU_KU_CACHE", "flag", "global", "1", "Krylov.c,backends/hip/src/Krylov.c", "") \
    X(KRYLOV_GPU_KU_RESERVE_MB, "OPENMX_KRYLOV_GPU_KU_RESERVE_MB", "mib", "global", "", "Krylov.c,backends/hip/src/Krylov.c", "") \
    X(KRYLOV_GPU_PROFILE, "OPENMX_KRYLOV_GPU_PROFILE", "flag", "global", "0", "Krylov.c,backends/hip/src/Krylov.c", "default 0: Fused-path breakdown (KRYFUSE lines)") \
    X(KRYLOV_GPU_RESERVE_MB, "OPENMX_KRYLOV_GPU_RESERVE_MB", "mib", "global", "", "backends/hip/src/Krylov.c", "Free-device-memory reserve in MiB for Krylov dense workspaces.") \
    X(KRYLOV_MEASURE_TIME, "OPENMX_KRYLOV_MEASURE_TIME", "flag", "global", "0", "backends/hip/src/Krylov.c", "Print detailed Krylov stage timings.") \
    X(MAGMA_FAILURE_DUMP, "OPENMX_MAGMA_FAILURE_DUMP", "string", "global", "unset", "openmx_magma_mgpu.cpp", "") \
    X(MAGMA_FP32, "OPENMX_MAGMA_FP32", "flag", "global", "0", "openmx_magma_mgpu.cpp", "") \
    X(MAGMA_FP32_COMPLEX_MAX_N, "OPENMX_MAGMA_FP32_COMPLEX_MAX_N", "int", "global", "0", "openmx_magma_mgpu.cpp", "") \
    X(MAGMA_HOST_CACHE_MB, "OPENMX_MAGMA_HOST_CACHE_MB", "mib", "global", "64", "openmx_magma_mgpu.cpp", "") \
    X(MAGMA_OZAKI2, "OPENMX_MAGMA_OZAKI2", "flag", "global", "1", "backends/hip/src/Cluster_DFT_Col.c,backends/hip/src/magma_openmx.cu", "Enable MAGMA Ozaki-II GEMMul8 acceleration.") \
    X(MAGMA_OZAKI2_RETRY, "OPENMX_MAGMA_OZAKI2_RETRY", "flag", "global", "1", "backends/hip/src/magma_openmx.cu", "Keep an input copy and retry failed Ozaki-II solves with the next backend when capacity permits.") \
    X(MAGMA_REFINE_RESERVE_MB, "OPENMX_MAGMA_REFINE_RESERVE_MB", "mib", "global", "4096", "openmx_mgpu_gemm.cu", "") \
    X(MAGMA_REFINE_RESIDENT_MB, "OPENMX_MAGMA_REFINE_RESIDENT_MB", "mib", "global", "0", "openmx_mgpu_gemm.cu", "") \
    X(MAGMA_REFINE_STEPS, "OPENMX_MAGMA_REFINE_STEPS", "int", "global", "2", "openmx_magma_refine.h", "") \
    X(MAGMA_RR_GPU_MIN, "OPENMX_MAGMA_RR_GPU_MIN", "int", "global", "1024", "openmx_magma_refine.h", "") \
    X(MAGMA_VALIDATE_FULL, "OPENMX_MAGMA_VALIDATE_FULL", "flag", "global", "0", "openmx_magma_mgpu.cpp", "") \
    X(MAKE_OPTIONS, "OPENMX_MAKE_OPTIONS", "tool", "tool", "", "build/tools", "CMake: native make variable assignments") \
    X(MGPU, "OPENMX_MGPU", "flag", "global", "auto", "openmx_mgpu_runtime.cu", "") \
    X(MGPU_BAND_SHARED, "OPENMX_MGPU_BAND_SHARED", "int", "global", "0", "openmx_mgpu_runtime.cu", "") \
    X(MGPU_GEMM, "OPENMX_MGPU_GEMM", "string", "global", "off", "openmx_mgpu_runtime.cu", "") \
    X(MGPU_GEMM_ASYNC_GATHER, "OPENMX_MGPU_GEMM_ASYNC_GATHER", "flag", "global", "1", "openmx_mgpu_gemm.cu", "") \
    X(MGPU_GEMM_BACKEND, "OPENMX_MGPU_GEMM_BACKEND", "string", "global", "bridge", "openmx_mgpu_gemm.cu", "") \
    X(MGPU_GEMM_FORCE_HOST, "OPENMX_MGPU_GEMM_FORCE_HOST", "flag", "global", "0", "openmx_mgpu_gemm.cu", "") \
    X(MGPU_GEMM_HOST_MB, "OPENMX_MGPU_GEMM_HOST_MB", "mib", "global", "256", "openmx_mgpu_gemm.cu", "") \
    X(MGPU_GEMM_HOST_RESULT_MB, "OPENMX_MGPU_GEMM_HOST_RESULT_MB", "mib", "global", "65536", "openmx_mgpu_gemm.cu", "") \
    X(MGPU_GEMM_MIN_N, "OPENMX_MGPU_GEMM_MIN_N", "int", "global", "8192", "openmx_mgpu_runtime.cu", "") \
    X(MGPU_GEMM_PAGEABLE, "OPENMX_MGPU_GEMM_PAGEABLE", "flag", "global", "0", "openmx_mgpu_gemm.cu", "") \
    X(MGPU_GEMM_RESERVE_MB, "OPENMX_MGPU_GEMM_RESERVE_MB", "mib", "global", "512", "openmx_mgpu_gemm.cu", "") \
    X(MGPU_GEMM_RESULT_HOST, "OPENMX_MGPU_GEMM_RESULT_HOST", "flag", "global", "0", "openmx_mgpu_gemm.cu", "") \
    X(MGPU_GEMM_RESULT_MB, "OPENMX_MGPU_GEMM_RESULT_MB", "mib", "global", "2048", "openmx_mgpu_gemm.cu", "") \
    X(MGPU_GEMM_TRAVERSAL, "OPENMX_MGPU_GEMM_TRAVERSAL", "string", "global", "auto", "openmx_mgpu_gemm.cu", "") \
    X(MGPU_GEMM_WORKSPACE_MB, "OPENMX_MGPU_GEMM_WORKSPACE_MB", "mib", "global", "1024", "openmx_mgpu_gemm.cu", "") \
    X(MGPU_MEMORY, "OPENMX_MGPU_MEMORY", "string", "global", "device", "openmx_mgpu_runtime.cu", "") \
    X(MGPU_NUM_GPUS, "OPENMX_MGPU_NUM_GPUS", "int", "global", "all visible", "openmx_mgpu_runtime.cu", "") \
    X(MGPU_SOLVER, "OPENMX_MGPU_SOLVER", "string", "global", "single", "openmx_mgpu_runtime.cu", "") \
    X(MGPU_SOLVER_MIN_N, "OPENMX_MGPU_SOLVER_MIN_N", "int", "global", "4096", "openmx_mgpu_runtime.cu", "") \
    X(MGPU_TUNE_TRIALS, "OPENMX_MGPU_TUNE_TRIALS", "int", "global", "2", "openmx_mgpu_runtime.cu", "") \
    X(MGPU_VERBOSE, "OPENMX_MGPU_VERBOSE", "int", "global", "0", "openmx_gpu_workers.c,openmx_mgpu_runtime.cu", "") \
    X(MIXH_GPU, "OPENMX_MIXH_GPU", "int", "global", "auto", "Mixing_H.c", "default auto: RMM-DIISH tier selection: unset=auto, 0=legacy, 1=force GPU tier, 2=force incremental CPU tier") \
    X(MIXH_GPU_RESERVE_MB, "OPENMX_MIXH_GPU_RESERVE_MB", "mib", "global", "max(total/10,1536)", "Mixing_H.c", "") \
    X(MIXH_GPU_VERBOSE, "OPENMX_MIXH_GPU_VERBOSE", "flag", "global", "0", "Mixing_H.c", "") \
    X(MIX_FAST, "OPENMX_MIX_FAST", "flag", "global", "1", "DIIS_Mixing_DM.c,DIIS_Mixing_Rhok.c,GR_Pulay_DM.c,Kerker_Mixing_Rhok.c,Mixing_V.c", "default 1: CPU fast paths of RMM-DIIS/DIISK/DIISV/GR-Pulay/Kerker (0 = legacy)") \
    X(NC_BACKTRANSFORM_BARRIER, "OPENMX_NC_BACKTRANSFORM_BARRIER", "flag", "global", "0", "backends/hip/src/Band_DFT_NonCol.c", "Opt in to a barrier around noncollinear band back-transforms.") \
    X(NC_DENSE_BUFFER_CACHE, "OPENMX_NC_DENSE_BUFFER_CACHE", "flag", "global", "0", "backends/hip/src/Band_DFT_NonCol.c", "Keep noncollinear band dense buffers; implied by device-eigenvector DM.") \
    X(NC_DEVICE_EIGENVECTOR_DM, "OPENMX_NC_DEVICE_EIGENVECTOR_DM", "flag", "global", "0", "backends/hip/src/Band_DFT_NonCol.c", "Opt in to noncollinear band DM generation from device-resident eigenvectors.") \
    X(NC_EIGEN_DEMOTE_SETHAM, "OPENMX_NC_EIGEN_DEMOTE_SETHAM", "flag", "global", "1", "backends/hip/src/DFT.c,backends/hip/src/Set_Hamiltonian.c", "Release orbital-table residency for noncollinear band dense eigensolves.") \
    X(NC_HAMILTONIAN_EIGENSOLVER, "OPENMX_NC_HAMILTONIAN_EIGENSOLVER", "string", "global", "", "backends/hip/src/Band_DFT_NonCol.c", "Set to hipsolver to select the HIP noncollinear Hamiltonian solver explicitly.") \
    X(NC_HIPSOLVER_TURN_RELEASE, "OPENMX_NC_HIPSOLVER_TURN_RELEASE", "flag", "global", "", "backends/hip/src/Band_DFT_NonCol.c", "Release hipSOLVER workspaces after a noncollinear band turn; default follows APU detection.") \
    X(NC_MAGMA_TURN_RELEASE, "OPENMX_NC_MAGMA_TURN_RELEASE", "flag", "global", "", "backends/hip/src/Band_DFT_NonCol.c", "Release MAGMA workspaces after a noncollinear band turn; default follows APU detection.") \
    X(OLPKIN_BESSEL_CACHE_MAX, "OPENMX_OLPKIN_BESSEL_CACHE_MAX", "int", "global", "128", "Set_OLP_Kin.c", "") \
    X(OLPKIN_RADIAL_GPU, "OPENMX_OLPKIN_RADIAL_GPU", "flag", "global", "0", "Set_OLP_Kin.c", "default 0: GPU radial integrals in Set_OLP_Kin (single thread required; usually slower — transfer-bound)") \
    X(ORBS_EVAL_PRECISION, "OPENMX_ORBS_EVAL_PRECISION", "string", "global", "df / fp64", "Set_Orbitals_Grid.c,backends/hip/src/Set_Orbitals_Grid.c", "default df / fp64: The arithmetic of the on-the-fly orbital evaluation (the device code of Set_Orbitals_Grid that the Hamiltonian and density tiles use when a rank ha...") \
    X(ORBS_GRID_GPU, "OPENMX_ORBS_GRID_GPU", "flag", "global", "1", "Set_Orbitals_Grid.c,backends/hip/src/Set_Orbitals_Grid.c", "") \
    X(ORBS_GRID_GPU_VERBOSE, "OPENMX_ORBS_GRID_GPU_VERBOSE", "flag", "global", "0", "Set_Orbitals_Grid.c,backends/hip/src/Set_Orbitals_Grid.c", "") \
    X(OTF_CHECK, "OPENMX_OTF_CHECK", "flag", "global", "0", "backends/hip/src/Set_Hamiltonian.c", "Diagnostic: compare on-demand and packed orbital matrix elements.") \
    X(SETDENSITY_GPU, "OPENMX_SETDENSITY_GPU", "flag", "global", "", "Set_Density_Grid_GPU.c,backends/hip/src/Set_Density_Grid_GPU.c", "") \
    X(SETDEN_FP64_CUDA, "OPENMX_SETDEN_FP64_CUDA", "flag", "global", "1", "set_hamiltonian_gpu.cu", "") \
    X(SETDEN_FP64_SPLIT_SPIN, "OPENMX_SETDEN_FP64_SPLIT_SPIN", "flag", "global", "auto", "set_hamiltonian_gpu.cu", "") \
    X(SETDEN_ONTHEFLY_MB, "OPENMX_SETDEN_ONTHEFLY_MB", "mib", "global", "128", "Set_Hamiltonian.c,backends/hip/src/Set_Hamiltonian.c", "default 128: Batch workspace bound (MiB) for on-the-fly density. The physical budget subtracts a 512 MiB node reserve, divides among on-the-fly ranks, then subt...") \
    X(SETHAM_BASE_GPU, "OPENMX_SETHAM_BASE_GPU", "flag", "global", "", "Set_Hamiltonian.c,backends/hip/src/Set_Hamiltonian.c", "default 0: GPU execution of the base combination of H (CPU is faster in practice)") \
    X(SETHAM_CUDA_KERNEL, "OPENMX_SETHAM_CUDA_KERNEL", "flag", "global", "1", "Set_Hamiltonian.c", "default 1: The dedicated CUDA kernel (0 = OpenACC kernel only; numerically identical)") \
    X(SETHAM_DEMOTE_ON_PRESSURE, "OPENMX_SETHAM_DEMOTE_ON_PRESSURE", "flag", "global", "1", "backends/hip/src/Set_Hamiltonian.c", "Allow resident Set_Hamiltonian tables to be evicted under eigensolver memory pressure.") \
    X(SETHAM_FP64_SPLIT, "OPENMX_SETHAM_FP64_SPLIT", "int", "global", "auto", "set_hamiltonian_gpu.cu", "") \
    X(SETHAM_FP64_TILED, "OPENMX_SETHAM_FP64_TILED", "flag", "global", "auto", "set_hamiltonian_gpu.cu", "") \
    X(SETHAM_GPU, "OPENMX_SETHAM_GPU", "flag", "global", "", "backends/hip/src/Set_Hamiltonian.c", "Enable HIP Hamiltonian matrix elements; automatic disables offload on oversubscribed devices.") \
    X(SETHAM_GPU_KOWNER_ONLY, "OPENMX_SETHAM_GPU_KOWNER_ONLY", "flag", "global", "0", "DFT.c,backends/hip/src/Set_Hamiltonian.c", "default 0: 1 restores the old k-owner-only GPU execution") \
    X(SETHAM_GPU_MAX_RANKS_PER_DEVICE, "OPENMX_SETHAM_GPU_MAX_RANKS_PER_DEVICE", "int", "global", "uncapped", "Set_Hamiltonian.c", "default uncapped: Explicit cap on ranks per wave") \
    X(SETHAM_GPU_RANK_OVERHEAD_MB, "OPENMX_SETHAM_GPU_RANK_OVERHEAD_MB", "mib", "global", "512", "Set_Hamiltonian.c", "") \
    X(SETHAM_GPU_RESERVE_MB, "OPENMX_SETHAM_GPU_RESERVE_MB", "mib", "global", "max(total/10,1536)", "Set_Hamiltonian.c", "") \
    X(SETHAM_GPU_RESIDENT, "OPENMX_SETHAM_GPU_RESIDENT", "flag", "global", "1", "Set_Hamiltonian.c", "") \
    X(SETHAM_GPU_RESIDENT_FLOOR_MB, "OPENMX_SETHAM_GPU_RESIDENT_FLOOR_MB", "mib", "global", "4096 or 16384", "Set_Hamiltonian.c", "") \
    X(SETHAM_GPU_SERIAL_WAVES, "OPENMX_SETHAM_GPU_SERIAL_WAVES", "flag", "global", "0", "Set_Hamiltonian.c", "default 0: 1 = run all waves on the GPU serially (default: wave 1 on GPU with remaining ranks on CPU concurrently) Under memory pressure on the A100 (whole-ra...") \
    X(SETHAM_GPU_TEST_NEED_FROM_CALL, "OPENMX_SETHAM_GPU_TEST_NEED_FROM_CALL", "int", "global", "1", "Set_Hamiltonian.c", "") \
    X(SETHAM_GPU_TEST_NEED_MB, "OPENMX_SETHAM_GPU_TEST_NEED_MB", "mib", "global", "0", "Set_Hamiltonian.c", "") \
    X(SETHAM_HIP_KERNEL, "OPENMX_SETHAM_HIP_KERNEL", "string", "global", "", "backends/hip/src/set_hamiltonian_hip.cu", "HIP matrix-elements kernel; scalar selects the reference kernel, otherwise use the tiled kernel.") \
    X(SETHAM_MATEL_SPLIT, "OPENMX_SETHAM_MATEL_SPLIT", "int", "global", "4", "set_hamiltonian_gpu.cu", "default 4: Blocks per atom pair along the grid points in the df matrix-element kernel (each block integrates whole 32-point tiles of its share and adds its pa...") \
    X(SETHAM_MATRIX_GPU, "OPENMX_SETHAM_MATRIX_GPU", "flag", "global", "1", "Set_Hamiltonian.c", "default 1: GPU execution of the dVH+Vxc+VNA grid integrals (0 = CPU)") \
    X(SETHAM_ONTHEFLY, "OPENMX_SETHAM_ONTHEFLY", "flag", "global", "1", "Set_Hamiltonian.c,backends/hip/src/Set_Hamiltonian.c", "default 1: A rank whose orbital tables do not fit on the device evaluates the orbitals of every batch of atom pairs on the device (the device code of Set_Orbi...") \
    X(SETHAM_ONTHEFLY_MARGIN_MB, "OPENMX_SETHAM_ONTHEFLY_MARGIN_MB", "mib", "global", "0", "Set_Hamiltonian.c", "") \
    X(SETHAM_ONTHEFLY_MB, "OPENMX_SETHAM_ONTHEFLY_MB", "mib", "global", "128", "Set_Hamiltonian.c,backends/hip/src/Set_Hamiltonian.c", "") \
    X(SETHAM_ONTHEFLY_PINNED_MB, "OPENMX_SETHAM_ONTHEFLY_PINNED_MB", "mib", "global", "1024", "backends/hip/src/Set_Hamiltonian.c", "Pinned host-memory cap in MiB for on-demand Hamiltonian staging.") \
    X(SETHAM_ONTHEFLY_PLAN_CACHE, "OPENMX_SETHAM_ONTHEFLY_PLAN_CACHE", "flag", "global", "1", "Set_Hamiltonian.c,backends/hip/src/Set_Hamiltonian.c", "default 1: The on-the-fly paths keep a plan per rank: the batches (which pairs, their offsets), the pair descriptors and the per-point index lists (Nc, and th...") \
    X(SETHAM_SIMD, "OPENMX_SETHAM_SIMD", "string", "global", "", "Set_Hamiltonian.c,backends/hip/src/Set_Hamiltonian.c,openmx_grid_assembly.c", "") \
    X(SETHAM_SIMD_TRACE, "OPENMX_SETHAM_SIMD_TRACE", "flag", "global", "", "openmx_grid_assembly.c", "") \
    X(SETHAM_STREAMING, "OPENMX_SETHAM_STREAMING", "int", "global", "1", "Set_Hamiltonian.c", "default 1: Streamed pair batches (256 MiB workspace per rank, OPENMX_SETHAM_STREAM_MB) when the device holds a single whole-rank allocation; 2 streams wheneve...") \
    X(SETHAM_STREAM_MB, "OPENMX_SETHAM_STREAM_MB", "mib", "global", "256", "Set_Hamiltonian.c", "") \
    X(SETHAM_TIMING, "OPENMX_SETHAM_TIMING", "int", "global", "0", "backends/hip/src/Set_Hamiltonian.c,backends/hip/src/set_hamiltonian_hip.cu", "Print HIP Hamiltonian timings; values above 1 include device-kernel timing.") \
    X(SETNL_BESSEL_CACHE_MAX, "OPENMX_SETNL_BESSEL_CACHE_MAX", "int", "global", "128", "Set_Nonlocal.c,backends/hip/src/Set_Nonlocal.c", "") \
    X(SETNL_OPENACC, "OPENMX_SETNL_OPENACC", "flag", "global", "1", "Set_Nonlocal.c", "default 1: The Set_Nonlocal GPU batch (0 = CPU path)") \
    X(SETNL_OPENMP, "OPENMX_SETNL_OPENMP", "flag", "global", "1", "backends/hip/src/Set_Nonlocal.c", "Enable the HIP/OpenMP nonlocal matrix path.") \
    X(SETPRO_DSVNA_GPU, "OPENMX_SETPRO_DSVNA_GPU", "flag", "global", "", "backends/hip/src/Set_ProExpn_VNA.c", "Override the VNA-projector derivative stage; otherwise inherit SETPRO_GPU and the device policy.") \
    X(SETPRO_GPU, "OPENMX_SETPRO_GPU", "flag", "global", "", "DFT.c,Set_ProExpn_VNA.c,backends/hip/src/Set_ProExpn_VNA.c", "default 1: The three Set_ProExpn_VNA batches (0 = CPU)") \
    X(SETPRO_HVNA_GPU, "OPENMX_SETPRO_HVNA_GPU", "flag", "global", "", "backends/hip/src/Set_ProExpn_VNA.c", "Override the VNA-projector energy stage; otherwise inherit SETPRO_GPU and the device policy.") \
    X(SETPRO_HVNA_TURNS, "OPENMX_SETPRO_HVNA_TURNS", "flag", "global", "1", "backends/hip/src/Set_ProExpn_VNA.c", "Use GPU turns for VNA-projector energy contractions.") \
    X(SETPRO_VERBOSE, "OPENMX_SETPRO_VERBOSE", "flag", "global", "0", "Set_ProExpn_VNA.c,backends/hip/src/Set_ProExpn_VNA.c", "default 0: One stderr line per rank and device stage of Set_ProExpn_VNA: the admission figures (need, budget, free memory, pair batch, rank turns), the batch...") \
    X(SETPRO_VNA23_GPU, "OPENMX_SETPRO_VNA23_GPU", "flag", "global", "", "backends/hip/src/Set_ProExpn_VNA.c", "Override the VNA23 force stage; otherwise inherit SETPRO_GPU and the device policy.") \
    X(SOLVE_TIMING, "OPENMX_SOLVE_TIMING", "flag", "global", "0", "backends/hip/src/Band_DFT_Col.c,backends/hip/src/Band_DFT_NonCol.c", "Print HIP band dense-solve timings.") \
    X(SOURCE_ROOT, "OPENMX_SOURCE_ROOT", "tool", "tool", "", "build/tools", "HIP Makefile: repository root of the staged build") \
    X(TOTALENERGY_GPU, "OPENMX_TOTALENERGY_GPU", "flag", "global", "1", "Total_Energy.c,Total_Energy_GPU.c,backends/hip/src/Total_Energy_GPU.c", "default 1: Master gate for the total-energy device kernels (including NC)") \
    X(UCELL_OLG_GPU, "OPENMX_UCELL_OLG_GPU", "flag", "global", "1", "truncation.c", "") \
    X(UCELL_OLG_GPU_VERBOSE, "OPENMX_UCELL_OLG_GPU_VERBOSE", "flag", "global", "0", "truncation.c", "") \
    X(XC_GRID_GPU, "OPENMX_XC_GRID_GPU", "flag", "global", "1", "Set_XC_Grid.c,backends/hip/src/Set_XC_Grid.c", "default 1: The GGA-PBE exchange-correlation potential on the grid (Set_XC_Grid, called by Set_Vpot every SCF step and by Total_Energy / Force) is evaluated on...") \
    X(ZHEEVD_SELFTEST, "OPENMX_ZHEEVD_SELFTEST", "flag", "global", "0", "backends/hip/src/Band_DFT_Col.c,backends/hip/src/DFT.c,openmx.c", "") \
    X(EXT_PMI_LOCAL_RANK, "PMI_LOCAL_RANK", "int", "local", "", "backends/cpu/device.c,backends/hip/src/set_hip_default_device_from_local_rank.c,set_cuda_default_device_from_local_rank.c", "PMI: node-local rank") \
    X(EXT_PMI_LOCAL_SIZE, "PMI_LOCAL_SIZE", "int", "local", "", "backends/cpu/device.c,backends/hip/src/set_hip_default_device_from_local_rank.c,set_cuda_default_device_from_local_rank.c", "PMI: ranks on this node") \
    X(EXT_PMI_RANK, "PMI_RANK", "string", "local", "", "backends/hip/src/gemmul8_bridge.cu,gemmul8_bridge.cu", "PMI: world rank (GEMMul8 statistics label)") \
    X(EXT_SLURM_LOCALID, "SLURM_LOCALID", "int", "local", "auto", "backends/cpu/device.c,backends/hip/src/Divide_Conquer.c,backends/hip/src/set_hip_default_device_from_local_rank.c,set_cuda_default_device_from_local_rank.c", "Slurm: node-local rank") \
    X(EXT_SLURM_NTASKS_PER_NODE, "SLURM_NTASKS_PER_NODE", "int", "local", "0", "backends/cpu/device.c,backends/hip/src/Cluster_DFT_Col.c,backends/hip/src/gemmul8_bridge.cu,backends/hip/src/set_hip_default_device_from_local_rank.c,gemmul8_bridge.cu,set_cuda_default_device_from_local_rank.c", "Slurm: ranks on this node")

#endif /* OPENMX_ENV_REGISTRY_H */

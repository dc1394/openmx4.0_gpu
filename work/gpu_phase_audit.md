# OpenMX 4.0 GPU v2.0 — GPU phase / fallback / env-knob / solver-path audit

- Repo: `/work/1/GPU2026/dc1394/openmx4.0_gpu`, checkout = commit `cd5f0d5` (branch main, HEAD f5fb5b3 at audit time; audited working tree).
- All file references are relative to `source/`. Every claim was verified in the source at the given `file:line`.
- Scope: WBS item 1 ("GPU phase / fallback / env knob / solver path 一覧") for the CPC paper.

---

## 0. Global routing primitives (read first)

| Item | Value | Evidence |
|---|---|---|
| `scf_eigen_lib_flag` enum | `ELPA1=1, ELPA2=2, GPUSOLVER=3, GPUSOLVER2=4` | `openmx_common.h:2807-2813` |
| `scf.eigen.lib` keyword map | `gpusolver`→GPUSOLVER (default, i_vec[0]), `lapack`→0, `elpa1`→ELPA1, `elpa2`→ELPA2, `cusolver`→deprecated alias of gpusolver (warning), `gpusolver2`→GPUSOLVER2 | `Input_std.c:783-792` |
| gpusolver2 folding | `GPUSOLVER2` sets `gpusolver2_flag=1` and folds `scf_eigen_lib_flag` back to `GPUSOLVER` | `Input_std.c:800-803`, `openmx_common.h:2814-2819` |
| gpusolver2 restriction | Only `scf.EigenvalueSolver=cluster` (Solver==2); otherwise printf + `MPI_Finalize/exit` | `Input_std.c:1183-1191` |
| `scf.gemmul8.enable` | default on; off routes all GPU dense GEMMs to plain cuBLAS via `openmx_gemmul8SetEnabled()` | `Input_std.c:812-822`, `gemmul8_bridge.cu:359-362` |
| Solver numbering | 1=Recursion, 2=Cluster, 3=Band, 4=NEGF, 5=DC, 6=GDC, 7=Cluster-DIIS, 8=Krylov, 9=Cluster2, 10=EGAC, 11=DC-LNO, 12=Cluster-LNO | `Input_std.c:740-750`, `DFT.c:933-936` |
| Global dense crossover macro | `GPU_CPU_SWITCH_NUM = 1000` (used by DMmu/CWF post-processing paths and as `DFT_GPU_DenseSwitchNum()` default for other solvers) | `openmx_common.h:4142`, `DFT.c:45-53` |
| Per-rank GPU probe | `gpu_rank_device_usable()`: CUDA context + OpenACC init behind an flock'd probe; `OPENMX_GPU=0` disables GPU for the whole run; failure banner on stderr; failing rank uses host paths | `set_cuda_default_device_from_local_rank.c:122-217` |
| Rank→device map | block map by default; `OPENMX_GPU_RANK_MAP=m...` switches to `local_rank % device_count` | `set_cuda_default_device_from_local_rank.c:75-93` |
| GPU init & demotion | `DFT_GPU_DeviceInit`: MPI_COMM_TYPE_SHARED per-node assignment; if ANY rank lacks a usable device (`MPI_MIN` allreduce), whole run demotes `scf_eigen_lib_flag=ELPA2` and `gpusolver2_flag=0` | `DFT.c:73-203` (demotion `DFT.c:158-161`) |
| `scf.Gpu.Num` | input keyword, default 30; caps GPUs used per node | `Input_std.c:808`, `DFT.c:119-125`, `set_cuda_default_device_from_local_rank.c:156,230` |
| Rank selection for Set_Hamiltonian GPU | `DFT_SetHamiltonianOpenACCSelectedRank` / `...WorkSelectedRank` wired via `Set_Hamiltonian_Set_OpenACC_Rank_Selected()`; work-rank default = every rank with a local H segment, `OPENMX_SETHAM_GPU_KOWNER_ONLY=1` restores legacy k-owner-only policy | `DFT.c:262-343` (knob `DFT.c:319`), `DFT.c:353-354`, `Set_Hamiltonian.c:87-109` |
| GPU-phase need registry | `OpenMX_GpuPhaseNeed_Register(name, group_bytes)` — 8 slots; consulted by Set_Hamiltonian residency (`Max` / `MaxPrefixed("band_")`) | `openmx_common.c:20252-20280`, `Set_Hamiltonian.c:550-551`; registrants: `Band_DFT_Col.c:806` ("band_col"), `Band_DFT_NonCol.c:1158-1168` ("band_noncol_ev/vec/dm"), `Set_Density_Grid_GPU.c:1160` ("density_grid_local") |

---

## 1. Dense-solver dispatch table

### 1.1 scf.eigen.lib value → path, per solver

Common: `lapack(0)` / `elpa1(1)` / `elpa2(2)` select CPU (ScaLAPACK/ELPA) code in all solvers below; inside the distributed branch `scf_eigen_lib_flag==1` calls `solve_evp_*` (ELPA1) and `==2 || ==GPUSOLVER` calls `elpa_solve_evp_*_2stage_double_impl` (ELPA2) — i.e. **GPUSOLVER's CPU fallback executes ELPA2 kernels** (e.g. `Band_DFT_Col.c:3413-3426`, `Cluster_DFT_NonCol.c:2597-2600,2829-2836`).

| Solver | GPU-dense gate (all conditions ANDed) | Threshold macro (value) / env override | Memory preflight | Fallback + banner |
|---|---|---|---|---|
| Cluster col (`Cluster_DFT_Col.c`) | `scf_eigen_lib_flag==GPUSOLVER && gpusolver2_flag==0 && Cluster_DFT_Col_GpuSwitchNum()<=n` → `ClusterCol_GpuDiagFits()` returns 1 (concurrent) or 2 (serialized) → `ClusterCol_GpuSolverRootDensePath` | `CLUSTER_GPU_CPU_SWITCH_NUM 400` (`:577`) / `OPENMX_CLUSTER_GPU_SWITCH_NUM` (`:584`) | Owner rank RESERVES full dense buffers + cusolver workspace (`ClusterCol_OwnerReserveProbe :994-1017`), plus `OPENMX_CLUSTER_GPU_DIAG_RESERVE_MB` margin (default 1024 MiB, `:943-951`); collective over `mpi_comm_level1` (`:1076`); serialized 2-spin tier if only one owner fits (`:1087-1096`), gated by `OPENMX_CLUSTER_GPU_DIAG_SERIAL` (default 1; 0=off, 2=skip concurrent probe; `:1022-1024,1046-1049`) | ELPA/ScaLAPACK branch (`:1766+`). Banners: `:1006` "could not reserve", `:1085` "cannot hold both spin owners'... serially", `:1104` "Falling back to the ELPA/ScaLAPACK diagonalization. Force the GPU path with OPENMX_CLUSTER_GPU_DIAG=1 or lower OPENMX_CLUSTER_GPU_DIAG_RESERVE_MB." Force knob `OPENMX_CLUSTER_GPU_DIAG` (`:1021`, =0 banner `:1036`) |
| Cluster col, gpusolver2 | `gpusolver2_flag==1`: distributed ELPA-GPU eigensolver `openmx_gs2_eigen_real` (`:1793-1800, 2041-2049`) + COSMA `openmx_gs2_pdgemm_` for the transforms (`:2006-2019, 2086-2087`) | n/a (always distributed when selected) | Per-call GPU/CPU verdict inside bridge (sec 1.4) | gs2 eigen failure = `MPI_Abort` after `Cluster_DFT_Col: the gpusolver2 overlap|Hamiltonian eigensolver failed (info=%d)` (`:1798,2047`); the bridge itself degrades ELPA→CPU kernels / COSMA→ScaLAPACK per memory verdict (sec 1.4) |
| Cluster NC (`Cluster_DFT_NonCol.c`) | `use_gpusolver_direct_cluster_dm = GPUSOLVER && gpusolver2_flag==0 && Cluster_DFT_NonCol_GpuSwitchNum()<=n2 && mode=="scf" && MO_fileout!=1 && xanes_calc!=1 && xanes_gs_fileout!=1 && !cal_partial_charge && !Dos_fileout && !DosGauss_fileout && ClusterNonCol_GpuDiagFits(...)` | `CLUSTER_NONCOL_GPU_CPU_SWITCH_NUM 400` vs `n2=2n` (`:1720`) / `OPENMX_CLUSTER_NONCOL_GPU_SWITCH_NUM` (`:1727`) | Host_ID reserves one device arena for the whole dense solve (`ClusterNonCol_DenseArena_TryEnsure :1648-1688`) + zheevdx workspace + transient overlap-diag workspace + `OPENMX_CLUSTER_GPU_DIAG_RESERVE_MB` (default 1024 MiB, `:1634-1643`); collective (`:1858`) | ELPA/ScaLAPACK branch. Banner `:1843-1852` "The dense owner could not reserve its GPU diagonalization buffers (...); falling back to the ELPA/ScaLAPACK diagonalization. Force the GPU path with OPENMX_CLUSTER_GPU_DIAG=1 or lower OPENMX_CLUSTER_GPU_DIAG_RESERVE_MB." Force `OPENMX_CLUSTER_GPU_DIAG` (`:1754`; =0 banner `:1766`; =1 aborts with diagnostic if reservation fails `:1778-1786`) |
| Cluster NC, gpusolver2 | `openmx_gs2_eigen_real` (S, `:2588-2596`) / `openmx_gs2_eigen_complex` (H, `:2823-2830`) + `openmx_gs2_pdgemm_` transforms (`:2698-2796`) | n/a | bridge verdict | failure = printf (`:2593,2829`) + `MPI_Abort` |
| Band col (`Band_DFT_Col.c`) | `use_gpusolver_dense = GPUSOLVER && Band_DFT_Col_GpuSwitchNum()<=n` (`:2398`), then `&& BandCol_GpuDenseFits(...)` (`:2429-2435`); requires full dense local matrices (`n==na_rows==na_cols`) else abort `"GPUSOLVER band path requires full dense matrices"` (`:2876-2879`) | `BAND_GPU_CPU_SWITCH_NUM 800` (`:554`) / `OPENMX_BAND_GPU_SWITCH_NUM` (`:561`) | Per k-owner rank: 3 dense n×n + cusolver `Xsyevdx` workspace (queried) + GEMMul8 workspace + staging transients + `OPENMX_BAND_GPU_RANK_OVERHEAD_MB` (256 MiB) must fit beside reserve max(total/10, 1536 MiB) (`BandCol_GpuTurnRequiredBytes :434-479`, reserve `:481-501`); collective MIN over `mpi_comm_level1`, sticky per (n, SCF cycle) (`:583-670`) | ScaLAPACK/ELPA branch (ELPA1/ELPA2 calls `:3413-3426,3547-3558,4745-4756,4838-4849`). Banner `:654-662` (sec 4). Force `OPENMX_BAND_GPU_DIAG` (`:587`; =0 banner `:598`) |
| Band NC (`Band_DFT_NonCol.c`) | `BandNonCol_UseDenseGpuMatrix = GPUSOLVER && Band_DFT_NonCol_GpuSwitchNum()<=n2 && na_rows==n && na_cols==n && na_rows2==n2 && na_cols2==n2 && BandNonCol_gpu_dense_verdict` (`:480-485`); verdict from `BandNonCol_GpuDiagFits` (`:981-1069`) | `BAND_NONCOL_GPU_CPU_SWITCH_NUM 800` vs `n2=2n` (`:462`) / `OPENMX_BAND_NONCOL_GPU_SWITCH_NUM` (`:469`) | max(root dense bytes, root DM bytes) + reserve max(total/10,1536 MiB) per k-owner; collective, sticky (`:1010-1043`); additionally a per-device group check can serialize k-worlds (banner `:1332`) | ScaLAPACK/ELPA branch. Banner `:1054-1062` (same wording as col, `<Band_DFT_NonCol>`). Force `OPENMX_BAND_GPU_DIAG` (`:985`; =0 banner `:996`) |
| DC-LNO (`Divide_Conquer_LNO.c`) | `use_gpu_accel = (scf_eigen_lib_flag==GPUSOLVER)` (`:1355`); per-atom task: `use_gpu_task = use_gpu_accel && NUM >= DCLNO_GpuEigenThreshold()` (`:2133-2134`; NC: `:4135-4136` with `Eigen_MPI_flag==0`) → `DCLNO_Solve_Col_GpuSolver` / `DCLNO_Solve_NonCol_GpuSolver` (`:2139,4149`) | col: `GPU_CPU_SWITCH_NUM2 800` (`:31,34`) / `OPENMX_DCLNO_GPU_THRESHOLD` (`:42`); NC: `DCLNO_NONCOL_GPU_CPU_SWITCH_NUM 400` vs `2*(Anum-1)` (`:56`) / `OPENMX_DCLNO_NONCOL_GPU_THRESHOLD` (`:64`) | `DCLNO_GpuGroupMemoryFits`: group-summed need (7 matrices + GEMMul8 ws + 256 MiB/rank) vs group-min free minus fixed 1536 MiB reserve, collective over the device group (`:1034-1082`); failing → `use_gpu_accel=0` for the whole atom loop (`:1531-1534`, NC `:2887-2890`) | CPU per-atom solve `DCLNO_Solve_Col_Local` (`:2146-2150`) / NC CPU flow. Banner `:1067-1073` "would need ... using the CPU eigensolver instead." Enable banners `:1363,2765`. Note: the node-local shared-GPU proxy is compiled out (`DCLNO_ENABLE_SHARED_GPU_PROXY 0`, `:121`, used `:1359`) |

DC (Solver==5, `Divide_Conquer.c`, not in the plan's list but sharing the pattern): per-atom gate `GPUSOLVER && !GemmDisabled && !EigenDisabled && DC_GPU_Threshold()<=NUM` (`:2201-2204`); thresholds `DC_GPU_CPU_SWITCH_NUM 400` / `DC_NONCOL_GPU_CPU_SWITCH_NUM 300` vs `2*NUM` (`:42-43`, env `:51,:70`, NC use `:4069`); GEMMul8→cuBLAS→CPU degradation per rank (`:645-683`); S-cache banner `:842`.

### 1.2 DFT.c startup routing (lines 39-203)

- `DFT_GPU_DenseSwitchNum()` (`DFT.c:42-52`): Solver==3 → band (NC/col) switch-num helper; Solver==2 → cluster helpers; else `GPU_CPU_SWITCH_NUM` (1000).
- Small-system notice (GPU kernels stay on, dense eigensolver falls back): banner `DFT.c:86` — emitted when `Solver!=5 && Solver!=8 && Solver!=11 && basis_count < DFT_GPU_DenseSwitchNum()` (`DFT.c:84-88`).
- Whole-run demotion to ELPA2 with reason codes 1-5 (cudaGetDeviceCount fail / 0 devices / OpenACC 0 devices / cudaSetDevice fail / context-module init impossible): banners `DFT.c:176-196`.
- Per-iteration solver banner: `<%s>  Solving the eigenvalue problem%s...` with ` (GPU-accelerated)` iff `DFT_GPU_EigensolverActive()` (`DFT.c:939-941`; active = GPUSOLVER && (Solver∈{5,8,11} or dim ≥ switch), `DFT.c:205-210`); one-time CPU-silence note `DFT.c:946-947`.

### 1.3 Band all_knum structure and jobz/NOVECTOR

- `all_knum = MPI_Allreduce(num_kloop0, MPI_PROD)`; ==1 means every k-world holds exactly one k point (`Band_DFT_Col.c:2798-2808`; special-case `SpinP_switch==1 && numprocs0==1` forces `all_knum=0`, `:2806-2808`).
- `all_knum==1`: single diagonalization pass computes eigenvectors too; `use_setham_packed_cache` may reuse Set_Hamiltonian's packed H/S (`:2812-2822`); DM built directly (`:3993-4464`); CPU path computes vectors via `pzgemm` + EVec1 exchange (`:3597-3630`).
- `all_knum!=1`: first pass eigenvalues-only, second pass (after chemical potential) re-diagonalizes for vectors + DM (`:4490+`; GPU turn ladder repeated `:4552+`).
- **NOVECTOR knob**: first (all_knum!=1) GPU pass uses `jobz=CUSOLVER_EIG_MODE_NOVECTOR` unless `OPENMX_BAND_SYEVDX_NOVECTOR=0` — `BandCol_EigenvaluesOnlyNoVectors` (`Band_DFT_Col.c:303-316`), applied via `want_vectors = build_eigenvectors || !BandCol_EigenvaluesOnlyNoVectors()` (`:2021`), jobz set `:1838,1888`. NC identical: `Band_DFT_NonCol.c:206-217`, applied `:2360`, jobz `:358`.
- GPU turn scheduling (both band solvers): concurrency cap = min(device group, `OPENMX_BAND_GPU_MAX_CONCURRENT_K` → alias `OPENMX_BAND_GPU_TURN_GROUP` → alias `OPENMX_BAND_GPU_MAX_RANKS_PER_DEVICE`, default INT_MAX = no cap) (`Band_DFT_Col.c:505-533`, `Band_DFT_NonCol.c:246-253`); within a group turns are serialized by default (`OPENMX_GPUSOLVER_SERIAL_GPU_TURNS`, default 1; `Band_DFT_Col.c:224-235`, `Band_DFT_NonCol.c:228`); plan banner `Band_DFT_Col.c:851`, `Band_DFT_NonCol.c:1218`.

### 1.4 gpusolver2 bridge (`elpa_cosma_bridge.c`)

- ELPA (NVIDIA GPU kernels) + COSMA; init banner `:218` `<DFT> gpusolver2: ELPA (API %d) with NVIDIA GPU kernels + COSMA initialized`.
- Eigensolver GPU/CPU verdict per configuration: need = `OPENMX_GS2_GPU_FACTOR`(3) × local matrix + `OPENMX_GS2_GPU_FIXED_MB`(128), × ranks-per-device + `OPENMX_GS2_GPU_RESERVE_MB`(512) vs free (`:86-93,150-179`); forced by `OPENMX_GS2_ELPA_GPU` (unset=auto, 0=CPU kernels, 1=GPU; `gs2_forced_mode` returns -1 when unset) (`:234,290-301`); solver stage via `OPENMX_GS2_ELPA_SOLVER` (=1 → ELPA 1stage, default 2stage; `:333-340`). Report banners `:190,194` (printed when verdict=CPU or `OPENMX_GS2_GPU_VERBOSE`).
- pdgemm/pzgemm: COSMA(GPU) vs ScaLAPACK(CPU) per call; bound = streams × tile products with `OPENMX_GS2_GPU_TILE`(1024) exported to `COSMA_GPU_MAX_TILE_{M,N,K}` (`:460-486`); forced by `OPENMX_GS2_GEMM_GPU` (`:539`); transition banners `:515,519`.
- Note (UNVERIFIED external behavior): `openmx_common.h:2814-2816` comment says "DLA-Future (eigensolver) and COSMA"; the code in `elpa_cosma_bridge.c` implements ELPA + COSMA. Paper text should follow the code.

### 1.5 Density-matrix construction knobs (per solver)

| Path | Gate | Evidence |
|---|---|---|
| Cluster col DM | `ClusterCol_UseGpuDM(n)`: default on when `ClusterCol_UseGpuAccel(n)` (= GPUSOLVER && switch<=n && rank selected); `OPENMX_CLUSTER_GPU_DM=0` → original CPU loop | `Cluster_DFT_Col.c:595-611`, used `:3139` |
| Cluster NC DM | part of the direct dense path `use_gpusolver_direct_cluster_dm` (no separate env knob) | `Cluster_DFT_NonCol.c:2397-2403` |
| Band col DM (ELPA-fallback loop) | `BandCol_UseGpuFallbackDM(n)`: GPUSOLVER && switch<=n, `OPENMX_BAND_GPU_DM=0` disables; per-k soft upload (needs free ≥ panel + 256 MiB else CPU loop) | `Band_DFT_Col.c:1523-1560`, used `:4165,4934` |
| Band NC DM (ELPA-fallback loop) | `BandNonCol_UseGpuFallbackDM(n2, evec_bytes)`: GPUSOLVER && switch<=n2, `OPENMX_BAND_NONCOL_GPU_DM=0` disables; refuses if evec panel + 256 MiB > free | `Band_DFT_NonCol.c:1921-1943` |
| Band GPU-dense flows | DM accumulated on device inside the turn ladder (col `:4552+`; NC via `AccumulateDMRootDenseK`, comment `:1921-1923`) | ibid. |

---

## 2. GPU phase table (G/H/F classification)

Classes: **G** = GPU fast path when its gate holds; **H** = hybrid (GPU + CPU cooperate by design); **F** = can fall back per-rank/per-call at runtime. A phase can carry several letters.

| Phase | Entry condition for GPU path | Fallback behaviour | Banners (stdout unless noted) | Class |
|---|---|---|---|---|
| **Set_Hamiltonian** — matrix elements (`Set_Hamiltonian.c` + `set_hamiltonian_gpu.cu`) | `Set_Hamiltonian_OpenACC_Enabled()` (= GPUSOLVER && `gpu_rank_device_usable()`, `:249-252`) && `OPENMX_SETHAM_MATRIX_GPU`≠0 (default 1, `:254-260`) && work-rank selected (`:1691-1692`); per-device turn plan admits ranks whose summed need (incl. `OPENMX_SETHAM_GPU_RANK_OVERHEAD_MB`, default 512 MiB, `:1996,2031`) fits beside reserve max(total/10,1536 MiB) (`OPENMX_SETHAM_GPU_RESERVE_MB` raise-only, `:302-318`); rank cap `OPENMX_SETHAM_GPU_MAX_RANKS_PER_DEVICE` (default INT_MAX, `:320-337`) | Ranks outside the first wave: CPU `Calc_MatrixElements_dVH_Vxc_VNA_CPU` (default hybrid) or serialized later GPU waves with `OPENMX_SETHAM_GPU_SERIAL_WAVES=1` (`:339-349,1746-1775`). Rank-level: CUDA/OpenACC init failure → stderr `:484`; memory query failure → stderr `:515`. Inside CUDA kernel: stale-error clear + per-batch defer to OpenACC kernel (`set_hamiltonian_gpu.cu:139-149,183-192`); kernel selectable via `OPENMX_SETHAM_CUDA_KERNEL` (default 1, `:263-267`) | Plan: `:683` `<Set_Hamiltonian> GPU device %d: %d Hamiltonian rank(s), GPU concurrency=%d, %s=%d, peak=%.3f GiB, free=%.3f GiB, reserve=%.3f GiB` (%s = "serialized waves" or "CPU fallback ranks"); phase banner `:1483` `<Set_Hamiltonian>  Hamiltonian matrix for VNA+dVH+Vxc%s...` (suffix ` (GPU-accelerated)`) | G+H+F |
| **Set_Hamiltonian** — resident cache | `OPENMX_SETHAM_GPU_RESIDENT` (default 1, `:352-361`); admitted only when every device rank is in one wave and budget = free − floor covers class bytes; floor `OPENMX_SETHAM_GPU_RESIDENT_FLOOR_MB` default 4096 MiB with registered phase need / 16384 MiB without (`:363-380`); test hooks `OPENMX_SETHAM_GPU_TEST_NEED_MB` / `_FROM_CALL` (default call 1, `:385-395`) | Mid-step eviction when a registered GPU phase needs the memory: release + stay transient for the MD step (`:1778-1792`) | `:920` `<Set_Hamiltonian> GPU resident cache: %d rank(s) hold %.3f GiB (%s) on device %d across SCF iterations`; `:1786` `<Set_Hamiltonian> GPU resident cache released: a GPU phase needs %.3f GiB on device %d; staying transient for this MD step` | G+F |
| **Set_Hamiltonian** — base add (H0+HNL→H) | opt-in `OPENMX_SETHAM_BASE_GPU=1`; forced off for Zeeman/xmcd (`:705-716`) | default CPU loop | (suffix of `:1483` banner covers it) | G (opt-in) |
| **Set_Nonlocal** (`Set_Nonlocal.c`) | `SetNonlocalUseOpenACC()` = GPUSOLVER && device usable; `OPENMX_SETNL_OPENACC=0` opts out (`:249-256`) | acc_malloc arena; on NULL the Bessel chunk shrinks and retries; if even the smallest footprint fails the function returns 0 and the host path runs (silent, by design) (`:272-279`) | phase label only: `DFT.c:544` `<MD=%2d>  Calculation of the nonlocal matrix%s` (suffix when GPUSOLVER) | G+F (silent) |
| **Set_ProExpn_VNA** (`Set_ProExpn_VNA.c`) | 3 device batches, each gated by GPUSOLVER + collective `OPENMX_SETPRO_GPU` (default 1; rank-0 value broadcast, `:140-153`): HVNA (`:168-169`), DS_VNA (`:761-762`), VNA23 (`:1479-1480`); per-node-rank memory checks (HVNA: (free−256 MiB)/node_ranks > need, `:201-217`; DS_VNA: exact table-sized estimate, `:923-955`; VNA23: fixed 512 MiB need, `:1487-1493`) | per-batch return 0 → CPU path for that batch | stderr `:210` `Set_ProExpn_VNA GPU: not enough free device memory; CPU fallback.`; stderr `:943` `Set_ProExpn_VNA DS_VNA GPU: needs %.3f GiB per rank, %d rank(s) share this device and only %.3f GiB is free; CPU fallback.`; VNA23 falls back silently. Phase label `DFT.c:560` with suffix from `DFT_SetProExpnVNAUseGPU` (`DFT.c:59-68`) | G+F |
| **Set_OLP_Kin** (`Set_OLP_Kin.c`) | radial-integral OpenACC batch only; gate `SetOLPKinRadialGpuEnabled() && SetOLPKinUseOpenACC() && Nthrds==1` (`:399`); `OPENMX_OLPKIN_RADIAL_GPU` is **opt-in (default 0)** (`:52-59`); Bessel cache `OPENMX_OLPKIN_BESSEL_CACHE_MAX` default 128 (`:77,79-90`) | default = CPU (opt-in GPU); no memory preflight banner | phase label `DFT.c:536` `<MD=%2d>  Calculation of the overlap matrix%s`, suffix from `DFT_SetOLPKinUseGPU()` = GPUSOLVER && omp_get_max_threads()==1 (`DFT.c:54-57`). **Discrepancy:** the suffix does not consult `OPENMX_OLPKIN_RADIAL_GPU`, so with defaults it can print " (GPU-accelerated)" while the radial batch stays on CPU | G (opt-in) |
| **Set_Density_Grid** (`Set_Density_Grid.c` + `Set_Density_Grid_GPU.c`) | Tier 1 "local": every rank integrates its own atoms; `Set_Density_Grid_GPU_Local_Prepare` gate: `OPENMX_DENSITY_GRID_GPU_LOCAL` (0=off, 1=auto: requires Set_Hamiltonian tables resident, 2=always) && `OPENMX_DENSITY_GRID_GPU` (default via legacy alias `OPENMX_SETDENSITY_GPU`, default 1) && GPUSOLVER && Solver∈{2,3} && Cnt_switch==0 && Cnt_kind∈{0,1} && SpinP_switch∈{0,1,3} && device usable (`Set_Density_Grid_GPU.c:1182-1236`); collective MIN (`Set_Density_Grid.c:84-96`). Tier 2 "service": single owner rank builds the full B partition; same input gate evaluated on owner (`Set_Density_Grid_GPU.c:850-863`), owner needs cache bytes + `OPENMX_DENSITY_GRID_GPU_RESERVE_MB` (256) free (`:873-925`); device residency across iterations opt-in `OPENMX_DENSITY_GRID_GPU_PERSISTENT` (default 0, `:958`) | Tier 1 fail → Tier 2; Tier 2 fail → original CPU/OpenMP loop (`Set_Density_Grid.c:98-105`) | `:91` `<Set_Density_Grid> distributed GPU quadrature: every rank integrates its own atoms`; stderr `Set_Density_Grid_GPU.c:916` `Set_Density_Grid GPU owner needs %.3f GiB plus %.3f GiB reserve, but only %.3f GiB is free; CPU fallback.` | G+F (two tiers) |
| **DM construction** | see sec 1.5 | per-k / per-call CPU loop | (no dedicated banners) | G+F |
| **Mixing** (`Mixing_H.c`, `DIIS_Mixing_Rhok.c`, `DIIS_Mixing_DM.c`, `GR_Pulay_DM.c`, `Kerker_Mixing_Rhok.c`, `Mixing_V.c`) | RMM-DIISH GPU history arena: `MixH_GpuPreflight` — `OPENMX_MIXH_GPU` unset=auto (arena + reserve `OPENMX_MIXH_GPU_RESERVE_MB` default max(total/10,1536 MiB) must fit group-wide, `Mixing_H.c:2202-2306`); =1 force GPU (alloc failure aborts, `:2311-2318`); =0 legacy CPU; =2 incremental CPU. `OPENMX_MIX_FAST` (default 1 under GPUSOLVER, 0 otherwise) enables cached-Gram incremental **CPU** mixing math in the five Rhok/DM/V mixers (`DIIS_Mixing_Rhok.c:71-76`, `DIIS_Mixing_DM.c:42`, `GR_Pulay_DM.c:31`, `Kerker_Mixing_Rhok.c:36`, `Mixing_V.c:48`) | GPU arena does not fit → incremental CPU Pulay (verdict 2) | `:2326` `<Mixing_H> The RMM-DIISH residual history (%.1f MiB per device) does not fit on the GPU (%.1f MiB free); using the incremental CPU Pulay mixing instead. OPENMX_MIXH_GPU=0 restores the legacy CPU mixing, OPENMX_MIXH_GPU=1 forces the GPU path.`; verbose-only (`OPENMX_MIXH_GPU_VERBOSE`): `:2237` forced-incremental, `:2343` `<Mixing_H> GPU RMM-DIISH mixing engaged (%d slots x %.1f MiB per rank).` | G+F |
| **Total_Energy** (`Total_Energy.c` + `Total_Energy_GPU.c`) | `TotalEnergyUseOpenACC()` = `OPENMX_TOTALENERGY_GPU` (default 1) && GPUSOLVER && device usable (`Total_Energy.c:67-81`); used at `:1346,1937,2075,2408,4425` | gate false → CPU loops. No runtime memory fallback: host malloc failure prints and `exit(1)` (`Total_Energy_GPU.c:390-393,788,926`; `Total_Energy.c:1375`) | force-part banners `Total_Energy.c:1028-1030,1607-1609` `  Force calculation #6%s` / `#7%s` (suffix ` (GPU-accelerated)`); phase label `DFT.c:2466` `<MD=%2d>  Total Energy` | G |
| **Force** (`Force.c`) | master gate `use_force_openacc = GPUSOLVER && OPENMX_FORCE_GPU (collective, default 1)` (`:2188-2190`); Force3: `OPENMX_FORCE3_GPU` (default 1) + budget probe ((free−256 MiB)/node_ranks over resident tables + floor 192 MiB, cap 384 MiB, `:860-902`), on-device orbital derivatives `OPENMX_FORCE3_GPU_DORB` (default 1, caps L0≤3, Mul≤8, uncontracted, `:904-927`); Force4 (ProExpn_VNA==0): GPU reduction under master gate (`:2191-2193`); Force4B: `OPENMX_FORCE4B_GPU` (default 1, `:7250`) but the OpenACC Force4B path is **hard-disabled in code** (`const int use_force4b_openacc = 0`, `:9037`), fused CPU trace `OPENMX_FORCE4B_FUSED` (default 1, `:9038-9039`); HNL: `OPENMX_FORCEHNL_GPU` (default 1) restricted to scalar-relativistic real branches (`:8027-8036`) | each sub-force falls back to its CPU trace when the gate/probe fails (return 0 pattern, e.g. `:872-873,7249-7251,8027-8029`) | `  Force calculation #3%s` (`:3561-3564`), `  Force calculation #4%s` with ` (GPU reduction)` / ` (GPU-accelerated)` (`:3588-3593`); phase label `DFT.c:2327` `<MD=%2d>  Force calculation`; profile lines via `OPENMX_FORCE_PROFILE` (`:2185-2186`, `Force_profile_report :80`) | H+F |
| **Krylov** (`Krylov.c`) | `Krylov_GPU_Enabled()` = compile-time `KRYLOV_ENABLE_GPU 1` && GPUSOLVER (`:27,117-124`); per-op thresholds: GEMM offload if m·n·k ≥ `OPENMX_KRYLOV_GPU_DGEMM_MIN_FLOPS` (default 5.0e7, `:31,175`), eigen offload if n ≥ `OPENMX_KRYLOV_GPU_EIGEN_MIN` (default 300, `:30,181,391,875`); fused projected solve `OPENMX_KRYLOV_GPU_FUSED` (default 1, `:187,800`); KU basis-panel device cache `OPENMX_KRYLOV_GPU_KU_CACHE` (default 1, `:511-517`) with reserve `OPENMX_KRYLOV_GPU_KU_RESERVE_MB` (default 4096 MiB, `:519-529`) | below thresholds → CPU BLAS/LAPACK; cusolver failure → per-call LAPACK fallback (stderr `:470-475` "…falling back to LAPACK.", `:771`); workspace-pool alloc failure aborts (`:219,281`); KU cache silently degrades (state -1, `:663-668`) | `:679` `<Krylov> GPU KU-cache: rank %d keeps %d of %d basis panels on the device (%.1f MiB)`; profile via `OPENMX_KRYLOV_GPU_PROFILE` (`:808`) | H+F |
| **Dense eigensolvers** (band/cluster/DC-LNO) | sec 1 | sec 1 | sec 1 / sec 4 | G+F |

---

## 3. Environment-knob reference table

### 3.1 The 61 knobs from knobs.txt (+ GEMMUL8_MAX_WORKSPACE_PERCENT)

"raise-only" = the env value can only increase the computed default, never lower it below the built-in floor.

| # | Knob | File:line (read site) | Default (code) | Effect |
|---|---|---|---|---|
| 1 | OPENMX_BAND_GEMMUL8_TURN_RELEASE | Band_DFT_Col.c:295; Band_DFT_NonCol.c:198; fallback for #16/#24 | 1 | Release GEMMul8 workspace when a rank finishes its GPU turns (band paths); 0 keeps it resident |
| 2 | OPENMX_BAND_GPU_DIAG | Band_DFT_Col.c:587; Band_DFT_NonCol.c:985 | unset = auto preflight | 1 forces band GPU dense diag, 0 forces ScaLAPACK/ELPA fallback (banner) |
| 3 | OPENMX_BAND_GPU_DM | Band_DFT_Col.c:1533 | on (only =0 disables) | GPU DM accumulation in the band-col ELPA-fallback DM loop (active when GPUSOLVER && n ≥ switch-num) |
| 4 | OPENMX_BAND_GPU_MAX_CONCURRENT_K | Band_DFT_Col.c:508; Band_DFT_NonCol.c:246 | INT_MAX (no cap) | Explicit cap on concurrent k-owner GPU turns per device (first alias) |
| 5 | OPENMX_BAND_GPU_MAX_RANKS_PER_DEVICE | Band_DFT_Col.c:515; Band_DFT_NonCol.c:253 | INT_MAX | Third alias of #4 |
| 6 | OPENMX_BAND_GPU_PERSISTENT | Band_DFT_Col.c:244 | 0 (off) | Keep band-col GPU solver context resident across SCF iterations |
| 7 | OPENMX_BAND_GPU_PERSISTENT_MIN_FREE_MB | Band_DFT_Col.c:265 | 2048 MiB | Free-memory threshold below which persistence (#6) stays off (stderr note :277-279) |
| 8 | OPENMX_BAND_GPU_RANK_OVERHEAD_MB | Band_DFT_Col.c:418; Band_DFT_NonCol.c:814 | 256 MiB | Per-owner-rank CUDA/OpenACC overhead charged in the band memory model |
| 9 | OPENMX_BAND_GPU_RESERVE_MB | Band_DFT_Col.c:483; Band_DFT_NonCol.c:930 | max(total/10, 1536 MiB); raise-only | Device headroom the band preflight/turn ladder keeps free. NOTE: fallback banners advise "lower OPENMX_BAND_GPU_RESERVE_MB" but the code cannot go below the floor (Band_DFT_Col.c:481-501) |
| 10 | OPENMX_BAND_GPU_SWITCH_NUM | Band_DFT_Col.c:561 | 800 (`BAND_GPU_CPU_SWITCH_NUM`, :554) | Band-col GPU/CPU crossover dimension (n) |
| 11 | OPENMX_BAND_GPU_TURN_GROUP | Band_DFT_Col.c:512; Band_DFT_NonCol.c:250 | INT_MAX | Second alias of #4 |
| 12 | OPENMX_BAND_NONCOL_GPU_DM | Band_DFT_NonCol.c:1929 | on (only =0 disables) | GPU DM in the band-NC ELPA-fallback loop (plus evec+256 MiB fit check) |
| 13 | OPENMX_BAND_NONCOL_GPU_SWITCH_NUM | Band_DFT_NonCol.c:469 | 800 vs n2 (`:462`) | Band-NC crossover (n2 = 2n) |
| 14 | OPENMX_BAND_PROFILE | Band_DFT_Col.c:64; Set_Hamiltonian.c:50; Set_OLP_Kin.c:184 | 0 | stderr phase-timing lines: `BANDPROF` (Band_DFT_Col.c:5388), `SETHPROF` (Set_Hamiltonian.c:1670), `SETHMEPROF` (:2704) |
| 15 | OPENMX_BAND_SYEVDX_NOVECTOR | Band_DFT_Col.c:309; Band_DFT_NonCol.c:212 | 1 | First multi-k pass runs cusolverDnXsyevdx with jobz=NOVECTOR (eigenvalues only); 0 always builds vectors |
| 16 | OPENMX_CLUSTER_GEMMUL8_TURN_RELEASE | Cluster_DFT_Col.c:557; Cluster_DFT_NonCol.c:824 | falls back to #1, then 1 | Release GEMMul8 workspace after the cluster diagonalization |
| 17 | OPENMX_CLUSTER_GPU_DIAG | Cluster_DFT_Col.c:1021; Cluster_DFT_NonCol.c:1754 | unset = auto reserve-probe | 1 forces cluster GPU dense diag (NC: aborts if reservation impossible, Cluster_DFT_NonCol.c:1772-1786), 0 forces ELPA/ScaLAPACK |
| 18 | OPENMX_CLUSTER_GPU_DIAG_RESERVE_MB | Cluster_DFT_Col.c:943; Cluster_DFT_NonCol.c:1635 | 1024 MiB (replaceable, can be lowered to 0) | Margin required on top of the owner's dense reservation |
| 19 | OPENMX_CLUSTER_GPU_DIAG_SERIAL | Cluster_DFT_Col.c:1022 | 1 | Serialized two-spin tier when both spin owners don't fit: 0=disable, 2=skip concurrent probe (testing) |
| 20 | OPENMX_CLUSTER_GPU_DM | Cluster_DFT_Col.c:605 | on (only =0 disables) | OpenACC cluster-col DM accumulation (requires UseGpuAccel gate) |
| 21 | OPENMX_CLUSTER_GPU_SWITCH_NUM | Cluster_DFT_Col.c:584 | 400 (`:577`) | Cluster-col crossover (n) |
| 22 | OPENMX_CLUSTER_NONCOL_GPU_SWITCH_NUM | Cluster_DFT_NonCol.c:1727 | 400 vs n2 (`:1720`) | Cluster-NC crossover |
| 23 | OPENMX_DCLNO_GPU_THRESHOLD | Divide_Conquer_LNO.c:42 | 800 (`GPU_CPU_SWITCH_NUM2`, :31) | DC-LNO col: per-atom GPU eigensolve when local NUM ≥ threshold |
| 24 | OPENMX_DCLNO_GPU_TURN_RELEASE | Divide_Conquer_LNO.c:962 | falls back to #1, then 1 | Release GEMMul8 workspace + solver buffers after the DC-LNO phase |
| 25 | OPENMX_DCLNO_NONCOL_GPU_THRESHOLD | Divide_Conquer_LNO.c:64 | 400 (`:56`) vs spinor dim | DC-LNO NC threshold |
| 26 | OPENMX_DC_GPU_THRESHOLD | Divide_Conquer.c:51 | 400 (`:42`) | DC col per-atom GPU threshold (NUM) |
| 27 | OPENMX_DC_NONCOL_GPU_THRESHOLD | Divide_Conquer.c:70 | 300 (`:43`) vs 2*NUM (:4069) | DC NC threshold |
| 28 | OPENMX_DENSITY_GRID_GPU_LOCAL | Set_Density_Grid_GPU.c:1195 | 1 (auto) | 0=off, 1=distributed per-rank quadrature only when Set_Hamiltonian tables are device-resident, 2=always (stage tables per call) |
| 29 | OPENMX_GEMMUL8_MAX_WORKSPACE_PERCENT | gemmul8_bridge.cu:186 (also NULL-check Band_DFT_NonCol.c:190) | 30 % (`kDefaultMaxWorkspacePercent`, :18), capped at 100 (:94-96). Band-NC dense init setenv's 50 when both spellings unset (Band_DFT_NonCol.c:188-193, called :3304) | GEMMul8 workspace may not exceed this % of device total; exceeding → native cuBLAS fallback for that GEMM |
| 30 | OPENMX_GPU | set_cuda_default_device_from_local_rank.c:129 | unset (GPU on) | =0 disables the GPU for the whole run (probe returns unusable; whole run demotes to ELPA2 via DFT.c:158-196) |
| 31 | OPENMX_GPUSOLVER_SERIAL_GPU_TURNS | Band_DFT_Col.c:229; Band_DFT_NonCol.c:228 | 1 | Serialize GPU turns of owner ranks within a concurrency group; 0 = concurrent submission |
| 32 | OPENMX_GPU_PROBE_RESERVE_MB | set_cuda_default_device_from_local_rank.c:103 | 256 MiB (`:99`) | Free memory a rank must still see after creating its context to claim the GPU |
| 33 | OPENMX_GPU_RANK_MAP | set_cuda_default_device_from_local_rank.c:77 | block map | value starting with `m`/`M` = modulo map (local_rank % device_count) |
| 34 | OPENMX_GS2_ELPA_SOLVER | elpa_cosma_bridge.c:335 | 2-stage | =1 selects ELPA 1-stage solver |
| 35 | OPENMX_KRYLOV_GPU_DGEMM_MIN_FLOPS | Krylov.c:175 | 5.0e7 (`:31`) | Minimum m·n·k product to offload a Krylov DGEMM |
| 36 | OPENMX_KRYLOV_GPU_EIGEN_MIN | Krylov.c:181 | 300 (`:30`) | Minimum dimension to offload a Krylov eigensolve |
| 37 | OPENMX_KRYLOV_GPU_FUSED | Krylov.c:187 | 1 | Fused on-device projected solve (only when every GEMM passes #35; Krylov.c:800-806) |
| 38 | OPENMX_KRYLOV_GPU_KU_CACHE | Krylov.c:513 | 1 | Keep repacked Krylov basis panels on the device |
| 39 | OPENMX_KRYLOV_GPU_KU_RESERVE_MB | Krylov.c:520 | 4096 MiB | Headroom kept free by the KU-cache admission |
| 40 | OPENMX_KRYLOV_GPU_PROFILE | Krylov.c:808 | 0 | Per-atom fused-solve timing output |
| 41 | OPENMX_MIXH_GPU | Mixing_H.c:2204 | unset = auto preflight | 1=force GPU RMM-DIISH history (abort if unallocatable, Mixing_H.c:2306-2313), 0=legacy CPU, 2=incremental CPU |
| 42 | OPENMX_MIXH_GPU_RESERVE_MB | Mixing_H.c:2275 | max(total/10, 1536 MiB) | Reserve for the Mixing_H GPU arena verdict |
| 43 | OPENMX_MIXH_GPU_VERBOSE | Mixing_H.c:2236,2342 | off | Prints the engage/forced-incremental banners |
| 44 | OPENMX_MIX_FAST | DIIS_Mixing_Rhok.c:73; DIIS_Mixing_DM.c:42; GR_Pulay_DM.c:31; Kerker_Mixing_Rhok.c:36; Mixing_V.c:48 | 1 under GPUSOLVER, forced 0 otherwise | Incremental cached-Gram CPU mixing math (identical results, fewer allreduces); 0 restores original loops |
| 45 | OPENMX_OLPKIN_BESSEL_CACHE_MAX | Set_OLP_Kin.c:84 | 128 entries (`:77`); 0 disables | Per-call spherical-Bessel table cache in Set_OLP_Kin (single-thread only) |
| 46 | OPENMX_OLPKIN_RADIAL_GPU | Set_OLP_Kin.c:56 | 0 (opt-in) | OpenACC radial integration in Set_OLP_Kin (measured slower than CPU on sidia333 → default off) |
| 47 | OPENMX_SETHAM_BASE_GPU | Set_Hamiltonian.c:711 | 0 (opt-in) | GPU packed base add H0+HNL(+HCH)→H; forced off for Zeeman/xmcd (`:710`) |
| 48 | OPENMX_SETHAM_CUDA_KERNEL | Set_Hamiltonian.c:264 | 1 | Use the CUDA matrix-elements kernel; 0 = OpenACC kernel |
| 49 | OPENMX_SETHAM_GPU_KOWNER_ONLY | DFT.c:319 | 0 (all ranks with local H) | 1 restores the legacy k-owner-only GPU-user policy for Set_Hamiltonian |
| 50 | OPENMX_SETHAM_GPU_MAX_RANKS_PER_DEVICE | Set_Hamiltonian.c:324 | INT_MAX | Explicit cap on Set_Hamiltonian first-wave GPU ranks per device |
| 51 | OPENMX_SETHAM_GPU_RANK_OVERHEAD_MB | Set_Hamiltonian.c:2031 | 512 MiB (`:1996`) | Per-active-rank runtime overhead charged by the matrix-elements memory model |
| 52 | OPENMX_SETHAM_GPU_RESERVE_MB | Set_Hamiltonian.c:305 | max(total/10, 1536 MiB); raise-only | Per-device reserve of the Set_Hamiltonian turn plan |
| 53 | OPENMX_SETHAM_GPU_RESIDENT | Set_Hamiltonian.c:355 | 1 | Keep SCF-invariant tables (orbitals/grid maps/pair metadata) resident across iterations |
| 54 | OPENMX_SETHAM_GPU_RESIDENT_FLOOR_MB | Set_Hamiltonian.c:369 | 4096 MiB (with registered phase need) / 16384 MiB (without) | Free memory that must remain after resident upload |
| 55 | OPENMX_SETHAM_GPU_SERIAL_WAVES | Set_Hamiltonian.c:339 | 0 (hybrid: first wave GPU, rest CPU) | 1 = push every later wave through the GPU serially |
| 56 | OPENMX_SETHAM_GPU_TEST_NEED_FROM_CALL | Set_Hamiltonian.c:389 | 1 | Testing hook: which matrix-elements call registers the fake need |
| 57 | OPENMX_SETHAM_GPU_TEST_NEED_MB | Set_Hamiltonian.c:388 | unset | Testing hook: pretend a GPU phase registered this need (MiB) |
| 58 | OPENMX_SETHAM_MATRIX_GPU | Set_Hamiltonian.c:255 | 1 | GPU path of the expensive matrix-elements quadrature; 0 = CPU comparison mode |
| 59 | OPENMX_SETNL_OPENACC | Set_Nonlocal.c:252 | 1 (on when GPUSOLVER + usable device) | Batched OpenACC radial Fourier integrals of DS_NL; 0 = OpenMP CPU path |
| 60 | OPENMX_SETPRO_GPU | Set_ProExpn_VNA.c:169,762,1480 (collective helper :140-153); DFT.c:66 (banner only) | 1 | The three Set_ProExpn_VNA device batches (HVNA / DS_VNA / VNA23); rank-0 value broadcast |
| 61 | OPENMX_TOTALENERGY_GPU | Total_Energy.c:76 | 1 | Batched OpenACC EH0/Exc0 total-energy terms (+ force #6/#7) |

### 3.2 Additional env knobs found in the tree but NOT in knobs.txt

These are read through wrapper helpers (`env_bool`/`DC_EnvU32`/`SDG_env_bool`/`Force_env_flag`/`gs2_env_long` etc.), which is why a plain `getenv("OPENMX_` grep missed them. A run-manifest implementation should record these too.

| Knob | File:line | Default | Effect |
|---|---|---|---|
| GEMMUL8_DISABLE / OPENMX_GEMMUL8_DISABLE / GEMMUL8_DISABLE_{D,Z} / OPENMX_GEMMUL8_DISABLE_{D,Z} | gemmul8_bridge.cu:109-117,308,367,380,420 | all off | Chained disable (generic → OPENMX-generic → typed → OPENMX-typed); disabled → native cuBLAS with one-time stderr "environment disable" note |
| GEMMUL8_FASTMODE_{D,Z} / OPENMX_GEMMUL8_FASTMODE_{D,Z} | gemmul8_bridge.cu:296,410 | false | GEMMul8 fastmode flag per type |
| GEMMUL8_NUM_MOD_{D,Z} / OPENMX_GEMMUL8_NUM_MOD_{D,Z} | gemmul8_bridge.cu:119-129,295,371,384,409 | 15 (clamped to [2,20], :15-16,124-126) | Number of moduli of the INT8 ozIMMU scheme |
| GEMMUL8_MIN_FREE_AFTER_MB / OPENMX_GEMMUL8_MIN_FREE_AFTER_MB | gemmul8_bridge.cu:185 (default :17); Divide_Conquer.c:381-382 | 1536 MiB | Free-memory reserve that must survive workspace allocation, else cuBLAS fallback |
| CUBLAS_MIN_FREE_AFTER_MB / OPENMX_CUBLAS_MIN_FREE_AFTER_MB | Divide_Conquer.c:391-392 | 3072 MiB | DC: free memory required before retrying a failed GEMM with native cuBLAS |
| GPUSOLVER_MIN_FREE_AFTER_MB / OPENMX_GPUSOLVER_MIN_FREE_AFTER_MB | Divide_Conquer.c:401-402 | 1536 MiB | DC: reserve for the per-atom eigensolver path |
| OPENMX_DC_GPU_S_CACHE | Divide_Conquer.c:687 | 1 | DC device cache of cluster overlap matrices (banner Divide_Conquer.c:842) |
| OPENMX_DC_GPU_S_CACHE_RESERVE_MB | Divide_Conquer.c:692 | 4096 MiB | Admission reserve of the DC S-cache |
| OPENMX_DENSITY_GRID_GPU (alias OPENMX_SETDENSITY_GPU) | Set_Density_Grid_GPU.c:851-852,1200 | 1 | Master on/off of both Set_Density_Grid GPU tiers |
| OPENMX_DENSITY_GRID_GPU_RESERVE_MB | Set_Density_Grid_GPU.c:873,1251 | 256 MiB | Owner/local reserve |
| OPENMX_DENSITY_GRID_GPU_PERSISTENT | Set_Density_Grid_GPU.c:958 | 0 | Keep the owner-service cache device-resident between iterations |
| OPENMX_FORCE_GPU | Force.c:2188 (collective) | 1 | Master gate of the Force OpenACC paths |
| OPENMX_FORCE3_GPU | Force.c:873,3562 | 1 | Force3 GPU trace (plus budget probe) |
| OPENMX_FORCE3_GPU_DORB | Force.c:915 | 1 | On-device orbital derivatives inside Force3 (L0≤3, Mul≤8, uncontracted only) |
| OPENMX_FORCE4B_GPU | Force.c:7250,3592 | 1 | Force4B separable-VNA GPU batch; NOTE independent hard switch `use_force4b_openacc = 0` at Force.c:9037 keeps the packed-array OpenACC contraction off |
| OPENMX_FORCE4B_FUSED | Force.c:9038-9039 | 1 | Fused CPU trace of Force4B |
| OPENMX_FORCEHNL_GPU | Force.c:8028 | 1 | HNL force GPU batch (scalar-relativistic real branches only) |
| OPENMX_FORCE_PROFILE | Force.c:2186,8957 | 0 | Per-part force timing report |
| OPENMX_GS2_ELPA_GPU | elpa_cosma_bridge.c:234 | unset (auto memory verdict) | 0=ELPA CPU kernels, 1=always GPU |
| OPENMX_GS2_GEMM_GPU | elpa_cosma_bridge.c:539 | unset (auto) | 0=ScaLAPACK, 1=always COSMA |
| OPENMX_GS2_GPU_FACTOR / _FIXED_MB / _RESERVE_MB / _TILE / _VERBOSE | elpa_cosma_bridge.c:90 / 91 / 157 / 468 / 298,554 | 3 / 128 MiB / 512 MiB / 1024 / 0 | gs2 memory model and COSMA tile caps (TILE exported to COSMA_GPU_MAX_TILE_{M,N,K}, :470-474) |
| OPENMX_ORBS_GRID_GPU | Set_Orbitals_Grid.c:63 | 1 | GPU orbital-grid evaluation (Cnt_kind==0, float Type_Orbs_Grid, L0 caps) |
| OPENMX_ORBS_GRID_GPU_VERBOSE | Set_Orbitals_Grid.c:703 | 0 | stderr `Set_Orbitals_Grid: GPU path active` (:704) |
| OPENMX_UCELL_OLG_GPU | truncation.c:4273 | 1 | GPU overlap-grid (UCell_Box) path |
| OPENMX_UCELL_OLG_GPU_VERBOSE | truncation.c:4460 | 0 | stderr `UCell_Box: GPU overlap-grid path active` (:4464) |

---

## 4. Banner-string inventory (run-manifest / log-scraper keys)

Format strings quoted verbatim (printf format, `\n` omitted). "once" = latched by a static flag. All stdout unless marked stderr.

### 4.1 Startup / routing (DFT.c, Input_std.c)

| # | file:line | Format string | Meaning |
|---|---|---|---|
| B01 | Input_std.c:790 | `Warning: scf.eigen.lib=cusolver is deprecated; use scf.eigen.lib=gpusolver instead.` | deprecated alias |
| B02 | Input_std.c:819 | `<Input_std> scf.gemmul8.enable=off: GPU dense GEMMs use plain cuBLAS instead of GEMMul8.` | GEMMul8 disabled by input |
| B03 | Input_std.c:1185-1186 | `scf.eigen.lib=gpusolver2 supports only scf.EigenvalueSolver=cluster` (+ second line) | gpusolver2 misuse, run exits |
| B04 | DFT.c:86 | `<DFT> GPUSOLVER requested; global matrix dimension %d is below %d, so global dense eigensolver paths use a CPU fallback while GPU kernels remain enabled.` | small-system dense fallback |
| B05 | DFT.c:121 | `<DFT> scf.Gpu.Num caps the GPUs used per node at %d of the %d detected.` | device cap |
| B06 | DFT.c:176 | `<DFT> GPUSOLVER requested, but GPU initialization failed on %d of %d MPI ranks; using ELPA2.` | whole-run demotion |
| B07 | DFT.c:180-195 | `<DFT>   cudaGetDeviceCount failed: %s (error %d).` / `<DFT>   nvidia-smi may still work...` / `<DFT>   cudaGetDeviceCount reported 0 devices; check CUDA_VISIBLE_DEVICES...` / `<DFT>   the OpenACC runtime reports no NVIDIA device (acc_get_num_devices=0)...` / `<DFT>   cudaSetDevice failed: %s (error %d).` / `<DFT>   some ranks could not initialize their GPU (device memory exhausted by the node's CUDA contexts, or OPENMX_GPU=0); reduce the MPI ranks per GPU or raise scf.Gpu.Num.` | demotion reason detail (cases 1-5) |
| B08 | DFT.c:536 / 544 / 553 / 560 | `<MD=%2d>  Calculation of the overlap matrix%s` / `...nonlocal matrix%s` / `...core hole matrix` / `...VNA projector matrix%s` — suffix `%s` = ` (GPU-accelerated)` or empty | phase labels + GPU suffix |
| B09 | DFT.c:939 | `<%s>  Solving the eigenvalue problem%s...` (`%s` #1 = solver name, #2 = ` (GPU-accelerated)`) | per-iteration solver evidence |
| B10 | DFT.c:946 | `<%s>  Note: the CPU eigensolver prints nothing until every k point of an SCF iteration is done (matrix dimension %d); minutes of silence here are normal.` (once) | CPU-diag notice |
| B11 | DFT.c:2327 / 2466 | `<MD=%2d>  Force calculation` / `<MD=%2d>  Total Energy` | phase labels |
| B12 | set_cuda_default_device_from_local_rank.c:164-166 (stderr) | `gpu_rank_device_usable: cudaSetDevice(%d) failed on this rank; using host paths.` | per-rank probe failure |
| B13 | set_cuda_default_device_from_local_rank.c:206-212 (stderr) | `gpu_rank_device_usable: GPU %d cannot be initialized on this rank (%.1f MiB free of %.1f MiB, %.1f MiB reserve); using host paths.` | per-rank probe failure |

### 4.2 Dense-solver dispatch / fallback

| # | file:line | Format string |
|---|---|---|
| B20 | Band_DFT_Col.c:598 | `<Band_DFT_Col> GPU dense diagonalization disabled by OPENMX_BAND_GPU_DIAG=0.` (once) |
| B21 | Band_DFT_Col.c:654-659 | `<Band_DFT_Col> A k-owner rank cannot fit even one k-point's dense GPU diagonalization (%.1f MiB needed per rank, %.1f MiB free); falling back to the ScaLAPACK/ELPA diagonalization. Force the GPU path with OPENMX_BAND_GPU_DIAG=1 or lower OPENMX_BAND_GPU_RESERVE_MB.` |
| B22 | Band_DFT_Col.c:851-854 | `<Band_DFT_Col> GPU device %d: %d k-owner rank(s), GPU concurrency=%d, turn group=%d, per-rank=%.3f GiB, peak=%.3f GiB, free=%.3f GiB, reserve=%.3f GiB` (re-printed on change) |
| B23 | Band_DFT_Col.c:868-870 (stderr) | `Band_DFT_Col: failed to query common GPU memory; using one GPU k-owner rank at a time.` |
| B24 | Band_DFT_Col.c:277-279 (stderr) | `Band_DFT_Col: persistent GPU buffers disabled (%.0f MiB free < %.0f MiB threshold).` |
| B25 | Band_DFT_NonCol.c:996 | `<Band_DFT_NonCol> GPU dense diagonalization disabled by OPENMX_BAND_GPU_DIAG=0.` (once) |
| B26 | Band_DFT_NonCol.c:1054-1059 | `<Band_DFT_NonCol> A k-owner rank cannot fit even one k-point's dense GPU diagonalization (%.1f MiB needed per rank, %.1f MiB free); falling back to the ScaLAPACK/ELPA diagonalization. Force the GPU path with OPENMX_BAND_GPU_DIAG=1 or lower OPENMX_BAND_GPU_RESERVE_MB.` |
| B27 | Band_DFT_NonCol.c:1218-1221 | `<Band_DFT_NonCol> GPU device %d: %d k-owner rank(s), GPU concurrency=%d (%s), turn group=%d, per-rank=%.3f GiB, peak=%.3f GiB, free=%.3f GiB, reserve=%.3f GiB` (`%s` = mode label ev/vec/dm) |
| B28 | Band_DFT_NonCol.c:1332-1334 | `<Band>  Serializing non-collinear dense GpuSolver k-worlds: GPU device %d is shared by %d owner rank(s), free %.3f MiB, need %.3f MiB plus %.3f MiB reserve.` |
| B29 | Cluster_DFT_Col.c:1006-1009 | `<Cluster_DFT_Col> The dense owner of spin world %d could not reserve its GPU diagonalization buffers%s (%.1f MiB of %.1f MiB free).` (`%s` = ` concurrently` / ` for the serialized mode` / empty) |
| B30 | Cluster_DFT_Col.c:1036 | `<Cluster_DFT_Col> GPU dense diagonalization disabled by OPENMX_CLUSTER_GPU_DIAG=0.` (once) |
| B31 | Cluster_DFT_Col.c:1085-1087 | `<Cluster_DFT_Col> The device cannot hold both spin owners' GPU buffers at once; spin world 0's owner solves both spins serially on the GPU (disable with OPENMX_CLUSTER_GPU_DIAG_SERIAL=0).` |
| B32 | Cluster_DFT_Col.c:1104-1106 | `<Cluster_DFT_Col> Falling back to the ELPA/ScaLAPACK diagonalization. Force the GPU path with OPENMX_CLUSTER_GPU_DIAG=1 or lower OPENMX_CLUSTER_GPU_DIAG_RESERVE_MB.` |
| B33 | Cluster_DFT_NonCol.c:1766 | `<Cluster_DFT_NonCol> GPU dense diagonalization disabled by OPENMX_CLUSTER_GPU_DIAG=0.` (once) |
| B34 | Cluster_DFT_NonCol.c:1843-1848 | `<Cluster_DFT_NonCol> The dense owner could not reserve its GPU diagonalization buffers (%.1f MiB of %.1f MiB free); falling back to the ELPA/ScaLAPACK diagonalization. Force the GPU path with OPENMX_CLUSTER_GPU_DIAG=1 or lower OPENMX_CLUSTER_GPU_DIAG_RESERVE_MB.` |
| B35 | Divide_Conquer_LNO.c:1363-1364 | `<DC-LNO> GPUSOLVER direct per-rank dispatch is enabled for local matrices with dimension >= %d on %d CUDA device(s).` (first call) |
| B36 | Divide_Conquer_LNO.c:2765-2766 | `<DC-LNO> GPUSOLVER direct per-rank dispatch (noncollinear) is enabled for local matrices with dimension >= %d on %d CUDA device(s).` |
| B37 | Divide_Conquer_LNO.c:1067-1073 | `<DC-LNO> GPU device %d: %d rank(s) would need %.3f GiB (free %.3f GiB, reserve %.3f GiB); using the CPU eigensolver instead.` (once) |
| B38 | Divide_Conquer.c:842-843 | `<DC> GPU S-cache: rank %d keeps %d of %d cluster overlaps on the device (%.1f MiB)` |
| B39 | Divide_Conquer.c:665-668 (stderr) | `<DC> rank %d: GEMMul8 failed in %s for GEMM(m=%d,n=%d,k=%d): %s (%d). Retrying this DC Hamiltonian solve with native cuBLAS.` |
| B40 | Cluster_DFT_Col.c:1798 / 2047; Cluster_DFT_NonCol.c:2593 / 2829 | `Cluster_DFT_Col: the gpusolver2 overlap|Hamiltonian eigensolver failed (info=%d)` / `Cluster_DFT_NonCol: ...` (followed by MPI_Abort) |
| B41 | elpa_cosma_bridge.c:190 / 194-197 | `<DFT> gpusolver2: ELPA n=%d nev=%d %s %s kernels (OPENMX_GS2_ELPA_GPU)` / `<DFT> gpusolver2: ELPA n=%d nev=%d %s %s kernels (GPU memory: need ~%llu MB/rank x %d ranks/GPU + reserve %llu MB %s free %llu MB)` |
| B42 | elpa_cosma_bridge.c:218 | `<DFT> gpusolver2: ELPA (API %d) with NVIDIA GPU kernels + COSMA initialized` |
| B43 | elpa_cosma_bridge.c:515 / 519-522 | `<DFT> gpusolver2: %s m=%d n=%d k=%d using %s (OPENMX_GS2_GEMM_GPU)` / `<DFT> gpusolver2: %s m=%d n=%d k=%d using %s (GPU memory: need ~%llu MB/rank x %d ranks/GPU + reserve %llu MB %s free %llu MB)` |
| B44 | Krylov.c:470-475 (stderr) | `Krylov: %s failed, info=%d; falling back to LAPACK.` and `Krylov: cusolverDnXsyevdx returned %lld eigenpairs, expected %d; falling back to LAPACK.` (also Krylov.c:771) |

### 4.3 GPU phases

| # | file:line | Format string |
|---|---|---|
| B50 | Set_Hamiltonian.c:683-686 | `<Set_Hamiltonian> GPU device %d: %d Hamiltonian rank(s), GPU concurrency=%d, %s=%d, peak=%.3f GiB, free=%.3f GiB, reserve=%.3f GiB` (`%s` = `serialized waves` or `CPU fallback ranks`; re-printed on change) |
| B51 | Set_Hamiltonian.c:920-923 | `<Set_Hamiltonian> GPU resident cache: %d rank(s) hold %.3f GiB (%s) on device %d across SCF iterations` |
| B52 | Set_Hamiltonian.c:1786-1788 | `<Set_Hamiltonian> GPU resident cache released: a GPU phase needs %.3f GiB on device %d; staying transient for this MD step` |
| B53 | Set_Hamiltonian.c:1483 | `<Set_Hamiltonian>  Hamiltonian matrix for VNA+dVH+Vxc%s...` (suffix ` (GPU-accelerated)`) |
| B54 | Set_Hamiltonian.c:484 (stderr) | `Set_Hamiltonian: rank %d %s cannot initialize CUDA/OpenACC; using CPU.` |
| B55 | Set_Hamiltonian.c:515 (stderr) | `Set_Hamiltonian: %s failed to query common GPU memory; using CPU.` |
| B56 | set_hamiltonian_gpu.cu:145-147 (stderr, ≤3×) | `Set_Hamiltonian_Cuda_MatrixElements: cleared stale CUDA error "%s" left by an earlier recovered failure; continuing.` |
| B57 | set_hamiltonian_gpu.cu:185-187 (stderr, ≤3×) | `Set_Hamiltonian_Cuda_MatrixElements: CUDA kernel failed with "%s" but the context is healthy; deferring to the OpenACC kernel for this batch.` |
| B58 | Set_ProExpn_VNA.c:210-211 (stderr) | `Set_ProExpn_VNA GPU: not enough free device memory; CPU fallback.` |
| B59 | Set_ProExpn_VNA.c:943-945 (stderr) | `Set_ProExpn_VNA DS_VNA GPU: needs %.3f GiB per rank, %d rank(s) share this device and only %.3f GiB is free; CPU fallback.` |
| B60 | Set_Density_Grid.c:91 | `<Set_Density_Grid> distributed GPU quadrature: every rank integrates its own atoms` (once) |
| B61 | Set_Density_Grid_GPU.c:916-919 (stderr) | `Set_Density_Grid GPU owner needs %.3f GiB plus %.3f GiB reserve, but only %.3f GiB is free; CPU fallback.` |
| B62 | Mixing_H.c:2326 | `<Mixing_H> The RMM-DIISH residual history (%.1f MiB per device) does not fit on the GPU (%.1f MiB free); using the incremental CPU Pulay mixing instead. OPENMX_MIXH_GPU=0 restores the legacy CPU mixing, OPENMX_MIXH_GPU=1 forces the GPU path.` (once) |
| B63 | Mixing_H.c:2237 (verbose) | `<Mixing_H> incremental CPU RMM-DIISH mixing forced by OPENMX_MIXH_GPU=2.` |
| B64 | Mixing_H.c:2343 (verbose) | `<Mixing_H> GPU RMM-DIISH mixing engaged (%d slots x %.1f MiB per rank).` |
| B65 | Krylov.c:679-681 | `<Krylov> GPU KU-cache: rank %d keeps %d of %d basis panels on the device (%.1f MiB)` |
| B66 | Force.c:3561-3564 / 3588-3593 | `  Force calculation #3%s` / `  Force calculation #4%s` (suffixes ` (GPU-accelerated)` / ` (GPU reduction)`) |
| B67 | Total_Energy.c:1028-1029 / 1607-1608 | `  Force calculation #6%s` / `  Force calculation #7%s` (suffix ` (GPU-accelerated)`) |
| B68 | Set_Orbitals_Grid.c:704 (stderr, verbose) | `Set_Orbitals_Grid: GPU path active` |
| B69 | truncation.c:4464 (stderr, verbose) | `UCell_Box: GPU overlap-grid path active` |
| B70 | gemmul8_bridge.cu:263-269 (stderr, once per type) | `openmx_gemmul8%sgemm: GEMMul8 workspace fallback by %s; need %.3f MiB, CUDA free %.3f MiB / total %.3f MiB, reserve %.3f MiB, max-workspace %u%%. Falling back to native cuBLAS.` (`%s` #1 = `D`/`Z`; #2 = reason: `workspace fraction policy` / `free memory reserve policy` / `cudaMalloc failure` / `environment disable` / `allocation failure`) |

### 4.4 Profiling lines (opt-in evidence)

| file:line | trigger | line prefix |
|---|---|---|
| Band_DFT_Col.c:5388 (stderr) | OPENMX_BAND_PROFILE=1 | `BANDPROF id=%d it=%d ...` |
| Set_Hamiltonian.c:1670 (stderr) | OPENMX_BAND_PROFILE=1 | `SETHPROF id=%d it=%d ...` |
| Set_Hamiltonian.c:2704 (stderr) | OPENMX_BAND_PROFILE=1 | `SETHMEPROF id=%d ...` |

Timing-only stdout lines that are NOT path evidence (do not key manifests off them): `<Band_DFT>  Eigen, time=...` (Band_DFT_Col.c:4478, Band_DFT_NonCol.c:4531), `<Band_DFT>  DM, time=...` (Band_DFT_Col.c:5109, Band_DFT_NonCol.c:5117).

---

## 5. GEMMul8 integration points

### 5.1 Bridge and interception

- Bridge: `gemmul8_bridge.cu` exports `openmx_gemmul8Dgemm` (`:276`) / `openmx_gemmul8Zgemm` (`:390`) / `openmx_gemmul8{D,Z}WorkspaceSize` (`:377,364`) / `openmx_gemmul8ReleaseWorkspaces` (`:333`) / `openmx_gemmul8SetEnabled` (`:359`). Template instantiation TU: `gemmul8_openmx.cu` (INT8-backend gemm for double and cuDoubleComplex + workSize only, `:31-65`); upstream sources under `third_party/GEMMul8/`.
- CUBLAS_OP_C is mapped to CUBLAS_OP_T for the real case (`gemmul8_bridge.cu:297-298`).
- Call sites (every GPU dense GEMM in the tree goes through the bridge; no direct `cublasDgemm/Zgemm` in solver code outside it):
  - Dgemm: `Cluster_DFT_Col.c:440`; `Cluster_DFT_Col_CWF.c:72`; `Cluster_DFT_Col_DMmu.c:70`; `Cluster_DFT_Col_LNO.c:69`; `Cluster_DFT_NonCol.c:1006`; `Divide_Conquer.c:656`; `Divide_Conquer_LNO.c:565,568,573`; `Krylov.c:379,841,847,885`.
  - Zgemm: `Band_DFT_Col.c:1992,2035,2039,2056`; `Band_DFT_Col_CWF.c:76`; `Band_DFT_Col_DMmu.c:71`; `Band_DFT_NonCol.c:440`; `Band_DFT_NonCol_CWF.c:73`; `Band_DFT_NonCol_DMmu.c:71`; `Cluster_DFT_NonCol.c:1021,1041`; `Cluster_DFT_NonCol_CWF.c:73`; `Cluster_DFT_NonCol_DMmu.c:70`; `Divide_Conquer_LNO.c:924,929,943`.

### 5.2 scf.gemmul8.enable routing

- `Input_std.c:812-822`: `input_logical("scf.gemmul8.enable", &gemmul8_enable, 1)` (default on) → `openmx_gemmul8SetEnabled()`; off prints B02 once. In the bridge, `g_input_enabled==0` short-circuits to `cublasDgemm`/`cublasZgemm` with no warning (`gemmul8_bridge.cu:302-306,414-418`), so A/B runs differ only in the GEMM backend.

### 5.3 Workspace policy

- One cached workspace per (device, stream) in a mutex-guarded map (`gemmul8_bridge.cu:21-44,191-192`).
- Required size = `gemmul8::workSize<is_complex, INT8>(m,n,k,num_moduli)` (`:179`); moduli default 15, clamp [2,20] (`:15-16,119-129`).
- Admission checks, in order (`ensure_workspace`, `:162-251`):
  1. Fraction policy: required > total × `MAX_WORKSPACE_PERCENT`/100 → release + `CUBLAS_STATUS_ALLOC_FAILED` (reason `workspace fraction policy`, `:202-211`). Percent default **30** (`:18`), env `OPENMX_GEMMUL8_MAX_WORKSPACE_PERCENT` overriding `GEMMUL8_MAX_WORKSPACE_PERCENT`, capped at 100 (`:89-99,186-188`). The band-NC dense path setenv's the OPENMX spelling to **50** when neither spelling is set (`Band_DFT_NonCol.c:188-193`, invoked at `:3304`) — process-wide from that point.
  2. Reserve policy: free after (re)allocation must keep ≥ `MIN_FREE_AFTER_MB` (default **1536 MiB**, `:17,185`; env pair `OPENMX_GEMMUL8_MIN_FREE_AFTER_MB`/`GEMMUL8_MIN_FREE_AFTER_MB`) → else ALLOC_FAILED (reason `free memory reserve policy`, `:226-236`).
  3. `cudaMalloc` failure (reason `cudaMalloc failure`, `:238-245`).
- Workspace lifetime: released explicitly by `openmx_gemmul8ReleaseWorkspaces()` at turn/phase boundaries, controlled by `OPENMX_BAND_GEMMUL8_TURN_RELEASE` / `OPENMX_CLUSTER_GEMMUL8_TURN_RELEASE` / `OPENMX_DCLNO_GPU_TURN_RELEASE` (all default release=1; sec 3.1 #1/#16/#24); DC-LNO also releases at phase teardown (`Divide_Conquer_LNO.c:1341`).

### 5.4 Fallback-to-cuBLAS conditions and evidence

| Condition | Behaviour | Evidence string |
|---|---|---|
| `scf.gemmul8.enable=off` | every call → native cuBLAS, silent | B02 at parse time only (`Input_std.c:819`) |
| env disable chain (`GEMMUL8_DISABLE` → `OPENMX_GEMMUL8_DISABLE` → typed → OPENMX-typed; `gemmul8_bridge.cu:109-117`) | native cuBLAS | B70 once, reason `environment disable` (`:308-312,420-424`) |
| workspace fraction policy exceeded | native cuBLAS for that call (workspace freed) | B70 once, reason `workspace fraction policy` |
| free-memory reserve violated | native cuBLAS | B70 once, reason `free memory reserve policy` |
| cudaMalloc failed | native cuBLAS | B70 once, reason `cudaMalloc failure` |
| DC solver: GEMMul8 returned non-SUCCESS | rank-local: disable GEMMul8 for the rest of the run, release workspaces, retry with native cuBLAS if `OPENMX_CUBLAS_MIN_FREE_AFTER_MB` (3072 MiB) headroom exists, else CPU path | B39 (`Divide_Conquer.c:645-683`) |
| Latched CUDA error left by a recovered GEMMul8 fallback | popped defensively before the Set_Hamiltonian CUDA kernel | B56 (`set_hamiltonian_gpu.cu:139-149`) |

### 5.5 Interaction with the memory planners

Every band/cluster/DC-LNO preflight charges `openmx_gemmul8{D,Z}WorkspaceSize(n,n,n)` explicitly (`Band_DFT_Col.c:454`, `Divide_Conquer_LNO.c:1055-1056`; Band NC comment `Band_DFT_NonCol.c:925-927`), and the workspace-size query returns 0 when GEMMul8 is input- or env-disabled (`gemmul8_bridge.cu:364-388`), so disabling GEMMul8 also shrinks the planners' reservations. The 1536 MiB floor of the band/Set_Hamiltonian reserves is documented in-code as GEMMul8 headroom (`Set_Hamiltonian.c:304-306`).

---

## Appendix: verified quirks worth a footnote in the paper

1. Reserve knobs are asymmetric: `OPENMX_BAND_GPU_RESERVE_MB` and `OPENMX_SETHAM_GPU_RESERVE_MB` are raise-only above max(total/10, 1536 MiB) (`Band_DFT_Col.c:481-501`, `Set_Hamiltonian.c:302-318`) although fallback banners B21/B26 suggest lowering them; `OPENMX_CLUSTER_GPU_DIAG_RESERVE_MB` genuinely replaces its default (`Cluster_DFT_Col.c:943-951`).
2. `<MD=..> overlap matrix (GPU-accelerated)` suffix (B08) does not reflect the opt-in `OPENMX_OLPKIN_RADIAL_GPU` gate (`DFT.c:54-57` vs `Set_OLP_Kin.c:399`).
3. Force4B OpenACC contraction is hard-disabled (`Force.c:9030-9037`) regardless of `OPENMX_FORCE4B_GPU`; the knob only gates the separable-VNA batch entered at `Force.c:7250`.
4. DC-LNO's shared-GPU proxy service is compiled out (`DCLNO_ENABLE_SHARED_GPU_PROXY 0`, `Divide_Conquer_LNO.c:121`); the live path is direct per-rank dispatch with the group memory check B37.
5. `openmx_common.h:2814-2816` describes gpusolver2 as "DLA-Future + COSMA"; the implementation is ELPA (NVIDIA GPU kernels) + COSMA (`elpa_cosma_bridge.c:190-236`). UNVERIFIED: whether any DLA-Future code path exists elsewhere — none was found in `source/`.
6. knobs.txt (61 names) misses 43 additional env names read through wrapper helpers (sec 3.2); log scrapers and run_manifest.json should use the union.

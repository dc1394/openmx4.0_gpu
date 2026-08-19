# run_manifest.json — design note (plan v2.6 sec. 10 / WBS item 3, P0)

Grounded in `gpu_phase_audit.md` (commit cd5f0d5). Goal: every run emits one
machine-readable JSON proving which fast paths actually executed, so no
"GPU版" label ever has to be taken on faith. This note fixes the schema and
the emission mechanism BEFORE any code is written, so the implementation is
mechanical and reviewable.

## 1. Emission mechanism

- One file per run: `<System.Name>.manifest.json`, written by Host_ID at the
  end of `openmx.c` main (after `Output_CompTime`), assembled via
  `MPI_Reduce`/`MPI_Gather` of per-rank counter structs. No I/O from other
  ranks.
- A tiny fixed-size counter registry in `openmx_common.c`:
  `OpenMX_Manifest_Count(key)` / `OpenMX_Manifest_Set(key, value)` /
  `OpenMX_Manifest_Note(key, string)`; keys are compile-time enum + string
  table so the reduce is a flat `long long` array (no dynamic allocation in
  hot paths; single atomic increment per event).
- Counters are incremented at EXACTLY the sites that today print the
  banners inventoried in audit sec. 4 (B01-B70). Rule: **one banner = one
  counter**; the printf stays, the counter is added beside it. That keeps
  log-scraping (current validators) and the manifest in provable agreement.
- Overhead budget: increments only at per-SCF or per-phase granularity
  (never per matrix element); target zero measurable wall-time change
  (P2's production/profiling split stays separate).

## 2. Schema (v1)

```json
{
  "manifest_version": 1,
  "source_commit": "<git describe --always --dirty>",
  "release_tag": "<v2.0_thesis or empty>",
  "gemmul8_commit": "<from build stamp>",
  "build": {"compiler": "...", "cuda": "...", "mpi": "...", "md5": "<argv0 md5>"},
  "input": {"file": "...", "sha256": "...", "system_name": "...",
             "solver": "band|cluster|dc|dclno|...", "spin": "off|on|nc",
             "kgrid": [2,2,2], "atoms": 216, "scf_maxiter": 25,
             "criterion": 1e-15, "eigen_lib": "gpusolver",
             "gemmul8_enable": true},
  "layout": {"ranks": 48, "threads": 1, "nodes": 1,
              "rank_to_gpu": "block|mod", "gpus_per_node": 1,
              "mps": {"detected": true, "pipe_dir": "..."}},
  "kpoints": {"computed": 8, "T_knum": ..., "numprocs1": ...,
               "all_knum": 1, "owners": [[0,"GPU-UUID"], ...],
               "k_per_gpu": 8},
  "dense_solver": {
    "path": "gpusolver|gpusolver2|elpa2-fallback|elpa2|elpa1|lapack",
    "switch_num": 800, "n": 5616, "jobz": "V", "novector_pass": 0,
    "maxn": ..., "reserve_mb": ..., "concurrency_plan": 8,
    "concurrency_actual_min": 4, "fallback_count": 0,
    "fallback_reasons": []},
  "phases": {
    "set_hamiltonian": {"gpu_ranks": 48, "cpu_ranks": 0, "resident": true,
                          "resident_bytes": ..., "waves": ..., "fallbacks": 0},
    "set_nonlocal":    {"path": "gpu|acc|cpu", "fallbacks": 0},
    "set_proexpn_vna": {"batches_gpu": 3, "batches_cpu": 0,
                          "vna23_silent_cpu": false},
    "set_density_grid":{"mode": "distributed-gpu|owner|cpu", "gpu_ranks": ...},
    "dm_edm":          {"path": "gpu-accumulate|cpu", "fallbacks": 0},
    "mixing":          {"tier": "gpu|incremental|legacy", "history": 7},
    "total_energy":    {"path": "gpu|cpu"},
    "force":           {"batches_gpu": ..., "batches_cpu": ..., "fallbacks": 0}
  },
  "gemmul8": {"enabled": true, "backend": "int8", "num_moduli": 15,
               "fastmode": false,
               "d_calls": ..., "z_calls": ...,
               "d_fallback": 0, "z_fallback": 0,
               "fallback_reasons": [],
               "workspace_percent": 30, "workspace_peak_mb": ...},
  "memory": {"peak_vram_mb_per_rank_max": ..., "peak_host_rss_mb_node0": ...},
  "wall": {"total_s": ..., "dft_s": ..., "diag_s": ..., "scf_iters": 25},
  "exit_status": 0
}
```

## 3. What already exists vs what must be added

| Field group | Status (audit evidence) |
|---|---|
| dense_solver path/thresholds | conditions all localized (audit sec 1); need counters at the 4 solver dispatch sites + the DFT.c:158 whole-run demotion |
| concurrency plan/actual | already printed in banners B-series (`GPU concurrency=`); counter = min over SCF iters |
| set_hamiltonian ranks/resident | already printed (`<Set_Hamiltonian> GPU device %d: %d Hamiltonian rank(s)...`, resident-cache line); counters at same sites |
| gemmul8 D/Z call & fallback counts | PARTIAL: bridge has 5 fallback conditions each with one-shot stderr strings (audit sec 5); call counters must be added in `gemmul8_bridge.cu` wrappers (P1) |
| vna23 silent CPU fallback | audit flags it as SILENT — needs BOTH a counter and a new one-shot banner (quirk fix) |
| kpoints/all_knum/jobz | values live in solver locals; export via Manifest_Set at first-iteration setup |
| peak VRAM | per-rank `cudaMemGetInfo` min-free tracking at phase boundaries (cheap); host RSS from `/proc/self/status` VmHWM at exit |
| input sha256 | small sha256 impl or call out to system openssl at startup on Host_ID |

## 4. Validation plan

1. Unit: run the four sip_ combos 3-SCF; assert manifest fields equal the
   log-scraped values (a `validate_manifest.sh` that greps the joblog the
   way today's summarizers do and diffs against the JSON).
2. `-runtest` 14/14 unchanged; production wall time within noise of the
   pre-manifest binary on one representative point (bnc216 o, 3 reps).
3. The three deliberate-fallback scenarios (OPENMX_BAND_GPU_DIAG=0;
   OPENMX_CLUSTER_GPU_DIAG_RESERVE_MB huge; tiny system below switch-num)
   must show nonzero fallback counters with matching reasons.

## 5. Out of scope for P0

Nsight union scripting (P5), timer split (P2), accuracy dumps (P6) — separate
WBS items. FP8/hook-mode fields — excluded from the first paper entirely.

## 6. Implementation notes (landed 2026-08-19, uncommitted tree on cd5f0d5)

What was built, and where it deviates from sections 1-3:

- Registry: `openmx_common.c` end ("Run manifest" block), enum + prototypes
  in `openmx_common.h` (`MANI_*`, `OpenMX_Manifest_{Count,Add,SetMax,SetMin,
  RankFlag,RankValue,Write}`). Reduce classes per key: SUM / MAX / MIN /
  FSUM (= local max, reduced SUM: "how many ranks", used for rank flags and
  per-device-leader values). Emission from `openmx.c` right after
  `Output_CompTime()`; wall fields come from CompTime slots [0]/[3]/[9]/[7].
- Instrumented files (counters beside the audited banners): DFT.c (B04-B07,
  B09, SCF-loop exit), set_cuda_default_device_from_local_rank.c (B12/B13),
  Band_DFT_Col.c (B20-B23 + Xsyevdx + 4 ELPA regions + all_knum/T_knum +
  k-owner flags + fallback-DM), Band_DFT_NonCol.c (B25-B28 + 2 Xsyevdx +
  4 ELPA regions x 2 branches + DM), Cluster_DFT_Col.c (B30-B32 + Xsyevdx +
  gs2 x2 + ELPA x4 + DM), Cluster_DFT_NonCol.c (B33/B34 + Xsyevdx + gs2 x2 +
  ELPA x4), Set_Hamiltonian.c (B50-B55 + per-rank GPU/CPU dispatch),
  Set_Nonlocal.c, Set_ProExpn_VNA.c, Set_Density_Grid.c/_GPU.c (B60/B61),
  Mixing_H.c (tier), Total_Energy.c (5 gate sites), Force.c (F3/F4/F4B/HNL),
  gemmul8_bridge.cu.
- Build stamp: `manifest_buildinfo.h` generated by both Makefiles
  (content-compare, no spurious recompiles), consumed via `__has_include`
  ("unknown" fallback), git-ignored. Records git describe --dirty, exact
  tag, GEMMUL8_COMMIT.
- P1 pulled forward: the GEMMul8 D/Z **call counters** and per-reason
  fallback counters landed inside `gemmul8_bridge.cu` together with P0
  (one atomic add per dense GEMM, negligible vs the GEMM itself), exposed
  through `openmx_gemmul8GetStats/GetConfig`; the manifest merges them at
  write time. B70's "allocation failure" and "cudaMalloc failure" reasons
  share one counter slot.
- Quirk fix from sec.3: the VNA23 silent CPU fallback now prints a one-shot
  banner (matching the B58/B59 style) and counts `vna_vna23_fallbacks`.
- Simplifications vs the sec.2 sketch: `kpoints.owners[]` (per-owner GPU
  UUID list) replaced by `kowner_ranks`; `memory.peak_vram_*` deferred to
  WBS item 5 (host VmHWM per rank + node0 sum are in); `dense_solver.n`,
  `jobz` folded into solve/novector counters; DC / DC-LNO / Krylov phase
  counters deferred (production matrix uses band/cluster only).
- Known blind spots (documented, acceptable for P0): the three silent
  `cudaMemGetInfo`-failure `return 0`s in Set_ProExpn_VNA are not counted;
  tiny-system LAPACK (`parallel_mode==0`) dense solves are not counted;
  `-runtest` runs several inputs in one process, so its manifest holds the
  accumulated counters of the whole suite under the last System.Name.
- Semantics notes from validation: `dense_cpu_solves` counts per-rank
  PARTICIPATIONS in distributed ELPA/ScaLAPACK eigensolves (48 ranks x
  4 solves = 192 for a 3-SCF band run), not global solve events;
  `memory.peak_node0_rss_sum_kb` is the SUM of per-rank VmHWM peaks — an
  upper bound on the node's simultaneous RSS (peaks at different times and
  shared pages double-count; the .smi sampler stays the ground truth until
  WBS item 5). First-build bug fixed same day: `build.md5` hashed
  /proc/self/exe inside the md5sum child (= md5sum itself); now resolved
  with readlink in the parent before hashing.
- Validation: `work/gen_simani.sh` + `work/validate_manifest.sh`;
  campaign simani_{bg,bo,nc,cc,cn,cpu,fb1,fb2,fb3,noise1-3} + runtest
  (job IDs in work/siband_jobids.txt).

## 7. P2 / P4 / P6 / P7 additions (landed 2026-08-19, same day)

- **P2 lightweight phase timing**: the manifest's `wall` section now
  carries all remaining CompTime slots (`phases_s`: Set_OLP_Kin,
  Set_Nonlocal, Set_ProExpn_VNA, Poisson, Mixing_DM, Force, Total_Energy,
  Set_Aden_Grid, Set_Orbitals_Grid, Set_Density_Grid) — zero new timers,
  MAX over ranks like the Max_Time column. A `profiling` section records
  the seven detailed-profiling knobs (OPENMX_BAND_PROFILE etc.), derives
  `mode: production|profiling`, and flags external injection
  (CUDA_INJECTION64_PATH / NSYS_PROFILING_SESSION_ID / NVTX_...), so the
  never-pool-profiled-runs rule is machine-checkable.
- **P4 VRAM peak**: `OpenMX_Manifest_VramSample()` (openmx_common.c) takes
  one cudaMemGetInfo sample; called only from GPU-path sites with an
  established context (the five Xsyevdx sites, the Set_Hamiltonian GPU
  branches) and additionally inside the GEMMul8 bridge's ensure_workspace
  (free/total already queried there; stats slots 11/12). memory section
  emits vram_total_mb / vram_min_free_mb / peak_device_vram_used_mb —
  device-global under MPS, which is the meaningful number when 48 ranks
  share one H100.
- **P7 environment snapshot**: the `environment` section dumps every set
  OPENMX_/GEMMUL8_/CUDA_/OMP_ variable as Host_ID sees it (mpirun -x
  forwards identically on this site).
- **P6 small accuracy dump** (Cluster_DFT_Col.c): OPENMX_ACC_DUMP=1 +
  OPENMX_ACC_DUMP_SCF=N make the dense owner write
  `<name>.accdump.scfN.spinS.{Htilde,Ytilde,ko}.bin` (+meta.txt) —
  the orthonormal-basis H~ right after the S~^T H S~ transform and the
  eigenpairs right after cusolver, before the back-transform.  Offline:
  `work/accdump_analysis.py` builds K~ = Y~ f Y~^T and reports eps_K,
  max|dK|, max|dH~|, max|d eps_i| and r_HK per plan sec. 13.2-C.
  Scope: cluster collinear GPU dense path only (the plan's Gamma-cluster
  representative); generator `work/gen_siacc64.sh` (Si 64-atom cluster,
  n=832, nocc=128).

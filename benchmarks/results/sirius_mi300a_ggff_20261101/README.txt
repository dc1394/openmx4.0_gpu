Sirius MI300A noncollinear band optimization, 2026-11-01
=====================================================

Scope and architecture policy
-----------------------------
GGFF is GaFeO3 with 40 atoms, SOC and Hubbard U, 64 k points (4x4x4),
456 basis functions, a 912-by-912 complex Hamiltonian and 432 requested
states. Its original input SHA-256 is
6a80fcc4f96e97944cc7b6affd20df04f24764cfb4852703f01e105ae3dc701f.

All measurements use one physical Sirius node with four MI300A APUs,
32 MPI ranks x 3 OpenMP threads, three disjoint CPU cores per rank,
ROCm 7.2.0 clang/flang, Open MPI 5.0.10 and AOCL 5.1.0 threaded BLIS.
Rank/GPU mapping remains block (eight ranks per GPU). CPU/NUMA placement
and MPI ownership of each k point are preserved. Only ordinary gpusolver
is timed; no gpusolver2/gpusolver3 or L2/L3 benchmark is added.

Radeon RX 9060 XT keeps the legacy noncollinear k-turn policy, streaming
constructor, blocking solver completion and existing GEMMul8/MAGMA policy.
The new owner-wave/constructor/density controls require MI300A and the
ordinary noncollinear SCF dense-k path. Batched result completion requires
MI300A rocSOLVER. gfx1200 compilation and policy models are validated on
Sirius; no physical Radeon GPU is available for a Radeon timing claim.

Changes
-------
* Schedule distinct k-point owners per GPU, bounded by that GPU's free
  memory and explicit owner cap. Retire an owner's whole ascending k queue
  together. The fixed global four-turn ceiling no longer limits MI300A.
  Partial device occupancy, unequal owner queues and multi-node device
  groups are covered by the host planner models; hardware timings here
  use four GPUs on a single physical node.
* Retain dense-construction indices across SCF calls, values within an
  SCF call (separate slots for overlap and seven Hamiltonian components),
  and phases within one k point. Upload asynchronously on the default
  stream, retaining the old scatter arithmetic and chunk boundaries.
  Default budget is 64 MiB per owner, included in the concurrency guard;
  insufficient budget/allocation falls back to the streaming constructor.
* Collect solver info, vector/value finite checks and requested eigenvalues
  into pinned host storage with async copies and one final stream wait.
  The caller still receives a completed solve. Retry/input backup and
  non-finite rejection remain intact. The old tiny info-copy wait represented
  outstanding GPU eigensolver work, not poor transfer bandwidth.
* When every owned eigenvector panel is cached, build density matrices on
  the admitted owners without per-k world collectives. Stage compact host
  panels at stride MaxN, rather than expanding to n2-by-n2. Check success
  at wave boundaries; preserve panels until global success. A local/remote
  failure discards all partial sums before the original collective recovery
  path replays. The final density reductions and per-owner k order remain.

Measured solver choice
----------------------
The actual first-SCF GGFF Hamiltonian was captured from the frozen previous
GPU binary. Typed hipSOLVER full D&C and index-range subset solvers were
compared on identical phased input, five timed repeats after warmup. With
one active rank per GPU, full heevd took 0.028894 s versus 0.293575 s for
subset heevdx. With all 32 ranks active, times were 0.260635 and 0.670284 s.
Residuals were <=1.31e-16 and eigenvalue differences <=4.22e-15. The full
D&C method is retained; computing fewer pairs was slower on this workload.

Validation
----------
* 12,001 owner schedules: device limits, full-node concurrency, uneven
  queues, empty owners, stable ownership/order and invalid input.
* Deferred-copy constructor model under ASan/UBSan: SCF/frame, phase,
  geometry and slot invalidation; budget decline; allocation failure;
  device-safe release and deviceless CPU cleanup.
* Concurrent density-wave model under ASan/UBSan: missing panels,
  empty owners, local/remote failures after partial progress, rollback,
  input retention and recovery-mode gating.
* The band lifecycle fixture executes the actual CU construction-cache
  release on two devices and two independent calculation contexts for 100
  cycles; collinear/noncollinear and CUDA/HIP variants pass.
* Eight targeted HIP host CTests, existing 243 architecture/release/entry
  policy cases, 607,303 GPU-worker assertions and 35,871 multi-GPU runtime
  assertions pass. The existing unregistered k-order fixture needs an
  ownership-boundary adapter after the earlier explicit-state migration;
  with that private fixture adapter, its 25,391 schedule layouts pass.
* Actual MI300A wrapper checks: real/complex, complete/partial spectra,
  residuals, orthogonality, finite checks, NaN failure preserving caller
  buffers, device/host/OpenMP-mapped arrays, repeated cache reuse, explicit
  releases and two host threads. Default, forced retry, blocking completion
  and retry/cache-disabled configurations pass. The standalone fixture
  now handles APU unified memory and links the actual environment registry.
* Changed native units compile for gfx942 and gfx1200. Environment registry
  and generated English/Japanese tables agree (2,252 host checks).

Five-SCF diagnostics are deliberately unconverged and do not establish
physical agreement. Full-input energy/force checks are reported separately.
Original geometry, basis, grids, SCF/MD limits and thresholds are preserved
in the full measurements. The original 25-SCF sidia333 deck remains an
unconverged comparison; it is never presented as a converged calculation.

GitHub Actions on test-fix commit 25b45161: all 18 jobs pass, including
CUDA/HIP native compilation, GPU contracts, CPU applications and macOS/
Linux host lifecycle tests. The native executable remains the validated
5925e71a production code; 25b45161 changes only the lifecycle fixture.

Converged/full-input measurements
--------------------------------
The historical CPU and previous GPU results are reused unchanged. Only
the optimized ordinary-gpusolver path is newly measured. All source,
executable and library identities are in manifest.json.

Workload                  Previous GPU(s) Optimized GPU(s) Speedup
S                                 140.310          153.200   0.916
L                                3032.940         2187.490   1.386
sidia333_nc_cluster               133.143          126.599   1.052
cghelgeo                          344.113          332.392   1.035

GGFF stages (full original input, standalone runs):
cache_full: elapsed=649.224 s, diagonalization=529.666 s, SCF=101
density_full: elapsed=304.061 s, diagonalization=183.937 s, SCF=101

Five-SCF diagnostics (unconverged, not accuracy evidence):
cache_all                  elapsed=  49.993 s, diagonalization=  33.450 s
cache_all_cap2             elapsed=  59.163 s, diagonalization=  42.156 s
cache_all_cap4             elapsed=  53.022 s, diagonalization=  36.504 s
cache_build_only           elapsed=  51.108 s, diagonalization=  34.376 s
cache_legacy               elapsed=  89.417 s, diagonalization=  72.962 s
cache_schedule_only        elapsed=  50.646 s, diagonalization=  34.511 s
density_all                elapsed=  26.834 s, diagonalization=  10.061 s
density_cap1               elapsed=  45.778 s, diagonalization=  29.521 s
density_cap4               elapsed=  30.861 s, diagonalization=  14.560 s
density_no_build           elapsed=  27.192 s, diagonalization=  11.176 s
density_no_panels          elapsed= 102.549 s, diagonalization=  86.048 s
density_serial             elapsed=  51.922 s, diagonalization=  33.238 s
schedule_cap1              elapsed=  67.421 s, diagonalization=  50.914 s
schedule_cap2              elapsed=  58.127 s, diagonalization=  41.677 s
schedule_cap4              elapsed=  52.537 s, diagonalization=  36.204 s
schedule_cap8              elapsed=  49.522 s, diagonalization=  33.291 s
schedule_legacy            elapsed=  93.214 s, diagonalization=  77.192 s

All 96 final-vs-reference comparisons pass 1e-7 Ha / 1e-7 Ha/bohr.
Maximum |delta E|=5.84650479141e-08 Ha; maximum |delta F|=3.6312e-08 Ha/bohr.

Matched S timing control, separate physical node a04:
old1: 144.42 s (48525.pbs)
optimized: 144.76 s (48526.pbs)
old2: 140.03 s (48526.pbs)
The old/new/old sequence spans 144.42/144.76/140.03 seconds; the historical
a05 S slowdown is not reproduced at that magnitude. This small sample does not
establish identical performance. One old1 control H2O-EF result differs from
the historical reference by 4.97e-6 Ha and 1.29e-6 Ha/bohr, with NormRD
stalled at 8.59e-6. The same old executable on the same node in old2 and the
optimized run agree with the historical reference. Diagnostic checks and raw
outputs are retained separately; none replace the strict primary a05
comparisons. The first control job stopped on that historical comparison, then
the completed control was reused when the sequence resumed.

MCCN completion-policy control, same physical node a04 (67 SCF each):
default: 164.762 s (48527.pbs)
sync: 169.040 s (48527.pbs)
previous: 223.333 s (48528.pbs)
The current binary with batched completion took 164.762 s, versus 169.040 s
with it disabled. The previous executable took 223.333 s in the later
same-node control; its historical 145.832 s was not reproduced. All three
controls pass the original energy/force tolerances. These limited controls do
not isolate the reason for historical timing changes, and provide no evidence
that disabling batched completion improves this Krylov case. No additional
small-matrix threshold was introduced on speculation.

All campaign jobs, including repaired test/launcher failures, consumed at most
6.75 points (each full-node wall time rounded up to 0.01 points). The
authorized cap was 40. No campaign GPU job remains. Initial harness failures
were an omitted -nt launch argument and assumptions that OpenMP host/device
mappings are separate on an APU; final numerical checks passed after the
fixtures were corrected.

Sirius MI300A benchmarks, 2026-10-31, 32 MPI x 3 OpenMP per physical node
=========================================================================

Scope: native -runtest, -runtestL and the versioned sidia333_nc_cluster.dat
and cghelgeo.dat decks. Every measurement uses one physical four-APU node
(32 ranks, 96 CPU cores, four GPUs). All ranks have three disjoint CPU places.
All 12 route/workload measurements and 96 calculations finished.
The user removed L2/L3 after the project-budget stop. Matching completed
CPU measurements and the final GPU S measurement were retained; only
the three missing GPU measurements were run after the scope change.
No separate gpusolver2 or gpusolver3 benchmark was requested or run.

Three routes: official OpenMX 4.0 plus official 4.0.1 fixes (CPU elpa2),
the unified GPU fork with CPU elpa2/OPENMX_GPU=0, and the fork with
ordinary gpusolver/OPENMX_GPU=1. Both fork routes use frozen executables
built after the ELPA OpenMP fix and remote merge, with identical shared DSOs.
All GPU measurements include remote 73c4a2c0, which guards
GPU ELPA handle destruction without changing numerical operations.
GPU S was repeated with that guard at the user request; its older result is excluded.
Exact source and executable versions are retained for each workload.
All use AMD ROCm 7.2.0 Clang/flang,
OpenMPI 5.0.10 and AOCL 5.1.0 (OpenMP BLIS preloaded first, three threads).
The official code retains ELPA 2018.05.001;
the fork uses ELPA 2026.02.002. Both ELPA libraries enable OpenMP.
Fork CPU handles explicitly use three threads; GPU ELPA host teams
remain one thread. The fork initializes MPI with MPI_THREAD_SERIALIZED.
Exact compiler/link flags,
source revision and executable/DSO hashes are in the manifests.
The Krylov fix binds each OpenMP worker to its rank GPU and initializes
rocBLAS on the calling thread before workers start their first GEMM.
A 32-rank, three-thread native BLAS probe reproduced stale HIP 209 with
numerically exact products; upfront rocBLAS initialization eliminated it.
All eight fork route/workload measurements were rerun after enabling
ELPA OpenMP. Earlier fork measurements are excluded, including the old
GPU-only Krylov rerun. All earlier native-suite measurements, including
official CPU, are also excluded: the helper overwrote BLIS_NUM_THREADS
with 1 even when its parent requested 3. Earlier direct BLAS diagnostics
observed three workers but did not test the suite helper. Only the four
direct named CPU measurements are retained from before the helper fix.
The corrected helper preserves explicit BLAS settings. Every accepted
native suite records three BLAS threads before MPI and on every rank.
A separate native C60/GaAs sequence through that helper observed three
actual BLIS workers on all 32 ranks in all three routes (PBS 48495).
AOCL selection had also preferred serial BLIS at link time; the corrected
build selects MT BLIS directly. An untimed direct audit confirms three
workers without LD_PRELOAD. Production keeps the common MT preload.
The production patch and validation records are retained alongside results.

Every physical input control, including SCF/MD limits, convergence criteria,
grids, geometry and basis, is preserved. Only deployment paths and the
CPU/GPU eigensolver selector differ. The 27 PAO and 20 VPS files have
identical SHA-256 values across all routes. The collector checks all atom
identities, coordinates, grids, energies and every force component.

Timing: sum of native per-case runtest result rows for each whole suite;
Elapsed.Time. in the output for named inputs. Launcher wall times are
retained separately. Each suite uses its native flag in one persistent
MPI process set. Prefixes of failed suites are never accepted as complete.
These are single measurements on allocated, exclusive physical nodes.
phases.csv retains the native minimum/maximum MPI-rank stage times; nested
stages and maxima from different ranks must not be added as a wall time.
OPENMX_GPU_VERBOSE=1 retains solver dispatch/timing diagnostics. Extra
Krylov stage profiling, GEMMul8 profiling and allocation tracing are off.

Additional measurement resource use is recorded separately in
additional_measurement_resources.json; the reused nine measurements
did not need another allocation.

Native times (seconds):
workload                  nodes  upstream CPU    fork CPU    fork GPU
S                           1         61.97       43.58      140.31
L                           1       2219.35     1533.41     3032.94
sidia333_nc_cluster         1        247.52      192.89      133.14
cghelgeo                    1        860.99      792.17      344.11

Numerical comparisons within 1e-7 Ha and 1e-7 Ha/bohr: 96/96.
Completion and numerical acceptance are reported separately. Fixed-SCF
decks, including sidia333 (25 SCF steps),
are not described as converged. Native stdout convergence statements
are retained for every MD step in summary.json and outputs.json.
SCF counts and every mismatch remain in cases.csv/comparisons.csv.

Observed one-node phase times (seconds; sums of native MPI-rank maxima):
  S Hamiltonian: fork CPU 6.581, GPU 63.296.
  S density grid: fork CPU 4.142, GPU 32.712.
  L diagonalization: fork CPU 1088.090, GPU 2090.796.
  sidia333_nc_cluster diagonalization: fork CPU 151.201, GPU 75.514.
  cghelgeo diagonalization: fork CPU 712.189, GPU 222.983.
  These measurements identify the expensive phases. They do not alone
  isolate kernel launch, memory movement, queue contention or communication.
  one-node-phases.json retains phase maxima for all 32 one-node cases per route.

Runtime settings and stability:
  OMP_NUM_THREADS=3, OMP_MAX_ACTIVE_LEVELS=1, OMP_DYNAMIC=FALSE
  BLIS_NUM_THREADS=3 (AOCL OpenMP BLIS), OPENBLAS_NUM_THREADS=1, MAGMA_NUM_THREADS=1
  HSA_XNACK=1, HSA_OVERRIDE_CPU_AFFINITY_DEBUG=0
  HSA_ENABLE_SDMA=1, HSA_NO_SCRATCH_RECLAIM=1
  LIBOMPTARGET_MEMORY_MANAGER_THRESHOLD=0
  UCX_TLS=self,cma,rc, UCX_RCACHE_ENABLE=n; default memory-type cache

The final per-measurement JSON files give the actual environment and
PBS allocation identities. Earlier UCX MM endpoint assertions occurred
in the official CPU code and the fork CPU/GPU code. Those attempts
were excluded. A separate GPU fault surfaced while waiting for native
on-demand orbital/density kernels; changing SDMA did not fix it.
The final GPU runs use HSA_NO_SCRATCH_RECLAIM=1. This retains scratch
on its queue; it is a tested runtime setting, not proof of fault cause.
ROCr reference: https://rocm.docs.amd.com/projects/ROCR-Runtime/en/docs-7.0.0/api-reference/environment_variables.html

GPU-enabled execution can select CPU eigensolvers for small matrices
or according to the normal capacity/sharing policy. Actual dispatch
must be read from retained run logs; no threshold or memory override
was used to force a benchmark route.

Full logs, PBS scripts, output files and interrupted attempts:
/work/GPU2026/dc1394/openmx4.0_gpu/work/codex_sirius_benchmark_20261031
prepare.py records the initial input snapshot; later build/launcher
updates are recorded by their exact source and binary versions in
manifest.json. fork-build-manifest.json records the final HIP toolchain
and upstream-build-manifest.json records the official CPU build.
The submitted PBS scripts retain the original local campaign paths.
Raw input locations and all data-file hashes are in manifest.json.
The original CPU scripts also ran superseded suite measurements;
only their direct named CPU measurements are accepted. job_scripts.json
maps every accepted measurement to its submitted PBS script.
The RSS sampling in suite helpers covers processes on the launcher node;
it is not a sum of remote-node memory usage.

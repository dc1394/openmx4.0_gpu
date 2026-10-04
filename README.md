# openmx4.0_gpu
## What does this code do?
This is a GPU-accelerated version of [OpenMX](https://www.openmx-square.org/), a first-principles calculation code based on numerical atomic orbitals (NAO). The dense eigensolvers of the band calculations (collinear and non-collinear) and of the cluster calculations (collinear and non-collinear), the O(N)-type solvers (DC, DC-LNO, Krylov), and the heavy matrix-construction stages around them (Hamiltonian/overlap assembly, charge-density and orbital grids, forces, charge mixing) are GPU-accelerated. Since v2.0, the cluster diagonalization can also be distributed over multiple GPUs and multiple nodes with ELPA (GPU kernels) + COSMA; see below.

## Code author
Hiroyuki Kawai (Niigata Univ.)</br>
X account: [@dc1394](https://x.com/dc1394)

## How to enable GPU acceleration
GPU acceleration is enabled by default: "scf.eigen.lib" defaults to "gpusolver", so no extra input line is required. Writing it explicitly is of course still fine:

```ini
scf.XcType                  GGA-PBE    # LDA|LSDA-CA|LSDA-PW|GGA-PBE
scf.SpinPolarization        off        # On|Off|NC
scf.ElectronicTemperature  300.0       # default=300 (K)
scf.energycutoff           150.0       # default=150 (Ry)
scf.maxIter                 40         # default=40
scf.EigenvalueSolver       band        # DC|GDC|Cluster|Band
scf.Kgrid                  9 9 9       # means n1 x n2 x n3
scf.Mixing.Type           rmm-diisk    # Simple|Rmm-Diis|Gr-Pulay|Kerker|Rmm-Diisk
scf.Init.Mixing.Weight     0.300       # default=0.30
scf.Min.Mixing.Weight      0.001       # default=0.001 
scf.Max.Mixing.Weight      0.700       # default=0.40 
scf.Mixing.History          7          # default=5
scf.Mixing.StartPulay       5          # default=6
scf.criterion             1.0e-10      # default=1.0e-6 (Hartree) 
scf.eigen.lib             gpusolver     # default=gpusolver
```

To run the conventional CPU paths instead, specify "elpa2" or "elpa1":

```ini
scf.eigen.lib             elpa2         # CPU (ELPA2) paths
```

A run that finds no usable GPU demotes itself to ELPA2 automatically, so the default is also safe on machines without an NVIDIA GPU. The environment variable `OPENMX_GPU=0` forces the same demotion on a machine with a GPU, which is convenient for CPU-vs-GPU comparisons with unmodified input files.

## New in v2.0: distributed multi-GPU cluster diagonalization (ELPA GPU + COSMA)
A cluster calculation has only one k-point, so with the default "gpusolver" its dense eigenvalue problem is solved on a single GPU and, on a multi-GPU machine or a multi-node GPU cluster, the other GPUs idle during the diagonalization (see "Multi-GPU parallelization" below). Since v2.0, selecting

```ini
scf.EigenvalueSolver       cluster
scf.eigen.lib              gpusolver2    # ELPA (GPU kernels) + COSMA
```

replaces the cluster diagonalization (collinear and non-collinear) with [ELPA](https://elpa.mpcdf.mpg.de/) 2026.02 (NVIDIA GPU kernels) and [COSMA](https://github.com/eth-cscs/COSMA) (communication-optimal distributed matrix multiplication, GPU backend), engaging every MPI rank and every GPU. This is the option for machines where the diagonalization must not be confined to one GPU, e.g. one GPU per node × many nodes. Notes:

- "gpusolver2" supports only `scf.EigenvalueSolver cluster`; any other solver stops with an input error. Everything outside the cluster diagonalization behaves exactly like "gpusolver", and a machine without a usable GPU demotes itself to ELPA2 as usual.
- When the GPU memory cannot hold a solve, that solve automatically falls back to the ELPA CPU kernels / ScaLAPACK instead of aborting the run.
- The required libraries (ELPA, COSMA, COSTA, Tiled-MM) are bundled as source archives under `source/third_party/dist/` and are built automatically by the first `make` — no network access is needed (autoconf, automake, libtool, m4 and python3 must be installed). The conventional "elpa1"/"elpa2" paths keep using the ELPA 2018.05 embedded in the OpenMX source; only "gpusolver2" links the bundled ELPA 2026.02.

## GEMMul8: FP64 matrix multiplication on integer tensor cores
The large dense matrix multiplications of the GPU eigensolver path are executed through [GEMMul8](https://github.com/RIKEN-RCCS/GEMMul8), which emulates FP64 GEMM on the INT8 tensor cores using the Ozaki scheme II. This is enabled by default and is particularly effective on consumer GPUs (GeForce), whose native FP64 throughput is limited, while the total energy stays at the ~1e-10 Hartree agreement level in our tests. To compare with plain cuBLAS FP64 GEMM, it can be switched off (keyword added in v2.0):

```ini
scf.gemmul8.enable         off           # default=on
```

## Multi-GPU parallelization
How many GPUs a run can actually use is bounded by the number of k-points requested with "scf.Kgrid". The MPI ranks are divided into one group per k-point, and the dense eigenvalue problem of each group is solved on a single GPU, so the eigenvalue solver keeps at most as many GPUs busy as there are k-points; any GPU beyond that number stays idle in this part of the calculation. (The Hamiltonian matrix elements and the grid work are distributed over all MPI ranks, and therefore over all GPUs.)

In particular, a cluster calculation — `scf.EigenvalueSolver cluster`, i.e. `scf.Kgrid 1 1 1` — has only one k-point, so with the default "gpusolver" **only one GPU is used for the diagonalization however many GPUs the node has**. Adding GPUs, or raising the upper bound `scf.Gpu.Num` (default 30, which effectively means "use every GPU found"), does not make such a run faster. (A collinear spin-polarized calculation solves the two spins in separate MPI worlds, so it can occupy two GPUs at most.)

Multiple GPUs therefore pay off for band calculations with a k-mesh; for a cluster calculation, either give the job one GPU and more CPU cores / MPI ranks, or — since v2.0 — select `scf.eigen.lib gpusolver2` (see above), which distributes the cluster diagonalization itself over all ranks and all GPUs.

## MPI vs. hybrid (MPI/OpenMP) parallelization
Use flat MPI. In OpenMX 4.0 GPU, hybrid MPI/OpenMP parallelization is not effective: OpenMP threads can still be requested as usual with the `-nt` option and such runs complete correctly, but they bring no speedup — only MPI parallelization is effective. Assign all the cores you want to use to MPI ranks instead, with one OpenMP thread per rank:

```sh
mpirun -np 16 ./openmx input.dat -nt 1
```

## NVIDIA MPS: recommended whenever several ranks share a GPU
With flat MPI, all the MPI ranks of a node normally share one GPU — so run the
[NVIDIA CUDA Multi-Process Service (MPS)](https://docs.nvidia.com/deploy/mps/).
Without MPS the kernels of the different ranks are serialized by context
switching on the device; under MPS they execute concurrently.
The effect is large: with 48 ranks sharing one H100 PCIe, the `-runtest`
suite below completes in 108.4 s without MPS and 90.2 s with it (17% faster;
mean of two back-to-back A/B pairs on the same node), and the `-runtestL`
suite drops from 3684.7 s without MPS to 1481.6 s with it — 2.5x faster.
Without MPS the 48 time-sliced CUDA contexts make the GPU `-runtestL` run
even slower than the CPU-only run (2203.6 s), so on bigger systems MPS is
not a tweak but a requirement for the GPU to pay off.
On the 18-rank RTX 5080 PC the same A/B gives 98.9 s → 90.2 s for `-runtest`
and 5005.4 s → 4361.9 s for `-runtestL` (13% faster; up to 1.6x on individual
krylov inputs), so fewer ranks per GPU soften the no-MPS penalty, but MPS
still wins everywhere we measured.
Start the control daemon once per node before `mpirun`, and stop it afterwards:

```sh
nvidia-cuda-mps-control -d             # start the MPS control daemon
mpirun -np 48 ./openmx input.dat -nt 1
echo quit | nvidia-cuda-mps-control    # stop it
```

On a multi-node batch job, start one daemon on every node (`/tmp` is usually
node-local, so per-node `CUDA_MPS_PIPE_DIRECTORY`/`CUDA_MPS_LOG_DIRECTORY`
paths work well). MPS requires native Linux; it is not available under WSL2.
An MPS server also accepts at most 48 client processes per GPU. With more
ranks than that on one GPU the extra ranks find no device, and OpenMX then
demotes the whole run to the CPU paths (the log reports `GPU initialization
failed on 16 of 64 MPI ranks; using ELPA2`), so keep at most 48 ranks per GPU
under MPS; a larger rank count can use the GPU only without MPS (see the
Kugui columns below).
The benchmark tables below list the GPU columns of all three machines without and with MPS.

## Build and install
Building and installing is more difficult than with standard OpenMX. The build requires the [NVIDIA HPC SDK](https://developer.nvidia.com/hpc-sdk) and OpenMPI. The Makefile contains build examples for several supercomputer systems, and ready-made site makefiles are included for the Pegasus supercomputer at the University of Tsukuba (`Makefile.pegasus`) and for System C "Kugui" at ISSP, Univ. of Tokyo (`Makefile.kugui`, which builds with the NVHPC 24.7 / CUDA 12.5 that Kugui offers; its header lists the two nvc 24.x code-generation problems it works around, one of them reproduced by `tests/nvc_diag_vectorizer_bug.c`); please refer to them. Since v2.0 the first `make` also builds the bundled ELPA/COSMA stack for "gpusolver2" automatically, which adds some time to the first build. A detailed implementation document (English and Japanese, including the list of GPU-related environment variables) is available under [doc/](doc/). If you're unsure about the build and installation process, feel free to ask in English via GitHub issues or [my X account](https://x.com/dc1394) (Japanese is also acceptable on my X account). I'll assist you as much as I can.

### HPC SDK 26.9 / CUDA 13.4 update

The default `source/Makefile` now uses HPC SDK 26.9 and CUDA 13.4. The
SDK's bundled toolkit and the separately installed toolkit can differ: on
this machine SDK 26.9 bundles CUDA 13.3, while CUDA 13.4.2 is installed at
`/usr/local/cuda-13.4`. The build selects one consistent set of CUDA headers,
runtime and math libraries, and obtains the MPI library path from the SDK's
MPI wrapper. Compiler/toolkit changes invalidate cached build products.

```sh
cd source
make -j18 GEMMUL8_GPU_ARCH=120             # RTX 5080; use 90 for H100
# An older installation remains selectable:
make -j18 NVHPC_ROOT=/opt/nvidia/hpc_sdk/Linux_x86_64/26.5 \
  NVHPC_CUDA_VERSION=13.2 \
  NVHPC_CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/26.5/cuda/13.2 \
  GEMMUL8_GPU_ARCH=120
```

GEMMul8 is pinned to upstream **v3.5.2**
(`603b52363715796a0af5e4aa1ed8d386349b4251`). Initialize/update submodules
with `git submodule update --init --recursive` after updating the repository.
The OpenMX adapter includes v3.5.2's memory-saving configuration and K-block
modular-reduction implementation. The adapter also enables memory-saving
blocking with a 256 MiB default target (`OPENMX_GEMMUL8_MAX_WORKSPACE_MB`),
and divides the available-memory budget between ranks sharing each GPU.

cuSOLVER handles enable FP64 fixed-point emulation with the performant
strategy and dynamic mantissa selection. With CUDA 13.4 Update 1 or later
(cuSOLVER 12.3.4+), complete-spectrum real and complex dense solves use
`CUSOLVER_ALG_0`: cuSOLVER automatically chooses the one-stage or two-stage
reduction from the matrix dimension, GPU architecture and math mode. No
fixed starting dimension or startup calibration is required. Partial-spectrum
requests retain SYEVDX, avoiding an unnecessary full eigendecomposition.
Explicit `two-stage` mode also supports leading partial spectra by computing
the full spectrum and returning the requested eigenpairs. The library chooses Ozaki-II when
beneficial; there is no public API to force Ozaki-II for every internal GEMM.
See [NVIDIA's cuSOLVER documentation](https://docs.nvidia.com/cuda/cusolver/)
and [CUDA release notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/).
Older cuSOLVER headers/libraries retain the existing SYEVD/SYEVDX path, and unsupported
algorithm selection or emulation modes fall back to the compatible implementation.
The cuSOLVER layer compiles against CUDA 12.3 as well; the complete bundled
GEMMul8 build requires newer cuBLASLt APIs, as the previous bundled version
already did. CUDA 13.2 and 13.4 GEMMul8 builds are verified.

For controlled comparisons:

```sh
OPENMX_CUSOLVER_VERBOSE=1 mpirun -np 18 ./openmx input.dat -nt 1
OPENMX_CUSOLVER_ALGORITHM=legacy OPENMX_CUSOLVER_EMULATION=0 \
  mpirun -np 18 ./openmx input.dat -nt 1
```

`OPENMX_CUSOLVER_ALGORITHM` accepts `auto` (default), `one-stage`,
`two-stage` and `legacy`. For a manual crossover,
`OPENMX_CUSOLVER_TWO_STAGE_MIN_N` alone selects two-stage dispatch at or
above that dimension; smaller full spectra use one-stage and smaller partial
spectra use SYEVDX. Explicit `auto` takes precedence over this manual minimum.
The automatic crossover belongs to cuSOLVER and can change with its version
and the GPU. `OPENMX_CUSOLVER_CACHE_MB` caps retained generic-solver
scratch per rank (default 64 MiB; 0 releases scratch after every solve).
`OPENMX_CUSOLVER_EMULATION=0` disables cuSOLVER emulation;
`scf.gemmul8.enable` independently controls OpenMX's GEMMul8 products.

The AMD version's run-boundary cache cleanup has been ported to the CUDA
band, cluster, DC, DC-LNO and Krylov paths. Small density-grid tables are
retained across SCF iterations; the defaults cap the global and local caches
at 64 MiB and 32 MiB, with a further free-memory allowance per sharing rank.
Set `OPENMX_DENSITY_GRID_GPU_CACHE_MB=0` and
`OPENMX_DENSITY_GRID_GPU_LOCAL_CACHE_MB=0` to disable this retention for an
A/B comparison. Nonlocal spin-orbit force assembly also avoids redundant
pointer traversal and clears only the active orbital block.

Collinear and noncollinear multi-k band calculations also retain transformed
overlap matrices across SCF iterations and reuse eigenvectors between the
eigenvalue and density-matrix passes. Host memory is bounded by the per-rank
limits below and the available node memory divided among local ranks. Uncached k points use
the existing solve path; eigenvector panels expire after each SCF call, and
overlap entries expire when the geometry or input changes.

| Setting | Default per-rank cap |
|---|---:|
| `OPENMX_BAND_COL_OVERLAP_CACHE_MB` | 512 MiB |
| `OPENMX_BAND_COL_KCACHE_MB` | 1024 MiB |
| `OPENMX_BAND_NONCOL_OVERLAP_CACHE_MB` | 512 MiB |
| `OPENMX_BAND_NONCOL_KCACHE_MB` | 1024 MiB |
| `OPENMX_BAND_NONCOL_PACK_CACHE_MB` | 128 MiB |

Set a cap to `0` to disable that cache. Each collinear cache and the
noncollinear overlap cache is further capped at 1/64 of available node RAM
divided by the number of local MPI ranks; the noncollinear eigenvector cache
uses 1/32. Overlap caches reset at the first SCF iteration and system cleanup;
noncollinear entries also require matching matrix dimensions and k points.
The noncollinear packed-matrix cache gathers the eight real-space matrices
once per SCF and reuses them across k-point groups and both passes. It is
also limited to 1/64 of available node RAM divided by local ranks. All owners
must fit; otherwise the existing per-group gathers remain in use. Its storage
is released at the end of each SCF call.
Noncollinear k-point groups interleave their owner ranks to use the admitted
GPU concurrency while preserving each rank's original k-point accumulation
order. `OPENMX_BAND_NONCOL_INTERLEAVE_K=0` restores consecutive k-point groups
for timing comparisons.
`OPENMX_BAND_CACHE_TRACE=1` and
`OPENMX_BAND_NONCOL_CACHE_TRACE=1` report reuse counts. Solver and GEMMul8
scratch is released before force evaluation. Band runs release shared orbital
and density caches after Force3; `OPENMX_FORCE_RELEASE_SCF_CACHES=0` disables
this latter release for comparisons.

When no complete Hamiltonian rank fits on its GPU, or only one rank would
use the GPU while its peers fall back to the CPU, matrix construction
automatically streams bounded pair batches through the CUDA kernel.
For the one-rank case, explicit serial waves and a one-rank GPU limit keep
their existing schedule. If no complete rank fits, streaming is still tried.
`OPENMX_SETHAM_STREAM_MB` caps the workspace per rank (default 256 MiB;
values must be at least 8 MiB), with a further limit from each rank's share
of free GPU memory after reserving space for the potential and runtime.
Oversized pairs or recoverable CUDA failures use the CPU for those pairs.
`OPENMX_SETHAM_STREAMING=0` disables this fallback; it also requires the
CUDA matrix kernel to be enabled (`OPENMX_SETHAM_CUDA_KERNEL`, default 1).

VNA projector construction reduces its pair batch to fit each rank's GPU
memory budget. The HVNA contraction also admits ranks in turns when their
combined buffers do not fit. This keeps the existing kernel and accumulation
order; only the scheduling and archive placement change. Turn execution
uses host archives with a node-wide budget of half of available RAM,
and falls back to the CPU if one rank cannot fit. Set
`OPENMX_SETPRO_HVNA_TURNS=0` to disable rank turns, or use the existing
`OPENMX_SETPRO_GPU=0` to disable the VNA GPU stages.

For VNA forces, `OPENMX_FORCE4B_CASE2_STREAM=auto` (also the unset default)
streams case-2 projector traces one centre at a time when the full GPU batch
does not fit. Set it to `0` to disable streaming or `1` to request streaming
even when the full batch fits. Memory admission still applies, with CPU
fallback if a rank's largest centre does not fit; `OPENMX_FORCE4B_GPU=0`
disables both the full-batch and streamed GPU paths.
For case 1, `OPENMX_FORCE4B_CASE1_TURNS=auto` (the unset default) can
retain received projector rows in host memory and run bounded groups of ranks
on each physical GPU when concurrent full archives do not fit. Group size
uses the free memory measured after earlier buffers are released, the largest
rank arena, a 256 MiB reserve and 64 MiB allowance per active rank.
`OPENMX_FORCE4B_CASE1_TURN_MAX_RANKS=1` restores one rank at a time for comparisons.
Each rank
uploads its local rows and halo once, using the existing kernel. The added
host allocations must fit both three quarters of available node RAM and
the amount remaining after reserving 32 GiB. Device allocation failure uses
the cached halo in the CPU contraction. Set this mode to `0` to disable it
or `1` to force it; `OPENMX_FORCE4B_CASE1_STREAM=1` takes precedence.
`OPENMX_FORCE4B_CASE1_STREAM=auto` similarly streams case-1 traces when the
full archive and rank turns are unavailable. It packs only the projector rows needed for each
received source, then accumulates results in the original atom/pair order.
`OPENMX_FORCE4B_CASE1_STREAM_MB` caps its per-rank arena at 1024 MiB by default;
`0` prevents streaming. The case-1 mode also accepts `0` (disabled) and `1`
(forced), independently of the case-2 mode. Both respect the parent GPU switch
and fall back to the CPU if their workspace cannot fit.

Large collinear cluster solves release disposable matrices, eigenvector panels
and solver workspace after forming the density matrix when less than half of
the GPU memory is free. The transformed overlap stays cached. Memory admission
is checked again before the next solve, including serialized spin execution.
`OPENMX_CLUSTER_GPU_RETAIN_SCRATCH=1` keeps the previous retention policy for
comparisons; `0` releases scratch at every SCF boundary, and leaving it unset
uses the automatic policy. `OPENMX_CUSOLVER_VERBOSE=1` reports released scratch.

`OPENMX_DCLNO_PROFILE=1` reports collinear DC-LNO CPU/GPU task counts,
matrix-dimension ranges and histograms, and SCF phase times aggregated as
MPI-rank maxima and means. Profiling is off by default. The existing
`OPENMX_DCLNO_GPU_THRESHOLD` overrides the collinear local-matrix GPU
crossover (default dimension 800) for controlled comparisons.
Collinear DC-LNO keeps LAPACK eigenvectors in their packed layout and stores
each atom/spin residue window in one contiguous allocation, preserving the
existing row layout and accumulation order.
If a local cuSOLVER solve returns positive INFO (numerical nonconvergence),
it retries that generalized eigenproblem on the CPU using the original host
matrices and existing workspace. The optional proxy path returns the retry
to the originating rank. Invalid arguments, API errors and inconsistent
eigenpair counts remain fatal. Profiling counts the initially selected backend;
its solve time includes any CPU retry, which is also reported in the log.

### Reproducible GPU regression runs

`tools/run_gpu_suites.py` and `tools/compare_gpu_suites.py` adapt the AMD
version's isolated suite runner and force-component comparison. They cover
all four built-in suites (`S`, `L`, `L2`, `L3`), preserve input/reference files,
and record binary hashes, launch settings, elapsed time, memory usage and
interruptions. Each input is a separate MPI job invoking its original
`-runtest*` option; this mode does not measure cache reuse between inputs.
Add `--together` to execute all selected inputs in one native suite call and
check cache cleanup between systems; the comparator supports both layouts.

```sh
python3 tools/run_gpu_suites.py --binary source/openmx --suite S --ranks 18 \
  --mpirun /opt/nvidia/hpc_sdk/Linux_x86_64/26.9/comm_libs/mpi/bin/mpirun \
  --output work/bench_after_S
python3 tools/compare_gpu_suites.py work/bench_after_S --reference
# After recording an equivalent before run:
python3 tools/compare_gpu_suites.py work/bench_before_S work/bench_after_S
```

Use identical MPI ranks, MPS settings and input data for timing comparisons.
The comparator checks every force component, since the built-in scalar
`diff Force` can hide cancellation. Large tests stop individually if the
memory reserve or per-input timeout is reached; interrupted cases are
reported as incomplete, never as passes.

Standalone GPU accuracy checks are also included:

```sh
tests/run_cusolver_smoke.sh 257
tests/run_gemmul8_smoke.sh
# Check bounded Hamiltonian batches, buffer reuse and recoverable overflow:
tests/run_set_hamiltonian_stream_smoke.sh
# Add CUDA device memory checking:
SETHAMILTONIAN_SANITIZE=1 tests/run_set_hamiltonian_stream_smoke.sh
# Exercise a GPU first SCF step followed by a memory-admission CPU fallback:
tests/run_cluster_fallback_smoke.sh
```

The scripts support `CUDA_HOME`, `CUDA_MATH_ROOT` and `NVHPC_ROOT` for
cross-version checks. They isolate the selected CUDA runtime from unrelated
`LD_LIBRARY_PATH` entries. GEMMul8's template translation unit takes several
minutes to compile. The cluster fallback injection check requires CUDA 13.4
Update 1 or later and a built OpenMX executable.

CPU checks for the new packing, allocation and scheduling paths use functions
extracted from the production sources:

```sh
python3 tests/run_force4b_stream_cpu.py
python3 tests/run_force4b_case1_cpu.py
python3 tests/run_force4b_case1_turns_cpu.py
python3 tests/run_dclno_residue_pool_smoke.py --sanitize
python3 tests/run_dclno_solver_retry.py --sanitize
python3 tests/run_band_noncol_k_order_cpu.py
```

These check arithmetic against independent CPU contractions, buffer boundaries,
allocation failure, empty ranks and preservation of each owner's k-point order.
The solver retry check injects CUDA failures through host stubs and checks the
recovery against independent LAPACK generalized eigenpairs; it needs LP64
LAPACK/BLAS libraries (`LAPACK_LIBS` can override `-llapack -lblas`).
They complement the GPU/MPI suite runs. `tools/bench_dclno_solver.py --build`
also prepares a CPU/GPU crossover benchmark using the production DC-LNO solver
functions and existing libraries. It prints the separate MPI launch command;
synthetic-matrix timings guide experiments and do not change the default
threshold. Use the same thread, MPI binding and MPS settings as the suite runs.
The timing harness rejects GPU nonconvergence instead of reporting CPU retry
time as a GPU measurement.

Measured results, accuracy checks, version coverage and remaining limitations
are recorded in [the CUDA 13.4 validation report](doc/cuda_13_4_validation.txt)
and [the subsequent GPU tuning measurements](doc/gpu_tuning_20261001.txt).

## Docker image
I have released the OpenMX 4.0 GPU Docker image.
You can easily try OpenMX 4.0 GPU on computers equipped with NVIDIA GPUs.
The steps are as follows:
1. Install [Docker](https://docs.docker.com/get-started/get-docker/).
2. Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
3. Run: `docker run --gpus all --shm-size=4gb --rm -it -v /path/to/inputs:/work dc1394/openmx4.0-gpu-ubuntu24.04:1.1.1`. Ensure `/path/to/inputs` is created beforehand.
4. Run tests with `cd openmx_work` and `mpirun -np 4 ./openmx -runtest`.

This should yield results like the following:

```ini
   1  input_example/Benzene.dat        Elapsed time(s)=    6.16  diff Utot= 0.000000000001  diff Force= 0.000000000000
   2  input_example/C60.dat            Elapsed time(s)=   10.61  diff Utot= 0.000000000030  diff Force= 0.000000000002
   3  input_example/CO.dat             Elapsed time(s)=    7.95  diff Utot= 0.000000000000  diff Force= 0.000000000004
   4  input_example/Cr2.dat            Elapsed time(s)=    8.37  diff Utot= 0.000000000000  diff Force= 0.000000000003
   5  input_example/Crys-MnO.dat       Elapsed time(s)=   13.44  diff Utot= 0.000000000013  diff Force= 0.000000000002
   6  input_example/GaAs.dat           Elapsed time(s)=   20.83  diff Utot= 0.000000000023  diff Force= 0.000000000001
   7  input_example/Glycine.dat        Elapsed time(s)=    4.93  diff Utot= 0.000000000001  diff Force= 0.000000000000
   8  input_example/Graphite4.dat      Elapsed time(s)=    3.76  diff Utot= 0.000000000002  diff Force= 0.000000000001
   9  input_example/H2O-EF.dat         Elapsed time(s)=    4.69  diff Utot= 0.000000000001  diff Force= 0.000000000001
  10  input_example/H2O.dat            Elapsed time(s)=    4.11  diff Utot= 0.000000000000  diff Force= 0.000000000001
  11  input_example/HMn.dat            Elapsed time(s)=   12.77  diff Utot= 0.000000000000  diff Force= 0.000000000000
  12  input_example/Methane.dat        Elapsed time(s)=    3.43  diff Utot= 0.000000000058  diff Force= 0.000000000001
  13  input_example/Mol_MnO.dat        Elapsed time(s)=    8.31  diff Utot= 0.000000000001  diff Force= 0.000000000000
  14  input_example/Ndia2.dat          Elapsed time(s)=    4.61  diff Utot= 0.000000000004  diff Force= 0.000000000000


Total elapsed time (s)      113.97
```

You can verify that the calculation is correct.

## Benchmarks
For benchmarks of GPU-accelerated OpenMX, please refer to the following literature.
https://journals.jps.jp/doi/10.7566/JPSJ.94.124003

However, the current version offers improved performance compared to the version described in this paper.

### Built-in test suites (-runtest / -runtestL)
The two standard OpenMX test suites were run on three machines with flat MPI (the PC and Pegasus with the NVIDIA HPC SDK 26.5 / CUDA 13.2, Kugui with NVHPC 24.7 / CUDA 12.5 through `Makefile.kugui`):

- **PC** — Core i9-10980XE (18 cores) + GeForce RTX 5080 (16 GB), 18 ranks sharing the single GPU;
- **Pegasus (CCS, Univ. of Tsukuba) node** — Xeon Platinum 8468 (48 cores) + H100 PCIe (80 GB), 48 ranks sharing the single GPU;
- **Kugui (ISSP, Univ. of Tokyo) node** — AMD EPYC 7763 (64 cores) + A100-SXM4 (40 GB), one of the node's four GPUs; 64 ranks share it in the CPU and no-MPS columns, and because MPS serves at most 48 clients per GPU, the MPS column uses 48 ranks and is paired with a 48-rank CPU column.

On each machine the same binary was used for all of its columns, and the GPU suites were run twice — without CUDA MPS (time-sliced contexts) and with it (see above). `OPENMX_GPU=0` demotes the whole run to the CPU (ELPA2) paths (the Pegasus CPU jobs additionally had no GPU allocated at all), and the GPU runs use the input-file defaults (`scf.eigen.lib gpusolver`, GEMMul8 on):

```sh
# GPU (defaults; GEMMul8 enabled); N = 18 on the PC, 48 on the Pegasus node, 64 or 48 on the Kugui node.
# On all three machines the GPU suites were run twice: with the MPS daemon up
# ("MPS" columns) and without it ("no MPS" columns).
mpirun -np N ./openmx -runtest  -nt 1
mpirun -np N ./openmx -runtestL -nt 1
# CPU reference (same binary)
OPENMX_GPU=0 mpirun -np N ./openmx -runtest  -nt 1
OPENMX_GPU=0 mpirun -np N ./openmx -runtestL -nt 1
```

`-runtest` (14 small systems, 2–60 atoms; elapsed seconds from runtest.result):

| input | i9 CPU (s) | 5080 GPU, no MPS (s) | 5080 GPU, MPS (s) | Xeon CPU (s) | H100 GPU, no MPS (s) | H100 GPU, MPS (s) | EPYC CPU, 64 ranks (s) | A100 GPU, no MPS, 64 ranks (s) | EPYC CPU, 48 ranks (s) | A100 GPU, MPS, 48 ranks (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Benzene | 6.14 | 7.41 | 5.60 | 13.63 | 15.92 | 11.78 | 48.85 | 48.88 | 23.73 | 18.58 |
| C60 | 10.29 | 9.96 | 8.13 | 7.36 | 15.33 | 6.35 | 18.95 | 29.19 | 6.07 | 5.67 |
| CO | 7.97 | 9.13 | 8.13 | 7.25 | 8.34 | 7.47 | 28.44 | 28.90 | 6.67 | 6.93 |
| Cr2 | 8.07 | 7.96 | 7.12 | 8.31 | 8.44 | 7.90 | 34.81 | 35.30 | 6.60 | 5.43 |
| Crys-MnO | 13.07 | 9.73 | 9.25 | 10.84 | 7.08 | 6.44 | 38.66 | 32.38 | 12.48 | 6.18 |
| GaAs | 21.37 | 13.73 | 13.27 | 16.36 | 9.45 | 9.28 | 46.76 | 39.53 | 18.82 | 10.56 |
| Glycine | 4.86 | 5.38 | 4.61 | 4.65 | 5.51 | 4.59 | 16.73 | 18.03 | 3.42 | 3.46 |
| Graphite4 | 3.91 | 2.86 | 2.60 | 4.16 | 3.58 | 3.30 | 16.47 | 16.07 | 3.32 | 2.35 |
| H2O-EF | 4.49 | 4.49 | 4.29 | 4.79 | 4.87 | 4.54 | 16.36 | 16.68 | 3.06 | 3.23 |
| H2O | 3.85 | 3.88 | 3.44 | 4.37 | 5.21 | 4.47 | 19.08 | 18.13 | 3.13 | 3.10 |
| HMn | 12.82 | 11.26 | 11.07 | 10.53 | 10.33 | 10.00 | 34.47 | 32.96 | 9.45 | 7.32 |
| Methane | 3.15 | 3.16 | 2.95 | 3.64 | 3.81 | 3.44 | 15.16 | 15.23 | 2.37 | 2.38 |
| Mol_MnO | 8.17 | 7.42 | 7.26 | 7.50 | 7.27 | 6.99 | 32.16 | 31.96 | 6.83 | 5.48 |
| Ndia2 | 4.64 | 2.60 | 2.49 | 5.01 | 3.31 | 3.04 | 17.29 | 16.32 | 4.23 | 2.45 |
| **Total** | **112.80** | **98.94** | **90.21** | **108.41** | **108.45** | **89.59** | **384.19** | **379.56** | **110.18** | **83.12** |

These systems are far below the GPU/CPU switching thresholds of the dense eigensolvers, so the diagonalization automatically falls back to the CPU and only the GPU-accelerated matrix-construction stages differ — the point of this table is that the whole suite passes on the GPU build with the same accuracy as the CPU paths (max diff Utot ≤ 5.5e-11 Hartree in all six columns; on each machine the MPS and non-MPS runs report identical diffs). On these tiny systems MPS is what gives the GPU columns their edge (RTX 5080: 98.94 → 90.21 s; H100: 108.45 → 89.59 s); the elevated first one or two cases of each Pegasus column are the one-time warm-up of the batch node (input-file cache, CUDA context creation). The two 64-rank Kugui columns measure that node's file system rather than the GPU: with all 64 cores running ranks, writing the cube files (`OutData`) takes about 12 s per input against 1 s at 48 ranks, which is where their 384.19 / 379.56 s totals come from; the 48-rank pair shows the usual MPS edge (110.18 → 83.12 s). Max diff Utot ≤ 6.1e-11 Hartree in all four Kugui columns.

`-runtestL` (16 medium/large systems; each "ratio" column is CPU / MPS-on GPU on the same machine, for Kugui both at 48 ranks):

| input | atoms | solver | i9 CPU (s) | 5080 GPU, no MPS (s) | 5080 GPU, MPS (s) | ratio | Xeon CPU (s) | H100 GPU, no MPS (s) | H100 GPU, MPS (s) | ratio | EPYC CPU, 64 ranks (s) | A100 GPU, no MPS, 64 ranks (s) | EPYC CPU, 48 ranks (s) | A100 GPU, MPS, 48 ranks (s) | ratio |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5_5_13COb2 | 155 | band | 106.33 | 78.80 | 73.49 | 1.45 | 52.66 | 65.14 | 35.60 | 1.48 | 91.38 | 111.10 | 79.48 | 46.74 | 1.70 |
| B2C62_Band | 64 | band | 704.51 | 540.50 | 395.99 | 1.78 | 320.14 | 878.20 | 157.35 | 2.03 | 402.92 | 439.06 | 448.13 | 223.54 | 2.00 |
| CG15c-DC-LNO | 650 | dc-lno | 161.87 | 136.06 | 124.79 | 1.30 | 65.92 | 95.57 | 49.83 | 1.32 | 72.98 | 112.98 | 74.38 | 60.92 | 1.22 |
| DIA512-1 | 512 | krylov | 184.10 | 185.44 | 116.49 | 1.58 | 68.28 | 353.95 | 56.04 | 1.22 | 71.46 | 120.42 | 74.86 | 63.20 | 1.18 |
| FeBCC | 16 | band (sp) | 183.65 | 190.18 | 177.53 | 1.03 | 78.78 | 83.68 | 66.27 | 1.19 | 97.42 | 95.26 | 92.56 | 74.14 | 1.25 |
| GEL | 40 | band | 56.58 | 52.34 | 45.81 | 1.24 | 31.23 | 52.37 | 22.00 | 1.42 | 53.74 | 52.01 | 43.59 | 27.23 | 1.60 |
| GFRAG | 54 | cluster | 45.10 | 43.25 | 38.35 | 1.18 | 23.22 | 41.56 | 15.50 | 1.50 | 28.77 | 60.75 | 29.43 | 19.90 | 1.48 |
| GGFF | 40 | band (NC) | 1657.70 | 1304.17 | 1080.95 | 1.53 | 573.69 | 591.56 | 332.88 | 1.72 | 451.92 | 429.62 | 784.23 | 267.99 | 2.93 |
| MCCN | 564 | krylov | 313.57 | 327.93 | 209.61 | 1.50 | 123.87 | 176.05 | 95.15 | 1.30 | 128.55 | 212.09 | 137.69 | 104.72 | 1.31 |
| Mn12_148_F | 148 | cluster (sp) | 121.04 | 88.59 | 84.37 | 1.43 | 57.85 | 70.06 | 29.04 | 1.99 | 82.39 | 70.47 | 72.36 | 44.79 | 1.62 |
| N1C999 | 1000 | dc-lno (sp) | 1663.46 | 1568.67 | 1549.51 | 1.07 | 489.76 | 828.62 | 458.58 | 1.07 | 527.90 | 509.34 | 554.09 | 524.81 | 1.06 |
| Ni63-O64 | 127 | band (sp) | 104.33 | 68.53 | 65.27 | 1.60 | 53.18 | 112.07 | 24.09 | 2.21 | 66.94 | 70.16 | 65.75 | 44.70 | 1.47 |
| Pt63 | 63 | cluster | 83.42 | 60.11 | 56.40 | 1.48 | 31.42 | 74.37 | 23.43 | 1.34 | 48.23 | 106.00 | 42.01 | 40.09 | 1.05 |
| SialicAcid | 40 | cluster | 25.57 | 23.05 | 20.50 | 1.25 | 14.33 | 32.42 | 12.96 | 1.11 | 31.40 | 48.17 | 15.36 | 14.30 | 1.07 |
| ZrB2_2x2 | 76 | band | 307.99 | 222.72 | 211.77 | 1.45 | 137.12 | 142.96 | 68.31 | 2.01 | 175.53 | 126.31 | 171.55 | 102.71 | 1.67 |
| nsV4Bz5 | 64 | cluster | 138.14 | 115.06 | 111.08 | 1.24 | 82.15 | 86.10 | 34.59 | 2.37 | 104.74 | 107.12 | 100.44 | 60.09 | 1.67 |
| **Total** | | | **5857.35** | **5005.41** | **4361.90** | **1.34** | **2203.59** | **3684.69** | **1481.62** | **1.49** | **2436.27** | **2670.86** | **2785.91** | **1719.87** | **1.62** |

All 16 inputs pass on the GPU (GEMMul8 on) on both machines — max diff Utot = 2.0e-9 Hartree on the RTX 5080 and 2.3e-9 on the H100 (on both machines identical with and without MPS), the same order as the official CPU reference results bundled in `work/large_example/runtestL.result_*` (whose largest deviation is also on Pt63, the case that reaches 2.3e-8 in our 48-rank CPU reference column). The H100 no-MPS column is the "NVIDIA MPS" section above in numbers: 48 time-sliced CUDA contexts drag the suite to 3684.69 s — 2.5x the MPS-on time, slower than the CPU-only run, with per-case penalties up to 6.3x (DIA512-1) — while accuracy is unaffected. On the 18-rank RTX 5080 the no-MPS run is a milder 15% slower overall, but the pattern is the same, with the krylov inputs hit hardest (DIA512-1: 185.44 vs 116.49 s). On the larger inputs the dense band/cluster diagonalizations run on the GPU through GEMMul8; when many ranks share one GPU, some construction stages transiently fall back to the CPU where the device-memory preflight says they do not fit (by design — the run continues and stays correct; this happens on the 16 GB RTX 5080 and, at 48 ranks, even on the 80 GB H100). Keep in mind that these test inputs are correctness tests, not performance showcases: they are small-to-medium systems dominated by stages other than the dense diagonalization, which is where the GPU gains the most. The speedup grows with the system size (see "Important notes" below), and calculations with hundreds of atoms and a dense solver benefit far more than the 1.34x / 1.49x totals above.

The Kugui columns add a 40 GB GPU to the picture. With 64 ranks on one A100 the MPS daemon cannot be used (48-client limit, see above), and without it the 64 time-sliced contexts leave about 13 GB of the 40 GB free, so the force and Hamiltonian stages fall back to the CPU on most inputs and the GPU run ends up 10% slower than the CPU run (2670.86 vs 2436.27 s). With 48 ranks and MPS the same binary completes the suite in 1719.87 s — 1.62x the 48-rank CPU run (2785.91 s) and 1.42x the 64-rank one — with the largest gains on the band inputs (GGFF 2.93x, B2C62_Band 2.00x) and none on N1C999 (1.06x), as on the H100. All 16 inputs pass in all four Kugui columns; the two GPU columns stay within max diff Utot = 2.3e-9 Hartree, and the 48-rank CPU column reaches 2.3e-8 on Pt63, exactly as the Pegasus CPU column does.

### Large test suites on two Kugui nodes (-runtestL3 / -runtestL2)
The two largest built-in suites were run on two Kugui GPU nodes (each AMD EPYC 7763, 64 cores, 4 × A100-SXM4-40GB, 251 GiB; F2acc queue) with the same binary as above and two layouts of 64 MPI ranks:

- **CPU** — 32 ranks × 2 OpenMP threads per node, `OPENMX_GPU=0`, `-nt 2`;
- **GPU** — 32 flat-MPI ranks per node sharing one of the node's A100s, with an MPS daemon on each node.

Both layouts give every rank two cores (`mpirun --map-by ppr:32:node:PE=2`), and every input ran as a separate MPI job through `tools/run_gpu_suites.py`; the times are those of the native result files.

`-runtestL3` (20 inputs, 4–1280 atoms; every input is capped at 3 SCF iterations; "ratio" is CPU / GPU):

| input | atoms | solver | CPU (s) | GPU, MPS (s) | ratio |
|---|---:|---|---:|---:|---:|
| 5_5_13COb2 | 155 | band | 15.68 | 16.34 | 0.96 |
| C1000 | 1000 | cluster | 153.21 | 95.53 | 1.60 |
| C60 | 60 | dc | 11.15 | 10.75 | 1.04 |
| CG15c | 650 | dc-lno | 70.79 | 42.49 | 1.67 |
| Crys-MnO | 4 | band (sp) | 13.91 | 11.21 | 1.24 |
| DIA512-1 | 512 | krylov | 30.21 | 21.86 | 1.38 |
| Fe1000 | 1000 | cluster (sp) | 267.59 | 100.74 | 2.66 |
| GEL | 40 | band | 19.13 | 14.48 | 1.32 |
| GFRAG | 54 | cluster | 12.80 | 12.31 | 1.04 |
| GGFF | 40 | band (NC) | 139.41 | 54.66 | 2.55 |
| MCCN | 564 | dc-lno | 65.83 | 42.13 | 1.56 |
| Mn12_148_F | 148 | cluster (sp) | 19.41 | 14.81 | 1.31 |
| N1C999 | 1000 | dc-lno (sp) | 487.56 | 285.44 | 1.71 |
| Ni63-O64 | 127 | band (sp) | 18.63 | 15.87 | 1.17 |
| Pt500 | 500 | cluster | 360.56 | 300.98 | 1.20 |
| Pt63 | 63 | cluster | 14.72 | 13.64 | 1.08 |
| Si1280-LNO | 1280 | dc-lno | 605.02 | 641.41 | 0.94 |
| SialicAcid | 40 | cluster | 13.14 | 12.88 | 1.02 |
| ZrB2_2x2 | 76 | band | 45.59 | 24.12 | 1.89 |
| nsV4Bz5 | 64 | cluster | 18.34 | 15.83 | 1.16 |
| **Total** | | | **2382.68** | **1747.48** | **1.36** |

`-runtestL2` (7 clusters of 500–1200 atoms converged to 1e-10 Hartree; "SCF" counts the iterations in the `.out` history, the last number being the bundled reference):

| input | atoms | solver | CPU (s) | GPU, MPS (s) | ratio | SCF (CPU / GPU / ref) |
|---|---:|---|---:|---:|---:|---:|
| C1000 | 1000 | cluster | 1088.21 | 621.75 | 1.75 | 41 / 41 / 41 |
| Fe1000 | 1000 | cluster (sp) | 26053.42 | 6488.62 | 4.02 | 507 / 409 / 366 |
| GRA1024 | 1024 | cluster | 1429.67 | 624.62 | 2.29 | 57 / 48 / 53 |
| Ih-Ice1200 | 1200 | cluster | 520.04 | 319.85 | 1.63 | 36 / 35 / 36 |
| Pt500 | 500 | cluster | 8676.55 | 4118.84 | 2.11 | 300 / 215 / 199 |
| R-TiO2-1050 | 1050 | cluster | 1564.01 | 729.24 | 2.14 | 36 / 39 / 41 |
| Si1000 | 1000 | cluster | 1156.51 | 648.14 | 1.78 | 42 / 41 / 42 |
| **Total** | | | **40488.41** | **13551.06** | **2.99** | |

On `-runtestL3` the GPU layout is 1.36x faster overall, with the largest gains on the big cluster and band inputs (Fe1000 2.66x, GGFF 2.55x) and no gain on Si1280-LNO (0.94x) or the small 5_5_13COb2 (0.96x). On `-runtestL2` it is 2.99x faster, but part of that gap is the SCF path rather than the hardware: the CPU run of Pt500 reached its `scf.maxIter` of 300 without meeting the criterion (it still agrees with the reference to 1.6e-8 Hartree), and Fe1000 needed 507 iterations on the CPU against 409 on the GPU and 366 in the reference. The final convergence of these metallic clusters is erratic — the energy change hovers between 1e-8 and 1e-7 Hartree for a hundred iterations or more before it drops below 1e-10 — so the time per iteration is the fairer comparison: Fe1000 51.4 s on the CPU vs 15.9 s on the GPU (3.2x), Pt500 28.9 vs 19.2 s (1.5x); the other five inputs take 35–57 iterations in both layouts and run 1.6–2.3x faster on the GPU. Two GPU nodes thus complete `-runtestL2` in about the time of the bundled Kugui reference `runtestL2.result_kugui` (13566.86 s; Intel oneAPI + Intel MPI, 6 nodes, 192 MPI processes).

All seven `-runtestL2` inputs agree with the references in both layouts (max diff Utot = 1.6e-8 Hartree, max diff Force = 1.5e-8 Hartree/bohr). On `-runtestL3` the force differences of Fe1000 and Pt500 in both layouts and of C1000 on the GPU (up to 8.5e-7 Hartree/bohr on the GPU and 4.8e-7 on the CPU, both for Fe1000) exceed the 1e-7 default tolerance of `tools/compare_gpu_suites.py`; with only 3 SCF iterations their forces are far from converged, and the bundled results of other machines differ from the reference by the same amount (Fe1000: 4.9e-7 Hartree/bohr). Host memory was not a constraint: the largest run (L3 Pt500 on the GPU) peaked at 166 GiB on its first node.

## Important notes
At present, GPU-accelerated OpenMX performs faster than standard OpenMX for calculations involving systems containing hundreds of atoms. For calculations involving systems with fewer than a hundred atoms, standard OpenMX should be used (or set `scf.eigen.lib elpa2` to run the CPU paths of this code). Please use with caution as it may contain bugs.

## About bug reports
I would appreciate it if you could actively report any bugs. Please report them via GitHub issues or send them to [my X account](https://x.com/dc1394). Bug reports sent to my X account can be in English.

## License

This project is a derivative work of **OpenMX** and is licensed under the
**GNU General Public License v3.0 or later (GPL-3.0-or-later)**, consistent
with the upstream OpenMX license. See the [LICENSE](LICENSE) file for the
full license text.

```
SPDX-License-Identifier: GPL-3.0-or-later
```

### Upstream Project

This project is based on:

- **OpenMX** (Open source package for Material eXplorer)
- **Version**: 4.0
- **Upstream**: <https://www.openmx-square.org/>
- **Copyright**: Copyright (c) Taisuke Ozaki and OpenMX contributors
- **License**: GNU General Public License v3.0 or later

The original OpenMX source code retains its original copyright notices and
license headers. Modifications made in this project (GPU acceleration for
NVIDIA CUDA and AMD HIP backends) are also licensed under GPL-3.0-or-later.

### Third-Party Components

This project incorporates source code from the following third-party projects.
Each component retains its original license; their original license files are
preserved under `third_party/<component>/`.

#### FFTW3

- **Source**: <https://www.fftw.org/>
- **Copyright**:
  - Copyright (c) 2003, 2007-14 Matteo Frigo
  - Copyright (c) 2003, 2007-14 Massachusetts Institute of Technology
- **License**: GNU General Public License v2.0 or later (`GPL-2.0-or-later`)
- **Used here under**: GPL v3 (selected via the "or any later version" clause)
- **Original license file**: [`third_party/fftw3/COPYING`](third_party/fftw3/COPYING)

#### GEMMul8

- **Source**: <https://github.com/RIKEN-RCCS/GEMMul8>
- **Copyright**: Copyright (c) 2025- RIKEN R-CCS
- **Responsible developer**: Yuki Uchino (RIKEN Center for Computational Science)
- **License**: MIT License (`MIT`)
- **Original license file**: [`third_party/gemmul8/LICENSE`](third_party/gemmul8/LICENSE)

GEMMul8 (GEMMulate) is a library for emulating high-precision matrix
multiplication (SGEMM, DGEMM, CGEMM, ZGEMM) using INT8/FP8 matrix engines
based on the Ozaki Scheme II. 

If you use this project in academic work, please also cite the GEMMul8
upstream references (see the GEMMul8 repository).

#### ELPA / COSMA (the "gpusolver2" stack, since v2.0)

The distributed multi-GPU cluster diagonalization (`scf.eigen.lib gpusolver2`)
links four additional libraries, bundled as source archives under
`source/third_party/dist/` and built automatically by the first `make`; each
archive carries its original license file:

- **ELPA** 2026.02 — <https://elpa.mpcdf.mpg.de/> — Copyright (c) the ELPA
  consortium — **GNU Lesser General Public License v3.0 (`LGPL-3.0`)**
  (`COPYING/` inside the archive). The conventional "elpa1"/"elpa2" CPU paths
  use the ELPA 2018.05 source embedded in upstream OpenMX under the same
  license.
- **COSMA** 2.8.4 — <https://github.com/eth-cscs/COSMA> — Copyright (c) ETH
  Zürich (parts: Advanced Micro Devices, Inc.) — **BSD 3-Clause License
  (`BSD-3-Clause`)**
- **COSTA** and **Tiled-MM** (COSMA dependencies) —
  <https://github.com/eth-cscs/COSTA>,
  <https://github.com/eth-cscs/Tiled-MM> — Copyright (c) ETH Zürich —
  **BSD 3-Clause License (`BSD-3-Clause`)**

### License Compatibility Summary

| Component | Original License | Compatible with GPL v3 |
|-----------|------------------|------------------------|
| OpenMX (upstream) | GPL-3.0-or-later | — (same license) |
| GPU modifications (this project) | GPL-3.0-or-later | — (same license) |
| FFTW3 | GPL-2.0-or-later | Yes (via "or later" clause) |
| GEMMul8 | MIT | Yes (permissive → strong copyleft) |
| ELPA (2018.05 embedded; 2026.02 for "gpusolver2") | LGPL-3.0 | Yes (LGPL v3 code may be conveyed under GPL v3) |
| COSMA (with COSTA, Tiled-MM) | BSD-3-Clause | Yes (permissive → strong copyleft) |

The combined work is distributed under the terms of GPL v3 or later.

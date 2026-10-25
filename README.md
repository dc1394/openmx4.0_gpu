# openmx4.0_gpu

This repository includes NVIDIA CUDA, AMD HIP and CPU-only builds. Select one
per build with CMake; the existing NVIDIA `make -C source` workflow also works.
Single-configuration CMake builds default to `Release`, including the C++ core.
Use `-DCMAKE_BUILD_TYPE=Debug` explicitly for debugging; an existing empty build
type becomes `Release` when reconfigured. For performance validation against the
Intel CPU build and earlier GPU versions, see [the regression procedure](docs/PERFORMANCE_REGRESSION.md).

```sh
# RTX 5080 example; use sm_80 for A100 or sm_90 for H100.
cmake -S . -B build/cuda -DOPENMX_GPU_BACKEND=CUDA -DOPENMX_GPU_ARCH=sm_120
cmake --build build/cuda

# AMD example; use gfx1200 for Radeon RX 9060 XT, gfx942 for MI300A.
cmake -S . -B build/hip -DOPENMX_GPU_BACKEND=HIP -DOPENMX_GPU_ARCH=gfx1200 \
  -DOPENMX_HIP_ROCM_ROOT=/opt/rocm \
  -DOPENMX_HIP_MPI=OPENMPI -DOPENMX_HIP_MPI_ROOT=/path/to/openmpi \
  -DOPENMX_HIP_LINALG=AOCL -DOPENMX_HIP_AOCL_ROOT=/path/to/aocl/aocc \
  -DOPENMX_HIP_AOCC_LIB=/path/to/aocc/lib
cmake --build build/hip

# Full CPU application: MPI, OpenMP, FFTW3 and LP64 ScaLAPACK/BLAS/LAPACK.
cmake -S . -B build/cpu -DOPENMX_GPU_BACKEND=CPU \
  -DOPENMX_CPU_COMPILER=GCC -DOPENMX_CPU_MPI=OPENMPI -DOPENMX_CPU_LINALG=OPENBLAS
cmake --build build/cpu --parallel 4
```

The executables are `build/cuda/openmx`, `build/hip/openmx` and `build/cpu/openmx`. Each successful
build also updates `work/openmx`; `OPENMX_WORK_DIR` changes this directory and
`OPENMX_COPY_TO_WORK=OFF` disables the copy. Objects and
dependency builds are isolated in each build directory. See
[GPU backend build instructions](docs/GPU_BACKENDS.md) for requirements,
configuration, AMD runtime settings and validation limits.

HIP builds accept `OPENMX_HIP_MPI=OPENMPI|INTEL` and
`OPENMX_HIP_LINALG=AOCL|MKL|OPENBLAS` (defaults: OpenMPI + AOCL).
The selected MPI and LP64 libraries are shared by OpenMX, ELPA, COSMA and
MAGMA. Use a separate build directory for each combination. For example:

```sh
cmake -S . -B build/hip-intel-mkl -DOPENMX_GPU_BACKEND=HIP -DOPENMX_GPU_ARCH=gfx1200 \
  -DOPENMX_HIP_ROCM_ROOT=/opt/rocm \
  -DOPENMX_HIP_MPI=INTEL -DOPENMX_HIP_MPI_ROOT="$I_MPI_ROOT" \
  -DOPENMX_HIP_LINALG=MKL -DOPENMX_HIP_MKL_ROOT="$MKLROOT"
cmake --build build/hip-intel-mkl
```

`OPENMX_HIP_OPENBLAS_ROOT` selects a non-system OpenBLAS installation.
OpenBLAS and AOCL use a compatible ScaLAPACK installation or build the pinned
Netlib release for the selected MPI; an OpenMPI ScaLAPACK cannot be reused with
Intel MPI. FFTW3 is selected separately with `OPENMX_HIP_FFTW_ROOT` (AOCL's
FFTW is preferred in AOCL builds). See [HIP options](docs/GPU_BACKENDS.md#amd--hip).

The CPU build uses ordinary C/C++ and Fortran compilers with a matching MPI,
requires no GPU SDK, and defaults to `scf.eigen.lib elpa2`. It builds the same
bundled ELPA 2026.02.002 release with GPU kernels disabled, and uses the host
Kokkos SIMD kernels. Existing `gpusolver`/`gpusolver2`/`gpusolver3` input decks
select CPU ELPA2 in this executable. CPU builds also copy the matching Core and
host SIMD libraries to `work/lib/`. See [CPU dependencies and options](docs/GPU_BACKENDS.md#cpu-専用の本体ビルド)
for external numerical libraries and MPI compiler selection.

CPU configuration accepts `OPENMX_CPU_COMPILER=GCC|CLANG|ICX`,
`OPENMX_CPU_LINALG=MKL|OPENBLAS`, and `OPENMX_CPU_MPI=INTEL|OPENMPI`.
Use a separate build directory for each combination. `OPENMX_CPU_MPI_ROOT`
selects the MPI installation; `OPENMX_CPU_MKL_ROOT` and
`OPENMX_CPU_OPENBLAS_ROOT` select numerical-library prefixes. For example:

```sh
cmake -S . -B build/cpu-icx-mkl -DOPENMX_GPU_BACKEND=CPU \
  -DOPENMX_CPU_COMPILER=ICX -DOPENMX_CPU_MPI=INTEL -DOPENMX_CPU_LINALG=MKL \
  -DOPENMX_CPU_MPI_ROOT="$I_MPI_ROOT" -DOPENMX_CPU_MKL_ROOT="$MKLROOT"
cmake --build build/cpu-icx-mkl --parallel 4
cmake --install build/cpu-icx-mkl
```

GNU Fortran is the default Fortran compiler for all three C/C++ families;
`-DCMAKE_Fortran_COMPILER=ifx` selects Intel Fortran explicitly. MKL selects
the matching Fortran interface and MPI BLACS, using sequential MKL alongside
OpenMX's MPI/OpenMP. OpenBLAS uses a matching Netlib ScaLAPACK installation
or builds the pinned release for the selected MPI. Clang also needs its
OpenMP development package (`libomp-dev` on Debian/Ubuntu).

To install after building, pass the same build directory used with `cmake -B`:

```sh
# Default: work/openmx and both shared libraries in work/lib/
cmake --install build/cuda
# For the HIP build: cmake --install build/hip
# For the CPU build: cmake --install build/cpu

# Optional prefix: $HOME/.local/bin/openmx and $HOME/.local/lib/libopenmx_core.so
cmake --install build/cuda --prefix "$HOME/.local"
```

With no prefix selection, installation puts `openmx` directly in `work/`;
`OPENMX_WORK_DIR` changes this local directory. A prefix set with
`-DCMAKE_INSTALL_PREFIX=...` during configuration or with `--prefix` during
installation uses the `bin/openmx` and `lib/libopenmx_core.so` layout.
Here `build` is only the parent directory; `cmake --install build` is valid
if the project was configured with `-B build`. Installation does not build
the executable, so finish `cmake --build` first.

To switch an existing build from a cached prefix (such as the previous
`/usr/local` default) to the local default, reconfigure once before installing:

```sh
cmake -S . -B build/cuda -UCMAKE_INSTALL_PREFIX
cmake --build build/cuda
cmake --install build/cuda
```

An explicit `OPENMX_GPU_ARCH` accepts `80` or `sm_80` and also selects the
Kokkos/CUDA target when `OPENMX_CORE_BACKEND=CUDA`. It needs no GPU on the build
node. When `OPENMX_GPU_ARCH` is unset or `auto`, CMake queries `nvidia-smi`
for the first GPU with a valid compute capability. Configuration fails if the
command is missing, fails, or reports no usable compute capability; specify
`-DOPENMX_GPU_ARCH=sm_80` (or the target GPU's architecture) on a login node.
A cached `CMAKE_CUDA_ARCHITECTURES` value does not bypass this check. Host-only
core builds and CPU application builds do not query a GPU.

The `refactor/cpp-kokkos-core` branch begins the migration of the OpenMX
application to C++, explicit ownership and Kokkos. The new `core/` library
contains radial interpolation and Fourier transforms for PAOs, PAO products, nonlocal
projectors and neutral-atom potentials, spherical Bessel functions and FFT-grid
validation. PAO grids, atomic densities, orbitals, VPS/nonlocal-projector data,
optional VNA expansion tables and their Fourier tables now have
calculation-scoped C++ ownership, with borrowed views for legacy consumers.
Ragged PAO-product tables use offsets into one contiguous allocation and
mdspan block views; their Gauss quadrature uses the common Kokkos kernels.
VNA separable-projector construction and orthogonalization use an independent
C++ module with RAII workspaces; their Fourier transforms also use the common
Kokkos kernels. The legacy PhiF interpolation shares the checked radial code.
Radial transforms borrow validated input views without copying the input tables;
their sampled data and workspaces use RAII. Periodic-cell inverse indexing and
the uniform reciprocal grid no longer use global arrays; PAO/projector
reciprocal interpolation shares the same C++ implementation across backends.
The VNA wavevector grid is local to each matrix-construction call.
Pair Bessel tables for overlap/kinetic integrals and the CPU nonlocal path
now use invocation-owned C++ workspaces. Cache hits borrow the existing
contiguous table through mdspan without copying it or allocating per pair.
Overlap/kinetic integral scratch matrices also use a per-thread C++ owner;
packed orbital indices replace their six-level pointer tables, and angular
assembly reads the radial results directly without an intermediate copy.
Nonlocal integrals use the same CPU evaluator in both backends. Their
workspaces own projector metadata and pack magnetic components into contiguous
arrays, replacing the temporary five-level pointer tables.
VNA2/VNA3 also share one CPU evaluator with per-thread RAII workspaces;
packed orbital matrices and reusable Bessel tables replace six-level pointer
tables and per-pair allocation. Their GPU batch state belongs to each call.
VNA preparation owns a snapshot of species basis metadata, projector indices
and contiguous weighted radial rows. Both backends share the CPU sampling
loop, and DS_VNA GPU batches also receive their state explicitly per call.
DS_VNA CPU assembly now shares one evaluator across CUDA/HIP. Per-thread C++
workspaces own packed angular matrices, radial sums and Bessel scratch;
species-local offsets replace temporary five-level pointers and omit empty
basis groups. The zero-wavevector Bessel derivative initializes the extra order
it reads, and per-atom timing updates are safe across threads.
HVNA archive offsets and projector energies now share one C++ owner across
CUDA/HIP. Resident archives, rank turns and capacity recovery receive explicit
invocation state; their backend memory budgets and contraction order are preserved.
CPU HVNA contraction also shares one evaluator for local and received atoms.
A C++ snapshot owns projector groups, energies and basis sizes; per-worker packed
matrices replace two-level scratch pointers. Geometry is passed explicitly, and
HIP returns its per-call stage diagnostic directly to the DFT caller.
All three VNA stages now share one assembly pipeline and an invocation owner
for basis metadata, ordered atom pairs, both geometric orientations and wavevectors.
CPU integrals receive explicit geometry and Fourier-table views. Projector
communication in VNA, nonlocal integrals, forces, stress and core-hole matrices
uses per-call C++ owners for progress, message sizes and reusable send/receive
buffers, replacing shared progress globals and repeated message allocation.
The final VNA matrices now also have calculation-scoped ownership: four
primitive and three optional contracted matrix families use contiguous ragged
blocks with mdspan views. Their seven global arrays are removed. DFT,
Hamiltonian assembly, total energy, forces and stress receive the matrix state
explicitly; CPU VNA assembly uses flat views, while native kernels borrow
stable row adapters. Repartitioning and teardown release these owners without
walking the old atom or neighbor arrays.
PAO/nonlocal Fourier tables, neutral projectors and energies, and orbital-product
Fourier tables now also travel through an explicit calculation view; their
seven global pointers and the null-context manual allocation/free paths are
removed. Host interpolation and VNA consumers use flat views. One C++ row plan
shares restart I/O and MPI distribution across all five Fourier transforms,
preserving their binary formats and row order. Files close through RAII, and
MPI ranks agree on restart failure before recomputing. The external C API owns
the same tables with host radial execution, allowing consecutive calls without
initializing or finalizing Kokkos for each input.
Species names, basis counts, meshes, charges and other basic species fields now
share a calculation-owned C++ store. Its contiguous arrays and label buffers
replace manual allocation of the corresponding input tables. Real-space atomic
densities, neutral/Hartree potentials, partial-core densities, VPS coordinates
and nonlocal projector rows are passed explicitly to input normalization,
Fourier transforms, initial-density grids, energy, force and stress consumers;
eight more global pointers are removed. The Hartree interpolator and its radial
derivative take both their row and asymptotic charge from the caller.
Orbital values, derivatives and their grids now receive explicit atomic views
through CPU evaluation, native GPU table/tile packing, nonlocal assembly,
optical/XANES calculations, transport analysis and orbital exports. The remaining
PAO-coordinate/orbital and nonlocal angular/energy global pointers are removed.
Evaluation scratch uses worker-local C++ owners with flat mdspan storage,
transactional allocation and reusable derivative buffers. No owner lookup or
allocation is added inside the reusable evaluator's grid-point loops.
The main SCF matrices now also belong to the calculation: 32 kinetic,
nonlocal, overlap, Hamiltonian, density and mixing-history families replace
their global arrays with contiguous C++ storage. Component/atom/neighbor
offsets describe their unequal block sizes and expose mdspan views. SCF,
solvers, energy, forces, restart and output pass these matrices explicitly;
legacy native kernels borrow row adapters into the same storage. Replacement
is transactional, and release does not depend on old atom or neighbor lists.
CUDA/HIP share this ownership layer while retaining their current kernel and
device-specific execution policies. Removing the remaining borrowed pointer
tables from numerical consumers is a later migration step.
Onsite density and Hubbard/Zeeman potentials, DC inverse overlaps, LNO
coefficients/populations and contracted-orbital coefficients now share another
calculation-owned store. Nine families use contiguous values with explicit
atom offsets or rectangular strides; ten globals, including one unused Zeeman
allocation, are removed. Their numerical consumers use flat row views through
SCF, forces, stress, orbital optimization and output. These nine families now
expose only flat views; their temporary row adapters are removed. Allocation
failure preserves the previous state, and repartitioning releases storage
independently of old topology. The same implementation serves CUDA/HIP and has
no vendor SDK types. Noncollinear Hamiltonian contraction also handles all four
spin components without overrunning its accumulation buffer.
Orbital optimization now runs in the common C++ core. Each DFT invocation owns
its histories, Hessian, copied topology, halo packet plans and reusable scratch.
Atoms/species optimization, DIIS/EF updates, normalization and gradients use flat
matrix/tensor views; MPI and native LAPACK are explicit service callbacks. Seven
more globals are removed, and the optimization norm travels as an argument.
CUDA/HIP also share the full-projector Hubbard force derivatives, with one
contiguous C++ scratch owner replacing their temporary pointer trees. Regression
tests compare the numerical paths with frozen implementations and check s/p
gradients with finite differences, including tied magnetic channels and radial
constraints. Exactly zero gradients and the species DIIS small-gradient limit
are handled explicitly.
The collinear/noncollinear Cluster and Band solvers, DC, DC-LNO and both
Krylov dispatches now share a C++ invocation owner for ordinary host scratch.
Matrices, integer/MPI work buffers, eigenvalue/PDOS rows and pole residues use
RAII storage; ragged residues replace six-level numerical pointer trees with
contiguous atom blocks and explicit offsets. Noncollinear Band spinor scratch
is local to each call, removing eight shared arrays. Native BLAS/eigensolver
interfaces borrow owned row adapters. Persistent Band/solver caches, device
buffers, pinned memory, streams and BLAS/solver handles in these seven CUDA/HIP
modules now have calculation-scoped C++ owners. Native cache metadata and SDK
aliases remain borrowed during this transition; existing cache budgets,
eviction and device-specific execution policies are preserved. DC-LNO histories
and the CUDA noncollinear Cluster GEMM handle reset with each calculation.
Deeper Krylov helpers use their own RAII scratch scopes and contiguous matrices;
FP32 eigenvalue refinement shares its caller's owner for both allocation and
release. The common ownership module contains no vendor SDK types. Remaining
native globals, borrowed row interfaces and kernels still require migration.
`Set_Hamiltonian` and `Set_Density_Grid` now share their host quadrature in
the C++ core. Explicit calculation workspaces own the contiguous density,
atom/pair plans, per-worker matrix/orbital scratch and A-to-B MPI work storage;
SCF iterations reuse these allocations. Density workers process whole atoms,
and Hamiltonian workers retain the small spin-unpolarized blocked integral.
The CPU quadratures use Kokkos experimental SIMD with explicit host ABIs:
AVX2 (4 doubles), AVX-512 (8 doubles), or AArch64 NEON (2 doubles), with a
scalar fallback. Hamiltonian matrix accumulators stay in registers over a
16-point tile; neither kernel changes the order of its floating-point sums.
Host ISA selection checks CPU/OS support. `OPENMX_SETHAM_SIMD`,
`OPENMX_DENSITY_GRID_COL_SIMD` and `OPENMX_DENSITY_GRID_NC_SIMD` accept
`scalar` or `avx2` to cap the default automatic selection; append `_TRACE=1`
to the corresponding variable name to report the selected implementation.
CUDA/HIP device kernels, cache budgets and GeForce versus A100/H100 dispatch
policies are retained. The new host kernels require neither a GPU SDK nor a
second OpenMP team/runtime. They also serve the CPU path when GPU memory
admission fails. Kokkos 5.2.2's default `simd<double>` is not used to select
the host ABI: CUDA/HIP/SYCL defaults would select width 1.

`libopenmx_host_simd` is built by an ordinary C++20 compiler against a private
SERIAL configuration of the same pinned Kokkos source. Its Kokkos/C++ symbols
are hidden, and only two functions exchanging plain descriptors cross the
library boundary. Device compilers never compile the host SIMD kernels.
CMake builds and installs this library alongside `libopenmx_core`; keep both
libraries when copying an installation. If the Core C++ compiler is a device
wrapper, specify `-DOPENMX_HOST_SIMD_CXX_COMPILER=/path/to/g++` (or `clang++`).
`-DOPENMX_CORE_HOST_SIMD=OFF` builds only the scalar implementation.
The optional `openmx_grid_benchmark` target, enabled with
`-DOPENMX_CORE_BUILD_BENCHMARKS=ON`, compares the CPU quadratures including
their packing and worker overhead; see [the performance checks](docs/PERFORMANCE_REGRESSION.md).
`Set_OLP_Kin`, `Set_Nonlocal` and the VNA2/3 CPU fallback in
`Set_ProExpn_VNA` share a C++ module for real-harmonic transformations,
Cartesian derivatives and overlap angular momentum. Per-worker RAII tensors
own its scratch and results; the numerical loop allocates no memory. Nonlocal
assembly reuses contiguous complex scratch across halo and local products,
and overlap/nonlocal pair lists use owned topology plans. Native output order,
onsite derivatives and overlap stabilization calls are preserved. The module
borrows explicit conversion and geometry views and contains no vendor SDK
types; the optimized CUDA/HIP device kernels retain their existing policies.
Force now owns its halo density matrices and per-atom accumulators in a C++
invocation. CPU derivative, grid, nonlocal and core-hole scratch uses reusable
per-worker tensors; stage scopes release large grid buffers before later GPU
archives are prepared. CUDA/HIP share the overlap and spin-density traces,
with explicit strides and independently tested spinor signs. The native GPU
batch kernels and device-specific memory/precision policies are retained.
Host tensors own contiguous storage and expose Kokkos/mdspan views, or
`std::mdspan` when the compiler and standard library support it. This is
an ongoing migration; most application globals and legacy GPU kernels remain.
See the [C++ core migration guide](docs/CPP_CORE_MIGRATION.md) for ownership,
CUDA/HIP configuration, tests and the remaining work.

The standalone core also supports a Kokkos SYCL build with an Intel SYCL
compiler. Device capabilities are described without vendor SDK types; a SYCL
device lacking FP64 uses the CPU radial implementation. Intel GPU support in
the full application and a Metal implementation remain future work.
The [2026-10-20 review response](docs/CODE_REVIEW_20261020.md) records these
boundaries, GPU error-handling fixes and validation.

The core defaults to the Serial backend independently of the legacy GPU
backend. Set `OPENMX_CORE_BACKEND=CUDA` or `HIP` to run these radial transforms
on the selected accelerator. Its unit tests can also run without MPI or a GPU:

```sh
cmake -S . -B build/core-host -DOPENMX_BUILD_APPLICATION=OFF
cmake --build build/core-host --parallel 4
ctest --test-dir build/core-host --output-on-failure
```

Both backends use the same GPU worker, memory-view, density-matrix, dense
transform, multi-GPU GEMM and MAGMA transaction code in `source/`. The
`openmx_gpu_*.h` interfaces select native CUDA or HIP APIs at compile time.
Backend-specific solver policies and OpenACC/OpenMP target kernels retain
their respective implementations; see [shared code](docs/GPU_BACKENDS.md#共通コードとバックエンド境界).

The NVIDIA build adds MAGMA, which the multi-GPU Cluster and Band solvers
need, when `nvidia-smi` on the build node lists two or more GPUs. To build
it on a single-GPU node or on a login node without GPUs for multi-GPU
compute nodes, add `WITH_MAGMA=1` (with CMake,
`-DOPENMX_MAKE_OPTIONS=WITH_MAGMA=1`). The HIP build always includes MAGMA.
See [Dependencies](#dependencies).

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

A run that finds no usable GPU demotes itself to ELPA2 automatically, so the default is also safe on machines without a GPU supported by the selected backend. The environment variable `OPENMX_GPU=0` forces the same demotion on a machine with a GPU, which is convenient for CPU-vs-GPU comparisons with unmodified input files.

The GPU paths follow OpenMX's `level.of.stdout` keyword. At the default (1) they print only errors, a GPU that cannot be used, and a GPU eigensolver given up for lack of device memory; `level.of.stdout 2` adds the device plans, the precision switches and the CPU fallbacks of single stages, and 3 adds the per-step, per-k-point and per-rank details of the refined FP32 eigensolver. Note that OpenMX itself prints per-atom debugging output from level 2 on.

## New in v2.0: distributed multi-GPU cluster diagonalization (ELPA GPU + COSMA)
The distributed ELPA/COSMA backend supports cluster diagonalization across MPI ranks and nodes. For single-node memory recovery and forced MAGMA, see the following section. Selecting

```ini
scf.EigenvalueSolver       cluster
scf.eigen.lib              gpusolver2    # ELPA (GPU kernels) + COSMA
```

replaces the cluster diagonalization (collinear and non-collinear) with [ELPA](https://elpa.mpcdf.mpg.de/) 2026.02 (NVIDIA or AMD GPU kernels, following the build's backend) and [COSMA](https://github.com/eth-cscs/COSMA) (communication-optimal distributed matrix multiplication, GPU backend), engaging every MPI rank and every GPU. This is the option for machines where the diagonalization must not be confined to one GPU, e.g. one GPU per node × many nodes. Notes:

- "gpusolver2" supports only `scf.EigenvalueSolver cluster`; any other solver stops with an input error. It retains the existing per-rank GPU paths outside cluster diagonalization, and a machine without a usable GPU demotes itself to ELPA2 as usual.
- When the GPU memory cannot hold a solve, that solve automatically falls back to the ELPA CPU kernels / ScaLAPACK instead of aborting the run.
- The required libraries (ELPA, COSMA, COSTA, Tiled-MM) are bundled as source archives under `source/third_party/dist/` and are built automatically from those archives by the first `make` (the build tools, and the one download that `Makefile.kugui` adds, are listed under [Dependencies](#dependencies)). The CPU "elpa1"/"elpa2" paths and "gpusolver2" share the same ELPA 2026.02.002 library. The CPU keywords select its one-stage and two-stage solvers with GPU execution disabled; the old embedded ELPA has been removed.

## Single-node multi-GPU Cluster and Band

The default `scf.eigen.lib gpusolver` uses cuSOLVER while the dense GPU
buffers fit. If they do not fit, it automatically uses multi-GPU MAGMA
for ordinary collinear and noncollinear Cluster and Band on one node.
`gpusolver3` forces that MAGMA path from the beginning; `gpusolver2`
remains the distributed ELPA/COSMA solver. These MAGMA paths need a build
that includes MAGMA; see [Dependencies](#dependencies).

```ini
scf.eigen.lib              gpusolver
scf.Gpu.Num                4
```

`scf.Gpu.Num` is a strict upper bound. CUDA visibility and
`OPENMX_MGPU_NUM_GPUS` can reduce it further. If the permitted GPU group
cannot hold MAGMA's per-device scratch, or MAGMA/cooperation is unavailable,
the memory fallback of a single-node run reports the reason and uses CPU
eigensolvers and matrix products. It never enlarges the GPU pool.
`scf.mgpu.enable off` and `OPENMX_MGPU=0` explicitly disable cooperation.

In a **multi-node** Cluster run (collinear or noncollinear) the default
`gpusolver` continues the same ladder across nodes. When the dense solve does
not fit on one GPU, it first tries MAGMA on the GPUs of rank 0's node (two or
more GPUs, order up to 32766, enough host memory there). If that is not
possible, it switches to the distributed gpusolver2 path (ELPA GPU kernels +
COSMA) on all ranks of the run; a spin-polarized run solves its two spins in
turn there. ELPA and COSMA use the GPU of each rank, so run at least one rank
per GPU. In a two-node job with one GPU per node the solve then moves to two
GPUs. With two GPUs per node, MAGMA on two GPUs comes first, then four GPUs.
If even the distributed GPU solve does not fit, it uses CPU kernels.
Its transforms use COSMA only while the host memory COSMA keeps
leaves a quarter of each node free (`OPENMX_GS2_HOST_RESERVE_MB`), and
ScaLAPACK otherwise. `OPENMX_CLUSTER_GS2_FALLBACK=off|on|force`
disables this step, allows it on one node, or skips MAGMA. Details:
[Cluster details](doc/cluster_mgpu.txt), "Multi-node capacity fallback".

The MAGMA capacity path keeps dense matrices on the host, distributes the
eigensolve across the group, and streams bounded tiles for matrix products.
It allocates neither the full cuSOLVER workspace nor a full dense matrix
on the owner GPU. Sufficient host RAM is still required. Native FP64 is
the default. `OPENMX_MAGMA_FP32=1` with `scf.fp32.enable on` enables a
MAGMA FP32 basis followed by FP64 eigenvector correction and residual
checks for all four Cluster/Band spin paths. The correction products use
GEMMul8 when enabled. FP32 requires every CUDA-visible GPU to have a known
FP32:FP64 throughput ratio of at least 8. Mixed H100/GeForce allocations
retain native FP64, including when H100 is outside the current Band group;
the FP32 request cannot override this guard.
See [two RTX 5090 GPUs](doc/rtx5090_magma_fp32.txt)
for the controls, memory requirements and reproducible tests.

DC, DC-LNO and Krylov use `gpusolver` to solve local atomic clusters on the
GPU assigned to each MPI rank.
Their collinear FP32/double-float grid kernels and GEMMul8 can run on both GPUs,
while their local cuSOLVER eigensolves remain FP64. See
[O(N) two-GPU validation](doc/rtx5090_ordern.txt) for the Si555/Si666 runner
and the distinction from the MAGMA FP32 path.

For Band, each of the **K actually computed k-points** receives
`floor(G/K)` GPUs from the permitted pool G. At least two GPUs per k-point
are required. Three k-points with four GPUs cannot cooperate; with eight
GPUs they use six, in groups `[0,1]`, `[2,3]`, `[4,5]`.
`OPENMX_MGPU_BAND_SHARED=1` instead assigns k-points cyclically to groups
of at least two GPUs and serializes access to each group. This allows a
Regular 2x2x2 mesh to use MAGMA with two GPUs, retaining every k-point.

Matrix products follow the eigensolver: a product whose operands are on a
GPU stays on that GPU, and several GPUs are used only on the capacity rungs
(the host-backed MAGMA products on the owner's node, then the distributed
gpusolver2 transforms over all nodes). Spreading such a product over the
GPUs for speed is experimental: `OPENMX_MGPU_GEMM=on` (or `auto`).

GPU work outside the generalized eigenproblem now uses capacity recovery
by default: the rank's GPU, permitted GPUs on its node, GPUs on other nodes,
then CPU. This covers orbital/grid kernels, Hamiltonian construction,
density matrices and density grids, XC, GPU force/energy terms and RMM-DIISH
history. Every tier respects the configured GPU cap; k-dependent dense work
also respects its exact GPU group. Allocation failures release private
buffers before retrying, so a failed attempt cannot partially add results.
`OPENMX_GPU_WORKERS=1` optionally distributes work for speed even when one
GPU fits; `0` disables borrowing and retains CPU recovery. Host memory is
still required for CPU work and communication buffers.
See [GPU worker ownership and controls](doc/gpu_workers.txt).
The new policy's numerical and allocation-failure checks are recorded in
[capacity recovery validation](doc/gpu_capacity_validation.txt).
Earlier performance-splitting measurements remain in
[GPU worker validation](doc/gpu_workers_validation.txt).

A binary built without MAGMA cannot take these paths. Where a solve would
need MAGMA it reports `MAGMA multi-GPU support is unavailable for the
selected devices` and continues with the remaining fallbacks above.
[Dependencies](#dependencies) describes when the build includes MAGMA.
The experimental paper implementation has been removed. See
[Cluster details](doc/cluster_mgpu.txt), [Band details](doc/band_mgpu.txt),
[matrix products](doc/mgpu_gemm.txt), and
[capacity-path validation](doc/mgpu_capacity_validation.txt).

## GEMMul8: FP64 matrix multiplication on integer tensor cores
The large dense matrix multiplications of the GPU eigensolver path use a device-aware bridge to [GEMMul8](https://github.com/RIKEN-RCCS/GEMMul8) or native cuBLAS FP64. GEMMul8 emulates FP64 GEMM on INT8 tensor cores using the Ozaki scheme II and remains the default on consumer GPUs (GeForce) with limited native FP64 throughput. General GEMMs on A100 and H100 instead default to native FP64, including the FP64 Tensor Core kernels selected by cuBLAS, avoiding integer decomposition and its workspace. The choice uses the device's FP32:FP64 throughput ratio: a known positive ratio below 8 selects native FP64. A100 has a separate size-dependent exception for the collinear cluster forward transform, described below.

Omitting `scf.gemmul8.enable` selects this device default. Explicit settings override it:

```ini
scf.gemmul8.enable         off           # force native cuBLAS FP64
# scf.gemmul8.enable       on            # enable GEMMul8, including on A100
```

`OPENMX_GEMMUL8_DISABLE=1` disables GEMMul8, while an explicit `=0` enables
it under the automatic input setting. The `_D` and `_Z` variants select real
and complex products separately. An input-file `off` always takes precedence;
an input-file `on` still honors environment disables.

The input keyword `scf.fp32.enable` defaults to `on`, preserving the existing
precision policies. To force FP64 arithmetic while retaining GEMMul8:

```text
scf.fp32.enable     off
scf.gemmul8.enable  on
```

`off` disables FP32 eigenvector refinement, FP32 and double-float grid
integrals, and double-float orbital evaluation. It also promotes the
legacy float products in VNA matrix, force and stress contractions to FP64.
It overrides `OPENMX_EIGEN_REFINE`, `OPENMX_GRID_FP32`,
`OPENMX_GRID_PRECISION` and `OPENMX_ORBS_EVAL_PRECISION` requests for FP32/DF.
GEMMul8 and cuSOLVER's FP64 fixed-point emulation remain governed by their
own settings. The existing float storage of orbital and VNA projector
tables is unchanged; this keyword controls arithmetic precision.

### A100 execution policy

On A100, on-the-fly orbital evaluation and grid quadrature use native FP64
arithmetic by default. GeForce retains the compensated double-float paths.
Native density quadrature now also uses a CUDA kernel that tiles overlap
points and stages the density matrix in shared memory. Its OpenACC fallback
remains available for unsupported sizes and resource limits. Orbital tile
storage retains its existing format; changing the arithmetic does not change
that storage contract.

On `sm_80`, native FP64 Hamiltonian quadrature uses 2x2 register tiles,
separate spin blocks and adaptive point chunks. `OPENMX_SETHAM_FP64_TILED=0`
restores the original FP64 kernel; `OPENMX_SETHAM_FP64_SPLIT=1..64` overrides
the chunk count for measurements. GeForce's double-float kernel is unchanged.

On full A100-class `sm_80` devices (at least 100 SMs), real square cluster
forward products at dimension 6144 or above automatically use full-workspace
GEMMul8 when memory permits. Smaller transforms keep native FP64; a full
workspace that cannot fit also selects native FP64. Explicit input, environment
and forward-stage selections take precedence. This exception is controlled by
`OPENMX_GEMMUL8_A100_FORWARD_MIN_N` (`0` disables it) and does not change
GeForce's existing selection.

`OPENMX_ORBS_EVAL_PRECISION=fp64|df` and `OPENMX_GRID_PRECISION=fp64|df`
select orbital and grid arithmetic explicitly. `OPENMX_SETDEN_FP64_CUDA=0`
selects the existing OpenACC implementation for native density quadrature.
Native CUDA density splits work by spin for orbital counts above 32 and more
than one spin; `OPENMX_SETDEN_FP64_SPLIT_SPIN=0|1` overrides that scheduling.
The lossy `OPENMX_GRID_FP32` experiment stays off unless requested.
FP32 eigenvector refinement already selects native FP64 on A100;
`OPENMX_EIGEN_REFINE` still provides an explicit override. On toolkits that
support cuSOLVER FP64 emulation, the default now follows the same hardware
boundary, avoiding emulation's full-matrix retry backup on A100.

Build instructions, control precedence, regression commands and the new
validation record are in [the A100 optimization report](doc/a100_optimization.txt).
The [2026-10-21 validation of the unified code](doc/a100_validation_20261021.txt)
includes the new quadrature, Kugui's optimized host FFTW build and CPU ELPA
recovery without a visible GPU.

### Forward transform of the collinear cluster solver
The two products of the orthogonalization transform, `B = H X` and `X^T B`, have entry points of their own (`openmx_gemmul8DgemmFixed`), because `X` stays fixed during the SCF.

- **Workspace.** When these products select GEMMul8, they take its full workspace whenever the device has room; every other GEMMul8 GEMM keeps the capped, blocked workspace (`OPENMX_GEMMUL8_MAX_WORKSPACE_MB`, 256 MiB). Under the cap each block rescales its operands. The workspace is returned right after the transform, so the eigensolver finds the memory it found before. For an explicit GEMMul8 selection or the GeForce default, a workspace that does not fit falls back to the capped path at the same precision, and `OPENMX_GEMMUL8_FORWARD_UNBLOCKED=0` restores that capped path. Under the A100 automatic policy, both conditions select native FP64 instead. As before, `OPENMX_GEMMUL8_FORWARD_REUSE=1` implies unblocked execution and takes precedence over `OPENMX_GEMMUL8_FORWARD_UNBLOCKED=0`. Below the cap the retained GEMMul8 results are bit-identical to the previous ones.
- **Precision stage**, for measurements: `OPENMX_GEMMUL8_FORWARD=fp64|gemmul8` selects plain cuBLAS FP64 or GEMMul8 with `OPENMX_GEMMUL8_FORWARD_NUM_MOD` moduli and `OPENMX_GEMMUL8_FORWARD_FASTMODE`; `OPENMX_GEMMUL8_FORWARD_REUSE=1` retains the prepared `X` across SCF steps (exact with fast scaling only); `OPENMX_GEMMUL8_FORWARD_TIMING=1` reports calls, preparations, reuses and wall time, and `OPENMX_CLUSTER_PROFILE=1` the phases of each dense solve.
- **Tools.** `tools/measure_gemm_probe.sh` (matrix-level times, phases and errors against a double-double reference), `tools/run_forward_variants.sh` (the same input under several settings, with a comparison table), and the checks `tests/run_gemmul8_forward_smoke.sh`. `OPENMX_GEMMUL8_TIMING=1` counts and times products through the general GEMM entry points and reports the totals per rank at the end of each SCF. Successful full-workspace fixed transforms, including large automatic A100 transforms, use the separate `OPENMX_GEMMUL8_FORWARD_TIMING=1` counters.

## Multi-GPU parallelization
With the default `gpusolver`, each dense eigenproblem normally uses one
GPU. If its buffers cannot fit, single-node Cluster and ordinary Band can
automatically use MAGMA (in a build that includes it) on the permitted GPU
group described above; a
multi-node Cluster run then tries MAGMA on rank 0's node and otherwise the
distributed gpusolver2 path on every node.
`gpusolver3` forces MAGMA; `gpusolver2` selects distributed ELPA/COSMA.
More GPUs do not guarantee a speedup when the problem already fits on one
GPU: MAGMA's transfers and CPU work can outweigh the parallel GPU work.

## MPI vs. hybrid (MPI/OpenMP) parallelization
Use flat MPI while every core you want to use can hold an MPI rank: the device phases (the eigensolvers, the grid integrals, the force batches) are shared by all the ranks of a node and gain nothing from threads. When device memory limits the number of ranks below the core count, give each rank several cores and as many OpenMP threads (`-nt`). Since 2026-10-13 the host-side phases of the GPU build run threaded: the Bessel-table cache of Set_OLP_Kin is kept per thread, the received-atom products of Set_Nonlocal and the CPU fallback of the cluster density matrix are threaded over the atoms, and the OpenMP regions of the original code (truncation, Set_ProExpn_VNA, Set_Orbitals_Grid, Force, Total_Energy, the CPU fallbacks) work as they do in the CPU build. Two changes of the same date pay at any thread count: every rank of the collinear cluster solver now accumulates its slice of the density matrix on the device (before, only the rank that owns the dense solve did, and the others spent longer on one core than the owner spent in the solve), and the final density of an MD step takes the GPU tables instead of the one-rank service.

```sh
# 6 ranks x 3 threads on an 18-core node: every rank bound to 3 cores, its threads inside them
mpirun --map-by ppr:6:node:PE=3 --bind-to core -np 6 \
       -x OMP_NUM_THREADS=3 -x OMP_PROC_BIND=close -x OMP_PLACES=cores \
       ./openmx input.dat -nt 3
```

Runs with threads are not bitwise reproducible from one run to the next,
because OpenMP reductions can change their summation order.
`OPENMX_BAND_PROFILE=1` prints, per rank, the section times of Set_Nonlocal
(`SETNLPROF`), the cluster density matrix (`CLUSTERDMPROF`), truncation and
Total_Energy beside the existing lines.

## NVIDIA MPS: recommended whenever several ranks share a GPU
With flat MPI, all the MPI ranks of a node normally share one GPU — so run the
[NVIDIA CUDA Multi-Process Service (MPS)](https://docs.nvidia.com/deploy/mps/).
Without MPS the kernels of the different ranks are serialized by context
switching on the device; under MPS they execute concurrently.
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
under MPS; a larger rank count can use the GPU only without MPS.

## WSL2 and other WDDM platforms

Under WSL2 the CUDA runtime's `cudaMemGetInfo` reports a per-process view of
the device memory: the allocations of other processes — the other ranks'
CUDA contexts and arrays — are not subtracted, and an allocation beyond the
physical VRAM does not fail but is paged through system memory at a small
fraction of the speed. Every device-memory budget of this code divides the
free memory among the ranks of the node, so with the CUDA figure 12 ranks on
a 12 GB GeForce each believed the whole card was theirs, oversubscribed it
and stalled for minutes. The budgets therefore use the physically free
memory reported by NVML (`nvidia-smi`'s library) whenever it is lower than
the CUDA figure; on native Linux the two agree. A one-time line
`<GPU> device memory: the CUDA runtime reports ... but ... is physically
free (NVML)` on stdout (`level.of.stdout 2` and above) says that the cap is
active;
`OPENMX_GPU_MEMINFO_NVML=0` restores the plain CUDA figure.

A kernel whose stack frame exceeds the driver's default limit of 1 KiB
per thread makes its first launch grow the context's local-memory pool,
sized for every thread the device can hold (about 26 MiB per rank on an
RTX 5080 for a 1224-byte frame), at launch time. The Force3 trace kernel
had such a frame, and since the force batches size their device arenas to
fill the card down to a node-wide reserve, with 18 ranks on 16 GB its late
launches raced the other ranks' arenas for the last MiB and sometimes lost
(`cuLaunchKernel` out of memory in `Force3_GpuTrace`, C1000). Its private
spherical-harmonic tables are now stored flat, which brings the frame to
936 bytes, so no launch grows the pool any more; as a guard for future
kernels, the device initialization raises the stack limit to the largest
frame of OpenMX's own kernels, which the Makefile reads from the linked
binary with `cuobjdump` and relinks once with (idle at present, every
frame being below 1 KiB; `OPENMX_GPU_STACK_BYTES` overrides the value, 0
disables the step). The node-wide reserve of the force-phase budgets also
grows by 16 MiB per rank sharing the device. In the same situation the cluster
dense diagonalization used to fall back to ELPA because its reservation
probe demanded room for the basis of the refined FP32 eigensolver (n²
doubles, 1289 MiB at n=13000) next to the 1024 MiB reserve; when the dense
buffers and the reserve fit but the basis does not, the GPU path is now
kept with the FP64 eigensolver for that SCF cycle.

## Build and install
Building and installing is more difficult than with standard OpenMX. The NVIDIA build requires the [NVIDIA HPC SDK](https://developer.nvidia.com/hpc-sdk) with its Open MPI, and the AMD build requires ROCm, OpenMPI or Intel MPI, and AOCL, MKL or OpenBLAS; [Dependencies](#dependencies) below lists the requirements. The Makefile contains build examples for several supercomputer systems, and a ready-made site makefile is included for System C "Kugui" at ISSP, Univ. of Tokyo (`Makefile.kugui`, which includes the stock Makefile and builds with the NVHPC 24.7 / CUDA 12.5 that Kugui offers; its header lists the two nvc 24.x code-generation problems it works around, one of them reproduced by `tests/nvc_diag_vectorizer_bug.c`); please refer to it. `Makefile.pegasus` (the Pegasus supercomputer at the University of Tsukuba) and `Makefile.bunshiken` are older standalone copies that do not list the sources added since early October 2026. The first `make` also builds the bundled ELPA/COSMA stack, used by the CPU eigensolvers and "gpusolver2", and MAGMA when it is selected; this adds some time to the first build. A detailed implementation document (English and Japanese, including the list of GPU-related environment variables) is available under [doc/](doc/). If you're unsure about the build and installation process, feel free to ask in English via GitHub issues or [my X account](https://x.com/dc1394) (Japanese is also acceptable on my X account). I'll assist you as much as I can.

### Dependencies

Installed on the build machine:

- **NVIDIA build**: the NVIDIA HPC SDK (nvc, nvc++ and nvfortran, its Open MPI, and its LP64 ScaLAPACK, LAPACK and BLAS) with a CUDA toolkit. The default is HPC SDK 26.9 with CUDA 13.4; CUDA 12.3 or later is supported.
- **AMD build**: ROCm (clang, hipcc, flang and OpenMP offload), OpenMPI or Intel MPI, LP64 numerical libraries (AOCL, MKL or OpenBLAS with matching ScaLAPACK), and FFTW3. See [GPU backend build instructions](docs/GPU_BACKENDS.md).
- **Both**: GNU make, a C++20 compiler and standard library for the new core, Python 3.8 or later, and CMake 3.25.2 or later (also used by direct make to build the core). The bundled ELPA needs autoconf 2.71 or later, automake, libtool, m4, python3 and unzip. MAGMA's build needs `patch`, and in the NVIDIA build also CMake with CUDA support. The NVIDIA build of FFTW3 needs `rsync`.

Built during configuration or the first build:

| Library | Version | Used for | Backend | Location |
|---|---|---|---|---|
| Kokkos / bundled mdspan | 5.2.2 | C++ core execution and multidimensional views | Serial, OpenMP, CUDA, HIP; SYCL in the standalone core | CMake FetchContent; pinned release and SHA-256 in `core/CMakeLists.txt` |
| ELPA | 2026.02.002 | CPU eigensolvers ("elpa1", "elpa2") and "gpusolver2" | both | `source/third_party/dist/` |
| COSMA, COSTA, Tiled-MM | 2.8.4 and the commits it pins | distributed matrix products of "gpusolver2" | both | `source/third_party/dist/` |
| GEMMul8 | v3.5.2 | FP64 matrix products emulated on integer tensor cores | both | Git submodule `source/third_party/GEMMul8` |
| FFTW3 | 3.3.11 | FFTs | NVIDIA (AMD uses an external FFTW3, including AOCL's) | Git submodule `source/third_party/fftw3` and release tarball `source/third_party/fftw-3.3.11.tar.gz` |
| MAGMA | 2.10.0 | multi-GPU Cluster and Band eigensolver | NVIDIA when selected (below); AMD always | `source/third_party/hip/magma/` |
| CMake | 3.31.12 | building COSMA when the system CMake is older than 3.24 | both | `source/third_party/dist/` |

Initialize the submodules with `git submodule update --init --recursive`;
the AMD build needs only GEMMul8. The first core configuration downloads
Kokkos. For an offline build, provide the extracted pinned release using
`-DFETCHCONTENT_SOURCE_DIR_KOKKOS=/absolute/path/to/kokkos-5.2.2`
(direct make: pass it in `OPENMX_CORE_CMAKE_OPTIONS`). After that the stock
Makefile and the CMake workflow need no network access. FFTW3's Git sources lack its
generated codelets and `configure`, so the build adds them from the bundled
3.3.11 release tarball after checking its SHA-256. It downloads the tarball
from fftw.org only when that file is missing. `Makefile.kugui` also builds
autoconf 2.72, automake 1.16.5 and libtool 2.4.7 on its first build and
downloads them from ftp.gnu.org, unless their release tarballs are already
in `source/build/host-tools/src`.

In the NVIDIA build, MAGMA follows the `make` variable `WITH_MAGMA` (with
CMake, add it to `OPENMX_MAKE_OPTIONS`):

- `auto` (default): when `nvidia-smi` on the build node lists two or more GPUs, MAGMA is linked, and built first if `MAGMA_PREFIX` has none. An existing installation is used as it is. With fewer GPUs, only an existing installation is linked.
- `1`: MAGMA is built and linked on any node, and rebuilt when its toolchain, build script or patch changes. Use it on a single-GPU node, or on a login node without GPUs that builds for multi-GPU compute nodes. `Makefile.kugui` sets it.
- `0`: the build omits MAGMA.

The build prints its decision (`MAGMA: ...`) whenever it changes. MAGMA is
built from the bundled archive after its SHA-256 is verified; an empty
`MAGMA_ARCHIVE` downloads the same archive instead. OpenMX's device-map
patch is applied, and the library is installed in
`source/third_party/magma-install`, or inside the build directory with
CMake. `MAGMA_JOBS` (default 3) and `MAGMA_CMAKE` select its parallelism
and CMake. The AMD build always builds MAGMA, with OpenMX's Ozaki-II patch.

For the Kugui A100 environment used for the current validation:

```sh
cd source
# First build only: the site's CMake 3.20.2 is below the core's 3.25.2 minimum.
python3.9 -m pip install --target build/cmake-python 'cmake==3.31.10'
make -f Makefile.kugui -j3 GPUSOLVER2_JOBS=3 \
  OPENMX_CORE_CMAKE="$PWD/build/cmake-python/cmake/data/bin/cmake" \
  GPUSOLVER2_CMAKE="$PWD/build/cmake-python/cmake/data/bin/cmake"
```

This selects NVHPC 24.7, CUDA 12.5, the site MPI and `sm_80`; the login node
does not need a visible GPU. It also sets `WITH_MAGMA=1`, because a GPU node
has four A100s while the login node shows none, so the first build also
builds MAGMA, with the pip-installed CMake above when present. The local CMake 3.31.10 path reproduces the
tested build; without a suitable override or system CMake, the stack builder
bootstraps its bundled CMake instead. Use one build at a time: `-j8` hit the
login environment's 500-thread quota during this validation. On another installation, select its SDK/MPI paths and
pass `GEMMUL8_GPU_ARCH=80` to the normal Makefile. GPU tests run in an
allocated GPU job.

The site validation above predates this C++ migration. The core additionally
needs a C++20 compiler. `Makefile.kugui` selects the site's GCC 12.2 for the
core as well as the CUDA adapters; override it with `OPENMX_CORE_CXX` if needed.
The updated branch has not yet been revalidated on Kugui. Older NVHPC versions
can be combined with a separate modern GCC, but there is no C++17 fallback;
the CUDA core also requires nvcc 12.2 or later. See the
[compiler compatibility notes](docs/CPP_CORE_MIGRATION.md#古い-nvidia-コンパイラとの組み合わせ).

### HPC SDK 26.9 / CUDA 13.4 update

The default `source/Makefile` now uses HPC SDK 26.9 and CUDA 13.4. The
SDK's bundled toolkit and the separately installed toolkit can differ: on
the reference workstation SDK 26.9 bundles CUDA 13.3, while CUDA 13.4.2 is installed at
`/usr/local/cuda-13.4`. The build selects one consistent set of CUDA headers,
runtime and math libraries, and obtains the MPI library path from the SDK's
MPI wrapper. Compiler/toolkit changes invalidate cached build products.

```sh
cd source
make -j18 GEMMUL8_GPU_ARCH=120             # RTX 5080; 80 for A100, 90 for H100
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

On GPUs with weak native FP64, cuSOLVER handles enable FP64 fixed-point
emulation with the performant strategy and dynamic mantissa selection when
the toolkit supports it. A100-class devices default to native FP64;
`OPENMX_CUSOLVER_EMULATION=0|1` overrides the hardware choice. With CUDA 13.4 Update 1 or later
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
The cuSOLVER layer also compiles against CUDA 12.3. The bundled GEMMul8
translation unit supplies compatibility declarations for newer cuBLASLt
attributes referenced by upstream headers; OpenMX's instantiated
`cublasHandle_t` path does not execute that newer Lt setup. CUDA 12.5 is
supported by `Makefile.kugui`; CUDA 13.2 and 13.4 builds were verified earlier.
CUDA 12.5 uses cuSOLVER 11.6.3 and does not expose the newer FP64-emulation
or two-stage-selection APIs.

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
`OPENMX_CUSOLVER_EMULATION=0` disables cuSOLVER emulation and `=1` requests it
where supported;
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
rank arena, a reserve of 256 MiB plus 16 MiB per node rank and a 64 MiB
allowance per active rank.
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
# Architecture selection and explicit override regression checks:
python3 tests/run_cusolver_policy_cpu.py
python3 tests/run_gemmul8_policy_cpu.py
python3 tests/run_orbs_precision_policy_cpu.py
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

The test suite writes elapsed times and differences from the bundled reference
results to `runtest.result`. Inspect the output files and compare every force
component with `tools/compare_gpu_suites.py` as described above.

## Benchmarks
Published benchmarks of GPU-accelerated OpenMX are available in
[the JPSJ paper](https://journals.jps.jp/doi/10.7566/JPSJ.94.124003).
For measurements of the current version on your hardware, use the commands in
[Reproducible GPU regression runs](#reproducible-gpu-regression-runs).

## Important notes
Performance depends on the system, solver, hardware and MPI/OpenMP layout.
Use the same binary and inputs when comparing CPU and GPU execution;
`OPENMX_GPU=0` selects the CPU paths for that comparison. Check numerical
agreement alongside execution time.

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
Each component retains its original license; its original license file is
in its Git submodule or source archive under `source/third_party/`.

#### FFTW3

- **Source**: <https://www.fftw.org/>
- **Version**: 3.3.11 (Git submodule `source/third_party/fftw3` and release tarball `source/third_party/fftw-3.3.11.tar.gz`), used by the NVIDIA build; the AMD build uses external FFTW3 (AOCL or system/user installation)
- **Copyright**:
  - Copyright (c) 2003, 2007-14 Matteo Frigo
  - Copyright (c) 2003, 2007-14 Massachusetts Institute of Technology
- **License**: GNU General Public License v2.0 or later (`GPL-2.0-or-later`)
- **Used here under**: GPL v3 (selected via the "or any later version" clause)
- **Original license file**: `COPYING` in the submodule (`source/third_party/fftw3/COPYING`)

#### GEMMul8

- **Source**: <https://github.com/RIKEN-RCCS/GEMMul8>
- **Version**: v3.5.2 (Git submodule `source/third_party/GEMMul8`)
- **Copyright**: Copyright (c) 2025- RIKEN R-CCS
- **Responsible developer**: Yuki Uchino (RIKEN Center for Computational Science)
- **License**: MIT License (`MIT`)
- **Original license file**: `LICENSE` in the submodule (`source/third_party/GEMMul8/LICENSE`)

GEMMul8 (GEMMulate) is a library for emulating high-precision matrix
multiplication (SGEMM, DGEMM, CGEMM, ZGEMM) using INT8/FP8 matrix engines
based on the Ozaki Scheme II. 

If you use this project in academic work, please also cite the GEMMul8
upstream references (see the GEMMul8 repository).

#### ELPA, COSMA, COSTA and Tiled-MM

ELPA serves the CPU eigensolvers (`scf.eigen.lib elpa1` and `elpa2`) and,
with COSMA, the distributed multi-GPU cluster diagonalization
(`scf.eigen.lib gpusolver2`). The four libraries are bundled as source
archives under `source/third_party/dist/` and built automatically by the
first `make`; each archive carries its original license file:

- **ELPA** 2026.02.002 — <https://elpa.mpcdf.mpg.de/> — Copyright (c) the ELPA
  consortium — **GNU Lesser General Public License v3.0 (`LGPL-3.0`)**
  (`COPYING/` inside the archive)
- **COSMA** 2.8.4 — <https://github.com/eth-cscs/COSMA> — Copyright (c) ETH
  Zürich (parts: Advanced Micro Devices, Inc.) — **BSD 3-Clause License
  (`BSD-3-Clause`)**
- **COSTA** and **Tiled-MM** (COSMA dependencies, at the commits COSMA 2.8.4
  pins) — <https://github.com/eth-cscs/COSTA>,
  <https://github.com/eth-cscs/Tiled-MM> — Copyright (c) ETH Zürich —
  **BSD 3-Clause License (`BSD-3-Clause`)**

#### MAGMA

- **Source**: <https://icl.utk.edu/magma/>
- **Version**: 2.10.0 (`source/third_party/hip/magma/magma-2.10.0.tar.gz`, used by both backends)
- **Copyright**: Copyright (c) 2009-2023, The University of Tennessee
- **License**: BSD 3-Clause License (`BSD-3-Clause`); `COPYRIGHT` inside the archive

MAGMA is the multi-GPU dense eigensolver of the Cluster and Band solvers.
The AMD build always links it; the NVIDIA build links it as described under
[Dependencies](#dependencies). The build applies OpenMX's patches to the
extracted sources: `source/third_party/magma-device-map.patch` (NVIDIA:
maps MAGMA's logical GPU numbers to the selected Band GPU group) and
`source/third_party/hip/magma-ozaki2.patch` (AMD: routes the matrix
products of the eigensolvers through GEMMul8's Ozaki-II emulation).

#### Kokkos and bundled mdspan

The C++ core uses [Kokkos 5.2.2](https://github.com/kokkos/kokkos/releases/tag/5.2.2)
and its bundled mdspan headers. Both carry `Apache-2.0 WITH LLVM-exception`
notices. The release's `LICENSE` is retained in the fetched source and installed
as `share/licenses/openmx_core/Kokkos-LICENSE`. The source URL and SHA-256 are
pinned in `core/CMakeLists.txt`.

#### CMake (build tool)

- **Source**: <https://cmake.org/>
- **Version**: 3.31.12 (`source/third_party/dist/cmake-3.31.12.tar.gz`)
- **Copyright**: Copyright 2000-2024 Kitware, Inc. and Contributors
- **License**: BSD 3-Clause License (`BSD-3-Clause`); `Copyright.txt` inside the archive

It is built only when the system CMake is older than the 3.24 that COSMA
needs, and it is not part of the OpenMX executable.

### License Compatibility Summary

| Component | Original License | Compatible with GPL v3 |
|-----------|------------------|------------------------|
| OpenMX (upstream) | GPL-3.0-or-later | — (same license) |
| GPU modifications (this project) | GPL-3.0-or-later | — (same license) |
| FFTW3 | GPL-2.0-or-later | Yes (via "or later" clause) |
| GEMMul8 | MIT | Yes (permissive → strong copyleft) |
| ELPA 2026.02.002 | LGPL-3.0 | Yes (LGPL v3 code may be conveyed under GPL v3) |
| COSMA (with COSTA, Tiled-MM) | BSD-3-Clause | Yes (permissive → strong copyleft) |
| MAGMA 2.10.0 | BSD-3-Clause | Yes (permissive → strong copyleft) |
| CMake (build tool only) | BSD-3-Clause | Yes (permissive → strong copyleft) |

The combined work is distributed under the terms of GPL v3 or later.

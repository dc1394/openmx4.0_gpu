# openmx4.0_gpu

Alongside GPU acceleration, this fork includes [extended lanthanide
pseudopotentials and PAO basis sets](#extended-lanthanide-pseudopotentials-and-pao)
in `DFT_DATA19`: additional datasets absent from upstream OpenMX 4.0, plus
updated datasets for elements already supported upstream.

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
AOCL defaults to OpenMP BLIS (`libblis-mt`); set
`OPENMX_HIP_AOCL_THREADING=SEQUENTIAL` to select sequential BLIS explicitly.
The CUDA and HIP ELPA archives are OpenMP-enabled, so CPU diagonalization uses
both the MPI ranks and their OpenMP threads.
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
Kokkos SIMD kernels. In every build the bundled ELPA also gets the x86 SIMD
kernels the build host can run (SSE/AVX/AVX2/AVX-512; `GPUSOLVER2_ELPA_KERNELS`
for the NVIDIA Makefile, `OPENMX_CPU_ELPA_KERNELS` for the CPU build, `auto`
or `generic`), and the ELPA2 path selects the fastest one the CPU supports at
run time (`ELPA_DEFAULT_real_kernel` / `ELPA_DEFAULT_complex_kernel` override it). Existing `gpusolver`/`gpusolver2`/`gpusolver3` input decks
select CPU ELPA2 in this executable. The bundled ELPA is compiled with `-O2`:
its C SIMD kernels built by gcc 14 with `-O3` returned wrong eigenvectors
(with correct eigenvalues) while the generic kernel was right, so the CPU
numerics test checks eigenvectors too. CPU builds also copy the matching Core and
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
OpenMX's MPI/OpenMP by default. When the C, C++ and Fortran compilers are all
Intel's, `-DOPENMX_CPU_MKL_THREADING=INTEL` (in a new build directory) links
the threaded MKL with the same libiomp5, so the BLAS that ScaLAPACK and ELPA
call outside OpenMX's parallel regions uses each rank's OpenMP threads. On
2 x 64-core AMD EPYC 7702 nodes with 32 ranks x 4 threads per node it cut a
650-atom Band SCF with forces and stress (2 nodes) from 1194 s to 928 s
(diagonalization 946 s to 673 s) and five SCF iterations of a 2064-atom
DC-LNO cell (8 nodes) from 356 s to 205 s (diagonalization 281 s to 127 s);
total energies agreed to 5e-12 and 2e-10 Ha. Flat MPI with sequential MKL
(128 ranks per node) was slower still: the Band case did not finish within
1800 s. OpenBLAS uses a matching Netlib ScaLAPACK installation
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

The goal of the `refactor/cpp-kokkos-core` branch is a set of independently
buildable, testable and reusable C++ scientific libraries, composed into OpenMX.
Ownership, RAII and Kokkos support that goal. See the
[library architecture and acceptance criteria](docs/CPP_LIBRARY_ARCHITECTURE.md).
The first package, [OpenMX::basis](modules/basis/README.md), provides the radial
and reciprocal interpolation used by the application, with C++ and C APIs,
and independent C++ PAO/VPS readers with owned data and radial evaluation.
It also provides atomic density/Hartree operations, Gauss–Legendre and Lebedev
quadrature, spherical Bessel functions, spherical harmonics and their derivatives,
real-orbital transformations, Gaunt angular integrals with owned coefficient tables,
and real-space orbital radial values and radial derivatives.
The orbital radial API preserves the native first-interval and origin-continuation
conventions, separately from the existing PAO/PhiF radial policy. Owned tables
and validated borrowed views support allocation-free evaluation. Native scalar
orbitals and CUDA/HIP CPU fallbacks share the same numerical kernel.
It builds without Kokkos, MPI, BLAS or a GPU SDK:

```sh
cmake -S modules/basis -B build/basis
cmake --build build/basis --parallel
ctest --test-dir build/basis --output-on-failure
cmake --install build/basis --prefix "$PWD/build/basis-install"
```

External CMake projects use `find_package(OpenMXBasis 0.1 CONFIG REQUIRED)`
and link `OpenMX::basis`. Complete PAO basis evaluation and two-center matrix
integrals are still being assembled from these components.
The `core/` library consumes this package and contains Fourier
transforms for PAOs, PAO products, nonlocal
projectors and neutral-atom potentials, spherical Bessel functions and FFT-grid
validation. PAO grids, atomic densities, orbitals, VPS/nonlocal-projector data,
optional VNA expansion tables and their Fourier tables now have
calculation-scoped C++ ownership, with borrowed views for legacy consumers.
Ragged PAO-product tables use offsets into one contiguous allocation and
mdspan block views; their Gauss quadrature uses the common Kokkos kernels.
The second package, [OpenMX::xc](modules/xc/README.md), provides the
exchange-correlation functionals (Ceperley–Alder LDA/LSDA, Perdew–Wang 92
correlation, Slater exchange and GGA-PBE with its density and gradient
derivatives) as pure functions with C++ and C APIs and no global state.
Its arithmetic is one set of inline C kernels (`openmx/xc/kernels.h`) that the
library, C callers and the application's GPU device code (OpenACC routines on
CUDA, OpenMP target regions on HIP) share. The application's CPU paths in
`Set_XC_Grid.c`, `Force.c`, `Stress.c` and `Total_Energy.c` call it through
`libopenmx_core.so`; the five original C routines are frozen as test fixtures
and the package tests show the library bitwise identical to them. External CMake projects use
`find_package(OpenMXXc 0.1 CONFIG REQUIRED)` and link `OpenMX::xc`.
The third package, [OpenMX::io](modules/io/README.md), reads OpenMX keyword
files (scalar, vector and block keywords) through an owning `KeywordFile`
and the original nesting model `KeywordFiles`, with C++ and C APIs, no global
state and no process exit. It also parses basis specifications
("Si7.0-s2p2d1"), the species, atom and unit-cell blocks, and composes them
with the structural keywords into typed `Settings`; `Input_std.c` reads those
blocks through it. The application's `Inputtools.c` is a thin
compatibility layer over it, and the frozen original is the reference of the
package tests. External CMake projects use
`find_package(OpenMXIo 0.1 CONFIG REQUIRED)` and link `OpenMX::io`.
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
and contiguous weighted radial rows. Its immutable Gaunt tables are initialized
on the first CPU request and reused by both VNA2/3 orientations; repeated access
allocates no memory. GPU-only calls do not allocate those host tables.
Both backends share the CPU sampling
loop, and DS_VNA GPU batches also receive their state explicitly per call.
DS_VNA CPU assembly now shares one evaluator across CUDA/HIP. Per-thread C++
workspaces own packed angular matrices, radial sums and Bessel scratch;
species-local offsets replace temporary five-level pointers and omit empty
basis groups. The zero-wavevector Bessel derivative initializes the extra order
it reads, and per-atom timing updates are safe across threads.
DS_VNA and VNA2/3 CPU SIMD scratch now belongs to those same C++ workers.
Lazy tensors replace 15 manual numeric allocations; explicit weight and Gaunt
strides remove five runtime row-pointer lists. Reacquisition preserves storage
without allocation, and empty basis groups skip absent angular buffers.
The changed VNA descriptors use distinct `_flat` C entry points and a versioned
private SIMD dispatcher. Rebuild and deploy the application, Core and SIMD
libraries together; mixed versions fail to resolve their interface symbols.
Shared CPU projector, conversion and radial-product transposes also use C++
owners. They snapshot explicit input views, expose immutable contiguous tables
to workers and release partial construction on failure. The flat radial-view
header can be used without including a calculation context. Native quadratures
and backend kernels keep their existing operation order.
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
SCF iterations reuse these allocations. Density workers process whole atoms
unless that leaves threads idle: when a rank owns fewer than four atoms per
thread (a small cell, or many ranks per atom) and cutting balances the team at
least a quarter better, the rank's pair points are cut into ranges of equal
work, one per thread. Each thread sums its range into a private copy of the
atom densities (at most 1 GiB per rank), and the last thread to finish adds the
copies in thread order. That order depends on the thread count, so the density
can then differ in the last bits between team sizes. Hamiltonian workers retain
the small spin-unpolarized blocked integral.
The CPU quadratures use Kokkos experimental SIMD with explicit host ABIs:
AVX2 (4 doubles), AVX-512 (8 doubles), AArch64 NEON (2 doubles), or Linux
AArch64 SVE (128/256/512 bits: 2/4/8 doubles), with a scalar fallback.
Hamiltonian matrix accumulators stay in registers over a
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
are hidden; C entry points exchange plain descriptors and function tables
across the library boundary. Device compilers never compile the host SIMD kernels.
CMake builds and installs this library alongside `libopenmx_core`; keep both
libraries when copying an installation. If the Core C++ compiler is a device
wrapper, specify `-DOPENMX_HOST_SIMD_CXX_COMPILER=/path/to/g++` (or `clang++`).
`-DOPENMX_CORE_HOST_SIMD=OFF` builds only the scalar implementation.
On Linux AArch64, GCC/Clang builds also enable SVE when the compiler supports
its fixed vector lengths. `OPENMX_HOST_SIMD=auto` (the default) or `sve`
selects SVE only when the OS enables it and the calling thread's current
vector length exactly matches a compiled variant. Other lengths fall back
to NEON. `scalar`, `neon`, `sve128`, `sve256` and `sve512` select the named
usable variant; an unavailable explicit request falls back to automatic
selection. `legacy` runs the original C loops. No thread's SVE vector length
is changed. Set `-DOPENMX_CORE_HOST_SVE=OFF` to keep NEON without SVE.
The private SVE ABI uses ARM ACLE with Kokkos's SIMD interface, including
masked tails and the half-width FP32/index vectors needed by these kernels.
It does not depend on Kokkos 5.2.2's incomplete SVE conversion support.
NEON FP64-to-FP32 conversion follows ordinary scalar rounding, including
ties, instead of the round-to-odd conversion in Kokkos 5.2.2.
For ARM numerical validation with real MPI calculations, see
[the CPU validation instructions](docs/GPU_BACKENDS.md#armneonsve-の検証).
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
- The local tile products of the COSMA transforms go through the same GEMMul8 bridge as the other dense products (CUDA build): GEMMul8 on GPUs with weak FP64 (the GeForce default), native cuBLAS FP64 on A100/H100, `scf.gemmul8.enable` and the workspace guards as elsewhere; `OPENMX_GS2_GEMMUL8=0` keeps COSMA on cuBLAS FP64. The hook is a small patch of Tiled-MM that the stack build applies (`source/third_party/tiledmm-openmx-gemm-hook.patch`); the ROCm Tiled-MM keeps rocBLAS.
- The eigensolves stay in FP64 (ELPA GPU kernels). The refined FP32 eigensolver of `gpusolver` (an FP32 solve plus Ogita-Aishima correction through GEMMul8) does not pay in the distributed path: each correction step needs at least four n×n×n distributed products, and on the measured sizes (sidia333, 8 ranks on one RTX 5080) one COSMA product costs as much as the time the FP32 solve saves (`OPENMX_GS2_FP32_PROBE=1` times the FP32 solve alone; see [docs/PERFORMANCE_REGRESSION.md](docs/PERFORMANCE_REGRESSION.md)).
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
The large dense matrix multiplications of the GPU eigensolver path (including the local tile products of the distributed `gpusolver2` transforms) use a device-aware bridge to [GEMMul8](https://github.com/RIKEN-RCCS/GEMMul8) or native cuBLAS FP64. GEMMul8 emulates FP64 GEMM on INT8 tensor cores using the Ozaki scheme II and remains the default on consumer GPUs (GeForce) with limited native FP64 throughput. General GEMMs on A100 and H100 instead default to native FP64, including the FP64 Tensor Core kernels selected by cuBLAS, avoiding integer decomposition and its workspace. The choice uses the device's FP32:FP64 throughput ratio: a known positive ratio below 8 selects native FP64. A100 has a separate size-dependent exception for the collinear cluster forward transform, described below.

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
Building and installing is more difficult than with standard OpenMX. The NVIDIA build requires the [NVIDIA HPC SDK](https://developer.nvidia.com/hpc-sdk) with its Open MPI, and the AMD build requires ROCm, OpenMPI or Intel MPI, and AOCL, MKL or OpenBLAS; [Dependencies](#dependencies) below lists the requirements. The Makefile contains build examples for several supercomputer systems, and a ready-made site makefile is included for System C "Kugui" at ISSP, Univ. of Tokyo (`Makefile.kugui`, which includes the stock Makefile and builds with the NVHPC 24.7 / CUDA 12.5 that Kugui offers; its header lists the two nvc 24.x code-generation problems it works around, one of them reproduced by `tests/nvc_diag_vectorizer_bug.c`); please refer to it. `Makefile.pegasus` also includes the shared Makefile, with Pegasus H100 defaults (NVHPC 26.5 / CUDA 13.2 and GCC 12 for the C++20 components). `Makefile.bunshiken` is an older standalone copy that does not list the sources added since early October 2026. The first `make` also builds the bundled ELPA/COSMA stack, used by the CPU eigensolvers and "gpusolver2", and MAGMA when it is selected; this adds some time to the first build. A detailed implementation document (English and Japanese, including the list of GPU-related environment variables) is available under [doc/](doc/). If you're unsure about the build and installation process, feel free to ask in English via GitHub issues or [my X account](https://x.com/dc1394) (Japanese is also acceptable on my X account). I'll assist you as much as I can.

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
| GEMMul8 | v3.5.4 | FP64 matrix products emulated on integer tensor cores | both | Git submodule `source/third_party/GEMMul8` |
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

For Pegasus H100, `make -C source -f Makefile.pegasus -j12 openmx` uses
NVHPC 26.5 / CUDA 13.2, GCC 12 and `sm_90`, without querying a GPU on the
login node. Supply `PEGASUS_CMAKE=/path/to/cmake` if CMake >=3.25.2 is not
available; the wrapper also recognizes the dependency builder's local
CMake installation. MAGMA is disabled by default for its one-GPU nodes.
The common Makefile applies an NVHPC 26.5 workaround to explicit `fmaf`
calls in three OpenACC source files while retaining the orbital evaluator's
`-gpu=nofma` accuracy policy. The isolated CMake build commands, H100
verification and MPI/OpenMP measurements are in
[the Pegasus validation record](doc/pegasus_refactor_validation_20261027.txt).
The [Pegasus job script](tools/pegasus_openmx.pbs) starts private MPS and
defaults to the measured 12 MPI x 4 OpenMP layout; both counts are configurable.

For Sirius MI300A, the [build and validation record](doc/sirius_unified_validation_20261027.txt)
covers the unified HIP build, one APU, four local GPUs and two physical
nodes with eight GPUs. The [Sirius PBS script](tools/sirius_openmx.pbs)
defaults to 8 MPI x 3 OpenMP per APU for CPU and GPU runs, hence
32 MPI x 3 OpenMP on a full four-APU node. It records rank placement and
loads the matching installed libraries. From `work/`, submit with
`qsub -v OMX_INPUT=input.dat ../tools/sirius_openmx.pbs`.
The [Sirius threading audit](doc/sirius_threading_validation_20261031.txt)
records ELPA/BLAS worker checks and the replacement of earlier native suite timings.
The [one-node 32 MPI x 3 OpenMP benchmark results](#benchmark-sirius-mi300a)
compare official CPU, fork CPU and ordinary GPU execution for S14, L16,
sidia333 and cghelgeo. All 96 energy/force comparisons pass 1e-7; the
original 25-SCF sidia333 deck is retained and remains unconverged.

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

GEMMul8 is pinned to upstream **v3.5.4**
(`f8fdf0b2a5eafd4dbbfab6be8489d3356e0da06a`). Initialize/update submodules
with `git submodule update --init --recursive` after updating the repository.
The update includes upstream's large-column kernel-launch fixes and FP8
workspace/reduction improvements; OpenMX continues to instantiate the INT8
backend for double and complex-double products. The adapter enables memory-saving
blocking with a 256 MiB default target (`OPENMX_GEMMUL8_MAX_WORKSPACE_MB`),
and divides the available-memory budget between ranks sharing each GPU.
The ordinary CUDA, Pegasus and HIP Makefiles pin the same release; Kugui
inherits the ordinary CUDA pin. See the [update validation](docs/GEMMUL8_3.5.4_VALIDATION.md)
for numerical and compiler checks.

`GEMMUL8_TEST_LARGE_N=1 tests/run_gemmul8_smoke.sh` additionally checks the
262,141-column launch boundary against a CPU `long double` reference. This
optional check disables workspace blocking and needs several GiB of free GPU
memory; run it alone on the device. The default smoke test retains its 256 MiB cap.

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
`cublasHandle_t` path does not execute that newer Lt setup. It also omits
unused FP8 register hints when the toolkit lacks `__maxnreg__`.
The CUDA 12.3 cuSOLVER check does not establish a complete GEMMul8 build:
nvcc 12.3 with GCC 12 rejects an upstream grouped-CRT template in both
GEMMul8 v3.5.2 and v3.5.4. Use CUDA 12.5 or newer for that build. CUDA 12.5 is
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

## Extended lanthanide pseudopotentials and PAO

[DFT_DATA19/VPS](DFT_DATA19/VPS) and [DFT_DATA19/PAO](DFT_DATA19/PAO) contain
this fork's additional and updated lanthanide datasets, versioned together
with the code. The reference for the original OpenMX 4.0 data is the
[official VPS/PAO database (2019)](https://www.openmx-square.org/vps_pao2019/).
This fork adds Eu, Gd and Er datasets with both explicit 4f valence states
and open-core treatment, and Tm open-core datasets; these elements have no
downloadable entries in that official database. It also adds Ho open-core
data and revises datasets for elements already present upstream, including
Yb with 4f in valence, Yb open core and Dy open core.

The extension includes the following combinations. `CA26` and `PBE26` are
the filenames of this fork's LDA and GGA-PBE VPS datasets, respectively;
they are not an upstream database release designation. `_OC` denotes
open-core treatment of the 4f states.

| Element(s) | 4f in valence | Open core |
| --- | --- | --- |
| Eu, Gd, Er | CA26/PBE26 VPS and PAO | CA26/PBE26_OC VPS and PAO |
| Tm | — | CA26/PBE26_OC VPS and PAO |
| Tb, Yb | Updated CA26/PBE26 VPS and PAO | Updated CA26/PBE26_OC VPS and PAO |
| Ho | Existing 2019 data | CA26/PBE26_OC VPS and PAO |
| Dy | Existing 2019 data | Updated CA26/PBE26_OC VPS and replaced PAO |
| Sm | Existing 2019 data | Updated CA26/PBE26_OC VPS with the existing PAO |

The Ho open-core data are
`Ho_CA26_OC.vps` / `Ho_PBE26_OC.vps` and `Ho8.0_OC.pao`, `Ho10.0_OC.pao`,
`Ho12.0_OC.pao`.

For Tb open-core calculations, use the supplied `Tb_CA26_OC.vps` or
`Tb_PBE26_OC.vps` with `Tb8.0_OC.pao`, `Tb10.0_OC.pao` or `Tb12.0_OC.pao`.
These form the versioned open-core set; the names without `_OC` select
the separate 4f-in-valence data.

PAO files use confinement radii of 8.0, 10.0 and 12.0 bohr. For example,
`Yb8.0.pao` pairs with `Yb_PBE26.vps` for explicit 4f valence states, while
`Yb8.0_OC.pao` pairs with `Yb_PBE26_OC.vps` for open-core treatment. Choose the
matching names in `Definition.of.Atomic.Species`, omitting the extensions:

```text
# Species   PAO basis              VPS
Yb          Yb8.0-s3p2d2f1         Yb_PBE26
# Alternative open-core definition for the same species:
# Yb        Yb8.0_OC-s3p2d2f1      Yb_PBE26_OC
```

The supplied `Yb_CA26_OC`/`Yb_PBE26_OC` files and their PAOs specify **11
valence electrons** (4f13 in the core), whereas the
[upstream Yb open-core dataset](https://www.openmx-square.org/vps_pao2019/Yb/Yb_OC/index.html)
specifies 10 valence electrons. Set initial spin populations and any
open-core occupations from the selected dataset. The supplied 4f-in-valence
Yb VPS files specify 24 valence electrons; the separately named
`Yb_PBE24.vps` is also retained under version control.

The newly added/replaced PAOs contain s, p, d and f radial channels
(`maxL.pao = 3`, `num.pao = 15`). In particular, the updated `Dy8.0_OC.pao`,
`Dy10.0_OC.pao` and `Dy12.0_OC.pao` replace the previous files at the same
paths. Existing 2019 VPS filenames remain available; record the repository
commit and exact PAO/VPS names when reproducing a calculation. These
datasets can be selected in CPU, CUDA and HIP builds.

## Benchmarks

Published benchmarks of GPU-accelerated OpenMX are available in
[the JPSJ paper](https://journals.jps.jp/doi/10.7566/JPSJ.94.124003).
For measurements of the current version on your hardware, use the commands in
[Reproducible GPU regression runs](#reproducible-gpu-regression-runs).

Two input decks are also tracked in [benchmarks/inputs](benchmarks/inputs),
copied verbatim from the supplied benchmark inputs:

| Input | System | Solver / spin | SCF limit / MD limit |
| --- | --- | --- | --- |
| [sidia333_nc_cluster.dat](benchmarks/inputs/sidia333_nc_cluster.dat) | 216 Si atoms | Cluster / noncollinear | 25 / 1 |
| [cghelgeo.dat](benchmarks/inputs/cghelgeo.dat) | 650 atoms (H, C, N, O, P) | Band / spin unpolarized, 1 × 1 × 3 k points | 200 / 1 |

Both inputs use `DATA.PATH ../../DFT_DATA19`. Prepare separate run
directories two levels below the checkout root so that the path resolves
to the bundled data and outputs stay separate from the versioned inputs:

```sh
# From the repository root:
mkdir -p work/bench_sidia333_nc_cluster work/bench_cghelgeo
cp benchmarks/inputs/sidia333_nc_cluster.dat work/bench_sidia333_nc_cluster/
cp benchmarks/inputs/cghelgeo.dat work/bench_cghelgeo/

# Example after building work/openmx; use the MPI matching that binary.
cd work/bench_sidia333_nc_cluster
OMP_NUM_THREADS=1 OPENMX_GPU=1 mpirun -np 18 ../openmx sidia333_nc_cluster.dat > run.log 2>&1
# For the other case, run ../openmx cghelgeo.dat from work/bench_cghelgeo.
```

For a CPU-path comparison with the same GPU-enabled binary, use
`OPENMX_GPU=0` in a fresh run directory. Enable MPS before NVIDIA GPU
measurements when comparing against the MPS baseline. Keep the input
settings unchanged and record the actual SCF iterations and convergence
status along with time, energy and forces; reaching an iteration limit
does not establish SCF convergence.

All five environments use the same reporting order: measurement conditions,
overall results, case results, accuracy and convergence, and measurement records.
Case timings are displayed below in this README; the CSV/JSON links provide
the underlying records. **Fork GPU** means CUDA on NVIDIA and HIP on AMD.
All timings are seconds. Ratios above 1 favor the denominator configuration;
upstream comparisons include compiler, MPI and numerical-library effects.
Compare routes within each environment: hardware, affinity, input controls
and timing aggregation are specified for each measurement. SCF counts cover
all MD steps. Per-case error columns are maxima over that environment's
recorded comparisons; the accuracy section states their scope.
A dash means no completed measurement is available. SCF-count lists follow
repeat order; `25 × 3` means three 25-step runs. `PASS`/`FAIL` describes
all recorded comparisons for that input, including geometry/grid checks,
with energy/force tolerances of `1e-7` Ha and `1e-7` Ha/bohr, separately
from SCF convergence.

| Environment | MPI × OpenMP per node | GPU backend | GPU(s) per node |
| --- | --- | --- | ---: |
| [GeForce RTX 5080](#benchmark-rtx5080) | 18 × 2 (SMT siblings) | CUDA + MPS | 1 |
| [Kugui A100](#benchmark-kugui-a100) | 32 × 2 (physical cores) | CUDA + MPS | 2 (4 reserved) |
| [Pegasus H100](#benchmark-pegasus-h100) | 24 × 2 (physical cores) | CUDA + MPS | 1 |
| [Radeon RX 9060 XT](#benchmark-radeon-rx9060xt) | 16 × 2 (physical cores) | HIP | 1 |
| [Sirius MI300A](#benchmark-sirius-mi300a) | 32 × 3 (physical cores) | HIP | 4 |

<a id="benchmark-rtx5080"></a>

### GeForce RTX 5080 / Core i9-10980XE (2026-10-31–2026-11-01)

#### Measurement conditions

`main` at `0681d63e` was pulled, rebuilt with CMake in Release mode and
installed before this measurement. The host has an Intel Core i9-10980XE
(18 physical cores, SMT enabled: 36 logical CPUs), 256 GiB RAM and one
GeForce RTX 5080 (16 GiB, `sm_120`, driver 595.99.02). All runs used
**18 MPI ranks × 2 OpenMP threads**: rank `i` used the two SMT siblings
`i` and `i+18` on one physical core. Affinity probes using each MPI/OpenMP
toolchain verified all 36 logical CPUs and two threads per rank.
Every invocation used MPI `-np 18`, OpenMX `-nt 2` and `OMP_NUM_THREADS=2`.

| Configuration | Build and execution |
| --- | --- |
| Upstream CPU | Existing `~/openmx4.0/work/openmx`; Intel icx/ifx 2026.0, Intel MPI 2021.18, LP64 oneMKL 2026.0; `scf.eigen.lib elpa2` |
| Fork CPU path | Installed NVIDIA Release build, NVHPC 26.9 / GCC 14.2 C++ core, Open MPI 5.0.10rc2, SDK LP64 OpenMP BLAS/LAPACK/ScaLAPACK, bundled OpenMP ELPA 2026.02.002, host SIMD; `OPENMX_GPU=0`, `scf.eigen.lib elpa2` |
| Fork CUDA path | The same installed executable and matching Core/SIMD libraries, CUDA 13.4, a private NVIDIA MPS daemon; `OPENMX_GPU=1`, `scf.eigen.lib gpusolver` |

Each configuration was measured once, sequentially, with no concurrent
calculation or build. The order for each test was CUDA, fork CPU, upstream;
the tests ran in the order `-runtest`, sidia333, cghelgeo, `-runtestL`.
These are individual measurements, not repeated-run averages. The CPU
comparison includes compiler, MPI, eigensolver and numerical-library effects.
CUDA timings include CPU work and any automatically selected fallbacks.
All four CUDA jobs were observed with 18 clients connected to MPS.
Each suite ran as one native `-runtest*` MPI job, retaining the normal
cache and cleanup behavior between cases.

The three configurations used the same 32 input definitions and 44 PAO/VPS
files. Only `DATA.PATH` (an absolute shared path) and `scf.eigen.lib` (ELPA2
for CPU execution) were changed in measurement copies. Geometry, grids,
SCF/MD limits, mixing and convergence criteria were preserved. OpenMP used
`OMP_DYNAMIC=FALSE`, `OMP_PROC_BIND=close`, `OMP_PLACES=threads`,
`OMP_MAX_ACTIVE_LEVELS=1` and a 64 MiB thread stack. The numerical-library
controls were `MKL_NUM_THREADS=2`, `MKL_DYNAMIC=FALSE`,
`OPENBLAS_NUM_THREADS=2` and `BLIS_NUM_THREADS=2`.
Open MPI used `--use-hwthread-cpus --map-by slot:PE=2 --bind-to hwthread`;
Intel MPI used the equivalent explicit CPU masks. Installed binaries and
matching libraries were frozen for the campaign; the Core/SIMD copies were
preloaded to keep their versions fixed.

Before timing, separate instrumented native C60/GaAs suite runs observed
two actual ELPA workers and two BLAS workers on **every one of the 18 ranks**
of both fork paths. CPU ELPA reported `requested=2 configured=2 MPI_thread=2`;
all three routes retained an OpenMP limit of two at exit. Upstream MKL also
reported `NThr:2`. The Intel compiler's intrinsic calls are not visible to
the public OpenMP ABI observer; its zero observation masks are not worker counts.
The production CPU ELPA adapter passed 24 real/complex eigenpair, residual,
orthogonality and leading-dimension cases per rank at 8 MPI × 2 threads.
The [threading audit](benchmarks/results/rtx5080_20261031/threading-audit.json)
is separate from the uninstrumented benchmark timings.

#### Overall results

Suite totals sum native per-case `runtest.result` / `runtestL.result` times.
Named inputs use the maximum-rank `Total Computational Time`. Each value
is one measurement. MPI launch/shutdown overhead is excluded.

<details>
<summary>MPI-job wall times, including the monitor</summary>

The following wall times include the MPI job and the measurement monitor
(two-second polling interval); they are reported separately from native calculation times.

| Test | Upstream wall (s) | Fork CPU wall (s) | Fork GPU wall (s) |
| --- | ---: | ---: | ---: |
| `-runtest` (14 cases) | 112.44 | 71.53 | 87.64 |
| `-runtestL` (16 cases) | 7558.02 | 5490.24 | 4360.74 |
| `sidia333_nc_cluster.dat` | 836.13 | 588.08 | 105.01 |
| `cghelgeo.dat` | 3216.51 | 3853.35 | 571.32 |

</details>

| Test | Nodes | Upstream CPU (s) | Fork CPU (s) | Fork GPU (s) | Upstream / Fork CPU | Upstream / GPU | Fork CPU / GPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `-runtest` (14 cases) | 1 | 110.130 | 68.860 | 82.640 | 1.599 | 1.333 | 0.833 |
| `-runtestL` (16 cases) | 1 | 7551.800 | 5485.470 | 4354.990 | 1.377 | 1.734 | 1.260 |
| `-runtestL2` (7 cases) | — | — | — | — | — | — | — |
| `-runtestL3` (20 cases) | — | — | — | — | — | — | — |
| `sidia333_nc_cluster.dat` | 1 | 832.660 | 584.729 | 101.277 | 1.424 | 8.222 | 5.774 |
| `cghelgeo.dat` | 1 | 3214.383 | 3850.342 | 566.828 | 0.835 | 5.671 | 6.793 |

#### Case results

##### `-runtest` (14 cases)

| Input | Upstream CPU (s) | Fork CPU (s) | Fork GPU (s) | Upstream / Fork CPU | Upstream / GPU | Fork CPU / GPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Benzene | 4.240 | 2.540 | 4.200 | 1.669 | 1.010 | 0.605 |
| C60 | 9.320 | 6.600 | 11.000 | 1.412 | 0.847 | 0.600 |
| CO | 7.770 | 6.070 | 20.070 | 1.280 | 0.387 | 0.302 |
| Cr2 | 7.610 | 4.640 | 4.330 | 1.640 | 1.758 | 1.072 |
| Crys-MnO | 14.450 | 8.210 | 7.290 | 1.760 | 1.982 | 1.126 |
| GaAs | 25.310 | 15.420 | 11.360 | 1.641 | 2.228 | 1.357 |
| Glycine | 4.240 | 2.750 | 3.140 | 1.542 | 1.350 | 0.876 |
| Graphite4 | 3.330 | 1.030 | 1.350 | 3.233 | 2.467 | 0.763 |
| H2O | 3.520 | 1.400 | 1.920 | 2.514 | 1.833 | 0.729 |
| H2O-EF | 3.830 | 2.940 | 3.010 | 1.303 | 1.272 | 0.977 |
| HMn | 12.200 | 9.460 | 7.650 | 1.290 | 1.595 | 1.237 |
| Methane | 2.820 | 1.260 | 1.710 | 2.238 | 1.649 | 0.737 |
| Mol_MnO | 7.430 | 4.820 | 4.420 | 1.541 | 1.681 | 1.090 |
| Ndia2 | 4.060 | 1.720 | 1.190 | 2.360 | 3.412 | 1.445 |
| **Total** | **110.130** | **68.860** | **82.640** | **1.599** | **1.333** | **0.833** |

<details>
<summary>SCF counts and numerical checks: -runtest (14 cases)</summary>

| Input | SCF steps (upstream / fork CPU / GPU) | Max abs ΔE (Ha) | Max abs ΔF (Ha/bohr) | Check |
| --- | --- | ---: | ---: | --- |
| Benzene | 19 / 19 / 19 | 9.948e-13 | 0.000e+00 | PASS |
| C60 | 27 / 27 / 27 | 5.201e-11 | 8.000e-12 | PASS |
| CO | 16 / 16 / 16 | 0.000e+00 | 2.100e-11 | PASS |
| Cr2 | 21 / 21 / 21 | 2.984e-12 | 1.720e-10 | PASS |
| Crys-MnO | 46 / 42 / 39 | 4.820e-10 | 6.390e-10 | PASS |
| GaAs | 13 / 13 / 13 | 2.001e-11 | 8.000e-12 | PASS |
| Glycine | 20 / 20 / 20 | 9.948e-13 | 1.000e-12 | PASS |
| Graphite4 | 14 / 14 / 14 | 3.002e-12 | 1.000e-12 | PASS |
| H2O | 31 / 31 / 31 | 1.002e-12 | 1.000e-12 | PASS |
| H2O-EF | 17 / 17 / 17 | 9.983e-13 | 1.000e-12 | PASS |
| HMn | 26 / 26 / 26 | 9.948e-13 | 0.000e+00 | PASS |
| Methane | 14 / 14 / 14 | 1.170e-10 | 1.300e-11 | PASS |
| Mol_MnO | 26 / 26 / 26 | 9.948e-13 | 1.000e-12 | PASS |
| Ndia2 | 13 / 13 / 13 | 4.000e-12 | 1.000e-12 | PASS |

</details>

##### `-runtestL` (16 cases)

| Input | Upstream CPU (s) | Fork CPU (s) | Fork GPU (s) | Upstream / Fork CPU | Upstream / GPU | Fork CPU / GPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 5_5_13COb2 | 101.410 | 89.280 | 55.660 | 1.136 | 1.822 | 1.604 |
| B2C62_Band | 758.240 | 454.940 | 366.610 | 1.667 | 2.068 | 1.241 |
| CG15c-DC-LNO | 174.400 | 136.070 | 199.670 | 1.282 | 0.873 | 0.681 |
| DIA512-1 | 173.090 | 119.960 | 220.500 | 1.443 | 0.785 | 0.544 |
| FeBCC | 204.730 | 133.670 | 137.490 | 1.532 | 1.489 | 0.972 |
| GEL | 58.520 | 36.480 | 35.010 | 1.604 | 1.672 | 1.042 |
| GFRAG | 47.970 | 39.190 | 30.730 | 1.224 | 1.561 | 1.275 |
| GGFF | 1871.230 | 1743.060 | 552.960 | 1.074 | 3.384 | 3.152 |
| MCCN | 248.630 | 244.900 | 358.550 | 1.015 | 0.693 | 0.683 |
| Mn12_148_F | 116.810 | 94.860 | 53.290 | 1.231 | 2.192 | 1.780 |
| N1C999 | 3153.740 | 1916.450 | 2050.110 | 1.646 | 1.538 | 0.935 |
| Ni63-O64 | 115.550 | 78.030 | 49.000 | 1.481 | 2.358 | 1.592 |
| Pt63 | 88.100 | 68.300 | 45.220 | 1.290 | 1.948 | 1.510 |
| SialicAcid | 24.820 | 19.320 | 14.460 | 1.285 | 1.716 | 1.336 |
| ZrB2_2x2 | 287.880 | 201.220 | 119.600 | 1.431 | 2.407 | 1.682 |
| nsV4Bz5 | 126.680 | 109.740 | 66.130 | 1.154 | 1.916 | 1.659 |
| **Total** | **7551.800** | **5485.470** | **4354.990** | **1.377** | **1.734** | **1.260** |

<details>
<summary>SCF counts and numerical checks: -runtestL (16 cases)</summary>

| Input | SCF steps (upstream / fork CPU / GPU) | Max abs ΔE (Ha) | Max abs ΔF (Ha/bohr) | Check |
| --- | --- | ---: | ---: | --- |
| 5_5_13COb2 | 67 / 66 / 67 | 5.900e-10 | 3.250e-10 | PASS |
| B2C62_Band | 203 / 203 / 204 | 6.798e-11 | 3.610e-08 | PASS |
| CG15c-DC-LNO | 40 / 42 / 42 | 1.372e-09 | 1.560e-10 | PASS |
| DIA512-1 | 29 / 29 / 29 | 7.913e-10 | 8.856e-09 | PASS |
| FeBCC | 62 / 62 / 62 | 1.114e-09 | 6.800e-11 | PASS |
| GEL | 58 / 58 / 58 | 2.390e-10 | 3.000e-10 | PASS |
| GFRAG | 60 / 60 / 60 | 4.130e-10 | 1.884e-09 | PASS |
| GGFF | 102 / 97 / 101 | 1.301e-10 | 1.290e-10 | PASS |
| MCCN | 67 / 67 / 67 | 3.811e-10 | 3.600e-11 | PASS |
| Mn12_148_F | 61 / 61 / 61 | 2.701e-10 | 5.440e-10 | PASS |
| N1C999 | 64 / 59 / 59 | 1.560e-09 | 3.200e-11 | PASS |
| Ni63-O64 | 68 / 69 / 62 | 3.471e-09 | 7.860e-10 | PASS |
| Pt63 | 75 / 70 / 67 | 4.563e-09 | 3.420e-10 | PASS |
| SialicAcid | 29 / 29 / 29 | 3.140e-10 | 2.150e-10 | PASS |
| ZrB2_2x2 | 49 / 49 / 49 | 2.019e-10 | 3.390e-10 | PASS |
| nsV4Bz5 | 57 / 57 / 57 | 2.220e-10 | 1.186e-09 | PASS |

</details>

##### Named inputs

| Input | Upstream CPU (s) | Fork CPU (s) | Fork GPU (s) | Upstream / Fork CPU | Upstream / GPU | Fork CPU / GPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `sidia333_nc_cluster.dat` | 832.660 | 584.729 | 101.277 | 1.424 | 8.222 | 5.774 |
| `cghelgeo.dat` | 3214.383 | 3850.342 | 566.828 | 0.835 | 5.671 | 6.793 |

<details>
<summary>SCF counts and numerical checks: Named inputs</summary>

| Input | SCF steps (upstream / fork CPU / GPU) | Max abs ΔE (Ha) | Max abs ΔF (Ha/bohr) | Check |
| --- | --- | ---: | ---: | --- |
| `sidia333_nc_cluster.dat` | 25 / 25 / 25 | 1.950e-10 | 1.840e-10 | PASS |
| `cghelgeo.dat` | 58 / 59 / 58 | 7.502e-08 | 1.260e-08 | PASS |

</details>

#### Accuracy and convergence

All **186 numerical checks passed**, including all 90 bundled-reference
comparisons. The largest absolute energy difference was `7.502e-8` Ha and
the largest force-component difference was `3.610e-8` Ha/bohr. Per-case
maxima above cover all three configuration pairs and bundled references.

Sidia333 reached its original 25-SCF limit and remains unconverged in all
three routes. The other inputs converged at every MD step. Named-input
SCF counts were 25/25/25 for sidia333 and 58/59/58 for cghelgeo
(upstream / fork CPU / GPU).

#### Measurement records

Versioned [timings, per-case checks and input hashes](benchmarks/results/rtx5080_20261031/manifest.json)
identify the exact binaries, libraries, controls and source commit.
The complete local logs, outputs, resource samples and affinity probes are
under `build/rtx5080-benchmark-20261031-210530/`.

<a id="benchmark-kugui-a100"></a>

### Kugui A100 benchmarks (2026-11-02 partial snapshot)

#### Measurement conditions

This is a fixed snapshot captured at **2026-11-02 08:01 JST**, while the
remaining measurements continued. Both fork routes use the frozen executable
built from main commit 0681d63e, with NVHPC 24.7, CUDA 12.5, GCC 12.2,
site Open MPI 4.1.6 and OpenMP ELPA 2026.02.002. The SDK LP64 OpenBLAS
uses OpenMP. CPU ELPA and host BLAS use two threads per MPI rank.
Upstream is official OpenMX 4.0 + patch 4.0.1, built with Intel oneAPI
2025.3 icx/ifx, Intel MPI 2021.17 and threaded MKL 2025.3.

All routes run on A100-SXM4-40GB GPU nodes with a 64-core AMD EPYC 7763.
**32 MPI ranks × 2 OpenMP threads per node** use 64 distinct CPU cores.
S/L and the two named inputs use one node; L2/L3 request two nodes,
64 MPI ranks total. The fork GPU route uses gpusolver and private NVIDIA
MPS, initially exposing two GPUs per node. Four GPUs per node are reserved
for capacity retries. The CPU routes use elpa2 with CUDA devices hidden.
GPU timings include host phases and any automatically selected fallbacks;
backend messages are retained with the per-run observations.

The 59 inputs and 47 PAO/VPS files are frozen. Coordinates, basis, cutoffs,
k points, mixing, SCF/MD limits and convergence criteria are preserved.
Each native suite case starts a **separate MPI invocation**. Times below
are maximum-rank native Total Computational Time, including application
initialization. Suite totals sum those independent case times; they are
not persistent-process whole-suite measurements. External launcher wall
times and nested phase times are retained in the data. Each is one
measurement, without statistical uncertainty estimates.

#### Overall results

At the cutoff, fork CPU completed 30 cases and GPU completed 29 cases;
one GPU case was running. No fork run failed at the cutoff. The remaining
two-node allocations were queued. Upstream completed all 16 L cases and
cghelgeo on November 2; its 14 S cases and sidia333 are retained
input-matched measurements from October 29 using the same upstream binary.
Running/pending calculations have no published final time. Incomplete
suite totals and speed ratios are omitted.

| Test | Nodes | Completed cases (upstream / CPU / GPU) | Upstream CPU (s) | Fork CPU (s) | Fork GPU (s) | Upstream / CPU | Upstream / GPU | CPU / GPU |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| runtest | 1 | 14/14 / 14/14 / 14/14 | 69.381 | 85.662 | 197.657 | 0.810 | 0.351 | 0.433 |
| runtestL | 1 | 16/16 / 15/16 / 14/16 | 2516.886 | — | — | — | — | — |
| runtestL2 | 2 | 0/7 / 0/7 / 0/7 | — | — | — | — | — | — |
| runtestL3 | 2 | 0/20 / 0/20 / 0/20 | — | — | — | — | — | — |
| sidia333_nc_cluster | 1 | 1/1 / 1/1 / 1/1 | 322.622 | 232.044 | 83.834 | 1.390 | 3.848 | 2.768 |
| cghelgeo | 1 | 1/1 / 0/1 / 0/1 | 1141.649 | — | — | — | — | — |

#### Case results

##### runtest (14 cases)

| Input | Upstream CPU (s) | Fork CPU (s) | Fork GPU (s) | CPU / GPU | SCF steps (upstream / CPU / GPU) | GPU vs CPU |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| Benzene | 5.631 | 5.126 | 12.092 | 0.424 | 19 / 19 / 19 | PASS |
| C60 | 5.313 | 7.063 | 16.671 | 0.424 | 27 / 27 / 27 | PASS |
| CO | 5.337 | 6.841 | 31.256 | 0.219 | 16 / 16 / 16 | PASS |
| Cr2 | 5.913 | 6.225 | 12.062 | 0.516 | 21 / 21 / 21 | PASS |
| Crys-MnO | 7.009 | 7.773 | 15.063 | 0.516 | 37 / 39 / 37 | PASS |
| GaAs | 11.268 | 10.261 | 15.499 | 0.662 | 13 / 13 / 13 | PASS |
| Glycine | 3.468 | 5.265 | 12.525 | 0.420 | 20 / 20 / 20 | PASS |
| Graphite4 | 2.826 | 4.231 | 11.081 | 0.382 | 14 / 14 / 14 | PASS |
| H2O-EF | 3.006 | 4.763 | 11.432 | 0.417 | 17 / 17 / 17 | PASS |
| H2O | 3.297 | 4.346 | 10.838 | 0.401 | 31 / 31 / 31 | PASS |
| HMn | 6.568 | 8.434 | 14.642 | 0.576 | 26 / 26 / 26 | PASS |
| Methane | 2.327 | 4.325 | 11.098 | 0.390 | 14 / 14 / 14 | PASS |
| Mol_MnO | 4.730 | 6.359 | 13.340 | 0.477 | 26 / 26 / 26 | PASS |
| Ndia2 | 2.688 | 4.650 | 10.058 | 0.462 | 13 / 13 / 13 | PASS |

##### runtestL (16 cases)

| Input | Upstream CPU (s) | Fork CPU (s) | Fork GPU (s) | CPU / GPU | SCF steps (upstream / CPU / GPU) | GPU vs CPU |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| 5_5_13COb2 | 34.932 | 37.839 | 39.337 | 0.962 | 66 / 66 / 66 | PASS |
| B2C62_Band | 265.212 | 215.047 | 195.847 | 1.098 | 203 / 203 / 204 | PASS |
| CG15c-DC-LNO | 81.915 | 62.343 | 97.070 | 0.642 | 43 / 42 / 42 | PASS |
| DIA512-1 | 53.377 | 48.664 | 90.672 | 0.537 | 29 / 29 / 29 | PASS |
| FeBCC | 64.699 | 63.506 | 73.200 | 0.868 | 61 / 62 / 62 | PASS |
| GEL | 24.408 | 24.277 | 29.248 | 0.830 | 58 / 58 / 58 | PASS |
| GFRAG | 17.689 | 19.794 | 28.258 | 0.700 | 60 / 60 / 60 | PASS |
| GGFF | 528.454 | 454.949 | 299.271 | 1.520 | 98 / 97 / 101 | PASS |
| MCCN | 96.975 | 100.896 | 149.114 | 0.677 | 67 / 67 / 67 | PASS |
| Mn12_148_F | 43.555 | 44.908 | 37.216 | 1.207 | 61 / 61 / 61 | PASS |
| N1C999 | 1082.719 | 542.644 | 725.564 | 0.748 | 63 / 58 / 59 | PASS |
| Ni63-O64 | 37.627 | 37.713 | 36.579 | 1.031 | 66 / 64 / 62 | PASS |
| Pt63 | 26.590 | 34.778 | 34.822 | 0.999 | 72 / 85 / 70 | PASS |
| SialicAcid | 9.428 | 11.679 | 17.280 | 0.676 | 29 / 29 / 29 | PASS |
| ZrB2_2x2 | 103.297 | 88.590 | — | — | 49 / 49 / — | — |
| nsV4Bz5 | 46.009 | — | — | — | 57 / — / — | — |

##### Named inputs

| Input | Upstream CPU (s) | Fork CPU (s) | Fork GPU (s) | CPU / GPU | SCF steps (upstream / CPU / GPU) | GPU vs CPU |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| sidia333_nc_cluster | 322.622 | 232.044 | 83.834 | 2.768 | 25 / 25 / 25 | PASS |
| cghelgeo | 1141.649 | — | — | — | 53 / — / — | — |

L2 (7 cases) and L3 (20 cases) have no completed measurement in this snapshot.

#### Accuracy and convergence

29/29 completed GPU/CPU comparisons pass the energy, force,
atom, coordinate, input and grid checks. Acceptance limits are absolute
energy 1e-7 Ha, maximum force-component difference 1e-7 Ha/bohr and
maximum printed-coordinate difference 1e-6 Å. GPU/CPU maxima are
2.372e-09 Ha and 3.865e-08 Ha/bohr.

All three completed route pairs are also recorded in accuracy.csv;
these checks do not include comparisons against bundled reference outputs.
For upstream comparisons, 12 coordinate checks differ by one last printed
unit (1e-5 Å) between the Intel and NVHPC output. Both outputs print
coordinates to five decimal places; the common input hashes match.
Those comparisons explicitly allow 1.000001e-5 Å for printed-coordinate
rounding. Their strict 1e-6 result and raw coordinate differences remain
in accuracy.csv. Energy and force tolerances remain 1e-7 in every pair.

cpu/upstream: 30/30 pass.
gpu/upstream: 29/29 pass.

Sidia333 reaches its original 25-SCF limit and remains unconverged in all
three routes. Normal termination and SCF convergence are separate fields.
SCF counts differ in several other cases; full-run ratios include those
differences. CG15c-DC-LNO and N1C999 complete on all three routes, and
their fork GPU/CPU comparisons pass.

#### Measurement records

[Snapshot manifest](benchmarks/results/kugui_a100_20261102/manifest.json),
[per-case timings](benchmarks/results/kugui_a100_20261102/case-results.csv),
[per-run records](benchmarks/results/kugui_a100_20261102/run-results.csv),
[numerical comparisons](benchmarks/results/kugui_a100_20261102/accuracy.csv)
and [execution audits](benchmarks/results/kugui_a100_20261102/execution-audit.json)
retain binary/input/output hashes, source identities, SCF counts, affinity
and MPS checks. Full numerical observations and phase times are in runs.json.
The raw fork campaign is work/codex_cpu_gpu_bugfix_20261031/;
the separate upstream campaign is work/codex_upstream_benchmark_20261031/.

<a id="benchmark-pegasus-h100"></a>

### Pegasus H100 benchmarks (2026-10-31–2026-11-01)

#### Measurement conditions

All six tests below were rerun from commit `0681d63e` with the same freshly
built and installed CUDA Release executable for both fork routes. Each node
had one Intel Xeon Platinum 8468, 48 allocated physical cores (SMT disabled),
about 124.5 GiB host memory and one H100 PCIe 80 GB. Every node used
**24 MPI ranks × 2 OpenMP threads**, bound to two cores per rank. `L2` and
`L3` used four nodes (96 ranks); the other tests used one node. Every CUDA
run started a private NVIDIA MPS server on every allocated node.

The build used NVHPC 26.5, CUDA 13.2 (`sm_90`), SDK HPC-X 2.50 / Open MPI,
ELPA 2026.02.002 with OpenMP enabled, and the SDK's threaded LP64 OpenBLAS.
Application OpenMP was also enabled. OpenMP and all BLAS thread controls
were explicitly exported as 2 to every MPI rank. CPU runs selected
`OPENMX_GPU=0` and `scf.eigen.lib elpa2`; CUDA runs selected `OPENMX_GPU=1`
and the original `gpusolver` setting, including its capacity fallbacks.
Placement, thread controls, MPS connections and file hashes passed all
690 checks in the [execution audit](benchmarks/results/pegasus_h100_20261031/execution-audit.json).

Each suite ran once. Each named input ran three times per route, with CPU
and CUDA sharing the same allocation sequentially within each repeat; the
second repeat reversed their order. Original geometries, grids, SCF/MD
limits and stopping criteria were preserved. Only deployment paths and
CPU eigensolver selection changed, as recorded in the
[input audit](benchmarks/results/pegasus_h100_20261031/input-audit.json).

The upstream CPU column uses the retained **October 29, 2026** measurements
of OpenMX 4.0 + official patch 4.0.1, built with Intel icx/ifx 2025.3.2,
Intel MPI 2021.17, threaded MKL 2025.3 and OpenMP ELPA 2018.05.001.
Those runs used the same 24 MPI × 2 OpenMP layout per node and the same
node counts as the latest fork runs. Input definitions and all 47 used PAO/VPS files match. The retained output hashes,
build and measurement dates are recorded in the
[upstream baseline](benchmarks/results/pegasus_h100_20261031/upstream-baseline.json).
Upstream `L2` has no completed baseline measurement, so its cells are marked
with a dash.

#### Overall results

Suite totals sum per-case output `Elapsed.Time`; named inputs use the
median of three such times per route. Launcher and MPS preparation time is
excluded. The upstream baseline uses the same aggregation.

| Test | Nodes | Upstream CPU (s) | Fork CPU (s) | Fork GPU (s) | Upstream / Fork CPU | Upstream / GPU | Fork CPU / GPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `-runtest` (14 cases) | 1 | 75.294 | 43.972 | 73.516 | 1.712 | 1.024 | 0.598 |
| `-runtestL` (16 cases) | 1 | 2847.609 | 1952.185 | 2049.231 | 1.459 | 1.390 | 0.953 |
| `-runtestL2` (7 cases) | 4 | — | 16325.286 | 8392.880 | — | — | 1.945 |
| `-runtestL3` (20 cases) | 4 | 1400.949 | 930.853 | 663.907 | 1.505 | 2.110 | 1.402 |
| `sidia333_nc_cluster.dat` | 1 | 315.049 | 228.009 | 72.137 | 1.382 | 4.367 | 3.161 |
| `cghelgeo.dat` | 1 | 1102.810 | 1304.279 | 243.684 | 0.846 | 4.526 | 5.352 |

#### Case results

##### `-runtest` (14 cases)

| Input | Upstream CPU (s) | Fork CPU (s) | Fork GPU (s) | Upstream / Fork CPU | Upstream / GPU | Fork CPU / GPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Benzene | 6.284 | 2.836 | 5.445 | 2.216 | 1.154 | 0.521 |
| C60 | 6.160 | 4.419 | 8.715 | 1.394 | 0.707 | 0.507 |
| CO | 4.908 | 3.158 | 19.889 | 1.554 | 0.247 | 0.159 |
| Cr2 | 7.081 | 4.261 | 4.704 | 1.662 | 1.505 | 0.906 |
| Crys-MnO | 7.400 | 5.119 | 5.580 | 1.446 | 1.326 | 0.917 |
| GaAs | 12.480 | 7.346 | 6.464 | 1.699 | 1.931 | 1.136 |
| Glycine | 3.196 | 1.911 | 2.961 | 1.672 | 1.079 | 0.645 |
| Graphite4 | 2.661 | 0.789 | 1.514 | 3.373 | 1.758 | 0.521 |
| H2O | 2.885 | 1.175 | 1.509 | 2.455 | 1.912 | 0.779 |
| H2O-EF | 3.151 | 1.648 | 2.491 | 1.912 | 1.265 | 0.662 |
| HMn | 8.221 | 5.937 | 7.056 | 1.385 | 1.165 | 0.841 |
| Methane | 2.432 | 1.076 | 1.935 | 2.260 | 1.257 | 0.556 |
| Mol_MnO | 5.397 | 3.197 | 4.423 | 1.688 | 1.220 | 0.723 |
| Ndia2 | 3.038 | 1.100 | 0.830 | 2.762 | 3.660 | 1.325 |
| **Total** | **75.294** | **43.972** | **73.516** | **1.712** | **1.024** | **0.598** |

<details>
<summary>SCF counts and numerical checks: -runtest (14 cases)</summary>

| Input | SCF steps (upstream / fork CPU / GPU) | Max abs ΔE (Ha) | Max abs ΔF (Ha/bohr) | Check |
| --- | --- | ---: | ---: | --- |
| Benzene | 19 / 19 / 19 | 9.948e-13 | 0.000e+00 | PASS |
| C60 | 27 / 27 / 27 | 3.098e-11 | 1.400e-11 | PASS |
| CO | 16 / 16 / 16 | 0.000e+00 | 1.300e-11 | PASS |
| Cr2 | 21 / 21 / 21 | 4.007e-12 | 2.170e-10 | PASS |
| Crys-MnO | 38 / 44 / 37 | 4.730e-10 | 6.400e-10 | PASS |
| GaAs | 13 / 13 / 13 | 2.100e-11 | 8.000e-12 | PASS |
| Glycine | 20 / 20 / 20 | 9.948e-13 | 1.000e-12 | PASS |
| Graphite4 | 14 / 14 / 14 | 2.000e-12 | 1.000e-12 | PASS |
| H2O | 31 / 31 / 31 | 1.002e-12 | 1.000e-12 | PASS |
| H2O-EF | 17 / 17 / 17 | 9.983e-13 | 0.000e+00 | PASS |
| HMn | 26 / 26 / 26 | 9.948e-13 | 0.000e+00 | PASS |
| Methane | 14 / 14 / 14 | 5.700e-11 | 6.000e-12 | PASS |
| Mol_MnO | 26 / 26 / 26 | 9.948e-13 | 1.000e-12 | PASS |
| Ndia2 | 13 / 13 / 13 | 4.000e-12 | 0.000e+00 | PASS |

</details>

##### `-runtestL` (16 cases)

| Input | Upstream CPU (s) | Fork CPU (s) | Fork GPU (s) | Upstream / Fork CPU | Upstream / GPU | Fork CPU / GPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 5_5_13COb2 | 40.022 | 34.512 | 29.411 | 1.160 | 1.361 | 1.173 |
| B2C62_Band | 311.743 | 205.887 | 182.550 | 1.514 | 1.708 | 1.128 |
| CG15c-DC-LNO | 91.053 | 57.680 | 123.308 | 1.579 | 0.738 | 0.468 |
| DIA512-1 | 64.126 | 44.928 | 116.060 | 1.427 | 0.553 | 0.387 |
| FeBCC | 96.602 | 76.955 | 82.373 | 1.255 | 1.173 | 0.934 |
| GEL | 26.502 | 18.293 | 18.266 | 1.449 | 1.451 | 1.001 |
| GFRAG | 17.599 | 16.708 | 14.789 | 1.053 | 1.190 | 1.130 |
| GGFF | 597.858 | 557.554 | 276.305 | 1.072 | 2.164 | 2.018 |
| MCCN | 103.045 | 103.568 | 201.110 | 0.995 | 0.512 | 0.515 |
| Mn12_148_F | 48.396 | 39.704 | 24.848 | 1.219 | 1.948 | 1.598 |
| N1C999 | 1187.484 | 596.837 | 842.069 | 1.990 | 1.410 | 0.709 |
| Ni63-O64 | 48.381 | 37.784 | 25.131 | 1.280 | 1.925 | 1.503 |
| Pt63 | 34.478 | 27.645 | 21.564 | 1.247 | 1.599 | 1.282 |
| SialicAcid | 11.082 | 8.271 | 9.066 | 1.340 | 1.222 | 0.912 |
| ZrB2_2x2 | 120.799 | 84.833 | 52.831 | 1.424 | 2.287 | 1.606 |
| nsV4Bz5 | 48.439 | 41.026 | 29.550 | 1.181 | 1.639 | 1.388 |
| **Total** | **2847.609** | **1952.185** | **2049.231** | **1.459** | **1.390** | **0.953** |

<details>
<summary>SCF counts and numerical checks: -runtestL (16 cases)</summary>

| Input | SCF steps (upstream / fork CPU / GPU) | Max abs ΔE (Ha) | Max abs ΔF (Ha/bohr) | Check |
| --- | --- | ---: | ---: | --- |
| 5_5_13COb2 | 66 / 67 / 67 | 5.730e-10 | 2.220e-10 | PASS |
| B2C62_Band | 203 / 203 / 204 | 8.902e-11 | 3.920e-08 | PASS |
| CG15c-DC-LNO | 49 / 42 / 42 | 1.264e-09 | 1.560e-10 | PASS |
| DIA512-1 | 29 / 29 / 29 | 7.531e-10 | 5.812e-09 | PASS |
| FeBCC | 62 / 62 / 62 | 1.115e-09 | 6.700e-11 | PASS |
| GEL | 58 / 58 / 58 | 2.390e-10 | 2.530e-10 | PASS |
| GFRAG | 60 / 60 / 60 | 5.770e-10 | 3.673e-09 | PASS |
| GGFF | 101 / 99 / 101 | 9.800e-11 | 1.370e-10 | PASS |
| MCCN | 67 / 67 / 67 | 3.638e-10 | 3.700e-11 | PASS |
| Mn12_148_F | 61 / 61 / 61 | 2.769e-10 | 4.560e-10 | PASS |
| N1C999 | 66 / 59 / 58 | 1.553e-09 | 3.000e-11 | PASS |
| Ni63-O64 | 68 / 69 / 62 | 2.748e-09 | 8.480e-10 | PASS |
| Pt63 | 81 / 70 / 66 | 4.847e-09 | 3.230e-10 | PASS |
| SialicAcid | 29 / 29 / 29 | 3.100e-10 | 3.680e-10 | PASS |
| ZrB2_2x2 | 49 / 49 / 49 | 2.060e-10 | 3.290e-10 | PASS |
| nsV4Bz5 | 57 / 57 / 57 | 2.070e-10 | 8.620e-10 | PASS |

</details>

##### `-runtestL2` (7 cases)

| Input | Upstream CPU (s) | Fork CPU (s) | Fork GPU (s) | Upstream / Fork CPU | Upstream / GPU | Fork CPU / GPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C1000 | — | 583.088 | 530.061 | — | — | 1.100 |
| Fe1000 | — | 10391.753 | 3195.273 | — | — | 3.252 |
| GRA1024 | — | 791.450 | 369.214 | — | — | 2.144 |
| Ih-Ice1200 | — | 247.206 | 217.339 | — | — | 1.137 |
| Pt500 | — | 2955.237 | 3097.813 | — | — | 0.954 |
| R-TiO2-1050 | — | 767.323 | 542.767 | — | — | 1.414 |
| Si1000 | — | 589.229 | 440.413 | — | — | 1.338 |
| **Total** | — | **16325.286** | **8392.880** | — | — | **1.945** |

<details>
<summary>SCF counts and numerical checks: -runtestL2 (7 cases)</summary>

| Input | SCF steps (upstream / fork CPU / GPU) | Max abs ΔE (Ha) | Max abs ΔF (Ha/bohr) | Check |
| --- | --- | ---: | ---: | --- |
| C1000 | — / 41 / 41 | 7.512e-10 | 6.050e-10 | PASS |
| Fe1000 | — / 436 / 342 | 3.783e-09 | 5.894e-09 | PASS |
| GRA1024 | — / 60 / 48 | 3.031e-09 | 8.850e-10 | PASS |
| Ih-Ice1200 | — / 36 / 35 | 7.857e-09 | 3.840e-10 | PASS |
| Pt500 | — / 204 / 181 | 1.465e-08 | 9.702e-08 | PASS |
| R-TiO2-1050 | — / 34 / 39 | 4.056e-09 | 8.800e-10 | PASS |
| Si1000 | — / 42 / 41 | 2.692e-10 | 1.170e-10 | PASS |

</details>

##### `-runtestL3` (20 cases)

| Input | Upstream CPU (s) | Fork CPU (s) | Fork GPU (s) | Upstream / Fork CPU | Upstream / GPU | Fork CPU / GPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 5_5_13COb2 | 7.517 | 3.385 | 8.136 | 2.221 | 0.924 | 0.416 |
| C1000 | 75.529 | 63.118 | 55.543 | 1.197 | 1.360 | 1.136 |
| C60 | 2.422 | 1.058 | 1.885 | 2.289 | 1.285 | 0.561 |
| CG15c | 43.160 | 22.137 | 24.001 | 1.950 | 1.798 | 0.922 |
| Crys-MnO | 5.273 | 1.363 | 1.396 | 3.869 | 3.777 | 0.976 |
| DIA512-1 | 12.696 | 6.815 | 9.222 | 1.863 | 1.377 | 0.739 |
| Fe1000 | 103.523 | 99.759 | 48.726 | 1.038 | 2.125 | 2.047 |
| GEL | 8.111 | 3.180 | 3.713 | 2.551 | 2.184 | 0.856 |
| GFRAG | 3.418 | 1.492 | 2.130 | 2.291 | 1.605 | 0.700 |
| GGFF | 72.409 | 64.476 | 68.840 | 1.123 | 1.052 | 0.937 |
| MCCN | 40.405 | 19.630 | 25.872 | 2.058 | 1.562 | 0.759 |
| Mn12_148_F | 7.765 | 3.713 | 3.419 | 2.091 | 2.271 | 1.086 |
| N1C999 | 402.708 | 236.021 | 114.270 | 1.706 | 3.524 | 2.065 |
| Ni63-O64 | 7.812 | 3.765 | 4.430 | 2.075 | 1.763 | 0.850 |
| Pt500 | 142.883 | 101.872 | 110.445 | 1.403 | 1.294 | 0.922 |
| Pt63 | 5.064 | 2.539 | 3.315 | 1.994 | 1.528 | 0.766 |
| Si1280-LNO | 431.747 | 282.175 | 164.167 | 1.530 | 2.630 | 1.719 |
| SialicAcid | 3.855 | 1.588 | 2.466 | 2.428 | 1.563 | 0.644 |
| ZrB2_2x2 | 17.233 | 9.076 | 8.180 | 1.899 | 2.107 | 1.110 |
| nsV4Bz5 | 7.419 | 3.691 | 3.751 | 2.010 | 1.978 | 0.984 |
| **Total** | **1400.949** | **930.853** | **663.907** | **1.505** | **2.110** | **1.402** |

<details>
<summary>SCF counts and numerical checks: -runtestL3 (20 cases)</summary>

| Input | SCF steps (upstream / fork CPU / GPU) | Max abs ΔE (Ha) | Max abs ΔF (Ha/bohr) | Check |
| --- | --- | ---: | ---: | --- |
| 5_5_13COb2 | 3 / 3 / 3 | 1.258e-08 | 5.860e-10 | PASS |
| C1000 | 3 / 3 / 3 | 4.305e-09 | 1.205e-09 | PASS |
| C60 | 3 / 3 / 3 | 3.098e-11 | 1.700e-11 | PASS |
| CG15c | 3 / 3 / 3 | 4.326e-09 | 1.464e-09 | PASS |
| Crys-MnO | 3 / 3 / 3 | 4.640e-10 | 5.350e-10 | PASS |
| DIA512-1 | 3 / 3 / 3 | 7.208e-10 | 8.486e-09 | PASS |
| Fe1000 | 3 / 3 / 3 | 6.301e-09 | 3.021e-07 | FAIL |
| GEL | 3 / 3 / 3 | 2.350e-10 | 1.973e-09 | PASS |
| GFRAG | 3 / 3 / 3 | 3.791e-09 | 2.821e-09 | PASS |
| GGFF | 3 / 3 / 3 | 1.000e-10 | 9.500e-11 | PASS |
| MCCN | 3 / 3 / 3 | 7.055e-09 | 1.448e-09 | PASS |
| Mn12_148_F | 3 / 3 / 3 | 5.443e-10 | 2.147e-09 | PASS |
| N1C999 | 3 / 3 / 3 | 1.586e-09 | 1.790e-10 | PASS |
| Ni63-O64 | 3 / 3 / 3 | 2.725e-09 | 1.135e-05 | FAIL |
| Pt500 | 3 / 3 / 3 | 1.111e-08 | 1.302e-09 | PASS |
| Pt63 | 3 / 3 / 3 | 5.249e-09 | 7.090e-10 | PASS |
| Si1280-LNO | 3 / 3 / 3 | 2.099e-08 | 2.640e-10 | PASS |
| SialicAcid | 3 / 3 / 3 | 3.020e-10 | 3.910e-10 | PASS |
| ZrB2_2x2 | 3 / 3 / 3 | 2.958e-09 | 1.088e-09 | PASS |
| nsV4Bz5 | 3 / 3 / 3 | 8.090e-10 | 2.851e-09 | PASS |

</details>

##### Named inputs

| Input | Upstream CPU (s) | Fork CPU (s) | Fork GPU (s) | Upstream / Fork CPU | Upstream / GPU | Fork CPU / GPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `sidia333_nc_cluster.dat` | 315.049 | 228.009 | 72.137 | 1.382 | 4.367 | 3.161 |
| `cghelgeo.dat` | 1102.810 | 1304.279 | 243.684 | 0.846 | 4.526 | 5.352 |

<details>
<summary>SCF counts and numerical checks: Named inputs</summary>

| Input | SCF steps (upstream / fork CPU / GPU) | Max abs ΔE (Ha) | Max abs ΔF (Ha/bohr) | Check |
| --- | --- | ---: | ---: | --- |
| `sidia333_nc_cluster.dat` | 25 × 3 / 25 × 3 / 25 × 3 | 2.210e-10 | 2.180e-10 | PASS |
| `cghelgeo.dat` | 53, 59, 61 / 59, 56, 59 / 60, 53, 60 | 3.191e-07 | 6.983e-08 | FAIL |

</details>

#### Accuracy and convergence

These ratios include differences in SCF iteration counts. For example,
`L2/Fe1000` used 436 CPU steps and 342 CUDA steps. `cghelgeo` used
59/56/59 steps on CPU and 60/53/60 on CUDA; its input limit remained 200.
Sidia333 reached its original 25-step cap in all six runs, and the `L3`
inputs use three SCF steps. Assess convergence from the recorded final
residuals alongside these timings.

Per-case maxima above combine paired CPU/GPU, repeat, bundled-reference and
retained-upstream checks. The comparison sets are summarized separately below.

All 20 CPU/CUDA runs completed normally, producing 126 case observations.
Paired CPU/CUDA checks compare total energy and every force component,
with tolerances of `1e-7` Ha and `1e-7` Ha/bohr. Of 63 paired comparisons,
59 were within tolerance. The four exceptions were:

| Test / input | Repeat | abs ΔE (Ha) | Max abs ΔF (Ha/bohr) |
| --- | ---: | ---: | ---: |
| L3 / Fe1000 | 1 | 5.04951e-09 | 2.33676e-07 |
| L3 / Ni63-O64 | 1 | 2.30648e-09 | 3.66447e-06 |
| cghelgeo | 2 | 2.23568e-07 | 6.98300e-08 |
| cghelgeo | 3 | 1.03287e-07 | 4.41100e-09 |

All seven `L2` paired comparisons were within tolerance. Its `Pt500`
maximum force-component difference was `9.7019e-8` Ha/bohr in this run.
Bundled-reference force checks also exceeded tolerance for `L3/Fe1000`
(CUDA) and `L3/Ni63-O64` (both CPU and CUDA). CUDA `cghelgeo` repeat 2
exceeded the energy tolerance against CUDA repeat 1. All 185 paired,
repeat and bundled-reference comparisons are retained separately in
[accuracy.csv](benchmarks/results/pegasus_h100_20261031/accuracy.csv).

The retained upstream outputs were also checked against the latest fork
outputs at the same energy/force tolerances, matching case and repeat
index. Of 56 comparisons per route, 54 CPU and 52 CUDA comparisons were
within tolerance. The following six exceptions are separate from the
CPU/CUDA comparisons above.

<details>
<summary>Numerical differences against the retained upstream CPU baseline</summary>

| Test / input | Latest fork route | Repeat | abs ΔE (Ha) | Max abs ΔF (Ha/bohr) |
| --- | --- | ---: | ---: | ---: |
| L3 / Fe1000 | CUDA | 1 | 4.80213e-09 | 3.02135e-07 |
| L3 / Ni63-O64 | CPU | 1 | 5.58430e-10 | 8.70978e-06 |
| L3 / Ni63-O64 | CUDA | 1 | 1.74805e-09 | 5.07938e-06 |
| cghelgeo | CPU | 1 | 2.34339e-07 | 6.85210e-08 |
| cghelgeo | CUDA | 1 | 3.19112e-07 | 6.56320e-08 |
| cghelgeo | CUDA | 2 | 2.88112e-07 | 6.39600e-08 |

All 112 comparisons, including input/output hashes, are recorded in
[upstream-accuracy.csv](benchmarks/results/pegasus_h100_20261031/upstream-accuracy.csv).

</details>

#### Measurement records

The campaign consumed **32.08 points**, within the 150-point cap. The ten
allocations totaled 32.072778 node-hours, including preparation and cleanup.
The [versioned CSV/JSON records](benchmarks/results/pegasus_h100_20261031/manifest.json)
include per-case times, SCF counts, final residuals, numerical comparisons,
build/input hashes, launch commands and allocation accounting. Full local
outputs, rank records and MPS logs remain under
`work/benchmarks_threaded_20261031/`.

The [overall comparison CSV](benchmarks/results/pegasus_h100_20261031/upstream-comparison.csv)
and [case comparison CSV](benchmarks/results/pegasus_h100_20261031/upstream-case-comparison.csv)
provide the source values for the inline tables.
[Whole-run time divided by SCF count](benchmarks/results/pegasus_h100_20261031/L2-iteration-reference.csv)
is an additional reference that includes initialization and force work.

<a id="benchmark-radeon-rx9060xt"></a>

### Radeon RX 9060 XT validation (2026-11-01 refresh)

#### Measurement conditions

After pulling `main`, the fork was rebuilt and installed from production
revision `c6648c29`. Both CPU and HIP paths were remeasured from
2026-11-01 13:45 to 2026-11-01 17:48 JST.
The October 29 upstream CPU measurement is reused unchanged.
The host was a Ryzen Threadripper 3970X (32 physical cores, SMT disabled),
256 GB RAM, and one Radeon RX 9060 XT with 16 GiB VRAM (`gfx1200`).
Every measurement used **16 MPI ranks × 2 OpenMP threads**, bound to 32 cores.
Jobs ran sequentially in CPU/HIP pairs for S, sidia333, cghelgeo and L.
After the cghelgeo CPU comparison failed, the queue was paused for the
separate controls below, then resumed for the remaining jobs. The failed
timed input was preserved and was not repeated to select a passing result.

| Path | Build and runtime |
| --- | --- |
| Upstream CPU | OpenMX 4.0 + [official patch 4.0.1](https://www.openmx-square.org/bugfixed/26May08/README.txt), Intel icx/ifx 2026.0, Intel MPI 2021.18, threaded LP64 MKL; `scf.eigen.lib elpa2` |
| Fork CPU | Release HIP build, ROCm SDK `core-10.1` Clang/Flang, OpenMPI 5.0.9, AOCL 5.2, OpenMP ELPA 2026.02.002; `OPENMX_GPU=0`, `scf.eigen.lib elpa2` |
| Fork GPU | Same installed executable and Core/SIMD libraries; `OPENMX_GPU=1`, `scf.eigen.lib gpusolver`, `HSA_ENABLE_SDMA=0` |

The pulled MI300A optimization also edits common HIP noncollinear band
control flow and state lifetime, so Radeon was remeasured. MI300A owner
waves, construction caches, cached density waves and batched solver
completion remain architecture-gated off on gfx1200. Radeon retains its
existing k-turn policy, streaming constructor and blocking completion.
Architecture-policy checks, 12,001 owner schedules, cache/density failure
models, band lifecycle checks and the ELPA/GPU-handle regressions passed.
Three new test scripts needed source-path fixes in the CMake-generated
native tree; all three now pass both there and from the source tree.
These test-only edits do not change the measured application code.

Both fork paths use `OMP_NUM_THREADS=2`, `BLIS_NUM_THREADS=2`,
`OMP_DYNAMIC=FALSE`, `OMP_PROC_BIND=close`, `OMP_PLACES=cores` and
`OMP_MAX_ACTIVE_LEVELS=1`, matching the previous fork campaign. All 16 rank
environments and CPU affinities were checked for every job. A separate
8×2 CPU ELPA numerical check observed two ELPA workers on every rank.
The upstream baseline uses two MKL threads per rank; its ratios also
include compiler, MPI and numerical-library differences.

Inputs, geometry, grids, SCF/MD limits, convergence thresholds, stdout
levels and arithmetic policy are unchanged. Only `DATA.PATH` and
`scf.eigen.lib` are set in the measurement copies. All 44 basis and
pseudopotential hashes match the previous record.
`LIBOMPTARGET_HEAP_SIZE=67108864` avoids heap re-execution and
`HSA_OVERRIDE_CPU_AFFINITY_DEBUG=0` retains rank CPU affinity for HSA helpers.
The HIP path retains `HSA_ENABLE_SDMA=0`; NVIDIA MPS does not apply to Radeon.

#### Overall results

Suite totals sum native per-case `runtest.result` / `runtestL.result` times.
Named inputs use the maximum-rank `Total Computational Time`. Each value
is one measurement. MPI launch/shutdown overhead is excluded. GPU-path
times include CPU fallbacks selected by the application.

| Test | Nodes | Upstream CPU (s) | Fork CPU (s) | Fork GPU (s) | Upstream / Fork CPU | Upstream / GPU | Fork CPU / GPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `-runtest` (14 cases) | 1 | 59.020 | 46.890 | 67.470 | 1.259 | 0.875 | 0.695 |
| `-runtestL` (16 cases) | 1 | 4319.340 | 2839.930 | 3944.630 | 1.521 | 1.095 | 0.720 |
| `-runtestL2` (7 cases) | — | — | — | — | — | — | — |
| `-runtestL3` (20 cases) | — | — | — | — | — | — | — |
| `sidia333_nc_cluster.dat` | 1 | 483.443 | 352.627 | 193.664 | 1.371 | 2.496 | 1.821 |
| `cghelgeo.dat` | 1 | 1896.871 | 2483.295 | 2587.862 | 0.764 | 0.733 | 0.960 |

#### Case results

##### `-runtest` (14 cases)

| Input | Upstream CPU (s) | Fork CPU (s) | Fork GPU (s) | Upstream / Fork CPU | Upstream / GPU | Fork CPU / GPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Benzene | 2.390 | 1.790 | 3.720 | 1.335 | 0.642 | 0.481 |
| C60 | 4.970 | 5.710 | 8.720 | 0.870 | 0.570 | 0.655 |
| CO | 5.990 | 3.690 | 16.790 | 1.623 | 0.357 | 0.220 |
| Cr2 | 4.070 | 3.190 | 3.140 | 1.276 | 1.296 | 1.016 |
| Crys-MnO | 6.570 | 5.140 | 5.350 | 1.278 | 1.228 | 0.961 |
| GaAs | 11.560 | 9.410 | 8.540 | 1.228 | 1.354 | 1.102 |
| Glycine | 2.350 | 1.760 | 3.620 | 1.335 | 0.649 | 0.486 |
| Graphite4 | 1.620 | 1.070 | 1.540 | 1.514 | 1.052 | 0.695 |
| H2O | 1.880 | 1.080 | 1.700 | 1.741 | 1.106 | 0.635 |
| H2O-EF | 2.150 | 1.540 | 2.110 | 1.396 | 1.019 | 0.730 |
| HMn | 7.660 | 6.930 | 5.830 | 1.105 | 1.314 | 1.189 |
| Methane | 1.460 | 0.930 | 1.680 | 1.570 | 0.869 | 0.554 |
| Mol_MnO | 4.090 | 3.320 | 3.740 | 1.232 | 1.094 | 0.888 |
| Ndia2 | 2.260 | 1.330 | 0.990 | 1.699 | 2.283 | 1.343 |
| **Total** | **59.020** | **46.890** | **67.470** | **1.259** | **0.875** | **0.695** |

<details>
<summary>SCF counts and numerical checks: -runtest (14 cases)</summary>

| Input | SCF steps (upstream / fork CPU / GPU) | Max abs ΔE (Ha) | Max abs ΔF (Ha/bohr) | Check |
| --- | --- | ---: | ---: | --- |
| Benzene | 19 / 19 / 19 | 9.948e-13 | 0.000e+00 | PASS |
| C60 | 27 / 27 / 27 | 3.200e-11 | 1.000e-12 | PASS |
| CO | 16 / 16 / 16 | 0.000e+00 | 2.200e-11 | PASS |
| Cr2 | 21 / 21 / 21 | 2.984e-12 | 2.170e-10 | PASS |
| Crys-MnO | 40 / 37 / 37 | 4.750e-10 | 6.410e-10 | PASS |
| GaAs | 13 / 13 / 13 | 1.998e-11 | 8.000e-12 | PASS |
| Glycine | 20 / 20 / 20 | 9.948e-13 | 1.000e-12 | PASS |
| Graphite4 | 14 / 14 / 14 | 2.000e-12 | 1.000e-12 | PASS |
| H2O | 31 / 31 / 31 | 1.002e-12 | 1.000e-12 | PASS |
| H2O-EF | 17 / 17 / 17 | 9.983e-13 | 1.000e-12 | PASS |
| HMn | 26 / 26 / 26 | 9.948e-13 | 0.000e+00 | PASS |
| Methane | 14 / 14 / 14 | 5.600e-11 | 6.000e-12 | PASS |
| Mol_MnO | 26 / 26 / 26 | 9.948e-13 | 1.000e-12 | PASS |
| Ndia2 | 13 / 13 / 13 | 4.000e-12 | 1.000e-12 | PASS |

</details>

##### `-runtestL` (16 cases)

| Input | Upstream CPU (s) | Fork CPU (s) | Fork GPU (s) | Upstream / Fork CPU | Upstream / GPU | Fork CPU / GPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 5_5_13COb2 | 63.350 | 58.650 | 64.770 | 1.080 | 0.978 | 0.906 |
| B2C62_Band | 363.230 | 276.170 | 796.290 | 1.315 | 0.456 | 0.347 |
| CG15c-DC-LNO | 137.220 | 93.590 | 113.100 | 1.466 | 1.213 | 0.827 |
| DIA512-1 | 87.110 | 75.080 | 104.790 | 1.160 | 0.831 | 0.716 |
| FeBCC | 96.300 | 90.900 | 105.930 | 1.059 | 0.909 | 0.858 |
| GEL | 27.110 | 22.940 | 103.250 | 1.182 | 0.263 | 0.222 |
| GFRAG | 28.730 | 25.870 | 36.650 | 1.111 | 0.784 | 0.706 |
| GGFF | 952.280 | 784.610 | 1053.170 | 1.214 | 0.904 | 0.745 |
| MCCN | 174.760 | 170.770 | 185.390 | 1.023 | 0.943 | 0.921 |
| Mn12_148_F | 66.390 | 60.710 | 76.790 | 1.094 | 0.865 | 0.791 |
| N1C999 | 1985.770 | 845.210 | 908.460 | 2.349 | 2.186 | 0.930 |
| Ni63-O64 | 58.470 | 54.520 | 66.920 | 1.072 | 0.874 | 0.815 |
| Pt63 | 35.530 | 33.640 | 58.360 | 1.056 | 0.609 | 0.576 |
| SialicAcid | 12.740 | 10.710 | 19.410 | 1.190 | 0.656 | 0.552 |
| ZrB2_2x2 | 155.010 | 148.290 | 174.490 | 1.045 | 0.888 | 0.850 |
| nsV4Bz5 | 75.340 | 88.270 | 76.860 | 0.854 | 0.980 | 1.148 |
| **Total** | **4319.340** | **2839.930** | **3944.630** | **1.521** | **1.095** | **0.720** |

<details>
<summary>SCF counts and numerical checks: -runtestL (16 cases)</summary>

| Input | SCF steps (upstream / fork CPU / GPU) | Max abs ΔE (Ha) | Max abs ΔF (Ha/bohr) | Check |
| --- | --- | ---: | ---: | --- |
| 5_5_13COb2 | 66 / 66 / 66 | 6.260e-10 | 3.200e-10 | PASS |
| B2C62_Band | 204 / 204 / 204 | 1.270e-10 | 3.517e-08 | PASS |
| CG15c-DC-LNO | 44 / 42 / 42 | 1.268e-09 | 1.560e-10 | PASS |
| DIA512-1 | 29 / 29 / 29 | 7.731e-10 | 6.133e-09 | PASS |
| FeBCC | 62 / 62 / 62 | 1.100e-09 | 7.300e-11 | PASS |
| GEL | 58 / 58 / 58 | 2.369e-10 | 4.410e-10 | PASS |
| GFRAG | 60 / 60 / 60 | 5.660e-10 | 1.916e-09 | PASS |
| GGFF | 102 / 98 / 101 | 5.200e-10 | 5.900e-11 | PASS |
| MCCN | 67 / 67 / 67 | 3.620e-10 | 3.700e-11 | PASS |
| Mn12_148_F | 61 / 61 / 61 | 2.510e-10 | 5.570e-10 | PASS |
| N1C999 | 79 / 59 / 59 | 1.545e-09 | 3.100e-11 | PASS |
| Ni63-O64 | 69 / 69 / 62 | 2.907e-09 | 8.010e-10 | PASS |
| Pt63 | 58 / 59 / 69 | 1.701e-08 | 4.596e-09 | PASS |
| SialicAcid | 29 / 29 / 29 | 3.120e-10 | 4.190e-10 | PASS |
| ZrB2_2x2 | 49 / 49 / 49 | 8.708e-11 | 5.120e-10 | PASS |
| nsV4Bz5 | 57 / 57 / 57 | 9.999e-11 | 1.161e-09 | PASS |

</details>

##### Named inputs

| Input | Upstream CPU (s) | Fork CPU (s) | Fork GPU (s) | Upstream / Fork CPU | Upstream / GPU | Fork CPU / GPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `cghelgeo.dat` | 1896.871 | 2483.295 | 2587.862 | 0.764 | 0.733 | 0.960 |
| `sidia333_nc_cluster.dat` | 483.443 | 352.627 | 193.664 | 1.371 | 2.496 | 1.821 |

<details>
<summary>SCF counts and numerical checks: Named inputs</summary>

| Input | SCF steps (upstream / fork CPU / GPU) | Max abs ΔE (Ha) | Max abs ΔF (Ha/bohr) | Check |
| --- | --- | ---: | ---: | --- |
| `cghelgeo.dat` | 60 / 59 / 61 | 2.615e-07 | 4.598e-08 | FAIL |
| `sidia333_nc_cluster.dat` | 25 / 25 / 25 | 2.000e-10 | 1.450e-10 | PASS |

</details>

#### Accuracy and convergence

All **64 new case executions** completed with finite total energies and
all 27,552 force components and matching geometry/grids.
The unchanged strict comparison requires absolute energy differences ≤
`1e-7` Ha and maximum force-component differences ≤ `1e-7` Ha/bohr.
**62/64 executions pass all applicable comparisons; 2 fail.**
Against the retained upstream alone, 63/64 outputs pass
(CPU 31/32; HIP 32/32).
The failed original measurements remain in the tables and records.
Per-case maxima cover both fork paths against bundled references, each
other, retained upstream CPU and the previous fork output where applicable.
Completion markers alone never determine acceptance. SCF counts sum all
MD steps; ratios compare whole inputs rather than fixed iteration counts.

The sidia333 input retains its 25-SCF limit; these timings do not establish
SCF convergence. All reported benchmark times are single measurements.

The original cghelgeo CPU run failed the strict total-energy comparison.

- `cghelgeo` / `fork-cpu`: previous-fork-cpu: abs ΔE=2.136e-07 Ha, max abs ΔF=3.911e-08 Ha/bohr, upstream-intel-previous: abs ΔE=2.214e-07 Ha, max abs ΔF=3.850e-08 Ha/bohr.
- `cghelgeo` / `fork-gpu`: fork-cpu: abs ΔE=2.615e-07 Ha, max abs ΔF=4.598e-08 Ha/bohr.

The cghelgeo HIP output passes the upstream and previous-HIP checks; its
strict execution flag fails on the comparison against the current CPU output.

Identical-restart CPU controls used `scf.restart=on`, `scf.criterion=1e-10`
and at most 20 SCF steps for both the current and previous frozen binaries.
They are separate diagnostics, excluded from every benchmark timing.

| Control | SCF steps | Total energy (Ha) | Final band-energy change (Ha) | Input stop test met |
| --- | ---: | ---: | ---: | --- |
| current | 20 | -4153.148190614701 | 9.955e-09 | False |
| previous | 20 | -4153.148190614704 | 9.953e-09 | False |

Both controls reached their diagnostic iteration cap without meeting
the tighter stop test; they do not establish SCF convergence.

The control pair differs by 2.728e-12 Ha in total energy
and 1.000e-12 Ha/bohr in force components.
Both controls are also within `1e-7` Ha and `1e-7` Ha/bohr of the retained
upstream output (maximum differences: 7.004e-8 Ha and 7.921e-8 Ha/bohr).
These controls do not retroactively make the original strict comparison pass.
The identical-state agreement suggests SCF trajectory/stopping sensitivity
rather than a new arithmetic difference in this path. The SCF stop test
uses a band-energy change and mixing residual; it does not bound the final
total-energy difference against another run.

**The cghelgeo GPU timing includes CPU dense diagonalization.** There were
61 dense-memory rejections across 61 SCF steps. The first verdict was:

```text
Band_DFT_Col: GPU device is shared by 16 rank(s), has 12794.000 MiB free (16304.000 MiB total), but 16004.569 MiB is required for the dense GPU diagonalization (512.000 MiB library reserve per rank); switching to CPU path.
```

The single Radeon could not form the cooperative MAGMA multi-GPU pool
(61 CPU-solve messages); other HIP phases remained enabled.
The pre-admission GPU banner alone does not establish GPU eigensolver use.

`-runtestL2` and `-runtestL3` have no completed three-route measurement.
L3 was not repeated after the earlier host-memory guard stopped that
campaign at the 32 GiB available-memory reserve. No complete L3 timing or
speedup is reported.

#### Measurement records

The [case-level CSV](benchmarks/results/radeon_rx9060xt_20261101_16x2.csv) records every
case timing, SCF count, comparison maximum and input hash. The accompanying
[provenance JSON](benchmarks/results/radeon_rx9060xt_20261101_16x2.json) records the measured
revision, build, frozen executable/library hashes, runtime controls, rank
audits, dataset hashes, source review, test fixes and all eight jobs.
Raw logs are under `work/codex_benchmark_20261101_mi300a_guard/`.
The [previous Radeon record](benchmarks/results/radeon_rx9060xt_20261031_16x2.json)
remains available separately.

<a id="benchmark-sirius-mi300a"></a>

### Sirius MI300A benchmarks (2026-10-31–2026-11-01)

#### Measurement conditions

The S/L suites and the two named inputs use **one physical node with four
MI300A APUs** (96 CPU cores and four GPUs), **32 MPI ranks × 3 OpenMP
threads**. L2/L3 use **two physical nodes and eight APUs**, with **64 MPI
ranks total and 32 ranks × 3 threads per node**. Every rank uses three
disjoint CPU cores. All three configurations used
AMD ROCm 7.2.0 Clang/flang, Open MPI 5.0.10 and AOCL 5.1.0, with the
OpenMP BLIS provider set to three threads per rank.

The upstream CPU executable is official OpenMX 4.0 plus the official
4.0.1 patches, using its OpenMP-enabled ELPA 2018.05.001. The fork CPU
path uses `OPENMX_GPU=0` and `scf.eigen.lib elpa2`; the HIP GPU path uses
`OPENMX_GPU=1` and ordinary `scf.eigen.lib gpusolver`. The fork uses
OpenMP-enabled ELPA 2026.02.002, with three threads for CPU handles.
The S/L/named CPU measurements are reused from 2026-10-31: the fork CPU executable
is frozen at `3e2f5c73`. All six GPU workloads were freshly measured on
2026-11-01 using optimized numerical code `5925e71a` and checkout
`c6648c29`, with the same frozen executable and Core/SIMD libraries as
the earlier optimization validation. The two CPU L3 suites were additionally
measured on two nodes; the eight S/L/named CPU workloads remain reused.
Executable/library/source and all 59 input hashes are in the
[new manifest](benchmarks/results/sirius_mi300a_gpu_rebench_20261101_32x3/manifest.json).
The [earlier optimized measurement](benchmarks/results/sirius_mi300a_ggff_20261101/manifest.json)
and [2026-10-31 measurement](benchmarks/results/sirius_mi300a_20261031_32x3/manifest.json)
remain available for comparison.

The [CPU reuse audit](benchmarks/results/sirius_mi300a_gpu_rebench_20261101_32x3/cpu-reuse-assessment.json)
confirms the corrected CPU ELPA/BLAS teams and unchanged Core/SIMD hashes.
The GPU changes do not change ordinary `elpa2` dense solves. The common
noncollinear band dispatch adds one small MPI reduction per SCF step;
the historical CPU executable is therefore disclosed explicitly, without
claiming byte-identical CPU execution or fresh S/L/named CPU timings.
The fresh fork CPU L3 uses the same optimized executable as the GPU route.

#### Overall results

Suite totals sum native per-case result rows; named inputs use output
`Elapsed.Time`. Each value is one measurement. Launcher wall times are
retained separately. L3 has fresh timings for all three configurations;
L2 has a fresh GPU total and no completed CPU total.

| Test | Nodes | Upstream CPU (s) | Fork CPU (s) | Fork GPU (s) | Upstream / Fork CPU | Upstream / GPU | Fork CPU / GPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `-runtest` (14 cases) | 1 | 61.970 | 43.580 | 126.480 | 1.422 | 0.490 | 0.345 |
| `-runtestL` (16 cases) | 1 | 2219.350 | 1533.410 | 2173.690 | 1.447 | 1.021 | 0.705 |
| `-runtestL2` (7 cases) | 2 | — | — | 20325.750 | — | — | — |
| `-runtestL3` (20 cases) | 2 | 2526.620 | 1707.120 | 1602.770 | 1.480 | 1.576 | 1.065 |
| `sidia333_nc_cluster.dat` | 1 | 247.520 | 192.886 | 120.946 | 1.283 | 2.047 | 1.595 |
| `cghelgeo.dat` | 1 | 860.989 | 792.170 | 362.277 | 1.087 | 2.377 | 2.187 |

#### MI300A optimization

GGFF completed in **268.509 s**, compared with
**1161.923 s** before optimization (**4.33×** faster); the fresh run uses
101 SCF iterations. A separate standalone full run took **304.061 s**,
with energy/force agreement checked against all three previous routes.

The MI300A dense-k path preserves MPI k-point ownership, rank/GPU mapping
and CPU/NUMA binding, while scheduling memory-bounded owners per GPU in
place of the global four-turn ceiling. It reuses construction data, queues
copies, batches eigensolver status/results into one final wait, and builds
cached density panels without per-k world collectives. Failed waves retain
the original collective recovery path. Radeon keeps its existing policies;
Radeon validation here covers gfx1200 compilation and policy models, with
no Radeon hardware timing claim.

| Workload | Previous HIP (s) | Fresh optimized HIP (s) | Speedup |
| --- | ---: | ---: | ---: |
| `-runtest` (14 cases) | 140.310 | 126.480 | 1.109 |
| `-runtestL` (16 cases) | 3032.940 | 2173.690 | 1.395 |
| `sidia333_nc_cluster.dat` | 133.143 | 120.946 | 1.101 |
| `cghelgeo.dat` | 344.113 | 362.277 | 0.950 |

The separate matched S control on node a04 measured old/new/old
**144.42 / 144.76 / 140.03 s**. The historical S slowdown was not reproduced
at the same magnitude. One old-control H2O-EF run had a residual floor and
a numerical deviation; its raw diagnostic results are kept separate from
the primary a05 checks. MCCN controls took **164.762 s** with batched
completion, **169.040 s** without it, and **223.333 s** for the previous
executable (67 SCF each). The historical MCCN time was not reproduced by
that old control. These small samples do not isolate the reason for timing
variation. Full controls and ablations are in the measurement record below.

#### Case results

##### `-runtest` (14 cases)

| Input | Upstream CPU (s) | Fork CPU (s) | Fork GPU (s) | Upstream / Fork CPU | Upstream / GPU | Fork CPU / GPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Benzene | 2.900 | 2.310 | 7.090 | 1.255 | 0.409 | 0.326 |
| C60 | 5.110 | 4.830 | 8.470 | 1.058 | 0.603 | 0.570 |
| CO | 6.140 | 5.190 | 42.600 | 1.183 | 0.144 | 0.122 |
| Cr2 | 4.460 | 3.230 | 5.350 | 1.381 | 0.834 | 0.604 |
| Crys-MnO | 8.170 | 5.110 | 9.020 | 1.599 | 0.906 | 0.567 |
| GaAs | 10.980 | 6.720 | 8.340 | 1.634 | 1.317 | 0.806 |
| Glycine | 2.730 | 2.050 | 5.500 | 1.332 | 0.496 | 0.373 |
| Graphite4 | 2.300 | 1.090 | 3.620 | 2.110 | 0.635 | 0.301 |
| H2O-EF | 2.420 | 1.630 | 3.870 | 1.485 | 0.625 | 0.421 |
| H2O | 2.440 | 1.460 | 3.830 | 1.671 | 0.637 | 0.381 |
| HMn | 5.810 | 4.450 | 14.570 | 1.306 | 0.399 | 0.305 |
| Methane | 1.730 | 1.080 | 3.180 | 1.602 | 0.544 | 0.340 |
| Mol_MnO | 4.320 | 3.100 | 9.220 | 1.394 | 0.469 | 0.336 |
| Ndia2 | 2.460 | 1.330 | 1.820 | 1.850 | 1.352 | 0.731 |
| **Total** | **61.970** | **43.580** | **126.480** | **1.422** | **0.490** | **0.345** |

<details>
<summary>SCF counts and numerical checks: -runtest (14 cases)</summary>

| Input | SCF steps (upstream / fork CPU / GPU) | Max abs ΔE (Ha) | Max abs ΔF (Ha/bohr) | Check |
| --- | --- | ---: | ---: | --- |
| Benzene | 19 / 19 / 19 | 0.000e+00 | 0.000e+00 | PASS |
| C60 | 27 / 27 / 27 | 2.802e-11 | 1.000e-12 | PASS |
| CO | 182 / 182 / 182 | 0.000e+00 | 1.500e-11 | PASS |
| Cr2 | 21 / 21 / 21 | 3.013e-12 | 2.010e-10 | PASS |
| Crys-MnO | 37 / 40 / 39 | 4.700e-10 | 6.470e-10 | PASS |
| GaAs | 13 / 13 / 13 | 1.600e-11 | 8.000e-12 | PASS |
| Glycine | 20 / 20 / 20 | 1.002e-12 | 1.000e-12 | PASS |
| Graphite4 | 14 / 14 / 14 | 2.000e-12 | 1.000e-12 | PASS |
| H2O-EF | 17 / 17 / 17 | 9.983e-13 | 1.000e-12 | PASS |
| H2O | 31 / 31 / 31 | 1.002e-12 | 1.000e-12 | PASS |
| HMn | 26 / 26 / 26 | 9.948e-13 | 0.000e+00 | PASS |
| Methane | 14 / 14 / 14 | 1.430e-10 | 1.600e-11 | PASS |
| Mol_MnO | 26 / 26 / 26 | 9.948e-13 | 1.000e-12 | PASS |
| Ndia2 | 13 / 13 / 13 | 3.000e-12 | 1.000e-12 | PASS |

</details>

##### `-runtestL` (16 cases)

| Input | Upstream CPU (s) | Fork CPU (s) | Fork GPU (s) | Upstream / Fork CPU | Upstream / GPU | Fork CPU / GPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 5_5_13COb2 | 49.980 | 45.820 | 42.590 | 1.091 | 1.174 | 1.076 |
| B2C62_Band | 270.330 | 174.900 | 382.190 | 1.546 | 0.707 | 0.458 |
| CG15c-DC-LNO | 69.520 | 51.520 | 56.180 | 1.349 | 1.237 | 0.917 |
| DIA512-1 | 43.240 | 30.890 | 94.880 | 1.400 | 0.456 | 0.326 |
| FeBCC | 70.560 | 51.160 | 72.280 | 1.379 | 0.976 | 0.708 |
| GEL | 26.580 | 17.780 | 106.590 | 1.495 | 0.249 | 0.167 |
| GFRAG | 31.140 | 28.660 | 39.370 | 1.087 | 0.791 | 0.728 |
| GGFF | 457.270 | 336.620 | 268.510 | 1.358 | 1.703 | 1.254 |
| MCCN | 65.100 | 58.560 | 165.640 | 1.112 | 0.393 | 0.354 |
| Mn12_148_F | 59.800 | 52.050 | 59.280 | 1.149 | 1.009 | 0.878 |
| N1C999 | 810.540 | 465.160 | 500.610 | 1.742 | 1.619 | 0.929 |
| Ni63-O64 | 54.350 | 49.480 | 48.910 | 1.098 | 1.111 | 1.012 |
| Pt63 | 43.760 | 46.290 | 54.940 | 0.945 | 0.797 | 0.843 |
| SialicAcid | 10.840 | 8.710 | 20.160 | 1.245 | 0.538 | 0.432 |
| ZrB2_2x2 | 103.470 | 72.770 | 140.570 | 1.422 | 0.736 | 0.518 |
| nsV4Bz5 | 52.870 | 43.040 | 120.990 | 1.228 | 0.437 | 0.356 |
| **Total** | **2219.350** | **1533.410** | **2173.690** | **1.447** | **1.021** | **0.705** |

<details>
<summary>SCF counts and numerical checks: -runtestL (16 cases)</summary>

| Input | SCF steps (upstream / fork CPU / GPU) | Max abs ΔE (Ha) | Max abs ΔF (Ha/bohr) | Check |
| --- | --- | ---: | ---: | --- |
| 5_5_13COb2 | 66 / 66 / 67 | 7.360e-10 | 1.330e-10 | PASS |
| B2C62_Band | 203 / 203 / 203 | 1.410e-10 | 4.720e-08 | PASS |
| CG15c-DC-LNO | 42 / 42 / 42 | 6.221e-10 | 1.790e-10 | PASS |
| DIA512-1 | 29 / 29 / 29 | 7.490e-10 | 9.659e-09 | PASS |
| FeBCC | 62 / 62 / 62 | 1.036e-09 | 6.900e-11 | PASS |
| GEL | 58 / 58 / 58 | 2.091e-10 | 4.110e-10 | PASS |
| GFRAG | 60 / 60 / 60 | 1.640e-10 | 1.776e-09 | PASS |
| GGFF | 100 / 98 / 101 | 8.617e-11 | 1.390e-10 | PASS |
| MCCN | 67 / 67 / 67 | 3.411e-10 | 3.000e-11 | PASS |
| Mn12_148_F | 61 / 61 / 61 | 2.910e-11 | 2.960e-10 | PASS |
| N1C999 | 61 / 59 / 58 | 1.570e-09 | 3.100e-11 | PASS |
| Ni63-O64 | 62 / 69 / 62 | 2.825e-09 | 6.220e-10 | PASS |
| Pt63 | 67 / 78 / 70 | 6.790e-09 | 2.320e-10 | PASS |
| SialicAcid | 29 / 29 / 29 | 3.013e-12 | 1.000e-12 | PASS |
| ZrB2_2x2 | 49 / 49 / 49 | 1.100e-10 | 3.820e-10 | PASS |
| nsV4Bz5 | 57 / 57 / 57 | 1.240e-10 | 6.840e-10 | PASS |

</details>

##### `-runtestL2` (7 cases, GPU only, two nodes)

| Input | Fork GPU (s) | SCF steps | SCF converged | Bundled abs ΔE (Ha) | Bundled max abs ΔF (Ha/bohr) | Strict reference check |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| C1000 | 1427.720 | 41 | yes | 4.375e-10 | 3.960e-10 | PASS |
| Fe1000 | 3992.710 | 315 | yes | 3.565e-09 | 1.153e-08 | PASS |
| GRA1024 | 1032.530 | 48 | yes | 1.945e-09 | 6.580e-10 | PASS |
| Ih-Ice1200 | 558.470 | 35 | yes | 2.574e-09 | 1.060e-10 | PASS |
| Pt500 | 10552.200 | 220 | yes | 1.797e-09 | 5.693e-08 | PASS |
| R-TiO2-1050 | 1526.480 | 39 | yes | 3.132e-09 | 4.140e-10 | PASS |
| Si1000 | 1235.640 | 41 | yes | 3.056e-10 | 1.010e-10 | PASS |
| **Total** | **20325.750** | — | — | — | — | — |

##### `-runtestL3` (20 cases, three routes, two nodes)

| Input | Upstream CPU (s) | Fork CPU (s) | Fork GPU (s) | Upstream / Fork CPU | Upstream / GPU | Fork CPU / GPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 5_5_13COb2 | 8.970 | 6.920 | 10.780 | 1.296 | 0.832 | 0.642 |
| C1000 | 269.370 | 242.490 | 140.650 | 1.111 | 1.915 | 1.724 |
| C60 | 2.360 | 1.500 | 2.720 | 1.573 | 0.868 | 0.551 |
| CG15c | 52.780 | 30.840 | 59.660 | 1.711 | 0.885 | 0.517 |
| Crys-MnO | 5.410 | 2.250 | 3.420 | 2.404 | 1.582 | 0.658 |
| DIA512-1 | 18.200 | 11.670 | 22.840 | 1.560 | 0.797 | 0.511 |
| Fe1000 | 220.520 | 165.970 | 108.380 | 1.329 | 2.035 | 1.531 |
| GEL | 10.960 | 6.440 | 15.320 | 1.702 | 0.715 | 0.420 |
| GFRAG | 6.260 | 4.700 | 5.990 | 1.332 | 1.045 | 0.785 |
| GGFF | 247.460 | 222.300 | 51.520 | 1.113 | 4.803 | 4.315 |
| MCCN | 44.510 | 24.540 | 37.660 | 1.814 | 1.182 | 0.652 |
| Mn12_148_F | 10.960 | 7.400 | 8.410 | 1.481 | 1.303 | 0.880 |
| N1C999 | 539.250 | 274.890 | 404.970 | 1.962 | 1.332 | 0.679 |
| Ni63-O64 | 13.150 | 9.620 | 9.580 | 1.367 | 1.373 | 1.004 |
| Pt500 | 456.030 | 335.360 | 253.450 | 1.360 | 1.799 | 1.323 |
| Pt63 | 8.620 | 6.520 | 7.590 | 1.322 | 1.136 | 0.859 |
| Si1280-LNO | 551.560 | 310.740 | 416.520 | 1.775 | 1.324 | 0.746 |
| SialicAcid | 5.250 | 3.600 | 6.040 | 1.458 | 0.869 | 0.596 |
| ZrB2_2x2 | 45.040 | 32.480 | 26.620 | 1.387 | 1.692 | 1.220 |
| nsV4Bz5 | 9.960 | 6.890 | 10.650 | 1.446 | 0.935 | 0.647 |
| **Total** | **2526.620** | **1707.120** | **1602.770** | **1.480** | **1.576** | **1.065** |

<details>
<summary>SCF counts and numerical checks: -runtestL3 (20 cases)</summary>

| Input | SCF steps (upstream / fork CPU / GPU) | Max abs ΔE (Ha) | Max abs ΔF (Ha/bohr) | Check |
| --- | --- | ---: | ---: | --- |
| 5_5_13COb2 | 3 / 3 / 3 | 8.024e-09 | 3.000e-10 | PASS |
| C1000 | 3 / 3 / 3 | 3.251e-09 | 1.039e-09 | PASS |
| C60 | 3 / 3 / 3 | 2.802e-11 | 1.000e-12 | PASS |
| CG15c | 3 / 3 / 3 | 1.490e-08 | 1.399e-09 | PASS |
| Crys-MnO | 3 / 3 / 3 | 4.550e-10 | 5.380e-10 | PASS |
| DIA512-1 | 3 / 3 / 3 | 7.658e-10 | 3.634e-09 | PASS |
| Fe1000 | 3 / 3 / 3 | 4.293e-09 | 6.124e-07 | MISMATCH |
| GEL | 3 / 3 / 3 | 2.840e-10 | 2.753e-09 | PASS |
| GFRAG | 3 / 3 / 3 | 7.170e-10 | 1.782e-09 | PASS |
| GGFF | 3 / 3 / 3 | 8.004e-11 | 9.500e-11 | PASS |
| MCCN | 3 / 3 / 3 | 1.098e-09 | 1.279e-09 | PASS |
| Mn12_148_F | 3 / 3 / 3 | 9.504e-11 | 1.392e-09 | PASS |
| N1C999 | 3 / 3 / 3 | 1.571e-09 | 1.240e-10 | PASS |
| Ni63-O64 | 3 / 3 / 3 | 1.597e-07 | 7.288e-06 | MISMATCH |
| Pt500 | 3 / 3 / 3 | 1.506e-09 | 8.180e-10 | PASS |
| Pt63 | 3 / 3 / 3 | 2.850e-09 | 4.340e-10 | PASS |
| Si1280-LNO | 3 / 3 / 3 | 1.383e-08 | 3.390e-10 | PASS |
| SialicAcid | 3 / 3 / 3 | 2.984e-12 | 1.000e-12 | PASS |
| ZrB2_2x2 | 3 / 3 / 3 | 7.358e-09 | 1.126e-09 | PASS |
| nsV4Bz5 | 3 / 3 / 3 | 1.224e-09 | 1.753e-09 | PASS |

</details>

##### Named inputs

| Input | Upstream CPU (s) | Fork CPU (s) | Fork GPU (s) | Upstream / Fork CPU | Upstream / GPU | Fork CPU / GPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `sidia333_nc_cluster.dat` | 247.520 | 192.886 | 120.946 | 1.283 | 2.047 | 1.595 |
| `cghelgeo.dat` | 860.989 | 792.170 | 362.277 | 1.087 | 2.377 | 2.187 |

<details>
<summary>SCF counts and numerical checks: Named inputs</summary>

| Input | SCF steps (upstream / fork CPU / GPU) | Max abs ΔE (Ha) | Max abs ΔF (Ha/bohr) | Check |
| --- | --- | ---: | ---: | --- |
| `sidia333_nc_cluster.dat` | 25 / 25 / 25 | 1.430e-10 | 2.010e-10 | PASS |
| `cghelgeo.dat` | 59 / 61 / 60 | 3.850e-08 | 1.293e-08 | PASS |

</details>

#### Accuracy and convergence

The primary three configurations pass **96 pairwise comparisons** over
the 32 S/L/named inputs at **1e-7 Ha** for energy and **1e-7 Ha/bohr**
for every force component. Maximum differences are **3.850437e-08 Ha** and
**4.720100e-08 Ha/bohr**. The fresh GPU results also pass **128 comparisons**
against the three 2026-10-31 routes and the previous optimized GPU run.
Geometry and grids agree. Input physics, basis, SCF/MD limits and
convergence thresholds were preserved. The original **25-SCF sidia333
remains unconverged**; numerical agreement is separate from convergence.
SCF counts sum every MD step.

The GPU L2/L3 results have 27 comparisons with bundled
reference outputs: **25 within** and **2 outside** the same strict
energy/force tolerances; the case tables report the actual differences.
Cases outside tolerance: L3/Fe1000, L3/Ni63-O64.
These references are separate from the matched Sirius CPU measurements.
L3 retains the original **3-SCF limit** and is a timing test of incomplete
SCF calculations. The fresh L3 trio additionally has **55 passing** and
**5 mismatching** strict pairwise comparisons; its detailed table reports
all differences. Ni63-O64 also exceeds the strict energy tolerance in
comparisons involving upstream CPU (maximum **1.597436e-07 Ha**).
No equivalence claim is made for out-of-tolerance energy or force, or
from finite output alone. Completed C1000 outputs from all three earlier
two-node routes are checked separately in the measurement record.
The historical H100 L3 Fe1000 and Ni63-O64 GPU measurements also exceeded
the strict force tolerance at three SCF steps; their existing comparison rows are
retained as context, without treating them as a matched Sirius CPU test
or identifying the cause of the present difference.

#### Measurement records

All six GPU workloads (59 cases) and both CPU L3 suites (40 case instances)
were freshly measured. The existing eight CPU measurements for S/L and
the named inputs were reused. The four fresh PBS jobs consumed at most **61.47 points** of the
authorized **100-point allowance**, rounding each job up to 0.01 points;
the maximum of cumulative actual spending plus outstanding hard
reservations was **98.75 points**. No campaign GPU job remains. No `gpusolver2`/`gpusolver3` timing was run.

[Measurement record](benchmarks/results/sirius_mi300a_gpu_rebench_20261101_32x3/README.txt),
[GPU case timings](benchmarks/results/sirius_mi300a_gpu_rebench_20261101_32x3/cases.csv),
[fresh CPU L3 timings](benchmarks/results/sirius_mi300a_gpu_rebench_20261101_32x3/cpu-cases.csv),
[current three-route checks](benchmarks/results/sirius_mi300a_gpu_rebench_20261101_32x3/current-three-route-comparisons.csv),
[previous GPU/CPU checks](benchmarks/results/sirius_mi300a_gpu_rebench_20261101_32x3/comparisons.csv),
[L3 three-route checks](benchmarks/results/sirius_mi300a_gpu_rebench_20261101_32x3/L3-three-route-comparisons.csv)
and [bundled L2/L3 reference checks](benchmarks/results/sirius_mi300a_gpu_rebench_20261101_32x3/bundled-reference-comparisons.csv)
retain commands, hashes, numerical results, convergence, thread/affinity
audits and job accounting. The
[GPU dispatch traces](benchmarks/results/sirius_mi300a_gpu_rebench_20261101_32x3/gpu-dispatch.json)
retain actual solver calls and any CPU fallback messages.
The [optimization method and controls](doc/sirius_mi300a_ggff_optimization_20261101.txt)
and [previous optimized measurements](benchmarks/results/sirius_mi300a_ggff_20261101/README.txt)
retain the separate 6.75-point optimization campaign; its cost is excluded
from this new 100-point allowance.

L2 is GPU-only in the requested scope. The earlier CPU attempt retained
only a completed C1000 prefix; no full CPU L2 total or speedup is available.
The [CPU extension estimate](benchmarks/results/sirius_mi300a_gpu_rebench_20261101_32x3/cpu-extension-estimate.json)
records the additional CPU L3 plan and the deferred CPU L2 scope.

## Important notes
Performance depends on the system, solver, hardware and MPI/OpenMP layout.
Use the same binary and inputs when comparing CPU and GPU execution;
`OPENMX_GPU=0` selects the CPU paths for that comparison. Check numerical
agreement alongside execution time.

## About bug reports
I would appreciate it if you could actively report any bugs. Please report them via GitHub issues or send them to [my X account](https://x.com/dc1394). Bug reports sent to my X account can be in English.

## License

This project is a derivative work of **OpenMX** and is licensed under the
**GNU General Public License v3.0 or later (GPL-3.0-or-later)**. See
[LICENSE](LICENSE) for the full license text.

```text
SPDX-License-Identifier: GPL-3.0-or-later
```

### Upstream Project and OpenMX Packages

- **OpenMX** (Open source package for Material eXplorer), version **4.0**
- **Upstream**: <https://www.openmx-square.org/>
- **Copyright**: Taisuke Ozaki and OpenMX contributors
- **Project license**: `GPL-3.0-or-later`

The original OpenMX files retain their copyright notices and license headers.
This project's CPU/GPU changes and C++ core are covered by the project license.
The independent `OpenMX::basis`, `OpenMX::xc` and `OpenMX::io` packages carry
copies in [modules/basis/LICENSE](modules/basis/LICENSE),
[modules/xc/LICENSE](modules/xc/LICENSE) and
[modules/io/LICENSE](modules/io/LICENSE).
The third-party components below retain their own licenses and notices.

### Third-Party Components

The build uses dependencies from Git submodules, source archives under
`source/third_party/`, and the pinned Kokkos release fetched by CMake. The
following entries describe those versions and identify their upstream license
texts and file-specific notices.

#### Kokkos, mdspan and desul atomics

- **Kokkos version**: **5.2.2**, pinned with its source URL and SHA-256 in
  [core/CMakeLists.txt](core/CMakeLists.txt)
- **Source**: <https://github.com/kokkos/kokkos/tree/5.2.2>
- **Copyright**: Contributors to the Kokkos project; see the release's
  [COPYRIGHT.md](https://github.com/kokkos/kokkos/blob/5.2.2/COPYRIGHT.md)
- **License**: Apache License 2.0 with LLVM Exceptions
  (`Apache-2.0 WITH LLVM-exception`)
- **Original license text**:
  [Kokkos 5.2.2 LICENSE](https://github.com/kokkos/kokkos/blob/5.2.2/LICENSE).
  CMake installs this file as
  `<prefix>/share/licenses/openmx_core/Kokkos-LICENSE`.

Kokkos provides execution spaces and views for the C++ core in CPU-only,
NVIDIA CUDA and AMD HIP builds, and in the standalone SYCL core. The isolated
host SIMD library uses a SERIAL configuration of the same pinned source.

The release also supplies two components with their own notices:

- **mdspan**: the bundled snapshot recorded in
  [tpls/mdspan-hash.txt](https://github.com/kokkos/kokkos/blob/5.2.2/tpls/mdspan-hash.txt)
  also uses `Apache-2.0 WITH LLVM-exception`. Its
  [header notices](https://github.com/kokkos/kokkos/blob/5.2.2/tpls/mdspan/include/mdspan/mdspan.hpp)
  identify NTESS and the applicable terms. The core uses these headers for
  multidimensional views; where supported, its public host views can instead
  use the toolchain's `std::mdspan` implementation.
- **desul atomics**: Kokkos bundles snapshot `83d26f34` of
  [desul/desul](https://github.com/desul/desul/tree/83d26f34db50b4fefe30ed4db38b10e04c7a9384)
  for atomic operations. These headers retain the copyright of Lawrence
  Livermore National Security, LLC and DESUL contributors and use
  **BSD 3-Clause** (`BSD-3-Clause`), as stated in their
  [bundled notices](https://github.com/kokkos/kokkos/blob/5.2.2/tpls/desul/include/desul/atomics.hpp)
  and [upstream LICENSE](https://github.com/desul/desul/blob/83d26f34db50b4fefe30ed4db38b10e04c7a9384/LICENSE).

#### FFTW3

- **Source**: <https://www.fftw.org/>
- **Version**: **3.3.11**, supplied by the Git submodule
  `source/third_party/fftw3` and the release archive
  `source/third_party/fftw-3.3.11.tar.gz`
- **Copyright**:
  - Copyright (c) 2003, 2007-14 Matteo Frigo
  - Copyright (c) 2003, 2007-14 Massachusetts Institute of Technology
- **License**: GNU General Public License v2.0 or later (`GPL-2.0-or-later`)
- **Original license files**: `COPYING` and the source-file notices in the
  submodule and archive

The NVIDIA build uses this bundled FFTW3. AMD and CPU-only builds use an
external FFTW3 selected from the build environment.

#### GEMMul8

- **Source**: <https://github.com/RIKEN-RCCS/GEMMul8>
- **Version**: **v3.5.4**, Git submodule `source/third_party/GEMMul8`
- **Copyright**: Copyright (c) 2025- RIKEN R-CCS
- **Responsible developer**: Yuki Uchino (RIKEN Center for Computational Science)
- **License**: MIT (`MIT`)
- **Original license file**: `source/third_party/GEMMul8/LICENSE`

GEMMul8 supplies emulated matrix products when selected by the GPU build and
runtime policy. For academic use, please also cite the upstream GEMMul8
references.

#### ELPA, COSMA, COSTA and Tiled-MM

The pinned source archives are in `source/third_party/dist/`. ELPA supplies
CPU eigensolvers and the ELPA/COSMA `gpusolver2` path; COSMA uses COSTA and
Tiled-MM for its distributed GPU matrix products.

| Component | Pinned version | Upstream license | License text in its archive |
|---|---|---|---|
| [ELPA](https://elpa.mpcdf.mpg.de/) | 2026.02.002 | GNU LGPL v3 (`LGPL-3.0-only`) | `LICENSE`, `COPYING/lgpl.txt` and `COPYING/gpl.txt` |
| [COSMA](https://github.com/eth-cscs/COSMA) | 2.8.4 | BSD 3-Clause (`BSD-3-Clause`) | `LICENSE` |
| [COSTA](https://github.com/eth-cscs/COSTA) | 2.3.0, commit `2484769535772f807d402901ffca63bb6678dd42` | BSD 3-Clause (`BSD-3-Clause`) | `LICENSE` |
| [Tiled-MM](https://github.com/eth-cscs/Tiled-MM) | Commit `0eb75179e670a04c649b50ae5e91bb71b43e4d06`, pinned by COSMA 2.8.4 | BSD 3-Clause (`BSD-3-Clause`) | `LICENSE` |

ELPA retains the copyright of its consortium and contributors. COSMA retains
ETH Zürich and Advanced Micro Devices notices; COSTA and Tiled-MM retain their
ETH Zürich notices. OpenMX's build patches preserve these upstream notices.

#### MAGMA

- **Source**: <https://icl.utk.edu/magma/>
- **Version**: **2.10.0**, archive
  `source/third_party/hip/magma/magma-2.10.0.tar.gz`, shared by both GPU backends
- **Copyright**: Copyright (c) 2009-2023, The University of Tennessee
- **License**: BSD 3-Clause (`BSD-3-Clause`)
- **Original license file**: `COPYRIGHT` inside the archive

MAGMA supplies the multi-GPU dense eigensolver when enabled. The build applies
`source/third_party/magma-device-map.patch` for NVIDIA and
`source/third_party/hip/magma-ozaki2.patch` for AMD.

### Build Tools and External Libraries

The bundled **CMake 3.31.12** archive,
`source/third_party/dist/cmake-3.31.12.tar.gz`, is a fallback build tool for the
ELPA/COSMA stack. Its main license is **BSD 3-Clause** (`BSD-3-Clause`), with
Copyright 2000-2024 Kitware, Inc. and Contributors in `Copyright.txt`. CMake is
not linked into the OpenMX executable; its archive retains its additional
third-party notices.

CUDA/NVIDIA HPC SDK, ROCm libraries, MPI, and the selected numerical libraries
(such as oneMKL, AOCL or OpenBLAS/ScaLAPACK) come from the build environment.
Those installations, the external FFTW3 and the compiler's standard library
have their own license and redistribution terms.

### License Summary

| Component | License | Use in this project |
|---|---|---|
| OpenMX and this project's CPU/GPU/C++ changes | `GPL-3.0-or-later` | Application, C++ core and the basis/xc/io packages |
| Kokkos 5.2.2 | `Apache-2.0 WITH LLVM-exception` | Core execution spaces, views and host SIMD |
| mdspan bundled with Kokkos 5.2.2 | `Apache-2.0 WITH LLVM-exception` | Multidimensional views |
| desul atomics bundled with Kokkos 5.2.2 | `BSD-3-Clause` | Atomic operations |
| FFTW3 3.3.11 | `GPL-2.0-or-later` | Bundled NVIDIA FFT implementation |
| GEMMul8 v3.5.4 | `MIT` | Selected GPU matrix products |
| ELPA 2026.02.002 | `LGPL-3.0-only` | CPU and distributed GPU eigensolvers |
| COSMA / COSTA / Tiled-MM | `BSD-3-Clause` | Distributed matrix products |
| MAGMA 2.10.0 | `BSD-3-Clause` | Multi-GPU dense eigensolver |
| CMake 3.31.12 | `BSD-3-Clause` | Build tool only |

The OpenMX-derived application is distributed under `GPL-3.0-or-later`;
third-party components retain the terms and notices identified above.

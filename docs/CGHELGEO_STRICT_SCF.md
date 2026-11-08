# cghelgeo strict SCF comparison (Radeon, 2026-11-08)

The upstream CPU, CPU AVX2, HIP with FP64 grid/orbital/GEMM arithmetic and the default Radeon precision policy agree within the predeclared energy/force tolerances when inherited VNA/force product rounding is retained.

The full `scf.fp32.enable=off` calculation is a broader arithmetic comparison: it also promotes VNA/force float products inherited from the CPU implementation, including their GPU implementations. Its numerical differences are shown separately in the tables; it must not be conflated with switching only grid/orbital/GEMM arithmetic to native FP64.

The full switch has an upstream energy difference of **1.627e-7 Ha**, exceeding the retained 1e-7 Ha energy tolerance; its maximum force difference is **3.506e-8 Ha/bohr**. This offset remains after convergence at 1e-12. The paired grid/orbital/GEMM-only FP64 control differs from the default HIP route by **3.911e-11 Ha** and **5.000e-12 Ha/bohr**. Together with the source-defined scope of `scf.fp32.enable`, this isolates the broader arithmetic-policy effect from the GPU grid/orbital/GEMM precision switch.

Each calculation used **8 MPI ranks × 2 OpenMP/BLAS threads**, with **16 distinct physical cores** verified from every rank's runtime affinity, and `scf.criterion=1e-12` from the start.
The current fork was rebuilt from [`5ab5f1ed`](https://github.com/dc1394/openmx4.0_gpu/commit/5ab5f1edf7f8e322bad5cc230ba2ec0451504c37) with ROCm/HIP, OpenMPI and AOCL.

## Energy and force differences

Differences in this table use upstream CPU as the reference. Forces cover all **1,950 Cartesian components of 650 atoms**.

| Route | SCF steps | Total energy (Ha) | abs ΔE to upstream (Ha) | max abs ΔF to upstream (Ha/bohr) |
| --- | ---: | ---: | ---: | ---: |
| Upstream OpenMX 4.0 CPU | 17 | -4153.148190613800 | 0.000e+00 | 0.000e+00 |
| Fork CPU (AVX2) | 22 | -4153.148190613815 | 1.546e-11 | 1.950e-10 |
| Fork HIP (full FP64 arithmetic; legacy products promoted) | 38 | -4153.148190776532 | 1.627e-07 | 3.506e-08 |
| Fork HIP (default emulation policy) | 19 | -4153.148190613721 | 7.822e-11 | 4.080e-10 |
| Fork HIP (FP64 grid/orbital/GEMM; legacy products unchanged) | 19 | -4153.148190613761 | 3.911e-11 | 4.080e-10 |

All pairwise differences:

| Reference | Candidate | abs ΔE (Ha) | max abs ΔF (Ha/bohr) |
| --- | --- | ---: | ---: |
| Upstream OpenMX 4.0 CPU | Fork CPU (AVX2) | 1.546e-11 | 1.950e-10 |
| Upstream OpenMX 4.0 CPU | Fork HIP (full FP64 arithmetic; legacy products promoted) | 1.627e-07 | 3.506e-08 |
| Upstream OpenMX 4.0 CPU | Fork HIP (default emulation policy) | 7.822e-11 | 4.080e-10 |
| Upstream OpenMX 4.0 CPU | Fork HIP (FP64 grid/orbital/GEMM; legacy products unchanged) | 3.911e-11 | 4.080e-10 |
| Fork CPU (AVX2) | Fork HIP (full FP64 arithmetic; legacy products promoted) | 1.627e-07 | 3.499e-08 |
| Fork CPU (AVX2) | Fork HIP (default emulation policy) | 9.368e-11 | 3.410e-10 |
| Fork CPU (AVX2) | Fork HIP (FP64 grid/orbital/GEMM; legacy products unchanged) | 5.457e-11 | 3.410e-10 |
| Fork HIP (full FP64 arithmetic; legacy products promoted) | Fork HIP (default emulation policy) | 1.628e-07 | 3.490e-08 |
| Fork HIP (full FP64 arithmetic; legacy products promoted) | Fork HIP (FP64 grid/orbital/GEMM; legacy products unchanged) | 1.628e-07 | 3.490e-08 |
| Fork HIP (default emulation policy) | Fork HIP (FP64 grid/orbital/GEMM; legacy products unchanged) | 3.911e-11 | 5.000e-12 |

## Controlled conditions

All routes start from independent copies of one near-converged upstream restart density (723 files). The seed is identical byte for byte. These are **local fixed-point continuation checks**, not independent cold-start runs or performance measurements.

The geometry, basis, pseudopotentials, 150 Ry cutoff, 300 K electronic temperature, 1×1×3 k-grid and 210×210×250 integration grid are unchanged. All routes use fixed 0.3 Hamiltonian mixing; `scf.Mixing.StartPulay=201` defers Pulay beyond the unchanged 200-step limit. No runtime SCF keyword file or mid-run criterion adjustment is used.

The upstream executable is the retained Intel oneAPI 2026.0 / Intel MPI 2021.18 / MKL 2026.0 build, byte-identical to `~/openmx4.0/source/openmx`. The fork uses ROCm Core SDK 10.1 / Clang-Flang 24, OpenMPI 5.0.9 and AOCL 5.2 OpenMP. CPU AVX2 is selected explicitly with `OPENMX_GPU=0` and `OPENMX_HOST_SIMD=avx2` and verified in stdout.

The native FP64 route sets `scf.fp32.enable off` and `scf.gemmul8.enable off`, with FP64 grid/orbital evaluation, refinement disabled and MAGMA Ozaki-II disabled. This also promotes inherited CPU VNA/force float products to FP64. Orbital/projector storage remains unchanged. The default HIP route leaves these precision options unset and retains the current device policy.
The grid/orbital/GEMM control keeps `scf.fp32.enable` at its default and sets `OPENMX_GRID_PRECISION=fp64`, `OPENMX_ORBS_EVAL_PRECISION=fp64` and `scf.gemmul8.enable=off`, with refinement and MAGMA Ozaki-II disabled. Thus inherited CPU products retain the upstream policy.

Actual dense CPU fallback counts and precision messages are recorded per run in the JSON. GPU initialization failures, incomplete outputs, nonfinite values or reaching the iteration limit cannot pass. The stop condition uses band-energy change and a mixing residual; it is not an absolute total-energy error bound. Printed diagnostics and final energy/forces have 12 decimal places: a displayed `dUele=1e-12` can round an internal value below that threshold. Early SCF exit is checked against the unmodified source stopping predicate.

| HIP setting | Dense CPU fallback messages | GPU initialization failures |
| --- | ---: | ---: |
| Fork HIP (full FP64 arithmetic; legacy products promoted) | 0 | 0 |
| Fork HIP (default emulation policy) | 0 | 0 |
| Fork HIP (FP64 grid/orbital/GEMM; legacy products unchanged) | 0 | 0 |

These 8×2 accuracy runs are separate from the older 16×2 performance measurements, whose GPU route included CPU dense diagonalization.

All successful runs use an unlimited process stack and `OMP_STACKSIZE=256M`. The first upstream attempt with the default 8 MiB process stack ended in a rank SIGSEGV followed by MPI cleanup; it published no result and is excluded. The same input and executable were rerun with sufficient stack space. The failed attempt remains in the raw archive.

## Records

[CSV](../benchmarks/results/radeon_cghelgeo_strict_20261108_8x2.csv) and [JSON](../benchmarks/results/radeon_cghelgeo_strict_20261108_8x2.json) contain the differences, full force vectors, energy components, SCF histories, input/output/executable/dataset/library hashes, commands, precision settings and rank affinity checks. Raw logs and restart snapshots are under `work/codex_cghelgeo_strict_20261108_8x2/`. No production source was modified.

# NVIDIA check of retained upstream float products (2026-11-09)

The correction in `288a2c7480705332ba608750a6248a556997ca5d` works on the tested RTX 5080 CUDA build.
`scf.fp32.enable off` retains the inherited VNA/force/stress float products while
disabling the optional FP32/double-float paths. A float product rounds before
being widened and accumulated in double. The existing projector and derivative
storage formats are unchanged.

These are **matched-seed continuation correctness checks**, not cold-start or
performance measurements. All four current-fork runs use byte-identical copies
of the converged default-GPU restart from the November 8 campaign (82 preparatory
SCF steps), the original 650-atom geometry and PAO/VPS files, and
`scf.criterion=1e-10`. The upstream CPU result is reused from that campaign;
its convergence and initialization details remain in the original records.
No upstream executable was rerun for this check. The four new runs each reach
the native SCF stop condition before the iteration cap and finish normally.

All runs use 18 MPI ranks × 2 OpenMP/BLAS threads. Rank i is placed on logical
CPUs i and i+18 (the two hardware threads of one physical core). GPU runs use
a private MPS service. The computer has an Intel Core i9-10980XE, RTX 5080
(sm_120), NVIDIA HPC SDK 26.9, CUDA 13.4 and OpenMPI 5.0.10rc2. The host SIMD
library uses GCC 14.2. Runtime maps verify the frozen libraries, NVHPC OpenMP
and the NVIDIA HPC-X MPI library on every rank.

| Route | SCF steps | Total energy (Ha) | abs ΔE to upstream (Ha) | max abs ΔF to upstream (Ha/bohr) |
| --- | ---: | ---: | ---: | ---: |
| Retained upstream OpenMX 4.0 CPU reference | 2 | -4153.148190613880 | 0.000e+00 | 0.000e+00 |
| CUDA: FP32 off, native FP64, GEMMul8 off | 2 | -4153.148190613758 | 1.220e-10 | 1.363e-09 |
| CUDA: FP32 off, default FP64 emulation/GEMMul8 | 2 | -4153.148190613754 | 1.260e-10 | 1.363e-09 |
| CUDA: default precision controls | 2 | -4153.148190613744 | 1.360e-10 | 1.363e-09 |
| CPU: FP32 off, AVX-512 | 2 | -4153.148190613765 | 1.150e-10 | 1.281e-09 |

All ten pairs pass the unchanged `1e-7` Ha energy and `1e-7` Ha/bohr maximum
force-component gates, including the printed-rounding bounds. The largest
printed pairwise differences are **1.360e-10 Ha** and
**1.363e-09 Ha/bohr**, over all 1,950 force components. Geometry, species,
FFT grid, input hashes, finite outputs, convergence traces and restart hashes
are checked alongside the numerical differences. `NormRD` is printed as the
square root of the stored norm; verification follows the stopping predicate in
`DFT.c` rather than comparing that display directly to the energy criterion.

## Precision and execution checks

- `gpu_off_native`: `scf.fp32.enable off`, `scf.gemmul8.enable off` and
  `OPENMX_CUSOLVER_EMULATION=0`; all audited dense solves use native FP64.
- `gpu_off_emulated`: `scf.fp32.enable off`, with GEMMul8 and cuSOLVER emulation
  left at their defaults; audited FP64 solves use emulated math mode.
- `gpu_default`: all precision controls are left unset.
- `cpu_avx512_off`: `OPENMX_GPU=0`, `OPENMX_HOST_SIMD=avx512`,
  `scf.fp32.enable off` and `scf.eigen.lib elpa2`; AVX-512 is reported at runtime.

The forwarding cuSOLVER observer records actual dimension-6,670 solve calls,
argument types, successful statuses and queried math modes. It forwards the
calls unchanged. GEMMul8/cuBLAS counters verify the selected GEMM routes.
GPU VNA and force stages are reported in every CUDA log. There are no dense
memory-admission rejections or GPU initialization failures. The bounded
Hamiltonian scheduler still assigns some ranks to its normal CPU fallback;
this is separate from the audited GPU dense solves. The private MPS service
is observed during execution and stopped after the GPU checks.

## Regression tests

The NVIDIA executable rebuilt successfully. Three precision/policy scripts,
all 11 targeted CTest cases, and the OpenACC VNA archive fixture on the actual
GPU pass. The latter exercises 46 archive/contraction cases, including capacity
recovery, allocation/copy failures, host dispatch, empty ranks and empty halos.
The CTest cases cover force resources/packing/recovery, VNA recovery, stress
and solver architecture policy. Full `-runtest`/`-runtestL` suites were not run.

The VNA recovery test reference was corrected to retain float-product rounding
with the precision switch both on and off. Its `1e-12` tolerance is unchanged.
An initial GCC sanitizer fixture also inherited NVHPC's `libgomp.so.1` alias
to `libnvomp.so`; clearing `LD_LIBRARY_PATH` for the CTest child makes the
unchanged sanitizer fixture pass. Initial failure logs and the successful
rerun are retained in `regression-tests.json` and the referenced logs.

## Records

`analysis.json` and `comparisons.csv` contain all ten comparisons. Each route
directory retains the exact input, full force vector, SCF trace, compressed
stdout/output, rank placement, runtime libraries and resource samples. Solver
events are compressed separately. `provenance.json` includes binary/library,
dataset and 695 seed-file hashes; `SHA256SUMS` covers the archived files.

The upstream reference remains in
[`rtx5080_cghelgeo_strict_20261108/upstream`](../rtx5080_cghelgeo_strict_20261108/upstream/).
Those historical promoted-product failures remain unchanged. The old off/native
matched-seed energy differed from upstream by `1.62646e-7` Ha; this corrected
off/native run differs by `1.220e-10` Ha.

Private binaries, the 247 MiB seed and complete runtime-map snapshots remain
under `/home/dc1394/openmx4.0_gpu/build/cuda-fp32-legacy-validation-20261109-092916`. They are not embedded in Git. The archived runner/observer scripts
record the procedure; replaying the identical continuation requires those
preserved build artifacts and the seed described by its manifest.

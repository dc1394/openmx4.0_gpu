# cghelgeo: corrected projector rounding and independent cold starts

The four routes were independently converged from atomic initial densities at `scf.criterion=1e-12`, using **8 MPI ranks × 2 OpenMP/BLAS threads**. No restart density, Hamiltonian or mixing history was supplied to any cold-start run.

Across all six pairs, the largest differences are **9.641e-11 Ha** in total energy and **4.080e-10 Ha/bohr** among all 1,950 force components. Each pair passes the predeclared 1e-7 Ha / 1e-7 Ha/bohr tolerances.

## Precision-option correction

`scf.fp32.enable=off` now disables optional accelerated FP32/double-float grid, orbital and eigenvector-refinement arithmetic while retaining the float products inherited from upstream OpenMX in VNA, force and stress contractions. The product rounds in float before being widened and accumulated in double. GEMMul8 remains an independent setting; the native comparison also disables it.

The correction covers the shared CUDA/CPU implementation and the HIP implementation, including host SIMD callers, normal device batches, bounded batches, capacity recovery and VNA assembly. The SIMD core interfaces are unchanged. Existing orbital/projector storage formats are unchanged.

The [earlier strict continuation comparison](CGHELGEO_STRICT_SCF.md) records the old behavior: promoting inherited products produced a 1.627e-7 Ha energy offset. Those historical results are retained and are not relabeled as passing.

## First: isolate the correction against upstream

Before starting the four cold runs, the corrected native-FP64 route was converged from the historical near-converged seed, with the same fixed 0.3 Hamiltonian mixing and strict threshold used for the retained upstream continuation. This separate diagnostic verifies the arithmetic correction; it is **not** counted as a cold-start result.

It converged in 22 SCF steps to -4153.148190613771 Ha. Differences to that upstream result are 2.910e-11 Ha and 4.080e-10 Ha/bohr.

## Then: four independent cold starts

| Route | SCF steps | Total energy (Ha) | abs ΔE to upstream (Ha) | max abs ΔF to upstream (Ha/bohr) |
| --- | ---: | ---: | ---: | ---: |
| Upstream OpenMX 4.0 CPU | 78 | -4153.148190613800 | 0.000e+00 | 0.000e+00 |
| Fork CPU (AVX2) | 77 | -4153.148190613826 | 2.638e-11 | 1.950e-10 |
| Fork HIP (native FP64 accelerated paths) | 78 | -4153.148190613756 | 4.366e-11 | 4.080e-10 |
| Fork HIP (default FP64 emulation policy) | 78 | -4153.148190613730 | 7.003e-11 | 4.080e-10 |

| Reference | Candidate | abs ΔE (Ha) | max abs ΔF (Ha/bohr) |
| --- | --- | ---: | ---: |
| Upstream OpenMX 4.0 CPU | Fork CPU (AVX2) | 2.638e-11 | 1.950e-10 |
| Upstream OpenMX 4.0 CPU | Fork HIP (native FP64 accelerated paths) | 4.366e-11 | 4.080e-10 |
| Upstream OpenMX 4.0 CPU | Fork HIP (default FP64 emulation policy) | 7.003e-11 | 4.080e-10 |
| Fork CPU (AVX2) | Fork HIP (native FP64 accelerated paths) | 7.003e-11 | 3.410e-10 |
| Fork CPU (AVX2) | Fork HIP (default FP64 emulation policy) | 9.641e-11 | 3.410e-10 |
| Fork HIP (native FP64 accelerated paths) | Fork HIP (default FP64 emulation policy) | 2.638e-11 | 5.000e-12 |

## Conditions and checks

Every cold-start route starts in its own fresh directory containing only its input. `scf.restart=off` is explicit, no `cghelgeo_rst` directory exists before launch, and none of the logs reports reading restart data. Restart files written by the completed calculations are outputs, not seeds. No mid-run threshold adjustment is used. A mixing-only SCF control file applies the common transition described below; it supplies no density or restart data.

The initial RMM-DIISH weights and Pulay start are common: initial/minimum weight .001, maximum .1 and Pulay start 23. All four cold starts use history 8 from the outset. For a route still running after 63 completed steps, an identical mixing-only SCF file sets minimum/maximum weights .3 and Pulay start 401. It takes effect around SCF64 and continues the same atomic-density calculation with simple Hamiltonian mixing. No stopping threshold is changed. The JSON records each file creation and the first SCF step actually reporting weight .3. The strict threshold is 1e-12 from the first SCF step. The iteration cap is increased from 200 to 400, identically in all four inputs; every accepted calculation exits before its cap.

An exploratory upstream cold-start trial with the original history 50 was intentionally stopped after 93 SCF steps: its energy changes continued to oscillate mostly around 1e-9 to 1e-8 Ha after the mixing residual became small. It did not satisfy 1e-12 and is not reported as a pass or an iteration-limit failure. Its logs and metadata remain under `archives/coldstart-history50-93SCF/`. The accepted history-8 calculations all begin again from atomic densities, without reusing this trial or any other restart.

The original 650-atom geometry, basis, pseudopotentials, 150 Ry cutoff, 300 K electronic temperature, 1×1×3 k-grid and 210×210×250 integration grid are unchanged. Input, dataset, executable, loaded-library, output and log hashes are recorded. Actual rank affinities verify eight ranks with two physical cores each and a total of 16 physical cores.

The upstream executable is the retained Intel oneAPI 2026.0 / Intel MPI 2021.18 / MKL 2026.0 build, byte-identical to `~/openmx4.0/source/openmx`. The corrected fork uses ROCm Core SDK 10.1 / Clang-Flang 24, OpenMPI 5.0.9 and AOCL 5.2 OpenMP on Radeon RX 9060 XT 16 GiB (gfx1200). The CPU route explicitly selects AVX2 and disables GPU execution.

The native HIP route uses `scf.fp32.enable=off`, `scf.gemmul8.enable=off`, FP64 grid/orbital overrides, and disabled eigenvector refinement and MAGMA Ozaki-II. The default HIP route leaves precision controls unset. Both retain upstream float-product rounding after this correction. Both GPU routes select MAGMA at every SCF step. The complete logs contain no CPU dense fallback messages or GPU solver failures, and Radeon activity was recorded during the solves. Backend headers are checked together with those fallback/error paths and activity records, separately from GPU matrix-construction markers.

| HIP route | Dense CPU fallback messages | GPU initialization failures |
| --- | ---: | ---: |
| Fork HIP (native FP64 accelerated paths) | 0 | 0 |
| Fork HIP (default FP64 emulation policy) | 0 | 0 |

Convergence is checked independently from the SCF trace and the unmodified stopping predicate, together with early exit, successful process status, complete finite energy/force outputs and the normal completion marker. A completion marker or reaching the iteration cap alone cannot pass. The SCF predicate tests band-energy change and mixing residual; it is not an absolute total-energy error bound. Diagnostics print 12 decimal places, so a displayed `dUele=1e-12` may round an internal value below the threshold.

Numerical regressions check VNA/force products and local/halo force/stress derivatives with the input switch both on and off. Force batches, allocation failure and recovery fixtures pass in both settings; grid/orbital policy tests confirm the switch still disables optional FP32/double-float paths. All 17 unified build/projection checks pass. Existing test fixtures were updated for the current optional SIMD argument and typed environment-reader dependencies.

## Records

[CSV](../benchmarks/results/radeon_cghelgeo_coldstart_20261108_8x2.csv) and [JSON](../benchmarks/results/radeon_cghelgeo_coldstart_20261108_8x2.json) include the independent cold-start SCF histories, full forces, energy components, all pairwise comparisons, commands, settings, source correction patch and hashes. The separate correction-check result is labeled by phase. Raw logs, inputs, source/build records, regression checks and private installed binaries are retained under `work/codex_cghelgeo_coldstart_20261108_8x2/`.

The last-stage mixing transition is part of each independent cold-start SCF run. No common near-converged density is introduced at that boundary.

These accuracy checks do not replace the earlier 16×2 timing benchmarks.

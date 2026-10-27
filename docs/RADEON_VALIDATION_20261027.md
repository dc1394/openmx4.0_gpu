# Radeon verification and density-kernel tuning (2026-10-27)

Validated `refactor/cpp-kokkos-core` at `16540c81` plus this change on a
Radeon RX 9060 XT (gfx1200, 16 GiB), Ryzen Threadripper 3970X, ROCm SDK 10.1
(Clang/Flang 24), Open MPI and AOCL 5.2. Jobs used at most 16 CPU cores.
The build is Release with HIP Core and the private host-SIMD library.
`OPENMX_COPY_TO_WORK=OFF`; the installed/work executable was not replaced.

## Changes

The HIP CSR density gather now specializes the one- and two-component spin
cases. This removes runtime spin tests and unused live accumulators inside
the orbital loops. The generic kernel remains for three/four components;
term order, double-float arithmetic, FP64 behavior and the public interface
are preserved. The change adds no mutable application global state and does
not alter CPU kernels or calculation-context ownership. The settings/structure
classifications and ownership audit baselines are therefore unchanged.

The standalone HIP GEMMul8 checker previously failed to link with undefined
`openmx_env_at`, `openmx_env_flag_at` and `openmx_env_ulong_at`. Its runner now
links the production environment registry, and works from the backend source
or projected native tree. The density checker also has a standalone runner
that links the environment/precision readers and is registered as the
`hip-density-precision` CTest. It checks finite values, actual bit equality,
80-orbital collinear paths, the generic three-component boundary and checksums.

The common GEMMul8 submodule was initially a clean v3.5.2 checkout despite the
parent pin being v3.5.4. It was synchronized to the existing pin
`f8fdf0b2a5eafd4dbbfab6be8489d3356e0da06a` before the build; this does not change
the parent dependency pin.

## Numerical verification

All 60 completed calculations passed finite-energy/force checks and bundled
reference comparisons, using a 1e-7 tolerance for energy (Ha) and every force
component (Ha/bohr). Completion markers alone were not used as acceptance.
Only `level.of.stdout=2` was changed in isolated copies of the suite inputs.

| Run | Cases | Maximum abs energy difference (Ha) | Maximum abs force difference (Ha/bohr) |
| --- | ---: | ---: | ---: |
| baseline-gpu-L-focused-4x4 | 2 | 2.07e-09 | 2.87e-10 |
| baseline-gpu-S-4x4 | 14 | 2.63e-10 | 2.96e-09 |
| final-cpu-S-4x4 | 14 | 4.2e-11 | 7.9e-11 |
| final-gpu-L-4x4 | 2 | 2.21e-09 | 2.87e-10 |
| final-gpu-S-4x4 | 14 | 3.38e-10 | 2.96e-09 |
| final-gpu-S-8x2 | 14 | 7e-11 | 2.96e-09 |

`S` is all 14 `-runtest` inputs. `L` is **only Mn12_148_F and Pt63** from
`-runtestL`, not the complete L suite. The initial GGFF run was intentionally
stopped during SCF 14 to focus this change's large-case validation on collinear
density. Its logs and stop reason are retained under `archives/`; it is not a
PASS. L2/L3 were not run in this verification.

The GPU runs use `OPENMX_GPU=1`, MAGMA/Ozaki2 and cluster/noncollinear-cluster
GPU crossover thresholds of 1 to exercise small solves. Other solver thresholds
remain at their defaults: GPU matrix construction is distinct from a global
dense CPU eigensolver fallback. The completed GPU S runs each recorded 432
successful MAGMA calls. The L runs recorded 73 before / 75 after, all successful.
The CPU run uses the HIP executable with `OPENMX_GPU=0`, not a separate CPU-only
build; it reports AVX2 and has no GPU eigensolver calls or GPU-accelerated stages.

Before/after comparison of all 16 selected inputs gives maximum differences
of **6.01e-10 Ha** and **6.6e-11 Ha/bohr**. The 4x4/8x2 GPU comparison and
CPU/GPU comparison also pass for all 14 S cases. Matching Core and host-SIMD
libraries are byte-identical before/after this optimization.

Additional checks:

- 58 focused Release Core/host-SIMD tests and 58 ASan/UBSan tests pass.
- Two native OLG/density ownership tests also pass with the sanitized Core.
- 17 native/HIP regressions pass, including real device reductions and capacity
  recovery. Two SDK-dependent probes initially skipped with the default SDK
  path; rerunning with the actual ROCm path passed both without skips.
- Density preparation passes 16 real-MPI rounds, including empty ranks,
  mixed CSR/OTF, allocation failure and collective agreement.
- The production local-density fixture passes 20 resident/staged, DF/FP64
  configurations for the refactored ownership code.
- GEMMul8 3.5.4 passes 99 checks with timing disabled and 99 with timing enabled;
  both produce checksum `220cb9ba58e97830`. This includes real/complex operations,
  workspace limits, trim and native fallback. The CUDA-only optional large-N
  launch-limit test was not run here.
- The final density CTest passes all nine shapes against independent long-double
  references, including both register and memory rows and LDS splitting.

## Density-kernel measurement

Eight alternating before/after process pairs use the same nine-case fixture.
The first pair is warm-up; the table gives medians of the other seven.
No application or compiler jobs run concurrently. GPU event time includes
in-place density-matrix splitting and CSR gather, excluding allocations and
host-to-device copies. All nine CSR checksums match across both binaries and
every repetition. The changed collinear cases improve by about 1–5%; generic
cases are within 0.4% of the baseline.

| Case | Before (ms) | After (ms) | Speedup |
| --- | ---: | ---: | ---: |
| collinear, unpolarized | 0.568 | 0.541 | 1.050x |
| collinear, polarized | 2.443 | 2.360 | 1.035x |
| non-collinear | 2.157 | 2.155 | 1.001x |
| non-collinear, 48 orbitals | 1.691 | 1.697 | 0.996x |
| polarized, 41 and 48 orbitals | 0.909 | 0.889 | 1.022x |
| non-collinear, 80 orbitals | 1.030 | 1.029 | 1.001x |
| unpolarized, 80 orbitals | 1.320 | 1.305 | 1.011x |
| polarized, 80 orbitals | 3.734 | 3.683 | 1.014x |
| three components | 0.248 | 0.247 | 1.004x |

This is a kernel-level improvement, not a demonstrated end-to-end speedup.
Mn12 took 66.61 / 67.03 s at 61 SCF iterations. Pt63 took 42.07 / 43.72 s,
but its SCF count changed from 68 to 70, so that pair does not isolate speed.
The application still contains atomic/non-bitwise-reproducible GPU paths;
the final numerical comparisons above remain within tolerance. No MI300A
runtime or performance claim is made.

## Reproduction and evidence

The local build directory is `build/hip-application-20261025/openmpi-aocl`.
Logs, isolated inputs/outputs, frozen baseline binaries/libraries, commands,
source/binary hashes and the complete machine-readable result are in
`work/codex_radeon_20261027/verification.json` and its neighboring files.
`density-paired/comparison.json` retains every timing sample and both binary
hashes. The `openmx-radeon-*.py` scripts in that work directory retain the local
SDK paths and environment used for the runs.

With the configured HIP build and a working GPU:

```sh
cmake --build build/hip-application-20261025/openmpi-aocl --target openmx --parallel 12
ctest --test-dir build/hip-application-20261025/openmpi-aocl \
  -R '^hip-density-precision$' --output-on-failure
```

For standalone density checks, set `ROCM` to the SDK root and `ARCH=gfx1200`,
then run `bash source/backends/hip/tools/run_check_density_df_hip.sh`.
For GEMMul8, use the projected `native/tools/run_check_gemmul8_hip.sh`, or set
`BRIDGE_OBJ` and `GEMMUL8_LIB` to the matching build artifacts when invoking
`source/backends/hip/tools/run_check_gemmul8_hip.sh` directly.

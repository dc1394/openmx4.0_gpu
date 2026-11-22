# Radeon RX 9060 XT / Threadripper 3970X — 2026-11-21, 8 MPI × 4 OpenMP

Measured 2026-11-21T13:14:55.505344+09:00 – 2026-11-21T18:53:09.215819+09:00 JST; revision `456731c912093299b6756ae081c55533e433b0cf`. One measurement per route and test; jobs ran sequentially. All eight MPI ranks were audited with four disjoint physical cores per rank, four OpenMP threads and four math-library threads (32 cores total, SMT disabled).

| Route | Build / runtime |
|---|---|
| Upstream CPU | Existing OpenMX 4.0 + official 4.0.1 patch; Intel icx/ifx 2026.0, Intel MPI 2021.18, threaded LP64 MKL, ELPA2. Executable hash matches the previously verified upstream build. |
| Fork CPU | Release HIP build; ROCm core-10.1 Clang/Flang, Open MPI 5.0.9, AOCL 5.2 OpenMP, ELPA2; `OPENMX_GPU=0`. Host SIMD selects AVX2 on this CPU. |
| Fork HIP | Same frozen executable/core/SIMD libraries; `OPENMX_GPU=1`, `scf.eigen.lib gpusolver`, `HSA_ENABLE_SDMA=0`; one gfx1200 Radeon RX 9060 XT, 16 GiB. NVIDIA MPS does not apply. |

Original geometry, grids, precision options, stdout, SCF/MD limits and convergence criteria are retained. Only `DATA.PATH` and `scf.eigen.lib` differ in measurement copies. Every job starts in a fresh directory without copied restart. All 44 basis/pseudopotential hashes stayed unchanged. Compiler, MPI and numerical-library differences are included in upstream comparisons.

Suite totals sum native per-case elapsed times; named inputs use maximum-rank `Total Computational Time`. These exclude MPI launch/shutdown; wall times are separately recorded. GPU-route totals include CPU work and any CPU fallback selected by the application.

| Test | Upstream CPU (s) | Fork CPU (s) | Fork HIP (s) | Upstream / CPU | Upstream / HIP | CPU / HIP |
|---|---:|---:|---:|---:|---:|---:|
| runtest (14 cases) | 54.220 | 52.190 | 63.070 | 1.039 | 0.860 | 0.827 |
| runtestL (16 cases) | 5662.590 | 3710.610 | 4025.250 | 1.526 | 1.407 | 0.922 |
| cghelgeo.dat | 2057.865 | 2392.246 | 1121.985 | 0.860 | 1.834 | 2.132 |
| sidia333_nc_cluster.dat | 514.484 | 370.987 | 181.548 | 1.387 | 2.834 | 2.043 |

All timings are single measurements. Ratios greater than one mean the denominator route is faster. Different SCF/MD counts can affect whole-input times. Case tables show total SCF steps [actual MD steps].

## runtest (14 cases) case details

| Input | Upstream CPU (s) | Fork CPU (s) | Fork HIP (s) | CPU / HIP | SCF [MD] steps (upstream / CPU / HIP) | Max pairwise ΔE (Ha) | Max pairwise ΔF (Ha/bohr) | Agreement |
|---|---:|---:|---:|---:|---|---:|---:|---|
| Benzene | 2.280 | 3.080 | 2.900 | 1.062 | 19 [1] / 19 [1] / 19 [1] | 9.948e-13 | 0.000e+00 | PASS |
| C60 | 4.910 | 7.770 | 8.790 | 0.884 | 27 [1] / 27 [1] / 27 [1] | 2.802e-11 | 1.000e-12 | PASS |
| CO | 3.470 | 2.770 | 11.470 | 0.241 | 16 [1] / 16 [1] / 16 [1] | 0.000e+00 | 2.100e-11 | PASS |
| Cr2 | 3.850 | 3.080 | 3.020 | 1.020 | 21 [1] / 21 [1] / 21 [1] | 2.984e-12 | 1.880e-10 | PASS |
| Crys-MnO | 7.140 | 5.870 | 6.680 | 0.879 | 37 [1] / 39 [1] / 37 [1] | 4.790e-10 | 6.400e-10 | PASS |
| GaAs | 12.810 | 12.290 | 11.490 | 1.070 | 13 [1] / 13 [1] / 13 [1] | 1.600e-11 | 8.000e-12 | PASS |
| Glycine | 2.140 | 1.770 | 2.850 | 0.621 | 20 [1] / 20 [1] / 20 [1] | 0.000e+00 | 1.000e-12 | PASS |
| Graphite4 | 1.250 | 0.860 | 1.150 | 0.748 | 14 [1] / 14 [1] / 14 [1] | 2.000e-12 | 1.000e-12 | PASS |
| H2O | 1.730 | 1.130 | 1.480 | 0.764 | 31 [2] / 31 [2] / 31 [2] | 1.002e-12 | 1.000e-12 | PASS |
| H2O-EF | 1.840 | 1.470 | 1.750 | 0.840 | 17 [1] / 17 [1] / 17 [1] | 9.983e-13 | 1.000e-12 | PASS |
| HMn | 6.310 | 6.310 | 6.300 | 1.002 | 26 [1] / 26 [1] / 26 [1] | 9.948e-13 | 0.000e+00 | PASS |
| Methane | 1.240 | 0.900 | 1.320 | 0.682 | 14 [1] / 14 [1] / 14 [1] | 6.200e-11 | 7.000e-12 | PASS |
| Mol_MnO | 3.630 | 3.660 | 2.790 | 1.312 | 26 [1] / 26 [1] / 26 [1] | 9.948e-13 | 1.000e-12 | PASS |
| Ndia2 | 1.620 | 1.230 | 1.080 | 1.139 | 13 [1] / 13 [1] / 13 [1] | 4.000e-12 | 1.000e-12 | PASS |

## runtestL (16 cases) case details

| Input | Upstream CPU (s) | Fork CPU (s) | Fork HIP (s) | CPU / HIP | SCF [MD] steps (upstream / CPU / HIP) | Max pairwise ΔE (Ha) | Max pairwise ΔF (Ha/bohr) | Agreement |
|---|---:|---:|---:|---:|---|---:|---:|---|
| 5_5_13COb2 | 63.720 | 59.700 | 58.790 | 1.015 | 66 [1] / 67 [1] / 66 [1] | 6.110e-10 | 1.570e-10 | PASS |
| B2C62_Band | 785.290 | 581.610 | 626.130 | 0.929 | 203 [11] / 203 [11] / 203 [11] | 1.190e-10 | 4.488e-08 | PASS |
| CG15c-DC-LNO | 151.680 | 107.380 | 116.910 | 0.918 | 42 [1] / 42 [1] / 42 [1] | 1.326e-09 | 1.560e-10 | PASS |
| DIA512-1 | 88.730 | 74.720 | 97.660 | 0.765 | 29 [1] / 29 [1] / 29 [1] | 7.321e-10 | 7.322e-09 | PASS |
| FeBCC | 141.710 | 130.790 | 136.280 | 0.960 | 62 [1] / 62 [1] / 62 [1] | 9.370e-10 | 7.400e-11 | PASS |
| GEL | 60.270 | 49.120 | 45.100 | 1.089 | 58 [1] / 58 [1] / 58 [1] | 2.150e-10 | 4.410e-10 | PASS |
| GFRAG | 27.990 | 25.370 | 27.410 | 0.926 | 60 [1] / 60 [1] / 60 [1] | 1.380e-10 | 6.930e-10 | PASS |
| GGFF | 1351.160 | 976.110 | 869.400 | 1.123 | 101 [1] / 100 [1] / 101 [1] | 7.981e-11 | 1.390e-10 | PASS |
| MCCN | 168.450 | 166.830 | 471.950 | 0.353 | 67 [1] / 67 [1] / 67 [1] | 3.402e-10 | 3.700e-11 | PASS |
| Mn12_148_F | 62.780 | 59.530 | 52.480 | 1.134 | 61 [1] / 61 [1] / 61 [1] | 1.864e-11 | 4.530e-10 | PASS |
| N1C999 | 2419.340 | 1159.360 | 1192.430 | 0.972 | 66 [1] / 58 [1] / 58 [1] | 1.430e-09 | 3.300e-11 | PASS |
| Ni63-O64 | 61.170 | 50.050 | 51.040 | 0.981 | 69 [1] / 62 [1] / 62 [1] | 2.752e-09 | 5.260e-10 | PASS |
| Pt63 | 43.400 | 38.940 | 40.240 | 0.968 | 73 [1] / 74 [1] / 67 [1] | 1.252e-09 | 1.620e-10 | PASS |
| SialicAcid | 11.940 | 10.660 | 12.880 | 0.828 | 29 [1] / 29 [1] / 29 [1] | 5.002e-12 | 4.000e-12 | PASS |
| ZrB2_2x2 | 154.380 | 137.760 | 170.520 | 0.808 | 49 [1] / 49 [1] / 49 [1] | 1.510e-10 | 3.870e-10 | PASS |
| nsV4Bz5 | 70.580 | 82.680 | 56.030 | 1.476 | 57 [1] / 57 [1] / 57 [1] | 1.110e-10 | 5.290e-10 | PASS |

## cghelgeo.dat case details

| Input | Upstream CPU (s) | Fork CPU (s) | Fork HIP (s) | CPU / HIP | SCF [MD] steps (upstream / CPU / HIP) | Max pairwise ΔE (Ha) | Max pairwise ΔF (Ha/bohr) | Agreement |
|---|---:|---:|---:|---:|---|---:|---:|---|
| cghelgeo | 2057.865 | 2392.246 | 1121.985 | 2.132 | 60 [1] / 57 [1] / 58 [1] | 1.318e-07 | 5.428e-08 | FAIL |

## sidia333_nc_cluster.dat case details

| Input | Upstream CPU (s) | Fork CPU (s) | Fork HIP (s) | CPU / HIP | SCF [MD] steps (upstream / CPU / HIP) | Max pairwise ΔE (Ha) | Max pairwise ΔF (Ha/bohr) | Agreement |
|---|---:|---:|---:|---:|---|---:|---:|---|
| sidia333_nc_cluster | 514.484 | 370.987 | 181.548 | 2.043 | 25 [1] / 25 [1] / 25 [1] | 2.290e-10 | 1.640e-10 | PASS |

## Numerical checks

All **96 completed case outputs** contain finite energies and all **41,328 force components**. Rank binding, runtime libraries, input integrity and frozen-binary hashes passed. Pairwise comparisons: **95/96 PASS**. Bundled S/L reference comparisons: **90/90 PASS**.

Agreement requires the same geometry/grids, absolute total-energy difference ≤1e-7 Ha and maximum force-component difference ≤1e-7 Ha/bohr. Completion markers alone do not establish PASS. All failed comparisons remain in the records. The original cghelgeo criterion is 1e-7; sidia333 retains criterion 1e-13 and its 25-SCF cap, so its capped timing does not establish convergence.

Failed route-pair comparisons:

- cghelgeo / cghelgeo, fork-cpu vs fork-gpu: |ΔE|=1.318e-07 Ha, max |ΔF|=5.428e-08 Ha/bohr.

## Records

- `case-results.csv`, `run-results.csv`: all per-case computational/native times, MPI-job wall times, SCF counts and memory use.
- `comparisons.csv` / JSON and `bundled-reference-comparisons.json`: route-pair and reference checks.
- `manifest.json`, `original-inputs.json`, `data-files.json`, `input-audit.json`, `execution-audit.json`: source/build, inputs/data, environments, rank affinity and loaded libraries.
- `gpu-path-audit.json`: GPU acceleration and fallback messages; matrix construction and dense diagonalization must be interpreted separately.
- `logs/`: compressed timing logs and all final outputs, commands and execution audits.
- `benchmark.py`, `audit.py`: reproduction and final-audit scripts.
- Full raw calculations, restart files, frozen binaries and CPU/memory samples: `/home/kawai/openmx4.0_gpu/work/codex_benchmark_20261121_8x4`.

## Stack-limit workaround and failed attempt

The first upstream cghelgeo attempt used the default 8 MiB process stack and crashed in its first eigensolve (SIGSEGV, no completed output). The same failure and workaround were already documented in `docs/CGHELGEO_STRICT_SCF.md`. It was rerun with **unlimited process stack and `OMP_STACKSIZE=256M`**, retaining the exact binary, scientific input and 8 MPI × 4-thread binding. Failed logs and attempt history are preserved; only the completed retry contributes a benchmark time. Remaining primary launches after the recorded stack adjustment inherited unlimited process stack. Earlier successful jobs are retained. Stack limits do not change arithmetic or SCF conditions. See `attempt-history.json`, `failed-attempts/` and `stack-limit-adjustment.json`.

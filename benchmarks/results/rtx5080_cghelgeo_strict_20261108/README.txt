Strict cghelgeo accuracy validation (RTX 5080, started 2026-11-08)

Six complete calculations at scf.criterion=1e-10, 18 MPI x 2 OpenMP: four requested modes and two matched-seed controls.
Geometry, cell, PAO/VPS, k mesh, cutoff, mixing and the 200-SCF limit are preserved.
The original MD.Type=Opt and MD.maxIter=1 are retained.
The upstream case uses scf.restart on with density/Hamiltonian from its same-geometry probe.
The three original fork cases start fresh. Both added controls use exactly the same converged default-GPU restart.
All six final runs converge and complete normally. Initialization differences preclude performance comparisons.
The original four-mode comparison is NOT all-pass at the unchanged 1e-7 Ha energy gate.
Full-FP64 energy-gate failures are retained in analysis.json and comparisons.csv.
All six comparisons with inherited projector-product rounding held fixed pass the same gates.
All 15 pairwise comparisons, including failures, appear in control-comparisons.csv.
The force check covers all 1950 components. Printed rounding bounds are included.

A preliminary upstream 1e-12 run exited normally after 200 SCF steps without convergence.
Its final dUele was about 2.7e-10 Ha. The common comparison criterion was set to 1e-10,
1000 times tighter than the original 1e-7 input. Energy/force comparison tolerances remain 1e-7.
The failed convergence probe is preserved under preliminary/ and is not counted as converged.

The CPU path uses the same CUDA executable with GPU execution disabled and AVX-512 selected.
Native FP64 disables FP32/DF arithmetic, GEMMul8 and cuSOLVER emulation.
scf.fp32.enable off also promotes inherited FP32 projector products; this changes the numerical contract of the original code.
The additional native-FP64 control preserves those products but forces FP64 dense solvers/GEMMs and grid/orbital evaluation.
The matched full-FP64 control differs only in scf.fp32.enable, with identical restart-seed hashes and numerical environment.
Existing float table storage is retained. The default GPU case keeps all default arithmetic policies.
cuSOLVER API queries and actual Xsyevd/Xsyevdx calls verify native and emulated modes;
runtime-validation.json records the matrix sizes and arithmetic types.
A 3x3 Hermitian eigenproblem verifies both forwarded APIs and both modes independently.
The forwarding audit shim and diagnostic logging are enabled for every GPU configuration.
Every GPU run uses a private MPS service, stopped after completion.

See provenance.json for the measured source commit, data/library/binary hashes and RPATH snapshot change.
See input.dat and result.json in each configuration directory for exact inputs, commands and environment.
Full stdout and OpenMX output files are gzip-compressed; runtime-library and rank-placement records accompany them.
Capture scripts require the original workstation paths, bin/lib/DFT_DATA19/runs layout and matching binaries/data.
This compressed evidence archive is not a standalone runnable benchmark installation.
Recorded times are diagnostic single runs and should not replace the README performance tables.

SCF steps and total energies:
upstream: 2 final SCF (+200 preparatory), -4153.148190613880 Ha; same-geometry 200-SCF 1e-12 probe restart
cpu_avx512: 95 final SCF (+0 preparatory), -4153.148190613648 Ha; fresh; no restart
gpu_native_fp64: 83 final SCF (+0 preparatory), -4153.148190776506 Ha; fresh; no restart
gpu_default: 82 final SCF (+0 preparatory), -4153.148190613619 Ha; fresh; no restart
gpu_native_legacy_products: 2 final SCF (+0 preparatory), -4153.148190613762 Ha; same-geometry converged default-GPU density/Hamiltonian; scf.restart on
gpu_full_fp64_same_seed: 60 final SCF (+0 preparatory), -4153.148190776526 Ha; same-geometry converged default-GPU density/Hamiltonian; scf.restart on

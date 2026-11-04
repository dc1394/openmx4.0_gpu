Neoverse N3 host scalar / SVE128 benchmark, 8 MPI x 1 OpenMP (2026-11-04)

Input: benchmarks/inputs/sidia333_nc_cluster.dat, 216 Si atoms, NC Cluster.
Measured source commit: 6ad8d53e75db6453e70d7bccf34725c541fbfbae.
CPU: 8-core ARM Neoverse-N3, Linux AArch64, SVE vector length 128 bits.
No GPU. Release CPU build; SERIAL Kokkos core, application OpenMP enabled.
GCC/G++/gfortran 12.2.0, Open MPI 4.1.4, CMake 3.31.6, OpenBLAS 0.3.21
(pthread, LP64), ScaLAPACK 2.2.1, FFTW 3.3.10, bundled ELPA 2026.02.002.

The same installed binary and matching Core/SIMD libraries were used for
both modes: OPENMX_HOST_SIMD=scalar and OPENMX_HOST_SIMD=sve128, OPENMX_GPU=0,
scf.eigen.lib=elpa2. "No SIMD" here means OpenMX's host scalar kernels;
compiler/BLAS/ELPA optimizations are shared, not disabled throughout the code.
Each of eight MPI ranks was bound to its own CPU core. The rank launcher
checked the selected SIMD implementation and fixed/inherited SVE VL=128
before exec. OpenMP and BLAS thread controls were 1; OpenMX also received -nt 1.
These rank audits establish launch settings, not an instrumented count of
threads inside every numerical library.

One warmup per mode, followed by three timed repetitions per mode. Runs were
sequential and alternated scalar/SVE128 order by repetition. Warmups are
excluded from timing statistics. Native times are the maximum MPI-rank
Elapsed.Time values merged into result.out, at 0.001 s precision. Launcher
wall times are also retained in timings.csv and results.json.gz.

Timed repetition | Scalar (s) | SVE128 (s)
1                | 1739.999   | 1553.806
2                | 1738.314   | 1550.315
3                | 1737.031   | 1550.631
Median           | 1738.314   | 1550.631

Median speedup: 1.1210365x; elapsed-time reduction: 10.7968% (187.683 s).

Phase                     | Scalar median (s) | SVE128 median (s) | Speedup
Set_Density_Grid          | 251.606           | 172.185           | 1.461x
Set_Hamiltonian           | 239.281           | 178.921           | 1.337x
Set_ProExpn_VNA           | 22.736            | 16.478            | 1.380x
Force                    | 79.462            | 71.388            | 1.113x
Diagonalization          | 1111.650          | 1080.086          | 1.029x

Phase statistics use each phase's reported maximum rank time. Separate
phase medians/maxima need not add to the overall time. Density-grid and
Hamiltonian construction show the largest savings. Diagonalization still
accounts for about 69.7% of SVE128 total time. The timing difference in that
phase alone does not establish a direct SVE optimization of ELPA/BLAS.

Numerical checks
----------------
All eight runs exited normally. All rank audits passed. Final total energy
was -887.30960736914 Ha. All printed energies, all 648 force components,
coordinates, grids (140 x 140 x 140), and 25-step SCF histories were equal
across both modes and their repetitions. Maximum printed delta E = 0 Ha;
maximum printed delta F = 0 Ha/bohr. This is equality at output precision,
not a comparison of unprinted floating-point bits. The comparison thresholds
were 1e-7 Ha for energies and 1e-7 Ha/bohr for forces.

IMPORTANT: all eight runs reached scf.maxIter=25 WITHOUT SCF convergence.
The original scf.criterion=1e-13, mixing, structure and MD settings were kept.
Numerical agreement of this fixed 25-iteration workload is established;
agreement of converged ground-state solutions is not established.
input-overrides.json lists the path/output/solver/restart changes. Input and
binary/data hashes are retained in manifest.json and results.json.gz.
cpu-numerical-toolchain and core-host-sve-abi CTest checks both passed.

Independent force-component audit
---------------------------------
The original result.out files were re-read and checked against their recorded
SHA256 hashes. The complete coordinates.forces block was extracted from each
file independently of the earlier binary-float parser. Decimal comparisons
retain all 12 printed fractional digits, and token/byte comparisons also
retain the two negative-zero components.

All 28 pairs of the eight runs agree: 16 scalar/SVE128 pairs and 12 same-mode
repeat pairs (15 pairs when considering only the six timed runs). Maximum
absolute component difference, component RMS difference, and maximum atom
force-vector difference are all 0 Ha/bohr. All 648-component tables are
byte-identical, including atom identities, printed coordinates and signed
zeros. Their common block SHA256 is:
25d5ccdf9882778ab2be29da1af91a7c75e6ed60167ce39f84b49a1abe35de8b

This is a nontrivial comparison: 646/648 reference components are nonzero;
the largest absolute component is 9.206316e-6 Ha/bohr. The largest atomic
force norm is 1.593541393689e-5 Ha/bohr (atom 82). Norms and net forces in
force-audit.json are calculated from the rounded printed components.

source/MD_pac.c prints -Gxyz[k][17], -Gxyz[k][18], -Gxyz[k][19] with %15.12f
in Hartree/Bohr, giving a printed quantum of 1e-12 Ha/bohr. Sub-quantum
differences or in-memory bitwise equality cannot be inferred. These remain
forces from the unconverged 25-SCF workload, not validated converged forces.

Files and reproduction
----------------------
results.json.gz: full original parsed records, including all forces, SCF
  histories, coordinates, commands, rank audits, hashes and pairwise checks.
timings.csv: all warmup/timed records, native and launcher wall times.
phase-timings.csv: all reported phase medians and speedup ratios.
logs/*.log.gz: original stdout/stderr for all eight runs.
input.dat.gz: exact common runtime input, including original whitespace
  (absolute DATA.PATH is machine-specific; decompress with gzip -dc).
environment.txt, manifest.json: toolchain, configuration and provenance.
analysis.json, analysis.txt: independent analysis and verification summary.
force-blocks.json.gz: exact extracted force blocks from all eight original
  outputs, their block hashes and full-output provenance hashes.
force-audit.json, force-audit.txt: detailed force magnitudes and all 28
  component/vector/RMS comparisons, including repeatability checks.
force-components.csv: all 648 reference components and their ranges across
  eight runs; force-pairs.csv: pairwise difference statistics.
audit_forces.py: independent Decimal extraction/audit and reproducibility
  check. test_audit_forces.py tests one-quantum errors, cancelling errors,
  signed zeros, coordinate changes, missing/duplicate rows and nonfinite data.
reproduce.sh: corrected script used for this completed campaign. Run as an
  ordinary user; sudo is used only for apt dependencies. A new invocation
  creates a fresh output directory and does not reuse the published timings.

From the repository root, validate the published records without running MPI:
  python3 benchmarks/results/neoverse_n3_20261104_8x1/validate.py

Verify only the force audit from the committed blocks and numerical archive:
  python3 benchmarks/results/neoverse_n3_20261104_8x1/audit_forces.py

Run its parser/comparison regression checks (no MPI calculations):
  python3 benchmarks/results/neoverse_n3_20261104_8x1/test_audit_forces.py -v

To re-extract the force audit from the original local campaign outputs:
  python3 benchmarks/results/neoverse_n3_20261104_8x1/audit_forces.py --collect /path/to/campaign
This verifies the full original output hashes before extracting the blocks.

To reproduce the complete build and measurements on Debian 12 AArch64:
  bash benchmarks/results/neoverse_n3_20261104_8x1/reproduce.sh

If dependencies are already installed:
  OPENMX_BENCH_SKIP_APT=1 bash benchmarks/results/neoverse_n3_20261104_8x1/reproduce.sh

The script checks out the measured commit in ~/openmx4.0_gpu by default;
set OPENMX_BENCH_ROOT to a fresh absolute path when running from an existing
checkout with a different commit. It preserves the 25-SCF input by default.

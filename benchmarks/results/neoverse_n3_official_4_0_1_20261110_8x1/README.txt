Official OpenMX 4.0 + patch4.0.1, Neoverse-N3, 8 MPI x 1 OpenMP (2026-11-10)

This measures the official CPU distribution, not the modified code in this
GitHub repository. It provides an official-source baseline for the published
neoverse_n3_20261104_8x1 campaign. The execution workload and rank/thread
settings match; the software builds and numerical-library versions differ.

Input: sidia333_nc_cluster.dat, 216 Si atoms, NC Cluster, GGA-PBE, 200 Ry.
CPU: 8-core ARM Neoverse-N3, Linux AArch64, inherited SVE vector length 128 bits.
No GPU. Eight MPI ranks, each bound to its own core; OpenMP and BLAS thread
controls were 1, and OpenMX received -nt 1. The preserved rank launcher checks
these settings immediately before exec. MPI helper threads may still exist;
these audits do not count threads inside every library throughout a run.

One warmup followed by ONE timed run, both sequential, using identical input
bytes in fresh directories with scf.restart=off. The warmup is excluded from
the measurement. No additional timed calculations were launched. This is a
single timing observation; no timing spread or statistical significance is
claimed.

Run          Native elapsed (s)    Launcher wall (s)    SCF iterations
warmup-01             1714.709             1715.884                25
timed-01              1721.589             1722.723                25

Measured result: 1721.589 s = 28 min 41.589 s.
Native elapsed is Elapsed.Time in result.out: the maximum MPI-rank elapsed
time merged by OpenMX, at 0.001 s precision. Launcher wall time also includes
startup and termination. Individual phase times are maximum rank times;
their sum need not equal the overall elapsed time.

Phase                         Official timed (s)    Published scalar/SVE128 medians (s)
Set_Density_Grid                          166.184                  251.606 / 172.185
Set_Hamiltonian                           212.512                  239.281 / 178.921
Set_ProExpn_VNA                            40.975                   22.736 /  16.478
Force                                      77.698                   79.462 /  71.388
Diagonalization                          1156.435                 1111.650 / 1080.086
Overall                                  1721.589                 1738.314 / 1550.631

For context, published scalar median / official single run = 1.009715;
published SVE128 median / official single run = 0.900698. The published values
are three-run medians. These ratios compare different builds/toolchains;
they do not isolate host SIMD, an ELPA change, or a statistically significant
performance difference.

Numerical validation
--------------------
Both runs exited 0 and reported normal completion. All eight rank audits
passed for each run. Grids were 140 x 140 x 140. The total energy was
-887.309607369157 Ha. All printed total energies, all 648 force components,
coordinates and the complete 25-step SCF histories agreed between warmup and
timed run at output precision; this is not in-memory bitwise equality.

Compared with the published campaign, maximum absolute total-energy delta
was 1.693933882052079e-11 Ha, and maximum absolute force-component delta was
1.1300000000078988e-10 Ha/bohr. Both are below the reference tolerances of
1e-7 Ha and 1e-7 Ha/bohr. Raw outputs are retained for independent re-parsing.

Both runs reached scf.maxIter=25 WITHOUT SCF convergence, as in the reference.
scf.criterion=1e-13, the structure, mixing and MD settings were preserved.
This validates the fixed 25-iteration workload, not converged ground-state
solutions or converged forces. input-overrides.json lists every input change.

Build and provenance
--------------------
Official source: https://www.openmx-square.org/openmx4.0.tar.gz
Official patch:  https://www.openmx-square.org/bugfixed/26May08/patch4.0.1.tar.gz
Patch procedure: https://www.openmx-square.org/bugfixed/26May08/README.txt

The patch replaces source/Band_DFT_Dosout.c and source/Mulliken_Charge.c and
work/GaAs.dat. Their hashes match the official patch. Comparing 1785 original
files with the source archive found only those three patched files and the
configured makefile changed. The version banner remains 4.0 because the
official patch does not update the version string.

Debian 13, GCC/gfortran 14.2.0, Open MPI 5.0.7, OpenBLAS 0.3.29 (pthread,
LP64), ScaLAPACK 2.2.2, FFTW 3.3.10, official bundled ELPA 2018.05.001.
The published comparison used Debian 12, GCC/gfortran 12.2.0, Open MPI 4.1.4,
OpenBLAS 0.3.21, ScaLAPACK 2.2.1, FFTW 3.3.10 and ELPA 2026.02.002.
The official source has no Kokkos or OPENMX_HOST_SIMD scalar/SVE128 toggle.

build/makefile.gz preserves the actual GCC/OpenMPI/OpenBLAS build settings:
-O3, application OpenMP enabled, no -ffast-math or -march=native. -Dnosse
selects the upstream portable Krylov implementation on AArch64. Legacy code
uses -fcommon, -Wno-implicit-function-declaration and Fortran
-fallow-argument-mismatch. Fortran modules were built serially in upstream
object order, followed by make -j4 openmx and make install without sudo.

Files
-----
results.json.gz: complete finalized two-run records, including coordinates,
  all forces, SCF histories, timings, launch commands, rank audits and hashes.
summary.json, timings.csv, phase-timings.csv: measurement and reference values.
input.dat.gz: exact common runtime input; original.dat.gz: unmodified source
  input at the published measured commit, SHA256 9c5466ab...9decd0.
logs/*.log.gz: original, unchanged stdout/stderr from both MPI launches.
outputs/*.out.gz: original, unchanged result.out files from both runs.
manifest.json: official source/patch URLs and hashes, binary/data hashes,
  flags, toolchain differences and decompressed evidence hashes.
environment.txt, installed-environment.txt, numerical-libraries.json:
  measured machine/toolchain, linked libraries and OpenBLAS configuration.
build/: actual makefile (gzip-compressed unchanged), official-patch hashes and source comparison record.
launcher/: actual launch scripts, including their original absolute paths.
  These are provenance, not portable entry points; adapt paths to replay.
validate.py, SHA256SUMS: offline verification without running calculations.

Generated restart/density/visualization files were deleted only after each
completed run; their names are recorded in results.json.gz. No binary or full
DFT data distribution is committed. Original local campaign bookkeeping is
retained on the measurement host; only the two completed runs are published.

Validate from the repository root (no OpenMX, MPI or numerical libraries needed):
  python3 benchmarks/results/neoverse_n3_official_4_0_1_20261110_8x1/validate.py

To replay with an independently installed official 4.0.1, decompress input.dat.gz
into a fresh run directory and change only DATA.PATH to that installation.
Adapt the two launcher script paths to the binary and rank launcher, then
run one warmup and one timed calculation in separate fresh directories.

Reference campaign:
  ../neoverse_n3_20261104_8x1/README.txt
  https://github.com/dc1394/openmx4.0_gpu/tree/main/benchmarks/results/neoverse_n3_20261104_8x1

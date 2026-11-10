# Intel CPU-only measurements on Core i9-10980XE (2026-11-04)

These are the historical measurements made on this computer on November 4,
2026, archived on November 10. No calculation was rerun for this publication.
The fork source revision was `9cb605dbbc892ea8c5d828c93c03a9d9be1d4147`.

## Build and execution

The CPU-only fork was built and installed in a separate directory with
Intel oneAPI 2026.0 (`icx`, `icpx`, `ifx`), Intel MPI 2021.18 and threaded
LP64 MKL 2026.0. The CMake choices were `OPENMX_GPU_BACKEND=CPU`,
`OPENMX_CPU_COMPILER=ICX`, `OPENMX_CPU_MPI=INTEL`,
`OPENMX_CPU_LINALG=MKL` and `OPENMX_CPU_MKL_THREADING=INTEL`.
The Release flags included `-O3 -DNDEBUG -xHost`. Kokkos 5.2.2 used its
Serial execution space, with the host SIMD library also compiled by Intel.
FFTW calls used MKL's FFTW-compatible API. The existing NVIDIA installation
in the original work directory was preserved.

The upstream reference is the existing `~/openmx4.0/work/openmx` executable,
copied unchanged and using the same Intel compiler/MPI/MKL versions.
The fork uses ELPA 2026.02.002; upstream uses ELPA 2018. Implementation,
linkage and precision policies also differ. This comparison does not isolate
the compiler's effect on performance.

The host has an Intel Core i9-10980XE, 18 physical cores with SMT (36 logical
CPUs), 256 GiB RAM and an RTX 5080. Both configurations use CPU execution
only, with **18 MPI ranks × 2 OpenMP threads** and **2 MKL threads**.
Rank i uses logical CPUs i and i+18, the two hardware threads of one core.
`OMP_DYNAMIC=FALSE`, `MKL_DYNAMIC=FALSE`, `OMP_PLACES=threads` and
`OMP_PROC_BIND=close` are common. The native MPI command uses `-np 18` and
OpenMX uses `-nt 2`. The exact environments and commands are retained in
each `result.json` and `build-commands.json`.

Each input/configuration was measured once in a fresh directory. The four
benchmark jobs ran sequentially after the build and preflight checks finished:
fork Intel then upstream for sidia333, followed by fork Intel then upstream
for cghelgeo.
There was no restart seed or input-specific warmup. Separate numerical and
threading probes were untimed and excluded from these measurements.
Only the absolute `DATA.PATH` and CPU `scf.eigen.lib=elpa2` selection were
adapted from the original inputs. Both configurations use byte-identical
prepared inputs and the same PAO/VPS data. Geometry, grids, mixing, SCF/MD
limits and convergence criteria were retained.

## Results

Primary times are the maximum-rank native `Total Computational Time` in
seconds. Wall times include MPI startup/shutdown and up to one two-second
monitoring interval. These are single observations, not averages or medians.

| Input | Upstream native (s) | Fork Intel native (s) | Upstream / fork | Upstream wall (s) | Fork wall (s) | SCF steps (upstream / fork) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| sidia333_nc_cluster | 882.152 | **632.439** | 1.395 | 886.317 | 634.790 | 25 / 25 |
| cghelgeo | 3361.253 | **3105.124** | 1.082 | 3365.348 | 3106.400 | 58 / 59 |

Both sidia333 runs reached the original 25-SCF cap **without convergence**.
The original criterion was `1e-13`; the reported times describe that capped
workload. Both cghelgeo runs converged at its original `1e-7` criterion.
All four processes exited successfully and printed normal completion markers.

The upstream sidia333 baseline here is the run made on November 4, paired
with the Intel fork measurement. The separate October 31 campaign reported
832.660 s; that value is not substituted into this campaign's ratios. The
upstream executable hashes and physical inputs match across those measurements,
but the source revision and toolchain of the fork comparison differ between
the campaigns. No claim about the cause of the timing variation is made.

## Accuracy and runtime checks

The historical analysis compares every force component and includes atom
identities, FFT grids and printed coordinates. Its energy and force-component
tolerances were `1e-7` Ha and `1e-7` Ha/bohr. Both input pairs pass.

| Input | Force components | Historical abs ΔE (Ha) | Historical max abs ΔF (Ha/bohr) |
| --- | ---: | ---: | ---: |
| sidia333_nc_cluster | 648 | 4.502e-11 | 1.710e-10 |
| cghelgeo | 1950 | 5.4114e-8 | 9.378e-9 |

The original `analysis.json` and `comparison.csv` retain the floating-point
analysis unchanged. The archive verifier independently parses the printed
energy/forces using decimal arithmetic and includes printed-rounding bounds;
its results are in `archive-validation.json`. Passing an output comparison
does not establish SCF convergence.

The separate affinity probe verifies two workers on the assigned SMT pair
for all 18 ranks. The MKL probe records `NThr:2` on all 18 ranks. An untimed
C60 Cluster run reports CPU ELPA `requested=2`, `configured=2`, `MPI_thread=2`.
Sampled rank maps from each measured job verify Intel MPI, Intel OpenMP,
threaded MKL and the frozen fork libraries. They are sampled ranks, not a
complete set of 18 runtime maps for every job.

Initial untimed C60 diagnostics needed two additional carbon data files and
an explicit Cluster selection to exercise ELPA; their original logs and the
successful final check are retained. The two-rank hybrid numerical probe
also prints a pin-mask warning. The separate successful 18-rank affinity
probe verifies the placement used by the benchmark. None of these probes
contributes to the reported timing.

## Records and verification

- `historical-manifest.json` is the original local manifest, preserved unchanged.
- `manifest.json` and `SHA256SUMS` describe this versioned archive. Hashes of
  the private executables, installed libraries and ELPA archive remain in
  the historical manifest; the binaries are not added to Git.
- `runs/<input>/<variant>/` contains exact input/result metadata, compressed
  stdout and native output, and sampled runtime maps. Each result includes
  peak RSS and minimum available RAM; no historical resource-sample files
  were saved by this driver.
- `inputs/` contains the original input definitions. PAO/VPS files are identified
  by their original hashes in `input-data-audit.json`.
- `build-evidence/`, `probes/` and `procedure/` retain the build, installation,
  numerical/threading diagnostics and original driver scripts. The procedure
  scripts retain original local paths and need those paths adjusted for replay.

Check the committed evidence without building or executing OpenMX:

```sh
python3 benchmarks/results/rtx5080_intel_cpu_20261104_18x2/verify.py
```

The complete original build, private installed binaries and uncompressed
calculation directories remain at
`/home/dc1394/openmx4.0_gpu_intel_benchmark_20261104-072751/`.

# GEMMul8 3.5.4 update (2026-10-27)

The dependency moves from v3.5.2 (`603b52363715796a0af5e4aa1ed8d386349b4251`)
to the upstream [v3.5.4 release](https://github.com/RIKEN-RCCS/GEMMul8/releases/tag/v3.5.4)
(`f8fdf0b2a5eafd4dbbfab6be8489d3356e0da06a`). The submodule contains the
unmodified release. The ordinary CUDA, Pegasus and HIP Makefiles use the same
pin; `Makefile.kugui` inherits the ordinary CUDA Makefile.

OpenMX still instantiates the INT8 backend for real and complex double
precision. The HIP adapter also retains its MAGMA entry points. This update
does not enable FP8, change the precision/workspace policy, or change the
GeForce, A100/H100 and AMD dispatch rules. Upstream's FP8 improvements therefore
do not imply an OpenMX speedup. The CUDA adapter suppresses the FP8
`__maxnreg__` hint only if the toolkit does not define it, and restores the
macro scope after the upstream includes.

## Matrix checks on RTX 5080

CUDA 13.4, NVHPC 26.9 and `sm_120`, using independent CPU `long double`
references for the accuracy cases:

| Check | Result |
| --- | --- |
| 39 general real/complex GEMMs, fast/accurate scaling, N/T/C, padding, K blocking and the 256 MiB cap | Maximum relative error 3.702e-13 |
| 72 prepared GEMMs, both fixed sides, operand version changes | Maximum relative error 4.188e-14; 48 preparations, 24 reuses, no fallback |
| Four large-column GEMMs, real/complex and fast/accurate | Maximum relative error 4.081e-16; no native fallback |
| Fixed forward-transform modes, cache ownership, trim and fallback | 229 checks passed |

The added large-column case uses `m=3, n=262141, k=3`. Its `n` exceeds
`65535 * 4`, the old undo-scaling launch limit, while the padded product still
fits one grouped-CRT tile. Workspace blocking is disabled in a separate
process so it cannot hide the limit. The test checks every result and the
padding, rejects nonfinite values, and fails if the bridge substitutes native
GEMM. Linking the same test with the saved v3.5.2 object passes
the existing 39 + 72 cases, then fails at the new case with `CUDA: invalid
argument` (exit 2). With v3.5.4 all four new cases pass.

```sh
GEMMUL8_TEST_LARGE_N=1 tests/run_gemmul8_smoke.sh
tests/run_gemmul8_forward_smoke.sh
```

Run the optional large-column test alone on the GPU with several GiB free.
`GEMMUL8_TEST_OBJECT=/absolute/path/to/gemmul8_openmx.o` reuses a matching
toolkit/architecture object. Omit it to compile the adapter from the current
checkout. Toolkit selection is documented in the scripts. The default test
does not enable the large allocation.

## Build and policy checks

The CUDA OpenMX application is built through CMake with `OPENMX_COPY_TO_WORK=OFF`.
The comparison baseline is rebuilt from `9ce5f11d` with v3.5.2, including the
latest A100 force optimization. Binaries, Core/SIMD shared libraries and hashes
are frozen separately for each dependency version.

Six focused CTests pass: `gemmul8-cache-resource-lifecycle`,
`gemmul8-architecture-policy`, `cusolver-refinement-architecture-policy`,
`build-backend-isolation`, `environment-registry` and `cmake-driver`.

The final CUDA adapter compiles with CUDA 12.5.1 and GCC 12 for both `sm_80`
and `sm_90`. The production bridge and matrix-check program compile and link
with that adapter and CUDA 12.5 cuBLAS/cuBLASLt. The toolkit components are
from NVIDIA's redistribution packages with their published SHA-256 hashes
verified. This is a compiler/link check, not an A100/H100 runtime test; neither
device is available here. CUDA 12.5's existing `__maxnreg__` definition is
preserved by the compatibility guard.

With the ROCm 7.2 SDK, the HIP adapter (including its MAGMA instantiations),
memory-saving configuration, production bridge and existing numerical-check
program compile at `-O3 --offload-arch=gfx1200 -DGPU_ARCH=gfx1200` and link
against hipBLAS/hipBLASLt. The adapter, configuration and bridge also pass
host/device syntax checks for `gfx942`, with `GPU_ARCH=gfx942` as required by
upstream's FP8/FNUZ header selection. The isolated source projection uses
the actual HIP backend headers. No AMD GPU is available here: these checks
do not validate AMD runtime behavior or performance. The `gfx942` check does
not include code generation or linking.

CUDA 12.3 with GCC 12 rejects `C_hi.ptr1` in the upstream grouped-CRT template
for a real GEMM. This also fails with the untouched v3.5.2 adapter and headers;
it is not introduced by this update. The `__maxnreg__` compatibility guard
removes the additional v3.5.4 parse errors but does not fix that pre-existing
compiler/template incompatibility. The previous CUDA 12.3 cuSOLVER-only check
must not be read as validation of the complete GEMMul8 application build.

## OpenMX numerical regression

Seven cases run before and after the update in each of CPU and GPU mode:
collinear/noncollinear cluster, collinear/noncollinear band, DC, DC-LNO and
Krylov. All 14 comparisons pass (28 calculations), with four MPI ranks and
two OpenMP threads per rank. The runner binds two cores to each rank and
preloads the matching frozen Core and host-SIMD libraries. CPU mode uses the
CUDA application's CPU path with `OPENMX_GPU=0` and ELPA2; this is not a new
CPU-only toolchain build.

| Mode | Maximum energy difference (Hartree) | Maximum force-component difference (Hartree/Bohr) |
| --- | --- | --- |
| CPU | 1.810e-10 | 9.621e-9 |
| GPU | 1.830e-10 | 8.094e-9 |

The acceptance limit is 1e-7 for both quantities. SCF counts and grids match
the baseline: the four Si8 cases converge in 13 iterations, and DC/DC-LNO/
Krylov in 19. GPU runs explicitly enable GEMMul8; the general-GEMM counters
confirm its use in DC and Krylov. Not every small solver case necessarily
enters GEMMul8, so the direct real/complex matrix checks above remain necessary.

Inputs retain the previous migration's Si8 regression fixtures. DC/DC-LNO/
Krylov derive from `work/ml_example/DIA8_*.dat` with `MD.Type=nomd`, one MD
step, `scf.maxIter=200`, stdout level 2, and absolute data paths. The Krylov
order is 600. These are convergence checks, not runs of the original short
MD inputs. Full provenance and hashes are in `input-provenance.json`.

## sidia333 timing comparison

The unchanged `work/sidia333_nc_cluster/sidia333_nc_cluster.dat` runs with
18 MPI ranks, one OpenMP thread each, and the existing MPS server. Each
version is measured once, sequentially, after all compiler/test jobs finish.

| GEMMul8 | Wall time (s) | Maximum reported computational time (s) |
| --- | --- | --- |
| v3.5.2 | 103.164 | 99.813 |
| v3.5.4 | 103.260 | 99.901 |

This pair differs by 0.09% in wall time and remains near the 105-second target;
it does not establish a speedup. Printed total energies agree, the largest
force-component difference is 1e-12 Hartree/Bohr, and the grids/SCF counts
match. Both runs reach the original 25-SCF limit. This is a fixed-workload
timing comparison, not evidence of SCF convergence; use the converged cases
above for the numerical regression.

Evidence, compiler commands, frozen binaries, inputs and numerical results
are retained locally under `build/gemmul8-3.5.4-update-20261027/`.

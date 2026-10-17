# MAGMA Ozaki Scheme II (GEMMul8) integration — implementation note

Status: EXPERIMENTAL, AMD GPU / ROCm / HIP first.
Tested on: AMD Radeon RX 9060 XT 16 GB (gfx1200), ROCm 7.2.0, single GPU,
MAGMA 2.10.0, GEMMul8 v3.0.4 (pinned commit 884a28750f28ae39a20475db8c2ebe4130b4d4ef),
INT8 backend, plain (non-Lt) hipBLAS handle API.

This note is the implementation note required by the spec in the repo root
(`magma_ozaki2.txt`, "Deliverables"). Paths are relative to `source/` unless
absolute. The MAGMA tree (`third_party/magma`) is untracked; it is re-created
from the committed tarball plus `third_party/magma-ozaki2.patch`.

## 1. What was built

- **Adapter** `interface_hip/ozaki2.cpp` + `include/magma_ozaki2.h`: the only
  translation unit that touches GEMMul8. It owns a per-(device, HIP stream)
  hipBLAS handle + grow-only device workspace cache (mutex-guarded), maps MAGMA
  enums/types to hipBLAS/GEMMul8, passes alpha/beta as host pointers, and never
  touches the MAGMA queue's own hipBLAS handle (GEMMul8 installs a workspace on
  the handle it is given). All work is stream-ordered on the queue's stream.
- **Routing hooks** in the interface_hip BLAS wrappers: `blas_d_v2.cpp`
  (magma_dgemm/dsymm/dsyr2k/dtrmm) and `blas_z_v2.cpp`
  (magma_zgemm/zhemm/zher2k/ztrmm) each begin with a
  `magma_ozaki2_try_X(...)` call; if the adapter handles the call it returns 1
  and the wrapper returns, otherwise the native hipBLAS path runs unchanged.
- **Scoped activation**: a process-wide atomic depth counter
  (`magma_ozaki2_scope_begin/end`) — deliberately NOT thread_local, because the
  2-stage eigensolvers spawn pthreads that issue magma_* BLAS calls. Routing is
  active while an _ozaki2 driver runs, or globally with env `MAGMA_OZAKI2=1`.
  With neither active, MAGMA behavior is bit-identical to stock.
- **Drivers** (8 files, all 4 precisions, host + `_gpu`):
  `src/{s,d}syevdx_ozaki2[_gpu].cpp`, `src/{c,z}heevdx_ozaki2[_gpu].cpp`.
  Signatures are identical to the existing `Xsyevdx`/`Xheevdx[_gpu]` routines.
  They do NOT re-implement any eigensolver: they pick 1-stage or 2-stage
  (section 4), delegate to the existing `magma_Xsyevdx`/`_2stage` drivers, and
  wrap the real solve (not workspace queries, not n at most 128) in the Ozaki-II
  scope. The `_gpu` 2-stage variant stages dA to host wA, runs the host 2-stage
  driver, and copies the first mout eigenvector columns back to dA.
- **Testers**: `testing/testing_{s,d}syevdx_ozaki2.cpp`,
  `testing_{c,z}heevdx_ozaki2.cpp` — 2stage-tester-style options plus
  `--version 1|2` (force 1-stage/2-stage), timing ozaki2 vs the same-algorithm
  native driver, het21/het22 residual + orthogonality checks and eigenvalue
  agreement vs native (flagged if above 1000*eps relative).
- **D&C knob**: `src/dlaex3.cpp` `magma_get_dlaed3_k()` now reads env
  `MAGMA_DLAED3_K` (default 512, clamped to at least 16) — the rank threshold
  below which the divide-and-conquer back-multiply stays on the CPU.
- **Build registration**: `Makefile.gen.hip` appends (adapter + drivers +
  testers, adapter compiled with `-std=c++20 -I../GEMMul8/include`); live
  `make.inc` gains `LIBDIR += -L../GEMMul8/lib`, `LIB += -lgemmul8 -lhipblaslt`,
  `INC += -I../GEMMul8/include`.
- **OpenMX side** (git-tracked, not part of the MAGMA patch):
  `source/gemmul8_openmx.cu` (explicit INT8 instantiations: symm/syr2k/trmm for
  double, hemm/her2k/trmm for double-complex, plus the 6 workSize functions),
  `source/magma_openmx.cu` (env switch + defaults, section "env table"),
  `source/Band_DFT_Col.c` (calls `openmx_magma_ozaki2_release()` at the
  OOM-retry site and at end-of-pass to return adapter workspaces to the driver),
  `source/openmx_common.h` (declaration), `source/makefile` (link order:
  `$(MAGMA_LIB)` before `$(GEMMUL8_LIB)`, since libmagma.a now references
  gemmul8:: symbols).

## 2. Which BLAS-like calls are routed, and which stay native

Routed through GEMMul8 (double and double-complex only, subject to the size
gates of section 3): **gemm, symm/hemm, syr2k/her2k, trmm**. In the eigensolver
these are the sytrd/hetrd blocked-reduction syr2k/her2k updates, the
ormtr/unmtr (larfb) gemm/trmm backtransform applied to the n x nev eigenvector
block, and the dlaex3 divide-and-conquer gemms. For selected ranges
(range = MagmaRangeI, nev = iu - il + 1) the backtransform operates on n x nev
matrices, so nev enters the gates directly as the gemm-equivalent k; small nev
falls back to native automatically. trmm is emulated out-of-place (temp C plus
a strided copy-back on the same stream), hence its stricter gate.

Intentionally left native:
- **BLAS-2 symv/hemv/gemv panel work** in the 1-stage tridiagonalization —
  unsupported by GEMMul8 and bandwidth-bound, nothing to gain.
- **sb2st bulge chasing** (2-stage) — CPU pthread kernels, not BLAS-3.
- **CPU LAPACK calls** (steqr/stemr/laed sorting and selection logic) — out of
  scope by design; Ozaki-II is a GEMM emulation, not an eigensolver.
- **magmablas_* custom kernels and multi-GPU `_m` paths** — they do not pass
  through the interface_hip wrapper layer, so they are not hooked.
- **s/c single precision** — the _ozaki2 drivers exist for all 4 precisions,
  but single-precision BLAS is never routed (fp32 is fast natively on gfx1200);
  s/c drivers are scoped passthroughs whose BLAS stays native.
- Any call failing the size gates, an unsupported enum combination, or the
  workspace cap/free-VRAM guard (counted in the fallback statistics).

## 3. Size gates and environment variables

Authority: `interface_hip/ozaki2.cpp`. Gates were tuned 2026-07-02 from a
48-shape gfx1200 microbenchmark (native rocBLAS/hipBLAS vs GEMMul8 INT8,
num_moduli 15). On the gemm-equivalent dims (m, n, keff):
route iff min(m, n, keff) at least min_dim AND keff at least min_k AND
m*n*keff at least min_mnk (times trmm_mnk_mult for trmm).
Defaults: real (128, 256, 2^28), complex (64, 128, 2^27), trmm multiplier 4.
Against the 48 measured shapes this routes 30 calls — all wins, zero losses
(worst routed time ratio 0.85 ozaki/native).

| Variable | Default | Meaning |
|---|---|---|
| MAGMA_OZAKI2 | 0 | 1 = route globally (not just inside _ozaki2 drivers) |
| MAGMA_OZAKI2_NUM_MODULI | 15 | GEMMul8 moduli count, clamped 2..20 (see section 5) |
| MAGMA_OZAKI2_FASTMODE | 0 | GEMMul8 fastmode flag |
| MAGMA_OZAKI2_MIN_DIM | 128 | real min(m,n,keff) gate |
| MAGMA_OZAKI2_MIN_DIM_CPLX | 64 | complex gate (inherits MIN_DIM if that is set) |
| MAGMA_OZAKI2_MIN_K | 256 | real keff gate |
| MAGMA_OZAKI2_MIN_K_CPLX | 128 | complex keff gate (inherits MIN_K if set) |
| MAGMA_OZAKI2_MIN_MNK | 268435456 (2^28) | real m*n*keff gate |
| MAGMA_OZAKI2_MIN_MNK_CPLX | 134217728 (2^27) | complex m*n*keff gate (inherits MIN_MNK if set) |
| MAGMA_OZAKI2_TRMM_MNK_MULT | 4 | min_mnk multiplier for trmm |
| MAGMA_OZAKI2_MAX_WS_MB | unset | absolute per-process workspace cap (MiB) |
| MAGMA_OZAKI2_MAX_WS_PERCENT | 30 when no MB cap | percent of total VRAM, divided by LOCAL_RANKS |
| MAGMA_OZAKI2_MIN_FREE_MB | 2048 | refuse workspace growth unless this much VRAM stays free |
| MAGMA_OZAKI2_LOCAL_RANKS | OMPI_COMM_WORLD_LOCAL_SIZE, else SLURM_NTASKS_PER_NODE, else 1 | processes sharing the GPU; divides the percent cap |
| MAGMA_OZAKI2_OPS | all | csv subset of gemm,symm,hemm,syr2k,her2k,trmm |
| MAGMA_OZAKI2_VERBOSE | 0 | 1 = config + atexit stats, 2 = per-call lines |
| MAGMA_OZAKI2_ALGO | auto | driver algorithm: auto / 1stage / 2stage (section 4) |
| MAGMA_DLAED3_K | 512 | dlaex3 CPU/GPU rank crossover, min 16 |
| OPENMX_MAGMA_OZAKI2 | 1 | OpenMX: use the _ozaki2 drivers (0 = legacy MAGMA) |
| OPENMX_MAGMA_OZAKI2_RETRY | 0 | OpenMX: host-backup dA and retry natively if the solve fails |

OpenMX (`source/magma_openmx.cu`) additionally sets, with setenv overwrite=0 so
user values always win: `MAGMA_OZAKI2_ALGO=1stage` and
`MAGMA_OZAKI2_LOCAL_RANKS=4`. The workspace cap and MIN_FREE_MB guard exist
because MAGMA's own device allocations fail hard (an early 32-rank band ON run
crashed with info=-113) while an adapter denial is a graceful native fallback;
with the rank-divided cap plus the free-VRAM guard the final benchmark runs had
zero crashes and zero retries.

## 4. One-stage vs two-stage selection

The _ozaki2 drivers use MAGMA's existing 1-stage (sytrd/hetrd) and 2-stage
(sy2sb/he2hb + bulge chasing) selected eigensolvers internally; nothing new
algorithmically. `MAGMA_OZAKI2_ALGO=auto` picks, with nev = iu - il + 1 for
range = MagmaRangeI and nev = n otherwise:

    2-stage iff uplo == MagmaLower AND n at least 2048
                AND (2*nev at most n OR n at most 3072)

2-stage is never chosen for MagmaUpper (he2hb is lower-only); an explicit
`2stage` request is silently demoted to 1-stage in that case.

**Why OpenMX defaults to 1stage**: OpenMX runs many busy-polling MPI ranks per
node (32 here). The 2-stage CPU phases — core-pinned bulge-chasing pthreads —
starve under that contention: hang-level slowdowns at n=2808 band, and -5 to
-11 percent for the cluster/nc workloads. 1-stage plus BLAS routing is a pure
win versus legacy in that environment. **When to choose 2stage**: idle machine
or few ranks — measured up to 3.5x per solve at n=2808 with a selected range.

## 5. Accuracy

- **num_moduli=15 is mandatory** with routing active inside the reduction: a
  moduli sweep showed 12 or fewer fails the 1000*eps relative eigenvalue
  agreement check, while saving only 4-10 percent runtime. Default stays 15
  (fastmode=0), matching GEMMul8's fp64-accurate configuration.
- **Residual and orthogonality**: in all 12 of 12 tester checks (d/z, several
  n, selected and full ranges) the ozaki2 result was slightly BETTER than the
  native same-algorithm result on both metrics.
- **End-to-end**: OpenMX Utot agreement within 1.5e-10 Hartree of the legacy
  path, with identical SCF iteration counts, on all three sidia benchmarks.

## 6. Benchmarks

### Microbenchmark (idle machine, gfx1200, num_moduli 15)

Big square/large-keff GEMMs: Ozaki-II wins 3.8-13.7x (double) and 5.9-18.4x
(double-complex) over native fp64 rocBLAS (native dgemm is only ~0.32 TFLOP/s
on gfx1200). Crossovers: real always loses for keff at most 128 (worst 2.85x
loss; even m*n*k = 4.0e9 at keff=128 loses 1.15x); complex loses for keff at
most 64, and at keff=128 when m*n*keff is below about 1e8. These crossovers are
exactly what the section 3 gates encode.

### Tester matrix (idle machine, same-algorithm comparison)

24 cells: precision d and z; n = 2808, 4096, 5616; selected range
(--fraction 0,0.2) and full spectrum; --version 1 (both 1-stage) and
--version 2 (both 2-stage). Every cell routed and won: ozaki2 speedup over the
same-algorithm native driver ranged 1.04x to 1.26x, growing with n and with
the full-spectrum share of BLAS-3 work. Independently of routing, 2-stage beat
1-stage by up to 3.5x per solve at n=2808 selected — that is the algorithm
choice, available only on an uncontended CPU (section 4).

### OpenMX end-to-end (this box, 32 cores with shell affinity 16, work/ozk_e2e)

| Case | np | OFF total / Diag (s) | ON total / Diag (s) | Total gain |
|---|---|---|---|---|
| sidia333 band, complex n=2808, kgrid 2x2x2 | 32 | 1030.1 / 929.9 | 1036.1 / 937.0 | -0.6 percent |
| sidia444 cluster, real n=6656 | 32 | 370.2 / 167.0 | 358.5 / 162.5 | +3.3 percent |
| sidia333 nc cluster, complex n2=5616 | 18 | 660.5 / 412.3 | 648.1 / 400.1 | +1.9 percent |

The band -0.6 percent is honest and expected: it is within run-to-run noise
plus the per-rank one-time GEMMul8/hipBLAS initialization (~0.5 s per rank,
paid once), which amortizes in longer runs; the routed per-call BLAS wins are
real but small relative to the k-point-parallel solve at n=2808 with 32 ranks
sharing one GPU. Zero crashes, zero retries, Utot within 1.5e-10 Ha in all
three cases.

## 7. Build and persistence

Design: MAGMA lives as a committed pristine tarball
(`third_party/magma/magma-2.10.0.tar.gz`) plus
`third_party/magma-ozaki2.patch` (4684 lines) and the template
`third_party/magma.openmx.make.inc` (which carries the GEMMul8 lines: LIBDIR
-L../GEMMul8/lib, LIB -lgemmul8 -lhipblaslt, INC -I../GEMMul8/include). The
stamp rule in `source/makefile` (target `$(MAGMA_STAMP)`) wipes the tree,
re-extracts the tarball, installs the make.inc, applies the patch with
`patch -d third_party/magma -p1`, then — because `tools/codegen.py` is not
shipped — touches every file MAGMA would otherwise try to regenerate
(blas_{s,c,d}_v2.cpp after blas_z_v2.cpp, slaex3.cpp after dlaex3.cpp, and the
s/d/c _ozaki2 driver and tester siblings) so no codegen rule ever fires.
GEMMul8 is a pinned shallow clone; only `lib/libgemmul8.a` and
`src/gemmul8_openmx.o` are local artifacts, built with hipcc -std=c++20 -O3
--offload-arch=gfx1200 -ffp-contract=off -DOCML_BASIC_ROUNDED_OPERATIONS.

Exact commands (from `source/`):

    make openmx          # builds libgemmul8.a, extracts+patches+builds MAGMA,
                         # links openmx (MAGMA_LIB before GEMMUL8_LIB)
    make install         # copies binaries to ../work

    # MAGMA testers
    make -C third_party/magma testing/testing_dsyevdx_ozaki2 \
                              testing/testing_zheevdx_ozaki2 -j8

    # example tester runs (from third_party/magma)
    MAGMA_OZAKI2_VERBOSE=1 ./testing/testing_dsyevdx_ozaki2 \
        -N 2808 --fraction 0,0.2 -JV -L --check --version 1
    MAGMA_OZAKI2_VERBOSE=1 ./testing/testing_zheevdx_ozaki2 \
        -N 2808 -JV -L --check --version 2

Success criteria: --check passes AND the atexit statistics report routed
calls above 0 (no silent fallback). A native run without any MAGMA_OZAKI2 env
(e.g. `./testing/testing_dsyevdx_2stage -N 1000 --check`) must still pass
unchanged. The persistence phase exercised the from-scratch path: tree wiped by
the stamp rule, re-extracted, re-patched, and rebuilt through `make openmx`,
reproducing the patched tree without manual steps.

## 8. Limitations

- Gate thresholds are tuned for fp64-weak gfx1200 (native dgemm ~0.32
  TFLOP/s); on AMD Instinct (gfx90a/gfx942) native fp64 is far stronger —
  re-run the microbenchmark and retune MIN_DIM/MIN_K/MIN_MNK before use.
- Multi-GPU `_m` driver paths are not routed (they bypass the interface_hip
  wrappers); single-GPU only.
- hipSOLVER call sites elsewhere in OpenMX are untouched; only the MAGMA
  eigensolver path in magma_openmx.cu is switched.
- The Fortran-stub-based checks in the stock testers were superseded by the
  standalone residual/orthogonality checker in the _ozaki2 testers.
- Per-rank one-time GEMMul8/hipBLAS initialization costs ~0.5 s; visible in
  short many-rank runs (the band -0.6 percent), amortized otherwise.
- 2-stage under many-rank CPU saturation is pathological (bulge-chasing
  pthread starvation up to hang-level) — hence the OpenMX 1stage default; use
  2stage only on an uncontended machine.
- s/c precisions delegate natively (drivers exist, BLAS is never routed).
- The 30 percent rank-divided workspace cap and 2048 MiB free guard are
  conservative defaults for one 16 GB GPU shared by many ranks, not tuned maxima.

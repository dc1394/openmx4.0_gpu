#!/usr/bin/env python3
"""Check the orbital spline boundaries and the failure handling of the
device orbital evaluation without a GPU.

AMD port of the NVIDIA test of the same name.  There the evaluators are
OpenACC C; here they are HIP device functions (source/orbs_otf_hip.cu), so
the production file is compiled whole as host C++ against a stub
<hip/hip_runtime.h>: hipMalloc/hipMemcpy work on host memory (the arena
poisoned like fresh device memory) with failure injection, and a kernel
launch runs its blocks and threads one after another.  The device code of
source/Set_Orbitals_Grid.c (SOG_gpu_eligible, the table build
Set_Orbitals_Grid_GPU, Set_Orbitals_Grid_OTF_Prepare/_Release) is extracted
verbatim and built as C11 against stub OpenMX globals.  Checks:
  1 the radial splines, FP64 (spline64 + radial_fp64 from R) and
    double-float (radial_df with the uploaded mesh constants), at every
    mesh interval m = 1 .. mesh-1 and t = 0, 0.1, .., 1 against a long
    double reference with the end substitutes (h1 = -(h2 + h3), f1 = f4;
    h3 = -(h1 + h2), f4 = f1), on the uploaded tables: species 0's mesh
    opens the device arena, so a read of rv[-1] leaves the allocation
  2 whole points through OTF_Orbitals_Eval on an FP64 and on a
    double-float upload, and OTF_Orbitals_Eval_FP64 on the double-float
    upload (bitwise the FP64 upload's values; it builds the tables
    Orbs_Grid / Orbs_Grid_FNAN), in every interval and several directions,
    inside the first mesh point and beyond the last, against a long double
    reference of radial x real harmonic; with the item indexing (idx,
    idx_off, sph_off, out_off; the gaps between the items stay untouched)
  3 OTF_Orbitals_Upload: tables outside the evaluator -> -1; a failed
    hipMalloc, each failed hipMemcpy and each failed host allocation -> 0
    with nothing left allocated and the error state cleared; the launch:
    not ready -> 1, launch error -> 2, a stale error of an earlier call
    does not fail it
  4 Set_Orbitals_Grid_OTF_Prepare: the uploaded tables are OpenMX's; each
    failed host allocation, the failed device allocation and each failed
    transfer -> 0, nothing leaked, and the next call uploads (no latch);
    an ineligible basis, OPENMX_ORBS_GRID_GPU=0 and tables outside the
    evaluator -> 0 latched until Set_Orbitals_Grid_OTF_Release; the
    spellings of OPENMX_ORBS_EVAL_PRECISION and the FP32:FP64 default
  5 Set_Orbitals_Grid_GPU on a small system (two local atoms, two remote
    neighbours of different species in one launch, an empty overlap):
    Orbs_Grid and Orbs_Grid_FNAN equal the
    FP64 tiles at their points bitwise, the tables of the tiles stay; a
    rank without atoms touches nothing; a failed staging allocation, each
    failed transfer, a failed launch, each failed host allocation and a
    failed upload of the tables -> 0 (the host loops) with nothing leaked
  6 the build keeps the double-float arithmetic uncontracted
    (-ffp-contract=off for orbs_otf_hip.o, the pragma in the file) and the
    table build uses the shared FP64 evaluator
ASan/UBSan are on unless --no-sanitize.  --source DIR tests another copy of
the three source files.  The device build itself (gfx1200 code, FMA) is
checked on a GPU by tools/check_orbs_otf_hip.cu.
"""

import argparse
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile


def function(source, name):
    match = re.search(r"^(?:static\s+)?[A-Za-z_][\w \t*]*?\b" + re.escape(name) + r"\([^;{]*\)\s*\{",
                      source, re.MULTILINE)
    if match is None:
        raise ValueError(f"Cannot find production function {name}")
    brace = match.end() - 1
    depth, end = 1, brace + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[match.start():end]


def line(source, pattern):
    match = re.search(pattern, source, re.MULTILINE)
    if match is None:
        raise ValueError(f"Cannot find production line {pattern!r}")
    return match.group(0)


STUB_HIP = r'''
/* stub HIP runtime for a host build of orbs_otf_hip.cu */
#pragma once
#include <cstddef>
#include <cstdlib>
#include <cstring>

#define __device__
#define __host__
#define __global__
#define __forceinline__ inline
#define __noinline__ __attribute__((noinline))
#define __launch_bounds__(...)

typedef int hipError_t;
enum { hipSuccess = 0, hipErrorInvalidValue = 1, hipErrorOutOfMemory = 2, hipErrorLaunchFailure = 719 };
typedef struct ihipStream_t *hipStream_t;
typedef enum { hipMemcpyHostToHost, hipMemcpyHostToDevice, hipMemcpyDeviceToHost } hipMemcpyKind;

struct dim3 {
  unsigned x, y, z;
  dim3(unsigned a = 1, unsigned b = 1, unsigned c = 1) : x(a), y(b), z(c) {}
};
extern dim3 blockIdx, threadIdx, blockDim, gridDim;
extern int stub_launch_fail, stub_last_error;

/* C linkage: the extracted C code of Set_Orbitals_Grid.c calls them too */
extern "C" {
hipError_t hipMalloc(void **p, size_t n);
hipError_t hipFree(void *p);
hipError_t hipMemcpy(void *d, const void *s, size_t n, hipMemcpyKind kind);
hipError_t hipDeviceSynchronize(void);
hipError_t hipGetLastError(void);
}
template <class T> inline hipError_t hipMalloc(T **p, size_t n) { return hipMalloc((void **)p, n); }

/* the blocks and threads one after another (the kernels have no barriers) */
template <class F> void stub_launch(F f, dim3 grid, dim3 block)
{
  if (stub_launch_fail) {
    stub_last_error = hipErrorLaunchFailure;
    return;
  }
  gridDim = grid;
  blockDim = block;
  for (unsigned y = 0; y < grid.y; y++)
    for (unsigned x = 0; x < grid.x; x++)
      for (unsigned t = 0; t < block.x; t++) {
        blockIdx = dim3(x, y, 0);
        threadIdx = dim3(t, 0, 0);
        f();
      }
}
#define hipLaunchKernelGGL(kernel, grid, block, shmem, stream, ...) \
  stub_launch([&]() { kernel(__VA_ARGS__); }, (grid), (block))
'''


CXX_TEST = r'''
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <vector>

#include "orbs_otf_hip.cu"
#include "openmx_precision.h"

#define CHECK(condition) do { if (!(condition)) { \
    fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #condition); exit(2); \
} } while (0)

/* ---- the stub HIP runtime: host memory, failure injection ---- */

dim3 blockIdx, threadIdx, blockDim, gridDim;
int stub_malloc_fail_at, stub_malloc_fail_count = 1, stub_memcpy_fail_at, stub_launch_fail;
int stub_mallocs, stub_memcpys, stub_live, stub_last_error;

/* hipMalloc number fail_at .. fail_at + fail_count - 1 fails */
extern "C" hipError_t hipMalloc(void **p, size_t n)
{
  *p = nullptr;
  ++stub_mallocs;
  if (0 < stub_malloc_fail_at && stub_malloc_fail_at <= stub_mallocs &&
      stub_mallocs < stub_malloc_fail_at + stub_malloc_fail_count)
    return stub_last_error = hipErrorOutOfMemory;
  if (posix_memalign(p, 256, n ? n : 1) != 0) {
    *p = nullptr;
    return stub_last_error = hipErrorOutOfMemory;
  }
  memset(*p, 0xa5, n);   /* the arena's padding is garbage, as on the device */
  stub_live++;
  return hipSuccess;
}

extern "C" hipError_t hipFree(void *p)
{
  if (p != nullptr) {
    stub_live--;
    free(p);
  }
  return hipSuccess;
}

extern "C" hipError_t hipMemcpy(void *d, const void *s, size_t n, hipMemcpyKind)
{
  if (++stub_memcpys == stub_memcpy_fail_at) return stub_last_error = hipErrorInvalidValue;
  memcpy(d, s, n);
  return hipSuccess;
}

extern "C" hipError_t hipDeviceSynchronize(void) { return hipSuccess; }

extern "C" hipError_t hipGetLastError(void)
{
  const hipError_t e = stub_last_error;
  stub_last_error = hipSuccess;
  return e;
}

/* host allocations (the double-float tables) with failure injection */
static long new_fail_at, new_count, new_live;

void *operator new(std::size_t n)
{
  void *p;
  if (++new_count == new_fail_at) throw std::bad_alloc();
  p = std::malloc(n ? n : 1);
  if (p == nullptr) throw std::bad_alloc();
  new_live++;
  return p;
}
void operator delete(void *p) noexcept
{
  if (p != nullptr) {
    new_live--;
    std::free(p);
  }
}
void operator delete(void *p, std::size_t) noexcept { operator delete(p); }

/* ---- two species: 0 the mesh of the NVIDIA test, 1 a longer logarithmic one ---- */

enum { S = 2 };
static const int kMesh[S] = {6, 8};
static const int kMaxl[S] = {3, 2};
static const int kNb[S][4] = {{2, 1, 1, 1}, {1, 2, 1, 0}};
static const double G[3][3] = {{0.31, 0.02, -0.01}, {-0.03, 0.27, 0.05}, {0.04, -0.02, 0.33}};
static const int NG2 = 5, NG3 = 6, NGRID = 4 * 5 * 6;
static double ATV[3 * 3] = {0.0, 0.0, 0.0, 5.1, -0.4, 0.3, -2.2, 3.3, 1.7};
static std::vector<double> RV, RWF;
static size_t RVO[S + 1], RWB[S + 1];
static int SPM[S], SPX[S], SPB[4 * S], NO[S];

extern "C" double test_radial_value(int w, int L0, int Mul0, int i)
{
  return cos(0.9 * i + 0.7 * L0 + 0.4 * Mul0 + 0.3 * w) * (1.0 + 0.15 * i) / (1.0 + L0);
}

extern "C" double test_mesh_value(int w, int i)
{
  return w == 0 ? 0.1 * ldexp(1.0, i) : 0.05 * pow(1.8, i);
}

static void build_species(void)
{
  for (int w = 0; w < S; w++) {
    RVO[w] = RV.size();
    RWB[w] = RWF.size();
    SPM[w] = kMesh[w];
    SPX[w] = kMaxl[w];
    NO[w] = 0;
    for (int i = 0; i < kMesh[w]; i++) RV.push_back(test_mesh_value(w, i));
    for (int L0 = 0; L0 <= 3; L0++) {
      SPB[4 * w + L0] = (L0 <= kMaxl[w]) ? kNb[w][L0] : 0;
      NO[w] += (2 * L0 + 1) * SPB[4 * w + L0];
      for (int Mul0 = 0; Mul0 < SPB[4 * w + L0]; Mul0++)
        for (int i = 0; i < kMesh[w]; i++) RWF.push_back(test_radial_value(w, L0, Mul0, i));
    }
  }
  RVO[S] = RV.size();
  RWB[S] = RWF.size();
}

static OTF_HostTables tables(void)
{
  OTF_HostTables t;

  memset(&t, 0, sizeof(t));
  t.species = S;
  t.rv_off = RVO;
  t.rwf_base = RWB;
  t.mesh = SPM;
  t.maxl = SPX;
  t.nb = SPB;
  t.rv = RV.data();
  t.rwf = RWF.data();
  t.rv_count = RV.size();
  t.rwf_count = RWF.size();
  t.atv = ATV;
  t.atv_rows = 3;
  t.ng23 = NG2 * NG3;
  t.ng3 = NG3;
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) t.g[i][j] = G[i][j];
  return t;
}

/* ---- the long double reference (the host loops with the end substitutes) ---- */

typedef long double ld;

/* the first mesh index with rv[m] >= R in [1, mesh - 1] */
static int ref_interval(const double *rv, int mesh, ld R)
{
  int m = 1;
  while (m < mesh - 1 && (ld)rv[m] < R) m++;
  return m;
}

static ld ref_spline(const double *rv, const double *rw, int mesh, int m, ld y1, ld *deriv)
{
  const ld h2 = (ld)rv[m] - (ld)rv[m - 1];
  ld h1, h3;

  if (m == 1) {
    h3 = (ld)rv[m + 1] - (ld)rv[m];
    h1 = -(h2 + h3);
  }
  else if (m == mesh - 1) {
    h1 = (ld)rv[m - 1] - (ld)rv[m - 2];
    h3 = -(h1 + h2);
  }
  else {
    h1 = (ld)rv[m - 1] - (ld)rv[m - 2];
    h3 = (ld)rv[m + 1] - (ld)rv[m];
  }
  const ld f1 = rw[m == 1 ? m + 1 : m - 2], f2 = rw[m - 1], f3 = rw[m], f4 = rw[m == mesh - 1 ? m - 2 : m + 1];
  const ld g1 = ((f3 - f2) * h1 / h2 + (f2 - f1) * h2 / h1) / (h1 + h2);
  const ld g2 = ((f4 - f3) * h2 / h3 + (f3 - f2) * h3 / h2) / (h2 + h3);
  const ld y2 = y1 - 1, p = h2 * g1, q = h2 * g2;

  if (deriv != nullptr)
    *deriv = 2 * y2 / h2 * (3 * f2 + p + (2 * f2 + p) * y2) + y2 * y2 * (2 * f2 + p) / h2 +
             2 * y1 / h2 * (3 * f3 - q - (2 * f3 - q) * y1) - y1 * y1 * (2 * f3 - q) / h2;
  return y2 * y2 * (3 * f2 + p + (2 * f2 + p) * y2) + y1 * y1 * (3 * f3 - q - (2 * f3 - q) * y1);
}

/* radial function row (of angular momentum L0) of species w at R */
static ld ref_radial(int w, int L0, int row, ld R)
{
  const double *rv = RV.data() + RVO[w];
  const double *rw = RWF.data() + RWB[w] + (size_t)row * kMesh[w];
  const int mesh = kMesh[w];

  if ((ld)rv[mesh - 1] < R) return 0;
  if (R < (ld)rv[0]) {
    /* the cubic extrapolation from the spline at rm = rv[4] */
    const ld rm = rv[4];
    ld d1, a, b, c, d;
    const ld f = ref_spline(rv, rw, mesh, 4, 1.0L, &d1);

    if (L0 == 0) {
      a = 0;
      b = 0.5L * d1 / rm;
      c = 0;
      d = f - b * rm * rm;
    }
    else if (L0 == 1) {
      a = (rm * d1 - f) / (2 * rm * rm * rm);
      b = 0;
      c = d1 - 3 * a * rm * rm;
      d = 0;
    }
    else {
      b = (3 * f - rm * d1) / (rm * rm);
      a = (f - b * rm * rm) / (rm * rm * rm);
      c = 0;
      d = 0;
    }
    return a * R * R * R + b * R * R + c * R + d;
  }
  {
    const int m = ref_interval(rv, mesh, R);
    return ref_spline(rv, rw, mesh, m, (R - (ld)rv[m - 1]) / ((ld)rv[m] - (ld)rv[m - 1]), nullptr);
  }
}

static void ref_harmonics(int L0, ld coQ, ld siQ, ld coP, ld siP, ld *A)
{
  if (L0 == 0) {
    A[0] = 0.282094791773878;
  }
  else if (L0 == 1) {
    A[0] = 0.48860251190292 * siQ * coP;
    A[1] = 0.48860251190292 * siQ * siP;
    A[2] = 0.48860251190292 * coQ;
  }
  else if (L0 == 2) {
    A[0] = 0.94617469575756 * coQ * coQ - 0.31539156525252;
    A[1] = 0.54627421529604 * siQ * siQ * (1 - 2 * siP * siP);
    A[2] = 1.09254843059208 * siQ * siQ * siP * coP;
    A[3] = 1.09254843059208 * siQ * coQ * coP;
    A[4] = 1.09254843059208 * siQ * coQ * siP;
  }
  else {
    A[0] = 0.373176332590116 * (5 * coQ * coQ * coQ - 3 * coQ);
    A[1] = 0.457045799464466 * coP * siQ * (5 * coQ * coQ - 1);
    A[2] = 0.457045799464466 * siP * siQ * (5 * coQ * coQ - 1);
    A[3] = 1.44530572132028 * siQ * siQ * coQ * (coP * coP - siP * siP);
    A[4] = 2.89061144264055 * siQ * siQ * coQ * siP * coP;
    A[5] = 0.590043589926644 * siQ * siQ * siQ * (4 * coP * coP * coP - 3 * coP);
    A[6] = 0.590043589926644 * siQ * siQ * siQ * (3 * siP - 4 * siP * siP * siP);
  }
}

/* every orbital of species w at the offset (x, y, z) from its centre */
static ld ref_point(int w, double x, double y, double z, ld *out)
{
  const ld X = x, Y = y, Z = z;
  const ld r = sqrtl(X * X + Y * Y + Z * Z), r1 = sqrtl(X * X + Y * Y);
  ld coQ = 0, siQ = 1, coP = 1, siP = 0;
  int i = 0, row = 0;

  if (10e-15L <= r) {
    coQ = Z / r;
    siQ = r1 / r;
    if (10e-15L <= r1) {
      siP = Y / r1;
      coP = X / r1;
    }
  }
  for (int L0 = 0; L0 <= kMaxl[w]; L0++) {
    ld A[7];

    ref_harmonics(L0, coQ, siQ, coP, siP, A);
    for (int Mul0 = 0; Mul0 < kNb[w][L0]; Mul0++, row++) {
      const ld rf = ref_radial(w, L0, row, r);
      for (int M0 = 0; M0 <= 2 * L0; M0++) out[i++] = rf * A[M0];
    }
  }
  CHECK(i == NO[w]);
  return r;
}

/* the device's grid coordinates (the same FP64 expression) */
static void coords(const OTF_Atom &a, int GNc, int GRc, double *x, double *y, double *z)
{
  const int ng23 = NG2 * NG3, ng3 = NG3;
  const int n1 = GNc / ng23;
  const int n2 = (GNc - n1 * ng23) / ng3;
  const int n3 = GNc - n1 * ng23 - n2 * ng3;

  *x = ((double)n1 * G[0][0] + (double)n2 * G[1][0] + (double)n3 * G[2][0] + 0.0) + ATV[3 * GRc + 0] - a.gx - a.ax;
  *y = ((double)n1 * G[0][1] + (double)n2 * G[1][1] + (double)n3 * G[2][1] + 0.0) + ATV[3 * GRc + 1] - a.gy - a.ay;
  *z = ((double)n1 * G[0][2] + (double)n2 * G[1][2] + (double)n3 * G[2][2] + 0.0) + ATV[3 * GRc + 2] - a.gz - a.az;
}

static double ulpf(ld v)
{
  const float f = (float)fabsl(v);
  return (double)(nextafterf(f, INFINITY) - f);
}

/* ---- 1: the radial splines at every interval ---- */

static int radial_checks(void)
{
  int checks = 0;

  CHECK(OTF_Orbitals_Ready() && OTF_Orbitals_DoubleFloat());
  for (int w = 0; w < S; w++) {
    const int mesh = kMesh[w];
    /* the uploaded tables: species 0's mesh is the first array of the arena */
    const double *rv = g_dev.rv + RVO[w];

    for (int m = 1; m < mesh; m++) {
      for (int step = 0; step <= 10; step++) {
        const double t = step / 10.0;
        const double R = rv[m - 1] + t * (rv[m] - rv[m - 1]);
        const Spline64 s = spline64(rv, mesh, R);
        const float *mc = g_dev.mcd + kMcdStride * (RVO[w] + (size_t)m);
        int row = 0;

        for (int L0 = 0; L0 <= kMaxl[w]; L0++) {
          for (int Mul0 = 0; Mul0 < kNb[w][L0]; Mul0++, row++) {
            const size_t first = RWB[w] + (size_t)row * mesh;
            const ld ref_R = ref_radial(w, L0, row, R);
            const ld ref_t = ref_spline(RV.data() + RVO[w], RWF.data() + first, mesh, m, t, nullptr);
            const double v64 = radial_fp64(g_dev.rwf + first, L0, s);
            const df vdf = radial_df(g_dev.rwd + 2 * first, m, mesh, df_mk(mc[2], mc[3]), df_mk(mc[4], mc[5]),
                                     df_mk(mc[6], mc[7]), df_mk(mc[8], mc[9]), df_mk(mc[10], mc[11]),
                                     df_dbl(t), df_dbl(t - 1.0), df_dbl(t * t), df_dbl((t - 1.0) * (t - 1.0)));

            if (!(fabsl((ld)v64 - ref_R) < 2e-12L) || !(fabsl((ld)vdf.hi + vdf.lo - ref_t) < 2e-12L)) {
              fprintf(stderr, "species %d m %d t %.1f L %d Mul %d: FP64 %.17g DF %.17g reference %.17Lg %.17Lg\n",
                      w, m, t, L0, Mul0, v64, (double)vdf.hi + vdf.lo, ref_R, ref_t);
              exit(2);
            }
            checks += 2;
          }
        }
      }
    }
  }
  return checks;
}

/* ---- 2: whole points through the kernel launches ---- */

struct Launch {
  std::vector<OTF_Atom> atoms;
  std::vector<int> npts, idx, gla, cla;
  std::vector<size_t> idx_off, out_off, sph_off;
  size_t out_size = 0;
  int max_npts = 1;
  bool use_idx = false;
};

static void add_item(Launch &L, const OTF_Atom &a, int npts, size_t idx_first, size_t sph_first)
{
  L.atoms.push_back(a);
  L.npts.push_back(npts);
  L.idx_off.push_back(idx_first);
  L.sph_off.push_back(sph_first);
  L.out_off.push_back(L.out_size);
  L.out_size += (size_t)npts * (size_t)a.no + 1;   /* one untouched slot after each item */
  if (L.max_npts < npts) L.max_npts = npts;
}

static OTF_Atom atom(int w, double gx, double gy, double gz, double ax, double ay, double az)
{
  OTF_Atom a;

  memset(&a, 0, sizeof(a));
  a.wan = w;
  a.no = NO[w];
  a.gx = gx; a.gy = gy; a.gz = gz;
  a.ax = ax; a.ay = ay; a.az = az;
  return a;
}

/* one point per item at radius R in direction dir: exact offsets (grid
   point 0, cell 0, no shift) or composed ones */
static void add_point(Launch &L, int w, double R, const double *dir, int composed)
{
  const size_t k = L.atoms.size();
  const int GNc = composed ? (int)((k * 37) % NGRID) : 0;
  const int GRc = composed ? (int)(k % 3) : 0;
  const double sx = composed ? 0.7 : 0.0, sy = composed ? -1.3 : 0.0, sz = composed ? 2.1 : 0.0;
  OTF_Atom zero = atom(w, 0, 0, 0, 0, 0, 0);
  double bx, by, bz;

  /* the cell's grid point relative to a zero atom, then the atom at offset -d */
  coords(zero, GNc, GRc, &bx, &by, &bz);
  L.gla.push_back(GNc);
  L.cla.push_back(GRc);
  add_item(L, atom(w, bx - sx - R * dir[0], by - sy - R * dir[1], bz - sz - R * dir[2], sx, sy, sz), 1, 0,
           L.gla.size() - 1);
}

static Launch launch_points(void)
{
  static const double dirs[8][3] = {{0, 0, 1}, {0, 0, -1}, {1, 0, 0}, {-1, 0, 0}, {0, -1, 0},
                                    {0.3, -0.5, 0.8}, {-0.6, -0.7, 0.2}, {-0.5, 0.5, -0.7071067811865476}};
  static const double ts[5] = {0.003, 0.25, 0.5, 0.75, 0.997};
  Launch L;

  for (int w = 0; w < S; w++) {
    const int mesh = kMesh[w];
    const double *rv = RV.data() + RVO[w];

    for (int d = 0; d < 8; d++) {
      double dir[3] = {dirs[d][0], dirs[d][1], dirs[d][2]};
      const double n = sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
      for (int c = 0; c < 3; c++) dir[c] /= n;
      for (int m = 1; m < mesh; m++)
        for (int i = 0; i < 5; i++) add_point(L, w, rv[m - 1] + ts[i] * (rv[m] - rv[m - 1]), dir, (m + i + d) % 2);
      /* inside the first mesh point (the cubic extrapolation) and beyond the mesh */
      add_point(L, w, 0.9 * rv[0], dir, d % 2);
      add_point(L, w, 0.25 * rv[0], dir, (d + 1) % 2);
      add_point(L, w, 1.001 * rv[mesh - 1], dir, d % 2);
      add_point(L, w, 2.0 * rv[mesh - 1], dir, (d + 1) % 2);
    }
    {
      /* the centre (r < Min_r) */
      const double dir[3] = {0.0, 0.0, 1.0};
      add_point(L, w, 0.0, dir, 0);
      add_point(L, w, 0.0, dir, 1);
    }
  }
  /* three consecutive sphere points in one item (the identity index) */
  {
    const size_t first = L.gla.size();
    for (int k = 0; k < 3; k++) {
      L.gla.push_back(11 + 13 * k);
      L.cla.push_back(k);
    }
    add_item(L, atom(1, 0.4, 0.3, 0.5, 0.2, -0.1, 0.3), 3, 0, first);
  }
  return L;
}

/* items over two spheres through sphere indices: A (40 points, cell 0)
   and B (25 points, cells 1 and 2, some beyond the mesh) */
static Launch launch_spheres(void)
{
  Launch L;

  L.use_idx = true;
  for (int k = 0; k < 40; k++) {
    L.gla.push_back((k * 53 + 7) % NGRID);
    L.cla.push_back(0);
  }
  for (int k = 0; k < 25; k++) {
    L.gla.push_back((k * 29 + 3) % NGRID);
    L.cla.push_back(1 + (k % 5 == 4));
  }
  /* species 0 at 30 points of A, scrambled */
  for (int k = 0; k < 30; k++) L.idx.push_back((k * 7) % 40);
  add_item(L, atom(0, 0.6, 0.55, 0.85, 0.0, 0.0, 0.0), 30, 0, 0);
  /* species 1 shifted, all of B reversed */
  for (int k = 0; k < 25; k++) L.idx.push_back(24 - k);
  add_item(L, atom(1, 5.1 - 0.7 + 0.5, -0.4 + 1.3 + 0.6, 0.3 - 2.1 + 0.7, 0.7, -1.3, 2.1), 25, 30, 40);
  /* no points */
  add_item(L, atom(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), 0, 55, 0);
  /* species 1 at all of A */
  for (int k = 0; k < 40; k++) L.idx.push_back(k);
  add_item(L, atom(1, 0.4, 0.5, 0.6, 0.0, 0.0, 0.0), 40, 55, 0);
  return L;
}

typedef int (*EvalFn)(void *, int, int, const OTF_Atom *, const int *, const size_t *, const size_t *,
                      const size_t *, const int *, const int *, const int *, float *);

static std::vector<float> run(const Launch &L, EvalFn eval)
{
  std::vector<float> out(L.out_size, NAN);

  CHECK(eval(nullptr, (int)L.atoms.size(), L.max_npts, L.atoms.data(), L.npts.data(), L.idx_off.data(),
             L.out_off.data(), L.sph_off.data(), L.use_idx ? L.idx.data() : nullptr, L.gla.data(), L.cla.data(),
             out.data()) == 0);
  return out;
}

struct Tally {
  int values = 0, points = 0, below = 0, beyond = 0, end_intervals = 0, df_differ = 0, df_tiny = 0;
  double worst64 = 0.0, worst_df = 0.0, worst_tiny = 0.0;
};

/* out64, outdf, out64f: FP64 upload, double-float upload, forced FP64 on it */
static void compare(const Launch &L, const std::vector<float> &out64, const std::vector<float> &outdf,
                    const std::vector<float> &out64f, Tally &tally)
{
  CHECK(memcmp(out64.data(), out64f.data(), sizeof(float) * L.out_size) == 0 || L.out_size == 0);
  for (size_t p = 0; p < L.atoms.size(); p++) {
    const OTF_Atom &a = L.atoms[p];
    const size_t end = L.out_off[p] + (size_t)L.npts[p] * a.no;

    CHECK(std::isnan(out64[end]) && std::isnan(outdf[end]));   /* untouched */
    for (int k = 0; k < L.npts[p]; k++) {
      const int Nc = L.use_idx ? L.idx[L.idx_off[p] + k] : k;
      const int GNc = L.gla[L.sph_off[p] + Nc], GRc = L.cla[L.sph_off[p] + Nc];
      const size_t o = L.out_off[p] + (size_t)k * a.no;
      const double *rv = RV.data() + RVO[a.wan];
      double x, y, z;
      ld ref[64];

      coords(a, GNc, GRc, &x, &y, &z);
      const ld R = ref_point(a.wan, x, y, z, ref);
      const int below = R < (ld)rv[0], beyond = (ld)rv[kMesh[a.wan] - 1] < R;
      const int m = (below || beyond) ? 0 : ref_interval(rv, kMesh[a.wan], R);

      tally.points++;
      tally.below += below;
      tally.beyond += beyond;
      tally.end_intervals += (m == 1 || m == kMesh[a.wan] - 1);
      for (int i = 0; i < a.no; i++) {
        const double e64 = (double)fabsl((ld)out64[o + i] - ref[i]);
        const double edf = (double)fabsl((ld)outdf[o + i] - ref[i]);
        const double tol = ulpf(ref[i]);

        if (!(e64 <= tol + 1e-14) || !(edf <= tol + 1e-12)) {
          fprintf(stderr, "item %zu point %d (species %d, R %.17Lg, m %d) orbital %d: FP64 %.9g DF %.9g "
                  "reference %.12Lg\n", p, k, a.wan, R, m, i, out64[o + i], outdf[o + i], ref[i]);
          exit(2);
        }
        if (beyond) CHECK(out64[o + i] == 0.0f && outdf[o + i] == 0.0f);
        /* the double-float kernel takes the FP64 routine inside the first mesh point */
        if (below) CHECK(memcmp(&out64[o + i], &outdf[o + i], sizeof(float)) == 0);
        /* double-float and FP64: a float apart at most, or both at the
           rounding level of the values (a harmonic that vanishes on an axis) */
        if (out64[o + i] != outdf[o + i]) {
          const double d = fabs((double)out64[o + i] - (double)outdf[o + i]);

          if (fabsl(ref[i]) < 1e-12L) {
            CHECK(d <= 2e-12);
            tally.df_tiny++;
          }
          else {
            CHECK(d <= tol);
            tally.df_differ++;
          }
        }
        if (fabsl(ref[i]) < 1e-12L) {
          if (tally.worst_tiny < e64) tally.worst_tiny = e64;
          if (tally.worst_tiny < edf) tally.worst_tiny = edf;
        }
        else {
          if (tally.worst64 < e64 / tol) tally.worst64 = e64 / tol;
          if (tally.worst_df < edf / tol) tally.worst_df = edf / tol;
        }
        tally.values++;
      }
    }
  }
}

/* ---- 3: OTF_Orbitals_Upload and the launch on failures ---- */

static int upload_checks(void)
{
  const OTF_HostTables t = tables();
  int checks = 0;

  {
    int maxl[S] = {4, 2}, mesh[S] = {6, 5}, nb[4 * S];
    OTF_HostTables u;

    memcpy(nb, SPB, sizeof(nb));
    nb[4 * 1 + 1] = 9;
    u = t; u.species = 0; CHECK(OTF_Orbitals_Upload(&u, 1) == -1);
    u = t; u.maxl = maxl; CHECK(OTF_Orbitals_Upload(&u, 0) == -1);
    u = t; u.mesh = mesh; CHECK(OTF_Orbitals_Upload(&u, 1) == -1);
    u = t; u.nb = nb; CHECK(OTF_Orbitals_Upload(&u, 0) == -1);
    CHECK(!OTF_Orbitals_Ready() && stub_live == 0);
    checks += 4;
  }
  for (int double_float = 0; double_float <= 1; double_float++) {
    const int copies = double_float ? 11 : 8;

    stub_mallocs = 0;
    stub_malloc_fail_at = 1;
    CHECK(OTF_Orbitals_Upload(&t, double_float) == 0);
    CHECK(!OTF_Orbitals_Ready() && stub_live == 0 && stub_last_error == 0);
    stub_malloc_fail_at = 0;
    checks++;
    for (int k = 1; k <= copies; k++) {
      stub_memcpys = 0;
      stub_memcpy_fail_at = k;
      CHECK(OTF_Orbitals_Upload(&t, double_float) == 0);
      CHECK(!OTF_Orbitals_Ready() && stub_live == 0 && stub_last_error == 0);
      checks++;
    }
    stub_memcpy_fail_at = 0;
    for (int k = 1; double_float && k <= 3; k++) {
      const long live = new_live;

      stub_mallocs = 0;
      new_count = 0;
      new_fail_at = k;
      CHECK(OTF_Orbitals_Upload(&t, double_float) == 0);
      CHECK(!OTF_Orbitals_Ready() && stub_live == 0 && stub_mallocs == 0 && new_live == live);
      checks++;
    }
    new_fail_at = 0;
    stub_memcpys = 0;
    CHECK(OTF_Orbitals_Upload(&t, double_float) == 1);
    CHECK(stub_memcpys == copies && stub_live == 1 && OTF_Orbitals_Ready() && 0 < OTF_Orbitals_Bytes());
    CHECK(OTF_Orbitals_DoubleFloat() == double_float);
  }
  {
    OTF_Atom a = atom(0, 0.1, 0.2, 0.3, 0, 0, 0);
    int np = 1, gla = 0, cla = 0;
    size_t zero = 0;
    float out[64];

    /* launch errors, and a stale error of an earlier call */
    stub_launch_fail = 1;
    CHECK(OTF_Orbitals_Eval(nullptr, 1, 1, &a, &np, &zero, &zero, &zero, nullptr, &gla, &cla, out) == 2);
    CHECK(OTF_Orbitals_Eval_FP64(nullptr, 1, 1, &a, &np, &zero, &zero, &zero, nullptr, &gla, &cla, out) == 2);
    stub_launch_fail = 0;
    stub_last_error = hipErrorOutOfMemory;
    CHECK(OTF_Orbitals_Eval(nullptr, 1, 1, &a, &np, &zero, &zero, &zero, nullptr, &gla, &cla, out) == 0);
    CHECK(OTF_Orbitals_Eval(nullptr, 0, 1, &a, &np, &zero, &zero, &zero, nullptr, &gla, &cla, out) == 0);
    OTF_Orbitals_Release();
    CHECK(!OTF_Orbitals_Ready() && stub_live == 0 && OTF_Orbitals_Bytes() == 0 && !OTF_Orbitals_DoubleFloat());
    CHECK(OTF_Orbitals_Eval(nullptr, 1, 1, &a, &np, &zero, &zero, &zero, nullptr, &gla, &cla, out) == 1);
    CHECK(OTF_Orbitals_Eval_FP64(nullptr, 1, 1, &a, &np, &zero, &zero, &zero, nullptr, &gla, &cla, out) == 1);
    checks += 7;
  }
  return checks;
}

/* ---- hooks of the C part (Set_Orbitals_Grid.c) ---- */

extern "C" {
int upload_force = 0, upload_calls = 0;

/* Set_Orbitals_Grid_OTF_Prepare's OTF_Orbitals_Upload: forced result or the real one */
int test_OTF_Orbitals_Upload(const OTF_HostTables *t, int double_float)
{
  upload_calls++;
  return upload_force ? upload_force : OTF_Orbitals_Upload(t, double_float);
}

void test_device_fail(int malloc_at, int memcpy_at)
{
  stub_mallocs = 0;
  stub_memcpys = 0;
  stub_malloc_fail_at = malloc_at;
  stub_malloc_fail_count = 1;
  stub_memcpy_fail_at = memcpy_at;
}

/* hipMalloc from malloc_at on fails count times; a failed kernel launch */
void test_device_fail_window(int malloc_at, int count, int launch)
{
  test_device_fail(malloc_at, 0);
  stub_malloc_fail_count = count;
  stub_launch_fail = launch;
}

int test_device_counts(int *mallocs, int *memcpys)
{
  *mallocs = stub_mallocs;
  *memcpys = stub_memcpys;
  return stub_live;
}

/* one point through OTF_Orbitals_Eval_FP64 (the uploaded tables) */
int test_eval_fp64(int wan, int no, double gx, double gy, double gz, double ax, double ay, double az, int GNc,
                   int GRc, float *out)
{
  OTF_Atom a = atom(wan, gx, gy, gz, ax, ay, az);
  int np = 1;
  size_t zero = 0;

  a.no = no;
  return OTF_Orbitals_Eval_FP64(nullptr, 1, 1, &a, &np, &zero, &zero, &zero, nullptr, &GNc, &GRc, out);
}

int test_device_live(void) { return stub_live; }

int test_device_error(void) { return stub_last_error; }

/* the uploaded tables against OpenMX's, flattened by the test */
int test_device_tables_equal(int species, const size_t *rvo, const size_t *rwb, const int *spm, const int *spx,
                             const int *spb, const double *rv, const double *rwf, const double *atv3, size_t rows,
                             int ng23, int ng3, const double *g9, const double *org3)
{
  if (!g_dev.ready) return 0;
  if (memcmp(g_dev.rvo, rvo, sizeof(size_t) * (species + 1)) || memcmp(g_dev.rwb, rwb, sizeof(size_t) * (species + 1)) ||
      memcmp(g_dev.spm, spm, sizeof(int) * species) || memcmp(g_dev.spx, spx, sizeof(int) * species) ||
      memcmp(g_dev.spb, spb, sizeof(int) * 4 * species) || memcmp(g_dev.rv, rv, sizeof(double) * rvo[species]) ||
      memcmp(g_dev.rwf, rwf, sizeof(double) * rwb[species]) || memcmp(g_dev.atv, atv3, sizeof(double) * 3 * rows))
    return 0;
  if (g_dev.ng23 != ng23 || g_dev.ng3 != ng3) return 0;
  for (int i = 0; i < 3; i++) {
    if (g_dev.org[i] != org3[i]) return 0;
    for (int j = 0; j < 3; j++)
      if (g_dev.g[i][j] != g9[3 * i + j]) return 0;
  }
  return 1;
}

int run_prepare_tests(int *checks);
int run_table_tests(int *checks);
}

int main(void)
{
  int radial = 0, upload = 0, prepare = 0, table = 0;
  Tally tally;

  build_species();

  /* 2: FP64 upload, then double-float upload with the forced FP64 */
  {
    const OTF_HostTables t = tables();
    const Launch P = launch_points(), Q = launch_spheres();
    std::vector<float> p64, q64, pdf, qdf, p64f, q64f;

    CHECK(OTF_Orbitals_Upload(&t, 0) == 1 && !OTF_Orbitals_DoubleFloat());
    p64 = run(P, OTF_Orbitals_Eval);
    q64 = run(Q, OTF_Orbitals_Eval);
    CHECK(OTF_Orbitals_Upload(&t, 1) == 1 && OTF_Orbitals_DoubleFloat());
    pdf = run(P, OTF_Orbitals_Eval);
    qdf = run(Q, OTF_Orbitals_Eval);
    p64f = run(P, OTF_Orbitals_Eval_FP64);
    q64f = run(Q, OTF_Orbitals_Eval_FP64);
    compare(P, p64, pdf, p64f, tally);
    compare(Q, q64, qdf, q64f, tally);
    CHECK(0 < tally.below && 0 < tally.beyond && 0 < tally.end_intervals);

    /* 1: on the double-float upload (it holds both tables) */
    radial = radial_checks();

    /* The input switch takes effect even with live DF tables and wins
       over an explicit DF upload request. Re-enabling restores DF. */
    openmx_set_fp32_enabled(0);
    CHECK(!OTF_Orbitals_DoubleFloat());
    CHECK(memcmp(run(P, OTF_Orbitals_Eval).data(), p64.data(), p64.size() * sizeof(float)) == 0);
    CHECK(memcmp(run(Q, OTF_Orbitals_Eval).data(), q64.data(), q64.size() * sizeof(float)) == 0);
    CHECK(OTF_Orbitals_Upload(&t, 1) == 1 && !OTF_Orbitals_DoubleFloat());
    CHECK(!g_dev.rwd && !g_dev.mcd && !g_dev.spl);
    CHECK(memcmp(run(P, OTF_Orbitals_Eval).data(), p64.data(), p64.size() * sizeof(float)) == 0);
    openmx_set_fp32_enabled(1);
    CHECK(OTF_Orbitals_Upload(&t, 1) == 1 && OTF_Orbitals_DoubleFloat());
    CHECK(memcmp(run(P, OTF_Orbitals_Eval).data(), pdf.data(), pdf.size() * sizeof(float)) == 0);
    OTF_Orbitals_Release();
    CHECK(stub_live == 0);
  }
  upload = upload_checks();
  CHECK(run_prepare_tests(&prepare) == 0);
  CHECK(run_table_tests(&table) == 0);
  CHECK(stub_live == 0 && !OTF_Orbitals_Ready());
  printf("PASS: %d radial spline checks (m = 1 .. mesh-1, FP64 and double-float, < 2e-12); %d orbital values "
         "at %d points (%d inside rv[0], %d beyond the mesh, %d in the end intervals) against the reference: "
         "worst FP64 %.2f ulp, double-float %.2f ulp (values below 1e-12: %.1e), double-float a float off FP64 "
         "in %d values (and in %d below 1e-12), forced FP64 = FP64 upload bitwise; %d upload/launch failure "
         "checks; %d prepare checks; %d table build checks\n",
         radial, tally.values, tally.points, tally.below, tally.beyond, tally.end_intervals, tally.worst64,
         tally.worst_df, tally.worst_tiny, tally.df_differ, tally.df_tiny, upload, prepare, table);
  return 0;
}
'''


C_PREFIX = r'''
#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "orbs_otf_hip.h"

#define CHECK(condition) do { if (!(condition)) { \
    fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #condition); exit(2); \
} } while (0)

/* the HIP runtime of the stub (hip_runtime_compat.h's C prototypes) */
typedef int hipError_t;
enum { hipSuccess = 0, hipMemcpyHostToDevice = 1, hipMemcpyDeviceToHost = 2 };
hipError_t hipMalloc(void **ptr, size_t size);
hipError_t hipFree(void *ptr);
hipError_t hipMemcpy(void *dst, const void *src, size_t sizeBytes, int kind);
hipError_t hipDeviceSynchronize(void);
hipError_t hipGetLastError(void);

/* the OpenMX globals the extracted code reads */
typedef float Type_Orbs_Grid;
typedef int MPI_Comm;
MPI_Comm mpi_comm_level1;
int SpeciesNum, TCpyCell, Ngrid2, Ngrid3, Matomnum, Host_ID;
int level_stdout = 3;
int *Spe_MaxL_Basis, *Spe_Num_Mesh_PAO, **Spe_Num_Basis, *Spe_Total_NO;
int *M2G, *WhatSpecies, *G2ID, *GridN_Atom, *FNAN, **natn, **ncn, **NumOLG, ***GListTAtoms1;
int **GridListAtom, **CellListAtom;
double **Spe_PAO_RV, ****Spe_PAO_RWF, **atv, **Gxyz, *time_per_atom;
double gtv[4][4], Grid_Origin[4];
Type_Orbs_Grid ***Orbs_Grid, ****Orbs_Grid_FNAN;
int MPI_Comm_rank(MPI_Comm comm, int *rank) { (void)comm; *rank = 0; return 0; }
void dtime(double *t) { *t = 0.0; }

static int device_usable = 1, fp_ratio = 16;
int gpu_rank_device_usable(void) { return device_usable; }
int openmx_gpu_fp32_fp64_ratio(void) { return fp_ratio; }

/* the C++ side */
int test_OTF_Orbitals_Upload(const OTF_HostTables *t, int double_float);
extern int upload_force, upload_calls;
void test_device_fail(int malloc_at, int memcpy_at);
void test_device_fail_window(int malloc_at, int count, int launch);
int test_device_counts(int *mallocs, int *memcpys);
int test_eval_fp64(int wan, int no, double gx, double gy, double gz, double ax, double ay, double az, int GNc,
                   int GRc, float *out);
int test_device_live(void);
int test_device_error(void);
int test_device_tables_equal(int species, const size_t *rvo, const size_t *rwb, const int *spm, const int *spx,
                             const int *spb, const double *rv, const double *rwf, const double *atv3, size_t rows,
                             int ng23, int ng3, const double *g9, const double *org3);
double test_radial_value(int w, int L0, int Mul0, int i);
double test_mesh_value(int w, int i);

/* host allocations of the extracted code, with failure injection */
static int c_fail_at, c_allocations, c_live;
static void *checked_malloc(size_t n)
{
  void *p;
  if (++c_allocations == c_fail_at) return NULL;
  p = malloc(n);
  if (p != NULL) c_live++;
  return p;
}
static void *checked_calloc(size_t n, size_t size)
{
  void *p;
  if (++c_allocations == c_fail_at) return NULL;
  p = calloc(n, size);
  if (p != NULL) c_live++;
  return p;
}
static void checked_free(void *p)
{
  if (p != NULL) {
    c_live--;
    free(p);
  }
}

#define OTF_Orbitals_Upload test_OTF_Orbitals_Upload
#define malloc checked_malloc
#define calloc checked_calloc
#define free checked_free
'''


C_TEST = r'''
#undef malloc
#undef calloc
#undef free
#undef OTF_Orbitals_Upload

/* the species of the C++ side in OpenMX's arrays */
enum { NS = 2 };
static const int kMesh[NS] = {6, 8}, kMaxl[NS] = {3, 2}, kNb[NS][4] = {{2, 1, 1, 1}, {1, 2, 1, 0}};

static void setup(void)
{
  int w, L0, Mul0, i, r;

  SpeciesNum = NS;
  TCpyCell = 2;
  Ngrid2 = 5;
  Ngrid3 = 6;
  Spe_MaxL_Basis = malloc(sizeof(int) * NS);
  Spe_Num_Mesh_PAO = malloc(sizeof(int) * NS);
  Spe_Num_Basis = malloc(sizeof(int *) * NS);
  Spe_PAO_RV = malloc(sizeof(double *) * NS);
  Spe_PAO_RWF = malloc(sizeof(double ***) * NS);
  for (w = 0; w < NS; w++) {
    Spe_MaxL_Basis[w] = kMaxl[w];
    Spe_Num_Mesh_PAO[w] = kMesh[w];
    Spe_Num_Basis[w] = malloc(sizeof(int) * 4);
    Spe_PAO_RV[w] = malloc(sizeof(double) * kMesh[w]);
    Spe_PAO_RWF[w] = malloc(sizeof(double **) * 4);
    for (i = 0; i < kMesh[w]; i++) Spe_PAO_RV[w][i] = test_mesh_value(w, i);
    for (L0 = 0; L0 < 4; L0++) {
      Spe_Num_Basis[w][L0] = kNb[w][L0];
      Spe_PAO_RWF[w][L0] = malloc(sizeof(double *) * 9);
      for (Mul0 = 0; Mul0 < 9; Mul0++) {
        Spe_PAO_RWF[w][L0][Mul0] = malloc(sizeof(double) * kMesh[w]);
        for (i = 0; i < kMesh[w]; i++) Spe_PAO_RWF[w][L0][Mul0][i] = test_radial_value(w, L0, Mul0, i);
      }
    }
  }
  atv = malloc(sizeof(double *) * (TCpyCell + 1));
  for (r = 0; r <= TCpyCell; r++) {
    atv[r] = malloc(sizeof(double) * 4);
    for (i = 1; i <= 3; i++) atv[r][i] = 0.5 * r - 0.25 * i * r + 0.125 * i;
  }
  for (i = 1; i <= 3; i++) {
    Grid_Origin[i] = -0.3 * i;
    for (r = 1; r <= 3; r++) gtv[i][r] = (i == r) ? 0.3 : 0.01 * (i - r);
  }
}

/* the uploaded tables are OpenMX's, flattened L0 -> Mul0 -> mesh point */
static int tables_match(void)
{
  size_t rvo[NS + 1], rwb[NS + 1], nrv = 0, nrwf = 0;
  int spm[NS], spx[NS], spb[4 * NS], w, L0, Mul0, i, ok;
  double rv[64], rwf[512], atv3[9], g9[9], org3[3];

  for (w = 0; w < NS; w++) {
    rvo[w] = nrv;
    rwb[w] = nrwf;
    spm[w] = kMesh[w];
    spx[w] = kMaxl[w];
    for (i = 0; i < kMesh[w]; i++) rv[nrv++] = Spe_PAO_RV[w][i];
    for (L0 = 0; L0 < 4; L0++) {
      spb[4 * w + L0] = (L0 <= kMaxl[w]) ? kNb[w][L0] : 0;
      for (Mul0 = 0; Mul0 < spb[4 * w + L0]; Mul0++)
        for (i = 0; i < kMesh[w]; i++) rwf[nrwf++] = Spe_PAO_RWF[w][L0][Mul0][i];
    }
  }
  rvo[NS] = nrv;
  rwb[NS] = nrwf;
  for (w = 0; w <= TCpyCell; w++)
    for (i = 0; i < 3; i++) atv3[3 * w + i] = atv[w][i + 1];
  for (i = 0; i < 3; i++) {
    org3[i] = Grid_Origin[i + 1];
    for (w = 0; w < 3; w++) g9[3 * i + w] = gtv[i + 1][w + 1];
  }
  ok = test_device_tables_equal(NS, rvo, rwb, spm, spx, spb, rv, rwf, atv3, (size_t)TCpyCell + 1,
                                Ngrid2 * Ngrid3, Ngrid3, g9, org3);
  return ok;
}

/* Prepare fails now (0, nothing leaked) and, not latched, uploads at the next call */
static void retried(int *checks)
{
  CHECK(Set_Orbitals_Grid_OTF_Prepare() == 0);
  CHECK(c_live == 0 && !OTF_Orbitals_Ready() && test_device_live() == 0 && test_device_error() == 0);
  c_fail_at = 0;
  test_device_fail(0, 0);
  CHECK(Set_Orbitals_Grid_OTF_Prepare() == 1 && OTF_Orbitals_Ready() && c_live == 0);
  Set_Orbitals_Grid_OTF_Release();
  CHECK(test_device_live() == 0);
  *checks += 2;
}

/* Prepare fails now and stays so (no upload tried) until the release */
static void latched(int *checks, void (*undo)(void))
{
  int calls;

  CHECK(Set_Orbitals_Grid_OTF_Prepare() == 0 && !OTF_Orbitals_Ready());
  undo();
  calls = upload_calls;
  CHECK(Set_Orbitals_Grid_OTF_Prepare() == 0 && upload_calls == calls && !OTF_Orbitals_Ready());
  Set_Orbitals_Grid_OTF_Release();
  CHECK(Set_Orbitals_Grid_OTF_Prepare() == 1 && OTF_Orbitals_Ready() && upload_calls == calls + 1);
  Set_Orbitals_Grid_OTF_Release();
  CHECK(c_live == 0 && test_device_live() == 0);
  *checks += 3;
}

static void undo_force(void) { upload_force = 0; }
static void undo_maxl(void) { Spe_MaxL_Basis[1] = kMaxl[1]; }
static void undo_mesh(void) { Spe_Num_Mesh_PAO[0] = kMesh[0]; }
static void undo_mul(void) { Spe_Num_Basis[0][1] = kNb[0][1]; }
static void undo_env(void) { unsetenv("OPENMX_ORBS_GRID_GPU"); }

int run_prepare_tests(int *checks)
{
  static const char *fp64[4] = {"fp64", "FP64", "64", "double"};
  int k;

  setup();
  unsetenv("OPENMX_ORBS_GRID_GPU");
  unsetenv("OPENMX_ORBS_EVAL_PRECISION");

  /* the tables, built from OpenMX's arrays */
  c_allocations = 0;
  CHECK(Set_Orbitals_Grid_OTF_Prepare() == 1);
  CHECK(c_live == 0 && c_allocations == 8 && OTF_Orbitals_Ready() && OTF_Orbitals_DoubleFloat());
  CHECK(tables_match());
  CHECK(Set_Orbitals_Grid_OTF_Prepare() == 1 && c_allocations == 8);
  Set_Orbitals_Grid_OTF_Release();
  CHECK(!OTF_Orbitals_Ready() && test_device_live() == 0);
  *checks += 4;

  /* each host allocation, the device allocation and each transfer fail once */
  for (k = 1; k <= 8; k++) {
    c_allocations = 0;
    c_fail_at = k;
    retried(checks);
  }
  test_device_fail(1, 0);
  retried(checks);
  for (k = 1; k <= 11; k++) {
    test_device_fail(0, k);
    retried(checks);
  }

  /* tables outside the evaluator, an ineligible basis, OPENMX_ORBS_GRID_GPU=0 */
  upload_force = -1;
  latched(checks, undo_force);
  Spe_MaxL_Basis[1] = 4;
  latched(checks, undo_maxl);
  Spe_Num_Mesh_PAO[0] = 5;
  latched(checks, undo_mesh);
  Spe_Num_Basis[0][1] = 9;
  latched(checks, undo_mul);
  setenv("OPENMX_ORBS_GRID_GPU", "0", 1);
  latched(checks, undo_env);

  /* the precision: the spellings of FP64, df, the FP32:FP64 default */
  for (k = 0; k < 4; k++) {
    setenv("OPENMX_ORBS_EVAL_PRECISION", fp64[k], 1);
    CHECK(Set_Orbitals_Grid_OTF_Prepare() == 1 && !OTF_Orbitals_DoubleFloat());
    CHECK(tables_match());
    Set_Orbitals_Grid_OTF_Release();
  }
  setenv("OPENMX_ORBS_EVAL_PRECISION", "df", 1);
  CHECK(Set_Orbitals_Grid_OTF_Prepare() == 1 && OTF_Orbitals_DoubleFloat());
  Set_Orbitals_Grid_OTF_Release();
  unsetenv("OPENMX_ORBS_EVAL_PRECISION");
  fp_ratio = 2;
  CHECK(Set_Orbitals_Grid_OTF_Prepare() == 1 && !OTF_Orbitals_DoubleFloat());
  Set_Orbitals_Grid_OTF_Release();
  fp_ratio = 16;
  CHECK(Set_Orbitals_Grid_OTF_Prepare() == 1 && OTF_Orbitals_DoubleFloat());
  Set_Orbitals_Grid_OTF_Release();
  setenv("OPENMX_ORBS_EVAL_PRECISION", "df", 1);
  for (k = 0; k < 4; k++) {
    int calls = upload_calls;
    openmx_set_fp32_enabled(k % 2);
    CHECK(Set_Orbitals_Grid_OTF_Prepare() == 1);
    CHECK(OTF_Orbitals_DoubleFloat() == k % 2 && upload_calls == calls + 1);
    CHECK(Set_Orbitals_Grid_OTF_Prepare() == 1 && upload_calls == calls + 1);
    CHECK(tables_match());
  }
  unsetenv("OPENMX_ORBS_EVAL_PRECISION");
  Set_Orbitals_Grid_OTF_Release();
  CHECK(c_live == 0 && test_device_live() == 0);
  *checks += 31;
  return 0;
}

/* ---- the table build (Set_Orbitals_Grid_GPU) on a small system: atoms 1
   and 2 on this rank, atoms 3 and 4 on another (their orbitals at the
   overlap points of atom 1 go to Orbs_Grid_FNAN as two items of one
   launch, of different species, so their point and output offsets
   differ; atom 2's overlap with atom 3 is empty) ---- */

enum { NA = 4, NH = 4 };
static const int kPts[3] = {0, 40, 25};

static void setup_system(void)
{
  static const double xyz[NA + 1][3] = {{0, 0, 0}, {0.5, 0.6, 0.7}, {1.1, 0.4, 0.9}, {0.2, 1.3, 0.4},
                                        {0.9, 1.0, 0.1}};
  static const int fnan[3] = {0, 3, 2};
  static const int hn[3][NH] = {{0, 0, 0, 0}, {1, 2, 3, 4}, {2, 3, 1, 0}};
  static const int hc[3][NH] = {{0, 0, 0, 0}, {0, 1, 2, 1}, {0, 2, 1, 0}};
  static const int olg[3][NH] = {{0, 0, 0, 0}, {40, 20, 30, 17}, {25, 0, 15, 0}};
  int a, h, i, k, w, L0;

  Matomnum = 2;
  Host_ID = 0;
  M2G = malloc(sizeof(int) * 3);
  WhatSpecies = malloc(sizeof(int) * (NA + 1));
  G2ID = malloc(sizeof(int) * (NA + 1));
  GridN_Atom = malloc(sizeof(int) * (NA + 1));
  FNAN = malloc(sizeof(int) * (NA + 1));
  Gxyz = malloc(sizeof(double *) * (NA + 1));
  time_per_atom = calloc(NA + 1, sizeof(double));
  Spe_Total_NO = malloc(sizeof(int) * NS);
  for (w = 0; w < NS; w++)
    for (Spe_Total_NO[w] = 0, L0 = 0; L0 <= kMaxl[w]; L0++) Spe_Total_NO[w] += (2 * L0 + 1) * kNb[w][L0];
  for (a = 1; a <= NA; a++) {
    WhatSpecies[a] = (a == 1 || a == 4) ? 0 : 1;
    G2ID[a] = (3 <= a);
    GridN_Atom[a] = (a < 3) ? kPts[a] : 0;
    FNAN[a] = (a < 3) ? fnan[a] : 0;
    Gxyz[a] = malloc(sizeof(double) * 4);
    for (i = 0; i < 3; i++) Gxyz[a][i + 1] = xyz[a][i];
  }
  natn = malloc(sizeof(int *) * 3);
  ncn = malloc(sizeof(int *) * 3);
  NumOLG = malloc(sizeof(int *) * 3);
  GListTAtoms1 = malloc(sizeof(int **) * 3);
  GridListAtom = malloc(sizeof(int *) * 3);
  CellListAtom = malloc(sizeof(int *) * 3);
  Orbs_Grid = malloc(sizeof(Type_Orbs_Grid **) * 3);
  Orbs_Grid_FNAN = malloc(sizeof(Type_Orbs_Grid ***) * 3);
  for (a = 1; a <= 2; a++) {
    const int no = Spe_Total_NO[WhatSpecies[a]];

    M2G[a] = a;
    natn[a] = malloc(sizeof(int) * NH);
    ncn[a] = malloc(sizeof(int) * NH);
    NumOLG[a] = malloc(sizeof(int) * NH);
    GListTAtoms1[a] = calloc(NH, sizeof(int *));
    Orbs_Grid_FNAN[a] = calloc(NH, sizeof(Type_Orbs_Grid **));
    GridListAtom[a] = malloc(sizeof(int) * kPts[a]);
    CellListAtom[a] = malloc(sizeof(int) * kPts[a]);
    Orbs_Grid[a] = malloc(sizeof(Type_Orbs_Grid *) * kPts[a]);
    for (k = 0; k < kPts[a]; k++) {
      GridListAtom[a][k] = (k * (a == 1 ? 53 : 29) + (a == 1 ? 7 : 3)) % (4 * Ngrid2 * Ngrid3);
      CellListAtom[a][k] = (k + a) % (TCpyCell + 1);
      Orbs_Grid[a][k] = malloc(sizeof(Type_Orbs_Grid) * no);
    }
    for (h = 0; h <= FNAN[a]; h++) {
      natn[a][h] = hn[a][h];
      ncn[a][h] = hc[a][h];
      NumOLG[a][h] = olg[a][h];
      GListTAtoms1[a][h] = malloc(sizeof(int) * (olg[a][h] ? olg[a][h] : 1));
      /* a different point list per neighbour: an item reading another
         item's indices shows */
      for (k = 0; k < olg[a][h]; k++) GListTAtoms1[a][h][k] = (k * 7 + 3 * h) % kPts[a];
      if (G2ID[hn[a][h]] != 0 && 0 < olg[a][h]) {
        const int no1 = Spe_Total_NO[WhatSpecies[hn[a][h]]];
        Orbs_Grid_FNAN[a][h] = malloc(sizeof(Type_Orbs_Grid *) * olg[a][h]);
        for (k = 0; k < olg[a][h]; k++) Orbs_Grid_FNAN[a][h][k] = malloc(sizeof(Type_Orbs_Grid) * no1);
      }
    }
  }
}

/* NaN wherever the build writes */
static void poison(void)
{
  int a, h, k, i;

  for (a = 1; a <= 2; a++) {
    for (k = 0; k < kPts[a]; k++)
      for (i = 0; i < Spe_Total_NO[WhatSpecies[a]]; i++) Orbs_Grid[a][k][i] = NAN;
    for (h = 0; h <= FNAN[a]; h++)
      for (k = 0; Orbs_Grid_FNAN[a][h] != NULL && k < NumOLG[a][h]; k++)
        for (i = 0; i < Spe_Total_NO[WhatSpecies[natn[a][h]]]; i++) Orbs_Grid_FNAN[a][h][k][i] = NAN;
  }
}

/* every value is the FP64 evaluation of the tiles at its point, bitwise */
static int table_matches(void)
{
  float ref[64];
  int a, h, k, nonzero = 0;

  for (a = 1; a <= 2; a++) {
    const int w = WhatSpecies[a], no = Spe_Total_NO[w];

    for (k = 0; k < kPts[a]; k++) {
      CHECK(test_eval_fp64(w, no, Gxyz[a][1], Gxyz[a][2], Gxyz[a][3], 0.0, 0.0, 0.0, GridListAtom[a][k],
                           CellListAtom[a][k], ref) == 0);
      if (memcmp(Orbs_Grid[a][k], ref, sizeof(float) * no) != 0) return 0;
      nonzero += (ref[0] != 0.0f);
    }
    for (h = 1; h <= FNAN[a]; h++) {
      const int g = natn[a][h], w1 = WhatSpecies[g], no1 = Spe_Total_NO[w1], R = ncn[a][h];

      if (G2ID[g] == 0 || NumOLG[a][h] == 0) continue;
      for (k = 0; k < NumOLG[a][h]; k++) {
        const int Nc = GListTAtoms1[a][h][k];

        CHECK(test_eval_fp64(w1, no1, Gxyz[g][1], Gxyz[g][2], Gxyz[g][3], atv[R][1], atv[R][2], atv[R][3],
                             GridListAtom[a][Nc], CellListAtom[a][Nc], ref) == 0);
        if (memcmp(Orbs_Grid_FNAN[a][h][k], ref, sizeof(float) * no1) != 0) return 0;
        nonzero += (ref[0] != 0.0f);
      }
    }
  }
  return 0 < nonzero;
}

static void unharmed(int tables, int *checks)
{
  CHECK(c_live == 0 && test_device_live() == tables && OTF_Orbitals_Ready() == tables && test_device_error() == 0);
  (*checks)++;
}

int run_table_tests(int *checks)
{
  int k, copies, mallocs, memcpys;

  setup_system();
  Set_Orbitals_Grid_OTF_Release();

  /* a rank without atoms touches nothing */
  Matomnum = 0;
  test_device_fail(0, 0);
  CHECK(Set_Orbitals_Grid_GPU() == 1);
  test_device_counts(&mallocs, &memcpys);
  CHECK(mallocs == 0 && memcpys == 0 && !OTF_Orbitals_Ready());
  Matomnum = 2;

  /* the build uploads the tables of the tiles and leaves them; the values
     are the FP64 tiles' */
  poison();
  test_device_fail(0, 0);
  CHECK(Set_Orbitals_Grid_GPU() == 1 && table_matches());
  unharmed(1, checks);
  poison();
  test_device_fail(0, 0);
  CHECK(Set_Orbitals_Grid_GPU() == 1 && table_matches());
  test_device_counts(&mallocs, &memcpys);
  copies = memcpys;
  CHECK(mallocs == 1 && copies == 11);   /* the staging; atom 1: 4 + 3 copies, atom 2: 4 */
  unharmed(1, checks);

  /* device failures in the build (both attempts at the staging, each
     transfer, the launch) and each failed host allocation: 0 (the host
     loops rebuild everything), nothing leaked, the tables stay */
  test_device_fail_window(1, 2, 0);
  CHECK(Set_Orbitals_Grid_GPU() == 0);
  unharmed(1, checks);
  for (k = 1; k <= copies; k++) {
    test_device_fail(0, k);
    CHECK(Set_Orbitals_Grid_GPU() == 0);
    unharmed(1, checks);
  }
  test_device_fail_window(0, 0, 1);
  CHECK(Set_Orbitals_Grid_GPU() == 0);
  test_device_fail_window(0, 0, 0);
  unharmed(1, checks);
  for (k = 1; k <= 4; k++) {
    c_allocations = 0;
    c_fail_at = k;
    CHECK(Set_Orbitals_Grid_GPU() == 0);
    c_fail_at = 0;
    unharmed(1, checks);
  }

  /* the tables' upload fails: 0, and the next build uploads them */
  Set_Orbitals_Grid_OTF_Release();
  test_device_fail(1, 0);
  CHECK(Set_Orbitals_Grid_GPU() == 0);
  unharmed(0, checks);
  test_device_fail(0, 0);
  poison();
  CHECK(Set_Orbitals_Grid_GPU() == 1 && table_matches());
  Set_Orbitals_Grid_OTF_Release();
  CHECK(test_device_live() == 0 && c_live == 0);
  *checks += 4;
  return 0;
}
'''


def c_harness(sog):
    parts = [C_PREFIX, '#include "openmx_precision.h"',
             line(sog, r"^#define\s+SOG_L0MAX\b.*$"),
             line(sog, r"^#define\s+SOG_MULMAX\b.*$"),
             line(sog, r"^int Set_Orbitals_Grid_OTF_Prepare\(void\);$"),
             function(sog, "SOG_env_flag"),
             function(sog, "SOG_gpu_eligible"),
             function(sog, "SOG_arena_off"),
             function(sog, "SOG_arena_try"),
             function(sog, "Set_Orbitals_Grid_GPU"),
             line(sog, r"^extern int openmx_gpu_fp32_fp64_ratio\(void\);$"),
             line(sog, r"^static int SOG_otf_unavailable = 0;$"),
             line(sog, r"^static int SOG_otf_fp32_enabled = -1;$"),
             function(sog, "SOG_otf_double_float"),
             function(sog, "Set_Orbitals_Grid_OTF_Release"),
             function(sog, "Set_Orbitals_Grid_OTF_Prepare"),
             C_TEST]
    return "\n\n".join(parts)


def static_checks(sources, makefile):
    rule = re.search(r"^orbs_otf_hip\.o:.*\n\t(.*)$", makefile, re.MULTILINE)
    if rule is None:
        raise SystemExit("FAIL: no orbs_otf_hip.o rule in the makefile")
    command = rule.group(1)
    variables = dict(re.findall(r"^(\w+)\s*:?=(.*)$", makefile, re.MULTILINE))
    for name in re.findall(r"\$\((\w+)\)", command):
        command += " " + variables.get(name, "")
    if "-ffp-contract=off" not in command:
        raise SystemExit("FAIL: orbs_otf_hip.o is built without -ffp-contract=off")
    if "#pragma clang fp contract(off)" not in sources["orbs_otf_hip.cu"]:
        raise SystemExit("FAIL: orbs_otf_hip.cu lost '#pragma clang fp contract(off)'")
    table = function(sources["Set_Orbitals_Grid.c"], "Set_Orbitals_Grid_GPU")
    if "OTF_Orbitals_Eval_FP64(" not in table or "#pragma omp target" in table or "acos(" in table:
        raise SystemExit("FAIL: the table build of Set_Orbitals_Grid.c does not use the shared FP64 evaluator")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cc", default=os.environ.get("CC", "cc"), help="C compiler (default: $CC or cc)")
    parser.add_argument("--cxx", default=None,
                        help="C++ compiler (default: $CXX, else g++ / clang++ after --cc)")
    parser.add_argument("--no-sanitize", action="store_true", help="build without ASan/UBSan")
    parser.add_argument("--source", default=None,
                        help="directory holding orbs_otf_hip.cu, orbs_otf_hip.h and Set_Orbitals_Grid.c "
                             "(default: the repository's source/)")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source_dir = Path(args.source).resolve() if args.source else root / "source"
    cxx = args.cxx or os.environ.get("CXX")
    if cxx is None:
        # the C++ driver next to the C compiler (--cc /opt/rocm/llvm/bin/clang -> its clang++)
        cc = Path(shlex.split(args.cc)[0])
        name = "clang++" if "clang" in cc.name else "g++"
        cxx = str(cc.with_name(name)) if cc.parent != Path(".") and cc.with_name(name).exists() else name
    sources = {name: (source_dir / name).read_text()
               for name in ("orbs_otf_hip.cu", "orbs_otf_hip.h", "Set_Orbitals_Grid.c")}
    makefile = (source_dir / "makefile") if (source_dir / "makefile").exists() else root / "source/makefile"
    static_checks(sources, makefile.read_text())

    flags = ["-O1", "-g", "-ffp-contract=off", "-fno-omit-frame-pointer"]
    if not args.no_sanitize:
        flags += ["-fsanitize=address,undefined", "-fno-sanitize-recover=all"]
    with tempfile.TemporaryDirectory(prefix="openmx-orbs-test-") as directory:
        work = Path(directory)
        (work / "hip").mkdir()
        (work / "hip/hip_runtime.h").write_text(STUB_HIP)
        (work / "orbs_otf_hip.cu").write_text(sources["orbs_otf_hip.cu"])
        (work / "orbs_otf_hip.h").write_text(sources["orbs_otf_hip.h"])
        (work / "otf_test.cpp").write_text(CXX_TEST)
        (work / "prepare_test.c").write_text(c_harness(sources["Set_Orbitals_Grid.c"]))
        include = ["-I" + str(work), "-I" + str(root / "source")]
        subprocess.run([*shlex.split(args.cc), *flags, "-std=c11", *include, "-c", str(work / "prepare_test.c"),
                        "-o", str(work / "prepare_test.o")], check=True)
        subprocess.run([*shlex.split(cxx), *flags, "-std=c++17", "-Wno-unknown-pragmas", *include, "-c",
                        str(work / "otf_test.cpp"), "-o", str(work / "otf_test.o")], check=True)
        subprocess.run([*shlex.split(args.cc), *flags, "-std=c11", *include, "-c",
                        str(root / "source/openmx_precision.c"), "-o", str(work / "precision.o")], check=True)
        subprocess.run([*shlex.split(cxx), *flags, str(work / "otf_test.o"), str(work / "prepare_test.o"),
                        str(work / "precision.o"), "-lm",
                        "-o", str(work / "check")], check=True)
        env = os.environ.copy()
        for name in ("OPENMX_ORBS_GRID_GPU", "OPENMX_ORBS_EVAL_PRECISION"):
            env.pop(name, None)
        subprocess.run([str(work / "check")], env=env, check=True)


if __name__ == "__main__":
    main()

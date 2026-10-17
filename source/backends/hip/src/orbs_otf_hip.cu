/**********************************************************************
  orbs_otf_hip.cu

  Device evaluation of the basis orbitals (see orbs_otf_hip.h).  HIP port
  of the NVIDIA version's SOG_point_eval / SOG_point_eval_df and
  SOG_Device_* (Set_Orbitals_Grid.c).  This translation unit must be
  compiled without floating-point contraction (-ffp-contract=off): the
  double-float arithmetic below relies on rounded products staying
  rounded (an error-free transformation fused into a later addition
  collapses to FP32 accuracy); the explicit fmaf() calls stay fused.
  The pragma below keeps that when the flag is lost (hipcc's default,
  fast-honor-pragmas, honors it).
***********************************************************************/
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <vector>

#include "orbs_otf_hip.h"
#include "openmx_precision.h"

#pragma clang fp contract(off)

namespace {

constexpr int kThreads = 128;
constexpr int kMcdStride = 12;   /* per mesh point: rv, h2, dum1..dum4 as (hi, lo) */
constexpr double kPi = 3.141592653589793238;

struct DeviceTables {
  int ready = 0;
  int double_float = 0;
  unsigned char *arena = nullptr;
  size_t bytes = 0;
  const double *rv = nullptr, *rwf = nullptr, *atv = nullptr;
  const size_t *rvo = nullptr, *rwb = nullptr;
  const int *spm = nullptr, *spx = nullptr, *spb = nullptr;
  const float *rwd = nullptr, *mcd = nullptr, *spl = nullptr;
  int ng23 = 0, ng3 = 0;
  double g[3][3] = {{0.0}};
  double org[3] = {0.0};
};

DeviceTables g_dev;

/* the kernel's view: plain pointers and values passed by argument */
struct TablesArg {
  const double *rv, *rwf, *atv;
  const size_t *rvo, *rwb;
  const int *spm, *spx, *spb;
  const float *rwd, *mcd, *spl;
  int ng23, ng3;
  double g11, g12, g13, g21, g22, g23, g31, g32, g33;
  double org1, org2, org3;
};

/* ---------------------------------------------------------------------
   FP64: a transcription of the inlined Get_Orbitals of the host loops of
   Set_Orbitals_Grid.c, without the angles (the real harmonics need only
   cos/sin of theta and phi: z/r, r1/r, x/r1, y/r1), and with the radial
   mesh index estimated from the logarithmic mesh, then made exact.  The
   tables Orbs_Grid / Orbs_Grid_FNAN are built with it too
   (OTF_Orbitals_Eval_FP64), so the FP64 tiles equal them bit for bit.
   The end intervals m = 1 and m = mesh - 1 take the substitutes of the
   host loops (reflected endpoint samples for h1/f1 and h3/f4) before
   any load and before forming the spline weights dum1..dum4, as the
   double-float tables do.  Each (L0, Mul0) is written as
   it is computed: locally indexed arrays would live in scratch memory.
   --------------------------------------------------------------------- */

/* the radial spline at R: the mesh interval m with its weights, or
   (below) R inside the first mesh point, extrapolated from the interval
   of rm = rv[4] */
struct Spline64 {
  int m, mesh, below;
  double R, rm, h2, dum1, dum2, dum3, dum4, y1, y2, y12, y22;
};

/* the spline at R (rv[0] <= R <= rv[mesh - 1], or below) on the mesh rv
   of mesh points */
__device__ inline Spline64 spline64(const double *rv, int mesh, double R)
{
  Spline64 s;
  double h1, h3, x1, x2, dum;

  s.mesh = mesh;
  s.R = R;
  s.below = (R < rv[0]);
  s.rm = 0.0;
  if (s.below) {
    s.m = 4;
    s.rm = rv[4];
    h1 = rv[3] - rv[2];
    s.h2 = rv[4] - rv[3];
    h3 = rv[5] - rv[4];
    x1 = s.rm - rv[3];
    x2 = s.rm - rv[4];
  }
  else {
    /* the first mesh index with rv[m] >= R (what the host's bisection
       finds): an estimate from the logarithmic mesh, then the exact
       neighbours */
    const double x0 = log(rv[0]);
    const double dx = (log(rv[mesh - 1]) - x0) / (double)(mesh - 1);
    int m = (int)((log(R) - x0) / dx) + 1;

    if (m < 1) m = 1;
    if (mesh - 1 < m) m = mesh - 1;
    while (1 < m && R <= rv[m - 1]) m--;
    while (m < mesh - 1 && rv[m] < R) m++;

    s.m = m;
    s.h2 = rv[m] - rv[m - 1];
    if (m == 1) {
      h3 = rv[m + 1] - rv[m];
      h1 = rv[m - 1] - rv[m + 1];
    }
    else if (m == mesh - 1) {
      h1 = rv[m - 1] - rv[m - 2];
      h3 = rv[m - 2] - rv[m];
    }
    else {
      h1 = rv[m - 1] - rv[m - 2];
      h3 = rv[m + 1] - rv[m];
    }
    x1 = R - rv[m - 1];
    x2 = R - rv[m];
  }

  s.y1 = x1 / s.h2;
  s.y2 = x2 / s.h2;
  s.y12 = s.y1 * s.y1;
  s.y22 = s.y2 * s.y2;

  dum = h1 + s.h2;
  s.dum1 = h1 / s.h2 / dum;
  s.dum2 = s.h2 / h1 / dum;
  dum = s.h2 + h3;
  s.dum3 = s.h2 / h3 / dum;
  s.dum4 = h3 / s.h2 / dum;
  return s;
}

/* one radial function (rw: its table) of angular momentum L0 */
__device__ inline double radial_fp64(const double *rw, int L0, const Spline64 &s)
{
  /* the boundary indices replaced before the loads */
  const int m = s.m;
  const int i1 = m == 1 ? m + 1 : m - 2;
  const int i4 = m == s.mesh - 1 ? m - 2 : m + 1;
  const double f1 = rw[i1];
  const double f2 = rw[m - 1];
  const double f3 = rw[m];
  const double f4 = rw[i4];
  const double h2 = s.h2, y1 = s.y1, y2 = s.y2, y12 = s.y12, y22 = s.y22;
  const double dum = f3 - f2;
  const double g1 = dum * s.dum1 + (f2 - f1) * s.dum2;
  const double g2 = (f4 - f3) * s.dum3 + dum * s.dum4;
  const double f = y22 * (3.0 * f2 + h2 * g1 + (2.0 * f2 + h2 * g1) * y2) +
                   y12 * (3.0 * f3 - h2 * g2 - (2.0 * f3 - h2 * g2) * y1);

  if (s.below) {
    const double rm = s.rm, R = s.R;
    const double df = 2.0 * y2 / h2 * (3.0 * f2 + h2 * g1 + (2.0 * f2 + h2 * g1) * y2) +
                      y22 * (2.0 * f2 + h2 * g1) / h2 +
                      2.0 * y1 / h2 * (3.0 * f3 - h2 * g2 - (2.0 * f3 - h2 * g2) * y1) -
                      y12 * (2.0 * f3 - h2 * g2) / h2;
    double a, b, c, d;

    if (L0 == 0) {
      a = 0.0;
      b = 0.5 * df / rm;
      c = 0.0;
      d = f - b * rm * rm;
    }
    else if (L0 == 1) {
      a = (rm * df - f) / (2.0 * rm * rm * rm);
      b = 0.0;
      c = df - 3.0 * a * rm * rm;
      d = 0.0;
    }
    else {
      b = (3.0 * f - rm * df) / (rm * rm);
      a = (f - b * rm * rm) / (rm * rm * rm);
      c = 0.0;
      d = 0.0;
    }
    return a * R * R * R + b * R * R + c * R + d;
  }
  return f;
}

__device__ __forceinline__ void otf_point_eval(const TablesArg &t, int wan, int no, double gx, double gy, double gz,
                                               double ax, double ay, double az, int GNc, int GRc, float *out)
{
  const int n1 = GNc / t.ng23;
  const int n2 = (GNc - n1 * t.ng23) / t.ng3;
  const int n3 = GNc - n1 * t.ng23 - n2 * t.ng3;

  const double x = ((double)n1 * t.g11 + (double)n2 * t.g21 + (double)n3 * t.g31 + t.org1) +
                   t.atv[3 * (size_t)GRc + 0] - gx - ax;
  const double y = ((double)n1 * t.g12 + (double)n2 * t.g22 + (double)n3 * t.g32 + t.org2) +
                   t.atv[3 * (size_t)GRc + 1] - gy - ay;
  const double z = ((double)n1 * t.g13 + (double)n2 * t.g23 + (double)n3 * t.g33 + t.org3) +
                   t.atv[3 * (size_t)GRc + 2] - gz - az;

  const int mesh = t.spm[wan];
  const int maxl = t.spx[wan];
  const double *rv = t.rv + t.rvo[wan];

  double R;
  double siQ, coQ, siP, coP;
  Spline64 s;
  size_t rrow;
  int L0, Mul0, i1;

  {
    const double Min_r = 10e-15;
    const double dum = x * x + y * y;
    const double r = sqrt(dum + z * z);
    const double r1 = sqrt(dum);

    if (Min_r <= r) {
      if (r < fabs(z)) {
        coQ = (z < 0.0 ? -1.0 : 1.0);
        siQ = 0.0;
      }
      else {
        coQ = z / r;
        siQ = r1 / r;
      }
      if (Min_r <= r1) {
        if (r1 < fabs(y)) {
          siP = (y < 0.0 ? -1.0 : 1.0);
          coP = 0.0;
        }
        else {
          siP = y / r1;
          coP = x / r1;
        }
      }
      else {
        siP = 0.0;
        coP = 1.0;
      }
    }
    else {
      coQ = 0.0;
      siQ = 1.0;
      siP = 0.0;
      coP = 1.0;
    }
    R = r;
  }

  if (rv[mesh - 1] < R) {
    /* outside the PAO mesh: every orbital is zero */
    for (i1 = 0; i1 < no; i1++) out[i1] = 0.0f;
    return;
  }

  s = spline64(rv, mesh, R);
  rrow = t.rwb[wan];
  i1 = 0;
  for (L0 = 0; L0 <= maxl; L0++) {
    const int nb = t.spb[wan * (OTF_L0MAX + 1) + L0];
    double A0, A1 = 0.0, A2 = 0.0, A3 = 0.0, A4 = 0.0, A5 = 0.0, A6 = 0.0;

    if (L0 == 0) {
      A0 = 0.282094791773878;
    }
    else if (L0 == 1) {
      const double dum = 0.48860251190292 * siQ;
      A0 = dum * coP;
      A1 = dum * siP;
      A2 = 0.48860251190292 * coQ;
    }
    else if (L0 == 2) {
      const double dum1 = siQ * siQ;
      const double dum2 = 1.09254843059208 * siQ * coQ;
      A0 = 0.94617469575756 * coQ * coQ - 0.31539156525252;
      A1 = 0.54627421529604 * dum1 * (1.0 - 2.0 * siP * siP);
      A2 = 1.09254843059208 * dum1 * siP * coP;
      A3 = dum2 * coP;
      A4 = dum2 * siP;
    }
    else {
      A0 = 0.373176332590116 * (5.0 * coQ * coQ * coQ - 3.0 * coQ);
      A1 = 0.457045799464466 * coP * siQ * (5.0 * coQ * coQ - 1.0);
      A2 = 0.457045799464466 * siP * siQ * (5.0 * coQ * coQ - 1.0);
      A3 = 1.44530572132028 * siQ * siQ * coQ * (coP * coP - siP * siP);
      A4 = 2.89061144264055 * siQ * siQ * coQ * siP * coP;
      A5 = 0.590043589926644 * siQ * siQ * siQ * (4.0 * coP * coP * coP - 3.0 * coP);
      A6 = 0.590043589926644 * siQ * siQ * siQ * (3.0 * siP - 4.0 * siP * siP * siP);
    }
    for (Mul0 = 0; Mul0 < nb; Mul0++) {
      const double RF = radial_fp64(t.rwf + rrow, L0, s);

      rrow += (size_t)mesh;
      out[i1] = (float)(RF * A0);
      if (1 <= L0) {
        out[i1 + 1] = (float)(RF * A1);
        out[i1 + 2] = (float)(RF * A2);
      }
      if (2 <= L0) {
        out[i1 + 3] = (float)(RF * A3);
        out[i1 + 4] = (float)(RF * A4);
      }
      if (3 <= L0) {
        out[i1 + 5] = (float)(RF * A5);
        out[i1 + 6] = (float)(RF * A6);
      }
      i1 += 2 * L0 + 1;
    }
  }
}

/* ---------------------------------------------------------------------
   Double-float: everything after the FP64 grid coordinates is a pair of
   floats (hi + lo, |lo| <= ulp(hi)/2, about 2^-48 relative); the radial
   tables and the per-mesh spline constants are held as such pairs, and
   the float written for an orbital is the hi part of its double-float
   product (the float nearest the FP64 value but within ~1e-14 of a
   rounding boundary).  Points inside the first mesh point (the cubic
   extrapolation) take the FP64 routine.  The algebra follows
   otf_point_eval.
   --------------------------------------------------------------------- */
struct df {
  float hi, lo;
};

__device__ inline df df_mk(float hi, float lo)
{
  df r;
  r.hi = hi;
  r.lo = lo;
  return r;
}

/* |a| >= |b| (or a == 0) */
__device__ inline df df_qts(float a, float b)
{
  df r;
  r.hi = a + b;
  r.lo = b - (r.hi - a);
  return r;
}

__device__ inline df df_dbl(double x)
{
  const float hi = (float)x;
  return df_mk(hi, (float)(x - (double)hi));
}

__device__ inline df df_add(df a, df b)
{
  const float s = a.hi + b.hi;
  const float v = s - a.hi;
  const float e = ((a.hi - (s - v)) + (b.hi - v)) + (a.lo + b.lo);
  return df_qts(s, e);
}

__device__ inline df df_sub(df a, df b)
{
  return df_add(a, df_mk(-b.hi, -b.lo));
}

__device__ inline df df_mul(df a, df b)
{
  const float p = a.hi * b.hi;
  const float e = fmaf(a.hi, b.hi, -p) + (a.hi * b.lo + a.lo * b.hi);
  return df_qts(p, e);
}

/* b an exact float (a small integer) */
__device__ inline df df_mulf(df a, float b)
{
  const float p = a.hi * b;
  const float e = fmaf(a.hi, b, -p) + a.lo * b;
  return df_qts(p, e);
}

__device__ inline df df_div(df a, df b)
{
  const float q1 = a.hi / b.hi;
  const float p = q1 * b.hi;
  const df r = df_sub(a, df_qts(p, fmaf(q1, b.hi, -p) + q1 * b.lo));
  return df_qts(q1, r.hi / b.hi);
}

__device__ inline df df_sqrt(df a)
{
  float s1, p;
  df r;

  if (a.hi <= 0.0f) return df_mk(0.0f, 0.0f);
  s1 = sqrtf(a.hi);
  p = s1 * s1;
  r = df_sub(a, df_mk(p, fmaf(s1, s1, -p)));
  return df_qts(s1, r.hi / (2.0f * s1));
}

/* the cubic spline of one radial function (rw2: its (hi, lo) table) at
   the mesh interval m with the interval's constants */
__device__ inline df radial_df(const float *rw2, int m, int mesh, df h2, df dum1, df dum2, df dum3, df dum4, df y1,
                               df y2, df y12, df y22)
{
  /* the boundary indices replaced before the loads (the end intervals
     would read outside the radial table otherwise) */
  const int i1 = m == 1 ? m + 1 : m - 2;
  const int i4 = m == mesh - 1 ? m - 2 : m + 1;
  const df f1 = df_mk(rw2[2 * i1], rw2[2 * i1 + 1]);
  const df f2 = df_mk(rw2[2 * (m - 1)], rw2[2 * (m - 1) + 1]);
  const df f3 = df_mk(rw2[2 * m], rw2[2 * m + 1]);
  const df f4 = df_mk(rw2[2 * i4], rw2[2 * i4 + 1]);
  df d32, g1, g2, P, Q, tt, u;

  d32 = df_sub(f3, f2);
  g1 = df_add(df_mul(d32, dum1), df_mul(df_sub(f2, f1), dum2));
  g2 = df_add(df_mul(df_sub(f4, f3), dum3), df_mul(d32, dum4));
  P = df_mul(h2, g1);
  Q = df_mul(h2, g2);
  /* y2^2 (3 f2 + P + (2 f2 + P) y2) + y1^2 (3 f3 - Q - (2 f3 - Q) y1) */
  tt = df_add(df_mulf(f2, 2.0f), P);
  tt = df_add(df_add(tt, f2), df_mul(tt, y2));
  u = df_sub(df_mulf(f3, 2.0f), Q);
  u = df_sub(df_add(u, f3), df_mul(u, y1));
  return df_add(df_mul(y22, tt), df_mul(y12, u));
}

__device__ void otf_point_eval_df(const TablesArg &t, int wan, int no, double gx, double gy, double gz, double ax,
                                  double ay, double az, int GNc, int GRc, float *out)
{
  /* the grid coordinates in FP64: the small difference of large values */
  const int n1 = GNc / t.ng23;
  const int n2 = (GNc - n1 * t.ng23) / t.ng3;
  const int n3 = GNc - n1 * t.ng23 - n2 * t.ng3;

  const double x = ((double)n1 * t.g11 + (double)n2 * t.g21 + (double)n3 * t.g31 + t.org1) +
                   t.atv[3 * (size_t)GRc + 0] - gx - ax;
  const double y = ((double)n1 * t.g12 + (double)n2 * t.g22 + (double)n3 * t.g32 + t.org2) +
                   t.atv[3 * (size_t)GRc + 1] - gy - ay;
  const double z = ((double)n1 * t.g13 + (double)n2 * t.g23 + (double)n3 * t.g33 + t.org3) +
                   t.atv[3 * (size_t)GRc + 2] - gz - az;

  const int mesh = t.spm[wan];
  const int maxl = t.spx[wan];
  const double *rv = t.rv + t.rvo[wan];
  const float *mcd = t.mcd + kMcdStride * t.rvo[wan];
  const df one = df_mk(1.0f, 0.0f), zero = df_mk(0.0f, 0.0f);
  const df xd = df_dbl(x), yd = df_dbl(y), zd = df_dbl(z);
  const df dsq = df_add(df_mul(xd, xd), df_mul(yd, yd));
  const df r = df_sqrt(df_add(dsq, df_mul(zd, zd)));
  const df r1 = df_sqrt(dsq);
  const double R = (double)r.hi + (double)r.lo;
  const double R1 = (double)r1.hi + (double)r1.lo;
  df siQ, coQ, siP, coP;
  df h2, dum1, dum2, dum3, dum4, y1, y2, y12, y22;
  size_t rrow;
  int m, Mul0, i1;

  if (rv[mesh - 1] < R) {
    for (i1 = 0; i1 < no; i1++) out[i1] = 0.0f;
    return;
  }
  if (R < rv[0]) {
    /* inlined: without its local arrays the FP64 routine costs this kernel
       no scratch (gfx1200: 116 VGPRs, 12 waves per SIMD), whereas a call
       (__noinline__) takes 216 VGPRs and 208 B of scratch per lane for the
       call frame (7 waves per SIMD) */
    otf_point_eval(t, wan, no, gx, gy, gz, ax, ay, az, GNc, GRc, out);
    return;
  }

  {
    const double Min_r = 10e-15;

    if (Min_r <= R) {
      if (R < fabs(z)) {
        coQ = df_mk(z < 0.0 ? -1.0f : 1.0f, 0.0f);
        siQ = zero;
      }
      else {
        const df ir = df_div(one, r);
        coQ = df_mul(zd, ir);
        siQ = df_mul(r1, ir);
      }
      if (Min_r <= R1) {
        if (R1 < fabs(y)) {
          siP = df_mk(y < 0.0 ? -1.0f : 1.0f, 0.0f);
          coP = zero;
        }
        else {
          const df ir1 = df_div(one, r1);
          siP = df_mul(yd, ir1);
          coP = df_mul(xd, ir1);
        }
      }
      else {
        siP = zero;
        coP = one;
      }
    }
    else {
      coQ = zero;
      siQ = one;
      siP = zero;
      coP = one;
    }
  }

  {
    const float x0 = t.spl[2 * wan], dx = t.spl[2 * wan + 1];

    m = (int)((logf(r.hi) - x0) / dx) + 1;
    if (m < 1) m = 1;
    if (mesh - 1 < m) m = mesh - 1;
    while (1 < m && R <= rv[m - 1]) m--;
    while (m < mesh - 1 && rv[m] < R) m++;
  }
  {
    const float *mc = mcd + kMcdStride * m;
    const df rvm1 = df_mk(mc[-kMcdStride], mc[-kMcdStride + 1]);

    h2 = df_mk(mc[2], mc[3]);
    dum1 = df_mk(mc[4], mc[5]);
    dum2 = df_mk(mc[6], mc[7]);
    dum3 = df_mk(mc[8], mc[9]);
    dum4 = df_mk(mc[10], mc[11]);
    y1 = df_div(df_sub(r, rvm1), h2);
    y2 = df_sub(y1, one);
    y12 = df_mul(y1, y1);
    y22 = df_mul(y2, y2);
  }

  rrow = t.rwb[wan];
  i1 = 0;
  {
    const int nb = t.spb[wan * (OTF_L0MAX + 1) + 0];
    const df c00 = df_mk(0.282094806432724f, -1.4658845692849809e-08f);

    for (Mul0 = 0; Mul0 < nb; Mul0++) {
      const df RF = radial_df(t.rwd + 2 * rrow, m, mesh, h2, dum1, dum2, dum3, dum4, y1, y2, y12, y22);
      rrow += (size_t)mesh;
      out[i1++] = df_mul(RF, c00).hi;
    }
  }
  if (1 <= maxl) {
    const int nb = t.spb[wan * (OTF_L0MAX + 1) + 1];
    const df c1 = df_mk(0.48860251903533936f, -7.1324195438648985e-09f);
    const df d = df_mul(c1, siQ);
    const df A0 = df_mul(d, coP);
    const df A1 = df_mul(d, siP);
    const df A2 = df_mul(c1, coQ);

    for (Mul0 = 0; Mul0 < nb; Mul0++) {
      const df RF = radial_df(t.rwd + 2 * rrow, m, mesh, h2, dum1, dum2, dum3, dum4, y1, y2, y12, y22);
      rrow += (size_t)mesh;
      out[i1 + 0] = df_mul(RF, A0).hi;
      out[i1 + 1] = df_mul(RF, A1).hi;
      out[i1 + 2] = df_mul(RF, A2).hi;
      i1 += 3;
    }
  }
  if (2 <= maxl) {
    const int nb = t.spb[wan * (OTF_L0MAX + 1) + 2];
    const df c20 = df_mk(0.946174681186676f, 1.4570884054876387e-08f);
    const df c21 = df_mk(0.31539157032966614f, -5.077146258969378e-09f);
    const df c22 = df_mk(0.5462742447853088f, -2.948926791646045e-08f);
    const df c23 = df_mk(1.0925484895706177f, -5.89785358329209e-08f);
    const df s2 = df_mul(siQ, siQ);
    const df d2 = df_mul(c23, df_mul(siQ, coQ));
    const df A0 = df_sub(df_mul(c20, df_mul(coQ, coQ)), c21);
    const df A1 = df_mul(df_mul(c22, s2), df_sub(one, df_mulf(df_mul(siP, siP), 2.0f)));
    const df A2 = df_mul(df_mul(c23, s2), df_mul(siP, coP));
    const df A3 = df_mul(d2, coP);
    const df A4 = df_mul(d2, siP);

    for (Mul0 = 0; Mul0 < nb; Mul0++) {
      const df RF = radial_df(t.rwd + 2 * rrow, m, mesh, h2, dum1, dum2, dum3, dum4, y1, y2, y12, y22);
      rrow += (size_t)mesh;
      out[i1 + 0] = df_mul(RF, A0).hi;
      out[i1 + 1] = df_mul(RF, A1).hi;
      out[i1 + 2] = df_mul(RF, A2).hi;
      out[i1 + 3] = df_mul(RF, A3).hi;
      out[i1 + 4] = df_mul(RF, A4).hi;
      i1 += 5;
    }
  }
  if (3 <= maxl) {
    const int nb = t.spb[wan * (OTF_L0MAX + 1) + 3];
    const df c30 = df_mk(0.37317633628845215f, -3.6983360818254596e-09f);
    const df c31 = df_mk(0.4570457935333252f, 5.931140911741295e-09f);
    const df c33 = df_mk(1.4453057050704956f, 1.6249783740818202e-08f);
    const df c34 = df_mk(2.890611410140991f, 3.249955682349537e-08f);
    const df c35 = df_mk(0.5900436043739319f, -1.4447287810526177e-08f);
    const df cQ2 = df_mul(coQ, coQ);
    const df sQ2 = df_mul(siQ, siQ);
    const df sQ3 = df_mul(sQ2, siQ);
    const df cP2 = df_mul(coP, coP);
    const df sP2 = df_mul(siP, siP);
    const df five_cQ2_m1 = df_sub(df_mulf(cQ2, 5.0f), one);
    const df A0 = df_mul(c30, df_mul(coQ, df_sub(df_mulf(cQ2, 5.0f), df_mk(3.0f, 0.0f))));
    const df A1 = df_mul(c31, df_mul(df_mul(coP, siQ), five_cQ2_m1));
    const df A2 = df_mul(c31, df_mul(df_mul(siP, siQ), five_cQ2_m1));
    const df A3 = df_mul(c33, df_mul(df_mul(sQ2, coQ), df_sub(cP2, sP2)));
    const df A4 = df_mul(c34, df_mul(df_mul(sQ2, coQ), df_mul(siP, coP)));
    const df A5 = df_mul(c35, df_mul(sQ3, df_mul(coP, df_sub(df_mulf(cP2, 4.0f), df_mk(3.0f, 0.0f)))));
    const df A6 = df_mul(c35, df_mul(sQ3, df_mul(siP, df_sub(df_mk(3.0f, 0.0f), df_mulf(sP2, 4.0f)))));

    for (Mul0 = 0; Mul0 < nb; Mul0++) {
      const df RF = radial_df(t.rwd + 2 * rrow, m, mesh, h2, dum1, dum2, dum3, dum4, y1, y2, y12, y22);
      rrow += (size_t)mesh;
      out[i1 + 0] = df_mul(RF, A0).hi;
      out[i1 + 1] = df_mul(RF, A1).hi;
      out[i1 + 2] = df_mul(RF, A2).hi;
      out[i1 + 3] = df_mul(RF, A3).hi;
      out[i1 + 4] = df_mul(RF, A4).hi;
      out[i1 + 5] = df_mul(RF, A5).hi;
      out[i1 + 6] = df_mul(RF, A6).hi;
      i1 += 7;
    }
  }
}

template <bool DF>
__global__ __launch_bounds__(kThreads) void otf_eval_kernel(TablesArg t, int count, const OTF_Atom *__restrict__ atoms,
                                                            const int *__restrict__ npts,
                                                            const size_t *__restrict__ idx_off,
                                                            const size_t *__restrict__ out_off,
                                                            const size_t *__restrict__ sph_off,
                                                            const int *__restrict__ idx,
                                                            const int *__restrict__ gla,
                                                            const int *__restrict__ cla, float *__restrict__ out)
{
  const int p = blockIdx.x;

  if (p >= count) return;
  {
    const OTF_Atom a = atoms[p];
    const int np = npts[p];
    const size_t ioff = (idx != nullptr) ? idx_off[p] : 0;
    const size_t ooff = out_off[p], soff = sph_off[p];

    for (int k = blockIdx.y * blockDim.x + threadIdx.x; k < np; k += gridDim.y * blockDim.x) {
      const int Nc = (idx != nullptr) ? idx[ioff + (size_t)k] : k;
      const int GNc = gla[soff + (size_t)Nc], GRc = cla[soff + (size_t)Nc];
      float *o = out + ooff + (size_t)k * (size_t)a.no;

      if (DF) otf_point_eval_df(t, a.wan, a.no, a.gx, a.gy, a.gz, a.ax, a.ay, a.az, GNc, GRc, o);
      else otf_point_eval(t, a.wan, a.no, a.gx, a.gy, a.gz, a.ax, a.ay, a.az, GNc, GRc, o);
    }
  }
}

size_t arena_off(size_t *pos, size_t bytes)
{
  const size_t off = *pos;

  *pos = (off + bytes + 511u) & ~(size_t)511u;
  return off;
}

/* the launch of OTF_Orbitals_Eval (double_float: the uploaded precision)
   and of OTF_Orbitals_Eval_FP64 */
int otf_launch(bool double_float, void *stream, int count, int max_npts, const OTF_Atom *atoms, const int *npts,
               const size_t *idx_off, const size_t *out_off, const size_t *sph_off, const int *idx, const int *gla,
               const int *cla, float *out)
{
  TablesArg t;
  unsigned ychunks;

  if (!g_dev.ready) return 1;
  if (count <= 0 || max_npts <= 0) return 0;
  t.rv = g_dev.rv;
  t.rwf = g_dev.rwf;
  t.atv = g_dev.atv;
  t.rvo = g_dev.rvo;
  t.rwb = g_dev.rwb;
  t.spm = g_dev.spm;
  t.spx = g_dev.spx;
  t.spb = g_dev.spb;
  t.rwd = g_dev.rwd;
  t.mcd = g_dev.mcd;
  t.spl = g_dev.spl;
  t.ng23 = g_dev.ng23;
  t.ng3 = g_dev.ng3;
  t.g11 = g_dev.g[0][0];
  t.g12 = g_dev.g[0][1];
  t.g13 = g_dev.g[0][2];
  t.g21 = g_dev.g[1][0];
  t.g22 = g_dev.g[1][1];
  t.g23 = g_dev.g[1][2];
  t.g31 = g_dev.g[2][0];
  t.g32 = g_dev.g[2][1];
  t.g33 = g_dev.g[2][2];
  t.org1 = g_dev.org[0];
  t.org2 = g_dev.org[1];
  t.org3 = g_dev.org[2];
  /* enough blocks per item to fill the device when the items are few (an
     atom's whole sphere), one when they are many (pairs) */
  ychunks = (unsigned)((max_npts + kThreads - 1) / kThreads);
  {
    unsigned cap = (unsigned)(256 / (count < 256 ? count : 256));

    if (cap < 1) cap = 1;
    if (ychunks > cap) ychunks = cap;
  }
  /* a failure left behind by an earlier call is not this launch's */
  (void)hipGetLastError();
  if (double_float)
    hipLaunchKernelGGL(otf_eval_kernel<true>, dim3((unsigned)count, ychunks), dim3(kThreads), 0,
                       (hipStream_t)stream, t, count, atoms, npts, idx_off, out_off, sph_off, idx, gla, cla, out);
  else
    hipLaunchKernelGGL(otf_eval_kernel<false>, dim3((unsigned)count, ychunks), dim3(kThreads), 0,
                       (hipStream_t)stream, t, count, atoms, npts, idx_off, out_off, sph_off, idx, gla, cla, out);
  return hipGetLastError() == hipSuccess ? 0 : 2;
}

} // namespace

extern "C" void OTF_Orbitals_Release(void)
{
  if (g_dev.arena != nullptr) {
    (void)hipDeviceSynchronize();
    (void)hipFree(g_dev.arena);
  }
  g_dev = DeviceTables();
}

extern "C" int OTF_Orbitals_Ready(void)
{
  return g_dev.ready;
}

extern "C" int OTF_Orbitals_DoubleFloat(void)
{
  return g_dev.ready && g_dev.double_float && openmx_fp32_enabled();
}

extern "C" size_t OTF_Orbitals_Bytes(void)
{
  return g_dev.ready ? g_dev.bytes : 0;
}

extern "C" int OTF_Orbitals_Upload(const OTF_HostTables *t, int double_float)
{
  const size_t rv_cnt = (t->rv_count == 0) ? 1 : t->rv_count;
  const size_t rwf_cnt = (t->rwf_count == 0) ? 1 : t->rwf_count;
  const int species = t->species;
  std::vector<float> rwd, mcd, spl;
  size_t pos = 0, o_rv, o_rwf, o_rvo, o_rwb, o_spm, o_spx, o_spb, o_atv, o_rwd = 0, o_mcd = 0, o_spl = 0;
  unsigned char *arena = nullptr;

  double_float = double_float && openmx_fp32_enabled();
  OTF_Orbitals_Release();
  if (species <= 0) return -1;
  for (int w = 0; w < species; w++) {
    if (t->maxl[w] > OTF_L0MAX || t->mesh[w] < 6) return -1;
    for (int L0 = 0; L0 <= t->maxl[w]; L0++) {
      if (t->nb[w * (OTF_L0MAX + 1) + L0] > OTF_MULMAX) return -1;
    }
  }
  if (double_float) {
    /* every radial value as (hi, lo); per mesh point rv, h2 and the spline
       weights dum1..dum4 as (hi, lo) (the ends m = 1 and m = mesh - 1 take
       the host's corrected h1 / h3); per species the logarithmic mesh */
    try {
      rwd.resize(2 * rwf_cnt, 0.0f);
      mcd.resize(kMcdStride * rv_cnt, 0.0f);
      spl.resize(2 * (size_t)species, 0.0f);
    }
    catch (const std::bad_alloc &) {
      return 0;
    }
    for (size_t r = 0; r < t->rwf_count; r++) {
      const float hi = (float)t->rwf[r];
      rwd[2 * r] = hi;
      rwd[2 * r + 1] = (float)(t->rwf[r] - (double)hi);
    }
    for (int w = 0; w < species; w++) {
      const int mesh = t->mesh[w];
      const double *rvw = t->rv + t->rv_off[w];
      float *mc = mcd.data() + kMcdStride * t->rv_off[w];

      spl[2 * w] = (float)log(rvw[0]);
      spl[2 * w + 1] = (float)((log(rvw[mesh - 1]) - log(rvw[0])) / (double)(mesh - 1));
      for (int m = 0; m < mesh; m++, mc += kMcdStride) {
        double v[6], h1, h2, h3, dum;

        v[0] = rvw[m];
        if (m < 1) {
          mc[0] = (float)v[0];
          mc[1] = (float)(v[0] - (double)mc[0]);
          continue;
        }
        h2 = rvw[m] - rvw[m - 1];
        h1 = (2 <= m) ? rvw[m - 1] - rvw[m - 2] : 0.0;
        h3 = (m <= mesh - 2) ? rvw[m + 1] - rvw[m] : 0.0;
        if (m == 1) h1 = rvw[m - 1] - rvw[m + 1];
        if (m == mesh - 1) h3 = rvw[m - 2] - rvw[m];
        v[1] = h2;
        dum = h1 + h2;
        v[2] = h1 / h2 / dum;
        v[3] = h2 / h1 / dum;
        dum = h2 + h3;
        v[4] = h2 / h3 / dum;
        v[5] = h3 / h2 / dum;
        for (int q = 0; q < 6; q++) {
          const float hi = (float)v[q];
          mc[2 * q] = hi;
          mc[2 * q + 1] = (float)(v[q] - (double)hi);
        }
      }
    }
  }

  o_rv = arena_off(&pos, sizeof(double) * rv_cnt);
  o_rwf = arena_off(&pos, sizeof(double) * rwf_cnt);
  o_rvo = arena_off(&pos, sizeof(size_t) * (size_t)(species + 1));
  o_rwb = arena_off(&pos, sizeof(size_t) * (size_t)(species + 1));
  o_spm = arena_off(&pos, sizeof(int) * (size_t)species);
  o_spx = arena_off(&pos, sizeof(int) * (size_t)species);
  o_spb = arena_off(&pos, sizeof(int) * (size_t)species * (OTF_L0MAX + 1));
  o_atv = arena_off(&pos, sizeof(double) * t->atv_rows * 3);
  if (double_float) {
    o_rwd = arena_off(&pos, sizeof(float) * rwd.size());
    o_mcd = arena_off(&pos, sizeof(float) * mcd.size());
    o_spl = arena_off(&pos, sizeof(float) * spl.size());
  }
  if (hipMalloc(&arena, pos) != hipSuccess) {
    (void)hipGetLastError();
    return 0;
  }
#define OTF_COPY(off, src, bytes)                                                                        \
  do {                                                                                                   \
    if (hipMemcpy(arena + (off), (src), (bytes), hipMemcpyHostToDevice) != hipSuccess) {                \
      (void)hipFree(arena);                                                                              \
      (void)hipGetLastError();                                                                           \
      return 0;                                                                                          \
    }                                                                                                    \
  } while (0)
  OTF_COPY(o_rv, t->rv, sizeof(double) * t->rv_count);
  OTF_COPY(o_rwf, t->rwf, sizeof(double) * t->rwf_count);
  OTF_COPY(o_rvo, t->rv_off, sizeof(size_t) * (size_t)(species + 1));
  OTF_COPY(o_rwb, t->rwf_base, sizeof(size_t) * (size_t)(species + 1));
  OTF_COPY(o_spm, t->mesh, sizeof(int) * (size_t)species);
  OTF_COPY(o_spx, t->maxl, sizeof(int) * (size_t)species);
  OTF_COPY(o_spb, t->nb, sizeof(int) * (size_t)species * (OTF_L0MAX + 1));
  OTF_COPY(o_atv, t->atv, sizeof(double) * t->atv_rows * 3);
  if (double_float) {
    OTF_COPY(o_rwd, rwd.data(), sizeof(float) * rwd.size());
    OTF_COPY(o_mcd, mcd.data(), sizeof(float) * mcd.size());
    OTF_COPY(o_spl, spl.data(), sizeof(float) * spl.size());
  }
#undef OTF_COPY

  g_dev.arena = arena;
  g_dev.bytes = pos;
  g_dev.double_float = double_float ? 1 : 0;
  g_dev.rv = reinterpret_cast<const double *>(arena + o_rv);
  g_dev.rwf = reinterpret_cast<const double *>(arena + o_rwf);
  g_dev.rvo = reinterpret_cast<const size_t *>(arena + o_rvo);
  g_dev.rwb = reinterpret_cast<const size_t *>(arena + o_rwb);
  g_dev.spm = reinterpret_cast<const int *>(arena + o_spm);
  g_dev.spx = reinterpret_cast<const int *>(arena + o_spx);
  g_dev.spb = reinterpret_cast<const int *>(arena + o_spb);
  g_dev.atv = reinterpret_cast<const double *>(arena + o_atv);
  if (double_float) {
    g_dev.rwd = reinterpret_cast<const float *>(arena + o_rwd);
    g_dev.mcd = reinterpret_cast<const float *>(arena + o_mcd);
    g_dev.spl = reinterpret_cast<const float *>(arena + o_spl);
  }
  g_dev.ng23 = t->ng23;
  g_dev.ng3 = t->ng3;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) g_dev.g[i][j] = t->g[i][j];
    g_dev.org[i] = t->org[i];
  }
  g_dev.ready = 1;
  return 1;
}

extern "C" int OTF_Orbitals_Eval(void *stream, int count, int max_npts, const OTF_Atom *atoms, const int *npts,
                                 const size_t *idx_off, const size_t *out_off, const size_t *sph_off, const int *idx,
                                 const int *gla, const int *cla, float *out)
{
  return otf_launch(g_dev.double_float != 0 && openmx_fp32_enabled(), stream, count, max_npts, atoms, npts, idx_off, out_off, sph_off, idx,
                    gla, cla, out);
}

extern "C" int OTF_Orbitals_Eval_FP64(void *stream, int count, int max_npts, const OTF_Atom *atoms, const int *npts,
                                      const size_t *idx_off, const size_t *out_off, const size_t *sph_off,
                                      const int *idx, const int *gla, const int *cla, float *out)
{
  return otf_launch(false, stream, count, max_npts, atoms, npts, idx_off, out_off, sph_off, idx, gla, cla, out);
}

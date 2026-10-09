/**********************************************************************
  Set_Orbitals_Grid.c:

   Set_Orbitals_Grid.c is a subroutine to calculate the value of basis
   functions on each grid point.

  Log of Set_Orbitals_Grid.c:

     22/Nov/2001  Released by T.Ozaki

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include "openmx_common.h"
#include "mpi.h"
#include <omp.h>
#include <openacc.h>
#include "set_cuda_default_device_from_local_rank.h"
#include "orbs_grid_gpu.h"

/***********************************************************************
   GPU evaluation of the primitive (Cnt_kind==0) basis orbitals.

   The per-point math is a verbatim transcription of the inlined
   Get_Orbitals in the host loops below; the device covers uncontracted
   bases with L0 <= SOG_L0MAX (the explicit real-harmonic formulas).
   Anything else (orbital optimization, L0 >= 4 via ComplexSH) keeps the
   untouched host path.  All transient device memory is claimed through
   one acc_malloc arena so that an out-of-memory device never aborts the
   run: acc_malloc reports NULL and the caller falls back to the host.
   Disable with OPENMX_ORBS_GRID_GPU=0.
***********************************************************************/

#define SOG_L0MAX  3
#define SOG_MULMAX 8

/* SOG_GpuPair: orbs_grid_gpu.h */

static int SOG_env_flag(const char *name, int default_value)
{
  const char *value = getenv(name);

  if (value == NULL || value[0] == '\0') return default_value;
  return atoi(value) != 0;
}

static int SOG_gpu_eligible(int Cnt_kind)
{
  static int device_checked = 0, device_ok = 0;
  int w, L0;

  if (Cnt_kind != 0) return 0;
  if (!SOG_env_flag("OPENMX_ORBS_GRID_GPU", 1)) return 0;

  /* the staging buffers and the device stores are float */
  if (sizeof(Type_Orbs_Grid) != sizeof(float)) return 0;

  for (w = 0; w < SpeciesNum; w++) {
    if (SOG_L0MAX < Spe_MaxL_Basis[w]) return 0;
    for (L0 = 0; L0 <= Spe_MaxL_Basis[w]; L0++) {
      if (SOG_MULMAX < Spe_Num_Basis[w][L0]) return 0;
    }
    if (Spe_Num_Mesh_PAO[w] < 6) return 0;
  }

  if (!device_checked) {
    device_checked = 1;
    /* the probe, not a bare device count: a rank whose context or module
       load cannot complete would abort inside acc_malloc otherwise */
    device_ok = gpu_rank_device_usable();
  }
  return device_ok;
}

static size_t SOG_arena_off(size_t *pos, size_t bytes)
{
  size_t off = *pos;

  *pos = (off + bytes + 511U) & ~(size_t)511U;
  return off;
}

/* one shot; the caller has a host fallback */
static void *SOG_arena_try(size_t bytes)
{
  void *arena = acc_malloc(bytes);

  if (arena == NULL) {
    /* our own freelist may hoard mismatched freed blocks */
    if (cudaDeviceSynchronize() == cudaSuccess) acc_clear_freelists();
    arena = acc_malloc(bytes);
  }
  return arena;
}

/* The orbitals of species wan at the grid point (GNc, cell GRc) about the
   atom at (gx, gy, gz) shifted by (ax, ay, az): a verbatim transcription
   of the inlined Get_Orbitals of the host loops below (keep the two in
   sync).  Shared by the table construction and the on-the-fly tiles of
   the grid integrals (SOG_Device_* below). */
#pragma acc routine seq
static void SOG_point_eval(int wan, int no, double gx, double gy, double gz,
                           double ax, double ay, double az, int GNc, int GRc,
                           const double *atvf, const double *rv_all, const double *rwf_all,
                           const size_t *rv_off, const size_t *rwf_base,
                           const int *sp_mesh, const int *sp_maxl, const int *sp_nb,
                           int ng23, int ng3,
                           double g11, double g12, double g13,
                           double g21, double g22, double g23,
                           double g31, double g32, double g33,
                           double org1, double org2, double org3,
                           float *out)
{
    /* Get_Grid_XYZ */
    const int n1 = GNc / ng23;
    const int n2 = (GNc - n1 * ng23) / ng3;
    const int n3 = GNc - n1 * ng23 - n2 * ng3;

    const double x = ((double)n1 * g11 + (double)n2 * g21 + (double)n3 * g31 + org1)
                     + atvf[3 * (size_t)GRc + 0] - gx - ax;
    const double y = ((double)n1 * g12 + (double)n2 * g22 + (double)n3 * g32 + org2)
                     + atvf[3 * (size_t)GRc + 1] - gy - ay;
    const double z = ((double)n1 * g13 + (double)n2 * g23 + (double)n3 * g33 + org3)
                     + atvf[3 * (size_t)GRc + 2] - gz - az;

    const int mesh = sp_mesh[wan];
    const int maxl = sp_maxl[wan];
    const double *rv = rv_all + rv_off[wan];

    double RF[SOG_L0MAX + 1][SOG_MULMAX];
    double AF[SOG_L0MAX + 1][2 * SOG_L0MAX + 1];
    double R, Q, P;
    int po = 0;
    int L0, Mul0, M0, i1;

    /* xyz2spherical, without the angles: the real harmonics below need
       only cos/sin of theta and phi, which are z/r, r1/r, x/r1, y/r1 (the
       host path goes through acos/asin and sin/cos, which agrees with this
       to the last bits of the double value) */
    double siQ, coQ, siP, coP;
    {
      const double Min_r = 10e-15;
      double dum = x * x + y * y;
      double r = sqrt(dum + z * z);
      double r1 = sqrt(dum);

      if (Min_r <= r) {
        if (r < fabs(z)) { coQ = (z < 0.0 ? -1.0 : 1.0); siQ = 0.0; }
        else { coQ = z / r; siQ = r1 / r; }
        if (Min_r <= r1) {
          if (r1 < fabs(y)) { siP = (y < 0.0 ? -1.0 : 1.0); coP = 0.0; }
          else { siP = y / r1; coP = x / r1; }
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
      Q = 0.0;
      P = 0.0;
    }

    /* radial spline of every (L0, Mul0) */
    if (rv[mesh - 1] < R) {
      /* outside the PAO mesh: every RF is zero, so every orbital is zero */
      po = 1;
    }
    else {

      const int below = (R < rv[0]);
      double h1, h2, h3, x1, x2, y1, y2, y12, y22;
      double dum, dum1, dum2, dum3, dum4;
      double rm = 0.0;
      size_t rrow = rwf_base[wan];
      int m;

      if (below) {
        m = 4;
        rm = rv[m];

        h1 = rv[m - 1] - rv[m - 2];
        h2 = rv[m] - rv[m - 1];
        h3 = rv[m + 1] - rv[m];

        x1 = rm - rv[m - 1];
        x2 = rm - rv[m];
      }
      else {
        /* the first mesh index with rv[m] >= R (what the bisection of the
           host path finds): an estimate from the logarithmic mesh, then
           the exact neighbours */
        const double x0 = log(rv[0]);
        const double dx = (log(rv[mesh - 1]) - x0) / (double)(mesh - 1);
        m = (int)((log(R) - x0) / dx) + 1;
        if (m < 1) m = 1;
        if (mesh - 1 < m) m = mesh - 1;
        while (1 < m && R <= rv[m - 1]) m--;
        while (m < mesh - 1 && rv[m] < R) m++;

        h1 = rv[m - 1] - rv[m - 2];
        h2 = rv[m] - rv[m - 1];
        h3 = rv[m + 1] - rv[m];

        x1 = R - rv[m - 1];
        x2 = R - rv[m];
      }

      y1 = x1 / h2;
      y2 = x2 / h2;
      y12 = y1 * y1;
      y22 = y2 * y2;

      dum = h1 + h2;
      dum1 = h1 / h2 / dum;
      dum2 = h2 / h1 / dum;
      dum = h2 + h3;
      dum3 = h2 / h3 / dum;
      dum4 = h3 / h2 / dum;

      for (L0 = 0; L0 <= maxl; L0++) {
        const int nb = sp_nb[wan * (SOG_L0MAX + 1) + L0];

        for (Mul0 = 0; Mul0 < nb; Mul0++) {

          const double *rw = rwf_all + rrow;
          double f1 = rw[m - 2];
          double f2 = rw[m - 1];
          double f3 = rw[m];
          double f4 = rw[m + 1];
          double g1, g2, f;

          rrow += (size_t)mesh;

          if (m == 1) {
            h1 = -(h2 + h3);
            f1 = f4;
          }
          else if (m == (mesh - 1)) {
            h3 = -(h1 + h2);
            f4 = f1;
          }

          dum = f3 - f2;
          g1 = dum * dum1 + (f2 - f1) * dum2;
          g2 = (f4 - f3) * dum3 + dum * dum4;

          f = y22 * (3.0 * f2 + h2 * g1 + (2.0 * f2 + h2 * g1) * y2)
            + y12 * (3.0 * f3 - h2 * g2 - (2.0 * f3 - h2 * g2) * y1);

          if (below) {

            double df = 2.0 * y2 / h2 * (3.0 * f2 + h2 * g1 + (2.0 * f2 + h2 * g1) * y2)
              + y22 * (2.0 * f2 + h2 * g1) / h2
              + 2.0 * y1 / h2 * (3.0 * f3 - h2 * g2 - (2.0 * f3 - h2 * g2) * y1)
              - y12 * (2.0 * f3 - h2 * g2) / h2;
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

            RF[L0][Mul0] = a * R * R * R + b * R * R + c * R + d;
          }
          else {
            RF[L0][Mul0] = f;
          }
        }
      }
    }

    if (po == 0) {

      /* Angular */
      /* siQ, coQ, siP, coP: from the coordinates above */
      double dum, dum1, dum2;

      for (L0 = 0; L0 <= maxl; L0++) {

        if (L0 == 0) {
          AF[0][0] = 0.282094791773878;
        }
        else if (L0 == 1) {
          dum = 0.48860251190292 * siQ;
          AF[1][0] = dum * coP;
          AF[1][1] = dum * siP;
          AF[1][2] = 0.48860251190292 * coQ;
        }
        else if (L0 == 2) {
          dum1 = siQ * siQ;
          dum2 = 1.09254843059208 * siQ * coQ;
          AF[2][0] = 0.94617469575756 * coQ * coQ - 0.31539156525252;
          AF[2][1] = 0.54627421529604 * dum1 * (1.0 - 2.0 * siP * siP);
          AF[2][2] = 1.09254843059208 * dum1 * siP * coP;
          AF[2][3] = dum2 * coP;
          AF[2][4] = dum2 * siP;
        }
        else if (L0 == 3) {
          AF[3][0] = 0.373176332590116 * (5.0 * coQ * coQ * coQ - 3.0 * coQ);
          AF[3][1] = 0.457045799464466 * coP * siQ * (5.0 * coQ * coQ - 1.0);
          AF[3][2] = 0.457045799464466 * siP * siQ * (5.0 * coQ * coQ - 1.0);
          AF[3][3] = 1.44530572132028 * siQ * siQ * coQ * (coP * coP - siP * siP);
          AF[3][4] = 2.89061144264055 * siQ * siQ * coQ * siP * coP;
          AF[3][5] = 0.590043589926644 * siQ * siQ * siQ * (4.0 * coP * coP * coP - 3.0 * coP);
          AF[3][6] = 0.590043589926644 * siQ * siQ * siQ * (3.0 * siP - 4.0 * siP * siP * siP);
        }
      }

      /* Chi */
      i1 = -1;
      for (L0 = 0; L0 <= maxl; L0++) {
        const int nb = sp_nb[wan * (SOG_L0MAX + 1) + L0];

        for (Mul0 = 0; Mul0 < nb; Mul0++) {
          for (M0 = 0; M0 <= 2 * L0; M0++) {
            i1++;
            out[i1] = (float)(RF[L0][Mul0] * AF[L0][M0]);
          }
        }
      }
    }
    else {
      for (i1 = 0; i1 < no; i1++) out[i1] = 0.0f;
    }
}

/* ---------------------------------------------------------------------
   Double-float evaluation of the on-the-fly tiles (the default; the
   FP64 transcription above stays for the tables and for
   OPENMX_ORBS_EVAL_PRECISION=fp64).  A GeForce runs FP64 at 1/64 of its
   FP32 rate and the FP64 routine is bound by that pipe (RTX 5080, 18
   ranks: 0.75 s per SCF step for the Hamiltonian tiles alone).  Here
   everything after the FP64 grid coordinates is a pair of floats
   (hi + lo, |lo| <= ulp(hi)/2, about 2^-48 relative): the PAO radial
   tables and the per-mesh spline constants are held as such pairs, the
   mesh index comes from an FP32 logarithm (the index search makes it
   exact), and the float written for an orbital is the hi part of its
   double-float product, i.e. the float nearest the FP64 value unless
   that value lies within ~1e-14 of a rounding boundary (about one
   value in 10^6 differs by one ulp from the FP64 routine).  Points
   inside the first mesh point (the cubic extrapolation) take the FP64
   routine.  The algebra follows SOG_point_eval; keep the two in sync.
   --------------------------------------------------------------------- */
typedef struct { float hi, lo; } SOG_df;

/* per mesh point of a species: rv, h2 = rv[m] - rv[m-1] and the four
   derivative weights of the spline (dum1..dum4), each as (hi, lo) */
#define SOG_MCD_STRIDE 12

#pragma acc routine seq
static inline SOG_df SOG_df_mk(float hi, float lo)
{
  SOG_df r; r.hi = hi; r.lo = lo; return r;
}

/* |a| >= |b| (or a == 0) */
#pragma acc routine seq
static inline SOG_df SOG_df_qts(float a, float b)
{
  SOG_df r; r.hi = a + b; r.lo = b - (r.hi - a); return r;
}

#pragma acc routine seq
static inline SOG_df SOG_df_dbl(double x)
{
  const float hi = (float)x;
  return SOG_df_mk(hi, (float)(x - (double)hi));
}

#pragma acc routine seq
static inline SOG_df SOG_df_add(SOG_df a, SOG_df b)
{
  const float s = a.hi + b.hi;
  const float v = s - a.hi;
  const float e = ((a.hi - (s - v)) + (b.hi - v)) + (a.lo + b.lo);
  return SOG_df_qts(s, e);
}

#pragma acc routine seq
static inline SOG_df SOG_df_sub(SOG_df a, SOG_df b)
{
  return SOG_df_add(a, SOG_df_mk(-b.hi, -b.lo));
}

#pragma acc routine seq
static inline SOG_df SOG_df_mul(SOG_df a, SOG_df b)
{
  const float p = a.hi * b.hi;
  const float e = fmaf(a.hi, b.hi, -p) + (a.hi * b.lo + a.lo * b.hi);
  return SOG_df_qts(p, e);
}

/* b an exact float (a small integer) */
#pragma acc routine seq
static inline SOG_df SOG_df_mulf(SOG_df a, float b)
{
  const float p = a.hi * b;
  const float e = fmaf(a.hi, b, -p) + a.lo * b;
  return SOG_df_qts(p, e);
}

#pragma acc routine seq
static inline SOG_df SOG_df_div(SOG_df a, SOG_df b)
{
  const float q1 = a.hi / b.hi;
  const float p = q1 * b.hi;
  const SOG_df r = SOG_df_sub(a, SOG_df_qts(p, fmaf(q1, b.hi, -p) + q1 * b.lo));
  return SOG_df_qts(q1, r.hi / b.hi);
}

#pragma acc routine seq
static inline SOG_df SOG_df_sqrt(SOG_df a)
{
  float s1, p;
  SOG_df r;

  if (a.hi <= 0.0f) return SOG_df_mk(0.0f, 0.0f);
  s1 = sqrtf(a.hi);
  p = s1 * s1;
  r = SOG_df_sub(a, SOG_df_mk(p, fmaf(s1, s1, -p)));
  return SOG_df_qts(s1, r.hi / (2.0f * s1));
}

/* the cubic spline of SOG_point_eval for one radial function (rw2: its
   (hi, lo) table) at the mesh interval m with the interval's constants */
#pragma acc routine seq
static inline SOG_df SOG_radial_df(const float *rw2, int m, int mesh, SOG_df h2,
                                   SOG_df dum1, SOG_df dum2, SOG_df dum3, SOG_df dum4,
                                   SOG_df y1, SOG_df y2, SOG_df y12, SOG_df y22)
{
  /* Correct the boundary indices before loading: replacing f1/f4 after
     the load still reads outside the radial table at the end intervals. */
  const int i1 = m == 1 ? m + 1 : m - 2;
  const int i4 = m == mesh - 1 ? m - 2 : m + 1;
  const SOG_df f1 = SOG_df_mk(rw2[2 * i1], rw2[2 * i1 + 1]);
  const SOG_df f2 = SOG_df_mk(rw2[2 * (m - 1)], rw2[2 * (m - 1) + 1]);
  const SOG_df f3 = SOG_df_mk(rw2[2 * m], rw2[2 * m + 1]);
  const SOG_df f4 = SOG_df_mk(rw2[2 * i4], rw2[2 * i4 + 1]);
  SOG_df d32, g1, g2, P, Q, t, u;
  d32 = SOG_df_sub(f3, f2);
  g1 = SOG_df_add(SOG_df_mul(d32, dum1), SOG_df_mul(SOG_df_sub(f2, f1), dum2));
  g2 = SOG_df_add(SOG_df_mul(SOG_df_sub(f4, f3), dum3), SOG_df_mul(d32, dum4));
  P = SOG_df_mul(h2, g1);
  Q = SOG_df_mul(h2, g2);
  /* y2^2 (3 f2 + P + (2 f2 + P) y2) + y1^2 (3 f3 - Q - (2 f3 - Q) y1) */
  t = SOG_df_add(SOG_df_mulf(f2, 2.0f), P);
  t = SOG_df_add(SOG_df_add(t, f2), SOG_df_mul(t, y2));
  u = SOG_df_sub(SOG_df_mulf(f3, 2.0f), Q);
  u = SOG_df_sub(SOG_df_add(u, f3), SOG_df_mul(u, y1));
  return SOG_df_add(SOG_df_mul(y22, t), SOG_df_mul(y12, u));
}

#pragma acc routine seq
static void SOG_point_eval_df(int wan, int no, double gx, double gy, double gz,
                              double ax, double ay, double az, int GNc, int GRc,
                              const double *atvf, const double *rv_all, const double *rwf_all,
                              const float *rwd_all, const float *mcd_all, const float *spl,
                              const size_t *rv_off, const size_t *rwf_base,
                              const int *sp_mesh, const int *sp_maxl, const int *sp_nb,
                              int ng23, int ng3,
                              double g11, double g12, double g13,
                              double g21, double g22, double g23,
                              double g31, double g32, double g33,
                              double org1, double org2, double org3,
                              float *out)
{
    /* Get_Grid_XYZ in FP64: the small difference of large coordinates */
    const int n1 = GNc / ng23;
    const int n2 = (GNc - n1 * ng23) / ng3;
    const int n3 = GNc - n1 * ng23 - n2 * ng3;

    const double x = ((double)n1 * g11 + (double)n2 * g21 + (double)n3 * g31 + org1)
                     + atvf[3 * (size_t)GRc + 0] - gx - ax;
    const double y = ((double)n1 * g12 + (double)n2 * g22 + (double)n3 * g32 + org2)
                     + atvf[3 * (size_t)GRc + 1] - gy - ay;
    const double z = ((double)n1 * g13 + (double)n2 * g23 + (double)n3 * g33 + org3)
                     + atvf[3 * (size_t)GRc + 2] - gz - az;

    const int mesh = sp_mesh[wan];
    const int maxl = sp_maxl[wan];
    const double *rv = rv_all + rv_off[wan];
    const float *mcd = mcd_all + SOG_MCD_STRIDE * rv_off[wan];
    const SOG_df one = SOG_df_mk(1.0f, 0.0f), zero = SOG_df_mk(0.0f, 0.0f);
    const SOG_df xd = SOG_df_dbl(x), yd = SOG_df_dbl(y), zd = SOG_df_dbl(z);
    const SOG_df dum = SOG_df_add(SOG_df_mul(xd, xd), SOG_df_mul(yd, yd));
    const SOG_df r = SOG_df_sqrt(SOG_df_add(dum, SOG_df_mul(zd, zd)));
    const SOG_df r1 = SOG_df_sqrt(dum);
    const double R = (double)r.hi + (double)r.lo;
    const double R1 = (double)r1.hi + (double)r1.lo;
    SOG_df siQ, coQ, siP, coP;
    SOG_df h2, dum1, dum2, dum3, dum4, y1, y2, y12, y22;
    size_t rrow;
    int m, Mul0, i1;

    if (rv[mesh - 1] < R) {
      /* outside the PAO mesh: every orbital is zero */
      for (i1 = 0; i1 < no; i1++) out[i1] = 0.0f;
      return;
    }
    if (R < rv[0]) {
      /* inside the first mesh point: the cubic extrapolation, in FP64 */
      SOG_point_eval(wan, no, gx, gy, gz, ax, ay, az, GNc, GRc, atvf, rv_all, rwf_all, rv_off, rwf_base,
                     sp_mesh, sp_maxl, sp_nb, ng23, ng3, g11, g12, g13, g21, g22, g23, g31, g32, g33,
                     org1, org2, org3, out);
      return;
    }

    /* xyz2spherical without the angles (see SOG_point_eval) */
    {
      const double Min_r = 10e-15;

      if (Min_r <= R) {
        if (R < fabs(z)) { coQ = SOG_df_mk(z < 0.0 ? -1.0f : 1.0f, 0.0f); siQ = zero; }
        else {
          const SOG_df ir = SOG_df_div(one, r);
          coQ = SOG_df_mul(zd, ir);
          siQ = SOG_df_mul(r1, ir);
        }
        if (Min_r <= R1) {
          if (R1 < fabs(y)) { siP = SOG_df_mk(y < 0.0 ? -1.0f : 1.0f, 0.0f); coP = zero; }
          else {
            const SOG_df ir1 = SOG_df_div(one, r1);
            siP = SOG_df_mul(yd, ir1);
            coP = SOG_df_mul(xd, ir1);
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

    /* the first mesh index with rv[m] >= R: an FP32 estimate from the
       logarithmic mesh, then the exact neighbours (FP64 compares) */
    {
      const float x0 = spl[2 * wan], dx = spl[2 * wan + 1];

      m = (int)((logf(r.hi) - x0) / dx) + 1;
      if (m < 1) m = 1;
      if (mesh - 1 < m) m = mesh - 1;
      while (1 < m && R <= rv[m - 1]) m--;
      while (m < mesh - 1 && rv[m] < R) m++;
    }
    {
      const float *mc = mcd + SOG_MCD_STRIDE * m;
      const SOG_df rvm1 = SOG_df_mk(mc[-SOG_MCD_STRIDE], mc[-SOG_MCD_STRIDE + 1]);

      h2 = SOG_df_mk(mc[2], mc[3]);
      dum1 = SOG_df_mk(mc[4], mc[5]);
      dum2 = SOG_df_mk(mc[6], mc[7]);
      dum3 = SOG_df_mk(mc[8], mc[9]);
      dum4 = SOG_df_mk(mc[10], mc[11]);
      y1 = SOG_df_div(SOG_df_sub(r, rvm1), h2);
      y2 = SOG_df_sub(y1, one);
      y12 = SOG_df_mul(y1, y1);
      y22 = SOG_df_mul(y2, y2);
    }

    /* Chi = RF * AF per (L0, Mul0, M0), in the order of SOG_point_eval */
    rrow = rwf_base[wan];
    i1 = 0;
    {
      const int nb = sp_nb[wan * (SOG_L0MAX + 1) + 0];
      const SOG_df c00 = SOG_df_mk(0.282094806432724f, -1.4658845692849809e-08f);

      for (Mul0 = 0; Mul0 < nb; Mul0++) {
        const SOG_df RF = SOG_radial_df(rwd_all + 2 * rrow, m, mesh, h2, dum1, dum2, dum3, dum4, y1, y2, y12, y22);
        rrow += (size_t)mesh;
        out[i1++] = SOG_df_mul(RF, c00).hi;
      }
    }
    if (1 <= maxl) {
      const int nb = sp_nb[wan * (SOG_L0MAX + 1) + 1];
      const SOG_df c1 = SOG_df_mk(0.48860251903533936f, -7.1324195438648985e-09f);
      const SOG_df d = SOG_df_mul(c1, siQ);
      const SOG_df A0 = SOG_df_mul(d, coP);
      const SOG_df A1 = SOG_df_mul(d, siP);
      const SOG_df A2 = SOG_df_mul(c1, coQ);

      for (Mul0 = 0; Mul0 < nb; Mul0++) {
        const SOG_df RF = SOG_radial_df(rwd_all + 2 * rrow, m, mesh, h2, dum1, dum2, dum3, dum4, y1, y2, y12, y22);
        rrow += (size_t)mesh;
        out[i1 + 0] = SOG_df_mul(RF, A0).hi;
        out[i1 + 1] = SOG_df_mul(RF, A1).hi;
        out[i1 + 2] = SOG_df_mul(RF, A2).hi;
        i1 += 3;
      }
    }
    if (2 <= maxl) {
      const int nb = sp_nb[wan * (SOG_L0MAX + 1) + 2];
      const SOG_df c20 = SOG_df_mk(0.946174681186676f, 1.4570884054876387e-08f);
      const SOG_df c21 = SOG_df_mk(0.31539157032966614f, -5.077146258969378e-09f);
      const SOG_df c22 = SOG_df_mk(0.5462742447853088f, -2.948926791646045e-08f);
      const SOG_df c23 = SOG_df_mk(1.0925484895706177f, -5.89785358329209e-08f);
      const SOG_df s2 = SOG_df_mul(siQ, siQ);
      const SOG_df d2 = SOG_df_mul(c23, SOG_df_mul(siQ, coQ));
      const SOG_df A0 = SOG_df_sub(SOG_df_mul(c20, SOG_df_mul(coQ, coQ)), c21);
      const SOG_df A1 = SOG_df_mul(SOG_df_mul(c22, s2), SOG_df_sub(one, SOG_df_mulf(SOG_df_mul(siP, siP), 2.0f)));
      const SOG_df A2 = SOG_df_mul(SOG_df_mul(c23, s2), SOG_df_mul(siP, coP));
      const SOG_df A3 = SOG_df_mul(d2, coP);
      const SOG_df A4 = SOG_df_mul(d2, siP);

      for (Mul0 = 0; Mul0 < nb; Mul0++) {
        const SOG_df RF = SOG_radial_df(rwd_all + 2 * rrow, m, mesh, h2, dum1, dum2, dum3, dum4, y1, y2, y12, y22);
        rrow += (size_t)mesh;
        out[i1 + 0] = SOG_df_mul(RF, A0).hi;
        out[i1 + 1] = SOG_df_mul(RF, A1).hi;
        out[i1 + 2] = SOG_df_mul(RF, A2).hi;
        out[i1 + 3] = SOG_df_mul(RF, A3).hi;
        out[i1 + 4] = SOG_df_mul(RF, A4).hi;
        i1 += 5;
      }
    }
    if (3 <= maxl) {
      const int nb = sp_nb[wan * (SOG_L0MAX + 1) + 3];
      const SOG_df c30 = SOG_df_mk(0.37317633628845215f, -3.6983360818254596e-09f);
      const SOG_df c31 = SOG_df_mk(0.4570457935333252f, 5.931140911741295e-09f);
      const SOG_df c33 = SOG_df_mk(1.4453057050704956f, 1.6249783740818202e-08f);
      const SOG_df c34 = SOG_df_mk(2.890611410140991f, 3.249955682349537e-08f);
      const SOG_df c35 = SOG_df_mk(0.5900436043739319f, -1.4447287810526177e-08f);
      const SOG_df cQ2 = SOG_df_mul(coQ, coQ);
      const SOG_df sQ2 = SOG_df_mul(siQ, siQ);
      const SOG_df sQ3 = SOG_df_mul(sQ2, siQ);
      const SOG_df cP2 = SOG_df_mul(coP, coP);
      const SOG_df sP2 = SOG_df_mul(siP, siP);
      const SOG_df five_cQ2_m1 = SOG_df_sub(SOG_df_mulf(cQ2, 5.0f), one);
      const SOG_df A0 = SOG_df_mul(c30, SOG_df_mul(coQ, SOG_df_sub(SOG_df_mulf(cQ2, 5.0f), SOG_df_mk(3.0f, 0.0f))));
      const SOG_df A1 = SOG_df_mul(c31, SOG_df_mul(SOG_df_mul(coP, siQ), five_cQ2_m1));
      const SOG_df A2 = SOG_df_mul(c31, SOG_df_mul(SOG_df_mul(siP, siQ), five_cQ2_m1));
      const SOG_df A3 = SOG_df_mul(c33, SOG_df_mul(SOG_df_mul(sQ2, coQ), SOG_df_sub(cP2, sP2)));
      const SOG_df A4 = SOG_df_mul(c34, SOG_df_mul(SOG_df_mul(sQ2, coQ), SOG_df_mul(siP, coP)));
      const SOG_df A5 = SOG_df_mul(c35, SOG_df_mul(sQ3, SOG_df_mul(coP, SOG_df_sub(SOG_df_mulf(cP2, 4.0f), SOG_df_mk(3.0f, 0.0f)))));
      const SOG_df A6 = SOG_df_mul(c35, SOG_df_mul(sQ3, SOG_df_mul(siP, SOG_df_sub(SOG_df_mk(3.0f, 0.0f), SOG_df_mulf(sP2, 4.0f)))));

      for (Mul0 = 0; Mul0 < nb; Mul0++) {
        const SOG_df RF = SOG_radial_df(rwd_all + 2 * rrow, m, mesh, h2, dum1, dum2, dum3, dum4, y1, y2, y12, y22);
        rrow += (size_t)mesh;
        out[i1 + 0] = SOG_df_mul(RF, A0).hi;
        out[i1 + 1] = SOG_df_mul(RF, A1).hi;
        out[i1 + 2] = SOG_df_mul(RF, A2).hi;
        out[i1 + 3] = SOG_df_mul(RF, A3).hi;
        out[i1 + 4] = SOG_df_mul(RF, A4).hi;
        out[i1 + 5] = SOG_df_mul(RF, A5).hi;
        out[i1 + 6] = SOG_df_mul(RF, A6).hi;
        i1 += 7;
      }
    }
}

/* One launch evaluates every grid point of one atom (part 1:
   single_pair && identity_idx, pairs[0] describes the atom itself) or
   every overlap point of all remote neighbours of one atom (part 2). */
static void SOG_gpu_eval(int npts, int identity_idx, int single_pair,
                         const int *nog, const int *ppr, const SOG_GpuPair *pairs,
                         const int *gla, const int *cla, const double *atvf,
                         const double *rv_all, const double *rwf_all,
                         const size_t *rv_off, const size_t *rwf_base,
                         const int *sp_mesh, const int *sp_maxl, const int *sp_nb,
                         int ng23, int ng3,
                         double g11, double g12, double g13,
                         double g21, double g22, double g23,
                         double g31, double g32, double g33,
                         double org1, double org2, double org3,
                         float *chi)
{
  int ip;
#pragma acc parallel loop gang vector vector_length(128) \
  deviceptr(nog, ppr, pairs, gla, cla, atvf, rv_all, rwf_all, rv_off, rwf_base, \
            sp_mesh, sp_maxl, sp_nb, chi)
  for (ip = 0; ip < npts; ip++) {
    const int pair = (single_pair ? 0 : ppr[ip]);
    const SOG_GpuPair pr = pairs[pair];
    const int Nc = (identity_idx ? ip : nog[ip]);
    float *out = chi + pr.out + (size_t)(ip - pr.pt0) * (size_t)pr.no;
    SOG_point_eval(pr.wan, pr.no, pr.gx, pr.gy, pr.gz, pr.ax, pr.ay, pr.az, gla[Nc], cla[Nc],
                   atvf, rv_all, rwf_all, rv_off, rwf_base, sp_mesh, sp_maxl, sp_nb,
                   ng23, ng3, g11, g12, g13, g21, g22, g23, g31, g32, g33, org1, org2, org3, out);
  }
}
/* returns 1 when the whole update ran on the device; 0 keeps the host path */
static int Set_Orbitals_Grid_GPU(void)
{
  static int fail_notice_done = 0, active_notice_done = 0;
  int myid;
  int Mc_AN, Gc_AN, Cwan, h_AN, Gh_AN, Hwan, Rnh, NO0, NO1;
  int w, L0, Mul0, i, npair_cap;
  size_t max_pts = 0, max_chi = 0, max_fpts = 0, max_fchi = 0;
  size_t rv_cnt = 0, rwf_cnt = 0, atv_rows, r, cnt;
  size_t *rv_off_h = NULL, *rwf_base_h = NULL;
  int *sp_mesh_h = NULL, *sp_maxl_h = NULL, *sp_nb_h = NULL;
  double *pao_rv = NULL, *pao_rwf = NULL, *atv_flat = NULL;
  float *chi_host = NULL;
  int *ppr_host = NULL, *pair_hAN = NULL;
  SOG_GpuPair *pair_host = NULL;
  unsigned char *arena = NULL;
  size_t pos = 0, arena_bytes;
  size_t o_rv, o_rwf, o_rvo, o_rwb, o_spm, o_spx, o_spb, o_atv;
  size_t o_gla, o_cla, o_nog, o_ppr, o_prs, o_chi;
  double Stime_atom, Etime_atom;

  const int ng23 = Ngrid2 * Ngrid3;
  const int ng3 = Ngrid3;
  const double g11 = gtv[1][1], g12 = gtv[1][2], g13 = gtv[1][3];
  const double g21 = gtv[2][1], g22 = gtv[2][2], g23 = gtv[2][3];
  const double g31 = gtv[3][1], g32 = gtv[3][2], g33 = gtv[3][3];
  const double org1 = Grid_Origin[1], org2 = Grid_Origin[2], org3 = Grid_Origin[3];

  MPI_Comm_rank(mpi_comm_level1, &myid);

  /* ---- per-atom staging maxima ---- */

  npair_cap = 0;
  for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
    size_t fpts = 0, fchi = 0;
    int npair = 0;

    Gc_AN = M2G[Mc_AN];
    Cwan = WhatSpecies[Gc_AN];
    NO0 = Spe_Total_NO[Cwan];

    if (max_pts < (size_t)GridN_Atom[Gc_AN]) max_pts = (size_t)GridN_Atom[Gc_AN];
    cnt = (size_t)GridN_Atom[Gc_AN] * (size_t)NO0;
    if (max_chi < cnt) max_chi = cnt;

    for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
      Gh_AN = natn[Gc_AN][h_AN];
      if (G2ID[Gh_AN] != myid && 0 < NumOLG[Mc_AN][h_AN]) {
        NO1 = Spe_Total_NO[WhatSpecies[Gh_AN]];
        fpts += (size_t)NumOLG[Mc_AN][h_AN];
        fchi += (size_t)NumOLG[Mc_AN][h_AN] * (size_t)NO1;
        npair++;
      }
    }
    if (max_fpts < fpts) max_fpts = fpts;
    if (max_fchi < fchi) max_fchi = fchi;
    if (npair_cap < npair) npair_cap = npair;
  }
  if (npair_cap < 1) npair_cap = 1;

  /* ---- flatten the PAO radial tables and atv ---- */

  rv_off_h = (size_t*)malloc(sizeof(size_t) * (size_t)(SpeciesNum + 1));
  rwf_base_h = (size_t*)malloc(sizeof(size_t) * (size_t)(SpeciesNum + 1));
  sp_mesh_h = (int*)malloc(sizeof(int) * (size_t)SpeciesNum);
  sp_maxl_h = (int*)malloc(sizeof(int) * (size_t)SpeciesNum);
  sp_nb_h = (int*)malloc(sizeof(int) * (size_t)SpeciesNum * (SOG_L0MAX + 1));

  if (rv_off_h == NULL || rwf_base_h == NULL || sp_mesh_h == NULL ||
      sp_maxl_h == NULL || sp_nb_h == NULL) goto host_fallback;

  for (w = 0; w < SpeciesNum; w++) {
    rv_off_h[w] = rv_cnt;
    rwf_base_h[w] = rwf_cnt;
    sp_mesh_h[w] = Spe_Num_Mesh_PAO[w];
    sp_maxl_h[w] = Spe_MaxL_Basis[w];
    for (L0 = 0; L0 <= SOG_L0MAX; L0++) {
      int nb = (L0 <= Spe_MaxL_Basis[w] ? Spe_Num_Basis[w][L0] : 0);

      sp_nb_h[w * (SOG_L0MAX + 1) + L0] = nb;
      rwf_cnt += (size_t)nb * (size_t)Spe_Num_Mesh_PAO[w];
    }
    rv_cnt += (size_t)Spe_Num_Mesh_PAO[w];
  }
  rv_off_h[SpeciesNum] = rv_cnt;
  rwf_base_h[SpeciesNum] = rwf_cnt;

  pao_rv = (double*)malloc(sizeof(double) * (rv_cnt == 0 ? 1 : rv_cnt));
  pao_rwf = (double*)malloc(sizeof(double) * (rwf_cnt == 0 ? 1 : rwf_cnt));
  atv_rows = (size_t)TCpyCell + 1;
  atv_flat = (double*)malloc(sizeof(double) * atv_rows * 3);
  chi_host = (float*)malloc(sizeof(float) * ((max_chi < max_fchi ? max_fchi : max_chi) == 0 ? 1 :
                                             (max_chi < max_fchi ? max_fchi : max_chi)));
  ppr_host = (int*)malloc(sizeof(int) * (max_fpts == 0 ? 1 : max_fpts));
  pair_hAN = (int*)malloc(sizeof(int) * (size_t)npair_cap);
  pair_host = (SOG_GpuPair*)malloc(sizeof(SOG_GpuPair) * (size_t)npair_cap);

  if (pao_rv == NULL || pao_rwf == NULL || atv_flat == NULL || chi_host == NULL ||
      ppr_host == NULL || pair_hAN == NULL || pair_host == NULL) goto host_fallback;

  for (w = 0; w < SpeciesNum; w++) {
    size_t rpos = rwf_base_h[w];

    for (i = 0; i < Spe_Num_Mesh_PAO[w]; i++) {
      pao_rv[rv_off_h[w] + (size_t)i] = Spe_PAO_RV[w][i];
    }
    for (L0 = 0; L0 <= Spe_MaxL_Basis[w]; L0++) {
      for (Mul0 = 0; Mul0 < Spe_Num_Basis[w][L0]; Mul0++) {
        for (i = 0; i < Spe_Num_Mesh_PAO[w]; i++) {
          pao_rwf[rpos++] = Spe_PAO_RWF[w][L0][Mul0][i];
        }
      }
    }
  }

  for (r = 0; r < atv_rows; r++) {
    atv_flat[3 * r + 0] = atv[r][1];
    atv_flat[3 * r + 1] = atv[r][2];
    atv_flat[3 * r + 2] = atv[r][3];
  }

  /* ---- one arena claim covers the whole footprint ---- */

  o_rv = SOG_arena_off(&pos, sizeof(double) * (rv_cnt == 0 ? 1 : rv_cnt));
  o_rwf = SOG_arena_off(&pos, sizeof(double) * (rwf_cnt == 0 ? 1 : rwf_cnt));
  o_rvo = SOG_arena_off(&pos, sizeof(size_t) * (size_t)(SpeciesNum + 1));
  o_rwb = SOG_arena_off(&pos, sizeof(size_t) * (size_t)(SpeciesNum + 1));
  o_spm = SOG_arena_off(&pos, sizeof(int) * (size_t)SpeciesNum);
  o_spx = SOG_arena_off(&pos, sizeof(int) * (size_t)SpeciesNum);
  o_spb = SOG_arena_off(&pos, sizeof(int) * (size_t)SpeciesNum * (SOG_L0MAX + 1));
  o_atv = SOG_arena_off(&pos, sizeof(double) * atv_rows * 3);
  o_gla = SOG_arena_off(&pos, sizeof(int) * (max_pts == 0 ? 1 : max_pts));
  o_cla = SOG_arena_off(&pos, sizeof(int) * (max_pts == 0 ? 1 : max_pts));
  o_nog = SOG_arena_off(&pos, sizeof(int) * (max_fpts == 0 ? 1 : max_fpts));
  o_ppr = SOG_arena_off(&pos, sizeof(int) * (max_fpts == 0 ? 1 : max_fpts));
  o_prs = SOG_arena_off(&pos, sizeof(SOG_GpuPair) * (size_t)npair_cap);
  cnt = (max_chi < max_fchi ? max_fchi : max_chi);
  o_chi = SOG_arena_off(&pos, sizeof(float) * (cnt == 0 ? 1 : cnt));
  arena_bytes = pos;

  arena = (unsigned char*)SOG_arena_try(arena_bytes);
  if (arena == NULL) {
    if (!fail_notice_done) {
      fail_notice_done = 1;
      fprintf(stderr,
              "Set_Orbitals_Grid: %.1f MiB of device staging unavailable; using the host path.\n",
              (double)arena_bytes / (1024.0 * 1024.0));
      fflush(stderr);
    }
    goto host_fallback;
  }

  acc_memcpy_to_device(arena + o_rv, pao_rv, sizeof(double) * (rv_cnt == 0 ? 1 : rv_cnt));
  acc_memcpy_to_device(arena + o_rwf, pao_rwf, sizeof(double) * (rwf_cnt == 0 ? 1 : rwf_cnt));
  acc_memcpy_to_device(arena + o_rvo, rv_off_h, sizeof(size_t) * (size_t)(SpeciesNum + 1));
  acc_memcpy_to_device(arena + o_rwb, rwf_base_h, sizeof(size_t) * (size_t)(SpeciesNum + 1));
  acc_memcpy_to_device(arena + o_spm, sp_mesh_h, sizeof(int) * (size_t)SpeciesNum);
  acc_memcpy_to_device(arena + o_spx, sp_maxl_h, sizeof(int) * (size_t)SpeciesNum);
  acc_memcpy_to_device(arena + o_spb, sp_nb_h, sizeof(int) * (size_t)SpeciesNum * (SOG_L0MAX + 1));
  acc_memcpy_to_device(arena + o_atv, atv_flat, sizeof(double) * atv_rows * 3);

  {
    const int *d_gla = (const int*)(const void*)(arena + o_gla);
    const int *d_cla = (const int*)(const void*)(arena + o_cla);
    const int *d_nog = (const int*)(const void*)(arena + o_nog);
    const int *d_ppr = (const int*)(const void*)(arena + o_ppr);
    const SOG_GpuPair *d_prs = (const SOG_GpuPair*)(const void*)(arena + o_prs);
    const double *d_atv = (const double*)(const void*)(arena + o_atv);
    const double *d_rv = (const double*)(const void*)(arena + o_rv);
    const double *d_rwf = (const double*)(const void*)(arena + o_rwf);
    const size_t *d_rvo = (const size_t*)(const void*)(arena + o_rvo);
    const size_t *d_rwb = (const size_t*)(const void*)(arena + o_rwb);
    const int *d_spm = (const int*)(const void*)(arena + o_spm);
    const int *d_spx = (const int*)(const void*)(arena + o_spx);
    const int *d_spb = (const int*)(const void*)(arena + o_spb);
    float *d_chi = (float*)(void*)(arena + o_chi);

    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {

      int npts, npair;
      size_t fpts, fchi;

      dtime(&Stime_atom);

      Gc_AN = M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];
      NO0 = Spe_Total_NO[Cwan];
      npts = GridN_Atom[Gc_AN];

      /* ---- part 1: the atom's own orbitals on its grid ---- */

      if (0 < npts) {

        int Nc;

        acc_memcpy_to_device((void*)d_gla, GridListAtom[Mc_AN], sizeof(int) * (size_t)npts);
        acc_memcpy_to_device((void*)d_cla, CellListAtom[Mc_AN], sizeof(int) * (size_t)npts);

        pair_host[0].wan = Cwan;
        pair_host[0].no = NO0;
        pair_host[0].pt0 = 0;
        pair_host[0].pad = 0;
        pair_host[0].out = 0;
        pair_host[0].gx = Gxyz[Gc_AN][1];
        pair_host[0].gy = Gxyz[Gc_AN][2];
        pair_host[0].gz = Gxyz[Gc_AN][3];
        pair_host[0].ax = 0.0;
        pair_host[0].ay = 0.0;
        pair_host[0].az = 0.0;
        acc_memcpy_to_device((void*)d_prs, pair_host, sizeof(SOG_GpuPair));

        SOG_gpu_eval(npts, 1, 1, d_nog, d_ppr, d_prs, d_gla, d_cla, d_atv,
                     d_rv, d_rwf, d_rvo, d_rwb, d_spm, d_spx, d_spb,
                     ng23, ng3, g11, g12, g13, g21, g22, g23, g31, g32, g33,
                     org1, org2, org3, d_chi);

        acc_memcpy_from_device(chi_host, d_chi, sizeof(float) * (size_t)npts * (size_t)NO0);

#pragma omp parallel for private(i)
        for (Nc = 0; Nc < npts; Nc++) {
          const float *src = chi_host + (size_t)Nc * (size_t)NO0;
          Type_Orbs_Grid *dst = Orbs_Grid[Mc_AN][Nc];

          for (i = 0; i < NO0; i++) dst[i] = src[i];
        }
      }

      /* ---- part 2: remote neighbours on the shared grid points ---- */

      npair = 0;
      fpts = 0;
      fchi = 0;
      for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {

        Gh_AN = natn[Gc_AN][h_AN];

        if (G2ID[Gh_AN] != myid && 0 < NumOLG[Mc_AN][h_AN]) {

          const int nolg = NumOLG[Mc_AN][h_AN];
          size_t k;

          Rnh = ncn[Gc_AN][h_AN];
          Hwan = WhatSpecies[Gh_AN];
          NO1 = Spe_Total_NO[Hwan];

          pair_host[npair].wan = Hwan;
          pair_host[npair].no = NO1;
          pair_host[npair].pt0 = (int)fpts;
          pair_host[npair].pad = 0;
          pair_host[npair].out = fchi;
          pair_host[npair].gx = Gxyz[Gh_AN][1];
          pair_host[npair].gy = Gxyz[Gh_AN][2];
          pair_host[npair].gz = Gxyz[Gh_AN][3];
          pair_host[npair].ax = atv[Rnh][1];
          pair_host[npair].ay = atv[Rnh][2];
          pair_host[npair].az = atv[Rnh][3];
          pair_hAN[npair] = h_AN;

          acc_memcpy_to_device((void*)(d_nog + fpts), GListTAtoms1[Mc_AN][h_AN],
                               sizeof(int) * (size_t)nolg);
          for (k = 0; k < (size_t)nolg; k++) ppr_host[fpts + k] = npair;

          fpts += (size_t)nolg;
          fchi += (size_t)nolg * (size_t)NO1;
          npair++;
        }
      }

      if (0 < npair) {

        int p;

        acc_memcpy_to_device((void*)d_ppr, ppr_host, sizeof(int) * fpts);
        acc_memcpy_to_device((void*)d_prs, pair_host, sizeof(SOG_GpuPair) * (size_t)npair);

        SOG_gpu_eval((int)fpts, 0, 0, d_nog, d_ppr, d_prs, d_gla, d_cla, d_atv,
                     d_rv, d_rwf, d_rvo, d_rwb, d_spm, d_spx, d_spb,
                     ng23, ng3, g11, g12, g13, g21, g22, g23, g31, g32, g33,
                     org1, org2, org3, d_chi);

        acc_memcpy_from_device(chi_host, d_chi, sizeof(float) * fchi);

        for (p = 0; p < npair; p++) {

          const int nolg = NumOLG[Mc_AN][pair_hAN[p]];
          const float *base = chi_host + pair_host[p].out;
          Type_Orbs_Grid **dst_rows = Orbs_Grid_FNAN[Mc_AN][pair_hAN[p]];
          int Nog, j;

          NO1 = pair_host[p].no;

#pragma omp parallel for private(j)
          for (Nog = 0; Nog < nolg; Nog++) {
            const float *src = base + (size_t)Nog * (size_t)NO1;
            Type_Orbs_Grid *dst = dst_rows[Nog];

            for (j = 0; j < NO1; j++) dst[j] = src[j];
          }
        }
      }

      dtime(&Etime_atom);
      time_per_atom[Gc_AN] += Etime_atom - Stime_atom;
    }
  }

  acc_free(arena);
  /* return the arena to CUDA instead of the process-local freelist, so
     the other ranks sharing this device can actually use the memory */
  if (cudaDeviceSynchronize() == cudaSuccess) acc_clear_freelists();
  free(rv_off_h); free(rwf_base_h); free(sp_mesh_h); free(sp_maxl_h); free(sp_nb_h);
  free(pao_rv); free(pao_rwf); free(atv_flat); free(chi_host);
  free(ppr_host); free(pair_hAN); free(pair_host);

  if (!active_notice_done) {
    active_notice_done = 1;
    if (SOG_env_flag("OPENMX_ORBS_GRID_GPU_VERBOSE", 0) && myid == Host_ID) {
      fprintf(stderr, "Set_Orbitals_Grid: GPU path active\n");
      fflush(stderr);
    }
  }
  return 1;

host_fallback:
  free(rv_off_h); free(rwf_base_h); free(sp_mesh_h); free(sp_maxl_h); free(sp_nb_h);
  free(pao_rv); free(pao_rwf); free(atv_flat); free(chi_host);
  free(ppr_host); free(pair_hAN); free(pair_host);
  return 0;
}





/* ---------------------------------------------------------------------
   On-the-fly orbital tiles for the grid integrals (orbs_grid_gpu.h): the
   species' PAO tables, the cell translations and the grid frame stay on
   the device; Set_Hamiltonian evaluates the orbitals of each batch of
   atom pairs from them instead of holding the multi-GiB tables.
   --------------------------------------------------------------------- */
static SOG_DeviceTables SOG_dev = {0};

/* OPENMX_ORBS_EVAL_PRECISION: "df" (double-float, the default) or "fp64"
   for the on-the-fly tiles; the tables are always built in FP64 */
static int SOG_eval_double_float(void)
{
  const char *value = getenv("OPENMX_ORBS_EVAL_PRECISION");

  if (value == NULL || value[0] == '\0') return 1;
  if (strcmp(value, "fp64") == 0 || strcmp(value, "FP64") == 0 || strcmp(value, "64") == 0 ||
      strcmp(value, "double") == 0) return 0;
  return 1;
}

void SOG_Device_Release(void)
{
  if (SOG_dev.arena != NULL) {
    acc_free(SOG_dev.arena);
    if (cudaDeviceSynchronize() == cudaSuccess) acc_clear_freelists();
  }
  memset(&SOG_dev, 0, sizeof(SOG_dev));
}

int SOG_Device_Prepare(void)
{
  int w, L0, Mul0, i;
  size_t rv_cnt = 0, rwf_cnt = 0, atv_rows, r, pos = 0;
  size_t *rv_off_h = NULL, *rwf_base_h = NULL;
  int *sp_mesh_h = NULL, *sp_maxl_h = NULL, *sp_nb_h = NULL;
  double *pao_rv = NULL, *pao_rwf = NULL, *atv_flat = NULL;
  float *rwd_h = NULL, *mcd_h = NULL, *spl_h = NULL;
  size_t o_rv, o_rwf, o_rvo, o_rwb, o_spm, o_spx, o_spb, o_atv, o_rwd, o_mcd, o_spl;
  unsigned char *arena;

  if (SOG_dev.ready) return 1;
  if (SOG_dev.unavailable) return 0;
  if (!SOG_gpu_eligible(0)) { SOG_dev.unavailable = 1; return 0; }
  SOG_dev.eval_df = SOG_eval_double_float();

  rv_off_h = (size_t*)malloc(sizeof(size_t) * (size_t)(SpeciesNum + 1));
  rwf_base_h = (size_t*)malloc(sizeof(size_t) * (size_t)(SpeciesNum + 1));
  sp_mesh_h = (int*)malloc(sizeof(int) * (size_t)SpeciesNum);
  sp_maxl_h = (int*)malloc(sizeof(int) * (size_t)SpeciesNum);
  sp_nb_h = (int*)malloc(sizeof(int) * (size_t)SpeciesNum * (SOG_L0MAX + 1));
  if (!rv_off_h || !rwf_base_h || !sp_mesh_h || !sp_maxl_h || !sp_nb_h) goto host_fallback;
  for (w = 0; w < SpeciesNum; w++) {
    rv_off_h[w] = rv_cnt;
    rwf_base_h[w] = rwf_cnt;
    sp_mesh_h[w] = Spe_Num_Mesh_PAO[w];
    sp_maxl_h[w] = Spe_MaxL_Basis[w];
    for (L0 = 0; L0 <= SOG_L0MAX; L0++) {
      int nb = (L0 <= Spe_MaxL_Basis[w] ? Spe_Num_Basis[w][L0] : 0);
      sp_nb_h[w * (SOG_L0MAX + 1) + L0] = nb;
      rwf_cnt += (size_t)nb * (size_t)Spe_Num_Mesh_PAO[w];
    }
    rv_cnt += (size_t)Spe_Num_Mesh_PAO[w];
  }
  rv_off_h[SpeciesNum] = rv_cnt;
  rwf_base_h[SpeciesNum] = rwf_cnt;
  pao_rv = (double*)malloc(sizeof(double) * (rv_cnt == 0 ? 1 : rv_cnt));
  pao_rwf = (double*)malloc(sizeof(double) * (rwf_cnt == 0 ? 1 : rwf_cnt));
  atv_rows = (size_t)TCpyCell + 1;
  atv_flat = (double*)malloc(sizeof(double) * atv_rows * 3);
  if (!pao_rv || !pao_rwf || !atv_flat) goto host_fallback;
  for (w = 0; w < SpeciesNum; w++) {
    size_t rpos = rwf_base_h[w];
    for (i = 0; i < Spe_Num_Mesh_PAO[w]; i++) pao_rv[rv_off_h[w] + (size_t)i] = Spe_PAO_RV[w][i];
    for (L0 = 0; L0 <= Spe_MaxL_Basis[w]; L0++)
      for (Mul0 = 0; Mul0 < Spe_Num_Basis[w][L0]; Mul0++)
        for (i = 0; i < Spe_Num_Mesh_PAO[w]; i++) pao_rwf[rpos++] = Spe_PAO_RWF[w][L0][Mul0][i];
  }
  for (r = 0; r < atv_rows; r++) {
    atv_flat[3 * r + 0] = atv[r][1];
    atv_flat[3 * r + 1] = atv[r][2];
    atv_flat[3 * r + 2] = atv[r][3];
  }
  /* the double-float tables: every radial value as (hi, lo), and per
     mesh point rv, h2 and the spline weights dum1..dum4 of SOG_point_eval
     as (hi, lo); the ends m = 1 and m = mesh - 1 (unreachable while the
     atom's cutoff lies inside the mesh) take the host's corrected h1 / h3 */
  rwd_h = (float*)malloc(sizeof(float) * 2 * (rwf_cnt == 0 ? 1 : rwf_cnt));
  mcd_h = (float*)malloc(sizeof(float) * SOG_MCD_STRIDE * (rv_cnt == 0 ? 1 : rv_cnt));
  spl_h = (float*)malloc(sizeof(float) * 2 * (size_t)SpeciesNum);
  if (!rwd_h || !mcd_h || !spl_h) goto host_fallback;
  for (r = 0; r < rwf_cnt; r++) {
    const float hi = (float)pao_rwf[r];
    rwd_h[2 * r] = hi;
    rwd_h[2 * r + 1] = (float)(pao_rwf[r] - (double)hi);
  }
  for (w = 0; w < SpeciesNum; w++) {
    const int mesh = Spe_Num_Mesh_PAO[w];
    const double *rvw = pao_rv + rv_off_h[w];
    float *mc = mcd_h + SOG_MCD_STRIDE * rv_off_h[w];
    int m;

    spl_h[2 * w] = (float)log(rvw[0]);
    spl_h[2 * w + 1] = (float)((log(rvw[mesh - 1]) - log(rvw[0])) / (double)(mesh - 1));
    for (m = 0; m < mesh; m++, mc += SOG_MCD_STRIDE) {
      double v[6], h1, h2, h3, dum;
      int q;

      for (q = 0; q < SOG_MCD_STRIDE; q++) mc[q] = 0.0f;
      v[0] = rvw[m];
      if (m < 1) {
        mc[0] = (float)v[0];
        mc[1] = (float)(v[0] - (double)mc[0]);
        continue;
      }
      h2 = rvw[m] - rvw[m - 1];
      h1 = (2 <= m) ? rvw[m - 1] - rvw[m - 2] : 0.0;
      h3 = (m <= mesh - 2) ? rvw[m + 1] - rvw[m] : 0.0;
      if (m == 1) h1 = -(h2 + h3);
      if (m == mesh - 1) h3 = -(h1 + h2);
      v[1] = h2;
      dum = h1 + h2;
      v[2] = h1 / h2 / dum;
      v[3] = h2 / h1 / dum;
      dum = h2 + h3;
      v[4] = h2 / h3 / dum;
      v[5] = h3 / h2 / dum;
      for (q = 0; q < 6; q++) {
        const float hi = (float)v[q];
        mc[2 * q] = hi;
        mc[2 * q + 1] = (float)(v[q] - (double)hi);
      }
    }
  }
  o_rv = SOG_arena_off(&pos, sizeof(double) * (rv_cnt == 0 ? 1 : rv_cnt));
  o_rwf = SOG_arena_off(&pos, sizeof(double) * (rwf_cnt == 0 ? 1 : rwf_cnt));
  o_rvo = SOG_arena_off(&pos, sizeof(size_t) * (size_t)(SpeciesNum + 1));
  o_rwb = SOG_arena_off(&pos, sizeof(size_t) * (size_t)(SpeciesNum + 1));
  o_spm = SOG_arena_off(&pos, sizeof(int) * (size_t)SpeciesNum);
  o_spx = SOG_arena_off(&pos, sizeof(int) * (size_t)SpeciesNum);
  o_spb = SOG_arena_off(&pos, sizeof(int) * (size_t)SpeciesNum * (SOG_L0MAX + 1));
  o_atv = SOG_arena_off(&pos, sizeof(double) * atv_rows * 3);
  o_rwd = SOG_arena_off(&pos, sizeof(float) * 2 * (rwf_cnt == 0 ? 1 : rwf_cnt));
  o_mcd = SOG_arena_off(&pos, sizeof(float) * SOG_MCD_STRIDE * (rv_cnt == 0 ? 1 : rv_cnt));
  o_spl = SOG_arena_off(&pos, sizeof(float) * 2 * (size_t)SpeciesNum);
  arena = (unsigned char*)SOG_arena_try(pos);
  if (arena == NULL) goto host_fallback;
  acc_memcpy_to_device(arena + o_rv, pao_rv, sizeof(double) * (rv_cnt == 0 ? 1 : rv_cnt));
  acc_memcpy_to_device(arena + o_rwf, pao_rwf, sizeof(double) * (rwf_cnt == 0 ? 1 : rwf_cnt));
  acc_memcpy_to_device(arena + o_rvo, rv_off_h, sizeof(size_t) * (size_t)(SpeciesNum + 1));
  acc_memcpy_to_device(arena + o_rwb, rwf_base_h, sizeof(size_t) * (size_t)(SpeciesNum + 1));
  acc_memcpy_to_device(arena + o_spm, sp_mesh_h, sizeof(int) * (size_t)SpeciesNum);
  acc_memcpy_to_device(arena + o_spx, sp_maxl_h, sizeof(int) * (size_t)SpeciesNum);
  acc_memcpy_to_device(arena + o_spb, sp_nb_h, sizeof(int) * (size_t)SpeciesNum * (SOG_L0MAX + 1));
  acc_memcpy_to_device(arena + o_atv, atv_flat, sizeof(double) * atv_rows * 3);
  acc_memcpy_to_device(arena + o_rwd, rwd_h, sizeof(float) * 2 * (rwf_cnt == 0 ? 1 : rwf_cnt));
  acc_memcpy_to_device(arena + o_mcd, mcd_h, sizeof(float) * SOG_MCD_STRIDE * (rv_cnt == 0 ? 1 : rv_cnt));
  acc_memcpy_to_device(arena + o_spl, spl_h, sizeof(float) * 2 * (size_t)SpeciesNum);
  SOG_dev.arena = arena;
  SOG_dev.bytes = pos;
  SOG_dev.rv = (const double*)(const void*)(arena + o_rv);
  SOG_dev.rwf = (const double*)(const void*)(arena + o_rwf);
  SOG_dev.rvo = (const size_t*)(const void*)(arena + o_rvo);
  SOG_dev.rwb = (const size_t*)(const void*)(arena + o_rwb);
  SOG_dev.spm = (const int*)(const void*)(arena + o_spm);
  SOG_dev.spx = (const int*)(const void*)(arena + o_spx);
  SOG_dev.spb = (const int*)(const void*)(arena + o_spb);
  SOG_dev.atv = (const double*)(const void*)(arena + o_atv);
  SOG_dev.rwd = (const float*)(const void*)(arena + o_rwd);
  SOG_dev.mcd = (const float*)(const void*)(arena + o_mcd);
  SOG_dev.spl = (const float*)(const void*)(arena + o_spl);
  SOG_dev.ng23 = Ngrid2 * Ngrid3;
  SOG_dev.ng3 = Ngrid3;
  for (i = 0; i < 3; i++) {
    SOG_dev.g[i][0] = gtv[i + 1][1];
    SOG_dev.g[i][1] = gtv[i + 1][2];
    SOG_dev.g[i][2] = gtv[i + 1][3];
    SOG_dev.org[i] = Grid_Origin[i + 1];
  }
  SOG_dev.ready = 1;
  free(rv_off_h); free(rwf_base_h); free(sp_mesh_h); free(sp_maxl_h); free(sp_nb_h);
  free(pao_rv); free(pao_rwf); free(atv_flat); free(rwd_h); free(mcd_h); free(spl_h);
  return 1;

host_fallback:
  free(rv_off_h); free(rwf_base_h); free(sp_mesh_h); free(sp_maxl_h); free(sp_nb_h);
  free(pao_rv); free(pao_rwf); free(atv_flat); free(rwd_h); free(mcd_h); free(spl_h);
  return 0;
}

const char *SOG_Device_EvalPrecisionName(void)
{
  return SOG_dev.eval_df ? "double-float" : "FP64";
}

const SOG_DeviceTables *SOG_Device_Tables(void)
{
  return SOG_dev.ready ? &SOG_dev : NULL;
}

size_t SOG_Device_Bytes(void)
{
  return SOG_dev.ready ? SOG_dev.bytes : 0;
}

/* The orbitals of the neighbour of every pair at the pair's overlap points:
   pair p has NOLG[p] points, the k-th at the central atom's sphere index
   nolg_Nc[nolg_off[p] + k] (its global grid index and cell in gla / cla at
   gla_off[p] + that index), and writes no[p] values per point from
   out[out_off[p] + k * no[p]]. */
void SOG_Device_EvalPairTiles(int pair_count, const SOG_GpuPair *pairs, const int *pair_NOLG,
                              const size_t *nolg_off, const size_t *out_off, const size_t *gla_off,
                              const int *nolg_Nc, const int *gla, const int *cla, float *out)
{
  const SOG_DeviceTables *t = &SOG_dev;
  const double *atvf = t->atv, *rv_all = t->rv, *rwf_all = t->rwf;
  const size_t *rv_off = t->rvo, *rwf_base = t->rwb;
  const int *sp_mesh = t->spm, *sp_maxl = t->spx, *sp_nb = t->spb;
  const int ng23 = t->ng23, ng3 = t->ng3;
  const double g11 = t->g[0][0], g12 = t->g[0][1], g13 = t->g[0][2];
  const double g21 = t->g[1][0], g22 = t->g[1][1], g23 = t->g[1][2];
  const double g31 = t->g[2][0], g32 = t->g[2][1], g33 = t->g[2][2];
  const double org1 = t->org[0], org2 = t->org[1], org3 = t->org[2];
  int p;

  if (!t->ready || pair_count <= 0) return;
  if (t->eval_df) {
    const float *rwd_all = t->rwd, *mcd_all = t->mcd, *spl = t->spl;
#pragma acc parallel loop gang \
  deviceptr(pairs, pair_NOLG, nolg_off, out_off, gla_off, nolg_Nc, gla, cla, out, \
            atvf, rv_all, rwf_all, rwd_all, mcd_all, spl, rv_off, rwf_base, sp_mesh, sp_maxl, sp_nb)
    for (p = 0; p < pair_count; p++) {
      const SOG_GpuPair pr = pairs[p];
      const int nolg = pair_NOLG[p];
      const size_t noff = nolg_off[p], ooff = out_off[p], goff = gla_off[p];
      int k;
#pragma acc loop vector
      for (k = 0; k < nolg; k++) {
        const int Nc = nolg_Nc[noff + (size_t)k];
        SOG_point_eval_df(pr.wan, pr.no, pr.gx, pr.gy, pr.gz, pr.ax, pr.ay, pr.az,
                          gla[goff + (size_t)Nc], cla[goff + (size_t)Nc],
                          atvf, rv_all, rwf_all, rwd_all, mcd_all, spl, rv_off, rwf_base, sp_mesh, sp_maxl, sp_nb,
                          ng23, ng3, g11, g12, g13, g21, g22, g23, g31, g32, g33, org1, org2, org3,
                          out + ooff + (size_t)k * (size_t)pr.no);
      }
    }
    return;
  }
#pragma acc parallel loop gang \
  deviceptr(pairs, pair_NOLG, nolg_off, out_off, gla_off, nolg_Nc, gla, cla, out, \
            atvf, rv_all, rwf_all, rv_off, rwf_base, sp_mesh, sp_maxl, sp_nb)
  for (p = 0; p < pair_count; p++) {
    const SOG_GpuPair pr = pairs[p];
    const int nolg = pair_NOLG[p];
    const size_t noff = nolg_off[p], ooff = out_off[p], goff = gla_off[p];
    int k;
#pragma acc loop vector
    for (k = 0; k < nolg; k++) {
      const int Nc = nolg_Nc[noff + (size_t)k];
      SOG_point_eval(pr.wan, pr.no, pr.gx, pr.gy, pr.gz, pr.ax, pr.ay, pr.az,
                     gla[goff + (size_t)Nc], cla[goff + (size_t)Nc],
                     atvf, rv_all, rwf_all, rv_off, rwf_base, sp_mesh, sp_maxl, sp_nb,
                     ng23, ng3, g11, g12, g13, g21, g22, g23, g31, g32, g33, org1, org2, org3,
                     out + ooff + (size_t)k * (size_t)pr.no);
    }
  }
}

/* The orbitals of one atom (pair: its species, orbital count and position,
   no shift) at all npts points of its sphere (gla / cla), no values per
   point from out */
void SOG_Device_EvalAtomTile(const SOG_GpuPair *pair, int npts, const int *gla, const int *cla, float *out)
{
  const SOG_DeviceTables *t = &SOG_dev;
  const double *atvf = t->atv, *rv_all = t->rv, *rwf_all = t->rwf;
  const size_t *rv_off = t->rvo, *rwf_base = t->rwb;
  const int *sp_mesh = t->spm, *sp_maxl = t->spx, *sp_nb = t->spb;
  const int ng23 = t->ng23, ng3 = t->ng3;
  const double g11 = t->g[0][0], g12 = t->g[0][1], g13 = t->g[0][2];
  const double g21 = t->g[1][0], g22 = t->g[1][1], g23 = t->g[1][2];
  const double g31 = t->g[2][0], g32 = t->g[2][1], g33 = t->g[2][2];
  const double org1 = t->org[0], org2 = t->org[1], org3 = t->org[2];
  const SOG_GpuPair pr = *pair;
  int ip;

  if (!t->ready || npts <= 0) return;
  if (t->eval_df) {
    const float *rwd_all = t->rwd, *mcd_all = t->mcd, *spl = t->spl;
#pragma acc parallel loop gang vector vector_length(128) \
  deviceptr(gla, cla, out, atvf, rv_all, rwf_all, rwd_all, mcd_all, spl, rv_off, rwf_base, sp_mesh, sp_maxl, sp_nb)
    for (ip = 0; ip < npts; ip++) {
      SOG_point_eval_df(pr.wan, pr.no, pr.gx, pr.gy, pr.gz, 0.0, 0.0, 0.0, gla[ip], cla[ip],
                        atvf, rv_all, rwf_all, rwd_all, mcd_all, spl, rv_off, rwf_base, sp_mesh, sp_maxl, sp_nb,
                        ng23, ng3, g11, g12, g13, g21, g22, g23, g31, g32, g33, org1, org2, org3,
                        out + (size_t)ip * (size_t)pr.no);
    }
    return;
  }
#pragma acc parallel loop gang vector vector_length(128) \
  deviceptr(gla, cla, out, atvf, rv_all, rwf_all, rv_off, rwf_base, sp_mesh, sp_maxl, sp_nb)
  for (ip = 0; ip < npts; ip++) {
    SOG_point_eval(pr.wan, pr.no, pr.gx, pr.gy, pr.gz, 0.0, 0.0, 0.0, gla[ip], cla[ip],
                   atvf, rv_all, rwf_all, rv_off, rwf_base, sp_mesh, sp_maxl, sp_nb,
                   ng23, ng3, g11, g12, g13, g21, g22, g23, g31, g32, g33, org1, org2, org3,
                   out + (size_t)ip * (size_t)pr.no);
  }
}

double Set_Orbitals_Grid(int Cnt_kind)
{
  SOG_Device_Release();   /* the grid frame may have changed */
  int i,j,n,Mc_AN,Gc_AN,Cwan,NO0,GNc,GRc;
  int Gh_AN,Mh_AN,Rnh,Hwan,NO1,Nog,h_AN;
  long int k,Nc;
  double time0;
  double x,y,z,dx,dy,dz;
  double TStime,TEtime;
  double Cxyz[4];
  int numprocs,myid,tag=999,ID,IDS,IDR;
  double Stime_atom,Etime_atom;

  MPI_Status stat;
  MPI_Request request;

  /* for OpenMP */
  int OMPID,Nthrds,Nprocs;

  /* Orbital values and their grid topology back the density GPU service
     cache.  Drop the previous epoch before any values are regenerated. */
  Set_Density_Grid_GPU_Invalidate();
  Set_Hamiltonian_Invalidate_OpenACC_MatrixElements_Cache();

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);
  
  dtime(&TStime);

  /* device fast path; on any ineligibility or allocation failure the
     untouched host loops below produce the same values */
  if (SOG_gpu_eligible(Cnt_kind) && Set_Orbitals_Grid_GPU()) {
    dtime(&TEtime);
    return TEtime - TStime;
  }

  /*****************************************************
                Calculate orbitals on grids
  *****************************************************/

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

    dtime(&Stime_atom);

    Gc_AN = M2G[Mc_AN];    
    Cwan = WhatSpecies[Gc_AN];

    if (Cnt_kind==0)  NO0 = Spe_Total_NO[Cwan];
    else              NO0 = Spe_Total_CNO[Cwan]; 

#pragma omp parallel shared(Comp2Real,Spe_PAO_RWF,Spe_Num_Basis,Spe_MaxL_Basis,Spe_PAO_RV,Spe_Num_Mesh_PAO,List_YOUSO,Orbs_Grid,Cnt_kind,Gxyz,atv,CellListAtom,GridListAtom,GridN_Atom,Gc_AN,Cwan,Mc_AN,NO0) private(OMPID,Nthrds,Nprocs,Nc,GNc,GRc,Cxyz,x,y,z,dx,dy,dz,i,j)
    {
      double *Chi0;
      double Cxyz0[4]; 
      double **RF;
      double **AF;
      int i,L0,Mul0,M0,i1;
      double S_coordinate[3];
      double dum,dum1,dum2,dum3,dum4,a,b,c,d;
      double siQ,coQ,siP,coP,Q,P,R;
      double rm,df,sum0,sum1;
      double SH[Supported_MaxL*2+1][2];
      double dSHt[Supported_MaxL*2+1][2];
      double dSHp[Supported_MaxL*2+1][2];

      /* Radial */
      int mp_min,mp_max,m,po,wan;
      double h1,h2,h3,f1,f2,f3,f4;
      double g1,g2,x1,x2,y1,y2,y12,y22,f;
      double r,r1,theta,phi,Min_r;

      /* allocation of array */

      Chi0 = (double*)malloc(sizeof(double)*List_YOUSO[7]);

      RF = (double**)malloc(sizeof(double*)*(List_YOUSO[25]+1));
      for (i=0; i<(List_YOUSO[25]+1); i++){
	RF[i] = (double*)malloc(sizeof(double)*List_YOUSO[24]);
      }

      AF = (double**)malloc(sizeof(double*)*(List_YOUSO[25]+1));
      for (i=0; i<(List_YOUSO[25]+1); i++){
	AF[i] = (double*)malloc(sizeof(double)*(2*(List_YOUSO[25]+1)+1));
      }

      /* get info. on OpenMP */ 

      OMPID = omp_get_thread_num();
      Nthrds = omp_get_num_threads();
      Nprocs = omp_get_num_procs();

      for (Nc=OMPID*GridN_Atom[Gc_AN]/Nthrds; Nc<(OMPID+1)*GridN_Atom[Gc_AN]/Nthrds; Nc++){

	GNc = GridListAtom[Mc_AN][Nc]; 
	GRc = CellListAtom[Mc_AN][Nc];

	Get_Grid_XYZ(GNc,Cxyz);
	x = Cxyz[1] + atv[GRc][1] - Gxyz[Gc_AN][1]; 
	y = Cxyz[2] + atv[GRc][2] - Gxyz[Gc_AN][2]; 
	z = Cxyz[3] + atv[GRc][3] - Gxyz[Gc_AN][3];

	if (Cnt_kind==0){

          /* Get_Orbitals(Cwan,x,y,z,Chi0); */
          /* start of inlining of Get_Orbitals */

          wan = Cwan;

          /* xyz2spherical(x,y,z,0.0,0.0,0.0,S_coordinate); */
          /* start of inlining of xyz2spherical */

	  Min_r = 10e-15;
	  dum = x*x + y*y; 
	  r = sqrt(dum + z*z);
	  r1 = sqrt(dum);

	  if (Min_r<=r){

	    if (r<fabs(z))
	      dum1 = sgn(z)*1.0;
	    else
	      dum1 = z/r;

	    theta = acos(dum1);

	    if (Min_r<=r1){
	      if (0.0<=x){

		if (r1<fabs(y))
		  dum1 = sgn(y)*1.0;
		else
		  dum1 = y/r1;        
  
		phi = asin(dum1);
	      }
	      else{

		if (r1<fabs(y))
		  dum1 = sgn(y)*1.0;
		else
		  dum1 = y/r1;        

		phi = PI - asin(dum1);
	      }
	    }
	    else{
	      phi = 0.0;
	    }
	  }
	  else{
	    theta = 0.5*PI;
	    phi = 0.0;
	  }

	  R = r;
	  Q = theta;
	  P = phi;

	  /* end of inlining of xyz2spherical */

	  po = 0;
	  mp_min = 0;
	  mp_max = Spe_Num_Mesh_PAO[wan] - 1;

	  if (Spe_PAO_RV[wan][Spe_Num_Mesh_PAO[wan]-1]<R){

	    for (L0=0; L0<=Spe_MaxL_Basis[wan]; L0++){
	      for (Mul0=0; Mul0<Spe_Num_Basis[wan][L0]; Mul0++){
		RF[L0][Mul0] = 0.0;
	      }
	    }

	    po = 1;
	  }

	  else if (R<Spe_PAO_RV[wan][0]){

	    m = 4;
	    rm = Spe_PAO_RV[wan][m];

	    h1 = Spe_PAO_RV[wan][m-1] - Spe_PAO_RV[wan][m-2];
	    h2 = Spe_PAO_RV[wan][m]   - Spe_PAO_RV[wan][m-1];
	    h3 = Spe_PAO_RV[wan][m+1] - Spe_PAO_RV[wan][m];

	    x1 = rm - Spe_PAO_RV[wan][m-1];
	    x2 = rm - Spe_PAO_RV[wan][m];
	    y1 = x1/h2;
	    y2 = x2/h2;
	    y12 = y1*y1;
	    y22 = y2*y2;

	    dum = h1 + h2;
	    dum1 = h1/h2/dum;
	    dum2 = h2/h1/dum;
	    dum = h2 + h3;
	    dum3 = h2/h3/dum;
	    dum4 = h3/h2/dum;

	    for (L0=0; L0<=Spe_MaxL_Basis[wan]; L0++){
	      for (Mul0=0; Mul0<Spe_Num_Basis[wan][L0]; Mul0++){

		f1 = Spe_PAO_RWF[wan][L0][Mul0][m-2];
		f2 = Spe_PAO_RWF[wan][L0][Mul0][m-1];
		f3 = Spe_PAO_RWF[wan][L0][Mul0][m];
		f4 = Spe_PAO_RWF[wan][L0][Mul0][m+1];

		if (m==1){
		  h1 = -(h2+h3);
		  f1 = f4;
		}
		else if (m==(Spe_Num_Mesh_PAO[wan]-1)){
		  h3 = -(h1+h2);
		  f4 = f1;
		}

		dum = f3 - f2;
		g1 = dum*dum1 + (f2-f1)*dum2;
		g2 = (f4-f3)*dum3 + dum*dum4;

		f =  y22*(3.0*f2 + h2*g1 + (2.0*f2 + h2*g1)*y2)
		  + y12*(3.0*f3 - h2*g2 - (2.0*f3 - h2*g2)*y1);

		df = 2.0*y2/h2*(3.0*f2 + h2*g1 + (2.0*f2 + h2*g1)*y2)
		  + y22*(2.0*f2 + h2*g1)/h2
		  + 2.0*y1/h2*(3.0*f3 - h2*g2 - (2.0*f3 - h2*g2)*y1)
		  - y12*(2.0*f3 - h2*g2)/h2;

		if (L0==0){
		  a = 0.0;
		  b = 0.5*df/rm;
		  c = 0.0;
		  d = f - b*rm*rm;
		}

		else if (L0==1){
		  a = (rm*df - f)/(2.0*rm*rm*rm);
		  b = 0.0;
		  c = df - 3.0*a*rm*rm;
		  d = 0.0;
		}

		else{
		  b = (3.0*f - rm*df)/(rm*rm);
		  a = (f - b*rm*rm)/(rm*rm*rm);
		  c = 0.0;
		  d = 0.0;
		}

		RF[L0][Mul0] = a*R*R*R + b*R*R + c*R + d;

	      }
	    }

	  }

	  else{

	    do{
	      m = (mp_min + mp_max)/2;
	      if (Spe_PAO_RV[wan][m]<R)
		mp_min = m;
	      else 
		mp_max = m;
	    }
	    while((mp_max-mp_min)!=1);
	    m = mp_max;

	    h1 = Spe_PAO_RV[wan][m-1] - Spe_PAO_RV[wan][m-2];
	    h2 = Spe_PAO_RV[wan][m]   - Spe_PAO_RV[wan][m-1];
	    h3 = Spe_PAO_RV[wan][m+1] - Spe_PAO_RV[wan][m];

	    x1 = R - Spe_PAO_RV[wan][m-1];
	    x2 = R - Spe_PAO_RV[wan][m];
	    y1 = x1/h2;
	    y2 = x2/h2;
	    y12 = y1*y1;
	    y22 = y2*y2;

	    dum = h1 + h2;
	    dum1 = h1/h2/dum;
	    dum2 = h2/h1/dum;
	    dum = h2 + h3;
	    dum3 = h2/h3/dum;
	    dum4 = h3/h2/dum;

	    for (L0=0; L0<=Spe_MaxL_Basis[wan]; L0++){
	      for (Mul0=0; Mul0<Spe_Num_Basis[wan][L0]; Mul0++){

		f1 = Spe_PAO_RWF[wan][L0][Mul0][m-2];
		f2 = Spe_PAO_RWF[wan][L0][Mul0][m-1];
		f3 = Spe_PAO_RWF[wan][L0][Mul0][m];
		f4 = Spe_PAO_RWF[wan][L0][Mul0][m+1];

		if (m==1){
		  h1 = -(h2+h3);
		  f1 = f4;
		}
		else if (m==(Spe_Num_Mesh_PAO[wan]-1)){
		  h3 = -(h1+h2);
		  f4 = f1;
		}

		dum = f3 - f2;
		g1 = dum*dum1 + (f2-f1)*dum2;
		g2 = (f4-f3)*dum3 + dum*dum4;

		f =  y22*(3.0*f2 + h2*g1 + (2.0*f2 + h2*g1)*y2)
		  + y12*(3.0*f3 - h2*g2 - (2.0*f3 - h2*g2)*y1);

		RF[L0][Mul0] = f;

	      }
	    } 

	  }

	  if (po==0){

	    /* Angular */
	    siQ = sin(Q);
	    coQ = cos(Q);
	    siP = sin(P);
	    coP = cos(P);

	    for (L0=0; L0<=Spe_MaxL_Basis[wan]; L0++){

	      if (L0==0){
		AF[0][0] = 0.282094791773878;
	      }
	      else if (L0==1){
		dum = 0.48860251190292*siQ;
		AF[1][0] = dum*coP;
		AF[1][1] = dum*siP;
		AF[1][2] = 0.48860251190292*coQ;
	      }
	      else if (L0==2){
		dum1 = siQ*siQ;
		dum2 = 1.09254843059208*siQ*coQ;
		AF[2][0] = 0.94617469575756*coQ*coQ - 0.31539156525252;
		AF[2][1] = 0.54627421529604*dum1*(1.0 - 2.0*siP*siP);
		AF[2][2] = 1.09254843059208*dum1*siP*coP;
		AF[2][3] = dum2*coP;
		AF[2][4] = dum2*siP;
	      }

	      else if (L0==3){
		AF[3][0] = 0.373176332590116*(5.0*coQ*coQ*coQ - 3.0*coQ);
		AF[3][1] = 0.457045799464466*coP*siQ*(5.0*coQ*coQ - 1.0);
		AF[3][2] = 0.457045799464466*siP*siQ*(5.0*coQ*coQ - 1.0);
		AF[3][3] = 1.44530572132028*siQ*siQ*coQ*(coP*coP-siP*siP);
		AF[3][4] = 2.89061144264055*siQ*siQ*coQ*siP*coP;
		AF[3][5] = 0.590043589926644*siQ*siQ*siQ*(4.0*coP*coP*coP - 3.0*coP);
		AF[3][6] = 0.590043589926644*siQ*siQ*siQ*(3.0*siP - 4.0*siP*siP*siP);
	      }

	      else if (4<=L0){

		/* calculation of complex spherical harmonics functions */
		for(m=-L0; m<=L0; m++){ 
		  ComplexSH(L0,m,Q,P,SH[L0+m],dSHt[L0+m],dSHp[L0+m]);
		}

		/* transformation of complex to real */
		for (i=0; i<(L0*2+1); i++){

		  sum0 = 0.0;
		  sum1 = 0.0; 
		  for (j=0; j<(L0*2+1); j++){
		    sum0 += Comp2Real[L0][i][j].r*SH[j][0] - Comp2Real[L0][i][j].i*SH[j][1]; 
		    sum1 += Comp2Real[L0][i][j].r*SH[j][1] + Comp2Real[L0][i][j].i*SH[j][0]; 
		  }
		  AF[L0][i] = sum0 + sum1; 
		}              

	      }
	    }
	  }

	  /* Chi0 */  
	  i1 = -1;
	  for (L0=0; L0<=Spe_MaxL_Basis[wan]; L0++){
	    for (Mul0=0; Mul0<Spe_Num_Basis[wan][L0]; Mul0++){
	      for (M0=0; M0<=2*L0; M0++){
		i1++;
		Chi0[i1] = RF[L0][Mul0]*AF[L0][M0];
	      }
	    }
	  }
	  /* end of inlining of Get_Orbitals */

	}
	else{
          Get_Cnt_Orbitals(Mc_AN,x,y,z,Chi0);
	}

	for (i=0; i<NO0; i++){
	  Orbs_Grid[Mc_AN][Nc][i] = (Type_Orbs_Grid)Chi0[i];/* AITUNE */
	}

      } /* Nc */

      /* freeing of array */

      free(Chi0);

      for (i=0; i<(List_YOUSO[25]+1); i++){
	free(RF[i]);
      }
      free(RF);

      for (i=0; i<(List_YOUSO[25]+1); i++){
	free(AF[i]);
      }
      free(AF);

    } /* #pragma omp parallel */

    dtime(&Etime_atom);
    time_per_atom[Gc_AN] += Etime_atom - Stime_atom;
  }

  /****************************************************
     Calculate Orbs_Grid_FNAN
  ****************************************************/

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

    Gc_AN = M2G[Mc_AN];    

    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

      Gh_AN = natn[Gc_AN][h_AN];

      if (G2ID[Gh_AN]!=myid){

        Mh_AN = F_G2M[Gh_AN];
        Rnh = ncn[Gc_AN][h_AN];
        Hwan = WhatSpecies[Gh_AN];

        if (Cnt_kind==0)  NO1 = Spe_Total_NO[Hwan];
        else              NO1 = Spe_Total_CNO[Hwan];

#pragma omp parallel shared(List_YOUSO,Orbs_Grid_FNAN,NO1,Mh_AN,Hwan,Cnt_kind,Rnh,Gh_AN,Gxyz,atv,NumOLG,Mc_AN,h_AN,GListTAtoms1,GridListAtom,CellListAtom) private(OMPID,Nthrds,Nprocs,Nog,Nc,GNc,GRc,x,y,z,j)
        {

          double *Chi0;
	  double Cxyz0[4]; 
          double **RF;
          double **AF;
	  int i,L0,Mul0,M0,i1;
	  double S_coordinate[3];
	  double dum,dum1,dum2,dum3,dum4,a,b,c,d;
	  double siQ,coQ,siP,coP,Q,P,R;
	  double rm,df,sum0,sum1;
	  double SH[Supported_MaxL*2+1][2];
	  double dSHt[Supported_MaxL*2+1][2];
	  double dSHp[Supported_MaxL*2+1][2];

	  /* Radial */
	  int mp_min,mp_max,m,po,wan;
	  double h1,h2,h3,f1,f2,f3,f4;
	  double g1,g2,x1,x2,y1,y2,y12,y22,f;
          double r,r1,theta,phi,Min_r;

          /* allocation of arrays */

	  Chi0 = (double*)malloc(sizeof(double)*List_YOUSO[7]);

	  RF = (double**)malloc(sizeof(double*)*(List_YOUSO[25]+1));
	  for (i=0; i<(List_YOUSO[25]+1); i++){
	    RF[i] = (double*)malloc(sizeof(double)*List_YOUSO[24]);
	  }

	  AF = (double**)malloc(sizeof(double*)*(List_YOUSO[25]+1));
	  for (i=0; i<(List_YOUSO[25]+1); i++){
	    AF[i] = (double*)malloc(sizeof(double)*(2*(List_YOUSO[25]+1)+1));
	  }

	  /* get info. on OpenMP */ 

	  OMPID = omp_get_thread_num();
	  Nthrds = omp_get_num_threads();
	  Nprocs = omp_get_num_procs();

	  for (Nog=OMPID*NumOLG[Mc_AN][h_AN]/Nthrds; Nog<(OMPID+1)*NumOLG[Mc_AN][h_AN]/Nthrds; Nog++){

	    Nc = GListTAtoms1[Mc_AN][h_AN][Nog];
	    GNc = GridListAtom[Mc_AN][Nc];
	    GRc = CellListAtom[Mc_AN][Nc]; 

	    Get_Grid_XYZ(GNc,Cxyz0);

	    x = Cxyz0[1] + atv[GRc][1] - Gxyz[Gh_AN][1] - atv[Rnh][1];
	    y = Cxyz0[2] + atv[GRc][2] - Gxyz[Gh_AN][2] - atv[Rnh][2];
	    z = Cxyz0[3] + atv[GRc][3] - Gxyz[Gh_AN][3] - atv[Rnh][3];

	    if (Cnt_kind==0){

              /* Get_Orbitals(Hwan,x,y,z,Chi0); */
              /* start of inlining of Get_Orbitals */

              wan = Hwan; 

	      /* xyz2spherical(x,y,z,0.0,0.0,0.0,S_coordinate); */
              /* start of inlining of xyz2spherical */

	      Min_r = 10e-15;
	      dum = x*x + y*y; 
	      r = sqrt(dum + z*z);
	      r1 = sqrt(dum);

	      if (Min_r<=r){

		if (r<fabs(z))
		  dum1 = sgn(z)*1.0;
		else
		  dum1 = z/r;

		theta = acos(dum1);

		if (Min_r<=r1){
		  if (0.0<=x){

		    if (r1<fabs(y))
		      dum1 = sgn(y)*1.0;
		    else
		      dum1 = y/r1;        
  
		    phi = asin(dum1);
		  }
		  else{

		    if (r1<fabs(y))
		      dum1 = sgn(y)*1.0;
		    else
		      dum1 = y/r1;        

		    phi = PI - asin(dum1);
		  }
		}
		else{
		  phi = 0.0;
		}
	      }
	      else{
		theta = 0.5*PI;
		phi = 0.0;
	      }

	      R = r;
	      Q = theta;
	      P = phi;

              /* end of inlining of xyz2spherical */

	      po = 0;
	      mp_min = 0;
	      mp_max = Spe_Num_Mesh_PAO[wan] - 1;

	      if (Spe_PAO_RV[wan][Spe_Num_Mesh_PAO[wan]-1]<R){

		for (L0=0; L0<=Spe_MaxL_Basis[wan]; L0++){
		  for (Mul0=0; Mul0<Spe_Num_Basis[wan][L0]; Mul0++){
		    RF[L0][Mul0] = 0.0;
		  }
		}

		po = 1;
	      }

	      else if (R<Spe_PAO_RV[wan][0]){

		m = 4;
		rm = Spe_PAO_RV[wan][m];

		h1 = Spe_PAO_RV[wan][m-1] - Spe_PAO_RV[wan][m-2];
		h2 = Spe_PAO_RV[wan][m]   - Spe_PAO_RV[wan][m-1];
		h3 = Spe_PAO_RV[wan][m+1] - Spe_PAO_RV[wan][m];

		x1 = rm - Spe_PAO_RV[wan][m-1];
		x2 = rm - Spe_PAO_RV[wan][m];
		y1 = x1/h2;
		y2 = x2/h2;
		y12 = y1*y1;
		y22 = y2*y2;

		dum = h1 + h2;
		dum1 = h1/h2/dum;
		dum2 = h2/h1/dum;
		dum = h2 + h3;
		dum3 = h2/h3/dum;
		dum4 = h3/h2/dum;

		for (L0=0; L0<=Spe_MaxL_Basis[wan]; L0++){
		  for (Mul0=0; Mul0<Spe_Num_Basis[wan][L0]; Mul0++){

		    f1 = Spe_PAO_RWF[wan][L0][Mul0][m-2];
		    f2 = Spe_PAO_RWF[wan][L0][Mul0][m-1];
		    f3 = Spe_PAO_RWF[wan][L0][Mul0][m];
		    f4 = Spe_PAO_RWF[wan][L0][Mul0][m+1];

		    if (m==1){
		      h1 = -(h2+h3);
		      f1 = f4;
		    }
		    else if (m==(Spe_Num_Mesh_PAO[wan]-1)){
		      h3 = -(h1+h2);
		      f4 = f1;
		    }

		    dum = f3 - f2;
		    g1 = dum*dum1 + (f2-f1)*dum2;
		    g2 = (f4-f3)*dum3 + dum*dum4;

		    f =  y22*(3.0*f2 + h2*g1 + (2.0*f2 + h2*g1)*y2)
		      + y12*(3.0*f3 - h2*g2 - (2.0*f3 - h2*g2)*y1);

		    df = 2.0*y2/h2*(3.0*f2 + h2*g1 + (2.0*f2 + h2*g1)*y2)
		      + y22*(2.0*f2 + h2*g1)/h2
		      + 2.0*y1/h2*(3.0*f3 - h2*g2 - (2.0*f3 - h2*g2)*y1)
		      - y12*(2.0*f3 - h2*g2)/h2;

		    if (L0==0){
		      a = 0.0;
		      b = 0.5*df/rm;
		      c = 0.0;
		      d = f - b*rm*rm;
		    }

		    else if (L0==1){
		      a = (rm*df - f)/(2.0*rm*rm*rm);
		      b = 0.0;
		      c = df - 3.0*a*rm*rm;
		      d = 0.0;
		    }

		    else{
		      b = (3.0*f - rm*df)/(rm*rm);
		      a = (f - b*rm*rm)/(rm*rm*rm);
		      c = 0.0;
		      d = 0.0;
		    }

		    RF[L0][Mul0] = a*R*R*R + b*R*R + c*R + d;

		  }
		}

	      }

	      else{

		do{
		  m = (mp_min + mp_max)/2;
		  if (Spe_PAO_RV[wan][m]<R)
		    mp_min = m;
		  else 
		    mp_max = m;
		}
		while((mp_max-mp_min)!=1);
		m = mp_max;

		h1 = Spe_PAO_RV[wan][m-1] - Spe_PAO_RV[wan][m-2];
		h2 = Spe_PAO_RV[wan][m]   - Spe_PAO_RV[wan][m-1];
		h3 = Spe_PAO_RV[wan][m+1] - Spe_PAO_RV[wan][m];

		x1 = R - Spe_PAO_RV[wan][m-1];
		x2 = R - Spe_PAO_RV[wan][m];
		y1 = x1/h2;
		y2 = x2/h2;
		y12 = y1*y1;
		y22 = y2*y2;

		dum = h1 + h2;
		dum1 = h1/h2/dum;
		dum2 = h2/h1/dum;
		dum = h2 + h3;
		dum3 = h2/h3/dum;
		dum4 = h3/h2/dum;

		for (L0=0; L0<=Spe_MaxL_Basis[wan]; L0++){
		  for (Mul0=0; Mul0<Spe_Num_Basis[wan][L0]; Mul0++){

		    f1 = Spe_PAO_RWF[wan][L0][Mul0][m-2];
		    f2 = Spe_PAO_RWF[wan][L0][Mul0][m-1];
		    f3 = Spe_PAO_RWF[wan][L0][Mul0][m];
		    f4 = Spe_PAO_RWF[wan][L0][Mul0][m+1];

		    if (m==1){
		      h1 = -(h2+h3);
		      f1 = f4;
		    }
		    else if (m==(Spe_Num_Mesh_PAO[wan]-1)){
		      h3 = -(h1+h2);
		      f4 = f1;
		    }

		    dum = f3 - f2;
		    g1 = dum*dum1 + (f2-f1)*dum2;
		    g2 = (f4-f3)*dum3 + dum*dum4;

		    f =  y22*(3.0*f2 + h2*g1 + (2.0*f2 + h2*g1)*y2)
		      + y12*(3.0*f3 - h2*g2 - (2.0*f3 - h2*g2)*y1);

		    RF[L0][Mul0] = f;

		  }
		} 

	      }

	      if (po==0){

		/* Angular */
		siQ = sin(Q);
		coQ = cos(Q);
		siP = sin(P);
		coP = cos(P);

		for (L0=0; L0<=Spe_MaxL_Basis[wan]; L0++){

		  if (L0==0){
		    AF[0][0] = 0.282094791773878;
		  }
		  else if (L0==1){
		    dum = 0.48860251190292*siQ;
		    AF[1][0] = dum*coP;
		    AF[1][1] = dum*siP;
		    AF[1][2] = 0.48860251190292*coQ;
		  }
		  else if (L0==2){
		    dum1 = siQ*siQ;
		    dum2 = 1.09254843059208*siQ*coQ;
		    AF[2][0] = 0.94617469575756*coQ*coQ - 0.31539156525252;
		    AF[2][1] = 0.54627421529604*dum1*(1.0 - 2.0*siP*siP);
		    AF[2][2] = 1.09254843059208*dum1*siP*coP;
		    AF[2][3] = dum2*coP;
		    AF[2][4] = dum2*siP;
		  }

		  else if (L0==3){
		    AF[3][0] = 0.373176332590116*(5.0*coQ*coQ*coQ - 3.0*coQ);
		    AF[3][1] = 0.457045799464466*coP*siQ*(5.0*coQ*coQ - 1.0);
		    AF[3][2] = 0.457045799464466*siP*siQ*(5.0*coQ*coQ - 1.0);
		    AF[3][3] = 1.44530572132028*siQ*siQ*coQ*(coP*coP-siP*siP);
		    AF[3][4] = 2.89061144264055*siQ*siQ*coQ*siP*coP;
		    AF[3][5] = 0.590043589926644*siQ*siQ*siQ*(4.0*coP*coP*coP - 3.0*coP);
		    AF[3][6] = 0.590043589926644*siQ*siQ*siQ*(3.0*siP - 4.0*siP*siP*siP);
		  }

		  else if (4<=L0){

		    /* calculation of complex spherical harmonics functions */
		    for(m=-L0; m<=L0; m++){ 
		      ComplexSH(L0,m,Q,P,SH[L0+m],dSHt[L0+m],dSHp[L0+m]);
		    }

		    /* transformation of complex to real */
		    for (i=0; i<(L0*2+1); i++){

		      sum0 = 0.0;
		      sum1 = 0.0; 
		      for (j=0; j<(L0*2+1); j++){
			sum0 += Comp2Real[L0][i][j].r*SH[j][0] - Comp2Real[L0][i][j].i*SH[j][1]; 
			sum1 += Comp2Real[L0][i][j].r*SH[j][1] + Comp2Real[L0][i][j].i*SH[j][0]; 
		      }
		      AF[L0][i] = sum0 + sum1; 
		    }              

		  }
		}
	      }

	      /* Chi0 */  
	      i1 = -1;
	      for (L0=0; L0<=Spe_MaxL_Basis[wan]; L0++){
		for (Mul0=0; Mul0<Spe_Num_Basis[wan][L0]; Mul0++){
		  for (M0=0; M0<=2*L0; M0++){
		    i1++;
		    Chi0[i1] = RF[L0][Mul0]*AF[L0][M0];
		  }
		}
	      }

              /* end of inlining of Get_Orbitals */

	    } /* if (Cnt_kind==0) */

	    else{
              Get_Cnt_Orbitals(Mh_AN,x,y,z,Chi0);
	    }

	    for (j=0; j<NO1; j++){
	      Orbs_Grid_FNAN[Mc_AN][h_AN][Nog][j] = (Type_Orbs_Grid)Chi0[j];/* AITUNE */
	    }

	  } /* Nog */

          /* freeing of arrays */

	  free(Chi0);

	  for (i=0; i<(List_YOUSO[25]+1); i++){
	    free(RF[i]);
	  }
	  free(RF);

	  for (i=0; i<(List_YOUSO[25]+1); i++){
	    free(AF[i]);
	  }
	  free(AF);

        } 
      }
    }
  }

  /* time */
  dtime(&TEtime);
  time0 = TEtime - TStime;

  return time0;
}

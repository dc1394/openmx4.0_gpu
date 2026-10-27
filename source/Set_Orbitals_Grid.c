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
#include "openmx_env.h"
#include "openmx_orbitals.h"
#include "openmx_gpu_workers.h"
#include "openmx_gpu_views.h"
#include "openmx_gpu_task.h"
#include "openmx_host_memory.h"
#include "openmx_host_simd.h"
#include "openmx_core_orbitals_simd.h"
#include <stdint.h>
#include "mpi.h"
#include <omp.h>
#include <openacc.h>
#include "set_cuda_default_device_from_local_rank.h"
#include "orbs_grid_gpu.h"
#include "openmx_precision.h"

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

static int SOG_gpu_requested(int Cnt_kind)
{
  int w, L0;

  if (Cnt_kind != 0) return 0;
  if (!openmx_env_flag_at(OPENMX_ENV_ORBS_GRID_GPU, 1)) return 0;

  /* the staging buffers and the device stores are float */
  if (sizeof(Type_Orbs_Grid) != sizeof(float)) return 0;

  for (w = 0; w < SpeciesNum; w++) {
    if (SOG_L0MAX < Spe_MaxL_Basis[w]) return 0;
    for (L0 = 0; L0 <= Spe_MaxL_Basis[w]; L0++) {
      if (SOG_MULMAX < Spe_Num_Basis[w][L0]) return 0;
    }
    if (Spe_Num_Mesh_PAO[w] < 6) return 0;
  }

  return 1;
}

static int SOG_gpu_eligible(int Cnt_kind)
{
  return SOG_gpu_requested(Cnt_kind) && (openmx_gpu_worker_device()>=0 || gpu_rank_device_usable());
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

        h1 = (m == 1) ? rv[m - 1] - rv[m + 1] : rv[m - 1] - rv[m - 2];
        h2 = rv[m] - rv[m - 1];
        h3 = (m == mesh - 1) ? rv[m - 2] - rv[m] : rv[m + 1] - rv[m];

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

        h1 = (m == 1) ? rv[m - 1] - rv[m + 1] : rv[m - 1] - rv[m - 2];
        h2 = rv[m] - rv[m - 1];
        h3 = (m == mesh - 1) ? rv[m - 2] - rv[m] : rv[m + 1] - rv[m];

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
          double f1 = rw[m == 1 ? m + 1 : m - 2];
          double f2 = rw[m - 1];
          double f3 = rw[m];
          double f4 = rw[m == mesh - 1 ? m - 2 : m + 1];
          double g1, g2, f;

          rrow += (size_t)mesh;

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
static int Set_Orbitals_Grid_GPU_Part(const OpenMXAtomicViews *atomic, int first, int last, int myid)
{
  static _Thread_local int fail_notice_done = 0, active_notice_done = 0;
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


  /* ---- per-atom staging maxima ---- */

  npair_cap = 0;
  for (Mc_AN = first; Mc_AN < last; Mc_AN++) {
    size_t fpts = 0, fchi = 0;
    int npair = 0;

    Gc_AN = M2G[Mc_AN];
    Cwan = WhatSpecies[Gc_AN];
    NO0 = atomic->species.orbitals[Cwan];

    if (max_pts < (size_t)GridN_Atom[Gc_AN]) max_pts = (size_t)GridN_Atom[Gc_AN];
    cnt = (size_t)GridN_Atom[Gc_AN] * (size_t)NO0;
    if (max_chi < cnt) max_chi = cnt;

    for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
      Gh_AN = natn[Gc_AN][h_AN];
      if (G2ID[Gh_AN] != myid && 0 < NumOLG[Mc_AN][h_AN]) {
        NO1 = atomic->species.orbitals[WhatSpecies[Gh_AN]];
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
    sp_mesh_h[w] = atomic->species.pao_mesh[w];
    sp_maxl_h[w] = atomic->species.basis_lmax[w];
    for (L0 = 0; L0 <= SOG_L0MAX; L0++) {
      int nb = (L0 <= atomic->species.basis_lmax[w] ? atomic->species.basis[(size_t)(w)*atomic->species.angular_channels+(L0)] : 0);

      sp_nb_h[w * (SOG_L0MAX + 1) + L0] = nb;
      rwf_cnt += (size_t)nb * (size_t)atomic->species.pao_mesh[w];
    }
    rv_cnt += (size_t)atomic->species.pao_mesh[w];
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

    for (i = 0; i < atomic->species.pao_mesh[w]; i++) {
      pao_rv[rv_off_h[w] + (size_t)i] = openmx_radial_row2(&atomic->pao_radii,w)[i];
    }
    for (L0 = 0; L0 <= atomic->species.basis_lmax[w]; L0++) {
      for (Mul0 = 0; Mul0 < atomic->species.basis[(size_t)(w)*atomic->species.angular_channels+(L0)]; Mul0++) {
        for (i = 0; i < atomic->species.pao_mesh[w]; i++) {
          pao_rwf[rpos++] = openmx_radial_row4(&atomic->orbitals,w,L0,Mul0)[i];
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
    if (!fail_notice_done && 2 <= level_stdout) {
      fail_notice_done = 1;
      fprintf(stderr,
              "Set_Orbitals_Grid: %.1f MiB of device staging unavailable; trying distributed capacity recovery.\n",
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

    for (Mc_AN = first; Mc_AN < last; Mc_AN++) {

      int npts, npair;
      size_t fpts, fchi;

      dtime(&Stime_atom);

      Gc_AN = M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];
      NO0 = atomic->species.orbitals[Cwan];
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

#pragma omp parallel for if(!omp_in_parallel()) private(i)
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
          NO1 = atomic->species.orbitals[Hwan];

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

#pragma omp parallel for if(!omp_in_parallel()) private(j)
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
    if (openmx_env_flag_at(OPENMX_ENV_ORBS_GRID_GPU_VERBOSE, 0) && myid == Host_ID) {
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





typedef struct {
  OpenMXGpuWorkers workers;
  const OpenMXAtomicViews *atomic;
  int myid;
} SOG_TableWork;

static int SOG_table_worker(void *opaque, int worker, int phase)
{
  SOG_TableWork *work=(SOG_TableWork*)opaque;
  if (phase!=OPENMX_GPU_WORKER_EXECUTE) return 1;
  const int first=1+(int)openmx_gpu_worker_cut(work->workers.items,worker,work->workers.count);
  const int last=1+(int)openmx_gpu_worker_cut(work->workers.items,worker+1,work->workers.count);
  /* Disjoint atoms, with overwrite semantics. A failed staging claim
     leaves the original whole-table CPU overwrite as a valid fallback. */
  return Set_Orbitals_Grid_GPU_Part(work->atomic, first,last,work->myid);
}

/* the Core's host kernel reads the same point and frame records */
typedef OpenMXOrbitalsPoint SOG_CapacityPoint;
typedef OpenMXOrbitalsFrame SOG_CapacityParameters;
/* Host evaluation of a point range through the Core's Kokkos::simd kernel
   (OPENMX_HOST_SIMD selects the ISA; legacy keeps the loop of the kernel
   below, which also serves any range the kernel refuses). Threads take
   contiguous ranges of points. */
static int SOG_host_points(const SOG_CapacityParameters *p,const SOG_CapacityPoint *points,size_t n,
    const double *rv,const double *rwf,const double *translations,const size_t *rvo,const size_t *rwbo,
    const int *mesh,const int *maxl,const int *nb,float *out)
{
  const int mode=OpenMX_HostSimdMode();
  if(mode==OPENMX_HOST_SIMD_LEGACY)return 0;
  const OpenMXOrbitalsTables tables={rv,rwf,translations,rvo,rwbo,mesh,maxl,nb,SpeciesNum,SOG_L0MAX,SOG_MULMAX,TCpyCell+1};
  if(n<4096 || omp_in_parallel() || omp_get_max_threads()<=1)
    return openmx_core_orbitals_points(&tables,p,points,n,out,mode);
  int ok=1;
#pragma omp parallel reduction(&&:ok)
  {
    const int thread=omp_get_thread_num(),threads=omp_get_num_threads();
    const size_t begin=n*(size_t)thread/threads,end=n*(size_t)(thread+1)/threads;
    ok=ok && openmx_core_orbitals_points(&tables,p,points+begin,end-begin,out+begin*(size_t)p->max_no,mode);
  }
  return ok;
}
static void SOG_capacity_kernel(size_t first,size_t n,const void *parameters,void *const v[],int gpu)
{
  (void)first;const SOG_CapacityParameters *p=parameters;
  const int ng23=p->ng23,ng3=p->ng3,max_no=p->max_no;
  const double g11=p->frame[0],g12=p->frame[1],g13=p->frame[2];
  const double g21=p->frame[3],g22=p->frame[4],g23=p->frame[5];
  const double g31=p->frame[6],g32=p->frame[7],g33=p->frame[8];
  const double org1=p->origin[0],org2=p->origin[1],org3=p->origin[2];
  const SOG_CapacityPoint *points=v[0];
  const double *rv=v[1],*rwf=v[2],*translations=v[3];
  const size_t *rvo=v[4],*rwbo=v[5];const int *mesh=v[6],*maxl=v[7],*nb=v[8];
  float *out=v[9];
  if(!gpu && SOG_host_points(p,points,n,rv,rwf,translations,rvo,rwbo,mesh,maxl,nb,out))return;
#pragma acc parallel loop gang vector vector_length(128) deviceptr(points,rv,rwf,translations,rvo,rwbo,mesh,maxl,nb,out) if(gpu)
  for (size_t i=0;i<n;i++) {
    SOG_CapacityPoint q=points[i];
    for (int j=q.no;j<max_no;j++) out[i*max_no+j]=0;
    SOG_point_eval(q.wan,q.no,q.xyz[0],q.xyz[1],q.xyz[2],q.shift[0],q.shift[1],q.shift[2],q.grid,q.cell,
      translations,rv,rwf,rvo,rwbo,mesh,maxl,nb,ng23,ng3,
      g11,g12,g13,g21,g22,g23,g31,g32,g33,org1,org2,org3,out+i*max_no);
  }
}

/* Failed owner staging is retried as independent point tiles. The radial
   tables are small, immutable lookups; no complete rank orbital table is
   replicated on a helper GPU or sent to another node. */
/* Every orbital row of this rank straight from the Core's segment kernel,
   for a rank whose rows are pending (the device part was voted out by the
   preflight, the device is unusable, or the device part failed): one
   segment per atom grid and per overlap with an atom of another rank, cut
   into pieces for the threads, written into Orbs_Grid and Orbs_Grid_FNAN
   without point records, an output tile or collectives. 0 (the legacy
   loops selected or the kernel refused) leaves every row to the tiled
   capacity path, which overwrites them all. */
#define SOG_HOST_PIECE 16384
static int SOG_host_rows(const OpenMXAtomicViews *atomic,const SOG_CapacityParameters *p,const double *rv,const double *rwf,
    const double *translations,const size_t *rvo,const size_t *rwbo,const int *mesh,const int *maxl,const int *nb,int rank)
{
  const int mode=OpenMX_HostSimdMode();
  if(mode==OPENMX_HOST_SIMD_LEGACY)return 0;
  const OpenMXOrbitalsTables tables={rv,rwf,translations,rvo,rwbo,mesh,maxl,nb,SpeciesNum,SOG_L0MAX,SOG_MULMAX,TCpyCell+1};
  size_t pieces=0,total=0;
  for(int mc=1;mc<=Matomnum;mc++) {
    const int gc=M2G[mc];
    for(int h=-1;h<=FNAN[gc];h++) {
      if(h>=0 && G2ID[natn[gc][h]]==rank)continue;
      const size_t n=h<0?(size_t)GridN_Atom[gc]:(size_t)NumOLG[mc][h];
      pieces+=(n+SOG_HOST_PIECE-1)/SOG_HOST_PIECE;total+=n;
    }
  }
  if(!pieces)return 1;
  OpenMXOrbitalsSegment *seg=malloc(pieces*sizeof(*seg));
  if(!seg)return 0;
  size_t at=0;
  for(int mc=1;mc<=Matomnum;mc++) {
    const int gc=M2G[mc];
    for(int h=-1;h<=FNAN[gc];h++) {
      const int gh=h<0?gc:natn[gc][h];
      if(h>=0 && G2ID[gh]==rank)continue;
      const size_t n=h<0?(size_t)GridN_Atom[gc]:(size_t)NumOLG[mc][h];
      for(size_t first=0;first<n;first+=SOG_HOST_PIECE) {
        OpenMXOrbitalsSegment *e=seg+at++;
        e->wan=WhatSpecies[gh];e->no=atomic->species.orbitals[e->wan];
        for(int d=0;d<3;d++){e->xyz[d]=Gxyz[gh][d+1];e->shift[d]=h<0?0.0:atv[ncn[gc][h]][d+1];}
        e->count=n-first<SOG_HOST_PIECE?n-first:SOG_HOST_PIECE;
        if(h<0){e->index=NULL;e->grid=GridListAtom[mc]+first;e->cell=CellListAtom[mc]+first;e->rows=(float *const *)Orbs_Grid[mc]+first;}
        else{e->index=GListTAtoms1[mc][h]+first;e->grid=GridListAtom[mc];e->cell=CellListAtom[mc];e->rows=(float *const *)Orbs_Grid_FNAN[mc][h]+first;}
      }
    }
  }
  double begin,end;
  dtime(&begin);
  int ok=1;
  if(omp_in_parallel() || omp_get_max_threads()<=1 || pieces<2) ok=openmx_core_orbitals_segments(&tables,p,seg,pieces,mode);
  else {
    long i;
#pragma omp parallel for schedule(dynamic) reduction(&&:ok)
    for(i=0;i<(long)pieces;i++) if(ok) ok=openmx_core_orbitals_segments(&tables,p,seg+i,1,mode);
  }
  dtime(&end);
  free(seg);
  if(3<=level_stdout){
    printf("<Set_Orbitals_Grid> rank %d host rows: %zu points in %zu pieces, %.3f s%s\n",rank,total,pieces,end-begin,ok?"":" (refused)");
    fflush(stdout);
  }
  return ok;
}

static int SOG_capacity_recover(const OpenMXAtomicViews *atomic, int done,int rank)
{
  int pending=!done,any=0;MPI_Allreduce(&pending,&any,1,MPI_INT,MPI_MAX,mpi_comm_level1);
  if (!any) return done;
  if (pending && !gpu_rank_device_usable()) OpenMX_HostSimdReport("<Set_Orbitals_Grid> capacity points:");
  SOG_CapacityParameters p={0};p.ng23=Ngrid2*Ngrid3;p.ng3=Ngrid3;
  for (int i=0;i<3;i++) {p.origin[i]=Grid_Origin[i+1];for(int j=0;j<3;j++)p.frame[3*i+j]=gtv[i+1][j+1];}
  size_t rows=0,rvn=0,rwfn=0,tile=262144;
  int ready=1;
  for (int w=0;w<SpeciesNum;w++) if (p.max_no<atomic->species.orbitals[w]) p.max_no=atomic->species.orbitals[w];
  size_t *rvo=NULL,*rwbo=NULL;int *mesh=NULL,*maxl=NULL,*nb=NULL;
  double *rv=NULL,*rwf=NULL,*translations=NULL;
  SOG_CapacityPoint *points=NULL;Type_Orbs_Grid **destination=NULL;float *out=NULL;
  if (pending) {
    rvo=malloc((size_t)SpeciesNum*sizeof(size_t));rwbo=malloc((size_t)SpeciesNum*sizeof(size_t));
    mesh=malloc((size_t)SpeciesNum*sizeof(int));maxl=malloc((size_t)SpeciesNum*sizeof(int));
    nb=calloc((size_t)SpeciesNum*(SOG_L0MAX+1),sizeof(int));
    ready=rvo && rwbo && mesh && maxl && nb;
    for (int w=0;ready && w<SpeciesNum;w++) {
      rvo[w]=rvn;rwbo[w]=rwfn;mesh[w]=atomic->species.pao_mesh[w];maxl[w]=atomic->species.basis_lmax[w];rvn+=mesh[w];
      for(int l=0;l<=maxl[w];l++){nb[w*(SOG_L0MAX+1)+l]=atomic->species.basis[(size_t)(w)*atomic->species.angular_channels+(l)];rwfn+=(size_t)mesh[w]*atomic->species.basis[(size_t)(w)*atomic->species.angular_channels+(l)];}
    }
    if (ready) {
      rv=malloc(rvn*sizeof(double));rwf=malloc(rwfn*sizeof(double));
      translations=malloc(((size_t)TCpyCell+1)*3*sizeof(double));ready=rv && rwf && translations;
    }
    for(int w=0;ready && w<SpeciesNum;w++) {
      memcpy(rv+rvo[w],openmx_radial_row2(&atomic->pao_radii,w),(size_t)mesh[w]*sizeof(double));size_t at=rwbo[w];
      for(int l=0;l<=maxl[w];l++)for(int m=0;m<atomic->species.basis[(size_t)(w)*atomic->species.angular_channels+(l)];m++) {
        memcpy(rwf+at,openmx_radial_row4(&atomic->orbitals,w,l,m),(size_t)mesh[w]*sizeof(double));at+=mesh[w];
      }
    }
    if(ready)for(int r=0;r<=TCpyCell;r++)for(int i=0;i<3;i++)translations[3*(size_t)r+i]=atv[r][i+1];
    /* a pending rank evaluates its rows on the host SIMD path; this has no
       collectives, so the other ranks only meet it in the step count below.
       The capacity fault injection keeps exercising the tiled device path. */
    const int direct=ready && !openmx_gpu_capacity_test_recovery() &&
        SOG_host_rows(atomic,&p,rv,rwf,translations,rvo,rwbo,mesh,maxl,nb,rank);
    for(int mc=1;ready && !direct && mc<=Matomnum;mc++) {
      int gc=M2G[mc];rows+=(size_t)GridN_Atom[gc];
      for(int h=0;h<=FNAN[gc];h++)if(G2ID[natn[gc][h]]!=rank)rows+=(size_t)NumOLG[mc][h];
    }
    if (tile>rows) tile=rows;
    size_t per=sizeof(*points)+sizeof(*destination)+(size_t)p.max_no*sizeof(float);
    unsigned long long room=openmx_host_mem_available();int sharing=openmx_gpu_local_size_noncollective();if(sharing<1)sharing=1;
    if(room && tile>room/(4ULL*sharing)/per)tile=(size_t)(room/(4ULL*sharing)/per);
    if(rows && !tile)ready=0;
    if(ready && tile){points=malloc(tile*sizeof(*points));destination=malloc(tile*sizeof(*destination));out=malloc(tile*(size_t)p.max_no*sizeof(float));ready=points && destination && out;}
  }
  if(!ready)rows=0;
  uint64_t steps=tile?(rows/tile+(rows%tile!=0)):0,max_steps=0;
  MPI_Allreduce(&steps,&max_steps,1,MPI_UINT64_T,MPI_MAX,mpi_comm_level1);
  for(uint64_t step=0;step<max_steps;step++) {
    size_t begin=step<steps?(size_t)step*tile:rows,end=begin;
    if(step<steps){end=rows-begin<tile?rows:begin+tile;}
    size_t cursor=0;
    for(int mc=1;begin<end && mc<=Matomnum;mc++) {
      int gc=M2G[mc];
      for(int h=-1;h<=FNAN[gc];h++) {
        int gh=h<0?gc:natn[gc][h];if(h>=0 && G2ID[gh]==rank)continue;
        size_t length=h<0?(size_t)GridN_Atom[gc]:(size_t)NumOLG[mc][h],lo=cursor,hi=cursor+length;
        cursor=hi;if(lo<begin)lo=begin;if(hi>end)hi=end;
        for(size_t row=lo;row<hi;row++) {
          size_t k=row-(cursor-length),i=row-begin;int nc=h<0?(int)k:GListTAtoms1[mc][h][k];
          SOG_CapacityPoint *q=points+i;q->wan=WhatSpecies[gh];q->no=atomic->species.orbitals[q->wan];
          q->grid=GridListAtom[mc][nc];q->cell=CellListAtom[mc][nc];
          for(int d=0;d<3;d++){q->xyz[d]=Gxyz[gh][d+1];q->shift[d]=h<0?0:atv[ncn[gc][h]][d+1];}
          destination[i]=h<0?Orbs_Grid[mc][k]:Orbs_Grid_FNAN[mc][h][k];
        }
      }
    }
    OpenMXGpuTask t={0};t.count=end-begin;t.inputs=9;t.outputs=1;t.parameters=&p;t.parameter_bytes=sizeof(p);t.kernel=SOG_capacity_kernel;
    t.view[0]=(OpenMXGpuTaskView){points,0,sizeof(*points)};
    t.view[1]=(OpenMXGpuTaskView){rv,rvn*sizeof(double),0};t.view[2]=(OpenMXGpuTaskView){rwf,rwfn*sizeof(double),0};
    t.view[3]=(OpenMXGpuTaskView){translations,translations?((size_t)TCpyCell+1)*3*sizeof(double):0,0};
    t.view[4]=(OpenMXGpuTaskView){rvo,rvo?(size_t)SpeciesNum*sizeof(size_t):0,0};
    t.view[5]=(OpenMXGpuTaskView){rwbo,rwbo?(size_t)SpeciesNum*sizeof(size_t):0,0};
    t.view[6]=(OpenMXGpuTaskView){mesh,mesh?(size_t)SpeciesNum*sizeof(int):0,0};
    t.view[7]=(OpenMXGpuTaskView){maxl,maxl?(size_t)SpeciesNum*sizeof(int):0,0};
    t.view[8]=(OpenMXGpuTaskView){nb,nb?(size_t)SpeciesNum*(SOG_L0MAX+1)*sizeof(int):0,0};
    t.view[9]=(OpenMXGpuTaskView){out,0,(size_t)p.max_no*sizeof(float)};
    openmx_gpu_task_execute(mpi_comm_level1,&t,1,"orbital-capacity");
    for(size_t i=0;i<t.count;i++)for(int j=0;j<points[i].no;j++)destination[i][j]=out[i*(size_t)p.max_no+j];
  }
  free(rvo);free(rwbo);free(mesh);free(maxl);free(nb);free(rv);free(rwf);free(translations);free(points);free(destination);free(out);
  return done || ready;
}

/* The device working set of Set_Orbitals_Grid_GPU_Part on atoms [first,last)
   of this rank: its single arena claim (the per-atom staging maxima and the
   PAO tables; keep in sync with the layout there). */
static size_t SOG_device_need(const OpenMXAtomicViews *atomic,int first,int last,int myid)
{
  size_t max_pts=0,max_chi=0,max_fpts=0,max_fchi=0,rv_cnt=0,rwf_cnt=0,pos=0,cnt;
  int npair_cap=0;
  for (int Mc_AN=first;Mc_AN<last;Mc_AN++) {
    size_t fpts=0,fchi=0;
    int npair=0;
    const int Gc_AN=M2G[Mc_AN];
    const int NO0=atomic->species.orbitals[WhatSpecies[Gc_AN]];
    if (max_pts<(size_t)GridN_Atom[Gc_AN]) max_pts=(size_t)GridN_Atom[Gc_AN];
    cnt=(size_t)GridN_Atom[Gc_AN]*(size_t)NO0;
    if (max_chi<cnt) max_chi=cnt;
    for (int h_AN=0;h_AN<=FNAN[Gc_AN];h_AN++) {
      const int Gh_AN=natn[Gc_AN][h_AN];
      if (G2ID[Gh_AN]!=myid && 0<NumOLG[Mc_AN][h_AN]) {
        fpts+=(size_t)NumOLG[Mc_AN][h_AN];
        fchi+=(size_t)NumOLG[Mc_AN][h_AN]*(size_t)atomic->species.orbitals[WhatSpecies[Gh_AN]];
        npair++;
      }
    }
    if (max_fpts<fpts) max_fpts=fpts;
    if (max_fchi<fchi) max_fchi=fchi;
    if (npair_cap<npair) npair_cap=npair;
  }
  if (npair_cap<1) npair_cap=1;
  for (int w=0;w<SpeciesNum;w++) {
    for (int L0=0;L0<=SOG_L0MAX;L0++) {
      const int nb=L0<=atomic->species.basis_lmax[w] ? atomic->species.basis[(size_t)w*atomic->species.angular_channels+(size_t)L0] : 0;
      rwf_cnt+=(size_t)nb*(size_t)atomic->species.pao_mesh[w];
    }
    rv_cnt+=(size_t)atomic->species.pao_mesh[w];
  }
  (void)SOG_arena_off(&pos,sizeof(double)*(rv_cnt==0 ? 1 : rv_cnt));
  (void)SOG_arena_off(&pos,sizeof(double)*(rwf_cnt==0 ? 1 : rwf_cnt));
  (void)SOG_arena_off(&pos,sizeof(size_t)*(size_t)(SpeciesNum+1));
  (void)SOG_arena_off(&pos,sizeof(size_t)*(size_t)(SpeciesNum+1));
  (void)SOG_arena_off(&pos,sizeof(int)*(size_t)SpeciesNum);
  (void)SOG_arena_off(&pos,sizeof(int)*(size_t)SpeciesNum);
  (void)SOG_arena_off(&pos,sizeof(int)*(size_t)SpeciesNum*(SOG_L0MAX+1));
  (void)SOG_arena_off(&pos,sizeof(double)*((size_t)TCpyCell+1)*3);
  (void)SOG_arena_off(&pos,sizeof(int)*(max_pts==0 ? 1 : max_pts));
  (void)SOG_arena_off(&pos,sizeof(int)*(max_pts==0 ? 1 : max_pts));
  (void)SOG_arena_off(&pos,sizeof(int)*(max_fpts==0 ? 1 : max_fpts));
  (void)SOG_arena_off(&pos,sizeof(int)*(max_fpts==0 ? 1 : max_fpts));
  (void)SOG_arena_off(&pos,sizeof(SOG_GpuPair)*(size_t)npair_cap);
  cnt=max_chi<max_fchi ? max_fchi : max_chi;
  (void)SOG_arena_off(&pos,sizeof(float)*(cnt==0 ? 1 : cnt));
  return pos;
}

static int Set_Orbitals_Grid_GPU(const OpenMXAtomicViews *atomic)
{
  SOG_TableWork work;
  work.atomic=atomic;
  MPI_Comm_rank(mpi_comm_level1,&work.myid);
  openmx_gpu_workers_local(&work.workers,(size_t)Matomnum);
  int done=0;
  /* Device preflight vote, collective over mpi_comm_level1 (every rank gets
     here once per call): when some rank's device cannot hold the stage, all
     ranks skip the device part and SOG_capacity_recover writes their rows on
     the host SIMD path. A device that holds it keeps the device part, and
     under the legacy policy a usable device always does (old routes). */
  const int host=OpenMX_HostSimdStageOnHost("Set_Orbitals_Grid",
      SOG_device_need(atomic,1,Matomnum+1,work.myid),OpenMX_HostSimdDeviceBudget(0));
  if ((!host || OpenMX_HostSimdMode()==OPENMX_HOST_SIMD_LEGACY) && !openmx_gpu_capacity_test_recovery() && gpu_rank_device_usable()) {
    if (work.workers.count<=1) done=Set_Orbitals_Grid_GPU_Part(atomic, 1,Matomnum+1,work.myid);
    else done=openmx_gpu_workers_run(&work.workers,SOG_table_worker,&work,"orbital-tables");
  }
  return SOG_capacity_recover(atomic, done,work.myid);
}

/* ---------------------------------------------------------------------
   On-the-fly orbital tiles for the grid integrals (orbs_grid_gpu.h): the
   species' PAO tables, the cell translations and the grid frame stay on
   the device; Set_Hamiltonian evaluates the orbitals of each batch of
   atom pairs from them instead of holding the multi-GiB tables.
   --------------------------------------------------------------------- */
static SOG_DeviceTables SOG_devices[OPENMX_GPU_WORKERS_MAX];
static int SOG_fp32_modes[OPENMX_GPU_WORKERS_MAX];
static int SOG_cache_device(void)
{
  int device=-1;
  if (cudaGetDevice(&device)!=cudaSuccess || device<0 || device>=OPENMX_GPU_WORKERS_MAX) {
    fprintf(stderr,"SOG: invalid CUDA device for orbital-table cache.\n");
    exit(EXIT_FAILURE);
  }
  return device;
}
#define SOG_dev SOG_devices[SOG_cache_device()]
#define SOG_dev_fp32_enabled SOG_fp32_modes[SOG_cache_device()]


/* OPENMX_ORBS_EVAL_PRECISION: "df" or "fp64" for the on-the-fly
   tiles.  Native FP64 avoids the error-free FP32 transformations on A100
   and other devices with a fast FP64 pipeline.  Match the grid-integral
   policy: use DF only when FP64 throughput is at least eight times lower.
   scf.fp32.enable off overrides this policy.  Otherwise an explicit
   environment setting wins; failed probes conservatively use FP64. */
static int SOG_eval_double_float(void)
{
  const char *value;

  if (!openmx_fp32_enabled()) return 0;
  value = openmx_env_at(OPENMX_ENV_ORBS_EVAL_PRECISION);
  if (value != NULL && value[0] != '\0') {
    if (strcmp(value, "fp64") == 0 || strcmp(value, "FP64") == 0 || strcmp(value, "64") == 0 ||
        strcmp(value, "double") == 0) return 0;
    return 1;
  }
  {
    int device = 0, perf_ratio = 0;
    return cudaGetDevice(&device) == cudaSuccess &&
           cudaDeviceGetAttribute(&perf_ratio, cudaDevAttrSingleToDoublePrecisionPerfRatio, device) == cudaSuccess &&
           perf_ratio >= 8;
  }
}

void SOG_Device_Release(void)
{
  if (SOG_dev.arena != NULL) {
    acc_free(SOG_dev.arena);
    if (cudaDeviceSynchronize() == cudaSuccess) acc_clear_freelists();
  }
  memset(&SOG_dev, 0, sizeof(SOG_dev));
  SOG_dev_fp32_enabled = -1;
}

/* Release every cache without initializing a GPU on a CPU-only run. */
void SOG_Device_ReleaseAll(void)
{
  int saved_cuda=-1,saved_acc=-1,have_saved=0;
  acc_device_t saved_type=acc_device_none;
  for (int device=0;device<OPENMX_GPU_WORKERS_MAX;device++) {
    if (SOG_devices[device].arena!=NULL) {
      if (!have_saved) {
        (void)cudaGetDevice(&saved_cuda);
        saved_type=acc_get_device_type();saved_acc=acc_get_device_num(saved_type);
        have_saved=1;
      }
      if (cudaSetDevice(device)!=cudaSuccess) {
        fprintf(stderr,"SOG: cannot bind device %d to release its orbital cache.\n",device);
        exit(EXIT_FAILURE);
      }
      acc_set_device_num(device,acc_device_nvidia);
      SOG_Device_Release();
    } else {
      memset(SOG_devices+device,0,sizeof(SOG_devices[device]));
      SOG_fp32_modes[device]=-1;
    }
  }
  if (have_saved) {
    acc_set_device_num(saved_acc,saved_type);
    if (saved_cuda>=0) (void)cudaSetDevice(saved_cuda);
  }
}

int SOG_Device_Prepare(const OpenMXAtomicViews *atomic)
{
  int w, L0, Mul0, i;
  size_t rv_cnt = 0, rwf_cnt = 0, atv_rows, r, pos = 0;
  size_t *rv_off_h = NULL, *rwf_base_h = NULL;
  int *sp_mesh_h = NULL, *sp_maxl_h = NULL, *sp_nb_h = NULL;
  double *pao_rv = NULL, *pao_rwf = NULL, *atv_flat = NULL;
  float *rwd_h = NULL, *mcd_h = NULL, *spl_h = NULL;
  size_t o_rv, o_rwf, o_rvo, o_rwb, o_spm, o_spx, o_spb, o_atv, o_rwd, o_mcd, o_spl;
  unsigned char *arena;

  if (SOG_dev.ready) {
    if (SOG_dev_fp32_enabled == openmx_fp32_enabled()) return 1;
    SOG_Device_Release();
  }
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
    sp_mesh_h[w] = atomic->species.pao_mesh[w];
    sp_maxl_h[w] = atomic->species.basis_lmax[w];
    for (L0 = 0; L0 <= SOG_L0MAX; L0++) {
      int nb = (L0 <= atomic->species.basis_lmax[w] ? atomic->species.basis[(size_t)(w)*atomic->species.angular_channels+(L0)] : 0);
      sp_nb_h[w * (SOG_L0MAX + 1) + L0] = nb;
      rwf_cnt += (size_t)nb * (size_t)atomic->species.pao_mesh[w];
    }
    rv_cnt += (size_t)atomic->species.pao_mesh[w];
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
    for (i = 0; i < atomic->species.pao_mesh[w]; i++) pao_rv[rv_off_h[w] + (size_t)i] = openmx_radial_row2(&atomic->pao_radii,w)[i];
    for (L0 = 0; L0 <= atomic->species.basis_lmax[w]; L0++)
      for (Mul0 = 0; Mul0 < atomic->species.basis[(size_t)(w)*atomic->species.angular_channels+(L0)]; Mul0++)
        for (i = 0; i < atomic->species.pao_mesh[w]; i++) pao_rwf[rpos++] = openmx_radial_row4(&atomic->orbitals,w,L0,Mul0)[i];
  }
  for (r = 0; r < atv_rows; r++) {
    atv_flat[3 * r + 0] = atv[r][1];
    atv_flat[3 * r + 1] = atv[r][2];
    atv_flat[3 * r + 2] = atv[r][3];
  }
  /* Native FP64 never reads the DF mesh/spline representation.  Avoid
     constructing, transferring and retaining that duplicate on A100. */
  if (SOG_dev.eval_df) {
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
      const int mesh = atomic->species.pao_mesh[w];
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
  }
  o_rv = SOG_arena_off(&pos, sizeof(double) * (rv_cnt == 0 ? 1 : rv_cnt));
  o_rwf = SOG_arena_off(&pos, sizeof(double) * (rwf_cnt == 0 ? 1 : rwf_cnt));
  o_rvo = SOG_arena_off(&pos, sizeof(size_t) * (size_t)(SpeciesNum + 1));
  o_rwb = SOG_arena_off(&pos, sizeof(size_t) * (size_t)(SpeciesNum + 1));
  o_spm = SOG_arena_off(&pos, sizeof(int) * (size_t)SpeciesNum);
  o_spx = SOG_arena_off(&pos, sizeof(int) * (size_t)SpeciesNum);
  o_spb = SOG_arena_off(&pos, sizeof(int) * (size_t)SpeciesNum * (SOG_L0MAX + 1));
  o_atv = SOG_arena_off(&pos, sizeof(double) * atv_rows * 3);
  o_rwd = o_mcd = o_spl = 0;
  if (SOG_dev.eval_df) {
    o_rwd = SOG_arena_off(&pos, sizeof(float) * 2 * (rwf_cnt == 0 ? 1 : rwf_cnt));
    o_mcd = SOG_arena_off(&pos, sizeof(float) * SOG_MCD_STRIDE * (rv_cnt == 0 ? 1 : rv_cnt));
    o_spl = SOG_arena_off(&pos, sizeof(float) * 2 * (size_t)SpeciesNum);
  }
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
  if (SOG_dev.eval_df) {
    acc_memcpy_to_device(arena + o_rwd, rwd_h, sizeof(float) * 2 * (rwf_cnt == 0 ? 1 : rwf_cnt));
    acc_memcpy_to_device(arena + o_mcd, mcd_h, sizeof(float) * SOG_MCD_STRIDE * (rv_cnt == 0 ? 1 : rv_cnt));
    acc_memcpy_to_device(arena + o_spl, spl_h, sizeof(float) * 2 * (size_t)SpeciesNum);
  }
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
  SOG_dev.rwd = SOG_dev.eval_df ? (const float*)(const void*)(arena + o_rwd) : NULL;
  SOG_dev.mcd = SOG_dev.eval_df ? (const float*)(const void*)(arena + o_mcd) : NULL;
  SOG_dev.spl = SOG_dev.eval_df ? (const float*)(const void*)(arena + o_spl) : NULL;
  SOG_dev.ng23 = Ngrid2 * Ngrid3;
  SOG_dev.ng3 = Ngrid3;
  for (i = 0; i < 3; i++) {
    SOG_dev.g[i][0] = gtv[i + 1][1];
    SOG_dev.g[i][1] = gtv[i + 1][2];
    SOG_dev.g[i][2] = gtv[i + 1][3];
    SOG_dev.org[i] = Grid_Origin[i + 1];
  }
  SOG_dev_fp32_enabled = openmx_fp32_enabled();
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
  return SOG_dev.eval_df && openmx_fp32_enabled() ? "double-float" : "FP64";
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
static void SOG_Device_EvalPairTiles_Part(const SOG_DeviceTables *t,int pair_count, const SOG_GpuPair *pairs, const int *pair_NOLG,
                              const size_t *nolg_off, const size_t *out_off, const size_t *gla_off,
                              const int *nolg_Nc, const int *gla, const int *cla, float *out)
{
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
  if (t->eval_df && openmx_fp32_enabled()) {
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
static void SOG_Device_EvalAtomTile_Part(const SOG_DeviceTables *t,const SOG_GpuPair *pair, int npts, const int *gla, const int *cla, float *out)
{
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
  if (t->eval_df && openmx_fp32_enabled()) {
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

/* One compact pair chunk, distributed across the device by grid point.
   The descriptor is on the host, all arrays on the device; nolg_Nc maps
   chunk rows back to the central atom's sphere.  PairTiles assigns a gang
   per pair and would leave nearly the whole GPU idle for a single chunk. */
static void SOG_Device_EvalPairChunk_Part(const SOG_DeviceTables *t,const SOG_GpuPair *pair, int npts, const int *nolg_Nc,
                              const int *gla, const int *cla, float *out)
{
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
  if (t->eval_df && openmx_fp32_enabled()) {
    const float *rwd_all = t->rwd, *mcd_all = t->mcd, *spl = t->spl;
#pragma acc parallel loop gang vector vector_length(128) \
  deviceptr(nolg_Nc, gla, cla, out, atvf, rv_all, rwf_all, rwd_all, mcd_all, spl, rv_off, rwf_base, sp_mesh, sp_maxl, sp_nb)
    for (ip = 0; ip < npts; ip++) {
      SOG_point_eval_df(pr.wan, pr.no, pr.gx, pr.gy, pr.gz, pr.ax, pr.ay, pr.az, gla[nolg_Nc[ip]], cla[nolg_Nc[ip]],
                        atvf, rv_all, rwf_all, rwd_all, mcd_all, spl, rv_off, rwf_base, sp_mesh, sp_maxl, sp_nb,
                        ng23, ng3, g11, g12, g13, g21, g22, g23, g31, g32, g33, org1, org2, org3,
                        out + (size_t)ip * (size_t)pr.no);
    }
    return;
  }
#pragma acc parallel loop gang vector vector_length(128) \
  deviceptr(nolg_Nc, gla, cla, out, atvf, rv_all, rwf_all, rv_off, rwf_base, sp_mesh, sp_maxl, sp_nb)
  for (ip = 0; ip < npts; ip++) {
    SOG_point_eval(pr.wan, pr.no, pr.gx, pr.gy, pr.gz, pr.ax, pr.ay, pr.az, gla[nolg_Nc[ip]], cla[nolg_Nc[ip]],
                   atvf, rv_all, rwf_all, rv_off, rwf_base, sp_mesh, sp_maxl, sp_nb,
                   ng23, ng3, g11, g12, g13, g21, g22, g23, g31, g32, g33, org1, org2, org3,
                   out + (size_t)ip * (size_t)pr.no);
  }
}

typedef struct {
  OpenMXGpuWorkers workers;
  SOG_DeviceTables table;
  int kind,n;
  const SOG_GpuPair *pairs;
  SOG_GpuPair pair,*host_pairs;
  const int *count,*nc,*gla,*cla;
  const size_t *noff,*ooff,*goff;
  int *host_count,*host_nc;
  size_t *host_noff,*host_ooff,*host_goff,nc_extent;
  float *out;
  struct {
    OpenMXGpuReadView view[10];
    SOG_DeviceTables table;
    size_t first,last,gfirst,glast,obase,oend;
    size_t *ooff,*goff;
    float *out;
  } device[OPENMX_GPU_WORKERS_MAX];
} SOG_TileWork;

static int SOG_tile_worker(void *opaque,int worker,int phase)
{
  SOG_TileWork *w=(SOG_TileWork*)opaque;
  const size_t first=openmx_gpu_worker_cut(w->workers.items,worker,w->workers.count);
  const size_t last=openmx_gpu_worker_cut(w->workers.items,worker+1,w->workers.count),n=last-first;
  if (phase==OPENMX_GPU_WORKER_PREPARE) {
    w->device[worker].first=first;w->device[worker].last=last;
    if (!openmx_gpu_read_view(&w->device[worker].view[0],w->table.arena,w->table.bytes)) return 0;
    SOG_DeviceTables *t=&w->device[worker].table;
    *t=w->table;t->arena=(unsigned char*)w->device[worker].view[0].pointer;
#define SOG_TILE_FIELD(field,type) t->field=w->table.field?(const type*)(t->arena+((const unsigned char*)w->table.field-w->table.arena)):NULL
    SOG_TILE_FIELD(rv,double);SOG_TILE_FIELD(rwf,double);SOG_TILE_FIELD(atv,double);
    SOG_TILE_FIELD(rvo,size_t);SOG_TILE_FIELD(rwb,size_t);
    SOG_TILE_FIELD(spm,int);SOG_TILE_FIELD(spx,int);SOG_TILE_FIELD(spb,int);
    SOG_TILE_FIELD(rwd,float);SOG_TILE_FIELD(mcd,float);SOG_TILE_FIELD(spl,float);
#undef SOG_TILE_FIELD
    size_t gfirst=0,glast=0,obase=0,oend=0;
    if (w->kind==0) {
      gfirst=(size_t)-1;obase=(size_t)-1;
      size_t *offsets=(size_t*)malloc(2*n*sizeof(size_t));
      if (!offsets) return 0;
      for (size_t p=first;p<last;p++) {
        if (w->host_goff[p]<gfirst) gfirst=w->host_goff[p];
        if (w->host_ooff[p]<obase) obase=w->host_ooff[p];
        size_t end=w->host_ooff[p]+(size_t)w->host_count[p]*w->host_pairs[p].no;
        if (end>oend) oend=end;
        for (int q=0;q<w->host_count[p];q++) {
          size_t gend=w->host_goff[p]+(size_t)w->host_nc[w->host_noff[p]+q]+1;
          if (gend>glast) glast=gend;
        }
      }
      if (glast<gfirst) glast=gfirst;
      for (size_t p=first;p<last;p++) {
        offsets[p-first]=w->host_ooff[p]-obase;
        offsets[n+p-first]=w->host_goff[p]-gfirst;
      }
      int ok=cudaMalloc((void**)&w->device[worker].ooff,2*n*sizeof(size_t))==cudaSuccess;
      if (ok) {
        w->device[worker].goff=w->device[worker].ooff+n;
        ok=cudaMemcpy(w->device[worker].ooff,offsets,2*n*sizeof(size_t),cudaMemcpyHostToDevice)==cudaSuccess;
      }
      free(offsets);
      if (!ok ||
          !openmx_gpu_read_view(&w->device[worker].view[1],w->pairs+first,n*sizeof(SOG_GpuPair)) ||
          !openmx_gpu_read_view(&w->device[worker].view[2],w->count+first,n*sizeof(int)) ||
          !openmx_gpu_read_view(&w->device[worker].view[3],w->noff+first,n*sizeof(size_t)) ||
          !openmx_gpu_read_view(&w->device[worker].view[4],w->nc,w->nc_extent*sizeof(int))) return 0;
    } else {
      obase=first*(size_t)w->pair.no;oend=last*(size_t)w->pair.no;
      if (w->kind==1) {gfirst=first;glast=last;}
      else {
        for (size_t p=first;p<last;p++) if ((size_t)w->host_nc[p]+1>glast) glast=(size_t)w->host_nc[p]+1;
        if (!openmx_gpu_read_view(&w->device[worker].view[4],w->nc+first,n*sizeof(int))) return 0;
      }
    }
    w->device[worker].obase=obase;w->device[worker].oend=oend;
    w->device[worker].gfirst=gfirst;w->device[worker].glast=glast;
    if (!openmx_gpu_read_view(&w->device[worker].view[5],w->gla+gfirst,(glast-gfirst)*sizeof(int)) ||
        !openmx_gpu_read_view(&w->device[worker].view[6],w->cla+gfirst,(glast-gfirst)*sizeof(int))) return 0;
    return cudaMalloc((void**)&w->device[worker].out,((oend>obase)?oend-obase:1)*sizeof(float))==cudaSuccess;
  }
  if (phase==OPENMX_GPU_WORKER_EXECUTE) {
    const SOG_DeviceTables *t=&w->device[worker].table;
    OpenMXGpuReadView *v=w->device[worker].view;
    if (w->kind==0) SOG_Device_EvalPairTiles_Part(t,(int)n,
      (const SOG_GpuPair*)v[1].pointer,(const int*)v[2].pointer,(const size_t*)v[3].pointer,
      w->device[worker].ooff,w->device[worker].goff,(const int*)v[4].pointer,
      (const int*)v[5].pointer,(const int*)v[6].pointer,w->device[worker].out);
    else if (w->kind==1) SOG_Device_EvalAtomTile_Part(t,&w->pair,(int)n,
      (const int*)v[5].pointer,(const int*)v[6].pointer,w->device[worker].out);
    else SOG_Device_EvalPairChunk_Part(t,&w->pair,(int)n,(const int*)v[4].pointer,
      (const int*)v[5].pointer,(const int*)v[6].pointer,w->device[worker].out);
  } else if (phase==OPENMX_GPU_WORKER_COMMIT) {
    if (w->kind==0) {
      for (size_t p=first;p<last;p++)
        if (!openmx_gpu_copy_device(w->out+w->host_ooff[p],
            w->device[worker].out+(w->host_ooff[p]-w->device[worker].obase),
            (size_t)w->host_count[p]*w->host_pairs[p].no*sizeof(float))) return 0;
    } else return openmx_gpu_copy_device(w->out+w->device[worker].obase,w->device[worker].out,
      (w->device[worker].oend-w->device[worker].obase)*sizeof(float));
  } else if (phase==OPENMX_GPU_WORKER_RELEASE) {
    for (int i=0;i<10;i++) openmx_gpu_read_view_release(&w->device[worker].view[i]);
    if (w->device[worker].out) (void)cudaFree(w->device[worker].out);
    if (w->device[worker].ooff) (void)cudaFree(w->device[worker].ooff);
  }
  return 1;
}

static int SOG_tiles_distribute(SOG_TileWork *w)
{
  if (w->n<=0 || !SOG_dev.ready) return 0;
  openmx_gpu_workers_local(&w->workers,(size_t)w->n);
  if (w->workers.count<=1) return 0;
  w->table=SOG_dev;
  int ok=cudaDeviceSynchronize()==cudaSuccess;
  if (w->kind==0 && ok) {
    size_t n=(size_t)w->n;
    w->host_pairs=(SOG_GpuPair*)malloc(n*sizeof(SOG_GpuPair));
    w->host_count=(int*)malloc(n*sizeof(int));
    w->host_noff=(size_t*)malloc(3*n*sizeof(size_t));
    ok=w->host_pairs && w->host_count && w->host_noff;
    if (ok) {
      w->host_ooff=w->host_noff+n;w->host_goff=w->host_noff+2*n;
      ok=cudaMemcpy(w->host_pairs,w->pairs,n*sizeof(SOG_GpuPair),cudaMemcpyDeviceToHost)==cudaSuccess &&
         cudaMemcpy(w->host_count,w->count,n*sizeof(int),cudaMemcpyDeviceToHost)==cudaSuccess &&
         cudaMemcpy(w->host_noff,w->noff,n*sizeof(size_t),cudaMemcpyDeviceToHost)==cudaSuccess &&
         cudaMemcpy(w->host_ooff,w->ooff,n*sizeof(size_t),cudaMemcpyDeviceToHost)==cudaSuccess &&
         cudaMemcpy(w->host_goff,w->goff,n*sizeof(size_t),cudaMemcpyDeviceToHost)==cudaSuccess;
      for (size_t p=0;ok && p<n;p++) {
        if (w->host_count[p]<0 || w->host_pairs[p].no<0) {ok=0;break;}
        size_t end=w->host_noff[p]+(size_t)w->host_count[p];
        if (end>w->nc_extent) w->nc_extent=end;
      }
    }
  } else if (w->kind==2) w->nc_extent=(size_t)w->n;
  if (ok && w->nc_extent) {
    w->host_nc=(int*)malloc(w->nc_extent*sizeof(int));
    ok=w->host_nc && cudaMemcpy(w->host_nc,w->nc,w->nc_extent*sizeof(int),cudaMemcpyDeviceToHost)==cudaSuccess;
    /* Compact pair layouts may leave padding between valid Nc slices.
       Inspect only entries referenced by a pair, never unused padding. */
    if (w->kind==0) {
      for (int p=0;ok && p<w->n;p++)
        for (int k=0;k<w->host_count[p];k++)
          if (w->host_nc[w->host_noff[p]+(size_t)k]<0) {ok=0;break;}
    } else {
      for (size_t p=0;ok && p<w->nc_extent;p++) if (w->host_nc[p]<0) ok=0;
    }
  }
  if (ok) ok=openmx_gpu_workers_run(&w->workers,SOG_tile_worker,w,
    w->kind==0?"orbital-pair-tiles":w->kind==1?"orbital-atom-tile":"orbital-pair-chunk");
  free(w->host_nc);free(w->host_noff);free(w->host_count);free(w->host_pairs);
  return ok;
}

void SOG_Device_EvalPairTiles(int n,const SOG_GpuPair *pairs,const int *count,
    const size_t *noff,const size_t *ooff,const size_t *goff,const int *nc,
    const int *gla,const int *cla,float *out)
{
  SOG_TileWork w={0};w.n=n;w.pairs=pairs;w.count=count;w.noff=noff;w.ooff=ooff;w.goff=goff;
  w.nc=nc;w.gla=gla;w.cla=cla;w.out=out;
  if (!SOG_tiles_distribute(&w)) SOG_Device_EvalPairTiles_Part(&SOG_dev,n,pairs,count,noff,ooff,goff,nc,gla,cla,out);
}

void SOG_Device_EvalAtomTile(const SOG_GpuPair *pair,int n,const int *gla,const int *cla,float *out)
{
  SOG_TileWork w={0};w.kind=1;w.n=n;w.pair=*pair;w.gla=gla;w.cla=cla;w.out=out;
  if (!SOG_tiles_distribute(&w)) SOG_Device_EvalAtomTile_Part(&SOG_dev,pair,n,gla,cla,out);
}

void SOG_Device_EvalPairChunk(const SOG_GpuPair *pair,int n,const int *nc,
    const int *gla,const int *cla,float *out)
{
  SOG_TileWork w={0};w.kind=2;w.n=n;w.pair=*pair;w.nc=nc;w.gla=gla;w.cla=cla;w.out=out;
  if (!SOG_tiles_distribute(&w)) SOG_Device_EvalPairChunk_Part(&SOG_dev,pair,n,nc,gla,cla,out);
}

double Set_Orbitals_Grid(OpenMXDensityGpuState *density_state, const OpenMXAtomicViews *atomic, int Cnt_kind, const OpenMXLocalBasisViews *local_basis)
{
  SOG_Device_ReleaseAll();   /* the grid frame may have changed */
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
  Set_Density_Grid_GPU_Invalidate(density_state);
  Set_Hamiltonian_Invalidate_OpenACC_MatrixElements_Cache();

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);
  
  dtime(&TStime);

  /* device fast path; on any ineligibility or allocation failure the
     untouched host loops below produce the same values */
  int gpu_requested=SOG_gpu_requested(Cnt_kind);
  MPI_Allreduce(MPI_IN_PLACE,&gpu_requested,1,MPI_INT,MPI_MIN,mpi_comm_level1);
  if (gpu_requested && Set_Orbitals_Grid_GPU(atomic)) {
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

    if (Cnt_kind==0)  NO0 = atomic->species.orbitals[Cwan];
    else              NO0 = atomic->species.contracted_orbitals[Cwan];

#pragma omp parallel shared(Comp2Real,List_YOUSO,Orbs_Grid,Cnt_kind,Gxyz,atv,CellListAtom,GridListAtom,GridN_Atom,Gc_AN,Cwan,Mc_AN,NO0,atomic) private(OMPID,Nthrds,Nprocs,Nc,GNc,GRc,Cxyz,x,y,z,dx,dy,dz,i,j)
    {
      double *Chi0;
      double Cxyz0[4]; 
      double *RF;
      double *AF;
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

      OpenMXOrbitalBuffers orbital_buffers;
      OpenMXOrbitalWorkspace *orbital_workspace=OpenMX_OrbitalWorkspace(
          (int)atomic->orbitals.extent[1]-1,(int)atomic->orbitals.extent[2],0,&orbital_buffers);
      RF=orbital_buffers.RF;
      AF=orbital_buffers.AF;

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
	  mp_max = atomic->species.pao_mesh[wan] - 1;

	  if (openmx_radial_row2(&atomic->pao_radii,wan)[atomic->species.pao_mesh[wan]-1]<R){

	    for (L0=0; L0<=atomic->species.basis_lmax[wan]; L0++){
	      for (Mul0=0; Mul0<atomic->species.basis[(size_t)(wan)*atomic->species.angular_channels+(L0)]; Mul0++){
		RF[(size_t)(L0)*orbital_buffers.multiplicity+(Mul0)] = 0.0;
	      }
	    }

	    po = 1;
	  }

	  else if (R<openmx_radial_row2(&atomic->pao_radii,wan)[0]){

	    m = 4;
	    rm = openmx_radial_row2(&atomic->pao_radii,wan)[m];

	    /* Use reflected endpoint samples before forming the spline weights. */
	    h1 = (m==1) ? openmx_radial_row2(&atomic->pao_radii,wan)[m-1] - openmx_radial_row2(&atomic->pao_radii,wan)[m+1]
	      : openmx_radial_row2(&atomic->pao_radii,wan)[m-1] - openmx_radial_row2(&atomic->pao_radii,wan)[m-2];
	    h2 = openmx_radial_row2(&atomic->pao_radii,wan)[m]   - openmx_radial_row2(&atomic->pao_radii,wan)[m-1];
	    h3 = (m==atomic->species.pao_mesh[wan]-1) ? openmx_radial_row2(&atomic->pao_radii,wan)[m-2] - openmx_radial_row2(&atomic->pao_radii,wan)[m]
	      : openmx_radial_row2(&atomic->pao_radii,wan)[m+1] - openmx_radial_row2(&atomic->pao_radii,wan)[m];

	    x1 = rm - openmx_radial_row2(&atomic->pao_radii,wan)[m-1];
	    x2 = rm - openmx_radial_row2(&atomic->pao_radii,wan)[m];
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

	    for (L0=0; L0<=atomic->species.basis_lmax[wan]; L0++){
	      for (Mul0=0; Mul0<atomic->species.basis[(size_t)(wan)*atomic->species.angular_channels+(L0)]; Mul0++){

		f1 = openmx_radial_row4(&atomic->orbitals,wan,L0,Mul0)[(m==1) ? m+1 : m-2];
		f2 = openmx_radial_row4(&atomic->orbitals,wan,L0,Mul0)[m-1];
		f3 = openmx_radial_row4(&atomic->orbitals,wan,L0,Mul0)[m];
		f4 = openmx_radial_row4(&atomic->orbitals,wan,L0,Mul0)[(m==atomic->species.pao_mesh[wan]-1) ? m-2 : m+1];

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

		RF[(size_t)(L0)*orbital_buffers.multiplicity+(Mul0)] = a*R*R*R + b*R*R + c*R + d;

	      }
	    }

	  }

	  else{

	    do{
	      m = (mp_min + mp_max)/2;
	      if (openmx_radial_row2(&atomic->pao_radii,wan)[m]<R)
		mp_min = m;
	      else 
		mp_max = m;
	    }
	    while((mp_max-mp_min)!=1);
	    m = mp_max;

	    h1 = (m==1) ? openmx_radial_row2(&atomic->pao_radii,wan)[m-1] - openmx_radial_row2(&atomic->pao_radii,wan)[m+1]
	      : openmx_radial_row2(&atomic->pao_radii,wan)[m-1] - openmx_radial_row2(&atomic->pao_radii,wan)[m-2];
	    h2 = openmx_radial_row2(&atomic->pao_radii,wan)[m]   - openmx_radial_row2(&atomic->pao_radii,wan)[m-1];
	    h3 = (m==atomic->species.pao_mesh[wan]-1) ? openmx_radial_row2(&atomic->pao_radii,wan)[m-2] - openmx_radial_row2(&atomic->pao_radii,wan)[m]
	      : openmx_radial_row2(&atomic->pao_radii,wan)[m+1] - openmx_radial_row2(&atomic->pao_radii,wan)[m];

	    x1 = R - openmx_radial_row2(&atomic->pao_radii,wan)[m-1];
	    x2 = R - openmx_radial_row2(&atomic->pao_radii,wan)[m];
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

	    for (L0=0; L0<=atomic->species.basis_lmax[wan]; L0++){
	      for (Mul0=0; Mul0<atomic->species.basis[(size_t)(wan)*atomic->species.angular_channels+(L0)]; Mul0++){

		f1 = openmx_radial_row4(&atomic->orbitals,wan,L0,Mul0)[(m==1) ? m+1 : m-2];
		f2 = openmx_radial_row4(&atomic->orbitals,wan,L0,Mul0)[m-1];
		f3 = openmx_radial_row4(&atomic->orbitals,wan,L0,Mul0)[m];
		f4 = openmx_radial_row4(&atomic->orbitals,wan,L0,Mul0)[(m==atomic->species.pao_mesh[wan]-1) ? m-2 : m+1];

		dum = f3 - f2;
		g1 = dum*dum1 + (f2-f1)*dum2;
		g2 = (f4-f3)*dum3 + dum*dum4;

		f =  y22*(3.0*f2 + h2*g1 + (2.0*f2 + h2*g1)*y2)
		  + y12*(3.0*f3 - h2*g2 - (2.0*f3 - h2*g2)*y1);

		RF[(size_t)(L0)*orbital_buffers.multiplicity+(Mul0)] = f;

	      }
	    } 

	  }

	  if (po==0){

	    /* Angular */
	    siQ = sin(Q);
	    coQ = cos(Q);
	    siP = sin(P);
	    coP = cos(P);
	    if (atomic->species.basis_lmax[wan] >= 4)
	      openmx_orbital_polar(x, y, z, R, &siQ, &coQ, &siP, &coP, &P);

	    for (L0=0; L0<=atomic->species.basis_lmax[wan]; L0++){

	      if (L0==0){
		AF[(size_t)(0)*orbital_buffers.magnetic_capacity+(0)] = 0.282094791773878;
	      }
	      else if (L0==1){
		dum = 0.48860251190292*siQ;
		AF[(size_t)(1)*orbital_buffers.magnetic_capacity+(0)] = dum*coP;
		AF[(size_t)(1)*orbital_buffers.magnetic_capacity+(1)] = dum*siP;
		AF[(size_t)(1)*orbital_buffers.magnetic_capacity+(2)] = 0.48860251190292*coQ;
	      }
	      else if (L0==2){
		dum1 = siQ*siQ;
		dum2 = 1.09254843059208*siQ*coQ;
		AF[(size_t)(2)*orbital_buffers.magnetic_capacity+(0)] = 0.94617469575756*coQ*coQ - 0.31539156525252;
		AF[(size_t)(2)*orbital_buffers.magnetic_capacity+(1)] = 0.54627421529604*dum1*(1.0 - 2.0*siP*siP);
		AF[(size_t)(2)*orbital_buffers.magnetic_capacity+(2)] = 1.09254843059208*dum1*siP*coP;
		AF[(size_t)(2)*orbital_buffers.magnetic_capacity+(3)] = dum2*coP;
		AF[(size_t)(2)*orbital_buffers.magnetic_capacity+(4)] = dum2*siP;
	      }

	      else if (L0==3){
		AF[(size_t)(3)*orbital_buffers.magnetic_capacity+(0)] = 0.373176332590116*(5.0*coQ*coQ*coQ - 3.0*coQ);
		AF[(size_t)(3)*orbital_buffers.magnetic_capacity+(1)] = 0.457045799464466*coP*siQ*(5.0*coQ*coQ - 1.0);
		AF[(size_t)(3)*orbital_buffers.magnetic_capacity+(2)] = 0.457045799464466*siP*siQ*(5.0*coQ*coQ - 1.0);
		AF[(size_t)(3)*orbital_buffers.magnetic_capacity+(3)] = 1.44530572132028*siQ*siQ*coQ*(coP*coP-siP*siP);
		AF[(size_t)(3)*orbital_buffers.magnetic_capacity+(4)] = 2.89061144264055*siQ*siQ*coQ*siP*coP;
		AF[(size_t)(3)*orbital_buffers.magnetic_capacity+(5)] = 0.590043589926644*siQ*siQ*siQ*(4.0*coP*coP*coP - 3.0*coP);
		AF[(size_t)(3)*orbital_buffers.magnetic_capacity+(6)] = 0.590043589926644*siQ*siQ*siQ*(3.0*siP - 4.0*siP*siP*siP);
	      }

	      else if (4<=L0){

		/* calculation of complex spherical harmonics functions */
		for(m=-L0; m<=L0; m++){ 
		  openmx_orbital_complex_sh(L0, m, siQ, coQ, P,SH[L0+m],dSHt[L0+m],dSHp[L0+m]);
		}

		/* transformation of complex to real */
		for (i=0; i<(L0*2+1); i++){

		  sum0 = 0.0;
		  sum1 = 0.0; 
		  for (j=0; j<(L0*2+1); j++){
		    sum0 += Comp2Real[L0][i][j].r*SH[j][0] - Comp2Real[L0][i][j].i*SH[j][1]; 
		    sum1 += Comp2Real[L0][i][j].r*SH[j][1] + Comp2Real[L0][i][j].i*SH[j][0]; 
		  }
		  AF[(size_t)(L0)*orbital_buffers.magnetic_capacity+(i)] = sum0 + sum1;
		}              

	      }
	    }
	  }

	  /* Chi0 */  
	  i1 = -1;
	  for (L0=0; L0<=atomic->species.basis_lmax[wan]; L0++){
	    for (Mul0=0; Mul0<atomic->species.basis[(size_t)(wan)*atomic->species.angular_channels+(L0)]; Mul0++){
	      for (M0=0; M0<=2*L0; M0++){
		i1++;
		Chi0[i1] = (po==0) ? RF[(size_t)(L0)*orbital_buffers.multiplicity+(Mul0)]*AF[(size_t)(L0)*orbital_buffers.magnetic_capacity+(M0)] : 0.0;
	      }
	    }
	  }
	  /* end of inlining of Get_Orbitals */

	}
	else{
          Get_Cnt_Orbitals(atomic, Mc_AN,x,y,z,Chi0, local_basis);
	}

	for (i=0; i<NO0; i++){
	  Orbs_Grid[Mc_AN][Nc][i] = (Type_Orbs_Grid)Chi0[i];/* AITUNE */
	}

      } /* Nc */

      /* freeing of array */

      free(Chi0);

      openmx_core_orbital_destroy(orbital_workspace);

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

        if (Cnt_kind==0)  NO1 = atomic->species.orbitals[Hwan];
        else              NO1 = atomic->species.contracted_orbitals[Hwan];

#pragma omp parallel shared(List_YOUSO,Orbs_Grid_FNAN,NO1,Mh_AN,Hwan,Cnt_kind,Rnh,Gh_AN,Gxyz,atv,NumOLG,Mc_AN,h_AN,GListTAtoms1,GridListAtom,CellListAtom) private(OMPID,Nthrds,Nprocs,Nog,Nc,GNc,GRc,x,y,z,j)
        {

          double *Chi0;
	  double Cxyz0[4]; 
          double *RF;
          double *AF;
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

	  OpenMXOrbitalBuffers orbital_buffers;
      OpenMXOrbitalWorkspace *orbital_workspace=OpenMX_OrbitalWorkspace(
          (int)atomic->orbitals.extent[1]-1,(int)atomic->orbitals.extent[2],0,&orbital_buffers);
      RF=orbital_buffers.RF;
      AF=orbital_buffers.AF;

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
	      mp_max = atomic->species.pao_mesh[wan] - 1;

	      if (openmx_radial_row2(&atomic->pao_radii,wan)[atomic->species.pao_mesh[wan]-1]<R){

		for (L0=0; L0<=atomic->species.basis_lmax[wan]; L0++){
		  for (Mul0=0; Mul0<atomic->species.basis[(size_t)(wan)*atomic->species.angular_channels+(L0)]; Mul0++){
		    RF[(size_t)(L0)*orbital_buffers.multiplicity+(Mul0)] = 0.0;
		  }
		}

		po = 1;
	      }

	      else if (R<openmx_radial_row2(&atomic->pao_radii,wan)[0]){

		m = 4;
		rm = openmx_radial_row2(&atomic->pao_radii,wan)[m];

		h1 = (m==1) ? openmx_radial_row2(&atomic->pao_radii,wan)[m-1] - openmx_radial_row2(&atomic->pao_radii,wan)[m+1]
		  : openmx_radial_row2(&atomic->pao_radii,wan)[m-1] - openmx_radial_row2(&atomic->pao_radii,wan)[m-2];
		h2 = openmx_radial_row2(&atomic->pao_radii,wan)[m]   - openmx_radial_row2(&atomic->pao_radii,wan)[m-1];
		h3 = (m==atomic->species.pao_mesh[wan]-1) ? openmx_radial_row2(&atomic->pao_radii,wan)[m-2] - openmx_radial_row2(&atomic->pao_radii,wan)[m]
		  : openmx_radial_row2(&atomic->pao_radii,wan)[m+1] - openmx_radial_row2(&atomic->pao_radii,wan)[m];

		x1 = rm - openmx_radial_row2(&atomic->pao_radii,wan)[m-1];
		x2 = rm - openmx_radial_row2(&atomic->pao_radii,wan)[m];
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

		for (L0=0; L0<=atomic->species.basis_lmax[wan]; L0++){
		  for (Mul0=0; Mul0<atomic->species.basis[(size_t)(wan)*atomic->species.angular_channels+(L0)]; Mul0++){

		    f1 = openmx_radial_row4(&atomic->orbitals,wan,L0,Mul0)[(m==1) ? m+1 : m-2];
		    f2 = openmx_radial_row4(&atomic->orbitals,wan,L0,Mul0)[m-1];
		    f3 = openmx_radial_row4(&atomic->orbitals,wan,L0,Mul0)[m];
		    f4 = openmx_radial_row4(&atomic->orbitals,wan,L0,Mul0)[(m==atomic->species.pao_mesh[wan]-1) ? m-2 : m+1];

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

		    RF[(size_t)(L0)*orbital_buffers.multiplicity+(Mul0)] = a*R*R*R + b*R*R + c*R + d;

		  }
		}

	      }

	      else{

		do{
		  m = (mp_min + mp_max)/2;
		  if (openmx_radial_row2(&atomic->pao_radii,wan)[m]<R)
		    mp_min = m;
		  else 
		    mp_max = m;
		}
		while((mp_max-mp_min)!=1);
		m = mp_max;

		h1 = (m==1) ? openmx_radial_row2(&atomic->pao_radii,wan)[m-1] - openmx_radial_row2(&atomic->pao_radii,wan)[m+1]
		  : openmx_radial_row2(&atomic->pao_radii,wan)[m-1] - openmx_radial_row2(&atomic->pao_radii,wan)[m-2];
		h2 = openmx_radial_row2(&atomic->pao_radii,wan)[m]   - openmx_radial_row2(&atomic->pao_radii,wan)[m-1];
		h3 = (m==atomic->species.pao_mesh[wan]-1) ? openmx_radial_row2(&atomic->pao_radii,wan)[m-2] - openmx_radial_row2(&atomic->pao_radii,wan)[m]
		  : openmx_radial_row2(&atomic->pao_radii,wan)[m+1] - openmx_radial_row2(&atomic->pao_radii,wan)[m];

		x1 = R - openmx_radial_row2(&atomic->pao_radii,wan)[m-1];
		x2 = R - openmx_radial_row2(&atomic->pao_radii,wan)[m];
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

		for (L0=0; L0<=atomic->species.basis_lmax[wan]; L0++){
		  for (Mul0=0; Mul0<atomic->species.basis[(size_t)(wan)*atomic->species.angular_channels+(L0)]; Mul0++){

		    f1 = openmx_radial_row4(&atomic->orbitals,wan,L0,Mul0)[(m==1) ? m+1 : m-2];
		    f2 = openmx_radial_row4(&atomic->orbitals,wan,L0,Mul0)[m-1];
		    f3 = openmx_radial_row4(&atomic->orbitals,wan,L0,Mul0)[m];
		    f4 = openmx_radial_row4(&atomic->orbitals,wan,L0,Mul0)[(m==atomic->species.pao_mesh[wan]-1) ? m-2 : m+1];

		    dum = f3 - f2;
		    g1 = dum*dum1 + (f2-f1)*dum2;
		    g2 = (f4-f3)*dum3 + dum*dum4;

		    f =  y22*(3.0*f2 + h2*g1 + (2.0*f2 + h2*g1)*y2)
		      + y12*(3.0*f3 - h2*g2 - (2.0*f3 - h2*g2)*y1);

		    RF[(size_t)(L0)*orbital_buffers.multiplicity+(Mul0)] = f;

		  }
		} 

	      }

	      if (po==0){

		/* Angular */
		siQ = sin(Q);
		coQ = cos(Q);
		siP = sin(P);
		coP = cos(P);
		if (atomic->species.basis_lmax[wan] >= 4)
		  openmx_orbital_polar(x, y, z, R, &siQ, &coQ, &siP, &coP, &P);

		for (L0=0; L0<=atomic->species.basis_lmax[wan]; L0++){

		  if (L0==0){
		    AF[(size_t)(0)*orbital_buffers.magnetic_capacity+(0)] = 0.282094791773878;
		  }
		  else if (L0==1){
		    dum = 0.48860251190292*siQ;
		    AF[(size_t)(1)*orbital_buffers.magnetic_capacity+(0)] = dum*coP;
		    AF[(size_t)(1)*orbital_buffers.magnetic_capacity+(1)] = dum*siP;
		    AF[(size_t)(1)*orbital_buffers.magnetic_capacity+(2)] = 0.48860251190292*coQ;
		  }
		  else if (L0==2){
		    dum1 = siQ*siQ;
		    dum2 = 1.09254843059208*siQ*coQ;
		    AF[(size_t)(2)*orbital_buffers.magnetic_capacity+(0)] = 0.94617469575756*coQ*coQ - 0.31539156525252;
		    AF[(size_t)(2)*orbital_buffers.magnetic_capacity+(1)] = 0.54627421529604*dum1*(1.0 - 2.0*siP*siP);
		    AF[(size_t)(2)*orbital_buffers.magnetic_capacity+(2)] = 1.09254843059208*dum1*siP*coP;
		    AF[(size_t)(2)*orbital_buffers.magnetic_capacity+(3)] = dum2*coP;
		    AF[(size_t)(2)*orbital_buffers.magnetic_capacity+(4)] = dum2*siP;
		  }

		  else if (L0==3){
		    AF[(size_t)(3)*orbital_buffers.magnetic_capacity+(0)] = 0.373176332590116*(5.0*coQ*coQ*coQ - 3.0*coQ);
		    AF[(size_t)(3)*orbital_buffers.magnetic_capacity+(1)] = 0.457045799464466*coP*siQ*(5.0*coQ*coQ - 1.0);
		    AF[(size_t)(3)*orbital_buffers.magnetic_capacity+(2)] = 0.457045799464466*siP*siQ*(5.0*coQ*coQ - 1.0);
		    AF[(size_t)(3)*orbital_buffers.magnetic_capacity+(3)] = 1.44530572132028*siQ*siQ*coQ*(coP*coP-siP*siP);
		    AF[(size_t)(3)*orbital_buffers.magnetic_capacity+(4)] = 2.89061144264055*siQ*siQ*coQ*siP*coP;
		    AF[(size_t)(3)*orbital_buffers.magnetic_capacity+(5)] = 0.590043589926644*siQ*siQ*siQ*(4.0*coP*coP*coP - 3.0*coP);
		    AF[(size_t)(3)*orbital_buffers.magnetic_capacity+(6)] = 0.590043589926644*siQ*siQ*siQ*(3.0*siP - 4.0*siP*siP*siP);
		  }

		  else if (4<=L0){

		    /* calculation of complex spherical harmonics functions */
		    for(m=-L0; m<=L0; m++){ 
		      openmx_orbital_complex_sh(L0, m, siQ, coQ, P,SH[L0+m],dSHt[L0+m],dSHp[L0+m]);
		    }

		    /* transformation of complex to real */
		    for (i=0; i<(L0*2+1); i++){

		      sum0 = 0.0;
		      sum1 = 0.0; 
		      for (j=0; j<(L0*2+1); j++){
			sum0 += Comp2Real[L0][i][j].r*SH[j][0] - Comp2Real[L0][i][j].i*SH[j][1]; 
			sum1 += Comp2Real[L0][i][j].r*SH[j][1] + Comp2Real[L0][i][j].i*SH[j][0]; 
		      }
		      AF[(size_t)(L0)*orbital_buffers.magnetic_capacity+(i)] = sum0 + sum1;
		    }              

		  }
		}
	      }

	      /* Chi0 */  
	      i1 = -1;
	      for (L0=0; L0<=atomic->species.basis_lmax[wan]; L0++){
		for (Mul0=0; Mul0<atomic->species.basis[(size_t)(wan)*atomic->species.angular_channels+(L0)]; Mul0++){
		  for (M0=0; M0<=2*L0; M0++){
		    i1++;
		    Chi0[i1] = (po==0) ? RF[(size_t)(L0)*orbital_buffers.multiplicity+(Mul0)]*AF[(size_t)(L0)*orbital_buffers.magnetic_capacity+(M0)] : 0.0;
		  }
		}
	      }

              /* end of inlining of Get_Orbitals */

	    } /* if (Cnt_kind==0) */

	    else{
              Get_Cnt_Orbitals(atomic, Mh_AN,x,y,z,Chi0, local_basis);
	    }

	    for (j=0; j<NO1; j++){
	      Orbs_Grid_FNAN[Mc_AN][h_AN][Nog][j] = (Type_Orbs_Grid)Chi0[j];/* AITUNE */
	    }

	  } /* Nog */

          /* freeing of arrays */

	  free(Chi0);

	  openmx_core_orbital_destroy(orbital_workspace);

        } 
      }
    }
  }

  /* time */
  dtime(&TEtime);
  time0 = TEtime - TStime;

  return time0;
}

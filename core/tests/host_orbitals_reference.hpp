#ifndef OPENMX_HOST_ORBITALS_REFERENCE_HPP
#define OPENMX_HOST_ORBITALS_REFERENCE_HPP
// Verbatim copy (2026-10-25) of SOG_point_eval from source/Set_Orbitals_Grid.c,
// the scalar device routine the orbitals family reproduces bit for bit. Only the
// name and linkage changed; keep in sync with the source when it changes.
#include <cmath>
#include <cstddef>
#define SOG_L0MAX  3
#define SOG_MULMAX 8
namespace openmx::test {
using std::sqrt; using std::fabs; using std::log; using std::acos; using std::asin; using std::sin; using std::cos;
inline void sog_point_eval(int wan, int no, double gx, double gy, double gz,
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

// Transcription (2026-10-25) of Get_dOrbitals_work from source/Get_dOrbitals.c
// (also inlined in Force3) on the flat species tables of the orbitals family,
// with xyz2spherical and sgn of openmx_common.c and OpenMX's PI literal; the
// L0 >= 4 complex-harmonics branch is not covered (the kernel refuses it).
// out[i*3 + c] = dChi[1 + c][i]. Keep in sync with the source when it changes.
inline double openmx_sgn_reference(double nu) { return nu<0.0?-1.0:1.0; }
inline void xyz2spherical_reference(double x,double y,double z,double S_coordinate[3]) {
  const double PI=3.1415926535897932384626;
  double dx,dy,dz,r,r1,theta,phi,dum,dum1,Min_r;
  Min_r = 10e-15;
  dx = x - 0.0;  dy = y - 0.0;  dz = z - 0.0;
  dum = dx*dx + dy*dy;
  r = sqrt(dum + dz*dz);
  r1 = sqrt(dum);
  if (Min_r<=r){
    if (r<fabs(dz)) dum1 = openmx_sgn_reference(dz)*1.0; else dum1 = dz/r;
    theta = acos(dum1);
    if (Min_r<=r1){
      if (0.0<=dx){
        if (r1<fabs(dy)) dum1 = openmx_sgn_reference(dy)*1.0; else dum1 = dy/r1;
        phi = asin(dum1);
      }
      else{
        if (r1<fabs(dy)) dum1 = openmx_sgn_reference(dy)*1.0; else dum1 = dy/r1;
        phi = PI - asin(dum1);
      }
    }
    else{ phi = 0.0; }
  }
  else{ theta = 0.5*PI; phi = 0.0; }
  S_coordinate[0] = r;  S_coordinate[1] = theta;  S_coordinate[2] = phi;
}
inline void get_dorbitals_reference(int maxl,const int *nb,int mesh,const double *rv,const double *rwf_rows,
                                    double x,double y,double z,double *out) {
  int i1, po, L0, Mul0, M0, mp_min, mp_max, m;
  double dum, dum1, dum2, dum3, dum4;
  double siQ, coQ, siP, coP, a, b, c;
  double rm, d, R, Q, P, Rmin;
  double S_coordinate[3];
  double RF[SOG_L0MAX+1][SOG_MULMAX], dRF[SOG_L0MAX+1][SOG_MULMAX];
  double AF[SOG_L0MAX+1][2*SOG_L0MAX+1], dAFQ[SOG_L0MAX+1][2*SOG_L0MAX+1], dAFP[SOG_L0MAX+1][2*SOG_L0MAX+1];
  double h1, h2, h3, f1, f2, f3, f4;
  double g1, g2, x1, x2, y1, y2, y12, y22, f, df;
  double dRx, dRy, dRz, dQx, dQy, dQz, dPx, dPy, dPz;
  double dChiR, dChiQ, dChiP;
  Rmin = 10e-14;
  xyz2spherical_reference(x, y, z, S_coordinate);
  R = S_coordinate[0]; Q = S_coordinate[1]; P = S_coordinate[2];
  if (R < Rmin) {
    x = x + Rmin; y = y + Rmin; z = z + Rmin;
    xyz2spherical_reference(x, y, z, S_coordinate);
    R = S_coordinate[0]; Q = S_coordinate[1]; P = S_coordinate[2];
  }
  po = 0; mp_min = 0; mp_max = mesh - 1;
  if (rv[mesh - 1] < R) {
    for (L0 = 0; L0 <= maxl; L0++) for (Mul0 = 0; Mul0 < nb[L0]; Mul0++) { RF[L0][Mul0] = 0.0; dRF[L0][Mul0] = 0.0; }
    po = 1;
  }
  else if (R < rv[0]) {
    m = 4; rm = rv[m];
    h1 = (m==1) ? rv[m-1] - rv[m+1] : rv[m-1] - rv[m-2];
    h2 = rv[m] - rv[m - 1];
    h3 = (m==mesh-1) ? rv[m-2] - rv[m] : rv[m+1] - rv[m];
    x1 = rm - rv[m - 1]; x2 = rm - rv[m];
    y1 = x1 / h2; y2 = x2 / h2; y12 = y1*y1; y22 = y2*y2;
    dum = h1 + h2; dum1 = h1 / h2 / dum; dum2 = h2 / h1 / dum;
    dum = h2 + h3; dum3 = h2 / h3 / dum; dum4 = h3 / h2 / dum;
    const double *rw = rwf_rows;
    for (L0 = 0; L0 <= maxl; L0++) {
      for (Mul0 = 0; Mul0 < nb[L0]; Mul0++, rw += mesh) {
        f1 = rw[(m==1) ? m+1 : m-2]; f2 = rw[m - 1]; f3 = rw[m]; f4 = rw[(m==mesh-1) ? m-2 : m+1];
        dum = f3 - f2;
        g1 = dum*dum1 + (f2 - f1)*dum2;
        g2 = (f4 - f3)*dum3 + dum*dum4;
        f = y22*(3.0*f2 + h2*g1 + (2.0*f2 + h2*g1)*y2)
          + y12*(3.0*f3 - h2*g2 - (2.0*f3 - h2*g2)*y1);
        df = 2.0*y2 / h2*(3.0*f2 + h2*g1 + (2.0*f2 + h2*g1)*y2)
          + y22*(2.0*f2 + h2*g1) / h2
          + 2.0*y1 / h2*(3.0*f3 - h2*g2 - (2.0*f3 - h2*g2)*y1)
          - y12*(2.0*f3 - h2*g2) / h2;
        if (L0 == 0) { a = 0.0; b = 0.5*df / rm; c = 0.0; d = f - b*rm*rm; }
        else if (L0 == 1) { a = (rm*df - f) / (2.0*rm*rm*rm); b = 0.0; c = df - 3.0*a*rm*rm; d = 0.0; }
        else { b = (3.0*f - rm*df) / (rm*rm); a = (f - b*rm*rm) / (rm*rm*rm); c = 0.0; d = 0.0; }
        RF[L0][Mul0] = a*R*R*R + b*R*R + c*R + d;
        dRF[L0][Mul0] = 3.0*a*R*R + 2.0*b*R + c;
      }
    }
  }
  else {
    do {
      m = (mp_min + mp_max) / 2;
      if (rv[m] < R) mp_min = m; else mp_max = m;
    } while ((mp_max - mp_min) != 1);
    m = mp_max;
    h1 = (m==1) ? rv[m-1] - rv[m+1] : rv[m-1] - rv[m-2];
    h2 = rv[m] - rv[m - 1];
    h3 = (m==mesh-1) ? rv[m-2] - rv[m] : rv[m+1] - rv[m];
    x1 = R - rv[m - 1]; x2 = R - rv[m];
    y1 = x1 / h2; y2 = x2 / h2; y12 = y1*y1; y22 = y2*y2;
    dum = h1 + h2; dum1 = h1 / h2 / dum; dum2 = h2 / h1 / dum;
    dum = h2 + h3; dum3 = h2 / h3 / dum; dum4 = h3 / h2 / dum;
    const double *rw = rwf_rows;
    for (L0 = 0; L0 <= maxl; L0++) {
      for (Mul0 = 0; Mul0 < nb[L0]; Mul0++, rw += mesh) {
        f1 = rw[(m==1) ? m+1 : m-2]; f2 = rw[m - 1]; f3 = rw[m]; f4 = rw[(m==mesh-1) ? m-2 : m+1];
        dum = f3 - f2;
        g1 = dum*dum1 + (f2 - f1)*dum2;
        g2 = (f4 - f3)*dum3 + dum*dum4;
        f = y22*(3.0*f2 + h2*g1 + (2.0*f2 + h2*g1)*y2)
          + y12*(3.0*f3 - h2*g2 - (2.0*f3 - h2*g2)*y1);
        df = 2.0*y2 / h2*(3.0*f2 + h2*g1 + (2.0*f2 + h2*g1)*y2)
          + y2*y2*(2.0*f2 + h2*g1) / h2
          + 2.0*y1 / h2*(3.0*f3 - h2*g2 - (2.0*f3 - h2*g2)*y1)
          - y1*y1*(2.0*f3 - h2*g2) / h2;
        RF[L0][Mul0] = f;
        dRF[L0][Mul0] = df;
      }
    }
  }
  if (po == 0) {
    siQ = sin(Q); coQ = cos(Q); siP = sin(P); coP = cos(P);
    dRx = siQ*coP; dRy = siQ*siP; dRz = coQ;
    if (Rmin < R) { dQx = coQ*coP / R; dQy = coQ*siP / R; dQz = -siQ / R; } else { dQx = 0.0; dQy = 0.0; dQz = 0.0; }
    if (Rmin < R) { dPx = -siP / R; dPy = coP / R; dPz = 0.0; } else { dPx = 0.0; dPy = 0.0; dPz = 0.0; }
    for (L0 = 0; L0 <= maxl; L0++) {
      if (L0 == 0) {
        AF[0][0] = 0.282094791773878; dAFQ[0][0] = 0.0; dAFP[0][0] = 0.0;
      } else if (L0 == 1) {
        dum = 0.48860251190292*siQ;
        AF[1][0] = dum*coP; AF[1][1] = dum*siP; AF[1][2] = 0.48860251190292*coQ;
        dAFQ[1][0] = 0.48860251190292*coQ*coP; dAFQ[1][1] = 0.48860251190292*coQ*siP; dAFQ[1][2] = -0.48860251190292*siQ;
        dAFP[1][0] = -0.48860251190292*siP; dAFP[1][1] = 0.48860251190292*coP; dAFP[1][2] = 0.0;
      } else if (L0 == 2) {
        dum1 = siQ*siQ;
        dum2 = 1.09254843059208*siQ*coQ;
        AF[2][0] = 0.94617469575756*coQ*coQ - 0.31539156525252;
        AF[2][1] = 0.54627421529604*dum1*(1.0 - 2.0*siP*siP);
        AF[2][2] = 1.09254843059208*dum1*siP*coP;
        AF[2][3] = dum2*coP;
        AF[2][4] = dum2*siP;
        dAFQ[2][0] = -1.89234939151512*siQ*coQ;
        dAFQ[2][1] = 1.09254843059208*siQ*coQ*(1.0 - 2.0*siP*siP);
        dAFQ[2][2] = 2.18509686118416*siQ*coQ*siP*coP;
        dAFQ[2][3] = 1.09254843059208*(1.0 - 2.0*siQ*siQ)*coP;
        dAFQ[2][4] = 1.09254843059208*(1.0 - 2.0*siQ*siQ)*siP;
        dAFP[2][0] = 0.0;
        dAFP[2][1] = -2.18509686118416*siQ*siP*coP;
        dAFP[2][2] = 1.09254843059208*siQ*(1.0 - 2.0*siP*siP);
        dAFP[2][3] = -1.09254843059208*coQ*siP;
        dAFP[2][4] = 1.09254843059208*coQ*coP;
      } else if (L0 == 3) {
        AF[3][0] = 0.373176332590116*(5.0*coQ*coQ*coQ - 3.0*coQ);
        AF[3][1] = 0.457045799464466*coP*siQ*(5.0*coQ*coQ - 1.0);
        AF[3][2] = 0.457045799464466*siP*siQ*(5.0*coQ*coQ - 1.0);
        AF[3][3] = 1.44530572132028*siQ*siQ*coQ*(coP*coP - siP*siP);
        AF[3][4] = 2.89061144264055*siQ*siQ*coQ*siP*coP;
        AF[3][5] = 0.590043589926644*siQ*siQ*siQ*(4.0*coP*coP*coP - 3.0*coP);
        AF[3][6] = 0.590043589926644*siQ*siQ*siQ*(3.0*siP - 4.0*siP*siP*siP);
        dAFQ[3][0] = 0.373176332590116*siQ*(-15.0*coQ*coQ + 3.0);
        dAFQ[3][1] = 0.457045799464466*coP*coQ*(15.0*coQ*coQ - 11.0);
        dAFQ[3][2] = 0.457045799464466*siP*coQ*(15.0*coQ*coQ - 11.0);
        dAFQ[3][3] = 1.44530572132028*(coP*coP - siP*siP)*siQ*(2.0*coQ*coQ - siQ*siQ);
        dAFQ[3][4] = 2.89061144264055*coP*siP*siQ*(2.0*coQ*coQ - siQ*siQ);
        dAFQ[3][5] = 1.770130769779932*coP*coQ*siQ*siQ*(-3.0 + 4.0*coP*coP);
        dAFQ[3][6] = 1.770130769779932*coQ*siP*siQ*siQ*(3.0 - 4.0*siP*siP);
        dAFP[3][0] = 0.0;
        dAFP[3][1] = 0.457045799464466*siP*(-5.0*coQ*coQ + 1.0);
        dAFP[3][2] = 0.457045799464466*coP*(5.0*coQ*coQ - 1.0);
        dAFP[3][3] = -5.781222885281120*coP*coQ*siP*siQ;
        dAFP[3][4] = 2.89061144264055*coQ*siQ*(coP*coP - siP*siP);
        dAFP[3][5] = 1.770130769779932*siP*siQ*siQ*(1.0 - 4.0*coP*coP);
        dAFP[3][6] = 1.770130769779932*coP*siQ*siQ*(1.0 - 4.0*siP*siP);
      }
    }
  }
  i1 = -1;
  for (L0 = 0; L0 <= maxl; L0++) {
    for (Mul0 = 0; Mul0 < nb[L0]; Mul0++) {
      for (M0 = 0; M0 <= 2 * L0; M0++) {
        i1++;
        if (po!=0){ out[3*i1] = out[3*i1+1] = out[3*i1+2] = 0.0; continue; }
        dChiR = dRF[L0][Mul0] * AF[L0][M0];
        dChiQ = RF[L0][Mul0] * dAFQ[L0][M0];
        dChiP = RF[L0][Mul0] * dAFP[L0][M0];
        out[3*i1]   = -dRx*dChiR - dQx*dChiQ - dPx*dChiP;
        out[3*i1+1] = -dRy*dChiR - dQy*dChiQ - dPy*dChiP;
        out[3*i1+2] = -dRz*dChiR - dQz*dChiQ - dPz*dChiP;
      }
    }
  }
}
}
#endif

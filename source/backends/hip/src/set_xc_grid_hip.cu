/**********************************************************************
  set_xc_grid_hip.cu

  Device evaluation of the GGA-PBE branch of Set_XC_Grid (XC_switch 4,
  XC_P_switch 0, 1 or 2, without the LDA+U cFLL rotation): the 27-point
  gradient stencil, the per-point PBE functional (device transcriptions
  of XC_PBE, XC_PW92C and XC_EX), the divergence stencil and the
  non-collinear rotation, as four kernels on the rank's D grid, in FP64
  as on the host and in the host's order of every sum (HIP port of the
  NVIDIA version's OpenACC kernels).  The results agree with the host
  loops to the last bits except for the transcendental functions of the
  device library.  All device memory is one transient arena; the
  outputs are written only when every kernel has completed, so a
  failure leaves them to the host loops.
***********************************************************************/
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstring>

namespace {

constexpr double kDenMin = 1.0e-14;
constexpr double kPi = 3.1415926535897932384626;
constexpr int kXcThreads = 128;

__device__ inline double sxg_largest(double a, double b)
{
    return (b <= a) ? a : b;
}

/* XC_EX with NSP == 1 */
__device__ void sxg_xc_ex1(double DS0, double *EX, double *VX)
{
    const double TRD = 0.333333333333333;
    const double COE1 = 0.6203504908994;
    const double COE2 = 0.610887057710857;
    double RS, VXP;

    if (DS0 <= kDenMin) RS = COE1 * pow(kDenMin, -1.0 / 3.0);
    else RS = COE1 * pow(DS0, -TRD);
    VXP = -COE2 / RS;
    *VX = VXP;
    *EX = 0.750 * VXP;
}

/* XC_PW92C without the LDA+U cFLL branch; dens may be clamped as there */
__device__ void sxg_xc_pw92c(double *dens, double *Ec, double *Vc)
{
    const double A[3] = {0.0310910, 0.0155450, 0.0168870};
    const double alpha1[3] = {0.2137000, 0.2054800, 0.1112500};
    const double beta1[3] = {7.5957000, 14.1189000, 10.3570000};
    const double beta2[3] = {3.5876000, 6.1977000, 3.6231000};
    const double beta3[3] = {1.6382000, 3.3662000, 0.8802600};
    const double beta4[3] = {0.4929400, 0.6251700, 0.4967100};
    double dtot, rs, srs, zeta, coe;
    double dum, dum1, dum2, b, c, dbdrs, dcdrs;
    double tmp0, tmp1, tmp2, tmp12, tmp14, tmp22, tmp24;
    double fpp0, f, dfdz;
    double G[3], dGdrs[3];
    double dEcdd[2];
    double dEcdrs, dEcdz;
    double drsdd, dzdd[2];

    coe = 0.6203504908994;
    dtot = dens[0] + dens[1];
    if (dtot <= kDenMin) {
        rs = 6203.504908994;
        dens[0] = 0.5 * 1.0e-14;
        dens[1] = 0.5 * 1.0e-14;
        dtot = kDenMin;
    }
    else
        rs = coe * pow(dtot, -0.33333333333333333333333);
    tmp0 = 1.0 / dtot;
    zeta = tmp0 * (dens[0] - dens[1]);
    if (0.99 < zeta) zeta = 0.99;
    if (zeta < -1.0) zeta = -0.99;
    drsdd = -0.3333333333333333333333 * rs * tmp0;
    dzdd[0] = tmp0 * (1.0 - zeta);
    dzdd[1] = tmp0 * (-1.0 - zeta);
    srs = sqrt(rs);
    for (int i = 0; i <= 2; i++) {
        b = beta1[i] * srs + rs * (beta2[i] + beta3[i] * srs + beta4[i] * rs);
        dbdrs = beta1[i] * 0.50 / srs + beta2[i] + beta3[i] * 1.50 * srs + beta4[i] * 2.0 * rs;
        c = 1.0 + 1.0 / (2.0 * A[i] * b);
        dcdrs = -(c - 1.0) * dbdrs / b;
        dum = log(c);
        dum1 = 1.0 + alpha1[i] * rs;
        G[i] = -2.0 * A[i] * dum1 * dum;
        dGdrs[i] = -2.0 * A[i] * (alpha1[i] * dum + dum1 * dcdrs / c);
    }
    c = 1.92366105093154;
    fpp0 = 1.70992093416137;
    dum1 = 1.0 + zeta;
    dum2 = 1.0 - zeta;
    tmp1 = pow(dum1, 0.333333333333333333);
    tmp2 = pow(dum2, 0.333333333333333333);
    tmp12 = tmp1 * tmp1;
    tmp22 = tmp2 * tmp2;
    tmp14 = tmp12 * tmp12;
    tmp24 = tmp22 * tmp22;
    f = (tmp14 + tmp24 - 2.0) * c;
    dfdz = 1.333333333333333333 * (tmp1 - tmp2) * c;
    dum1 = zeta * zeta * zeta;
    dum = dum1 * zeta;
    Ec[0] = G[0] - G[2] * f / fpp0 * (1.0 - dum) + (G[1] - G[0]) * f * dum;
    dEcdrs = dGdrs[0] - dGdrs[2] * f / fpp0 * (1.0 - dum) + (dGdrs[1] - dGdrs[0]) * f * dum;
    dEcdz = -G[2] / fpp0 * (dfdz * (1.0 - dum) - f * 4.0 * dum1) + (G[1] - G[0]) * (dfdz * dum + f * 4.0 * dum1);
    dum = dEcdrs * drsdd;
    dEcdd[0] = dum + dEcdz * dzdd[0];
    dEcdd[1] = dum + dEcdz * dzdd[1];
    Vc[0] = Ec[0] + dtot * dEcdd[0];
    Vc[1] = Ec[0] + dtot * dEcdd[1];
}

/* XC_PBE without the LDA+U cFLL branch */
__device__ void sxg_xc_pbe(double *dens, double GDENS[3][2], double *Exc, double *DEXDD, double *DECDD,
                           double DEXDGD[3][2], double DECDGD[3][2])
{
    const double HALF = 0.50, THD = (1.0 / 3.0), TWO = 2.0, TWOTHD = (2.0 / 3.0);
    const double beta = 0.06672455060314922, kappa = 0.8040;
    double Ec_unif[1], Vc_unif[2];
    double dt, rs, zeta;
    double den_min, gd_min, phi, t, ks, kF, f1, f2, f3, f4;
    double A, H, Fc, Fx;
    double GDMT, GDT[3];
    double DKFDD, DKSDD, DZDD[2], DPDZ;
    double DECUDD, DPDD, DTDD, DF1DD, DF2DD, DF3DD, DF4DD, DADD;
    double DHDD, DFCDD[2], DTDGD, DF3DGD, DF4DGD, DHDGD, DFCDGD[3][2];
    double DS[2], GDMS, KFS, s, f, DFDD, DFXDD[2], Vx_unif[2], Ex_unif[1];
    double GDS, DSDGD, DSDD, DF1DGD, DFDGD, DFXDGD[3][2];
    double D[2], GD[3][2], GDM[2];
    double gamma, mu;

    gamma = (1.0 - log(TWO)) / (kPi * kPi);
    mu = beta * kPi * kPi / 3.0;
    den_min = 1.0e-14;
    gd_min = 1.0e-14;
    dens[0] = sxg_largest(0.5 * den_min, dens[0]);
    dens[1] = sxg_largest(0.5 * den_min, dens[1]);
    D[0] = dens[0];
    D[1] = dens[1];
    dt = dens[0] + dens[1];
    for (int IX = 0; IX <= 2; IX++) {
        GD[IX][0] = GDENS[IX][0];
        GD[IX][1] = GDENS[IX][1];
        GDT[IX] = GDENS[IX][0] + GDENS[IX][1];
    }
    GDM[0] = sqrt(GD[0][0] * GD[0][0] + GD[1][0] * GD[1][0] + GD[2][0] * GD[2][0]);
    GDM[1] = sqrt(GD[0][1] * GD[0][1] + GD[1][1] * GD[1][1] + GD[2][1] * GD[2][1]);
    GDMT = sqrt(GDT[0] * GDT[0] + GDT[1] * GDT[1] + GDT[2] * GDT[2]);
    GDMT = sxg_largest(gd_min, GDMT);
    sxg_xc_pw92c(dens, Ec_unif, Vc_unif);
    rs = pow(3.0 / (4.0 * kPi * dt), THD);
    kF = pow(3.0 * kPi * kPi * dt, THD);
    ks = sqrt(4.0 * kF / kPi);
    zeta = (dens[0] - dens[1]) / dt;
    if (0.99 < zeta) zeta = 0.99;
    if (zeta < -1.0) zeta = -0.99;
    phi = 0.50 * (pow(1.0 + zeta, TWOTHD) + pow(1.0 - zeta, TWOTHD));
    t = GDMT / (2.0 * phi * ks * dt);
    f1 = Ec_unif[0] / (gamma * phi * phi * phi);
    f2 = exp(-f1);
    A = beta / gamma / (f2 - 1.0);
    f3 = t * t + A * t * t * t * t;
    f4 = beta / gamma * f3 / (1.0 + A * f3);
    H = gamma * phi * phi * phi * log(1.0 + f4);
    Fc = Ec_unif[0] + H;
    (void)rs;
    DKFDD = THD * kF / dt;
    DKSDD = HALF * ks * DKFDD / kF;
    DZDD[0] = 1.0 / dt - zeta / dt;
    DZDD[1] = -(1.0 / dt) - zeta / dt;
    DPDZ = HALF * TWOTHD * (1.0 / pow(1.0 + zeta, THD) - 1.0 / pow(1.0 - zeta, THD));
    for (int IS = 0; IS <= 1; IS++) {
        DECUDD = (Vc_unif[IS] - Ec_unif[0]) / dt;
        DPDD = DPDZ * DZDD[IS];
        DTDD = (-t) * (DPDD / phi + DKSDD / ks + 1.0 / dt);
        DF1DD = f1 * (DECUDD / Ec_unif[0] - 3.0 * DPDD / phi);
        DF2DD = (-f2) * DF1DD;
        DADD = (-A) * DF2DD / (f2 - 1.0);
        DF3DD = (2.0 * t + 4.0 * A * t * t * t) * DTDD + DADD * t * t * t * t;
        DF4DD = f4 * (DF3DD / f3 - (DADD * f3 + A * DF3DD) / (1.0 + A * f3));
        DHDD = 3.0 * H * DPDD / phi;
        DHDD = DHDD + gamma * phi * phi * phi * DF4DD / (1.0 + f4);
        DFCDD[IS] = Vc_unif[IS] + H + dt * DHDD;
        for (int IX = 0; IX <= 2; IX++) {
            DTDGD = (t / GDMT) * GDT[IX] / GDMT;
            DF3DGD = DTDGD * (2.0 * t + 4.0 * A * t * t * t);
            DF4DGD = f4 * DF3DGD * (1.0 / f3 - A / (1.0 + A * f3));
            DHDGD = gamma * phi * phi * phi * DF4DGD / (1.0 + f4);
            DFCDGD[IX][IS] = dt * DHDGD;
        }
    }
    Fx = 0.0;
    for (int IS = 0; IS <= 1; IS++) {
        DS[IS] = sxg_largest(den_min, 2.0 * D[IS]);
        GDMS = sxg_largest(gd_min, 2.0 * GDM[IS]);
        KFS = pow(3.0 * kPi * kPi * DS[IS], THD);
        s = GDMS / (2.0 * KFS * DS[IS]);
        f1 = 1.0 + mu * s * s / kappa;
        f = 1.0 + kappa - kappa / f1;
        sxg_xc_ex1(DS[IS], Ex_unif, Vx_unif);
        Fx = Fx + DS[IS] * Ex_unif[0] * f;
        DKFDD = THD * KFS / DS[IS];
        DSDD = s * (-(DKFDD / KFS) - 1.0 / DS[IS]);
        DF1DD = 2.0 * (f1 - 1.0) * DSDD / s;
        DFDD = kappa * DF1DD / (f1 * f1);
        DFXDD[IS] = Vx_unif[0] * f + DS[IS] * Ex_unif[0] * DFDD;
        for (int IX = 0; IX <= 2; IX++) {
            GDS = 2.0 * GD[IX][IS];
            DSDGD = (s / GDMS) * GDS / GDMS;
            DF1DGD = 2.0 * mu * s * DSDGD / kappa;
            DFDGD = kappa * DF1DGD / (f1 * f1);
            DFXDGD[IX][IS] = DS[IS] * Ex_unif[0] * DFDGD;
        }
    }
    Fx = HALF * Fx / dt;
    Exc[0] = Fx;
    Exc[1] = Fc;
    for (int IS = 0; IS <= 1; IS++) {
        DEXDD[IS] = DFXDD[IS];
        DECDD[IS] = DFCDD[IS];
        for (int IX = 0; IX <= 2; IX++) {
            DEXDGD[IX][IS] = DFXDGD[IX][IS];
            DECDGD[IX][IS] = DFCDGD[IX][IS];
        }
    }
}

/* 1. dDen_Grid: the 27-point interpolation (dDen_Grid_calc_method 2) */
__global__ __launch_bounds__(kXcThreads) void sxg_gradient(int n, int Ng1, int Ng2, int Ng3, int pcc,
                                                           const double *__restrict__ den0,
                                                           const double *__restrict__ den1,
                                                           const double *__restrict__ pcc0,
                                                           const double *__restrict__ pcc1,
                                                           const double *__restrict__ inv, double *__restrict__ dden)
{
    const int MN = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (MN >= n) return;
    const std::size_t sn = static_cast<std::size_t>(n);

    if (kDenMin < (den0[MN] + den1[MN])) {
        const int ii = MN / (Ng2 * Ng3);
        const int jj = (MN - ii * Ng2 * Ng3) / Ng3;
        const int kk = MN - ii * Ng2 * Ng3 - jj * Ng3;
        if (ii == 0 || ii == (Ng1 - 1) || jj == 0 || jj == (Ng2 - 1) || kk == 0 || kk == (Ng3 - 1)) {
            for (int dd = 0; dd < 6; dd++) dden[dd * sn + MN] = 0.0;
        }
        else {
            double f0[27], f1[27];
            /* the 27 neighbours in the host's order (a, b, c = -1..1, c fastest) */
            for (int pp = 0; pp < 27; pp++) {
                const int a = pp / 9 - 1, b = (pp / 3) % 3 - 1, c = pp % 3 - 1;
                const int MN2 = (ii + a) * Ng2 * Ng3 + (jj + b) * Ng3 + (kk + c);
                if (pcc) {
                    f0[pp] = den0[MN2] + pcc0[MN2];
                    f1[pp] = den1[MN2] + pcc1[MN2];
                }
                else {
                    f0[pp] = den0[MN2];
                    f1[pp] = den1[MN2];
                }
            }
            /* x: row 9, y: row 3, z: row 1 */
            for (int axis = 0; axis < 3; axis++) {
                const int row = (axis == 0) ? 9 : (axis == 1) ? 3 : 1;
                double s0 = 0.0, s1 = 0.0;
                for (int qq = 0; qq < 27; qq++) {
                    s0 += inv[row * 27 + qq] * f0[qq];
                    s1 += inv[row * 27 + qq] * f1[qq];
                }
                dden[axis * sn + MN] = s0;
                dden[(axis + 3) * sn + MN] = s1;
            }
        }
    }
    else {
        for (int dd = 0; dd < 6; dd++) dden[dd * sn + MN] = 0.0;
    }
}

/* 2. the PBE functional at every point */
__global__ __launch_bounds__(kXcThreads) void sxg_pbe(int n, int XC_P_switch, int pcc, const double *__restrict__ den0,
                                                      const double *__restrict__ den1,
                                                      const double *__restrict__ pcc0,
                                                      const double *__restrict__ pcc1,
                                                      const double *__restrict__ dden, double *__restrict__ dexc,
                                                      double *__restrict__ vxc0, double *__restrict__ vxc1)
{
    const int MN = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (MN >= n) return;
    const std::size_t sn = static_cast<std::size_t>(n);
    double ED[2], GDENS[3][2], Exc[2], DEXDD[2], DECDD[2], DEXDGD[3][2], DECDGD[3][2];

    ED[0] = den0[MN];
    ED[1] = den1[MN];
    if ((ED[0] + ED[1]) < kDenMin) {
        vxc0[MN] = 0.0;
        vxc1[MN] = 0.0;
        if (XC_P_switch != 0)
            for (int dd = 0; dd < 6; dd++) dexc[dd * sn + MN] = 0.0;
        return;
    }
    GDENS[0][0] = dden[0 * sn + MN];
    GDENS[1][0] = dden[1 * sn + MN];
    GDENS[2][0] = dden[2 * sn + MN];
    GDENS[0][1] = dden[3 * sn + MN];
    GDENS[1][1] = dden[4 * sn + MN];
    GDENS[2][1] = dden[5 * sn + MN];
    if (pcc) {
        ED[0] += pcc0[MN];
        ED[1] += pcc1[MN];
    }
    sxg_xc_pbe(ED, GDENS, Exc, DEXDD, DECDD, DEXDGD, DECDGD);
    if (XC_P_switch == 0) {
        vxc0[MN] = Exc[0] + Exc[1];
        vxc1[MN] = Exc[0] + Exc[1];
    }
    else if (XC_P_switch == 1) {
        vxc0[MN] = DEXDD[0] + DECDD[0];
        vxc1[MN] = DEXDD[1] + DECDD[1];
    }
    else {
        vxc0[MN] = Exc[0] + Exc[1] - DEXDD[0] - DECDD[0];
        vxc1[MN] = Exc[0] + Exc[1] - DEXDD[1] - DECDD[1];
    }
    if (XC_P_switch != 0) {
        dexc[0 * sn + MN] = DEXDGD[0][0] + DECDGD[0][0];
        dexc[1 * sn + MN] = DEXDGD[1][0] + DECDGD[1][0];
        dexc[2 * sn + MN] = DEXDGD[2][0] + DECDGD[2][0];
        dexc[3 * sn + MN] = DEXDGD[0][1] + DECDGD[0][1];
        dexc[4 * sn + MN] = DEXDGD[1][1] + DECDGD[1][1];
        dexc[5 * sn + MN] = DEXDGD[2][1] + DECDGD[2][1];
    }
}

/* 3. the divergence of dEXC_dGD (the second part of the GGA potential):
   d/dx of the x components (row 9), then d/dy (row 3), then d/dz (row 1),
   one running sum as on the host */
__global__ __launch_bounds__(kXcThreads) void sxg_divergence(int n, int Ng1, int Ng2, int Ng3, int XC_P_switch,
                                                             const double *__restrict__ den0,
                                                             const double *__restrict__ den1,
                                                             const double *__restrict__ inv,
                                                             const double *__restrict__ dexc,
                                                             double *__restrict__ dden, double *__restrict__ vxc0,
                                                             double *__restrict__ vxc1)
{
    const int MN = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (MN >= n) return;
    const std::size_t sn = static_cast<std::size_t>(n);

    if (kDenMin < (den0[MN] + den1[MN])) {
        const int ii = MN / (Ng2 * Ng3);
        const int jj = (MN - ii * Ng2 * Ng3) / Ng3;
        const int kk = MN - ii * Ng2 * Ng3 - jj * Ng3;
        if (ii <= 1 || (Ng1 - 2) <= ii || jj <= 1 || (Ng2 - 2) <= jj || kk <= 1 || (Ng3 - 2) <= kk) {
            vxc0[MN] = 0.0;
            vxc1[MN] = 0.0;
        }
        else {
            double s0 = 0.0, s1 = 0.0;
            for (int dd = 0; dd < 3; dd++) {
                const int row = (dd == 0) ? 9 : (dd == 1) ? 3 : 1;
                for (int pp = 0; pp < 27; pp++) {
                    const int a = pp / 9 - 1, b = (pp / 3) % 3 - 1, c = pp % 3 - 1;
                    const int MN2 = (ii + a) * Ng2 * Ng3 + (jj + b) * Ng3 + (kk + c);
                    s0 += inv[row * 27 + pp] * dexc[dd * sn + MN2];
                    s1 += inv[row * 27 + pp] * dexc[(dd + 3) * sn + MN2];
                }
            }
            if (XC_P_switch == 1) {
                vxc0[MN] -= s0;
                vxc1[MN] -= s1;
            }
            else {
                vxc0[MN] += s0;
                vxc1[MN] += s1;
            }
        }
    }
    else {
        for (int dd = 0; dd < 6; dd++) dden[dd * sn + MN] = 0.0;
    }
}

/* 4. non-collinear: rotate (Vxc0, Vxc1) by (theta, phi) into the 2x2 form */
__global__ __launch_bounds__(kXcThreads) void sxg_rotate(int n, const double *__restrict__ den2,
                                                         const double *__restrict__ den3, double *__restrict__ vxc0,
                                                         double *__restrict__ vxc1, double *__restrict__ vxc2,
                                                         double *__restrict__ vxc3)
{
    const int MN = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (MN >= n) return;
    const double tmp0 = 0.5 * (vxc0[MN] + vxc1[MN]);
    const double tmp1 = 0.5 * (vxc0[MN] - vxc1[MN]);
    const double theta = den2[MN];
    const double phi = den3[MN];
    const double sit = sin(theta), cot = cos(theta), sip = sin(phi), cop = cos(phi);

    vxc3[MN] = -tmp1 * sit * sip;
    vxc2[MN] = tmp1 * sit * cop;
    vxc1[MN] = tmp0 - cot * tmp1;
    vxc0[MN] = tmp0 + cot * tmp1;
}

std::size_t arena_off(std::size_t *pos, std::size_t bytes)
{
    const std::size_t off = *pos;
    *pos = (off + bytes + 511u) & ~static_cast<std::size_t>(511u);
    return off;
}

} // namespace

/* Returns 1 when the whole evaluation ran on the device (the outputs are in
   vxc0..3 and, where the caller passed them, dden_out[6] / dexc_out[6]),
   0 when nothing was written (the caller runs the host loops). */
extern "C" int Set_XC_Grid_HIP(int n, int Ng1, int Ng2, int Ng3, int XC_P_switch, int pcc, int rotate,
                               const double *inv27, const double *Den0, const double *Den1, const double *Den2,
                               const double *Den3, const double *Pcc0, const double *Pcc1, double *Vxc0,
                               double *Vxc1, double *Vxc2, double *Vxc3, double *const *dden_out,
                               double *const *dexc_out)
{
    const int second = (XC_P_switch == 1 || XC_P_switch == 2);
    const std::size_t nb = sizeof(double) * static_cast<std::size_t>(n);
    std::size_t pos = 0, free_bytes = 0, total_bytes = 0;
    unsigned char *arena = nullptr;
    hipError_t status = hipSuccess;
    int ok = 0;

    if (n <= 0) return 1;
    (void)hipGetLastError();
    const std::size_t o_den0 = arena_off(&pos, nb);
    const std::size_t o_den1 = arena_off(&pos, nb);
    const std::size_t o_den2 = rotate ? arena_off(&pos, nb) : 0;
    const std::size_t o_den3 = rotate ? arena_off(&pos, nb) : 0;
    const std::size_t o_pcc0 = pcc ? arena_off(&pos, nb) : 0;
    const std::size_t o_pcc1 = pcc ? arena_off(&pos, nb) : 0;
    const std::size_t o_dden = arena_off(&pos, 6 * nb);
    const std::size_t o_dexc = arena_off(&pos, 6 * nb);
    const std::size_t o_vxc0 = arena_off(&pos, nb);
    const std::size_t o_vxc1 = arena_off(&pos, nb);
    const std::size_t o_vxc2 = rotate ? arena_off(&pos, nb) : 0;
    const std::size_t o_vxc3 = rotate ? arena_off(&pos, nb) : 0;
    const std::size_t o_inv = arena_off(&pos, sizeof(double) * 27 * 27);

    /* leave room for the other ranks sharing the device */
    if (hipMemGetInfo(&free_bytes, &total_bytes) != hipSuccess ||
        free_bytes < pos + (static_cast<std::size_t>(256) << 20)) {
        (void)hipGetLastError();
        return 0;
    }
    if (hipMalloc(reinterpret_cast<void **>(&arena), pos) != hipSuccess) {
        (void)hipGetLastError();
        return 0;
    }
    double *d_den0 = reinterpret_cast<double *>(arena + o_den0);
    double *d_den1 = reinterpret_cast<double *>(arena + o_den1);
    double *d_den2 = rotate ? reinterpret_cast<double *>(arena + o_den2) : nullptr;
    double *d_den3 = rotate ? reinterpret_cast<double *>(arena + o_den3) : nullptr;
    double *d_pcc0 = pcc ? reinterpret_cast<double *>(arena + o_pcc0) : nullptr;
    double *d_pcc1 = pcc ? reinterpret_cast<double *>(arena + o_pcc1) : nullptr;
    double *d_dden = reinterpret_cast<double *>(arena + o_dden);
    double *d_dexc = reinterpret_cast<double *>(arena + o_dexc);
    double *d_vxc0 = reinterpret_cast<double *>(arena + o_vxc0);
    double *d_vxc1 = reinterpret_cast<double *>(arena + o_vxc1);
    double *d_vxc2 = rotate ? reinterpret_cast<double *>(arena + o_vxc2) : nullptr;
    double *d_vxc3 = rotate ? reinterpret_cast<double *>(arena + o_vxc3) : nullptr;
    double *d_inv = reinterpret_cast<double *>(arena + o_inv);
    const dim3 grid(static_cast<unsigned>((n + kXcThreads - 1) / kXcThreads)), block(kXcThreads);

#define SXG_CHECK(call) do { status = (call); if (status != hipSuccess) goto done; } while (0)
    SXG_CHECK(hipMemcpy(d_den0, Den0, nb, hipMemcpyHostToDevice));
    SXG_CHECK(hipMemcpy(d_den1, Den1, nb, hipMemcpyHostToDevice));
    if (rotate) {
        SXG_CHECK(hipMemcpy(d_den2, Den2, nb, hipMemcpyHostToDevice));
        SXG_CHECK(hipMemcpy(d_den3, Den3, nb, hipMemcpyHostToDevice));
    }
    if (pcc) {
        SXG_CHECK(hipMemcpy(d_pcc0, Pcc0, nb, hipMemcpyHostToDevice));
        SXG_CHECK(hipMemcpy(d_pcc1, Pcc1, nb, hipMemcpyHostToDevice));
    }
    SXG_CHECK(hipMemcpy(d_inv, inv27, sizeof(double) * 27 * 27, hipMemcpyHostToDevice));

    hipLaunchKernelGGL(sxg_gradient, grid, block, 0, 0, n, Ng1, Ng2, Ng3, pcc, d_den0, d_den1, d_pcc0, d_pcc1, d_inv,
                       d_dden);
    SXG_CHECK(hipGetLastError());
    hipLaunchKernelGGL(sxg_pbe, grid, block, 0, 0, n, XC_P_switch, pcc, d_den0, d_den1, d_pcc0, d_pcc1, d_dden, d_dexc,
                       d_vxc0, d_vxc1);
    SXG_CHECK(hipGetLastError());
    if (second) {
        hipLaunchKernelGGL(sxg_divergence, grid, block, 0, 0, n, Ng1, Ng2, Ng3, XC_P_switch, d_den0, d_den1, d_inv,
                           d_dexc, d_dden, d_vxc0, d_vxc1);
        SXG_CHECK(hipGetLastError());
    }
    if (rotate) {
        hipLaunchKernelGGL(sxg_rotate, grid, block, 0, 0, n, d_den2, d_den3, d_vxc0, d_vxc1, d_vxc2, d_vxc3);
        SXG_CHECK(hipGetLastError());
    }
    SXG_CHECK(hipDeviceSynchronize());

    /* the host rotation's store order: the callers may alias the outputs,
       and then the diagonal Vxc0 has to be written last */
    if (rotate) {
        SXG_CHECK(hipMemcpy(Vxc3, d_vxc3, nb, hipMemcpyDeviceToHost));
        SXG_CHECK(hipMemcpy(Vxc2, d_vxc2, nb, hipMemcpyDeviceToHost));
        SXG_CHECK(hipMemcpy(Vxc1, d_vxc1, nb, hipMemcpyDeviceToHost));
        SXG_CHECK(hipMemcpy(Vxc0, d_vxc0, nb, hipMemcpyDeviceToHost));
    }
    else {
        SXG_CHECK(hipMemcpy(Vxc0, d_vxc0, nb, hipMemcpyDeviceToHost));
        SXG_CHECK(hipMemcpy(Vxc1, d_vxc1, nb, hipMemcpyDeviceToHost));
    }
    if (dden_out != nullptr) {
        for (int i = 0; i < 6; i++)
            SXG_CHECK(hipMemcpy(dden_out[i], d_dden + static_cast<std::size_t>(i) * n, nb, hipMemcpyDeviceToHost));
    }
    if (dexc_out != nullptr && XC_P_switch != 0) {
        for (int i = 0; i < 6; i++)
            SXG_CHECK(hipMemcpy(dexc_out[i], d_dexc + static_cast<std::size_t>(i) * n, nb, hipMemcpyDeviceToHost));
    }
    ok = 1;
#undef SXG_CHECK

done:
    hipFree(arena);
    (void)hipGetLastError();
    return ok;
}

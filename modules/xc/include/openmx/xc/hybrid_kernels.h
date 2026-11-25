#ifndef OPENMX_XC_HYBRID_KERNELS_H
#define OPENMX_XC_HYBRID_KERNELS_H

/* The semilocal parts of the hybrid functionals as inline C functions, in the
   form of kernels.h and with its conventions: one implementation for the C++
   library (functionals.cpp, wpbeh_kernel.cpp, special_functions.cpp), for C
   callers, and for the application's GPU device code.  Like kernels.h, the
   functions agree bitwise with the library only in a unit compiled without
   floating-point contraction.

   Short-range omega-PBE exchange (wPBEh) of one spin channel: the exchange
   energy of the screened interaction erfc(omega r)/r for the PBE exchange
   hole, in the analytic form of J. Heyd.

   Source.  The energy expressions and the constants were transcribed from the
   routines pbexsr and wpbe_analy_erfc_approx_grad of Quantum ESPRESSO 7.6
   (XClib/qe_funct_exch_gga.f90, tag qe-7.6; Copyright (C) 2020 Quantum
   ESPRESSO group, distributed under the terms of the GNU General Public
   License), through the multi-precision reference tools/wpbeh_reference.py of
   this repository, which lists the papers.  This file is a new
   double-precision implementation and is distributed under the licence of
   this library, the GNU General Public License (see LICENSE).

     kF    = (6 pi^2 rho_s)^(1/3)                  w = omega / kF
     s_raw = |grad rho_s| / (2 (6 pi^2)^(1/3) rho_s^(4/3))
     s     = s_raw up to 8.3, 8.572844 - 18.796223 / s_raw^2 above
     e_s   = -(3/4) (6/pi)^(1/3) rho_s^(4/3) f(w, s)
     f     = -(8/9) (term1 + term2 + term3 + term4 + term5)

   The constants are the published, rounded ones.  The hole function EG(s) is
   a three-term polynomial up to s = 0.08 and a closed expression above; up to
   w = 14, term1 uses a fit of erfc with eight coefficients, above one
   Gaussian.  f therefore has steps at s = 0.08 (up to 7e-9 of its value), at
   s_raw = 8.3 (up to 1e-10) and at w = 14 (up to 3e-4); the derivatives
   returned are those of the expression in use.

   Differences from the Fortran routine:
     - The derivatives are those of the energy above, the rescaling of s
       included.  The Fortran routine leaves the derivative of the rescaling
       out of its potential above s_raw = 8.3.
     - exp(x) E1(x) and exp(x^2) erfc(x) are evaluated to full precision for
       every argument (openmx_xc_kernel_e1_scaled, openmx_xc_kernel_erfcx).
       The Fortran routine uses asymptotic approximations above w = 14.
     - s_raw below 1e-10 is replaced by 1e-10.  term1 has logarithmic
       singularities at w = 0, s = 0 that cancel in the sum, and
       f(w, s) - f(w, 0) is of the order s^2, so this changes f by less than
       1e-19 of its value.  (At omega = 0 and a vanishing gradient the Fortran
       routine returns f = 1; the formula tends to 1 - 4.8e-9.)

   The evaluation follows the double-precision form of the reference
   (double_channel_energy): the hole function EG(s) enters only through
   Q = E + s^2 EG(s), which is computed without a division by s^2, and
   log(Hsbw/DHsbw) is computed as -log1p(D/Hsbw).  Between w = 4 and w = 14
   the terms of term1 cancel and digits are lost; the tolerances of the
   fixture (tests/fixtures/wpbeh_reference.txt) describe the remaining
   accuracy.

   The derivatives with respect to rho_s and |grad rho_s| are carried along
   with the value (forward mode, openmx_xc_dual), so they have the rounding
   behaviour of the value and no separately transcribed formulas.  No
   function here calls itself, so that the device code can call all of
   them. */

#include "kernels.h"

#if defined(OPENMX_XC_KERNELS_OMP_TARGET)
#pragma omp declare target
#endif

/* exp(x) E1(x) and erfcx(x) = exp(x^2) erfc(x) in double precision, each
   with its first derivative.  Small arguments use the power series of E1 and
   the product exp(x^2) erfc(x); larger arguments use the continued fractions

     exp(x) E1(x) = 1 / (x + 1 - 1 / (x + 3 - 4 / (x + 5 - 9 / (x + 7 - ...))))
     erfcx(x)     = (1/sqrt(pi)) / (x + (1/2) / (x + 1 / (x + (3/2) / (x + ...))))

   evaluated from a fixed depth upwards.  The depths were chosen so that the
   truncation is below the rounding error on each interval, with a margin of
   one step of the table.  The derivatives are taken from the same fraction
   (exp(x) E1(x) - 1/x and 2 x erfcx(x) - 2/sqrt(pi) cancel for large x).
   tests/test_wpbeh.cpp compares values and derivatives with 50-digit
   references between 1e-30 and 1e8. */

/* out[0] = exp(x) E1(x), out[1] = exp(x) E1(x) - 1/x, for x > 0.
   x = 0 gives (+infinity, -infinity); negative x and NaN give NaN. */
OPENMX_XC_KERNEL void openmx_xc_kernel_e1_scaled(double x, double out[2])
{
  if (!(x > 0.0)) {
    if (x == 0.0) {
      out[0] = INFINITY;
      out[1] = -INFINITY;
    }
    else {
      out[0] = NAN;
      out[1] = NAN;
    }
    return;
  }
  if (x < 1.0) {
    /* E1(x) = -gamma - ln x - sum_{k >= 1} (-x)^k / (k k!) */
    const double euler_gamma = 0.57721566490153286061;
    double sum = 0.0, term = 1.0, value;
    int k;
    for (k = 1; k <= 40; ++k) {
      double add;
      term *= -x / k;
      add = -term / k;
      sum += add;
      if (fabs(add) <= 1e-17 * fabs(sum)) break;
    }
    value = exp(x) * (-euler_gamma - log(x) + sum);
    out[0] = value;
    out[1] = value - 1.0 / x;
    return;
  }
  {
    const int depth = x < 1.5 ? 120 : x < 2.0 ? 100 : x < 3.0 ? 60 : x < 5.0 ? 48 : x < 10.0 ? 32 : x < 20.0 ? 20 : 16;
    double tail = 0.0, value;
    int k;
    for (k = depth; k >= 1; --k) tail = ((double)k * k) / (x + (2 * k + 1) - tail);
    value = 1.0 / (x + 1.0 - tail);
    out[0] = value;
    out[1] = (tail - 1.0) * value / x;
  }
}

/* erfcx(x) and its derivative 2 x erfcx(x) - 2/sqrt(pi) for x >= 0 */
OPENMX_XC_KERNEL void openmx_xc_kernel_erfcx_nonnegative(double x, double out[2])
{
  const double two_over_sqrt_pi = 1.1283791670955125739, one_over_sqrt_pi = 0.56418958354775628695;
  if (x < 2.0) {
    const double value = exp(x * x) * erfc(x);
    out[0] = value;
    out[1] = 2.0 * x * value - two_over_sqrt_pi;
    return;
  }
  {
    const int depth = x < 2.5 ? 100 : x < 3.0 ? 80 : x < 4.0 ? 50 : x < 6.0 ? 30 : x < 10.0 ? 20 : 12;
    double tail = 0.0;
    int k;
    for (k = depth; k >= 1; --k) tail = (0.5 * k) / (x + tail);
    out[0] = one_over_sqrt_pi / (x + tail);
    out[1] = -two_over_sqrt_pi * tail / (x + tail);
  }
}

/* out[0] = erfcx(x), out[1] = 2 x erfcx(x) - 2/sqrt(pi); NaN gives NaN */
OPENMX_XC_KERNEL void openmx_xc_kernel_erfcx(double x, double out[2])
{
  const double two_over_sqrt_pi = 1.1283791670955125739;
  if (isnan(x)) {
    out[0] = x;
    out[1] = x;
    return;
  }
  if (x < 0.0) {
    double positive[2], value;
    openmx_xc_kernel_erfcx_nonnegative(-x, positive);
    value = 2.0 * exp(x * x) - positive[0];
    out[0] = value;
    out[1] = 2.0 * x * value - two_over_sqrt_pi;
    return;
  }
  openmx_xc_kernel_erfcx_nonnegative(x, out);
}

/* A value with its partial derivatives with respect to the spin density (r)
   and the norm of its gradient (g).  Each function is one operation of the
   value, with the derivatives in the order of the operands; the sums and
   products of a value and a constant serve both orders of the operands. */
typedef struct {
  double v, r, g;
} openmx_xc_dual;

OPENMX_XC_KERNEL openmx_xc_dual openmx_xc_dual_constant(double c)
{
  const openmx_xc_dual z = {c, 0.0, 0.0};
  return z;
}
/* f(a) with the derivative df = f'(a.v) */
OPENMX_XC_KERNEL openmx_xc_dual openmx_xc_dual_chain(openmx_xc_dual a, double f, double df)
{
  const openmx_xc_dual z = {f, df * a.r, df * a.g};
  return z;
}
OPENMX_XC_KERNEL openmx_xc_dual openmx_xc_dual_add(openmx_xc_dual a, openmx_xc_dual b)
{
  const openmx_xc_dual z = {a.v + b.v, a.r + b.r, a.g + b.g};
  return z;
}
/* a + c and c + a */
OPENMX_XC_KERNEL openmx_xc_dual openmx_xc_dual_add_constant(openmx_xc_dual a, double c)
{
  const openmx_xc_dual z = {a.v + c, a.r, a.g};
  return z;
}
OPENMX_XC_KERNEL openmx_xc_dual openmx_xc_dual_negate(openmx_xc_dual a)
{
  const openmx_xc_dual z = {-a.v, -a.r, -a.g};
  return z;
}
OPENMX_XC_KERNEL openmx_xc_dual openmx_xc_dual_subtract(openmx_xc_dual a, openmx_xc_dual b)
{
  const openmx_xc_dual z = {a.v - b.v, a.r - b.r, a.g - b.g};
  return z;
}
/* a - c */
OPENMX_XC_KERNEL openmx_xc_dual openmx_xc_dual_subtract_constant(openmx_xc_dual a, double c)
{
  const openmx_xc_dual z = {a.v - c, a.r, a.g};
  return z;
}
/* c - a */
OPENMX_XC_KERNEL openmx_xc_dual openmx_xc_dual_constant_minus(double c, openmx_xc_dual a)
{
  const openmx_xc_dual z = {c - a.v, -a.r, -a.g};
  return z;
}
OPENMX_XC_KERNEL openmx_xc_dual openmx_xc_dual_multiply(openmx_xc_dual a, openmx_xc_dual b)
{
  const openmx_xc_dual z = {a.v * b.v, a.r * b.v + a.v * b.r, a.g * b.v + a.v * b.g};
  return z;
}
/* c a and a c */
OPENMX_XC_KERNEL openmx_xc_dual openmx_xc_dual_scale(double c, openmx_xc_dual a)
{
  const openmx_xc_dual z = {c * a.v, c * a.r, c * a.g};
  return z;
}
OPENMX_XC_KERNEL openmx_xc_dual openmx_xc_dual_divide(openmx_xc_dual a, openmx_xc_dual b)
{
  const double q = a.v / b.v;
  const openmx_xc_dual z = {q, (a.r - q * b.r) / b.v, (a.g - q * b.g) / b.v};
  return z;
}
/* a / c */
OPENMX_XC_KERNEL openmx_xc_dual openmx_xc_dual_divide_constant(openmx_xc_dual a, double c)
{
  const openmx_xc_dual z = {a.v / c, a.r / c, a.g / c};
  return z;
}
/* c / a */
OPENMX_XC_KERNEL openmx_xc_dual openmx_xc_dual_constant_over(double c, openmx_xc_dual a)
{
  const double q = c / a.v;
  const openmx_xc_dual z = {q, -q * a.r / a.v, -q * a.g / a.v};
  return z;
}
OPENMX_XC_KERNEL openmx_xc_dual openmx_xc_dual_sqrt(openmx_xc_dual a)
{
  const double f = sqrt(a.v);
  return openmx_xc_dual_chain(a, f, 0.5 / f);
}
OPENMX_XC_KERNEL openmx_xc_dual openmx_xc_dual_cbrt(openmx_xc_dual a)
{
  const double f = cbrt(a.v);
  return openmx_xc_dual_chain(a, f, f / (3.0 * a.v));
}
OPENMX_XC_KERNEL openmx_xc_dual openmx_xc_dual_log1p(openmx_xc_dual a)
{
  return openmx_xc_dual_chain(a, log1p(a.v), 1.0 / (1.0 + a.v));
}
OPENMX_XC_KERNEL openmx_xc_dual openmx_xc_dual_e1_scaled(openmx_xc_dual a)
{
  double f[2];
  openmx_xc_kernel_e1_scaled(a.v, f);
  return openmx_xc_dual_chain(a, f[0], f[1]);
}
OPENMX_XC_KERNEL openmx_xc_dual openmx_xc_dual_erfcx(openmx_xc_dual a)
{
  double f[2];
  openmx_xc_kernel_erfcx(a.v, f);
  return openmx_xc_dual_chain(a, f[0], f[1]);
}

/* Short names for the channel energy below; undefined after it. */
#define OXD_ADD openmx_xc_dual_add
#define OXD_ADDC openmx_xc_dual_add_constant
#define OXD_NEG openmx_xc_dual_negate
#define OXD_SUB openmx_xc_dual_subtract
#define OXD_SUBC openmx_xc_dual_subtract_constant
#define OXD_CSUB openmx_xc_dual_constant_minus
#define OXD_MUL openmx_xc_dual_multiply
#define OXD_SCALE openmx_xc_dual_scale
#define OXD_DIV openmx_xc_dual_divide
#define OXD_DIVC openmx_xc_dual_divide_constant
#define OXD_CDIV openmx_xc_dual_constant_over
#define OXD_SQRT openmx_xc_dual_sqrt

/* The short-range exchange of one spin channel: out[0] the energy per volume
   of the channel, out[1] its derivative with respect to the spin density,
   out[2] with respect to the norm of its gradient.  density > 0 is the spin
   density rho_s, gradient_norm >= 0 is |grad rho_s|, omega >= 0 is the
   screening parameter in 1/bohr. */
OPENMX_XC_KERNEL void openmx_xc_kernel_wpbeh_channel(double density, double gradient_norm, double omega,
                                                     double out[3])
{
  /* PBE exchange hole: A, B, C, D, E, the coefficients of H(s), and
     F(s) = Fc1 H(s) + Fc2 */
  const double A = 1.0161144, B = -0.37170836, C = -0.077215461, D = 0.57786348, E = -0.051955731;
  const double ha0 = 0.00979681, ha1 = 0.0410834, ha2 = 0.187440, ha3 = 0.00120824, ha4 = 0.0347188;
  const double fc1 = 6.4753871, fc2 = 0.47965830;
  /* EG(s) = ega0 + ega1 s^2 + ega2 s^4 up to s = eg_s_cut */
  const double ega0 = -0.02628417880, ega1 = -0.07117647788, ega2 = 0.08534541323;
  const double eg_s_cut = 0.08;
  /* erfc(x) ~ (1 + sum_k ea_k x^(k+1)) exp(-eb1 x^2) up to w = w_cut,
     exp(-2 x^2) above */
  const double ea0 = -1.128223946706117, ea1 = 1.452736265762971, ea2 = -1.243162299390327,
               ea3 = 0.971824836115601, ea4 = -0.568861079687373, ea5 = 0.246880514820192,
               ea6 = -0.065032363850763, ea7 = 0.008401793031216;
  const double eb1_fit = 1.455915450052607, eb1_gaussian = 2.0, w_cut = 14.0;
  /* rescaling of the reduced gradient above s_trans, and the guard of this
     file at small s */
  const double s_trans = 8.3, s_max = 8.572844, s_const = 18.796223, s_floor = 1.0e-10;
  const double pi = 3.14159265358979323846;
  const double sqrt_pi = 1.77245385090551602730;
  const double sqrt_A = 1.00802499968998784748;            /* sqrt(A) */
  const double A_32 = 1.02426871774499218765;              /* A^(3/2) */
  const double A_52 = 1.04077419357022208975;              /* A^(5/2) */
  const double cbrt_6pi2 = 3.89777708972075395896;         /* (6 pi^2)^(1/3) */
  const double exchange_prefactor = 0.930525736349100025002;  /* (3/4) (6/pi)^(1/3) */

  const openmx_xc_dual rho = {density, 1.0, 0.0}, gamma = {gradient_norm, 0.0, 1.0};
  const openmx_xc_dual rho13 = openmx_xc_dual_cbrt(rho), rho43 = OXD_MUL(rho, rho13);
  const openmx_xc_dual w = OXD_CDIV(omega, OXD_SCALE(cbrt_6pi2, rho13));
  const openmx_xc_dual s_raw = OXD_DIV(gamma, OXD_SCALE(2.0 * cbrt_6pi2, rho43));
  openmx_xc_dual s = s_raw, s2, s4, h, f_s, w2, dhs, dhsw, hsbw, hsbwa94, dhsbw, dhs_2, dhs_3, q;
  openmx_xc_dual dhsw_2, dhsw_52, w3, w5, term1, term2, term3, term4, term5, f, e;
  int gaussian;
  double eb1;

  if (s_raw.v > s_trans) {
    s = OXD_CSUB(s_max, OXD_CDIV(s_const, OXD_MUL(s_raw, s_raw)));
  }
  else if (s_raw.v < s_floor) {
    s = openmx_xc_dual_constant(s_floor);
  }

  /* The names follow the Fortran routine: Hsbw = s^2 H + eb1 w^2,
     DHsbw = D + Hsbw, DHs = D + s^2 H, DHsw = DHs + w^2,
     HsbwA94 = 9 Hsbw / (4 A). */
  gaussian = w.v > w_cut;
  eb1 = gaussian ? eb1_gaussian : eb1_fit;
  s2 = OXD_MUL(s, s);
  s4 = OXD_MUL(s2, s2);
  h = OXD_DIV(OXD_ADD(OXD_SCALE(ha0, s2), OXD_SCALE(ha1, s4)),
              OXD_ADD(OXD_ADD(OXD_ADDC(OXD_SCALE(ha2, s4), 1.0), OXD_MUL(OXD_SCALE(ha3, s4), s)),
                      OXD_MUL(OXD_SCALE(ha4, s4), s2)));
  f_s = OXD_ADDC(OXD_SCALE(fc1, h), fc2);
  w2 = OXD_MUL(w, w);
  dhs = OXD_ADDC(OXD_MUL(s2, h), D);
  dhsw = OXD_ADD(dhs, w2);
  hsbw = OXD_ADD(OXD_MUL(s2, h), OXD_SCALE(eb1, w2));
  hsbwa94 = OXD_DIVC(OXD_SCALE(9.0, hsbw), 4.0 * A);
  dhsbw = OXD_ADDC(hsbw, D);
  dhs_2 = OXD_MUL(dhs, dhs);
  dhs_3 = OXD_MUL(dhs_2, dhs);

  /* Q = E + s^2 EG(s) */
  if (s.v > eg_s_cut) {
    /* EG = -(3 pi/4 + Ga)/Gb with Gb = 15 sqrt(pi) s^2 / (16 DHs^(7/2)),
       hence Q = E - (3 pi/4 + Ga(s)) 16 DHs^(7/2) / (15 sqrt(pi)) */
    const openmx_xc_dual dhs_72 = OXD_MUL(dhs_3, OXD_SQRT(dhs));
    const openmx_xc_dual ga_sum =
        OXD_ADD(OXD_ADD(OXD_ADDC(OXD_MUL(OXD_SCALE(6.0 * C, OXD_ADDC(OXD_MUL(f_s, s2), 1.0)), dhs), 15.0 * E),
                        OXD_SCALE(4.0 * B, dhs_2)),
                OXD_SCALE(8.0 * A, dhs_3));
    const openmx_xc_dual ga =
        OXD_SUB(OXD_DIV(OXD_SCALE(sqrt_pi, ga_sum), OXD_SCALE(16.0, dhs_72)),
                OXD_SCALE(0.75 * pi * sqrt_A,
                          openmx_xc_dual_erfcx(OXD_SQRT(OXD_DIVC(OXD_MUL(OXD_SCALE(9.0, h), s2), 4.0 * A)))));
    q = OXD_CSUB(E, OXD_DIVC(OXD_MUL(OXD_SCALE(16.0, OXD_ADDC(ga, 0.75 * pi)), dhs_72), 15.0 * sqrt_pi));
  }
  else {
    q = OXD_ADDC(OXD_MUL(s2, OXD_ADD(OXD_ADDC(OXD_SCALE(ega1, s2), ega0), OXD_SCALE(ega2, s4))), E);
  }

  dhsw_2 = OXD_MUL(dhsw, dhsw);
  dhsw_52 = OXD_MUL(dhsw_2, OXD_SQRT(dhsw));
  w3 = OXD_MUL(w2, w);
  w5 = OXD_MUL(w3, w2);
  term2 = OXD_DIV(OXD_ADD(OXD_ADD(OXD_ADD(OXD_SCALE(B, dhs_2), OXD_SCALE(C, dhs)),
                                  OXD_MUL(OXD_SCALE(C, OXD_MUL(dhs, s2)), f_s)),
                          OXD_SCALE(2.0, q)),
                  OXD_SCALE(2.0, dhs_3));
  term3 = OXD_DIV(OXD_MUL(OXD_NEG(w),
                          OXD_ADD(OXD_ADD(OXD_ADD(OXD_SCALE(B, OXD_SCALE(4.0, dhsw_2)),
                                                  OXD_SCALE(C, OXD_SCALE(6.0, dhsw))),
                                          OXD_MUL(OXD_SCALE(C, OXD_MUL(OXD_SCALE(6.0, dhsw), s2)), f_s)),
                                  OXD_SCALE(15.0, q))),
                  OXD_MUL(OXD_SCALE(8.0, dhs), dhsw_52));
  term4 = OXD_DIV(OXD_MUL(OXD_NEG(w3),
                          OXD_ADD(OXD_ADD(OXD_SCALE(C, dhsw), OXD_MUL(OXD_SCALE(C, OXD_MUL(dhsw, s2)), f_s)),
                                  OXD_SCALE(5.0, q))),
                  OXD_MUL(OXD_SCALE(2.0, dhs_2), dhsw_52));
  term5 = OXD_DIV(OXD_MUL(OXD_NEG(w5), q), OXD_MUL(dhs_3, dhsw_52));

  /* expei = exp(x) Ei(-x) = -exp(x) E1(x) at x = HsbwA94 */
  if (gaussian) {
    term1 = OXD_SCALE(-0.5 * A, OXD_ADD(OXD_NEG(openmx_xc_dual_e1_scaled(hsbwa94)),
                                        openmx_xc_dual_log1p(OXD_CDIV(D, hsbw))));
  }
  else {
    const openmx_xc_dual w4 = OXD_MUL(w2, w2), w6 = OXD_MUL(w3, w3), w7 = OXD_MUL(w6, w),
                         w8 = OXD_MUL(w4, w4);
    const openmx_xc_dual np1 =
        OXD_ADD(OXD_SUB(OXD_ADD(OXD_SCALE(-1.5 * ea0 * sqrt_A, w), OXD_DIVC(OXD_SCALE(27.0 * ea2, w3), 8.0 * sqrt_A)),
                        OXD_DIVC(OXD_SCALE(243.0 * ea4, w5), 32.0 * A_32)),
                OXD_DIVC(OXD_SCALE(2187.0 * ea6, w7), 128.0 * A_52));
    const openmx_xc_dual np2 =
        OXD_SUB(OXD_ADD(OXD_SUB(OXD_ADDC(OXD_DIVC(OXD_SCALE(9.0 * ea1, w2), 4.0), -A),
                                OXD_DIVC(OXD_SCALE(81.0 * ea3, w4), 16.0 * A)),
                        OXD_DIVC(OXD_SCALE(729.0 * ea5, w6), 64.0 * A * A)),
                OXD_DIVC(OXD_SCALE(6561.0 * ea7, w8), 256.0 * A * A * A));
    const openmx_xc_dual t1 =
        OXD_SCALE(0.5, OXD_SUB(OXD_MUL(OXD_SCALE(pi, np1), openmx_xc_dual_erfcx(OXD_SQRT(hsbwa94))),
                               OXD_MUL(np2, openmx_xc_dual_e1_scaled(hsbwa94))));
    const openmx_xc_dual t10 = OXD_SCALE(-0.5 * A, openmx_xc_dual_log1p(OXD_CDIV(D, hsbw)));
    term1 = OXD_ADD(t1, t10);
    if (w.v != 0.0) {
      const openmx_xc_dual sqrt_hsbw = OXD_SQRT(hsbw), sqrt_dhsbw = OXD_SQRT(dhsbw);
      const openmx_xc_dual hsbw_2 = OXD_MUL(hsbw, hsbw), hsbw_3 = OXD_MUL(hsbw_2, hsbw);
      const openmx_xc_dual hsbw_32 = OXD_MUL(hsbw, sqrt_hsbw), hsbw_52 = OXD_MUL(hsbw_2, sqrt_hsbw);
      const openmx_xc_dual dhsbw_2 = OXD_MUL(dhsbw, dhsbw), dhsbw_3 = OXD_MUL(dhsbw_2, dhsbw);
      const openmx_xc_dual dhsbw_4 = OXD_MUL(dhsbw_2, dhsbw_2);
      const openmx_xc_dual dhsbw_32 = OXD_MUL(dhsbw, sqrt_dhsbw), dhsbw_52 = OXD_MUL(dhsbw_2, sqrt_dhsbw);
      const openmx_xc_dual dhsbw_72 = OXD_MUL(dhsbw_3, sqrt_dhsbw);
      const openmx_xc_dual f2 = OXD_CDIV(0.5 * ea0 * sqrt_pi * A, sqrt_dhsbw);
      const openmx_xc_dual f3 = OXD_CDIV(0.5 * ea1 * A, dhsbw);
      const openmx_xc_dual f4 =
          OXD_SCALE(ea2 * sqrt_pi, OXD_ADD(OXD_CDIV(-9.0, OXD_SCALE(8.0, sqrt_hsbw)), OXD_CDIV(0.25 * A, dhsbw_32)));
      const openmx_xc_dual f5 = OXD_SCALE(ea3 / 128.0, OXD_ADD(OXD_CDIV(-144.0, hsbw), OXD_CDIV(64.0 * A, dhsbw_2)));
      const openmx_xc_dual f6 =
          OXD_DIV(OXD_SCALE(ea4, OXD_SCALE(3.0 * sqrt_pi,
                                           OXD_ADD(OXD_MUL(OXD_SCALE(3.0, dhsbw_52),
                                                           OXD_SUBC(OXD_SCALE(9.0, hsbw), 2.0 * A)),
                                                   OXD_SCALE(A * A, OXD_SCALE(4.0, hsbw_32))))),
                  OXD_SCALE(A, OXD_MUL(OXD_SCALE(32.0, dhsbw_52), hsbw_32)));
      const openmx_xc_dual f7 =
          OXD_DIVC(OXD_SCALE(ea5, OXD_ADD(OXD_CDIV(32.0 * A, dhsbw_3),
                                          OXD_DIV(OXD_ADDC(OXD_DIVC(OXD_MUL(OXD_SCALE(81.0, s2), h), A), -36.0),
                                                  hsbw_2))),
                   32.0);
      const openmx_xc_dual f8 =
          OXD_DIV(OXD_SCALE(ea6, OXD_SCALE(-3.0 * sqrt_pi,
                                           OXD_ADD(OXD_SCALE(A * A * A, OXD_SCALE(-40.0, hsbw_52)),
                                                   OXD_MUL(OXD_SCALE(9.0, dhsbw_72),
                                                           OXD_ADDC(OXD_SUB(OXD_SCALE(27.0, hsbw_2),
                                                                            OXD_SCALE(A, OXD_SCALE(6.0, hsbw))),
                                                                    4.0 * (A * A)))))),
                  OXD_SCALE(A * A, OXD_MUL(OXD_SCALE(128.0, dhsbw_72), hsbw_52)));
      const openmx_xc_dual f9 =
          OXD_DIV(OXD_ADD(OXD_SCALE(A, OXD_MUL(OXD_SCALE(324.0 * ea5 * eb1, dhsbw_4), hsbw)),
                          OXD_SCALE(ea7, OXD_ADD(OXD_SCALE(A * A * A, OXD_SCALE(384.0, hsbw_3)),
                                                 OXD_MUL(dhsbw_4,
                                                         OXD_SUBC(OXD_ADD(OXD_SCALE(-729.0, hsbw_2),
                                                                          OXD_SCALE(A, OXD_SCALE(324.0, hsbw))),
                                                                  288.0 * (A * A)))))),
                  OXD_SCALE(A * A, OXD_MUL(OXD_SCALE(128.0, dhsbw_4), hsbw_3)));
      const openmx_xc_dual series =
          OXD_ADD(OXD_ADD(OXD_ADD(OXD_ADD(OXD_ADD(OXD_ADD(OXD_ADD(OXD_MUL(f2, w), OXD_MUL(f3, w2)), OXD_MUL(f4, w3)),
                                                  OXD_MUL(f5, w4)),
                                          OXD_MUL(f6, w5)),
                                  OXD_MUL(f7, w6)),
                          OXD_MUL(f8, w7)),
                  OXD_MUL(f9, w8));
      term1 = OXD_ADD(term1, series);
    }
  }

  f = OXD_SCALE(-8.0 / 9.0, OXD_ADD(OXD_ADD(OXD_ADD(OXD_ADD(term1, term2), term3), term4), term5));
  e = OXD_MUL(OXD_SCALE(-exchange_prefactor, rho43), f);
  out[0] = e.v;
  out[1] = e.r;
  out[2] = e.g;
}

#undef OXD_ADD
#undef OXD_ADDC
#undef OXD_NEG
#undef OXD_SUB
#undef OXD_SUBC
#undef OXD_CSUB
#undef OXD_MUL
#undef OXD_SCALE
#undef OXD_DIV
#undef OXD_DIVC
#undef OXD_CDIV
#undef OXD_SQRT

/* Short-range omega-PBE exchange of both spin channels: ex[0] the energy
   density per particle, dexdd per spin, dexdgd [direction][spin]; omega in
   1/bohr.  The floors and the symmetrization are those of the exchange part
   of openmx_xc_kernel_pbe: each spin density is at least 0.5e-14 (dens is
   rewritten with the floored values), the norm of each spin gradient at
   least 0.5e-14, and the derivative with respect to the gradient vector
   points along the unfloored components.  The channel energy depends on the
   density and the norm of the gradient only: without spin polarization, and
   with the symmetrization, the two channels have the same ones, and the
   second channel takes the result of the first, the bits of a second
   call. */
OPENMX_XC_KERNEL void openmx_xc_kernel_wpbeh_short_range(double dens[2], double gdens[3][2], int symmetrize,
                                                         double omega, double ex[1], double dexdd[2],
                                                         double dexdgd[3][2])
{
  const double den_min = 1.0e-14, gd_min = 1.0e-14;
  double D[2], GD[3][2], floored_norm[2], channel[3] = {0.0, 0.0, 0.0}, energy = 0.0;
  int s, d;

  dens[0] = openmx_xc_kernel_largest(0.5*den_min, dens[0]);
  dens[1] = openmx_xc_kernel_largest(0.5*den_min, dens[1]);
  if (symmetrize) {
    D[0] = (dens[0] + dens[1])/2.0;
    D[1] = (dens[0] + dens[1])/2.0;
  }
  else {
    D[0] = dens[0];
    D[1] = dens[1];
  }
  for (d = 0; d < 3; ++d) {
    if (symmetrize) {
      GD[d][0] = (gdens[d][0] + gdens[d][1])/2.0;
      GD[d][1] = (gdens[d][0] + gdens[d][1])/2.0;
    }
    else {
      GD[d][0] = gdens[d][0];
      GD[d][1] = gdens[d][1];
    }
  }
  for (s = 0; s < 2; ++s) {
    const double norm = sqrt(GD[0][s]*GD[0][s] + GD[1][s]*GD[1][s] + GD[2][s]*GD[2][s]);
    floored_norm[s] = openmx_xc_kernel_largest(0.5*gd_min, norm);
  }
  for (s = 0; s < 2; ++s) {
    if (s == 0 || D[1] != D[0] || floored_norm[1] != floored_norm[0])
      openmx_xc_kernel_wpbeh_channel(D[s], floored_norm[s], omega, channel);
    energy += channel[0];
    dexdd[s] = channel[1];
    for (d = 0; d < 3; ++d) dexdgd[d][s] = channel[2]*(GD[d][s]/floored_norm[s]);
  }
  ex[0] = energy/(dens[0] + dens[1]);
}

/* Semilocal part of a screened hybrid functional (HSE06): the outputs of
   openmx_xc_kernel_pbe with fraction times the short-range exchange above
   removed from exc[0], dexdd and dexdgd.  A fraction of exactly 0 gives the
   outputs of openmx_xc_kernel_pbe bit for bit.  dens is floored as there. */
OPENMX_XC_KERNEL void openmx_xc_kernel_pbe_screened_hybrid(double dens[2], double gdens[3][2], int symmetrize,
                                                           double fraction, double omega,
                                                           double exc[2], double dexdd[2], double decdd[2],
                                                           double dexdgd[3][2], double decdgd[3][2])
{
  double sr_dens[2], sr_ex[1], sr_dexdd[2], sr_dexdgd[3][2];
  int s, d;

  sr_dens[0] = dens[0];
  sr_dens[1] = dens[1];
  openmx_xc_kernel_pbe(dens, gdens, symmetrize, exc, dexdd, decdd, dexdgd, decdgd);
  if (fraction == 0.0) return;
  openmx_xc_kernel_wpbeh_short_range(sr_dens, gdens, symmetrize, omega, sr_ex, sr_dexdd, sr_dexdgd);
  exc[0] -= fraction*sr_ex[0];
  for (s = 0; s < 2; ++s) {
    dexdd[s] -= fraction*sr_dexdd[s];
    for (d = 0; d < 3; ++d) dexdgd[d][s] -= fraction*sr_dexdgd[d][s];
  }
}

/* Semilocal part of a global hybrid functional (PBE0): the outputs of
   openmx_xc_kernel_pbe with exc[0], dexdd and dexdgd multiplied by
   1 - fraction.  A fraction of exactly 0 gives the outputs of
   openmx_xc_kernel_pbe bit for bit.  dens is floored as there. */
OPENMX_XC_KERNEL void openmx_xc_kernel_pbe_global_hybrid(double dens[2], double gdens[3][2], int symmetrize,
                                                         double fraction,
                                                         double exc[2], double dexdd[2], double decdd[2],
                                                         double dexdgd[3][2], double decdgd[3][2])
{
  double kept;
  int s, d;

  openmx_xc_kernel_pbe(dens, gdens, symmetrize, exc, dexdd, decdd, dexdgd, decdgd);
  if (fraction == 0.0) return;
  kept = 1.0 - fraction;
  exc[0] *= kept;
  for (s = 0; s < 2; ++s) {
    dexdd[s] *= kept;
    for (d = 0; d < 3; ++d) dexdgd[d][s] *= kept;
  }
}

#if defined(OPENMX_XC_KERNELS_OMP_TARGET)
#pragma omp end declare target
#endif

#endif

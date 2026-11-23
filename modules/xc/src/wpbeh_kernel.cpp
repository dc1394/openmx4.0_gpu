// Short-range omega-PBE exchange (wPBEh) of one spin channel: the exchange
// energy of the screened interaction erfc(omega r)/r for the PBE exchange
// hole, in the analytic form of J. Heyd.
//
// Source.  The energy expressions and the constants were transcribed from the
// routines pbexsr and wpbe_analy_erfc_approx_grad of Quantum ESPRESSO 7.6
// (XClib/qe_funct_exch_gga.f90, tag qe-7.6; Copyright (C) 2020 Quantum
// ESPRESSO group, distributed under the terms of the GNU General Public
// License), through the multi-precision reference tools/wpbeh_reference.py of
// this repository, which lists the papers.  This file is a new
// double-precision implementation and is distributed under the licence of
// this library, the GNU General Public License (see LICENSE).
//
//   kF    = (6 pi^2 rho_s)^(1/3)                  w = omega / kF
//   s_raw = |grad rho_s| / (2 (6 pi^2)^(1/3) rho_s^(4/3))
//   s     = s_raw up to 8.3, 8.572844 - 18.796223 / s_raw^2 above
//   e_s   = -(3/4) (6/pi)^(1/3) rho_s^(4/3) f(w, s)
//   f     = -(8/9) (term1 + term2 + term3 + term4 + term5)
//
// The constants are the published, rounded ones.  The hole function EG(s) is
// a three-term polynomial up to s = 0.08 and a closed expression above; up to
// w = 14, term1 uses a fit of erfc with eight coefficients, above one
// Gaussian.  f therefore has steps at s = 0.08 (up to 7e-9 of its value), at
// s_raw = 8.3 (up to 1e-10) and at w = 14 (up to 3e-4); the derivatives
// returned are those of the expression in use.
//
// Differences from the Fortran routine:
//   - The derivatives are those of the energy above, the rescaling of s
//     included.  The Fortran routine leaves the derivative of the rescaling
//     out of its potential above s_raw = 8.3.
//   - exp(x) E1(x) and exp(x^2) erfc(x) are evaluated to full precision for
//     every argument (special_functions.cpp).  The Fortran routine uses
//     asymptotic approximations above w = 14.
//   - s_raw below 1e-10 is replaced by 1e-10.  term1 has logarithmic
//     singularities at w = 0, s = 0 that cancel in the sum, and
//     f(w, s) - f(w, 0) is of the order s^2, so this changes f by less than
//     1e-19 of its value.  (At omega = 0 and a vanishing gradient the Fortran
//     routine returns f = 1; the formula tends to 1 - 4.8e-9.)
//
// The evaluation follows the double-precision form of the reference
// (double_channel_energy): the hole function EG(s) enters only through
// Q = E + s^2 EG(s), which is computed without a division by s^2, and
// log(Hsbw/DHsbw) is computed as -log1p(D/Hsbw).  Between w = 4 and w = 14
// the terms of term1 cancel and digits are lost; the tolerances of the
// fixture (tests/fixtures/wpbeh_reference.txt) describe the remaining
// accuracy.
//
// The derivatives with respect to rho_s and |grad rho_s| are carried along
// with the value (forward mode), so they have the rounding behaviour of the
// value and no separately transcribed formulas.

#include "wpbeh_kernel.hpp"

#include "special_functions.hpp"

#include <cmath>

namespace openmx::xc::detail {
namespace {

// A value with its partial derivatives with respect to the spin density (r)
// and the norm of its gradient (g).
struct Dual {
  double v, r, g;
};

constexpr Dual constant(double c) noexcept { return {c, 0.0, 0.0}; }
// f(a) with the derivative df = f'(a.v)
constexpr Dual chain(Dual a, double f, double df) noexcept { return {f, df * a.r, df * a.g}; }

constexpr Dual operator+(Dual a, Dual b) noexcept { return {a.v + b.v, a.r + b.r, a.g + b.g}; }
constexpr Dual operator+(Dual a, double b) noexcept { return {a.v + b, a.r, a.g}; }
constexpr Dual operator+(double a, Dual b) noexcept { return {a + b.v, b.r, b.g}; }
constexpr Dual operator-(Dual a) noexcept { return {-a.v, -a.r, -a.g}; }
constexpr Dual operator-(Dual a, Dual b) noexcept { return {a.v - b.v, a.r - b.r, a.g - b.g}; }
constexpr Dual operator-(Dual a, double b) noexcept { return {a.v - b, a.r, a.g}; }
constexpr Dual operator-(double a, Dual b) noexcept { return {a - b.v, -b.r, -b.g}; }
constexpr Dual operator*(Dual a, Dual b) noexcept { return {a.v * b.v, a.r * b.v + a.v * b.r, a.g * b.v + a.v * b.g}; }
constexpr Dual operator*(Dual a, double b) noexcept { return {a.v * b, a.r * b, a.g * b}; }
constexpr Dual operator*(double a, Dual b) noexcept { return {a * b.v, a * b.r, a * b.g}; }
constexpr Dual operator/(Dual a, Dual b) noexcept
{
  const double q = a.v / b.v;
  return {q, (a.r - q * b.r) / b.v, (a.g - q * b.g) / b.v};
}
constexpr Dual operator/(Dual a, double b) noexcept { return {a.v / b, a.r / b, a.g / b}; }
constexpr Dual operator/(double a, Dual b) noexcept
{
  const double q = a / b.v;
  return {q, -q * b.r / b.v, -q * b.g / b.v};
}

Dual sqrt(Dual a) noexcept { const double f = std::sqrt(a.v); return chain(a, f, 0.5 / f); }
Dual cbrt(Dual a) noexcept { const double f = std::cbrt(a.v); return chain(a, f, f / (3.0 * a.v)); }
Dual log1p(Dual a) noexcept { return chain(a, std::log1p(a.v), 1.0 / (1.0 + a.v)); }
Dual scaled_e1(Dual a) noexcept { const ValueAndDerivative f = detail::e1_scaled(a.v); return chain(a, f.value, f.derivative); }
Dual scaled_erfc(Dual a) noexcept { const ValueAndDerivative f = detail::erfcx(a.v); return chain(a, f.value, f.derivative); }

// PBE exchange hole: A, B, C, D, E, the coefficients of H(s), and F(s) = Fc1 H(s) + Fc2.
constexpr double A = 1.0161144, B = -0.37170836, C = -0.077215461, D = 0.57786348, E = -0.051955731;
constexpr double ha[5] = {0.00979681, 0.0410834, 0.187440, 0.00120824, 0.0347188};
constexpr double fc1 = 6.4753871, fc2 = 0.47965830;
// EG(s) = ega[0] + ega[1] s^2 + ega[2] s^4 up to s = eg_s_cut
constexpr double ega[3] = {-0.02628417880, -0.07117647788, 0.08534541323};
constexpr double eg_s_cut = 0.08;
// erfc(x) ~ (1 + sum_k ea[k] x^(k+1)) exp(-eb1 x^2) up to w = w_cut, exp(-2 x^2) above
constexpr double ea[8] = {-1.128223946706117, 1.452736265762971, -1.243162299390327, 0.971824836115601,
                          -0.568861079687373, 0.246880514820192, -0.065032363850763, 0.008401793031216};
constexpr double eb1_fit = 1.455915450052607, eb1_gaussian = 2.0, w_cut = 14.0;
// rescaling of the reduced gradient above s_trans, and the guard of this file at small s
constexpr double s_trans = 8.3, s_max = 8.572844, s_const = 18.796223, s_floor = 1.0e-10;

constexpr double pi = 3.14159265358979323846;
constexpr double sqrt_pi = 1.77245385090551602730;
constexpr double sqrt_A = 1.00802499968998784748;          // sqrt(A)
constexpr double A_32 = 1.02426871774499218765;            // A^(3/2)
constexpr double A_52 = 1.04077419357022208975;            // A^(5/2)
constexpr double cbrt_6pi2 = 3.89777708972075395896;       // (6 pi^2)^(1/3)
constexpr double exchange_prefactor = 0.930525736349100025002;  // (3/4) (6/pi)^(1/3)

}  // namespace

ChannelEnergy wpbeh_channel_energy(double density, double gradient_norm, double omega) noexcept
{
  const Dual rho{density, 1.0, 0.0}, gamma{gradient_norm, 0.0, 1.0};
  const Dual rho13 = cbrt(rho), rho43 = rho * rho13;
  const Dual w = omega / (cbrt_6pi2 * rho13);
  const Dual s_raw = gamma / (2.0 * cbrt_6pi2 * rho43);

  Dual s = s_raw;
  if (s_raw.v > s_trans) {
    s = s_max - s_const / (s_raw * s_raw);
  } else if (s_raw.v < s_floor) {
    s = constant(s_floor);
  }

  // The names follow the Fortran routine: Hsbw = s^2 H + eb1 w^2, DHsbw = D + Hsbw,
  // DHs = D + s^2 H, DHsw = DHs + w^2, HsbwA94 = 9 Hsbw / (4 A).
  const bool gaussian = w.v > w_cut;
  const double eb1 = gaussian ? eb1_gaussian : eb1_fit;
  const Dual s2 = s * s, s4 = s2 * s2;
  const Dual h = (ha[0] * s2 + ha[1] * s4) / (1.0 + ha[2] * s4 + ha[3] * s4 * s + ha[4] * s4 * s2);
  const Dual f_s = fc1 * h + fc2;
  const Dual w2 = w * w;
  const Dual dhs = D + s2 * h;
  const Dual dhsw = dhs + w2;
  const Dual hsbw = s2 * h + eb1 * w2;
  const Dual hsbwa94 = 9.0 * hsbw / (4.0 * A);
  const Dual dhsbw = D + hsbw;
  const Dual dhs_2 = dhs * dhs, dhs_3 = dhs_2 * dhs;

  // Q = E + s^2 EG(s)
  Dual q;
  if (s.v > eg_s_cut) {
    // EG = -(3 pi/4 + Ga)/Gb with Gb = 15 sqrt(pi) s^2 / (16 DHs^(7/2)), hence
    // Q = E - (3 pi/4 + Ga(s)) 16 DHs^(7/2) / (15 sqrt(pi))
    const Dual dhs_72 = dhs_3 * sqrt(dhs);
    const Dual ga = sqrt_pi * (15.0 * E + 6.0 * C * (1.0 + f_s * s2) * dhs + 4.0 * B * dhs_2 + 8.0 * A * dhs_3) / (16.0 * dhs_72)
                    - 0.75 * pi * sqrt_A * scaled_erfc(sqrt(9.0 * h * s2 / (4.0 * A)));
    q = E - (0.75 * pi + ga) * 16.0 * dhs_72 / (15.0 * sqrt_pi);
  } else {
    q = E + s2 * (ega[0] + ega[1] * s2 + ega[2] * s4);
  }

  const Dual dhsw_2 = dhsw * dhsw, dhsw_52 = dhsw_2 * sqrt(dhsw);
  const Dual w3 = w2 * w, w5 = w3 * w2;
  const Dual term2 = (dhs_2 * B + dhs * C + dhs * s2 * C * f_s + 2.0 * q) / (2.0 * dhs_3);
  const Dual term3 = -w * (4.0 * dhsw_2 * B + 6.0 * dhsw * C + 6.0 * dhsw * s2 * C * f_s + 15.0 * q) / (8.0 * dhs * dhsw_52);
  const Dual term4 = -w3 * (dhsw * C + dhsw * s2 * C * f_s + 5.0 * q) / (2.0 * dhs_2 * dhsw_52);
  const Dual term5 = -w5 * q / (dhs_3 * dhsw_52);

  // expei = exp(x) Ei(-x) = -exp(x) E1(x) at x = HsbwA94
  Dual term1;
  if (gaussian) {
    term1 = -0.5 * A * (-scaled_e1(hsbwa94) + log1p(D / hsbw));
  } else {
    const Dual w4 = w2 * w2, w6 = w3 * w3, w7 = w6 * w, w8 = w4 * w4;
    const Dual np1 = -1.5 * ea[0] * sqrt_A * w + 27.0 * ea[2] * w3 / (8.0 * sqrt_A)
                     - 243.0 * ea[4] * w5 / (32.0 * A_32) + 2187.0 * ea[6] * w7 / (128.0 * A_52);
    const Dual np2 = -A + 9.0 * ea[1] * w2 / 4.0 - 81.0 * ea[3] * w4 / (16.0 * A)
                     + 729.0 * ea[5] * w6 / (64.0 * A * A) - 6561.0 * ea[7] * w8 / (256.0 * A * A * A);
    const Dual t1 = 0.5 * (np1 * pi * scaled_erfc(sqrt(hsbwa94)) - np2 * scaled_e1(hsbwa94));
    const Dual t10 = -0.5 * A * log1p(D / hsbw);
    term1 = t1 + t10;
    if (w.v != 0.0) {
      const Dual sqrt_hsbw = sqrt(hsbw), sqrt_dhsbw = sqrt(dhsbw);
      const Dual hsbw_2 = hsbw * hsbw, hsbw_3 = hsbw_2 * hsbw, hsbw_32 = hsbw * sqrt_hsbw, hsbw_52 = hsbw_2 * sqrt_hsbw;
      const Dual dhsbw_2 = dhsbw * dhsbw, dhsbw_3 = dhsbw_2 * dhsbw, dhsbw_4 = dhsbw_2 * dhsbw_2;
      const Dual dhsbw_32 = dhsbw * sqrt_dhsbw, dhsbw_52 = dhsbw_2 * sqrt_dhsbw, dhsbw_72 = dhsbw_3 * sqrt_dhsbw;
      const Dual f2 = 0.5 * ea[0] * sqrt_pi * A / sqrt_dhsbw;
      const Dual f3 = 0.5 * ea[1] * A / dhsbw;
      const Dual f4 = ea[2] * sqrt_pi * (-9.0 / (8.0 * sqrt_hsbw) + 0.25 * A / dhsbw_32);
      const Dual f5 = (ea[3] / 128.0) * (-144.0 / hsbw + 64.0 * A / dhsbw_2);
      const Dual f6 = ea[4] * (3.0 * sqrt_pi * (3.0 * dhsbw_52 * (9.0 * hsbw - 2.0 * A) + 4.0 * hsbw_32 * (A * A)))
                      / (32.0 * dhsbw_52 * hsbw_32 * A);
      const Dual f7 = ea[5] * (32.0 * A / dhsbw_3 + (-36.0 + 81.0 * s2 * h / A) / hsbw_2) / 32.0;
      const Dual f8 = ea[6] * (-3.0 * sqrt_pi * (-40.0 * hsbw_52 * (A * A * A)
                                                 + 9.0 * dhsbw_72 * (27.0 * hsbw_2 - 6.0 * hsbw * A + 4.0 * (A * A))))
                      / (128.0 * dhsbw_72 * hsbw_52 * (A * A));
      const Dual f9 = (324.0 * ea[5] * eb1 * dhsbw_4 * hsbw * A
                       + ea[7] * (384.0 * hsbw_3 * (A * A * A) + dhsbw_4 * (-729.0 * hsbw_2 + 324.0 * hsbw * A - 288.0 * (A * A))))
                      / (128.0 * dhsbw_4 * hsbw_3 * (A * A));
      term1 = term1 + (f2 * w + f3 * w2 + f4 * w3 + f5 * w4 + f6 * w5 + f7 * w6 + f8 * w7 + f9 * w8);
    }
  }

  const Dual f = (-8.0 / 9.0) * (term1 + term2 + term3 + term4 + term5);
  const Dual e = -exchange_prefactor * rho43 * f;
  return {e.v, e.r, e.g};
}

}  // namespace openmx::xc::detail

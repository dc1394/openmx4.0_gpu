#!/usr/bin/env python3
# The energy expressions and the constants evaluated here were transcribed from
# the routines pbexsr and wpbe_analy_erfc_approx_grad of Quantum ESPRESSO 7.6
# (XClib/qe_funct_exch_gga.f90, tag qe-7.6; Copyright (C) 2020 Quantum ESPRESSO
# group, distributed under the terms of the GNU General Public License).  This
# file is a new multi-precision implementation written for testing.  It is
# distributed under the licence of this repository, the GNU General Public
# License (see LICENSE in the root directory).
"""
wpbeh_reference.py -- multi-precision reference ("oracle") for the short-range
omega-PBE exchange (wPBEh) of HSE06, milestone M1 of EXX_IMPLEMENTATION_GUIDE.md
(OpenMX 4.0 GPU).

What it evaluates
  The exchange energy of the screened interaction erfc(omega r)/r for the PBE exchange
  hole, in the analytic form of J. Heyd, with the constants and the branch rules of
  Quantum ESPRESSO 7.6 (pbexsr and wpbe_analy_erfc_approx_grad).  The special functions
  are evaluated with 50 significant digits (mpmath); the approximations that Quantum
  ESPRESSO uses for them at large arguments are not part of the definition.  The file
  calls neither Quantum ESPRESSO nor any other electronic-structure code.

Definition (one spin channel with density rho_s and gradient grad rho_s)
  kF    = (6 pi^2 rho_s)^(1/3)                     w = omega / kF
  s_raw = |grad rho_s| / (2 (6 pi^2)^(1/3) rho_s^(4/3))
  s     = s_raw                          for s_raw <= 8.3
          8.572844 - 18.796223 / s_raw^2 for s_raw >  8.3
  e_s   = -(3/4) (6/pi)^(1/3) rho_s^(4/3) f(w, s)                 energy per volume
  e     = e_up + e_down
  f(w, s) = -(8/9) (term1 + term2 + term3 + term4 + term5) is the enhancement factor.
  Its constants are the published, rounded ones (E, Fc1 and Fc2 included).  The hole
  function EG(s) is a three-term polynomial for s <= 0.08 and a closed expression above.
  Up to w = 14, term1 uses a fit of erfc with eight coefficients; above, one Gaussian.
  Quantum ESPRESSO applies the same expressions to 2 rho_s and 4 |grad rho_s|^2 and
  halves the energy, which gives the formulas above.

  One guard is not from Quantum ESPRESSO: s_raw below 1e-10 is replaced by 1e-10.  The
  terms have logarithmic singularities at w = 0, s = 0 that cancel in the sum, and
  f(w, s) - f(w, 0) is of the order s^2, so the guard changes f by less than 1e-19.
  (Quantum ESPRESSO returns f = 1 at w = 0, s = 0; the limit of its formula is 1 - 4.8e-9.)

Output conventions (those of openmx_xc_kernel_pbe in modules/xc)
  exc      e / (rho_up + rho_down)
  dexdd    d e / d rho_s
  dexdgd   d e / d (d_i rho_s)
  The derivatives are those of the energy defined above, the rescaling of s included.
  Quantum ESPRESSO 7.6 leaves the derivative of the rescaling out of its potential above
  s_raw = 8.3 (--compare-qe accounts for it).

Usage
  python3 wpbeh_reference.py                   # self-test
  python3 wpbeh_reference.py --report          # properties of the definition
  python3 wpbeh_reference.py --dump FILE       # fixture for the unit tests
  python3 wpbeh_reference.py --point RU RD GUX GUY GUZ GDX GDY GDZ   # one point, with intermediates
  python3 wpbeh_reference.py --double          # double-precision prototype against the fixture (NumPy, SciPy)
  python3 wpbeh_reference.py --qe-files DIR    # write a driver for pbexsr of Quantum ESPRESSO and its input
  python3 wpbeh_reference.py --compare-qe FILE # compare the output of that driver with this file (NumPy)

Sources
  J. Heyd, G. E. Scuseria and M. Ernzerhof, J. Chem. Phys. 118, 8207 (2003): the screened
      exchange hole (eq. 8) and the enhancement factor as an integral over it (eq. 9).
  J. Heyd, "Screened Coulomb Hybrid Density Functionals", PhD thesis, Rice University (2004):
      the fit of erfc and the Gaussian above 14 (section 2.2.2), the rescaling of s that
      keeps the local Lieb-Oxford bound (section 2.2.3).
  M. Ernzerhof and J. P. Perdew, J. Chem. Phys. 109, 3313 (1998): the PBE exchange hole,
      cited by the first source.
  Quantum ESPRESSO 7.6, XClib/qe_funct_exch_gga.f90: every constant and every expression of
      this file was read there.

How the closed form was checked
  The self-test integrates the model exchange hole numerically and compares the result with
  the closed form (T12).  --qe-files and --compare-qe compare the Fortran routine of Quantum
  ESPRESSO with this file at the fixture points; the routine is compiled and run by hand.
"""
import argparse
import os
import sys

from mpmath import mp, mpf, sqrt, pi, exp, erf, erfc, e1, log, cbrt, diff, nstr, quad, inf

mp.dps = 50
OMEGA_HSE06 = "0.11"

# ----------------------------------------------------------------------------
# constants of wpbe_analy_erfc_approx_grad
# erfc(x) ~ (1 + sum_k EA[k] x^(k+1)) exp(-EB1 x^2)
EA = [mpf(x) for x in ("-1.128223946706117", "1.452736265762971", "-1.243162299390327",
                       "0.971824836115601", "-0.568861079687373", "0.246880514820192",
                       "-0.065032363850763", "0.008401793031216")]
EB1 = mpf("1.455915450052607")
WCUTOFF = 14                   # above: one Gaussian with the exponent 2 in place of the fit
# PBE exchange hole
A = mpf("1.0161144")
B = mpf("-0.37170836")
C = mpf("-0.077215461")
D = mpf("0.57786348")
E = mpf("-0.051955731")
HA = [mpf(x) for x in ("0.00979681", "0.0410834", "0.187440", "0.00120824", "0.0347188")]
FC1, FC2 = mpf("6.4753871"), mpf("0.47965830")
# EG at small s
EGA = [mpf(x) for x in ("-0.02628417880", "-0.07117647788", "0.08534541323")]
EGSCUT = mpf("0.08")
# constants of pbexsr: rescaling of the reduced gradient
S_TRANS, S_MAX, S_CONST = mpf("8.3"), mpf("8.572844"), mpf("18.796223")
# guard of this implementation (see the definition above)
S_FLOOR = mpf("1e-10")
# PBE exchange, for the comparison with the omega = 0 limit
KAPPA = mpf("0.804")
MU = mpf("0.06672455060314922") * pi ** 2 / 3


def erfcx(u):
    return exp(u * u) * erfc(u)


def e1_scaled(x):
    return exp(x) * e1(x)


# ----------------------------------------------------------------------------
# enhancement factor
def hole_h(s):
    return (HA[0] * s ** 2 + HA[1] * s ** 4) / (1 + HA[2] * s ** 4 + HA[3] * s ** 5 + HA[4] * s ** 6)


def hole_f(s):
    return FC1 * hole_h(s) + FC2


def g_a(s, e_const=None):
    """G_a(s); e_const replaces the constant E (used by the self-test)."""
    e_const = E if e_const is None else e_const
    s2 = s * s
    h = hole_h(s)
    dhs = D + s2 * h
    return (sqrt(pi) * (15 * e_const + 6 * C * (1 + hole_f(s) * s2) * dhs + 4 * B * dhs ** 2 + 8 * A * dhs ** 3)
            / (16 * dhs ** (mpf(7) / 2)) - (3 * pi / 4) * sqrt(A) * erfcx(sqrt(9 * h * s2 / (4 * A))))


def g_b(s):
    return 15 * sqrt(pi) * s * s / (16 * (D + s * s * hole_h(s)) ** (mpf(7) / 2))


def hole_eg_closed(s, e_const=None):
    """EG(s) = -(3 pi/4 + G_a)/G_b, the expression used above s = 0.08."""
    return -(3 * pi / 4 + g_a(s, e_const)) / g_b(s)


def hole_eg(s):
    if s > EGSCUT:
        return hole_eg_closed(s)
    return EGA[0] + EGA[1] * s ** 2 + EGA[2] * s ** 4


def enhancement_terms(w, s):
    """Terms of f(w, s) = -(8/9) (term1 + term2 + term3 + term4 + term5); s is the rescaled gradient.

    The names follow the Fortran routine: Hsbw = s^2 H + eb1 w^2, DHsbw = D + Hsbw,
    DHs = D + s^2 H, DHsw = DHs + w^2, HsbwA94 = 9 Hsbw / (4 A),
    piexperf = pi exp(x) erfc(sqrt(x)) and expei = -exp(x) E1(x) at x = HsbwA94."""
    w, s = mpf(w), mpf(s)
    if w == 0 and s == 0:
        raise ValueError("w = 0 and s = 0: the terms are singular; channel_energy never passes this point")
    eb1 = mpf(2) if w > WCUTOFF else EB1
    h = hole_h(s)
    f_s = hole_f(s)
    eg = hole_eg(s)
    s2 = s * s
    hsbw = s2 * h + eb1 * w * w
    dhsbw = D + hsbw
    hsbwa94 = 9 * hsbw / (4 * A)
    dhs = D + s2 * h
    dhsw = dhs + w * w
    dhsw52 = dhsw ** (mpf(5) / 2)
    term2 = (dhs ** 2 * B + dhs * C + 2 * E + dhs * s2 * C * f_s + 2 * s2 * eg) / (2 * dhs ** 3)
    term3 = -w * (4 * dhsw ** 2 * B + 6 * dhsw * C + 15 * E + 6 * dhsw * s2 * C * f_s + 15 * s2 * eg) / (8 * dhs * dhsw52)
    term4 = -w ** 3 * (dhsw * C + 5 * E + dhsw * s2 * C * f_s + 5 * s2 * eg) / (2 * dhs ** 2 * dhsw52)
    term5 = -w ** 5 * (E + s2 * eg) / (dhs ** 3 * dhsw52)
    expei = -e1_scaled(hsbwa94)
    if w > WCUTOFF:
        term1 = -A / 2 * (expei + log(dhsbw) - log(hsbw))
    else:
        piexperf = pi * erfcx(sqrt(hsbwa94))
        np1 = (-mpf("1.5") * EA[0] * sqrt(A) * w + 27 * EA[2] * w ** 3 / (8 * sqrt(A))
               - 243 * EA[4] * w ** 5 / (32 * A ** (mpf(3) / 2)) + 2187 * EA[6] * w ** 7 / (128 * A ** (mpf(5) / 2)))
        np2 = (-A + 9 * EA[1] * w ** 2 / 4 - 81 * EA[3] * w ** 4 / (16 * A)
               + 729 * EA[5] * w ** 6 / (64 * A ** 2) - 6561 * EA[7] * w ** 8 / (256 * A ** 3))
        t1 = (np1 * piexperf + np2 * expei) / 2
        t10 = A / 2 * log(hsbw / dhsbw)
        t2t9 = mpf(0)
        if w != 0:
            f2 = EA[0] * sqrt(pi) * A / sqrt(dhsbw) / 2
            f3 = EA[1] * A / dhsbw / 2
            f4 = EA[2] * sqrt(pi) * (-mpf(9) / (8 * sqrt(hsbw)) + A / (4 * dhsbw ** (mpf(3) / 2)))
            f5 = (EA[3] / 128) * (-144 / hsbw + 64 * A / dhsbw ** 2)
            f6 = EA[4] * (3 * sqrt(pi) * (3 * dhsbw ** (mpf(5) / 2) * (9 * hsbw - 2 * A) + 4 * hsbw ** (mpf(3) / 2) * A ** 2)) \
                / (32 * dhsbw ** (mpf(5) / 2) * hsbw ** (mpf(3) / 2) * A)
            f7 = EA[5] * (32 * A / dhsbw ** 3 + (-36 + 81 * s2 * h / A) / hsbw ** 2) / 32
            f8 = EA[6] * (-3 * sqrt(pi) * (-40 * hsbw ** (mpf(5) / 2) * A ** 3
                                           + 9 * dhsbw ** (mpf(7) / 2) * (27 * hsbw ** 2 - 6 * hsbw * A + 4 * A ** 2))) \
                / (128 * dhsbw ** (mpf(7) / 2) * hsbw ** (mpf(5) / 2) * A ** 2)
            f9 = (324 * EA[5] * eb1 * dhsbw ** 4 * hsbw * A
                  + EA[7] * (384 * hsbw ** 3 * A ** 3 + dhsbw ** 4 * (-729 * hsbw ** 2 + 324 * hsbw * A - 288 * A ** 2))) \
                / (128 * dhsbw ** 4 * hsbw ** 3 * A ** 2)
            t2t9 = (f2 * w + f3 * w ** 2 + f4 * w ** 3 + f5 * w ** 4 + f6 * w ** 5 + f7 * w ** 6
                    + f8 * w ** 7 + f9 * w ** 8)
        term1 = t1 + t2t9 + t10
    return {"H": h, "F": f_s, "EG": eg, "term1": term1, "term2": term2, "term3": term3,
            "term4": term4, "term5": term5}


def enhancement(w, s):
    t = enhancement_terms(w, s)
    return -mpf(8) / 9 * (t["term1"] + t["term2"] + t["term3"] + t["term4"] + t["term5"])


def rescale(s_raw):
    s_raw = mpf(s_raw)
    if s_raw > S_TRANS:
        return S_MAX - S_CONST / s_raw ** 2
    return max(s_raw, S_FLOOR)


def f_pbe(s):
    return 1 + KAPPA - KAPPA / (1 + MU * mpf(s) ** 2 / KAPPA)


def f_lda_sr(w):
    """Exact short-range attenuation of LDA exchange for erfc(omega r)/r, a = omega/(2 kF)."""
    a = mpf(w) / 2
    if a == 0:
        return mpf(1)
    b = exp(-1 / (4 * a * a)) - 1
    return 1 - mpf(8) / 3 * a * (sqrt(pi) * erf(1 / (2 * a)) + 2 * a * (b - (2 * a * a * b + mpf(1) / 2)))


# ----------------------------------------------------------------------------
# energy density and derivatives
def channel_variables(rho_s, gamma_s, omega):
    k_f = cbrt(6 * pi ** 2 * rho_s)
    s_raw = gamma_s / (2 * cbrt(6 * pi ** 2) * rho_s ** (mpf(4) / 3))
    return omega / k_f, s_raw


def channel_energy(rho_s, gamma_s, omega):
    """Short-range exchange energy per volume of one spin channel; gamma_s = |grad rho_s|."""
    w, s_raw = channel_variables(rho_s, gamma_s, omega)
    return -mpf(3) / 4 * cbrt(6 / pi) * rho_s ** (mpf(4) / 3) * enhancement(w, rescale(s_raw))


def tolerance(w):
    """Relative tolerance for a double-precision implementation: 100 times an upper envelope of the
    rounding error of the direct formula (largest between w = 4 and w = 14, where the terms cancel).
    Above w = 14 the value is accurate to about 1e-11, the gradient derivative to about 1e-9."""
    w = float(w)
    if w > WCUTOFF:
        return 1e-9
    if w <= 4.0:
        return max(1e-12, 1e-13 * 10.0 ** (1.5 * w))
    return 1e-7 * 10.0 ** (0.5 * (w - 4.0))


def rounding_error(w, s):
    """Relative error of a 16-digit evaluation of the direct formula."""
    ref = enhancement(w, s)
    old = mp.dps
    mp.dps = 16
    try:
        val = +enhancement(w, s)
    finally:
        mp.dps = old
    return abs(val - ref) / abs(ref)


def evaluate(rho, grad, omega=OMEGA_HSE06):
    """rho = (up, down); grad = ((x, y, z) of up, (x, y, z) of down).  Returns a dict of mpf values."""
    omega = mpf(omega)
    out = {"e_s": [], "dexdd": [], "dexdgd": [], "vsigma": [], "w": [], "s_raw": [], "s": [], "tol": []}
    for sp in range(2):
        r = mpf(rho[sp])
        g = [mpf(x) for x in grad[sp]]
        gamma = sqrt(g[0] ** 2 + g[1] ** 2 + g[2] ** 2)
        if gamma == 0:
            raise ValueError("zero gradient: the derivative with respect to the gradient is taken per direction")
        w, s_raw = channel_variables(r, gamma, omega)
        de_dgamma = diff(lambda x: channel_energy(r, x, omega), gamma)
        out["e_s"].append(channel_energy(r, gamma, omega))
        out["dexdd"].append(diff(lambda x: channel_energy(x, gamma, omega), r))
        out["dexdgd"].append([de_dgamma * g[i] / gamma for i in range(3)])
        out["vsigma"].append(de_dgamma / (2 * gamma))
        out["w"].append(w)
        out["s_raw"].append(s_raw)
        out["s"].append(rescale(s_raw))
        out["tol"].append(tolerance(w))
    out["e"] = out["e_s"][0] + out["e_s"][1]
    out["exc"] = out["e"] / (mpf(rho[0]) + mpf(rho[1]))
    return out


# ----------------------------------------------------------------------------
# fixture
SPIN_DENSITIES = ("5", "0.5", "0.05", "5e-3", "5e-4", "5e-5", "5e-6", "5e-7", "4.4e-8", "5e-9")
# both sides of the two changes of the formula (s = 0.08 and s_raw = 8.3) are included
REDUCED_GRADIENTS = ("0.003", "0.05", "0.079", "0.081", "0.13", "0.4", "0.9", "1.3",
                     "1.7", "3", "6", "8.29", "8.31", "9", "15", "40")
DIRECTIONS = (("0.6", "-0.48", "0.64"), ("-0.28", "0.96", "0"))
COLUMNS = ("rho_up rho_dn gup_x gup_y gup_z gdn_x gdn_y gdn_z e_up e_dn exc dexdd_up dexdd_dn "
           "dexdgd_x_up dexdgd_y_up dexdgd_z_up dexdgd_x_dn dexdgd_y_dn dexdgd_z_dn vsigma_upup vsigma_dndn "
           "w_up s_up w_dn s_dn tol_up tol_dn").split()


def fixture_points():
    """Each (density, reduced gradient) pair appears once in the up and once in the down channel."""
    combos = [(mpf(r), mpf(s)) for r in SPIN_DENSITIES for s in REDUCED_GRADIENTS]
    n = len(combos)
    points = []
    for i in range(n):
        chans = (combos[i], combos[(7 * i + 3) % n])
        rho, grad = [], []
        for sp, (r, s_raw) in enumerate(chans):
            gamma = s_raw * 2 * cbrt(6 * pi ** 2) * r ** (mpf(4) / 3)
            sign = -1 if (i + sp) % 2 else 1
            # the reference values belong to the double-precision inputs that are written to the file
            rho.append(mpf(float(r)))
            grad.append([mpf(float(sign * gamma * mpf(c))) for c in DIRECTIONS[sp]])
        points.append((rho, grad))
    return points


def fixture_rows(omega=OMEGA_HSE06):
    rows = []
    for rho, grad in fixture_points():
        v = evaluate(rho, grad, omega)
        rows.append([rho[0], rho[1]] + grad[0] + grad[1] + v["e_s"] + [v["exc"]] + v["dexdd"]
                    + v["dexdgd"][0] + v["dexdgd"][1] + v["vsigma"]
                    + [v["w"][0], v["s"][0], v["w"][1], v["s"][1], mpf(v["tol"][0]), mpf(v["tol"][1])])
    return rows


def write_fixture(path, omega=OMEGA_HSE06):
    rows = fixture_rows(omega)
    with open(path, "w") as fh:
        fh.write("wpbeh_fixture 2\n")
        fh.write("definition short-range omega-PBE exchange in the analytic form of J. Heyd, constants and branch rules "
                 "of Quantum ESPRESSO 7.6 (evaluated with %d digits, not an output of Quantum ESPRESSO)\n" % mp.dps)
        fh.write("omega %s\n" % omega)
        fh.write("npoints %d\n" % len(rows))
        fh.write("ncolumns %d\n" % len(COLUMNS))
        fh.write("# " + " ".join(COLUMNS) + "\n")
        fh.write("# e_up + e_dn is the energy per volume; s_up and s_dn are the rescaled reduced gradients;\n")
        fh.write("# tol_* is the relative tolerance of that channel: |d e| <= tol_up |e_up| + tol_dn |e_dn|,\n")
        fh.write("# |d dexdd_s| <= 10 tol_s |dexdd_s|, |d dexdgd_i_s| <= 10 tol_s max(|dexdgd_s|, 1e-6 |e_s| / |grad rho_s|)\n")
        for row in rows:
            fh.write(" ".join("%.17e" % float(x) for x in row) + "\n")
    if read_fixture(path) != [[float(x) for x in row] for row in rows]:
        raise RuntimeError("the fixture does not read back identically")
    return len(rows)


def read_fixture(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if parts and parts[0][0] in "+-0123456789":
                rows.append([float(x) for x in parts])
    return rows


# ----------------------------------------------------------------------------
# the defining integral (independent of the closed form)
def hole_integrals(w, s):
    """f(w, s) = -(8/9) (I1 + I2) by numerical quadrature of the model exchange hole.

    I1 = int dy y [-A/(y^2 (1 + 4A y^2/9)) + A exp(-D y^2)/y^2] exp(-s^2 H y^2) erfc_fit(w y)
    I2 = int dy y [B + C (1 + s^2 F) y^2 + (E + s^2 EG) y^4] exp(-(D + s^2 H) y^2) erfc(w y)
    erfc_fit(x) = (1 + sum_k ea_k x^k) exp(-eb1 x^2) up to w = 14 and exp(-2 x^2) above.
    Returns (I1, I2); the closed form has term1 = I1 and term2 + ... + term5 = I2.
    """
    w, s = mpf(w), mpf(s)
    h = hole_h(s)
    f_s = hole_f(s)
    q = E + s * s * hole_eg(s)

    def erfc_fit(x):
        if w > WCUTOFF:
            return exp(-2 * x * x)
        return (1 + sum(EA[k] * x ** (k + 1) for k in range(8))) * exp(-EB1 * x * x)

    def i1(y):
        if y == 0:
            return mpf(0)
        return (y * (-A / (y * y * (1 + 4 * A * y * y / 9)) + A * exp(-D * y * y) / (y * y))
                * exp(-s * s * h * y * y) * erfc_fit(w * y))

    def i2(y):
        return y * (B + C * (1 + s * s * f_s) * y * y + q * y ** 4) * exp(-(D + s * s * h) * y * y) * erfc(w * y)

    pts = [0, mpf("0.05"), mpf("0.3"), 1, 3, 10, 40, 200, inf]
    return quad(i1, pts), quad(i2, pts)


def hole_integral_deviation(points):
    """Largest |closed form - quadrature| / |f| over the given (w, s) pairs, at 30 digits."""
    old = mp.dps
    mp.dps = 30
    try:
        worst = mpf(0)
        for w, s in points:
            a, b = hole_integrals(w, s)
            t = enhancement_terms(w, s)
            f = enhancement(w, s)
            worst = max(worst, abs(a - t["term1"]) / abs(f),
                        abs(b - (t["term2"] + t["term3"] + t["term4"] + t["term5"])) / abs(f))
        return worst
    finally:
        mp.dps = old


# ----------------------------------------------------------------------------
# double-precision prototype
def double_channel_energy(rho, gamma, omega):
    """Double precision (complex arguments allowed, for complex-step derivatives).

    The hole function EG enters only as Q = E + s^2 EG.  Above s = 0.08, Q is evaluated as
    E - (3 pi/4 + Ga(s)) 16 DHs^(7/2) / (15 sqrt(pi)), without a division by s^2; up to 0.08,
    from the polynomial.  log(Hsbw/DHsbw) is evaluated as -log1p(D/Hsbw).
    """
    import numpy as np
    from scipy.special import erfcx as np_erfcx, exp1

    a_, b_, c_, d_, e_ = float(A), float(B), float(C), float(D), float(E)
    ha = [float(x) for x in HA]
    ea = [float(x) for x in EA]
    ega = [float(x) for x in EGA]
    fc1, fc2 = float(FC1), float(FC2)
    sqpi = np.sqrt(np.pi)

    def e1s(x):
        if abs(x) < 30.0:
            return np.exp(x) * exp1(x)
        val = 0.0
        for k in range(60, 0, -1):      # continued fraction 1/(x + 1 - 1/(x + 3 - 4/(x + 5 - ...)))
            val = k * k / (x + 2 * k + 1 - val)
        return 1.0 / (x + 1 - val)

    kf = (6 * np.pi ** 2 * rho) ** (1.0 / 3.0)
    s = gamma / (2 * (6 * np.pi ** 2) ** (1.0 / 3.0) * rho ** (4.0 / 3.0))
    w = omega / kf
    if s.real > float(S_TRANS):
        s = float(S_MAX) - float(S_CONST) / (s * s)
    elif s.real < float(S_FLOOR):
        s = float(S_FLOOR) + 0 * s
    eb1 = 2.0 if w.real > WCUTOFF else float(EB1)
    h = (ha[0] * s ** 2 + ha[1] * s ** 4) / (1 + ha[2] * s ** 4 + ha[3] * s ** 5 + ha[4] * s ** 6)
    f_s = fc1 * h + fc2
    s2 = s * s
    dhs = d_ + s2 * h
    dhsw = dhs + w * w
    hsbw = s2 * h + eb1 * w * w
    hsbwa94 = 9 * hsbw / (4 * a_)
    dhsbw = d_ + hsbw
    if s.real > float(EGSCUT):
        dhs72 = dhs ** 3.5
        ga = sqpi * (15 * e_ + 6 * c_ * (1 + f_s * s2) * dhs + 4 * b_ * dhs ** 2 + 8 * a_ * dhs ** 3) / (16 * dhs72) \
            - 0.75 * np.pi * np.sqrt(a_) * np_erfcx(np.sqrt(9 * h * s2 / (4 * a_)))
        q = e_ - (0.75 * np.pi + ga) * 16 * dhs72 / (15 * sqpi)
    else:
        q = e_ + s2 * (ega[0] + ega[1] * s2 + ega[2] * s2 * s2)
    p52 = dhsw ** 2.5
    term2 = (dhs ** 2 * b_ + dhs * c_ + dhs * s2 * c_ * f_s + 2 * q) / (2 * dhs ** 3)
    term3 = -w * (4 * dhsw ** 2 * b_ + 6 * dhsw * c_ + 6 * dhsw * s2 * c_ * f_s + 15 * q) / (8 * dhs * p52)
    term4 = -w ** 3 * (dhsw * c_ + dhsw * s2 * c_ * f_s + 5 * q) / (2 * dhs ** 2 * p52)
    term5 = -w ** 5 * q / (dhs ** 3 * p52)
    if w.real > WCUTOFF:
        term1 = -a_ / 2 * (-e1s(hsbwa94) + np.log1p(d_ / hsbw))
    else:
        sa = np.sqrt(a_)
        np1 = (-1.5 * ea[0] * sa * w + 27 * ea[2] * w ** 3 / (8 * sa) - 243 * ea[4] * w ** 5 / (32 * a_ ** 1.5)
               + 2187 * ea[6] * w ** 7 / (128 * a_ ** 2.5))
        np2 = (-a_ + 9 * ea[1] * w ** 2 / 4 - 81 * ea[3] * w ** 4 / (16 * a_) + 729 * ea[5] * w ** 6 / (64 * a_ ** 2)
               - 6561 * ea[7] * w ** 8 / (256 * a_ ** 3))
        t1 = 0.5 * (np1 * np.pi * np_erfcx(np.sqrt(hsbwa94)) - np2 * e1s(hsbwa94))
        t10 = -0.5 * a_ * np.log1p(d_ / hsbw)
        f2 = 0.5 * ea[0] * sqpi * a_ / np.sqrt(dhsbw)
        f3 = 0.5 * ea[1] * a_ / dhsbw
        f4 = ea[2] * sqpi * (-9 / (8 * np.sqrt(hsbw)) + 0.25 * a_ / dhsbw ** 1.5)
        f5 = (ea[3] / 128) * (-144 / hsbw + 64 * a_ / dhsbw ** 2)
        f6 = ea[4] * (3 * sqpi * (3 * dhsbw ** 2.5 * (9 * hsbw - 2 * a_) + 4 * hsbw ** 1.5 * a_ ** 2)) \
            / (32 * dhsbw ** 2.5 * hsbw ** 1.5 * a_)
        f7 = ea[5] * (32 * a_ / dhsbw ** 3 + (-36 + 81 * s2 * h / a_) / hsbw ** 2) / 32
        f8 = ea[6] * (-3 * sqpi * (-40 * hsbw ** 2.5 * a_ ** 3
                                   + 9 * dhsbw ** 3.5 * (27 * hsbw ** 2 - 6 * hsbw * a_ + 4 * a_ ** 2))) \
            / (128 * dhsbw ** 3.5 * hsbw ** 2.5 * a_ ** 2)
        f9 = (324 * ea[5] * eb1 * dhsbw ** 4 * hsbw * a_
              + ea[7] * (384 * hsbw ** 3 * a_ ** 3 + dhsbw ** 4 * (-729 * hsbw ** 2 + 324 * hsbw * a_ - 288 * a_ ** 2))) \
            / (128 * dhsbw ** 4 * hsbw ** 3 * a_ ** 2)
        term1 = t1 + t10 + (f2 * w + f3 * w ** 2 + f4 * w ** 3 + f5 * w ** 4 + f6 * w ** 5 + f7 * w ** 6
                            + f8 * w ** 7 + f9 * w ** 8)
    f = -8.0 / 9.0 * (term1 + term2 + term3 + term4 + term5)
    return -0.75 * (6 / np.pi) ** (1.0 / 3.0) * rho ** (4.0 / 3.0) * f


def double_check(omega=OMEGA_HSE06):
    """Double-precision prototype against the fixture values, with the tolerances of the fixture."""
    import numpy as np
    rows = [[float(x) for x in row] for row in fixture_rows(omega)]
    col = {name: i for i, name in enumerate(COLUMNS)}
    om = float(omega)
    classes = {}
    worst = 0.0
    for row in rows:
        for sp, name in enumerate(("up", "dn")):
            rho = row[sp]
            g = np.array(row[2 + 3 * sp:5 + 3 * sp])
            gamma = float(np.sqrt(np.sum(g * g)))
            tol = row[col["tol_" + name]]
            e = double_channel_energy(complex(rho), complex(gamma), om).real
            de_dr = double_channel_energy(complex(rho, rho * 1e-30), complex(gamma), om).imag / (rho * 1e-30)
            de_dg = double_channel_energy(complex(rho), complex(gamma, gamma * 1e-30), om).imag / (gamma * 1e-30)
            e_ref, d_ref = row[col["e_" + name]], row[col["dexdd_" + name]]
            g_ref = np.array([row[col["dexdgd_%s_%s" % (ax, name)]] for ax in "xyz"])
            scale = max(float(np.sqrt(np.sum(g_ref * g_ref))), 1e-6 * abs(e_ref) / gamma)
            dev = (abs(e - e_ref) / abs(e_ref), abs(de_dr - d_ref) / abs(d_ref),
                   float(np.max(np.abs(de_dg * g / gamma - g_ref))) / scale)
            c = classes.setdefault(tol, [0.0, 0.0, 0.0])
            for i in range(3):
                c[i] = max(c[i], dev[i])
            worst = max(worst, dev[0] / tol, dev[1] / (10 * tol), dev[2] / (10 * tol))
    print("largest relative deviation of the double-precision prototype from the fixture, by tolerance class")
    print("   tol        energy     d/d rho    d/d grad (on the scale given in the fixture)")
    for tol in sorted(classes):
        print("   %.1e    %.1e    %.1e    %.1e" % ((tol,) + tuple(classes[tol])))
    ok = worst <= 1.0
    print("largest deviation / allowed = %.3g  %s" % (worst, "ok" if ok else "FAILED"))
    return ok


# ----------------------------------------------------------------------------
# checks and reports
def q_closed(s, e_const=None):
    """Q = E + s^2 EG with the closed expression of EG."""
    e_const = E if e_const is None else e_const
    return e_const + s * s * hole_eg_closed(s, e_const)


def self_test():
    ok = True

    def check(name, value, low, high):
        nonlocal ok
        good = low <= value <= high
        ok = ok and good
        print("%-74s %-12s %s" % (name, nstr(value, 4), "ok" if good else "FAILED"))

    omega = mpf(OMEGA_HSE06)
    above = 1 + mpf("1e-30")
    print("-- the constants and the exchange hole")
    check("T1  3pi/4 + Ga(0) with the published E (reason for the polynomial)",
          3 * pi / 4 + g_a(mpf("1e-12")), 1.3e-8, 1.5e-8)
    worst = max(abs(q_closed(mpf(s)) - q_closed(mpf(s), mpf(0))) for s in ("0.081", "1", "5"))
    check("T2  E + s^2 EG with the closed EG does not depend on E", worst, 0, 1e-45)
    q_poly = E + EGSCUT ** 2 * hole_eg(EGSCUT)
    check("T3  jump of E + s^2 EG at s = 0.08 (closed - polynomial)", q_closed(EGSCUT * above) - q_poly, -1.6e-9, -1.4e-9)
    worst = max(abs(enhancement(w, EGSCUT * above) / enhancement(w, EGSCUT) - 1) for w in ("0", "0.1", "0.3", "1", "3"))
    check("T4  largest relative jump of f at s = 0.08 (w = 0 ... 3)", worst, 0, 1e-8)
    print("-- limits of the enhancement factor")
    check("T5  1 - f(w=0, s->0)   (LDA limit with the published constants)", 1 - enhancement(0, S_FLOOR), 4.7e-9, 4.9e-9)
    worst = max(abs(enhancement(w, S_FLOOR) / f_lda_sr(w) - 1) for w in ("0.05", "0.2", "0.5", "1", "2", "5"))
    check("T6  max |f(w, s->0) / exact short-range LDA - 1|, w <= 5", worst, 0, 1e-2)
    worst = max(abs(enhancement("13.999", s) * (mpf("13.999") / mpf("14.001")) ** 2 / enhancement("14.001", s) - 1)
                for s in ("0.3", "1.5", "5"))
    check("T7  mismatch of the two branches across w = 14 (cancelling terms)", worst, 0, 1e-3)
    worst = max(abs(enhancement(0, s) / f_pbe(s) - 1) for s in ("0.2", "0.87", "2", "4.08"))
    check("T8  max |f(w=0, s) / F_PBE(s) - 1| at four s (model hole, not exact)", worst, 4e-3, 6e-3)
    print("-- energy density and derivatives")
    cases = ((("0.31", "0.07"), (("0.11", "-0.2", "0.05"), ("-0.03", "0.02", "0.04"))),
             (("2e-3", "0.07"), (("0.011", "-0.02", "0.005"), ("-0.03", "0.02", "0.04"))))
    worst_r, worst_g, s_raws = mpf(0), mpf(0), []
    for rho, grad in cases:
        v = evaluate(rho, grad)
        s_raws.append(v["s_raw"][0])
        r, g = mpf(rho[0]), [mpf(x) for x in grad[0]]
        gamma = sqrt(sum(x * x for x in g))
        h = r * mpf("1e-14")
        fd = (channel_energy(r + h, gamma, omega) - channel_energy(r - h, gamma, omega)) / (2 * h)
        worst_r = max(worst_r, abs(fd / v["dexdd"][0] - 1))
        hx = abs(g[0]) * mpf("1e-14")

        def e_of_gx(x):
            return channel_energy(r, sqrt(x * x + g[1] ** 2 + g[2] ** 2), omega)

        fd = (e_of_gx(g[0] + hx) - e_of_gx(g[0] - hx)) / (2 * hx)
        worst_g = max(worst_g, abs(fd / v["dexdgd"][0][0] - 1))
    if not (s_raws[0] < S_TRANS < s_raws[1]):
        raise RuntimeError("the two test points must lie on the two sides of s_raw = 8.3")
    check("T9  d e/d rho against a central difference (s_raw = %s and %s)" % (nstr(s_raws[0], 3), nstr(s_raws[1], 3)),
          worst_r, 0, 1e-20)
    check("T10 d e/d(d_x rho) against a central difference (same points)", worst_g, 0, 1e-20)
    n_tot, g_tot = mpf("0.4"), mpf("0.25")
    w_tot = omega / cbrt(3 * pi ** 2 * n_tot)
    s_tot = g_tot / (2 * cbrt(3 * pi ** 2) * n_tot ** (mpf(4) / 3))
    e_unpol = -mpf(3) / 4 * cbrt(3 / pi) * n_tot ** (mpf(4) / 3) * enhancement(w_tot, rescale(s_tot))
    half = evaluate((n_tot / 2, n_tot / 2), ((g_tot / 2, 0, 0), (g_tot / 2, 0, 0)))
    check("T11 spin scaling: two equal channels against the total-density form", abs(half["e"] / e_unpol - 1), 0, 1e-40)
    print("-- closed form against the defining integral")
    pairs = (("0", "0.5"), ("0.05", "0.02"), ("0.3", "1.3"), ("1", "4"), ("2.5", "0.5"), ("6", "8.5"), ("20", "1.3"))
    check("T12 max |closed form - quadrature of the hole| / |f|, 7 points", hole_integral_deviation(pairs), 0, 1e-18)
    print("-- the rescaling of the reduced gradient")
    check("T13 jump of the rescaled s at s_raw = 8.3", rescale(S_TRANS * above) - S_TRANS, 2.2e-9, 2.5e-9)
    check("T14 f(w=0) at the largest rescaled s, 8.572844 (local Lieb-Oxford bound 1.804)", enhancement(0, S_MAX), 1.80, 1.804)
    print("-- tolerance envelope at the fixture points")
    worst = mpf(0)
    for rho_s in SPIN_DENSITIES:
        for s_raw in REDUCED_GRADIENTS:
            w, _ = channel_variables(mpf(rho_s), mpf(1), omega)
            worst = max(worst, rounding_error(w, rescale(s_raw)) / tolerance(w))
    check("T15 max (16-digit rounding error of the direct formula) / tolerance", worst, 0, 0.1)
    print("self-test passed" if ok else "self-test FAILED")
    return ok


def report():
    above = 1 + mpf("1e-30")

    def grid(a, b, n):
        return [mpf(a) + (mpf(b) - mpf(a)) * i / n for i in range(n + 1)]

    print("Properties of the enhancement factor f(w, s); w = omega/kF")
    print("(a) s = 0.08: EG is the polynomial up to 0.08 and the closed expression above")
    print("    EG: polynomial %s, closed %s" % (nstr(hole_eg(EGSCUT), 12), nstr(hole_eg(EGSCUT * above), 12)))
    print("    E + s^2 EG: polynomial %s, closed %s" % (
        nstr(E + EGSCUT ** 2 * hole_eg(EGSCUT), 12), nstr(q_closed(EGSCUT * above), 12)))
    print("    relative jump of f: " + ", ".join(
        "%s (w = %s)" % (nstr(enhancement(w, EGSCUT * above) / enhancement(w, EGSCUT) - 1, 3), w)
        for w in ("0", "0.1", "0.3", "1", "3")))
    print("    1 - f(w = 0, s -> 0) = %s" % nstr(1 - enhancement(0, S_FLOOR), 4))
    print("(b) s_raw = 8.3: s = 8.572844 - 18.796223 / s_raw^2 above")
    print("    jump of the rescaled s: %s; d s / d s_raw just above 8.3: %s (1 below)" % (
        nstr(rescale(S_TRANS * above) - S_TRANS, 4), nstr(2 * S_CONST / S_TRANS ** 3, 4)))
    print("    relative jump of f: " + ", ".join(
        "%s (w = %s)" % (nstr(enhancement(w, rescale(S_TRANS * above)) / enhancement(w, S_TRANS) - 1, 3), w)
        for w in ("0", "0.1", "0.3", "1")))
    print("    f(w = 0): %s at s = 8.3, %s at s = 8.572844 (local Lieb-Oxford bound 1.804)" % (
        nstr(enhancement(0, S_TRANS), 8), nstr(enhancement(0, S_MAX), 8)))
    print("(c) omega = 0 limit against the PBE enhancement factor, (f - F_PBE)/F_PBE")
    pos, neg = (mpf(0), None), (mpf(0), None)
    for s in grid("0.001", "5", 500):
        d = enhancement(0, s) / f_pbe(s) - 1
        pos = max(pos, (d, s))
        neg = min(neg, (d, s))
    print("    0 < s <= 5: from %s (s = %s) to %s (s = %s)" % (nstr(neg[0], 3), nstr(neg[1], 3), nstr(pos[0], 3), nstr(pos[1], 3)))
    print("(d) rounding error of a 16-digit evaluation of the direct formula (largest over s = 0.05 ... 8)")
    for w in ("0.3", "1", "2", "4", "6", "8", "10", "12", "13.9", "14.1", "50"):
        err = max(rounding_error(w, s) for s in ("0.05", "0.09", "0.2", "0.5", "0.999", "2", "4", "8"))
        print("    w = %-5s %-9s tolerance %.1e   f(s = 0.5) = %s" % (w, nstr(err, 2), tolerance(w), nstr(enhancement(w, "0.5"), 4)))
    return True


def print_point(values, omega):
    rho, grad = values[0:2], (values[2:5], values[5:8])
    v = evaluate(rho, grad, omega)
    for sp, name in enumerate(("up", "down")):
        t = enhancement_terms(v["w"][sp], v["s"][sp])
        print("channel %-4s w = %s  s_raw = %s  s = %s" % (name, nstr(v["w"][sp], 20), nstr(v["s_raw"][sp], 20), nstr(v["s"][sp], 20)))
        for key in ("H", "F", "EG", "term1", "term2", "term3", "term4", "term5"):
            print("   %-6s %s" % (key, nstr(t[key], 20)))
        print("   f      %s" % nstr(enhancement(v["w"][sp], v["s"][sp]), 20))
        print("   e_s    %s   dexdd %s   vsigma %s" % (nstr(v["e_s"][sp], 20), nstr(v["dexdd"][sp], 20), nstr(v["vsigma"][sp], 20)))
        print("   dexdgd %s" % " ".join(nstr(x, 20) for x in v["dexdgd"][sp]))
    print("exc = %s   (energy per volume %s)" % (nstr(v["exc"], 20), nstr(v["e"], 20)))
    return True


# ----------------------------------------------------------------------------
# comparison with the Fortran routine of Quantum ESPRESSO
QE_KIND = """module kind_l
  implicit none
  integer, parameter :: DP = selected_real_kind(14, 200)
end module kind_l
"""
QE_DRIVER = """program wpbeh_qe_driver
  ! reads lines "rho grho omega" and prints sxsr, v1xsr, v2xsr of pbexsr and its error flag
  use kind_l, only: DP
  use exch_gga, only: pbexsr
  implicit none
  real(DP) :: rho, grho, omega, sxsr, v1xsr, v2xsr
  integer :: in_err, ios
  do
    read (*, *, iostat=ios) rho, grho, omega
    if (ios /= 0) exit
    in_err = 0
    call pbexsr(rho, grho, sxsr, v1xsr, v2xsr, omega, in_err)
    write (*, '(3es44.33e4,i3)') sxsr, v1xsr, v2xsr, in_err
  end do
end program wpbeh_qe_driver
"""
# Quantum ESPRESSO replaces exp(x) Ei(-x) by -(x^2 + c1 x + c2) / (x (x^2 + c3 x + c4)) from x = 700 on
QE_EXPEI = [mpf(x) for x in ("4.03640", "1.15198", "5.03627", "4.19160")]
QE_EXPFCUTOFF = 700


def write_qe_files(directory, omega=OMEGA_HSE06):
    """Write what a run of pbexsr of Quantum ESPRESSO at the fixture points needs.

    kind_l.f90 (the one module that qe_funct_exch_gga.f90 uses), wpbeh_qe_driver.f90 and
    wpbeh_qe_input.txt, one line "2 rho_s, 4 |grad rho_s|^2, omega" per channel: the arguments
    with which the spin-polarised driver of Quantum ESPRESSO calls the routine.  This script
    does not compile or run anything; the commands are printed.
    """
    os.makedirs(directory, exist_ok=True)
    lines = []
    for row in fixture_rows(omega):
        values = [float(x) for x in row]
        for sp in range(2):
            g = values[2 + 3 * sp:5 + 3 * sp]
            lines.append("%.17e %.17e %.17e" % (2 * values[sp], 4 * (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]), float(omega)))
    for name, text in (("kind_l.f90", QE_KIND), ("wpbeh_qe_driver.f90", QE_DRIVER),
                       ("wpbeh_qe_input.txt", "\n".join(lines) + "\n")):
        with open(os.path.join(directory, name), "w") as fh:
            fh.write(text)
    print("wrote kind_l.f90, wpbeh_qe_driver.f90 and wpbeh_qe_input.txt (%d channels) to %s" % (len(lines), directory))
    print("with QE = the path of XClib/qe_funct_exch_gga.f90 of Quantum ESPRESSO 7.6 (tag qe-7.6 of https://github.com/QEF/q-e):")
    print("  gfortran -O1 -J %s -o %s/wpbeh_qe_driver %s/kind_l.f90 $QE %s/wpbeh_qe_driver.f90" % ((directory,) * 4))
    print("  %s/wpbeh_qe_driver < %s/wpbeh_qe_input.txt > %s/wpbeh_qe_output.txt" % ((directory,) * 3))
    print("  python3 wpbeh_reference.py --compare-qe %s/wpbeh_qe_output.txt" % directory)
    print("adding -freal-8-real-16 to the gfortran command evaluates the routine with 33 digits, which removes its rounding error")
    return True


def compare_qe(output, omega=OMEGA_HSE06):
    """Compare the fixture points with the output of the driver of --qe-files.

    Half of sxsr is the energy per volume of the channel, v1xsr is dexdd and v2xsr is vsigma.
    Two properties of the Fortran routine are taken into account, so that what is compared is
    the definition and not these:

    (1) Above s_raw = 8.3 its potential lacks the derivative s' of the rescaled s (it uses
        d s/d rho = -4 s/(3 rho) and d s/d|grad rho| = s_raw/|grad rho|).  The reference values
        for that convention are vsigma / s' and
        dexdd - (8 |grad rho_s|^2 vsigma / (3 rho_s s_raw s')) (s - s' s_raw).
    (2) From HsbwA94 = 700 on, which is the case exactly above w = 14, it replaces
        expei = exp(x) Ei(-x) by a rational function R(x).  With d = R(x) - expei and
        P = e_s / f this changes f by (4A/9) d, the derivative of f with respect to Hsbw by d
        (the routine keeps the exact expression of that derivative and inserts R), hence
        dexdd by d (4 P / (3 rho_s)) (4A/9 - w^2 - s G) and vsigma by
        d P G s_raw / (2 |grad rho_s|^2), with G = d(s^2 H)/ds.

    What remains is the rounding of the Fortran arithmetic and the convergence threshold 1e-12
    of its exponential integral; the cancellation of the terms amplifies both.
    """
    import numpy as np
    with open(output) as fh:
        out = [line.split() for line in fh if line.strip()]
    points = fixture_points()
    rows = fixture_rows(omega)
    arr = np.array([[float(x) for x in row] for row in rows])
    if len(out) != 2 * len(arr) or any(len(words) != 4 for words in out):
        print("%s does not hold one line of four numbers per channel (%d channels)" % (output, 2 * len(arr)))
        return False
    col = {name: i for i, name in enumerate(COLUMNS)}
    classes = {}
    ok = True
    for n in range(len(arr)):
        for sp, name in enumerate(("up", "dn")):
            sxsr, v1, v2 = (mpf(x) for x in out[2 * n + sp][:3])
            err = int(out[2 * n + sp][3])
            tol = arr[n, col["tol_" + name]]
            rho = points[n][0][sp]
            gamma2 = sum(x * x for x in points[n][1][sp])
            w, s_raw = channel_variables(rho, sqrt(gamma2), mpf(omega))
            s = rescale(s_raw)
            e_ref, d_ref = rows[n][col["e_" + name]], rows[n][col["dexdd_" + name]]
            vs_ref = rows[n][col["vsigma_upup" if sp == 0 else "vsigma_dndn"]]
            rescaled = s_raw > S_TRANS
            if rescaled:
                slope = 2 * S_CONST / s_raw ** 3
                d_ref = d_ref - 8 * gamma2 * vs_ref / (3 * rho * s_raw * slope) * (s - slope * s_raw)
                vs_ref = vs_ref / slope
            large_w = w > WCUTOFF
            x = 9 * (s * s * hole_h(s) + (2 if large_w else EB1) * w * w) / (4 * A)
            if (x >= QE_EXPFCUTOFF) != large_w:
                raise RuntimeError("the rational approximation is expected exactly above w = 14")
            if large_w:
                d = -(x * x + QE_EXPEI[0] * x + QE_EXPEI[1]) / (x * (x * x + QE_EXPEI[2] * x + QE_EXPEI[3])) + e1_scaled(x)
                f = enhancement(w, s)
                p = e_ref / f
                slope_h = diff(lambda t: t * t * hole_h(t), s)
                d_ref = d_ref + d * (4 * p / (3 * rho)) * (4 * A / 9 - w * w - s * slope_h)
                vs_ref = vs_ref + d * p * slope_h * s_raw / (2 * gamma2)
                e_ref = e_ref * (1 + 4 * A * d / (9 * f))
            dev_e = float(abs(sxsr / 2 / e_ref - 1))
            dev_d = float(abs(v1 / d_ref - 1))
            dev_v = float(abs(v2 - vs_ref) / max(abs(vs_ref), mpf("1e-6") * abs(e_ref) / gamma2))
            ratio = max(dev_e / (100 * tol), dev_d / (1000 * tol), dev_v / (1000 * tol))
            label = ("w > 14" if large_w else "tol %.1e" % tol) + (", s_raw > 8.3" if rescaled else "")
            entry = classes.setdefault((large_w, tol, rescaled, label), [0.0, 0.0, 0.0, 0.0, 0])
            for i, dev in enumerate((dev_e, dev_d, dev_v, ratio)):
                entry[i] = max(entry[i], dev)
            entry[4] += 1
            if err != 0 or not ratio <= 1.0:
                print("point %d (%s): w = %s, s_raw = %s, error flag %d, deviations %.2e %.2e %.2e"
                      % (n, name, nstr(w, 3), nstr(s_raw, 3), err, dev_e, dev_d, dev_v))
                ok = False
    print("largest relative deviation of pbexsr (Quantum ESPRESSO 7.6) from the reference")
    print("   class                      channels  energy     d/d rho    vsigma     largest / allowed")
    for key in sorted(classes):
        e = classes[key]
        print("   %-26s %-9d %.1e    %.1e    %.1e    %.2g" % (key[3], e[4], e[0], e[1], e[2], e[3]))
    print("allowed: 100 x tol for the energy and 1000 x tol for the derivatives (tol from the fixture, 1e-9 above w = 14);")
    print("vsigma on the scale max(|vsigma|, 1e-6 |e_s| / |grad rho_s|^2); above s_raw = 8.3 and above w = 14 the")
    print("reference values are converted to what the Fortran routine computes (see the source of this script)")
    print("comparison passed" if ok else "comparison FAILED")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", action="store_true", help="print properties of the definition")
    ap.add_argument("--dump", metavar="FILE", help="write the fixture for the unit tests")
    ap.add_argument("--point", nargs=8, metavar="X", help="rho_up rho_dn and the two gradients")
    ap.add_argument("--double", action="store_true", help="double-precision prototype against the fixture")
    ap.add_argument("--qe-files", metavar="DIR", help="write a driver for pbexsr of Quantum ESPRESSO 7.6 and its input")
    ap.add_argument("--compare-qe", metavar="FILE", help="compare the output of that driver with the reference")
    ap.add_argument("--omega", default=OMEGA_HSE06, help="screening parameter in 1/bohr (default 0.11)")
    args = ap.parse_args()
    if args.report:
        return 0 if report() else 1
    if args.dump:
        print("wrote", args.dump, "with", write_fixture(args.dump, args.omega), "points")
        return 0
    if args.point:
        return 0 if print_point(args.point, args.omega) else 1
    if args.double:
        return 0 if double_check(args.omega) else 1
    if args.qe_files:
        return 0 if write_qe_files(args.qe_files, args.omega) else 1
    if args.compare_qe:
        return 0 if compare_qe(args.compare_qe, args.omega) else 1
    return 0 if self_test() else 1


if __name__ == "__main__":
    sys.exit(main())

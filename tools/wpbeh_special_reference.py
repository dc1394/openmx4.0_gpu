#!/usr/bin/env python3
"""
wpbeh_special_reference.py -- reference values of the two special functions of the
short-range omega-PBE exchange, for the unit test xc-wpbeh-special of modules/xc
(milestone M1 of docs/EXX_IMPLEMENTATION_GUIDE.md).

What it evaluates
  e1_scaled(x) = exp(x) E1(x)          and its derivative  e1_scaled(x) - 1/x
  erfcx(x)     = exp(x^2) erfc(x)      and its derivative  2 x erfcx(x) - 2/sqrt(pi)
  with 50 significant digits (mpmath), at double-precision arguments between 1e-30 and 1e8.

Points
  16 per decade from 1e-30 to 1e8, 64 per decade from 0.1 to 100, and each argument at which
  the implementation (modules/xc/src/special_functions.cpp) changes its formula, with the
  neighbouring doubles on both sides.

Usage
  python3 wpbeh_special_reference.py --dump FILE    # write the fixture
  python3 wpbeh_special_reference.py --check FILE   # compare a stored fixture with fresh values
"""
import argparse
import math
import sys

from mpmath import mp, mpf, e1, erfc, exp, sqrt, pi

mp.dps = 50
COLUMNS = ("x", "e1_scaled", "de1_scaled", "erfcx", "derfcx")
# arguments at which special_functions.cpp switches between the series, the product and
# the depths of the continued fractions
SWITCHES = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 10.0, 20.0)


def points():
    xs = set()
    for i in range(38 * 16 + 1):
        xs.add(float(mpf(10) ** (mpf(-30) + mpf(i) / 16)))
    for i in range(3 * 64 + 1):
        xs.add(float(mpf(10) ** (mpf(-1) + mpf(i) / 64)))
    for t in SWITCHES:
        xs.update((math.nextafter(t, 0.0), t, math.nextafter(t, math.inf)))
    xs.update((1e-30, 1e8))
    return sorted(x for x in xs if 1e-30 <= x <= 1e8)


def reference(x):
    """Values for the double x (the argument is converted exactly)."""
    x = mpf(x)
    a = exp(x) * e1(x)
    b = exp(x * x) * erfc(x)
    return [x, a, a - 1 / x, b, 2 * x * b - 2 / sqrt(pi)]


def rows():
    return [[float(v) for v in reference(x)] for x in points()]


def read_fixture(path):
    out = []
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if parts and parts[0][0] in "+-0123456789":
                out.append([float(v) for v in parts])
    return out


def write_fixture(path):
    data = rows()
    with open(path, "w") as fh:
        fh.write("wpbeh_special_fixture 1\n")
        fh.write("definition exp(x) E1(x), erfcx(x) and their derivatives (evaluated with %d digits)\n" % mp.dps)
        fh.write("npoints %d\n" % len(data))
        fh.write("ncolumns %d\n" % len(COLUMNS))
        fh.write("# " + " ".join(COLUMNS) + "\n")
        fh.write("# the values belong to the double-precision x written in the first column\n")
        for row in data:
            fh.write(" ".join("%.17e" % v for v in row) + "\n")
    if read_fixture(path) != data:
        raise RuntimeError("the fixture does not read back identically")
    return len(data)


def check_fixture(path):
    stored, fresh = read_fixture(path), rows()
    if len(stored) != len(fresh):
        print("number of points: stored %d, expected %d" % (len(stored), len(fresh)))
        return False
    worst = 0.0
    for a, b in zip(stored, fresh):
        for u, v in zip(a, b):
            if u != v:
                worst = max(worst, abs(u - v) / max(abs(u), abs(v)))
    print("%d points, largest relative difference from fresh values %.3g" % (len(stored), worst))
    return worst <= 1e-15


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", metavar="FILE", help="write the fixture for the unit test")
    ap.add_argument("--check", metavar="FILE", help="compare a stored fixture with fresh values")
    args = ap.parse_args()
    if args.dump:
        print("wrote", args.dump, "with", write_fixture(args.dump), "points")
        return 0
    if args.check:
        return 0 if check_fixture(args.check) else 1
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())

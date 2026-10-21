#!/usr/bin/env python3
"""Compare production D3 C6 interpolation to unshifted high-precision sums."""
import ctypes
from decimal import Decimal, localcontext
import math
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
source = (ROOT / "source/Total_Energy.c").read_text()
start = source.index("/* Calculate C6, C8 coefficient and derivatives with Gaussian-distance (L) */")
start = source.index("Z = 0.0;", start)
end = source.index("/*  CALCULATE ENERGY AND TWO FIRST PART OF GRADIENTS*/", start)
interpolation = source[start:end]
wrapper = r'''
#include <math.h>
void interpolate(int na,int nb,double ca,double cb,double k,const double *ref,double *out) {
 int maxcn_dftD[2]={na,nb},wanA=0,wanB=1,Gc_AN=1,Gc_BN=2,i,j;
 double C6ab_dftD[2][2][8][8][3]={0},CN[3]={0,ca,cb},k3_dftD=k;
 double r2r4ab_dftD[2][2]={{0,2.3},{2.3,0}},dC6ij[3][3]={0};
 double Z,W,dZi,dZj,dWi,dWj,dAi,dBj,exparg,Lij,C6_ref,C6,C8,C8C6;
 for(i=0;i<na;i++)for(j=0;j<nb;j++)for(int v=0;v<3;v++)
  C6ab_dftD[0][1][i][j][v]=ref[3*(i*nb+j)+v];
''' + interpolation + r'''
 out[0]=C6;out[1]=C8;out[2]=dC6ij[1][2];
}
'''


def reference(table, x, y, k):
    with localcontext() as context:
        context.prec = 100
        z = w = dz = dw = Decimal(0)
        dx, dy, dk = map(lambda v: Decimal(str(v)), (x, y, k))
        for coeff, a, b in table:
            if coeff <= 1e-12:
                continue
            c, a, b = map(lambda v: Decimal(str(v)), (coeff, a, b))
            weight = (-dk * ((dx-a)**2 + (dy-b)**2)).exp()
            z += c*weight
            w += weight
            dz += c*weight*2*dk*(dx-a)
            dw += weight*2*dk*(dx-a)
        if not w:
            return (0.0, 0.0, 0.0)
        c6 = z/w
        # OpenMX stores minus the derivative with respect to CN_i.
        return (float(c6), float(c6*Decimal('6.9')), float((dz*w-dw*z)/(w*w)))


with tempfile.TemporaryDirectory(prefix="openmx-d3-") as directory:
    p = Path(directory)
    (p / "check.c").write_text(wrapper)
    subprocess.run(shlex.split(os.environ.get("CC", "cc")) + [
        "-std=c11", "-O2", "-fPIC", "-shared", str(p / "check.c"),
        "-lm", "-o", str(p / "check.so")], check=True)
    library = ctypes.CDLL(str(p / "check.so"))
    function = library.interpolate
    function.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double,
                         ctypes.c_double, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
    function.restype = None

    def evaluate(na, nb, table, x, y, k):
        data = (ctypes.c_double * (3*len(table)))(*(v for row in table for v in row))
        out = (ctypes.c_double * 3)()
        function(na, nb, x, y, k, data, out)
        return tuple(out)

    cases = 0
    for na, nb in ((1, 1), (2, 3), (5, 4)):
        table = [(20.0+3*i+2*j, 1.1*i, .8*j) for i in range(na) for j in range(nb)]
        for x, y in ((.4, .7), (1.2, 1.1), (10, 12), (40, 41), (150, 155)):
            for k in (1.0, 4.0):
                got = evaluate(na, nb, table, x, y, k)
                expected = reference(table, x, y, k)
                for a, b in zip(got, expected):
                    assert math.isfinite(a) and math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-9), (got, expected)
                h = 1e-5
                finite_difference = -(evaluate(na, nb, table, x+h, y, k)[0] -
                                      evaluate(na, nb, table, x-h, y, k)[0])/(2*h)
                assert math.isclose(got[2], finite_difference, rel_tol=1e-5, abs_tol=1e-7)
                assert got[0] > 0, "Large CN must not silently lose its dispersion coefficient"
                cases += 1
        table[0] = (0.0, table[0][1], table[0][2])
        got = evaluate(na, nb, table, 15, 16, 4.0)
        expected = reference(table, 15, 16, 4.0)
        assert all(math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-9) for a, b in zip(got, expected))
        cases += 1
    assert evaluate(1, 1, [(0.0, 0.0, 0.0)], 20, 30, 4.0) == (0.0, 0.0, 0.0)
    print(f"PASS: {cases} D3 interpolation/gradient cases and missing-reference handling")

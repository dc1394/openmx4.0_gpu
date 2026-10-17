#!/usr/bin/env python3
"""Check the capacity fallback with real LAPACK/BLAS (no HIP required)."""
import ctypes as c
import ctypes.util
import math
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory(prefix="openmx-dense-cpu-") as folder:
    lib = Path(folder) / "cpu.so"
    blas = ctypes.util.find_library("openblas")
    libs = ["-l:" + blas] if blas else ["-llapack", "-lblas"]
    libs = shlex.split(os.environ.get("BLAS_LIBS", "")) or libs
    subprocess.run(shlex.split(os.environ.get("CC", "cc")) + [
        "-std=c11", "-O1", "-Wall", "-Wextra", "-Werror", "-fPIC", "-shared",
        str(root / "source/openmx_dense_cpu.c"), *libs, "-lm", "-o", str(lib)], check=True)
    api = c.CDLL(str(lib))
    solve = api.openmx_dense_cpu_eigen
    solve.argtypes = [c.c_int, c.c_int, c.c_int, c.c_void_p, c.c_int, c.c_void_p]
    gemm = api.openmx_dense_cpu_gemm
    gemm.argtypes = [c.c_char, c.c_char, c.c_int, c.c_int, c.c_int, c.c_int,
                     c.c_void_p, c.c_int, c.c_void_p, c.c_int, c.c_void_p, c.c_int]
    cases = 0

    def buffer(values, z):
        flat = [q for v in values for q in ([complex(v).real, complex(v).imag] if z else [v])]
        return (c.c_double * len(flat))(*flat)

    def value(buf, index, z):
        return complex(buf[2*index], buf[2*index+1]) if z else buf[index]

    for z in (0, 1):
        for n in (1, 4, 19):
            lda = n + 3
            h = [[complex(math.sin(i+j+1)/n, (i-j)/(n*n)) if z else math.sin(i+j+1)/n
                  for j in range(n)] for i in range(n)]
            for i in range(n):
                h[i][i] += i+1
            for nev in sorted({1, n//2+1, n}):
                # Poison the unused triangle: only lower should be read.
                a = buffer([h[i][j] if i>=j and i<n else 98765.0
                            for j in range(n) for i in range(lda)], z)
                original = bytes(a)
                w = buffer([-999.0]*nev, 0)
                assert solve(n, nev, z, a, lda, w) == 1
                assert all(w[j]>=w[j-1] for j in range(1, nev))
                for j in range(nev):
                    v = [value(a, j*lda+i, z) for i in range(n)]
                    residual = max(abs(sum(h[i][k]*v[k] for k in range(n))-w[j]*v[i]) for i in range(n))
                    assert residual < 2e-12*n, residual
                    for k in range(nev):
                        dot = sum(complex(v[i]).conjugate()*value(a, k*lda+i, z) for i in range(n))
                        assert abs(dot-(j==k)) < 2e-12*n
                stride = 8*(2 if z else 1)
                for j in range(n):
                    start = (j*lda + (n if j<nev else 0))*stride
                    end = (j+1)*lda*stride
                    assert bytes(a)[start:end] == original[start:end]
                saved = bytes(a), bytes(w)
                assert solve(n, n+1, z, a, lda, w) == 0
                assert (bytes(a), bytes(w)) == saved
                cases += 1

        for ta in "NTC":
            for tb in "NTC":
                m, n, k = 3, 4, 5
                ar, ac = (m, k) if ta=="N" else (k, m)
                br, bc = (k, n) if tb=="N" else (n, k)
                lda, ldb, ldc = ar+2, br+3, m+1
                av = [complex(math.sin(i+1), math.cos(i+2)) if z else math.sin(i+1) for i in range(lda*ac)]
                bv = [complex(math.cos(i+2), math.sin(i+3)) if z else math.cos(i+2) for i in range(ldb*bc)]
                a, b, out = buffer(av,z), buffer(bv,z), buffer([12345.0]*(ldc*n),z)
                assert gemm(ta.encode(),tb.encode(),z,m,n,k,a,lda,b,ldb,out,ldc)==1

                def op(v, ld, trans, i, j):
                    x = v[j*ld+i] if trans=="N" else v[i*ld+j]
                    return complex(x).conjugate() if trans=="C" else x

                for j in range(n):
                    for i in range(m):
                        ref = sum(op(av,lda,ta,i,q)*op(bv,ldb,tb,q,j) for q in range(k))
                        assert abs(value(out,j*ldc+i,z)-ref) < 2e-12
                    assert value(out,j*ldc+m,z) == 12345.0
                cases += 1
    print("PASS: {} real/complex eigenproblem and GEMM cases with real LAPACK/BLAS".format(cases))

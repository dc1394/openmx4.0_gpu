#!/usr/bin/env python3
"""Check the noncollinear DC-LNO GPU local solve on the GPU, single process.

The production functions of the noncollinear GPU solve
(DCLNO_Solve_NonCol_GpuSolver and its DCLNO_GpuSolverZ_* helpers) are
extracted verbatim from source/Divide_Conquer_LNO.c and linked against the
existing OpenMX objects magma_openmx.o (openmx_hipsolver_zheevd_gpu) and
gemmul8_bridge.o (openmx_gemmul8Zgemm), libgemmul8.a, MAGMA and AOCL; the
rocSOLVER, hipBLAS and GEMMul8 calls are the real ones.  Synthetic local
problems of three kinds are solved for NUM2 = NUM (SCF 1-2) and NUM2 = 0.6 NUM
(the partial spectrum of SCF >= 3):
  noncollinear    complex spin off-diagonal blocks
  real-blockdiag  real block-diagonal Hamiltonian (collinear spins, no SOC)
  nonmagnetic     equal real spin blocks (exactly degenerate pairs)
with a real overlap, as in DC_NonCol.  The eigenvalues and the projector
sum_k conj(x_k) x_k^T over the returned states are compared with LAPACK zhegv
for conj(H) x = lambda S x, the problem the CPU flow of DC_NonCol solves.
The real overlap and the real Hamiltonians are the inputs for which ROCm
7.2's complex divide and conquer returns NaN vectors without the phase
similarity of openmx_hipsolver_zheevd_gpu, and the nonmagnetic partial case
is where the zheevdx of the former path returned wrong eigenvalues.

It never rebuilds OpenMX.  Build the OpenMX objects first, then e.g.
  python3 tools/check_dclno_noncol_gpu.py --build-dir /tmp/dclno-nc-gpu --run 150 250
(NUMH sizes; NUM = 2 NUMH; 150 250 take about 10 s, 400 600 about 3 min of
which most is the CPU reference).  Run it alone on the GPU.  --objects DIR
links magma_openmx.o and gemmul8_bridge.o from DIR instead of source/, e.g.
objects compiled into a scratch directory with the makefile's command lines
after magma_openmx.cu or gemmul8_bridge.cu changed.
"""

import argparse
import os
from pathlib import Path
import re
import shlex
import subprocess


def function(source, name):
    match = re.search(r"^static [^;{}]*\b" + re.escape(name) + r"\([^;]*?\)\s*\{",
                      source, re.MULTILINE)
    if match is None:
        raise ValueError(f"Cannot find production function {name}")
    brace = source.index("{", match.start())
    depth, end = 1, brace + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[match.start():end]


def define(source, name):
    match = re.search(r"^#define\s+" + re.escape(name) + r"\b.*$", source, re.MULTILINE)
    if match is None:
        raise ValueError(f"Cannot find production macro {name}")
    return match.group(0)


PREFIX = r'''
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "hip_runtime_compat.h"
#include "hipblas_compat.h"
#include "hip_complex_compat.h"
typedef struct { double r, i; } dcomplex;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); exit(2); } } while (0)
#define wait_hipfunc(call) CHECK((int)(call) == 0)
static void DCLNO_AbortWithMessage(const char *m) { fprintf(stderr, "FATAL: %s\n", m); exit(3); }
hipblasStatus_t openmx_gemmul8Zgemm(hipblasHandle_t, hipblasOperation_t, hipblasOperation_t, int, int, int,
    const hipDoubleComplex *, const hipDoubleComplex *, int, const hipDoubleComplex *, int,
    const hipDoubleComplex *, hipDoubleComplex *, int);
hipblasStatus_t hipblasZgeam(hipblasHandle_t, hipblasOperation_t, hipblasOperation_t, int, int,
    const hipDoubleComplex *, const hipDoubleComplex *, int, const hipDoubleComplex *,
    const hipDoubleComplex *, int, hipDoubleComplex *, int);
int openmx_hipsolver_zheevd_gpu(int n, int maxn, void *d_A, double *w, int *mout);
int openmx_hipsolver_release_workspace(void);
'''

WRAPPER = r'''
int check_nc_solve(int n, int nh, int n2, const dcomplex *s, dcomplex *h, double *ko)
{ return DCLNO_Solve_NonCol_GpuSolver(n, nh, n2, 1, s, h, ko); }
void check_nc_release(void) { DCLNO_GpuSolverZ_ReleaseDeviceMemory(); (void)openmx_hipsolver_release_workspace(); }
void check_nc_destroy(void) { DCLNO_GpuSolverZ_Destroy(); }
'''

MAIN = r'''
#define _POSIX_C_SOURCE 199309L
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct { double r, i; } dcomplex;
typedef struct { double re, im; } zc;
int MYID_MPI_COMM_WORLD = 0;
int check_nc_solve(int n, int nh, int n2, const dcomplex *s, dcomplex *h, double *ko);
void check_nc_release(void);
void check_nc_destroy(void);
extern void zhegv_(const int *, const char *, const char *, const int *, zc *, const int *,
                   zc *, const int *, double *, zc *, const int *, double *, int *);

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9*t.tv_nsec; }
static unsigned seed = 4242u;
static double rnd(void) { seed = seed*1664525u + 1013904223u; return (double)(seed >> 8)/(double)(1u << 24) - 0.5; }

static void matrices(int nh, int kind, dcomplex *s, dcomplex *h)
{
    const int n = 2*nh;
    memset(s, 0, sizeof(dcomplex)*(size_t)nh*nh);
    memset(h, 0, sizeof(dcomplex)*(size_t)n*n);
    for (int j = 0; j < nh; j++)
        for (int i = j; i < nh; i++) {
            double v = (i == j) ? 1.2 + 0.3*rnd() : 0.15*rnd()*exp(-0.05*(i - j));
            s[(size_t)j*nh+i].r = v; s[(size_t)i*nh+j].r = v;
        }
    for (int b = 0; b < 2; b++)
        for (int j = 0; j < nh; j++)
            for (int i = j; i < nh; i++) {
                double v = (kind == 2 && b == 1) ? h[(size_t)j*n+i].r
                         : (i == j) ? -1.0 + 2.0*i/nh + 0.05*b : 0.2*rnd()*exp(-0.03*(i - j));
                h[(size_t)(j + b*nh)*n + i + b*nh].r = v;
                h[(size_t)(i + b*nh)*n + j + b*nh].r = v;
            }
    if (kind == 0)
        for (int j = 0; j < nh; j++)
            for (int i = 0; i < nh; i++) {
                double re = 0.05*rnd()*exp(-0.03*abs(i - j)), im = 0.05*rnd()*exp(-0.03*abs(i - j));
                h[(size_t)(j + nh)*n + i].r = re; h[(size_t)(j + nh)*n + i].i = im;
                h[(size_t)i*n + j + nh].r = re;   h[(size_t)i*n + j + nh].i = -im;
            }
}

static void reference(int nh, const dcomplex *s, const dcomplex *h, double *w, zc *x)
{
    const int n = 2*nh, itype = 1, lwork = 64*n;
    zc *b = calloc((size_t)n*n, sizeof(zc)), *work = malloc(sizeof(zc)*lwork);
    double *rwork = malloc(sizeof(double)*3*n);
    int info;
    for (size_t i = 0; i < (size_t)n*n; i++) { x[i].re = h[i].r; x[i].im = -h[i].i; }
    for (int bk = 0; bk < 2; bk++)
        for (int j = 0; j < nh; j++)
            for (int i = 0; i < nh; i++) b[(size_t)(j + bk*nh)*n + i + bk*nh].re = s[(size_t)j*nh+i].r;
    zhegv_(&itype, "V", "L", &n, x, &n, b, &n, w, work, &lwork, rwork, &info);
    if (info != 0) { fprintf(stderr, "zhegv info %d\n", info); exit(1); }
    free(b); free(work); free(rwork);
}

static double projector_error(int n, int m, const zc *xref, const dcomplex *hout)
{
    double worst = 0.0;
    for (int b = 0; b < n; b++)
        for (int a = 0; a < n; a++) {
            double r1 = 0, i1 = 0, r2 = 0, i2 = 0;
            for (int k = 0; k < m; k++) {
                zc u = xref[(size_t)k*n + a], v = xref[(size_t)k*n + b];
                dcomplex p = hout[(size_t)a*n + k], q = hout[(size_t)b*n + k];
                r1 += u.re*v.re + u.im*v.im; i1 += u.re*v.im - u.im*v.re;
                r2 += p.r*q.r + p.i*q.i;     i2 += p.r*q.i - p.i*q.r;
            }
            double e = hypot(r1 - r2, i1 - i2);
            if (!(e <= worst)) worst = e;
        }
    return worst;
}

int main(int argc, char **argv)
{
    const char *kname[3] = {"noncollinear", "real-blockdiag", "nonmagnetic"};
    int sizes[16], nsizes = 0, bad = 0;
    for (int i = 1; i < argc && nsizes < 16; i++) sizes[nsizes++] = atoi(argv[i]);
    if (nsizes == 0) { sizes[0] = 150; sizes[1] = 250; nsizes = 2; }
    for (int a = 0; a < nsizes; a++) {
        const int nh = sizes[a], n = 2*nh;
        dcomplex *s = malloc(sizeof(dcomplex)*(size_t)nh*nh), *h = malloc(sizeof(dcomplex)*(size_t)n*n);
        dcomplex *hw = malloc(sizeof(dcomplex)*(size_t)n*n);
        zc *x = malloc(sizeof(zc)*(size_t)n*n);
        double *w = malloc(sizeof(double)*n), *ko = malloc(sizeof(double)*(n + 2));
        for (int kind = 0; kind < 3; kind++) {
            seed = 1000u + 17u*(unsigned)kind + (unsigned)nh;
            matrices(nh, kind, s, h);
            reference(nh, s, h, w, x);
            for (int part = 0; part < 2; part++) {
                int n2 = part ? (int)(0.6*n) : n, status = 0;
                double tbest = 1e30, ew = 0.0, ep;
                if (n2 % 2) n2--;           /* whole degenerate pairs */
                for (int rep = 0; rep < 3; rep++) {
                    double t0;
                    memcpy(hw, h, sizeof(dcomplex)*(size_t)n*n);
                    for (int i = 0; i < n + 2; i++) ko[i] = NAN;
                    t0 = now();
                    status = check_nc_solve(n, nh, n2, s, hw, ko);
                    if (now() - t0 < tbest) tbest = now() - t0;
                }
                for (int k = 0; k < n2; k++) {
                    double e = fabs(ko[k+1] - w[k]);
                    if (!(e <= ew)) ew = e;
                }
                ep = (status == 0) ? projector_error(n, n2, x, hw) : NAN;
                printf("NUM=%5d NUM2=%5d %-15s status=%d max|dlambda|=%.2e max|dP|=%.2e best time=%.4f s\n",
                       n, n2, kname[kind], status, ew, ep, tbest);
                fflush(stdout);
                if (!(status == 0 && ew < 1e-9 && ep < 1e-8)) bad++;
            }
        }
        check_nc_release();
        free(s); free(h); free(hw); free(x); free(w); free(ko);
    }
    check_nc_destroy();
    printf("%s: %s\n", bad ? "FAIL" : "PASS",
           bad ? "the GPU solve differs from zhegv" : "the GPU solve matches zhegv in every case");
    return bad ? 1 : 0;
}
'''


def main():
    root = Path(__file__).resolve().parents[1]
    src = root / "source"
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build-dir", type=Path, default=root / "work/dclno_noncol_gpu_check")
    parser.add_argument("--rocm", default=os.environ.get("ROCM", "/opt/rocm-7.2.0"))
    parser.add_argument("--aocl", default=os.environ.get("AOCL_HOME", "/opt/AMD/aocl/aocl-linux-aocc-5.2.0/aocc"))
    parser.add_argument("--arch", default="gfx1200")
    parser.add_argument("--objects", type=Path, default=src,
                        help="directory of magma_openmx.o and gemmul8_bridge.o (default: source/)")
    parser.add_argument("--run", nargs="*", type=int, metavar="NUMH",
                        help="run after building, for these overlap orders (default 150 250)")
    args = parser.parse_args()
    out = args.build_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    objects = args.objects.resolve()
    for needed in (objects / "magma_openmx.o", objects / "gemmul8_bridge.o",
                   src / "third_party/GEMMul8/lib/libgemmul8.a"):
        if not needed.exists():
            parser.error(f"{needed} is missing: build OpenMX first")

    source = (src / "Divide_Conquer_LNO.c").read_text()
    parts = [PREFIX]
    for name in ["DCLNO_MAGMA_ERR_HOST_ALLOC", "DCLNO_MAGMA_ERR_DEVICE_ALLOC", "DCLNO_RETRY_NO_MEMORY",
                 "DCLNO_RETRY_GPU_ERROR", "DCLNO_MAGMA_ERR", "DCLNO_MAGMA_ERR_ILLEGAL_VALUE",
                 "DCLNO_MAGMA_ERR_UNKNOWN"]:
        parts.append(define(source, name))
    for name in ["DCLNO_CheckedArrayBytes", "DCLNO_CheckedMulCount", "DCLNO_MallocArray",
                 "DCLNO_RetryTest", "DCLNO_ReportGpuRetry"]:
        parts.append(function(source, name))
    end = source.index("} DCLNO_GpuSolverZCtx;") + len("} DCLNO_GpuSolverZCtx;")
    parts.append(source[source.rfind("typedef struct {", 0, end):end] +
                 "\nstatic DCLNO_GpuSolverZCtx DCLNO_gpusolverz_ctx = {0};\n")
    for name in ["DCLNO_GpuSolverZ_FreeBuffers", "DCLNO_GpuSolverZ_Destroy", "DCLNO_GpuSolverZ_Init",
                 "DCLNO_GpuSolverZ_EnsureMatrixCapacity", "DCLNO_GpuSolverZ_Eigen",
                 "DCLNO_NonColGpuCallFailed", "DCLNO_Solve_NonCol_GpuSolver",
                 "DCLNO_GpuSolverZ_ReleaseDeviceMemory"]:
        parts.append(function(source, name))
    parts.append(WRAPPER)
    (out / "nc_solve.c").write_text("\n\n".join(parts))
    (out / "nc_main.c").write_text(MAIN)

    clang = f"{args.rocm}/llvm/bin/clang"
    magma = src / "third_party/magma"
    run = lambda cmd: subprocess.run(cmd, check=True, cwd=out)
    run([clang, "-O2", "-std=c11", "-D__HIP_PLATFORM_AMD__", "-Wno-implicit-function-declaration",
         f"-I{src}", f"-I{args.rocm}/include", "-c", "nc_solve.c", "-o", "nc_solve.o"])
    run([clang, "-O2", "-std=c11", "-c", "nc_main.c", "-o", "nc_main.o"])
    run([f"{args.rocm}/bin/hipcc", f"--offload-arch={args.arch}", "nc_main.o", "nc_solve.o",
         str(objects / "gemmul8_bridge.o"), str(objects / "magma_openmx.o"),
         str(src / "third_party/GEMMul8/lib/libgemmul8.a"),
         f"-L{magma}/lib", "-lmagma", f"-Wl,-rpath,{magma}/lib",
         f"-L{args.aocl}/lib", "-lflame", "-lblis", f"-Wl,-rpath,{args.aocl}/lib",
         f"-L{args.rocm}/lib", "-lhipsolver", "-lhipblas", "-lhipblaslt", "-lrocblas",
         "-o", "check_dclno_noncol_gpu"])
    print(f"Built {out / 'check_dclno_noncol_gpu'}")
    if args.run is not None:
        env = os.environ.copy()
        env.pop("OPENMX_DCLNO_RETRY_TEST", None)
        env.setdefault("OMP_NUM_THREADS", "4")
        env.setdefault("BLIS_NUM_THREADS", "4")
        subprocess.run([str(out / "check_dclno_noncol_gpu"), *map(str, args.run)], check=True, env=env)


if __name__ == "__main__":
    main()

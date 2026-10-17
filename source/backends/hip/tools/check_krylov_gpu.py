#!/usr/bin/env python3
"""Check the GPU projected solve of the Krylov O(N) solver on the GPU, single
process.

The production GPU helpers of source/Krylov.c (the per-thread workspace,
Krylov_Dgemm, Krylov_Eigen2, the KU cache, the device-memory preflight and
the fused Krylov_ProjectedSolve_GPU) are extracted verbatim and linked
against the existing OpenMX objects gemmul8_bridge.o (openmx_gemmul8Dgemm,
openmx_gemmul8DWorkspaceSize) and magma_openmx.o (openmx_magma_dsyevdx_gpu),
libgemmul8.a, MAGMA and AOCL; MPI is a one-rank stub.  For synthetic local
problems of Krylov's shapes (n = Msize, m3 = Msize3) every combination of

  OPENMX_KRYLOV_GEMMUL8   gemmul8 (openmx_gemmul8Dgemm) or hipblas (hipblasDgemm)
  OPENMX_KRYLOV_GPU_FUSED fused or per-call
  eigensolver             magma (OPENMX_KRYLOV_GPU_EIGEN_MIN <= m3) or lapack

is run after the preflight of the call (which must admit it) and compared
with the CPU sequence of Krylov_Col (BLAS/LAPACK): the eigenvalues and the
eigenvector rows of C up to their sign.  The best time of --reps solves is
printed with the GEMMul8 workspace of the three products, so the two GEMM
routes and the two eigensolvers can be compared at these sizes.

It never rebuilds OpenMX.  Build the OpenMX objects first, then e.g.
  python3 tools/check_krylov_gpu.py --build-dir /tmp/krylov-gpu --run 500x300 900x540
(nxm3 shapes; the default four take about 10 s).  Run it alone on the GPU.
--objects DIR links magma_openmx.o and gemmul8_bridge.o from DIR instead of
source/, e.g. objects compiled into a scratch directory with the makefile's
command lines after magma_openmx.cu or gemmul8_bridge.cu changed.
"""

import argparse
import os
from pathlib import Path
import re
import subprocess


def function(source, name, static=True):
    prefix = r"^static [^;{}]*\b" if static else r"^[a-z][^;{}()]*\b"
    match = re.search(prefix + re.escape(name) + r"\([^;]*?\)\s*\{", source, re.MULTILINE)
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


def variable(source, name):
    match = re.search(r"^static [^;(]*\b" + re.escape(name) + r"\b[^;(]*;", source, re.MULTILINE)
    if match is None:
        raise ValueError(f"Cannot find production variable {name}")
    return match.group(0)


def typedef(source, name):
    end = source.index("} " + name + ";") + len("} " + name + ";")
    return source[source.rfind("typedef struct {", 0, end):end]


def between(source, first, last):
    start = source.index(first)
    return source[start:source.index(last, start) + len(last)]


PREFIX = r'''
#define _POSIX_C_SOURCE 199309L
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "hip_runtime_compat.h"
#include "hipblas_compat.h"
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #c); exit(2); } } while (0)
#define wait_hipfunc(call) CHECK((int)(call) == 0)
#define F77_NAME(lower, upper) lower ## _
extern void dgemm_(const char *, const char *, const int *, const int *, const int *,
                   const double *, const double *, const int *, const double *,
                   const int *, const double *, double *, const int *);
extern void dsyevd_(const char *, const char *, const int *, double *, const int *,
                    double *, double *, const int *, int *, const int *, int *);
hipblasStatus_t openmx_gemmul8Dgemm(hipblasHandle_t, hipblasOperation_t, hipblasOperation_t, int, int, int,
    const double *, const double *, int, const double *, int, const double *, double *, int);
size_t openmx_gemmul8DWorkspaceSize(int, int, int);
void openmx_gemmul8ReleaseWorkspaces(void);
int openmx_magma_dsyevdx_gpu(int n, int maxn, double *d_A, double *w, int *mout);
int MYID_MPI_COMM_WORLD = 0;

/* MPI, one rank */
typedef int MPI_Comm; typedef int MPI_Datatype; typedef int MPI_Op; typedef int MPI_Info;
#define MPI_COMM_NULL (-1)
#define MPI_COMM_WORLD 0
#define MPI_INFO_NULL 0
#define MPI_COMM_TYPE_SHARED 0
#define MPI_UNSIGNED_LONG_LONG 1
#define MPI_SUM 1
#define MPI_MIN 2
static MPI_Comm mpi_comm_level1 = 0;
static int Host_ID = 0, level_stdout = 1;
static int MPI_Comm_split_type(MPI_Comm c, int t, int k, MPI_Info i, MPI_Comm *out) { *out = 1; return 0; }
static int MPI_Comm_split(MPI_Comm c, int color, int key, MPI_Comm *out) { *out = 2; return 0; }
static int MPI_Comm_free(MPI_Comm *c) { *c = MPI_COMM_NULL; return 0; }
static int MPI_Comm_size(MPI_Comm c, int *size) { *size = 1; return 0; }
static int MPI_Comm_rank(MPI_Comm c, int *rank) { *rank = 0; return 0; }
static int MPI_Allreduce(const void *s, void *r, int count, MPI_Datatype t, MPI_Op op, MPI_Comm c)
{ memcpy(r, s, sizeof(unsigned long long)); return 0; }
static int MPI_Barrier(MPI_Comm c) { return 0; }
static int MPI_Abort(MPI_Comm c, int code) { exit(4); }

/* OpenMX globals and helpers */
#define GPUSOLVER 7
static int scf_eigen_lib_flag = GPUSOLVER;
static int openmx_band_gpu_dense_oversubscribed(void) { return 0; }
static void dtime(double *t) { struct timespec s; clock_gettime(CLOCK_MONOTONIC, &s); *t = s.tv_sec + 1e-9*s.tv_nsec; }
static int OpenMX_GpuMemGetInfo(size_t *f, size_t *t) { return hipMemGetInfo(f, t); }
/* Eigen_lapack2 of Krylov: a[(i+1)*csize+j+1] in, eigenvector i at a[i*n+j]
   out, ko[1..EVmax] */
static void Eigen_lapack2(double *a, int csize, double *ko, int n, int EVmax)
{
    int lwork = 1 + 6*n + 2*n*n, liwork = 3 + 5*n, info;
    double *A = malloc(sizeof(double)*(size_t)n*n), *w = malloc(sizeof(double)*n);
    double *work = malloc(sizeof(double)*lwork);
    int *iwork = malloc(sizeof(int)*liwork);
    CHECK(A && w && work && iwork);
    for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) A[i*n+j] = a[(i+1)*csize+j+1];
    dsyevd_("V", "L", &n, A, &n, w, work, &lwork, iwork, &liwork, &info);
    CHECK(info == 0);
    for (int i = EVmax; i >= 1; i--) ko[i] = w[i-1];
    for (int i = 0; i < EVmax; i++) for (int j = 0; j < n; j++) a[i*n+j] = A[i*n+j];
    free(A); free(w); free(work); free(iwork);
}
'''

MAIN = r'''
static double worst;

/* the CPU sequence of Krylov_Col (ws == NULL) or its per-call GPU path */
static void per_call(Krylov_GPU_Workspace *ws, int n, int m3, int msize2, double ***Ku, double ****EC,
                     double *H_DC, double *KU, double *C, double *ko)
{
    for (int i = 0; i < n; i++) for (int j = 0; j < m3; j++) KU[j*n+i] = Ku[0][1][j*msize2+i+1];
    Krylov_Dgemm(ws, HIPBLAS_OP_N, HIPBLAS_OP_N, n, m3, n, H_DC, n, KU, n, C, n);
    Krylov_Dgemm(ws, HIPBLAS_OP_T, HIPBLAS_OP_N, m3, m3, n, KU, n, C, n, H_DC, m3);
    for (int i = 0; i < (int)Ku[0][1][0]; i++) H_DC[i*m3+i] = 1.0e+3;
    for (int i = m3-1; 0 <= i; i--) for (int j = 0; j < m3; j++)
        H_DC[(i+1)*(m3+1)+(j+1)] = H_DC[i*m3+j] + EC[0][1][i+1][j+1];
    Krylov_Eigen2(ws, H_DC, m3+1, ko, m3, m3);
    Krylov_Dgemm(ws, HIPBLAS_OP_T, HIPBLAS_OP_T, m3, n, m3, H_DC, m3, KU, n, C, m3);
}

int main(int argc, char **argv)
{
    int shapes[16][2], nshapes = 0, reps = 3;
    Krylov_GPU_Workspace *ws;

    for (int a = 1; a < argc; a++) {
        if (strncmp(argv[a], "--reps=", 7) == 0) { reps = atoi(argv[a] + 7); continue; }
        CHECK(nshapes < 16 && sscanf(argv[a], "%dx%d", &shapes[nshapes][0], &shapes[nshapes][1]) == 2);
        nshapes++;
    }
    if (nshapes == 0) {
        const int def[4][2] = {{500, 300}, {700, 420}, {900, 540}, {1100, 660}};
        memcpy(shapes, def, sizeof(def));
        nshapes = 4;
    }
    if (reps < 1) reps = 1;

    Krylov_GPU_InitOnce();
    Krylov_GPU_PreparePool(1);
    ws = Krylov_GPU_GetWorkspace(0);
    CHECK(ws != NULL && ws->active);
    for (int s = 0; s < nshapes; s++) {
        const int n = shapes[s][0], m3 = shapes[s][1], msize2 = n + 3, csize = n + 40;
        int Msize[2] = {1, n}, Msize2[2] = {1, msize2}, Msize3[2] = {1, m3};
        double *u = malloc(sizeof(double)*((size_t)msize2*m3 + 1)), *Ua[2], **Us[1], ***Ku;
        double **ecr = malloc(sizeof(double*)*(m3+1)), **Ea[2], ***Es[1], ****EC;
        double *H = malloc(sizeof(double)*(size_t)n*n), *H_DC = malloc(sizeof(double)*(size_t)csize*csize);
        double *KU = malloc(sizeof(double)*(size_t)(msize2+2)*m3);
        double *Cref = malloc(sizeof(double)*(size_t)csize*csize), *C = malloc(sizeof(double)*(size_t)csize*csize);
        double *koref = malloc(sizeof(double)*csize), *ko = malloc(sizeof(double)*csize);
        CHECK(n > 0 && m3 > 0 && m3 <= n && u && ecr && H && H_DC && KU && Cref && C && koref && ko);
        for (int j = 0; j < n; j++) for (int i = 0; i <= j; i++) {
            double v = (i == j) ? -1.0 + 2.0*i/n : 0.2*sin(1.0 + i + 2.3*j)*exp(-0.01*(j - i));
            H[(size_t)j*n+i] = v; H[(size_t)i*n+j] = v;
        }
        u[0] = 2.0;   /* ZeroNum */
        for (int j = 0; j < m3; j++) for (int i = 0; i < msize2; i++)
            u[j*msize2+i+1] = cos(0.3 + 1.7*i + 0.9*j*j)*exp(-0.002*abs(i - j)) + (i == j ? 1.0 : 0.0);
        for (int i = 0; i <= m3; i++) ecr[i] = malloc(sizeof(double)*(m3+1));
        for (int i = 0; i <= m3; i++) for (int j = 0; j <= i; j++)
            ecr[i][j] = ecr[j][i] = 0.02*cos(0.5*i + 1.1*j) + (i == j ? 0.01*i : 0.0);
        Ua[1] = u; Us[0] = Ua; Ku = Us; Ea[1] = ecr; Es[0] = Ea; EC = Es;
        memcpy(H_DC, H, sizeof(double)*(size_t)n*n);
        per_call(NULL, n, m3, msize2, Ku, EC, H_DC, KU, Cref, koref);
        printf("n=%5d m3=%4d GEMMul8 workspace (H*u1, u1^T*C, back transformation): %.1f / %.1f / %.1f MiB\n",
               n, m3, openmx_gemmul8DWorkspaceSize(n, m3, n)/1048576.0,
               openmx_gemmul8DWorkspaceSize(m3, m3, n)/1048576.0,
               openmx_gemmul8DWorkspaceSize(m3, n, m3)/1048576.0);
        for (int g8 = 1; g8 >= 0; g8--)
        for (int fused = 1; fused >= 0; fused--)
        for (int lapack = 0; lapack < 2; lapack++) {
            double t0, t1, best = 1e30, err = 0.0;
            Krylov_gpu_gemmul8 = g8;
            Krylov_gpu_fused = fused;
            Krylov_gpu_eigen_min = lapack ? m3 + 1 : 1;
            Krylov_gpu_dgemm_min_flops = 0.0;
            Krylov_GPU_Preflight(1, Msize, Msize3);
            CHECK(Krylov_gpu_fits);
            for (int rep = 0; rep < reps; rep++) {
                memcpy(H_DC, H, sizeof(double)*(size_t)n*n);
                dtime(&t0);
                if (Krylov_ProjectedSolve_GPU(ws, 0, 1, Msize, Msize2, Msize3, Ku, EC, H_DC, KU, C, ko) == 0)
                    per_call(ws, n, m3, msize2, Ku, EC, H_DC, KU, C, ko);
                dtime(&t1);
                if (t1 - t0 < best) best = t1 - t0;
            }
            CHECK(Krylov_gpu_fits);
            for (int k = 0; k < m3; k++) {
                double dot = 0.0, e = fabs(ko[k+1] - koref[k+1]);
                if (!(e <= err)) err = e;
                for (int j = 0; j < n; j++) dot += C[k + (size_t)j*m3]*Cref[k + (size_t)j*m3];
                for (int j = 0; j < n; j++) {
                    e = fabs(C[k + (size_t)j*m3] - (dot < 0 ? -1.0 : 1.0)*Cref[k + (size_t)j*m3]);
                    if (!(e <= err)) err = e;
                }
            }
            printf("n=%5d m3=%4d %-8s %-8s eig=%-6s max err %.2e  best time %.4f s (need %.1f MiB)\n", n, m3,
                   g8 ? "gemmul8" : "hipblas", fused ? "fused" : "per-call", lapack ? "lapack" : "magma",
                   err, best, Krylov_gpu_need_total/1048576.0);
            fflush(stdout);
            if (!(err < 1e-8)) worst = 1.0;
        }
        for (int i = 0; i <= m3; i++) free(ecr[i]);
        free(u); free(ecr); free(H); free(H_DC); free(KU); free(Cref); free(C); free(koref); free(ko);
    }
    Krylov_Release_GPU_Caches();
    openmx_gemmul8ReleaseWorkspaces();
    printf("%s\n", worst == 0.0 ? "PASS: every GPU route matches the CPU sequence"
                                : "FAIL: a GPU route differs from the CPU sequence");
    return worst == 0.0 ? 0 : 1;
}
'''


def main():
    root = Path(__file__).resolve().parents[1]
    src = root / "source"
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build-dir", type=Path, default=root / "work/krylov_gpu_check")
    parser.add_argument("--rocm", default=os.environ.get("ROCM", "/opt/rocm-7.2.0"))
    parser.add_argument("--aocl", default=os.environ.get("AOCL_HOME", "/opt/AMD/aocl/aocl-linux-aocc-5.2.0/aocc"))
    parser.add_argument("--arch", default="gfx1200")
    parser.add_argument("--objects", type=Path, default=src,
                        help="directory of magma_openmx.o and gemmul8_bridge.o (default: source/)")
    parser.add_argument("--reps", type=int, default=3, help="solves per route; the best time is printed")
    parser.add_argument("--run", nargs="*", metavar="NxM3",
                        help="run after building, for these shapes (default 500x300 700x420 900x540 1100x660)")
    args = parser.parse_args()
    out = args.build_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    objects = args.objects.resolve()
    for needed in (objects / "magma_openmx.o", objects / "gemmul8_bridge.o",
                   src / "third_party/GEMMul8/lib/libgemmul8.a"):
        if not needed.exists():
            parser.error(f"{needed} is missing: build OpenMX first")

    source = (src / "Krylov.c").read_text()
    parts = [PREFIX]
    for name in ["KRYLOV_ENABLE_GPU", "KRYLOV_GPU_EIGEN_MIN", "KRYLOV_GPU_DGEMM_MIN_FLOPS"]:
        parts.append(define(source, name))
    for name in ["Krylov_gpu_dgemm_min_flops", "Krylov_gpu_eigen_min", "Krylov_gpu_fused", "Krylov_gpu_gemmul8",
                 "Krylov_gpu_fits", "Krylov_gpu_alloc_failure_reported", "Krylov_gpu_device_ranks",
                 "Krylov_gpu_need_total"]:
        parts.append(variable(source, name))
    parts.append(function(source, "Krylov_GPU_Enabled"))
    parts.append(typedef(source, "Krylov_GPU_Workspace"))
    parts.append(between(source, "static void Krylov_GPU_InitOnce(void);",
                         "static void Krylov_Eigen2(Krylov_GPU_Workspace *ws, double *a, int csize, "
                         "double *ko, int n, int EVmax);"))
    parts.append(variable(source, "Krylov_gpu_ws_pool"))
    parts.append(variable(source, "Krylov_gpu_ws_pool_size"))
    for name in ["Krylov_GPU_InitOnce", "Krylov_GPU_PreparePool", "Krylov_GPU_GetWorkspace",
                 "Krylov_GPU_ReportAllocFailure", "Krylov_GPU_EnsureDouble", "Krylov_GPU_Usable",
                 "Krylov_GPU_EnsureHostMatrix", "Krylov_GPU_Workspace_Init", "Krylov_GPU_Workspace_Free",
                 "Krylov_GPU_EnsureHipblas", "Krylov_GPU_DeviceDgemm", "Krylov_GPU_GemmFailed",
                 "Krylov_Dgemm", "Krylov_Eigen2"]:
        parts.append(function(source, name))
    parts += [typedef(source, "Krylov_KUCache"), variable(source, "Krylov_kucache"),
              function(source, "Krylov_Release_GPU_KUCache", static=False),
              function(source, "Krylov_Release_GPU_Caches", static=False)]
    for name in ["Krylov_GPU_Preflight", "Krylov_KUCache_Get", "Krylov_ProjectedSolve_GPU"]:
        parts.append(function(source, name))
    (out / "krylov_gpu.c").write_text("\n\n".join(parts + [MAIN]))

    clang = f"{args.rocm}/llvm/bin/clang"
    magma = src / "third_party/magma"
    run = lambda cmd: subprocess.run(cmd, check=True, cwd=out)
    run([clang, "-O2", "-std=c11", "-D__HIP_PLATFORM_AMD__", "-Wno-implicit-function-declaration",
         f"-I{src}", f"-I{args.rocm}/include", "-c", "krylov_gpu.c", "-o", "krylov_gpu.o"])
    run([f"{args.rocm}/bin/hipcc", f"--offload-arch={args.arch}", "krylov_gpu.o",
         str(objects / "gemmul8_bridge.o"), str(objects / "magma_openmx.o"),
         str(src / "third_party/GEMMul8/lib/libgemmul8.a"),
         f"-L{magma}/lib", "-lmagma", f"-Wl,-rpath,{magma}/lib",
         f"-L{args.aocl}/lib", "-lflame", "-lblis", f"-Wl,-rpath,{args.aocl}/lib",
         f"-L{args.rocm}/lib", "-lhipsolver", "-lhipblas", "-lhipblaslt", "-lrocblas",
         "-o", "check_krylov_gpu"])
    print(f"Built {out / 'check_krylov_gpu'}")
    if args.run is not None:
        env = os.environ.copy()
        for name in ("OPENMX_KRYLOV_GEMMUL8", "OPENMX_KRYLOV_GPU_FUSED", "OPENMX_KRYLOV_GPU_EIGEN_MIN",
                     "OPENMX_KRYLOV_GPU_DGEMM_MIN_FLOPS", "OPENMX_KRYLOV_GPU"):
            env.pop(name, None)
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("BLIS_NUM_THREADS", "1")
        subprocess.run([str(out / "check_krylov_gpu"), f"--reps={args.reps}", *args.run], check=True, env=env)


if __name__ == "__main__":
    main()

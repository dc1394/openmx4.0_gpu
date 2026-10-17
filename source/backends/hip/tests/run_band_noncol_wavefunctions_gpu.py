#!/usr/bin/env python3
"""Check the non-collinear band host-path wave functions on the GPU (small, one process).

BandNonCol_DenseWavefunctions_OpenMP and ..._PresentOpenMP of
source/Band_DFT_NonCol.c (the spin-half back transform: rows MaxN.. of Hs2
cleared, then one MaxN x n x n product against S per spin half, with only the
first MaxN columns of Cs2 mapped) are cut out of the production file and
built as an OpenMP-offload program; openmx_gemmul8Zgemm is replaced by plain
hipBLAS ZGEMM so that the check isolates the mapping, the offsets and the
leading dimensions from GEMMul8's rounding.  The result must match a host
evaluation of the former n2 x n2 x n2 product Hs2 = Cs2^T Ss2^T with
Ss2 = diag(S, S) in its first MaxN rows (to 1e-12 relative), the rows from
MaxN on must be zero, the host Cs2 and S unchanged, and nothing may remain
mapped.  Several shapes, MaxN = n2 included.  Needs ROCm (hipBLAS) and a GPU;
uses a few MiB of device memory.
"""

import argparse
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def function(source, name):
    match = re.search(r"^static [^;{}]*\b" + re.escape(name) + r"\([^;{]*?\)\s*\{", source, re.M)
    if match is None:
        raise ValueError(f"Cannot find production function {name}")
    pos, depth = source.index("{", match.start()) + 1, 1
    while depth:
        depth += (source[pos] == "{") - (source[pos] == "}")
        pos += 1
    return source[match.start():pos]


PREFIX = r'''
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <hip/hip_runtime_api.h>
#include <hip/hip_complex.h>
#include <hipblas/hipblas.h>

#define CHECK(condition) do { if (!(condition)) { \
    fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #condition); exit(2); } } while (0)
#define wait_hipfunc(call) CHECK((int)(call) == 0)

typedef struct { double r, i; } dcomplex;
typedef struct { int initialized; hipStream_t stream; hipblasHandle_t hipblas; } BandNonColDMGpuWorkspace;
static BandNonColDMGpuWorkspace BandNonCol_dm_gpu_workspace;
static int gemm_calls;

static void BandNonCol_DMGpu_Init(void)
{
    BandNonColDMGpuWorkspace *w = &BandNonCol_dm_gpu_workspace;
    if (w->initialized) return;
    wait_hipfunc(hipStreamCreateWithFlags(&w->stream, hipStreamNonBlocking));
    wait_hipfunc(hipblasCreate(&w->hipblas));
    wait_hipfunc(hipblasSetStream(w->hipblas, w->stream));
    w->initialized = 1;
}

/* plain hipBLAS for GEMMul8: the same call, FP64 rounding */
static hipblasStatus_t openmx_gemmul8Zgemm(hipblasHandle_t handle, hipblasOperation_t transa,
                                           hipblasOperation_t transb, int m, int n, int k,
                                           const hipDoubleComplex *alpha, const hipDoubleComplex *A, int lda,
                                           const hipDoubleComplex *B, int ldb, const hipDoubleComplex *beta,
                                           hipDoubleComplex *C, int ldc)
{
    gemm_calls++;
    return hipblasZgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}
'''

HARNESS = r'''
static void run_case(int n, int MaxN, unsigned seed)
{
    const int n2 = 2 * n;
    const size_t nn = (size_t)n * n, n2n2 = (size_t)n2 * n2;
    dcomplex *Cs2 = malloc(sizeof(dcomplex) * n2n2), *S = malloc(sizeof(dcomplex) * nn);
    dcomplex *Hs2 = malloc(sizeof(dcomplex) * n2n2), *Cs2_copy = malloc(sizeof(dcomplex) * n2n2);
    dcomplex *S_copy = malloc(sizeof(dcomplex) * nn);
    double worst = 0.0, scale = 0.0;

    CHECK(Cs2 && S && Hs2 && Cs2_copy && S_copy);
    srand(seed);
    for (size_t i = 0; i < n2n2; i++) {
        Cs2[i].r = (double)rand() / RAND_MAX - 0.5;
        Cs2[i].i = (double)rand() / RAND_MAX - 0.5;
        Hs2[i].r = 1.0e30;   /* stale contents must not survive */
        Hs2[i].i = -1.0e30;
    }
    for (size_t i = 0; i < nn; i++) {
        S[i].r = (double)rand() / RAND_MAX - 0.5;
        S[i].i = (double)rand() / RAND_MAX - 0.5;
    }
    memcpy(Cs2_copy, Cs2, sizeof(dcomplex) * n2n2);
    memcpy(S_copy, S, sizeof(dcomplex) * nn);
    gemm_calls = 0;

    BandNonCol_DenseWavefunctions_OpenMP(n, n2, MaxN, Cs2, S, Hs2);

    CHECK(gemm_calls == 2);
    CHECK(memcmp(Cs2, Cs2_copy, sizeof(dcomplex) * n2n2) == 0 && memcmp(S, S_copy, sizeof(dcomplex) * nn) == 0);
    CHECK(!omp_target_is_present(Hs2, omp_get_default_device()));
    CHECK(!omp_target_is_present(Cs2, omp_get_default_device()));
    CHECK(!omp_target_is_present(S, omp_get_default_device()));
    /* the former product: Hs2[l + i n2] = sum_j Cs2[j + l n2] Ss2[i + j n2], Ss2 = diag(S, S) */
    for (int l = 0; l < n2; l++) {
        for (int i = 0; i < n2; i++) {
            const dcomplex h = Hs2[(size_t)l + (size_t)i * n2];
            if (l >= MaxN) {
                CHECK(h.r == 0.0 && h.i == 0.0);
                continue;
            }
            double re = 0.0, im = 0.0;
            for (int j = 0; j < n2; j++) {
                const int same = (i < n) == (j < n);
                if (!same) continue;
                {
                    const dcomplex c = Cs2[(size_t)j + (size_t)l * n2];
                    const dcomplex s = S[(size_t)(i % n) + (size_t)(j % n) * n];
                    re += c.r * s.r - c.i * s.i;
                    im += c.r * s.i + c.i * s.r;
                }
            }
            worst = fmax(worst, fmax(fabs(h.r - re), fabs(h.i - im)));
            scale = fmax(scale, fmax(fabs(re), fabs(im)));
        }
    }
    CHECK(worst <= 1.0e-12 * (scale > 1.0 ? scale : 1.0));
    printf("n=%d n2=%d MaxN=%d: max |dHs2| %.2e (scale %.2e), rows %d.. zero\n", n, n2, MaxN, worst, scale, MaxN);
    free(Cs2); free(S); free(Hs2); free(Cs2_copy); free(S_copy);
}

int main(void)
{
    CHECK(omp_get_num_devices() > 0);
    run_case(16, 11, 1u);
    run_case(48, 61, 2u);
    run_case(48, 96, 3u);   /* MaxN = n2: no row to clear */
    run_case(100, 37, 4u);
    printf("PASS: spin-half wave functions against the n2^3 product, cleared rows, inputs kept, nothing left mapped\n");
    return 0;
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rocm", default=os.environ.get("ROCM_PATH", "/opt/rocm-7.2.0"))
    parser.add_argument("--arch", default=os.environ.get("OFFLOAD_ARCH", "gfx1200"))
    args = parser.parse_args()
    nc = (ROOT / "source/Band_DFT_NonCol.c").read_text()
    source_text = "\n\n".join([PREFIX, function(nc, "BandNonCol_OmpTargetIsPresent"),
                               function(nc, "BandNonCol_DenseWavefunctions_PresentOpenMP"),
                               function(nc, "BandNonCol_DenseWavefunctions_OpenMP"), HARNESS])
    clang = os.path.join(args.rocm, "llvm/bin/clang")
    with tempfile.TemporaryDirectory(prefix="openmx-band-wf-gpu-") as directory:
        source = Path(directory) / "wf.c"
        binary = Path(directory) / "wf"
        source.write_text(source_text)
        subprocess.run([clang, "-O2", "-std=c11", "-D__HIP_PLATFORM_AMD__", "-fopenmp",
                        f"--offload-arch={args.arch}", "-I" + os.path.join(args.rocm, "include"),
                        str(source), "-o", str(binary), "-L" + os.path.join(args.rocm, "lib"),
                        "-Wl,-rpath," + os.path.join(args.rocm, "lib"), "-lhipblas", "-lamdhip64", "-lm"],
                       check=True)
        subprocess.run([str(binary)], check=True, timeout=120)


if __name__ == "__main__":
    main()

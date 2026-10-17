#!/usr/bin/env python3
"""Check the host-staged back transform of the non-collinear cluster solver
without a GPU.

ClusterNonCol_BackTransformHostStaged of source/Cluster_DFT_NonCol.c is
extracted verbatim and built against host stubs: the device attempt
(ClusterNonCol_BackTransformHost_HIP) either reports no room, so that the
CPU form runs (the transpose of the solved vectors and one real dgemm per
spin half against the transformed overlap S), or succeeds, in which case
the CPU form must not touch the output.  The vectors of the transformed
problem V (n2 x n2 complex, the first maxn columns solved) and S (n x n) are
random; the result (state l of basis i at l + i ld) is checked against a
long-double evaluation of

    dense_evec[l + i ld] = sum_j S[i mod n + (j mod n) n] V[j + l n2]

over the spin block of i, i.e. the former complex GEMM against
diag(S, S); the rows maxn..ld-1 of a wider output must stay untouched.
This checks the index arithmetic of the CPU form (only reached when the
device has no room for the device form) with real BLAS, not the GPU.
--sanitize adds ASan/UBSan.
"""

import argparse
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile


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


PREFIX = r'''
#include <math.h>
#include <stdint.h>
#include <stdio.h>
static int level_stdout=3;
#include <stdlib.h>
#include <string.h>

#define CHECK(condition) do { if (!(condition)) { \
    fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #condition); exit(2); \
} } while (0)
typedef int INTEGER;
typedef void *hipblasHandle_t;
typedef struct { double r, i; } dcomplex;
#define F77_NAME(lower, upper) lower ## _
extern void dgemm_(const char *, const char *, const int *, const int *, const int *,
                   const double *, const double *, const int *, const double *,
                   const int *, const double *, double *, const int *);

static int device_rc;      /* what the device attempt returns */
static int device_calls;

static size_t ClusterNonCol_CheckedMulCount(size_t a, size_t b, const char *label)
{
    (void)label;
    CHECK(a == 0 || b <= SIZE_MAX / a);
    return a * b;
}

static void *ClusterNonCol_MallocArray(size_t count, size_t elem, const char *label)
{
    void *ptr = malloc(ClusterNonCol_CheckedMulCount(count, elem, label) + 1);
    CHECK(ptr != NULL);
    return ptr;
}

static int ClusterNonCol_MpiRankForLog(void) { return 0; }

static int ClusterNonCol_BackTransformHost_HIP(hipblasHandle_t handle, int n, int n2, int maxn,
                                               const dcomplex *V, const double *S, dcomplex *evec, int ld)
{
    (void)handle; (void)n; (void)n2; (void)maxn; (void)V; (void)S; (void)evec; (void)ld;
    device_calls++;
    return device_rc;
}
'''

MAIN = r'''
static double uniform(uint64_t *state)
{
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return 2.0 * ((double)((*state * 0x2545F4914F6CDD1DULL) >> 11) / 9007199254740992.0) - 1.0;
}

static double run(int n, int maxn, int ld, int rc)
{
    const int n2 = 2 * n;
    const double sentinel = 12345.0;
    uint64_t state = 0x9E3779B97F4A7C15ULL ^ ((uint64_t)n << 20) ^ (uint64_t)(maxn * 7 + ld);
    double *S = malloc(sizeof(double) * (size_t)n * n);
    dcomplex *V = malloc(sizeof(dcomplex) * (size_t)n2 * n2);
    dcomplex *out = malloc(sizeof(dcomplex) * (size_t)ld * n2);
    double max_ref = 0.0, max_err = 0.0;

    CHECK(S != NULL && V != NULL && out != NULL);
    for (size_t i = 0; i < (size_t)n * n; i++) S[i] = uniform(&state);
    for (size_t i = 0; i < (size_t)n2 * n2; i++) {
        V[i].r = uniform(&state);
        V[i].i = uniform(&state);
    }
    for (size_t i = 0; i < (size_t)ld * n2; i++) out[i].r = out[i].i = sentinel;

    device_rc = rc;
    device_calls = 0;
    ClusterNonCol_BackTransformHostStaged(n, n2, maxn, V, S, out, ld, NULL);
    CHECK(device_calls == 1);

    for (int i = 0; i < n2; i++) {
        for (int l = 0; l < ld; l++) {
            const dcomplex got = out[(size_t)l + (size_t)i * ld];

            if (rc == 0 || maxn <= l) {
                /* the device form wrote it (stubbed: nothing), or no state */
                CHECK(got.r == sentinel && got.i == sentinel);
                continue;
            }
            {
                const int half = i / n, ii = i % n;
                long double re = 0.0L, im = 0.0L;

                for (int jj = 0; jj < n; jj++) {
                    const dcomplex v = V[(size_t)(half * n + jj) + (size_t)l * n2];
                    const long double s = S[(size_t)ii + (size_t)jj * n];
                    re += s * v.r;
                    im += s * v.i;
                }
                if (max_ref < fabs((double)re)) max_ref = fabs((double)re);
                if (max_ref < fabs((double)im)) max_ref = fabs((double)im);
                if (max_err < fabs(got.r - (double)re)) max_err = fabs(got.r - (double)re);
                if (max_err < fabs(got.i - (double)im)) max_err = fabs(got.i - (double)im);
            }
        }
    }
    free(out);
    free(V);
    free(S);
    return (0.0 < max_ref) ? max_err / max_ref : 0.0;
}

int main(void)
{
    static const int shapes[][3] = {{7, 14, 14}, {7, 9, 9}, {13, 9, 20}, {37, 50, 61}, {64, 128, 128},
                                    {101, 77, 77}};
    double worst = 0.0;
    int checks = 0;

    for (size_t s = 0; s < sizeof(shapes) / sizeof(shapes[0]); s++) {
        const int n = shapes[s][0], maxn = shapes[s][1], ld = shapes[s][2];
        double err = run(n, maxn, ld, 1);

        CHECK(err < 1e-13);
        if (worst < err) worst = err;
        (void)run(n, maxn, ld, 0);
        checks += 2;
    }
    printf("PASS: %d shapes (CPU form and device-form short cut), max relative error %.2e\n", checks, worst);
    return 0;
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cc", default=os.environ.get("CC", "cc"))
    parser.add_argument("--sanitize", action="store_true")
    parser.add_argument("--libs", default=os.environ.get("BLAS_LIBS", "-lblas"),
                        help="LP64 CPU BLAS libraries (default: -lblas)")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    source = (root / "source" / "Cluster_DFT_NonCol.c").read_text()
    code = PREFIX + function(source, "ClusterNonCol_BackTransformHostStaged") + "\n" + MAIN
    flags = ["-O1", "-g", "-Wall", "-Wno-unused-function"]
    if args.sanitize:
        flags += ["-fsanitize=address,undefined", "-fno-omit-frame-pointer"]
    with tempfile.TemporaryDirectory(prefix="nc-host-back-") as work:
        harness = Path(work) / "harness.c"
        binary = Path(work) / "harness"
        harness.write_text(code)
        subprocess.run([*shlex.split(args.cc), *flags, "-std=c11", str(harness),
                        *shlex.split(args.libs), "-lm", "-o", str(binary)], check=True)
        env = dict(os.environ)
        if args.sanitize:
            env.setdefault("ASAN_OPTIONS", "detect_leaks=1")
        subprocess.run([str(binary)], env=env, check=True)


if __name__ == "__main__":
    main()

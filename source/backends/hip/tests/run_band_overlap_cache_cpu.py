#!/usr/bin/env python3
"""Check the band solvers' transformed-overlap caches on the CPU.

The k-point overlap caches of source/Band_DFT_NonCol.c (BandNonCol_KOverlap*,
OPENMX_BAND_NONCOL_OVERLAP_CACHE_MB, a host copy of the transformed S(k) per
k-point of the k-dense path) and of source/Band_DFT_Col.c (BandCol_KOverlap*,
OPENMX_BAND_COL_OVERLAP_CACHE_MB, the transformed S(k) on the device and the
overlap eigenvalues behind it) are cut out of the production files and built
against host stubs of the HIP copies and of the solver context; no device
code runs.  Checked: a saved k-point restores bit for bit (and the
collinear one its n overlap eigenvalues to ko[1..n] and koS[1..n], ko[0]
untouched), a k-point of other coordinates or another index misses, an
index out of range is refused on save and restore, the per-rank budget
(the variable's MiB, at most 1/fraction of MemAvailable over the ranks)
fills in order and stops the later k-points (hits and misses mixed), the
budget 0 disables the cache, a new size or k count resets it, the reset
frees every entry, and the collinear restore sizes the device buffers
(BandCol_GpuSolver_EnsureMatrixCapacity) before its copy.
--cc (or CC) and CFLAGS select the compiler, --sanitize adds ASan/UBSan.
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


def block(source, first, last):
    start = source.index(first)
    end = source.index(last, start) + len(last)
    return source[start:end]


PREFIX = r'''
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(condition) do { if (!(condition)) { \
    fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #condition); exit(2); } } while (0)

typedef struct { double r, i; } dcomplex;
typedef int hipError_t;
enum { hipSuccess = 0, hipMemcpyHostToDevice = 1, hipMemcpyDeviceToHost = 2 };
#define wait_hipfunc(call) CHECK((call) == hipSuccess)

static int BandCol_mgpu_host_dense=0;
static dcomplex *BandCol_host_s;
static void BandCol_HostEnsure(int n){(void)n;CHECK(0);}
static int memcpy_h2d, memcpy_d2h;
static hipError_t hipMemcpy(void *dst, const void *src, size_t bytes, int kind)
{
    if (kind == hipMemcpyHostToDevice) memcpy_h2d++;
    else memcpy_d2h++;
    memcpy(dst, src, bytes);
    return hipSuccess;
}

static int local_ranks = 4;
int openmx_gpu_local_size_noncollective(void) { return local_ranks; }

/* the collinear solver context: only what the cache touches */
typedef struct {
    dcomplex *d_S;
    int       matrix_dim;
    int       transformed_s_valid;
    int       transformed_s_dim;
} BandColGpuSolverCtx;
static BandColGpuSolverCtx BandCol_gpusolver_ctx;
static int ensure_calls;
static void BandCol_GpuSolver_EnsureMatrixCapacity(int n)
{
    ensure_calls++;
    if (n <= BandCol_gpusolver_ctx.matrix_dim) return;
    free(BandCol_gpusolver_ctx.d_S);
    BandCol_gpusolver_ctx.d_S = calloc((size_t)n * (size_t)n, sizeof(dcomplex));
    CHECK(BandCol_gpusolver_ctx.d_S != NULL);
    BandCol_gpusolver_ctx.matrix_dim = n;
    BandCol_gpusolver_ctx.transformed_s_valid = 0;
}
'''

HARNESS = r'''
static unsigned long checks;

static double available_mib(void)
{
    FILE *file = fopen("/proc/meminfo", "r");
    char line[256];
    unsigned long long kib = 0;
    CHECK(file != NULL);
    while (fgets(line, sizeof(line), file) != NULL)
        if (sscanf(line, "MemAvailable: %llu kB", &kib) == 1) break;
    fclose(file);
    return (double)kib / 1024.0;
}

static void fill(dcomplex *m, size_t count, double seed)
{
    for (size_t i = 0; i < count; i++) {
        m[i].r = seed + 0.001 * (double)i;
        m[i].i = -seed + 1.0e-7 * (double)i;
    }
}

static void test_noncol(void)
{
    const int n = 24, points = 9;
    const size_t nn = (size_t)n * (size_t)n, bytes = nn * sizeof(dcomplex);
    dcomplex *a = malloc(bytes), *b = malloc(bytes);
    CHECK(a != NULL && b != NULL);

    /* the budget: the variable's MiB, at most 1/64 of MemAvailable over the ranks */
    setenv("OPENMX_BAND_NONCOL_OVERLAP_CACHE_MB", "3", 1);
    {
        double avail = available_mib() / (double)local_ranks / 64.0;
        double expected = (avail < 3.0) ? avail : 3.0;
        double limit = (double)BandNonCol_KOverlapLimit() / 1048576.0;
        CHECK(limit <= 3.0 + 1e-9);
        CHECK(limit >= expected - 0.01);
        checks += 2;
    }

    /* three entries fit (bytes = 9216 each with n = 24) */
    {
        char value[64];
        snprintf(value, sizeof(value), "%.10f", 3.5 * (double)bytes / 1048576.0);
        setenv("OPENMX_BAND_NONCOL_OVERLAP_CACHE_MB", value, 1);
    }
    BandNonCol_KOverlapPrepare(n, points);
    CHECK(BandNonCol_k_overlap != NULL && BandNonCol_k_overlap_bytes == bytes);
    for (int k = 0; k < points; k++) CHECK(!BandNonCol_KOverlapRestore(k, 0.1 * k, 0.2, 0.3, b));
    for (int k = 0; k < points; k++) {
        fill(a, nn, 1.0 + k);
        BandNonCol_KOverlapSave(k, 0.1 * k, 0.2, 0.3, a);
    }
    CHECK(BandNonCol_k_overlap_used == 3 * bytes);
    for (int k = 0; k < points; k++) {
        const int hit = BandNonCol_KOverlapRestore(k, 0.1 * k, 0.2, 0.3, b);
        CHECK(hit == (k < 3));
        if (hit) {
            fill(a, nn, 1.0 + k);
            CHECK(memcmp(a, b, bytes) == 0);
        }
        checks++;
    }
    /* other coordinates or index: a miss; out of range: refused */
    CHECK(!BandNonCol_KOverlapRestore(1, 0.1, 0.2, 0.31, b));
    CHECK(!BandNonCol_KOverlapRestore(-1, 0.0, 0.2, 0.3, b));
    CHECK(!BandNonCol_KOverlapRestore(points, 0.0, 0.2, 0.3, b));
    BandNonCol_KOverlapSave(points, 0.0, 0.2, 0.3, a);
    BandNonCol_KOverlapSave(-3, 0.0, 0.2, 0.3, a);
    CHECK(BandNonCol_k_overlap_used == 3 * bytes);
    /* a later save of a cached k-point replaces it in place */
    fill(a, nn, 42.0);
    BandNonCol_KOverlapSave(2, 0.5, 0.6, 0.7, a);
    CHECK(BandNonCol_k_overlap_used == 3 * bytes);
    CHECK(!BandNonCol_KOverlapRestore(2, 0.2, 0.2, 0.3, b));
    CHECK(BandNonCol_KOverlapRestore(2, 0.5, 0.6, 0.7, b) && memcmp(a, b, bytes) == 0);
    /* the same shape keeps the entries, another one resets them */
    BandNonCol_KOverlapPrepare(n, points);
    CHECK(BandNonCol_KOverlapRestore(0, 0.0, 0.2, 0.3, b));
    BandNonCol_KOverlapPrepare(n, points + 1);
    CHECK(BandNonCol_k_overlap_used == 0 && !BandNonCol_KOverlapRestore(0, 0.0, 0.2, 0.3, b));
    BandNonCol_KOverlapReset();
    CHECK(BandNonCol_k_overlap == NULL && BandNonCol_k_overlap_points == 0);
    /* 0 disables, and a too small budget too */
    setenv("OPENMX_BAND_NONCOL_OVERLAP_CACHE_MB", "0", 1);
    BandNonCol_KOverlapPrepare(n, points);
    CHECK(BandNonCol_k_overlap == NULL);
    BandNonCol_KOverlapSave(0, 0.0, 0.2, 0.3, a);
    CHECK(!BandNonCol_KOverlapRestore(0, 0.0, 0.2, 0.3, b));
    setenv("OPENMX_BAND_NONCOL_OVERLAP_CACHE_MB", "0.001", 1);
    BandNonCol_KOverlapPrepare(n, points);
    CHECK(BandNonCol_k_overlap == NULL);
    unsetenv("OPENMX_BAND_NONCOL_OVERLAP_CACHE_MB");
    BandNonCol_KOverlapPrepare(n, points);
    CHECK(BandNonCol_k_overlap != NULL);
    BandNonCol_KOverlapReset();
    free(a);
    free(b);
    checks += 20;
}

static void test_col(void)
{
    const int n = 20, points = 5;
    const size_t nn = (size_t)n * (size_t)n, matrix_bytes = nn * sizeof(dcomplex);
    double ko[21], koS[21], saved[21];
    dcomplex *dev;

    setenv("OPENMX_BAND_COL_OVERLAP_CACHE_MB", "1", 1);
    BandCol_KOverlapPrepare(n, points);
    CHECK(BandCol_k_overlap != NULL);
    CHECK(BandCol_k_overlap_bytes == matrix_bytes + (size_t)n * sizeof(double));
    BandCol_GpuSolver_EnsureMatrixCapacity(n);
    dev = BandCol_gpusolver_ctx.d_S;
    for (int k = 0; k < points; k++) {
        fill(dev, nn, 3.0 + k);
        for (int l = 1; l <= n; l++) koS[l] = 0.5 + 0.01 * l + k;
        BandCol_KOverlapSave(k, 0.25 * k, 0.0, 0.0, koS);
    }
    for (int k = points - 1; 0 <= k; k--) {
        /* as a later k-point leaves them */
        for (int l = 0; l <= n; l++) ko[l] = koS[l] = -77.0;
        fill(BandCol_gpusolver_ctx.d_S, nn, 0.0);
        BandCol_gpusolver_ctx.transformed_s_valid = 0;
        ensure_calls = 0;
        CHECK(BandCol_KOverlapRestore(k, 0.25 * k, 0.0, 0.0, ko, koS));
        CHECK(ensure_calls == 1);
        CHECK(BandCol_gpusolver_ctx.transformed_s_valid == 1 && BandCol_gpusolver_ctx.transformed_s_dim == n);
        {
            dcomplex *expected = malloc(matrix_bytes);
            CHECK(expected != NULL);
            fill(expected, nn, 3.0 + k);
            CHECK(memcmp(expected, BandCol_gpusolver_ctx.d_S, matrix_bytes) == 0);
            free(expected);
        }
        for (int l = 1; l <= n; l++) saved[l] = 0.5 + 0.01 * l + k;
        CHECK(memcmp(ko + 1, saved + 1, sizeof(double) * (size_t)n) == 0);
        CHECK(memcmp(koS + 1, saved + 1, sizeof(double) * (size_t)n) == 0);
        CHECK(ko[0] == -77.0 && koS[0] == -77.0);
        checks += 6;
    }
    /* a restore sizes the device buffers itself (released after a turn) */
    free(BandCol_gpusolver_ctx.d_S);
    BandCol_gpusolver_ctx.d_S = NULL;
    BandCol_gpusolver_ctx.matrix_dim = 0;
    CHECK(BandCol_KOverlapRestore(1, 0.25, 0.0, 0.0, ko, koS) && BandCol_gpusolver_ctx.d_S != NULL);
    CHECK(!BandCol_KOverlapRestore(1, 0.26, 0.0, 0.0, ko, koS));
    CHECK(!BandCol_KOverlapRestore(points, 0.0, 0.0, 0.0, ko, koS));
    CHECK(!BandCol_KOverlapRestore(-1, 0.0, 0.0, 0.0, ko, koS));
    BandCol_KOverlapSave(points, 0.0, 0.0, 0.0, koS);
    BandCol_KOverlapSave(-1, 0.0, 0.0, 0.0, koS);
    /* budget: with room for two entries the later k-points miss */
    BandCol_KOverlapReset();
    {
        char value[64];
        snprintf(value, sizeof(value), "%.10f",
                 2.5 * (double)(matrix_bytes + (size_t)n * sizeof(double)) / 1048576.0);
        setenv("OPENMX_BAND_COL_OVERLAP_CACHE_MB", value, 1);
    }
    BandCol_KOverlapPrepare(n, points);
    for (int k = 0; k < points; k++) BandCol_KOverlapSave(k, 0.25 * k, 0.0, 0.0, koS);
    for (int k = 0; k < points; k++) CHECK(BandCol_KOverlapRestore(k, 0.25 * k, 0.0, 0.0, ko, koS) == (k < 2));
    BandCol_KOverlapReset();
    CHECK(BandCol_k_overlap == NULL && BandCol_k_overlap_used == 0);
    free(BandCol_gpusolver_ctx.d_S);
    checks += 12;
}

int main(void)
{
    test_noncol();
    test_col();
    printf("PASS: %lu checks; non-collinear and collinear overlap caches: bitwise restore, eigenvalues, "
           "misses, index range, budget, disable, reset\n", checks);
    return 0;
}
'''


def make_harness():
    nc = (ROOT / "source/Band_DFT_NonCol.c").read_text()
    col = (ROOT / "source/Band_DFT_Col.c").read_text()
    parts = [PREFIX,
             "static void BandNonCol_AbortWithMessage(const char *msg) { fprintf(stderr, \"%s\\n\", msg); exit(3); }",
             function(nc, "BandNonCol_HostCacheLimit"),
             block(nc, "typedef struct {\n    dcomplex *matrix;\n    double k1, k2, k3;\n} BandNonColKOverlap;",
                   "static size_t BandNonCol_k_overlap_bytes = 0, BandNonCol_k_overlap_used = 0;"),
             *(function(nc, name) for name in ("BandNonCol_KOverlapReset", "BandNonCol_KOverlapLimit",
                                               "BandNonCol_KOverlapPrepare", "BandNonCol_KOverlapRestore",
                                               "BandNonCol_KOverlapSave")),
             block(col, "typedef struct {\n    dcomplex *matrix;\n    double k1, k2, k3;\n} BandColKOverlap;",
                   "static size_t BandCol_k_overlap_bytes = 0, BandCol_k_overlap_used = 0;"),
             *(function(col, name) for name in ("BandCol_KOverlapReset", "BandCol_HostCacheLimit",
                                                "BandCol_KOverlapPrepare", "BandCol_KOverlapRestore",
                                                "BandCol_KOverlapSave")),
             HARNESS]
    return "\n\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cc", default=os.environ.get("CC", "cc"))
    parser.add_argument("--sanitize", action="store_true")
    args = parser.parse_args()
    flags = shlex.split(os.environ.get("CFLAGS", "-O2"))
    if args.sanitize:
        flags += ["-g", "-fsanitize=address,undefined", "-fno-omit-frame-pointer"]
    with tempfile.TemporaryDirectory(prefix="openmx-band-overlap-") as directory:
        source = Path(directory) / "overlap.c"
        binary = Path(directory) / "overlap"
        source.write_text(make_harness())
        subprocess.run([*shlex.split(args.cc), *flags, "-std=c11", "-D_DEFAULT_SOURCE", str(source),
                        "-o", str(binary)], check=True)
        subprocess.run([str(binary)], check=True)


if __name__ == "__main__":
    main()

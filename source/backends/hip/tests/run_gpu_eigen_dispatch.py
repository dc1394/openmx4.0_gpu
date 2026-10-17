#!/usr/bin/env python3
"""Check the production dense eigensolver dispatch and retry without a GPU.

Extract the policy, public entries and retry loop verbatim from
magma_openmx.cu.  HIP discovery and the numerical backends are stubbed:
record which backend runs, overwrite the matrix before an injected failure,
and require a retry to see the original matrix.  This catches a direct
cluster entry bypassing the policy even when the policy helper is correct.
"""

import argparse
import os
from pathlib import Path
import re
import subprocess
import tempfile


def function(source, name):
    match = re.search(r'^(?:extern "C" )?[^;{}\n]*\b' + re.escape(name)
                      + r'\([^;]*?\)\s*\{', source, re.MULTILINE)
    if match is None:
        raise ValueError(f"Cannot find production function {name}")
    brace = source.index("{", match.start())
    depth, end = 1, brace + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[match.start():end]


STUBS = r'''
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <strings.h>
#include <string>

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); \
    std::exit(2); } } while (0)
#define OPENMX_STDOUT_LEVEL(n) 1
constexpr int hipSuccess = 0, hipMemcpyDefault = 0;
constexpr int MAGMA_ERR_ILLEGAL_VALUE = -101, MAGMA_ERR_UNKNOWN = -199;
struct magmaDoubleComplex { double r, i; };
struct hipDeviceProp_t { char gcnArchName[256]; int integrated; };
static int MYID_MPI_COMM_WORLD = 0;
static std::mutex g_magma_mutex;
static std::string attempts;
static int device_queries = 0;
static int hipGetDevice(int *dev) { ++device_queries; *dev = 0; return 0; }
static int hipGetDeviceProperties(hipDeviceProp_t *p, int) {
    ++device_queries;
    if (std::getenv("TEST_DISCOVERY_FAIL")) return 1;
    std::snprintf(p->gcnArchName, sizeof(p->gcnArchName), "%s", std::getenv("TEST_ARCH"));
    p->integrated = std::atoi(std::getenv("TEST_INTEGRATED"));
    return 0;
}
static int hipGetLastError() { return 0; }
static int hipFree(void *p) { std::free(p); return 0; }
static int hipMemcpy(void *dst, const void *src, size_t n, int) { std::memcpy(dst, src, n); return 0; }
static int hipDeviceSynchronize() { return 0; }
#define OPENMX_AMDGPU_HIP_RUNTIME_COMPAT_H
#include "openmx_gpu_arch_policy.h"
struct EigenBackup { void *ptr = nullptr; size_t bytes = 0; int host = 0, kept = 0; };
static int backup_save(EigenBackup *b, void *a, size_t n, size_t) {
    b->ptr = std::malloc(n); b->bytes = n; std::memcpy(b->ptr, a, n); return 1;
}
static int backup_restore(EigenBackup *b, void *a) { std::memcpy(a, b->ptr, b->bytes); return 0; }
static void backup_free(EigenBackup *b) { std::free(b->ptr); }
static void backup_report(EigenBackup *, int, int, int, const char *) {}
static void *solve_copy_alloc(size_t n) { return std::malloc(n); }
enum { ATTEMPT_ROCSOLVER, ATTEMPT_MAGMA_OZAKI2, ATTEMPT_MAGMA };
static int backend(char which, int n, int maxn, void *a, double *w, int *mout, int *touched) {
    CHECK(((double *)a)[0] == 12.0); // Every retry must restore the damaged input.
    attempts += which;
    *touched = 1;
    ((double *)a)[0] = -99.0;
    const char *fail = std::getenv("TEST_FAIL_BACKEND");
    if (fail && std::strchr(fail, which)) return std::atoi(std::getenv("TEST_FAIL_INFO"));
    for (int i = 0; i < maxn; ++i) w[i] = i + 1.0;
    *mout = maxn;
    ((double *)a)[0] = 42.0;
    return 0;
}
static int rocsolver_evd_locked(int, int n, int maxn, void *a, double *w, int *mout, int *touched, int) {
    return backend('R', n, maxn, a, w, mout, touched);
}
static int magma_evdx_locked(int, int ozaki, int n, int maxn, void *a, double *w, int *mout, int *touched) {
    return backend(ozaki ? 'O' : 'M', n, maxn, a, w, mout, touched);
}
'''

MAIN = r'''
int main(int argc, char **argv) {
    CHECK(argc == 5);
    const int cplx = std::atoi(argv[1]), retry_entry = std::atoi(argv[2]), expected_info = std::atoi(argv[4]);
    const int n = 4, maxn = 2;
    double a[32] = {12.0}, w[4] = {-1, -1, -1, -1};
    int mout = -1;
    int info = retry_entry ? openmx_magma_eigen_gpu(cplx, n, maxn, a, w, &mout) :
        (cplx ? openmx_magma_zheevdx_gpu(n, maxn, a, w, &mout) :
                openmx_magma_dsyevdx_gpu(n, maxn, a, w, &mout));
    CHECK(info == expected_info);
    CHECK(attempts == argv[3]);
    if (info == 0) { CHECK(mout == maxn); CHECK(a[0] == 42.0); CHECK(w[0] == 1.0 && w[1] == 2.0); }
    CHECK(w[2] == -1 && w[3] == -1);
    std::printf("dispatch %s cplx=%d retry_entry=%d attempts=%s info=%d\n",
                std::getenv("TEST_ARCH"), cplx, retry_entry, attempts.c_str(), info);
    return 0;
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cxx", default="g++")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    source = (root / "source/magma_openmx.cu").read_text()
    names = ["use_magma_ozaki2", "env_flag", "env_count", "eigen_retry_policy",
             "eigensolver_env_choice", "eigensolver_magma_min_n", "magma_entry_reroute",
             "openmx_gpu_eigensolver_use_hipsolver", "openmx_gpu_eigen_backend",
             "attempt_name", "attempt_retryable", "dense_eigen_locked",
             "openmx_magma_dsyevdx_gpu", "openmx_magma_zheevdx_gpu", "openmx_magma_eigen_gpu"]
    program = STUBS + "\n" + "\n\n".join(function(source, name) for name in names) + MAIN
    # Each process has fresh production static policy caches.
    cases = [
        ("mi300a auto", "gfx942:sramecc+:xnack+", 1, {}, "R", 0, 0),
        ("radeon auto", "gfx1200", 0, {}, "O", 0, 0),
        ("mi300x auto", "gfx942", 0, {}, "R", 0, 0),
        ("force magma mi300a", "gfx942", 1, {"OPENMX_GPU_EIGENSOLVER": "magma"}, "O", 0, 0),
        ("force roc radeon", "gfx1200", 0, {"OPENMX_GPU_EIGENSOLVER": "hipsolver"}, "R", 0, 0),
        ("force ozaki mi300a", "gfx942", 1, {"OPENMX_GPU_EIGENSOLVER": "magma", "OPENMX_MAGMA_OZAKI2": "1"}, "O", 0, 0),
        ("disable ozaki radeon", "gfx1200", 0, {"OPENMX_MAGMA_OZAKI2": "0"}, "M", 0, 0),
        ("disable ozaki mi300a", "gfx942", 1, {"OPENMX_GPU_EIGENSOLVER": "magma", "OPENMX_MAGMA_OZAKI2": "0"}, "M", 0, 0),
        ("small-size override", "gfx1200", 0, {"OPENMX_GPU_EIGENSOLVER_MAGMA_MIN_N": "5"}, "R", 0, 0),
        ("threshold equality", "gfx1200", 0, {"OPENMX_GPU_EIGENSOLVER_MAGMA_MIN_N": "4"}, "O", 0, 0),
        ("explicit over threshold", "gfx1200", 0, {"OPENMX_GPU_EIGENSOLVER": "magma", "OPENMX_GPU_EIGENSOLVER_MAGMA_MIN_N": "100"}, "O", 0, 0),
        ("roc failed mi300a", "gfx942", 1, {"TEST_FAIL_BACKEND": "R", "TEST_FAIL_INFO": "3"}, "RO", 0, 0),
        ("roc nonfinite mi300a", "gfx942", 1, {"TEST_FAIL_BACKEND": "R", "TEST_FAIL_INFO": "-199"}, "RO", 0, 0),
        ("ozaki failure radeon", "gfx1200", 0, {"TEST_FAIL_BACKEND": "O", "TEST_FAIL_INFO": "3"}, "OM", 0, 0),
        ("all fail", "gfx942", 1, {"TEST_FAIL_BACKEND": "ROM", "TEST_FAIL_INFO": "3"}, "ROM", 0, 3),
        ("retry disabled", "gfx942", 1, {"OPENMX_MAGMA_OZAKI2_RETRY": "0", "TEST_FAIL_BACKEND": "R", "TEST_FAIL_INFO": "3"}, "R", 0, 3),
        ("deliberate magma retry mi300a", "gfx942", 1, {}, "O", 1, 0),
        ("deliberate magma retry forced roc", "gfx1200", 0, {"OPENMX_GPU_EIGENSOLVER": "hipsolver"}, "O", 1, 0),
        ("discovery failure", "gfx942", 1, {"TEST_DISCOVERY_FAIL": "1"}, "O", 0, 0),
    ]
    with tempfile.TemporaryDirectory(prefix="openmx-eigen-dispatch-") as tmp:
        cpp, binary = Path(tmp) / "test.cpp", Path(tmp) / "test"
        cpp.write_text(program)
        subprocess.run([args.cxx, "-std=c++17", "-O2", "-pthread", "-I" + str(root / "source"),
                        str(cpp), "-o", str(binary)], check=True)
        base_env = {k: v for k, v in os.environ.items()
                    if not k.startswith(("OPENMX_", "MAGMA_OZAKI2_", "TEST_"))}
        for label, arch, integrated, overrides, expected, retry, info in cases:
            for cplx in (0, 1):
                env = dict(base_env, TEST_ARCH=arch, TEST_INTEGRATED=str(integrated),
                           OPENMX_GPU_EIGENSOLVER_PROFILE="1", **overrides)
                result = subprocess.run([str(binary), str(cplx), str(retry), expected, str(info)],
                                        check=True, text=True, capture_output=True, env=env)
                records = re.findall(r"GPUEIGENPROF.*backend=(\S+).*info=(-?\d+)", result.stderr)
                observed = "".join({"rocsolver": "R", "magma_ozaki2": "O", "magma": "M"}[r[0]]
                                   for r in records)
                if observed != expected:
                    raise AssertionError(f"{label}: diagnostic backend {observed!r} != {expected!r}")
                if len(expected) > 1 and "stage=backup_restore" not in result.stderr:
                    raise AssertionError(f"{label}: retry restore profiling is missing")
            print(f"PASS {label}: {expected}")
        for verbose in ("0", "1"):
            for cplx in (0, 1):
                env = dict(base_env, TEST_ARCH="gfx942", TEST_INTEGRATED="1", OPENMX_GPU_VERBOSE=verbose)
                result = subprocess.run([str(binary), str(cplx), "0", "R", "0"],
                                        check=True, text=True, capture_output=True, env=env)
                if "GPUEIGENDETAIL" in result.stderr:
                    raise AssertionError("Detailed profiling must require OPENMX_GPU_EIGENSOLVER_PROFILE")
        print("PASS detailed profiling remains opt-in even with GPU verbose logging")
    print(f"gpu_eigen_dispatch: {len(cases) * 2 + 4} checks passed")


if __name__ == "__main__":
    main()

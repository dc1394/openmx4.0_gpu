#!/usr/bin/env python3
"""Check the selection of the grid integrals' precision without a GPU.

The AMD port of the NVIDIA test of the same name.  This version has no
lossy FP32 stage (and so no state machine to step through): what is left is
the choice between the double-float and the FP64 kernels, made once per
process.  Extracted verbatim and built as host C++ against HIP stubs:
grid_precision_configure, openmx_grid_precision_df and
openmx_grid_precision_announce of source/set_hamiltonian_hip.cu, and
openmx_gpu_fp32_fp64_ratio of source/eigen_refine_gpu.cu (the gcnArchName
table).  The stubbed hipGetDeviceProperties reports TEST_GCN_ARCH ("fail"
makes the query fail).  Checked per case, each in a new process:
OPENMX_GRID_PRECISION as the NVIDIA version reads it (df, DF and 2 give
double-float, any other non-empty value FP64, with a note when it is not
fp64, FP64, double or 0), the default from the architecture (RDNA and the
consumer gfx9 parts double-float, the data-centre gfx9 parts and a failed
query FP64), OPENMX_GPU_FP32_FP64_RATIO, the decision staying fixed when
the environment changes later, and the banner printed once.

Run: python3 tests/run_grid_precision_state.py [--cxx CXX]
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
    match = re.search(r"^[^\n;]*\b" + re.escape(name) + r"\([^;]*?\)\n\{", source, re.M)
    if not match:
        raise RuntimeError(f"function not found: {name}")
    pos = match.end()
    depth = 1
    while depth:
        if source[pos] == "{":
            depth += 1
        elif source[pos] == "}":
            depth -= 1
        pos += 1
    return source[match.start():pos]


def line(source, pattern):
    match = re.search(pattern, source, re.M)
    if not match:
        raise RuntimeError(f"line not found: {pattern}")
    return match.group(0)


STUBS = r'''
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "openmx_stdout_level.h"
#include "openmx_precision.h"
#ifdef TEST_STDOUT_DEFINED
int level_stdout = 3;
#endif
typedef int hipError_t;
enum { hipSuccess = 0, hipErrorNoDevice = 100 };
struct hipDeviceProp_t { char gcnArchName[256]; };
static int queries;
static hipError_t hipGetDevice(int *device) { *device = 0; return hipSuccess; }
static hipError_t hipGetDeviceProperties(hipDeviceProp_t *prop, int device)
{
    const char *arch = std::getenv("TEST_GCN_ARCH");
    (void)device;
    queries++;
    if (arch == nullptr || std::strcmp(arch, "fail") == 0) return hipErrorNoDevice;
    std::snprintf(prop->gcnArchName, sizeof(prop->gcnArchName), "%s", arch);
    return hipSuccess;
}
static hipError_t hipGetLastError(void) { return hipSuccess; }
'''

HARNESS = r'''
int main(int argc, char **argv)
{
    if (argc != 2) return 2;
#ifdef TEST_STDOUT_DEFINED
    level_stdout = std::atoi(std::getenv("TEST_STDOUT_LEVEL"));
#endif
    const int expect_df = std::atoi(argv[1]);
    /* A disabled first input must not cache its veto or consume the banner. */
    openmx_set_fp32_enabled(0);
    if (openmx_grid_precision_df() != 0 || queries != 0) return 11;
    openmx_grid_precision_announce();
    openmx_set_fp32_enabled(1);
    if (openmx_grid_precision_df() != expect_df) {
        std::fprintf(stderr, "openmx_grid_precision_df() = %d, expected %d\n", openmx_grid_precision_df(), expect_df);
        return 1;
    }
    /* decided once: a later environment does not change it */
    setenv("OPENMX_GRID_PRECISION", expect_df ? "fp64" : "df", 1);
    setenv("OPENMX_GPU_FP32_FP64_RATIO", expect_df ? "1" : "64", 1);
    if (openmx_grid_precision_df() != expect_df || grid_precision_df() != (expect_df != 0)) {
        std::fprintf(stderr, "the decision changed\n");
        return 1;
    }
    if (1 < queries) {
        std::fprintf(stderr, "the device was queried %d times\n", queries);
        return 1;
    }
    /* Repeated inputs override the cached environment in both directions. */
    openmx_set_fp32_enabled(0);
    if (openmx_grid_precision_df() != 0 || grid_precision_df()) return 12;
    openmx_grid_precision_announce();
    openmx_set_fp32_enabled(1);
    if (openmx_grid_precision_df() != expect_df) return 13;
    openmx_grid_precision_announce();
    openmx_grid_precision_announce();
    return 0;
}
'''

DF_BANNER = ("<DFT>  grid integrals: double-float kernels (FP32 arithmetic with error-free transformations; "
             "OPENMX_GRID_PRECISION=fp64 restores the FP64 kernels)\n")

# (environment, expected double-float, expected stdout: "df" banner, "note" or "")
CASES = [
    ({"TEST_GCN_ARCH": "gfx1200"}, 1, "df"),
    ({"TEST_GCN_ARCH": "gfx1100"}, 1, "df"),
    ({"TEST_GCN_ARCH": "gfx900"}, 1, "df"),
    ({"TEST_GCN_ARCH": "gfx90a:sramecc+:xnack-"}, 0, ""),
    ({"TEST_GCN_ARCH": "gfx942"}, 0, ""),
    ({"TEST_GCN_ARCH": "gfx906"}, 0, ""),
    ({"TEST_GCN_ARCH": "fail"}, 0, ""),
    ({"TEST_GCN_ARCH": "gfx1200", "OPENMX_GPU_FP32_FP64_RATIO": "4"}, 0, ""),
    ({"TEST_GCN_ARCH": "gfx90a", "OPENMX_GPU_FP32_FP64_RATIO": "8"}, 1, "df"),
    ({"TEST_GCN_ARCH": "gfx90a", "OPENMX_GRID_PRECISION": "df"}, 1, "df"),
    ({"TEST_GCN_ARCH": "gfx90a", "OPENMX_GRID_PRECISION": "DF"}, 1, "df"),
    ({"TEST_GCN_ARCH": "fail", "OPENMX_GRID_PRECISION": "2"}, 1, "df"),
    ({"TEST_GCN_ARCH": "gfx1200", "OPENMX_GRID_PRECISION": "fp64"}, 0, ""),
    ({"TEST_GCN_ARCH": "gfx1200", "OPENMX_GRID_PRECISION": "FP64"}, 0, ""),
    ({"TEST_GCN_ARCH": "gfx1200", "OPENMX_GRID_PRECISION": "double"}, 0, ""),
    ({"TEST_GCN_ARCH": "gfx1200", "OPENMX_GRID_PRECISION": "0"}, 0, ""),
    ({"TEST_GCN_ARCH": "gfx1200", "OPENMX_GRID_PRECISION": "1"}, 0, "note"),
    ({"TEST_GCN_ARCH": "gfx1200", "OPENMX_GRID_PRECISION": "single"}, 0, "note"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cxx", default=os.environ.get("CXX", "c++"))
    args = parser.parse_args()

    sh = (ROOT / "source/set_hamiltonian_hip.cu").read_text()
    er = (ROOT / "source/eigen_refine_gpu.cu").read_text()
    code = "\n\n".join([
        STUBS,
        function(er, "openmx_gpu_fp32_fp64_ratio"),
        line(sh, r"^int grid_precision_mode = .*;$"),
        line(sh, r"^char grid_precision_unrecognized\[.*;$"),
        function(sh, "grid_precision_configure"),
        function(sh, "grid_precision_df"),
        function(sh, "openmx_grid_precision_df"),
        function(sh, "openmx_grid_precision_announce"),
        HARNESS,
    ])
    env = {k: v for k, v in os.environ.items()
           if k not in ("OPENMX_GRID_PRECISION", "OPENMX_GPU_FP32_FP64_RATIO", "TEST_GCN_ARCH")}
    with tempfile.TemporaryDirectory(prefix="openmx-grid-precision-") as tmp:
        cfile, binary = Path(tmp) / "check.cpp", Path(tmp) / "check"
        cfile.write_text(code)
        for defined in (False, True):
            flags = ["-DTEST_STDOUT_DEFINED"] if defined else []
            subprocess.run(shlex.split(args.cxx) + ["-std=c++17", "-O1", "-Wall", "-Wextra", "-Werror",
                            "-I", str(ROOT / "source"), *flags, str(cfile),
                            str(ROOT / "source/openmx_precision.c"), "-o", str(binary)], check=True)
            for level in ((0, 1, 2, 3) if defined else (3,)):
                for extra, expect_df, expect_out in CASES:
                    run = subprocess.run([str(binary), str(expect_df)],
                                         env=env | extra | {"TEST_STDOUT_LEVEL": str(level)},
                                         capture_output=True, text=True)
                    if run.returncode != 0:
                        raise SystemExit(f"FAIL level={level} {extra}: {run.stderr.strip()}")
                    if expect_out == "df":
                        ok = run.stdout == (DF_BANNER if level >= 2 else "")
                    elif expect_out == "note":
                        value = extra["OPENMX_GRID_PRECISION"]
                        ok = run.stdout == (f"<DFT>  grid integrals: OPENMX_GRID_PRECISION={value} is neither df nor fp64; "
                                            "FP64 kernels\n")
                    else:
                        ok = run.stdout == ""
                    if not ok:
                        raise SystemExit(f"FAIL level={level} {extra}: unexpected banner {run.stdout!r}")
    print(f"PASS: {len(CASES) * 5} grid precision/output configurations (levels 0..3, standalone weak "
          "reference, input warnings at every level, per-input FP32 veto/reset, one banner)")


if __name__ == "__main__":
    main()

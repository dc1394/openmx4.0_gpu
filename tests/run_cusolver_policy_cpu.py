#!/usr/bin/env python3
"""Exercise production native/emulated eigensolver policy without a GPU.

Extract the production handle-creation and refinement configuration functions,
and stub CUDA device attributes and cuSOLVER calls. Test fast-FP64/GeForce and
mixed-device use, explicit overrides, unsupported APIs and creation failures.
Actual numerical accuracy is covered by run_cusolver_smoke.sh on a GPU.
"""
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile


def function(source, name):
    match = re.search(r"^(?:static )?(?:int|cusolverStatus_t) " + re.escape(name)
                      + r"\([^;]*?\)\s*\{", source, re.MULTILINE)
    if match is None:
        raise ValueError(f"Cannot find production function {name}")
    start = source.index("{", match.start())
    depth, end = 1, start + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[match.start():end]


PREFIX = r'''
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#define CHECK(x) do { if (!(x)) { fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#x); exit(2); } } while (0)
typedef int cusolverStatus_t;
typedef void *cusolverDnHandle_t;
enum { cudaSuccess = 0, CUSOLVER_STATUS_SUCCESS = 0,
       CUSOLVER_STATUS_NOT_SUPPORTED = 1, CUSOLVER_STATUS_INVALID_VALUE = 2,
       CUSOLVER_DEFAULT_MATH = 3, CUSOLVER_FP64_EMULATED_FIXEDPOINT_MATH = 4,
       CUDA_EMULATION_STRATEGY_PERFORMANT = 5,
       CUDA_EMULATION_MANTISSA_CONTROL_DYNAMIC = 6,
       cudaDevAttrSingleToDoublePrecisionPerfRatio = 7 };
static int ratio = 2, query_fail, query_calls, create_fail, setting_fail,
           setting_call, destroyed, mode, gemmul8 = 1, checks;
static int cudaGetDevice(int *device) { *device = 2; query_calls++; return query_fail; }
static int cudaDeviceGetAttribute(int *result, int attr, int device) {
    CHECK(attr == cudaDevAttrSingleToDoublePrecisionPerfRatio && device == 2);
    *result = ratio; return query_fail;
}
static int cudaGetLastError(void) { return 0; }
static int cusolverDnCreate(cusolverDnHandle_t *handle) {
    if (create_fail) return create_fail;
    *handle = (void *)1; mode = CUSOLVER_DEFAULT_MATH; return 0;
}
static int cusolverDnDestroy(cusolverDnHandle_t handle) {
    CHECK(handle == (void *)1); destroyed++; return 0;
}
static int cusolverDnSetMathMode(cusolverDnHandle_t handle, int value) {
    CHECK(handle == (void *)1); setting_call++;
    if (value != CUSOLVER_DEFAULT_MATH && setting_fail) return setting_fail;
    mode = value; return 0;
}
static int cusolverDnSetEmulationStrategy(cusolverDnHandle_t handle, int value) {
    CHECK(handle == (void *)1 && value == CUDA_EMULATION_STRATEGY_PERFORMANT);
    setting_call++; return 0;
}
static int cusolverDnSetFixedPointEmulationMantissaControl(cusolverDnHandle_t handle, int value) {
    CHECK(handle == (void *)1 && value == CUDA_EMULATION_MANTISSA_CONTROL_DYNAMIC);
    setting_call++; return 0;
}
static int openmx_gemmul8DgemmEnabled(void) { return gemmul8; }
static int openmx_gemmul8ZgemmEnabled(void) { return gemmul8; }
'''

HARNESS = r'''
static void solver_case(int device_ratio, const char *override, int expected) {
    cusolverDnHandle_t handle = NULL;
    ratio = device_ratio; query_calls = setting_call = destroyed = 0;
    if (override) setenv("OPENMX_CUSOLVER_EMULATION", override, 1);
    else unsetenv("OPENMX_CUSOLVER_EMULATION");
    CHECK(openmx_cusolverDnCreate(&handle) == 0);
    CHECK(handle == (void *)1 && !destroyed);
#if CUSOLVER_VERSION >= 12200
    CHECK(mode == (expected ? CUSOLVER_FP64_EMULATED_FIXEDPOINT_MATH : CUSOLVER_DEFAULT_MATH));
    CHECK(setting_call == (expected ? 3 : 0));
    CHECK(query_calls == (override && *override ? 0 : 1));
#else
    CHECK(mode == CUSOLVER_DEFAULT_MATH && !setting_call && !query_calls);
#endif
    checks++;
}
int main(void) {
    for (int c = 0; c < 2; c++) {
        int out;
        /* A100 then GeForce then A100: detect the current device per handle. */
        solver_case(2, NULL, 0);
        solver_case(64, NULL, 1);
        solver_case(2, NULL, 0);
        solver_case(7, NULL, 0);
        solver_case(8, NULL, 1);
        solver_case(2, "1", 1);
        solver_case(64, "0", 0);
        solver_case(2, "", 0);
        solver_case(0, NULL, 1);
        query_fail = 1; solver_case(2, NULL, 1); query_fail = 0;
        unsetenv("OPENMX_EIGEN_REFINE");
        ratio = 2;
        CHECK(openmx_eigen_refine_configure(c, 0, &out) == 0 && out == 0);
        ratio = 64;
        CHECK(openmx_eigen_refine_configure(c, 0, &out) == 2 && out == 64);
        gemmul8 = 0;
        CHECK(openmx_eigen_refine_configure(c, 0, &out) == 0 && out == 0);
        gemmul8 = 1;
        setenv("OPENMX_EIGEN_REFINE", "0", 1);
        CHECK(openmx_eigen_refine_configure(c, 0, &out) == 0 && out == 0);
        ratio = 2; setenv("OPENMX_EIGEN_REFINE", "3", 1);
        CHECK(openmx_eigen_refine_configure(c, 0, &out) == 3 && out == 0);
        checks += 5;
    }
    unsetenv("OPENMX_CUSOLVER_EMULATION"); ratio = 64;
#if CUSOLVER_VERSION >= 12200
    for (int failure = 1; failure <= 3; failure++) {
        cusolverDnHandle_t handle = NULL;
        setting_fail = failure; destroyed = 0;
        int result = openmx_cusolverDnCreate(&handle);
        if (failure < 3) {
            CHECK(!result && handle == (void *)1 && !destroyed && mode == CUSOLVER_DEFAULT_MATH);
        } else CHECK(result == failure && handle == NULL && destroyed == 1);
        checks++;
    }
#endif
    { cusolverDnHandle_t handle = NULL;
      create_fail = 9; query_calls = setting_call = 0;
      CHECK(openmx_cusolverDnCreate(&handle) == 9 && !query_calls && !setting_call);
      checks++;
    }
    printf("PASS cuSOLVER %d native/emulation policy: %d checks\n", CUSOLVER_VERSION, checks);
    return 0;
}
'''


def main():
    root = Path(__file__).resolve().parents[1]
    solver = (root / "source/openmx_cusolver_compat.c").read_text()
    refine = (root / "source/eigen_refine_gpu.c").read_text()
    source = (PREFIX + function(solver, "openmx_solver_verbose")
              + "\n#if CUSOLVER_VERSION >= 12200\n"
              + function(solver, "openmx_solver_emulation_default") + "\n#endif\n"
              + function(solver, "openmx_cusolverDnCreate")
              + function(refine, "openmx_eigen_refine_configure") + HARNESS)
    with tempfile.TemporaryDirectory(prefix="openmx-cusolver-policy-") as tmp:
        src = Path(tmp) / "policy.c"
        src.write_text(source)
        for version in (12100, 12200, 12304):
            exe = Path(tmp) / f"policy-{version}"
            subprocess.run(shlex.split(os.environ.get("CC", "cc")) + [
                "-std=c11", "-O2", "-Wall", "-Wextra", "-Wno-unused-function",
                "-Wno-unused-parameter", f"-DCUSOLVER_VERSION={version}",
                str(src), "-o", str(exe)], check=True)
            subprocess.run([str(exe)], check=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Exercise production native/emulated eigensolver policy without a GPU.

Extract the production handle-creation and refinement configuration functions,
and stub CUDA device attributes and cuSOLVER calls. Test fast-FP64/GeForce and
mixed-device use, explicit overrides, unsupported APIs and creation failures.
The input FP32 veto also covers environment overrides and the four solvers'
cached settings across off/on/off input sequences in a single process.
Actual numerical accuracy is covered by run_cusolver_smoke.sh on a GPU.
"""
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile


def function(source, name):
    match = re.search(r"^(?:static )?(?:int|void|cusolverStatus_t) " + re.escape(name)
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
#include "openmx_precision.h"
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

CACHE_PREFIX = r'''
typedef struct { int iterations, defaulted; double until; } TestRefineState;
typedef TestRefineState ClusterColRefineState;
typedef TestRefineState ClusterNonColRefineState;
typedef TestRefineState BandColRefineState;
typedef TestRefineState BandNonColRefineState;
static TestRefineState ClusterCol_refine = {-1, 0, 0.0};
static TestRefineState ClusterNonCol_refine = {-1, 0, 0.0};
static TestRefineState BandCol_refine = {-1, 0, 0.0};
static TestRefineState BandNonCol_refine = {-1, 0, 0.0};
static int mpi_calls;
enum { mpi_comm_level1 = 0, Host_ID = 0, level_stdout = 0, MPI_INT = 0 };
static void MPI_Comm_rank(int comm, int *rank) { *rank = 0; mpi_calls++; }
static void MPI_Bcast(void *buffer, int count, int datatype, int root, int comm) { mpi_calls++; }
static double openmx_eigen_refine_until(void) { return 1e-5; }
'''

HARNESS = r'''
static void cache_case(TestRefineState *state, void (*configure)(void), int (*iterations)(void)) {
    /* Each solver may first be used by an FP64-only -runtest input. */
    openmx_set_fp32_enabled(0);
    setenv("OPENMX_EIGEN_REFINE", "3", 1);
    mpi_calls = 0;
    configure();
    CHECK(state->iterations == -1 && iterations() == 0 && mpi_calls == 0);
    openmx_set_fp32_enabled(1);
    configure();
    CHECK(state->iterations == 3 && iterations() == 3 && mpi_calls == 2);
    CHECK(state->defaulted == 0 && state->until == 1e-5);
    /* A later input must veto an already enabled static configuration. */
    openmx_set_fp32_enabled(0);
    configure();
    CHECK(state->iterations == 3 && iterations() == 0 && mpi_calls == 2);
    openmx_set_fp32_enabled(1);
    configure();
    CHECK(iterations() == 3 && mpi_calls == 2);
    checks += 5;
}

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
    CHECK(openmx_fp32_enabled());
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
        openmx_set_fp32_enabled(0);
        query_calls = 0;
        out = -1;
        CHECK(openmx_eigen_refine_configure(c, 0, &out) == 0 && out == 0 && !query_calls);
        unsetenv("OPENMX_EIGEN_REFINE"); ratio = 64; out = -1;
        CHECK(openmx_eigen_refine_configure(c, 0, &out) == 0 && out == 0 && !query_calls);
        /* FP64 emulation and GEMMul8 are independent of the FP32 veto. */
        CHECK(openmx_gemmul8DgemmEnabled() && openmx_gemmul8ZgemmEnabled());
        solver_case(64, NULL, 1);
        openmx_set_fp32_enabled(1);
        CHECK(openmx_eigen_refine_configure(c, 0, &out) == 2 && out == 64);
        checks += 4;
    }
    cache_case(&ClusterCol_refine, ClusterCol_RefineConfigure, ClusterCol_RefineIterations);
    cache_case(&ClusterNonCol_refine, ClusterNonCol_RefineConfigure, ClusterNonCol_RefineIterations);
    cache_case(&BandCol_refine, BandCol_RefineConfigure, BandCol_RefineIterations);
    cache_case(&BandNonCol_refine, BandNonCol_RefineConfigure, BandNonCol_RefineIterations);
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
    cached = ""
    for filename, prefix in (("Cluster_DFT_Col.c", "ClusterCol"),
                             ("Cluster_DFT_NonCol.c", "ClusterNonCol"),
                             ("Band_DFT_Col.c", "BandCol"),
                             ("Band_DFT_NonCol.c", "BandNonCol")):
        solver_source = (root / "source" / filename).read_text()
        cached += function(solver_source, prefix + "_RefineConfigure")
        cached += function(solver_source, prefix + "_RefineIterations")
    source = (PREFIX + function(solver, "openmx_solver_verbose")
              + "\n#if CUSOLVER_VERSION >= 12200\n"
              + function(solver, "openmx_solver_emulation_default") + "\n#endif\n"
              + function(solver, "openmx_cusolverDnCreate")
              + function(refine, "openmx_eigen_refine_configure")
              + CACHE_PREFIX + cached + HARNESS)
    with tempfile.TemporaryDirectory(prefix="openmx-cusolver-policy-") as tmp:
        src = Path(tmp) / "policy.c"
        src.write_text(source)
        for version in (12100, 12200, 12304):
            exe = Path(tmp) / f"policy-{version}"
            subprocess.run(shlex.split(os.environ.get("CC", "cc")) + [
                "-std=c11", "-O2", "-Wall", "-Wextra", "-Wno-unused-function",
                "-Wno-unused-parameter", f"-DCUSOLVER_VERSION={version}",
                "-I" + str(root / "source"), str(src),
                str(root / "source/openmx_precision.c"), "-o", str(exe)], check=True)
            subprocess.run([str(exe)], check=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Exercise the production grid precision state machine without a GPU.

Only CUDA device discovery is stubbed.  Each case uses a new process so the
production one-time environment configuration is tested too.
"""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
CUDA_STUB = r"""
#include <stdlib.h>
enum { cudaSuccess = 0, cudaDevAttrSingleToDoublePrecisionPerfRatio = 1 };
static inline int cudaGetDevice(int *device) { *device = 0; return 0; }
static inline int cudaDeviceGetAttribute(int *ratio, int attribute, int device)
{
    const char *value = getenv("TEST_GPU_RATIO");
    (void)attribute; (void)device;
    *ratio = value ? atoi(value) : 64;
    return *ratio < 0;
}
"""
HARNESS = r"""
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include "grid_precision.h"

int main(int argc, char **argv)
{
    int active, kernel, reset;
    assert(argc == 4);
    active = atoi(argv[1]); kernel = atoi(argv[2]); reset = atoi(argv[3]);
    assert(Grid_Precision_Kernel() == kernel);
    assert(Grid_Precision_StopCheck(1, 0, 0) == 1);
    Grid_Precision_BeginStep(1, 4, 1.0, 0, 0);
    assert(Grid_Precision_Fp32() == active);
    assert(Grid_Precision_StopCheck(0, 0, 0) == 0);
    assert(Grid_Precision_StopCheck(1, 0, 0) == !active);
    Grid_Precision_BeginStep(2, 4, 1.0, 0, 0);
    assert(Grid_Precision_Kernel() == kernel);
    assert(Grid_Precision_TakeHistoryReset() == (active && reset));
    assert(Grid_Precision_TakeHistoryReset() == 0);
    Grid_Precision_BeginStep(3, 4, 1.0, 0, 0);
    assert(!Grid_Precision_Fp32());

    Grid_Precision_EndCycle();
    Grid_Precision_BeginStep(1, 1, 1.0, 0, 0);
    assert(Grid_Precision_Kernel() == kernel);
    assert(Grid_Precision_TakeHistoryReset() == 0);
    Grid_Precision_EndCycle();
    Grid_Precision_BeginStep(1, 2, 1.0, 0, 0);
    assert(Grid_Precision_Fp32() == active);
    Grid_Precision_BeginStep(2, 2, 1.0, 0, 0);
    assert(Grid_Precision_Kernel() == kernel);
    assert(Grid_Precision_TakeHistoryReset() == (active && reset));

    Grid_Precision_EndCycle();
    Grid_Precision_BeginStep(1, 5, 1.0, 0, 0);
    Grid_Precision_BeginStep(2, 5, 1.0e-4, 0, 0);
    assert(Grid_Precision_Fp32() == active);
    Grid_Precision_BeginStep(3, 5, 1.0e-7, 0, 0);
    assert(Grid_Precision_Kernel() == kernel);
    assert(Grid_Precision_TakeHistoryReset() == (active && reset));
    Grid_Precision_BeginStep(4, 5, 1.0, 0, 0);
    assert(!Grid_Precision_Fp32());

    for (int i = 0; i < 2; ++i) {
        Grid_Precision_EndCycle();
        Grid_Precision_BeginStep(1, 5, 1.0, 0, 0);
        Grid_Precision_BeginStep(2, 5, i ? INFINITY : NAN, 0, 0);
        assert(Grid_Precision_Kernel() == kernel);
        assert(Grid_Precision_TakeHistoryReset() == (active && reset));
    }
    Grid_Precision_EndCycle();
    assert(Grid_Precision_Kernel() == kernel);
    return 0;
}
"""


def main():
    cases = [
        ({}, 0, 2, 1),
        ({"TEST_GPU_RATIO": "2"}, 0, 0, 1),
        ({"TEST_GPU_RATIO": "-1"}, 0, 0, 1),
        ({"OPENMX_GRID_PRECISION": "fp64"}, 0, 0, 1),
        ({"OPENMX_GRID_FP32": "1"}, 1, 2, 1),
        ({"OPENMX_GRID_FP32": "1", "OPENMX_GRID_PRECISION": "fp64"}, 1, 0, 1),
        ({"OPENMX_GRID_FP32": "1", "OPENMX_GRID_FP32_RESTART": "0"}, 1, 2, 0),
        ({"OPENMX_GRID_FP32": "1", "OPENMX_GRID_FP32_UNTIL": "nan"}, 1, 2, 1),
        ({"OPENMX_GRID_FP32": "1", "OPENMX_GRID_FP32_UNTIL": "inf"}, 1, 2, 1),
    ]
    env = {k: v for k, v in os.environ.items() if not k.startswith("OPENMX_GRID_")}
    env.pop("TEST_GPU_RATIO", None)
    with tempfile.TemporaryDirectory(prefix="openmx-grid-state-") as tmp:
        temp = Path(tmp)
        (temp / "cuda_runtime.h").write_text(CUDA_STUB)
        (temp / "main.c").write_text(HARNESS)
        binary = temp / "grid-state"
        subprocess.run(shlex.split(os.environ.get("CC", "cc")) +
                       ["-std=c99", "-O2", "-Wall", "-Wextra", "-Werror",
                        "-I" + tmp, "-I" + str(ROOT / "source"),
                        str(ROOT / "source/grid_precision.c"), str(temp / "main.c"),
                        "-lm", "-o", str(binary)], check=True)
        for extra, active, kernel, reset in cases:
            subprocess.run([str(binary), str(active), str(kernel), str(reset)],
                           env=env | extra, check=True)
    print(f"PASS: {len(cases)} grid precision configurations and SCF state transitions")


if __name__ == "__main__":
    main()

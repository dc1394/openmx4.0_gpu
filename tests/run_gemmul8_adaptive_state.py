#!/usr/bin/env python3
"""CPU test of the production GEMMul8 adaptive controller.

Extract the self-contained controller from the CUDA bridge, stubbing only
the forward-product stage setter; no CUDA installation/device is needed.
"""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
PREFIX = r"""
#include <cassert>
#include <cmath>
#include <cstdio>
#include <limits>
constexpr unsigned kDefaultNumModuli = 15u;
constexpr unsigned kMaxNumModuli = 20u;
enum { kForwardFp64 = 1, kForwardGemmul8 = 2 };
static void openmx_gemmul8SetForwardStage(int, int, int, int, int) {}
"""
HARNESS = r"""
static void configure(double tolerance)
{
    const int moduli[] = {12, 15}, fast[] = {1, 0};
    const double promote[] = {1e-4, 1e-7}, tolerances[] = {tolerance, tolerance};
    openmx_gemmul8AdaptiveConfigure(2, 1, moduli, fast, promote, tolerances,
                                   0, 1, 2, 0, 0, 2, 1, 8, 5, 1e-8);
    openmx_gemmul8AdaptiveStart();
    assert(openmx_gemmul8AdaptiveBeginTrial() == 0);
}

int main()
{
    const double inf = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double invalid[] = {nan, inf, -inf};
    int rejected;
    double eta;
    char description[64];
    long long counters[5];
    for (double tolerance : {0.0, 1e-6}) {
        for (double bad : invalid) {
            configure(tolerance);
            openmx_gemmul8AdaptiveReport(-1.0, 0);
            openmx_gemmul8AdaptiveTrialStatus(&rejected, &eta);
            assert(!rejected && eta == -1.0);
            openmx_gemmul8AdaptiveReport(bad, 0);
            openmx_gemmul8AdaptiveReport(0.0, 0);
            openmx_gemmul8AdaptiveTrialStatus(&rejected, &eta);
            assert(rejected && eta == inf);
            openmx_gemmul8AdaptiveDescribe(description, sizeof(description), counters);
            assert(counters[3] == 2);
            openmx_gemmul8AdaptiveReject();
            assert(openmx_gemmul8AdaptiveBeginTrial() == 1);
            openmx_gemmul8AdaptiveTrialStatus(&rejected, &eta);
            assert(!rejected && eta == -1.0);
            openmx_gemmul8AdaptiveReport(bad, 0);
            openmx_gemmul8AdaptiveTrialStatus(&rejected, &eta);
            assert(rejected && eta == inf);
            openmx_gemmul8AdaptiveReject();
            assert(openmx_gemmul8AdaptiveBeginTrial() == 2);
            assert(openmx_gemmul8AdaptiveProbeColumns() == 0);
        }
        configure(tolerance);
        openmx_gemmul8AdaptiveReport(1e-7, 0);
        openmx_gemmul8AdaptiveTrialStatus(&rejected, &eta);
        assert(!rejected && eta == 1e-7);
        openmx_gemmul8AdaptiveReport(1e-5, 0);
        openmx_gemmul8AdaptiveTrialStatus(&rejected, &eta);
        assert(rejected == (tolerance > 0.0) && eta == 1e-5);
        openmx_gemmul8AdaptiveReport(-1.0, 1);
        openmx_gemmul8AdaptiveTrialStatus(&rejected, &eta);
        assert(rejected && eta == 1e-5);
    }
    configure(1e-6);
    assert(!openmx_gemmul8AdaptiveStopCheck(1));
    assert(openmx_gemmul8AdaptiveTakeHistoryReset());
    assert(!openmx_gemmul8AdaptiveTakeHistoryReset());
    assert(openmx_gemmul8AdaptiveBeginTrial() == 2);
    assert(!openmx_gemmul8AdaptiveStopCheck(1));
    assert(!openmx_gemmul8AdaptiveStopCheck(1));
    assert(openmx_gemmul8AdaptiveStopCheck(1));
    openmx_gemmul8AdaptiveStart();
    assert(openmx_gemmul8AdaptiveBeginTrial() == 0);
}
"""


def main():
    source = (ROOT / "source/gemmul8_bridge.cu").read_text()
    begin = source.index("constexpr int kMaxAdaptiveStages = 8;")
    controller = "namespace {\n" + source[begin:]
    with tempfile.TemporaryDirectory(prefix="openmx-adaptive-state-") as tmp:
        temp = Path(tmp)
        cpp = temp / "state.cpp"
        cpp.write_text(PREFIX + "#include <initializer_list>\n" + controller + HARNESS)
        binary = temp / "state"
        subprocess.run(shlex.split(os.environ.get("CXX", "c++")) +
                       ["-std=c++14", "-O2", "-Wall", "-Wextra", "-Werror",
                        str(cpp), "-o", str(binary)], check=True)
        subprocess.run([str(binary)], check=True)
    print("PASS: adaptive error rejection, finite sentinel/tolerance, escalation, and final-stage reset")


if __name__ == "__main__":
    main()

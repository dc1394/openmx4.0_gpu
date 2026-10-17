#!/usr/bin/env python3
"""Check real input/bridge policy code against HIP stubs, without a GPU.

Protects the MI300A/Radeon split, explicit input and environment overrides,
the absence of HIP calls while parsing CPU inputs, and repeated -runtest
input resets.  Failed HIP queries must not poison the later device choice.
Concurrent first calls exercise the shared architecture cache and retry;
the HIP tracking stubs use atomic counters as well.
Run: python3 tests/run_gemmul8_arch_policy.py [--cxx CXX]
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
    pos, depth = match.end(), 1
    while depth:
        depth += (source[pos] == "{") - (source[pos] == "}")
        pos += 1
    return source[match.start():pos]


STUBS = r'''
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <thread>
#include <vector>
#define OPENMX_AMDGPU_HIP_RUNTIME_COMPAT_H
typedef int hipError_t;
enum { hipSuccess = 0, hipErrorNoDevice = 100 };
struct hipDeviceProp_t { char gcnArchName[256]; int integrated; };
static std::atomic<int> hip_calls{0}, property_queries{0}, fail_query{0};
static std::atomic<int> query_gate_total{0}, query_gate_arrivals{0};
static int input_setting = -1;
static hipError_t hipGetDevice(int *device) { hip_calls++; *device = 0; return hipSuccess; }
static hipError_t hipGetDeviceProperties(hipDeviceProp_t *prop, int)
{
    hip_calls++; property_queries++;
    if (query_gate_total.load()) {
        query_gate_arrivals++;
        while (query_gate_arrivals.load() < query_gate_total.load()) std::this_thread::yield();
    }
    if (fail_query) return hipErrorNoDevice;
    std::snprintf(prop->gcnArchName, sizeof(prop->gcnArchName), "%s", std::getenv("TEST_ARCH"));
    prop->integrated = std::atoi(std::getenv("TEST_INTEGRATED"));
    return hipSuccess;
}
static hipError_t hipGetLastError(void) { hip_calls++; return hipSuccess; }
#include "openmx_gpu_arch_policy.h"
static int myid, Host_ID, level_stdout = 1;
static int input_logical(const char *key, int *value, int dflt)
{
    if (std::strcmp(key, "scf.gemmul8.enable")) std::abort();
    *value = input_setting < 0 ? dflt : input_setting;
    return input_setting < 0 ? 0 : 1;
}
'''

HARNESS = r'''
static void expect(bool ok, const char *message)
{
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", message); std::exit(1); }
}
static void parse(int setting)
{
    const int before = hip_calls;
    input_setting = setting;
    parse_input();
    expect(hip_calls == before, "input parsing initialized HIP");
}
static void check(int d, int z)
{
    expect(openmx_gemmul8DgemmEnabled() == d, "DGEMM policy");
    expect(openmx_gemmul8ZgemmEnabled() == z, "ZGEMM policy");
}
static void concurrent_check(int expected, bool bridge, int repeats)
{
    std::vector<std::thread> workers;
    for (int i=0; i<16; ++i) {
        workers.emplace_back([=]() {
            for (int j=0; j<repeats; ++j) {
                if (bridge) check(expected, expected);
                else expect(openmx_gpu_device_is_mi300a() == expected, "concurrent device choice");
            }
        });
    }
    for (auto &worker : workers) worker.join();
}
static void concurrent_first_calls(int auto_enabled)
{
    parse(-1);
    setenv("OPENMX_GPU", "0", 1);
    concurrent_check(0, false, 32);
    expect(hip_calls == 0, "concurrent CPU-only calls initialized HIP");
    unsetenv("OPENMX_GPU");

    /* Hold every initial query until all 16 workers entered it, making
       simultaneous cache misses deterministic even on a busy machine. */
    query_gate_total = 16;
    fail_query = 1;
    concurrent_check(0, false, 1);
    expect(property_queries == 16, "failed concurrent queries were cached");

    query_gate_arrivals = 0;
    fail_query = 0;
    concurrent_check(!auto_enabled, false, 1);
    expect(property_queries == 32, "successful concurrent first-query count");
    query_gate_total = 0;
    const int calls = hip_calls;
    concurrent_check(auto_enabled, true, 128);
    expect(hip_calls == calls, "concurrent successful cache was not reused");

    setenv("OPENMX_GPU", "0", 1);
    concurrent_check(0, false, 32);
    expect(hip_calls == calls, "CPU-only override ignored after concurrent cache fill");
    std::puts("PASS concurrent first calls, retry and cached bridge policy");
}
int main(int argc, char **argv)
{
    if (argc != 2 && argc != 3) return 2;
    const int auto_enabled = std::atoi(argv[1]);
    if (argc == 3) { concurrent_first_calls(auto_enabled); return 0; }
    /* Parsing explicit or omitted inputs never queries a device. */
    parse(1); check(1, 1);
    parse(0); check(0, 0);
    expect(hip_calls == 0, "explicit on/off queried HIP");
    parse(-1);
    setenv("OPENMX_GPU", "0", 1);
    check(1, 1); /* unused CPU policy, preserving the old non-MI300A default */
    expect(hip_calls == 0, "OPENMX_GPU=0 queried HIP");
    unsetenv("OPENMX_GPU");

    /* A temporary failure cannot permanently select the wrong architecture. */
    fail_query = 1;
    check(1, 1);
    fail_query = 0;
    check(auto_enabled, auto_enabled);
    const int queries = property_queries;
    check(auto_enabled, auto_enabled);
    expect(property_queries == queries, "successful property query not cached");

    /* Successive inputs of -runtest must each recover the automatic policy. */
    parse(1); check(1, 1);
    parse(-1); check(auto_enabled, auto_enabled);
    parse(0); check(0, 0);
    parse(-1); check(auto_enabled, auto_enabled);

    /* Preserve global / alias / per-type priority in both directions. */
    setenv("GEMMUL8_DISABLE", "1", 1); check(0, 0);
    setenv("OPENMX_GEMMUL8_DISABLE", "0", 1); check(1, 1);
    setenv("GEMMUL8_DISABLE_D", "1", 1); check(0, 1);
    setenv("OPENMX_GEMMUL8_DISABLE_D", "0", 1); check(1, 1);
    setenv("OPENMX_GEMMUL8_DISABLE_Z", "1", 1); check(1, 0);
    parse(1); check(1, 0); /* explicit on still respects environment disable */
    parse(0); check(0, 0); /* explicit off wins over environment enable */
    for (const char *name : {"GEMMUL8_DISABLE", "OPENMX_GEMMUL8_DISABLE", "GEMMUL8_DISABLE_D",
                             "OPENMX_GEMMUL8_DISABLE_D", "OPENMX_GEMMUL8_DISABLE_Z"}) unsetenv(name);
    parse(-1); check(auto_enabled, auto_enabled);
    /* Empty overrides retain the architecture default. */
    setenv("OPENMX_GEMMUL8_DISABLE", "", 1); check(auto_enabled, auto_enabled);
    expect(property_queries == queries, "input reset repeated a successful HIP query");
    std::puts("PASS");
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", default="c++")
    parser.add_argument("--sanitize-thread", action="store_true",
                        help="compile the host harness with ThreadSanitizer")
    args = parser.parse_args()
    bridge = (ROOT / "source/gemmul8_bridge.cu").read_text()
    input_source = (ROOT / "source/Input_std.c").read_text()
    input_block = re.search(r"  \{\n    int gemmul8_enable;.*?\n  \}", input_source, re.S)
    if input_block is None:
        raise RuntimeError("GEMMul8 input block not found")
    functions = "\n".join(function(bridge, name) for name in (
        "env_bool", "gemmul8_disabled", "openmx_gemmul8SetEnabled",
        "openmx_gemmul8DgemmEnabled", "openmx_gemmul8ZgemmEnabled"))
    state = re.search(r"^int g_input_enabled = [^;]+;", bridge, re.M).group(0)
    source = ("#include <initializer_list>\n" + STUBS + "\n" + state + "\n" + functions +
              "\nstatic void parse_input(void)\n{\n" + input_block.group(0) + "\n}\n" + HARNESS)
    cases = [("gfx942", 1, 0), ("gfx942:sramecc+:xnack+", 1, 0),
             ("gfx942", 0, 1), ("gfx1200", 0, 1), ("gfx1200", 1, 1),
             ("gfx90a", 0, 1), ("gfx90c", 1, 1), ("gfx9420", 1, 1)]
    with tempfile.TemporaryDirectory(prefix="openmx_gemmul8_policy_") as tmp:
        src, exe = Path(tmp) / "policy.cpp", Path(tmp) / "policy"
        src.write_text(source)
        sanitizer = ["-g", "-fsanitize=thread", "-fno-omit-frame-pointer"] if args.sanitize_thread else []
        subprocess.run(shlex.split(args.cxx) + ["-std=c++11", "-Wall", "-Wextra", "-Werror", "-pthread", *sanitizer,
                       "-I" + str(ROOT / "source"), str(src), "-o", str(exe)], check=True)
        env = {key: value for key, value in os.environ.items()
               if not key.startswith(("OPENMX_", "GEMMUL8_", "TEST_"))}
        for arch, integrated, expected in cases:
            run_env = dict(env, TEST_ARCH=arch, TEST_INTEGRATED=str(integrated))
            result = subprocess.run([str(exe), str(expected)], env=run_env,
                                    capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError(f"{arch} integrated={integrated}: {result.stdout}{result.stderr}")
            concurrent = subprocess.run([str(exe), str(expected), "concurrent"], env=run_env,
                                        capture_output=True, text=True)
            if concurrent.returncode:
                raise RuntimeError(f"concurrent {arch} integrated={integrated}: {concurrent.stdout}{concurrent.stderr}")
            print(f"PASS {arch} integrated={integrated}: default {'GEMMul8' if expected else 'native FP64'}")
    print("PASS input reset, deferred HIP initialization, failed-query recovery and all override priorities")
    print("PASS 16-thread simultaneous first queries, failed-query recovery, cached calls and CPU-only bypass for all 8 devices")


if __name__ == "__main__":
    main()

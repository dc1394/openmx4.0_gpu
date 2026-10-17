#!/usr/bin/env python3
"""Run the Force4B resident/stream/turn harnesses on the GPU (one process).

Builds tests/run_force4b_resident_cpu.py and run_force4b_case1_turns_cpu.py
with clang -fopenmp --offload-arch (default gfx1200), replacing the host
stand-ins for device memory by the HIP runtime: Force_gpu_arena_try,
ForceHipMalloc and Force_gpu_arena_wait call hipMalloc, the copies call
hipMemcpy, hipFree/hipHostMalloc/hipHostFree are the real ones (counted).
The shared OpenMP target kernels of source/Force.c then execute on the device
with the production packing, so this checks the forces against the CPU fused
traces and that the resident, streamed and rank-turn paths agree bit for bit
on the GPU.  The admission inputs (free memory, MPI) stay single-process
stubs.  Needs a few MiB of device memory and a few seconds.

    python3 tests/run_force4b_gpu_smoke.py [--arch gfx1200] [--rocm /opt/rocm-7.2.0]
"""

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile


sys.dont_write_bytecode = True  # sibling harness modules, no __pycache__
import run_force4b_case1_turns_cpu as turns
import run_force4b_resident_cpu as resident
import run_force4b_stream_cpu as common


HOST_STUBS = (
    '''static void* Force_gpu_arena_try(size_t n)
{
    void* ptr = allocation_fail ? NULL : malloc(n);
    if (ptr != NULL) device_live++;
    return ptr;
}
static int hipFree(void* ptr)
{
    if (ptr != NULL) device_live--;
    free(ptr);
    return hipSuccess;
}
static void ForceHipMemcpyToDevice(void* dst, const void* src, size_t n) { memcpy(dst, src, n); }
static void ForceHipMemcpyFromDevice(void* dst, const void* src, size_t n) { memcpy(dst, src, n); }
''',
    '''static int hipHostMalloc(void** ptr, size_t n, unsigned int flags)
{
    (void)flags;
    *ptr = (pinned_fail || host_allocation_fail) ? NULL : malloc(n);
    if (*ptr == NULL) return 2;
    pinned_live++;
    return hipSuccess;
}
static int hipHostFree(void* ptr)
{
    if (ptr != NULL) pinned_live--;
    free(ptr);
    return hipSuccess;
}
''')

HIP_STUBS = (
    '''extern int hipMalloc(void** ptr, size_t size);
extern int hipFree(void* ptr);
extern int hipMemcpy(void* dst, const void* src, size_t size, int kind);
extern int hipHostMalloc(void** ptr, size_t size, unsigned int flags);
extern int hipHostFree(void* ptr);
extern int hipDeviceSynchronize(void);
static void* Force_gpu_arena_try(size_t n)
{
    void* ptr = NULL;
    if (allocation_fail || hipMalloc(&ptr, n == 0 ? 1 : n) != hipSuccess) return NULL;
    device_live++;
    return ptr;
}
static int hip_free_counted(void* ptr)
{
    if (ptr != NULL) device_live--;
    return ptr != NULL ? hipFree(ptr) : hipSuccess;
}
static void ForceHipMemcpyToDevice(void* dst, const void* src, size_t n)
{
    CHECK(n == 0 || hipMemcpy(dst, src, n, 1) == hipSuccess);
}
static void ForceHipMemcpyFromDevice(void* dst, const void* src, size_t n)
{
    CHECK(n == 0 || hipMemcpy(dst, src, n, 2) == hipSuccess);
}
''',
    '''static int hip_host_malloc_counted(void** ptr, size_t n, unsigned int flags)
{
    *ptr = NULL;
    if (pinned_fail || host_allocation_fail || hipHostMalloc(ptr, n, flags) != hipSuccess) return 2;
    pinned_live++;
    return hipSuccess;
}
static int hip_host_free_counted(void* ptr)
{
    if (ptr != NULL) pinned_live--;
    return ptr != NULL ? hipHostFree(ptr) : hipSuccess;
}
#define hipFree hip_free_counted
#define hipHostMalloc hip_host_malloc_counted
#define hipHostFree hip_host_free_counted
''')

RESIDENT_HOST = '''static int hipDeviceSynchronize(void) { return hipSuccess; }
'''
RESIDENT_DEVICE_HOST = '''static void* ForceHipMalloc(size_t bytes)
{
    void* ptr = allocation_fail ? NULL : malloc(bytes == 0 ? 1 : bytes);
    if (ptr != NULL) device_live++;
    return ptr;
}
static void* Force_gpu_arena_wait(size_t bytes, const char* what)
{
    void* ptr = malloc(bytes == 0 ? 1 : bytes);
    (void)what;
    CHECK(ptr != NULL);
    last_wait_bytes = bytes;
    device_live++;
    return ptr;
}
'''
RESIDENT_DEVICE_HIP = '''static void* ForceHipMalloc(size_t bytes)
{
    void* ptr = NULL;
    if (allocation_fail || hipMalloc(&ptr, bytes == 0 ? 1 : bytes) != hipSuccess) return NULL;
    device_live++;
    return ptr;
}
static void* Force_gpu_arena_wait(size_t bytes, const char* what)
{
    void* ptr = NULL;
    (void)what;
    CHECK(hipMalloc(&ptr, bytes == 0 ? 1 : bytes) == hipSuccess);
    last_wait_bytes = bytes;
    device_live++;
    return ptr;
}
'''


def to_device(harness: str) -> str:
    """Swap the host stand-ins of device memory for the HIP runtime."""
    for host, hip in zip(HOST_STUBS, HIP_STUBS):
        if harness.count(host) != 1:
            raise ValueError("harness stubs changed; update run_force4b_gpu_smoke.py")
        harness = harness.replace(host, hip)
    if RESIDENT_DEVICE_HOST in harness:
        harness = harness.replace(RESIDENT_HOST, "").replace(RESIDENT_DEVICE_HOST, RESIDENT_DEVICE_HIP)
    return harness


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rocm", default=os.environ.get("ROCM_PATH", "/opt/rocm-7.2.0"))
    parser.add_argument("--arch", default="gfx1200")
    args = parser.parse_args()
    rocm = Path(args.rocm)
    clang = rocm / "llvm" / "bin" / "clang"
    source = (common.ROOT / "source" / "Force.c").read_text()
    status = 0
    try:
        builds = (("resident", resident.make_harness(source)), ("turns", turns.make_harness(source)))
        with tempfile.TemporaryDirectory(prefix="openmx-force4b-gpu-") as directory:
            for name, harness in builds:
                src = Path(directory) / f"{name}.c"
                exe = Path(directory) / name
                src.write_text(to_device(harness))
                subprocess.run(
                    [str(clang), "-O2", "-std=c11", "-D_POSIX_C_SOURCE=200809L", "-fopenmp",
                     f"--offload-arch={args.arch}",
                     "-Wno-unused-function", "-Wno-unused-variable", str(src),
                     f"-L{rocm / 'lib'}", f"-Wl,-rpath,{rocm / 'lib'}", "-lamdhip64", "-lm",
                     "-o", str(exe)],
                    check=True,
                )
                env = dict(os.environ, OMP_TARGET_OFFLOAD="MANDATORY")
                result = subprocess.run([str(exe)], text=True, capture_output=True, env=env)
                if result.returncode:
                    sys.stderr.write(result.stderr)
                    sys.stdout.write(result.stdout)
                    status = 1
                else:
                    sys.stdout.write(f"[GPU {args.arch}] {result.stdout}")
        return status
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"Force4B GPU smoke test failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

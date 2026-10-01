#!/usr/bin/env python3
"""CPU-only checks of Force4B cached-halo rank turns and allocation fallback.

Extracts current production planning, packing, kernel and accumulation functions.
MPI/CUDA are single-process CPU stubs, so this does not test real MPI progress,
CUDA execution, device concurrency, or Linux allocation under memory pressure.
The fixture/reference is shared with run_force4b_case1_cpu.py. CC/CFLAGS are
honored; generated sources and binaries live only in a TemporaryDirectory.
"""

import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

import run_force4b_case1_cpu as case1
import run_force4b_stream_cpu as common


TURN_STUBS = r'''
#define MPI_MIN 0
#define MPI_BYTE 3
#define MPI_UNDEFINED (-1)
static int cuda_properties_fail;
static size_t host_available = (size_t)128 * 1024 * 1024 * 1024;
static int turns_fail_countdown = -1;
struct cudaDeviceProp { struct { char bytes[16]; } uuid; };
static int cudaGetDevice(int* device) { *device = 0; return cudaSuccess; }
static int cudaGetDeviceProperties(struct cudaDeviceProp* prop, int device)
{
    memset(prop, 0, sizeof(*prop));
    return cuda_properties_fail ? 1 : cudaSuccess;
}
static int MPI_Comm_split(int comm, int color, int key, int* result)
{
    *result = color == MPI_UNDEFINED ? MPI_COMM_NULL : 1;
    return 0;
}
static int MPI_Allreduce(const void* src, void* dst, int count, int type, int op, int comm)
{
    memcpy(dst, src, (size_t)count * (type == MPI_UNSIGNED_LONG_LONG
        ? sizeof(unsigned long long) : sizeof(int)));
    return 0;
}
static int MPI_Allgather(const void* src, int count, int type,
    void* dst, int receive_count, int receive_type, int comm)
{
    CHECK(type == MPI_BYTE && receive_type == MPI_BYTE && count == receive_count);
    memcpy(dst, src, (size_t)count);
    return 0;
}
static size_t Force4B_GpuCase1HostAvailable(void) { return host_available; }
#undef calloc
static void* turns_calloc(size_t count, size_t width)
{
    if (turns_fail_countdown == 0) return NULL;
    if (turns_fail_countdown > 0) turns_fail_countdown--;
    return test_calloc(count, width);
}
#define calloc turns_calloc
'''


TURN_MAIN = r'''
static void check_declined(int full_batch, int stream_mode, int mode)
{
    CHECK(!Force4B_GpuCase1TurnsBegin(full_batch, stream_mode, mode, cdm));
    /* Even an inactive rank can own a communicator for its admitted peers. */
    Force4B_GpuCase1TurnsRun(cdm, ds);
    CHECK(F4B_turns1.device_comm == MPI_COMM_NULL);
    CHECK(F4B_turns1.halo_host == NULL && F4B_turns1.metadata == NULL);
    CHECK(device_live == 0);
}

int main(void)
{
    int fallback, spin, seed, mc, q, source, direction, atom, k, i, l, checks = 0;
    const size_t GiB = (size_t)1024 * 1024 * 1024;
    const size_t MiB = (size_t)1024 * 1024;
    double max_error = 0.0;
    unsetenv("OPENMX_FORCE4B_GPU");
    unsetenv("OPENMX_FORCE4B_CASE1_TURNS");
    unsetenv("OPENMX_FORCE4B_CASE1_TURN_MAX_RANKS");
    allocate_case1_fixture();
    CHECK(Force4B_GpuCase1TurnsConcurrency(16 * GiB, 10 * GiB, 8, 0) == 1);
    CHECK(Force4B_GpuCase1TurnsConcurrency(12 * GiB, GiB, 18, 0) == 11);
    CHECK(Force4B_GpuCase1TurnsConcurrency(12 * GiB, GiB, 18, 2) == 2);
    CHECK(Force4B_GpuCase1TurnsConcurrency(12 * GiB, GiB, 4, 0) == 4);
    CHECK(Force4B_GpuCase1TurnsConcurrency(320 * MiB + GiB, GiB, 18, 0) == 1);
    CHECK(Force4B_GpuCase1TurnsConcurrency(256 * MiB, GiB, 18, 0) == 1);
    CHECK(Force4B_GpuCase1TurnsConcurrency(0, GiB, 18, 0) == 1);
    CHECK(Force4B_GpuCase1TurnsConcurrency(SIZE_MAX, SIZE_MAX, 18, 0) == 1);
    CHECK(Force4B_GpuCase1TurnsConcurrency(SIZE_MAX, 1, INT_MAX, 0) >= 1);
    CHECK(Force4B_GpuCase1TurnsConcurrency(SIZE_MAX, 0, 18, 0) == 1);
    CHECK(Force4B_GpuCase1HostLimit(32 * GiB) == 0);
    CHECK(Force4B_GpuCase1HostLimit(64 * GiB) == 32 * GiB);
    CHECK(Force4B_GpuCase1HostLimit(92 * GiB) == 60 * GiB);
    CHECK(Force4B_GpuCase1HostLimit(128 * GiB) == 96 * GiB);
    CHECK(Force4B_GpuCase1HostLimit(SIZE_MAX) < SIZE_MAX);
    for (fallback = 0; fallback <= 1; fallback++) {
        for (spin = 0; spin < 3; spin++) {
            for (seed = 0; seed < 7; seed++) {
                double expected[LOCAL_SLOTS][3] = {{0}};
                SpinP_switch = spin == 2 ? 3 : spin;
                for (direction = 0; direction < 4; direction++)
                    for (atom = 0; atom < ATOMS; atom++)
                        for (k = 0; k < NEIGHBORS; k++)
                            for (i = 0; i < ORBITALS; i++)
                                for (l = 0; l < PROJECTORS; l++)
                                    true_ds[direction][atom][k][i][l] =
                                        0.3 * sin(seed + 0.17 * direction + 0.23 * atom + 0.29 * k + 0.31 * i + 0.37 * l);
                for (mc = 1; mc <= Matomnum; mc++) {
                    for (q = 1; q <= FNAN[M2G[mc]]; q++) {
                        stage_source(F_G2M[natn[M2G[mc]][q]]);
                        Force4B_case1_trace_fused(mc, M2G[mc], q, cdm, ds,
                            &expected[mc][0], &expected[mc][1], &expected[mc][2]);
                    }
                    for (i = 0; i < 3; i++) Gxyz[M2G[mc]][41 + i] = 0.37;
                }
                CHECK(Force4B_GpuCase1TurnsBegin(0, -1, seed % 2 ? 1 : -1, cdm));
                CHECK(F4B_turns1.enabled && device_live == 0);
                CHECK(F4B_turns1.host_bytes < F4B_turns1.plan.bytes);
                for (source = Matomnum + MatomnumF; source > Matomnum; source--) {
                    stage_source(source);
                    Force4B_GpuCase1TurnsArchive(ds, source);
                }
                /* The receive slot is no longer a usable halo. Both device
                   packing and the allocation-failure path must use the cache. */
                for (k = 0; k < NEIGHBORS; k++)
                    for (i = 0; i < ORBITALS; i++)
                        for (l = 0; l < PROJECTORS; l++) ds[0][Matomnum + 1][k][i][l] = NAN;
                allocation_fail = fallback;
                {
                    const char* caps[] = { "1", "2", "0", "-1", "bad", "999999999999999999999", "" };
                    setenv("OPENMX_FORCE4B_CASE1_TURN_MAX_RANKS", caps[seed], 1);
                }
                Force4B_GpuCase1TurnsRun(cdm, ds);
                allocation_fail = 0;
                for (mc = 1; mc <= Matomnum; mc++) {
                    for (i = 0; i < 3; i++) {
                        const double error = fabs(Gxyz[M2G[mc]][41 + i] - (0.37 + expected[mc][i]));
                        CHECK(isfinite(error) && error <= 1e-11);
                        if (max_error < error) max_error = error;
                    }
                    checks++;
                }
                CHECK(F4B_turns1.device_comm == MPI_COMM_NULL);
                CHECK(F4B_turns1.halo_host == NULL && F4B_turns1.metadata == NULL);
                CHECK(device_live == 0);
            }
        }
    }
    CHECK(Force4B_GpuCase1TurnsMode() == -1);
    unsetenv("OPENMX_FORCE4B_CASE1_TURN_MAX_RANKS");
    setenv("OPENMX_FORCE4B_CASE1_TURNS", "auto", 1);
    CHECK(Force4B_GpuCase1TurnsMode() == -1);
    setenv("OPENMX_FORCE4B_CASE1_TURNS", "0", 1);
    CHECK(Force4B_GpuCase1TurnsMode() == 0);
    setenv("OPENMX_FORCE4B_CASE1_TURNS", "1", 1);
    CHECK(Force4B_GpuCase1TurnsMode() == 1);
    unsetenv("OPENMX_FORCE4B_CASE1_TURNS");
    check_declined(1, -1, -1); /* Full resident case already admitted. */
    check_declined(0, -1, 0);
    check_declined(0, 1, 1); /* Explicit source streaming takes precedence. */
    setenv("OPENMX_FORCE4B_GPU", "0", 1);
    check_declined(0, -1, 1);
    unsetenv("OPENMX_FORCE4B_GPU");
    host_available = 32 * GiB;
    check_declined(0, -1, 1);
    host_available = 128 * GiB;
    low_memory = 1;
    check_declined(0, -1, 1);
    low_memory = 0;
    cuda_properties_fail = 1;
    check_declined(0, -1, 1);
    cuda_properties_fail = 0;
    /* UUID table plus all six persistent host allocations fail one at a
       time. Every partial allocation must be released before CPU fallback. */
    for (int fail_after = 0; fail_after < 7; fail_after++) {
        turns_fail_countdown = fail_after;
        check_declined(0, -1, 1);
        turns_fail_countdown = -1;
    }
    List_YOUSO[35] = INT_MAX;
    check_declined(0, -1, 1);
    List_YOUSO[35] = 1;
    MatomnumF = INT_MAX;
    check_declined(0, -1, 1);
    MatomnumF = 3;
    Matomnum = 0;
    check_declined(0, -1, 1);
    Matomnum = 3;
    for (atom = 0; atom < ATOMS; atom++) FNAN[atom] = 0;
    check_declined(0, -1, 1);
    Force4B_GpuCase1TurnsEnd();
    free_case1_fixture();
    printf("PASS: %d CPU case-1 rank-turn comparisons; max force error %.3g; "
        "cached-halo device/CPU paths, overwritten receive slot, allocation failure, "
        "empty rank, size bounds, host budget and modes PASS\n", checks, max_error);
    return 0;
}
'''


def make_harness(source: str) -> str:
    # Reuse fixture declarations/helpers, omitting the source-stream test main.
    parts = [case1.make_harness(source).split("\nint main(void)\n", 1)[0], TURN_STUBS]
    for name in ("Force4BCase1Turns", "Force4BCase1Device"):
        parts.append(common.extract_struct(source, name))
    parts.append("static Force4BCase1Turns F4B_turns1 = { .device_comm = MPI_COMM_NULL };\n")
    for name in (
        "Force4B_GpuCase1TurnsMode", "Force4B_GpuCase1TurnsEnd", "Force4B_GpuCase1HostLimit",
        "Force4B_GpuCase1TurnsPlan", "Force4B_GpuCase1TurnsAllocate", "Force4B_GpuCase1TurnsBegin",
        "Force4B_GpuCase1TurnsArchive", "Force4B_GpuCase1TurnsDevice",
        "Force4B_GpuCase1TurnsAccumulate", "Force4B_GpuCase1TurnsConcurrency",
        "Force4B_GpuCase1TurnsRun",
    ):
        parts.append(common.extract_function(source, name))
    parts.append(TURN_MAIN)
    return "\n".join(parts)


def main() -> int:
    try:
        harness = make_harness((common.ROOT / "source" / "Force.c").read_text())
        compiler = shlex.split(os.environ.get("CC", "cc"))
        if not compiler:
            raise ValueError("CC must name a C compiler")
        flags = shlex.split(os.environ.get("CFLAGS", "-O2"))
        with tempfile.TemporaryDirectory(prefix="openmx-force4b-case1-turns-cpu-") as directory:
            source = Path(directory) / "turns.c"
            binary = Path(directory) / "turns"
            source.write_text(harness)
            subprocess.run(
                [*compiler, *flags, "-std=c11", "-Wno-unknown-pragmas", str(source), "-lm", "-o", str(binary)],
                check=True,
            )
            result = subprocess.run([str(binary)], text=True, capture_output=True)
            if result.returncode:
                sys.stderr.write(result.stderr)
                sys.stdout.write(result.stdout)
                return 1
            sys.stdout.write(result.stdout)
        return 0
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"Force4B case-1 rank-turn CPU test failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

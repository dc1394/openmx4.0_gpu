#!/usr/bin/env python3
"""CPU-only checks of Force4B cached-halo rank turns and allocation fallback.

The AMD port of the NVIDIA test of the same name.  Extracts the production
planning, packing, kernel and accumulation functions of source/Force.c.
MPI/HIP are single-process CPU stubs (the GPU grouping uses
hipDeviceGetPCIBusId instead of the CUDA UUID), so this does not test real
MPI progress, GPU execution, device concurrency, or Linux allocation under
memory pressure.  The fixture/reference is shared with
run_force4b_case1_cpu.py.  The receive slot is overwritten with NaN before
each turn, so both the device packing and the allocation-failure CPU trace
must read the cached halo rows; an unreadable /proc/meminfo (0) must decline.

    python3 tests/run_force4b_case1_turns_cpu.py --cc /usr/bin/gcc [--sanitize]
"""

import subprocess
import sys


sys.dont_write_bytecode = True  # sibling harness modules, no __pycache__
import run_force4b_case1_cpu as case1
import run_force4b_stream_cpu as common


TURN_STUBS = r'''
#define MPI_MIN 0
#define MPI_BYTE 3
#define MPI_UNDEFINED (-1)
static int pci_fail;
static size_t host_available = (size_t)128 * 1024 * 1024 * 1024;
static int hipGetDevice(int* device) { *device = 0; return hipSuccess; }
static int hipDeviceGetPCIBusId(char* bus, int length, int device)
{
    (void)device;
    if (pci_fail) return 1;
    snprintf(bus, (size_t)length, "0000:03:00.0");
    return hipSuccess;
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
'''


TURN_MAIN = r'''
static void check_declined(int full_batch, int stream_mode, int mode)
{
    CHECK(!Force4B_GpuCase1TurnsBegin(full_batch, stream_mode, mode, cdm));
    /* Even an inactive rank can own a communicator for its admitted peers. */
    Force4B_GpuCase1TurnsRun(cdm, ds);
    CHECK(F4B_turns1.device_comm == MPI_COMM_NULL);
    CHECK(F4B_turns1.halo_host == NULL && F4B_turns1.metadata == NULL);
    CHECK(F4B_turns1.stage == NULL);
    CHECK(device_live == 0 && pinned_live == 0);
}

int main(void)
{
    const char *precision = getenv("TEST_FP32_ENABLE");
    openmx_set_fp32_enabled(precision == NULL || atoi(precision));
    int fallback, spin, seed, mc, q, source, direction, atom, k, i, l, checks = 0, identical = 0;
    const size_t GiB = (size_t)1024 * 1024 * 1024;
    const size_t MiB = (size_t)1024 * 1024;
    double max_error = 0.0;
    unsetenv("OPENMX_FORCE4B_GPU");
    unsetenv("OPENMX_FORCE4B_CASE1_TURNS");
    unsetenv("OPENMX_FORCE4B_CASE1_TURN_MAX_RANKS");
    allocate_case1_fixture();
    CHECK(Force4B_GpuCase1TurnsConcurrency(16 * GiB, 10 * GiB, 8, 0) == 1);
    CHECK(Force4B_GpuCase1TurnsConcurrency(12 * GiB, GiB, 18, 0) == 10);
    CHECK(Force4B_GpuCase1TurnsConcurrency(12 * GiB, GiB, 18, 2) == 2);
    CHECK(Force4B_GpuCase1TurnsConcurrency(12 * GiB, GiB, 4, 0) == 4);
    CHECK(Force4B_GpuCase1TurnsConcurrency(320 * MiB + GiB, GiB, 18, 0) == 1);
    CHECK(Force4B_GpuCase1TurnsConcurrency(256 * MiB, GiB, 18, 0) == 1);
    CHECK(Force4B_GpuCase1TurnsConcurrency(0, GiB, 18, 0) == 1);
    CHECK(Force4B_GpuCase1TurnsConcurrency(SIZE_MAX, SIZE_MAX, 18, 0) == 1);
    CHECK(Force4B_GpuCase1TurnsConcurrency(SIZE_MAX, 1, INT_MAX, 0) >= 1);
    CHECK(Force4B_GpuCase1TurnsConcurrency(SIZE_MAX, 0, 18, 0) == 1);
    /* RX 9060 XT: 16 ranks, 15 GiB free, 2 GiB arenas -> 7 per wave */
    CHECK(Force4B_GpuCase1TurnsConcurrency(15 * GiB, 2 * GiB, 16, 0) == 7);
    CHECK(Force4B_GpuCase1HostLimit(0) == 0);
    CHECK(Force4B_GpuCase1HostLimit(32 * GiB) == 0);
    CHECK(Force4B_GpuCase1HostLimit(64 * GiB) == 32 * GiB);
    CHECK(Force4B_GpuCase1HostLimit(92 * GiB) == 60 * GiB);
    CHECK(Force4B_GpuCase1HostLimit(128 * GiB) == 96 * GiB);
    CHECK(Force4B_GpuCase1HostLimit(246 * GiB) == 184 * GiB + 512 * MiB);
    CHECK(Force4B_GpuCase1HostLimit(SIZE_MAX) < SIZE_MAX);
    for (fallback = 0; fallback <= 1; fallback++) {
        for (spin = 0; spin < 3; spin++) {
            for (seed = 0; seed < 7; seed++) {
                double expected[LOCAL_SLOTS][3] = {{0}};
                SpinP_switch = spin == 2 ? 3 : spin;
                pinned_fail = (seed + fallback) % 2; /* pinned and pageable stage */
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
                /* CDM0 x pref is packed here, before the receive loop */
                CHECK(Force4B_GpuCase1TurnsBegin(0, -1, seed % 2 ? 1 : -1, cdm));
                CHECK(F4B_turns1.enabled && device_live == 0);
                CHECK(F4B_turns1.stage_pinned == !pinned_fail);
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
                if (!fallback) {
                    /* the streamed path runs the same kernel and accumulates
                       in the same mc,q order: identical bits */
                    double turns_f[LOCAL_SLOTS][3];
                    for (mc = 1; mc <= Matomnum; mc++)
                        for (i = 0; i < 3; i++) {
                            turns_f[mc][i] = Gxyz[M2G[mc]][41 + i];
                            Gxyz[M2G[mc]][41 + i] = 0.37;
                        }
                    CHECK(Force4B_GpuCase1StreamBegin(0, -1));
                    for (source = Matomnum + MatomnumF; source >= 1; source--) {
                        stage_source(source);
                        Force4B_GpuCase1StreamSource(source, cdm, ds);
                    }
                    Force4B_GpuCase1StreamAccumulate(cdm);
                    Force4B_GpuCase1StreamEnd();
                    for (mc = 1; mc <= Matomnum; mc++)
                        CHECK(memcmp(turns_f[mc], &Gxyz[M2G[mc]][41], sizeof(turns_f[mc])) == 0);
                    identical++;
                }
                CHECK(F4B_turns1.device_comm == MPI_COMM_NULL);
                CHECK(F4B_turns1.halo_host == NULL && F4B_turns1.metadata == NULL);
                CHECK(device_live == 0 && pinned_live == 0);
            }
        }
    }
    pinned_fail = 0;
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
    stage_default = 0; /* APU or OPENMX_FORCE_GPU=0 */
    check_declined(0, -1, 1);
    stage_default = 1;
    host_available = 32 * GiB;
    check_declined(0, -1, 1);
    host_available = 0; /* unreadable /proc/meminfo: no rank turns */
    check_declined(0, -1, 1);
    host_available = 128 * GiB;
    low_memory = 1;
    check_declined(0, -1, 1);
    low_memory = 0;
    pci_fail = 1;
    check_declined(0, -1, 1);
    pci_fail = 0;
    /* Device table plus all six persistent host allocations fail one at a
       time (the stage through calloc). Every partial allocation must be
       released before the CPU fallback. */
    pinned_fail = 1;
    for (int fail_after = 0; fail_after < 7; fail_after++) {
        alloc_fail_countdown = fail_after;
        check_declined(0, -1, 1);
        alloc_fail_countdown = -1;
    }
    pinned_fail = 0;
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
        "%d turn/stream results bit-identical; "
        "cached-halo device/CPU paths, overwritten receive slot, allocation failure, "
        "pinned/pageable stage, empty rank, size bounds, host budget (incl. unreadable "
        "meminfo), APU gate and modes PASS\n", checks, max_error, identical);
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
        parts.append(common.extract_function(source, name+"_Single" if name in ("Force4B_GpuCase1Kernel","Force4B_GpuCase2Kernel") else name).replace(name+"_Single",name))
    parts.append(TURN_MAIN)
    return "\n".join(parts)


def main() -> int:
    try:
        harness = make_harness((common.ROOT / "source" / "Force.c").read_text())
        return common.build_and_run(harness, "force4b-case1-turns-cpu")
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"Force4B case-1 rank-turn CPU test failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

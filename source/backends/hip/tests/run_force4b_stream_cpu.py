#!/usr/bin/env python3
"""Check Force4B case-2 streaming packing, arithmetic and fallback on the CPU.

The AMD port of the NVIDIA test of the same name.  Production functions are
extracted verbatim from source/Force.c.  Without -fopenmp the OpenMP target
pragmas of the shared kernels compile as host loops, while MPI, HIP and the
device allocations/copies (hipMalloc arenas, hipFree, hipMemcpy, pinned host
arenas, OpenMX_GpuMemGetInfo) use single-process CPU stubs.  The numerical
reference is the existing CPU fused contraction.  This is not a GPU
correctness, MPI concurrency or performance test.

    python3 tests/run_force4b_stream_cpu.py --cc /usr/bin/gcc
    python3 tests/run_force4b_stream_cpu.py --cc /usr/bin/gcc --sanitize

CC and CFLAGS are honoured as defaults.  All generated files live in a
TemporaryDirectory; all C fixtures are freed.
"""

import argparse
from pathlib import Path
import os
import re
import shlex
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def extract_function(source: str, name: str) -> str:
    """Extract a static definition, ignoring braces in comments and literals."""
    match = re.search(r"^static [^\n]*\b" + re.escape(name) + r"\([^;{}]*\)\s*\{", source, re.M)
    if match is None:
        raise ValueError(f"missing production function: {name}")
    start = match.start()
    opening = match.end()-1
    # This scanner also fails explicitly if a declaration replaces a definition.
    if ";" in source[match.end():opening]:
        raise ValueError(f"expected definition, found declaration: {name}")
    token = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|/\*.*?\*/|//[^\n]*|[{}]', re.S)
    depth = 0
    for item in token.finditer(source, opening):
        if item.group() == "{":
            depth += 1
        elif item.group() == "}":
            depth -= 1
            if depth == 0:
                return source[start:item.end()] + "\n"
    raise ValueError(f"unterminated production function: {name}")


def extract_struct(source: str, name: str) -> str:
    match = re.search(r"typedef struct\s*\{[^{}]*\}\s*" + re.escape(name) + r";", source)
    if match is None:
        raise ValueError(f"missing production structure: {name}")
    return match.group() + "\n"


STUBS = r'''
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
static int level_stdout=3;
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <limits.h>
#include "openmx_precision.h"

typedef float Type_DS_VNA;
typedef int MPI_Comm;
#define MPI_COMM_NULL 0
#define MPI_COMM_TYPE_SHARED 0
#define MPI_INFO_NULL 0
#define MPI_UNSIGNED_LONG_LONG 1
#define MPI_INT 2
#define MPI_MAX 0
#define MPI_SUM 0
#define GPUSOLVER 6
#define hipSuccess 0

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #condition); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

static int scf_eigen_lib_flag = GPUSOLVER, mpi_comm_level1;
static int Matomnum = 3, SpinP_switch, Solver, List_YOUSO[50];
static int *M2G, *FNAN, *Spe_Total_CNO, *WhatSpecies, *F_G2M, **natn, ***RMI1;
static double *Spe_Atom_Cut1, **Dis, **Gxyz;
static int allocation_fail, host_allocation_fail, pinned_fail, low_memory;
static int device_live, pinned_live, stage_default = 1;

static int MPI_Comm_split_type(int a, int b, int c, int d, int* e) { *e = 1; return 0; }
static int MPI_Comm_rank(int a, int* b) { *b = 0; return 0; }
static int MPI_Comm_size(int a, int* b) { *b = 1; return 0; }
static int MPI_Comm_free(int* a) { *a = 0; return 0; }
static int MPI_Barrier(int a) { return 0; }
static int MPI_Bcast(void* a, int b, int c, int d, int e) { return 0; }
static int MPI_Reduce(void* a, void* b, int n, int t, int op, int root, int comm)
{
    memcpy(b, a, n * (t == MPI_UNSIGNED_LONG_LONG ? sizeof(unsigned long long) : sizeof(int)));
    return 0;
}
static int OpenMX_GpuMemGetInfo(size_t* free_bytes, size_t* total_bytes)
{
    *free_bytes = low_memory ? 0 : 1024ULL * 1024 * 1024;
    *total_bytes = 1024ULL * 1024 * 1024;
    return hipSuccess;
}
static void Force_gpu_pool_flush(void) {}
/* the production code clears HIP's last error after a handled failure */
static int hipGetLastError(void) { return hipSuccess; }
static void* Force_gpu_arena_try(size_t n)
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
/* page-locked host arenas: failure injected by pinned_fail (calloc follows)
   or host_allocation_fail (both host allocators fail) */
static int hipHostMalloc(void** ptr, size_t n, unsigned int flags)
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
/* AMD gate: OPENMX_FORCE_GPU umbrella, 0 on an APU */
static int Force_gpu_stage_default(void) { return stage_default; }
static int Force_collective_env_flag(const char* name, int default_value, int comm)
{
    const char* value = getenv(name);
    return value != NULL && value[0] != '\0' ? atoi(value) != 0 : default_value;
}
static void Force4B_gpu_abort(const char* message)
{
    fprintf(stderr, "%s\n", message);
    exit(EXIT_FAILURE);
}
/* Both reference and streaming receive the same deterministic damping.
   This test checks the contraction/packing, not the damping implementation. */
static double dampingF(double cutoff, double distance)
{
    return 1.0 / (1.0 + cutoff + distance);
}
/* alloc_fail_countdown >= 0 fails the host allocation after that many
   successes (every calloc of the extracted functions is counted) */
static int alloc_fail_countdown = -1;
static void* test_calloc(size_t count, size_t size)
{
    if (host_allocation_fail || alloc_fail_countdown == 0) return NULL;
    if (alloc_fail_countdown > 0) alloc_fail_countdown--;
    return calloc(count, size);
}
#define calloc test_calloc
'''


FIXTURE = r'''
#define ATOMS 6
#define NEIGHBORS 6
#define ORBITALS 4
#define PROJECTORS 12
#define LOCAL_SLOTS 5

static double***** cdm;
static Type_DS_VNA***** ds;

static void allocate_fixture(void)
{
    int atom, mc, spin, direction, k, i, j;
    cdm = calloc(2, sizeof(*cdm));
    ds = calloc(4, sizeof(*ds));
    M2G = calloc(LOCAL_SLOTS, sizeof(int));
    FNAN = calloc(ATOMS, sizeof(int));
    WhatSpecies = calloc(ATOMS, sizeof(int));
    F_G2M = calloc(ATOMS, sizeof(int));
    Spe_Total_CNO = calloc(3, sizeof(int));
    Spe_Atom_Cut1 = calloc(3, sizeof(double));
    natn = calloc(ATOMS, sizeof(*natn));
    Dis = calloc(ATOMS, sizeof(*Dis));
    Gxyz = calloc(ATOMS, sizeof(*Gxyz));
    RMI1 = calloc(LOCAL_SLOTS, sizeof(*RMI1));
    for (i = 0; i < 3; i++) {
        Spe_Total_CNO[i] = i + 2;
        Spe_Atom_Cut1[i] = 2.0 + i;
    }
    for (atom = 0; atom < ATOMS; atom++) {
        FNAN[atom] = NEIGHBORS - 1;
        WhatSpecies[atom] = atom % 3;
        F_G2M[atom] = atom % 3 + 1;
        natn[atom] = calloc(NEIGHBORS, sizeof(int));
        Dis[atom] = calloc(NEIGHBORS, sizeof(double));
        Gxyz[atom] = calloc(44, sizeof(double));
        for (j = 0; j < NEIGHBORS; j++) {
            natn[atom][j] = (atom + j) % ATOMS;
            Dis[atom][j] = 0.1 + 0.2 * j;
        }
    }
    for (mc = 1; mc <= Matomnum; mc++) {
        M2G[mc] = mc;
        RMI1[mc] = calloc(NEIGHBORS, sizeof(*RMI1[mc]));
        for (i = 0; i < NEIGHBORS; i++) {
            RMI1[mc][i] = calloc(NEIGHBORS, sizeof(int));
            for (j = 0; j < NEIGHBORS; j++)
                RMI1[mc][i][j] = (i && j && i != j && (i + j) % 4 == 0) ? -1 : j;
        }
    }
    for (spin = 0; spin < 2; spin++) {
        cdm[spin] = calloc(LOCAL_SLOTS, sizeof(*cdm[spin]));
        for (mc = 0; mc < LOCAL_SLOTS; mc++) {
            cdm[spin][mc] = calloc(NEIGHBORS, sizeof(*cdm[spin][mc]));
            for (k = 0; k < NEIGHBORS; k++) {
                cdm[spin][mc][k] = calloc(ORBITALS, sizeof(*cdm[spin][mc][k]));
                for (i = 0; i < ORBITALS; i++) {
                    cdm[spin][mc][k][i] = calloc(ORBITALS, sizeof(double));
                    for (j = 0; j < ORBITALS; j++)
                        cdm[spin][mc][k][i][j] = sin(0.2 + spin + 0.3 * mc + 0.4 * k + 0.5 * i + 0.6 * j) * 0.2;
                }
            }
        }
    }
    for (direction = 0; direction < 4; direction++) {
        ds[direction] = calloc(LOCAL_SLOTS, sizeof(*ds[direction]));
        ds[direction][Matomnum + 1] = calloc(NEIGHBORS, sizeof(*ds[direction][Matomnum + 1]));
        for (k = 0; k < NEIGHBORS; k++) {
            ds[direction][Matomnum + 1][k] = calloc(ORBITALS, sizeof(*ds[direction][Matomnum + 1][k]));
            for (i = 0; i < ORBITALS; i++)
                ds[direction][Matomnum + 1][k][i] = calloc(PROJECTORS, sizeof(float));
        }
    }
    List_YOUSO[35] = 1;
    List_YOUSO[34] = 3;
}

static void free_fixture(void)
{
    int atom, mc, spin, direction, k, i;
    for (direction = 0; direction < 4; direction++) {
        for (k = 0; k < NEIGHBORS; k++) {
            for (i = 0; i < ORBITALS; i++) free(ds[direction][Matomnum + 1][k][i]);
            free(ds[direction][Matomnum + 1][k]);
        }
        free(ds[direction][Matomnum + 1]);
        free(ds[direction]);
    }
    free(ds);
    for (spin = 0; spin < 2; spin++) {
        for (mc = 0; mc < LOCAL_SLOTS; mc++) {
            for (k = 0; k < NEIGHBORS; k++) {
                for (i = 0; i < ORBITALS; i++) free(cdm[spin][mc][k][i]);
                free(cdm[spin][mc][k]);
            }
            free(cdm[spin][mc]);
        }
        free(cdm[spin]);
    }
    free(cdm);
    for (mc = 1; mc <= Matomnum; mc++) {
        for (i = 0; i < NEIGHBORS; i++) free(RMI1[mc][i]);
        free(RMI1[mc]);
    }
    for (atom = 0; atom < ATOMS; atom++) {
        free(natn[atom]); free(Dis[atom]); free(Gxyz[atom]);
    }
    free(RMI1); free(natn); free(Dis); free(Gxyz);
    free(M2G); free(FNAN); free(WhatSpecies); free(F_G2M);
    free(Spe_Total_CNO); free(Spe_Atom_Cut1);
}

/* AMD auto policy for the O(N) solvers (collective flag stubbed by getenv) */
static void check_ordern_policy(void)
{
    const int solvers[] = {2, 3, 4, 5, 8, 11};
    int s;
    unsetenv("OPENMX_FORCE4B_ORDERN_AUTO");
    for (s = 0; s < 6; s++) {
        Solver = solvers[s];
        CHECK(Force4B_GpuOrderNLegacy() == (Solver == 5 || Solver == 8 || Solver == 11));
    }
    setenv("OPENMX_FORCE4B_ORDERN_AUTO", "1", 1);
    for (s = 0; s < 6; s++) {
        Solver = solvers[s];
        CHECK(!Force4B_GpuOrderNLegacy());
    }
    unsetenv("OPENMX_FORCE4B_ORDERN_AUTO");
}

int main(void)
{
    const char *precision = getenv("TEST_FP32_ENABLE");
    openmx_set_fp32_enabled(precision == NULL || atoi(precision));
    int mc, direction, k, i, j, l, spin, solver, seed, checks = 0;
    const int solvers[] = {3, 5, 8, 11};
    double max_error = 0.0;

    unsetenv("OPENMX_FORCE4B_GPU");
    unsetenv("OPENMX_FORCE4B_CASE2_STREAM");
    allocate_fixture();
    check_ordern_policy();
    for (spin = 0; spin < 3; spin++) {
        for (solver = 0; solver < 4; solver++) {
            for (seed = 0; seed < 7; seed++) {
                SpinP_switch = spin == 2 ? 3 : spin;
                Solver = solvers[solver];
                pinned_fail = seed % 2; /* pinned and pageable host arenas */
                CHECK(Force4B_GpuCase2StreamBegin(0, -1));
                CHECK(F4B_stream.host_pinned == !pinned_fail);
                for (mc = 1; mc <= Matomnum; mc++) {
                    double expected[3] = {0.0, 0.0, 0.0};
                    for (direction = 0; direction < 4; direction++)
                        for (k = 0; k < NEIGHBORS; k++)
                            for (i = 0; i < ORBITALS; i++)
                                for (l = 0; l < PROJECTORS; l++)
                                    ds[direction][Matomnum + 1][k][i][l] =
                                        cos(seed + 0.17 * mc + 0.23 * direction + 0.11 * k + 0.25 * i + 0.51 * l) * 0.3;
                    for (i = 1; i < NEIGHBORS; i++) {
                        for (j = Solver == 3 ? i : 0; j < NEIGHBORS; j++) {
                            if (!j || i == j || RMI1[mc][i][j] < 0) continue;
                            Force4B_case2_trace_fused(mc, mc, i, j, cdm, ds,
                                &expected[0], &expected[1], &expected[2]);
                        }
                    }
                    for (i = 0; i < 3; i++) Gxyz[mc][41 + i] = 0.37;
                    CHECK(Force4B_GpuCase2StreamRun(mc, cdm, ds));
                    /* Results must wait for the host special-pair accumulation. */
                    for (i = 0; i < 3; i++) CHECK(Gxyz[mc][41 + i] == 0.37);
                    Force4B_GpuCase2StreamAccumulate(mc);
                    for (i = 0; i < 3; i++) {
                        const double error = fabs(Gxyz[mc][41 + i] - (0.37 + expected[i]));
                        CHECK(isfinite(error) && error <= 1e-11);
                        if (max_error < error) max_error = error;
                    }
                    checks++;
                }
                Force4B_GpuCase2StreamEnd();
                CHECK(device_live == 0 && pinned_live == 0);
            }
        }
    }
    pinned_fail = 0;

    CHECK(Force4B_GpuCase2StreamMode() == -1);
    setenv("OPENMX_FORCE4B_CASE2_STREAM", "auto", 1);
    CHECK(Force4B_GpuCase2StreamMode() == -1);
    setenv("OPENMX_FORCE4B_CASE2_STREAM", "0", 1);
    CHECK(Force4B_GpuCase2StreamMode() == 0);
    setenv("OPENMX_FORCE4B_CASE2_STREAM", "1", 1);
    CHECK(Force4B_GpuCase2StreamMode() == 1);
    unsetenv("OPENMX_FORCE4B_CASE2_STREAM");
    CHECK(!Force4B_GpuCase2StreamBegin(0, 0));
    CHECK(Force4B_GpuCase2StreamBegin(0, 1));
    Force4B_GpuCase2StreamEnd();
    setenv("OPENMX_FORCE4B_GPU", "0", 1);
    CHECK(!Force4B_GpuCase2StreamBegin(0, 1));
    unsetenv("OPENMX_FORCE4B_GPU");
    /* APU/umbrella default off: no stream unless OPENMX_FORCE4B_GPU=1 */
    stage_default = 0;
    CHECK(!Force4B_GpuCase2StreamBegin(0, 1));
    setenv("OPENMX_FORCE4B_GPU", "1", 1);
    CHECK(Force4B_GpuCase2StreamBegin(0, 1));
    Force4B_GpuCase2StreamEnd();
    unsetenv("OPENMX_FORCE4B_GPU");
    stage_default = 1;
    scf_eigen_lib_flag = 0;
    CHECK(!Force4B_GpuCase2StreamBegin(0, 1));
    scf_eigen_lib_flag = GPUSOLVER;

    allocation_fail = 1;
    CHECK(!Force4B_GpuCase2StreamBegin(0, -1));
    CHECK(!Force4B_GpuCase2StreamRun(1, cdm, ds));
    allocation_fail = 0;
    host_allocation_fail = 1;
    CHECK(!Force4B_GpuCase2StreamBegin(0, -1));
    CHECK(device_live == 0 && pinned_live == 0);
    host_allocation_fail = 0;
    low_memory = 1;
    CHECK(!Force4B_GpuCase2StreamBegin(0, -1));
    low_memory = 0;
    CHECK(!Force4B_GpuCase2StreamBegin(1, -1));
    CHECK(Force4B_GpuCase2StreamBegin(0, -1));
    F4B_stream.capacity = 1;
    CHECK(!Force4B_GpuCase2StreamRun(1, cdm, ds));
    Force4B_GpuCase2StreamEnd();

    /* Empty centre while the other centres still enable streaming. */
    FNAN[1] = 0;
    CHECK(Force4B_GpuCase2StreamBegin(0, -1));
    CHECK(Force4B_GpuCase2StreamRun(1, cdm, ds));
    CHECK(F4B_stream.pending_items == 0);
    Force4B_GpuCase2StreamAccumulate(1);
    Force4B_GpuCase2StreamEnd();
    for (i = 0; i < ATOMS; i++) FNAN[i] = 0;
    CHECK(!Force4B_GpuCase2StreamBegin(0, -1));
    Matomnum = 0;
    CHECK(!Force4B_GpuCase2StreamBegin(0, -1));
    Force4B_GpuCase2StreamEnd();
    Matomnum = 3;
    CHECK(device_live == 0 && pinned_live == 0);
    free_fixture();
    printf("PASS: %d CPU mixed-species/spin/solver cases; max force error %.3g; "
        "configuration, APU gate, O(N) policy, OOM, pinned/pageable arenas, low-memory, "
        "full-batch, undersized-arena, empty-centre/rank checks PASS\n",
        checks, max_error);
    return 0;
}
'''


def make_harness(source: str) -> str:
    parts = [STUBS, extract_struct(source, "Force4BGpuItem")]
    for name in ("Force_gpu_node_reserve", "Force_gpu_arena_off", "Force_host_arena_alloc", "Force_host_arena_free",
                 "Force4B_GpuOrderNLegacy", "Force4B_case2_trace_fused", "Force4B_GpuCase2Kernel"):
        parts.append(extract_function(source, name+"_Single" if name in ("Force4B_GpuCase1Kernel","Force4B_GpuCase2Kernel") else name).replace(name+"_Single",name))
    parts.extend(extract_struct(source, name) for name in ("Force4BCase2Plan", "Force4BCase2Stream"))
    parts.append("static Force4BCase2Stream F4B_stream = { 0 };\n")
    for name in (
        "Force4B_GpuCase2StreamMode", "Force4B_GpuCase2Plan", "Force4B_GpuCase2StreamEnd",
        "Force4B_GpuCase2StreamBegin", "Force4B_GpuCase2StreamRun", "Force4B_GpuCase2StreamAccumulate",
    ):
        parts.append(extract_function(source, name+"_Single" if name in ("Force4B_GpuCase1Kernel","Force4B_GpuCase2Kernel") else name).replace(name+"_Single",name))
    parts.append(FIXTURE)
    return "\n".join(parts)


def build_and_run(harness: str, name: str, argv=None) -> int:
    """Compile one harness with the requested compiler and run it."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cc", default=os.environ.get("CC", "cc"))
    parser.add_argument("--cflags", default=os.environ.get("CFLAGS", "-O2"))
    parser.add_argument("--sanitize", action="store_true",
                        help="add -O1 -g -fsanitize=address,undefined")
    args = parser.parse_args(argv)
    compiler = shlex.split(args.cc)
    if not compiler:
        raise ValueError("--cc must name a C compiler")
    flags = shlex.split(args.cflags)
    if args.sanitize:
        flags += ["-O1", "-g", "-fsanitize=address,undefined", "-fno-sanitize-recover=all"]
    with tempfile.TemporaryDirectory(prefix=f"openmx-{name}-") as directory:
        source = Path(directory) / f"{name}.c"
        binary = Path(directory) / name
        source.write_text(harness)
        subprocess.run(
            [*compiler, *flags, "-std=c11", "-Wall", "-Wno-unknown-pragmas", "-Wno-unused-function", "-Wno-unused-variable",
             "-I", str(ROOT / "source"), str(source),
             str(ROOT / "source/openmx_precision.c"), "-lm", "-o", str(binary)],
            check=True,
        )
        env = dict(os.environ)
        env.setdefault("ASAN_OPTIONS", "detect_leaks=1")
        for enabled in ("1", "0"):
            env["TEST_FP32_ENABLE"] = enabled
            result = subprocess.run([str(binary)], text=True, capture_output=True, env=env)
            if result.returncode:
                sys.stderr.write(result.stderr)
                sys.stdout.write(result.stdout)
                return 1
            sys.stdout.write(f"scf.fp32.enable={enabled}: {result.stdout}")
    return 0


def main() -> int:
    try:
        harness = make_harness((ROOT / "source" / "Force.c").read_text())
        return build_and_run(harness, "force4b-stream-cpu")
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"Force4B CPU test failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

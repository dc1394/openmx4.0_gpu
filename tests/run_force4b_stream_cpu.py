#!/usr/bin/env python3
"""Check Force4B streaming packing, arithmetic and fallback on the CPU.

Production functions are extracted from source/Force.c. OpenACC pragmas are
ignored, while MPI, CUDA and device allocations/copies use single-process CPU
stubs. The numerical reference is the existing CPU fused contraction. This is
not a GPU correctness, MPI concurrency or performance test.

Run from any directory with Python 3 and a C compiler, for example:
    python3 tests/run_force4b_stream_cpu.py
    CC=gcc CFLAGS='-O1 -g -fsanitize=address,undefined' python3 tests/run_force4b_stream_cpu.py

All generated files live in a TemporaryDirectory; all C fixtures are freed.
"""

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
    match = re.search(r"^static [^\n]*\b" + re.escape(name) + r"\(", source, re.M)
    if match is None:
        raise ValueError(f"missing production function: {name}")
    start = match.start()
    opening = source.index("{", match.end())
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
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

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
#define cudaSuccess 0

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
static int allocation_fail, host_allocation_fail, low_memory, device_live;

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
    return cudaSuccess;
}
static void Force_gpu_pool_flush(void) {}
static void* Force_gpu_arena_try(size_t n)
{
    void* ptr = allocation_fail ? NULL : malloc(n);
    if (ptr != NULL) device_live++;
    return ptr;
}
static void acc_free(void* ptr)
{
    if (ptr != NULL) device_live--;
    free(ptr);
}
static void acc_memcpy_to_device(void* dst, void* src, size_t n) { memcpy(dst, src, n); }
static void acc_memcpy_from_device(void* dst, void* src, size_t n) { memcpy(dst, src, n); }
static int Force_collective_env_flag(const char* name, int default_value, int comm)
{
    const char* value = getenv(name);
    return value != NULL ? atoi(value) : default_value;
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
static void* test_calloc(size_t count, size_t size)
{
    return host_allocation_fail ? NULL : calloc(count, size);
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

int main(void)
{
    int mc, direction, k, i, j, l, spin, solver, seed, checks = 0;
    const int solvers[] = {3, 5, 8, 11};
    double max_error = 0.0;

    unsetenv("OPENMX_FORCE4B_GPU");
    unsetenv("OPENMX_FORCE4B_CASE2_STREAM");
    allocate_fixture();
    for (spin = 0; spin < 3; spin++) {
        for (solver = 0; solver < 4; solver++) {
            for (seed = 0; seed < 7; seed++) {
                SpinP_switch = spin == 2 ? 3 : spin;
                Solver = solvers[solver];
                CHECK(Force4B_GpuCase2StreamBegin(0, -1));
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
                CHECK(device_live == 0);
            }
        }
    }

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

    allocation_fail = 1;
    CHECK(!Force4B_GpuCase2StreamBegin(0, -1));
    CHECK(!Force4B_GpuCase2StreamRun(1, cdm, ds));
    allocation_fail = 0;
    host_allocation_fail = 1;
    CHECK(!Force4B_GpuCase2StreamBegin(0, -1));
    CHECK(device_live == 0);
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
    CHECK(device_live == 0);
    free_fixture();
    printf("PASS: %d CPU mixed-species/spin/solver cases; max force error %.3g; "
        "configuration, OOM, low-memory, full-batch, undersized-arena, empty-centre/rank checks PASS\n",
        checks, max_error);
    return 0;
}
'''


def make_harness(source: str) -> str:
    parts = [STUBS, extract_struct(source, "Force4BGpuItem")]
    for name in ("Force_gpu_arena_off", "Force4B_case2_trace_fused", "Force4B_GpuCase2Kernel"):
        parts.append(extract_function(source, name))
    parts.extend(extract_struct(source, name) for name in ("Force4BCase2Plan", "Force4BCase2Stream"))
    parts.append("static Force4BCase2Stream F4B_stream = { 0 };\n")
    for name in (
        "Force4B_GpuCase2StreamMode", "Force4B_GpuCase2Plan", "Force4B_GpuCase2StreamEnd",
        "Force4B_GpuCase2StreamBegin", "Force4B_GpuCase2StreamRun", "Force4B_GpuCase2StreamAccumulate",
    ):
        parts.append(extract_function(source, name))
    parts.append(FIXTURE)
    return "\n".join(parts)


def main() -> int:
    try:
        harness = make_harness((ROOT / "source" / "Force.c").read_text())
        compiler = shlex.split(os.environ.get("CC", "cc"))
        if not compiler:
            raise ValueError("CC must name a C compiler")
        flags = shlex.split(os.environ.get("CFLAGS", "-O2"))
        with tempfile.TemporaryDirectory(prefix="openmx-force4b-cpu-") as directory:
            source = Path(directory) / "force4b_stream_cpu.c"
            binary = Path(directory) / "force4b_stream_cpu"
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
        print(f"Force4B CPU test failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

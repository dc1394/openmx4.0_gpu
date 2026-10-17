#!/usr/bin/env python3
"""Check the HVNA batch of source/Set_ProExpn_VNA.c without MPI (AMD).

Extracts SetPro_HVNA_GpuBegin/ArchiveHalo/GpuRun/GpuEnd with their helpers
and builds them against single-process stubs.  MPI node reductions emulate
FAKE_NODE identical ranks (SUM multiplies, MIN/MAX copy), so the turn
admission can be exercised.  Checks:
  - HVNA from the device arena and from the host fallback (forced hipMalloc
    failure) against an independent reference of the contraction;
  - rank turns, OPENMX_SETPRO_HVNA_TURNS=0, the node RAM cap
    (MemAvailable/2 over the node sum of need) and an unreadable
    /proc/meminfo (no cap);
  - the level1 agreement on the host archives (an allocation failure sends
    every rank to the CPU path; nothing leaks);
  - the stage gates (OPENMX_SETPRO_GPU, OPENMX_SETPRO_HVNA_GPU, APU default);
  - OPENMX_SETPRO_VERBOSE notes only when requested.

    python3 tests/run_setpro_hvna_cpu.py --cc /usr/bin/gcc [--sanitize]
    python3 tests/run_setpro_hvna_cpu.py --gpu     # kernel on the GPU (gfx1200)
"""

import argparse
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def extract_function(source: str, name: str) -> str:
    """Extract a static definition (forward declarations are skipped)."""
    token = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|/\*.*?\*/|//[^\n]*|[{}]', re.S)
    for match in re.finditer(r"^static [^\n]*\b" + re.escape(name) + r"\(", source, re.M):
        opening = source.index("{", match.end())
        if ";" in source[match.end():opening]:
            continue
        depth = 0
        for item in token.finditer(source, opening):
            if item.group() == "{":
                depth += 1
            elif item.group() == "}":
                depth -= 1
                if depth == 0:
                    return source[match.start():item.end()] + "\n"
        raise ValueError(f"unterminated production function: {name}")
    raise ValueError(f"missing production function: {name}")


def extract_struct(source: str, name: str) -> str:
    match = re.search(r"typedef struct\s*\{[^{}]*\}\s*" + re.escape(name) + r";", source)
    if match is None:
        raise ValueError(f"missing production structure: {name}")
    return match.group() + "\n"


PREFIX = r'''
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
static int level_stdout=3;
static int SetPro_gpu_used=0;
#include <stdlib.h>
#include <string.h>

#define CHECK(condition) do { if (!(condition)) { \
    fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #condition); exit(2); } } while (0)

typedef float Type_DS_VNA;
typedef int MPI_Comm;
#define MPI_COMM_NULL 0
#define MPI_COMM_TYPE_SHARED 0
#define MPI_INFO_NULL 0
#define MPI_UNSIGNED_LONG_LONG 1
#define MPI_INT 2
#define MPI_MAX 10
#define MPI_MIN 11
#define MPI_SUM 12
#define GPUSOLVER 6
#define FAKE_NODE_COMM 7
enum { hipSuccess = 0, hipMemcpyHostToDevice = 1, hipMemcpyDeviceToHost = 2 };

static int scf_eigen_lib_flag = GPUSOLVER, mpi_comm_level1 = 1;
static int Matomnum = 3, MatomnumF = 3, SpeciesNum = 2, List_YOUSO[50];
static int *M2G, *F_M2G, *F_G2M, *FNAN, **natn, ***RMI1, *WhatSpecies, *Spe_Total_NO;
static double ***VNA_proj_ene, *Spe_Atom_Cut1, **Dis;
static int fake_node_ranks = 1, fake_node_rank = 0, apu = 0;
static size_t free_bytes_stub = (size_t)1 << 30;
static unsigned long long host_kib_stub = ULLONG_MAX;
static int device_fail, device_live, malloc_countdown = -1;

static int MPI_Comm_split_type(int a, int b, int c, int d, int* e) { *e = FAKE_NODE_COMM; return 0; }
static int MPI_Comm_rank(int comm, int* rank) { *rank = comm == FAKE_NODE_COMM ? fake_node_rank : 0; return 0; }
static int MPI_Comm_size(int comm, int* size) { *size = comm == FAKE_NODE_COMM ? fake_node_ranks : 1; return 0; }
static int MPI_Comm_free(int* comm) { *comm = MPI_COMM_NULL; return 0; }
static int MPI_Barrier(int comm) { return 0; }
static int MPI_Abort(int comm, int code) { exit(3); }
static int MPI_Bcast(void* a, int b, int c, int d, int e) { return 0; }
/* identical ranks on the node: SUM scales, MIN/MAX copy */
static int MPI_Allreduce(const void* src, void* dst, int count, int type, int op, int comm)
{
    const int scale = (op == MPI_SUM && comm == FAKE_NODE_COMM) ? fake_node_ranks : 1;
    if (type == MPI_UNSIGNED_LONG_LONG) {
        for (int i = 0; i < count; i++)
            ((unsigned long long*)dst)[i] = ((const unsigned long long*)src)[i] * (unsigned)scale;
    } else {
        for (int i = 0; i < count; i++) ((int*)dst)[i] = ((const int*)src)[i] * scale;
    }
    return 0;
}
static int openmx_gpu_is_apu(void) { return apu; }
static int OpenMX_GpuMemGetInfo(size_t* free_bytes, size_t* total_bytes)
{
    *free_bytes = free_bytes_stub;
    *total_bytes = (size_t)16 << 30;
    return hipSuccess;
}
static unsigned long long SetPro_host_available_kib(void) { return host_kib_stub; }
static void dtime(double* t) { *t = 0.0; }
static double dampingF(double rcut, double r) { return 1.0 / (1.0 + rcut + r); }
static void* test_malloc(size_t n)
{
    if (malloc_countdown == 0) { malloc_countdown = -1; return NULL; }
    if (malloc_countdown > 0) malloc_countdown--;
    return malloc(n);
}
#define malloc test_malloc
'''

HOST_DEVICE = r'''
static int hipMalloc(void** ptr, size_t n)
{
    *ptr = device_fail ? NULL : malloc(n);
    if (*ptr == NULL) return 2;
    device_live++;
    return hipSuccess;
}
static int hipFree(void* ptr) { if (ptr != NULL) device_live--; free(ptr); return hipSuccess; }
static int hipMemcpy(void* dst, const void* src, size_t n, int kind) { memcpy(dst, src, n); return hipSuccess; }
static int hipDeviceSynchronize(void) { return hipSuccess; }
static int hipGetLastError(void) { return hipSuccess; }
'''

HIP_DEVICE = r'''
extern int hipMalloc(void** ptr, size_t n);
extern int hipFree(void* ptr);
extern int hipMemcpy(void* dst, const void* src, size_t n, int kind);
extern int hipDeviceSynchronize(void);
extern int hipGetLastError(void);
static int hip_malloc_counted(void** ptr, size_t n)
{
    *ptr = NULL;
    if (device_fail || hipMalloc(ptr, n) != hipSuccess) return 2;
    device_live++;
    return hipSuccess;
}
static int hip_free_counted(void* ptr) { if (ptr != NULL) device_live--; return ptr ? hipFree(ptr) : 0; }
#define hipMalloc hip_malloc_counted
#define hipFree hip_free_counted
'''

MAIN = r'''
#define ATOMS 6
#define NB 5          /* neighbours per atom, natn[g][0] == g */
#define PROJ 12       /* (1+1)^2 * 3 */
#define NRVNA 6       /* 3 radial projectors x L = 0, 1 */

static Type_DS_VNA***** ds;          /* production view: local + receive slot */
static Type_DS_VNA**** true0;        /* direction-0 rows of every atom */
static double**** hvna;
static int VNA_List[NRVNA], VNA_List2[NRVNA];

static void* xcalloc(size_t n, size_t s) { void* p = calloc(n, s); CHECK(p != NULL); return p; }

static void fixture(void)
{
    int a, j, k, i, l, mc, sp, L;
    M2G = xcalloc(Matomnum + 2, sizeof(int));
    F_M2G = xcalloc(Matomnum + MatomnumF + 2, sizeof(int));
    F_G2M = xcalloc(ATOMS + 1, sizeof(int));
    FNAN = xcalloc(ATOMS + 1, sizeof(int));
    WhatSpecies = xcalloc(ATOMS + 1, sizeof(int));
    Spe_Total_NO = xcalloc(SpeciesNum, sizeof(int));
    Spe_Atom_Cut1 = xcalloc(SpeciesNum, sizeof(double));
    natn = xcalloc(ATOMS + 1, sizeof(int*));
    Dis = xcalloc(ATOMS + 1, sizeof(double*));
    Spe_Total_NO[0] = 2; Spe_Total_NO[1] = 3;
    Spe_Atom_Cut1[0] = 4.0; Spe_Atom_Cut1[1] = 5.0;
    /* global atoms 1..6: 1..3 local, 4..6 halo (Original_Mc_AN 4..6) */
    for (a = 1; a <= ATOMS; a++) {
        WhatSpecies[a] = a % 2;
        FNAN[a] = NB - 1;
        natn[a] = xcalloc(NB, sizeof(int));
        Dis[a] = xcalloc(NB, sizeof(double));
        for (j = 0; j < NB; j++) {
            natn[a][j] = (a - 1 + j) % ATOMS + 1;
            Dis[a][j] = 0.3 + 0.7 * j;
        }
    }
    for (mc = 1; mc <= Matomnum; mc++) { M2G[mc] = mc; F_M2G[mc] = mc; F_G2M[mc] = mc; }
    for (mc = Matomnum + 1; mc <= Matomnum + MatomnumF; mc++) { F_M2G[mc] = mc; F_G2M[mc] = mc; }
    RMI1 = xcalloc(Matomnum + 1, sizeof(int**));
    for (mc = 1; mc <= Matomnum; mc++) {
        RMI1[mc] = xcalloc(NB, sizeof(int*));
        for (j = 0; j < NB; j++) {
            RMI1[mc][j] = xcalloc(NB, sizeof(int));
            for (k = 0; k < NB; k++)
                RMI1[mc][j][k] = ((j + 2 * k + mc) % 5 == 0) ? -1 : (j + k + mc) % NB;
        }
    }
    VNA_proj_ene = xcalloc(SpeciesNum, sizeof(double**));
    for (sp = 0; sp < SpeciesNum; sp++) {
        VNA_proj_ene[sp] = xcalloc(2, sizeof(double*));
        for (L = 0; L < 2; L++) {
            VNA_proj_ene[sp][L] = xcalloc(3, sizeof(double));
            for (i = 0; i < 3; i++) VNA_proj_ene[sp][L][i] = 0.5 + 0.25 * sp - 0.1 * L + 0.07 * i;
        }
    }
    for (l = 0, L = 0; L < 2; L++)
        for (i = 0; i < 3; i++, l++) { VNA_List[l] = L; VNA_List2[l] = i; }
    List_YOUSO[35] = 1; List_YOUSO[34] = 3;
    true0 = xcalloc(ATOMS + 1, sizeof(Type_DS_VNA***));
    for (a = 1; a <= ATOMS; a++) {
        true0[a] = xcalloc(NB, sizeof(Type_DS_VNA**));
        for (k = 0; k < NB; k++) {
            true0[a][k] = xcalloc(3, sizeof(Type_DS_VNA*));
            for (i = 0; i < 3; i++) {
                true0[a][k][i] = xcalloc(PROJ, sizeof(Type_DS_VNA));
                for (l = 0; l < PROJ; l++)
                    true0[a][k][i][l] = (Type_DS_VNA)(0.4 * sin(0.3 * a + 0.7 * k + 1.1 * i + 0.13 * l));
            }
        }
    }
    ds = xcalloc(1, sizeof(Type_DS_VNA****));
    ds[0] = xcalloc(Matomnum + 2, sizeof(Type_DS_VNA***));
    for (mc = 1; mc <= Matomnum; mc++) ds[0][mc] = true0[mc];
    ds[0][Matomnum + 1] = xcalloc(NB, sizeof(Type_DS_VNA**));
    for (k = 0; k < NB; k++) {
        ds[0][Matomnum + 1][k] = xcalloc(3, sizeof(Type_DS_VNA*));
        for (i = 0; i < 3; i++) ds[0][Matomnum + 1][k][i] = xcalloc(PROJ, sizeof(Type_DS_VNA));
    }
    hvna = xcalloc(Matomnum + 1, sizeof(double***));
    for (mc = 1; mc <= Matomnum; mc++) {
        hvna[mc] = xcalloc(NB, sizeof(double**));
        for (j = 0; j < NB; j++) {
            hvna[mc][j] = xcalloc(3, sizeof(double*));
            for (i = 0; i < 3; i++) hvna[mc][j][i] = xcalloc(3, sizeof(double));
        }
    }
}

static void free_fixture(void)
{
    int a, j, k, i, mc, sp;
    for (mc = 1; mc <= Matomnum; mc++) {
        for (j = 0; j < NB; j++) {
            for (i = 0; i < 3; i++) free(hvna[mc][j][i]);
            free(hvna[mc][j]); free(RMI1[mc][j]);
        }
        free(hvna[mc]); free(RMI1[mc]);
    }
    free(hvna); free(RMI1);
    for (k = 0; k < NB; k++) {
        for (i = 0; i < 3; i++) free(ds[0][Matomnum + 1][k][i]);
        free(ds[0][Matomnum + 1][k]);
    }
    free(ds[0][Matomnum + 1]); free(ds[0]); free(ds);
    for (a = 1; a <= ATOMS; a++) {
        for (k = 0; k < NB; k++) {
            for (i = 0; i < 3; i++) free(true0[a][k][i]);
            free(true0[a][k]);
        }
        free(true0[a]); free(natn[a]); free(Dis[a]);
    }
    free(true0);
    for (sp = 0; sp < SpeciesNum; sp++) {
        for (j = 0; j < 2; j++) free(VNA_proj_ene[sp][j]);
        free(VNA_proj_ene[sp]);
    }
    free(VNA_proj_ene);
    free(M2G); free(F_M2G); free(F_G2M); free(FNAN); free(WhatSpecies);
    free(Spe_Total_NO); free(Spe_Atom_Cut1); free(natn); free(Dis);
}

/* independent reference: dmp * sum_k sum_L ene_L sum_l a*b, CPU order */
static double reference(int mc, int j, int m, int n)
{
    const int gc = M2G[mc], jg = natn[gc][j];
    double nlh = 0.0;
    for (int k = 0; k <= FNAN[gc]; k++) {
        const int kl = RMI1[mc][j][k];
        const int spk = WhatSpecies[natn[gc][k]];
        double sum = 0.0;
        int l = 0;
        if (kl < 0) continue;
        for (int L1 = 0; L1 < NRVNA; L1++) {
            double tmp0 = 0.0;
            for (int l3 = 0; l3 < 2 * VNA_List[L1] + 1; l3++, l++)
                tmp0 += (double)(true0[gc][k][m][l] * true0[jg][kl][n][l]);
            sum += VNA_proj_ene[spk][VNA_List[L1]][VNA_List2[L1]] * tmp0;
        }
        nlh += sum;
    }
    return dampingF(Spe_Atom_Cut1[WhatSpecies[gc]] + Spe_Atom_Cut1[WhatSpecies[jg]], Dis[gc][j]) * nlh;
}

static double run_batch(void)
{
    double max_error = 0.0;
    int source, k, i, mc, j, m, n;
    for (source = Matomnum + 1; source <= Matomnum + MatomnumF; source++) {
        for (k = 0; k < NB; k++)
            for (i = 0; i < Spe_Total_NO[WhatSpecies[source]]; i++)
                memcpy(ds[0][Matomnum + 1][k][i], true0[source][k][i], sizeof(Type_DS_VNA) * PROJ);
        SetPro_HVNA_GpuArchiveHalo(ds, source, F_M2G[source]);
    }
    for (k = 0; k < NB; k++)            /* the receive slot is stale now */
        for (i = 0; i < 3; i++)
            for (j = 0; j < PROJ; j++) ds[0][Matomnum + 1][k][i][j] = NAN;
    SetPro_gpu_used = 0;
    SetPro_HVNA_GpuRun(hvna);
    CHECK(SetPro_gpu_used == !device_fail);
    for (mc = 1; mc <= Matomnum; mc++)
        for (j = 0; j <= FNAN[mc]; j++)
            for (m = 0; m < Spe_Total_NO[WhatSpecies[mc]]; m++)
                for (n = 0; n < Spe_Total_NO[WhatSpecies[natn[mc][j]]]; n++) {
                    const double ref = reference(mc, j, m, n);
                    const double err = fabs(hvna[mc][j][m][n] - ref);
                    CHECK(isfinite(hvna[mc][j][m][n]) && err <= 1e-12 * (1.0 + fabs(ref)));
                    if (max_error < err) max_error = err;
                }
    return max_error;
}

static int begin(void) { return SetPro_HVNA_GpuBegin(ds, VNA_List, VNA_List2, NRVNA); }

int main(void)
{
    const size_t MiB = (size_t)1 << 20;
    double err, max_error = 0.0;
    int countdown;

    unsetenv("OPENMX_SETPRO_GPU");
    unsetenv("OPENMX_SETPRO_HVNA_GPU");
    unsetenv("OPENMX_SETPRO_HVNA_TURNS");
    fixture();

    /* device batch, then the host fallback of a failed claim */
    CHECK(begin() && SetPro_gpu.turns == 1 && SetPro_gpu.turn == 0);
    err = run_batch(); if (max_error < err) max_error = err;
    CHECK(device_live == 0);
    SetPro_HVNA_GpuEnd();
    CHECK(begin());
    device_fail = 1;
    err = run_batch(); if (max_error < err) max_error = err;
    device_fail = 0;
    CHECK(device_live == 0);
    SetPro_HVNA_GpuEnd();

    /* four identical node ranks, ~64 MiB need each (the 64 MiB slack) */
    fake_node_ranks = 4;
    free_bytes_stub = 256 * MiB + 200 * MiB;     /* 3 per turn */
    fake_node_rank = 3;
    CHECK(begin() && SetPro_gpu.turns == 2 && SetPro_gpu.turn == 1);
    SetPro_HVNA_GpuEnd();
    free_bytes_stub = 256 * MiB + 100 * MiB;     /* 1 per turn */
    CHECK(begin() && SetPro_gpu.turns == 4 && SetPro_gpu.turn == 3);
    err = run_batch(); if (max_error < err) max_error = err;
    SetPro_HVNA_GpuEnd();
    setenv("OPENMX_SETPRO_HVNA_TURNS", "0", 1);  /* turns needed: CPU */
    CHECK(!begin());
    free_bytes_stub = (size_t)16 << 30;          /* all fit at once */
    CHECK(begin() && SetPro_gpu.turns == 1);
    SetPro_HVNA_GpuEnd();
    unsetenv("OPENMX_SETPRO_HVNA_TURNS");
    free_bytes_stub = 256 * MiB + 60 * MiB;      /* not even one rank */
    CHECK(!begin());
    free_bytes_stub = (size_t)16 << 30;
    /* node RAM cap: 4 x ~64 MiB against half of MemAvailable */
    host_kib_stub = 400 * 1024;                  /* 200 MiB budget */
    CHECK(!begin());
    host_kib_stub = 1024 * 1024;                 /* 512 MiB budget */
    CHECK(begin());
    SetPro_HVNA_GpuEnd();
    host_kib_stub = ULLONG_MAX;                  /* unreadable: no cap */
    CHECK(begin());
    SetPro_HVNA_GpuEnd();
    fake_node_ranks = 1;
    fake_node_rank = 0;

    /* host archive failures (3rd: flat, 6th: halo malloc) */
    for (countdown = 2; countdown <= 5; countdown += 3) {
        malloc_countdown = countdown;
        CHECK(!begin());
        CHECK(SetPro_gpu.flat == NULL && SetPro_gpu.halo == NULL && SetPro_gpu.enabled == 0);
        malloc_countdown = -1;
    }

    /* stage gates */
    setenv("OPENMX_SETPRO_GPU", "0", 1);
    CHECK(!begin());
    setenv("OPENMX_SETPRO_HVNA_GPU", "1", 1);
    CHECK(begin());
    SetPro_HVNA_GpuEnd();
    unsetenv("OPENMX_SETPRO_GPU");
    unsetenv("OPENMX_SETPRO_HVNA_GPU");
    apu = 1;
    CHECK(!begin());
    apu = 0;
    scf_eigen_lib_flag = 0;
    CHECK(!begin());
    scf_eigen_lib_flag = GPUSOLVER;

    free_fixture();
    printf("PASS: HVNA device batch and host fallback against the reference (max error %.3g); "
        "rank turns, OPENMX_SETPRO_HVNA_TURNS=0, node RAM cap, unreadable meminfo, host archive "
        "agreement and stage gates PASS\n", max_error);
    return 0;
}
'''


def make_harness(source: str, gpu: bool) -> str:
    parts = [PREFIX, '#include "openmx_precision.h"', HIP_DEVICE if gpu else HOST_DEVICE,
             extract_struct(source, "SetProGpuContext"),
             "static SetProGpuContext SetPro_gpu = { 0 };\n"]
    for name in ("SetPro_gpu_abort", "SetPro_checked_malloc", "SetPro_collective_env_flag",
                 "SetPro_gpu_default_enabled", "SetPro_gpu_stage_enabled", "SetPro_verbose",
                 "SetPro_free_mib", "SetPro_note", "SetPro_arena_off", "SetPro_device_try",
                 "SetPro_HVNA_element", "SetPro_HVNA_GpuBegin", "SetPro_HVNA_GpuEnd",
                 "SetPro_HVNA_GpuArchiveHalo", "SetPro_HVNA_GpuRun"):
        parts.append(extract_function(source, name))
    parts.append(MAIN)
    return "\n".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cc", default=os.environ.get("CC", "cc"))
    parser.add_argument("--cflags", default=os.environ.get("CFLAGS", "-O2"))
    parser.add_argument("--sanitize", action="store_true")
    parser.add_argument("--gpu", action="store_true", help="offload the kernel (clang of --rocm)")
    parser.add_argument("--rocm", default=os.environ.get("ROCM_PATH", "/opt/rocm-7.2.0"))
    parser.add_argument("--arch", default="gfx1200")
    args = parser.parse_args()
    try:
        harness = make_harness((ROOT / "source" / "Set_ProExpn_VNA.c").read_text(), args.gpu)
        if args.gpu:
            rocm = Path(args.rocm)
            command = [str(rocm / "llvm" / "bin" / "clang"), "-O2", "-D_POSIX_C_SOURCE=200809L",
                       "-fopenmp", f"--offload-arch={args.arch}",
                       f"-L{rocm / 'lib'}", f"-Wl,-rpath,{rocm / 'lib'}", "-lamdhip64"]
        else:
            command = [*shlex.split(args.cc), *shlex.split(args.cflags), "-D_POSIX_C_SOURCE=200809L",
                       "-Wno-unknown-pragmas"]
            if args.sanitize:
                command += ["-O1", "-g", "-fsanitize=address,undefined", "-fno-sanitize-recover=all"]
        with tempfile.TemporaryDirectory(prefix="openmx-setpro-hvna-") as directory:
            src = Path(directory) / "hvna.c"
            exe = Path(directory) / "hvna"
            src.write_text(harness)
            subprocess.run([*command, "-std=c11", "-Wno-unused-function", "-I", str(ROOT / "source"), str(src), str(ROOT / "source/openmx_precision.c"), "-lm", "-o", str(exe)],
                           check=True)
            status = 0
            for verbose in ("", "1"):
                env = dict(os.environ, OPENMX_SETPRO_VERBOSE=verbose)
                if args.gpu:
                    env["OMP_TARGET_OFFLOAD"] = "MANDATORY"
                result = subprocess.run([str(exe)], text=True, capture_output=True, env=env)
                notes = result.stderr.count("<Set_ProExpn_VNA> rank 0 ")
                if result.returncode or (notes == 0) == bool(verbose):
                    sys.stderr.write(result.stderr)
                    sys.stdout.write(result.stdout)
                    print(f"verbose={verbose!r}: rc {result.returncode}, {notes} note line(s)",
                          file=sys.stderr)
                    status = 1
                else:
                    tag = "[GPU] " if args.gpu else ""
                    sys.stdout.write(f"{tag}OPENMX_SETPRO_VERBOSE={verbose or 'unset'} "
                                     f"({notes} note lines): {result.stdout}")
            return status
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"Set_ProExpn_VNA HVNA test failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

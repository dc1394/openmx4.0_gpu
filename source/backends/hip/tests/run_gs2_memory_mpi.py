#!/usr/bin/env python3
"""Exercise the production ELPA/COSMA admission and handle lifecycle with MPI.

HIP, ELPA and BLACS are mocked; MPI collectives are real. No GPU is needed.
Run with mpicc/mpirun in PATH (CC/MPICC/MPIEXEC can override the tools).
"""
from pathlib import Path
import os
import shlex
import signal
import subprocess
import tempfile
from mock_hip_sources import stage

ROOT = Path(__file__).resolve().parents[1]

HIP = r'''
#pragma once
#include <stddef.h>
typedef int hipError_t;
enum { hipSuccess = 0 };
typedef struct { char bytes[16]; } hipUUID_t;
struct hipDeviceProp_t { hipUUID_t uuid; };
int hipGetDevice(int *);
int hipGetDeviceCount(int *);
int hipGetDeviceProperties(struct hipDeviceProp_t *, int);
int hipMemGetInfo(size_t *,size_t *);
int hipGetLastError(void);
'''
ELPA = r'''
#pragma once
#include <complex.h>
typedef void *elpa_t;
enum { ELPA_OK = 0, ELPA_API_VERSION = 1,
       ELPA_SOLVER_1STAGE = 1, ELPA_SOLVER_2STAGE = 2 };
int elpa_init(int);
void elpa_uninit(int *);
elpa_t elpa_allocate(int *);
void elpa_deallocate(elpa_t, int *);
void elpa_set_integer(elpa_t, const char *, int, int *);
int elpa_setup(elpa_t);
const char *elpa_strerr(int);
void elpa_eigenvectors_double(elpa_t, double *, double *, double *, int *);
void elpa_eigenvectors_double_complex(elpa_t, double complex *, double *, double complex *, int *);
'''
DRIVER = r'''
#define _POSIX_C_SOURCE 200809L
#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "hip_runtime_compat.h"
static int rank_id, rank_count, visible = 4, device, physical;
static int allocation_failure, property_failure, launcher_size;
static size_t device_free = (size_t)16 << 30;
int SCF_Gpu_Num = 4, level_stdout = 0;
static MPI_Comm grid_comm;
static void *test_malloc(size_t n) {
    return allocation_failure && rank_id == 1 ? NULL : malloc(n);
}
#define malloc test_malloc
int openmx_gpu_physical_uuid(int device,unsigned char uuid[16]){struct hipDeviceProp_t p; if(hipGetDeviceProperties(&p,device)!=hipSuccess)return 0;memcpy(uuid,p.uuid.bytes,16);return 1;}
#include "elpa_cosma_bridge.c"
#undef malloc

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "rank %d line %d: %s\n", rank_id, __LINE__, #x); MPI_Abort(MPI_COMM_WORLD, 1); } } while (0)
int hipGetDevice(int *d) { *d = device; return 0; }
int hipGetDeviceCount(int *n) { *n = visible; return 0; }
int hipGetDeviceProperties(struct hipDeviceProp_t *p, int d) {
    (void)d;
    memset(p, 0, sizeof(*p)); p->uuid.bytes[0] = (char)(physical + 1);
    return property_failure && rank_id == 1;
}
int hipGetLastError(void) { return 0; }
hipError_t hipMemGetInfo(size_t *f, size_t *t) { *f = device_free; *t = (size_t)40 << 30; return 0; }
int openmx_gpu_local_size_noncollective(void) { return launcher_size; }
/* The actual rank map is appended from set_hip_default_device_from_local_rank.c. */

void Cblacs_gridinfo(int ctx, int *nr, int *nc, int *mr, int *mc) {
    (void)ctx;
    MPI_Comm_size(grid_comm, nr); MPI_Comm_rank(grid_comm, mr); *nc = 1; *mc = 0;
}
void Cblacs_get(int ctx, int what, int *out) { (void)what; *out = ctx; }
MPI_Comm Cblacs2sys_handle(int ctx) { (void)ctx; return grid_comm; }
int numroc_(int *n, int *nb, int *rank, int *src, int *np) {
    int count = 0;
    for (int i = 0; i < *n; ++i) count += ((i / *nb + *src) % *np == *rank);
    return count;
}
unsigned long long openmx_host_mem_available(void) { return 1ULL << 40; }
unsigned long long openmx_host_mem_total(void) { return 1ULL << 40; }
double openmx_gs2_cosma_host_bytes(char a, char b, int m, int n, int k,
                                  const int *da, const int *db, const int *dc, int ctx, int z) {
    (void)a; (void)b; (void)m; (void)n; (void)k; (void)da; (void)db; (void)dc; (void)ctx; (void)z;
    return 1024.0;
}
static int live_handles, allocations, last_block, fail_setup, fail_allocate, fail_init;
static const char *fail_key;
int elpa_init(int v) { (void)v; return fail_init ? 42 : 0; }
void elpa_uninit(int *err) { *err = 0; }
elpa_t elpa_allocate(int *err) {
    if (fail_allocate) { *err = 42; return NULL; }
    ++live_handles; ++allocations; *err = 0; return &live_handles;
}
void elpa_deallocate(elpa_t h, int *err) { CHECK(h == &live_handles); --live_handles; *err = 0; }
void elpa_set_integer(elpa_t h, const char *key, int v, int *err) {
    CHECK(h == &live_handles);
    if (strcmp(key, "nblk") == 0) last_block = v;
    *err = fail_key && strcmp(key, fail_key) == 0 ? 42 : 0;
}
int elpa_setup(elpa_t h) {
    CHECK(h == &live_handles);
    MPI_Barrier(grid_comm); /* All ranks must enter setup together. */
    return fail_setup ? 42 : 0;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_id); MPI_Comm_size(MPI_COMM_WORLD, &rank_count);
    CHECK(rank_count == 4 && argc == 2);
    grid_comm = MPI_COMM_WORLD;
    launcher_size = rank_count;
    unsetenv("OPENMX_GS2_ELPA_GPU"); unsetenv("OPENMX_GS2_GEMM_GPU");
    unsetenv("OPENMX_GPU_RANK_MAP");
    setenv("OPENMX_GS2_GPU_RESERVE_MB", "0", 1);
    setenv("OPENMX_GS2_HOST_RESERVE_MB", "0", 1);
    setenv("OMPI_COMM_WORLD_LOCAL_SIZE", "4", 1);

    if (strcmp(argv[1], "cap") == 0) {
        MPI_Comm sub;
        SCF_Gpu_Num = 1; device = physical = 0;
        MPI_Comm_split(MPI_COMM_WORLD, rank_id / 2, rank_id, &sub);
        int count = gs2_ranks_on_device(sub);
        CHECK(count == 4); /* two concurrent spin worlds on the one permitted GPU */
        gs2_mem_details det;
        device_free = 3 * ((size_t)1 << 30);
        CHECK(!gs2_gpu_verdict((size_t)1 << 30, count, sub, -1, 0, &det));
        MPI_Comm_free(&sub);
    }
    else if (strcmp(argv[1], "block") == 0 || strcmp(argv[1], "mod") == 0) {
        visible = SCF_Gpu_Num = 3;
        int modulo = strcmp(argv[1], "mod") == 0;
        if (modulo) setenv("OPENMX_GPU_RANK_MAP", "mod", 1);
        device = modulo ? rank_id % 3 : rank_id / 2; physical = device;
        CHECK(gs2_ranks_on_device(MPI_COMM_SELF) == (modulo ? (device == 0 ? 2 : 1) : 2));
    }
    else if (strcmp(argv[1], "uuid") == 0) {
        /* Different visibility orderings: ordinal 0/1 does not identify a GPU. */
        launcher_size = 1; setenv("OMPI_COMM_WORLD_LOCAL_SIZE", "1", 1);
        device = rank_id % 2; physical = rank_id == 0 ? 0 : 1;
        CHECK(gs2_ranks_on_device(MPI_COMM_WORLD) == (rank_id == 0 ? 1 : 3));
    }
    else if (strcmp(argv[1], "allocation") == 0 || strcmp(argv[1], "query") == 0) {
        allocation_failure = strcmp(argv[1], "allocation") == 0;
        property_failure = !allocation_failure; device = physical = rank_id;
        /* One rank cannot census the devices: every rank must take a
           conservative bound without skipping a collective or hanging. */
        CHECK(gs2_ranks_on_device(MPI_COMM_WORLD) == 4);
    }
    else if (strcmp(argv[1], "grid-change") == 0 || strcmp(argv[1], "grid-cache-skew") == 0) {
        launcher_size = 1; setenv("OMPI_COMM_WORLD_LOCAL_SIZE", "1", 1);
        device = physical = rank_id / 2; device_free = (size_t)20 << 20;
        int desc[9] = {1, 0, 512, 512, 64, 64, 0, 0, 512};
        grid_comm = MPI_COMM_SELF;
        CHECK(gs2_gemm_use_gpu("N", "N", 512, 512, 512, desc, desc, desc, 0));
        if (strcmp(argv[1], "grid-cache-skew") == 0 && rank_id == 0) openmx_gs2_grid_free(0);
        grid_comm = MPI_COMM_WORLD; desc[1] = strcmp(argv[1], "grid-change") == 0 ? 1 : 0;
        CHECK(!gs2_gemm_use_gpu("N", "N", 512, 512, 512, desc, desc, desc, 0));
    }
    else if (strcmp(argv[1], "overflow") == 0) {
        gs2_mem_details det;
        CHECK(!gs2_gpu_verdict(SIZE_MAX / 2 + 1, 2, MPI_COMM_WORLD, -1, 0, &det));
        CHECK(!gs2_gpu_verdict(SIZE_MAX - 32, 1, MPI_COMM_WORLD, 0, 0, &det));
    }
    else if (strncmp(argv[1], "mixed-", 6) == 0) {
        const int gemm = strcmp(argv[1], "mixed-gemm") == 0;
        if (rank_id == 0) setenv(gemm ? "OPENMX_GS2_GEMM_GPU" : "OPENMX_GS2_ELPA_GPU", "0", 1);
        if (gemm) {
            int desc[9] = {1, 0, 64, 64, 16, 16, 0, 0, 64};
            (void)gs2_gemm_use_gpu("N", "N", 64, 64, 64, desc, desc, desc, 0);
        } else {
            int err;
            (void)gs2_get_handle(0, 64, 64, 0, 16, &err);
        }
        CHECK(0); /* inconsistent kernel modes must stop collectively */
    }
    else if (strcmp(argv[1], "handle-errors") == 0) {
        setenv("OPENMX_GS2_ELPA_GPU", "0", 1);
        const char *keys[] = {"na", "amd-gpu", "solver", NULL};
        for (int i = 0; i < 4; ++i) {
            int err;
            fail_key = keys[i]; fail_setup = i == 3;
            CHECK(gs2_get_handle(0, 64, 64, 0, 16, &err) == NULL && err == 42);
            CHECK(live_handles == 0);
        }
        fail_key = NULL; fail_setup = 0;
        int err;
        CHECK(gs2_get_handle(0, 64, 64, 0, 16, &err) && err == 0 && live_handles == 1);
        openmx_gs2_grid_free(0);
        CHECK(live_handles == 0);
    }
    else if (strcmp(argv[1], "small-solve-report") == 0) {
        int err;
        level_stdout = 2;
        gs2_handle_slot *s = gs2_get_handle(0, 64, 64, 0, 16, &err);
        CHECK(s && err == 0 && !s->gpu);
        CHECK(gs2_get_handle(0, 64, 64, 0, 16, &err) == s && allocations == 1);
        openmx_gs2_grid_free(0); CHECK(live_handles == 0);
    }
    else if (strcmp(argv[1], "block-change") == 0) {
        int err;
        setenv("OPENMX_GS2_ELPA_GPU", "0", 1);
        CHECK(gs2_get_handle(0, 64, 64, 0, 16, &err) && last_block == 16);
        CHECK(gs2_get_handle(0, 64, 64, 0, 8, &err) && last_block == 8);
        CHECK(allocations == 2 && live_handles == 1);
        openmx_gs2_grid_free(0); CHECK(live_handles == 0);
    }
    else if (strcmp(argv[1], "asymmetric-handle-errors") == 0) {
        int err;
        setenv("OPENMX_GS2_ELPA_GPU", "0", 1);
        fail_init = rank_id == 0;
        CHECK(gs2_get_handle(0, 64, 64, 0, 16, &err) == NULL && err != 0);
        fail_init = 0;
        const char *keys[] = {"na", "amd-gpu", "solver", NULL, NULL};
        for (int j = 0; j < 5; ++j) {
            fail_key = rank_id == 0 ? keys[j] : NULL;
            fail_setup = rank_id == 0 && j == 3;
            fail_allocate = rank_id == 0 && j == 4;
            CHECK(gs2_get_handle(0, 64, 64, 0, 16, &err) == NULL && err == 42);
            CHECK(live_handles == 0);
        }
        fail_key = NULL; fail_setup = fail_allocate = 0;
        CHECK(gs2_get_handle(0, 64, 64, 0, 16, &err) && err == 0);
        openmx_gs2_grid_free(0); CHECK(live_handles == 0);
    }
    else if (strcmp(argv[1], "handle-cache-skew") == 0 ||
             strcmp(argv[1], "handle-comm-change") == 0 ||
             strcmp(argv[1], "handle-mode-skew") == 0) {
        int err, comm_change = strcmp(argv[1], "handle-comm-change") == 0;
        setenv("OPENMX_GS2_ELPA_GPU", "0", 1);
        if (comm_change) grid_comm = MPI_COMM_SELF;
        CHECK(gs2_get_handle(0, 64, 64, 0, 16, &err));
        if (strcmp(argv[1], "handle-cache-skew") == 0 && rank_id == 0) openmx_gs2_grid_free(0);
        if (strcmp(argv[1], "handle-mode-skew") == 0 && rank_id == 0) gs2_slots[0].gpu = 1;
        grid_comm = MPI_COMM_WORLD;
        CHECK(gs2_get_handle(0, 64, 64, 0, 16, &err) && err == 0);
        CHECK(allocations == 2 && live_handles == 1);
        CHECK(gs2_get_handle(0, 64, 64, 0, 16, &err) && allocations == 2);
        openmx_gs2_grid_free(0); CHECK(live_handles == 0);
    }
    else CHECK(0);
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank_id == 0) printf("PASS: %s\n", argv[1]);
    MPI_Finalize();
    return 0;
}
'''


def rank_map():
    source = (ROOT / "source/set_hip_default_device_from_local_rank.c").read_text()
    start = source.index("int openmx_gpu_map_rank_to_device(")
    end = source.index("\n}\n", start) + 3
    return source[start:end]


def main():
    cases = ["cap", "block", "mod", "uuid", "allocation", "query", "grid-change", "grid-cache-skew",
             "overflow", "mixed-elpa", "mixed-gemm", "handle-errors", "block-change", "small-solve-report",
             "asymmetric-handle-errors", "handle-cache-skew", "handle-comm-change", "handle-mode-skew"]
    failures = []
    with tempfile.TemporaryDirectory(prefix="openmx-gs2-memory-") as directory:
        tmp = Path(directory)
        (tmp / "elpa").mkdir()
        (tmp / "hip_runtime_compat.h").write_text(HIP)
        (tmp / "elpa/elpa.h").write_text(ELPA)
        (tmp / "check.c").write_text(DRIVER + rank_map())
        stage(ROOT,tmp,"elpa_cosma_bridge.c")
        (tmp/'openmx_mgpu_runtime.h').write_text('#pragma once\nint openmx_gpu_physical_uuid(int,unsigned char[16]);\n')
        exe = tmp / "check"
        command = shlex.split(os.environ.get("MPICC", "mpicc")) + [
            "-std=gnu11", "-O1", "-g", "-Wall", "-Wextra", "-ffunction-sections", "-fdata-sections",
            "-fsanitize=undefined", "-fno-sanitize-recover=all", "-Wl,--gc-sections",
            "-I" + str(tmp), "-I" + str(ROOT / "source"), str(tmp / "check.c"), "-o", str(exe)]
        subprocess.run(command, check=True)
        for case in cases:
            command = shlex.split(os.environ.get("MPIEXEC", "mpirun --oversubscribe"))
            proc = subprocess.Popen(command + ["-np", "4", str(exe), case],
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, start_new_session=True)
            try:
                output, _ = proc.communicate(timeout=15)
                if case.startswith("mixed-"):
                    ok = proc.returncode != 0 and "inconsistent" in output and "OPENMX_GS2_" in output
                else:
                    ok = proc.returncode == 0 and "PASS: " + case in output
                    if case == "small-solve-report":
                        ok &= "CPU kernels (small matrix, n < 256)" in output
                        ok &= "GPU memory:" not in output and "OPENMX_GS2_ELPA_GPU)" not in output
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                output, _ = proc.communicate()
                output += "\nMPI collective timed out\n"
                ok = False
            print(("PASS: " if ok else "FAIL: ") + case, flush=True)
            if not ok:
                failures.append(case)
                print(output, flush=True)
    print(f"{len(cases) - len(failures)}/{len(cases)} distributed admission/lifecycle cases passed")
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""CPU-only checks of the resident Force4B batches (AMD).

Extracts Force4B_GpuBegin/GpuEnd/Case2Begin, the archive functions, Case1Run,
Case2Run and the workspace sizing from source/Force.c, with the CPU/MPI/HIP
stubs of run_force4b_stream_cpu.py.  Checks:
  - the exact admission (max of the case-1 and case-2 lifetimes, 64 MiB
    workspace floors) against the earlier summed bound kept for the O(N)
    solvers in auto mode, and the deferred-case variants;
  - the lifetime split: case-1 archives only at Begin, the case-2 archive
    only after Case2Begin(1), a failed device claim uses host archives;
  - Force4B_GpuWorkspaceSizes equals the arenas Case1Run/Case2Run request;
  - forces against the CPU fused traces, and bit-identical forces between
    the resident and the streamed paths of both cases.

    python3 tests/run_force4b_resident_cpu.py --cc /usr/bin/gcc [--sanitize]
"""

import subprocess
import sys


sys.dont_write_bytecode = True  # sibling harness modules, no __pycache__
import run_force4b_case1_cpu as case1
import run_force4b_stream_cpu as common


RESIDENT_STUBS = r'''
#define hipMemcpyDeviceToHost 1
static int hipMemcpy(void *a,const void *b,size_t n,int mode) {(void)mode;memcpy(a,b,n);return hipSuccess;}
#define MPI_MIN 0
#define MPI_IN_PLACE ((void*)1)
static int MPI_Allreduce(const void *a,void *b,int n,int type,int op,MPI_Comm comm) {
    (void)op;(void)comm;
    if (a!=MPI_IN_PLACE) memcpy(b,a,n*(type==MPI_INT?sizeof(int):sizeof(unsigned long long)));
    return 0;
}
static int gpu_rank_device_usable(void) {return 1;}
static int openmx_gpu_capacity_test_recovery(void) {return 0;}
static unsigned long long openmx_host_mem_available(void) {return 1ULL<<40;}
static void openmx_target_wait(void) {}
static void openmx_target_clear_freelists(void) {}
static void openmx_target_free(void *p) {hipFree(p);}
static void openmx_target_upload(void *a,const void *b,size_t n) {memcpy(a,b,n);}
static void openmx_target_download(void *a,const void *b,size_t n) {memcpy(a,b,n);}

static int hipDeviceSynchronize(void) { return hipSuccess; }
static void* Force_checked_malloc(size_t size, const char* file, int line)
{
    void* ptr = malloc(size == 0 ? 1 : size);
    (void)file; (void)line;
    CHECK(ptr != NULL);
    return ptr;
}
static void* ForceHipMalloc(size_t bytes)
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


RESIDENT_MAIN = r'''
static void fill_case2_slot(int seed, int mc)
{
    int direction, k, i, l;
    for (direction = 0; direction < 4; direction++)
        for (k = 0; k < NEIGHBORS; k++)
            for (i = 0; i < ORBITALS; i++)
                for (l = 0; l < PROJECTORS; l++)
                    ds[direction][Matomnum + 1][k][i][l] =
                        cos(seed + 0.17 * mc + 0.23 * direction + 0.11 * k + 0.25 * i + 0.51 * l) * 0.3;
}

static void reset_forces(void)
{
    int mc, i;
    for (mc = 1; mc <= Matomnum; mc++)
        for (i = 0; i < 3; i++) Gxyz[M2G[mc]][41 + i] = 0.37;
}

static void check_admission(void)
{
    const size_t MiB = (size_t)1024 * 1024;
    size_t work1, work2;

    Force4B_GpuWorkspaceSizes(&work1, &work2);
    CHECK(work1 > 0 && work2 > 0 && work1 < 64 * MiB && work2 < 64 * MiB);
    /* 100 MiB per rank: the exact need (64 MiB workspace floor plus the
       small fixture archives) fits, the earlier summed bound (+192 MiB)
       does not */
    free_override = 256 * MiB + 100 * MiB;
    CHECK(Force4B_GpuBegin(ds, 0, 0, 0));
    CHECK(F4B_gpu.flat_dev != NULL && F4B_gpu.halo_dev != NULL && F4B_gpu.c2_dev == NULL);
    Force4B_GpuEnd();
    CHECK(Force4B_GpuBegin(ds, 0, 0, 1));
    CHECK(F4B_gpu.flat_host && F4B_gpu.halo_host);
    Force4B_GpuEnd();
    CHECK(device_live == 0);
    free_override = 256 * MiB + 60 * MiB;
    CHECK(Force4B_GpuBegin(ds, 0, 0, 0));
    CHECK(F4B_gpu.flat_host && F4B_gpu.halo_host);
    Force4B_GpuEnd();
    free_override = 256 * MiB + 400 * MiB;
    CHECK(Force4B_GpuBegin(ds, 0, 0, 1));
    Force4B_GpuEnd();
    free_override = 0;
    /* both cases deferred: no resident context */
    CHECK(!Force4B_GpuBegin(ds, 1, 1, 0));
    /* case 1 deferred: no case-1 archives, case 2 resident later */
    CHECK(Force4B_GpuBegin(ds, 1, 0, 0));
    CHECK(F4B_gpu.flat_dev == NULL && F4B_gpu.halo_dev == NULL && F4B_gpu.c2_dev == NULL);
    CHECK(Force4B_GpuCase2Begin(1));
    CHECK(F4B_gpu.c2_dev != NULL);
    Force4B_GpuEnd();
    /* case 2 forced to stream: Case2Begin(0) only releases case 1 */
    CHECK(Force4B_GpuBegin(ds, 0, 1, 0));
    CHECK(!Force4B_GpuCase2Begin(0));
    CHECK(F4B_gpu.flat_dev == NULL && F4B_gpu.halo_dev == NULL && F4B_gpu.c2_dev == NULL);
    Force4B_GpuEnd();
    /* a failed case-2 claim is reported and leaves case 2 to the caller */
    CHECK(Force4B_GpuBegin(ds, 0, 0, 0));
    allocation_fail = 1;
    CHECK(Force4B_GpuCase2Begin(1));
    CHECK(F4B_gpu.c2_host != NULL);
    allocation_fail = 0;
    CHECK(F4B_gpu.c2_dev == NULL);
    Force4B_GpuEnd();
    /* device allocation failure at Begin: CPU fallback, nothing leaked */
    allocation_fail = 1;
    CHECK(Force4B_GpuBegin(ds, 0, 0, 0));
    CHECK(F4B_gpu.flat_host && F4B_gpu.halo_host);
    Force4B_GpuEnd();
    allocation_fail = 0;
    /* low memory, umbrella/APU gate and non-GPU solver */
    low_memory = 1;
    CHECK(Force4B_GpuBegin(ds, 0, 0, 0));
    CHECK(F4B_gpu.flat_host && F4B_gpu.halo_host);
    Force4B_GpuEnd();
    low_memory = 0;
    stage_default = 0;
    CHECK(!Force4B_GpuBegin(ds, 0, 0, 0));
    setenv("OPENMX_FORCE4B_GPU", "1", 1);
    CHECK(Force4B_GpuBegin(ds, 0, 0, 0));
    Force4B_GpuEnd();
    unsetenv("OPENMX_FORCE4B_GPU");
    stage_default = 1;
    setenv("OPENMX_FORCE4B_GPU", "0", 1);
    CHECK(!Force4B_GpuBegin(ds, 0, 0, 0));
    unsetenv("OPENMX_FORCE4B_GPU");
    scf_eigen_lib_flag = 0;
    CHECK(!Force4B_GpuBegin(ds, 0, 0, 0));
    scf_eigen_lib_flag = GPUSOLVER;
    CHECK(device_live == 0 && pinned_live == 0);
}

int main(void)
{
    const char *precision = getenv("TEST_FP32_ENABLE");
    openmx_set_fp32_enabled(precision == NULL || atoi(precision));
    const int solvers[] = {3, 5, 8, 11};
    int failure, spin, solver, seed, mc, q, source, direction, atom, k, i, j, l, checks = 0, identical = 0;
    double max_error = 0.0;
    unsetenv("OPENMX_FORCE4B_GPU");
    allocate_case1_fixture();
    check_admission();
    for (failure = 0; failure < 2; failure++)
    for (spin = 0; spin < 3; spin++) {
        for (solver = 0; solver < 4; solver++) {
            for (seed = 0; seed < 3; seed++) {
                double expected1[LOCAL_SLOTS][3] = {{0}}, expected2[LOCAL_SLOTS][3] = {{0}};
                double resident[LOCAL_SLOTS][3];
                size_t work1, work2;
                SpinP_switch = spin == 2 ? 3 : spin;
                Solver = solvers[solver];
                Force4B_GpuWorkspaceSizes(&work1, &work2);
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
                            &expected1[mc][0], &expected1[mc][1], &expected1[mc][2]);
                    }
                }

                /* resident case 1: archives at Begin, halo rows as received */
                reset_forces();
                allocation_fail = failure;
                CHECK(Force4B_GpuBegin(ds, 0, 0, 0));
                CHECK(F4B_gpu.c2_dev == NULL);
                for (source = Matomnum + 1; source <= Matomnum + MatomnumF; source++) {
                    stage_source(source);
                    Force4B_GpuArchiveHalo(ds, source, F_M2G[source]);
                }
                last_wait_bytes = 0;
                Force4B_GpuCase1Run(cdm);
                CHECK(failure || last_wait_bytes == work1);
                for (mc = 1; mc <= Matomnum; mc++) {
                    for (i = 0; i < 3; i++) {
                        const double error = fabs(Gxyz[M2G[mc]][41 + i] - (0.37 + expected1[mc][i]));
                        CHECK(isfinite(error) && error <= 1e-11);
                        if (max_error < error) max_error = error;
                        resident[mc][i] = Gxyz[M2G[mc]][41 + i];
                    }
                    checks++;
                }

                /* lifetime split: case-1 archives released before case 2 */
                CHECK(!Force4B_GpuCase2Begin(0));
                CHECK(F4B_gpu.flat_dev == NULL && F4B_gpu.halo_dev == NULL && F4B_gpu.c2_dev == NULL);
                CHECK(Force4B_GpuCase2Begin(1));
                CHECK(failure ? F4B_gpu.c2_host != NULL : F4B_gpu.c2_dev != NULL);
                allocation_fail = 0;

                /* streamed case 1 over the same rows: identical bits */
                reset_forces();
                CHECK(Force4B_GpuCase1StreamBegin(0, -1));
                for (source = 1; source <= Matomnum + MatomnumF; source++) {
                    stage_source(source);
                    Force4B_GpuCase1StreamSource(source, cdm, ds);
                }
                Force4B_GpuCase1StreamAccumulate(cdm);
                Force4B_GpuCase1StreamEnd();
                for (mc = 1; mc <= Matomnum; mc++)
                    CHECK(memcmp(resident[mc], &Gxyz[M2G[mc]][41], sizeof(resident[mc])) == 0);
                identical++;

                /* resident case 2: the staged slot of each centre archived */
                for (mc = 1; mc <= Matomnum; mc++) {
                    fill_case2_slot(seed, mc);
                    for (i = 1; i < NEIGHBORS; i++) {
                        for (j = Solver == 3 ? i : 0; j < NEIGHBORS; j++) {
                            if (!j || i == j || RMI1[mc][i][j] < 0) continue;
                            Force4B_case2_trace_fused(mc, M2G[mc], i, j, cdm, ds,
                                &expected2[mc][0], &expected2[mc][1], &expected2[mc][2]);
                        }
                    }
                    Force4B_GpuArchiveCase2(ds, mc, M2G[mc]);
                }
                reset_forces();
                last_wait_bytes = 0;
                allocation_fail = failure;
                Force4B_GpuCase2Run(cdm);
                allocation_fail = 0;
                CHECK(failure || last_wait_bytes == work2);
                Force4B_GpuEnd();
                CHECK(device_live == 0);
                for (mc = 1; mc <= Matomnum; mc++) {
                    for (i = 0; i < 3; i++) {
                        const double error = fabs(Gxyz[M2G[mc]][41 + i] - (0.37 + expected2[mc][i]));
                        CHECK(isfinite(error) && error <= 1e-11);
                        if (max_error < error) max_error = error;
                        resident[mc][i] = Gxyz[M2G[mc]][41 + i];
                    }
                    checks++;
                }

                /* streamed case 2, accumulated per centre: identical bits */
                reset_forces();
                CHECK(Force4B_GpuCase2StreamBegin(0, -1));
                for (mc = 1; mc <= Matomnum; mc++) {
                    fill_case2_slot(seed, mc);
                    CHECK(Force4B_GpuCase2StreamRun(mc, cdm, ds));
                    Force4B_GpuCase2StreamAccumulate(M2G[mc]);
                }
                Force4B_GpuCase2StreamEnd();
                for (mc = 1; mc <= Matomnum; mc++)
                    CHECK(memcmp(resident[mc], &Gxyz[M2G[mc]][41], sizeof(resident[mc])) == 0);
                identical++;
                CHECK(device_live == 0 && pinned_live == 0);
            }
        }
    }
    free_case1_fixture();
    printf("PASS: %d CPU resident-batch comparisons; max force error %.3g; %d resident/stream "
        "results bit-identical; resident and capacity admission, deferred cases, lifetime split, "
        "workspace sizes, allocation failures and gates PASS\n", checks, max_error, identical);
    return 0;
}
'''


def make_harness(source: str) -> str:
    stubs = common.STUBS.replace(
        "    *free_bytes = low_memory ? 0 : 1024ULL * 1024 * 1024;",
        "    *free_bytes = low_memory ? 0 : free_override != 0 ? free_override : 1024ULL * 1024 * 1024;")
    if stubs == common.STUBS:
        raise ValueError("free-memory stub changed; update the resident harness")
    stubs = stubs.replace("static int device_live, pinned_live, stage_default = 1;",
                          "static int device_live, pinned_live, stage_default = 1;\n"
                          "static size_t free_override, last_wait_bytes;")
    stubs = stubs.replace("static void* Force_gpu_arena_try(size_t n)\n{",
                          "static void* Force_gpu_arena_try(size_t n)\n{\n    last_wait_bytes = n;")
    task_source = (common.ROOT / "source/openmx_gpu_task.c").read_text()
    task_api = (common.ROOT / "source/openmx_gpu_task.h").read_text().replace("#include <mpi.h>", "")
    cpu_task = task_api + "\n".join(common.extract_function(task_source,name) for name in
        ("task_add","task_view_bytes","task_bytes","task_copy_view","task_cpu_bytes","task_cpu"))
    cpu_task += "\nint openmx_gpu_task_execute(MPI_Comm c,OpenMXGpuTask *t,int e,const char *l) {(void)c;(void)e;(void)l;task_cpu(t);return 0;}\n"
    parts = [stubs, cpu_task, case1.GLOBALS, RESIDENT_STUBS,
             common.extract_struct(source, "Force4BGpuItem"),
             common.extract_struct(source, "Force4BGpuContext"),
             "static Force4BGpuContext F4B_gpu = { 0 };\n"]
    for name in (
        "Force_gpu_node_reserve", "Force_gpu_arena_off", "Force_host_arena_alloc", "Force_host_arena_free",
        "Force4B_case2_trace_fused", "Force4B_case1_trace_fused_rows", "Force4B_case1_trace_fused",
        "Force4B_GpuWorkspaceSizes", "Force4B_GpuBegin", "Force4B_GpuEnd", "Force4B_GpuCase2Begin",
        "Force4B_GpuArchiveHalo", "Force4B_GpuArchiveCase2", "Force4B_GpuCase1Kernel",
        "Force4B_GpuCase1Accumulate",
    ):
        actual = name + "_Single" if name in ("Force4B_GpuCase1Kernel","Force4B_GpuCase2Kernel") else name
        parts.append(common.extract_function(source, actual).replace(actual,name))
    parts.append(common.extract_struct(source,"Force4BCapacityParameters"))
    for name in ("Force4B_capacity_kernel","Force4B_capacity_copy","Force4B_capacity"):
        parts.append(common.extract_function(source,name))
    for name in ("Force4BCase1Plan", "Force4BCase1Stream"):
        parts.append(common.extract_struct(source, name))
    parts.append("static Force4BCase1Stream F4B_stream1 = { 0 };\n")
    for name in (
        "Force4B_GpuCase1StreamEnd", "Force4B_GpuCase1SizeAdd", "Force4B_GpuCase1SizeMul",
        "Force4B_GpuCase1ArenaOff", "Force4B_GpuCase1HaloCount", "Force4B_GpuCase1Layout",
        "Force4B_GpuCase1PlanAdd", "Force4B_GpuCase1StreamBegin", "Force4B_GpuCase1StreamSource",
        "Force4B_GpuCase1StreamAccumulate", "Force4B_GpuCase1Run", "Force4B_GpuCase2Kernel",
    ):
        actual = name + "_Single" if name in ("Force4B_GpuCase1Kernel","Force4B_GpuCase2Kernel") else name
        parts.append(common.extract_function(source, actual).replace(actual,name))
    for name in ("Force4BCase2Plan", "Force4BCase2Stream"):
        parts.append(common.extract_struct(source, name))
    parts.append("static Force4BCase2Stream F4B_stream = { 0 };\n")
    for name in (
        "Force4B_GpuCase2Plan", "Force4B_GpuCase2StreamEnd", "Force4B_GpuCase2StreamBegin",
        "Force4B_GpuCase2StreamRun", "Force4B_GpuCase2StreamAccumulate", "Force4B_GpuCase2Run",
    ):
        actual = name + "_Single" if name in ("Force4B_GpuCase1Kernel","Force4B_GpuCase2Kernel") else name
        parts.append(common.extract_function(source, actual).replace(actual,name))
    # Case 2 reads CDM0 of halo atoms too (F_G2M of the case-1 fixture maps
    # them to 4..6), so every CDM slot up to Matomnum+MatomnumF exists.
    fixture = common.FIXTURE.split("/* AMD auto policy", 1)[0]
    if fixture.count("#define LOCAL_SLOTS 5") != 1:
        raise ValueError("fixture changed; update the resident harness")
    parts.append(fixture.replace("#define LOCAL_SLOTS 5", "#define LOCAL_SLOTS 7"))
    parts.append(case1.CASE1_FIXTURE.split("static void check_size_limits(void)", 1)[0])
    parts.append(RESIDENT_MAIN)
    return "\n".join(parts)


def main() -> int:
    try:
        harness = make_harness((common.ROOT / "source" / "Force.c").read_text())
        return common.build_and_run(harness, "force4b-resident-cpu")
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"Force4B resident CPU test failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

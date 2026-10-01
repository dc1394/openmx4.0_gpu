#!/usr/bin/env python3
"""CPU-only checks for bounded Force4B case-1 streaming.

Uses the production kernel, packing, planner and damping accumulation with the
same CPU/MPI/CUDA stubs as run_force4b_stream_cpu.py. The reference is the original
CPU fused trace. Exercises mixed orbital counts, periodic-image duplicates,
zero common-projector rows, reversed receive order and deliberately split batches.
This does not exercise CUDA execution or real MPI progress. CC/CFLAGS are honored.
"""

import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

import run_force4b_stream_cpu as common


GLOBALS = r'''
#include <limits.h>
static int MatomnumF = 3;
static int *F_M2G, **ncn;
static double **atv, ****HVNA;
static double deri_dampingF(double cutoff, double distance)
{
    const double denominator = 1.0 + cutoff + distance;
    return -1.0 / (denominator * denominator);
}
'''


CASE1_FIXTURE = r'''
static Type_DS_VNA***** true_ds;

static void allocate_case1_fixture(void)
{
    int mc, direction, atom, k, i, j;
    allocate_fixture();
    F_M2G = calloc(7, sizeof(int));
    F_M2G[1] = 1; F_M2G[2] = 2; F_M2G[3] = 3;
    F_M2G[4] = 0; F_M2G[5] = 4; F_M2G[6] = 5;
    for (mc = 1; mc <= 6; mc++) F_G2M[F_M2G[mc]] = mc;
    ncn = calloc(ATOMS, sizeof(*ncn));
    atv = calloc(2, sizeof(*atv));
    for (i = 0; i < 2; i++) {
        atv[i] = calloc(4, sizeof(double));
        for (j = 1; j <= 3; j++) atv[i][j] = 0.1 * i * j;
    }
    for (atom = 0; atom < ATOMS; atom++) {
        ncn[atom] = calloc(NEIGHBORS, sizeof(int));
        for (k = 0; k < NEIGHBORS; k++) ncn[atom][k] = k % 2;
        for (i = 1; i <= 3; i++) Gxyz[atom][i] = 0.17 * atom * i;
    }
    HVNA = calloc(LOCAL_SLOTS, sizeof(*HVNA));
    for (mc = 1; mc <= Matomnum; mc++) {
        /* Distinct periodic q slots can refer to the same source atom. */
        natn[mc][NEIGHBORS - 1] = natn[mc][1];
        HVNA[mc] = calloc(NEIGHBORS, sizeof(*HVNA[mc]));
        for (k = 0; k < NEIGHBORS; k++) {
            HVNA[mc][k] = calloc(ORBITALS, sizeof(*HVNA[mc][k]));
            for (i = 0; i < ORBITALS; i++) {
                HVNA[mc][k][i] = calloc(ORBITALS, sizeof(double));
                for (j = 0; j < ORBITALS; j++)
                    HVNA[mc][k][i][j] = 0.03 * cos(mc + k + i + j);
            }
        }
    }
    /* This pair has only its damping derivative contribution. */
    for (k = 0; k < NEIGHBORS; k++) RMI1[1][2][k] = -1;
    true_ds = calloc(4, sizeof(*true_ds));
    for (direction = 0; direction < 4; direction++) {
        true_ds[direction] = calloc(ATOMS, sizeof(*true_ds[direction]));
        for (atom = 0; atom < ATOMS; atom++) {
            true_ds[direction][atom] = calloc(NEIGHBORS, sizeof(*true_ds[direction][atom]));
            for (k = 0; k < NEIGHBORS; k++) {
                true_ds[direction][atom][k] = calloc(ORBITALS, sizeof(*true_ds[direction][atom][k]));
                for (i = 0; i < ORBITALS; i++)
                    true_ds[direction][atom][k][i] = calloc(PROJECTORS, sizeof(float));
            }
        }
        /* Local rows alias immutable fixture data; halo reception uses
           the separate Matomnum+1 slot allocated by allocate_fixture. */
        for (mc = 1; mc <= Matomnum; mc++)
            ds[direction][mc] = true_ds[direction][M2G[mc]];
    }
}

static void stage_source(int source)
{
    int k, i;
    if (source <= Matomnum) return;
    for (k = 0; k < NEIGHBORS; k++)
        for (i = 0; i < ORBITALS; i++)
            memcpy(ds[0][Matomnum + 1][k][i], true_ds[0][F_M2G[source]][k][i],
                PROJECTORS * sizeof(float));
}

static void free_case1_fixture(void)
{
    int direction, atom, mc, k, i;
    for (direction = 0; direction < 4; direction++) {
        for (atom = 0; atom < ATOMS; atom++) {
            for (k = 0; k < NEIGHBORS; k++) {
                for (i = 0; i < ORBITALS; i++) free(true_ds[direction][atom][k][i]);
                free(true_ds[direction][atom][k]);
            }
            free(true_ds[direction][atom]);
        }
        free(true_ds[direction]);
    }
    free(true_ds);
    for (mc = 1; mc <= Matomnum; mc++) {
        for (k = 0; k < NEIGHBORS; k++) {
            for (i = 0; i < ORBITALS; i++) free(HVNA[mc][k][i]);
            free(HVNA[mc][k]);
        }
        free(HVNA[mc]);
    }
    free(HVNA);
    for (atom = 0; atom < ATOMS; atom++) free(ncn[atom]);
    free(ncn);
    for (i = 0; i < 2; i++) free(atv[i]);
    free(atv); free(F_M2G);
    /* free_fixture releases ds pointer tables and the halo staging slot;
       the local aliased data was freed above. */
    free_fixture();
}

static void check_size_limits(void)
{
    Force4BCase1Plan plan = {0};
    size_t position = 0;
    int saved, species;

    CHECK(Force4B_GpuCase1SizeAdd(7, 9) == 16);
    CHECK(Force4B_GpuCase1SizeAdd(SIZE_MAX - 3, 4) == SIZE_MAX);
    CHECK(Force4B_GpuCase1SizeMul(7, 9) == 63);
    CHECK(Force4B_GpuCase1SizeMul(SIZE_MAX / 2, 2) == SIZE_MAX - 1);
    CHECK(Force4B_GpuCase1SizeMul(SIZE_MAX / 2 + 1, 2) == SIZE_MAX);
    CHECK(Force4B_GpuCase1SizeMul(SIZE_MAX, 0) == SIZE_MAX);
    CHECK(Force4B_GpuCase1ArenaOff(&position, 1, 1) == 0 && position == 512);
    position = SIZE_MAX - 511U;
    CHECK(Force4B_GpuCase1ArenaOff(&position, 0, 1) == SIZE_MAX - 511U);
    CHECK(position == SIZE_MAX - 511U);
    CHECK(Force4B_GpuCase1ArenaOff(&position, 1, 1) == SIZE_MAX && position == SIZE_MAX);

    plan.nitems = 1;
    plan.rows = (size_t)INT_MAX + 1U;
    Force4B_GpuCase1Layout(&plan);
    CHECK(plan.bytes == SIZE_MAX);
    memset(&plan, 0, sizeof(plan));
    plan.nitems = 1;
    plan.flat_stride = SIZE_MAX / (4U * sizeof(float)) + 1U;
    Force4B_GpuCase1Layout(&plan);
    CHECK(plan.bytes == SIZE_MAX);
    memset(&plan, 0, sizeof(plan));
    plan.nitems = 1;
    plan.halo_count = SIZE_MAX / sizeof(float) + 1U;
    Force4B_GpuCase1Layout(&plan);
    CHECK(plan.bytes == SIZE_MAX);
    memset(&plan, 0, sizeof(plan));
    plan.nitems = 1;
    plan.cdm_count = SIZE_MAX / sizeof(double) + 1U;
    Force4B_GpuCase1Layout(&plan);
    CHECK(plan.bytes == SIZE_MAX);

    CHECK(Force4B_GpuCase1StreamBegin(0, -1));
    CHECK(F4B_stream1.pair_rows[0] > 0);
    memset(&plan, 0, sizeof(plan));
    plan.nitems = INT_MAX;
    Force4B_GpuCase1PlanAdd(&plan, 0);
    CHECK(plan.bytes == SIZE_MAX);
    memset(&plan, 0, sizeof(plan));
    plan.rows = INT_MAX;
    Force4B_GpuCase1PlanAdd(&plan, 0);
    CHECK(plan.bytes == SIZE_MAX);
    memset(&plan, 0, sizeof(plan));
    plan.flat_stride = SIZE_MAX - 1U;
    Force4B_GpuCase1PlanAdd(&plan, 0);
    CHECK(plan.bytes == SIZE_MAX);
    memset(&plan, 0, sizeof(plan));
    plan.cdm_count = SIZE_MAX - 1U;
    Force4B_GpuCase1PlanAdd(&plan, 0);
    CHECK(plan.bytes == SIZE_MAX);
    species = WhatSpecies[M2G[F4B_stream1.pair_mc[0]]];
    saved = Spe_Total_CNO[species];
    Spe_Total_CNO[species] = INT_MAX;
    memset(&plan, 0, sizeof(plan));
    Force4B_GpuCase1PlanAdd(&plan, 0);
    CHECK(plan.bytes == SIZE_MAX); /* ian*jan is also int in the kernel. */
    Spe_Total_CNO[species] = saved;
    Force4B_GpuCase1StreamEnd();

    saved = FNAN[0];
    FNAN[0] = INT_MAX;
    CHECK(Force4B_GpuCase1HaloCount(0, PROJECTORS) == SIZE_MAX);
    FNAN[0] = saved;
    saved = Spe_Total_CNO[WhatSpecies[0]];
    Spe_Total_CNO[WhatSpecies[0]] = INT_MAX;
    CHECK(Force4B_GpuCase1HaloCount(0, INT_MAX) == SIZE_MAX);
    Spe_Total_CNO[WhatSpecies[0]] = saved;
    List_YOUSO[35] = INT_MAX;
    CHECK(!Force4B_GpuCase1StreamBegin(0, -1));
    List_YOUSO[35] = 1;
    List_YOUSO[34] = INT_MAX;
    CHECK(!Force4B_GpuCase1StreamBegin(0, -1));
    List_YOUSO[34] = 3;
    MatomnumF = INT_MAX;
    CHECK(!Force4B_GpuCase1StreamBegin(0, -1));
    MatomnumF = 3;
    saved = FNAN[1];
    FNAN[1] = INT_MAX;
    CHECK(!Force4B_GpuCase1StreamBegin(0, -1));
    FNAN[1] = saved;
    CHECK(device_live == 0);
}

int main(void)
{
    int chunked, spin, seed, mc, q, source, direction, atom, k, i, l, checks = 0;
    double max_error = 0.0;
    unsetenv("OPENMX_FORCE4B_GPU");
    unsetenv("OPENMX_FORCE4B_CASE1_STREAM");
    unsetenv("OPENMX_FORCE4B_CASE1_STREAM_MB");
    allocate_case1_fixture();
    check_size_limits();
    for (chunked = 0; chunked <= 1; chunked++) {
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
                CHECK(Force4B_GpuCase1StreamBegin(0, -1));
                if (chunked) {
                    size_t bound = 0, pair;
                    for (pair = 0; pair < F4B_stream1.pair_count; pair++) {
                        Force4BCase1Plan plan = {0};
                        const int gc = M2G[F4B_stream1.pair_mc[pair]];
                        const int gq = natn[gc][F4B_stream1.pair_q[pair]];
                        plan.halo_count = (size_t)(FNAN[gq] + 1)
                            * (size_t)Spe_Total_CNO[WhatSpecies[gq]] * PROJECTORS;
                        Force4B_GpuCase1PlanAdd(&plan, (int)pair);
                        if (bound < plan.bytes) bound = plan.bytes;
                    }
                    CHECK(bound < F4B_stream1.capacity);
                    F4B_stream1.capacity = bound;
                }
                /* Reverse source order tests deferred mc/q accumulation;
                   periodic duplicate slots must still be evaluated once each. */
                for (source = Matomnum + MatomnumF; source >= 1; source--) {
                    stage_source(source);
                    Force4B_GpuCase1StreamSource(source, cdm, ds);
                }
                for (mc = 1; mc <= Matomnum; mc++)
                    for (i = 0; i < 3; i++) CHECK(Gxyz[M2G[mc]][41 + i] == 0.37);
                Force4B_GpuCase1StreamAccumulate(cdm);
                for (mc = 1; mc <= Matomnum; mc++) {
                    for (i = 0; i < 3; i++) {
                        const double error = fabs(Gxyz[M2G[mc]][41 + i] - (0.37 + expected[mc][i]));
                        CHECK(isfinite(error) && error <= 1e-11);
                        if (max_error < error) max_error = error;
                    }
                    checks++;
                }
                Force4B_GpuCase1StreamEnd();
                CHECK(device_live == 0);
            }
        }
    }
    CHECK(Force4B_GpuCase1StreamMode() == -1);
    setenv("OPENMX_FORCE4B_CASE1_STREAM", "auto", 1);
    CHECK(Force4B_GpuCase1StreamMode() == -1);
    setenv("OPENMX_FORCE4B_CASE1_STREAM", "0", 1);
    CHECK(Force4B_GpuCase1StreamMode() == 0);
    setenv("OPENMX_FORCE4B_CASE1_STREAM", "1", 1);
    CHECK(Force4B_GpuCase1StreamMode() == 1);
    unsetenv("OPENMX_FORCE4B_CASE1_STREAM");
    CHECK(!Force4B_GpuCase1StreamBegin(0, 0));
    CHECK(!Force4B_GpuCase1StreamBegin(1, -1));
    setenv("OPENMX_FORCE4B_GPU", "0", 1);
    CHECK(!Force4B_GpuCase1StreamBegin(0, 1));
    unsetenv("OPENMX_FORCE4B_GPU");
    setenv("OPENMX_FORCE4B_CASE1_STREAM_MB", "0", 1);
    CHECK(!Force4B_GpuCase1StreamBegin(0, -1));
    unsetenv("OPENMX_FORCE4B_CASE1_STREAM_MB");
    allocation_fail = 1;
    CHECK(!Force4B_GpuCase1StreamBegin(0, -1));
    allocation_fail = 0;
    host_allocation_fail = 1;
    CHECK(!Force4B_GpuCase1StreamBegin(0, -1));
    host_allocation_fail = 0;
    low_memory = 1;
    CHECK(!Force4B_GpuCase1StreamBegin(0, -1));
    low_memory = 0;
    for (atom = 0; atom < ATOMS; atom++) FNAN[atom] = 0;
    CHECK(!Force4B_GpuCase1StreamBegin(0, -1));
    Matomnum = 0;
    CHECK(!Force4B_GpuCase1StreamBegin(0, -1));
    Matomnum = 3;
    Force4B_GpuCase1StreamEnd();
    CHECK(device_live == 0);
    free_case1_fixture();
    printf("PASS: %d CPU case-1 comparisons; max force error %.3g; bounded batches, "
        "periodic duplicates, empty rows/rank, size bounds, configuration and allocation fallback PASS\n", checks, max_error);
    return 0;
}
'''


def make_harness(source: str) -> str:
    parts = [common.STUBS, GLOBALS, common.extract_struct(source, "Force4BGpuItem")]
    for name in (
        "Force_gpu_arena_off", "Force4B_case1_trace_fused_rows", "Force4B_case1_trace_fused", "Force4B_GpuCase1Kernel",
        "Force4B_GpuCase1Accumulate",
    ):
        parts.append(common.extract_function(source, name))
    for name in ("Force4BCase1Plan", "Force4BCase1Stream"):
        parts.append(common.extract_struct(source, name))
    parts.append("static Force4BCase1Stream F4B_stream1 = { 0 };\n")
    for name in (
        "Force4B_GpuCase1StreamMode", "Force4B_GpuCase1StreamEnd",
        "Force4B_GpuCase1SizeAdd", "Force4B_GpuCase1SizeMul", "Force4B_GpuCase1ArenaOff",
        "Force4B_GpuCase1HaloCount", "Force4B_GpuCase1Layout",
        "Force4B_GpuCase1PlanAdd", "Force4B_GpuCase1StreamBegin", "Force4B_GpuCase1StreamSource",
        "Force4B_GpuCase1StreamAccumulate",
    ):
        parts.append(common.extract_function(source, name))
    parts.append(common.FIXTURE.split("int main(void)", 1)[0])
    parts.append(CASE1_FIXTURE)
    return "\n".join(parts)


def main() -> int:
    try:
        harness = make_harness((common.ROOT / "source" / "Force.c").read_text())
        compiler = shlex.split(os.environ.get("CC", "cc"))
        if not compiler:
            raise ValueError("CC must name a C compiler")
        flags = shlex.split(os.environ.get("CFLAGS", "-O2"))
        with tempfile.TemporaryDirectory(prefix="openmx-force4b-case1-cpu-") as directory:
            source = Path(directory) / "case1.c"
            binary = Path(directory) / "case1"
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
        print(f"Force4B case-1 CPU test failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

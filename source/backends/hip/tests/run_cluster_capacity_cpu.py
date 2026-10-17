#!/usr/bin/env python3
"""MPI check of the cluster capacity decision (source/openmx_cluster_capacity.c).

Compiles the production file with the multi-GPU runtime, the gpusolver2
bridge and HIP mocked in host memory, then runs every decision of
openmx_cluster_capacity_select / openmx_cluster_gs2_admit under mpirun with
three ranks: single/multi node, forced MAGMA, the OPENMX_CLUSTER_GS2_FALLBACK
modes, the GPU-free MAGMA limits before the host-memory test (whose reason
must win), MAGMA plan success/failure, the host-memory test (fixed through
OPENMX_CLUSTER_HOST_MEM_AVAILABLE_MB, read by the production
openmx_host_memory.c), the admission of the distributed solve with its retry
after Set_Hamiltonian's resident tables step aside, and its placement report
(nodes, rank-bound GPUs, idle permitted GPUs).
Every rank must return the same decision, including the final CPU fallback.

  python3 tests/run_cluster_capacity_cpu.py   (needs mpicc and mpirun in PATH)
"""
from pathlib import Path
import os
import subprocess
import sys
import tempfile
from mock_hip_sources import stage

root = Path(__file__).resolve().parents[1]

headers = {
"hip_runtime_compat.h": r'''
#pragma once
#include <stddef.h>
typedef int hipError_t;
enum { hipSuccess = 0 };
typedef struct { char bytes[16]; } hipUUID_t;
struct hipDeviceProp_t { hipUUID_t uuid; };
hipError_t hipGetDevice(int *device);
hipError_t hipGetDeviceCount(int *count);
hipError_t hipGetDeviceProperties(struct hipDeviceProp_t *prop, int device);
hipError_t hipGetLastError(void);
''',
"hip_runtime_api.h": '#pragma once\n#include "hip_runtime_compat.h"\n',
"hipblas_compat.h": r'''
#pragma once
typedef void *hipblasHandle_t; typedef int hipblasOperation_t; typedef int hipblasStatus_t;
typedef int hipblasFillMode_t; typedef struct { double x, y; } hipDoubleComplex;
''',
"hipsolver_compat.h": r'''
#pragma once
#include <stdint.h>
typedef void *hipsolverDnHandle_t; typedef int hipsolverEigMode_t; typedef int hipsolverStatus_t;
typedef int hipDataType;
''',
}

mocks = r'''
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "hip_runtime_compat.h"
#include "openmx_cluster_capacity.h"

int level_stdout = 1;
static int env(const char *name) { const char *v = getenv(name); return v ? atoi(v) : 0; }
static int plan_calls = 0, precheck_calls = 0, report_on = 0, yield_calls = 0, yielded = 0;
static char host_error[256];

int openmx_mgpu_host_required(void) { return env("T_FORCED"); }
int openmx_mgpu_multi_node(void) { return env("T_MULTI"); }
int openmx_mgpu_gpu_limit(void) { return env("T_LIMIT"); }
/* T_PRE: 1 (default) passes, 0 fails on the pool, 2 on the LP64 order */
int openmx_mgpu_host_precheck(int kp, int n, int z) {
    (void)kp; (void)z; ++precheck_calls;
    const char *v = getenv("T_PRE");
    int pre = v ? atoi(v) : 1;
    if (pre == 0) {
        snprintf(host_error, sizeof(host_error), "MAGMA needs 2..8 GPUs per matrix: available pool=1, n=%d; GPU limit will not be exceeded", n);
        return 0;
    }
    if (pre == 2) {
        snprintf(host_error, sizeof(host_error), "unsupported MAGMA order/group: n=%d, gpus=2 (LP64 order limit 32766)", n);
        return 0;
    }
    return 2;
}
/* T_RESIDENT: rank 1 holds resident tables; once every rank yielded, the
   distributed solve fits */
size_t Set_Hamiltonian_DemoteResidency(void) {
    int r; MPI_Comm_rank(MPI_COMM_WORLD, &r);
    ++yield_calls; yielded = 1;
    return env("T_RESIDENT") && r == 1;
}
int openmx_mgpu_host_plan(int kp, int n, int z) {
    (void)kp; (void)z; ++plan_calls;
    if (env("T_PLAN")) return 2;
    snprintf(host_error, sizeof(host_error), "MAGMA needs 2..8 GPUs per matrix: available pool=1, n=%d; GPU limit will not be exceeded", n);
    return 0;
}
const char *openmx_mgpu_host_error(void) { return host_error; }
int openmx_gs2_gpu_fits(int ictxt, int n, int nblk, int z, unsigned long long *need,
                        unsigned long long *freem, int *ranks) {
    (void)ictxt; (void)n; (void)nblk; (void)z;
    int r; MPI_Comm_rank(MPI_COMM_WORLD, &r);
    *need = 1000 + (unsigned long long)r; *freem = 5000 - (unsigned long long)r; *ranks = 2;
    /* the bridge's verdict is collective: every rank of a grid agrees */
    if (env("T_RESIDENT") && yielded) return 1;
    return env("T_FITS") == 2 ? -1 : env("T_FITS");
}
void openmx_gs2_set_fallback_report(int on) { report_on = on; }
/* the three ranks share this host (one node) and are bound to GPUs 0,1,0 */
hipError_t hipGetDevice(int *d) { int r; MPI_Comm_rank(MPI_COMM_WORLD, &r); *d = r % 2; return hipSuccess; }
hipError_t hipGetDeviceCount(int *c) { *c = env("T_VISIBLE") ? env("T_VISIBLE") : 2; return hipSuccess; }
hipError_t hipGetDeviceProperties(struct hipDeviceProp_t *p, int d) {
    memset(p, 0, sizeof(*p)); p->uuid.bytes[0] = (char)(d + 1);
    return hipSuccess;
}
hipError_t hipGetLastError(void) { return hipSuccess; }
int openmx_gpu_physical_uuid(int device,unsigned char uuid[16]){memset(uuid,0,16);uuid[0]=device+1;return 1;}

int main(int argc, char **argv) {
    int myid, decision, all_min, all_max;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    decision = openmx_cluster_capacity_select(MPI_COMM_WORLD, env("T_N"), env("T_COMPLEX"), "Cluster test");
    if (decision == OPENMX_CLUSTER_CAPACITY_GS2)
        if (!openmx_cluster_gs2_admit(MPI_COMM_WORLD, 0, env("T_N"), 64, env("T_COMPLEX"), "Cluster test"))
            decision = OPENMX_CLUSTER_CAPACITY_CPU;
    MPI_Allreduce(&decision, &all_min, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&decision, &all_max, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (myid == 0)
        printf("RESULT decision=%d agree=%d plan_calls=%d report=%d precheck_calls=%d yields=%d\n",
               decision, all_min == all_max, plan_calls, report_on, precheck_calls, yield_calls);
    MPI_Finalize();
    return 0;
}
'''

# (name, env, expected decision or None for an abort, expected text, plan calls).
# The host memory is fixed by OPENMX_CLUSTER_HOST_MEM_AVAILABLE_MB (1 TiB
# unless a case sets it), so no case depends on the test machine.
cases = [
    ("single-auto-magma",  dict(T_MULTI=0, T_PLAN=1), 1, "", 1),
    ("single-auto-cpu",    dict(T_MULTI=0, T_PLAN=0), 0, "GPU limit will not be exceeded", 1),
    ("multi-auto-magma",   dict(T_MULTI=1, T_PLAN=1, T_N=13000), 1, "", 1),
    ("multi-auto-gs2",     dict(T_MULTI=1, T_PLAN=0, T_FITS=1), 2,
     "distributed gpusolver2 (ELPA GPU kernels + COSMA) on 1 node(s), 2 GPU(s), 3 rank(s).", 1),
    ("multi-hostmem-gs2",  dict(T_MULTI=1, T_PLAN=1, T_FITS=1, T_N=26000, T_COMPLEX=1,
                                OPENMX_CLUSTER_HOST_MEM_AVAILABLE_MB=60000), 2, "of host memory on rank 0's node", 0),
    ("multi-hostmem-magma", dict(T_MULTI=1, T_PLAN=1, T_N=26000, T_COMPLEX=1,
                                 OPENMX_CLUSTER_HOST_MEM_AVAILABLE_MB=100000), 1, "", 1),
    ("multi-idle-warning", dict(T_MULTI=1, T_PLAN=0, T_FITS=1, T_VISIBLE=4), 2,
     "Warning: 2 permitted GPU(s) have no MPI rank", 1),
    ("multi-idle-limit",   dict(T_MULTI=1, T_PLAN=0, T_FITS=1, T_VISIBLE=4, T_LIMIT=2), 2,
     "on 1 node(s), 2 GPU(s), 3 rank(s).", 1),
    ("multi-idle-cpu",     dict(T_MULTI=1, T_PLAN=0, T_FITS=0, T_VISIBLE=4), 0,
     "2 permitted GPU(s) have no MPI rank: run at least one rank per GPU", 1),
    ("multi-forced-cpu",   dict(T_MULTI=1, T_PLAN=0, T_FORCED=1), 0, "using the CPU eigensolver", 1),
    ("multi-forced-magma", dict(T_MULTI=1, T_PLAN=1, T_FORCED=1, OPENMX_CLUSTER_GS2_FALLBACK="force"), 1, "", 1),
    ("multi-off-cpu",      dict(T_MULTI=1, T_PLAN=0, OPENMX_CLUSTER_GS2_FALLBACK="off"), 0, "using the CPU eigensolver", 1),
    ("single-on-gs2",      dict(T_MULTI=0, T_PLAN=0, T_FITS=1, OPENMX_CLUSTER_GS2_FALLBACK="on"), 2, "pool=1", 1),
    ("single-force-gs2",   dict(T_MULTI=0, T_PLAN=1, T_FITS=1, OPENMX_CLUSTER_GS2_FALLBACK="force"), 2, "MAGMA skipped", 0),
    ("multi-gs2-nofit",    dict(T_MULTI=1, T_PLAN=0, T_FITS=0), 0, "fits neither one GPU, MAGMA nor the distributed", 1),
    ("multi-gs2-forcedkernels", dict(T_MULTI=1, T_PLAN=0, T_FITS=2), 2, "kernels set by OPENMX_GS2_ELPA_GPU", 1),
    ("single-hostmem-cpu", dict(T_MULTI=0, T_PLAN=1, T_N=26000, T_COMPLEX=1,
                                OPENMX_CLUSTER_HOST_MEM_AVAILABLE_MB=1000), 0, "of host memory on rank 0's node", 0),
    ("forced-hostmem-cpu", dict(T_MULTI=1, T_PLAN=1, T_FORCED=1, T_N=26000, T_COMPLEX=1,
                                OPENMX_CLUSTER_HOST_MEM_AVAILABLE_MB=1000), 0, "using the CPU eigensolver", 0),
    # the GPU-free limits come before the host-memory test: their reason wins
    ("multi-precheck-pool", dict(T_MULTI=1, T_PRE=0, T_PLAN=1, T_FITS=1, T_N=26000, T_COMPLEX=1,
                                 OPENMX_CLUSTER_HOST_MEM_AVAILABLE_MB=60000), 2,
     "hipSOLVER memory limit -> MAGMA needs 2..8 GPUs per matrix: available pool=1", 0, "of host memory"),
    ("multi-precheck-order", dict(T_MULTI=1, T_PRE=2, T_PLAN=1, T_FITS=1, T_N=37440, T_COMPLEX=1,
                                  OPENMX_CLUSTER_HOST_MEM_AVAILABLE_MB=60000), 2,
     "unsupported MAGMA order/group: n=37440", 0, "of host memory"),
    ("single-precheck-cpu", dict(T_MULTI=0, T_PRE=0, T_PLAN=1), 0, "pool=1", 0),
    # the distributed solve is admitted again once the resident tables yield
    ("multi-gs2-yield",     dict(T_MULTI=1, T_PLAN=0, T_FITS=0, T_RESIDENT=1), 2,
     "distributed gpusolver2 (ELPA GPU kernels + COSMA) on 1 node(s), 2 GPU(s), 3 rank(s).", 1, "fits neither"),
    ("multi-gs2-fits-noyield", dict(T_MULTI=1, T_PLAN=0, T_FITS=1, T_RESIDENT=1), 2, "yields=0", 1),
]

failures = 0
with tempfile.TemporaryDirectory(prefix="openmx-cluster-capacity-") as folder:
    tmp = Path(folder)
    for name, text in headers.items():
        (tmp / name).write_text(text)
    (tmp / "openmx_stdout_level.h").write_text(
        '#pragma once\nextern int level_stdout;\n#define OPENMX_STDOUT_LEVEL(l) ((l) <= level_stdout)\n')
    (tmp / "test.c").write_text(mocks)
    stage(root,tmp,"openmx_cluster_capacity.c","openmx_host_memory.c")
    exe = tmp / "test"
    # the mock headers come first; the production headers come from source/
    cmd = ["mpicc", "-std=gnu11", "-O1", "-Wall", "-I" + str(tmp), "-I" + str(root / "source"),
           str(tmp / "openmx_cluster_capacity.c"), str(tmp / "openmx_host_memory.c"),
           str(tmp / "test.c"), "-o", str(exe)]
    subprocess.run(cmd, check=True)
    for name, settings, want, text, plans, *absent in cases:
        run_env = dict(os.environ)
        run_env.pop("OPENMX_CLUSTER_GS2_FALLBACK", None)
        run_env.update({"T_N": "26000", "T_COMPLEX": "0", "OPENMX_CLUSTER_HOST_MEM_AVAILABLE_MB": "1048576"})
        run_env.update({k: str(v) for k, v in settings.items()})
        launch = ["mpirun", "--oversubscribe", "-np", "3"]
        for key in ("T_N", "T_COMPLEX", "T_MULTI", "T_PLAN", "T_FORCED", "T_FITS", "T_VISIBLE", "T_LIMIT",
                    "T_PRE", "T_RESIDENT", "OPENMX_CLUSTER_GS2_FALLBACK", "OPENMX_CLUSTER_HOST_MEM_AVAILABLE_MB"):
            if key in run_env:
                launch += ["-x", key]
        proc = subprocess.run(launch + [str(exe)], env=run_env, capture_output=True, text=True, timeout=120)
        log = proc.stdout + proc.stderr
        ok = True
        if want is None:
            ok = proc.returncode != 0 and "RESULT" not in log
        else:
            line = next((l for l in log.splitlines() if l.startswith("RESULT")), "")
            ok = (proc.returncode == 0 and f"decision={want} " in line and "agree=1" in line
                  and f"plan_calls={plans} " in line and ("report=1" in line) == (want == 2))
        if text and text not in log:
            ok = False
        if absent and absent[0] in log:
            ok = False
        if name == "multi-gs2-yield" and "yields=1" not in log:
            ok = False
        print(f"{'PASS' if ok else 'FAIL'}: {name}")
        if not ok:
            failures += 1
            print(log)
print(f"{len(cases) - failures}/{len(cases)} cluster capacity decisions passed")
sys.exit(1 if failures else 0)

/**********************************************************************
  openmx_cluster_capacity.c

  Capacity decision of the cluster solvers for scf.eigen.lib=gpusolver;
  see openmx_cluster_capacity.h.  The MAGMA plan and the host-memory test
  run on rank 0 only (the MAGMA path keeps its dense matrices on rank 0's
  node) and the decision is broadcast before any rank commits to a path,
  since the MAGMA path and the distributed path have different collective
  sequences.
***********************************************************************/

#include "openmx_cluster_capacity.h"
#include "openmx_mgpu_runtime.h"
#include "elpa_cosma_bridge.h"
#include "openmx_stdout_level.h"
#include "openmx_host_memory.h"

#include "hip_runtime_compat.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

enum { FALLBACK_OFF = 0, FALLBACK_AUTO = 1, FALLBACK_ON = 2, FALLBACK_FORCE = 3 };

/* why MAGMA was not taken (rank 0) */
static char capacity_reason[768];

static int fallback_mode(void)
{
    const char *v = getenv("OPENMX_CLUSTER_GS2_FALLBACK");

    if (v == NULL || v[0] == '\0' || strcasecmp(v, "auto") == 0) return FALLBACK_AUTO;
    if (strcmp(v, "0") == 0 || strcasecmp(v, "off") == 0) return FALLBACK_OFF;
    if (strcmp(v, "1") == 0 || strcasecmp(v, "on") == 0) return FALLBACK_ON;
    if (strcasecmp(v, "force") == 0) return FALLBACK_FORCE;
    fprintf(stderr, "Warning: OPENMX_CLUSTER_GS2_FALLBACK=%s is not auto, off, on or force;"
                    " using auto.\n", v);
    return FALLBACK_AUTO;
}

/* host memory available on rank 0's node for the MAGMA test, in bytes (0:
   unknown); OPENMX_CLUSTER_HOST_MEM_AVAILABLE_MB overrides the measurement */
static unsigned long long magma_host_available(void)
{
    const char *override = getenv("OPENMX_CLUSTER_HOST_MEM_AVAILABLE_MB");

    if (override != NULL && override[0] != '\0') {
        return strtoull(override, NULL, 10) * 1048576ULL;
    }
    return openmx_host_mem_available();
}

/* Host memory the MAGMA path adds on rank 0's node over the pre-solve
   baseline, with a margin. In the NVIDIA four-GPU Si1000 capacity runs
   used to size this upstream estimate: about 10-12 GiB for the collinear
   order 13000 (~70 n^2 bytes) and 70-73 GiB for the noncollinear spinor
   order 26000 (~115 n^2 bytes). */
static unsigned long long magma_host_bytes(int n, int complex_matrix)
{
    const unsigned long long nn = (unsigned long long)n * (unsigned long long)n;

    return (complex_matrix ? 128ULL : 80ULL) * nn + (8ULL << 30);
}

int openmx_cluster_capacity_select(MPI_Comm comm, int n, int complex_matrix,
                                   const char *label)
{
    int myid = 0;
    int decision = 0;

    MPI_Comm_rank(comm, &myid);

    if (myid == 0) {
        const int forced = openmx_mgpu_host_required();
        const int multi = openmx_mgpu_multi_node();
        const int mode = fallback_mode();
        const int gs2 = !forced && (mode == FALLBACK_FORCE || mode == FALLBACK_ON ||
                                    (mode == FALLBACK_AUTO && multi));
        int magma = 0;

        capacity_reason[0] = '\0';
        if (forced || mode != FALLBACK_FORCE) {
            /* the limits that need no GPU first (group size within the pool,
               LP64 order): when they rule MAGMA out, theirs is the reason */
            int ok = openmx_mgpu_host_precheck(0, n, complex_matrix);

            if (!ok) {
                snprintf(capacity_reason, sizeof(capacity_reason), "%s",
                         openmx_mgpu_host_error());
            }
            if (ok) {
                /* CPU distribution is also available on a single node.
                   Never try a root-only MAGMA allocation that exceeds the
                   host's remaining memory just because there is no GPU tier. */
                const unsigned long long need = magma_host_bytes(n, complex_matrix);
                const unsigned long long avail = magma_host_available();

                if (avail != 0 && avail < need) {
                    ok = 0;
                    snprintf(capacity_reason, sizeof(capacity_reason),
                             "MAGMA n=%d type=%c needs ~%.1f GiB of host memory on rank 0's node,"
                             " %.1f GiB available",
                             n, complex_matrix ? 'Z' : 'D',
                             (double)need / 1073741824.0, (double)avail / 1073741824.0);
                }
            }
            if (ok) {
                magma = openmx_mgpu_host_plan(0, n, complex_matrix);
                if (!magma) {
                    snprintf(capacity_reason, sizeof(capacity_reason), "%s",
                             openmx_mgpu_host_error());
                }
            }
        }
        else {
            snprintf(capacity_reason, sizeof(capacity_reason),
                     "MAGMA skipped (OPENMX_CLUSTER_GS2_FALLBACK=force)");
        }

        if (magma) {
            decision = OPENMX_CLUSTER_CAPACITY_MAGMA;
        }
        else if (gs2) {
            decision = OPENMX_CLUSTER_CAPACITY_GS2;
        }
        else {
            decision = OPENMX_CLUSTER_CAPACITY_CPU;
            fprintf(stderr, "<OpenMX_MGPU> %s n=%d: %s; using the CPU eigensolver and"
                            " matrix products within the existing MPI layout.\n",
                    label, n, capacity_reason);
            fflush(stderr);
        }
    }

    MPI_Bcast(&decision, 1, MPI_INT, 0, comm);
    return decision;
}

/* distinct nodes and GPUs among the ranks of comm, and the permitted GPUs
   (visible, capped by scf.Gpu.Num) of their nodes that no rank is bound to:
   ELPA and COSMA use the GPU of each rank */
/* Set_Hamiltonian.c: release the resident matrix-elements tables of this
   rank and keep them transient for the rest of the MD step (1: there were
   some) */
extern size_t Set_Hamiltonian_DemoteResidency(void);

#define PLACEMENT_INTS 6
static void count_placement(MPI_Comm comm, int *nodes, int *gpus, int *idle)
{
    int myid, numprocs, i, j, mine[PLACEMENT_INTS], node_key;
    int *all;
    MPI_Comm node_comm;

    MPI_Comm_rank(comm, &myid);
    MPI_Comm_size(comm, &numprocs);

    node_key = myid;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node_comm);
    MPI_Bcast(&node_key, 1, MPI_INT, 0, node_comm);
    MPI_Comm_free(&node_comm);

    memset(mine, 0, sizeof(mine));
    mine[0] = node_key;
    {
        int dev = -1, count = 0;
        unsigned char uuid[16];
        const int limit = openmx_mgpu_gpu_limit();

        if (hipGetDevice(&dev) == hipSuccess &&
            openmx_gpu_physical_uuid(dev, uuid)) {
            memcpy(&mine[1], uuid, 16);
        }
        else {
            (void)hipGetLastError();
            mine[1] = -1 - node_key; /* one anonymous device per node */
        }
        if (hipGetDeviceCount(&count) != hipSuccess) {
            (void)hipGetLastError();
            count = 1;
        }
        if (0 < limit && limit < count) count = limit;
        mine[5] = count;
    }

    *nodes = 0;
    *gpus = 0;
    *idle = 0;
    all = (int *)malloc(sizeof(int) * PLACEMENT_INTS * (size_t)numprocs);
    if (all == NULL) {
        fprintf(stderr, "openmx_cluster_capacity: out of host memory (%d ranks)\n", numprocs);
        MPI_Abort(comm, 1);
    }
    MPI_Allgather(mine, PLACEMENT_INTS, MPI_INT, all, PLACEMENT_INTS, MPI_INT, comm);
    for (i = 0; i < numprocs; i++) {
        const int *ri = &all[PLACEMENT_INTS * i];
        int new_node = 1, new_gpu = 1;
        for (j = 0; j < i; j++) {
            const int *rj = &all[PLACEMENT_INTS * j];
            if (rj[0] == ri[0]) new_node = 0;
            if (memcmp(&rj[1], &ri[1], 4 * sizeof(int)) == 0) new_gpu = 0;
        }
        *gpus += new_gpu;
        if (new_node) {
            /* the first rank of a node: its permitted GPUs minus the
               distinct GPUs its ranks are bound to */
            int permitted = 0, bound = 0, k, m;
            *nodes += 1;
            for (k = i; k < numprocs; k++) {
                const int *rk = &all[PLACEMENT_INTS * k];
                int seen = 0;
                if (rk[0] != ri[0]) continue;
                if (permitted < rk[5]) permitted = rk[5];
                for (m = i; m < k; m++) {
                    const int *rm = &all[PLACEMENT_INTS * m];
                    if (rm[0] == ri[0] && memcmp(&rm[1], &rk[1], 4 * sizeof(int)) == 0) seen = 1;
                }
                if (!seen) bound++;
            }
            if (bound < permitted) *idle += permitted - bound;
        }
    }
    free(all);
}

int openmx_cluster_gs2_admit(MPI_Comm comm, int ictxt, int n, int nblk,
                              int complex_matrix, const char *label)
{
    unsigned long long need_mb = 0, free_mb = 0, worst[2], least_free;
    int ranks_on_dev = 0;
    int fits, admitted, all_admitted, forced, any_forced;
    int myid, numprocs, nodes = 0, gpus = 0, idle = 0;

    MPI_Comm_rank(comm, &myid);
    MPI_Comm_size(comm, &numprocs);

    /* collective over this rank's grid, then over comm (the collinear spin
       worlds run their grids side by side) */
    fits = openmx_gs2_gpu_fits(ictxt, n, nblk, complex_matrix, &need_mb, &free_mb, &ranks_on_dev);
    admitted = (fits != 0);
    forced = (fits < 0);
    MPI_Allreduce(&admitted, &all_admitted, 1, MPI_INT, MPI_MIN, comm);
    MPI_Allreduce(&forced, &any_forced, 1, MPI_INT, MPI_MAX, comm);
    if (!all_admitted) {
        /* Set_Hamiltonian's resident tables (kept when every rank of a GPU
           builds in one wave, up to ~20 GiB per GPU) are the tenant that
           can step aside: every rank releases its own, and the solve is
           admitted again on the memory they leave */
        int resident = (Set_Hamiltonian_DemoteResidency()!=0), any_resident = 0;
        MPI_Allreduce(&resident, &any_resident, 1, MPI_INT, MPI_MAX, comm);
        if (any_resident) {
            if (myid == 0 && OPENMX_STDOUT_LEVEL(2)) {
                printf("<OpenMX_MGPU> %s n=%d: Set_Hamiltonian's resident GPU tables released for the"
                       " distributed solve (transient for this MD step)\n", label, n);
                fflush(stdout);
            }
            fits = openmx_gs2_gpu_fits(ictxt, n, nblk, complex_matrix, &need_mb, &free_mb,
                                       &ranks_on_dev);
            admitted = (fits != 0);
            forced = (fits < 0);
            MPI_Allreduce(&admitted, &all_admitted, 1, MPI_INT, MPI_MIN, comm);
            MPI_Allreduce(&forced, &any_forced, 1, MPI_INT, MPI_MAX, comm);
        }
    }
    worst[0] = need_mb;
    worst[1] = (unsigned long long)ranks_on_dev;
    MPI_Allreduce(MPI_IN_PLACE, worst, 2, MPI_UNSIGNED_LONG_LONG, MPI_MAX, comm);
    least_free = free_mb;
    MPI_Allreduce(MPI_IN_PLACE, &least_free, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN, comm);
    count_placement(comm, &nodes, &gpus, &idle);

    if (myid == 0) {
        if (all_admitted) {
            if (OPENMX_STDOUT_LEVEL(1)) {
                printf("<OpenMX_MGPU> %s n=%d: hipSOLVER memory limit -> %s -> distributed gpusolver2"
                       " (ELPA %s + COSMA) on %d node(s), %d GPU(s), %d rank(s).\n",
                       label, n, capacity_reason,
                       any_forced ? "kernels set by OPENMX_GS2_ELPA_GPU" : "GPU kernels",
                       nodes, gpus, numprocs);
                if (idle > 0) {
                    printf("<OpenMX_MGPU> Warning: %d permitted GPU(s) have no MPI rank and stay idle"
                           " in the distributed solve (ELPA and COSMA use the GPU of each rank); run at"
                           " least one rank per GPU.\n", idle);
                }
                fflush(stdout);
            }
        }
        else {
            char hint[160];
            if (idle > 0) {
                snprintf(hint, sizeof(hint), "%d permitted GPU(s) have no MPI rank: run at least one"
                         " rank per GPU", idle);
            }
            else {
                snprintf(hint, sizeof(hint), "more GPUs or nodes, or OPENMX_GS2_ELPA_GPU=0 for ELPA's"
                         " CPU kernels");
            }
            fprintf(stderr, "%s\n", capacity_reason);
            fprintf(stderr, "%s n=%d: the dense solve fits neither one GPU, MAGMA nor the distributed"
                            " gpusolver2 solve on the %d rank-bound GPU(s) of %d node(s) of this run:"
                            " ELPA/COSMA need ~%llu MB per rank x %llu ranks per GPU + reserve, %llu MB"
                            " free (%s); using the CPU eigensolver and matrix products.\n",
                    label, n, gpus, nodes, worst[0], worst[1], least_free, hint);
            fflush(stderr);
        }
    }
    openmx_gs2_set_fallback_report(all_admitted);
    return all_admitted;
}

/**********************************************************************
  openmx_cluster_capacity.h

  What the cluster solvers (Cluster_DFT_Col.c, Cluster_DFT_NonCol.c) do
  with scf.eigen.lib=gpusolver once their collective native verdict
  (*_GpuDiagFits) finds that the dense solve does not fit on one GPU:

    1. host-backed multi-GPU MAGMA on the GPUs of rank 0's node, when that
       node has two or more permitted GPUs, the order is within MAGMA's
       LP64 limit, every GPU of the group admits MAGMA's scratch and, when
       step 2 is possible, rank 0's node has the host memory for it;
    2. otherwise, in a run that spans several nodes, the distributed
       gpusolver2 path (ELPA GPU kernels + COSMA) on a BLACS grid over every
       rank of the run (a spin-polarized collinear run solves its two spins
       in turn on such a grid), i.e. both nodes of a two-node job.  ELPA
       and COSMA use the GPU each rank is bound to, so with at least one
       rank per GPU: 1 GPU per node gives 2 GPUs, 2 GPUs per node 4 GPUs;
    3. otherwise the distributed CPU eigensolver and matrix products.

  The host-memory test uses MemAvailable bounded by the cgroup limits of
  rank 0; OPENMX_CLUSTER_HOST_MEM_AVAILABLE_MB overrides it.

  OPENMX_CLUSTER_GS2_FALLBACK (read on rank 0 only):
    unset/auto  as above (step 2 only in multi-node runs)
    0/off       never step 2 (use the CPU when MAGMA cannot take it)
    1/on        step 2 also in a single-node run
    force       skip step 1 and go straight to step 2 (tests on inputs
                that MAGMA could take)
  Forced MAGMA (scf.eigen.lib=gpusolver3 or OPENMX_MGPU_MEMORY=host)
  never takes step 2.
***********************************************************************/
#ifndef OPENMX_CLUSTER_CAPACITY_H
#define OPENMX_CLUSTER_CAPACITY_H

#include <mpi.h>

#define OPENMX_CLUSTER_CAPACITY_CPU   0
#define OPENMX_CLUSTER_CAPACITY_MAGMA 1
#define OPENMX_CLUSTER_CAPACITY_GS2   2

/* Collective over comm, called by every rank after the native buffers were
   released: rank 0 decides and broadcasts. Returns CPU, MAGMA or GS2 on
   every rank. n is the order of the dense
   problem (2n, the spinor order, for the noncollinear solver). */
int openmx_cluster_capacity_select(MPI_Comm comm, int n, int complex_matrix,
                                   const char *label);

/* Collective over comm, after a GS2 decision: admits the distributed solve
   of order n on this rank's BLACS grid (ictxt, nblk; each collinear spin
   world passes its own grid), reports the placement (nodes, GPUs, ranks)
   at level.of.stdout 1 and turns on the bridge's fallback reporting.
   Returns 1 on admission and 0 on every rank when the caller must use its
   CPU path. OPENMX_GS2_ELPA_GPU explicitly fixes the bridge's kernel side. */
int openmx_cluster_gs2_admit(MPI_Comm comm, int ictxt, int n, int nblk,
                              int complex_matrix, const char *label);

#endif /* OPENMX_CLUSTER_CAPACITY_H */

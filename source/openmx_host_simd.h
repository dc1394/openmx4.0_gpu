#ifndef OPENMX_HOST_SIMD_H
#define OPENMX_HOST_SIMD_H
#include "openmx_core_host_simd.h"
#ifdef __cplusplus
extern "C" {
#endif
/* The OPENMX_HOST_SIMD policy of this rank for the Core's host kernel
   families: OPENMX_HOST_SIMD_LEGACY (the C loops), or an OPENMX_GRID_* mode
   that this processor executes. The registry has already synchronised the
   value from rank 0, so every rank resolves the same ISA on a homogeneous
   cluster; the resolution is a few string compares and CPUID bit tests. */
int OpenMX_HostSimdMode(void);
/* Rank 0 names the resolved policy once per call site at level.of.stdout >= 2
   ("<label> host SIMD = AVX-512"). */
void OpenMX_HostSimdReport(const char *label);
/* The grid kernels of Set_Hamiltonian / Set_Density_Grid keep their own knobs
   (OPENMX_SETHAM_SIMD, OPENMX_DENSITY_GRID_COL_SIMD, _NC_SIMD, registry id
   env_id); when the knob is unset they follow OPENMX_HOST_SIMD, legacy meaning
   the scalar Core kernel (those paths have no C loop any more). */
int OpenMX_HostSimdGridMode(int env_id);

/* Device-or-host decision of one GPU stage, collective over mpi_comm_level1:
   every rank calls it at the same point with its own figures and all ranks
   get the same answer. 1 means: run the whole stage on the host SIMD paths.
   A rank votes for the host when its device is unusable, when need_bytes
   (the stage's device working set; SIZE_MAX = unknown/never fits) exceeds
   budget_bytes (what the stage may use on this rank's device), or when
   fitting it would split the work into more than OPENMX_HOST_SIMD_MAX_PIECES
   device pieces (default 8): small pieces are dominated by transfers and by
   the collective steps of the capacity recovery, where the host SIMD kernels
   finish first. One host vote moves every rank (the ranks share devices and
   the recovery loops are collective). Under OPENMX_HOST_SIMD=legacy only an
   unusable device votes for the host, which keeps the earlier behaviour.
   Rank 0 reports host decisions at level.of.stdout >= 2. */
int OpenMX_HostSimdStageOnHost(const char *stage, size_t need_bytes, size_t budget_bytes);
/* Stages named in OPENMX_HOST_SIMD_PREFER (comma-separated, case-insensitive
   substrings of the stage label such as "UCell_Box", "Set_ProExpn_VNA",
   "Set_Hamiltonian", "Force3"; "all" = every stage) vote for the host even
   when the device holds them: on hosts whose SIMD paths beat the device for a
   stage (e.g. the overlap-grid search and the projector expansion on an
   RTX 5080 with an 18-core AVX-512 host). Empty by default; ignored under
   OPENMX_HOST_SIMD=legacy. The value is synchronised from rank 0 like every
   registry entry, so the vote stays collective-consistent. */
int OpenMX_HostSimdPreferred(const char *stage);
/* This rank's device budget: free memory (OpenMX_GpuMemGetInfo) minus
   reserve_bytes, or 0 when the device is unusable or the query fails. */
size_t OpenMX_HostSimdDeviceBudget(size_t reserve_bytes);
#ifdef __cplusplus
}
#endif
#endif

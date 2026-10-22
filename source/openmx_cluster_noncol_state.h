#ifndef OPENMX_CLUSTER_NONCOL_STATE_H
#define OPENMX_CLUSTER_NONCOL_STATE_H
#include "openmx_core.h"
#include "openmx_complex.h"
#include "openmx_density_cache_hooks.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Borrowed calculation state, shared only by that calculation's SCF steps.
   Clear/finish drains native resources and invalidates the view before MPI
   finalization. Binding and teardown require quiescent users. */
typedef struct OpenMXClusterNonColState OpenMXClusterNonColState;
OpenMXClusterNonColState *Cluster_DFT_NonCol_Bind_Resources(OpenMXCoreContext*);
void Cluster_DFT_NonCol_Release_GPU_Solver(OpenMXClusterNonColState*);
int Cluster_DFT_NonCol_RefineLastRefined(OpenMXClusterNonColState*);
void Cluster_DFT_NonCol_RefineAfterMixing(OpenMXClusterNonColState*, double normrd);
void Cluster_DFT_NonCol_DemoteGpuSolverCachedEVec(OpenMXClusterNonColState*);
double Cluster_DFT_NonCol_ScatterGpuSolverCachedEVec(OpenMXClusterNonColState*,
    int n2, int *is2, int *ie2, dcomplex *eigenvectors);
int Cluster_DFT_NonCol_MgpuCachedEDM(OpenMXClusterNonColState*, int n, int size_H1,
    int *MP, double *eigenvalues, double *****density, double *elapsed);
#ifdef __cplusplus
}
#endif
static inline void OpenMX_ClusterNonCol_Demote(void *state)
{
  Cluster_DFT_NonCol_DemoteGpuSolverCachedEVec((OpenMXClusterNonColState *)state);
}
static inline OpenMXDensityCacheHooks OpenMX_ClusterNonCol_DensityHooks(OpenMXClusterNonColState *state)
{
  OpenMXDensityCacheHooks hooks={state,OpenMX_ClusterNonCol_Demote};
  return hooks;
}
#endif

#ifndef OPENMX_CLUSTER_COL_STATE_H
#define OPENMX_CLUSTER_COL_STATE_H
#include "openmx_core.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Borrowed from an explicit calculation; never store this in a global/TLS.
   SCF steps reuse the same state. Finish/clear invalidates it after draining
   its native resources, before MPI_Finalize. Binding requires quiescent users. */
typedef struct OpenMXClusterColState OpenMXClusterColState;
OpenMXClusterColState *Cluster_DFT_Col_Bind_Resources(OpenMXCoreContext*);
void Cluster_DFT_Col_Release_GPU_Solver(OpenMXClusterColState*);
int Cluster_DFT_Col_RefineLastRefined(OpenMXClusterColState*);
void Cluster_DFT_Col_RefineForceFP64(OpenMXClusterColState*, int on);
void Cluster_DFT_Col_RefineAfterMixing(OpenMXClusterColState*, double normrd);
#ifdef __cplusplus
}
#endif
#endif

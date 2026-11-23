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
/* The exact exchange of a hybrid functional for the solves that follow, or
   NULL without one. The solver adds the exchange matrix to its dense
   Hamiltonian and hands the vectors and occupations of a collection over
   (openmx_exact_exchange.h). The state is borrowed from the same calculation. */
struct OpenMXExactExchangeState;
void Cluster_DFT_Col_ExactExchange(OpenMXClusterColState*, struct OpenMXExactExchangeState*);
#ifdef __cplusplus
}
#endif
#endif

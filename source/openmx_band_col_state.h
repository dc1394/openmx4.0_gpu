#ifndef OPENMX_BAND_COL_STATE_H
#define OPENMX_BAND_COL_STATE_H
#include "openmx_core.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Borrowed calculation metadata. Binding/clear/finish require quiescent
   users; clear/finish invalidates the view before MPI finalization. */
typedef struct OpenMXBandColState OpenMXBandColState;
OpenMXBandColState *Band_DFT_Col_Bind_Resources(OpenMXCoreContext *core);
int Band_DFT_Col_RefineLastRefined(OpenMXBandColState *state);
void Band_DFT_Col_RefineForceFP64(OpenMXBandColState *state, int on);
void Band_DFT_Col_RefineAfterMixing(OpenMXBandColState *state, double normrd);
void Band_DFT_Col_RefineEndCycle(OpenMXBandColState *state);
/* The exact exchange of a hybrid functional for the solves that follow, or
   NULL without one. The solver adds the exchange matrix K(k) to its dense
   Hamiltonian at every k point and hands the vectors and occupations of a
   collection over (openmx_exact_exchange.h). The state is borrowed from the
   same calculation. */
struct OpenMXExactExchangeState;
void Band_DFT_Col_ExactExchange(OpenMXBandColState *state, struct OpenMXExactExchangeState *exact_exchange);
#ifdef __cplusplus
}
#endif
#endif

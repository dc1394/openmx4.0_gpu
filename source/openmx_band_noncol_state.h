#ifndef OPENMX_BAND_NONCOL_STATE_H
#define OPENMX_BAND_NONCOL_STATE_H
#include "openmx_core.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Borrowed calculation metadata. Binding/clear/finish require quiescent
   users; clear/finish invalidates the view before MPI finalization. */
typedef struct OpenMXBandNonColState OpenMXBandNonColState;
OpenMXBandNonColState *Band_DFT_NonCol_Bind_Resources(OpenMXCoreContext *core);
int Band_DFT_NonCol_RefineLastRefined(OpenMXBandNonColState *state);
void Band_DFT_NonCol_RefineForceFP64(OpenMXBandNonColState *state, int on);
void Band_DFT_NonCol_RefineAfterMixing(OpenMXBandNonColState *state, double normrd);
void Band_DFT_NonCol_RefineEndCycle(OpenMXBandNonColState *state);
#ifdef __cplusplus
}
#endif
#endif

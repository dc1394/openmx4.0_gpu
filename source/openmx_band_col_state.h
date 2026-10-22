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
#ifdef __cplusplus
}
#endif
#endif

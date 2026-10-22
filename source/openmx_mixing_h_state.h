#ifndef OPENMX_MIXING_H_STATE_H
#define OPENMX_MIXING_H_STATE_H
#include "openmx_core.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Borrowed calculation state. Bind, clear and finish require quiescent
   users. GPU work must finish before release; clear/finish invalidates the
   view. Persistent CPU/GPU histories belong to this exact calculation. */
typedef struct OpenMXMixingHState OpenMXMixingHState;
OpenMXMixingHState *Mixing_H_Bind_Resources(OpenMXCoreContext *core);
void Mixing_H_Release_GPU(OpenMXMixingHState *state);
#ifdef __cplusplus
}
#endif
#endif

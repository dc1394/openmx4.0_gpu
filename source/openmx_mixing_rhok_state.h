#ifndef OPENMX_MIXING_RHOK_STATE_H
#define OPENMX_MIXING_RHOK_STATE_H
#include "openmx_core.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Borrowed density-mixing state. Bind, clear and finish require quiescent
   users; clear/finish invalidates this view and releases its host history. */
typedef struct OpenMXMixingRhokState OpenMXMixingRhokState;
OpenMXMixingRhokState *Mixing_Rhok_Bind_Resources(OpenMXCoreContext *core);
#ifdef __cplusplus
}
#endif
#endif

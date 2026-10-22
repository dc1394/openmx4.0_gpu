#ifndef OPENMX_MIXING_DM_STATE_H
#define OPENMX_MIXING_DM_STATE_H
#include "openmx_core.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Borrowed calculation state; clear/finish requires quiescent users and
   invalidates the state. Numerical scratch has a shorter invocation lifetime. */
typedef struct OpenMXMixingDMState OpenMXMixingDMState;
OpenMXMixingDMState *Mixing_DM_Bind_Resources(OpenMXCoreContext *core);
#ifdef __cplusplus
}
#endif
#endif

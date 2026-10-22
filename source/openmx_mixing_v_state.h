#ifndef OPENMX_MIXING_V_STATE_H
#define OPENMX_MIXING_V_STATE_H
#include "openmx_core.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Borrowed calculation state. Clear/finish requires quiescent users and
   invalidates this state; numerical scratch has an invocation lifetime. */
typedef struct OpenMXMixingVState OpenMXMixingVState;
OpenMXMixingVState *Mixing_V_Bind_Resources(OpenMXCoreContext *core);
#ifdef __cplusplus
}
#endif
#endif

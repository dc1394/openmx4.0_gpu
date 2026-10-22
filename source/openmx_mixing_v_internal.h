#ifndef OPENMX_MIXING_V_INTERNAL_H
#define OPENMX_MIXING_V_INTERNAL_H
#include "openmx_mixing_v_state.h"
#include "openmx_core_native_resources.h"
typedef struct {
  int valid,last_scf_iter,last_nummix,spinmax,l38;
  double *A; /* [l38*l38] cached Allreduced Gram matrix */
} MixVKSkFastState;
struct OpenMXMixingVState {
  OpenMXNativeResources *resources;
  int initialized,diis_first,kerker_first;
  MixVKSkFastState fast;
};
typedef OpenMXMixingVState MixingVState;
#endif

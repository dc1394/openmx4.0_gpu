#ifndef OPENMX_MIXING_RHOK_INTERNAL_H
#define OPENMX_MIXING_RHOK_INTERNAL_H
#include "openmx_mixing_rhok_state.h"
#include "openmx_core_native_resources.h"
#include "openmx_core_mixing.h"

/* Only the two native density-mixing implementations share this layout.
   The C++ calculation resource group owns this POD and the Gram storage. */
typedef struct {
  int valid;
  int last_scf_iter;
  int last_nummix;
  int spinmax;
  int l38;
  double *A; /* [l38*l38] cached Allreduced Gram (index [i*l38+j]) */
} MixRhokFastState;

struct OpenMXMixingRhokState {
  OpenMXMixingHistory *history; /* borrowed from this calculation */
  OpenMXNativeResources *resources;
  int initialized;
  int diis_first_normal,diis_first_negf;
  int kerker_first_normal,kerker_first_negf;
  MixRhokFastState fast;
};
typedef OpenMXMixingRhokState MixingRhokState;
#endif

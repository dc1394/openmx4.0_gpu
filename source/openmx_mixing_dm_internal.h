#ifndef OPENMX_MIXING_DM_INTERNAL_H
#define OPENMX_MIXING_DM_INTERNAL_H
#include "openmx_mixing_dm_state.h"
#include "openmx_core_native_resources.h"
#include "openmx_core_mixing.h"
#include "openmx_core_energy.h"
struct OpenMXMixingDMState {
  OpenMXMixingHistory *history; /* shared with reciprocal/Hamiltonian/potential mixing */
  OpenMXBandEnergies *energies; /* borrowed from this calculation */
  int initialized;
  int diis_first,gr_first;
};
#endif

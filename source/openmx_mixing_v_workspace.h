#ifndef OPENMX_MIXING_V_WORKSPACE_H
#define OPENMX_MIXING_V_WORKSPACE_H
#include "openmx_mixing_v_internal.h"
#include "openmx_solver_workspace.h"
typedef struct {
  OpenMXMixingVState *state;
  int md_iteration,scf_iteration,first_iteration;
} MixV_ScratchArguments;
#endif

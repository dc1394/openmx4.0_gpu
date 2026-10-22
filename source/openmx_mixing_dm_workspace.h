#ifndef OPENMX_MIXING_DM_WORKSPACE_H
#define OPENMX_MIXING_DM_WORKSPACE_H
#include "openmx_mixing_dm_internal.h"
#include "openmx_solver_workspace.h"

typedef struct MixDM_ScratchArguments MixDM_ScratchArguments;
typedef void (*MixDM_NumericalOperation)(OpenMXSolverInvocation*,const MixDM_ScratchArguments*);
struct MixDM_ScratchArguments {
  MixDM_NumericalOperation operation;
  OpenMXMixingDMState *state;
  int iteration;
  double ******residual,******imaginary_residual;
  const OpenMXElectronicViews *electronic;
};
static int MixDM_ScratchOperation(OpenMXSolverInvocation *scratch,void *opaque,double *result)
{
  const MixDM_ScratchArguments *arguments=(const MixDM_ScratchArguments*)opaque;
  arguments->operation(scratch,arguments);
  *result=0.0;
  return 1;
}
static inline void MixDM_WithScratch(MixDM_NumericalOperation operation,OpenMXMixingDMState *state,
    int iteration,double ******residual,double ******imaginary_residual,const OpenMXElectronicViews *electronic)
{
  MixDM_ScratchArguments arguments={operation,state,iteration,residual,imaginary_residual,electronic};
  (void)OpenMX_Solver_Run(MixDM_ScratchOperation,&arguments);
}
#endif

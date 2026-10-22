#ifndef OPENMX_MIXING_RHOK_WORKSPACE_H
#define OPENMX_MIXING_RHOK_WORKSPACE_H
#include "openmx_mixing_rhok_internal.h"
#include "openmx_solver_workspace.h"

/* The native grid routines borrow scratch from one C++ invocation. The
   opaque calculation state separately owns persistent density history. */
typedef void (*MixRhok_NumericalOperation)(OpenMXSolverInvocation*,MixingRhokState*,
    int,double,double***,double***,double**,double**,double*,double*,double*,double*);
typedef struct {
  MixRhok_NumericalOperation operation;
  MixingRhokState *state;
  int control;
  double weight;
  double ***real,***imag,**residual_real,**residual_imag;
  double *potential_real,*potential_imag,*atomic_real,*atomic_imag;
} MixRhok_ScratchArguments;

static int MixRhok_ScratchOperation(OpenMXSolverInvocation *scratch,void *opaque,double *result)
{
  const MixRhok_ScratchArguments *a=(const MixRhok_ScratchArguments*)opaque;
  a->operation(scratch,a->state,a->control,a->weight,a->real,a->imag,
      a->residual_real,a->residual_imag,a->potential_real,a->potential_imag,
      a->atomic_real,a->atomic_imag);
  *result=0.0;
  return 1;
}

static inline void MixRhok_WithScratch(MixRhok_NumericalOperation operation,MixingRhokState *state,
    int control,double weight,double ***real,double ***imag,double **residual_real,double **residual_imag,
    double *potential_real,double *potential_imag,double *atomic_real,double *atomic_imag)
{
  MixRhok_ScratchArguments arguments={operation,state,control,weight,real,imag,residual_real,residual_imag,
      potential_real,potential_imag,atomic_real,atomic_imag};
  (void)OpenMX_Solver_Run(MixRhok_ScratchOperation,&arguments);
}
#endif

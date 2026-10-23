#ifndef OPENMX_STRESS_WORKSPACE_H
#define OPENMX_STRESS_WORKSPACE_H
#include "openmx_solver_workspace.h"
#include "openmx_xc_workspace.h"

/* Nine active components per worker. A 256-byte row separates their writes
   even for 128-byte cache lines; unused padding is never read or zeroed.
   Prepare only between joined worker regions. All stages reuse this slot. */
static inline void OpenMX_Stress_ThreadsPrepare(OpenMXSolverInvocation *owner,
    int workers,OpenMXSolverRealTensor *out)
{
  char error[256];const size_t shape[4]={(size_t)workers,32,1,1};
  if(workers<1)OpenMX_Solver_Failure("invalid stress worker count");
  if(!openmx_core_solver_real_overwrite(owner,omp_get_thread_num(),"stress:thread-sums",
        shape,out,error,sizeof(error)))OpenMX_Solver_Failure(error);
}

/* FFT/MPI interfaces retain a nonnull row on empty ranks. Even columns
   preserve the alignment of both borrowed real/imaginary rows. */
static inline void OpenMX_Stress_ReciprocalPrepare(OpenMXSolverInvocation *owner,
    int points,OpenMXSolverRealTensor *out)
{
  char error[256];const size_t columns=OpenMX_Solver_SimdColumns(points>0?(size_t)points:1);
  const size_t shape[4]={2,columns,1,1};
  if(points<0)OpenMX_Solver_Failure("invalid stress reciprocal grid count");
  if(!openmx_core_solver_real_overwrite(owner,omp_get_thread_num(),"stress:hartree-density",
        shape,out,error,sizeof(error)))OpenMX_Solver_Failure(error);
}
#endif

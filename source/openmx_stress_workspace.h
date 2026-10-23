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

typedef struct {
  OpenMXSolverRealTensor potential,derivatives,displacements;
} OpenMXStressGridScratch;

static inline void OpenMX_Stress_OverwritePrepare(OpenMXSolverInvocation *owner,
    const char *name,const size_t shape[4],OpenMXSolverRealTensor *out)
{
  char error[256];
  if(!openmx_core_solver_real_overwrite(owner,omp_get_thread_num(),name,
        shape,out,error,sizeof(error)))OpenMX_Solver_Failure(error);
}

/* Main grid caches are shared across joined worksharing regions; only
   active points/orbitals are produced and consumed. Empty shapes keep a
   nonnull row for native interfaces. No numeric values are initialized. */
static inline void OpenMX_Stress_GridPrepare(OpenMXSolverInvocation *owner,
    int spin,int points,int orbitals,OpenMXStressGridScratch *out)
{
  if((spin!=0 && spin!=1 && spin!=3)||points<0||orbitals<1)
    OpenMX_Solver_Failure("invalid stress grid shape");
  const size_t count=points>0?(size_t)points:1;
  const size_t potential[4]={(size_t)spin+1,OpenMX_Solver_SimdColumns(count),1,1};
  const size_t derivatives[4]={count,(size_t)orbitals,3,1};
  const size_t displacements[4]={count,3,1,1};
  OpenMX_Stress_OverwritePrepare(owner,"stress:grid:potential",potential,&out->potential);
  OpenMX_Stress_OverwritePrepare(owner,"stress:grid:derivatives",derivatives,&out->derivatives);
  OpenMX_Stress_OverwritePrepare(owner,"stress:grid:displacements",displacements,&out->displacements);
}
static inline double *OpenMX_Stress_DerivativeRow(const OpenMXSolverRealTensor *view,
    size_t point,size_t orbital)
{
  return view->values+(point*view->extent[1]+orbital)*view->extent[2];
}

/* Four fixed borrowed rows adapt the existing orbital producer ABI. */
static inline void OpenMX_Stress_OrbitalPrepare(OpenMXSolverInvocation *owner,
    int orbitals,OpenMXSolverRealTensor *out)
{
  if(orbitals<1)OpenMX_Solver_Failure("invalid stress orbital count");
  const size_t shape[4]={4,OpenMX_Solver_SimdColumns((size_t)orbitals),1,1};
  OpenMX_Stress_OverwritePrepare(owner,"stress:grid:orbital-rows",shape,out);
}

/* [field (XC derivative/density gradient),spin,axis,grid] replaces four
   pointer trees. Bind the fixed legacy D rows only for Set_XC_Grid. */
static inline void OpenMX_Stress_GradientPrepare(OpenMXSolverInvocation *owner,
    const char *name,int points,OpenMXSolverRealTensor *out)
{
  if(points<0)OpenMX_Solver_Failure("invalid stress gradient grid count");
  const size_t shape[4]={2,2,3,OpenMX_Solver_SimdColumns(points>0?(size_t)points:1)};
  OpenMX_Stress_OverwritePrepare(owner,name,shape,out);
}
static inline double *OpenMX_Stress_GradientRow(const OpenMXSolverRealTensor *view,
    size_t field,size_t spin,size_t axis)
{
  return view->values+((field*view->extent[1]+spin)*view->extent[2]+axis)*view->extent[3];
}
#endif

#ifndef OPENMX_ENERGY_WORKSPACE_H
#define OPENMX_ENERGY_WORKSPACE_H
#include "openmx_solver_workspace.h"
#include <limits.h>

/* Borrowed for one Total_Energy invocation. The C++ owner holds a flat
   [atom/neighbor,pair] integer tensor, with the legacy padding element.
   No process/global/thread-local plan is installed. */
typedef struct {
  OpenMXSolverInvocation *owner;
  int pair_count;
  OpenMXSolverIntTensor pair_indices;
} OpenMXEnergyWorkspace;

static inline void OpenMX_Energy_PairsPrepare(OpenMXSolverInvocation *owner,
    size_t count, OpenMXEnergyWorkspace *out)
{
  char error[256];
  const size_t shape[4]={2,count+1,1,1};
  if(count>INT_MAX)OpenMX_Solver_Failure("too many total-energy atom pairs");
  if(!openmx_core_solver_int_tensor(owner,omp_get_thread_num(),"total-energy:pairs",
        shape,&out->pair_indices,error,sizeof(error)))OpenMX_Solver_Failure(error);
  out->owner=owner;
  out->pair_count=(int)count;
}

/* Borrowed EH0 batch. Integer rows are base species, other species and the
   derivative flag; real rows are distance, direction xyz and four outputs.
   Empty MPI ranks retain one padding element per row for the native ABI. */
typedef struct {
  int count;
  OpenMXSolverIntTensor indices;
  OpenMXSolverRealTensor values;
} OpenMXEnergyEH0Batch;

static inline void OpenMX_Energy_EH0BatchPrepare(OpenMXSolverInvocation *owner,
    size_t count,OpenMXEnergyEH0Batch *out)
{
  char error[256];
  const size_t capacity=count ? count : 1;
  const size_t integers[4]={3,capacity,1,1},reals[4]={8,capacity,1,1};
  if(count>INT_MAX)OpenMX_Solver_Failure("too many EH0 atom pairs");
  if(!openmx_core_solver_int_tensor(owner,omp_get_thread_num(),"total-energy:eh0-indices",
        integers,&out->indices,error,sizeof(error)))OpenMX_Solver_Failure(error);
  if(!openmx_core_solver_real_overwrite(owner,omp_get_thread_num(),"total-energy:eh0-batch",
        reals,&out->values,error,sizeof(error)))OpenMX_Solver_Failure(error);
  out->count=(int)count;
}

/* Explicit output parameters also avoid large-aggregate returns in native
   OpenMP code. Both [spin,CWF] matrices live until the invocation ends. */
static inline void OpenMX_Energy_CWFScratch(OpenMXSolverInvocation *owner,
    int orbitals,OpenMXSolverRealTensor *double_count,OpenMXSolverRealTensor *atomic)
{
  char error[256];
  const size_t shape[4]={2,(size_t)orbitals,1,1};
  if(orbitals<1)OpenMX_Solver_Failure("empty CWF energy decomposition");
  if(!openmx_core_solver_real_overwrite(owner,omp_get_thread_num(),
        "total-energy:cwf-double-count",shape,double_count,error,sizeof(error)))
    OpenMX_Solver_Failure(error);
  if(!openmx_core_solver_real_overwrite(owner,omp_get_thread_num(),
        "total-energy:cwf-atomic",shape,atomic,error,sizeof(error)))
    OpenMX_Solver_Failure(error);
}

/* The legacy angular-grid producer and GPU entry points borrow a row table;
   its metadata and the flat [point,xyz/weight] data share this invocation. */
static inline void OpenMX_Energy_LebedevPrepare(OpenMXSolverInvocation *owner,
    int points,OpenMXSolverRealTensor *out,double ***legacy_rows)
{
  char error[256];
  const size_t shape[4]={(size_t)points,4,1,1};
  if(points!=590 && points!=1202 && points!=2030)
    OpenMX_Solver_Failure("unsupported Lebedev grid: use 590, 1202 or 2030 points");
  if(!openmx_core_solver_real_overwrite(owner,omp_get_thread_num(),"total-energy:lebedev",
        shape,out,error,sizeof(error)))OpenMX_Solver_Failure(error);
  *legacy_rows=OpenMX_Solver_RealAlias(owner,"total-energy:lebedev-rows",out->values,(size_t)points,4);
}

/* Flat [component,worker,neighbor] partials, shared only during one atom's
   joined worker region. Capacity is constant across atoms in this evaluation. */
static inline void OpenMX_Energy_XCPartialsPrepare(OpenMXSolverInvocation *owner,
    size_t workers,size_t neighbors,OpenMXSolverRealTensor *out)
{
  char error[256];const size_t shape[4]={3,workers,neighbors,1};
  if(!openmx_core_solver_real_overwrite(owner,omp_get_thread_num(),"total-energy:xc-partials",
        shape,out,error,sizeof(error)))OpenMX_Solver_Failure(error);
}
static inline double *OpenMX_Energy_XCForceRow(const OpenMXSolverRealTensor *view,
    size_t component,size_t worker)
{
  return view->extent[2] ? view->values+(component*view->extent[1]+worker)*view->extent[2] : view->values;
}

/* Each worker reuses six derivative/sum rows. Keep an output parameter in
   native OpenMP regions; no large aggregate is returned from this helper. */
static inline void OpenMX_Energy_XCWorkerPrepare(OpenMXSolverInvocation *owner,
    size_t neighbors,OpenMXSolverRealTensor *out)
{
  char error[256];const size_t shape[4]={6,neighbors,1,1};
  if(!openmx_core_solver_real_overwrite(owner,omp_get_thread_num(),"total-energy:xc-derivatives",
        shape,out,error,sizeof(error)))OpenMX_Solver_Failure(error);
}
#endif

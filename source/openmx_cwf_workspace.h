#ifndef OPENMX_CWF_WORKSPACE_H
#define OPENMX_CWF_WORKSPACE_H
#include "openmx_solver_workspace.h"
#include "openmx_core_cwf_workspace.h"

static inline void OpenMX_CWF_Blocks(OpenMXSolverInvocation *owner,const char *name,
    size_t spins,size_t ni,size_t nj,size_t nk,size_t elements,size_t bytes,OpenMXCWFBlocks *out)
{
  const OpenMXCWFBlockShape shape={spins,{ni,nj,nk},elements};char error[256];
  if(!openmx_core_solver_cwf_blocks(owner,omp_get_thread_num(),name,&shape,bytes,out,error,sizeof(error)))
    OpenMX_Solver_Failure(error);
}
#endif

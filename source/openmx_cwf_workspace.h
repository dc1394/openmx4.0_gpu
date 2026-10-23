#ifndef OPENMX_CWF_WORKSPACE_H
#define OPENMX_CWF_WORKSPACE_H
#include "openmx_solver_workspace.h"
#include "openmx_core_cwf_workspace.h"
#include <limits.h>

static inline void OpenMX_CWF_Blocks(OpenMXSolverInvocation *owner,const char *name,
    size_t spins,size_t ni,size_t nj,size_t nk,size_t elements,size_t bytes,OpenMXCWFBlocks *out)
{
  const OpenMXCWFBlockShape shape={spins,{ni,nj,nk},elements};char error[256];
  if(!openmx_core_solver_cwf_blocks(owner,omp_get_thread_num(),name,&shape,bytes,out,error,sizeof(error)))
    OpenMX_Solver_Failure(error);
}

/* Each path keeps only its own points, spins and eigenvalue padding. The
   descriptor array and every numeric block belong to the same invocation. */
static inline OpenMXCWFBlocks *OpenMX_CWF_PathValues(OpenMXSolverInvocation *owner,
    int paths,const int *points,int spins,int bands)
{
  int path;OpenMXCWFBlocks *views;char name[96];
  if(paths<0 || paths==INT_MAX || !points || spins<1 || bands<0)
    OpenMX_Solver_Failure("invalid CWF band path dimensions");
  views=(OpenMXCWFBlocks*)OpenMX_Solver_Bytes(owner,"band-cwf:dispersion:paths",
      (size_t)paths+1,sizeof(*views));
  for(path=0;path<=paths;path++){
    if(points[path]<0)OpenMX_Solver_Failure("negative CWF band path point count");
    snprintf(name,sizeof(name),"band-cwf:dispersion:EigenVal:%d",path);
    OpenMX_CWF_Blocks(owner,name,spins,(size_t)points[path]+1,1,1,
        (size_t)bands+2,sizeof(double),&views[path]);
  }
  return views;
}
static inline double *OpenMX_CWF_PathRow(const OpenMXCWFBlocks *paths,
    size_t path,size_t point,size_t spin)
{ return (double*)openmx_cwf_block_cell(&paths[path],spin,point,0,0); }

/* Preserve multiply-before-divide for ordinary paths. A one-point path is
   its start point; dummy index zero is never a physical k point. */
static inline double OpenMX_CWF_PathCoordinate(double start,double end,int point,int count)
{
  if(point<1 || point>count)OpenMX_Solver_Failure("invalid CWF band path point");
  return count==1 ? start : start+(end-start)*(point-1)/(count-1);
}
static inline void OpenMX_CWF_SVDStatus(const char *stage,int info)
{
  if(info!=0){
    char error[160];snprintf(error,sizeof(error),"%s failed (INFO=%d)",stage,info);
    OpenMX_Solver_Failure(error);
  }
}
static inline int OpenMX_CWF_SVDWorkSize(double requested)
{
  /* The comparisons reject NaN and infinities before conversion to int. */
  if(!(requested>=1.0 && requested<=INT_MAX) || (double)(int)requested!=requested)
    OpenMX_Solver_Failure("invalid CWF SVD workspace query size");
  return (int)requested;
}
#endif

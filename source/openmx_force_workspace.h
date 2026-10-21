#ifndef OPENMX_FORCE_WORKSPACE_H
#define OPENMX_FORCE_WORKSPACE_H
#include "openmx_core_force.h"
#include "openmx_core_local_basis.h"
#include "openmx_solver_workspace.h"

static inline double OpenMX_ForceRun(const OpenMXForceInput *input,OpenMXForceOperation operation,void *args)
{
  double result;char error[256];
  if(!openmx_core_force_run(input,omp_get_max_threads(),operation,args,&result,error,sizeof(error)))
    OpenMX_Solver_Failure(error);
  return result;
}
static inline dcomplex ***OpenMX_ForceComplexVolume(OpenMXSolverInvocation *owner,
    const char *name,size_t components,size_t rows,size_t columns)
{
  const size_t shape[4]={components,rows,columns,1};
  OpenMXSolverComplexTensor view;dcomplex ***aliases;char error[256];
  if(!openmx_core_solver_complex(owner,omp_get_thread_num(),name,shape,&view,error,sizeof(error)) ||
     !openmx_core_solver_complex_alias(owner,omp_get_thread_num(),name,&view,&aliases,error,sizeof(error)))
    OpenMX_Solver_Failure(error);
  return aliases;
}
static inline dcomplex ****OpenMX_ForceSpinorMatrix(OpenMXSolverInvocation *owner,
    const char *name,size_t orbitals)
{
  dcomplex ***planes=OpenMX_ForceComplexVolume(owner,name,4,orbitals,orbitals);
  dcomplex ****spin=OpenMX_Solver_Bytes(owner,name,2,sizeof(*spin));
  spin[0]=planes;spin[1]=planes+2;return spin;
}
static inline int **OpenMX_ForceIntMatrix(OpenMXSolverInvocation *owner,
    const char *name,size_t rows,size_t columns)
{
  OpenMXSolverIntTensor values=OpenMX_Solver_IntView(owner,name,rows,columns);
  int **aliases=OpenMX_Solver_Bytes(owner,name,rows,sizeof(*aliases));
  for(size_t i=0;i<rows;++i)aliases[i]=columns?values.values+i*columns:values.values;
  return aliases;
}
static inline OpenMXLocalTensor3 OpenMX_ForceHubbardScratch(OpenMXSolverInvocation *owner,size_t orbitals)
{
  const size_t shape[4]={9,orbitals,orbitals,1};OpenMXSolverRealTensor view;char error[256];
  if(!openmx_core_solver_real(owner,omp_get_thread_num(),"Force:Hubbard-transpose",shape,&view,error,sizeof(error)))
    OpenMX_Solver_Failure(error);
  return (OpenMXLocalTensor3){9,orbitals,orbitals,view.values};
}
static inline void OpenMX_ForceSpinTrace(int spin,int spinor,double prefactor,int rows,int columns,
    double *****density,double *****imaginary,int atom,int neighbor,dcomplex ***dx,dcomplex ***dy,
    dcomplex ***dz,size_t derivative_rows,size_t derivative_stride,double *fx,double *fy,double *fz)
{
  OpenMXForceSpinTrace input={0};input.spin=spin;input.spinor=spinor;
  input.rows=rows;input.columns=columns;input.prefactor=prefactor;
  input.derivative_rows=derivative_rows;input.derivative_stride=derivative_stride;
  if(rows && columns){
    for(int c=0;c<(spinor?4:(spin?2:1));++c)input.density[c]=density[c][atom][neighbor][0];
    if(spinor)for(int c=0;c<2;++c)input.imaginary[c]=imaginary[c][atom][neighbor][0];
    input.derivative[0]=dx[0][0];input.derivative[1]=dy[0][0];input.derivative[2]=dz[0][0];
  }
  double force[3]={*fx,*fy,*fz};char error[256];
  if(!openmx_core_force_spin_trace(&input,force,error,sizeof(error)))OpenMX_Solver_Failure(error);
  *fx=force[0];*fy=force[1];*fz=force[2];
}
#endif

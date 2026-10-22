#ifndef OPENMX_SOLVER_WORKSPACE_H
#define OPENMX_SOLVER_WORKSPACE_H
#include "openmx_core_solver_workspace.h"
#include "mpi.h"
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

/* Native numerical functions borrow these views. Their enclosing C++ call
   owns the storage and releases it automatically on every callback return. */
static inline void OpenMX_Solver_Failure(const char *error)
{
  fprintf(stderr,"OpenMX solver workspace: %s\n",error);
  MPI_Abort(mpi_comm_level1,1);abort();
}
static inline double OpenMX_Solver_Run(OpenMXSolverOperation operation,void *arguments)
{
  double result;char error[256];
  if(!openmx_core_solver_run(omp_get_max_threads(),operation,arguments,&result,error,sizeof(error)))
    OpenMX_Solver_Failure(error);
  return result;
}
static inline OpenMXNativeResources *OpenMX_Solver_Resources(OpenMXSolverInvocation *owner,const char *name)
{
  OpenMXNativeResources *view;char error[256];
  if(!openmx_core_solver_resources(owner,name,&view,error,sizeof(error)))OpenMX_Solver_Failure(error);
  return view;
}
static inline double *OpenMX_Solver_Real(OpenMXSolverInvocation *owner,const char *name,size_t count)
{
  const size_t shape[4]={count,1,1,1};OpenMXSolverRealTensor view;char error[256];
  if(!openmx_core_solver_real(owner,omp_get_thread_num(),name,shape,&view,error,sizeof(error)))OpenMX_Solver_Failure(error);
  return view.values;
}
/* Keep the output parameter: returning this large aggregate by value inside
   Force3's OpenMP region produced invalid loop indices with NVHPC 26.9. */
static inline void OpenMX_Solver_Orbitals(OpenMXSolverInvocation *owner,
    const char *name,int maximum_angular,int multiplicity,int derivatives,OpenMXOrbitalBuffers *view)
{
  const OpenMXOrbitalShape shape={maximum_angular,multiplicity,derivatives};
  char error[256];
  if(!openmx_core_solver_orbitals(owner,omp_get_thread_num(),name,&shape,view,error,sizeof(error)))
    OpenMX_Solver_Failure(error);
}
static inline dcomplex *OpenMX_Solver_Complex(OpenMXSolverInvocation *owner,const char *name,size_t count)
{
  const size_t shape[4]={count,1,1,1};OpenMXSolverComplexTensor view;char error[256];
  if(!openmx_core_solver_complex(owner,omp_get_thread_num(),name,shape,&view,error,sizeof(error)))OpenMX_Solver_Failure(error);
  return view.values;
}
static inline int *OpenMX_Solver_Int(OpenMXSolverInvocation *owner,const char *name,size_t count)
{
  int *view;char error[256];
  if(!openmx_core_solver_int(owner,omp_get_thread_num(),name,count,&view,error,sizeof(error)))OpenMX_Solver_Failure(error);
  return view;
}
static inline OpenMXSolverIntTensor OpenMX_Solver_IntView(OpenMXSolverInvocation *owner,const char *name,size_t rows,size_t columns)
{
  const size_t shape[4]={rows,columns,1,1};OpenMXSolverIntTensor view;char error[256];
  if(!openmx_core_solver_int_tensor(owner,omp_get_thread_num(),name,shape,&view,error,sizeof(error)))OpenMX_Solver_Failure(error);
  return view;
}
static inline void *OpenMX_Solver_Bytes(OpenMXSolverInvocation *owner,const char *name,size_t count,size_t bytes)
{
  void *view;char error[256];
  if(!openmx_core_solver_bytes(owner,omp_get_thread_num(),name,count,bytes,&view,error,sizeof(error)))OpenMX_Solver_Failure(error);
  return view;
}
static inline double **OpenMX_Solver_RealMatrix(OpenMXSolverInvocation *owner,const char *name,size_t rows,size_t columns)
{
  double **view;char error[256];
  if(!openmx_core_solver_real_rows(owner,omp_get_thread_num(),name,rows,columns,&view,error,sizeof(error)))OpenMX_Solver_Failure(error);
  return view;
}
static inline OpenMXSolverRealTensor OpenMX_Solver_RealView(OpenMXSolverInvocation *owner,const char *name,size_t rows,size_t columns)
{
  const size_t shape[4]={rows,columns,1,1};OpenMXSolverRealTensor view;char error[256];
  if(!openmx_core_solver_real(owner,omp_get_thread_num(),name,shape,&view,error,sizeof(error)))OpenMX_Solver_Failure(error);
  return view;
}
static inline OpenMXSolverComplexTensor OpenMX_Solver_ComplexView(OpenMXSolverInvocation *owner,const char *name,size_t rows,size_t columns)
{
  const size_t shape[4]={rows,columns,1,1};OpenMXSolverComplexTensor view;char error[256];
  if(!openmx_core_solver_complex(owner,omp_get_thread_num(),name,shape,&view,error,sizeof(error)))OpenMX_Solver_Failure(error);
  return view;
}
static inline dcomplex **OpenMX_Solver_ComplexMatrix(OpenMXSolverInvocation *owner,const char *name,size_t rows,size_t columns)
{
  dcomplex **view;char error[256];
  if(!openmx_core_solver_complex_rows(owner,omp_get_thread_num(),name,rows,columns,&view,error,sizeof(error)))OpenMX_Solver_Failure(error);
  return view;
}
static inline double ***OpenMX_Solver_RealPlanes(OpenMXSolverInvocation *owner,const char *name,double *values,size_t components,size_t rows,size_t columns)
{
  const OpenMXSolverRealTensor view={{components,rows,columns,1},values};double ***aliases;char error[256];
  if(!openmx_core_solver_real_alias(owner,omp_get_thread_num(),name,&view,&aliases,error,sizeof(error)))OpenMX_Solver_Failure(error);
  return aliases;
}
static inline double **OpenMX_Solver_RealAlias(OpenMXSolverInvocation *owner,const char *name,double *values,size_t rows,size_t columns)
{ return OpenMX_Solver_RealPlanes(owner,name,values,1,rows,columns)[0]; }
static inline double ***OpenMX_Solver_RealVolume(OpenMXSolverInvocation *owner,const char *name,size_t components,size_t rows,size_t columns)
{
  const size_t shape[4]={components,rows,columns,1};OpenMXSolverRealTensor view;char error[256];
  if(!openmx_core_solver_real(owner,omp_get_thread_num(),name,shape,&view,error,sizeof(error)))OpenMX_Solver_Failure(error);
  return OpenMX_Solver_RealPlanes(owner,name,view.values,components,rows,columns);
}
/* Legacy SIMD kernels may map unaligned intrinsics to aligned loads on some
   targets. An even row stride retains the former per-row malloc alignment. */
static inline size_t OpenMX_Solver_SimdColumns(size_t columns)
{
  if(columns==(size_t)-1)OpenMX_Solver_Failure("solver SIMD row stride overflows");
  return columns+(columns&1U);
}
static inline double **OpenMX_Solver_AlignedRealMatrix(OpenMXSolverInvocation *owner,const char *name,size_t rows,size_t columns)
{ return OpenMX_Solver_RealMatrix(owner,name,rows,OpenMX_Solver_SimdColumns(columns)); }
static inline double ***OpenMX_Solver_AlignedRealVolume(OpenMXSolverInvocation *owner,const char *name,size_t components,size_t rows,size_t columns)
{ return OpenMX_Solver_RealVolume(owner,name,components,rows,OpenMX_Solver_SimdColumns(columns)); }
static inline dcomplex **OpenMX_Solver_ComplexAlias(OpenMXSolverInvocation *owner,const char *name,dcomplex *values,size_t rows,size_t columns)
{
  const OpenMXSolverComplexTensor view={{1,rows,columns,1},values};dcomplex ***aliases;char error[256];
  if(!openmx_core_solver_complex_alias(owner,omp_get_thread_num(),name,&view,&aliases,error,sizeof(error)))OpenMX_Solver_Failure(error);
  return aliases[0];
}
typedef struct {
  OpenMXSolverRealTensor ReEVec_i1,ImEVec_i1,ReEVec_i2,ImEVec_i2;
  OpenMXSolverRealTensor ReEVec_j1,ImEVec_j1,ReEVec_j2,ImEVec_j2;
} OpenMXBandSpinorScratch;
static inline OpenMXSolverRealFieldView OpenMX_Solver_Field(OpenMXSolverInvocation *owner,size_t components,size_t atoms)
{
  OpenMXSolverRealFieldView view;char error[256];
  if(!openmx_core_solver_field_create(owner,components,atoms,&view,error,sizeof(error)))OpenMX_Solver_Failure(error);
  return view;
}
static inline void OpenMX_Solver_FieldResize(OpenMXSolverRealFieldView view,size_t component,size_t atom,size_t count)
{
  char error[256];
  if(!openmx_core_solver_field_resize(view.owner,component,atom,count,error,sizeof(error)))OpenMX_Solver_Failure(error);
}
static inline OpenMXSolverResidueView OpenMX_Solver_Residues(OpenMXSolverInvocation *owner,int components,int padding,int complex_values)
{
  OpenMXSolverResidueView view;char error[256];
  const OpenMXSolverResidueInput in={atomnum,Matomnum,SpeciesNum,components,padding,complex_values,
      M2G,WhatSpecies,Spe_Total_CNO,FNAN,natn};
  if(!openmx_core_solver_residue_create(owner,&in,&view,error,sizeof(error)))OpenMX_Solver_Failure(error);
  return view;
}
static inline void OpenMX_Solver_ResidueResize(OpenMXSolverResidueView view,size_t component,size_t atom,size_t states)
{
  char error[256];
  if(!openmx_core_solver_residue_resize(view.owner,component,atom,states,error,sizeof(error)))OpenMX_Solver_Failure(error);
}
#endif

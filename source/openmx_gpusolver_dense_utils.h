#ifndef OPENMX_GPUSOLVER_DENSE_UTILS_H
#define OPENMX_GPUSOLVER_DENSE_UTILS_H

#include <stdio.h>
#include <string.h>
#include "mpi.h"

static void OpenMX_GpuSolver_DenseAbort(const char *where, int info)
{
  fprintf(stderr,"%s failed, info=%d\n",where,info);
  fflush(stderr);
  MPI_Abort(mpi_comm_level1,1);
}

static void OpenMX_GpuSolver_DenseDsyevx_1based(double *A, double *Z, double *W,
                                               int n, int maxn, const char *where)
{
  int l, copy_cols;
  int info = gpusolver_Syevdx(A,W,n,maxn);

  if (info!=0) OpenMX_GpuSolver_DenseAbort(where,info);

  for (l=maxn; 1<=l; l--) W[l] = W[l-1];

  if (Z!=NULL && Z!=A){
    copy_cols = maxn;
    if (n<copy_cols) copy_cols = n;
    memcpy(Z,A,sizeof(double)*(size_t)n*(size_t)copy_cols);
  }
}

static void OpenMX_GpuSolver_DenseDsyevx_0based(double *A, double *Z, double *W,
                                               int n, int maxn, const char *where)
{
  int copy_cols;
  int info = gpusolver_Syevdx(A,W,n,maxn);

  if (info!=0) OpenMX_GpuSolver_DenseAbort(where,info);

  if (Z!=NULL && Z!=A){
    copy_cols = maxn;
    if (n<copy_cols) copy_cols = n;
    memcpy(Z,A,sizeof(double)*(size_t)n*(size_t)copy_cols);
  }
}

static void OpenMX_GpuSolver_DenseZheevx_1based(dcomplex *A, dcomplex *Z, double *W,
                                               int n, int maxn, const char *where)
{
  int l, copy_cols;
  int info = gpusolver_Syevdx_Complex(A,W,n,maxn);

  if (info!=0) OpenMX_GpuSolver_DenseAbort(where,info);

  for (l=maxn; 1<=l; l--) W[l] = W[l-1];

  if (Z!=NULL && Z!=A){
    copy_cols = maxn;
    if (n<copy_cols) copy_cols = n;
    memcpy(Z,A,sizeof(dcomplex)*(size_t)n*(size_t)copy_cols);
  }
}

static void OpenMX_GpuSolver_DenseZheevx_0based(dcomplex *A, dcomplex *Z, double *W,
                                               int n, int maxn, const char *where)
{
  int copy_cols;
  int info = gpusolver_Syevdx_Complex(A,W,n,maxn);

  if (info!=0) OpenMX_GpuSolver_DenseAbort(where,info);

  if (Z!=NULL && Z!=A){
    copy_cols = maxn;
    if (n<copy_cols) copy_cols = n;
    memcpy(Z,A,sizeof(dcomplex)*(size_t)n*(size_t)copy_cols);
  }
}

#endif

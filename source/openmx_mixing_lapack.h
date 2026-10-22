#ifndef OPENMX_MIXING_LAPACK_H
#define OPENMX_MIXING_LAPACK_H
#include "openmx_core_lu_ops.h"
/* Include after the native LAPACK declarations. The SDK-free core receives
   explicit operations; native symbols and library selection stay here. */
static inline void OpenMX_Mixing_LuFactor(void *context,int *m,int *n,double *a,int *lda,int *pivots,int *info)
{
  (void)context;
  F77_NAME(dgetrf,DGETRF)(m,n,a,lda,pivots,info);
}
static inline void OpenMX_Mixing_LuInvert(void *context,int *n,double *a,int *lda,int *pivots,double *work,int *lwork,int *info)
{
  (void)context;
  F77_NAME(dgetri,DGETRI)(n,a,lda,pivots,work,lwork,info);
}
#endif

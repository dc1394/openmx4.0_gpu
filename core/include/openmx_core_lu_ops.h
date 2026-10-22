#ifndef OPENMX_CORE_LU_OPS_H
#define OPENMX_CORE_LU_OPS_H
#ifdef __cplusplus
extern "C" {
#endif
/* Explicit LP64 LAPACK operations supplied by the native caller. This header
   introduces no SDK, MPI or BLAS link dependency and installs no provider.
   Operations complete synchronously and preserve LAPACK's input dimensions. */
typedef void (*OpenMXRealLuFactor)(void*,int*,int*,double*,int*,int*,int*);
typedef void (*OpenMXRealLuInvert)(void*,int*,double*,int*,int*,double*,int*,int*);
typedef struct {
  void *context; /* Borrowed only during this call; never cached by the core. */
  OpenMXRealLuFactor factor;
  OpenMXRealLuInvert invert;
} OpenMXRealLuOperations;
#ifdef __cplusplus
}
#endif
#endif

#ifndef OPENMX_AMDGPU_CUCOMPLEX_COMPAT_H
#define OPENMX_AMDGPU_CUCOMPLEX_COMPAT_H

#ifdef __cplusplus
#include <hip/hip_complex.h>
typedef hipFloatComplex cuComplex;
typedef hipFloatComplex cuFloatComplex;
typedef hipDoubleComplex cuDoubleComplex;
#else
typedef struct {
    float x;
    float y;
} cuComplex;
typedef cuComplex cuFloatComplex;
typedef struct {
    double x;
    double y;
} cuDoubleComplex;
#define make_cuComplex(r, i) ((cuComplex){(float)(r), (float)(i)})
#define make_cuFloatComplex(r, i) ((cuFloatComplex){(float)(r), (float)(i)})
#define make_cuDoubleComplex(r, i) ((cuDoubleComplex){(double)(r), (double)(i)})
#endif

#ifdef __cplusplus
#define make_cuComplex make_hipFloatComplex
#define make_cuFloatComplex make_hipFloatComplex
#define make_cuDoubleComplex make_hipDoubleComplex
#define cuCreal hipCreal
#define cuCimag hipCimag
#define cuCrealf hipCrealf
#define cuCimagf hipCimagf
#define cuCadd hipCadd
#define cuCsub hipCsub
#define cuCmul hipCmul
#endif

#endif

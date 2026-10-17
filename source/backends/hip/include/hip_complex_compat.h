#ifndef OPENMX_AMDGPU_HIP_COMPLEX_COMPAT_H
#define OPENMX_AMDGPU_HIP_COMPLEX_COMPAT_H

/*
 * HIP complex compatibility header for OpenMX's own sources.
 * C++ TUs use the real ROCm hip complex header; plain C TUs get a matching
 * POD definition and helper macros.  See hip_runtime_compat.h for the rationale
 * behind keeping the CUDA-named shim headers for the third-party libraries.
 */

#ifdef __cplusplus
#include <hip/hip_complex.h>
#else
typedef struct {
    float x;
    float y;
} hipFloatComplex;
typedef hipFloatComplex hipComplex;
typedef struct {
    double x;
    double y;
} hipDoubleComplex;
#define make_hipFloatComplex(r, i) ((hipFloatComplex){(float)(r), (float)(i)})
#define make_hipComplex(r, i) ((hipComplex){(float)(r), (float)(i)})
#define make_hipDoubleComplex(r, i) ((hipDoubleComplex){(double)(r), (double)(i)})
#endif

#endif

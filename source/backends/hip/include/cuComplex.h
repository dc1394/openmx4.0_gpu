#ifndef OPENMX_AMDGPU_CUCOMPLEX_COMPAT_H
#define OPENMX_AMDGPU_CUCOMPLEX_COMPAT_H

#include "hip_complex_compat.h"
typedef hipFloatComplex cuComplex;
typedef hipFloatComplex cuFloatComplex;
typedef hipDoubleComplex cuDoubleComplex;
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

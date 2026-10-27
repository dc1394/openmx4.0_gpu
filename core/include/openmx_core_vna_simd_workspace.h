#ifndef OPENMX_CORE_VNA_SIMD_WORKSPACE_H
#define OPENMX_CORE_VNA_SIMD_WORKSPACE_H
#include "openmx_complex.h"
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct OpenMXVnaWorkspace OpenMXVnaWorkspace;
typedef struct OpenMXDsvnaWorkspace OpenMXDsvnaWorkspace;

/* Mutable scratch borrowed from one worker. No row pointer tables. All
   strides/counts are in elements; angular planes retain their zero padding. */
typedef struct {
  int orders;
  double *arguments, *bessel, *bessel_derivative, *sum, *sum_derivative;
  dcomplex *harmonics; /* [3][orders][orders] */
} OpenMXVnaSimdBuffers;
typedef struct {
  int pad, llmax;
  size_t cy_stride, cy_length, plane, sum_slice, weight_stride;
  double *arguments, *weighted; /* [radial_capacity][2][points] */
  double *harmonics, *accumulators, *sums; /* [6][cy_length], [8][plane], [2][LL][sum_slice] */
  double *cpow_real, *cpow_imag, *radial, *radial_derivative; /* each [llmax+1] */
} OpenMXDsvnaSimdBuffers;

/* Lazily attach scratch to the existing worker owner. Legacy/GPU callers need
   not acquire it. Repeated acquisition with the same options allocates nothing
   and preserves values/addresses; different options are rejected. Failure
   preserves output and permits retry. Views expire with the parent workspace.
   A workspace belongs to one worker; distinct workers may acquire concurrently.
   orders is a positive multiple of 8 within the parent's angular capacity.
   angular is 0 or 1; zero omits the angular planes but retains radial scratch. */
int openmx_core_vna_simd_workspace(OpenMXVnaWorkspace *workspace, int orders,
    OpenMXVnaSimdBuffers *out, char *error, size_t capacity);
int openmx_core_dsvna_simd_workspace(OpenMXDsvnaWorkspace *workspace, int angular,
    OpenMXDsvnaSimdBuffers *out, char *error, size_t capacity);
#ifdef __cplusplus
}
#endif
#endif

#ifndef OPENMX_GPU_TRANSFORM_H
#define OPENMX_GPU_TRANSFORM_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Device-pointer operations in the exact current dense solver group.
   Return 1 after successful publication, 0 with output unchanged if the
   operation is not eligible or cannot allocate its private worker results.
   Leading dimensions and offsets are in elements. Output padding is kept. */
enum { OPENMX_GPU_ZERO, OPENMX_GPU_COPY, OPENMX_GPU_TRANSPOSE,
       OPENMX_GPU_ADD, OPENMX_GPU_SCALE_ROWS, OPENMX_GPU_SCALE_COLUMNS,
       OPENMX_GPU_HERMITIAN, OPENMX_GPU_INVSQRT };
/* Standalone native-kernel fixtures can omit the cooperative runtime. */
#if defined(__GNUC__) && !defined(OPENMX_GPU_TRANSFORM_IMPLEMENTATION)
#define OPENMX_TRANSFORM_OPTIONAL __attribute__((weak))
#else
#define OPENMX_TRANSFORM_OPTIONAL
#endif
int openmx_gpu_transform(int operation,int complex_values,int rows,int columns,
    const void *a,int lda,const void *b,int ldb,void *output,int ldout,
    int complex_weights,double floor_value) OPENMX_TRANSFORM_OPTIONAL;
/* Combine real or complex 11,22,12 spin blocks into a Hermitian spinor
   matrix. The lower off-diagonal block is the conjugate transpose of 12. */
int openmx_gpu_spinor(int n,int real_parts,const void *r11,const void *r22,
    const void *r12,const double *i11,const double *i22,const double *i12,void *out) OPENMX_TRANSFORM_OPTIONAL;
/* Sparse scatter. Indices are device buffers. format 0: one dense index;
   format 3: (value,dense,phase); format 7: (value,dense,mirror,l1,l2,l3,phase).
   Format 0 has real output, 3/7 complex. Values and phases are on device. */
int openmx_gpu_scatter(int n,int entries,int format,const int *index,
    int value_count,const double *values,int phase_count,const double *phase_r,
    const double *phase_i,int imaginary_values,void *out) OPENMX_TRANSFORM_OPTIONAL;
#undef OPENMX_TRANSFORM_OPTIONAL
#ifdef __cplusplus
}
#endif
#endif

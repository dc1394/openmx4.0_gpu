#ifndef OPENMX_CORE_HOST_SIMD_H
#define OPENMX_CORE_HOST_SIMD_H
#include "openmx_core_grid.h"
#ifdef __cplusplus
extern "C" {
#endif

/* The legacy C loops instead of the Core's host kernels. */
#define OPENMX_HOST_SIMD_LEGACY (-1)

/* Resolves the OPENMX_HOST_SIMD policy shared by the Core's host kernel
   families (overlap grids, projector expansions, forces, energies, ...):
   unset/auto/sve = the widest ISA usable here; scalar, avx2, avx512, neon,
   sve128, sve256 or sve512 = that ISA when usable, else the widest usable;
   legacy/off/0/no = the legacy loops. SVE requires an exact match with the
   calling thread's current vector length; this function never changes it.
   Never initializes Kokkos or a device, even in a GPU-default Core build. */
int openmx_core_host_simd_mode(const char *value);
/* "legacy", "scalar", "AVX2", "AVX-512", "NEON", "SVE128", "SVE256", "SVE512". */
const char *openmx_core_host_simd_name(int mode);
/* Compiled into the Core and executable by this processor and OS. */
int openmx_core_host_simd_usable(int mode);

/* Update disjoint FP64 rows. Strides and extents count doubles. The kernel
   rounds the multiplication before adding it to the destination. */
typedef void (*OpenMXHostAddRows)(double *dst, size_t dst_stride, const double *src, size_t src_stride,
                                 size_t rows, size_t columns, double factor);
/* NULL when the mode has no row kernel. Resolve on the thread using it
   for an SVE mode, because fixed-width SVE depends on that thread's VL. */
OpenMXHostAddRows openmx_core_host_add_rows(int mode);

/* Accumulate a Bloch block in a column-major complex matrix. The source
   planes are contiguous and imag may be NULL. The destination is interleaved
   doubles and ld counts complex elements. */
typedef void (*OpenMXHostBlochAdd)(double *dst, int ld, const double *real, const double *imag,
                                 int rows, int cols, double cosine, double sine);
OpenMXHostBlochAdd openmx_core_host_bloch_add(int mode);

#ifdef __cplusplus
}
#endif
#endif

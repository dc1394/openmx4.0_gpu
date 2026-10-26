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

#ifdef __cplusplus
}
#endif
#endif

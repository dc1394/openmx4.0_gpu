#ifndef OPENMX_HOST_SIMD_H
#define OPENMX_HOST_SIMD_H
#include "openmx_core_host_simd.h"
#ifdef __cplusplus
extern "C" {
#endif
/* The OPENMX_HOST_SIMD policy of this rank for the Core's host kernel
   families: OPENMX_HOST_SIMD_LEGACY (the C loops), or an OPENMX_GRID_* mode
   that this processor executes. The registry has already synchronised the
   value from rank 0, so every rank resolves the same ISA on a homogeneous
   cluster; the resolution is a few string compares and CPUID bit tests. */
int OpenMX_HostSimdMode(void);
/* Rank 0 names the resolved policy once per call site at level.of.stdout >= 2
   ("<label> host SIMD = AVX-512"). */
void OpenMX_HostSimdReport(const char *label);
/* The grid kernels of Set_Hamiltonian / Set_Density_Grid keep their own knobs
   (OPENMX_SETHAM_SIMD, OPENMX_DENSITY_GRID_COL_SIMD, _NC_SIMD, registry id
   env_id); when the knob is unset they follow OPENMX_HOST_SIMD, legacy meaning
   the scalar Core kernel (those paths have no C loop any more). */
int OpenMX_HostSimdGridMode(int env_id);
#ifdef __cplusplus
}
#endif
#endif

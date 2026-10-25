/**********************************************************************
  openmx_host_simd.c:

  The OPENMX_HOST_SIMD policy shared by the Core's host kernel families
  (Kokkos::simd kernels of the C++ core, selected per ISA at run time).
  The GPU paths are untouched; these kernels serve the host fallbacks
  (no device, device memory shortage, OPENMX_GPU=0) and the CPU build.
***********************************************************************/
#include <stdio.h>
#include "openmx_common.h"
#include "openmx_env.h"
#include "openmx_host_simd.h"
#include "openmx_core_grid.h"

int OpenMX_HostSimdMode(void)
{
  return openmx_core_host_simd_mode(openmx_env_at(OPENMX_ENV_HOST_SIMD));
}

void OpenMX_HostSimdReport(const char *label)
{
  if (MYID_MPI_COMM_WORLD!=Host_ID || level_stdout<2) return;
  printf("%s host SIMD = %s (OPENMX_HOST_SIMD)\n",label,openmx_core_host_simd_name(OpenMX_HostSimdMode()));
  fflush(stdout);
}

int OpenMX_HostSimdGridMode(int env_id)
{
  const char *value=openmx_env_at(env_id);
  if (value!=NULL && value[0]!='\0') return openmx_core_grid_simd(value);
  const int mode=OpenMX_HostSimdMode();
  return mode==OPENMX_HOST_SIMD_LEGACY ? OPENMX_GRID_SCALAR : mode;
}

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
#include "openmx_gpu_device.h"
#include "openmx_gpu_runtime.h"
#include <stdint.h>
#include "mpi.h"

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

size_t OpenMX_HostSimdDeviceBudget(size_t reserve_bytes)
{
  size_t free_bytes=0,total_bytes=0;
  if (!gpu_rank_device_usable()) return 0;
  /* backend-neutral status: cudaSuccess / hipSuccess (HIP returns int 0) */
  if ((int)OpenMX_GpuMemGetInfo(&free_bytes,&total_bytes)!=(int)OPENMX_GPU_SUCCESS) {
    (void)OpenMXGpuGetLastError();
    return 0;
  }
  return free_bytes>reserve_bytes ? free_bytes-reserve_bytes : 0;
}

int OpenMX_HostSimdStageOnHost(const char *stage, size_t need_bytes, size_t budget_bytes)
{
  const int legacy=OpenMX_HostSimdMode()==OPENMX_HOST_SIMD_LEGACY;
  const long max_pieces=openmx_env_long_at(OPENMX_ENV_HOST_SIMD_MAX_PIECES,8);
  double local[2]={0.0,0.0},all[2]={0.0,0.0};  /* host vote, device pieces */
  if (!gpu_rank_device_usable()) local[0]=1.0;
  else if (!legacy && need_bytes>0) {
    if (need_bytes==SIZE_MAX || budget_bytes==0) { local[0]=1.0; local[1]=-1.0; }
    else {
      local[1]=(double)need_bytes/(double)budget_bytes;
      /* OPENMX_HOST_SIMD_MAX_PIECES<=0: any split moves the stage to the host */
      if (1.0<local[1] && (max_pieces<=0 || (double)max_pieces<local[1])) local[0]=1.0;
    }
  }
  MPI_Allreduce(local,all,2,MPI_DOUBLE,MPI_MAX,mpi_comm_level1);
  if (0.0<all[0] && MYID_MPI_COMM_WORLD==Host_ID && 2<=level_stdout) {
    if (0.0<all[1])
      printf("<%s> the device would need up to %.0f pieces on some rank (limit %ld); host SIMD path (%s)\n",
             stage,all[1],max_pieces,openmx_core_host_simd_name(OpenMX_HostSimdMode()));
    else
      printf("<%s> the device cannot hold the stage on some rank; host SIMD path (%s)\n",
             stage,openmx_core_host_simd_name(OpenMX_HostSimdMode()));
    fflush(stdout);
  }
  return 0.0<all[0];
}

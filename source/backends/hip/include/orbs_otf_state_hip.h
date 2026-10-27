#ifndef ORBS_OTF_STATE_HIP_H
#define ORBS_OTF_STATE_HIP_H
#include "openmx_orbital_gpu_state.h"
#include "openmx_core_native_resources.h"

/* Private C/C++ POD shared by the host facade and the HIP upload/launch unit.
   Array pointers borrow the arena registered with this calculation's owner. */
typedef struct {
  int ready, double_float;
  unsigned char *arena;
  size_t bytes;
  const double *rv, *rwf, *atv;
  const size_t *rvo, *rwb;
  const int *spm, *spx, *spb;
  const float *rwd, *mcd, *spl;
  int ng23, ng3;
  double g[3][3], org[3];
} OTF_DeviceTables;

struct OpenMXOrbitalGpuState {
  OpenMXNativeResources *resources;
  int initialized, unavailable, fp32_enabled;
  int table_notice_done, fail_notice_done, device_notice_done, active_notice_done;
  OTF_DeviceTables tables;
};
#endif

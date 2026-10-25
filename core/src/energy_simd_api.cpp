#include "openmx_core_energy_simd.h"
#include "host_energy_kernels.hpp"

extern "C" int openmx_core_energy_neutral_points(const OpenMXEnergyNeutralPoints* s,int mode) {
  const auto* kernels=openmx::core::host::openmx_host_energy_kernels(mode);
  if(!kernels || !s || s->count<0)return 0;
  if(!s->count)return 1;
  // The extrapolation inside the first mesh point reads radii[2..5].
  if(s->mesh<6 || !s->log_radii || !s->radii || !s->hartree)return 0;
  if(!s->x || !s->y || !s->z || !s->density || !s->weight || !s->values)return 0;
  if(s->derivative && !s->derivatives)return 0;
  kernels->neutral_points(*s);
  return 1;
}
extern "C" int openmx_core_energy_band_block(const OpenMXEnergyBandBlock* b,int mode) {
  const auto* kernels=openmx::core::host::openmx_host_energy_kernels(mode);
  if(!kernels || !b || b->rows<0 || b->columns<0 || b->bands<0)return 0;
  if(!b->rows || !b->columns)return 1;
  if(b->stride%8 || b->stride<size_t(b->columns) || !b->cdm || !b->edm)return 0;
  if(b->bands && (!b->re0 || !b->im0 || !b->re1T || !b->im1T || !b->eigen))return 0;
  kernels->band_block(*b);
  return 1;
}

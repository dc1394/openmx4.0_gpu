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
extern "C" int openmx_core_energy_neutral_sum(const OpenMXEnergyNeutralPoints* s,double* sum,double* sumr,int mode) {
  const auto* kernels=openmx::core::host::openmx_host_energy_kernels(mode);
  if(!kernels || !s || s->count<0 || !sum || (s->derivative && !sumr))return 0;
  if(!s->count)return 1;
  if(s->mesh<6 || !s->log_radii || !s->radii || !s->hartree)return 0;
  if(!s->x || !s->y || !s->z || !s->density || !s->weight)return 0;
  kernels->neutral_sum(*s,sum,sumr);
  return 1;
}
extern "C" size_t openmx_core_energy_xc_shell_scratch(int neighbors) {
  // per-neighbour table constants, then the three gradient components of
  // every neighbour for eight lanes
  return neighbors<0?0:size_t(neighbors)*8+size_t(neighbors)*3*8;
}
extern "C" int openmx_core_energy_xc_shell(const OpenMXEnergyXcShell* s,int mode) {
  const auto* kernels=openmx::core::host::openmx_host_energy_kernels(mode);
  if(!kernels || !s || s->points<0 || s->neighbors<1)return 0;
  if(!s->den0 || !s->sumt || !s->sum_gx || !s->sum_gy || !s->sum_gz)return 0;
  if(!s->nx || !s->ny || !s->nz || !s->cutoff2 || !s->mesh || !s->log_radii || !s->radii || !s->density)return 0;
  for(int h=0;h<s->neighbors;++h)
    if(s->mesh[h]<6 || !s->log_radii[h] || !s->radii[h] || !s->density[h])return 0;
  if(!s->scratch || s->scratch_count<openmx_core_energy_xc_shell_scratch(s->neighbors))return 0;
  if(!s->points)return 1;
  if(!s->lebedev || s->lebedev_stride<4)return 0;
  kernels->xc_shell(*s);
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
extern "C" int openmx_core_energy_nc_block(const OpenMXEnergyNcBlock* b,int mode) {
  const auto* kernels=openmx::core::host::openmx_host_energy_kernels(mode);
  if(!kernels || !b || b->rows<0 || b->columns<0 || b->bands<0)return 0;
  if(!b->rows || !b->columns)return 1;
  if(b->stride%8 || b->stride<size_t(b->columns) || b->out_stride<size_t(b->columns))return 0;
  if(b->bands && !b->eigen)for(int q=6;q<10;++q)if(b->out[q])return 0;
  if(b->bands && (!b->re0_1 || !b->im0_1 || !b->re0_2 || !b->im0_2 ||
                  !b->re1T_1 || !b->im1T_1 || !b->re1T_2 || !b->im1T_2))return 0;
  kernels->nc_block(*b);
  return 1;
}

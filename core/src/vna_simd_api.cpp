#include "openmx_core_vna_simd.h"
#include "host_vna_kernels.hpp"

namespace {
const openmx::core::host::VnaKernels* kernels(int mode) { return openmx::core::host::openmx_host_vna_kernels(mode); }
}
extern "C" int openmx_core_vna_contract(const OpenMXVnaContract* p,int mode) {
  const auto* k=kernels(mode);
  if(!k || !p || p->points<0 || p->rows<0 || p->count<0)return 0;
  if(p->table_stride<size_t(p->rows) || p->out_stride<size_t(p->rows))return 0;
  if(!p->rows || !p->count)return 1;
  if(!p->out0 || !p->out1 || !p->w0 || !p->w1)return 0;
  if(p->points && !p->table)return 0;
  for(int m=0;m<p->count;++m)if(p->points && (!p->w0[m] || !p->w1[m]))return 0;
  k->contract(*p);
  return 1;
}
extern "C" int openmx_core_vna_bessel(const OpenMXVnaBessel* p,int mode) {
  const auto* k=kernels(mode);
  if(!k || !p || p->points<0 || p->lmax<0 || p->lmax>30)return 0;
  if(!p->points)return 1;
  if(!p->x || !p->sb || !p->dsb || p->stride<size_t(p->points))return 0;
  if((p->transposed || p->transposed_derivative) &&
      (!p->transposed || !p->transposed_derivative || p->transposed_stride<size_t(p->lmax)+1))return 0;
  k->bessel(*p);
  return 1;
}
extern "C" int openmx_core_vna_quadrature(const OpenMXVnaQuadrature* p,int mode) {
  const auto* k=kernels(mode);
  if(!k || !p || p->points<0 || p->orders<0 || p->table_stride<size_t(p->orders))return 0;
  if(!p->orders)return 1;
  if(!p->sum || !p->sum_derivative)return 0;
  if(p->points && (!p->product || !p->bessel || !p->bessel_derivative || !p->crude || !p->weights || !p->wavevectors))return 0;
  k->quadrature(*p);
  return 1;
}
extern "C" int openmx_core_vna_hvna_block(const OpenMXVnaHvnaBlock* p,int mode) {
  const auto* k=kernels(mode);
  if(!k || !p || p->rows<0 || p->columns<0 || p->groups<0)return 0;
  if(!p->rows || !p->columns)return 1;
  if(!p->matrix || p->stride<size_t(p->columns))return 0;
  size_t components=0;
  if(p->groups && (!p->lengths || !p->energies))return 0;
  for(int g=0;g<p->groups;++g){if(p->lengths[g]<0)return 0;components+=size_t(p->lengths[g]);}
  if(components) {
    if(!p->left || !p->right)return 0;
    if(p->right_transposed ? (p->left_stride<components || p->right_stride<size_t(p->columns))
                           : (p->left_stride<size_t(p->rows) || p->right_stride<components))return 0;
  }
  k->hvna_block(*p);
  return 1;
}

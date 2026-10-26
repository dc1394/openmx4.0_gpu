#include "openmx_core_vna_simd.h"
#include "host_vna_kernels.hpp"
#include <algorithm>

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
    const bool doubles=!p->fp32 && p->left_double && p->right_double;
    if(!doubles && (!p->left || !p->right))return 0;
    if(p->right_transposed ? (p->left_stride<components || p->right_stride<size_t(p->columns))
                           : (p->left_stride<size_t(p->rows) || p->right_stride<components))return 0;
  }
  k->hvna_block(*p);
  return 1;
}
extern "C" int openmx_core_vna_angular(const OpenMXVnaAngular* p,int mode) {
  const auto* k=kernels(mode);
  if(!k || !p || p->l0<0 || p->l1<0 || p->rows!=2*p->l0+1 || p->columns!=2*p->l1+1)return 0;
  if(p->ll_first<0 || p->ll_count<0 || p->pad<p->l0+p->l1)return 0;
  if(p->gaunt_stride<size_t(p->columns) || p->acc_stride<size_t(p->columns))return 0;
  for(int c=0;c<8;++c)if(!p->acc[c])return 0;
  if(p->ll_count) {
    const int top=p->ll_first+p->ll_count-1;
    const int reach=std::max(2*top+1,top+p->l0+p->l1+8);
    if(p->cy_stride<size_t(p->pad+reach) || p->cy_length<size_t(top+1)*p->cy_stride)return 0;
    if(!p->cpow_r || !p->cpow_i || !p->s0 || !p->s1 || !p->gaunt)return 0;
    for(int c=0;c<6;++c)if(!p->cy[c])return 0;
    for(int t=0;t<p->ll_count;++t)if(!p->gaunt[t])return 0;
  }
  k->angular(*p);
  return 1;
}
extern "C" int openmx_core_vna_convert(const OpenMXVnaConvert* p,int mode) {
  const auto* k=kernels(mode);
  if(!k || !p || p->rows0<1 || p->rows0>16 || p->columns<1 || p->columns>32)return 0;
  if(p->acc_stride<size_t(p->columns) || p->width<size_t(p->rows0) || p->ld<size_t(p->columns) || p->out_stride<size_t(p->columns))return 0;
  if(!p->conv0 || !p->convt_r || !p->convt_i)return 0;
  for(int c=0;c<8;++c)if(!p->acc[c])return 0;
  for(int c=0;c<4;++c)if(!p->out[c])return 0;
  k->convert(*p);
  return 1;
}
extern "C" int openmx_core_vna_overlap_radial(const OpenMXVnaOverlapRadial* p,int mode) {
  const auto* k=kernels(mode);
  if(!k || !p || p->points<0 || p->orders<0 || p->combos<0)return 0;
  if(!p->orders || !p->combos)return 1;
  for(int q=0;q<4;++q)if(!p->out[q])return 0;
  if(p->out_stride<size_t(p->combos))return 0;
  if(p->points) {
    if(!p->k || !p->rf0t || !p->rf1t || !p->sb || !p->sbp)return 0;
    if(p->table_stride<size_t(p->combos) || p->sb_stride<size_t(p->points))return 0;
  }
  k->overlap_radial(*p);
  return 1;
}
extern "C" int openmx_core_vna_nonlocal_radial(const OpenMXVnaNonlocalRadial* p,int mode) {
  const auto* k=kernels(mode);
  if(!k || !p || p->points<0 || p->orders<0 || p->combos<0)return 0;
  if(!p->orders || !p->combos)return 1;
  for(int q=0;q<2;++q)if(!p->out[q])return 0;
  if(p->out_stride<size_t(p->combos))return 0;
  if(p->points) {
    if(!p->k || !p->rft || !p->nlrft || !p->sb || !p->sbp)return 0;
    if(p->table_stride<size_t(p->combos) || p->sb_stride<size_t(p->points))return 0;
  }
  k->nonlocal_radial(*p);
  return 1;
}

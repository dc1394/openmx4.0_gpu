#include "openmx_core_force_simd.h"
#include "host_force_kernels.hpp"
#include <cmath>
#include <cstdint>

// Validation of every force-family request; the kernels trust their input.
namespace {
using openmx::core::host::force_padded;
bool rows_present(const float* const* rows,int count) {
  if(count<=0)return true;
  if(!rows)return false;
  for(int i=0;i<count;++i)if(!rows[i])return false;
  return true;
}
}
extern "C" size_t openmx_core_force_vna_scratch(int columns,int projectors,int terms) {
  if(columns<0 || projectors<0 || (terms!=1 && terms!=2))return 0;
  return size_t(terms==2?4:1)*size_t(projectors)*force_padded(columns)*sizeof(float);
}
extern "C" int openmx_core_force_vna_block(const OpenMXForceVnaBlock* b,void* scratch,size_t scratch_bytes,int mode) {
  const auto* kernels=openmx::core::host::openmx_host_force_kernels(mode);
  if(!kernels || !b || b->rows<0 || b->columns<0 || b->projectors<0 || (b->terms!=1 && b->terms!=2))return 0;
  if(size_t(b->columns)>b->output_stride && b->rows && b->columns)return 0;
  const bool active=b->rows && b->columns;
  if(active) {
    for(int c=0;c<3;++c)if(!b->output[c])return 0;
    if(b->projectors) {
      for(int c=0;c<3;++c)if(!rows_present(b->derivative[c],b->rows))return 0;
      if(!rows_present(b->value,b->columns))return 0;
      if(b->terms==2) {
        if(!rows_present(b->row_value,b->rows))return 0;
        for(int c=0;c<3;++c)if(!rows_present(b->column_derivative[c],b->columns))return 0;
      }
    }
    const size_t needed=openmx_core_force_vna_scratch(b->columns,b->projectors,b->terms);
    if(needed && (!scratch || scratch_bytes<needed))return 0;
  }
  kernels->vna_block(*b,scratch);
  return 1;
}
extern "C" size_t openmx_core_force_grid_scratch(int rows,int columns,int spins) {
  if(rows<0 || columns<0 || spins<1 || spins>4)return 0;
  return (size_t(spins)*size_t(columns)+1)*force_padded(rows)*sizeof(double);
}
extern "C" int openmx_core_force_grid_pair(const OpenMXForceGridPair* g,double sums[3],void* scratch,size_t scratch_bytes,int mode) {
  const auto* kernels=openmx::core::host::openmx_host_force_kernels(mode);
  if(!kernels || !g || !sums || g->rows<0 || g->columns<0 || g->points<0 || g->spins<1 || g->spins>4)return 0;
  if(g->points && g->rows) {
    if(!g->center || !g->derivative || (g->columns && !g->neighbor_rows))return 0;
    for(int s=0;s<g->spins;++s) {
      if(!g->potential || !g->potential[s])return 0;
      if(!g->density[s])return 0;
      for(int i=0;i<g->rows;++i)if(!g->density[s][i])return 0;
    }
    const size_t needed=openmx_core_force_grid_scratch(g->rows,g->columns,g->spins);
    if(!scratch || scratch_bytes<needed)return 0;
  }
  kernels->grid_pair(*g,sums,scratch);
  return 1;
}
extern "C" size_t openmx_core_force_nl_scratch(int columns,size_t components,size_t terms) {
  if(columns<0 || terms>2)return 0;
  return terms*3*components*force_padded(columns)*sizeof(double);
}
extern "C" int openmx_core_force_nl_block(const OpenMXForceProjectorSpecies* sp,const OpenMXProjectorForceBlock* b,
    void* scratch,size_t scratch_bytes,int mode) {
  const auto* kernels=openmx::core::host::openmx_host_force_kernels(mode);
  if(!kernels || !sp || !b || sp->relativistic || b->term_count>2 || !b->terms)return 0;
  if(b->rows>b->output_rows || b->columns>b->output_stride || b->columns>size_t(INT32_MAX))return 0;
  if(sp->count && !sp->channels)return 0;
  for(size_t j=0;j<sp->count;++j) {
    const auto& channel=sp->channels[j];
    if(channel.twice_angular<0 || channel.offset+size_t(channel.twice_angular)>=sp->components)return 0;
  }
  const bool active=b->rows && b->columns;
  if(active)for(int axis=0;axis<3;++axis)if(!b->output[axis])return 0;
  for(size_t k=0;k<b->term_count;++k) {
    const auto& term=b->terms[k];
    if(!(term.sign==1.0 || term.sign==-1.0) || (term.reverse_energy!=0 && term.reverse_energy!=1) ||
        (term.transpose!=0 && term.transpose!=1) || (term.value_first!=0 && term.value_first!=1))return 0;
    if(term.derivative_stride<sp->components || term.value_stride<sp->components)return 0;
    if(term.derivative_rows<(term.transpose?b->columns:b->rows) || term.value_rows<(term.transpose?b->rows:b->columns))return 0;
    if(sp->components && active) {
      for(int axis=0;axis<3;++axis)if(!term.derivative[0][axis])return 0;
      if(!term.value[0])return 0;
    }
  }
  if(active) {
    const size_t needed=openmx_core_force_nl_scratch(int(b->columns),sp->components,b->term_count);
    if(needed && (!scratch || scratch_bytes<needed))return 0;
  }
  kernels->nl_block(*sp,*b,scratch);
  return 1;
}

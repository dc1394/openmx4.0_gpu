#include "openmx_core_force_simd.h"
#include "host_force_kernels.hpp"
#include <cmath>
#include <cstdint>

// Validation of every force-family request; the kernels trust their input.
namespace {
using openmx::core::host::force_padded;
using openmx::core::host::force_vna_planes;
bool rows_present(const float* const* rows,int count) {
  if(count<=0)return true;
  if(!rows)return false;
  for(int i=0;i<count;++i)if(!rows[i])return false;
  return true;
}
}
extern "C" size_t openmx_core_force_vna_lanes_bytes(int count,int projectors,int components) {
  if(count<0 || projectors<0 || components<1 || components>4)return 0;
  return size_t(components)*size_t(projectors)*force_padded(count)*sizeof(float);
}
extern "C" int openmx_core_force_vna_lanes(const float* const* const* rows,int components,int count,int projectors,
    void* memory,size_t bytes,OpenMXForceVnaLanes* lanes,int mode) {
  const auto* kernels=openmx::core::host::openmx_host_force_kernels(mode);
  if(!kernels || !lanes || count<0 || projectors<0 || components<1 || components>4)return 0;
  const size_t needed=openmx_core_force_vna_lanes_bytes(count,projectors,components);
  if(needed && (!memory || bytes<needed))return 0;
  if(count && projectors) {
    if(!rows)return 0;
    for(int c=0;c<components;++c)if(!rows_present(rows[c],count))return 0;
  }
  float* planes=static_cast<float*>(memory);
  if(needed)kernels->vna_lanes(rows,components,count,projectors,planes,force_padded(count));
  lanes->planes=planes;lanes->count=count;lanes->projectors=projectors;lanes->components=components;lanes->pad=force_padded(count);
  return 1;
}
extern "C" size_t openmx_core_force_vna_scratch(int lane_count,int projectors,int terms,int lanes_rows) {
  if(lane_count<0 || projectors<0 || (terms!=1 && terms!=2) || (lanes_rows!=0 && lanes_rows!=1))return 0;
  return size_t(force_vna_planes(terms,lanes_rows))*size_t(projectors)*force_padded(lane_count)*sizeof(float);
}
extern "C" int openmx_core_force_vna_block(const OpenMXForceVnaBlock* b,void* scratch,size_t scratch_bytes,int mode) {
  const auto* kernels=openmx::core::host::openmx_host_force_kernels(mode);
  if(!kernels || !b || b->rows<0 || b->columns<0 || b->projectors<0 || (b->terms!=1 && b->terms!=2) ||
      (b->lanes_rows!=0 && b->lanes_rows!=1))return 0;
  const bool active=b->rows && b->columns;
  const int lane_count=b->lanes_rows?b->rows:b->columns;
  if(active && size_t(lane_count)>b->output_stride)return 0;
  if(active) {
    for(int c=0;c<3;++c)if(!b->output[c])return 0;
    if(b->projectors) {
      const bool need_rows=b->lanes==nullptr || b->lanes_rows==0;      // derivative rows streamed or transposed here
      const bool need_columns=b->lanes==nullptr || b->lanes_rows==1;   // value rows streamed or transposed here
      if(need_rows)for(int c=0;c<3;++c)if(!rows_present(b->derivative[c],b->rows))return 0;
      if(need_columns && !rows_present(b->value,b->columns))return 0;
      if(b->terms==2) {
        if(need_rows && !rows_present(b->row_value,b->rows))return 0;
        if(need_columns)for(int c=0;c<3;++c)if(!rows_present(b->column_derivative[c],b->columns))return 0;
      }
    }
    if(b->lanes) {
      const auto& l=*b->lanes;
      if(!l.planes || l.count!=lane_count || l.projectors!=b->projectors || l.pad!=force_padded(lane_count) ||
          l.components<force_vna_planes(b->terms,b->lanes_rows))return 0;
    }else {
      const size_t needed=openmx_core_force_vna_scratch(lane_count,b->projectors,b->terms,b->lanes_rows);
      if(needed && (!scratch || scratch_bytes<needed))return 0;
    }
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
extern "C" size_t openmx_core_force_grid_density_bytes(int rows,int columns_max,int spins,int pairs) {
  if(rows<0 || columns_max<0 || spins<1 || spins>4 || pairs<0)return 0;
  return size_t(pairs)*size_t(spins)*size_t(columns_max)*force_padded(rows)*sizeof(double);
}
extern "C" int openmx_core_force_grid_density(const double* const* density[4],int rows,int columns,int columns_max,int spins,
    int pair,double* dm,size_t dm_bytes,int mode) {
  const auto* kernels=openmx::core::host::openmx_host_force_kernels(mode);
  if(!kernels || rows<0 || columns<0 || columns>columns_max || spins<1 || spins>4 || pair<0 || !density)return 0;
  const size_t pad=force_padded(rows),plane=size_t(spins)*size_t(columns_max)*pad;
  if(!plane)return 1;
  if(!dm || dm_bytes<(size_t(pair)+1)*plane*sizeof(double))return 0;
  if(rows)for(int s=0;s<spins;++s) {
    if(!density[s])return 0;
    for(int i=0;i<rows;++i)if(!density[s][i])return 0;
  }
  kernels->grid_density(density,rows,columns,columns_max,spins,dm+size_t(pair)*plane,pad);
  return 1;
}
extern "C" size_t openmx_core_force_grid_points_scratch(int rows) {
  return rows<0?0:force_padded(rows)*sizeof(double);
}
extern "C" int openmx_core_force_grid_points(const OpenMXForceGridPoints* g,int first,int count,void* scratch,size_t scratch_bytes,int mode) {
  const auto* kernels=openmx::core::host::openmx_host_force_kernels(mode);
  if(!kernels || !g || g->rows<0 || g->points<0 || g->columns_max<0 || g->spins<1 || g->spins>4 || first<0 || count<0 ||
      first>g->points || count>g->points-first)return 0;
  if(g->pad_rows!=force_padded(g->rows) || g->row_stride<3 || g->point_stride<size_t(g->rows)*g->row_stride)return 0;
  if(!count)return 1;
  if(!g->point_offsets || !g->potential || !g->terms || !g->derivative)return 0;
  for(int s=0;s<g->spins;++s)if(!g->potential[s])return 0;
  const size_t visits=g->point_offsets[first+count]-g->point_offsets[first];
  if(visits && (!g->visit_pair || !g->visit_row || !g->visit_term || !g->pair_rows || !g->pair_columns || (g->rows && !g->dm)))return 0;
  const size_t needed=openmx_core_force_grid_points_scratch(g->rows);
  if(!scratch || scratch_bytes<needed)return 0;
  kernels->grid_points(*g,first,count,scratch);
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

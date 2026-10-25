#include "host_grid_kernels.hpp"
#include "host_simd_abi.hpp"
#include <algorithm>
// The explicit host ABIs, the lane counts and load/store come from
// host_simd_abi.hpp, shared by every per-ISA translation unit of this library.
#define OPENMX_GRID_FACTORY OPENMX_HOST_SIMD_FACTORY(grid)

namespace openmx::core::host {
namespace {
template<class T> double orbital(const OpenMXGridOrbitals& o,int point,int index) {
  const auto rows=static_cast<const T* const*>(o.rows);
  return o.transposed?double(rows[index][point]):double(rows[point][index]);
}
double orbital(const OpenMXGridOrbitals& o,int point,int index) {
  return o.element_bytes==int(sizeof(float))?orbital<float>(o,point,index):orbital<double>(o,point,index);
}

// Each lane owns one point; j, i and repeated-center scatter order are exact.
template<int Spins>
void density(const OpenMXGridPair& p,GridScratch w,double* const* output) {
  for(int i=0;i<p.rows;++i)for(int j=0;j<p.columns;++j)for(int s=0;s<Spins;++s)
    w.dm[(size_t(i)*p.columns+j)*Spins+s]=p.matrix[s][i][j];
  for(int first=0;first<p.points;) {
    const int count=std::min(lanes,p.points-first);
    for(int i=0;i<p.rows;++i)for(int g=0;g<lanes;++g)
      w.orb0[size_t(i)*lanes+g]=g<count?orbital(p.center,p.center_points[first+g],i):0.0;
    for(int j=0;j<p.columns;++j)for(int g=0;g<lanes;++g)
      w.orb1[size_t(j)*lanes+g]=g<count?orbital(p.neighbor,
          p.neighbor.overlap_rows?first+g:p.neighbor_points[first+g],j):0.0;
    Vector sum[Spins];for(auto& v:sum)v=Vector(0.0);
    for(int i=0;i<p.rows;++i) {
      Vector tmp[Spins];for(auto& v:tmp)v=Vector(0.0);
      for(int j=0;j<p.columns;++j) {
        const auto b=load(w.orb1+size_t(j)*lanes);
        for(int s=0;s<Spins;++s)tmp[s]+=Vector(w.dm[(size_t(i)*p.columns+j)*Spins+s])*b;
      }
      const auto a=load(w.orb0+size_t(i)*lanes);
      for(int s=0;s<Spins;++s)sum[s]+=a*tmp[s];
    }
    for(int s=0;s<Spins;++s) {
      double values[lanes];store(sum[s],values);
      for(int g=0;g<count;++g)output[s][p.center_points[first+g]]+=values[g];
    }
    first+=count;
  }
}

// SIMD spans independent matrix columns, never the quadrature reduction.
template<int Chunks>
void accumulate_tile(double* row,const double* b,size_t stride,const double* a,int count) {
  Vector value[Chunks];
  for(int c=0;c<Chunks;++c)value[c]=load(row+c*lanes);
  for(int g=0;g<count;++g) {
    const auto weight=Vector(a[g]);
    for(int c=0;c<Chunks;++c)value[c]+=weight*load(b+g*stride+c*lanes);
  }
  for(int c=0;c<Chunks;++c)store(value[c],row+c*lanes);
}
template<int Spins>
void hamiltonian(const OpenMXGridPair& p,GridScratch w) {
  if(!p.points || !p.rows || !p.columns)return;
  const bool blocked=Spins==1 && p.rows<=16 && p.columns<=16 && !p.center.transposed;
  const size_t stride=blocked?16:w.orbitals;
  // A fixed row stride does not require computing every padded column.
  // Small H/C blocks need only their active host SIMD vectors; retain the
  // stride and ascending quadrature order of every physical matrix entry.
  const size_t columns=(size_t(p.columns)+lanes-1)/lanes*lanes;
  for(int s=0;s<Spins;++s)for(int i=0;i<p.rows;++i)for(size_t j=0;j<stride;++j)
    w.h[(size_t(s)*p.rows+i)*stride+j]=blocked || j>=size_t(p.columns)?0.0:p.matrix[s][i][j];
  // Scratch already owns a multiple-of-eight row. Zero only the padding,
  // allowing full-width updates without reading beyond borrowed input rows.
  for(size_t g=0;g<16;++g)
    std::fill(w.orb1+g*w.orbitals+p.columns,w.orb1+g*w.orbitals+columns,0.0);
  for(int first=0;first<p.points;) {
    const int count=std::min(16,p.points-first);
    for(int g=0;g<count;++g) {
      const int nc=p.center_points[first+g];
      const int nh=p.neighbor.overlap_rows?first+g:p.neighbor_points[first+g];
      for(int i=0;i<p.rows;++i)w.orb0[g*w.orbitals+i]=orbital(p.center,nc,i);
      for(int j=0;j<p.columns;++j)w.orb1[g*w.orbitals+j]=orbital(p.neighbor,nh,j);
    }
    double potential[Spins][16];
    for(int s=0;s<Spins;++s)for(int g=0;g<count;++g)
      potential[s][g]=p.volume*p.potential[s][p.potential_index[p.center_points[first+g]]];
    for(int s=0;s<Spins;++s)for(int i=0;i<p.rows;++i) {
      double* row=w.h+(size_t(s)*p.rows+i)*stride;
      double a[16];for(int g=0;g<count;++g)a[g]=potential[s][g]*w.orb0[g*w.orbitals+i];
      size_t j=0;
      // Keep the output vector in a register across the 16-point tile. Each
      // element still sees ascending grid points, with no horizontal sum/FMA.
      for(;j+4*lanes<=columns;j+=4*lanes)
        accumulate_tile<4>(row+j,w.orb1+j,w.orbitals,a,count);
      if(j+2*lanes<=columns) {
        accumulate_tile<2>(row+j,w.orb1+j,w.orbitals,a,count);j+=2*lanes;
      }
      if(j<columns)accumulate_tile<1>(row+j,w.orb1+j,w.orbitals,a,count);
    }
    first+=count;
  }
  for(int s=0;s<Spins;++s)for(int i=0;i<p.rows;++i)for(int j=0;j<p.columns;++j) {
    if(blocked)p.matrix[s][i][j]+=w.h[(size_t(s)*p.rows+i)*stride+j];
    else p.matrix[s][i][j]=w.h[(size_t(s)*p.rows+i)*stride+j];
  }
}
void density_dispatch(const OpenMXGridPair& p,GridScratch w,double* const* out,int spins) {
  if(spins==1)density<1>(p,w,out);else if(spins==2)density<2>(p,w,out);else density<4>(p,w,out);
}
void hamiltonian_dispatch(const OpenMXGridPair& p,GridScratch w,int spins) {
  if(spins==1)hamiltonian<1>(p,w);else if(spins==2)hamiltonian<2>(p,w);else hamiltonian<4>(p,w);
}
template<int Chunks>
void packed_columns(const OpenMXGridPackedHamiltonian& p,int spin,int i,int j,double* row) {
  Vector value[Chunks];for(auto& v:value)v=Vector(0.0);
  for(int k=0;k<p.points;++k) {
    const double weight=p.volume*p.potential[size_t(k)*p.spins+spin];
    const auto a=Vector(weight*p.left[size_t(k)*p.rows+i]);
    for(int c=0;c<Chunks;++c)value[c]+=a*load(p.right+size_t(k)*p.columns+j+c*lanes);
  }
  for(int c=0;c<Chunks;++c)store(value[c],row+j+c*lanes);
}
void packed_hamiltonian(const OpenMXGridPackedHamiltonian& p) {
  const size_t plane=p.output_stride*p.output_stride;
  if(!plane)return;
  std::fill_n(p.output,size_t(p.spins)*plane,0.0);
  for(int spin=0;spin<p.spins;++spin)for(int i=0;i<p.rows;++i) {
    double* row=p.output+size_t(spin)*plane+size_t(i)*p.output_stride;
    int j=0;
    for(;j<=p.columns-4*lanes;j+=4*lanes)packed_columns<4>(p,spin,i,j,row);
    if(j<=p.columns-2*lanes){packed_columns<2>(p,spin,i,j,row);j+=2*lanes;}
    if(j<=p.columns-lanes){packed_columns<1>(p,spin,i,j,row);j+=lanes;}
    if(j<p.columns) {
      Vector value(0.0);const int count=p.columns-j;
      const typename Vector::mask_type mask([&](auto lane){return int(lane)<count;});
      for(int k=0;k<p.points;++k) {
        const double weight=p.volume*p.potential[size_t(k)*p.spins+spin];
        const auto a=Vector(weight*p.left[size_t(k)*p.rows+i]);
        const auto b=simd::simd_partial_load<Vector>(p.right+size_t(k)*p.columns+j,mask);
        value+=a*b;
      }
      double values[lanes];store(value,values);
      for(int lane=0;lane<count;++lane)row[j+lane]=values[lane];
    }
  }
}
constexpr GridKernels kernels{lanes,density_dispatch,hamiltonian_dispatch,packed_hamiltonian};
}
const GridKernels& OPENMX_GRID_FACTORY() noexcept {return kernels;}
}

#include "host_grid_kernels.hpp"
#include "host_simd_abi.hpp"
#include <algorithm>
// The explicit host ABIs, the lane counts and load/store come from
// host_simd_abi.hpp, shared by every per-ISA translation unit of this library.
#define OPENMX_GRID_FACTORY OPENMX_HOST_SIMD_FACTORY(grid)

#include <type_traits>

namespace openmx::core::host {
namespace {
// Lane vectors (one per orbital, points across lanes) of a block of points.
// The block's row pointers (or column indices of a transposed table) are
// resolved once per block, outside the orbital loop; masked-off lanes are
// zero, as before. Element gathers were measured slower than this loop on
// Skylake-X for every shape (the rows of a block sit in L1/L2).
// The helpers stay inline in their kernels (an out-of-line call per point
// costs more than a short row's conversion).
template<class T>
[[gnu::always_inline]] inline void fill_lanes(const OpenMXGridOrbitals& o,const int* points,bool positional,int first,
    int count,int n,double* dst) {
  const auto rows=static_cast<const T* const*>(o.rows);
  int index[lanes];
  for(int g=0;g<lanes;++g)index[g]=g<count?(positional?first+g:points[g]):0;
  if(o.transposed) {
    for(int i=0;i<n;++i)for(int g=0;g<lanes;++g)
      dst[size_t(i)*lanes+g]=g<count?double(rows[i][index[g]]):0.0;
    return;
  }
  const T* row[lanes];
  for(int g=0;g<lanes;++g)row[g]=rows[index[g]];
  for(int i=0;i<n;++i)for(int g=0;g<lanes;++g)
    dst[size_t(i)*lanes+g]=g<count?double(row[g][i]):0.0;
}
[[gnu::always_inline]] inline void fill_lanes(const OpenMXGridOrbitals& o,const int* points,bool positional,
    int first,int count,int n,double* dst) {
  if(o.element_bytes==int(sizeof(float)))fill_lanes<float>(o,points,positional,first,count,n,dst);
  else fill_lanes<double>(o,points,positional,first,count,n,dst);
}
// One point's row of n orbitals as doubles (point-major scratch): whole
// vectors are converted at once (float rows ride the double ABI's lane
// count; the conversion is exact), the tail and transposed tables element by
// element.
using FloatLanes=simd::basic_simd<float,HostAbi>;
template<class T> using LanesOf=std::conditional_t<std::is_same_v<T,float>,FloatLanes,Vector>;
template<class T> [[gnu::always_inline]] inline void copy_row(const OpenMXGridOrbitals& o,int point,int n,double* dst) {
  const auto rows=static_cast<const T* const*>(o.rows);
  if(o.transposed){for(int i=0;i<n;++i)dst[i]=double(rows[i][point]);return;}
  const T* row=rows[point];
  int i=0;
  if constexpr(lanes>1)for(;i+lanes<=n;i+=lanes)store(Vector(simd::simd_unchecked_load<LanesOf<T>>(row+i)),dst+i);
  for(;i<n;++i)dst[i]=double(row[i]);
}
[[gnu::always_inline]] inline void copy_row(const OpenMXGridOrbitals& o,int point,int n,double* dst) {
  if(o.element_bytes==int(sizeof(float)))copy_row<float>(o,point,n,dst);else copy_row<double>(o,point,n,dst);
}

// Lanes across density-matrix rows i instead of points. Per point the j-sum
// of every row runs in its own lane in ascending j and the i-sum stays a
// sequential scalar loop, which is exactly the per-point arithmetic of the
// lane-per-point kernel below, so both give bitwise the same density. Rows
// are converted with whole vectors; no per-point transposition is needed.
// Blocks of Points points share the density-matrix loads and interleave
// their scalar i-sums; the output adds keep ascending point order.
template<int Spins,int Points>
#if defined(OPENMX_HOST_SIMD_NEON) || (defined(OPENMX_HOST_SIMD_SVE) && OPENMX_HOST_SIMD_SVE == 128)
// GCC 12 inlining into the shape dispatcher regressed SVE128 single-spin
// timings. Keep this sizable kernel independent of the dispatch branches.
[[gnu::noinline]]
#endif
void density_rowwise(const OpenMXGridPair& p,GridScratch w,double* const* output) {
  const int chunks=(p.rows+lanes-1)/lanes;
  const size_t pad=size_t(chunks)*lanes;
  // dm[s][j][pad] (zero beyond the rows); pad <= w.orbitals, so it fits.
  for(int s=0;s<Spins;++s)for(int j=0;j<p.columns;++j) {
    double* column=w.dm+(size_t(s)*p.columns+j)*pad;
    for(int i=0;i<p.rows;++i)column[i]=p.matrix[s][i][j];
    for(size_t i=size_t(p.rows);i<pad;++i)column[i]=0.0;
  }
  double* const a=w.orb0;  // [Points][w.orbitals]
  double* const b=w.orb1;  // [Points][w.orbitals]
  double* const t=w.h;     // [Points][Spins][pad]
  for(int first=0;first<p.points;first+=Points) {
    const int count=std::min(Points,p.points-first);
    int nc[Points];
    for(int q=0;q<count;++q) {
      nc[q]=p.center_points[first+q];
      const int nh=p.neighbor.overlap_rows?first+q:p.neighbor_points[first+q];
      copy_row(p.center,nc[q],p.rows,a+size_t(q)*w.orbitals);
      copy_row(p.neighbor,nh,p.columns,b+size_t(q)*w.orbitals);
    }
    for(int c=0;c<chunks;++c) {
      Vector tmp[Points][Spins];
      for(auto& point:tmp)for(auto& v:point)v=Vector(0.0);
      for(int j=0;j<p.columns;++j) {
        Vector column[Spins];
        for(int s=0;s<Spins;++s)column[s]=load(w.dm+(size_t(s)*p.columns+j)*pad+size_t(c)*lanes);
        for(int q=0;q<Points;++q) {
          if(q>=count)break;
          const Vector weight(b[size_t(q)*w.orbitals+j]);
          for(int s=0;s<Spins;++s)tmp[q][s]+=column[s]*weight;
        }
      }
      for(int q=0;q<count;++q)for(int s=0;s<Spins;++s)
        store(tmp[q][s],t+(size_t(q)*Spins+s)*pad+size_t(c)*lanes);
    }
    for(int s=0;s<Spins;++s) {
      double sum[Points];
      for(int q=0;q<Points;++q)sum[q]=0.0;
      for(int i=0;i<p.rows;++i)for(int q=0;q<Points;++q) {
        if(q>=count)break;
        sum[q]+=a[size_t(q)*w.orbitals+i]*t[(size_t(q)*Spins+s)*pad+i];
      }
      for(int q=0;q<count;++q)output[s][nc[q]]+=sum[q];
    }
  }
}

// Each lane owns one point; j, i and repeated-center scatter order are exact.
template<int Spins>
void density(const OpenMXGridPair& p,GridScratch w,double* const* output) {
  for(int i=0;i<p.rows;++i)for(int j=0;j<p.columns;++j)for(int s=0;s<Spins;++s)
    w.dm[(size_t(i)*p.columns+j)*Spins+s]=p.matrix[s][i][j];
  for(int first=0;first<p.points;) {
    const int count=std::min(lanes,p.points-first);
    fill_lanes(p.center,p.center_points+first,false,first,count,p.rows,w.orb0);
    fill_lanes(p.neighbor,p.neighbor_points+first,p.neighbor.overlap_rows!=0,first,count,p.columns,w.orb1);
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
      copy_row(p.center,nc,p.rows,w.orb0+g*w.orbitals);
      copy_row(p.neighbor,nh,p.columns,w.orb1+g*w.orbitals);
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
// The committed kernels of this ISA, compiled in their own translation unit.
const GridKernels& committed() noexcept {return OPENMX_HOST_SIMD_FACTORY(grid_committed)();}
// Every kernel here and the committed ones give bitwise the same result, so
// the choice is only about speed. Measured per ISA on an i9-10980XE against the committed kernels
// (2000-point pairs; rows 1-19 x columns 1-19, spins 1/2/4; float rows):
// a shape takes a new kernel only where it was faster, otherwise the
// committed one. The NEON/SVE128 choices below were measured separately on
// Neoverse-N3; wider SVE variants retain their existing policy.
enum class DensityKernel {committed,points,rows2,rows4};
DensityKernel density_kernel(int rows,int columns,int spins) {
#if defined(OPENMX_HOST_SIMD_AVX512)
  // points across lanes: 1.13-4x everywhere; rows across lanes from 8 rows
  // when the neighbour block is wide.
  return spins==1 && rows>=8 && columns>=16?DensityKernel::rows4:DensityKernel::points;
#elif defined(OPENMX_HOST_SIMD_AVX2)
  // points across lanes: 1.04-1.6x everywhere.
  if(spins==2 && rows>=8 && columns>=13)return DensityKernel::rows4;
  if(spins==1 && rows>=16 && columns>=13)return DensityKernel::rows2;
  return DensityKernel::points;
#elif defined(OPENMX_HOST_SIMD_NEON)
  if(rows==1 && columns==1)return DensityKernel::committed;
  // Hoist row-pointer resolution out of the orbital loop. Four points share
  // the matrix loads for larger single-spin blocks; narrow and multi-spin
  // blocks benefit more from keeping points across the two SIMD lanes.
  if(spins==1 && ((rows>=8 && columns>=8) || (rows>=4 && columns>=12)))
    return DensityKernel::rows4;
  return DensityKernel::points;
#elif defined(OPENMX_HOST_SIMD_SVE) && OPENMX_HOST_SIMD_SVE == 128
  if(rows==1 && columns==1)return DensityKernel::committed;
  if(spins==1 && (rows>=8 || (rows>=4 && columns>=12)))return DensityKernel::rows4;
  return DensityKernel::points;
#else
  // the scalar build gains from interleaved rows (several points per pass)
  if(spins==1)return columns>=2?DensityKernel::rows4:DensityKernel::committed;
  if(spins==2) {
    if(columns>=13)return DensityKernel::rows4;
    return columns>=8 && rows>=4?DensityKernel::rows2:DensityKernel::committed;
  }
  return rows>=8 && columns<=4?DensityKernel::points:DensityKernel::committed;
#endif
}
bool hamiltonian_rows(int rows,int columns) {
#if defined(OPENMX_HOST_SIMD_AVX512)
  // whole-vector row conversion pays from 8 orbitals; narrow blocks keep
  // the committed element loop.
  return !(columns<=2 || (rows<8 && columns<8));
#elif defined(OPENMX_HOST_SIMD_AVX2)
  return !((rows<8 && columns<=2) || (rows<=3 && columns<=5));
#elif defined(OPENMX_HOST_SIMD_NEON) || (defined(OPENMX_HOST_SIMD_SVE) && OPENMX_HOST_SIMD_SVE == 128)
  // Whole-vector FP32->FP64 row copies pay for their setup beyond tiny
  // blocks on Neoverse-N3; retain the element loop for the small cases.
  return rows>=4 || columns>=4;
#else
  // one-lane builds have no vector conversion to gain
  (void)rows;(void)columns;return false;
#endif
}
void density_dispatch(const OpenMXGridPair& p,GridScratch w,double* const* out,int spins) {
  switch(density_kernel(p.rows,p.columns,spins)) {
    case DensityKernel::rows2:
      if(spins==1)density_rowwise<1,2>(p,w,out);else if(spins==2)density_rowwise<2,2>(p,w,out);
      else density_rowwise<4,2>(p,w,out);
      return;
    case DensityKernel::rows4:
      if(spins==1)density_rowwise<1,4>(p,w,out);else if(spins==2)density_rowwise<2,4>(p,w,out);
      else density_rowwise<4,4>(p,w,out);
      return;
    case DensityKernel::points:
      if(spins==1)density<1>(p,w,out);else if(spins==2)density<2>(p,w,out);else density<4>(p,w,out);
      return;
    case DensityKernel::committed:
      committed().density(p,w,out,spins);
      return;
  }
}
void hamiltonian_dispatch(const OpenMXGridPair& p,GridScratch w,int spins) {
  if(hamiltonian_rows(p.rows,p.columns)) {
    if(spins==1)hamiltonian<1>(p,w);else if(spins==2)hamiltonian<2>(p,w);else hamiltonian<4>(p,w);
    return;
  }
  committed().hamiltonian(p,w,spins);
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

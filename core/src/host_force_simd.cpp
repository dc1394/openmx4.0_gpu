#include "host_force_kernels.hpp"
#include "host_simd_abi.hpp"
#include <algorithm>
#include <cstring>

// Force-stage host kernels. Lanes always span independent output elements
// (matrix columns n, or density-matrix rows i); every reduction runs in the
// legacy order, and every product rounds where the legacy C expression
// rounds (float products before widening when fp32 is set).
namespace openmx::core::host {
namespace {
using NarrowFloat=simd::basic_simd<float,HostAbi>;  // float lanes of Vector's width: converts to Vector exactly
static_assert(NarrowFloat::size()==lanes);
#if defined(OPENMX_HOST_SIMD_AVX512)
constexpr int chunk=4;  // output vectors kept in registers per column chunk
#else
constexpr int chunk=2;
#endif
NarrowFloat loadf(const float* p) { return simd::simd_unchecked_load<NarrowFloat>(p); }

// rows[n][l] -> out[l*pad+n], padding columns zero.
void transpose(const float* const* rows,int count,int length,float* out,size_t pad) {
  for(int l=0;l<length;++l) {
    float* o=out+size_t(l)*pad;
    for(int n=0;n<count;++n)o[n]=rows[n][l];
    std::fill(o+count,o+pad,0.0f);
  }
}
void transpose(const double* base,size_t stride,int count,size_t length,double* out,size_t pad) {
  for(size_t l=0;l<length;++l) {
    double* o=out+l*pad;
    for(int n=0;n<count;++n)o[n]=base[size_t(n)*stride+l];
    std::fill(o+count,o+pad,0.0);
  }
}
// Store the first count lanes of v at p.
void store_partial(Vector v,double* p,int count) {
  if(count>=lanes){store(v,p);return;}
  double values[lanes];store(v,values);
  for(int k=0;k<count;++k)p[k]=values[k];
}

template<bool Fp32,int Terms>
void vna_rows(const OpenMXForceVnaBlock& b,const float* value_t,const float* const* deriv_t,size_t pad) {
  const int vectors=int(pad/lanes);
  for(int m=0;m<b.rows;++m) {
    const float* dx=b.derivative[0][m];const float* dy=b.derivative[1][m];const float* dz=b.derivative[2][m];
    const float* h0=Terms==2?b.row_value[m]:nullptr;
    for(int v0=0;v0<vectors;v0+=chunk) {
      const int nv=std::min(chunk,vectors-v0);
      Vector acc[3][chunk];
      for(auto& row:acc)for(auto& a:row)a=Vector(0.0);
      for(int l=0;l<b.projectors;++l) {
        const size_t at=size_t(l)*pad+size_t(v0)*lanes;
        if constexpr(Fp32) {
          const NarrowFloat ax(dx[l]),ay(dy[l]),az(dz[l]);
          if constexpr(Terms==1) {
            for(int v=0;v<nv;++v) {
              const NarrowFloat q=loadf(value_t+at+size_t(v)*lanes);
              acc[0][v]+=Vector(ax*q);acc[1][v]+=Vector(ay*q);acc[2][v]+=Vector(az*q);
            }
          }else {
            const NarrowFloat h(h0[l]);
            for(int v=0;v<nv;++v) {
              const size_t o=at+size_t(v)*lanes;
              const NarrowFloat q=loadf(value_t+o),qx=loadf(deriv_t[0]+o),qy=loadf(deriv_t[1]+o),qz=loadf(deriv_t[2]+o);
              acc[0][v]-=Vector(ax*q)+Vector(h*qx);
              acc[1][v]-=Vector(ay*q)+Vector(h*qy);
              acc[2][v]-=Vector(az*q)+Vector(h*qz);
            }
          }
        }else {
          const Vector ax(double(dx[l])),ay(double(dy[l])),az(double(dz[l]));
          if constexpr(Terms==1) {
            for(int v=0;v<nv;++v) {
              const Vector q(loadf(value_t+at+size_t(v)*lanes));
              acc[0][v]+=ax*q;acc[1][v]+=ay*q;acc[2][v]+=az*q;
            }
          }else {
            const Vector h(double(h0[l]));
            for(int v=0;v<nv;++v) {
              const size_t o=at+size_t(v)*lanes;
              const Vector q(loadf(value_t+o)),qx(loadf(deriv_t[0]+o)),qy(loadf(deriv_t[1]+o)),qz(loadf(deriv_t[2]+o));
              acc[0][v]-=ax*q+h*qx;
              acc[1][v]-=ay*q+h*qy;
              acc[2][v]-=az*q+h*qz;
            }
          }
        }
      }
      for(int c=0;c<3;++c)for(int v=0;v<nv;++v) {
        const int n=(v0+v)*lanes;
        store_partial(acc[c][v],b.output[c]+size_t(m)*b.output_stride+n,std::min(lanes,b.columns-n));
      }
    }
  }
}
// One-lane variant: the legacy loops themselves (no transposition).
template<bool Fp32,int Terms>
void vna_plain(const OpenMXForceVnaBlock& b) {
  for(int m=0;m<b.rows;++m)for(int n=0;n<b.columns;++n) {
    const float* dx=b.derivative[0][m];const float* dy=b.derivative[1][m];const float* dz=b.derivative[2][m];
    const float* q0=b.value[n];
    double sx=0.0,sy=0.0,sz=0.0;
    if constexpr(Terms==1) {
      for(int l=0;l<b.projectors;++l) {
        if constexpr(Fp32){sx+=double(dx[l]*q0[l]);sy+=double(dy[l]*q0[l]);sz+=double(dz[l]*q0[l]);}
        else{sx+=double(dx[l])*double(q0[l]);sy+=double(dy[l])*double(q0[l]);sz+=double(dz[l])*double(q0[l]);}
      }
    }else {
      const float* h0=b.row_value[m];const float* qx=b.column_derivative[0][n];const float* qy=b.column_derivative[1][n];const float* qz=b.column_derivative[2][n];
      for(int l=0;l<b.projectors;++l) {
        if constexpr(Fp32) {
          sx-=double(dx[l]*q0[l])+double(h0[l]*qx[l]);sy-=double(dy[l]*q0[l])+double(h0[l]*qy[l]);sz-=double(dz[l]*q0[l])+double(h0[l]*qz[l]);
        }else {
          sx-=double(dx[l])*double(q0[l])+double(h0[l])*double(qx[l]);
          sy-=double(dy[l])*double(q0[l])+double(h0[l])*double(qy[l]);
          sz-=double(dz[l])*double(q0[l])+double(h0[l])*double(qz[l]);
        }
      }
    }
    const size_t at=size_t(m)*b.output_stride+n;
    b.output[0][at]=sx;b.output[1][at]=sy;b.output[2][at]=sz;
  }
}
void vna_block(const OpenMXForceVnaBlock& b,void* scratch) {
  if(!b.rows || !b.columns)return;
  if constexpr(lanes==1) {
    if(b.fp32){if(b.terms==1)vna_plain<true,1>(b);else vna_plain<true,2>(b);}
    else{if(b.terms==1)vna_plain<false,1>(b);else vna_plain<false,2>(b);}
    (void)scratch;return;
  }
  const size_t pad=force_padded(b.columns),plane=size_t(b.projectors)*pad;
  float* value_t=static_cast<float*>(scratch);
  const float* deriv_t[3]={nullptr,nullptr,nullptr};
  transpose(b.value,b.columns,b.projectors,value_t,pad);
  if(b.terms==2)for(int c=0;c<3;++c) {
    float* t=value_t+size_t(c+1)*plane;
    transpose(b.column_derivative[c],b.columns,b.projectors,t,pad);deriv_t[c]=t;
  }
  if(b.fp32){if(b.terms==1)vna_rows<true,1>(b,value_t,deriv_t,pad);else vna_rows<true,2>(b,value_t,deriv_t,pad);}
  else{if(b.terms==1)vna_rows<false,1>(b,value_t,deriv_t,pad);else vna_rows<false,2>(b,value_t,deriv_t,pad);}
}

void grid_pair(const OpenMXForceGridPair& g,double* sums,void* scratch) {
  if(!g.points || !g.rows)return;
  const size_t pad=force_padded(g.rows);
  double* dm_t=static_cast<double*>(scratch);               // [spins][columns][pad]
  double* tmp0=dm_t+size_t(g.spins)*size_t(g.columns)*pad;  // [pad]
  for(int s=0;s<g.spins;++s)for(int j=0;j<g.columns;++j) {
    double* o=dm_t+(size_t(s)*g.columns+j)*pad;
    for(int i=0;i<g.rows;++i)o[i]=g.density[s][i][j];
    std::fill(o+g.rows,o+pad,0.0);
  }
  const int vectors=int(pad/lanes);
  for(int p=0;p<g.points;++p) {
    const int nc=g.center[p];
    const float* orbitals=g.neighbor_rows[g.neighbor?g.neighbor[p]:p];
    const double* const* derivative=g.derivative[nc];
    for(int s=0;s<g.spins;++s) {
      const double* dm=dm_t+size_t(s)*size_t(g.columns)*pad;
      for(int v0=0;v0<vectors;v0+=chunk) {
        const int nv=std::min(chunk,vectors-v0);
        Vector acc[chunk];
        for(auto& a:acc)a=Vector(0.0);
        for(int j=0;j<g.columns;++j) {
          const Vector w(double(orbitals[j]));
          const double* row=dm+size_t(j)*pad+size_t(v0)*lanes;
          for(int v=0;v<nv;++v)acc[v]+=w*load(row+size_t(v)*lanes);
        }
        for(int v=0;v<nv;++v)store(acc[v],tmp0+size_t(v0+v)*lanes);
      }
      double tx=0.0,ty=0.0,tz=0.0;
      for(int i=0;i<g.rows;++i) {
        const double* d=derivative[i];
        tx+=d[0]*tmp0[i];ty+=d[1]*tmp0[i];tz+=d[2]*tmp0[i];
      }
      const double potential=g.potential[s][nc];
      sums[0]+=tx*potential;sums[1]+=ty*potential;sums[2]+=tz*potential;
    }
  }
}

// One term of one row m into sum[axis][v]: the lane operand (value rows for
// a direct term, derivative rows for a transposed one) comes transposed as
// [component][n]; the other operand is a scalar of row m.
template<bool Transpose,bool ValueFirst,bool Subtract>
void nl_term(const OpenMXForceProjectorSpecies& sp,const OpenMXProjectorForceTerm& term,size_t m,
    const double* const lane[3],size_t pad,int v0,int nv,Vector (*sum)[chunk]) {
  for(size_t j=0;j<sp.count;++j) {
    const auto& channel=sp.channels[j];
    const double energy=term.reverse_energy?-channel.energy[0]:channel.energy[0];
    for(int magnetic=0;magnetic<=channel.twice_angular;++magnetic) {
      const size_t p=channel.offset+magnetic,at=p*pad+size_t(v0)*lanes;
      if constexpr(!Transpose) {
        const double a[3]={term.derivative[0][0][m*term.derivative_stride+p],
            term.derivative[0][1][m*term.derivative_stride+p],term.derivative[0][2][m*term.derivative_stride+p]};
        for(int v=0;v<nv;++v) {
          const Vector value=load(lane[0]+at+size_t(v)*lanes);
          for(int axis=0;axis<3;++axis) {
            Vector x;
            if constexpr(ValueFirst)x=(Vector(energy)*value)*Vector(a[axis]);
            else x=Vector(energy*a[axis])*value;
            if constexpr(Subtract)sum[axis][v]-=x;else sum[axis][v]+=x;
          }
        }
      }else {
        const double value=term.value[0][m*term.value_stride+p];
        for(int v=0;v<nv;++v)for(int axis=0;axis<3;++axis) {
          const Vector a=load(lane[axis]+at+size_t(v)*lanes);
          Vector x;
          if constexpr(ValueFirst)x=Vector(energy*value)*a;
          else x=(Vector(energy)*a)*Vector(value);
          if constexpr(Subtract)sum[axis][v]-=x;else sum[axis][v]+=x;
        }
      }
    }
  }
}
template<bool Transpose,bool ValueFirst>
void nl_term(const OpenMXForceProjectorSpecies& sp,const OpenMXProjectorForceTerm& term,size_t m,
    const double* const lane[3],size_t pad,int v0,int nv,Vector (*sum)[chunk]) {
  if(term.sign==1.0)nl_term<Transpose,ValueFirst,false>(sp,term,m,lane,pad,v0,nv,sum);
  else nl_term<Transpose,ValueFirst,true>(sp,term,m,lane,pad,v0,nv,sum);
}
// One-lane variant: the scalar Core block itself, specialized on the flags.
template<bool ValueFirst,bool Subtract>
void nl_plain_term(const OpenMXForceProjectorSpecies& sp,const OpenMXProjectorForceTerm& term,size_t ai,size_t bi,double sum[3]) {
  for(size_t j=0;j<sp.count;++j) {
    const auto& channel=sp.channels[j];
    const double energy=term.reverse_energy?-channel.energy[0]:channel.energy[0];
    for(int magnetic=0;magnetic<=channel.twice_angular;++magnetic) {
      const size_t p=channel.offset+magnetic;const double v=term.value[0][bi+p];
      for(int axis=0;axis<3;++axis) {
        const double a=term.derivative[0][axis][ai+p];
        double x;
        if constexpr(ValueFirst)x=energy*v*a;else x=energy*a*v;
        if constexpr(Subtract)sum[axis]-=x;else sum[axis]+=x;
      }
    }
  }
}
void nl_plain(const OpenMXForceProjectorSpecies& sp,const OpenMXProjectorForceBlock& b) {
  for(size_t m=0;m<b.rows;++m)for(size_t n=0;n<b.columns;++n) {
    double sum[3]={0.0,0.0,0.0};
    for(size_t k=0;k<b.term_count;++k) {
      const auto& term=b.terms[k];
      const size_t ai=(term.transpose?n:m)*term.derivative_stride,bi=(term.transpose?m:n)*term.value_stride;
      if(term.value_first){if(term.sign==1.0)nl_plain_term<true,false>(sp,term,ai,bi,sum);else nl_plain_term<true,true>(sp,term,ai,bi,sum);}
      else{if(term.sign==1.0)nl_plain_term<false,false>(sp,term,ai,bi,sum);else nl_plain_term<false,true>(sp,term,ai,bi,sum);}
    }
    for(int axis=0;axis<3;++axis) {
      dcomplex* out=b.output[axis];
      if(b.accumulate)for(size_t spin=0;spin<2;++spin)out[(spin*b.output_rows+m)*b.output_stride+n].r+=sum[axis];
      else {
        for(size_t spin=0;spin<3;++spin)out[(spin*b.output_rows+m)*b.output_stride+n]=dcomplex{0.0,0.0};
        out[(0*b.output_rows+m)*b.output_stride+n].r=sum[axis];out[(1*b.output_rows+m)*b.output_stride+n].r=sum[axis];
      }
    }
  }
}
void nl_block(const OpenMXForceProjectorSpecies& sp,const OpenMXProjectorForceBlock& b,void* scratch) {
  const int columns=int(b.columns);
  if(!b.rows || !columns)return;
  if constexpr(lanes==1){nl_plain(sp,b);(void)scratch;return;}
  const size_t pad=force_padded(columns),plane=sp.components*pad;
  double* t=static_cast<double*>(scratch);
  // The lane operand of each term, transposed to [component][n].
  const double* lane[2][3]={{nullptr,nullptr,nullptr},{nullptr,nullptr,nullptr}};
  size_t used=0;
  for(size_t k=0;k<b.term_count;++k) {
    const auto& term=b.terms[k];
    if(!term.transpose) {
      transpose(term.value[0],term.value_stride,columns,sp.components,t+used*plane,pad);lane[k][0]=t+used*plane;++used;
    }else for(int axis=0;axis<3;++axis) {
      transpose(term.derivative[0][axis],term.derivative_stride,columns,sp.components,t+used*plane,pad);lane[k][axis]=t+used*plane;++used;
    }
  }
  const int vectors=int(pad/lanes);
  for(size_t m=0;m<b.rows;++m) {
    for(int v0=0;v0<vectors;v0+=chunk) {
      const int nv=std::min(chunk,vectors-v0);
      Vector sum[3][chunk];
      for(auto& row:sum)for(auto& a:row)a=Vector(0.0);
      for(size_t k=0;k<b.term_count;++k) {
        const auto& term=b.terms[k];
        if(term.transpose){if(term.value_first)nl_term<true,true>(sp,term,m,lane[k],pad,v0,nv,sum);else nl_term<true,false>(sp,term,m,lane[k],pad,v0,nv,sum);}
        else{if(term.value_first)nl_term<false,true>(sp,term,m,lane[k],pad,v0,nv,sum);else nl_term<false,false>(sp,term,m,lane[k],pad,v0,nv,sum);}
      }
      for(int axis=0;axis<3;++axis)for(int v=0;v<nv;++v) {
        const int n0=(v0+v)*lanes,count=std::min(lanes,columns-n0);
        double values[lanes];store(sum[axis][v],values);
        dcomplex* out=b.output[axis];
        for(int c=0;c<count;++c) {
          const size_t n=size_t(n0+c);
          if(b.accumulate)for(size_t spin=0;spin<2;++spin)out[(spin*b.output_rows+m)*b.output_stride+n].r+=values[c];
          else {
            for(size_t spin=0;spin<3;++spin)out[(spin*b.output_rows+m)*b.output_stride+n]=dcomplex{0.0,0.0};
            out[(0*b.output_rows+m)*b.output_stride+n].r=values[c];
            out[(1*b.output_rows+m)*b.output_stride+n].r=values[c];
          }
        }
      }
    }
  }
}
constexpr ForceKernels kernels{lanes,vna_block,grid_pair,nl_block};
}
const ForceKernels& OPENMX_HOST_SIMD_FACTORY(force)() noexcept {return kernels;}
}

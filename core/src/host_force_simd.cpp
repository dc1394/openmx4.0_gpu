#include "host_force_kernels.hpp"
#include "host_simd_abi.hpp"
#include <algorithm>
#include <cstring>

// Force-stage host kernels. Lanes always span independent output elements
// (matrix columns n or rows m, density-matrix rows i); every reduction runs in
// the legacy order, and every product rounds where the legacy C expression
// rounds (float products before widening when fp32 is set).
namespace openmx::core::host {
namespace {
using NarrowFloat=simd::basic_simd<float,HostAbi>;  // float lanes of Vector's width: converts to Vector exactly
static_assert(NarrowFloat::size()==lanes);
#if defined(OPENMX_HOST_SIMD_AVX512)
constexpr int chunk=4;  // output vectors kept in registers per lane chunk
#else
constexpr int chunk=2;
#endif
NarrowFloat loadf(const float* p) { return simd::simd_unchecked_load<NarrowFloat>(p); }

// rows[n][l] -> out[l*pad+n], padding lanes zero.
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

void vna_lanes(const float* const* const* rows,int components,int count,int projectors,float* planes,size_t pad) {
  for(int c=0;c<components;++c)transpose(rows[c],count,projectors,planes+size_t(c)*size_t(projectors)*pad,pad);
}

// Lane planes: value, column derivatives x,y,z (lanes over n) or derivatives
// x,y,z, row value (lanes over m). The other operand streams row by row.
template<bool Fp32,int Terms,bool LanesRows>
void vna_rows(const OpenMXForceVnaBlock& b,const float* const planes[4],size_t pad) {
  const int outer=LanesRows?b.columns:b.rows,lane_count=LanesRows?b.rows:b.columns;
  const int vectors=int(pad/lanes);
  for(int o=0;o<outer;++o) {
    const float* r0;const float* r1=nullptr;const float* r2=nullptr;const float* r3=nullptr;
    if constexpr(!LanesRows) {
      r0=b.derivative[0][o];r1=b.derivative[1][o];r2=b.derivative[2][o];
      if constexpr(Terms==2)r3=b.row_value[o];
    }else {
      r0=b.value[o];
      if constexpr(Terms==2){r1=b.column_derivative[0][o];r2=b.column_derivative[1][o];r3=b.column_derivative[2][o];}
    }
    for(int v0=0;v0<vectors;v0+=chunk) {
      const int nv=std::min(chunk,vectors-v0);
      Vector acc[3][chunk];
      for(auto& row:acc)for(auto& a:row)a=Vector(0.0);
      for(int l=0;l<b.projectors;++l) {
        const size_t at=size_t(l)*pad+size_t(v0)*lanes;
        if constexpr(!LanesRows) {
          if constexpr(Fp32) {
            const NarrowFloat ax(r0[l]),ay(r1[l]),az(r2[l]);
            if constexpr(Terms==1) {
              for(int v=0;v<nv;++v) {
                const NarrowFloat q=loadf(planes[0]+at+size_t(v)*lanes);
                acc[0][v]+=Vector(ax*q);acc[1][v]+=Vector(ay*q);acc[2][v]+=Vector(az*q);
              }
            }else {
              const NarrowFloat h(r3[l]);
              for(int v=0;v<nv;++v) {
                const size_t o2=at+size_t(v)*lanes;
                const NarrowFloat q=loadf(planes[0]+o2),qx=loadf(planes[1]+o2),qy=loadf(planes[2]+o2),qz=loadf(planes[3]+o2);
                acc[0][v]-=Vector(ax*q)+Vector(h*qx);
                acc[1][v]-=Vector(ay*q)+Vector(h*qy);
                acc[2][v]-=Vector(az*q)+Vector(h*qz);
              }
            }
          }else {
            const Vector ax(double(r0[l])),ay(double(r1[l])),az(double(r2[l]));
            if constexpr(Terms==1) {
              for(int v=0;v<nv;++v) {
                const Vector q(loadf(planes[0]+at+size_t(v)*lanes));
                acc[0][v]+=ax*q;acc[1][v]+=ay*q;acc[2][v]+=az*q;
              }
            }else {
              const Vector h(double(r3[l]));
              for(int v=0;v<nv;++v) {
                const size_t o2=at+size_t(v)*lanes;
                const Vector q(loadf(planes[0]+o2)),qx(loadf(planes[1]+o2)),qy(loadf(planes[2]+o2)),qz(loadf(planes[3]+o2));
                acc[0][v]-=ax*q+h*qx;acc[1][v]-=ay*q+h*qy;acc[2][v]-=az*q+h*qz;
              }
            }
          }
        }else {
          if constexpr(Fp32) {
            const NarrowFloat q(r0[l]);
            if constexpr(Terms==1) {
              for(int v=0;v<nv;++v) {
                const size_t o2=at+size_t(v)*lanes;
                const NarrowFloat hx=loadf(planes[0]+o2),hy=loadf(planes[1]+o2),hz=loadf(planes[2]+o2);
                acc[0][v]+=Vector(hx*q);acc[1][v]+=Vector(hy*q);acc[2][v]+=Vector(hz*q);
              }
            }else {
              const NarrowFloat qx(r1[l]),qy(r2[l]),qz(r3[l]);
              for(int v=0;v<nv;++v) {
                const size_t o2=at+size_t(v)*lanes;
                const NarrowFloat hx=loadf(planes[0]+o2),hy=loadf(planes[1]+o2),hz=loadf(planes[2]+o2),h0=loadf(planes[3]+o2);
                acc[0][v]-=Vector(hx*q)+Vector(h0*qx);
                acc[1][v]-=Vector(hy*q)+Vector(h0*qy);
                acc[2][v]-=Vector(hz*q)+Vector(h0*qz);
              }
            }
          }else {
            const Vector q(double(r0[l]));
            if constexpr(Terms==1) {
              for(int v=0;v<nv;++v) {
                const size_t o2=at+size_t(v)*lanes;
                const Vector hx(loadf(planes[0]+o2)),hy(loadf(planes[1]+o2)),hz(loadf(planes[2]+o2));
                acc[0][v]+=hx*q;acc[1][v]+=hy*q;acc[2][v]+=hz*q;
              }
            }else {
              const Vector qx(double(r1[l])),qy(double(r2[l])),qz(double(r3[l]));
              for(int v=0;v<nv;++v) {
                const size_t o2=at+size_t(v)*lanes;
                const Vector hx(loadf(planes[0]+o2)),hy(loadf(planes[1]+o2)),hz(loadf(planes[2]+o2)),h0(loadf(planes[3]+o2));
                acc[0][v]-=hx*q+h0*qx;acc[1][v]-=hy*q+h0*qy;acc[2][v]-=hz*q+h0*qz;
              }
            }
          }
        }
      }
      for(int c=0;c<3;++c)for(int v=0;v<nv;++v) {
        const int n0=(v0+v)*lanes;
        store_partial(acc[c][v],b.output[c]+size_t(o)*b.output_stride+n0,std::min(lanes,lane_count-n0));
      }
    }
  }
}
// One-lane variant: the legacy loops themselves over the raw rows when the
// caller passed both sides (the glue always does), else the lane operand is
// read from its prepared planes.
template<bool Fp32,int Terms>
void vna_plain_rows(const OpenMXForceVnaBlock& b) {
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
    const size_t at=b.lanes_rows?size_t(n)*b.output_stride+m:size_t(m)*b.output_stride+n;
    b.output[0][at]=sx;b.output[1][at]=sy;b.output[2][at]=sz;
  }
}
template<bool Fp32,int Terms>
void vna_plain(const OpenMXForceVnaBlock& b) {
  const bool raw=b.derivative[0] && b.derivative[1] && b.derivative[2] && b.value &&
      (Terms==1 || (b.row_value && b.column_derivative[0] && b.column_derivative[1] && b.column_derivative[2]));
  if(raw || !b.lanes){vna_plain_rows<Fp32,Terms>(b);return;}
  const float* planes=b.lanes->planes;
  const size_t pad=b.lanes->pad,plane=size_t(b.projectors)*pad;
  // lane operand element: component c of lane index idx at projector l
  const auto lane=[&](int c,int idx,int l)->float {return planes[size_t(c)*plane+size_t(l)*pad+size_t(idx)];};
  for(int m=0;m<b.rows;++m)for(int n=0;n<b.columns;++n) {
    double sx=0.0,sy=0.0,sz=0.0;
    for(int l=0;l<b.projectors;++l) {
      float hx,hy,hz,q0,h0=0.0f,qx=0.0f,qy=0.0f,qz=0.0f;
      if(!b.lanes_rows) {
        hx=b.derivative[0][m][l];hy=b.derivative[1][m][l];hz=b.derivative[2][m][l];q0=lane(0,n,l);
        if constexpr(Terms==2){h0=b.row_value[m][l];qx=lane(1,n,l);qy=lane(2,n,l);qz=lane(3,n,l);}
      }else {
        hx=lane(0,m,l);hy=lane(1,m,l);hz=lane(2,m,l);q0=b.value[n][l];
        if constexpr(Terms==2){h0=lane(3,m,l);qx=b.column_derivative[0][n][l];qy=b.column_derivative[1][n][l];qz=b.column_derivative[2][n][l];}
      }
      if constexpr(Terms==1) {
        if constexpr(Fp32){sx+=double(hx*q0);sy+=double(hy*q0);sz+=double(hz*q0);}
        else{sx+=double(hx)*double(q0);sy+=double(hy)*double(q0);sz+=double(hz)*double(q0);}
      }else {
        if constexpr(Fp32){sx-=double(hx*q0)+double(h0*qx);sy-=double(hy*q0)+double(h0*qy);sz-=double(hz*q0)+double(h0*qz);}
        else{sx-=double(hx)*double(q0)+double(h0)*double(qx);sy-=double(hy)*double(q0)+double(h0)*double(qy);sz-=double(hz)*double(q0)+double(h0)*double(qz);}
      }
    }
    const size_t at=b.lanes_rows?size_t(n)*b.output_stride+m:size_t(m)*b.output_stride+n;
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
  const int lane_count=b.lanes_rows?b.rows:b.columns,components=force_vna_planes(b.terms,b.lanes_rows);
  const size_t pad=force_padded(lane_count),plane=size_t(b.projectors)*pad;
  const float* planes[4]={nullptr,nullptr,nullptr,nullptr};
  if(b.lanes) {
    for(int c=0;c<components;++c)planes[c]=b.lanes->planes+size_t(c)*plane;
  }else {
    float* t=static_cast<float*>(scratch);
    const float* const* source[4];
    if(!b.lanes_rows){source[0]=b.value;source[1]=b.column_derivative[0];source[2]=b.column_derivative[1];source[3]=b.column_derivative[2];}
    else{source[0]=b.derivative[0];source[1]=b.derivative[1];source[2]=b.derivative[2];source[3]=b.row_value;}
    for(int c=0;c<components;++c){transpose(source[c],lane_count,b.projectors,t+size_t(c)*plane,pad);planes[c]=t+size_t(c)*plane;}
  }
  if(b.lanes_rows) {
    if(b.fp32){if(b.terms==1)vna_rows<true,1,true>(b,planes,pad);else vna_rows<true,2,true>(b,planes,pad);}
    else{if(b.terms==1)vna_rows<false,1,true>(b,planes,pad);else vna_rows<false,2,true>(b,planes,pad);}
  }else {
    if(b.fp32){if(b.terms==1)vna_rows<true,1,false>(b,planes,pad);else vna_rows<true,2,false>(b,planes,pad);}
    else{if(b.terms==1)vna_rows<false,1,false>(b,planes,pad);else vna_rows<false,2,false>(b,planes,pad);}
  }
}

// tmp0[i] = sum_j orbitals[j] * dm[j][i] over the density planes of one pair
// and spin (lanes over i, j ascending), stored for the scalar i-reduction.
void density_rows(const double* dm,int columns,size_t pad,const float* orbitals,double* tmp0) {
  const int vectors=int(pad/lanes);
  for(int v0=0;v0<vectors;v0+=chunk) {
    const int nv=std::min(chunk,vectors-v0);
    Vector acc[chunk];
    for(auto& a:acc)a=Vector(0.0);
    for(int j=0;j<columns;++j) {
      const Vector w(double(orbitals[j]));
      const double* row=dm+size_t(j)*pad+size_t(v0)*lanes;
      for(int v=0;v<nv;++v)acc[v]+=w*load(row+size_t(v)*lanes);
    }
    for(int v=0;v<nv;++v)store(acc[v],tmp0+size_t(v0+v)*lanes);
  }
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
  for(int p=0;p<g.points;++p) {
    const int nc=g.center[p];
    const float* orbitals=g.neighbor_rows[g.neighbor?g.neighbor[p]:p];
    const double* const* derivative=g.derivative[nc];
    for(int s=0;s<g.spins;++s) {
      density_rows(dm_t+size_t(s)*size_t(g.columns)*pad,g.columns,pad,orbitals,tmp0);
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
void grid_density(const double* const* density[4],int rows,int columns,int columns_max,int spins,double* planes,size_t pad) {
  for(int s=0;s<spins;++s)for(int j=0;j<columns;++j) {
    double* o=planes+(size_t(s)*columns_max+j)*pad;
    for(int i=0;i<rows;++i)o[i]=density[s][i][j];
    std::fill(o+rows,o+pad,0.0);
  }
}
// Point-major: one derivative row per point, every visit of the point in
// turn; the terms land in the slots the caller sums in the legacy order.
void grid_points(const OpenMXForceGridPoints& g,int first,int count,void* scratch) {
  double* tmp0=static_cast<double*>(scratch);
  const size_t pad=g.pad_rows,plane=size_t(g.columns_max)*pad;
  for(int p=first;p<first+count;++p) {
    const double* d=g.derivative+size_t(p)*g.point_stride;
    for(size_t v=g.point_offsets[p];v<g.point_offsets[p+1];++v) {
      const int pair=g.visit_pair[v],columns=g.pair_columns[pair];
      const float* orbitals=g.pair_rows[pair][g.visit_row[v]];
      double* term=g.terms+size_t(g.visit_term[v])*size_t(g.spins)*3;
      for(int s=0;s<g.spins;++s) {
        density_rows(g.dm+(size_t(pair)*g.spins+s)*plane,columns,pad,orbitals,tmp0);
        double tx=0.0,ty=0.0,tz=0.0;
        for(int i=0;i<g.rows;++i) {
          const double* di=d+size_t(i)*g.row_stride;
          tx+=di[0]*tmp0[i];ty+=di[1]*tmp0[i];tz+=di[2]*tmp0[i];
        }
        const double potential=g.potential[s][p];
        term[s*3+0]=tx*potential;term[s*3+1]=ty*potential;term[s*3+2]=tz*potential;
      }
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
constexpr ForceKernels kernels{lanes,vna_lanes,vna_block,grid_pair,grid_density,grid_points,nl_block};
}
const ForceKernels& OPENMX_HOST_SIMD_FACTORY(force)() noexcept {return kernels;}
}

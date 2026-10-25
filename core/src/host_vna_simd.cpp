#include "host_vna_kernels.hpp"
#include "host_simd_abi.hpp"
#include <algorithm>
#include <cmath>

// Host kernels of Set_ProExpn_VNA. Lanes span independent outputs (projector
// rows, Bessel arguments, Bessel orders, orbital rows or columns); every
// reduction keeps the legacy ascending order and association, so each
// element is bitwise the scalar loop's result without contraction.
namespace openmx::core::host {
namespace {
using Mask=typename Vector::mask_type;
using FloatLanes=simd::basic_simd<float,HostAbi>;  // the float lanes that widen to one Vector
using FloatMask=typename FloatLanes::mask_type;
constexpr int V=lanes;

Mask lane_mask(int count) { return Mask([=](auto lane){return int(lane)<count;}); }
FloatMask float_mask(int count) { return FloatMask([=](auto lane){return int(lane)<count;}); }
Vector select(Mask m,Vector a,Vector b) { return simd::condition(m,a,b); }
Vector load_partial(const double* p,int count) {
  return count>=V?load(p):simd::simd_partial_load<Vector>(p,lane_mask(count));
}
FloatLanes load_float(const float* p,int count) {
  return count>=V?simd::simd_unchecked_load<FloatLanes>(p):simd::simd_partial_load<FloatLanes>(p,float_mask(count));
}
void store_partial(Vector v,double* p,int count) {
  if(count>=V)store(v,p);else simd::simd_partial_store(v,p,lane_mask(count));
}
#if defined(OPENMX_HOST_SIMD_AVX512)
bool any(Mask m) { return static_cast<__mmask8>(m)!=0; }
#else
bool any(Mask m) { return simd::any_of(m); }
#endif
Vector truncated(Vector v) { return Vector([&](auto i){return std::trunc(v[i]);}); }
// Per-lane libm, as the scalar routine calls it; no ISA header dependency.
Vector sine(Vector v) { return Vector([&](auto i){return std::sin(v[i]);}); }
Vector cosine(Vector v) { return Vector([&](auto i){return std::cos(v[i]);}); }
Vector magnitude(Vector v) { return select(v<Vector(0.0),Vector(0.0)-v,v); }
double lane_max(Vector v) { double r=v[0];for(int i=1;i<V;++i)r=std::max(r,v[i]);return r; }

// ---- Gauss-Legendre contraction: lanes over projector rows ---------------
// MG radial functions share each table load; C row vectors per pass keep
// enough independent accumulators to cover the add latency.
constexpr int contract_vectors=V>=8?2:1;
template<int MG,int C,bool FULL> void contract_tile(const OpenMXVnaContract& p,int m0,int r0,int tail) {
  Vector acc0[MG][C],acc1[MG][C];
  for(int m=0;m<MG;++m)for(int c=0;c<C;++c){acc0[m][c]=Vector(0.0);acc1[m][c]=Vector(0.0);}
  const double* table=p.table+r0;
  const size_t ts=p.table_stride;
  const double* w0[MG];const double* w1[MG];
  for(int m=0;m<MG;++m){w0[m]=p.w0[m0+m];w1[m]=p.w1[m0+m];}
  const Mask mask=lane_mask(tail);
  for(int i=0;i<p.points;++i) {
    Vector t[C];
    for(int c=0;c<C-1;++c)t[c]=load(table+size_t(i)*ts+c*V);
    t[C-1]=FULL?load(table+size_t(i)*ts+(C-1)*V):simd::simd_partial_load<Vector>(table+size_t(i)*ts+(C-1)*V,mask);
    for(int m=0;m<MG;++m) {
      const Vector a(w0[m][i]),b(w1[m][i]);
      for(int c=0;c<C;++c){acc0[m][c]+=a*t[c];acc1[m][c]+=b*t[c];}
    }
  }
  for(int m=0;m<MG;++m)for(int c=0;c<C;++c) {
    const int count=c<C-1?V:tail;
    store_partial(acc0[m][c],p.out0+size_t(m0+m)*p.out_stride+r0+c*V,count);
    store_partial(acc1[m][c],p.out1+size_t(m0+m)*p.out_stride+r0+c*V,count);
  }
}
template<int C> void contract_pass(const OpenMXVnaContract& p,int r0,int tail) {
  int m0=0;
  if(tail>=V) {
    for(;m0+4<=p.count;m0+=4)contract_tile<4,C,true>(p,m0,r0,tail);
    for(;m0+2<=p.count;m0+=2)contract_tile<2,C,true>(p,m0,r0,tail);
    for(;m0<p.count;++m0)contract_tile<1,C,true>(p,m0,r0,tail);
  } else {
    for(;m0+4<=p.count;m0+=4)contract_tile<4,C,false>(p,m0,r0,tail);
    for(;m0+2<=p.count;m0+=2)contract_tile<2,C,false>(p,m0,r0,tail);
    for(;m0<p.count;++m0)contract_tile<1,C,false>(p,m0,r0,tail);
  }
}
void contract(const OpenMXVnaContract& p) {
  constexpr int C=contract_vectors;
  for(int r0=0;r0<p.rows;r0+=C*V) {
    const int remaining=p.rows-r0;
    if(C==2 && remaining>V)contract_pass<C>(p,r0,std::min(V,remaining-V));
    else contract_pass<1>(p,r0,std::min(V,remaining));
  }
}

// ---- HVNA23 quadrature: lanes over Bessel orders --------------------------
void quadrature(const OpenMXVnaQuadrature& p) {
  for(int l0=0;l0<p.orders;l0+=V) {
    const int count=std::min(V,p.orders-l0);
    Vector sum(0.0),sumr(0.0);
    for(int i=0;i<p.points;++i) {
      const size_t at=size_t(i)*p.table_stride+l0;
      const Vector k(p.wavevectors[i]);
      Vector t=Vector(p.scale*p.crude[i])*load_partial(p.product+at,count);
      t=t*Vector(p.weights[i]);t=t*k;t=t*k;
      sum+=t*load_partial(p.bessel+at,count);
      sumr+=(t*load_partial(p.bessel_derivative+at,count))*k;
    }
    store_partial(sum,p.sum+l0,count);
    store_partial(sumr,p.sum_derivative+l0,count);
  }
}

// ---- Spherical_Bessel2: lanes over arguments ------------------------------
// NB vectors advance together so the latency-bound recurrences overlap.
template<int NB> void bessel_blocks(const OpenMXVnaBessel& p,int first,int count) {
  const int lmax=p.lmax;
  Vector x[NB],invx[NB],limit[NB],vsb0[NB],vsb1[NB],vsbi[NB];
  Mask positive[NB];
  int top=0;
  for(int b=0;b<NB;++b) {
    const int c=std::clamp(count-b*V,0,V);
    Vector xv=c>0?load_partial(p.x+first+b*V,c):Vector(1.0);
    // Arguments at or below xmin (0) take the small-x branch; such lanes and
    // tail lanes recur with x = 1 and are overwritten before the stores.
    positive[b]=(xv>Vector(0.0))&&lane_mask(c);
    xv=select(positive[b],xv,Vector(1.0));
    x[b]=xv;invx[b]=Vector(1.0)/xv;
    Vector nmax=truncated(Vector(double(lmax))+Vector(3.0)*xv+Vector(20.0));
    nmax=select(nmax<Vector(100.0),Vector(100.0),nmax);
    limit[b]=nmax;
    vsb0[b]=Vector(0.0);vsb1[b]=Vector(1.0e-14);vsbi[b]=Vector(0.0);
    if(c>0)top=std::max(top,int(lane_max(select(positive[b],nmax,Vector(0.0)))));
  }
  // downward recurrence from nmax-1 to lmax+3 (a lane joins at its own nmax)
  for(int n=top-1;(lmax+2)<n;--n) {
    const Vector coef{2.0*n+1.0},nd{double(n)};
    for(int b=0;b<NB;++b) {
      const Mask active=nd<limit[b];
      Vector vsb2=coef*invx[b]*vsb1[b]-vsb0[b];
      const Mask big=active&&(Vector(1.0e+250)<vsb2);
      if(any(big)) {
        const Vector tmp=Vector(1.0)/vsb2;
        vsb2=select(big,vsb2*tmp,vsb2);vsb1[b]=select(big,vsb1[b]*tmp,vsb1[b]);
      }
      vsbi[b]=select(active,vsb0[b],vsbi[b]);
      vsb0[b]=select(active,vsb1[b],vsb0[b]);
      vsb1[b]=select(active,vsb2,vsb1[b]);
    }
  }
  for(int b=0;b<NB;++b) {
    const int c=std::clamp(count-b*V,0,V);
    if(c<=0)continue;
    Vector tsb[36],sb[33];
    int n=lmax+3;
    tsb[n-1]=vsb1[b];tsb[n]=vsb0[b];tsb[n+1]=vsbi[b];
    {const Vector tmp=tsb[n-1];tsb[n-1]=tsb[n-1]/tmp;tsb[n]=tsb[n]/tmp;}
    for(n=lmax+2;0<n;--n) {
      tsb[n-1]=Vector(2.0*n+1.0)*invx[b]*tsb[n]-tsb[n+1];
      const Mask big=Vector(1.0e+250)<tsb[n-1];
      if(any(big)) {
        const Vector tmp=tsb[n-1];
        for(int m=n-1;m<=lmax+1;++m)tsb[m]=select(big,tsb[m]/tmp,tsb[m]);
      }
    }
    const Vector si=sine(x[b]),co=cosine(x[b]),ix=Vector(1.0)/x[b];
    const Vector j0=si*ix,j1=si*ix*ix-co*ix;
    const Vector sf=select(magnitude(tsb[1])<magnitude(tsb[0]),j0/tsb[0],j1/tsb[1]);
    for(n=0;n<=lmax+1;++n)sb[n]=select(positive[b],tsb[n]*sf,Vector(n==0?1.0:0.0));
    const int point=first+b*V;
    double* dsb0=p.dsb+point;
    store_partial(select(positive[b],co*ix-si*ix*ix,Vector(0.0)),dsb0,c);
    store_partial(sb[0],p.sb+point,c);
    Vector dsb[31];
    dsb[0]=select(positive[b],co*ix-si*ix*ix,Vector(0.0));
    for(n=1;n<=lmax;++n) {
      dsb[n]=(Vector(double(n))*sb[n-1]-Vector(double(n)+1.0)*sb[n+1])/Vector(2.0*double(n)+1.0);
      store_partial(sb[n],p.sb+size_t(n)*p.stride+point,c);
      store_partial(dsb[n],p.dsb+size_t(n)*p.stride+point,c);
    }
    if(p.transposed)
      for(int lane=0;lane<c;++lane) {
        double* t=p.transposed+size_t(point+lane)*p.transposed_stride;
        double* td=p.transposed_derivative+size_t(point+lane)*p.transposed_stride;
        for(n=0;n<=lmax;++n){t[n]=sb[n][lane];td[n]=dsb[n][lane];}
      }
  }
}
void bessel(const OpenMXVnaBessel& p) {
  int first=0;
  for(;first+2*V<=p.points;first+=2*V)bessel_blocks<2>(p,first,2*V);
  if(first<p.points)bessel_blocks<2>(p,first,p.points-first);
}

// ---- HVNA projector contraction -------------------------------------------
// Left block transposed: lanes over rows m, NB columns at a time.
template<bool FP32,int NB> void hvna_rows(const OpenMXVnaHvnaBlock& p,int m0,int mcount,int n0) {
  Vector sum[NB];
  for(auto& s:sum)s=Vector(0.0);
  const float* left=p.left+m0;
  const float* column[NB];
  for(int c=0;c<NB;++c)column[c]=p.right+size_t(n0+c)*p.right_stride;
  const size_t ls=p.left_stride;
  const int* lengths=p.lengths;const double* energies=p.energies;const int groups=p.groups;
  int L=0;
  for(int g=0;g<groups;++g) {
    Vector tmp[NB];
    for(auto& t:tmp)t=Vector(0.0);
    const int members=lengths[g];
    for(int member=0;member<members;++member,++L) {
      const FloatLanes lf=load_float(left+size_t(L)*ls,mcount);
      if constexpr(FP32) {
        for(int c=0;c<NB;++c)tmp[c]+=Vector(lf*FloatLanes(column[c][L]));
      } else {
        const Vector ld(lf);
        for(int c=0;c<NB;++c)tmp[c]+=ld*Vector(double(column[c][L]));
      }
    }
    const Vector ene(energies[g]);
    for(int c=0;c<NB;++c)sum[c]+=ene*tmp[c];
  }
  for(int c=0;c<NB;++c)for(int lane=0;lane<mcount;++lane) {
    double& d=p.matrix[size_t(m0+lane)*p.stride+size_t(n0+c)];
    d=p.first?sum[c][lane]:d+sum[c][lane];
  }
}
// Right block transposed: lanes over columns n, NB rows at a time.
template<bool FP32,int NB> void hvna_columns(const OpenMXVnaHvnaBlock& p,int m0,int n0,int ncount) {
  Vector sum[NB];
  for(auto& s:sum)s=Vector(0.0);
  const float* right=p.right+n0;
  const float* row[NB];
  for(int r=0;r<NB;++r)row[r]=p.left+size_t(m0+r)*p.left_stride;
  const size_t rs=p.right_stride;
  const int* lengths=p.lengths;const double* energies=p.energies;const int groups=p.groups;
  int L=0;
  for(int g=0;g<groups;++g) {
    Vector tmp[NB];
    for(auto& t:tmp)t=Vector(0.0);
    const int members=lengths[g];
    for(int member=0;member<members;++member,++L) {
      const FloatLanes rf=load_float(right+size_t(L)*rs,ncount);
      if constexpr(FP32) {
        for(int r=0;r<NB;++r)tmp[r]+=Vector(FloatLanes(row[r][L])*rf);
      } else {
        const Vector rd(rf);
        for(int r=0;r<NB;++r)tmp[r]+=Vector(double(row[r][L]))*rd;
      }
    }
    const Vector ene(energies[g]);
    for(int r=0;r<NB;++r)sum[r]+=ene*tmp[r];
  }
  for(int r=0;r<NB;++r) {
    double* d=p.matrix+size_t(m0+r)*p.stride+n0;
    store_partial(p.first?sum[r]:load_partial(d,ncount)+sum[r],d,ncount);
  }
}
template<bool FP32> void hvna_block_typed(const OpenMXVnaHvnaBlock& p) {
  if(p.right_transposed) {
    for(int n0=0;n0<p.columns;n0+=V) {
      const int ncount=std::min(V,p.columns-n0);
      int m0=0;
      for(;m0+4<=p.rows;m0+=4)hvna_columns<FP32,4>(p,m0,n0,ncount);
      for(;m0+2<=p.rows;m0+=2)hvna_columns<FP32,2>(p,m0,n0,ncount);
      for(;m0<p.rows;++m0)hvna_columns<FP32,1>(p,m0,n0,ncount);
    }
  } else {
    for(int m0=0;m0<p.rows;m0+=V) {
      const int mcount=std::min(V,p.rows-m0);
      int n0=0;
      for(;n0+4<=p.columns;n0+=4)hvna_rows<FP32,4>(p,m0,mcount,n0);
      for(;n0+2<=p.columns;n0+=2)hvna_rows<FP32,2>(p,m0,mcount,n0);
      for(;n0<p.columns;++n0)hvna_rows<FP32,1>(p,m0,mcount,n0);
    }
  }
}
void hvna_block(const OpenMXVnaHvnaBlock& p) {
  if(p.fp32)hvna_block_typed<true>(p);else hvna_block_typed<false>(p);
}
constexpr VnaKernels kernels{V,contract,bessel,quadrature,hvna_block};
}
const VnaKernels& OPENMX_HOST_SIMD_FACTORY(vna)() noexcept {return kernels;}
}

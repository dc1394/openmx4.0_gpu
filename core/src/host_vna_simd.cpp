#include "host_vna_kernels.hpp"
#include "host_simd_abi.hpp"
#include <algorithm>
#include <cmath>
#include <type_traits>

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
// Every output (m, n) is sum_g ene_g * tmp_g with tmp_g the ascending member
// sum of product(left[m][L], right[n][L]). One operand block is transposed to
// [L][lanes] (lane operand); the other keeps its legacy rows (broadcast
// operand). A tile holds BC broadcast rows x LV lane vectors of accumulators:
// every loaded lane vector serves BC outputs, every broadcast value LV
// vectors. Products are commutative and exact to reorder (a*b == b*a), lanes
// span independent outputs, and each output keeps the legacy member/group
// order, so the result is bitwise the legacy rows'. Tail lanes use masks
// computed once per tile, outside the component loop. Under fp32 the float
// lanes are multiplied in float and the product widened (the legacy rounding
// point); under fp64 with double operands (DOUBLES) both factors are already
// widened, so the inner loop is one multiply and one add per lane vector.
constexpr int hvna_lane_vectors=V>=8?2:V>=4?2:4;   // lane vectors per tile
constexpr int hvna_broadcasts=V>=8?4:2;             // broadcast rows per tile

template<bool FP32,bool DOUBLES,bool FULL,int LV,int BC,class LaneT,class BroadcastT>
void hvna_tile(const OpenMXVnaHvnaBlock& p,const LaneT* lanes,size_t lane_stride,
    const BroadcastT* const* broadcast,FloatMask ftail,Mask tail,Vector (&sum)[BC][LV]) {
  static_assert(!(FP32 && DOUBLES),"double operands serve the fp64 products only");
  for(auto& row:sum)for(auto& v:row)v=Vector(0.0);
  const int* lengths=p.lengths;const double* energies=p.energies;const int groups=p.groups;
  int L=0;
  for(int g=0;g<groups;++g) {
    Vector tmp[BC][LV];
    for(auto& row:tmp)for(auto& v:row)v=Vector(0.0);
    const int members=lengths[g];
    for(int member=0;member<members;++member,++L) {
      const LaneT* at=lanes+size_t(L)*lane_stride;
      if constexpr(DOUBLES) {
        Vector ld[LV];
        for(int c=0;c<LV-1;++c)ld[c]=load(at+c*V);
        if constexpr(FULL)ld[LV-1]=load(at+(LV-1)*V);
        else ld[LV-1]=simd::simd_partial_load<Vector>(at+(LV-1)*V,tail);
        for(int b=0;b<BC;++b) {
          const Vector a(broadcast[b][L]);
          for(int c=0;c<LV;++c)tmp[b][c]+=a*ld[c];
        }
      } else {
        FloatLanes lf[LV];
        for(int c=0;c<LV-1;++c)lf[c]=simd::simd_unchecked_load<FloatLanes>(at+c*V);
        if constexpr(FULL)lf[LV-1]=simd::simd_unchecked_load<FloatLanes>(at+(LV-1)*V);
        else lf[LV-1]=simd::simd_partial_load<FloatLanes>(at+(LV-1)*V,ftail);
        if constexpr(FP32) {
          // (double)(left*right): the float product, then the exact widening
          for(int b=0;b<BC;++b) {
            const FloatLanes a(broadcast[b][L]);
            for(int c=0;c<LV;++c)tmp[b][c]+=Vector(a*lf[c]);
          }
        } else {
          // (double)left*(double)right: lanes widened once, shared by BC rows
          Vector ld[LV];
          for(int c=0;c<LV;++c)ld[c]=Vector(lf[c]);
          for(int b=0;b<BC;++b) {
            const Vector a(double(broadcast[b][L]));
            for(int c=0;c<LV;++c)tmp[b][c]+=a*ld[c];
          }
        }
      }
    }
    const Vector ene(energies[g]);
    for(int b=0;b<BC;++b)for(int c=0;c<LV;++c)sum[b][c]+=ene*tmp[b][c];
  }
}
template<bool DOUBLES> struct HvnaOperands {
  using T=std::conditional_t<DOUBLES,double,float>;
  static const T* left(const OpenMXVnaHvnaBlock& p) {
    if constexpr(DOUBLES)return p.left_double;else return p.left;
  }
  static const T* right(const OpenMXVnaHvnaBlock& p) {
    if constexpr(DOUBLES)return p.right_double;else return p.right;
  }
};
// Right block transposed: lanes over columns n, broadcast rows m.
template<bool FP32,bool DOUBLES,bool FULL,int LV,int BC>
void hvna_columns(const OpenMXVnaHvnaBlock& p,int m0,int n0,FloatMask ftail,Mask tail) {
  using O=HvnaOperands<DOUBLES>;
  const typename O::T* rows[BC];
  for(int b=0;b<BC;++b)rows[b]=O::left(p)+size_t(m0+b)*p.left_stride;
  Vector sum[BC][LV];
  hvna_tile<FP32,DOUBLES,FULL,LV,BC>(p,O::right(p)+n0,p.right_stride,rows,ftail,tail,sum);
  for(int b=0;b<BC;++b)for(int c=0;c<LV;++c) {
    double* d=p.matrix+size_t(m0+b)*p.stride+size_t(n0+c*V);
    if(FULL || c<LV-1)store(p.first?sum[b][c]:load(d)+sum[b][c],d);
    else simd::simd_partial_store(p.first?sum[b][c]:simd::simd_partial_load<Vector>(d,tail)+sum[b][c],d,tail);
  }
}
// Left block transposed: lanes over rows m, broadcast columns n.
template<bool FP32,bool DOUBLES,bool FULL,int LV,int BC>
void hvna_rows(const OpenMXVnaHvnaBlock& p,int m0,int n0,int mcount,FloatMask ftail,Mask tail) {
  using O=HvnaOperands<DOUBLES>;
  const typename O::T* columns[BC];
  for(int b=0;b<BC;++b)columns[b]=O::right(p)+size_t(n0+b)*p.right_stride;
  Vector sum[BC][LV];
  hvna_tile<FP32,DOUBLES,FULL,LV,BC>(p,O::left(p)+m0,p.left_stride,columns,ftail,tail,sum);
  for(int b=0;b<BC;++b)for(int c=0;c<LV;++c) {
    const int count=std::min(V,mcount-c*V);
    for(int lane=0;lane<count;++lane) {
      double& d=p.matrix[size_t(m0+c*V+lane)*p.stride+size_t(n0+b)];
      d=p.first?sum[b][c][lane]:d+sum[b][c][lane];
    }
  }
}
// One lane chunk (LV vectors, the last FULL or masked) over every broadcast index.
template<bool FP32,bool DOUBLES,bool FULL,int LV>
void hvna_chunk(const OpenMXVnaHvnaBlock& p,int lane0,int lanes_left,int broadcasts) {
  const int tail_count=lanes_left-(LV-1)*V;
  const FloatMask ftail=float_mask(tail_count);
  const Mask tail=lane_mask(tail_count);
  constexpr int B=hvna_broadcasts;
  int b0=0;
  if(p.right_transposed) {
    for(;b0+B<=broadcasts;b0+=B)hvna_columns<FP32,DOUBLES,FULL,LV,B>(p,b0,lane0,ftail,tail);
    if constexpr(B>2)for(;b0+2<=broadcasts;b0+=2)hvna_columns<FP32,DOUBLES,FULL,LV,2>(p,b0,lane0,ftail,tail);
    for(;b0<broadcasts;++b0)hvna_columns<FP32,DOUBLES,FULL,LV,1>(p,b0,lane0,ftail,tail);
  } else {
    for(;b0+B<=broadcasts;b0+=B)hvna_rows<FP32,DOUBLES,FULL,LV,B>(p,lane0,b0,lanes_left,ftail,tail);
    if constexpr(B>2)for(;b0+2<=broadcasts;b0+=2)hvna_rows<FP32,DOUBLES,FULL,LV,2>(p,lane0,b0,lanes_left,ftail,tail);
    for(;b0<broadcasts;++b0)hvna_rows<FP32,DOUBLES,FULL,LV,1>(p,lane0,b0,lanes_left,ftail,tail);
  }
}
template<bool FP32,bool DOUBLES,int LV>
void hvna_chunk_dispatch(const OpenMXVnaHvnaBlock& p,int lane0,int lanes_left,int broadcasts) {
  if(lanes_left-(LV-1)*V==V)hvna_chunk<FP32,DOUBLES,true,LV>(p,lane0,lanes_left,broadcasts);
  else hvna_chunk<FP32,DOUBLES,false,LV>(p,lane0,lanes_left,broadcasts);
}
template<bool FP32,bool DOUBLES> void hvna_block_typed(const OpenMXVnaHvnaBlock& p) {
  const int lanes_total=p.right_transposed?p.columns:p.rows;
  const int broadcasts=p.right_transposed?p.rows:p.columns;
  constexpr int chunk=hvna_lane_vectors*V;
  for(int lane0=0;lane0<lanes_total;lane0+=chunk) {
    const int lanes_left=std::min(chunk,lanes_total-lane0);
    const int vectors=(lanes_left+V-1)/V;
    if constexpr(hvna_lane_vectors>=4) {
      if(vectors==4){hvna_chunk_dispatch<FP32,DOUBLES,4>(p,lane0,lanes_left,broadcasts);continue;}
      if(vectors==3){hvna_chunk_dispatch<FP32,DOUBLES,3>(p,lane0,lanes_left,broadcasts);continue;}
    }
    if(vectors==2)hvna_chunk_dispatch<FP32,DOUBLES,2>(p,lane0,lanes_left,broadcasts);
    else hvna_chunk_dispatch<FP32,DOUBLES,1>(p,lane0,lanes_left,broadcasts);
  }
}
void hvna_block(const OpenMXVnaHvnaBlock& p) {
  if(p.fp32)hvna_block_typed<true,false>(p);
  else if(p.left_double && p.right_double)hvna_block_typed<false,true>(p);
  else hvna_block_typed<false,false>(p);
}
// ---- DS_VNA m sum: lanes over M1, LL in registers ----------------------------
// For fixed (L0, Mul0, L) each pair (M0, M1) receives one update per LL, the
// one with m = M0-M1, so lanes across M1 with LL ascending inside keep the
// legacy per-element order. The reversed harmonic tables make the lanes of one
// M0 a contiguous window; lanes with |m| > LL keep their value for that LL.
template<bool FULL> void angular_lanes(const OpenMXVnaAngular& p,int a,int j0,Mask tail,Vector lane) {
  const int m0=a-p.l0;
  const size_t row=size_t(a)*p.acc_stride;
  Vector acc[8];
  for(auto& v:acc)v=Vector(0.0);
  for(int t=0;t<p.ll_count;++t) {
    const int ll=p.ll_first+t,r0=ll-m0-p.l1;
    const int jlo=std::max(0,-r0),jhi=std::min(p.columns-1,2*ll-r0);
    if(j0>jhi || j0+V-1<jlo)continue;
    const Mask valid=(lane>=Vector(double(jlo)))&&(lane<=Vector(double(jhi)));
    const size_t at=size_t(ll)*p.cy_stride+size_t(p.pad+r0+j0);
    const Vector yr=load(p.cy[0]+at),yi=load(p.cy[1]+at),tr=load(p.cy[2]+at),ti=load(p.cy[3]+at),
        pr=load(p.cy[4]+at),pi=load(p.cy[5]+at);
    const Vector cr(p.cpow_r[t]),ci(p.cpow_i[t]);
    // Cmul(Cpow, Y): r = a.r*b.r - a.i*b.i, i = a.i*b.r + a.r*b.i
    const Vector y1r=cr*yr-ci*yi,y1i=ci*yr+cr*yi;
    const Vector t1r=cr*tr-ci*ti,t1i=ci*tr+cr*ti;
    const Vector p1r=cr*pr-ci*pi,p1i=ci*pr+cr*pi;
    const Vector g=FULL?load(p.gaunt[t]+size_t(a)*p.gaunt_stride+j0)
                       :simd::simd_partial_load<Vector>(p.gaunt[t]+size_t(a)*p.gaunt_stride+j0,tail);
    const Vector tmp2=g*Vector(p.s0[t]),tmp0=g*Vector(p.s1[t]);
    acc[0]=select(valid,acc[0]+y1r*tmp2,acc[0]);acc[1]=select(valid,acc[1]+y1i*tmp2,acc[1]);
    acc[2]=select(valid,acc[2]+y1r*tmp0,acc[2]);acc[3]=select(valid,acc[3]+y1i*tmp0,acc[3]);
    acc[4]=select(valid,acc[4]+t1r*tmp2,acc[4]);acc[5]=select(valid,acc[5]+t1i*tmp2,acc[5]);
    acc[6]=select(valid,acc[6]+p1r*tmp2,acc[6]);acc[7]=select(valid,acc[7]+p1i*tmp2,acc[7]);
  }
  for(int q=0;q<8;++q) {
    double* d=p.acc[q]+row+j0;
    if constexpr(FULL)store(acc[q],d);else simd::simd_partial_store(acc[q],d,tail);
  }
}
void angular(const OpenMXVnaAngular& p) {
  const Vector lanes_d([](auto lane){return double(lane);});
  for(int a=0;a<p.rows;++a)for(int j0=0;j0<p.columns;j0+=V) {
    const int count=std::min(V,p.columns-j0);
    const Mask tail=lane_mask(count);
    const Vector lane=Vector(double(j0))+lanes_d;
    if(count>=V)angular_lanes<true>(p,a,j0,tail,lane);else angular_lanes<false>(p,a,j0,tail,lane);
  }
}

// ---- DS_VNA complex-to-real conversions: lanes over M1 ---------------------
template<bool FULL> Vector load_lanes(const double* d,Mask tail) {
  if constexpr(FULL)return load(d);else return simd::simd_partial_load<Vector>(d,tail);
}
template<bool FULL> void store_lanes(Vector v,double* d,Mask tail) {
  if constexpr(FULL)store(v,d);else simd::simd_partial_store(v,d,tail);
}
constexpr int convert_rows=16,convert_columns=32;
// First pass: C[q][M0][M1] = sum_k conj(conv0[M0][k]) * acc_q[k][M1], legacy Cmul/Cadd order.
template<bool FULL> void convert_first(const OpenMXVnaConvert& p,double (*cm)[8][convert_columns],int a,int j0,Mask tail) {
  Vector sum[8];
  for(auto& v:sum)v=Vector(0.0);
  for(int k=0;k<p.rows0;++k) {
    const double* c=p.conv0+2*(size_t(a)*p.width+size_t(k));
    const Vector ar(c[0]),ai(-c[1]);
    for(int q=0;q<4;++q) {
      const Vector tr=load_lanes<FULL>(p.acc[2*q]+size_t(k)*p.acc_stride+j0,tail);
      const Vector ti=load_lanes<FULL>(p.acc[2*q+1]+size_t(k)*p.acc_stride+j0,tail);
      sum[2*q]+=ar*tr-ai*ti;
      sum[2*q+1]+=ai*tr+ar*ti;
    }
  }
  for(int q=0;q<8;++q)store_lanes<FULL>(sum[q],&cm[a][q][j0],tail);
}
// Second pass: real part of sum_k C[q][M0][k]*convt[k][M1] (the legacy adds a.r*b.r - a.i*b.i per k), then the DS_VNA rows.
template<bool FULL> void convert_second(const OpenMXVnaConvert& p,const double (*cm)[8][convert_columns],int a,int j0,int count,Mask tail) {
  Vector re[4];
  for(auto& v:re)v=Vector(0.0);
  for(int k=0;k<p.columns;++k) {
    const Vector vr=load_lanes<FULL>(p.convt_r+size_t(k)*p.ld+j0,tail),vi=load_lanes<FULL>(p.convt_i+size_t(k)*p.ld+j0,tail);
    const double* c=cm[a][0]+k;   // plane q of row a at c[q*convert_columns]
    for(int q=0;q<4;++q) {
      const Vector mr(c[(2*q)*convert_columns]),mi(c[(2*q+1)*convert_columns]);
      re[q]+=mr*vr-mi*vi;
    }
  }
  const FloatMask fm=float_mask(count);
  const auto emit=[&](float* row,Vector v,double scale) {
    const FloatLanes f(v);          // (Type_DS_VNA) cast
    const Vector d(f);
    simd::simd_partial_store(FloatLanes(Vector(scale)*d),row,fm);
  };
  const size_t at=size_t(a)*p.out_stride+size_t(j0);
  emit(p.out[0]+at,re[0],8.0);
  if(p.onsite) {
    for(int c=1;c<4;++c)simd::simd_partial_store(FloatLanes(0.0f),p.out[c]+at,fm);
  } else {
    Vector v1=Vector(p.c1)*re[1]+Vector(p.c2)*re[2];
    Vector v2=Vector(p.c4)*re[1]+Vector(p.c5)*re[2];
    if(!p.polar){v1=v1-Vector(p.c3)*re[3];v2=v2+Vector(p.c6)*re[3];}
    const Vector v3=Vector(p.c7)*re[1]-Vector(p.c8)*re[2];
    emit(p.out[1]+at,v1,-8.0);emit(p.out[2]+at,v2,-8.0);emit(p.out[3]+at,v3,-8.0);
  }
}
void convert(const OpenMXVnaConvert& p) {
  // [M0][plane][k]: the eight planes of one row stay 256 bytes apart
  alignas(64) double cm[convert_rows][8][convert_columns];
  for(int a=0;a<p.rows0;++a)for(int j0=0;j0<p.columns;j0+=V) {
    const int count=std::min(V,p.columns-j0);
    const Mask tail=lane_mask(count);
    if(count>=V)convert_first<true>(p,cm,a,j0,tail);else convert_first<false>(p,cm,a,j0,tail);
  }
  for(int a=0;a<p.rows0;++a)for(int j0=0;j0<p.columns;j0+=V) {
    const int count=std::min(V,p.columns-j0);
    const Mask tail=lane_mask(count);
    if(count>=V)convert_second<true>(p,cm,a,j0,count,tail);else convert_second<false>(p,cm,a,j0,count,tail);
  }
}
constexpr VnaKernels kernels{V,contract,bessel,quadrature,hvna_block,angular,convert};
}
const VnaKernels& OPENMX_HOST_SIMD_FACTORY(vna)() noexcept {return kernels;}
}

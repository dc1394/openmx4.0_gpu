#include "host_orbitals_kernels.hpp"
#include "host_simd_abi.hpp"
#include <Kokkos_SIMD_Common_Math.hpp>  // per-lane acos/asin/sin/cos; after the ABI header
#include <algorithm>
#include <cstdint>
#include <span>

// Primitive orbital values (points) and their Cartesian derivatives (dpoints)
// on grid points, with SIMD lanes across the points of one species. Both are
// transcriptions of the scalar routines (SOG_point_eval of Set_Orbitals_Grid.c
// and Get_dOrbitals_work of Get_dOrbitals.c): every branch is evaluated for
// all lanes and selected per lane in the same operation order, so each lane
// is bitwise the scalar result. The scalar routines' estimate (points) or
// bisection (dpoints) of the spline interval only decides where the search
// starts; both settle on the first mesh point at or beyond the radius, which
// here comes from a per-species seed table and two masked correction loops.
namespace openmx::core::host {
namespace {
using D=Vector;
using I=IndexVector;
using F=simd::basic_simd<float,HostAbi>;
using DM=typename D::mask_type;
using IM=typename I::mask_type;
constexpr int L0MAX=3,MULMAX=8;
constexpr int MAXNO=MULMAX*(L0MAX+1)*(L0MAX+1);  // orbitals of one species at most

[[gnu::always_inline]] inline D gather(const double* p,size_t n,I index) {
  return simd::unchecked_gather_from<D>(std::span<const double>(p,n),index);
}
[[gnu::always_inline]] inline D select(DM m,D a,D b) { return simd::condition(m,a,b); }
// Exact negation (Kokkos computes -v as 0 - v on AVX, which turns -0 into +0).
[[gnu::always_inline]] inline D negate(D v) { return D(-1.0)*v; }
// Whether any lane is set (Kokkos's any_of tests the lanes one by one).
[[gnu::always_inline]] inline bool any(DM m) {
#if defined(OPENMX_HOST_SIMD_AVX512)
  return static_cast<__mmask8>(m)!=0;
#elif defined(OPENMX_HOST_SIMD_AVX2)
  return _mm256_movemask_pd(static_cast<__m256d>(m))!=0;
#else
  return simd::any_of(m);
#endif
}

constexpr int max_bins=2048;
struct Species {
  int mesh,maxl,no,bins;
  const int* nb;
  const double* rv;
  size_t rwf_base;
  double rm,rv0,rvlast,rv3,seed_scale;
  std::int32_t seed[max_bins];  // first interval whose radius bounds each bin's lower edge
};
void setup(Species& s,const OpenMXOrbitalsTables& t,int wan) {
  s.mesh=t.mesh[wan];s.maxl=t.maxl[wan];s.nb=t.nb+size_t(wan)*size_t(t.l0max+1);
  s.rv=t.rv+t.rv_off[wan];s.rwf_base=t.rwf_base[wan];
  s.rv0=s.rv[0];s.rvlast=s.rv[s.mesh-1];s.rv3=s.rv[3];s.rm=s.rv[4];
  s.no=0;for(int l=0;l<=s.maxl;++l)s.no+=s.nb[l]*(2*l+1);
  // seed[k] = the first interval m in [1, mesh-1] with rv[m] >= k*width:
  // never above the interval of a radius in bin k, so the upward loop
  // finishes the search. Two bins per mesh point keep the setup cheap for
  // small calls; a mesh denser than the bins costs iterations only next to
  // the nucleus.
  s.bins=std::min(max_bins,std::max(64,2*s.mesh));
  const double width=s.rvlast/double(s.bins);
  s.seed_scale=1.0/width;
  for(int k=0,m=1;k<s.bins;++k) {
    const double edge=k*width;
    while(m<s.mesh-1 && s.rv[m]<edge)++m;
    s.seed[k]=m;
  }
}

// The spline interval of every lane: the first mesh point m at or beyond R
// (clamped to [1, mesh-1]), with lo = rv[m-1] and hi = rv[m] carried out of
// the correction loops; interval 4 for radii inside the first mesh point.
// m is kept as a small exact double so that every mask is a double mask.
struct Interval { D m,lo,hi; };
[[gnu::always_inline]] inline Interval interval(const Species& s,D R,DM interior,DM below) {
  const D zero(0.0),one(1.0),last(double(s.mesh-1));
  const size_t mesh=size_t(s.mesh);
  D bin=select(interior,R*D(s.seed_scale),zero);
  bin=select(bin>D(double(s.bins-1)),D(double(s.bins-1)),bin);
  Interval v;
  v.m=D(simd::unchecked_gather_from<I>(std::span<const std::int32_t>(s.seed,size_t(s.bins)),I(bin)));
  v.lo=gather(s.rv,mesh,I(v.m-one));
  for(;;) {
    const DM active=interior&&(one<v.m)&&(R<=v.lo);
    if(!any(active))break;
    v.m=select(active,v.m-one,v.m);
    v.lo=gather(s.rv,mesh,I(v.m-one));
  }
  v.hi=gather(s.rv,mesh,I(v.m));
  for(;;) {
    const DM active=interior&&(v.m<last)&&(v.hi<R);
    if(!any(active))break;
    v.m=select(active,v.m+one,v.m);
    v.lo=select(active,v.hi,v.lo);
    v.hi=gather(s.rv,mesh,I(v.m));
  }
  v.m=select(below,D(4.0),v.m);
  v.lo=select(below,D(s.rv3),v.lo);
  v.hi=select(below,D(s.rm),v.hi);
  return v;
}

// The interval's constants of the cubic spline (the scalar routines' h1..h3,
// x1, x2, y1, y2, dum1..dum4) and the four sample indices.
struct Spline {
  I im1,im,i1,i4;
  D h2,y1,y2,y12,y22,dum1,dum2,dum3,dum4;
};
[[gnu::always_inline]] inline Spline spline(const Species& s,D R,DM below,const Interval& v) {
  const D one(1.0),two(2.0),last(double(s.mesh-1));
  const size_t mesh=size_t(s.mesh);
  Spline sp;
  sp.im1=I(v.m-one);sp.im=I(v.m);sp.i1=I(select(v.m==one,v.m+one,v.m-two));sp.i4=I(select(v.m==last,v.m-two,v.m+one));
  const D h1=v.lo-gather(s.rv,mesh,sp.i1),h3=gather(s.rv,mesh,sp.i4)-v.hi;
  sp.h2=v.hi-v.lo;
  const D Rb=select(below,D(s.rm),R);
  const D x1=Rb-v.lo,x2=Rb-v.hi;
  sp.y1=x1/sp.h2;sp.y2=x2/sp.h2;sp.y12=sp.y1*sp.y1;sp.y22=sp.y2*sp.y2;
  D dh=h1+sp.h2;
  sp.dum1=h1/sp.h2/dh;sp.dum2=sp.h2/h1/dh;
  dh=sp.h2+h3;
  sp.dum3=sp.h2/h3/dh;sp.dum4=h3/sp.h2/dh;
  return sp;
}
// The spline value f and derivative df of one radial function at the lanes'
// intervals (Get_dOrbitals_work's mesh-branch df equals the other branch's:
// y2*y2 there is the same product as y22).
[[gnu::always_inline]] inline void radial(const Spline& sp,const double* rw,size_t mesh,D& f,D& df) {
  const D two(2.0),three(3.0);
  const D f1=gather(rw,mesh,sp.i1),f2=gather(rw,mesh,sp.im1),f3=gather(rw,mesh,sp.im),f4=gather(rw,mesh,sp.i4);
  const D d32=f3-f2;
  const D g1=d32*sp.dum1+(f2-f1)*sp.dum2;
  const D g2=(f4-f3)*sp.dum3+d32*sp.dum4;
  f=sp.y22*(three*f2+sp.h2*g1+(two*f2+sp.h2*g1)*sp.y2)+sp.y12*(three*f3-sp.h2*g2-(two*f3-sp.h2*g2)*sp.y1);
  df=two*sp.y2/sp.h2*(three*f2+sp.h2*g1+(two*f2+sp.h2*g1)*sp.y2)+sp.y22*(two*f2+sp.h2*g1)/sp.h2
      +two*sp.y1/sp.h2*(three*f3-sp.h2*g2-(two*f3-sp.h2*g2)*sp.y1)-sp.y12*(two*f3-sp.h2*g2)/sp.h2;
}
// The inner extrapolation's cubic coefficients of one radial function.
[[gnu::always_inline]] inline void extrapolation(int L0,D rm,D f,D df,D& a,D& b,D& c,D& d) {
  const D zero(0.0),two(2.0),three(3.0);
  if(L0==0){a=zero;b=D(0.5)*df/rm;c=zero;d=f-b*rm*rm;}
  else if(L0==1){a=(rm*df-f)/(two*rm*rm*rm);b=zero;c=df-three*a*rm*rm;d=zero;}
  else{b=(three*f-rm*df)/(rm*rm);a=(f-b*rm*rm)/(rm*rm*rm);c=zero;d=zero;}
}

// Get_Grid_XYZ of every lane's grid index (the integer quotients, exact in
// doubles: a rounded quotient is corrected by at most one), plus the cell
// translation (a broadcast when the block shares one cell), minus the atom
// position and its shift: the device routine's arithmetic, per lane.
struct Displacement { D x,y,z; };
[[gnu::always_inline]] inline Displacement displacement(const OpenMXOrbitalsTables& t,const OpenMXOrbitalsFrame& f,
    const std::int32_t* grida,const std::int32_t* cella,bool uniform,D gx,D gy,D gz,D ax,D ay,D az) {
  const D one(1.0);
  const D gn=D(simd::simd_unchecked_load<I>(grida)),ng23(double(f.ng23)),ng3(double(f.ng3));
  D n1=D(I(gn/ng23));
  n1=select(n1*ng23>gn,n1-one,n1);n1=select((n1+one)*ng23<=gn,n1+one,n1);
  const D rest=gn-n1*ng23;
  D n2=D(I(rest/ng3));
  n2=select(n2*ng3>rest,n2-one,n2);n2=select((n2+one)*ng3<=rest,n2+one,n2);
  const D n3=rest-n2*ng3;
  D tx,ty,tz;
  if(uniform) {
    const double* c=t.translations+cella[0];
    tx=D(c[0]);ty=D(c[1]);tz=D(c[2]);
  } else {
    const I cell3=simd::simd_unchecked_load<I>(cella);
    const size_t shifts=3*size_t(t.cells);
    tx=gather(t.translations,shifts,cell3);ty=gather(t.translations,shifts,cell3+I(1));tz=gather(t.translations,shifts,cell3+I(2));
  }
  Displacement d;
  d.x=(n1*D(f.frame[0])+n2*D(f.frame[3])+n3*D(f.frame[6])+D(f.origin[0]))+tx-gx-ax;
  d.y=(n1*D(f.frame[1])+n2*D(f.frame[4])+n3*D(f.frame[7])+D(f.origin[1]))+ty-gy-ay;
  d.z=(n1*D(f.frame[2])+n2*D(f.frame[5])+n3*D(f.frame[8])+D(f.origin[2]))+tz-gz-az;
  return d;
}

// The orbital values at the lanes' displacements: sink(j, value) for every
// orbital j of the species in order, zero outside the PAO mesh.
template<class Sink>
[[gnu::always_inline]] inline void values(const OpenMXOrbitalsTables& t,const Species& s,const Displacement& p,Sink&& sink) {
  const D zero(0.0),one(1.0),minus(-1.0),two(2.0),three(3.0);
  const D x=p.x,y=p.y,z=p.z;
  // Direction cosines without the angles (the device routine's branches).
  const D dum=x*x+y*y;
  const D r=Kokkos::sqrt(dum+z*z),r1=Kokkos::sqrt(dum);
  const DM normal=D(10e-15)<=r,zaxis=r<Kokkos::abs(z),inplane=normal&&(D(10e-15)<=r1),yaxis=r1<Kokkos::abs(y);
  const D sgnz=select(z<zero,minus,one),sgny=select(y<zero,minus,one);
  const D coQ=select(normal,select(zaxis,sgnz,z/r),zero);
  const D siQ=select(normal,select(zaxis,zero,r1/r),one);
  const D siP=select(inplane,select(yaxis,sgny,y/r1),zero);
  const D coP=select(inplane,select(yaxis,zero,x/r1),one);
  const D R=r;
  // Radial functions of every (L0, Mul0): the spline, or inside the first
  // mesh point the cubic extrapolation (only when a lane needs it).
  const DM outside=D(s.rvlast)<R,below=R<D(s.rv0),interior=!outside&&!below;
  const Spline sp=spline(s,R,below,interval(s,R,interior,below));
  D RF[L0MAX+1][MULMAX];
  const D rm(s.rm);
  const bool extrapolate=any(below);
  size_t row=s.rwf_base;
  for(int L0=0;L0<=s.maxl;++L0)for(int Mul0=0;Mul0<s.nb[L0];++Mul0) {
    const double* rw=t.rwf+row;row+=size_t(s.mesh);
    D fv,df;radial(sp,rw,size_t(s.mesh),fv,df);
    if(!extrapolate){RF[L0][Mul0]=fv;continue;}
    D a,b,c,d;extrapolation(L0,rm,fv,df,a,b,c,d);
    RF[L0][Mul0]=select(below,a*R*R*R+b*R*R+c*R+d,fv);
  }
  // Real harmonics up to L0 = 3.
  D AF[L0MAX+1][2*L0MAX+1];
  AF[0][0]=D(0.282094791773878);
  if(s.maxl>=1) {
    const D d1=D(0.48860251190292)*siQ;
    AF[1][0]=d1*coP;AF[1][1]=d1*siP;AF[1][2]=D(0.48860251190292)*coQ;
  }
  if(s.maxl>=2) {
    const D d1=siQ*siQ,d2=D(1.09254843059208)*siQ*coQ;
    AF[2][0]=D(0.94617469575756)*coQ*coQ-D(0.31539156525252);
    AF[2][1]=D(0.54627421529604)*d1*(one-two*siP*siP);
    AF[2][2]=D(1.09254843059208)*d1*siP*coP;
    AF[2][3]=d2*coP;AF[2][4]=d2*siP;
  }
  if(s.maxl>=3) {
    AF[3][0]=D(0.373176332590116)*(D(5.0)*coQ*coQ*coQ-three*coQ);
    AF[3][1]=D(0.457045799464466)*coP*siQ*(D(5.0)*coQ*coQ-one);
    AF[3][2]=D(0.457045799464466)*siP*siQ*(D(5.0)*coQ*coQ-one);
    AF[3][3]=D(1.44530572132028)*siQ*siQ*coQ*(coP*coP-siP*siP);
    AF[3][4]=D(2.89061144264055)*siQ*siQ*coQ*siP*coP;
    AF[3][5]=D(0.590043589926644)*siQ*siQ*siQ*(D(4.0)*coP*coP*coP-three*coP);
    AF[3][6]=D(0.590043589926644)*siQ*siQ*siQ*(three*siP-D(4.0)*siP*siP*siP);
  }
  int index=0;
  for(int L0=0;L0<=s.maxl;++L0)for(int Mul0=0;Mul0<s.nb[L0];++Mul0)for(int M0=0;M0<=2*L0;++M0)
    sink(index++,select(outside,zero,RF[L0][Mul0]*AF[L0][M0]));
}

// One block of `lanes` point records of one species; lanes >= n replay
// point n-1 and their output is masked out. Rows of max_no floats, zero
// beyond the species' orbitals.
void block(const OpenMXOrbitalsTables& t,const OpenMXOrbitalsFrame& f,const Species& s,
    const OpenMXOrbitalsPoint* q,int n,float* out) {
  alignas(64) double gxa[lanes],gya[lanes],gza[lanes],axa[lanes],aya[lanes],aza[lanes];
  alignas(64) std::int32_t grida[lanes],cella[lanes];
  bool uniform=true;
  for(int k=0;k<lanes;++k) {
    const OpenMXOrbitalsPoint& e=q[std::min(k,n-1)];
    grida[k]=e.grid;cella[k]=3*e.cell;uniform=uniform && cella[k]==cella[0];
    gxa[k]=e.xyz[0];gya[k]=e.xyz[1];gza[k]=e.xyz[2];axa[k]=e.shift[0];aya[k]=e.shift[1];aza[k]=e.shift[2];
  }
  const Displacement p=displacement(t,f,grida,cella,uniform,load(gxa),load(gya),load(gza),load(axa),load(aya),load(aza));
  const I lane([](auto k){return std::int32_t(k);});
  const IM active=lane<I(n);
  const I rows=lane*I(f.max_no);
  std::span<float> destination(out,size_t(n)*size_t(f.max_no));
  values(t,s,p,[&](int j,D v){simd::unchecked_scatter_to(F(v),destination,active,rows+I(j));});
  for(int j=s.no;j<f.max_no;++j)simd::unchecked_scatter_to(F(D(0.0)),destination,active,rows+I(j));
}

// One block of `lanes` points of a segment, from point `first` on; lanes
// >= n replay the last point and are not stored. Each point's no floats go
// to its own row: by a masked scatter when the block's rows are contiguous,
// else through a transposed scratch.
void sblock(const OpenMXOrbitalsTables& t,const OpenMXOrbitalsFrame& f,const Species& s,
    const OpenMXOrbitalsSegment& g,size_t first,int n) {
  alignas(64) std::int32_t grida[lanes],cella[lanes];
  bool uniform=true;
  for(int k=0;k<lanes;++k) {
    const size_t p=first+size_t(std::min(k,n-1)),i=g.index?size_t(g.index[p]):p;
    grida[k]=g.grid[i];cella[k]=3*g.cell[i];uniform=uniform && cella[k]==cella[0];
  }
  const Displacement p=displacement(t,f,grida,cella,uniform,D(g.xyz[0]),D(g.xyz[1]),D(g.xyz[2]),
      D(g.shift[0]),D(g.shift[1]),D(g.shift[2]));
  float* const* rows=g.rows+first;
  bool contiguous=true;
  for(int k=1;k<n;++k)contiguous=contiguous && rows[k]==rows[0]+size_t(k)*size_t(s.no);
  if(contiguous) {
    const I lane([](auto k){return std::int32_t(k);});
    const IM active=lane<I(n);
    const I base=lane*I(s.no);
    std::span<float> destination(rows[0],size_t(n)*size_t(s.no));
    values(t,s,p,[&](int j,D v){simd::unchecked_scatter_to(F(v),destination,active,base+I(j));});
  } else {
    alignas(64) float scratch[MAXNO][lanes];
    values(t,s,p,[&](int j,D v){simd::simd_unchecked_store(F(v),&scratch[j][0]);});
    for(int k=0;k<n;++k){float* row=rows[k];for(int j=0;j<s.no;++j)row[j]=scratch[j][k];}
  }
}

void points(const OpenMXOrbitalsTables& t,const OpenMXOrbitalsFrame& f,const OpenMXOrbitalsPoint* q,size_t count,float* out) {
  for(size_t first=0;first<count;) {
    const int wan=q[first].wan;
    size_t last=first+1;
    while(last<count && q[last].wan==wan)++last;
    Species s;setup(s,t,wan);
    for(size_t i=first;i<last;i+=lanes)
      block(t,f,s,q+i,int(std::min<size_t>(lanes,last-i)),out+i*size_t(f.max_no));
    first=last;
  }
}

void segments(const OpenMXOrbitalsTables& t,const OpenMXOrbitalsFrame& f,const OpenMXOrbitalsSegment* g,size_t count) {
  Species s;int current=-1;
  for(size_t i=0;i<count;++i) {
    const OpenMXOrbitalsSegment& e=g[i];
    if(!e.count)continue;
    if(e.wan!=current){setup(s,t,e.wan);current=e.wan;}
    for(size_t first=0;first<e.count;first+=lanes)
      sblock(t,f,s,e,first,int(std::min<size_t>(lanes,e.count-first)));
  }
}

// One block of `lanes` displacements of one species: Get_dOrbitals_work. With
// exact_angles the angles come from per-lane acos/asin/sin/cos (this build's
// libm), as in the scalar routine; otherwise sin/cos of theta and phi are the
// direction cosines r1/R, z/R, y/r1, x/r1 of the same branches (last-ulp
// differences, no libm). The derivative of every orbital is written per lane
// as three consecutive doubles.
template<bool exact_angles>
void dblock(const OpenMXOrbitalsTables& t,const Species& s,const OpenMXOrbitalsDisplacement* q,int n,double* out) {
  alignas(64) double xa[lanes],ya[lanes],za[lanes];
  for(int k=0;k<lanes;++k){const OpenMXOrbitalsDisplacement& e=q[std::min(k,n-1)];xa[k]=e.x;ya[k]=e.y;za[k]=e.z;}
  const D zero(0.0),one(1.0),minus(-1.0),two(2.0),three(3.0),rmin(10e-14),minr(10e-15),pi(3.1415926535897932384626);
  D x=load(xa),y=load(ya),z=load(za);
  // xyz2spherical's radius decides the Rmin shift of a point on the atom; the
  // spherical coordinates then follow from the (possibly shifted) point.
  {
    const D dum=x*x+y*y;
    const DM small=Kokkos::sqrt(dum+z*z)<rmin;
    x=select(small,x+rmin,x);y=select(small,y+rmin,y);z=select(small,z+rmin,z);
  }
  const D dum=x*x+y*y;
  const D R=Kokkos::sqrt(dum+z*z),r1=Kokkos::sqrt(dum);
  const DM normal=minr<=R,zaxis=R<Kokkos::abs(z),inplane=normal&&(minr<=r1),yaxis=r1<Kokkos::abs(y);
  const D sgnz=select(z<zero,minus,one),sgny=select(y<zero,minus,one);
  const D cosine=select(zaxis,sgnz*one,z/R),sine=select(yaxis,sgny*one,y/r1);
  D siQ,coQ,siP,coP;
  if constexpr(exact_angles) {
    const D Q=select(normal,Kokkos::acos(cosine),D(0.5)*pi);
    const D arc=Kokkos::asin(sine);
    const D P=select(inplane,select(zero<=x,arc,pi-arc),zero);
    siQ=Kokkos::sin(Q);coQ=Kokkos::cos(Q);siP=Kokkos::sin(P);coP=Kokkos::cos(P);
  } else {
    coQ=select(normal,cosine,zero);
    siQ=select(normal,select(zaxis,zero,r1/R),one);
    siP=select(inplane,sine,zero);
    coP=select(inplane,select(yaxis,zero,x/r1),one);
  }
  // Radial functions and their derivatives.
  const DM outside=D(s.rvlast)<R,below=R<D(s.rv0),interior=!outside&&!below;
  const Spline sp=spline(s,R,below,interval(s,R,interior,below));
  D RF[L0MAX+1][MULMAX],dRF[L0MAX+1][MULMAX];
  const D rm(s.rm);
  const bool extrapolate=any(below);
  size_t row=s.rwf_base;
  for(int L0=0;L0<=s.maxl;++L0)for(int Mul0=0;Mul0<s.nb[L0];++Mul0) {
    const double* rw=t.rwf+row;row+=size_t(s.mesh);
    D fv,df;radial(sp,rw,size_t(s.mesh),fv,df);
    if(!extrapolate){RF[L0][Mul0]=fv;dRF[L0][Mul0]=df;continue;}
    D a,b,c,d;extrapolation(L0,rm,fv,df,a,b,c,d);
    RF[L0][Mul0]=select(below,a*R*R*R+b*R*R+c*R+d,fv);
    dRF[L0][Mul0]=select(below,three*a*R*R+two*b*R+c,df);
  }
  // dr/dx,y,z, dQ/dx,y,z, dP/dx,y,z and the angular functions.
  const D dRx=siQ*coP,dRy=siQ*siP,dRz=coQ;
  const DM beyond=rmin<R;
  const D dQx=select(beyond,coQ*coP/R,zero),dQy=select(beyond,coQ*siP/R,zero),dQz=select(beyond,negate(siQ)/R,zero);
  const D dPx=select(beyond,negate(siP)/R,zero),dPy=select(beyond,coP/R,zero),dPz=zero;
  D AF[L0MAX+1][2*L0MAX+1],dAFQ[L0MAX+1][2*L0MAX+1],dAFP[L0MAX+1][2*L0MAX+1];
  AF[0][0]=D(0.282094791773878);dAFQ[0][0]=zero;dAFP[0][0]=zero;
  if(s.maxl>=1) {
    const D d1=D(0.48860251190292)*siQ;
    AF[1][0]=d1*coP;AF[1][1]=d1*siP;AF[1][2]=D(0.48860251190292)*coQ;
    dAFQ[1][0]=D(0.48860251190292)*coQ*coP;dAFQ[1][1]=D(0.48860251190292)*coQ*siP;dAFQ[1][2]=D(-0.48860251190292)*siQ;
    dAFP[1][0]=D(-0.48860251190292)*siP;dAFP[1][1]=D(0.48860251190292)*coP;dAFP[1][2]=zero;
  }
  if(s.maxl>=2) {
    const D d1=siQ*siQ,d2=D(1.09254843059208)*siQ*coQ;
    AF[2][0]=D(0.94617469575756)*coQ*coQ-D(0.31539156525252);
    AF[2][1]=D(0.54627421529604)*d1*(one-two*siP*siP);
    AF[2][2]=D(1.09254843059208)*d1*siP*coP;
    AF[2][3]=d2*coP;AF[2][4]=d2*siP;
    dAFQ[2][0]=D(-1.89234939151512)*siQ*coQ;
    dAFQ[2][1]=D(1.09254843059208)*siQ*coQ*(one-two*siP*siP);
    dAFQ[2][2]=D(2.18509686118416)*siQ*coQ*siP*coP;
    dAFQ[2][3]=D(1.09254843059208)*(one-two*siQ*siQ)*coP;
    dAFQ[2][4]=D(1.09254843059208)*(one-two*siQ*siQ)*siP;
    dAFP[2][0]=zero;
    dAFP[2][1]=D(-2.18509686118416)*siQ*siP*coP;
    dAFP[2][2]=D(1.09254843059208)*siQ*(one-two*siP*siP);
    dAFP[2][3]=D(-1.09254843059208)*coQ*siP;
    dAFP[2][4]=D(1.09254843059208)*coQ*coP;
  }
  if(s.maxl>=3) {
    AF[3][0]=D(0.373176332590116)*(D(5.0)*coQ*coQ*coQ-three*coQ);
    AF[3][1]=D(0.457045799464466)*coP*siQ*(D(5.0)*coQ*coQ-one);
    AF[3][2]=D(0.457045799464466)*siP*siQ*(D(5.0)*coQ*coQ-one);
    AF[3][3]=D(1.44530572132028)*siQ*siQ*coQ*(coP*coP-siP*siP);
    AF[3][4]=D(2.89061144264055)*siQ*siQ*coQ*siP*coP;
    AF[3][5]=D(0.590043589926644)*siQ*siQ*siQ*(D(4.0)*coP*coP*coP-three*coP);
    AF[3][6]=D(0.590043589926644)*siQ*siQ*siQ*(three*siP-D(4.0)*siP*siP*siP);
    dAFQ[3][0]=D(0.373176332590116)*siQ*(D(-15.0)*coQ*coQ+three);
    dAFQ[3][1]=D(0.457045799464466)*coP*coQ*(D(15.0)*coQ*coQ-D(11.0));
    dAFQ[3][2]=D(0.457045799464466)*siP*coQ*(D(15.0)*coQ*coQ-D(11.0));
    dAFQ[3][3]=D(1.44530572132028)*(coP*coP-siP*siP)*siQ*(two*coQ*coQ-siQ*siQ);
    dAFQ[3][4]=D(2.89061144264055)*coP*siP*siQ*(two*coQ*coQ-siQ*siQ);
    dAFQ[3][5]=D(1.770130769779932)*coP*coQ*siQ*siQ*(D(-3.0)+D(4.0)*coP*coP);
    dAFQ[3][6]=D(1.770130769779932)*coQ*siP*siQ*siQ*(three-D(4.0)*siP*siP);
    dAFP[3][0]=zero;
    dAFP[3][1]=D(0.457045799464466)*siP*(D(-5.0)*coQ*coQ+one);
    dAFP[3][2]=D(0.457045799464466)*coP*(D(5.0)*coQ*coQ-one);
    dAFP[3][3]=D(-5.781222885281120)*coP*coQ*siP*siQ;
    dAFP[3][4]=D(2.89061144264055)*coQ*siQ*(coP*coP-siP*siP);
    dAFP[3][5]=D(1.770130769779932)*siP*siQ*siQ*(one-D(4.0)*coP*coP);
    dAFP[3][6]=D(1.770130769779932)*coP*siQ*siQ*(one-D(4.0)*siP*siP);
  }
  // dChi/dx,y,z per orbital, zero outside the mesh.
  const I lane([](auto k){return std::int32_t(k);});
  const IM active=lane<I(n);
  const I rows=lane*I(3*s.no);
  std::span<double> destination(out,size_t(n)*size_t(s.no)*3);
  int index=0;
  for(int L0=0;L0<=s.maxl;++L0)for(int Mul0=0;Mul0<s.nb[L0];++Mul0)for(int M0=0;M0<=2*L0;++M0) {
    const D dChiR=dRF[L0][Mul0]*AF[L0][M0],dChiQ=RF[L0][Mul0]*dAFQ[L0][M0],dChiP=RF[L0][Mul0]*dAFP[L0][M0];
    simd::unchecked_scatter_to(select(outside,zero,negate(dRx)*dChiR-dQx*dChiQ-dPx*dChiP),destination,active,rows+I(3*index));
    simd::unchecked_scatter_to(select(outside,zero,negate(dRy)*dChiR-dQy*dChiQ-dPy*dChiP),destination,active,rows+I(3*index+1));
    simd::unchecked_scatter_to(select(outside,zero,negate(dRz)*dChiR-dQz*dChiQ-dPz*dChiP),destination,active,rows+I(3*index+2));
    ++index;
  }
}

// Rows follow the points: point i's row starts at 3 * (sum of the orbital
// counts of the points before it), i.e. at i*no*3 when one species is used.
template<bool exact_angles>
void dpoints_impl(const OpenMXOrbitalsTables& t,const OpenMXOrbitalsDisplacement* q,size_t count,double* out) {
  size_t offset=0;
  for(size_t first=0;first<count;) {
    const int wan=q[first].wan;
    size_t last=first+1;
    while(last<count && q[last].wan==wan)++last;
    Species s;setup(s,t,wan);
    for(size_t i=first;i<last;i+=lanes)
      dblock<exact_angles>(t,s,q+i,int(std::min<size_t>(lanes,last-i)),out+offset+(i-first)*size_t(s.no)*3);
    offset+=(last-first)*size_t(s.no)*3;
    first=last;
  }
}
void dpoints(const OpenMXOrbitalsTables& t,const OpenMXOrbitalsDisplacement* q,size_t count,double* out) {
  dpoints_impl<true>(t,q,count,out);
}
void dpoints_cosines(const OpenMXOrbitalsTables& t,const OpenMXOrbitalsDisplacement* q,size_t count,double* out) {
  dpoints_impl<false>(t,q,count,out);
}
constexpr OrbitalsKernels kernels{lanes,points,dpoints,dpoints_cosines,segments};
}
const OrbitalsKernels& OPENMX_HOST_SIMD_FACTORY(orbitals)() noexcept {return kernels;}
}

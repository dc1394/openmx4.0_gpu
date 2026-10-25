#include "host_orbitals_kernels.hpp"
#include "host_simd_abi.hpp"
#include <algorithm>
#include <cstdint>
#include <span>

// Primitive orbital values on grid points, with SIMD lanes across the points
// of one species: a transcription of SOG_point_eval (Set_Orbitals_Grid.c),
// every branch of the scalar routine evaluated for all lanes and selected per
// lane in the same operation order, so each lane is bitwise the scalar result.
// The scalar routine's logarithmic-mesh estimate of the spline interval only
// seeds its two correction loops, and the interval they settle on does not
// depend on the seed; here the seed comes from a per-species table indexed by
// the radius (no lane-wise log()) and the loops run as masked iterations.
namespace openmx::core::host {
namespace {
using D=Vector;
using I=IndexVector;
using F=simd::basic_simd<float,HostAbi>;
using DM=typename D::mask_type;
using IM=typename I::mask_type;
constexpr int L0MAX=3,MULMAX=8;

D gather(const double* p,size_t n,I index) {
  return simd::unchecked_gather_from<D>(std::span<const double>(p,n),index);
}
D select(DM m,D a,D b) { return simd::condition(m,a,b); }

constexpr int seed_bins=2048;
struct Species {
  int mesh,maxl;
  const int* nb;
  const double* rv;
  size_t rwf_base;
  double rm,rv0,rvlast,seed_scale;
  std::int32_t seed[seed_bins];  // first interval whose radius bounds each bin's lower edge
};

// One block of `lanes` points of one species; lanes >= n replay point n-1 and
// their output is masked out.
void block(const OpenMXOrbitalsTables& t,const OpenMXOrbitalsFrame& f,const Species& s,
    const OpenMXOrbitalsPoint* q,int n,float* out) {
  alignas(64) double gxa[lanes],gya[lanes],gza[lanes],axa[lanes],aya[lanes],aza[lanes];
  alignas(64) std::int32_t grida[lanes],cella[lanes];
  for(int k=0;k<lanes;++k) {
    const OpenMXOrbitalsPoint& e=q[std::min(k,n-1)];
    grida[k]=e.grid;cella[k]=3*e.cell;
    gxa[k]=e.xyz[0];gya[k]=e.xyz[1];gza[k]=e.xyz[2];axa[k]=e.shift[0];aya[k]=e.shift[1];aza[k]=e.shift[2];
  }
  const D zero(0.0),one(1.0),minus(-1.0),two(2.0),three(3.0);
  // Get_Grid_XYZ: the integer quotients of the grid index, exact in doubles
  // (a rounded quotient is corrected by at most one), then the displacement
  // from the shifted atom.
  const D gn=D(simd::simd_unchecked_load<I>(grida)),ng23(double(f.ng23)),ng3(double(f.ng3));
  D n1=D(I(gn/ng23));
  n1=select(n1*ng23>gn,n1-one,n1);n1=select((n1+one)*ng23<=gn,n1+one,n1);
  const D rest=gn-n1*ng23;
  D n2=D(I(rest/ng3));
  n2=select(n2*ng3>rest,n2-one,n2);n2=select((n2+one)*ng3<=rest,n2+one,n2);
  const D n3=rest-n2*ng3;
  const I cell3=simd::simd_unchecked_load<I>(cella);
  const size_t shifts=3*size_t(t.cells);
  const D x=(n1*D(f.frame[0])+n2*D(f.frame[3])+n3*D(f.frame[6])+D(f.origin[0]))+gather(t.translations,shifts,cell3)-load(gxa)-load(axa);
  const D y=(n1*D(f.frame[1])+n2*D(f.frame[4])+n3*D(f.frame[7])+D(f.origin[1]))+gather(t.translations,shifts,cell3+I(1))-load(gya)-load(aya);
  const D z=(n1*D(f.frame[2])+n2*D(f.frame[5])+n3*D(f.frame[8])+D(f.origin[2]))+gather(t.translations,shifts,cell3+I(2))-load(gza)-load(aza);
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
  // The spline interval: a seed from the species table, then the exact
  // neighbours by the scalar routine's two loops (the interval they settle on
  // does not depend on the seed); points inside the first mesh point use
  // interval 4. The index stays a small exact double so that every mask is a
  // double mask; it becomes an int32 only as a gather index.
  const int mesh=s.mesh;
  const D last(double(mesh-1));
  const DM outside=D(s.rvlast)<R,below=R<D(s.rv0),interior=!outside&&!below;
  D bin=select(interior,R*D(s.seed_scale),zero);
  bin=select(bin>D(double(seed_bins-1)),D(double(seed_bins-1)),bin);
  D m=D(simd::unchecked_gather_from<I>(std::span<const std::int32_t>(s.seed,size_t(seed_bins)),I(bin)));
  for(;;) {
    const DM active=interior&&(one<m)&&(R<=gather(s.rv,size_t(mesh),I(m-one)));
    if(!simd::any_of(active))break;
    m=select(active,m-one,m);
  }
  for(;;) {
    const DM active=interior&&(m<last)&&(gather(s.rv,size_t(mesh),I(m))<R);
    if(!simd::any_of(active))break;
    m=select(active,m+one,m);
  }
  m=select(below,D(4.0),m);
  const I im1=I(m-one),im=I(m),i1=I(select(m==one,m+one,m-two)),i4=I(select(m==last,m-two,m+one));
  const D rvm1=gather(s.rv,size_t(mesh),im1),rvm=gather(s.rv,size_t(mesh),im);
  const D h1=rvm1-gather(s.rv,size_t(mesh),i1),h2=rvm-rvm1,h3=gather(s.rv,size_t(mesh),i4)-rvm;
  const D Rb=select(below,D(s.rm),R);
  const D x1=Rb-rvm1,x2=Rb-rvm;
  const D y1=x1/h2,y2=x2/h2,y12=y1*y1,y22=y2*y2;
  D dh=h1+h2;
  const D dum1=h1/h2/dh,dum2=h2/h1/dh;
  dh=h2+h3;
  const D dum3=h2/h3/dh,dum4=h3/h2/dh;
  // Radial functions of every (L0, Mul0).
  D RF[L0MAX+1][MULMAX];
  const D rm(s.rm);
  const bool extrapolate=simd::any_of(below);
  size_t row=s.rwf_base;
  for(int L0=0;L0<=s.maxl;++L0)for(int Mul0=0;Mul0<s.nb[L0];++Mul0) {
    const double* rw=t.rwf+row;row+=size_t(mesh);
    const D f1=gather(rw,size_t(mesh),i1),f2=gather(rw,size_t(mesh),im1),f3=gather(rw,size_t(mesh),im),f4=gather(rw,size_t(mesh),i4);
    const D d32=f3-f2;
    const D g1=d32*dum1+(f2-f1)*dum2;
    const D g2=(f4-f3)*dum3+d32*dum4;
    const D fv=y22*(three*f2+h2*g1+(two*f2+h2*g1)*y2)+y12*(three*f3-h2*g2-(two*f3-h2*g2)*y1);
    if(!extrapolate){RF[L0][Mul0]=fv;continue;}
    const D df=two*y2/h2*(three*f2+h2*g1+(two*f2+h2*g1)*y2)+y22*(two*f2+h2*g1)/h2
        +two*y1/h2*(three*f3-h2*g2-(two*f3-h2*g2)*y1)-y12*(two*f3-h2*g2)/h2;
    D a,b,c,d;
    if(L0==0){a=zero;b=D(0.5)*df/rm;c=zero;d=fv-b*rm*rm;}
    else if(L0==1){a=(rm*df-fv)/(two*rm*rm*rm);b=zero;c=df-three*a*rm*rm;d=zero;}
    else{b=(three*fv-rm*df)/(rm*rm);a=(fv-b*rm*rm)/(rm*rm*rm);c=zero;d=zero;}
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
  // Chi = RF*AF per orbital, zero outside the mesh and beyond the species.
  const I lane([](auto k){return std::int32_t(k);});
  const IM active=lane<I(n);
  const I rows=lane*I(f.max_no);
  std::span<float> destination(out,size_t(n)*size_t(f.max_no));
  int index=0;
  for(int L0=0;L0<=s.maxl;++L0)for(int Mul0=0;Mul0<s.nb[L0];++Mul0)for(int M0=0;M0<=2*L0;++M0) {
    const D value=select(outside,zero,RF[L0][Mul0]*AF[L0][M0]);
    simd::unchecked_scatter_to(F(value),destination,active,rows+I(index));
    ++index;
  }
  for(;index<f.max_no;++index)simd::unchecked_scatter_to(F(zero),destination,active,rows+I(index));
}

void points(const OpenMXOrbitalsTables& t,const OpenMXOrbitalsFrame& f,const OpenMXOrbitalsPoint* q,size_t count,float* out) {
  for(size_t first=0;first<count;) {
    const int wan=q[first].wan;
    size_t last=first+1;
    while(last<count && q[last].wan==wan)++last;
    Species s;
    s.mesh=t.mesh[wan];s.maxl=t.maxl[wan];s.nb=t.nb+size_t(wan)*size_t(t.l0max+1);
    s.rv=t.rv+t.rv_off[wan];s.rwf_base=t.rwf_base[wan];
    s.rv0=s.rv[0];s.rvlast=s.rv[s.mesh-1];s.rm=s.rv[4];
    // seed[k] = the first interval m in [1, mesh-1] with rv[m] >= k*width:
    // never above the interval of a radius in bin k, so the upward loop
    // finishes the search (a mesh denser than the bins costs iterations only
    // next to the nucleus).
    const double width=s.rvlast/double(seed_bins);
    s.seed_scale=1.0/width;
    for(int k=0,m=1;k<seed_bins;++k) {
      const double edge=k*width;
      while(m<s.mesh-1 && s.rv[m]<edge)++m;
      s.seed[k]=m;
    }
    for(size_t i=first;i<last;i+=lanes)
      block(t,f,s,q+i,int(std::min<size_t>(lanes,last-i)),out+i*size_t(f.max_no));
    first=last;
  }
}
constexpr OrbitalsKernels kernels{lanes,points};
}
const OrbitalsKernels& OPENMX_HOST_SIMD_FACTORY(orbitals)() noexcept {return kernels;}
}

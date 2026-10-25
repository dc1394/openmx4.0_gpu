#include "host_energy_kernels.hpp"
#include "host_simd_abi.hpp"
// The generic lane-wise log/sqrt/floor fallbacks (after the ISA header).
#include <Kokkos_SIMD_Common_Math.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <span>

// Energy family. Lanes span independent outputs only: grid points of the
// neutral-atom quadrature (each point's value is stored, the caller sums in
// the legacy order) and matrix columns of a band density-matrix block (the
// band sum of every element stays sequential). Every expression keeps the
// legacy association; the translation unit is built without contraction.
namespace openmx::core::host {
namespace {
using Mask=typename Vector::mask_type;
using IndexMask=typename IndexVector::mask_type;
static_assert(lanes<=8,"band blocks pad their transposed rows to a multiple of 8 columns");

// A masked gather takes the mask type of its index vector.
Vector gather(const double* p,size_t n,Mask m,IndexVector index) {
  return simd::unchecked_gather_from<Vector>(std::span<const double>(p,n),IndexMask(m),index);
}
Vector select(Mask m,Vector a,Vector b) { return simd::condition(m,a,b); }
// Integral doubles to lane indices (exact: the values are floor results).
#if defined(OPENMX_HOST_SIMD_AVX512)
IndexVector to_index(Vector integral) { return IndexVector(integral); }
#else
IndexVector to_index(Vector integral) {
  alignas(64) double values[lanes];
  alignas(64) std::int32_t indices[lanes];
  simd::simd_unchecked_store(integral,values);
  for(int k=0;k<lanes;++k)indices[k]=std::int32_t(values[k]);
  return simd::simd_unchecked_load<IndexVector>(indices);
}
#endif
// The scalar libm log of every lane (the same function the legacy loop calls).
Vector lane_log(Vector v) {
  alignas(64) double values[lanes];
  simd::simd_unchecked_store(v,values);
  for(int k=0;k<lanes;++k)values[k]=std::log(values[k]);
  return simd::simd_unchecked_load<Vector>(values);
}

// The cubic extrapolation of VH_AtomF/Dr_VH_AtomF inside the first mesh
// point, formed per call exactly as the legacy routines form it.
struct Extrapolation { double a,b; };
Extrapolation extrapolation(const double* rv,const double* yv) {
  const int m=4;
  const double rm=rv[m];
  const double h1=rv[m-1]-rv[m-2];
  const double h2=rv[m]-rv[m-1];
  const double h3=rv[m+1]-rv[m];
  const double f1=yv[m-2],f2=yv[m-1],f3=yv[m],f4=yv[m+1];
  const double g1=((f3-f2)*h1/h2+(f2-f1)*h2/h1)/(h1+h2);
  const double g2=((f4-f3)*h2/h3+(f3-f2)*h3/h2)/(h2+h3);
  const double x1=rm-rv[m-1];
  const double x2=rm-rv[m];
  const double y1=x1/h2;
  const double y2=x2/h2;
  const double y12=y1*y1;
  const double y22=y2*y2;
  const double f=y22*(3.0*f2+h2*g1+(2.0*f2+h2*g1)*y2)
      +y12*(3.0*f3-h2*g2-(2.0*f3-h2*g2)*y1);
  const double df=2.0*y2/h2*(3.0*f2+h2*g1+(2.0*f2+h2*g1)*y2)
      +y22*(2.0*f2+h2*g1)/h2
      +2.0*y1/h2*(3.0*f3-h2*g2-(2.0*f3-h2*g2)*y1)
      -y12*(2.0*f3-h2*g2)/h2;
  const double a=0.5*df/rm;
  const double b=f-a*rm*rm;
  return {a,b};
}
struct Potential {
  const double *xv,*rv,*yv;
  int mesh;
  double charge,xmin,xmax,rv0;
  double count,width,tmp;   // (double)N-1.0, xmax-xmin, Dr_VH_AtomF's tmp
  Extrapolation e;
};
Potential potential(const OpenMXEnergyNeutralPoints& s) {
  const double xmin=s.log_radii[0],xmax=s.log_radii[s.mesh-1];
  return {s.log_radii,s.radii,s.hartree,s.mesh,s.charge,xmin,xmax,s.radii[0],
      double(s.mesh)-1.0,xmax-xmin,(double(s.mesh)-1.0)/(xmax-xmin),extrapolation(s.radii,s.hartree)};
}
struct Table { Vector y0,y1,y2,y3,dt; };
// The four table values and the fractional position of VH_AtomF's third
// branch for the lanes in `table` (the others gather nothing).
Table table_lookup(const Potential& p,Vector t,Mask table) {
  const Vector fl=Kokkos::floor(t);
  const IndexVector i=to_index(select(table,fl,Vector(0.0)));
  const size_t n=size_t(p.mesh)+2;
  return {gather(p.yv,n,table,i),gather(p.yv,n,table,i+IndexVector(1)),
      gather(p.yv,n,table,i+IndexVector(2)),gather(p.yv,n,table,i+IndexVector(3)),t-fl};
}
// VH_AtomF(charge, mesh, xx, r, ...) at lanes (xx, r), every lane taking its own branch.
Vector hartree(const Potential& p,Vector xx,Vector r,Mask active) {
  const Mask beyond=Vector(p.xmax)<=xx;
  const Mask inner=(!beyond)&&(r<Vector(p.rv0));
  const Mask table=active&&(!beyond)&&(!inner);
  const Vector x=select(xx>Vector(p.xmin),xx,Vector(p.xmin));
  const Vector t=(Vector(p.count)*(x-Vector(p.xmin)))/Vector(p.width);
  const Table v=table_lookup(p,t,table);
  const Vector q=(v.y3-v.y0)-Vector(3.0)*(v.y2-v.y1);
  const Vector w=(((q*v.dt)-v.y3)+Vector(4.0)*v.y2)-Vector(5.0)*v.y1+Vector(2.0)*v.y0;
  const Vector u=(w*v.dt)+(v.y2-v.y0);
  const Vector interpolated=(Vector(0.5)*u)*v.dt+v.y1;
  const Vector far=Vector(p.charge)/r;
  const Vector near=(Vector(p.e.a)*r)*r+Vector(p.e.b);
  return select(beyond,far,select(inner,near,interpolated));
}
// Dr_VH_AtomF(charge, mesh, xx, r, ...) likewise (its own position formula).
Vector hartree_derivative(const Potential& p,Vector xx,Vector r,Mask active) {
  const Mask beyond=Vector(p.xmax)<=xx;
  const Mask inner=(!beyond)&&(r<Vector(p.rv0));
  const Mask table=active&&(!beyond)&&(!inner);
  const Vector x=select(xx>Vector(p.xmin),xx,Vector(p.xmin));
  const Vector t=(x-Vector(p.xmin))*Vector(p.tmp);
  const Table v=table_lookup(p,t,table);
  const Vector q=Vector(3.0)*((v.y3-v.y0)-Vector(3.0)*(v.y2-v.y1));
  const Vector w=Vector(2.0)*((((-v.y3)+Vector(4.0)*v.y2)-Vector(5.0)*v.y1)+Vector(2.0)*v.y0);
  const Vector u=((q*v.dt)+w)*v.dt+(v.y2-v.y0);
  const Vector interpolated=((Vector(0.5)*u)*Vector(p.tmp))/r;
  const Vector far=(Vector(-p.charge)/r)/r;
  const Vector near=Vector(2.0*p.e.a)*r;
  return select(beyond,far,select(inner,near,interpolated));
}

void neutral_points(const OpenMXEnergyNeutralPoints& s) {
  const Potential p=potential(s);
  const Vector separation(s.separation),half(0.5),floor_r(1.0e-10),zero(0.0);
  alignas(64) double values[lanes],derivatives[lanes];
  const Mask every(true);
  for(int first=0;first<s.count;first+=lanes) {
    const int n=std::min(lanes,s.count-first);
    const bool full=n==lanes;
    const Mask active=full?every:Mask([&](auto k){return int(k)<n;});
    const Vector x=full?load(s.x+first):simd::simd_partial_load<Vector>(s.x+first,active);
    const Vector y=full?load(s.y+first):simd::simd_partial_load<Vector>(s.y+first,active);
    const Vector z=full?load(s.z+first):simd::simd_partial_load<Vector>(s.z+first,active);
    const Vector rho=full?load(s.density+first):simd::simd_partial_load<Vector>(s.density+first,active);
    const Vector wt=full?load(s.weight+first):simd::simd_partial_load<Vector>(s.weight+first,active);
    const Vector z2=z-separation;
    const Vector r2=(x*x+y*y)+z2*z2;
    const Vector length=Kokkos::sqrt(r2);
    const Vector xx=half*lane_log(r2);
    const Vector r=s.guard?select(length<floor_r,floor_r,length):length;
    simd::simd_unchecked_store((wt*hartree(p,xx,r,active))*rho,values);
    for(int k=0;k<n;++k)s.values[first+k]=values[k];
    if(s.derivative) {
      const Mask far=active&&(Vector(1.0e-14)<r);
      const Vector d=(((wt*rho)*hartree_derivative(p,xx,r,far))*z2)/r;
      simd::simd_unchecked_store(select(far,d,zero),derivatives);
      for(int k=0;k<n;++k)s.derivatives[first+k]=derivatives[k];
    }
  }
}

// Rows [i, i+Rows) of one block for the columns [j0, j0+count), count <=
// Chunks*lanes; the transposed rows are padded, so whole vectors are read, and
// every loaded neighbour vector serves all Rows rows. Each element's band sum
// stays sequential.
template<int Rows,int Chunks> void band_rows(const OpenMXEnergyBandBlock& b,int i,int j0,int count) {
  alignas(64) double s1[Chunks*lanes],s2[Chunks*lanes],s3[Chunks*lanes],s4[Chunks*lanes];
  Vector d1[Rows][Chunks],d2[Rows][Chunks],d3[Rows][Chunks],d4[Rows][Chunks];
  for(int r=0;r<Rows;++r)for(int c=0;c<Chunks;++c)d1[r][c]=d2[r][c]=d3[r][c]=d4[r][c]=Vector(0.0);
  const double* r0[Rows];
  const double* i0[Rows];
  for(int r=0;r<Rows;++r){r0[r]=b.re0[i+r];i0[r]=b.im0[i+r];}
  for(int k=0;k<b.bands;++k) {
    const Vector e(b.eigen[k]);
    const double* x=b.re1T+size_t(k)*b.stride+j0;
    const double* y=b.im1T+size_t(k)*b.stride+j0;
    Vector re1[Chunks],im1[Chunks];
    for(int c=0;c<Chunks;++c){re1[c]=load(x+c*lanes);im1[c]=load(y+c*lanes);}
    for(int r=0;r<Rows;++r) {
      const Vector a(r0[r][k]),g(i0[r][k]);
      for(int c=0;c<Chunks;++c) {
        const Vector re=a*re1[c]+g*im1[c];
        const Vector im=a*im1[c]-g*re1[c];
        d1[r][c]+=re;d2[r][c]+=im;d3[r][c]+=re*e;d4[r][c]+=im*e;
      }
    }
  }
  for(int r=0;r<Rows;++r) {
    for(int c=0;c<Chunks;++c){store(d1[r][c],s1+c*lanes);store(d2[r][c],s2+c*lanes);store(d3[r][c],s3+c*lanes);store(d4[r][c],s4+c*lanes);}
    double* cdm=b.cdm+size_t(i+r)*b.columns+j0;
    double* edm=b.edm+size_t(i+r)*b.columns+j0;
    for(int j=0;j<count;++j){cdm[j]+=b.co*s1[j]-b.si*s2[j];edm[j]+=b.co*s3[j]-b.si*s4[j];}
  }
}
void band_block(const OpenMXEnergyBandBlock& b) {
  for(int j0=0;j0<b.columns;j0+=2*lanes) {
    const int count=std::min(2*lanes,b.columns-j0);
    int i=0;
    for(;i+2<=b.rows;i+=2){if(count>lanes)band_rows<2,2>(b,i,j0,count);else band_rows<2,1>(b,i,j0,count);}
    if(i<b.rows){if(count>lanes)band_rows<1,2>(b,i,j0,count);else band_rows<1,1>(b,i,j0,count);}
  }
}
constexpr EnergyKernels kernels{lanes,neutral_points,band_block};
}
const EnergyKernels& OPENMX_HOST_SIMD_FACTORY(energy)() noexcept {return kernels;}
}

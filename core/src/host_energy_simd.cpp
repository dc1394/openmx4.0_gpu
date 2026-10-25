#include "host_energy_kernels.hpp"
#include "host_simd_abi.hpp"
// The generic lane-wise sqrt/floor fallbacks (after the ISA header).
#include <Kokkos_SIMD_Common_Math.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <span>

// Energy family. Lanes span independent outputs only: grid points of the
// neutral-atom quadrature and Lebedev points of the atomic exchange-
// correlation shell (each point's value is stored or added in point order),
// and matrix columns of a band density-matrix block (the band sum of every
// element stays sequential). Every expression keeps the legacy association;
// the translation unit is built without contraction. log/exp/pow have no
// exact vector form: they are the scalar libm calls, lane by lane, made only
// for the lanes whose branch needs them.
namespace openmx::core::host {
namespace {
using Mask=typename Vector::mask_type;
using IndexMask=typename IndexVector::mask_type;
static_assert(lanes<=8,"transposed band rows and XC scratch are padded to 8 columns");

Vector gather(const double* p,size_t n,Mask m,IndexVector index) {
  return simd::unchecked_gather_from<Vector>(std::span<const double>(p,n),IndexMask(m),index);
}
Vector select(Mask m,Vector a,Vector b) { return simd::condition(m,a,b); }
template<class M> bool any(const M& m) { return simd::any_of(m); }
template<class M> bool all(const M& m) { return simd::all_of(m); }
Mask active_lanes(int n) { return n>=lanes?Mask(true):Mask([&](auto k){return int(k)<n;}); }
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
// The scalar libm function of every lane in `m` (the other lanes are zero).
template<class F> Vector lane_call(Vector v,Mask m,F&& f) {
  alignas(64) double values[lanes];
  simd::simd_unchecked_store(v,values);
  for(int k=0;k<lanes;++k)values[k]=m[k]?f(values[k]):0.0;
  return simd::simd_unchecked_load<Vector>(values);
}
Vector lane_log(Vector v,Mask m) { return lane_call(v,m,[](double x){return std::log(x);}); }
Vector lane_exp(Vector v,Mask m) { return lane_call(v,m,[](double x){return std::exp(x);}); }

// The cubic extrapolation inside the first mesh point shared by VH_AtomF,
// Dr_VH_AtomF, KumoF and Dr_KumoF, formed per table exactly as they form it.
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
// A logarithmic radial table (xv = log radii, rv radii, yv values with two
// trailing entries) and the per-table constants of the four routines.
struct Table {
  const double *xv,*rv,*yv;
  int mesh;
  double xmin,xmax,rv0;
  double count,width,tmp;   // (double)N-1.0, xmax-xmin, Dr_*'s tmp
  Extrapolation e;
};
Table table(int mesh,const double* xv,const double* rv,const double* yv) {
  const double xmin=xv[0],xmax=xv[mesh-1];
  return {xv,rv,yv,mesh,xmin,xmax,rv[0],double(mesh)-1.0,xmax-xmin,(double(mesh)-1.0)/(xmax-xmin),extrapolation(rv,yv)};
}
struct Samples { Vector y0,y1,y2,y3; };
Samples samples(const Table& t,IndexVector i,Mask m) {
  // The legacy routines read yv[i+3] with i up to N-1 (KumoF's clamp); the
  // gathers touch exactly those entries, nothing else.
  const size_t n=size_t(t.mesh)+3;
  return {gather(t.yv,n,m,i),gather(t.yv,n,m,i+IndexVector(1)),gather(t.yv,n,m,i+IndexVector(2)),gather(t.yv,n,m,i+IndexVector(3))};
}
// The 4-point formula of VH_AtomF/KumoF at the fractional position dt.
Vector value_formula(const Samples& v,Vector dt) {
  const Vector q=(v.y3-v.y0)-Vector(3.0)*(v.y2-v.y1);
  const Vector w=(((q*dt)-v.y3)+Vector(4.0)*v.y2)-Vector(5.0)*v.y1+Vector(2.0)*v.y0;
  const Vector u=(w*dt)+(v.y2-v.y0);
  return (Vector(0.5)*u)*dt+v.y1;
}
// The 4-point formula of Dr_VH_AtomF/Dr_KumoF up to "0.5*(...)" (then *tmp/r).
Vector derivative_formula(const Samples& v,Vector dt) {
  const Vector q=Vector(3.0)*((v.y3-v.y0)-Vector(3.0)*(v.y2-v.y1));
  const Vector w=Vector(2.0)*((((-v.y3)+Vector(4.0)*v.y2)-Vector(5.0)*v.y1)+Vector(2.0)*v.y0);
  const Vector u=((q*dt)+w)*dt+(v.y2-v.y0);
  return Vector(0.5)*u;
}
// The table branches of a value routine and its derivative routine at the
// clamped abscissa x, for the lanes in `m`. The two routines locate the
// interval with different arithmetic (t = (N-1)*(x-xmin)/(xmax-xmin) against
// t = (x-xmin)*tmp); when every lane lands in the same interval the four
// gathered samples serve both. derivative = "0.5*(...)*tmp" (the caller
// divides by its radius).
struct Interpolated { Vector value,derivative; };
Interpolated interpolate(const Table& t,Vector x,Mask m,bool derivative) {
  const Vector shift=x-Vector(t.xmin);
  const Vector tv=(Vector(t.count)*shift)/Vector(t.width);
  const Vector fv=Kokkos::floor(tv);
  const IndexVector iv=to_index(select(m,fv,Vector(0.0)));
  const Samples sv=samples(t,iv,m);
  Interpolated out{value_formula(sv,tv-fv),Vector(0.0)};
  if(derivative) {
    const Vector td=shift*Vector(t.tmp);
    const Vector fd=Kokkos::floor(td);
    const IndexVector id=to_index(select(m,fd,Vector(0.0)));
    const bool shared=all((id==iv)||(!IndexMask(m)));
    out.derivative=derivative_formula(shared?sv:samples(t,id,m),td-fd)*Vector(t.tmp);
  }
  return out;
}

/* ---- the neutral-atom integrand: VH_AtomF and Dr_VH_AtomF ---- */
void neutral_points(const OpenMXEnergyNeutralPoints& s) {
  const Table t=table(s.mesh,s.log_radii,s.radii,s.hartree);
  const Vector separation(s.separation),half(0.5),floor_r(1.0e-10),zero(0.0);
  const Vector charge(s.charge),negative_charge(-s.charge),rv0(t.rv0),xmin(t.xmin),xmax(t.xmax);
  const Vector a(t.e.a),b(t.e.b),twice_a(2.0*t.e.a);
  // "xmax <= xx" with xx = 0.5*log(r*r) is undecided only within a thin band
  // around r = exp(xmax): elsewhere it follows from r alone (log is monotonic
  // and libm's log is within an ulp), and lanes inside the first mesh point
  // never use xx either, so those lanes skip the scalar log.
  const double edge=std::exp(t.xmax),outer=edge*(1.0+1.0e-9),inner=edge*(1.0-1.0e-9);
  const Vector outer2(outer*outer),inner2(inner*inner);
  const bool derivative=s.derivative!=0;
  alignas(64) double values[lanes],derivatives[lanes];
  for(int first=0;first<s.count;first+=lanes) {
    const int n=std::min(lanes,s.count-first);
    const bool full=n==lanes;
    const Mask active=active_lanes(n);
    const Vector x=full?load(s.x+first):simd::simd_partial_load<Vector>(s.x+first,active);
    const Vector y=full?load(s.y+first):simd::simd_partial_load<Vector>(s.y+first,active);
    const Vector z=full?load(s.z+first):simd::simd_partial_load<Vector>(s.z+first,active);
    const Vector rho=full?load(s.density+first):simd::simd_partial_load<Vector>(s.density+first,active);
    const Vector wt=full?load(s.weight+first):simd::simd_partial_load<Vector>(s.weight+first,active);
    const Vector z2=z-separation;
    const Vector r2=(x*x+y*y)+z2*z2;
    const Vector length=Kokkos::sqrt(r2);
    const Vector r=s.guard?select(length<floor_r,floor_r,length):length;
    const Mask surely_beyond=r2>outer2;
    const Mask undecided=(!(r2>outer2))&&(!(r2<inner2));
    const Mask below=r<rv0;
    const Mask need_log=active&&(undecided||((!surely_beyond)&&(!below)));
    const Vector xx=half*lane_log(r2,need_log);
    const Mask beyond=surely_beyond||(undecided&&(xmax<=xx));
    const Mask near=(!beyond)&&below;
    const Mask tab=active&&(!beyond)&&(!near);
    Vector va0=zero,dr=zero;
    if(any(active&&beyond)) {
      va0=select(beyond,charge/r,va0);
      if(derivative)dr=select(beyond,(negative_charge/r)/r,dr);
    }
    if(any(active&&near)) {
      va0=select(near,(a*r)*r+b,va0);
      if(derivative)dr=select(near,twice_a*r,dr);
    }
    if(any(tab)) {
      const Interpolated p=interpolate(t,select(xx>xmin,xx,xmin),tab,derivative);
      va0=select(tab,p.value,va0);
      if(derivative)dr=select(tab,p.derivative/r,dr);
    }
    simd::simd_unchecked_store((wt*va0)*rho,values);
    for(int k=0;k<n;++k)s.values[first+k]=values[k];
    if(derivative) {
      const Mask far=active&&(Vector(1.0e-14)<r);
      const Vector d=(((wt*rho)*dr)*z2)/r;
      simd::simd_unchecked_store(select(far,d,zero),derivatives);
      for(int k=0;k<n;++k)s.derivatives[first+k]=derivatives[k];
    }
  }
}

/* ---- the atomic exchange-correlation shell: KumoF, Dr_KumoF, XC_Ceperly_Alder ---- */
// XC_Ceperly_Alder(den, 0) and (den, 3) for the lanes in `active`; both
// switches share rs, Ex, Ec and their derivatives as the routine computes them.
struct Xc { Vector energy,derivative; };
Xc ceperley_alder(Vector den,Mask active) {
  const Vector zero(0.0);
  const Mask live=active&&(!(den<=Vector(1.0e-15)));
  const double coe=0.6203504908994;
  const Vector rs=Vector(coe)*lane_call(den,live,[](double d){return std::pow(d,-0.3333333333333333333);});
  const Vector tmp0=Vector(0.458165293632163)/rs;
  const Vector Ex=-tmp0;
  const Vector dEx=tmp0/rs;
  const Mask high=live&&(Vector(1.0)<=rs);
  const Vector root=Kokkos::sqrt(rs);
  const Vector dum=(Vector(1.0)+Vector(1.0529)*root)+Vector(0.3334)*rs;
  const Vector tmp1=Vector(0.1423)/dum;
  const Vector Ec_high=-tmp1;
  const Vector dEc_high=(tmp1/dum)*((Vector(0.52645)/root)+Vector(0.3334));
  const Mask low=live&&(!high);
  const Vector lg=lane_log(rs,low);
  const Vector Ec_low=(Vector(-0.0480)+Vector(0.0311)*lg)+rs*((Vector(0.0020)*lg)-Vector(0.0116));
  const Vector dEc_low=((Vector(0.0311)/rs)+Vector(0.0020)*lg)-Vector(0.0096);
  const Vector Ec=select(high,Ec_high,Ec_low),dEc=select(high,dEc_high,dEc_low);
  const double scale=-0.3333333333333333333/(coe*coe*coe);
  return {select(live,Ex+Ec,zero),select(live,((((Vector(scale)*rs)*rs)*rs)*rs)*(dEx+dEc),zero)};
}
// Per-neighbour table constants cached in the scratch for one shell.
constexpr int table_doubles=8;
void cache_table(double* c,const Table& t) {
  c[0]=t.xmin;c[1]=t.xmax;c[2]=t.rv0;c[3]=t.count;c[4]=t.width;c[5]=t.tmp;c[6]=t.e.a;c[7]=t.e.b;
}
Table cached_table(const double* c,int mesh,const double* xv,const double* rv,const double* yv) {
  return {xv,rv,yv,mesh,c[0],c[1],c[2],c[3],c[4],c[5],{c[6],c[7]}};
}
void xc_shell(const OpenMXEnergyXcShell& s) {
  const int nb=s.neighbors;
  double* consts=s.scratch;                            // [nb][table_doubles]
  double* g=s.scratch+size_t(nb)*table_doubles;        // [3][nb][8] lanes
  for(int h=0;h<nb;++h)cache_table(consts+size_t(h)*table_doubles,table(s.mesh[h],s.log_radii[h],s.radii[h],s.density[h]));
  const Vector radius(s.radius),cx(s.center[0]),cy(s.center[1]),cz(s.center[2]),vxc(s.vxc),half(0.5),zero(0.0);
  double carry=*s.den0;
  alignas(64) double lx[lanes],ly[lanes],lz[lanes],lw[lanes],terms[lanes],den0v[lanes],products[lanes];
  for(int first=0;first<s.points;first+=lanes) {
    const int n=std::min(lanes,s.points-first);
    const Mask active=active_lanes(n);
    for(int k=0;k<lanes;++k) {
      const double* row=s.lebedev+size_t(first+std::min(k,n-1))*s.lebedev_stride;
      lx[k]=row[0];ly[k]=row[1];lz[k]=row[2];lw[k]=row[3];
    }
    const Vector x0=radius*load(lx)+cx,y0=radius*load(ly)+cy,z0=radius*load(lz)+cz;
    Vector den=zero,den0=zero;
    Mask pass0(false);
    for(int h=0;h<nb;++h) {
      const Vector dx=Vector(s.nx[h])-x0,dy=Vector(s.ny[h])-y0,dz=Vector(s.nz[h])-z0;
      const Vector r2=(dx*dx+dy*dy)+dz*dz;
      const Mask inside=active&&(r2<Vector(s.cutoff2[h]));
      Vector gx=zero,gy=zero,gz=zero;
      if(any(inside)) {
        const Table t=cached_table(consts+size_t(h)*table_doubles,s.mesh[h],s.log_radii[h],s.radii[h],s.density[h]);
        const Vector x=half*lane_log(r2,inside);
        const Mask below=inside&&(x<Vector(t.xmin));   // x<xv[0]
        const Mask tab=inside&&(!below);
        const Vector r1=Kokkos::sqrt(r2);              // sqrt(dx*dx + dy*dy + dz*dz)
        Vector value=zero,slope=zero;
        if(any(tab)) {
          const Vector clamped_high=select(x<Vector(t.xmax),x,Vector(t.xmax));          // MIN(x,xmax)
          const Vector clamped=select(clamped_high>Vector(t.xmin),clamped_high,Vector(t.xmin));  // MAX(x,xmin)
          const Interpolated p=interpolate(t,clamped,tab,h!=0);
          value=select(tab,p.value,value);
          if(h)slope=select(tab,p.derivative/r1,slope);
        }
        if(any(below)) {
          const Vector rb=lane_exp(x,below);
          value=select(below,(Vector(t.e.a)*rb)*rb+Vector(t.e.b),value);
          if(h)slope=select(below,Vector(2.0*t.e.a)*rb,slope);
        }
        den=select(inside,den+value*vxc,den);
        if(h==0){den0=select(inside,den,den0);pass0=inside;}
        else {
          const Vector q=(slope*vxc)/r1;               // gden0/r1
          gx=select(inside,q*dx,zero);gy=select(inside,q*dy,zero);gz=select(inside,q*dz,zero);
        }
      }
      if(h) {
        store(gx,g+(size_t(0)*nb+h)*8);store(gy,g+(size_t(1)*nb+h)*8);store(gz,g+(size_t(2)*nb+h)*8);
      }
    }
    // The centre density of a point whose centre test failed is the legacy
    // variable's previous value.
    simd::simd_unchecked_store(den0,den0v);
    if(all(pass0||(!active)))carry=den0v[n-1];
    else for(int k=0;k<n;++k){if(!pass0[k])den0v[k]=carry;carry=den0v[k];}
    den0=simd::simd_unchecked_load<Vector>(den0v);
    const Xc xc=ceperley_alder(den,active);
    const Vector w=load(lw);
    const Vector energy=(w*den0)*xc.energy,factor=(w*den0)*xc.derivative;
    simd::simd_unchecked_store(energy,terms);
    for(int k=0;k<n;++k)*s.sumt+=terms[k];
    double* const sums[3]={s.sum_gx,s.sum_gy,s.sum_gz};
    for(int c=0;c<3;++c)for(int h=1;h<nb;++h) {
      simd::simd_unchecked_store(factor*load(g+(size_t(c)*nb+h)*8),products);
      for(int k=0;k<n;++k)sums[c][h]+=products[k];
    }
  }
  *s.den0=carry;
}

/* ---- band density-matrix block ---- */
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
constexpr EnergyKernels kernels{lanes,neutral_points,xc_shell,band_block};
}
const EnergyKernels& OPENMX_HOST_SIMD_FACTORY(energy)() noexcept {return kernels;}
}

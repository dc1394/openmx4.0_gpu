#include "openmx_core_orbitals_simd.h"
#include "openmx_core_host_simd.h"
#include "host_orbitals_reference.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

// The orbitals family must reproduce SOG_point_eval bit for bit on every
// usable ISA: random points inside and outside the PAO mesh, below its first
// point, on the axes and at the atom, mixed species runs, every lane tail.
namespace {
void require(bool ok,const char* what) { if(!ok)throw std::runtime_error(what); }
struct Species {
  int mesh=0,maxl=0,nb[4]={0,0,0,0};
  std::vector<double> rv;
  std::vector<std::vector<double>> rwf;  // (L0, Mul0) order
  int orbitals() const { int n=0;for(int l=0;l<=maxl;++l)n+=nb[l]*(2*l+1);return n; }
};
Species make_species(std::mt19937& rng,int mesh,std::initializer_list<int> counts,double rcut,bool logarithmic) {
  Species s;s.mesh=mesh;s.maxl=int(counts.size())-1;
  int l=0;for(int c:counts)s.nb[l++]=c;
  const double r0=1.0e-4;
  for(int m=0;m<mesh;++m)
    s.rv.push_back(logarithmic?std::exp(std::log(r0)+(std::log(rcut)-std::log(r0))*m/double(mesh-1)):r0+(rcut-r0)*m/double(mesh-1));
  std::uniform_real_distribution<double> sign(-1.0,1.0);
  for(int L0=0;L0<=s.maxl;++L0)for(int Mul0=0;Mul0<s.nb[L0];++Mul0) {
    std::vector<double> f(mesh);const double scale=sign(rng)<0?-1.0:1.0;
    for(int m=0;m<mesh;++m){const double r=s.rv[m];f[m]=scale*std::pow(r,L0)*std::exp(-(0.7+0.3*Mul0)*r)*(1.0+0.25*std::sin((2.0+L0)*r+Mul0));}
    s.rwf.push_back(f);
  }
  return s;
}
struct Setup {
  std::vector<Species> species;
  std::vector<double> rv,rwf,translations;
  std::vector<size_t> rv_off,rwf_base;
  std::vector<int> mesh,maxl,nb;
  int cells=27,l0max=3;
  OpenMXOrbitalsFrame frame{};
  int ng1=0;
  double tv[3][3];
  void finish() {
    for(const auto& s:species) {
      rv_off.push_back(rv.size());rwf_base.push_back(rwf.size());mesh.push_back(s.mesh);maxl.push_back(s.maxl);
      for(int l=0;l<=l0max;++l)nb.push_back(l<=s.maxl?s.nb[l]:0);
      rv.insert(rv.end(),s.rv.begin(),s.rv.end());
      for(const auto& f:s.rwf)rwf.insert(rwf.end(),f.begin(),f.end());
    }
    for(int c=0;c<cells;++c){const int l1=c/9-1,l2=c/3%3-1,l3=c%3-1;
      for(int a=0;a<3;++a)translations.push_back(l1*tv[0][a]+l2*tv[1][a]+l3*tv[2][a]);}
  }
  OpenMXOrbitalsTables tables() const {
    return {rv.data(),rwf.data(),translations.data(),rv_off.data(),rwf_base.data(),mesh.data(),maxl.data(),nb.data(),
        int(species.size()),l0max,8,cells};
  }
  // The absolute coordinate of a grid point in a cell, in the kernel's arithmetic.
  void coordinate(int grid,int cell,double c[3]) const {
    const int n1=grid/frame.ng23,n2=(grid-n1*frame.ng23)/frame.ng3,n3=grid-n1*frame.ng23-n2*frame.ng3;
    for(int a=0;a<3;++a)c[a]=((double)n1*frame.frame[a]+(double)n2*frame.frame[3+a]+(double)n3*frame.frame[6+a]+frame.origin[a])+translations[3*size_t(cell)+a];
  }
  int grids() const { return ng1*frame.ng23; }
};
Setup make_setup(std::mt19937& rng) {
  Setup u;
  u.species.push_back(make_species(rng,500,{2,2,1},7.0,true));     // s2p2d1 on a logarithmic mesh
  u.species.push_back(make_species(rng,97,{1,1,1,1},5.5,false));   // s1p1d1f1 on a uniform mesh
  u.species.push_back(make_species(rng,300,{3,3,2,1},6.0,true));   // wider multiplicities
  u.species.push_back(make_species(rng,6,{1},4.0,true));           // the smallest mesh, L0 = 0
  u.species.push_back(make_species(rng,200,{1,2,8,8},6.5,true));   // the largest multiplicity
  const double g[3][3]={{0.21,0.01,0.0},{0.02,0.19,0.01},{0.0,0.03,0.2}};
  u.ng1=48;u.frame.ng3=48;u.frame.ng23=48*48;
  for(int i=0;i<3;++i)for(int j=0;j<3;++j){u.frame.frame[3*i+j]=g[i][j];u.tv[i][j]=48*g[i][j];}
  u.frame.origin[0]=-5.0;u.frame.origin[1]=-4.5;u.frame.origin[2]=-5.5;
  u.frame.max_no=0;for(const auto& s:u.species)u.frame.max_no=std::max(u.frame.max_no,s.orbitals());
  u.finish();
  return u;
}
// A point of species wan whose displacement from the atom has length about
// `distance` in a random direction (or exactly the given direction).
OpenMXOrbitalsPoint point(std::mt19937& rng,const Setup& u,int wan,double distance,const double* direction=nullptr,bool shifted=false) {
  OpenMXOrbitalsPoint p{};
  p.wan=wan;p.no=u.species[wan].orbitals();
  p.grid=std::uniform_int_distribution<int>(0,u.grids()-1)(rng);
  p.cell=std::uniform_int_distribution<int>(0,u.cells-1)(rng);
  double c[3];u.coordinate(p.grid,p.cell,c);
  double d[3];
  if(direction){for(int a=0;a<3;++a)d[a]=direction[a];}
  else {std::normal_distribution<double> n;double norm=0;for(int a=0;a<3;++a){d[a]=n(rng);norm+=d[a]*d[a];}
    norm=std::sqrt(norm);for(int a=0;a<3;++a)d[a]/=norm;}
  const int shift=shifted?std::uniform_int_distribution<int>(0,u.cells-1)(rng):13;
  for(int a=0;a<3;++a){p.shift[a]=u.translations[3*size_t(shift)+a];p.xyz[a]=c[a]-distance*d[a]-p.shift[a];}
  return p;
}
std::vector<float> expected(const Setup& u,const std::vector<OpenMXOrbitalsPoint>& q) {
  std::vector<float> e(q.size()*size_t(u.frame.max_no),0.0f);
  const auto& f=u.frame;
  for(size_t i=0;i<q.size();++i) {
    const auto& p=q[i];
    openmx::test::sog_point_eval(p.wan,p.no,p.xyz[0],p.xyz[1],p.xyz[2],p.shift[0],p.shift[1],p.shift[2],p.grid,p.cell,
        u.translations.data(),u.rv.data(),u.rwf.data(),u.rv_off.data(),u.rwf_base.data(),u.mesh.data(),u.maxl.data(),u.nb.data(),
        f.ng23,f.ng3,f.frame[0],f.frame[1],f.frame[2],f.frame[3],f.frame[4],f.frame[5],f.frame[6],f.frame[7],f.frame[8],
        f.origin[0],f.origin[1],f.origin[2],e.data()+i*size_t(f.max_no));
  }
  return e;
}
std::vector<int> modes() {
  std::vector<int> m;
  for(int mode=OPENMX_GRID_SCALAR;mode<=OPENMX_GRID_NEON;++mode)if(openmx_core_host_simd_usable(mode))m.push_back(mode);
  require(!m.empty() && m[0]==OPENMX_GRID_SCALAR,"scalar host kernels must always be usable");
  return m;
}
void compare(const Setup& u,const std::vector<OpenMXOrbitalsPoint>& q,const char* what) {
  const auto e=expected(u,q);
  const auto tables=u.tables();
  for(int mode:modes()) {
    std::vector<float> out(e.size()+2,-7.0f);
    require(openmx_core_orbitals_points(&tables,&u.frame,q.data(),q.size(),out.data()+1,mode),"the kernel refused a valid request");
    require(out.front()==-7.0f && out.back()==-7.0f,"the kernel wrote outside its output");
    for(size_t i=0;i<e.size();++i)
      if(std::memcmp(&out[i+1],&e[i],sizeof(float))!=0) {
        std::cerr<<what<<": mode "<<openmx_core_host_simd_name(mode)<<" point "<<i/size_t(u.frame.max_no)<<" orbital "<<i%size_t(u.frame.max_no)
            <<" got "<<out[i+1]<<" expected "<<e[i]<<'\n';
        throw std::runtime_error("host orbital kernel differs from SOG_point_eval");
      }
  }
}
void equivalence() {
  std::mt19937 rng(2025u);
  const Setup u=make_setup(rng);
  std::uniform_real_distribution<double> radius(0.0,1.0);
  for(int count:{1,2,3,7,8,9,15,16,17,31,33,100,1000,4097})for(int wan=0;wan<int(u.species.size());++wan) {
    std::vector<OpenMXOrbitalsPoint> q;
    const double rcut=u.species[wan].rv.back();
    for(int i=0;i<count;++i)q.push_back(point(rng,u,wan,1.3*rcut*radius(rng),nullptr,i%3==0));
    compare(u,q,"equivalence");
  }
}
void runs() {
  std::mt19937 rng(77u);
  const Setup u=make_setup(rng);
  std::uniform_real_distribution<double> radius(0.0,1.0);
  std::uniform_int_distribution<int> length(1,40),which(0,int(u.species.size())-1);
  std::vector<OpenMXOrbitalsPoint> q;
  while(q.size()<3000) {
    const int wan=which(rng),n=length(rng);
    for(int i=0;i<n;++i)q.push_back(point(rng,u,wan,1.2*u.species[wan].rv.back()*radius(rng),nullptr,i%2==0));
  }
  compare(u,q,"runs");
}
void edges() {
  std::mt19937 rng(5u);
  const Setup u=make_setup(rng);
  std::vector<OpenMXOrbitalsPoint> q;
  const double ex[3]={1,0,0},ey[3]={0,1,0},ez[3]={0,0,1},mz[3]={0,0,-1},my[3]={0,-1,0},mx[3]={-1,0,0};
  for(int wan=0;wan<int(u.species.size());++wan) {
    const auto& s=u.species[wan];
    // At the atom, on every axis in both directions, inside the first mesh
    // point, at mesh points, just inside and beyond the last one, far away.
    q.push_back(point(rng,u,wan,0.0));
    for(const double* d:{ex,ey,ez,mz,my,mx})for(double dist:{0.5*s.rv[0],s.rv[1],s.rv[4],0.5*s.rv.back(),s.rv.back(),1.001*s.rv.back()})
      q.push_back(point(rng,u,wan,dist,d));
    for(int m:{1,2,3,4,5,s.mesh/2,s.mesh-2,s.mesh-1})q.push_back(point(rng,u,wan,s.rv[m]));
    q.push_back(point(rng,u,wan,2.0e-14));q.push_back(point(rng,u,wan,0.9e-14));
    q.push_back(point(rng,u,wan,50.0));
  }
  // Repeated identical points fill whole blocks with equal lanes.
  for(int i=0;i<20;++i)q.push_back(q[3]);
  compare(u,q,"edges");
  // An empty request writes nothing; padding beyond the species is zero.
  const auto tables=u.tables();float sentinel=3.0f;
  for(int mode:modes())require(openmx_core_orbitals_points(&tables,&u.frame,nullptr,0,&sentinel,mode) && sentinel==3.0f,"empty request");
  Setup padded=u;padded.frame.max_no=u.frame.max_no+5;
  std::vector<OpenMXOrbitalsPoint> few(q.begin(),q.begin()+37);
  compare(padded,few,"padding");
}
void invalid() {
  std::mt19937 rng(9u);
  Setup u=make_setup(rng);
  std::vector<OpenMXOrbitalsPoint> q;
  for(int i=0;i<10;++i)q.push_back(point(rng,u,i%2,1.0));
  const auto tables=u.tables();
  std::vector<float> out(q.size()*size_t(u.frame.max_no),5.0f);
  auto refused=[&](const OpenMXOrbitalsTables* t,const OpenMXOrbitalsFrame* f,const OpenMXOrbitalsPoint* p,size_t n,float* o,int mode,const char* what) {
    require(!openmx_core_orbitals_points(t,f,p,n,o,mode),what);
    for(float v:out)require(v==5.0f,"a refused call wrote output");
  };
  refused(&tables,&u.frame,q.data(),q.size(),out.data(),99,"unknown mode accepted");
  refused(&tables,&u.frame,q.data(),q.size(),out.data(),OPENMX_HOST_SIMD_LEGACY,"legacy accepted as a kernel");
  refused(nullptr,&u.frame,q.data(),q.size(),out.data(),OPENMX_GRID_SCALAR,"missing tables accepted");
  refused(&tables,nullptr,q.data(),q.size(),out.data(),OPENMX_GRID_SCALAR,"missing frame accepted");
  refused(&tables,&u.frame,nullptr,q.size(),out.data(),OPENMX_GRID_SCALAR,"missing points accepted");
  refused(&tables,&u.frame,q.data(),q.size(),nullptr,OPENMX_GRID_SCALAR,"missing output accepted");
  auto t=tables;t.rwf=nullptr;refused(&t,&u.frame,q.data(),q.size(),out.data(),OPENMX_GRID_SCALAR,"missing radial table accepted");
  t=tables;t.cells=0;refused(&t,&u.frame,q.data(),q.size(),out.data(),OPENMX_GRID_SCALAR,"no cells accepted");
  auto f=u.frame;f.ng3=0;refused(&tables,&f,q.data(),q.size(),out.data(),OPENMX_GRID_SCALAR,"empty grid accepted");
  f=u.frame;f.max_no=q[0].no-1;refused(&tables,&f,q.data(),q.size(),out.data(),OPENMX_GRID_SCALAR,"short output row accepted");
  auto bad=q;bad[3].no+=1;refused(&tables,&u.frame,bad.data(),bad.size(),out.data(),OPENMX_GRID_SCALAR,"wrong orbital count accepted");
  bad=q;bad[4].wan=int(u.species.size());refused(&tables,&u.frame,bad.data(),bad.size(),out.data(),OPENMX_GRID_SCALAR,"unknown species accepted");
  bad=q;bad[5].cell=u.cells;refused(&tables,&u.frame,bad.data(),bad.size(),out.data(),OPENMX_GRID_SCALAR,"unknown cell accepted");
  bad=q;bad[6].grid=-1;refused(&tables,&u.frame,bad.data(),bad.size(),out.data(),OPENMX_GRID_SCALAR,"negative grid index accepted");
  // Shapes beyond the explicit harmonics or the radial storage.
  std::vector<int> maxl=u.maxl;maxl[0]=4;t=tables;t.maxl=maxl.data();
  refused(&t,&u.frame,q.data(),q.size(),out.data(),OPENMX_GRID_SCALAR,"L0 = 4 accepted");
  std::vector<int> nb=u.nb;nb[1]=9;t=tables;t.nb=nb.data();
  refused(&t,&u.frame,q.data(),q.size(),out.data(),OPENMX_GRID_SCALAR,"nine radial functions accepted");
  std::vector<int> mesh=u.mesh;mesh[1]=5;t=tables;t.mesh=mesh.data();
  refused(&t,&u.frame,q.data(),q.size(),out.data(),OPENMX_GRID_SCALAR,"a five-point mesh accepted");
}
// The legacy OpenMP loops of Set_Orbitals_Grid (angles through acos/asin and
// sin/cos, bisection for the mesh interval) for the difference report.
void legacy_point(const Setup& u,const OpenMXOrbitalsPoint& p,float* out) {
  const auto& s=u.species[p.wan];const double* rv=s.rv.data();const int mesh=s.mesh;
  double c[3];u.coordinate(p.grid,p.cell,c);
  const double x=c[0]-p.xyz[0]-p.shift[0],y=c[1]-p.xyz[1]-p.shift[1],z=c[2]-p.xyz[2]-p.shift[2];
  const double PI=3.141592653589793238462643383279502884;
  auto sgn=[](double a){return a<0.0?-1.0:1.0;};
  const double Min_r=10e-15,dum=x*x+y*y,r=std::sqrt(dum+z*z),r1=std::sqrt(dum);
  double theta,phi,dum1;
  if(Min_r<=r) {
    dum1=r<std::fabs(z)?sgn(z)*1.0:z/r;theta=std::acos(dum1);
    if(Min_r<=r1) {
      dum1=r1<std::fabs(y)?sgn(y)*1.0:y/r1;
      phi=0.0<=x?std::asin(dum1):PI-std::asin(dum1);
    } else phi=0.0;
  } else {theta=0.5*PI;phi=0.0;}
  const double R=r,Q=theta,P=phi;
  double RF[4][8],AF[4][7];int po=0;
  if(rv[mesh-1]<R)po=1;
  else {
    int m;double rm=0.0,h1,h2,h3,x1,x2;
    const bool below=R<rv[0];
    if(below){m=4;rm=rv[m];}
    else {int lo=0,hi=mesh-1;do{m=(lo+hi)/2;if(rv[m]<R)lo=m;else hi=m;}while(hi-lo!=1);m=hi;}
    h1=(m==1)?rv[m-1]-rv[m+1]:rv[m-1]-rv[m-2];h2=rv[m]-rv[m-1];h3=(m==mesh-1)?rv[m-2]-rv[m]:rv[m+1]-rv[m];
    x1=(below?rm:R)-rv[m-1];x2=(below?rm:R)-rv[m];
    const double y1=x1/h2,y2=x2/h2,y12=y1*y1,y22=y2*y2;
    double d=h1+h2;const double dum1_=h1/h2/d,dum2=h2/h1/d;d=h2+h3;const double dum3=h2/h3/d,dum4=h3/h2/d;
    int k=0;
    for(int L0=0;L0<=s.maxl;++L0)for(int Mul0=0;Mul0<s.nb[L0];++Mul0,++k) {
      const double* rw=s.rwf[k].data();
      const double f1=rw[(m==1)?m+1:m-2],f2=rw[m-1],f3=rw[m],f4=rw[(m==mesh-1)?m-2:m+1];
      const double dd=f3-f2,g1=dd*dum1_+(f2-f1)*dum2,g2=(f4-f3)*dum3+dd*dum4;
      const double f=y22*(3.0*f2+h2*g1+(2.0*f2+h2*g1)*y2)+y12*(3.0*f3-h2*g2-(2.0*f3-h2*g2)*y1);
      if(below) {
        const double df=2.0*y2/h2*(3.0*f2+h2*g1+(2.0*f2+h2*g1)*y2)+y22*(2.0*f2+h2*g1)/h2
            +2.0*y1/h2*(3.0*f3-h2*g2-(2.0*f3-h2*g2)*y1)-y12*(2.0*f3-h2*g2)/h2;
        double a,b,cc,dq;
        if(L0==0){a=0.0;b=0.5*df/rm;cc=0.0;dq=f-b*rm*rm;}
        else if(L0==1){a=(rm*df-f)/(2.0*rm*rm*rm);b=0.0;cc=df-3.0*a*rm*rm;dq=0.0;}
        else{b=(3.0*f-rm*df)/(rm*rm);a=(f-b*rm*rm)/(rm*rm*rm);cc=0.0;dq=0.0;}
        RF[L0][Mul0]=a*R*R*R+b*R*R+cc*R+dq;
      } else RF[L0][Mul0]=f;
    }
  }
  int i1=0;
  if(po==0) {
    const double siQ=std::sin(Q),coQ=std::cos(Q),siP=std::sin(P),coP=std::cos(P);
    AF[0][0]=0.282094791773878;
    double d1=0.48860251190292*siQ;AF[1][0]=d1*coP;AF[1][1]=d1*siP;AF[1][2]=0.48860251190292*coQ;
    const double e1=siQ*siQ,e2=1.09254843059208*siQ*coQ;
    AF[2][0]=0.94617469575756*coQ*coQ-0.31539156525252;AF[2][1]=0.54627421529604*e1*(1.0-2.0*siP*siP);
    AF[2][2]=1.09254843059208*e1*siP*coP;AF[2][3]=e2*coP;AF[2][4]=e2*siP;
    AF[3][0]=0.373176332590116*(5.0*coQ*coQ*coQ-3.0*coQ);AF[3][1]=0.457045799464466*coP*siQ*(5.0*coQ*coQ-1.0);
    AF[3][2]=0.457045799464466*siP*siQ*(5.0*coQ*coQ-1.0);AF[3][3]=1.44530572132028*siQ*siQ*coQ*(coP*coP-siP*siP);
    AF[3][4]=2.89061144264055*siQ*siQ*coQ*siP*coP;AF[3][5]=0.590043589926644*siQ*siQ*siQ*(4.0*coP*coP*coP-3.0*coP);
    AF[3][6]=0.590043589926644*siQ*siQ*siQ*(3.0*siP-4.0*siP*siP*siP);
    for(int L0=0;L0<=s.maxl;++L0)for(int Mul0=0;Mul0<s.nb[L0];++Mul0)for(int M0=0;M0<=2*L0;++M0)out[i1++]=float(RF[L0][Mul0]*AF[L0][M0]);
  } else for(;i1<p.no;++i1)out[i1]=0.0f;
}
void legacy_difference() {
  std::mt19937 rng(31u);
  const Setup u=make_setup(rng);
  std::uniform_real_distribution<double> radius(0.0,1.0);
  std::vector<OpenMXOrbitalsPoint> q;
  for(int wan=0;wan<int(u.species.size());++wan)for(int i=0;i<4000;++i)q.push_back(point(rng,u,wan,1.1*u.species[wan].rv.back()*radius(rng)));
  const auto e=expected(u,q);
  std::vector<float> legacy(q.size()*size_t(u.frame.max_no),0.0f);
  for(size_t i=0;i<q.size();++i)legacy_point(u,q[i],legacy.data()+i*size_t(u.frame.max_no));
  size_t differing=0,values=0;double worst=0.0,worst_relative=0.0;
  for(size_t i=0;i<e.size();++i) {
    if(e[i]==0.0f && legacy[i]==0.0f)continue;
    ++values;
    const double d=std::fabs(double(e[i])-double(legacy[i]));
    if(std::memcmp(&e[i],&legacy[i],sizeof(float))!=0){++differing;worst=std::max(worst,d);
      worst_relative=std::max(worst_relative,d/std::max(std::fabs(double(e[i])),std::fabs(double(legacy[i]))));}
    require(d<=1.0e-6*std::max(1.0,std::fabs(double(e[i]))),"the legacy angle path and SOG_point_eval disagree beyond rounding");
  }
  std::cout<<"legacy acos/asin path vs SOG_point_eval: "<<differing<<" of "<<values<<" nonzero floats differ, max |diff| "
      <<worst<<", max relative "<<worst_relative<<'\n';
}
}
int main(int argc,char** argv) {
  try {
    const std::string name=argc>1?argv[1]:"equivalence";
    if(name=="equivalence")equivalence();
    else if(name=="runs")runs();
    else if(name=="edges")edges();
    else if(name=="invalid")invalid();
    else if(name=="legacy-difference")legacy_difference();
    else throw std::runtime_error("unknown host orbitals case");
    std::cout<<"PASS host orbitals "<<name<<" (modes:";
    for(int mode:modes())std::cout<<' '<<openmx_core_host_simd_name(mode);
    std::cout<<")\n";return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}

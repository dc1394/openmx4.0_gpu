#include <openmx/core/two_center.hpp>
#include "openmx_core_overlap.h"
#include "openmx_core_nonlocal.h"
#include "openmx_core_vna.h"
#include "allocation_faults.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <string>

using namespace openmx::core;
namespace {
using Complex=std::complex<long double>;
void require(bool ok,const char* why) {if(!ok)throw std::runtime_error(why);}
Complex value(dcomplex c) {return {c.r,c.i};}
void near(double actual,long double expected,const char* why)
{require(std::abs(actual-expected)<2e-11L*(1+std::abs(expected)),why);}
struct Fixture {
  int l0,l1;std::size_t channels,rows,cols;
  Tensor<dcomplex,3> source,conversion;
  TwoCenterWorkspace work;
  OpenMXTwoCenterInput in;
  Fixture(int a,int b,std::size_t n):l0(a),l1(b),channels(n),rows(2*a+5),cols(2*b+6),
    source({n,rows,cols}),conversion({15,31,31}),work(n,a,b) {
    for(std::size_t ch=0;ch<n;++ch)for(std::size_t i=0;i<rows;++i)for(std::size_t j=0;j<cols;++j)
      element(source.view(),ch,i,j)={std::sin(0.17+ch*0.3+i*0.19+j*0.41),std::cos(ch*0.23+i*0.31-j*0.37)};
    // The same real-harmonic change of basis, including g orbitals. Its rows
    // are unitary, and both signs of the imaginary entries are exercised.
    const double s=1/std::sqrt(2.);
    for(int l=0;l<15;++l) {
      element(conversion.view(),l,l,l)={1,0};
      for(int m=1;m<=l;++m) {
        const double sign=m%2?-1:1;
        element(conversion.view(),l,l+m,l-m)={s,0};
        element(conversion.view(),l,l+m,l+m)={sign*s,0};
        element(conversion.view(),l,l-m,l-m)={0,-s};
        element(conversion.view(),l,l-m,l+m)={0,sign*s};
      }
    }
    in={source.data(),rows,cols,2,3,a,b,conversion.data(),15,31,
        {1.73,std::sin(.81),std::cos(.81),std::sin(-.43),std::cos(-.43),0}};
  }
  Complex c(int l,int i,int k) const {return value(element(conversion.view(),l,i,k));}
  Complex spherical(std::size_t ch,int i,int j) const {
    Complex sum{};
    // Independent direct matrix product, using long-double complex math.
    for(int k=0;k<=2*l0;++k)for(int q=0;q<=2*l1;++q)
      sum+=std::conj(c(l0,i,k))*value(element(source.view(),ch,in.row+k,in.column+q))*c(l1,j,q);
    return sum;
  }
  Complex angular_momentum(int axis,int i,int j) const {
    Complex sum{};
    for(int k=0;k<=2*l0;++k)for(int q=0;q<=2*l1;++q)for(int p=0;p<=2*l1;++p) {
      const int m=p-l1;Complex L{};
      if(axis==2 && q==p)L=m;
      if(axis<2 && q==p+1) {
        const long double a=.5L*std::sqrt((long double)(l1-m)*(l1+m+1));
        L=axis==0?Complex{a,0}:Complex{0,-a};
      }
      if(axis<2 && q==p-1) {
        const long double a=.5L*std::sqrt((long double)(l1+m)*(l1-m+1));
        L=axis==0?Complex{a,0}:Complex{0,a};
      }
      sum+=std::conj(c(l0,i,k))*value(element(source.view(),0,in.row+k,in.column+q))*L*c(l1,j,p);
    }
    return sum;
  }
  void evaluate() {
    const auto b=work.buffers();char error[256];
    require(openmx_core_two_center_transform(&in,&b,error,sizeof(error)),error);
  }
  void reference() {
    evaluate();const auto b=work.buffers();const auto& g=in.geometry;
    for(int i=0;i<=2*l0;++i)for(int j=0;j<=2*l1;++j) {
      for(std::size_t base=0;base<channels;base+=4) {
        const long double scale=base?4:8;
        const auto v=spherical(base,i,j).real(),r=spherical(base+1,i,j).real(),
            t=spherical(base+2,i,j).real(),p=spherical(base+3,i,j).real();
        long double x=-scale*(g.sin_theta*g.cos_phi*r+g.cos_theta*g.cos_phi/g.distance*t),
                    y=-scale*(g.sin_theta*g.sin_phi*r+g.cos_theta*g.sin_phi/g.distance*t),
                    z=-scale*(g.cos_theta*r-g.sin_theta/g.distance*t);
        if(std::abs(g.sin_theta)>=10e-14) {
          x+=scale*g.sin_phi/g.sin_theta/g.distance*p;
          y-=scale*g.cos_phi/g.sin_theta/g.distance*p;
        }
        const std::array expected{scale*v,g.onsite?0.L:x,g.onsite?0.L:y,g.onsite?0.L:z};
        for(int ch=0;ch<4;++ch)near(b.values[((base+ch)*b.rows+i)*b.columns+j],expected[ch],"two-center matrix/derivative reference differs");
      }
      if(channels==8)for(int axis=0;axis<3;++axis)
        near(b.momenta[(axis*b.rows+i)*b.columns+j],8*angular_momentum(axis,i,j).imag(),"angular momentum reference differs");
    }
  }
};
void reference_test(std::size_t channels) {
  for(int a=0;a<=4;++a)for(int b=0;b<=4;++b) {Fixture f(a,b,channels);f.reference();}
  Fixture left_maximum(14,0,channels);left_maximum.reference();
  Fixture right_maximum(0,14,channels);right_maximum.reference();
}
void geometry_test() {
  Fixture f(1,2,8);
  for(double sine:{0.,1e-14,-1e-14,2e-13,-2e-13}) {
    f.in.geometry.sin_theta=sine;f.in.geometry.cos_theta=std::sqrt(1-sine*sine);f.reference();
  }
  f.in.geometry.onsite=1;f.reference();
  Fixture d(0,0,4);const auto g=d.in.geometry;
  const std::array<double,3> position={g.distance*g.sin_theta*g.cos_phi,g.distance*g.sin_theta*g.sin_phi,g.distance*g.cos_theta};
  const auto function=[](auto p){return 1.2*p[0]+.4*p[1]-.7*p[2];};
  element(d.source.view(),0,2,3)={function(position),0};
  element(d.source.view(),1,2,3)={1.2*g.sin_theta*g.cos_phi+.4*g.sin_theta*g.sin_phi-.7*g.cos_theta,0};
  element(d.source.view(),2,2,3)={g.distance*(1.2*g.cos_theta*g.cos_phi+.4*g.cos_theta*g.sin_phi+.7*g.sin_theta),0};
  element(d.source.view(),3,2,3)={g.distance*g.sin_theta*(-1.2*g.sin_phi+.4*g.cos_phi),0};
  d.evaluate();const auto result=d.work.buffers();
  for(int axis=0;axis<3;++axis) {
    auto plus=position,minus=position;plus[axis]+=1e-5;minus[axis]-=1e-5;
    near(result.values[axis+1],-8*(function(plus)-function(minus))/(2e-5),"Cartesian force differs from finite difference");
  }
}
void invalid_test() {
  Fixture f(2,1,4);f.evaluate();const auto original=f.work.buffers();
  const std::vector<double> saved(original.values,original.values+original.channels*original.rows*original.columns);
  const std::vector<dcomplex> left(original.left,original.left+original.channels*original.rows*original.columns);
  const auto reject=[&](OpenMXTwoCenterInput in,OpenMXTwoCenterBuffers b) {
    char error[19];require(!openmx_core_two_center_transform(&in,&b,error,sizeof(error)) && error[0] && !error[18],"invalid transform accepted or error unterminated");
    require(std::equal(saved.begin(),saved.end(),original.values),"failure changed values");
    for(std::size_t i=0;i<left.size();++i)require(left[i].r==original.left[i].r && left[i].i==original.left[i].i,"failure changed scratch");
  };
  for(int which=0;which<17;++which) {
    auto in=f.in;auto b=original;
    switch(which) {
      case 0:in.angular=nullptr;break;case 1:in.conversion=nullptr;break;
      case 2:in.left_angular=-1;break;case 3:in.right_angular=15;break;
      case 4:in.row=std::numeric_limits<std::size_t>::max();break;
      case 5:in.column=in.source_columns;break;case 6:in.conversion_width=2;break;
      case 7:in.conversion_orders=1;break;case 8:b.left=nullptr;break;
      case 9:b.values=nullptr;break;case 10:b.channels=5;break;case 11:b.rows=1;break;
      case 12:in.geometry.distance=0;break;case 13:in.geometry.sin_theta=std::nan("");break;
      case 14:in.source_rows=std::numeric_limits<std::size_t>::max();break;
      case 15:in.geometry.onsite=2;break;case 16:b.columns=0;break;
    }
    reject(in,b);
  }
  char error[1];require(!openmx_core_two_center_transform(nullptr,&original,error,sizeof(error)) && !error[0],"null input failure");
  require(!openmx_core_two_center_transform(&f.in,nullptr,nullptr,0),"null buffers accepted");
}
void lifecycle_test() {
  Fixture f(4,3,8);f.evaluate();const auto b=f.work.buffers();
  openmx::test::fail_allocation_after(0);
  f.evaluate();
  openmx::test::fail_allocation_after(-1);
  require(b.values==f.work.buffers().values && b.left==f.work.buffers().left,"transform reallocated storage");
  int failures=0;bool success=false;
  for(int n=0;n<20 && !success;++n) {
    openmx::test::fail_allocation_after(n);
    try {TwoCenterWorkspace owner(8,4,3);success=true;}
    catch(const std::bad_alloc&){++failures;}
    openmx::test::fail_allocation_after(-1);
  }
  require(success && failures==3,"two-center owner allocation cleanup not exercised");
  Fixture other(0,0,4);require(other.work.buffers().left!=b.left,"workspaces share scratch");
}
void adapters_test() {
  char error[256];
  OpenMXOverlapBuffers o{};const OpenMXOverlapShape os{4,2,3,0,1,0};
  std::unique_ptr<OpenMXOverlapWorkspace,decltype(&openmx_core_overlap_destroy)> ow(openmx_core_overlap_create(&os,&o,error,sizeof(error)),openmx_core_overlap_destroy);
  require(bool(ow),error);require(o.two_center.channels==8 && o.two_center.left==o.conversion[0] && o.two_center.momenta,"overlap lost shared transform owner");
  OpenMXVnaBuffers v{};const OpenMXVnaShape vs{4,2,3};
  std::unique_ptr<OpenMXVnaWorkspace,decltype(&openmx_core_vna_destroy)> vw(openmx_core_vna_create(&vs,&v,error,sizeof(error)),openmx_core_vna_destroy);
  require(bool(vw),error);require(v.two_center.channels==4 && v.two_center.left==v.conversion[0] && !v.two_center.momenta,"VNA lost shared transform owner");
  int count=2,angular[]={0,3,1};const OpenMXNonlocalShape ns{2,2,3,1,&count,angular,3,0,nullptr,0};OpenMXNonlocalBuffers n{};
  std::unique_ptr<OpenMXNonlocalWorkspace,decltype(&openmx_core_nonlocal_destroy)> nw(openmx_core_nonlocal_create(&ns,&n,error,sizeof(error)),openmx_core_nonlocal_destroy);
  require(bool(nw),error);require(n.two_center.rows==5 && n.two_center.columns==7 && n.two_center.left==n.conversion[0],"nonlocal lost rectangular transform owner");
  Fixture f(0,3,4);
  const OpenMXTwoCenterInput in={n.angular[0],n.orbital_capacity,n.projector_capacity,0,0,0,3,
      f.conversion.data(),15,31,{1,0,1,0,1,1}};
  // Use a live independent conversion table; the source is the native packed layout.
  auto input=in;
  require(openmx_core_two_center_transform(&input,&n.two_center,error,sizeof(error)),error);
}
void concurrent_test() {
  std::array<std::future<void>,4> jobs;
  for(int i=0;i<4;++i)jobs[i]=std::async(std::launch::async,[i] {Fixture f(i,4-i,i%2?4:8);for(int j=0;j<3;++j)f.reference();});
  for(auto& job:jobs)job.get();
}
}
int main(int argc,char** argv) {
  try {
    const std::string test=argc>1?argv[1]:"";
    if(test=="overlap")reference_test(8);else if(test=="nonlocal-vna")reference_test(4);
    else if(test=="geometry")geometry_test();else if(test=="invalid")invalid_test();
    else if(test=="lifecycle")lifecycle_test();else if(test=="adapters")adapters_test();
    else if(test=="concurrent")concurrent_test();else throw std::runtime_error("unknown test");
    return 0;
  } catch(const std::exception& e) {openmx::test::fail_allocation_after(-1);std::cerr<<e.what()<<'\n';return 1;}
}

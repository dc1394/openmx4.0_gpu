#include <openmx/basis/gaunt.hpp>
#include <openmx/basis/harmonics.hpp>
#include <openmx/basis/lebedev.hpp>
#include "allocation_gate.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <climits>
#include <cmath>
#include <cstdint>
#include <future>
#include <iostream>
#include <numbers>
#include <stdexcept>
#include <string_view>

extern "C" double legacy_Gaunt(int,int,int,int,int,int);
namespace {
using namespace openmx::basis;
void require(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
void near(double actual,double expected,double tolerance=3e-12) {
  require(std::isfinite(actual) && std::abs(actual-expected)<=tolerance*(1+std::abs(expected)),"Gaunt physical value differs");
}
void bits(double actual,double expected) {
  require(std::isfinite(actual) && std::bit_cast<std::uint64_t>(actual)==std::bit_cast<std::uint64_t>(expected),"legacy Gaunt arithmetic changed");
}
template<class F> void rejects(F&& f) {
  bool caught=false;try{f();}catch(const std::exception&){caught=true;}require(caught,"invalid Gaunt request accepted");
}
void coefficients() {
  std::size_t count=0;
  for(int l=0;l<=6;++l)for(int m=-l;m<=l;++m)
    for(int l1=0;l1<=6;++l1)for(int m1=-l1;m1<=l1;++m1)
      for(int l2=0;l2<=6;++l2)for(int m2=-l2;m2<=l2;++m2) {
        bits(gaunt({l,m},{l1,m1},{l2,m2}),legacy_Gaunt(l,m,l1,m1,l2,m2));++count;
      }
  for(const auto ls:{std::array{19,19,0},std::array{19,0,19},std::array{13,13,12},std::array{0,19,19}})
    for(int m=-ls[0];m<=ls[0];++m)for(int m1=-ls[1];m1<=ls[1];++m1) {
      const int m2=m-m1;if(std::abs(m2)>ls[2])continue;
      bits(gaunt({ls[0],m},{ls[1],m1},{ls[2],m2}),legacy_Gaunt(ls[0],m,ls[1],m1,ls[2],m2));++count;
    }
  std::cout<<count<<" frozen scalar values\n";
}
void physics() {
  const double s=1/std::sqrt(4*std::numbers::pi);
  for(int l=0;l<=19;++l)for(int m=-l;m<=l;++m) {
    near(gaunt({l,m},{l,m},{0,0}),s);
    near(gaunt({0,0},{l,m},{l,-m}),m%2?-s:s);
  }
  // Independent Cartesian moments: Y10=sqrt(3/4pi)z and
  // Y20=sqrt(5/16pi)(3z^2-1). Sphere moments <z^2>=1/3, <z^4>=1/5.
  near(gaunt({2,0},{1,0},{1,0}),std::sqrt(5/std::numbers::pi)/5);
  near(gaunt({1,0},{1,0},{1,0}),0.);
  near(gaunt({4,0},{1,0},{1,0}),0.);
  near(gaunt({2,1},{1,0},{1,0}),0.);
  for(int l=0;l<=5;++l)for(int l1=0;l1<=5;++l1)for(int l2=0;l2<=5;++l2)
    for(int m=-l;m<=l;++m) {
      const int m1=std::min(l1,std::max(-l1,m)),m2=m-m1;
      if(std::abs(m2)>l2)continue;
      const double value=gaunt({l,m},{l1,m1},{l2,m2});
      near(value,gaunt({l,m},{l2,m2},{l1,m1}));
      near(value,gaunt({l,-m},{l1,-m1},{l2,-m2}));
      if(l>l1+l2 || l<std::abs(l1-l2) || (l+l1+l2)%2)near(value,0.);
    }
}
void integration() {
  const LebedevRule quadrature(590);
  // Numerical solid-angle integration is independent of the 3j factorial sum.
  const std::array triples{
    std::array{HarmonicIndex{2,0},HarmonicIndex{1,0},HarmonicIndex{1,0}},
    std::array{HarmonicIndex{4,-3},HarmonicIndex{3,-1},HarmonicIndex{3,-2}},
    std::array{HarmonicIndex{6,2},HarmonicIndex{4,3},HarmonicIndex{4,-1}},
    std::array{HarmonicIndex{5,1},HarmonicIndex{3,-2},HarmonicIndex{4,3}},
    std::array{HarmonicIndex{13,5},HarmonicIndex{13,2},HarmonicIndex{12,3}},
    std::array{HarmonicIndex{19,-9},HarmonicIndex{19,-9},HarmonicIndex{0,0}}};
  for(auto t:triples) {
    std::complex<double> sum{};
    for(std::size_t p=0;p<quadrature.size();++p) {
      const auto q=quadrature.point(p);const double theta=std::acos(std::clamp(q[2],-1.,1.)),phi=std::atan2(q[1],q[0]);
      const auto a=integral_harmonic(t[0].degree,t[0].order,theta,phi).value;
      const auto b=integral_harmonic(t[1].degree,t[1].order,theta,phi).value;
      const auto c=integral_harmonic(t[2].degree,t[2].order,theta,phi).value;
      sum+=q[3]*std::conj(a)*b*c;
    }
    near(4*std::numbers::pi*sum.real(),gaunt(t[0],t[1],t[2]));near(sum.imag(),0.);
  }
}
void tables() {
  for(auto shape:{std::array{0,0,0},std::array{4,3,8},std::array{5,5,10},std::array{19,0,19},std::array{4,12,24},std::array{38,0,0}}) {
    GauntTable table(shape[0],shape[1],shape[2]);const auto data=table.view();
    require(table.values().size()==std::size_t(data.l0max+1)*(data.l0max+1)*(data.l1max+1)*(data.llmax+1)*data.row_stride,"Gaunt table dimensions differ");
    for(int l0=0;l0<=data.l0max;++l0)for(int m0=-l0;m0<=l0;++m0)
      for(int ll=0;ll<=data.llmax;++ll)for(int l1=0;l1<=data.l1max;++l1) {
        const auto row=table.row(l0,m0,ll,l1);
        require(row.size()==std::size_t(2*l1+1) && row.data()==data.values+openmx_gaunt_table_row_offset(&data,l0,m0,ll,l1),"Gaunt row is copied or has wrong stride");
        for(int m1=-l1;m1<=l1;++m1)
          bits(table.coefficient(l0,m0,l1,m1,ll),legacy_Gaunt(l0,m0,l1,m1,ll,m0-m1));
        for(std::size_t i=2*l1+1;i<data.row_stride;++i)bits(row.data()[i],0.);
      }
    std::vector<double> buffer(table.values().size()+3,17.);OpenMXGauntTableData view{};char error[160];
    require(openmx_basis_gaunt_table_fill(shape[0],shape[1],shape[2],buffer.data(),buffer.size(),&view,error,sizeof(error)),"C Gaunt fill failed");
    require(view.values==buffer.data() && error[0]=='\0',"C Gaunt descriptor differs");
    for(std::size_t i=0;i<table.values().size();++i)bits(buffer[i],table.values()[i]);
    for(std::size_t i=table.values().size();i<buffer.size();++i)bits(buffer[i],17.);
  }
}
void invalid() {
  for(int l:{INT_MIN,-1,39,INT_MAX}) {
    rejects([&]{(void)gaunt({l,0},{0,0},{0,0});});rejects([&]{GauntTable bad(l,0,0);});
    rejects([&]{GauntTable bad(0,l,0);});rejects([&]{GauntTable bad(0,0,l);});
  }
  rejects([]{(void)gaunt({19,0},{19,0},{1,0});});rejects([]{GauntTable bad(19,19,1);});
  for(int m:{INT_MIN,-3,3,INT_MAX})rejects([&]{(void)gaunt({2,m},{1,0},{1,0});});
  GauntTable table(2,2,4);
  for(int i:{INT_MIN,-1,3,INT_MAX})rejects([&]{(void)table.row(i,0,0,0);});
  rejects([&]{(void)table.row(2,3,0,0);});rejects([&]{(void)table.row(0,0,5,0);});
  rejects([&]{(void)table.row(0,0,0,3);});rejects([&]{(void)table.coefficient(0,0,2,3,1);});
  double value=17.;char error[128];
  require(!openmx_basis_gaunt(-1,0,0,0,0,0,&value,error,sizeof(error)) && value==17. && *error,"invalid C scalar modified output");
  require(!openmx_basis_gaunt(0,0,0,0,0,0,nullptr,nullptr,0),"null scalar accepted");
  std::array<double,4> storage{1,2,3,4};const auto saved=storage;
  OpenMXGauntTableData out{7,8,9,10,storage.data()};const auto old=out;
  require(!openmx_basis_gaunt_table_fill(1,1,1,storage.data(),4,&out,error,sizeof(error)),"short Gaunt table accepted");
  require(!openmx_basis_gaunt_table_fill(-1,0,0,storage.data(),4,&out,error,sizeof(error)),"negative table accepted");
  require(!openmx_basis_gaunt_table_fill(0,0,0,nullptr,4,&out,error,sizeof(error)),"null table accepted");
  require(!openmx_basis_gaunt_table_fill(0,0,0,storage.data(),4,nullptr,error,sizeof(error)),"null descriptor accepted");
  require(storage==saved && out.values==old.values && out.l0max==old.l0max && out.l1max==old.l1max && out.llmax==old.llmax && out.row_stride==old.row_stride,"failed C table changed output");
  char tiny='x';require(!openmx_basis_gaunt(-1,0,0,0,0,0,&value,&tiny,1) && tiny=='\0',"error not terminated");
}
void ownership() {
  GauntTable first(4,3,8);auto copy=first;const double expected=first.coefficient(2,0,1,0,1);
  require(copy.view().values!=first.view().values,"copied tables share storage");
  auto moved=std::move(copy);require(copy.values().empty() && !copy.view().values,"move source exposes storage");
  rejects([&]{(void)copy.row(0,0,0,0);});
  copy=moved;first=GauntTable(0,0,0);moved=std::move(first);near(copy.coefficient(2,0,1,0,1),expected);
  auto* self=&copy;copy=*self;copy=std::move(*self);near(copy.coefficient(2,0,1,0,1),expected);
}
void allocation() {
  GauntTable original(4,3,8),target(0,0,0);const auto address=target.view().values;
  bool failed=false;allocations_before_failure=0;
  try{target=original;}catch(const std::bad_alloc&){failed=true;}
  allocations_before_failure=-1;require(failed && target.view().values==address && target.view().l0max==0,"failed assignment changed owner");
  failed=false;allocations_before_failure=0;try{GauntTable bad(4,3,8);}catch(const std::bad_alloc&){failed=true;}
  allocations_before_failure=-1;require(failed,"construction allocation fault not reached");
  std::array<double,72> buffer{};OpenMXGauntTableData out{};char error[128];double scalar=0.;
  allocations_before_failure=0;
  const auto row=original.row(2,0,1,1);const auto view=original.view();const double direct=gaunt({2,0},{1,0},{1,0});
  const int a=openmx_basis_gaunt(2,0,1,0,1,0,&scalar,error,sizeof(error));
  const int b=openmx_basis_gaunt_table_fill(1,1,1,buffer.data(),buffer.size(),&out,error,sizeof(error));
  allocations_before_failure=-1;
  require(a && b && view.values && !row.empty(),"allocation-free evaluation failed");near(scalar,direct);
}
void concurrent() {
  const GauntTable shared(4,3,8);std::array<std::future<void>,6> workers;
  for(int i=0;i<6;++i)workers[i]=std::async(std::launch::async,[&,i] {
    GauntTable own(3+i%2,3,6+i%2);
    for(int repeat=0;repeat<30;++repeat)for(int m=-3;m<=3;++m) {
      near(shared.coefficient(3,m,3,m,0),gaunt({3,m},{3,m},{0,0}));
      near(own.coefficient(3,m,3,m,0),shared.coefficient(3,m,3,m,0));
    }
  });
  for(auto& worker:workers)worker.get();
}
} // namespace
int main(int argc,char** argv) {
  try {
    require(argc==2,"expected Gaunt test name");const std::string_view name=argv[1];
    if(name=="coefficients")coefficients();else if(name=="physics")physics();else if(name=="integration")integration();
    else if(name=="tables")tables();else if(name=="invalid")invalid();else if(name=="ownership")ownership();
    else if(name=="allocation")allocation();else if(name=="concurrent")concurrent();else throw std::invalid_argument("unknown Gaunt test");
    std::cout<<"PASS Gaunt "<<name<<'\n';
  }catch(const std::exception& error){allocations_before_failure=-1;std::cerr<<error.what()<<'\n';return 1;}
}

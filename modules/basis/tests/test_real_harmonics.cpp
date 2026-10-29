#include <openmx/basis/real_harmonics.hpp>
#include <openmx/basis/harmonics.hpp>
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

namespace {
using openmx::basis::RealHarmonicTransform;
using ComplexValue=std::complex<double>;
void require(bool ok,const char* why) {if(!ok)throw std::runtime_error(why);}
void near(ComplexValue actual,ComplexValue expected) {
  require(std::isfinite(actual.real()) && std::isfinite(actual.imag()) &&
      std::abs(actual-expected)<3e-12*(1+std::abs(expected)),"real harmonic value differs");
}
template<class F> void rejects(F&& operation) {
  bool caught=false;try{operation();}catch(const std::exception&){caught=true;}
  require(caught,"invalid real harmonic request accepted");
}
// This frozen initializer is only used by tests; production has no global table.
constexpr int YOUSO36=14;
dcomplex Comp2Real[15][31][31]{};
dcomplex Complex(double r,double i) {return {r,i};}
#include "fixtures/real_harmonics_before_owner.inc"

void values() {
  Set_Comp2Real();
  for(int maximum=0;maximum<=14;++maximum) {
    const RealHarmonicTransform transform(maximum);const auto view=transform.view();
    require(view.orders==std::size_t(maximum+1) && view.width==std::size_t(2*maximum+1),"transform shape differs");
    require(transform.coefficients().size()==view.orders*view.width*view.width,"table not contiguous");
    for(std::size_t l=0;l<view.orders;++l)for(std::size_t row=0;row<view.width;++row)
      for(std::size_t col=0;col<view.width;++col) {
        const auto current=openmx_real_harmonic_coefficient(&view,l,row,col),old=Comp2Real[l][row][col];
        require(std::bit_cast<std::uint64_t>(current.r)==std::bit_cast<std::uint64_t>(old.r) &&
            std::bit_cast<std::uint64_t>(current.i)==std::bit_cast<std::uint64_t>(old.i),"legacy coefficient/padding changed");
        if(row<=2*l && col<=2*l)
          require(&transform.coefficient(l,row,col)==view.coefficients+(l*view.width+row)*view.width+col,"checked coefficient copies storage");
      }
  }
}
void unitarity() {
  const RealHarmonicTransform transform(64);
  for(int l=0;l<=64;++l) {
    const int n=2*l+1;
    for(int row=0;row<n;++row)for(int other=0;other<n;++other) {
      ComplexValue product=0.;
      for(int col=0;col<n;++col) {
        auto a=transform.coefficient(l,row,col),b=transform.coefficient(l,other,col);
        product+=ComplexValue(a.r,a.i)*std::conj(ComplexValue(b.r,b.i));
      }
      near(product,row==other?1.:0.);
    }
  }
}
std::vector<double> cartesian(int l,double x,double y,double z) {
  const double pi=std::numbers::pi;
  if(l==0)return {1/std::sqrt(4*pi)};
  if(l==1){const auto a=std::sqrt(3/(4*pi));return {a*x,a*y,a*z};}
  if(l==2)return {std::sqrt(5/(16*pi))*(3*z*z-1),std::sqrt(15/(16*pi))*(x*x-y*y),
      std::sqrt(15/(4*pi))*x*y,std::sqrt(15/(4*pi))*x*z,std::sqrt(15/(4*pi))*y*z};
  if(l==3)return {std::sqrt(7/(16*pi))*z*(5*z*z-3),std::sqrt(21/(32*pi))*x*(5*z*z-1),
      std::sqrt(21/(32*pi))*y*(5*z*z-1),std::sqrt(105/(16*pi))*z*(x*x-y*y),
      std::sqrt(105/(4*pi))*x*y*z,std::sqrt(35/(32*pi))*x*(x*x-3*y*y),
      std::sqrt(35/(32*pi))*y*(3*x*x-y*y)};
  const double xx=x*x,yy=y*y,zz=z*z;
  return {3/(16*std::sqrt(pi))*(35*zz*zz-30*zz+3),3./8*std::sqrt(10/pi)*x*z*(7*zz-3),
      3./8*std::sqrt(10/pi)*y*z*(7*zz-3),3./8*std::sqrt(5/pi)*(xx-yy)*(7*zz-1),
      3./4*std::sqrt(5/pi)*x*y*(7*zz-1),3./8*std::sqrt(70/pi)*x*z*(xx-3*yy),
      3./8*std::sqrt(70/pi)*y*z*(3*xx-yy),3./16*std::sqrt(35/pi)*(xx*xx-6*xx*yy+yy*yy),
      3./4*std::sqrt(35/pi)*x*y*(xx-yy)};
}
void orbitals() {
  const RealHarmonicTransform transform(4);
  for(double theta:{0.,.21,.87,1.51,2.18,2.9,std::numbers::pi})for(double phi:{-2.6,-.3,0.,.71,2.1}) {
    const auto x=std::sin(theta)*std::cos(phi),y=std::sin(theta)*std::sin(phi),z=std::cos(theta);
    for(int l=0;l<=4;++l) {
      const auto expected=cartesian(l,x,y,z);std::vector<ComplexValue> input(2*l+1),output(input.size());
      for(int m=-l;m<=l;++m)input[l+m]=openmx::basis::integral_harmonic(l,m,theta,phi).value;
      transform.transform(l,input,output);
      for(int row=0;row<=2*l;++row)near(output[row],expected[row]);
      transform.transform(l,input,input);for(int row=0;row<=2*l;++row)near(input[row],expected[row]);
    }
  }
  // Arbitrary complex inputs need not produce real output; partial overlap is
  // supported as well as exact in-place use.
  std::array<ComplexValue,6> overlapping{{{.3,.2},{-.4,.7},{.5,-.1},{.8,.9},{-.2,.6},{0.,0.}}};
  std::array<ComplexValue,5> expected{};
  transform.transform(2,std::span(overlapping).first<5>(),expected);
  transform.transform(2,std::span(overlapping).first<5>(),std::span(overlapping).subspan<1>());
  for(int i=0;i<5;++i)near(overlapping[i+1],expected[i]);
}
void invalid() {
  for(int maximum:{INT_MIN,-1,65,INT_MAX})rejects([&]{RealHarmonicTransform bad(maximum);});
  const RealHarmonicTransform transform(4);
  for(int l:{-1,5,INT_MAX})rejects([&]{(void)transform.coefficient(l,0,0);});
  for(int i:{-1,9,INT_MAX}) {
    rejects([&]{(void)transform.coefficient(4,i,0);});rejects([&]{(void)transform.coefficient(4,0,i);});
  }
  std::array<ComplexValue,9> input{},output;output.fill({17.,31.});const auto saved=output;
  rejects([&]{transform.transform(5,input,output);});
  rejects([&]{transform.transform(4,std::span(input).first<8>(),output);});
  rejects([&]{transform.transform(4,input,std::span(output).first<8>());});
  require(output==saved,"invalid transform changed output");
  std::array<dcomplex,405> coefficients;coefficients.fill({13.,29.});char error[128];
  for(int degree:{-1,65})require(!openmx_basis_real_harmonics_fill(degree,coefficients.data(),405,error,sizeof error) && *error,"invalid C degree accepted");
  require(!openmx_basis_real_harmonics_fill(4,coefficients.data(),404,error,sizeof error),"short C table accepted");
  require(!openmx_basis_real_harmonics_fill(4,nullptr,405,nullptr,0),"null C table accepted");
  char tiny='x';require(!openmx_basis_real_harmonics_fill(-1,coefficients.data(),405,&tiny,1) && tiny=='\0',"C error not terminated");
  for(auto z:coefficients)require(z.r==13. && z.i==29.,"failed C fill changed coefficients");
}
void ownership() {
  RealHarmonicTransform first(4);auto copy=first;
  require(first.view().coefficients!=copy.view().coefficients,"copies share storage");
  auto moved=std::move(copy);require(copy.maximum_degree()==-1 && !copy.view().coefficients && copy.coefficients().empty(),"moved-from view still exposes data");
  rejects([&]{(void)copy.coefficient(0,0,0);});
  first=RealHarmonicTransform(1);copy=moved;moved=std::move(first);
  require(copy.maximum_degree()==4 && moved.maximum_degree()==1 && first.maximum_degree()==-1,"assignments corrupted owners");
  moved=std::move(moved);copy=copy;require(moved.coefficient(0,0,0).r==1 && copy.coefficient(4,0,4).r==1,"self assignment damaged data");
}
void allocation() {
  bool success=false;int failures=0;
  for(int index=0;index<8;++index) {
    allocations_before_failure=index;
    try {const RealHarmonicTransform owner(14);success=owner.maximum_degree()==14;}catch(const std::bad_alloc&){++failures;}
    allocations_before_failure=-1;if(success)break;
  }
  require(success && failures>0,"constructor fault sweep incomplete");
  RealHarmonicTransform small(1),large(14);const auto before=small.view();bool failed=false;
  allocations_before_failure=0;try{small=large;}catch(const std::bad_alloc&){failed=true;}allocations_before_failure=-1;
  require(failed && small.view().coefficients==before.coefficients && small.maximum_degree()==1 && small.coefficient(0,0,0).r==1,"failed copy assignment changed owner");
  std::array<ComplexValue,29> input{},output;input[14]=1.;
  std::array<dcomplex,407> c_values;c_values.fill({71.,73.});char error[160];
  allocations_before_failure=0;
  large.transform(14,input,output);
  const int ok=openmx_basis_real_harmonics_fill(4,c_values.data(),c_values.size(),error,sizeof error);
  const auto remaining=allocations_before_failure.load();allocations_before_failure=-1;
  require(ok && !*error && remaining==0 && c_values.back().r==71. && c_values.back().i==73.,"C fill or transform allocated/overran output");
  near(output[0],1.);for(std::size_t i=1;i<output.size();++i)near(output[i],0.);
}
void concurrent() {
  const RealHarmonicTransform shared(14);std::array<std::future<void>,4> workers;
  for(int t=0;t<4;++t)workers[t]=std::async(std::launch::async,[&,t]{
    const RealHarmonicTransform own(4+t);
    for(int repeat=0;repeat<20;++repeat)for(int l=3;l<=4+t;++l) {
      std::array<ComplexValue,29> data{};data[l]=repeat+t+.25;
      shared.transform(l,std::span(data).first(2*l+1),std::span(data).first(2*l+1));
      near(data[0],repeat+t+.25);require(own.coefficient(l,0,l).r==1.,"independent transforms interfere");
    }
  });
  for(auto& worker:workers)worker.get();
}
} // namespace
int main(int argc,char** argv) {
  try {
    require(argc==2,"expected real harmonic test name");const std::string_view name=argv[1];
    if(name=="values")values();else if(name=="unitarity")unitarity();else if(name=="orbitals")orbitals();
    else if(name=="invalid")invalid();else if(name=="ownership")ownership();else if(name=="allocation")allocation();
    else if(name=="concurrent")concurrent();else throw std::invalid_argument("unknown real harmonic test");
    std::cout<<"PASS real harmonics "<<name<<'\n';
  }catch(const std::exception& error){allocations_before_failure=-1;std::cerr<<error.what()<<'\n';return 1;}
}

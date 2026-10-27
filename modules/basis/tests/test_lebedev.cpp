#include <openmx/basis/lebedev.hpp>
#include <openmx/basis/lebedev.h>
#include "allocation_gate.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <climits>
#include <cmath>
#include <cstdint>
#include <future>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {
using openmx::basis::LebedevRule;
void require(bool ok,const char* why) { if(!ok)throw std::runtime_error(why); }
void near(long double x,long double y) { require(std::isfinite(x) && std::abs(x-y)<3e-12L,"Lebedev moment differs"); }
template<class F> void rejects(F&& f) {
  bool caught=false;try {f();}catch(const std::exception&) {caught=true;}
  require(caught,"invalid Lebedev request accepted");
}
long double double_factorial(int n) { long double value=1;for(int i=n;i>1;i-=2)value*=i;return value; }
long double moment(int a,int b,int c) {
  if(a%2 || b%2 || c%2)return 0;
  return double_factorial(a-1)*double_factorial(b-1)*double_factorial(c-1)/double_factorial(a+b+c+1);
}
void moments() {
  for(int count:{590,1202,2030}) {
    const LebedevRule rule(count);require(rule.size()==static_cast<std::size_t>(count),"wrong count");
    for(std::size_t i=0;i<rule.size();++i) {
      const auto point=rule.point(i);near(point[0]*point[0]+point[1]*point[1]+point[2]*point[2],1);
      require(point[3]>0 && std::isfinite(point[3]),"invalid angular weight");
    }
    // Unit-sphere averages of monomials, including mixed/odd components.
    for(int a=0;a<=10;++a)for(int b=0;b<=10-a;++b)for(int c=0;c<=10-a-b;++c) {
      long double sum=0;
      for(std::size_t i=0;i<rule.size();++i) {
        const auto p=rule.point(i);
        sum+=static_cast<long double>(p[3])*std::pow((long double)p[0],a)*
            std::pow((long double)p[1],b)*std::pow((long double)p[2],c);
      }
      near(sum,moment(a,b,c));
    }
    for(int degree:{12,16,20,24,32}) {
      long double sum=0;for(std::size_t i=0;i<rule.size();++i)sum+=rule.point(i)[3]*std::pow((long double)rule.point(i)[0],degree);
      near(sum,1.L/(degree+1));
    }
  }
}
void coefficients() {
  // Word-wise fingerprints of the frozen OpenMX table at 68322894. All 15,288
  // coefficients and their point order participate; independent of byte order.
  const std::array<int,3> sizes{590,1202,2030};
  const std::array<std::uint64_t,3> expected{0xbc2169c64e1a0775ULL,0xfca6b0e9c7532a59ULL,0x06ccb52429d058d1ULL};
  for(std::size_t n=0;n<sizes.size();++n) {
    const LebedevRule rule(sizes[n]);std::uint64_t hash=14695981039346656037ULL;
    for(double x:rule.xyzw())hash=(hash^std::bit_cast<std::uint64_t>(x))*1099511628211ULL;
    require(hash==expected[n],"legacy coefficient or point order changed");
    static_assert(std::is_same_v<decltype(rule.point(0)),std::span<const double,4>>);
    for(std::size_t i=0;i<rule.size();++i)require(rule.point(i).data()==rule.xyzw().data()+4*i,"point is not a contiguous view");
  }
}
void invalid() {
  LebedevRule rule(590);const auto before=rule.xyzw();std::vector<double> saved(before.begin(),before.end());
  for(int n:{INT_MIN,-1,0,1,591,1201,2031,INT_MAX}) {
    rejects([&]{LebedevRule bad(n);});rejects([&]{rule.reset(n);});
    require(rule.xyzw().data()==before.data() && rule.size()==590 && std::equal(saved.begin(),saved.end(),rule.xyzw().begin()),"invalid reset damaged rule");
  }
  rejects([&]{(void)rule.point(590);});rejects([&]{(void)rule.point(static_cast<std::size_t>(-1));});
  std::array<double,4*590+2> output;output.fill(-77);char error[128];
  for(int n:{-1,0,591})require(!openmx_basis_lebedev_fill(n,output.data(),output.size(),error,sizeof error) && *error,"invalid C count accepted");
  require(!openmx_basis_lebedev_fill(590,nullptr,output.size(),error,sizeof error),"null C output accepted");
  require(!openmx_basis_lebedev_fill(590,output.data(),4*590-1,error,sizeof error),"short C output accepted");
  for(double x:output)require(x==-77,"failed C call changed output");
  char tiny='x';require(!openmx_basis_lebedev_fill(591,output.data(),output.size(),&tiny,1) && tiny=='\0',"C diagnostic not terminated");
  require(!openmx_basis_lebedev_fill(591,output.data(),output.size(),nullptr,0),"C call requires error buffer");
}
void ownership() {
  LebedevRule original(590);auto copy=original;
  require(copy.xyzw().data()!=original.xyzw().data(),"copied rule aliases original");
  auto moved=std::move(copy);require(copy.size()==0 && copy.xyzw().empty(),"moved-from rule still exposes data");
  rejects([&]{(void)copy.point(0);});copy.reset(1202);
  original.reset(2030);moved=copy;copy=std::move(original);
  require(original.size()==0 && moved.size()==1202 && copy.size()==2030,"assignment changed another owner");
  copy=std::move(copy);moved=moved;require(copy.size()==2030 && moved.size()==1202,"self assignment damaged owner");
  original.reset(590);near(original.point(0)[3],.000309512129531L);
}
void allocation() {
  bool succeeded=false;int failures=0;
  for(int i=0;i<8;++i) {
    allocations_before_failure=i;
    try {const LebedevRule rule(590);succeeded=rule.size()==590;}catch(const std::bad_alloc&) {++failures;}
    allocations_before_failure=-1;if(succeeded)break;
  }
  require(succeeded && failures>0,"constructor allocation failures not covered");
  LebedevRule rule(590);const auto before=rule.xyzw();std::vector<double> saved(before.begin(),before.end());
  allocations_before_failure=0;bool failed=false;
  try {rule.reset(2030);}catch(const std::bad_alloc&) {failed=true;}
  allocations_before_failure=-1;
  require(failed && rule.size()==590 && rule.xyzw().data()==before.data() && std::equal(saved.begin(),saved.end(),rule.xyzw().begin()),"failed growth changed rule");
  const LebedevRule other(2030);allocations_before_failure=0;failed=false;
  try {rule=other;}catch(const std::bad_alloc&) {failed=true;}
  allocations_before_failure=-1;
  require(failed && rule.xyzw().data()==before.data() && rule.size()==590,"failed copy assignment changed rule");
  rule.reset(2030);const auto address=rule.xyzw().data();
  allocations_before_failure=0;
  for(int n:{2030,590,1202,2030,590}) {
    rule.reset(n);require(rule.xyzw().data()==address,"capacity reuse moved data");
  }
  std::array<double,4*2030+2> data;data.fill(-99);char error[128];
  const int ok=openmx_basis_lebedev_fill(2030,data.data(),data.size(),error,sizeof error);
  allocations_before_failure=-1;
  require(ok && !error[0] && data[4*2030]==-99 && data.back()==-99,"C fill allocated or damaged padding");
  require(std::equal(other.xyzw().begin(),other.xyzw().end(),data.begin()),"C fill differs from C++ rule");
}
void concurrent() {
  const LebedevRule shared(1202);std::array<std::future<void>,4> workers;
  for(int t=0;t<4;++t)workers[t]=std::async(std::launch::async,[&]{
    LebedevRule local(2030);for(int repeat=0;repeat<10;++repeat)for(int n:{590,1202,2030}) {
      local.reset(n);near(local.point(0)[0],1);near(shared.point(0)[0],1);
      long double sum=0;for(std::size_t i=0;i<local.size();++i)sum+=local.point(i)[3];near(sum,1);
    }
  });
  for(auto& worker:workers)worker.get();
}
} // namespace
int main(int argc,char** argv) {
  try {
    if(argc!=2)throw std::invalid_argument("expected Lebedev test case");const std::string name=argv[1];
    if(name=="moments")moments();else if(name=="coefficients")coefficients();else if(name=="invalid")invalid();
    else if(name=="ownership")ownership();else if(name=="allocation")allocation();else if(name=="concurrent")concurrent();
    else throw std::invalid_argument("unknown Lebedev test case");
    std::cout<<"PASS Lebedev "<<name<<'\n';return 0;
  }catch(const std::exception& error){allocations_before_failure=-1;std::cerr<<error.what()<<'\n';return 1;}
}

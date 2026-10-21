#include <openmx/core/core_hole.hpp>
#include "allocation_faults.hpp"
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
using namespace openmx::core;
namespace {
void require(bool ok,const char *why){if(!ok)throw std::runtime_error(why);}
void near(double a,double b){require(std::isfinite(a) && std::abs(a-b)<1e-13*(1+std::abs(b)),"core-hole value differs");}
struct Input {
  int basis[5]{2,2,2,2,1};
  OpenMXCoreHoleInput input{7,3,1,1,0,4,41,basis,1.75};
};
void coefficients()
{
  for(int spin:{0,1,3})for(int angular=0;angular<=3;++angular)for(int state=0;state<=2*(2*angular+1);++state) {
    Input in;in.input.spin=spin;in.input.angular=angular;in.input.state=state;
    CoreHole owner(in.input);const auto view=owner.view();
    int offset=0;for(int l=0;l<angular;l++)offset+=(2*l+1)*in.basis[l];
    const int width=2*angular+1;
    require(view.offset==std::size_t(offset) && view.orbitals==41 && view.active && view.atom==7,"core-hole channel layout");
    for(int s=0;s<2;s++)for(int i=0;i<41;i++) {
      const bool selected=state==0?(i>=offset && i<offset+width):
          (s==(state<=width?0:1) && i==offset+(state-1)%width);
      near(view.collinear[s][i],selected?1.75:0);
    }
    double sum=0,opposite=0;
    for(int i=0;i<OPENMX_CH_WEIGHTS;i++){sum+=view.weights[i];opposite+=view.opposite_weights[i];}
    near(sum,state==0?3.5:1.75/width);near(opposite,-sum);
  }
  Input in;in.basis[1]=0;in.input.orbitals-=6;
  CoreHole absent(in.input);require(!absent.view().active,"absent shell became active");
  OpenMXCoreHoleTrace empty{};empty.sign=1;std::array<dcomplex,9> result{};
  core_hole_trace(absent.view(),empty,result);
  for(auto x:result)require(x.r==0 && x.i==0,"absent shell changed trace");
}
int snapshot(OpenMXSolverInvocation *scope,void*,double *result)
{
  Input in;OpenMXCoreHoleView first,second;char error[256];
  require(openmx_core_core_hole_prepare(scope,&in.input,&first,error,sizeof(error)),error);
  in.basis[0]=100;in.input.penalty=-99;
  require(first.offset==2 && first.penalty==1.75 && first.collinear[0][2]==1.75,"core-hole owner borrowed mutable metadata");
  in=Input{};in.input.multiplicity=in.basis;
  for(int i=0;i<20;i++) {
    in.input.penalty=i+.5;
    require(openmx_core_core_hole_prepare(scope,&in.input,&second,error,sizeof(error)),error);
    near(second.weights[OPENMX_CH_D32_P32],(i+.5)/3.0);
    near(first.collinear[1][4],1.75);
  }
  in.input.state=7;const auto saved=second;
  require(!openmx_core_core_hole_prepare(scope,&in.input,&second,error,sizeof(error)),"invalid core-hole state accepted");
  require(second.weights==saved.weights && second.collinear[0]==saved.collinear[0],"failed preparation changed view");
  *result=19;return 1;
}
void invalid()
{
  Input in;CoreHole owner(in.input);auto view=owner.view();
  std::array<double,41> a{};OpenMXCoreHoleTrace trace{{a.data(),a.data(),a.data()},a.data(),41,1,0};
  std::array<dcomplex,9> sums;for(int i=0;i<9;i++)sums[i]={double(i),-double(i)};const auto saved=sums;
  char error[256];
  const auto unchanged=[&]{for(int i=0;i<9;i++)require(sums[i].r==saved[i].r && sums[i].i==saved[i].i,"invalid trace partially changed output");};
  for(int mode=0;mode<6;mode++) {
    auto input=trace;auto hole=view;
    if(mode==0)input.columns=2;
    if(mode==1)input.derivative[2]=nullptr;
    if(mode==2)input.sign=std::numeric_limits<double>::quiet_NaN();
    if(mode==3)input.reverse_penalty=2;
    if(mode==4)hole.opposite_weights=nullptr,input.reverse_penalty=1;
    if(mode==5)hole.angular=4;
    require(!openmx_core_core_hole_trace(&hole,&input,sums.data(),error,sizeof(error)) && error[0],"invalid trace accepted");unchanged();
  }
  for(int mode=0;mode<5;mode++) {
    Input bad;
    if(mode==0)bad.input.state=-1;
    if(mode==1)bad.input.orbitals=40;
    if(mode==2)bad.basis[0]=-1;
    if(mode==3)bad.input.multiplicity=nullptr;
    if(mode==4)bad.input.penalty=std::numeric_limits<double>::infinity();
    bool threw=false;try{CoreHole rejected(bad.input);}catch(const std::exception&){threw=true;}
    require(threw,"invalid owner configuration accepted");
  }
}
int lifetime_callback(OpenMXSolverInvocation *scope,void *opaque,double *result)
{
  snapshot(scope,nullptr,result);const int mode=*static_cast<int*>(opaque);
  if(mode==1)return 0;if(mode==2)throw std::runtime_error("injected core-hole exception");return 1;
}
void lifetime()
{
  for(int mode=0;mode<3;mode++) {
    const auto live=openmx::test::live_allocations();char error[256];double value=31;
    const int ok=openmx_core_solver_run(2,lifetime_callback,&mode,&value,error,sizeof(error));
    require(ok==(mode==0) && openmx::test::live_allocations()==live,"core-hole invocation leaked");
    near(value,mode==0?19:31);
  }
}
int allocate(OpenMXSolverInvocation *scope,void*,double *result)
{
  Input in;OpenMXCoreHoleView view;char error[256];
  for(int i=0;i<3;i++)if(!openmx_core_core_hole_prepare(scope,&in.input,&view,error,sizeof(error)))return 0;
  *result=19;return 1;
}
void allocation()
{
  char error[256];double result=31;openmx::test::fail_allocation_after(10000);
  const int ok=openmx_core_solver_run(2,allocate,nullptr,&result,error,sizeof(error));
  const long count=10000-openmx::test::remaining_allocations();openmx::test::fail_allocation_after(-1);
  require(ok && count>5,"allocation interception failed");
  for(long i=0;i<count;i++) {
    const auto live=openmx::test::live_allocations();result=31;openmx::test::fail_allocation_after(i);
    const int passed=openmx_core_solver_run(2,allocate,nullptr,&result,error,sizeof(error));openmx::test::fail_allocation_after(-1);
    require(!passed && openmx::test::live_allocations()==live,"partial core-hole preparation leaked");near(result,31);
  }
}
void concurrent()
{
  Input input;CoreHole owner(input.input);const auto hole=owner.view();std::array<int,4> done{};std::vector<std::thread> threads;
  for(int worker=0;worker<4;worker++)threads.emplace_back([&,worker]{try {
    std::array<double,41> row;for(int i=0;i<41;i++)row[i]=std::sin(i*.1+worker*.3);
    OpenMXCoreHoleTrace trace{{row.data(),row.data(),row.data()},row.data(),41,1,worker%2};
    std::array<dcomplex,9> a{},b{};core_hole_trace(hole,trace,a);
    for(int i=0;i<20;i++)core_hole_trace(hole,trace,b);
    for(int i=0;i<9;i++){near(b[i].r,20*a[i].r);near(b[i].i,20*a[i].i);}done[worker]=1;
  }catch(...){done[worker]=-1;}});
  for(auto& t:threads)t.join();for(auto status:done)require(status==1,"shared core-hole coefficients raced");
}
}
int main(int argc,char **argv)
{
  try {
    const std::string name=argc==2?argv[1]:"";
    if(name=="coefficients")coefficients();else if(name=="invalid")invalid();else if(name=="lifetime")lifetime();
    else if(name=="allocation")allocation();else if(name=="concurrent")concurrent();else if(name=="snapshot") {
      char error[256];double result;require(openmx_core_solver_run(1,snapshot,nullptr,&result,error,sizeof(error)),error);near(result,19);
    }else throw std::runtime_error("unknown core-hole test");
    std::cout<<"PASS core-hole "<<name<<'\n';return 0;
  }catch(const std::exception& e){openmx::test::fail_allocation_after(-1);std::cerr<<e.what()<<'\n';return 1;}
}

#include "openmx_core_angular_grid.h"
#include "allocation_faults.hpp"
#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <vector>
namespace {
void require(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
void rule(const OpenMXLebedevRuleData& view)
{
  require(view.xyzw && (view.count==590 || view.count==1202 || view.count==2030),"invalid rule shape");
  double sum=0,moment=0;
  for(std::size_t i=0;i<view.count;++i) {
    const auto* p=openmx_lebedev_point(&view,i);
    require(p==view.xyzw+4*i && p[3]>0,"invalid flat point or weight");
    require(std::abs(p[0]*p[0]+p[1]*p[1]+p[2]*p[2]-1)<3e-12,"point not on sphere");
    sum+=p[3];moment+=p[3]*p[0]*p[0];
  }
  require(std::abs(sum-1)<3e-12 && std::abs(moment-1./3)<3e-12,"rule not initialized");
}
struct Request {std::string_view action;int exit_mode=0;double seed=71;};
int callback(OpenMXSolverInvocation* scope,void* opaque,double* result)
{
  const auto r=*static_cast<Request*>(opaque);char error[160];OpenMXLebedevRuleData view{};
  if(r.action=="initial-failures") {
    const auto live=openmx::test::live_allocations();int failures=0;bool complete=false;
    for(int n=0;n<12;++n) {
      openmx::test::fail_allocation_after(n);
      const int ok=openmx_core_solver_lebedev_rule(scope,590,&view,error,sizeof error);
      openmx::test::fail_allocation_after(-1);
      if(ok){complete=true;rule(view);break;}
      ++failures;require(error[0] && !view.xyzw && !view.count && openmx::test::live_allocations()==live,"failed initial rule leaked or changed output");
    }
    require(complete && failures>=2,"initial owner/data fault sweep incomplete");
    *result=r.seed;return 1;
  }
  const int initial=r.action=="failures"?590:2030;
  require(openmx_core_solver_lebedev_rule(scope,initial,&view,error,sizeof error),error);rule(view);
  const auto saved=view;const std::vector<double> values(view.xyzw,view.xyzw+4*view.count);
  auto unchanged=[&] {
    return view.xyzw==saved.xyzw && view.count==saved.count &&
        std::equal(values.begin(),values.end(),view.xyzw);
  };
  if(r.action=="reuse" || r.action=="layout") {
    for(int points:{590,1202,590,2030,1202,2030}) {
      openmx::test::fail_allocation_after(0);
      const int ok=openmx_core_solver_lebedev_rule(scope,points,&view,error,sizeof error);
      const bool no_allocation=openmx::test::remaining_allocations()==0;openmx::test::fail_allocation_after(-1);
      require(ok && no_allocation && view.xyzw==saved.xyzw && view.count==std::size_t(points),"capacity reuse allocated or moved");rule(view);
    }
  }else if(r.action=="invalid") {
    for(int points:{0,-1,591,INT_MAX}) {
      require(!openmx_core_solver_lebedev_rule(scope,points,&view,error,sizeof error) && error[0],"unsupported grid accepted");
      require(unchanged(),"invalid points damaged view/storage");
    }
    require(!openmx_core_solver_lebedev_rule(nullptr,590,&view,error,sizeof error),"null scope accepted");
    require(!openmx_core_solver_lebedev_rule(scope,590,nullptr,error,sizeof error),"null output accepted");
    require(unchanged(),"invalid outputs damaged existing rule");
    char tiny='x';require(!openmx_core_solver_lebedev_rule(scope,591,&view,&tiny,1) && tiny=='\0',"tiny diagnostic not terminated");
    require(!openmx_core_solver_lebedev_rule(scope,591,&view,nullptr,0),"diagnostic required for failure");
  }else if(r.action=="failures") {
    const auto live=openmx::test::live_allocations();int failures=0;bool complete=false;
    for(int n=0;n<12;++n) {
      openmx::test::fail_allocation_after(n);
      const int ok=openmx_core_solver_lebedev_rule(scope,2030,&view,error,sizeof error);
      openmx::test::fail_allocation_after(-1);
      if(ok){complete=true;require(view.count==2030,"grown shape differs");rule(view);break;}
      ++failures;require(error[0] && unchanged() && openmx::test::live_allocations()==live,"failed growth damaged rule or leaked");
    }
    require(complete && failures>0,"growth fault sweep incomplete");
  }
  if(r.exit_mode==1)return 0;
  if(r.exit_mode==2)throw std::runtime_error("intentional angular callback exception");
  rule(view);*result=r.seed;return 1;
}
void run(std::string_view action,int mode=0)
{
  const auto live=openmx::test::live_allocations();Request request{action,mode};double result=-9;char error[160];
  const int ok=openmx_core_solver_run(1,callback,&request,&result,error,sizeof error);
  require((ok==1)==(mode==0) && result==(ok?request.seed:-9),"callback exit/result changed");
  require(openmx::test::live_allocations()==live,"rule survived invocation");
}
void concurrent()
{
  const auto live=openmx::test::live_allocations();std::array<std::thread,4> workers;std::array<int,4> ok{};
  for(int t=0;t<4;++t)workers[t]=std::thread([&,t]{
    try {for(int i=0;i<20;++i){Request request{"lifetime",0,100*t+i+.5};double result;char error[160];
      require(openmx_core_solver_run(1,callback,&request,&result,error,sizeof error) && result==request.seed,error);}ok[t]=1;}catch(...){}
  });
  for(auto& worker:workers)worker.join();for(int value:ok)require(value==1,"rule invocations interfered");
  require(openmx::test::live_allocations()==live,"concurrent owners leaked");
}
}
int main(int argc,char** argv)
{
  try {
    require(argc==2,"expected angular test name");const std::string_view action=argv[1];
    if(action=="concurrent")concurrent();
    else if(action=="lifetime")for(int mode=0;mode<3;++mode)for(int i=0;i<20;++i)run(action,mode);
    else if(action=="layout" || action=="reuse" || action=="invalid")run(action);
    else if(action=="failures"){run("initial-failures");run(action);}
    else throw std::runtime_error("unknown angular test");
  }catch(const std::exception& e){openmx::test::fail_allocation_after(-1);std::cerr<<e.what()<<'\n';return 1;}
}

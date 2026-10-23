#include <openmx/core/angular_grid.hpp>
#include "allocation_faults.hpp"
#include <array>
#include <climits>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <thread>
using openmx::core::AngularGrid;
namespace {
void require(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
void layout()
{
  for(int count:{590,1202,2030}) {
    AngularGrid owner(count);auto span=owner.grid();auto view=owner.view();auto** rows=owner.legacy_rows();
    require(span.extent(0)==std::size_t(count)&&span.extent(1)==4&&view.extent[0]==std::size_t(count)&&
        view.extent[1]==4&&view.extent[2]==1&&view.extent[3]==1,"angular shape lost");
    for(int i=0;i<count;++i)for(int axis=0;axis<4;++axis){
      auto& value=openmx::core::element(span,i,axis);require(&value==rows[i]+axis&& &value==view.values+4*i+axis,"angular row mapping differs");
      value=.25+4*i+axis;require(rows[i][axis]==value,"borrowed row detached from mdspan");
    }
  }
}
struct Request {std::string_view action;int exit_mode=0;double seed=71;};
int callback(OpenMXSolverInvocation* scope,void* opaque,double* result)
{
  const auto r=*static_cast<Request*>(opaque);char error[160];OpenMXSolverRealTensor view{};double** rows=nullptr;
  const int initial=r.action=="failures"?590:2030;
  require(openmx_core_solver_angular_grid(scope,initial,&view,&rows,error,sizeof(error)),error);
  for(int i=0;i<initial*4;++i)view.values[i]=r.seed+i;
  const auto saved=view;auto** saved_rows=rows;
  if(r.action=="reuse") {
    for(int points:{590,1202,590,2030,1202,2030}) {
      openmx::test::fail_allocation_after(0);
      const int ok=openmx_core_solver_angular_grid(scope,points,&view,&rows,error,sizeof(error));
      const bool no_allocation=openmx::test::remaining_allocations()==0;openmx::test::fail_allocation_after(-1);
      require(ok&&no_allocation&&view.values==saved.values&&rows==saved_rows,"capacity rebind allocated or moved");
      require(view.extent[0]==std::size_t(points)&&view.extent[1]==4,"rebind lost logical shape");
      for(int i=0;i<points;++i){require(rows[i]==view.values+4*i,"rebind lost row pointer");require(rows[i][3]==r.seed+4*i+3,"rebind reset numeric values");}
    }
  }else if(r.action=="invalid") {
    for(int points:{0,-1,591,INT_MAX}) {
      require(!openmx_core_solver_angular_grid(scope,points,&view,&rows,error,sizeof(error))&&error[0],"unsupported grid accepted");
      require(rows==saved_rows&&view.values==saved.values&&view.extent[0]==saved.extent[0]&&view.values[0]==r.seed,"invalid points damaged view/storage");
    }
    require(!openmx_core_solver_angular_grid(nullptr,590,&view,&rows,error,sizeof(error)),"null scope accepted");
    require(!openmx_core_solver_angular_grid(scope,590,nullptr,&rows,error,sizeof(error)),"null tensor accepted");
    require(!openmx_core_solver_angular_grid(scope,590,&view,nullptr,error,sizeof(error)),"null rows output accepted");
    require(rows==saved_rows&&view.values==saved.values&&view.values[0]==r.seed,"invalid outputs damaged existing grid");
    char tiny='x';require(!openmx_core_solver_angular_grid(scope,591,&view,&rows,&tiny,1)&&tiny=='\0',"tiny diagnostic not terminated");
    require(!openmx_core_solver_angular_grid(scope,591,&view,&rows,nullptr,0),"diagnostic buffer required for failure");
  }else if(r.action=="failures") {
    const auto live=openmx::test::live_allocations();int failures=0;
    for(int n=0;n<16;++n) {
      openmx::test::fail_allocation_after(n);
      const int ok=openmx_core_solver_angular_grid(scope,2030,&view,&rows,error,sizeof(error));
      openmx::test::fail_allocation_after(-1);
      if(ok){require(failures>=3,"failure sweep missed numeric/metadata construction");require(view.extent[0]==2030&&rows[2029]==view.values+4*2029,"grown shape differs");break;}
      ++failures;require(error[0]&&view.values==saved.values&&rows==saved_rows&&view.extent[0]==590&&
          view.values[0]==r.seed&&rows[589][3]==r.seed+4*589+3&&openmx::test::live_allocations()==live,"failed growth damaged output/storage or leaked");
      require(n<15,"every growth request failed");
    }
  }
  if(r.exit_mode==1)return 0;
  if(r.exit_mode==2)throw std::runtime_error("intentional angular callback exception");
  require(r.action=="failures"||view.values[0]==r.seed,"independent grid was overwritten");
  *result=r.seed;return 1;
}
void run(std::string_view action,int mode=0)
{
  const auto live=openmx::test::live_allocations();Request request{action,mode};double result=-9;char error[160];
  const int ok=openmx_core_solver_run(1,callback,&request,&result,error,sizeof(error));
  require((ok==1)==(mode==0)&&result==(ok?request.seed:-9),"callback exit/result changed");
  require(openmx::test::live_allocations()==live,"angular owner survived invocation");
}
void concurrent()
{
  const auto live=openmx::test::live_allocations();std::array<std::thread,4> workers;std::array<int,4> ok{};
  for(int t=0;t<4;++t)workers[t]=std::thread([&,t]{
    try {for(int i=0;i<20;++i){Request request{"layout",0,100*t+i+.5};double result;char error[160];
      require(openmx_core_solver_run(1,callback,&request,&result,error,sizeof(error))&&result==request.seed,error);}ok[t]=1;}catch(...){}
  });
  for(auto& worker:workers)worker.join();for(int value:ok)require(value==1,"angular invocations interfered");
  require(openmx::test::live_allocations()==live,"concurrent owners leaked");
}
}
int main(int argc,char** argv)
{
  try {
    require(argc==2,"expected angular test name");const std::string_view action=argv[1];
    if(action=="layout")layout();else if(action=="concurrent")concurrent();
    else if(action=="lifetime")for(int mode=0;mode<3;++mode)for(int i=0;i<20;++i)run(action,mode);
    else if(action=="reuse"||action=="invalid"||action=="failures")run(action);
    else throw std::runtime_error("unknown angular test");
  }catch(const std::exception& e){openmx::test::fail_allocation_after(-1);std::cerr<<e.what()<<'\n';return 1;}
}

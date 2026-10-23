#include <openmx/core/dispersion_workspace.hpp>
#include "allocation_faults.hpp"
#include <array>
#include <climits>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <thread>
using openmx::core::DispersionWorkspace;
namespace {
void require(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
std::size_t count(const OpenMXDispersionView& v)
{return v.atoms*v.atoms*v.images[0]*v.images[1]*v.images[2];}
void seed(OpenMXDispersionView v)
{
  for(std::size_t a=0;a<v.atoms;++a)v.coordination[a]=7+a;
  for(std::size_t i=0;i<v.atoms*v.atoms;++i){v.c6_derivative[i]=11+i;v.energy_derivative[i]=13+i;}
  for(std::size_t i=0;i<count(v);++i)v.coordination_derivative[i]=17+i;
}
void zero(OpenMXDispersionView v)
{
  for(std::size_t a=0;a<v.atoms;++a)require(v.coordination[a]==0,"coordination was not reset");
  for(std::size_t i=0;i<v.atoms*v.atoms;++i)
    require(v.c6_derivative[i]==0 && v.energy_derivative[i]==0,"pair buffers were not reset");
  for(std::size_t i=0;i<count(v);++i)require(v.coordination_derivative[i]==0,"image buffer was not reset");
}
void layout()
{
  for(int atoms:{0,1,4})for(const auto radii:{std::array{0,0,0},std::array{1,0,2},std::array{0,2,1}}) {
    DispersionWorkspace owner({atoms,{radii[0],radii[1],radii[2]}});auto v=owner.view();zero(v);
    require(v.atoms==std::size_t(atoms+1),"atom padding lost");
    using Tensor=openmx::core::Tensor<double,5>;
    Tensor::view_type md(v.coordination_derivative,v.atoms,v.atoms,v.images[0],v.images[1],v.images[2]);
    std::size_t offset=0;
    for(std::size_t a=0;a<v.atoms;++a)for(std::size_t b=0;b<v.atoms;++b) {
      require(openmx_dispersion_c6(&v,a,b)==v.c6_derivative+a*v.atoms+b,"C6 row differs");
      require(openmx_dispersion_energy(&v,a,b)==v.energy_derivative+a*v.atoms+b,"energy row differs");
      for(std::size_t i=0;i<v.images[0];++i)for(std::size_t j=0;j<v.images[1];++j)for(std::size_t k=0;k<v.images[2];++k) {
        auto* cell=openmx_dispersion_coordination(&v,a,b,i,j,k);
        require(cell==v.coordination_derivative+offset++ && cell==&openmx::core::element(md,a,b,i,j,k),"five-dimensional mapping differs");
        *cell=double(offset);
      }
    }
    seed(v);owner.clear();zero(v);
  }
}
struct Request {std::string_view action;int exit_mode=0;};
int callback(OpenMXSolverInvocation* scope,void* opaque,double* result)
{
  const auto& request=*static_cast<Request*>(opaque);char error[160];
  const OpenMXDispersionShape small{2,{1,0,2}},large{4,{2,1,0}};
  OpenMXDispersionView view{};
  require(openmx_core_dispersion_prepare(scope,&small,&view,error,sizeof(error)),error);seed(view);
  const auto saved=view;
  if(request.action=="reuse") {
    openmx::test::fail_allocation_after(0);
    const int ok=openmx_core_dispersion_prepare(scope,&small,&view,error,sizeof(error));
    const bool unallocated=openmx::test::remaining_allocations()==0;
    openmx::test::fail_allocation_after(-1);
    require(ok && unallocated && view.coordination==saved.coordination &&
        view.coordination_derivative==saved.coordination_derivative,"same shape allocated or moved");zero(view);
    require(openmx_core_dispersion_prepare(scope,&large,&view,error,sizeof(error)),error);zero(view);
    require(view.atoms==5 && view.images[0]==5 && view.images[1]==3 && view.images[2]==1,"shape change lost axes");
  }else if(request.action=="invalid" || request.action=="overflow") {
    for(int mode=0;mode<9;++mode) {
      auto shape=small;if(mode==0)shape.atoms=-1;if(mode==1)shape.atoms=INT_MAX;
      if(mode==2)shape.image_radius[1]=-1;if(mode==3)shape.image_radius[2]=INT_MAX;
      if(mode==4)shape={INT_MAX-1,{(INT_MAX-1)/2,(INT_MAX-1)/2,(INT_MAX-1)/2}};
      int ok=openmx_core_dispersion_prepare(mode==5?nullptr:scope,mode==6?nullptr:&shape,
          mode==7?nullptr:&view,mode==8?nullptr:error,mode==8?0:sizeof(error));
      // Mode eight repeats a valid request without a diagnostic buffer.
      if(mode==8){require(ok,"optional error buffer required");zero(view);continue;}
      require(!ok && error[0] && view.coordination==saved.coordination && view.coordination[1]==8 &&
          view.coordination_derivative[count(view)-1]==16+count(view),"invalid prepare damaged live storage/output");
    }
  }else if(request.action=="failures") {
    const auto allocations=openmx::test::live_allocations();
    int failed=0;
    for(int n=0;n<16;++n) {
      openmx::test::fail_allocation_after(n);
      const int ok=openmx_core_dispersion_prepare(scope,&large,&view,error,sizeof(error));
      openmx::test::fail_allocation_after(-1);
      if(ok){require(failed>=5,"allocation failures did not reach each owned tensor");zero(view);break;}
      ++failed;require(error[0] && view.coordination==saved.coordination && view.coordination[1]==8 &&
          view.coordination_derivative[count(view)-1]==16+count(view) &&
          openmx::test::live_allocations()==allocations,"partial construction leaked or damaged prior view");
      require(n<15,"all replacements failed");
    }
  }
  if(request.exit_mode==1)return 0;
  if(request.exit_mode==2)throw std::runtime_error("intentional callback exception");
  *result=41;return 1;
}
void invocation(std::string_view action,int exit_mode=0)
{
  const auto allocations=openmx::test::live_allocations();Request request{action,exit_mode};
  double result=-7;char error[160];
  const int ok=openmx_core_solver_run(2,callback,&request,&result,error,sizeof(error));
  require((ok==1)==(exit_mode==0),"invocation exit status differs");
  require(result==(ok?41:-7),"failed callback published a result");
  require(openmx::test::live_allocations()==allocations,"invocation retained dispersion storage");
}
void concurrent()
{
  std::array<int,4> ok{};std::array<std::thread,4> threads;
  // Cross-thread allocations make the per-invocation live-count assertion
  // inappropriate here; only compare the complete joined lifetime.
  const auto allocations=openmx::test::live_allocations();
  for(int t=0;t<4;++t)threads[t]=std::thread([&,t]{
    try {for(int i=0;i<20;++i){Request r{"layout"};double result;char error[160];
      require(openmx_core_solver_run(1,callback,&r,&result,error,sizeof(error)) && result==41,error);}ok[t]=1;}catch(...){}
  });
  for(auto& thread:threads)thread.join();for(int value:ok)require(value==1,"concurrent invocation failed");
  require(openmx::test::live_allocations()==allocations,"joined owners leaked");
}
}
int main(int argc,char** argv)
{
  try {
    require(argc==2,"expected dispersion test name");const std::string_view action=argv[1];
    if(action=="layout")layout();else if(action=="concurrent")concurrent();
    else if(action=="lifetime")for(int mode=0;mode<3;++mode)for(int i=0;i<20;++i)invocation(action,mode);
    else if(action=="reuse" || action=="invalid" || action=="overflow" || action=="failures")invocation(action);
    else throw std::runtime_error("unknown dispersion test");
  }catch(const std::exception& e){openmx::test::fail_allocation_after(-1);std::cerr<<e.what()<<'\n';return 1;}
}

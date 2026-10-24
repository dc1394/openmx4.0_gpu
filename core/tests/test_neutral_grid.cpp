#include "openmx_core_neutral_grid.h"
#include "allocation_faults.hpp"
#include <array>
#include <climits>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <thread>

namespace {
void require(bool condition,const char* message){if(!condition)throw std::runtime_error(message);}
struct Context {
  OpenMXCoreContext* value;
  Context(){char error[256];value=openmx_core_create_host(error,sizeof(error));require(value,error);}
  ~Context(){openmx_core_destroy(value);}
  Context(const Context&)=delete;Context& operator=(const Context&)=delete;
};
OpenMXNeutralGridTables prepare(Context& context,int species,int capacity){
  const OpenMXNeutralGridShape shape{species,capacity};OpenMXNeutralGridTables view{};char error[256];
  require(openmx_core_neutral_grid_prepare(context.value,&shape,&view,error,sizeof(error)),error);
  return view;
}
void fill(const OpenMXNeutralGridTables& view,double seed){
  require(view.extent[0]==5&&view.extent[1]>0&&view.extent[2]>0&&view.values&&view.legacy_rows,"missing grid shape/storage");
  for(std::size_t component=0;component<5;++component)for(std::size_t species=0;species<view.extent[1];++species){
    auto* row=view.legacy_rows[component*view.extent[1]+species];
    require(row==view.values+(component*view.extent[1]+species)*view.extent[2],"row not mapped over flat component/species/point storage");
    for(std::size_t point=0;point<view.extent[2];++point)row[point]=seed+10000*component+100*species+point;
  }
}
void check(const OpenMXNeutralGridTables& view,double seed){
  for(std::size_t c=0;c<5;++c)for(std::size_t s=0;s<view.extent[1];++s)for(std::size_t p=0;p<view.extent[2];++p)
    require(view.values[(c*view.extent[1]+s)*view.extent[2]+p]==seed+10000*c+100*s+p,"flat values and borrowed rows disagree");
}
void layout(){
  Context context;
  for(int species:{1,3,2})for(int capacity:{1,17,5}){
    openmx::test::fill_allocations_with(0xa5);auto view=prepare(context,species,capacity);openmx::test::fill_allocations_with(-1);
    const auto* bytes=reinterpret_cast<const unsigned char*>(view.values);
    for(std::size_t i=0;i<5*view.extent[1]*view.extent[2]*sizeof(double);++i)require(bytes[i]==0xa5,"numeric allocation eagerly initialized");
    fill(view,.125);check(view,.125);
  }
}
void replace(){
  Context a,b;char error[256];
  auto other=prepare(b,4,19);fill(other,71);
  for(int n:{3,29,7,29,1}){
    auto view=prepare(a,n==29?3:2,n);fill(view,n+.25);check(view,n+.25);check(other,71);
    OpenMXNeutralGridTables fetched{};require(openmx_core_neutral_grid_get(a.value,&fetched,error,sizeof(error)),error);
    require(fetched.values==view.values&&fetched.legacy_rows==view.legacy_rows&&fetched.extent[1]==view.extent[1]&&fetched.extent[2]==view.extent[2],"get detached current owner");
  }
}
void invalid(){
  Context context;auto view=prepare(context,2,13);fill(view,17);const auto saved=view;char error[256];
  for(const auto shape:std::array<OpenMXNeutralGridShape,5>{{{0,8},{2,0},{-1,8},{2,-1},{INT_MAX,INT_MAX}}}){
    require(!openmx_core_neutral_grid_prepare(context.value,&shape,&view,error,sizeof(error))&&error[0],"invalid/overflow shape accepted");
    require(std::memcmp(&view,&saved,sizeof(view))==0,"invalid request committed output");check(saved,17);
  }
  const OpenMXNeutralGridShape shape{1,3};
  require(!openmx_core_neutral_grid_prepare(nullptr,&shape,&view,error,sizeof(error)),"null context accepted");
  require(!openmx_core_neutral_grid_prepare(context.value,nullptr,&view,error,sizeof(error)),"null shape accepted");
  require(!openmx_core_neutral_grid_prepare(context.value,&shape,nullptr,error,sizeof(error)),"null output accepted");
  require(!openmx_core_neutral_grid_get(nullptr,&view,error,sizeof(error))&&std::memcmp(&view,&saved,sizeof(view))==0,"invalid get changed output");
  require(!openmx_core_neutral_grid_get(context.value,nullptr,error,sizeof(error)),"null get output accepted");
  char tiny='x';const OpenMXNeutralGridShape bad{0,0};
  require(!openmx_core_neutral_grid_prepare(context.value,&bad,&view,&tiny,1)&&tiny=='\0',"tiny diagnostic not terminated");
  require(!openmx_core_neutral_grid_prepare(context.value,&bad,&view,nullptr,0),"diagnostic buffer required");check(saved,17);
}
void failures(){
  Context context;auto view=prepare(context,2,5);fill(view,23);const auto saved=view;const auto live=openmx::test::live_allocations();
  const OpenMXNeutralGridShape shape{3,47};char error[256];int failures=0;bool success=false;
  for(int n=0;n<16;++n){
    openmx::test::fail_allocation_after(n);const int ok=openmx_core_neutral_grid_prepare(context.value,&shape,&view,error,sizeof(error));openmx::test::fail_allocation_after(-1);
    if(ok){success=true;break;}
    ++failures;require(error[0]&&std::memcmp(&view,&saved,sizeof(view))==0,"allocation failure changed output");
    OpenMXNeutralGridTables current{};require(openmx_core_neutral_grid_get(context.value,&current,error,sizeof(error))&&current.values==saved.values&&current.legacy_rows==saved.legacy_rows,"failed prepare replaced old owner");
    check(saved,23);require(openmx::test::live_allocations()==live,"failed prepare leaked");
  }
  require(success&&failures>=3&&view.extent[1]==3&&view.extent[2]==47,"failure sweep missed object/numeric/row allocations");fill(view,31);check(view,31);
}
void lifetime(){
  const auto live=openmx::test::live_allocations();
  for(int i=0;i<16;++i){
    Context context;char error[256];OpenMXNeutralGridTables view{};
    require(openmx_core_neutral_grid_get(context.value,&view,error,sizeof(error))&&!view.values&&!view.legacy_rows,"absent owner not empty");
    for(int repeat=0;repeat<3;++repeat){view=prepare(context,2,11);fill(view,repeat);openmx_core_neutral_grid_release(context.value);openmx_core_neutral_grid_release(context.value);
      require(openmx_core_neutral_grid_get(context.value,&view,error,sizeof(error))&&!view.values&&!view.legacy_rows,"released owner still visible");}
    view=prepare(context,3,7);openmx_core_finish(context.value);openmx_core_finish(context.value);const auto saved=view;
    require(!openmx_core_neutral_grid_get(context.value,&view,error,sizeof(error))&&std::memcmp(&view,&saved,sizeof(view))==0,"finished get modified output");
    const OpenMXNeutralGridShape shape{1,1};require(!openmx_core_neutral_grid_prepare(context.value,&shape,&view,error,sizeof(error)),"finished context accepted prepare");
    openmx_core_neutral_grid_release(context.value);
  }
  require(openmx::test::live_allocations()==live,"owner/context survived lifetime boundaries");openmx_core_neutral_grid_release(nullptr);
}
void concurrent(){
  const auto live=openmx::test::live_allocations();std::array<std::thread,4> workers;std::array<int,4> ok{};
  for(int t=0;t<4;++t)workers[t]=std::thread([&,t]{try{Context context;for(int i=0;i<12;++i){auto view=prepare(context,t+1,3+i);fill(view,t*100+i);check(view,t*100+i);}ok[t]=1;}catch(...){}});
  for(auto& worker:workers)worker.join();for(int result:ok)require(result==1,"independent contexts interfered");
  require(openmx::test::live_allocations()==live,"concurrent contexts leaked");
}
}
int main(int argc,char** argv){
  try{require(argc==2,"expected neutral-grid test case");const std::string_view name=argv[1];
    if(name=="layout")layout();else if(name=="replace")replace();else if(name=="invalid")invalid();else if(name=="failures")failures();else if(name=="lifetime")lifetime();else if(name=="concurrent")concurrent();else throw std::runtime_error("unknown neutral-grid case");
  }catch(const std::exception& error){openmx::test::fail_allocation_after(-1);openmx::test::fill_allocations_with(-1);std::cerr<<error.what()<<'\n';return 1;}
}

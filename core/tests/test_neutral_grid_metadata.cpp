#include "openmx_core_neutral_grid_metadata.h"
#include "openmx_core_neutral_grid.h"
#include "allocation_faults.hpp"
#include <array>
#include <climits>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <utility>

namespace {
void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
struct Context {
  OpenMXCoreContext* value;
  Context(){char error[256];value=openmx_core_create_host(error,sizeof(error));require(value,error);}
  ~Context(){openmx_core_destroy(value);}
  Context(const Context&)=delete;Context& operator=(const Context&)=delete;
};
OpenMXNeutralGridMetadataView prepare(Context& context,std::size_t species){
  const OpenMXNeutralGridMetadataShape shape{species};OpenMXNeutralGridMetadataView view{};char error[256];
  require(openmx_core_neutral_grid_metadata_prepare(context.value,&shape,&view,error,sizeof(error)),error);return view;
}
void fill(const OpenMXNeutralGridMetadataView& view,int seed){
  require(view.counts&&view.spacing&&view.species,"missing metadata storage");
  for(std::size_t s=0;s<view.species;++s){view.counts[s]=seed+int(s)*31;view.spacing[s]=seed*.125+s*.0625;}
}
void check(const OpenMXNeutralGridMetadataView& view,int seed){
  for(std::size_t s=0;s<view.species;++s)require(view.counts[s]==seed+int(s)*31&&view.spacing[s]==seed*.125+s*.0625,"metadata overwritten or wrongly projected");
}
void layout(){
  Context context;
  for(std::size_t species:{1,7,3,19}){
    openmx::test::fill_allocations_with(0xa5);auto view=prepare(context,species);openmx::test::fill_allocations_with(-1);
    require(view.species==species,"metadata shape differs");
    for(auto item:{std::pair<const void*,std::size_t>{view.counts,species*sizeof(int)},std::pair<const void*,std::size_t>{view.spacing,species*sizeof(double)}}){
      const auto* bytes=static_cast<const unsigned char*>(item.first);for(std::size_t i=0;i<item.second;++i)require(bytes[i]==0xa5,"metadata eagerly initialized");
    }
    fill(view,11);check(view,11);
    // Native input uses the same double cells for raw AtomSpecies, including
    // fractional and ghost species. Preserve their bits until overwritten.
    const double raw_species[]={.25,.75,-0.0,14.0};
    for(double raw:raw_species){view.spacing[species-1]=raw;require(std::memcmp(&view.spacing[species-1],&raw,sizeof(raw))==0,"raw AtomSpecies lost precision/sign");}
  }
}
void replace(){
  Context a,b;auto other=prepare(b,5);fill(other,83);char error[256];
  const OpenMXNeutralGridShape grid_shape{2,17};OpenMXNeutralGridTables grid{};
  require(openmx_core_neutral_grid_prepare(a.value,&grid_shape,&grid,error,sizeof(error)),error);grid.values[169]=91;
  for(std::size_t species:{2,17,3,17,1}){
    auto view=prepare(a,species);fill(view,13);check(view,13);check(other,83);require(grid.values[169]==91,"metadata prepare invalidated independent grid");
    OpenMXNeutralGridMetadataView current{};require(openmx_core_neutral_grid_metadata_get(a.value,&current,error,sizeof(error)),error);
    require(current.counts==view.counts&&current.spacing==view.spacing&&current.species==species,"get detached metadata owner");
  }
  openmx_core_neutral_grid_metadata_release(a.value);require(grid.values[169]==91,"metadata release invalidated grid");
  auto view=prepare(a,3);fill(view,29);openmx_core_neutral_grid_release(a.value);check(view,29);
}
void invalid(){
  Context context;auto view=prepare(context,4);fill(view,17);const auto saved=view;char error[256];
  for(std::size_t species:{std::size_t(0),std::size_t(INT_MAX)+1,std::numeric_limits<std::size_t>::max()}){
    const OpenMXNeutralGridMetadataShape shape{species};require(!openmx_core_neutral_grid_metadata_prepare(context.value,&shape,&view,error,sizeof(error))&&error[0],"invalid native index range accepted");
    require(std::memcmp(&view,&saved,sizeof(view))==0,"invalid shape committed metadata output");check(saved,17);
  }
  const OpenMXNeutralGridMetadataShape shape{3};
  require(!openmx_core_neutral_grid_metadata_prepare(nullptr,&shape,&view,error,sizeof(error)),"null context accepted");
  require(!openmx_core_neutral_grid_metadata_prepare(context.value,nullptr,&view,error,sizeof(error)),"null shape accepted");
  require(!openmx_core_neutral_grid_metadata_prepare(context.value,&shape,nullptr,error,sizeof(error)),"null output accepted");
  require(!openmx_core_neutral_grid_metadata_get(nullptr,&view,error,sizeof(error))&&std::memcmp(&view,&saved,sizeof(view))==0,"invalid get changed output");
  require(!openmx_core_neutral_grid_metadata_get(context.value,nullptr,error,sizeof(error)),"null get output accepted");
  char tiny='x';const OpenMXNeutralGridMetadataShape bad{0};
  require(!openmx_core_neutral_grid_metadata_prepare(context.value,&bad,&view,&tiny,1)&&tiny=='\0',"tiny diagnostic not terminated");
  require(!openmx_core_neutral_grid_metadata_prepare(context.value,&bad,&view,nullptr,0),"diagnostic buffer required");check(saved,17);
}
void failures(){
  Context context;auto view=prepare(context,3);fill(view,23);const auto saved=view;const auto live=openmx::test::live_allocations();char error[256];
  const OpenMXNeutralGridMetadataShape shape{37};int failures=0;bool success=false;
  for(int n=0;n<16;++n){
    openmx::test::fail_allocation_after(n);const int ok=openmx_core_neutral_grid_metadata_prepare(context.value,&shape,&view,error,sizeof(error));openmx::test::fail_allocation_after(-1);
    if(ok){success=true;break;}++failures;
    require(error[0]&&std::memcmp(&view,&saved,sizeof(view))==0,"failed prepare committed output");
    OpenMXNeutralGridMetadataView current{};require(openmx_core_neutral_grid_metadata_get(context.value,&current,error,sizeof(error))&&current.counts==saved.counts&&current.spacing==saved.spacing,"failed prepare replaced old owner");
    check(saved,23);require(openmx::test::live_allocations()==live,"failed prepare leaked");
  }
  require(success&&failures>=3&&view.species==37,"failure sweep missed owner/count/spacing allocations");fill(view,31);check(view,31);
}
void lifetime(){
  const auto live=openmx::test::live_allocations();
  for(int i=0;i<16;++i){
    Context context;char error[256];OpenMXNeutralGridMetadataView view{};
    require(openmx_core_neutral_grid_metadata_get(context.value,&view,error,sizeof(error))&&!view.counts&&!view.spacing&&!view.species,"absent owner not empty");
    for(int r=0;r<3;++r){view=prepare(context,7);fill(view,r);openmx_core_neutral_grid_metadata_release(context.value);openmx_core_neutral_grid_metadata_release(context.value);
      require(openmx_core_neutral_grid_metadata_get(context.value,&view,error,sizeof(error))&&!view.counts&&!view.spacing&&!view.species,"released metadata still visible");}
    view=prepare(context,5);openmx_core_finish(context.value);openmx_core_finish(context.value);const auto saved=view;
    require(!openmx_core_neutral_grid_metadata_get(context.value,&view,error,sizeof(error))&&std::memcmp(&view,&saved,sizeof(view))==0,"finished get modified output");
    const OpenMXNeutralGridMetadataShape shape{2};require(!openmx_core_neutral_grid_metadata_prepare(context.value,&shape,&view,error,sizeof(error)),"finished context accepted prepare");openmx_core_neutral_grid_metadata_release(context.value);
  }
  require(openmx::test::live_allocations()==live,"owner/context survived lifetime boundaries");openmx_core_neutral_grid_metadata_release(nullptr);
}
void concurrent(){
  const auto live=openmx::test::live_allocations();std::array<std::thread,4> workers;std::array<int,4> ok{};
  for(int t=0;t<4;++t)workers[t]=std::thread([&,t]{try{Context context;for(int i=0;i<16;++i){auto view=prepare(context,t+i+1);fill(view,100*t+i);check(view,100*t+i);}ok[t]=1;}catch(...){}});
  for(auto& worker:workers)worker.join();for(int result:ok)require(result==1,"independent metadata contexts interfered");require(openmx::test::live_allocations()==live,"concurrent metadata contexts leaked");
}
}
int main(int argc,char** argv){
  try{require(argc==2,"expected metadata test case");const std::string_view name=argv[1];
    if(name=="layout")layout();else if(name=="replace")replace();else if(name=="invalid")invalid();else if(name=="failures")failures();else if(name=="lifetime")lifetime();else if(name=="concurrent")concurrent();else throw std::runtime_error("unknown metadata case");
  }catch(const std::exception& error){openmx::test::fail_allocation_after(-1);openmx::test::fill_allocations_with(-1);std::cerr<<error.what()<<'\n';return 1;}
}

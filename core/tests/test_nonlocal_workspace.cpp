#include <openmx/core/nonlocal_workspace.hpp>
#include "openmx_core_nonlocal.h"
#include "allocation_faults.hpp"
#include <array>
#include <algorithm>
#include <climits>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>

namespace {
using namespace openmx::core;
void require(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }
template<class F> void rejects(F&& function) {
  bool caught = false;
  try { function(); } catch (const std::exception&) { caught = true; }
  require(caught,"invalid nonlocal input accepted");
}
using Owner = std::unique_ptr<OpenMXNonlocalWorkspace,decltype(&openmx_core_nonlocal_destroy)>;

void layout_test()
{
  std::array<int,15> angular{};
  for (int i = 0; i < 15; ++i) angular[i] = i;
  const NonlocalSpecies species[]{{angular},{{}},{{angular.data(),2}}};
  NonlocalWorkspace work({14,2,14},species);
  require(work.angular().extent(1)==450 && work.angular().extent(2)==225,"rectangular magnetic padding remains");
  require(work.conversion().extent(1)==29 && work.conversion().extent(2)==29,"conversion extents mismatch");
  const auto& layout=work.projectors();
  require(layout.counts()[0]==15 && layout.counts()[1]==0 && layout.slots()==16,"projector counts/order mismatch");
  require(layout.species_offsets()[0]==0 && layout.species_offsets()[1]==16 &&
          layout.species_offsets()[2]==17 && layout.species_offsets()[3]==20,"empty species lost its dummy slot");
  std::size_t column=0;
  for (int l=0;l<=14;++l) {
    require(layout.offsets()[l+1]==column && layout.angular()[l+1]==l,"projector layout reordered angular channels");
    column += 2*l+1;
  }
  require(column==225,"projector magnetic count mismatch");
  for (std::size_t i=0;i<work.angular().size();++i)
    require(work.angular().data_handle()[i].r==0 && work.angular().data_handle()[i].i==0,"angular buffer not zeroed");
  element(work.angular(),3,449,224)={123,-7};
  require(work.angular().data_handle()[work.angular().size()-1].r==123,"packed mdspan row order mismatch");
  static_assert(!std::is_copy_constructible_v<NonlocalWorkspace>);
  const auto& fixed=work;
  static_assert(std::is_const_v<std::remove_reference_t<decltype(element(fixed.angular(),0,0,0))>>);
}

void ownership_test()
{
  int counts[]{3,0,1}, rows[][4]{{-99,0,2,1},{0,0,0,0},{-99,3,0,0}};
  std::array grid{0.,0.5,1.,2.};
  const OpenMXNonlocalShape shape{3,2,3,3,counts,&rows[0][0],4,4,grid.data(),1};
  OpenMXNonlocalBuffers views{}; char error[256];
  Owner work(openmx_core_nonlocal_create(&shape,&views,error,sizeof(error)),openmx_core_nonlocal_destroy);
  require(work!=nullptr,error);
  require(views.orbital_capacity==32 && views.projector_capacity==9 && views.projector_slots==4,
          "C capacities mismatch");
  require(views.conversion_columns==7,"projector conversion leading dimension mismatch");
  rows[0][1]=14; rows[2][1]=0; counts[0]=0; grid[1]=100;
  require(views.projector_counts[0]==3 && views.projector_angular[1]==0 &&
          views.projector_angular[views.species_offsets[2]+1]==3,"metadata is still borrowed from input");
  const auto* offsets=views.projector_offsets;
  for (int channel=0;channel<4;++channel) {
    std::array<bool,32*9> visited{};
    for (int l=0;l<=3;++l) for (int radial=0;radial<2;++radial) for (int m=0;m<=2*l;++m)
      for (int p=1;p<=3;++p) for (int pm=0;pm<=2*views.projector_angular[p];++pm) {
        const auto index=openmx_core_nonlocal_pair(&views,offsets,l,radial,m,p,pm);
        require(index<visited.size() && !visited[index],"nonlocal pair index alias");
        visited[index]=true;
        views.angular[channel][index]={static_cast<double>(index+1),static_cast<double>(channel)};
      }
    require(std::all_of(visited.begin(),visited.end(),[](bool v){return v;}),"nonlocal pair layout has gaps");
    require(views.angular[channel]==views.angular[0]+channel*32*9,"angular channels overlap");
    require(views.conversion[channel]==views.conversion[0]+channel*7*7,"conversion channels overlap");
  }
  for (int l=0;l<=3;++l) for (int radial=0;radial<2;++radial) for (int p=0;p<4;++p) {
    const auto i=openmx_core_nonlocal_sum(&views,l,radial,p);
    require(views.sums[0][i]==0 && views.sums[1][i]==0,"stale radial sums");
    views.sums[1][i]=static_cast<double>(i+1);
  }
  require(views.sums[1]==views.sums[0]+8*4 && views.nonlocal_radial[1]==views.nonlocal_radial[0]+4*4,
          "radial/spin channels overlap");
  OpenMXPairBesselTable table{};
  require(openmx_core_nonlocal_sample(work.get(),1.,2,&table,error,sizeof(error)),error);
  require(std::abs(table.values[1]-std::sin(.5)/.5)<1e-14,"Bessel grid was borrowed rather than copied");
  const auto* saved=table.values;
  require(openmx_core_nonlocal_sample(work.get(),2.,4,&table,error,sizeof(error)),error);
  require(openmx_core_nonlocal_sample(work.get(),1.,2,&table,error,sizeof(error)),error);
  require(table.values==saved && views.angular[3][0].r==1 && views.sums[1][31]==32,
          "sampling invalidated cached values or work buffers");
  const OpenMXNonlocalShape small{0,1,0,1,views.projector_counts,nullptr,0,0,nullptr,0};
  int zero=0;
  auto empty=small; empty.projector_counts=&zero;
  OpenMXNonlocalBuffers independent{};
  Owner another(openmx_core_nonlocal_create(&empty,&independent,error,sizeof(error)),openmx_core_nonlocal_destroy);
  require(another!=nullptr && independent.angular[0][0].r==0,error);
  require(independent.angular[0]!=views.angular[0],"independent workspaces share storage");
}

void angular_only_test()
{
  const int angular[]{2,0}; const NonlocalSpecies species[]{{angular}};
  NonlocalWorkspace work({1,1,2},species);
  require(work.radial().size()==0 && work.nonlocal_radial().size()==0 && work.sums().size()==0,
          "device angular assembly allocated host radial buffers");
  rejects([&]{work.sample(1.,0);});
  const int count=2; int row[]{0,2,0};
  const OpenMXNonlocalShape shape{1,1,2,1,&count,row,3,0,nullptr,0};
  OpenMXNonlocalBuffers views{};char error[128];
  Owner owner(openmx_core_nonlocal_create(&shape,&views,error,sizeof(error)),openmx_core_nonlocal_destroy);
  require(owner!=nullptr,error);
  require(!views.radial && !views.sums[0] && !views.sums[1] && !views.nonlocal_radial[0] &&
          !views.nonlocal_radial[1],"angular-only C views expose radial storage");
  OpenMXPairBesselTable table{};
  require(!openmx_core_nonlocal_sample(owner.get(),1.,0,&table,error,sizeof(error)) && !table.values,
          "angular-only sample published a table");
}

void failure_test()
{
  const int angles[]{0,2};const NonlocalSpecies species[]{{angles}};
  for (NonlocalShape bad : {NonlocalShape{-1,1,2}, {15,1,2},{2,0,2},{2,INT_MAX,2},{2,1,15},{2,1,-1}})
    rejects([&]{NonlocalWorkspace work(bad,species);});
  rejects([&]{NonlocalWorkspace work({2,1,2},{});});
  for (int bad : {-1,3}) {
    const NonlocalSpecies invalid[]{{{&bad,1}}};
    rejects([&]{NonlocalWorkspace work({2,1,2},invalid);});
  }
  const std::array one{1.};
  rejects([&]{NonlocalWorkspace work({2,1,2},species,one);});
  const std::array nan_grid{0.,std::numeric_limits<double>::quiet_NaN()};
  rejects([&]{NonlocalWorkspace work({2,1,2},species,nan_grid);});
  int count=2, row[]{0,0,2};std::array grid{0.,1.,2.};
  OpenMXNonlocalShape shape{2,1,2,1,&count,row,3,3,grid.data(),0};
  OpenMXNonlocalBuffers views{}; char error[128];
  Owner owner(openmx_core_nonlocal_create(&shape,&views,error,sizeof(error)),openmx_core_nonlocal_destroy);
  require(owner!=nullptr,error);const auto saved=views;
  require(!openmx_core_nonlocal_create(nullptr,&views,error,sizeof(error)) && views.angular[0]==saved.angular[0],
          "null shape changed C views");
  auto bad=shape;bad.species=0;
  require(!openmx_core_nonlocal_create(&bad,&views,error,sizeof(error)),"empty species accepted");
  bad=shape;bad.wavevectors=nullptr;
  require(!openmx_core_nonlocal_create(&bad,&views,error,sizeof(error)),"missing grid accepted");
  bad=shape;bad.projector_angular=nullptr;
  require(!openmx_core_nonlocal_create(&bad,&views,error,sizeof(error)),"missing projector row accepted");
  bad=shape;bad.projector_angular_stride=2;
  require(!openmx_core_nonlocal_create(&bad,&views,error,sizeof(error)),"short input stride accepted");
  bad=shape;bad.projector_angular_stride=std::numeric_limits<std::size_t>::max();
  require(!openmx_core_nonlocal_create(&bad,&views,error,sizeof(error)),"overflowing input stride accepted");
  require(views.angular[0]==saved.angular[0],"invalid flat metadata changed output");
  count=-1;
  require(!openmx_core_nonlocal_create(&shape,&views,error,sizeof(error)),"negative count accepted");
  OpenMXPairBesselTable table{};
  require(openmx_core_nonlocal_sample(owner.get(),1.,2,&table,error,sizeof(error)),error);
  const auto old=table;
  require(!openmx_core_nonlocal_sample(owner.get(),-1.,2,&table,error,sizeof(error)) && table.values==old.values,
          "failed sample changed borrowed output");
  require(!openmx_core_nonlocal_sample(nullptr,1.,2,&table,error,sizeof(error)),"null owner accepted");
  require(!openmx_core_nonlocal_sample(owner.get(),1.,2,nullptr,error,sizeof(error)),"null table accepted");
  openmx_core_nonlocal_destroy(nullptr);
}

void allocation_test()
{
  int counts[]{3,1}, rows[][4]{{0,0,2,1},{0,1,0,0}};
  const std::array grid{0.,1.,2.};
  for (int points : {0,3}) {
    const OpenMXNonlocalShape shape{2,2,2,2,counts,&rows[0][0],4,points,points?grid.data():nullptr,2};
    OpenMXNonlocalBuffers views{};char error[256];
    openmx::test::fail_allocation_after(1000);
    auto* retained=openmx_core_nonlocal_create(&shape,&views,error,sizeof(error));
    const auto allocations=1000-openmx::test::remaining_allocations();
    openmx::test::fail_allocation_after(-1);
    require(retained && allocations>=8,"allocation interception inactive");
    const auto saved=views;views.angular[0][0]={19,-1};
    for (long fail=0;fail<allocations;++fail) {
      const auto before=openmx::test::live_allocations();
      openmx::test::fail_allocation_after(fail);
      auto* failed=openmx_core_nonlocal_create(&shape,&views,error,sizeof(error));
      openmx::test::fail_allocation_after(-1);
      if (failed) openmx_core_nonlocal_destroy(failed);
      require(!failed && error[0],"allocation failure escaped C ABI");
      require(openmx::test::live_allocations()==before,"partial nonlocal workspace leaked");
      require(views.angular[0]==saved.angular[0] && views.projector_angular==saved.projector_angular &&
              views.angular[0][0].r==19,"failed allocation published incomplete views");
    }
    openmx_core_nonlocal_destroy(retained);
  }
}
} // namespace

int main(int argc,char** argv)
{
  try {
    const std::string name=argc==2 ? argv[1] : "";
    if(name=="layout")layout_test();
    else if(name=="ownership")ownership_test();
    else if(name=="angular-only")angular_only_test();
    else if(name=="failure")failure_test();
    else if(name=="allocation")allocation_test();
    else throw std::runtime_error("unknown nonlocal workspace test");
    std::cout<<"PASS nonlocal workspace "<<name<<'\n';return 0;
  }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}

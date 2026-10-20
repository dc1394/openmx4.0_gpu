#include <openmx/core/orbital_workspace.hpp>
#include "openmx_core_orbital_workspace.h"
#include "allocation_faults.hpp"

#include <array>
#include <climits>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <type_traits>

namespace {
using namespace openmx::core;
void require(bool value, const char *message) { if (!value) throw std::runtime_error(message); }
using Owner=std::unique_ptr<OpenMXOrbitalWorkspace,decltype(&openmx_core_orbital_destroy)>;
Owner create(OpenMXOrbitalShape shape, OpenMXOrbitalBuffers& views) {
  char error[256];
  Owner owner(openmx_core_orbital_create(&shape,&views,error,sizeof(error)),openmx_core_orbital_destroy);
  require(bool(owner),error);
  return owner;
}
void layout_test() {
  static_assert(!std::is_copy_constructible_v<OrbitalWorkspace>);
  static_assert(!std::is_move_constructible_v<OrbitalWorkspace>);
  for (int l : {0,1,4,14}) for (int m : {1,3,7}) for (int derivatives : {0,1}) {
    OrbitalWorkspace work({l,m,derivatives!=0});
    const auto r=work.radial(),a=work.angular();
    require(r.extent(0)==std::size_t(derivatives ? 2 : 1) && r.extent(1)==std::size_t(l+1) &&
            r.extent(2)==std::size_t(m),"wrong radial dimensions");
    require(a.extent(0)==std::size_t(derivatives ? 3 : 1) && a.extent(1)==std::size_t(l+1) &&
            a.extent(2)==std::size_t(2*(l+1)+1),"wrong angular padding");
    for (std::size_t i=0;i<r.size();++i) require(r.data_handle()[i]==0,"radial not initialized");
    for (std::size_t i=0;i<a.size();++i) require(a.data_handle()[i]==0,"angular not initialized");
    element(r,r.extent(0)-1,l,m-1)=31;
    element(a,a.extent(0)-1,l,a.extent(2)-1)=47;
    require(r.data_handle()[r.size()-1]==31 && a.data_handle()[a.size()-1]==47,"mdspan mapping changed");
    const auto& fixed=work;
    static_assert(std::is_const_v<std::remove_reference_t<decltype(element(fixed.radial(),0,0,0))>>);
  }
}
void c_api_test() {
  for (int d : {0,1}) {
    OpenMXOrbitalBuffers b{};auto owner=create({4,3,d},b);
    require(b.angular_channels==5 && b.multiplicity==3 && b.magnetic_capacity==11,"wrong C capacities");
    require(b.RF && b.AF && bool(b.dRF)==bool(d) && bool(b.dAFQ)==bool(d) && bool(b.dAFP)==bool(d),"wrong channel presence");
    if (d) require(b.dRF==b.RF+15 && b.dAFQ==b.AF+55 && b.dAFP==b.AF+110,"channels overlap");
    for (std::size_t i=0;i<15;++i) b.RF[i]=double(i+1);
    for (std::size_t i=0;i<55;++i) b.AF[i]=double(2*i+1);
    if (d) {
      for (std::size_t i=0;i<15;++i) require(b.dRF[i]==0,"value writes overwrote radial derivative");
      for (std::size_t i=0;i<55;++i) require(b.dAFQ[i]==0 && b.dAFP[i]==0,"value writes overwrote angular derivative");
    }
  }
  openmx_core_orbital_destroy(nullptr);
}
void independent_test() {
  OpenMXOrbitalBuffers retained{};auto first=create({4,3,1},retained);
  retained.RF[14]=71;retained.dAFP[54]=29;
  for (int i=0;i<32;++i) {
    OpenMXOrbitalBuffers other{};auto next=create({i%5,1+i%7,i%2},other);
    require(other.RF!=retained.RF && other.AF!=retained.AF,"owners share storage");
    other.RF[0]=-3;other.AF[0]=-5;
    require(retained.RF[14]==71 && retained.dAFP[54]==29,"owner address or values changed");
  }
}
void invalid_test() {
  OpenMXOrbitalBuffers b{};auto retained=create({1,2,1},b);const auto saved=b;char error[7];
  for (auto shape : {OpenMXOrbitalShape{-1,1,0},{15,1,1},{INT_MAX,1,1},{1,0,1},{1,-1,0},
                     {14,INT_MAX,1},{1,1,-1},{1,1,2}}) {
    std::memset(error,'x',sizeof(error));
    require(!openmx_core_orbital_create(&shape,&b,error,sizeof(error)),"invalid shape accepted");
    require(error[0] && error[6]=='\0' && std::memcmp(&b,&saved,sizeof(b))==0,"failed create modified output or overflowed error");
  }
  const OpenMXOrbitalShape shape{1,2,1};
  require(!openmx_core_orbital_create(nullptr,&b,error,sizeof(error)),"null shape accepted");
  require(!openmx_core_orbital_create(&shape,nullptr,nullptr,0),"null output accepted");
  require(std::memcmp(&b,&saved,sizeof(b))==0,"null create changed output");
}
void allocation_test() {
  OpenMXOrbitalBuffers b{};auto retained=create({3,3,1},b);const auto saved=b;char error[256];
  const OpenMXOrbitalShape shape{4,3,1};
  openmx::test::fail_allocation_after(1000);
  auto *probe=openmx_core_orbital_create(&shape,&b,error,sizeof(error));
  const long count=1000-openmx::test::remaining_allocations();
  openmx::test::fail_allocation_after(-1);
  require(probe && count==3,"workspace should own exactly two numeric allocations and one handle");
  openmx_core_orbital_destroy(probe);b=saved;b.RF[0]=37;
  for (long fail=0;fail<count;++fail) {
    const auto live=openmx::test::live_allocations();
    openmx::test::fail_allocation_after(fail);
    auto *bad=openmx_core_orbital_create(&shape,&b,error,sizeof(error));
    openmx::test::fail_allocation_after(-1);
    if (bad) openmx_core_orbital_destroy(bad);
    require(!bad && error[0],"allocation failure escaped boundary");
    require(openmx::test::live_allocations()==live,"partial allocation leaked");
    require(std::memcmp(&b,&saved,sizeof(b))==0 && b.RF[0]==37,"failure invalidated retained workspace");
  }
}
void concurrent_test() {
  std::array<std::exception_ptr,4> errors{};std::array<std::thread,4> workers;
  for (std::size_t t=0;t<workers.size();++t) workers[t]=std::thread([&,t] {
    try { for (int i=0;i<32;++i) {OpenMXOrbitalBuffers b{};auto owner=create({i%5,1+i%3,1},b);
      b.RF[0]=double(t+1);b.dAFP[0]=double(i+3);
      require(b.RF[0]==double(t+1) && b.dAFP[0]==double(i+3),"workers share workspace");}}
    catch (...) {errors[t]=std::current_exception();}
  });
  for (auto& worker:workers) worker.join();
  for (auto error:errors) if (error) std::rethrow_exception(error);
}
}
int main(int argc,char **argv) {
  try {
    const std::string name=argc==2 ? argv[1] : "";
    if(name=="layout")layout_test();else if(name=="c-api")c_api_test();
    else if(name=="independent")independent_test();else if(name=="invalid")invalid_test();
    else if(name=="allocation")allocation_test();else if(name=="concurrent")concurrent_test();
    else throw std::runtime_error("unknown orbital workspace test");
    std::cout<<"PASS "<<name<<'\n';return 0;
  } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}

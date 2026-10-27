#include "openmx_core_vna_preparation.h"
#include <openmx/basis/gaunt.hpp>
#include "allocation_faults.hpp"
#include <array>
#include <cmath>
#include <future>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <type_traits>

namespace {
using Owner=std::unique_ptr<OpenMXVnaPreparation,decltype(&openmx_core_vna_preparation_destroy)>;
void require(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
Owner make(int angular=4,int projector=3,int multiplicity=2) {
  std::array<int,15> counts;counts.fill(1);int* rows[]{counts.data()};
  const OpenMXVnaPreparationShape shape{angular,1,projector,multiplicity,3,1,&angular,rows};
  OpenMXVnaPreparationBuffers data{};char error[160];
  Owner result(openmx_core_vna_preparation_create(&shape,&data,error,sizeof(error)),openmx_core_vna_preparation_destroy);
  require(bool(result),error);counts.fill(0);angular=-1;return result;
}
OpenMXGauntTableData get(const Owner& owner,OpenMXVnaGauntKind kind) {
  OpenMXGauntTableData data{};char error[160];
  require(openmx_core_vna_gaunt_table(owner.get(),kind,&data,error,sizeof(error)),error);
  return data;
}
void equal(const OpenMXGauntTableData& actual,const openmx::basis::GauntTable& expected) {
  const auto reference=expected.view();
  require(actual.l0max==reference.l0max && actual.l1max==reference.l1max && actual.llmax==reference.llmax && actual.row_stride==reference.row_stride,"VNA Gaunt table shape differs");
  for(std::size_t i=0;i<expected.values().size();++i)
    require(actual.values[i]==reference.values[i] && std::signbit(actual.values[i])==std::signbit(reference.values[i]),"VNA Gaunt coefficient/padding differs");
}
void contents() {
  static_assert(std::is_const_v<std::remove_reference_t<decltype(*OpenMXGauntTableData{}.values)>>);
  auto owner=make();const auto a=get(owner,OPENMX_VNA_GAUNT_PROJECTOR),b=get(owner,OPENMX_VNA_GAUNT_PRODUCT);
  equal(a,openmx::basis::GauntTable(4,3,8));equal(b,openmx::basis::GauntTable(4,4,8));
  require(a.values!=b.values,"independent table kinds share mutable allocation");
  const auto before=openmx::test::live_allocations();
  openmx::test::fail_allocation_after(0);
  const auto aa=get(owner,OPENMX_VNA_GAUNT_PROJECTOR),bb=get(owner,OPENMX_VNA_GAUNT_PRODUCT);
  openmx::test::fail_allocation_after(-1);
  require(a.values==aa.values && b.values==bb.values && openmx::test::live_allocations()==before,"table reacquisition allocates or invalidates view");
  // No projectors: the old native lfi used the basis maximum, not the unused
  // projector capacity, while the rectangular l1 dimension retained it.
  auto empty=make(1,3,0);equal(get(empty,OPENMX_VNA_GAUNT_PROJECTOR),openmx::basis::GauntTable(1,3,2));
}
void invalid() {
  auto owner=make();double sentinel=71.;OpenMXGauntTableData out{11,12,13,14,&sentinel};const auto saved=out;char error[128];
  require(!openmx_core_vna_gaunt_table(nullptr,OPENMX_VNA_GAUNT_PROJECTOR,&out,error,sizeof(error)) && *error,"null owner accepted");
  require(!openmx_core_vna_gaunt_table(owner.get(),OPENMX_VNA_GAUNT_PRODUCT,nullptr,error,sizeof(error)),"null out accepted");
  require(!openmx_core_vna_gaunt_table(owner.get(),2,&out,error,sizeof(error)),"invalid table kind accepted");
  // Supported preparation metadata need not imply the factorial sum supports
  // every high-angular host table. Fail before publication or numeric writes.
  auto high=make(14,14);
  require(!openmx_core_vna_gaunt_table(high.get(),OPENMX_VNA_GAUNT_PRODUCT,&out,error,sizeof(error)) && *error,"unsupported Gaunt degree sum accepted");
  require(out.values==saved.values && out.l0max==saved.l0max && out.l1max==saved.l1max && out.llmax==saved.llmax && out.row_stride==saved.row_stride && sentinel==71.,"failed acquisition changed output");
  (void)get(owner,OPENMX_VNA_GAUNT_PRODUCT);
}
void lifetime() {
  // Count actual allocations in the shared production core, including tables.
  const auto start=openmx::test::live_allocations();
  {
    auto first=make(),second=make(2,1);const auto a=get(first,OPENMX_VNA_GAUNT_PROJECTOR);
    const auto b=get(first,OPENMX_VNA_GAUNT_PRODUCT),c=get(second,OPENMX_VNA_GAUNT_PRODUCT);
    require(a.values!=b.values && a.values!=c.values && b.values!=c.values,"calculations share Gaunt storage");
    first.reset();equal(c,openmx::basis::GauntTable(2,2,4));
  }
  require(openmx::test::live_allocations()==start,"Gaunt ownership leaks after preparation destruction");
}
void allocation() {
  for(auto kind:{OPENMX_VNA_GAUNT_PROJECTOR,OPENMX_VNA_GAUNT_PRODUCT}) {
    long allocations;
    {
      auto owner=make();openmx::test::fail_allocation_after(1000);const auto data=get(owner,kind);
      allocations=1000-openmx::test::remaining_allocations();openmx::test::fail_allocation_after(-1);
      require(data.values && allocations>=2,"Gaunt owner/vector allocations not intercepted");
    }
    for(long failure=0;failure<allocations;++failure) {
      auto owner=make();const auto other=kind==OPENMX_VNA_GAUNT_PROJECTOR?OPENMX_VNA_GAUNT_PRODUCT:OPENMX_VNA_GAUNT_PROJECTOR;
      const auto kept=get(owner,other);const auto live=openmx::test::live_allocations();
      OpenMXGauntTableData output=kept;char error[160];
      openmx::test::fail_allocation_after(failure);
      const int ok=openmx_core_vna_gaunt_table(owner.get(),kind,&output,error,sizeof(error));
      openmx::test::fail_allocation_after(-1);
      require(!ok && *error && output.values==kept.values && output.row_stride==kept.row_stride && openmx::test::live_allocations()==live,"partial Gaunt initialization escaped/leaked/changed output");
      require(get(owner,other).values==kept.values,"failure invalidated previously borrowed table");
      const auto retried=get(owner,kind);require(retried.values && retried.values!=kept.values,"failed once initialization cannot retry");
    }
  }
}
void concurrent() {
  auto owner=make();std::array<std::future<OpenMXGauntTableData>,8> tasks;
  for(int worker=0;worker<8;++worker)tasks[worker]=std::async(std::launch::async,[&,worker] {
    const auto kind=worker%2?OPENMX_VNA_GAUNT_PROJECTOR:OPENMX_VNA_GAUNT_PRODUCT;
    auto first=get(owner,kind);
    for(int repeat=0;repeat<50;++repeat)require(get(owner,kind).values==first.values,"concurrent acquisition changed storage");
    return first;
  });
  const auto product=tasks[0].get(),projector=tasks[1].get();
  for(int i=2;i<8;++i)require(tasks[i].get().values==(i%2?projector:product).values,"cold concurrent acquisition created different owners");
  equal(projector,openmx::basis::GauntTable(4,3,8));equal(product,openmx::basis::GauntTable(4,4,8));
}
} // namespace
int main(int argc,char** argv) {
  try {
    require(argc==2,"expected VNA Gaunt test name");const std::string_view name=argv[1];
    if(name=="contents")contents();else if(name=="invalid")invalid();else if(name=="lifetime")lifetime();
    else if(name=="allocation")allocation();else if(name=="concurrent")concurrent();else throw std::invalid_argument("unknown VNA Gaunt test");
    std::cout<<"PASS VNA Gaunt "<<name<<'\n';
  }catch(const std::exception& error){openmx::test::fail_allocation_after(-1);std::cerr<<error.what()<<'\n';return 1;}
}

#include <openmx/core/dsvna_workspace.hpp>
#include "openmx_core_dsvna.h"
#include "allocation_faults.hpp"
#include <array>
#include <iostream>
#include <memory>
#include <string>
#include <type_traits>

namespace {
using namespace openmx::core;
void require(bool value,const char* reason) { if (!value) throw std::runtime_error(reason); }

void layout_test()
{
  static_assert(!std::is_copy_constructible_v<DsvnaWorkspace>);
  static_assert(std::is_const_v<std::remove_reference_t<decltype(
      element(std::declval<const DsvnaWorkspace&>().angular(),0,0,0))>>);
  const int first[]{2,0,1},second[]{0,2,0};
  const std::array basis{SpeciesBasisView{2,first},SpeciesBasisView{2,second}};
  VnaPreparation preparation({2,2,2,2,7},basis);
  DsvnaWorkspace work(preparation);
  require(work.angular().extent(0)==4 && work.angular().extent(1)==7 && work.angular().extent(2)==18,
          "DS_VNA angular storage has rectangular basis padding");
  require(work.sums().extent(1)==3 && work.sums().extent(2)==6,"DS_VNA sums include inactive radial rows");
  require(work.conversion().extent(1)==5 && work.conversion().extent(2)==5 &&
          work.bessel().extent(1)==5 && work.bessel().extent(2)==7 && work.scratch().extent(1)==7,
          "DS_VNA scratch dimensions are wrong");
  element(work.angular(),3,6,17)={41,-13};
  require(work.angular().data_handle()[work.angular().size()-1].i==-13,"angular mdspan stride mismatch");
  for (int l=0;l<=14;++l) {
    const int count[]{1};const std::array small{SpeciesBasisView{0,count}};
    VnaPreparation wide({0,1,l,1,1},small);DsvnaWorkspace rectangle(wide);
    require(rectangle.conversion().extent(1)==1 && rectangle.conversion().extent(2)==2*l+1 &&
            rectangle.bessel().extent(1)==2*l+1 && rectangle.angular().extent(2)==(l+1)*(l+1),
            "projector angular capacity was truncated to the PAO basis");
  }
}

void ownership_test()
{
  std::unique_ptr<DsvnaWorkspace> retained;
  for (int call=0;call<24;++call) {
    const int counts[]{1,0,2};const std::array basis{SpeciesBasisView{2,counts}};
    VnaPreparation preparation({2,2,1,2,call+1},basis);
    auto next=std::make_unique<DsvnaWorkspace>(preparation);
    element(next->angular(),0,0,0)={double(call),-31};
    element(next->radial(),1,call)=59;
    if (retained) require(element(retained->angular(),0,0,0).r==call-1,"workspaces share memory");
    retained=std::move(next);
  }
  require(element(retained->angular(),0,0,0).r==23 && element(retained->radial(),1,23)==59,
          "workspace retained a borrowed preparation buffer");
}

void empty_test()
{
  for (int active=0;active<2;++active) for (int projectors=0;projectors<2;++projectors) {
    const int count[]{1};const std::array basis{SpeciesBasisView{active?0:-1,
        active?std::span<const int>(count):std::span<const int>{}}};
    VnaPreparation preparation({0,1,0,projectors,3},basis);DsvnaWorkspace work(preparation);
    require(work.angular().size()==4*active*projectors && work.sums().size()==2*active*projectors,
            "empty DS_VNA buffers allocate numeric padding");
    require(work.bessel().size()==6 && work.scratch().size()==6,"empty basis lost valid Bessel scratch");
  }
}

void c_api_test()
{
  int angular[]{2,1},first[]{2,0,1},second[]{0,2};int* rows[]{first,second};
  const OpenMXVnaPreparationShape shape{2,2,2,2,7,2,angular,rows};
  OpenMXVnaPreparationBuffers basis{};OpenMXDsvnaBuffers buffers{};char error[256];
  const auto prep_destroy=&openmx_core_vna_preparation_destroy;
  std::unique_ptr<OpenMXVnaPreparation,decltype(prep_destroy)> preparation(
      openmx_core_vna_preparation_create(&shape,&basis,error,sizeof(error)),prep_destroy);
  require(preparation!=nullptr,error);
  const auto destroy=&openmx_core_dsvna_destroy;
  std::unique_ptr<OpenMXDsvnaWorkspace,decltype(destroy)> work(
      openmx_core_dsvna_create(preparation.get(),&buffers,error,sizeof(error)),destroy);
  require(work!=nullptr,error);
  for (int s=0;s<2;++s) {
    std::size_t orbital=0,radial=0;
    for (int l=0;l<=angular[s];++l) for (int r=0;r<rows[s][l];++r,++radial) {
      for (int p=0;p<6;++p)
        require(openmx_core_dsvna_sum(&buffers,&basis,s,l,r,p)==radial*6+p,"radial row has a gap");
      for (int m=0;m<2*l+1;++m,++orbital) {
        std::size_t component=0;
        for (int p=0;p<6;++p) for (int q=0;q<2*basis.projector_angular[p]+1;++q,++component) {
          const auto i=openmx_core_dsvna_pair(&buffers,&basis,s,l,r,m,p,q);
          require(i==orbital*18+component,"packed orbital/projector order differs from output");
          buffers.angular[3][i]={double(s+1),double(i)};
        }
      }
    }
    require(orbital==basis.orbitals[s],"packed rows differ from species orbital count");
  }
  require(buffers.angular[3]==buffers.angular[0]+3*7*18 && buffers.conversion[3]==buffers.conversion[0]+3*25 &&
          buffers.sums[1]==buffers.sums[0]+18 && buffers.radial[1]==buffers.radial[0]+7 &&
          buffers.bessel[1]==buffers.bessel[0]+35 && buffers.scratch[1]==buffers.scratch[0]+7,
          "C workspace channels overlap");
  const auto retained=buffers;
  require(!openmx_core_dsvna_create(nullptr,&buffers,error,sizeof(error)) && error[0],"null preparation accepted");
  require(!openmx_core_dsvna_create(preparation.get(),nullptr,error,sizeof(error)),"null output accepted");
  require(buffers.angular[0]==retained.angular[0],"failed create modified published buffers");
  preparation.reset();buffers.radial[0][0]=71;
  require(buffers.radial[0][0]==71,"preparation destruction invalidated workspace");
  openmx_core_dsvna_destroy(nullptr);
  const int absent[]{-1};const OpenMXVnaPreparationShape empty{0,1,0,0,1,1,absent,nullptr};
  preparation.reset(openmx_core_vna_preparation_create(&empty,&basis,error,sizeof(error)));
  require(preparation!=nullptr,error);
  work.reset(openmx_core_dsvna_create(preparation.get(),&buffers,error,sizeof(error)));
  require(work && !buffers.angular[0] && !buffers.sums[1] && buffers.bessel[0],"empty C view is invalid");
}

void allocation_test()
{
  int angular[]{2},counts[]{2,0,1};int* rows[]{counts};
  const OpenMXVnaPreparationShape shape{2,2,1,2,7,1,angular,rows};
  OpenMXVnaPreparationBuffers basis{};OpenMXDsvnaBuffers views{};char error[256];
  const auto destroy=&openmx_core_vna_preparation_destroy;
  std::unique_ptr<OpenMXVnaPreparation,decltype(destroy)> preparation(
      openmx_core_vna_preparation_create(&shape,&basis,error,sizeof(error)),destroy);
  require(preparation!=nullptr,error);
  openmx::test::fail_allocation_after(1000);
  auto* retained=openmx_core_dsvna_create(preparation.get(),&views,error,sizeof(error));
  const long count=1000-openmx::test::remaining_allocations();
  openmx::test::fail_allocation_after(-1);
  require(retained && count>=7,"workspace allocations were not intercepted");
  const auto original=views;views.radial[0][0]=83;
  for (long failure=0;failure<count;++failure) {
    const auto before=openmx::test::live_allocations();
    openmx::test::fail_allocation_after(failure);
    auto* failed=openmx_core_dsvna_create(preparation.get(),&views,error,sizeof(error));
    openmx::test::fail_allocation_after(-1);
    if (failed) openmx_core_dsvna_destroy(failed);
    require(!failed && error[0] && openmx::test::live_allocations()==before,"partial DS_VNA workspace leaked");
    require(views.angular[0]==original.angular[0] && views.radial[0][0]==83,"failure published partial views");
  }
  openmx_core_dsvna_destroy(retained);
}
} // namespace

int main(int argc,char** argv)
{
  try {
    const std::string name=argc==2?argv[1]:"";
    if (name=="layout") layout_test();
    else if (name=="ownership") ownership_test();
    else if (name=="empty") empty_test();
    else if (name=="c-api") c_api_test();
    else if (name=="allocation") allocation_test();
    else throw std::runtime_error("unknown DS_VNA test");
    std::cout<<"PASS DS_VNA workspace "<<name<<'\n';return 0;
  } catch (const std::exception& error) { std::cerr<<error.what()<<'\n';return 1; }
}

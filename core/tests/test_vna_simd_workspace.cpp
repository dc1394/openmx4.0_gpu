#include "openmx_core_vna.h"
#include "openmx_core_dsvna.h"
#include "allocation_faults.hpp"
#include <algorithm>
#include <array>
#include <climits>
#include <cstring>
#include <future>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string_view>

namespace {
using Vna=std::unique_ptr<OpenMXVnaWorkspace,decltype(&openmx_core_vna_destroy)>;
using Dsvna=std::unique_ptr<OpenMXDsvnaWorkspace,decltype(&openmx_core_dsvna_destroy)>;
using Preparation=std::unique_ptr<OpenMXVnaPreparation,decltype(&openmx_core_vna_preparation_destroy)>;
void require(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
Vna make_vna() {
  const OpenMXVnaShape shape{5,2,7};OpenMXVnaBuffers data{};char error[160];
  Vna owner(openmx_core_vna_create(&shape,&data,error,sizeof(error)),openmx_core_vna_destroy);
  require(bool(owner),error);return owner;
}
Dsvna make_dsvna(bool empty=false,int multiplicity=2) {
  int angular[]{empty?-1:4,empty?-1:2},first[]{1,0,2,1,1},second[]{1,1,1};
  int* rows[]{first,second};const OpenMXVnaPreparationShape shape{5,2,3,multiplicity,7,2,angular,rows};
  OpenMXVnaPreparationBuffers basis{};char error[160];
  Preparation prep(openmx_core_vna_preparation_create(&shape,&basis,error,sizeof(error)),openmx_core_vna_preparation_destroy);
  require(bool(prep),error);OpenMXDsvnaBuffers data{};
  Dsvna owner(openmx_core_dsvna_create(prep.get(),&data,error,sizeof(error)),openmx_core_dsvna_destroy);
  require(bool(owner),error);
  // Creation copied capacities: SIMD acquisition must survive the preparation.
  std::fill_n(first,5,0);std::fill_n(second,3,0);prep.reset();return owner;
}
OpenMXVnaSimdBuffers get(const Vna& owner,int orders=16) {
  OpenMXVnaSimdBuffers data{};char error[160];
  require(openmx_core_vna_simd_workspace(owner.get(),orders,&data,error,sizeof(error)),error);return data;
}
OpenMXDsvnaSimdBuffers get(const Dsvna& owner,int angular=1) {
  OpenMXDsvnaSimdBuffers data{};char error[160];
  require(openmx_core_dsvna_simd_workspace(owner.get(),angular,&data,error,sizeof(error)),error);return data;
}
template<class T> bool identical(const T& a,const T& b){return std::memcmp(&a,&b,sizeof(T))==0;}
void contents() {
  auto v=make_vna();auto a=get(v);
  require(a.orders==16 && a.bessel_derivative==a.bessel+7*16 && a.sum_derivative==a.sum+16,"VNA flat strides differ");
  a.arguments[6]=3.;a.bessel_derivative[7*16-1]=5.;a.sum_derivative[15]=7.;a.harmonics[3*16*16-1]={9.,11.};
  openmx::test::fail_allocation_after(0);const auto again=get(v);openmx::test::fail_allocation_after(-1);
  require(a.arguments==again.arguments && a.harmonics==again.harmonics && again.harmonics[3*16*16-1].i==11.,"VNA scratch reallocated or reset");
  auto d=make_dsvna();auto b=get(d);
  // Capacity l=5, active l=4, projector l=3; holes/multiple radial functions.
  require(b.pad==7 && b.llmax==8 && b.cy_stride==38 && b.cy_length==342 &&
      b.plane==27*32 && b.sum_slice==5*8 && b.weight_stride==14,"DSVNA active capacities differ");
  require(b.cpow_imag==b.cpow_real+9 && b.radial==b.cpow_real+18 && b.radial_derivative==b.cpow_real+27,"DSVNA coefficient planes differ");
  for(size_t i=0;i<6*b.cy_length;++i)require(b.harmonics[i]==0.,"harmonic padding is not zero");
  b.arguments[6]=13.;b.weighted[5*14-1]=17.;b.harmonics[6*342-1]=19.;
  b.accumulators[8*27*32-1]=23.;b.sums[2*9*5*8-1]=29.;b.radial_derivative[8]=31.;
  openmx::test::fail_allocation_after(0);const auto repeated=get(d);openmx::test::fail_allocation_after(-1);
  require(b.harmonics==repeated.harmonics && b.weighted==repeated.weighted && repeated.harmonics[6*342-1]==19. && repeated.sums[2*9*5*8-1]==29.,"DSVNA acquisition reallocates or resets");
  auto radial=make_dsvna();const auto r=get(radial,0);
  require(r.arguments && r.weighted && !r.harmonics && !r.accumulators && !r.sums && !r.cpow_real && !r.radial,"radial-only mode allocates angular planes");
  auto empty=make_dsvna(true);const auto e=get(empty);
  require(e.arguments && !e.weighted && !e.accumulators && !e.sums && e.harmonics && e.pad==3 && e.llmax==6,"empty-basis workspace dimensions differ");
  auto no_projectors=make_dsvna(false,0);const auto z=get(no_projectors,0);
  require(z.weighted && !z.accumulators && !z.sums && z.llmax==8 && z.plane==0,"zero-projector workspace differs");
}
void invalid() {
  auto v=make_vna();OpenMXVnaSimdBuffers out{};out.orders=19;double sentinel=17.;out.arguments=&sentinel;const auto saved=out;char error[160];
  for(int orders:{INT_MIN,-1,0,7,9,24,INT_MAX}) {
    require(!openmx_core_vna_simd_workspace(v.get(),orders,&out,error,sizeof(error)) && *error,"invalid padded orders accepted");
    require(identical(out,saved) && sentinel==17.,"failed VNA acquisition changed output");
  }
  require(!openmx_core_vna_simd_workspace(nullptr,16,&out,error,sizeof(error)),"null VNA owner accepted");
  require(!openmx_core_vna_simd_workspace(v.get(),16,nullptr,error,sizeof(error)),"null VNA output accepted");
  const auto valid=get(v);
  require(!openmx_core_vna_simd_workspace(v.get(),8,&out,error,sizeof(error)) && identical(out,saved),"VNA options changed after publication");
  require(get(v).arguments==valid.arguments,"invalid options damaged VNA storage");
  auto d=make_dsvna();OpenMXDsvnaSimdBuffers ds{};ds.pad=9;ds.arguments=&sentinel;const auto kept=ds;
  for(int angular:{-1,2,INT_MAX}) {
    require(!openmx_core_dsvna_simd_workspace(d.get(),angular,&ds,error,sizeof(error)) && *error,"invalid angular mode accepted");
    require(identical(ds,kept),"failed DSVNA acquisition changed output");
  }
  require(!openmx_core_dsvna_simd_workspace(nullptr,1,&ds,error,sizeof(error)),"null DSVNA owner accepted");
  require(!openmx_core_dsvna_simd_workspace(d.get(),1,nullptr,error,sizeof(error)),"null DSVNA output accepted");
  const auto published=get(d);
  require(!openmx_core_dsvna_simd_workspace(d.get(),0,&ds,error,sizeof(error)) && identical(ds,kept),"DSVNA options changed after publication");
  require(get(d).weighted==published.weighted,"invalid options damaged DSVNA storage");
}
void allocation() {
  {
    auto owner=make_vna();openmx::test::fail_allocation_after(1000);(void)get(owner);
    const auto allocations=1000-openmx::test::remaining_allocations();openmx::test::fail_allocation_after(-1);
    require(allocations>=5,"VNA scratch owner/arrays not intercepted");
    for(long failure=0;failure<allocations;++failure) {
      auto trial=make_vna();const auto live=openmx::test::live_allocations();
      OpenMXVnaSimdBuffers out{};out.orders=71;const auto saved=out;char error[160];
      openmx::test::fail_allocation_after(failure);
      const int ok=openmx_core_vna_simd_workspace(trial.get(),16,&out,error,sizeof(error));
      openmx::test::fail_allocation_after(-1);
      require(!ok && *error && identical(out,saved) && openmx::test::live_allocations()==live,"partial VNA scratch construction escaped/leaked");
      require(get(trial).orders==16,"VNA acquisition cannot retry");
    }
  }
  for(int angular:{0,1}) {
    auto owner=make_dsvna();openmx::test::fail_allocation_after(1000);(void)get(owner,angular);
    const auto allocations=1000-openmx::test::remaining_allocations();openmx::test::fail_allocation_after(-1);
    require(allocations>=(angular?7:3),"DSVNA scratch owner/arrays not intercepted");
    for(long failure=0;failure<allocations;++failure) {
      auto trial=make_dsvna();const auto live=openmx::test::live_allocations();
      OpenMXDsvnaSimdBuffers out{};out.pad=71;const auto saved=out;char error[160];
      openmx::test::fail_allocation_after(failure);
      const int ok=openmx_core_dsvna_simd_workspace(trial.get(),angular,&out,error,sizeof(error));
      openmx::test::fail_allocation_after(-1);
      require(!ok && *error && identical(out,saved) && openmx::test::live_allocations()==live,"partial DSVNA scratch construction escaped/leaked");
      require(get(trial,angular).arguments,"DSVNA acquisition cannot retry");
    }
  }
}
void lifetime() {
  const auto before=openmx::test::live_allocations();
  {
    auto a=make_vna(),b=make_vna();auto c=make_dsvna(),d=make_dsvna();
    auto av=get(a),bv=get(b);auto cv=get(c),dv=get(d);
    require(av.harmonics!=bv.harmonics && cv.weighted!=dv.weighted,"workers share mutable scratch");
    bv.sum[0]=41.;dv.arguments[0]=43.;a.reset();c.reset();
    require(get(b).sum[0]==41. && get(d).arguments[0]==43.,"destroying one worker invalidated another");
  }
  require(openmx::test::live_allocations()==before,"worker SIMD scratch leaked");
}
void concurrent() {
  std::array<std::future<double>,8> tasks;
  for(int worker=0;worker<8;++worker)tasks[worker]=std::async(std::launch::async,[worker] {
    auto v=make_vna();auto d=make_dsvna();const auto a=get(v);const auto b=get(d);
    for(int repeat=0;repeat<100;++repeat) {
      a.sum[0]=worker+1.;b.weighted[0]=worker+2.;
      require(get(v).sum[0]==worker+1. && get(d).weighted[0]==worker+2.,"concurrent workers interfered");
    }
    return a.sum[0]+b.weighted[0];
  });
  for(int worker=0;worker<8;++worker)require(tasks[worker].get()==2*worker+3.,"parallel scratch result differs");
}
} // namespace
int main(int argc,char** argv) {
  try {
    require(argc==2,"expected VNA SIMD workspace test name");const std::string_view name=argv[1];
    if(name=="contents")contents();else if(name=="invalid")invalid();else if(name=="allocation")allocation();
    else if(name=="lifetime")lifetime();else if(name=="concurrent")concurrent();else throw std::invalid_argument("unknown test");
    std::cout<<"PASS VNA SIMD workspace "<<name<<'\n';
  }catch(const std::exception& error){openmx::test::fail_allocation_after(-1);std::cerr<<error.what()<<'\n';return 1;}
}

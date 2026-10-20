#include <openmx/core/atom_pairs.hpp>
#include <openmx/core/projector_exchange.hpp>
#include "openmx_core_vna_invocation.h"
#include "openmx_core_projector_exchange.h"
#include "allocation_faults.hpp"
#include <climits>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <type_traits>

namespace {
using namespace openmx::core;
void require(bool value,const char *reason) { if (!value) throw std::runtime_error(reason); }
template<class T,auto Destroy> using Owner=std::unique_ptr<T,decltype(Destroy)>;
using Pairs=Owner<OpenMXAtomPairPlan,openmx_core_atom_pairs_destroy>;
using Exchange=Owner<OpenMXProjectorExchange,openmx_core_projector_exchange_destroy>;
using Invocation=Owner<OpenMXVnaInvocation,openmx_core_vna_invocation_destroy>;
struct Fixture {
  int map[3]{0,3,1},species[4]{0,0,1,1},counts[4]{0,2,0,1};
  int first[3]{1,2,3},third[2]{3,1},image1[3]{0,1,0},image3[2]{0,1};
  int *neighbors[4]{nullptr,first,nullptr,third},*images[4]{nullptr,image1,nullptr,image3};
  double xyz[4][4]{{},{0,.2,-.3,1.4},{0,1e16,2,.4},{0,1e16,-.1,-.2}};
  double *positions[4]{xyz[0],xyz[1],xyz[2],xyz[3]};
  double vectors[2][4]{{},{0,-1e16,.3,.7}},*translations[2]{vectors[0],vectors[1]};
  int angular[2]{1,0},basis0[2]{1,1},basis1[1]{2},*basis[2]{basis0,basis1};
  double abscissae[3]{-1,0,1};
  OpenMXAtomPairInput input{3,2,2,2,map,species,counts,neighbors,images,positions,translations};
  OpenMXAtomPairBuffers view{};char error[256]{};
  OpenMXVnaInvocationInput call{{1,2,1,1,3,2,angular,basis},input,.1,3,abscissae};
  OpenMXVnaInvocationBuffers call_view{};
  OpenMXAtomPairPlan *create() { return openmx_core_atom_pairs_create(&input,&view,error,sizeof(error)); }
  OpenMXVnaInvocation *invoke() { return openmx_core_vna_invocation_create(&call,&call_view,error,sizeof(error)); }
};

void pairs_layout()
{
  Fixture f;Pairs owner(f.create(),openmx_core_atom_pairs_destroy);require(bool(owner),f.error);
  require(f.view.count==5 && f.view.local_atoms==2,"ragged pair counts");
  const int mc[]{1,1,2,2,2},h[]{0,1,0,1,2},sp[]{1,1,0,0,0};
  require(!std::memcmp(mc,f.view.local,sizeof(mc)) && !std::memcmp(h,f.view.neighbor,sizeof(h)),"legacy pair order");
  require(!std::memcmp(sp,f.view.central_species,sizeof(sp)),"species snapshot");
  require(f.view.offsets[0]==0 && f.view.offsets[1]==2 && f.view.offsets[2]==5,"CSR atom offsets");
  bool rounding_sensitive=false;
  for (int p=0;p<f.view.count;++p) {
    const int global=f.map[mc[p]],other=f.neighbors[global][h[p]],image=f.images[global][h[p]];
    require(f.view.central_global[p]==global && f.view.neighbor_species[p]==f.species[other],"global/species identity");
    for (int d=0;d<3;++d) {
      const double a=f.xyz[global][d+1],b=f.xyz[other][d+1],t=f.vectors[image][d+1];
      const double forward=b+t-a,mirror=a-b-t;
      require(f.view.displacements[p*6+d]==forward && f.view.displacements[p*6+3+d]==mirror,"displacement arithmetic order");
      rounding_sensitive|=mirror!=-forward;
    }
  }
  require(rounding_sensitive,"fixture does not detect mirrored reassociation");
  const std::array<int,2> local{1,2};
  const int ns[]{1},is[]{0};
  const std::array atoms{AtomSiteView{0,{0,0,0},ns,is},AtomSiteView{1,{1,2,3},ns,is}};
  const std::array<std::array<double,3>,1> translations{{{0,0,0}}};
  AtomPairPlan plan(atoms,local,translations,2);
  static_assert(std::is_const_v<std::remove_reference_t<decltype(element(plan.displacements(),0,0,0))>>);
  require(plan.displacements().extent(0)==2 && element(plan.displacements(),1,0,2)==-3,"pair mdspan layout");
}

void pairs_ownership()
{
  OpenMXAtomPairBuffers retained{};Pairs owner(nullptr,openmx_core_atom_pairs_destroy);
  for (int round=0;round<10;++round) {
    Fixture f;Pairs next(f.create(),openmx_core_atom_pairs_destroy);require(bool(next),f.error);
    if (owner) require(retained.local!=f.view.local && retained.local[4]==2,"pair owners share storage");
    retained=f.view;owner=std::move(next);
    std::memset(f.xyz,0,sizeof(f.xyz));f.map[1]=-1;f.species[3]=-1;f.first[0]=-1;
    require(retained.central_global[0]==3 && retained.neighbor[4]==2 && retained.central_species[0]==1,"pair plan borrows input");
  }
  require(retained.central_global[4]==1 && retained.count==5,"pair input destruction invalidates view");
}

void pairs_empty()
{
  OpenMXAtomPairInput input{};OpenMXAtomPairBuffers view{};char error[128];
  Pairs plan(openmx_core_atom_pairs_create(&input,&view,error,sizeof(error)),openmx_core_atom_pairs_destroy);
  require(bool(plan) && !view.count && !view.local_atoms && view.offsets && !view.offsets[0],"empty rank plan");
  input.atoms=100;input.species=2;
  Pairs another(openmx_core_atom_pairs_create(&input,&view,error,sizeof(error)),openmx_core_atom_pairs_destroy);
  require(bool(another) && !view.count,"empty rank unnecessarily reads geometry");
}

void pairs_failure()
{
  for (int scenario=0;scenario<13;++scenario) {
    Fixture f;Pairs good(f.create(),openmx_core_atom_pairs_destroy);require(bool(good),f.error);const auto original=f.view;
    switch(scenario) {
      case 0:f.input.local_atoms=-1;break;case 1:f.map[1]=4;break;case 2:f.counts[3]=-1;break;
      case 3:f.counts[3]=INT_MAX;break;case 4:f.third[0]=0;break;case 5:f.image3[1]=2;break;
      case 6:f.species[3]=2;break;case 7:f.xyz[1][2]=NAN;break;case 8:f.vectors[1][3]=INFINITY;break;
      case 9:f.positions[2]=nullptr;break;case 10:f.input.images=nullptr;break;
      case 11:f.input.translations=0;break;case 12:f.input.species=-1;break;
    }
    Pairs bad(f.create(),openmx_core_atom_pairs_destroy);
    require(!bad && f.error[0] && !std::memcmp(&original,&f.view,sizeof(original)),"bad pair input published views");
  }
}

void exchange_layout()
{
  for (int scalar:{OPENMX_EXCHANGE_FLOAT,OPENMX_EXCHANGE_DOUBLE}) {
    OpenMXProjectorExchangeBuffers view{};char error[128];
    Exchange owner(openmx_core_projector_exchange_create(5,scalar,&view,error,sizeof(error)),openmx_core_projector_exchange_destroy);
    require(bool(owner),error);
    require(view.received==view.sent+5 && view.send_sizes==view.sent+10 && view.receive_sizes==view.sent+15,"flat exchange state");
    for (int i=0;i<20;++i) require(view.sent[i]==0,"uninitialized exchange state");
    ProjectorExchange<double> workspace(3);element(workspace.state(),2,1)=14;
    require(workspace.state().data_handle()[7]==14,"state mdspan layout");
  }
}

void exchange_buffers()
{
  for (int scalar:{OPENMX_EXCHANGE_FLOAT,OPENMX_EXCHANGE_DOUBLE}) {
    OpenMXProjectorExchangeBuffers view{};char error[128];
    Exchange owner(openmx_core_projector_exchange_create(3,scalar,&view,error,sizeof(error)),openmx_core_projector_exchange_destroy);
    require(bool(owner),error);void *sent=nullptr,*received=nullptr,*reuse=nullptr;
    require(openmx_core_projector_exchange_buffer(owner.get(),0,17,&sent,error,sizeof(error)),error);
    const auto bytes=(scalar==OPENMX_EXCHANGE_FLOAT?sizeof(float):sizeof(double))*17;
    std::memset(sent,0x53,bytes);
    for (int n:{0,1,1024,32768}) {
      require(openmx_core_projector_exchange_buffer(owner.get(),1,n,&received,error,sizeof(error)),error);
      require(received && received!=sent,"receive growth invalidated send storage");
      for (std::size_t i=0;i<bytes;++i) require(static_cast<unsigned char*>(sent)[i]==0x53,"receive overwrote in-flight send");
    }
    for (int n:{0,1,17,3}) {
      require(openmx_core_projector_exchange_buffer(owner.get(),0,n,&reuse,error,sizeof(error)),error);
      require(reuse==sent,"smaller send needlessly reallocated");
    }
    for (int scenario=0;scenario<3;++scenario) {
      reuse=sent;
      require(!openmx_core_projector_exchange_buffer(scenario==2?nullptr:owner.get(),scenario==0?2:0,
          scenario==1?-1:1,&reuse,error,sizeof(error)) && error[0] && reuse==sent,"invalid buffer changed output");
    }
  }
}

void exchange_ownership()
{
  OpenMXProjectorExchangeBuffers a{},b{};char error[128];
  Exchange first(openmx_core_projector_exchange_create(2,1,&a,error,sizeof(error)),openmx_core_projector_exchange_destroy);
  Exchange second(openmx_core_projector_exchange_create(2,2,&b,error,sizeof(error)),openmx_core_projector_exchange_destroy);
  require(bool(first) && bool(second),error);a.sent[0]=7;b.sent[0]=4;
  first.reset();require(b.sent[0]==4,"exchange owners share progress");
  const auto saved=b;
  for (int scenario=0;scenario<3;++scenario) {
    Exchange bad(openmx_core_projector_exchange_create(scenario==0?0:2,scenario==1?5:2,
        scenario==2?nullptr:&b,error,sizeof(error)),openmx_core_projector_exchange_destroy);
    require(!bad && error[0] && !std::memcmp(&saved,&b,sizeof(b)),"invalid exchange published views");
  }
}

void invocation_lifetime()
{
  Fixture f;Invocation first(f.invoke(),openmx_core_vna_invocation_destroy);require(bool(first),f.error);
  const auto original=f.call_view;
  f.abscissae[0]=100;f.basis0[0]=0;f.xyz[3][1]=-99;
  require(original.wavevectors[0]==.5*((3-.1)*(-1)+(3+.1)) && original.wavevectors[2]==3,"wavevectors borrow abscissae");
  require(original.preparation.orbitals[0]==4 && original.pairs.central_global[0]==3,"invocation metadata lifetime");
  Invocation second(f.invoke(),openmx_core_vna_invocation_destroy);require(bool(second),f.error);
  require(original.wavevectors!=f.call_view.wavevectors && original.preparation.orbitals!=f.call_view.preparation.orbitals,"invocations share owners");
  second.reset();require(original.preparation.orbitals[0]==4,"second destruction invalidated first");
}

void invocation_failure()
{
  for (int scenario=0;scenario<9;++scenario) {
    Fixture f;Invocation good(f.invoke(),openmx_core_vna_invocation_destroy);require(bool(good),f.error);const auto saved=f.call_view;
    switch(scenario) {
      case 0:f.call.maximum_wavevector=-1;break;case 1:f.call.minimum_wavevector=-1;break;
      case 2:f.call.basis.species=1;break;case 3:f.call.abscissae=nullptr;break;case 4:f.abscissae[2]=NAN;break;
      case 5:f.third[0]=4;break;case 6:f.basis0[0]=-1;break;case 7:f.call.basis.points=0;break;
      case 8:f.call.minimum_wavevector=f.call.maximum_wavevector=1e308;break;
    }
    const auto before=openmx::test::live_allocations();
    Invocation bad(f.invoke(),openmx_core_vna_invocation_destroy);
    require(!bad && f.error[0] && !std::memcmp(&saved,&f.call_view,sizeof(saved)) &&
        openmx::test::live_allocations()==before,"failed invocation leaked or published views");
  }
}

void allocation_test()
{
  Fixture f;
  for (int kind=0;kind<3;++kind) {
    OpenMXProjectorExchangeBuffers exchange_view{};
    auto create=[&]() -> void* {
      if(kind==0)return f.create();if(kind==1)return f.invoke();
      return openmx_core_projector_exchange_create(3,OPENMX_EXCHANGE_DOUBLE,&exchange_view,f.error,sizeof(f.error));
    };
    auto destroy=[&](void* ptr) {
      if(kind==0)openmx_core_atom_pairs_destroy(static_cast<OpenMXAtomPairPlan*>(ptr));
      else if(kind==1)openmx_core_vna_invocation_destroy(static_cast<OpenMXVnaInvocation*>(ptr));
      else openmx_core_projector_exchange_destroy(static_cast<OpenMXProjectorExchange*>(ptr));
    };
    openmx::test::fail_allocation_after(1000);void* good=create();
    const auto count=1000-openmx::test::remaining_allocations();openmx::test::fail_allocation_after(-1);
    require(good && count>=2,"allocations not intercepted");destroy(good);
    const auto original=f.view;const auto invocation=f.call_view;const auto exchange=exchange_view;
    for (long failure=0;failure<count;++failure) {
      const auto before=openmx::test::live_allocations();openmx::test::fail_allocation_after(failure);
      void* bad=create();openmx::test::fail_allocation_after(-1);if(bad)destroy(bad);
      require(!bad && f.error[0] && openmx::test::live_allocations()==before,"partial pipeline leaked");
      require(!std::memcmp(&original,&f.view,sizeof(original)) && !std::memcmp(&invocation,&f.call_view,sizeof(invocation)),"failed construction published views");
      require(!std::memcmp(&exchange,&exchange_view,sizeof(exchange)),"failed exchange construction published views");
    }
  }
  OpenMXProjectorExchangeBuffers view{};char error[128];
  Exchange owner(openmx_core_projector_exchange_create(2,2,&view,error,sizeof(error)),openmx_core_projector_exchange_destroy);
  require(bool(owner),error);void *old=nullptr;
  require(openmx_core_projector_exchange_buffer(owner.get(),0,4,&old,error,sizeof(error)),error);
  static_cast<double*>(old)[3]=19;void* next=old;
  const auto live=openmx::test::live_allocations();openmx::test::fail_allocation_after(0);
  const int ok=openmx_core_projector_exchange_buffer(owner.get(),0,100,&next,error,sizeof(error));
  openmx::test::fail_allocation_after(-1);
  require(!ok && error[0] && next==old && static_cast<double*>(old)[3]==19 && openmx::test::live_allocations()==live,"failed growth destroyed prior send");
}
}

int main(int argc,char** argv)
{
  try {
    const std::string name=argc==2?argv[1]:"";
    if(name=="pairs-layout")pairs_layout();else if(name=="pairs-ownership")pairs_ownership();
    else if(name=="pairs-empty")pairs_empty();else if(name=="pairs-failure")pairs_failure();
    else if(name=="exchange-layout")exchange_layout();else if(name=="exchange-buffers")exchange_buffers();
    else if(name=="exchange-ownership")exchange_ownership();else if(name=="invocation-lifetime")invocation_lifetime();
    else if(name=="invocation-failure")invocation_failure();else if(name=="allocation")allocation_test();
    else throw std::runtime_error("unknown projector pipeline test");
    std::cout<<"PASS projector pipeline "<<name<<'\n';return 0;
  } catch(const std::exception& error) { openmx::test::fail_allocation_after(-1);std::cerr<<error.what()<<'\n';return 1; }
}

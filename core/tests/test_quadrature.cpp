#include "openmx_core_quadrature.h"
#include "allocation_faults.hpp"
#include "openmx_core_native_resources.h"

#include <cmath>
#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {
void require(bool value,const char* reason) { if(!value)throw std::runtime_error(reason); }
struct State { int calls=0,mode=0; };
int consume(void* user,const OpenMXGaussRuleData* rule,char* error,std::size_t capacity)
{
  auto& state=*static_cast<State*>(user); ++state.calls;
  require(rule && rule->count>=3 && rule->nodes && rule->weights,"invalid borrowed quadrature");
  double moment=0;
  for(int i=0;i<rule->count;++i)moment+=rule->weights[i]*rule->nodes[i]*rule->nodes[i];
  require(std::abs(moment-2./3)<2e-11,"borrowed quadrature moment differs");
  if(state.mode==1) { if(error && capacity)std::snprintf(error,capacity,"consumer diagnostic"); return 0; }
  if(state.mode==2)throw std::runtime_error("consumer exception");
  if(state.mode==3)throw 7;
  if(state.mode==4) {
    const double first=rule->nodes[0]; State inner; char nested_error[128];
    require(openmx_core_with_gauss_rule(3,consume,&inner,nested_error,sizeof nested_error),nested_error);
    require(inner.calls==1 && rule->count==32 && rule->nodes[0]==first,"nested rule changed outer storage");
  }
  if(state.mode==5)return 0;
  return 1;
}
void contents()
{
  State state; char error[128]; const auto live=openmx::test::live_allocations();
  require(openmx_core_with_gauss_rule(1500,consume,&state,error,sizeof error),error);
  require(state.calls==1 && !*error,"quadrature consumer call count or diagnostic differs");
  require(openmx::test::live_allocations()==live,"quadrature storage retained after consumer returned");
}
void lifetime()
{
  for(int mode=1;mode<=5;++mode) {
    State state{0,mode}; char error[128]; const auto live=openmx::test::live_allocations();
    const int ok=openmx_core_with_gauss_rule(32,consume,&state,error,sizeof error);
    require(ok==(mode==4) && state.calls==1,"consumer status lost");
    require(openmx::test::live_allocations()==live,"consumer failure retained quadrature storage");
    if(mode!=4)require(*error,"consumer failure lost diagnostic");
    if(mode==1)require(std::string_view(error)=="consumer diagnostic","consumer diagnostic overwritten");
    if(mode==2)require(std::string_view(error)=="consumer exception","exception diagnostic lost");
  }
}
void invalid()
{
  State state; char error[128];
  require(!openmx_core_with_gauss_rule(0,consume,&state,error,sizeof error),"zero order accepted");
  require(*error && !state.calls,"failed generation invoked consumer");
  require(!openmx_core_with_gauss_rule(3,nullptr,&state,error,sizeof error),"null consumer accepted");
  char short_error[2]={'x','x'};
  require(!openmx_core_with_gauss_rule(-1,consume,&state,short_error,2),"negative order accepted");
  require(short_error[0] && !short_error[1] && !state.calls,"short diagnostic or callback count differs");
  require(!openmx_core_with_gauss_rule(-1,consume,&state,nullptr,0),"invalid order without diagnostic accepted");
}
void allocation()
{
  bool complete=false; int failures=0;
  for(int index=0;index<32;++index) {
    State state; char error[128]; const auto live=openmx::test::live_allocations();
    openmx::test::fail_allocation_after(index);
    const int ok=openmx_core_with_gauss_rule(32,consume,&state,error,sizeof error);
    openmx::test::fail_allocation_after(-1);
    require(openmx::test::live_allocations()==live,"quadrature allocation failure leaked storage");
    if(ok) { complete=true; require(state.calls==1,"successful allocation skipped consumer"); break; }
    ++failures; require(*error && !state.calls,"allocation failure invoked consumer or lost diagnostic");
  }
  require(complete && failures>0,"quadrature allocation fault sweep incomplete");
}

using Context=std::unique_ptr<OpenMXCoreContext,decltype(&openmx_core_destroy)>;
Context context()
{
  char error[128];
  Context owner(openmx_core_create_host(error,sizeof error),openmx_core_destroy);
  require(bool(owner),error);
  return owner;
}
bool same(OpenMXGaussRuleData a,OpenMXGaussRuleData b)
{ return a.count==b.count && a.nodes==b.nodes && a.weights==b.weights; }
struct Boundary {
  decltype(&openmx_core_radial_quadrature_initialize) initialize=openmx_core_radial_quadrature_initialize;
  decltype(&openmx_core_radial_quadrature_get) get=openmx_core_radial_quadrature_get;
};
void context_contents(const Boundary& api)
{
  auto a=context(),b=context(); char error[128]; OpenMXGaussRuleData x,y;
  require(api.initialize(a.get(),300,&x,error,sizeof error),error);
  require(api.initialize(b.get(),17,&y,error,sizeof error),error);
  require(x.nodes!=y.nodes && x.weights!=y.weights,"calculations share quadrature storage");
  State state; require(consume(&state,&x,error,sizeof error) && consume(&state,&y,error,sizeof error),error);
  OpenMXGaussRuleData reused{};
  openmx::test::fail_allocation_after(0);
  const int get=api.get(a.get(),&reused,error,sizeof error);
  const int init=api.initialize(a.get(),300,&reused,error,sizeof error);
  openmx::test::fail_allocation_after(-1);
  require(get && init && same(reused,x),"quadrature reuse allocated or changed borrowed addresses");
  openmx_core_finish(b.get());
  require(api.get(a.get(),&reused,error,sizeof error) && same(reused,x),
      "finishing another calculation invalidated quadrature");
  require(consume(&state,&reused,error,sizeof error),error);
}
void context_invalid(const Boundary& api)
{
  auto owner=context(); char error[128]; const double sentinel=7;
  const OpenMXGaussRuleData saved{-1,&sentinel,&sentinel}; auto out=saved;
  require(!api.get(owner.get(),&out,error,sizeof error) && same(out,saved) && *error,
      "uninitialized quadrature exposed storage");
  require(!api.initialize(nullptr,3,&out,error,sizeof error) && same(out,saved),
      "null context accepted");
  require(!api.initialize(owner.get(),3,nullptr,error,sizeof error),"null output accepted");
  require(!api.get(nullptr,&out,nullptr,0) && same(out,saved),"null getter context accepted");
  require(!api.get(owner.get(),nullptr,error,sizeof error),"null getter output accepted");
  for(int count : {0,-1}) {
    require(!api.initialize(owner.get(),count,&out,error,sizeof error) && same(out,saved) && *error,
        "invalid count changed quadrature output");
  }
  OpenMXGaussRuleData original;
  require(api.initialize(owner.get(),3,&original,error,sizeof error),error);
  char short_error[2]={'x','x'};
  require(!api.initialize(owner.get(),4,&out,short_error,2) && same(out,saved),
      "quadrature resize accepted or changed failed output");
  require(short_error[0] && !short_error[1],"short diagnostic not terminated");
  require(api.get(owner.get(),&out,error,sizeof error) && same(out,original),
      "failed resize invalidated prior views");
  openmx_core_finish(owner.get()); out=saved;
  require(!api.get(owner.get(),&out,error,sizeof error) && same(out,saved),
      "finished quadrature exposed storage");
  require(!api.initialize(owner.get(),3,&out,error,sizeof error) && same(out,saved),
      "finished calculation reinitialized quadrature");
}
struct Drain {
  OpenMXCoreContext* context;
  OpenMXGaussRuleData rule;
  bool* called;
  bool* valid;
  decltype(&openmx_core_radial_quadrature_get) get;
};
void drain(OpenMXNativeResources*,void* pointer)
{
  const auto& state=*static_cast<Drain*>(pointer);
  OpenMXGaussRuleData rule{}; char error[128];
  *state.called=true;
  *state.valid=state.get(state.context,&rule,error,sizeof error) &&
      same(rule,state.rule) && rule.nodes[0]<0 && rule.weights[0]>0;
}
void context_lifetime(const Boundary& api)
{
  for(bool explicit_finish : {false,true}) {
    const auto before=openmx::test::live_allocations();
    bool called=false,valid=false;
    {
      auto owner=context(); char error[128]; OpenMXGaussRuleData rule;
      require(api.initialize(owner.get(),32,&rule,error,sizeof error),error);
      OpenMXNativeResources* resources=nullptr; void* pointer=nullptr;
      require(openmx_core_native_resources_state(owner.get(),"quadrature drain",sizeof(Drain),drain,
          &resources,&pointer,error,sizeof error),error);
      *static_cast<Drain*>(pointer)={owner.get(),rule,&called,&valid,api.get};
      if(explicit_finish) {
        openmx_core_finish(owner.get());
        require(called && valid,"finish freed quadrature before native drain");
        openmx_core_finish(owner.get());
      }
    }
    require(called && valid,"destruction freed quadrature before native drain");
    require(openmx::test::live_allocations()==before,"calculation retained quadrature storage");
  }
}
void context_allocation(const Boundary& api)
{
  bool complete=false; int failures=0;
  for(int index=0;index<32;++index) {
    auto owner=context(); char error[128]; const double sentinel=1;
    const OpenMXGaussRuleData saved{-1,&sentinel,&sentinel}; auto out=saved;
    const auto before=openmx::test::live_allocations();
    openmx::test::fail_allocation_after(index);
    const int ok=api.initialize(owner.get(),32,&out,error,sizeof error);
    openmx::test::fail_allocation_after(-1);
    if(ok) { complete=true; State state; require(consume(&state,&out,error,sizeof error),error); break; }
    ++failures;
    require(*error && same(out,saved),"allocation failure changed output or lost diagnostic");
    require(openmx::test::live_allocations()==before,"quadrature initialization failure leaked storage");
    require(!api.get(owner.get(),&out,error,sizeof error) && same(out,saved),
        "partial quadrature was published");
    require(api.initialize(owner.get(),32,&out,error,sizeof error),error);
  }
  require(complete && failures>=3,"context quadrature fault sweep incomplete");
}
void independent_rules()
{
  auto owner=context(); char error[128]; OpenMXGaussRuleData radial,neutral;
  require(openmx_core_radial_quadrature_initialize(owner.get(),300,&radial,error,sizeof error),error);
  neutral=radial;
  const auto before=openmx::test::live_allocations();
  openmx::test::fail_allocation_after(0);
  const int ok=openmx_core_neutral_quadrature_initialize(owner.get(),150,&neutral,error,sizeof error);
  openmx::test::fail_allocation_after(-1);
  require(!ok && same(neutral,radial) && openmx::test::live_allocations()==before,
      "failed neutral initialization changed borrowed storage");
  OpenMXGaussRuleData retained{};
  require(openmx_core_radial_quadrature_get(owner.get(),&retained,error,sizeof error) && same(retained,radial),
      "neutral failure invalidated projector quadrature");
  require(openmx_core_neutral_quadrature_initialize(owner.get(),150,&neutral,error,sizeof error),error);
  require(neutral.count==150 && radial.count==300 && neutral.nodes!=radial.nodes && neutral.weights!=radial.weights,
      "independent quadrature purposes alias");
  State state;
  require(consume(&state,&radial,error,sizeof error) && consume(&state,&neutral,error,sizeof error),error);
  require(openmx_core_radial_quadrature_get(owner.get(),&retained,error,sizeof error) && same(retained,radial),
      "neutral initialization changed projector addresses");
  require(openmx_core_neutral_quadrature_get(owner.get(),&retained,error,sizeof error) && same(retained,neutral),
      "neutral retrieval changed addresses");
}
}

int main(int argc,char** argv)
{
  try {
    if(argc!=2)throw std::invalid_argument("expected quadrature bridge test case"); std::string name=argv[1];
    Boundary api;
    if(name.starts_with("neutral-")) {
      api={openmx_core_neutral_quadrature_initialize,openmx_core_neutral_quadrature_get};
      name.erase(0,8);
    }
    if(name=="contents")contents(); else if(name=="lifetime")lifetime(); else if(name=="invalid")invalid(); else if(name=="allocation")allocation();
    else if(name=="context")context_contents(api); else if(name=="context-invalid")context_invalid(api);
    else if(name=="context-lifetime")context_lifetime(api); else if(name=="context-allocation")context_allocation(api);
    else if(name=="independent")independent_rules();
    else throw std::invalid_argument("unknown quadrature bridge test case");
    std::cout<<"PASS quadrature bridge "<<name<<'\n'; return 0;
  } catch(const std::exception& error) { openmx::test::fail_allocation_after(-1); std::cerr<<error.what()<<'\n'; return 1; }
}

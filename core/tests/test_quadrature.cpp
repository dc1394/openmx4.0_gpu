#include "openmx_core_quadrature.h"
#include "allocation_faults.hpp"

#include <cmath>
#include <cstdio>
#include <iostream>
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
}

int main(int argc,char** argv)
{
  try {
    if(argc!=2)throw std::invalid_argument("expected quadrature bridge test case"); const std::string name=argv[1];
    if(name=="contents")contents(); else if(name=="lifetime")lifetime(); else if(name=="invalid")invalid(); else if(name=="allocation")allocation();
    else throw std::invalid_argument("unknown quadrature bridge test case");
    std::cout<<"PASS quadrature bridge "<<name<<'\n'; return 0;
  } catch(const std::exception& error) { openmx::test::fail_allocation_after(-1); std::cerr<<error.what()<<'\n'; return 1; }
}

#include "openmx_mixing_rhok_internal.h"
#include "openmx_env.h"
#include "allocation_faults.hpp"
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>

constexpr int GPUSOLVER=3,mpi_comm_level1=0;
int scf_eigen_lib_flag=GPUSOLVER,List_YOUSO[60]={},log_kind=0,prints[4]={};
struct AllocationAbort:std::exception {};
void MPI_Abort(int,int){throw AllocationAbort();}
void check(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
void PrintMemory(const char*,double,const char*){prints[log_kind]++;}
#include "mixing_rhok_state.inc"

namespace {
using Context=std::unique_ptr<OpenMXCoreContext,decltype(&openmx_core_destroy)>;
Context calculation(){
  char error[256];Context owner(openmx_core_create_host(error,sizeof(error)),openmx_core_destroy);
  check(bool(owner),error);return owner;
}
void flags(MixingRhokState* state){
  check(state->diis_first_normal && state->diis_first_negf &&
        state->kerker_first_normal && state->kerker_first_negf,"new calculation inherited memory-log flags");
}
void logs(MixingRhokState* state){
  log_kind=0;log_diis_normal(state);log_kind=1;log_diis_negf(state);
  log_kind=2;log_kerker_normal(state);log_kind=3;log_kerker_negf(state);
}
void seed(MixingRhokState* state,int iteration,int dimension,double tag){
  state->fast.valid=1;state->fast.last_scf_iter=iteration;state->fast.last_nummix=dimension;
  for(int i=0;i<state->fast.l38*state->fast.l38;i++)state->fast.A[i]=tag+i*.125;
}
void mode(const char* value){check(!openmx_env_set("OPENMX_MIX_FAST",value),"set density-mixing policy");}
void rotation(){
  double a=1,b=2,c=3,d=4;double* v[]={&a,&b,&c,&d};double** slots[]={v,v+1,v+2,v+3};
  MixRhok_RotateSlots(slots,4);
  check(slots[0]==v+3 && slots[1]==v && slots[2]==v+1 && slots[3]==v+2,"density history pointer rotation changed");
}
void failures(){
  auto first=calculation(),second=calculation();auto* a=Mixing_Rhok_Bind_Resources(first.get());auto* b=Mixing_Rhok_Bind_Resources(second.get());
  List_YOUSO[38]=7;mode("1");check(!preflight(a,4,3,1) && !preflight(b,4,3,1),"fresh preflight became incremental");seed(a,4,3,100);seed(b,4,3,200);
  List_YOUSO[38]=9;openmx::test::fail_allocation_after(0);bool aborted=false;
  try{(void)preflight(a,5,3,1);}catch(const AllocationAbort&){aborted=true;}
  openmx::test::fail_allocation_after(-1);
  check(aborted && !a->fast.A && openmx_core_native_resource_count(a->resources)==0,"cache registration failure was not drained");
  check(b->fast.valid && b->fast.A[8]==201 && openmx_core_native_resource_count(b->resources)==1,"failed cache resize damaged another context");
  check(!preflight(a,5,3,1) && a->fast.l38==9 && !a->fast.valid,"failed cache did not recover with a fresh layout");
  seed(a,5,3,300);openmx::test::fail_allocation_after(0);
  openmx_core_finish(first.get());openmx_core_native_resources_clear(second.get());
  openmx::test::fail_allocation_after(-1);
}
}
int main(){try{
  const auto initial=openmx::test::live_allocations();
  rotation();
  for(int repeat=0;repeat<100;repeat++){
    auto first=calculation(),second=calculation();auto* a=Mixing_Rhok_Bind_Resources(first.get());auto* b=Mixing_Rhok_Bind_Resources(second.get());
    check(a!=b && a->resources!=b->resources,"density-mixing contexts share state");
    OpenMXMixingHistory* owned=nullptr;char error[256];check(openmx_core_mixing_history(first.get(),&owned,error,sizeof(error)) && a->history==owned && b->history!=owned,"density mixing borrowed the wrong shared history");flags(a);flags(b);
    check(!a->fast.valid && a->fast.last_scf_iter==-1 && !a->fast.A,"new Gram cache lost its initial sentinels");
    std::fill(std::begin(prints),std::end(prints),0);
    for(int n=0;n<3;n++)logs(a);
    check(prints[0]==3 && prints[1]==3 && prints[2]==1 && prints[3]==1,"memory logging did not run exactly once per path");flags(b);logs(b);
    check(prints[0]==6 && prints[1]==6 && prints[2]==2 && prints[3]==2,"memory-log state leaked between calculations");
    const int history=repeat%5+7,spin=repeat%3+1,new_spin=spin%3+1;List_YOUSO[38]=history;mode("1");
    check(!preflight(a,4,3,spin) && !preflight(b,4,3,spin),"unseeded cache was reused");
    check(a->fast.A && b->fast.A && a->fast.A!=b->fast.A && a->fast.l38==history,"Gram numeric storage is not calculation-owned");
    seed(a,4,3,100);seed(b,4,3,200);auto* address=a->fast.A;
    openmx::test::fail_allocation_after(0);
    check(Mixing_Rhok_Bind_Resources(first.get())==a && preflight(a,5,3,spin) && preflight(a,5,4,spin),"cache lookup/continuity allocated or reset state");
    openmx::test::fail_allocation_after(-1);
    check(a->fast.A==address && a->fast.A[8]==101 && b->fast.A[8]==201,"cache reuse lost the payload or selected another calculation");
    check(!preflight(a,4,3,spin) && !preflight(a,6,3,spin) && !preflight(a,5,5,spin),"SCF rewind/gap or dimension jump reused stale Gram entries");
    mode("0");check(!preflight(a,5,3,spin) && a->fast.A==address,"disabled fast path allocated/replaced cache");
    mode("1");scf_eigen_lib_flag=0;check(!preflight(a,5,3,spin),"CPU eigen route enabled GPU-only mixing optimization");scf_eigen_lib_flag=GPUSOLVER;
    check(!preflight(a,5,3,new_spin) && !a->fast.valid && a->fast.last_scf_iter==-1,"spin change did not invalidate the Gram layout");
    check(b->fast.valid && b->fast.spinmax==spin && b->fast.A[8]==201,"layout change touched another calculation");
    List_YOUSO[38]=history+1;check(!preflight(a,5,3,new_spin) && a->fast.l38==history+1,"history resize kept the old layout");
    openmx::test::fail_allocation_after(0);openmx_core_native_resources_clear(first.get());openmx_core_finish(second.get());openmx::test::fail_allocation_after(-1);
    auto* fresh=Mixing_Rhok_Bind_Resources(first.get());flags(fresh);
    check(!fresh->fast.A && !fresh->fast.valid && fresh->fast.last_scf_iter==-1,"next input inherited the previous cache");
  }
  failures();
  check(openmx::test::live_allocations()==initial,"density-mixing lifecycle leaked C++ allocations");
  std::cout<<"PASS: 100 dual-context density-mixing lifecycles, four logging paths, continuity/rewind, history/spin resize, policy, real allocation failure and exact-owner cleanup\n";return 0;
}catch(const std::exception& error){openmx::test::fail_allocation_after(-1);std::cerr<<error.what()<<'\n';return 1;}}

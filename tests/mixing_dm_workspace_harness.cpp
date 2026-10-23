#include "openmx_mixing_dm_internal.h"
#include "allocation_faults.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>

int mpi_comm_level1=0,atomnum=1,Matomnum=1,SpeciesNum=1;
int M2G[]={0,1},WhatSpecies[]={0,0},Spe_Total_CNO[]={1},FNAN[]={0,0},neighbor[]={1};
int* natn[]={neighbor,neighbor};int List_YOUSO[60]={};
int SpinP_switch=3,SO_switch=1,Hub_U_switch=0,Constraint_NCS_switch=0,Zeeman_NCS_switch=0,Zeeman_NCO_switch=0;
int log_kind=0,prints[2]={};
struct AllocationAbort:std::exception {};
void MPI_Abort(int,int){throw AllocationAbort();}
void allocation_abort(){throw AllocationAbort();}
void check(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
void equal(long double a,long double b){check(std::isfinite(a) && std::abs(a-b)<1e-11L*(1+std::abs(b)),"mixing numerical value differs");}
void PrintMemory(const char*,double bytes,const char*){equal(bytes,26*sizeof(double));prints[log_kind]++;}
#include "openmx_mixing_dm_workspace.h"

MixDM_ScratchArguments expected{};int selected=-1,behavior=0;
static void operation(int tag,OpenMXSolverInvocation* scratch,const MixDM_ScratchArguments* args){
  check(args->state==expected.state && args->iteration==expected.iteration && args->residual==expected.residual &&
    args->imaginary_residual==expected.imaginary_residual && args->electronic==expected.electronic,"scratch bridge changed borrowed arguments");
  selected=tag;
  const auto matrix=OpenMX_Solver_MatrixScratch(scratch,&args->electronic->fields[OPENMX_ELECTRONIC_DENSITY],0);
  auto* x=OpenMX_Solver_OverwriteReal(scratch,"bridge-vector",11);
  for(size_t i=0;i<matrix.elements;++i)matrix.values[i]=i+.25;
  x[10]=19;
  if(behavior==1)return;
  if(behavior==2)throw std::runtime_error("injected numerical callback exception");
  equal(x[10],19);equal(matrix.values[matrix.elements-1],25.25);
}
static void DIIS_Mixing_DM_Compute(OpenMXSolverInvocation* scratch,const MixDM_ScratchArguments* args){operation(0,scratch,args);}
static void ADIIS_Mixing_DM_Compute(OpenMXSolverInvocation* scratch,const MixDM_ScratchArguments* args){operation(1,scratch,args);}
static void GR_Pulay_DM_Compute(OpenMXSolverInvocation* scratch,const MixDM_ScratchArguments* args){operation(2,scratch,args);}
// Production binding's fatal process exit is modeled only in this fixture.
#define abort allocation_abort
#include "mixing_dm_workspace.inc"
#undef abort

namespace {
using Context=std::unique_ptr<OpenMXCoreContext,decltype(&openmx_core_destroy)>;
Context calculation(){char error[256];Context c(openmx_core_create_host(error,sizeof(error)),openmx_core_destroy);check(bool(c),error);return c;}
void flags(OpenMXMixingDMState* s){check(s->initialized && s->diis_first && s->gr_first,"fresh calculation inherited memory-log state");}
void logs(OpenMXMixingDMState* s){log_kind=0;log_diis(s);log_kind=1;log_gr(s);}
void state_lifecycle(){
  for(int repeat=0;repeat<100;++repeat){
    auto first=calculation(),second=calculation();auto* a=Mixing_DM_Bind_Resources(first.get());auto* b=Mixing_DM_Bind_Resources(second.get());
    check(a!=b,"calculations share real-space mixing state");
    OpenMXMixingHistory* owned=nullptr;char error[256];check(openmx_core_mixing_history(first.get(),&owned,error,sizeof(error)) && a->history==owned && b->history!=owned,"real-space mixing borrowed the wrong shared history");flags(a);flags(b);prints[0]=prints[1]=0;
    for(int j=0;j<3;++j)logs(a);
    check(prints[0]==2 && prints[1]==1 && !a->diis_first && !a->gr_first,"production logging did not run once per calculation");flags(b);
    logs(b);check(prints[0]==4 && prints[1]==2,"logging state leaked between calculations");
    openmx::test::fail_allocation_after(0);auto* same=Mixing_DM_Bind_Resources(first.get());openmx::test::fail_allocation_after(-1);
    check(same==a && !same->diis_first && !same->gr_first,"state lookup allocated or reset flags");
    openmx::test::fail_allocation_after(0);openmx_core_native_resources_clear(first.get());openmx_core_finish(second.get());openmx::test::fail_allocation_after(-1);
    flags(Mixing_DM_Bind_Resources(first.get()));flags(Mixing_DM_Bind_Resources(second.get()));
  }
  auto owner=calculation();openmx::test::fail_allocation_after(10000);(void)Mixing_DM_Bind_Resources(owner.get());
  const long count=10000-openmx::test::remaining_allocations();openmx::test::fail_allocation_after(-1);
  check(count>2,"state allocation interception missed the actual core");openmx_core_native_resources_clear(owner.get());
  for(long failure=0;failure<count;++failure){
    const auto before=openmx::test::live_allocations();bool failed=false;openmx::test::fail_allocation_after(failure);
    try{(void)Mixing_DM_Bind_Resources(owner.get());}catch(const AllocationAbort&){failed=true;}
    openmx::test::fail_allocation_after(-1);check(failed && openmx::test::live_allocations()==before,"partial state construction leaked");
    flags(Mixing_DM_Bind_Resources(owner.get()));openmx_core_native_resources_clear(owner.get());
  }
}
void invoke(int tag){
  const auto& a=expected;
  if(tag==0)DIIS_Mixing_DM(a.state,a.iteration,a.residual,a.imaginary_residual,a.electronic);
  else if(tag==1)ADIIS_Mixing_DM(a.state,a.iteration,a.residual,a.imaginary_residual,a.electronic);
  else GR_Pulay_DM(a.state,a.iteration,a.residual,a.electronic);
}
void dispatch(){
  OpenMXMixingDMState state{};const size_t components[]={0,2,4},atoms[]={0,1,3,4,6};
  const int shapes[]={1,1,2,3,3,2,1,1,2,3,3,2};OpenMXElectronicViews electronic{};
  electronic.fields[OPENMX_ELECTRONIC_DENSITY]={1,2,2,4,6,26,components,atoms,nullptr,shapes,nullptr};
  double***** borrowed[2]={};expected={nullptr,&state,17,borrowed,borrowed+1,&electronic};
  for(int tag=0;tag<3;++tag){
    expected.state=&state;expected.imaginary_residual=tag==2?nullptr:borrowed+1;
    for(behavior=0;behavior<3;++behavior){
      selected=-1;const auto before=openmx::test::live_allocations();bool failed=false;
      try{invoke(tag);}catch(const AllocationAbort&){failed=true;}
      check(selected==tag && failed==(behavior==2) && openmx::test::live_allocations()==before,"entry dispatch or return/exception cleanup failed");
    }
  }
  expected.state=&state;expected.imaginary_residual=borrowed+1;behavior=0;
  openmx::test::fail_allocation_after(10000);invoke(0);const long count=10000-openmx::test::remaining_allocations();openmx::test::fail_allocation_after(-1);
  check(count>10,"scratch allocation interception missed the actual core");
  for(long failure=0;failure<count;++failure){
    const auto before=openmx::test::live_allocations();bool failed=false;openmx::test::fail_allocation_after(failure);
    try{invoke(0);}catch(const AllocationAbort&){failed=true;}
    openmx::test::fail_allocation_after(-1);check(failed && openmx::test::live_allocations()==before,"partial scratch construction leaked");
  }
}
int inverses(OpenMXSolverInvocation* scratch,void*,double* result){
  List_YOUSO[16]=7;auto a=OpenMX_Solver_OverwriteView(scratch,"input",7,7),b=OpenMX_Solver_OverwriteView(scratch,"inverse",7,7);
  for(auto inverse:{diis::Inverse,adiis::Inverse,gr::Inverse}){
    for(int size:{1,3,2,4,3}){
      std::fill(a.values,a.values+49,-71);std::fill(b.values,b.values+49,-91);
      for(int i=0;i<size;++i)for(int j=0;j<size;++j)a.values[i*7+j]=i==j?4+i:.125*(1+i+2*j);
      double original[49];std::copy(a.values,a.values+49,original);inverse(scratch,size-1,&a,&b);
      check(!std::memcmp(a.values,original,sizeof(original)),"inverse did not restore its borrowed input");
      for(int i=0;i<size;++i)for(int j=0;j<size;++j){long double product=0;
        for(int k=0;k<size;++k)product+=static_cast<long double>(original[i*7+k])*b.values[k*7+j];equal(product,i==j?1:0);}
      check(b.values[48]==-91,"inverse overwrote inactive subspace");
    }
    inverse(scratch,-1,&a,&b);for(int i=0;i<49;++i)check(a.values[i]==0,"empty inverse sentinel changed");
  }
  *result=0;return 1;
}
}
int main(){try{
  const auto initial=openmx::test::live_allocations();state_lifecycle();dispatch();
  char error[256];double result;check(openmx_core_solver_run(2,inverses,nullptr,&result,error,sizeof(error)),error);
  check(openmx::test::live_allocations()==initial,"real-space mixing lifetime leaked C++ allocations");
  std::cout<<"PASS: 100 dual-context real-space mixing lifetimes, production logging, entry routing, actual allocation failures/cleanup and three production inverse identities\n";return 0;
}catch(const std::exception& error){openmx::test::fail_allocation_after(-1);std::cerr<<error.what()<<'\n';return 1;}}

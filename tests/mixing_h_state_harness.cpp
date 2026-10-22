#include "fixtures/native_resources/sdk_model.hpp"
#include "openmx_mixing_h_state.h"
#include "openmx_gpu_workers.h"
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>

constexpr int mpi_comm_level1=0,Host_ID=0,MPI_INT=1,MPI_UNSIGNED_LONG_LONG=2,
              MPI_MIN=0,MPI_SUM=1,MPI_COMM_TYPE_SHARED=0,MPI_INFO_NULL=0;
const void* const MPI_IN_PLACE=reinterpret_cast<void*>(1);
using MPI_Comm=int;
struct PreflightAllocationAbort:std::exception {};
int level_stdout=0,Matomnum=2,SpinP_switch=0,Num_Mixing_pDM=7;
int M2G[]={0,1,2},WhatSpecies[]={0,0,1},FNAN[]={0,1,1},Spe_Total_NO[]={2,3};
int neighbors[3][2]={{0,0},{1,2},{2,1}};
int* natn[]={neighbors[0],neighbors[1],neighbors[2]};
bool capacity_ok=true;
unsigned long long host_available=1ULL<<40;
int next_comm=1,live_comms=0,capacity_queries=0;
constexpr int OPENMX_GPU_COPY_HOST_TO_DEVICE=0;
static int OpenMXGpuGetLastError(){return 0;}
static int OpenMXGpuDeviceSynchronize(){return 0;}
static int OpenMXGpuMemcpy(void* to,const void* from,size_t bytes,int){std::memcpy(to,from,bytes);return 0;}
static int OpenMX_GpuMemGetInfo(size_t* available,size_t* total){*available=*total=size_t(8)<<30;return 0;}
static int gpu_rank_device_usable(){return 1;}
static unsigned long long openmx_host_mem_available(){return host_available;}
static void acc_wait_all(){}
static void acc_memcpy_from_device(void* to,const void* from,size_t bytes){std::memcpy(to,from,bytes);}
static void openmx_target_download(void* to,const void* from,size_t bytes){std::memcpy(to,from,bytes);}
static void MPI_Abort(int,int){throw PreflightAllocationAbort();}
static void MPI_Barrier(int){}
static void MPI_Comm_size(int,int* size){*size=1;}
static void MPI_Comm_split_type(int,int,int,int,int* comm){*comm=next_comm++;live_comms++;}
static void MPI_Comm_free(int* comm){check(*comm>0,"MPI group freed twice");*comm=-1;live_comms--;}
static void MPI_Allreduce(const void* from,void* to,int count,int type,int,int){
  if(from!=MPI_IN_PLACE)std::memcpy(to,from,count*(type==MPI_INT?sizeof(int):sizeof(unsigned long long)));
}
int openmx_gpu_workers_capacity(OpenMXGpuWorkers* workers,size_t items,int,
                               OpenMXGpuWorkerCost cost,void* context){
  check(context,"capacity callback has no calculation");capacity_queries++;
  if(!capacity_ok){workers->count=0;return 0;}
  workers->count=2;workers->items=items;workers->devices[0]=1;workers->devices[1]=2;
  for(int i=0;i<2;i++)check(cost(context,i,2)<SIZE_MAX,"incorrect capacity callback state");
  return 1;
}
int openmx_gpu_workers_run(const OpenMXGpuWorkers* workers,
                          int (*callback)(void*,int,int),void* context,const char*){
  const int previous=current_device,previous_acc=acc_device;
  bool ok=true;
  for(int phase=OPENMX_GPU_WORKER_PREPARE;phase<=OPENMX_GPU_WORKER_RELEASE;phase++){
    if(!ok && phase!=OPENMX_GPU_WORKER_RELEASE)continue;
    for(int i=0;i<workers->count;i++){
      current_device=acc_device=workers->devices[i];
      if(!callback(context,i,phase))ok=false;
    }
  }
  current_device=previous;acc_device=previous_acc;return ok;
}

#include "mixing_h_state.inc"
namespace {
using Context=std::unique_ptr<OpenMXCoreContext,decltype(&openmx_core_destroy)>;
Context calculation(){
  char error[256];Context c(openmx_core_create_host(error,sizeof(error)),openmx_core_destroy);
  check(bool(c),error);return c;
}
void mode(const char* value){check(!openmx_env_set("OPENMX_MIXH_GPU",value),"set mixing policy");}
void clean(MixingHState* state){
  Mixing_H_Release_GPU(state);Mixing_H_Release_GPU(state);
  check(!openmx_core_native_resource_count(state->resources),"mixing resources remain after release");
  check(state->gpu.verdict==-1 && state->gpu.last_scf_iter==-1 && state->inc.last_scf_iter==-1,
        "mixing release lost preflight/history sentinels");
  check(!state->workers.count && !state->host_ring && !state->host_weight,"worker/host history survived release");
}
void contractions(MixingHState* state){
  const size_t n=state->gpu.L;
  for(int slot=0;slot<2;slot++){
    for(size_t k=0;k<n;k++)state->gpu.h_flat[k]=(slot+1)*.125+(k%7)*.0625;
    MixH_Upload(state,slot);
  }
  for(size_t k=0;k<n;k++)state->gpu.h_flat[k]=.25+(k%3)*.125;
  MixH_Upload(state,-1);
  const int pm[]={0,1,1},pn[]={0,0,1},permutation[]={1,0};
  const double coefficient[]={.75,-.25};
  double triangles[6]={};
  MixH_Work work={};work.state=state;work.mode=MIXH_GRAM;work.dim=2;work.npair=3;
  work.pm=pm;work.pn=pn;work.pslot=permutation;work.tri=triangles;
  check(openmx_gpu_workers_run(&state->workers,MixH_Worker,&work,"test-Gram"),"Gram callback failed");
  for(int p=0;p<3;p++){
    long double expected=0;
    for(size_t k=0;k<n;k++){
      const long double a=(permutation[pm[p]]+1)*.125+(k%7)*.0625;
      const long double b=(permutation[pn[p]]+1)*.125+(k%7)*.0625;
      expected+=(.25L+(k%3)*.125L)*a*b;
    }
    check(std::abs(triangles[p]+triangles[3+p]-expected)<1e-12L*(1+std::abs(expected)),
          "permuted weighted Gram used another calculation's history");
  }
  work.mode=MIXH_RESIDUAL;work.coeff=coefficient;
  check(openmx_gpu_workers_run(&state->workers,MixH_Worker,&work,"test-residual"),"residual callback failed");
  for(size_t k=0;k<n;k++){
    const double expected=.75*(.25+(k%7)*.0625)-.25*(.125+(k%7)*.0625);
    check(state->gpu.h_r[k]==expected,"residual download used another calculation's staging buffer");
  }
}
void failures(){
  Matomnum=2;SpinP_switch=3;mode("1");
  auto other=calculation();auto* b=Mixing_H_Bind_Resources(other.get());
  check(MixH_GpuPreflight(b,1,0)==1,"second context preflight");
  const auto retained=openmx_core_native_resource_count(b->resources);
  for(long position=0;position<20;position++){
    event_count=0;auto c=calculation();auto* a=Mixing_H_Bind_Resources(c.get());
    openmx::test::fail_allocation_after(position);
    try{(void)MixH_GpuPreflight(a,1,0);}catch(const PreflightAllocationAbort&){}
    openmx::test::fail_allocation_after(-1);clean(a);
    check(openmx_core_native_resource_count(b->resources)==retained && b->workers.count==2,
          "partial allocation/release damaged another owner");
  }
  event_count=0;auto c=calculation();auto* a=Mixing_H_Bind_Resources(c.get());
  fail_status=OPENMX_GPU_OUT_OF_MEMORY;
  check(MixH_GpuPreflight(a,1,0)==1 && a->host_ring,"SDK allocation failure did not try host-backed history");
  fail_status=0;clean(a);clean(b);
}
}
int main(){try{
  for(int repeat=0;repeat<100;repeat++){
    event_count=0;capacity_ok=true;host_available=1ULL<<40;
    Matomnum=repeat%7?2:0;SpinP_switch=repeat%3==2?3:repeat%3;
    auto first=calculation(),second=calculation();
    auto* a=Mixing_H_Bind_Resources(first.get());auto* b=Mixing_H_Bind_Resources(second.get());
    check(a!=b && a->resources!=b->resources,"mixing metadata/resources shared across contexts");
    check(a->gpu.verdict==-1 && a->inc.last_scf_iter==-1,"mixing constructor lost sentinels");
    mode("1");check(MixH_GpuPreflight(a,1,0)==1 && !a->host_ring,"resident tier missing");
    const size_t expected=Matomnum?(SpinP_switch==3?175:25*(SpinP_switch+1)):0;
    check(a->gpu.L==expected,"spin/empty-rank residual layout changed");
    contractions(a);a->gpu.valid=1;a->gpu.last_scf_iter=13;
    if(repeat%2){mode("2");check(MixH_GpuPreflight(b,2,0)==2 && b->inc.nslot==7,"incremental CPU tier missing");}
    else{capacity_ok=false;check(MixH_GpuPreflight(b,2,0)==1 && b->host_ring,"host-backed tier missing");}
    const auto second_count=openmx_core_native_resource_count(b->resources);
    openmx::test::fail_allocation_after(0);
    check(Mixing_H_Bind_Resources(first.get())==a && MixH_GpuPreflight(a,1,0)==1,
          "SCF state/preflight lookup allocated or reset cached data");
    openmx::test::fail_allocation_after(-1);
    check(a->gpu.valid && a->gpu.last_scf_iter==13,"another context changed history continuity");
    current_device=acc_device=7;clean(a);
    check(current_device==7 && acc_device==7,"worker cleanup changed caller's device binding");
    check(openmx_core_native_resource_count(b->resources)==second_count,"pre-Force release touched another context");
    event_count=0;openmx::test::fail_allocation_after(0);
    openmx_core_native_resources_clear(first.get());openmx_core_finish(second.get());
    openmx::test::fail_allocation_after(-1);
    check(!live && !live_comms && current_device==7,"RAII teardown left GPU/MPI resources or changed binding");
    auto* fresh=Mixing_H_Bind_Resources(second.get());
    check(fresh->gpu.verdict==-1 && !fresh->inc.valid && !fresh->host_ring,"new input inherited stale history");
  }
  failures();
  {auto c=calculation();auto* a=Mixing_H_Bind_Resources(c.get());mode("0");
   check(MixH_GpuPreflight(a,1,0)==0,"forced legacy CPU tier changed");clean(a);
   capacity_ok=false;host_available=1;mode("1");
   check(MixH_GpuPreflight(a,1,0)==2 && a->inc.nslot==7,"host capacity rejection did not reach CPU cache");clean(a);}
  Mixing_H_Release_GPU(nullptr);check(!live && !live_comms,"mixing resources leaked");
  std::cout<<"PASS: 100 dual-context mixing lifecycles, all tiers, spin/empty layouts, "
             "real allocation failures, callback routing, weighted/permuted contractions and exact-owner cleanup\n";
  return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}}

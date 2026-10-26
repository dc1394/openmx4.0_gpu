#include "openmx_core_solver_workspace.h"
#include "openmx_core_atomic.h"
#include "fixtures/native_resources/sdk_model.hpp"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <iostream>
#include <omp.h>

// SDK-independent model; the production adapters and C++ owner are real.
static int usable=1,recovery=0,adoption_failure=0,requested=1,collective=1;
static unsigned long long host_room=1ULL<<40;
static int waits=0,clears=0,communicators=0;
static int OpenMXGpuDeviceSynchronize(){record(400);return 0;}
static int OpenMX_GpuMemGetInfo(size_t* free_bytes,size_t* total){*free_bytes=*total=1ULL<<40;return 0;}
static void acc_wait_all(){++waits;}
static void acc_clear_freelists(){++clears;}
static void acc_memcpy_to_device(void* d,const void* s,size_t n){std::memcpy(d,s,n);}
static void openmx_target_wait(){++waits;check(!OpenMXGpuDeviceSynchronize(),"HIP wait");}
static void openmx_target_clear_freelists(){}
static void openmx_target_upload(void* d,const void* s,size_t n){std::memcpy(d,s,n);}
static int gpu_rank_device_usable(){return usable;}
static int openmx_gpu_capacity_test_recovery(){return recovery;}
static unsigned long long openmx_host_mem_available(){return host_room;}
static int Force_gpu_stage_default(){return collective;}
using MPI_Comm=int;
constexpr int MPI_COMM_NULL=0,MPI_COMM_TYPE_SHARED=1,MPI_INFO_NULL=0,MPI_INT=1,MPI_MIN=1;
#define MPI_IN_PLACE ((void*)1)
static MPI_Comm mpi_comm_level1=1;
static void MPI_Allreduce(const void* in,void* out,int,int,int,MPI_Comm){if(in!=MPI_IN_PLACE)std::memcpy(out,in,sizeof(int));}
static void MPI_Comm_split_type(MPI_Comm,int,int,int,MPI_Comm* out){*out=2;++communicators;}
static void MPI_Comm_size(MPI_Comm,int* out){*out=1;}
static void MPI_Comm_rank(MPI_Comm,int* out){*out=0;}
static void MPI_Barrier(MPI_Comm){}
static void MPI_Comm_free(MPI_Comm* out){*out=MPI_COMM_NULL;--communicators;}
static int Force_collective_env_flag(const char*,int value,MPI_Comm){return value;}
static size_t Force_gpu_node_reserve(int){return 256ULL*1024*1024;}
static void* Force_gpu_arena_try(size_t bytes){
  void* p=nullptr;
#if OPENMX_GPU_HIP
  (void)OpenMXGpuMalloc(&p,bytes);
#else
  p=openmx_test_acc_malloc(bytes);
#endif
  if(p && adoption_failure)openmx::test::fail_allocation_after(0);
  return p;
}
static void OpenMX_Solver_Failure(const char* error){throw std::runtime_error(error);}
static void Force4B_gpu_abort(const char* error){throw std::runtime_error(error);}
/* OPENMX_GPU_RESOURCE_ADAPTER */

constexpr int GPUSOLVER=9;
static int scf_eigen_lib_flag=GPUSOLVER,F_NL_flag=1,SpinP_switch=0,SO_switch=0,Hub_U_switch=0;
static int Constraint_NCS_switch=0,Zeeman_NCS_switch=0,Zeeman_NCO_switch=0;
static int SpeciesNum=2,Matomnum=1,MatomnumF=1;
static int M2G[]{0,1},F_M2G[]{0,1,2},WhatSpecies[]{-1,0,1},FNAN[]{-1,1,1};
static int n1[]{1,2},n2[]{2,1};static int* natn[]{nullptr,n1,n2};
static int Spe_Total_CNO[]{2,1},radial[]{2,1},angular[2][3]{{0,0,1},{0,2,0}},relativistic[]{0,0};
static double energies[1][2][2]{{{1.25,-.75},{2,0}}};
static double values[4][3][2][3][8];
static double* rows[4][3][2][3];static double** blocks[4][3][2];
static double*** atoms[4][3];static double**** directions[4];static double***** components[]{directions};
typedef struct ForceHNLGpuContext ForceHNLGpuContext;
static void Force_HNL_GpuEnd(ForceHNLGpuContext*);
#include "openmx_host_simd.h"
/* The stage's host SIMD policy vote, recorded: the device answer keeps the
   archive paths of modes 0-15; mode 16 checks the host answer. */
static int host_vote=0,votes=0;static size_t vote_need=0,vote_budget=0;
int OpenMX_HostSimdStageOnHost(const char*,size_t need,size_t budget){++votes;vote_need=need;vote_budget=budget;return host_vote;}
#include "openmx_force_hnl_gpu.h"

namespace {
OpenMXAtomicViews atomic(){
  OpenMXAtomicViews a{};a.species.species=2;a.species.radial_projectors=radial;
  a.species.relativistic=relativistic;a.species.contracted_orbitals=Spe_Total_CNO;
  a.projector_angular=&angular[0][0];a.projector_angular_species=2;a.projector_angular_stride=3;
  a.projector_energies={&energies[0][0][0],{1,2,2}};return a;
}
void reset(int mode){
  usable=collective=1;recovery=adoption_failure=0;host_room=1ULL<<40;fail_status=0;
  current_device=acc_device=3;event_count=queries=waits=clears=0;
  scf_eigen_lib_flag=GPUSOLVER;F_NL_flag=1;SpinP_switch=SO_switch=0;relativistic[0]=0;
  Matomnum=MatomnumF=1;radial[0]=2;radial[1]=1;
  if(mode==2)recovery=1;if(mode==4)host_room=1;if(mode==5)fail_status=2;
  if(mode==6)adoption_failure=1;if(mode==7)scf_eigen_lib_flag=0;if(mode==8)collective=0;
  if(mode==9)F_NL_flag=0;if(mode==10){SpinP_switch=3;SO_switch=1;}
  if(mode==11)relativistic[0]=1;if(mode==12)usable=0;
  if(mode==13)Matomnum=MatomnumF=0;if(mode==14)radial[0]=radial[1]=0;
  host_vote=mode==16;votes=0;vote_need=vote_budget=0;
}
void packed(const ForceHNLGpuContext* g,int mode){
  check(g && g->enabled && g->resources,"missing invocation-owned archive");
  if(mode==13 || mode==14){check(!g->flat_stride && !g->c2_stride && !g->halo_count,"empty archive gained values");return;}
  check(g->flat_stride==18 && g->c2_stride==12 && g->halo_count==9,"archive sizes differ");
  check(g->sp_nlp[0]==4 && g->sp_nlp[1]==5 && g->ene_off[0]==0 && g->ene_off[1]==4 && g->ene_off[2]==9,"projector layout differs");
  check(g->mck_base[1]==0 && g->mck_off[0]==0 && g->mck_off[1]==8 && g->c2_off[0]==0 && g->c2_off[1]==8,
      "local/staged archive offsets differ");
  check(g->halo_base[1]==0 && g->halo_off[0]==0 && g->halo_off[1]==5,"halo archive offsets differ");
  const double expected[]{1.25,-.75,-.75,-.75,2,2,2,2,2};
  check(std::equal(expected,expected+9,g->ene),"expanded projector energies differ");
  for(int d=0;d<4;++d)for(int h=0;h<2;++h)for(int r=0;r<2;++r)for(int j=0;j<(h?5:4);++j)
    check(g->flat[d*18+(h?8:0)+r*(h?5:4)+j]==values[d][1][h][r][j],"local archive lost row order/stride");
  if(g->flat_dev)check(!std::memcmp(g->flat_dev,g->flat,4*18*sizeof(double)),"GPU upload differs from host archive");
  Force_HNL_GpuArchiveHalo(const_cast<ForceHNLGpuContext*>(g),components,2,2);
  for(int h=0;h<2;++h)for(int j=0;j<(h?4:5);++j)
    check(g->halo[(h?5:0)+j]==values[0][2][h][0][j],"halo archive differs");
  Force_HNL_GpuArchiveCase2(const_cast<ForceHNLGpuContext*>(g),components,1,1);
  for(int d=0;d<4;++d)for(int h=0;h<2;++h)for(int r=0;r<(h?1:2);++r)for(int j=0;j<4;++j)
    check(g->c2[d*12+(h?8:0)+r*4+j]==values[d][2][h][r][j],"staged archive differs");
}
int nested(OpenMXSolverInvocation* scope,void* outer,double* result){
  auto a=atomic();ForceHNLGpuContext* g=nullptr;
  check(Force_HNL_GpuBegin(&a,components,scope,&g),"nested GPU archive declined");
  check(g!=outer,"nested invocation reused outer archive state");packed(g,0);*result=1;return 1;
}
int operation(OpenMXSolverInvocation* scope,void* opaque,double* result){
  const int mode=*static_cast<int*>(opaque);auto a=atomic();ForceHNLGpuContext* g=nullptr;
  const int ok=Force_HNL_GpuBegin(&a,components,scope,&g);
  if(mode>=7 && mode<=11){check(!ok && !g && !queries && !waits && !communicators && !votes,"disabled stage touched GPU state");*result=1;return 1;}
  if(mode==4 || mode==16){check(!ok && !g,"host capacity decline published state");
    check(!openmx_core_native_resource_count(OpenMX_Solver_Resources(scope,"force-hnl-archives")),"capacity decline retained partial host archive");
    check(votes==(mode==16),"host capacity decline reached the policy vote");
    *result=1;return 1;
  }
  /* need = the archives (4*18 local + 9 halo + 4*12 staged doubles) plus the
     128 MiB margin; budget = free minus the node reserve; the capacity test
     and an unusable device keep their recovery paths without a figure */
  const size_t archives=(mode==13 || mode==14?0:(4*18+9+4*12)*sizeof(double))+134217728;
  check(votes==1 && vote_need==(mode==2?0:archives),"policy vote need differs");
  check(vote_budget==(mode==2 || mode==12?0:(size_t(1)<<40)-256ULL*1024*1024),"policy vote budget differs");
  check(ok,"GPU archive preparation declined");packed(g,mode);
  const bool resident=mode!=2 && mode!=5 && mode!=6 && mode!=12;
  check(!!g->flat_dev==resident,"GPU capacity/adoption fallback changed");
  check(openmx_core_native_resource_count(g->resources)==size_t(11+resident),"archive allocations not all owned");
  if(mode==15){
    char error[256];double output;check(openmx_core_solver_run(1,nested,g,&output,error,sizeof(error)),error);
    check(output==1,"nested callback result differs");packed(g,0);
  }
  if(mode==0){
    auto* saved=g;check(Force_HNL_GpuBegin(&a,components,scope,&g) && g==saved,"repeated preparation moved metadata");
    packed(g,0);check(openmx_core_native_resource_count(g->resources)==12,"repeated preparation leaked buffers");
    Force_HNL_GpuEnd(g);Force_HNL_GpuEnd(g);
    check(!g->enabled && !g->flat && !g->flat_dev && !openmx_core_native_resource_count(g->resources),"explicit archive release leaked");
  }
  current_device=acc_device=7;
  if(mode==2)return 0;if(mode==3)throw std::runtime_error("injected native archive exception");
  *result=1;return 1;
}
}
int main(){
  try{
    for(int d=0;d<4;++d){directions[d]=atoms[d];for(int a=0;a<3;++a){atoms[d][a]=blocks[d][a];
      for(int h=0;h<2;++h){blocks[d][a][h]=rows[d][a][h];for(int r=0;r<3;++r){rows[d][a][h][r]=values[d][a][h][r];
        for(int j=0;j<8;++j)values[d][a][h][r][j]=10000*d+1000*a+100*h+10*r+j+.125;}}}}
    omp_set_dynamic(0);int comparisons=0;
    for(int threads:{1,2,4,8}){omp_set_num_threads(threads);for(int mode=0;mode<17;++mode){
      reset(mode);char error[256];double result=31;
      const int ok=openmx_core_solver_run(threads,operation,&mode,&result,error,sizeof(error));
      check(ok==(mode!=2 && mode!=3),error);check(result==(ok?1:31),"failed archive stage published output");
      check(!live && !communicators,"archive scope leaked SDK allocation or communicator");
      if(mode==1 || mode==3 || mode==15)check(current_device==7 && acc_device==7,"RAII release changed caller's device");
      ++comparisons;
    }}
    omp_set_num_threads(1);reset(1);int mode=1;char error[256];double result;
    openmx::test::fail_allocation_after(10000);check(openmx_core_solver_run(1,operation,&mode,&result,error,sizeof(error)),error);
    const auto count=10000-openmx::test::remaining_allocations();openmx::test::fail_allocation_after(-1);
    for(long fail=0;fail<count;++fail){
      reset(1);const auto before=openmx::test::live_allocations();result=31;openmx::test::fail_allocation_after(fail);
      const int ok=openmx_core_solver_run(1,operation,&mode,&result,error,sizeof(error));openmx::test::fail_allocation_after(-1);
      // Failure of the optional GPU registration is a valid host fallback.
      check((ok && result==1) || (!ok && result==31),"failed archive preparation published output");
      check(!live && openmx::test::live_allocations()==before,"partial GPU archive preparation leaked");
    }
    std::cout<<"PASS: "<<comparisons<<" actual nonlocal GPU archive cases; "<<count<<" C++ allocation failures; backend="<<OPENMX_GPU_HIP<<'\n';return 0;
  }catch(const std::exception& e){openmx::test::fail_allocation_after(-1);std::cerr<<e.what()<<'\n';return 1;}
}

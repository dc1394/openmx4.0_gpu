#include "openmx_core_solver_workspace.h"
#include "fixtures/native_resources/sdk_model.hpp"
#include "openmx_env.h"
#include <cstdio>
#include <cstring>
#include <iostream>

namespace {
int held,denied,inject,low_memory,allocation,copies,fail_allocation,fail_copy,fail_sync,fail_info;
int weak_fp64,device_query_failure,usable=1,MYID_MPI_COMM_WORLD;
int admission_token;
constexpr int cudaSuccess=0,cudaMemcpyHostToDevice=1;
int gpu_rank_device_usable(){return usable;}
int openmx_gpu_capacity_test_recovery(){return inject;}
int openmx_gpu_local_size_noncollective(){return 18;}
int cudaGetDevice(int *device){*device=current_device;return device_query_failure;}
int openmx_gpu_device_weak_fp64(int){return weak_fp64;}
int cudaGetLastError(){return 0;}
int cudaDeviceSynchronize(){check(held,"sync without admission");return fail_sync;}
int OpenMXGpuDeviceSynchronize(){check(held,"RAII drain lost admission");record(401);return 0;}
void acc_wait_all(){check(held,"wait without admission");}
void acc_clear_freelists(){check(held,"clear without admission");}
int OpenMX_GpuMemGetInfo(size_t *free,size_t *total){
  check(held,"query without admission");*free=*total=low_memory?0:1073741824;return fail_info;
}
void *openmx_mgpu_admit_devices(const int *device,int n){
  check(n==1 && *device==3,"wrong admission device");
  if(denied || held)return nullptr;held=1;return &admission_token;
}
void openmx_mgpu_release_devices(void *token){
  check(token==&admission_token && held && !live,"admission released before archive storage");held=0;record(501);
}
int cudaMalloc(void **pointer,size_t bytes){
  check(held,"allocation without admission");*pointer=nullptr;
  return ++allocation==fail_allocation?2:OpenMXGpuMalloc(pointer,bytes);
}
int cudaFree(void *pointer){check(held,"free without admission");return OpenMXGpuFree(pointer);}
int cudaMemcpy(void *to,const void *from,size_t bytes,int kind){
  check(held && kind==cudaMemcpyHostToDevice,"copy without admission");
  if(++copies==fail_copy)return 1;std::memcpy(to,from,bytes);return 0;
}
void Force4B_GpuWorkspaceSizes(size_t *a,size_t *b){*a=4096;*b=8192;}
/* OPENMX_FORCE_ARCHIVE_PRODUCTION */

struct Case {int kind,scenario,mode;};
int operation(OpenMXSolverInvocation *scope,void *opaque,double *result){
  auto c=*static_cast<Case*>(opaque);char error[256];OpenMXNativeResources *resources=nullptr;
  check(openmx_core_solver_resources(scope,"force-archive-lease-test",&resources,error,sizeof(error)),error);
  float flat[4*37],halo[29],c2[4*43];
  for(int i=0;i<4*37;++i)flat[i]=i+.125f;
  for(int i=0;i<29;++i)halo[i]=i-.25f;
  for(int i=0;i<4*43;++i)c2[i]=.5f-i;
  Force4BGpuContext g={};g.resources=resources;g.flat_host=flat;g.halo_host=halo;g.c2_host=c2;
  g.flat_stride=37;g.halo_count=29;g.c2_stride=43;
  if(c.scenario==8){g.flat_dev=flat;g.halo_dev=halo;g.c2_dev=c2;}
  if(c.scenario==11)g.flat_host=g.c2_host=nullptr;
  Force4BArchiveLease *lease=nullptr;
  const int ready=Force4B_ArchiveBegin(&g,c.kind,&lease);
  if(c.scenario>=0)check(ready==(c.scenario==0),"archive admission policy changed");
  if(ready){
    check(lease && held && live==(c.kind==1?2:1),"archive storage escaped lease");
    check(openmx_core_native_resource_count(resources)==1,"lease not owned by invocation");
    check(!std::memcmp(lease->first,c.kind==1?flat:c2,c.kind==1?sizeof(flat):sizeof(c2)),"first upload changed");
    if(c.kind==1)check(!std::memcmp(lease->second,halo,sizeof(halo)),"halo upload changed");
    Force4BArchiveLease *duplicate=lease;
    check(!Force4B_ArchiveBegin(&g,c.kind,&duplicate) && !duplicate,"active lease overwritten");
  }else check(!lease && !held && !live && !openmx_core_native_resource_count(resources),"declined archive leaked");
  if(c.scenario==8)check(g.flat_dev==flat && g.halo_dev==halo && g.c2_dev==c2,"resident context changed");
  else check(!g.flat_dev && !g.halo_dev && !g.c2_dev,"temporary pointers published into resident context");
  if(c.mode==0){
    Force4B_ArchiveEnd(lease);Force4B_ArchiveEnd(lease);
    check(!held && !live && !openmx_core_native_resource_count(resources),"explicit cleanup leaked");
    if(ready){
      Force4BArchiveLease *again=nullptr;check(Force4B_ArchiveBegin(&g,c.kind,&again),"lease not reusable");
      check(again==lease,"lease descriptor moved");Force4B_ArchiveEnd(again);
    }
  }
  current_device=acc_device=7;
  if(c.mode==2)return 0;if(c.mode==3)throw std::runtime_error("injected archive callback exception");
  *result=1;return 1;
}
void reset(int scenario){
  check(!held && !live,"previous archive scope leaked");
  current_device=acc_device=3;event_count=0;usable=1;device_query_failure=0;
  denied=inject=low_memory=allocation=copies=fail_allocation=fail_copy=fail_sync=fail_info=0;
  if(scenario==1)denied=1;if(scenario==2)low_memory=1;if(scenario==3)inject=1;
  if(scenario==4)fail_allocation=1;if(scenario==5)fail_allocation=2;
  if(scenario==6)fail_copy=1;if(scenario==7)fail_copy=2;
  if(scenario==9)fail_sync=1;if(scenario==10)fail_info=1;
  if(scenario==12)usable=0;if(scenario==13)device_query_failure=1;
}
}
int main(){
  try{
    openmx_env_reset();int policy_cases=0,cases=0,failures=0;
    const char *controls[]={nullptr,"auto","0","1"};
    for(int setting=0;setting<4;++setting)for(int resident=0;resident<2;++resident)
    for(usable=0;usable<2;++usable)for(weak_fp64=0;weak_fp64<2;++weak_fp64)
    for(inject=0;inject<2;++inject)for(device_query_failure=0;device_query_failure<2;++device_query_failure){
      check(!openmx_env_set("OPENMX_FORCE4B_ARCHIVE_GPU",controls[setting]?controls[setting]:"auto"),"override failed");
      const int expected=resident || inject || (setting==2?0:setting==3?1:!usable || device_query_failure || !weak_fp64);
      check(Force4B_ArchiveGpuPreferred(resident)==expected,"GeForce/datacenter archive policy changed");++policy_cases;
    }
    check(!openmx_env_unset("OPENMX_FORCE4B_ARCHIVE_GPU"),"clear override failed");
    for(int kind=1;kind<=2;++kind)for(int scenario=0;scenario<14;++scenario)for(int mode=0;mode<4;++mode){
      if(kind==2 && (scenario==5 || scenario==7))continue;
      reset(scenario);Case c{kind,scenario,mode};char error[256];double result=31;
      const auto before=openmx::test::live_allocations();
      const int ok=openmx_core_solver_run(1,operation,&c,&result,error,sizeof(error));
      check(ok==(mode<2) && result==(ok?1:31),error);
      check(!held && !live && openmx::test::live_allocations()==before,"archive scope leaked");
      check(current_device==7 && acc_device==7,"archive cleanup changed caller device");++cases;
    }
    for(int kind=1;kind<=2;++kind){
      reset(0);Case c{kind,-1,1};char error[256];double result;
      openmx::test::fail_allocation_after(10000);check(openmx_core_solver_run(1,operation,&c,&result,error,sizeof(error)),error);
      const auto count=10000-openmx::test::remaining_allocations();openmx::test::fail_allocation_after(-1);
      for(long fail=0;fail<count;++fail){
        reset(0);const auto before=openmx::test::live_allocations();result=31;
        openmx::test::fail_allocation_after(fail);const int ok=openmx_core_solver_run(1,operation,&c,&result,error,sizeof(error));
        openmx::test::fail_allocation_after(-1);
        check(result==(ok?1:31),"failed archive setup published result");
        check(!held && !live && openmx::test::live_allocations()==before,"partial archive setup leaked");++failures;
      }
    }
    std::cout<<"PASS: "<<cases<<" archive lifetime cases; "<<policy_cases<<" hardware/override policies; "
        <<failures<<" C++ allocation failure sites\n";return 0;
  }catch(const std::exception& e){openmx::test::fail_allocation_after(-1);std::cerr<<e.what()<<'\n';return 1;}
}

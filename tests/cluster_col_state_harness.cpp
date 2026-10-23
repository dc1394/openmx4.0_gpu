#include "openmx_core_chemical.h"
#include "fixtures/native_resources/sdk_model.hpp"
#include "openmx_cluster_col_state.h"
#include "openmx_core_energy.h"
#include "openmx_env.h"
#include "openmx_precision.h"
#include <cstdint>
#include <cstring>
#include <memory>
#include <iostream>
#include <algorithm>
#include <climits>
using cudaStream_t=void*;
using cublasHandle_t=void*;
using cusolverDnHandle_t=void*;
using cusolverDnParams_t=void*;
using hipStream_t=void*;
using hipblasHandle_t=void*;
using cudaError_t=int;
using hipError_t=int;
constexpr int cudaSuccess=0,hipSuccess=0,cudaErrorMemoryAllocation=2;
constexpr double WAITTIME=0.001;
constexpr int CLUSTER_GPU_CPU_SWITCH_NUM=400;
constexpr int mpi_comm_level1=0,Host_ID=0,MPI_INT=0,MPI_MAX=0;
int Solver=2,gpusolver2_flag=0,level_stdout=0,Cnt_switch=0,DFTSCF_loop=50,xanes_calc=0,
    empty_occupation_flag=0,empty_states_flag=0;
int policy_iterations=2,grid_drains=0,prepared_drains=0,refine_device_drains=0;
static int cudaGetDevice(int* d){return OpenMXGpuGetDevice(d);}
static int cudaSetDevice(int d){return OpenMXGpuSetDevice(d);}
static int cudaGetLastError(){return 0;}
static int hipGetLastError(){return 0;}
static int cudaStreamSynchronize(void* s){return OpenMXGpuStreamSynchronize(s);}
static void MPI_Comm_rank(int,int* rank){*rank=0;}
static void MPI_Bcast(void*,int,int,int,int){}
static void MPI_Allreduce(const void* input,void* output,int count,int,int,int){std::memcpy(output,input,count*sizeof(int));}
static void MPI_Abort(int,int){throw std::runtime_error("unexpected production abort");}
static double MPI_Wtime(){static double t=0;return t+=0.1;}
static void openmx_gs2_set_fallback_report(int){}
static void openmx_gs2_grid_free(int){grid_drains++;}
static void Cfree_blacs_system_handle(int){}
static void Cblacs_gridexit(int){}
static void openmx_gemmul8ReleasePreparedOwner(const void*){prepared_drains++;}
static int openmx_eigen_refine_configure(int,int,int* ratio){*ratio=0;return policy_iterations;}
static double openmx_eigen_refine_until(){return 1e-5;}
static void refine_check(int status,const char* message){check(status==0,message);}
#define wait_cudafunc(status) refine_check(status,"CUDA operation")
#define wait_hipfunc(status) refine_check(status,"HIP operation")
#define OpenMXBlasWait(status) refine_check(status,"BLAS operation")
#define OpenMXSolverWait(status) refine_check(status,"solver operation")
// The refinement library's private SDK handle cleanup is modeled; its state
// buffer cleanup below is the production implementation, not a replacement.
#include "cluster_col_state.inc"
namespace {
using Context=std::unique_ptr<OpenMXCoreContext,decltype(&openmx_core_destroy)>;
Context calculation(){char error[256];Context c(openmx_core_create_host(error,sizeof(error)),openmx_core_destroy);check(bool(c),error);return c;}
template<class T> void device_buffer(ClusterColState* s,T** p){check(!OpenMX_Resource_GpuMalloc(s->resources,reinterpret_cast<void**>(p),64),"device allocation");}
template<class T> T* host_buffer(ClusterColState* s){auto* p=static_cast<T*>(openmx_core_native_malloc(s->resources,1,64,1));check(p,"host allocation");return p;}
void populate(ClusterColState* s,int device){
    current_device=device;auto& g=s->gpusolver_ctx;g.initialized=1;g.device_id=device;g.matrix_dim=4;
    device_buffer(s,&g.d_S);device_buffer(s,&g.d_H);device_buffer(s,&g.d_tmp);device_buffer(s,&g.d_W);
    device_buffer(s,&g.d_info);device_buffer(s,&g.d_work);
    check(!OpenMX_Resource_StreamCreate(s->resources,&g.stream,0),"stream creation");
#if OPENMX_GPU_HIP
    check(!OpenMX_Resource_BlasCreate(s->resources,&g.hipblas),"HIP BLAS creation");
    s->SerialOverlap=host_buffer<double>(s);s->SerialOverlapN=4;
    s->DistributedOverlapN=4;s->DeviceEvecSpin=1;
    s->root_dm_workspace.occ=host_buffer<double>(s);s->root_dm_workspace.occ_e=host_buffer<double>(s);
    s->root_dm_workspace.pocc=host_buffer<double>(s);s->root_dm_workspace.dm_buffer=host_buffer<double>(s);
    s->CachedGpuSolverDenseN=4;s->CachedGpuSolverDenseMaxN=3;
#else
    check(!OpenMX_Resource_BlasCreate(s->resources,&g.cublas),"CUDA BLAS creation");
    check(!OpenMX_Resource_SolverCreate(s->resources,&g.gpusolver),"CUDA solver creation");
    g.h_work=host_buffer<double>(s);
    s->dense_index_cache.dense_index=host_buffer<int>(s);s->dense_index_cache.tnum=4;s->dense_index_cache.valid=1;
    s->cpu_overlap_n=4;
#endif
    for(int spin=0;spin<2;spin++){
        auto& r=s->refine_state[spin];r.resources=s->resources;r.capacity=4;r.basis_valid=1;r.basis_n=4;
        device_buffer(s,&r.lam);device_buffer(s,&r.occ);device_buffer(s,&r.own);device_buffer(s,&s->refine_x[spin]);
#if OPENMX_GPU_HIP
        s->refine_h[spin]=host_buffer<double>(s);s->refine_h_valid[spin]=1;
        s->CachedGpuSolverDenseEVec[spin]=host_buffer<double>(s);s->CachedGpuSolverDenseValid[spin]=1;
#else
        device_buffer(s,&g.d_evec_stash[spin]);g.evec_stash_valid[spin]=1;
#endif
    }
    s->refine_x_n=4;s->dm_entry_cache.basis0=host_buffer<int>(s);s->dm_entry_cache.basis1=host_buffer<int>(s);
    s->dm_entry_cache.valid=1;s->gs2_fallback_n=4;s->mgpu_host_dense=1;
    s->ugrid.n=4;s->ugrid.Ss=host_buffer<double>(s);s->ugrid.Hs=host_buffer<double>(s);s->ugrid.Cs=host_buffer<double>(s);
}
void policy(ClusterColState* a,ClusterColState* b){
    check(a->refine.iterations==-1 && b->refine.iterations==-1 && a->refine.peer==-1,"initial refinement sentinels");
    openmx_set_fp32_enabled(0);ClusterCol_RefineConfigure(a);check(a->refine.iterations==-1,"FP32 veto was cached");
    openmx_set_fp32_enabled(1);policy_iterations=2;ClusterCol_RefineConfigure(a);
    policy_iterations=1;ClusterCol_RefineConfigure(b);
    check(ClusterCol_RefineIterations(a)==2 && ClusterCol_RefineIterations(b)==1,"contexts share refinement policy");
    a->refine.scf_mode=1;ClusterCol_RefineBeginSolve(a,1,0,-1);
    check(Cluster_DFT_Col_RefineLastRefined(a)==1 && !Cluster_DFT_Col_RefineLastRefined(b),"refinement solve controls leaked");
    a->refine.solved[0]=1;ClusterCol_RefineEndSolve(a);Cluster_DFT_Col_RefineAfterMixing(a,1e-6);
    check(a->refine.final_stage && !b->refine.final_stage,"mixing stage affected another calculation");
    Cluster_DFT_Col_RefineForceFP64(b,1);check(b->refine.force_fp64 && !a->refine.force_fp64,"FP64 control affected another calculation");
    Cluster_DFT_Col_RefineForceFP64(b,0);
}
void callbacks(ClusterColState* a,ClusterColState* b){
    EigenRefineDevice device{};device.allocation_context=a;device.try_malloc_owned=ClusterCol_RefineDeviceMalloc;
    device.host_malloc_owned=ClusterCol_RefineHostAllocate;
    const auto before=openmx_core_native_resource_count(a->resources);current_device=3;
    void* pointer=nullptr;check(!openmx_eigen_refine_malloc(&device,&pointer,32),"owned allocation callback failed");
    check(openmx_core_native_resource_count(a->resources)==before+1 && !openmx_core_native_resource_count(b->resources),"callback registered with another owner");
    check(!OpenMX_Resource_GpuFree(a->resources,pointer),"callback memory release failed");
    void* host=openmx_eigen_refine_host_malloc(&device,4,8,"owned host");check(host,"host callback failed");
    openmx_core_native_free(a->resources,host);
    fail_status=2;pointer=reinterpret_cast<void*>(17);check(openmx_eigen_refine_malloc(&device,&pointer,32)==2 && !pointer,"SDK callback failure lost status/output");fail_status=0;
    openmx::test::fail_allocation_after(0);
    int status=openmx_eigen_refine_malloc(&device,&pointer,32);
    const auto pending=openmx::test::remaining_allocations();openmx::test::fail_allocation_after(-1);
    if (status!=2 && OPENMX_GPU_HIP) std::cerr<<"registration fault diagnostic: status="<<status<<" pointer="<<pointer<<" pending="<<pending<<" live="<<live<<" resources="<<openmx_core_native_resource_count(a->resources)<<"\n";
#if OPENMX_GPU_HIP
    check(status==2 && !pointer,"HIP callback registration failure changed bounded allocation");
#else
    // Native CUDA policy retries a transient registration failure, as it
    // retries a transient SDK failure. The failed attempt must be released.
    check(status==0 && pointer && live==1,"CUDA callback lost bounded retry");
    check(!OpenMX_Resource_GpuFree(a->resources,pointer),"retry result release failed");
#endif
    check(openmx_core_native_resource_count(a->resources)==before && !live,"registration OOM leaked callback storage");
}
}
OpenMXChemicalPotentials* owned_chemical(OpenMXCoreContext* c) {
  OpenMXChemicalPotentials* out=nullptr;char error[256];
  check(openmx_core_chemical_potentials(c,&out,error,sizeof(error)),error);return out;
}
int main(){try{
    for(const auto* value: {"", "0", "-1", "27", "invalid", "400"}) {
        setenv("OPENMX_CLUSTER_GPU_SWITCH_NUM",value,1);
        check(Cluster_DFT_Col_GpuSwitchNum()==(!std::strcmp(value,"27")?27:400),"GPU crossover input/default changed");
    }
    unsetenv("OPENMX_CLUSTER_GPU_SWITCH_NUM");
    for(int cycle=0;cycle<100;cycle++){
        event_count=0;queries=0;auto first=calculation();auto second=calculation();
        auto* a=Cluster_DFT_Col_Bind_Resources(first.get());auto* b=Cluster_DFT_Col_Bind_Resources(second.get());
    check(a->chemical==owned_chemical(first.get()) && b->chemical==owned_chemical(second.get()) && a->chemical!=b->chemical,
          "native solver borrowed another calculation's chemical potentials");
    a->chemical->chemical_potential=.125;a->chemical->xanes[0]=1.5;
    b->chemical->chemical_potential=-.25;b->chemical->xanes[1]=2.5;
        check(a!=b && a->resources!=b->resources && !queries && !live,"binding touched SDK/shared a state");
        check(a->firsttime==1 && b->firsttime==1,"first-call sentinel lost");
#if OPENMX_GPU_HIP
        check(a->owners_on_device==1 && a->DeviceEvecSpin==-1 && a->forward_unblocked==-1 && a->measure_time_enabled==-1,"HIP sentinel initialization changed");
#else
        check(a->diag_verdict==-1 && !a->diag_verdict_n && !a->transformed_s_serial,"CUDA sentinel initialization changed");
#endif
        policy(a,b);callbacks(a,b);
        a->sample_geometry=71;
        openmx::test::fail_allocation_after(0);auto* again=Cluster_DFT_Col_Bind_Resources(first.get());openmx::test::fail_allocation_after(-1);
        check(again==a && a->sample_geometry==71 && !b->sample_geometry,"SCF lookup allocated/reset/shared metadata");
        populate(a,1);populate(b,2);const auto second_count=openmx_core_native_resource_count(b->resources);
        current_device=7;Cluster_DFT_Col_Release_GPU_Solver(a);
        check(!a->gpusolver_ctx.initialized && !a->refine_x_n && current_device==7,"pre-Force release changed device/left solver live");
        check(openmx_core_native_resource_count(b->resources)==second_count && b->gpusolver_ctx.initialized && b->refine_state[0].basis_valid,"pre-Force release invalidated another calculation");
        ClusterCol_ReleaseCaches(a);ClusterCol_ReleaseCaches(a);
        check(!openmx_core_native_resource_count(a->resources) && a->refine.iterations==-1 && a->refine.peer==-1,"run-boundary cleanup leaked/did not restore policy");
        event_count=0;openmx::test::fail_allocation_after(0);openmx_core_native_resources_clear(first.get());
    check(owned_chemical(first.get())->chemical_potential==0 && owned_chemical(first.get())->xanes[0]==0 &&
          b->chemical->chemical_potential==-.25 && b->chemical->xanes[1]==2.5,
          "clearing a calculation retained chemical potentials or reset another calculation");openmx::test::fail_allocation_after(-1);
        check(openmx_core_native_resource_count(b->resources)==second_count && b->refine_state[0].basis_valid,"first context destruction touched second");
        openmx_core_native_resources_clear(first.get());first.reset();
        event_count=0;openmx::test::fail_allocation_after(0);openmx_core_finish(second.get());openmx::test::fail_allocation_after(-1);
        check(!live && current_device==7,"context destruction leaked SDK state/changed caller device");
        openmx_core_finish(second.get());auto* fresh=Cluster_DFT_Col_Bind_Resources(second.get());
        check(fresh->firsttime==1 && fresh->refine.iterations==-1 && !fresh->sample_geometry,"next input inherited stale metadata");
    }
    check(grid_drains==200,"union grids not destroyed exactly once");
    std::cout<<"PASS: 100 dual-context calculation lifecycles; SCF reuse, allocation callback failures, refinement/FP64 stages, pre-Force release, union grids and exact-owner teardown\n";return 0;
}catch(const std::exception& e){openmx::test::fail_allocation_after(-1);std::cerr<<e.what()<<'\n';return 1;}}

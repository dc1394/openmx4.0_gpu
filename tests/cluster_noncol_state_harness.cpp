#include "openmx_core_chemical.h"
#include "fixtures/native_resources/sdk_model.hpp"
#include "hamiltonian_cache_binding_fixture.hpp"
#include "orbital_state_binding_fixture.hpp"
#include "openmx_cluster_noncol_state.h"
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

int unmaps=0;
constexpr int MPI_COMM_WORLD=0;
static void MPI_Initialized(int* initialized) { *initialized=1; }
static void acc_unmap_data(void*) { unmaps++; }
constexpr int hipHostMallocDefault=0,hipMemcpyDeviceToHost=1;
static int hipMemcpy(void* to,const void* from,size_t bytes,int) { std::memcpy(to,from,bytes);return 0; }
static int hipDeviceSynchronize(){return 0;}
static int cudaDeviceSynchronize(){return 0;}
static double MPI_Wtime();
static void dtime(double* time){*time=MPI_Wtime();}
#include "cluster_noncol_state.inc"
namespace {
using Context=std::unique_ptr<OpenMXCoreContext,decltype(&openmx_core_destroy)>;
Context calculation(){char error[256];Context c(openmx_core_create_host(error,sizeof(error)),openmx_core_destroy);check(bool(c),error);return c;}
template<class T> void device_buffer(ClusterNonColState* s,T** p){check(!OpenMX_Resource_GpuMalloc(s->resources,reinterpret_cast<void**>(p),512),"device allocation");}
template<class T> T* host_buffer(ClusterNonColState* s){auto* p=static_cast<T*>(openmx_core_native_malloc(s->resources,1,512,1));check(p,"host allocation");return p;}
void populate(ClusterNonColState* s){
    current_device=0;
    s->refine_state.resources=s->resources;s->refine_state.capacity=2;s->refine_state.basis_valid=1;s->refine_state.basis_n=2;
    device_buffer(s,&s->refine_state.lam);device_buffer(s,&s->refine_state.occ);device_buffer(s,&s->refine_state.own);
    s->dm_entry_cache.basis0=host_buffer<int>(s);s->dm_entry_cache.basis1=host_buffer<int>(s);
    s->root_dm_workspace.occ=host_buffer<double>(s);s->root_dm_workspace.occ_e=host_buffer<double>(s);s->root_dm_workspace.dm_buffer=host_buffer<double>(s);
    s->gs2_fallback_n2=2;s->cpu_fallback_n2=2;s->mgpu_host_dense=1;
#if OPENMX_GPU_HIP
    auto& d=s->dsolve;d.n=1;d.n2=2;d.x_on_device=1;
    device_buffer(s,&d.d_A);device_buffer(s,&d.d_S);device_buffer(s,&d.d_arena);device_buffer(s,&d.d_x);device_buffer(s,&d.d_w);device_buffer(s,&d.d_info);device_buffer(s,&d.d_work);
    check(!OpenMX_Resource_BlasCreate(s->resources,&d.hipblas),"BLAS allocation");
    s->host_stage.S=host_buffer<double>(s);s->host_stage.n=1;s->host_stage.S_valid=1;
    s->host_stage.evec=host_buffer<dcomplex>(s);s->CachedGpuSolverDenseEVec=s->host_stage.evec;s->CachedGpuSolverDenseN2=2;s->CachedGpuSolverDenseLd=2;
#else
    s->dense_arena.valid=1;s->dense_arena.n=1;s->dense_arena.n2=2;device_buffer(s,&s->dense_arena.base);
    s->dense_scache.S=host_buffer<double>(s);s->dense_scache.s_on_device=1;s->dense_scache.transformed_s_valid=1;
    s->root_solve_workspace.Cs=host_buffer<double>(s);s->root_solve_workspace.rHs11=host_buffer<double>(s);
    s->dense_index_cache.dense_index=host_buffer<int>(s);s->dense_index_cache.device_valid=1;
    s->CachedGpuSolverDenseEVec=host_buffer<dcomplex>(s);s->CachedGpuSolverDenseN2=2;s->CachedGpuSolverDenseOnDevice=1;
    auto& d=s->gpu_eigen_ctx;d.inited=1;device_buffer(s,&d.d_work);device_buffer(s,&d.d_info);d.h_work=host_buffer<double>(s);
    check(!OpenMX_Resource_StreamCreate(s->resources,&d.stream,0),"stream allocation");check(!OpenMX_Resource_SolverCreate(s->resources,&d.handle),"solver allocation");
#endif
}
}
OpenMXChemicalPotentials* owned_chemical(OpenMXCoreContext* c) {
  OpenMXChemicalPotentials* out=nullptr;char error[256];
  check(openmx_core_chemical_potentials(c,&out,error,sizeof(error)),error);return out;
}
int main(){try{
 for(int cycle=0;cycle<100;cycle++){
  event_count=0;queries=0;auto first=calculation();auto second=calculation();
  auto* a=Cluster_DFT_NonCol_Bind_Resources(first.get());auto* b=Cluster_DFT_NonCol_Bind_Resources(second.get());
    check(a->chemical==owned_chemical(first.get()) && b->chemical==owned_chemical(second.get()) && a->chemical!=b->chemical,
          "native solver borrowed another calculation's chemical potentials");
    check(a->hamiltonian_cache==Set_Hamiltonian_Cache_Bind_Resources(first.get()) &&
          b->hamiltonian_cache==Set_Hamiltonian_Cache_Bind_Resources(second.get()) &&
          a->hamiltonian_cache!=b->hamiltonian_cache,
          "solver borrowed another calculation's packed Hamiltonian cache");
    check(a->orbital_state==Set_Orbitals_Grid_GPU_Bind_Resources(first.get()) &&
          b->orbital_state==Set_Orbitals_Grid_GPU_Bind_Resources(second.get()) &&
          a->orbital_state!=b->orbital_state,"solver borrowed another calculation's orbital tables");
    a->chemical->chemical_potential=.125;a->chemical->xanes[0]=1.5;
    b->chemical->chemical_potential=-.25;b->chemical->xanes[1]=2.5;
  check(a!=b && a->resources!=b->resources && !queries && !live,"binding shared metadata/touched SDK");
  check(a->firsttime==1 && a->refine.iterations==-1 && b->refine.iterations==-1,"initial sentinels changed");
  policy_iterations=2;openmx_set_fp32_enabled(1);ClusterNonCol_RefineConfigure(a);policy_iterations=1;ClusterNonCol_RefineConfigure(b);
  check(ClusterNonCol_RefineIterations(a)==2 && ClusterNonCol_RefineIterations(b)==1,"configuration crossed calculations");
  a->refine.cycle_refined=1;Cluster_DFT_NonCol_RefineAfterMixing(a,1e-6);check(a->refine.final_stage && !b->refine.final_stage,"mixing stage crossed calculations");
  openmx::test::fail_allocation_after(0);auto* again=Cluster_DFT_NonCol_Bind_Resources(first.get());openmx::test::fail_allocation_after(-1);check(again==a && a->refine.final_stage,"SCF binding allocated/reset state");
  populate(a);populate(b);const auto other=openmx_core_native_resource_count(b->resources);auto* host=a->CachedGpuSolverDenseEVec;
  const auto hooks=OpenMX_ClusterNonCol_DensityHooks(a);check(hooks.context==a && hooks.demote_eigenvectors,"density hook lost explicit owner");hooks.demote_eigenvectors(hooks.context);
  check(a->CachedGpuSolverDenseEVec==host && a->CachedGpuSolverDenseN2==2 && b->CachedGpuSolverDenseN2==2 && openmx_core_native_resource_count(b->resources)==other,"density pressure invalidated another calculation/host stash");
#if OPENMX_GPU_HIP
  check(!a->dsolve.d_A && b->dsolve.d_A,"pressure used the wrong retained solver");
#else
  check(!a->CachedGpuSolverDenseOnDevice && b->CachedGpuSolverDenseOnDevice,"pressure used the wrong mapped vectors");
#endif
  Cluster_DFT_NonCol_Release_GPU_Solver(a);ClusterNonCol_ReleaseCaches(a);ClusterNonCol_ReleaseCaches(a);
  check(!openmx_core_native_resource_count(a->resources) && a->refine.iterations==-1,"explicit cleanup left resources/policy");
  openmx::test::fail_allocation_after(0);openmx_core_native_resources_clear(first.get());
    check(owned_chemical(first.get())->chemical_potential==0 && owned_chemical(first.get())->xanes[0]==0 &&
          b->chemical->chemical_potential==-.25 && b->chemical->xanes[1]==2.5,
          "clearing a calculation retained chemical potentials or reset another calculation");openmx::test::fail_allocation_after(-1);
  check(openmx_core_native_resource_count(b->resources)==other && b->refine.iterations==1,"clearing first touched second state");
  event_count=0;openmx::test::fail_allocation_after(0);openmx_core_finish(second.get());openmx::test::fail_allocation_after(-1);check(!live,"native teardown leaked");
  openmx_core_finish(second.get());auto* fresh=Cluster_DFT_NonCol_Bind_Resources(second.get());check(fresh->refine.iterations==-1 && fresh->firsttime==1,"new input inherited metadata");
 }
 std::cout<<"PASS: 100 dual-context noncollinear lifecycles, refinement, exact-owner density pressure and deep cleanup\n";return 0;
}catch(const std::exception& e){openmx::test::fail_allocation_after(-1);std::cerr<<e.what()<<'\n';return 1;}}

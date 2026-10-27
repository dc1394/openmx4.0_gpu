#include "openmx_core_chemical.h"
#include "openmx_core_energy.h"
#include "fixtures/native_resources/sdk_model.hpp"
#include "hamiltonian_cache_binding_fixture.hpp"
#include "hamiltonian_state_binding_fixture.hpp"
#include "density_state_binding_fixture.hpp"
#include "openmx_complex.h"
#include "openmx_band_col_state.h"
#include "openmx_band_noncol_state.h"
#include "openmx_env.h"
#include "openmx_precision.h"
#include <cstdint>
#include <cstring>
#include <memory>
#include <iostream>
#include <climits>
#include <type_traits>
using cudaStream_t=void*;
using cublasHandle_t=void*;
using cusolverDnHandle_t=void*;
using cusolverDnParams_t=void*;
using hipStream_t=void*;
using hipblasHandle_t=void*;
using hipsolverDnHandle_t=void*;
using cudaError_t=int;
using hipError_t=int;
using EigenRefineOccupation=double(*)(double,int,void*);
constexpr int cudaSuccess=0,hipSuccess=0,cudaErrorMemoryAllocation=2,hipErrorOutOfMemory=2;
constexpr double WAITTIME=0.001;
constexpr int mpi_comm_level1=0,Host_ID=0,MPI_INT=1,MPI_DOUBLE=2,MPI_MAX=0;
int level_stdout=0,Cnt_switch=0,DFTSCF_loop=50,xanes_calc=0,empty_occupation_flag=0,empty_states_flag=0;
int policy_iterations=2,refine_device_drains=0,external_workspace_drains=0;
static int cudaGetDevice(int* device){return OpenMXGpuGetDevice(device);}
static int cudaSetDevice(int device){return OpenMXGpuSetDevice(device);}
static int cudaGetLastError(){return 0;}
static int hipGetDevice(int* device){return OpenMXGpuGetDevice(device);}
static int hipSetDevice(int device){return OpenMXGpuSetDevice(device);}
static int hipGetLastError(){return 0;}
static int hipDeviceSynchronize(){return 0;}
static void acc_wait_all(){}
static void acc_clear_freelists(){}
static void MPI_Comm_rank(int,int* rank){*rank=0;}
static void MPI_Bcast(void*,int,int,int,int){}
static void MPI_Allreduce(const void* input,void* output,int count,int type,int,int){
    std::memcpy(output,input,count*(type==MPI_DOUBLE?sizeof(double):sizeof(int)));
}
static void MPI_Abort(int,int){throw std::runtime_error("unexpected production abort");}
static int openmx_eigen_refine_configure(int,int,int* ratio){*ratio=0;return policy_iterations;}
static double openmx_eigen_refine_until(){return 1e-5;}
static int BandCol_ReleaseDMDeviceWorkspace_HIP(){external_workspace_drains++;return 0;}
static void openmx_gemmul8ReleaseWorkspaces(){external_workspace_drains++;}
static void openmx_gemmul8TrimWorkspaces(){external_workspace_drains++;}
static void refine_check(int status,const char* message){check(status==0,message);}
#define wait_cudafunc(status) refine_check(status,"CUDA operation")
#define wait_hipfunc(status) refine_check(status,"HIP operation")
#define OpenMXBlasWait(status) refine_check(status,"BLAS operation")
#define OpenMXSolverWait(status) refine_check(status,"solver operation")
#include "band_state.inc"
namespace {
using Context=std::unique_ptr<OpenMXCoreContext,decltype(&openmx_core_destroy)>;
Context calculation(){char error[256];Context c(openmx_core_create_host(error,sizeof(error)),openmx_core_destroy);check(bool(c),error);return c;}
template<class T> void device_buffer(BandState* s,T** pointer){check(!OpenMX_Resource_GpuMalloc(s->resources,reinterpret_cast<void**>(pointer),64),"device allocation");}
template<class T> T* host_buffer(BandState* s,size_t count=4){auto* p=static_cast<T*>(openmx_core_native_malloc(s->resources,count,sizeof(T),1));check(p,"host allocation");return p;}
void populate(BandState* s,int device){
    current_device=device;s->mgpu_host_dense=1;
    auto& r=s->refine_state;r.resources=s->resources;r.capacity=4;r.basis_valid=1;r.basis_n=4;
    device_buffer(s,&r.lam);device_buffer(s,&r.occ);device_buffer(s,&r.own);device_buffer(s,&s->refine_x);
    s->refine_x_n=4;s->refine_e0=host_buffer<double>(s);s->refine_e0_n=4;
    s->kref_slots=2;s->kref=host_buffer<std::remove_pointer_t<decltype(s->kref)>>(s,2);s->kref_used=128;
    s->k_overlap=host_buffer<std::remove_pointer_t<decltype(s->k_overlap)>>(s,2);s->k_overlap_points=2;s->k_overlap_n=4;
    for(int k=0;k<2;k++){
        s->kref[k].basis=host_buffer<dcomplex>(s,16);s->kref[k].lam=host_buffer<double>(s);s->kref[k].valid=1;
        s->k_overlap[k].matrix=host_buffer<dcomplex>(s,16);
    }
    auto& dm=s->dm_entry_cache;dm.basis0=host_buffer<int>(s);dm.basis1=host_buffer<int>(s);
    dm.phase_index=host_buffer<int>(s);dm.phase_r=host_buffer<double>(s);dm.phase_i=host_buffer<double>(s);dm.valid=1;
#if BAND_NONCOL
    s->kref_n2=4;auto& g=s->gpusolver_workspace;g.initialized=1;g.device_id=device;
    device_buffer(s,&g.d_A);device_buffer(s,&g.d_W);device_buffer(s,&g.d_info);device_buffer(s,&g.d_work);g.h_work=host_buffer<double>(s);
    check(!OpenMX_Resource_StreamCreate(s->resources,&g.stream,0),"solver stream");
    check(!OpenMX_Resource_SolverCreate(s->resources,&g.gpusolver),"solver handle");
    auto& d=s->dm_gpu_workspace;d.initialized=1;d.device_id=device;device_buffer(s,&d.d_vec_up);device_buffer(s,&d.d_mat12);
    check(!OpenMX_Resource_StreamCreate(s->resources,&d.stream,0),"DM stream");
#if OPENMX_GPU_HIP
    check(!OpenMX_Resource_BlasCreate(s->resources,&d.hipblas),"DM BLAS");
    auto& dense=s->dense_gpu_workspace;dense.initialized=1;dense.device_id=device;
    device_buffer(s,&dense.d_S);device_buffer(s,&dense.d_H2);device_buffer(s,&dense.d_C2);device_buffer(s,&dense.d_scale);
    dense.h_scale=host_buffer<dcomplex>(s);check(!OpenMX_Resource_BlasCreate(s->resources,&dense.hipblas),"dense BLAS");
    device_buffer(s,&s->refine_buf.d_scratch);device_buffer(s,&s->refine_buf.d_w);device_buffer(s,&s->refine_buf.d_info);device_buffer(s,&s->refine_buf.d_work);
#else
    check(!OpenMX_Resource_BlasCreate(s->resources,&d.cublas),"DM BLAS");
    s->dm_occ_workspace.occ=host_buffer<double>(s);s->dm_occ_workspace.eig_occ=host_buffer<double>(s);
#endif
    auto& w=s->root_dense_workspace;w.s_all=host_buffer<dcomplex>(s);w.h11=host_buffer<dcomplex>(s);w.h22=host_buffer<dcomplex>(s);
    w.h12=host_buffer<dcomplex>(s);w.work=host_buffer<dcomplex>(s);w.hs2=host_buffer<dcomplex>(s);w.cs2=host_buffer<dcomplex>(s);
#else
    s->kref_n=4;auto& ks=s->kref_state;ks.resources=s->resources;ks.capacity=4;device_buffer(s,&ks.lam);device_buffer(s,&ks.occ);
    auto& g=s->gpusolver_ctx;g.initialized=1;g.device_id=device;g.matrix_dim=4;
    device_buffer(s,&g.d_S);device_buffer(s,&g.d_H);device_buffer(s,&g.d_tmp);device_buffer(s,&g.d_evec);
    device_buffer(s,&g.d_scale);device_buffer(s,&g.d_W);device_buffer(s,&g.d_info);device_buffer(s,&g.d_work);
    g.h_work=host_buffer<double>(s);g.h_scale=host_buffer<dcomplex>(s);g.h_evec=host_buffer<dcomplex>(s);g.h_transformed_s=host_buffer<dcomplex>(s);
#if OPENMX_GPU_HIP
    check(!OpenMX_Resource_BlasCreate(s->resources,&g.hipblas),"solver BLAS");
#else
    check(!OpenMX_Resource_BlasCreate(s->resources,&g.cublas),"solver BLAS");
#endif
    check(!OpenMX_Resource_SolverCreate(s->resources,&g.gpusolver),"solver handle");
    s->host_s=host_buffer<dcomplex>(s);s->host_h=host_buffer<dcomplex>(s);s->host_tmp=host_buffer<dcomplex>(s);s->host_n=4;
    s->dm_workspace.buf_Re0=host_buffer<double>(s);s->dm_workspace.OccWeight=host_buffer<double>(s);
#endif
}
void policy(BandState* a,BandState* b){
    check(a->refine.iterations==-1 && b->refine.iterations==-1,"initial refinement sentinels");
    openmx_set_fp32_enabled(0);band_configure(a);check(a->refine.iterations==-1,"FP32 veto was cached");
    openmx_set_fp32_enabled(1);policy_iterations=2;band_configure(a);policy_iterations=1;band_configure(b);
    check(band_iterations(a)==2 && band_iterations(b)==1,"contexts share refinement policy");
    band_begin(a,1,1,1,1);check(band_last(a) && !band_last(b),"step control leaked");
    a->refine.solved=1;band_end(a);band_mixing(a,1e-6);
    check(a->refine.final_stage && !b->refine.final_stage,"mixing changed another context");
    band_force_fp64(b,1);check(b->refine.force_fp64 && !a->refine.force_fp64,"FP64 control leaked");band_force_fp64(b,0);
}
void callbacks(BandState* a,BandState* b){
    EigenRefineDevice dev{};dev.allocation_context=a;dev.try_malloc_owned=band_malloc;dev.host_malloc_owned=band_host_malloc;
    current_device=3;void* pointer=nullptr;check(!openmx_eigen_refine_malloc(&dev,&pointer,32),"owned SDK callback failed");
    check(openmx_core_native_resource_count(a->resources)==1 && !openmx_core_native_resource_count(b->resources),"SDK callback used another owner");
    check(!OpenMX_Resource_GpuFree(a->resources,pointer),"callback free failed");
    void* host=openmx_eigen_refine_host_malloc(&dev,4,8,"owned host");check(host,"owned host callback failed");openmx_core_native_free(a->resources,host);
    fail_status=2;pointer=reinterpret_cast<void*>(17);check(openmx_eigen_refine_malloc(&dev,&pointer,32)==2 && !pointer,"SDK failure lost output/status");fail_status=0;
    check(!live && !openmx_core_native_resource_count(a->resources),"callback leaked");
}
}
OpenMXChemicalPotentials* owned_chemical(OpenMXCoreContext* c) {
  OpenMXChemicalPotentials* out=nullptr;char error[256];
  check(openmx_core_chemical_potentials(c,&out,error,sizeof(error)),error);return out;
}
int main(){try{
    for(int cycle=0;cycle<100;cycle++){
        event_count=0;queries=0;auto first=calculation();auto second=calculation();
        auto* a=band_bind(first.get());auto* b=band_bind(second.get());
    check(a->chemical==owned_chemical(first.get()) && b->chemical==owned_chemical(second.get()) && a->chemical!=b->chemical,
          "native solver borrowed another calculation's chemical potentials");
    check(a->hamiltonian_cache==Set_Hamiltonian_Cache_Bind_Resources(first.get()) &&
          b->hamiltonian_cache==Set_Hamiltonian_Cache_Bind_Resources(second.get()) &&
          a->hamiltonian_cache!=b->hamiltonian_cache,
          "solver borrowed another calculation's packed Hamiltonian cache");
#if OPENMX_GPU_HIP && !BAND_NONCOL
    check(a->hamiltonian_state==Set_Hamiltonian_GPU_Bind_Resources(first.get()) &&
          b->hamiltonian_state==Set_Hamiltonian_GPU_Bind_Resources(second.get()) &&
          a->hamiltonian_state!=b->hamiltonian_state,"band solver borrowed another calculation's Hamiltonian GPU cache");
#endif
    a->chemical->chemical_potential=.125;a->chemical->xanes[0]=1.5;
    b->chemical->chemical_potential=-.25;b->chemical->xanes[1]=2.5;
        check(a!=b && a->resources!=b->resources && !queries && !live,"binding shared state or touched SDK");
        check(a->refine.iterations==-1 && b->refine.iterations==-1,"constructor sentinel lost");
#if !OPENMX_GPU_HIP
        check(a->QueryGpuSolverWorkBytes_cached_n[0]==-1 && a->QueryGpuSolverWorkBytes_cached_n[1]==-1,"query sentinel array lost");
#endif
        policy(a,b);callbacks(a,b);a->refine.scf_iter=71;
        openmx::test::fail_allocation_after(0);auto* same=band_bind(first.get());openmx::test::fail_allocation_after(-1);
        check(same==a && a->refine.scf_iter==71 && b->refine.scf_iter!=71,"SCF lookup allocated/reset/shared state");
        populate(a,1);populate(b,2);const auto count_b=openmx_core_native_resource_count(b->resources);
        current_device=7;band_end_cycle(a);band_end_cycle(a);
        check(!a->kref && !a->refine_x && !a->refine_state.basis_valid,"cycle end left warm bases live");
        check(a->k_overlap && openmx_core_native_resource_count(a->resources)>0,"cycle end lost retained overlap/solver cache");
        check(openmx_core_native_resource_count(b->resources)==count_b && b->kref && b->refine_state.basis_valid,"cycle end invalidated another calculation");
        band_release(a);band_release(a);
        check(!openmx_core_native_resource_count(a->resources) && a->refine.iterations==-1,"run-boundary drain leaked or lost policy reset");
        event_count=0;openmx::test::fail_allocation_after(0);openmx_core_native_resources_clear(first.get());
    check(owned_chemical(first.get())->chemical_potential==0 && owned_chemical(first.get())->xanes[0]==0 &&
          b->chemical->chemical_potential==-.25 && b->chemical->xanes[1]==2.5,
          "clearing a calculation retained chemical potentials or reset another calculation");openmx::test::fail_allocation_after(-1);
        check(openmx_core_native_resource_count(b->resources)==count_b && b->kref,"clear affected another calculation");
        openmx_core_native_resources_clear(first.get());first.reset();
        event_count=0;openmx::test::fail_allocation_after(0);openmx_core_finish(second.get());openmx::test::fail_allocation_after(-1);
        check(!live && current_device==7,"finish leaked SDK resources or changed caller device");
        openmx_core_finish(second.get());auto* fresh=band_bind(second.get());
        check(fresh->refine.iterations==-1 && !fresh->refine.scf_iter && !fresh->kref,"next input inherited old policy/cache");
    }
    std::cout<<"PASS: 100 dual-context band lifecycles; SCF reuse, sentinel arrays, SDK callbacks, refinement/FP64 stages, warm bases, retained overlap and exact-owner teardown\n";
    return 0;
}catch(const std::exception& e){openmx::test::fail_allocation_after(-1);std::cerr<<e.what()<<'\n';return 1;}}

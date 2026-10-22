#include "fixtures/native_resources/sdk_model.hpp"
#include "openmx_complex.h"
#include <cstdint>
#include <cstring>
#include <memory>
#include <iostream>
#include <climits>
#include <type_traits>
#include <set>
using cudaStream_t=void*;
using cublasHandle_t=void*;
using cusolverDnHandle_t=void*;
using hipStream_t=void*;
using hipblasHandle_t=void*;
using hipsolverDnHandle_t=void*;
using hipDoubleComplex=dcomplex;
using cudaError_t=int;
using hipError_t=int;
#if LOCAL_MPI_REAL
#include <mpi.h>
#else
using MPI_Comm=int;
constexpr int MPI_COMM_NULL=-19;
#endif
constexpr int cudaSuccess=0,hipSuccess=0;
constexpr int DC_GPU_CPU_SWITCH_NUM=400,DC_NONCOL_GPU_CPU_SWITCH_NUM=300;
constexpr int DC_GPUSOLVER_GEMMUL8_DISABLED_DEFAULT=OPENMX_GPU_HIP;
constexpr int KRYLOV_GPU_EIGEN_MIN=300;
constexpr double KRYLOV_GPU_DGEMM_MIN_FLOPS=50000000.0;
std::set<MPI_Comm> live_comms;
int next_comm=1,comm_drains=0,external_drains=0;
#if LOCAL_MPI_REAL
static MPI_Comm new_comm(){
    int rank;MPI_Comm c;MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    check(MPI_Comm_split(MPI_COMM_WORLD,rank%2,rank,&c)==MPI_SUCCESS,"MPI split");
    check(live_comms.insert(c).second,"MPI reused live descriptor");return c;
}
static void tracked_comm_free(MPI_Comm* comm){
    check(live_comms.erase(*comm)==1,"communicator freed twice/on another owner");
    check(MPI_Comm_free(comm)==MPI_SUCCESS,"MPI free");comm_drains++;
}
#define MPI_Comm_free tracked_comm_free
#else
static int new_comm(){int c=next_comm++;live_comms.insert(c);return c;}
static void MPI_Comm_free(MPI_Comm* comm){check(live_comms.erase(*comm)==1,"communicator freed twice/on another owner");*comm=MPI_COMM_NULL;comm_drains++;}
#endif
static int cudaGetDevice(int* device){return OpenMXGpuGetDevice(device);}
static int cudaSetDevice(int device){return OpenMXGpuSetDevice(device);}
static const char* cudaGetErrorString(int){return "SDK model";}
static int hipGetDevice(int* device){return OpenMXGpuGetDevice(device);}
static int hipSetDevice(int device){return OpenMXGpuSetDevice(device);}
static int cudaStreamSynchronize(void* stream){return OpenMXGpuStreamSynchronize(stream);}
static int hipStreamSynchronize(void* stream){return OpenMXGpuStreamSynchronize(stream);}
static void openmx_magma_ozaki2_release(){external_drains++;}
static void refine_check(int status){check(status==0,"SDK status");}
#define wait_cudafunc(status) refine_check(status)
#define wait_hipfunc(status) refine_check(status)
#define OpenMXBlasWait(status) refine_check(status)
#define OpenMXSolverWait(status) refine_check(status)
#include "local_solver_state.inc"
namespace {
using Context=std::unique_ptr<OpenMXCoreContext,decltype(&openmx_core_destroy)>;
Context calculation(){char error[256];Context c(openmx_core_create_host(error,sizeof(error)),openmx_core_destroy);check(bool(c),error);return c;}
template<class T> void device_buffer(LocalState* s,T** out){check(!OpenMX_Resource_GpuMalloc(s->resources,reinterpret_cast<void**>(out),64),"device allocation");}
template<class T> T* host_buffer(LocalState* s,size_t count=4){auto* p=static_cast<T*>(openmx_core_native_malloc(s->resources,count,sizeof(T),1));check(p,"host allocation");return p;}
void sentinel(LocalState* s){
#if LOCAL_DC
    check(s->Col_owned_firsttime==1 && s->NonCol_owned_firsttime==1,"DC first-use state lost");
    check(s->gpusolver_gemmul8_disabled==DC_GPUSOLVER_GEMMUL8_DISABLED_DEFAULT,"DC backend default lost");
#if OPENMX_GPU_HIP
    check(s->GPU_Threshold_threshold==400 && s->NonCol_GPU_Threshold_threshold==300 && s->GpuSolver_RankAllowed_allowed==1,"HIP DC admission defaults lost");
#endif
#elif LOCAL_DCLNO
    check(s->DC_Col_owned_firsttime==1 && s->DC_NonCol_owned_firsttime==1,"DC-LNO first-use state lost");
    check(s->node_comm==MPI_COMM_NULL && s->gpu_group_comm==MPI_COMM_NULL && s->node_size==1 && s->gpu_group_size==1,"DC-LNO communicator sentinels lost");
#if OPENMX_GPU_HIP
    check(s->ProfileEnabled_enabled==-1 && s->GpuEigenThreshold_threshold==-1 && s->GpuOwnerThreshold_threshold==-1 && s->NonColGpuEigenThreshold_threshold==-1 && s->RetryTest_enabled==-1,"HIP DC-LNO policy sentinels lost");
#endif
#else
    check(s->Col_owned_firsttime==1 && s->Col_owned_recalc_firsttime==1 && s->Col_trd_owned_firsttime==1 && s->Col_trd_owned_recalc_firsttime==1,"Krylov history sentinels lost");
    check(s->gpu_eigen_min==300 && s->gpu_dgemm_min_flops==50000000.0 && s->gpu_fused==1,"Krylov thresholds lost");
#if OPENMX_GPU_HIP
    check(s->GPU_Preflight_device_comm==MPI_COMM_NULL && s->GPU_Preflight_reserve_bytes==size_t(512)*1024*1024 && s->GPU_Enabled_requested==-1 && s->ProjectedSolve_GPU_profile==-1 && s->measure_time_env==-1,"HIP Krylov sentinels lost");
#else
    check(s->gpu_device==-1,"Krylov device sentinel lost");
#endif
#endif
}
void populate(LocalState* s,int device){
    current_device=device;
#if LOCAL_DC
    auto& g=s->gpusolver_ctx;g.initialized=1;g.device_id=device;g.matrix_dim=4;
    device_buffer(s,&g.d_S);device_buffer(s,&g.d_H);device_buffer(s,&g.d_tmp);device_buffer(s,&g.d_A);device_buffer(s,&g.d_C);check(!OpenMX_Resource_HostAlloc(s->resources,reinterpret_cast<void**>(&g.h_matrix),64,0),"DC pinned matrix");
    check(!OpenMX_Resource_StreamCreate(s->resources,&g.stream,0),"DC stream");
#if OPENMX_GPU_HIP
    check(!OpenMX_Resource_BlasCreate(s->resources,&g.hipblas),"DC BLAS");
    auto& n=s->noncol_gpu;n.num=4;device_buffer(s,&n.d_A);device_buffer(s,&n.d_Y);device_buffer(s,&n.d_T);
    check(!OpenMX_Resource_HostAlloc(s->resources,&n.h_stage,64,0),"NC pinned stage");
#else
    check(!OpenMX_Resource_BlasCreate(s->resources,&g.cublas),"DC BLAS");
    check(!OpenMX_Resource_SolverCreate(s->resources,&g.gpusolver),"DC solver");
    device_buffer(s,&g.d_W);device_buffer(s,&g.d_info);device_buffer(s,&g.d_work);g.h_work=host_buffer<double>(s);
#endif
    auto& c=s->scache;c.state=1;c.dim=host_buffer<int>(s);c.valid=host_buffer<int>(s);c.off=host_buffer<size_t>(s);device_buffer(s,&c.d_arena);
#elif LOCAL_DCLNO
    s->gpu_proxy_initialized=1;s->node_comm=new_comm();s->gpu_group_comm=new_comm();
    auto& g=s->gpusolver_ctx;g.initialized=1;g.max_num=4;
    auto& z=s->gpusolverz_ctx;z.initialized=1;z.max_num=4;
    device_buffer(s,&g.d_S);device_buffer(s,&g.d_H);device_buffer(s,&g.d_tmp);device_buffer(s,&g.d_W);
    device_buffer(s,&z.d_S);device_buffer(s,&z.d_H);device_buffer(s,&z.d_tmp);device_buffer(s,&z.d_scale);z.h_scale=host_buffer<dcomplex>(s);
    check(!OpenMX_Resource_StreamCreate(s->resources,&g.stream,0) && !OpenMX_Resource_StreamCreate(s->resources,&z.stream,0),"DC-LNO streams");
#if OPENMX_GPU_HIP
    check(!OpenMX_Resource_BlasCreate(s->resources,&g.hipblas) && !OpenMX_Resource_BlasCreate(s->resources,&z.hipblas),"DC-LNO BLAS");
#else
    check(!OpenMX_Resource_BlasCreate(s->resources,&g.cublas) && !OpenMX_Resource_BlasCreate(s->resources,&z.cublas),"DC-LNO BLAS");
    check(!OpenMX_Resource_SolverCreate(s->resources,&g.gpusolver) && !OpenMX_Resource_SolverCreate(s->resources,&z.gpusolver),"DC-LNO solvers");
    device_buffer(s,&g.d_info);device_buffer(s,&g.d_work);g.h_work=host_buffer<double>(s);
    device_buffer(s,&z.d_W);device_buffer(s,&z.d_info);device_buffer(s,&z.d_work);z.h_work=host_buffer<double>(s);
#endif
#else
    s->gpu_ws_pool=host_buffer<Krylov_GPU_Workspace>(s,2);s->gpu_ws_pool_size=2;
#if OPENMX_GPU_HIP
    s->GPU_Preflight_device_comm=new_comm();
#else
    s->gpu_device=device;
#endif
    for(int i=0;i<2;i++){
        auto& w=s->gpu_ws_pool[i];w.active=1;device_buffer(s,&w.d_A);device_buffer(s,&w.d_B);device_buffer(s,&w.d_C);w.h_A=host_buffer<double>(s);
        check(!OpenMX_Resource_StreamCreate(s->resources,&w.stream,0),"Krylov stream");
#if OPENMX_GPU_HIP
        check(!OpenMX_Resource_BlasCreate(s->resources,&w.hipblas),"Krylov BLAS");
#else
        check(!OpenMX_Resource_BlasCreate(s->resources,&w.cublas),"Krylov BLAS");
        check(!OpenMX_Resource_SolverCreate(s->resources,&w.gpusolver),"Krylov solver");
        device_buffer(s,&w.d_W);device_buffer(s,&w.d_info);device_buffer(s,&w.d_work);w.h_work=host_buffer<double>(s);
#endif
    }
    auto& k=s->kucache;k.state=1;k.dim_n=host_buffer<int>(s);k.dim_m=host_buffer<int>(s);k.valid=host_buffer<int>(s);k.off=host_buffer<size_t>(s);device_buffer(s,&k.d_arena);
#endif
}
}
int main(int argc,char** argv){
#if LOCAL_MPI_REAL
    MPI_Init(&argc,&argv);
#endif
try{
    for(int cycle=0;cycle<100;cycle++){
        event_count=0;queries=0;auto first=calculation();auto second=calculation();
        auto* a=local_bind(first.get());auto* b=local_bind(second.get());
        check(a!=b && a->resources!=b->resources && !queries && !live,"bind shared state or touched SDK");sentinel(a);sentinel(b);
        populate(a,1);populate(b,2);auto count_b=openmx_core_native_resource_count(b->resources);auto comms=live_comms.size();
        a->initialized=71;openmx::test::fail_allocation_after(0);auto* same=local_bind(first.get());openmx::test::fail_allocation_after(-1);
        check(same==a && a->initialized==71 && b->initialized==1,"SCF lookup allocated/reset/shared metadata");
        current_device=7;local_release(a);local_release(a);
        check(!openmx_core_native_resource_count(a->resources),"direct cleanup retained resources");
        check(openmx_core_native_resource_count(b->resources)==count_b && current_device==7,"cleanup affected another owner/device");
        check(live_comms.size()==comms/2,"cleanup retained/freed another owner's communicator");
        event_count=0;openmx::test::fail_allocation_after(0);openmx_core_native_resources_clear(first.get());openmx::test::fail_allocation_after(-1);
        check(openmx_core_native_resource_count(b->resources)==count_b,"clear drained another owner");openmx_core_native_resources_clear(first.get());
        auto* fresh=local_bind(first.get());sentinel(fresh);check(openmx_core_native_resource_count(fresh->resources)==0,"next input inherited resources");
        // A partially initialized object still falls back to exact-owner drain.
        current_device=3;void* orphan=nullptr;check(!OpenMX_Resource_GpuMalloc(fresh->resources,&orphan,64),"partial allocation");
        current_device=7;event_count=0;openmx::test::fail_allocation_after(0);openmx_core_finish(first.get());openmx::test::fail_allocation_after(-1);
        check(openmx_core_native_resource_count(b->resources)==count_b,"partial-state fallback drained another owner");
        event_count=0;openmx::test::fail_allocation_after(0);openmx_core_finish(second.get());openmx::test::fail_allocation_after(-1);
        check(!live && live_comms.empty() && current_device==7,"finish leaked SDK/MPI resources or caller device");
        openmx_core_finish(second.get());fresh=local_bind(second.get());sentinel(fresh);
    }
    std::cout<<"PASS: 100 dual-context local-solver lifecycles; sentinels, reuse, deep/idempotent cleanup, partial-state fallback, exact-owner SDK/MPI drains, no-allocation teardown\n";
#if LOCAL_MPI_REAL
    MPI_Finalize();
#endif
    return 0;
}catch(const std::exception& e){openmx::test::fail_allocation_after(-1);std::cerr<<e.what()<<'\n';return 1;}}

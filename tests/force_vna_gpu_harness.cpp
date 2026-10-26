#include "openmx_core_solver_workspace.h"
#include "fixtures/native_resources/sdk_model.hpp"
#include "openmx_env.h"
#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <omp.h>

// Nonzero null handle catches accidental zero-initialization of MPI handles.
using MPI_Comm=int;
constexpr int MPI_COMM_NULL=-17,MPI_COMM_TYPE_SHARED=1,MPI_INFO_NULL=0;
constexpr int MPI_INT=1,MPI_UNSIGNED_LONG_LONG=2,MPI_BYTE=3,MPI_MIN=1,MPI_MAX=2,MPI_SUM=3,MPI_UNDEFINED=-1;
#define MPI_IN_PLACE ((void*)1)
static MPI_Comm mpi_comm_level1=1;
static int communicators=0,barriers=0,peer_enabled=0,waits=0,clears=0;
static int usable=1,recovery=0,adoption_failure=0,weak_fp64=0,properties_failure=0;
static unsigned long long host_room=1ULL<<40;
static size_t free_room=1ULL<<40;
static int MPI_Comm_free(MPI_Comm *comm){check(*comm!=MPI_COMM_NULL,"free of null MPI handle");*comm=MPI_COMM_NULL;--communicators;return 0;}
static void MPI_Comm_split_type(MPI_Comm,int,int,int,MPI_Comm *comm){*comm=2;++communicators;}
static void MPI_Comm_split(MPI_Comm,int color,int,MPI_Comm *comm){*comm=color==MPI_UNDEFINED?MPI_COMM_NULL:4;if(*comm!=MPI_COMM_NULL)++communicators;}
static void MPI_Comm_size(MPI_Comm,int *n){*n=1;}
static void MPI_Comm_rank(MPI_Comm,int *n){*n=0;}
static void MPI_Barrier(MPI_Comm){++barriers;}
static void MPI_Bcast(void*,int,int,int,MPI_Comm){}
static void MPI_Allreduce(const void *s,void *d,int count,int type,int op,MPI_Comm comm){
    if(s!=MPI_IN_PLACE)std::memcpy(d,s,count*(type==MPI_UNSIGNED_LONG_LONG?sizeof(unsigned long long):sizeof(int)));
    if(comm==4 && type==MPI_INT && op==MPI_SUM)*static_cast<int*>(d)+=peer_enabled;
}
static void MPI_Reduce(const void *s,void *d,int n,int type,int,int,MPI_Comm){MPI_Allreduce(s,d,n,type,0,0);}
static void MPI_Allgather(const void *s,int n,int,void *d,int,int,MPI_Comm){std::memcpy(d,s,n);}
static int OpenMXGpuDeviceSynchronize(){record(400);return 0;}
static int OpenMXGpuGetLastError(){return 0;}
static int cudaDeviceSynchronize(){return OpenMXGpuDeviceSynchronize();}
static int hipGetLastError(){return 0;}
static int hipDeviceSynchronize(){return OpenMXGpuDeviceSynchronize();}
static int cudaGetDevice(int *device){return OpenMXGpuGetDevice(device);}
static int hipGetDevice(int *device){return OpenMXGpuGetDevice(device);}
static int gpu_rank_device_usable(){return usable;}
static int openmx_gpu_capacity_test_recovery(){return recovery;}
static int openmx_gpu_device_weak_fp64(int){return weak_fp64;}
static unsigned long long openmx_host_mem_available(){return host_room;}
static int OpenMX_GpuMemGetInfo(size_t *f,size_t *t){*f=free_room;*t=1ULL<<40;return 0;}
static void acc_wait_all(){++waits;}
static void acc_clear_freelists(){++clears;}
static void acc_memcpy_to_device(void *d,const void *s,size_t n){std::memcpy(d,s,n);}
static void openmx_target_wait(){++waits;}
static void openmx_target_clear_freelists(){}
static void openmx_target_upload(void *d,const void *s,size_t n){std::memcpy(d,s,n);}
static void Force_gpu_pool_flush(){}
static int Force_gpu_stage_default(){return openmx_env_flag_at(OPENMX_ENV_FORCE_GPU,1);}
static int Force_collective_env_flag(const char *name,int fallback,MPI_Comm){return openmx_env_flag(name,fallback);}
static int Force_env_flag(const char *name,int fallback){return openmx_env_flag(name,fallback);}
static void OpenMX_Solver_Failure(const char *message){throw std::runtime_error(message);}
static void Force4B_gpu_abort(const char *message){throw std::runtime_error(message);}
static void *Force_gpu_arena_try(size_t bytes){
    void *p=nullptr;
#if OPENMX_GPU_HIP
    (void)OpenMXGpuMalloc(&p,bytes);
#else
    p=openmx_test_acc_malloc(bytes);
#endif
    if(p && adoption_failure)openmx::test::fail_allocation_after(0);
    return p;
}
/* OPENMX_GPU_RESOURCE_ADAPTER */
constexpr int cudaSuccess=0,hipSuccess=0,GPUSOLVER=9;
static int scf_eigen_lib_flag=GPUSOLVER,level_stdout=0,Solver=2,SpinP_switch=0;
static int Matomnum=1,MatomnumF=2,List_YOUSO[50];
static int M2G[]{0,1},F_M2G[]{0,1,2,3},F_G2M[]{0,1,2,3};
static int WhatSpecies[]{-1,0,1,0},FNAN[]{-1,2,2,2},Spe_Total_CNO[]{2,1};
static int n1[]{1,2,3},n2[]{2,1,3},n3[]{3,2,1};
static int *natn[]{nullptr,n1,n2,n3};
static int r1[]{0,1,2},r2[]{1,0,2},r3[]{2,1,0};
static int *rmi_rows[]{r1,r2,r3};static int **RMI1[]{nullptr,rmi_rows};
struct cudaDeviceProp{struct{unsigned char bytes[16];}uuid;};
static int cudaGetDeviceProperties(cudaDeviceProp *p,int){std::memset(p,0,sizeof(*p));return properties_failure;}
static int hipDeviceGetPCIBusId(char *bus,int n,int){std::snprintf(bus,n,"0000:03:00.0");return properties_failure;}
using Type_DS_VNA=float;
static float values[4][3][3][2][3];
static float *rows[4][3][3][2];static float **blocks[4][3][3];
static float ***atoms[4][3];static float ****directions[4];
static double density[2][2][3][2][2];static double *density_rows[2][2][3][2];
static double **density_blocks[2][2][3];static double ***density_atoms[2][2];static double ****cdm[2];
#include "openmx_force_gpu_resources.h"
#include "openmx_host_simd.h"
/* The stage's host SIMD policy vote (Force4B_GpuBegin of both backends),
   recorded: the device answer keeps the paths of modes 0-13; mode 14 checks
   the host one. */
static int host_vote=0,votes=0;static size_t vote_need=0,vote_budget=0;
int OpenMX_HostSimdStageOnHost(const char*,size_t need,size_t budget){++votes;vote_need=need;vote_budget=budget;return host_vote;}
static void Force4B_GpuEnd(Force4BGpuContext*);
/* OPENMX_FORCE4B_PRODUCTION */

namespace {
struct Case {int stage,mode;void *outer=nullptr;};
void reset(int mode){
    check(!live && !communicators,"previous case leaked SDK/MPI objects");
    openmx_env_reset();
    for(const char *name:{"OPENMX_FORCE_GPU","OPENMX_FORCE4B_GPU","OPENMX_FORCE4B_ARCHIVE_GPU","OPENMX_FORCE4B_CASE1_STREAM_MB"})
        check(!openmx_env_unset(name),"clear override");
    current_device=acc_device=3;event_count=queries=barriers=waits=clears=peer_enabled=0;
    usable=1;recovery=adoption_failure=weak_fp64=properties_failure=fail_status=0;
    host_room=1ULL<<40;free_room=1ULL<<40;scf_eigen_lib_flag=GPUSOLVER;
    Matomnum=1;MatomnumF=2;List_YOUSO[34]=3;List_YOUSO[35]=0;
    if(mode==4)free_room=1;if(mode==5)fail_status=2;
    if(mode==6)adoption_failure=1;if(mode==7)scf_eigen_lib_flag=0;
    if(mode==8)check(!openmx_env_set("OPENMX_FORCE4B_GPU","0"),"set override");
    if(mode==9)recovery=1;
    if(mode==10){recovery=1;host_room=1;}
    if(mode==11)peer_enabled=1;
    if(mode==13)Matomnum=MatomnumF=0;
    host_vote=mode==14;votes=0;vote_need=vote_budget=0;
}
void archives(Force4BGpuContext *g){
    check(g->flat_stride==18 && g->halo_count==27 && g->c2_stride==15,"archive shapes changed");
    const float *flat=g->flat_dev?g->flat_dev:g->flat_host;
    for(int d=0;d<4;++d)for(int h=0;h<3;++h)for(int r=0;r<2;++r)for(int p=0;p<3;++p)
        check(flat[d*18+h*6+r*3+p]==values[d][1][h][r][p],"local archive packing changed");
    for(int source=2;source<=3;++source){
        Force4B_GpuArchiveHalo(g,directions,source,source);
        const float *halo=g->halo_dev?g->halo_dev:g->halo_host;
        const int orbital=source==2?1:2;const size_t base=source==2?0:9;
        for(int h=0;h<3;++h)for(int r=0;r<orbital;++r)for(int p=0;p<3;++p)
            check(halo[base+h*orbital*3+r*3+p]==values[0][2][h][r][p],"halo archive packing changed");
    }
    check(Force4B_GpuCase2Begin(g,1),"case 2 archive declined");
    check(!g->flat_dev && !g->halo_dev && !g->flat_host && !g->halo_host,"case-1 archives survived into case 2");
    Force4B_GpuArchiveCase2(g,directions,1,1);
    const float *c2=g->c2_dev?g->c2_dev:g->c2_host;
    for(int d=0;d<4;++d){int offset=0;for(int h=0;h<3;++h){
        const int orbitals=h==1?1:2;
        for(int r=0;r<orbitals;++r)for(int p=0;p<3;++p)
            check(c2[d*15+offset+r*3+p]==values[d][2][h][r][p],"case-2 archive packing changed");
        offset+=orbitals*3;
    }}
}
int operation(OpenMXSolverInvocation *scope,void *opaque,double *result){
    auto &c=*static_cast<Case*>(opaque);OpenMXNativeResources *resources=nullptr;void *view=nullptr;
    Force4BGpuContext *archive=nullptr;Force4BCase1Stream *stream1=nullptr;
    Force4BCase1Turns *turns=nullptr;Force4BCase2Stream *stream2=nullptr;
    int enabled=0;
    auto begin=[&]{
        if(c.stage==0){
#if OPENMX_GPU_HIP
            enabled=Force4B_GpuBegin(scope,&archive,directions,0,0,0);
#else
            enabled=Force4B_GpuBegin(scope,&archive,directions,0,0);
#endif
            view=archive;if(archive)resources=archive->resources;
        }else if(c.stage==1){enabled=Force4B_GpuCase1StreamBegin(scope,&stream1,0,1);view=stream1;if(stream1)resources=stream1->resources;}
        else if(c.stage==2){enabled=Force4B_GpuCase1TurnsBegin(scope,&turns,c.mode==11,0,1,cdm);view=turns;if(turns)resources=turns->resources;}
        else{enabled=Force4B_GpuCase2StreamBegin(scope,&stream2,0,1);view=stream2;if(stream2)resources=stream2->resources;}
    };
    auto end=[&]{if(c.stage==0)Force4B_GpuEnd(archive);else if(c.stage==1)Force4B_GpuCase1StreamEnd(stream1);
        else if(c.stage==2)Force4B_GpuCase1TurnsEnd(turns);else Force4B_GpuCase2StreamEnd(stream2);};
    begin();
    if(c.mode==7 || c.mode==8){check(!enabled && !view && !queries && !communicators && !votes,"disabled stage allocated or queried GPU state");*result=1;return 1;}
    /* The admission votes once with the residency figures: need = the case
       lifetime, budget = free minus the node reserve (272 MiB for one node
       rank), zero without a usable device answer; the capacity test passes
       need 0. The streams and rank turns do not vote. */
    if(!c.outer){
        check(votes==(c.stage==0),"policy vote count differs");
        if(c.stage==0){
            check(c.mode==9 || c.mode==10?vote_need==0:vote_need>0,"policy vote need differs");
            check(vote_budget==(c.mode==4 || c.mode==9 || c.mode==10?0:(size_t(1)<<40)-272ULL*1024*1024),"policy vote budget differs");
        }
    }
    check(view && view!=c.outer && resources,"missing independent invocation-owned state");
    /* mode 14 votes for the host */
    const bool expected_enabled=c.stage==0?c.mode!=10 && c.mode!=14:c.stage==2?
        (c.mode!=4 && c.mode!=10 && c.mode!=11 && c.mode!=13):
        (c.mode!=4 && c.mode!=5 && c.mode!=6 && c.mode!=13);
    check(bool(enabled)==expected_enabled,"GPU admission changed");
    if(c.mode==11 && c.stage==2){
        check(!enabled && turns->device_comm!=MPI_COMM_NULL,"inactive peer lost collective participation");
        check(openmx_core_native_resource_count(resources)==1,"peer communicator is not owned");
    }else if(enabled){
        if(c.stage==0){check(openmx_core_native_resource_count(resources)==8,"archive allocations escaped owner");
            if(Matomnum)archives(archive);else check(!archive->flat_stride && !archive->halo_count && !archive->c2_stride,"empty rank gained archive values");}
        else if(c.stage==1)check(openmx_core_native_resource_count(resources)==9,"case-1 stream allocations escaped owner");
        else if(c.stage==2)check(openmx_core_native_resource_count(resources)==7,"rank-turn allocations escaped owner");
        else check(openmx_core_native_resource_count(resources)==3,"case-2 stream allocations escaped owner");
    }else check(!openmx_core_native_resource_count(resources),"declined stage kept partial allocations");
    if(c.outer){*result=1;return 1;}
    if(c.mode==0){
        void *saved=view;begin();check(saved==view,"repeated preparation moved native metadata");
        end();end();check(!openmx_core_native_resource_count(resources),"explicit/repeated cleanup leaked");
    }
    if(c.mode==12){const size_t owned=openmx_core_native_resource_count(resources);Case nested{c.stage,1,view};double nested_result;char error[256];
        check(openmx_core_solver_run(1,operation,&nested,&nested_result,error,sizeof(error)),error);
        check(nested_result==1 && openmx_core_native_resource_count(resources)==owned,"nested call altered outer resources");
    }
    current_device=acc_device=7;
    if(c.mode==2)return 0;if(c.mode==3)throw std::runtime_error("injected Force4B callback exception");
    *result=1;return 1;
}
}
int main(){
    try{
        for(int d=0;d<4;++d){directions[d]=atoms[d];for(int a=0;a<3;++a){atoms[d][a]=blocks[d][a];
            for(int h=0;h<3;++h){blocks[d][a][h]=rows[d][a][h];for(int r=0;r<2;++r){rows[d][a][h][r]=values[d][a][h][r];
                for(int p=0;p<3;++p)values[d][a][h][r][p]=10000*d+1000*a+100*h+10*r+p+.125f;}}}}
        for(int s=0;s<2;++s){cdm[s]=density_atoms[s];for(int a=0;a<2;++a){density_atoms[s][a]=density_blocks[s][a];
            for(int h=0;h<3;++h){density_blocks[s][a][h]=density_rows[s][a][h];for(int r=0;r<2;++r){density_rows[s][a][h][r]=density[s][a][h][r];
                for(int j=0;j<2;++j)density[s][a][h][r][j]=.01*(1+s+a+h+r+j);}}}}
        omp_set_dynamic(0);int cases=0,failures=0;
        for(int threads:{1,2,4,8}){omp_set_num_threads(threads);for(int stage=0;stage<4;++stage)for(int mode=0;mode<15;++mode){
            reset(mode);Case c{stage,mode};char error[256];double result=31;
            const int ok=openmx_core_solver_run(threads,operation,&c,&result,error,sizeof(error));
            if(ok!=(mode!=2 && mode!=3))throw std::runtime_error("stage="+std::to_string(stage)+" mode="+std::to_string(mode)+": "+error);check(result==(ok?1:31),"failed callback published output");
            check(!live && !communicators,"scope leaked SDK allocation or communicator");
            check((current_device==7 && acc_device==7) || mode==7 || mode==8,"RAII cleanup changed caller's device");++cases;
        }}
        for(int stage=0;stage<4;++stage){reset(1);Case c{stage,1};char error[256];double result;
            openmx::test::fail_allocation_after(10000);check(openmx_core_solver_run(1,operation,&c,&result,error,sizeof(error)),error);
            const long count=10000-openmx::test::remaining_allocations();openmx::test::fail_allocation_after(-1);
            for(long fail=0;fail<count;++fail){reset(1);const auto before=openmx::test::live_allocations();result=31;
                openmx::test::fail_allocation_after(fail);const int ok=openmx_core_solver_run(1,operation,&c,&result,error,sizeof(error));
                openmx::test::fail_allocation_after(-1);check((ok && result==1)||(!ok && result==31),"failed setup published output");
                check(!live && !communicators && openmx::test::live_allocations()==before,"partial preparation leaked");++failures;
            }
        }
        std::cout<<"PASS: "<<cases<<" actual VNA GPU resource cases; "<<failures<<" C++ allocation failures; backend="<<OPENMX_GPU_HIP<<'\n';return 0;
    }catch(const std::exception &e){openmx::test::fail_allocation_after(-1);std::cerr<<e.what()<<'\n';return 1;}
}

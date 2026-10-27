#include "fixtures/native_resources/sdk_model.hpp"
#include "openmx_hamiltonian_gpu_state.h"
#include "openmx_orbital_gpu_state.h"
#include "openmx_env.h"
#include <cassert>
#include <cstdint>
#include <climits>
#include <cstring>
#include <iostream>
#include <memory>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

using Type_Orbs_Grid = float;
using MPI_Comm = int;
struct OpenMXAtomicViews {};
struct BindingAbort {};
[[noreturn]] static void binding_abort() { throw BindingAbort{}; }
// Only the borrowed orbital binding is modeled here; its production upload
// and native-owner lifetime are covered by orbital-calculation-state-lifecycle.
struct OpenMXOrbitalGpuState { int unused; };
static void orbital_reset(OpenMXNativeResources*,void*) {}
OpenMXOrbitalGpuState *Set_Orbitals_Grid_GPU_Bind_Resources(OpenMXCoreContext *core) {
    OpenMXNativeResources *owner=nullptr;void *metadata=nullptr;char error[256];
    if(!openmx_core_native_resources_state(core,"Set_Orbitals_Grid_GPU",
        sizeof(OpenMXOrbitalGpuState),orbital_reset,&owner,&metadata,error,sizeof(error)))binding_abort();
    return static_cast<OpenMXOrbitalGpuState*>(metadata);
}
static size_t host_calls=0,host_fail=0,host_live=0,device_calls=0,device_fail=0;
static bool pin_fail=false,copy_fail=false;
static std::array<void*,512> host_records;
static void *host_malloc(size_t n) {
    if (++host_calls==host_fail) return nullptr;
    void *p=std::malloc(n?n:1);assert(p);
    for(auto &slot:host_records) if(!slot) { slot=p;host_live++;return p; }
    throw std::runtime_error("host tracker exhausted");
}
static void *host_calloc(size_t n,size_t z) { void *p=host_malloc(n*z);if(p)std::memset(p,0,n*z);return p; }
static void host_free(void *p) {
    if(!p)return;
    for(auto &slot:host_records)if(slot==p){slot=nullptr;assert(host_live);host_live--;std::free(p);return;}
    throw std::runtime_error("unowned host allocation");
}
static void *Set_Hamiltonian_malloc(size_t n,const char*,int) {
    void *p=host_malloc(n);if(!p)binding_abort();return p;
}
static void Set_Hamiltonian_abort(const char*,const char*,int) { binding_abort(); }
static size_t Set_Hamiltonian_checked_mul(size_t a,size_t b,const char*,int) {
    assert(!a || b<=SIZE_MAX/a);return a*b;
}
static void Set_Hamiltonian_add_array_bytes(size_t *v,size_t n,size_t z,const char *s,int r) {
    size_t x=Set_Hamiltonian_checked_mul(n,z,s,r);assert(x<=SIZE_MAX-*v);*v+=x;
}
static int device_malloc(void **p,size_t n) {
    if(++device_calls==device_fail){*p=nullptr;return 2;}return OpenMXGpuMalloc(p,n);
}
static int copy(void *d,const void *s,size_t n,int) {
    if(copy_fail){copy_fail=false;return 2;}std::memcpy(d,s,n);return 0;
}
struct Mapping { void *host=nullptr,*device=nullptr;int gpu=0,offload=0; };
static std::array<Mapping,128> mappings;
static int omp_device=0;
static int omp_get_default_device(){return omp_device;}
static void omp_set_default_device(int n){omp_device=n;}
static void *mapped(void *p,int d) {
    for(auto &m:mappings)if(m.host==p && m.gpu==current_device && m.offload==d)return m.device;
    return nullptr;
}
static int associate(void *h,void *p,int offload) {
    assert(!mapped(h,offload));
    for(auto &m:mappings)if(!m.host){m={h,p,current_device,offload};return 0;}
    throw std::runtime_error("mapping table exhausted");
}
static void disassociate(void *h,int offload) {
    for(auto &m:mappings)if(m.host==h && m.gpu==current_device && m.offload==offload){m={};return;}
    throw std::runtime_error("mapping removed on the wrong device");
}
#if OPENMX_GPU_HIP
constexpr int hipSuccess=0,hipMemcpyHostToDevice=0;
static int hipMalloc(void **p,size_t n){return device_malloc(p,n);}
static int hipFree(void *p){return OpenMXGpuFree(p);}
static int hipMemcpy(void *d,const void *s,size_t n,int k){return copy(d,s,n,k);}
static int hipHostMalloc(void **p,size_t n,unsigned f){if(pin_fail){*p=nullptr;return 2;}return OpenMXGpuHostAlloc(p,n,f);}
static int hipHostFree(void *p){return OpenMXGpuFreeHost(p);}
static int hipGetLastError(){return 0;}
static int hipDeviceSynchronize(){return 0;}
static int omp_target_associate_ptr(void *h,void *p,size_t,size_t,int d){return associate(h,p,d);}
static void *omp_get_mapped_ptr(void *h,int d){return mapped(h,d);}
static int omp_target_disassociate_ptr(void *h,int d){disassociate(h,d);return 0;}
typedef struct { int wan,no,pad0,pad1;double gx,gy,gz,ax,ay,az; } OTF_Atom;
static size_t Set_Hamiltonian_OTF_Budget(int){return 20000;}
#else
constexpr int cudaSuccess=0,cudaMemcpyHostToDevice=0;
static void *model_acc_malloc(size_t n){
    if(++device_calls==device_fail)return nullptr;
    void *p=nullptr;(void)create(&p,n,3);return p;
}
static int cudaMalloc(void **p,size_t n){return device_malloc(p,n);}
static int cudaFree(void *p){return OpenMXGpuFree(p);}
static int cudaMemcpy(void *d,const void *s,size_t n,int k){return copy(d,s,n,k);}
static int cudaMallocHost(void **p,size_t n){if(pin_fail){*p=nullptr;return 2;}return OpenMXGpuHostAlloc(p,n,0);}
static int cudaFreeHost(void *p){return OpenMXGpuFreeHost(p);}
static int cudaGetLastError(){return 0;}
static int cudaDeviceSynchronize(){return 0;}
static acc_device_t acc_get_device_type(){return acc_device_nvidia;}
static void acc_wait_all(){}
static void acc_clear_freelists(){}
static void *acc_deviceptr(void *h){return mapped(h,acc_device);}
static void acc_map_data(void *h,void *p,size_t){associate(h,p,acc_device);}
static void acc_unmap_data(void *h){disassociate(h,acc_device);}
static void acc_memcpy_to_device(void *d,const void *s,size_t n){std::memcpy(d,s,n);}
struct SOG_GpuPair { int wan,no,pt0,pad;size_t out;double gx,gy,gz,ax,ay,az; };
static int SOG_Device_Prepare(OpenMXOrbitalGpuState*,const OpenMXAtomicViews*){return 1;}
static const void *SOG_Device_Tables(OpenMXOrbitalGpuState*){return reinterpret_cast<void*>(1);}
#endif
#include "openmx_grid_device_scope.h"
static int mpi_calls=0,mpi_comm_level1=0;
static void MPI_Comm_rank(MPI_Comm,int *rank){mpi_calls++;*rank=0;}
static int SpinP_switch=0,Cnt_switch=0,Matomnum=3,Host_ID=0,level_stdout=0;
static int M2G[4]={0,1,2,3},F_G2M[4]={0,1,2,3},G2ID[4]={0};
static int Spe_Total_NO[4]={0,3,5,7},Spe_Total_CNO[4]={0,3,5,7},WhatSpecies[4]={0,1,2,3};
static int GridN_Atom[4]={0,31,47,59},FNAN[4]={0,2,2,2};
static int natn[4][3],ncn[4][3],NumOLG[4][3],glist[4][3][60],*GListTAtoms1[4][3],*GListTAtoms2[4][3];
static int mglist[4][60],*MGridListAtom[4],*GridListAtom[4],*CellListAtom[4];
static int cells[60];
static double Gxyz[4][4],atv[3][4],potential[4][1024],*Vpot_Grid[4];
static float orb_values[4][60][7],*orb_rows[4][60],**Orbs_Grid[4],*Orbs_Grid_FNAN[4][3][60];

// Borrowed read-only public view, exactly as used by density and forces.
#include "hamiltonian_tables.inc"
#include "hamiltonian_state.inc"

using Context=std::unique_ptr<OpenMXCoreContext,decltype(&openmx_core_destroy)>;
static Context calculation(){char error[256];Context c(openmx_core_create_host(error,sizeof(error)),openmx_core_destroy);assert(c);return c;}
static void bind_device(int d){current_device=acc_device=omp_device=d;}
static void geometry() {
    for(int a=1;a<=3;a++){
        MGridListAtom[a]=GridListAtom[a]=mglist[a];CellListAtom[a]=cells;Orbs_Grid[a]=orb_rows[a];
        for(int k=0;k<60;k++){
            mglist[a][k]=a*100+k;orb_rows[a][k]=orb_values[a][k];
            for(int i=0;i<7;i++)orb_values[a][k][i]=float(a*1000+k*10+i);
        }
        for(int h=0;h<3;h++){
            natn[a][h]=h==0?a:1+(a+h)%3;ncn[a][h]=h;
            NumOLG[a][h]=(a*13+h*7)%GridN_Atom[a];GListTAtoms1[a][h]=GListTAtoms2[a][h]=glist[a][h];
            for(int k=0;k<NumOLG[a][h];k++)glist[a][h][k]=(k*11+h)%GridN_Atom[a];
        }
    }
    for(int s=0;s<4;s++){Vpot_Grid[s]=potential[s];for(int k=0;k<1024;k++)potential[s][k]=s*100+k+.25;}
}

static void tables(OpenMXHamiltonianGpuState *s,int mask) {
#if OPENMX_GPU_HIP
    auto &c=s->cache;auto &t=c.t;
    // HIP builds these flat arrays inside its offload routine. Independent
    // fixture data exercises the production association/release boundary.
    c.pair_Mc_AN=(int*)host_calloc(1,sizeof(int));c.pair_h_AN=(int*)host_calloc(1,sizeof(int));
    c.pair_NO0=(int*)host_calloc(1,sizeof(int));c.pair_NO1=(int*)host_calloc(1,sizeof(int));
    c.pair_NOLG=(int*)host_calloc(1,sizeof(int));c.nolg_MN=(int*)host_calloc(1,sizeof(int));
    c.nolg_Nc=(int*)host_calloc(1,sizeof(int));
    c.pair_h_offset=(size_t*)host_calloc(1,sizeof(size_t));c.pair_nolg_offset=(size_t*)host_calloc(1,sizeof(size_t));
    c.pair_orbs0_offset=(size_t*)host_calloc(1,sizeof(size_t));c.pair_orbs1_offset=(size_t*)host_calloc(1,sizeof(size_t));
    c.orbs0buf=(float*)host_calloc(1,sizeof(float));c.orbs1buf=(float*)host_calloc(1,sizeof(float));
    *c.orbs0buf=1.25f;*c.orbs1buf=2.5f;
    void *p[]={c.pair_NO0,c.pair_NO1,c.pair_NOLG,c.nolg_MN,c.nolg_Nc,c.pair_h_offset,
        c.pair_nolg_offset,c.pair_orbs0_offset,c.pair_orbs1_offset,c.orbs0buf,c.orbs1buf};
    if(mask)for(void *v:p)assert(Set_Hamiltonian_ME_Associate(s,v,sizeof(int)));
    t.orbs0buf=c.orbs0buf;t.orbs1buf=c.orbs1buf;
    t.orbs0_resident=t.orbs1_resident=t.nolg_resident=t.meta_resident=!!mask;
    c.ready=1;c.cnt_kind=0;
#else
    auto *c=Set_Hamiltonian_Ensure_OpenACC_MatrixElements_Cache(s,0,0,mask&7);
    SetHamiltonianGpuTurnPlan p{};p.resident_admit_mask=mask;
    if(mask)Set_Hamiltonian_ME_EnterDeviceCache(s,&p);
    assert(c->ready && c->pair_count==9);
    SetHamiltonianMETables t{};assert(Set_Hamiltonian_GetMatrixElementsTables(s,0,&t));
    for(int pair=0;pair<t.pair_count;pair++){
        int a=t.pair_Mc_AN[pair],h=t.pair_h_AN[pair],b=natn[a][h];
        const float *o=t.orbs1_resident?(float*)acc_deviceptr((void*)t.orbs1buf):t.orbs1buf;
        for(int k=0;k<NumOLG[a][h];k++)for(int j=0;j<Spe_Total_NO[b];j++)
            assert(o[t.pair_orbs1_offset[pair]+k*Spe_Total_NO[b]+j]==orb_values[b][glist[a][h][k]][j]);
    }
#endif
}

static void plans(OpenMXHamiltonianGpuState *s) {
    OpenMXAtomicViews atomic{};
#if OPENMX_GPU_HIP
    assert(Set_Hamiltonian_OTF_Ensure(s,&atomic));
    auto *a=SetH_OTF_GetPlan(s,0,1),*b=SetH_OTF_GetPlan(s,1,1);
    assert(a && b && a!=b && a->valid && b->valid);
    size_t n=host_calls;assert(SetH_OTF_GetPlan(s,0,1)==a && host_calls==n);
#else
    assert(Set_Hamiltonian_OTF_Ensure(s,&atomic,0,0));
    unsigned int out[]={0,0,31,78};
    auto *a=SETH_OTF_GetPlan(s,0,0,1,20000,nullptr,0),*b=SETH_OTF_GetPlan(s,1,0,1,20000,out,0);
    assert(a && b && a!=b && a->ready && b->ready);
    auto *stage=SETH_OTF_StagePotential(s,1,1024);assert(stage && stage[1023]==1023.25);
    size_t n=host_calls;assert(SETH_OTF_GetPlan(s,0,0,1,20000,nullptr,0)==a && host_calls==n);
    assert(SETH_OTF_StagePotential(s,1,512)==stage);
#endif
}

int main(){try{
    geometry();
    {auto c=calculation();auto *s=Set_Hamiltonian_GPU_Bind_Resources(c.get());
        assert(s->orbital_state==Set_Orbitals_Grid_GPU_Bind_Resources(c.get()));
        assert(!Set_Hamiltonian_MatrixElementsTables_Ready(s,0));openmx_core_finish(c.get());}
    assert(!queries && !live && !host_live);
    for(int cycle=0;cycle<8;cycle++){
        event_count=0;auto a=calculation(),b=calculation();
        auto *sa=Set_Hamiltonian_GPU_Bind_Resources(a.get()),*sb=Set_Hamiltonian_GPU_Bind_Resources(b.get());
        assert(sa!=sb && sa==Set_Hamiltonian_GPU_Bind_Resources(a.get()));
        bind_device(0);tables(sa,15);plans(sa);
        int db=cycle%2?0:1;bind_device(db);tables(sb,15);plans(sb);
        SetHamiltonianMETables snapshot{};assert(Set_Hamiltonian_GetMatrixElementsTables(sb,0,&snapshot));
        int old_mpi=mpi_calls;Matomnum=0;current_device=7;acc_device=6;omp_device=5;
        openmx_core_native_resources_clear(a.get());
        assert(current_device==7 && acc_device==6 && omp_device==5 && mpi_calls==old_mpi);
        Matomnum=3;bind_device(db);SetHamiltonianMETables still{};
        assert(Set_Hamiltonian_GetMatrixElementsTables(sb,0,&still) && still.orbs1buf==snapshot.orbs1buf);
        event_count=0;sa=Set_Hamiltonian_GPU_Bind_Resources(a.get());bind_device(2);tables(sa,15);plans(sa);
        current_device=7;acc_device=6;omp_device=5;Matomnum=0;openmx_core_finish(a.get());a.reset();b.reset();Matomnum=3;
        assert(!live && !host_live && current_device==7 && acc_device==6 && omp_device==5);
        for(auto &m:mappings)assert(!m.host);
    }
#if !OPENMX_GPU_HIP
    for(int mask:{0,1,2,4,7,15}){
        event_count=0;auto c=calculation();auto *s=Set_Hamiltonian_GPU_Bind_Resources(c.get());bind_device(0);
        int q=queries;tables(s,mask);if(!mask)assert(queries==q);
        if(mask){assert(Set_Hamiltonian_Yield_Resident_DeviceCache(s));assert(!s->cache.host_discarded);
            assert(Set_Hamiltonian_MatrixElementsTables_Ready(s,0));}
        current_device=7;acc_device=6;c.reset();assert(!live && !host_live);
    }
    // All flat-array allocation failures leave a destructible owner.
    for(int f=1;f<=13;f++){
        event_count=0;auto c=calculation();auto *s=Set_Hamiltonian_GPU_Bind_Resources(c.get());bind_device(0);
        host_calls=0;host_fail=f;bool failed=false;
        try{tables(s,0);}catch(const BindingAbort&){failed=true;}
        host_fail=0;assert(failed);c.reset();assert(!live && !host_live);
    }
#endif
#if OPENMX_GPU_HIP
    // Recoverable pageable-plan and sphere-index allocation failures.
    pin_fail=true;
    for(int f=1;f<=30;f++){
        event_count=0;auto c=calculation();auto *s=Set_Hamiltonian_GPU_Bind_Resources(c.get());bind_device(0);
        OpenMXAtomicViews atomic{};host_calls=0;host_fail=f;
        if(Set_Hamiltonian_OTF_Ensure(s,&atomic)){
            (void)SetH_OTF_GetPlan(s,0,1);(void)SetH_OTF_GetPlan(s,1,1);
        }
        host_fail=0;current_device=7;acc_device=6;omp_device=5;c.reset();assert(!live && !host_live);
    }
    pin_fail=false;
    {event_count=0;auto c=calculation();auto *s=Set_Hamiltonian_GPU_Bind_Resources(c.get());bind_device(0);
        OpenMXAtomicViews atomic{};copy_fail=true;assert(!Set_Hamiltonian_OTF_Ensure(s,&atomic));
        assert(!live && !host_live);c.reset();}
#else
    // Failure after direct mappings exist must still leave a releasable owner.
    {event_count=0;auto c=calculation();auto *s=Set_Hamiltonian_GPU_Bind_Resources(c.get());bind_device(0);
        host_calls=0;host_fail=14;bool failed=false;
        try{tables(s,7);}catch(const BindingAbort&){failed=true;}
        host_fail=0;assert(failed);current_device=7;acc_device=6;c.reset();assert(!live && !host_live);}
#endif
    // Device allocation and pinned allocation failures are recoverable. The
    // two backends keep their native policies for pageable plan fallbacks.
    for(int f=1;f<=3;f++){
        event_count=0;auto c=calculation();auto *s=Set_Hamiltonian_GPU_Bind_Resources(c.get());bind_device(0);
        OpenMXAtomicViews atomic{};device_calls=0;device_fail=f;
#if OPENMX_GPU_HIP
        assert(!Set_Hamiltonian_OTF_Ensure(s,&atomic));
#else
        assert(!Set_Hamiltonian_OTF_Ensure(s,&atomic,0,0));
#endif
        device_fail=0;assert(!live && !host_live);plans(s);
        current_device=7;acc_device=6;omp_device=5;c.reset();assert(!live && !host_live);
    }
    {event_count=0;auto c=calculation();auto *s=Set_Hamiltonian_GPU_Bind_Resources(c.get());bind_device(0);
        pin_fail=true;
#if OPENMX_GPU_HIP
        plans(s);assert(!s->pinned_bytes);
#else
        OpenMXAtomicViews atomic{};assert(Set_Hamiltonian_OTF_Ensure(s,&atomic,0,0));
        auto *p=SETH_OTF_GetPlan(s,0,0,1,20000,nullptr,0);assert(p && p->hbuf_pageable);
        assert(!SETH_OTF_StagePotential(s,1,1024));
#endif
        pin_fail=false;c.reset();assert(!live && !host_live);}
    for(int f=0;f<6;f++){
        event_count=0;auto c=calculation();openmx::test::fail_allocation_after(f);
        try{(void)Set_Hamiltonian_GPU_Bind_Resources(c.get());}catch(const BindingAbort&){}
        openmx::test::fail_allocation_after(-1);c.reset();assert(!live && !host_live);
    }
    // Retained-device reuse on another primary GPU must fail explicitly.
    {event_count=0;auto c=calculation();auto *s=Set_Hamiltonian_GPU_Bind_Resources(c.get());bind_device(0);plans(s);
        pid_t child=fork();assert(child>=0);
        if(!child){if(!freopen("/dev/null","w",stderr))_exit(2);bind_device(1);OpenMX_GridDeviceCapture(&s->affinity);_exit(0);}
        int status;assert(waitpid(child,&status,0)==child && WIFSIGNALED(status) && WTERMSIG(status)==SIGABRT);
        c.reset();assert(!live && !host_live);}
    assert(!host_live && !live);
    std::cout<<"PASS: Hamiltonian "<<(OPENMX_GPU_HIP?"HIP":"CUDA")
        <<" 8 dual contexts, same/different devices, real core clear/finish/destroy, isolated tables/plans, failures and restored bindings\n";
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}

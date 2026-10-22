#include "bridge_under_test.cu"
#include "allocation_faults.hpp"
#include <iostream>
#include <stdexcept>

extern "C" { int level_stdout=0; }
using namespace gpu_views_model;
using openmx::gpu::DeviceBuffer;
namespace {
void check(bool ok,const char *message){if(!ok)throw std::runtime_error(message);}
constexpr int alloc_failed=3,internal_error=14;
void clear_caches(){
#if !OPENMX_GPU_HIP
    openmx_gemmul8ReleasePrepared();
#endif
    openmx_gemmul8ReleaseWorkspaces();
    decltype(g_workspaces) empty;g_workspaces.swap(empty);
}
int normal(void **out){
    WorkspaceReport report;
#if OPENMX_GPU_HIP
    return ensure_workspace<false>(nullptr,3,2,4,15,out,&report,0);
#else
    return ensure_workspace<false>(nullptr,3,2,4,15,false,out,&report);
#endif
}
int scratch(void **out){
#if OPENMX_GPU_HIP
    return ensure_scratch(nullptr,mock_worksize,out,0);
#else
    return ensure_scratch(nullptr,mock_worksize,out);
#endif
}
void explicit_status_and_retry(){
    reset(1);DeviceBuffer<double> buffer;
    check(buffer.allocate_status(SIZE_MAX)==OPENMX_GPU_INVALID_VALUE,"typed overflow status lost");
    check(buffer.allocate_status(17)==0 && buffer.allocate_status(1)==OPENMX_GPU_INVALID_VALUE,"live storage silently replaced");
    void *pointer=buffer.get();
    reset(3);trace.fail_kind=release;trace.fail_at=1;
    check(buffer.try_reset()==MODEL_ERROR && buffer.get()==pointer && buffer.size()==17 && trace.current==3,
        "failed free consumed cache ownership or binding");
    reset(3);check(buffer.try_reset()==0 && !buffer && !live && trace.current==3,"failed free could not be retried");
    for(gpu_views_model::Call kind:{get_device,set_device}){
        reset(1);check(buffer.allocate_status(9)==0,"binding case allocation failed");
        reset(3);trace.fail_kind=kind;trace.fail_at=1;
        check(buffer.try_reset()==MODEL_ERROR && buffer && trace.current==3,"failed release binding lost ownership");
        reset(3);check(buffer.try_reset()==0 && !live && trace.current==3,"binding failure could not be retried");
    }
    reset(1);check(buffer.allocate(9),"restore case allocation failed");
    reset(3);trace.fail_kind=set_device;trace.fail_at=2;
    check(buffer.try_reset()==MODEL_ERROR && !buffer && !buffer.size() && !live && trace.current==3,
        "successful free/failed restore retained a dangling pointer");
    const int frees=trace.count[release];check(buffer.try_reset()==0 && trace.count[release]==frees,"restore retry freed twice");
    reset(0);std::cout<<"PASS: explicit SDK status and 4 release/binding/restore failure paths\n";
}
void reuse_and_cleanup(){
    clear_caches();mock_worksize=128;reset(1);mock_stream=reinterpret_cast<void*>(1);
    void *first=nullptr,*again=nullptr;check(normal(&first)==0 && first,"normal cache allocation failed");
    const int allocations=trace.count[allocate];check(normal(&again)==0 && again==first && trace.count[allocate]==allocations,"cache hit reallocated");
    mock_worksize=64;check(normal(&again)==0 && again==first,"smaller request lost retained capacity");
    mock_worksize=256;check(scratch(&again)==0 && g_workspaces.size()==1 && live==1,"scratch growth leaked old allocation");
    trace.current=2;check(normal(&again)==0 && g_workspaces.size()==2 && live==2,"devices shared a workspace");
    mock_stream=reinterpret_cast<void*>(2);check(normal(&again)==0 && g_workspaces.size()==3 && live==3,"streams shared a workspace");
    trace.current=3;openmx_gemmul8ReleaseWorkspaces();
    check(g_workspaces.empty() && !live && !trace.wrong_device_frees && trace.current==3,"global release used the caller device");
    try{
        std::unordered_map<WorkspaceKey,Workspace,WorkspaceKeyHash> local;
        trace.current=2;check(local[WorkspaceKey{2,mock_stream}].allocate(32),"local cache allocation failed");
        trace.current=0;throw std::runtime_error("scope exit");
    }catch(const std::runtime_error&){}
    check(!live && trace.current==0,"cache exception/destruction leaked storage");
    reset(0);std::cout<<"PASS: reuse/growth, separate device/stream caches, caller-device cleanup and exception destruction\n";
}
void release_failure_and_trim(){
    mock_worksize=128;reset(1);mock_stream=nullptr;void *out=nullptr;
    check(scratch(&out)==0,"retained release case allocation failed");
    reset(3);trace.fail_kind=release;trace.fail_at=1;openmx_gemmul8ReleaseWorkspaces();
    check(g_workspaces.size()==1 && live==1 && trace.current==3,"failed cache release was discarded");
    reset(3);openmx_gemmul8ReleaseWorkspaces();check(g_workspaces.empty() && !live,"cache release retry leaked");
    // The production cap has a 256 MiB minimum. No data are read/written;
    // the SDK model allocates uninitialized virtual storage for the trim.
    mock_worksize=(size_t(256)<<20)+16;reset(1);check(scratch(&out)==0 && live==1,"large scratch allocation failed");
    trace.current=3;openmx_gemmul8TrimWorkspaces();
    check(!live && trace.current==3 && !trace.wrong_device_frees,"trim did not release on the allocation device");
    clear_caches();reset(0);std::cout<<"PASS: retained free failure/retry and production 256 MiB scratch trim\n";
}
void preparation_failures(){
    int sdk=0,host=0;
    for(auto function:{normal,scratch}){
        clear_caches();mock_worksize=128;reset(1);mock_stream=nullptr;void *out=nullptr;
        openmx::test::fail_allocation_after(10000);const int ok=function(&out);
        const long allocations=10000-openmx::test::remaining_allocations();openmx::test::fail_allocation_after(-1);
        check(ok==0 && out,"allocation control failed");const auto observed=trace.count;
        clear_caches();
        for(gpu_views_model::Call kind:{get_device,allocate})for(int at=1;at<=observed[kind];++at){
            clear_caches();reset(1);trace.fail_kind=kind;trace.fail_at=at;out=nullptr;
            const int status=function(&out);
            check((status==alloc_failed || status==internal_error) && !out && !live && trace.current==1,
                "SDK preparation failure published or retained a workspace");
            clear_caches();++sdk;
        }
        for(long at=0;at<allocations;++at){
            clear_caches();reset(1);out=nullptr;const long heap=openmx::test::live_allocations();
            openmx::test::fail_allocation_after(at);const int status=function(&out);openmx::test::fail_allocation_after(-1);
            check(status==alloc_failed && !out && !live && trace.current==1,"cache metadata OOM escaped or leaked");
            clear_caches();check(openmx::test::live_allocations()==heap,"cache metadata OOM retained host allocations");++host;
        }
    }
    clear_caches();mock_worksize=128;reset(1);void *out=nullptr;
    mock_free=0;check(normal(&out)==alloc_failed && !out && !live,"reserve policy was bypassed");
    mock_free=size_t(40)<<30;mock_total=128;
    check(normal(&out)==alloc_failed && !out && !live,"rank fraction policy was bypassed");
    mock_total=size_t(40)<<30;clear_caches();reset(0);
    std::cout<<"PASS: normal/scratch caches; "<<sdk<<" SDK preparation failures, "<<host<<" metadata allocation failures, reserve/fraction declines\n";
}
#if !OPENMX_GPU_HIP
void prepared_matrices(){
    clear_caches();reset(0);const PreparedKey key{0,7,1,0};
    Prepared *entry=ensure_prepared(key,64);check(entry && entry->buffer.size()==64,"prepared allocation failed");
    void *pointer=entry->buffer.get();entry->valid=true;entry->version=19;
    check(ensure_prepared(key,32)==entry && entry->buffer.get()==pointer && entry->valid && entry->version==19,"prepared reuse lost metadata");
    entry=ensure_prepared(key,128);check(entry && !entry->valid && !entry->version && live==1,"prepared growth retained stale metadata");
    entry->valid=true;reset(0);trace.fail_kind=release;trace.fail_at=1;
    check(!ensure_prepared(key,256) && entry->buffer.size()==128 && !entry->valid && live==1,"failed prepared release lost storage or allowed reuse");
    reset(0);entry=ensure_prepared(key,256);check(entry && entry->buffer.size()==256 && live==1,"prepared release retry failed");
    trace.current=1;check(ensure_prepared(PreparedKey{1,8,0,1},64) && live==2,"complex/other-device prepared allocation failed");
    long long counters[5];double seconds[2];check(openmx_gemmul8ForwardCounters(counters,seconds)==320,"prepared retention byte count changed");
    trace.current=3;openmx_gemmul8ReleasePrepared();check(!live && g_prepared.empty() && trace.current==3 && !trace.wrong_device_frees,"prepared cleanup used the caller's device");
    {
        Prepared scoped;check(scoped.buffer.allocate(32),"scoped prepared allocation failed");trace.current=0;
    }
    check(!live && trace.current==0,"scoped prepared storage leaked");
    decltype(g_prepared) empty;g_prepared.swap(empty);const long heap=openmx::test::live_allocations();
    openmx::test::fail_allocation_after(0);Prepared *failed=ensure_prepared(key,64);openmx::test::fail_allocation_after(-1);
    check(!failed && !live && openmx::test::live_allocations()==heap,"prepared metadata OOM escaped or leaked");
    reset(0);std::cout<<"PASS: prepared reuse/version/growth, failed free/retry, per-device cleanup, scoped destruction and metadata OOM\n";
}
void prepared_owner_lifetimes(){
    clear_caches();reset(1);int first,second;
    const PreparedKey a{1,0,0,0,&first},b{1,0,0,0,&second},legacy{1,0,0,0};
    auto* x=ensure_prepared(a,32);auto* y=ensure_prepared(b,32);auto* z=ensure_prepared(legacy,32);
    check(x && y && z && x!=y && y!=z && live==3,"owner identities shared prepared storage");
    x->valid=y->valid=z->valid=true;x->version=y->version=z->version=1;
    void* saved=y->buffer.get();trace.current=3;
    openmx_gemmul8ReleasePreparedOwner(&first);
    check(live==2 && !g_prepared.count(a) && g_prepared.at(b).buffer.get()==saved &&
        g_prepared.at(b).valid && g_prepared.at(legacy).valid && trace.current==3 && !trace.wrong_device_frees,
        "owner teardown invalidated another calculation/legacy cache or selected wrong device");
    openmx_gemmul8ReleasePreparedOwner(&first);check(live==2,"owner teardown was not idempotent");
    x=ensure_prepared(a,32);check(x && !x->valid && !x->version,"reused owner identity inherited a stale generation");
    openmx_gemmul8ReleasePreparedOwner(&second);check(!g_prepared.count(b) && live==2,"second owner teardown retained storage");
    clear_caches();check(!live,"owner cache cleanup leaked");reset(0);
    std::cout<<"PASS: distinct calculation/legacy namespaces, local generations, selective/idempotent drain and reused owner identity\n";
}
#endif
}
int main(){
    try{
        check(!live && !trace.count[get_device],"cache initialization touched SDK");
        openmx_gemmul8SetEnabled(1);setenv("OPENMX_GEMMUL8_MIN_FREE_AFTER_MB","0",1);
        setenv("OPENMX_GEMMUL8_MAX_WORKSPACE_MB","256",1);
        explicit_status_and_retry();reuse_and_cleanup();release_failure_and_trim();preparation_failures();
#if !OPENMX_GPU_HIP
        prepared_matrices();
        prepared_owner_lifetimes();
#endif
        clear_caches();check(!live && !trace.wrong_device_frees,"cache test retained or freed foreign device storage");return 0;
    }catch(const std::exception& error){openmx::test::fail_allocation_after(-1);std::cerr<<error.what()<<'\n';return 1;}
}

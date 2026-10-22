#include "fixtures/gpu_views_model.hpp"
#include "openmx_gpu_views.h"
#include "allocation_faults.hpp"
#include <iostream>
#include <limits>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>

using namespace gpu_views_model;
extern "C" int gpu_views_c_layout(size_t,size_t,size_t);
extern "C" int gpu_views_c_lifetime(const void*,size_t,int);
extern "C" int gpu_views_test_select(int device){return OpenMXGpuSetDevice(device);}
extern "C" int gpu_views_test_current(){return trace.current;}
namespace {
void check(bool ok,const char *message){if(!ok)throw std::runtime_error(message);}
struct Buffer {
    unsigned char *data=nullptr;size_t bytes;int device;
    Buffer(size_t n,int d):bytes(n),device(d){
        const int saved=trace.current;trace.current=d;
        check(!OpenMXGpuMalloc(reinterpret_cast<void**>(&data),n),"model source allocation failed");trace.current=saved;
        for(size_t i=0;i<n;++i)data[i]=static_cast<unsigned char>(17*i+3);
    }
    ~Buffer(){const int saved=trace.current;trace.current=device;(void)OpenMXGpuFree(data);trace.current=saved;}
};
void layouts_and_empty(){
    static_assert(std::is_standard_layout_v<OpenMXGpuReadView>);
    static_assert(!std::is_copy_constructible_v<OpenMXGpuReadView>);
    static_assert(std::is_nothrow_move_constructible_v<OpenMXGpuReadView>);
    check(gpu_views_c_layout(sizeof(OpenMXGpuReadView),offsetof(OpenMXGpuReadView,allocated),offsetof(OpenMXGpuReadView,device)),"C/C++ view layouts differ");
    struct Legacy {void *pointer;int allocated;};
    if(sizeof(void*)==8)check(sizeof(OpenMXGpuReadView)==sizeof(Legacy),"64-bit view ABI grew");
    reset(2);OpenMXGpuReadView view;
    check(openmx_gpu_read_view(&view,nullptr,0) && !view.pointer && !view.allocated && view.device==-1,"empty view changed");
    check(openmx_gpu_copy_device(nullptr,nullptr,0),"empty copy touched runtime");
    check(!openmx_gpu_read_view(nullptr,nullptr,0),"null view accepted");
    openmx_gpu_read_view_release(nullptr);
    unsigned char host=0;
    check(!openmx_gpu_read_view(&view,&host,1) && !view.pointer && !view.allocated,"host pointer accepted");
    check(!openmx_gpu_copy_device(&host,&host,std::numeric_limits<size_t>::max()),"overflowing copy accepted");
    check(!trace.count[get_device] && !trace.count[allocate] && trace.current==2,"empty/invalid view initialized a device");
}
void borrowing_and_copy(){
    Buffer source(151,0),destination(153,0);setenv("OPENMX_GPU_WORKER_COPY","0",1);
    possible=false;reset(0);{
        OpenMXGpuReadView view;check(openmx_gpu_read_view(&view,source.data,source.bytes),"same-device view failed");
        check(view.pointer==source.data && !view.allocated && !trace.count[allocate] && !trace.count[set_device],"same-device view allocated or switched");
    }
    possible=true;reset(1);{
        OpenMXGpuReadView view;check(openmx_gpu_read_view(&view,source.data,source.bytes),"peer view failed");
        check(view.pointer==source.data && !view.allocated && !trace.count[allocate],"peer view copied");
        openmx_gpu_read_view_release(&view);check(!view.pointer && trace.current==1,"borrow release changed binding");
        check(openmx_gpu_read_view(&view,source.data,source.bytes) && !trace.pending,"existing peer mapping left an error");
    }
    reset(3);check(openmx_gpu_copy_device(destination.data+1,source.data,source.bytes),"same-device copy from a different binding failed");
    check(trace.current==3 && trace.count[set_device]==2 && !std::memcmp(destination.data+1,source.data,source.bytes),"same-device copy did not restore its caller");
    check(destination.data[0]==3 && destination.data[152]==static_cast<unsigned char>(17*152+3),"copy overwrote guards");
    Buffer remote(151,1);reset(3);
    check(openmx_gpu_copy_device(remote.data,source.data,source.bytes) && trace.count[peer_copy]==1 && !trace.count[copy] && trace.current==3,"peer transfer changed policy or binding");
    reset(3);trace.fail_kind=peer_copy;trace.fail_at=1;
    check(!openmx_gpu_copy_device(remote.data,source.data,source.bytes) && trace.current==3 && !trace.pending,"failed peer copy leaked binding/error");
    reset(1);trace.fail_kind=peer_enable;trace.fail_at=1;{
        OpenMXGpuReadView view;check(openmx_gpu_read_view(&view,source.data,source.bytes) && view.allocated,"failed peer enable did not recover with a private copy");
        check(!std::memcmp(view.pointer,source.data,source.bytes),"peer-enable recovery lost data");
    }
    reset(0);
}
void ownership(){
    Buffer source(153,0);setenv("OPENMX_GPU_WORKER_COPY","1",1);possible=true;
    const int baseline=live;reset(1);{
        OpenMXGpuReadView first;check(openmx_gpu_read_view(&first,source.data,source.bytes),"owned view failed");
        check(first.allocated && first.device==1 && first.pointer!=source.data,"owned view lost allocation device");
        void *pointer=first.pointer;OpenMXGpuReadView moved(std::move(first));
        check(!first.pointer && !first.allocated && moved.pointer==pointer,"move duplicated ownership");
        reset(2);OpenMXGpuReadView second;check(openmx_gpu_read_view(&second,source.data,source.bytes),"second owned view failed");
        reset(3);second=std::move(moved);check(live==baseline+1 && trace.current==3,"move assignment failed to release destination");
        check(second.pointer==pointer && !moved.pointer,"move assignment lost allocation");
        second=std::move(second);check(second.pointer==pointer && live==baseline+1,"self move released storage");
    }
    check(live==baseline && trace.current==3 && !trace.wrong_device_frees,"scope exit freed on the caller's device");
    reset(1);try {
        OpenMXGpuReadView view;check(openmx_gpu_read_view(&view,source.data,source.bytes),"exception view preparation failed");
        trace.current=2;throw std::runtime_error("native caller exception");
    }catch(const std::runtime_error&){}
    check(live==baseline && trace.current==2,"exception did not release/restore the owned view");
    reset(1);check(gpu_views_c_lifetime(source.data,source.bytes,3)==1 && live==baseline,"C release/device restoration failed");
    reset(1);trace.fail_kind=copy;trace.fail_at=1;
    check(gpu_views_c_lifetime(source.data,source.bytes,3)==0 && live==baseline && trace.current==1,"C failed preparation retained storage");
    reset(0);
}
void bounded_staging_and_failures(){
    constexpr size_t limit=16U*1024*1024,n=limit+29;
    Buffer source(n,0),destination(n,1);possible=false;setenv("OPENMX_GPU_WORKER_COPY","1",1);
    const int baseline=live;reset(1);std::array<int,calls> observed;
    {
        OpenMXGpuReadView view;
        openmx::test::fail_allocation_after(10000);
        const int ok=openmx_gpu_read_view(&view,source.data,n);
        const long allocations=10000-openmx::test::remaining_allocations();openmx::test::fail_allocation_after(-1);
        check(ok && allocations==1,"staging must use one host allocation");
        check(trace.download_max==limit && trace.upload_max==limit && trace.count[copy]==4,"staging exceeded 16 MiB or lost the tail");
        check(!std::memcmp(view.pointer,source.data,n),"staged data differs");observed=trace.count;
    }
    int failures=0;
    for(Call kind:{get_device,set_device,attributes,allocate,copy})for(int at=1;at<=observed[kind];++at){
        reset(1);trace.fail_kind=kind;trace.fail_at=at;const long heap=openmx::test::live_allocations();
        OpenMXGpuReadView view;
        check(!openmx_gpu_read_view(&view,source.data,n),"injected preparation/copy error reported success");
        check(!view.pointer && !view.allocated && view.device==-1,"failed view published ownership");
        check(trace.current==1 && live==baseline && !trace.wrong_device_frees,"failed view leaked or changed the caller device");
        check(openmx::test::live_allocations()==heap,"failed staging leaked host memory");++failures;
    }
    reset(3);const long heap=openmx::test::live_allocations();openmx::test::fail_allocation_after(0);
    const int ok=openmx_gpu_copy_device(destination.data,source.data,n);openmx::test::fail_allocation_after(-1);
    check(!ok && trace.current==3 && openmx::test::live_allocations()==heap,"host OOM escaped C ABI or lost device restoration");
    reset(1);OpenMXGpuReadView view;openmx::test::fail_allocation_after(0);
    const int prepared=openmx_gpu_read_view(&view,source.data,n);openmx::test::fail_allocation_after(-1);
    check(!prepared && !view.pointer && !view.allocated && live==baseline && trace.current==1,"host OOM retained the device allocation");
    // The final switch itself fails once: the guard retries during destruction.
    possible=true;setenv("OPENMX_GPU_WORKER_COPY","0",1);reset(3);trace.fail_kind=set_device;trace.fail_at=2;
    check(!openmx_gpu_copy_device(destination.data,source.data,n) && trace.current==3,"failed restore did not run guard cleanup");
    std::cout<<"PASS: bounded 16 MiB staging; "<<failures<<" SDK failures, 2 host OOM paths, restore failure\n";
    reset(0);
}
void concurrent(){
    Buffer source(4099,0);possible=true;setenv("OPENMX_GPU_WORKER_COPY","0",1);
    const int baseline=live;std::array<int,4> ok{};std::array<std::thread,4> threads;
    for(int i=0;i<4;++i)threads[i]=std::thread([&,i]{
        reset(i);for(int iteration=0;iteration<64;++iteration){
            OpenMXGpuReadView view;
            if(!openmx_gpu_read_view(&view,source.data,source.bytes) || view.allocated || view.pointer!=source.data)return;
        }
        if(trace.current==i && !trace.pending)ok[i]=1;
    });
    for(auto& thread:threads)thread.join();for(int value:ok)check(value==1,"concurrent peer view failed");
    possible=false;setenv("OPENMX_GPU_WORKER_COPY","1",1);
    for(int i=0;i<4;++i)threads[i]=std::thread([&,i]{
        reset(1+i%3);for(int iteration=0;iteration<16;++iteration){
            OpenMXGpuReadView view;
            if(!openmx_gpu_read_view(&view,source.data,source.bytes) || !view.allocated || std::memcmp(view.pointer,source.data,source.bytes))return;
            const int saved=trace.current;trace.current=0;openmx_gpu_read_view_release(&view);
            if(trace.current!=0 || trace.wrong_device_frees)return;trace.current=saved;
        }
        ok[i]=2;
    });
    for(auto& thread:threads)thread.join();for(int value:ok)check(value==2,"concurrent private view failed");
    check(live==baseline,"concurrent views leaked");reset(0);
}
}
int main(){
    try {
        layouts_and_empty();borrowing_and_copy();ownership();bounded_staging_and_failures();concurrent();
        check(!live,"test retained device storage");
        std::cout<<"PASS: C/C++ layout, borrowed/owned/moved views, exception/device cleanup, 320 concurrent views\n";return 0;
    }catch(const std::exception& error){openmx::test::fail_allocation_after(-1);std::cerr<<error.what()<<'\n';return 1;}
}

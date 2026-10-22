#include "fixtures/gpu_worker_model.hpp"
#include "openmx_gpu_memory.hpp"
#include "openmx_gpu_dm.h"
#include "openmx_gpu_transform.h"
#include "openmx_gpu_workers.h"
#include "allocation_faults.hpp"
#include <iostream>
#include <stdexcept>
#include <thread>

using namespace gpu_views_model;
using openmx::gpu::DeviceBuffer;
namespace {
void check(bool ok,const char *message){if(!ok)throw std::runtime_error(message);}
std::array<int,calls> before_commit{};
long host_before_commit=-1;
int commits=0;
bool omit_release=false;
constexpr int group=3;
}
extern "C" void openmx_gpu_workers_scope(OpenMXGpuWorkers *p,size_t n){
    p->items=n;p->count=int(std::min(n,size_t(group)));
    for(int i=0;i<p->count;++i)p->devices[i]=i;
}
extern "C" int openmx_gpu_workers_capacity_device(OpenMXGpuWorkers *p,size_t n,int,
    OpenMXGpuWorkerDeviceCost cost,void *context){
    openmx_gpu_workers_scope(p,n);
    for(int i=0;i<p->count;++i)if(cost(context,i,p->count,p->devices[i])==SIZE_MAX)return 0;
    return 1;
}
extern "C" int openmx_gpu_workers_run(const OpenMXGpuWorkers *p,
    int (*callback)(void*,int,int),void *context,const char*){
    const int saved=trace.current;bool ok=true;
    std::array<bool,OPENMX_GPU_WORKERS_MAX> started{};
    for(int phase=OPENMX_GPU_WORKER_PREPARE;phase<=OPENMX_GPU_WORKER_RELEASE;++phase){
        if(!ok && phase!=OPENMX_GPU_WORKER_RELEASE)continue;
        // Model an executor returning before its usual cleanup. Work's own
        // destructors must still release all partially/completely prepared buffers.
        if(omit_release && phase==OPENMX_GPU_WORKER_EXECUTE){trace.current=saved;return 0;}
        if(phase==OPENMX_GPU_WORKER_COMMIT){
            before_commit=trace.count;host_before_commit=openmx::test::remaining_allocations();
        }
        for(int i=0;i<p->count;++i){
            if(phase==OPENMX_GPU_WORKER_RELEASE && !started[i])continue;
            trace.current=p->devices[i];
            if(phase==OPENMX_GPU_WORKER_PREPARE)started[i]=true;
            if(phase==OPENMX_GPU_WORKER_COMMIT)++commits;
            ok=bool(callback(context,i,phase)) && ok;
        }
    }
    trace.current=saved;return ok;
}
extern "C" int openmx_gpu_workers_run_prepared(const OpenMXGpuWorkers *p,
    int (*callback)(void*,int,int),void *context,const char *label){
    return openmx_gpu_workers_run(p,callback,context,label);
}
namespace {
void buffer_lifetimes(){
    static_assert(!std::is_copy_constructible_v<DeviceBuffer<double>>);
    static_assert(std::is_nothrow_move_constructible_v<DeviceBuffer<double>>);
    static_assert(std::is_nothrow_move_assignable_v<DeviceBuffer<double>>);
    reset(2);const long heap=openmx::test::live_allocations();
    {
        DeviceBuffer<double> empty;
        check(empty.allocate(0) && !empty && !empty.size() && empty.device()==-1,"empty storage touched SDK");
        check(!empty.allocate(SIZE_MAX/sizeof(double)+1),"typed allocation overflow accepted");
        check(!trace.count[get_device] && !trace.count[allocate],"invalid storage initialized a device");
        check(empty.allocate(23) && empty.size()==23 && empty.device()==2,"typed storage shape/device wrong");
        check(!empty.allocate(1),"live allocation replaced without reset");
        check(!trace.count[fill] && !trace.count[copy],"device allocation implicitly initialized storage");
        void *pointer=empty.get();DeviceBuffer<double> first(std::move(empty));
        check(!empty && !empty.size() && first.get()==pointer,"buffer move duplicated ownership");
        trace.current=1;DeviceBuffer<double> second;check(second.allocate(7),"second buffer allocation failed");
        trace.current=3;second=std::move(first);
        check(!first && !first.size() && live==1 && second.get()==pointer && trace.current==3,"move assignment lost allocation device");
        second=std::move(second);second.reset();second.reset();
        check(!second && !second.size() && second.device()==-1 && !live && trace.current==3,"reset was not idempotent");
        check(second.allocate(9) && second.device()==3,"reset buffer not reusable");
    }
    check(!live && !trace.wrong_device_frees && openmx::test::live_allocations()==heap,"buffer scope leaked or allocated host bookkeeping");
    for(Call kind:{get_device,allocate}){
        reset(1);trace.fail_kind=kind;trace.fail_at=1;DeviceBuffer<int> buffer;
        check(!buffer.allocate(9) && !buffer && !buffer.size() && !live,"failed typed allocation retained storage");
    }
    reset(1);try {
        DeviceBuffer<int> buffer;check(buffer.allocate(17),"exception allocation failed");
        trace.current=3;throw std::runtime_error("unwind buffer");
    }catch(const std::runtime_error&){}
    check(!live && trace.current==3,"exception cleanup lost caller binding");
    std::array<std::thread,4> threads;std::array<bool,4> ok{};
    for(int i=0;i<4;++i)threads[i]=std::thread([&,i]{
        reset(i);for(int repeat=0;repeat<64;++repeat){
            DeviceBuffer<int> buffer;if(!buffer.allocate(19))return;
            trace.current=(i+1)%4;buffer.reset();
            if(trace.current!=(i+1)%4 || trace.wrong_device_frees)return;
            trace.current=i;
        }ok[i]=true;
    });
    for(auto& thread:threads)thread.join();for(bool value:ok)check(value,"concurrent typed storage failed");
    check(!live,"concurrent buffers leaked");reset(0);
    std::cout<<"PASS: typed/moved/empty/reused/overflow buffers, 2 allocation failures, exception and 256 concurrent lifetimes\n";
}

struct DmCase {
    static constexpr int n=4,states=3,entries=5;
    std::array<ModelComplex,2*n*states> input{};
    std::array<int,entries> left{0,3,1,2,0},right{3,1,2,0,1},phase{0,1,0,1,1};
    std::array<double,states> occ{0.7,0.2,0.1},energy{-1,2,3},partial{0.3,0.6,0.4};
    double real[2]{0.8,0.6},imag[2]{-0.6,0.8};
    std::array<std::array<double,entries>,10> output{};
    DmCase(){for(size_t i=0;i<input.size();++i)input[i]={double(i+1)*0.07,double(i%5)*-0.13};clear();}
    void clear(){for(auto& channel:output)channel.fill(-931.5);}
    int run(bool cpu=false){
        double *targets[10];for(int c=0;c<10;++c)targets[c]=output[c].data();
        auto function=cpu?openmx_gpu_dm_entries:openmx_gpu_dm_try_entries;
        return function(OPENMX_GPU_DM_NC,n,states,states,entries,input.data(),1,left.data(),right.data(),
            phase.data(),2,real,imag,occ.data(),energy.data(),partial.data(),targets,1);
    }
    bool unchanged()const{for(auto& channel:output)for(double value:channel)if(value!=-931.5)return false;return true;}
};
struct TransformCase {
    DeviceBuffer<double> input,output;
    TransformCase(){
        reset(0);check(input.allocate(36) && output.allocate(48),"transform input preparation failed");
        for(int i=0;i<36;++i)input.get()[i]=i*0.17-1;clear();
    }
    void clear(){for(int i=0;i<48;++i)output.get()[i]=-643.5;}
    int run(){return openmx_gpu_transform(OPENMX_GPU_TRANSPOSE,0,6,6,input.get(),6,nullptr,0,output.get(),8,0,0);}
    bool unchanged()const{for(int i=0;i<48;++i)if(output.get()[i]!=-643.5)return false;return true;}
};
template<class Case> void failures(Case& test,const char *label){
    const int base=live;const long heap=openmx::test::live_allocations();
    test.clear();reset(3);commits=0;before_commit={};
    openmx::test::fail_allocation_after(10000);const int passed=test.run();
    // COMMIT failures are fatal in the native executor and may have published
    // earlier chunks. Inject recoverable failures only before publication.
    const long allocations=10000-host_before_commit;openmx::test::fail_allocation_after(-1);
    check(passed && commits==group && live==base && trace.current==3,"control worker execution failed");
    const auto observed=before_commit;
    int sdk_failures=0;
    for(Call kind:{allocate,copy,fill,peek})for(int at=1;at<=observed[kind];++at){
        test.clear();reset(3);trace.fail_kind=kind;trace.fail_at=at;commits=0;
        const int result=test.run();
        check(!result && !commits && test.unchanged(),"failed worker published incomplete results");
        check(live==base && trace.current==3 && !trace.wrong_device_frees,"failed worker leaked GPU storage or binding");
        check(openmx::test::live_allocations()==heap,"failed worker leaked host storage");++sdk_failures;
    }
    for(long at=0;at<allocations;++at){
        test.clear();reset(3);commits=0;openmx::test::fail_allocation_after(at);
        const int result=test.run();openmx::test::fail_allocation_after(-1);
        check(!result && !commits && test.unchanged(),"host OOM published incomplete results");
        check(live==base && trace.current==3 && openmx::test::live_allocations()==heap,"host OOM leaked worker storage");
    }
    test.clear();reset(3);commits=0;omit_release=true;
    const int early=test.run();omit_release=false;
    check(!early && !commits && test.unchanged() && live==base && trace.current==3,"early executor return bypassed batch ownership");
    check(openmx::test::live_allocations()==heap,"early executor return leaked host storage");
    std::cout<<"PASS: "<<label<<"; "<<sdk_failures<<" SDK failures, "<<allocations<<" host allocation failures, omitted RELEASE phase\n";
}
void worker_lifetimes(){
    setenv("OPENMX_GPU_WORKER_COPY","1",1);possible=false;
    DmCase density;failures(density,"density matrix");
    density.clear();reset(3);trace.fail_kind=allocate;trace.fail_at=2;trace.fail_status=OPENMX_GPU_OUT_OF_MEMORY;
    check(density.run(true) && !live && trace.current==3,"late GPU OOM did not recover on CPU");
    {
        TransformCase transform;failures(transform,"dense transform");
    }
    check(!live,"worker tests retained device storage");reset(0);
}
}
int main(){
    try{buffer_lifetimes();worker_lifetimes();return 0;}
    catch(const std::exception& error){openmx::test::fail_allocation_after(-1);std::cerr<<error.what()<<'\n';return 1;}
}

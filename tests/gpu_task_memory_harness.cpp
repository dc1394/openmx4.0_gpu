#include "fixtures/gpu_task_model.hpp"
#include "openmx_gpu_task_worker.cpp"
#include <iostream>
#include <stdexcept>

extern "C" void *__real_malloc(size_t);
extern "C" void __real_free(void*);
namespace {
using namespace gpu_views_model;
int host_calls=0,host_fail_at=0,host_live=0;
std::array<void*,128> host_pointers{};
std::array<int,calls> before_commit{};
int host_before_commit=0,commits=0;
bool omit_release=false,throw_before_execute=false;
constexpr int group=3;
void check(bool ok,const char *message){if(!ok)throw std::runtime_error(message);}
}
extern "C" void *__wrap_malloc(size_t bytes){
    if(gpu_task_model::sdk_allocation)return __real_malloc(bytes);
    if(++host_calls==host_fail_at)return nullptr;
    void *pointer=__real_malloc(bytes);
    if(pointer)for(auto& entry:host_pointers)if(!entry){entry=pointer;++host_live;return pointer;}
    return pointer;
}
extern "C" void __wrap_free(void *pointer){
    if(pointer)for(auto& entry:host_pointers)if(entry==pointer){entry=nullptr;--host_live;break;}
    __real_free(pointer);
}
extern "C" int openmx_gpu_workers_capacity(OpenMXGpuWorkers *p,size_t n,int,
    OpenMXGpuWorkerCost cost,void *context){
    p->items=n;p->count=int(std::min(n,size_t(group)));
    // Model a pool where the other layouts do not fit. A failed preparation
    // must decline this layout without publishing any output.
    for(int i=0;i<p->count;++i){p->devices[i]=i;if(cost(context,i,p->count)==SIZE_MAX)return 0;}
    return 1;
}
extern "C" int openmx_gpu_workers_run_prepared(const OpenMXGpuWorkers *p,
    int(*callback)(void*,int,int),void *context,const char*){
    const int saved=trace.current;bool ok=true;
    for(int phase=OPENMX_GPU_WORKER_PREPARE;phase<=OPENMX_GPU_WORKER_RELEASE;++phase){
        if(!ok && phase!=OPENMX_GPU_WORKER_RELEASE)continue;
        if(phase==OPENMX_GPU_WORKER_EXECUTE && (omit_release || throw_before_execute)){
            trace.current=saved;
            if(throw_before_execute)throw std::runtime_error("executor unwind");
            return 0;
        }
        if(phase==OPENMX_GPU_WORKER_COMMIT){before_commit=trace.count;host_before_commit=host_calls;}
        for(int i=0;i<p->count;++i){
            trace.current=p->devices[i];
            if(phase==OPENMX_GPU_WORKER_COMMIT)++commits;
            ok=bool(callback(context,i,phase)) && ok;
        }
    }
    trace.current=saved;return ok;
}
namespace {
void kernel(size_t first,size_t n,const void*,void *const views[],int){
    const auto *dense=static_cast<const float*>(views[0]);
    const auto *offsets=static_cast<const size_t*>(views[1]);
    const auto *items=static_cast<const int*>(openmx_gpu_task_ragged_data(views[1],n));
    const auto *lookup=static_cast<const int*>(views[2]);
    auto *out=static_cast<double*>(views[3]),*reduced=static_cast<double*>(views[4]);
    auto *scratch=static_cast<double*>(views[5]);reduced[0]=reduced[1]=0;
    for(size_t i=0;i<n;++i){
        scratch[i]=2*dense[i]+lookup[(first+i)%3];
        for(size_t k=offsets[i];k<offsets[i+1];++k)scratch[i]+=items[k];
        out[i]=scratch[i];reduced[0]+=out[i];reduced[1]+=first+i;
    }
}
struct Case {
    static constexpr size_t n=7;
    float dense[n]{.5f,1.f,-2.f,3.f,4.5f,0.f,-1.f};
    size_t offsets[n+1]{0,0,2,3,3,3,6,7};
    int items[7]{2,-1,3,7,8,-5,4},lookup[3]{-2,1,4};
    double output[n+2],reduced[2];
    OpenMXGpuTask task{};
    Case(){
        task.first=11;task.count=n;task.inputs=3;task.outputs=2;task.scratch=1;task.kernel=kernel;
        task.view[0]={dense,0,sizeof(float),nullptr,0};
        task.view[1]={items,0,0,offsets,sizeof(int)};
        task.view[2]={lookup,sizeof(lookup),0,nullptr,0};
        task.view[3]={output+1,0,sizeof(double),nullptr,0};
        task.view[4]={reduced,sizeof(reduced),0,nullptr,0};
        task.view[5]={nullptr,0,sizeof(double),nullptr,0};clear();
    }
    void clear(){for(double& value:output)value=-731;reduced[0]=reduced[1]=0;}
    int run(){return openmx_gpu_task_local(&task,"task-lifecycle");}
    bool unchanged()const{for(double value:output)if(value!=-731)return false;return reduced[0]==0 && reduced[1]==0;}
    void verify()const{
        double sum=0,indices=0;
        for(size_t i=0;i<n;++i){
            double expected=2*dense[i]+lookup[(11+i)%3];
            for(size_t k=offsets[i];k<offsets[i+1];++k)expected+=items[k];
            check(output[i+1]==expected,"dense/CSR/lookup result differs");sum+=expected;indices+=11+i;
        }
        check(output[0]==-731 && output[n+1]==-731 && reduced[0]==sum && reduced[1]==indices,"guards or reduction differ");
    }
};
void begin(){reset(3);host_calls=host_fail_at=commits=0;before_commit={};host_before_commit=0;}
void verify_empty(){
    begin();OpenMXGpuTask empty{};
    check(openmx_gpu_task_local(&empty,"empty") && !trace.count[get_device] && !host_calls,"empty task touched SDK/heap");
}
void failures(){
    Case test;begin();check(test.run()==1 && commits==group,"control task failed");test.verify();
    check(!live && !host_live && trace.current==3 && !trace.wrong_device_frees,"control task leaked storage/binding");
    const auto observed=before_commit;const int host_sites=host_before_commit;int sdk_sites=0;
    for(Call kind:{get_device,allocate,copy})for(int at=1;at<=observed[kind];++at){
        test.clear();begin();trace.fail_kind=kind;trace.fail_at=at;
        check(!test.run() && !commits && test.unchanged(),"preparation failure published task output");
        check(!live && !host_live && trace.current==3 && !trace.wrong_device_frees,"failed task leaked storage/binding");++sdk_sites;
    }
    for(int at=1;at<=host_sites;++at){
        test.clear();begin();host_fail_at=at;
        check(!test.run() && !commits && test.unchanged(),"host OOM published task output");
        check(!live && !host_live && trace.current==3,"host OOM leaked task buffers");
    }
    test.clear();begin();omit_release=true;check(!test.run(),"omitted cleanup returned success");omit_release=false;
    check(test.unchanged() && !live && !host_live && trace.current==3,"omitted RELEASE leaked buffers");
    test.clear();begin();throw_before_execute=true;bool caught=false;
    try{(void)test.run();}catch(const std::runtime_error&){caught=true;}
    throw_before_execute=false;
    check(caught && test.unchanged() && !live && !host_live && trace.current==3,"executor exception bypassed scope cleanup");
    std::cout<<"PASS: actual task worker; "<<sdk_sites<<" SDK preparation failures, "<<host_sites
        <<" host allocation failures, omitted RELEASE and exception cleanup; dense/CSR/lookup/scratch/reduction guards\n";
}
void release_retry(){
    begin();Case test;
    {
        TaskWork work;work.task=&test.task;work.pool.count=1;work.pool.items=Case::n;
        trace.current=2;check(task_worker(&work,0,OPENMX_GPU_WORKER_PREPARE),"release retry preparation failed");
        reset(0);trace.fail_kind=release;trace.fail_at=1;
        check(!task_worker(&work,0,OPENMX_GPU_WORKER_RELEASE) && work.device[0] && live==1 && trace.current==0,
            "failed explicit release lost device ownership");
        reset(0);check(task_worker(&work,0,OPENMX_GPU_WORKER_RELEASE) && !live && trace.current==0,"release retry failed");
    }
    check(!host_live && !trace.wrong_device_frees,"release retry retained host storage or foreign frees");
    std::cout<<"PASS: task buffer release on its allocation device, failure retention and retry\n";
}
}
int main(){try{verify_empty();failures();release_retry();return 0;}
    catch(const std::exception& error){host_fail_at=0;std::cerr<<error.what()<<'\n';return 1;}}

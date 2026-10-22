#pragma once
// One translation unit per test executable. The SDK model owns independent
// storage; only host scratch allocations participate in heap fault injection.
#include "gpu_task_model.hpp"
#include "openmx_gpu_workers.h"
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
void begin(){reset(3);host_calls=host_fail_at=commits=0;before_commit={};host_before_commit=0;}
}

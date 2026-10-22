#pragma once
// Independent SDK model. Production transport and the public C/C++ view are
// compiled unchanged; physical CUDA/HIP execution is covered separately.
#define OPENMX_GPU_RUNTIME_H
#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <mutex>

using OpenMXGpuError=int;
enum { OPENMX_GPU_SUCCESS=0,OPENMX_GPU_DEVICE_MEMORY=1,
       OPENMX_GPU_OUT_OF_MEMORY=2,
       OPENMX_GPU_INVALID_VALUE=11,
       OPENMX_GPU_PEER_ALREADY_ENABLED=3,MODEL_ERROR=7,
       OPENMX_GPU_COPY_DEVICE_TO_DEVICE=0,OPENMX_GPU_COPY_DEVICE_TO_HOST=1,
       OPENMX_GPU_COPY_HOST_TO_DEVICE=2 };
struct OpenMXGpuPointerAttributes { int device,type; };
namespace gpu_views_model {
enum Call { get_device,set_device,attributes,allocate,release,copy,peer_possible,peer_enable,peer_copy,fill,peek,calls };
struct Trace {
    int current=0,pending=0,fail_kind=-1,fail_at=0,fail_status=MODEL_ERROR,wrong_device_frees=0;
    std::array<int,calls> count{};
    size_t download_max=0,upload_max=0;
};
inline thread_local Trace trace;
struct Allocation { void *pointer=nullptr;size_t bytes=0;int device=-1; };
inline std::array<Allocation,128> allocations;
inline std::mutex mutex;
inline bool possible=false,enabled[4][4]{};
inline int live=0;
inline bool fail(Call kind) {
    const int at=++trace.count[kind];
    if(trace.fail_kind==kind && trace.fail_at==at){trace.pending=trace.fail_status;return true;}
    return false;
}
inline Allocation* find(const void *pointer,size_t bytes=0) {
    const auto address=reinterpret_cast<std::uintptr_t>(pointer);
    for(auto& a:allocations){
        const auto base=reinterpret_cast<std::uintptr_t>(a.pointer);
        if(a.pointer && address>=base && address-base<=a.bytes && bytes<=a.bytes-(address-base))return &a;
    }
    return nullptr;
}
inline void reset(int device) { trace={};trace.current=device; }
inline int error(){trace.pending=MODEL_ERROR;return MODEL_ERROR;}
}
inline int OpenMXGpuGetDevice(int *device) {
    using namespace gpu_views_model;if(fail(get_device))return MODEL_ERROR;*device=trace.current;return 0;
}
inline int OpenMXGpuSetDevice(int device) {
    using namespace gpu_views_model;if(fail(set_device))return MODEL_ERROR;
    if(device<0 || device>=4)return error();trace.current=device;return 0;
}
inline int OpenMXGpuGetLastError() { auto& t=gpu_views_model::trace;int e=t.pending;t.pending=0;return e; }
inline int OpenMXGpuPointerGetAttributes(OpenMXGpuPointerAttributes *out,const void *pointer) {
    using namespace gpu_views_model;if(fail(attributes))return MODEL_ERROR;
    std::lock_guard<std::mutex> lock(mutex);auto* a=find(pointer);
    *out=a?OpenMXGpuPointerAttributes{a->device,OPENMX_GPU_DEVICE_MEMORY}:OpenMXGpuPointerAttributes{-1,0};return 0;
}
inline int OpenMXGpuMalloc(void **out,size_t bytes) {
    using namespace gpu_views_model;*out=nullptr;if(fail(allocate))return MODEL_ERROR;
    std::lock_guard<std::mutex> lock(mutex);
    for(auto& a:allocations)if(!a.pointer){
        void *p=std::malloc(bytes?bytes:1);if(!p)return error();
        a={p,bytes?bytes:1,trace.current};*out=p;++live;return 0;
    }
    return error();
}
inline int OpenMXGpuFree(void *pointer) {
    using namespace gpu_views_model;if(fail(release))return MODEL_ERROR;if(!pointer)return 0;
    std::lock_guard<std::mutex> lock(mutex);auto* a=find(pointer);
    if(!a || a->pointer!=pointer)return error();
    if(a->device!=trace.current){++trace.wrong_device_frees;return error();}
    std::free(a->pointer);*a={};--live;return 0;
}
inline int OpenMXGpuMemcpy(void *destination,const void *source,size_t bytes,int kind) {
    using namespace gpu_views_model;if(fail(copy))return MODEL_ERROR;
    std::lock_guard<std::mutex> lock(mutex);auto* dst=find(destination,bytes);auto* src=find(source,bytes);
    if(kind==OPENMX_GPU_COPY_DEVICE_TO_DEVICE){
        if(!src || !dst || src->device!=trace.current || dst->device!=trace.current)return error();
    }else if(kind==OPENMX_GPU_COPY_DEVICE_TO_HOST){
        if(!src || src->device!=trace.current)return error();trace.download_max=std::max(trace.download_max,bytes);
    }else if(kind==OPENMX_GPU_COPY_HOST_TO_DEVICE){
        if(!dst || dst->device!=trace.current)return error();trace.upload_max=std::max(trace.upload_max,bytes);
    }else return error();
    std::memcpy(destination,source,bytes);return 0;
}
inline int OpenMXGpuDeviceCanAccessPeer(int *out,int,int) {
    using namespace gpu_views_model;if(fail(peer_possible))return MODEL_ERROR;*out=possible;return 0;
}
inline int OpenMXGpuDeviceEnablePeerAccess(int source,unsigned) {
    using namespace gpu_views_model;if(fail(peer_enable))return MODEL_ERROR;
    if(enabled[trace.current][source]){trace.pending=OPENMX_GPU_PEER_ALREADY_ENABLED;return OPENMX_GPU_PEER_ALREADY_ENABLED;}
    enabled[trace.current][source]=true;return 0;
}
inline int OpenMXGpuMemcpyPeer(void *destination,int target,const void *source,int origin,size_t bytes) {
    using namespace gpu_views_model;if(fail(peer_copy))return MODEL_ERROR;
    std::lock_guard<std::mutex> lock(mutex);auto* dst=find(destination,bytes);auto* src=find(source,bytes);
    if(!src || !dst || src->device!=origin || dst->device!=target || trace.current!=target)return error();
    std::memcpy(destination,source,bytes);return 0;
}

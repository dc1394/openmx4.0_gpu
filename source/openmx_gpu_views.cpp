#include "openmx_gpu_backend.h"
#include "openmx_env.h"
#include "openmx_gpu_runtime.h"
#include "openmx_gpu_views.h"

#include <cstddef>
#include <limits>
#include <memory>

namespace {
/* SDK bindings are thread state. Every temporary switch is paired on all
   exits, including host allocation failure. OpenACC/OpenMP bindings are not
   changed by this transport layer. */
class DeviceBinding {
 public:
    explicit DeviceBinding(int device) noexcept {
        if(OpenMXGpuGetDevice(&saved_)!=OPENMX_GPU_SUCCESS)return;
        current_=saved_;
        valid_=select(device);
    }
    ~DeviceBinding() noexcept { if(changed_)(void)OpenMXGpuSetDevice(saved_); }
    DeviceBinding(const DeviceBinding&)=delete;
    DeviceBinding& operator=(const DeviceBinding&)=delete;
    explicit operator bool() const noexcept { return valid_; }
    bool select(int device) noexcept {
        if(current_==device)return true;
        changed_=true;
        if(OpenMXGpuSetDevice(device)!=OPENMX_GPU_SUCCESS)return false;
        current_=device;changed_=current_!=saved_;return true;
    }
    bool restore() noexcept {
        if(changed_){
            if(OpenMXGpuSetDevice(saved_)!=OPENMX_GPU_SUCCESS)return false;
            changed_=false;current_=saved_;
        }
        return valid_;
    }
 private:
    int saved_=-1,current_=-1;
    bool changed_=false,valid_=false;
};
struct DeviceFree {
    int device;
    void operator()(void *pointer) const noexcept {
        if(!pointer)return;
        DeviceBinding binding(device);
        if(binding)(void)OpenMXGpuFree(pointer);
    }
};
using DeviceOwner=std::unique_ptr<void,DeviceFree>;
void clear_view(OpenMXGpuReadView *view) noexcept
{ view->pointer=nullptr;view->allocated=0;view->device=-1; }

/* CUDA peer enablement is context state. Serialize its first use; accept a
   mapping installed by another OpenMX backend and never disable it here. */
static int gpu_peer_access(int target,int source)
{
    int ok=0;
    if (target==source) return 1;
    if (openmx_env_flag_at(OPENMX_ENV_GPU_WORKER_COPY, 0)) return 0;
#pragma omp critical(openmx_worker_peer_enable)
    {
        int possible=0;
        if (OpenMXGpuDeviceCanAccessPeer(&possible,target,source)==OPENMX_GPU_SUCCESS && possible) {
            OpenMXGpuError status=OpenMXGpuDeviceEnablePeerAccess(source,0);
            ok=status==OPENMX_GPU_SUCCESS || status==OPENMX_GPU_PEER_ALREADY_ENABLED;
            if (status!=OPENMX_GPU_SUCCESS) (void)OpenMXGpuGetLastError();
        } else (void)OpenMXGpuGetLastError();
    }
    return ok;
}

static int device_pointer(const void *pointer,int *device)
{
    OpenMXGpuPointerAttributes attributes;
    if (OpenMXGpuPointerGetAttributes(&attributes,pointer)!=OPENMX_GPU_SUCCESS) {
        (void)OpenMXGpuGetLastError();return 0;
    }
    if (attributes.type!=OPENMX_GPU_DEVICE_MEMORY) return 0;
    *device=attributes.device;
    return 1;
}
} // namespace

extern "C" int openmx_gpu_copy_device(void *destination,const void *source,size_t bytes) noexcept
{
    if (!bytes) return 1;
    if(bytes>std::size_t(std::numeric_limits<std::ptrdiff_t>::max()))return 0;
    int target=-1,origin=-1,ok=0;
    if (!destination || !source || !device_pointer(destination,&target) ||
        !device_pointer(source,&origin)) return 0;
    try {
    DeviceBinding binding(target);
    if(!binding)return 0;
    if(target==origin) {
        ok=OpenMXGpuMemcpy(destination,source,bytes,OPENMX_GPU_COPY_DEVICE_TO_DEVICE)==OPENMX_GPU_SUCCESS;
    } else if (gpu_peer_access(target,origin)) {
        ok=OpenMXGpuMemcpyPeer(destination,target,source,origin,bytes)==OPENMX_GPU_SUCCESS;
    } else {
        constexpr size_t limit=16U*1024*1024;
        const size_t capacity=bytes<limit?bytes:limit;
        /* Every byte is overwritten by download before upload. Keep staging
           bounded and avoid value-initializing an extra 16 MiB per copy. */
        std::unique_ptr<unsigned char[]> staging(new unsigned char[capacity]);
        ok=1;
        for (size_t offset=0;ok && offset<bytes;) {
            const size_t chunk=bytes-offset<capacity ? bytes-offset : capacity;
            ok=binding.select(origin) &&
               OpenMXGpuMemcpy(staging.get(),(const char*)source+offset,chunk,OPENMX_GPU_COPY_DEVICE_TO_HOST)==OPENMX_GPU_SUCCESS &&
               binding.select(target) &&
               OpenMXGpuMemcpy((char*)destination+offset,staging.get(),chunk,OPENMX_GPU_COPY_HOST_TO_DEVICE)==OPENMX_GPU_SUCCESS;
            offset+=chunk;
        }
    }
    if(!binding.restore())ok=0;
    if (!ok) (void)OpenMXGpuGetLastError();
    return ok;
    }catch(...) {
        (void)OpenMXGpuGetLastError();return 0;
    }
}

extern "C" int openmx_gpu_read_view(OpenMXGpuReadView *view,const void *source,size_t bytes) noexcept
{
    if(!view)return 0;
    int current=-1,origin=-1;
    clear_view(view);
    if (!bytes) return 1;
    if(bytes>std::size_t(std::numeric_limits<std::ptrdiff_t>::max()))return 0;
    if (!source || !device_pointer(source,&origin) || OpenMXGpuGetDevice(&current)!=OPENMX_GPU_SUCCESS) return 0;
    if (gpu_peer_access(current,origin)) { view->pointer=(void*)source;view->device=origin;return 1; }
    void *pointer=nullptr;
    const auto status=OpenMXGpuMalloc(&pointer,bytes);
    DeviceOwner owner(pointer,DeviceFree{current});
    if(status!=OPENMX_GPU_SUCCESS || !pointer) {
        (void)OpenMXGpuGetLastError();return 0;
    }
    if(!openmx_gpu_copy_device(pointer,source,bytes))return 0;
    view->pointer=owner.release();view->allocated=1;view->device=current;
    return 1;
}

extern "C" void openmx_gpu_read_view_release(OpenMXGpuReadView *view) noexcept
{
    if(!view)return;
    DeviceOwner owner(view->allocated?view->pointer:nullptr,DeviceFree{view->device});
    clear_view(view);
}

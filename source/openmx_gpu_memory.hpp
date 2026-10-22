#ifndef OPENMX_GPU_MEMORY_HPP
#define OPENMX_GPU_MEMORY_HPP

#include "openmx_gpu_runtime.h"

#include <cstddef>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>

namespace openmx::gpu {
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

/* Native linear storage: no host allocation, zero fill, implicit kernel
   launch or Kokkos initialization. The current SDK device owns each new
   allocation. Synchronize consumers before reset/destruction. SDK errors
   remain available to the caller's existing recovery policy. */
template<class T> class DeviceBuffer {
    static_assert(std::is_trivially_copyable_v<T> && !std::is_const_v<T>);
 public:
    DeviceBuffer() noexcept=default;
    DeviceBuffer(const DeviceBuffer&)=delete;
    DeviceBuffer& operator=(const DeviceBuffer&)=delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept
        :memory_(std::move(other.memory_)),count_(std::exchange(other.count_,0)) {}
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if(this!=&other){
            memory_=std::move(other.memory_);
            count_=std::exchange(other.count_,0);
        }
        return *this;
    }
    bool allocate(std::size_t count) noexcept {
        if(memory_ || count>std::numeric_limits<std::size_t>::max()/sizeof(T))return false;
        if(!count)return true;
        int device=-1;
        if(OpenMXGpuGetDevice(&device)!=OPENMX_GPU_SUCCESS)return false;
        void *pointer=nullptr;
        const auto status=OpenMXGpuMalloc(&pointer,count*sizeof(T));
        DeviceOwner pending(pointer,DeviceFree{device});
        if(status!=OPENMX_GPU_SUCCESS || !pointer)return false;
        memory_.get_deleter().device=device;
        memory_.reset(static_cast<T*>(pending.release()));
        count_=count;
        return true;
    }
    void reset() noexcept { memory_.reset();count_=0; }
    T *get() const noexcept { return memory_.get(); }
    std::size_t size() const noexcept { return count_; }
    int device() const noexcept { return memory_?memory_.get_deleter().device:-1; }
    explicit operator bool() const noexcept { return bool(memory_); }
 private:
    std::unique_ptr<T,DeviceFree> memory_{nullptr,DeviceFree{-1}};
    std::size_t count_=0;
};
} // namespace openmx::gpu
#endif

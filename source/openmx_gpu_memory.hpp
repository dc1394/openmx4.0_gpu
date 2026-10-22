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
    explicit DeviceBinding(int device,OpenMXGpuError *error=nullptr) noexcept {
        const auto status=OpenMXGpuGetDevice(&saved_);
        if(error)*error=status;
        if(status!=OPENMX_GPU_SUCCESS)return;
        current_=saved_;
        valid_=select(device,error);
    }
    ~DeviceBinding() noexcept { if(changed_)(void)OpenMXGpuSetDevice(saved_); }
    DeviceBinding(const DeviceBinding&)=delete;
    DeviceBinding& operator=(const DeviceBinding&)=delete;
    explicit operator bool() const noexcept { return valid_; }
    bool select(int device,OpenMXGpuError *error=nullptr) noexcept {
        if(current_==device){if(error)*error=OPENMX_GPU_SUCCESS;return true;}
        changed_=true;
        const auto status=OpenMXGpuSetDevice(device);
        if(error)*error=status;
        if(status!=OPENMX_GPU_SUCCESS)return false;
        current_=device;changed_=current_!=saved_;return true;
    }
    bool restore(OpenMXGpuError *error=nullptr) noexcept {
        if(changed_){
            const auto status=OpenMXGpuSetDevice(saved_);
            if(error)*error=status;
            if(status!=OPENMX_GPU_SUCCESS)return false;
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
        return allocate_status(count)==OPENMX_GPU_SUCCESS;
    }
    OpenMXGpuError allocate_status(std::size_t count) noexcept {
        if(memory_ || count>std::numeric_limits<std::size_t>::max()/sizeof(T))return OPENMX_GPU_INVALID_VALUE;
        if(!count)return OPENMX_GPU_SUCCESS;
        int device=-1;
        const auto queried=OpenMXGpuGetDevice(&device);
        if(queried!=OPENMX_GPU_SUCCESS)return queried;
        void *pointer=nullptr;
        const auto status=OpenMXGpuMalloc(&pointer,count*sizeof(T));
        DeviceOwner pending(pointer,DeviceFree{device});
        if(status!=OPENMX_GPU_SUCCESS)return status;
        if(!pointer)return OPENMX_GPU_OUT_OF_MEMORY;
        memory_.get_deleter().device=device;
        memory_.reset(static_cast<T*>(pending.release()));
        count_=count;
        return OPENMX_GPU_SUCCESS;
    }
    /* A retained cache can retry a failed explicit release. Successful SDK
       free consumes ownership even when restoring the caller fails; a
       subsequent retry must never free the same pointer twice. */
    OpenMXGpuError try_reset() noexcept {
        if(!memory_)return OPENMX_GPU_SUCCESS;
        OpenMXGpuError status=OPENMX_GPU_SUCCESS;
        DeviceBinding binding(device(),&status);
        if(!binding)return status;
        status=OpenMXGpuFree(memory_.get());
        if(status==OPENMX_GPU_SUCCESS){(void)memory_.release();count_=0;}
        OpenMXGpuError restored=OPENMX_GPU_SUCCESS;
        if(!binding.restore(&restored) && status==OPENMX_GPU_SUCCESS)status=restored;
        return status;
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

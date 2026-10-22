#ifndef OPENMX_GPU_VIEWS_H
#define OPENMX_GPU_VIEWS_H

#include "openmx_gpu_backend.h"
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#define OPENMX_GPU_VIEW_NOEXCEPT noexcept
#else
#define OPENMX_GPU_VIEW_NOEXCEPT
#endif
/* Read-only peer access when available; otherwise an owned device copy.
   Contexts and source inputs stay owned by their original caller. A C caller
   releases each initialized view; C++ additionally releases on scope exit and
   transfers ownership only by move. Release an owned view before reusing it
   as an output. Failed preparation leaves an empty view. */
typedef struct OpenMXGpuReadView OpenMXGpuReadView;
void openmx_gpu_read_view_release(OpenMXGpuReadView *view) OPENMX_GPU_VIEW_NOEXCEPT;
struct OpenMXGpuReadView {
    void *pointer;
    int allocated;
    int device;
#ifdef __cplusplus
    OpenMXGpuReadView() noexcept :pointer(nullptr),allocated(0),device(-1) {}
    ~OpenMXGpuReadView() noexcept { if(allocated)openmx_gpu_read_view_release(this); }
    OpenMXGpuReadView(const OpenMXGpuReadView&)=delete;
    OpenMXGpuReadView& operator=(const OpenMXGpuReadView&)=delete;
    OpenMXGpuReadView(OpenMXGpuReadView&& other) noexcept
        :pointer(other.pointer),allocated(other.allocated),device(other.device) {
        other.pointer=nullptr;other.allocated=0;other.device=-1;
    }
    OpenMXGpuReadView& operator=(OpenMXGpuReadView&& other) noexcept {
        if(this!=&other){
            if(allocated)openmx_gpu_read_view_release(this);
            pointer=other.pointer;allocated=other.allocated;device=other.device;
            other.pointer=nullptr;other.allocated=0;other.device=-1;
        }
        return *this;
    }
#endif
};
int openmx_gpu_read_view(OpenMXGpuReadView *view,const void *source,size_t bytes) OPENMX_GPU_VIEW_NOEXCEPT;
/* Explicit cross-device transfer, with bounded host staging when necessary.
   Call only after producer kernels complete. Neither CUDA nor OpenACC
   binding changes on return, including a failed copy. */
int openmx_gpu_copy_device(void *destination,const void *source,size_t bytes) OPENMX_GPU_VIEW_NOEXCEPT;
#ifdef __cplusplus
}
#endif
#undef OPENMX_GPU_VIEW_NOEXCEPT
#endif

#include "openmx_gpu_backend.h"
#include "openmx_gpu_runtime.h"
#include "openmx_gpu_views.h"

#include <stdlib.h>
#include <string.h>

/* CUDA peer enablement is context state. Serialize its first use; accept a
   mapping installed by another OpenMX backend and never disable it here. */
static int gpu_peer_access(int target,int source)
{
    int ok=0;
    if (target==source) return 1;
    const char *copy=getenv("OPENMX_GPU_WORKER_COPY");
    if (copy && atoi(copy)) return 0;
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

int openmx_gpu_copy_device(void *destination,const void *source,size_t bytes)
{
    int target=-1,origin=-1,saved=-1,ok=0;
    if (!bytes) return 1;
    if (!destination || !source || !device_pointer(destination,&target) ||
        !device_pointer(source,&origin) || OpenMXGpuGetDevice(&saved)!=OPENMX_GPU_SUCCESS) return 0;
    if (target==origin) return OpenMXGpuMemcpy(destination,source,bytes,OPENMX_GPU_COPY_DEVICE_TO_DEVICE)==OPENMX_GPU_SUCCESS;
    if (OpenMXGpuSetDevice(target)!=OPENMX_GPU_SUCCESS) return 0;
    if (gpu_peer_access(target,origin)) {
        ok=OpenMXGpuMemcpyPeer(destination,target,source,origin,bytes)==OPENMX_GPU_SUCCESS;
    } else {
        const size_t capacity=bytes<(size_t)16*1024*1024 ? bytes : (size_t)16*1024*1024;
        void *staging=malloc(capacity);
        ok=staging!=NULL;
        for (size_t offset=0;ok && offset<bytes;) {
            const size_t chunk=bytes-offset<capacity ? bytes-offset : capacity;
            ok=OpenMXGpuSetDevice(origin)==OPENMX_GPU_SUCCESS &&
               OpenMXGpuMemcpy(staging,(const char*)source+offset,chunk,OPENMX_GPU_COPY_DEVICE_TO_HOST)==OPENMX_GPU_SUCCESS &&
               OpenMXGpuSetDevice(target)==OPENMX_GPU_SUCCESS &&
               OpenMXGpuMemcpy((char*)destination+offset,staging,chunk,OPENMX_GPU_COPY_HOST_TO_DEVICE)==OPENMX_GPU_SUCCESS;
            offset+=chunk;
        }
        free(staging);
    }
    if (OpenMXGpuSetDevice(saved)!=OPENMX_GPU_SUCCESS) ok=0;
    if (!ok) (void)OpenMXGpuGetLastError();
    return ok;
}

int openmx_gpu_read_view(OpenMXGpuReadView *view,const void *source,size_t bytes)
{
    int current=-1,origin=-1;
    view->pointer=NULL;view->allocated=0;
    if (!bytes) return 1;
    if (!source || !device_pointer(source,&origin) || OpenMXGpuGetDevice(&current)!=OPENMX_GPU_SUCCESS) return 0;
    if (gpu_peer_access(current,origin)) { view->pointer=(void*)source;return 1; }
    if (OpenMXGpuMalloc(&view->pointer,bytes)!=OPENMX_GPU_SUCCESS) {
        (void)OpenMXGpuGetLastError();view->pointer=NULL;return 0;
    }
    view->allocated=1;
    return openmx_gpu_copy_device(view->pointer,source,bytes);
}

void openmx_gpu_read_view_release(OpenMXGpuReadView *view)
{
    if (view->allocated && view->pointer) (void)OpenMXGpuFree(view->pointer);
    view->pointer=NULL;view->allocated=0;
}

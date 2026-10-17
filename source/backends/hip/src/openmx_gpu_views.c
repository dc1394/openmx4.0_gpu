#include "openmx_gpu_views.h"
#include "hip_runtime_compat.h"
#include <stdlib.h>
#include <string.h>

/* HIP peer enablement is context state. Serialize its first use; accept a
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
        if (hipDeviceCanAccessPeer(&possible,target,source)==hipSuccess && possible) {
            hipError_t status=hipDeviceEnablePeerAccess(source,0);
            ok=status==hipSuccess || status==hipErrorPeerAccessAlreadyEnabled;
            if (status!=hipSuccess) (void)hipGetLastError();
        } else (void)hipGetLastError();
    }
    return ok;
}

static int device_pointer(const void *pointer,int *device)
{
    hipPointerAttribute_t attributes;
    if (hipPointerGetAttributes(&attributes,pointer)!=hipSuccess) {
        (void)hipGetLastError();return 0;
    }
    if (attributes.type!=hipMemoryTypeDevice) return 0;
    *device=attributes.device;
    return 1;
}

int openmx_gpu_copy_device(void *destination,const void *source,size_t bytes)
{
    int target=-1,origin=-1,saved=-1,ok=0;
    if (!bytes) return 1;
    if (!destination || !source || !device_pointer(destination,&target) ||
        !device_pointer(source,&origin) || hipGetDevice(&saved)!=hipSuccess) return 0;
    if (target==origin) return hipMemcpy(destination,source,bytes,hipMemcpyDeviceToDevice)==hipSuccess;
    if (hipSetDevice(target)!=hipSuccess) return 0;
    if (gpu_peer_access(target,origin)) {
        ok=hipMemcpyPeer(destination,target,source,origin,bytes)==hipSuccess;
    } else {
        const size_t capacity=bytes<(size_t)16*1024*1024 ? bytes : (size_t)16*1024*1024;
        void *staging=malloc(capacity);
        ok=staging!=NULL;
        for (size_t offset=0;ok && offset<bytes;) {
            const size_t chunk=bytes-offset<capacity ? bytes-offset : capacity;
            ok=hipSetDevice(origin)==hipSuccess &&
               hipMemcpy(staging,(const char*)source+offset,chunk,hipMemcpyDeviceToHost)==hipSuccess &&
               hipSetDevice(target)==hipSuccess &&
               hipMemcpy((char*)destination+offset,staging,chunk,hipMemcpyHostToDevice)==hipSuccess;
            offset+=chunk;
        }
        free(staging);
    }
    if (hipSetDevice(saved)!=hipSuccess) ok=0;
    if (!ok) (void)hipGetLastError();
    return ok;
}

int openmx_gpu_read_view(OpenMXGpuReadView *view,const void *source,size_t bytes)
{
    int current=-1,origin=-1;
    view->pointer=NULL;view->allocated=0;
    if (!bytes) return 1;
    if (!source || !device_pointer(source,&origin) || hipGetDevice(&current)!=hipSuccess) return 0;
    if (gpu_peer_access(current,origin)) { view->pointer=(void*)source;return 1; }
    if (hipMalloc(&view->pointer,bytes)!=hipSuccess) {
        (void)hipGetLastError();view->pointer=NULL;return 0;
    }
    view->allocated=1;
    return openmx_gpu_copy_device(view->pointer,source,bytes);
}

void openmx_gpu_read_view_release(OpenMXGpuReadView *view)
{
    if (view->allocated && view->pointer) (void)hipFree(view->pointer);
    view->pointer=NULL;view->allocated=0;
}

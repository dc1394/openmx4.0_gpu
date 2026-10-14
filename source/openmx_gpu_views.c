#include "openmx_gpu_views.h"
#include <cuda_runtime_api.h>
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
        if (cudaDeviceCanAccessPeer(&possible,target,source)==cudaSuccess && possible) {
            cudaError_t status=cudaDeviceEnablePeerAccess(source,0);
            ok=status==cudaSuccess || status==cudaErrorPeerAccessAlreadyEnabled;
            if (status!=cudaSuccess) (void)cudaGetLastError();
        } else (void)cudaGetLastError();
    }
    return ok;
}

static int device_pointer(const void *pointer,int *device)
{
    struct cudaPointerAttributes attributes;
    if (cudaPointerGetAttributes(&attributes,pointer)!=cudaSuccess) {
        (void)cudaGetLastError();return 0;
    }
    if (attributes.type!=cudaMemoryTypeDevice) return 0;
    *device=attributes.device;
    return 1;
}

int openmx_gpu_copy_device(void *destination,const void *source,size_t bytes)
{
    int target=-1,origin=-1,saved=-1,ok=0;
    if (!bytes) return 1;
    if (!destination || !source || !device_pointer(destination,&target) ||
        !device_pointer(source,&origin) || cudaGetDevice(&saved)!=cudaSuccess) return 0;
    if (target==origin) return cudaMemcpy(destination,source,bytes,cudaMemcpyDeviceToDevice)==cudaSuccess;
    if (cudaSetDevice(target)!=cudaSuccess) return 0;
    if (gpu_peer_access(target,origin)) {
        ok=cudaMemcpyPeer(destination,target,source,origin,bytes)==cudaSuccess;
    } else {
        const size_t capacity=bytes<(size_t)16*1024*1024 ? bytes : (size_t)16*1024*1024;
        void *staging=malloc(capacity);
        ok=staging!=NULL;
        for (size_t offset=0;ok && offset<bytes;) {
            const size_t chunk=bytes-offset<capacity ? bytes-offset : capacity;
            ok=cudaSetDevice(origin)==cudaSuccess &&
               cudaMemcpy(staging,(const char*)source+offset,chunk,cudaMemcpyDeviceToHost)==cudaSuccess &&
               cudaSetDevice(target)==cudaSuccess &&
               cudaMemcpy((char*)destination+offset,staging,chunk,cudaMemcpyHostToDevice)==cudaSuccess;
            offset+=chunk;
        }
        free(staging);
    }
    if (cudaSetDevice(saved)!=cudaSuccess) ok=0;
    if (!ok) (void)cudaGetLastError();
    return ok;
}

int openmx_gpu_read_view(OpenMXGpuReadView *view,const void *source,size_t bytes)
{
    int current=-1,origin=-1;
    view->pointer=NULL;view->allocated=0;
    if (!bytes) return 1;
    if (!source || !device_pointer(source,&origin) || cudaGetDevice(&current)!=cudaSuccess) return 0;
    if (gpu_peer_access(current,origin)) { view->pointer=(void*)source;return 1; }
    if (cudaMalloc(&view->pointer,bytes)!=cudaSuccess) {
        (void)cudaGetLastError();view->pointer=NULL;return 0;
    }
    view->allocated=1;
    return openmx_gpu_copy_device(view->pointer,source,bytes);
}

void openmx_gpu_read_view_release(OpenMXGpuReadView *view)
{
    if (view->allocated && view->pointer) (void)cudaFree(view->pointer);
    view->pointer=NULL;view->allocated=0;
}

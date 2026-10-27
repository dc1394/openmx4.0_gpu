#ifndef OPENMX_GRID_DEVICE_SCOPE_H
#define OPENMX_GRID_DEVICE_SCOPE_H
#include "openmx_gpu_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#if OPENMX_GPU_HIP
#include <omp.h>
#else
#include <openacc.h>
#endif

/* Primary-device affinity of retained grid data. A calculation keeps its
   rank-to-device mapping until clear/finish; workers use separate transient
   buffers. Empty host states never query or initialize a device. */
typedef struct {
    int valid, native_device, offload_device, offload_type;
} OpenMXGridDeviceAffinity;
typedef struct {
    int active, native_device, offload_device, offload_type;
} OpenMXGridDeviceScope;

static inline void OpenMX_GridDeviceError(const char *operation)
{
    fprintf(stderr, "OpenMX grid cache: %s. Clear the calculation before changing its primary GPU.\n", operation);
    abort();
}

static inline void OpenMX_GridDeviceCapture(OpenMXGridDeviceAffinity *affinity)
{
    int device, offload, type = 0;
    if (OpenMXGpuGetDevice(&device) != OPENMX_GPU_SUCCESS)
        OpenMX_GridDeviceError("cannot query the allocation device");
#if OPENMX_GPU_HIP
    offload = omp_get_default_device();
#else
    type = (int)acc_get_device_type();
    offload = acc_get_device_num((acc_device_t)type);
#endif
    if (affinity->valid && (affinity->native_device != device ||
        affinity->offload_device != offload || affinity->offload_type != type))
        OpenMX_GridDeviceError("retained data was requested on another device");
    affinity->native_device = device;
    affinity->offload_device = offload;
    affinity->offload_type = type;
    affinity->valid = 1;
}

static inline OpenMXGridDeviceScope OpenMX_GridDeviceEnter(const OpenMXGridDeviceAffinity *affinity)
{
    OpenMXGridDeviceScope previous = {0};
    if (!affinity->valid) return previous;
    if (OpenMXGpuGetDevice(&previous.native_device) != OPENMX_GPU_SUCCESS)
        OpenMX_GridDeviceError("cannot query the calling device for release");
#if OPENMX_GPU_HIP
    previous.offload_device = omp_get_default_device();
    omp_set_default_device(affinity->offload_device);
#else
    previous.offload_type = (int)acc_get_device_type();
    previous.offload_device = acc_get_device_num((acc_device_t)previous.offload_type);
    acc_set_device_num(affinity->offload_device, (acc_device_t)affinity->offload_type);
#endif
    if (OpenMXGpuSetDevice(affinity->native_device) != OPENMX_GPU_SUCCESS)
        OpenMX_GridDeviceError("cannot bind the allocation device for release");
    previous.active = 1;
    return previous;
}

static inline void OpenMX_GridDeviceLeave(OpenMXGridDeviceScope previous)
{
    if (!previous.active) return;
#if OPENMX_GPU_HIP
    omp_set_default_device(previous.offload_device);
#else
    acc_set_device_num(previous.offload_device, (acc_device_t)previous.offload_type);
#endif
    if (OpenMXGpuSetDevice(previous.native_device) != OPENMX_GPU_SUCCESS)
        OpenMX_GridDeviceError("cannot restore the calling device after release");
}
#endif

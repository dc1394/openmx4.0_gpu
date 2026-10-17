/* OpenMX extension for the pinned MAGMA 2.10.0 HIP interface.
   This file is copied beside interface_hip/interface.cpp by apply_magma_device_map.py.
   It is included exactly once, after HIP/MAGMA type definitions.

   Only OpenMX's one-stage host adapter uses this map. Its process mutex
   covers setup, every MAGMA call, GPU synchronization, and map removal.
   The map must be process-wide: MAGMA can create OpenMP worker threads.
   Unmapped mode preserves upstream behavior. No HIP visibility, ordinal,
   property cache, or application-side HIP API is changed. */
#ifndef OPENMX_MAGMA_DEVICE_MAP_H
#define OPENMX_MAGMA_DEVICE_MAP_H

static int openmx_magma_map_count = 0;
static int openmx_magma_map_devices[8];

/* Versioned capability symbol also permits runtime detection when OpenMX
   links against a normal external (unpatched) MAGMA installation. Clearing
   never calls HIP and cannot fail, including exception cleanup paths. */
extern "C" int openmx_magma_device_map_v1(int count, const int *devices)
{
    if (count == 0) { openmx_magma_map_count = 0; return 0; }
    if (!devices || count < 2 || count > 8 || openmx_magma_map_count) return -1;
    int visible = 0;
    if (hipGetDeviceCount(&visible) != hipSuccess) return -1;
    for (int i = 0; i < count; ++i) {
        if (devices[i] < 0 || devices[i] >= visible) return -1;
        for (int j = 0; j < i; ++j) if (devices[i] == devices[j]) return -1;
    }
    for (int i = 0; i < count; ++i) openmx_magma_map_devices[i] = devices[i];
    openmx_magma_map_count = count;
    return 0;
}

static int openmx_magma_map_to_physical(int logical)
{
    if (!openmx_magma_map_count) return logical;
    return logical >= 0 && logical < openmx_magma_map_count
        ? openmx_magma_map_devices[logical] : -1;
}

static int openmx_magma_map_to_logical(int physical)
{
    if (!openmx_magma_map_count) return physical;
    for (int i = 0; i < openmx_magma_map_count; ++i)
        if (openmx_magma_map_devices[i] == physical) return i;
    /* The adapter enters on its first selected device. Never reinterpret an
       unrelated physical ordinal as an in-group logical ordinal. */
    return -1;
}

#endif

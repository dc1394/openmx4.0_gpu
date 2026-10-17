#ifndef OPENMX_GPU_ARCH_POLICY_H
#define OPENMX_GPU_ARCH_POLICY_H

/* Keep the measured MI300A policies separate from discrete gfx942 (MI300X)
   and from the consumer APUs and Radeon GPUs.  The suffix contains target
   features, for example gfx942:sramecc+:xnack+. */
#include <string.h>

static inline int openmx_gpu_is_mi300a_arch(const char *arch, int integrated)
{
    return integrated && arch != NULL && strncmp(arch, "gfx942", 6) == 0 &&
           (arch[6] == '\0' || arch[6] == ':');
}

#ifdef __cplusplus
#include <atomic>
#include <cstdlib>
#include "hip_runtime_compat.h"

/* Call only when choosing a GPU path, never while parsing an input.  A CPU
   run must not start the HIP runtime just to establish unused defaults.
   OpenMX selects a fixed device per rank before its GPU work; cache a
   successful query, but let a transient failure be retried.  Krylov can
   reach this helper concurrently from OpenMP workers.  Only the scalar
   choice is shared, so relaxed atomic access is sufficient; concurrent
   first calls may independently query the same fixed device. */
static inline int openmx_gpu_device_is_mi300a(void)
{
    static std::atomic<int> cached{-1};
    const char *enabled = std::getenv("OPENMX_GPU");
    int choice;

    if (enabled != NULL && enabled[0] != '\0' && std::atoi(enabled) == 0) return 0;
    choice = cached.load(std::memory_order_relaxed);
    if (choice < 0) {
        hipDeviceProp_t prop;
        int device = 0;

        if (hipGetDevice(&device) != hipSuccess || hipGetDeviceProperties(&prop, device) != hipSuccess) {
            (void)hipGetLastError();
            return 0;
        }
        choice = openmx_gpu_is_mi300a_arch(prop.gcnArchName, prop.integrated);
        cached.store(choice, std::memory_order_relaxed);
    }
    return choice;
}
#endif

#endif

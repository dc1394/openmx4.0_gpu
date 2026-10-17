#ifndef OPENMX_TARGET_H
#define OPENMX_TARGET_H

/* Explicit HIP allocations associated with the calling OpenMP device.
   Allocation failure is returned to capacity planners, before publication. */
#include "hip_runtime_compat.h"
#include <omp.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

static inline void *openmx_target_alloc(size_t bytes)
{
    void *p = NULL;
    if (hipMalloc(&p, bytes ? bytes : 1) != hipSuccess) {
        (void)hipGetLastError();
        return NULL;
    }
    return p;
}

/* Allocation admission is recoverable. An unchecked transfer/launch fault
   must never let a legacy path publish incomplete numerical results. */
static inline void openmx_target_check(hipError_t status, const char *operation)
{
    if (status == hipSuccess) return;
    fprintf(stderr, "OpenMX: HIP %s failed: %s\n", operation, hipGetErrorString(status));
    abort();
}
static inline void openmx_target_free(void *p) { if (p) (void)hipFree(p); }
static inline void openmx_target_wait(void) { openmx_target_check(hipDeviceSynchronize(), "synchronization"); }
static inline void *openmx_target_deviceptr(const void *p)
{ return omp_get_mapped_ptr(p, omp_get_default_device()); }
static inline int openmx_target_present(const void *p, size_t bytes)
{ (void)bytes; return omp_target_is_present(p, omp_get_default_device()); }
static inline void openmx_target_map(void *host, void *device, size_t bytes)
{ if (omp_target_associate_ptr(host, device, bytes, 0, omp_get_default_device()) != 0) {
    fprintf(stderr, "OpenMX: OpenMP device pointer association failed.\n"); abort();
} }
static inline void openmx_target_unmap(void *host)
{ (void)omp_target_disassociate_ptr(host, omp_get_default_device()); }
static inline int openmx_target_upload(void *device, const void *host, size_t bytes)
{ if (bytes) openmx_target_check(hipMemcpy(device, host, bytes, hipMemcpyHostToDevice), "upload"); return 1; }
static inline int openmx_target_download(void *host, const void *device, size_t bytes)
{ if (bytes) openmx_target_check(hipMemcpy(host, device, bytes, hipMemcpyDeviceToHost), "download"); return 1; }
static inline void openmx_target_update_device(void *host, size_t bytes)
{ openmx_target_upload(openmx_target_deviceptr(host), host, bytes); }
static inline void openmx_target_update_self(void *host, size_t bytes)
{ openmx_target_download(host, openmx_target_deviceptr(host), bytes); }
static inline void openmx_target_delete(void *host, size_t bytes)
{
    (void)bytes;
    void *device = openmx_target_deviceptr(host);
    if (device) { openmx_target_unmap(host); openmx_target_free(device); }
}
/* HIP frees immediately; there is no OpenACC allocation freelist. */
static inline void openmx_target_clear_freelists(void) {}

#endif

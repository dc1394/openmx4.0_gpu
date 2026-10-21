#ifndef OPENMX_CORE_NATIVE_RESOURCES_H
#define OPENMX_CORE_NATIVE_RESOURCES_H
#include "openmx_core.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef struct OpenMXNativeResources OpenMXNativeResources;
typedef int (*OpenMXNativeResourceRelease)(void *value, int device);
typedef void (*OpenMXNativeResourceReset)(void);

/* The explicit calculation owns each named group. reset runs before resource
   destruction, so native caches drain streams, unmap and invalidate borrowed
   aliases in their original order. Finish/destroy before MPI_Finalize. */
int openmx_core_native_resources(OpenMXCoreContext*, const char *name,
    OpenMXNativeResourceReset reset, OpenMXNativeResources **out, char*, size_t);
void openmx_core_native_resources_clear(OpenMXCoreContext*);

/* Ownership transfers only on success. Separate domains distinguish a device
   address from a host/handle address with the same numerical value. Operations
   on one group may run concurrently; reset/clear require quiescent users. */
int openmx_core_native_adopt(OpenMXNativeResources*, int domain, void *value,
    int device, OpenMXNativeResourceRelease release,
    OpenMXNativeResourceRelease synchronize, char*, size_t);
/* Return 1 for an owned object (including an SDK error), 0 for a foreign object.
   A release attempt consumes ownership; SDK status is returned in status. */
int openmx_core_native_release(OpenMXNativeResources*, int domain, void *value,
    int *status);
void *openmx_core_native_malloc(OpenMXNativeResources*, size_t count, size_t bytes,
    int zero);
void *openmx_core_native_realloc(OpenMXNativeResources*, void*, size_t bytes);
void openmx_core_native_free(OpenMXNativeResources*, void*);
/* Zero-initialized POD metadata; storage has max_align_t alignment and stays
   stable until group destruction. A repeated name must keep its byte size. */
int openmx_core_native_state(OpenMXNativeResources*, const char *name, size_t bytes,
    void **out, char*, size_t);
size_t openmx_core_native_resource_count(OpenMXNativeResources*);

#ifdef __cplusplus
}
#endif
#endif

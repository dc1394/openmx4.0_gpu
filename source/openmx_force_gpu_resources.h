#ifndef OPENMX_FORCE_GPU_RESOURCES_H
#define OPENMX_FORCE_GPU_RESOURCES_H
/* GPU Force storage belongs to the enclosing solver invocation. The native
   context pointers are borrowed; registration preserves optional-allocation
   fallbacks and the original allocation/free APIs of each backend. */
#include "openmx_native_resources.h"

typedef struct {
    int enabled;
    float *flat_host,*halo_host,*c2_host;
    int num_proj;

    /* local DS_VNA rows, direction-major: flat[dir*stride + block]    */
    float* flat_dev;
    size_t flat_stride;
    size_t* mck_off;   /* (Mc,k) block base: mck_off[mck_base[Mc]+k]   */
    int* mck_base;     /* Mc -> first (Mc,k) slot                      */

    /* case-1 halo archive: direction 0 rows of every received atom    */
    float* halo_dev;
    size_t halo_count;
    size_t* halo_off;  /* (halo_idx, k) block base                     */
    int* halo_base;    /* halo_idx -> first slot                       */

    /* case-2 per-centre archive: 4 directions of the staged slot;
       block sizes follow the NEIGHBOUR's orbital count                 */
    float* c2_dev;
    size_t c2_stride;
    size_t* c2_off;    /* (Mc,k) block base, indexed via mck_base      */

    /* host bounce buffer holding one atom's largest one-direction block */
    float* stage;
    size_t stage_len;
    OpenMXNativeResources *resources;
} Force4BGpuContext;

typedef struct {
    size_t cdm_off;
    int k_off;
    int k_n;
    int ian;
    int jan;
} Force4BGpuItem;

typedef struct {
    int nitems;
    size_t flat_stride, halo_count, rows, cdm_count, bytes;
    size_t flat, halo, items, row_a, row_b, row_halo, cdm, item_f;
} Force4BCase1Plan;

typedef struct {
    char *host, *device;
    size_t capacity, pair_count;
    int num_proj, host_pinned;
    int *source_head, *pair_next, *pair_mc, *pair_q, *pair_rows;
    double* forces;
    unsigned char* done;
    OpenMXNativeResources *resources;
} Force4BCase1Stream;

typedef struct {
    Force4BCase1Plan plan;
    MPI_Comm device_comm;
    int enabled, num_proj, device_rank, device_ranks, stage_pinned;
    size_t host_bytes, stage_count;
    size_t *local_off, *halo_off;
    Type_DS_VNA* halo_host;
    float* stage;
    char* metadata;
    unsigned char* seen;
    OpenMXNativeResources *resources;
} Force4BCase1Turns;

typedef struct {
    size_t stride, bytes, cdm_count;
    size_t c2, items, hoff, qoff, cdm, item_f;
    int nitems;
} Force4BCase2Plan;

typedef struct {
    char* host;
    char* device;
    size_t capacity;
    size_t* row_off;
    int num_proj, host_pinned;
    int pending_items;
    size_t pending_f;
    OpenMXNativeResources *resources;
} Force4BCase2Stream;

static void *ForceGpu_Metadata(OpenMXNativeResources *resources,const char *name,size_t bytes)
{
    void *state=NULL;char error[256];
    if(!openmx_core_native_state(resources,name,bytes,&state,error,sizeof(error)))
        OpenMX_Solver_Failure(error);
    return state;
}

static void *ForceGpu_State(OpenMXSolverInvocation *scratch,const char *name,
    size_t bytes,OpenMXNativeResources **resources)
{
    *resources=OpenMX_Solver_Resources(scratch,name);
    return ForceGpu_Metadata(*resources,"state",bytes);
}

static void *ForceGpu_Host(OpenMXNativeResources *resources,size_t count,size_t bytes,int zero)
{
    return openmx_core_native_malloc(resources,count,bytes,zero);
}

static void *ForceGpu_RequiredHost(OpenMXNativeResources *resources,size_t bytes)
{
    void *values=ForceGpu_Host(resources,1,bytes?bytes:1,0);
    if(!values)OpenMX_Solver_Failure("GPU Force host allocation failed");
    return values;
}

static int ForceGpu_DeviceSync(void *pointer,int device)
{
    int previous,status=OpenMX_Resource_BindDevice(device,&previous);(void)pointer;
    if(status)return status;
    status=(int)OpenMXGpuDeviceSynchronize();
    OpenMX_Resource_RestoreDevice(device,previous);return status;
}

static void *ForceGpu_Arena(OpenMXNativeResources *resources,size_t bytes)
{
    void *pointer=Force_gpu_arena_try(bytes);
    if(!pointer)return NULL;
#if OPENMX_GPU_HIP
    if(!OpenMX_Resource_Adopt(resources,OPENMX_RESOURCE_DEVICE,pointer,
        OpenMX_Resource_DeviceFree,ForceGpu_DeviceSync)) {
        (void)OpenMXGpuFree(pointer);return NULL;
    }
#else
    if(!OpenMX_Resource_Adopt(resources,OPENMX_RESOURCE_ACC,pointer,
        OpenMX_Resource_AccFreeCallback,ForceGpu_DeviceSync)) {
        acc_free(pointer);return NULL;
    }
#endif
    return pointer;
}

static void ForceGpu_ArenaFree(OpenMXNativeResources *resources,void *pointer)
{
#if OPENMX_GPU_HIP
    (void)OpenMX_Resource_GpuFree(resources,pointer);
#else
    OpenMX_Resource_AccFree(resources,pointer);
#endif
}

/* HIP keeps page-locked staging for its many small PCIe transfers. A failed
   SDK allocation or registration still permits the zeroed pageable fallback. */
static void *ForceGpu_PinnedHost(OpenMXNativeResources *resources,size_t bytes,int *pinned)
{
    *pinned=0;
    if(!bytes)bytes=1;
#if OPENMX_GPU_HIP
    void *pointer=NULL;
    if(OpenMXGpuHostAlloc(&pointer,bytes,0)==OPENMX_GPU_SUCCESS && pointer) {
        if(OpenMX_Resource_Adopt(resources,OPENMX_RESOURCE_PINNED,pointer,
            OpenMX_Resource_PinnedFree,ForceGpu_DeviceSync)) {
            *pinned=1;return pointer;
        }
        (void)OpenMXGpuFreeHost(pointer);
    }
    (void)OpenMXGpuGetLastError();
#endif
    return ForceGpu_Host(resources,1,bytes,1);
}

static void ForceGpu_PinnedHostFree(OpenMXNativeResources *resources,void *pointer,int pinned)
{
    if(!pointer)return;
    if(pinned)(void)OpenMX_Resource_HostFree(resources,pointer);
    else openmx_core_native_free(resources,pointer);
}

/* MPI handles need stable storage until the C++ invocation is destroyed;
   the field below lives in native_state, never on a returned C stack frame. */
enum { OPENMX_FORCE_RESOURCE_MPI=7 };
static int ForceGpu_CommRelease(void *pointer,int device)
{
    MPI_Comm *comm=(MPI_Comm*)pointer;(void)device;
    return *comm==MPI_COMM_NULL?0:MPI_Comm_free(comm);
}

static void ForceGpu_CommAdopt(OpenMXNativeResources *resources,MPI_Comm *comm)
{
    char error[256];
    if(*comm==MPI_COMM_NULL)return;
    if(!openmx_core_native_adopt(resources,OPENMX_FORCE_RESOURCE_MPI,comm,-1,
        ForceGpu_CommRelease,NULL,error,sizeof(error))) {
        (void)MPI_Comm_free(comm);OpenMX_Solver_Failure(error);
    }
}

static void ForceGpu_CommFree(OpenMXNativeResources *resources,MPI_Comm *comm)
{
    int status;
    if(*comm!=MPI_COMM_NULL && !openmx_core_native_release(resources,
        OPENMX_FORCE_RESOURCE_MPI,comm,&status)) (void)MPI_Comm_free(comm);
}

static MPI_Comm *ForceGpu_NodeComm(OpenMXNativeResources *resources)
{
    MPI_Comm *comm=(MPI_Comm*)ForceGpu_Metadata(resources,"node-comm",sizeof(MPI_Comm));
    *comm=MPI_COMM_NULL;
    return comm;
}


static void Force4B_GpuEnd(Force4BGpuContext *g)
{
    if(!g)return;
    OpenMXNativeResources *resources=g->resources;

    const int enabled=g->enabled;

    if (g->flat_dev != NULL) ForceGpu_ArenaFree(g->resources,g->flat_dev);
    if (g->halo_dev != NULL) ForceGpu_ArenaFree(g->resources,g->halo_dev);
    if (g->c2_dev != NULL) ForceGpu_ArenaFree(g->resources,g->c2_dev);
#if !OPENMX_GPU_HIP
    if(enabled)acc_clear_freelists();
#else
    (void)enabled;
#endif

    openmx_core_native_free(g->resources,g->flat_host);openmx_core_native_free(g->resources,g->halo_host);openmx_core_native_free(g->resources,g->c2_host);
    openmx_core_native_free(g->resources,g->stage);
    openmx_core_native_free(g->resources,g->c2_off);
    openmx_core_native_free(g->resources,g->halo_off);
    openmx_core_native_free(g->resources,g->halo_base);
    openmx_core_native_free(g->resources,g->mck_off);
    openmx_core_native_free(g->resources,g->mck_base);
    memset(g, 0, sizeof(*g));
    g->resources=resources;
}

static void Force4B_GpuCase1StreamEnd(Force4BCase1Stream *g)
{
    if(!g)return;
    OpenMXNativeResources *resources=g->resources;

    if (g->device != NULL) {
        ForceGpu_ArenaFree(g->resources,g->device);
        Force_gpu_pool_flush();
    }
    openmx_core_native_free(g->resources,g->done);
    openmx_core_native_free(g->resources,g->forces);
    openmx_core_native_free(g->resources,g->pair_rows);
    openmx_core_native_free(g->resources,g->pair_q);
    openmx_core_native_free(g->resources,g->pair_mc);
    openmx_core_native_free(g->resources,g->pair_next);
    openmx_core_native_free(g->resources,g->source_head);
    ForceGpu_PinnedHostFree(g->resources,g->host,g->host_pinned);
    memset(g, 0, sizeof(*g));
    g->resources=resources;
}

static void Force4B_GpuCase1TurnsEnd(Force4BCase1Turns *g)
{
    if(!g)return;
    OpenMXNativeResources *resources=g->resources;
    openmx_core_native_free(g->resources,g->seen);
    openmx_core_native_free(g->resources,g->metadata);
    ForceGpu_PinnedHostFree(g->resources,g->stage,g->stage_pinned);
    openmx_core_native_free(g->resources,g->halo_host);
    openmx_core_native_free(g->resources,g->halo_off);
    openmx_core_native_free(g->resources,g->local_off);
    if (g->device_comm != MPI_COMM_NULL) ForceGpu_CommFree(g->resources,&g->device_comm);
    memset(g, 0, sizeof(*g));
    g->resources=resources;
    g->device_comm = MPI_COMM_NULL;
}

static void Force4B_GpuCase2StreamEnd(Force4BCase2Stream *g)
{
    if(!g)return;
    OpenMXNativeResources *resources=g->resources;
    if (g->device != NULL) {
        ForceGpu_ArenaFree(g->resources,g->device);
        Force_gpu_pool_flush();
    }
    openmx_core_native_free(g->resources,g->row_off);
    ForceGpu_PinnedHostFree(g->resources,g->host,g->host_pinned);
    memset(g, 0, sizeof(*g));
    g->resources=resources;
}

#endif

#ifndef OPENMX_FORCE_GPU_RESOURCES_MODEL_H
#define OPENMX_FORCE_GPU_RESOURCES_MODEL_H
#include <stddef.h>
#define OPENMX_GPU_HIP FORCE_FIXTURE_HIP
#include "openmx_env.h"
typedef struct OpenMXNativeResources OpenMXNativeResources;
typedef struct OpenMXSolverInvocation OpenMXSolverInvocation;

/* Only allocation/copy models belong in these arithmetic fixtures. The real
   ownership implementation is linked by force-vna-gpu-resources instead. */
static void *ForceGpu_State(OpenMXSolverInvocation *scope,const char *name,
    size_t bytes,OpenMXNativeResources **resources)
{
    static struct { max_align_t alignment; unsigned char data[2048]; MPI_Comm node; } states[4];
    int index=!strcmp(name,"force4b-archives")?0:!strcmp(name,"force4b-case1-stream")?1:
        !strcmp(name,"force4b-case1-turns")?2:3;
    (void)scope;CHECK(bytes<=sizeof(states[index].data));
    *resources=(OpenMXNativeResources*)&states[index];
    return states[index].data;
}
static void *(*force_fixture_calloc)(size_t,size_t)=test_calloc;
static void *ForceGpu_Host(OpenMXNativeResources *owner,size_t count,size_t bytes,int zero)
{ (void)owner;return zero?force_fixture_calloc(count,bytes):(host_allocation_fail?NULL:malloc(count*bytes)); }
static void *ForceGpu_RequiredHost(OpenMXNativeResources *owner,size_t bytes)
{ void *p=ForceGpu_Host(owner,1,bytes?bytes:1,0);CHECK(p);return p; }
static void openmx_core_native_free(OpenMXNativeResources *owner,void *p)
{ (void)owner;free(p); }
static void *ForceGpu_Arena(OpenMXNativeResources *owner,size_t bytes)
{ (void)owner;return Force_gpu_arena_try(bytes); }
static void ForceGpu_ArenaFree(OpenMXNativeResources *owner,void *p)
{
    (void)owner;
#if FORCE_FIXTURE_HIP
    (void)hipFree(p);
#else
    acc_free(p);
#endif
}
static void *ForceGpu_PinnedHost(OpenMXNativeResources *owner,size_t bytes,int *pinned)
{
    *pinned=0;
#if FORCE_FIXTURE_HIP
    void *p=NULL;
    if(hipHostMalloc(&p,bytes,0)==hipSuccess && p) { *pinned=1;return p; }
    (void)hipGetLastError();
#endif
    return ForceGpu_Host(owner,1,bytes,1);
}
static void ForceGpu_PinnedHostFree(OpenMXNativeResources *owner,void *p,int pinned)
{
#if FORCE_FIXTURE_HIP
    if(pinned) { (void)hipHostFree(p);return; }
#else
    (void)pinned;
#endif
    openmx_core_native_free(owner,p);
}
static void ForceGpu_CommAdopt(OpenMXNativeResources *owner,MPI_Comm *comm)
{ (void)owner;(void)comm; }
static void ForceGpu_CommFree(OpenMXNativeResources *owner,MPI_Comm *comm)
{ (void)owner;if(*comm!=MPI_COMM_NULL)(void)MPI_Comm_free(comm); }
static MPI_Comm *ForceGpu_NodeComm(OpenMXNativeResources *owner)
{
    /* A fresh node handle is sufficient for the single-rank MPI model. */
    static MPI_Comm node;(void)owner;node=MPI_COMM_NULL;return &node;
}
#if !FORCE_FIXTURE_HIP
static int Force_gpu_stage_default(void)
{ return openmx_env_flag_at(OPENMX_ENV_FORCE_GPU,1); }
#endif
#endif

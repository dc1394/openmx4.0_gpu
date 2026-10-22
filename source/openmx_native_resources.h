#ifndef OPENMX_NATIVE_RESOURCES_H
#define OPENMX_NATIVE_RESOURCES_H
#include "openmx_core_native_resources.h"
#include "openmx_gpu_runtime.h"
#include "openmx_gpu_blas.h"
#include "openmx_gpu_solver.h"
#include <stdlib.h>
#include <stdio.h>
#if !OPENMX_GPU_HIP
#include <openacc.h>
#endif

/* These adapters preserve native status codes and optional-allocation fallbacks.
   GPU objects remember their allocation device; freeing on an OpenMP/main
   thread temporarily binds that device and restores the previous one. */
enum { OPENMX_RESOURCE_DEVICE=1, OPENMX_RESOURCE_PINNED=2,
       OPENMX_RESOURCE_ACC=3, OPENMX_RESOURCE_SOLVER=4,
       OPENMX_RESOURCE_BLAS=5, OPENMX_RESOURCE_STREAM=6 };
static inline int OpenMX_Resource_BindDevice(int device,int *previous)
{
  OpenMXGpuError status=OpenMXGpuGetDevice(previous);
  if(status!=OPENMX_GPU_SUCCESS)return (int)status;
  return *previous==device?0:(int)OpenMXGpuSetDevice(device);
}
static inline void OpenMX_Resource_RestoreDevice(int device,int previous)
{ if(device!=previous)(void)OpenMXGpuSetDevice(previous); }
static inline int OpenMX_Resource_DeviceFree(void *pointer,int device)
{
  int previous,status=OpenMX_Resource_BindDevice(device,&previous);if(status)return status;
  status=(int)OpenMXGpuFree(pointer);OpenMX_Resource_RestoreDevice(device,previous);return status;
}
static inline int OpenMX_Resource_PinnedFree(void *pointer,int device)
{
  int previous,status=OpenMX_Resource_BindDevice(device,&previous);if(status)return status;
  status=(int)OpenMXGpuFreeHost(pointer);OpenMX_Resource_RestoreDevice(device,previous);return status;
}
static inline int OpenMX_Resource_StreamSync(void *pointer,int device)
{
  int previous,status=OpenMX_Resource_BindDevice(device,&previous);if(status)return status;
  status=(int)OpenMXGpuStreamSynchronize((OpenMXGpuStream)pointer);
  OpenMX_Resource_RestoreDevice(device,previous);return status;
}
static inline int OpenMX_Resource_StreamFree(void *pointer,int device)
{
  int previous,status=OpenMX_Resource_BindDevice(device,&previous);if(status)return status;
  (void)OpenMXGpuStreamSynchronize((OpenMXGpuStream)pointer);
  status=(int)OpenMXGpuStreamDestroy((OpenMXGpuStream)pointer);
  OpenMX_Resource_RestoreDevice(device,previous);return status;
}
static inline int OpenMX_Resource_BlasFree(void *pointer,int device)
{
  int previous,status=OpenMX_Resource_BindDevice(device,&previous);
  if(status)return OPENMX_BLAS_EXECUTION_FAILED;
  status=(int)OpenMXBlasDestroy((OpenMXBlasHandle)pointer);
  OpenMX_Resource_RestoreDevice(device,previous);return status;
}
static inline int OpenMX_Resource_SolverFree(void *pointer,int device)
{
  int previous,status=OpenMX_Resource_BindDevice(device,&previous);
  if(status)return OPENMX_SOLVER_EXECUTION_FAILED;
  status=(int)OpenMXSolverDestroy((OpenMXSolverHandle)pointer);
  OpenMX_Resource_RestoreDevice(device,previous);return status;
}
static inline int OpenMX_Resource_Adopt(OpenMXNativeResources *owner,int domain,void *pointer,
    OpenMXNativeResourceRelease release,OpenMXNativeResourceRelease synchronize)
{
  int device;char error[256];
  if(!owner || !pointer)return 1;
  if(OpenMXGpuGetDevice(&device)!=OPENMX_GPU_SUCCESS)return 0;
  return openmx_core_native_adopt(owner,domain,pointer,device,release,synchronize,error,sizeof(error));
}
static inline OpenMXGpuError OpenMX_Resource_GpuMalloc(OpenMXNativeResources *owner,void **out,size_t bytes)
{
  void *pointer=NULL;OpenMXGpuError status=OpenMXGpuMalloc(&pointer,bytes);
  if(status!=OPENMX_GPU_SUCCESS) { *out=NULL;return status; }
  if(!OpenMX_Resource_Adopt(owner,OPENMX_RESOURCE_DEVICE,pointer,OpenMX_Resource_DeviceFree,NULL)) {
    (void)OpenMXGpuFree(pointer);*out=NULL;return OPENMX_GPU_OUT_OF_MEMORY;
  }
  *out=pointer;return status;
}
static inline OpenMXGpuError OpenMX_Resource_GpuFree(OpenMXNativeResources *owner,void *pointer)
{
  int status;
  if(openmx_core_native_release(owner,OPENMX_RESOURCE_DEVICE,pointer,&status))return (OpenMXGpuError)status;
  return OpenMXGpuFree(pointer);
}
static inline OpenMXGpuError OpenMX_Resource_HostAlloc(OpenMXNativeResources *owner,void **out,size_t bytes,unsigned int flags)
{
  void *pointer=NULL;OpenMXGpuError status=OpenMXGpuHostAlloc(&pointer,bytes,flags);
  if(status!=OPENMX_GPU_SUCCESS) { *out=NULL;return status; }
  if(!OpenMX_Resource_Adopt(owner,OPENMX_RESOURCE_PINNED,pointer,OpenMX_Resource_PinnedFree,NULL)) {
    (void)OpenMXGpuFreeHost(pointer);*out=NULL;return OPENMX_GPU_OUT_OF_MEMORY;
  }
  *out=pointer;return status;
}
static inline OpenMXGpuError OpenMX_Resource_HostFree(OpenMXNativeResources *owner,void *pointer)
{
  int status;
  if(openmx_core_native_release(owner,OPENMX_RESOURCE_PINNED,pointer,&status))return (OpenMXGpuError)status;
  return OpenMXGpuFreeHost(pointer);
}
static inline OpenMXGpuError OpenMX_Resource_StreamCreate(OpenMXNativeResources *owner,OpenMXGpuStream *out,unsigned int flags)
{
  OpenMXGpuStream stream=NULL;OpenMXGpuError status=OpenMXGpuStreamCreateWithFlags(&stream,flags);
  if(status!=OPENMX_GPU_SUCCESS)return status;
  if(!OpenMX_Resource_Adopt(owner,OPENMX_RESOURCE_STREAM,(void*)stream,OpenMX_Resource_StreamFree,OpenMX_Resource_StreamSync)) {
    (void)OpenMXGpuStreamDestroy(stream);return OPENMX_GPU_OUT_OF_MEMORY;
  }
  *out=stream;return status;
}
static inline OpenMXGpuError OpenMX_Resource_StreamDestroy(OpenMXNativeResources *owner,OpenMXGpuStream stream)
{
  int status;
  if(openmx_core_native_release(owner,OPENMX_RESOURCE_STREAM,(void*)stream,&status))return (OpenMXGpuError)status;
  return OpenMXGpuStreamDestroy(stream);
}
static inline OpenMXBlasStatus OpenMX_Resource_BlasCreate(OpenMXNativeResources *owner,OpenMXBlasHandle *out)
{
  OpenMXBlasHandle handle=NULL;OpenMXBlasStatus status=OpenMXBlasCreate(&handle);
  if(status!=OPENMX_BLAS_SUCCESS)return status;
  if(!OpenMX_Resource_Adopt(owner,OPENMX_RESOURCE_BLAS,(void*)handle,OpenMX_Resource_BlasFree,NULL)) {
    (void)OpenMXBlasDestroy(handle);return OPENMX_BLAS_ALLOC_FAILED;
  }
  *out=handle;return status;
}
static inline OpenMXBlasStatus OpenMX_Resource_BlasDestroy(OpenMXNativeResources *owner,OpenMXBlasHandle handle)
{
  int status;
  if(openmx_core_native_release(owner,OPENMX_RESOURCE_BLAS,(void*)handle,&status))return (OpenMXBlasStatus)status;
  return OpenMXBlasDestroy(handle);
}
static inline OpenMXSolverStatus OpenMX_Resource_SolverCreate(OpenMXNativeResources *owner,OpenMXSolverHandle *out)
{
  OpenMXSolverHandle handle=NULL;OpenMXSolverStatus status=OpenMXSolverCreate(&handle);
  if(status!=OPENMX_SOLVER_SUCCESS)return status;
  if(!OpenMX_Resource_Adopt(owner,OPENMX_RESOURCE_SOLVER,(void*)handle,OpenMX_Resource_SolverFree,NULL)) {
    (void)OpenMXSolverDestroy(handle);return OPENMX_SOLVER_ALLOC_FAILED;
  }
  *out=handle;return status;
}
static inline OpenMXSolverStatus OpenMX_Resource_SolverDestroy(OpenMXNativeResources *owner,OpenMXSolverHandle handle)
{
  int status;
  if(openmx_core_native_release(owner,OPENMX_RESOURCE_SOLVER,(void*)handle,&status))return (OpenMXSolverStatus)status;
  return OpenMXSolverDestroy(handle);
}
#if !OPENMX_GPU_HIP
static inline int OpenMX_Resource_AccFreeCallback(void *pointer,int device)
{
  int previous,status=OpenMX_Resource_BindDevice(device,&previous);if(status)return status;
  int acc_previous=acc_get_device_num(acc_device_nvidia);
  if(acc_previous!=device)acc_set_device_num(device,acc_device_nvidia);
  acc_free(pointer);
  if(acc_previous!=device)acc_set_device_num(acc_previous,acc_device_nvidia);
  OpenMX_Resource_RestoreDevice(device,previous);return 0;
}
static inline void *OpenMX_Resource_AccMalloc(OpenMXNativeResources *owner,size_t bytes)
{
  void *pointer=acc_malloc(bytes);
  if(!OpenMX_Resource_Adopt(owner,OPENMX_RESOURCE_ACC,pointer,OpenMX_Resource_AccFreeCallback,NULL)) {
    acc_free(pointer);return NULL;
  }
  return pointer;
}
static inline void OpenMX_Resource_AccFree(OpenMXNativeResources *owner,void *pointer)
{
  int status;if(!openmx_core_native_release(owner,OPENMX_RESOURCE_ACC,pointer,&status))acc_free(pointer);
}
#endif
static inline void *OpenMX_Resource_State(OpenMXNativeResources *owner,const char *name,size_t bytes)
{
  void *state=NULL;char error[256];
  if(!openmx_core_native_state(owner,name,bytes,&state,error,sizeof(error))) {
    fprintf(stderr,"OpenMX native state: %s\n",error);abort();
  }
  return state;
}

/* Legacy modules below borrow their owners from an explicit core at DFT entry.
   There is no default/global core. Native cache metadata remains in its
   existing modules during this ownership migration; reset invalidates it
   before the calculation releases the group. */
#define OPENMX_NATIVE_RESOURCE_GROUP(module,release_all) \
  static OpenMXNativeResources *module##_resources=NULL; \
  static void module##_resources_reset(void) { release_all(); module##_resources=NULL; } \
  void module##_Bind_Resources(OpenMXCoreContext *core) { \
    char error[256]; \
    if(!openmx_core_native_resources(core,#module,module##_resources_reset, \
        &module##_resources,error,sizeof(error))) { \
      fprintf(stderr,"OpenMX native resources: %s\n",error);abort(); \
    } \
  }

#include "openmx_cluster_col_state.h"
void Cluster_DFT_NonCol_Bind_Resources(OpenMXCoreContext*);
void Band_DFT_Col_Bind_Resources(OpenMXCoreContext*);
void Band_DFT_NonCol_Bind_Resources(OpenMXCoreContext*);
void Divide_Conquer_Bind_Resources(OpenMXCoreContext*);
void Divide_Conquer_LNO_Bind_Resources(OpenMXCoreContext*);
void Krylov_Bind_Resources(OpenMXCoreContext*);
static inline OpenMXClusterColState *OpenMX_Native_Resources_Bind(OpenMXCoreContext *core)
{
  OpenMXClusterColState *col=Cluster_DFT_Col_Bind_Resources(core);
  Cluster_DFT_NonCol_Bind_Resources(core);
  Band_DFT_Col_Bind_Resources(core);Band_DFT_NonCol_Bind_Resources(core);
  Divide_Conquer_Bind_Resources(core);Divide_Conquer_LNO_Bind_Resources(core);Krylov_Bind_Resources(core);
  return col;
}
#endif

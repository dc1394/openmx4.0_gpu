#pragma once
// Keep SDK allocations separate from host scratch fault injection.
#define OpenMXGpuMalloc model_task_gpu_malloc
#include "gpu_views_model.hpp"
#undef OpenMXGpuMalloc
namespace gpu_task_model {
inline thread_local bool sdk_allocation=false;
}
inline int OpenMXGpuMalloc(void **out,size_t bytes){
    gpu_task_model::sdk_allocation=true;
    const int status=model_task_gpu_malloc(out,bytes);
    gpu_task_model::sdk_allocation=false;
    return status;
}

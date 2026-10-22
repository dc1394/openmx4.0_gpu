#pragma once
#include "gpu_views_model.hpp"

// Independent host execution of the unchanged grid-stride kernels. Native
// SDK aliases, device code generation and actual ordering need GPU tests.
#define OPENMX_GPU_COMPLEX_H
#define __device__
#define __global__
struct ModelComplex { double x,y; };
using OpenMXGpuComplex=ModelComplex;
inline ModelComplex OpenMXGpuMakeComplex(double x,double y){return {x,y};}
struct ModelDim {unsigned x;};
inline constexpr ModelDim blockIdx{0},threadIdx{0},blockDim{1},gridDim{1};
inline double atomicAdd(double *pointer,double value){double old=*pointer;*pointer+=value;return old;}
inline int OpenMXGpuPeekAtLastError(){
    using namespace gpu_views_model;
    (void)fail(peek);return trace.pending;
}
inline int OpenMXGpuMemset(void *pointer,int value,size_t bytes){
    using namespace gpu_views_model;if(fail(fill))return MODEL_ERROR;
    std::lock_guard<std::mutex> lock(mutex);auto* a=find(pointer,bytes);
    if(!a || a->device!=trace.current)return error();
    std::memset(pointer,value,bytes);return 0;
}

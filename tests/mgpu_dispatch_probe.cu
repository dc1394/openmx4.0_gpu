/* Hardware check for the opt-in NVLink auto gate and cluster scope.
   Small matrices use a reduced test threshold, keeping this probe cheap. */
#include "openmx_mgpu_runtime.h"
#include "openmx_mgpu_gemm.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CUDA_OK(x) do { auto e=(x); if(e!=cudaSuccess) { \
    std::fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e));return 1; } } while(0)
int main()
{
    constexpr int n=192;
    setenv("OPENMX_MGPU_GEMM_MIN_N","128",1);
    unsetenv("OPENMX_MGPU_GEMM");
    setenv("OPENMX_MGPU_VERBOSE","1",1);
    openmx_mgpu_configure(1,2,1);
    int visible=0;CUDA_OK(cudaGetDeviceCount(&visible));if(visible<2)return 2;
    CUDA_OK(cudaSetDevice(1));
    double *a,*b,*c,one=1,zero=0;
    CUDA_OK(cudaMalloc(&a,n*n*sizeof(double)));CUDA_OK(cudaMalloc(&b,n*n*sizeof(double)));
    CUDA_OK(cudaMalloc(&c,n*n*sizeof(double)));
    std::vector<double> h(n*n,1),out(n*n);
    CUDA_OK(cudaMemcpy(a,h.data(),n*n*sizeof(double),cudaMemcpyHostToDevice));
    for(auto&v:h)v=2;
    CUDA_OK(cudaMemcpy(b,h.data(),n*n*sizeof(double),cudaMemcpyHostToDevice));
    cublasHandle_t handle;if(cublasCreate(&handle)!=CUBLAS_STATUS_SUCCESS)return 1;
    cublasStatus_t status=CUBLAS_STATUS_SUCCESS;
    if(openmx_mgpu_dispatch_dgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,n,n,n,&one,a,n,b,n,&zero,c,n,&status)) {
        std::fprintf(stderr,"FAIL: dispatch outside cluster scope\n");return 1;
    }
    int previous=openmx_mgpu_scope_enter(1);
    if(openmx_mgpu_dispatch_dgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,n,n,n,
        &one,a,n,b,n,&zero,c,n,&status)) {
        std::fprintf(stderr,"FAIL: device-resident GEMM should stay on one GPU by default\n");return 1;
    }
    setenv("OPENMX_MGPU_GEMM","auto",1);
    const int handled=openmx_mgpu_dispatch_dgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,n,n,n,
        &one,a,n,b,n,&zero,c,n,&status);
    if(!handled || status!=CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr,"FAIL: expected direct-NVLink auto dispatch (handled=%d status=%d)\n",handled,int(status));
        return 1;
    }
    openmx_mgpu_gemm_report report{};openmx_mgpuGemmLastReport(&report);
    CUDA_OK(cudaMemcpy(out.data(),c,n*n*sizeof(double),cudaMemcpyDeviceToHost));
    for(auto v:out)if(v!=2*n)return 1;
    if(report.active_gpus!=2)return 1;
    openmx_mgpu_scope_leave(previous);
    if(openmx_mgpu_dispatch_dgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,n,n,n,&one,a,n,b,n,&zero,c,n,&status))return 1;
    cublasDestroy(handle);CUDA_OK(cudaFree(c));CUDA_OK(cudaFree(b));CUDA_OK(cudaFree(a));
    std::puts("PASS: actual NVLink auto admission, numeric GEMM, two GPUs, cluster scope exit");
}

/* Hardware check for the runtime's NVLink auto gate and cluster scope.
   Small matrices use a reduced test threshold, keeping this probe cheap. */
#include "openmx_mgpu_runtime.h"
#include "openmx_mgpu_gemm.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#define HIP_OK(x) do { auto e=(x); if(e!=hipSuccess) { \
    std::fprintf(stderr,"%s: %s\n",#x,hipGetErrorString(e));return 1; } } while(0)
int main()
{
    constexpr int n=192;
    setenv("OPENMX_MGPU_GEMM_MIN_N","128",1);
    unsetenv("OPENMX_MGPU_GEMM");
    setenv("OPENMX_MGPU_VERBOSE","1",1);
    openmx_mgpu_configure(1,2,1);
    int visible=0;HIP_OK(hipGetDeviceCount(&visible));if(visible<2)return 2;
    HIP_OK(hipSetDevice(1));
    double *a,*b,*c,one=1,zero=0;
    HIP_OK(hipMalloc(&a,n*n*sizeof(double)));HIP_OK(hipMalloc(&b,n*n*sizeof(double)));
    HIP_OK(hipMalloc(&c,n*n*sizeof(double)));
    std::vector<double> h(n*n,1),out(n*n);
    HIP_OK(hipMemcpy(a,h.data(),n*n*sizeof(double),hipMemcpyHostToDevice));
    for(auto&v:h)v=2;
    HIP_OK(hipMemcpy(b,h.data(),n*n*sizeof(double),hipMemcpyHostToDevice));
    hipblasHandle_t handle;if(hipblasCreate(&handle)!=HIPBLAS_STATUS_SUCCESS)return 1;
    hipblasStatus_t status=HIPBLAS_STATUS_SUCCESS;
    if(openmx_mgpu_dispatch_dgemm(handle,HIPBLAS_OP_N,HIPBLAS_OP_N,n,n,n,&one,a,n,b,n,&zero,c,n,&status)) {
        std::fprintf(stderr,"FAIL: dispatch outside cluster scope\n");return 1;
    }
    int previous=openmx_mgpu_scope_enter(1);
    const int handled=openmx_mgpu_dispatch_dgemm(handle,HIPBLAS_OP_N,HIPBLAS_OP_N,n,n,n,
        &one,a,n,b,n,&zero,c,n,&status);
    if(!handled || status!=HIPBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr,"FAIL: expected direct-NVLink auto dispatch (handled=%d status=%d)\n",handled,int(status));
        return 1;
    }
    openmx_mgpu_gemm_report report{};openmx_mgpuGemmLastReport(&report);
    HIP_OK(hipMemcpy(out.data(),c,n*n*sizeof(double),hipMemcpyDeviceToHost));
    for(auto v:out)if(v!=2*n)return 1;
    if(report.active_gpus!=2)return 1;
    openmx_mgpu_scope_leave(previous);
    if(openmx_mgpu_dispatch_dgemm(handle,HIPBLAS_OP_N,HIPBLAS_OP_N,n,n,n,&one,a,n,b,n,&zero,c,n,&status))return 1;
    hipblasDestroy(handle);HIP_OK(hipFree(c));HIP_OK(hipFree(b));HIP_OK(hipFree(a));
    std::puts("PASS: actual NVLink auto admission, numeric GEMM, two GPUs, cluster scope exit");
}

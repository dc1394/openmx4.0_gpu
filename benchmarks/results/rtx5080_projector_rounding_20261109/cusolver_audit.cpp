#include <cusolverDn.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
static void record(const char *event,cusolverDnHandle_t handle,int requested,cusolverStatus_t status,
                   int data_type=-1,int64_t n=0) {
    const char *path=std::getenv("OMX_PRECISION_AUDIT_LOG");
    if(!path)return;
    using Get=cusolverStatus_t(*)(cusolverDnHandle_t,cusolverMathMode_t*);
    auto get=reinterpret_cast<Get>(dlsym(RTLD_NEXT,"cusolverDnGetMathMode"));
    cusolverMathMode_t mode=CUSOLVER_DEFAULT_MATH;
    int read_status=get?int(get(handle,&mode)):-1;
    const char *arithmetic=data_type==CUDA_C_64F?"complex_fp64":data_type==CUDA_R_64F?"real_fp64":
        data_type==CUDA_C_32F?"complex_fp32":data_type==CUDA_R_32F?"real_fp32":"none";
    if(FILE *f=std::fopen(path,"a")) {
        std::fprintf(f,"{\"pid\":%d,\"event\":\"%s\",\"handle\":\"%p\",\"requested\":%d,\"mode\":%d,\"status\":%d,\"read_status\":%d,\"native_mode\":%d,\"emulated_mode\":%d,\"arithmetic\":\"%s\",\"n\":%lld}\n",
            int(getpid()),event,(void*)handle,requested,int(mode),int(status),read_status,
            int(CUSOLVER_DEFAULT_MATH),int(CUSOLVER_FP64_EMULATED_FIXEDPOINT_MATH),arithmetic,(long long)n);
        std::fclose(f);
    }
}
extern "C" cusolverStatus_t cusolverDnCreate(cusolverDnHandle_t *handle) {
    using Function=cusolverStatus_t(*)(cusolverDnHandle_t*);
    static auto real=reinterpret_cast<Function>(dlsym(RTLD_NEXT,"cusolverDnCreate"));
    if(!real)std::abort();
    auto status=real(handle);
    if(status==CUSOLVER_STATUS_SUCCESS)record("create",*handle,-1,status);
    return status;
}
extern "C" cusolverStatus_t cusolverDnSetMathMode(cusolverDnHandle_t handle,cusolverMathMode_t mode) {
    using Function=cusolverStatus_t(*)(cusolverDnHandle_t,cusolverMathMode_t);
    static auto real=reinterpret_cast<Function>(dlsym(RTLD_NEXT,"cusolverDnSetMathMode"));
    if(!real)std::abort();
    auto status=real(handle,mode);record("set",handle,int(mode),status);return status;
}
extern "C" cusolverStatus_t cusolverDnXsyevdx(
    cusolverDnHandle_t h,cusolverDnParams_t p,cusolverEigMode_t job,cusolverEigRange_t range,
    cublasFillMode_t uplo,int64_t n,cudaDataType typeA,void *A,int64_t lda,void *vl,void *vu,
    int64_t il,int64_t iu,int64_t *meig,cudaDataType typeW,void *W,cudaDataType compute,
    void *device,size_t device_bytes,void *host,size_t host_bytes,int *info) {
    static auto real=reinterpret_cast<decltype(&cusolverDnXsyevdx)>(dlsym(RTLD_NEXT,"cusolverDnXsyevdx"));
    if(!real)std::abort();
    auto status=real(h,p,job,range,uplo,n,typeA,A,lda,vl,vu,il,iu,meig,typeW,W,compute,
                     device,device_bytes,host,host_bytes,info);
    record("Xsyevdx",h,-1,status,int(typeA),n);return status;
}
extern "C" cusolverStatus_t cusolverDnXsyevd(
    cusolverDnHandle_t h,cusolverDnParams_t p,cusolverEigMode_t job,cublasFillMode_t uplo,
    int64_t n,cudaDataType typeA,void *A,int64_t lda,cudaDataType typeW,void *W,cudaDataType compute,
    void *device,size_t device_bytes,void *host,size_t host_bytes,int *info) {
    static auto real=reinterpret_cast<decltype(&cusolverDnXsyevd)>(dlsym(RTLD_NEXT,"cusolverDnXsyevd"));
    if(!real)std::abort();
    auto status=real(h,p,job,uplo,n,typeA,A,lda,typeW,W,compute,device,device_bytes,host,host_bytes,info);
    record("Xsyevd",h,-1,status,int(typeA),n);return status;
}

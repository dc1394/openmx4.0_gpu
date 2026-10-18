// Test-only LD_PRELOAD observer. Forward calls unchanged; count the actual
// cuSOLVER precision/device used by each MPI rank without GPU synchronization.
#include <cusolverDn.h>
#include <cuda_runtime_api.h>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <unistd.h>

static std::atomic<unsigned long long> counts[64][4];
static std::atomic<unsigned long long> errors;

template<class T> static T next(const char *name)
{
    auto p = reinterpret_cast<T>(dlsym(RTLD_NEXT, name));
    if (!p) { std::fprintf(stderr, "ORDERNTRACE missing=%s\n", name); std::abort(); }
    return p;
}

static const char *rank()
{
    const char *value = std::getenv("OMPI_COMM_WORLD_RANK");
    return value ? value : "unknown";
}

static void record(const char *api, int64_t n, cudaDataType type, cusolverStatus_t status)
{
    static auto device_fn = next<decltype(&cudaGetDevice)>("cudaGetDevice");
    int device = -1;
    const int kind = type == CUDA_R_64F ? 0 : type == CUDA_C_64F ? 1 :
                     type == CUDA_R_32F ? 2 : type == CUDA_C_32F ? 3 : -1;
    if (status != CUSOLVER_STATUS_SUCCESS || device_fn(&device) != cudaSuccess ||
        device < 0 || device >= 64 || kind < 0) {
        ++errors;
        std::fprintf(stderr, "ORDERNTRACE error rank=%s status=%d device=%d type=%d\n",
                     rank(), int(status), device, int(type));
        return;
    }
    if (counts[device][kind].fetch_add(1) == 0)
        std::fprintf(stderr, "ORDERNTRACE first rank=%s pid=%d device=%d api=%s n=%lld type=%d\n",
                     rank(), int(getpid()), device, api, (long long)n, int(type));
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnXsyevd(
    cusolverDnHandle_t h, cusolverDnParams_t p, cusolverEigMode_t j,
    cublasFillMode_t u, int64_t n, cudaDataType a_type, void *a, int64_t lda,
    cudaDataType w_type, void *w, cudaDataType compute, void *db, size_t ds,
    void *hb, size_t hs, int *info)
{
    static auto fn = next<decltype(&cusolverDnXsyevd)>("cusolverDnXsyevd");
    auto status = fn(h,p,j,u,n,a_type,a,lda,w_type,w,compute,db,ds,hb,hs,info);
    record("Xsyevd", n, a_type, status);
    return status;
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnXsyevdx(
    cusolverDnHandle_t h, cusolverDnParams_t p, cusolverEigMode_t j,
    cusolverEigRange_t r, cublasFillMode_t u, int64_t n, cudaDataType a_type,
    void *a, int64_t lda, void *vl, void *vu, int64_t il, int64_t iu,
    int64_t *meig, cudaDataType w_type, void *w, cudaDataType compute,
    void *db, size_t ds, void *hb, size_t hs, int *info)
{
    static auto fn = next<decltype(&cusolverDnXsyevdx)>("cusolverDnXsyevdx");
    auto status = fn(h,p,j,r,u,n,a_type,a,lda,vl,vu,il,iu,meig,w_type,w,compute,db,ds,hb,hs,info);
    record("Xsyevdx", n, a_type, status);
    return status;
}

__attribute__((destructor)) static void report()
{
    for (int d = 0; d < 64; ++d) {
        auto real64=counts[d][0].load(), complex64=counts[d][1].load();
        auto real32=counts[d][2].load(), complex32=counts[d][3].load();
        if (real64 || complex64 || real32 || complex32)
            std::fprintf(stderr, "ORDERNTRACE total rank=%s pid=%d device=%d real64=%llu complex64=%llu real32=%llu complex32=%llu errors=%llu\n",
                         rank(), int(getpid()), d, real64, complex64, real32, complex32, errors.load());
    }
}

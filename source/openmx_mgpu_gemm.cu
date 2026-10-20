#include "openmx_gpu_backend.h"
#include "openmx_mgpu_gemm.h"
#include "openmx_gpu_runtime.h"
#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

extern "C" __attribute__((weak)) OpenMXGpuError OpenMX_GpuMemGetInfo(size_t *, size_t *);
extern "C" __attribute__((weak)) OpenMXBlasStatus openmx_gemmul8Dgemm(
    OpenMXBlasHandle,OpenMXBlasOperation,OpenMXBlasOperation,int,int,int,const double *,
    const double *,int,const double *,int,const double *,double *,int);
extern "C" __attribute__((weak)) OpenMXBlasStatus openmx_gemmul8Zgemm(
    OpenMXBlasHandle,OpenMXBlasOperation,OpenMXBlasOperation,int,int,int,const OpenMXGpuComplex *,
    const OpenMXGpuComplex *,int,const OpenMXGpuComplex *,int,const OpenMXGpuComplex *,OpenMXGpuComplex *,int);

namespace {
constexpr size_t MiB = size_t(1024)*1024;
thread_local bool in_worker = false;
#if !OPENMX_GPU_HIP
thread_local unsigned staging_depth = 0;
#endif
thread_local openmx_mgpu_gemm_report report = {};

size_t budget(const char *name, size_t fallback_mib)
{
    const char *value = std::getenv(name);
    if (!value || !*value) return fallback_mib*MiB;
    char *end = nullptr;
    const unsigned long long parsed = std::strtoull(value,&end,10);
    if (*value=='-' || end==value || *end || parsed>std::numeric_limits<size_t>::max()/MiB)
        return fallback_mib*MiB;
    return size_t(parsed)*MiB;
}
bool option(const char *name) { const char *v=std::getenv(name);return v && *v=='1'; }
bool product(size_t a,size_t b,size_t *out)
{
    if (b && a>std::numeric_limits<size_t>::max()/b) return false;
    *out=a*b;return true;
}
OpenMXGpuError meminfo(size_t *free_bytes,size_t *total)
{
    return OpenMX_GpuMemGetInfo ? OpenMX_GpuMemGetInfo(free_bytes,total) : OpenMXGpuMemGetInfo(free_bytes,total);
}
bool available(size_t bytes,size_t reserve)
{
    size_t free_bytes=0,total=0;
    return meminfo(&free_bytes,&total)==OPENMX_GPU_SUCCESS && bytes<=free_bytes && reserve<=free_bytes-bytes;
}

/* Handles and streams survive calls; device operand buffers do not. Stable
 * stream identities also bound the existing bridge's workspace cache. */
struct Context {
    int device;
    OpenMXGpuStream stream=nullptr;
    OpenMXBlasHandle blas=nullptr;
#if !OPENMX_GPU_HIP
    void *staging=nullptr;
    size_t staging_bytes=0;
#endif
    explicit Context(int id):device(id) {}
    ~Context() {
        int saved=-1;OpenMXGpuGetDevice(&saved);OpenMXGpuSetDevice(device);
        if(stream) OpenMXGpuStreamSynchronize(stream);
#if !OPENMX_GPU_HIP
        if(staging) OpenMXGpuFreeHost(staging);
#endif
        if(blas) OpenMXBlasDestroy(blas);
        if(stream) OpenMXGpuStreamDestroy(stream);
        if(saved>=0) OpenMXGpuSetDevice(saved);
    }
};
thread_local std::vector<std::unique_ptr<Context>> contexts;
Context *context(int device)
{
    for(auto &c:contexts) if(c->device==device) return c.get();
    if(OpenMXGpuSetDevice(device)!=OPENMX_GPU_SUCCESS) return nullptr;
    auto c=std::make_unique<Context>(device);
    if(OpenMXGpuStreamCreateWithFlags(&c->stream,OPENMX_GPU_STREAM_NONBLOCKING)!=OPENMX_GPU_SUCCESS ||
       OpenMXBlasCreate(&c->blas)!=OPENMX_BLAS_SUCCESS ||
       OpenMXBlasSetStream(c->blas,c->stream)!=OPENMX_BLAS_SUCCESS) return nullptr;
    contexts.push_back(std::move(c));return contexts.back().get();
}
struct Worker {
    Context *ctx=nullptr;
    void *buffer=nullptr,*host=nullptr;
    size_t bytes=0;
#if OPENMX_GPU_HIP
    bool peer=false,used=false;
#else
    bool peer=false,used=false,cached_host=false;
#endif
    int cached_row=-1,cached_m=-1,cached_col=-1,cached_n=-1;
    int row=0,col=0,m=0,n=0;
};
struct Operation {
    int origin=-1;
#if OPENMX_GPU_HIP
    bool host_matrices=false,host_result=false;
#else
    bool host_matrices=false,host_result=false,pageable=false;
#endif
    OpenMXGpuStream copy_stream=nullptr;
    OpenMXGpuEvent ready=nullptr;
    void *result=nullptr;
    std::vector<Worker> workers;
    bool clean=true;
    ~Operation() { cleanup(); }
    void cleanup() {
        if(origin<0) return;
        for(auto &w:workers) {
            if(w.ctx && OpenMXGpuSetDevice(w.ctx->device)==OPENMX_GPU_SUCCESS) {
                if(OpenMXGpuStreamSynchronize(w.ctx->stream)!=OPENMX_GPU_SUCCESS) clean=false;
            }
        }
        if(OpenMXGpuSetDevice(origin)!=OPENMX_GPU_SUCCESS) clean=false;
        if(copy_stream) {
            if(OpenMXGpuStreamSynchronize(copy_stream)!=OPENMX_GPU_SUCCESS) clean=false;
            if(OpenMXGpuStreamDestroy(copy_stream)!=OPENMX_GPU_SUCCESS) clean=false;
            copy_stream=nullptr;
        }
        for(auto &w:workers) {
            if(w.buffer && (OpenMXGpuSetDevice(w.ctx->device)!=OPENMX_GPU_SUCCESS || OpenMXGpuFree(w.buffer)!=OPENMX_GPU_SUCCESS)) clean=false;
#if OPENMX_GPU_HIP
            if(w.host && OpenMXGpuFreeHost(w.host)!=OPENMX_GPU_SUCCESS) clean=false;
#else
            if(w.host && !w.cached_host && OpenMXGpuFreeHost(w.host)!=OPENMX_GPU_SUCCESS) clean=false;
#endif
            w.buffer=w.host=nullptr;
        }
        if(OpenMXGpuSetDevice(origin)!=OPENMX_GPU_SUCCESS) clean=false;
        if(ready) { if(OpenMXGpuEventDestroy(ready)!=OPENMX_GPU_SUCCESS) clean=false;ready=nullptr; }
        if(result) {
            if(host_result) std::free(result);
            else if(OpenMXGpuFree(result)!=OPENMX_GPU_SUCCESS) clean=false;
            result=nullptr;
        }
    }
};

/* A pitched peer copy retains storage orientation. CUDA's copy engine moves
 * raw A/B submatrices; cuBLAS applies the requested transpose/conjugation. */
OpenMXGpuError peer_copy(void *dst,int dstdev,size_t dpitch,const void *src,int srcdev,size_t spitch,
                      size_t width,size_t height,OpenMXGpuStream stream)
{
    if(srcdev==dstdev) return OpenMXGpuMemcpy2DAsync(dst,dpitch,src,spitch,width,height,OPENMX_GPU_COPY_DEVICE_TO_DEVICE,stream);
    OpenMXGpuPeerCopy3D p={};
    p.srcDevice=srcdev;p.dstDevice=dstdev;
    p.srcPtr=OpenMXGpuMakePitchedPtr(const_cast<void *>(src),spitch,width,height);
    p.dstPtr=OpenMXGpuMakePitchedPtr(dst,dpitch,width,height);
    p.extent=OpenMXGpuMakeExtent(width,height,1);
    return OpenMXGpuMemcpy3DPeerAsync(&p,stream);
}

bool input_copy(Operation &op,Worker &w,void *dst,const void *src,size_t spitch,
                size_t width,size_t height,void *host)
{
    if(OpenMXGpuSetDevice(w.ctx->device)!=OPENMX_GPU_SUCCESS) return false;
    if(op.host_matrices) {
#if !OPENMX_GPU_HIP
        if(op.pageable) {
            if(OpenMXGpuMemcpy2DAsync(dst,width,src,spitch,width,height,OPENMX_GPU_COPY_HOST_TO_DEVICE,w.ctx->stream)!=OPENMX_GPU_SUCCESS)
                return false;
            ++report.host_transfers;return true;
        }
        #pragma omp parallel for schedule(static) if(width*height>16*MiB)
#endif
        for(size_t j=0;j<height;++j)
            std::memcpy(static_cast<char *>(host)+j*width,static_cast<const char *>(src)+j*spitch,width);
        if(OpenMXGpuMemcpyAsync(dst,host,width*height,OPENMX_GPU_COPY_HOST_TO_DEVICE,w.ctx->stream)!=OPENMX_GPU_SUCCESS) return false;
        ++report.host_transfers;return true;
    }
    if(w.peer) {
        const OpenMXGpuError err=peer_copy(dst,w.ctx->device,width,src,op.origin,spitch,width,height,w.ctx->stream);
        if(err==OPENMX_GPU_SUCCESS) {if(w.ctx->device!=op.origin) ++report.peer_transfers;return true;}
        return false; // no user output was touched: original backend may retry
    }
    if(OpenMXGpuSetDevice(op.origin)!=OPENMX_GPU_SUCCESS ||
       OpenMXGpuMemcpy2DAsync(host,width,src,spitch,width,height,OPENMX_GPU_COPY_DEVICE_TO_HOST,op.copy_stream)!=OPENMX_GPU_SUCCESS ||
       OpenMXGpuStreamSynchronize(op.copy_stream)!=OPENMX_GPU_SUCCESS ||
       OpenMXGpuSetDevice(w.ctx->device)!=OPENMX_GPU_SUCCESS ||
       OpenMXGpuMemcpyAsync(dst,host,width*height,OPENMX_GPU_COPY_HOST_TO_DEVICE,w.ctx->stream)!=OPENMX_GPU_SUCCESS) return false;
    ++report.host_transfers;return true;
}
bool output_copy(Operation &op,Worker &w,void *dst,size_t dpitch,const void *src,
                 size_t width,size_t height,void *host)
{
    if(OpenMXGpuSetDevice(w.ctx->device)!=OPENMX_GPU_SUCCESS) return false;
    if(op.host_result) {
#if !OPENMX_GPU_HIP
        if(op.pageable) {
            if(OpenMXGpuMemcpy2DAsync(dst,dpitch,src,width,width,height,OPENMX_GPU_COPY_DEVICE_TO_HOST,w.ctx->stream)!=OPENMX_GPU_SUCCESS)
                return false;
            ++report.host_transfers;return true;
        }
#endif
        if(OpenMXGpuMemcpyAsync(host,src,width*height,OPENMX_GPU_COPY_DEVICE_TO_HOST,w.ctx->stream)!=OPENMX_GPU_SUCCESS ||
           OpenMXGpuStreamSynchronize(w.ctx->stream)!=OPENMX_GPU_SUCCESS) return false;
#if !OPENMX_GPU_HIP
        #pragma omp parallel for schedule(static) if(width*height>16*MiB)
#endif
        for(size_t j=0;j<height;++j)
            std::memcpy(static_cast<char *>(dst)+j*dpitch,static_cast<const char *>(host)+j*width,width);
        ++report.host_transfers;return true;
    }
    if(w.peer) {
        const OpenMXGpuError err=peer_copy(dst,op.origin,dpitch,src,w.ctx->device,width,width,height,w.ctx->stream);
        if(err==OPENMX_GPU_SUCCESS) {if(w.ctx->device!=op.origin) ++report.peer_transfers;return true;}
        return false;
    }
    if(OpenMXGpuMemcpyAsync(host,src,width*height,OPENMX_GPU_COPY_DEVICE_TO_HOST,w.ctx->stream)!=OPENMX_GPU_SUCCESS ||
       OpenMXGpuStreamSynchronize(w.ctx->stream)!=OPENMX_GPU_SUCCESS ||
       OpenMXGpuSetDevice(op.origin)!=OPENMX_GPU_SUCCESS ||
       OpenMXGpuMemcpy2DAsync(dst,dpitch,host,width,width,height,OPENMX_GPU_COPY_HOST_TO_DEVICE,op.copy_stream)!=OPENMX_GPU_SUCCESS ||
       OpenMXGpuStreamSynchronize(op.copy_stream)!=OPENMX_GPU_SUCCESS) return false;
    ++report.host_transfers;return true;
}
template<class T> bool zero(T v) {
    if constexpr(std::is_same_v<T,double>) return v==0;
    else return v.x==0 && v.y==0;
}
template<class T> OpenMXBlasStatus local_gemm(OpenMXBlasHandle h,OpenMXBlasOperation ta,OpenMXBlasOperation tb,
    int m,int n,int k,const T *alpha,const T *a,int lda,const T *b,int ldb,const T *beta,T *c,int ldc)
{
    const char *backend=std::getenv("OPENMX_MGPU_GEMM_BACKEND");
    const bool bridge=!(backend && std::strcmp(backend,"native")==0);
    struct Guard { Guard(){in_worker=true;} ~Guard(){in_worker=false;} } guard;
    if constexpr(std::is_same_v<T,double>) {
        if(bridge && openmx_gemmul8Dgemm) {
            ++report.bridge_calls;return openmx_gemmul8Dgemm(h,ta,tb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc);
        }
        return OpenMXBlasDgemm(h,ta,tb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc);
    } else {
        if(bridge && openmx_gemmul8Zgemm) {
            ++report.bridge_calls;return openmx_gemmul8Zgemm(h,ta,tb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc);
        }
        return OpenMXBlasZgemm(h,ta,tb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc);
    }
}
bool owned_pointer(const void *p,int device)
{
    OpenMXGpuPointerAttributes attr={};
    const auto error=OpenMXGpuPointerGetAttributes(&attr,p);
    if(error!=OPENMX_GPU_SUCCESS) {
        if(error==OPENMX_GPU_INVALID_VALUE && OpenMXGpuPeekAtLastError()==error)
            (void)OpenMXGpuGetLastError(); // our unsupported host-pointer query
        return false;
    }
    return attr.type==OPENMX_GPU_DEVICE_MEMORY && attr.device==device;
}
bool valid_op(OpenMXBlasOperation op) { return op==OPENMX_BLAS_OP_N || op==OPENMX_BLAS_OP_T || op==OPENMX_BLAS_OP_C; }
bool enable_peer_pair(int first,int second)
{
    // Peer enablement belongs to CUDA contexts, not handles or host threads.
    // Cache both successful and unsupported directions to avoid re-enabling
    // an existing mapping on every GEMM (and racing another host thread).
    static std::mutex mutex;
    static std::unordered_map<unsigned long long,bool> enabled;
    std::lock_guard<std::mutex> lock(mutex);
    for(int direction=0;direction<2;++direction) {
        const int from=direction?second:first,to=direction?first:second;
        const auto key=(static_cast<unsigned long long>(from)<<32)|static_cast<unsigned>(to);
        const auto known=enabled.find(key);
        if(known!=enabled.end()) {if(!known->second)return false;continue;}
        if(OpenMXGpuSetDevice(from)!=OPENMX_GPU_SUCCESS) return false;
        const OpenMXGpuError e=OpenMXGpuDeviceEnablePeerAccess(to,0);
        if(e!=OPENMX_GPU_SUCCESS) {
            if(e==OPENMX_GPU_PEER_ALREADY_ENABLED || e==OPENMX_GPU_PEER_UNSUPPORTED) {
                if(OpenMXGpuPeekAtLastError()==e) (void)OpenMXGpuGetLastError();
            }
            if(e!=OPENMX_GPU_PEER_ALREADY_ENABLED) {
                if(e==OPENMX_GPU_PEER_UNSUPPORTED) enabled[key]=false;
                return false;
            }
        }
        enabled[key]=true;
    }
    // Peer mappings are process/context state; keep enabled mappings rather
    // than disabling access another CUDA library may subsequently use.
    return true;
}

template<class T> int gemm(OpenMXBlasHandle handle,int ngpu,const int *devices,OpenMXBlasOperation ta,
    OpenMXBlasOperation tb,int m,int n,int k,const T *alpha,const T *a,int lda,const T *b,int ldb,
    const T *beta,T *c,int ldc,OpenMXBlasStatus *status,bool host_matrices=false)
{
    const auto start=std::chrono::steady_clock::now();
    if(in_worker) {if(status) *status=OPENMX_BLAS_SUCCESS;return 0;}
    report={};report.requested_gpus=ngpu;report.reason="not eligible";
    if(!status) return 0;
    *status=OPENMX_BLAS_SUCCESS;
    if(OpenMXGpuPeekAtLastError()!=OPENMX_GPU_SUCCESS) {
        *status=OPENMX_BLAS_EXECUTION_FAILED;
        report.reason="pending caller GPU error";return 1;
    }
    if(ngpu<2 || ngpu>64 || !devices || m<=0 || n<=0 || k<=0) return 0;
    if(!valid_op(ta) || !valid_op(tb) || lda<(ta==OPENMX_BLAS_OP_N?m:k) || ldb<(tb==OPENMX_BLAS_OP_N?k:n) ||
       ldc<m || !alpha || !beta || !a || !b || !c) {
        *status=OPENMX_BLAS_INVALID_VALUE;report.reason="invalid GEMM arguments";return 1;
    }
    Operation op;
    op.host_matrices=host_matrices;
#if !OPENMX_GPU_HIP
    op.pageable=host_matrices && option("OPENMX_MGPU_GEMM_PAGEABLE");
#endif
    if(OpenMXGpuGetDevice(&op.origin)!=OPENMX_GPU_SUCCESS) {report.reason="current device unavailable";return 0;}
    auto finish=[&](int handled,const char *reason) {
        op.cleanup();
        if(!op.clean) {handled=1;*status=OPENMX_BLAS_EXECUTION_FAILED;reason=OPENMX_GPU_RUNTIME_NAME " cleanup/synchronization failure";}
        if(OpenMXGpuSetDevice(op.origin)!=OPENMX_GPU_SUCCESS) {handled=1;*status=OPENMX_BLAS_EXECUTION_FAILED;reason="restore device failure";}
        report.reason=reason;report.seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        if(!handled) {
            const auto pending=OpenMXGpuPeekAtLastError();
            if(pending==OPENMX_GPU_OUT_OF_MEMORY) (void)OpenMXGpuGetLastError();
            else if(pending!=OPENMX_GPU_SUCCESS) {
                handled=1;*status=OPENMX_BLAS_EXECUTION_FAILED;
                report.reason="GPU execution error during preparation";
            }
        }
        op.origin=-1;return handled;
    };
    if(!host_matrices && (!owned_pointer(a,op.origin) || !owned_pointer(b,op.origin) || !owned_pointer(c,op.origin)))
        return finish(0,"operands are not on the current device");
    OpenMXGpuStream original=nullptr;
    OpenMXBlasPointerMode pointer_mode=OPENMX_BLAS_POINTER_HOST;
    OpenMXBlasMathMode math_mode=OPENMX_BLAS_DEFAULT_MATH;
    OpenMXBlasAtomicsMode atomics_mode=OPENMX_BLAS_ATOMICS_NOT_ALLOWED;
    if(!host_matrices && (OpenMXBlasGetStream(handle,&original)!=OPENMX_BLAS_SUCCESS ||
       OpenMXBlasGetPointerMode(handle,&pointer_mode)!=OPENMX_BLAS_SUCCESS ||
       OpenMXBlasGetMathMode(handle,&math_mode)!=OPENMX_BLAS_SUCCESS ||
       OpenMXBlasGetAtomicsMode(handle,&atomics_mode)!=OPENMX_BLAS_SUCCESS)) return finish(0,"handle query failed");
    OpenMXGpuCaptureStatus capture;
    if(OpenMXGpuStreamIsCapturing(original,&capture)!=OPENMX_GPU_SUCCESS || capture!=OPENMX_GPU_CAPTURE_NONE)
        return finish(0,OPENMX_GPU_RUNTIME_NAME " graph capture is unsupported");
    T host_alpha,host_beta;
    if(pointer_mode==OPENMX_BLAS_POINTER_DEVICE) {
        if(OpenMXGpuMemcpyAsync(&host_alpha,alpha,sizeof(T),OPENMX_GPU_COPY_DEVICE_TO_HOST,original)!=OPENMX_GPU_SUCCESS ||
           OpenMXGpuMemcpyAsync(&host_beta,beta,sizeof(T),OPENMX_GPU_COPY_DEVICE_TO_HOST,original)!=OPENMX_GPU_SUCCESS ||
           OpenMXGpuStreamSynchronize(original)!=OPENMX_GPU_SUCCESS) return finish(0,"device scalar copy failed");
    } else {host_alpha=*alpha;host_beta=*beta;}
    if(zero(host_alpha)) return finish(0,"zero alpha uses the original backend");
    size_t elements=0,result_bytes=0;
    if(!product(size_t(m),size_t(n),&elements) || !product(elements,sizeof(T),&result_bytes))
        return finish(0,"transaction size overflow");
    const size_t reserve=budget("OPENMX_MGPU_GEMM_RESERVE_MB",512);
    size_t tile_budget=budget("OPENMX_MGPU_GEMM_WORKSPACE_MB",1024);
    const size_t host_budget=budget("OPENMX_MGPU_GEMM_HOST_MB",256);
    if(!tile_budget) return finish(0,"worker budget is zero");
    /* A full private result must not require a second full matrix on the
       owner GPU. Keep the transaction in host memory when that would
       exceed the device budget, reserving room for the owner's tile. */
    op.host_result=host_matrices || option("OPENMX_MGPU_GEMM_RESULT_HOST") ||
        result_bytes>budget("OPENMX_MGPU_GEMM_RESULT_MB",2048) ||
        result_bytes>std::numeric_limits<size_t>::max()-tile_budget ||
        !available(result_bytes+tile_budget,reserve);
    if(op.host_result) {
        if(result_bytes>budget("OPENMX_MGPU_GEMM_HOST_RESULT_MB",65536))
            return finish(0,"host transaction budget exceeded");
        op.result=std::malloc(result_bytes);
        tile_budget=std::min(tile_budget,host_budget);
        if(!op.result) return finish(0,"host transaction allocation failed");
    } else if(const auto error=OpenMXGpuMalloc(&op.result,result_bytes); error!=OPENMX_GPU_SUCCESS) {
        if(error!=OPENMX_GPU_OUT_OF_MEMORY) {
            *status=OPENMX_BLAS_EXECUTION_FAILED;
            return finish(1,"device transaction allocation failed");
        }
        if(OpenMXGpuPeekAtLastError()==OPENMX_GPU_OUT_OF_MEMORY) (void)OpenMXGpuGetLastError();
        op.host_result=true;
        if(result_bytes>budget("OPENMX_MGPU_GEMM_HOST_RESULT_MB",65536))
            return finish(0,"host transaction budget exceeded");
        op.result=std::malloc(result_bytes);
        tile_budget=std::min(tile_budget,host_budget);
        if(!op.result) return finish(0,"host transaction allocation failed");
    }
    int count=0;if(OpenMXGpuGetDeviceCount(&count)!=OPENMX_GPU_SUCCESS) return finish(0,"device count unavailable");
    std::vector<std::string> buses;
    for(int i=0;i<ngpu;++i) {
        char bus[64]={};
        if(devices[i]<0 || devices[i]>=count || OpenMXGpuDeviceGetPCIBusId(bus,sizeof(bus),devices[i])!=OPENMX_GPU_SUCCESS)
            return finish(0,"invalid device list");
        if(std::find(buses.begin(),buses.end(),bus)!=buses.end()) return finish(0,"devices share one physical GPU");
        buses.emplace_back(bus);
        Worker w;w.ctx=context(devices[i]);if(!w.ctx) return finish(0,"worker context unavailable");
        int can=devices[i]==op.origin,reverse=can;
        if(!host_matrices && !can && (OpenMXGpuDeviceCanAccessPeer(&can,devices[i],op.origin)!=OPENMX_GPU_SUCCESS ||
                    OpenMXGpuDeviceCanAccessPeer(&reverse,op.origin,devices[i])!=OPENMX_GPU_SUCCESS)) return finish(0,"peer query failed");
        w.peer=!host_matrices && (devices[i]==op.origin || (can && reverse && !option("OPENMX_MGPU_GEMM_FORCE_HOST") &&
                                       enable_peer_pair(devices[i],op.origin)));
        if(OpenMXGpuPeekAtLastError()!=OPENMX_GPU_SUCCESS)
            return finish(0,"peer setup failed");
        if(!w.peer) tile_budget=std::min(tile_budget,host_budget);
        op.workers.push_back(w);
    }
    int tile_m=m,tile_n=std::max(1,int((size_t(n)+ngpu-1)/ngpu));
    if(n<ngpu) tile_m=std::max(1,int((size_t(m)+((ngpu+n-1)/n)-1)/((ngpu+n-1)/n)));
    auto bytes_for=[&](int tm,int tn) -> size_t {
        const size_t total=size_t(tm)*k+size_t(k)*tn+size_t(tm)*tn;
        return total>std::numeric_limits<size_t>::max()/sizeof(T)?std::numeric_limits<size_t>::max():total*sizeof(T);
    };
    while(bytes_for(tile_m,tile_n)>tile_budget && (tile_m>1 || tile_n>1)) {
        if(tile_m>=tile_n && tile_m>1) tile_m=(tile_m+1)/2;
        else tile_n=(tile_n+1)/2;
    }
    // Ceil(n/ngpu) can yield fewer than ngpu tiles when n is only just
    // above ngpu (e.g. five columns on four GPUs). Split further, retaining
    // full K, so skinny back transforms can still use every selected GPU.
    while(((size_t(m)+tile_m-1)/tile_m)*((size_t(n)+tile_n-1)/tile_n)<size_t(ngpu) &&
          (tile_m>1 || tile_n>1)) {
        if(tile_n>1) tile_n=(tile_n+1)/2;
        else tile_m=(tile_m+1)/2;
    }
    const size_t worker_bytes=bytes_for(tile_m,tile_n);
    if(worker_bytes>tile_budget) return finish(0,"full K does not fit the worker budget");
    const size_t tile_rows=(size_t(m)+tile_m-1)/tile_m,tile_cols=(size_t(n)+tile_n-1)/tile_n;
    if(tile_rows*tile_cols<size_t(ngpu)) return finish(0,"too few output tiles");
    report.transaction_bytes=result_bytes;report.worker_bytes=worker_bytes;
    if(OpenMXGpuSetDevice(op.origin)!=OPENMX_GPU_SUCCESS) return finish(0,"origin device selection failed");
    if(OpenMXGpuStreamCreateWithFlags(&op.copy_stream,OPENMX_GPU_STREAM_NONBLOCKING)!=OPENMX_GPU_SUCCESS ||
       OpenMXGpuEventCreateWithFlags(&op.ready,OPENMX_GPU_EVENT_DISABLE_TIMING)!=OPENMX_GPU_SUCCESS ||
       OpenMXGpuEventRecord(op.ready,original)!=OPENMX_GPU_SUCCESS ||
       OpenMXGpuStreamWaitEvent(op.copy_stream,op.ready,0)!=OPENMX_GPU_SUCCESS) return finish(0,"input dependency setup failed");
    for(auto &w:op.workers) {
        if(OpenMXGpuSetDevice(w.ctx->device)!=OPENMX_GPU_SUCCESS || !available(worker_bytes,reserve) ||
           OpenMXGpuMalloc(&w.buffer,worker_bytes)!=OPENMX_GPU_SUCCESS) return finish(0,"worker allocation failed");
        w.bytes=worker_bytes;
#if OPENMX_GPU_HIP
        if((!w.peer || op.host_result) && OpenMXGpuHostAlloc(&w.host,worker_bytes,OPENMX_GPU_HOST_ALLOC_PORTABLE)!=OPENMX_GPU_SUCCESS)
#else
        if((!w.peer || op.host_result) && !op.pageable) {
            if(staging_depth && host_matrices) {
                /* Reuse registration only inside an explicitly bounded host
                   GEMM sequence. Device scratch still returns to MAGMA after
                   every product. Never retain more than the current budget. */
                auto &ctx=*w.ctx;
                if(ctx.staging_bytes<worker_bytes || ctx.staging_bytes>tile_budget) {
                    if(ctx.staging && OpenMXGpuFreeHost(ctx.staging)!=OPENMX_GPU_SUCCESS)
                        return finish(0,"cached staging release failed");
                    ctx.staging=nullptr;ctx.staging_bytes=0;
                }
                if(!ctx.staging) {
                    if(OpenMXGpuHostAlloc(&ctx.staging,worker_bytes,OPENMX_GPU_HOST_ALLOC_PORTABLE)!=OPENMX_GPU_SUCCESS)
#endif
                        return finish(0,"pinned staging allocation failed");
#if !OPENMX_GPU_HIP
                    ctx.staging_bytes=worker_bytes;
                }
                w.host=ctx.staging;w.cached_host=true;
            } else if(OpenMXGpuHostAlloc(&w.host,worker_bytes,OPENMX_GPU_HOST_ALLOC_PORTABLE)!=OPENMX_GPU_SUCCESS)
                return finish(0,"pinned staging allocation failed");
        }
#endif
        if(OpenMXGpuStreamWaitEvent(w.ctx->stream,op.ready,0)!=OPENMX_GPU_SUCCESS ||
           OpenMXBlasSetPointerMode(w.ctx->blas,OPENMX_BLAS_POINTER_HOST)!=OPENMX_BLAS_SUCCESS ||
           OpenMXBlasSetMathMode(w.ctx->blas,math_mode)!=OPENMX_BLAS_SUCCESS ||
           OpenMXBlasSetAtomicsMode(w.ctx->blas,atomics_mode)!=OPENMX_BLAS_SUCCESS)
            return finish(0,"worker handle setup failed");
    }
    const size_t a_elements=size_t(tile_m)*k,b_elements=size_t(k)*tile_n;
    for(size_t first=0;first<tile_rows*tile_cols;first+=ngpu) {
        const int active=int(std::min(size_t(ngpu),tile_rows*tile_cols-first));
        for(int i=0;i<active;++i) {
            Worker &w=op.workers[i];
            w.row=int((first+i)/tile_cols)*tile_m;w.col=int((first+i)%tile_cols)*tile_n;
            w.m=std::min(tile_m,m-w.row);w.n=std::min(tile_n,n-w.col);
            T *wa=static_cast<T *>(w.buffer),*wb=wa+a_elements,*wc=wb+b_elements;
            T *ha=static_cast<T *>(w.host),*hb=ha?ha+a_elements:nullptr,*hc=ha?hb+b_elements:nullptr;
            if(w.cached_row!=w.row || w.cached_m!=w.m) {
                const T *src=a+(ta==OPENMX_BLAS_OP_N?size_t(w.row):size_t(w.row)*lda);
                if(!input_copy(op,w,wa,src,size_t(lda)*sizeof(T),size_t(ta==OPENMX_BLAS_OP_N?w.m:k)*sizeof(T),
                               ta==OPENMX_BLAS_OP_N?k:w.m,ha)) return finish(0,"operand A transfer failed");
                w.cached_row=w.row;w.cached_m=w.m;
            }
            if(w.cached_col!=w.col || w.cached_n!=w.n) {
                const T *src=b+(tb==OPENMX_BLAS_OP_N?size_t(w.col)*ldb:size_t(w.col));
                if(!input_copy(op,w,wb,src,size_t(ldb)*sizeof(T),size_t(tb==OPENMX_BLAS_OP_N?k:w.n)*sizeof(T),
                               tb==OPENMX_BLAS_OP_N?w.n:k,hb)) return finish(0,"operand B transfer failed");
                w.cached_col=w.col;w.cached_n=w.n;
            }
            if(!zero(host_beta) && !input_copy(op,w,wc,c+size_t(w.col)*ldc+w.row,size_t(ldc)*sizeof(T),
                                              size_t(w.m)*sizeof(T),w.n,hc)) return finish(0,"input C transfer failed");
            if(OpenMXGpuSetDevice(w.ctx->device)!=OPENMX_GPU_SUCCESS) return finish(0,"worker device selection failed");
            const OpenMXBlasStatus local=local_gemm(w.ctx->blas,ta,tb,w.m,w.n,k,&host_alpha,wa,
                ta==OPENMX_BLAS_OP_N?w.m:k,wb,tb==OPENMX_BLAS_OP_N?k:w.n,&host_beta,wc,w.m);
            if(local!=OPENMX_BLAS_SUCCESS) {
                if(local==OPENMX_BLAS_ALLOC_FAILED) return finish(0,"worker GEMM allocation failed before commit");
                *status=local;return finish(1,"worker GEMM failed before commit");
            }
            if(!w.used) {w.used=true;++report.active_gpus;}
            ++report.tiles;
        }
        // All workers have queued their GEMMs before any host-staged gather waits.
        for(int i=0;i<active;++i) {
            Worker &w=op.workers[i];
            T *wc=static_cast<T *>(w.buffer)+a_elements+b_elements;
            T *hc=w.host?static_cast<T *>(w.host)+a_elements+b_elements:nullptr;
            T *dst=static_cast<T *>(op.result)+size_t(w.col)*m+w.row;
            if(!output_copy(op,w,dst,size_t(m)*sizeof(T),wc,size_t(w.m)*sizeof(T),w.n,hc))
                return finish(0,"result transfer failed before commit");
        }
        // Bound buffer lifetimes: the next tile can overwrite host/device input
        // staging only after its prior transfers and GEMM have completed.
        for(int i=0;i<active;++i) {
            Worker &w=op.workers[i];
            if(OpenMXGpuSetDevice(w.ctx->device)!=OPENMX_GPU_SUCCESS || OpenMXGpuStreamSynchronize(w.ctx->stream)!=OPENMX_GPU_SUCCESS)
                return finish(0,"worker synchronization failed before commit");
        }
    }
    if(OpenMXGpuSetDevice(op.origin)!=OPENMX_GPU_SUCCESS || OpenMXGpuStreamSynchronize(op.copy_stream)!=OPENMX_GPU_SUCCESS)
        return finish(0,"gather synchronization failed before commit");
    // The only write to caller C. Preserve all padding and make beta retries
    // unambiguous by treating every commit error as handled, never declined.
    if(host_matrices) {
#if !OPENMX_GPU_HIP
        #pragma omp parallel for schedule(static) if(result_bytes>16*MiB)
#endif
        for(int j=0;j<n;++j)
            std::memcpy(c+size_t(j)*ldc,static_cast<T *>(op.result)+size_t(j)*m,size_t(m)*sizeof(T));
    } else if(OpenMXGpuMemcpy2DAsync(c,size_t(ldc)*sizeof(T),op.result,size_t(m)*sizeof(T),size_t(m)*sizeof(T),n,
                         op.host_result?OPENMX_GPU_COPY_HOST_TO_DEVICE:OPENMX_GPU_COPY_DEVICE_TO_DEVICE,original)!=OPENMX_GPU_SUCCESS || OpenMXGpuStreamSynchronize(original)!=OPENMX_GPU_SUCCESS) {
        *status=OPENMX_BLAS_EXECUTION_FAILED;return finish(1,"output commit failed");
    }
    return finish(1,"success");
}
} // namespace

extern "C" int openmx_mgpuGemmInWorker(void) { return in_worker?1:0; }
extern "C" void openmx_mgpuGemmLastReport(openmx_mgpu_gemm_report *out) { if(out) *out=report; }
extern "C" void openmx_mgpuGemmRelease(void) { if(!in_worker) contexts.clear(); }
#if !OPENMX_GPU_HIP
extern "C" void openmx_mgpuGemmStagingBegin(void) { ++staging_depth; }
extern "C" void openmx_mgpuGemmStagingEnd(void) {
    if(!staging_depth || --staging_depth) return;
    for(auto &ctx:contexts) if(ctx->staging) {
        /* Try has synchronized every worker before returning. */
        if(OpenMXGpuFreeHost(ctx->staging)==OPENMX_GPU_SUCCESS) {
            ctx->staging=nullptr;ctx->staging_bytes=0;
        }
    }
}
#endif
extern "C" int openmx_mgpuDgemmHostTry(int ngpu,const int *devices,OpenMXBlasOperation ta,
    OpenMXBlasOperation tb,int m,int n,int k,const double *alpha,const double *a,int lda,const double *b,int ldb,
    const double *beta,double *c,int ldc,OpenMXBlasStatus *status)
{
    try {return gemm(nullptr,ngpu,devices,ta,tb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc,status,true);}
    catch(const std::bad_alloc &) {report.reason="host allocation failed";if(status)*status=OPENMX_BLAS_ALLOC_FAILED;return 0;}
}
extern "C" int openmx_mgpuZgemmHostTry(int ngpu,const int *devices,OpenMXBlasOperation ta,
    OpenMXBlasOperation tb,int m,int n,int k,const OpenMXGpuComplex *alpha,const OpenMXGpuComplex *a,int lda,
    const OpenMXGpuComplex *b,int ldb,const OpenMXGpuComplex *beta,OpenMXGpuComplex *c,int ldc,OpenMXBlasStatus *status)
{
    try {return gemm(nullptr,ngpu,devices,ta,tb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc,status,true);}
    catch(const std::bad_alloc &) {report.reason="host allocation failed";if(status)*status=OPENMX_BLAS_ALLOC_FAILED;return 0;}
}
extern "C" int openmx_mgpuDgemmTry(OpenMXBlasHandle h,int ngpu,const int *devices,OpenMXBlasOperation ta,
    OpenMXBlasOperation tb,int m,int n,int k,const double *alpha,const double *a,int lda,const double *b,int ldb,
    const double *beta,double *c,int ldc,OpenMXBlasStatus *status)
{
    try {return gemm(h,ngpu,devices,ta,tb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc,status);}
    catch(const std::bad_alloc &) {report.reason="host metadata allocation failed";if(status)*status=OPENMX_BLAS_SUCCESS;return 0;}
}
extern "C" int openmx_mgpuZgemmTry(OpenMXBlasHandle h,int ngpu,const int *devices,OpenMXBlasOperation ta,
    OpenMXBlasOperation tb,int m,int n,int k,const OpenMXGpuComplex *alpha,const OpenMXGpuComplex *a,int lda,
    const OpenMXGpuComplex *b,int ldb,const OpenMXGpuComplex *beta,OpenMXGpuComplex *c,int ldc,OpenMXBlasStatus *status)
{
    try {return gemm(h,ngpu,devices,ta,tb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc,status);}
    catch(const std::bad_alloc &) {report.reason="host metadata allocation failed";if(status)*status=OPENMX_BLAS_SUCCESS;return 0;}
}
#if !OPENMX_GPU_HIP

namespace {
struct RefineWorkspace {
    int n,origin=-1;
    bool full_project=false;
    std::vector<Worker> workers;
    explicit RefineWorkspace(int size):n(size) {}
    virtual ~RefineWorkspace() { release(); }
    int release() {
        int saved=-1,error=0;
        if(OpenMXGpuGetDevice(&saved)!=OPENMX_GPU_SUCCESS) error=1;
        for(auto &w:workers) if(w.buffer) {
            if(OpenMXGpuSetDevice(w.ctx->device)!=OPENMX_GPU_SUCCESS ||
               OpenMXGpuStreamSynchronize(w.ctx->stream)!=OPENMX_GPU_SUCCESS || OpenMXGpuFree(w.buffer)!=OPENMX_GPU_SUCCESS) error=1;
            else w.buffer=nullptr;
        }
        if(saved>=0 && OpenMXGpuSetDevice(saved)!=OPENMX_GPU_SUCCESS) error=1;
        return error;
    }
    int synchronize() {
        int error=0;
        for(auto &w:workers)
            if(OpenMXGpuSetDevice(w.ctx->device)!=OPENMX_GPU_SUCCESS || OpenMXGpuStreamSynchronize(w.ctx->stream)!=OPENMX_GPU_SUCCESS) error=1;
        return error;
    }
    virtual int project(const void*,const void*,int,void*,void*)=0;
    virtual int update(const void*,void*)=0;
};
template<class T> struct ResidentRefine:RefineWorkspace {
    explicit ResidentRefine(int size):RefineWorkspace(size) {}
    bool allocate(int ngpu,const int *devices) {
        const size_t nn=size_t(n)*n,columns=(size_t(n)+ngpu-1)/ngpu;
        size_t bytes;
        if(columns*size_t(n)>INT_MAX || nn>SIZE_MAX/3 ||
           !product(2*nn+columns*n,sizeof(T),&bytes) ||
           bytes>budget("OPENMX_MAGMA_REFINE_RESIDENT_MB",0)) return false;
        std::vector<std::string> buses;
        for(int j=0;j<ngpu;++j) {
            char bus[64];
            if(OpenMXGpuSetDevice(devices[j])!=OPENMX_GPU_SUCCESS ||
               OpenMXGpuDeviceGetPCIBusId(bus,sizeof(bus),devices[j])!=OPENMX_GPU_SUCCESS ||
               std::find(buses.begin(),buses.end(),bus)!=buses.end() ||
               !available(bytes,budget("OPENMX_MAGMA_REFINE_RESERVE_MB",4096))) return false;
            buses.emplace_back(bus);
            Worker w;w.ctx=context(devices[j]);
            if(!w.ctx) return false;
            workers.push_back(w);
            if(OpenMXGpuMalloc(&workers.back().buffer,bytes)!=OPENMX_GPU_SUCCESS) return false;
            workers.back().bytes=bytes;
        }
        return true;
    }
    static T one() { if constexpr(std::is_same_v<T,double>) return 1;else return {1,0}; }
    static T zero_value() { if constexpr(std::is_same_v<T,double>) return 0;else return {0,0}; }
    int project(const void *ha,const void *hx,int end,void *hh,void *hg) override {
        full_project=false;
        if(!ha || !hx || !hh || !hg || end<1 || end>n) return 1;
        const size_t nn=size_t(n)*n;
        const T alpha=one(),beta=zero_value();
        /* All workers retain X. Their Y panels never leave device memory;
           A's storage is recycled for the projected H after A X completes. */
        for(size_t j=0;j<workers.size();++j) {
            auto &w=workers[j];
            w.col=int(size_t(end)*j/workers.size());
            w.n=int(size_t(end)*(j+1)/workers.size())-w.col;
            if(!w.n) continue;
            T *da=static_cast<T*>(w.buffer),*dx=da+nn,*dy=dx+nn;
            if(OpenMXGpuSetDevice(w.ctx->device)!=OPENMX_GPU_SUCCESS ||
               OpenMXBlasSetPointerMode(w.ctx->blas,OPENMX_BLAS_POINTER_HOST)!=OPENMX_BLAS_SUCCESS ||
               OpenMXGpuMemcpyAsync(da,ha,nn*sizeof(T),OPENMX_GPU_COPY_HOST_TO_DEVICE,w.ctx->stream)!=OPENMX_GPU_SUCCESS ||
               OpenMXGpuMemcpyAsync(dx,hx,nn*sizeof(T),OPENMX_GPU_COPY_HOST_TO_DEVICE,w.ctx->stream)!=OPENMX_GPU_SUCCESS ||
               local_gemm(w.ctx->blas,OPENMX_BLAS_OP_N,OPENMX_BLAS_OP_N,n,w.n,n,&alpha,da,n,
                          dx+size_t(w.col)*n,n,&beta,dy,n)!=OPENMX_BLAS_SUCCESS ||
               local_gemm(w.ctx->blas,OPENMX_BLAS_OP_C,OPENMX_BLAS_OP_N,end,w.n,n,&alpha,dx,n,dy,n,
                          &beta,da,end)!=OPENMX_BLAS_SUCCESS) return 1;
        }
        for(auto &w:workers) {
            if(!w.n) continue;
            T *da=static_cast<T*>(w.buffer),*dx=da+nn,*dy=dx+nn;
            if(OpenMXGpuSetDevice(w.ctx->device)!=OPENMX_GPU_SUCCESS ||
               OpenMXGpuMemcpyAsync(static_cast<T*>(hh)+size_t(w.col)*end,da,size_t(end)*w.n*sizeof(T),
                               OPENMX_GPU_COPY_DEVICE_TO_HOST,w.ctx->stream)!=OPENMX_GPU_SUCCESS ||
               local_gemm(w.ctx->blas,OPENMX_BLAS_OP_C,OPENMX_BLAS_OP_N,end,w.n,n,&alpha,dx,n,
                          dx+size_t(w.col)*n,n,&beta,dy,end)!=OPENMX_BLAS_SUCCESS) return 1;
        }
        for(auto &w:workers) {
            if(!w.n) continue;
            T *dy=static_cast<T*>(w.buffer)+2*nn;
            if(OpenMXGpuSetDevice(w.ctx->device)!=OPENMX_GPU_SUCCESS ||
               OpenMXGpuMemcpyAsync(static_cast<T*>(hg)+size_t(w.col)*end,dy,size_t(end)*w.n*sizeof(T),
                               OPENMX_GPU_COPY_DEVICE_TO_HOST,w.ctx->stream)!=OPENMX_GPU_SUCCESS) return 1;
        }
        const int error=synchronize();
        full_project=!error && end==n;
        return error;
    }
    int update(const void *he,void *hx) override {
        if(!full_project || !he || !hx) return 1;
        full_project=false;
        const size_t nn=size_t(n)*n;
        const T alpha=one(),beta=zero_value();
        for(auto &w:workers) {
            if(!w.n) continue;
            T *da=static_cast<T*>(w.buffer),*dx=da+nn,*de=dx+nn;
            const size_t count=size_t(n)*w.n,offset=size_t(n)*w.col;
            if(OpenMXGpuSetDevice(w.ctx->device)!=OPENMX_GPU_SUCCESS ||
               OpenMXGpuMemcpyAsync(de,static_cast<const T*>(he)+offset,count*sizeof(T),OPENMX_GPU_COPY_HOST_TO_DEVICE,w.ctx->stream)!=OPENMX_GPU_SUCCESS ||
               local_gemm(w.ctx->blas,OPENMX_BLAS_OP_N,OPENMX_BLAS_OP_N,n,w.n,n,&alpha,dx,n,de,n,
                          &beta,da,n)!=OPENMX_BLAS_SUCCESS) return 1;
            OpenMXBlasStatus status;
            if constexpr(std::is_same_v<T,double>)
                status=cublasDaxpy(w.ctx->blas,int(count),&alpha,da,1,dx+offset,1);
            else status=cublasZaxpy(w.ctx->blas,int(count),&alpha,da,1,dx+offset,1);
            if(status!=OPENMX_BLAS_SUCCESS) return 1;
        }
        for(auto &w:workers) {
            if(!w.n) continue;
            T *dx=static_cast<T*>(w.buffer)+nn;
            const size_t count=size_t(n)*w.n,offset=size_t(n)*w.col;
            if(OpenMXGpuSetDevice(w.ctx->device)!=OPENMX_GPU_SUCCESS ||
               OpenMXGpuMemcpyAsync(static_cast<T*>(hx)+offset,dx+offset,count*sizeof(T),OPENMX_GPU_COPY_DEVICE_TO_HOST,w.ctx->stream)!=OPENMX_GPU_SUCCESS) return 1;
        }
        return synchronize();
    }
};
struct RestoreDevice {
    int previous=-1;
    RestoreDevice() { OpenMXGpuGetDevice(&previous); }
    ~RestoreDevice() { if(previous>=0) OpenMXGpuSetDevice(previous); }
};
}
extern "C" void *openmx_mgpuRefineCreate(int ngpu,const int *devices,int n,int is_complex) {
    RestoreDevice restore;
    if(restore.previous<0 || ngpu<2 || ngpu>64 || !devices || n<ngpu) return nullptr;
    try {
        if(is_complex) {
            auto p=std::make_unique<ResidentRefine<OpenMXGpuComplex>>(n);
            if(!p->allocate(ngpu,devices)) return nullptr;
            return p.release();
        }
        auto p=std::make_unique<ResidentRefine<double>>(n);
        if(!p->allocate(ngpu,devices)) return nullptr;
        return p.release();
    } catch(const std::bad_alloc&) {return nullptr;}
}
extern "C" int openmx_mgpuRefineProject(void *p,const void *a,const void *x,int end,void *h,void *g) {
    RestoreDevice restore;
    if(!p || restore.previous<0) return 1;
    auto *workspace=static_cast<RefineWorkspace*>(p);
    int error=1;
    try { error=workspace->project(a,x,end,h,g); } catch(...) {}
    /* On failure, transfers may still refer to the private host vectors.
       Complete them before the adapter can destroy those vectors. */
    if(workspace->synchronize()) error=1;
    if(error) workspace->full_project=false;
    if(OpenMXGpuSetDevice(restore.previous)!=OPENMX_GPU_SUCCESS) error=1;
    return error;
}
extern "C" int openmx_mgpuRefineUpdate(void *p,const void *e,void *x) {
    RestoreDevice restore;
    if(!p || restore.previous<0) return 1;
    auto *workspace=static_cast<RefineWorkspace*>(p);
    int error=1;
    try { error=workspace->update(e,x); } catch(...) {}
    if(workspace->synchronize()) error=1;
    if(error) workspace->full_project=false;
    if(OpenMXGpuSetDevice(restore.previous)!=OPENMX_GPU_SUCCESS) error=1;
    return error;
}
extern "C" int openmx_mgpuRefineRelease(void *p) {
    if(!p) return 0;
    auto *workspace=static_cast<RefineWorkspace*>(p);
    const int error=workspace->release();
    delete workspace;
    return error;
}
#endif

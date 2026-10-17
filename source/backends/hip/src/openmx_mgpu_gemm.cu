#include "openmx_mgpu_gemm.h"
#include "hip_runtime_compat.h"

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

extern "C" __attribute__((weak)) hipError_t OpenMX_GpuMemGetInfo(size_t *, size_t *);
extern "C" __attribute__((weak)) hipblasStatus_t openmx_gemmul8Dgemm(
    hipblasHandle_t,hipblasOperation_t,hipblasOperation_t,int,int,int,const double *,
    const double *,int,const double *,int,const double *,double *,int);
extern "C" __attribute__((weak)) hipblasStatus_t openmx_gemmul8Zgemm(
    hipblasHandle_t,hipblasOperation_t,hipblasOperation_t,int,int,int,const hipDoubleComplex *,
    const hipDoubleComplex *,int,const hipDoubleComplex *,int,const hipDoubleComplex *,hipDoubleComplex *,int);

namespace {
constexpr size_t MiB = size_t(1024)*1024;
thread_local bool in_worker = false;
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
hipError_t meminfo(size_t *free_bytes,size_t *total)
{
    return OpenMX_GpuMemGetInfo ? OpenMX_GpuMemGetInfo(free_bytes,total) : hipMemGetInfo(free_bytes,total);
}
bool available(size_t bytes,size_t reserve)
{
    size_t free_bytes=0,total=0;
    return meminfo(&free_bytes,&total)==hipSuccess && bytes<=free_bytes && reserve<=free_bytes-bytes;
}

/* Handles and streams survive calls; large operand buffers do not. Stable
 * stream identities also bound the existing bridge's workspace cache. */
struct Context {
    int device;
    hipStream_t stream=nullptr;
    hipblasHandle_t blas=nullptr;
    explicit Context(int id):device(id) {}
    ~Context() {
        int saved=-1;hipGetDevice(&saved);hipSetDevice(device);
        if(stream) hipStreamSynchronize(stream);
        if(blas) hipblasDestroy(blas);
        if(stream) hipStreamDestroy(stream);
        if(saved>=0) hipSetDevice(saved);
    }
};
thread_local std::vector<std::unique_ptr<Context>> contexts;
Context *context(int device)
{
    for(auto &c:contexts) if(c->device==device) return c.get();
    if(hipSetDevice(device)!=hipSuccess) return nullptr;
    auto c=std::make_unique<Context>(device);
    if(hipStreamCreateWithFlags(&c->stream,hipStreamNonBlocking)!=hipSuccess ||
       hipblasCreate(&c->blas)!=HIPBLAS_STATUS_SUCCESS ||
       hipblasSetStream(c->blas,c->stream)!=HIPBLAS_STATUS_SUCCESS) return nullptr;
    contexts.push_back(std::move(c));return contexts.back().get();
}
struct Worker {
    Context *ctx=nullptr;
    void *buffer=nullptr,*host=nullptr;
    size_t bytes=0;
    bool peer=false,used=false;
    int cached_row=-1,cached_m=-1,cached_col=-1,cached_n=-1;
    int row=0,col=0,m=0,n=0;
};
struct Operation {
    int origin=-1;
    bool host_matrices=false,host_result=false;
    hipStream_t copy_stream=nullptr;
    hipEvent_t ready=nullptr;
    void *result=nullptr;
    std::vector<Worker> workers;
    bool clean=true;
    ~Operation() { cleanup(); }
    void cleanup() {
        if(origin<0) return;
        for(auto &w:workers) {
            if(w.ctx && hipSetDevice(w.ctx->device)==hipSuccess) {
                if(hipStreamSynchronize(w.ctx->stream)!=hipSuccess) clean=false;
            }
        }
        if(hipSetDevice(origin)!=hipSuccess) clean=false;
        if(copy_stream) {
            if(hipStreamSynchronize(copy_stream)!=hipSuccess) clean=false;
            if(hipStreamDestroy(copy_stream)!=hipSuccess) clean=false;
            copy_stream=nullptr;
        }
        for(auto &w:workers) {
            if(w.buffer && (hipSetDevice(w.ctx->device)!=hipSuccess || hipFree(w.buffer)!=hipSuccess)) clean=false;
            if(w.host && hipFreeHost(w.host)!=hipSuccess) clean=false;
            w.buffer=w.host=nullptr;
        }
        if(hipSetDevice(origin)!=hipSuccess) clean=false;
        if(ready) { if(hipEventDestroy(ready)!=hipSuccess) clean=false;ready=nullptr; }
        if(result) {
            if(host_result) std::free(result);
            else if(hipFree(result)!=hipSuccess) clean=false;
            result=nullptr;
        }
    }
};

/* A pitched peer copy retains storage orientation. HIP's copy engine moves
 * raw A/B submatrices; hipBLAS applies the requested transpose/conjugation. */
hipError_t peer_copy(void *dst,int dstdev,size_t dpitch,const void *src,int srcdev,size_t spitch,
                      size_t width,size_t height,hipStream_t stream)
{
    if(srcdev==dstdev) return hipMemcpy2DAsync(dst,dpitch,src,spitch,width,height,hipMemcpyDeviceToDevice,stream);
    hipMemcpy3DPeerParms p={};
    p.srcDevice=srcdev;p.dstDevice=dstdev;
    p.srcPtr=make_hipPitchedPtr(const_cast<void *>(src),spitch,width,height);
    p.dstPtr=make_hipPitchedPtr(dst,dpitch,width,height);
    p.extent=make_hipExtent(width,height,1);
    return hipMemcpy3DPeerAsync(&p,stream);
}

bool input_copy(Operation &op,Worker &w,void *dst,const void *src,size_t spitch,
                size_t width,size_t height,void *host)
{
    if(hipSetDevice(w.ctx->device)!=hipSuccess) return false;
    if(op.host_matrices) {
        for(size_t j=0;j<height;++j)
            std::memcpy(static_cast<char *>(host)+j*width,static_cast<const char *>(src)+j*spitch,width);
        if(hipMemcpyAsync(dst,host,width*height,hipMemcpyHostToDevice,w.ctx->stream)!=hipSuccess) return false;
        ++report.host_transfers;return true;
    }
    if(w.peer) {
        const hipError_t err=peer_copy(dst,w.ctx->device,width,src,op.origin,spitch,width,height,w.ctx->stream);
        if(err==hipSuccess) {if(w.ctx->device!=op.origin) ++report.peer_transfers;return true;}
        (void)hipGetLastError();
        return false; // no user output was touched: original backend may retry
    }
    if(hipSetDevice(op.origin)!=hipSuccess ||
       hipMemcpy2DAsync(host,width,src,spitch,width,height,hipMemcpyDeviceToHost,op.copy_stream)!=hipSuccess ||
       hipStreamSynchronize(op.copy_stream)!=hipSuccess ||
       hipSetDevice(w.ctx->device)!=hipSuccess ||
       hipMemcpyAsync(dst,host,width*height,hipMemcpyHostToDevice,w.ctx->stream)!=hipSuccess) return false;
    ++report.host_transfers;return true;
}
bool output_copy(Operation &op,Worker &w,void *dst,size_t dpitch,const void *src,
                 size_t width,size_t height,void *host)
{
    if(hipSetDevice(w.ctx->device)!=hipSuccess) return false;
    if(op.host_result) {
        if(hipMemcpyAsync(host,src,width*height,hipMemcpyDeviceToHost,w.ctx->stream)!=hipSuccess ||
           hipStreamSynchronize(w.ctx->stream)!=hipSuccess) return false;
        for(size_t j=0;j<height;++j)
            std::memcpy(static_cast<char *>(dst)+j*dpitch,static_cast<const char *>(host)+j*width,width);
        ++report.host_transfers;return true;
    }
    if(w.peer) {
        const hipError_t err=peer_copy(dst,op.origin,dpitch,src,w.ctx->device,width,width,height,w.ctx->stream);
        if(err==hipSuccess) {if(w.ctx->device!=op.origin) ++report.peer_transfers;return true;}
        (void)hipGetLastError();return false;
    }
    if(hipMemcpyAsync(host,src,width*height,hipMemcpyDeviceToHost,w.ctx->stream)!=hipSuccess ||
       hipStreamSynchronize(w.ctx->stream)!=hipSuccess ||
       hipSetDevice(op.origin)!=hipSuccess ||
       hipMemcpy2DAsync(dst,dpitch,host,width,width,height,hipMemcpyHostToDevice,op.copy_stream)!=hipSuccess ||
       hipStreamSynchronize(op.copy_stream)!=hipSuccess) return false;
    ++report.host_transfers;return true;
}
template<class T> bool zero(T v) {
    if constexpr(std::is_same_v<T,double>) return v==0;
    else return v.x==0 && v.y==0;
}
template<class T> hipblasStatus_t local_gemm(hipblasHandle_t h,hipblasOperation_t ta,hipblasOperation_t tb,
    int m,int n,int k,const T *alpha,const T *a,int lda,const T *b,int ldb,const T *beta,T *c,int ldc)
{
    const char *backend=std::getenv("OPENMX_MGPU_GEMM_BACKEND");
    const bool bridge=!(backend && std::strcmp(backend,"native")==0);
    struct Guard { Guard(){in_worker=true;} ~Guard(){in_worker=false;} } guard;
    if constexpr(std::is_same_v<T,double>) {
        if(bridge && openmx_gemmul8Dgemm) {
            ++report.bridge_calls;return openmx_gemmul8Dgemm(h,ta,tb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc);
        }
        return hipblasDgemm(h,ta,tb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc);
    } else {
        if(bridge && openmx_gemmul8Zgemm) {
            ++report.bridge_calls;return openmx_gemmul8Zgemm(h,ta,tb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc);
        }
        return hipblasZgemm(h,ta,tb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc);
    }
}
bool owned_pointer(const void *p,int device)
{
    hipPointerAttribute_t attr={};
    if(hipPointerGetAttributes(&attr,p)!=hipSuccess) {(void)hipGetLastError();return false;}
    return attr.type==hipMemoryTypeDevice && attr.device==device;
}
bool valid_op(hipblasOperation_t op) { return op==HIPBLAS_OP_N || op==HIPBLAS_OP_T || op==HIPBLAS_OP_C; }
bool enable_peer_pair(int first,int second)
{
    // Peer enablement belongs to HIP contexts, not handles or host threads.
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
        if(hipSetDevice(from)!=hipSuccess) return false;
        const hipError_t e=hipDeviceEnablePeerAccess(to,0);
        if(e!=hipSuccess) {
            (void)hipGetLastError();
            if(e!=hipErrorPeerAccessAlreadyEnabled) {enabled[key]=false;return false;}
        }
        enabled[key]=true;
    }
    // Peer mappings are process/context state; keep enabled mappings rather
    // than disabling access another HIP library may subsequently use.
    return true;
}

template<class T> int gemm(hipblasHandle_t handle,int ngpu,const int *devices,hipblasOperation_t ta,
    hipblasOperation_t tb,int m,int n,int k,const T *alpha,const T *a,int lda,const T *b,int ldb,
    const T *beta,T *c,int ldc,hipblasStatus_t *status,bool host_matrices=false)
{
    const auto start=std::chrono::steady_clock::now();
    if(in_worker) {if(status) *status=HIPBLAS_STATUS_SUCCESS;return 0;}
    report={};report.requested_gpus=ngpu;report.reason="not eligible";
    if(!status) return 0;
    *status=HIPBLAS_STATUS_SUCCESS;
    if(ngpu<2 || ngpu>64 || !devices || m<=0 || n<=0 || k<=0) return 0;
    if(!valid_op(ta) || !valid_op(tb) || lda<(ta==HIPBLAS_OP_N?m:k) || ldb<(tb==HIPBLAS_OP_N?k:n) ||
       ldc<m || !alpha || !beta || !a || !b || !c) {
        *status=HIPBLAS_STATUS_INVALID_VALUE;report.reason="invalid GEMM arguments";return 1;
    }
    Operation op;
    op.host_matrices=host_matrices;
    if(hipGetDevice(&op.origin)!=hipSuccess) {report.reason="current device unavailable";return 0;}
    auto finish=[&](int handled,const char *reason) {
        op.cleanup();
        if(!op.clean) {handled=1;*status=HIPBLAS_STATUS_EXECUTION_FAILED;reason="HIP cleanup/synchronization failure";}
        if(hipSetDevice(op.origin)!=hipSuccess) {handled=1;*status=HIPBLAS_STATUS_EXECUTION_FAILED;reason="restore device failure";}
        report.reason=reason;report.seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        if(!handled) (void)hipGetLastError();
        op.origin=-1;return handled;
    };
    if(!host_matrices && (!owned_pointer(a,op.origin) || !owned_pointer(b,op.origin) || !owned_pointer(c,op.origin)))
        return finish(0,"operands are not on the current device");
    hipStream_t original=nullptr;
    hipblasPointerMode_t pointer_mode=HIPBLAS_POINTER_MODE_HOST;
    hipblasMath_t math_mode=HIPBLAS_DEFAULT_MATH;
    hipblasAtomicsMode_t atomics_mode=HIPBLAS_ATOMICS_NOT_ALLOWED;
    if(!host_matrices && (hipblasGetStream(handle,&original)!=HIPBLAS_STATUS_SUCCESS ||
       hipblasGetPointerMode(handle,&pointer_mode)!=HIPBLAS_STATUS_SUCCESS ||
       hipblasGetMathMode(handle,&math_mode)!=HIPBLAS_STATUS_SUCCESS ||
       hipblasGetAtomicsMode(handle,&atomics_mode)!=HIPBLAS_STATUS_SUCCESS)) return finish(0,"handle query failed");
    hipStreamCaptureStatus capture;
    if(hipStreamIsCapturing(original,&capture)!=hipSuccess || capture!=hipStreamCaptureStatusNone)
        return finish(0,"HIP graph capture is unsupported");
    T host_alpha,host_beta;
    if(pointer_mode==HIPBLAS_POINTER_MODE_DEVICE) {
        if(hipMemcpyAsync(&host_alpha,alpha,sizeof(T),hipMemcpyDeviceToHost,original)!=hipSuccess ||
           hipMemcpyAsync(&host_beta,beta,sizeof(T),hipMemcpyDeviceToHost,original)!=hipSuccess ||
           hipStreamSynchronize(original)!=hipSuccess) return finish(0,"device scalar copy failed");
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
    } else if(hipMalloc(&op.result,result_bytes)!=hipSuccess) {
        (void)hipGetLastError();
        op.host_result=true;
        if(result_bytes>budget("OPENMX_MGPU_GEMM_HOST_RESULT_MB",65536))
            return finish(0,"host transaction budget exceeded");
        op.result=std::malloc(result_bytes);
        tile_budget=std::min(tile_budget,host_budget);
        if(!op.result) return finish(0,"host transaction allocation failed");
    }
    int count=0;if(hipGetDeviceCount(&count)!=hipSuccess) return finish(0,"device count unavailable");
    std::vector<std::string> buses;
    for(int i=0;i<ngpu;++i) {
        char bus[64]={};
        if(devices[i]<0 || devices[i]>=count || hipDeviceGetPCIBusId(bus,sizeof(bus),devices[i])!=hipSuccess)
            return finish(0,"invalid device list");
        if(std::find(buses.begin(),buses.end(),bus)!=buses.end()) return finish(0,"devices share one physical GPU");
        buses.emplace_back(bus);
        Worker w;w.ctx=context(devices[i]);if(!w.ctx) return finish(0,"worker context unavailable");
        int can=devices[i]==op.origin,reverse=can;
        if(!host_matrices && !can && (hipDeviceCanAccessPeer(&can,devices[i],op.origin)!=hipSuccess ||
                    hipDeviceCanAccessPeer(&reverse,op.origin,devices[i])!=hipSuccess)) return finish(0,"peer query failed");
        w.peer=!host_matrices && (devices[i]==op.origin || (can && reverse && !option("OPENMX_MGPU_GEMM_FORCE_HOST") &&
                                       enable_peer_pair(devices[i],op.origin)));
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
    if(hipSetDevice(op.origin)!=hipSuccess) return finish(0,"origin device selection failed");
    if(hipStreamCreateWithFlags(&op.copy_stream,hipStreamNonBlocking)!=hipSuccess ||
       hipEventCreateWithFlags(&op.ready,hipEventDisableTiming)!=hipSuccess ||
       hipEventRecord(op.ready,original)!=hipSuccess ||
       hipStreamWaitEvent(op.copy_stream,op.ready,0)!=hipSuccess) return finish(0,"input dependency setup failed");
    for(auto &w:op.workers) {
        if(hipSetDevice(w.ctx->device)!=hipSuccess || !available(worker_bytes,reserve) ||
           hipMalloc(&w.buffer,worker_bytes)!=hipSuccess) return finish(0,"worker allocation failed");
        w.bytes=worker_bytes;
        if((!w.peer || op.host_result) && hipHostAlloc(&w.host,worker_bytes,hipHostAllocPortable)!=hipSuccess)
            return finish(0,"pinned staging allocation failed");
        if(hipStreamWaitEvent(w.ctx->stream,op.ready,0)!=hipSuccess ||
           hipblasSetPointerMode(w.ctx->blas,HIPBLAS_POINTER_MODE_HOST)!=HIPBLAS_STATUS_SUCCESS ||
           hipblasSetMathMode(w.ctx->blas,math_mode)!=HIPBLAS_STATUS_SUCCESS ||
           hipblasSetAtomicsMode(w.ctx->blas,atomics_mode)!=HIPBLAS_STATUS_SUCCESS)
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
                const T *src=a+(ta==HIPBLAS_OP_N?size_t(w.row):size_t(w.row)*lda);
                if(!input_copy(op,w,wa,src,size_t(lda)*sizeof(T),size_t(ta==HIPBLAS_OP_N?w.m:k)*sizeof(T),
                               ta==HIPBLAS_OP_N?k:w.m,ha)) return finish(0,"operand A transfer failed");
                w.cached_row=w.row;w.cached_m=w.m;
            }
            if(w.cached_col!=w.col || w.cached_n!=w.n) {
                const T *src=b+(tb==HIPBLAS_OP_N?size_t(w.col)*ldb:size_t(w.col));
                if(!input_copy(op,w,wb,src,size_t(ldb)*sizeof(T),size_t(tb==HIPBLAS_OP_N?k:w.n)*sizeof(T),
                               tb==HIPBLAS_OP_N?w.n:k,hb)) return finish(0,"operand B transfer failed");
                w.cached_col=w.col;w.cached_n=w.n;
            }
            if(!zero(host_beta) && !input_copy(op,w,wc,c+size_t(w.col)*ldc+w.row,size_t(ldc)*sizeof(T),
                                              size_t(w.m)*sizeof(T),w.n,hc)) return finish(0,"input C transfer failed");
            if(hipSetDevice(w.ctx->device)!=hipSuccess) return finish(0,"worker device selection failed");
            const hipblasStatus_t local=local_gemm(w.ctx->blas,ta,tb,w.m,w.n,k,&host_alpha,wa,
                ta==HIPBLAS_OP_N?w.m:k,wb,tb==HIPBLAS_OP_N?k:w.n,&host_beta,wc,w.m);
            if(local!=HIPBLAS_STATUS_SUCCESS) return finish(0,"worker GEMM failed before commit");
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
            if(hipSetDevice(w.ctx->device)!=hipSuccess || hipStreamSynchronize(w.ctx->stream)!=hipSuccess)
                return finish(0,"worker synchronization failed before commit");
        }
    }
    if(hipSetDevice(op.origin)!=hipSuccess || hipStreamSynchronize(op.copy_stream)!=hipSuccess)
        return finish(0,"gather synchronization failed before commit");
    // The only write to caller C. Preserve all padding and make beta retries
    // unambiguous by treating every commit error as handled, never declined.
    if(host_matrices) {
        for(int j=0;j<n;++j)
            std::memcpy(c+size_t(j)*ldc,static_cast<T *>(op.result)+size_t(j)*m,size_t(m)*sizeof(T));
    } else if(hipMemcpy2DAsync(c,size_t(ldc)*sizeof(T),op.result,size_t(m)*sizeof(T),size_t(m)*sizeof(T),n,
                         op.host_result?hipMemcpyHostToDevice:hipMemcpyDeviceToDevice,original)!=hipSuccess || hipStreamSynchronize(original)!=hipSuccess) {
        *status=HIPBLAS_STATUS_EXECUTION_FAILED;return finish(1,"output commit failed");
    }
    return finish(1,"success");
}
} // namespace

extern "C" int openmx_mgpuGemmInWorker(void) { return in_worker?1:0; }
extern "C" void openmx_mgpuGemmLastReport(openmx_mgpu_gemm_report *out) { if(out) *out=report; }
extern "C" void openmx_mgpuGemmRelease(void) { if(!in_worker) contexts.clear(); }
extern "C" int openmx_mgpuDgemmHostTry(int ngpu,const int *devices,hipblasOperation_t ta,
    hipblasOperation_t tb,int m,int n,int k,const double *alpha,const double *a,int lda,const double *b,int ldb,
    const double *beta,double *c,int ldc,hipblasStatus_t *status)
{
    try {return gemm(nullptr,ngpu,devices,ta,tb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc,status,true);}
    catch(const std::bad_alloc &) {report.reason="host allocation failed";if(status)*status=HIPBLAS_STATUS_ALLOC_FAILED;return 0;}
}
extern "C" int openmx_mgpuZgemmHostTry(int ngpu,const int *devices,hipblasOperation_t ta,
    hipblasOperation_t tb,int m,int n,int k,const hipDoubleComplex *alpha,const hipDoubleComplex *a,int lda,
    const hipDoubleComplex *b,int ldb,const hipDoubleComplex *beta,hipDoubleComplex *c,int ldc,hipblasStatus_t *status)
{
    try {return gemm(nullptr,ngpu,devices,ta,tb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc,status,true);}
    catch(const std::bad_alloc &) {report.reason="host allocation failed";if(status)*status=HIPBLAS_STATUS_ALLOC_FAILED;return 0;}
}
extern "C" int openmx_mgpuDgemmTry(hipblasHandle_t h,int ngpu,const int *devices,hipblasOperation_t ta,
    hipblasOperation_t tb,int m,int n,int k,const double *alpha,const double *a,int lda,const double *b,int ldb,
    const double *beta,double *c,int ldc,hipblasStatus_t *status)
{
    try {return gemm(h,ngpu,devices,ta,tb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc,status);}
    catch(const std::bad_alloc &) {report.reason="host metadata allocation failed";if(status)*status=HIPBLAS_STATUS_SUCCESS;return 0;}
}
extern "C" int openmx_mgpuZgemmTry(hipblasHandle_t h,int ngpu,const int *devices,hipblasOperation_t ta,
    hipblasOperation_t tb,int m,int n,int k,const hipDoubleComplex *alpha,const hipDoubleComplex *a,int lda,
    const hipDoubleComplex *b,int ldb,const hipDoubleComplex *beta,hipDoubleComplex *c,int ldc,hipblasStatus_t *status)
{
    try {return gemm(h,ngpu,devices,ta,tb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc,status);}
    catch(const std::bad_alloc &) {report.reason="host metadata allocation failed";if(status)*status=HIPBLAS_STATUS_SUCCESS;return 0;}
}

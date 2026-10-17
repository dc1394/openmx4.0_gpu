/* GPU-resident GEMM/forward pair: one-GPU hipBLAS/bridge versus the multi-GPU core.
 * Wall timing includes scratch allocation, transfers, GEMMs and commit. */
#include "openmx_mgpu_gemm.h"
#include "hip_runtime_compat.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <vector>
extern "C" {int level_stdout=0;}
#ifdef OPENMX_MGPU_BENCH_BRIDGE
extern "C" hipblasStatus_t openmx_gemmul8Dgemm(hipblasHandle_t,hipblasOperation_t,hipblasOperation_t,int,int,int,const double*,const double*,int,const double*,int,const double*,double*,int);
extern "C" hipblasStatus_t openmx_gemmul8DgemmUnblocked(hipblasHandle_t,hipblasOperation_t,hipblasOperation_t,int,int,int,const double*,const double*,int,const double*,int,const double*,double*,int);
extern "C" hipblasStatus_t openmx_gemmul8Zgemm(hipblasHandle_t,hipblasOperation_t,hipblasOperation_t,int,int,int,const hipDoubleComplex*,const hipDoubleComplex*,int,const hipDoubleComplex*,int,const hipDoubleComplex*,hipDoubleComplex*,int);
extern "C" hipblasStatus_t openmx_gemmul8ZgemmUnblocked(hipblasHandle_t,hipblasOperation_t,hipblasOperation_t,int,int,int,const hipDoubleComplex*,const hipDoubleComplex*,int,const hipDoubleComplex*,int,const hipDoubleComplex*,hipDoubleComplex*,int);
extern "C" void openmx_gemmul8TrimWorkspaces(void);
extern "C" void openmx_gemmul8ReleaseWorkspaces(void);
#endif
static void check(hipError_t e){if(e!=hipSuccess){std::fprintf(stderr,"HIP %s\n",hipGetErrorString(e));std::exit(2);}}
static void check(hipblasStatus_t e){if(e!=HIPBLAS_STATUS_SUCCESS){std::fprintf(stderr,"hipBLAS %d\n",int(e));std::exit(2);}}
template<class T> T value(double r,double i=0){if constexpr(std::is_same_v<T,double>)return r;else return make_hipDoubleComplex(r,i);}
template<class T> std::complex<long double> wide(T x){if constexpr(std::is_same_v<T,double>)return {x,0};else return {x.x,x.y};}
template<class T> void native(hipblasHandle_t h,hipblasOperation_t ta,int m,int n,int k,const T *a,int lda,
                              const T *b,int ldb,T *c,int ldc){
    const T one=value<T>(1),zero=value<T>(0);
#ifdef OPENMX_MGPU_BENCH_BRIDGE
    if constexpr(std::is_same_v<T,double>)check(openmx_gemmul8Dgemm(h,ta,HIPBLAS_OP_N,m,n,k,&one,a,lda,b,ldb,&zero,c,ldc));
    else check(openmx_gemmul8Zgemm(h,ta,HIPBLAS_OP_N,m,n,k,&one,a,lda,b,ldb,&zero,c,ldc));
#else
    if constexpr(std::is_same_v<T,double>)check(hipblasDgemm(h,ta,HIPBLAS_OP_N,m,n,k,&one,a,lda,b,ldb,&zero,c,ldc));
    else check(hipblasZgemm(h,ta,HIPBLAS_OP_N,m,n,k,&one,a,lda,b,ldb,&zero,c,ldc));
#endif
}
template<class T> void multi(hipblasHandle_t h,const std::vector<int> &devices,hipblasOperation_t ta,int m,int n,int k,
                             const T *a,int lda,const T *b,int ldb,T *c,int ldc){
    const T one=value<T>(1),zero=value<T>(0);hipblasStatus_t status;int handled;
    if constexpr(std::is_same_v<T,double>)handled=openmx_mgpuDgemmTry(h,int(devices.size()),devices.data(),ta,HIPBLAS_OP_N,m,n,k,&one,a,lda,b,ldb,&zero,c,ldc,&status);
    else handled=openmx_mgpuZgemmTry(h,int(devices.size()),devices.data(),ta,HIPBLAS_OP_N,m,n,k,&one,a,lda,b,ldb,&zero,c,ldc,&status);
    if(!handled){openmx_mgpu_gemm_report r;openmx_mgpuGemmLastReport(&r);std::fprintf(stderr,"DECLINED: %s\n",r.reason);std::exit(4);}check(status);
}
template<class T> void run(int ngpu,int m,int n,int k,int iters,bool forward){
    std::vector<int> devices;for(int i=0;i<ngpu;++i)devices.push_back(i);
    std::vector<T> a(size_t(m)*k),b(size_t(k)*n),reference(size_t(m)*n),out(reference.size());
    for(size_t i=0;i<a.size();++i)a[i]=value<T>(std::sin(i*.17),std::cos(i*.09)*.3);
    for(size_t i=0;i<b.size();++i)b[i]=value<T>(std::cos(i*.13),std::sin(i*.07)*.2);
    T *da,*db,*dc,*tmp=nullptr;check(hipMalloc(&da,a.size()*sizeof(T)));check(hipMalloc(&db,b.size()*sizeof(T)));check(hipMalloc(&dc,out.size()*sizeof(T)));
    if(forward)check(hipMalloc(&tmp,out.size()*sizeof(T)));
    check(hipMemcpy(da,a.data(),a.size()*sizeof(T),hipMemcpyHostToDevice));check(hipMemcpy(db,b.data(),b.size()*sizeof(T),hipMemcpyHostToDevice));
    hipblasHandle_t h;check(hipblasCreate(&h));hipStream_t stream;check(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking));check(hipblasSetStream(h,stream));
    auto product=[&](bool mgpu){

        if(mgpu)multi(h,devices,HIPBLAS_OP_N,m,n,k,da,m,db,k,forward?tmp:dc,m);
        else native(h,HIPBLAS_OP_N,m,n,k,da,m,db,k,forward?tmp:dc,m);
        if(forward){const hipblasOperation_t adjoint=std::is_same_v<T,double> ? HIPBLAS_OP_T : HIPBLAS_OP_C;
            if(mgpu)multi(h,devices,adjoint,n,n,n,db,n,tmp,n,dc,n);else native(h,adjoint,n,n,n,db,n,tmp,n,dc,n);}
    };
    for(bool mgpu:{false,true}){
        product(mgpu);check(hipStreamSynchronize(stream));
        const auto start=std::chrono::steady_clock::now();for(int i=0;i<iters;++i)product(mgpu);check(hipStreamSynchronize(stream));
        const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()/iters;
        check(hipMemcpy(out.data(),dc,out.size()*sizeof(T),hipMemcpyDeviceToHost));if(!mgpu)reference=out;
        long double error=0,scale=0;for(size_t i=0;i<out.size();++i){error=std::max(error,std::abs(wide(out[i])-wide(reference[i])));scale=std::max(scale,std::abs(wide(reference[i])));}
        const double rel=double(error/std::max(1.L,scale));
        openmx_mgpu_gemm_report r={};if(mgpu)openmx_mgpuGemmLastReport(&r);
        const char *type=std::is_same_v<T,double> ? "D" : "Z";
#ifdef OPENMX_MGPU_BENCH_BRIDGE
        const char *baseline="single-bridge";
#else
        const char *baseline="native";
#endif
        std::printf("%s,%d,%d,%d,%s,%s,%d,%d,%.6f,%.3e,%d,%llu,%llu,%zu,%zu\n",type,m,n,k,
            forward?"forward":"general",mgpu?"multi":baseline,mgpu?ngpu:1,iters,ms,rel,r.active_gpus,r.peer_transfers,r.host_transfers,r.transaction_bytes,r.worker_bytes);std::fflush(stdout);
        if(!std::isfinite(rel)||rel>1e-11||(mgpu&&r.active_gpus!=ngpu)){std::fprintf(stderr,"accuracy/device-count check failed\n");std::exit(3);}
    }
#ifdef OPENMX_MGPU_BENCH_BRIDGE
    openmx_gemmul8ReleaseWorkspaces();
#endif
    check(hipFree(da));check(hipFree(db));check(hipFree(dc));if(tmp)check(hipFree(tmp));check(hipblasDestroy(h));check(hipStreamDestroy(stream));openmx_mgpuGemmRelease();
}
int main(int argc,char **argv){
    int ngpu=2,m=4096,n=4096,k=4096,iters=5;bool complex=false,forward=false;
    for(int i=1;i<argc;++i){
        if(!std::strcmp(argv[i],"--complex"))complex=true;
        else if(!std::strcmp(argv[i],"--forward"))forward=true;
        else if(i+1<argc){const char *key=argv[i++];const int v=std::atoi(argv[i]);
            if(!std::strcmp(key,"--ngpu"))ngpu=v;else if(!std::strcmp(key,"--m"))m=v;else if(!std::strcmp(key,"--n"))n=v;else if(!std::strcmp(key,"--k"))k=v;else if(!std::strcmp(key,"--iterations"))iters=v;else return 2;
        }else return 2;
    }
    if(m<1||n<1||k<1||ngpu<2||iters<1||(forward&&(m!=n||n!=k)))return 2;
    int count=0;check(hipGetDeviceCount(&count));if(count<ngpu){std::fprintf(stderr,"need %d GPUs\n",ngpu);return 77;}check(hipSetDevice(0));
    std::puts("type,m,n,k,path,backend,gpus,iterations,wall_ms,relative_to_single_gpu,active_gpus,peer_copies,host_copies,transaction_bytes,worker_bytes");
    if(complex)run<hipDoubleComplex>(ngpu,m,n,k,iters,forward);else run<double>(ngpu,m,n,k,iters,forward);
}

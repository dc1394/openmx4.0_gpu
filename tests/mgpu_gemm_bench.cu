/* GPU-resident GEMM/forward pair: one-GPU cuBLAS/bridge versus the multi-GPU core.
 * Wall timing includes scratch allocation, transfers, GEMMs and commit. */
#include "openmx_mgpu_gemm.h"
#include <cuda_runtime.h>
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
extern "C" cublasStatus_t openmx_gemmul8Dgemm(cublasHandle_t,cublasOperation_t,cublasOperation_t,int,int,int,const double*,const double*,int,const double*,int,const double*,double*,int);
extern "C" cublasStatus_t openmx_gemmul8Zgemm(cublasHandle_t,cublasOperation_t,cublasOperation_t,int,int,int,const cuDoubleComplex*,const cuDoubleComplex*,int,const cuDoubleComplex*,int,const cuDoubleComplex*,cuDoubleComplex*,int);
extern "C" cublasStatus_t openmx_gemmul8DgemmFixed(cublasHandle_t,int,cublasOperation_t,int,int,int,const double*,int,const double*,int,double*,int,int,unsigned long long);
extern "C" cublasStatus_t openmx_gemmul8ZgemmFixed(cublasHandle_t,int,cublasOperation_t,int,int,int,const cuDoubleComplex*,int,const cuDoubleComplex*,int,cuDoubleComplex*,int,int,unsigned long long);
extern "C" void openmx_gemmul8SetForwardStage(int,int,int,int,int);
extern "C" void openmx_gemmul8TrimWorkspaces(void);
extern "C" void openmx_gemmul8ReleaseWorkspaces(void);
#endif
static void check(cudaError_t e){if(e!=cudaSuccess){std::fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e));std::exit(2);}}
static void check(cublasStatus_t e){if(e!=CUBLAS_STATUS_SUCCESS){std::fprintf(stderr,"cuBLAS %d\n",int(e));std::exit(2);}}
template<class T> T value(double r,double i=0){if constexpr(std::is_same_v<T,double>)return r;else return make_cuDoubleComplex(r,i);}
template<class T> std::complex<long double> wide(T x){if constexpr(std::is_same_v<T,double>)return {x,0};else return {x.x,x.y};}
template<class T> void native(cublasHandle_t h,cublasOperation_t ta,int m,int n,int k,const T *a,int lda,
                              const T *b,int ldb,T *c,int ldc){
    const T one=value<T>(1),zero=value<T>(0);
#ifdef OPENMX_MGPU_BENCH_BRIDGE
    if constexpr(std::is_same_v<T,double>)check(openmx_gemmul8Dgemm(h,ta,CUBLAS_OP_N,m,n,k,&one,a,lda,b,ldb,&zero,c,ldc));
    else check(openmx_gemmul8Zgemm(h,ta,CUBLAS_OP_N,m,n,k,&one,a,lda,b,ldb,&zero,c,ldc));
#else
    if constexpr(std::is_same_v<T,double>)check(cublasDgemm(h,ta,CUBLAS_OP_N,m,n,k,&one,a,lda,b,ldb,&zero,c,ldc));
    else check(cublasZgemm(h,ta,CUBLAS_OP_N,m,n,k,&one,a,lda,b,ldb,&zero,c,ldc));
#endif
}
template<class T> void multi(cublasHandle_t h,const std::vector<int> &devices,cublasOperation_t ta,int m,int n,int k,
                             const T *a,int lda,const T *b,int ldb,T *c,int ldc){
    const T one=value<T>(1),zero=value<T>(0);cublasStatus_t status;int handled;
    if constexpr(std::is_same_v<T,double>)handled=openmx_mgpuDgemmTry(h,int(devices.size()),devices.data(),ta,CUBLAS_OP_N,m,n,k,&one,a,lda,b,ldb,&zero,c,ldc,&status);
    else handled=openmx_mgpuZgemmTry(h,int(devices.size()),devices.data(),ta,CUBLAS_OP_N,m,n,k,&one,a,lda,b,ldb,&zero,c,ldc,&status);
    if(!handled){openmx_mgpu_gemm_report r;openmx_mgpuGemmLastReport(&r);std::fprintf(stderr,"DECLINED: %s\n",r.reason);std::exit(4);}check(status);
}
template<class T> void run(int ngpu,int m,int n,int k,int iters,bool forward){
    std::vector<int> devices;for(int i=0;i<ngpu;++i)devices.push_back(i);
    std::vector<T> a(size_t(m)*k),b(size_t(k)*n),reference(size_t(m)*n),out(reference.size());
    for(size_t i=0;i<a.size();++i)a[i]=value<T>(std::sin(i*.17),std::cos(i*.09)*.3);
    for(size_t i=0;i<b.size();++i)b[i]=value<T>(std::cos(i*.13),std::sin(i*.07)*.2);
    T *da,*db,*dc,*tmp=nullptr;check(cudaMalloc(&da,a.size()*sizeof(T)));check(cudaMalloc(&db,b.size()*sizeof(T)));check(cudaMalloc(&dc,out.size()*sizeof(T)));
    if(forward)check(cudaMalloc(&tmp,out.size()*sizeof(T)));
    check(cudaMemcpy(da,a.data(),a.size()*sizeof(T),cudaMemcpyHostToDevice));check(cudaMemcpy(db,b.data(),b.size()*sizeof(T),cudaMemcpyHostToDevice));
    cublasHandle_t h;check(cublasCreate(&h));cudaStream_t stream;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));check(cublasSetStream(h,stream));
    auto product=[&](bool mgpu){
#ifdef OPENMX_MGPU_BENCH_BRIDGE
        if(!mgpu && forward){
            openmx_gemmul8SetForwardStage(0,15,0,0,1);
            if constexpr(std::is_same_v<T,double>){
                check(openmx_gemmul8DgemmFixed(h,0,CUBLAS_OP_N,n,n,n,db,n,da,n,tmp,n,0,1));
                check(openmx_gemmul8DgemmFixed(h,1,CUBLAS_OP_T,n,n,n,db,n,tmp,n,dc,n,0,1));
            }else{
                check(openmx_gemmul8ZgemmFixed(h,0,CUBLAS_OP_N,n,n,n,db,n,da,n,tmp,n,0,1));
                check(openmx_gemmul8ZgemmFixed(h,1,CUBLAS_OP_C,n,n,n,db,n,tmp,n,dc,n,0,1));
            }
            check(cudaStreamSynchronize(stream));openmx_gemmul8TrimWorkspaces();return;
        }
#endif
        if(mgpu)multi(h,devices,CUBLAS_OP_N,m,n,k,da,m,db,k,forward?tmp:dc,m);
        else native(h,CUBLAS_OP_N,m,n,k,da,m,db,k,forward?tmp:dc,m);
        if(forward){const cublasOperation_t adjoint=std::is_same_v<T,double> ? CUBLAS_OP_T : CUBLAS_OP_C;
            if(mgpu)multi(h,devices,adjoint,n,n,n,db,n,tmp,n,dc,n);else native(h,adjoint,n,n,n,db,n,tmp,n,dc,n);}
    };
    for(bool mgpu:{false,true}){
        product(mgpu);check(cudaStreamSynchronize(stream));
        const auto start=std::chrono::steady_clock::now();for(int i=0;i<iters;++i)product(mgpu);check(cudaStreamSynchronize(stream));
        const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()/iters;
        check(cudaMemcpy(out.data(),dc,out.size()*sizeof(T),cudaMemcpyDeviceToHost));if(!mgpu)reference=out;
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
    check(cudaFree(da));check(cudaFree(db));check(cudaFree(dc));if(tmp)check(cudaFree(tmp));check(cublasDestroy(h));check(cudaStreamDestroy(stream));openmx_mgpuGemmRelease();
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
    int count=0;check(cudaGetDeviceCount(&count));if(count<ngpu){std::fprintf(stderr,"need %d GPUs\n",ngpu);return 77;}check(cudaSetDevice(0));
    std::puts("type,m,n,k,path,backend,gpus,iterations,wall_ms,relative_to_single_gpu,active_gpus,peer_copies,host_copies,transaction_bytes,worker_bytes");
    if(complex)run<cuDoubleComplex>(ngpu,m,n,k,iters,forward);else run<double>(ngpu,m,n,k,iters,forward);
}

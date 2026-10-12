/* Compare native FP64, the device default, and explicitly enabled GEMMul8.
   Timings exclude allocation/copies and include a warmup. Full output is
   checked against native FP64, and selected entries against CPU long double. */
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuComplex.h>
#include <algorithm>
#include <cmath>
#include <complex>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>
#include "gemmul8.hpp"

extern "C" cublasStatus_t openmx_gemmul8Dgemm(cublasHandle_t,cublasOperation_t,cublasOperation_t,int,int,int,const double*,const double*,int,const double*,int,const double*,double*,int);
extern "C" cublasStatus_t openmx_gemmul8Zgemm(cublasHandle_t,cublasOperation_t,cublasOperation_t,int,int,int,const cuDoubleComplex*,const cuDoubleComplex*,int,const cuDoubleComplex*,int,const cuDoubleComplex*,cuDoubleComplex*,int);
extern "C" cublasStatus_t openmx_gemmul8DgemmUnblocked(cublasHandle_t,cublasOperation_t,cublasOperation_t,int,int,int,const double*,const double*,int,const double*,int,const double*,double*,int);
extern "C" cublasStatus_t openmx_gemmul8ZgemmUnblocked(cublasHandle_t,cublasOperation_t,cublasOperation_t,int,int,int,const cuDoubleComplex*,const cuDoubleComplex*,int,const cuDoubleComplex*,int,const cuDoubleComplex*,cuDoubleComplex*,int);
extern "C" cublasStatus_t openmx_gemmul8DgemmFixed(cublasHandle_t,int,cublasOperation_t,int,int,int,const double*,int,const double*,int,double*,int,int,unsigned long long);
extern "C" cublasStatus_t openmx_gemmul8ZgemmFixed(cublasHandle_t,int,cublasOperation_t,int,int,int,const cuDoubleComplex*,int,const cuDoubleComplex*,int,cuDoubleComplex*,int,int,unsigned long long);
extern "C" void openmx_gemmul8SetForwardStage(int,int,int,int,int);
extern "C" void openmx_gemmul8SetEnabled(int);
extern "C" int openmx_gemmul8DgemmEnabled();
extern "C" int openmx_gemmul8ZgemmEnabled();
extern "C" long long openmx_gemmul8NativeCalls(int);
extern "C" void openmx_gemmul8ReleaseWorkspaces();
extern "C" void openmx_gemmul8TrimWorkspaces();
extern "C" { int level_stdout = 1; }

static void check(cudaError_t status) {
    if (status != cudaSuccess) { std::fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(status));std::exit(2); }
}
static void check(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) { std::fprintf(stderr,"cuBLAS: %d\n",int(status));std::exit(2); }
}
template<class T> static T scalar(double re,double im=0) {
    if constexpr(std::is_same_v<T,double>) return re;
    else return make_cuDoubleComplex(re,im);
}
template<class T> static std::complex<long double> wide(T value) {
    if constexpr(std::is_same_v<T,double>) return {value,0};
    else return {value.x,value.y};
}
struct Shape { int m,n,k; };
struct Sample { size_t index; std::complex<long double> exact; };

template<class T> static void benchmark(cublasHandle_t handle,Shape shape,int iterations,const std::string &path,bool trim_after_pair) {
    const int m=shape.m,n=shape.n,k=shape.k;
    const bool unblocked=path=="unblocked",fixed=path=="fixed";
    std::vector<T> a(size_t(m)*k),b(size_t(k)*n),reference(size_t(m)*n),output(reference.size());
    for(size_t i=0;i<a.size();++i) a[i]=scalar<T>(std::sin(double(i)*.127),std::cos(double(i)*.213)*.4);
    for(size_t i=0;i<b.size();++i) b[i]=scalar<T>(std::cos(double(i)*.079),std::sin(double(i)*.179)*.3);
    const T alpha=scalar<T>(.73,.17),beta=scalar<T>(0);
    std::vector<Sample> samples;
    /* Fixed-path output is X^T(H X), or X^H(H X) in the complex case.
       Compute one full column of H X in long double, then sample sixteen
       rows of the second product. The O(n^2) CPU work runs once per shape,
       outside all timed regions, with column-major sequential reads. */
    if(fixed) {
        const int column=n-1;
        if constexpr(std::is_same_v<T,double>) {
            std::vector<long double> hx(n);
            for(int q=0;q<n;++q) for(int p=0;p<n;++p)
                hx[p]+=static_cast<long double>(a[p+size_t(q)*n])*b[q+size_t(column)*n];
            for(int sample=0;sample<16;++sample) {
                const int row=(size_t(sample)*7919+size_t(n)-1)%n;
                long double exact=0;
                for(int p=0;p<n;++p) exact+=static_cast<long double>(b[p+size_t(row)*n])*hx[p];
                samples.push_back({row+size_t(column)*n,{exact,0}});
            }
        } else {
            std::vector<std::complex<long double>> hx(n);
            for(int q=0;q<n;++q) for(int p=0;p<n;++p)
                hx[p]+=wide(a[p+size_t(q)*n])*wide(b[q+size_t(column)*n]);
            for(int sample=0;sample<16;++sample) {
                const int row=(size_t(sample)*7919+size_t(n)-1)%n;
                std::complex<long double> exact=0;
                for(int p=0;p<n;++p) exact+=std::conj(wide(b[p+size_t(row)*n]))*hx[p];
                samples.push_back({row+size_t(column)*n,exact});
            }
        }
    } else {
        for(int sample=0;sample<16;++sample) {
            const int row=(size_t(sample)*7919+size_t(m)-1)%m;
            const int column=(size_t(sample)*3571+size_t(n)-1)%n;
            std::complex<long double> sum=0;
            for(int p=0;p<k;++p) sum+=wide(a[row+size_t(p)*m])*wide(b[p+size_t(column)*k]);
            samples.push_back({row+size_t(column)*m,wide(alpha)*sum});
        }
    }
    T *da,*db,*dc,*temporary=nullptr;
    check(cudaMalloc(&da,a.size()*sizeof(T)));check(cudaMalloc(&db,b.size()*sizeof(T)));
    check(cudaMalloc(&dc,output.size()*sizeof(T)));
    if(fixed) check(cudaMalloc(&temporary,output.size()*sizeof(T)));
    check(cudaMemcpy(da,a.data(),a.size()*sizeof(T),cudaMemcpyHostToDevice));
    check(cudaMemcpy(db,b.data(),b.size()*sizeof(T),cudaMemcpyHostToDevice));
    cudaStream_t stream=nullptr;check(cublasGetStream(handle,&stream));
    const auto product=[&]() {
        if constexpr(std::is_same_v<T,double>) {
            if(fixed) {
                check(openmx_gemmul8DgemmFixed(handle,0,CUBLAS_OP_N,n,n,n,db,n,da,n,temporary,n,0,1));
                check(openmx_gemmul8DgemmFixed(handle,1,CUBLAS_OP_T,n,n,n,db,n,temporary,n,dc,n,0,1));
            } else {
                auto gemm=unblocked?openmx_gemmul8DgemmUnblocked:openmx_gemmul8Dgemm;
                check(gemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,&alpha,da,m,db,k,&beta,dc,m));
            }
        } else {
            if(fixed) {
                check(openmx_gemmul8ZgemmFixed(handle,0,CUBLAS_OP_N,n,n,n,db,n,da,n,temporary,n,0,1));
                check(openmx_gemmul8ZgemmFixed(handle,1,CUBLAS_OP_C,n,n,n,db,n,temporary,n,dc,n,0,1));
            } else {
                auto gemm=unblocked?openmx_gemmul8ZgemmUnblocked:openmx_gemmul8Zgemm;
                check(gemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,&alpha,da,m,db,k,&beta,dc,m));
            }
        }
        if(fixed && trim_after_pair) {
            check(cudaStreamSynchronize(stream));
            openmx_gemmul8TrimWorkspaces();
        }
    };
    cudaEvent_t start,end;check(cudaEventCreate(&start));check(cudaEventCreate(&end));
    for(int mode:{0,-1,1}) {
        openmx_gemmul8ReleaseWorkspaces();
        openmx_gemmul8SetEnabled(mode);
        if(fixed) openmx_gemmul8SetForwardStage(0,15,0,0,1);
        const bool ozaki=std::is_same_v<T,double> ? openmx_gemmul8DgemmEnabled() : openmx_gemmul8ZgemmEnabled();
        if ((mode==0 && ozaki) || (mode==1 && !ozaki)) {
            std::fprintf(stderr,"Unexpected environment override of benchmark mode\n");std::exit(2);
        }
        if(mode==1 && (unblocked || fixed)) {
            size_t free_bytes=0,total_bytes=0;
            const size_t required=gemmul8::workSize<!std::is_same_v<T,double>,gemmul8::Backend::INT8>(
                size_t(m),size_t(n),size_t(k),15,false,false,nullptr,nullptr,false);
            check(cudaMemGetInfo(&free_bytes,&total_bytes));
            std::printf("# %s %dx%dx%d full_workspace_MiB=%.1f free_MiB=%.1f\n",
                (std::is_same_v<T,double>) ? "D" : "Z",m,n,k,required/1048576.0,free_bytes/1048576.0);
            if(free_bytes<required) {
                std::fprintf(stderr,"Insufficient memory for full-workspace comparison; refusing a blocked fallback\n");
                std::exit(4);
            }
        }
        const long long native_before=openmx_gemmul8NativeCalls(!std::is_same_v<T,double>);
        product();check(cudaDeviceSynchronize());
        const auto wall_start=std::chrono::steady_clock::now();
        check(cudaEventRecord(start,stream));
        for(int iteration=0;iteration<iterations;++iteration) product();
        check(cudaEventRecord(end,stream));check(cudaEventSynchronize(end));
        const double wall_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-wall_start).count()/iterations;
        float elapsed=0;check(cudaEventElapsedTime(&elapsed,start,end));
        const long long native_calls=openmx_gemmul8NativeCalls(!std::is_same_v<T,double>)-native_before;
        const char *effective=native_calls==(fixed?2:1)*(iterations+1)?"fp64":native_calls==0?"gemmul8":"mixed";
        check(cudaMemcpy(output.data(),dc,output.size()*sizeof(T),cudaMemcpyDeviceToHost));
        if(mode==0) reference=output;
        long double error=0,scale=0,sample_error=0,sample_scale=0;
        for(size_t i=0;i<output.size();++i) {
            if(!std::isfinite(double(std::abs(wide(output[i]))))) {
                std::fprintf(stderr,"Nonfinite result\n");std::exit(3);
            }
            error=std::max(error,std::abs(wide(output[i])-wide(reference[i])));
            scale=std::max(scale,std::abs(wide(reference[i])));
        }
        for(const Sample &sample:samples) {
            sample_error=std::max(sample_error,std::abs(wide(output[sample.index])-sample.exact));
            sample_scale=std::max(sample_scale,std::abs(sample.exact));
        }
        const double relative=double(error/std::max(1.L,scale));
        const double sample_relative=double(sample_error/std::max(1.L,sample_scale));
        const double ms=elapsed/iterations;
        std::printf("%s,%d,%d,%d,%s,%s,%s,%s,%d,%.6f,%.6f,%.3f,%.3e,%.3e\n",
            (std::is_same_v<T,double>) ? "D" : "Z",m,n,k,path.c_str(),
            fixed?(trim_after_pair?"trimmed":"retained"):(unblocked?"retained":"capped"),
            mode==0?"native":mode<0?"auto":"gemmul8",
            effective,iterations,ms,wall_ms,
            ((std::is_same_v<T,double>) ? 2.0 : 8.0)*(fixed?2:1)*m*n*k/(ms*1.e6),relative,sample_relative);
        std::fflush(stdout);
        if(relative>1.e-11 || sample_relative>1.e-11) {
            std::fprintf(stderr,"Accuracy check failed\n");std::exit(3);
        }
    }
    check(cudaEventDestroy(start));check(cudaEventDestroy(end));
    openmx_gemmul8ReleaseWorkspaces();
    check(cudaFree(da));check(cudaFree(db));check(cudaFree(dc));
    if(temporary) check(cudaFree(temporary));
}

int main(int argc,char **argv) {
    int iterations=5;
    bool trim_after_pair=true;
    std::string sizes="1024x1024x1024,2048x2048x2048,4096x512x4096";
    std::string path="general",types="both";
    for(int i=1;i<argc;++i) {
        if(std::strcmp(argv[i],"--iterations")==0 && i+1<argc) iterations=std::atoi(argv[++i]);
        else if(std::strcmp(argv[i],"--sizes")==0 && i+1<argc) sizes=argv[++i];
        else if(std::strcmp(argv[i],"--path")==0 && i+1<argc) path=argv[++i];
        else if(std::strcmp(argv[i],"--types")==0 && i+1<argc) types=argv[++i];
        else if(std::strcmp(argv[i],"--retain-workspace")==0) trim_after_pair=false;
        else if(std::strcmp(argv[i],"--trim-after-pair")==0) trim_after_pair=true;
        else {std::fprintf(stderr,"Usage: %s [--iterations N] [--sizes MxNxK,...] "
            "[--path general|unblocked|fixed] [--types D|Z|both] "
            "[--trim-after-pair|--retain-workspace]\n",argv[0]);return 2;}
    }
    if(iterations<1 || iterations>100000) {std::fprintf(stderr,"Invalid iterations\n");return 2;}
    if(path!="general" && path!="unblocked" && path!="fixed") {std::fprintf(stderr,"Invalid path\n");return 2;}
    if(types=="D") types="real";
    if(types=="Z") types="complex";
    if(types!="real" && types!="complex" && types!="both") {std::fprintf(stderr,"Invalid types\n");return 2;}
    std::vector<Shape> shapes;
    size_t begin=0;
    while(begin<sizes.size()) {
        const size_t end=sizes.find(',',begin);
        const std::string entry=sizes.substr(begin,end==std::string::npos?end:end-begin);
        Shape shape;char extra;
        if(std::sscanf(entry.c_str(),"%dx%dx%d%c",&shape.m,&shape.n,&shape.k,&extra)!=3 ||
            shape.m<1 || shape.n<1 || shape.k<1) {std::fprintf(stderr,"Invalid size: %s\n",entry.c_str());return 2;}
        if(path=="fixed" && (shape.m!=shape.n || shape.n!=shape.k)) {
            std::fprintf(stderr,"Fixed forward pairs require square M=N=K\n");return 2;
        }
        shapes.push_back(shape);
        if(end==std::string::npos) break;
        begin=end+1;
    }
    if(shapes.empty()) {std::fprintf(stderr,"No sizes specified\n");return 2;}
    setenv("OPENMX_GEMMUL8_NUM_MOD_D","15",1);setenv("OPENMX_GEMMUL8_NUM_MOD_Z","15",1);
    setenv("OPENMX_GEMMUL8_FASTMODE_D","0",1);setenv("OPENMX_GEMMUL8_FASTMODE_Z","0",1);
    if(path!="general") setenv("OPENMX_GEMMUL8_MIN_FREE_AFTER_MB","0",1);
    cublasHandle_t handle;check(cublasCreate(&handle));
    int device=0,ratio=0;check(cudaGetDevice(&device));
    cudaDeviceProp properties;check(cudaGetDeviceProperties(&properties,device));
    check(cudaDeviceGetAttribute(&ratio,cudaDevAttrSingleToDoublePrecisionPerfRatio,device));
    std::printf("# device=%s sm=%d%d FP32:FP64=%d\n",properties.name,properties.major,properties.minor,ratio);
    if(path=="fixed") std::printf("# fixed: total H X plus X^T/H(H X), full workspace, L15 accurate, no reuse; trim_after_pair=%d\n",int(trim_after_pair));
    std::puts("type,m,n,k,path,workspace,requested,effective,iterations,ms,wall_ms,gflops,relative_to_native,relative_to_long_double_samples");
    for(const Shape shape:shapes) {
        if(types!="complex") benchmark<double>(handle,shape,iterations,path,trim_after_pair);
        if(types!="real") benchmark<cuDoubleComplex>(handle,shape,iterations,path,trim_after_pair);
    }
    check(cublasDestroy(handle));
}

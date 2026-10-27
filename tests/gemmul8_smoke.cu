/* Run with tests/run_gemmul8_smoke.sh.  The reference is evaluated on the
   CPU in long double; no cuBLAS result is used as the accuracy oracle. */
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuComplex.h>
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <vector>

extern "C" cublasStatus_t openmx_gemmul8Dgemm(cublasHandle_t,cublasOperation_t,cublasOperation_t,int,int,int,const double*,const double*,int,const double*,int,const double*,double*,int);
extern "C" cublasStatus_t openmx_gemmul8Zgemm(cublasHandle_t,cublasOperation_t,cublasOperation_t,int,int,int,const cuDoubleComplex*,const cuDoubleComplex*,int,const cuDoubleComplex*,int,const cuDoubleComplex*,cuDoubleComplex*,int);
extern "C" cublasStatus_t openmx_gemmul8DgemmFixed(cublasHandle_t,int,cublasOperation_t,int,int,int,const double*,int,const double*,int,double*,int,int,unsigned long long);
extern "C" cublasStatus_t openmx_gemmul8ZgemmFixed(cublasHandle_t,int,cublasOperation_t,int,int,int,const cuDoubleComplex*,int,const cuDoubleComplex*,int,cuDoubleComplex*,int,int,unsigned long long);
extern "C" void openmx_gemmul8SetForwardStage(int,int,int,int,int);
extern "C" size_t openmx_gemmul8ForwardCounters(long long[5],double[2]);
extern "C" void openmx_gemmul8ReleasePrepared();
extern "C" void openmx_gemmul8ReleaseWorkspaces();
extern "C" size_t openmx_gemmul8ZWorkspaceSize(int,int,int);
extern "C" long long openmx_gemmul8NativeCalls(int);

void check(cudaError_t status) { if (status != cudaSuccess) { fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(status)); exit(2); } }
void check(cublasStatus_t status) { if (status != CUBLAS_STATUS_SUCCESS) { fprintf(stderr,"cuBLAS: %d\n",int(status)); exit(2); } }
template<class T> T scalar(double r,double i) { if constexpr (std::is_same_v<T,double>) return r; else return make_cuDoubleComplex(r,i); }
template<class T> std::complex<long double> wide(T value) { if constexpr (std::is_same_v<T,double>) return {value,0}; else return {value.x,value.y}; }

template<class T> double run(cublasHandle_t handle,cublasOperation_t ta,cublasOperation_t tb,int m,int n,int k,
                            bool fixed=false,bool left=false) {
    const int lda=(ta==CUBLAS_OP_N?m:k)+3, ldb=(tb==CUBLAS_OP_N?k:n)+5, ldc=m+7;
    std::vector<T> a(size_t(lda)*(ta==CUBLAS_OP_N?k:m)),b(size_t(ldb)*(tb==CUBLAS_OP_N?n:k)),c(size_t(ldc)*n),out(c.size());
    for (size_t i=0;i<a.size();++i) a[i]=scalar<T>(sin(double(i)*.127),cos(double(i)*.213)*.4);
    for (size_t i=0;i<b.size();++i) b[i]=scalar<T>(cos(double(i)*.079),sin(double(i)*.179)*.3);
    for (size_t i=0;i<c.size();++i) c[i]=scalar<T>(sin(double(i)*.093)*.1,cos(double(i)*.083)*.1);
    const T alpha=fixed?scalar<T>(1,0):scalar<T>(.73,.17),beta=fixed?scalar<T>(0,0):scalar<T>(-.31,.11);
    T *da,*db,*dc;
    check(cudaMalloc(&da,a.size()*sizeof(T))); check(cudaMalloc(&db,b.size()*sizeof(T))); check(cudaMalloc(&dc,c.size()*sizeof(T)));
    check(cudaMemcpy(da,a.data(),a.size()*sizeof(T),cudaMemcpyHostToDevice)); check(cudaMemcpy(db,b.data(),b.size()*sizeof(T),cudaMemcpyHostToDevice)); check(cudaMemcpy(dc,c.data(),c.size()*sizeof(T),cudaMemcpyHostToDevice));
    double worst=0;
    // Fixed operands are prepared, reused, then changed with a new version.
    // Comparing every result to the CPU detects reuse of stale residues.
    for(int pass=0;pass<(fixed?3:1);++pass) {
        if(fixed) {
            if(pass==2) {
                auto &x=left?a:b;
                for(size_t i=0;i<x.size();++i) x[i]=scalar<T>(sin(double(i)*.107+.2),cos(double(i)*.119+.4)*.6);
                check(cudaMemcpy(left?da:db,x.data(),x.size()*sizeof(T),cudaMemcpyHostToDevice));
            }
            if constexpr (std::is_same_v<T,double>)
                check(openmx_gemmul8DgemmFixed(handle,left,left?ta:tb,m,n,k,left?da:db,left?lda:ldb,
                    left?db:da,left?ldb:lda,dc,ldc,1,pass==2?2:1));
            else
                check(openmx_gemmul8ZgemmFixed(handle,left,left?ta:tb,m,n,k,left?da:db,left?lda:ldb,
                    left?db:da,left?ldb:lda,dc,ldc,1,pass==2?2:1));
        } else {
            if constexpr (std::is_same_v<T,double>) check(openmx_gemmul8Dgemm(handle,ta,tb,m,n,k,&alpha,da,lda,db,ldb,&beta,dc,ldc));
            else check(openmx_gemmul8Zgemm(handle,ta,tb,m,n,k,&alpha,da,lda,db,ldb,&beta,dc,ldc));
        }
        check(cudaGetLastError()); check(cudaMemcpy(out.data(),dc,out.size()*sizeof(T),cudaMemcpyDeviceToHost));
        long double error=0,scale=0;
        for (int j=0;j<n;++j) for (int i=0;i<m;++i) {
            std::complex<long double> product=0;
            for (int p=0;p<k;++p) {
                auto av=wide(a[ta==CUBLAS_OP_N?i+p*lda:p+i*lda]);
                auto bv=wide(b[tb==CUBLAS_OP_N?p+j*ldb:j+p*ldb]);
                if (ta==CUBLAS_OP_C) av=std::conj(av);
                if (tb==CUBLAS_OP_C) bv=std::conj(bv);
                product+=av*bv;
            }
            auto ref=wide(alpha)*product+wide(beta)*wide(c[i+j*ldc]);
            if (!std::isfinite(wide(out[i+j*ldc]).real()) || !std::isfinite(wide(out[i+j*ldc]).imag())) {
                fprintf(stderr,"Nonfinite GEMM result\n"); exit(3);
            }
            error=std::max(error,std::abs(wide(out[i+j*ldc])-ref)); scale=std::max(scale,std::abs(ref));
        }
        for (int j=0;j<n;++j) for(int i=m;i<ldc;++i) if(wide(c[i+j*ldc])!=wide(out[i+j*ldc])) {fprintf(stderr,"Padding overwritten\n");exit(3);}
        double relative=double(error/std::max(1.L,scale));
        if(relative>1e-12) {fprintf(stderr,"Residual exceeded tolerance: %.3e\n",relative);exit(3);}
        worst=std::max(worst,relative);
    }
    check(cudaFree(da)); check(cudaFree(db)); check(cudaFree(dc));
    return worst;
}

int main(int argc, char **argv) {
    const bool large_n=argc==2 && std::strcmp(argv[1],"--large-n")==0;
    if (argc!=1 && !large_n) { fprintf(stderr,"Usage: %s [--large-n]\n",argv[0]); return 1; }
    cublasHandle_t handle; check(cublasCreate(&handle));
    int cuda_version=0,blas_version=0;
    check(cudaRuntimeGetVersion(&cuda_version)); check(cublasGetVersion(handle,&blas_version));
    printf("CUDA runtime %d, cuBLAS %d\n",cuda_version,blas_version);
    double worst=0; int count=0;
    if (large_n) {
        // v3.5.4 splits undo-scaling launches at grid.y's 65535-block limit.
        // Four columns per block: this is the first invalid size in v3.5.2,
        // while the padded product still fits one grouped-CRT tile. Disable
        // workspace blocking in the runner so it cannot hide the boundary.
        constexpr int columns=65535*4+1;
        for (const char *fast:{"0","1"}) {
            setenv("OPENMX_GEMMUL8_FASTMODE_D",fast,1);
            setenv("OPENMX_GEMMUL8_FASTMODE_Z",fast,1);
            worst=std::max(worst,run<double>(handle,CUBLAS_OP_N,CUBLAS_OP_T,3,columns,3));
            openmx_gemmul8ReleaseWorkspaces();
            worst=std::max(worst,run<cuDoubleComplex>(handle,CUBLAS_OP_C,CUBLAS_OP_C,3,columns,3));
            openmx_gemmul8ReleaseWorkspaces();
        }
        if (openmx_gemmul8NativeCalls(0)!=0 || openmx_gemmul8NativeCalls(1)!=0) {
            fprintf(stderr,"Large-N test fell back to native GEMM; run alone with sufficient GPU memory\n"); return 6;
        }
        check(cudaGetLastError()); check(cublasDestroy(handle));
        printf("PASS: 4 real/complex large-N GEMMs (%d columns), fast/accurate, no fallback; max relative error %.3e\n",columns,worst);
        return 0;
    }
    for(const char *fast:{"0","1"}) {
        setenv("OPENMX_GEMMUL8_FASTMODE_D",fast,1); setenv("OPENMX_GEMMUL8_FASTMODE_Z",fast,1);
        for(auto ta:{CUBLAS_OP_N,CUBLAS_OP_T,CUBLAS_OP_C}) for(auto tb:{CUBLAS_OP_N,CUBLAS_OP_T,CUBLAS_OP_C}) {
            worst=std::max(worst,run<double>(handle,ta,tb,65,47,71));
            worst=std::max(worst,run<cuDoubleComplex>(handle,ta,tb,65,47,71)); count+=2;
        }
    }
    // Exercise the new modular reduction kernels beyond a single K block.
    worst=std::max(worst,run<double>(handle,CUBLAS_OP_N,CUBLAS_OP_N,3,2,131075));
    worst=std::max(worst,run<cuDoubleComplex>(handle,CUBLAS_OP_N,CUBLAS_OP_C,3,2,131075)); count+=2;
    // Force GEMMul8's memory-saving block path with a long skinny product.
    if (openmx_gemmul8ZWorkspaceSize(3,2,524291)!=size_t(256)*1024*1024) {
        fprintf(stderr,"Expected the 256 MiB blocked workspace\n"); return 4;
    }
    worst=std::max(worst,run<cuDoubleComplex>(handle,CUBLAS_OP_N,CUBLAS_OP_C,3,2,524291)); ++count;
    printf("PASS: %d real/complex GEMMs, fast/accurate scaling, N/T/C, padded leading dimensions, K blocking; max relative error %.3e\n",count,worst);
    double fixed_worst=0;
    for(int fast:{0,1}) {
        openmx_gemmul8SetForwardStage(2,15,fast,1,1);
        for(bool left:{false,true}) for(auto op:{CUBLAS_OP_N,CUBLAS_OP_T,CUBLAS_OP_C}) {
            const auto ta=left?op:CUBLAS_OP_N,tb=left?CUBLAS_OP_N:op;
            fixed_worst=std::max(fixed_worst,run<double>(handle,ta,tb,65,47,71,true,left));
            fixed_worst=std::max(fixed_worst,run<cuDoubleComplex>(handle,ta,tb,65,47,71,true,left));
        }
    }
    long long counters[5];double seconds[2];
    const size_t retained=openmx_gemmul8ForwardCounters(counters,seconds);
    if(!retained || counters[0]!=36 || counters[1]!=36 || counters[2]!=48 || counters[3]!=24 || counters[4]!=0) {
        fprintf(stderr,"Expected 48 preparations, 24 reuses, no capacity fallbacks and retained storage\n");return 5;
    }
    openmx_gemmul8ReleasePrepared();
    if(openmx_gemmul8ForwardCounters(counters,seconds)!=0) {fprintf(stderr,"Prepared cache retained storage after release\n");return 5;}
    openmx_gemmul8ReleaseWorkspaces(); check(cudaGetLastError()); check(cublasDestroy(handle));
    printf("PASS: 72 prepared real/complex GEMMs, both sides, N/T/C, fast/accurate, version invalidation; 48 preparations, 24 reuses, no fallback; max relative error %.3e\n",fixed_worst);
}

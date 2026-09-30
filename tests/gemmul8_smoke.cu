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
#include <type_traits>
#include <vector>

extern "C" cublasStatus_t openmx_gemmul8Dgemm(cublasHandle_t,cublasOperation_t,cublasOperation_t,int,int,int,const double*,const double*,int,const double*,int,const double*,double*,int);
extern "C" cublasStatus_t openmx_gemmul8Zgemm(cublasHandle_t,cublasOperation_t,cublasOperation_t,int,int,int,const cuDoubleComplex*,const cuDoubleComplex*,int,const cuDoubleComplex*,int,const cuDoubleComplex*,cuDoubleComplex*,int);
extern "C" void openmx_gemmul8ReleaseWorkspaces();
extern "C" size_t openmx_gemmul8ZWorkspaceSize(int,int,int);

void check(cudaError_t status) { if (status != cudaSuccess) { fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(status)); exit(2); } }
void check(cublasStatus_t status) { if (status != CUBLAS_STATUS_SUCCESS) { fprintf(stderr,"cuBLAS: %d\n",int(status)); exit(2); } }
template<class T> T scalar(double r,double i) { if constexpr (std::is_same_v<T,double>) return r; else return make_cuDoubleComplex(r,i); }
template<class T> std::complex<long double> wide(T value) { if constexpr (std::is_same_v<T,double>) return {value,0}; else return {value.x,value.y}; }

template<class T> double run(cublasHandle_t handle,cublasOperation_t ta,cublasOperation_t tb,int m,int n,int k) {
    const int lda=(ta==CUBLAS_OP_N?m:k)+3, ldb=(tb==CUBLAS_OP_N?k:n)+5, ldc=m+7;
    std::vector<T> a(size_t(lda)*(ta==CUBLAS_OP_N?k:m)),b(size_t(ldb)*(tb==CUBLAS_OP_N?n:k)),c(size_t(ldc)*n),out(c.size());
    for (size_t i=0;i<a.size();++i) a[i]=scalar<T>(sin(double(i)*.127),cos(double(i)*.213)*.4);
    for (size_t i=0;i<b.size();++i) b[i]=scalar<T>(cos(double(i)*.079),sin(double(i)*.179)*.3);
    for (size_t i=0;i<c.size();++i) c[i]=scalar<T>(sin(double(i)*.093)*.1,cos(double(i)*.083)*.1);
    const T alpha=scalar<T>(.73,.17),beta=scalar<T>(-.31,.11);
    T *da,*db,*dc;
    check(cudaMalloc(&da,a.size()*sizeof(T))); check(cudaMalloc(&db,b.size()*sizeof(T))); check(cudaMalloc(&dc,c.size()*sizeof(T)));
    check(cudaMemcpy(da,a.data(),a.size()*sizeof(T),cudaMemcpyHostToDevice)); check(cudaMemcpy(db,b.data(),b.size()*sizeof(T),cudaMemcpyHostToDevice)); check(cudaMemcpy(dc,c.data(),c.size()*sizeof(T),cudaMemcpyHostToDevice));
    if constexpr (std::is_same_v<T,double>) check(openmx_gemmul8Dgemm(handle,ta,tb,m,n,k,&alpha,da,lda,db,ldb,&beta,dc,ldc));
    else check(openmx_gemmul8Zgemm(handle,ta,tb,m,n,k,&alpha,da,lda,db,ldb,&beta,dc,ldc));
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
        error=std::max(error,std::abs(wide(out[i+j*ldc])-ref)); scale=std::max(scale,std::abs(ref));
    }
    for (int j=0;j<n;++j) for(int i=m;i<ldc;++i) if(wide(c[i+j*ldc])!=wide(out[i+j*ldc])) {fprintf(stderr,"Padding overwritten\n");exit(3);}
    check(cudaFree(da)); check(cudaFree(db)); check(cudaFree(dc));
    double relative=double(error/std::max(1.L,scale));
    if(relative>1e-12) {fprintf(stderr,"Residual exceeded tolerance: %.3e\n",relative);exit(3);}
    return relative;
}

int main() {
    cublasHandle_t handle; check(cublasCreate(&handle));
    int cuda_version=0,blas_version=0;
    check(cudaRuntimeGetVersion(&cuda_version)); check(cublasGetVersion(handle,&blas_version));
    printf("CUDA runtime %d, cuBLAS %d\n",cuda_version,blas_version);
    double worst=0; int count=0;
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
    openmx_gemmul8ReleaseWorkspaces(); check(cublasDestroy(handle));
    printf("PASS: %d real/complex GEMMs, fast/accurate scaling, N/T/C, padded leading dimensions, K blocking; max relative error %.3e\n",count,worst);
}

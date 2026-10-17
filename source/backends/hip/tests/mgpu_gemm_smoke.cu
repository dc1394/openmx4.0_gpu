/* Independent long-double numerical oracle, actual separate HIP devices.
 * Build with run_mgpu_gemm_smoke.sh. No MPI or scheduler submission here. */
#include "openmx_mgpu_gemm.h"
#include "hip_runtime_compat.h"
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>

static int bridge_entries=0;
static int fail_bridge_at=0;
static bool host_operands=false;
static void require(bool ok,const char *why) {if(!ok){std::fprintf(stderr,"FAIL: %s\n",why);std::exit(3);}}
static void check(hipError_t e) {if(e!=hipSuccess){std::fprintf(stderr,"HIP %s\n",hipGetErrorString(e));std::exit(2);}}
static void check(hipblasStatus_t e) {if(e!=HIPBLAS_STATUS_SUCCESS){std::fprintf(stderr,"hipBLAS %d\n",int(e));std::exit(2);}}
#ifdef OPENMX_MGPU_TEST_BRIDGE
/* A lightweight bridge substitutes native arithmetic while checking the
 * real recursive-dispatch contract. Production GEMMul8 can be linked instead. */
extern "C" hipblasStatus_t openmx_gemmul8Dgemm(hipblasHandle_t h,hipblasOperation_t ta,hipblasOperation_t tb,
    int m,int n,int k,const double *alpha,const double *a,int lda,const double *b,int ldb,const double *beta,double *c,int ldc)
{
    require(openmx_mgpuGemmInWorker()==1,"D worker recursion guard");++bridge_entries;
    if(bridge_entries==fail_bridge_at) return HIPBLAS_STATUS_ALLOC_FAILED;
    hipblasStatus_t ignored;
    require(openmx_mgpuDgemmTry(h,2,nullptr,ta,tb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc,&ignored)==0,"recursive D dispatch declined");
    return hipblasDgemm(h,ta,tb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc);
}
extern "C" hipblasStatus_t openmx_gemmul8Zgemm(hipblasHandle_t h,hipblasOperation_t ta,hipblasOperation_t tb,
    int m,int n,int k,const hipDoubleComplex *alpha,const hipDoubleComplex *a,int lda,const hipDoubleComplex *b,int ldb,
    const hipDoubleComplex *beta,hipDoubleComplex *c,int ldc)
{
    require(openmx_mgpuGemmInWorker()==1,"Z worker recursion guard");++bridge_entries;
    if(bridge_entries==fail_bridge_at) return HIPBLAS_STATUS_ALLOC_FAILED;
    return hipblasZgemm(h,ta,tb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc);
}
#endif
template<class T> T scalar(double r,double i=0) {
    if constexpr(std::is_same_v<T,double>) return r;else return make_hipDoubleComplex(r,i);
}
template<class T> std::complex<long double> wide(T x) {
    if constexpr(std::is_same_v<T,double>) return {x,0};else return {x.x,x.y};
}
template<class T> int call(hipblasHandle_t h,const std::vector<int> &devices,hipblasOperation_t ta,hipblasOperation_t tb,
    int m,int n,int k,const T *alpha,const T *a,int lda,const T *b,int ldb,const T *beta,T *c,int ldc,hipblasStatus_t *s) {
    if (host_operands) {
        if constexpr(std::is_same_v<T,double>) return openmx_mgpuDgemmHostTry(int(devices.size()),devices.data(),ta,tb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc,s);
        else return openmx_mgpuZgemmHostTry(int(devices.size()),devices.data(),ta,tb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc,s);
    }
    if constexpr(std::is_same_v<T,double>) return openmx_mgpuDgemmTry(h,int(devices.size()),devices.data(),ta,tb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc,s);
    else return openmx_mgpuZgemmTry(h,int(devices.size()),devices.data(),ta,tb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc,s);
}
__global__ void delay_input(unsigned long long cycles) {
    const unsigned long long start=clock64();while(clock64()-start<cycles) {}
}
static double worst=0;
static int cases=0;
template<class T> void run(hipblasHandle_t h,const std::vector<int> &devices,hipblasOperation_t ta,hipblasOperation_t tb,
    int m,int n,int k,bool device_scalars=false,bool beta_zero=false,bool expect_decline=false) {
    int origin=-1;check(hipGetDevice(&origin));hipStream_t stream;check(hipblasGetStream(h,&stream));
    hipblasMath_t original_math;check(hipblasGetMathMode(h,&original_math));
    hipblasAtomicsMode_t original_atomics;check(hipblasGetAtomicsMode(h,&original_atomics));
    const int lda=(ta==HIPBLAS_OP_N?m:k)+3,ldb=(tb==HIPBLAS_OP_N?k:n)+5,ldc=m+7;
    std::vector<T> a(size_t(lda)*(ta==HIPBLAS_OP_N?k:m)),b(size_t(ldb)*(tb==HIPBLAS_OP_N?n:k)),c(size_t(ldc)*n),out(c.size());
    for(size_t i=0;i<a.size();++i) a[i]=scalar<T>(std::sin(i*.127),std::cos(i*.213)*.4);
    for(size_t i=0;i<b.size();++i) b[i]=scalar<T>(std::cos(i*.079),std::sin(i*.179)*.3);
    for(size_t i=0;i<c.size();++i) c[i]=scalar<T>(std::sin(i*.093)*.1,std::cos(i*.083)*.1);
    if(beta_zero) for(int j=0;j<n;++j) for(int i=0;i<m;++i) c[i+size_t(j)*ldc]=scalar<T>(std::numeric_limits<double>::quiet_NaN());
    const T alpha=scalar<T>(.73,.17),beta=beta_zero?scalar<T>(0):scalar<T>(-.31,.11);
    T *da=nullptr,*db=nullptr,*dc=nullptr,*ds=nullptr;void *pinned=nullptr;
    check(hipMalloc(&da,a.size()*sizeof(T)));check(hipMalloc(&db,b.size()*sizeof(T)));check(hipMalloc(&dc,c.size()*sizeof(T)));
    const size_t host_bytes=(a.size()+b.size()+c.size())*sizeof(T);
    check(hipHostAlloc(&pinned,host_bytes,hipHostAllocPortable));
    T *pa=static_cast<T *>(pinned),*pb=pa+a.size(),*pc=pb+b.size();
    std::memcpy(pa,a.data(),a.size()*sizeof(T));std::memcpy(pb,b.data(),b.size()*sizeof(T));std::memcpy(pc,c.data(),c.size()*sizeof(T));
    delay_input<<<1,1,0,stream>>>(1000000);
    check(hipMemcpyAsync(da,pa,a.size()*sizeof(T),hipMemcpyHostToDevice,stream));
    check(hipMemcpyAsync(db,pb,b.size()*sizeof(T),hipMemcpyHostToDevice,stream));
    check(hipMemcpyAsync(dc,pc,c.size()*sizeof(T),hipMemcpyHostToDevice,stream));
    const T *ap=&alpha,*bp=&beta;
    if(device_scalars) {
        check(hipMalloc(&ds,2*sizeof(T)));check(hipMemcpyAsync(ds,&alpha,sizeof(T),hipMemcpyHostToDevice,stream));
        check(hipMemcpyAsync(ds+1,&beta,sizeof(T),hipMemcpyHostToDevice,stream));ap=ds;bp=ds+1;
        check(hipblasSetPointerMode(h,HIPBLAS_POINTER_MODE_DEVICE));
    }
    hipblasStatus_t status=HIPBLAS_STATUS_INTERNAL_ERROR;
    out=c;
    const int handled=host_operands ? call(h,devices,ta,tb,m,n,k,&alpha,a.data(),lda,b.data(),ldb,&beta,out.data(),ldc,&status)
        : call(h,devices,ta,tb,m,n,k,ap,da,lda,db,ldb,bp,dc,ldc,&status);
    openmx_mgpu_gemm_report rep;openmx_mgpuGemmLastReport(&rep);
    if(handled!=!expect_decline) {std::fprintf(stderr,"dispatch %dx%dx%d reason: %s\n",m,n,k,rep.reason);require(false,"expected handled/declined result");}
    if(handled) check(status);
    int after=-1;check(hipGetDevice(&after));require(after==origin,"current device restored");
    require(openmx_mgpuGemmInWorker()==0,"worker recursion guard restored");
    hipStream_t after_stream;check(hipblasGetStream(h,&after_stream));require(after_stream==stream,"caller stream unchanged");
    hipblasPointerMode_t pm;check(hipblasGetPointerMode(h,&pm));
    require(pm==(device_scalars?HIPBLAS_POINTER_MODE_DEVICE:HIPBLAS_POINTER_MODE_HOST),"caller pointer mode unchanged");
    hipblasMath_t after_math;check(hipblasGetMathMode(h,&after_math));require(after_math==original_math,"caller math mode unchanged");
    hipblasAtomicsMode_t after_atomics;check(hipblasGetAtomicsMode(h,&after_atomics));require(after_atomics==original_atomics,"caller atomics mode unchanged");
    check(hipStreamSynchronize(stream));if(!host_operands)check(hipMemcpy(out.data(),dc,out.size()*sizeof(T),hipMemcpyDeviceToHost));
    if(expect_decline) require(std::memcmp(out.data(),c.data(),c.size()*sizeof(T))==0,"declined call leaves C byte-identical");
    else {
        require(rep.active_gpus==int(devices.size()),"all requested GPUs performed products");
        require(rep.tiles>=rep.active_gpus,"report counts worker tiles");
        if(std::getenv("OPENMX_MGPU_GEMM_FORCE_HOST")) require(rep.host_transfers>0,"forced host staging exercised");
        long double error=0,scale=0;
        for(int j=0;j<n;++j) for(int i=0;i<m;++i) {
            std::complex<long double> sum=0;
            for(int p=0;p<k;++p) {
                auto av=wide(a[ta==HIPBLAS_OP_N?i+size_t(p)*lda:p+size_t(i)*lda]);
                auto bv=wide(b[tb==HIPBLAS_OP_N?p+size_t(j)*ldb:j+size_t(p)*ldb]);
                if(ta==HIPBLAS_OP_C) av=std::conj(av);if(tb==HIPBLAS_OP_C) bv=std::conj(bv);
                sum+=av*bv;
            }
            auto exact=wide(alpha)*sum;if(!beta_zero) exact+=wide(beta)*wide(c[i+size_t(j)*ldc]);
            require(std::isfinite(double(std::abs(wide(out[i+size_t(j)*ldc])))),"finite result (including beta=0 with NaN C)");
            error=std::max(error,std::abs(wide(out[i+size_t(j)*ldc])-exact));scale=std::max(scale,std::abs(exact));
        }
        for(int j=0;j<n;++j) require(std::memcmp(out.data()+m+size_t(j)*ldc,c.data()+m+size_t(j)*ldc,(ldc-m)*sizeof(T))==0,"C padding preserved");
        const double relative=double(error/std::max(1.L,scale));worst=std::max(worst,relative);
        if(relative>1e-11) {std::fprintf(stderr,"relative error %.3e\n",relative);require(false,"long-double numerical oracle");}
    }
    check(hipblasSetPointerMode(h,HIPBLAS_POINTER_MODE_HOST));
    check(hipFree(da));check(hipFree(db));check(hipFree(dc));if(ds)check(hipFree(ds));check(hipFreeHost(pinned));++cases;
}
template<class T> void forward(hipblasHandle_t h,const std::vector<int> &devices) {
    constexpr int n=97;std::vector<T> a(n*n),x(n*n),out(n*n);
    for(int i=0;i<n*n;++i) {a[i]=scalar<T>(std::sin(i*.17),std::cos(i*.07)*.2);x[i]=scalar<T>(std::cos(i*.11),std::sin(i*.13)*.3);}
    T *da,*dx,*tmp;check(hipMalloc(&da,n*n*sizeof(T)));check(hipMalloc(&dx,n*n*sizeof(T)));check(hipMalloc(&tmp,n*n*sizeof(T)));
    check(hipMemcpy(da,a.data(),n*n*sizeof(T),hipMemcpyHostToDevice));check(hipMemcpy(dx,x.data(),n*n*sizeof(T),hipMemcpyHostToDevice));
    const T one=scalar<T>(1),zero=scalar<T>(0);hipblasStatus_t status;
    require(call(h,devices,HIPBLAS_OP_N,HIPBLAS_OP_N,n,n,n,&one,da,n,dx,n,&zero,tmp,n,&status)==1,"forward first product handled");check(status);
    const hipblasOperation_t adjoint=std::is_same_v<T,double> ? HIPBLAS_OP_T : HIPBLAS_OP_C;
    require(call(h,devices,adjoint,HIPBLAS_OP_N,n,n,n,&one,dx,n,tmp,n,&zero,da,n,&status)==1,"forward second product handled");check(status);
    check(hipMemcpy(out.data(),da,n*n*sizeof(T),hipMemcpyDeviceToHost));
    std::vector<std::complex<long double>> first(n*n);
    for(int j=0;j<n;++j) for(int i=0;i<n;++i) for(int p=0;p<n;++p) first[i+j*n]+=wide(a[i+p*n])*wide(x[p+j*n]);
    long double error=0,scale=0;
    for(int j=0;j<n;++j) for(int i=0;i<n;++i) {
        std::complex<long double> exact=0;for(int p=0;p<n;++p) exact+=std::conj(wide(x[p+i*n]))*first[p+j*n];
        error=std::max(error,std::abs(wide(out[i+j*n])-exact));scale=std::max(scale,std::abs(exact));
    }
    const double relative=double(error/std::max(1.L,scale));require(relative<1e-11,"full X^T/H H X forward pair");worst=std::max(worst,relative);
    check(hipFree(da));check(hipFree(dx));check(hipFree(tmp));++cases;
}
int main(int argc,char **argv) {
    int ngpu=2;for(int i=1;i<argc;++i) {
        if(std::strcmp(argv[i],"--ngpu")==0 && i+1<argc) ngpu=std::atoi(argv[++i]);else {std::fprintf(stderr,"usage: %s [--ngpu N]\n",argv[0]);return 2;}
    }
    int count=0;const auto query=hipGetDeviceCount(&count);
    if(query!=hipSuccess || count<ngpu || ngpu<2) {std::puts("SKIP: need at least two physical HIP GPUs");return 77;}
    std::vector<int> devices;for(int i=0;i<ngpu;++i) devices.push_back(i);
    check(hipSetDevice(0));hipStream_t stream;check(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking));
    hipblasHandle_t h;check(hipblasCreate(&h));check(hipblasSetStream(h,stream));
    setenv("OPENMX_MGPU_GEMM_WORKSPACE_MB","1",1);setenv("OPENMX_MGPU_GEMM_HOST_MB","1",1);
    for(int host:{0,1}) {
        if(host)setenv("OPENMX_MGPU_GEMM_FORCE_HOST","1",1);else unsetenv("OPENMX_MGPU_GEMM_FORCE_HOST");
        for(auto ta:{HIPBLAS_OP_N,HIPBLAS_OP_T,HIPBLAS_OP_C}) for(auto tb:{HIPBLAS_OP_N,HIPBLAS_OP_T,HIPBLAS_OP_C}) {
            run<double>(h,devices,ta,tb,65,47,71);run<hipDoubleComplex>(h,devices,ta,tb,65,47,71);
        }
        run<double>(h,devices,HIPBLAS_OP_T,HIPBLAS_OP_N,337,223,311);
        run<hipDoubleComplex>(h,devices,HIPBLAS_OP_C,HIPBLAS_OP_T,337,223,311);
        run<double>(h,devices,HIPBLAS_OP_N,HIPBLAS_OP_T,3,5,17001);
        run<hipDoubleComplex>(h,devices,HIPBLAS_OP_C,HIPBLAS_OP_N,3,5,17001);
        run<double>(h,devices,HIPBLAS_OP_N,HIPBLAS_OP_N,17,1,31,true,true);
        run<hipDoubleComplex>(h,devices,HIPBLAS_OP_C,HIPBLAS_OP_C,17,1,31,true,true);
        forward<double>(h,devices);forward<hipDoubleComplex>(h,devices);
        run<double>(h,devices,HIPBLAS_OP_T,HIPBLAS_OP_T,47,97,97);
        run<hipDoubleComplex>(h,devices,HIPBLAS_OP_T,HIPBLAS_OP_T,47,97,97);
    }
    unsetenv("OPENMX_MGPU_GEMM_FORCE_HOST");
    setenv("OPENMX_MGPU_GEMM_RESULT_MB","0",1);
    // A device transaction exceeding its budget now uses host scratch.
    for(auto ta:{HIPBLAS_OP_N,HIPBLAS_OP_T,HIPBLAS_OP_C}) for(auto tb:{HIPBLAS_OP_N,HIPBLAS_OP_T,HIPBLAS_OP_C}) {
        run<double>(h,devices,ta,tb,65,47,71);
        run<hipDoubleComplex>(h,devices,ta,tb,65,47,71,true,true);
    }
#ifdef OPENMX_MGPU_TEST_BRIDGE
    fail_bridge_at=bridge_entries+ngpu+1;
    run<double>(h,devices,HIPBLAS_OP_T,HIPBLAS_OP_N,337,223,311,false,false,true);
    fail_bridge_at=0;
#endif
    setenv("OPENMX_MGPU_GEMM_HOST_RESULT_MB","0",1);
    run<double>(h,devices,HIPBLAS_OP_N,HIPBLAS_OP_N,65,47,71,false,false,true);
    run<hipDoubleComplex>(h,devices,HIPBLAS_OP_C,HIPBLAS_OP_T,65,47,71,false,false,true);
    unsetenv("OPENMX_MGPU_GEMM_HOST_RESULT_MB");
    unsetenv("OPENMX_MGPU_GEMM_RESULT_MB");setenv("OPENMX_MGPU_GEMM_WORKSPACE_MB","0",1);
    run<double>(h,devices,HIPBLAS_OP_N,HIPBLAS_OP_N,65,47,71,false,false,true);
    setenv("OPENMX_MGPU_GEMM_WORKSPACE_MB","1",1);
    auto duplicate=devices;duplicate[1]=duplicate[0];
    run<double>(h,duplicate,HIPBLAS_OP_N,HIPBLAS_OP_N,65,47,71,false,false,true);
    setenv("OPENMX_MGPU_GEMM_RESERVE_MB","1048576",1);
    run<double>(h,devices,HIPBLAS_OP_N,HIPBLAS_OP_N,65,47,71,false,false,true);
    unsetenv("OPENMX_MGPU_GEMM_RESERVE_MB");
#ifdef OPENMX_MGPU_TEST_BRIDGE
    const char *backend=std::getenv("OPENMX_MGPU_GEMM_BACKEND");
    if(!(backend && std::strcmp(backend,"native")==0)) {
        // One full wave already gathered into transaction scratch, then the
        // bridge rejects a later tile. User C must still be byte-identical.
        fail_bridge_at=bridge_entries+ngpu+1;
        run<double>(h,devices,HIPBLAS_OP_T,HIPBLAS_OP_N,337,223,311,false,false,true);
        fail_bridge_at=0;
    }
#endif
    host_operands=true;
    for(auto ta:{HIPBLAS_OP_N,HIPBLAS_OP_T,HIPBLAS_OP_C}) for(auto tb:{HIPBLAS_OP_N,HIPBLAS_OP_T,HIPBLAS_OP_C}) {
        run<double>(h,devices,ta,tb,65,47,71);
        run<hipDoubleComplex>(h,devices,ta,tb,65,47,71,false,true);
    }
    run<hipDoubleComplex>(h,devices,HIPBLAS_OP_T,HIPBLAS_OP_T,3,5,17001);
    if(count>=4) {
        std::vector<int> nonprefix={2,3};
        run<double>(h,nonprefix,HIPBLAS_OP_N,HIPBLAS_OP_T,65,47,71);
        run<hipDoubleComplex>(h,nonprefix,HIPBLAS_OP_C,HIPBLAS_OP_N,65,47,71);
    }
    setenv("OPENMX_MGPU_GEMM_HOST_RESULT_MB","0",1);
    run<double>(h,devices,HIPBLAS_OP_T,HIPBLAS_OP_N,337,223,311,false,false,true);
    unsetenv("OPENMX_MGPU_GEMM_HOST_RESULT_MB");
#ifdef OPENMX_MGPU_TEST_BRIDGE
    fail_bridge_at=bridge_entries+ngpu+1;
    run<double>(h,devices,HIPBLAS_OP_T,HIPBLAS_OP_N,337,223,311,false,false,true);
    fail_bridge_at=0;
#endif
    host_operands=false;
    // Source GPU need not be device zero; repeat after restoring a different handle/stream.
    check(hipblasDestroy(h));check(hipStreamDestroy(stream));check(hipSetDevice(1));
    check(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking));check(hipblasCreate(&h));check(hipblasSetStream(h,stream));
    run<double>(h,devices,HIPBLAS_OP_N,HIPBLAS_OP_C,65,47,71);
    run<hipDoubleComplex>(h,devices,HIPBLAS_OP_C,HIPBLAS_OP_N,65,47,71);
    check(hipblasDestroy(h));check(hipStreamDestroy(stream));openmx_mgpuGemmRelease();
    int restored=-1;check(hipGetDevice(&restored));require(restored==1,"release restores current device");
    std::printf("PASS: %d cases on %d GPUs; N/T/C, real/complex, alpha/beta, device scalars, padding, streams, full K tiles, forward/back, peer/host, transactional decline; max error %.3e; bridge calls %d\n",cases,ngpu,worst,bridge_entries);
}

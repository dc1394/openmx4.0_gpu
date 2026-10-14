/* Exercise the actual OpenMX cuSOLVER hooks, including first-call tuning and
   cached calls. A=P H diag(lambda) H P^H has an independent O(n*nev) residual
   oracle, where H=I-2*11^T/n and P is a complex diagonal phase matrix. */
#include "openmx_cusolver_compat.h"
#include "openmx_mgpu_runtime.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK_CUDA(call) do { cudaError_t e=(call); if(e!=cudaSuccess) { \
    std::fprintf(stderr,"%s: %s\n",#call,cudaGetErrorString(e)); std::exit(1); } } while(0)
#define CHECK_SOLVER(call) do { cusolverStatus_t e=(call); if(e!=CUSOLVER_STATUS_SUCCESS) { \
    std::fprintf(stderr,"%s: status %d\n",#call,int(e)); std::exit(1); } } while(0)
template<class T> T conjv(T x) { return x; }
template<> std::complex<double> conjv(std::complex<double> x) { return std::conj(x); }
template<class T> T phase(int) { return T(1); }
template<> std::complex<double> phase(int i) { return std::polar(1.0,0.37*i); }

template<class T> int run(int ngpu,int n,int nev,int upper,int repeat,int full_api)
{
    const int lda=n+11;
    const auto type=sizeof(T)==8?CUDA_R_64F:CUDA_C_64F;
    const auto uplo=upper?CUBLAS_FILL_MODE_UPPER:CUBLAS_FILL_MODE_LOWER;
    std::vector<T> a(size_t(lda)*n,T(-777)),q(a.size()),p(n);
    std::vector<double> d(n),w(n+1);
    double sumd=0;
    for(int i=0;i<n;++i) { d[i]=1+2.0*i/(n-1); sumd+=d[i];p[i]=phase<T>(i); }
    const double correction=4*sumd/(double(n)*n);
    for(int j=0;j<n;++j) for(int i=0;i<n;++i)
        a[size_t(j)*lda+i]=p[i]*conjv(p[j])*((i==j?d[i]:0)-2*(d[i]+d[j])/n+correction);
    CHECK_CUDA(cudaSetDevice(ngpu-1));
    cusolverDnHandle_t h; cudaStream_t stream;
    CHECK_SOLVER(cusolverDnCreate(&h));
    CHECK_CUDA(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    CHECK_SOLVER(cusolverDnSetStream(h,stream));
    void *da,*work; double *dw; int *di;
    CHECK_CUDA(cudaMalloc(&da,a.size()*sizeof(T)));
    CHECK_CUDA(cudaMalloc((void**)&dw,w.size()*sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&di,sizeof(int)));
    size_t db=0,hb=0; double lo=0,hi=0; int64_t meig=0;
    if(full_api) CHECK_SOLVER(cusolverDnXsyevd_bufferSize(h,nullptr,CUSOLVER_EIG_MODE_VECTOR,
        uplo,n,type,da,lda,CUDA_R_64F,dw,type,&db,&hb));
    else CHECK_SOLVER(cusolverDnXsyevdx_bufferSize(h,nullptr,CUSOLVER_EIG_MODE_VECTOR,
        CUSOLVER_EIG_RANGE_I,uplo,n,type,da,lda,&lo,&hi,1,nev,&meig,CUDA_R_64F,dw,type,&db,&hb));
    CHECK_CUDA(cudaMalloc(&work,std::max(size_t(1),db)));
    std::vector<char> host(hb);
    for(int r=0;r<repeat;++r) {
        w[n]=-555;
        CHECK_CUDA(cudaMemcpyAsync(da,a.data(),a.size()*sizeof(T),cudaMemcpyHostToDevice,stream));
        CHECK_CUDA(cudaMemcpyAsync(dw+n,w.data()+n,sizeof(double),cudaMemcpyHostToDevice,stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
        const auto start=std::chrono::steady_clock::now();
        if(full_api) {
            CHECK_SOLVER(cusolverDnXsyevd(h,nullptr,CUSOLVER_EIG_MODE_VECTOR,uplo,n,type,da,lda,
                CUDA_R_64F,dw,type,work,db,host.data(),hb,di)); meig=n;
        } else CHECK_SOLVER(cusolverDnXsyevdx(h,nullptr,CUSOLVER_EIG_MODE_VECTOR,
            CUSOLVER_EIG_RANGE_I,uplo,n,type,da,lda,&lo,&hi,1,nev,&meig,CUDA_R_64F,dw,type,
            work,db,host.data(),hb,di));
        CHECK_CUDA(cudaStreamSynchronize(stream));
        const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        int info=-1,current=-1;
        CHECK_CUDA(cudaGetDevice(&current));
        CHECK_CUDA(cudaMemcpy(&info,di,sizeof(int),cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(q.data(),da,q.size()*sizeof(T),cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(w.data(),dw,w.size()*sizeof(double),cudaMemcpyDeviceToHost));
        if(info || meig!=nev || current!=ngpu-1 || w[n]!=-555) {
            std::fprintf(stderr,"FAIL metadata info=%d meig=%lld device=%d guard=%g\n",info,(long long)meig,current,w[n]);
            return 1;
        }
        double eval_error=0,residual=0,orth=0;
        for(int j=0;j<nev;++j) {
            eval_error=std::max(eval_error,std::abs(w[j]-d[j]));
            T sum=0,weighted=0;
            for(int i=0;i<n;++i) { T x=conjv(p[i])*q[size_t(j)*lda+i];sum+=x;weighted+=d[i]*x; }
            double r2=0;
            for(int i=0;i<n;++i) {
                T x=conjv(p[i])*q[size_t(j)*lda+i];
                T ri=(d[i]-w[j])*x-2.0/n*(d[i]*sum+weighted)+correction*sum;
                r2+=std::abs(ri)*std::abs(ri);
            }
            residual=std::max(residual,std::sqrt(r2)/(3+std::abs(w[j])));
            for(int k: {j,std::max(0,j-1),nev-1}) {
                T dot=0;
                for(int i=0;i<n;++i) dot+=conjv(q[size_t(k)*lda+i])*q[size_t(j)*lda+i];
                orth=std::max(orth,std::abs(dot-T(j==k?1:0)));
            }
        }
        /* Native cuSOLVER may use lda padding as scratch; verify that the
           strided eigenvector columns themselves are correct below. The
           direct MAGMA/paper fixtures check their tighter padding contract. */
        const bool pass=eval_error<1e-10 && residual<1e-10 && orth<1e-10;
        std::printf("%s backend=%s type=%c ngpu=%d n=%d nev=%d upper=%d full_api=%d repeat=%d seconds=%.6f eval=%.3e residual=%.3e orth=%.3e\n",
            pass?"PASS":"FAIL",std::getenv("OPENMX_MGPU_SOLVER")?std::getenv("OPENMX_MGPU_SOLVER"):"auto",
            sizeof(T)==8?'D':'Z',ngpu,n,nev,upper,full_api,r,seconds,eval_error,residual,orth);
        if(!pass) return 1;
    }
    CHECK_CUDA(cudaFree(work));CHECK_CUDA(cudaFree(di));CHECK_CUDA(cudaFree(dw));CHECK_CUDA(cudaFree(da));
    CHECK_SOLVER(cusolverDnDestroy(h));CHECK_CUDA(cudaStreamDestroy(stream));
    return 0;
}
int main(int argc,char**argv)
{
    int ngpu=2,n=257,nev=0,upper=0,repeat=2,complex=0,full_api=0,enabled=1,scope=1,single_node=1;
    for(int i=1;i<argc;++i) {
        if(!std::strcmp(argv[i],"--complex")) complex=1;
        else if(!std::strcmp(argv[i],"--upper")) upper=1;
        else if(!std::strcmp(argv[i],"--full-api")) full_api=1;
        else if(!std::strcmp(argv[i],"--disabled")) enabled=0;
        else if(!std::strcmp(argv[i],"--outside-cluster")) scope=0;
        else if(!std::strcmp(argv[i],"--multi-node")) single_node=0;
        else if(i+1<argc) {
            if(!std::strcmp(argv[i],"--n")) n=std::atoi(argv[++i]);
            else if(!std::strcmp(argv[i],"--nev")) nev=std::atoi(argv[++i]);
            else if(!std::strcmp(argv[i],"--ngpu")) ngpu=std::atoi(argv[++i]);
            else if(!std::strcmp(argv[i],"--repeat")) repeat=std::atoi(argv[++i]);
            else if(!std::strcmp(argv[i],"--backend")) setenv("OPENMX_MGPU_SOLVER",argv[++i],1);
            else return 2;
        } else return 2;
    }
    if(!nev) nev=n;
    if(n<2 || n>16384 || nev<1 || nev>n || ngpu<1 || repeat<1 || (full_api && nev!=n)) return 2;
    int devices=0;CHECK_CUDA(cudaGetDeviceCount(&devices)); if(devices<ngpu) return 2;
    setvbuf(stdout,nullptr,_IOLBF,0);
    openmx_mgpu_configure(enabled,ngpu,single_node);
    int previous=openmx_mgpu_scope_enter(scope);
    int result=complex?run<std::complex<double>>(ngpu,n,nev,upper,repeat,full_api):run<double>(ngpu,n,nev,upper,repeat,full_api);
    openmx_mgpu_scope_leave(previous);
    return result;
}

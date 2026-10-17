/* Independent dense Householder similarity oracle for the real dependency.
   A=P H diag(lambda) H P^H, H=I-2*11^T/n; P supplies complex phases.
   Eigenvalues and full residuals are analytic without a second eigensolver. */
#include "openmx_magma_mgpu.h"
#include "hip_runtime_compat.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

template<class T> T phase(int) { return T(1); }
template<> std::complex<double> phase(int i) { return std::polar(1.0,0.37*i); }
template<class T> T conjugate(T x) { return x; }
template<> std::complex<double> conjugate(std::complex<double> x) { return std::conj(x); }

template<class T> int run(int ngpu,const int *devices,int n,int first,int last,int vectors,int lower)
{
    const int lda=n+11,count=last-first+1;
    std::vector<double> diagonal(n);
    std::vector<T> phases(n);
    double sum_diagonal=0;
    for(int i=0;i<n;++i) {
        diagonal[i]=1+2.0*i/(n-1);
        sum_diagonal+=diagonal[i];
        phases[i]=phase<T>(i);
    }
    const double correction=4*sum_diagonal/(double(n)*n);
    std::vector<T>a(size_t(lda)*n,T(-777));
    for(int j=0;j<n;++j) for(int i=0;i<n;++i) {
        T value=phases[i]*conjugate(phases[j])*
            ((i==j?diagonal[i]:0)-2*(diagonal[i]+diagonal[j])/n+correction);
        if(i!=j && (i>j)!=bool(lower)) value=T(std::numeric_limits<double>::quiet_NaN());
        a[size_t(j)*lda+i]=value;
    }
    const auto original=a;
    std::vector<double>w(count+7,-555);
    int original_device=-1;
    if(hipGetDevice(&original_device)!=hipSuccess) return 1;
    const auto start=std::chrono::steady_clock::now();
    const int status=devices?
        openmx_magma_mgpu_solve_host_devices(ngpu,devices,sizeof(T)!=8,vectors,lower,n,first,last,a.data(),lda,w.data()):
        openmx_magma_mgpu_solve_host(ngpu,sizeof(T)!=8,vectors,lower,n,first,last,a.data(),lda,w.data());
    const double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    if(status) {
        std::fprintf(stderr,"MAGMA failure %d: %s\n",status,openmx_magma_mgpu_last_error());return 1;
    }
    int current=-1;
    if(hipGetDevice(&current)!=hipSuccess || current!=original_device) return 1;
    double eigen_error=0,residual=0,orthogonality=0;
    for(int j=0;j<count;++j) {
        const double expected=diagonal[first+j-1];
        eigen_error=std::max(eigen_error,std::abs(w[j]-expected));
        if(vectors) {
            double r2=0;
            T sum=0,weighted_sum=0;
            for(int i=0;i<n;++i) {
                const T x=conjugate(phases[i])*a[size_t(j)*lda+i];
                sum+=x;
                weighted_sum+=diagonal[i]*x;
            }
            for(int i=0;i<n;++i) {
                const T x=conjugate(phases[i])*a[size_t(j)*lda+i];
                const T r=(diagonal[i]-w[j])*x-2.0/n*(diagonal[i]*sum+weighted_sum)+correction*sum;
                r2+=std::abs(r)*std::abs(r);
            }
            residual=std::max(residual,std::sqrt(r2)/(3+std::abs(w[j])));
            // Full norms and orthogonality against nearby and distant columns.
            for(int k: {j,std::max(0,j-1),count-1}) {
                T dot=0;
                for(int i=0;i<n;++i) dot+=conjugate(a[size_t(k)*lda+i])*a[size_t(j)*lda+i];
                orthogonality=std::max(orthogonality,std::abs(dot-T(j==k?1:0)));
            }
        }
    }
    for(int j=count;j<int(w.size());++j) if(w[j]!=-555) return 1;
    for(int j=0;j<n;++j) for(int i=0;i<lda;++i)
        if((!vectors || j>=count || i>=n) && std::memcmp(&a[size_t(j)*lda+i],&original[size_t(j)*lda+i],sizeof(T))) return 1;
    const bool pass=eigen_error<1e-10 && residual<1e-10 && orthogonality<1e-10;
    std::string group;
    for(int i=0;i<ngpu;++i) group+=(i?",":"")+std::to_string(devices?devices[i]:i);
    std::printf("%c ngpu=%d devices=%s owner=%d n=%d range=%d:%d vectors=%d lower=%d seconds=%.6f eigen_error=%.3e residual=%.3e orth=%.3e %s\n",
        sizeof(T)==8?'D':'Z',ngpu,group.c_str(),original_device,n,first,last,vectors,lower,elapsed,eigen_error,residual,orthogonality,pass?"PASS":"FAIL");
    return pass?0:1;
}

int main(int argc,char **argv)
{
    const int ngpu=argc>1?std::atoi(argv[1]):2;
    const int n=argc>2?std::atoi(argv[2]):257;
    if(ngpu<2 || ngpu>8 || n<32 || n>16000 || !openmx_magma_mgpu_available()) return 2;
    int available=0;
    if(hipGetDeviceCount(&available)!=hipSuccess || available<ngpu) return 2;
    // Optional comma-separated physical ordinals and independent data owner:
    //   magma_mgpu_smoke 2 257 2,3 0
    //   magma_mgpu_smoke 2 257 3,2 1
    //   magma_mgpu_smoke 2 257 1,3 0
    std::vector<int> devices;
    if(argc>3) {
        const char *p=argv[3];
        while(*p) {
            char *end=nullptr;const long value=std::strtol(p,&end,10);
            if(end==p || value<0 || value>=available || (*end && *end!=',')) return 2;
            if(std::find(devices.begin(),devices.end(),int(value))!=devices.end()) return 2;
            devices.push_back(int(value));
            if(*end && !end[1]) return 2;
            p=*end?end+1:end;
        }
        if(devices.size()!=size_t(ngpu)) return 2;
    }
    const int owner=argc>4?std::atoi(argv[4]):ngpu-1;
    if(owner<0 || owner>=available || hipSetDevice(owner)!=hipSuccess) return 2;
    const int *group=devices.empty()?nullptr:devices.data();
    for(int vectors=0;vectors<2;++vectors) for(int lower=0;lower<2;++lower) {
        if(run<double>(ngpu,group,n,1,n,vectors,lower) ||
           run<std::complex<double>>(ngpu,group,n,1,n,vectors,lower) ||
           run<double>(ngpu,group,n,3,19,vectors,lower) ||
           run<std::complex<double>>(ngpu,group,n,3,19,vectors,lower)) return 1;
    }
    std::puts("PASS: real/complex, full/subset, upper/lower, vectors/values-only MAGMA multi-GPU solves");
}

#include "openmx_paper_mgpu.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>
extern "C" void dsyev_(const char*,const char*,const int*,double*,const int*,double*,double*,const int*,int*);

static bool test(int ngpu,int n,int nev,const char *kind) {
    int lda=n+3;std::vector<double>a(size_t(lda)*n,-123.25),ref(size_t(n)*n),orig(size_t(n)*n),w(n,-999),wr(n);
    std::vector<double>v(n),d(n);double vv=0,vdv=0;
    for(int i=0;i<n;++i){v[i]=std::sin(i+1.23);vv+=v[i]*v[i];}
    for(int i=0;i<n;++i){v[i]/=std::sqrt(vv);double t=n>1?double(i)/(n-1):0;
        if(!std::strcmp(kind,"geometric"))d[i]=1e6*std::pow(1e-8,1-t);
        else if(!std::strcmp(kind,"cluster0"))d[i]=i==n-1?1e6:.01;
        else if(!std::strcmp(kind,"cluster1"))d[i]=i==0?.01:1e6;
        else d[i]=.01+(1e6-.01)*t;
        vdv+=v[i]*v[i]*d[i];}
    bool prescribed=std::strcmp(kind,"normal")!=0;
    for(int c=0;c<n;++c)for(int r=0;r<n;++r){
        double x=prescribed?((r==c?d[r]:0)-2*v[r]*v[c]*(d[r]+d[c])+4*v[r]*v[c]*vdv):
            std::sin((r+1.)*(c+1.)*.731)+std::cos((r+c+2.)*.337)+(r==c?0.125*r:0);
        orig[r+size_t(c)*n]=ref[r+size_t(c)*n]=x;
        a[r+size_t(c)*lda]=r>=c?x:std::numeric_limits<double>::quiet_NaN();
    }
    char job='N',uplo='L';int info=0,lwork=-1;double query=0;
    dsyev_(&job,&uplo,&n,ref.data(),&n,wr.data(),&query,&lwork,&info);lwork=std::max(1,int(query));std::vector<double>work(lwork);
    dsyev_(&job,&uplo,&n,ref.data(),&n,wr.data(),work.data(),&lwork,&info);
    if(info){std::fprintf(stderr,"reference DSYEV failed %d\n",info);return false;}
    auto begin=std::chrono::steady_clock::now();int status=openmx_paper_mgpu_dsyevdx_host(ngpu,n,nev,a.data(),lda,w.data());
    double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
    if(status){std::printf("FAIL n=%d nev=%d type=%s status=%d %s\n",n,nev,kind,status,openmx_paper_mgpu_last_error());return false;}
    long double norm=0,res=0,orth=0;double eval=0;
    for(double x:orig)norm+=(long double)x*x;
    for(int c=0;c<nev;++c)for(int r=0;r<n;++r){long double z=0;
        for(int k=0;k<n;++k)z+=(long double)orig[r+size_t(k)*n]*a[k+size_t(c)*lda];
        z-=(long double)w[c]*a[r+size_t(c)*lda];res+=z*z;}
    for(int c=0;c<nev;++c)for(int r=0;r<nev;++r){long double z=r==c?-1:0;
        for(int k=0;k<n;++k)z+=(long double)a[k+size_t(c)*lda]*a[k+size_t(r)*lda];orth+=z*z;}
    for(int i=0;i<nev;++i)eval=std::max(eval,std::abs(w[i]-wr[i])/std::max(1.0,std::abs(wr[i])));
    double rn=std::sqrt(double(res/norm))/n,on=std::sqrt(double(orth))/n;
    bool ok=rn<2e-14 && on<2e-14 && eval<5e-7;
    for(int i=nev;i<n;++i)if(w[i]!=-999)ok=false;
    for(int c=0;c<n;++c)for(int r=n;r<lda;++r)if(a[r+size_t(c)*lda]!=-123.25)ok=false;
    for(int c=nev;c<n;++c)for(int r=c;r<n;++r)if(a[r+size_t(c)*lda]!=orig[r+size_t(c)*n])ok=false;
    std::printf("%s ngpu=%d n=%d nev=%d type=%s residual=%.3e orth=%.3e eval_rel=%.3e solver_ms=%.3f\n",ok?"PASS":"FAIL",ngpu,n,nev,kind,rn,on,eval,ms);
    return ok;
}
static bool test_failures(int ngpu) {
    const int n=129,nev=73,lda=n+3;
    using Value=double;
    std::vector<Value> original(size_t(lda)*n,Value(1.25));
    for(int c=0;c<n;++c)for(int r=c;r<n;++r)
        original[r+size_t(c)*lda]=Value(r==c?2.0+c:0.01*std::sin(r+c));
    for(const char *stage:{"thread","sbr","bc","back","final"}) {
        std::vector<Value>a=original;std::vector<double>w(nev,-777);
        setenv("OPENMX_PAPER_TEST_FAIL",stage,1);
        int code=openmx_paper_mgpu_dsyevdx_host(ngpu,n,nev,a.data(),lda,w.data());
        unsetenv("OPENMX_PAPER_TEST_FAIL");
        bool ok=code!=0 && a==original && std::all_of(w.begin(),w.end(),[](double x){return x==-777;});
        std::printf("%s real failure_stage=%s unchanged=%d error=%s\n",ok?"PASS":"FAIL",stage,int(ok),openmx_paper_mgpu_last_error());
        if(!ok)return false;
    }
    return test(ngpu,n,nev,"normal");
}
int main(int argc,char **argv) {
    int ngpu=argc>1?std::atoi(argv[1]):2;
    if(openmx_paper_mgpu_available()<ngpu){std::fprintf(stderr,"Need %d visible GPUs\n",ngpu);return 77;}
    bool ok=true;
    if(argc>2 && !std::strcmp(argv[2],"--failures"))return test_failures(ngpu)?0:1;
    if(argc>2){int n=std::atoi(argv[2]);int nev=argc>3?std::atoi(argv[3]):n;ok=test(ngpu,n,nev,argc>4?argv[4]:"normal");}
    else {
        for(int n:{1,2,17,31,32,33,63,64,65,97,128,257})ok=test(ngpu,n,n,"normal")&&ok;
        for(const char *kind:{"cluster0","cluster1","geometric","arithmetic"})ok=test(ngpu,129,73,kind)&&ok;
    }
    double a=2,w=-1;int rc=openmx_paper_mgpu_dsyevdx_host(ngpu,1,2,&a,1,&w);
    if(!rc || a!=2 || w!=-1){std::fprintf(stderr,"Invalid-argument transaction test failed\n");ok=false;}
    return ok?0:1;
}

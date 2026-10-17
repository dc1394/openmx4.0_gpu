#include "openmx_gpu_transform.h"
#include "openmx_gpu_dm.h"
#include "openmx_gpu_workers.h"
#include "hip_runtime_compat.h"
#include <omp.h>
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <vector>
extern "C" {extern int probe_gpus,probe_first,probe_count;}
using Z=std::complex<double>;
using L=std::complex<long double>;
static int cases=0;
static double worst=0;
static void require(bool yes,const char *what){if(!yes){std::fprintf(stderr,"FAIL: %s (case %d)\n",what,cases);std::exit(1);}}
static void check(hipError_t e){if(e!=hipSuccess){std::fprintf(stderr,"HIP: %s\n",hipGetErrorString(e));std::exit(1);}}
template<class T> struct Device {
    T *p=nullptr;size_t size;
    explicit Device(const std::vector<T>& v):size(v.size()){
        check(hipMalloc((void**)&p,size*sizeof(T)));check(hipMemcpy(p,v.data(),size*sizeof(T),hipMemcpyHostToDevice));
    }
    ~Device(){check(hipFree(p));}
    std::vector<T> read(){std::vector<T> v(size);check(hipMemcpy(v.data(),p,size*sizeof(T),hipMemcpyDeviceToHost));return v;}
};
template<class T> static T sample(int i){return T(std::sin(i*.713)*.7+.2);}
template<> Z sample<Z>(int i){return {std::sin(i*.713)*.7+.2,std::cos(i*.317)*.4};}
template<class T> static T conjugate(T v){return v;}
template<> Z conjugate(Z v){return std::conj(v);}
template<class T> static double real_part(T v){return v;}
template<> double real_part(Z v){return v.real();}
template<class T> static void compare(const std::vector<T>& a,const std::vector<T>& b,const char *name){
    require(a.size()==b.size(),"comparison size");
    for(size_t i=0;i<a.size();i++){
        const double error=std::abs(a[i]-b[i]);worst=std::max(worst,error);
        if(!(error<=2e-11*(1+std::abs(b[i])))){
            std::fprintf(stderr,"%s index=%zu error=%.17g\n",name,i,error);require(false,name);
        }
    }
    ++cases;
}
template<class T> static void transform(int op,bool alias,bool complex_weights){
    const bool cpx=sizeof(T)==sizeof(Z);
    const int rows=op==OPENMX_GPU_INVSQRT?1:13;
    const int cols=op==OPENMX_GPU_HERMITIAN || alias?rows:19;
    const int ar=op==OPENMX_GPU_TRANSPOSE?cols:rows,ac=op==OPENMX_GPU_TRANSPOSE?rows:cols;
    const int lda=ar+3,ldb=rows+4,ldout=alias?lda:rows+5;
    std::vector<T> a(2+size_t(lda)*ac),b(2+size_t(ldb)*cols),out(2+size_t(ldout)*cols,T(91));
    for(size_t i=0;i<a.size();i++)a[i]=sample<T>(int(i));
    for(size_t i=0;i<b.size();i++)b[i]=sample<T>(int(i)+101);
    std::vector<double> weights(std::max(rows,cols));
    for(size_t i=0;i<weights.size();i++)weights[i]=.1+i*.03;
    std::vector<T> expected=alias?a:out;
    for(int j=0;j<cols;j++)for(int i=0;i<rows;i++){
        T v=op==OPENMX_GPU_ZERO?T(0):a[1+(op==OPENMX_GPU_TRANSPOSE?j+i*lda:i+j*lda)];
        if(op==OPENMX_GPU_ADD)v+=b[1+i+j*ldb];
        if(op==OPENMX_GPU_SCALE_ROWS || op==OPENMX_GPU_SCALE_COLUMNS){
            int w=op==OPENMX_GPU_SCALE_ROWS?i:j;
            v*=complex_weights?b[1+w]:T(weights[w]);
        }
        if(op==OPENMX_GPU_HERMITIAN)v=i==j?T(real_part(v)):(v+conjugate(a[1+j+i*lda]))*.5;
        if(op==OPENMX_GPU_INVSQRT)v=T(1/std::sqrt(std::max(.03,real_part(v))));
        expected[1+i+j*ldout]=v;
    }
    Device<T> da(a),db(b),dout(out);Device<double> dw(weights);
    void *target=alias?da.p+1:dout.p+1;
    const void *second=op==OPENMX_GPU_ADD || complex_weights?(void*)(db.p+1):(void*)dw.p;
    require(openmx_gpu_transform(op,cpx,rows,cols,da.p+1,lda,second,ldb,target,ldout,complex_weights,.03)==1,"transform handled");
    compare(alias?da.read():dout.read(),expected,"transform");
}
static void spinor(bool real){
    int n=7;std::vector<double> a[6];std::vector<Z> z[3];
    for(int k=0;k<6;k++){a[k].resize(n*n);for(int i=0;i<n*n;i++)a[k][i]=sample<double>(i+k*31);}
    for(int k=0;k<3;k++){z[k].resize(n*n);for(int i=0;i<n*n;i++)z[k][i]={a[k][i],a[k+3][i]};}
    Device<double> a0(a[0]),a1(a[1]),a2(a[2]),a3(a[3]),a4(a[4]),a5(a[5]);
    Device<Z> z0(z[0]),z1(z[1]),z2(z[2]);
    std::vector<Z> out(4*n*n),expected(out.size());Device<Z> d(out);
    for(int j=0;j<2*n;j++)for(int i=0;i<2*n;i++){
        int block=i<n?(j<n?0:2):(j<n?2:1);
        expected[i+j*2*n]=i>=n && j<n?std::conj(z[block][j+(i-n)*n]):z[block][i%n+(j%n)*n];
    }
    require(openmx_gpu_spinor(n,real,real?(void*)a0.p:(void*)z0.p,real?(void*)a1.p:(void*)z1.p,
        real?(void*)a2.p:(void*)z2.p,a3.p,a4.p,a5.p,d.p)==1,"spinor handled");
    compare(d.read(),expected,"spinor");
}
static void scatter(int format,bool imaginary){
    int n=11,count=317,phases=7,step=format?format:1;
    std::vector<int> index(count*step);std::vector<double> values(count),pr(phases),pi(phases);
    std::vector<Z> expected(n*n);std::vector<double> real(n*n);
    for(int i=0;i<phases;i++){pr[i]=std::cos(i*.2);pi[i]=std::sin(i*.2);}
    for(int i=0;i<count;i++){
        values[i]=sample<double>(i);int target=(i*37)%(n*n),ph=i%phases;
        if(!format){index[i]=target;real[target]+=values[i];continue;}
        int *e=index.data()+i*step;e[0]=i;e[1]=target;e[step-1]=ph;
        Z v=values[i]*Z(pr[ph],pi[ph])*(imaginary?Z(0,1):Z(1));expected[target]+=v;
        if(format==7){e[2]=(i%3)?(i*23)%(n*n):-1;if(e[2]>=0)expected[e[2]]+=std::conj(v);}
    }
    Device<int> di(index);Device<double> dv(values),dr(pr),dp(pi);
    if(format){Device<Z> out(std::vector<Z>(n*n,Z(77)));
        require(openmx_gpu_scatter(n,count,format,di.p,count,dv.p,phases,dr.p,dp.p,imaginary,out.p)==1,"scatter handled");compare(out.read(),expected,"scatter complex");
    }else{Device<double> out(std::vector<double>(n*n,77));
        require(openmx_gpu_scatter(n,count,format,di.p,count,dv.p,0,nullptr,nullptr,0,out.p)==1,"scatter real handled");compare(out.read(),real,"scatter real");}
}
static void density(int kind,bool host,bool phased,bool accumulate,bool energy){
    const int n=7,states=17,stride=23,entries=39,phases=5;
    const bool gather=kind>=OPENMX_GPU_DM_GATHER,isreal=kind==OPENMX_GPU_DM_REAL;
    size_t count=gather?size_t(2*n)*2*n:(size_t(kind==OPENMX_GPU_DM_NC?2*n:n)-1)*stride+states;
    std::vector<Z> v(count);std::vector<double> rv(count),occ(states),ew(states),pw(states),pr(phases),pi(phases);
    for(size_t i=0;i<count;i++){v[i]=sample<Z>(int(i));rv[i]=v[i].real();}
    for(int k=0;k<states;k++){occ[k]=.4+.01*k;ew[k]=sample<double>(k+94);pw[k]=sample<double>(k+231);}
    for(int k=0;k<phases;k++){pr[k]=std::cos(.7*k);pi[k]=std::sin(.7*k);}
    std::vector<int> a(entries),b(entries),ph(entries);std::vector<double> expected[10],out[10];double *outputs[10];
    for(int c=0;c<10;c++){out[c].assign(entries,7+c*.3);expected[c]=out[c];outputs[c]=out[c].data();}
    for(int p=0;p<entries;p++){
        a[p]=(p*3)%n;b[p]=(p*5+1)%n;ph[p]=p%phases;
        long double r[10]={0};L dm[3]{},edm[3]{};
        if(gather){L x=v[a[p]+b[p]*2*n],y=v[a[p]+n+(b[p]+n)*2*n],z=v[a[p]+(b[p]+n)*2*n];
            r[0]=x.real();r[1]=y.real();r[2]=z.real();r[3]=z.imag();
            if(kind==OPENMX_GPU_DM_GATHER){r[4]=x.imag();r[5]=y.imag();}
        }else for(int k=0;k<states;k++){
            if(isreal){long double prod=(long double)rv[a[p]*stride+k]*rv[b[p]*stride+k];
                r[0]+=occ[k]*prod;if(energy)r[1]+=occ[k]*ew[k]*prod;r[2]+=pw[k]*prod;
            }else{
                L va=v[a[p]*stride+k],vb=v[b[p]*stride+k];
                if(kind==OPENMX_GPU_DM_COL){L prod=std::conj(va)*vb*(long double)occ[k]*(long double)occ[k];dm[0]+=prod;if(energy)edm[0]+=prod*(long double)ew[k];}
                else {L vc=v[(a[p]+n)*stride+k],vd=v[(b[p]+n)*stride+k];L prod[3]={std::conj(va)*vb,std::conj(vc)*vd,std::conj(va)*vd};
                    for(int c=0;c<3;c++){dm[c]+=prod[c]*(long double)occ[k];if(energy)edm[c]+=prod[c]*(long double)ew[k];}}
            }
        }
        if(!gather && !isreal){L phase=phased?L(pr[ph[p]],pi[ph[p]]):L(1);
            for(int c=0;c<3;c++){dm[c]*=phase;edm[c]*=phase;}
            if(kind==OPENMX_GPU_DM_COL){r[0]=dm[0].real();r[1]=edm[0].real();}
            else {r[0]=dm[0].real();r[1]=dm[1].real();r[2]=dm[2].real();r[3]=dm[2].imag();r[4]=dm[0].imag();r[5]=dm[1].imag();
                r[6]=edm[0].real();r[7]=edm[1].real();r[8]=edm[2].real();r[9]=edm[2].imag();}
        }
        for(int c=0;c<10;c++)expected[c][p]=(accumulate?out[c][p]:0)+double(r[c]);
    }
    Device<Z> dv(v);Device<double> dr(rv);
    const void *vectors=isreal?(host?(void*)rv.data():(void*)dr.p):(host?(void*)v.data():(void*)dv.p);
    require(openmx_gpu_dm_entries(kind,n,states,stride,entries,vectors,host,a.data(),b.data(),phased?ph.data():nullptr,
        phases,pr.data(),pi.data(),occ.data(),energy?ew.data():nullptr,isreal?pw.data():nullptr,outputs,accumulate)==1,"density handled");
    for(int c=0;c<10;c++)compare(out[c],expected[c],"density channel");
    // A partially prepared group must release its private allocations and
    // leave every host accumulator unchanged when an index is rejected.
    a.back()=n;
    require(openmx_gpu_dm_entries(kind,n,states,stride,entries,vectors,host,a.data(),b.data(),nullptr,
        0,nullptr,nullptr,occ.data(),energy?ew.data():nullptr,nullptr,outputs,1)==0,"density invalid index declined");
    for(int c=0;c<10;c++)compare(out[c],expected[c],"density rollback");
}
int main(){
    int count=0;check(hipGetDeviceCount(&count));if(count<4){std::puts("SKIP: requires four HIP GPUs");return 77;}
    probe_gpus=4;openmx_gpu_workers_configure(0,1,4);
    for(int config=0;config<6;config++){
        probe_first=config%3==2?2:0;probe_count=config%3==0?4:2;
        const int origin=config%2?3:0;
        check(hipSetDevice(origin));omp_set_default_device(origin);
        if(config>=3)setenv("OPENMX_GPU_WORKER_COPY","1",1);else unsetenv("OPENMX_GPU_WORKER_COPY");
        for(int op=OPENMX_GPU_ZERO;op<=OPENMX_GPU_INVSQRT;op++){
            transform<double>(op,false,false);transform<Z>(op,false,false);
            if(op!=OPENMX_GPU_INVSQRT){transform<double>(op,true,false);transform<Z>(op,true,false);}
            if(op==OPENMX_GPU_SCALE_ROWS || op==OPENMX_GPU_SCALE_COLUMNS)transform<Z>(op,false,true);
        }
        spinor(false);spinor(true);scatter(0,false);
        for(int format:{3,7})for(bool imaginary:{false,true})scatter(format,imaginary);
        for(int kind=0;kind<=OPENMX_GPU_EDM_GATHER;kind++)for(bool host:{false,true})for(bool acc:{false,true}){
            bool phase=kind==OPENMX_GPU_DM_COL || kind==OPENMX_GPU_DM_NC;
            density(kind,host,phase,acc,true);
            if(kind<=OPENMX_GPU_DM_NC)density(kind,host,false,acc,false);
        }
        int restored=-1;check(hipGetDevice(&restored));require(restored==origin,"HIP binding restored");
        require(omp_get_default_device()==origin,"OpenMP binding restored");
        std::printf("configuration %d group=%d..%d origin=%d PASS\n",config,probe_first,probe_first+probe_count-1,origin);std::fflush(stdout);
    }
    std::printf("PASS: %d dense worker comparisons; max absolute error %.3e\n",cases,worst);
}

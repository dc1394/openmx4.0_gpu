/* The builder extracts the production OpenACC workers into this translation
   unit. Host references use pair/spin/orbital loops, independent of the
   workers' flattened output cuts. This also catches missing GPU codegen. */
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <openacc.h>
#include <cuda_runtime_api.h>
#include "openmx_gpu_workers.h"
#include "openmx_gpu_views.h"
typedef float Type_Orbs_Grid;
int SpinP_switch;
#include "setham_workers.inc"
extern int probe_gpus, probe_first, probe_count;
static double max_error;
static int comparisons;
static void check(const double *got, const double *want, size_t n)
{
    for (size_t i=0;i<n;i++) {
        double error=fabs(got[i]-want[i]);
        if (!isfinite(got[i]) || error>2e-12*(1+fabs(want[i]))) {
            fprintf(stderr,"mismatch at %zu: %.17g != %.17g\n",i,got[i],want[i]);
            abort();
        }
        if(error>max_error) max_error=error;
    }
    ++comparisons;
}
static void run_case(int pairs,int spins)
{
    enum {CAP=8192};
    int no0[7]={5,1,3,7,2,4,1},no1[7]={7,2,4,1,6,3,1};
    int nolg[7]={9,0,3,8,1,7,2},mn[CAP],nc[CAP];
    size_t mo[7],ho[7],go[7],o0[7],o1[7],mat=0,nh=0,ng=0,n0=0,n1=0;
    double kin[CAP],nl[CAP],vna[CAP],hub[CAP],hole[CAP],potential[4*17];
    double output[CAP],expected[CAP];
    Type_Orbs_Grid phi0[CAP],phi1[CAP];
    SpinP_switch=spins==4?3:spins-1;
    for(int p=0;p<pairs;p++) {
        mo[p]=mat;ho[p]=nh;go[p]=ng;o0[p]=n0;o1[p]=n1;
        mat+=no0[p]*no1[p];nh+=spins*no0[p]*no1[p];
        ng+=nolg[p];n0+=11*no0[p];n1+=nolg[p]*no1[p];
        for(int q=0;q<nolg[p];q++) {mn[go[p]+q]=(q*7+p)%17;nc[go[p]+q]=(q*3+p)%11;}
    }
    assert(nh+2<CAP && ng<CAP && n0<CAP && n1<CAP);
    for(size_t i=0;i<mat;i++) {kin[i]=sin(i+.3);vna[i]=cos(i*.2);}
    for(size_t i=0;i<nh;i++) {nl[i]=cos(i*.4);hub[i]=sin(i*.7);hole[i]=.013*i-.91;}
    for(size_t i=0;i<n0;i++) phi0[i]=(float)sin(i*.3+.7);
    for(size_t i=0;i<n1;i++) phi1[i]=(float)cos(i*.9+.2);
    for(int i=0;i<spins*17;i++) potential[i]=sin(i*.11)-.3;
#define MAP(a,n) assert(acc_copyin(a,(n)*sizeof(*(a)))!=NULL)
#define UNMAP(a,n) acc_delete(a,(n)*sizeof(*(a)))
    MAP(no0,pairs);MAP(no1,pairs);MAP(mo,pairs);MAP(ho,pairs);
    MAP(kin,mat);MAP(nl,nh);MAP(vna,mat);MAP(hub,nh);MAP(hole,nh);
    for(int flags=0;flags<8;flags++) {
        for(size_t i=0;i<nh+2;i++) output[i]=expected[i]=-777.25;
        for(int p=0;p<pairs;p++) for(int s=0;s<spins;s++)
            for(int i=0;i<no0[p];i++) for(int j=0;j<no1[p];j++) {
                size_t a=mo[p]+i*no1[p]+j,b=ho[p]+s*no0[p]*no1[p]+i*no1[p]+j;
                double x=0;
                if(spins!=4 || s<2) x=.7*kin[a]+((flags&1)?1.2*vna[a]:0);
                if(spins!=4 || s<3) x+=-.8*nl[b]+((flags&2)?hub[b]:0)+((flags&4)?hole[b]:0);
                expected[b+1]=x;
            }
        MAP(output,nh+2);
        assert(SetH_Base_Distributed(pairs,spins,!!(flags&2),!!(flags&4),!!(flags&1),
            nh,mat,.7,-.8,1.2,no0,no1,mo,ho,kin,nl,vna,hub,hole,output+1));
        acc_update_self(output,(nh+2)*sizeof(double));UNMAP(output,nh+2);
        check(output,expected,nh+2);
    }
    UNMAP(kin,mat);UNMAP(nl,nh);UNMAP(vna,mat);UNMAP(hub,nh);UNMAP(hole,nh);UNMAP(mo,pairs);
    MAP(nolg,pairs);MAP(mn,ng);MAP(nc,ng);MAP(go,pairs);MAP(o0,pairs);MAP(o1,pairs);
    MAP(phi0,n0);MAP(phi1,n1);MAP(potential,spins*17);
    for(size_t i=0;i<nh+2;i++) output[i]=expected[i]=sin(i*.23)+.123;
    for(int p=0;p<pairs;p++) for(int s=0;s<spins;s++)
        for(int i=0;i<no0[p];i++) for(int j=0;j<no1[p];j++) {
            size_t b=1+ho[p]+s*no0[p]*no1[p]+i*no1[p]+j;
            for(int q=0;q<nolg[p];q++) expected[b]+=.17*potential[s*17+mn[go[p]+q]]*
                (double)phi0[o0[p]+nc[go[p]+q]*no0[p]+i]*(double)phi1[o1[p]+q*no1[p]+j];
        }
    MAP(output,nh+2);
    assert(SetH_Fallback_Distributed(pairs,spins,nh,ng,n0,n1,17,spins*17,.17,
        no0,no1,nolg,mn,nc,ho,go,o0,o1,phi0,phi1,potential,output+1));
    acc_update_self(output,(nh+2)*sizeof(double));UNMAP(output,nh+2);
    check(output,expected,nh+2);
    UNMAP(no0,pairs);UNMAP(no1,pairs);UNMAP(ho,pairs);UNMAP(nolg,pairs);
    UNMAP(mn,ng);UNMAP(nc,ng);UNMAP(go,pairs);UNMAP(o0,pairs);UNMAP(o1,pairs);
    UNMAP(phi0,n0);UNMAP(phi1,n1);UNMAP(potential,spins*17);
}
int main(void)
{
    int visible=0;assert(cudaGetDeviceCount(&visible)==cudaSuccess && visible>=2);
    setenv("OPENMX_GPU_WORKERS","1",1);
    for(int g=2;g<=4 && g<=visible;g+=2) {
        probe_gpus=probe_count=g;probe_first=0;
        for(int copy=0;copy<2;copy++) for(int owner=0;owner<g;owner+=g-1) {
            setenv("OPENMX_GPU_WORKER_COPY",copy?"1":"0",1);
            assert(cudaSetDevice(owner)==cudaSuccess);acc_set_device_num(owner,acc_device_nvidia);
            for(int s=1;s<=4;s*=2) {run_case(1,s);run_case(7,s);}
        }
    }
    printf("PASS: %d OpenACC base/quadrature comparisons, guards, core-hole and Hubbard terms; max error %.3e\n",comparisons,max_error);
    return 0;
}

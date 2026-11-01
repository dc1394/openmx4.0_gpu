#!/usr/bin/env python3
"""Exercise production cache lifetime with deferred fake HIP operations."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

root = Path(__file__).resolve().parents[4]
source = (root/'source/backends/hip/src/Cluster_DFT_NonCol_dm_hip.cu').read_text()
kernel = source[source.index('__global__ static void BandNonColDenseMsKernel'):
                source.index('extern "C" int BandNonCol_BuildDenseMs_HIP')]
cache = source[source.index('namespace {\nstruct Mi300aBandBuildCache'):
               source.index('__global__ static void BandNonColAddDenseKernel')]
prefix = r'''
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <map>
#include <new>
#include <tuple>
#include <vector>
#define CHECK(x) do {if(!(x)){fprintf(stderr,"line %d: %s\n",__LINE__,#x);exit(1);}}while(0)
#define __global__
struct dcomplex { double r,i; };
struct BandNonColHipConstructEntry { int m_index,dense_index,phase_index; };
struct dim3 {unsigned x,y,z;dim3(unsigned a=1,unsigned b=1,unsigned c=1):x(a),y(b),z(c){}};
static dim3 blockIdx,blockDim,threadIdx;
using hipError_t=int;
static const int hipSuccess=0,hipMemcpyHostToDevice=1;
static int device=0,allocation_calls=0,fail_at=-1,queries=0;
static std::map<void*,int> allocations;
static std::vector<std::function<void()>> pending[2];
static int hipStreamSynchronize(void*) {
    auto work=std::move(pending[device]);pending[device].clear();
    for(auto &f:work)f();
    return 0;
}
static int hipGetDevice(int *d){queries++;*d=device;return 0;}
static int hipSetDevice(int d){CHECK(d>=0 && d<2);device=d;return 0;}
template<class T>static int hipMalloc(T **p,size_t bytes){
    if(++allocation_calls==fail_at){*p=nullptr;return 1;}
    *p=(T*)malloc(bytes);CHECK(*p);allocations[*p]=device;return 0;
}
static int hipFree(void *p){
    hipStreamSynchronize(nullptr);CHECK(allocations.count(p) && allocations[p]==device);
    allocations.erase(p);free(p);return 0;
}
static int hipMemcpyAsync(void *d,const void *s,size_t bytes,int kind,void*){
    CHECK(kind==hipMemcpyHostToDevice);pending[device].push_back([=](){memcpy(d,s,bytes);});return 0;
}
static int hipMemsetAsync(void *d,int value,size_t bytes,void*){
    pending[device].push_back([=](){memset(d,value,bytes);});return 0;
}
static int hipGetLastError(){return 0;}
static void atomicAdd(double *p,double v){*p+=v;}
template<class F,class... A>static void launch(F f,dim3 grid,dim3 block,A... args){
    auto values=std::make_tuple(args...);
    pending[device].push_back([=](){blockDim=block;
        for(unsigned b=0;b<grid.x;b++)for(unsigned t=0;t<block.x;t++){
            blockIdx.x=b;threadIdx.x=t;std::apply(f,values);
        }});
}
#define hipLaunchKernelGGL(f,grid,block,shared,stream,...) launch(f,grid,block,__VA_ARGS__)
'''
main = r'''
int main(){
    void *state=nullptr;
    BandNonColHipConstructEntry entries[4]={{0,0,0},{1,1,1},{2,2,0},{3,3,1}};
    double pr[2]={1,0},pi[2]={0,1},a[4]={1,2,3,4},b[4]={11,12,13,14};
    dcomplex out[4]={{-99,0}};
    auto build=[&](unsigned long long geometry,unsigned long long frame,int slot,int cpx,
                   const double *v,double k,size_t limit=1024){
        return BandNonCol_BuildDenseMsCached_HIP(&state,geometry,frame,slot,cpx,4,4,2,2,
                                                entries,pr,pi,v,k,0,0,out,limit);
    };
    CHECK(build(10,1,1,0,a,0)==0 && out[0].r==-99);
    hipStreamSynchronize(nullptr);
    CHECK(out[0].r==1 && out[1].i==2 && out[2].r==3 && out[3].i==4);
    int warm_allocations=allocation_calls;
    a[0]=7; // Values are immutable within a frame, even if the host pointer is reused.
    CHECK(build(10,1,1,0,a,0)==0);hipStreamSynchronize(nullptr);
    CHECK(out[0].r==1 && allocation_calls==warm_allocations);
    CHECK(build(10,2,1,0,a,0)==0);hipStreamSynchronize(nullptr);CHECK(out[0].r==7);
    CHECK(build(10,2,2,1,b,0)==0);hipStreamSynchronize(nullptr);
    CHECK(out[0].i==11 && out[1].r==-12 && out[2].i==13 && out[3].r==-14);
    pr[1]=1;pi[1]=0;
    CHECK(build(10,2,1,0,a,0.25)==0);hipStreamSynchronize(nullptr);
    CHECK(out[1].r==2 && out[1].i==0);
    entries[0].dense_index=3;entries[3].dense_index=0;
    CHECK(build(11,2,1,0,a,0.25)==0);hipStreamSynchronize(nullptr);
    CHECK(out[0].r==4 && out[3].r==7); // Same dimensions/pointers, changed geometry.
    unsigned long long stats[4];BandNonCol_BuildDenseCacheStats_HIP(state,stats);
    CHECK(stats[0]==6 && stats[1]==2 && stats[2]==4 && stats[3]==3);
    CHECK(build(11,3,1,0,a,0,1)==2 && !state && allocations.empty());
    CHECK(build(12,3,1,0,a,0)==0);hipStreamSynchronize(nullptr);
    device=1;BandNonCol_BuildDenseCacheRelease_HIP(&state);
    CHECK(device==1 && !state && allocations.empty());
    int before=queries;BandNonCol_BuildDenseCacheRelease_HIP(&state);CHECK(queries==before);
    device=0;
    for(int failure=1;failure<=4;failure++){
        fail_at=allocation_calls+failure;
        CHECK(build(20,4,1,0,a,0)==2 && !state && allocations.empty());
        fail_at=-1;
    }
    CHECK(BandNonCol_BuildDenseMsCached_HIP(&state,1,1,8,0,4,4,2,2,entries,pr,pi,a,0,0,0,out,1024)==1);
    CHECK(!state && allocations.empty());
    puts("PASS: deferred copies; SCF/geometry/phase/slot invalidation; bounded decline; allocation failure; device-safe and CPU-only cleanup");
}
'''
with tempfile.TemporaryDirectory(prefix='openmx-mi300a-build-cache-') as d:
    path=Path(d)
    (path/'test.cpp').write_text(prefix+kernel+cache+main)
    subprocess.run(shlex.split(os.environ.get('CXX','c++'))+
                   shlex.split(os.environ.get('CXXFLAGS',''))+
                   ['-std=c++17','-O1','-Wall','-Wextra',str(path/'test.cpp'),'-o',str(path/'test')],check=True)
    subprocess.run([str(path/'test')],check=True)

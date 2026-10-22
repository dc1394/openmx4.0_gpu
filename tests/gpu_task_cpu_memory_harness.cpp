#include "openmx_gpu_task_cpu.h"
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

extern "C" void *__real_malloc(size_t);
extern "C" void __real_free(void*);
namespace {
size_t limit=std::numeric_limits<size_t>::max(),requests[128],calls,allocations,releases;
void *live=nullptr;
bool measuring=false,fail_first=false,overflow_must_not_allocate=false;
void check(bool ok,const char *message){if(!ok)throw std::runtime_error(message);}
void begin(size_t cap=std::numeric_limits<size_t>::max(),bool first=false){
    check(!live,"previous CPU workspace leaked");limit=cap;calls=allocations=releases=0;
    fail_first=first;measuring=true;
}
void finish(){
    measuring=false;check(!live && allocations==releases,"CPU workspace allocation/free imbalance");
}
struct Chunks {size_t calls=0,first[64]{},count[64]{};};
struct Parameters {double factor=.5;Chunks *chunks=nullptr;bool throw_after_write=false;};
void kernel(size_t first,size_t n,const void *opaque,void *const views[],int gpu){
    check(!gpu,"CPU recovery selected a GPU kernel");
    const auto& p=*static_cast<const Parameters*>(opaque);
    auto& trace=*p.chunks;check(trace.calls<64,"too many CPU chunks");
    trace.first[trace.calls]=first;trace.count[trace.calls++]=n;
    const auto *dense=static_cast<const float*>(views[0]);
    const auto *offsets=static_cast<const size_t*>(views[1]);
    const auto *items=static_cast<const int*>(openmx_gpu_task_ragged_data(views[1],n));
    const auto *lookup=static_cast<const int*>(views[2]);
    auto *out=static_cast<double*>(views[3]),*sum=static_cast<double*>(views[4]);
    auto *scratch=static_cast<double*>(views[5]);sum[0]=sum[1]=0;
    check(offsets[0]==0,"CSR offsets not normalized for this chunk");
    for(size_t i=0;i<n;++i){
        scratch[i]=p.factor*dense[i]+lookup[(first+i)%3];
        for(size_t k=offsets[i];k<offsets[i+1];++k)scratch[i]+=items[k];
        out[i]=scratch[i];
        if(p.throw_after_write)throw std::runtime_error("injected CPU kernel exception");
        sum[0]+=out[i];sum[1]+=first+i;
    }
}
struct Case {
    static constexpr size_t n=19;
    std::array<float,n> dense{};
    std::array<size_t,n+1> offsets{};
    std::array<int,n*4> items{};
    int lookup[3]{7,-3,2};
    double output[n+2],sum[2]{};
    Chunks chunks;Parameters parameters{.5,&chunks,false};OpenMXGpuTask task{};
    Case(){
        for(size_t i=0;i<n;++i){dense[i]=static_cast<float>(static_cast<int>(i)-7)*.25f;offsets[i+1]=offsets[i]+i%5;}
        for(size_t k=0;k<items.size();++k)items[k]=static_cast<int>(k%7)-3;
        std::fill(std::begin(output),std::end(output),-947.);
        task.first=11;task.count=n;task.parameters=&parameters;task.parameter_bytes=sizeof(parameters);
        task.inputs=3;task.outputs=2;task.scratch=1;task.kernel=kernel;
        task.view[0]={dense.data(),0,sizeof(float),nullptr,0};
        task.view[1]={items.data(),0,0,offsets.data(),sizeof(int)};
        task.view[2]={lookup,sizeof(lookup),0,nullptr,0};
        task.view[3]={output+1,0,sizeof(double),nullptr,0};
        task.view[4]={sum,sizeof(sum),0,nullptr,0};
        task.view[5]={nullptr,0,sizeof(double),nullptr,0};
    }
    void verify()const{
        long double total=0,indices=0;
        for(size_t i=0;i<n;++i){
            long double expected=.5L*dense[i]+lookup[(11+i)%3];
            for(size_t k=offsets[i];k<offsets[i+1];++k)expected+=items[k];
            check(output[i+1]==expected,"dense/CSR/lookup result differs");total+=expected;indices+=11+i;
        }
        check(output[0]==-947 && output[n+1]==-947 && sum[0]==total && sum[1]==indices,"CPU guards/reduction differ");
        size_t next=11;
        for(size_t k=0;k<chunks.calls;++k){check(chunks.first[k]==next && chunks.count[k]>0,"CPU chunks reordered or empty");next+=chunks.count[k];}
        check(next==11+n && chunks.calls==allocations,"CPU chunk coverage or allocation count differs");
    }
};
void dense_and_ragged(){
    for(int mode=0;mode<3;++mode){
        Case c;const auto dense=c.dense;const auto offsets=c.offsets;const auto items=c.items;
        begin(mode==2?128:std::numeric_limits<size_t>::max(),mode==1);
        openmx_gpu_task_cpu(&c.task);finish();c.verify();
        check(c.dense==dense && c.offsets==offsets && c.items==items,"CPU modified borrowed inputs");
        if(mode){check(c.chunks.count[0]<Case::n && calls>allocations,"host OOM did not shrink the CPU range");}
        if(mode==2)for(size_t i=0;i<calls;++i)if(requests[i]>128)check(i+1<calls,"oversized final allocation unexpectedly succeeded");
    }
    Case c;c.parameters.throw_after_write=true;begin();bool caught=false;
    try{openmx_gpu_task_cpu(&c.task);}catch(const std::runtime_error&){caught=true;}
    finish();check(caught && allocations==1 && c.output[1]!=-947,"kernel exception did not unwind a live workspace");
    check(c.output[0]==-947 && c.output[Case::n+1]==-947,"kernel exception corrupted outer guards");
    puts("PASS: CPU task dense/CSR/lookup/reduction, host OOM range shrinking, borrowed inputs and exception cleanup");
}
struct Large {size_t stride;Chunks chunks;};
void large_kernel(size_t first,size_t n,const void *opaque,void *const views[],int gpu){
    check(!gpu,"large CPU task used GPU");auto& p=*const_cast<Large*>(static_cast<const Large*>(opaque));
    p.chunks.first[p.chunks.calls]=first;p.chunks.count[p.chunks.calls++]=n;
    auto *out=static_cast<int*>(views[0]);auto *scratch=static_cast<unsigned char*>(views[1]);
    scratch[0]=13;scratch[n*p.stride-1]=27;
    for(size_t i=0;i<n;++i)out[i]=static_cast<int>(first+i);
}
void bounds_and_empty(){
    constexpr size_t mib=1024*1024;
    for(auto shape: {std::pair<size_t,size_t>{3,40*mib},{1,64*mib+1}}){
        Large p{shape.second,{}};int output[5]{-991,-991,-991,-991,-991};OpenMXGpuTask t{};
        t.first=9;t.count=shape.first;t.parameters=&p;t.outputs=1;t.scratch=1;t.kernel=large_kernel;
        t.view[0]={output+1,0,sizeof(int),nullptr,0};t.view[1]={nullptr,0,p.stride,nullptr,0};
        begin();openmx_gpu_task_cpu(&t);finish();
        check(p.chunks.calls==shape.first && allocations==shape.first,"64 MiB CPU range bound changed");
        for(size_t i=0;i<shape.first;++i)check(p.chunks.count[i]==1 && output[i+1]==9+static_cast<int>(i),"large chunk result differs");
        check(output[0]==-991 && output[shape.first+1]==-991,"large chunk guards differ");
    }
    OpenMXGpuTask empty{};begin();openmx_gpu_task_cpu(&empty);finish();check(!calls,"empty CPU task touched heap");
    puts("PASS: 64 MiB CPU chunk bound, indivisible larger range and allocation-free empty task");
}
void direct_kernel(size_t,size_t n,const void*,void *const views[],int){
    const auto *in=static_cast<const double*>(views[0]);auto *out=static_cast<double*>(views[1]);
    for(size_t i=0;i<n;++i)out[i]=2*in[i];
}
void no_scratch(){
    double in[3]{1,-2,.5},out[5]{-23,-23,-23,-23,-23};OpenMXGpuTask t{};
    t.count=3;t.inputs=1;t.outputs=1;t.kernel=direct_kernel;
    t.view[0]={in,0,sizeof(double),nullptr,0};t.view[1]={out+1,0,sizeof(double),nullptr,0};
    begin();openmx_gpu_task_cpu(&t);finish();check(calls==1 && requests[0]==1,"zero workspace allocation policy changed");
    check(out[0]==-23 && out[4]==-23 && out[1]==2 && out[2]==-4 && out[3]==1,"direct CPU task differs");
    puts("PASS: direct views and original one-byte allocation for a nonempty task without scratch");
}
}
extern "C" void *__wrap_malloc(size_t bytes){
    if(!measuring)return __real_malloc(bytes);
    if(overflow_must_not_allocate)std::_Exit(29);
    if(calls>=128 || live)std::_Exit(30);
    requests[calls++]=bytes;if(bytes>limit || (fail_first && calls==1))return nullptr;
    live=__real_malloc(bytes);if(live)++allocations;return live;
}
extern "C" void __wrap_free(void *pointer){
    if(pointer && pointer==live){live=nullptr;++releases;}
    __real_free(pointer);
}
int main(int argc,char **argv){try{
    if(argc>1){
        OpenMXGpuTask t{};t.count=2;t.outputs=1;t.scratch=1;t.kernel=direct_kernel;
        t.view[1].item_bytes=std::numeric_limits<size_t>::max();
        begin();overflow_must_not_allocate=std::strcmp(argv[1],"overflow")==0;
        if(!overflow_must_not_allocate){t.count=1;t.view[1].item_bytes=8;limit=0;}
        openmx_gpu_task_cpu(&t);return 31;
    }
    dense_and_ragged();bounds_and_empty();no_scratch();return 0;
}catch(const std::exception& error){measuring=false;std::fprintf(stderr,"%s\n",error.what());return 1;}}

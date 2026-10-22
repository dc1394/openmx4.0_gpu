#include "openmx_gpu_task_packet.h"
#include "openmx_gpu_task_layout.h"
#include "openmx_gpu_task_cpu.h"
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>

extern "C" size_t openmx_test_task_packet_golden(void*);
extern "C" void openmx_test_task_packet_free(void*);
extern "C" void *__real_calloc(size_t,size_t);
extern "C" void __real_free(void*);
namespace {
void *owned[8]{};
size_t allocation_calls,live,fail_at,local_calls,kernel_calls;
int worker_mode;
const unsigned char *active_packet=nullptr;
size_t active_bytes;
void check(bool ok,const char *message){if(!ok)throw std::runtime_error(message);}
void begin(size_t failure=0,int mode=0){
    check(!live,"previous packet/result allocation leaked");allocation_calls=local_calls=kernel_calls=0;
    fail_at=failure;worker_mode=mode;
}
void kernel(size_t first,size_t n,const void *parameters,void *const views[],int gpu){
    ++kernel_calls;check(!gpu,"SDK-free worker model used a GPU kernel");
    const auto *factor=static_cast<const unsigned char*>(parameters);
    const auto *dense=static_cast<const float*>(views[0]);
    const auto *offsets=static_cast<const size_t*>(views[1]);
    const auto *items=static_cast<const int*>(openmx_gpu_task_ragged_data(views[1],n));
    const auto *lookup=static_cast<const int*>(views[2]);
    auto *out=static_cast<double*>(views[3]),*sum=static_cast<double*>(views[4]);
    auto *scratch=static_cast<double*>(views[5]);sum[0]=sum[1]=0;
    for(size_t i=0;i<n;++i){
        scratch[i]=factor[0]*dense[i]+lookup[(first+i)%2];
        for(size_t k=offsets[i];k<offsets[i+1];++k)scratch[i]+=items[k];
        out[i]=scratch[i];sum[0]+=out[i];sum[1]+=first+i;
    }
}
struct Case {
    float dense[7]{.25f,-1.f,.5f,3.f,-2.f,7.f,-.5f};
    size_t offsets[8]{0,0,2,3,5,5,8,9};
    int items[9]{-2,7,4,-3,6,8,-4,5,9},lookup[2]{5,-1};
    unsigned char parameters[13]{2,3,4,5,6,7,8,9,10,11,12,13,14};
    double output[9],sum[2]{};OpenMXGpuTask task{};
    Case(){
        for(double& value:output)value=-919;
        task.first=11;task.count=7;task.parameters=parameters;task.parameter_bytes=sizeof(parameters);
        task.inputs=3;task.outputs=2;task.scratch=1;task.kernel=kernel;task.scope_kind=2;
        task.view[0]={dense,0,sizeof(float),nullptr,0};
        task.view[1]={items,0,0,offsets,sizeof(int)};
        task.view[2]={lookup,sizeof(lookup),0,nullptr,0};
        task.view[3]={output+1,0,sizeof(double),nullptr,0};
        task.view[4]={sum,sizeof(sum),0,nullptr,0};
        task.view[5]={nullptr,0,sizeof(double),nullptr,0};
    }
    bool unchanged()const{
        for(double value:output)if(value!=-919)return false;
        return sum[0]==0 && sum[1]==0;
    }
};
struct Golden {
    alignas(16) std::array<unsigned char,1200> data{};
    size_t bytes=openmx_test_task_packet_golden(data.data());
    OpenMXGpuTaskPacket& header(){return *reinterpret_cast<OpenMXGpuTaskPacket*>(data.data());}
};
void check_result(const void *pointer,size_t bytes){
    check(bytes==48,"remote result layout differs");
    const double expected[6]{4,14,-5,28,41,58};
    check(std::memcmp(pointer,expected,sizeof(expected))==0,"remote dense/CSR/reduction oracle differs");
}
void golden_roundtrip(){
    Case c;Golden expected;void *packet=nullptr,*result=nullptr;size_t bytes=0,result_bytes=0;
    begin();check(openmx_gpu_task_pack(&c.task,2,6,&packet,&bytes),"packet construction failed");
    check(allocation_calls==1 && live==1 && bytes==expected.bytes,"packet allocation count or size differs");
    check(std::memcmp(packet,expected.data.data(),bytes)==0,"C++ encoding differs from independent C wire fixture");
    active_packet=expected.data.data();active_bytes=expected.bytes;
    check(openmx_gpu_task_remote(&c.task,active_packet,active_bytes,&result,&result_bytes),"C fixture decoding failed");
    check(allocation_calls==2 && live==2 && local_calls==1 && kernel_calls==1 && c.unchanged(),"remote published early or allocated extra packet storage");
    check_result(result,result_bytes);
    task_commit(&c.task,2,6,result,result_bytes);
    const double want[9]{-919,-919,-919,4,14,-5,28,-919,-919};
    check(std::memcmp(c.output,want,sizeof(want))==0 && c.sum[0]==41 && c.sum[1]==58,"commit slice or guards differ");
    openmx_test_task_packet_free(packet);openmx_test_task_packet_free(result);check(!live,"C free did not reclaim transferred storage");
    puts("PASS: independent C wire encoding/decoding, dense/CSR/lookup/reduction oracle, deferred commit and C ownership transfer");
}
void allocation_and_exception_failures(){
    Case c;Golden golden;int sentinel;void *out=&sentinel;size_t bytes=0;
    begin(1);check(!openmx_gpu_task_pack(&c.task,2,6,&out,&bytes) && out==&sentinel && allocation_calls==1 && !live,"failed pack changed ownership");
    active_packet=golden.data.data();active_bytes=golden.bytes;
    begin(1);check(!openmx_gpu_task_remote(&c.task,active_packet,active_bytes,&out,&bytes) && !out && !live && !local_calls && allocation_calls==1,"remote OOM leaked or invoked worker");
    begin(0,1);check(!openmx_gpu_task_remote(&c.task,active_packet,active_bytes,&out,&bytes) && out && live==1 && local_calls==1 && c.unchanged(),"ordinary GPU failure did not retain collective ownership contract");
    for(size_t i=0;i<bytes;++i)check(static_cast<unsigned char*>(out)[i]==0,"failed remote buffer not zero initialized");
    openmx_test_task_packet_free(out);check(!live,"failed remote output leaked");
    begin(0,2);out=&sentinel;bool caught=false;
    try{(void)openmx_gpu_task_remote(&c.task,active_packet,active_bytes,&out,&bytes);}catch(const std::runtime_error&){caught=true;}
    check(caught && out==&sentinel && local_calls==1 && kernel_calls==1 && !live && c.unchanged(),"worker exception leaked unpublished result or changed origin");
    puts("PASS: pack/result calloc failures, ordinary failed-result transfer and exception cleanup before ownership transfer");
}
void malformed(){
    Case c;Golden golden;int sentinel;void *out;size_t bytes=0,rejected=0;
    begin();
    for(size_t n=0;n<golden.bytes;++n){
        auto prefix=std::make_unique<unsigned char[]>(n);if(n)std::memcpy(prefix.get(),golden.data.data(),n);
        out=&sentinel;
        check(!openmx_gpu_task_remote(&c.task,prefix.get(),n,&out,&bytes) && out==&sentinel,"truncated packet accepted or changed ownership");++rejected;
    }
    for(int kind=0;kind<22;++kind){
        Golden bad;auto& h=bad.header();size_t size=bad.bytes;
        auto *index=reinterpret_cast<size_t*>(bad.data.data()+1040);
        switch(kind){
        case 0:h.inputs=-1;break;case 1:h.outputs=0;break;case 2:h.scratch=-1;break;
        case 3:h.outputs=41;break;case 4:h.inputs=40;break;case 5:h.scratch=40;break;
        case 6:h.count=UINT64_MAX;break;case 7:h.first=UINT64_MAX;break;
        case 8:h.parameter_bytes=UINT64_MAX;break;case 9:h.view[0].item_bytes=UINT64_MAX;break;
        case 10:h.view[2].bytes=UINT64_MAX;break;case 11:h.view[1].element_bytes=UINT64_MAX;break;
        case 12:h.view[3].element_bytes=1;break;case 13:index[0]=1;break;
        case 14:index[2]=0;break;case 15:index[4]=SIZE_MAX;break;
        case 16:h.view[4].bytes=7;break;case 17:h.view[3].item_bytes=UINT64_MAX;break;
        case 18:h.view[4].bytes=UINT64_MAX-7;break;case 19:++size;break;
        case 20:h.count=SIZE_MAX/sizeof(size_t);break;case 21:h.view[1].element_bytes=0;break;
        }
        out=&sentinel;
        check(!openmx_gpu_task_remote(&c.task,bad.data.data(),size,&out,&bytes) && out==&sentinel,"malformed packet accepted or changed ownership");++rejected;
    }
    check(!allocation_calls && !local_calls && !live && c.unchanged(),"malformed packet allocated or ran work");
    c.task.parameter_bytes=SIZE_MAX;out=&sentinel;
    check(!openmx_gpu_task_pack(&c.task,2,6,&out,&bytes) && bytes==SIZE_MAX && out==&sentinel && !allocation_calls,"saturated pack reached allocator");
    std::printf("PASS: %zu truncated prefixes and 22 malformed packets rejected without allocation/worker calls; saturated pack declined\n",golden.bytes);
    check(rejected==golden.bytes+22,"malformed cases skipped");
}
void empty(){
    Case c;void *packet=nullptr,*result=nullptr;size_t bytes=0,result_bytes=0;
    begin();check(openmx_gpu_task_pack(&c.task,3,3,&packet,&bytes),"empty pack failed");
    active_packet=static_cast<const unsigned char*>(packet);active_bytes=bytes;
    check(openmx_gpu_task_remote(&c.task,packet,bytes,&result,&result_bytes),"empty remote failed");
    check(allocation_calls==2 && result_bytes==16 && local_calls==1 && !kernel_calls && c.unchanged(),"empty remote semantics changed");
    const double zero[2]{};check(std::memcmp(result,zero,sizeof(zero))==0,"empty reduction not zero");
    openmx_test_task_packet_free(packet);openmx_test_task_packet_free(result);check(!live,"empty buffers leaked");
    puts("PASS: empty remote range, replicated data, zero reductions and transferred buffer cleanup");
}
}
extern "C" void *__wrap_calloc(size_t count,size_t width){
    ++allocation_calls;if(allocation_calls==fail_at)return nullptr;
    void *pointer=__real_calloc(count,width);if(!pointer)return nullptr;
    for(void*& slot:owned)if(!slot){slot=pointer;++live;return pointer;}
    std::abort();
}
extern "C" void __wrap_free(void *pointer){
    if(pointer)for(void*& slot:owned)if(slot==pointer){slot=nullptr;--live;break;}
    __real_free(pointer);
}
extern "C" int openmx_gpu_task_local(OpenMXGpuTask *t,const char*){
    ++local_calls;
    check(t->inputs==3 && t->outputs==2 && t->scratch==1 && t->scope_kind==2 && t->kernel==kernel,"decoded task metadata differs");
    check(t->parameters==active_packet+1008 && t->view[0].data==active_packet+1024 &&
          t->view[1].offsets==reinterpret_cast<const size_t*>(active_packet+(t->count?1040:1024)),"remote copied or misplaced borrowed wire data");
    check(t->parameter_bytes==13 && active_bytes>=1008+13,"parameter view differs");
    if(worker_mode==1)return 0;
    if(t->count)openmx_gpu_task_cpu(t);
    if(worker_mode==2)throw std::runtime_error("injected worker exception after private output write");
    return 1;
}
int main(){try{golden_roundtrip();allocation_and_exception_failures();malformed();empty();check(!live,"packet allocation leaked");return 0;}
    catch(const std::exception& error){fail_at=0;std::fprintf(stderr,"%s\n",error.what());return 1;}}

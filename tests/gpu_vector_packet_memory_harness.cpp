#include "openmx_gpu_vector_packet.h"
#include <array>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <stdexcept>
extern "C" size_t openmx_test_vector_packet_golden(void*,int);
extern "C" void openmx_test_vector_packet_free(void*);
extern "C" void *__real_malloc(size_t);
extern "C" void *__real_calloc(size_t,size_t);
extern "C" void __real_free(void*);
namespace {
void *owned[8]{};size_t live,malloc_calls,calloc_calls,local_calls;
int fail_malloc,fail_calloc,worker_mode;
const unsigned char *active_packet;
void check(bool ok,const char *message){if(!ok)throw std::runtime_error(message);}
void begin(int allocation=0,int mode=0){
    check(!live,"previous vector packet allocation leaked");malloc_calls=calloc_calls=local_calls=0;
    fail_malloc=allocation==1;fail_calloc=allocation==2;worker_mode=mode;
}
void remember(void *pointer){
    if(!pointer)return;
    for(void*& slot:owned)if(!slot){slot=pointer;++live;return;}
    std::abort();
}
struct Case {
    double data[21]{1,2,3,4,5,901,902,-1,0,2,-2,1,903,904,4,-3,1,2,-1,905,906};
    double weights[5]{1,2,-1,.5,3},coefficients[3]{2,-.5,1},output[7];
    int left[4]{2,0,1,2},right[4]{0,1,1,2};OpenMXGpuVector job{};
    explicit Case(int mode){
        for(double& value:output)value=-977;
        job.n=5;job.ld=7;job.mode=mode;job.columns=3;job.pairs=4;
        job.data=data;job.weight=weights;job.coeff=coefficients;job.left=left;job.right=right;job.output=output+1;
    }
    bool unchanged()const{for(double value:output)if(value!=-977)return false;return true;}
};
struct Golden {
    alignas(16) std::array<unsigned char,160> data{};size_t bytes;
    explicit Golden(int mode){::new(data.data())OpenMXGpuVectorPacket;bytes=openmx_test_vector_packet_golden(data.data(),mode);}
    OpenMXGpuVectorPacket& header(){return *reinterpret_cast<OpenMXGpuVectorPacket*>(data.data());}
};
const double gram[4]{-11,-10,-2,19},linear[3]{1,6,11};
void check_result(const void *result,size_t bytes,int mode){
    check(bytes==(mode?24:32) && std::memcmp(result,mode?linear:gram,bytes)==0,"vector remote oracle differs");
}
void roundtrip_and_failures(int mode){
    Case c(mode);Golden golden(mode);void *packet=nullptr,*result=nullptr;size_t bytes=0,result_bytes=0;
    begin();check(openmx_gpu_vector_pack(&c.job,1,4,&packet,&bytes),"vector pack failed");
    check(malloc_calls==1 && !calloc_calls && live==1 && bytes==golden.bytes,"pack allocation count/size differs");
    check(std::memcmp(packet,golden.data.data(),bytes)==0,"encoding or canonical header padding differs from C fixture");
    active_packet=golden.data.data();
    check(openmx_gpu_vector_remote(nullptr,active_packet,golden.bytes,&result,&result_bytes),"C vector fixture decoding failed");
    check_result(result,result_bytes,mode);
    check(local_calls==1 && malloc_calls==1 && calloc_calls==1 && live==2 && c.unchanged(),"remote copied/published extra data or added allocations");
    openmx_test_vector_packet_free(packet);openmx_test_vector_packet_free(result);check(!live,"C transfer leaked vector storage");
    /* Previously sent tail padding is unspecified, and remains accepted. */
    std::memset(golden.data.data()+20,0xbd,4);begin();result=nullptr;
    check(openmx_gpu_vector_remote(nullptr,active_packet,golden.bytes,&result,&result_bytes),"legacy padding rejected");
    check_result(result,result_bytes,mode);openmx_test_vector_packet_free(result);check(!live,"legacy result leaked");
    int sentinel;void *out=&sentinel;
    begin(1);check(!openmx_gpu_vector_pack(&c.job,1,4,&out,&bytes) && out==&sentinel && !live && malloc_calls==1,"pack OOM changed ownership");
    begin(2);check(!openmx_gpu_vector_remote(nullptr,active_packet,golden.bytes,&out,&bytes) && !out && !live && !local_calls && calloc_calls==1,"remote OOM leaked or ran worker");
    begin(0,1);check(!openmx_gpu_vector_remote(nullptr,active_packet,golden.bytes,&out,&bytes) && out && live==1 && c.unchanged(),"ordinary failed result ownership changed");
    for(size_t i=0;i<bytes;++i)check(static_cast<unsigned char*>(out)[i]==0,"ordinary failed result not zero");
    openmx_test_vector_packet_free(out);check(!live,"ordinary failed result leaked");
    begin(0,2);out=&sentinel;bool caught=false;
    try{(void)openmx_gpu_vector_remote(nullptr,active_packet,golden.bytes,&out,&bytes);}catch(const std::runtime_error&){caught=true;}
    check(caught && out==&sentinel && !live && local_calls==1 && c.unchanged(),"worker exception leaked or transferred unpublished result");
    std::printf("PASS: vector mode %d C wire/oracle, canonical and legacy padding, allocator failures, ordinary result transfer and exception cleanup\n",mode);
}
void malformed(int mode){
    Golden golden(mode);int sentinel;void *out;size_t bytes=0;
    begin();
    for(size_t n=0;n<golden.bytes;++n){
        auto prefix=std::make_unique<unsigned char[]>(n);if(n)std::memcpy(prefix.get(),golden.data.data(),n);
        out=&sentinel;check(!openmx_gpu_vector_remote(nullptr,prefix.get(),n,&out,&bytes) && out==&sentinel,"truncated vector packet accepted");
    }
    const int conditions=mode?10:15;
    for(int kind=0;kind<conditions;++kind){
        Golden bad(mode);auto& h=bad.header();size_t size=bad.bytes;
        switch(kind){
        case 0:h.mode=-1;break;case 1:h.mode=2;break;case 2:h.columns=0;break;
        case 3:h.columns=-1;break;case 4:h.pairs=-1;break;case 5:h.n=UINT64_MAX;break;
        case 6:h.columns=INT_MAX;break;case 7:h.n=SIZE_MAX/sizeof(double);break;
        case 8:++size;break;case 9:h.n=0;break;
        case 10:reinterpret_cast<int*>(bad.data.data()+120)[0]=-1;break;
        case 11:reinterpret_cast<int*>(bad.data.data()+136)[0]=-1;break;
        case 12:reinterpret_cast<int*>(bad.data.data()+120)[3]=3;break;
        case 13:reinterpret_cast<int*>(bad.data.data()+136)[3]=3;break;
        case 14:h.pairs=INT_MAX;break;
        }
        out=&sentinel;check(!openmx_gpu_vector_remote(nullptr,bad.data.data(),size,&out,&bytes) && out==&sentinel,"malformed vector packet accepted");
    }
    check(!malloc_calls && !calloc_calls && !local_calls && !live,"invalid vector packet allocated or ran work");
    Case c(mode);out=&sentinel;c.job.n=SIZE_MAX;
    check(!openmx_gpu_vector_pack(&c.job,0,SIZE_MAX,&out,&bytes) && out==&sentinel,"saturated vector pack accepted");
    c.job.n=5;
    check(!openmx_gpu_vector_pack(&c.job,4,3,&out,&bytes) && !openmx_gpu_vector_pack(&c.job,0,6,&out,&bytes),"invalid vector slice accepted");
    check(!malloc_calls && !calloc_calls && !local_calls,"invalid vector pack reached allocator");
    std::printf("PASS: vector mode %d: %zu exact truncated prefixes, %d malformed headers/indices, overflow and invalid slices rejected before allocation/worker\n",mode,golden.bytes,conditions);
}
void empty(int mode){
    Case c(mode);void *packet=nullptr,*result=nullptr;size_t bytes=0,result_bytes=0;
    begin();check(openmx_gpu_vector_pack(&c.job,2,2,&packet,&bytes),"empty vector pack failed");active_packet=static_cast<const unsigned char*>(packet);
    check(bytes==(mode?48:56) && openmx_gpu_vector_remote(nullptr,packet,bytes,&result,&result_bytes),"empty vector decode failed");
    check(result && result_bytes==(mode?0:32) && malloc_calls==1 && calloc_calls==1 && live==2 && local_calls==1 && c.unchanged(),"empty vector ownership/size differs");
    for(size_t i=0;i<(result_bytes?result_bytes:1);++i)check(static_cast<unsigned char*>(result)[i]==0,"empty vector result not zero");
    openmx_test_vector_packet_free(packet);openmx_test_vector_packet_free(result);check(!live,"empty vector leaked");
    std::printf("PASS: vector mode %d empty range and zero-sized output allocation contract\n",mode);
}
}
extern "C" void *__wrap_malloc(size_t bytes){
    ++malloc_calls;if(fail_malloc)return nullptr;
    void *p=__real_malloc(bytes);if(p)std::memset(p,0xa5,bytes);remember(p);return p;
}
extern "C" void *__wrap_calloc(size_t count,size_t width){
    ++calloc_calls;if(fail_calloc)return nullptr;
    void *p=__real_calloc(count,width);remember(p);return p;
}
extern "C" void __wrap_free(void *pointer){
    if(pointer)for(void*& slot:owned)if(slot==pointer){slot=nullptr;--live;break;}
    __real_free(pointer);
}
extern "C" int openmx_gpu_vector_local(OpenMXGpuVector *j){
    ++local_calls;
    check(j->n==j->ld && j->columns==3 && j->pairs==4 && (j->n==0 || j->n==3),"decoded vector metadata differs");
    check(j->data==reinterpret_cast<const double*>(active_packet+24) && j->coeff==j->weight &&
          j->weight==j->data+3*j->n,"remote copied or misplaced vector input");
    if(j->n){
        Golden expected(j->mode);
        check(std::memcmp(active_packet+24,expected.data.data()+24,expected.bytes-24)==0,"borrowed vector values or pair indices differ");
    }
    if(worker_mode==1)return 0;
    if(j->n)std::memcpy(j->output,j->mode?linear:gram,(j->mode?3:4)*sizeof(double));
    if(worker_mode==2)throw std::runtime_error("injected vector worker exception after private write");
    return 1;
}
int main(){try{for(int mode:{0,1}){roundtrip_and_failures(mode);malformed(mode);empty(mode);}check(!live,"vector packet leak");return 0;}
    catch(const std::exception& error){fail_malloc=fail_calloc=0;std::fprintf(stderr,"%s\n",error.what());return 1;}}

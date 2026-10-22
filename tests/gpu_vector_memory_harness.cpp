#include "fixtures/gpu_worker_lifecycle.hpp"
#include "openmx_gpu_vector_worker.cpp"
#include <cmath>
#include <climits>

namespace {
struct Case {
    static constexpr size_t n=7,ld=11;
    static constexpr int columns=3,pairs=6;
    double data[columns*ld],weight[n],coeff[columns]{.5,-.25,.125},output[n+2];
    int left[pairs]{0,1,1,2,2,2},right[pairs]{0,0,1,0,1,2};
    OpenMXGpuVector job{};
    explicit Case(int mode){
        for(int c=0;c<columns;++c)for(size_t i=0;i<ld;++i)data[c*ld+i]=(int(i%5)-2)*.125+c*.25;
        for(size_t i=0;i<n;++i)weight[i]=(int(i%3)-1)*.5;
        job={n,ld,mode,columns,pairs,data,weight,coeff,left,right,output+1};clear();
    }
    void clear(){for(double& value:output)value=-731;}
    int run(){return openmx_gpu_vector_local(&job);}
    bool unchanged()const{for(double value:output)if(value!=-731)return false;return true;}
    void verify()const{
        const size_t outputs=job.mode?n:pairs;
        for(size_t i=0;i<outputs;++i){
            long double expected=0;
            if(job.mode)for(int c=0;c<columns;++c)expected+=(long double)coeff[c]*data[c*ld+i];
            else for(size_t k=0;k<n;++k)expected+=(long double)weight[k]*data[left[i]*ld+k]*data[right[i]*ld+k];
            check(std::abs((long double)output[i+1]-expected)<1e-12,"weighted contraction/linear result differs");
        }
        check(output[0]==-731 && output[outputs+1]==-731,"output guards differ");
    }
};
void empty_and_overflow(){
    begin();OpenMXGpuVector empty{};
    check(openmx_gpu_vector_local(&empty) && !trace.count[get_device] && !host_calls,"empty vector touched SDK/heap");
    Case test(0);test.job.n=SIZE_MAX;test.job.columns=INT_MAX;
    check(!test.run() && !live && !host_calls && !trace.count[get_device],"overflow reached an allocator");
}
void failures(int mode){
    Case test(mode);begin();check(test.run()==1 && commits==group,"control vector failed");test.verify();
    check(!live && !host_live && trace.current==3 && !trace.wrong_device_frees,"control vector leaked storage/binding");
    const auto observed=before_commit;const int host_sites=host_before_commit;int sdk_sites=0;
    for(Call kind:{get_device,allocate,copy})for(int at=1;at<=observed[kind];++at){
        test.clear();begin();trace.fail_kind=kind;trace.fail_at=at;
        check(!test.run() && !commits && test.unchanged(),"preparation failure published vector output");
        check(!live && !host_live && trace.current==3 && !trace.wrong_device_frees,"failed vector leaked storage/binding");++sdk_sites;
    }
    for(int at=1;at<=host_sites;++at){
        test.clear();begin();host_fail_at=at;
        check(!test.run() && !commits && test.unchanged(),"host OOM published vector output");
        check(!live && !host_live && trace.current==3,"host OOM leaked vector buffers");
    }
    test.clear();begin();omit_release=true;check(!test.run(),"omitted cleanup returned success");omit_release=false;
    check(test.unchanged() && !live && !host_live && trace.current==3,"omitted RELEASE leaked vector buffers");
    test.clear();begin();throw_before_execute=true;bool caught=false;
    try{(void)test.run();}catch(const std::runtime_error&){caught=true;}
    throw_before_execute=false;
    check(caught && test.unchanged() && !live && !host_live && trace.current==3,"exception bypassed vector scope cleanup");
    std::cout<<"PASS: actual vector worker mode="<<mode<<"; "<<sdk_sites<<" SDK preparation failures, "<<host_sites
        <<" host allocation failures, omitted RELEASE and exception cleanup; long-double oracle and guards\n";
}
void release_retry(){
    begin();Case test(0);
    {
        VectorWork work;work.job=test.job;work.workers.count=1;work.workers.items=Case::n;
        trace.current=2;check(vector_worker(&work,0,OPENMX_GPU_WORKER_PREPARE),"release retry preparation failed");
        reset(0);trace.fail_kind=release;trace.fail_at=1;
        check(!vector_worker(&work,0,OPENMX_GPU_WORKER_RELEASE) && work.arena[0] && live==1 && trace.current==0,
            "failed explicit release lost vector ownership");
        reset(0);check(vector_worker(&work,0,OPENMX_GPU_WORKER_RELEASE) && !live && trace.current==0,"release retry failed");
    }
    check(!host_live && !trace.wrong_device_frees,"release retry retained host storage or foreign frees");
    std::cout<<"PASS: vector buffer release on its allocation device, failure retention and retry\n";
}
}
int main(){try{empty_and_overflow();failures(0);failures(1);release_retry();return 0;}
    catch(const std::exception& error){host_fail_at=0;std::cerr<<error.what()<<'\n';return 1;}}

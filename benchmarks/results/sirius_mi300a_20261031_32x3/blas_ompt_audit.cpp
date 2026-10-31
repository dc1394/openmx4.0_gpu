#include <omp-tools.h>
#include <dlfcn.h>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
static std::atomic<unsigned long long> seen{0};
static constexpr uint64_t mask=0xffff000000000000ULL, mt=0xb1a5000000000000ULL, serial=0xb1a6000000000000ULL;
static void parallel_begin(ompt_data_t*,const ompt_frame_t*,ompt_data_t *data,
                          unsigned requested,int,const void *address) {
    data->value=0;Dl_info info{};
    if(!address || !dladdr(address,&info) || !info.dli_fname)return;
    if(std::strstr(info.dli_fname,"libblis-mt"))data->value=mt|requested;
    else if(std::strstr(info.dli_fname,"libblis."))data->value=serial|requested;
}
static void implicit_task(ompt_scope_endpoint_t endpoint,ompt_data_t *parallel,
                          ompt_data_t*,unsigned actual,unsigned index,int) {
    if(endpoint!=ompt_scope_begin || !parallel || index || actual>=32)return;
    uint64_t tag=parallel->value&mask;
    if(tag!=mt && tag!=serial)return;
    const unsigned long long bit=1ULL<<(actual*2+(tag==mt));
    if(seen.fetch_or(bit)&bit)return;
    std::fprintf(stderr,"BLAS_OMPT rank=%s provider=%s requested=%u actual=%u\n",
        std::getenv("OMPI_COMM_WORLD_RANK"),tag==mt?"libblis-mt":"libblis-serial",
        unsigned(parallel->value&0xffff),actual);
}
static int initialize(ompt_function_lookup_t lookup,int,ompt_data_t*) {
    auto set=reinterpret_cast<ompt_set_callback_t>(lookup("ompt_set_callback"));
    if(!set)return 0;
    set(ompt_callback_parallel_begin,reinterpret_cast<ompt_callback_t>(parallel_begin));
    set(ompt_callback_implicit_task,reinterpret_cast<ompt_callback_t>(implicit_task));
    return 1;
}
static void finalize(ompt_data_t*) {}
extern "C" ompt_start_tool_result_t *ompt_start_tool(unsigned int,const char*) {
    static ompt_start_tool_result_t result{initialize,finalize,{0}};return &result;
}

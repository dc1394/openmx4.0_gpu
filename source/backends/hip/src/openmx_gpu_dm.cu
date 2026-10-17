#include "openmx_gpu_dm_entry.h"
#include "openmx_gpu_workers.h"
#include "openmx_gpu_views.h"
#include "hip_runtime_compat.h"
#include "hip_complex_compat.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>
#include <new>
#include <limits>
#include <cstdint>
namespace {
using Z=hipDoubleComplex;
struct Work {
    OpenMXGpuWorkers workers{};
    int kind,n,states,stride,entries,phases,host,accumulate,origin;
    const void *vectors;
    size_t vector_bytes;
    const int *basis0,*basis1,*phase;
    const double *real,*imag,*occ,*energy,*partial;
    double *target[10]{};
    struct Device {
        OpenMXGpuReadView vectors{};
        void *owned_vectors=nullptr;
        int *indices=nullptr;
        double *weights=nullptr,*phase=nullptr,*output=nullptr;
        std::vector<double> result;
        std::vector<int> basis;
        int n=0,stride=0;
    } device[OPENMX_GPU_WORKERS_MAX];
};
__global__ void kernel(int kind,int n,int states,int stride,int entries,const void *vectors,
    const int *indices,const double *weights,const double *phases,double *output,int phased,int edm,int pdm)
{
    for(int p=int(blockIdx.x*blockDim.x+threadIdx.x);p<entries;p+=int(blockDim.x*gridDim.x))
        openmx_gpu_dm_entry(kind,n,states,stride,entries,vectors,indices,weights,phases,output,phased,edm,pdm,p);
}
std::vector<int> basis_rows(const Work &w,size_t first,size_t last)
{
    std::vector<int> rows;rows.reserve(2*(last-first));
    rows.insert(rows.end(),w.basis0+first,w.basis0+last);rows.insert(rows.end(),w.basis1+first,w.basis1+last);
    std::sort(rows.begin(),rows.end());rows.erase(std::unique(rows.begin(),rows.end()),rows.end());return rows;
}
size_t capacity_cost(void *opaque,int worker,int count)
{
    const auto &w=*static_cast<Work*>(opaque);
    size_t first=openmx_gpu_worker_cut(w.entries,worker,count),last=openmx_gpu_worker_cut(w.entries,worker+1,count);
    size_t bytes=(last-first)*(2*sizeof(int)+12*sizeof(double))+3*size_t(w.states)*sizeof(double);
    if(w.host || count>1){
        try{
            if(w.kind>=OPENMX_GPU_DM_GATHER)return SIZE_MAX; // Gather reads an already resident matrix.
            auto rows=basis_rows(w,first,last);size_t width=w.kind==OPENMX_GPU_DM_REAL?8:16;
            size_t size=rows.size()*(w.kind==OPENMX_GPU_DM_NC?2:1)*size_t(w.states);
            if(size>(SIZE_MAX-bytes)/width)return SIZE_MAX;bytes+=size*width;
        }catch(const std::bad_alloc&){return SIZE_MAX;}
    }
    return bytes;
}

int worker(void *opaque,int worker,int phase)
{
    auto &w=*static_cast<Work*>(opaque);auto &d=w.device[worker];
    const size_t first=openmx_gpu_worker_cut(w.entries,worker,w.workers.count),last=openmx_gpu_worker_cut(w.entries,worker+1,w.workers.count),count=last-first;
    if(phase==OPENMX_GPU_WORKER_PREPARE) {
        for(size_t p=first;p<last;p++)
            if(w.basis0[p]<0 || w.basis0[p]>=w.n || w.basis1[p]<0 || w.basis1[p]>=w.n)return 0;
        d.n=w.n;d.stride=w.stride;
        std::vector<int> indices(2*count);
        if(w.host || w.workers.count>1){
            if(w.kind>=OPENMX_GPU_DM_GATHER)return 0;
            d.basis=basis_rows(w,first,last);d.n=int(d.basis.size());d.stride=w.states;
            const size_t width=w.kind==OPENMX_GPU_DM_REAL?sizeof(double):sizeof(Z);
            size_t bytes=d.basis.size()*(w.kind==OPENMX_GPU_DM_NC?2:1)*size_t(w.states)*width;
            if(hipMalloc(&d.owned_vectors,bytes)!=hipSuccess)return 0;
            d.vectors.pointer=d.owned_vectors;
            for(int spin=0;spin<(w.kind==OPENMX_GPU_DM_NC?2:1);spin++)for(size_t i=0;i<d.basis.size();i++){
                void *dst=static_cast<char*>(d.owned_vectors)+(i+spin*d.basis.size())*w.states*width;
                const void *src=static_cast<const char*>(w.vectors)+(size_t(d.basis[i]+spin*w.n)*w.stride)*width;
                if(w.host){if(hipMemcpy(dst,src,size_t(w.states)*width,hipMemcpyHostToDevice)!=hipSuccess)return 0;}
                else if(!openmx_gpu_copy_device(dst,src,size_t(w.states)*width))return 0;
            }
            for(size_t i=first;i<last;i++){
                indices[i-first]=int(std::lower_bound(d.basis.begin(),d.basis.end(),w.basis0[i])-d.basis.begin());
                indices[count+i-first]=int(std::lower_bound(d.basis.begin(),d.basis.end(),w.basis1[i])-d.basis.begin());
            }
        }else{
            if(!openmx_gpu_read_view(&d.vectors,w.vectors,w.vector_bytes))return 0;
            std::copy(w.basis0+first,w.basis0+last,indices.begin());std::copy(w.basis1+first,w.basis1+last,indices.begin()+count);
        }
        if(hipMalloc(reinterpret_cast<void**>(&d.indices),2*count*sizeof(int))!=hipSuccess ||
           hipMalloc(reinterpret_cast<void**>(&d.output),10*count*sizeof(double))!=hipSuccess)return 0;
        if(hipMemcpy(d.indices,indices.data(),2*count*sizeof(int),hipMemcpyHostToDevice)!=hipSuccess)return 0;
        d.result.resize(10*count);
        for(int c=0;c<10;c++)if(w.accumulate && w.target[c])std::copy(w.target[c]+first,w.target[c]+last,d.result.begin()+c*count);
        if(hipMemcpy(d.output,d.result.data(),10*count*sizeof(double),hipMemcpyHostToDevice)!=hipSuccess)return 0;
        if(w.kind<=OPENMX_GPU_DM_NC) {
            std::vector<double> weights(3*size_t(w.states),0);
            std::copy(w.occ,w.occ+w.states,weights.begin());
            if(w.energy)std::copy(w.energy,w.energy+w.states,weights.begin()+w.states);
            if(w.partial)std::copy(w.partial,w.partial+w.states,weights.begin()+2*w.states);
            if(hipMalloc(reinterpret_cast<void**>(&d.weights),weights.size()*sizeof(double))!=hipSuccess ||
               hipMemcpy(d.weights,weights.data(),weights.size()*sizeof(double),hipMemcpyHostToDevice)!=hipSuccess)return 0;
        }
        if(w.phase) {
            std::vector<double> phases(2*count);
            for(size_t i=first;i<last;i++) {
                const int p=w.phase[i];if(p<0 || p>=w.phases)return 0;
                phases[i-first]=w.real[p];phases[count+i-first]=w.imag[p];
            }
            if(hipMalloc(reinterpret_cast<void**>(&d.phase),phases.size()*sizeof(double))!=hipSuccess ||
               hipMemcpy(d.phase,phases.data(),phases.size()*sizeof(double),hipMemcpyHostToDevice)!=hipSuccess)return 0;
        }
    } else if(phase==OPENMX_GPU_WORKER_EXECUTE) {
        kernel<<<unsigned(std::min(size_t(65535),(count+127)/128)),128>>>(w.kind,d.n,w.states,d.stride,int(count),d.vectors.pointer,d.indices,
            d.weights,d.phase,d.output,w.phase!=nullptr,w.energy!=nullptr,w.partial!=nullptr);
        if(hipGetLastError()!=hipSuccess || hipMemcpy(d.result.data(),d.output,10*count*sizeof(double),hipMemcpyDeviceToHost)!=hipSuccess)return 0;
    } else if(phase==OPENMX_GPU_WORKER_COMMIT) {
        for(int c=0;c<10;c++)if(w.target[c])std::copy(d.result.begin()+c*count,d.result.begin()+(c+1)*count,w.target[c]+first);
    } else if(phase==OPENMX_GPU_WORKER_RELEASE) {
        if(d.owned_vectors)hipFree(d.owned_vectors);else openmx_gpu_read_view_release(&d.vectors);
        if(d.indices)hipFree(d.indices);if(d.weights)hipFree(d.weights);if(d.phase)hipFree(d.phase);if(d.output)hipFree(d.output);
    }
    return 1;
}
/* A bounded CPU row needs at most four state vectors. A late GPU OOM
   must not recreate the complete eigenvector matrix on either processor. */
int cpu_rows(Work &w)
{
    const int states=w.kind>=OPENMX_GPU_DM_GATHER?0:w.states;
    const size_t width=w.kind==OPENMX_GPU_DM_REAL?sizeof(double):sizeof(Z);
    std::vector<double> storage(std::max(size_t(32),size_t(4)*states*width/sizeof(double)),0),weights(3*size_t(states),0);
    std::vector<double> result(10*size_t(w.entries),0);
    if(states){std::copy(w.occ,w.occ+states,weights.begin());if(w.energy)std::copy(w.energy,w.energy+states,weights.begin()+states);if(w.partial)std::copy(w.partial,w.partial+states,weights.begin()+2*states);}
    for(int p=0;p<w.entries;p++){
        if(w.basis0[p]<0 || w.basis0[p]>=w.n || w.basis1[p]<0 || w.basis1[p]>=w.n)return 0;
        auto copy=[&](size_t target,size_t source,size_t bytes){
            void *dst=reinterpret_cast<char*>(storage.data())+target*width;
            const void *src=static_cast<const char*>(w.vectors)+source*width;
            if(w.host){std::memcpy(dst,src,bytes);return true;}
            return hipMemcpy(dst,src,bytes,hipMemcpyDeviceToHost)==hipSuccess;
        };
        if(w.kind>=OPENMX_GPU_DM_GATHER){
            if(!copy(4,size_t(w.basis0[p])+size_t(w.basis1[p])*2*w.n,sizeof(Z)) ||
               !copy(14,size_t(w.basis0[p]+w.n)+size_t(w.basis1[p]+w.n)*2*w.n,sizeof(Z)) ||
               !copy(12,size_t(w.basis0[p])+size_t(w.basis1[p]+w.n)*2*w.n,sizeof(Z)))return 0;
        }else for(int spin=0;spin<(w.kind==OPENMX_GPU_DM_NC?2:1);spin++){
            if(!copy(size_t(2*spin)*states,size_t(w.basis0[p]+spin*w.n)*w.stride,size_t(states)*width) ||
               !copy(size_t(2*spin+1)*states,size_t(w.basis1[p]+spin*w.n)*w.stride,size_t(states)*width))return 0;
        }
        int indices[2]={0,1};double phases[2]={1,0},output[10]={0};
        if(w.phase){int ph=w.phase[p];if(ph<0 || ph>=w.phases)return 0;phases[0]=w.real[ph];phases[1]=w.imag[ph];}
        openmx_gpu_dm_entry(w.kind,2,states,states,1,storage.data(),indices,weights.data(),phases,output,w.phase!=nullptr,w.energy!=nullptr,w.partial!=nullptr,0);
        for(int c=0;c<10;c++)result[size_t(c)*w.entries+p]=output[c];
    }
    for(int c=0;c<10;c++)if(w.target[c])for(int p=0;p<w.entries;p++){
        if(w.accumulate)w.target[c][p]+=result[size_t(c)*w.entries+p];
        else w.target[c][p]=result[size_t(c)*w.entries+p];
    }
    return 1;
}
int safe_worker(void *w,int i,int p){try{return worker(w,i,p);}catch(const std::bad_alloc&){return 0;}}
}
static int dm_entries(int kind,int n,int states,int stride,int entries,const void *vectors,int host_vectors,
    const int *basis0,const int *basis1,const int *phase_index,int phase_count,const double *phase_r,const double *phase_i,
    const double *occupation,const double *energy_weight,const double *partial_weight,double *const outputs[10],int accumulate,int cpu)
{
    if(kind<OPENMX_GPU_DM_REAL || kind>OPENMX_GPU_EDM_GATHER || n<1 || entries<1 || !vectors || !basis0 || !basis1 || !outputs)return 0;
    if(kind<=OPENMX_GPU_DM_NC && (states<1 || stride<states || !occupation))return 0;
    if(n>2147483647/2 || (phase_index && (phase_count<1 || !phase_r || !phase_i)))return 0;
    for(int p=0;p<entries;p++)if(basis0[p]<0 || basis0[p]>=n || basis1[p]<0 || basis1[p]>=n ||
        (phase_index && (phase_index[p]<0 || phase_index[p]>=phase_count)))return 0;
    Work w{};w.kind=kind;w.n=n;w.states=states;w.stride=stride;w.entries=entries;w.phases=phase_count;
    w.vectors=vectors;w.host=host_vectors;w.basis0=basis0;w.basis1=basis1;w.phase=phase_index;w.real=phase_r;w.imag=phase_i;
    w.occ=occupation;w.energy=energy_weight;w.partial=partial_weight;w.accumulate=accumulate;
    for(int i=0;i<10;i++)w.target[i]=outputs[i];
    w.vector_bytes=kind>=OPENMX_GPU_DM_GATHER?size_t(2*n)*size_t(2*n)*sizeof(Z):
        ((size_t(n)*(kind==OPENMX_GPU_DM_NC?2:1)-1)*stride+states)* (kind==OPENMX_GPU_DM_REAL?sizeof(double):sizeof(Z));
    if(hipGetDevice(&w.origin)!=hipSuccess)w.origin=0;
    int ok=openmx_gpu_workers_capacity(&w.workers,entries,2,capacity_cost,&w);
    if(ok)ok=openmx_gpu_workers_run(&w.workers,safe_worker,&w,"density-matrix-entries");
    if(ok)return 1;
    (void)hipGetLastError();
    if(!cpu)return 0;
    try{return cpu_rows(w);}catch(const std::bad_alloc&){return 0;}
}
extern "C" int openmx_gpu_dm_entries(int kind,int n,int states,int stride,int entries,const void *vectors,int host_vectors,
    const int *basis0,const int *basis1,const int *phase_index,int phase_count,const double *phase_r,const double *phase_i,
    const double *occupation,const double *energy_weight,const double *partial_weight,double *const outputs[10],int accumulate)
{ return dm_entries(kind,n,states,stride,entries,vectors,host_vectors,basis0,basis1,phase_index,phase_count,phase_r,phase_i,occupation,energy_weight,partial_weight,outputs,accumulate,1); }
extern "C" int openmx_gpu_dm_try_entries(int kind,int n,int states,int stride,int entries,const void *vectors,int host_vectors,
    const int *basis0,const int *basis1,const int *phase_index,int phase_count,const double *phase_r,const double *phase_i,
    const double *occupation,const double *energy_weight,const double *partial_weight,double *const outputs[10],int accumulate)
{ return dm_entries(kind,n,states,stride,entries,vectors,host_vectors,basis0,basis1,phase_index,phase_count,phase_r,phase_i,occupation,energy_weight,partial_weight,outputs,accumulate,0); }

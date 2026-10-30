#include "openmx_gpu_backend.h"
#include "openmx_gpu_complex.h"
#include "openmx_gpu_runtime.h"
#include "openmx_gpu_dm_entry.h"
#include "openmx_gpu_workers.h"
#include "openmx_gpu_views.h"
#include "openmx_gpu_memory.hpp"


#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>
#include <new>
#include <limits>
#include <cstdint>
#include <memory>
namespace {
using Z=OpenMXGpuComplex;
using openmx::gpu::DeviceBuffer;
struct Work {
    OpenMXGpuWorkers workers{};
    int kind,n,states,stride,entries,phases,host,accumulate,origin,output_channels;
    const void *vectors;
    size_t vector_bytes;
    const int *basis0,*basis1,*phase;
    const double *real,*imag,*occ,*energy,*partial;
    double *target[10]{};
    int layout_count=0;
    std::vector<double> host_weights;
    struct Device {
        OpenMXGpuReadView vectors{};
        DeviceBuffer<unsigned char> owned_vectors;
        DeviceBuffer<int> indices;
        DeviceBuffer<double> weights,phase,output;
        std::unique_ptr<double[]> result;
        std::vector<int> basis,host_indices;
        std::vector<double> host_phase;
        std::vector<unsigned char> host_panel;
        const void *host_vectors=nullptr;
        int n=0,stride=0;
        bool identity=false;
    } device[OPENMX_GPU_WORKERS_MAX];
};
__global__ void kernel(int kind,int n,int states,int stride,int entries,const void *vectors,
    const int *indices,const double *weights,const double *phases,double *output,int phased,int edm,int pdm)
{
    for(int p=int(blockIdx.x*blockDim.x+threadIdx.x);p<entries;p+=int(blockDim.x*gridDim.x))
        openmx_gpu_dm_entry(kind,n,states,stride,entries,vectors,indices,weights,phases,output,phased,edm,pdm,p);
}
/* Gathering a completed dense product needs no state reduction or ten-channel
   scratch. Keep the same channel order and private publication boundary. */
__global__ void gather_kernel(int kind,int n,int entries,const Z *vectors,
    const int *indices,double *output,int channels,int accumulate)
{
    for(int p=int(blockIdx.x*blockDim.x+threadIdx.x);p<entries;p+=int(blockDim.x*gridDim.x)){
        const size_t a=indices[p],b=indices[entries+p],n2=2*size_t(n);
        const Z x=vectors[a+b*n2],y=vectors[a+n+(b+n)*n2],z=vectors[a+(b+n)*n2];
        const double values[10]={x.x,y.x,z.x,z.y,
            kind==OPENMX_GPU_DM_GATHER?x.y:0,kind==OPENMX_GPU_DM_GATHER?y.y:0,0,0,0,0};
        for(int channel=0;channel<channels;channel++){
            double *target=output+size_t(channel)*entries+p;
            if(accumulate)*target+=values[channel];else *target=values[channel];
        }
    }
}
/* Capacity probes and execution share one layout for the candidate worker
   count. Build an inverse map in O(n+entries), instead of sorting sparse
   entries and binary-searching each entry again under the shared GPU lock.
   Retain only the current candidate's host storage. */
void prepare_layouts(Work &w,int workers)
{
    if(w.layout_count==workers)return;
    w.layout_count=0;
    for(auto &d:w.device){
        std::vector<int>().swap(d.basis);
        std::vector<int>().swap(d.host_indices);
    }
    const bool compact=w.host || workers>1;
    std::vector<int> inverse(compact?size_t(w.n):0,-1);
    for(int worker=0;worker<workers;worker++){
        auto &d=w.device[worker];
        const size_t first=openmx_gpu_worker_cut(w.entries,worker,workers);
        const size_t last=openmx_gpu_worker_cut(w.entries,worker+1,workers),count=last-first;
        d.n=w.n;d.stride=w.stride;
        d.identity=true;
        if(compact){
            std::fill(inverse.begin(),inverse.end(),-1);
            for(size_t i=first;i<last;i++)inverse[w.basis0[i]]=inverse[w.basis1[i]]=0;
            for(int i=0;i<w.n;i++)if(inverse[i]==0){
                inverse[i]=int(d.basis.size());d.basis.push_back(i);
            }
            d.n=int(d.basis.size());d.stride=w.states;
            d.identity=d.n==w.n;
            if(!d.identity){
                d.host_indices.resize(2*count);
                for(size_t i=first;i<last;i++){
                    d.host_indices[i-first]=inverse[w.basis0[i]];
                    d.host_indices[count+i-first]=inverse[w.basis1[i]];
                }
            }
        }
    }
    w.layout_count=workers;
}
size_t capacity_cost(void *opaque,int worker,int count,int device)
{
    auto &w=*static_cast<Work*>(opaque);
    size_t first=openmx_gpu_worker_cut(w.entries,worker,count),last=openmx_gpu_worker_cut(w.entries,worker+1,count);
    const int channels=w.kind>=OPENMX_GPU_DM_GATHER?w.output_channels:10;
    size_t bytes=(last-first)*(2*sizeof(int)+size_t(channels+2)*sizeof(double))+3*size_t(w.states)*sizeof(double);
    try{
        if((w.host || count>1) && w.kind>=OPENMX_GPU_DM_GATHER)return SIZE_MAX;
        prepare_layouts(w,count);
        if(w.host || count>1){
            const size_t width=w.kind==OPENMX_GPU_DM_REAL?8:16;
            size_t size=w.device[worker].basis.size()*(w.kind==OPENMX_GPU_DM_NC?2:1)*size_t(w.states);
            if(size>(SIZE_MAX-bytes)/width)return SIZE_MAX;bytes+=size*width;
        }else if(device!=w.origin){
            /* read_view may need a full private copy when P2P is unavailable. */
            if(w.vector_bytes>SIZE_MAX-bytes)return SIZE_MAX;
            bytes+=w.vector_bytes;
        }
    }catch(const std::bad_alloc&){return SIZE_MAX;}
    return bytes;
}

/* Prepare private host storage before GPU admission, so ranks can pack
   their panels concurrently. Device allocation/execution remain locked. */
void prepare_host(Work &w)
{
    prepare_layouts(w,w.workers.count);
    if(w.kind<=OPENMX_GPU_DM_NC){
        w.host_weights.assign(3*size_t(w.states),0);
        std::copy(w.occ,w.occ+w.states,w.host_weights.begin());
        if(w.energy)std::copy(w.energy,w.energy+w.states,w.host_weights.begin()+w.states);
        if(w.partial)std::copy(w.partial,w.partial+w.states,w.host_weights.begin()+2*w.states);
    }
    for(int worker=0;worker<w.workers.count;worker++){
        auto &d=w.device[worker];
        const size_t first=openmx_gpu_worker_cut(w.entries,worker,w.workers.count);
        const size_t last=openmx_gpu_worker_cut(w.entries,worker+1,w.workers.count),count=last-first;
        if(w.kind<OPENMX_GPU_DM_GATHER)d.result.reset(new double[size_t(w.output_channels)*count]);
        if(w.host){
            const size_t width=w.kind==OPENMX_GPU_DM_REAL?sizeof(double):sizeof(Z);
            const int spins=w.kind==OPENMX_GPU_DM_NC?2:1;
            if(d.n==w.n && w.stride==w.states)d.host_vectors=w.vectors;
            else{
                d.host_panel.resize(size_t(d.n)*spins*w.states*width);
                for(int spin=0;spin<spins;spin++)for(size_t i=0;i<d.basis.size();i++){
                    void *dst=d.host_panel.data()+(i+size_t(spin)*d.n)*w.states*width;
                    const void *src=static_cast<const char*>(w.vectors)+(size_t(d.basis[i]+spin*w.n)*w.stride)*width;
                    std::memcpy(dst,src,size_t(w.states)*width);
                }
                d.host_vectors=d.host_panel.data();
            }
        }
        if(w.phase){
            d.host_phase.resize(2*count);
            for(size_t i=first;i<last;i++){
                d.host_phase[i-first]=w.real[w.phase[i]];
                d.host_phase[count+i-first]=w.imag[w.phase[i]];
            }
        }
    }
}

int worker(void *opaque,int worker,int phase)
{
    auto &w=*static_cast<Work*>(opaque);auto &d=w.device[worker];
    const size_t first=openmx_gpu_worker_cut(w.entries,worker,w.workers.count),last=openmx_gpu_worker_cut(w.entries,worker+1,w.workers.count),count=last-first;
    if(phase==OPENMX_GPU_WORKER_PREPARE) {
        if(w.host || w.workers.count>1){
            if(w.kind>=OPENMX_GPU_DM_GATHER)return 0;
            const size_t width=w.kind==OPENMX_GPU_DM_REAL?sizeof(double):sizeof(Z);
            size_t bytes=d.basis.size()*(w.kind==OPENMX_GPU_DM_NC?2:1)*size_t(w.states)*width;
            if(!d.owned_vectors.allocate(bytes))return 0;
            d.vectors.pointer=d.owned_vectors.get();
            if(w.host){
                if(OpenMXGpuMemcpy(d.owned_vectors.get(),d.host_vectors,bytes,OPENMX_GPU_COPY_HOST_TO_DEVICE)!=OPENMX_GPU_SUCCESS)return 0;
            }else for(int spin=0;spin<(w.kind==OPENMX_GPU_DM_NC?2:1);spin++){
                for(size_t i=0;i<d.basis.size();){
                    size_t end=i+1;
                    if(w.stride==w.states)
                        while(end<d.basis.size() && d.basis[end]==d.basis[end-1]+1)++end;
                    void *dst=d.owned_vectors.get()+(i+spin*d.basis.size())*w.states*width;
                    const void *src=static_cast<const char*>(w.vectors)+(size_t(d.basis[i]+spin*w.n)*w.stride)*width;
                    if(!openmx_gpu_copy_device(dst,src,(end-i)*w.states*width))return 0;
                    i=end;
                }
            }
        }else{
            if(!openmx_gpu_read_view(&d.vectors,w.vectors,w.vector_bytes))return 0;
        }
        const int channels=w.kind>=OPENMX_GPU_DM_GATHER?w.output_channels:10;
        if(!d.indices.allocate(2*count) || !d.output.allocate(size_t(channels)*count))return 0;
        if(d.identity){
            /* A full cluster panel already uses the caller's basis numbers.
               Avoid another full sparse-index array on every MPI rank. */
            if(OpenMXGpuMemcpy(d.indices.get(),w.basis0+first,count*sizeof(int),OPENMX_GPU_COPY_HOST_TO_DEVICE)!=OPENMX_GPU_SUCCESS ||
               OpenMXGpuMemcpy(d.indices.get()+count,w.basis1+first,count*sizeof(int),OPENMX_GPU_COPY_HOST_TO_DEVICE)!=OPENMX_GPU_SUCCESS)return 0;
        }else if(OpenMXGpuMemcpy(d.indices.get(),d.host_indices.data(),2*count*sizeof(int),OPENMX_GPU_COPY_HOST_TO_DEVICE)!=OPENMX_GPU_SUCCESS)return 0;
        if((w.kind<OPENMX_GPU_DM_GATHER || w.accumulate) &&
           OpenMXGpuMemset(d.output.get(),0,size_t(channels)*count*sizeof(double))!=OPENMX_GPU_SUCCESS)return 0;
        for(int channel=0;channel<w.output_channels;channel++)if(w.accumulate && w.target[channel])
            if(OpenMXGpuMemcpy(d.output.get()+size_t(channel)*count,w.target[channel]+first,count*sizeof(double),
                OPENMX_GPU_COPY_HOST_TO_DEVICE)!=OPENMX_GPU_SUCCESS)return 0;
        if(w.kind<=OPENMX_GPU_DM_NC) {
            if(!d.weights.allocate(w.host_weights.size()) ||
               OpenMXGpuMemcpy(d.weights.get(),w.host_weights.data(),w.host_weights.size()*sizeof(double),OPENMX_GPU_COPY_HOST_TO_DEVICE)!=OPENMX_GPU_SUCCESS)return 0;
        }
        if(w.phase) {
            if(!d.phase.allocate(d.host_phase.size()) ||
               OpenMXGpuMemcpy(d.phase.get(),d.host_phase.data(),d.host_phase.size()*sizeof(double),OPENMX_GPU_COPY_HOST_TO_DEVICE)!=OPENMX_GPU_SUCCESS)return 0;
        }
    } else if(phase==OPENMX_GPU_WORKER_EXECUTE) {
        const unsigned blocks=unsigned(std::min(size_t(65535),(count+127)/128));
        if(w.kind>=OPENMX_GPU_DM_GATHER)
            gather_kernel<<<blocks,128>>>(w.kind,d.n,int(count),static_cast<const Z*>(d.vectors.pointer),
                d.indices.get(),d.output.get(),w.output_channels,w.accumulate);
        else
            kernel<<<blocks,128>>>(w.kind,d.n,w.states,d.stride,int(count),d.vectors.pointer,d.indices.get(),
                d.weights.get(),d.phase.get(),d.output.get(),w.phase!=nullptr,w.energy!=nullptr,w.partial!=nullptr);
        if(OpenMXGpuPeekAtLastError()!=OPENMX_GPU_SUCCESS)return 0;
        if(w.kind<OPENMX_GPU_DM_GATHER && OpenMXGpuMemcpy(d.result.get(),d.output.get(),
            size_t(w.output_channels)*count*sizeof(double),OPENMX_GPU_COPY_DEVICE_TO_HOST)!=OPENMX_GPU_SUCCESS)return 0;
    } else if(phase==OPENMX_GPU_WORKER_COMMIT) {
        /* The completed private GPU array is sufficient staging for a gather.
           Publish only after every worker's kernel and synchronization succeed;
           publication failures are fatal in the common executor. This avoids a
           second large host allocation, page registration and host-to-host copy. */
        for(int channel=0;channel<w.output_channels;channel++)if(w.target[channel]){
            if(w.kind>=OPENMX_GPU_DM_GATHER){
                if(OpenMXGpuMemcpy(w.target[channel]+first,d.output.get()+size_t(channel)*count,
                    count*sizeof(double),OPENMX_GPU_COPY_DEVICE_TO_HOST)!=OPENMX_GPU_SUCCESS)return 0;
            }
        }
    } else if(phase==OPENMX_GPU_WORKER_RELEASE) {
        openmx_gpu_read_view_release(&d.vectors);d.owned_vectors.reset();
        d.indices.reset();d.weights.reset();d.phase.reset();d.output.reset();
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
            return OpenMXGpuMemcpy(dst,src,bytes,OPENMX_GPU_COPY_DEVICE_TO_HOST)==OPENMX_GPU_SUCCESS;
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
    for(int channel=0;channel<10;channel++){
        w.target[channel]=outputs[channel];
        if(outputs[channel])w.output_channels=channel+1;
    }
    if(!w.output_channels)return 1;
    /* A failed caller launch is not a capacity failure. Preserve it and do
       not turn the operation into a successful CPU result. */
    if(OpenMXGpuPeekAtLastError()!=OPENMX_GPU_SUCCESS)return 0;
    w.vector_bytes=kind>=OPENMX_GPU_DM_GATHER?size_t(2*n)*size_t(2*n)*sizeof(Z):
        ((size_t(n)*(kind==OPENMX_GPU_DM_NC?2:1)-1)*stride+states)* (kind==OPENMX_GPU_DM_REAL?sizeof(double):sizeof(Z));
    if(OpenMXGpuGetDevice(&w.origin)!=OPENMX_GPU_SUCCESS)return 0;
    int ok=openmx_gpu_workers_capacity_device(&w.workers,entries,2,capacity_cost,&w);
    if(ok)try{prepare_host(w);}catch(const std::bad_alloc&){ok=0;}
    if(ok)ok=openmx_gpu_workers_run_prepared(&w.workers,safe_worker,&w,"density-matrix-entries");
    if(ok){
        /* All private results are ready and device ownership is released.
           Publish host-only accumulators without holding the GPU lock. */
        if(kind<OPENMX_GPU_DM_GATHER)for(int worker=0;worker<w.workers.count;worker++){
            const size_t first=openmx_gpu_worker_cut(entries,worker,w.workers.count);
            const size_t count=openmx_gpu_worker_cut(entries,worker+1,w.workers.count)-first;
            for(int channel=0;channel<w.output_channels;channel++)if(w.target[channel])
                std::copy_n(w.device[worker].result.get()+size_t(channel)*count,count,w.target[channel]+first);
        }
        return 1;
    }
    const auto pending=OpenMXGpuPeekAtLastError();
    if(pending==OPENMX_GPU_OUT_OF_MEMORY) (void)OpenMXGpuGetLastError();
    else if(pending!=OPENMX_GPU_SUCCESS) return 0;
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

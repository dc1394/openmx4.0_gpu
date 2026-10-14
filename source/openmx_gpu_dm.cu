#include "openmx_gpu_dm.h"
#include "openmx_gpu_workers.h"
#include "openmx_gpu_views.h"
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>
#include <new>
namespace {
using Z=cuDoubleComplex;
struct Work {
    OpenMXGpuWorkers workers{};
    int kind,n,states,stride,entries,phases,host,accumulate;
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
    } device[OPENMX_GPU_WORKERS_MAX];
};
__global__ void kernel(int kind,int n,int states,int stride,int entries,
    const void *vectors,const int *indices,const double *weights,const double *phases,
    double *output,int phased,int edm,int pdm)
{
    for(int p=int(blockIdx.x*blockDim.x+threadIdx.x);p<entries;p+=int(blockDim.x*gridDim.x)) {
        const int ia=indices[p],ib=indices[entries+p];
        const double co=phased?phases[p]:1,si=phased?phases[entries+p]:0;
        double r[10]={0};
        if(kind==OPENMX_GPU_DM_GATHER || kind==OPENMX_GPU_EDM_GATHER) {
            const Z *v=static_cast<const Z*>(vectors);
            Z x=v[size_t(ia)+size_t(ib)*2*n],y=v[size_t(ia+n)+size_t(ib+n)*2*n],z=v[size_t(ia)+size_t(ib+n)*2*n];
            r[0]=x.x;r[1]=y.x;r[2]=z.x;r[3]=z.y;
            if(kind==OPENMX_GPU_DM_GATHER){r[4]=x.y;r[5]=y.y;}
        } else if(kind==OPENMX_GPU_DM_REAL) {
            const double *v=static_cast<const double*>(vectors);
            for(int k=0;k<states;k++) {
                const double a=v[size_t(ia)*stride+k],b=v[size_t(ib)*stride+k];
                const double product=weights[k]*a*b;
                r[0]+=product;r[1]+=product*weights[states+k];
                if(pdm)r[2]+=weights[2*states+k]*a*b;
            }
        } else if(kind==OPENMX_GPU_DM_COL) {
            const Z *v=static_cast<const Z*>(vectors);
            double d1=0,d2=0,d3=0,d4=0;
            for(int k=0;k<states;k++) {
                const Z a=v[size_t(ia)*stride+k],b=v[size_t(ib)*stride+k];
                const double w=weights[k],ar=a.x*w,ai=a.y*w,br=b.x*w,bi=b.y*w;
                const double re=ar*br+ai*bi,im=ar*bi-ai*br;
                d1+=re;d2+=im;d3+=re*weights[states+k];d4+=im*weights[states+k];
            }
            r[0]=co*d1-si*d2;r[1]=co*d3-si*d4;
        } else {
            const Z *v=static_cast<const Z*>(vectors);
            double e11i=0,e22i=0;
            for(int k=0;k<states;k++) {
                const Z a=v[size_t(ia)*stride+k],b=v[size_t(ib)*stride+k],c=v[size_t(ia+n)*stride+k],d=v[size_t(ib+n)*stride+k];
                const double x=a.x*b.x+a.y*b.y,xi=a.x*b.y-a.y*b.x;
                const double y=c.x*d.x+c.y*d.y,yi=c.x*d.y-c.y*d.x;
                const double z=a.x*d.x+a.y*d.y,zi=a.x*d.y-a.y*d.x;
                const double w=weights[k],ew=weights[states+k];
                r[0]+=w*x;r[1]+=w*y;r[2]+=w*z;r[3]+=w*zi;r[4]+=w*xi;r[5]+=w*yi;
                if(edm){r[6]+=ew*x;r[7]+=ew*y;r[8]+=ew*z;r[9]+=ew*zi;e11i+=ew*xi;e22i+=ew*yi;}
            }
            if(phased) {
                double x=r[0],y=r[1],z=r[2];
                r[0]=co*x-si*r[4];r[4]=co*r[4]+si*x;
                r[1]=co*y-si*r[5];r[5]=co*r[5]+si*y;
                r[2]=co*z-si*r[3];r[3]=co*r[3]+si*z;
                if(edm){
                    r[6]=co*r[6]-si*e11i;r[7]=co*r[7]-si*e22i;
                    double re=r[8];r[8]=co*re-si*r[9];r[9]=co*r[9]+si*re;
                }
            }
        }
        for(int c=0;c<10;c++)output[size_t(c)*entries+p]+=r[c];
    }
}
int worker(void *opaque,int worker,int phase)
{
    auto &w=*static_cast<Work*>(opaque);auto &d=w.device[worker];
    const size_t first=openmx_gpu_worker_cut(w.entries,worker,w.workers.count),last=openmx_gpu_worker_cut(w.entries,worker+1,w.workers.count),count=last-first;
    if(phase==OPENMX_GPU_WORKER_PREPARE) {
        for(size_t p=first;p<last;p++)
            if(w.basis0[p]<0 || w.basis0[p]>=w.n || w.basis1[p]<0 || w.basis1[p]>=w.n)return 0;
        if(w.host) {
            if(cudaMalloc(&d.owned_vectors,w.vector_bytes)!=cudaSuccess || cudaMemcpy(d.owned_vectors,w.vectors,w.vector_bytes,cudaMemcpyHostToDevice)!=cudaSuccess)return 0;
            d.vectors.pointer=d.owned_vectors;
        } else if(!openmx_gpu_read_view(&d.vectors,w.vectors,w.vector_bytes))return 0;
        if(cudaMalloc(reinterpret_cast<void**>(&d.indices),2*count*sizeof(int))!=cudaSuccess ||
           cudaMalloc(reinterpret_cast<void**>(&d.output),10*count*sizeof(double))!=cudaSuccess)return 0;
        if(cudaMemcpy(d.indices,w.basis0+first,count*sizeof(int),cudaMemcpyHostToDevice)!=cudaSuccess ||
           cudaMemcpy(d.indices+count,w.basis1+first,count*sizeof(int),cudaMemcpyHostToDevice)!=cudaSuccess)return 0;
        d.result.resize(10*count);
        for(int c=0;c<10;c++)if(w.accumulate && w.target[c])std::copy(w.target[c]+first,w.target[c]+last,d.result.begin()+c*count);
        if(cudaMemcpy(d.output,d.result.data(),10*count*sizeof(double),cudaMemcpyHostToDevice)!=cudaSuccess)return 0;
        if(w.kind<=OPENMX_GPU_DM_NC) {
            std::vector<double> weights(3*size_t(w.states),0);
            std::copy(w.occ,w.occ+w.states,weights.begin());
            if(w.energy)std::copy(w.energy,w.energy+w.states,weights.begin()+w.states);
            if(w.partial)std::copy(w.partial,w.partial+w.states,weights.begin()+2*w.states);
            if(cudaMalloc(reinterpret_cast<void**>(&d.weights),weights.size()*sizeof(double))!=cudaSuccess ||
               cudaMemcpy(d.weights,weights.data(),weights.size()*sizeof(double),cudaMemcpyHostToDevice)!=cudaSuccess)return 0;
        }
        if(w.phase) {
            std::vector<double> phases(2*count);
            for(size_t i=first;i<last;i++) {
                const int p=w.phase[i];if(p<0 || p>=w.phases)return 0;
                phases[i-first]=w.real[p];phases[count+i-first]=w.imag[p];
            }
            if(cudaMalloc(reinterpret_cast<void**>(&d.phase),phases.size()*sizeof(double))!=cudaSuccess ||
               cudaMemcpy(d.phase,phases.data(),phases.size()*sizeof(double),cudaMemcpyHostToDevice)!=cudaSuccess)return 0;
        }
    } else if(phase==OPENMX_GPU_WORKER_EXECUTE) {
        kernel<<<unsigned(std::min(size_t(65535),(count+127)/128)),128>>>(w.kind,w.n,w.states,w.stride,int(count),d.vectors.pointer,d.indices,
            d.weights,d.phase,d.output,w.phase!=nullptr,w.energy!=nullptr,w.partial!=nullptr);
        if(cudaGetLastError()!=cudaSuccess || cudaMemcpy(d.result.data(),d.output,10*count*sizeof(double),cudaMemcpyDeviceToHost)!=cudaSuccess)return 0;
    } else if(phase==OPENMX_GPU_WORKER_COMMIT) {
        for(int c=0;c<10;c++)if(w.target[c])std::copy(d.result.begin()+c*count,d.result.begin()+(c+1)*count,w.target[c]+first);
    } else if(phase==OPENMX_GPU_WORKER_RELEASE) {
        if(d.owned_vectors)cudaFree(d.owned_vectors);else openmx_gpu_read_view_release(&d.vectors);
        if(d.indices)cudaFree(d.indices);if(d.weights)cudaFree(d.weights);if(d.phase)cudaFree(d.phase);if(d.output)cudaFree(d.output);
    }
    return 1;
}
int safe_worker(void *w,int i,int p){try{return worker(w,i,p);}catch(const std::bad_alloc&){return 0;}}
}
extern "C" int openmx_gpu_dm_entries(int kind,int n,int states,int stride,int entries,const void *vectors,int host_vectors,
    const int *basis0,const int *basis1,const int *phase_index,int phase_count,const double *phase_r,const double *phase_i,
    const double *occupation,const double *energy_weight,const double *partial_weight,double *const outputs[10],int accumulate)
{
    if(kind<OPENMX_GPU_DM_REAL || kind>OPENMX_GPU_EDM_GATHER || n<1 || entries<1 || !vectors || !basis0 || !basis1 || !outputs)return 0;
    if(kind<=OPENMX_GPU_DM_NC && (states<1 || stride<states || !occupation))return 0;
    if(n>2147483647/2 || (phase_index && (phase_count<1 || !phase_r || !phase_i)))return 0;
    Work w{};w.kind=kind;w.n=n;w.states=states;w.stride=stride;w.entries=entries;w.phases=phase_count;
    w.vectors=vectors;w.host=host_vectors;w.basis0=basis0;w.basis1=basis1;w.phase=phase_index;w.real=phase_r;w.imag=phase_i;
    w.occ=occupation;w.energy=energy_weight;w.partial=partial_weight;w.accumulate=accumulate;
    for(int i=0;i<10;i++)w.target[i]=outputs[i];
    w.vector_bytes=kind>=OPENMX_GPU_DM_GATHER?size_t(2*n)*size_t(2*n)*sizeof(Z):
        ((size_t(n)*(kind==OPENMX_GPU_DM_NC?2:1)-1)*stride+states)* (kind==OPENMX_GPU_DM_REAL?sizeof(double):sizeof(Z));
    openmx_gpu_workers_scope(&w.workers,entries);
    if(w.workers.count<2)return 0;
    return openmx_gpu_workers_run(&w.workers,safe_worker,&w,"density-matrix-entries");
}

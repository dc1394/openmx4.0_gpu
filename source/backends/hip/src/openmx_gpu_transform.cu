#define OPENMX_GPU_TRANSFORM_IMPLEMENTATION
#include "openmx_gpu_transform.h"
#include "openmx_gpu_workers.h"
#include "openmx_gpu_views.h"
#include "hip_runtime_compat.h"
#include "hip_complex_compat.h"
#include <cmath>
#include <cstring>
#include <limits>

namespace {
using Z=hipDoubleComplex;
struct Transform {
    OpenMXGpuWorkers workers{};
    int op=0,complex=0,rows=0,cols=0,lda=0,ldb=0,ldout=0,complex_weights=0;
    int spinor=0,real_parts=0,scatter=0,entries=0,format=0,imaginary=0;
    double floor=0;
    const void *source[6]{};
    size_t bytes[6]{};
    void *target=nullptr,*output[OPENMX_GPU_WORKERS_MAX]{};
    OpenMXGpuReadView input[OPENMX_GPU_WORKERS_MAX][6]{};
};
__device__ double product(double a,double b) {return a*b;}
__device__ Z product(Z a,Z b) {return make_hipDoubleComplex(a.x*b.x-a.y*b.y,a.x*b.y+a.y*b.x);}
__device__ Z product(Z a,double b) {return make_hipDoubleComplex(a.x*b,a.y*b);}
__device__ double sum(double a,double b) {return a+b;}
__device__ Z sum(Z a,Z b) {return make_hipDoubleComplex(a.x+b.x,a.y+b.y);}
__device__ double zero(double) {return 0;}
__device__ Z zero(Z) {return make_hipDoubleComplex(0,0);}
__device__ double hermitian(double a,double b,int diagonal) {return diagonal?a:0.5*(a+b);}
__device__ Z hermitian(Z a,Z b,int diagonal) {return make_hipDoubleComplex(diagonal?a.x:0.5*(a.x+b.x),diagonal?0:0.5*(a.y-b.y));}
__device__ double invsqrt(double a,double floor) {return 1.0/sqrt(a<floor?floor:a);}
__device__ Z invsqrt(Z a,double floor) {return make_hipDoubleComplex(invsqrt(a.x,floor),0);}

template<class T> __global__ void transform_kernel(int op,int rows,int cols,int lda,int ldb,int first,int last,
    const T *a,const void *b,T *out,int complex_weights,double floor)
{
    const size_t count=size_t(rows)*size_t(last-first);
    for(size_t index=size_t(blockIdx.x)*blockDim.x+threadIdx.x;index<count;index+=size_t(blockDim.x)*gridDim.x) {
        int row=int(index%rows),col=first+int(index/rows);
        T value{};
        if(op!=OPENMX_GPU_ZERO) {
            value=a[op==OPENMX_GPU_TRANSPOSE?size_t(col)+size_t(row)*lda:size_t(row)+size_t(col)*lda];
            if(op==OPENMX_GPU_ADD) value=sum(value,static_cast<const T*>(b)[size_t(row)+size_t(col)*ldb]);
            else if(op==OPENMX_GPU_HERMITIAN) value=hermitian(value,a[size_t(col)+size_t(row)*lda],row==col);
            else if(op==OPENMX_GPU_INVSQRT) value=invsqrt(value,floor);
            else if(op==OPENMX_GPU_SCALE_ROWS || op==OPENMX_GPU_SCALE_COLUMNS) {
                int weight=op==OPENMX_GPU_SCALE_ROWS?row:col;
                if(complex_weights) value=product(value,static_cast<const T*>(b)[weight]);
                else value=product(value,static_cast<const double*>(b)[weight]);
            }
        }
        out[index]=value;
    }
}
__global__ void spinor_kernel(int n,int first,int last,int real_parts,
    const void *r11,const void *r22,const void *r12,const double *i11,const double *i22,const double *i12,Z *out)
{
    int n2=2*n;
    size_t count=size_t(n2)*size_t(last-first);
    for(size_t index=size_t(blockIdx.x)*blockDim.x+threadIdx.x;index<count;index+=size_t(blockDim.x)*gridDim.x) {
        int row=int(index%n2),col=first+int(index/n2),r=row%n,c=col%n;
        bool lower=row>=n && col<n;
        size_t j=lower?size_t(c)+size_t(r)*n:size_t(r)+size_t(c)*n;
        const void *real=(row<n && col<n)?r11:(row>=n && col>=n)?r22:r12;
        const double *imag=(row<n && col<n)?i11:(row>=n && col>=n)?i22:i12;
        Z value=real_parts?make_hipDoubleComplex(static_cast<const double*>(real)[j],imag[j]):static_cast<const Z*>(real)[j];
        if(lower)value.y=-value.y;
        out[index]=value;
    }
}
__global__ void scatter_kernel(int entries,int format,int imaginary,size_t first,size_t last,
    const int *index,const double *values,const double *pr,const double *pi,void *output)
{
    for(size_t p=size_t(blockIdx.x)*blockDim.x+threadIdx.x;p<size_t(entries);p+=size_t(blockDim.x)*gridDim.x) {
        if(format==0) {
            const int target=index[p];
            if(target>=0 && size_t(target)>=first && size_t(target)<last)
                atomicAdd(static_cast<double*>(output)+size_t(target)-first,values[p]);
        } else {
            const int *e=index+p*format;
            const int value_index=e[0],phase=e[format-1];
            const double re=values[value_index]*(imaginary?-pi[phase]:pr[phase]);
            const double im=values[value_index]*(imaginary?pr[phase]:pi[phase]);
            const int target=e[1];
            if(target>=0 && size_t(target)>=first && size_t(target)<last) {
                Z *out=static_cast<Z*>(output)+size_t(target)-first;
                atomicAdd(&out->x,re);atomicAdd(&out->y,im);
            }
            if(format==7) {
                const int mirror=e[2];
                if(mirror>=0 && size_t(mirror)>=first && size_t(mirror)<last) {
                    Z *out=static_cast<Z*>(output)+size_t(mirror)-first;
                    atomicAdd(&out->x,re);atomicAdd(&out->y,-im);
                }
            }
        }
    }
}
int worker(void *opaque,int device,int phase)
{
    auto &w=*static_cast<Transform*>(opaque);
    const int first=int(openmx_gpu_worker_cut(w.cols,device,w.workers.count));
    const int last=int(openmx_gpu_worker_cut(w.cols,device+1,w.workers.count));
    const size_t width=w.complex?sizeof(Z):sizeof(double),count=size_t(w.rows)*size_t(last-first);
    if(phase==OPENMX_GPU_WORKER_PREPARE) {
        for(int i=0;i<6;i++)if(!openmx_gpu_read_view(&w.input[device][i],w.source[i],w.bytes[i]))return 0;
        if(hipMalloc(&w.output[device],count*width)!=hipSuccess)return 0;
    } else if(phase==OPENMX_GPU_WORKER_EXECUTE) {
        const auto *v=w.input[device];
        const unsigned blocks=unsigned(count/256+1>65535?65535:count/256+1);
        if(w.scatter) {
            if(hipMemset(w.output[device],0,count*width)!=hipSuccess)return 0;
            scatter_kernel<<<blocks,256>>>(w.entries,w.format,w.imaginary,size_t(first)*w.rows,size_t(last)*w.rows,
                static_cast<const int*>(v[0].pointer),static_cast<const double*>(v[1].pointer),
                static_cast<const double*>(v[2].pointer),static_cast<const double*>(v[3].pointer),w.output[device]);
        } else if(w.spinor) {
            spinor_kernel<<<blocks,256>>>(w.rows/2,first,last,w.real_parts,v[0].pointer,v[1].pointer,v[2].pointer,
                static_cast<const double*>(v[3].pointer),static_cast<const double*>(v[4].pointer),
                static_cast<const double*>(v[5].pointer),static_cast<Z*>(w.output[device]));
        } else if(w.complex) {
            transform_kernel<<<blocks,256>>>(w.op,w.rows,w.cols,w.lda,w.ldb,first,last,
                static_cast<const Z*>(v[0].pointer),v[1].pointer,static_cast<Z*>(w.output[device]),w.complex_weights,w.floor);
        } else {
            transform_kernel<<<blocks,256>>>(w.op,w.rows,w.cols,w.lda,w.ldb,first,last,
                static_cast<const double*>(v[0].pointer),v[1].pointer,static_cast<double*>(w.output[device]),w.complex_weights,w.floor);
        }
        return hipGetLastError()==hipSuccess;
    } else if(phase==OPENMX_GPU_WORKER_COMMIT) {
        if(w.ldout==w.rows) return openmx_gpu_copy_device(static_cast<char*>(w.target)+size_t(first)*w.ldout*width,w.output[device],count*width);
        for(int col=first;col<last;col++)
            if(!openmx_gpu_copy_device(static_cast<char*>(w.target)+size_t(col)*w.ldout*width,
                static_cast<const char*>(w.output[device])+size_t(col-first)*w.rows*width,size_t(w.rows)*width))return 0;
    } else if(phase==OPENMX_GPU_WORKER_RELEASE) {
        if(w.output[device])hipFree(w.output[device]);
        for(auto &v:w.input[device])openmx_gpu_read_view_release(&v);
    }
    return 1;
}
int run(Transform &w,const char *label)
{
    openmx_gpu_workers_scope(&w.workers,size_t(w.cols));
    if(w.workers.count<2)return 0;
    return openmx_gpu_workers_run(&w.workers,worker,&w,label);
}
size_t matrix_bytes(int rows,int cols,int ld,size_t width)
{ return rows>0 && cols>0 ? (size_t(cols-1)*size_t(ld)+size_t(rows))*width : 0; }
}

extern "C" int openmx_gpu_transform(int op,int complex,int rows,int cols,const void *a,int lda,
    const void *b,int ldb,void *out,int ldout,int complex_weights,double floor)
{
    if(rows<1 || cols<1 || ldout<rows || !out || op<OPENMX_GPU_ZERO || op>OPENMX_GPU_INVSQRT)return 0;
    if(op!=OPENMX_GPU_ZERO && (!a || lda<(op==OPENMX_GPU_TRANSPOSE?cols:rows)))return 0;
    if(op==OPENMX_GPU_HERMITIAN && rows!=cols)return 0;
    Transform w{};w.op=op;w.complex=complex;w.rows=rows;w.cols=cols;w.lda=lda;w.ldb=ldb;w.ldout=ldout;
    w.complex_weights=complex_weights;w.floor=floor;w.target=out;
    const size_t width=complex?sizeof(Z):sizeof(double);
    if(op!=OPENMX_GPU_ZERO) {
        w.source[0]=a;
        w.bytes[0]=op==OPENMX_GPU_TRANSPOSE?matrix_bytes(cols,rows,lda,width):matrix_bytes(rows,cols,lda,width);
    }
    if(op==OPENMX_GPU_ADD || op==OPENMX_GPU_SCALE_ROWS || op==OPENMX_GPU_SCALE_COLUMNS) {
        if(!b || (op==OPENMX_GPU_ADD && ldb<rows))return 0;
        w.source[1]=b;
        w.bytes[1]=op==OPENMX_GPU_ADD?matrix_bytes(rows,cols,ldb,width):
            size_t(op==OPENMX_GPU_SCALE_ROWS?rows:cols)*(complex_weights?width:sizeof(double));
    }
    return run(w,"dense-transform");
}
extern "C" int openmx_gpu_spinor(int n,int real_parts,const void *r11,const void *r22,const void *r12,
    const double *i11,const double *i22,const double *i12,void *out)
{
    if(n<1 || n>std::numeric_limits<int>::max()/2 || !out)return 0;
    Transform w{};w.rows=w.cols=w.ldout=2*n;w.spinor=1;w.real_parts=real_parts;w.complex=1;w.target=out;
    const void *inputs[6]={r11,r22,r12,i11,i22,i12};
    for(int i=0;i<(real_parts?6:3);i++) {w.source[i]=inputs[i];w.bytes[i]=size_t(n)*n*(real_parts?sizeof(double):sizeof(Z));}
    return run(w,"dense-spinor-build");
}
extern "C" int openmx_gpu_scatter(int n,int entries,int format,const int *index,int values_count,
    const double *values,int phase_count,const double *phase_r,const double *phase_i,int imaginary,void *out)
{
    if(n<1 || entries<0 || values_count<0 || phase_count<0 || (format!=0 && format!=3 && format!=7))return 0;
    Transform w{};w.rows=w.cols=w.ldout=n;w.entries=entries;w.format=format;w.imaginary=imaginary;
    w.scatter=1;w.complex=format!=0;w.target=out;
    w.source[0]=index;w.bytes[0]=size_t(entries)*(format?format:1)*sizeof(int);
    w.source[1]=values;w.bytes[1]=size_t(values_count)*sizeof(double);
    if(format) {w.source[2]=phase_r;w.source[3]=phase_i;w.bytes[2]=w.bytes[3]=size_t(phase_count)*sizeof(double);}
    return run(w,"dense-sparse-scatter");
}

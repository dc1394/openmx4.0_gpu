#ifndef OPENMX_GPU_DM_ENTRY_H
#define OPENMX_GPU_DM_ENTRY_H
#include <stddef.h>
#include "openmx_gpu_dm.h"
typedef struct {double x,y;} OpenMXDMComplex;
#if defined(__HIPCC__)
__host__ __device__
#elif defined(_OPENMP)
#pragma omp declare target
#endif
static inline void openmx_gpu_dm_entry(int kind,int n,int states,int stride,int entries,
    const void *vectors,const int *indices,const double *weights,const double *phases,
    double *output,int phased,int edm,int pdm,int p)
{
        const int ia=indices[p],ib=indices[entries+p];
        const double co=phased?phases[p]:1,si=phased?phases[entries+p]:0;
        double r[10]={0};
        if(kind==OPENMX_GPU_DM_GATHER || kind==OPENMX_GPU_EDM_GATHER) {
            const OpenMXDMComplex *v=(const OpenMXDMComplex*)vectors;
            OpenMXDMComplex x=v[(size_t)(ia)+(size_t)(ib)*2*n],y=v[(size_t)(ia+n)+(size_t)(ib+n)*2*n],z=v[(size_t)(ia)+(size_t)(ib+n)*2*n];
            r[0]=x.x;r[1]=y.x;r[2]=z.x;r[3]=z.y;
            if(kind==OPENMX_GPU_DM_GATHER){r[4]=x.y;r[5]=y.y;}
        } else if(kind==OPENMX_GPU_DM_REAL) {
            const double *v=(const double*)vectors;
            for(int k=0;k<states;k++) {
                const double a=v[(size_t)(ia)*stride+k],b=v[(size_t)(ib)*stride+k];
                const double product=weights[k]*a*b;
                r[0]+=product;r[1]+=product*weights[states+k];
                if(pdm)r[2]+=weights[2*states+k]*a*b;
            }
        } else if(kind==OPENMX_GPU_DM_COL) {
            const OpenMXDMComplex *v=(const OpenMXDMComplex*)vectors;
            double d1=0,d2=0,d3=0,d4=0;
            for(int k=0;k<states;k++) {
                const OpenMXDMComplex a=v[(size_t)(ia)*stride+k],b=v[(size_t)(ib)*stride+k];
                const double w=weights[k],ar=a.x*w,ai=a.y*w,br=b.x*w,bi=b.y*w;
                const double re=ar*br+ai*bi,im=ar*bi-ai*br;
                d1+=re;d2+=im;d3+=re*weights[states+k];d4+=im*weights[states+k];
            }
            r[0]=co*d1-si*d2;r[1]=co*d3-si*d4;
        } else {
            const OpenMXDMComplex *v=(const OpenMXDMComplex*)vectors;
            double e11i=0,e22i=0;
            for(int k=0;k<states;k++) {
                const OpenMXDMComplex a=v[(size_t)(ia)*stride+k],b=v[(size_t)(ib)*stride+k],c=v[(size_t)(ia+n)*stride+k],d=v[(size_t)(ib+n)*stride+k];
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
        for(int c=0;c<10;c++)output[(size_t)(c)*entries+p]+=r[c];
}
#if defined(_OPENMP) && !defined(__HIPCC__)
#pragma omp end declare target
#endif
#endif

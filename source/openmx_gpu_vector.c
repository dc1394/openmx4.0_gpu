#include "openmx_gpu_backend.h"
#include "openmx_gpu_device.h"
#include "openmx_gpu_runtime.h"
#include "openmx_gpu_vector.h"
#include "openmx_gpu_vector_worker.h"
#include "openmx_gpu_workers.h"
#include "openmx_gpu_collective.h"


#if OPENMX_GPU_HIP
#include <omp.h>
#else
#include <openacc.h>
#endif
#include <stdint.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>

typedef struct { uint64_t n; int mode,columns,pairs; } VectorPacket;

static int vector_packet_size(size_t n,int mode,int columns,int pairs,size_t *bytes)
{
    if ((mode!=0 && mode!=1) || columns<1 || pairs<0) return 0;
    size_t indices=mode?0:(size_t)pairs*2*sizeof(int);
    size_t weights=mode?(size_t)columns:n;
    if (indices>SIZE_MAX-sizeof(VectorPacket) ||
        weights>(SIZE_MAX-sizeof(VectorPacket)-indices)/sizeof(double)) return 0;
    size_t available=(SIZE_MAX-sizeof(VectorPacket)-indices)/sizeof(double)-weights;
    if (n>available/(size_t)columns) return 0;
    *bytes=sizeof(VectorPacket)+((size_t)columns*n+weights)*sizeof(double)+indices;
    return 1;
}

static int vector_pack(void *opaque,size_t first,size_t last,void **packet,size_t *bytes)
{
    OpenMXGpuVector *j=opaque;const size_t n=last-first;
    if (last<first || last>j->n || !vector_packet_size(n,j->mode,j->columns,j->pairs,bytes)) return 0;
    VectorPacket *p=malloc(*bytes);if (!p) return 0;
    *packet=p;p->n=n;p->mode=j->mode;p->columns=j->columns;p->pairs=j->pairs;
    double *data=(double*)(p+1);
    for (int col=0;col<j->columns;col++) memcpy(data+(size_t)col*n,j->data+(size_t)col*j->ld+first,n*sizeof(double));
    data+=(size_t)j->columns*n;
    memcpy(data,j->mode?j->coeff:j->weight+first,(j->mode?(size_t)j->columns:n)*sizeof(double));
    if (!j->mode) {
        int *list=(int*)(data+n);
        memcpy(list,j->left,(size_t)j->pairs*sizeof(int));
        memcpy(list+j->pairs,j->right,(size_t)j->pairs*sizeof(int));
    }
    return 1;
}

static int vector_remote(void *context,const void *packet,size_t bytes,void **result,size_t *result_bytes)
{
    (void)context;
    if (bytes<sizeof(VectorPacket)) return 0;
    const VectorPacket *p=packet;
    size_t expected=0;
    if (p->n>SIZE_MAX || !vector_packet_size((size_t)p->n,p->mode,p->columns,p->pairs,&expected) || bytes!=expected) return 0;
    OpenMXGpuVector j={0};j.n=j.ld=(size_t)p->n;j.mode=p->mode;j.columns=p->columns;j.pairs=p->pairs;
    j.data=(const double*)(p+1);j.coeff=j.weight=j.data+(size_t)j.columns*j.n;
    if (!j.mode) {
        j.left=(const int*)(j.weight+j.n);j.right=j.left+j.pairs;
        for (int i=0;i<j.pairs;i++)
            if (j.left[i]<0 || j.left[i]>=j.columns || j.right[i]<0 || j.right[i]>=j.columns) return 0;
    }
    *result_bytes=(j.mode?j.n:(size_t)j.pairs)*sizeof(double);
    *result=calloc(*result_bytes?*result_bytes:1,1);if (!*result) return 0;
    j.output=*result;
    return openmx_gpu_vector_local(&j);
}

static void vector_commit(void *opaque,size_t first,size_t last,const void *result,size_t bytes)
{
    OpenMXGpuVector *j=opaque;(void)bytes;
    if (j->mode) memcpy(j->output+first,result,(last-first)*sizeof(double));
    else for (int p=0;p<j->pairs;p++) j->output[p]+=((const double*)result)[p];
}

int openmx_gpu_vector_execute(MPI_Comm comm,OpenMXGpuVector *j,int enabled,const char *label)
{
    const size_t outputs=j->mode?j->n:(size_t)j->pairs;
    memset(j->output,0,outputs*sizeof(double));
    int borrow=enabled && openmx_gpu_workers_capacity_enabled();
    int usable=enabled && (gpu_rank_device_usable() || borrow);
    int done=!j->n || (usable && openmx_gpu_vector_local(j));
    static const OpenMXGpuCollectiveOps ops={vector_pack,vector_remote,vector_commit};
    done=openmx_gpu_collective_retry(comm,&ops,j,j->n,done?1:borrow?0:-1,borrow && usable,label);
    if (!done) {
        if (j->mode) {
#pragma omp parallel for
            for (size_t i=0;i<j->n;i++) {
                double sum=0;
                for (int col=0;col<j->columns;col++) sum+=j->coeff[col]*j->data[(size_t)col*j->ld+i];
                j->output[i]=sum;
            }
        }
        else {
#pragma omp parallel for
            for (int p=0;p<j->pairs;p++) {
                double sum=0;
                const double *a=j->data+(size_t)j->left[p]*j->ld,*b=j->data+(size_t)j->right[p]*j->ld;
                for (size_t i=0;i<j->n;i++) sum+=j->weight[i]*a[i]*b[i];
                j->output[p]=sum;
            }
        }
    }
    return done;
}

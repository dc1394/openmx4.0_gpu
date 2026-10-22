#include "openmx_gpu_vector_packet.h"
#include "openmx_gpu_vector_worker.h"
#include "openmx_host_buffer.hpp"
#include <cstdlib>
#include <cstring>
#include <new>
#include <type_traits>

static_assert(std::is_trivial<OpenMXGpuVectorPacket>::value &&
              std::is_standard_layout<OpenMXGpuVectorPacket>::value,
              "Vector wire headers must retain their plain C layout");

static int vector_packet_size(size_t n,int mode,int columns,int pairs,size_t *bytes)
{
    if ((mode!=0 && mode!=1) || columns<1 || pairs<0) return 0;
    size_t indices=mode?0:(size_t)pairs*2*sizeof(int);
    size_t weights=mode?(size_t)columns:n;
    if (indices>SIZE_MAX-sizeof(OpenMXGpuVectorPacket) ||
        weights>(SIZE_MAX-sizeof(OpenMXGpuVectorPacket)-indices)/sizeof(double)) return 0;
    size_t available=(SIZE_MAX-sizeof(OpenMXGpuVectorPacket)-indices)/sizeof(double)-weights;
    if (n>available/(size_t)columns) return 0;
    *bytes=sizeof(OpenMXGpuVectorPacket)+((size_t)columns*n+weights)*sizeof(double)+indices;
    return 1;
}

extern "C" int openmx_gpu_vector_pack(void *opaque,size_t first,size_t last,void **packet,size_t *bytes)
{
    auto *j=static_cast<OpenMXGpuVector*>(opaque);const size_t n=last-first;
    if (last<first || last>j->n || !vector_packet_size(n,j->mode,j->columns,j->pairs,bytes)) return 0;
    openmx::host::Buffer<unsigned char> storage(static_cast<unsigned char*>(std::malloc(*bytes)));
    if (!storage) return 0;
    /* Canonicalize the header, including its native tail padding. Numeric
       payload sections retain the existing copies without initialization. */
    std::memset(storage.get(),0,sizeof(OpenMXGpuVectorPacket));
    auto *p=::new(storage.get()) OpenMXGpuVectorPacket;
    p->n=n;p->mode=j->mode;p->columns=j->columns;p->pairs=j->pairs;
    double *data=(double*)(p+1);
    for (int col=0;col<j->columns;col++) memcpy(data+(size_t)col*n,j->data+(size_t)col*j->ld+first,n*sizeof(double));
    data+=(size_t)j->columns*n;
    memcpy(data,j->mode?j->coeff:j->weight+first,(j->mode?(size_t)j->columns:n)*sizeof(double));
    if (!j->mode) {
        int *list=(int*)(data+n);
        memcpy(list,j->left,(size_t)j->pairs*sizeof(int));
        memcpy(list+j->pairs,j->right,(size_t)j->pairs*sizeof(int));
    }
    *packet=storage.release();
    return 1;
}

extern "C" int openmx_gpu_vector_remote(void *context,const void *packet,size_t bytes,void **result,size_t *result_bytes)
{
    (void)context;
    if (bytes<sizeof(OpenMXGpuVectorPacket)) return 0;
    const auto *p=static_cast<const OpenMXGpuVectorPacket*>(packet);
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
    openmx::host::Buffer<unsigned char> storage(static_cast<unsigned char*>(std::calloc(*result_bytes?*result_bytes:1,1)));
    if (!storage) {*result=nullptr;return 0;}
    j.output=reinterpret_cast<double*>(storage.get());
    const int completed=openmx_gpu_vector_local(&j);
    /* Ordinary failed results still belong to the collective on return. */
    *result=storage.release();
    return completed;
}

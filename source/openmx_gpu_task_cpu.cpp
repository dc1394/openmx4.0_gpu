#include "openmx_gpu_task_cpu.h"
#include "openmx_gpu_task_layout.h"
#include "openmx_host_buffer.hpp"
#include <cstdio>
#include <cstdlib>

namespace {
static size_t task_cpu_bytes(const OpenMXGpuTask *t,size_t lo,size_t hi)
{
    size_t nb=task_bytes(t,lo,hi,t->inputs+t->outputs,t->inputs+t->outputs+t->scratch);
    for (int i=0;i<t->inputs+t->outputs;i++)
        if (t->view[i].offsets || (i>=t->inputs && !t->view[i].item_bytes))
            nb=task_add(nb,task_view_bytes(t->view+i,lo,hi));
    return nb;
}
} // namespace

extern "C" void openmx_gpu_task_cpu(OpenMXGpuTask *t)
{
    const int output_end=t->inputs+t->outputs,end=output_end+t->scratch;
    /* Keep scratch and CSR packing bounded on the CPU too. A final CPU
       fallback must not duplicate an entire large device workspace. */
    for (size_t first=0;first<t->count;) {
        size_t n=t->count-first,nb;
        openmx::host::Buffer<unsigned char> arena;
        while (n>1 && task_cpu_bytes(t,first,first+n)>67108864) n=(n+1)/2;
        for (;;) {
            nb=task_cpu_bytes(t,first,first+n);
            if (nb!=SIZE_MAX) arena.reset(static_cast<unsigned char*>(malloc(nb?nb:1)));
            if (arena || n<=1) break;
            n=(n+1)/2;
        }
        if (!arena) {fprintf(stderr,"OpenMX: cannot allocate a single CPU range workspace.\n");abort();}
        void *p[OPENMX_GPU_TASK_VIEWS];size_t offset=0;
        for (int i=0;i<output_end;i++) {
            p[i]=t->view[i].data;
            if (t->view[i].offsets) {
                p[i]=arena.get()+offset;task_copy_view(p[i],t->view+i,first,first+n);
                offset=task_add(offset,task_view_bytes(t->view+i,first,first+n));
            } else if (t->view[i].item_bytes) p[i]=(unsigned char*)p[i]+first*t->view[i].item_bytes;
            else if (i>=t->inputs) {p[i]=arena.get()+offset;offset=task_add(offset,t->view[i].bytes);}
        }
        for (int i=output_end;i<end;i++) {p[i]=arena.get()+offset;offset=task_add(offset,task_view_bytes(t->view+i,first,first+n));}
        t->kernel(t->first+first,n,t->parameters,p,0);
        for (int i=t->inputs;i<output_end;i++) if (!t->view[i].item_bytes)
            for (size_t k=0;k<t->view[i].bytes/sizeof(double);k++) ((double*)t->view[i].data)[k]+=((double*)p[i])[k];
        first+=n;
    }
}

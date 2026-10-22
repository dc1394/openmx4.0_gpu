#include "openmx_gpu_task_packet.h"
#include "openmx_gpu_task_layout.h"
#include "openmx_host_buffer.hpp"
#include <cstdlib>
#include <cstring>
#include <new>
#include <type_traits>

static_assert(std::is_trivial<OpenMXGpuTaskPacket>::value &&
              std::is_standard_layout<OpenMXGpuTaskPacket>::value,
              "Task wire headers must retain their plain C layout");

extern "C" int openmx_gpu_task_pack(void *opaque,size_t lo,size_t hi,void **packet,size_t *bytes)
{
    const auto *t=static_cast<const OpenMXGpuTask*>(opaque);size_t n=hi-lo;
    size_t offset=task_add(0,sizeof(OpenMXGpuTaskPacket));
    offset=task_add(offset,t->parameter_bytes);
    *bytes=task_add(offset,task_bytes(t,lo,hi,0,t->inputs));
    if (*bytes==SIZE_MAX) return 0;
    openmx::host::Buffer<unsigned char> storage(static_cast<unsigned char*>(std::calloc(*bytes?*bytes:1,1)));
    if (!storage) return 0;
    auto *p=::new(storage.get()) OpenMXGpuTaskPacket;
    p->first=t->first+lo;p->count=n;p->parameter_bytes=t->parameter_bytes;
    p->inputs=t->inputs;p->outputs=t->outputs;p->scratch=t->scratch;p->scope_kind=t->scope_kind;
    for (int i=0;i<t->inputs+t->outputs+t->scratch;i++) {p->view[i].bytes=t->view[i].bytes;p->view[i].item_bytes=t->view[i].item_bytes;p->view[i].element_bytes=t->view[i].element_bytes;}
    if (t->parameter_bytes) memcpy((unsigned char*)p+task_add(0,sizeof(*p)),t->parameters,t->parameter_bytes);
    for (int i=0;i<t->inputs;i++) {
        size_t nb=task_view_bytes(t->view+i,lo,hi);
        if (nb) task_copy_view((unsigned char*)p+offset,t->view+i,lo,hi);
        offset=task_add(offset,nb);
    }
    *packet=storage.release();
    return 1;
}
extern "C" int openmx_gpu_task_remote(void *context,const void *packet,size_t bytes,void **result,size_t *result_bytes)
{
    const auto *original=static_cast<const OpenMXGpuTask*>(context);
    if (bytes<sizeof(OpenMXGpuTaskPacket)) return 0;
    const auto *p=static_cast<const OpenMXGpuTaskPacket*>(packet);
    if (p->inputs<0 || p->outputs<1 || p->scratch<0 || p->outputs>OPENMX_GPU_TASK_VIEWS ||
        p->inputs>OPENMX_GPU_TASK_VIEWS-p->outputs ||
        p->scratch>OPENMX_GPU_TASK_VIEWS-p->outputs-p->inputs || p->count>SIZE_MAX ||
        p->first>SIZE_MAX-p->count || p->parameter_bytes>SIZE_MAX) return 0;
    OpenMXGpuTask t={0};t.first=p->first;t.count=p->count;t.parameter_bytes=p->parameter_bytes;
    t.inputs=p->inputs;t.outputs=p->outputs;t.scratch=p->scratch;t.kernel=original->kernel;t.scope_kind=p->scope_kind;
    size_t offset=task_add(task_add(0,sizeof(*p)),t.parameter_bytes);
    if (offset>bytes) return 0;
    t.parameters=(const unsigned char*)p+task_add(0,sizeof(*p));
    for (int i=0;i<t.inputs+t.outputs+t.scratch;i++) {
        if (p->view[i].bytes>SIZE_MAX || p->view[i].item_bytes>SIZE_MAX || p->view[i].element_bytes>SIZE_MAX) return 0;
        t.view[i].bytes=p->view[i].bytes;t.view[i].item_bytes=p->view[i].item_bytes;
        if (p->view[i].element_bytes) {
            if (i>=t.inputs || t.count>SIZE_MAX/sizeof(size_t)-1) return 0;
            size_t index_bytes=task_add(0,(t.count+1)*sizeof(size_t));
            if (index_bytes>bytes-offset) return 0;
            t.view[i].offsets=(const size_t*)((const unsigned char*)packet+offset);
            t.view[i].element_bytes=p->view[i].element_bytes;
            if (t.view[i].offsets[0]) return 0;
            for (size_t k=0;k<t.count;k++) if (t.view[i].offsets[k+1]<t.view[i].offsets[k]) return 0;
        }
        if (i<t.inputs) {
            size_t next=task_add(offset,task_view_bytes(t.view+i,0,t.count));
            if (next>bytes) return 0;
            t.view[i].data=t.view[i].offsets?openmx_gpu_task_ragged_data((unsigned char*)packet+offset,t.count):(unsigned char*)packet+offset;
            offset=next;
        }
        else if (i<t.inputs+t.outputs && !t.view[i].item_bytes && t.view[i].bytes%sizeof(double)) return 0;
    }
    if (offset!=bytes) return 0;
    *result_bytes=task_bytes(&t,0,t.count,t.inputs,t.inputs+t.outputs);
    if (*result_bytes==SIZE_MAX) return 0;
    openmx::host::Buffer<unsigned char> storage(static_cast<unsigned char*>(std::calloc(*result_bytes?*result_bytes:1,1)));
    if (!storage) {*result=nullptr;return 0;}
    offset=0;
    for (int i=t.inputs;i<t.inputs+t.outputs;i++) {
        t.view[i].data=storage.get()+offset;
        offset=task_add(offset,task_view_bytes(t.view+i,0,t.count));
    }
    const int completed=openmx_gpu_task_local(&t,"remote-range");
    /* The collective also owns returned buffers after an ordinary failed
       run. Keep that contract, but retain ownership while the worker runs. */
    *result=storage.release();
    return completed;
}

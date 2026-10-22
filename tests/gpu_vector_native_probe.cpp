#include "gpu_vector_native_probe.h"
#include "openmx_gpu_vector_worker.cpp"
#include <new>

extern "C" void *openmx_test_vector_create(OpenMXGpuVector *job){
    auto *work=new(std::nothrow) VectorWork;
    if(work){work->job=*job;work->workers.count=1;work->workers.items=job->n;}
    return work;
}
extern "C" int openmx_test_vector_phase(void *opaque,int phase){return vector_worker(opaque,0,phase);}
extern "C" void *openmx_test_vector_result(void *opaque){return static_cast<VectorWork*>(opaque)->partial[0].get();}
extern "C" void *openmx_test_vector_arena(void *opaque){return static_cast<VectorWork*>(opaque)->arena[0].get();}
extern "C" void openmx_test_vector_destroy(void *opaque){delete static_cast<VectorWork*>(opaque);}

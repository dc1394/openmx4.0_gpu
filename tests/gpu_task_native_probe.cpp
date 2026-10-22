#include "gpu_task_native_probe.h"
#include "openmx_gpu_task_worker.cpp"
#include <new>

extern "C" void *openmx_test_task_create(OpenMXGpuTask *task){
    auto *work=new(std::nothrow) TaskWork;
    if(work){work->task=task;work->pool.count=1;work->pool.items=task->count;}
    return work;
}
extern "C" int openmx_test_task_phase(void *opaque,int phase){
    return task_worker(opaque,0,phase);
}
extern "C" void *openmx_test_task_result(void *opaque){
    return static_cast<TaskWork*>(opaque)->result[0].get();
}
extern "C" void openmx_test_task_destroy(void *opaque){delete static_cast<TaskWork*>(opaque);}

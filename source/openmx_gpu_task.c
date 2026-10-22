#include "openmx_gpu_backend.h"
#include "openmx_gpu_device.h"
#include "openmx_gpu_runtime.h"
#include "openmx_gpu_task.h"
#include "openmx_gpu_task_layout.h"
#include "openmx_gpu_task_cpu.h"
#include "openmx_gpu_task_packet.h"
#include "openmx_gpu_workers.h"
#include "openmx_gpu_collective.h"


#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>




int openmx_gpu_task_execute(MPI_Comm comm,OpenMXGpuTask *t,int enabled,const char *label)
{
    const int views=t->inputs+t->outputs;
    /* Reduction destinations are private to this operation; callers add
       existing accumulated values after successful/CPU completion. */
    for (int i=t->inputs;i<views;i++) if (!t->view[i].item_bytes)
        memset(t->view[i].data,0,t->view[i].bytes);
    int borrow=enabled && openmx_gpu_workers_capacity_enabled();
    int usable=enabled && (gpu_rank_device_usable() || borrow);
    int done=!t->count || (usable && openmx_gpu_task_local(t,label));
    if (comm!=MPI_COMM_NULL) {
        static const OpenMXGpuCollectiveOps ops={openmx_gpu_task_pack,openmx_gpu_task_remote,task_commit};
        done=openmx_gpu_collective_retry(comm,&ops,t,t->count,done?1:borrow?0:-1,borrow && usable,label);
    }
    if (!done) openmx_gpu_task_cpu(t);
    return done;
}

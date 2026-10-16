#ifndef OPENMX_GPU_TASK_H
#define OPENMX_GPU_TASK_H
#include <stddef.h>
#include <mpi.h>

#define OPENMX_GPU_TASK_VIEWS 40
/* A host-backed range kernel. Inputs are contiguous arrays: item_bytes>0
   slices by item, otherwise bytes is the size of a replicated lookup table.
   Outputs use the same convention; fixed outputs are double reductions.
   The kernel overwrites every output and must not access process globals.
   use_gpu=1 receives device pointers, use_gpu=0 receives host pointers.
   Every rank enters execute with the same kernel, even for empty work.
   MPI_COMM_NULL provides the node-local/CPU ladder at noncollective sites. */
typedef void (*OpenMXGpuTaskKernel)(size_t first,size_t count,const void *parameters,
    void *const views[],int use_gpu);
typedef struct {
    void *data;
    size_t bytes,item_bytes;
    /* Optional CSR input: offsets[count+1] counts elements of element_bytes.
       A kernel receives normalized offsets followed by aligned packed data.
       This avoids padding every atom to the largest neighbor/grid list. */
    const size_t *offsets;
    size_t element_bytes;
} OpenMXGpuTaskView;
static inline void *openmx_gpu_task_ragged_data(void *view,size_t count)
{
    return (unsigned char*)view+(((count+1)*sizeof(size_t)+15)&~(size_t)15);
}
typedef struct {
    size_t first,count,parameter_bytes;
    const void *parameters;
    int inputs,outputs,scratch;
    OpenMXGpuTaskView view[OPENMX_GPU_TASK_VIEWS];
    OpenMXGpuTaskKernel kernel;
    int scope_kind; /* 2: exact dense/k-point GPU group; otherwise node pool. */
} OpenMXGpuTask;

/* Returns 1 when GPU work completed, 0 after CPU completion. GPU capacity
   is always checked before allocation; unsuccessful GPU attempts leave
   the original outputs untouched. Outputs may not alias the inputs. */
int openmx_gpu_task_execute(MPI_Comm comm,OpenMXGpuTask *task,int enabled,const char *label);
#endif

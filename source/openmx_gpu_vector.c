#include "openmx_gpu_backend.h"
#include "openmx_gpu_device.h"
#include "openmx_gpu_runtime.h"
#include "openmx_gpu_vector.h"
#include "openmx_gpu_vector_worker.h"
#include "openmx_gpu_vector_packet.h"
#include "openmx_gpu_workers.h"
#include "openmx_gpu_collective.h"

#include <string.h>

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
    static const OpenMXGpuCollectiveOps ops={openmx_gpu_vector_pack,openmx_gpu_vector_remote,vector_commit};
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

#ifndef OPENMX_GPU_DM_COLLECTIVE_H
#define OPENMX_GPU_DM_COLLECTIVE_H
#include <mpi.h>
#include "openmx_gpu_dm.h"
/* All ranks in comm enter, including helpers with entries=0. The ordinary
   resident kernel is attempted first; recovery packets contain only the
   needed basis rows, never a complete eigenvector matrix. */
void openmx_gpu_dm_collective(MPI_Comm comm,int kind,int n,int states,int stride,int entries,
    const void *vectors,int host_vectors,const int *basis0,const int *basis1,
    const int *phase_index,int phase_count,const double *phase_r,const double *phase_i,
    const double *occupation,const double *energy_weight,const double *partial_weight,
    double *const outputs[10],int accumulate);
#endif

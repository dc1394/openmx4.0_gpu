#ifndef OPENMX_GPU_VECTOR_H
#define OPENMX_GPU_VECTOR_H
#include <mpi.h>
#include <stddef.h>
/* Capacity ladder for host-backed vector contractions. Every rank enters.
   mode=0: output[p]=sum_i weight[i]*data[left[p]][i]*data[right[p]][i].
   mode=1: output[i]=sum_j coeff[j]*data[j][i]. Column-major data uses ld.
   The local GPU pool precedes remote node shards, then CPU arithmetic. */
typedef struct {
    size_t n,ld;
    int mode,columns,pairs;
    const double *data,*weight,*coeff;
    const int *left,*right;
    double *output;
} OpenMXGpuVector;
int openmx_gpu_vector_execute(MPI_Comm comm,OpenMXGpuVector *job,int enabled,const char *label);
#endif

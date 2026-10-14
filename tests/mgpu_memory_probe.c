/* Query actual native cuSOLVER workspace sizes without allocating matrices.
   Link the production compatibility layer and run inside a GPU allocation. */
#include "openmx_cusolver_compat.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

int main(void)
{
    cusolverDnHandle_t handle;
    size_t free_bytes=0,total_bytes=0;
    if (cusolverDnCreate(&handle)!=CUSOLVER_STATUS_SUCCESS ||
        cudaMemGetInfo(&free_bytes,&total_bytes)!=cudaSuccess) return 1;
    printf("GPU_MEMORY free_bytes=%zu total_bytes=%zu\n",free_bytes,total_bytes);
    const int orders[]={6656,13000,13312,26000};
    for (int z=0;z<2;++z) for (int k=0;k<4;++k) {
        const int n=orders[k];
        cudaDataType type=z?CUDA_C_64F:CUDA_R_64F;
        const int requests[]={n/4,n<4800?n:4800,n};
        for (int sample=0;sample<3;++sample) {
            const int nev=requests[sample], full=nev==n;
            size_t device_bytes=0,host_bytes=0;
            int64_t found=0;
            double lo=0,hi=0;
            cusolverStatus_t status=cusolverDnXsyevdx_bufferSize(handle,NULL,CUSOLVER_EIG_MODE_VECTOR,
                full?CUSOLVER_EIG_RANGE_ALL:CUSOLVER_EIG_RANGE_I,CUBLAS_FILL_MODE_LOWER,
                n,type,NULL,n,&lo,&hi,1,nev,&found,CUDA_R_64F,NULL,type,&device_bytes,&host_bytes);
            printf("WORKSPACE type=%c n=%d nev=%d status=%d matrix_bytes=%zu device_bytes=%zu host_bytes=%zu\n",
                z?'Z':'D',n,nev,(int)status,(size_t)n*n*(z?16:8),device_bytes,host_bytes);
            if (status!=CUSOLVER_STATUS_SUCCESS) return 2;
        }
    }
    cusolverDnDestroy(handle);
    return 0;
}

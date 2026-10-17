/* Query actual native hipSOLVER workspace sizes without allocating matrices.
   Link the production compatibility layer and run inside a GPU allocation. */
#include "hipsolver_compat.h"
#include "hip_runtime_compat.h"
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

int main(void)
{
    hipsolverDnHandle_t handle;
    size_t free_bytes=0,total_bytes=0;
    if (hipsolverDnCreate(&handle)!=HIPSOLVER_STATUS_SUCCESS ||
        hipMemGetInfo(&free_bytes,&total_bytes)!=hipSuccess) return 1;
    printf("GPU_MEMORY free_bytes=%zu total_bytes=%zu\n",free_bytes,total_bytes);
    const int orders[]={6656,13000,13312,26000};
    for (int z=0;z<2;++z) for (int k=0;k<4;++k) {
        const int n=orders[k];
        hipDataType type=z?HIP_C_64F:HIP_R_64F;
        const int requests[]={n/4,n<4800?n:4800,n};
        for (int sample=0;sample<3;++sample) {
            const int nev=requests[sample], full=nev==n;
            size_t device_bytes=0,host_bytes=0;
            int64_t found=0;
            double lo=0,hi=0;
            hipsolverStatus_t status=hipsolverDnXsyevdx_bufferSize(handle,NULL,HIPSOLVER_EIG_MODE_VECTOR,
                full?HIPSOLVER_EIG_RANGE_ALL:HIPSOLVER_EIG_RANGE_I,HIPBLAS_FILL_MODE_LOWER,
                n,type,NULL,n,&lo,&hi,1,nev,&found,HIP_R_64F,NULL,type,&device_bytes,&host_bytes);
            printf("WORKSPACE type=%c n=%d nev=%d status=%d matrix_bytes=%zu device_bytes=%zu host_bytes=%zu\n",
                z?'Z':'D',n,nev,(int)status,(size_t)n*n*(z?16:8),device_bytes,host_bytes);
            if (status!=HIPSOLVER_STATUS_SUCCESS) return 2;
        }
    }
    hipsolverDnDestroy(handle);
    return 0;
}

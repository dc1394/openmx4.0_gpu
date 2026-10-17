#ifndef OPENMX_MGPU_DENSITY_H
#define OPENMX_MGPU_DENSITY_H
/* Included after openmx_common.h. Eigenvectors are state-major within each
   basis column (ldv=n). The host result is P=V^H diag(weight) V. The streamed
   GEMM keeps both the vector panel and the dense DM off the owner GPU. */
#include "openmx_mgpu_runtime.h"
#include <stdlib.h>
#include <stdio.h>
static int OpenMX_MgpuDensityHost(int n,int occupied,const double *weight,
                                const dcomplex *v,dcomplex *p)
{
    dcomplex *weighted=malloc((size_t)n*occupied*sizeof(dcomplex));
    if (!weighted) {
        fprintf(stderr,"<OpenMX_MGPU> Cannot allocate %.3f GiB host density panel\n",
                (double)n*occupied*sizeof(dcomplex)/(1024.0*1024.0*1024.0));
        return 0;
    }
#pragma omp parallel for
    for (int j=0;j<n;++j) for (int k=0;k<occupied;++k) {
        dcomplex x=v[(size_t)j*n+k];
        weighted[(size_t)j*occupied+k]=(dcomplex){weight[k]*x.r,weight[k]*x.i};
    }
    int ok=openmx_mgpu_host_zgemm(HIPBLAS_OP_C,HIPBLAS_OP_N,n,n,occupied,
                                 weighted,occupied,v,n,p,n);
    free(weighted);
    return ok;
}
#endif

/* Test-only interposer: make the second SCF step fail memory admission,
   without allocating or competing for GPU memory. Requires the CUDA 13.4
   automatic full-spectrum route (overlap + Hamiltonian = two SYEVD calls). */
#define _GNU_SOURCE
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <dlfcn.h>
#include <stdatomic.h>
#include <stdio.h>

static atomic_int full_solves;

cusolverStatus_t cusolverDnXsyevd(
    cusolverDnHandle_t h, cusolverDnParams_t p, cusolverEigMode_t job,
    cublasFillMode_t uplo, int64_t n, cudaDataType t, void *a, int64_t lda,
    cudaDataType wt, void *w, cudaDataType ct, void *dw, size_t db,
    void *hw, size_t hb, int *info)
{
    __typeof__(&cusolverDnXsyevd) real_call = dlsym(RTLD_NEXT,"cusolverDnXsyevd");
    if (!real_call) return CUSOLVER_STATUS_INTERNAL_ERROR;
    cusolverStatus_t status = real_call(h,p,job,uplo,n,t,a,lda,wt,w,ct,dw,db,hw,hb,info);
    if (status==CUSOLVER_STATUS_SUCCESS) atomic_fetch_add(&full_solves,1);
    return status;
}

cudaError_t cudaMemGetInfo(size_t *free_bytes, size_t *total_bytes)
{
    __typeof__(&cudaMemGetInfo) real_call = dlsym(RTLD_NEXT,"cudaMemGetInfo");
    static atomic_int announced;
    if (!real_call) return cudaErrorUnknown;
    cudaError_t status = real_call(free_bytes,total_bytes);
    if (status==cudaSuccess && atomic_load(&full_solves)>=2){
        if (*free_bytes>64u*1024u*1024u) *free_bytes=64u*1024u*1024u;
        if (!atomic_exchange(&announced,1))
            fprintf(stderr,"TEST_MEMORY_PRESSURE: reporting 64 MiB free after two full solves.\n");
    }
    return status;
}

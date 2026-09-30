#ifndef OPENMX_CUSOLVER_COMPAT_H
#define OPENMX_CUSOLVER_COMPAT_H

#include <cusolverDn.h>
#include <stdint.h>

/* Keep the public cuSOLVER signatures so existing workspace caches and error
   handling also cover the two-stage backend. Native calls are used internally. */
#ifdef __cplusplus
extern "C" {
#endif

void openmx_gpusolver_cache_release(void);

cusolverStatus_t openmx_cusolverDnCreate(cusolverDnHandle_t *handle);

cusolverStatus_t openmx_cusolverDnXsyevd_bufferSize(
    cusolverDnHandle_t handle,
    cusolverDnParams_t params,
    cusolverEigMode_t  jobz,
    cublasFillMode_t   uplo,
    int64_t            n,
    cudaDataType       dataTypeA,
    const void *       A,
    int64_t            lda,
    cudaDataType       dataTypeW,
    const void *       W,
    cudaDataType       computeType,
    size_t *           workspaceInBytesOnDevice,
    size_t *           workspaceInBytesOnHost);

cusolverStatus_t openmx_cusolverDnXsyevd(
    cusolverDnHandle_t handle,
    cusolverDnParams_t params,
    cusolverEigMode_t  jobz,
    cublasFillMode_t   uplo,
    int64_t            n,
    cudaDataType       dataTypeA,
    void *             A,
    int64_t            lda,
    cudaDataType       dataTypeW,
    void *             W,
    cudaDataType       computeType,
    void *             bufferOnDevice,
    size_t             workspaceInBytesOnDevice,
    void *             bufferOnHost,
    size_t             workspaceInBytesOnHost,
    int *              info);

cusolverStatus_t openmx_cusolverDnXsyevdx_bufferSize(
    cusolverDnHandle_t handle,
    cusolverDnParams_t params,
    cusolverEigMode_t  jobz,
    cusolverEigRange_t range,
    cublasFillMode_t   uplo,
    int64_t            n,
    cudaDataType       dataTypeA,
    const void *       A,
    int64_t            lda,
    void *             vl,
    void *             vu,
    int64_t            il,
    int64_t            iu,
    int64_t *          h_meig,
    cudaDataType       dataTypeW,
    const void *       W,
    cudaDataType       computeType,
    size_t *           workspaceInBytesOnDevice,
    size_t *           workspaceInBytesOnHost);

cusolverStatus_t openmx_cusolverDnXsyevdx(
    cusolverDnHandle_t handle,
    cusolverDnParams_t params,
    cusolverEigMode_t  jobz,
    cusolverEigRange_t range,
    cublasFillMode_t   uplo,
    int64_t            n,
    cudaDataType       dataTypeA,
    void *             A,
    int64_t            lda,
    void *             vl,
    void *             vu,
    int64_t            il,
    int64_t            iu,
    int64_t *          meig64,
    cudaDataType       dataTypeW,
    void *             W,
    cudaDataType       computeType,
    void *             bufferOnDevice,
    size_t             workspaceInBytesOnDevice,
    void *             bufferOnHost,
    size_t             workspaceInBytesOnHost,
    int *              info);

#ifdef __cplusplus
}
#endif

#ifndef OPENMX_CUSOLVER_IMPLEMENTATION
#define cusolverDnCreate openmx_cusolverDnCreate
#define cusolverDnXsyevd_bufferSize openmx_cusolverDnXsyevd_bufferSize
#define cusolverDnXsyevd openmx_cusolverDnXsyevd
#define cusolverDnXsyevdx_bufferSize openmx_cusolverDnXsyevdx_bufferSize
#define cusolverDnXsyevdx openmx_cusolverDnXsyevdx
#endif
#endif

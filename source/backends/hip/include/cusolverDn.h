#ifndef OPENMX_AMDGPU_CUSOLVERDN_COMPAT_H
#define OPENMX_AMDGPU_CUSOLVERDN_COMPAT_H

#include <stdint.h>
#include <stddef.h>
#include "cuda_runtime.h"
#include "cublas_v2.h"

#ifdef __cplusplus
#include <hipsolver/hipsolver.h>
typedef hipsolverDnHandle_t cusolverDnHandle_t;
typedef hipsolverStatus_t cusolverStatus_t;
typedef hipsolverEigMode_t cusolverEigMode_t;
typedef hipsolverEigRange_t cusolverEigRange_t;
#else
typedef struct hipsolverDnContext *cusolverDnHandle_t;
typedef int cusolverStatus_t;
typedef enum {
    HIPSOLVER_EIG_MODE_NOVECTOR = 201,
    HIPSOLVER_EIG_MODE_VECTOR = 202
} cusolverEigMode_t;
typedef enum {
    HIPSOLVER_EIG_RANGE_ALL = 221,
    HIPSOLVER_EIG_RANGE_V = 222,
    HIPSOLVER_EIG_RANGE_I = 223
} cusolverEigRange_t;
enum { HIPSOLVER_STATUS_SUCCESS = 0 };
cusolverStatus_t hipsolverDnCreate(cusolverDnHandle_t *handle);
cusolverStatus_t hipsolverDnDestroy(cusolverDnHandle_t handle);
cusolverStatus_t hipsolverDnSetStream(cusolverDnHandle_t handle, cudaStream_t streamId);
cusolverStatus_t hipsolverDnDsyevdx_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz,
    cusolverEigRange_t range, cublasFillMode_t uplo, int n, double *A, int lda, double vl, double vu,
    int il, int iu, int *nev, double *W, int *lwork);
cusolverStatus_t hipsolverDnDsyevdx(cusolverDnHandle_t handle, cusolverEigMode_t jobz,
    cusolverEigRange_t range, cublasFillMode_t uplo, int n, double *A, int lda, double vl, double vu,
    int il, int iu, int *nev, double *W, double *work, int lwork, int *devInfo);
cusolverStatus_t hipsolverDnDsyevd_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz,
    cublasFillMode_t uplo, int n, double *A, int lda, double *W, int *lwork);
cusolverStatus_t hipsolverDnDsyevd(cusolverDnHandle_t handle, cusolverEigMode_t jobz,
    cublasFillMode_t uplo, int n, double *A, int lda, double *W, double *work, int lwork,
    int *devInfo);
cusolverStatus_t hipsolverDnZheevdx_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz,
    cusolverEigRange_t range, cublasFillMode_t uplo, int n, cuDoubleComplex *A, int lda, double vl, double vu,
    int il, int iu, int *nev, double *W, int *lwork);
cusolverStatus_t hipsolverDnZheevdx(cusolverDnHandle_t handle, cusolverEigMode_t jobz,
    cusolverEigRange_t range, cublasFillMode_t uplo, int n, cuDoubleComplex *A, int lda, double vl, double vu,
    int il, int iu, int *nev, double *W, cuDoubleComplex *work, int lwork, int *devInfo);
#endif

#define CUSOLVER_STATUS_SUCCESS HIPSOLVER_STATUS_SUCCESS
#define CUSOLVER_STATUS_NOT_INITIALIZED 1
#define CUSOLVER_STATUS_ALLOC_FAILED 2
#define CUSOLVER_STATUS_INVALID_VALUE 3
#define CUSOLVER_STATUS_MAPPING_ERROR 4
#define CUSOLVER_STATUS_EXECUTION_FAILED 5
#define CUSOLVER_STATUS_INTERNAL_ERROR 6
#define CUSOLVER_STATUS_NOT_SUPPORTED 7
#define CUSOLVER_STATUS_ARCH_MISMATCH 8
#define CUSOLVER_STATUS_HANDLE_IS_NULLPTR 9
#define CUSOLVER_STATUS_INVALID_ENUM 10
#define CUSOLVER_STATUS_UNKNOWN 11
#define CUSOLVER_STATUS_ZERO_PIVOT 12
#define CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED 13
#define CUSOLVER_STATUS_INVALID_LICENSE CUSOLVER_STATUS_UNKNOWN
#define CUSOLVER_STATUS_INVALID_WORKSPACE CUSOLVER_STATUS_INVALID_VALUE
#define CUSOLVER_EIG_MODE_NOVECTOR HIPSOLVER_EIG_MODE_NOVECTOR
#define CUSOLVER_EIG_MODE_VECTOR HIPSOLVER_EIG_MODE_VECTOR
#define CUSOLVER_EIG_RANGE_ALL HIPSOLVER_EIG_RANGE_ALL
#define CUSOLVER_EIG_RANGE_I HIPSOLVER_EIG_RANGE_I
#define CUSOLVER_EIG_RANGE_V HIPSOLVER_EIG_RANGE_V

#define cusolverDnCreate hipsolverDnCreate
#define cusolverDnDestroy hipsolverDnDestroy
#define cusolverDnSetStream hipsolverDnSetStream

static inline cusolverStatus_t cusolverDnXsyevd_bufferSize(
    cusolverDnHandle_t handle, void *params, cusolverEigMode_t jobz,
    cublasFillMode_t uplo, int64_t n, cudaDataType dataTypeA, const void *A,
    int64_t lda, cudaDataType dataTypeW, const void *W,
    cudaDataType computeType, size_t *workspaceInBytesOnDevice,
    size_t *workspaceInBytesOnHost)
{
    (void)params;
    (void)dataTypeW;
    (void)computeType;

    int lwork = 0;
    cusolverStatus_t status;

    if (dataTypeA == CUDA_R_64F) {
        status = hipsolverDnDsyevd_bufferSize(
            handle, jobz, uplo, (int)n, (double *)A, (int)lda,
            (double *)W, &lwork);
        *workspaceInBytesOnDevice = (size_t)lwork * sizeof(double);
    } else {
        return CUSOLVER_STATUS_NOT_SUPPORTED;
    }

    if (workspaceInBytesOnHost != NULL) {
        *workspaceInBytesOnHost = 0;
    }
    return status;
}

static inline cusolverStatus_t cusolverDnXsyevd(
    cusolverDnHandle_t handle, void *params, cusolverEigMode_t jobz,
    cublasFillMode_t uplo, int64_t n, cudaDataType dataTypeA, void *A,
    int64_t lda, cudaDataType dataTypeW, void *W, cudaDataType computeType,
    void *d_work, size_t workspaceInBytesOnDevice, void *h_work,
    size_t workspaceInBytesOnHost, int *devInfo)
{
    (void)params;
    (void)dataTypeW;
    (void)computeType;
    (void)h_work;
    (void)workspaceInBytesOnHost;

    if (dataTypeA == CUDA_R_64F) {
        return hipsolverDnDsyevd(
            handle, jobz, uplo, (int)n, (double *)A, (int)lda,
            (double *)W, (double *)d_work,
            (int)(workspaceInBytesOnDevice / sizeof(double)), devInfo);
    }

    return CUSOLVER_STATUS_NOT_SUPPORTED;
}

static inline cusolverStatus_t cusolverDnXsyevdx_bufferSize(
    cusolverDnHandle_t handle, void *params, cusolverEigMode_t jobz,
    cusolverEigRange_t range, cublasFillMode_t uplo, int64_t n,
    cudaDataType dataTypeA, const void *A, int64_t lda, const double *vl,
    const double *vu, int64_t il, int64_t iu, int64_t *h_meig,
    cudaDataType dataTypeW, const void *W, cudaDataType computeType,
    size_t *workspaceInBytesOnDevice, size_t *workspaceInBytesOnHost)
{
    (void)params;
    (void)dataTypeW;
    (void)computeType;

    int lwork = 0;
    int nev = 0;
    cusolverStatus_t status;
    const double vl_value = (vl != NULL) ? *vl : 0.0;
    const double vu_value = (vu != NULL) ? *vu : 0.0;

    if (dataTypeA == CUDA_R_64F) {
        status = hipsolverDnDsyevdx_bufferSize(
            handle, jobz, range, uplo, (int)n, (double *)A, (int)lda,
            vl_value, vu_value, (int)il, (int)iu, &nev, (double *)W, &lwork);
        *workspaceInBytesOnDevice = (size_t)lwork * sizeof(double);
    } else {
        status = hipsolverDnZheevdx_bufferSize(
            handle, jobz, range, uplo, (int)n, (cuDoubleComplex *)A, (int)lda,
            vl_value, vu_value, (int)il, (int)iu, &nev, (double *)W, &lwork);
        *workspaceInBytesOnDevice = (size_t)lwork * sizeof(cuDoubleComplex);
    }

    if (workspaceInBytesOnHost != NULL) {
        *workspaceInBytesOnHost = 0;
    }
    if (h_meig != NULL) {
        *h_meig = (range == HIPSOLVER_EIG_RANGE_ALL) ? n : (iu - il + 1);
    }
    return status;
}

static inline cusolverStatus_t cusolverDnXsyevdx(
    cusolverDnHandle_t handle, void *params, cusolverEigMode_t jobz,
    cusolverEigRange_t range, cublasFillMode_t uplo, int64_t n,
    cudaDataType dataTypeA, void *A, int64_t lda, const double *vl,
    const double *vu, int64_t il, int64_t iu, int64_t *h_meig,
    cudaDataType dataTypeW, void *W, cudaDataType computeType, void *d_work,
    size_t workspaceInBytesOnDevice, void *h_work, size_t workspaceInBytesOnHost,
    int *devInfo)
{
    (void)params;
    (void)dataTypeW;
    (void)computeType;
    (void)h_work;
    (void)workspaceInBytesOnHost;

    int nev = 0;
    int lwork;
    cusolverStatus_t status;
    const double vl_value = (vl != NULL) ? *vl : 0.0;
    const double vu_value = (vu != NULL) ? *vu : 0.0;

    if (dataTypeA == CUDA_R_64F) {
        lwork = (int)(workspaceInBytesOnDevice / sizeof(double));
        status = hipsolverDnDsyevdx(
            handle, jobz, range, uplo, (int)n, (double *)A, (int)lda,
            vl_value, vu_value, (int)il, (int)iu, &nev, (double *)W,
            (double *)d_work, lwork, devInfo);
    } else {
        lwork = (int)(workspaceInBytesOnDevice / sizeof(cuDoubleComplex));
        status = hipsolverDnZheevdx(
            handle, jobz, range, uplo, (int)n, (cuDoubleComplex *)A, (int)lda,
            vl_value, vu_value, (int)il, (int)iu, &nev, (double *)W,
            (cuDoubleComplex *)d_work, lwork, devInfo);
    }

    if (h_meig != NULL) {
        *h_meig = (range == HIPSOLVER_EIG_RANGE_ALL) ? n : (iu - il + 1);
    }
    return status;
}

#endif

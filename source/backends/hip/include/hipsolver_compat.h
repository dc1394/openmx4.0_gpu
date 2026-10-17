#ifndef OPENMX_AMDGPU_HIPSOLVER_COMPAT_H
#define OPENMX_AMDGPU_HIPSOLVER_COMPAT_H

/*
 * hipSOLVER compatibility header for OpenMX's own sources.
 * C++ TUs use the real ROCm hipSOLVER header; plain C TUs get the small set of
 * hipSOLVER prototypes/enums OpenMX calls, plus the generic-API emulation
 * wrappers (hipsolverDnXsyevd / hipsolverDnXsyevdx) built on the typed hipSOLVER
 * entry points.  See hip_runtime_compat.h for why the CUDA-named shim headers
 * are kept in the tree for the third-party libraries.
 *
 * The wrappers only translate the generic API into the typed one (FP32/FP64,
 * real/complex).  The solver policy of the NVIDIA build's compatibility layer
 * (full spectra through the D&C driver, the phase similarity against the
 * rocSOLVER complex D&C defect, the retry of a failed solve) lives in
 * magma_openmx.cu and gpusolver_Syevdx.c.
 */

#include <limits.h>
#include <stdint.h>
#include <stddef.h>
#include "hip_runtime_compat.h"
#include "hipblas_compat.h"

#ifdef __cplusplus
#include <hipsolver/hipsolver.h>
#else
typedef struct hipsolverDnContext *hipsolverDnHandle_t;
typedef int hipsolverStatus_t;
typedef enum {
    HIPSOLVER_EIG_MODE_NOVECTOR = 201,
    HIPSOLVER_EIG_MODE_VECTOR = 202
} hipsolverEigMode_t;
typedef enum {
    HIPSOLVER_EIG_RANGE_ALL = 221,
    HIPSOLVER_EIG_RANGE_V = 222,
    HIPSOLVER_EIG_RANGE_I = 223
} hipsolverEigRange_t;
/* the values of ROCm 7.2's hipsolver/internal/hipsolver-types.h */
enum {
    HIPSOLVER_STATUS_SUCCESS = 0,
    HIPSOLVER_STATUS_NOT_INITIALIZED = 1,
    HIPSOLVER_STATUS_ALLOC_FAILED = 2,
    HIPSOLVER_STATUS_INVALID_VALUE = 3,
    HIPSOLVER_STATUS_MAPPING_ERROR = 4,
    HIPSOLVER_STATUS_EXECUTION_FAILED = 5,
    HIPSOLVER_STATUS_INTERNAL_ERROR = 6,
    HIPSOLVER_STATUS_NOT_SUPPORTED = 7,
    HIPSOLVER_STATUS_ARCH_MISMATCH = 8,
    HIPSOLVER_STATUS_HANDLE_IS_NULLPTR = 9,
    HIPSOLVER_STATUS_INVALID_ENUM = 10,
    HIPSOLVER_STATUS_UNKNOWN = 11
};
hipsolverStatus_t hipsolverDnCreate(hipsolverDnHandle_t *handle);
hipsolverStatus_t hipsolverDnDestroy(hipsolverDnHandle_t handle);
hipsolverStatus_t hipsolverDnSetStream(hipsolverDnHandle_t handle, hipStream_t streamId);
hipsolverStatus_t hipsolverDnGetStream(hipsolverDnHandle_t handle, hipStream_t *streamId);
hipsolverStatus_t hipsolverDnSsyevd_bufferSize(hipsolverDnHandle_t handle, hipsolverEigMode_t jobz,
    hipblasFillMode_t uplo, int n, const float *A, int lda, const float *W, int *lwork);
hipsolverStatus_t hipsolverDnSsyevd(hipsolverDnHandle_t handle, hipsolverEigMode_t jobz,
    hipblasFillMode_t uplo, int n, float *A, int lda, float *W, float *work, int lwork,
    int *devInfo);
hipsolverStatus_t hipsolverDnDsyevd_bufferSize(hipsolverDnHandle_t handle, hipsolverEigMode_t jobz,
    hipblasFillMode_t uplo, int n, const double *A, int lda, const double *W, int *lwork);
hipsolverStatus_t hipsolverDnDsyevd(hipsolverDnHandle_t handle, hipsolverEigMode_t jobz,
    hipblasFillMode_t uplo, int n, double *A, int lda, double *W, double *work, int lwork,
    int *devInfo);
hipsolverStatus_t hipsolverDnCheevd_bufferSize(hipsolverDnHandle_t handle, hipsolverEigMode_t jobz,
    hipblasFillMode_t uplo, int n, const hipFloatComplex *A, int lda, const float *W, int *lwork);
hipsolverStatus_t hipsolverDnCheevd(hipsolverDnHandle_t handle, hipsolverEigMode_t jobz,
    hipblasFillMode_t uplo, int n, hipFloatComplex *A, int lda, float *W, hipFloatComplex *work,
    int lwork, int *devInfo);
hipsolverStatus_t hipsolverDnZheevd_bufferSize(hipsolverDnHandle_t handle, hipsolverEigMode_t jobz,
    hipblasFillMode_t uplo, int n, const hipDoubleComplex *A, int lda, const double *W, int *lwork);
hipsolverStatus_t hipsolverDnZheevd(hipsolverDnHandle_t handle, hipsolverEigMode_t jobz,
    hipblasFillMode_t uplo, int n, hipDoubleComplex *A, int lda, double *W, hipDoubleComplex *work,
    int lwork, int *devInfo);
hipsolverStatus_t hipsolverDnSsyevdx_bufferSize(hipsolverDnHandle_t handle, hipsolverEigMode_t jobz,
    hipsolverEigRange_t range, hipblasFillMode_t uplo, int n, const float *A, int lda, float vl, float vu,
    int il, int iu, int *nev, const float *W, int *lwork);
hipsolverStatus_t hipsolverDnSsyevdx(hipsolverDnHandle_t handle, hipsolverEigMode_t jobz,
    hipsolverEigRange_t range, hipblasFillMode_t uplo, int n, float *A, int lda, float vl, float vu,
    int il, int iu, int *nev, float *W, float *work, int lwork, int *devInfo);
hipsolverStatus_t hipsolverDnDsyevdx_bufferSize(hipsolverDnHandle_t handle, hipsolverEigMode_t jobz,
    hipsolverEigRange_t range, hipblasFillMode_t uplo, int n, const double *A, int lda, double vl, double vu,
    int il, int iu, int *nev, const double *W, int *lwork);
hipsolverStatus_t hipsolverDnDsyevdx(hipsolverDnHandle_t handle, hipsolverEigMode_t jobz,
    hipsolverEigRange_t range, hipblasFillMode_t uplo, int n, double *A, int lda, double vl, double vu,
    int il, int iu, int *nev, double *W, double *work, int lwork, int *devInfo);
hipsolverStatus_t hipsolverDnCheevdx_bufferSize(hipsolverDnHandle_t handle, hipsolverEigMode_t jobz,
    hipsolverEigRange_t range, hipblasFillMode_t uplo, int n, const hipFloatComplex *A, int lda,
    float vl, float vu, int il, int iu, int *nev, const float *W, int *lwork);
hipsolverStatus_t hipsolverDnCheevdx(hipsolverDnHandle_t handle, hipsolverEigMode_t jobz,
    hipsolverEigRange_t range, hipblasFillMode_t uplo, int n, hipFloatComplex *A, int lda,
    float vl, float vu, int il, int iu, int *nev, float *W, hipFloatComplex *work, int lwork,
    int *devInfo);
hipsolverStatus_t hipsolverDnZheevdx_bufferSize(hipsolverDnHandle_t handle, hipsolverEigMode_t jobz,
    hipsolverEigRange_t range, hipblasFillMode_t uplo, int n, const hipDoubleComplex *A, int lda,
    double vl, double vu, int il, int iu, int *nev, const double *W, int *lwork);
hipsolverStatus_t hipsolverDnZheevdx(hipsolverDnHandle_t handle, hipsolverEigMode_t jobz,
    hipsolverEigRange_t range, hipblasFillMode_t uplo, int n, hipDoubleComplex *A, int lda,
    double vl, double vu, int il, int iu, int *nev, double *W, hipDoubleComplex *work, int lwork,
    int *devInfo);
#endif

/*
 * Generic-API emulation wrappers (equivalent to cusolverDnXsyevd/Xsyevdx on the
 * NVIDIA build) implemented on top of the typed hipSOLVER entry points.
 * dataTypeA selects the routine (HIP_R_32F: Ssy, HIP_R_64F: Dsy, HIP_C_32F: Che,
 * HIP_C_64F: Zhe); dataTypeW must be the matching real type and computeType
 * equal dataTypeA, anything else is HIPSOLVER_STATUS_NOT_SUPPORTED.  vl and
 * vu point to values of dataTypeW, as in the generic API.  The typed API takes
 * int sizes and an element count of workspace, so a dimension or a workspace
 * that does not fit an int is HIPSOLVER_STATUS_INVALID_VALUE.  There is no
 * host workspace.
 */

/* element size of dataTypeA for the supported combinations, else 0 */
static inline size_t openmx_hipsolver_x_width(hipDataType dataTypeA, hipDataType dataTypeW,
                                              hipDataType computeType)
{
    if (computeType != dataTypeA) return 0;
    switch (dataTypeA) {
    case HIP_R_32F: return (dataTypeW == HIP_R_32F) ? sizeof(float) : 0;
    case HIP_R_64F: return (dataTypeW == HIP_R_64F) ? sizeof(double) : 0;
    case HIP_C_32F: return (dataTypeW == HIP_R_32F) ? sizeof(hipFloatComplex) : 0;
    case HIP_C_64F: return (dataTypeW == HIP_R_64F) ? sizeof(hipDoubleComplex) : 0;
    default:        return 0;
    }
}

/* *bound as dataTypeW, 0 when bound is NULL */
static inline double openmx_hipsolver_x_bound(const void *bound, hipDataType dataTypeW)
{
    if (bound == NULL) return 0.0;
    return (dataTypeW == HIP_R_32F) ? (double)*(const float *)bound : *(const double *)bound;
}

static inline int openmx_hipsolver_x_fits_int(int64_t value)
{
    return 0 <= value && value <= (int64_t)INT_MAX;
}

/* workspace bytes -> element count, or -1 when it does not fit an int */
static inline int openmx_hipsolver_x_lwork(size_t bytes, size_t width)
{
    size_t count = bytes / width;

    return (count <= (size_t)INT_MAX) ? (int)count : -1;
}

/* element count -> workspace bytes, 0 when the query failed; a negative
   count (an int overflow inside the library) is HIPSOLVER_STATUS_INVALID_VALUE */
static inline hipsolverStatus_t openmx_hipsolver_x_bytes(hipsolverStatus_t status, int lwork,
                                                         size_t width, size_t *bytes)
{
    if (bytes == NULL) return HIPSOLVER_STATUS_INVALID_VALUE;
    *bytes = 0;
    if (status != HIPSOLVER_STATUS_SUCCESS) return status;
    if (lwork < 0) return HIPSOLVER_STATUS_INVALID_VALUE;
    *bytes = (size_t)lwork * width;
    return status;
}

static inline hipsolverStatus_t hipsolverDnXsyevd_bufferSize(
    hipsolverDnHandle_t handle, void *params, hipsolverEigMode_t jobz,
    hipblasFillMode_t uplo, int64_t n, hipDataType dataTypeA, const void *A,
    int64_t lda, hipDataType dataTypeW, const void *W,
    hipDataType computeType, size_t *workspaceInBytesOnDevice,
    size_t *workspaceInBytesOnHost)
{
    const size_t width = openmx_hipsolver_x_width(dataTypeA, dataTypeW, computeType);
    int lwork = 0;
    hipsolverStatus_t status;

    (void)params;

    if (workspaceInBytesOnHost != NULL) {
        *workspaceInBytesOnHost = 0;
    }
    if (workspaceInBytesOnDevice != NULL) {
        *workspaceInBytesOnDevice = 0;
    }
    if (width == 0) {
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
    if (!openmx_hipsolver_x_fits_int(n) || !openmx_hipsolver_x_fits_int(lda)) {
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }

    switch (dataTypeA) {
    case HIP_R_32F:
        status = hipsolverDnSsyevd_bufferSize(handle, jobz, uplo, (int)n, (const float *)A, (int)lda,
                                              (const float *)W, &lwork);
        break;
    case HIP_R_64F:
        status = hipsolverDnDsyevd_bufferSize(handle, jobz, uplo, (int)n, (const double *)A, (int)lda,
                                              (const double *)W, &lwork);
        break;
    case HIP_C_32F:
        status = hipsolverDnCheevd_bufferSize(handle, jobz, uplo, (int)n, (const hipFloatComplex *)A,
                                              (int)lda, (const float *)W, &lwork);
        break;
    default:
        status = hipsolverDnZheevd_bufferSize(handle, jobz, uplo, (int)n, (const hipDoubleComplex *)A,
                                              (int)lda, (const double *)W, &lwork);
        break;
    }
    return openmx_hipsolver_x_bytes(status, lwork, width, workspaceInBytesOnDevice);
}

static inline hipsolverStatus_t hipsolverDnXsyevd(
    hipsolverDnHandle_t handle, void *params, hipsolverEigMode_t jobz,
    hipblasFillMode_t uplo, int64_t n, hipDataType dataTypeA, void *A,
    int64_t lda, hipDataType dataTypeW, void *W, hipDataType computeType,
    void *d_work, size_t workspaceInBytesOnDevice, void *h_work,
    size_t workspaceInBytesOnHost, int *devInfo)
{
    const size_t width = openmx_hipsolver_x_width(dataTypeA, dataTypeW, computeType);
    int lwork;

    (void)params;
    (void)h_work;
    (void)workspaceInBytesOnHost;

    if (width == 0) {
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
    lwork = openmx_hipsolver_x_lwork(workspaceInBytesOnDevice, width);
    if (!openmx_hipsolver_x_fits_int(n) || !openmx_hipsolver_x_fits_int(lda) || lwork < 0) {
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }

    switch (dataTypeA) {
    case HIP_R_32F:
        return hipsolverDnSsyevd(handle, jobz, uplo, (int)n, (float *)A, (int)lda, (float *)W,
                                 (float *)d_work, lwork, devInfo);
    case HIP_R_64F:
        return hipsolverDnDsyevd(handle, jobz, uplo, (int)n, (double *)A, (int)lda, (double *)W,
                                 (double *)d_work, lwork, devInfo);
    case HIP_C_32F:
        return hipsolverDnCheevd(handle, jobz, uplo, (int)n, (hipFloatComplex *)A, (int)lda, (float *)W,
                                 (hipFloatComplex *)d_work, lwork, devInfo);
    default:
        return hipsolverDnZheevd(handle, jobz, uplo, (int)n, (hipDoubleComplex *)A, (int)lda, (double *)W,
                                 (hipDoubleComplex *)d_work, lwork, devInfo);
    }
}

/* h_meig of the query: the count the request implies (n for RANGE_ALL and
   for RANGE_V, which has no count before the solve; iu - il + 1 for RANGE_I) */
static inline hipsolverStatus_t hipsolverDnXsyevdx_bufferSize(
    hipsolverDnHandle_t handle, void *params, hipsolverEigMode_t jobz,
    hipsolverEigRange_t range, hipblasFillMode_t uplo, int64_t n,
    hipDataType dataTypeA, const void *A, int64_t lda, const void *vl,
    const void *vu, int64_t il, int64_t iu, int64_t *h_meig,
    hipDataType dataTypeW, const void *W, hipDataType computeType,
    size_t *workspaceInBytesOnDevice, size_t *workspaceInBytesOnHost)
{
    const size_t width = openmx_hipsolver_x_width(dataTypeA, dataTypeW, computeType);
    int lwork = 0;
    int nev = 0;
    hipsolverStatus_t status;
    double vl_value, vu_value;

    (void)params;

    if (workspaceInBytesOnHost != NULL) {
        *workspaceInBytesOnHost = 0;
    }
    if (workspaceInBytesOnDevice != NULL) {
        *workspaceInBytesOnDevice = 0;
    }
    if (h_meig != NULL) {
        *h_meig = 0;
    }
    if (width == 0) {
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
    if (!openmx_hipsolver_x_fits_int(n) || !openmx_hipsolver_x_fits_int(lda) ||
        !openmx_hipsolver_x_fits_int(il) || !openmx_hipsolver_x_fits_int(iu)) {
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }
    /* read only once dataTypeW is known to be a supported real type */
    vl_value = openmx_hipsolver_x_bound(vl, dataTypeW);
    vu_value = openmx_hipsolver_x_bound(vu, dataTypeW);

    switch (dataTypeA) {
    case HIP_R_32F:
        status = hipsolverDnSsyevdx_bufferSize(handle, jobz, range, uplo, (int)n, (const float *)A, (int)lda,
                                               (float)vl_value, (float)vu_value, (int)il, (int)iu, &nev,
                                               (const float *)W, &lwork);
        break;
    case HIP_R_64F:
        status = hipsolverDnDsyevdx_bufferSize(handle, jobz, range, uplo, (int)n, (const double *)A, (int)lda,
                                               vl_value, vu_value, (int)il, (int)iu, &nev,
                                               (const double *)W, &lwork);
        break;
    case HIP_C_32F:
        status = hipsolverDnCheevdx_bufferSize(handle, jobz, range, uplo, (int)n, (const hipFloatComplex *)A,
                                               (int)lda, (float)vl_value, (float)vu_value, (int)il, (int)iu,
                                               &nev, (const float *)W, &lwork);
        break;
    default:
        status = hipsolverDnZheevdx_bufferSize(handle, jobz, range, uplo, (int)n, (const hipDoubleComplex *)A,
                                               (int)lda, vl_value, vu_value, (int)il, (int)iu, &nev,
                                               (const double *)W, &lwork);
        break;
    }

    status = openmx_hipsolver_x_bytes(status, lwork, width, workspaceInBytesOnDevice);
    if (h_meig != NULL && status == HIPSOLVER_STATUS_SUCCESS) {
        *h_meig = (range == HIPSOLVER_EIG_RANGE_I) ? (iu - il + 1) : n;
    }
    return status;
}

/* h_meig is the count the solver reports (hipSOLVER returns it to the host
   before the call returns) */
static inline hipsolverStatus_t hipsolverDnXsyevdx(
    hipsolverDnHandle_t handle, void *params, hipsolverEigMode_t jobz,
    hipsolverEigRange_t range, hipblasFillMode_t uplo, int64_t n,
    hipDataType dataTypeA, void *A, int64_t lda, const void *vl,
    const void *vu, int64_t il, int64_t iu, int64_t *h_meig,
    hipDataType dataTypeW, void *W, hipDataType computeType, void *d_work,
    size_t workspaceInBytesOnDevice, void *h_work, size_t workspaceInBytesOnHost,
    int *devInfo)
{
    const size_t width = openmx_hipsolver_x_width(dataTypeA, dataTypeW, computeType);
    int nev = 0;
    int lwork;
    hipsolverStatus_t status;
    double vl_value, vu_value;

    (void)params;
    (void)h_work;
    (void)workspaceInBytesOnHost;

    if (h_meig != NULL) {
        *h_meig = 0;
    }
    if (width == 0) {
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
    lwork = openmx_hipsolver_x_lwork(workspaceInBytesOnDevice, width);
    if (!openmx_hipsolver_x_fits_int(n) || !openmx_hipsolver_x_fits_int(lda) ||
        !openmx_hipsolver_x_fits_int(il) || !openmx_hipsolver_x_fits_int(iu) || lwork < 0) {
        return HIPSOLVER_STATUS_INVALID_VALUE;
    }
    vl_value = openmx_hipsolver_x_bound(vl, dataTypeW);
    vu_value = openmx_hipsolver_x_bound(vu, dataTypeW);

    switch (dataTypeA) {
    case HIP_R_32F:
        status = hipsolverDnSsyevdx(handle, jobz, range, uplo, (int)n, (float *)A, (int)lda,
                                    (float)vl_value, (float)vu_value, (int)il, (int)iu, &nev, (float *)W,
                                    (float *)d_work, lwork, devInfo);
        break;
    case HIP_R_64F:
        status = hipsolverDnDsyevdx(handle, jobz, range, uplo, (int)n, (double *)A, (int)lda,
                                    vl_value, vu_value, (int)il, (int)iu, &nev, (double *)W,
                                    (double *)d_work, lwork, devInfo);
        break;
    case HIP_C_32F:
        status = hipsolverDnCheevdx(handle, jobz, range, uplo, (int)n, (hipFloatComplex *)A, (int)lda,
                                    (float)vl_value, (float)vu_value, (int)il, (int)iu, &nev, (float *)W,
                                    (hipFloatComplex *)d_work, lwork, devInfo);
        break;
    default:
        status = hipsolverDnZheevdx(handle, jobz, range, uplo, (int)n, (hipDoubleComplex *)A, (int)lda,
                                    vl_value, vu_value, (int)il, (int)iu, &nev, (double *)W,
                                    (hipDoubleComplex *)d_work, lwork, devInfo);
        break;
    }

    if (h_meig != NULL && status == HIPSOLVER_STATUS_SUCCESS) {
        *h_meig = nev;
    }
    return status;
}

#endif

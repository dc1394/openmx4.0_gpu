#include "openmx_common.h"
#include "hip_runtime_compat.h"
#include <omp.h>
#include <stdlib.h>
#include <stdio.h>

extern void F77_NAME(zgemm, ZGEMM)(char *TRANSA, char *TRANSB, INTEGER *M, INTEGER *N, INTEGER *K, dcomplex *ALPHA,
                                   dcomplex *A, INTEGER *LDA, dcomplex *B, INTEGER *LDB, dcomplex *BETA,
                                   dcomplex *C, INTEGER *LDC);

static const char *OpenMX_ZgemmHipblasStatusName(hipblasStatus_t status)
{
    switch (status) {
    case HIPBLAS_STATUS_SUCCESS:
        return "HIPBLAS_STATUS_SUCCESS";
    case HIPBLAS_STATUS_NOT_INITIALIZED:
        return "HIPBLAS_STATUS_NOT_INITIALIZED";
    case HIPBLAS_STATUS_ALLOC_FAILED:
        return "HIPBLAS_STATUS_ALLOC_FAILED";
    case HIPBLAS_STATUS_INVALID_VALUE:
        return "HIPBLAS_STATUS_INVALID_VALUE";
    case HIPBLAS_STATUS_ARCH_MISMATCH:
        return "HIPBLAS_STATUS_ARCH_MISMATCH";
    case HIPBLAS_STATUS_MAPPING_ERROR:
        return "HIPBLAS_STATUS_MAPPING_ERROR";
    case HIPBLAS_STATUS_EXECUTION_FAILED:
        return "HIPBLAS_STATUS_EXECUTION_FAILED";
    case HIPBLAS_STATUS_INTERNAL_ERROR:
        return "HIPBLAS_STATUS_INTERNAL_ERROR";
    case HIPBLAS_STATUS_NOT_SUPPORTED:
        return "HIPBLAS_STATUS_NOT_SUPPORTED";
    default:
        return "HIPBLAS_STATUS_UNKNOWN";
    }
}

static int OpenMX_ZgemmRank(void)
{
    int initialized = 0;
    int rank = -1;

    MPI_Initialized(&initialized);
    if (initialized) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    }

    return rank;
}

static int OpenMX_ZgemmVerbose(void)
{
    const char *value = getenv("OPENMX_GPU_VERBOSE");

    if (value == NULL || value[0] == '\0') {
        value = getenv("OPENMX_GEMM_VERBOSE");
    }

    return (value != NULL && value[0] == '1');
}

static char OpenMX_ZgemmOpChar(hipblasOperation_t op)
{
    if (op == HIPBLAS_OP_C) {
        return 'C';
    }
    return (op == HIPBLAS_OP_N) ? 'N' : 'T';
}

static void OpenMX_ZgemmCpu(hipblasOperation_t transa, hipblasOperation_t transb, int m, int n, int k,
                            dcomplex const *A, dcomplex const *B, dcomplex *C)
{
    char transa_cpu = OpenMX_ZgemmOpChar(transa);
    char transb_cpu = OpenMX_ZgemmOpChar(transb);
    INTEGER mi = m;
    INTEGER ni = n;
    INTEGER ki = k;
    INTEGER lda = m;
    INTEGER ldb = k;
    INTEGER ldc = m;
    dcomplex alpha = {1.0, 0.0};
    dcomplex beta = {0.0, 0.0};

    F77_NAME(zgemm, ZGEMM)(&transa_cpu, &transb_cpu, &mi, &ni, &ki, &alpha, (dcomplex *)A, &lda, (dcomplex *)B,
                           &ldb, &beta, C, &ldc);
}

static void OpenMX_ZgemmLogGemmul8Retry(const char *where, hipblasStatus_t status, int m, int n, int k)
{
    fprintf(stderr,
            "<GEMM> rank %d: GEMMul8 failed in %s for ZGEMM(m=%d,n=%d,k=%d): %s (%d). "
            "Retrying with native hipBLAS.\n",
            OpenMX_ZgemmRank(), where, m, n, k, OpenMX_ZgemmHipblasStatusName(status), (int)status);
    fflush(stderr);
}

static void OpenMX_ZgemmLogCpuFallback(const char *where, const char *backend, hipblasStatus_t status, int m, int n,
                                       int k)
{
    fprintf(stderr,
            "<GEMM> rank %d: %s failed in %s for ZGEMM(m=%d,n=%d,k=%d): %s (%d). "
            "Falling back to CPU BLAS.\n",
            OpenMX_ZgemmRank(), backend, where, m, n, k, OpenMX_ZgemmHipblasStatusName(status), (int)status);
    fflush(stderr);
}

static void OpenMX_ZgemmLogHipCpuFallback(const char *where, hipError_t status, int m, int n, int k)
{
    fprintf(stderr,
            "<GEMM> rank %d: GPU setup/transfer failed in %s for ZGEMM(m=%d,n=%d,k=%d): %s (%d). "
            "Falling back to CPU BLAS.\n",
            OpenMX_ZgemmRank(), where, m, n, k, hipGetErrorString(status), (int)status);
    fflush(stderr);
}

static void OpenMX_ZgemmLogBackendOnce(const char *backend, int m, int n, int k)
{
    static int logged_gemmul8 = 0;
    static int logged_native = 0;
    int rank = OpenMX_ZgemmRank();
    int *logged = (backend[0] == 'G') ? &logged_gemmul8 : &logged_native;

    if (!OpenMX_ZgemmVerbose()) return;

    if (rank != 0 || *logged) return;

    fprintf(stderr,
            "<GEMM> rank %d: using %s for ZGEMM(m=%d,n=%d,k=%d). "
            "If it fails, OpenMX retries native hipBLAS and then CPU BLAS.\n",
            rank, backend, m, n, k);
    fflush(stderr);
    *logged = 1;
}

static hipblasStatus_t OpenMX_ZgemmTryGpu(hipblasHandle_t handle, hipblasOperation_t transa, hipblasOperation_t transb,
                                         int m, int n, int k, dcomplex const *A, dcomplex const *B, dcomplex *C,
                                         const char *where)
{
    hipDoubleComplex const alpha = make_hipDoubleComplex(1.0, 0.0);
    hipDoubleComplex const beta = make_hipDoubleComplex(0.0, 0.0);
    hipblasStatus_t status;
    long long const native_before = openmx_gemmul8NativeCalls(1);

    status = openmx_gemmul8Zgemm(handle, transa, transb, m, n, k, &alpha, (hipDoubleComplex const *)A, m,
                                 (hipDoubleComplex const *)B, k, &beta, (hipDoubleComplex *)C, m);
    if (status == HIPBLAS_STATUS_SUCCESS) {
        OpenMX_ZgemmLogBackendOnce(openmx_gemmul8NativeCalls(1) == native_before ? "GEMMul8" : "native hipBLAS",
                                   m, n, k);
        return status;
    }

    OpenMX_ZgemmLogGemmul8Retry(where, status, m, n, k);
    status = hipblasZgemm(handle, transa, transb, m, n, k, &alpha, (hipDoubleComplex const *)A, m,
                         (hipDoubleComplex const *)B, k, &beta, (hipDoubleComplex *)C, m);
    if (status == HIPBLAS_STATUS_SUCCESS) {
        OpenMX_ZgemmLogBackendOnce("native hipBLAS", m, n, k);
    }
    return status;
}

void my_hipblasZgemm(hipblasOperation_t transa, hipblasOperation_t transb, int m, int n, int k, dcomplex const * A,
    dcomplex const * B, dcomplex * C)
{
    hipblasHandle_t handle = NULL;
    hipblasStatus_t status;
    hipError_t hip_status;
    hipDoubleComplex *d_A = NULL;
    hipDoubleComplex *d_B = NULL;
    hipDoubleComplex *d_C = NULL;

    status = hipblasCreate(&handle);
    if (status != HIPBLAS_STATUS_SUCCESS) {
        OpenMX_ZgemmLogCpuFallback("my_hipblasZgemm:hipblasCreate", "native hipBLAS", status, m, n, k);
        OpenMX_ZgemmCpu(transa, transb, m, n, k, A, B, C);
        return;
    }

    hip_status = hipMalloc((void **)&d_A, m * k * sizeof(hipDoubleComplex));
    if (hip_status != hipSuccess) {
        OpenMX_ZgemmLogHipCpuFallback("my_hipblasZgemm:hipMalloc(A)", hip_status, m, n, k);
        goto cpu_fallback;
    }
    hip_status = hipMalloc((void **)&d_B, n * k * sizeof(hipDoubleComplex));
    if (hip_status != hipSuccess) {
        OpenMX_ZgemmLogHipCpuFallback("my_hipblasZgemm:hipMalloc(B)", hip_status, m, n, k);
        goto cpu_fallback;
    }
    hip_status = hipMalloc((void **)&d_C, m * n * sizeof(hipDoubleComplex));
    if (hip_status != hipSuccess) {
        OpenMX_ZgemmLogHipCpuFallback("my_hipblasZgemm:hipMalloc(C)", hip_status, m, n, k);
        goto cpu_fallback;
    }

    hip_status = hipMemcpy(d_A, A, m * k * sizeof(hipDoubleComplex), hipMemcpyHostToDevice);
    if (hip_status != hipSuccess) {
        OpenMX_ZgemmLogHipCpuFallback("my_hipblasZgemm:hipMemcpy(A)", hip_status, m, n, k);
        goto cpu_fallback;
    }
    hip_status = hipMemcpy(d_B, B, n * k * sizeof(hipDoubleComplex), hipMemcpyHostToDevice);
    if (hip_status != hipSuccess) {
        OpenMX_ZgemmLogHipCpuFallback("my_hipblasZgemm:hipMemcpy(B)", hip_status, m, n, k);
        goto cpu_fallback;
    }

    status = OpenMX_ZgemmTryGpu(handle, transa, transb, m, n, k, (dcomplex const *)d_A, (dcomplex const *)d_B,
                                (dcomplex *)d_C, "my_hipblasZgemm");
    if (status != HIPBLAS_STATUS_SUCCESS) {
        OpenMX_ZgemmLogCpuFallback("my_hipblasZgemm", "native hipBLAS", status, m, n, k);
        goto cpu_fallback;
    }

    hip_status = hipMemcpy(C, d_C, m * n * sizeof(hipDoubleComplex), hipMemcpyDeviceToHost);
    if (hip_status != hipSuccess) {
        OpenMX_ZgemmLogHipCpuFallback("my_hipblasZgemm:hipMemcpy(C)", hip_status, m, n, k);
        goto cpu_fallback;
    }

    if (d_A != NULL)
        wait_hipfunc(hipFree(d_A));
    if (d_B != NULL)
        wait_hipfunc(hipFree(d_B));
    if (d_C != NULL)
        wait_hipfunc(hipFree(d_C));
    wait_hipfunc(hipblasDestroy(handle));
    return;

cpu_fallback:
    if (d_A != NULL)
        hipFree(d_A);
    if (d_B != NULL)
        hipFree(d_B);
    if (d_C != NULL)
        hipFree(d_C);
    if (handle != NULL)
        hipblasDestroy(handle);
    OpenMX_ZgemmCpu(transa, transb, m, n, k, A, B, C);
}

void my_hipblasZgemm_openmp(hipblasOperation_t transa, hipblasOperation_t transb, int m, int n, int k, dcomplex const * A,
    dcomplex const * B, dcomplex * C)
{
    hipblasHandle_t handle = NULL;
    hipblasStatus_t status;

    status = hipblasCreate(&handle);

#pragma omp target data map(present, alloc: A[0 : m * k], B[0 : k * n], C[0 : m * n])
    {
    if (status == HIPBLAS_STATUS_SUCCESS) {
#pragma omp target data use_device_ptr(A, B, C)
        {
            status = OpenMX_ZgemmTryGpu(handle, transa, transb, m, n, k, A, B, C, "my_hipblasZgemm_openmp");
        }
    }

    if (handle != NULL) {
        hipblasStatus_t destroy_status = hipblasDestroy(handle);
        if (destroy_status != HIPBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "<GEMM> rank %d: hipblasDestroy failed after ZGEMM: %s (%d).\n", OpenMX_ZgemmRank(),
                    OpenMX_ZgemmHipblasStatusName(destroy_status), (int)destroy_status);
            fflush(stderr);
        }
    }

    if (status != HIPBLAS_STATUS_SUCCESS) {
        OpenMX_ZgemmLogCpuFallback("my_hipblasZgemm_openmp", "native hipBLAS", status, m, n, k);
#pragma omp target update from(A[0 : m * k], B[0 : k * n])
        OpenMX_ZgemmCpu(transa, transb, m, n, k, A, B, C);
#pragma omp target update to(C[0 : m * n])
    }
    }
}

#include <hip/hip_runtime.h>
#include "openmx_gpu_transform.h"
#include <hipblas/hipblas.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    double r;
    double i;
} dcomplex;

extern "C" hipblasStatus_t openmx_gemmul8Dgemm(hipblasHandle_t, hipblasOperation_t, hipblasOperation_t, int, int, int,
                                               const double *, const double *, int, const double *, int,
                                               const double *, double *, int);
extern "C" hipblasStatus_t openmx_gemmul8DgemmUnblocked(hipblasHandle_t, hipblasOperation_t, hipblasOperation_t, int,
                                                        int, int, const double *, const double *, int, const double *,
                                                        int, const double *, double *, int);
extern "C" hipblasStatus_t openmx_gemmul8DgemmSolve(hipblasHandle_t, hipblasOperation_t, hipblasOperation_t, int, int,
                                                    int, const double *, const double *, int, const double *, int,
                                                    const double *, double *, int);
extern "C" hipblasStatus_t openmx_gemmul8ZgemmSolve(hipblasHandle_t, hipblasOperation_t, hipblasOperation_t, int, int,
                                                    int, const hipDoubleComplex *, const hipDoubleComplex *, int,
                                                    const hipDoubleComplex *, int, const hipDoubleComplex *,
                                                    hipDoubleComplex *, int);

static int ClusterNonColDMReportHipError(const char *where, hipError_t err)
{
    if (err == hipSuccess) {
        return 0;
    }

    fprintf(stderr, "<Cluster> rank 0: HIP density-matrix GPU path failed at %s: %s (%d).\n",
            where, hipGetErrorString(err), (int)err);
    fflush(stderr);
    return 1;
}

/* the same for the stages of the root dense solve */
static int ClusterNonColSolveReportHipError(const char *where, hipError_t err)
{
    if (err == hipSuccess) {
        return 0;
    }

    fprintf(stderr, "<Cluster> HIP non-collinear root dense solve failed at %s: %s (%d).\n",
            where, hipGetErrorString(err), (int)err);
    fflush(stderr);
    return 1;
}

/* the stream of a hipBLAS handle (the default stream without one) */
static hipStream_t ClusterNonColHandleStream(hipblasHandle_t handle)
{
    hipStream_t stream = NULL;

    if (handle != NULL && hipblasGetStream(handle, &stream) != HIPBLAS_STATUS_SUCCESS) stream = NULL;
    return stream;
}

static unsigned ClusterNonColGrid(size_t count, unsigned threads)
{
    size_t blocks = (count + threads - 1) / threads;

    if (blocks < 1) blocks = 1;
    if (blocks > 65535u * 16u) blocks = 65535u * 16u;
    return (unsigned)blocks;
}

/* Device buffers carved from a caller's scratch while it has room, else
   allocated on their own (given back by ClusterNonColCarveRelease); an
   allocation failure is returned, not retried. */
typedef struct
{
    unsigned char *base;
    size_t         size;
    size_t         used;
    void          *own[8];
    int            owned;
} ClusterNonColCarve;

static hipError_t ClusterNonColCarveTake(ClusterNonColCarve *carve, size_t bytes, void **out)
{
    size_t const padded = (bytes + 255u) / 256u * 256u;
    hipError_t   err;

    *out = NULL;
    if (carve->base != NULL && padded <= carve->size - carve->used) {
        *out = carve->base + carve->used;
        carve->used += padded;
        return hipSuccess;
    }
    if (8 <= carve->owned) return hipErrorOutOfMemory;
    err = hipMalloc(out, (padded == 0) ? 256 : padded);
    if (err != hipSuccess) {
        *out = NULL;
        (void)hipGetLastError();
        return err;
    }
    carve->own[carve->owned++] = *out;
    return hipSuccess;
}

static void ClusterNonColCarveRelease(ClusterNonColCarve *carve)
{
    for (int i = 0; i < carve->owned; i++) (void)hipFree(carve->own[i]);
    carve->owned = 0;
    carve->used = 0;
}

/* Room on the device for bytes of a path's own plus margin_mib, the room its
   products need (the capped GEMMul8 workspace, 256 MiB, unless the turn
   holds it already, and the library's scratch: a native hipBLAS fallback
   ends the process when hipBLASLt cannot allocate), so that a path with a
   cheaper alternative takes that instead. */
static int ClusterNonColDeviceRoom(size_t bytes, size_t margin_mib)
{
    size_t free_bytes = 0, total_bytes = 0;

    if (hipMemGetInfo(&free_bytes, &total_bytes) != hipSuccess) {
        (void)hipGetLastError();
        return 0;
    }
    return bytes + margin_mib * 1024 * 1024 <= free_bytes;
}

static int BandNonColReportHipError(const char *where, hipError_t err)
{
    if (err == hipSuccess) {
        return 0;
    }

    fprintf(stderr, "<Band> HIP non-collinear GPU path failed at %s: %s (%d).\n",
            where, hipGetErrorString(err), (int)err);
    fflush(stderr);
    return 1;
}

__device__ static double ClusterNonColWarpReduceSum(double value)
{
    for (int offset = warpSize / 2; 0 < offset; offset >>= 1) {
        value += __shfl_down(value, offset, warpSize);
    }
    return value;
}

__global__ static void ClusterNonColDMKernel(int entry_count, int size_H1, int n, int n2, int nk_occ,
                                             int calc_edm, const int *basis0, const int *basis1,
                                             const double *occ, const double *occ_e,
                                             const dcomplex *dense_evec, int ld, double *dm_buffer)
{
    const int warps_per_block = blockDim.x / warpSize;
    const int warp_in_block = threadIdx.x / warpSize;
    const int lane = threadIdx.x - warp_in_block * warpSize;
    const int p = (int)(blockIdx.x * warps_per_block + warp_in_block);

    if (entry_count <= p) {
        return;
    }

    const int ia = basis0[p];
    const int ib = basis1[p];
    double dm11_r = 0.0;
    double dm22_r = 0.0;
    double dm12_r = 0.0;
    double dm12_i = 0.0;
    double dm11_i = 0.0;
    double dm22_i = 0.0;
    double edm11_r = 0.0;
    double edm22_r = 0.0;
    double edm12_r = 0.0;
    double edm12_i = 0.0;

    for (int k = lane; k < nk_occ; k += warpSize) {
        const double w = occ[k];
        const dcomplex va_up = dense_evec[(size_t)ia * (size_t)ld + (size_t)k];
        const dcomplex vb_up = dense_evec[(size_t)ib * (size_t)ld + (size_t)k];
        const dcomplex va_dn = dense_evec[(size_t)(ia + n) * (size_t)ld + (size_t)k];
        const dcomplex vb_dn = dense_evec[(size_t)(ib + n) * (size_t)ld + (size_t)k];
        const double re11 = va_up.r * vb_up.r + va_up.i * vb_up.i;
        const double im11 = va_up.r * vb_up.i - va_up.i * vb_up.r;
        const double re22 = va_dn.r * vb_dn.r + va_dn.i * vb_dn.i;
        const double im22 = va_dn.r * vb_dn.i - va_dn.i * vb_dn.r;
        const double re12 = va_up.r * vb_dn.r + va_up.i * vb_dn.i;
        const double im12 = va_up.r * vb_dn.i - va_up.i * vb_dn.r;

        dm11_r += w * re11;
        dm22_r += w * re22;
        dm12_r += w * re12;
        dm12_i += w * im12;
        dm11_i += w * im11;
        dm22_i += w * im22;

        if (calc_edm) {
            const double ew = occ_e[k];
            edm11_r += ew * re11;
            edm22_r += ew * re22;
            edm12_r += ew * re12;
            edm12_i += ew * im12;
        }
    }

    dm11_r = ClusterNonColWarpReduceSum(dm11_r);
    dm22_r = ClusterNonColWarpReduceSum(dm22_r);
    dm12_r = ClusterNonColWarpReduceSum(dm12_r);
    dm12_i = ClusterNonColWarpReduceSum(dm12_i);
    dm11_i = ClusterNonColWarpReduceSum(dm11_i);
    dm22_i = ClusterNonColWarpReduceSum(dm22_i);
    if (calc_edm) {
        edm11_r = ClusterNonColWarpReduceSum(edm11_r);
        edm22_r = ClusterNonColWarpReduceSum(edm22_r);
        edm12_r = ClusterNonColWarpReduceSum(edm12_r);
        edm12_i = ClusterNonColWarpReduceSum(edm12_i);
    }

    if (lane == 0) {
        dm_buffer[(size_t)size_H1 * 0u + (size_t)p] = dm11_r;
        dm_buffer[(size_t)size_H1 * 1u + (size_t)p] = dm22_r;
        dm_buffer[(size_t)size_H1 * 2u + (size_t)p] = dm12_r;
        dm_buffer[(size_t)size_H1 * 3u + (size_t)p] = dm12_i;
        dm_buffer[(size_t)size_H1 * 4u + (size_t)p] = dm11_i;
        dm_buffer[(size_t)size_H1 * 5u + (size_t)p] = dm22_i;

        if (calc_edm) {
            dm_buffer[(size_t)size_H1 * 6u + (size_t)p] = edm11_r;
            dm_buffer[(size_t)size_H1 * 7u + (size_t)p] = edm22_r;
            dm_buffer[(size_t)size_H1 * 8u + (size_t)p] = edm12_r;
            dm_buffer[(size_t)size_H1 * 9u + (size_t)p] = edm12_i;
        }
    }
}

/* dense_evec: the state-major eigenvectors (state k of basis i at
   k + i ld_host) on the host, of which the first nk_occ states are copied to
   the device; or, with d_evec_in, already on the device with leading
   dimension ld_in.  The device buffers are carved from d_scratch
   (scratch_bytes, NULL for none: the root solve's arena, idle once the
   vectors are back-transformed) as far as it has room, the rest allocated
   for the call. */
static int ClusterNonCol_CalcDMRootDense_Run(int entry_count, int size_H1, int n, int n2, int nk_occ,
                                            int calc_edm, const int *basis0, const int *basis1,
                                            const double *occ, const double *occ_e,
                                            const dcomplex *dense_evec, int ld_host, const dcomplex *d_evec_in,
                                            int ld_in, void *d_scratch, size_t scratch_bytes, double *dm_buffer)
{
    const int dm_components = calc_edm ? 10 : 6;
    const int block_size = 256;
    int device_id = 0;
    hipDeviceProp_t prop;
    int warp_size = 64;
    int entries_per_block;
    dim3 block(block_size);
    dim3 grid;
    ClusterNonColCarve carve = {(unsigned char *)d_scratch, (d_scratch != NULL) ? scratch_bytes : 0, 0, {NULL}, 0};
    int *d_basis0 = NULL;
    int *d_basis1 = NULL;
    double *d_occ = NULL;
    double *d_occ_e = NULL;
    dcomplex *d_dense_evec = NULL;
    double *d_dm_buffer = NULL;
    hipError_t err;
    size_t basis_bytes = sizeof(int) * (size_t)entry_count;
    size_t occ_bytes = sizeof(double) * (size_t)nk_occ;
    size_t evec_bytes = sizeof(dcomplex) * (size_t)n2 * (size_t)nk_occ;
    size_t evec_dst_pitch = sizeof(dcomplex) * (size_t)nk_occ;
    size_t evec_src_pitch = sizeof(dcomplex) * (size_t)ld_host;
    size_t evec_width = sizeof(dcomplex) * (size_t)nk_occ;
    size_t evec_height = (size_t)n2;
    size_t dm_bytes = sizeof(double) * (size_t)size_H1 * (size_t)dm_components;
    int failed = 0;

    err = hipGetDevice(&device_id);
    if (ClusterNonColDMReportHipError("hipGetDevice", err)) goto cleanup_failed;
    err = hipGetDeviceProperties(&prop, device_id);
    if (ClusterNonColDMReportHipError("hipGetDeviceProperties", err)) goto cleanup_failed;
    if (0 < prop.warpSize) {
        warp_size = prop.warpSize;
    }
    entries_per_block = block_size / warp_size;
    if (entries_per_block <= 0) {
        entries_per_block = 1;
    }
    grid = dim3((unsigned int)((entry_count + entries_per_block - 1) / entries_per_block));

    err = ClusterNonColCarveTake(&carve, dm_bytes, (void **)&d_dm_buffer);
    if (ClusterNonColDMReportHipError("hipMalloc(dm_buffer)", err)) goto cleanup_failed;
    err = ClusterNonColCarveTake(&carve, basis_bytes, (void **)&d_basis0);
    if (ClusterNonColDMReportHipError("hipMalloc(basis0)", err)) goto cleanup_failed;
    err = ClusterNonColCarveTake(&carve, basis_bytes, (void **)&d_basis1);
    if (ClusterNonColDMReportHipError("hipMalloc(basis1)", err)) goto cleanup_failed;
    err = ClusterNonColCarveTake(&carve, occ_bytes, (void **)&d_occ);
    if (ClusterNonColDMReportHipError("hipMalloc(occ)", err)) goto cleanup_failed;
    if (calc_edm) {
        err = ClusterNonColCarveTake(&carve, occ_bytes, (void **)&d_occ_e);
        if (ClusterNonColDMReportHipError("hipMalloc(occ_e)", err)) goto cleanup_failed;
    }
    if (d_evec_in == NULL) {
        err = ClusterNonColCarveTake(&carve, evec_bytes, (void **)&d_dense_evec);
        if (ClusterNonColDMReportHipError("hipMalloc(dense_evec)", err)) goto cleanup_failed;
    }

    err = hipMemcpy(d_basis0, basis0, basis_bytes, hipMemcpyHostToDevice);
    if (ClusterNonColDMReportHipError("hipMemcpy(basis0)", err)) goto cleanup_failed;
    err = hipMemcpy(d_basis1, basis1, basis_bytes, hipMemcpyHostToDevice);
    if (ClusterNonColDMReportHipError("hipMemcpy(basis1)", err)) goto cleanup_failed;
    err = hipMemcpy(d_occ, occ, occ_bytes, hipMemcpyHostToDevice);
    if (ClusterNonColDMReportHipError("hipMemcpy(occ)", err)) goto cleanup_failed;
    if (calc_edm) {
        err = hipMemcpy(d_occ_e, occ_e, occ_bytes, hipMemcpyHostToDevice);
        if (ClusterNonColDMReportHipError("hipMemcpy(occ_e)", err)) goto cleanup_failed;
    }
    if (d_evec_in == NULL) {
        err = hipMemcpy2D(d_dense_evec, evec_dst_pitch, dense_evec, evec_src_pitch,
                          evec_width, evec_height, hipMemcpyHostToDevice);
        if (ClusterNonColDMReportHipError("hipMemcpy2D(dense_evec)", err)) goto cleanup_failed;
    }

    hipLaunchKernelGGL(ClusterNonColDMKernel, grid, block, 0, 0,
                       entry_count, size_H1, n, n2, nk_occ, calc_edm,
                       d_basis0, d_basis1, d_occ, d_occ_e,
                       (d_evec_in != NULL) ? d_evec_in : d_dense_evec, (d_evec_in != NULL) ? ld_in : nk_occ,
                       d_dm_buffer);
    err = hipGetLastError();
    if (ClusterNonColDMReportHipError("ClusterNonColDMKernel launch", err)) goto cleanup_failed;
    err = hipDeviceSynchronize();
    if (ClusterNonColDMReportHipError("ClusterNonColDMKernel synchronize", err)) goto cleanup_failed;

    err = hipMemcpy(dm_buffer, d_dm_buffer, dm_bytes, hipMemcpyDeviceToHost);
    if (ClusterNonColDMReportHipError("hipMemcpy(dm_buffer)", err)) goto cleanup_failed;

    goto cleanup;

cleanup_failed:
    failed = 1;
    (void)hipDeviceSynchronize();

cleanup:
    ClusterNonColCarveRelease(&carve);
    return failed;
}

/* dense_evec: state k of basis i at k + i ld on the host */
extern "C" int ClusterNonCol_CalcDMRootDense_HIP(int entry_count, int size_H1, int n, int n2, int nk_occ,
                                                 int calc_edm, const int *basis0, const int *basis1,
                                                 const double *occ, const double *occ_e,
                                                 const dcomplex *dense_evec, int ld, double *dm_buffer)
{
    return ClusterNonCol_CalcDMRootDense_Run(entry_count, size_H1, n, n2, nk_occ, calc_edm, basis0, basis1, occ,
                                             occ_e, dense_evec, ld, NULL, 0, NULL, 0, dm_buffer);
}

/* the same from eigenvectors that stay on the device (d_dense_evec, state
   k of basis i at k + i ld), with the buffers of the sums carved from
   d_scratch (scratch_bytes; NULL for none) */
extern "C" int ClusterNonCol_CalcDMRootDenseDevice_HIP(int entry_count, int size_H1, int n, int n2, int nk_occ,
                                                       int calc_edm, const int *basis0, const int *basis1,
                                                       const double *occ, const double *occ_e,
                                                       const dcomplex *d_dense_evec, int ld, void *d_scratch,
                                                       size_t scratch_bytes, double *dm_buffer)
{
    return ClusterNonCol_CalcDMRootDense_Run(entry_count, size_H1, n, n2, nk_occ, calc_edm, basis0, basis1, occ,
                                             occ_e, NULL, 0, d_dense_evec, ld, d_scratch, scratch_bytes, dm_buffer);
}

/* T (cols x rows, leading dimension ldt) = A^T of the first cols columns of
   A (rows x cols, leading dimension lda), without conjugation */
__global__ static void ClusterNonColTransposeKernel(const dcomplex *__restrict__ A, int lda, dcomplex *__restrict__ T,
                                                    int ldt, int rows, int cols)
{
    const size_t count = (size_t)rows * (size_t)cols;

    for (size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x; idx < count;
         idx += (size_t)gridDim.x * blockDim.x) {
        const int r = (int)(idx % (size_t)cols);   /* row of T: column of A */
        const int c = (int)(idx / (size_t)cols);   /* column of T: row of A */

        T[(size_t)r + (size_t)c * (size_t)ldt] = A[(size_t)c + (size_t)r * (size_t)lda];
    }
}

/* The back transform of the root dense path on the device: the first maxn
   columns of d_A (n2 x n2, the eigenvectors V of the transformed problem)
   become the state-major vectors (diag(S, S) V)^T (state l of basis i at
   l + i n2, rows maxn.. untouched) in place.  V^T goes to d_vt (maxn x n2,
   scratch), then each spin half of its real view (2 maxn x n: the real and
   imaginary parts of a state on consecutive rows) is multiplied by S^T in
   one real GEMM through the GEMMul8 bridge -- a quarter of the work of the
   complex product with diag(S, S), and no n2 x n2 copy of it.  d_S is the
   n x n transformed overlap.  Returns 0 on success. */
extern "C" int ClusterNonCol_BackTransformDevice_HIP(hipblasHandle_t handle, int n, int n2, int maxn, void *d_A,
                                                     void *d_vt, const double *d_S)
{
    hipStream_t     stream = NULL;
    hipblasStatus_t status = hipblasGetStream(handle, &stream);
    const double    one = 1.0, zero = 0.0;
    const size_t    count = (size_t)n2 * (size_t)maxn;
    size_t          blocks = (count + 255) / 256;
    hipError_t      err;

    if (status != HIPBLAS_STATUS_SUCCESS) return 1;
    if (blocks > 65535u * 16u) blocks = 65535u * 16u;
    if (!openmx_gpu_transform ||
        !openmx_gpu_transform(OPENMX_GPU_TRANSPOSE,1,maxn,n2,d_A,n2,NULL,0,d_vt,maxn,0,0.0))
    hipLaunchKernelGGL(ClusterNonColTransposeKernel, dim3((unsigned)blocks), dim3(256), 0, stream,
                       (const dcomplex *)d_A, n2, (dcomplex *)d_vt, maxn, n2, maxn);
    err = hipGetLastError();
    if (ClusterNonColDMReportHipError("ClusterNonColTransposeKernel launch", err)) return 1;
    for (int half = 0; half < 2; half++) {
        const int col = half * n;

        /* a product of the dense solve, which the other ranks wait for: the
           solve reserve of the GEMMul8 bridge */
        status = openmx_gemmul8DgemmUnblocked(handle, HIPBLAS_OP_N, HIPBLAS_OP_T, 2 * maxn, n, n, &one,
                                     (const double *)d_vt + (size_t)2 * (size_t)maxn * (size_t)col, 2 * maxn, d_S, n,
                                     &zero, (double *)d_A + (size_t)2 * (size_t)n2 * (size_t)col, 2 * n2);
        if (status != HIPBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "<Cluster> back transform GEMM failed (hipBLAS status %d)\n", (int)status);
            fflush(stderr);
            return 1;
        }
    }
    err = hipStreamSynchronize(stream);
    if (ClusterNonColDMReportHipError("back transform synchronize", err)) return 1;
    return 0;
}

/* ------------------------------------------------------------------
   The forward stage of the root dense solve on the device: the gathered
   sparse blocks scattered into dense n x n matrices, ImNL[2] added to the
   imaginary part of the off-diagonal spin block, and, after the forward
   transform (the hipBLAS products of Cluster_DFT_NonCol.c), the assembly of
   Hs2 in place of the host loop and its 505 MB upload (n2 = 5616).  Every
   kernel runs on the stream of the solve's hipBLAS handle. */

__global__ static void ClusterNonColScatterKernel(int tnum, const int *__restrict__ index,
                                                  const double *__restrict__ values, double *__restrict__ H)
{
    for (size_t p = blockIdx.x * (size_t)blockDim.x + threadIdx.x; p < (size_t)tnum;
         p += (size_t)gridDim.x * blockDim.x) {
        atomicAdd(&H[index[p]], values[p]);
    }
}

/* The dense n x n matrix d_H of tnum gathered entries: values (host) added at
   the dense positions (row + col n) that d_index holds on the device, after
   index (host; NULL when d_index holds this matrix's positions already) is
   uploaded; d_values (tnum doubles) stages the values.  Zeroed and added as
   by the OpenMP build of Cluster_DFT_NonCol.c, so the matrix is the same to
   the bit.  Synchronized; 0 on success. */
extern "C" int ClusterNonCol_ScatterDense_HIP(hipblasHandle_t handle, int tnum, int n, const double *values,
                                              const int *index, double *d_values, int *d_index, double *d_H)
{
    hipStream_t stream = ClusterNonColHandleStream(handle);
    hipError_t  err;

    if (index != NULL) {
        err = hipMemcpyAsync(d_index, index, (size_t)tnum * sizeof(int), hipMemcpyHostToDevice, stream);
        if (ClusterNonColSolveReportHipError("dense index upload", err)) return 1;
    }
    err = hipMemcpyAsync(d_values, values, (size_t)tnum * sizeof(double), hipMemcpyHostToDevice, stream);
    if (ClusterNonColSolveReportHipError("gathered matrix upload", err)) return 1;
    if (openmx_gpu_scatter &&
        openmx_gpu_scatter(n,tnum,0,d_index,tnum,d_values,0,NULL,NULL,0,d_H)) return 0;
    err = hipMemsetAsync(d_H, 0, (size_t)n * (size_t)n * sizeof(double), stream);
    if (ClusterNonColSolveReportHipError("dense matrix clear", err)) return 1;
    if (0 < tnum) {
        hipLaunchKernelGGL(ClusterNonColScatterKernel, dim3(ClusterNonColGrid((size_t)tnum, 256)), dim3(256), 0,
                           stream, tnum, d_index, d_values, d_H);
        err = hipGetLastError();
        if (ClusterNonColSolveReportHipError("ClusterNonColScatterKernel launch", err)) return 1;
    }
    /* the host buffers of the gather go away after the call */
    err = hipStreamSynchronize(stream);
    if (ClusterNonColSolveReportHipError("dense matrix synchronize", err)) return 1;
    return 0;
}

__global__ static void ClusterNonColAddKernel(size_t count, double *__restrict__ y, const double *__restrict__ x)
{
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < count; i += (size_t)gridDim.x * blockDim.x) {
        y[i] += x[i];
    }
}

/* d_y += d_x (count doubles), the host loop's single additions */
extern "C" int ClusterNonCol_AddDense_HIP(hipblasHandle_t handle, size_t count, double *d_y, const double *d_x)
{
    hipStream_t stream = ClusterNonColHandleStream(handle);
    hipError_t  err;

    if (count == 0) return 0;
    if (count<=INT32_MAX && openmx_gpu_transform &&
        openmx_gpu_transform(OPENMX_GPU_ADD,0,1,(int)count,d_y,1,d_x,1,d_y,1,0,0.0)) return 0;
    hipLaunchKernelGGL(ClusterNonColAddKernel, dim3(ClusterNonColGrid(count, 256)), dim3(256), 0, stream, count, d_y,
                       d_x);
    err = hipGetLastError();
    if (ClusterNonColSolveReportHipError("ClusterNonColAddKernel launch", err)) return 1;
    return 0;
}

#define CLUSTERNONCOL_TILE 32

/* the right half of Hs2 (columns n..2n-1 of the n2 x n2 matrix A): H12 over
   H22, Hxy = rxy + i ixy (n x n, leading dimension n) */
__global__ static void ClusterNonColAssembleRightKernel(int n, const double *__restrict__ r12,
                                                        const double *__restrict__ i12, const double *__restrict__ r22,
                                                        const double *__restrict__ i22, dcomplex *__restrict__ A)
{
    const size_t n2 = 2 * (size_t)n;
    const size_t count = n2 * (size_t)n;

    for (size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x; idx < count;
         idx += (size_t)gridDim.x * blockDim.x) {
        const size_t row = idx % n2, jj = idx / n2;
        dcomplex     z;

        if (row < (size_t)n) {
            z.r = r12[row + jj * (size_t)n];
            z.i = i12[row + jj * (size_t)n];
        }
        else {
            z.r = r22[(row - (size_t)n) + jj * (size_t)n];
            z.i = i22[(row - (size_t)n) + jj * (size_t)n];
        }
        A[row + ((size_t)n + jj) * n2] = z;
    }
}

/* the upper left block of Hs2, H11 = r11 + i i11 */
__global__ static void ClusterNonColAssembleUpperLeftKernel(int n, const double *__restrict__ r11,
                                                            const double *__restrict__ i11, dcomplex *__restrict__ A)
{
    const size_t n2 = 2 * (size_t)n;
    const size_t count = (size_t)n * (size_t)n;

    for (size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x; idx < count;
         idx += (size_t)gridDim.x * blockDim.x) {
        const size_t row = idx % (size_t)n, col = idx / (size_t)n;
        dcomplex     z;

        z.r = r11[idx];
        z.i = i11[idx];
        A[row + col * n2] = z;
    }
}

/* the lower left block of Hs2, H12^H, from the upper right block H12
   already in A: A[(n + ii) + col n2] = conj(A[col + (n + ii) n2]), through
   tiles in shared memory (32 x 8 threads per 32 x 32 tile) */
__global__ static void ClusterNonColAssembleLowerLeftKernel(int n, dcomplex *A)
{
    __shared__ dcomplex tile[CLUSTERNONCOL_TILE][CLUSTERNONCOL_TILE + 1];
    const size_t n2 = 2 * (size_t)n;
    const int    r0 = blockIdx.x * CLUSTERNONCOL_TILE;   /* rows of H12: the columns of the block written */
    const int    c0 = blockIdx.y * CLUSTERNONCOL_TILE;   /* columns of H12: its rows */

    for (int dy = threadIdx.y; dy < CLUSTERNONCOL_TILE; dy += blockDim.y) {
        const int r = r0 + threadIdx.x, c = c0 + dy;

        if (r < n && c < n) tile[dy][threadIdx.x] = A[(size_t)r + ((size_t)n + (size_t)c) * n2];
    }
    __syncthreads();
    for (int dy = threadIdx.y; dy < CLUSTERNONCOL_TILE; dy += blockDim.y) {
        const int ii = c0 + threadIdx.x, col = r0 + dy;

        if (ii < n && col < n) {
            const dcomplex u = tile[threadIdx.x][dy];
            dcomplex       z;

            z.r = u.r;
            z.i = -u.i;
            A[((size_t)n + (size_t)ii) + (size_t)col * n2] = z;
        }
    }
}

/* Hs2 in d_A (n2 x n2 complex) from the transformed real blocks, as the host
   loop of Cluster_DFT_NonCol.c builds it: H11 and H12 over H12^H and H22,
   Hxy = rxy + i ixy.  The right half (columns n..2n-1) is written first, from
   r12, i12, r22 and i22, which may lie in the left half of d_A; then the
   left half from r11 and i11, which must lie outside d_A, and the conjugate
   transpose of the H12 block just written.  Every element is a copy (or a
   copy with its sign flipped), so the matrix is the same to the bit.
   Synchronized; 0 on success. */
extern "C" int ClusterNonCol_AssembleHs2_HIP(hipblasHandle_t handle, int n, const double *r11, const double *i11,
                                             const double *r12, const double *i12, const double *r22,
                                             const double *i22, void *d_A)
{
    hipStream_t    stream = ClusterNonColHandleStream(handle);
    const size_t   nn = (size_t)n * (size_t)n;
    const unsigned tiles = (unsigned)((n + CLUSTERNONCOL_TILE - 1) / CLUSTERNONCOL_TILE);
    dcomplex      *A = (dcomplex *)d_A;
    hipError_t     err;

    if (openmx_gpu_spinor && openmx_gpu_spinor(n,1,r11,r22,r12,i11,i22,i12,d_A)) return 0;
    hipLaunchKernelGGL(ClusterNonColAssembleRightKernel, dim3(ClusterNonColGrid(2 * nn, 256)), dim3(256), 0, stream,
                       n, r12, i12, r22, i22, A);
    err = hipGetLastError();
    if (ClusterNonColSolveReportHipError("ClusterNonColAssembleRightKernel launch", err)) return 1;
    hipLaunchKernelGGL(ClusterNonColAssembleUpperLeftKernel, dim3(ClusterNonColGrid(nn, 256)), dim3(256), 0, stream,
                       n, r11, i11, A);
    err = hipGetLastError();
    if (ClusterNonColSolveReportHipError("ClusterNonColAssembleUpperLeftKernel launch", err)) return 1;
    hipLaunchKernelGGL(ClusterNonColAssembleLowerLeftKernel, dim3(tiles, tiles), dim3(CLUSTERNONCOL_TILE, 8), 0,
                       stream, n, A);
    err = hipGetLastError();
    if (ClusterNonColSolveReportHipError("ClusterNonColAssembleLowerLeftKernel launch", err)) return 1;
    err = hipStreamSynchronize(stream);
    if (ClusterNonColSolveReportHipError("Hs2 assembly synchronize", err)) return 1;
    return 0;
}

/* The back transform of the host-staged solve (no room for the
   device-resident one, or OPENMX_CLUSTER_NC_DEVICE_SOLVE=0): V (host,
   n2 x n2 complex, the eigenvectors in its first maxn columns) and the
   transformed overlap S (host, n x n) go to the device for the transpose and
   the two real GEMMs of ClusterNonCol_BackTransformDevice_HIP (with the
   capped GEMMul8 workspace), and the state-major vectors come back to evec
   (host, state l of basis i at l + i ld) -- instead of the complex GEMM
   against an n2 x n2 copy of diag(S, S), four times the work.  The device
   holds 2 n2 maxn complex and n^2 doubles for the call; handle may be NULL
   (one of its own then).  Returns 0 on success, 1 when a buffer, a copy or
   a product failed (the caller computes on the CPU). */
extern "C" int ClusterNonCol_BackTransformHost_HIP(hipblasHandle_t handle, int n, int n2, int maxn, const dcomplex *V,
                                                   const double *S, dcomplex *evec, int ld)
{
    const size_t    panel = (size_t)n2 * (size_t)maxn * sizeof(dcomplex);
    const double    one = 1.0, zero = 0.0;
    hipblasHandle_t own = NULL;
    dcomplex       *d_v = NULL, *d_vt = NULL;
    double         *d_s = NULL;
    hipStream_t     stream;
    int             failed = 1;

    if (!ClusterNonColDeviceRoom(2 * panel + (size_t)n * (size_t)n * sizeof(double), 512)) return 1;
    if (handle == NULL) {
        if (hipblasCreate(&own) != HIPBLAS_STATUS_SUCCESS) return 1;
        handle = own;
    }
    stream = ClusterNonColHandleStream(handle);
    do {
        hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;
        hipError_t      err;

        if (hipMalloc((void **)&d_v, panel) != hipSuccess || hipMalloc((void **)&d_vt, panel) != hipSuccess ||
            hipMalloc((void **)&d_s, (size_t)n * (size_t)n * sizeof(double)) != hipSuccess) {
            (void)hipGetLastError();
            break;
        }
        /* the first maxn columns of V are contiguous */
        err = hipMemcpy(d_v, V, panel, hipMemcpyHostToDevice);
        if (ClusterNonColSolveReportHipError("host-staged back transform upload", err)) break;
        err = hipMemcpy(d_s, S, (size_t)n * (size_t)n * sizeof(double), hipMemcpyHostToDevice);
        if (ClusterNonColSolveReportHipError("host-staged back transform upload", err)) break;
        hipLaunchKernelGGL(ClusterNonColTransposeKernel, dim3(ClusterNonColGrid(panel / sizeof(dcomplex), 256)),
                           dim3(256), 0, stream, (const dcomplex *)d_v, n2, d_vt, maxn, n2, maxn);
        err = hipGetLastError();
        if (ClusterNonColSolveReportHipError("ClusterNonColTransposeKernel launch", err)) break;
        /* the vectors (maxn x n2, leading dimension maxn) in place of V,
           whose copy the transpose consumed */
        for (int half = 0; half < 2 && status == HIPBLAS_STATUS_SUCCESS; half++) {
            const size_t col = (size_t)half * (size_t)n;

            status = openmx_gemmul8DgemmSolve(handle, HIPBLAS_OP_N, HIPBLAS_OP_T, 2 * maxn, n, n, &one,
                                              (const double *)d_vt + (size_t)2 * (size_t)maxn * col, 2 * maxn, d_s, n,
                                              &zero, (double *)d_v + (size_t)2 * (size_t)maxn * col, 2 * maxn);
        }
        if (status != HIPBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "<Cluster> host-staged back transform GEMM failed (hipBLAS status %d)\n", (int)status);
            fflush(stderr);
            break;
        }
        err = hipStreamSynchronize(stream);
        if (ClusterNonColSolveReportHipError("host-staged back transform synchronize", err)) break;
        err = hipMemcpy2D(evec, (size_t)ld * sizeof(dcomplex), d_v, (size_t)maxn * sizeof(dcomplex),
                          (size_t)maxn * sizeof(dcomplex), (size_t)n2, hipMemcpyDeviceToHost);
        if (ClusterNonColSolveReportHipError("host-staged back transform download", err)) break;
        failed = 0;
    } while (0);
    (void)hipDeviceSynchronize();
    if (d_s != NULL) (void)hipFree(d_s);
    if (d_vt != NULL) (void)hipFree(d_vt);
    if (d_v != NULL) (void)hipFree(d_v);
    if (own != NULL) (void)hipblasDestroy(own);
    return failed;
}

/* ------------------------------------------------------------------
   The density matrix of the root dense solve as products
   (OPENMX_CLUSTER_GPU_DM_GEMM of Cluster_DFT_NonCol.c): with V the
   state-major vectors (state k of basis i at k + i ld) and W = diag(w) V
   over the nk occupied states (nk x n2, leading dimension nk), the spin
   blocks P11 = W1^H W1, P12 = W1^H W2 and P22 = W2^H W2 (W1, W2: the
   columns of the two spin halves; P21 is never needed) are three
   n x n x nk complex products through the GEMMul8 bridge, w = sqrt(f); the
   energy density matrix (orbital optimization) the same way with w = f e
   against V itself.  A gather after each product takes the sparse entries
   into the device copy of the DM buffer, which comes back in one copy.
   Ported from the NVIDIA version (P = W^H W in full, n2 x n2), which
   needs four such blocks; the weights go to a buffer of their own, not over
   the eigenvalues (the NVIDIA version wrote them over its device ko). */

/* W[k + ia nk] = wt[k] V[k + ia ld], k < nk, ia < n2 */
__global__ static void ClusterNonColDMWeightKernel(int nk, int n2, const double *__restrict__ wt,
                                                   const dcomplex *__restrict__ V, int ld, dcomplex *__restrict__ W)
{
    const size_t count = (size_t)nk * (size_t)n2;

    for (size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x; idx < count;
         idx += (size_t)gridDim.x * blockDim.x) {
        const size_t   k = idx % (size_t)nk, ia = idx / (size_t)nk;
        const dcomplex v = V[k + ia * (size_t)ld];

        W[idx].r = wt[k] * v.r;
        W[idx].i = wt[k] * v.i;
    }
}

/* the entries p of one block P (n x n complex, leading dimension n):
   dm[comp_r size_H1 + p] = Re P[ia + ib n], and Im to comp_i unless negative */
__global__ static void ClusterNonColDMGatherKernel(int entry_count, int size_H1, int n, const int *__restrict__ basis0,
                                                   const int *__restrict__ basis1, const dcomplex *__restrict__ P,
                                                   int comp_r, int comp_i, double *__restrict__ dm)
{
    for (size_t p = blockIdx.x * (size_t)blockDim.x + threadIdx.x; p < (size_t)entry_count;
         p += (size_t)gridDim.x * blockDim.x) {
        const dcomplex z = P[(size_t)basis0[p] + (size_t)basis1[p] * (size_t)n];

        dm[(size_t)comp_r * (size_t)size_H1 + p] = z.r;
        if (0 <= comp_i) dm[(size_t)comp_i * (size_t)size_H1 + p] = z.i;
    }
}

/* The DM buffer (size_H1 x 6 doubles, 10 with calc_edm, in the order of
   ClusterNonColDMKernel) from the device vectors d_V (leading dimension
   ld), the occupations occ and occ_e = f e of the nk_occ occupied states.
   The device DM buffer, the entry tables, W and one n x n block of the
   products are carved from d_scratch (scratch_bytes; the arena of the
   solve, idle once the vectors are back-transformed) as far as it has room,
   the rest allocated for the call.  Returns 0 on success, 1 when a buffer or a
   product failed (dm_buffer then holds nothing usable: the caller takes the
   per-entry sums of ClusterNonColDMKernel instead). */
extern "C" int ClusterNonCol_CalcDMRootDenseGemm_HIP(hipblasHandle_t handle, int entry_count, int size_H1, int n,
                                                     int n2, int nk_occ, int calc_edm, const int *basis0,
                                                     const int *basis1, const double *occ, const double *occ_e,
                                                     const dcomplex *d_V, int ld, void *d_scratch,
                                                     size_t scratch_bytes, double *dm_buffer)
{
    /* the components of the blocks 11, 12 and 22: of the density matrix,
       then of the energy density matrix (-1: the imaginary part is not kept) */
    static const int       comp_r[2][3] = {{0, 2, 1}, {6, 8, 7}};
    static const int       comp_i[2][3] = {{4, 3, 5}, {-1, 9, -1}};
    const int              comps = calc_edm ? 10 : 6;
    const size_t           nk = (size_t)nk_occ;
    const hipDoubleComplex one = make_hipDoubleComplex(1.0, 0.0), zero = make_hipDoubleComplex(0.0, 0.0);
    hipStream_t            stream = ClusterNonColHandleStream(handle);
    ClusterNonColCarve     carve = {(unsigned char *)d_scratch, (d_scratch != NULL) ? scratch_bytes : 0, 0, {NULL}, 0};
    dcomplex              *W = NULL, *P = NULL;
    double                *d_dm = NULL, *d_wt = NULL, *wt;
    int                   *d_b0 = NULL, *d_b1 = NULL;
    int                    failed = 1;

    if (handle == NULL || d_V == NULL || nk_occ <= 0 || n <= 0 || n2 != 2 * n || ld < nk_occ || entry_count <= 0 ||
        size_H1 < entry_count) {
        return 1;
    }
    wt = (double *)malloc(nk * sizeof(double));
    if (wt == NULL) return 1;
    do {
        hipError_t err;
        int        pass_failed = 0;

        /* the buffers of the per-entry sums first (they fill the arena of an
           FP64 solve), then W and the product block */
        if (ClusterNonColCarveTake(&carve, (size_t)size_H1 * (size_t)comps * sizeof(double), (void **)&d_dm) !=
                hipSuccess ||
            ClusterNonColCarveTake(&carve, (size_t)entry_count * sizeof(int), (void **)&d_b0) != hipSuccess ||
            ClusterNonColCarveTake(&carve, (size_t)entry_count * sizeof(int), (void **)&d_b1) != hipSuccess ||
            ClusterNonColCarveTake(&carve, nk * sizeof(double), (void **)&d_wt) != hipSuccess ||
            ClusterNonColCarveTake(&carve, nk * (size_t)n2 * sizeof(dcomplex), (void **)&W) != hipSuccess ||
            ClusterNonColCarveTake(&carve, (size_t)n * (size_t)n * sizeof(dcomplex), (void **)&P) != hipSuccess ||
            !ClusterNonColDeviceRoom(0, 256)) {
            break;
        }
        err = hipMemcpy(d_b0, basis0, (size_t)entry_count * sizeof(int), hipMemcpyHostToDevice);
        if (ClusterNonColDMReportHipError("hipMemcpy(basis0)", err)) break;
        err = hipMemcpy(d_b1, basis1, (size_t)entry_count * sizeof(int), hipMemcpyHostToDevice);
        if (ClusterNonColDMReportHipError("hipMemcpy(basis1)", err)) break;

        for (int pass = 0; pass < (calc_edm ? 2 : 1) && !pass_failed; pass++) {
            for (size_t k = 0; k < nk; k++) wt[k] = (pass == 0) ? sqrt(fmax(0.0, occ[k])) : occ_e[k];
            /* the previous pass may still read the weights */
            err = hipStreamSynchronize(stream);
            if (ClusterNonColDMReportHipError("DM products synchronize", err)) {
                pass_failed = 1;
                break;
            }
            err = hipMemcpy(d_wt, wt, nk * sizeof(double), hipMemcpyHostToDevice);
            if (ClusterNonColDMReportHipError("hipMemcpy(weights)", err)) {
                pass_failed = 1;
                break;
            }
            hipLaunchKernelGGL(ClusterNonColDMWeightKernel, dim3(ClusterNonColGrid(nk * (size_t)n2, 256)), dim3(256), 0,
                               stream, nk_occ, n2, d_wt, d_V, ld, W);
            err = hipGetLastError();
            if (ClusterNonColDMReportHipError("ClusterNonColDMWeightKernel launch", err)) {
                pass_failed = 1;
                break;
            }
            for (int b = 0; b < 3; b++) {
                const dcomplex *A = W + ((b == 2) ? (size_t)n * nk : 0);
                const dcomplex *B = (pass == 0) ? W + ((b == 0) ? 0 : (size_t)n * nk)
                                                : d_V + ((b == 0) ? 0 : (size_t)n * (size_t)ld);
                const int       ldb = (pass == 0) ? nk_occ : ld;
                hipblasStatus_t status;

                /* a product of the dense solve, which the other ranks wait
                   for: the solve reserve of the GEMMul8 bridge */
                status = openmx_gemmul8ZgemmSolve(handle, HIPBLAS_OP_C, HIPBLAS_OP_N, n, n, nk_occ, &one,
                                                  (const hipDoubleComplex *)A, nk_occ, (const hipDoubleComplex *)B,
                                                  ldb, &zero, (hipDoubleComplex *)P, n);
                if (status != HIPBLAS_STATUS_SUCCESS) {
                    fprintf(stderr, "<Cluster> density-matrix product failed (hipBLAS status %d)\n", (int)status);
                    fflush(stderr);
                    pass_failed = 1;
                    break;
                }
                hipLaunchKernelGGL(ClusterNonColDMGatherKernel, dim3(ClusterNonColGrid((size_t)entry_count, 256)),
                                   dim3(256), 0, stream, entry_count, size_H1, n, d_b0, d_b1, P, comp_r[pass][b],
                                   comp_i[pass][b], d_dm);
                err = hipGetLastError();
                if (ClusterNonColDMReportHipError("ClusterNonColDMGatherKernel launch", err)) {
                    pass_failed = 1;
                    break;
                }
            }
        }
        if (pass_failed) break;
        err = hipStreamSynchronize(stream);
        if (ClusterNonColDMReportHipError("DM products synchronize", err)) break;
        err = hipMemcpy(dm_buffer, d_dm, (size_t)size_H1 * (size_t)comps * sizeof(double), hipMemcpyDeviceToHost);
        if (ClusterNonColDMReportHipError("hipMemcpy(dm_buffer)", err)) break;
        failed = 0;
    } while (0);
    /* nothing of the call may still run when its buffers go */
    (void)hipDeviceSynchronize();
    ClusterNonColCarveRelease(&carve);
    free(wt);
    return failed;
}

typedef struct
{
    int m_index;
    int dense_index;
    int phase_index;
} BandNonColHipConstructEntry;

__global__ static void BandNonColDenseMsKernel(int cpx_flag, int count,
                                               const BandNonColHipConstructEntry *entries,
                                               const double *M1, const double *phase_r,
                                               const double *phase_i, dcomplex *d_M)
{
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);

    if (idx < count) {
        const BandNonColHipConstructEntry entry = entries[idx];
        const double pr = phase_r[entry.phase_index];
        const double pi = phase_i[entry.phase_index];
        const double value = M1[entry.m_index];
        double vr;
        double vi;

        if (cpx_flag == 0) {
            vr = value * pr;
            vi = value * pi;
        } else {
            vr = -value * pi;
            vi =  value * pr;
        }

        atomicAdd(&d_M[entry.dense_index].r, vr);
        atomicAdd(&d_M[entry.dense_index].i, vi);
    }
}

extern "C" int BandNonCol_BuildDenseMs_HIP(int cpx_flag, int count, int h_count, int phase_count, int n,
                                           const BandNonColHipConstructEntry *entries,
                                           const double *phase_r, const double *phase_i,
                                           const double *M1, dcomplex *d_M)
{
    const int block_size = 256;
    dim3 block(block_size);
    BandNonColHipConstructEntry *d_entries = NULL;
    double *d_M1 = NULL;
    double *d_phase_r = NULL;
    double *d_phase_i = NULL;
    hipError_t err;
    const int entry_chunk = 262144;
    int alloc_entries = count < entry_chunk ? count : entry_chunk;
    size_t entry_bytes = sizeof(BandNonColHipConstructEntry) * (size_t)alloc_entries;
    size_t h_bytes = sizeof(double) * (size_t)h_count;
    size_t phase_bytes = sizeof(double) * (size_t)phase_count;
    size_t dense_bytes = sizeof(dcomplex) * (size_t)n * (size_t)n;
    int failed = 0;

    if (count < 0 || h_count <= 0 || phase_count <= 0 || n <= 0 ||
        entries == NULL || phase_r == NULL || phase_i == NULL || M1 == NULL || d_M == NULL) {
        fprintf(stderr, "<Band> HIP non-collinear dense matrix build received invalid arguments.\n");
        fflush(stderr);
        return 1;
    }

    if (0 < count) {
        err = hipMalloc((void **)&d_entries, entry_bytes);
        if (BandNonColReportHipError("hipMalloc(dense entries)", err)) goto cleanup_failed;
    }
    err = hipMalloc((void **)&d_M1, h_bytes);
    if (BandNonColReportHipError("hipMalloc(dense M1)", err)) goto cleanup_failed;
    err = hipMalloc((void **)&d_phase_r, phase_bytes);
    if (BandNonColReportHipError("hipMalloc(dense phase_r)", err)) goto cleanup_failed;
    err = hipMalloc((void **)&d_phase_i, phase_bytes);
    if (BandNonColReportHipError("hipMalloc(dense phase_i)", err)) goto cleanup_failed;

    err = hipMemcpy(d_M1, M1, h_bytes, hipMemcpyHostToDevice);
    if (BandNonColReportHipError("hipMemcpy(dense M1)", err)) goto cleanup_failed;
    err = hipMemcpy(d_phase_r, phase_r, phase_bytes, hipMemcpyHostToDevice);
    if (BandNonColReportHipError("hipMemcpy(dense phase_r)", err)) goto cleanup_failed;
    err = hipMemcpy(d_phase_i, phase_i, phase_bytes, hipMemcpyHostToDevice);
    if (BandNonColReportHipError("hipMemcpy(dense phase_i)", err)) goto cleanup_failed;
    static_assert(sizeof(BandNonColHipConstructEntry)==3*sizeof(int), "cooperative scatter entry layout");
    if (count>0 && count<=entry_chunk && openmx_gpu_scatter) {
        err=hipMemcpy(d_entries,entries,entry_bytes,hipMemcpyHostToDevice);
        if (BandNonColReportHipError("cooperative dense entries upload",err)) goto cleanup_failed;
        if (openmx_gpu_scatter(n,count,3,(const int*)d_entries,h_count,d_M1,phase_count,d_phase_r,d_phase_i,cpx_flag,d_M)) goto cleanup;
    }
    err = hipMemset(d_M, 0, dense_bytes);
    if (BandNonColReportHipError("hipMemset(dense matrix)", err)) goto cleanup_failed;

    for (int offset = 0; offset < count; offset += entry_chunk) {
        int chunk_count = count - offset;
        size_t chunk_bytes;
        dim3 grid;

        if (entry_chunk < chunk_count) {
            chunk_count = entry_chunk;
        }
        chunk_bytes = sizeof(BandNonColHipConstructEntry) * (size_t)chunk_count;
        grid = dim3((unsigned int)((chunk_count + block_size - 1) / block_size));

        err = hipMemcpy(d_entries, entries + offset, chunk_bytes, hipMemcpyHostToDevice);
        if (BandNonColReportHipError("hipMemcpy(dense entries)", err)) goto cleanup_failed;

        hipLaunchKernelGGL(BandNonColDenseMsKernel, grid, block, 0, 0,
                           cpx_flag, chunk_count, d_entries, d_M1, d_phase_r, d_phase_i, d_M);
        err = hipGetLastError();
        if (BandNonColReportHipError("BandNonColDenseMsKernel launch", err)) goto cleanup_failed;
        err = hipDeviceSynchronize();
        if (BandNonColReportHipError("BandNonColDenseMsKernel synchronize", err)) goto cleanup_failed;
    }

    goto cleanup;

cleanup_failed:
    failed = 1;

cleanup:
    if (d_phase_i != NULL) hipFree(d_phase_i);
    if (d_phase_r != NULL) hipFree(d_phase_r);
    if (d_M1 != NULL) hipFree(d_M1);
    if (d_entries != NULL) hipFree(d_entries);
    return failed;
}

__global__ static void BandNonColAddDenseKernel(int count, dcomplex *dst, const dcomplex *src)
{
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < count) {
        dst[idx].r += src[idx].r;
        dst[idx].i += src[idx].i;
    }
}

extern "C" int BandNonCol_AddDense_HIP(int count, dcomplex *d_dst, const dcomplex *d_src)
{
    const int block_size = 256;
    dim3 block(block_size);
    dim3 grid((unsigned int)((count + block_size - 1) / block_size));
    hipError_t err;

    if (count < 0 || d_dst == NULL || d_src == NULL) {
        fprintf(stderr, "<Band> HIP non-collinear dense add received invalid arguments.\n");
        fflush(stderr);
        return 1;
    }
    if (count == 0) {
        return 0;
    }

    if (openmx_gpu_transform &&
        openmx_gpu_transform(OPENMX_GPU_ADD,1,1,count,d_dst,1,d_src,1,d_dst,1,0,0.0)) return 0;
    hipLaunchKernelGGL(BandNonColAddDenseKernel, grid, block, 0, 0, count, d_dst, d_src);
    err = hipGetLastError();
    if (BandNonColReportHipError("BandNonColAddDenseKernel launch", err)) return 1;
    err = hipDeviceSynchronize();
    if (BandNonColReportHipError("BandNonColAddDenseKernel synchronize", err)) return 1;
    return 0;
}

__global__ static void BandNonColSymmetrizeHermitianKernel(int n, dcomplex *A)
{
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int nn = n * n;

    if (idx < nn) {
        int j = idx / n;
        int i = idx - j * n;

        if (i == j) {
            A[(size_t)i + (size_t)j * (size_t)n].i = 0.0;
        } else if (j < i) {
            size_t lij = (size_t)i + (size_t)j * (size_t)n;
            size_t uji = (size_t)j + (size_t)i * (size_t)n;
            double ar = 0.5 * (A[lij].r + A[uji].r);
            double ai = 0.5 * (A[lij].i - A[uji].i);
            A[lij].r = ar;
            A[lij].i = ai;
            A[uji].r =  ar;
            A[uji].i = -ai;
        }
    }
}

extern "C" int BandNonCol_SymmetrizeDenseHermitian_HIP(int n, dcomplex *d_A)
{
    const int block_size = 256;
    int count = n * n;
    dim3 block(block_size);
    dim3 grid((unsigned int)((count + block_size - 1) / block_size));
    hipError_t err;

    if (n <= 0 || d_A == NULL) {
        fprintf(stderr, "<Band> HIP non-collinear symmetrization received invalid arguments.\n");
        fflush(stderr);
        return 1;
    }

    if (openmx_gpu_transform &&
        openmx_gpu_transform(OPENMX_GPU_HERMITIAN,1,n,n,d_A,n,NULL,0,d_A,n,0,0.0)) return 0;
    hipLaunchKernelGGL(BandNonColSymmetrizeHermitianKernel, grid, block, 0, 0, n, d_A);
    err = hipGetLastError();
    if (BandNonColReportHipError("BandNonColSymmetrizeHermitianKernel launch", err)) return 1;
    err = hipDeviceSynchronize();
    if (BandNonColReportHipError("BandNonColSymmetrizeHermitianKernel synchronize", err)) return 1;
    return 0;
}

__global__ static void BandNonColBuildHs2Kernel(int n, int n2, const dcomplex *H11,
                                                const dcomplex *H22, const dcomplex *H12,
                                                dcomplex *H2)
{
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int n2n2 = n2 * n2;

    if (idx < n2n2) {
        int j = idx / n2;
        int i = idx - j * n2;
        size_t idx2 = (size_t)i + (size_t)j * (size_t)n2;

        if (i < n && j < n) {
            size_t src = (size_t)i + (size_t)j * (size_t)n;
            H2[idx2] = H11[src];
        } else if (i < n && n <= j) {
            int jj = j - n;
            size_t src = (size_t)i + (size_t)jj * (size_t)n;
            H2[idx2] = H12[src];
        } else if (n <= i && j < n) {
            int ii = i - n;
            size_t src = (size_t)j + (size_t)ii * (size_t)n;
            H2[idx2].r =  H12[src].r;
            H2[idx2].i = -H12[src].i;
        } else {
            int ii = i - n;
            int jj = j - n;
            size_t src = (size_t)ii + (size_t)jj * (size_t)n;
            H2[idx2] = H22[src];
        }
    }
}

extern "C" int BandNonCol_BuildDenseHs2_HIP(int n, int n2, const dcomplex *d_H11,
                                            const dcomplex *d_H22, const dcomplex *d_H12,
                                            dcomplex *d_H2)
{
    const int block_size = 256;
    int count = n2 * n2;
    dim3 block(block_size);
    dim3 grid((unsigned int)((count + block_size - 1) / block_size));
    hipError_t err;

    if (n <= 0 || n2 <= 0 || d_H11 == NULL || d_H22 == NULL || d_H12 == NULL || d_H2 == NULL) {
        fprintf(stderr, "<Band> HIP non-collinear Hs2 build received invalid arguments.\n");
        fflush(stderr);
        return 1;
    }

    if (n2==2*n && openmx_gpu_spinor &&
        openmx_gpu_spinor(n,0,d_H11,d_H22,d_H12,NULL,NULL,NULL,d_H2)) return 0;
    hipLaunchKernelGGL(BandNonColBuildHs2Kernel, grid, block, 0, 0, n, n2, d_H11, d_H22, d_H12, d_H2);
    err = hipGetLastError();
    if (BandNonColReportHipError("BandNonColBuildHs2Kernel launch", err)) return 1;
    err = hipDeviceSynchronize();
    if (BandNonColReportHipError("BandNonColBuildHs2Kernel synchronize", err)) return 1;
    return 0;
}

__global__ static void BandNonColBuildSs2Kernel(int n, int n2, const dcomplex *S, dcomplex *S2)
{
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int n2n2 = n2 * n2;

    if (idx < n2n2) {
        int i = idx / n2;
        int j = idx - i * n2;
        int idx2 = n2 * i + j;

        if (i < n && j < n) {
            int src = n * i + j;
            S2[idx2] = S[src];
        } else if (n <= i && n <= j) {
            int src = n * (i - n) + (j - n);
            S2[idx2] = S[src];
        } else {
            S2[idx2].r = 0.0;
            S2[idx2].i = 0.0;
        }
    }
}

extern "C" int BandNonCol_BuildDenseSs2_HIP(int n, int n2, const dcomplex *d_S, dcomplex *d_S2)
{
    const int block_size = 256;
    int count = n2 * n2;
    dim3 block(block_size);
    dim3 grid((unsigned int)((count + block_size - 1) / block_size));
    hipError_t err;

    if (n <= 0 || n2 <= 0 || d_S == NULL || d_S2 == NULL) {
        fprintf(stderr, "<Band> HIP non-collinear Ss2 build received invalid arguments.\n");
        fflush(stderr);
        return 1;
    }

    hipLaunchKernelGGL(BandNonColBuildSs2Kernel, grid, block, 0, 0, n, n2, d_S, d_S2);
    err = hipGetLastError();
    if (BandNonColReportHipError("BandNonColBuildSs2Kernel launch", err)) return 1;
    err = hipDeviceSynchronize();
    if (BandNonColReportHipError("BandNonColBuildSs2Kernel synchronize", err)) return 1;
    return 0;
}

__global__ static void BandNonColDMRootDenseKernel(int entry_count, int n, int n2, int nk_occ,
                                                   int dense_evec_ld,
                                                   const int *basis0, const int *basis1,
                                                   const int *phase_index,
                                                   const double *phase_r, const double *phase_i,
                                                   const double *occ, const double *eig_occ,
                                                   const dcomplex *dense_evec, double *dm_buffer,
                                                   int size_H1)
{
    const int warps_per_block = blockDim.x / warpSize;
    const int warp_in_block = threadIdx.x / warpSize;
    const int lane = threadIdx.x - warp_in_block * warpSize;
    const int p = (int)(blockIdx.x * warps_per_block + warp_in_block);

    if (entry_count <= p) {
        return;
    }

    const int ia = basis0[p];
    const int ib = basis1[p];
    const int ph = phase_index[p];
    const double co = phase_r[ph];
    const double si = phase_i[ph];
    double dm11_r = 0.0;
    double dm11_i = 0.0;
    double dm22_r = 0.0;
    double dm22_i = 0.0;
    double dm12_r = 0.0;
    double dm12_i = 0.0;
    double edm11_r = 0.0;
    double edm11_i = 0.0;
    double edm22_r = 0.0;
    double edm22_i = 0.0;

    for (int k = lane; k < nk_occ; k += warpSize) {
        const double w = occ[k];
        const double ew = eig_occ[k];
        const dcomplex va_up = dense_evec[(size_t)ia * (size_t)dense_evec_ld + (size_t)k];
        const dcomplex vb_up = dense_evec[(size_t)ib * (size_t)dense_evec_ld + (size_t)k];
        const dcomplex va_dn = dense_evec[(size_t)(ia + n) * (size_t)dense_evec_ld + (size_t)k];
        const dcomplex vb_dn = dense_evec[(size_t)(ib + n) * (size_t)dense_evec_ld + (size_t)k];
        const double re11 = va_up.r * vb_up.r + va_up.i * vb_up.i;
        const double im11 = va_up.r * vb_up.i - va_up.i * vb_up.r;
        const double re22 = va_dn.r * vb_dn.r + va_dn.i * vb_dn.i;
        const double im22 = va_dn.r * vb_dn.i - va_dn.i * vb_dn.r;
        const double re12 = va_up.r * vb_dn.r + va_up.i * vb_dn.i;
        const double im12 = va_up.r * vb_dn.i - va_up.i * vb_dn.r;

        dm11_r += w * re11;
        dm11_i += w * im11;
        dm22_r += w * re22;
        dm22_i += w * im22;
        dm12_r += w * re12;
        dm12_i += w * im12;
        edm11_r += ew * re11;
        edm11_i += ew * im11;
        edm22_r += ew * re22;
        edm22_i += ew * im22;
    }

    dm11_r = ClusterNonColWarpReduceSum(dm11_r);
    dm11_i = ClusterNonColWarpReduceSum(dm11_i);
    dm22_r = ClusterNonColWarpReduceSum(dm22_r);
    dm22_i = ClusterNonColWarpReduceSum(dm22_i);
    dm12_r = ClusterNonColWarpReduceSum(dm12_r);
    dm12_i = ClusterNonColWarpReduceSum(dm12_i);
    edm11_r = ClusterNonColWarpReduceSum(edm11_r);
    edm11_i = ClusterNonColWarpReduceSum(edm11_i);
    edm22_r = ClusterNonColWarpReduceSum(edm22_r);
    edm22_i = ClusterNonColWarpReduceSum(edm22_i);

    if (lane == 0) {
        dm_buffer[(size_t)size_H1 * 0u + (size_t)p] = co * dm11_r - si * dm11_i;
        dm_buffer[(size_t)size_H1 * 1u + (size_t)p] = co * dm22_r - si * dm22_i;
        dm_buffer[(size_t)size_H1 * 2u + (size_t)p] = co * dm12_r - si * dm12_i;
        dm_buffer[(size_t)size_H1 * 3u + (size_t)p] = co * dm12_i + si * dm12_r;
        dm_buffer[(size_t)size_H1 * 4u + (size_t)p] = co * dm11_i + si * dm11_r;
        dm_buffer[(size_t)size_H1 * 5u + (size_t)p] = co * dm22_i + si * dm22_r;
        dm_buffer[(size_t)size_H1 * 6u + (size_t)p] = co * edm11_r - si * edm11_i;
        dm_buffer[(size_t)size_H1 * 7u + (size_t)p] = co * edm22_r - si * edm22_i;
    }
}

typedef struct {
    int initialized;
    int device_id;
    size_t entry_capacity;
    size_t pair_capacity;
    size_t occ_capacity;
    size_t dm_capacity;
    int *d_basis0;
    int *d_basis1;
    int *d_phase_index;
    double *d_phase_r;
    double *d_phase_i;
    double *d_occ;
    double *d_eig_occ;
    double *d_dm_buffer;
} BandNonColDMDeviceWorkspace;

static BandNonColDMDeviceWorkspace BandNonCol_dm_device_workspace = {0};
static int BandNonCol_dm_device_atexit_registered = 0;

/* This routine frees only storage owned by the DM helper.  In particular,
   the d_C2 eigenvector pointer supplied by Band_DFT_NonCol is borrowed and
   is deliberately neither stored in this workspace nor freed here. */
static void BandNonCol_DMDeviceWorkspaceFreeOwned(BandNonColDMDeviceWorkspace *w)
{
    if (w->d_dm_buffer  != NULL) (void)hipFree(w->d_dm_buffer);
    if (w->d_eig_occ    != NULL) (void)hipFree(w->d_eig_occ);
    if (w->d_occ        != NULL) (void)hipFree(w->d_occ);
    if (w->d_phase_i    != NULL) (void)hipFree(w->d_phase_i);
    if (w->d_phase_r    != NULL) (void)hipFree(w->d_phase_r);
    if (w->d_phase_index!= NULL) (void)hipFree(w->d_phase_index);
    if (w->d_basis1     != NULL) (void)hipFree(w->d_basis1);
    if (w->d_basis0     != NULL) (void)hipFree(w->d_basis0);

    memset(w,0,sizeof(*w));
    w->device_id = -1;
}

static int BandNonCol_DMDeviceWorkspaceRelease(int report_errors)
{
    BandNonColDMDeviceWorkspace *w = &BandNonCol_dm_device_workspace;
    int saved_device = -1;
    int switched_device = 0;
    int release_failed = 0;
    hipError_t err;

    if (!w->initialized) return 0;

    err = hipGetDevice(&saved_device);
    if (err!=hipSuccess){
        if (report_errors) BandNonColReportHipError("hipGetDevice(DM workspace release)",err);
        return 1;
    }
    if (saved_device!=w->device_id){
        err = hipSetDevice(w->device_id);
        if (err!=hipSuccess){
            if (report_errors) BandNonColReportHipError("hipSetDevice(DM workspace owner)",err);
            return 1;
        }
        switched_device = 1;
    }

    /* The normal path has already synchronized, while the failure path may
       still have outstanding work referring to these allocations. */
    err = hipDeviceSynchronize();
    if (err!=hipSuccess){
        if (report_errors){
            BandNonColReportHipError("hipDeviceSynchronize(DM workspace release)",err);
        }
        release_failed = 1;
    }
    BandNonCol_DMDeviceWorkspaceFreeOwned(w);

    if (switched_device){
        err = hipSetDevice(saved_device);
        if (err!=hipSuccess){
            if (report_errors) BandNonColReportHipError("hipSetDevice(DM workspace restore)",err);
            return 1;
        }
    }
    return release_failed;
}

static void BandNonCol_DMDeviceWorkspaceProcessCleanup(void)
{
    BandNonColDMDeviceWorkspace *w = &BandNonCol_dm_device_workspace;
    int saved_device = -1;

    /* MPI may already be finalized here, so cleanup is intentionally silent. */
    if (!w->initialized) return;
    (void)hipGetDevice(&saved_device);
    if (hipSetDevice(w->device_id)==hipSuccess){
        (void)hipDeviceSynchronize();
        BandNonCol_DMDeviceWorkspaceFreeOwned(w);
    }
    if (0<=saved_device) (void)hipSetDevice(saved_device);
}

static int BandNonCol_DMDeviceWorkspaceRegisterCleanup(void)
{
    if (!BandNonCol_dm_device_atexit_registered){
        if (atexit(BandNonCol_DMDeviceWorkspaceProcessCleanup)!=0){
            fprintf(stderr,"<Band> Failed to register HIP DM workspace cleanup.\n");
            fflush(stderr);
            return 1;
        }
        BandNonCol_dm_device_atexit_registered = 1;
    }
    return 0;
}

static int BandNonCol_DMDeviceWorkspaceEnsure(int device_id, size_t entry_count,
                                              size_t pair_count, size_t occ_count,
                                              size_t dm_count)
{
    BandNonColDMDeviceWorkspace *w = &BandNonCol_dm_device_workspace;
    BandNonColDMDeviceWorkspace replacement;
    hipError_t err;

    if (BandNonCol_DMDeviceWorkspaceRegisterCleanup()!=0) return 1;

    if (w->initialized && w->device_id!=device_id){
        if (BandNonCol_DMDeviceWorkspaceRelease(1)!=0) return 1;
    }
    if (w->initialized && entry_count<=w->entry_capacity && pair_count<=w->pair_capacity &&
        occ_count<=w->occ_capacity && dm_count<=w->dm_capacity){
        return 0;
    }

    memset(&replacement,0,sizeof(replacement));
    replacement.device_id = device_id;
    replacement.entry_capacity = (w->entry_capacity<entry_count) ? entry_count : w->entry_capacity;
    replacement.pair_capacity = (w->pair_capacity<pair_count) ? pair_count : w->pair_capacity;
    replacement.occ_capacity = (w->occ_capacity<occ_count) ? occ_count : w->occ_capacity;
    replacement.dm_capacity = (w->dm_capacity<dm_count) ? dm_count : w->dm_capacity;

    err = hipMalloc((void **)&replacement.d_basis0,
                    sizeof(int)*replacement.entry_capacity);
    if (BandNonColReportHipError("hipMalloc(cached DM basis0)",err)) goto allocation_failed;
    err = hipMalloc((void **)&replacement.d_basis1,
                    sizeof(int)*replacement.entry_capacity);
    if (BandNonColReportHipError("hipMalloc(cached DM basis1)",err)) goto allocation_failed;
    err = hipMalloc((void **)&replacement.d_phase_index,
                    sizeof(int)*replacement.entry_capacity);
    if (BandNonColReportHipError("hipMalloc(cached DM phase_index)",err)) goto allocation_failed;
    err = hipMalloc((void **)&replacement.d_phase_r,
                    sizeof(double)*replacement.pair_capacity);
    if (BandNonColReportHipError("hipMalloc(cached DM phase_r)",err)) goto allocation_failed;
    err = hipMalloc((void **)&replacement.d_phase_i,
                    sizeof(double)*replacement.pair_capacity);
    if (BandNonColReportHipError("hipMalloc(cached DM phase_i)",err)) goto allocation_failed;
    err = hipMalloc((void **)&replacement.d_occ,
                    sizeof(double)*replacement.occ_capacity);
    if (BandNonColReportHipError("hipMalloc(cached DM occ)",err)) goto allocation_failed;
    err = hipMalloc((void **)&replacement.d_eig_occ,
                    sizeof(double)*replacement.occ_capacity);
    if (BandNonColReportHipError("hipMalloc(cached DM eig_occ)",err)) goto allocation_failed;
    err = hipMalloc((void **)&replacement.d_dm_buffer,
                    sizeof(double)*replacement.dm_capacity);
    if (BandNonColReportHipError("hipMalloc(cached DM buffer)",err)) goto allocation_failed;
    replacement.initialized = 1;

    if (w->initialized){
        /* Both workspaces belong to the current device at this point. */
        BandNonCol_DMDeviceWorkspaceFreeOwned(w);
    }
    *w = replacement;
    return 0;

allocation_failed:
    BandNonCol_DMDeviceWorkspaceFreeOwned(&replacement);
    return 1;
}

static int BandNonCol_CalcDMRootDenseImpl_HIP(int entry_count, int pair_count, int size_H1,
                                             int n, int n2, int nk_occ,
                                             const int *basis0, const int *basis1,
                                             const int *phase_index, const double *phase_r,
                                             const double *phase_i, const double *occ,
                                             const double *eig_occ, const dcomplex *dense_evec,
                                             int dense_evec_ld, int dense_evec_on_device,
                                             double *dm_buffer)
{
    const int block_size = 256;
    int device_id = 0;
    hipDeviceProp_t prop;
    int warp_size = 64;
    int entries_per_block;
    dim3 block(block_size);
    dim3 grid;
    BandNonColDMDeviceWorkspace *device_ws = NULL;
    int *d_basis0 = NULL;
    int *d_basis1 = NULL;
    int *d_phase_index = NULL;
    double *d_phase_r = NULL;
    double *d_phase_i = NULL;
    double *d_occ = NULL;
    double *d_eig_occ = NULL;
    dcomplex *d_dense_evec_copy = NULL;
    const dcomplex *d_dense_evec = NULL;
    double *d_dm_buffer = NULL;
    hipError_t err;
    size_t entry_int_bytes;
    size_t pair_bytes;
    size_t occ_bytes;
    size_t evec_bytes;
    size_t evec_dst_pitch;
    size_t evec_src_pitch;
    size_t evec_width;
    size_t evec_height;
    size_t dm_count;
    size_t dm_bytes;
    int failed = 0;

    if (entry_count <= 0 || pair_count <= 0 || size_H1 <= 0 ||
        n <= 0 || n2 <= 0 || (size_t)n2 != 2U*(size_t)n ||
        nk_occ <= 0 || dense_evec_ld < nk_occ ||
        basis0 == NULL || basis1 == NULL || phase_index == NULL ||
        phase_r == NULL || phase_i == NULL || occ == NULL || eig_occ == NULL ||
        dense_evec == NULL || dm_buffer == NULL) {
        fprintf(stderr, "<Band> HIP non-collinear density matrix received invalid arguments.\n");
        fflush(stderr);
        return 1;
    }

    entry_int_bytes = sizeof(int)*(size_t)entry_count;
    pair_bytes = sizeof(double)*(size_t)pair_count;
    occ_bytes = sizeof(double)*(size_t)nk_occ;
    evec_dst_pitch = sizeof(dcomplex)*(size_t)nk_occ;
    evec_src_pitch = sizeof(dcomplex)*(size_t)dense_evec_ld;
    evec_width = sizeof(dcomplex)*(size_t)nk_occ;
    evec_height = (size_t)n2;
    dm_count = (size_t)size_H1*8U;
    dm_bytes = sizeof(double)*dm_count;
    if ((size_t)n2>SIZE_MAX/(size_t)nk_occ ||
        (size_t)n2*(size_t)nk_occ>SIZE_MAX/sizeof(dcomplex)){
        fprintf(stderr,"<Band> HIP non-collinear density-matrix eigenvector size overflow.\n");
        fflush(stderr);
        return 1;
    }
    evec_bytes = sizeof(dcomplex)*(size_t)n2*(size_t)nk_occ;

    err = hipGetDevice(&device_id);
    if (BandNonColReportHipError("hipGetDevice(BandNonCol DM)", err)) goto cleanup_failed;
    err = hipGetDeviceProperties(&prop, device_id);
    if (BandNonColReportHipError("hipGetDeviceProperties(BandNonCol DM)", err)) goto cleanup_failed;
    if (0 < prop.warpSize) {
        warp_size = prop.warpSize;
    }
    entries_per_block = block_size / warp_size;
    if (entries_per_block <= 0) {
        entries_per_block = 1;
    }
    grid = dim3((unsigned int)((entry_count + entries_per_block - 1) / entries_per_block));

    if (dense_evec_on_device) {
        if (BandNonCol_DMDeviceWorkspaceEnsure(device_id,(size_t)entry_count,
                                               (size_t)pair_count,(size_t)nk_occ,
                                               dm_count)!=0){
            goto cleanup_failed;
        }
        device_ws = &BandNonCol_dm_device_workspace;
        d_basis0 = device_ws->d_basis0;
        d_basis1 = device_ws->d_basis1;
        d_phase_index = device_ws->d_phase_index;
        d_phase_r = device_ws->d_phase_r;
        d_phase_i = device_ws->d_phase_i;
        d_occ = device_ws->d_occ;
        d_eig_occ = device_ws->d_eig_occ;
        d_dm_buffer = device_ws->d_dm_buffer;
        d_dense_evec = dense_evec;
    }
    else {
        err = hipMalloc((void **)&d_basis0, entry_int_bytes);
        if (BandNonColReportHipError("hipMalloc(DM basis0)", err)) goto cleanup_failed;
        err = hipMalloc((void **)&d_basis1, entry_int_bytes);
        if (BandNonColReportHipError("hipMalloc(DM basis1)", err)) goto cleanup_failed;
        err = hipMalloc((void **)&d_phase_index, entry_int_bytes);
        if (BandNonColReportHipError("hipMalloc(DM phase_index)", err)) goto cleanup_failed;
        err = hipMalloc((void **)&d_phase_r, pair_bytes);
        if (BandNonColReportHipError("hipMalloc(DM phase_r)", err)) goto cleanup_failed;
        err = hipMalloc((void **)&d_phase_i, pair_bytes);
        if (BandNonColReportHipError("hipMalloc(DM phase_i)", err)) goto cleanup_failed;
        err = hipMalloc((void **)&d_occ, occ_bytes);
        if (BandNonColReportHipError("hipMalloc(DM occ)", err)) goto cleanup_failed;
        err = hipMalloc((void **)&d_eig_occ, occ_bytes);
        if (BandNonColReportHipError("hipMalloc(DM eig_occ)", err)) goto cleanup_failed;
        err = hipMalloc((void **)&d_dense_evec_copy, evec_bytes);
        if (BandNonColReportHipError("hipMalloc(DM dense_evec)", err)) goto cleanup_failed;
        d_dense_evec = d_dense_evec_copy;
        err = hipMalloc((void **)&d_dm_buffer, dm_bytes);
        if (BandNonColReportHipError("hipMalloc(DM buffer)", err)) goto cleanup_failed;
    }

    err = hipMemcpy(d_basis0, basis0, entry_int_bytes, hipMemcpyHostToDevice);
    if (BandNonColReportHipError("hipMemcpy(DM basis0)", err)) goto cleanup_failed;
    err = hipMemcpy(d_basis1, basis1, entry_int_bytes, hipMemcpyHostToDevice);
    if (BandNonColReportHipError("hipMemcpy(DM basis1)", err)) goto cleanup_failed;
    err = hipMemcpy(d_phase_index, phase_index, entry_int_bytes, hipMemcpyHostToDevice);
    if (BandNonColReportHipError("hipMemcpy(DM phase_index)", err)) goto cleanup_failed;
    err = hipMemcpy(d_phase_r, phase_r, pair_bytes, hipMemcpyHostToDevice);
    if (BandNonColReportHipError("hipMemcpy(DM phase_r)", err)) goto cleanup_failed;
    err = hipMemcpy(d_phase_i, phase_i, pair_bytes, hipMemcpyHostToDevice);
    if (BandNonColReportHipError("hipMemcpy(DM phase_i)", err)) goto cleanup_failed;
    err = hipMemcpy(d_occ, occ, occ_bytes, hipMemcpyHostToDevice);
    if (BandNonColReportHipError("hipMemcpy(DM occ)", err)) goto cleanup_failed;
    err = hipMemcpy(d_eig_occ, eig_occ, occ_bytes, hipMemcpyHostToDevice);
    if (BandNonColReportHipError("hipMemcpy(DM eig_occ)", err)) goto cleanup_failed;
    if (!dense_evec_on_device) {
        err = hipMemcpy2D(d_dense_evec_copy, evec_dst_pitch, dense_evec, evec_src_pitch,
                          evec_width, evec_height, hipMemcpyHostToDevice);
        if (BandNonColReportHipError("hipMemcpy2D(DM dense_evec)", err)) goto cleanup_failed;
    }
    err = hipMemset(d_dm_buffer, 0, dm_bytes);
    if (BandNonColReportHipError("hipMemset(DM buffer)", err)) goto cleanup_failed;

    hipLaunchKernelGGL(BandNonColDMRootDenseKernel, grid, block, 0, 0,
                       entry_count, n, n2, nk_occ,
                       dense_evec_on_device ? dense_evec_ld : nk_occ,
                       d_basis0, d_basis1, d_phase_index,
                       d_phase_r, d_phase_i, d_occ, d_eig_occ, d_dense_evec,
                       d_dm_buffer, size_H1);
    err = hipGetLastError();
    if (BandNonColReportHipError("BandNonColDMRootDenseKernel launch", err)) goto cleanup_failed;
    err = hipDeviceSynchronize();
    if (BandNonColReportHipError("BandNonColDMRootDenseKernel synchronize", err)) goto cleanup_failed;
    err = hipMemcpy(dm_buffer, d_dm_buffer, dm_bytes, hipMemcpyDeviceToHost);
    if (BandNonColReportHipError("hipMemcpy(DM buffer back)", err)) goto cleanup_failed;

    goto cleanup;

cleanup_failed:
    failed = 1;

cleanup:
    if (dense_evec_on_device){
        if (failed) (void)BandNonCol_DMDeviceWorkspaceRelease(1);
        return failed;
    }
    if (d_dm_buffer != NULL) hipFree(d_dm_buffer);
    if (d_dense_evec_copy != NULL) hipFree(d_dense_evec_copy);
    if (d_eig_occ != NULL) hipFree(d_eig_occ);
    if (d_occ != NULL) hipFree(d_occ);
    if (d_phase_i != NULL) hipFree(d_phase_i);
    if (d_phase_r != NULL) hipFree(d_phase_r);
    if (d_phase_index != NULL) hipFree(d_phase_index);
    if (d_basis1 != NULL) hipFree(d_basis1);
    if (d_basis0 != NULL) hipFree(d_basis0);
    return failed;
}

extern "C" int BandNonCol_CalcDMRootDense_HIP(int entry_count, int pair_count, int size_H1,
                                              int n, int n2, int nk_occ,
                                              const int *basis0, const int *basis1,
                                              const int *phase_index, const double *phase_r,
                                              const double *phase_i, const double *occ,
                                              const double *eig_occ, const dcomplex *dense_evec,
                                              double *dm_buffer)
{
    return BandNonCol_CalcDMRootDenseImpl_HIP(entry_count,pair_count,size_H1,n,n2,nk_occ,
                                              basis0,basis1,phase_index,phase_r,phase_i,
                                              occ,eig_occ,dense_evec,n2,0,dm_buffer);
}

extern "C" int BandNonCol_CalcDMRootDenseDevice_HIP(int entry_count, int pair_count, int size_H1,
                                                    int n, int n2, int nk_occ,
                                                    const int *basis0, const int *basis1,
                                                    const int *phase_index, const double *phase_r,
                                                    const double *phase_i, const double *occ,
                                                    const double *eig_occ,
                                                    const dcomplex *d_dense_evec,
                                                    int dense_evec_ld, double *dm_buffer)
{
    return BandNonCol_CalcDMRootDenseImpl_HIP(entry_count,pair_count,size_H1,n,n2,nk_occ,
                                              basis0,basis1,phase_index,phase_r,phase_i,
                                              occ,eig_occ,d_dense_evec,dense_evec_ld,1,dm_buffer);
}

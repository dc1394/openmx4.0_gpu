#include <hip/hip_runtime.h>
#include "openmx_gpu_transform.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static double ClusterCol_hip_dense_build_kernel_time = 0.0;
static double ClusterCol_hip_dm_kernel_time = 0.0;

/* Complete the lower triangle in place before the GEMM-based symmetric
   product. A padded LDS transpose coalesces both sides of the copy. */
__global__ static void ClusterColSymmetrizeLowerKernel(int n, double *a)
{
    if (blockIdx.x < blockIdx.y) return;
    __shared__ double tile[32][33];
    const int x = blockIdx.x * 32 + threadIdx.x;
    const int y = blockIdx.y * 32 + threadIdx.y;
    for (int j = 0; j < 32; j += 8) {
        if (x < n && y + j < n)
            tile[threadIdx.y + j][threadIdx.x] = a[(size_t)(y + j) * n + x];
    }
    __syncthreads();
    const int row = blockIdx.y * 32 + threadIdx.x;
    const int col = blockIdx.x * 32 + threadIdx.y;
    for (int j = 0; j < 32; j += 8) {
        if (row < n && col + j < n && row < col + j)
            a[(size_t)(col + j) * n + row] = tile[threadIdx.x][threadIdx.y + j];
    }
}

extern "C" int ClusterCol_SymmetrizeLower_HIP(int n, double *a, hipStream_t stream)
{
    hipLaunchKernelGGL(ClusterColSymmetrizeLowerKernel,
                       dim3((n + 31) / 32, (n + 31) / 32), dim3(32, 8), 0, stream, n, a);
    return (int)hipGetLastError();
}

extern "C" void ClusterCol_HIPDetailTimer_Reset(void)
{
    ClusterCol_hip_dense_build_kernel_time = 0.0;
    ClusterCol_hip_dm_kernel_time = 0.0;
}

extern "C" void ClusterCol_HIPDetailTimer_Get(double *dense_build_kernel, double *dm_kernel)
{
    if (dense_build_kernel != NULL) {
        *dense_build_kernel = ClusterCol_hip_dense_build_kernel_time;
    }
    if (dm_kernel != NULL) {
        *dm_kernel = ClusterCol_hip_dm_kernel_time;
    }
}

static int ClusterColDMReportHipError(const char *where, hipError_t err)
{
    if (err == hipSuccess) {
        return 0;
    }

    fprintf(stderr, "<Cluster> rank 0: HIP collinear density-matrix GPU path failed at %s: %s (%d).\n",
            where, hipGetErrorString(err), (int)err);
    fflush(stderr);
    return 1;
}

__global__ static void ClusterColDenseBuildKernel(int tnum, const double *H1, const int *dense_index, double *d_H)
{
    int p = (int)(blockIdx.x * blockDim.x + threadIdx.x);

    if (p < tnum) {
        atomicAdd(&d_H[dense_index[p]], H1[p]);
    }
}

extern "C" int ClusterCol_BuildDeviceDenseFromPacked_HIP(const double *H1, const int *dense_index,
                                                         int tnum, int n, double *d_H)
{
    const int block_size = 256;
    dim3 block(block_size);
    dim3 grid((unsigned int)((tnum + block_size - 1) / block_size));
    double *d_H1 = NULL;
    int *d_dense_index = NULL;
    hipError_t err;
    size_t sparse_bytes = sizeof(double) * (size_t)tnum;
    size_t index_bytes = sizeof(int) * (size_t)tnum;
    size_t dense_bytes = sizeof(double) * (size_t)n * (size_t)n;
    int failed = 0;
    hipEvent_t ev_start = NULL;
    hipEvent_t ev_stop = NULL;
    float kernel_ms = 0.0f;

    if (d_H == NULL) {
        fprintf(stderr, "<Cluster> rank 0: HIP dense matrix build received a NULL device matrix.\n");
        fflush(stderr);
        return 1;
    }

    err = hipMalloc((void **)&d_H1, sparse_bytes);
    if (ClusterColDMReportHipError("hipMalloc(dense H1)", err)) goto cleanup_failed;
    err = hipMalloc((void **)&d_dense_index, index_bytes);
    if (ClusterColDMReportHipError("hipMalloc(dense index)", err)) goto cleanup_failed;

    err = hipMemcpy(d_H1, H1, sparse_bytes, hipMemcpyHostToDevice);
    if (ClusterColDMReportHipError("hipMemcpy(dense H1)", err)) goto cleanup_failed;
    err = hipMemcpy(d_dense_index, dense_index, index_bytes, hipMemcpyHostToDevice);
    if (ClusterColDMReportHipError("hipMemcpy(dense index)", err)) goto cleanup_failed;

    if (openmx_gpu_scatter &&
        openmx_gpu_scatter(n,tnum,0,d_dense_index,tnum,d_H1,0,NULL,NULL,0,d_H)) goto cleanup;

    err = hipMemset(d_H, 0, dense_bytes);
    if (ClusterColDMReportHipError("hipMemset(dense matrix)", err)) goto cleanup_failed;

    err = hipEventCreate(&ev_start);
    if (ClusterColDMReportHipError("hipEventCreate(dense start)", err)) goto cleanup_failed;
    err = hipEventCreate(&ev_stop);
    if (ClusterColDMReportHipError("hipEventCreate(dense stop)", err)) goto cleanup_failed;
    err = hipEventRecord(ev_start, 0);
    if (ClusterColDMReportHipError("hipEventRecord(dense start)", err)) goto cleanup_failed;

    hipLaunchKernelGGL(ClusterColDenseBuildKernel, grid, block, 0, 0, tnum, d_H1, d_dense_index, d_H);
    err = hipGetLastError();
    if (ClusterColDMReportHipError("ClusterColDenseBuildKernel launch", err)) goto cleanup_failed;
    err = hipEventRecord(ev_stop, 0);
    if (ClusterColDMReportHipError("hipEventRecord(dense stop)", err)) goto cleanup_failed;
    err = hipEventSynchronize(ev_stop);
    if (ClusterColDMReportHipError("hipEventSynchronize(dense stop)", err)) goto cleanup_failed;
    err = hipEventElapsedTime(&kernel_ms, ev_start, ev_stop);
    if (ClusterColDMReportHipError("hipEventElapsedTime(dense)", err)) goto cleanup_failed;
    ClusterCol_hip_dense_build_kernel_time += (double)kernel_ms * 1.0e-3;
    err = hipDeviceSynchronize();
    if (ClusterColDMReportHipError("ClusterColDenseBuildKernel synchronize", err)) goto cleanup_failed;

    goto cleanup;

cleanup_failed:
    failed = 1;

cleanup:
    if (ev_stop != NULL) hipEventDestroy(ev_stop);
    if (ev_start != NULL) hipEventDestroy(ev_start);
    if (d_dense_index != NULL) hipFree(d_dense_index);
    if (d_H1 != NULL) hipFree(d_H1);
    return failed;
}

typedef struct
{
    double r;
    double i;
} BandColHipComplex;

typedef struct
{
    int h_index;
    int index0;
    int index1;
    int l1;
    int l2;
    int l3;
    int phase_index;
} BandColHipConstructEntry;

static int BandColReportHipError(const char *where, hipError_t err)
{
    if (err == hipSuccess) {
        return 0;
    }

    fprintf(stderr, "<Band> HIP collinear dense matrix GPU path failed at %s: %s (%d).\n",
            where, hipGetErrorString(err), (int)err);
    fflush(stderr);
    return 1;
}

__global__ static void BandColDenseCsHsKernel(int need_s, int count,
                                              const BandColHipConstructEntry *entries,
                                              const double *H1, const double *S1,
                                              const double *phase_r, const double *phase_i,
                                              BandColHipComplex *d_H, BandColHipComplex *d_S)
{
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);

    if (idx < count) {
        const BandColHipConstructEntry entry = entries[idx];
        const int h_index = entry.h_index;
        const int index0 = entry.index0;
        const int index1 = entry.index1;
        const int phase_index = entry.phase_index;
        const double pr = phase_r[phase_index];
        const double pi = phase_i[phase_index];
        const double h_real = H1[h_index] * pr;
        const double h_imag = H1[h_index] * pi;

        atomicAdd(&d_H[index0].r, h_real);
        atomicAdd(&d_H[index0].i, h_imag);

        if (0 <= index1) {
            atomicAdd(&d_H[index1].r, h_real);
            atomicAdd(&d_H[index1].i, -h_imag);
        }

        if (need_s) {
            const double s_real = S1[h_index] * pr;
            const double s_imag = S1[h_index] * pi;

            atomicAdd(&d_S[index0].r, s_real);
            atomicAdd(&d_S[index0].i, s_imag);

            if (0 <= index1) {
                atomicAdd(&d_S[index1].r, s_real);
                atomicAdd(&d_S[index1].i, -s_imag);
            }
        }
    }
}

/* d_entries is DEVICE-resident: Band_DFT_Col.c keeps the construct entry
   table on the device only (uploaded once per geometry), so this build no
   longer stages the multi-hundred-MB table on every k point. */
extern "C" int BandCol_BuildDenseCsHs_HIP(int need_s, int count, int h_count, int phase_count, int n,
                                          const BandColHipConstructEntry *d_entries, const double *phase_r,
                                          const double *phase_i, const double *H1, const double *S1,
                                          BandColHipComplex *d_H, BandColHipComplex *d_S)
{
    const int block_size = 256;
    dim3 block(block_size);
    dim3 grid((unsigned int)((count + block_size - 1) / block_size));
    double *d_H1 = NULL;
    double *d_S1 = NULL;
    double *d_phase_r = NULL;
    double *d_phase_i = NULL;
    hipError_t err;
    size_t h_bytes = sizeof(double) * (size_t)h_count;
    size_t phase_bytes = sizeof(double) * (size_t)phase_count;
    size_t dense_bytes = sizeof(BandColHipComplex) * (size_t)n * (size_t)n;
    int failed = 0;

    if (count < 0 || h_count <= 0 || phase_count <= 0 || n <= 0 ||
        (0 < count && d_entries == NULL) || phase_r == NULL || phase_i == NULL || H1 == NULL || d_H == NULL) {
        fprintf(stderr, "<Band> HIP dense matrix build received invalid arguments.\n");
        fflush(stderr);
        return 1;
    }
    if (need_s && (S1 == NULL || d_S == NULL)) {
        fprintf(stderr, "<Band> HIP dense overlap build received invalid S arguments.\n");
        fflush(stderr);
        return 1;
    }

    err = hipMalloc((void **)&d_H1, h_bytes);
    if (BandColReportHipError("hipMalloc(H1)", err)) goto cleanup_failed;
    if (need_s) {
        err = hipMalloc((void **)&d_S1, h_bytes);
        if (BandColReportHipError("hipMalloc(S1)", err)) goto cleanup_failed;
    }
    err = hipMalloc((void **)&d_phase_r, phase_bytes);
    if (BandColReportHipError("hipMalloc(phase_r)", err)) goto cleanup_failed;
    err = hipMalloc((void **)&d_phase_i, phase_bytes);
    if (BandColReportHipError("hipMalloc(phase_i)", err)) goto cleanup_failed;

    err = hipMemcpy(d_H1, H1, h_bytes, hipMemcpyHostToDevice);
    if (BandColReportHipError("hipMemcpy(H1)", err)) goto cleanup_failed;
    if (need_s) {
        err = hipMemcpy(d_S1, S1, h_bytes, hipMemcpyHostToDevice);
        if (BandColReportHipError("hipMemcpy(S1)", err)) goto cleanup_failed;
    }
    err = hipMemcpy(d_phase_r, phase_r, phase_bytes, hipMemcpyHostToDevice);
    if (BandColReportHipError("hipMemcpy(phase_r)", err)) goto cleanup_failed;
    err = hipMemcpy(d_phase_i, phase_i, phase_bytes, hipMemcpyHostToDevice);
    if (BandColReportHipError("hipMemcpy(phase_i)", err)) goto cleanup_failed;

    static_assert(sizeof(BandColHipConstructEntry)==7*sizeof(int), "cooperative scatter entry layout");
    if (openmx_gpu_scatter) {
        int done=openmx_gpu_scatter(n,count,7,(const int*)d_entries,h_count,d_H1,phase_count,d_phase_r,d_phase_i,0,d_H);
        if (done && need_s) done=openmx_gpu_scatter(n,count,7,(const int*)d_entries,h_count,d_S1,phase_count,d_phase_r,d_phase_i,0,d_S);
        if (done) goto cleanup;
    }

    err = hipMemset(d_H, 0, dense_bytes);
    if (BandColReportHipError("hipMemset(H)", err)) goto cleanup_failed;
    if (need_s) {
        err = hipMemset(d_S, 0, dense_bytes);
        if (BandColReportHipError("hipMemset(S)", err)) goto cleanup_failed;
    }

    if (0 < count) {
        hipLaunchKernelGGL(BandColDenseCsHsKernel, grid, block, 0, 0,
                           need_s, count, d_entries, d_H1, d_S1,
                           d_phase_r, d_phase_i, d_H, d_S);
        err = hipGetLastError();
        if (BandColReportHipError("BandColDenseCsHsKernel launch", err)) goto cleanup_failed;
        err = hipDeviceSynchronize();
        if (BandColReportHipError("BandColDenseCsHsKernel synchronize", err)) goto cleanup_failed;
    }

    goto cleanup;

cleanup_failed:
    failed = 1;

cleanup:
    if (d_phase_i != NULL) hipFree(d_phase_i);
    if (d_phase_r != NULL) hipFree(d_phase_r);
    if (d_S1 != NULL) hipFree(d_S1);
    if (d_H1 != NULL) hipFree(d_H1);
    return failed;
}

__device__ static double ClusterColWarpReduceSum(double value)
{
    for (int offset = warpSize / 2; 0 < offset; offset >>= 1) {
        value += __shfl_down(value, offset, warpSize);
    }
    return value;
}

__global__ static void BandColDMKernel(int entry_count, int nk, int evec_stride,
                                       const int *basis0, const int *basis1, const int *phase_index,
                                       const double *phase_r, const double *phase_i,
                                       const double *eigen, const double *occ_weight,
                                       const BandColHipComplex *evec, double *cdm, double *edm)
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
    double d1 = 0.0;
    double d2 = 0.0;
    double d3 = 0.0;
    double d4 = 0.0;

    for (int k = lane; k < nk; k += warpSize) {
        const double w = occ_weight[k];
        const BandColHipComplex va = evec[(size_t)ia * (size_t)evec_stride + (size_t)k];
        const BandColHipComplex vb = evec[(size_t)ib * (size_t)evec_stride + (size_t)k];
        const double r0 = va.r * w;
        const double im0 = va.i * w;
        const double r1 = vb.r * w;
        const double im1 = vb.i * w;
        const double reA = r0 * r1 + im0 * im1;
        const double imA = r0 * im1 - im0 * r1;

        d1 += reA;
        d2 += imA;
        d3 += reA * eigen[k];
        d4 += imA * eigen[k];
    }

    d1 = ClusterColWarpReduceSum(d1);
    d2 = ClusterColWarpReduceSum(d2);
    d3 = ClusterColWarpReduceSum(d3);
    d4 = ClusterColWarpReduceSum(d4);

    if (lane == 0) {
        cdm[p] += co * d1 - si * d2;
        edm[p] += co * d3 - si * d4;
    }
}

typedef struct {
    int initialized;
    int device_id;
    size_t entry_capacity;
    size_t pair_capacity;
    size_t state_capacity;
    int *d_basis0;
    int *d_basis1;
    int *d_phase_index;
    double *d_phase_r;
    double *d_phase_i;
    double *d_eigen;
    double *d_occ_weight;
    double *d_cdm;
    double *d_edm;
} BandColDMDeviceWorkspace;

static BandColDMDeviceWorkspace BandCol_dm_device_workspace = {0};
static int BandCol_dm_device_atexit_registered = 0;

/* The eigenvector matrix is owned by Band_DFT_Col.  Keep only the small
   metadata and output arrays here so repeated SCF steps do not serialize on
   nine hipMalloc/hipFree pairs per k-point owner. */
static void BandCol_DMDeviceWorkspaceFreeOwned(BandColDMDeviceWorkspace *w)
{
    if (w->d_edm        != NULL) (void)hipFree(w->d_edm);
    if (w->d_cdm        != NULL) (void)hipFree(w->d_cdm);
    if (w->d_occ_weight != NULL) (void)hipFree(w->d_occ_weight);
    if (w->d_eigen      != NULL) (void)hipFree(w->d_eigen);
    if (w->d_phase_i    != NULL) (void)hipFree(w->d_phase_i);
    if (w->d_phase_r    != NULL) (void)hipFree(w->d_phase_r);
    if (w->d_phase_index!= NULL) (void)hipFree(w->d_phase_index);
    if (w->d_basis1     != NULL) (void)hipFree(w->d_basis1);
    if (w->d_basis0     != NULL) (void)hipFree(w->d_basis0);

    memset(w,0,sizeof(*w));
    w->device_id = -1;
}

static int BandCol_DMDeviceWorkspaceRelease(int report_errors)
{
    BandColDMDeviceWorkspace *w = &BandCol_dm_device_workspace;
    int saved_device = -1;
    int switched_device = 0;
    int release_failed = 0;
    hipError_t err;

    if (!w->initialized) return 0;

    err = hipGetDevice(&saved_device);
    if (err!=hipSuccess){
        if (report_errors) BandColReportHipError("hipGetDevice(DM workspace release)",err);
        return 1;
    }
    if (saved_device!=w->device_id){
        err = hipSetDevice(w->device_id);
        if (err!=hipSuccess){
            if (report_errors) BandColReportHipError("hipSetDevice(DM workspace owner)",err);
            return 1;
        }
        switched_device = 1;
    }

    err = hipDeviceSynchronize();
    if (err!=hipSuccess){
        if (report_errors){
            BandColReportHipError("hipDeviceSynchronize(DM workspace release)",err);
        }
        release_failed = 1;
    }
    BandCol_DMDeviceWorkspaceFreeOwned(w);

    if (switched_device){
        err = hipSetDevice(saved_device);
        if (err!=hipSuccess){
            if (report_errors) BandColReportHipError("hipSetDevice(DM workspace restore)",err);
            return 1;
        }
    }
    return release_failed;
}

extern "C" int BandCol_ReleaseDMDeviceWorkspace_HIP(void)
{
    return BandCol_DMDeviceWorkspaceRelease(1);
}

static void BandCol_DMDeviceWorkspaceProcessCleanup(void)
{
    BandColDMDeviceWorkspace *w = &BandCol_dm_device_workspace;
    int saved_device = -1;

    /* MPI may already be finalized when atexit handlers run. */
    if (!w->initialized) return;
    (void)hipGetDevice(&saved_device);
    if (hipSetDevice(w->device_id)==hipSuccess){
        (void)hipDeviceSynchronize();
        BandCol_DMDeviceWorkspaceFreeOwned(w);
    }
    if (0<=saved_device) (void)hipSetDevice(saved_device);
}

static int BandCol_DMDeviceWorkspaceRegisterCleanup(void)
{
    if (!BandCol_dm_device_atexit_registered){
        if (atexit(BandCol_DMDeviceWorkspaceProcessCleanup)!=0){
            fprintf(stderr,"<Band> Failed to register HIP collinear DM workspace cleanup.\n");
            fflush(stderr);
            return 1;
        }
        BandCol_dm_device_atexit_registered = 1;
    }
    return 0;
}

static int BandCol_DMDeviceWorkspaceEnsure(int device_id, size_t entry_count,
                                           size_t pair_count, size_t state_count)
{
    BandColDMDeviceWorkspace *w = &BandCol_dm_device_workspace;
    BandColDMDeviceWorkspace replacement;
    hipError_t err;

    if (BandCol_DMDeviceWorkspaceRegisterCleanup()!=0) return 1;

    if (w->initialized && w->device_id!=device_id){
        if (BandCol_DMDeviceWorkspaceRelease(1)!=0) return 1;
    }
    if (w->initialized && entry_count<=w->entry_capacity &&
        pair_count<=w->pair_capacity && state_count<=w->state_capacity){
        return 0;
    }

    memset(&replacement,0,sizeof(replacement));
    replacement.device_id = device_id;
    replacement.entry_capacity = w->entry_capacity<entry_count ? entry_count : w->entry_capacity;
    replacement.pair_capacity = w->pair_capacity<pair_count ? pair_count : w->pair_capacity;
    replacement.state_capacity = w->state_capacity<state_count ? state_count : w->state_capacity;

    err = hipMalloc((void **)&replacement.d_basis0,
                    sizeof(int)*replacement.entry_capacity);
    if (BandColReportHipError("hipMalloc(cached DM basis0)",err)) goto allocation_failed;
    err = hipMalloc((void **)&replacement.d_basis1,
                    sizeof(int)*replacement.entry_capacity);
    if (BandColReportHipError("hipMalloc(cached DM basis1)",err)) goto allocation_failed;
    err = hipMalloc((void **)&replacement.d_phase_index,
                    sizeof(int)*replacement.entry_capacity);
    if (BandColReportHipError("hipMalloc(cached DM phase_index)",err)) goto allocation_failed;
    err = hipMalloc((void **)&replacement.d_phase_r,
                    sizeof(double)*replacement.pair_capacity);
    if (BandColReportHipError("hipMalloc(cached DM phase_r)",err)) goto allocation_failed;
    err = hipMalloc((void **)&replacement.d_phase_i,
                    sizeof(double)*replacement.pair_capacity);
    if (BandColReportHipError("hipMalloc(cached DM phase_i)",err)) goto allocation_failed;
    err = hipMalloc((void **)&replacement.d_eigen,
                    sizeof(double)*replacement.state_capacity);
    if (BandColReportHipError("hipMalloc(cached DM eigen)",err)) goto allocation_failed;
    err = hipMalloc((void **)&replacement.d_occ_weight,
                    sizeof(double)*replacement.state_capacity);
    if (BandColReportHipError("hipMalloc(cached DM occ)",err)) goto allocation_failed;
    err = hipMalloc((void **)&replacement.d_cdm,
                    sizeof(double)*replacement.entry_capacity);
    if (BandColReportHipError("hipMalloc(cached DM cdm)",err)) goto allocation_failed;
    err = hipMalloc((void **)&replacement.d_edm,
                    sizeof(double)*replacement.entry_capacity);
    if (BandColReportHipError("hipMalloc(cached DM edm)",err)) goto allocation_failed;
    replacement.initialized = 1;

    if (w->initialized) BandCol_DMDeviceWorkspaceFreeOwned(w);
    *w = replacement;
    return 0;

allocation_failed:
    BandCol_DMDeviceWorkspaceFreeOwned(&replacement);
    return 1;
}

extern "C" int BandCol_AccumulateDenseTransposedDM_HIP(int entry_count, int pair_count, int nk, int evec_stride,
                                                       const int *basis0, const int *basis1, const int *phase_index,
                                                       const double *phase_r, const double *phase_i,
                                                       const double *eigen, const double *occ_weight,
                                                       const BandColHipComplex *evec_device,
                                                       double *CDM1, double *EDM1)
{
    const int block_size = 256;
    int device_id = 0;
    hipDeviceProp_t prop;
    int warp_size = 64;
    int entries_per_block;
    dim3 block(block_size);
    dim3 grid;
    BandColDMDeviceWorkspace *device_ws = NULL;
    int *d_basis0;
    int *d_basis1;
    int *d_phase_index;
    double *d_phase_r;
    double *d_phase_i;
    double *d_eigen;
    double *d_occ_weight;
    double *d_cdm;
    double *d_edm;
    hipError_t err;
    size_t entry_int_bytes = sizeof(int) * (size_t)entry_count;
    size_t pair_bytes = sizeof(double) * (size_t)pair_count;
    size_t state_bytes = sizeof(double) * (size_t)nk;
    size_t dm_bytes = sizeof(double) * (size_t)entry_count;
    int failed = 0;

    if (entry_count <= 0 || pair_count <= 0 || nk <= 0 || evec_stride <= 0 ||
        basis0 == NULL || basis1 == NULL || phase_index == NULL ||
        phase_r == NULL || phase_i == NULL || eigen == NULL || occ_weight == NULL ||
        evec_device == NULL || CDM1 == NULL || EDM1 == NULL) {
        fprintf(stderr, "<Band> HIP density matrix build received invalid arguments.\n");
        fflush(stderr);
        return 1;
    }

    err = hipGetDevice(&device_id);
    if (BandColReportHipError("hipGetDevice(DM)", err)) goto cleanup_failed;
    err = hipGetDeviceProperties(&prop, device_id);
    if (BandColReportHipError("hipGetDeviceProperties(DM)", err)) goto cleanup_failed;
    if (0 < prop.warpSize) {
        warp_size = prop.warpSize;
    }
    entries_per_block = block_size / warp_size;
    if (entries_per_block <= 0) {
        entries_per_block = 1;
    }
    grid = dim3((unsigned int)((entry_count + entries_per_block - 1) / entries_per_block));

    if (BandCol_DMDeviceWorkspaceEnsure(device_id,(size_t)entry_count,
                                        (size_t)pair_count,(size_t)nk)!=0){
        goto cleanup_failed;
    }
    device_ws = &BandCol_dm_device_workspace;
    d_basis0 = device_ws->d_basis0;
    d_basis1 = device_ws->d_basis1;
    d_phase_index = device_ws->d_phase_index;
    d_phase_r = device_ws->d_phase_r;
    d_phase_i = device_ws->d_phase_i;
    d_eigen = device_ws->d_eigen;
    d_occ_weight = device_ws->d_occ_weight;
    d_cdm = device_ws->d_cdm;
    d_edm = device_ws->d_edm;

    err = hipMemcpy(d_basis0, basis0, entry_int_bytes, hipMemcpyHostToDevice);
    if (BandColReportHipError("hipMemcpy(DM basis0)", err)) goto cleanup_failed;
    err = hipMemcpy(d_basis1, basis1, entry_int_bytes, hipMemcpyHostToDevice);
    if (BandColReportHipError("hipMemcpy(DM basis1)", err)) goto cleanup_failed;
    err = hipMemcpy(d_phase_index, phase_index, entry_int_bytes, hipMemcpyHostToDevice);
    if (BandColReportHipError("hipMemcpy(DM phase_index)", err)) goto cleanup_failed;
    err = hipMemcpy(d_phase_r, phase_r, pair_bytes, hipMemcpyHostToDevice);
    if (BandColReportHipError("hipMemcpy(DM phase_r)", err)) goto cleanup_failed;
    err = hipMemcpy(d_phase_i, phase_i, pair_bytes, hipMemcpyHostToDevice);
    if (BandColReportHipError("hipMemcpy(DM phase_i)", err)) goto cleanup_failed;
    err = hipMemcpy(d_eigen, eigen, state_bytes, hipMemcpyHostToDevice);
    if (BandColReportHipError("hipMemcpy(DM eigen)", err)) goto cleanup_failed;
    err = hipMemcpy(d_occ_weight, occ_weight, state_bytes, hipMemcpyHostToDevice);
    if (BandColReportHipError("hipMemcpy(DM occ)", err)) goto cleanup_failed;
    err = hipMemcpy(d_cdm, CDM1, dm_bytes, hipMemcpyHostToDevice);
    if (BandColReportHipError("hipMemcpy(DM cdm)", err)) goto cleanup_failed;
    err = hipMemcpy(d_edm, EDM1, dm_bytes, hipMemcpyHostToDevice);
    if (BandColReportHipError("hipMemcpy(DM edm)", err)) goto cleanup_failed;

    hipLaunchKernelGGL(BandColDMKernel, grid, block, 0, 0,
                       entry_count, nk, evec_stride, d_basis0, d_basis1, d_phase_index,
                       d_phase_r, d_phase_i, d_eigen, d_occ_weight, evec_device, d_cdm, d_edm);
    err = hipGetLastError();
    if (BandColReportHipError("BandColDMKernel launch", err)) goto cleanup_failed;
    err = hipDeviceSynchronize();
    if (BandColReportHipError("BandColDMKernel synchronize", err)) goto cleanup_failed;

    err = hipMemcpy(CDM1, d_cdm, dm_bytes, hipMemcpyDeviceToHost);
    if (BandColReportHipError("hipMemcpy(DM cdm back)", err)) goto cleanup_failed;
    err = hipMemcpy(EDM1, d_edm, dm_bytes, hipMemcpyDeviceToHost);
    if (BandColReportHipError("hipMemcpy(DM edm back)", err)) goto cleanup_failed;

    goto cleanup;

cleanup_failed:
    failed = 1;

cleanup:
    if (failed && device_ws!=NULL) (void)BandCol_DMDeviceWorkspaceRelease(1);
    return failed;
}

__global__ static void ClusterColDMKernel(int entry_count, int size_H1, int maxn, int nk_occ, int calc_pdm,
                                          const int *basis0, const int *basis1,
                                          const double *occ, const double *occ_e, const double *pocc,
                                          const double *dense_evec, double *dm_buffer)
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
    double dm = 0.0;
    double edm = 0.0;
    double pdm = 0.0;

    for (int k = lane; k < nk_occ; k += warpSize) {
        const double prod = dense_evec[(size_t)ia * (size_t)maxn + (size_t)k] *
                            dense_evec[(size_t)ib * (size_t)maxn + (size_t)k];

        dm  += occ[k] * prod;
        edm += occ_e[k] * prod;
        if (calc_pdm) {
            pdm += pocc[k] * prod;
        }
    }

    dm = ClusterColWarpReduceSum(dm);
    edm = ClusterColWarpReduceSum(edm);
    if (calc_pdm) {
        pdm = ClusterColWarpReduceSum(pdm);
    }

    if (lane == 0) {
        dm_buffer[(size_t)p] = dm;
        dm_buffer[(size_t)size_H1 + (size_t)p] = edm;
        if (calc_pdm) {
            dm_buffer[(size_t)size_H1 * 2u + (size_t)p] = pdm;
        }
    }
}

extern "C" int ClusterCol_CalcDMRootDense_HIP(int entry_count, int size_H1, int n, int maxn, int nk_occ, int calc_pdm,
                                              const int *basis0, const int *basis1,
                                              const double *occ, const double *occ_e, const double *pocc,
                                              const double *dense_evec, const double *device_evec, double *dm_buffer)
{
    const int dm_components = calc_pdm ? 3 : 2;
    const int block_size = 256;
    int device_id = 0;
    hipDeviceProp_t prop;
    int warp_size = 64;
    int entries_per_block;
    dim3 block(block_size);
    dim3 grid;
    int *d_basis0 = NULL;
    int *d_basis1 = NULL;
    double *d_occ = NULL;
    double *d_occ_e = NULL;
    double *d_pocc = NULL;
    double *d_dense_evec = NULL;
    double *d_dm_buffer = NULL;
    hipError_t err;
    size_t basis_bytes = sizeof(int) * (size_t)entry_count;
    size_t occ_bytes = sizeof(double) * (size_t)nk_occ;
    size_t evec_bytes = sizeof(double) * (size_t)n * (size_t)maxn;
    size_t dm_bytes = sizeof(double) * (size_t)size_H1 * (size_t)dm_components;
    int failed = 0;
    hipEvent_t ev_start = NULL;
    hipEvent_t ev_stop = NULL;
    float kernel_ms = 0.0f;

    err = hipGetDevice(&device_id);
    if (ClusterColDMReportHipError("hipGetDevice", err)) goto cleanup_failed;
    err = hipGetDeviceProperties(&prop, device_id);
    if (ClusterColDMReportHipError("hipGetDeviceProperties", err)) goto cleanup_failed;
    if (0 < prop.warpSize) {
        warp_size = prop.warpSize;
    }
    entries_per_block = block_size / warp_size;
    if (entries_per_block <= 0) {
        entries_per_block = 1;
    }
    grid = dim3((unsigned int)((entry_count + entries_per_block - 1) / entries_per_block));

    err = hipMalloc((void **)&d_basis0, basis_bytes);
    if (ClusterColDMReportHipError("hipMalloc(basis0)", err)) goto cleanup_failed;
    err = hipMalloc((void **)&d_basis1, basis_bytes);
    if (ClusterColDMReportHipError("hipMalloc(basis1)", err)) goto cleanup_failed;
    err = hipMalloc((void **)&d_occ, occ_bytes);
    if (ClusterColDMReportHipError("hipMalloc(occ)", err)) goto cleanup_failed;
    err = hipMalloc((void **)&d_occ_e, occ_bytes);
    if (ClusterColDMReportHipError("hipMalloc(occ_e)", err)) goto cleanup_failed;
    if (calc_pdm) {
        err = hipMalloc((void **)&d_pocc, occ_bytes);
        if (ClusterColDMReportHipError("hipMalloc(pocc)", err)) goto cleanup_failed;
    }
    if (device_evec == NULL) {
        err = hipMalloc((void **)&d_dense_evec, evec_bytes);
        if (ClusterColDMReportHipError("hipMalloc(dense_evec)", err)) goto cleanup_failed;
    }
    err = hipMalloc((void **)&d_dm_buffer, dm_bytes);
    if (ClusterColDMReportHipError("hipMalloc(dm_buffer)", err)) goto cleanup_failed;

    err = hipMemcpy(d_basis0, basis0, basis_bytes, hipMemcpyHostToDevice);
    if (ClusterColDMReportHipError("hipMemcpy(basis0)", err)) goto cleanup_failed;
    err = hipMemcpy(d_basis1, basis1, basis_bytes, hipMemcpyHostToDevice);
    if (ClusterColDMReportHipError("hipMemcpy(basis1)", err)) goto cleanup_failed;
    err = hipMemcpy(d_occ, occ, occ_bytes, hipMemcpyHostToDevice);
    if (ClusterColDMReportHipError("hipMemcpy(occ)", err)) goto cleanup_failed;
    err = hipMemcpy(d_occ_e, occ_e, occ_bytes, hipMemcpyHostToDevice);
    if (ClusterColDMReportHipError("hipMemcpy(occ_e)", err)) goto cleanup_failed;
    if (calc_pdm) {
        err = hipMemcpy(d_pocc, pocc, occ_bytes, hipMemcpyHostToDevice);
        if (ClusterColDMReportHipError("hipMemcpy(pocc)", err)) goto cleanup_failed;
    }
    if (device_evec == NULL) {
        err = hipMemcpy(d_dense_evec, dense_evec, evec_bytes, hipMemcpyHostToDevice);
        if (ClusterColDMReportHipError("hipMemcpy(dense_evec)", err)) goto cleanup_failed;
    }

    err = hipEventCreate(&ev_start);
    if (ClusterColDMReportHipError("hipEventCreate(DM start)", err)) goto cleanup_failed;
    err = hipEventCreate(&ev_stop);
    if (ClusterColDMReportHipError("hipEventCreate(DM stop)", err)) goto cleanup_failed;
    err = hipEventRecord(ev_start, 0);
    if (ClusterColDMReportHipError("hipEventRecord(DM start)", err)) goto cleanup_failed;

    hipLaunchKernelGGL(ClusterColDMKernel, grid, block, 0, 0,
                       entry_count, size_H1, maxn, nk_occ, calc_pdm,
                       d_basis0, d_basis1, d_occ, d_occ_e, d_pocc,
                       device_evec != NULL ? device_evec : d_dense_evec, d_dm_buffer);
    err = hipGetLastError();
    if (ClusterColDMReportHipError("ClusterColDMKernel launch", err)) goto cleanup_failed;
    err = hipEventRecord(ev_stop, 0);
    if (ClusterColDMReportHipError("hipEventRecord(DM stop)", err)) goto cleanup_failed;
    err = hipEventSynchronize(ev_stop);
    if (ClusterColDMReportHipError("hipEventSynchronize(DM stop)", err)) goto cleanup_failed;
    err = hipEventElapsedTime(&kernel_ms, ev_start, ev_stop);
    if (ClusterColDMReportHipError("hipEventElapsedTime(DM)", err)) goto cleanup_failed;
    ClusterCol_hip_dm_kernel_time += (double)kernel_ms * 1.0e-3;
    err = hipDeviceSynchronize();
    if (ClusterColDMReportHipError("ClusterColDMKernel synchronize", err)) goto cleanup_failed;

    err = hipMemcpy(dm_buffer, d_dm_buffer, dm_bytes, hipMemcpyDeviceToHost);
    if (ClusterColDMReportHipError("hipMemcpy(dm_buffer)", err)) goto cleanup_failed;

    goto cleanup;

cleanup_failed:
    failed = 1;

cleanup:
    if (ev_stop != NULL) hipEventDestroy(ev_stop);
    if (ev_start != NULL) hipEventDestroy(ev_start);
    if (d_dm_buffer != NULL) hipFree(d_dm_buffer);
    if (d_dense_evec != NULL) hipFree(d_dense_evec);
    if (d_pocc != NULL) hipFree(d_pocc);
    if (d_occ_e != NULL) hipFree(d_occ_e);
    if (d_occ != NULL) hipFree(d_occ);
    if (d_basis1 != NULL) hipFree(d_basis1);
    if (d_basis0 != NULL) hipFree(d_basis0);
    return failed;
}

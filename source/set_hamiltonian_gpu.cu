#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>

namespace {

constexpr int kThreads = 256;
constexpr int kPreferredGridTile = 32;

/* FP32: the weighted orbital tile is float, the products are FP32 and the
   sum is compensated (Neumaier), so the result is within a few FP32 ulps
   of the exact sum of the FP32 products; the FP64 base of the block is
   kept apart.  __fmul_rn / __fadd_rn keep the compiler from fusing the
   product into the sum, which the compensation relies on. */
/* MODE 2 (double-float): the weighted orbital is split into two floats
   (hi + lo = the FP64 value to 1e-14), each product with the FP32 orbital
   is formed exactly (TwoProd through FMA): the hi parts of a tile are
   summed with Neumaier's compensation and the lo parts (2^-24 smaller)
   plainly; the tile's (sum, compensation, lo sum) go into the FP64
   accumulator.  The error per tile is of order 32 x eps32^2, so the
   result is FP64-grade at FP32 cost.  The potential of a tile's points
   is split once into (hi, lo) (the kernel's only FP64 arithmetic: on a
   GeForce, with FP64 at 1/64 of the FP32 rate, the former FP64
   weighting of every orbital of every point by every y-block bounded
   the kernel). */
__device__ __forceinline__ void neumaier_add(float &s, float &c, float t)
{
    const float u = __fadd_rn(s, t);
    if (fabsf(s) >= fabsf(t)) c = __fadd_rn(c, __fadd_rn(__fadd_rn(s, -u), t));
    else c = __fadd_rn(c, __fadd_rn(__fadd_rn(t, -u), s));
    s = u;
}

template <int MODE>
__global__ void matrix_elements_kernel(
    int pair_count,
    int spin_count,
    std::size_t vpot_len,
    double grid_vol,
    int max_no,
    int grid_tile,
    const int *__restrict__ pair_NO0,
    const int *__restrict__ pair_NO1,
    const int *__restrict__ pair_NOLG,
    const int *__restrict__ nolg_MN,
    const int *__restrict__ nolg_Nc,
    const std::size_t *__restrict__ pair_h_offset,
    const std::size_t *__restrict__ pair_nolg_offset,
    const std::size_t *__restrict__ pair_orbs0_offset,
    const std::size_t *__restrict__ pair_orbs1_offset,
    const float *__restrict__ orbs0buf,
    const float *__restrict__ orbs1buf,
    const double *__restrict__ vpotgrid,
    double *__restrict__ hbuf)
{
    const int pair = static_cast<int>(blockIdx.x);
    const int NO0 = pair_NO0[pair];
    const int NO1 = pair_NO1[pair];
    const int NOLG = pair_NOLG[pair];
    const std::size_t mat_size = static_cast<std::size_t>(NO0) * static_cast<std::size_t>(NO1);
    const std::size_t output_count = static_cast<std::size_t>(spin_count) * mat_size;
    const std::size_t e = static_cast<std::size_t>(blockIdx.y) * blockDim.x + threadIdx.x;

    if (output_count <= static_cast<std::size_t>(blockIdx.y) * blockDim.x) return;

    extern __shared__ unsigned char shared_bytes[];
    const std::size_t weighted_count =
        static_cast<std::size_t>(spin_count) * static_cast<std::size_t>(grid_tile) * static_cast<std::size_t>(max_no);
    double *weighted0 = reinterpret_cast<double *>(shared_bytes);
    float *weighted0f = reinterpret_cast<float *>(shared_bytes);
    float2 *weighted0d = reinterpret_cast<float2 *>(shared_bytes);
    float *tile1 = (MODE == 1) ? (weighted0f + weighted_count)
                 : (MODE == 2) ? reinterpret_cast<float *>(weighted0d + weighted_count)
                 : reinterpret_cast<float *>(weighted0 + weighted_count);
    /* MODE 2: grid_vol * vpot of the tile's points per spin as (hi, lo) */
    float2 *vtile = (MODE == 2) ? reinterpret_cast<float2 *>(tile1 + static_cast<std::size_t>(grid_tile) * max_no)
                                : nullptr;

    const bool active = e < output_count;
    int spin = 0;
    int i = 0;
    int j = 0;
    double sum = 0.0;
    float sumf = 0.0f, compf = 0.0f;
    const std::size_t h_off = pair_h_offset[pair];
    const std::size_t nolg_off = pair_nolg_offset[pair];
    const std::size_t orbs0_off = pair_orbs0_offset[pair];
    const std::size_t orbs1_off = pair_orbs1_offset[pair];

    if (active) {
        spin = static_cast<int>(e / mat_size);
        const std::size_t ij = e - static_cast<std::size_t>(spin) * mat_size;
        i = static_cast<int>(ij / static_cast<std::size_t>(NO1));
        j = static_cast<int>(ij - static_cast<std::size_t>(i) * static_cast<std::size_t>(NO1));
        sum = hbuf[h_off + e];
    }

    for (int base = 0; base < NOLG; base += grid_tile) {
        const int count = (base + grid_tile <= NOLG) ? grid_tile : (NOLG - base);
        const int weighted0_count = spin_count * count * NO0;
        const int tile1_count = count * NO1;

        if (MODE == 2) {
            for (int index = static_cast<int>(threadIdx.x); index < spin_count * count; index += blockDim.x) {
                const int s = index / count;
                const int grid = index - s * count;
                const std::size_t pt = nolg_off + static_cast<std::size_t>(base + grid);
                const double v = grid_vol * vpotgrid[static_cast<std::size_t>(s) * vpot_len +
                                                     static_cast<std::size_t>(nolg_MN[pt])];
                const float hi = static_cast<float>(v);
                vtile[static_cast<std::size_t>(s) * grid_tile + grid] =
                    make_float2(hi, static_cast<float>(v - static_cast<double>(hi)));
            }
            __syncthreads();
        }
        for (int index = static_cast<int>(threadIdx.x); index < weighted0_count; index += blockDim.x) {
            const int spin_grid_size = count * NO0;
            const int s = index / spin_grid_size;
            const int spin_index = index - s * spin_grid_size;
            const int grid = spin_index / NO0;
            const int orbital = spin_index - grid * NO0;
            const std::size_t pt = nolg_off + static_cast<std::size_t>(base + grid);
            if (MODE == 2) {
                /* (v.x + v.y) * phi0 as (hi, lo): the product exact through an FMA */
                const float2 v = vtile[static_cast<std::size_t>(s) * grid_tile + grid];
                const float p0 = orbs0buf[orbs0_off + static_cast<std::size_t>(nolg_Nc[pt]) * NO0 + orbital];
                const float hi = __fmul_rn(v.x, p0);
                const float lo = __fadd_rn(__fmaf_rn(v.x, p0, -hi), __fmul_rn(v.y, p0));
                weighted0d[(static_cast<std::size_t>(s) * grid_tile + grid) * max_no + orbital] = make_float2(hi, lo);
                continue;
            }
            const double w =
                (grid_vol * vpotgrid[static_cast<std::size_t>(s) * vpot_len +
                                     static_cast<std::size_t>(nolg_MN[pt])]) *
                static_cast<double>(orbs0buf[
                    orbs0_off + static_cast<std::size_t>(nolg_Nc[pt]) * NO0 + orbital]);
            if (MODE == 1) {
                weighted0f[(static_cast<std::size_t>(s) * grid_tile + grid) * max_no + orbital] = static_cast<float>(w);
            }
            else {
                weighted0[(static_cast<std::size_t>(s) * grid_tile + grid) * max_no + orbital] = w;
            }
        }
        for (int index = static_cast<int>(threadIdx.x); index < tile1_count; index += blockDim.x) {
            const int grid = index / NO1;
            const int orbital = index - grid * NO1;
            tile1[static_cast<std::size_t>(grid) * max_no + orbital] =
                orbs1buf[orbs1_off + static_cast<std::size_t>(base + grid) * NO1 + orbital];
        }
        __syncthreads();

        if (active) {
            if (MODE == 1) {
                for (int grid = 0; grid < count; grid++) {
                    const float t = __fmul_rn(weighted0f[(static_cast<std::size_t>(spin) * grid_tile + grid) * max_no + i],
                                              tile1[static_cast<std::size_t>(grid) * max_no + j]);
                    neumaier_add(sumf, compf, t);
                }
            }
            else if (MODE == 2) {
                float ts = 0.0f, tc = 0.0f, qs = 0.0f;
                for (int grid = 0; grid < count; grid++) {
                    const float2 w = weighted0d[(static_cast<std::size_t>(spin) * grid_tile + grid) * max_no + i];
                    const float t1 = tile1[static_cast<std::size_t>(grid) * max_no + j];
                    const float p = __fmul_rn(w.x, t1);
                    const float e = __fmaf_rn(w.x, t1, -p);        /* p + e = w.x * t1 exactly */
                    const float q = __fmaf_rn(w.y, t1, e);         /* the small part */
                    neumaier_add(ts, tc, p);
                    qs = __fadd_rn(qs, q);
                }
                sum += static_cast<double>(ts) + static_cast<double>(tc) + static_cast<double>(qs);
            }
            else {
                for (int grid = 0; grid < count; grid++) {
                    sum += weighted0[(static_cast<std::size_t>(spin) * grid_tile + grid) * max_no + i] *
                           static_cast<double>(tile1[static_cast<std::size_t>(grid) * max_no + j]);
                }
            }
        }
        __syncthreads();
    }

    if (active) hbuf[h_off + e] = (MODE == 1) ? (sum + static_cast<double>(sumf) + static_cast<double>(compf)) : sum;
}


/* The on-the-fly density of a batch of atom pairs, double-float (the
   density matrix as hi + lo floats): one thread per overlap point, the
   pair's density matrix staged once per block in shared memory (float2),
   the point's orbital rows phi1 (NO1) and phi0 (NO0) in registers.  Per
   point and spin: rho += sum_i phi0[i] * (sum_j DM[i][j] phi1[j]); the
   products are formed exactly (TwoProd through FMA), the hi parts of a row
   summed with Neumaier's compensation, the lo parts plainly, the rows
   accumulated as a float pair (TwoSum), and the pair goes into the FP64
   density by one atomic add.  Register arrays need compile-time bounds,
   so the kernel comes in tiers NMAX = 16, 32, 48 chosen by the batch's
   largest orbital count; the loops are unrolled with a break at the
   pair's counts so that every index is static.  The OpenACC kernel of
   Set_Hamiltonian.c (same arithmetic, DM read from global memory per
   product) stays as the fallback and the FP64 / plain-FP32 modes. */
template <int NMAX>
__global__ void density_kernel(
    int spin_count,
    const int *__restrict__ pair_NO0,
    const int *__restrict__ pair_NO1,
    const int *__restrict__ pair_NOLG,
    const std::size_t *__restrict__ pair_h_offset,
    const std::size_t *__restrict__ pair_nolg_offset,
    const std::size_t *__restrict__ pair_orbs0_offset,
    const std::size_t *__restrict__ pair_orbs1_offset,
    const std::size_t *__restrict__ pair_out_base,
    const int *__restrict__ nolg_Nc,
    const float *__restrict__ orbs0buf,
    const float *__restrict__ orbs1buf,
    const float *__restrict__ dmf,
    const float *__restrict__ dml,
    double *__restrict__ tmpden,
    std::size_t output_count)
{
    const int pair = static_cast<int>(blockIdx.x);
    const int NO0 = pair_NO0[pair];
    const int NO1 = pair_NO1[pair];
    const int NOLG = pair_NOLG[pair];
    const int k = static_cast<int>(blockIdx.y) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);

    if (static_cast<int>(blockIdx.y) * static_cast<int>(blockDim.x) >= NOLG) return;

    extern __shared__ float2 dm_s[];
    const std::size_t mat = static_cast<std::size_t>(NO0) * static_cast<std::size_t>(NO1);
    const std::size_t hoff = pair_h_offset[pair];
    const std::size_t dm_count = static_cast<std::size_t>(spin_count) * mat;
    for (std::size_t idx = threadIdx.x; idx < dm_count; idx += blockDim.x)
        dm_s[idx] = make_float2(dmf[hoff + idx], dml[hoff + idx]);
    __syncthreads();
    if (k >= NOLG) return;

    const int Nc = nolg_Nc[pair_nolg_offset[pair] + static_cast<std::size_t>(k)];
    const float *phi1 = orbs1buf + pair_orbs1_offset[pair] + static_cast<std::size_t>(k) * NO1;
    const float *phi0 = orbs0buf + pair_orbs0_offset[pair] + static_cast<std::size_t>(Nc) * NO0;
    float f1[NMAX], f0[NMAX];
#pragma unroll
    for (int j = 0; j < NMAX; j++) f1[j] = (j < NO1) ? phi1[j] : 0.0f;
#pragma unroll
    for (int i = 0; i < NMAX; i++) f0[i] = (i < NO0) ? phi0[i] : 0.0f;
    const std::size_t out = pair_out_base[pair] + static_cast<std::size_t>(Nc);

    for (int s = 0; s < spin_count; s++) {
        const float2 *dms = dm_s + static_cast<std::size_t>(s) * mat;
        float eh = 0.0f, el = 0.0f;
#pragma unroll
        for (int i = 0; i < NMAX; i++) {
            if (i >= NO0) break;
            const float2 *row = dms + static_cast<std::size_t>(i) * NO1;
            float ts = 0.0f, tc = 0.0f, qs = 0.0f;
#pragma unroll
            for (int j = 0; j < NMAX; j++) {
                if (j >= NO1) break;
                const float2 d = row[j];
                const float p = __fmul_rn(f1[j], d.x);
                const float e = __fmaf_rn(f1[j], d.x, -p);
                const float q = __fmaf_rn(f1[j], d.y, e);
                neumaier_add(ts, tc, p);
                qs = __fadd_rn(qs, q);
            }
            {
                const float ph = __fmul_rn(f0[i], ts);
                const float pl = __fadd_rn(__fmaf_rn(f0[i], ts, -ph), __fmul_rn(f0[i], __fadd_rn(tc, qs)));
                const float su = __fadd_rn(eh, ph);
                const float v = __fadd_rn(su, -eh);
                el = __fadd_rn(el, __fadd_rn(__fadd_rn(__fadd_rn(eh, -__fadd_rn(su, -v)), __fadd_rn(ph, -v)), pl));
                eh = su;
            }
        }
        atomicAdd(tmpden + static_cast<std::size_t>(s) * output_count + out,
                  static_cast<double>(eh) + static_cast<double>(el));
    }
}

template <int NMAX>
static int density_launch(int pair_count, int spin_count, int max_nolg, std::size_t shared_bytes,
    const int *pair_NO0, const int *pair_NO1, const int *pair_NOLG,
    const std::size_t *pair_h_offset, const std::size_t *pair_nolg_offset,
    const std::size_t *pair_orbs0_offset, const std::size_t *pair_orbs1_offset,
    const std::size_t *pair_out_base, const int *nolg_Nc, const float *orbs0buf, const float *orbs1buf,
    const float *dmf, const float *dml, double *tmpden, std::size_t output_count)
{
    constexpr int kDensityThreads = 128;
    static std::size_t attribute_bytes = 0;
    if (shared_bytes > 48 * 1024 && shared_bytes > attribute_bytes) {
        if (cudaFuncSetAttribute(density_kernel<NMAX>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 static_cast<int>(shared_bytes)) != cudaSuccess) {
            (void)cudaGetLastError();
            return 1;
        }
        attribute_bytes = shared_bytes;
    }
    const dim3 block(kDensityThreads, 1u, 1u);
    const dim3 grid(static_cast<unsigned>(pair_count),
                    static_cast<unsigned>((max_nolg + kDensityThreads - 1) / kDensityThreads), 1u);
    density_kernel<NMAX><<<grid, block, shared_bytes>>>(spin_count, pair_NO0, pair_NO1, pair_NOLG,
        pair_h_offset, pair_nolg_offset, pair_orbs0_offset, pair_orbs1_offset, pair_out_base, nolg_Nc,
        orbs0buf, orbs1buf, dmf, dml, tmpden, output_count);
    {
        const cudaError_t launch = cudaGetLastError();
        if (launch != cudaSuccess) return 1;   /* nothing ran: the caller takes the OpenACC kernel */
    }
    return (cudaDeviceSynchronize() == cudaSuccess) ? 0 : -1;
}

} // namespace

/* 0 = done, 1 = not taken (nothing was written; use the OpenACC kernel),
   -1 = the kernel failed after running (the density is suspect) */
extern "C" int Set_Hamiltonian_Cuda_Density(
    int pair_count, int spin_count, int max_no, int max_nolg,
    const int *pair_NO0, const int *pair_NO1, const int *pair_NOLG,
    const std::size_t *pair_h_offset, const std::size_t *pair_nolg_offset,
    const std::size_t *pair_orbs0_offset, const std::size_t *pair_orbs1_offset,
    const std::size_t *pair_out_base, const int *nolg_Nc, const float *orbs0buf, const float *orbs1buf,
    const float *dmf, const float *dml, double *tmpden, std::size_t output_count)
{
    if (pair_count <= 0 || spin_count <= 0 || max_no <= 0 || max_nolg <= 0) return 0;
    if (max_no > 48) return 1;
    (void)cudaGetLastError();
    const std::size_t shared_bytes = static_cast<std::size_t>(spin_count) * static_cast<std::size_t>(max_no) *
                                     static_cast<std::size_t>(max_no) * sizeof(float2);
    int device = 0, max_shared = 0;
    if (cudaGetDevice(&device) != cudaSuccess) return 1;
    if (cudaDeviceGetAttribute(&max_shared, cudaDevAttrMaxSharedMemoryPerBlockOptin, device) != cudaSuccess) return 1;
    if (shared_bytes > static_cast<std::size_t>(max_shared)) return 1;
    if (max_no <= 16)
        return density_launch<16>(pair_count, spin_count, max_nolg, shared_bytes, pair_NO0, pair_NO1, pair_NOLG,
            pair_h_offset, pair_nolg_offset, pair_orbs0_offset, pair_orbs1_offset, pair_out_base, nolg_Nc,
            orbs0buf, orbs1buf, dmf, dml, tmpden, output_count);
    if (max_no <= 32)
        return density_launch<32>(pair_count, spin_count, max_nolg, shared_bytes, pair_NO0, pair_NO1, pair_NOLG,
            pair_h_offset, pair_nolg_offset, pair_orbs0_offset, pair_orbs1_offset, pair_out_base, nolg_Nc,
            orbs0buf, orbs1buf, dmf, dml, tmpden, output_count);
    return density_launch<48>(pair_count, spin_count, max_nolg, shared_bytes, pair_NO0, pair_NO1, pair_NOLG,
        pair_h_offset, pair_nolg_offset, pair_orbs0_offset, pair_orbs1_offset, pair_out_base, nolg_Nc,
        orbs0buf, orbs1buf, dmf, dml, tmpden, output_count);
}

extern "C" int Set_Hamiltonian_Cuda_MatrixElements(
    int pair_count,
    int spin_count,
    std::size_t vpot_len,
    double grid_vol,
    int max_no,
    int max_output_count,
    const int *pair_NO0,
    const int *pair_NO1,
    const int *pair_NOLG,
    const int *nolg_MN,
    const int *nolg_Nc,
    const std::size_t *pair_h_offset,
    const std::size_t *pair_nolg_offset,
    const std::size_t *pair_orbs0_offset,
    const std::size_t *pair_orbs1_offset,
    const float *orbs0buf,
    const float *orbs1buf,
    const double *vpotgrid,
    double *hbuf,
    int mode)
{
    int device = 0;
    int max_shared = 0;
    int grid_tile = kPreferredGridTile;

    if (pair_count <= 0 || spin_count <= 0 || max_no <= 0 || max_output_count <= 0) return 0;

    /* cudaGetLastError() reports the last unconsumed error of the whole host
       thread, not of the launch below.  A library that attempted a device
       allocation, failed, and recovered by falling back (e.g. the GEMMul8
       cuBLAS hook under memory pressure) leaves that error latched; pop it
       here so it cannot be misattributed to this kernel. */
    {
        cudaError_t stale = cudaGetLastError();
        if (stale != cudaSuccess) {
            static int stale_reports = 0;
            if (stale_reports < 3) {
                stale_reports++;
                fprintf(stderr,
                        "Set_Hamiltonian_Cuda_MatrixElements: cleared stale CUDA error \"%s\" left by an earlier recovered failure; continuing.\n",
                        cudaGetErrorString(stale));
                fflush(stderr);
            }
        }
    }

    if (cudaGetDevice(&device) != cudaSuccess) return -1;
    if (cudaDeviceGetAttribute(&max_shared, cudaDevAttrMaxSharedMemoryPerBlock, device) != cudaSuccess) return -1;

    auto shared_size = [=](int tile) {
        return static_cast<std::size_t>(tile) * static_cast<std::size_t>(max_no) *
               (static_cast<std::size_t>(spin_count) * (mode == 1 ? sizeof(float) : sizeof(double)) + sizeof(float)) +
               (mode == 2 ? static_cast<std::size_t>(tile) * static_cast<std::size_t>(spin_count) * sizeof(float2) : 0);
    };

    while (1 < grid_tile && static_cast<std::size_t>(max_shared) < shared_size(grid_tile)) grid_tile /= 2;
    if (static_cast<std::size_t>(max_shared) < shared_size(grid_tile)) return 1;

    const dim3 block(kThreads, 1u, 1u);
    const dim3 grid(static_cast<unsigned>(pair_count),
                    static_cast<unsigned>((max_output_count + kThreads - 1) / kThreads), 1u);
    if (mode == 1)
        matrix_elements_kernel<1><<<grid, block, shared_size(grid_tile)>>>(
            pair_count, spin_count, vpot_len, grid_vol, max_no, grid_tile,
            pair_NO0, pair_NO1, pair_NOLG, nolg_MN, nolg_Nc, pair_h_offset, pair_nolg_offset,
            pair_orbs0_offset, pair_orbs1_offset, orbs0buf, orbs1buf, vpotgrid, hbuf);
    else if (mode == 2)
        matrix_elements_kernel<2><<<grid, block, shared_size(grid_tile)>>>(
            pair_count, spin_count, vpot_len, grid_vol, max_no, grid_tile,
            pair_NO0, pair_NO1, pair_NOLG, nolg_MN, nolg_Nc, pair_h_offset, pair_nolg_offset,
            pair_orbs0_offset, pair_orbs1_offset, orbs0buf, orbs1buf, vpotgrid, hbuf);
    else
        matrix_elements_kernel<0><<<grid, block, shared_size(grid_tile)>>>(
            pair_count, spin_count, vpot_len, grid_vol, max_no, grid_tile,
            pair_NO0, pair_NO1, pair_NOLG, nolg_MN, nolg_Nc, pair_h_offset, pair_nolg_offset,
            pair_orbs0_offset, pair_orbs1_offset, orbs0buf, orbs1buf, vpotgrid, hbuf);

    cudaError_t status = cudaGetLastError();
    if (status == cudaSuccess) status = cudaDeviceSynchronize();
    if (status == cudaSuccess) return 0;

    /* The kernel may have run partially, so the device H blocks are suspect
       either way.  Distinguish a transient failure (context still usable ->
       the caller restores the H blocks and redoes this batch with the
       OpenACC kernel) from a sticky, context-poisoning one. */
    (void)cudaGetLastError();
    if (cudaDeviceSynchronize() == cudaSuccess) {
        static int recover_reports = 0;
        if (recover_reports < 3) {
            recover_reports++;
            fprintf(stderr,
                    "Set_Hamiltonian_Cuda_MatrixElements: CUDA kernel failed with \"%s\" but the context is healthy; deferring to the OpenACC kernel for this batch.\n",
                    cudaGetErrorString(status));
            fflush(stderr);
        }
        return 2;
    }
    (void)cudaGetLastError();
    return -static_cast<int>(status);
}

namespace {

/* A phase-local arena keeps streamed batches bounded without entering them
   into OpenACC's persistent present table.  The potential is uploaded once
   and all pair arrays reuse the same allocation for every batch. */
struct StreamWorkspace {
    unsigned char *arena;
    double *potential;
    std::size_t bytes;
};

int stream_status(cudaError_t status)
{
    if (status == cudaSuccess) return 0;
    (void)cudaGetLastError();
    if (cudaDeviceSynchronize() == cudaSuccess) return 2;
    (void)cudaGetLastError();
    return -static_cast<int>(status);
}

template<typename T>
T *stream_upload(StreamWorkspace *work, std::size_t &offset, const T *source,
                 std::size_t count, cudaError_t &status)
{
    if (status != cudaSuccess) return nullptr;
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T) ||
        offset > std::numeric_limits<std::size_t>::max() - 255) {
        status = cudaErrorInvalidValue;
        return nullptr;
    }
    offset = (offset + 255) & ~std::size_t(255);
    const std::size_t bytes = count * sizeof(T);
    if (offset > work->bytes || bytes > work->bytes - offset) {
        status = cudaErrorInvalidValue;
        return nullptr;
    }
    T *target = reinterpret_cast<T *>(work->arena + offset);
    if (bytes) status = cudaMemcpy(target, source, bytes, cudaMemcpyHostToDevice);
    offset += bytes;
    return target;
}

} // namespace

extern "C" void Set_Hamiltonian_Cuda_StreamDestroy(void *opaque)
{
    auto *work = static_cast<StreamWorkspace *>(opaque);
    if (!work) return;
    if (work->potential) cudaFree(work->potential);
    if (work->arena) cudaFree(work->arena);
    std::free(work);
}

extern "C" void *Set_Hamiltonian_Cuda_StreamCreate(std::size_t workspace_bytes,
                                                   std::size_t vpot_count, const double *vpotgrid)
{
    if (!workspace_bytes || vpot_count > std::numeric_limits<std::size_t>::max() / sizeof(double)) return nullptr;
    auto *work = static_cast<StreamWorkspace *>(std::calloc(1, sizeof(StreamWorkspace)));
    if (!work) return nullptr;
    work->bytes = workspace_bytes;
    cudaError_t status = cudaMalloc(reinterpret_cast<void **>(&work->arena), workspace_bytes);
    if (status == cudaSuccess && vpot_count)
        status = cudaMalloc(reinterpret_cast<void **>(&work->potential), vpot_count * sizeof(double));
    if (status == cudaSuccess && vpot_count)
        status = cudaMemcpy(work->potential, vpotgrid, vpot_count * sizeof(double), cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        Set_Hamiltonian_Cuda_StreamDestroy(work);
        (void)cudaGetLastError();
        return nullptr;
    }
    return work;
}

extern "C" int Set_Hamiltonian_Cuda_StreamRun(
    void *opaque, int pair_count, int spin_count, std::size_t vpot_len,
    double grid_vol, int max_no, int max_output_count,
    std::size_t total_h, std::size_t total_nolg, std::size_t total_orbs0, std::size_t total_orbs1,
    const int *pair_NO0, const int *pair_NO1, const int *pair_NOLG,
    const int *nolg_MN, const int *nolg_Nc,
    const std::size_t *pair_h_offset, const std::size_t *pair_nolg_offset,
    const std::size_t *pair_orbs0_offset, const std::size_t *pair_orbs1_offset,
    const float *orbs0buf, const float *orbs1buf, double *hbuf, int mode)
{
    auto *work = static_cast<StreamWorkspace *>(opaque);
    if (!work || pair_count < 0) return 1;
    std::size_t offset = 0;
    cudaError_t status = cudaSuccess;
    const auto *d_NO0 = stream_upload(work, offset, pair_NO0, pair_count, status);
    const auto *d_NO1 = stream_upload(work, offset, pair_NO1, pair_count, status);
    const auto *d_NOLG = stream_upload(work, offset, pair_NOLG, pair_count, status);
    const auto *d_h_off = stream_upload(work, offset, pair_h_offset, pair_count, status);
    const auto *d_nolg_off = stream_upload(work, offset, pair_nolg_offset, pair_count, status);
    const auto *d_orbs0_off = stream_upload(work, offset, pair_orbs0_offset, pair_count, status);
    const auto *d_orbs1_off = stream_upload(work, offset, pair_orbs1_offset, pair_count, status);
    const auto *d_MN = stream_upload(work, offset, nolg_MN, total_nolg, status);
    const auto *d_Nc = stream_upload(work, offset, nolg_Nc, total_nolg, status);
    const auto *d_orbs0 = stream_upload(work, offset, orbs0buf, total_orbs0, status);
    const auto *d_orbs1 = stream_upload(work, offset, orbs1buf, total_orbs1, status);
    auto *d_h = stream_upload(work, offset, hbuf, total_h, status);
    if (status != cudaSuccess) return stream_status(status);
    const int result = Set_Hamiltonian_Cuda_MatrixElements(pair_count, spin_count, vpot_len,
        grid_vol, max_no, max_output_count, d_NO0, d_NO1, d_NOLG, d_MN, d_Nc,
        d_h_off, d_nolg_off, d_orbs0_off, d_orbs1_off, d_orbs0, d_orbs1, work->potential, d_h, mode);
    if (result) return result;
    status = cudaMemcpy(hbuf, d_h, total_h * sizeof(double), cudaMemcpyDeviceToHost);
    return stream_status(status);
}

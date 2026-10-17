#include "openmx_stdout_level.h"
#include "openmx_precision.h"
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

constexpr int kThreads = 256;

/* Each team forms a 16x16 orbital tile.  Cache 32 grid points in LDS and
   apply the potential to the left orbital once, rather than once per
   matrix element.  FP64 accumulation still follows the original grid order.
   In particular, the float orbitals are converted before multiplication. */
__global__ __launch_bounds__(kThreads) void matrix_elements_tiled_kernel(
    int pair_count, std::size_t total_nolg,
    const int *__restrict__ pair_NO0, const int *__restrict__ pair_NO1,
    const int *__restrict__ pair_NOLG,
    const std::size_t *__restrict__ pair_h_offset,
    const std::size_t *__restrict__ pair_nolg_offset,
    const std::size_t *__restrict__ pair_orbs0_offset,
    const std::size_t *__restrict__ pair_orbs1_offset,
    const int *__restrict__ nolg_Nc,
    const double *__restrict__ vpotbuf, const float *__restrict__ orbs0buf,
    const float *__restrict__ orbs1buf, double *__restrict__ hbuf)
{
    const int pair = blockIdx.x;
    const int spin = blockIdx.y;
    if (pair >= pair_count) return;
    const int no0 = pair_NO0[pair], no1 = pair_NO1[pair];
    const int nolg = pair_NOLG[pair];
    const std::size_t hbase = pair_h_offset[pair] +
                            static_cast<std::size_t>(spin) * no0 * no1;
    const double *v = vpotbuf + static_cast<std::size_t>(spin) * total_nolg + pair_nolg_offset[pair];
    /* orbs0 holds the central atom's orbitals once per point of its sphere:
       the pair's overlap points select theirs through nolg_Nc */
    const float *a = orbs0buf + pair_orbs0_offset[pair];
    const int *nc = nolg_Nc + pair_nolg_offset[pair];
    const float *b = orbs1buf + pair_orbs1_offset[pair];
    const int lane = threadIdx.x;
    const int row = lane / 16, col = lane % 16;
    __shared__ double left[32][16];
    __shared__ double right[32][16];

    for (int i0 = 0; i0 < no0; i0 += 16) {
        for (int j0 = 0; j0 < no1; j0 += 16) {
            const bool active = i0 + row < no0 && j0 + col < no1;
            const std::size_t hidx = hbase + static_cast<std::size_t>(i0 + row) * no1 + j0 + col;
            double sum = active ? hbuf[hidx] : 0.0;
            for (int g0 = 0; g0 < nolg; g0 += 32) {
                const int count = nolg - g0 < 32 ? nolg - g0 : 32;
                for (int t = lane; t < 32 * 16; t += kThreads) {
                    const int g = t / 16, o = t % 16;
                    left[g][o] = (g < count && i0 + o < no0)
                        ? v[g0 + g] * static_cast<double>(a[static_cast<std::size_t>(nc[g0 + g]) * no0 + i0 + o]) : 0.0;
                    right[g][o] = (g < count && j0 + o < no1)
                        ? static_cast<double>(b[static_cast<std::size_t>(g0 + g) * no1 + j0 + o]) : 0.0;
                }
                __syncthreads();
                if (active) {
                    for (int g = 0; g < count; ++g)
                        sum += left[g][row] * right[g][col];
                }
                __syncthreads();
            }
            if (active) hbuf[hidx] = sum;
        }
    }
}

/* ---------------------------------------------------------------------
   Double-float grid kernels (the NVIDIA version's MODE 2 kernels,
   matrix_elements_kernel_df2 and density_kernel<NMAX>): FP64-grade sums
   from FP32 arithmetic, for GPUs whose FP64 rate is a small fraction of
   their FP32 rate (gfx1200: one FP64 instruction costs about 32 FP32
   ones).  The weighted orbital (or the density matrix element) is split
   into hi + lo floats; each product with an FP32 orbital is formed exactly
   (TwoProd through FMA), the hi parts are summed with an exact TwoSum
   compensation and the lo parts plainly, and the partial sums go into
   FP64 once per tile of grid points.  The error per tile is of order
   tile x eps32^2 of the sum of the magnitudes.  The error-free
   transformations need every product and sum rounded as written, so FP
   contraction is off in these functions and the wanted FMAs are explicit.
   gfx1200 has no FP64 atomic add (a compare-and-swap loop): the matrix
   element kernel has one block per output, the on-the-fly density kernel
   one atomic per point and spin, and the local mode's CSR density kernel
   (Set_Density_Grid_GPU.c) is a gather without atomics.
   OPENMX_GRID_PRECISION selects the kernels (grid_precision_configure);
   the default is double-float where the FP32 rate is at least 8 times
   the FP64 rate (openmx_gpu_fp32_fp64_ratio).
   --------------------------------------------------------------------- */
__device__ __forceinline__ void df_acc(float &s, float &c, float t)
{
#pragma clang fp contract(off)
    const float u = s + t;
    const float bv = u - s;
    const float av = u - bv;
    c += (s - av) + (t - bv);
    s = u;
}

__device__ __forceinline__ void df_term(float2 w, float t, float &ts, float &tc, float &qs)
{
#pragma clang fp contract(off)
    const float p = w.x * t;
    const float e = __builtin_fmaf(w.x, t, -p);
    const float q = __builtin_fmaf(w.y, t, e);
    df_acc(ts, tc, p);
    qs += q;
}

constexpr int kDfGridTile = 64;
constexpr int kDfFlushPoints = 128;

/* One block per (pair, y-block) for every spin: a thread owns the 2x2
   elements (spin, i0..i0+1, j0..j0+1) of its tile (an odd NO0 or NO1 gives
   edge tiles whose second row or column repeats the first and is not
   stored), so the block's outputs are its own and no atomics are needed.
   Per tile of grid points the block stages the potentials (hi, lo) of
   every spin, gathers each central orbital phi0 once and stores the
   weighted w = v phi0 of every spin as (hi, lo), and stages the neighbour
   orbitals phi1, all in LDS.  The FP32 partial sums go into FP64 every
   kDfFlushPoints points (an error bound of order 128 eps32^2 of the sum of
   the magnitudes, against 1e4 eps64 for a plain FP64 sum over the
   pair's points). */
__global__ __launch_bounds__(kThreads) void matrix_elements_df_kernel(
    int pair_count, int spin_count, int yblocks, int max_no, int grid_tile, int chunk, std::size_t total_nolg,
    std::size_t total_h, double *__restrict__ part,
    const int *__restrict__ pair_NO0, const int *__restrict__ pair_NO1, const int *__restrict__ pair_NOLG,
    const std::size_t *__restrict__ pair_h_offset, const std::size_t *__restrict__ pair_nolg_offset,
    const std::size_t *__restrict__ pair_orbs0_offset, const std::size_t *__restrict__ pair_orbs1_offset,
    const int *__restrict__ nolg_Nc, const double *__restrict__ vpotbuf, const float *__restrict__ orbs0buf,
    const float *__restrict__ orbs1buf, double *__restrict__ hbuf)
{
#pragma clang fp contract(off)
    extern __shared__ float2 df_shared[];
    const int pair = static_cast<int>(blockIdx.x);
    const int yb = static_cast<int>(blockIdx.y);
    if (pair >= pair_count) return;
    const int no0 = pair_NO0[pair], no1 = pair_NO1[pair], nolg = pair_NOLG[pair];
    const int tj = (no1 + 1) >> 1;
    const int per_spin = ((no0 + 1) >> 1) * tj;
    const int tiles = spin_count * per_spin;
    const int first = yb * static_cast<int>(blockDim.x);
    if (first >= tiles) return;   /* uniform over the block */
    (void)yblocks;
    /* the block's share of the pair's points: chunk z of gridDim.z (the
       partial sums of a split go to part[z] for the fixed-order reduction;
       a block past the pair's points stores zeros there) */
    const int point_begin = static_cast<int>(blockIdx.z) * chunk;
    const int point_end = (point_begin + chunk < nolg) ? point_begin + chunk : nolg;

    const int q = first + static_cast<int>(threadIdx.x);
    const bool active = q < tiles;
    int spin = 0, i0 = 0, i1 = 0, j0 = 0, j1 = 0;
    bool has_i1 = false, has_j1 = false;
    if (active) {
        spin = q / per_spin;
        const int r = q - spin * per_spin;
        const int ii = r / tj;
        i0 = 2 * ii;
        j0 = 2 * (r - ii * tj);
        has_i1 = i0 + 1 < no0;
        has_j1 = j0 + 1 < no1;
        i1 = has_i1 ? i0 + 1 : i0;
        j1 = has_j1 ? j0 + 1 : j0;
    }
    const std::size_t noff = pair_nolg_offset[pair];
    const float *a = orbs0buf + pair_orbs0_offset[pair];
    const int *nc = nolg_Nc + noff;
    const float *b = orbs1buf + pair_orbs1_offset[pair];
    const std::size_t wstride = static_cast<std::size_t>(grid_tile) * max_no;
    float2 *vt = df_shared;                                     /* spin_count x grid_tile */
    float2 *w0 = vt + static_cast<std::size_t>(spin_count) * grid_tile;   /* spin_count x grid_tile x max_no */
    float *t1 = reinterpret_cast<float *>(w0 + static_cast<std::size_t>(spin_count) * wstride);  /* grid_tile x max_no */
    const float2 *wrow = w0 + static_cast<std::size_t>(spin) * wstride;
    double s00 = 0.0, s01 = 0.0, s10 = 0.0, s11 = 0.0;
    float ts00 = 0.0f, tc00 = 0.0f, qs00 = 0.0f, ts01 = 0.0f, tc01 = 0.0f, qs01 = 0.0f;
    float ts10 = 0.0f, tc10 = 0.0f, qs10 = 0.0f, ts11 = 0.0f, tc11 = 0.0f, qs11 = 0.0f;
    int pending = 0;

    for (int base = point_begin; base < point_end; base += grid_tile) {
        const int count = (point_end - base < grid_tile) ? point_end - base : grid_tile;

        for (int idx = static_cast<int>(threadIdx.x); idx < spin_count * count; idx += static_cast<int>(blockDim.x)) {
            const int sp = idx / count, g = idx - sp * count;
            const double vv = vpotbuf[static_cast<std::size_t>(sp) * total_nolg + noff + base + g];
            const float hi = static_cast<float>(vv);
            vt[sp * grid_tile + g] = make_float2(hi, static_cast<float>(vv - static_cast<double>(hi)));
        }
        __syncthreads();
        for (int idx = static_cast<int>(threadIdx.x); idx < count * no0; idx += static_cast<int>(blockDim.x)) {
            const int g = idx / no0, o = idx - g * no0;
            const float p0 = a[static_cast<std::size_t>(nc[base + g]) * no0 + o];
            for (int sp = 0; sp < spin_count; ++sp) {
                const float2 vv = vt[sp * grid_tile + g];
                const float hi = vv.x * p0;
                const float lo = __builtin_fmaf(vv.x, p0, -hi) + vv.y * p0;
                w0[sp * wstride + g * max_no + o] = make_float2(hi, lo);
            }
        }
        for (int idx = static_cast<int>(threadIdx.x); idx < count * no1; idx += static_cast<int>(blockDim.x)) {
            const int g = idx / no1, o = idx - g * no1;
            t1[g * max_no + o] = b[static_cast<std::size_t>(base + g) * no1 + o];
        }
        __syncthreads();
        if (active) {
            for (int g = 0; g < count; ++g) {
                const float2 wa = wrow[g * max_no + i0], wb = wrow[g * max_no + i1];
                const float ta = t1[g * max_no + j0], tb = t1[g * max_no + j1];
                df_term(wa, ta, ts00, tc00, qs00);
                df_term(wa, tb, ts01, tc01, qs01);
                df_term(wb, ta, ts10, tc10, qs10);
                df_term(wb, tb, ts11, tc11, qs11);
            }
            pending += count;
            if (kDfFlushPoints <= pending || point_end <= base + grid_tile) {
                s00 += static_cast<double>(ts00) + static_cast<double>(tc00 + qs00);
                s01 += static_cast<double>(ts01) + static_cast<double>(tc01 + qs01);
                s10 += static_cast<double>(ts10) + static_cast<double>(tc10 + qs10);
                s11 += static_cast<double>(ts11) + static_cast<double>(tc11 + qs11);
                ts00 = tc00 = qs00 = ts01 = tc01 = qs01 = 0.0f;
                ts10 = tc10 = qs10 = ts11 = tc11 = qs11 = 0.0f;
                pending = 0;
            }
        }
        __syncthreads();
    }
    if (active) {
        const std::size_t hb = pair_h_offset[pair] + static_cast<std::size_t>(spin) * no0 * no1;
        if (part == nullptr) {
            double *h = hbuf + hb;
            h[i0 * no1 + j0] += s00;
            if (has_j1) h[i0 * no1 + j1] += s01;
            if (has_i1) h[i1 * no1 + j0] += s10;
            if (has_i1 && has_j1) h[i1 * no1 + j1] += s11;
        }
        else {
            double *h = part + static_cast<std::size_t>(blockIdx.z) * total_h + hb;
            h[i0 * no1 + j0] = s00;
            if (has_j1) h[i0 * no1 + j1] = s01;
            if (has_i1) h[i1 * no1 + j0] = s10;
            if (has_i1 && has_j1) h[i1 * no1 + j1] = s11;
        }
    }
}

/* H += the partial sums of the z blocks, in the order of z */
__global__ void matrix_elements_df_reduce(std::size_t total_h, int zcount, const double *__restrict__ part,
                                          double *__restrict__ hbuf)
{
    for (std::size_t e = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; e < total_h;
         e += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        double sum = 0.0;
        for (int z = 0; z < zcount; ++z) sum += part[static_cast<std::size_t>(z) * total_h + e];
        hbuf[e] += sum;
    }
}

extern "C" int openmx_gpu_fp32_fp64_ratio(void);

/* The arithmetic of the grid integrals' kernels, as the NVIDIA version's
   grid_precision.c decides it (without its lossy FP32 stage): 1 for the
   double-float kernels, 0 for the FP64 ones.  OPENMX_GRID_PRECISION=df, DF
   or 2 selects double-float and any other non-empty value FP64 (fp64,
   FP64, double and 0 as documented, anything else noted once by
   openmx_grid_precision_announce); unset, double-float where the FP32 rate
   is at least 8 times the FP64 rate (openmx_gpu_fp32_fp64_ratio: the
   gcnArchName table, or OPENMX_GPU_FP32_FP64_RATIO; a failed device query
   gives 0, so FP64).  Decided once per process and silently, since every
   rank asks. */
int grid_precision_mode = -1;
char grid_precision_unrecognized[64] = "";

int grid_precision_configure()
{
    /* Keep the cached device/environment preference for subsequent inputs. */
    if (!openmx_fp32_enabled()) return 0;
    if (grid_precision_mode < 0) {
        const char *value = std::getenv("OPENMX_GRID_PRECISION");
        if (value != nullptr && value[0] != '\0') {
            if (std::strcmp(value, "df") == 0 || std::strcmp(value, "DF") == 0 || std::strcmp(value, "2") == 0) {
                grid_precision_mode = 1;
            }
            else {
                grid_precision_mode = 0;
                if (!(std::strcmp(value, "fp64") == 0 || std::strcmp(value, "FP64") == 0 ||
                      std::strcmp(value, "double") == 0 || std::strcmp(value, "0") == 0))
                    std::snprintf(grid_precision_unrecognized, sizeof(grid_precision_unrecognized), "%s", value);
            }
        }
        else {
            grid_precision_mode = (8 <= openmx_gpu_fp32_fp64_ratio()) ? 1 : 0;
        }
    }
    return grid_precision_mode;
}

bool grid_precision_df()
{
    return grid_precision_configure() == 1;
}

int device_shared_limit()
{
    static int limit = -1;

    if (limit < 0) {
        int device = 0;
        hipDeviceProp_t prop;
        if (hipGetDevice(&device) == hipSuccess && hipGetDeviceProperties(&prop, device) == hipSuccess)
            limit = static_cast<int>(prop.sharedMemPerBlock);
        else {
            (void)hipGetLastError();
            limit = 0;
        }
    }
    return limit;
}

bool use_tiled_kernel()
{
    /* Keep the original kernel available for reproducible GPU A/B runs. */
    const char *value = std::getenv("OPENMX_SETHAM_HIP_KERNEL");
    return value == nullptr || std::strcmp(value, "scalar") != 0;
}

__global__ void matrix_elements_kernel(
    int pair_count, int spin_count, std::size_t total_nolg,
    const int *pair_NO0, const int *pair_NO1, const int *pair_NOLG,
    const std::size_t *pair_h_offset, const std::size_t *pair_nolg_offset,
    const std::size_t *pair_orbs0_offset, const std::size_t *pair_orbs1_offset,
    const int *nolg_Nc,
    const double *vpotbuf, const float *orbs0buf, const float *orbs1buf,
    double *hbuf)
{
    const int pair = static_cast<int>(blockIdx.x);
    if (pair >= pair_count) return;

    const int no0 = pair_NO0[pair];
    const int no1 = pair_NO1[pair];
    const int nolg = pair_NOLG[pair];
    const std::size_t mat_size = static_cast<std::size_t>(no0) * no1;
    const std::size_t count = static_cast<std::size_t>(spin_count) * mat_size;
    const std::size_t e = static_cast<std::size_t>(blockIdx.y) * blockDim.x + threadIdx.x;
    if (e >= count) return;

    const int spin = static_cast<int>(e / mat_size);
    const std::size_t ij = e - static_cast<std::size_t>(spin) * mat_size;
    const int i = static_cast<int>(ij / no1);
    const int j = static_cast<int>(ij - static_cast<std::size_t>(i) * no1);
    const std::size_t h_off = pair_h_offset[pair];
    const std::size_t nolg_off = pair_nolg_offset[pair];
    const std::size_t orbs0_off = pair_orbs0_offset[pair];
    const std::size_t orbs1_off = pair_orbs1_offset[pair];
    double sum = hbuf[h_off + e];

    for (int grid = 0; grid < nolg; ++grid) {
        sum += vpotbuf[static_cast<std::size_t>(spin) * total_nolg + nolg_off + grid] *
               static_cast<double>(orbs0buf[orbs0_off + static_cast<std::size_t>(nolg_Nc[nolg_off + grid]) * no0 + i]) *
               static_cast<double>(orbs1buf[orbs1_off + static_cast<std::size_t>(grid) * no1 + j]);
    }
    hbuf[h_off + e] = sum;
}

template <typename T>
bool device_alloc_copy(T **dst, const T *src, std::size_t count)
{
    if (hipMalloc(reinterpret_cast<void **>(dst), sizeof(T) * count) != hipSuccess) return false;
    if (hipMemcpy(*dst, src, sizeof(T) * count, hipMemcpyHostToDevice) != hipSuccess) return false;
    return true;
}

int launch_matrix_elements(int pair_count, int spin_count, int max_output_count, int max_no, int max_nolg,
                           std::size_t total_h, std::size_t total_nolg,
                           const int *d_no0, const int *d_no1, const int *d_nolg,
                           const std::size_t *d_hoff, const std::size_t *d_noff,
                           const std::size_t *d_o0off, const std::size_t *d_o1off, const int *d_nc,
                           const double *d_vpot, const float *d_o0, const float *d_o1, double *d_h)
{
    const dim3 block(kThreads);
    const dim3 grid(static_cast<unsigned>(pair_count),
                    static_cast<unsigned>((max_output_count + kThreads - 1) / kThreads));
    const bool tiled = use_tiled_kernel();
    const char *trace = std::getenv("OPENMX_SETHAM_TIMING");
    const bool profile = trace != nullptr && std::atoi(trace) > 1;
    std::chrono::steady_clock::time_point start;
    if (profile) start = std::chrono::steady_clock::now();
    if (tiled && 0 < max_no && grid_precision_df()) {
        /* the block: the 2x2 tiles of every spin of the largest pair, whole
           waves; the grid tile: LDS of at most 32 KiB, two blocks per CU */
        const int half = (max_no + 1) / 2;
        const int tiles = spin_count * half * half;
        const int threads = tiles < kThreads ? ((tiles + 31) / 32) * 32 : kThreads;
        const int yblocks = (tiles + threads - 1) / threads;
        int grid_tile = kDfGridTile;
        std::size_t shared = 0;

        for (;;) {
            shared = static_cast<std::size_t>(grid_tile) *
                     (static_cast<std::size_t>(spin_count) * (sizeof(float2) + static_cast<std::size_t>(max_no) *
                                                              sizeof(float2)) +
                      static_cast<std::size_t>(max_no) * sizeof(float));
            if (shared <= 32768 || grid_tile <= 8) break;
            grid_tile /= 2;
        }
        if (static_cast<int>(shared) <= device_shared_limit()) {
            /* the points of a pair split over at most 32 blocks of at least
               2048 points (the pair of an atom with itself holds its whole
               sphere), the partial sums at most 64 MiB */
            int zcount = 1, chunk = (0 < max_nolg) ? max_nolg : 1;
            double *part = nullptr;

            if (0 < max_nolg && 0 < total_h) {
                std::size_t zmem = (static_cast<std::size_t>(64) << 20) / (sizeof(double) * total_h);
                zcount = (max_nolg + 2047) / 2048;
                if (zcount > 32) zcount = 32;
                if (static_cast<std::size_t>(zcount) > zmem) zcount = static_cast<int>(zmem);
                if (zcount < 1) zcount = 1;
                chunk = (max_nolg + zcount - 1) / zcount;
                chunk = ((chunk + grid_tile - 1) / grid_tile) * grid_tile;
                if (1 < zcount && hipMalloc(reinterpret_cast<void **>(&part),
                                            sizeof(double) * total_h * static_cast<std::size_t>(zcount)) != hipSuccess) {
                    (void)hipGetLastError();
                    part = nullptr;
                    zcount = 1;
                    chunk = max_nolg;
                }
            }
            hipLaunchKernelGGL(matrix_elements_df_kernel,
                               dim3(static_cast<unsigned>(pair_count), static_cast<unsigned>(yblocks),
                                    static_cast<unsigned>(zcount)),
                               dim3(static_cast<unsigned>(threads)), shared, 0, pair_count, spin_count, yblocks, max_no,
                               grid_tile, chunk, total_nolg, total_h, part, d_no0, d_no1, d_nolg, d_hoff, d_noff,
                               d_o0off, d_o1off, d_nc, d_vpot, d_o0, d_o1, d_h);
            {
                /* the launch status is read once: hipGetLastError clears it */
                hipError_t status = hipGetLastError();

                if (status == hipSuccess && part != nullptr) {
                    const std::size_t blocks = (total_h + kThreads - 1) / kThreads;
                    hipLaunchKernelGGL(matrix_elements_df_reduce,
                                       dim3(static_cast<unsigned>(blocks < 4096 ? blocks : 4096)), dim3(kThreads), 0,
                                       0, total_h, zcount, part, d_h);
                    status = hipGetLastError();
                }
                if (status == hipSuccess) status = hipDeviceSynchronize();
                if (part != nullptr) hipFree(part);
                if (status != hipSuccess) {
                    (void)hipGetLastError();
                    return 2;
                }
            }
            if (profile) {
                const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
                std::fprintf(stdout, "SETHAM_HIP kernel=df pairs=%d spins=%d max_no=%d split=%d time=%.6f s\n",
                             pair_count, spin_count, max_no, zcount, seconds);
                std::fflush(stdout);
            }
            return 0;
        }
    }
    if (tiled) {
        hipLaunchKernelGGL(matrix_elements_tiled_kernel,
                           dim3(static_cast<unsigned>(pair_count), static_cast<unsigned>(spin_count)),
                           block, 0, 0, pair_count, total_nolg, d_no0, d_no1, d_nolg,
                           d_hoff, d_noff, d_o0off, d_o1off, d_nc, d_vpot, d_o0, d_o1, d_h);
    } else {
        hipLaunchKernelGGL(matrix_elements_kernel, grid, block, 0, 0,
                           pair_count, spin_count, total_nolg, d_no0, d_no1, d_nolg,
                           d_hoff, d_noff, d_o0off, d_o1off, d_nc, d_vpot, d_o0, d_o1, d_h);
    }
    if (hipGetLastError() != hipSuccess || hipDeviceSynchronize() != hipSuccess) return 2;
    if (profile) {
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        std::fprintf(stdout, "SETHAM_HIP kernel=%s pairs=%d spins=%d time=%.6f s\n",
                     tiled ? "tiled" : "scalar", pair_count, spin_count, seconds);
        std::fflush(stdout);
    }
    return 0;
}

} // namespace

extern "C" int Set_Hamiltonian_Hip_MatrixElements(
    int pair_count, int spin_count, int max_output_count, int max_no, int max_nolg,
    std::size_t total_h, std::size_t total_nolg, std::size_t total_orbs0, std::size_t total_orbs1,
    const int *pair_NO0, const int *pair_NO1, const int *pair_NOLG,
    const std::size_t *pair_h_offset, const std::size_t *pair_nolg_offset,
    const std::size_t *pair_orbs0_offset, const std::size_t *pair_orbs1_offset, const int *nolg_Nc,
    const double *vpotbuf, const float *orbs0buf, const float *orbs1buf, double *hbuf)
{
    int *d_no0 = nullptr, *d_no1 = nullptr, *d_nolg = nullptr, *d_nc = nullptr;
    std::size_t *d_hoff = nullptr, *d_noff = nullptr, *d_o0off = nullptr, *d_o1off = nullptr;
    double *d_vpot = nullptr, *d_h = nullptr;
    float *d_o0 = nullptr, *d_o1 = nullptr;
    int result = 0;

    if (pair_count <= 0 || spin_count <= 0 || max_output_count <= 0) return 0;
    (void)hipGetLastError();

#define COPY_DEVICE(dst, src, count) do { if (!device_alloc_copy(&(dst), (src), (count))) { result = 1; goto cleanup; } } while (0)
    COPY_DEVICE(d_no0, pair_NO0, pair_count);
    COPY_DEVICE(d_no1, pair_NO1, pair_count);
    COPY_DEVICE(d_nolg, pair_NOLG, pair_count);
    COPY_DEVICE(d_hoff, pair_h_offset, pair_count);
    COPY_DEVICE(d_noff, pair_nolg_offset, pair_count);
    COPY_DEVICE(d_o0off, pair_orbs0_offset, pair_count);
    COPY_DEVICE(d_o1off, pair_orbs1_offset, pair_count);
    COPY_DEVICE(d_nc, nolg_Nc, total_nolg);
    COPY_DEVICE(d_vpot, vpotbuf, static_cast<std::size_t>(spin_count) * total_nolg);
    COPY_DEVICE(d_o0, orbs0buf, total_orbs0);
    COPY_DEVICE(d_o1, orbs1buf, total_orbs1);
    COPY_DEVICE(d_h, hbuf, total_h);
#undef COPY_DEVICE

    result = launch_matrix_elements(pair_count, spin_count, max_output_count, max_no, max_nolg, total_h, total_nolg,
                                    d_no0, d_no1, d_nolg, d_hoff, d_noff, d_o0off, d_o1off, d_nc,
                                    d_vpot, d_o0, d_o1, d_h);
    if (result != 0) goto cleanup;
    if (hipMemcpy(hbuf, d_h, sizeof(double) * total_h, hipMemcpyDeviceToHost) != hipSuccess) result = 2;

cleanup:
    hipFree(d_h); hipFree(d_o1); hipFree(d_o0); hipFree(d_vpot); hipFree(d_nc);
    hipFree(d_o1off); hipFree(d_o0off); hipFree(d_noff); hipFree(d_hoff);
    hipFree(d_nolg); hipFree(d_no1); hipFree(d_no0);
    (void)hipGetLastError();
    return result;
}

/* The same kernel on the tables Set_Hamiltonian keeps resident on the device
   between SCF iterations (device pointers of the associated copies): only
   the potential and the output travel. */
extern "C" int Set_Hamiltonian_Hip_MatrixElements_Resident(
    int pair_count, int spin_count, int max_output_count, int max_no, int max_nolg,
    std::size_t total_h, std::size_t total_nolg,
    const int *d_no0, const int *d_no1, const int *d_nolg,
    const std::size_t *d_hoff, const std::size_t *d_noff,
    const std::size_t *d_o0off, const std::size_t *d_o1off, const int *d_nc,
    const float *d_o0, const float *d_o1,
    const double *vpotbuf, double *hbuf)
{
    double *d_vpot = nullptr, *d_h = nullptr;
    int result = 0;

    if (pair_count <= 0 || spin_count <= 0 || max_output_count <= 0) return 0;
    if (d_no0 == nullptr || d_no1 == nullptr || d_nolg == nullptr || d_hoff == nullptr ||
        d_noff == nullptr || d_o0off == nullptr || d_o1off == nullptr || d_nc == nullptr ||
        d_o0 == nullptr || d_o1 == nullptr) {
        return 1;
    }
    (void)hipGetLastError();

    if (!device_alloc_copy(&d_vpot, vpotbuf, static_cast<std::size_t>(spin_count) * total_nolg) ||
        !device_alloc_copy(&d_h, hbuf, total_h)) {
        result = 1;
        goto cleanup;
    }
    result = launch_matrix_elements(pair_count, spin_count, max_output_count, max_no, max_nolg, total_h, total_nolg,
                                    d_no0, d_no1, d_nolg, d_hoff, d_noff, d_o0off, d_o1off, d_nc,
                                    d_vpot, d_o0, d_o1, d_h);
    if (result != 0) goto cleanup;
    if (hipMemcpy(hbuf, d_h, sizeof(double) * total_h, hipMemcpyDeviceToHost) != hipSuccess) result = 2;

cleanup:
    hipFree(d_h); hipFree(d_vpot);
    (void)hipGetLastError();
    return result;
}

namespace {

/* the potential at the overlap points of a batch of pairs, from the rank's
   grid (vgrid, spin_count x vgrid_len): vb[s * batch_nolg + noff[p] + k] =
   gridvol * vgrid[s][mgla[sph[p] + nc[noff[p] + k]]] */
__global__ void otf_vpot_gather_kernel(int pair_count, int spin_count, std::size_t batch_nolg,
                                       const int *__restrict__ nolg, const std::size_t *__restrict__ noff,
                                       const std::size_t *__restrict__ sph, const int *__restrict__ nc,
                                       const int *__restrict__ mgla, const double *__restrict__ vgrid,
                                       std::size_t vgrid_len, double gridvol, double *__restrict__ vb)
{
    const int p = blockIdx.x;
    if (p >= pair_count) return;
    const int np = nolg[p];
    const std::size_t off = noff[p], so = sph[p];

    for (int k = blockIdx.y * blockDim.x + threadIdx.x; k < np; k += gridDim.y * blockDim.x) {
        const int MN = mgla[so + static_cast<std::size_t>(nc[off + k])];
        for (int s = 0; s < spin_count; ++s)
            vb[static_cast<std::size_t>(s) * batch_nolg + off + k] =
                gridvol * vgrid[static_cast<std::size_t>(s) * vgrid_len + static_cast<std::size_t>(MN)];
    }
}

} // namespace

/* The on-the-fly batches of Set_Hamiltonian (Set_Hamiltonian.c): the
   potential gathered at the batch's overlap points ... */
extern "C" int Set_Hamiltonian_Hip_OTF_VpotGather(int pair_count, int max_nolg, int spin_count,
                                                  std::size_t batch_nolg, const int *d_nolg,
                                                  const std::size_t *d_noff, const std::size_t *d_sph,
                                                  const int *d_nc, const int *d_mgla, const double *d_vgrid,
                                                  std::size_t vgrid_len, double gridvol, double *d_vb)
{
    unsigned ychunks;

    if (pair_count <= 0 || max_nolg <= 0) return 0;
    ychunks = static_cast<unsigned>((max_nolg + kThreads - 1) / kThreads);
    if (ychunks > 8u) ychunks = 8u;
    hipLaunchKernelGGL(otf_vpot_gather_kernel, dim3(static_cast<unsigned>(pair_count), ychunks), dim3(kThreads), 0, 0,
                       pair_count, spin_count, batch_nolg, d_nolg, d_noff, d_sph, d_nc, d_mgla, d_vgrid, vgrid_len,
                       gridvol, d_vb);
    return hipGetLastError() == hipSuccess ? 0 : 2;
}

/* ... and the matrix elements of the batch, every operand on the device
   (d_h holds the batch's H blocks on entry and on return) */
extern "C" int Set_Hamiltonian_Hip_MatrixElements_Device(
    int pair_count, int spin_count, int max_output_count, int max_no, int max_nolg, std::size_t total_h,
    std::size_t total_nolg,
    const int *d_no0, const int *d_no1, const int *d_nolg,
    const std::size_t *d_hoff, const std::size_t *d_noff,
    const std::size_t *d_o0off, const std::size_t *d_o1off, const int *d_nc,
    const double *d_vpot, const float *d_o0, const float *d_o1, double *d_h)
{
    if (pair_count <= 0 || spin_count <= 0 || max_output_count <= 0) return 0;
    return launch_matrix_elements(pair_count, spin_count, max_output_count, max_no, max_nolg, total_h, total_nolg,
                                  d_no0, d_no1, d_nolg, d_hoff, d_noff, d_o0off, d_o1off, d_nc, d_vpot, d_o0, d_o1,
                                  d_h);
}

namespace {

/* the on-the-fly density of a batch: every overlap point k of pair p adds
   sum_ij DM[s][i][j] phi0_i(Nc) phi1_j(k) to its central atom's sphere
   point Nc (den[s * den_stride + denoff[p] + Nc]); the pair's density
   matrix blocks are staged in LDS */
constexpr int kDenMaxDm = 4096;   /* doubles of one pair's DM blocks in LDS (spin x NO0 x NO1) */

__global__ __launch_bounds__(kThreads) void otf_density_kernel(
    int pair_count, int spin_count, const int *__restrict__ no0, const int *__restrict__ no1,
    const int *__restrict__ nolg, const std::size_t *__restrict__ dmoff, const std::size_t *__restrict__ noff,
    const std::size_t *__restrict__ o0off, const std::size_t *__restrict__ o1off,
    const std::size_t *__restrict__ denoff, const int *__restrict__ nc, const double *__restrict__ dm,
    const float *__restrict__ o0, const float *__restrict__ o1, std::size_t den_stride, double *__restrict__ den)
{
    __shared__ double sdm[kDenMaxDm];
    const int p = blockIdx.x;
    if (p >= pair_count) return;
    const int n0 = no0[p], n1 = no1[p], np = nolg[p];
    const int mat = n0 * n1;
    const int dm_count = spin_count * mat;
    const double *dmp = dm + dmoff[p];
    const bool staged = dm_count <= kDenMaxDm;

    if (staged) {
        for (int e = threadIdx.x; e < dm_count; e += blockDim.x) sdm[e] = dmp[e];
        __syncthreads();
    }
    for (int k = blockIdx.y * blockDim.x + threadIdx.x; k < np; k += gridDim.y * blockDim.x) {
        const int Nc = nc[noff[p] + k];
        const float *phi0 = o0 + o0off[p] + static_cast<std::size_t>(Nc) * n0;
        const float *phi1 = o1 + o1off[p] + static_cast<std::size_t>(k) * n1;

        for (int s = 0; s < spin_count; ++s) {
            const double *d = staged ? sdm + s * mat : dmp + static_cast<std::size_t>(s) * mat;
            double acc = 0.0;

            for (int i = 0; i < n0; ++i) {
                double t = 0.0;
                for (int j = 0; j < n1; ++j) t += d[i * n1 + j] * static_cast<double>(phi1[j]);
                acc += static_cast<double>(phi0[i]) * t;
            }
            atomicAdd(den + static_cast<std::size_t>(s) * den_stride + denoff[p] + static_cast<std::size_t>(Nc), acc);
        }
    }
}

/* The double-float density (the NVIDIA version's density_kernel<NMAX>):
   one thread per overlap point, the pair's density matrix blocks staged
   once per block in LDS as (hi, lo) floats, the point's neighbour orbitals
   phi1 in registers (NMAX, unrolled with a break at NO1 so that every
   index is static) and its central orbitals read once per row.  Per point
   and spin: rho += sum_i phi0[i] (sum_j DM[i][j] phi1[j]), each product
   exact (TwoProd), the hi parts of a row summed with TwoSum compensation
   and the lo parts plainly, the rows accumulated as a float pair, and the
   pair added to the FP64 density by one atomic add.  SPLIT gives every
   spin its own blocks (blockIdx.z), each staging only that spin's block:
   1/S of the LDS, for the pairs whose S x NO0 x NO1 (hi, lo) floats do not
   fit (non-collinear with 46 or more orbitals), and NMAX = 0 reads the
   neighbour orbitals from memory for more than 64 orbitals; the
   arithmetic per point and spin is the same in every variant (the role
   of the NVIDIA version's OpenACC double-float fallback, which reads a
   host-split DM from memory). */
constexpr int kDenDfThreads = 128;

template <int NMAX, bool SPLIT>
__global__ __launch_bounds__(kDenDfThreads) void otf_density_df_kernel(
    int spin_count, const int *__restrict__ no0, const int *__restrict__ no1, const int *__restrict__ nolg,
    const std::size_t *__restrict__ dmoff, const std::size_t *__restrict__ noff,
    const std::size_t *__restrict__ o0off, const std::size_t *__restrict__ o1off,
    const std::size_t *__restrict__ denoff, const int *__restrict__ nc, const double *__restrict__ dm,
    const float *__restrict__ o0, const float *__restrict__ o1, std::size_t den_stride, double *__restrict__ den)
{
#pragma clang fp contract(off)
    extern __shared__ float2 dms[];
    const int p = static_cast<int>(blockIdx.x);
    const int n0 = no0[p], n1 = no1[p], np = nolg[p];
    if (static_cast<int>(blockIdx.y * blockDim.x) >= np) return;   /* uniform over the block */
    /* the block's spins: all of them, or the one of blockIdx.z */
    const int s_first = SPLIT ? static_cast<int>(blockIdx.z) : 0;
    const int s_count = SPLIT ? 1 : spin_count;
    const int mat = n0 * n1;
    const int cnt = s_count * mat;
    const double *dp = dm + dmoff[p] + static_cast<std::size_t>(s_first) * static_cast<std::size_t>(mat);

    for (int e = static_cast<int>(threadIdx.x); e < cnt; e += static_cast<int>(blockDim.x)) {
        const double x = dp[e];
        const float hi = static_cast<float>(x);
        dms[e] = make_float2(hi, static_cast<float>(x - static_cast<double>(hi)));
    }
    __syncthreads();
    const int k = static_cast<int>(blockIdx.y * blockDim.x + threadIdx.x);
    if (k >= np) return;
    const int Nc = nc[noff[p] + static_cast<std::size_t>(k)];
    const float *phi1 = o1 + o1off[p] + static_cast<std::size_t>(k) * n1;
    const float *phi0 = o0 + o0off[p] + static_cast<std::size_t>(Nc) * n0;
    float f1[NMAX > 0 ? NMAX : 1];
    if constexpr (0 < NMAX) {
#pragma unroll
        for (int j = 0; j < NMAX; ++j) f1[j] = (j < n1) ? phi1[j] : 0.0f;
    }
    float eh[4] = {0.0f, 0.0f, 0.0f, 0.0f}, el[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int i = 0; i < n0; ++i) {
        const float a = phi0[i];
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            if (s >= s_count) break;
            const float2 *row = dms + s * mat + i * n1;
            float ts = 0.0f, tc = 0.0f, qs = 0.0f;
            if constexpr (0 < NMAX) {
#pragma unroll
                for (int j = 0; j < NMAX; ++j) {
                    if (j >= n1) break;
                    df_term(row[j], f1[j], ts, tc, qs);
                }
            }
            else {
                for (int j = 0; j < n1; ++j) df_term(row[j], phi1[j], ts, tc, qs);
            }
            {
                const float ph = a * ts;
                const float pl = __builtin_fmaf(a, ts, -ph) + a * (tc + qs);
                df_acc(eh[s], el[s], ph);
                el[s] += pl;
            }
        }
    }
    double *out = den + denoff[p] + static_cast<std::size_t>(Nc);
#pragma unroll
    for (int s = 0; s < 4; ++s) {
        if (s >= s_count) break;
        atomicAdd(out + static_cast<std::size_t>(s_first + s) * den_stride,
                  static_cast<double>(eh[s]) + static_cast<double>(el[s]));
    }
}

template <int NMAX, bool SPLIT>
int otf_density_df_launch(int pair_count, int max_nolg, int spin_count, std::size_t shared, const int *d_no0,
                          const int *d_no1, const int *d_nolg, const std::size_t *d_dmoff, const std::size_t *d_noff,
                          const std::size_t *d_o0off, const std::size_t *d_o1off, const std::size_t *d_denoff,
                          const int *d_nc, const double *d_dm, const float *d_o0, const float *d_o1,
                          std::size_t den_stride, double *d_den)
{
    hipLaunchKernelGGL((otf_density_df_kernel<NMAX, SPLIT>),
                       dim3(static_cast<unsigned>(pair_count),
                            static_cast<unsigned>((max_nolg + kDenDfThreads - 1) / kDenDfThreads),
                            static_cast<unsigned>(SPLIT ? spin_count : 1)),
                       dim3(kDenDfThreads), shared, 0, spin_count, d_no0, d_no1, d_nolg, d_dmoff, d_noff, d_o0off,
                       d_o1off, d_denoff, d_nc, d_dm, d_o0, d_o1, den_stride, d_den);
    return hipGetLastError() == hipSuccess ? 0 : 2;
}

template <bool SPLIT>
int otf_density_df_dispatch(int pair_count, int max_nolg, int spin_count, int max_no, std::size_t shared,
                            const int *d_no0, const int *d_no1, const int *d_nolg, const std::size_t *d_dmoff,
                            const std::size_t *d_noff, const std::size_t *d_o0off, const std::size_t *d_o1off,
                            const std::size_t *d_denoff, const int *d_nc, const double *d_dm, const float *d_o0,
                            const float *d_o1, std::size_t den_stride, double *d_den)
{
    if (max_no <= 16)
        return otf_density_df_launch<16, SPLIT>(pair_count, max_nolg, spin_count, shared, d_no0, d_no1, d_nolg,
                                                d_dmoff, d_noff, d_o0off, d_o1off, d_denoff, d_nc, d_dm, d_o0, d_o1,
                                                den_stride, d_den);
    if (max_no <= 32)
        return otf_density_df_launch<32, SPLIT>(pair_count, max_nolg, spin_count, shared, d_no0, d_no1, d_nolg,
                                                d_dmoff, d_noff, d_o0off, d_o1off, d_denoff, d_nc, d_dm, d_o0, d_o1,
                                                den_stride, d_den);
    if (max_no <= 64)
        return otf_density_df_launch<64, SPLIT>(pair_count, max_nolg, spin_count, shared, d_no0, d_no1, d_nolg,
                                                d_dmoff, d_noff, d_o0off, d_o1off, d_denoff, d_nc, d_dm, d_o0, d_o1,
                                                den_stride, d_den);
    return otf_density_df_launch<0, SPLIT>(pair_count, max_nolg, spin_count, shared, d_no0, d_no1, d_nolg, d_dmoff,
                                           d_noff, d_o0off, d_o1off, d_denoff, d_nc, d_dm, d_o0, d_o1, den_stride,
                                           d_den);
}

} // namespace

extern "C" int Set_Hamiltonian_Hip_OTF_Density(int pair_count, int max_nolg, int spin_count, int max_no,
                                               const int *d_no0, const int *d_no1, const int *d_nolg,
                                               const std::size_t *d_dmoff, const std::size_t *d_noff,
                                               const std::size_t *d_o0off, const std::size_t *d_o1off,
                                               const std::size_t *d_denoff, const int *d_nc, const double *d_dm,
                                               const float *d_o0, const float *d_o1, std::size_t den_stride,
                                               double *d_den)
{
    unsigned ychunks;

    if (pair_count <= 0 || max_nolg <= 0) return 0;
    if (0 < max_no && 0 < spin_count && spin_count <= 4 && grid_precision_df()) {
        /* the (hi, lo) DM blocks of a pair: every spin's in one block when
           they fit in LDS, else one spin's per block (FP64 only when even
           that does not fit: beyond 90 orbitals on 64 KiB) */
        const std::size_t block = static_cast<std::size_t>(max_no) * static_cast<std::size_t>(max_no) * sizeof(float2);
        const std::size_t limit = static_cast<std::size_t>(device_shared_limit() < 0 ? 0 : device_shared_limit());

        if (static_cast<std::size_t>(spin_count) * block <= limit)
            return otf_density_df_dispatch<false>(pair_count, max_nolg, spin_count, max_no,
                                                  static_cast<std::size_t>(spin_count) * block, d_no0, d_no1, d_nolg,
                                                  d_dmoff, d_noff, d_o0off, d_o1off, d_denoff, d_nc, d_dm, d_o0, d_o1,
                                                  den_stride, d_den);
        if (block <= limit)
            return otf_density_df_dispatch<true>(pair_count, max_nolg, spin_count, max_no, block, d_no0, d_no1,
                                                 d_nolg, d_dmoff, d_noff, d_o0off, d_o1off, d_denoff, d_nc, d_dm, d_o0,
                                                 d_o1, den_stride, d_den);
    }
    ychunks = static_cast<unsigned>((max_nolg + kThreads - 1) / kThreads);
    if (ychunks > 16u) ychunks = 16u;
    hipLaunchKernelGGL(otf_density_kernel, dim3(static_cast<unsigned>(pair_count), ychunks), dim3(kThreads), 0, 0,
                       pair_count, spin_count, d_no0, d_no1, d_nolg, d_dmoff, d_noff, d_o0off, d_o1off, d_denoff,
                       d_nc, d_dm, d_o0, d_o1, den_stride, d_den);
    return hipGetLastError() == hipSuccess ? 0 : 2;
}

namespace {

/* The double-float density of the distributed quadrature's CSR gather
   (Set_Density_Grid_GPU.c, the NVIDIA version's mode-2 local kernel), for
   the ranks whose Set_Hamiltonian tables are device-resident.  One thread
   per output point (a sphere point of one of the rank's atoms) sums its
   terms -- its overlap points with the neighbours, in the CSR order, which
   is the CPU loop's (h_AN, Nog) order -- each term as the on-the-fly
   kernel above sums a point of a pair: the density matrix as (hi, lo)
   floats, every product exact (TwoProd), the hi parts of a row summed with
   TwoSum compensation and the lo parts plainly, the rows of the pair
   accumulated as a float pair; then the pair's sum goes into the point's
   FP64 density (FP64 between the pairs, as the FP64 kernel adds them).  A
   gather: no atomics, the same result on every run.  The neighbour
   orbitals of a term sit in a register row (NMAX >= every NO1; the
   64-float row partly spills to scratch and is still several times faster
   than reading the orbitals in the inner loop), beyond 64 orbitals NMAX =
   0 reads them from memory.  Every variant gives the same bits. */
constexpr int kCsrDfThreads = 128;

/* the density matrix of the CSR kernel as (hi, lo) floats, in place: a
   float2 takes the 8 bytes of its double */
__global__ void csr_dm_split_kernel(std::size_t count, double *dm)
{
    float2 *dmf = reinterpret_cast<float2 *>(dm);

    for (std::size_t e = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; e < count;
         e += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        const double x = dm[e];
        const float hi = static_cast<float>(x);
        dmf[e] = make_float2(hi, static_cast<float>(x - static_cast<double>(hi)));
    }
}

template <int NMAX>
__global__ __launch_bounds__(kCsrDfThreads) void csr_density_df_kernel(
    std::size_t output_count, int spin_count, const std::uint32_t *__restrict__ out_ptr,
    const std::uint32_t *__restrict__ term_pt, const std::uint32_t *__restrict__ pt_pair,
    const int *__restrict__ no0, const int *__restrict__ no1, const std::size_t *__restrict__ hoff,
    const std::size_t *__restrict__ noff, const std::size_t *__restrict__ o0off,
    const std::size_t *__restrict__ o1off, const int *__restrict__ nc, const float *__restrict__ o0,
    const float *__restrict__ o1, const float2 *__restrict__ dmf, double *__restrict__ den)
{
#pragma clang fp contract(off)
    const std::size_t out = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (out >= output_count) return;
    double g[4] = {0.0, 0.0, 0.0, 0.0};

    for (std::uint32_t q = out_ptr[out]; q < out_ptr[out + 1]; ++q) {
        const std::uint32_t pt = term_pt[q];
        const std::uint32_t pair = pt_pair[pt];
        const int n0 = no0[pair], n1 = no1[pair];
        const std::size_t mat = static_cast<std::size_t>(n0) * static_cast<std::size_t>(n1);
        const float2 *dp = dmf + hoff[pair];
        /* the central atom's sphere point and the neighbour's overlap point */
        const float *phi0 = o0 + o0off[pair] + static_cast<std::size_t>(nc[pt]) * static_cast<std::size_t>(n0);
        const float *phi1 = o1 + o1off[pair] + (static_cast<std::size_t>(pt) - noff[pair]) * static_cast<std::size_t>(n1);
        float eh[4] = {0.0f, 0.0f, 0.0f, 0.0f}, el[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float f1[NMAX > 0 ? NMAX : 1];

        if constexpr (0 < NMAX) {
#pragma unroll
            for (int j = 0; j < NMAX; ++j) f1[j] = (j < n1) ? phi1[j] : 0.0f;
        }
        for (int i = 0; i < n0; ++i) {
            const float a = phi0[i];
#pragma unroll
            for (int s = 0; s < 4; ++s) {
                if (s >= spin_count) break;
                const float2 *row = dp + static_cast<std::size_t>(s) * mat + static_cast<std::size_t>(i) * n1;
                float ts = 0.0f, tc = 0.0f, qs = 0.0f;
                if constexpr (0 < NMAX) {
#pragma unroll
                    for (int j = 0; j < NMAX; ++j) {
                        if (j >= n1) break;
                        df_term(row[j], f1[j], ts, tc, qs);
                    }
                }
                else {
                    for (int j = 0; j < n1; ++j) df_term(row[j], phi1[j], ts, tc, qs);
                }
                {
                    const float ph = a * ts;
                    const float pl = __builtin_fmaf(a, ts, -ph) + a * (tc + qs);
                    df_acc(eh[s], el[s], ph);
                    el[s] += pl;
                }
            }
        }
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            if (s >= spin_count) break;
            g[s] += static_cast<double>(eh[s]) + static_cast<double>(el[s]);
        }
    }
#pragma unroll
    for (int s = 0; s < 4; ++s) {
        if (s >= spin_count) break;
        den[static_cast<std::size_t>(s) * output_count + out] = g[s];
    }
}

template <int NMAX>
hipError_t csr_density_df_launch(std::size_t output_count, int spin_count, const std::uint32_t *d_out_ptr,
                                 const std::uint32_t *d_term_pt, const std::uint32_t *d_pt_pair, const int *d_no0,
                                 const int *d_no1, const std::size_t *d_hoff, const std::size_t *d_noff,
                                 const std::size_t *d_o0off, const std::size_t *d_o1off, const int *d_nc,
                                 const float *d_o0, const float *d_o1, const float2 *d_dmf, double *d_den)
{
    hipLaunchKernelGGL(csr_density_df_kernel<NMAX>,
                       dim3(static_cast<unsigned>((output_count + kCsrDfThreads - 1) / kCsrDfThreads)),
                       dim3(kCsrDfThreads), 0, 0, output_count, spin_count, d_out_ptr, d_term_pt, d_pt_pair, d_no0,
                       d_no1, d_hoff, d_noff, d_o0off, d_o1off, d_nc, d_o0, d_o1, d_dmf, d_den);
    return hipGetLastError();
}

} // namespace

/* The precision of the grid integrals for C code: 1 when the double-float
   kernels run (grid_precision_configure), 0 for FP64. */
extern "C" int openmx_grid_precision_df(void)
{
    return grid_precision_df() ? 1 : 0;
}

/* The banner of the NVIDIA version's Grid_Precision_BeginStep, once per
   process; the caller prints it on one rank. */
extern "C" void openmx_grid_precision_announce(void)
{
    static int announced = 0;

    if (announced || !openmx_fp32_enabled()) return;
    const int mode = grid_precision_configure();
    if (mode == 1 && !OPENMX_STDOUT_LEVEL(2)) return;
    announced = 1;
    if (mode == 1) {
        std::printf("<DFT>  grid integrals: double-float kernels (FP32 arithmetic with error-free transformations; "
                    "OPENMX_GRID_PRECISION=fp64 restores the FP64 kernels)\n");
    }
    else if (grid_precision_unrecognized[0] != '\0') {
        std::printf("<DFT>  grid integrals: OPENMX_GRID_PRECISION=%s is neither df nor fp64; FP64 kernels\n",
                    grid_precision_unrecognized);
    }
    std::fflush(stdout);
}

/* The density of the local mode's CSR gather (Set_Density_Grid_GPU.c) in
   double-float, every operand on the device: the rank's tables of
   Set_Hamiltonian (d_no0 .. d_o1), the CSR over the output points, the
   packed density matrix d_dm (dm_count doubles, overwritten by its (hi,
   lo) split) and the density d_den (spin_count x output_count, written).
   max_no bounds the NO1 of every pair the terms reach.  Returns 0 when
   d_den holds the density, 1 when nothing ran (the FP64 mode or an
   argument outside the kernel's range: the caller's FP64 kernel takes
   over, d_dm intact), 2 on a device failure (d_dm and d_den undefined). */
extern "C" int Set_Density_Grid_Hip_LocalDensity(std::size_t output_count, int spin_count, int max_no,
                                                 std::size_t dm_count, const std::uint32_t *d_out_ptr,
                                                 const std::uint32_t *d_term_pt, const std::uint32_t *d_pt_pair,
                                                 const int *d_no0, const int *d_no1, const std::size_t *d_hoff,
                                                 const std::size_t *d_noff, const std::size_t *d_o0off,
                                                 const std::size_t *d_o1off, const int *d_nc, const float *d_o0,
                                                 const float *d_o1, double *d_dm, double *d_den)
{
    const float2 *d_dmf = reinterpret_cast<const float2 *>(d_dm);
    hipError_t status = hipSuccess;

    if (spin_count < 1 || 4 < spin_count || max_no < 0 || !grid_precision_df()) return 1;
    if (output_count == 0) return 0;
    (void)hipGetLastError();
    if (0 < dm_count) {
        std::size_t blocks = (dm_count + kThreads - 1) / kThreads;
        if (blocks > 4096) blocks = 4096;
        hipLaunchKernelGGL(csr_dm_split_kernel, dim3(static_cast<unsigned>(blocks)), dim3(kThreads), 0, 0, dm_count,
                           d_dm);
        status = hipGetLastError();
    }
    if (status == hipSuccess) {
        if (max_no <= 16)
            status = csr_density_df_launch<16>(output_count, spin_count, d_out_ptr, d_term_pt, d_pt_pair, d_no0, d_no1,
                                               d_hoff, d_noff, d_o0off, d_o1off, d_nc, d_o0, d_o1, d_dmf, d_den);
        else if (max_no <= 32)
            status = csr_density_df_launch<32>(output_count, spin_count, d_out_ptr, d_term_pt, d_pt_pair, d_no0, d_no1,
                                               d_hoff, d_noff, d_o0off, d_o1off, d_nc, d_o0, d_o1, d_dmf, d_den);
        else if (max_no <= 64)
            status = csr_density_df_launch<64>(output_count, spin_count, d_out_ptr, d_term_pt, d_pt_pair, d_no0, d_no1,
                                               d_hoff, d_noff, d_o0off, d_o1off, d_nc, d_o0, d_o1, d_dmf, d_den);
        else
            status = csr_density_df_launch<0>(output_count, spin_count, d_out_ptr, d_term_pt, d_pt_pair, d_no0, d_no1,
                                              d_hoff, d_noff, d_o0off, d_o1off, d_nc, d_o0, d_o1, d_dmf, d_den);
    }
    if (status == hipSuccess) status = hipDeviceSynchronize();
    if (status != hipSuccess) {
        (void)hipGetLastError();
        return 2;
    }
    return 0;
}

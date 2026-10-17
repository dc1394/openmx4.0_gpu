/* Refined FP32 eigensolver of the dense GPU paths (see eigen_refine_gpu.h).

   Algorithm (Ogita and Aishima, RefSyEv, with clusters): A is made symmetric
   or Hermitian from its lower triangle, the part the eigensolver reads; all
   n eigenpairs are solved in FP32 (the full solve is what the refinement
   needs as its correction basis), X = the n vectors in FP64 and X1 = the
   maxn wanted ones.  K times

       Y = A X1,  S = X^H Y,  G = X^H X1,  R = I - G,
       lambda_j = s_jj / g_jj for the wanted j (the rest keep their FP32
       eigenvalues),
       E_ij = (s_ij + lambda_j r_ij) / (lambda_j - lambda_i) when
       |lambda_j - lambda_i| > delta and the occupations of i and j differ,
       else E_ij = r_ij / 2 (the pair is only orthonormalized),
       X1 <- X1 + X E,

   delta = 2 (max_{i != j} |s_ij| + spectral radius * max |r_ij|) of the
   first step, kept for every step: recomputed later it only grows (the
   pairs left as clusters keep their residuals), while the estimates of the
   states left as clusters are off by at most the width of their cluster.
   Pairs of equal occupation are left alone because the density matrix needs
   the occupied subspace, not the rotations inside it (rotating them breaks
   the orthonormality of nearly degenerate pairs), and nearly degenerate
   pairs inside it are where one Newton step fails.  Afterwards every
   cluster of consecutive estimates closer than 30 delta whose occupations
   differ gets a Rayleigh-Ritz step (its H and overlap, the small generalized
   problem, the rotation): pairs with a gap of at least 30 delta start with
   mixings of at most max |s_ij| / gap, which two Newton steps take below
   1e-10, those just above delta start near 0.1 and do not.  The products
   go through the GEMMul8 bridge (15 moduli match FP64) with the full
   workspace when the device has room.

   The energy density matrix of the refined vectors is off by up to 2e-6
   (the vectors inside the fully occupied clusters stay FP32 mixtures),
   which is why the solvers solve the step that ends the SCF in FP64.

   When fewer than n columns are refined (the band solvers: maxn of n), the
   unrefined columns are rotated against the refined ones with the same
   formula every step: left alone, their FP32-level components along the
   occupied subspace and the first-order mixing inside the occupied block
   tilt the refined columns' fixed point out of the occupied subspace by
   about 1e-8 (an SCF floor of 1e-7, 2-5 more SCF steps, and a bias of
   2e-10 Ha across warm starts; OPENMX_EIGEN_REFINE_COMPLEMENT).

   HIP port of the NVIDIA module: the FP32 full solve is rocSOLVER's
   divide and conquer (hipsolverDn{S,C}{sy,he}evd; 0.35 s for a 2808 x 2808
   and 2.64 s for a 5616 x 5616 complex problem on an RX 9060 XT, where
   MAGMA's hybrid solver reduces matrices of order below 3000 on the CPU),
   and every loop is a kernel on the stream of the caller's hipBLAS handle.
*/
#include "eigen_refine_gpu.h"
#include "openmx_precision.h"
#include "openmx_stdout_level.h"

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <hipsolver/hipsolver.h>
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <strings.h>

extern "C" {
/* the GEMMul8 bridge (gemmul8_bridge.cu) */
hipblasStatus_t openmx_gemmul8Dgemm(hipblasHandle_t, hipblasOperation_t, hipblasOperation_t, int, int, int,
                                    const double *, const double *, int, const double *, int, const double *,
                                    double *, int);
hipblasStatus_t openmx_gemmul8Zgemm(hipblasHandle_t, hipblasOperation_t, hipblasOperation_t, int, int, int,
                                    const hipDoubleComplex *, const hipDoubleComplex *, int, const hipDoubleComplex *,
                                    int, const hipDoubleComplex *, hipDoubleComplex *, int);
hipblasStatus_t openmx_gemmul8DgemmUnblocked(hipblasHandle_t, hipblasOperation_t, hipblasOperation_t, int, int, int,
                                             const double *, const double *, int, const double *, int,
                                             const double *, double *, int);
hipblasStatus_t openmx_gemmul8ZgemmUnblocked(hipblasHandle_t, hipblasOperation_t, hipblasOperation_t, int, int, int,
                                             const hipDoubleComplex *, const hipDoubleComplex *, int,
                                             const hipDoubleComplex *, int, const hipDoubleComplex *,
                                             hipDoubleComplex *, int);
hipblasStatus_t openmx_gemmul8DgemmSolve(hipblasHandle_t, hipblasOperation_t, hipblasOperation_t, int, int, int,
                                         const double *, const double *, int, const double *, int, const double *,
                                         double *, int);
hipblasStatus_t openmx_gemmul8ZgemmSolve(hipblasHandle_t, hipblasOperation_t, hipblasOperation_t, int, int, int,
                                         const hipDoubleComplex *, const hipDoubleComplex *, int,
                                         const hipDoubleComplex *, int, const hipDoubleComplex *, hipDoubleComplex *,
                                         int);
int       openmx_gemmul8DgemmEnabled(void);
int       openmx_gemmul8ZgemmEnabled(void);
long long openmx_gemmul8NativeCalls(int is_complex);

/* MAGMA's hybrid FP32 solvers (magma_openmx.cu): all eigenpairs, the
   eigenvalues ascending into w (host) */
int openmx_magma_ssyevd_gpu(int n, float *d_A, float *w);
int openmx_magma_cheevd_gpu(int n, void *d_A, float *w);

/* LAPACK (AOCL libFLAME), with interleaved complex arrays */
void dsygv_(int *itype, const char *jobz, const char *uplo, int *n, double *a, int *lda, double *b, int *ldb,
            double *w, double *work, int *lwork, int *info);
void zhegv_(int *itype, const char *jobz, const char *uplo, int *n, double *a, int *lda, double *b, int *ldb,
            double *w, double *work, int *lwork, double *rwork, int *info);
}

namespace {

constexpr int kThreads = 256;

int env_flag(const char *name, int fallback)
{
    const char *value = getenv(name);

    if (value == NULL || value[0] == '\0') return fallback;
    return atoi(value) != 0;
}

void refine_check(hipError_t status, const char *what)
{
    if (status != hipSuccess) {
        fprintf(stderr, "eigen_refine_gpu: %s: %s\n", what, hipGetErrorString(status));
        fflush(stderr);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

void refine_check_blas(hipblasStatus_t status, const char *what)
{
    if (status != HIPBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "eigen_refine_gpu: %s: hipBLAS status %d\n", what, (int)status);
        fflush(stderr);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

hipStream_t refine_stream(const EigenRefineDevice *dev)
{
    hipStream_t stream = nullptr;

    refine_check_blas(hipblasGetStream(dev->hipblas, &stream), "hipblasGetStream");
    return stream;
}

unsigned grid_for(size_t count)
{
    size_t blocks = (count + kThreads - 1) / kThreads;

    if (blocks < 1) blocks = 1;
    if (blocks > 65535u * 16u) blocks = 65535u * 16u;
    return (unsigned)blocks;
}

/* ------------------------------------------------------------------ */
/* kernels                                                            */

__global__ void k_f64_to_f32(const double *__restrict__ a, float *__restrict__ af, size_t count)
{
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < count; i += (size_t)gridDim.x * blockDim.x)
        af[i] = (float)a[i];
}

/* The complex conversion goes through the unitary similarity D^H A D,
   D = diag(d_k), d_k = exp(2 pi i frac(k phi)) (phi the golden ratio), and
   every exactly zero off-diagonal element of the result is replaced by a
   value of order 1e-30.  rocSOLVER's Hermitian divide and conquer (cheevd
   and zheevd of ROCm 7.2) returns info 0 with NaN vectors for any complex
   matrix whose elements are all real -- the transformed Hamiltonian of a
   non-collinear calculation without spin-orbit coupling -- and for an exactly
   block-diagonal one, [[H, 0], [0, H]] of a non-magnetic one: the phases make
   the matrix genuinely complex without changing its spectrum, the dither
   couples the blocks 23 orders of magnitude below the FP32 rounding of the
   elements.  The vectors v of D^H A D give those of A as D v
   (k_c32_to_c64_phase); the refinement works on the exact FP64 matrix. */
__device__ inline void refine_phase(int k, double *c, double *s)
{
    double t = (double)k * 0.6180339887498949;

    t -= floor(t);
    sincospi(2.0 * t, s, c);
}

__global__ void k_f64_to_c32_phase(const double *__restrict__ a, float *__restrict__ af, int n)
{
    const size_t nn = (size_t)n * (size_t)n;

    for (size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x; idx < nn; idx += (size_t)gridDim.x * blockDim.x) {
        const int    row = (int)(idx % (size_t)n), col = (int)(idx / (size_t)n);
        const double re = a[2 * idx], im = a[2 * idx + 1];
        double       cr, sr, cc, sc;

        refine_phase(row, &cr, &sr);
        refine_phase(col, &cc, &sc);
        {
            const double pr = cr * cc + sr * sc, pi = cr * sc - sr * cc;   /* conj(d_row) d_col */
            float        fr = (float)(re * pr - im * pi), fi = (float)(re * pi + im * pr);

            /* 1e-30 still let ROCm 7.2's cheevd return NaN vectors for
               matrices of two tridiagonal blocks (n = 100 to 1000); 1e-25,
               far below the FP32 rounding of any element, did not */
            if (fr == 0.0f && fi == 0.0f && row != col)
                fr = 1.0e-25f * (float)(1 + (unsigned)(row * 7 + col * 13) % 17u);
            af[2 * idx] = fr;
            af[2 * idx + 1] = fi;
        }
    }
}

/* the vectors v (n x n complex FP32) of D^H A D to those of A in FP64: D v */
__global__ void k_c32_to_c64_phase(const float *__restrict__ af, double *__restrict__ x, int n)
{
    const size_t nn = (size_t)n * (size_t)n;

    for (size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x; idx < nn; idx += (size_t)gridDim.x * blockDim.x) {
        const double vr = af[2 * idx], vi = af[2 * idx + 1];
        double       c, s;

        refine_phase((int)(idx % (size_t)n), &c, &s);
        x[2 * idx] = c * vr - s * vi;
        x[2 * idx + 1] = c * vi + s * vr;
    }
}

/* clears *flag when one of count floats is not finite */
__global__ void k_finite_f32(const float *__restrict__ x, size_t count, int *flag)
{
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < count; i += (size_t)gridDim.x * blockDim.x)
        if (!isfinite(x[i])) *flag = 0;
}

__global__ void k_f32_to_f64(const float *__restrict__ af, double *__restrict__ x, size_t count)
{
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < count; i += (size_t)gridDim.x * blockDim.x)
        x[i] = (double)af[i];
}

/* the upper triangle from the lower one; for a complex matrix also the
   diagonal made real */
__global__ void k_symmetrize(double *a, int n, int cplx)
{
    const size_t nn = (size_t)n * (size_t)n;

    for (size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x; idx < nn; idx += (size_t)gridDim.x * blockDim.x) {
        const int row = (int)(idx % (size_t)n), col = (int)(idx / (size_t)n);

        if (row < col) {
            const size_t low = (size_t)col + (size_t)row * (size_t)n;

            if (cplx) {
                a[2 * idx]     = a[2 * low];
                a[2 * idx + 1] = -a[2 * low + 1];
            }
            else {
                a[idx] = a[low];
            }
        }
        else if (row == col && cplx) {
            a[2 * idx + 1] = 0.0;
        }
    }
}

/* per block: max offdiag |S|, max |G - I|, and whether every entry is finite
   with a positive diagonal of G; S in b2 and G in b1, n x kc */
__global__ void k_maxima(const double *__restrict__ b1, const double *__restrict__ b2, int n, int kc, int width,
                         double *partial)
{
    __shared__ double s_s[kThreads], s_r[kThreads], s_f[kThreads];
    const size_t count = (size_t)n * (size_t)kc;
    double max_s = 0.0, max_r = 0.0, finite = 1.0;

    for (size_t at = blockIdx.x * (size_t)blockDim.x + threadIdx.x; at < count; at += (size_t)gridDim.x * blockDim.x) {
        const int    i = (int)(at % (size_t)n), j = (int)(at / (size_t)n);
        const double gr = b1[width * at] - ((i == j) ? 1.0 : 0.0);
        const double gi = (width == 2) ? b1[width * at + 1] : 0.0;
        const double sr = b2[width * at];
        const double si = (width == 2) ? b2[width * at + 1] : 0.0;

        /* fmax alone discards NaNs, so an invalid product would look
           perfectly converged and overwrite the FP64 fallback input */
        if (!isfinite(gr) || !isfinite(gi) || !isfinite(sr) || !isfinite(si)) finite = 0.0;
        if (i == j && b1[width * at] <= 0.0) finite = 0.0;
        if (i != j) max_s = fmax(max_s, hypot(sr, si));
        max_r = fmax(max_r, hypot(gr, gi));
    }
    s_s[threadIdx.x] = max_s;
    s_r[threadIdx.x] = max_r;
    s_f[threadIdx.x] = finite;
    __syncthreads();
    for (int half = blockDim.x / 2; half > 0; half >>= 1) {
        if ((int)threadIdx.x < half) {
            s_s[threadIdx.x] = fmax(s_s[threadIdx.x], s_s[threadIdx.x + half]);
            s_r[threadIdx.x] = fmax(s_r[threadIdx.x], s_r[threadIdx.x + half]);
            s_f[threadIdx.x] = fmin(s_f[threadIdx.x], s_f[threadIdx.x + half]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        partial[3 * blockIdx.x]     = s_s[0];
        partial[3 * blockIdx.x + 1] = s_r[0];
        partial[3 * blockIdx.x + 2] = s_f[0];
    }
}

__global__ void k_lambda(const double *__restrict__ b1, const double *__restrict__ b2, double *lam, int n, int kc,
                         int width)
{
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < kc; j += gridDim.x * blockDim.x) {
        const size_t at = (size_t)j + (size_t)j * (size_t)n;

        lam[j] = b2[width * at] / b1[width * at];
    }
}

/* E in place of G (b1), n x kc */
__global__ void k_correction(double *b1, const double *__restrict__ b2, const double *__restrict__ lam,
                             const double *__restrict__ occ, int n, int kc, int width, double delta)
{
    const size_t count = (size_t)n * (size_t)kc;

    for (size_t at = blockIdx.x * (size_t)blockDim.x + threadIdx.x; at < count; at += (size_t)gridDim.x * blockDim.x) {
        const int    i = (int)(at % (size_t)n), j = (int)(at / (size_t)n);
        const double rr = ((i == j) ? 1.0 : 0.0) - b1[width * at];
        const double ri = (width == 2) ? -b1[width * at + 1] : 0.0;
        double       er = 0.5 * rr, ei = 0.5 * ri;

        if (i != j) {
            const double d = lam[j] - lam[i];

            if (delta < fabs(d) && 1.0e-12 <= fabs(occ[i] - occ[j])) {
                er = (b2[width * at] + lam[j] * rr) / d;
                ei = (width == 2) ? (b2[width * at + 1] + lam[j] * ri) / d : 0.0;
            }
        }
        b1[width * at] = er;
        if (width == 2) b1[width * at + 1] = ei;
    }
}

/* E_rc in place of G_rc (b3), kc x nc, S_rc in b2 */
__global__ void k_complement(double *b3, const double *__restrict__ b2, const double *__restrict__ lam,
                             const double *__restrict__ occ, int kc, int nc, int width, double delta)
{
    const size_t count = (size_t)kc * (size_t)nc;

    for (size_t at = blockIdx.x * (size_t)blockDim.x + threadIdx.x; at < count; at += (size_t)gridDim.x * blockDim.x) {
        const int    i = (int)(at % (size_t)kc), jc = (int)(at / (size_t)kc);
        const int    j = kc + jc;
        const double rr = -b3[width * at];
        const double ri = (width == 2) ? -b3[width * at + 1] : 0.0;
        const double d = lam[j] - lam[i];
        double       er = 0.5 * rr, ei = 0.5 * ri;

        if (delta < fabs(d) && 1.0e-12 <= fabs(occ[i] - occ[j])) {
            er = (b2[width * at] + lam[j] * rr) / d;
            ei = (width == 2) ? (b2[width * at + 1] + lam[j] * ri) / d : 0.0;
        }
        b3[width * at] = er;
        if (width == 2) b3[width * at + 1] = ei;
    }
}

/* clears *flag when an entry of x (count doubles) or of lam (kc) is not finite */
__global__ void k_finite(const double *__restrict__ x, const double *__restrict__ lam, size_t count, int kc, int *flag)
{
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < count; i += (size_t)gridDim.x * blockDim.x) {
        if (!isfinite(x[i])) *flag = 0;
        if (i < (size_t)kc && !isfinite(lam[i])) *flag = 0;
    }
}

/* diagnostic: per column the residual |A x_j - lam_j x_j| / |x_j| (b1 holds A X) */
__global__ void k_check_residual(const double *__restrict__ b1, const double *__restrict__ x,
                                 const double *__restrict__ lam, double *out, int n, int kc, int width)
{
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < kc; j += gridDim.x * blockDim.x) {
        double num = 0.0, den = 0.0;

        for (int i = 0; i < n; i++) {
            const size_t at = (size_t)i + (size_t)j * (size_t)n;
            const double xr = x[width * at], xi = (width == 2) ? x[width * at + 1] : 0.0;
            const double yr = b1[width * at] - lam[j] * xr;
            const double yi = (width == 2) ? b1[width * at + 1] - lam[j] * xi : 0.0;

            num += yr * yr + yi * yi;
            den += xr * xr + xi * xi;
        }
        out[j] = sqrt(num / den);
    }
}

/* diagnostic: per column max_i |x_i^H x_j - delta_ij| (b2 holds X^H X1) */
__global__ void k_check_orthogonality(const double *__restrict__ b2, double *out, int n, int kc, int width)
{
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < kc; j += gridDim.x * blockDim.x) {
        double worst = 0.0;

        for (int i = 0; i < n; i++) {
            const size_t at = (size_t)i + (size_t)j * (size_t)n;
            const double gr = b2[width * at] - ((i == j) ? 1.0 : 0.0);
            const double gi = (width == 2) ? b2[width * at + 1] : 0.0;

            worst = fmax(worst, hypot(gr, gi));
        }
        out[j] = worst;
    }
}

/* ------------------------------------------------------------------ */
/* the module's hipSOLVER handle (FP32 full solve, device Rayleigh-Ritz) */

hipsolverDnHandle_t g_solver = nullptr;

hipsolverDnHandle_t refine_solver(hipStream_t stream)
{
    if (g_solver == nullptr && hipsolverDnCreate(&g_solver) != HIPSOLVER_STATUS_SUCCESS) {
        g_solver = nullptr;
        return nullptr;
    }
    if (hipsolverDnSetStream(g_solver, stream) != HIPSOLVER_STATUS_SUCCESS) return nullptr;
    return g_solver;
}

/* OPENMX_EIGEN_FP32_SOLVER: rocsolver (default) or magma */
int fp32_backend_magma()
{
    static int magma = -1;

    if (magma < 0) {
        const char *value = getenv("OPENMX_EIGEN_FP32_SOLVER");
        magma = (value != NULL && strcasecmp(value, "magma") == 0) ? 1 : 0;
    }
    return magma;
}

/* ------------------------------------------------------------------ */

void refine_blocks_release(EigenRefineState *st)
{
    if (st->own != NULL) refine_check(hipFree(st->own), "hipFree blocks");
    st->own = NULL;
    st->b1 = st->b2 = st->b3 = NULL;
    st->products_ready = 0;
}

/* the layout of the product blocks of a solve of maxn wanted states: three
   n x kc blocks while the unrefined columns are rotated against the refined
   ones, two when every column is refined */
void refine_blocks_layout(int cplx, int n, int maxn, int *blocks, size_t *block_bytes)
{
    const int    kc = env_flag("OPENMX_EIGEN_REFINE_ALL", 0) ? n : maxn;
    const size_t nk = (size_t)n * (size_t)kc;

    *blocks = (kc < n && env_flag("OPENMX_EIGEN_REFINE_COMPLEMENT", 1)) ? 3 : 2;
    *block_bytes = (nk * (cplx ? 2 : 1) * sizeof(double) + 511u) / 512u * 512u;
}

/* the product blocks: the FP32 scratch once the solve is done, else a
   buffer of their own; 0 when there is no room */
int refine_blocks_ensure(EigenRefineState *st, EigenRefineDevice *dev, const EigenRefineProblem *pb,
                         EigenRefineReport *report)
{
    int    blocks;
    size_t block_bytes;

    refine_blocks_layout(pb->cplx, pb->n, pb->maxn, &blocks, &block_bytes);
    refine_blocks_release(st);
    if ((size_t)blocks * block_bytes <= pb->region) {
        st->b1 = (double *)pb->fp32;
    }
    else {
        if (dev->try_malloc(&st->own, (size_t)blocks * block_bytes) != hipSuccess) {
            if (OPENMX_STDOUT_LEVEL(2)) {
                printf("<eigen_refine_gpu> no room for the %.1f MiB of refinement products; FP64 eigensolver%s\n",
                       (double)blocks * (double)block_bytes / (1024.0 * 1024.0),
                       pb->transient ? "" : " for the rest of the SCF cycle");
                fflush(stdout);
            }
            report->persistent = pb->transient ? 0 : 1;
            return 0;
        }
        st->b1 = (double *)st->own;
    }
    st->b2 = (double *)((unsigned char *)st->b1 + block_bytes);
    st->b3 = (blocks == 3) ? (double *)((unsigned char *)st->b2 + block_bytes) : NULL;
    return 1;
}

/* The products: GEMMul8 with the full workspace when the device has room
   for it, else the blocked product; OPENMX_EIGEN_REFINE_UNBLOCKED=0 takes
   the blocked product always, and so does a device shared by solving ranks
   (dev->shared); both with the solve reserve of the bridge.  C = op(A) op(B) + beta C. */
hipblasStatus_t refine_gemm(const EigenRefineDevice *dev, int cplx, hipblasOperation_t transa,
                            hipblasOperation_t transb, int m, int n, int k, const void *A, int lda, const void *B,
                            int ldb, double beta, void *C, int ldc)
{
    static int unblocked = -1;

    if (unblocked < 0) unblocked = env_flag("OPENMX_EIGEN_REFINE_UNBLOCKED", 1);
    if (cplx) {
        const hipDoubleComplex one = make_hipDoubleComplex(1.0, 0.0), b = make_hipDoubleComplex(beta, 0.0);

        if (unblocked && !dev->shared)
            return openmx_gemmul8ZgemmUnblocked(dev->hipblas, transa, transb, m, n, k, &one,
                                                (const hipDoubleComplex *)A, lda, (const hipDoubleComplex *)B, ldb,
                                                &b, (hipDoubleComplex *)C, ldc);
        return openmx_gemmul8ZgemmSolve(dev->hipblas, transa, transb, m, n, k, &one, (const hipDoubleComplex *)A,
                                        lda, (const hipDoubleComplex *)B, ldb, &b, (hipDoubleComplex *)C, ldc);
    }
    else {
        const double             one = 1.0;
        const hipblasOperation_t ta = (transa == HIPBLAS_OP_C) ? HIPBLAS_OP_T : transa;
        const hipblasOperation_t tb = (transb == HIPBLAS_OP_C) ? HIPBLAS_OP_T : transb;

        if (unblocked && !dev->shared)
            return openmx_gemmul8DgemmUnblocked(dev->hipblas, ta, tb, m, n, k, &one, (const double *)A, lda,
                                                (const double *)B, ldb, &beta, (double *)C, ldc);
        return openmx_gemmul8DgemmSolve(dev->hipblas, ta, tb, m, n, k, &one, (const double *)A, lda,
                                        (const double *)B, ldb, &beta, (double *)C, ldc);
    }
}

/* plain hipBLAS FP64, for the small products of the Rayleigh-Ritz step */
hipblasStatus_t native_gemm(const EigenRefineDevice *dev, int cplx, hipblasOperation_t transa,
                            hipblasOperation_t transb, int m, int n, int k, const void *A, int lda, const void *B,
                            int ldb, void *C, int ldc)
{
    if (cplx) {
        const hipDoubleComplex one = make_hipDoubleComplex(1.0, 0.0), zero = make_hipDoubleComplex(0.0, 0.0);

        return hipblasZgemm(dev->hipblas, transa, transb, m, n, k, &one, (const hipDoubleComplex *)A, lda,
                            (const hipDoubleComplex *)B, ldb, &zero, (hipDoubleComplex *)C, ldc);
    }
    else {
        const double             one = 1.0, zero = 0.0;
        const hipblasOperation_t ta = (transa == HIPBLAS_OP_C) ? HIPBLAS_OP_T : transa;
        const hipblasOperation_t tb = (transb == HIPBLAS_OP_C) ? HIPBLAS_OP_T : transb;

        return hipblasDgemm(dev->hipblas, ta, tb, m, n, k, &one, (const double *)A, lda, (const double *)B, ldb,
                            &zero, (double *)C, ldc);
    }
}

/* ------------------------------------------------------------------ */

/* the device workspace given back once the stream is done with it */
void refine_work_release(EigenRefineDevice *dev, hipStream_t stream)
{
    if (*dev->d_work != NULL) {
        refine_check(hipStreamSynchronize(stream), "synchronize");
        refine_check(hipFree(*dev->d_work), "hipFree workspace");
        *dev->d_work = NULL;
    }
    *dev->d_work_bytes = 0;
}

/* the device workspace grown to bytes; 0 when there is no room */
int refine_work_ensure(EigenRefineDevice *dev, size_t bytes, const char *what)
{
    if (*dev->d_work_bytes < bytes) {
        void *grown = NULL;

        if (*dev->d_work != NULL) {
            refine_check(hipFree(*dev->d_work), "hipFree workspace");
            *dev->d_work = NULL;
            *dev->d_work_bytes = 0;
        }
        if (dev->try_malloc(&grown, bytes) != hipSuccess) {
            if (OPENMX_STDOUT_LEVEL(2)) {
                printf("<eigen_refine_gpu> no room for the %.1f MiB %s workspace\n", (double)bytes / (1024.0 * 1024.0),
                       what);
                fflush(stdout);
            }
            return 0;
        }
        *dev->d_work = grown;
        *dev->d_work_bytes = bytes;
    }
    return 1;
}

/* rocSOLVER's FP32 divide and conquer of af (n x n, device) in place: the
   vectors in af, the eigenvalues ascending in wf (device).  Returns 1 on
   success; otherwise says why and returns 0. */
int refine_fp32_rocsolver(EigenRefineDevice *dev, hipStream_t stream, int cplx, int n, float *af, float *wf)
{
    hipsolverDnHandle_t const solver = refine_solver(stream);
    int                       lwork = 0;
    int                       info = 0;
    hipsolverStatus_t         status;

    if (solver == nullptr) {
        if (OPENMX_STDOUT_LEVEL(2)) {
            printf("<eigen_refine_gpu> no hipSOLVER handle for the FP32 eigensolver\n");
            fflush(stdout);
        }
        return 0;
    }
    if (cplx)
        status = hipsolverDnCheevd_bufferSize(solver, HIPSOLVER_EIG_MODE_VECTOR, HIPBLAS_FILL_MODE_LOWER, n,
                                              (hipFloatComplex *)af, n, wf, &lwork);
    else
        status = hipsolverDnSsyevd_bufferSize(solver, HIPSOLVER_EIG_MODE_VECTOR, HIPBLAS_FILL_MODE_LOWER, n, af, n,
                                              wf, &lwork);
    if (status != HIPSOLVER_STATUS_SUCCESS) {
        if (OPENMX_STDOUT_LEVEL(2)) {
            printf("<eigen_refine_gpu> FP32 eigensolver workspace query failed (status %d)\n", (int)status);
            fflush(stdout);
        }
        return 0;
    }
    if (lwork < 1) lwork = 1;
    if (!refine_work_ensure(dev, (size_t)lwork * (cplx ? sizeof(hipFloatComplex) : sizeof(float)),
                            "FP32 eigensolver"))
        return 0;
    if (cplx)
        status = hipsolverDnCheevd(solver, HIPSOLVER_EIG_MODE_VECTOR, HIPBLAS_FILL_MODE_LOWER, n,
                                   (hipFloatComplex *)af, n, wf, (hipFloatComplex *)*dev->d_work, lwork, dev->d_info);
    else
        status = hipsolverDnSsyevd(solver, HIPSOLVER_EIG_MODE_VECTOR, HIPBLAS_FILL_MODE_LOWER, n, af, n, wf,
                                   (float *)*dev->d_work, lwork, dev->d_info);
    if (status == HIPSOLVER_STATUS_SUCCESS)
        refine_check(hipMemcpyAsync(&info, dev->d_info, sizeof(int32_t), hipMemcpyDeviceToHost, stream),
                     "download info");
    refine_check(hipStreamSynchronize(stream), "synchronize");
    if (status != HIPSOLVER_STATUS_SUCCESS || info != 0) {
        if (OPENMX_STDOUT_LEVEL(2)) {
            printf("<eigen_refine_gpu> FP32 rocSOLVER eigensolver failed (status %d, info %d)\n", (int)status, info);
            fflush(stdout);
        }
        return 0;
    }
    {
        /* the solver reports success with NaN vectors in some cases (see
           k_f64_to_c32_phase): check them before they become the basis */
        const size_t count = (size_t)(cplx ? 2 : 1) * (size_t)n * (size_t)n + (size_t)n;
        int         *flag = NULL;
        int          finite = 1;

        if (dev->try_malloc((void **)&flag, sizeof(int)) != hipSuccess) return 0;
        refine_check(hipMemcpyAsync(flag, &finite, sizeof(int), hipMemcpyHostToDevice, stream), "upload flag");
        hipLaunchKernelGGL(k_finite_f32, dim3(grid_for(count)), dim3(kThreads), 0, stream, af, count, flag);
        refine_check(hipGetLastError(), "FP32 finite check");
        refine_check(hipMemcpyAsync(&finite, flag, sizeof(int), hipMemcpyDeviceToHost, stream), "download flag");
        refine_check(hipStreamSynchronize(stream), "synchronize");
        refine_check(hipFree(flag), "hipFree flag");
        if (!finite) {
            if (OPENMX_STDOUT_LEVEL(2)) {
                printf("<eigen_refine_gpu> FP32 rocSOLVER eigensolver returned non-finite eigenpairs\n");
                fflush(stdout);
            }
            return 0;
        }
    }
    return 1;
}

/* The FP32 solve of all n eigenpairs of a: rocSOLVER (or MAGMA with
   OPENMX_EIGEN_FP32_SOLVER=magma), and when that one fails the other one
   from a fresh conversion of a, before anything falls to FP64.  a is left
   untouched; the vectors land in af, the eigenvalues ascending in e0 (host).
   Returns 1 on success; otherwise says why and returns 0. */
int refine_fp32_solve(EigenRefineDevice *dev, hipStream_t stream, int cplx, int n, const double *a, float *af,
                      float *wf, double *e0)
{
    const size_t nn = (size_t)n * (size_t)n;
    const size_t count = cplx ? 2 * nn : nn;
    const int    first_magma = fp32_backend_magma();

    {
        static int announced = 0;
        if (!announced && OPENMX_STDOUT_LEVEL(2)) {
            announced = 1;
            printf("<eigen_refine_gpu> FP32 full solve: %s\n",
                   first_magma ? "MAGMA hybrid divide and conquer (OPENMX_EIGEN_FP32_SOLVER=magma)"
                               : "rocSOLVER divide and conquer (hipsolverDn{S,C}{sy,he}evd)");
            fflush(stdout);
        }
    }
    for (int attempt = 0; attempt < 2; attempt++) {
        const int use_magma = (attempt == 0) ? first_magma : !first_magma;
        int       ok = 0;

        if (attempt == 1) {
            if (OPENMX_STDOUT_LEVEL(2)) {
                printf("<eigen_refine_gpu> retrying the FP32 solve with %s\n", use_magma ? "MAGMA" : "rocSOLVER");
                fflush(stdout);
            }
            /* the room of the other solver's workspace goes to this one (MAGMA
               allocates its own, and a library that cannot allocate ends the
               process) */
            refine_work_release(dev, stream);
        }
        if (cplx)
            hipLaunchKernelGGL(k_f64_to_c32_phase, dim3(grid_for(nn)), dim3(kThreads), 0, stream, a, af, n);
        else
            hipLaunchKernelGGL(k_f64_to_f32, dim3(grid_for(count)), dim3(kThreads), 0, stream, a, af, count);
        refine_check(hipGetLastError(), "convert to FP32");
        if (use_magma) {
            float *w = (float *)dev->host_malloc((size_t)n, sizeof(float), "FP32 eigenvalues");
            int    info;

            refine_check(hipStreamSynchronize(stream), "synchronize");
            info = cplx ? openmx_magma_cheevd_gpu(n, af, w) : openmx_magma_ssyevd_gpu(n, af, w);
            if (info == 0) {
                for (int i = 0; i < n; i++) e0[i] = (double)w[i];
                ok = 1;
            }
            else {
                if (OPENMX_STDOUT_LEVEL(2)) {
                    printf("<eigen_refine_gpu> FP32 MAGMA eigensolver failed (info %d)\n", info);
                    fflush(stdout);
                }
            }
            free(w);
        }
        else if (refine_fp32_rocsolver(dev, stream, cplx, n, af, wf)) {
            float *w = (float *)dev->host_malloc((size_t)n, sizeof(float), "FP32 eigenvalues");

            refine_check(hipMemcpy(w, wf, (size_t)n * sizeof(float), hipMemcpyDeviceToHost), "download eigenvalues");
            for (int i = 0; i < n; i++) e0[i] = (double)w[i];
            free(w);
            ok = 1;
        }
        if (ok) {
            if (attempt == 1) {
                if (OPENMX_STDOUT_LEVEL(2)) {
                    printf("<eigen_refine_gpu> the retried FP32 solve succeeded\n");
                    fflush(stdout);
                }
            }
            return 1;
        }
    }
    if (OPENMX_STDOUT_LEVEL(2)) {
        printf("<eigen_refine_gpu> FP32 eigensolver failed; solving in FP64\n");
        fflush(stdout);
    }
    return 0;
}

/* max offdiag |S| and max |I - G| of the products S (b2) and G (b1), n x kc;
   0 when an entry is not finite */
int refine_maxima(const EigenRefineDevice *dev, hipStream_t stream, int cplx, const double *b1, const double *b2,
                  int n, int kc, double *max_s_out, double *max_r_out)
{
    const size_t count = (size_t)n * (size_t)kc;
    unsigned     blocks = grid_for(count);
    double      *partial = NULL, *host;
    double       max_s = 0.0, max_r = 0.0, finite = 1.0;

    if (blocks > 1024) blocks = 1024;
    if (dev->try_malloc((void **)&partial, (size_t)blocks * 3 * sizeof(double)) != hipSuccess) {
        /* no room even for the check: the caller solves in FP64 */
        if (OPENMX_STDOUT_LEVEL(2)) {
            printf("<eigen_refine_gpu> no room for the refinement check\n");
            fflush(stdout);
        }
        return 0;
    }
    hipLaunchKernelGGL(k_maxima, dim3(blocks), dim3(kThreads), 0, stream, b1, b2, n, kc, cplx ? 2 : 1, partial);
    refine_check(hipGetLastError(), "refinement maxima");
    host = (double *)dev->host_malloc((size_t)blocks * 3, sizeof(double), "refinement maxima");
    refine_check(hipMemcpyAsync(host, partial, (size_t)blocks * 3 * sizeof(double), hipMemcpyDeviceToHost, stream),
                 "download maxima");
    refine_check(hipStreamSynchronize(stream), "synchronize");
    refine_check(hipFree(partial), "hipFree maxima");
    for (unsigned b = 0; b < blocks; b++) {
        max_s = fmax(max_s, host[3 * b]);
        max_r = fmax(max_r, host[3 * b + 1]);
        finite = fmin(finite, host[3 * b + 2]);
    }
    free(host);
    *max_s_out = max_s;
    *max_r_out = max_r;
    return finite == 1.0 && std::isfinite(max_s) && std::isfinite(max_r);
}

int refine_nonfinite(EigenRefineState *st, const EigenRefineProblem *pb, EigenRefineReport *report)
{
    if (OPENMX_STDOUT_LEVEL(2)) {
        printf("<eigen_refine_gpu> invalid refinement values; solving the preserved matrix in FP64\n");
        fflush(stdout);
    }
    refine_blocks_release(st);
    st->basis_valid = 0;
    report->persistent = pb->transient ? 0 : 1;
    return 0;
}

/* Check the final vectors before they replace a. The last Newton update
   and Rayleigh-Ritz rotation run after the last product check. */
int refine_result_finite(const EigenRefineDevice *dev, hipStream_t stream, int width, const double *x,
                         const double *lam, int n, int kc)
{
    const size_t count = (size_t)width * (size_t)n * (size_t)kc;
    int         *flag = NULL;
    int          finite = 1;

    if (dev->try_malloc((void **)&flag, sizeof(int)) != hipSuccess) {
        /* no room even for the check: the caller solves in FP64 */
        if (OPENMX_STDOUT_LEVEL(2)) {
            printf("<eigen_refine_gpu> no room for the refinement check\n");
            fflush(stdout);
        }
        return 0;
    }
    refine_check(hipMemcpyAsync(flag, &finite, sizeof(int), hipMemcpyHostToDevice, stream), "upload flag");
    hipLaunchKernelGGL(k_finite, dim3(grid_for(count)), dim3(kThreads), 0, stream, x, lam, count, kc, flag);
    refine_check(hipGetLastError(), "finite check");
    refine_check(hipMemcpyAsync(&finite, flag, sizeof(int), hipMemcpyDeviceToHost, stream), "download flag");
    refine_check(hipStreamSynchronize(stream), "synchronize");
    refine_check(hipFree(flag), "hipFree flag");
    return finite;
}

/* Rayleigh-Ritz inside one cluster: the m columns xc (n x m, device) span an
   invariant subspace to the accuracy of the refinement but inside it they
   are the FP32 mixtures the refinement only orthonormalized.  The cluster's
   H and overlap (m x m) are formed through z (n x m) and small (2 m^2),
   each product through GEMMul8 when it is large enough for that to pay
   (about 1e9 flops) and in hipBLAS FP64 otherwise; the generalized problem
   goes to LAPACK on the host below 256 states and to hipSOLVER on the device
   beyond (w_dev, m doubles, receives the eigenvalues there).  xc is rotated
   onto the eigenvectors and their eigenvalues go to w.  Returns 0 (xc
   unchanged) if the solve fails. */
int refine_rayleigh_ritz(EigenRefineDevice *dev, hipStream_t stream, int cplx, const void *a, double *xc, int n, int m,
                         double *z, double *small, double *w_dev, double *w)
{
    const int    width = cplx ? 2 : 1;
    const size_t mm = (size_t)m * (size_t)m;
    const size_t nm = (size_t)n * (size_t)m;
    const int    big = (1.0e9 <= 8.0 * (double)n * (double)n * (double)m);
    const int    mid = (1.0e9 <= 8.0 * (double)m * (double)m * (double)n);
    double      *hc = small, *nc = small + width * mm;
    int          info = 0;

    /* z = A X_c, H_c = X_c^H z, N_c = X_c^H X_c */
    refine_check_blas(big ? refine_gemm(dev, cplx, HIPBLAS_OP_N, HIPBLAS_OP_N, n, m, n, a, n, xc, n, 0.0, z, n)
                          : native_gemm(dev, cplx, HIPBLAS_OP_N, HIPBLAS_OP_N, n, m, n, a, n, xc, n, z, n),
                      "A X_c");
    refine_check_blas(mid ? refine_gemm(dev, cplx, HIPBLAS_OP_C, HIPBLAS_OP_N, m, m, n, xc, n, z, n, 0.0, hc, m)
                          : native_gemm(dev, cplx, HIPBLAS_OP_C, HIPBLAS_OP_N, m, m, n, xc, n, z, n, hc, m),
                      "X_c^H A X_c");
    refine_check_blas(mid ? refine_gemm(dev, cplx, HIPBLAS_OP_C, HIPBLAS_OP_N, m, m, n, xc, n, xc, n, 0.0, nc, m)
                          : native_gemm(dev, cplx, HIPBLAS_OP_C, HIPBLAS_OP_N, m, m, n, xc, n, xc, n, nc, m),
                      "X_c^H X_c");
    refine_check(hipStreamSynchronize(stream), "synchronize");

    if (m < 256) {
        double *h = (double *)dev->host_malloc(2 * width * mm, sizeof(double), "cluster Rayleigh-Ritz");
        double *s = h + width * mm;
        double *rwork = (double *)dev->host_malloc((size_t)(3 * m), sizeof(double), "cluster Rayleigh-Ritz");
        double  query[2] = {0.0, 0.0};
        double *work;
        int     itype = 1, dim = m, lwork = -1;

        refine_check(hipMemcpy(h, small, 2 * width * mm * sizeof(double), hipMemcpyDeviceToHost), "download cluster");
        if (cplx) zhegv_(&itype, "V", "L", &dim, h, &dim, s, &dim, w, query, &lwork, rwork, &info);
        else dsygv_(&itype, "V", "L", &dim, h, &dim, s, &dim, w, query, &lwork, &info);
        lwork = (info == 0 && 1.0 <= query[0]) ? (int)query[0] : 3 * m;
        if (lwork < 3 * m) lwork = 3 * m;
        work = (double *)dev->host_malloc((size_t)width * (size_t)lwork, sizeof(double), "cluster Rayleigh-Ritz");
        if (cplx) zhegv_(&itype, "V", "L", &dim, h, &dim, s, &dim, w, work, &lwork, rwork, &info);
        else dsygv_(&itype, "V", "L", &dim, h, &dim, s, &dim, w, work, &lwork, &info);
        free(work);
        free(rwork);
        if (info == 0) {
            refine_check(hipMemcpyAsync(small, h, width * mm * sizeof(double), hipMemcpyHostToDevice, stream),
                         "upload Y");
            refine_check(hipStreamSynchronize(stream), "synchronize");
        }
        free(h);
        if (info != 0) return 0;
    }
    else {
        hipsolverDnHandle_t const solver = refine_solver(stream);
        hipsolverStatus_t         status;
        int32_t                   dev_info = 0;
        int                       lwork = 0;
        void                     *work = NULL;
        int                       own_work = 0;
        size_t                    need;

        if (solver == nullptr) return 0;
        if (cplx)
            status = hipsolverDnZhegvd_bufferSize(solver, HIPSOLVER_EIG_TYPE_1, HIPSOLVER_EIG_MODE_VECTOR,
                                                  HIPBLAS_FILL_MODE_LOWER, m, (hipDoubleComplex *)hc, m,
                                                  (hipDoubleComplex *)nc, m, w_dev, &lwork);
        else
            status = hipsolverDnDsygvd_bufferSize(solver, HIPSOLVER_EIG_TYPE_1, HIPSOLVER_EIG_MODE_VECTOR,
                                                  HIPBLAS_FILL_MODE_LOWER, m, hc, m, nc, m, w_dev, &lwork);
        if (status != HIPSOLVER_STATUS_SUCCESS) return 0;
        if (lwork < 1) lwork = 1;
        need = (size_t)lwork * (size_t)width * sizeof(double);
        if (*dev->d_work_bytes < need) {
            if (dev->try_malloc(&work, need) != hipSuccess) return 0;
            own_work = 1;
        }
        else {
            work = *dev->d_work;
        }
        if (cplx)
            status = hipsolverDnZhegvd(solver, HIPSOLVER_EIG_TYPE_1, HIPSOLVER_EIG_MODE_VECTOR, HIPBLAS_FILL_MODE_LOWER,
                                       m, (hipDoubleComplex *)hc, m, (hipDoubleComplex *)nc, m, w_dev,
                                       (hipDoubleComplex *)work, lwork, dev->d_info);
        else
            status = hipsolverDnDsygvd(solver, HIPSOLVER_EIG_TYPE_1, HIPSOLVER_EIG_MODE_VECTOR, HIPBLAS_FILL_MODE_LOWER,
                                       m, hc, m, nc, m, w_dev, (double *)work, lwork, dev->d_info);
        if (status == HIPSOLVER_STATUS_SUCCESS)
            refine_check(hipMemcpyAsync(&dev_info, dev->d_info, sizeof(int32_t), hipMemcpyDeviceToHost, stream),
                         "download info");
        refine_check(hipStreamSynchronize(stream), "synchronize");
        if (own_work) refine_check(hipFree(work), "hipFree");
        if (status != HIPSOLVER_STATUS_SUCCESS || dev_info != 0) return 0;
        refine_check(hipMemcpy(w, w_dev, (size_t)m * sizeof(double), hipMemcpyDeviceToHost), "download w");
    }

    /* xc <- xc Y through z */
    refine_check_blas(mid ? refine_gemm(dev, cplx, HIPBLAS_OP_N, HIPBLAS_OP_N, n, m, m, xc, n, hc, m, 0.0, z, n)
                          : native_gemm(dev, cplx, HIPBLAS_OP_N, HIPBLAS_OP_N, n, m, m, xc, n, hc, m, z, n),
                      "X_c Y");
    refine_check(hipMemcpyAsync(xc, z, (size_t)width * nm * sizeof(double), hipMemcpyDeviceToDevice, stream),
                 "copy X_c");
    refine_check(hipStreamSynchronize(stream), "synchronize");
    return 1;
}

} // namespace

/* ------------------------------------------------------------------ */

extern "C" int openmx_gpu_fp32_fp64_ratio(void)
{
    static int ratio = -1;

    if (ratio < 0) {
        const char *value = getenv("OPENMX_GPU_FP32_FP64_RATIO");
        hipDeviceProp_t prop;
        int device = 0;

        if (value != NULL && value[0] != '\0' && 0 < atoi(value)) {
            ratio = atoi(value);
        }
        else if (hipGetDevice(&device) != hipSuccess || hipGetDeviceProperties(&prop, device) != hipSuccess) {
            (void)hipGetLastError();
            ratio = 0;
        }
        else if (strncmp(prop.gcnArchName, "gfx9", 4) == 0) {
            /* GCN5 and CDNA: the data-centre parts (gfx906 MI50/60, gfx908,
               gfx90a, gfx94x, gfx95x) run FP64 at half or full rate; the
               consumer Vega parts and APUs at 1/16 */
            const char *arch = prop.gcnArchName;
            int consumer = (strncmp(arch, "gfx900", 6) == 0 || strncmp(arch, "gfx902", 6) == 0 ||
                            strncmp(arch, "gfx904", 6) == 0 || strncmp(arch, "gfx909", 6) == 0 ||
                            strncmp(arch, "gfx90c", 6) == 0);
            ratio = consumer ? 16 : 2;
        }
        else {
            /* GCN before Vega and RDNA (gfx10, gfx11, gfx12): FP64 at 1/16
               of the FP32 rate or less (GEMM 12.5 on gfx1200) */
            ratio = 16;
        }
    }
    return ratio;
}

extern "C" int openmx_eigen_refine_configure(int cplx, int verbose, int *ratio)
{
    const char *value = getenv("OPENMX_EIGEN_REFINE");
    int         iterations = 0;

    *ratio = 0;
    if (!openmx_fp32_enabled()) return 0;
    if (value != NULL && value[0] != '\0') {
        iterations = (0 < atoi(value)) ? atoi(value) : 0;
        if (verbose) {
            printf("<eigen_refine_gpu> OPENMX_EIGEN_REFINE=%s: refined FP32 eigensolver %s\n", value,
                   iterations ? "on" : "off");
            fflush(stdout);
        }
        return iterations;
    }
    {
        const int perf_ratio = openmx_gpu_fp32_fp64_ratio();

        if (8 <= perf_ratio) {
            if (cplx ? openmx_gemmul8ZgemmEnabled() : openmx_gemmul8DgemmEnabled()) {
                iterations = 2;
                *ratio = perf_ratio;
                if (verbose) {
                    printf("<eigen_refine_gpu> refined FP32 eigensolver on by default (FP32:FP64 throughput ratio %d); "
                           "OPENMX_EIGEN_REFINE=0 turns it off\n", perf_ratio);
                    fflush(stdout);
                }
            }
            else if (verbose) {
                printf("<eigen_refine_gpu> the refined FP32 eigensolver is not used: the %s GEMMul8 products are off "
                       "(FP32:FP64 throughput ratio %d)\n", cplx ? "complex" : "real", perf_ratio);
                fflush(stdout);
            }
        }
    }
    return iterations;
}

extern "C" double openmx_eigen_refine_until(void)
{
    const char *value = getenv("OPENMX_EIGEN_REFINE_UNTIL");

    return (value != NULL && 0.0 < atof(value)) ? atof(value) : 0.0;
}

extern "C" int openmx_eigen_refine_warm_enabled(void)
{
    return env_flag("OPENMX_EIGEN_REFINE_WARM", 1);
}

extern "C" int openmx_eigen_refine_warm_band_enabled(void)
{
    return openmx_eigen_refine_warm_enabled() && env_flag("OPENMX_EIGEN_REFINE_WARM_BAND", 1);
}

extern "C" int openmx_eigen_refine_warm_band_period(void)
{
    const char *value = getenv("OPENMX_EIGEN_REFINE_WARM_PERIOD");

    if (value == NULL || value[0] == '\0') return 0;
    return (0 < atoi(value)) ? atoi(value) : 0;
}

extern "C" int openmx_eigen_refine_warm_band_step(int scf_iter)
{
    const int period = openmx_eigen_refine_warm_band_period();

    return openmx_eigen_refine_warm_band_enabled() && !(0 < period && scf_iter % period == 0);
}

extern "C" int openmx_eigen_refine_size_ok(int n)
{
    static long min_n = -1;

    if (min_n < 0) {
        const char *value = getenv("OPENMX_EIGEN_REFINE_MIN_N");

        min_n = 1000;
        if (value != NULL && value[0] != '\0') {
            char *end = NULL;
            long  parsed = strtol(value, &end, 10);

            if (end != value && 0 <= parsed) min_n = parsed;
        }
    }
    return (long)n >= min_n;
}

extern "C" size_t openmx_eigen_refine_fp32_work_bytes(int cplx, int n)
{
    static int    cached_n[2] = {0, 0};
    static size_t cached_bytes[2] = {0, 0};
    const int     slot = cplx ? 1 : 0;
    hipStream_t   stream = nullptr;

    if (n <= 0) return 0;
    if (cached_n[slot] == n) return cached_bytes[slot];
    if (fp32_backend_magma()) {
        cached_bytes[slot] = 0;   /* MAGMA keeps its workspaces on the host */
    }
    else {
        hipsolverDnHandle_t const solver = refine_solver(stream);
        int                       lwork = 0;
        hipsolverStatus_t         status = HIPSOLVER_STATUS_NOT_INITIALIZED;

        if (solver != nullptr) {
            if (cplx)
                status = hipsolverDnCheevd_bufferSize(solver, HIPSOLVER_EIG_MODE_VECTOR, HIPBLAS_FILL_MODE_LOWER, n,
                                                      nullptr, n, nullptr, &lwork);
            else
                status = hipsolverDnSsyevd_bufferSize(solver, HIPSOLVER_EIG_MODE_VECTOR, HIPBLAS_FILL_MODE_LOWER, n,
                                                      nullptr, n, nullptr, &lwork);
        }
        if (status != HIPSOLVER_STATUS_SUCCESS || lwork < 1) {
            /* rocSOLVER's divide and conquer takes about 2.25 n^2 elements */
            lwork = (int)std::min<double>(2147483647.0, 2.3 * (double)n * (double)n + 1.0e6);
        }
        cached_bytes[slot] = (size_t)lwork * (cplx ? sizeof(hipFloatComplex) : sizeof(float));
    }
    cached_n[slot] = n;
    return cached_bytes[slot];
}

/* bytes of the product blocks of a refined solve of maxn wanted states of an
   n x n problem, whether they then take the FP32 scratch (when its region
   holds them) or a buffer of their own: the memory models of the solvers
   budget the latter */
extern "C" size_t openmx_eigen_refine_product_bytes(int cplx, int n, int maxn)
{
    int    blocks;
    size_t block_bytes;

    if (n <= 0 || maxn <= 0) return 0;
    refine_blocks_layout(cplx, n, std::min(maxn, n), &blocks, &block_bytes);
    return (size_t)blocks * block_bytes;
}

extern "C" void openmx_eigen_refine_abandon(EigenRefineState *st)
{
    refine_blocks_release(st);
}

extern "C" void openmx_eigen_refine_state_release(EigenRefineState *st)
{
    refine_blocks_release(st);
    if (st->lam != NULL) refine_check(hipFree(st->lam), "hipFree lam");
    if (st->occ != NULL) refine_check(hipFree(st->occ), "hipFree occ");
    st->lam = st->occ = NULL;
    st->capacity = 0;
    st->basis_valid = 0;
}

extern "C" void openmx_eigen_refine_device_release(EigenRefineDevice *dev)
{
    (void)dev;
    if (g_solver != nullptr) (void)hipsolverDnDestroy(g_solver);
    g_solver = nullptr;
}

/* ------------------------------------------------------------------ */

extern "C" int openmx_eigen_refine_prepare(EigenRefineState *st, EigenRefineDevice *dev,
                                           const EigenRefineProblem *pb, double *e0, EigenRefineReport *report)
{
    const int    cplx = pb->cplx, n = pb->n;
    const size_t nn = (size_t)n * (size_t)n;
    double      *a = (double *)pb->a;
    double      *x = (double *)pb->x;
    float       *af = (float *)pb->fp32;
    float       *wf;
    hipStream_t  stream;

    memset(report, 0, sizeof(*report));
    if (!openmx_fp32_enabled()) {
        st->basis_valid = 0;
        return 0;
    }
    wf = af + (cplx ? 2 * nn : nn);
    stream = refine_stream(dev);
    st->cplx = cplx;

    /* the caller's products that built a may still run on other streams */
    refine_check(hipDeviceSynchronize(), "synchronize before the refined solve");

    /* symmetric or Hermitian from the lower triangle, the part every solve
       reads: the matrix comes from products and is so only to rounding */
    hipLaunchKernelGGL(k_symmetrize, dim3(grid_for(nn)), dim3(kThreads), 0, stream, a, n, cplx);
    refine_check(hipGetLastError(), "symmetrize");

    if (st->capacity < n) {
        openmx_eigen_refine_state_release(st);
        if (dev->try_malloc((void **)&st->lam, (size_t)n * sizeof(double)) != hipSuccess ||
            dev->try_malloc((void **)&st->occ, (size_t)n * sizeof(double)) != hipSuccess) {
            if (st->lam != NULL) refine_check(hipFree(st->lam), "hipFree lam");
            st->lam = NULL;
            if (OPENMX_STDOUT_LEVEL(2)) {
                printf("<eigen_refine_gpu> no room for the refinement eigenvalues; FP64 eigensolver%s\n",
                       pb->transient ? "" : " for the rest of the SCF cycle");
                fflush(stdout);
            }
            report->persistent = pb->transient ? 0 : 1;
            return 0;
        }
        st->capacity = n;
    }

    refine_blocks_release(st);
    if (pb->warm) {
        /* the first refinement step's products on the old vectors (X = X1):
           Y = A X1, S = X^H Y to b2, G = X^H X1 to b1.  Their maxima are the
           residual of the old vectors on the new matrix: above 2e-5 (an
           FP32 solve leaves 4e-6) that solve follows after all, else finish
           starts from these products */
        const int       kc = env_flag("OPENMX_EIGEN_REFINE_ALL", 0) ? n : pb->maxn;
        const long long native0 = openmx_gemmul8NativeCalls(cplx);
        double          max_s = 0.0, max_r = 0.0;
        double         *b1, *b2, *yb;

        refine_check(hipMemcpy(e0, st->lam, (size_t)n * sizeof(double), hipMemcpyDeviceToHost), "download lam");
        if (!refine_blocks_ensure(st, dev, pb, report)) return 0;
        b1 = st->b1;
        b2 = st->b2;
        yb = (st->b3 != NULL) ? st->b3 : b1;
        refine_check_blas(refine_gemm(dev, cplx, HIPBLAS_OP_N, HIPBLAS_OP_N, n, kc, n, a, n, x, n, 0.0, yb, n),
                          "A X1");
        refine_check_blas(refine_gemm(dev, cplx, HIPBLAS_OP_C, HIPBLAS_OP_N, n, kc, n, x, n, yb, n, 0.0, b2, n),
                          "X^H Y");
        refine_check_blas(refine_gemm(dev, cplx, HIPBLAS_OP_C, HIPBLAS_OP_N, n, kc, n, x, n, x, n, 0.0, b1, n),
                          "X^H X1");
        refine_check(hipStreamSynchronize(stream), "synchronize");
        if (pb->defaulted && openmx_gemmul8NativeCalls(cplx) != native0) {
            if (OPENMX_STDOUT_LEVEL(2)) {
                printf("<eigen_refine_gpu> the refinement products fell back to plain hipBLAS FP64 (GEMMul8 workspace "
                       "policy); FP64 eigensolver for the rest of the run\n");
                fflush(stdout);
            }
            refine_blocks_release(st);
            st->basis_valid = 0;
            report->persistent = 2;
            return 0;
        }
        if (!refine_maxima(dev, stream, cplx, b1, b2, n, kc, &max_s, &max_r)) return refine_nonfinite(st, pb, report);
        if (2.0e-5 < max_s || 2.0e-5 < max_r) {
            if (OPENMX_STDOUT_LEVEL(3)) {
                printf("<eigen_refine_gpu> warm start: residual %.1e above 2e-5; FP32 eigensolver\n", fmax(max_s, max_r));
                fflush(stdout);
            }
            refine_blocks_release(st);
            return -1;
        }
        st->products_ready = 1;
    }
    else {
        const size_t count = cplx ? 2 * nn : nn;

        st->basis_valid = 0;
        if (!refine_fp32_solve(dev, stream, cplx, n, a, af, wf, e0)) {
            /* a failure of the solve recurs on the same kind of matrix */
            report->persistent = pb->transient ? 0 : 1;
            return 0;
        }
        report->fp32_solved = 1;
        /* the FP32 solver's workspace is idle from here on: its room goes to
           the products */
        refine_work_release(dev, stream);
        if (cplx)
            hipLaunchKernelGGL(k_c32_to_c64_phase, dim3(grid_for(nn)), dim3(kThreads), 0, stream, af, x, n);
        else
            hipLaunchKernelGGL(k_f32_to_f64, dim3(grid_for(count)), dim3(kThreads), 0, stream, af, x, count);
        refine_check(hipGetLastError(), "convert the basis");
        for (int i = 0; i < n; i++) {
            if (!std::isfinite(e0[i])) return refine_nonfinite(st, pb, report);
        }
        refine_check(hipMemcpyAsync(st->lam, e0, (size_t)n * sizeof(double), hipMemcpyHostToDevice, stream),
                     "upload lam");
        refine_check(hipStreamSynchronize(stream), "synchronize");
    }
    st->anorm = fmax(fabs(e0[0]), fabs(e0[n - 1]));
    return 1;
}

extern "C" int openmx_eigen_refine_finish(EigenRefineState *st, EigenRefineDevice *dev, const EigenRefineProblem *pb,
                                          const double *f0, EigenRefineReport *report)
{
    const int    cplx = pb->cplx, n = pb->n, maxn = pb->maxn, width = cplx ? 2 : 1;
    const int    kc = env_flag("OPENMX_EIGEN_REFINE_ALL", 0) ? n : pb->maxn;
    const size_t nk = (size_t)n * (size_t)kc;
    double      *a = (double *)pb->a;
    double      *x = (double *)pb->x;
    double      *lam = st->lam, *occ = st->occ;
    double      *b1, *b2, *yb;
    double       max_s = 0.0, max_r = 0.0, delta = 0.0, first_delta = 0.0, chain;
    double const anorm = st->anorm;
    long long    native0;
    int          rr_clusters = 0, rr_largest = 0, rr_ok = 1;
    /* the cluster's H and overlap (2 m^2) go into b2 (n kc); beyond 2048
       states the cluster solve would cost more than the FP64 eigensolver
       saves */
    const int    rr_limit = (int)fmin(2048.0, sqrt(0.5 * (double)n * (double)kc));
    hipStream_t  stream = refine_stream(dev);

    for (int i = 0; i < n; i++) {
        if (!std::isfinite(f0[i])) return refine_nonfinite(st, pb, report);
    }
    refine_check(hipMemcpyAsync(occ, f0, (size_t)n * sizeof(double), hipMemcpyHostToDevice, stream),
                 "upload occupations");

    /* the product blocks: the FP32 scratch now that its vectors are in x
       and its eigenvalues on the host, else a buffer of their own */
    if (st->b1 == NULL && !refine_blocks_ensure(st, dev, pb, report)) return 0;
    b1 = st->b1;
    b2 = st->b2;
    yb = (st->b3 != NULL) ? st->b3 : b1;

    native0 = openmx_gemmul8NativeCalls(cplx);
    for (int it = 0; it < pb->iterations; it++) {
        /* Y = A X1 to b3 (b1 without a complement), S = X^H Y to b2,
           G = X^H X1 to b1 (a warm prepare left the first step's) */
        if (!(it == 0 && st->products_ready)) {
            refine_check_blas(refine_gemm(dev, cplx, HIPBLAS_OP_N, HIPBLAS_OP_N, n, kc, n, a, n, x, n, 0.0, yb, n),
                              "A X1");
            refine_check_blas(refine_gemm(dev, cplx, HIPBLAS_OP_C, HIPBLAS_OP_N, n, kc, n, x, n, yb, n, 0.0, b2, n),
                              "X^H Y");
            refine_check_blas(refine_gemm(dev, cplx, HIPBLAS_OP_C, HIPBLAS_OP_N, n, kc, n, x, n, x, n, 0.0, b1, n),
                              "X^H X1");
        }
        if (pb->defaulted && openmx_gemmul8NativeCalls(cplx) != native0) {
            /* plain FP64 products cost more than the FP64 eigensolver on the
               GPUs this default is for */
            if (OPENMX_STDOUT_LEVEL(2)) {
                printf("<eigen_refine_gpu> the refinement products fell back to plain hipBLAS FP64 (GEMMul8 workspace "
                       "policy); FP64 eigensolver for the rest of the run\n");
                fflush(stdout);
            }
            refine_blocks_release(st);
            st->basis_valid = 0;
            report->persistent = 2;
            return 0;
        }

        hipLaunchKernelGGL(k_lambda, dim3(grid_for((size_t)kc)), dim3(kThreads), 0, stream, b1, b2, lam, n, kc, width);
        refine_check(hipGetLastError(), "eigenvalue estimates");

        if (!refine_maxima(dev, stream, cplx, b1, b2, n, kc, &max_s, &max_r)) return refine_nonfinite(st, pb, report);
        delta = 2.0 * (max_s + anorm * max_r);
        if (!std::isfinite(delta)) return refine_nonfinite(st, pb, report);
        if (it == 0) {
            /* a warm start's residual is far below an FP32 solve's, which
               would shrink delta and the cluster chain to nothing: keep the
               FP32 solve's delta as the floor, so nearly degenerate pairs are
               treated as after a cold start (OPENMX_EIGEN_REFINE_WARM_DELTA_FLOOR=0
               drops the floor) */
            if (pb->warm) {
                if (env_flag("OPENMX_EIGEN_REFINE_WARM_DELTA_FLOOR", 1) && delta < st->delta_cold) delta = st->delta_cold;
            }
            else {
                st->delta_cold = delta;
            }
            first_delta = delta;
        }
        delta = first_delta;

        /* E in place of G */
        hipLaunchKernelGGL(k_correction, dim3(grid_for(nk)), dim3(kThreads), 0, stream, b1, b2, lam, occ, n, kc,
                           width, delta);
        refine_check(hipGetLastError(), "correction");

        if (st->b3 != NULL && kc < n) {
            /* The unrefined columns are rotated against the refined ones with
               the same formula (E_rc, refined rows i, unrefined columns j):
               left alone, they keep their FP32-level components along the
               occupied subspace, and the refined columns, whose rotations
               inside the occupied block are left to first order, then settle
               at a fixed point tilted out of the occupied subspace.
               S_rc = Y^H X_c and G_rc = X_r^H X_c (kc x nc) go to b2 and b3,
               E_rc in place of G_rc, then X_c += X_r E_rc.
               OPENMX_EIGEN_REFINE_COMPLEMENT=0 leaves them alone. */
            const int nc = n - kc;
            double   *b3 = st->b3;
            double   *xc = x + (size_t)width * (size_t)kc * (size_t)n;

            refine_check_blas(refine_gemm(dev, cplx, HIPBLAS_OP_C, HIPBLAS_OP_N, kc, nc, n, b3, n, xc, n, 0.0, b2, kc),
                              "Y^H X_c");
            refine_check_blas(refine_gemm(dev, cplx, HIPBLAS_OP_C, HIPBLAS_OP_N, kc, nc, n, x, n, xc, n, 0.0, b3, kc),
                              "X_r^H X_c");
            hipLaunchKernelGGL(k_complement, dim3(grid_for((size_t)kc * (size_t)nc)), dim3(kThreads), 0, stream, b3, b2,
                               lam, occ, kc, nc, width, delta);
            refine_check(hipGetLastError(), "complement correction");
            refine_check_blas(refine_gemm(dev, cplx, HIPBLAS_OP_N, HIPBLAS_OP_N, n, nc, kc, x, n, b3, kc, 1.0, xc, n),
                              "X_c + X_r E_rc");
        }

        /* X1 + X E, through b2 */
        refine_check(hipMemcpyAsync(b2, x, (size_t)width * nk * sizeof(double), hipMemcpyDeviceToDevice, stream),
                     "copy X1");
        refine_check_blas(refine_gemm(dev, cplx, HIPBLAS_OP_N, HIPBLAS_OP_N, n, kc, n, x, n, b1, n, 1.0, b2, n),
                          "X1 + X E");
        refine_check(hipMemcpyAsync(x, b2, (size_t)width * nk * sizeof(double), hipMemcpyDeviceToDevice, stream),
                     "update X1");
    }
    refine_check(hipStreamSynchronize(stream), "synchronize");
    if (pb->defaulted && openmx_gemmul8NativeCalls(cplx) != native0) {
        printf("<eigen_refine_gpu> the refinement products fell back to plain hipBLAS FP64 (GEMMul8 workspace "
               "policy); FP64 eigensolver for the rest of the run\n");
        fflush(stdout);
        refine_blocks_release(st);
        st->basis_valid = 0;
        report->persistent = 2;
        return 0;
    }

    if (!refine_result_finite(dev, stream, width, x, lam, n, kc)) return refine_nonfinite(st, pb, report);

    /* Rayleigh-Ritz inside every cluster of consecutive estimates closer
       than chain whose occupations differ (for Fermi occupations the same
       as holding a partially occupied state, and also true of an orbital
       emptied by index) */
    chain = 30.0 * first_delta;
    {
        double *lh = (double *)dev->host_malloc((size_t)kc, sizeof(double), "refined eigenvalues");
        double *fh = (double *)dev->host_malloc((size_t)kc, sizeof(double), "refined occupations");
        int    *c0s = (int *)dev->host_malloc((size_t)kc, sizeof(int), "refinement clusters");
        int    *ms = (int *)dev->host_malloc((size_t)kc, sizeof(int), "refinement clusters");
        int     clusters = 0;

        refine_check(hipMemcpy(lh, lam, (size_t)kc * sizeof(double), hipMemcpyDeviceToHost), "download lam");
        for (int j = 0; j < kc; j++) fh[j] = pb->occupation(lh[j], j + 1, pb->occupation_ctx);
        for (int c0 = 0; c0 < kc;) {
            int    c1 = c0;
            double f_low = fh[c0], f_high = fh[c0];

            while (c1 + 1 < kc && fabs(lh[c1 + 1] - lh[c1]) <= chain) c1++;
            for (int j = c0; j <= c1; j++) {
                f_low = fmin(f_low, fh[j]);
                f_high = fmax(f_high, fh[j]);
            }
            if (c0 < c1 && 1.0e-12 <= f_high - f_low) {
                c0s[clusters] = c0;
                ms[clusters] = c1 - c0 + 1;
                if (rr_largest < ms[clusters]) rr_largest = ms[clusters];
                clusters++;
            }
            c0 = c1 + 1;
        }
        if (rr_limit < rr_largest) {
            if (OPENMX_STDOUT_LEVEL(2)) {
                printf("<eigen_refine_gpu> a partially occupied cluster of %d states (limit %d) is too large for the "
                       "refined FP32 eigensolver; FP64 eigensolver for the rest of the SCF cycle\n", rr_largest, rr_limit);
                fflush(stdout);
            }
            rr_ok = 0;
        }
        for (int c = 0; rr_ok && c < clusters; c++) {
            /* occ is free by now: it serves as the device eigenvalue buffer */
            if (refine_rayleigh_ritz(dev, stream, cplx, a, x + (size_t)width * (size_t)c0s[c] * (size_t)n, n, ms[c],
                                     b1, b2, occ, lh + c0s[c])) {
                rr_clusters++;
            }
            else {
                if (OPENMX_STDOUT_LEVEL(2)) {
                    printf("<eigen_refine_gpu> the Rayleigh-Ritz step of a cluster of %d states failed; FP64 eigensolver "
                           "for the rest of the SCF cycle\n", ms[c]);
                    fflush(stdout);
                }
                rr_ok = 0;
            }
        }
        if (rr_ok)
            refine_check(hipMemcpy(lam, lh, (size_t)kc * sizeof(double), hipMemcpyHostToDevice), "upload lam");
        if (rr_ok && env_flag("OPENMX_EIGEN_REFINE_CHECK", 0)) {
            /* diagnostic: the residuals |A x_j - lam_j x_j| / |x_j| and the
               orthogonality max_i |x_i^H x_j - delta_ij| of the refined
               columns by occupation class, with plain hipBLAS FP64 products
               (occ is free: it holds the per-column values) */
            double *rh = (double *)dev->host_malloc((size_t)kc, sizeof(double), "refinement check");
            double  res_occ = 0.0, res_part = 0.0, res_empty = 0.0, orth = 0.0;
            int     n_part = 0, worst_occ = -1;

            refine_check_blas(native_gemm(dev, cplx, HIPBLAS_OP_N, HIPBLAS_OP_N, n, kc, n, a, n, x, n, b1, n),
                              "check A X");
            hipLaunchKernelGGL(k_check_residual, dim3(grid_for((size_t)kc)), dim3(kThreads), 0, stream, b1, x, lam,
                               occ, n, kc, width);
            refine_check(hipGetLastError(), "check residual");
            refine_check(hipMemcpyAsync(rh, occ, (size_t)kc * sizeof(double), hipMemcpyDeviceToHost, stream),
                         "download check");
            refine_check(hipStreamSynchronize(stream), "synchronize");
            for (int j = 0; j < kc; j++) {
                if (0.5 < fh[j]) {
                    if (res_occ < rh[j]) worst_occ = j;
                    res_occ = fmax(res_occ, rh[j]);
                }
                else if (1.0e-12 <= fh[j]) {
                    res_part = fmax(res_part, rh[j]);
                    n_part++;
                }
                else {
                    res_empty = fmax(res_empty, rh[j]);
                }
            }
            refine_check_blas(native_gemm(dev, cplx, HIPBLAS_OP_C, HIPBLAS_OP_N, n, kc, n, x, n, x, n, b2, n),
                              "check X^H X");
            hipLaunchKernelGGL(k_check_orthogonality, dim3(grid_for((size_t)kc)), dim3(kThreads), 0, stream, b2, occ,
                               n, kc, width);
            refine_check(hipGetLastError(), "check orthogonality");
            refine_check(hipMemcpyAsync(rh, occ, (size_t)kc * sizeof(double), hipMemcpyDeviceToHost, stream),
                         "download check");
            refine_check(hipStreamSynchronize(stream), "synchronize");
            for (int j = 0; j < kc; j++) orth = fmax(orth, rh[j]);
            printf("<eigen_refine_gpu> check: %d columns; residual max occupied %.1e (column %d), partial %.1e (%d "
                   "columns), empty %.1e; orthogonality max %.1e\n",
                   kc, res_occ, worst_occ + 1, res_part, n_part, res_empty, orth);
            fflush(stdout);
            free(rh);
        }
        free(lh);
        free(fh);
        free(c0s);
        free(ms);
    }
    if (!rr_ok) {
        /* a still holds the symmetric matrix: the caller solves it in FP64 */
        refine_blocks_release(st);
        st->basis_valid = 0;
        report->persistent = 1;
        return 0;
    }

    if (rr_clusters && !refine_result_finite(dev, stream, width, x, lam, n, kc)) return refine_nonfinite(st, pb, report);

    refine_check(hipMemcpyAsync(a, x, (size_t)width * (size_t)n * (size_t)maxn * sizeof(double),
                                hipMemcpyDeviceToDevice, stream),
                 "copy the refined vectors");
    refine_check(hipMemcpyAsync(pb->w, lam, (size_t)maxn * sizeof(double), hipMemcpyDeviceToDevice, stream),
                 "copy the refined eigenvalues");
    refine_check(hipStreamSynchronize(stream), "synchronize");

    refine_blocks_release(st);
    st->basis_valid = 1;
    st->basis_n = n;

    report->columns = kc;
    report->max_r = max_r;
    report->max_s = max_s;
    report->delta = first_delta;
    report->chain = chain;
    report->rr_clusters = rr_clusters;
    report->rr_largest = rr_largest;
    return 1;
}

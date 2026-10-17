#include "openmx_stdout_level.h"
#include <magma_v2.h>
#include "hip_runtime_compat.h"
#include "openmx_gpu_arch_policy.h"
#include "openmx_magma_lock.h"
#include "openmx_mgpu_runtime.h"
#include <vector>

#include <hipsolver/hipsolver.h>

#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <strings.h>

extern "C" int MYID_MPI_COMM_WORLD;
/* Ranks sharing this rank's GPU (set_hip_default_device_from_local_rank.c).
   Weak because this object is also linked into standalone testers without
   the device-selection object; one rank per device is assumed there. */
extern "C" __attribute__((weak)) int openmx_gpu_ranks_per_device_noncollective(void);

namespace {

std::once_flag g_magma_once;
std::mutex    &g_magma_mutex = openmx_magma_runtime_mutex();
int            g_magma_init_info = MAGMA_SUCCESS;

double      *g_host_matrix = nullptr;
size_t       g_host_matrix_elems = 0;
double      *g_work = nullptr;
magma_int_t  g_lwork = 0;
magma_int_t *g_iwork = nullptr;
magma_int_t  g_liwork = 0;

magmaDoubleComplex *g_z_host_matrix = nullptr;
size_t              g_z_host_matrix_elems = 0;
magmaDoubleComplex *g_z_work = nullptr;
magma_int_t         g_z_lwork = 0;
double             *g_z_rwork = nullptr;
magma_int_t         g_z_lrwork = 0;
magma_int_t        *g_z_iwork = nullptr;
magma_int_t         g_z_liwork = 0;

/* the device copy of the input kept for the retry of small solves */
void  *g_backup = nullptr;
size_t g_backup_bytes = 0;

int ensure_magma_initialized()
{
    std::call_once(g_magma_once, []() {
        g_magma_init_info = magma_init();
        if (g_magma_init_info != MAGMA_SUCCESS) {
            std::fprintf(stderr, "<Cluster> MAGMA initialization failed: info=%d\n", g_magma_init_info);
            std::fflush(stderr);
        }
    });
    return g_magma_init_info;
}

/* MAGMA Ozaki-II (GEMMul8) routing of the dense eigensolves.  The _ozaki2
   drivers delegate to the stock MAGMA algorithms and only route the BLAS-3
   phases (syr2k/her2k, larfb gemm/trmm, dlaex3 gemm) through GEMMul8 INT8
   emulation inside a scoped activation; OPENMX_MAGMA_OZAKI2=0 restores the
   stock drivers bit-identically.  The two defaults follow the tuning note
   (third_party/MAGMA_OZAKI2_NOTES.md): 1-stage because the 2-stage bulge
   chasing starves under many busy MPI ranks, and a 4-way split of the
   percent-of-VRAM workspace cap because only the dense root rank solves. */
int use_magma_ozaki2()
{
    static int use = -1;

    if (use < 0) {
        const char *env = std::getenv("OPENMX_MAGMA_OZAKI2");
        use = (env == nullptr || env[0] == '\0') ? 1 : (std::atoi(env) != 0);
        if (use) {
            setenv("MAGMA_OZAKI2_ALGO", "1stage", 0);
            setenv("MAGMA_OZAKI2_LOCAL_RANKS", "4", 0);
            /* A modest per-process workspace plus a 1 GiB free guard routes
               3x as many backtransform GEMMs on a 16 GB card shared by 16
               ranks than the adapter's conservative defaults; GEMMul8's
               memory-saving mode blocks any GEMM to fit the cap. */
            setenv("MAGMA_OZAKI2_MAX_WS_MB", "512", 0);
            setenv("MAGMA_OZAKI2_MIN_FREE_MB", "1024", 0);
        }
    }
    return use;
}

int env_flag(const char *name, int dflt)
{
    const char *v = std::getenv(name);

    if (v == nullptr || v[0] == '\0') return dflt;
    return std::atoi(v) != 0;
}

/* a nonnegative int from the environment; unset or malformed: dflt */
int env_count(const char *name, int dflt)
{
    const char *v = std::getenv(name);
    char       *end = nullptr;
    long        value;

    if (v == nullptr || v[0] == '\0') return dflt;
    errno = 0;
    value = std::strtol(v, &end, 10);
    if (errno != 0 || end == v || *end != '\0' || value < 0 || value > 1000000000L) {
        std::fprintf(stderr, "OpenMX: invalid %s=%s; using %d.\n", name, v, dflt);
        return dflt;
    }
    return (int)value;
}

/* Retry of a failed dense FP64 eigensolve (the AMD counterpart of the NVIDIA
   build's native-FP64 retry of an emulated cuSOLVER solve, 8d3a0931): the
   input is copied before the solve (to the device when this rank's share of
   the free memory allows, else to the host, else no retry) and a failed
   solve is solved again from the copy with the next backend of the chain:
   MAGMA Ozaki-II -> stock MAGMA (on no convergence, info > 0) and rocSOLVER
   -> MAGMA (on any failure, non-finite vectors included).
   OPENMX_MAGMA_OZAKI2_RETRY=0 keeps neither a copy nor a retry;
   OPENMX_GPUSOLVER_RETRY_TEST=1 (or the NVIDIA name OPENMX_CUSOLVER_RETRY_TEST)
   solves every solve again with the next backend (testing; about twice the
   cost).  0: off, 1: on failure, 2: test */
int eigen_retry_policy()
{
    static const int policy = []() {
        if (!env_flag("OPENMX_MAGMA_OZAKI2_RETRY", 1)) return 0;
        return (env_flag("OPENMX_GPUSOLVER_RETRY_TEST", 0) || env_flag("OPENMX_CUSOLVER_RETRY_TEST", 0)) ? 2 : 1;
    }();
    return policy;
}

/* OPENMX_GPU_EIGENSOLVER: 1 hipsolver, 0 magma, -1 unset or anything else (auto) */
int eigensolver_env_choice()
{
    static const int choice = []() {
        const char *env = std::getenv("OPENMX_GPU_EIGENSOLVER");

        if (env != nullptr) {
            if (strcasecmp(env, "hipsolver") == 0) return 1;
            if (strcasecmp(env, "magma") == 0) return 0;
        }
        return -1;
    }();
    return choice;
}

/* OPENMX_GPU_EIGENSOLVER_MAGMA_MIN_N (default 0, off): in the automatic
   backend choice the dense FP64 solves of order below it use rocSOLVER
   instead of MAGMA (MAGMA's {sy,he}trd reduce n < 3000 entirely on the CPU),
   for measuring the crossover in situ */
int eigensolver_magma_min_n()
{
    static const int min_n = env_count("OPENMX_GPU_EIGENSOLVER_MAGMA_MIN_N", 0);
    return min_n;
}

/* The size override in the automatic policy; explicit backend choices
   take precedence.  Direct dense entry points must also apply the device
   policy through openmx_gpu_eigen_backend, not this size override alone. */
int magma_entry_reroute(int n)
{
    return eigensolver_env_choice() < 0 && 0 < eigensolver_magma_min_n() && n < eigensolver_magma_min_n();
}

/* OPENMX_GPUSOLVER_CACHE_MB (alias OPENMX_CUSOLVER_CACHE_MB, the NVIDIA
   name; default 64): device scratch the generic wrappers keep per thread
   between solves, and the largest retry copy kept here */
size_t gpusolver_cache_limit_bytes()
{
    static const size_t limit = []() {
        const char *s = std::getenv("OPENMX_GPUSOLVER_CACHE_MB");
        size_t      bytes = static_cast<size_t>(64) << 20;

        if (s == nullptr || s[0] == '\0') s = std::getenv("OPENMX_CUSOLVER_CACHE_MB");
        if (s != nullptr && s[0] != '\0' && s[0] != '-') {
            char              *end = nullptr;
            unsigned long long mb;

            errno = 0;
            mb = std::strtoull(s, &end, 10);
            if (errno == 0 && end != s && *end == '\0' && mb <= SIZE_MAX / (1024u * 1024u))
                bytes = static_cast<size_t>(mb) << 20;
        }
        return bytes;
    }();
    return limit;
}

int ensure_host_matrix(size_t elems)
{
    double *new_ptr = nullptr;

    if (elems <= g_host_matrix_elems) {
        return MAGMA_SUCCESS;
    }
    if (magma_dmalloc_cpu(&new_ptr, elems) != MAGMA_SUCCESS) {
        return MAGMA_ERR_HOST_ALLOC;
    }
    if (g_host_matrix != nullptr) {
        magma_free_cpu(g_host_matrix);
    }
    g_host_matrix = new_ptr;
    g_host_matrix_elems = elems;
    return MAGMA_SUCCESS;
}

int ensure_work(magma_int_t lwork, magma_int_t liwork)
{
    if (lwork > g_lwork) {
        double *new_work = nullptr;

        if (magma_dmalloc_cpu(&new_work, static_cast<size_t>(lwork)) != MAGMA_SUCCESS) {
            return MAGMA_ERR_HOST_ALLOC;
        }
        if (g_work != nullptr) {
            magma_free_cpu(g_work);
        }
        g_work = new_work;
        g_lwork = lwork;
    }

    if (liwork > g_liwork) {
        magma_int_t *new_iwork = nullptr;

        if (magma_imalloc_cpu(&new_iwork, static_cast<size_t>(liwork)) != MAGMA_SUCCESS) {
            return MAGMA_ERR_HOST_ALLOC;
        }
        if (g_iwork != nullptr) {
            magma_free_cpu(g_iwork);
        }
        g_iwork = new_iwork;
        g_liwork = liwork;
    }

    return MAGMA_SUCCESS;
}

int ensure_z_host_matrix(size_t elems)
{
    magmaDoubleComplex *new_ptr = nullptr;

    if (elems <= g_z_host_matrix_elems) {
        return MAGMA_SUCCESS;
    }
    if (magma_zmalloc_cpu(&new_ptr, elems) != MAGMA_SUCCESS) {
        return MAGMA_ERR_HOST_ALLOC;
    }
    if (g_z_host_matrix != nullptr) {
        magma_free_cpu(g_z_host_matrix);
    }
    g_z_host_matrix = new_ptr;
    g_z_host_matrix_elems = elems;
    return MAGMA_SUCCESS;
}

int ensure_z_work(magma_int_t lwork, magma_int_t lrwork, magma_int_t liwork)
{
    if (lwork > g_z_lwork) {
        magmaDoubleComplex *new_work = nullptr;

        if (magma_zmalloc_cpu(&new_work, static_cast<size_t>(lwork)) != MAGMA_SUCCESS) {
            return MAGMA_ERR_HOST_ALLOC;
        }
        if (g_z_work != nullptr) {
            magma_free_cpu(g_z_work);
        }
        g_z_work = new_work;
        g_z_lwork = lwork;
    }

    if (lrwork > g_z_lrwork) {
        double *new_rwork = nullptr;

        if (magma_dmalloc_cpu(&new_rwork, static_cast<size_t>(lrwork)) != MAGMA_SUCCESS) {
            return MAGMA_ERR_HOST_ALLOC;
        }
        if (g_z_rwork != nullptr) {
            magma_free_cpu(g_z_rwork);
        }
        g_z_rwork = new_rwork;
        g_z_lrwork = lrwork;
    }

    if (liwork > g_z_liwork) {
        magma_int_t *new_iwork = nullptr;

        if (magma_imalloc_cpu(&new_iwork, static_cast<size_t>(liwork)) != MAGMA_SUCCESS) {
            return MAGMA_ERR_HOST_ALLOC;
        }
        if (g_z_iwork != nullptr) {
            magma_free_cpu(g_z_iwork);
        }
        g_z_iwork = new_iwork;
        g_z_liwork = liwork;
    }

    return MAGMA_SUCCESS;
}

hipsolverDnHandle_t g_hs_handle = nullptr;
void               *g_hs_work = nullptr;     /* hipDoubleComplex or double */
size_t              g_hs_work_bytes = 0;
int                 g_hs_query_n = 0;        /* (n, type) of the last workspace query */
int                 g_hs_query_cplx = -1;
int                 g_hs_query_lwork = 0;
double             *g_hs_dW = nullptr;
double             *g_hs_hW = nullptr;
int                 g_hs_W_n = 0;
int                *g_hs_dinfo = nullptr;    /* [0] info, [1] non-finite flag */

int ensure_hipsolver_ready(int n)
{
    if (g_hs_handle == nullptr) {
        if (hipsolverDnCreate(&g_hs_handle) != HIPSOLVER_STATUS_SUCCESS) {
            g_hs_handle = nullptr;
            return -1;
        }
    }
    if (g_hs_dinfo == nullptr) {
        if (hipMalloc((void **)&g_hs_dinfo, 2 * sizeof(int)) != hipSuccess) {
            (void)hipGetLastError();
            g_hs_dinfo = nullptr;
            return -1;
        }
    }
    if (n > g_hs_W_n) {
        if (g_hs_dW != nullptr) hipFree(g_hs_dW);
        if (g_hs_hW != nullptr) std::free(g_hs_hW);
        g_hs_dW = nullptr;
        g_hs_hW = nullptr;
        g_hs_W_n = 0;
        if (hipMalloc((void **)&g_hs_dW, sizeof(double) * (size_t)n) != hipSuccess) {
            (void)hipGetLastError();
            g_hs_dW = nullptr;
            return -1;
        }
        g_hs_hW = (double *)std::malloc(sizeof(double) * (size_t)n);
        if (g_hs_hW == nullptr) {
            return -1;
        }
        g_hs_W_n = n;
    }
    return 0;
}

/* free device memory, or 0 when it cannot be read */
size_t device_free_bytes()
{
    size_t free_bytes = 0, total_bytes = 0;

    if (hipMemGetInfo(&free_bytes, &total_bytes) != hipSuccess) {
        (void)hipGetLastError();
        return 0;
    }
    return free_bytes;
}

} // namespace

/* Backend selection for the dense GPU eigensolvers.  rocSOLVER's zheevd runs
   entirely on the GPU and is far faster than hybrid MAGMA on CDNA server parts
   (MI100/MI200/MI300 report gfx9xx), while on RDNA consumer parts
   (gfx10xx/11xx/12xx, e.g. RX 9060XT = gfx1200) MAGMA is the faster of the
   two.  Override with OPENMX_GPU_EIGENSOLVER=hipsolver|magma (anything else,
   including "auto", keeps the detection). */
extern "C" int openmx_gpu_eigensolver_use_hipsolver(void)
{
    static int cached = -1;

    if (cached >= 0) {
        return cached;
    }

    if (eigensolver_env_choice() >= 0) {
        cached = eigensolver_env_choice();
        return cached;
    }

    hipDeviceProp_t prop;
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess || hipGetDeviceProperties(&prop, dev) != hipSuccess) {
        cached = 0;
        return cached;
    }
    cached = (std::strncmp(prop.gcnArchName, "gfx9", 4) == 0) ? 1 : 0;
    return cached;
}

extern "C" int openmx_gpu_is_mi300a(void)
{
    return openmx_gpu_device_is_mi300a();
}

/* Size-aware backend of a dense FP64 eigensolve of order n (maxn wanted
   pairs, complex when cplx): 1 rocSOLVER (openmx_hipsolver_zheevd_gpu), 0
   MAGMA.  The choice above, except that in the automatic mode an n below
   OPENMX_GPU_EIGENSOLVER_MAGMA_MIN_N (default 0: none, today's choice) also
   selects rocSOLVER.  The MAGMA entry points apply the same threshold on
   their own, so both the device policy and the knob reach the cluster, DC,
   DC-LNO and Krylov solvers without changes there. */
extern "C" int openmx_gpu_eigen_backend(int n, int maxn, int cplx)
{
    (void)maxn;
    (void)cplx;
    if (openmx_gpu_eigensolver_use_hipsolver()) {
        return 1;
    }
    return magma_entry_reroute(n);
}

/* For the generic wrappers (gpusolver_Syevdx.c): 1 when the request for the
   lowest maxn of n eigenpairs is solved by rocSOLVER's full D&C ({sy,he}evd,
   of the n eigenpairs the first maxn are used), 0 for the subset driver
   ({sy,he}evdx).  The full D&C serves every complete spectrum: on gfx1200 it
   takes 0.26 s against 0.37 s of the subset driver at n = 1000 and 0.92
   against 2.30 s at n = 2000 (complex; real 0.73 against 1.68 s).  A partial
   request keeps the subset driver (also NVIDIA's automatic policy) unless n
   reaches OPENMX_GPUSOLVER_PARTIAL_EVD_MIN_N (default 0: never); the full
   solve was also faster for half the spectrum from n = 1000 and for 100 of
   2000 pairs (0.92 against 2.07 s), slower for 40 of 400. */
extern "C" int openmx_gpusolver_use_evd(int n, int maxn)
{
    static const int partial_min_n = env_count("OPENMX_GPUSOLVER_PARTIAL_EVD_MIN_N", 0);

    if (n == maxn) {
        return 1;
    }
    return 0 < partial_min_n && partial_min_n <= n;
}

/* the retry policy above, for gpusolver_Syevdx.c */
extern "C" int openmx_gpu_eigen_retry_policy(void)
{
    return eigen_retry_policy();
}

/* the scratch bound above, for gpusolver_Syevdx.c */
extern "C" size_t openmx_gpusolver_cache_limit(void)
{
    return gpusolver_cache_limit_bytes();
}

/* True when the GPU is an APU sharing physical memory with the CPU (MI300A):
   host arrays are device-accessible and host<->device staging copies are pure
   waste.  Discrete CDNA parts (MI300X, MI2xx) report integrated=0 and still
   need real copies.  Override with OPENMX_GPU_ASSUME_APU=0/1. */
extern "C" int openmx_gpu_is_apu(void)
{
    static int cached = -1;

    if (cached >= 0) {
        return cached;
    }

    const char *env = std::getenv("OPENMX_GPU_ASSUME_APU");
    if (env != nullptr && *env != '\0') {
        cached = (std::atoi(env) != 0) ? 1 : 0;
        return cached;
    }

    hipDeviceProp_t prop;
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess || hipGetDeviceProperties(&prop, dev) != hipSuccess) {
        cached = 0;
        return cached;
    }
    cached = (prop.integrated != 0) ? 1 : 0;

    if (std::getenv("OPENMX_GPU_VERBOSE") != nullptr && MYID_MPI_COMM_WORLD == 0) {
        std::fprintf(stderr, "openmx: GPU '%s' integrated(APU)=%d\n", prop.gcnArchName, cached);
        std::fflush(stderr);
    }
    return cached;
}

/* rocSOLVER's Hermitian divide and conquer (zheevd of ROCm 7.2) returns
   info 0 with NaN vectors for a complex matrix whose elements are all real
   (the non-collinear Hamiltonian without spin-orbit coupling, at real k) and
   for an exactly block-diagonal one (a non-magnetic non-collinear one).  The
   solve therefore takes the unitary similarity D^H A D, D = diag(d_k),
   d_k = exp(2 pi i frac(k phi)), with every exactly zero off-diagonal element
   set to about 1e-100, and returns D V: same eigenvalues, the vectors of A.
   The dither was 1e-200 until two-block matrices (two identical dense blocks
   at n = 100, tridiagonal blocks at n = 100 to 2000) still came back with
   NaN vectors, probably because its square underflows and the reduction
   meets exact zeros again; with 1e-100 (square 1e-200) every case was
   finite, and the change of the eigenvalues stays far below rounding.  (The
   subset driver zheevdx returned finite vectors for all-real and
   block-diagonal matrices at n = 300 and 1200.)  An exactly diagonal matrix
   still often comes back with NaN vectors (distinct entries at n = 100 to
   200, the zero matrix at every n from 100 to 1000), whatever the dither
   from 1e-200 to 1e-20; the finiteness check catches it and the solve goes
   to the MAGMA retry (eigen_retry_policy), so with
   OPENMX_MAGMA_OZAKI2_RETRY=0 it fails. */
namespace {

__device__ inline void zheevd_phase(int k, double *c, double *s)
{
    double t = (double)k * 0.6180339887498949;

    t -= floor(t);
    sincospi(2.0 * t, s, c);
}

__global__ void k_zheevd_phase_in(double *a, int n)
{
    const size_t nn = (size_t)n * (size_t)n;

    for (size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x; idx < nn; idx += (size_t)gridDim.x * blockDim.x) {
        const int    row = (int)(idx % (size_t)n), col = (int)(idx / (size_t)n);
        const double re = a[2 * idx], im = a[2 * idx + 1];
        double       cr, sr, cc, sc;

        zheevd_phase(row, &cr, &sr);
        zheevd_phase(col, &cc, &sc);
        {
            const double pr = cr * cc + sr * sc, pi = cr * sc - sr * cc;   /* conj(d_row) d_col */
            double       r2 = re * pr - im * pi, i2 = re * pi + im * pr;

            if (r2 == 0.0 && i2 == 0.0 && row != col) r2 = 1.0e-100 * (double)(1 + (unsigned)(row * 7 + col * 13) % 17u);
            a[2 * idx] = r2;
            a[2 * idx + 1] = i2;
        }
    }
}

/* v <- D v for the first cols columns; sets *bad on a non-finite entry */
__global__ void k_zheevd_phase_out(double *v, int n, int cols, int *bad)
{
    const size_t count = (size_t)n * (size_t)cols;

    for (size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x; idx < count;
         idx += (size_t)gridDim.x * blockDim.x) {
        const double vr = v[2 * idx], vi = v[2 * idx + 1];
        double       c, s;

        if (!isfinite(vr) || !isfinite(vi)) *bad = 1;
        zheevd_phase((int)(idx % (size_t)n), &c, &s);
        v[2 * idx] = c * vr - s * vi;
        v[2 * idx + 1] = c * vi + s * vr;
    }
}

/* sets *bad when one of the count values is not finite */
__global__ void k_flag_nonfinite(const double *x, size_t count, int *bad)
{
    for (size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x; idx < count;
         idx += (size_t)gridDim.x * blockDim.x) {
        if (!isfinite(x[idx])) *bad = 1;
    }
}

/* sets *bad when an element of the lower triangle of the n x n matrix a
   (complex when cplx) is not finite: the part the solvers read */
__global__ void k_flag_nonfinite_lower(const double *a, int n, int cplx, int *bad)
{
    const size_t nn = (size_t)n * (size_t)n;

    for (size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x; idx < nn; idx += (size_t)gridDim.x * blockDim.x) {
        if ((int)(idx % (size_t)n) < (int)(idx / (size_t)n)) continue;
        if (!isfinite(a[cplx ? 2 * idx : idx]) || (cplx && !isfinite(a[2 * idx + 1]))) *bad = 1;
    }
}

unsigned zheevd_grid(size_t count)
{
    size_t blocks = (count + 255) / 256;

    if (blocks > 65535u * 16u) blocks = 65535u * 16u;
    if (blocks < 1) blocks = 1;
    return (unsigned)blocks;
}

/* Clears *d_bad, then sets it when one of the first cols eigenvector
   columns of d_V (n rows; complex when cplx) or one of the first nw values
   of d_w (may be NULL) is not finite.  With phase the complex vectors are
   also multiplied by D (k_zheevd_phase_out).  Stream-ordered on stream. */
hipError_t eigen_check(int cplx, int phase, void *d_V, int n, int cols, const double *d_w, int nw, int *d_bad,
                       hipStream_t stream)
{
    const size_t count = (size_t)n * (size_t)cols * (cplx ? 2u : 1u);

    (void)hipGetLastError();
    if (hipMemsetAsync(d_bad, 0, sizeof(int), stream) != hipSuccess) return hipGetLastError();
    if (cplx && phase)
        hipLaunchKernelGGL(k_zheevd_phase_out, dim3(zheevd_grid((size_t)n * (size_t)cols)), dim3(256), 0, stream,
                           (double *)d_V, n, cols, d_bad);
    else
        hipLaunchKernelGGL(k_flag_nonfinite, dim3(zheevd_grid(count)), dim3(256), 0, stream, (const double *)d_V,
                           count, d_bad);
    if (d_w != nullptr && nw > 0)
        hipLaunchKernelGGL(k_flag_nonfinite, dim3(zheevd_grid((size_t)nw)), dim3(256), 0, stream, d_w, (size_t)nw,
                           d_bad);
    return hipGetLastError();
}

} // namespace

/* The phase similarity and the finiteness test for the generic wrappers
   (gpusolver_Syevdx.c), which drive hipSOLVER on their own handle and
   stream: phase_in before hipsolverDnZheevd on the n x n matrix (leading
   dimension n), check after the solve (with phase = 1 completing the
   similarity on the first cols vectors).  Both return a hipError_t. */
extern "C" int openmx_gpu_zheevd_phase_in(void *d_A, int n, hipStream_t stream)
{
    if (d_A == nullptr || n <= 0) {
        return (int)hipErrorInvalidValue;
    }
    (void)hipGetLastError();
    hipLaunchKernelGGL(k_zheevd_phase_in, dim3(zheevd_grid((size_t)n * (size_t)n)), dim3(256), 0, stream,
                       (double *)d_A, n);
    return (int)hipGetLastError();
}

extern "C" int openmx_gpu_eigen_check(int cplx, int phase, void *d_V, int n, int cols, const double *d_w,
                                      int *d_bad, hipStream_t stream)
{
    if (d_V == nullptr || d_bad == nullptr || n <= 0 || cols <= 0 || cols > n) {
        return (int)hipErrorInvalidValue;
    }
    return (int)eigen_check(cplx, phase, d_V, n, cols, d_w, cols, d_bad, stream);
}

/* Clears *d_bad, then sets it when the lower triangle of the input d_A
   (n x n, complex when cplx) has a non-finite element.  A solver can turn
   such an input into finite eigenpairs of another matrix (MAGMA dsyevdx
   treated a NaN as zero), so the wrappers refuse it before solving. */
extern "C" int openmx_gpu_eigen_check_input(int cplx, const void *d_A, int n, int *d_bad, hipStream_t stream)
{
    if (d_A == nullptr || d_bad == nullptr || n <= 0) {
        return (int)hipErrorInvalidValue;
    }
    (void)hipGetLastError();
    if (hipMemsetAsync(d_bad, 0, sizeof(int), stream) != hipSuccess) return (int)hipGetLastError();
    hipLaunchKernelGGL(k_flag_nonfinite_lower, dim3(zheevd_grid((size_t)n * (size_t)n)), dim3(256), 0, stream,
                       (const double *)d_A, n, cplx, d_bad);
    return (int)hipGetLastError();
}

namespace {

/* rocSOLVER's full divide and conquer on d_A (n x n, lower triangle):
   all n eigenvectors (ascending) in d_A, of which callers consume the first
   maxn columns, and the lowest maxn eigenvalues in w (host).  Complex
   matrices go through the phase similarity; a non-finite eigenvector element
   or eigenvalue is a failure (MAGMA_ERR_UNKNOWN), as is any nonzero info.
   *touched is set once d_A may have changed.  With preflight the solve is
   declined (MAGMA_ERR_DEVICE_ALLOC, d_A untouched) when less than 512 MiB
   of device memory would stay free beside the workspace: rocSOLVER's BLAS
   may run hipBLASLt, and an allocation failing there ends the process. */
int rocsolver_evd_locked(int cplx, int n, int maxn, void *d_A, double *w, int *mout_out, int *touched,
                         int preflight)
{
    static const int detail = env_flag("OPENMX_GPU_EIGENSOLVER_PROFILE", 0);
    const size_t width = cplx ? sizeof(hipDoubleComplex) : sizeof(double);
    const char  *name = cplx ? "zheevd" : "dsyevd";
    hipsolverStatus_t status;
    size_t       work_bytes;
    int          info = 0;
    int          bad = 0;
    auto         phase_start = detail ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    double       setup_seconds = 0.0, phase_in_seconds = 0.0, solve_seconds = 0.0, check_seconds = 0.0;

    if (ensure_hipsolver_ready(n) != 0) {
        return MAGMA_ERR_HOST_ALLOC;
    }

    if (n != g_hs_query_n || cplx != g_hs_query_cplx) {
        int lwork = 0;

        if (cplx)
            status = hipsolverDnZheevd_bufferSize(g_hs_handle, HIPSOLVER_EIG_MODE_VECTOR, HIPSOLVER_FILL_MODE_LOWER, n,
                                                  (hipDoubleComplex *)d_A, n, g_hs_dW, &lwork);
        else
            status = hipsolverDnDsyevd_bufferSize(g_hs_handle, HIPSOLVER_EIG_MODE_VECTOR, HIPSOLVER_FILL_MODE_LOWER, n,
                                                  (double *)d_A, n, g_hs_dW, &lwork);
        if (status != HIPSOLVER_STATUS_SUCCESS) {
            return (int)status;
        }
        if (lwork < 1) {
            lwork = 1;
        }
        g_hs_query_n = n;
        g_hs_query_cplx = cplx;
        g_hs_query_lwork = lwork;
    }

    work_bytes = width * (size_t)g_hs_query_lwork;
    if (preflight) {
        const size_t need = (work_bytes > g_hs_work_bytes ? work_bytes : 0) + (static_cast<size_t>(512) << 20);

        if (device_free_bytes() < need) {
            return MAGMA_ERR_DEVICE_ALLOC;
        }
    }
    if (work_bytes > g_hs_work_bytes) {
        if (g_hs_work != nullptr) {
            hipFree(g_hs_work);
            g_hs_work = nullptr;
            g_hs_work_bytes = 0;
        }
        if (hipMalloc(&g_hs_work, work_bytes) != hipSuccess) {
            (void)hipGetLastError();
            g_hs_work = nullptr;
            return MAGMA_ERR_DEVICE_ALLOC;
        }
        g_hs_work_bytes = work_bytes;
    }

    /* Only the explicit profiler serializes phase boundaries.  The normal
       solve retains its existing stream ordering and synchronization. */
    if (detail) {
        if (hipDeviceSynchronize() != hipSuccess) return MAGMA_ERR_UNKNOWN;
        const auto now = std::chrono::steady_clock::now();
        setup_seconds = std::chrono::duration<double>(now - phase_start).count();
        phase_start = now;
    }

    *touched = 1;
    if (cplx) {
        (void)hipGetLastError();
        hipLaunchKernelGGL(k_zheevd_phase_in, dim3(zheevd_grid((size_t)n * (size_t)n)), dim3(256), 0, 0,
                           (double *)d_A, n);
        if (hipGetLastError() != hipSuccess) {
            return MAGMA_ERR_UNKNOWN;
        }
    }
    if (detail) {
        if (hipDeviceSynchronize() != hipSuccess) return MAGMA_ERR_UNKNOWN;
        const auto now = std::chrono::steady_clock::now();
        phase_in_seconds = std::chrono::duration<double>(now - phase_start).count();
        phase_start = now;
    }
    if (cplx) {
        status = hipsolverDnZheevd(g_hs_handle, HIPSOLVER_EIG_MODE_VECTOR, HIPSOLVER_FILL_MODE_LOWER, n,
                                   (hipDoubleComplex *)d_A, n, g_hs_dW, (hipDoubleComplex *)g_hs_work,
                                   g_hs_query_lwork, g_hs_dinfo);
    }
    else {
        status = hipsolverDnDsyevd(g_hs_handle, HIPSOLVER_EIG_MODE_VECTOR, HIPSOLVER_FILL_MODE_LOWER, n,
                                   (double *)d_A, n, g_hs_dW, (double *)g_hs_work, g_hs_query_lwork, g_hs_dinfo);
    }
    if (status != HIPSOLVER_STATUS_SUCCESS) {
        return (int)status;
    }

    if (hipMemcpy(&info, g_hs_dinfo, sizeof(int), hipMemcpyDeviceToHost) != hipSuccess) {
        (void)hipGetLastError();
        return MAGMA_ERR_UNKNOWN;
    }
    if (info != 0) {
        return info;
    }
    if (detail) {
        const auto now = std::chrono::steady_clock::now();
        solve_seconds = std::chrono::duration<double>(now - phase_start).count();
        phase_start = now;
    }
    /* g_hs_dinfo[1] flags a non-finite vector element or eigenvalue */
    if (eigen_check(cplx, cplx, d_A, n, n, g_hs_dW, maxn, g_hs_dinfo + 1, 0) != hipSuccess ||
        hipMemcpy(&bad, g_hs_dinfo + 1, sizeof(int), hipMemcpyDeviceToHost) != hipSuccess) {
        (void)hipGetLastError();
        return MAGMA_ERR_UNKNOWN;
    }
    if (detail) {
        const auto now = std::chrono::steady_clock::now();
        check_seconds = std::chrono::duration<double>(now - phase_start).count();
        phase_start = now;
    }
    if (bad) {
        std::fprintf(stderr, "OpenMX: rocSOLVER %s returned non-finite eigenpairs (n=%d)\n", name, n);
        std::fflush(stderr);
        return MAGMA_ERR_UNKNOWN;
    }
    if (hipMemcpy(g_hs_hW, g_hs_dW, sizeof(double) * (size_t)n, hipMemcpyDeviceToHost) != hipSuccess) {
        (void)hipGetLastError();
        return MAGMA_ERR_UNKNOWN;
    }
    std::memcpy(w, g_hs_hW, sizeof(double) * (size_t)maxn);
    hipDeviceSynchronize();

    if (detail) {
        const double values_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - phase_start).count();

        std::fprintf(stderr, "GPUEIGENDETAIL rank=%d n=%d maxn=%d type=%s stage=rocsolver "
                     "setup=%.6f phase_in=%.6f solve=%.6f phase_out_check=%.6f values=%.6f workspace_mib=%.3f\n",
                     MYID_MPI_COMM_WORLD, n, maxn, cplx ? "complex" : "real", setup_seconds, phase_in_seconds,
                     solve_seconds, check_seconds, values_seconds, (double)g_hs_work_bytes / 1048576.0);
        std::fflush(stderr);
    }

    *mout_out = maxn;
    return 0;
}

/* MAGMA's dsyevdx_gpu / zheevdx_gpu (the Ozaki-II drivers when ozaki2) on
   d_A for the lowest maxn eigenpairs: the vectors in the first maxn columns
   of d_A, the eigenvalues in w (host, n entries of room).  *touched is set
   once d_A may have changed. */
int magma_evdx_locked(int cplx, int ozaki2, int n, int maxn, void *d_A, double *w, int *mout_out, int *touched)
{
    magma_int_t mn = static_cast<magma_int_t>(n);
    magma_int_t mmaxn = static_cast<magma_int_t>(maxn);
    magma_int_t il = 1;
    magma_int_t iu = mmaxn;
    magma_int_t mout = 0;
    magma_int_t info = 0;
    magma_int_t ret;
    magma_range_t range = (n == maxn) ? MagmaRangeAll : MagmaRangeI;
    int err;

    err = ensure_magma_initialized();
    if (err != MAGMA_SUCCESS) {
        return err;
    }

    if (!cplx) {
        double work_query = 0.0;
        magma_int_t iwork_query = 0;

        err = ensure_host_matrix(static_cast<size_t>(n) * static_cast<size_t>(n));
        if (err != MAGMA_SUCCESS) {
            return err;
        }

        ret = (ozaki2 ? magma_dsyevdx_ozaki2_gpu : magma_dsyevdx_gpu)(
                                        MagmaVec, range, MagmaLower,
                                        mn, reinterpret_cast<magmaDouble_ptr>(d_A), mn,
                                        0.0, 0.0, il, iu, &mout, w,
                                        g_host_matrix, mn,
                                        &work_query, -1, &iwork_query, -1, &info);
        if (ret != MAGMA_SUCCESS) {
            return static_cast<int>(ret);
        }
        if (info != 0) {
            return static_cast<int>(info);
        }

        magma_int_t lwork = static_cast<magma_int_t>(work_query);
        magma_int_t liwork = iwork_query;
        if (lwork < 1) {
            lwork = 1;
        }
        if (liwork < 1) {
            liwork = 1;
        }

        err = ensure_work(lwork, liwork);
        if (err != MAGMA_SUCCESS) {
            return err;
        }

        mout = 0;
        info = 0;
        *touched = 1;
        ret = (ozaki2 ? magma_dsyevdx_ozaki2_gpu : magma_dsyevdx_gpu)(
                                MagmaVec, range, MagmaLower,
                                mn, reinterpret_cast<magmaDouble_ptr>(d_A), mn,
                                0.0, 0.0, il, iu, &mout, w,
                                g_host_matrix, mn,
                                g_work, lwork, g_iwork, liwork, &info);
    }
    else {
        magmaDoubleComplex work_query = MAGMA_Z_ZERO;
        double rwork_query = 0.0;
        magma_int_t iwork_query = 0;

        err = ensure_z_host_matrix(static_cast<size_t>(n) * static_cast<size_t>(n));
        if (err != MAGMA_SUCCESS) {
            return err;
        }

        ret = (ozaki2 ? magma_zheevdx_ozaki2_gpu : magma_zheevdx_gpu)(
                                        MagmaVec, range, MagmaLower,
                                        mn, reinterpret_cast<magmaDoubleComplex_ptr>(d_A), mn,
                                        0.0, 0.0, il, iu, &mout, w,
                                        g_z_host_matrix, mn,
                                        &work_query, -1,
                                        &rwork_query, -1,
                                        &iwork_query, -1, &info);
        if (ret != MAGMA_SUCCESS) {
            return static_cast<int>(ret);
        }
        if (info != 0) {
            return static_cast<int>(info);
        }

        magma_int_t lwork = static_cast<magma_int_t>(MAGMA_Z_REAL(work_query));
        magma_int_t lrwork = static_cast<magma_int_t>(rwork_query);
        magma_int_t liwork = iwork_query;
        if (lwork < 1) {
            lwork = 1;
        }
        if (lrwork < 1) {
            lrwork = 1;
        }
        if (liwork < 1) {
            liwork = 1;
        }

        err = ensure_z_work(lwork, lrwork, liwork);
        if (err != MAGMA_SUCCESS) {
            return err;
        }

        mout = 0;
        info = 0;
        *touched = 1;
        ret = (ozaki2 ? magma_zheevdx_ozaki2_gpu : magma_zheevdx_gpu)(
                                MagmaVec, range, MagmaLower,
                                mn, reinterpret_cast<magmaDoubleComplex_ptr>(d_A), mn,
                                0.0, 0.0, il, iu, &mout, w,
                                g_z_host_matrix, mn,
                                g_z_work, lwork,
                                g_z_rwork, lrwork,
                                g_z_iwork, liwork, &info);
    }
    hipDeviceSynchronize();

    /* for n <= 128 MAGMA solves the whole spectrum with LAPACK dsyevd /
       zheevd and reports mout = n: the first maxn pairs are the request */
    if (ret == MAGMA_SUCCESS && info == 0 && mout > mmaxn) {
        mout = mmaxn;
    }
    *mout_out = static_cast<int>(mout);
    if (ret != MAGMA_SUCCESS) {
        return static_cast<int>(ret);
    }
    return static_cast<int>(info);
}

/* The copy of the input of a solve that may be retried. */
struct EigenBackup {
    void  *ptr = nullptr;
    size_t bytes = 0;
    int    host = 0;     /* malloc'd host copy (else device) */
    int    kept = 0;     /* the device copy is g_backup (kept for the next solve) */
};

/* HIP copies between pageable host memory and device memory it did not
   allocate itself (the OpenMP runtime's mappings, which the band solver
   hands over) on the host and faults, while a copy that has HIP device
   memory on one side works for every device pointer: host copies go through
   a HIP staging buffer in chunks. */
int staged_host_copy(void *dst, const void *src, size_t bytes, int to_host)
{
    const size_t chunk = bytes < (static_cast<size_t>(32) << 20) ? bytes : (static_cast<size_t>(32) << 20);
    char        *stage = nullptr;
    int          ok = 1;

    if (hipMalloc((void **)&stage, chunk) != hipSuccess) {
        (void)hipGetLastError();
        return -1;
    }
    for (size_t off = 0; ok && off < bytes; off += chunk) {
        const size_t len = (bytes - off < chunk) ? bytes - off : chunk;

        if (to_host)
            ok = hipMemcpy(stage, (const char *)src + off, len, hipMemcpyDefault) == hipSuccess &&
                 hipMemcpy((char *)dst + off, stage, len, hipMemcpyDeviceToHost) == hipSuccess;
        else
            ok = hipMemcpy(stage, (const char *)src + off, len, hipMemcpyHostToDevice) == hipSuccess &&
                 hipMemcpy((char *)dst + off, stage, len, hipMemcpyDefault) == hipSuccess;
    }
    if (hipDeviceSynchronize() != hipSuccess) ok = 0;
    if (!ok) (void)hipGetLastError();
    (void)hipFree(stage);
    return ok ? 0 : -1;
}

void backup_free(EigenBackup *b)
{
    if (b->ptr != nullptr) {
        if (b->host) std::free(b->ptr);
        else if (!b->kept) (void)hipFree(b->ptr);
    }
    *b = EigenBackup();
}

/* device copy when the copy and the solver's own device workspace
   (solver_bytes) fit this rank's share of the free memory above 1 GiB
   (the free-memory guard OpenMX gives the Ozaki-II workspaces); a copy up to
   OPENMX_GPUSOLVER_CACHE_MB is kept for the next solve.  Otherwise a host
   copy.  Returns 0 when no copy could be made (the solve then runs once). */
int backup_save(EigenBackup *b, const void *d_A, size_t bytes, size_t solver_bytes)
{
    *b = EigenBackup();
    b->bytes = bytes;
    if (hipDeviceSynchronize() != hipSuccess) {
        (void)hipGetLastError();
        return 0;
    }

    if (bytes <= g_backup_bytes) {
        b->ptr = g_backup;
        b->kept = 1;
    }
    else {
        const size_t reserve = static_cast<size_t>(1) << 30;
        const size_t free_bytes = device_free_bytes();
        int          ranks = (openmx_gpu_ranks_per_device_noncollective != nullptr)
                                 ? openmx_gpu_ranks_per_device_noncollective() : 1;
        size_t       share;

        if (ranks < 1) ranks = 1;
        share = (free_bytes > reserve) ? (free_bytes - reserve) / static_cast<size_t>(ranks) : 0;
        if (bytes <= share && solver_bytes <= share - bytes) {
            const int keep = bytes <= gpusolver_cache_limit_bytes();
            void     *p = nullptr;

            if (keep && g_backup != nullptr) {
                (void)hipFree(g_backup);
                g_backup = nullptr;
                g_backup_bytes = 0;
            }
            if (hipMalloc(&p, bytes) == hipSuccess) {
                b->ptr = p;
                if (keep) {
                    g_backup = p;
                    g_backup_bytes = bytes;
                    b->kept = 1;
                }
            }
            else {
                (void)hipGetLastError();
            }
        }
    }
    if (b->ptr != nullptr) {
        if (hipMemcpy(b->ptr, d_A, bytes, hipMemcpyDefault) == hipSuccess && hipDeviceSynchronize() == hipSuccess) {
            return 1;
        }
        (void)hipGetLastError();
        backup_free(b);
        b->bytes = bytes;
    }

    b->ptr = std::malloc(bytes);
    if (b->ptr == nullptr) {
        return 0;
    }
    b->host = 1;
    if (staged_host_copy(b->ptr, d_A, bytes, 1) != 0) {
        backup_free(b);
        return 0;
    }
    return 1;
}

int backup_restore(const EigenBackup *b, void *d_A)
{
    if (b->host) {
        return staged_host_copy(d_A, b->ptr, b->bytes, 0);
    }
    if (hipMemcpy(d_A, b->ptr, b->bytes, hipMemcpyDefault) != hipSuccess || hipDeviceSynchronize() != hipSuccess) {
        (void)hipGetLastError();
        return -1;
    }
    return 0;
}

/* when even the last backend fails: what did the input look like? */
void backup_report(const EigenBackup *b, int cplx, int n, int info, const char *what)
{
    const size_t count = static_cast<size_t>(n) * static_cast<size_t>(n) * (cplx ? 2u : 1u);
    const double *h = nullptr;
    double       *tmp = nullptr;
    size_t        nonfinite = 0;
    double        amax = 0.0, asym = 0.0;

    if (b->ptr == nullptr) {
        std::fprintf(stderr, "OpenMX: %s failed too (info=%d, n=%d); the input was not kept.\n", what, info, n);
        std::fflush(stderr);
        return;
    }
    if (b->host) {
        h = static_cast<const double *>(b->ptr);
    }
    else {
        tmp = static_cast<double *>(std::malloc(count * sizeof(double)));
        if (tmp == nullptr || hipMemcpy(tmp, b->ptr, count * sizeof(double), hipMemcpyDeviceToHost) != hipSuccess) {
            (void)hipGetLastError();
            std::free(tmp);
            std::fprintf(stderr, "OpenMX: %s failed too (info=%d, n=%d); the input could not be examined.\n",
                         what, info, n);
            std::fflush(stderr);
            return;
        }
        h = tmp;
    }
    for (size_t i = 0; i < count; i++) {
        if (!std::isfinite(h[i])) nonfinite++;
        else if (std::fabs(h[i]) > amax) amax = std::fabs(h[i]);
    }
    for (int c = 0; c < n; c++) {
        for (int r = c + 1; r < n; r++) {
            const size_t lo = static_cast<size_t>(c) * n + r, up = static_cast<size_t>(r) * n + c;
            const double d = cplx ? std::fabs(h[2 * lo] - h[2 * up]) + std::fabs(h[2 * lo + 1] + h[2 * up + 1])
                                  : std::fabs(h[lo] - h[up]);

            if (d > asym) asym = d;
        }
    }
    std::fprintf(stderr, "OpenMX: %s failed too (info=%d, n=%d): the input has %zu non-finite values, "
                 "max |a| = %.3e, max |a_ij - conj(a_ji)| = %.3e (both triangles; the solvers read the lower)\n",
                 what, info, n, nonfinite, amax, asym);
    std::fflush(stderr);
    std::free(tmp);
}

enum { ATTEMPT_ROCSOLVER, ATTEMPT_MAGMA_OZAKI2, ATTEMPT_MAGMA };

const char *attempt_name(int attempt, int cplx)
{
    switch (attempt) {
    case ATTEMPT_ROCSOLVER:    return cplx ? "rocSOLVER zheevd" : "rocSOLVER dsyevd";
    case ATTEMPT_MAGMA_OZAKI2: return cplx ? "MAGMA Ozaki-II zheevdx" : "MAGMA Ozaki-II dsyevdx";
    default:                   return cplx ? "MAGMA zheevdx" : "MAGMA dsyevdx";
    }
}

/* may a failed attempt be solved again with the next backend? */
int attempt_retryable(int attempt, int info)
{
    if (attempt == ATTEMPT_ROCSOLVER) return info != MAGMA_ERR_ILLEGAL_VALUE;
    return info > 0;   /* MAGMA: no convergence; allocation and argument errors would recur */
}

/* HIP device memory for the MAGMA attempts on a foreign d_A (see
   dense_eigen_locked), or nullptr when less than 512 MiB would stay free
   beside it (MAGMA's BLAS may run hipBLASLt, which ends the process when its
   own allocation fails).  rocSOLVER's workspace goes first when memory is
   short; the next rocSOLVER solve allocates it again. */
void *solve_copy_alloc(size_t bytes)
{
    const size_t guard = static_cast<size_t>(512) << 20;
    void        *p = nullptr;

    if (device_free_bytes() < bytes + guard && g_hs_work != nullptr) {
        (void)hipFree(g_hs_work);
        g_hs_work = nullptr;
        g_hs_work_bytes = 0;
    }
    if (device_free_bytes() < bytes + guard) return nullptr;
    if (hipMalloc(&p, bytes) != hipSuccess) {
        (void)hipGetLastError();
        return nullptr;
    }
    return p;
}

/* The dense FP64 eigensolve behind every entry point: the backend chain
   [rocSOLVER (first_rocsolver),] MAGMA Ozaki-II (use_magma_ozaki2()), stock
   MAGMA, each later backend only after the previous one failed (or, in the
   retry test, after the first one succeeded), from the copy of the input
   kept before the first solve (eigen_retry_policy).  A rocSOLVER attempt
   that failed before touching d_A (no workspace, or the memory preflight of
   a rerouted solve) is followed by MAGMA without a copy.  With foreign, d_A
   may be device memory HIP did not allocate (the OpenMP mappings the band
   solver hands to openmx_hipsolver_zheevd_gpu) or APU host memory: HIP
   faults on a copy between such an array and pageable host memory, which
   MAGMA makes, so the MAGMA attempts then solve a HIP copy and their
   vectors (the first maxn columns) are copied to d_A device to device.
   Returns the result of the last attempt made; the caller holds
   g_magma_mutex. */
int dense_eigen_locked(int cplx, int first_rocsolver, int preflight, int foreign, int n, int maxn, void *d_A,
                       double *w, int *mout_out)
{
    static int  test_announced = 0;
    static const int profile = env_flag("OPENMX_GPU_EIGENSOLVER_PROFILE", 0) ||
                               env_flag("OPENMX_GPU_VERBOSE", 0);
    static const int detail = env_flag("OPENMX_GPU_EIGENSOLVER_PROFILE", 0);
    const int   policy = eigen_retry_policy();
    const int   ozaki2 = use_magma_ozaki2();
    const size_t abytes = static_cast<size_t>(n) * static_cast<size_t>(n) *
                          (cplx ? sizeof(magmaDoubleComplex) : sizeof(double));
    int         attempts[3];
    int         count = 0;
    int         last = 0;
    EigenBackup backup;
    int         have_backup = 0;
    void       *solve = d_A;      /* the matrix the attempts work on */
    int         on_copy = 0;      /* the last attempt solved the HIP copy */
    int         info = 0;
    int         mout = 0;
    int         touched = 0;
    int         retried = 0;

    if (first_rocsolver) attempts[count++] = ATTEMPT_ROCSOLVER;
    attempts[count++] = ozaki2 ? ATTEMPT_MAGMA_OZAKI2 : ATTEMPT_MAGMA;
    if (ozaki2) attempts[count++] = ATTEMPT_MAGMA;

    if (policy != 0 && count > 1) {
        const auto started = detail ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};

        /* MAGMA's device workspace is about the matrix again (1.5 n^2 real
           words, n^2 complex ones), rocSOLVER's two to three times it */
        have_backup = backup_save(&backup, d_A, abytes, (first_rocsolver ? 4 : 2) * abytes);
        if (detail) {
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();

            std::fprintf(stderr, "GPUEIGENDETAIL rank=%d n=%d maxn=%d type=%s stage=backup_save "
                         "seconds=%.6f mib=%.3f location=%s cached=%d\n", MYID_MPI_COMM_WORLD, n, maxn,
                         cplx ? "complex" : "real", seconds, (double)abytes / 1048576.0,
                         !have_backup ? "none" : (backup.host ? "host" : "device"), have_backup && backup.kept);
            std::fflush(stderr);
        }
    }

    for (int k = 0; k < count; k++) {
        const int forced = (k == 0 && policy == 2);

        if (k > 0 && touched && !have_backup) {
            /* (in the retry test the first, successful result stands) */
            if (info != 0) {
                std::fprintf(stderr, "OpenMX: %s failed (info=%d, n=%d); no copy of the input was kept, "
                             "not solved again.\n", attempt_name(attempts[k - 1], cplx), info, n);
                std::fflush(stderr);
            }
            break;
        }
        if (k > 0 && foreign && solve == d_A && attempts[k] != ATTEMPT_ROCSOLVER) {
            void *copy = solve_copy_alloc(abytes);

            /* an untouched d_A is the input; else the copy is restored below */
            if (copy != nullptr && !touched &&
                (hipMemcpy(copy, d_A, abytes, hipMemcpyDefault) != hipSuccess ||
                 hipDeviceSynchronize() != hipSuccess)) {
                (void)hipGetLastError();
                (void)hipFree(copy);
                copy = nullptr;
            }
            if (copy == nullptr) {
                if (info != 0) {
                    std::fprintf(stderr, "OpenMX: %s failed (info=%d, n=%d); no device memory for %s, "
                                 "not solved again.\n", attempt_name(attempts[k - 1], cplx), info, n,
                                 attempt_name(attempts[k], cplx));
                    std::fflush(stderr);
                }
                break;
            }
            solve = copy;
        }
        if (k > 0 && touched) {
            const auto started = detail ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            const int restore_info = backup_restore(&backup, solve);

            if (detail) {
                const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();

                std::fprintf(stderr, "GPUEIGENDETAIL rank=%d n=%d maxn=%d type=%s stage=backup_restore "
                             "seconds=%.6f info=%d\n", MYID_MPI_COMM_WORLD, n, maxn, cplx ? "complex" : "real",
                             seconds, restore_info);
                std::fflush(stderr);
            }
            if (restore_info != 0) {
                std::fprintf(stderr, "OpenMX: %s failed (info=%d, n=%d); the input could not be restored.\n",
                             attempt_name(attempts[k - 1], cplx), info, n);
                std::fflush(stderr);
                /* d_A may be half restored (a half restored copy is dropped) */
                if (info == 0 && solve == d_A) info = MAGMA_ERR_UNKNOWN;
                break;
            }
            retried = 1;
        }

        last = k;
        on_copy = (solve != d_A);
        touched = 0;
        mout = 0;
        const auto started = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        info = (attempts[k] == ATTEMPT_ROCSOLVER)
                   ? rocsolver_evd_locked(cplx, n, maxn, solve, w, &mout, &touched, preflight)
                   : magma_evdx_locked(cplx, attempts[k] == ATTEMPT_MAGMA_OZAKI2, n, maxn, solve, w, &mout, &touched);
        if (profile) {
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();

            std::fprintf(stderr, "GPUEIGENPROF rank=%d n=%d maxn=%d type=%s backend=%s attempt=%d "
                         "seconds=%.6f info=%d mout=%d\n", MYID_MPI_COMM_WORLD, n, maxn,
                         cplx ? "complex" : "real", attempts[k] == ATTEMPT_ROCSOLVER ? "rocsolver" :
                         (attempts[k] == ATTEMPT_MAGMA_OZAKI2 ? "magma_ozaki2" : "magma"),
                         k + 1, seconds, info, mout);
            std::fflush(stderr);
        }

        if (k + 1 == count) break;
        if (info == 0) {
            if (!forced) break;
            if (!test_announced) {
                test_announced = 1;
                std::fprintf(stderr, "OpenMX: GPU eigensolver retry test (OPENMX_GPUSOLVER_RETRY_TEST=1): "
                             "every solve is solved again (first: %s -> %s, n=%d, input copy in %s memory).\n",
                             attempt_name(attempts[k], cplx), attempt_name(attempts[k + 1], cplx), n,
                             !have_backup ? "no" : (backup.host ? "host" : "device"));
                std::fflush(stderr);
            }
            continue;
        }
        if (!touched) {
            /* MAGMA's allocations would fail again; d_A is intact for MAGMA
               after a rocSOLVER attempt */
            if (attempts[k] == ATTEMPT_ROCSOLVER && (policy != 0 || preflight)) continue;
            break;
        }
        if (policy == 0 || !attempt_retryable(attempts[k], info)) break;
        if (OPENMX_STDOUT_LEVEL(2)) {
            std::fprintf(stderr, "OpenMX: %s failed (info=%d, n=%d); solving again with %s.\n",
                         attempt_name(attempts[k], cplx), info, n, attempt_name(attempts[k + 1], cplx));
            std::fflush(stderr);
        }
    }

    /* MAGMA's eigenvectors, the first maxn columns, into the caller's array
       (each solve that ran on the copy ended with a device sync) */
    if (info == 0 && on_copy &&
        (hipMemcpy(d_A, solve, abytes / static_cast<size_t>(n) * static_cast<size_t>(maxn), hipMemcpyDefault) !=
             hipSuccess || hipDeviceSynchronize() != hipSuccess)) {
        (void)hipGetLastError();
        std::fprintf(stderr, "OpenMX: the %s eigenvectors could not be copied back (n=%d).\n",
                     attempt_name(attempts[last], cplx), n);
        std::fflush(stderr);
        info = MAGMA_ERR_UNKNOWN;
    }
    if (solve != d_A) {
        (void)hipFree(solve);
    }
    if (retried && info == 0 && policy != 2) {
        if (OPENMX_STDOUT_LEVEL(2)) {
            std::fprintf(stderr, "OpenMX: the %s solve succeeded (n=%d).\n", attempt_name(attempts[last], cplx), n);
            std::fflush(stderr);
        }
    }
    if (retried && info != 0) {
        backup_report(&backup, cplx, n, info, attempt_name(attempts[last], cplx));
    }
    if (have_backup) {
        const auto started = detail ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};

        backup_free(&backup);
        if (detail) {
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();

            std::fprintf(stderr, "GPUEIGENDETAIL rank=%d n=%d maxn=%d type=%s stage=backup_free seconds=%.6f\n",
                         MYID_MPI_COMM_WORLD, n, maxn, cplx ? "complex" : "real", seconds);
            std::fflush(stderr);
        }
    }
    *mout_out = mout;
    return info;
}

} // namespace

/* Drop-in replacement for openmx_magma_zheevdx_gpu backed by
   hipsolverDnZheevd.  rocSOLVER's full-spectrum divide&conquer is much faster
   than its partial zheevdx, so all n eigenpairs are computed and only the
   lowest maxn eigenvalues are stored into w; d_A is overwritten with all n
   eigenvectors (ascending), of which callers consume the first maxn columns.
   A failed solve (nonzero info, an error or non-finite vectors) is solved
   again with MAGMA from a copy of the input (eigen_retry_policy), on HIP
   memory of its own because d_A may be an OpenMP mapping or APU host memory
   (then only the first maxn columns of d_A are vectors); w then needs the n
   entries the MAGMA path needs. */
extern "C" int openmx_hipsolver_zheevd_gpu(int n, int maxn, void *d_A, double *w, int *mout_out)
{
    std::lock_guard<std::mutex> lock(g_magma_mutex);
    int mout = 0;
    int info;

    if (mout_out != nullptr) {
        *mout_out = 0;
    }
    if (n <= 0 || maxn <= 0 || maxn > n || d_A == nullptr || w == nullptr) {
        return MAGMA_ERR_ILLEGAL_VALUE;
    }
    info = dense_eigen_locked(1, 1, 0, 1, n, maxn, d_A, w, &mout);
    if (mout_out != nullptr) {
        *mout_out = mout;
    }
    return info;
}

/* Release only the large Zheevd scratch allocation.  Handles, eigenvalue
   storage, and host buffers are small and remain cached.  The caller invokes
   this after the synchronous solve when another MPI owner or a large
   back-transform needs the shared MI300A memory. */
extern "C" int openmx_hipsolver_release_workspace(void)
{
    std::lock_guard<std::mutex> lock(g_magma_mutex);

    if (g_hs_work != nullptr) {
        if (hipFree(g_hs_work) != hipSuccess) {
            return -1;
        }
        g_hs_work = nullptr;
    }
    g_hs_work_bytes = 0;
    g_hs_query_n = 0;
    g_hs_query_cplx = -1;
    g_hs_query_lwork = 0;
    return 0;
}

/* Diagnostic: run a standalone zheevd on a fresh random Hermitian matrix and
   print the wall time, to locate the point in the run where GPU performance
   collapses.  Enabled via OPENMX_ZHEEVD_SELFTEST=1. */
extern "C" void openmx_gpu_zheevd_selftest(const char *label)
{
    const int n = 2808;
    size_t    nn = (size_t)n * (size_t)n;
    hipDoubleComplex *h_A = (hipDoubleComplex *)std::malloc(sizeof(hipDoubleComplex) * nn);
    hipDoubleComplex *d_A = nullptr;
    double           *w = (double *)std::malloc(sizeof(double) * n);

    if (h_A == nullptr || w == nullptr) {
        std::free(h_A);
        std::free(w);
        return;
    }

    unsigned seed = 12345u;
    for (int j = 0; j < n; j++) {
        for (int i = j; i < n; i++) {
            seed = seed * 1664525u + 1013904223u;
            double re = (double)(seed >> 8) / (double)(1u << 24) - 0.5;
            seed = seed * 1664525u + 1013904223u;
            double im = (i == j) ? 0.0 : (double)(seed >> 8) / (double)(1u << 24) - 0.5;
            h_A[(size_t)j * n + i] = {re, im};
            h_A[(size_t)i * n + j] = {re, -im};
        }
    }

    if (hipMalloc((void **)&d_A, sizeof(hipDoubleComplex) * nn) != hipSuccess) {
        std::free(h_A);
        std::free(w);
        return;
    }

    for (int rep = 0; rep < 2; rep++) {
        hipMemcpy(d_A, h_A, sizeof(hipDoubleComplex) * nn, hipMemcpyHostToDevice);
        hipDeviceSynchronize();
        double t0 = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
        int    mout = 0;
        int    info = openmx_hipsolver_zheevd_gpu(n, n, d_A, w, &mout);
        hipDeviceSynchronize();
        double t1 = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
        std::printf("[ZHEEVD_SELFTEST] %-12s rep%d  %8.3f s (info=%d w0=%.4f)\n", label, rep, t1 - t0, info,
                    w[0]);
        std::fflush(stdout);
    }

    hipFree(d_A);
    std::free(h_A);
    std::free(w);
}

/* MAGMA dense FP64 eigensolves (cluster, band, DC, DC-LNO, Krylov): the
   lowest maxn eigenpairs of d_A (n x n, lower triangle, leading dimension
   n), the vectors in the first maxn columns of d_A and the eigenvalues in w
   (host, n entries of room).  Returns 0, MAGMA's positive info (no
   convergence) or a negative MAGMA error (MAGMA_ERR_HOST_ALLOC/-112 and
   MAGMA_ERR_DEVICE_ALLOC/-113 when a workspace could not be allocated).
   A non-converged Ozaki-II solve is solved again with stock MAGMA and a
   device whose default backend is rocSOLVER, an explicit hipsolver choice,
   or an order below OPENMX_GPU_EIGENSOLVER_MAGMA_MIN_N goes to rocSOLVER
   first (dense_eigen_locked).  Despite the historical MAGMA names these
   are the common dense entry points, including the cluster root solve. */
static int openmx_magma_dsyevdx_gpu_single(int n, int maxn, double *d_A, double *w, int *mout_out)
{
    std::lock_guard<std::mutex> lock(g_magma_mutex);
    int mout = 0;
    int info;

    if (mout_out != nullptr) {
        *mout_out = 0;
    }
    if (n <= 0 || maxn <= 0 || maxn > n || d_A == nullptr || w == nullptr) {
        return MAGMA_ERR_ILLEGAL_VALUE;
    }
    {
        const int reroute = openmx_gpu_eigen_backend(n, maxn, 0);

        info = dense_eigen_locked(0, reroute, reroute, 0, n, maxn, d_A, w, &mout);
    }
    if (mout_out != nullptr) {
        *mout_out = mout;
    }
    return info;
}

static int openmx_magma_zheevdx_gpu_single(int n, int maxn, void *d_A, double *w, int *mout_out)
{
    std::lock_guard<std::mutex> lock(g_magma_mutex);
    int mout = 0;
    int info;

    if (mout_out != nullptr) {
        *mout_out = 0;
    }
    if (n <= 0 || maxn <= 0 || maxn > n || d_A == nullptr || w == nullptr) {
        return MAGMA_ERR_ILLEGAL_VALUE;
    }
    {
        const int reroute = openmx_gpu_eigen_backend(n, maxn, 1);

        info = dense_eigen_locked(1, reroute, reroute, 0, n, maxn, d_A, w, &mout);
    }
    if (mout_out != nullptr) {
        *mout_out = mout;
    }
    return info;
}

namespace {
struct OpenMXMagmaNativeCall {
    int complex_matrix,n,nev; void *a; double *w; int mout;
    static int run(void *opaque,double *output) {
        auto *c=static_cast<OpenMXMagmaNativeCall*>(opaque);
        int result=c->complex_matrix?
            openmx_magma_zheevdx_gpu_single(c->n,c->nev,c->a,output,&c->mout):
            openmx_magma_dsyevdx_gpu_single(c->n,c->nev,(double*)c->a,output,&c->mout);
        return result==0 && c->mout!=c->nev ? MAGMA_ERR_ILLEGAL_VALUE : result;
    }
};
int openmx_magma_dense_dispatch(int complex_matrix,int n,int nev,void *a,double *w,int *mout)
{
    /* No extra allocation in the normal single-GPU path. The runtime hook
       performs the GPU count and mode checks before allocating its scratch. */
    OpenMXMagmaNativeCall call{complex_matrix,n,nev,a,w,0};
    int result;
    if (openmx_mgpu_solve_host_result_try &&
        openmx_mgpu_solve_host_result_try(complex_matrix,n,nev,a,w,mout,
            OpenMXMagmaNativeCall::run,&call,&result)) return result;
    result=OpenMXMagmaNativeCall::run(&call,w);
    if (mout) *mout=call.mout;
    return result;
}
}
extern "C" int openmx_magma_dsyevdx_gpu(int n,int nev,double *a,double *w,int *mout)
{ return openmx_magma_dense_dispatch(0,n,nev,a,w,mout); }
extern "C" int openmx_magma_zheevdx_gpu(int n,int nev,void *a,double *w,int *mout)
{ return openmx_magma_dense_dispatch(1,n,nev,a,w,mout); }

/* The MAGMA chain alone (no rocSOLVER first, whatever the size): the retry
   of the generic wrappers' rocSOLVER solves (gpusolver_Syevdx.c). */
extern "C" int openmx_magma_eigen_gpu(int cplx, int n, int maxn, void *d_A, double *w, int *mout_out)
{
    std::lock_guard<std::mutex> lock(g_magma_mutex);
    int mout = 0;
    int info;

    if (mout_out != nullptr) {
        *mout_out = 0;
    }
    if (n <= 0 || maxn <= 0 || maxn > n || d_A == nullptr || w == nullptr) {
        return MAGMA_ERR_ILLEGAL_VALUE;
    }
    info = dense_eigen_locked(cplx != 0, 0, 0, 0, n, maxn, d_A, w, &mout);
    if (mout_out != nullptr) {
        *mout_out = mout;
    }
    return info;
}

/* FP32 full solves for the refined eigensolver (eigen_refine_gpu.cu, with
   OPENMX_EIGEN_FP32_SOLVER=magma or as the retry of a failed rocSOLVER solve):
   all n eigenpairs of the n x n matrix d_A (device, ldda n, lower triangle),
   the vectors in place, the eigenvalues ascending in w (host).  The host
   workspaces stay cached like those of the FP64 solves. */
namespace {

float              *g_f_host_matrix = nullptr;
size_t              g_f_host_elems = 0;
float              *g_f_work = nullptr;
magma_int_t         g_f_lwork = 0;
float              *g_f_rwork = nullptr;
magma_int_t         g_f_lrwork = 0;
magma_int_t        *g_f_iwork = nullptr;
magma_int_t         g_f_liwork = 0;

template <class T>
int grow_cpu(T **ptr, magma_int_t *have, magma_int_t want)
{
    if (want <= *have) return MAGMA_SUCCESS;
    T *grown = nullptr;
    if (magma_malloc_cpu((void **)&grown, sizeof(T) * static_cast<size_t>(want)) != MAGMA_SUCCESS)
        return MAGMA_ERR_HOST_ALLOC;
    if (*ptr != nullptr) magma_free_cpu(*ptr);
    *ptr = grown;
    *have = want;
    return MAGMA_SUCCESS;
}

int fp32_solve(int cplx, int n, void *d_A, float *w)
{
    magma_int_t mn = static_cast<magma_int_t>(n), info = 0;
    float       work_query[2] = {0.0f, 0.0f};
    float       rwork_query = 0.0f;
    magma_int_t iwork_query = 0;
    size_t      host_elems = static_cast<size_t>(n) * static_cast<size_t>(n) * (cplx ? 2u : 1u);
    int         err;

    if (n <= 0 || d_A == nullptr || w == nullptr) return MAGMA_ERR_ILLEGAL_VALUE;
    err = ensure_magma_initialized();
    if (err != MAGMA_SUCCESS) return err;
    if (host_elems > g_f_host_elems) {
        float *grown = nullptr;
        if (magma_smalloc_cpu(&grown, host_elems) != MAGMA_SUCCESS) return MAGMA_ERR_HOST_ALLOC;
        if (g_f_host_matrix != nullptr) magma_free_cpu(g_f_host_matrix);
        g_f_host_matrix = grown;
        g_f_host_elems = host_elems;
    }
    if (cplx)
        magma_cheevd_gpu(MagmaVec, MagmaLower, mn, reinterpret_cast<magmaFloatComplex_ptr>(d_A), mn, w,
                         reinterpret_cast<magmaFloatComplex *>(g_f_host_matrix), mn,
                         reinterpret_cast<magmaFloatComplex *>(work_query), -1, &rwork_query, -1, &iwork_query, -1,
                         &info);
    else
        magma_ssyevd_gpu(MagmaVec, MagmaLower, mn, reinterpret_cast<magmaFloat_ptr>(d_A), mn, w, g_f_host_matrix, mn,
                         work_query, -1, &iwork_query, -1, &info);
    if (info != 0) return static_cast<int>(info);
    {
        /* MAGMA's device workspace (chetrd2_gpu/cunmtr_gpu/cstedx, see
           cheevd_gpu.cpp) and room for the queue's library handles: a hipBLAS
           or hipBLASLt call that cannot allocate ends the process, so the
           solve is declined instead when the device is that full */
        const size_t nb = static_cast<size_t>(cplx ? magma_get_chetrd_nb(mn) : magma_get_ssytrd_nb(mn));
        const size_t nn = static_cast<size_t>(n) * static_cast<size_t>(n);
        size_t       words = static_cast<size_t>(n) * ((static_cast<size_t>(n) + 63) / 64) +
                             2 * static_cast<size_t>(n) * nb;
        size_t       free_bytes = 0, total_bytes = 0;

        if (words < nn) words = nn;
        words *= cplx ? 2 : 1;
        if (words < 3 * static_cast<size_t>(n) * (static_cast<size_t>(n) / 2 + 1))
            words = 3 * static_cast<size_t>(n) * (static_cast<size_t>(n) / 2 + 1);
        if (hipMemGetInfo(&free_bytes, &total_bytes) != hipSuccess ||
            free_bytes < words * sizeof(float) + (static_cast<size_t>(512) << 20)) {
            if (OPENMX_STDOUT_LEVEL(2)) {
                std::printf("<eigen_refine_gpu> MAGMA FP32 solve declined: %.1f MiB free, %.1f MiB needed\n",
                            free_bytes / 1048576.0, (words * sizeof(float) + (static_cast<size_t>(512) << 20)) / 1048576.0);
            }
            std::fflush(stdout);
            return MAGMA_ERR_DEVICE_ALLOC;
        }
    }

    magma_int_t lwork = static_cast<magma_int_t>(work_query[0]) + 1;
    magma_int_t lrwork = static_cast<magma_int_t>(rwork_query) + 1;
    magma_int_t liwork = iwork_query + 1;

    if (grow_cpu(&g_f_work, &g_f_lwork, lwork * (cplx ? 2 : 1)) != MAGMA_SUCCESS ||
        (cplx && grow_cpu(&g_f_rwork, &g_f_lrwork, lrwork) != MAGMA_SUCCESS) ||
        grow_cpu(&g_f_iwork, &g_f_liwork, liwork) != MAGMA_SUCCESS)
        return MAGMA_ERR_HOST_ALLOC;
    info = 0;
    if (cplx)
        magma_cheevd_gpu(MagmaVec, MagmaLower, mn, reinterpret_cast<magmaFloatComplex_ptr>(d_A), mn, w,
                         reinterpret_cast<magmaFloatComplex *>(g_f_host_matrix), mn,
                         reinterpret_cast<magmaFloatComplex *>(g_f_work), lwork, g_f_rwork, lrwork, g_f_iwork, liwork,
                         &info);
    else
        magma_ssyevd_gpu(MagmaVec, MagmaLower, mn, reinterpret_cast<magmaFloat_ptr>(d_A), mn, w, g_f_host_matrix, mn,
                         g_f_work, lwork, g_f_iwork, liwork, &info);
    hipDeviceSynchronize();
    return static_cast<int>(info);
}

} // namespace

extern "C" int openmx_magma_ssyevd_gpu(int n, float *d_A, float *w)
{
    std::lock_guard<std::mutex> lock(g_magma_mutex);
    return fp32_solve(0, n, d_A, w);
}

extern "C" int openmx_magma_cheevd_gpu(int n, void *d_A, float *w)
{
    std::lock_guard<std::mutex> lock(g_magma_mutex);
    return fp32_solve(1, n, d_A, w);
}

extern "C" void openmx_magma_release_fp32_workspace(void)
{
    std::lock_guard<std::mutex> lock(g_magma_mutex);
    if (g_f_host_matrix != nullptr) magma_free_cpu(g_f_host_matrix);
    if (g_f_work != nullptr) magma_free_cpu(g_f_work);
    if (g_f_rwork != nullptr) magma_free_cpu(g_f_rwork);
    if (g_f_iwork != nullptr) magma_free_cpu(g_f_iwork);
    g_f_host_matrix = nullptr;
    g_f_work = g_f_rwork = nullptr;
    g_f_iwork = nullptr;
    g_f_host_elems = 0;
    g_f_lwork = g_f_lrwork = g_f_liwork = 0;
}

/* Returns the Ozaki-II adapter's per-stream hipBLAS handles and device
   workspaces; a no-op when nothing was routed. */
extern "C" void openmx_magma_ozaki2_release(void)
{
    magma_ozaki2_release_workspaces();
}

namespace {

/* the retry copy kept for small solves (dense_eigen_locked) */
void release_backup_locked()
{
    if (g_backup != nullptr) (void)hipFree(g_backup);
    g_backup = nullptr;
    g_backup_bytes = 0;
}

} // namespace

/* MAGMA's range-I path caches a dense host copy and several large pinned host
   work arrays.  On a unified-memory APU those allocations consume the same
   physical pool as the following GPU turns, so NC band calculations release
   them at a safe, synchronous turn boundary. */
extern "C" int openmx_magma_release_z_workspace(void)
{
    std::lock_guard<std::mutex> lock(g_magma_mutex);

    magma_ozaki2_release_workspaces();
    release_backup_locked();
    if (g_z_host_matrix != nullptr) magma_free_cpu(g_z_host_matrix);
    if (g_z_work != nullptr)        magma_free_cpu(g_z_work);
    if (g_z_rwork != nullptr)       magma_free_cpu(g_z_rwork);
    if (g_z_iwork != nullptr)       magma_free_cpu(g_z_iwork);
    g_z_host_matrix = nullptr;
    g_z_work = nullptr;
    g_z_rwork = nullptr;
    g_z_iwork = nullptr;
    g_z_host_matrix_elems = 0;
    g_z_lwork = 0;
    g_z_lrwork = 0;
    g_z_liwork = 0;
    return 0;
}

/* The real counterpart: the dense host copy (n^2 doubles) and the work
   arrays openmx_magma_dsyevdx_gpu keeps between solves, the retry copy and
   the Ozaki-II workspaces.  For the input boundary of a test suite
   (Free_Arrays(0)) and before the forces, like the complex one. */
extern "C" int openmx_magma_release_d_workspace(void)
{
    std::lock_guard<std::mutex> lock(g_magma_mutex);

    magma_ozaki2_release_workspaces();
    release_backup_locked();
    if (g_host_matrix != nullptr) magma_free_cpu(g_host_matrix);
    if (g_work != nullptr)        magma_free_cpu(g_work);
    if (g_iwork != nullptr)       magma_free_cpu(g_iwork);
    g_host_matrix = nullptr;
    g_work = nullptr;
    g_iwork = nullptr;
    g_host_matrix_elems = 0;
    g_lwork = 0;
    g_liwork = 0;
    return 0;
}

#include <cublas_v2.h>
#include "openmx_env.h"
#include "openmx_gpu_memory.hpp"
#include <cuda_runtime.h>

/* cudaMemGetInfo capped at the physically free device memory (NVML); see
   set_cuda_default_device_from_local_rank.h.  A weak reference: the
   standalone probe builds (tests/run_gemmul8_smoke.sh and friends) link
   this file without that object and take the plain runtime call. */
extern "C" __attribute__((weak)) cudaError_t OpenMX_GpuMemGetInfo(size_t *free_bytes, size_t *total_bytes);
static inline cudaError_t device_mem_get_info(size_t *free_bytes, size_t *total_bytes)
{
    return OpenMX_GpuMemGetInfo != nullptr ? OpenMX_GpuMemGetInfo(free_bytes, total_bytes)
                                           : cudaMemGetInfo(free_bytes, total_bytes);
}
#include <cuComplex.h>

#include <atomic>
#include <chrono>
#include <climits>
#include <new>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <type_traits>
#include <unordered_map>

#include "gemmul8.hpp"
#include "openmx_stdout_level.h"
#include "openmx_mgpu_runtime.h"

namespace {

/* Cooperative GEMM queues its worker products on the calling host thread.
   Carry an explicit forward precision stage into those products without
   changing global precision controls or another thread's operations. */
thread_local bool forward_worker_stage=false;
thread_local bool forward_worker_x_left=false;
thread_local bool inside_forward_tile=false;
template<class T> cublasStatus_t forward_tile(cublasHandle_t,cublasOperation_t,cublasOperation_t,
    int,int,int,const T*,int,const T*,int,T*,int);

constexpr unsigned kDefaultNumModuli = 15u;
constexpr unsigned kMaxNumModuli     = 20u;
constexpr unsigned kDefaultMinFreeAfterMiB = 1536u;
constexpr unsigned kDefaultMaxWorkspacePercent = 30u;
constexpr size_t   kMiB = 1024u * 1024u;
constexpr unsigned kDefaultMemorySavingMiB = 256u;
constexpr size_t   kMinMemorySavingBytes = size_t(256) * kMiB;
constexpr unsigned kDefaultA100ForwardMinN = 6144u;

struct WorkspaceKey {
    int          device;
    cudaStream_t stream;

    bool operator==(const WorkspaceKey &other) const
    {
        return device == other.device && stream == other.stream;
    }
};

struct WorkspaceKeyHash {
    std::size_t operator()(const WorkspaceKey &key) const
    {
        return (static_cast<std::size_t>(key.device) << 32) ^ (reinterpret_cast<std::uintptr_t>(key.stream) << 1);
    }
};

using Workspace=openmx::gpu::DeviceBuffer<unsigned char>;

std::mutex g_workspace_mutex;
std::unordered_map<WorkspaceKey, Workspace, WorkspaceKeyHash> g_workspaces;

/* scf.gemmul8.enable from the input file; written once per input parse
   (Input_std.c, before any GEMM runs) via openmx_gemmul8SetEnabled().
   -1 selects the device default, 0 forces plain cuBLAS FP64 GEMM, and
   1 enables GEMMul8 subject to the existing environment overrides. */
int g_input_enabled = -1;

/* OPENMX_GEMMUL8_TIMING=1: every call of openmx_gemmul8{D,Z}gemm is timed
   with a stream synchronization before and after it, and counted by type
   (real, complex) and path (GEMMul8, plain cuBLAS); the release of the
   prepared forms reports and resets the totals.  The synchronization keeps
   the products from overlapping other work, so the totals describe the
   products, not a run without the option. */
struct GeneralCounters {
    long long calls[2][2]   = {{0, 0}, {0, 0}}; /* [real, complex][GEMMul8, cuBLAS] */
    double    seconds[2][2] = {{0.0, 0.0}, {0.0, 0.0}};
    double    flops[2][2]   = {{0.0, 0.0}, {0.0, 0.0}};
};
GeneralCounters g_general_counters;
/* calls of openmx_gemmul8{D,Z}gemm that ran in plain cuBLAS FP64, timed or
   not: [real, complex] */
std::atomic<long long> g_general_native[2];

struct WorkspaceReport {
    size_t      required_bytes = 0;
    size_t      free_bytes     = 0;
    size_t      total_bytes    = 0;
    size_t      reserve_bytes  = 0;
    unsigned    max_workspace_percent = 0;
    unsigned    ranks_per_gpu = 1;
    const char *reason = "allocation failure";
};

unsigned env_u32(int id, unsigned fallback)
{
    const unsigned long long parsed = openmx_env_ulong_at(id, fallback);

    return parsed <= UINT_MAX ? static_cast<unsigned>(parsed) : fallback;
}

bool verbose_logging_enabled()
{
    bool verbose = openmx_env_flag_at(OPENMX_ENV_GPU_VERBOSE, false);
    verbose = openmx_env_flag_at(OPENMX_ENV_GEMM_VERBOSE, verbose);
    verbose = openmx_env_flag_at(OPENMX_ENV_GEMMUL8_VERBOSE, verbose);
    return openmx_env_flag_at(OPENMX_ENV_EXT_GEMMUL8_VERBOSE, verbose);
}

unsigned env_percent(int openmx_env, int gemmul8_env, unsigned fallback)
{
    unsigned percent = env_u32(gemmul8_env, fallback);
    percent          = env_u32(openmx_env, percent);

    if (100u < percent) {
        percent = 100u;
    }

    return percent;
}

size_t env_mib(int openmx_env, int gemmul8_env, unsigned fallback)
{
    unsigned mib = env_u32(gemmul8_env, fallback);
    mib          = env_u32(openmx_env, mib);

    return static_cast<size_t>(mib) * kMiB;
}

/* General GEMMs on A100 and other GPUs with full-rate FP64 use native
   double-precision GEMM (including its FP64 Tensor Core kernels).  The
   INT8 decomposition is intended for GPUs with weak FP64 hardware.  Use
   the same throughput boundary as grid_precision and eigen_refine_gpu,
   rather than assuming every Ampere device has A100's FP64 hardware.
   Cache the current device per thread; cudaSetDevice must not leave a
   previous device's policy active.  Unknown hardware keeps the old default. */
bool native_fp64_device_default()
{
    static thread_local int cached_device = -1;
    static thread_local bool native = false;
    int device = -1;

    if (cudaGetDevice(&device) != cudaSuccess) return false;
    if (device != cached_device) {
        int ratio = 0;
        if (cudaDeviceGetAttribute(&ratio, cudaDevAttrSingleToDoublePrecisionPerfRatio, device) != cudaSuccess)
            return false;
        native = 0 < ratio && ratio < 8;
        cached_device = device;
        if (native && (verbose_logging_enabled() || OPENMX_STDOUT_LEVEL(2))) {
            std::fprintf(stderr,
                         "openmx_gemmul8: device %d uses native cuBLAS FP64 for general GEMMs by default "
                         "(FP32:FP64 throughput ratio %d); scf.gemmul8.enable=on or "
                         "OPENMX_GEMMUL8_DISABLE=0 enables GEMMul8.\n", device, ratio);
        }
    }
    return native;
}

bool gemmul8_environment_override(int openmx_env, int gemmul8_env, bool *disabled)
{
    /* Highest-priority override first.  In particular, an explicit zero
       opts back into GEMMul8 even on native-FP64 devices. */
    for (int id : {openmx_env, gemmul8_env, int(OPENMX_ENV_GEMMUL8_DISABLE), int(OPENMX_ENV_EXT_GEMMUL8_DISABLE)}) {
        const int value = openmx_env_flag_at(id, -1);
        if (0 <= value) {
            *disabled = value != 0;
            return true;
        }
    }
    return false;
}

bool gemmul8_disabled(int openmx_env, int gemmul8_env)
{
    bool disabled = false;
    if (gemmul8_environment_override(openmx_env, gemmul8_env, &disabled)) return disabled;
    return g_input_enabled < 0 && native_fp64_device_default();
}

/* Measured on full A100 sm_80: native FP64 wins small forward products,
   while full-workspace Ozaki wins large real square products even after
   the production scratch trim. Do not extrapolate that crossover to MIG,
   A30, GeForce, other architectures, complex products or capped workspaces.
   Explicit arithmetic choices are handled before considering this policy. */
bool a100_large_forward_candidate(int m, int n, int k)
{
    const unsigned min_n = env_u32(OPENMX_ENV_GEMMUL8_A100_FORWARD_MIN_N, kDefaultA100ForwardMinN);
    bool disabled = false;
    if (g_input_enabled >= 0 || min_n == 0 || m != n || m != k || m <= 0 ||
        static_cast<unsigned>(m) < min_n ||
        gemmul8_environment_override(OPENMX_ENV_GEMMUL8_DISABLE_D, OPENMX_ENV_EXT_GEMMUL8_DISABLE_D, &disabled)) return false;
    if (!native_fp64_device_default()) return false;

    static thread_local int cached_device = -1;
    static thread_local bool full_a100 = false;
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess) return false;
    if (device != cached_device) {
        int major = 0, minor = 0, sms = 0;
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess ||
            cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device) != cudaSuccess ||
            cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device) != cudaSuccess) return false;
        full_a100 = major == 8 && minor == 0 && sms >= 100;
        cached_device = device;
    }
    return full_a100;
}

/* Ported from the AMD bridge: keep Ozaki-II enabled for large matrices by
   letting GEMMul8 block within a per-rank workspace cap.  Zero restores the
   uncapped policy.  Smaller nonzero caps are raised to a viable block size. */
size_t memory_saving_cap_bytes()
{
    static const size_t cap = [] {
        size_t bytes = env_mib(OPENMX_ENV_GEMMUL8_MAX_WORKSPACE_MB, OPENMX_ENV_EXT_GEMMUL8_MAX_WORKSPACE_MB,
                               kDefaultMemorySavingMiB);
        if (bytes != 0 && bytes < kMinMemorySavingBytes) {
            std::fprintf(stderr,
                         "openmx_gemmul8: workspace cap below %zu MiB; using %zu MiB.\n",
                         kMinMemorySavingBytes / kMiB, kMinMemorySavingBytes / kMiB);
            bytes = kMinMemorySavingBytes;
        }
        return bytes;
    }();
    return cap;
}

size_t capped_workspace_size(size_t required)
{
    const size_t cap = memory_saving_cap_bytes();
    return cap != 0 && cap < required ? cap : required;
}

void apply_memory_saving(cublasHandle_t handle)
{
    const size_t cap = memory_saving_cap_bytes();
    gemmul8::set_memory_saving(handle, cap != 0);
    if (cap != 0) {
        gemmul8::set_max_worksize(handle, cap);
    }
}

/* Each MPI process owns a workspace.  Budget the aggregate allocation on a
   shared GPU, rounding up for uneven rank placement. */
unsigned ranks_sharing_gpu()
{
    static const unsigned ranks = [] {
        unsigned local_size = env_u32(OPENMX_ENV_GEMMUL8_LOCAL_RANKS, 0u);
        if (local_size == 0u) local_size = env_u32(OPENMX_ENV_EXT_OMPI_COMM_WORLD_LOCAL_SIZE, 0u);
        if (local_size == 0u) local_size = env_u32(OPENMX_ENV_EXT_SLURM_NTASKS_PER_NODE, 0u);
        if (local_size == 0u) local_size = 1u;
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || count < 1) count = 1;
        const unsigned devices = static_cast<unsigned>(count);
        return local_size / devices + (local_size % devices != 0u);
    }();
    return ranks;
}

unsigned gemmul8_num_moduli(int openmx_env, int gemmul8_env)
{
    unsigned num_moduli = env_u32(gemmul8_env, kDefaultNumModuli);
    num_moduli          = env_u32(openmx_env, num_moduli);

    if (num_moduli < 2u || kMaxNumModuli < num_moduli) {
        num_moduli = kDefaultNumModuli;
    }

    return num_moduli;
}

cudaError_t release_workspace(Workspace &workspace)
{
    return workspace.try_reset();
}

bool workspace_exceeds_fraction(size_t required, size_t total, unsigned max_percent, unsigned ranks_per_gpu)
{
    return total != 0 && max_percent != 0 &&
           (total * static_cast<size_t>(max_percent)) / 100u / ranks_per_gpu < required;
}

bool free_after_workspace_is_too_low(size_t free_bytes, size_t workspace_size, size_t required, size_t reserve)
{
    if (required <= workspace_size) {
        return free_bytes < reserve;
    }

    const size_t extra_required = required - workspace_size;
    return free_bytes < extra_required || free_bytes - extra_required < reserve;
}

template <bool is_complex>
cublasStatus_t ensure_workspace(cublasHandle_t handle, size_t m, size_t n, size_t k, unsigned num_moduli,
                                bool fastmode, void **work, WorkspaceReport *report)
try {
    cudaStream_t stream = nullptr;
    int          device = -1;

    cublasStatus_t cublas_status = cublasGetStream(handle, &stream);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        return cublas_status;
    }

    cudaError_t cuda_status = cudaGetDevice(&device);
    if (cuda_status != cudaSuccess) {
        return CUBLAS_STATUS_INTERNAL_ERROR;
    }

    const size_t required = capped_workspace_size(gemmul8::workSize<is_complex, gemmul8::Backend::INT8>(
        m, n, k, num_moduli, false, false, nullptr, nullptr, fastmode));
    WorkspaceKey key      = {device, stream};
    const unsigned ranks_per_gpu = ranks_sharing_gpu();

    if (report != nullptr) {
        report->required_bytes = required;
        report->reserve_bytes =
            env_mib(OPENMX_ENV_GEMMUL8_MIN_FREE_AFTER_MB, OPENMX_ENV_EXT_GEMMUL8_MIN_FREE_AFTER_MB, kDefaultMinFreeAfterMiB);
        report->max_workspace_percent = env_percent(OPENMX_ENV_GEMMUL8_MAX_WORKSPACE_PERCENT,
                                                     OPENMX_ENV_EXT_GEMMUL8_MAX_WORKSPACE_PERCENT,
                                                     kDefaultMaxWorkspacePercent);
        report->ranks_per_gpu = ranks_per_gpu;
    }

    std::lock_guard<std::mutex> lock(g_workspace_mutex);
    Workspace                  &workspace = g_workspaces[key];

    size_t free_bytes  = 0;
    size_t total_bytes = 0;
    cuda_status        = device_mem_get_info(&free_bytes, &total_bytes);
    if (cuda_status == cudaSuccess && report != nullptr) {
        report->free_bytes  = free_bytes;
        report->total_bytes = total_bytes;
    }

    if (cuda_status == cudaSuccess &&
        workspace_exceeds_fraction(required, total_bytes, report != nullptr ? report->max_workspace_percent : 0u,
                                   ranks_per_gpu)) {
        if (report != nullptr) {
            report->reason = "workspace fraction policy";
        }
        if (release_workspace(workspace) != cudaSuccess) {
            return CUBLAS_STATUS_INTERNAL_ERROR;
        }
        return CUBLAS_STATUS_ALLOC_FAILED;
    }

    if (workspace.size() < required) {
        cuda_status = release_workspace(workspace);
        if (cuda_status != cudaSuccess) {
            return CUBLAS_STATUS_INTERNAL_ERROR;
        }

        cuda_status = device_mem_get_info(&free_bytes, &total_bytes);
        if (cuda_status == cudaSuccess && report != nullptr) {
            report->free_bytes  = free_bytes;
            report->total_bytes = total_bytes;
        }
    }

    if (cuda_status == cudaSuccess &&
        free_after_workspace_is_too_low(free_bytes, workspace.size(), required,
                                        report != nullptr ? report->reserve_bytes : 0u)) {
        if (report != nullptr) {
            report->reason = "free memory reserve policy";
        }
        if (release_workspace(workspace) != cudaSuccess) {
            return CUBLAS_STATUS_INTERNAL_ERROR;
        }
        return CUBLAS_STATUS_ALLOC_FAILED;
    }

    if (workspace.size() < required) {
        cuda_status = workspace.allocate_status(required);
        if (cuda_status != cudaSuccess) {
            if (report != nullptr) {
                report->reason = "cudaMalloc failure";
            }
            return CUBLAS_STATUS_ALLOC_FAILED;
        }
    }

    *work = workspace.get();
    return CUBLAS_STATUS_SUCCESS;
} catch (const std::bad_alloc &) {
    return CUBLAS_STATUS_ALLOC_FAILED;
}

template <bool is_complex>
void log_workspace_fallback_once(const WorkspaceReport &report)
{
    static bool warned = false;

    if (!verbose_logging_enabled() && !OPENMX_STDOUT_LEVEL(2)) {
        return;
    }

    std::lock_guard<std::mutex> lock(g_workspace_mutex);
    if (warned) {
        return;
    }

    fprintf(stderr,
            "openmx_gemmul8%sgemm: GEMMul8 workspace fallback by %s; "
            "need %.3f MiB, CUDA free %.3f MiB / total %.3f MiB, "
            "reserve %.3f MiB, max-workspace %u%% shared by %u rank(s). Falling back to native cuBLAS.\n",
            is_complex ? "Z" : "D", report.reason, (double)report.required_bytes / (1024.0 * 1024.0),
            (double)report.free_bytes / (1024.0 * 1024.0), (double)report.total_bytes / (1024.0 * 1024.0),
            (double)report.reserve_bytes / (1024.0 * 1024.0), report.max_workspace_percent, report.ranks_per_gpu);
    fflush(stderr);
    warned = true;
}

/* body(&native) performs the product and sets native when it took plain
   cuBLAS; flops counts the real floating-point operations of the product */
template <class Body>
cublasStatus_t timed_general_gemm(cublasHandle_t handle, int is_complex, double flops, Body body)
{
    const bool timing = openmx_env_flag_at(OPENMX_ENV_GEMMUL8_TIMING, 0) != 0;
    bool              native = false;

    if (!timing) {
        const cublasStatus_t status = body(&native);
        if (native) g_general_native[is_complex].fetch_add(1, std::memory_order_relaxed);
        return status;
    }

    cudaStream_t   stream = nullptr;
    cublasStatus_t status = cublasGetStream(handle, &stream);
    if (status != CUBLAS_STATUS_SUCCESS) return status;
    if (cudaStreamSynchronize(stream) != cudaSuccess) return CUBLAS_STATUS_INTERNAL_ERROR;
    const auto start = std::chrono::steady_clock::now();
    status = body(&native);
    if (native) g_general_native[is_complex].fetch_add(1, std::memory_order_relaxed);
    if (cudaStreamSynchronize(stream) != cudaSuccess) return CUBLAS_STATUS_INTERNAL_ERROR;
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

    std::lock_guard<std::mutex> lock(g_workspace_mutex);
    const int path = native ? 1 : 0;
    ++g_general_counters.calls[is_complex][path];
    g_general_counters.seconds[is_complex][path] += seconds;
    g_general_counters.flops[is_complex][path] += flops;
    return status;
}

} // namespace

namespace {

cublasStatus_t general_dgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n,
                             int k, const double *alpha, const double *A, int lda, const double *B, int ldb,
                             const double *beta, double *C, int ldc, bool *native)
{
    const cublasOperation_t gemmul8_transa = (transa == CUBLAS_OP_C) ? CUBLAS_OP_T : transa;
    const cublasOperation_t gemmul8_transb = (transb == CUBLAS_OP_C) ? CUBLAS_OP_T : transb;

    if (g_input_enabled == 0) {
        /* scf.gemmul8.enable off: the fallback is what the user asked for,
           so no warning (Input_std already reported it once) */
        *native = true;
        return cublasDgemm(handle, gemmul8_transa, gemmul8_transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }

    if (gemmul8_disabled(OPENMX_ENV_GEMMUL8_DISABLE_D, OPENMX_ENV_EXT_GEMMUL8_DISABLE_D)) {
        *native = true;
        return cublasDgemm(handle, gemmul8_transa, gemmul8_transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }

    const unsigned num_moduli = gemmul8_num_moduli(OPENMX_ENV_GEMMUL8_NUM_MOD_D, OPENMX_ENV_EXT_GEMMUL8_NUM_MOD_D);
    const bool fastmode = openmx_env_flag_at(OPENMX_ENV_GEMMUL8_FASTMODE_D, openmx_env_flag_at(OPENMX_ENV_EXT_GEMMUL8_FASTMODE_D, false));
    void *work = nullptr;
    WorkspaceReport report;
    apply_memory_saving(handle);

    cublasStatus_t status =
        ensure_workspace<false>(handle, static_cast<size_t>(m), static_cast<size_t>(n), static_cast<size_t>(k),
                                num_moduli, fastmode, &work, &report);
    if (status == CUBLAS_STATUS_ALLOC_FAILED) {
        log_workspace_fallback_once<false>(report);
        *native = true;
        return cublasDgemm(handle, gemmul8_transa, gemmul8_transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }
    if (status != CUBLAS_STATUS_SUCCESS) {
        return status;
    }

    (void)gemmul8::gemm<double, gemmul8::Backend::INT8>(handle, gemmul8_transa, gemmul8_transb, static_cast<size_t>(m),
                                                        static_cast<size_t>(n), static_cast<size_t>(k), alpha, A,
                                                        static_cast<size_t>(lda), B, static_cast<size_t>(ldb), beta, C,
                                                        static_cast<size_t>(ldc), num_moduli, fastmode, work);

    return CUBLAS_STATUS_SUCCESS;
}

} // namespace

extern "C" cublasStatus_t openmx_gemmul8Dgemm(cublasHandle_t handle,
                                               cublasOperation_t transa,
                                               cublasOperation_t transb,
                                               int m,
                                               int n,
                                               int k,
                                               const double *alpha,
                                               const double *A,
                                               int lda,
                                               const double *B,
                                               int ldb,
                                               const double *beta,
                                               double *C,
                                               int ldc)
{
    if (m <= 0 || n <= 0 || k <= 0) {
        return CUBLAS_STATUS_SUCCESS;
    }
    if (forward_worker_stage && !inside_forward_tile)
        return forward_tile(handle,transa,transb,m,n,k,A,lda,B,ldb,C,ldc);
    cublasStatus_t mgpu_status;
    if (openmx_mgpu_dispatch_dgemm && openmx_mgpu_dispatch_dgemm(handle,
            transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, &mgpu_status))
        return mgpu_status;
    return timed_general_gemm(handle, 0, 2.0 * m * n * k, [&](bool *native) {
        return general_dgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, native);
    });
}

extern "C" void openmx_gemmul8ReleaseWorkspaces(void)
{
    std::lock_guard<std::mutex> lock(g_workspace_mutex);
    cudaError_t                 first_error = cudaSuccess;

    for (auto it = g_workspaces.begin(); it != g_workspaces.end();) {
        cudaError_t status = release_workspace(it->second);
        if (status == cudaSuccess) {
            it = g_workspaces.erase(it);
        } else {
            if (first_error == cudaSuccess) {
                first_error = status;
            }
            ++it;
        }
    }

    if (first_error != cudaSuccess) {
        std::fprintf(stderr,
                     "openmx_gemmul8ReleaseWorkspaces: cudaFree failed: %s\n",
                     cudaGetErrorString(first_error));
        std::fflush(stderr);
    }
}

/* Frees the scratch of unblocked forward products: every workspace larger
   than the cap of the blocked path.  The capped workspace of the other
   products stays, and without a cap nothing is freed. */
extern "C" void openmx_gemmul8TrimWorkspaces(void)
{
    const size_t                cap = memory_saving_cap_bytes();
    std::lock_guard<std::mutex> lock(g_workspace_mutex);

    if (cap == 0) return;
    for (auto &item : g_workspaces) {
        if (cap < item.second.size() && release_workspace(item.second) != cudaSuccess) {
            std::fprintf(stderr, "openmx_gemmul8TrimWorkspaces: cudaFree failed\n");
            std::fflush(stderr);
        }
    }
}

/* scf.gemmul8.enable: absent=-1 (device default), off=0, on=1. */
extern "C" void openmx_gemmul8SetEnabled(int enabled)
{
    g_input_enabled = enabled < 0 ? -1 : (enabled != 0);
}

/* General-GEMM policy only; large A100 Fixed transforms may select GEMMul8
   separately. A permitted call can still fall back for want of workspace. */
extern "C" int openmx_gemmul8ZgemmEnabled(void)
{
    return g_input_enabled != 0 && !gemmul8_disabled(OPENMX_ENV_GEMMUL8_DISABLE_Z, OPENMX_ENV_EXT_GEMMUL8_DISABLE_Z) ? 1 : 0;
}

/* the same for openmx_gemmul8Dgemm */
extern "C" int openmx_gemmul8DgemmEnabled(void)
{
    return g_input_enabled != 0 && !gemmul8_disabled(OPENMX_ENV_GEMMUL8_DISABLE_D, OPENMX_ENV_EXT_GEMMUL8_DISABLE_D) ? 1 : 0;
}

/* calls of openmx_gemmul8Dgemm (is_complex 0) or openmx_gemmul8Zgemm (1) so
   far that ran in plain cuBLAS FP64: switched off, or a workspace fallback */
extern "C" long long openmx_gemmul8NativeCalls(int is_complex)
{
    return g_general_native[is_complex ? 1 : 0].load(std::memory_order_relaxed);
}

extern "C" size_t openmx_gemmul8ZWorkspaceSize(int m, int n, int k)
{
    if (m <= 0 || n <= 0 || k <= 0 || g_input_enabled == 0 ||
        gemmul8_disabled(OPENMX_ENV_GEMMUL8_DISABLE_Z, OPENMX_ENV_EXT_GEMMUL8_DISABLE_Z)) {
        return 0;
    }

    const unsigned num_moduli = gemmul8_num_moduli(OPENMX_ENV_GEMMUL8_NUM_MOD_Z, OPENMX_ENV_EXT_GEMMUL8_NUM_MOD_Z);
    const bool fastmode = openmx_env_flag_at(OPENMX_ENV_GEMMUL8_FASTMODE_Z, openmx_env_flag_at(OPENMX_ENV_EXT_GEMMUL8_FASTMODE_Z, false));

    return capped_workspace_size(gemmul8::workSize<true, gemmul8::Backend::INT8>(
        static_cast<size_t>(m), static_cast<size_t>(n), static_cast<size_t>(k), num_moduli,
        false, false, nullptr, nullptr, fastmode));
}

extern "C" size_t openmx_gemmul8DWorkspaceSize(int m, int n, int k)
{
    if (m <= 0 || n <= 0 || k <= 0 || g_input_enabled == 0 ||
        gemmul8_disabled(OPENMX_ENV_GEMMUL8_DISABLE_D, OPENMX_ENV_EXT_GEMMUL8_DISABLE_D)) {
        return 0;
    }

    const unsigned num_moduli = gemmul8_num_moduli(OPENMX_ENV_GEMMUL8_NUM_MOD_D, OPENMX_ENV_EXT_GEMMUL8_NUM_MOD_D);
    const bool fastmode = openmx_env_flag_at(OPENMX_ENV_GEMMUL8_FASTMODE_D, openmx_env_flag_at(OPENMX_ENV_EXT_GEMMUL8_FASTMODE_D, false));

    return capped_workspace_size(gemmul8::workSize<false, gemmul8::Backend::INT8>(
        static_cast<size_t>(m), static_cast<size_t>(n), static_cast<size_t>(k), num_moduli,
        false, false, nullptr, nullptr, fastmode));
}

namespace {

cublasStatus_t general_zgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n,
                             int k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda,
                             const cuDoubleComplex *B, int ldb, const cuDoubleComplex *beta, cuDoubleComplex *C,
                             int ldc, bool *native)
{
    if (g_input_enabled == 0) {
        /* scf.gemmul8.enable off: the fallback is what the user asked for,
           so no warning (Input_std already reported it once) */
        *native = true;
        return cublasZgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }

    if (gemmul8_disabled(OPENMX_ENV_GEMMUL8_DISABLE_Z, OPENMX_ENV_EXT_GEMMUL8_DISABLE_Z)) {
        *native = true;
        return cublasZgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }

    const unsigned num_moduli = gemmul8_num_moduli(OPENMX_ENV_GEMMUL8_NUM_MOD_Z, OPENMX_ENV_EXT_GEMMUL8_NUM_MOD_Z);
    const bool fastmode = openmx_env_flag_at(OPENMX_ENV_GEMMUL8_FASTMODE_Z, openmx_env_flag_at(OPENMX_ENV_EXT_GEMMUL8_FASTMODE_Z, false));
    void *work = nullptr;
    WorkspaceReport report;
    apply_memory_saving(handle);

    cublasStatus_t status =
        ensure_workspace<true>(handle, static_cast<size_t>(m), static_cast<size_t>(n), static_cast<size_t>(k),
                               num_moduli, fastmode, &work, &report);
    if (status == CUBLAS_STATUS_ALLOC_FAILED) {
        log_workspace_fallback_once<true>(report);
        *native = true;
        return cublasZgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }
    if (status != CUBLAS_STATUS_SUCCESS) {
        return status;
    }

    (void)gemmul8::gemm<cuDoubleComplex, gemmul8::Backend::INT8>(
        handle, transa, transb, static_cast<size_t>(m), static_cast<size_t>(n), static_cast<size_t>(k), alpha, A,
        static_cast<size_t>(lda), B, static_cast<size_t>(ldb), beta, C, static_cast<size_t>(ldc), num_moduli, fastmode,
        work);

    return CUBLAS_STATUS_SUCCESS;
}

} // namespace

extern "C" cublasStatus_t openmx_gemmul8Zgemm(cublasHandle_t handle,
                                               cublasOperation_t transa,
                                               cublasOperation_t transb,
                                               int m,
                                               int n,
                                               int k,
                                               const cuDoubleComplex *alpha,
                                               const cuDoubleComplex *A,
                                               int lda,
                                               const cuDoubleComplex *B,
                                               int ldb,
                                               const cuDoubleComplex *beta,
                                               cuDoubleComplex *C,
                                               int ldc)
{
    if (m <= 0 || n <= 0 || k <= 0) {
        return CUBLAS_STATUS_SUCCESS;
    }
    if (forward_worker_stage && !inside_forward_tile)
        return forward_tile(handle,transa,transb,m,n,k,A,lda,B,ldb,C,ldc);
    cublasStatus_t mgpu_status;
    if (openmx_mgpu_dispatch_zgemm && openmx_mgpu_dispatch_zgemm(handle,
            transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, &mgpu_status))
        return mgpu_status;
    return timed_general_gemm(handle, 1, 8.0 * m * n * k, [&](bool *native) {
        return general_zgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, native);
    });
}

/* ------------------------------------------------------------------------
 * Forward transform of the dense eigensolver: B = H X and C = X^† B.
 *
 * X (the transformed overlap) stays fixed during an SCF, while H changes
 * every step.  These two products get their own precision stage, separate
 * from every other GEMM routed through this bridge, and X's prepared
 * (scaled, residue) form can be retained between calls with GEMMul8's
 * skip_scal mechanism.  The caller names each X with an id and a version
 * number that it bumps whenever the contents change; a retained form is
 * only reused while version, shape, operation, moduli count and scaling
 * mode all match, never because a pointer compares equal.
 *
 * Stage (openmx_gemmul8SetForwardStage, or the environment until it is
 * called):
 *   OPENMX_GEMMUL8_FORWARD            default | fp64 | gemmul8
 *     default  the settings of openmx_gemmul8{D,Z}gemm
 *     fp64     plain cuBLAS FP64
 *     gemmul8  GEMMul8 with the settings below
 *   OPENMX_GEMMUL8_FORWARD_NUM_MOD    moduli count (2..20, default 15)
 *   OPENMX_GEMMUL8_FORWARD_FASTMODE   1: fast scaling (default 0)
 *   OPENMX_GEMMUL8_FORWARD_REUSE      1: retain the prepared X (default 0)
 *   OPENMX_GEMMUL8_FORWARD_UNBLOCKED  0: always the capped, blocked workspace
 *                                     of openmx_gemmul8{D,Z}gemm (default 1,
 *                                     and implied by reuse: GEMMul8 disables
 *                                     skip_scal in its memory-saving mode)
 *   OPENMX_GEMMUL8_FORWARD_TIMING     1: synchronize around each call and
 *                                     report the accumulated wall time
 * By default these two products take the full workspace whenever the device
 * has room for it: only the rank that owns the dense solve reaches them, and
 * under the cap every block of the blocked product rescales its operands,
 * which doubles the time of a large product.  A retained form or a full
 * workspace that does not fit falls back to the next path of the same
 * precision (re-preparing every call, then blocked), and is counted;
 * capacity never lowers the precision.
 * ---------------------------------------------------------------------- */
namespace {

enum : int { kForwardDefault = 0, kForwardFp64 = 1, kForwardGemmul8 = 2 };

struct ForwardStage {
    int      mode       = kForwardDefault;
    unsigned num_moduli = kDefaultNumModuli;
    bool     fastmode   = false;
    bool     reuse      = false;
    bool     unblocked  = true;
};

struct PreparedKey {
    int device;
    int id;
    int x_is_left;
    int is_complex;
    const void *owner = nullptr;

    bool operator==(const PreparedKey &other) const
    {
        return device == other.device && id == other.id && x_is_left == other.x_is_left &&
               is_complex == other.is_complex && owner == other.owner;
    }
};

struct PreparedKeyHash {
    std::size_t operator()(const PreparedKey &key) const
    {
        return std::hash<const void*>{}(key.owner) ^
               (static_cast<std::size_t>(key.device) << 40) ^ (static_cast<std::size_t>(key.id) << 8) ^
               (static_cast<std::size_t>(key.x_is_left) << 1) ^ static_cast<std::size_t>(key.is_complex);
    }
};

struct Prepared {
    openmx::gpu::DeviceBuffer<unsigned char> buffer;
    bool               valid = false;
    unsigned long long version = 0;
    int                m = 0, n = 0, k = 0, op = 0;
    unsigned           num_moduli = 0;
    bool               fastmode   = false;
};

struct ForwardCounters {
    long long calls[2]           = {0, 0}; /* X on the right, X on the left */
    double    seconds[2]         = {0.0, 0.0};
    long long prepared           = 0;
    long long reused             = 0;
    long long capacity_fallbacks = 0;
};

ForwardStage    g_forward_stage;
bool            g_forward_stage_set = false;
ForwardCounters g_forward_counters;
std::unordered_map<PreparedKey, Prepared, PreparedKeyHash> g_prepared;

ForwardStage forward_stage()
{
    std::lock_guard<std::mutex> lock(g_workspace_mutex);

    if (!g_forward_stage_set) {
        const char *mode = openmx_env_at(OPENMX_ENV_GEMMUL8_FORWARD);
        ForwardStage stage;

        if (mode != nullptr && std::strcmp(mode, "fp64") == 0) stage.mode = kForwardFp64;
        if (mode != nullptr && std::strcmp(mode, "gemmul8") == 0) stage.mode = kForwardGemmul8;
        stage.num_moduli = gemmul8_num_moduli(OPENMX_ENV_GEMMUL8_FORWARD_NUM_MOD, OPENMX_ENV_GEMMUL8_FORWARD_NUM_MOD);
        stage.fastmode   = openmx_env_flag_at(OPENMX_ENV_GEMMUL8_FORWARD_FASTMODE, false);
        stage.reuse      = openmx_env_flag_at(OPENMX_ENV_GEMMUL8_FORWARD_REUSE, false);
        stage.unblocked  = stage.reuse || openmx_env_flag_at(OPENMX_ENV_GEMMUL8_FORWARD_UNBLOCKED, true);
        g_forward_stage     = stage;
        g_forward_stage_set = true;
    }
    return g_forward_stage;
}

/* Scratch of an unblocked call: the shared per-(device, stream) workspace
   grown to `required` bytes.  Only the free-memory reserve applies; the
   rank-share policy of ensure_workspace() assumes every rank holds such a
   workspace, whereas only the dense-owning rank reaches these products. */
cublasStatus_t ensure_scratch(cublasHandle_t handle, size_t required, void **work, bool automatic_forward = false)
try {
    cudaStream_t stream = nullptr;
    int          device = -1;

    cublasStatus_t status = cublasGetStream(handle, &stream);
    if (status != CUBLAS_STATUS_SUCCESS) return status;
    if (cudaGetDevice(&device) != cudaSuccess) return CUBLAS_STATUS_INTERNAL_ERROR;

    const size_t reserve =
        env_mib(OPENMX_ENV_GEMMUL8_MIN_FREE_AFTER_MB, OPENMX_ENV_EXT_GEMMUL8_MIN_FREE_AFTER_MB, kDefaultMinFreeAfterMiB);
    std::lock_guard<std::mutex> lock(g_workspace_mutex);
    Workspace &workspace = g_workspaces[WorkspaceKey{device, stream}];

    if (automatic_forward) {
        size_t free_bytes = 0, total_bytes = 0;
        unsigned percent = env_percent(OPENMX_ENV_GEMMUL8_MAX_WORKSPACE_PERCENT, OPENMX_ENV_EXT_GEMMUL8_MAX_WORKSPACE_PERCENT,
                                       0u);
        /* Only dense owners reach this short-lived scratch (one, or two
           spin owners for a cluster), not every local MPI rank. Follow
           the existing full-workspace free/reserve budget by default;
           honor an explicit percentage cap with its existing rank share. */
        if (device_mem_get_info(&free_bytes, &total_bytes) != cudaSuccess || total_bytes == 0 ||
            (percent != 0 && workspace_exceeds_fraction(required, total_bytes, percent, ranks_sharing_gpu())) ||
            free_after_workspace_is_too_low(free_bytes, workspace.size(), required, reserve)) {
            if (release_workspace(workspace) != cudaSuccess) return CUBLAS_STATUS_INTERNAL_ERROR;
            return CUBLAS_STATUS_ALLOC_FAILED;
        }
    }

    if (workspace.size() < required) {
        size_t free_bytes = 0, total_bytes = 0;

        if (release_workspace(workspace) != cudaSuccess) return CUBLAS_STATUS_INTERNAL_ERROR;
        const cudaError_t memory_status = device_mem_get_info(&free_bytes, &total_bytes);
        if ((automatic_forward && memory_status != cudaSuccess) ||
            (memory_status == cudaSuccess && free_bytes < required + reserve))
            return CUBLAS_STATUS_ALLOC_FAILED;
        if (workspace.allocate_status(required) != cudaSuccess) {
            (void)cudaGetLastError();
            return CUBLAS_STATUS_ALLOC_FAILED;
        }
    }
    *work = workspace.get();
    return CUBLAS_STATUS_SUCCESS;
} catch (const std::bad_alloc &) {
    return CUBLAS_STATUS_ALLOC_FAILED;
}

/* The retained form of one X, at least `bytes` large; nullptr when the
   device cannot hold it.  Growing it discards the old contents. */
Prepared *ensure_prepared(const PreparedKey &key, size_t bytes)
try {
    const size_t reserve =
        env_mib(OPENMX_ENV_GEMMUL8_MIN_FREE_AFTER_MB, OPENMX_ENV_EXT_GEMMUL8_MIN_FREE_AFTER_MB, kDefaultMinFreeAfterMiB);
    std::lock_guard<std::mutex> lock(g_workspace_mutex);
    Prepared &entry = g_prepared[key];

    if (entry.buffer.size() < bytes) {
        size_t free_bytes = 0, total_bytes = 0;

        if (entry.buffer.try_reset() != cudaSuccess) {entry.valid=false;return nullptr;}
        entry = Prepared{};
        if (device_mem_get_info(&free_bytes, &total_bytes) == cudaSuccess && free_bytes < bytes + reserve) return nullptr;
        if (entry.buffer.allocate_status(bytes) != cudaSuccess) {
            (void)cudaGetLastError();
            return nullptr;
        }
    }
    return &entry;
} catch (const std::bad_alloc &) {
    return nullptr;
}

template <typename T> T forward_scalar(double value)
{
    if constexpr (std::is_same_v<T, double>) return value;
    else return make_cuDoubleComplex(value, 0.0);
}

template <typename T>
cublasStatus_t native_gemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n,
                           int k, const T *alpha, const T *A, int lda, const T *B, int ldb, const T *beta, T *C,
                           int ldc)
{
    if constexpr (std::is_same_v<T, double>) {
        return cublasDgemm(handle, transa == CUBLAS_OP_C ? CUBLAS_OP_T : transa,
                           transb == CUBLAS_OP_C ? CUBLAS_OP_T : transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    } else {
        return cublasZgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }
}

/* C = op_x(X) V (x_is_left) or C = V op_x(X) with the precision stage of
   the forward transform */
template <typename T>
cublasStatus_t forward_gemm(cublasHandle_t handle, bool x_is_left, cublasOperation_t op_x, int m, int n, int k,
                            const T *X, int ldx, const T *V, int ldv, T *C, int ldc, int x_id,
                            unsigned long long x_version, const void *owner = nullptr)
{
    constexpr bool          is_complex = !std::is_same_v<T, double>;
    const T                 one = forward_scalar<T>(1.0), zero = forward_scalar<T>(0.0);
    const cublasOperation_t transa = x_is_left ? op_x : CUBLAS_OP_N;
    const cublasOperation_t transb = x_is_left ? CUBLAS_OP_N : op_x;
    const T *const          A = x_is_left ? X : V;
    const T *const          B = x_is_left ? V : X;
    const int               lda = x_is_left ? ldx : ldv, ldb = x_is_left ? ldv : ldx;
    ForwardStage            stage = forward_stage();
    /* Tile buffers are reused for different slices. Their pointer and the
       parent's X version do not identify their contents; never retain a
       prepared X across tile uploads. Single-device reuse is unchanged. */
    if (inside_forward_tile) stage.reuse=false;

    if (m <= 0 || n <= 0 || k <= 0) return CUBLAS_STATUS_SUCCESS;

    const bool timing = openmx_env_flag_at(OPENMX_ENV_GEMMUL8_FORWARD_TIMING, 0) != 0;
    cudaStream_t      stream = nullptr;
    cublasStatus_t    status = cublasGetStream(handle, &stream);
    if (status != CUBLAS_STATUS_SUCCESS) return status;
    if (timing && cudaStreamSynchronize(stream) != cudaSuccess) return CUBLAS_STATUS_INTERNAL_ERROR;
    const auto start = std::chrono::steady_clock::now();

    /* the default stage follows the settings of openmx_gemmul8{D,Z}gemm */
    const bool default_stage = stage.mode == kForwardDefault;
    const bool reuse_stage   = !default_stage && stage.reuse;
    if (!inside_forward_tile) {
        struct ForwardWorkerStage {
            bool active,left;
            ForwardWorkerStage(bool use,bool xleft):active(forward_worker_stage),left(forward_worker_x_left) {
                forward_worker_stage=use;forward_worker_x_left=xleft;
            }
            ~ForwardWorkerStage(){forward_worker_stage=active;forward_worker_x_left=left;}
        } worker_stage(!default_stage,x_is_left);
        if constexpr (is_complex) {
            if (openmx_mgpu_dispatch_zgemm && openmx_mgpu_dispatch_zgemm(handle,
                    transa, transb, m, n, k, &one, A, lda, B, ldb, &zero, C, ldc, &status))
                return status;
        } else {
            if (openmx_mgpu_dispatch_dgemm && openmx_mgpu_dispatch_dgemm(handle,
                    transa, transb, m, n, k, &one, A, lda, B, ldb, &zero, C, ldc, &status))
                return status;
        }
    }
    unsigned   num_moduli    = stage.num_moduli;
    bool       fastmode      = stage.fastmode;
    bool       native        = stage.mode == kForwardFp64 || g_input_enabled == 0;

    if (default_stage) {
        num_moduli = gemmul8_num_moduli(is_complex ? OPENMX_ENV_GEMMUL8_NUM_MOD_Z : OPENMX_ENV_GEMMUL8_NUM_MOD_D,
                                        is_complex ? OPENMX_ENV_EXT_GEMMUL8_NUM_MOD_Z : OPENMX_ENV_EXT_GEMMUL8_NUM_MOD_D);
        fastmode   = openmx_env_flag_at(is_complex ? OPENMX_ENV_GEMMUL8_FASTMODE_Z : OPENMX_ENV_GEMMUL8_FASTMODE_D,
                              openmx_env_flag_at(is_complex ? OPENMX_ENV_EXT_GEMMUL8_FASTMODE_Z : OPENMX_ENV_EXT_GEMMUL8_FASTMODE_D, false));
        native     = native || gemmul8_disabled(is_complex ? OPENMX_ENV_GEMMUL8_DISABLE_Z : OPENMX_ENV_GEMMUL8_DISABLE_D,
                                                is_complex ? OPENMX_ENV_EXT_GEMMUL8_DISABLE_Z : OPENMX_ENV_EXT_GEMMUL8_DISABLE_D);
    }
    const bool automatic_forward = !is_complex && default_stage && native && stage.unblocked &&
                                   a100_large_forward_candidate(m, n, k);
    if (automatic_forward) native = false;
    const auto general_entry = [&]() {
        if constexpr (is_complex) {
            return openmx_gemmul8Zgemm(handle, transa, transb, m, n, k, &one, A, lda, B, ldb, &zero, C, ldc);
        } else {
            return openmx_gemmul8Dgemm(handle, transa, transb, m, n, k, &one, A, lda, B, ldb, &zero, C, ldc);
        }
    };

    if (default_stage && (native || !stage.unblocked)) {
        status = general_entry();
    } else if (native) {
        status = native_gemm<T>(handle, transa, transb, m, n, k, &one, A, lda, B, ldb, &zero, C, ldc);
    } else {
        const cublasOperation_t gemmul8_transa = (!is_complex && transa == CUBLAS_OP_C) ? CUBLAS_OP_T : transa;
        const cublasOperation_t gemmul8_transb = (!is_complex && transb == CUBLAS_OP_C) ? CUBLAS_OP_T : transb;
        const size_t            sm = static_cast<size_t>(m), sn = static_cast<size_t>(n), sk = static_cast<size_t>(k);
        void                   *work = nullptr;
        bool                    done = false;

        if (stage.unblocked) {
            int       device = -1;
            Prepared *entry = nullptr;
            size_t    keep = 0, size_a = 0, size_b = 0;
            size_t    total = gemmul8::workSize<is_complex, gemmul8::Backend::INT8>(
                sm, sn, sk, num_moduli, reuse_stage && x_is_left, reuse_stage && !x_is_left, &size_a, &size_b,
                fastmode);

            if (reuse_stage && cudaGetDevice(&device) == cudaSuccess) {
                keep  = x_is_left ? size_a : size_b;
                entry = ensure_prepared(PreparedKey{device, x_id, x_is_left ? 1 : 0, is_complex ? 1 : 0, owner}, keep);
            }
            if (reuse_stage && entry == nullptr) {
                /* no room for the retained form: prepare X on every call */
                keep  = 0;
                total = gemmul8::workSize<is_complex, gemmul8::Backend::INT8>(sm, sn, sk, num_moduli, false, false,
                                                                              nullptr, nullptr, fastmode);
                std::lock_guard<std::mutex> lock(g_workspace_mutex);
                ++g_forward_counters.capacity_fallbacks;
            }

            status = ensure_scratch(handle, total - keep, &work, automatic_forward);
            if (status == CUBLAS_STATUS_SUCCESS) {
                const bool reuse = entry != nullptr;
                const bool skip  = reuse && entry->valid && entry->version == x_version && entry->m == m &&
                                   entry->n == n && entry->k == k && entry->op == static_cast<int>(op_x) &&
                                   entry->num_moduli == num_moduli && entry->fastmode == fastmode;

                gemmul8::set_memory_saving(handle, false);
                (void)gemmul8::gemm<T, gemmul8::Backend::INT8>(
                    handle, gemmul8_transa, gemmul8_transb, sm, sn, sk, &one, A, static_cast<size_t>(lda), B,
                    static_cast<size_t>(ldb), &zero, C, static_cast<size_t>(ldc), static_cast<int>(num_moduli),
                    fastmode, work, (reuse && x_is_left) ? entry->buffer.get() : nullptr,
                    (reuse && !x_is_left) ? entry->buffer.get() : nullptr, reuse && x_is_left, reuse && !x_is_left,
                    skip && x_is_left, skip && !x_is_left);

                std::lock_guard<std::mutex> lock(g_workspace_mutex);
                if (skip) {
                    ++g_forward_counters.reused;
                } else if (reuse) {
                    entry->valid      = true;
                    entry->version    = x_version;
                    entry->m          = m;
                    entry->n          = n;
                    entry->k          = k;
                    entry->op         = static_cast<int>(op_x);
                    entry->num_moduli = num_moduli;
                    entry->fastmode   = fastmode;
                    ++g_forward_counters.prepared;
                }
                done = true;
            } else if (status == CUBLAS_STATUS_ALLOC_FAILED) {
                /* no room for the full workspace either: blocked path below */
                std::lock_guard<std::mutex> lock(g_workspace_mutex);
                ++g_forward_counters.capacity_fallbacks;
            } else {
                return status;
            }
        }

        if (!done && default_stage) {
            status = general_entry();
        } else if (!done) {
            WorkspaceReport report;

            apply_memory_saving(handle);
            status = ensure_workspace<is_complex>(handle, sm, sn, sk, num_moduli, fastmode, &work, &report);
            if (status == CUBLAS_STATUS_ALLOC_FAILED) {
                log_workspace_fallback_once<is_complex>(report);
                status = native_gemm<T>(handle, transa, transb, m, n, k, &one, A, lda, B, ldb, &zero, C, ldc);
            } else if (status == CUBLAS_STATUS_SUCCESS) {
                (void)gemmul8::gemm<T, gemmul8::Backend::INT8>(
                    handle, gemmul8_transa, gemmul8_transb, sm, sn, sk, &one, A, static_cast<size_t>(lda), B,
                    static_cast<size_t>(ldb), &zero, C, static_cast<size_t>(ldc), static_cast<int>(num_moduli),
                    fastmode, work);
            }
        }
    }
    if (status != CUBLAS_STATUS_SUCCESS) return status;

    if (timing && cudaStreamSynchronize(stream) != cudaSuccess) return CUBLAS_STATUS_INTERNAL_ERROR;
    std::lock_guard<std::mutex> lock(g_workspace_mutex);
    ++g_forward_counters.calls[x_is_left ? 1 : 0];
    if (timing)
        g_forward_counters.seconds[x_is_left ? 1 : 0] +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return CUBLAS_STATUS_SUCCESS;
}

template<class T> cublasStatus_t forward_tile(cublasHandle_t handle,cublasOperation_t ta,cublasOperation_t tb,
    int m,int n,int k,const T *a,int lda,const T *b,int ldb,T *c,int ldc)
{
    struct TileGuard {
        bool previous;
        TileGuard():previous(inside_forward_tile){inside_forward_tile=true;}
        ~TileGuard(){inside_forward_tile=previous;}
    } guard;
    const bool left=forward_worker_x_left;
    return forward_gemm(handle,left,left?ta:tb,m,n,k,left?a:b,left?lda:ldb,
                        left?b:a,left?ldb:lda,c,ldc,0,0);
}

} // namespace

/* COSMA's local tile products of the gpusolver2 transforms.  Tiled-MM,
   patched at stack-build time (third_party/tiledmm-openmx-gemm-hook.patch),
   calls these entry points instead of cuBLAS when they are linked in, so
   the distributed transforms take the general-GEMM policy above: GEMMul8
   on GPUs with weak FP64 (the GeForce default), native FP64 on A100/H100,
   scf.gemmul8.enable and the workspace guards as for every other product.
   Tiled-MM binds one cuBLAS handle to each of its streams, so each stream
   keeps its own capped workspace (its tiles are at most 1024 wide,
   OPENMX_GS2_GPU_TILE).  OPENMX_GS2_GEMMUL8=0 keeps COSMA on cuBLAS FP64
   for comparison.  Counted and timed with the general products of
   OPENMX_GEMMUL8_TIMING. */
namespace {
bool cosma_products_through_bridge()
{
    static const bool through = openmx_env_flag_at(OPENMX_ENV_GS2_GEMMUL8, 1) != 0;
    return through;
}
} // namespace

extern "C" cublasStatus_t openmx_tiledmm_dgemm(cublasHandle_t handle, cublasOperation_t transa,
                                                cublasOperation_t transb, int m, int n, int k,
                                                const double *alpha, const double *A, int lda,
                                                const double *B, int ldb, const double *beta,
                                                double *C, int ldc)
{
    if (!cosma_products_through_bridge())
        return cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    return openmx_gemmul8Dgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

extern "C" cublasStatus_t openmx_tiledmm_zgemm(cublasHandle_t handle, cublasOperation_t transa,
                                                cublasOperation_t transb, int m, int n, int k,
                                                const cuDoubleComplex *alpha, const cuDoubleComplex *A,
                                                int lda, const cuDoubleComplex *B, int ldb,
                                                const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc)
{
    if (!cosma_products_through_bridge())
        return cublasZgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    return openmx_gemmul8Zgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

extern "C" cublasStatus_t openmx_gemmul8DgemmFixed(cublasHandle_t handle, int x_is_left, cublasOperation_t op_x,
                                                    int m, int n, int k, const double *X, int ldx, const double *V,
                                                    int ldv, double *C, int ldc, int x_id,
                                                    unsigned long long x_version)
{
    return forward_gemm<double>(handle, x_is_left != 0, op_x, m, n, k, X, ldx, V, ldv, C, ldc, x_id, x_version);
}

/* An explicit calculation can use a local generation counter. Distinct
   calculations never share a prepared operand, even at the same generation.
   Erase its entries before destroying/reusing the owner identity. */
extern "C" cublasStatus_t openmx_gemmul8DgemmFixedOwned(const void *owner,
    cublasHandle_t handle, int x_is_left, cublasOperation_t op_x, int m, int n, int k,
    const double *X, int ldx, const double *V, int ldv, double *C, int ldc, int x_id,
    unsigned long long x_version)
{
    return forward_gemm<double>(handle, x_is_left != 0, op_x, m, n, k, X, ldx, V, ldv,
                                C, ldc, x_id, x_version, owner);
}

extern "C" void openmx_gemmul8ReleasePreparedOwner(const void *owner)
{
    std::lock_guard<std::mutex> lock(g_workspace_mutex);
    for (auto it = g_prepared.begin(); it != g_prepared.end();) {
        if (it->first.owner == owner) it = g_prepared.erase(it);
        else ++it;
    }
}

extern "C" cublasStatus_t openmx_gemmul8ZgemmFixed(cublasHandle_t handle, int x_is_left, cublasOperation_t op_x,
                                                    int m, int n, int k, const cuDoubleComplex *X, int ldx,
                                                    const cuDoubleComplex *V, int ldv, cuDoubleComplex *C, int ldc,
                                                    int x_id, unsigned long long x_version)
{
    return forward_gemm<cuDoubleComplex>(handle, x_is_left != 0, op_x, m, n, k, X, ldx, V, ldv, C, ldc, x_id,
                                         x_version);
}

/* mode: 0 default, 1 fp64, 2 gemmul8 (see the block comment above) */
/* openmx_gemmul8Zgemm with the full GEMMul8 workspace when the device has
   room for it (the scratch of the forward transform; free memory of at
   least the workspace plus the reserve of OPENMX_GEMMUL8_MIN_FREE_AFTER_MB),
   so that no block rescales its operands: about 1.7x faster than the
   blocked product at the 256 MiB cap for the n2 x MaxN x n2 products of the
   refined eigensolver.  Otherwise, and when GEMMul8 is off, the same as
   openmx_gemmul8Zgemm, native fallbacks counted alike. */
extern "C" cublasStatus_t openmx_gemmul8ZgemmUnblocked(cublasHandle_t handle, cublasOperation_t transa,
                                                       cublasOperation_t transb, int m, int n, int k,
                                                       const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda,
                                                       const cuDoubleComplex *B, int ldb, const cuDoubleComplex *beta,
                                                       cuDoubleComplex *C, int ldc)
{
    if (m <= 0 || n <= 0 || k <= 0) return CUBLAS_STATUS_SUCCESS;
    cublasStatus_t mgpu_status;
    if (openmx_mgpu_dispatch_zgemm && openmx_mgpu_dispatch_zgemm(handle,
            transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, &mgpu_status))
        return mgpu_status;
    if (g_input_enabled == 0 || gemmul8_disabled(OPENMX_ENV_GEMMUL8_DISABLE_Z, OPENMX_ENV_EXT_GEMMUL8_DISABLE_Z))
        return openmx_gemmul8Zgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    return timed_general_gemm(handle, 1, 8.0 * m * n * k, [&](bool *native) -> cublasStatus_t {
        const unsigned num_moduli = gemmul8_num_moduli(OPENMX_ENV_GEMMUL8_NUM_MOD_Z, OPENMX_ENV_EXT_GEMMUL8_NUM_MOD_Z);
        const bool     fastmode   = openmx_env_flag_at(OPENMX_ENV_GEMMUL8_FASTMODE_Z, openmx_env_flag_at(OPENMX_ENV_EXT_GEMMUL8_FASTMODE_Z, false));
        const size_t   sm = static_cast<size_t>(m), sn = static_cast<size_t>(n), sk = static_cast<size_t>(k);
        const size_t   total = gemmul8::workSize<true, gemmul8::Backend::INT8>(sm, sn, sk, num_moduli, false, false,
                                                                                nullptr, nullptr, fastmode);
        void          *work = nullptr;
        cublasStatus_t status = ensure_scratch(handle, total, &work);

        if (status == CUBLAS_STATUS_SUCCESS) {
            gemmul8::set_memory_saving(handle, false);
            (void)gemmul8::gemm<cuDoubleComplex, gemmul8::Backend::INT8>(
                handle, transa, transb, sm, sn, sk, alpha, A, static_cast<size_t>(lda), B, static_cast<size_t>(ldb),
                beta, C, static_cast<size_t>(ldc), num_moduli, fastmode, work);
            return CUBLAS_STATUS_SUCCESS;
        }
        if (status != CUBLAS_STATUS_ALLOC_FAILED) return status;
        return general_zgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, native);
    });
}

/* the same for real products (openmx_gemmul8Dgemm) */
extern "C" cublasStatus_t openmx_gemmul8DgemmUnblocked(cublasHandle_t handle, cublasOperation_t transa,
                                                       cublasOperation_t transb, int m, int n, int k,
                                                       const double *alpha, const double *A, int lda, const double *B,
                                                       int ldb, const double *beta, double *C, int ldc)
{
    if (m <= 0 || n <= 0 || k <= 0) return CUBLAS_STATUS_SUCCESS;
    cublasStatus_t mgpu_status;
    if (openmx_mgpu_dispatch_dgemm && openmx_mgpu_dispatch_dgemm(handle,
            transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, &mgpu_status))
        return mgpu_status;
    if (g_input_enabled == 0 || gemmul8_disabled(OPENMX_ENV_GEMMUL8_DISABLE_D, OPENMX_ENV_EXT_GEMMUL8_DISABLE_D))
        return openmx_gemmul8Dgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    return timed_general_gemm(handle, 0, 2.0 * m * n * k, [&](bool *native) -> cublasStatus_t {
        const unsigned num_moduli = gemmul8_num_moduli(OPENMX_ENV_GEMMUL8_NUM_MOD_D, OPENMX_ENV_EXT_GEMMUL8_NUM_MOD_D);
        const bool     fastmode   = openmx_env_flag_at(OPENMX_ENV_GEMMUL8_FASTMODE_D, openmx_env_flag_at(OPENMX_ENV_EXT_GEMMUL8_FASTMODE_D, false));
        const cublasOperation_t ta = (transa == CUBLAS_OP_C) ? CUBLAS_OP_T : transa;
        const cublasOperation_t tb = (transb == CUBLAS_OP_C) ? CUBLAS_OP_T : transb;
        const size_t   sm = static_cast<size_t>(m), sn = static_cast<size_t>(n), sk = static_cast<size_t>(k);
        const size_t   total = gemmul8::workSize<false, gemmul8::Backend::INT8>(sm, sn, sk, num_moduli, false, false,
                                                                                 nullptr, nullptr, fastmode);
        void          *work = nullptr;
        cublasStatus_t status = ensure_scratch(handle, total, &work);

        if (status == CUBLAS_STATUS_SUCCESS) {
            gemmul8::set_memory_saving(handle, false);
            (void)gemmul8::gemm<double, gemmul8::Backend::INT8>(
                handle, ta, tb, sm, sn, sk, alpha, A, static_cast<size_t>(lda), B, static_cast<size_t>(ldb), beta, C,
                static_cast<size_t>(ldc), num_moduli, fastmode, work);
            return CUBLAS_STATUS_SUCCESS;
        }
        if (status != CUBLAS_STATUS_ALLOC_FAILED) return status;
        return general_dgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, native);
    });
}

extern "C" void openmx_gemmul8SetForwardStage(int mode, int num_moduli, int fastmode, int reuse, int unblocked)
{
    std::lock_guard<std::mutex> lock(g_workspace_mutex);
    ForwardStage                stage;

    stage.mode = (mode == kForwardFp64 || mode == kForwardGemmul8) ? mode : kForwardDefault;
    if (2 <= num_moduli && num_moduli <= static_cast<int>(kMaxNumModuli))
        stage.num_moduli = static_cast<unsigned>(num_moduli);
    stage.fastmode      = fastmode != 0;
    stage.reuse         = reuse != 0;
    stage.unblocked     = stage.reuse || unblocked != 0;
    g_forward_stage     = stage;
    g_forward_stage_set = true;
}

/* counters[0..4]: calls with X on the right, calls with X on the left,
   preparations, reuses, capacity fallbacks; seconds[0..1]: wall time per
   side (OPENMX_GEMMUL8_FORWARD_TIMING=1); returns the retained bytes */
extern "C" size_t openmx_gemmul8ForwardCounters(long long counters[5], double seconds[2])
{
    std::lock_guard<std::mutex> lock(g_workspace_mutex);
    size_t                      bytes = 0;

    counters[0] = g_forward_counters.calls[0];
    counters[1] = g_forward_counters.calls[1];
    counters[2] = g_forward_counters.prepared;
    counters[3] = g_forward_counters.reused;
    counters[4] = g_forward_counters.capacity_fallbacks;
    seconds[0]  = g_forward_counters.seconds[0];
    seconds[1]  = g_forward_counters.seconds[1];
    for (const auto &item : g_prepared) bytes += item.second.buffer.size();
    return bytes;
}

/* Frees every retained form of X; the next forward product prepares anew.
   With OPENMX_GEMMUL8_FORWARD_TIMING or verbose logging, a rank that ran
   forward products since the last release reports and resets its counters. */
extern "C" void openmx_gemmul8ReleasePrepared(void)
{
    std::lock_guard<std::mutex> lock(g_workspace_mutex);
    const ForwardCounters      &c = g_forward_counters;
    size_t                      bytes = 0;

    for (auto &item : g_prepared) {
        bytes += item.second.buffer.size();
        item.second.buffer.reset();
    }
    g_prepared.clear();

    if (c.calls[0] + c.calls[1] > 0 &&
        (verbose_logging_enabled() || openmx_env_flag_at(OPENMX_ENV_GEMMUL8_FORWARD_TIMING, false))) {
        const ForwardStage &s = g_forward_stage;
        /* the default stage uses the settings of the real products */
        const bool     general  = s.mode == kForwardDefault;
        const unsigned moduli   = general ? gemmul8_num_moduli(OPENMX_ENV_GEMMUL8_NUM_MOD_D, OPENMX_ENV_EXT_GEMMUL8_NUM_MOD_D)
                                          : s.num_moduli;
        const bool     fastmode = general ? openmx_env_flag_at(OPENMX_ENV_GEMMUL8_FASTMODE_D, openmx_env_flag_at(OPENMX_ENV_EXT_GEMMUL8_FASTMODE_D, false))
                                          : s.fastmode;
        std::printf("<openmx_gemmul8> forward transform: mode=%s moduli=%u scaling=%s reuse=%d unblocked=%d; "
                    "calls %lld + %lld, prepared %lld, reused %lld, capacity fallbacks %lld, retained %.1f MiB, "
                    "wall %.6f + %.6f s\n",
                    s.mode == kForwardFp64 ? "fp64" : (s.mode == kForwardGemmul8 ? "gemmul8" : "default"),
                    moduli, fastmode ? "fast" : "accurate", int(s.reuse), int(s.unblocked), c.calls[0],
                    c.calls[1], c.prepared, c.reused, c.capacity_fallbacks, double(bytes) / double(kMiB),
                    c.seconds[0], c.seconds[1]);
        std::fflush(stdout);
    }
    g_forward_counters = ForwardCounters{};

    const GeneralCounters &g = g_general_counters;
    if (g.calls[0][0] + g.calls[0][1] + g.calls[1][0] + g.calls[1][1] > 0 &&
        openmx_env_flag_at(OPENMX_ENV_GEMMUL8_TIMING, false)) {
        const char *rank = openmx_env_at(OPENMX_ENV_EXT_OMPI_COMM_WORLD_RANK);
        if (rank == nullptr) rank = openmx_env_at(OPENMX_ENV_EXT_PMI_RANK);
        std::printf("<openmx_gemmul8> general GEMMs, rank %s: real %lld + %lld calls, %.6f + %.6f s, %.3f + %.3f GFlop; "
                    "complex %lld + %lld calls, %.6f + %.6f s, %.3f + %.3f GFlop (GEMMul8 + cuBLAS); "
                    "moduli %u / %u, scaling %s / %s (real / complex)\n",
                    rank != nullptr ? rank : "-", g.calls[0][0], g.calls[0][1], g.seconds[0][0], g.seconds[0][1],
                    1e-9 * g.flops[0][0], 1e-9 * g.flops[0][1], g.calls[1][0], g.calls[1][1], g.seconds[1][0],
                    g.seconds[1][1], 1e-9 * g.flops[1][0], 1e-9 * g.flops[1][1],
                    gemmul8_num_moduli(OPENMX_ENV_GEMMUL8_NUM_MOD_D, OPENMX_ENV_EXT_GEMMUL8_NUM_MOD_D),
                    gemmul8_num_moduli(OPENMX_ENV_GEMMUL8_NUM_MOD_Z, OPENMX_ENV_EXT_GEMMUL8_NUM_MOD_Z),
                    openmx_env_flag_at(OPENMX_ENV_GEMMUL8_FASTMODE_D, openmx_env_flag_at(OPENMX_ENV_EXT_GEMMUL8_FASTMODE_D, false)) ? "fast" : "accurate",
                    openmx_env_flag_at(OPENMX_ENV_GEMMUL8_FASTMODE_Z, openmx_env_flag_at(OPENMX_ENV_EXT_GEMMUL8_FASTMODE_Z, false)) ? "fast" : "accurate");
        std::fflush(stdout);
    }
    g_general_counters = GeneralCounters{};
}

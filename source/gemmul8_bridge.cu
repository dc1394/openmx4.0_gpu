#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuComplex.h>

#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <type_traits>
#include <unordered_map>

#include "gemmul8.hpp"

namespace {

constexpr unsigned kDefaultNumModuli = 15u;
constexpr unsigned kMaxNumModuli     = 20u;
constexpr unsigned kDefaultMinFreeAfterMiB = 1536u;
constexpr unsigned kDefaultMaxWorkspacePercent = 30u;
constexpr size_t   kMiB = 1024u * 1024u;
constexpr unsigned kDefaultMemorySavingMiB = 256u;
constexpr size_t   kMinMemorySavingBytes = size_t(256) * kMiB;

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

struct Workspace {
    void *ptr   = nullptr;
    size_t size = 0;
};

std::mutex g_workspace_mutex;
std::unordered_map<WorkspaceKey, Workspace, WorkspaceKeyHash> g_workspaces;

/* scf.gemmul8.enable from the input file; written once per input parse
   (Input_std.c, before any GEMM runs) via openmx_gemmul8SetEnabled().
   0 sends every call straight to plain cuBLAS FP64 GEMM, so the GEMMul8
   contribution can be isolated without touching the environment. */
int g_input_enabled = 1;

struct WorkspaceReport {
    size_t      required_bytes = 0;
    size_t      free_bytes     = 0;
    size_t      total_bytes    = 0;
    size_t      reserve_bytes  = 0;
    unsigned    max_workspace_percent = 0;
    unsigned    ranks_per_gpu = 1;
    const char *reason = "allocation failure";
};

unsigned env_u32(const char *name, unsigned fallback)
{
    const char *value = std::getenv(name);
    char       *end   = nullptr;

    if (value == nullptr || *value == '\0') {
        return fallback;
    }

    unsigned long parsed = std::strtoul(value, &end, 10);
    if (end == value || *end != '\0') {
        return fallback;
    }

    return static_cast<unsigned>(parsed);
}

bool env_bool(const char *name, bool fallback)
{
    const char *value = std::getenv(name);

    if (value == nullptr || *value == '\0') {
        return fallback;
    }

    return value[0] == '1';
}

bool verbose_logging_enabled()
{
    bool verbose = env_bool("OPENMX_GPU_VERBOSE", false);
    verbose = env_bool("OPENMX_GEMM_VERBOSE", verbose);
    verbose = env_bool("OPENMX_GEMMUL8_VERBOSE", verbose);
    return env_bool("GEMMUL8_VERBOSE", verbose);
}

unsigned env_percent(const char *openmx_env, const char *gemmul8_env, unsigned fallback)
{
    unsigned percent = env_u32(gemmul8_env, fallback);
    percent          = env_u32(openmx_env, percent);

    if (100u < percent) {
        percent = 100u;
    }

    return percent;
}

size_t env_mib(const char *openmx_env, const char *gemmul8_env, unsigned fallback)
{
    unsigned mib = env_u32(gemmul8_env, fallback);
    mib          = env_u32(openmx_env, mib);

    return static_cast<size_t>(mib) * kMiB;
}

bool gemmul8_disabled(const char *openmx_env, const char *gemmul8_env)
{
    bool disabled = env_bool("GEMMUL8_DISABLE", false);
    disabled      = env_bool("OPENMX_GEMMUL8_DISABLE", disabled);
    disabled      = env_bool(gemmul8_env, disabled);
    disabled      = env_bool(openmx_env, disabled);

    return disabled;
}

/* Ported from the AMD bridge: keep Ozaki-II enabled for large matrices by
   letting GEMMul8 block within a per-rank workspace cap.  Zero restores the
   uncapped policy.  Smaller nonzero caps are raised to a viable block size. */
size_t memory_saving_cap_bytes()
{
    static const size_t cap = [] {
        size_t bytes = env_mib("OPENMX_GEMMUL8_MAX_WORKSPACE_MB", "GEMMUL8_MAX_WORKSPACE_MB",
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
        unsigned local_size = env_u32("OPENMX_GEMMUL8_LOCAL_RANKS", 0u);
        if (local_size == 0u) local_size = env_u32("OMPI_COMM_WORLD_LOCAL_SIZE", 0u);
        if (local_size == 0u) local_size = env_u32("SLURM_NTASKS_PER_NODE", 0u);
        if (local_size == 0u) local_size = 1u;
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || count < 1) count = 1;
        const unsigned devices = static_cast<unsigned>(count);
        return local_size / devices + (local_size % devices != 0u);
    }();
    return ranks;
}

unsigned gemmul8_num_moduli(const char *openmx_env, const char *gemmul8_env)
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
    if (workspace.ptr == nullptr) {
        workspace.size = 0;
        return cudaSuccess;
    }

    cudaError_t status = cudaFree(workspace.ptr);
    if (status == cudaSuccess) {
        workspace.ptr  = nullptr;
        workspace.size = 0;
    }

    return status;
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
{
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
            env_mib("OPENMX_GEMMUL8_MIN_FREE_AFTER_MB", "GEMMUL8_MIN_FREE_AFTER_MB", kDefaultMinFreeAfterMiB);
        report->max_workspace_percent = env_percent("OPENMX_GEMMUL8_MAX_WORKSPACE_PERCENT",
                                                     "GEMMUL8_MAX_WORKSPACE_PERCENT",
                                                     kDefaultMaxWorkspacePercent);
        report->ranks_per_gpu = ranks_per_gpu;
    }

    std::lock_guard<std::mutex> lock(g_workspace_mutex);
    Workspace                  &workspace = g_workspaces[key];

    size_t free_bytes  = 0;
    size_t total_bytes = 0;
    cuda_status        = cudaMemGetInfo(&free_bytes, &total_bytes);
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

    if (workspace.size < required) {
        cuda_status = release_workspace(workspace);
        if (cuda_status != cudaSuccess) {
            return CUBLAS_STATUS_INTERNAL_ERROR;
        }

        cuda_status = cudaMemGetInfo(&free_bytes, &total_bytes);
        if (cuda_status == cudaSuccess && report != nullptr) {
            report->free_bytes  = free_bytes;
            report->total_bytes = total_bytes;
        }
    }

    if (cuda_status == cudaSuccess &&
        free_after_workspace_is_too_low(free_bytes, workspace.size, required,
                                        report != nullptr ? report->reserve_bytes : 0u)) {
        if (report != nullptr) {
            report->reason = "free memory reserve policy";
        }
        if (release_workspace(workspace) != cudaSuccess) {
            return CUBLAS_STATUS_INTERNAL_ERROR;
        }
        return CUBLAS_STATUS_ALLOC_FAILED;
    }

    if (workspace.size < required) {
        cuda_status = cudaMalloc(&workspace.ptr, required);
        if (cuda_status != cudaSuccess) {
            if (report != nullptr) {
                report->reason = "cudaMalloc failure";
            }
            return CUBLAS_STATUS_ALLOC_FAILED;
        }
        workspace.size = required;
    }

    *work = workspace.ptr;
    return CUBLAS_STATUS_SUCCESS;
}

template <bool is_complex>
void log_workspace_fallback_once(const WorkspaceReport &report)
{
    static bool warned = false;

    if (!verbose_logging_enabled()) {
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

    const unsigned num_moduli = gemmul8_num_moduli("OPENMX_GEMMUL8_NUM_MOD_D", "GEMMUL8_NUM_MOD_D");
    const bool     fastmode   = env_bool("OPENMX_GEMMUL8_FASTMODE_D", env_bool("GEMMUL8_FASTMODE_D", false));
    const cublasOperation_t gemmul8_transa = (transa == CUBLAS_OP_C) ? CUBLAS_OP_T : transa;
    const cublasOperation_t gemmul8_transb = (transb == CUBLAS_OP_C) ? CUBLAS_OP_T : transb;
    void          *work = nullptr;
    WorkspaceReport report;

    if (!g_input_enabled) {
        /* scf.gemmul8.enable off: the fallback is what the user asked for,
           so no warning (Input_std already reported it once) */
        return cublasDgemm(handle, gemmul8_transa, gemmul8_transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }

    if (gemmul8_disabled("OPENMX_GEMMUL8_DISABLE_D", "GEMMUL8_DISABLE_D")) {
        report.reason = "environment disable";
        log_workspace_fallback_once<false>(report);
        return cublasDgemm(handle, gemmul8_transa, gemmul8_transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }

    apply_memory_saving(handle);

    cublasStatus_t status =
        ensure_workspace<false>(handle, static_cast<size_t>(m), static_cast<size_t>(n), static_cast<size_t>(k),
                                num_moduli, fastmode, &work, &report);
    if (status == CUBLAS_STATUS_ALLOC_FAILED) {
        log_workspace_fallback_once<false>(report);
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

/* scf.gemmul8.enable from the input file (Input_std.c); default on */
extern "C" void openmx_gemmul8SetEnabled(int enabled)
{
    g_input_enabled = (enabled != 0);
}

extern "C" size_t openmx_gemmul8ZWorkspaceSize(int m, int n, int k)
{
    if (m <= 0 || n <= 0 || k <= 0 || !g_input_enabled ||
        gemmul8_disabled("OPENMX_GEMMUL8_DISABLE_Z", "GEMMUL8_DISABLE_Z")) {
        return 0;
    }

    const unsigned num_moduli = gemmul8_num_moduli("OPENMX_GEMMUL8_NUM_MOD_Z", "GEMMUL8_NUM_MOD_Z");
    const bool fastmode = env_bool("OPENMX_GEMMUL8_FASTMODE_Z", env_bool("GEMMUL8_FASTMODE_Z", false));

    return capped_workspace_size(gemmul8::workSize<true, gemmul8::Backend::INT8>(
        static_cast<size_t>(m), static_cast<size_t>(n), static_cast<size_t>(k), num_moduli,
        false, false, nullptr, nullptr, fastmode));
}

extern "C" size_t openmx_gemmul8DWorkspaceSize(int m, int n, int k)
{
    if (m <= 0 || n <= 0 || k <= 0 || !g_input_enabled ||
        gemmul8_disabled("OPENMX_GEMMUL8_DISABLE_D", "GEMMUL8_DISABLE_D")) {
        return 0;
    }

    const unsigned num_moduli = gemmul8_num_moduli("OPENMX_GEMMUL8_NUM_MOD_D", "GEMMUL8_NUM_MOD_D");
    const bool fastmode = env_bool("OPENMX_GEMMUL8_FASTMODE_D", env_bool("GEMMUL8_FASTMODE_D", false));

    return capped_workspace_size(gemmul8::workSize<false, gemmul8::Backend::INT8>(
        static_cast<size_t>(m), static_cast<size_t>(n), static_cast<size_t>(k), num_moduli,
        false, false, nullptr, nullptr, fastmode));
}

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

    const unsigned num_moduli = gemmul8_num_moduli("OPENMX_GEMMUL8_NUM_MOD_Z", "GEMMUL8_NUM_MOD_Z");
    const bool     fastmode   = env_bool("OPENMX_GEMMUL8_FASTMODE_Z", env_bool("GEMMUL8_FASTMODE_Z", false));
    void          *work = nullptr;
    WorkspaceReport report;

    if (!g_input_enabled) {
        /* scf.gemmul8.enable off: the fallback is what the user asked for,
           so no warning (Input_std already reported it once) */
        return cublasZgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }

    if (gemmul8_disabled("OPENMX_GEMMUL8_DISABLE_Z", "GEMMUL8_DISABLE_Z")) {
        report.reason = "environment disable";
        log_workspace_fallback_once<true>(report);
        return cublasZgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }

    apply_memory_saving(handle);

    cublasStatus_t status =
        ensure_workspace<true>(handle, static_cast<size_t>(m), static_cast<size_t>(n), static_cast<size_t>(k),
                               num_moduli, fastmode, &work, &report);
    if (status == CUBLAS_STATUS_ALLOC_FAILED) {
        log_workspace_fallback_once<true>(report);
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
 *     default  same path and settings as openmx_gemmul8{D,Z}gemm
 *     fp64     plain cuBLAS FP64
 *     gemmul8  GEMMul8 with the settings below
 *   OPENMX_GEMMUL8_FORWARD_NUM_MOD    moduli count (2..20, default 15)
 *   OPENMX_GEMMUL8_FORWARD_FASTMODE   1: fast scaling (default 0)
 *   OPENMX_GEMMUL8_FORWARD_REUSE      1: retain the prepared X (default 0)
 *   OPENMX_GEMMUL8_FORWARD_UNBLOCKED  1: full workspace instead of the
 *                                     capped, blocked one (implied by reuse:
 *                                     GEMMul8 disables skip_scal in its
 *                                     memory-saving mode)
 *   OPENMX_GEMMUL8_FORWARD_TIMING     1: synchronize around each call and
 *                                     report the accumulated wall time
 * A retained form or a full workspace that does not fit falls back to the
 * next path of the same precision (re-preparing every call, then blocked),
 * and is counted; capacity never lowers the precision.
 * ---------------------------------------------------------------------- */
namespace {

enum : int { kForwardDefault = 0, kForwardFp64 = 1, kForwardGemmul8 = 2 };

struct ForwardStage {
    int      mode       = kForwardDefault;
    unsigned num_moduli = kDefaultNumModuli;
    bool     fastmode   = false;
    bool     reuse      = false;
    bool     unblocked  = false;
};

struct PreparedKey {
    int device;
    int id;
    int x_is_left;
    int is_complex;

    bool operator==(const PreparedKey &other) const
    {
        return device == other.device && id == other.id && x_is_left == other.x_is_left &&
               is_complex == other.is_complex;
    }
};

struct PreparedKeyHash {
    std::size_t operator()(const PreparedKey &key) const
    {
        return (static_cast<std::size_t>(key.device) << 40) ^ (static_cast<std::size_t>(key.id) << 8) ^
               (static_cast<std::size_t>(key.x_is_left) << 1) ^ static_cast<std::size_t>(key.is_complex);
    }
};

struct Prepared {
    void              *ptr   = nullptr;
    size_t             size  = 0;
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
        const char *mode = std::getenv("OPENMX_GEMMUL8_FORWARD");
        ForwardStage stage;

        if (mode != nullptr && std::strcmp(mode, "fp64") == 0) stage.mode = kForwardFp64;
        if (mode != nullptr && std::strcmp(mode, "gemmul8") == 0) stage.mode = kForwardGemmul8;
        stage.num_moduli = gemmul8_num_moduli("OPENMX_GEMMUL8_FORWARD_NUM_MOD", "OPENMX_GEMMUL8_FORWARD_NUM_MOD");
        stage.fastmode   = env_bool("OPENMX_GEMMUL8_FORWARD_FASTMODE", false);
        stage.reuse      = env_bool("OPENMX_GEMMUL8_FORWARD_REUSE", false);
        stage.unblocked  = stage.reuse || env_bool("OPENMX_GEMMUL8_FORWARD_UNBLOCKED", false);
        g_forward_stage     = stage;
        g_forward_stage_set = true;
    }
    return g_forward_stage;
}

/* Scratch of an unblocked call: the shared per-(device, stream) workspace
   grown to `required` bytes.  Only the free-memory reserve applies; the
   rank-share policy of ensure_workspace() assumes every rank holds such a
   workspace, whereas only the dense-owning rank reaches these products. */
cublasStatus_t ensure_scratch(cublasHandle_t handle, size_t required, void **work)
{
    cudaStream_t stream = nullptr;
    int          device = -1;

    cublasStatus_t status = cublasGetStream(handle, &stream);
    if (status != CUBLAS_STATUS_SUCCESS) return status;
    if (cudaGetDevice(&device) != cudaSuccess) return CUBLAS_STATUS_INTERNAL_ERROR;

    const size_t reserve =
        env_mib("OPENMX_GEMMUL8_MIN_FREE_AFTER_MB", "GEMMUL8_MIN_FREE_AFTER_MB", kDefaultMinFreeAfterMiB);
    std::lock_guard<std::mutex> lock(g_workspace_mutex);
    Workspace &workspace = g_workspaces[WorkspaceKey{device, stream}];

    if (workspace.size < required) {
        size_t free_bytes = 0, total_bytes = 0;

        if (release_workspace(workspace) != cudaSuccess) return CUBLAS_STATUS_INTERNAL_ERROR;
        if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess && free_bytes < required + reserve)
            return CUBLAS_STATUS_ALLOC_FAILED;
        if (cudaMalloc(&workspace.ptr, required) != cudaSuccess) {
            workspace.ptr = nullptr;
            (void)cudaGetLastError();
            return CUBLAS_STATUS_ALLOC_FAILED;
        }
        workspace.size = required;
    }
    *work = workspace.ptr;
    return CUBLAS_STATUS_SUCCESS;
}

/* The retained form of one X, at least `bytes` large; nullptr when the
   device cannot hold it.  Growing it discards the old contents. */
Prepared *ensure_prepared(const PreparedKey &key, size_t bytes)
{
    const size_t reserve =
        env_mib("OPENMX_GEMMUL8_MIN_FREE_AFTER_MB", "GEMMUL8_MIN_FREE_AFTER_MB", kDefaultMinFreeAfterMiB);
    std::lock_guard<std::mutex> lock(g_workspace_mutex);
    Prepared &entry = g_prepared[key];

    if (entry.size < bytes) {
        size_t free_bytes = 0, total_bytes = 0;

        if (entry.ptr != nullptr) (void)cudaFree(entry.ptr);
        entry = Prepared{};
        if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess && free_bytes < bytes + reserve) return nullptr;
        if (cudaMalloc(&entry.ptr, bytes) != cudaSuccess) {
            entry.ptr = nullptr;
            (void)cudaGetLastError();
            return nullptr;
        }
        entry.size = bytes;
    }
    return &entry;
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
                            unsigned long long x_version)
{
    constexpr bool          is_complex = !std::is_same_v<T, double>;
    const T                 one = forward_scalar<T>(1.0), zero = forward_scalar<T>(0.0);
    const cublasOperation_t transa = x_is_left ? op_x : CUBLAS_OP_N;
    const cublasOperation_t transb = x_is_left ? CUBLAS_OP_N : op_x;
    const T *const          A = x_is_left ? X : V;
    const T *const          B = x_is_left ? V : X;
    const int               lda = x_is_left ? ldx : ldv, ldb = x_is_left ? ldv : ldx;
    const ForwardStage      stage = forward_stage();

    if (m <= 0 || n <= 0 || k <= 0) return CUBLAS_STATUS_SUCCESS;

    static const bool timing = env_bool("OPENMX_GEMMUL8_FORWARD_TIMING", false);
    cudaStream_t      stream = nullptr;
    cublasStatus_t    status = cublasGetStream(handle, &stream);
    if (status != CUBLAS_STATUS_SUCCESS) return status;
    if (timing && cudaStreamSynchronize(stream) != cudaSuccess) return CUBLAS_STATUS_INTERNAL_ERROR;
    const auto start = std::chrono::steady_clock::now();

    if (stage.mode == kForwardDefault) {
        if constexpr (is_complex) {
            status = openmx_gemmul8Zgemm(handle, transa, transb, m, n, k, &one, A, lda, B, ldb, &zero, C, ldc);
        } else {
            status = openmx_gemmul8Dgemm(handle, transa, transb, m, n, k, &one, A, lda, B, ldb, &zero, C, ldc);
        }
    } else if (stage.mode == kForwardFp64 || !g_input_enabled) {
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
                sm, sn, sk, stage.num_moduli, stage.reuse && x_is_left, stage.reuse && !x_is_left, &size_a, &size_b,
                stage.fastmode);

            if (stage.reuse && cudaGetDevice(&device) == cudaSuccess) {
                keep  = x_is_left ? size_a : size_b;
                entry = ensure_prepared(PreparedKey{device, x_id, x_is_left ? 1 : 0, is_complex ? 1 : 0}, keep);
            }
            if (stage.reuse && entry == nullptr) {
                /* no room for the retained form: prepare X on every call */
                keep  = 0;
                total = gemmul8::workSize<is_complex, gemmul8::Backend::INT8>(sm, sn, sk, stage.num_moduli, false,
                                                                              false, nullptr, nullptr, stage.fastmode);
                std::lock_guard<std::mutex> lock(g_workspace_mutex);
                ++g_forward_counters.capacity_fallbacks;
            }

            status = ensure_scratch(handle, total - keep, &work);
            if (status == CUBLAS_STATUS_SUCCESS) {
                const bool reuse = entry != nullptr;
                const bool skip  = reuse && entry->valid && entry->version == x_version && entry->m == m &&
                                   entry->n == n && entry->k == k && entry->op == static_cast<int>(op_x) &&
                                   entry->num_moduli == stage.num_moduli && entry->fastmode == stage.fastmode;

                gemmul8::set_memory_saving(handle, false);
                (void)gemmul8::gemm<T, gemmul8::Backend::INT8>(
                    handle, gemmul8_transa, gemmul8_transb, sm, sn, sk, &one, A, static_cast<size_t>(lda), B,
                    static_cast<size_t>(ldb), &zero, C, static_cast<size_t>(ldc), static_cast<int>(stage.num_moduli),
                    stage.fastmode, work, (reuse && x_is_left) ? entry->ptr : nullptr,
                    (reuse && !x_is_left) ? entry->ptr : nullptr, reuse && x_is_left, reuse && !x_is_left,
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
                    entry->num_moduli = stage.num_moduli;
                    entry->fastmode   = stage.fastmode;
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

        if (!done) {
            WorkspaceReport report;

            apply_memory_saving(handle);
            status = ensure_workspace<is_complex>(handle, sm, sn, sk, stage.num_moduli, stage.fastmode, &work, &report);
            if (status == CUBLAS_STATUS_ALLOC_FAILED) {
                log_workspace_fallback_once<is_complex>(report);
                status = native_gemm<T>(handle, transa, transb, m, n, k, &one, A, lda, B, ldb, &zero, C, ldc);
            } else if (status == CUBLAS_STATUS_SUCCESS) {
                (void)gemmul8::gemm<T, gemmul8::Backend::INT8>(
                    handle, gemmul8_transa, gemmul8_transb, sm, sn, sk, &one, A, static_cast<size_t>(lda), B,
                    static_cast<size_t>(ldb), &zero, C, static_cast<size_t>(ldc), static_cast<int>(stage.num_moduli),
                    stage.fastmode, work);
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

} // namespace

extern "C" cublasStatus_t openmx_gemmul8DgemmFixed(cublasHandle_t handle, int x_is_left, cublasOperation_t op_x,
                                                    int m, int n, int k, const double *X, int ldx, const double *V,
                                                    int ldv, double *C, int ldc, int x_id,
                                                    unsigned long long x_version)
{
    return forward_gemm<double>(handle, x_is_left != 0, op_x, m, n, k, X, ldx, V, ldv, C, ldc, x_id, x_version);
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
    for (const auto &item : g_prepared) bytes += item.second.size;
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
        bytes += item.second.size;
        if (item.second.ptr != nullptr) (void)cudaFree(item.second.ptr);
    }
    g_prepared.clear();

    if (c.calls[0] + c.calls[1] > 0 &&
        (verbose_logging_enabled() || env_bool("OPENMX_GEMMUL8_FORWARD_TIMING", false))) {
        const ForwardStage &s = g_forward_stage;
        std::printf("<openmx_gemmul8> forward transform: mode=%s moduli=%u scaling=%s reuse=%d unblocked=%d; "
                    "calls %lld + %lld, prepared %lld, reused %lld, capacity fallbacks %lld, retained %.1f MiB, "
                    "wall %.6f + %.6f s\n",
                    s.mode == kForwardFp64 ? "fp64" : (s.mode == kForwardGemmul8 ? "gemmul8" : "default"),
                    s.num_moduli, s.fastmode ? "fast" : "accurate", int(s.reuse), int(s.unblocked), c.calls[0],
                    c.calls[1], c.prepared, c.reused, c.capacity_fallbacks, double(bytes) / double(kMiB),
                    c.seconds[0], c.seconds[1]);
        std::fflush(stdout);
    }
    g_forward_counters = ForwardCounters{};
}

/* ------------------------------------------------------------------------
 * Precision controller of the forward transform during an SCF.
 *
 * A policy lists GEMMul8 stages (moduli count, scaling mode) of increasing
 * precision; the stage after the last one is plain cuBLAS FP64.  Every rank
 * runs the same state machine on the same global quantities, so the stage
 * needs no broadcast; the rank that performs the dense solve additionally
 * reports the error indicator of a trial, which the caller reduces over MPI.
 *
 *   trial        BeginTrial sets the stage of the two forward GEMMs.
 *   rejection    An indicator above the stage's tolerance, or a failure of
 *                the solve, rejects the trial: Reject moves to the next
 *                stage and the caller repeats the solve from the same H.
 *   promotion    AfterMixing moves up when the SCF residual has stayed below
 *                the stage's threshold for `window` iterations, or when the
 *                residual has not improved for `stall` iterations.
 *   FP64 tail    StopCheck never lets a GEMMul8 stage end the SCF: when the
 *                usual stop condition holds (or the iteration budget is
 *                spent) it switches to the FP64 stage, and the SCF stops
 *                only after the stop condition has held `final_window`
 *                times in a row for differences between FP64 iterations.
 * The controller never returns to a lower stage.
 * ---------------------------------------------------------------------- */
namespace {

constexpr int kMaxAdaptiveStages = 8;

struct AdaptiveStage {
    unsigned num_moduli    = kDefaultNumModuli;
    bool     fastmode      = false;
    double   promote       = 0.0; /* SCF residual below which the next stage starts */
    double   eta_tolerance = 0.0; /* largest accepted error indicator; 0: none */
};

struct AdaptivePolicy {
    bool          enabled = false;
    int           nstage  = 0; /* GEMMul8 stages; index nstage is FP64 */
    AdaptiveStage stage[kMaxAdaptiveStages];
    bool          reuse          = true;
    bool          unblocked      = true;
    int           window         = 2;
    int           stall          = 0; /* 0: no stagnation rule */
    int           budget         = 0; /* 0: no iteration budget */
    int           final_window   = 2;
    bool          clear_history  = true; /* restart the mixing history at the FP64 stage */
    int           probe_columns  = 8; /* 0: no error indicator */
    int           probe_interval = 5; /* 0: only at stage entry and on request */
    double        probe_floor    = 1.0e-8; /* Hartree; lower bound of the indicator's denominator */
};

struct AdaptiveState {
    int    stage               = 0;
    int    iterations_in_stage = 0; /* accepted iterations */
    int    iterations_gemmul8  = 0;
    int    below               = 0;
    int    since_improvement   = 0;
    double best_residual       = -1.0;
    int    fp64_iterations     = 0;
    int    final_ok            = 0;
    bool   probe_requested     = false;
    bool   probe_this_trial    = false;
    bool   history_reset       = false;
    bool   rejected_local      = false;
    double eta_local           = -1.0;
    /* statistics of this SCF */
    long long trials = 0, rejections = 0, probes = 0;
    int       switch_iteration = 0; /* accepted iterations before the FP64 stage */
};

AdaptivePolicy g_adaptive_policy;
AdaptiveState  g_adaptive;

void adaptive_enter_stage(int stage)
{
    g_adaptive.stage               = stage;
    g_adaptive.iterations_in_stage = 0;
    g_adaptive.below               = 0;
    g_adaptive.since_improvement   = 0;
    g_adaptive.best_residual       = -1.0;
    g_adaptive.probe_requested     = true;
    if (stage >= g_adaptive_policy.nstage) {
        g_adaptive.fp64_iterations  = 0;
        g_adaptive.final_ok         = 0;
        g_adaptive.history_reset    = g_adaptive_policy.clear_history;
        g_adaptive.switch_iteration = g_adaptive.iterations_gemmul8;
    }
}

} // namespace

/* One line per stage: moduli count, scaling mode (0 accurate, 1 fast), the
   SCF residual below which the next stage starts, and the largest accepted
   error indicator (0: none).  nstage 0 disables the controller. */
extern "C" void openmx_gemmul8AdaptiveConfigure(int nstage, const int *moduli, const int *fastmode,
                                                 const double *promote, const double *eta_tolerance, int reuse,
                                                 int unblocked, int window, int stall, int budget, int final_window,
                                                 int clear_history, int probe_columns, int probe_interval,
                                                 double probe_floor)
{
    AdaptivePolicy policy;

    if (nstage > kMaxAdaptiveStages) nstage = kMaxAdaptiveStages;
    policy.enabled = nstage > 0;
    policy.nstage  = nstage > 0 ? nstage : 0;
    for (int i = 0; i < policy.nstage; ++i) {
        if (2 <= moduli[i] && moduli[i] <= static_cast<int>(kMaxNumModuli))
            policy.stage[i].num_moduli = static_cast<unsigned>(moduli[i]);
        policy.stage[i].fastmode      = fastmode[i] != 0;
        policy.stage[i].promote       = promote[i];
        policy.stage[i].eta_tolerance = eta_tolerance[i];
    }
    policy.reuse          = reuse != 0;
    policy.unblocked      = policy.reuse || unblocked != 0;
    policy.window         = window > 0 ? window : 1;
    policy.stall          = stall > 0 ? stall : 0;
    policy.budget         = budget > 0 ? budget : 0;
    policy.final_window   = final_window > 0 ? final_window : 1;
    policy.clear_history  = clear_history != 0;
    policy.probe_columns  = probe_columns > 0 ? probe_columns : 0;
    policy.probe_interval = probe_interval > 0 ? probe_interval : 0;
    if (probe_floor > 0.0) policy.probe_floor = probe_floor;
    g_adaptive_policy     = policy;
    g_adaptive            = AdaptiveState{};
}

extern "C" int openmx_gemmul8AdaptiveEnabled(void) { return g_adaptive_policy.enabled ? 1 : 0; }

/* start of an SCF */
extern "C" void openmx_gemmul8AdaptiveStart(void)
{
    if (!g_adaptive_policy.enabled) return;
    g_adaptive = AdaptiveState{};
    adaptive_enter_stage(0);
}

/* Sets the stage of the next solve and returns its index (the number of
   GEMMul8 stages for FP64). */
extern "C" int openmx_gemmul8AdaptiveBeginTrial(void)
{
    const AdaptivePolicy &p = g_adaptive_policy;
    AdaptiveState        &s = g_adaptive;

    if (!p.enabled) return -1;
    s.rejected_local = false;
    s.eta_local      = -1.0;
    ++s.trials;
    if (s.stage >= p.nstage) {
        s.probe_this_trial = false;
        openmx_gemmul8SetForwardStage(kForwardFp64, 0, 0, 0, 0);
    } else {
        const AdaptiveStage &stage = p.stage[s.stage];
        s.probe_this_trial = p.probe_columns > 0 &&
                             (s.probe_requested ||
                              (p.probe_interval > 0 && s.iterations_in_stage % p.probe_interval == 0));
        openmx_gemmul8SetForwardStage(kForwardGemmul8, static_cast<int>(stage.num_moduli), stage.fastmode ? 1 : 0,
                                      p.reuse ? 1 : 0, p.unblocked ? 1 : 0);
    }
    return s.stage;
}

/* columns of the error indicator the solve of this trial must evaluate (0: none) */
extern "C" int openmx_gemmul8AdaptiveProbeColumns(void)
{
    return (g_adaptive_policy.enabled && g_adaptive.probe_this_trial) ? g_adaptive_policy.probe_columns : 0;
}

extern "C" double openmx_gemmul8AdaptiveProbeFloor(void) { return g_adaptive_policy.probe_floor; }

/* The rank performing the dense solve reports the error indicator of a
   forward transform (eta < 0: not evaluated) and whether the solve failed
   (non-finite eigenvalues).  Several reports per trial keep the worst. */
extern "C" void openmx_gemmul8AdaptiveReport(double eta, int failed)
{
    const AdaptivePolicy &p = g_adaptive_policy;
    AdaptiveState        &s = g_adaptive;

    if (!p.enabled || s.stage >= p.nstage) return;
    if (eta >= 0.0) {
        const double tolerance = p.stage[s.stage].eta_tolerance;
        ++s.probes;
        if (eta > s.eta_local) s.eta_local = eta;
        if (!(eta == eta) || (tolerance > 0.0 && eta > tolerance)) s.rejected_local = true;
    }
    if (failed != 0) s.rejected_local = true;
}

/* this rank's verdict on the trial, to be reduced over all ranks (maximum) */
extern "C" void openmx_gemmul8AdaptiveTrialStatus(int *rejected, double *eta)
{
    *rejected = g_adaptive.rejected_local ? 1 : 0;
    *eta      = g_adaptive.eta_local;
}

/* every rank, after a rejected trial: the same H is solved again one stage up */
extern "C" void openmx_gemmul8AdaptiveReject(void)
{
    if (!g_adaptive_policy.enabled || g_adaptive.stage >= g_adaptive_policy.nstage) return;
    ++g_adaptive.rejections;
    adaptive_enter_stage(g_adaptive.stage + 1);
}

/* Every rank, once per accepted iteration, with the usual stop condition of
   the SCF: returns 1 when the SCF may stop. */
extern "C" int openmx_gemmul8AdaptiveStopCheck(int stop_condition)
{
    const AdaptivePolicy &p = g_adaptive_policy;
    AdaptiveState        &s = g_adaptive;

    if (!p.enabled) return stop_condition;
    ++s.iterations_in_stage;
    s.probe_requested = false;
    if (s.stage < p.nstage) {
        ++s.iterations_gemmul8;
        if (stop_condition != 0 || (p.budget > 0 && s.iterations_gemmul8 >= p.budget)) adaptive_enter_stage(p.nstage);
        return 0;
    }
    ++s.fp64_iterations;
    /* the first FP64 iteration compares its energy with a GEMMul8 one */
    if (stop_condition != 0 && s.fp64_iterations >= 2) ++s.final_ok;
    else s.final_ok = 0;
    return s.final_ok >= p.final_window ? 1 : 0;
}

/* returns 1 once after the switch to the FP64 stage: the caller restarts the
   mixing history, which was built from GEMMul8 iterations */
extern "C" int openmx_gemmul8AdaptiveTakeHistoryReset(void)
{
    const bool reset = g_adaptive_policy.enabled && g_adaptive.history_reset;
    g_adaptive.history_reset = false;
    return reset ? 1 : 0;
}

/* Every rank, after the mixing of an accepted iteration that did not end
   the SCF, with the SCF residual: returns 1 when the stage changed. */
extern "C" int openmx_gemmul8AdaptiveAfterMixing(double residual)
{
    const AdaptivePolicy &p = g_adaptive_policy;
    AdaptiveState        &s = g_adaptive;

    if (!p.enabled || s.stage >= p.nstage) return 0;

    if (s.best_residual < 0.0 || residual < 0.7 * s.best_residual) {
        s.best_residual     = residual;
        s.since_improvement = 0;
    } else {
        ++s.since_improvement;
    }
    if (residual < p.stage[s.stage].promote) ++s.below;
    else s.below = 0;

    const bool promote = s.stage + 1 < p.nstage && s.below >= p.window;
    const bool stalled = p.stall > 0 && s.since_improvement >= p.stall;
    if (promote || stalled) {
        adaptive_enter_stage(s.stage + 1);
        return 1;
    }
    /* halfway to the stagnation rule: look at the error indicator */
    if (p.stall > 0 && 2 * s.since_improvement >= p.stall) s.probe_requested = true;
    return 0;
}

/* description of the current stage and the statistics of this SCF:
   counters[0..4] = stage index, trials, rejections, indicator evaluations,
   accepted GEMMul8 iterations before the FP64 stage (or so far) */
extern "C" void openmx_gemmul8AdaptiveDescribe(char *text, int size, long long counters[5])
{
    const AdaptivePolicy &p = g_adaptive_policy;
    const AdaptiveState  &s = g_adaptive;

    if (!p.enabled) {
        std::snprintf(text, static_cast<size_t>(size), "off");
    } else if (s.stage >= p.nstage) {
        std::snprintf(text, static_cast<size_t>(size), "fp64");
    } else {
        std::snprintf(text, static_cast<size_t>(size), "moduli=%u %s%s", p.stage[s.stage].num_moduli,
                      p.stage[s.stage].fastmode ? "fast" : "accurate", p.reuse ? " reuse" : "");
    }
    counters[0] = s.stage;
    counters[1] = s.trials;
    counters[2] = s.rejections;
    counters[3] = s.probes;
    counters[4] = s.stage >= p.nstage ? s.switch_iteration : s.iterations_gemmul8;
}

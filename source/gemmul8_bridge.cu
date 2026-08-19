#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuComplex.h>

#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <unordered_map>

#include "gemmul8.hpp"

namespace {

constexpr unsigned kDefaultNumModuli = 15u;
constexpr unsigned kMaxNumModuli     = 20u;
constexpr unsigned kDefaultMinFreeAfterMiB = 1536u;
constexpr unsigned kDefaultMaxWorkspacePercent = 30u;
constexpr size_t   kMiB = 1024u * 1024u;

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

/* rank-local statistics for the run manifest (openmx_gemmul8GetStats).
   Slot layout (keep in sync with OpenMX_Manifest_Write):
     0 d_calls (GEMMul8 executed)   1 z_calls
     2 d_fallbacks (to cuBLAS)      3 z_fallbacks
     4 reason: workspace fraction   5 reason: free memory reserve
     6 reason: cudaMalloc/alloc     7 reason: environment disable
     8 d input-off calls            9 z input-off calls
    10 peak workspace bytes        11-15 reserved */
long long g_manifest_stats[16] = {0};
std::mutex g_manifest_stats_mutex;

void stats_add(int slot, long long v)
{
    std::lock_guard<std::mutex> lock(g_manifest_stats_mutex);
    g_manifest_stats[slot] += v;
}

void stats_max(int slot, long long v)
{
    std::lock_guard<std::mutex> lock(g_manifest_stats_mutex);
    if (g_manifest_stats[slot] < v) {
        g_manifest_stats[slot] = v;
    }
}

/* one of the five B70 reason strings -> stats slot; "allocation failure"
   and "cudaMalloc failure" share slot 6 (both are device allocation) */
int stats_reason_slot(const char *reason)
{
    if (reason == nullptr) return 6;
    if (std::strcmp(reason, "workspace fraction policy") == 0) return 4;
    if (std::strcmp(reason, "free memory reserve policy") == 0) return 5;
    if (std::strcmp(reason, "environment disable") == 0) return 7;
    return 6;
}

struct WorkspaceReport {
    size_t      required_bytes = 0;
    size_t      free_bytes     = 0;
    size_t      total_bytes    = 0;
    size_t      reserve_bytes  = 0;
    unsigned    max_workspace_percent = 0;
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

bool workspace_exceeds_fraction(size_t required, size_t total, unsigned max_percent)
{
    return total != 0 && max_percent != 0 && (total * static_cast<size_t>(max_percent)) / 100u < required;
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
cublasStatus_t ensure_workspace(cublasHandle_t handle, size_t m, size_t n, size_t k, unsigned num_moduli, void **work,
                                WorkspaceReport *report)
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

    const size_t required = gemmul8::workSize<is_complex, gemmul8::Backend::INT8>(m, n, k, num_moduli);
    WorkspaceKey key      = {device, stream};

    if (report != nullptr) {
        report->required_bytes = required;
        report->reserve_bytes =
            env_mib("OPENMX_GEMMUL8_MIN_FREE_AFTER_MB", "GEMMUL8_MIN_FREE_AFTER_MB", kDefaultMinFreeAfterMiB);
        report->max_workspace_percent = env_percent("OPENMX_GEMMUL8_MAX_WORKSPACE_PERCENT",
                                                     "GEMMUL8_MAX_WORKSPACE_PERCENT",
                                                     kDefaultMaxWorkspacePercent);
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
        workspace_exceeds_fraction(required, total_bytes, report != nullptr ? report->max_workspace_percent : 0u)) {
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

    std::lock_guard<std::mutex> lock(g_workspace_mutex);
    if (warned) {
        return;
    }

    fprintf(stderr,
            "openmx_gemmul8%sgemm: GEMMul8 workspace fallback by %s; "
            "need %.3f MiB, CUDA free %.3f MiB / total %.3f MiB, "
            "reserve %.3f MiB, max-workspace %u%%. Falling back to native cuBLAS.\n",
            is_complex ? "Z" : "D", report.reason, (double)report.required_bytes / (1024.0 * 1024.0),
            (double)report.free_bytes / (1024.0 * 1024.0), (double)report.total_bytes / (1024.0 * 1024.0),
            (double)report.reserve_bytes / (1024.0 * 1024.0), report.max_workspace_percent);
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
        stats_add(8, 1);
        return cublasDgemm(handle, gemmul8_transa, gemmul8_transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }

    if (gemmul8_disabled("OPENMX_GEMMUL8_DISABLE_D", "GEMMUL8_DISABLE_D")) {
        report.reason = "environment disable";
        log_workspace_fallback_once<false>(report);
        stats_add(2, 1);
        stats_add(7, 1);
        return cublasDgemm(handle, gemmul8_transa, gemmul8_transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }

    cublasStatus_t status =
        ensure_workspace<false>(handle, static_cast<size_t>(m), static_cast<size_t>(n), static_cast<size_t>(k),
                                num_moduli, &work, &report);
    if (status == CUBLAS_STATUS_ALLOC_FAILED) {
        log_workspace_fallback_once<false>(report);
        stats_add(2, 1);
        stats_add(stats_reason_slot(report.reason), 1);
        return cublasDgemm(handle, gemmul8_transa, gemmul8_transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }
    if (status != CUBLAS_STATUS_SUCCESS) {
        return status;
    }

    (void)gemmul8::gemm<double, gemmul8::Backend::INT8>(handle, gemmul8_transa, gemmul8_transb, static_cast<size_t>(m),
                                                        static_cast<size_t>(n), static_cast<size_t>(k), alpha, A,
                                                        static_cast<size_t>(lda), B, static_cast<size_t>(ldb), beta, C,
                                                        static_cast<size_t>(ldc), num_moduli, fastmode, work);

    stats_add(0, 1);
    stats_max(10, static_cast<long long>(report.required_bytes));
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

    return gemmul8::workSize<true, gemmul8::Backend::INT8>(
        static_cast<size_t>(m), static_cast<size_t>(n), static_cast<size_t>(k), num_moduli);
}

extern "C" size_t openmx_gemmul8DWorkspaceSize(int m, int n, int k)
{
    if (m <= 0 || n <= 0 || k <= 0 || !g_input_enabled ||
        gemmul8_disabled("OPENMX_GEMMUL8_DISABLE_D", "GEMMUL8_DISABLE_D")) {
        return 0;
    }

    const unsigned num_moduli = gemmul8_num_moduli("OPENMX_GEMMUL8_NUM_MOD_D", "GEMMUL8_NUM_MOD_D");

    return gemmul8::workSize<false, gemmul8::Backend::INT8>(
        static_cast<size_t>(m), static_cast<size_t>(n), static_cast<size_t>(k), num_moduli);
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
        stats_add(9, 1);
        return cublasZgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }

    if (gemmul8_disabled("OPENMX_GEMMUL8_DISABLE_Z", "GEMMUL8_DISABLE_Z")) {
        report.reason = "environment disable";
        log_workspace_fallback_once<true>(report);
        stats_add(3, 1);
        stats_add(7, 1);
        return cublasZgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }

    cublasStatus_t status =
        ensure_workspace<true>(handle, static_cast<size_t>(m), static_cast<size_t>(n), static_cast<size_t>(k),
                               num_moduli, &work, &report);
    if (status == CUBLAS_STATUS_ALLOC_FAILED) {
        log_workspace_fallback_once<true>(report);
        stats_add(3, 1);
        stats_add(stats_reason_slot(report.reason), 1);
        return cublasZgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }
    if (status != CUBLAS_STATUS_SUCCESS) {
        return status;
    }

    (void)gemmul8::gemm<cuDoubleComplex, gemmul8::Backend::INT8>(
        handle, transa, transb, static_cast<size_t>(m), static_cast<size_t>(n), static_cast<size_t>(k), alpha, A,
        static_cast<size_t>(lda), B, static_cast<size_t>(ldb), beta, C, static_cast<size_t>(ldc), num_moduli, fastmode,
        work);

    stats_add(1, 1);
    stats_max(10, static_cast<long long>(report.required_bytes));
    return CUBLAS_STATUS_SUCCESS;
}

/* rank-local counters for the run manifest; slot layout documented at
   g_manifest_stats.  Called once per run by OpenMX_Manifest_Write. */
extern "C" void openmx_gemmul8GetStats(long long out[16])
{
    std::lock_guard<std::mutex> lock(g_manifest_stats_mutex);
    for (int i = 0; i < 16; i++) {
        out[i] = g_manifest_stats[i];
    }
}

/* effective GEMMul8 configuration exactly as the wrappers resolve it */
extern "C" void openmx_gemmul8GetConfig(int *enabled, int *num_moduli_d, int *num_moduli_z,
                                        int *fastmode_d, int *fastmode_z,
                                        unsigned *max_workspace_percent,
                                        unsigned long long *min_free_after_mib)
{
    if (enabled != nullptr) *enabled = g_input_enabled;
    if (num_moduli_d != nullptr) {
        *num_moduli_d = (int)gemmul8_num_moduli("OPENMX_GEMMUL8_NUM_MOD_D", "GEMMUL8_NUM_MOD_D");
    }
    if (num_moduli_z != nullptr) {
        *num_moduli_z = (int)gemmul8_num_moduli("OPENMX_GEMMUL8_NUM_MOD_Z", "GEMMUL8_NUM_MOD_Z");
    }
    if (fastmode_d != nullptr) {
        *fastmode_d = env_bool("OPENMX_GEMMUL8_FASTMODE_D", env_bool("GEMMUL8_FASTMODE_D", false)) ? 1 : 0;
    }
    if (fastmode_z != nullptr) {
        *fastmode_z = env_bool("OPENMX_GEMMUL8_FASTMODE_Z", env_bool("GEMMUL8_FASTMODE_Z", false)) ? 1 : 0;
    }
    if (max_workspace_percent != nullptr) {
        *max_workspace_percent = env_percent("OPENMX_GEMMUL8_MAX_WORKSPACE_PERCENT",
                                             "GEMMUL8_MAX_WORKSPACE_PERCENT", kDefaultMaxWorkspacePercent);
    }
    if (min_free_after_mib != nullptr) {
        *min_free_after_mib = (unsigned long long)(env_mib("OPENMX_GEMMUL8_MIN_FREE_AFTER_MB",
                                                           "GEMMUL8_MIN_FREE_AFTER_MB",
                                                           kDefaultMinFreeAfterMiB) / kMiB);
    }
}

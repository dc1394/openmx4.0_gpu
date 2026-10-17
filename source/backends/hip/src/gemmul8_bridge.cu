#include "hipblas_compat.h"
#include "hip_runtime_compat.h"
#include "hip_complex_compat.h"

#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <atomic>
#include <chrono>
#include <mutex>
#include <unordered_map>

#include "gemmul8.hpp"
#include "openmx_mgpu_runtime.h"
#include "openmx_stdout_level.h"
#include "openmx_gpu_arch_policy.h"

namespace {

constexpr unsigned kDefaultNumModuli = 15u;
constexpr unsigned kMaxNumModuli     = 20u;
constexpr unsigned kDefaultMinFreeAfterMiB = 1536u;
constexpr unsigned kDefaultMaxWorkspacePercent = 30u;
constexpr size_t   kMiB = 1024u * 1024u;

struct WorkspaceKey {
    int          device;
    hipStream_t stream;

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
   0 sends every call straight to plain hipBLAS FP64 GEMM, so the GEMMul8
   contribution can be isolated without touching the environment.  -1 is
   automatic: MI300A uses native FP64, other GPUs keep GEMMul8.  Defer that
   device query until a GPU caller needs the policy; parsing every input of
   a CPU run or a -runtest suite only stores this setting. */
int g_input_enabled = -1;

/* calls of openmx_gemmul8{D,Z}gemm that ran in plain hipBLAS FP64 (switched
   off, disabled from the environment, or a workspace fallback): [real,
   complex].  The refined FP32 eigensolver compares them before and after its
   products: on GPUs with slow FP64 a refinement through plain FP64 products
   costs more than the FP64 eigensolver it replaces. */
std::atomic<long long> g_general_native[2];

/* OPENMX_GEMMUL8_TIMING=1: every general GEMM of the bridge (openmx_gemmul8
   {D,Z}gemm and their Solve and Unblocked variants) is timed between two
   synchronizations of its stream and counted by type (real, complex) and
   path (GEMMul8, plain hipBLAS); openmx_gemmul8ReportTiming() reports and
   resets the totals.  The synchronizations keep the products from
   overlapping other work, so the totals describe the products, not a run
   without the option.  Ported from the NVIDIA version. */
struct GeneralCounters {
    long long calls[2][2]   = {{0, 0}, {0, 0}}; /* [real, complex][GEMMul8, hipBLAS] */
    double    seconds[2][2] = {{0.0, 0.0}, {0.0, 0.0}};
    double    flops[2][2]   = {{0.0, 0.0}, {0.0, 0.0}};
};
GeneralCounters g_general_counters;

struct WorkspaceReport {
    size_t      required_bytes = 0;
    size_t      free_bytes     = 0;
    size_t      total_bytes    = 0;
    size_t      reserve_bytes  = 0;
    unsigned    max_workspace_percent = 0;
    unsigned    ranks_per_gpu  = 1;
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
    verbose = env_bool("GEMMUL8_VERBOSE", verbose);

    return verbose;
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
    const char *names[] = {"GEMMUL8_DISABLE", "OPENMX_GEMMUL8_DISABLE", gemmul8_env, openmx_env};
    bool disabled = false, overridden = false;

    /* Preserve the existing specificity/alias precedence.  Explicit zero
       also overrides the MI300A automatic policy; an input-file off still
       wins over every environment variable at the entry points. */
    for (const char *name : names) {
        const char *value = std::getenv(name);
        if (value != nullptr && value[0] != '\0') {
            disabled = env_bool(name, false);
            overridden = true;
        }
    }
    if (overridden) return disabled;
    return g_input_enabled < 0 && openmx_gpu_device_is_mi300a();
}

/* GEMMul8 >= 3.3.0 memory-saving mode: the per-GEMM workspace is capped and
   GEMMul8 runs the GEMM in blocks that fit the cap, instead of this bridge
   falling back to plain FP64 hipBLAS.  Default cap: 256 MiB per rank -- the
   blocked path matches the uncapped result bit-for-bit in accuracy class
   (verified against hipBLAS FP64 at n=4096) and keeps many-rank shared GPUs
   out of workspace-driven OOM.  OPENMX_GEMMUL8_MAX_WORKSPACE_MB overrides:
   0 disables the cap (pre-3.3.0 behavior, uncapped workspace with the
   fraction-based fallback below); values under 256 MiB are raised to 256,
   below the minimum viable block GEMMul8 asserts (and with NDEBUG would
   silently skip the multiplication). */
constexpr size_t kMinMemorySavingBytes = size_t(256) * kMiB;
constexpr unsigned kDefaultMemorySavingMiB = 256u;

size_t memory_saving_cap_bytes()
{
    static std::once_flag once;
    static size_t         cap = 0;

    std::call_once(once, [] {
        cap = env_mib("OPENMX_GEMMUL8_MAX_WORKSPACE_MB", "GEMMUL8_MAX_WORKSPACE_MB",
                      kDefaultMemorySavingMiB);
        if (cap != 0 && cap < kMinMemorySavingBytes) {
            std::fprintf(stderr,
                         "openmx_gemmul8: OPENMX_GEMMUL8_MAX_WORKSPACE_MB below the %zu MiB minimum; using %zu MiB.\n",
                         kMinMemorySavingBytes / kMiB, kMinMemorySavingBytes / kMiB);
            std::fflush(stderr);
            cap = kMinMemorySavingBytes;
        }
    });
    return cap;
}

/* apply the handle-local memory-saving configuration before a GEMM; cheap
   (a registry lookup inside GEMMul8), and idempotent */
void apply_memory_saving(hipblasHandle_t handle)
{
    const size_t cap = memory_saving_cap_bytes();

    if (cap != 0) {
        gemmul8::set_memory_saving(handle, true);
        gemmul8::set_max_worksize(handle, cap);
    }
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

/* Number of MPI ranks sharing this GPU. The workspace budget must be divided by
   this: each rank checks free memory independently, so 32 ranks can each "see"
   room for a 1-GiB workspace and collectively exhaust the device, after which
   hipBLASLt's lazy initialization inside GEMMul8 dies with out-of-memory. */
unsigned ranks_sharing_gpu()
{
    unsigned local_size = env_u32("OPENMX_GEMMUL8_LOCAL_RANKS", 0u);

    if (local_size == 0u) {
        local_size = env_u32("OMPI_COMM_WORLD_LOCAL_SIZE", 0u);
    }
    if (local_size == 0u) {
        local_size = env_u32("SLURM_NTASKS_PER_NODE", 0u);
    }
    if (local_size == 0u) {
        local_size = 1u;
    }

    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count < 1) {
        device_count = 1;
    }

    unsigned ranks = local_size / static_cast<unsigned>(device_count);
    return (ranks == 0u) ? 1u : ranks;
}

hipError_t release_workspace(Workspace &workspace)
{
    if (workspace.ptr == nullptr) {
        workspace.size = 0;
        return hipSuccess;
    }

    hipError_t status = hipFree(workspace.ptr);
    if (status == hipSuccess) {
        workspace.ptr  = nullptr;
        workspace.size = 0;
    }

    return status;
}

bool workspace_exceeds_fraction(size_t required, size_t total, unsigned max_percent, unsigned ranks_per_gpu)
{
    if (total == 0 || max_percent == 0) {
        return false;
    }
    if (ranks_per_gpu == 0u) {
        ranks_per_gpu = 1u;
    }
    return (total * static_cast<size_t>(max_percent)) / 100u / ranks_per_gpu < required;
}

bool free_after_workspace_is_too_low(size_t free_bytes, size_t workspace_size, size_t required, size_t reserve)
{
    if (required <= workspace_size) {
        return free_bytes < reserve;
    }

    const size_t extra_required = required - workspace_size;
    return free_bytes < extra_required || free_bytes - extra_required < reserve;
}

/* free device memory a product's workspace must leave: the general
   reserve of OPENMX_GEMMUL8_MIN_FREE_AFTER_MB (1536 MiB), for products any
   rank may run while the others run theirs */
size_t general_reserve()
{
    return env_mib("OPENMX_GEMMUL8_MIN_FREE_AFTER_MB", "GEMMUL8_MIN_FREE_AFTER_MB", kDefaultMinFreeAfterMiB);
}

/* the reserve of the products of a dense eigensolve (the *Unblocked entry
   points), which one rank runs while the others wait for its result:
   OPENMX_GEMMUL8_SOLVE_MIN_FREE_AFTER_MB, default 256 MiB, enough for the
   library handles' own scratch.  With the general reserve the solve's own
   buffers left too little room on a 16 GB GPU shared by 16 ranks, and every
   product fell back to plain FP64. */
size_t solve_reserve()
{
    return env_mib("OPENMX_GEMMUL8_SOLVE_MIN_FREE_AFTER_MB", "GEMMUL8_SOLVE_MIN_FREE_AFTER_MB", 256u);
}

template <bool is_complex>
hipblasStatus_t ensure_workspace(hipblasHandle_t handle, size_t m, size_t n, size_t k, unsigned num_moduli, void **work,
                                WorkspaceReport *report, size_t reserve)
{
    hipStream_t stream = nullptr;
    int          device = -1;

    hipblasStatus_t hipblas_status = hipblasGetStream(handle, &stream);
    if (hipblas_status != HIPBLAS_STATUS_SUCCESS) {
        return hipblas_status;
    }

    hipError_t hip_status = hipGetDevice(&device);
    if (hip_status != hipSuccess) {
        return HIPBLAS_STATUS_INTERNAL_ERROR;
    }

    size_t required = gemmul8::workSize<is_complex, gemmul8::Backend::INT8>(m, n, k, num_moduli);
    /* with the memory-saving cap the GEMM runs blocked inside cap bytes, so
       that is all the workspace it will touch */
    const size_t ms_cap = memory_saving_cap_bytes();
    if (ms_cap != 0 && ms_cap < required) {
        required = ms_cap;
    }
    WorkspaceKey key      = {device, stream};

    const unsigned ranks_per_gpu = ranks_sharing_gpu();

    if (report != nullptr) {
        report->required_bytes = required;
        report->reserve_bytes = reserve;
        report->max_workspace_percent = env_percent("OPENMX_GEMMUL8_MAX_WORKSPACE_PERCENT",
                                                     "GEMMUL8_MAX_WORKSPACE_PERCENT",
                                                     kDefaultMaxWorkspacePercent);
        report->ranks_per_gpu = ranks_per_gpu;
    }

    std::lock_guard<std::mutex> lock(g_workspace_mutex);
    Workspace                  &workspace = g_workspaces[key];

    size_t free_bytes  = 0;
    size_t total_bytes = 0;
    hip_status        = hipMemGetInfo(&free_bytes, &total_bytes);
    if (hip_status == hipSuccess && report != nullptr) {
        report->free_bytes  = free_bytes;
        report->total_bytes = total_bytes;
    }

    if (hip_status == hipSuccess &&
        workspace_exceeds_fraction(required, total_bytes, report != nullptr ? report->max_workspace_percent : 0u,
                                   ranks_per_gpu)) {
        if (report != nullptr) {
            report->reason = "workspace fraction policy";
        }
        if (release_workspace(workspace) != hipSuccess) {
            return HIPBLAS_STATUS_INTERNAL_ERROR;
        }
        return HIPBLAS_STATUS_ALLOC_FAILED;
    }

    if (workspace.size < required) {
        hip_status = release_workspace(workspace);
        if (hip_status != hipSuccess) {
            return HIPBLAS_STATUS_INTERNAL_ERROR;
        }

        hip_status = hipMemGetInfo(&free_bytes, &total_bytes);
        if (hip_status == hipSuccess && report != nullptr) {
            report->free_bytes  = free_bytes;
            report->total_bytes = total_bytes;
        }
    }

    if (hip_status == hipSuccess && free_after_workspace_is_too_low(free_bytes, workspace.size, required, reserve)) {
        if (report != nullptr) {
            report->reason = "free memory reserve policy";
        }
        if (release_workspace(workspace) != hipSuccess) {
            return HIPBLAS_STATUS_INTERNAL_ERROR;
        }
        return HIPBLAS_STATUS_ALLOC_FAILED;
    }

    if (workspace.size < required) {
        hip_status = hipMalloc(&workspace.ptr, required);
        if (hip_status != hipSuccess) {
            if (report != nullptr) {
                report->reason = "hipMalloc failure";
            }
            return HIPBLAS_STATUS_ALLOC_FAILED;
        }
        workspace.size = required;
    }

    *work = workspace.ptr;
    return HIPBLAS_STATUS_SUCCESS;
}

template <bool is_complex>
void log_workspace_fallback_once(const WorkspaceReport &report, const char *target)
{
    static bool warned = false;

    if (!verbose_logging_enabled() && !OPENMX_STDOUT_LEVEL(2)) {
        return;
    }

    std::lock_guard<std::mutex> lock(g_workspace_mutex);
    if (warned) {
        return;
    }

    if (std::strcmp(report.reason, "native FP64 policy") == 0) {
        fprintf(stderr,
                "openmx_gemmul8%sgemm: native FP64 selected by the device policy or environment; using %s.\n",
                is_complex ? "Z" : "D", target);
    }
    else {
        fprintf(stderr,
                "openmx_gemmul8%sgemm: GEMMul8 workspace fallback by %s; "
                "need %.3f MiB, HIP device free %.3f MiB / total %.3f MiB, "
                "reserve %.3f MiB, max-workspace %u%% shared by %u rank(s). Falling back to %s.\n",
                is_complex ? "Z" : "D", report.reason, (double)report.required_bytes / (1024.0 * 1024.0),
                (double)report.free_bytes / (1024.0 * 1024.0), (double)report.total_bytes / (1024.0 * 1024.0),
                (double)report.reserve_bytes / (1024.0 * 1024.0), report.max_workspace_percent,
                report.ranks_per_gpu, target);
    }
    fflush(stderr);
    warned = true;
}

/* Scratch of an unblocked call: the shared per-(device, stream) workspace
   grown to `required` bytes.  Only the free-memory reserve applies; the
   rank-share policy of ensure_workspace() assumes every rank holds such a
   workspace, whereas only the rank that owns a dense solve reaches these
   products. */
hipblasStatus_t ensure_scratch(hipblasHandle_t handle, size_t required, void **work, size_t reserve)
{
    hipStream_t stream = nullptr;
    int         device = -1;

    hipblasStatus_t status = hipblasGetStream(handle, &stream);
    if (status != HIPBLAS_STATUS_SUCCESS) return status;
    if (hipGetDevice(&device) != hipSuccess) return HIPBLAS_STATUS_INTERNAL_ERROR;

    std::lock_guard<std::mutex> lock(g_workspace_mutex);
    Workspace &workspace = g_workspaces[WorkspaceKey{device, stream}];

    if (workspace.size < required) {
        size_t free_bytes = 0, total_bytes = 0;

        if (release_workspace(workspace) != hipSuccess) return HIPBLAS_STATUS_INTERNAL_ERROR;
        if (hipMemGetInfo(&free_bytes, &total_bytes) == hipSuccess && free_bytes < required + reserve)
            return HIPBLAS_STATUS_ALLOC_FAILED;
        if (hipMalloc(&workspace.ptr, required) != hipSuccess) {
            workspace.ptr = nullptr;
            (void)hipGetLastError();
            return HIPBLAS_STATUS_ALLOC_FAILED;
        }
        workspace.size = required;
    }
    *work = workspace.ptr;
    return HIPBLAS_STATUS_SUCCESS;
}

/* OPENMX_GEMMUL8_TIMING, read once */
bool general_timing()
{
    static const bool timing = env_bool("OPENMX_GEMMUL8_TIMING", false);

    return timing;
}

/* body(&native) performs the product and sets native when it took plain
   hipBLAS; every such call is counted (g_general_native) whether timed or
   not.  flops counts the real floating-point operations of the product. */
template <class Body>
hipblasStatus_t timed_general_gemm(hipblasHandle_t handle, int is_complex, double flops, Body body)
{
    bool native = false;

    if (!general_timing()) {
        const hipblasStatus_t status = body(&native);
        if (native) g_general_native[is_complex].fetch_add(1, std::memory_order_relaxed);
        return status;
    }

    hipStream_t     stream = nullptr;
    hipblasStatus_t status = hipblasGetStream(handle, &stream);
    if (status != HIPBLAS_STATUS_SUCCESS) return status;
    if (hipStreamSynchronize(stream) != hipSuccess) return HIPBLAS_STATUS_INTERNAL_ERROR;
    const auto start = std::chrono::steady_clock::now();
    status = body(&native);
    if (native) g_general_native[is_complex].fetch_add(1, std::memory_order_relaxed);
    if (hipStreamSynchronize(stream) != hipSuccess) return HIPBLAS_STATUS_INTERNAL_ERROR;
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

/* *native is set when the product ran in plain hipBLAS FP64 (the callers
   count it, see timed_general_gemm) */
hipblasStatus_t dgemm_capped(hipblasHandle_t handle, hipblasOperation_t transa, hipblasOperation_t transb, int m, int n,
                             int k, const double *alpha, const double *A, int lda, const double *B, int ldb,
                             const double *beta, double *C, int ldc, size_t reserve, bool *native)
{
    if (m <= 0 || n <= 0 || k <= 0) {
        return HIPBLAS_STATUS_SUCCESS;
    }

    const unsigned num_moduli = gemmul8_num_moduli("OPENMX_GEMMUL8_NUM_MOD_D", "GEMMUL8_NUM_MOD_D");
    const bool     fastmode   = env_bool("OPENMX_GEMMUL8_FASTMODE_D", env_bool("GEMMUL8_FASTMODE_D", false));
    const hipblasOperation_t gemmul8_transa = (transa == HIPBLAS_OP_C) ? HIPBLAS_OP_T : transa;
    const hipblasOperation_t gemmul8_transb = (transb == HIPBLAS_OP_C) ? HIPBLAS_OP_T : transb;
    void          *work = nullptr;
    WorkspaceReport report;

    if (!g_input_enabled) {
        /* scf.gemmul8.enable off: the fallback is what the user asked for,
           so no warning (Input_std already reported it once) */
        *native = true;
        return hipblasDgemm(handle, gemmul8_transa, gemmul8_transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }

    if (gemmul8_disabled("OPENMX_GEMMUL8_DISABLE_D", "GEMMUL8_DISABLE_D")) {
        report.reason = "native FP64 policy";
        log_workspace_fallback_once<false>(report, "native hipBLAS");
        *native = true;
        return hipblasDgemm(handle, gemmul8_transa, gemmul8_transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }

    apply_memory_saving(handle);

    hipblasStatus_t status =
        ensure_workspace<false>(handle, static_cast<size_t>(m), static_cast<size_t>(n), static_cast<size_t>(k),
                                num_moduli, &work, &report, reserve);
    if (status == HIPBLAS_STATUS_ALLOC_FAILED) {
        log_workspace_fallback_once<false>(report, "native hipBLAS");
        *native = true;
        return hipblasDgemm(handle, gemmul8_transa, gemmul8_transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }
    if (status != HIPBLAS_STATUS_SUCCESS) {
        return status;
    }

    (void)gemmul8::gemm<double, gemmul8::Backend::INT8>(handle, gemmul8_transa, gemmul8_transb, static_cast<size_t>(m),
                                                        static_cast<size_t>(n), static_cast<size_t>(k), alpha, A,
                                                        static_cast<size_t>(lda), B, static_cast<size_t>(ldb), beta, C,
                                                        static_cast<size_t>(ldc), num_moduli, fastmode, work);

    return HIPBLAS_STATUS_SUCCESS;
}

} // namespace

extern "C" hipblasStatus_t openmx_gemmul8Dgemm(hipblasHandle_t handle, hipblasOperation_t transa,
                                               hipblasOperation_t transb, int m, int n, int k, const double *alpha,
                                               const double *A, int lda, const double *B, int ldb, const double *beta,
                                               double *C, int ldc)
{
    if (m <= 0 || n <= 0 || k <= 0) return HIPBLAS_STATUS_SUCCESS;
    hipblasStatus_t mgpu_status;
    if (openmx_mgpu_dispatch_dgemm && openmx_mgpu_dispatch_dgemm(handle,
            transa,transb,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc,&mgpu_status))
        return mgpu_status;
    return timed_general_gemm(handle, 0, 2.0 * m * n * k, [&](bool *native) {
        return dgemm_capped(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, general_reserve(),
                            native);
    });
}

/* scf.gemmul8.enable: -1 automatic, 0 off, 1 on; reset at every input. */
extern "C" void openmx_gemmul8SetEnabled(int enabled)
{
    g_input_enabled = (enabled < 0) ? -1 : (enabled != 0);
}

extern "C" size_t openmx_gemmul8ZWorkspaceSize(int m, int n, int k)
{
    if (m<=0 || n<=0 || k<=0 || !g_input_enabled ||
        gemmul8_disabled("OPENMX_GEMMUL8_DISABLE_Z","GEMMUL8_DISABLE_Z")) return 0;
    const unsigned num_moduli=gemmul8_num_moduli("OPENMX_GEMMUL8_NUM_MOD_Z","GEMMUL8_NUM_MOD_Z");
    size_t size = gemmul8::workSize<true,gemmul8::Backend::INT8>((size_t)m,(size_t)n,(size_t)k,num_moduli);
    const size_t ms_cap = memory_saving_cap_bytes();
    if (ms_cap != 0 && ms_cap < size) size = ms_cap;
    return size;
}

extern "C" size_t openmx_gemmul8DWorkspaceSize(int m, int n, int k)
{
    if (m<=0 || n<=0 || k<=0 || !g_input_enabled ||
        gemmul8_disabled("OPENMX_GEMMUL8_DISABLE_D","GEMMUL8_DISABLE_D")) return 0;
    const unsigned num_moduli=gemmul8_num_moduli("OPENMX_GEMMUL8_NUM_MOD_D","GEMMUL8_NUM_MOD_D");
    size_t size = gemmul8::workSize<false,gemmul8::Backend::INT8>((size_t)m,(size_t)n,(size_t)k,num_moduli);
    const size_t ms_cap = memory_saving_cap_bytes();
    if (ms_cap != 0 && ms_cap < size) size = ms_cap;
    return size;
}

extern "C" void openmx_gemmul8ReleaseWorkspaces(void)
{
    std::lock_guard<std::mutex> lock(g_workspace_mutex);
    hipError_t                 first_error = hipSuccess;

    for (auto it = g_workspaces.begin(); it != g_workspaces.end();) {
        hipError_t status = release_workspace(it->second);
        if (status == hipSuccess) {
            it = g_workspaces.erase(it);
        } else {
            if (first_error == hipSuccess) {
                first_error = status;
            }
            ++it;
        }
    }

    if (first_error != hipSuccess) {
        std::fprintf(stderr,
                     "openmx_gemmul8ReleaseWorkspaces: hipFree failed: %s\n",
                     hipGetErrorString(first_error));
        std::fflush(stderr);
    }
}

namespace {

/* *native as in dgemm_capped */
hipblasStatus_t zgemm_capped(hipblasHandle_t handle, hipblasOperation_t transa, hipblasOperation_t transb, int m, int n,
                             int k, const hipDoubleComplex *alpha, const hipDoubleComplex *A, int lda,
                             const hipDoubleComplex *B, int ldb, const hipDoubleComplex *beta, hipDoubleComplex *C,
                             int ldc, size_t reserve, bool *native)
{
    if (m <= 0 || n <= 0 || k <= 0) {
        return HIPBLAS_STATUS_SUCCESS;
    }

    const unsigned num_moduli = gemmul8_num_moduli("OPENMX_GEMMUL8_NUM_MOD_Z", "GEMMUL8_NUM_MOD_Z");
    const bool     fastmode   = env_bool("OPENMX_GEMMUL8_FASTMODE_Z", env_bool("GEMMUL8_FASTMODE_Z", false));
    void          *work = nullptr;
    WorkspaceReport report;

    if (!g_input_enabled) {
        /* scf.gemmul8.enable off: the fallback is what the user asked for,
           so no warning (Input_std already reported it once) */
        *native = true;
        return hipblasZgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }

    if (gemmul8_disabled("OPENMX_GEMMUL8_DISABLE_Z", "GEMMUL8_DISABLE_Z")) {
        report.reason = "native FP64 policy";
        log_workspace_fallback_once<true>(report, "native hipBLAS");
        *native = true;
        return hipblasZgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }

    apply_memory_saving(handle);

    hipblasStatus_t status =
        ensure_workspace<true>(handle, static_cast<size_t>(m), static_cast<size_t>(n), static_cast<size_t>(k),
                               num_moduli, &work, &report, reserve);
    if (status == HIPBLAS_STATUS_ALLOC_FAILED) {
        /* GPU memory is too tight for the GEMMul8 workspace: fall back to native
           hipBLAS (hipblasZgemm), which needs no extra workspace. If even that fails,
           its status propagates so the caller can fall back to CPU BLAS. */
        log_workspace_fallback_once<true>(report, "native hipBLAS");
        *native = true;
        return hipblasZgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }
    if (status != HIPBLAS_STATUS_SUCCESS) {
        return status;
    }

    (void)gemmul8::gemm<hipDoubleComplex, gemmul8::Backend::INT8>(
        handle, transa, transb, static_cast<size_t>(m), static_cast<size_t>(n), static_cast<size_t>(k), alpha, A,
        static_cast<size_t>(lda), B, static_cast<size_t>(ldb), beta, C, static_cast<size_t>(ldc), num_moduli, fastmode,
        work);

    return HIPBLAS_STATUS_SUCCESS;
}

} // namespace

extern "C" hipblasStatus_t openmx_gemmul8Zgemm(hipblasHandle_t handle, hipblasOperation_t transa,
                                               hipblasOperation_t transb, int m, int n, int k,
                                               const hipDoubleComplex *alpha, const hipDoubleComplex *A, int lda,
                                               const hipDoubleComplex *B, int ldb, const hipDoubleComplex *beta,
                                               hipDoubleComplex *C, int ldc)
{
    if (m <= 0 || n <= 0 || k <= 0) return HIPBLAS_STATUS_SUCCESS;
    hipblasStatus_t mgpu_status;
    if (openmx_mgpu_dispatch_zgemm && openmx_mgpu_dispatch_zgemm(handle,
            transa,transb,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc,&mgpu_status))
        return mgpu_status;
    return timed_general_gemm(handle, 1, 8.0 * m * n * k, [&](bool *native) {
        return zgemm_capped(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, general_reserve(),
                            native);
    });
}

/* 1 when openmx_gemmul8Zgemm takes the GEMMul8 path at all: explicit on or
   the architecture default, and not disabled from the environment (a call
   can still fall back for want of workspace). */
extern "C" int openmx_gemmul8ZgemmEnabled(void)
{
    return g_input_enabled && !gemmul8_disabled("OPENMX_GEMMUL8_DISABLE_Z", "GEMMUL8_DISABLE_Z") ? 1 : 0;
}

/* the same for openmx_gemmul8Dgemm */
extern "C" int openmx_gemmul8DgemmEnabled(void)
{
    return g_input_enabled && !gemmul8_disabled("OPENMX_GEMMUL8_DISABLE_D", "GEMMUL8_DISABLE_D") ? 1 : 0;
}

/* calls of openmx_gemmul8Dgemm (is_complex 0) or openmx_gemmul8Zgemm (1),
   unblocked ones included, that ran in plain hipBLAS FP64 so far */
extern "C" long long openmx_gemmul8NativeCalls(int is_complex)
{
    return g_general_native[is_complex ? 1 : 0].load(std::memory_order_relaxed);
}

/* Frees the scratch of unblocked products: every workspace larger than the
   cap of the blocked path.  The capped workspaces of the other products
   stay, and without a cap nothing is freed. */
extern "C" void openmx_gemmul8TrimWorkspaces(void)
{
    const size_t                cap = memory_saving_cap_bytes();
    std::lock_guard<std::mutex> lock(g_workspace_mutex);

    if (cap == 0) return;
    for (auto &item : g_workspaces) {
        if (cap < item.second.size && release_workspace(item.second) != hipSuccess) {
            std::fprintf(stderr, "openmx_gemmul8TrimWorkspaces: hipFree failed\n");
            std::fflush(stderr);
        }
    }
}

/* openmx_gemmul8Zgemm for the products of a dense eigensolve whose rank may
   share the device with other solving ranks (the band solvers' concurrent
   k-point turns): the workspace capped as in openmx_gemmul8Zgemm, the free
   memory reserve that of the solve (OPENMX_GEMMUL8_SOLVE_MIN_FREE_AFTER_MB) */
extern "C" hipblasStatus_t openmx_gemmul8ZgemmSolve(hipblasHandle_t handle, hipblasOperation_t transa,
                                                    hipblasOperation_t transb, int m, int n, int k,
                                                    const hipDoubleComplex *alpha, const hipDoubleComplex *A, int lda,
                                                    const hipDoubleComplex *B, int ldb, const hipDoubleComplex *beta,
                                                    hipDoubleComplex *C, int ldc)
{
    if (m <= 0 || n <= 0 || k <= 0) return HIPBLAS_STATUS_SUCCESS;
    hipblasStatus_t mgpu_status;
    if (openmx_mgpu_dispatch_zgemm && openmx_mgpu_dispatch_zgemm(handle,
            transa,transb,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc,&mgpu_status))
        return mgpu_status;
    return timed_general_gemm(handle, 1, 8.0 * m * n * k, [&](bool *native) {
        return zgemm_capped(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, solve_reserve(),
                            native);
    });
}

/* the same for real products */
extern "C" hipblasStatus_t openmx_gemmul8DgemmSolve(hipblasHandle_t handle, hipblasOperation_t transa,
                                                    hipblasOperation_t transb, int m, int n, int k, const double *alpha,
                                                    const double *A, int lda, const double *B, int ldb,
                                                    const double *beta, double *C, int ldc)
{
    if (m <= 0 || n <= 0 || k <= 0) return HIPBLAS_STATUS_SUCCESS;
    hipblasStatus_t mgpu_status;
    if (openmx_mgpu_dispatch_dgemm && openmx_mgpu_dispatch_dgemm(handle,
            transa,transb,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc,&mgpu_status))
        return mgpu_status;
    return timed_general_gemm(handle, 0, 2.0 * m * n * k, [&](bool *native) {
        return dgemm_capped(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, solve_reserve(),
                            native);
    });
}

/* openmx_gemmul8Zgemm for the products of a dense eigensolve, with the full
   GEMMul8 workspace when the device has room for it (free memory of at
   least the workspace plus the solve reserve, OPENMX_GEMMUL8_SOLVE_MIN_FREE_
   AFTER_MB), so that no block rescales its operands: on an RX 9060 XT the
   5616 x 1987 x 5616 product of the refined eigensolver takes 63 ms instead
   of 108 ms at the 256 MiB cap.  Otherwise the capped path with the same
   reserve, and when GEMMul8 is off the same as openmx_gemmul8Zgemm, native
   fallbacks counted alike.  The scratch stays until
   openmx_gemmul8TrimWorkspaces or openmx_gemmul8ReleaseWorkspaces. */
extern "C" hipblasStatus_t openmx_gemmul8ZgemmUnblocked(hipblasHandle_t handle, hipblasOperation_t transa,
                                                        hipblasOperation_t transb, int m, int n, int k,
                                                        const hipDoubleComplex *alpha, const hipDoubleComplex *A,
                                                        int lda, const hipDoubleComplex *B, int ldb,
                                                        const hipDoubleComplex *beta, hipDoubleComplex *C, int ldc)
{
    if (m <= 0 || n <= 0 || k <= 0) return HIPBLAS_STATUS_SUCCESS;
    hipblasStatus_t mgpu_status;
    if (openmx_mgpu_dispatch_zgemm && openmx_mgpu_dispatch_zgemm(handle,
            transa,transb,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc,&mgpu_status))
        return mgpu_status;
    if (!g_input_enabled || gemmul8_disabled("OPENMX_GEMMUL8_DISABLE_Z", "GEMMUL8_DISABLE_Z"))
        return openmx_gemmul8Zgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);

    return timed_general_gemm(handle, 1, 8.0 * m * n * k, [&](bool *native) -> hipblasStatus_t {
        const unsigned num_moduli = gemmul8_num_moduli("OPENMX_GEMMUL8_NUM_MOD_Z", "GEMMUL8_NUM_MOD_Z");
        const bool     fastmode   = env_bool("OPENMX_GEMMUL8_FASTMODE_Z", env_bool("GEMMUL8_FASTMODE_Z", false));
        const size_t   sm = static_cast<size_t>(m), sn = static_cast<size_t>(n), sk = static_cast<size_t>(k);
        const size_t   total = gemmul8::workSize<true, gemmul8::Backend::INT8>(sm, sn, sk, num_moduli);
        void          *work = nullptr;
        hipblasStatus_t status = ensure_scratch(handle, total, &work, solve_reserve());

        if (status == HIPBLAS_STATUS_SUCCESS) {
            gemmul8::set_memory_saving(handle, false);
            (void)gemmul8::gemm<hipDoubleComplex, gemmul8::Backend::INT8>(
                handle, transa, transb, sm, sn, sk, alpha, A, static_cast<size_t>(lda), B, static_cast<size_t>(ldb),
                beta, C, static_cast<size_t>(ldc), num_moduli, fastmode, work);
            return HIPBLAS_STATUS_SUCCESS;
        }
        if (status != HIPBLAS_STATUS_ALLOC_FAILED) return status;
        return zgemm_capped(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, solve_reserve(),
                            native);
    });
}

/* the same for real products (openmx_gemmul8Dgemm) */
extern "C" hipblasStatus_t openmx_gemmul8DgemmUnblocked(hipblasHandle_t handle, hipblasOperation_t transa,
                                                        hipblasOperation_t transb, int m, int n, int k,
                                                        const double *alpha, const double *A, int lda,
                                                        const double *B, int ldb, const double *beta, double *C,
                                                        int ldc)
{
    if (m <= 0 || n <= 0 || k <= 0) return HIPBLAS_STATUS_SUCCESS;
    hipblasStatus_t mgpu_status;
    if (openmx_mgpu_dispatch_dgemm && openmx_mgpu_dispatch_dgemm(handle,
            transa,transb,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc,&mgpu_status))
        return mgpu_status;
    if (!g_input_enabled || gemmul8_disabled("OPENMX_GEMMUL8_DISABLE_D", "GEMMUL8_DISABLE_D"))
        return openmx_gemmul8Dgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);

    return timed_general_gemm(handle, 0, 2.0 * m * n * k, [&](bool *native) -> hipblasStatus_t {
        const unsigned num_moduli = gemmul8_num_moduli("OPENMX_GEMMUL8_NUM_MOD_D", "GEMMUL8_NUM_MOD_D");
        const bool     fastmode   = env_bool("OPENMX_GEMMUL8_FASTMODE_D", env_bool("GEMMUL8_FASTMODE_D", false));
        const hipblasOperation_t ta = (transa == HIPBLAS_OP_C) ? HIPBLAS_OP_T : transa;
        const hipblasOperation_t tb = (transb == HIPBLAS_OP_C) ? HIPBLAS_OP_T : transb;
        const size_t   sm = static_cast<size_t>(m), sn = static_cast<size_t>(n), sk = static_cast<size_t>(k);
        const size_t   total = gemmul8::workSize<false, gemmul8::Backend::INT8>(sm, sn, sk, num_moduli);
        void          *work = nullptr;
        hipblasStatus_t status = ensure_scratch(handle, total, &work, solve_reserve());

        if (status == HIPBLAS_STATUS_SUCCESS) {
            gemmul8::set_memory_saving(handle, false);
            (void)gemmul8::gemm<double, gemmul8::Backend::INT8>(handle, ta, tb, sm, sn, sk, alpha, A,
                                                                static_cast<size_t>(lda), B, static_cast<size_t>(ldb),
                                                                beta, C, static_cast<size_t>(ldc), num_moduli, fastmode,
                                                                work);
            return HIPBLAS_STATUS_SUCCESS;
        }
        if (status != HIPBLAS_STATUS_ALLOC_FAILED) return status;
        return dgemm_capped(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, solve_reserve(),
                            native);
    });
}

/* OPENMX_GEMMUL8_TIMING=1: the totals of the general GEMMs since the last
   report, one line on stdout per rank that made such calls (the rank from
   OMPI_COMM_WORLD_RANK or PMI_RANK), then reset.  DFT.c calls it at the end
   of each SCF; without the variable it does nothing. */
extern "C" void openmx_gemmul8ReportTiming(void)
{
    std::lock_guard<std::mutex> lock(g_workspace_mutex);
    const GeneralCounters      &g = g_general_counters;

    if (general_timing() && 0 < g.calls[0][0] + g.calls[0][1] + g.calls[1][0] + g.calls[1][1]) {
        const char *rank = std::getenv("OMPI_COMM_WORLD_RANK");

        if (rank == nullptr) rank = std::getenv("PMI_RANK");
        std::printf("<openmx_gemmul8> general GEMMs, rank %s: real %lld + %lld calls, %.6f + %.6f s, %.3f + %.3f GFlop; "
                    "complex %lld + %lld calls, %.6f + %.6f s, %.3f + %.3f GFlop (GEMMul8 + hipBLAS); "
                    "moduli %u / %u, scaling %s / %s (real / complex)\n",
                    rank != nullptr ? rank : "-", g.calls[0][0], g.calls[0][1], g.seconds[0][0], g.seconds[0][1],
                    1e-9 * g.flops[0][0], 1e-9 * g.flops[0][1], g.calls[1][0], g.calls[1][1], g.seconds[1][0],
                    g.seconds[1][1], 1e-9 * g.flops[1][0], 1e-9 * g.flops[1][1],
                    gemmul8_num_moduli("OPENMX_GEMMUL8_NUM_MOD_D", "GEMMUL8_NUM_MOD_D"),
                    gemmul8_num_moduli("OPENMX_GEMMUL8_NUM_MOD_Z", "GEMMUL8_NUM_MOD_Z"),
                    env_bool("OPENMX_GEMMUL8_FASTMODE_D", env_bool("GEMMUL8_FASTMODE_D", false)) ? "fast" : "accurate",
                    env_bool("OPENMX_GEMMUL8_FASTMODE_Z", env_bool("GEMMUL8_FASTMODE_Z", false)) ? "fast" : "accurate");
        std::fflush(stdout);
    }
    g_general_counters = GeneralCounters{};
}

/* bytes of the GEMMul8 workspaces this process holds (all devices and
   streams): what openmx_gemmul8ReleaseWorkspaces would return */
extern "C" size_t openmx_gemmul8WorkspaceBytes(void)
{
    std::lock_guard<std::mutex> lock(g_workspace_mutex);
    size_t                      bytes = 0;

    for (const auto &item : g_workspaces) bytes += item.second.size;
    return bytes;
}

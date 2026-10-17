// Numerical checks for the entry points instantiated by gemmul8_openmx.cu,
// and for the bridge (source/gemmul8_bridge.cu) itself: the unblocked entry
// points against the capped ones (bit-identical while the full workspace fits
// the 256 MiB cap, within 1e-12 above it), their fallback to the capped path
// when the full workspace does not fit the free-memory reserve, the trim of
// the workspaces above the cap, the native-call counter and the Enabled
// queries, and the general-GEMM timing report (OPENMX_GEMMUL8_TIMING=1).
// Link with source/gemmul8_bridge.o, source/third_party/GEMMul8/lib/libgemmul8.a,
// hipBLAS and hipBLASLt; use the same HIP architecture as the OpenMX build
// (tools/run_check_gemmul8_hip.sh builds and runs it, with and without the
// timing).  Run it alone on the GPU: the trim and fallback checks read the
// device's free memory.
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include "gemmul8.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

extern "C" hipblasStatus_t openmx_gemmul8Dgemm(
    hipblasHandle_t, hipblasOperation_t, hipblasOperation_t, int, int, int,
    const double *, const double *, int, const double *, int,
    const double *, double *, int);
extern "C" hipblasStatus_t openmx_gemmul8Zgemm(
    hipblasHandle_t, hipblasOperation_t, hipblasOperation_t, int, int, int,
    const hipDoubleComplex *, const hipDoubleComplex *, int,
    const hipDoubleComplex *, int, const hipDoubleComplex *, hipDoubleComplex *, int);
extern "C" hipblasStatus_t openmx_gemmul8DgemmSolve(
    hipblasHandle_t, hipblasOperation_t, hipblasOperation_t, int, int, int,
    const double *, const double *, int, const double *, int,
    const double *, double *, int);
extern "C" hipblasStatus_t openmx_gemmul8ZgemmSolve(
    hipblasHandle_t, hipblasOperation_t, hipblasOperation_t, int, int, int,
    const hipDoubleComplex *, const hipDoubleComplex *, int,
    const hipDoubleComplex *, int, const hipDoubleComplex *, hipDoubleComplex *, int);
extern "C" hipblasStatus_t openmx_gemmul8DgemmUnblocked(
    hipblasHandle_t, hipblasOperation_t, hipblasOperation_t, int, int, int,
    const double *, const double *, int, const double *, int,
    const double *, double *, int);
extern "C" hipblasStatus_t openmx_gemmul8ZgemmUnblocked(
    hipblasHandle_t, hipblasOperation_t, hipblasOperation_t, int, int, int,
    const hipDoubleComplex *, const hipDoubleComplex *, int,
    const hipDoubleComplex *, int, const hipDoubleComplex *, hipDoubleComplex *, int);
extern "C" void openmx_gemmul8SetEnabled(int);
extern "C" void openmx_gemmul8ReleaseWorkspaces();
extern "C" void openmx_gemmul8TrimWorkspaces(void);
extern "C" size_t openmx_gemmul8WorkspaceBytes(void);
extern "C" long long openmx_gemmul8NativeCalls(int);
extern "C" int openmx_gemmul8DgemmEnabled(void);
extern "C" int openmx_gemmul8ZgemmEnabled(void);
extern "C" void openmx_gemmul8ReportTiming(void);

void check(hipError_t status) {
    if (status != hipSuccess) throw std::runtime_error(hipGetErrorString(status));
}
void check(hipblasStatus_t status) {
    if (status != HIPBLAS_STATUS_SUCCESS)
        throw std::runtime_error("hipBLAS status " + std::to_string(status));
}

struct Buffer {
    void *ptr = nullptr;
    explicit Buffer(size_t bytes) { check(hipMalloc(&ptr, bytes)); }
    ~Buffer() { if (ptr) (void)hipFree(ptr); }
    Buffer(const Buffer &) = delete;
    template <class T> T *as() { return static_cast<T *>(ptr); }
};

template <class T> constexpr bool complex_v = !std::is_same_v<T, double>;
template <class T> T scalar(double re, double im = 0) {
    if constexpr (complex_v<T>) return {re, im};
    else return re;
}
template <class T> double norm(T value) {
    if constexpr (complex_v<T>) return std::hypot(value.x, value.y);
    else return std::abs(value);
}
template <class T> double distance(T a, T b) {
    if constexpr (complex_v<T>) return std::hypot(a.x - b.x, a.y - b.y);
    else return std::abs(a - b);
}
template <class T> std::vector<T> matrix(int ld, int columns, int seed) {
    std::vector<T> result(size_t(ld) * columns);
    for (size_t i = 0; i < result.size(); ++i) {
        // Unequal signed real/imaginary components, with a modest dynamic range.
        double re = std::ldexp(double(int((i * 37 + seed) % 101) - 50) / 51,
                               int(i % 5) - 2);
        double im = std::ldexp(double(int((i * 43 + seed * 3) % 97) - 48) / 49,
                               int(i % 7) - 3);
        result[i] = scalar<T>(re, im);
    }
    return result;
}

int passed = 0;
template <class T, class Native, class Emulated>
void compare(const std::string &name, const std::vector<T> &initial,
             Native native, Emulated emulated) {
    const size_t bytes = initial.size() * sizeof(T);
    Buffer reference(bytes), candidate(bytes);
    check(hipMemcpy(reference.ptr, initial.data(), bytes, hipMemcpyHostToDevice));
    check(hipMemcpy(candidate.ptr, initial.data(), bytes, hipMemcpyHostToDevice));
    native(reference.as<T>());
    emulated(candidate.as<T>());
    check(hipDeviceSynchronize());
    std::vector<T> a(initial.size()), b(initial.size());
    check(hipMemcpy(a.data(), reference.ptr, bytes, hipMemcpyDeviceToHost));
    check(hipMemcpy(b.data(), candidate.ptr, bytes, hipMemcpyDeviceToHost));
    double error = 0, scale = 1;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(norm(a[i])) || !std::isfinite(norm(b[i])))
            throw std::runtime_error(name + ": nonfinite output");
        error = std::max(error, distance(a[i], b[i]));
        scale = std::max(scale, norm(a[i]));
    }
    // This also checks untouched triangles and leading-dimension padding.
    if (error > 2e-12 * scale) throw std::runtime_error(name + ": numerical mismatch");
    std::printf("PASS %-46s max_abs_error=%.3e scaled_error=%.3e\n",
                name.c_str(), error, error / scale);
    std::fflush(stdout);
    ++passed;
}

template <class T> void gemm(hipblasHandle_t handle, hipblasOperation_t ta,
                            hipblasOperation_t tb, int m, int n, int k,
                            bool enabled, bool require_blocking = false) {
    int lda = (ta == HIPBLAS_OP_N ? m : k) + 3;
    int ldb = (tb == HIPBLAS_OP_N ? k : n) + 4;
    int ldc = m + 5;
    auto a = matrix<T>(lda, ta == HIPBLAS_OP_N ? k : m, 7);
    auto b = matrix<T>(ldb, tb == HIPBLAS_OP_N ? n : k, 11);
    auto c = matrix<T>(ldc, n, 17);
    Buffer da(a.size() * sizeof(T)), db(b.size() * sizeof(T));
    check(hipMemcpy(da.ptr, a.data(), a.size() * sizeof(T), hipMemcpyHostToDevice));
    check(hipMemcpy(db.ptr, b.data(), b.size() * sizeof(T), hipMemcpyHostToDevice));
    const T alpha = scalar<T>(0.75, -0.25), beta = scalar<T>(-0.5, 0.125);
    const size_t full = gemmul8::workSize<complex_v<T>, gemmul8::Backend::INT8>(m, n, k, 15);
    if (require_blocking && full <= (size_t(256) << 20))
        throw std::runtime_error("The large GEMM does not exercise workspace blocking");
    openmx_gemmul8SetEnabled(enabled);
    std::string name = complex_v<T> ? "ZGEMM" : "DGEMM";
    name += " " + std::to_string(ta) + "/" + std::to_string(tb) + " " +
            std::to_string(m) + "x" + std::to_string(n) + "x" + std::to_string(k);
    name += enabled ? " emulated" : " disabled";
    if (require_blocking)
        std::printf("blocking workspace: %zu bytes, bridge cap: 268435456 bytes\n", full);
    compare<T>(name, c, [&](T *out) {
        if constexpr (complex_v<T>)
            check(hipblasZgemm(handle, ta, tb, m, n, k, &alpha, da.as<T>(), lda,
                              db.as<T>(), ldb, &beta, out, ldc));
        else
            check(hipblasDgemm(handle, ta, tb, m, n, k, &alpha, da.as<T>(), lda,
                              db.as<T>(), ldb, &beta, out, ldc));
    }, [&](T *out) {
        if constexpr (complex_v<T>)
            check(openmx_gemmul8Zgemm(handle, ta, tb, m, n, k, &alpha, da.as<T>(), lda,
                                    db.as<T>(), ldb, &beta, out, ldc));
        else
            check(openmx_gemmul8Dgemm(handle, ta, tb, m, n, k, &alpha, da.as<T>(), lda,
                                    db.as<T>(), ldb, &beta, out, ldc));
    });
}

template <class T> void structured(hipblasHandle_t h) {
    constexpr int m = 129, n = 137, k = 145, ldc = m + 5;
    const T alpha = scalar<T>(0.75, -0.25), beta = scalar<T>(-0.5);
    const double real_beta = -0.5;
    const char *type = complex_v<T> ? "Z" : "D";
    gemmul8::set_memory_saving(h, false);
    for (auto side : {HIPBLAS_SIDE_LEFT, HIPBLAS_SIDE_RIGHT}) {
        int order = side == HIPBLAS_SIDE_LEFT ? m : n;
        int lda = order + 3, ldb = m + 4;
        auto a = matrix<T>(lda, order, 19), b = matrix<T>(ldb, n, 23);
        for (int i = 0; i < order; ++i) a[size_t(i) * lda + i] = scalar<T>(2);
        Buffer da(a.size() * sizeof(T)), db(b.size() * sizeof(T));
        check(hipMemcpy(da.ptr, a.data(), a.size() * sizeof(T), hipMemcpyHostToDevice));
        check(hipMemcpy(db.ptr, b.data(), b.size() * sizeof(T), hipMemcpyHostToDevice));
        for (auto uplo : {HIPBLAS_FILL_MODE_LOWER, HIPBLAS_FILL_MODE_UPPER}) {
            Buffer work(gemmul8::workSize<complex_v<T>, gemmul8::Backend::INT8,
                        complex_v<T> ? gemmul8::Func::hemm : gemmul8::Func::symm>(m, n, order, 15));
            auto c = matrix<T>(ldc, n, 29);
            std::string suffix = " " + std::to_string(side) + "/" + std::to_string(uplo);
            compare<T>(std::string(type) + (complex_v<T> ? "HEMM" : "SYMM") + suffix, c,
                [&](T *out) {
                    if constexpr (complex_v<T>) check(hipblasZhemm(h, side, uplo, m, n, &alpha,
                        da.as<T>(), lda, db.as<T>(), ldb, &beta, out, ldc));
                    else check(hipblasDsymm(h, side, uplo, m, n, &alpha,
                        da.as<T>(), lda, db.as<T>(), ldb, &beta, out, ldc));
                }, [&](T *out) {
                    if constexpr (complex_v<T>) gemmul8::hemm<T>(h, side, uplo, m, n, &alpha,
                        da.as<T>(), lda, db.as<T>(), ldb, &beta, out, ldc, 15, false, work.ptr);
                    else gemmul8::symm<T>(h, side, uplo, m, n, &alpha,
                        da.as<T>(), lda, db.as<T>(), ldb, &beta, out, ldc, 15, false, work.ptr);
                });
            Buffer tw(gemmul8::workSize<complex_v<T>, gemmul8::Backend::INT8,
                      gemmul8::Func::trmm>(m, n, order, 15));
            for (auto op : {HIPBLAS_OP_N, HIPBLAS_OP_T, HIPBLAS_OP_C}) {
                for (auto diag : {HIPBLAS_DIAG_NON_UNIT, HIPBLAS_DIAG_UNIT}) {
                    compare<T>(std::string(type) + "TRMM" + suffix + " " +
                        std::to_string(op) + "/" + std::to_string(diag), c,
                        [&](T *out) {
                            if constexpr (complex_v<T>) check(hipblasZtrmm(h, side, uplo, op, diag,
                                m, n, &alpha, da.as<T>(), lda, db.as<T>(), ldb, out, ldc));
                            else check(hipblasDtrmm(h, side, uplo, op, diag,
                                m, n, &alpha, da.as<T>(), lda, db.as<T>(), ldb, out, ldc));
                        }, [&](T *out) {
                            gemmul8::trmm<T>(h, side, uplo, op, diag, m, n, &alpha,
                                da.as<T>(), lda, db.as<T>(), ldb, out, ldc, 15, false, tw.ptr);
                        });
                }
            }
        }
    }
    for (auto uplo : {HIPBLAS_FILL_MODE_LOWER, HIPBLAS_FILL_MODE_UPPER}) {
        for (auto op : {HIPBLAS_OP_N, complex_v<T> ? HIPBLAS_OP_C : HIPBLAS_OP_T}) {
            int lda = (op == HIPBLAS_OP_N ? m : k) + 3;
            int ldb = lda + 1, cols = op == HIPBLAS_OP_N ? k : m;
            auto a = matrix<T>(lda, cols, 31), b = matrix<T>(ldb, cols, 37);
            auto c = matrix<T>(ldc, m, 41);
            Buffer da(a.size() * sizeof(T)), db(b.size() * sizeof(T));
            check(hipMemcpy(da.ptr, a.data(), a.size() * sizeof(T), hipMemcpyHostToDevice));
            check(hipMemcpy(db.ptr, b.data(), b.size() * sizeof(T), hipMemcpyHostToDevice));
            Buffer work(gemmul8::workSize<complex_v<T>, gemmul8::Backend::INT8,
                        complex_v<T> ? gemmul8::Func::her2k : gemmul8::Func::syr2k>(m, m, k, 15));
            compare<T>(std::string(type) + (complex_v<T> ? "HER2K" : "SYR2K") + " " +
                std::to_string(uplo) + "/" + std::to_string(op), c,
                [&](T *out) {
                    if constexpr (complex_v<T>) check(hipblasZher2k(h, uplo, op, m, k, &alpha,
                        da.as<T>(), lda, db.as<T>(), ldb, &real_beta, out, ldc));
                    else check(hipblasDsyr2k(h, uplo, op, m, k, &alpha,
                        da.as<T>(), lda, db.as<T>(), ldb, &beta, out, ldc));
                }, [&](T *out) {
                    if constexpr (complex_v<T>) gemmul8::her2k<T>(h, uplo, op, m, k, &alpha,
                        da.as<T>(), lda, db.as<T>(), ldb, &real_beta, out, ldc, 15, false, work.ptr);
                    else gemmul8::syr2k<T>(h, uplo, op, m, k, &alpha,
                        da.as<T>(), lda, db.as<T>(), ldb, &beta, out, ldc, 15, false, work.ptr);
                });
        }
    }
}

// ---------------------------------------------------------------------------
// The bridge's own entry points.  The runner sets the bridge defaults:
// OPENMX_GEMMUL8_MAX_WORKSPACE_MB=256, 15 moduli, accurate scaling.

constexpr size_t kCap = size_t(256) << 20;   // the memory-saving cap of the blocked path
constexpr size_t kMiB = size_t(1) << 20;

// FNV-1a over the result of every bridge call below: the runner compares it
// between a run with OPENMX_GEMMUL8_TIMING=1 and one without
uint64_t result_checksum = 1469598103934665603ULL;
template <class T> void add_checksum(const std::vector<T> &values) {
    const unsigned char *p = reinterpret_cast<const unsigned char *>(values.data());
    for (size_t i = 0; i < values.size() * sizeof(T); ++i) {
        result_checksum ^= p[i];
        result_checksum *= 1099511628211ULL;
    }
}

void require(bool condition, const std::string &what) {
    if (!condition) throw std::runtime_error(what);
}
void pass(const std::string &name, const std::string &detail = "") {
    std::printf("PASS %-46s %s\n", name.c_str(), detail.c_str());
    std::fflush(stdout);
    ++passed;
}
std::string mib(size_t bytes) {
    char text[64];
    std::snprintf(text, sizeof(text), "%.1f MiB", double(bytes) / double(kMiB));
    return text;
}
size_t device_free() {
    size_t free_bytes = 0, total_bytes = 0;
    check(hipMemGetInfo(&free_bytes, &total_bytes));
    return free_bytes;
}

template <class T> using Entry = hipblasStatus_t (*)(hipblasHandle_t, hipblasOperation_t, hipblasOperation_t,
                                                     int, int, int, const T *, const T *, int, const T *, int,
                                                     const T *, T *, int);
template <class T> Entry<T> capped_entry() {
    if constexpr (complex_v<T>) return openmx_gemmul8Zgemm; else return openmx_gemmul8Dgemm;
}
template <class T> Entry<T> solve_entry() {
    if constexpr (complex_v<T>) return openmx_gemmul8ZgemmSolve; else return openmx_gemmul8DgemmSolve;
}
template <class T> Entry<T> unblocked_entry() {
    if constexpr (complex_v<T>) return openmx_gemmul8ZgemmUnblocked; else return openmx_gemmul8DgemmUnblocked;
}

// C = alpha op(A) op(B) + beta C0 with padded leading dimensions; every
// buffer is allocated up front, so a run allocates nothing but what the
// bridge does
template <class T> struct Product {
    hipblasOperation_t ta, tb;
    int m, n, k, lda, ldb, ldc;
    T alpha = scalar<T>(0.75, -0.25), beta = scalar<T>(-0.5, 0.125);
    std::vector<T> a, b, c0;
    Buffer da, db, dc;
    Product(hipblasOperation_t ta_, hipblasOperation_t tb_, int m_, int n_, int k_)
        : ta(ta_), tb(tb_), m(m_), n(n_), k(k_),
          lda((ta_ == HIPBLAS_OP_N ? m_ : k_) + 3), ldb((tb_ == HIPBLAS_OP_N ? k_ : n_) + 4), ldc(m_ + 5),
          a(matrix<T>(lda, ta_ == HIPBLAS_OP_N ? k_ : m_, 7)), b(matrix<T>(ldb, tb_ == HIPBLAS_OP_N ? n_ : k_, 11)),
          c0(matrix<T>(ldc, n_, 17)), da(a.size() * sizeof(T)), db(b.size() * sizeof(T)), dc(c0.size() * sizeof(T)) {
        check(hipMemcpy(da.ptr, a.data(), a.size() * sizeof(T), hipMemcpyHostToDevice));
        check(hipMemcpy(db.ptr, b.data(), b.size() * sizeof(T), hipMemcpyHostToDevice));
    }
    std::string name(const char *what) const {
        return std::string(complex_v<T> ? "Z" : "D") + what + " " + std::to_string(ta) + "/" + std::to_string(tb) +
               " " + std::to_string(m) + "x" + std::to_string(n) + "x" + std::to_string(k);
    }
    size_t full_workspace() const {
        return gemmul8::workSize<complex_v<T>, gemmul8::Backend::INT8>(m, n, k, 15);
    }
    std::vector<T> result() {
        std::vector<T> c(c0.size());
        check(hipDeviceSynchronize());
        check(hipMemcpy(c.data(), dc.ptr, c.size() * sizeof(T), hipMemcpyDeviceToHost));
        return c;
    }
    std::vector<T> run(hipblasHandle_t h, Entry<T> entry) {
        check(hipMemcpy(dc.ptr, c0.data(), c0.size() * sizeof(T), hipMemcpyHostToDevice));
        check(entry(h, ta, tb, m, n, k, &alpha, da.as<T>(), lda, db.as<T>(), ldb, &beta, dc.as<T>(), ldc));
        auto c = result();
        add_checksum(c);
        return c;
    }
    std::vector<T> native(hipblasHandle_t h) {
        check(hipMemcpy(dc.ptr, c0.data(), c0.size() * sizeof(T), hipMemcpyHostToDevice));
        if constexpr (complex_v<T>)
            check(hipblasZgemm(h, ta, tb, m, n, k, &alpha, da.as<T>(), lda, db.as<T>(), ldb, &beta, dc.as<T>(), ldc));
        else
            check(hipblasDgemm(h, ta, tb, m, n, k, &alpha, da.as<T>(), lda, db.as<T>(), ldb, &beta, dc.as<T>(), ldc));
        return result();
    }
};

template <class T> bool identical(const std::vector<T> &x, const std::vector<T> &y) {
    return x.size() == y.size() && std::memcmp(x.data(), y.data(), x.size() * sizeof(T)) == 0;
}
// max |x - y| over max(1, max |y|)
template <class T> double scaled_difference(const std::vector<T> &x, const std::vector<T> &y) {
    double error = 0, scale = 1;
    for (size_t i = 0; i < x.size(); ++i) {
        if (!std::isfinite(norm(x[i])) || !std::isfinite(norm(y[i]))) return INFINITY;
        error = std::max(error, distance(x[i], y[i]));
        scale = std::max(scale, norm(y[i]));
    }
    return error / scale;
}

// The unblocked entry point against the capped one (and its Solve variant,
// the same path with the solve reserve): bit-identical while the full
// workspace fits the cap, within 1e-12 above it; the scratch it leaves is the
// full workspace, which the trim returns only when it is above the cap.
template <class T> void unblocked_vs_capped(hipblasHandle_t h, hipblasOperation_t ta, hipblasOperation_t tb,
                                            int m, int n, int k) {
    Product<T> p(ta, tb, m, n, k);
    const size_t full = p.full_workspace();
    const bool fits = full <= kCap;
    char detail[160];

    openmx_gemmul8SetEnabled(1);
    openmx_gemmul8ReleaseWorkspaces();
    const auto native = p.native(h);
    const auto capped = p.run(h, capped_entry<T>());
    require(openmx_gemmul8WorkspaceBytes() == std::min(full, kCap), p.name("GEMM") + ": capped workspace size");
    const auto solve = p.run(h, solve_entry<T>());
    require(identical(solve, capped), p.name("GEMM") + ": the Solve entry differs from the capped one");
    const long long native_calls = openmx_gemmul8NativeCalls(complex_v<T> ? 1 : 0);
    const auto unblocked = p.run(h, unblocked_entry<T>());
    require(openmx_gemmul8NativeCalls(complex_v<T> ? 1 : 0) == native_calls, p.name("GEMM") + ": native fallback");
    require(openmx_gemmul8WorkspaceBytes() == full,
            p.name("GEMM") + ": the unblocked product holds " + mib(openmx_gemmul8WorkspaceBytes()) + ", not " +
            mib(full));
    const double to_capped = scaled_difference(unblocked, capped);
    const double to_native = scaled_difference(unblocked, native);
    if (fits)
        require(identical(unblocked, capped), p.name("GEMM") + ": unblocked differs from capped below the cap");
    else
        require(to_capped <= 1e-12, p.name("GEMM") + ": unblocked differs from capped beyond 1e-12");
    require(to_native <= 2e-12, p.name("GEMM") + ": unblocked differs from native hipBLAS beyond 2e-12");
    openmx_gemmul8TrimWorkspaces();
    require(openmx_gemmul8WorkspaceBytes() == (fits ? full : 0),
            p.name("GEMM") + ": the trim left " + mib(openmx_gemmul8WorkspaceBytes()));
    std::snprintf(detail, sizeof(detail), "workspace %s, %s, to capped %.1e, to native %.1e", mib(full).c_str(),
                  fits ? "bit-identical" : "blocked", to_capped, to_native);
    pass(p.name("GEMM unblocked"), detail);
}

// Two streams: the capped workspace of one (exactly the cap) stays through
// the trim, the full scratch of the other (above the cap) goes, and the
// device sees the memory again.
void trim_keeps_capped() {
    hipStream_t s1, s2;
    hipblasHandle_t h1, h2;
    check(hipStreamCreate(&s1));
    check(hipStreamCreate(&s2));
    check(hipblasCreate(&h1));
    check(hipblasCreate(&h2));
    check(hipblasSetStream(h1, s1));
    check(hipblasSetStream(h2, s2));
    {
        Product<double> p(HIPBLAS_OP_T, HIPBLAS_OP_N, 2808, 2808, 2808);
        const size_t full = p.full_workspace();

        require(kCap < full, "the trim check needs a workspace above the cap");
        openmx_gemmul8SetEnabled(1);
        openmx_gemmul8ReleaseWorkspaces();
        const auto capped = p.run(h2, openmx_gemmul8Dgemm);
        const auto unblocked = p.run(h1, openmx_gemmul8DgemmUnblocked);
        require(openmx_gemmul8WorkspaceBytes() == kCap + full,
                "two streams hold " + mib(openmx_gemmul8WorkspaceBytes()) + ", not " + mib(kCap + full));
        const size_t free_before = device_free();
        openmx_gemmul8TrimWorkspaces();
        const size_t free_after = device_free();
        require(openmx_gemmul8WorkspaceBytes() == kCap,
                "the trim left " + mib(openmx_gemmul8WorkspaceBytes()) + ", not the capped " + mib(kCap));
        require(free_before + full <= free_after + 16 * kMiB,
                "the trim returned " + mib(free_after - free_before) + " of the " + mib(full) + " scratch");
        openmx_gemmul8ReleaseWorkspaces();
        require(openmx_gemmul8WorkspaceBytes() == 0, "the release left workspaces behind");
        require(scaled_difference(unblocked, capped) <= 1e-12, "two-stream results differ beyond 1e-12");
        pass("trim: capped kept, scratch above the cap freed",
             "freed " + mib(free_after - free_before) + " of " + mib(full));
    }
    gemmul8::clear_config(h1);
    gemmul8::clear_config(h2);
    check(hipblasDestroy(h1));
    check(hipblasDestroy(h2));
    check(hipStreamDestroy(s1));
    check(hipStreamDestroy(s2));
}

// The unblocked product whose full workspace would leave less than the solve
// reserve (OPENMX_GEMMUL8_SOLVE_MIN_FREE_AFTER_MB) free takes the capped path
// with the same reserve: the capped result bit for bit, the capped workspace
// held, and no native call.
void unblocked_falls_back(hipblasHandle_t h) {
    Product<double> p(HIPBLAS_OP_N, HIPBLAS_OP_N, 2808, 2808, 2808);
    const size_t full = p.full_workspace();

    require(kCap < full, "the fallback check needs a workspace above the cap");
    openmx_gemmul8SetEnabled(1);
    openmx_gemmul8ReleaseWorkspaces();
    const auto capped = p.run(h, openmx_gemmul8DgemmSolve);
    openmx_gemmul8ReleaseWorkspaces();
    // free - full < reserve <= free - cap, halfway
    const size_t reserve = device_free() - (kCap + full) / 2;
    setenv("OPENMX_GEMMUL8_SOLVE_MIN_FREE_AFTER_MB", std::to_string(reserve / kMiB).c_str(), 1);
    const long long native_calls = openmx_gemmul8NativeCalls(0);
    const auto fallback = p.run(h, openmx_gemmul8DgemmUnblocked);
    const size_t held = openmx_gemmul8WorkspaceBytes();
    unsetenv("OPENMX_GEMMUL8_SOLVE_MIN_FREE_AFTER_MB");
    require(held == kCap, "the unblocked product without room holds " + mib(held) + ", not the capped " + mib(kCap));
    require(openmx_gemmul8NativeCalls(0) == native_calls, "the fallback went native");
    require(identical(fallback, capped), "the fallback differs from the capped product");
    openmx_gemmul8ReleaseWorkspaces();
    pass("unblocked without room: capped path", "reserve " + mib(reserve) + ", held " + mib(held));
}

// openmx_gemmul8NativeCalls counts the products that ran in plain hipBLAS:
// every entry point with scf.gemmul8.enable off, and the real or complex ones
// disabled from the environment; the Enabled queries follow both switches.
template <class T> void count_native(hipblasHandle_t h, const char *disable_env) {
    const int is_complex = complex_v<T> ? 1 : 0;
    Product<T> p(complex_v<T> ? HIPBLAS_OP_C : HIPBLAS_OP_T, HIPBLAS_OP_N, 129, 137, 145);
    const auto native = p.native(h);
    const long long other = openmx_gemmul8NativeCalls(1 - is_complex);
    long long expected = openmx_gemmul8NativeCalls(is_complex);
    auto enabled = [&]() { return complex_v<T> ? openmx_gemmul8ZgemmEnabled() : openmx_gemmul8DgemmEnabled(); };

    openmx_gemmul8SetEnabled(1);
    require(enabled() == 1, p.name("GEMM") + ": Enabled 0 with GEMMul8 on");
    for (auto entry : {capped_entry<T>(), solve_entry<T>(), unblocked_entry<T>()}) {
        require(scaled_difference(p.run(h, entry), native) <= 2e-12, p.name("GEMM") + ": GEMMul8 result");
        require(openmx_gemmul8NativeCalls(is_complex) == expected, p.name("GEMM") + ": counted with GEMMul8 on");
    }
    for (int how = 0; how < 2; ++how) {
        // 0: scf.gemmul8.enable off; 1: OPENMX_GEMMUL8_DISABLE_{D,Z}=1
        if (how == 0) openmx_gemmul8SetEnabled(0);
        else setenv(disable_env, "1", 1);
        require(enabled() == 0, p.name("GEMM") + ": Enabled 1 while switched off");
        require((complex_v<T> ? openmx_gemmul8DgemmEnabled() : openmx_gemmul8ZgemmEnabled()) == (how == 0 ? 0 : 1),
                p.name("GEMM") + ": the other type's Enabled");
        for (auto entry : {capped_entry<T>(), solve_entry<T>(), unblocked_entry<T>()}) {
            require(scaled_difference(p.run(h, entry), native) <= 2e-12, p.name("GEMM") + ": native result");
            require(openmx_gemmul8NativeCalls(is_complex) == ++expected, p.name("GEMM") + ": native call not counted");
        }
        require(openmx_gemmul8NativeCalls(1 - is_complex) == other, p.name("GEMM") + ": the other type counted");
        if (how == 0) openmx_gemmul8SetEnabled(1);
        else unsetenv(disable_env);
    }
    require(enabled() == 1, p.name("GEMM") + ": Enabled 0 after switching back on");
    pass(p.name("GEMM native counter"), "3 + 3 native calls counted");
}

int main() {
    try {
        std::printf("GEMMul8 %s, INT8, num_moduli=15\n", gemmul8::version_string);
        hipblasHandle_t h;
        check(hipblasCreate(&h));
        {
            // the bridge's entry points, before the checks below switch
            // GEMMul8 off: unblocked against capped below and above the cap
            // (the forward transform of the collinear cluster solver is
            // N/N and T/N n x n x n), the trim, the fallback, the counters
            unblocked_vs_capped<double>(h, HIPBLAS_OP_N, HIPBLAS_OP_N, 129, 137, 145);
            unblocked_vs_capped<hipDoubleComplex>(h, HIPBLAS_OP_C, HIPBLAS_OP_N, 129, 137, 145);
            unblocked_vs_capped<double>(h, HIPBLAS_OP_N, HIPBLAS_OP_N, 2048, 2048, 2048);
            unblocked_vs_capped<double>(h, HIPBLAS_OP_T, HIPBLAS_OP_N, 2048, 2048, 2048);
            unblocked_vs_capped<hipDoubleComplex>(h, HIPBLAS_OP_N, HIPBLAS_OP_C, 1024, 1024, 1024);
            unblocked_vs_capped<double>(h, HIPBLAS_OP_N, HIPBLAS_OP_N, 2808, 2808, 2808);
            unblocked_vs_capped<double>(h, HIPBLAS_OP_T, HIPBLAS_OP_N, 2808, 2808, 2808);
            unblocked_vs_capped<double>(h, HIPBLAS_OP_N, HIPBLAS_OP_T, 4097, 2049, 2049);
            unblocked_vs_capped<hipDoubleComplex>(h, HIPBLAS_OP_C, HIPBLAS_OP_N, 2049, 2049, 2049);
            trim_keeps_capped();
            unblocked_falls_back(h);
            count_native<double>(h, "OPENMX_GEMMUL8_DISABLE_D");
            count_native<hipDoubleComplex>(h, "OPENMX_GEMMUL8_DISABLE_Z");
            openmx_gemmul8ReleaseWorkspaces();
        }
        for (auto a : {HIPBLAS_OP_N, HIPBLAS_OP_T, HIPBLAS_OP_C}) {
            for (auto b : {HIPBLAS_OP_N, HIPBLAS_OP_T, HIPBLAS_OP_C}) {
                gemm<double>(h, a, b, 129, 137, 145, true);
                gemm<hipDoubleComplex>(h, a, b, 129, 137, 145, true);
            }
        }
        gemm<double>(h, HIPBLAS_OP_N, HIPBLAS_OP_N, 4097, 2049, 2049, true, true);
        gemm<hipDoubleComplex>(h, HIPBLAS_OP_C, HIPBLAS_OP_N, 2049, 2049, 2049, true, true);
        gemm<double>(h, HIPBLAS_OP_T, HIPBLAS_OP_N, 129, 137, 145, false);
        gemm<hipDoubleComplex>(h, HIPBLAS_OP_C, HIPBLAS_OP_T, 129, 137, 145, false);
        openmx_gemmul8ReleaseWorkspaces();
        structured<double>(h);
        structured<hipDoubleComplex>(h);
        gemmul8::clear_config(h);
        check(hipblasDestroy(h));
        // OPENMX_GEMMUL8_TIMING=1: the general-GEMM totals of this run
        openmx_gemmul8ReportTiming();
        std::printf("bridge result checksum: %016llx\n", (unsigned long long)result_checksum);
        std::printf("All %d checks passed.\n", passed);
        return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}

/* Offline study of the two forward GEMMs of the dense eigensolver path,
 *   B = H X   and   C = X^T B   (X: transformed overlap, fixed during an SCF),
 * on matrices written with OPENMX_GEMM_SAMPLE_DIR (Cluster_DFT_Col.c) or on a
 * synthetic pair.  Per moduli count and scaling mode it reports
 *   - whether GEMMul8's skip_scal reuse of the prepared X reproduces the
 *     normal path bit for bit, also for a Hamiltonian other than the one X
 *     was prepared with;
 *   - the errors of cuBLAS FP64 and of GEMMul8 (with and without reuse)
 *     against a double-double reference on sampled columns;
 *   - the wall time of each GEMM with and without reuse, and GEMMul8's phase
 *     times when linked against a GEMMul8_PROFILE=1 object;
 *   - the bytes of the retained representations and of the scratch area;
 *   - errors and times of the memory-saving (blocked) path under OpenMX's
 *     default workspace cap, which is what production runs execute;
 *   - the random-direction error indicator of a computed C (eta), with and
 *     without the eigensolver's one-triangle view, and its cost.
 * "capacity" and "selftest" need no GPU.
 * Build and run with tests/run_gemmul8_reuse_probe.sh. */
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform_reduce.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "gemmul8.hpp"

namespace {

using gemmul8::Backend;
using Clock = std::chrono::steady_clock;

[[noreturn]] void die(const char *format, ...)
{
    va_list args;
    va_start(args, format);
    vfprintf(stderr, format, args);
    va_end(args);
    fputc('\n', stderr);
    exit(2);
}

void check(cudaError_t status, const char *what)
{
    if (status != cudaSuccess) die("%s: %s", what, cudaGetErrorString(status));
}

void check(cublasStatus_t status, const char *what)
{
    if (status != CUBLAS_STATUS_SUCCESS) die("%s: cuBLAS status %d", what, int(status));
}

double seconds_since(Clock::time_point start)
{
    return std::chrono::duration<double>(Clock::now() - start).count();
}

double median(std::vector<double> values)
{
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

std::vector<int> parse_list(const char *text)
{
    std::vector<int> values;
    while (*text != '\0') {
        char *end;
        const long value = strtol(text, &end, 10);
        if (end == text) die("bad list element near \"%s\"", text);
        values.push_back(int(value));
        text = (*end == ',') ? end + 1 : end;
    }
    return values;
}

/* ------------------------------------------------------------------------
 * Workspace of one (matrix order, moduli count, scaling mode): the normal
 * call keeps everything in one scratch area; with reuse, X needs one
 * retained representation per role and the scratch holds the rest.
 * ---------------------------------------------------------------------- */
struct Sizes {
    size_t normal;   /* scratch of a call without reuse */
    size_t keep_xr;  /* X as the B operand of H X */
    size_t keep_xl;  /* X as the A operand of X^T B */
    size_t rest;     /* scratch of the two calls with reuse */
    size_t scratch;  /* scratch serving the normal and both reuse calls */
};

template <bool COMPLEX> Sizes sizes(size_t n, int moduli, bool fast)
{
    Sizes s{};
    size_t a = 0, b = 0;

    s.normal = gemmul8::workSize<COMPLEX, Backend::INT8>(n, n, n, moduli, false, false, nullptr, nullptr, fast);
    const size_t with_xr = gemmul8::workSize<COMPLEX, Backend::INT8>(n, n, n, moduli, false, true, &a, &b, fast);
    s.keep_xr = b;
    const size_t rest_xr = with_xr - b;
    const size_t with_xl = gemmul8::workSize<COMPLEX, Backend::INT8>(n, n, n, moduli, true, false, &a, &b, fast);
    s.keep_xl = a;
    const size_t rest_xl = with_xl - a;
    s.rest = std::max(rest_xr, rest_xl);
    s.scratch = std::max(s.normal, s.rest);
    return s;
}

int capacity(int argc, char **argv)
{
    const double mib = 1024.0 * 1024.0;

    if (argc < 1) die("capacity: give at least one matrix order");
    printf("# INT8 backend, square n x n operands; MiB.  normal = scratch of a call without reuse;\n");
    printf("# keep_XR, keep_XL = retained X for H X and for X^T B, rest = scratch of those two calls;\n");
    printf("# dense = X, H, B and C in FP64; total = dense + normal, total_reuse = dense + rest + keep_XR + keep_XL\n");
    printf("%-8s %6s %3s %-5s %10s %10s %10s %10s %10s %12s %12s\n", "type", "n", "L", "mode", "normal", "keep_XR",
           "keep_XL", "rest", "dense", "total", "total_reuse");
    for (int arg = 0; arg < argc; ++arg) {
        const size_t n = size_t(atoll(argv[arg]));
        for (int complex = 0; complex < 2; ++complex) {
            const double dense = 4.0 * double(n) * double(n) * (complex ? 16.0 : 8.0);
            for (int moduli : {8, 10, 12, 14, 15, 16, 18, 20}) {
                for (int fast = 0; fast < 2; ++fast) {
                    const Sizes s = complex ? sizes<true>(n, moduli, fast != 0) : sizes<false>(n, moduli, fast != 0);
                    printf("%-8s %6zu %3d %-5s %10.1f %10.1f %10.1f %10.1f %10.1f %12.1f %12.1f\n",
                           complex ? "complex" : "real", n, moduli, fast ? "fast" : "accu", s.normal / mib,
                           s.keep_xr / mib, s.keep_xl / mib, s.rest / mib, dense / mib, (dense + s.normal) / mib,
                           (dense + s.rest + s.keep_xr + s.keep_xl) / mib);
                }
            }
        }
    }
    return 0;
}

/* ------------------------------------------------------------------------
 * Double-double reference on the host (Ogita-Rump-Oishi compensated dot
 * products): errors about eps^2 * sum |terms|, far below those of FP64.
 * ---------------------------------------------------------------------- */
inline void two_sum(double a, double b, double &sum, double &error)
{
    sum = a + b;
    const double bb = sum - a;
    error = (a - (sum - bb)) + (b - bb);
}

/* (s, c) += a * b */
inline void dd_add_product(double a, double b, double &s, double &c)
{
    const double p = a * b;
    const double pe = std::fma(a, b, -p);
    double sum, se;
    two_sum(s, p, sum, se);
    s = sum;
    c += pe + se;
}

/* b = H x for a column-major n x n H */
void dd_matvec(const double *H, const double *x, size_t n, double *bh, double *bl)
{
    const size_t block = 256;
#pragma omp parallel
    {
        std::vector<double> s(block), c(block);
#pragma omp for schedule(static)
        for (long long first = 0; first < (long long)n; first += (long long)block) {
            const size_t i0 = size_t(first), len = std::min(block, n - i0);
            std::fill(s.begin(), s.end(), 0.0);
            std::fill(c.begin(), c.end(), 0.0);
            for (size_t k = 0; k < n; ++k) {
                const double xk = x[k];
                const double *h = H + i0 + k * n;
                for (size_t i = 0; i < len; ++i) dd_add_product(h[i], xk, s[i], c[i]);
            }
            for (size_t i = 0; i < len; ++i) two_sum(s[i], c[i], bh[i0 + i], bl[i0 + i]);
        }
    }
}

/* c = X^T (bh + bl) */
void dd_tmatvec(const double *X, const double *bh, const double *bl, size_t n, double *ch, double *cl)
{
#pragma omp parallel for schedule(static)
    for (long long col = 0; col < (long long)n; ++col) {
        const double *x = X + size_t(col) * n;
        double s = 0.0, c = 0.0;
        for (size_t k = 0; k < n; ++k) {
            dd_add_product(x[k], bh[k], s, c);
            c += x[k] * bl[k];
        }
        two_sum(s, c, ch[col], cl[col]);
    }
}

struct Reference {
    std::vector<double> bh, bl; /* H x_j,        n values per sampled column */
    std::vector<double> ch, cl; /* X^T (H x_j),  n values per sampled column */
};

Reference reference(const std::vector<double> &X, const std::vector<double> &H, size_t n, const std::vector<int> &columns)
{
    Reference r;
    const size_t count = n * columns.size();

    r.bh.resize(count); r.bl.resize(count); r.ch.resize(count); r.cl.resize(count);
    for (size_t c = 0; c < columns.size(); ++c) {
        dd_matvec(H.data(), X.data() + size_t(columns[c]) * n, n, &r.bh[c * n], &r.bl[c * n]);
        dd_tmatvec(X.data(), &r.bh[c * n], &r.bl[c * n], n, &r.ch[c * n], &r.cl[c * n]);
    }
    return r;
}

/* ------------------------------------------------------------------------
 * Inputs
 * ---------------------------------------------------------------------- */
std::vector<double> read_matrix(const std::string &path, size_t &n)
{
    FILE *fp = fopen(path.c_str(), "rb");
    if (fp == nullptr) die("cannot open %s", path.c_str());
    if (fseeko(fp, 0, SEEK_END) != 0) die("cannot seek in %s", path.c_str());
    const size_t count = size_t(ftello(fp)) / sizeof(double);
    const size_t order = size_t(std::llround(std::sqrt(double(count))));
    if (order == 0 || order * order * sizeof(double) != size_t(ftello(fp)))
        die("%s is not a square matrix of doubles", path.c_str());
    if (n == 0) n = order;
    if (n != order) die("%s has order %zu, expected %zu", path.c_str(), order, n);
    rewind(fp);
    std::vector<double> a(count);
    if (fread(a.data(), sizeof(double), count, fp) != count) die("short read from %s", path.c_str());
    fclose(fp);
    return a;
}

/* X = (I - 2 v v^T) diag(s^-1/2) with overlap eigenvalues s log-spaced over
   [1e-6, 2]; H_t = H0 + 2^-t D with exponentially decaying symmetric H0, D. */
void synthetic(size_t n, int steps, std::vector<double> &X, std::vector<std::vector<double>> &H)
{
    std::mt19937_64 engine(20261005);
    std::uniform_real_distribution<double> uniform(-1.0, 1.0);
    std::vector<double> v(n), base(n * n), delta(n * n);
    double norm = 0.0;

    for (size_t i = 0; i < n; ++i) { v[i] = uniform(engine); norm += v[i] * v[i]; }
    X.assign(n * n, 0.0);
    for (size_t j = 0; j < n; ++j) {
        const double s = 1.0e-6 * std::pow(2.0e6, double(j) / double(n > 1 ? n - 1 : 1));
        const double d = 1.0 / std::sqrt(s);
        for (size_t i = 0; i < n; ++i) X[i + j * n] = ((i == j ? 1.0 : 0.0) - 2.0 * v[i] * v[j] / norm) * d;
    }
    for (size_t j = 0; j < n; ++j) {
        for (size_t i = j; i < n; ++i) {
            const double decay = std::exp(-double(i - j) / 12.0);
            base[i + j * n] = base[j + i * n] = decay * uniform(engine) - (i == j ? 1.0 + double(i) / double(n) : 0.0);
            delta[i + j * n] = delta[j + i * n] = 0.05 * decay * uniform(engine);
        }
    }
    H.assign(size_t(steps), std::vector<double>(n * n));
    for (int t = 0; t < steps; ++t)
        for (size_t i = 0; i < n * n; ++i) H[size_t(t)][i] = base[i] + std::ldexp(delta[i], -t);
}

int selftest()
{
    const size_t n = 300;
    std::vector<double> X;
    std::vector<std::vector<double>> H;
    synthetic(n, 1, X, H);

    const std::vector<int> columns = {0, 137, int(n) - 1};
    const Reference r = reference(X, H[0], n, columns);
    long double worst_dd = 0, worst_fp64 = 0;
    for (size_t c = 0; c < columns.size(); ++c) {
        const double *x = &X[size_t(columns[c]) * n];
        std::vector<long double> b(n);
        std::vector<double> b64(n);
        for (size_t i = 0; i < n; ++i) {
            long double sum = 0, magnitude = 0;
            double sum64 = 0;
            for (size_t k = 0; k < n; ++k) {
                sum += (long double)H[0][i + k * n] * x[k];
                sum64 += H[0][i + k * n] * x[k];
                magnitude += fabsl((long double)H[0][i + k * n] * x[k]);
            }
            b[i] = sum; b64[i] = sum64;
            worst_dd = std::max(worst_dd, fabsl(((long double)r.bh[c * n + i] - sum) + r.bl[c * n + i]) / magnitude);
            worst_fp64 = std::max(worst_fp64, fabsl((long double)sum64 - sum) / magnitude);
        }
        for (size_t i = 0; i < n; ++i) {
            long double sum = 0, magnitude = 0;
            double sum64 = 0;
            for (size_t k = 0; k < n; ++k) {
                sum += (long double)X[k + i * n] * b[k];
                sum64 += X[k + i * n] * b64[k];
                magnitude += fabsl((long double)X[k + i * n] * b[k]);
            }
            worst_dd = std::max(worst_dd, fabsl(((long double)r.ch[c * n + i] - sum) + r.cl[c * n + i]) / magnitude);
            worst_fp64 = std::max(worst_fp64, fabsl((long double)sum64 - sum) / magnitude);
        }
    }
    /* long double carries 11 more bits than double, so the double-double
       values must agree with it far better than plain FP64 does */
    printf("selftest: double-double vs long double %.3Le, FP64 vs long double %.3Le (relative to sum |terms|)\n",
           worst_dd, worst_fp64);
    if (!(worst_dd < 1.0e-16L && worst_dd * 64 < worst_fp64)) {
        printf("FAIL\n");
        return 1;
    }
    printf("PASS\n");
    return 0;
}

/* ------------------------------------------------------------------------
 * Device side
 * ---------------------------------------------------------------------- */
struct BitsDiffer {
    __host__ __device__ long long operator()(uint64_t a, uint64_t b) const { return a != b ? 1 : 0; }
};
struct AbsDifference {
    __host__ __device__ double operator()(double a, double b) const { return fabs(a - b); }
};
struct SquaredDifference {
    __host__ __device__ double operator()(double a, double b) const { const double d = a - b; return d * d; }
};
struct AsymmetrySquared {
    const double *a;
    size_t        n;
    __host__ __device__ double operator()(size_t index) const
    {
        const size_t i = index % n, j = index / n;
        if (i <= j) return 0.0;
        const double d = a[index] - a[j + i * n];
        return d * d;
    }
};

long long bit_mismatches(const double *a, const double *b, size_t count)
{
    const auto pa = thrust::device_pointer_cast(reinterpret_cast<const uint64_t *>(a));
    const auto pb = thrust::device_pointer_cast(reinterpret_cast<const uint64_t *>(b));
    return thrust::inner_product(pa, pa + count, pb, 0LL, thrust::plus<long long>(), BitsDiffer());
}

double max_abs_difference(const double *a, const double *b, size_t count)
{
    const auto pa = thrust::device_pointer_cast(a), pb = thrust::device_pointer_cast(b);
    return thrust::inner_product(pa, pa + count, pb, 0.0, thrust::maximum<double>(), AbsDifference());
}

double frobenius_difference(const double *a, const double *b, size_t count)
{
    const auto pa = thrust::device_pointer_cast(a), pb = thrust::device_pointer_cast(b);
    return std::sqrt(thrust::inner_product(pa, pa + count, pb, 0.0, thrust::plus<double>(), SquaredDifference()));
}

double frobenius(const double *a, size_t count)
{
    const auto pa = thrust::device_pointer_cast(a);
    return std::sqrt(thrust::inner_product(pa, pa + count, pa, 0.0));
}

/* || A - A^T ||_F */
double asymmetry(const double *a, size_t n)
{
    return std::sqrt(2.0 * thrust::transform_reduce(thrust::device, thrust::counting_iterator<size_t>(0),
                                                    thrust::counting_iterator<size_t>(n * n), AsymmetrySquared{a, n},
                                                    0.0, thrust::plus<double>()));
}

enum class Reuse { off, prepare, use };

/* GEMMul8's memory-saving mode with a workspace cap, as the OpenMX bridge
   sets it by default; cap 0 returns to the full workspace */
void set_blocking(cublasHandle_t handle, size_t cap_bytes)
{
    gemmul8::set_memory_saving(handle, cap_bytes != 0);
    gemmul8::set_max_worksize(handle, cap_bytes);
}

struct Device {
    cublasHandle_t handle = nullptr;
    size_t         n = 0;
    double        *X = nullptr, *H = nullptr;
    double        *B = nullptr, *C = nullptr;   /* path under test */
    double        *Bn = nullptr, *Cn = nullptr; /* normal path */
    double        *R = nullptr;                 /* FP64 result or a third product */
    void          *work = nullptr, *keep_xr = nullptr, *keep_xl = nullptr;
};

struct Call {
    double              wall = 0.0;
    std::vector<double> phases; /* scaling, products, reduction, CRT (GEMMul8_PROFILE=1) */
};

/* out = H X */
Call gemm_hx(const Device &d, int moduli, bool fast, Reuse reuse, double *out)
{
    const double one = 1.0, zero = 0.0;
    const bool   keep = reuse != Reuse::off;
    Call         call;

    check(cudaDeviceSynchronize(), "synchronize");
    const auto start = Clock::now();
    call.phases = gemmul8::gemm<double, Backend::INT8>(d.handle, CUBLAS_OP_N, CUBLAS_OP_N, d.n, d.n, d.n, &one, d.H,
                                                       d.n, d.X, d.n, &zero, out, d.n, moduli, fast, d.work, nullptr,
                                                       keep ? d.keep_xr : nullptr, false, keep, false,
                                                       reuse == Reuse::use);
    check(cudaDeviceSynchronize(), "GEMMul8 H X");
    call.wall = seconds_since(start);
    return call;
}

/* out = X^T in */
Call gemm_xtb(const Device &d, int moduli, bool fast, Reuse reuse, const double *in, double *out)
{
    const double one = 1.0, zero = 0.0;
    const bool   keep = reuse != Reuse::off;
    Call         call;

    check(cudaDeviceSynchronize(), "synchronize");
    const auto start = Clock::now();
    call.phases = gemmul8::gemm<double, Backend::INT8>(d.handle, CUBLAS_OP_T, CUBLAS_OP_N, d.n, d.n, d.n, &one, d.X,
                                                       d.n, in, d.n, &zero, out, d.n, moduli, fast, d.work,
                                                       keep ? d.keep_xl : nullptr, nullptr, keep, false,
                                                       reuse == Reuse::use, false);
    check(cudaDeviceSynchronize(), "GEMMul8 X^T B");
    call.wall = seconds_since(start);
    return call;
}

void fp64_pair(const Device &d, double *outB, double *outC, double wall[2])
{
    const double one = 1.0, zero = 0.0;
    const int    n = int(d.n);

    check(cudaDeviceSynchronize(), "synchronize");
    auto start = Clock::now();
    check(cublasDgemm(d.handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, d.H, n, d.X, n, &zero, outB, n), "cublasDgemm");
    check(cudaDeviceSynchronize(), "cublasDgemm H X");
    wall[0] = seconds_since(start);
    start = Clock::now();
    check(cublasDgemm(d.handle, CUBLAS_OP_T, CUBLAS_OP_N, n, n, n, &one, d.X, n, outB, n, &zero, outC, n), "cublasDgemm");
    check(cudaDeviceSynchronize(), "cublasDgemm X^T B");
    wall[1] = seconds_since(start);
}

/* Random-direction indicator of the error of a computed C = X^T H X: with
   Omega (n x b) and Z = X Omega prepared once per X, V = X^T (H Z) costs two
   thin FP64 products and is compared with C Omega.  "full" takes C as
   computed, "herm" the symmetric matrix the eigensolver builds from the lower
   triangle of C. */
struct Monitor {
    int                 b = 16;
    double             *omega = nullptr, *z = nullptr, *hz = nullptr, *v = nullptr, *u = nullptr, *uh = nullptr;
    std::vector<double> host_v, host_u, host_uh;
};

struct Eta {
    double full[3] = {0, 0, 0}; /* first 4, 8 and 16 columns of Omega */
    double herm[3] = {0, 0, 0};
    double wall = 0.0;          /* the three thin products of the "full" variant */
};

Eta monitor_eta(const Device &d, Monitor &m, const double *C)
{
    const double one = 1.0, zero = 0.0;
    const int    n = int(d.n), b = m.b;
    const size_t bytes = d.n * size_t(b) * sizeof(double);
    Eta          e;

    check(cudaDeviceSynchronize(), "synchronize");
    const auto start = Clock::now();
    check(cublasDgemm(d.handle, CUBLAS_OP_N, CUBLAS_OP_N, n, b, n, &one, d.H, n, m.z, n, &zero, m.hz, n), "H Z");
    check(cublasDgemm(d.handle, CUBLAS_OP_T, CUBLAS_OP_N, n, b, n, &one, d.X, n, m.hz, n, &zero, m.v, n), "X^T (H Z)");
    check(cublasDgemm(d.handle, CUBLAS_OP_N, CUBLAS_OP_N, n, b, n, &one, C, n, m.omega, n, &zero, m.u, n), "C Omega");
    check(cudaDeviceSynchronize(), "indicator products");
    e.wall = seconds_since(start);
    check(cublasDsymm(d.handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, n, b, &one, C, n, m.omega, n, &zero, m.uh, n),
          "sym(C) Omega");
    check(cudaMemcpy(m.host_v.data(), m.v, bytes, cudaMemcpyDeviceToHost), "download V");
    check(cudaMemcpy(m.host_u.data(), m.u, bytes, cudaMemcpyDeviceToHost), "download U");
    check(cudaMemcpy(m.host_uh.data(), m.uh, bytes, cudaMemcpyDeviceToHost), "download U (herm)");

    double full = 0.0, herm = 0.0, norm = 0.0;
    for (int col = 0, slot = 0; col < b; ++col) {
        for (size_t i = size_t(col) * d.n; i < size_t(col + 1) * d.n; ++i) {
            const double df = m.host_u[i] - m.host_v[i], dh = m.host_uh[i] - m.host_v[i];
            full += df * df;
            herm += dh * dh;
            norm += m.host_v[i] * m.host_v[i];
        }
        if (col + 1 == 4 || col + 1 == 8 || col + 1 == 16) {
            e.full[slot] = std::sqrt(full / norm);
            e.herm[slot] = std::sqrt(herm / norm);
            ++slot;
        }
    }
    return e;
}

struct Error {
    double relative = 0.0; /* Frobenius norm over the sampled columns */
    double max_abs = 0.0;
};

Error column_error(const double *device, size_t n, const std::vector<int> &columns, const std::vector<double> &hi,
                   const std::vector<double> &lo)
{
    std::vector<double> column(n);
    double              num = 0.0, den = 0.0;
    Error               e;

    for (size_t c = 0; c < columns.size(); ++c) {
        check(cudaMemcpy(column.data(), device + size_t(columns[c]) * n, n * sizeof(double), cudaMemcpyDeviceToHost),
              "download a column");
        for (size_t i = 0; i < n; ++i) {
            const double difference = (column[i] - hi[c * n + i]) - lo[c * n + i];
            num += difference * difference;
            den += hi[c * n + i] * hi[c * n + i];
            e.max_abs = std::max(e.max_abs, std::fabs(difference));
        }
    }
    e.relative = std::sqrt(num / den);
    return e;
}

/* ------------------------------------------------------------------------
 * One JSON object per line
 * ---------------------------------------------------------------------- */
struct Json {
    std::string text = "{";

    void key(const char *name)
    {
        if (text.size() > 1) text += ",";
        text += "\"";
        text += name;
        text += "\":";
    }
    Json &str(const char *name, const std::string &value)
    {
        key(name);
        text += "\"" + value + "\"";
        return *this;
    }
    Json &num(const char *name, double value)
    {
        char buffer[40];
        key(name);
        if (std::isfinite(value)) snprintf(buffer, sizeof(buffer), "%.9g", value);
        else snprintf(buffer, sizeof(buffer), "null");
        text += buffer;
        return *this;
    }
    Json &integer(const char *name, long long value)
    {
        key(name);
        text += std::to_string(value);
        return *this;
    }
    Json &list(const char *name, const std::vector<double> &values)
    {
        key(name);
        text += "[";
        for (size_t i = 0; i < values.size(); ++i) {
            char buffer[40];
            snprintf(buffer, sizeof(buffer), "%s%.6g", i ? "," : "", values[i]);
            text += buffer;
        }
        text += "]";
        return *this;
    }
    void emit(FILE *fp)
    {
        if (fp == nullptr) return;
        fprintf(fp, "%s}\n", text.c_str());
        fflush(fp);
    }
};

std::string base_name(const std::string &path)
{
    const size_t slash = path.find_last_of('/');
    std::string  name = slash == std::string::npos ? path : path.substr(slash + 1);
    if (name.size() > 4 && name.compare(name.size() - 4, 4, ".bin") == 0) name.resize(name.size() - 4);
    return name;
}

[[noreturn]] void usage()
{
    fprintf(stderr,
            "usage: gemmul8_reuse_probe capacity N [N ...]\n"
            "       gemmul8_reuse_probe selftest\n"
            "       gemmul8_reuse_probe run (--x FILE --h FILE [--h FILE ...] | --synthetic N [--steps T])\n"
            "           [--label NAME] [--moduli 8,10,12,14,15,16,18,20] [--fast 0,1] [--prep INDEX]\n"
            "           [--cols 8] [--reps 3] [--blocked-mb 256] [--json FILE]\n"
            "  --prep        index of the --h matrix X is prepared with (default 0)\n"
            "  --blocked-mb  workspace cap of the additional memory-saving path (0: skip it)\n");
    exit(2);
}

int run(int argc, char **argv)
{
    std::string              x_path, label = "probe", json_path;
    std::vector<std::string> h_paths;
    std::vector<int>         moduli_list = {8, 10, 12, 14, 15, 16, 18, 20}, fast_list = {0, 1};
    size_t                   n = 0;
    int                      steps = 3, prep = 0, n_columns = 8, reps = 3, blocked_mb = 256;

    for (int i = 0; i < argc; ++i) {
        const std::string option = argv[i];
        const auto        value = [&]() -> const char * {
            if (i + 1 >= argc) usage();
            return argv[++i];
        };
        if (option == "--x") x_path = value();
        else if (option == "--h") h_paths.push_back(value());
        else if (option == "--synthetic") n = size_t(atoll(value()));
        else if (option == "--steps") steps = atoi(value());
        else if (option == "--label") label = value();
        else if (option == "--moduli") moduli_list = parse_list(value());
        else if (option == "--fast") fast_list = parse_list(value());
        else if (option == "--prep") prep = atoi(value());
        else if (option == "--cols") n_columns = atoi(value());
        else if (option == "--reps") reps = atoi(value());
        else if (option == "--blocked-mb") blocked_mb = atoi(value());
        else if (option == "--json") json_path = value();
        else usage();
    }

    /* inputs */
    std::vector<double>              X;
    std::vector<std::vector<double>> H;
    std::vector<std::string>         h_names;
    if (!x_path.empty()) {
        if (h_paths.empty()) usage();
        n = 0;
        X = read_matrix(x_path, n);
        for (const auto &path : h_paths) {
            H.push_back(read_matrix(path, n));
            h_names.push_back(base_name(path));
        }
    } else if (n > 0) {
        synthetic(n, steps, X, H);
        for (int t = 0; t < steps; ++t) h_names.push_back("synthetic_t" + std::to_string(t));
    } else {
        usage();
    }
    const size_t T = H.size(), count = n * n;
    if (prep < 0 || size_t(prep) >= T) die("--prep must name one of the %zu Hamiltonians", T);
    if (reps < 1) reps = 1;

    /* X = U diag(s^-1/2) with orthonormal U: the column norms give back the
       overlap eigenvalues */
    std::vector<double> column_norm(n);
#pragma omp parallel for schedule(static)
    for (long long j = 0; j < (long long)n; ++j) {
        double sum = 0.0;
        for (size_t i = 0; i < n; ++i) sum += X[i + size_t(j) * n] * X[i + size_t(j) * n];
        column_norm[size_t(j)] = std::sqrt(sum);
    }
    const auto   extremes = std::minmax_element(column_norm.begin(), column_norm.end());
    const double s_min = 1.0 / (*extremes.second * *extremes.second), s_max = 1.0 / (*extremes.first * *extremes.first);
    long long    clamped = 0;
    for (double norm : column_norm) clamped += (1.0 / (norm * norm) <= 1.0000001e-10);

    /* sampled columns: the two extreme columns of X and evenly spaced ones */
    std::vector<int> columns = {int(extremes.second - column_norm.begin()), int(extremes.first - column_norm.begin())};
    const int        spaced = std::max(0, std::min<int>(n_columns, int(n)) - 2);
    for (int c = 0; c < spaced; ++c) {
        const int candidate = int((2 * size_t(c) + 1) * n / (2 * size_t(spaced)));
        if (std::find(columns.begin(), columns.end(), candidate) == columns.end()) columns.push_back(candidate);
    }

    FILE *json = json_path.empty() ? nullptr : fopen(json_path.c_str(), "a");
    if (!json_path.empty() && json == nullptr) die("cannot open %s", json_path.c_str());

    printf("label %s: n = %zu, %zu Hamiltonian(s), X prepared with %s\n", label.c_str(), n, T, h_names[size_t(prep)].c_str());
    printf("overlap eigenvalues implied by X: min %.3e, max %.3e, clamped at 1e-10: %lld; ||X||_2 = %.3e\n", s_min,
           s_max, clamped, *extremes.second);

    auto                   start = Clock::now();
    std::vector<Reference> refs;
    for (size_t t = 0; t < T; ++t) refs.push_back(reference(X, H[t], n, columns));
    printf("double-double reference on %zu columns: %.1f s\n", columns.size(), seconds_since(start));

    /* device */
    Device d;
    d.n = n;
    check(cublasCreate(&d.handle), "cublasCreate");
    gemmul8::set_memory_saving(d.handle, false);
    int    cublas_version = 0, device_id = 0;
    size_t free_bytes = 0, total_bytes = 0;
    check(cublasGetVersion(d.handle, &cublas_version), "cublasGetVersion");
    check(cudaGetDevice(&device_id), "cudaGetDevice");
    cudaDeviceProp properties;
    check(cudaGetDeviceProperties(&properties, device_id), "cudaGetDeviceProperties");
    for (double **matrix : {&d.X, &d.H, &d.B, &d.C, &d.Bn, &d.Cn, &d.R})
        check(cudaMalloc(matrix, count * sizeof(double)), "allocate a dense matrix");
    check(cudaMemcpy(d.X, X.data(), count * sizeof(double), cudaMemcpyHostToDevice), "upload X");
    check(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
    printf("GPU %s (%.1f GiB), cuBLAS %d, GEMMul8 %s; %.1f GiB free after the dense matrices\n", properties.name,
           total_bytes / 1073741824.0, cublas_version, GEMMUL8_VERSION_STRING, free_bytes / 1073741824.0);

    /* configurations that fit, and the largest work areas among them */
    struct Config {
        int   moduli;
        bool  fast;
        Sizes size;
    };
    std::vector<Config> configs;
    size_t              scratch_bytes = 0, keep_bytes = 0;
    const size_t        reserve = size_t(512) << 20;
    for (int moduli : moduli_list) {
        for (int fast : fast_list) {
            const Sizes s = sizes<false>(n, moduli, fast != 0);
            if (s.scratch + s.keep_xr + s.keep_xl + reserve > free_bytes) {
                printf("skip L=%d %s: needs %.1f GiB\n", moduli, fast ? "fast" : "accu",
                       (s.scratch + s.keep_xr + s.keep_xl) / 1073741824.0);
                continue;
            }
            configs.push_back({moduli, fast != 0, s});
            scratch_bytes = std::max(scratch_bytes, s.scratch);
            keep_bytes = std::max({keep_bytes, s.keep_xr, s.keep_xl});
        }
    }
    if (scratch_bytes + 2 * keep_bytes + reserve > free_bytes) die("the selected configurations do not fit together");
    const size_t blocked_bytes = size_t(std::max(0, blocked_mb)) << 20;
    scratch_bytes = std::max(scratch_bytes, blocked_bytes);
    check(cudaMalloc(&d.work, scratch_bytes), "allocate the scratch area");
    check(cudaMalloc(&d.keep_xr, keep_bytes), "allocate the retained X (right)");
    check(cudaMalloc(&d.keep_xl, keep_bytes), "allocate the retained X (left)");

    const auto upload_h = [&](size_t t) {
        check(cudaMemcpy(d.H, H[t].data(), count * sizeof(double), cudaMemcpyHostToDevice), "upload H");
    };

    /* error indicator: Omega with entries in (-1, 1) and Z = X Omega */
    Monitor monitor;
    {
        const size_t        thin = n * size_t(monitor.b);
        const double        one = 1.0, zero = 0.0;
        std::vector<double> omega(thin);
        std::mt19937_64     engine(4301);
        std::uniform_real_distribution<double> uniform(-1.0, 1.0);
        for (double &value : omega) value = uniform(engine);
        for (double **thin_matrix : {&monitor.omega, &monitor.z, &monitor.hz, &monitor.v, &monitor.u, &monitor.uh})
            check(cudaMalloc(thin_matrix, thin * sizeof(double)), "allocate an indicator matrix");
        monitor.host_v.resize(thin); monitor.host_u.resize(thin); monitor.host_uh.resize(thin);
        check(cudaMemcpy(monitor.omega, omega.data(), thin * sizeof(double), cudaMemcpyHostToDevice), "upload Omega");
        check(cublasDgemm(d.handle, CUBLAS_OP_N, CUBLAS_OP_N, int(n), monitor.b, int(n), &one, d.X, int(n), monitor.omega,
                          int(n), &zero, monitor.z, int(n)), "Z = X Omega");
    }
    const auto add_eta = [](Json &j, const Eta &e) {
        j.num("eta_full_b4", e.full[0]).num("eta_full_b8", e.full[1]).num("eta_full_b16", e.full[2])
            .num("eta_herm_b4", e.herm[0]).num("eta_herm_b8", e.herm[1]).num("eta_herm_b16", e.herm[2])
            .num("eta_wall_b16", e.wall);
    };

    /* cuBLAS FP64 */
    std::vector<std::vector<double>> c64(T, std::vector<double>(count));
    std::vector<double>              fp64_wall[2];
    printf("\n%-5s %-5s %-22s %-8s %10s %10s %10s %10s %10s %10s %10s %9s %9s %9s\n", "L", "mode", "H", "path",
           "errB_rel", "errC_rel", "errC_max", "dC64_rel", "asymC_rel", "eta16", "eta16_herm", "bitsB", "bitsC",
           "bitsC_g2");
    for (size_t t = 0; t < T; ++t) {
        double wall[2];
        upload_h(t);
        fp64_pair(d, d.Bn, d.Cn, wall);
        for (int rep = 0; t + 1 == T && rep < reps; ++rep) {
            fp64_pair(d, d.Bn, d.Cn, wall);
            fp64_wall[0].push_back(wall[0]);
            fp64_wall[1].push_back(wall[1]);
        }
        const Error  eb = column_error(d.Bn, n, columns, refs[t].bh, refs[t].bl);
        const Error  ec = column_error(d.Cn, n, columns, refs[t].ch, refs[t].cl);
        const double norm_c = frobenius(d.Cn, count), asym = asymmetry(d.Cn, n) / norm_c;
        const Eta    eta = monitor_eta(d, monitor, d.Cn);
        check(cudaMemcpy(c64[t].data(), d.Cn, count * sizeof(double), cudaMemcpyDeviceToHost), "download C");
        printf("%-5s %-5s %-22s %-8s %10.3e %10.3e %10.3e %10s %10.3e %10.3e %10.3e\n", "-", "fp64", h_names[t].c_str(),
               "cublas", eb.relative, ec.relative, ec.max_abs, "-", asym, eta.full[2], eta.herm[2]);
        Json j;
        j.str("record", "error").str("label", label).integer("n", (long long)n).str("h", h_names[t])
            .str("path", "fp64").num("errB_rel", eb.relative).num("errB_max", eb.max_abs)
            .num("errC_rel", ec.relative).num("errC_max", ec.max_abs).num("asymC_rel", asym)
            .num("normC", norm_c);
        add_eta(j, eta);
        j.emit(json);
    }
    printf("fp64 wall: H X %.3f ms, X^T B %.3f ms\n", 1e3 * median(fp64_wall[0]), 1e3 * median(fp64_wall[1]));
    Json().str("record", "time").str("label", label).integer("n", (long long)n).str("path", "fp64")
        .num("wall_hx", median(fp64_wall[0])).num("wall_xtb", median(fp64_wall[1])).emit(json);

    /* GEMMul8 */
    for (const Config &config : configs) {
        const int   L = config.moduli;
        const bool  fast = config.fast;
        const char *mode = fast ? "fast" : "accu";

        /* prepare both retained representations of X with H[prep] */
        upload_h(size_t(prep));
        gemm_hx(d, L, fast, Reuse::off, d.Bn);
        gemm_xtb(d, L, fast, Reuse::off, d.Bn, d.Cn);
        gemm_hx(d, L, fast, Reuse::prepare, d.B);
        gemm_xtb(d, L, fast, Reuse::prepare, d.B, d.C);
        const long long prepare_bits_b = bit_mismatches(d.B, d.Bn, count), prepare_bits_c = bit_mismatches(d.C, d.Cn, count);

        for (size_t t = 0; t < T; ++t) {
            upload_h(t);
            gemm_hx(d, L, fast, Reuse::off, d.Bn);
            gemm_xtb(d, L, fast, Reuse::off, d.Bn, d.Cn);
            gemm_hx(d, L, fast, Reuse::use, d.B);
            gemm_xtb(d, L, fast, Reuse::use, d.B, d.C);
            gemm_xtb(d, L, fast, Reuse::use, d.Bn, d.R); /* second GEMM alone, on the normal B */

            const long long bits_b = bit_mismatches(d.B, d.Bn, count), bits_c = bit_mismatches(d.C, d.Cn, count);
            const long long bits_c_g2 = bit_mismatches(d.R, d.Cn, count);
            const double    max_b = max_abs_difference(d.B, d.Bn, count), max_c = max_abs_difference(d.C, d.Cn, count);

            check(cudaMemcpy(d.R, c64[t].data(), count * sizeof(double), cudaMemcpyHostToDevice), "upload C (FP64)");
            const double norm64 = frobenius(d.R, count);
            for (int path = 0; path < 2; ++path) {
                const double *B = path ? d.B : d.Bn, *C = path ? d.C : d.Cn;
                const Error   eb = column_error(B, n, columns, refs[t].bh, refs[t].bl);
                const Error   ec = column_error(C, n, columns, refs[t].ch, refs[t].cl);
                const double  d64 = frobenius_difference(C, d.R, count) / norm64;
                const double  d64_max = max_abs_difference(C, d.R, count);
                const double  asym = asymmetry(C, n) / norm64;
                const Eta     eta = monitor_eta(d, monitor, C);
                if (path == 0)
                    printf("%-5d %-5s %-22s %-8s %10.3e %10.3e %10.3e %10.3e %10.3e %10.3e %10.3e\n", L, mode,
                           h_names[t].c_str(), "normal", eb.relative, ec.relative, ec.max_abs, d64, asym, eta.full[2],
                           eta.herm[2]);
                else
                    printf("%-5d %-5s %-22s %-8s %10.3e %10.3e %10.3e %10.3e %10.3e %10.3e %10.3e %9lld %9lld %9lld\n",
                           L, mode, h_names[t].c_str(), "reuse", eb.relative, ec.relative, ec.max_abs, d64, asym,
                           eta.full[2], eta.herm[2], bits_b, bits_c, bits_c_g2);
                Json j;
                j.str("record", "error").str("label", label).integer("n", (long long)n).str("h", h_names[t])
                    .integer("L", L).integer("fast", fast).str("path", path ? "reuse" : "normal")
                    .num("errB_rel", eb.relative).num("errB_max", eb.max_abs).num("errC_rel", ec.relative)
                    .num("errC_max", ec.max_abs).num("dC64_rel", d64).num("dC64_max", d64_max).num("asymC_rel", asym);
                if (path)
                    j.str("prepared_with", h_names[size_t(prep)]).integer("bitsB", bits_b).integer("bitsC", bits_c)
                        .integer("bitsC_gemm2_only", bits_c_g2).num("maxdiffB", max_b).num("maxdiffC", max_c)
                        .integer("prepare_bitsB", prepare_bits_b).integer("prepare_bitsC", prepare_bits_c);
                add_eta(j, eta);
                j.emit(json);
            }

            if (blocked_bytes != 0) {
                set_blocking(d.handle, blocked_bytes);
                gemm_hx(d, L, fast, Reuse::off, d.B);
                gemm_xtb(d, L, fast, Reuse::off, d.B, d.C);
                set_blocking(d.handle, 0);
                const Error  eb = column_error(d.B, n, columns, refs[t].bh, refs[t].bl);
                const Error  ec = column_error(d.C, n, columns, refs[t].ch, refs[t].cl);
                const double d64 = frobenius_difference(d.C, d.R, count) / norm64;
                const double asym = asymmetry(d.C, n) / norm64;
                printf("%-5d %-5s %-22s %-8s %10.3e %10.3e %10.3e %10.3e %10.3e %9lld %9lld\n", L, mode,
                       h_names[t].c_str(), "blocked", eb.relative, ec.relative, ec.max_abs, d64, asym,
                       bit_mismatches(d.B, d.Bn, count), bit_mismatches(d.C, d.Cn, count));
                Json().str("record", "error").str("label", label).integer("n", (long long)n).str("h", h_names[t])
                    .integer("L", L).integer("fast", fast).str("path", "blocked").integer("blocked_mb", blocked_mb)
                    .num("errB_rel", eb.relative).num("errB_max", eb.max_abs).num("errC_rel", ec.relative)
                    .num("errC_max", ec.max_abs).num("dC64_rel", d64).num("asymC_rel", asym)
                    .integer("bitsB", bit_mismatches(d.B, d.Bn, count)).integer("bitsC", bit_mismatches(d.C, d.Cn, count))
                    .emit(json);
            }
        }

        /* wall time on the last Hamiltonian; X stays prepared with H[prep] */
        std::vector<double>              wall[8];
        std::vector<std::vector<double>> phases(8, std::vector<double>(4, 0.0));
        const auto                       record = [&](int slot, const Call &call) {
            wall[slot].push_back(call.wall);
            for (size_t p = 0; p < 4 && p < call.phases.size(); ++p) phases[size_t(slot)][p] += call.phases[p] / reps;
        };
        for (int rep = 0; rep < reps; ++rep) {
            record(0, gemm_hx(d, L, fast, Reuse::off, d.Bn));
            record(1, gemm_xtb(d, L, fast, Reuse::off, d.Bn, d.Cn));
            record(2, gemm_hx(d, L, fast, Reuse::use, d.B));
            record(3, gemm_xtb(d, L, fast, Reuse::use, d.B, d.C));
        }
        for (int rep = 0; rep < reps; ++rep) {
            record(4, gemm_hx(d, L, fast, Reuse::prepare, d.B));
            record(5, gemm_xtb(d, L, fast, Reuse::prepare, d.B, d.C));
        }
        for (int rep = 0; blocked_bytes != 0 && rep < reps; ++rep) {
            set_blocking(d.handle, blocked_bytes);
            record(6, gemm_hx(d, L, fast, Reuse::off, d.B));
            record(7, gemm_xtb(d, L, fast, Reuse::off, d.B, d.C));
            set_blocking(d.handle, 0);
        }
        const double normal = median(wall[0]) + median(wall[1]), reuse = median(wall[2]) + median(wall[3]);
        printf("%-5d %-5s time: normal %.3f + %.3f ms, reuse %.3f + %.3f ms (%.1f%% saved), prepare %.3f + %.3f ms; "
               "prepare-vs-normal bits %lld / %lld; keep %.2f + %.2f GiB, rest %.2f GiB, normal %.2f GiB\n",
               L, mode, 1e3 * median(wall[0]), 1e3 * median(wall[1]), 1e3 * median(wall[2]), 1e3 * median(wall[3]),
               100.0 * (normal - reuse) / normal, 1e3 * median(wall[4]), 1e3 * median(wall[5]), prepare_bits_b, prepare_bits_c,
               config.size.keep_xr / 1073741824.0, config.size.keep_xl / 1073741824.0,
               config.size.rest / 1073741824.0, config.size.normal / 1073741824.0);
        if (blocked_bytes != 0)
            printf("%-5d %-5s time: blocked (%d MiB cap) %.3f + %.3f ms\n", L, mode, blocked_mb, 1e3 * median(wall[6]),
                   1e3 * median(wall[7]));
        if (phases[0][0] > 0.0)
            printf("%-5d %-5s phases in ms (scaling, products, reduction, CRT) H X: normal %.3f %.3f %.3f %.3f, reuse "
                   "%.3f %.3f %.3f %.3f; X^T B: normal %.3f %.3f %.3f %.3f, reuse %.3f %.3f %.3f %.3f\n",
                   L, mode, 1e3 * phases[0][0], 1e3 * phases[0][1], 1e3 * phases[0][2], 1e3 * phases[0][3],
                   1e3 * phases[2][0], 1e3 * phases[2][1], 1e3 * phases[2][2], 1e3 * phases[2][3], 1e3 * phases[1][0],
                   1e3 * phases[1][1], 1e3 * phases[1][2], 1e3 * phases[1][3], 1e3 * phases[3][0], 1e3 * phases[3][1],
                   1e3 * phases[3][2], 1e3 * phases[3][3]);
        Json().str("record", "time").str("label", label).integer("n", (long long)n).integer("L", L)
            .integer("fast", fast).str("h", h_names[T - 1]).str("path", "gemmul8")
            .num("wall_hx_normal", median(wall[0])).num("wall_xtb_normal", median(wall[1]))
            .num("wall_hx_reuse", median(wall[2])).num("wall_xtb_reuse", median(wall[3]))
            .num("wall_hx_prepare", median(wall[4])).num("wall_xtb_prepare", median(wall[5]))
            .num("wall_hx_blocked", median(wall[6])).num("wall_xtb_blocked", median(wall[7])).integer("blocked_mb", blocked_mb)
            .list("phases_hx_blocked", phases[6]).list("phases_xtb_blocked", phases[7])
            .list("phases_hx_normal", phases[0]).list("phases_xtb_normal", phases[1])
            .list("phases_hx_reuse", phases[2]).list("phases_xtb_reuse", phases[3])
            .list("phases_hx_prepare", phases[4]).list("phases_xtb_prepare", phases[5])
            .integer("bytes_normal", (long long)config.size.normal).integer("bytes_keep_xr", (long long)config.size.keep_xr)
            .integer("bytes_keep_xl", (long long)config.size.keep_xl).integer("bytes_rest", (long long)config.size.rest)
            .emit(json);
    }

    if (json != nullptr) fclose(json);
    for (void *pointer : {(void *)d.X, (void *)d.H, (void *)d.B, (void *)d.C, (void *)d.Bn, (void *)d.Cn, (void *)d.R,
                          d.work, d.keep_xr, d.keep_xl, (void *)monitor.omega, (void *)monitor.z, (void *)monitor.hz,
                          (void *)monitor.v, (void *)monitor.u, (void *)monitor.uh})
        check(cudaFree(pointer), "cudaFree");
    check(cublasDestroy(d.handle), "cublasDestroy");
    return 0;
}

} // namespace

int main(int argc, char **argv)
{
    if (argc >= 2 && strcmp(argv[1], "capacity") == 0) return capacity(argc - 2, argv + 2);
    if (argc >= 2 && strcmp(argv[1], "selftest") == 0) return selftest();
    if (argc >= 2 && strcmp(argv[1], "run") == 0) return run(argc - 2, argv + 2);
    usage();
}

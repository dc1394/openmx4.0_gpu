/* Run with tests/run_gemmul8_forward_smoke.sh.  Checks the forward-transform
   entry points of the GEMMul8 bridge, openmx_gemmul8{D,Z}gemmFixed: the
   three precision stages, the reuse of the prepared X, and every change that
   must make the bridge prepare X again.  Accuracy is judged against a long
   double product on the CPU.  The state machine of the precision controller
   (openmx_gemmul8Adaptive*) is checked first; "check logic" runs only that
   part, which needs no GPU. */
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuComplex.h>
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <vector>

extern "C" cublasStatus_t openmx_gemmul8Dgemm(cublasHandle_t,cublasOperation_t,cublasOperation_t,int,int,int,const double*,const double*,int,const double*,int,const double*,double*,int);
extern "C" cublasStatus_t openmx_gemmul8Zgemm(cublasHandle_t,cublasOperation_t,cublasOperation_t,int,int,int,const cuDoubleComplex*,const cuDoubleComplex*,int,const cuDoubleComplex*,int,const cuDoubleComplex*,cuDoubleComplex*,int);
extern "C" cublasStatus_t openmx_gemmul8DgemmFixed(cublasHandle_t,int,cublasOperation_t,int,int,int,const double*,int,const double*,int,double*,int,int,unsigned long long);
extern "C" cublasStatus_t openmx_gemmul8ZgemmFixed(cublasHandle_t,int,cublasOperation_t,int,int,int,const cuDoubleComplex*,int,const cuDoubleComplex*,int,cuDoubleComplex*,int,int,unsigned long long);
extern "C" void openmx_gemmul8SetForwardStage(int,int,int,int,int);
extern "C" size_t openmx_gemmul8ForwardCounters(long long*,double*);
extern "C" void openmx_gemmul8ReleasePrepared();
extern "C" void openmx_gemmul8ReleaseWorkspaces();
extern "C" void openmx_gemmul8AdaptiveConfigure(int,const int*,const int*,const double*,const double*,int,int,int,int,int,int,int,int,int,double);
extern "C" int openmx_gemmul8AdaptiveEnabled();
extern "C" void openmx_gemmul8AdaptiveStart();
extern "C" int openmx_gemmul8AdaptiveBeginTrial();
extern "C" int openmx_gemmul8AdaptiveProbeColumns();
extern "C" void openmx_gemmul8AdaptiveReport(double,int);
extern "C" void openmx_gemmul8AdaptiveTrialStatus(int*,double*);
extern "C" void openmx_gemmul8AdaptiveReject();
extern "C" int openmx_gemmul8AdaptiveStopCheck(int);
extern "C" int openmx_gemmul8AdaptiveTakeHistoryReset();
extern "C" int openmx_gemmul8AdaptiveAfterMixing(double);
extern "C" void openmx_gemmul8AdaptiveDescribe(char*,int,long long*);

namespace {

enum { DEFAULT = 0, FP64 = 1, GEMMUL8 = 2 };

int failures = 0, checks = 0;

void expect(bool condition, const char *what, const char *type, const char *shape)
{
    ++checks;
    if (!condition) {
        ++failures;
        printf("FAIL (%s, %s): %s\n", type, shape, what);
    }
}

void check(cudaError_t status) { if (status != cudaSuccess) { fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(status)); exit(2); } }
void check(cublasStatus_t status) { if (status != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "cuBLAS: %d\n", int(status)); exit(2); } }

template <class T> T scalar(double r, double i)
{
    if constexpr (std::is_same_v<T, double>) return r;
    else return make_cuDoubleComplex(r, i);
}
template <class T> std::complex<long double> wide(T value)
{
    if constexpr (std::is_same_v<T, double>) return {value, 0};
    else return {value.x, value.y};
}

/* one shape of the forward transform: C = op(X) V (left) or C = V op(X) */
struct Shape {
    const char       *name;
    bool              left;
    cublasOperation_t op;
    int               m, n, k;
    int x_rows() const { return left ? (op == CUBLAS_OP_N ? m : k) : (op == CUBLAS_OP_N ? k : n); }
    int x_cols() const { return left ? (op == CUBLAS_OP_N ? k : m) : (op == CUBLAS_OP_N ? n : k); }
    int v_rows() const { return left ? k : m; }
    int v_cols() const { return left ? n : k; }
};

template <class T> std::vector<T> fill(size_t count, double seed)
{
    std::vector<T> a(count);
    for (size_t i = 0; i < count; ++i)
        a[i] = scalar<T>(sin(double(i) * 0.127 + seed) * exp(0.5 * cos(double(i) * 0.011 + seed)),
                         cos(double(i) * 0.213 + 2.0 * seed) * 0.4);
    return a;
}

template <class T> struct Bench {
    cublasHandle_t handle;
    Shape          shape;
    T             *dX = nullptr, *dV = nullptr, *dC = nullptr;

    Bench(cublasHandle_t h, const Shape &s) : handle(h), shape(s)
    {
        check(cudaMalloc(&dX, size_t(s.x_rows()) * s.x_cols() * sizeof(T)));
        check(cudaMalloc(&dV, size_t(s.v_rows()) * s.v_cols() * sizeof(T)));
        check(cudaMalloc(&dC, size_t(s.m) * s.n * sizeof(T)));
    }
    ~Bench() { cudaFree(dX); cudaFree(dV); cudaFree(dC); }

    void upload(const std::vector<T> &X, const std::vector<T> &V)
    {
        check(cudaMemcpy(dX, X.data(), X.size() * sizeof(T), cudaMemcpyHostToDevice));
        check(cudaMemcpy(dV, V.data(), V.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
    std::vector<T> download()
    {
        std::vector<T> c(size_t(shape.m) * shape.n);
        check(cudaGetLastError());
        check(cudaMemcpy(c.data(), dC, c.size() * sizeof(T), cudaMemcpyDeviceToHost));
        return c;
    }
    /* the entry point under test, with the stage that is currently set */
    std::vector<T> fixed(int id, unsigned long long version)
    {
        const Shape &s = shape;
        if constexpr (std::is_same_v<T, double>)
            check(openmx_gemmul8DgemmFixed(handle, s.left, s.op, s.m, s.n, s.k, dX, s.x_rows(), dV, s.v_rows(), dC, s.m, id, version));
        else
            check(openmx_gemmul8ZgemmFixed(handle, s.left, s.op, s.m, s.n, s.k, dX, s.x_rows(), dV, s.v_rows(), dC, s.m, id, version));
        return download();
    }
    /* the same product through the general entry points or plain cuBLAS */
    std::vector<T> general(bool gemmul8)
    {
        const Shape            &s = shape;
        const T                 one = scalar<T>(1, 0), zero = scalar<T>(0, 0);
        const cublasOperation_t ta = s.left ? s.op : CUBLAS_OP_N, tb = s.left ? CUBLAS_OP_N : s.op;
        const T                *A = s.left ? dX : dV, *B = s.left ? dV : dX;
        const int               lda = s.left ? s.x_rows() : s.v_rows(), ldb = s.left ? s.v_rows() : s.x_rows();
        if constexpr (std::is_same_v<T, double>) {
            const cublasOperation_t ra = ta == CUBLAS_OP_C ? CUBLAS_OP_T : ta, rb = tb == CUBLAS_OP_C ? CUBLAS_OP_T : tb;
            if (gemmul8) check(openmx_gemmul8Dgemm(handle, ta, tb, s.m, s.n, s.k, &one, A, lda, B, ldb, &zero, dC, s.m));
            else check(cublasDgemm(handle, ra, rb, s.m, s.n, s.k, &one, A, lda, B, ldb, &zero, dC, s.m));
        } else {
            if (gemmul8) check(openmx_gemmul8Zgemm(handle, ta, tb, s.m, s.n, s.k, &one, A, lda, B, ldb, &zero, dC, s.m));
            else check(cublasZgemm(handle, ta, tb, s.m, s.n, s.k, &one, A, lda, B, ldb, &zero, dC, s.m));
        }
        return download();
    }
    /* GEMMul8 without reuse, in the operation order of the reuse path */
    std::vector<T> fresh(int moduli, int fast)
    {
        openmx_gemmul8SetForwardStage(GEMMUL8, moduli, fast, 0, 1);
        return fixed(99, 0);
    }
};

template <class T> bool same(const std::vector<T> &a, const std::vector<T> &b)
{
    return a.size() == b.size() && memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0;
}

/* largest error relative to the largest exact entry */
template <class T> double error(const Shape &s, const std::vector<T> &X, const std::vector<T> &V, const std::vector<T> &C)
{
    long double worst = 0, scale = 0;
    for (int j = 0; j < s.n; ++j) for (int i = 0; i < s.m; ++i) {
        std::complex<long double> sum = 0;
        for (int p = 0; p < s.k; ++p) {
            std::complex<long double> x, v;
            if (s.left) {
                x = wide(X[s.op == CUBLAS_OP_N ? i + size_t(p) * s.x_rows() : p + size_t(i) * s.x_rows()]);
                v = wide(V[p + size_t(j) * s.v_rows()]);
            } else {
                x = wide(X[s.op == CUBLAS_OP_N ? p + size_t(j) * s.x_rows() : j + size_t(p) * s.x_rows()]);
                v = wide(V[i + size_t(p) * s.v_rows()]);
            }
            if (s.op == CUBLAS_OP_C) x = std::conj(x);
            sum += x * v;
        }
        worst = std::max(worst, std::abs(wide(C[i + size_t(j) * s.m]) - sum));
        scale = std::max(scale, std::abs(sum));
    }
    return double(worst / scale);
}

struct Counters {
    long long c[5];
    double    seconds[2];
    size_t    bytes;
    Counters() { bytes = openmx_gemmul8ForwardCounters(c, seconds); }
    long long prepared() const { return c[2]; }
    long long reused() const { return c[3]; }
    long long fallbacks() const { return c[4]; }
};

template <class T> void run(cublasHandle_t handle, const Shape &s, const char *type)
{
    Bench<T>             bench(handle, s);
    const size_t         x_count = size_t(s.x_rows()) * s.x_cols(), v_count = size_t(s.v_rows()) * s.v_cols();
    const std::vector<T> X1 = fill<T>(x_count, 0.3), X2 = fill<T>(x_count, 1.7);
    const std::vector<T> V1 = fill<T>(v_count, 0.9), V2 = fill<T>(v_count, 2.2);
    const int            L = 15;

    openmx_gemmul8ReleasePrepared();

    /* stages without reuse are the existing paths, bit for bit */
    bench.upload(X1, V1);
    openmx_gemmul8SetForwardStage(DEFAULT, 0, 0, 0, 0);
    expect(same(bench.fixed(0, 1), bench.general(true)), "default stage differs from openmx_gemmul8?gemm", type, s.name);
    openmx_gemmul8SetForwardStage(FP64, 0, 0, 0, 0);
    expect(same(bench.fixed(0, 1), bench.general(false)), "fp64 stage differs from cuBLAS", type, s.name);
    openmx_gemmul8SetForwardStage(GEMMUL8, L, 0, 0, 0);
    expect(error(s, X1, V1, bench.fixed(0, 1)) < 1e-12, "blocked GEMMul8 stage is inaccurate", type, s.name);

    /* fast scaling: reuse reproduces the path without reuse bit for bit */
    const std::vector<T> x1v1 = bench.fresh(L, 1);
    expect(error(s, X1, V1, x1v1) < 1e-12, "unblocked GEMMul8 stage is inaccurate", type, s.name);
    openmx_gemmul8SetForwardStage(GEMMUL8, L, 1, 1, 0);
    Counters before;
    expect(same(bench.fixed(0, 1), x1v1), "preparing call differs from the call without reuse", type, s.name);
    bench.upload(X1, V2);
    const std::vector<T> x1v2 = bench.fresh(L, 1);
    openmx_gemmul8SetForwardStage(GEMMUL8, L, 1, 1, 0);
    expect(same(bench.fixed(0, 1), x1v2), "reuse with another partner differs from the call without reuse", type, s.name);
    Counters after;
    expect(after.prepared() - before.prepared() == 1 && after.reused() - before.reused() == 1 && after.bytes > 0,
           "expected one preparation and one reuse", type, s.name);

    /* new contents behind the same pointer: only the version tells */
    bench.upload(X2, V2);
    const std::vector<T> x2v2 = bench.fresh(L, 1);
    openmx_gemmul8SetForwardStage(GEMMUL8, L, 1, 1, 0);
    expect(same(bench.fixed(0, 1), x1v2), "an unchanged version must keep using the retained X", type, s.name);
    expect(same(bench.fixed(0, 2), x2v2), "a new version must prepare X again", type, s.name);

    /* a second X under another id does not disturb the first */
    bench.upload(X1, V2);
    openmx_gemmul8SetForwardStage(GEMMUL8, L, 1, 1, 0);
    expect(same(bench.fixed(1, 7), x1v2), "second id: wrong result", type, s.name);
    bench.upload(X2, V2);
    before = Counters();
    expect(same(bench.fixed(0, 2), x2v2), "first id after the second: wrong result", type, s.name);
    after = Counters();
    expect(after.reused() - before.reused() == 1 && after.prepared() == before.prepared(),
           "first id should still be retained", type, s.name);

    /* a new moduli count prepares again */
    const std::vector<T> x2v2_12 = bench.fresh(12, 1);
    openmx_gemmul8SetForwardStage(GEMMUL8, 12, 1, 1, 0);
    before = Counters();
    expect(same(bench.fixed(0, 2), x2v2_12), "moduli change: wrong result", type, s.name);
    after = Counters();
    expect(after.prepared() - before.prepared() == 1, "moduli change should prepare again", type, s.name);

    /* accurate scaling: reuse changes the bits, not the validity */
    openmx_gemmul8SetForwardStage(GEMMUL8, L, 0, 1, 0);
    before = Counters();
    expect(error(s, X2, V2, bench.fixed(0, 2)) < 1e-12, "accurate scaling, preparing call", type, s.name);
    bench.upload(X2, V1);
    expect(error(s, X2, V1, bench.fixed(0, 2)) < 1e-11, "accurate scaling, reuse with another partner", type, s.name);
    after = Counters();
    expect(after.prepared() - before.prepared() == 1 && after.reused() - before.reused() == 1,
           "scaling-mode change should prepare once, then reuse", type, s.name);

    /* released forms are prepared again */
    openmx_gemmul8ReleasePrepared();
    openmx_gemmul8SetForwardStage(GEMMUL8, L, 1, 1, 0);
    bench.upload(X1, V1);
    expect(same(bench.fixed(0, 3), x1v1), "after a release: wrong result", type, s.name);

    /* no room for anything: the call falls back path by path (here down to
       cuBLAS FP64) without losing accuracy, and the fallback is counted */
    openmx_gemmul8ReleasePrepared();
    openmx_gemmul8ReleaseWorkspaces();
    setenv("OPENMX_GEMMUL8_MIN_FREE_AFTER_MB", "100000000", 1);
    openmx_gemmul8SetForwardStage(GEMMUL8, L, 1, 1, 0);
    before = Counters();
    expect(error(s, X1, V1, bench.fixed(0, 3)) < 1e-12, "capacity fallback is inaccurate", type, s.name);
    after = Counters();
    expect(after.fallbacks() > before.fallbacks() && after.bytes == 0, "capacity fallback was not counted", type, s.name);
    setenv("OPENMX_GEMMUL8_MIN_FREE_AFTER_MB", "0", 1);
    openmx_gemmul8ReleasePrepared();
}

/* the precision controller: three GEMMul8 stages, then FP64 */
void controller_logic()
{
    const char  *type = "controller", *shape = "logic";
    const int    moduli[] = {8, 10, 12}, fast[] = {1, 1, 0};
    const double promote[] = {1e-2, 1e-5, 0.0}, tolerance[] = {1e-5, 1e-7, 1e-9};
    char         text[64];
    long long    counters[5];
    int          rejected;
    double       eta;
    const auto   configure = [&](int nstage, int stall, int budget) {
        openmx_gemmul8AdaptiveConfigure(nstage, moduli, fast, promote, tolerance, 1, 1, 2, stall, budget, 2, 1, 8, 5, 1e-8);
        openmx_gemmul8AdaptiveStart();
    };
    const auto accept = [&](int stop, double residual) { /* one accepted iteration that does not end the SCF */
        const int may_stop = openmx_gemmul8AdaptiveStopCheck(stop);
        return may_stop ? -1 : openmx_gemmul8AdaptiveAfterMixing(residual);
    };

    configure(3, 0, 0);
    expect(openmx_gemmul8AdaptiveEnabled() == 1, "controller should be enabled", type, shape);
    expect(openmx_gemmul8AdaptiveBeginTrial() == 0 && openmx_gemmul8AdaptiveProbeColumns() == 8, "stage 0 starts with an indicator", type, shape);
    openmx_gemmul8AdaptiveReport(1e-7, 0);
    openmx_gemmul8AdaptiveTrialStatus(&rejected, &eta);
    expect(rejected == 0 && eta == 1e-7, "indicator within the tolerance is accepted", type, shape);
    openmx_gemmul8AdaptiveDescribe(text, sizeof(text), counters);
    expect(strcmp(text, "moduli=8 fast reuse") == 0 && counters[0] == 0 && counters[1] == 1, "description of stage 0", type, shape);
    expect(accept(0, 1.0) == 0, "no promotion above the threshold", type, shape);
    expect(openmx_gemmul8AdaptiveBeginTrial() == 0 && openmx_gemmul8AdaptiveProbeColumns() == 0, "no indicator between the scheduled ones", type, shape);
    expect(accept(0, 5e-3) == 0, "one iteration below the threshold does not promote", type, shape);
    openmx_gemmul8AdaptiveBeginTrial();
    expect(accept(0, 4e-3) == 1, "two iterations below the threshold promote", type, shape);

    /* stage 1: a rejected trial moves on and is probed again */
    expect(openmx_gemmul8AdaptiveBeginTrial() == 1 && openmx_gemmul8AdaptiveProbeColumns() == 8, "stage 1 starts with an indicator", type, shape);
    openmx_gemmul8AdaptiveReport(1e-9, 0);
    openmx_gemmul8AdaptiveReport(1e-6, 0); /* second spin: the worst counts */
    openmx_gemmul8AdaptiveTrialStatus(&rejected, &eta);
    expect(rejected == 1 && eta == 1e-6, "indicator above the tolerance rejects", type, shape);
    openmx_gemmul8AdaptiveReject();
    expect(openmx_gemmul8AdaptiveBeginTrial() == 2 && openmx_gemmul8AdaptiveProbeColumns() == 8, "rejection enters stage 2 with an indicator", type, shape);
    openmx_gemmul8AdaptiveTrialStatus(&rejected, &eta);
    expect(rejected == 0 && eta < 0, "a new trial starts clean", type, shape);
    openmx_gemmul8AdaptiveDescribe(text, sizeof(text), counters);
    expect(strcmp(text, "moduli=12 accurate reuse") == 0 && counters[2] == 1, "description of stage 2", type, shape);
    expect(accept(0, 1e-6) == 0 && accept(0, 1e-7) == 0 && accept(0, 1e-8) == 0, "the last GEMMul8 stage is not left by the residual", type, shape);
    for (int i = 0; i < 2; ++i) { openmx_gemmul8AdaptiveBeginTrial(); accept(0, 1e-8); }
    expect(openmx_gemmul8AdaptiveBeginTrial() == 2 && openmx_gemmul8AdaptiveProbeColumns() == 8, "indicator every fifth iteration of a stage", type, shape);
    openmx_gemmul8AdaptiveReport(-1.0, 1);
    openmx_gemmul8AdaptiveTrialStatus(&rejected, &eta);
    expect(rejected == 1, "a failed solve rejects", type, shape);
    openmx_gemmul8AdaptiveReject();

    /* FP64: only differences between FP64 iterations can end the SCF */
    expect(openmx_gemmul8AdaptiveBeginTrial() == 3 && openmx_gemmul8AdaptiveProbeColumns() == 0, "rejection of the last stage enters FP64", type, shape);
    openmx_gemmul8AdaptiveDescribe(text, sizeof(text), counters);
    expect(strcmp(text, "fp64") == 0, "description of the FP64 stage", type, shape);
    expect(openmx_gemmul8AdaptiveTakeHistoryReset() == 1 && openmx_gemmul8AdaptiveTakeHistoryReset() == 0, "the mixing restart is requested once", type, shape);
    expect(openmx_gemmul8AdaptiveStopCheck(1) == 0, "the first FP64 iteration cannot stop", type, shape);
    expect(openmx_gemmul8AdaptiveAfterMixing(1e-9) == 0, "FP64 is the last stage", type, shape);
    expect(openmx_gemmul8AdaptiveStopCheck(1) == 0, "one FP64 difference is not enough", type, shape);
    expect(openmx_gemmul8AdaptiveStopCheck(0) == 0 && openmx_gemmul8AdaptiveStopCheck(1) == 0, "an unmet condition restarts the count", type, shape);
    expect(openmx_gemmul8AdaptiveStopCheck(1) == 1, "two FP64 differences in a row stop", type, shape);

    /* the stop condition in a GEMMul8 stage switches to FP64 instead */
    configure(3, 0, 0);
    openmx_gemmul8AdaptiveBeginTrial();
    expect(openmx_gemmul8AdaptiveStopCheck(1) == 0 && openmx_gemmul8AdaptiveBeginTrial() == 3, "a GEMMul8 stage never ends the SCF", type, shape);
    expect(openmx_gemmul8AdaptiveStopCheck(1) == 0 && openmx_gemmul8AdaptiveStopCheck(1) == 0 && openmx_gemmul8AdaptiveStopCheck(1) == 1,
           "FP64 tail of three iterations", type, shape);

    /* iteration budget */
    configure(3, 0, 3);
    for (int i = 0; i < 2; ++i) { openmx_gemmul8AdaptiveBeginTrial(); accept(0, 1.0); }
    openmx_gemmul8AdaptiveBeginTrial();
    expect(openmx_gemmul8AdaptiveStopCheck(0) == 0 && openmx_gemmul8AdaptiveBeginTrial() == 3, "the budget ends the GEMMul8 stages", type, shape);

    /* stagnation: no improvement by 30 % for four iterations */
    configure(3, 4, 0);
    openmx_gemmul8AdaptiveBeginTrial();
    accept(0, 1.0);
    int promoted = 0, iterations = 0;
    while (!promoted && iterations < 10) { openmx_gemmul8AdaptiveBeginTrial(); promoted = accept(0, 0.9); ++iterations; }
    expect(promoted == 1 && iterations == 4 && openmx_gemmul8AdaptiveBeginTrial() == 1, "stagnation moves to the next stage", type, shape);

    /* a single stage is a fixed setting with the FP64 tail */
    configure(1, 0, 0);
    openmx_gemmul8AdaptiveBeginTrial();
    expect(accept(0, 1e-9) == 0 && openmx_gemmul8AdaptiveBeginTrial() == 0, "a single stage stays", type, shape);

    /* disabled: the SCF stops as usual */
    configure(0, 0, 0);
    expect(openmx_gemmul8AdaptiveEnabled() == 0 && openmx_gemmul8AdaptiveBeginTrial() == -1 &&
               openmx_gemmul8AdaptiveStopCheck(1) == 1 && openmx_gemmul8AdaptiveStopCheck(0) == 0 &&
               openmx_gemmul8AdaptiveProbeColumns() == 0 && openmx_gemmul8AdaptiveTakeHistoryReset() == 0,
           "a disabled controller must not interfere", type, shape);
    openmx_gemmul8SetForwardStage(DEFAULT, 0, 0, 0, 0);
}

} // namespace

int main(int argc, char **argv)
{
    controller_logic();
    if (argc >= 2 && strcmp(argv[1], "logic") == 0) {
        if (failures != 0) { printf("FAILED: %d of %d controller checks\n", failures, checks); return 1; }
        printf("PASS: %d checks of the precision controller\n", checks);
        return 0;
    }

    cublasHandle_t handle;
    check(cublasCreate(&handle));
    setenv("OPENMX_GEMMUL8_MIN_FREE_AFTER_MB", "0", 1);

    const Shape shapes[] = {
        {"H X: X right, N", false, CUBLAS_OP_N, 70, 50, 60},
        {"X^T B: X left, T", true, CUBLAS_OP_T, 70, 50, 60},
        {"X^H B: X left, C", true, CUBLAS_OP_C, 70, 50, 60},
        {"square, X right, N", false, CUBLAS_OP_N, 300, 300, 300},
        {"square, X left, C", true, CUBLAS_OP_C, 300, 300, 300},
    };
    for (const Shape &shape : shapes) {
        run<double>(handle, shape, "real");
        run<cuDoubleComplex>(handle, shape, "complex");
    }

    openmx_gemmul8ReleasePrepared();
    openmx_gemmul8ReleaseWorkspaces();
    check(cublasDestroy(handle));
    if (failures != 0) {
        printf("FAILED: %d of %d checks\n", failures, checks);
        return 1;
    }
    printf("PASS: %d checks of the precision controller, the forward-transform stages and the reuse of the prepared X, "
           "real and complex\n", checks);
    return 0;
}

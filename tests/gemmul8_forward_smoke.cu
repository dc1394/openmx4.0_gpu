/* Run with tests/run_gemmul8_forward_smoke.sh.  Checks the forward-transform
   entry points of the GEMMul8 bridge, openmx_gemmul8{D,Z}gemmFixed: the
   three fixed precision modes, the reuse of the prepared X, and every change
   that must make the bridge prepare X again.  Accuracy is judged against a long
   double product on the CPU. */
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
extern "C" cublasStatus_t openmx_gemmul8DgemmFixedOwned(const void*,cublasHandle_t,int,cublasOperation_t,int,int,int,const double*,int,const double*,int,double*,int,int,unsigned long long);
extern "C" void openmx_gemmul8ReleasePreparedOwner(const void*);
extern "C" void openmx_gemmul8ReleaseWorkspaces();
extern "C" void openmx_gemmul8TrimWorkspaces();

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
    expect(same(bench.fixed(0, 1), bench.general(true)), "capped default stage differs from openmx_gemmul8?gemm", type, s.name);
    openmx_gemmul8SetForwardStage(DEFAULT, 0, 0, 0, 1);
    expect(same(bench.fixed(0, 1), bench.general(true)), "default stage differs from openmx_gemmul8?gemm below the cap", type, s.name);
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

/* Above the workspace cap (256 MiB in this test) the default stage takes the
   full workspace, and that scratch can be returned without touching the
   capped workspace of the general entry points.  The memory tells the two
   paths apart: at this size the blocked product splits only the columns of
   C and agrees with the full-workspace one bit for bit. */
void above_the_cap(cublasHandle_t handle)
{
    const Shape   s = {"square above the cap, X right, N", false, CUBLAS_OP_N, 3000, 3000, 3000};
    const char   *type = "real";
    const size_t  count = size_t(s.m) * s.n, cap = size_t(256) << 20, slack = size_t(64) << 20;
    Bench<double> bench(handle, s);
    size_t        before = 0, now = 0, total = 0;
    const auto    used = [&]() {
        check(cudaMemGetInfo(&now, &total));
        return before > now ? before - now : size_t(0);
    };

    bench.upload(fill<double>(count, 0.3), fill<double>(count, 0.9));
    openmx_gemmul8SetForwardStage(GEMMUL8, 15, 0, 0, 1);
    const std::vector<double> full = bench.fixed(0, 1);
    openmx_gemmul8SetForwardStage(GEMMUL8, 15, 0, 0, 0);
    const std::vector<double> blocked = bench.fixed(0, 1);
    openmx_gemmul8ReleaseWorkspaces();
    check(cudaMemGetInfo(&before, &total));

    openmx_gemmul8SetForwardStage(DEFAULT, 0, 0, 0, 1);
    expect(same(bench.fixed(0, 1), full), "default stage does not take the full workspace above the cap", type, s.name);
    expect(used() > cap + slack, "no scratch above the cap after the default stage", type, s.name);
    openmx_gemmul8TrimWorkspaces();
    expect(used() < slack, "trim does not return the scratch", type, s.name);

    openmx_gemmul8SetForwardStage(DEFAULT, 0, 0, 0, 0);
    expect(same(bench.fixed(0, 1), blocked), "capped default stage differs from the blocked GEMMul8 stage", type, s.name);
    expect(same(bench.general(true), blocked), "openmx_gemmul8Dgemm differs from the blocked GEMMul8 stage", type, s.name);
    expect(cap - slack < used() && used() < cap + slack, "capped workspace of unexpected size", type, s.name);
    openmx_gemmul8TrimWorkspaces();
    expect(cap - slack < used(), "trim frees the capped workspace", type, s.name);
}

/* Equal local generations in interleaved calculations must never select
   each other's retained X, even when the device upload pointer is reused. */
void calculation_owners(cublasHandle_t handle)
{
    int first,second;
    for (bool left : {false,true}) {
        const Shape s = {"calculation ownership",left,left?CUBLAS_OP_T:CUBLAS_OP_N,50,50,50};
        Bench<double> bench(handle,s);
        const auto x1=fill<double>(2500,0.1),x2=fill<double>(2500,0.7),x3=fill<double>(2500,1.3);
        const auto v=fill<double>(2500,0.4);
        auto owned=[&](const void* owner) {
            check(openmx_gemmul8DgemmFixedOwned(owner,handle,left,s.op,50,50,50,
                bench.dX,50,bench.dV,50,bench.dC,50,0,1));return bench.download();
        };
        openmx_gemmul8ReleasePrepared();openmx_gemmul8SetForwardStage(GEMMUL8,15,1,1,1);
        bench.upload(x1,v);const auto a=owned(&first);
        bench.upload(x2,v);const auto b=owned(&second);
        expect(error(s,x1,v,a)<1e-12 && error(s,x2,v,b)<1e-12,"equal generations mixed owners","real",s.name);
        Counters before;
        expect(same(owned(&first),a),"interleaving lost first owner's prepared X","real",s.name);
        bench.upload(x3,v);const auto legacy=bench.fixed(0,1);
        expect(error(s,x3,v,legacy)<1e-12 && same(owned(&second),b),"legacy/caller namespaces mixed","real",s.name);
        Counters after;
        expect(after.reused()-before.reused()==2,"interleaving did not reuse the correct owners","real",s.name);
        openmx_gemmul8ReleasePreparedOwner(&first);openmx_gemmul8ReleasePreparedOwner(&first);
        before=Counters();
        expect(same(owned(&second),b),"selective drain invalidated second owner","real",s.name);
        const auto fresh=owned(&first);after=Counters();
        expect(error(s,x3,v,fresh)<1e-12 && after.prepared()-before.prepared()==1 &&
            after.reused()-before.reused()==1,"reused owner identity retained stale X","real",s.name);
    }
    openmx_gemmul8ReleasePrepared();
}

} // namespace

int main()
{
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

    above_the_cap(handle);
    calculation_owners(handle);

    openmx_gemmul8ReleasePrepared();
    openmx_gemmul8ReleaseWorkspaces();
    check(cublasDestroy(handle));
    if (failures != 0) {
        printf("FAILED: %d of %d checks\n", failures, checks);
        return 1;
    }
    printf("PASS: %d checks of the fixed forward-transform modes and the reuse of the prepared X, "
           "real and complex\n", checks);
    return 0;
}

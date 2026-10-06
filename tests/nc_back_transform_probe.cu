/* Back transform of the non-collinear cluster solver, in isolation.

   The old form is one complex GEMM against Ss2 = diag(S, S),

       C = V(:, 0:maxn)^T Ss2^T              (openmx_gemmul8Zgemm, T T),

   the real form (ClusterNonCol_BackTransform in Cluster_DFT_NonCol.c) the
   transpose vt = V(:, 0:maxn)^T (cublasZgeam) and, per spin half, one real
   GEMM of the real view of vt with S^T (openmx_gemmul8Dgemm, N T), and the
   halves form, per spin half, one complex GEMM of V's rows of that half
   with a complex copy of S (openmx_gemmul8Zgemm, T T, maxn x n x n): half
   the work of the old form, no transpose.  V (n2 x n2 complex, n2 = 2n)
   and S (n x n real) are random; the results are compared with the old
   form and all forms timed: the median of REPS calls after one warm-up, the
   device synchronized around each call.  With --release the bridge's
   workspaces are released before every call, as the solver does once per
   SCF step, so the timed call allocates them again.

     nc_back_transform_probe <n> <maxn> [reps] [--fp64] [--release]

   --fp64 sends both forms to plain cuBLAS FP64 (scf.gemmul8.enable off);
   otherwise the GEMMul8 settings come from the bridge's environment
   variables (OPENMX_GEMMUL8_NUM_MOD_D/Z, OPENMX_GEMMUL8_FASTMODE_D/Z, ...). */
#include <cublas_v2.h>
#include <cuComplex.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" {
cublasStatus_t openmx_gemmul8Dgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m,
                                   int n, int k, const double *alpha, const double *A, int lda, const double *B,
                                   int ldb, const double *beta, double *C, int ldc);
cublasStatus_t openmx_gemmul8Zgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m,
                                   int n, int k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda,
                                   const cuDoubleComplex *B, int ldb, const cuDoubleComplex *beta, cuDoubleComplex *C,
                                   int ldc);
void openmx_gemmul8ReleaseWorkspaces(void);
void openmx_gemmul8SetEnabled(int enabled);
}

namespace {

void check(cudaError_t status, const char *what)
{
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(status));
        std::exit(1);
    }
}

void check(cublasStatus_t status, const char *what)
{
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "%s: cuBLAS status %d\n", what, static_cast<int>(status));
        std::exit(1);
    }
}

double uniform(uint64_t &state)
{
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    return 2.0 * (static_cast<double>((state * 0x2545F4914F6CDD1DULL) >> 11) / 9007199254740992.0) - 1.0;
}

template <class Body>
double median_ms(int reps, bool release, Body body)
{
    std::vector<double> times;

    if (release) openmx_gemmul8ReleaseWorkspaces();
    body(); /* warm-up */
    for (int r = 0; r < reps; r++) {
        if (release) openmx_gemmul8ReleaseWorkspaces();
        check(cudaDeviceSynchronize(), "synchronize");
        auto t0 = std::chrono::steady_clock::now();
        body();
        check(cudaDeviceSynchronize(), "synchronize");
        times.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}

} // namespace

int main(int argc, char **argv)
{
    int positional[3] = {0, 0, 5};
    int count = 0;
    bool fp64 = false, release = false;

    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--fp64") == 0) fp64 = true;
        else if (std::strcmp(argv[i], "--release") == 0) release = true;
        else if (count < 3) positional[count++] = std::atoi(argv[i]);
    }
    const int n = positional[0], maxn = positional[1], reps = std::max(1, positional[2]);
    const int n2 = 2 * n;
    if (count < 2 || n < 1 || maxn < 1 || n2 < maxn) {
        std::fprintf(stderr, "usage: %s <n> <maxn> [reps] [--fp64] [--release]   (maxn <= 2n)\n", argv[0]);
        return 2;
    }
    if (fp64) openmx_gemmul8SetEnabled(0);

    const size_t nn = static_cast<size_t>(n) * n, n2n2 = static_cast<size_t>(n2) * n2;
    std::vector<double> h_s(nn);
    std::vector<cuDoubleComplex> h_v(n2n2), h_ss2(n2n2, make_cuDoubleComplex(0.0, 0.0)), h_sc(nn);
    uint64_t state = 0x9E3779B97F4A7C15ULL ^ (static_cast<uint64_t>(n) << 20) ^ static_cast<uint64_t>(maxn);

    for (auto &x : h_s) x = uniform(state);
    for (auto &z : h_v) {
        const double re = uniform(state);
        z = make_cuDoubleComplex(re, uniform(state));
    }
    for (int col = 0; col < n; col++) {
        for (int row = 0; row < n; row++) {
            const cuDoubleComplex s = make_cuDoubleComplex(h_s[row + static_cast<size_t>(col) * n], 0.0);
            h_ss2[row + static_cast<size_t>(col) * n2] = s;
            h_ss2[(row + n) + static_cast<size_t>(col + n) * n2] = s;
            h_sc[row + static_cast<size_t>(col) * n] = s;
        }
    }

    double *d_s = nullptr;
    cuDoubleComplex *d_v = nullptr, *d_ss2 = nullptr, *d_old = nullptr, *d_new = nullptr, *d_vt = nullptr;
    cuDoubleComplex *d_sc = nullptr, *d_halves = nullptr;
    check(cudaMalloc(&d_s, nn * sizeof(double)), "cudaMalloc S");
    check(cudaMalloc(&d_v, n2n2 * sizeof(cuDoubleComplex)), "cudaMalloc V");
    check(cudaMalloc(&d_ss2, n2n2 * sizeof(cuDoubleComplex)), "cudaMalloc Ss2");
    check(cudaMalloc(&d_old, n2n2 * sizeof(cuDoubleComplex)), "cudaMalloc C old");
    check(cudaMalloc(&d_new, n2n2 * sizeof(cuDoubleComplex)), "cudaMalloc C new");
    check(cudaMalloc(&d_vt, static_cast<size_t>(maxn) * n2 * sizeof(cuDoubleComplex)), "cudaMalloc vt");
    check(cudaMalloc(&d_sc, nn * sizeof(cuDoubleComplex)), "cudaMalloc complex S");
    check(cudaMalloc(&d_halves, n2n2 * sizeof(cuDoubleComplex)), "cudaMalloc C halves");
    check(cudaMemcpy(d_sc, h_sc.data(), nn * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice), "upload complex S");
    check(cudaMemset(d_halves, 0, n2n2 * sizeof(cuDoubleComplex)), "clear C halves");
    check(cudaMemcpy(d_s, h_s.data(), nn * sizeof(double), cudaMemcpyHostToDevice), "upload S");
    check(cudaMemcpy(d_v, h_v.data(), n2n2 * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice), "upload V");
    check(cudaMemcpy(d_ss2, h_ss2.data(), n2n2 * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice), "upload Ss2");
    check(cudaMemset(d_old, 0, n2n2 * sizeof(cuDoubleComplex)), "clear C old");
    check(cudaMemset(d_new, 0, n2n2 * sizeof(cuDoubleComplex)), "clear C new");

    cublasHandle_t handle = nullptr;
    check(cublasCreate(&handle), "cublasCreate");
    const cuDoubleComplex one = make_cuDoubleComplex(1.0, 0.0), zero = make_cuDoubleComplex(0.0, 0.0);
    const double alpha = 1.0, beta = 0.0;

    auto old_form = [&] {
        check(openmx_gemmul8Zgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, maxn, n2, n2, &one, d_v, n2, d_ss2, n2, &zero,
                                  d_old, n2), "old back transform");
    };
    auto transpose = [&] {
        check(cublasZgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, maxn, n2, &one, d_v, n2, &zero, d_v, n2, d_vt, maxn),
              "transpose");
    };
    auto real_gemms = [&] {
        for (int half = 0; half < 2; half++) {
            const size_t col = static_cast<size_t>(half) * n;
            check(openmx_gemmul8Dgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, 2 * maxn, n, n, &alpha,
                                      reinterpret_cast<const double *>(d_vt) + 2 * static_cast<size_t>(maxn) * col,
                                      2 * maxn, d_s, n, &beta,
                                      reinterpret_cast<double *>(d_new) + 2 * static_cast<size_t>(n2) * col, 2 * n2),
                  "new back transform");
        }
    };

    auto halves_form = [&] {
        for (int half = 0; half < 2; half++) {
            const size_t row = static_cast<size_t>(half) * n;
            check(openmx_gemmul8Zgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, maxn, n, n, &one, d_v + row, n2, d_sc, n,
                                      &zero, d_halves + row * n2, n2), "halves back transform");
        }
    };

    const double t_old = median_ms(reps, release, old_form);
    const double t_geam = median_ms(reps, false, transpose);
    const double t_new = median_ms(reps, release, [&] { transpose(); real_gemms(); });
    const double t_halves = median_ms(reps, release, halves_form);

    std::vector<cuDoubleComplex> c_old(n2n2), c_new(n2n2), c_halves(n2n2);
    check(cudaMemcpy(c_old.data(), d_old, n2n2 * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost), "download old");
    check(cudaMemcpy(c_new.data(), d_new, n2n2 * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost), "download new");
    check(cudaMemcpy(c_halves.data(), d_halves, n2n2 * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost),
          "download halves");
    double max_ref = 0.0, diff_new = 0.0, diff_halves = 0.0;
    for (int col = 0; col < n2; col++) {
        for (int row = 0; row < maxn; row++) {
            const size_t at = row + static_cast<size_t>(col) * n2;
            const cuDoubleComplex a = c_old[at], b = c_new[at], c = c_halves[at];
            max_ref = std::max(max_ref, std::hypot(cuCreal(a), cuCimag(a)));
            diff_new = std::max(diff_new, std::hypot(cuCreal(a) - cuCreal(b), cuCimag(a) - cuCimag(b)));
            diff_halves = std::max(diff_halves, std::hypot(cuCreal(a) - cuCreal(c), cuCimag(a) - cuCimag(c)));
        }
    }
    const double scale = max_ref > 0.0 ? 1.0 / max_ref : 1.0;
    const double gflop_old = 8.0 * maxn * static_cast<double>(n2) * n2 * 1e-9;
    const double gflop_new = 2.0 * 2.0 * (2.0 * maxn) * static_cast<double>(n) * n * 1e-9;

    std::printf("n=%d n2=%d maxn=%d %s%s old_ms=%.3f real_ms=%.3f (transpose_ms=%.3f) halves_ms=%.3f "
                "old_GFlop=%.1f real_GFlop=%.1f halves_GFlop=%.1f diff_real=%.2e diff_halves=%.2e\n",
                n, n2, maxn, fp64 ? "fp64" : "gemmul8", release ? " release" : "", t_old, t_new, t_geam, t_halves,
                gflop_old, gflop_new, gflop_old / 2.0, diff_new * scale, diff_halves * scale);

    cublasDestroy(handle);
    cudaFree(d_s);
    cudaFree(d_v);
    cudaFree(d_ss2);
    cudaFree(d_old);
    cudaFree(d_new);
    cudaFree(d_vt);
    cudaFree(d_sc);
    cudaFree(d_halves);
    return 0;
}

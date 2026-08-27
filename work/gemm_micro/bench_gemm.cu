/* bench_gemm.cu: M2 single-GEMM microbenchmark for the GEMMul8 campaign.
 *
 * For every non-comment line of the shapes file
 *     label  type(d|z)  opA(N|T|C)  opB(N|T|C)  m  n  k
 * this times, on the current GPU, back-to-back (ABAB per rep):
 *     cublasDgemm/cublasZgemm            (FP64 baseline)
 *     gemmul8::gemm<*, Backend::INT8>    (num_moduli, fastmode=false)
 * exactly as source/gemmul8_bridge.cu issues them from the dense solvers
 * (alpha=1, beta=0, ld = natural dims, one shared device workspace of
 * gemmul8::workSize bytes), and prints one CSV row per label to stdout.
 *
 * Usage: bench_gemm shapes.txt [reps=7] [num_moduli=15]
 * Build: ./build.sh (links the campaign's libgemmul8.a; same nvcc flags as
 *        source/Makefile's NVCC_GEMMUL8_FLAGS).
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "gemmul8.hpp"

#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        cudaError_t err_ = (call);                                                       \
        if (err_ != cudaSuccess) {                                                       \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_),        \
                    __FILE__, __LINE__);                                                 \
            exit(1);                                                                     \
        }                                                                                \
    } while (0)

#define CUBLAS_CHECK(call)                                                               \
    do {                                                                                 \
        cublasStatus_t st_ = (call);                                                     \
        if (st_ != CUBLAS_STATUS_SUCCESS) {                                              \
            fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)st_, __FILE__, __LINE__); \
            exit(1);                                                                     \
        }                                                                                \
    } while (0)

static double median_of(std::vector<double> v)
{
    std::sort(v.begin(), v.end());
    size_t h = v.size() / 2;
    return (v.size() % 2) ? v[h] : 0.5 * (v[h - 1] + v[h]);
}

static cublasOperation_t parse_op(const char *s)
{
    switch (s[0]) {
    case 'N': case 'n': return CUBLAS_OP_N;
    case 'T': case 't': return CUBLAS_OP_T;
    case 'C': case 'c': return CUBLAS_OP_C;
    }
    fprintf(stderr, "bad op '%s'\n", s);
    exit(1);
}

static void fill_random(double *host, size_t count)
{
    for (size_t i = 0; i < count; i++) {
        host[i] = ((double)rand() / (double)RAND_MAX) - 0.5;
    }
}

/* one timed call, device-synchronized via events on the default stream */
template <typename F>
static double timed_ms(F &&fn)
{
    cudaEvent_t ev0, ev1;
    float       ms = 0.0f;

    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(ev0));
    fn();
    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));
    return (double)ms;
}

template <bool IS_COMPLEX>
static void bench_one(cublasHandle_t handle, const char *label, cublasOperation_t opA, cublasOperation_t opB,
                      size_t m, size_t n, size_t k, int reps, int num_moduli)
{
    typedef typename std::conditional<IS_COMPLEX, cuDoubleComplex, double>::type T;

    const size_t rowsA = (opA == CUBLAS_OP_N) ? m : k;
    const size_t colsA = (opA == CUBLAS_OP_N) ? k : m;
    const size_t rowsB = (opB == CUBLAS_OP_N) ? k : n;
    const size_t colsB = (opB == CUBLAS_OP_N) ? n : k;
    const size_t lda = rowsA, ldb = rowsB, ldc = m;

    const size_t cntA = rowsA * colsA, cntB = rowsB * colsB, cntC = m * n;

    T *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, sizeof(T) * cntA));
    CUDA_CHECK(cudaMalloc(&dB, sizeof(T) * cntB));
    CUDA_CHECK(cudaMalloc(&dC, sizeof(T) * cntC));

    {   /* deterministic host fill, then upload */
        std::vector<double> host(std::max(cntA, cntB) * (IS_COMPLEX ? 2 : 1));
        srand(12345);
        fill_random(host.data(), cntA * (IS_COMPLEX ? 2 : 1));
        CUDA_CHECK(cudaMemcpy(dA, host.data(), sizeof(T) * cntA, cudaMemcpyHostToDevice));
        fill_random(host.data(), cntB * (IS_COMPLEX ? 2 : 1));
        CUDA_CHECK(cudaMemcpy(dB, host.data(), sizeof(T) * cntB, cudaMemcpyHostToDevice));
    }

    const size_t ws_bytes = gemmul8::workSize<IS_COMPLEX, gemmul8::Backend::INT8>(m, n, k, (unsigned)num_moduli);
    void        *dW       = nullptr;
    CUDA_CHECK(cudaMalloc(&dW, ws_bytes));

    T alpha, beta;
    if constexpr (IS_COMPLEX) { alpha = make_cuDoubleComplex(1.0, 0.0); beta = make_cuDoubleComplex(0.0, 0.0); }
    else                      { alpha = 1.0; beta = 0.0; }

    auto run_cublas = [&]() {
        if constexpr (IS_COMPLEX)
            CUBLAS_CHECK(cublasZgemm(handle, opA, opB, (int)m, (int)n, (int)k, &alpha, dA, (int)lda, dB, (int)ldb,
                                     &beta, dC, (int)ldc));
        else
            CUBLAS_CHECK(cublasDgemm(handle, opA, opB, (int)m, (int)n, (int)k, &alpha, dA, (int)lda, dB, (int)ldb,
                                     &beta, dC, (int)ldc));
    };
    auto run_g8 = [&]() {
        (void)gemmul8::gemm<T, gemmul8::Backend::INT8>(handle, opA, opB, m, n, k, &alpha, dA, lda, dB, ldb, &beta,
                                                       dC, ldc, num_moduli, false, dW);
    };

    /* warm-up both paths */
    run_cublas();
    run_g8();
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> t_cublas, t_g8;
    for (int r = 0; r < reps; r++) {          /* ABAB: equal thermal footing */
        t_cublas.push_back(timed_ms(run_cublas));
        t_g8.push_back(timed_ms(run_g8));
    }

    /* accuracy of the emulated GEMM against the FP64 result (same inputs) */
    double relerr = 0.0;
    {
        std::vector<double> ref(cntC * (IS_COMPLEX ? 2 : 1)), emu(cntC * (IS_COMPLEX ? 2 : 1));
        run_cublas();
        CUDA_CHECK(cudaMemcpy(ref.data(), dC, sizeof(T) * cntC, cudaMemcpyDeviceToHost));
        run_g8();
        CUDA_CHECK(cudaMemcpy(emu.data(), dC, sizeof(T) * cntC, cudaMemcpyDeviceToHost));
        double maxref = 0.0, maxdiff = 0.0;
        for (size_t i = 0; i < ref.size(); i++) {
            maxref  = std::max(maxref, std::fabs(ref[i]));
            maxdiff = std::max(maxdiff, std::fabs(ref[i] - emu[i]));
        }
        relerr = (maxref > 0.0) ? maxdiff / maxref : 0.0;
    }

    const double flops = (IS_COMPLEX ? 8.0 : 2.0) * (double)m * (double)n * (double)k;
    const double c_med = median_of(t_cublas), g_med = median_of(t_g8);
    const double c_min = *std::min_element(t_cublas.begin(), t_cublas.end());
    const double g_min = *std::min_element(t_g8.begin(), t_g8.end());

    printf("%s,%c,%c,%c,%zu,%zu,%zu,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.1f,%.1f,%.1f,%.3e\n", label,
           IS_COMPLEX ? 'z' : 'd', opA == CUBLAS_OP_N ? 'N' : (opA == CUBLAS_OP_T ? 'T' : 'C'),
           opB == CUBLAS_OP_N ? 'N' : (opB == CUBLAS_OP_T ? 'T' : 'C'), m, n, k, reps, num_moduli, c_med, c_min,
           g_med, g_min, c_med / g_med, flops / c_med * 1e-6, flops / g_med * 1e-6,
           (double)ws_bytes / (1024.0 * 1024.0), relerr);
    fflush(stdout);

    CUDA_CHECK(cudaFree(dW));
    CUDA_CHECK(cudaFree(dC));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dA));
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: %s shapes.txt [reps=7] [num_moduli=15]\n", argv[0]);
        return 1;
    }
    const int reps       = (argc > 2) ? atoi(argv[2]) : 7;
    const int num_moduli = (argc > 3) ? atoi(argv[3]) : 15;

    FILE *fp = fopen(argv[1], "r");
    if (!fp) {
        fprintf(stderr, "cannot open %s\n", argv[1]);
        return 1;
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    fprintf(stderr, "# GPU: %s (cc %d.%d), reps=%d num_moduli=%d fastmode=false\n", prop.name, prop.major, prop.minor,
            reps, num_moduli);

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    printf("label,type,opA,opB,m,n,k,reps,num_moduli,cublas_ms_med,cublas_ms_min,g8_ms_med,g8_ms_min,"
           "s_gemm_med,cublas_gflops,g8_gflops,ws_MiB,relerr\n");

    char line[512];
    while (fgets(line, sizeof(line), fp)) {
        char label[128], type[8], a[8], b[8];
        long m, n, k;
        if (line[0] == '#' || line[0] == '\n') continue;
        if (sscanf(line, "%127s %7s %7s %7s %ld %ld %ld", label, type, a, b, &m, &n, &k) != 7) {
            fprintf(stderr, "skipping malformed line: %s", line);
            continue;
        }
        if (type[0] == 'z' || type[0] == 'Z')
            bench_one<true>(handle, label, parse_op(a), parse_op(b), (size_t)m, (size_t)n, (size_t)k, reps, num_moduli);
        else
            bench_one<false>(handle, label, parse_op(a), parse_op(b), (size_t)m, (size_t)n, (size_t)k, reps, num_moduli);
    }

    fclose(fp);
    CUBLAS_CHECK(cublasDestroy(handle));
    return 0;
}

/* Production CUDA quadrature with full atom rows and bounded overlap tiles.
   The oracle is independent long-double arithmetic.  In particular, a
   chunk's local orbital row must never replace the original density index. */
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
extern "C" void openmx_gpu_test_use_visible_pool(void);
extern "C" void openmx_gpu_test_worker_failures(void);
extern "C" int Set_Hamiltonian_Cuda_MatrixElements(
    int, int, std::size_t, double, int, int, const int *, const int *, const int *,
    const int *, const int *, const std::size_t *, const std::size_t *,
    const std::size_t *, const std::size_t *, const float *, const float *,
    const double *, double *, int);

extern "C" int Set_Hamiltonian_Cuda_Density(
    int, int, int, int, const int *, const int *, const int *,
    const std::size_t *, const std::size_t *, const std::size_t *, const std::size_t *,
    const std::size_t *, const int *, const float *, const float *,
    const float *, const float *, double *, std::size_t, int);
extern "C" int Set_Hamiltonian_Cuda_Density_FP64(
    int, int, int, int, const int *, const int *, const int *,
    const std::size_t *, const std::size_t *, const std::size_t *, const std::size_t *,
    const std::size_t *, const int *, const float *, const float *,
    const double *, double *, std::size_t, int);
extern "C" int Set_Hamiltonian_Cuda_Density_Range(
    int, int, int, int, const int *, const int *, const int *,
    const std::size_t *, const std::size_t *, const std::size_t *, const std::size_t *,
    const std::size_t *, const int *, const float *, const float *,
    const float *, const float *, double *, std::size_t, int, std::size_t, std::size_t);
extern "C" int Set_Hamiltonian_Cuda_Density_FP64_Range(
    int, int, int, int, const int *, const int *, const int *,
    const std::size_t *, const std::size_t *, const std::size_t *, const std::size_t *,
    const std::size_t *, const int *, const float *, const float *,
    const double *, double *, std::size_t, int, std::size_t, std::size_t);
extern "C" void *Set_Hamiltonian_Cuda_StreamCreate(std::size_t, std::size_t, const double *);
extern "C" void Set_Hamiltonian_Cuda_StreamDestroy(void *);
extern "C" int Set_Hamiltonian_Cuda_StreamRun(
    void *, int, int, std::size_t, double, int, int,
    std::size_t, std::size_t, std::size_t, std::size_t,
    const int *, const int *, const int *, const int *, const int *,
    const std::size_t *, const std::size_t *, const std::size_t *, const std::size_t *,
    const float *, const float *, double *, int);

static void check(cudaError_t status)
{
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(status));
        std::exit(2);
    }
}

template <class T> struct Device {
    T *p;
    explicit Device(const std::vector<T> &v) {
        check(cudaMalloc(reinterpret_cast<void **>(&p), v.size() * sizeof(T)));
        check(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
    ~Device() { cudaFree(p); }
    Device(const Device &) = delete;
};

static double sample(unsigned x)
{
    x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15;
    x *= 0x846ca68bU; x ^= x >> 16;
    return (static_cast<double>(x & 65535U) - 32768.0) / 65536.0;
}

static double compare(const char *label, const std::vector<double> &actual,
                      const std::vector<long double> &expected)
{
    double worst = 0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const long double error = std::fabs(actual[i] - expected[i]) /
                                  std::max(1.0L, std::fabs(expected[i]));
        if (!std::isfinite(actual[i]) || error > 2e-12L) {
            std::fprintf(stderr, "%s[%zu]: %.17g != %.21Lg (%.3Le)\n",
                         label, i, actual[i], expected[i], error);
            std::exit(1);
        }
        worst = std::max(worst, static_cast<double>(error));
    }
    return worst;
}

static double exercise(int no0, int no1, int spins, int chunk, bool compact)
{
    constexpr int points = 273, rows = 719, potential_rows = 83;
    constexpr std::size_t noff = 3, aoff = 5, boff = 7, hoff = 9, outbase = 11;
    constexpr std::size_t outcount = outbase + rows + 13;
    constexpr double volume = 0.071238947;
    const int matrix_count = spins * no0 * no1;
    std::vector<int> original_nc(points), mn(points);
    std::vector<float> a(aoff + rows * no0 + 17, 111.0f);
    std::vector<float> b(boff + points * no1 + 19, 222.0f);
    std::vector<double> dm(hoff + matrix_count + 23, 333.0);
    std::vector<float> dmh(dm.size()), dml(dm.size());
    std::vector<double> potential(spins * potential_rows);
    std::vector<double> density(spins * outcount);
    std::vector<double> matrix(hoff + matrix_count + 23, 999.0);
    for (int g = 0; g < points; ++g) {
        /* Repeated, nonmonotonic indices, many larger than the compact tile. */
        original_nc[g] = ((g / 2) * 37 + 317) % rows;
        mn[g] = (g * 17 + 5) % potential_rows;
        for (int j = 0; j < no1; ++j) b[boff + g * no1 + j] = sample(g * 71 + j + 131);
    }
    for (int r = 0; r < rows; ++r)
        for (int i = 0; i < no0; ++i) a[aoff + r * no0 + i] = sample(r * 43 + i + 97);
    for (int e = 0; e < matrix_count; ++e) {
        dm[hoff + e] = sample(e + 104729) * 0.371;
        matrix[hoff + e] = sample(e + 7919) * 0.17;
    }
    for (std::size_t i = 0; i < dm.size(); ++i) {
        dmh[i] = static_cast<float>(dm[i]);
        dml[i] = static_cast<float>(dm[i] - static_cast<double>(dmh[i]));
    }
    for (std::size_t i = 0; i < potential.size(); ++i) potential[i] = sample(i + 271) * 0.731;
    for (std::size_t i = 0; i < density.size(); ++i) density[i] = 0.17 + sample(i + 83);
    std::vector<long double> expected_density(density.begin(), density.end());
    std::vector<long double> expected_matrix(matrix.begin(), matrix.end());
    for (int s = 0; s < spins; ++s)
        for (int g = 0; g < points; ++g) {
            long double rho = 0;
            for (int i = 0; i < no0; ++i)
                for (int j = 0; j < no1; ++j) {
                    const auto e = hoff + (s * no0 + i) * no1 + j;
                    const long double ab = static_cast<long double>(a[aoff + original_nc[g] * no0 + i]) *
                                           b[boff + g * no1 + j];
                    rho += dm[e] * ab;
                    expected_matrix[e] += volume * static_cast<long double>(potential[s * potential_rows + mn[g]]) * ab;
                }
            expected_density[s * outcount + outbase + original_nc[g]] += rho;
        }
    Device<double> d_density(density), d_density64(density), d_dm(dm);
    Device<float> d_dmh(dmh), d_dml(dml);
    Device<int> d_no0(std::vector<int>{no0}), d_no1(std::vector<int>{no1});
    Device<std::size_t> d_hoff(std::vector<std::size_t>{hoff});
    Device<std::size_t> d_noff(std::vector<std::size_t>{noff});
    Device<std::size_t> d_aoff(std::vector<std::size_t>{aoff});
    Device<std::size_t> d_boff(std::vector<std::size_t>{boff});
    Device<std::size_t> d_outbase(std::vector<std::size_t>{outbase});
    std::vector<double> h_fp64 = matrix, h_df = matrix;
    void *workspace = Set_Hamiltonian_Cuda_StreamCreate(2U << 20, potential.size(), potential.data());
    if (!workspace) std::exit(2);
    for (int first = 0; first < points; first += chunk) {
        const int count = std::min(chunk, points - first);
        std::vector<int> nc(noff + count + 29, 0), compact_nc = nc, mnc = nc;
        std::vector<float> ac = compact ? std::vector<float>(aoff + count * no0 + 17, 111.0f) : a;
        std::vector<float> bc(boff + count * no1 + 19, 222.0f);
        for (int k = 0; k < count; ++k) {
            nc[noff + k] = original_nc[first + k];
            compact_nc[noff + k] = compact ? k : nc[noff + k];
            mnc[noff + k] = mn[first + k];
            if (compact) std::copy_n(a.data() + aoff + nc[noff + k] * no0, no0, ac.data() + aoff + k * no0);
            std::copy_n(b.data() + boff + (first + k) * no1, no1, bc.data() + boff + k * no1);
        }
        Device<int> d_count(std::vector<int>{count}), d_nc(nc);
        Device<float> d_a(ac), d_b(bc);
        const int status = Set_Hamiltonian_Cuda_Density(1, spins, std::max(no0, no1), count,
            d_no0.p, d_no1.p, d_count.p, d_hoff.p, d_noff.p, d_aoff.p, d_boff.p, d_outbase.p,
            d_nc.p, d_a.p, d_b.p, d_dmh.p, d_dml.p, d_density.p, outcount, compact);
        if (status != 0) {
            std::fprintf(stderr, "density launch failed: %d\n", status); std::exit(2);
        }
        const int status64 = Set_Hamiltonian_Cuda_Density_FP64(1, spins, std::max(no0, no1), count,
            d_no0.p, d_no1.p, d_count.p, d_hoff.p, d_noff.p, d_aoff.p, d_boff.p, d_outbase.p,
            d_nc.p, d_a.p, d_b.p, d_dm.p, d_density64.p, outcount, compact);
        if (status64 != 0) {
            std::fprintf(stderr, "FP64 density launch failed: %d\n", status64); std::exit(2);
        }
        for (int mode : {0, 2}) {
            auto &h = mode == 0 ? h_fp64 : h_df;
            const int hs = Set_Hamiltonian_Cuda_StreamRun(workspace, 1, spins, potential_rows,
                volume, std::max(no0, no1), matrix_count, h.size(), nc.size(), ac.size(), bc.size(),
                &no0, &no1, &count, mnc.data(), compact_nc.data(), &hoff, &noff, &aoff, &boff,
                ac.data(), bc.data(), h.data(), mode);
            if (hs != 0) { std::fprintf(stderr, "matrix launch failed: %d\n", hs); std::exit(2); }
        }
    }
    Set_Hamiltonian_Cuda_StreamDestroy(workspace);
    check(cudaMemcpy(density.data(), d_density.p, density.size() * sizeof(double), cudaMemcpyDeviceToHost));
    const double df_error = compare("DF density", density, expected_density);
    check(cudaMemcpy(density.data(), d_density64.p, density.size() * sizeof(double), cudaMemcpyDeviceToHost));
    return std::max({df_error, compare("FP64 density", density, expected_density),
                     compare("FP64 matrix", h_fp64, expected_matrix),
                     compare("DF matrix", h_df, expected_matrix)});
}

/* Heterogeneous batches share output rows across pairs, as the neighbours
   of one atom do.  Include an empty pair and padding in every packed array. */
static double exercise_multi(int spins, bool compact)
{
    constexpr int pairs = 3, rows = 719;
    constexpr std::size_t output_count = rows + 41;
    std::vector<int> no0{13, 26, 48}, no1{11, 17, 45}, nolg{273, 129, 0};
    std::vector<std::size_t> hoff(pairs), noff(pairs), aoff(pairs), boff(pairs), outbase{11, 11, 17};
    std::vector<int> nc(5, 0);
    std::vector<float> a(7, 111.0f), b(9, 222.0f);
    std::vector<double> dm(3, 333.0), density(spins * output_count, 0.17);
    for (int p = 0; p < pairs; ++p) {
        hoff[p] = dm.size(); noff[p] = nc.size();
        aoff[p] = a.size(); boff[p] = b.size();
        const int left_rows = compact ? nolg[p] : rows;
        dm.resize(dm.size() + spins * no0[p] * no1[p] + 13, 333.0);
        nc.resize(nc.size() + nolg[p] + 7, 0);
        a.resize(a.size() + left_rows * no0[p] + 17, 111.0f);
        b.resize(b.size() + nolg[p] * no1[p] + 19, 222.0f);
        for (int k = 0; k < nolg[p]; ++k) {
            nc[noff[p] + k] = ((k / 2) * 37 + 317) % rows;
            for (int j = 0; j < no1[p]; ++j)
                b[boff[p] + k * no1[p] + j] = sample(p * 977 + k * 71 + j + 131);
        }
        for (int k = 0; k < left_rows; ++k) {
            const int row = compact ? nc[noff[p] + k] : k;
            for (int i = 0; i < no0[p]; ++i)
                a[aoff[p] + k * no0[p] + i] = sample(p * 983 + row * 43 + i + 97);
        }
        for (int e = 0; e < spins * no0[p] * no1[p]; ++e)
            dm[hoff[p] + e] = sample(p * 991 + e + 104729) * 0.371;
    }
    std::vector<float> hi(dm.size()), lo(dm.size());
    for (std::size_t e = 0; e < dm.size(); ++e) {
        hi[e] = static_cast<float>(dm[e]);
        lo[e] = static_cast<float>(dm[e] - static_cast<double>(hi[e]));
    }
    std::vector<long double> expected(density.begin(), density.end());
    for (int p = 0; p < pairs; ++p)
        for (int s = 0; s < spins; ++s)
            for (int k = 0; k < nolg[p]; ++k) {
                const int Nc = nc[noff[p] + k], row = compact ? k : Nc;
                long double rho = 0.0L;
                for (int i = 0; i < no0[p]; ++i)
                    for (int j = 0; j < no1[p]; ++j)
                        rho += static_cast<long double>(a[aoff[p] + row * no0[p] + i]) *
                               dm[hoff[p] + (s * no0[p] + i) * no1[p] + j] * b[boff[p] + k * no1[p] + j];
                expected[s * output_count + outbase[p] + Nc] += rho;
            }
    Device<int> d_no0(no0), d_no1(no1), d_nolg(nolg), d_nc(nc);
    Device<std::size_t> d_hoff(hoff), d_noff(noff), d_aoff(aoff), d_boff(boff), d_outbase(outbase);
    Device<float> d_a(a), d_b(b), d_hi(hi), d_lo(lo);
    Device<double> d_dm(dm), d_den64(density), d_dendf(density);
    const int fp64 = Set_Hamiltonian_Cuda_Density_FP64(pairs, spins, 48, 273,
        d_no0.p, d_no1.p, d_nolg.p, d_hoff.p, d_noff.p, d_aoff.p, d_boff.p, d_outbase.p,
        d_nc.p, d_a.p, d_b.p, d_dm.p, d_den64.p, output_count, compact);
    const int df = Set_Hamiltonian_Cuda_Density(pairs, spins, 48, 273,
        d_no0.p, d_no1.p, d_nolg.p, d_hoff.p, d_noff.p, d_aoff.p, d_boff.p, d_outbase.p,
        d_nc.p, d_a.p, d_b.p, d_hi.p, d_lo.p, d_dendf.p, output_count, compact);
    if (fp64 || df) { std::fprintf(stderr, "multipair launch failed: %d/%d\n", fp64, df); std::exit(2); }
    check(cudaMemcpy(density.data(), d_den64.p, density.size() * sizeof(double), cudaMemcpyDeviceToHost));
    const double error64 = compare("multipair FP64 density", density, expected);
    check(cudaMemcpy(density.data(), d_dendf.p, density.size() * sizeof(double), cudaMemcpyDeviceToHost));
    double worst = std::max(error64, compare("multipair DF density", density, expected));
    /* A worker stores only its owned output interval, with a local spin
       stride. Include intervals containing no contributing points, cuts
       through repeated indices, one-element tails and allocation guards. */
    const std::size_t cuts[] = {0, 1, 11, 12, 127, 256, 512, output_count-1, output_count};
    for (int mode : {0, 2}) {
        for (std::size_t part = 0; part + 1 < sizeof(cuts)/sizeof(*cuts); ++part) {
            const std::size_t first = cuts[part], last = cuts[part+1], count = last-first;
            constexpr std::size_t guard = 13;
            std::vector<double> local(2*guard + spins*count, 987.125);
            std::fill(local.begin()+guard, local.end()-guard, 0.17);
            std::vector<long double> reference(local.begin(), local.end());
            for (int s = 0; s < spins; ++s)
                for (std::size_t i = 0; i < count; ++i)
                    reference[guard+s*count+i] += 2*(expected[s*output_count+first+i]-0.17L);
            Device<double> out(local);
            auto launch = [&](std::size_t start, std::size_t end) {
                return mode == 0 ? Set_Hamiltonian_Cuda_Density_FP64_Range(pairs, spins, 48, 273,
                    d_no0.p, d_no1.p, d_nolg.p, d_hoff.p, d_noff.p, d_aoff.p, d_boff.p, d_outbase.p,
                    d_nc.p, d_a.p, d_b.p, d_dm.p, out.p+guard, output_count, compact, start, end) :
                    Set_Hamiltonian_Cuda_Density_Range(pairs, spins, 48, 273,
                    d_no0.p, d_no1.p, d_nolg.p, d_hoff.p, d_noff.p, d_aoff.p, d_boff.p, d_outbase.p,
                    d_nc.p, d_a.p, d_b.p, d_hi.p, d_lo.p, out.p+guard, output_count, compact, start, end);
            };
            if (launch(last, first) != 1 || launch(0, output_count+1) != 1 ||
                launch(first, first) != 0 || launch(first, last) || launch(first, last)) {
                std::fprintf(stderr, "output-range launch/validation failed\n"); std::exit(2);
            }
            check(cudaMemcpy(local.data(), out.p, local.size()*sizeof(double), cudaMemcpyDeviceToHost));
            worst = std::max(worst, compare(mode == 0 ? "range FP64" : "range DF", local, reference));
        }
    }
    return worst;
}

/* Optional resident-input microbenchmark, separate from the long-double
   correctness oracle above.  Measures the production API including its
   synchronization; excludes host packing, transfers and orbital evaluation. */
static void benchmark_density(int no, int spins, int repeats)
{
    const char *pairs_env = std::getenv("SETPOINT_BENCH_PAIRS"), *points_env = std::getenv("SETPOINT_BENCH_POINTS");
    const int pairs = pairs_env ? std::atoi(pairs_env) : 16;
    const int points = points_env ? std::atoi(points_env) : 8192;
    if (pairs < 1 || pairs > 4096 || points < 1 || points > 8192 ||
        static_cast<std::size_t>(pairs) * points > 1048576) {
        std::fprintf(stderr, "invalid quadrature benchmark size\n"); std::exit(2);
    }
    const std::size_t output_count = static_cast<std::size_t>(pairs) * points;
    const std::size_t mat = static_cast<std::size_t>(no) * no;
    std::vector<int> nos(pairs, no), counts(pairs, points), nc(pairs * points);
    std::vector<std::size_t> hoff(pairs), noff(pairs), ooff(pairs), outbase(pairs);
    std::vector<float> a(pairs * points * no), b(a.size());
    std::vector<double> dm(pairs * spins * mat), den(spins * output_count, 0.0);
    std::vector<float> hi(dm.size()), lo(dm.size());
    for (int p = 0; p < pairs; ++p) {
        hoff[p] = p * spins * mat;
        noff[p] = p * points;
        ooff[p] = static_cast<std::size_t>(p) * points * no;
        outbase[p] = p * points;
        for (int k = 0; k < points; ++k) nc[p * points + k] = k;
    }
    for (std::size_t j = 0; j < a.size(); ++j) {
        a[j] = sample(static_cast<unsigned>(j) + 47);
        b[j] = sample(static_cast<unsigned>(j) + 941);
    }
    for (std::size_t j = 0; j < dm.size(); ++j) {
        dm[j] = sample(static_cast<unsigned>(j) + 104729) * 0.371;
        hi[j] = static_cast<float>(dm[j]);
        lo[j] = static_cast<float>(dm[j] - static_cast<double>(hi[j]));
    }
    Device<int> d_no(nos), d_count(counts), d_nc(nc);
    Device<std::size_t> d_hoff(hoff), d_noff(noff), d_ooff(ooff), d_outbase(outbase);
    Device<float> d_a(a), d_b(b), d_hi(hi), d_lo(lo);
    Device<double> d_dm(dm), d_den(den);
    auto launch = [&](int mode) {
        return mode == 0 ? Set_Hamiltonian_Cuda_Density_FP64(pairs, spins, no, points,
            d_no.p, d_no.p, d_count.p, d_hoff.p, d_noff.p, d_ooff.p, d_ooff.p, d_outbase.p,
            d_nc.p, d_a.p, d_b.p, d_dm.p, d_den.p, output_count, 0) :
            Set_Hamiltonian_Cuda_Density(pairs, spins, no, points,
            d_no.p, d_no.p, d_count.p, d_hoff.p, d_noff.p, d_ooff.p, d_ooff.p, d_outbase.p,
            d_nc.p, d_a.p, d_b.p, d_hi.p, d_lo.p, d_den.p, output_count, 0);
    };
    double elapsed[2];
    for (int mode : {0, 2}) {
        for (int i = 0; i < 2; ++i)
            if (launch(mode)) { std::fprintf(stderr, "density benchmark warmup failed\n"); std::exit(2); }
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < repeats; ++i)
            if (launch(mode)) { std::fprintf(stderr, "density benchmark failed\n"); std::exit(2); }
        elapsed[mode == 0 ? 0 : 1] =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() / repeats;
    }
    std::printf("DENSITY_BENCH orbitals=%d spins=%d pairs=%d points=%d repeats=%d "
                "native_fp64_ms=%.6f df_ms=%.6f df_over_native=%.3f\n",
                no, spins, pairs, points, repeats, elapsed[0], elapsed[1], elapsed[1] / elapsed[0]);
    /* Include the production wrapper's synchronization/metadata cost, not
       just kernel-event time. Reuse the same heterogeneous-data layout. */
    std::vector<double> potential(spins * points), matrix(dm.size(), 0.0);
    for (std::size_t j = 0; j < potential.size(); ++j) potential[j] = sample(j + 271) * 0.731;
    Device<double> d_potential(potential), d_matrix(matrix);
    auto matrix_launch = [&](int mode) {
        return Set_Hamiltonian_Cuda_MatrixElements(pairs, spins, points, 0.071238947,
            no, spins * mat, d_no.p, d_no.p, d_count.p, d_nc.p, d_nc.p,
            d_hoff.p, d_noff.p, d_ooff.p, d_ooff.p, d_a.p, d_b.p,
            d_potential.p, d_matrix.p, mode);
    };
    for (int mode : {0, 2}) {
        for (int i = 0; i < 2; ++i)
            if (matrix_launch(mode)) { std::fprintf(stderr, "matrix benchmark warmup failed\n"); std::exit(2); }
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < repeats; ++i)
            if (matrix_launch(mode)) { std::fprintf(stderr, "matrix benchmark failed\n"); std::exit(2); }
        elapsed[mode == 0 ? 0 : 1] =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() / repeats;
    }
    std::printf("MATRIX_BENCH orbitals=%d spins=%d pairs=%d points=%d repeats=%d "
                "native_fp64_ms=%.6f df_ms=%.6f df_over_native=%.3f\n",
                no, spins, pairs, points, repeats, elapsed[0], elapsed[1], elapsed[1] / elapsed[0]);
}

int main()
{
    check(cudaSetDevice(0));
    openmx_gpu_test_use_visible_pool();
    openmx_gpu_test_worker_failures();
    std::printf("OPENMX_SETDEN_FP64_SPLIT_SPIN=%s\n", std::getenv("OPENMX_SETDEN_FP64_SPLIT_SPIN") ?
                std::getenv("OPENMX_SETDEN_FP64_SPLIT_SPIN") : "auto (max_no > 32 and spins > 1)");
    double worst = 0;
    int cases = 0;
    for (int no : {1, 16, 17, 32, 41, 48})
        for (int spins : {1, 2, 4}) {
            worst = std::max(worst, exercise(no, std::max(1, no - 3), spins, 273, false)); ++cases;
            for (int chunk : {31, 128, 273}) {
                worst = std::max(worst, exercise(no, std::max(1, no - 3), spins, chunk, true)); ++cases;
            }
        }
    for (int spins : {1, 2, 4}) {
        worst = std::max(worst, exercise_multi(spins, false)); ++cases;
        worst = std::max(worst, exercise_multi(spins, true)); ++cases;
    }
    check(cudaDeviceSynchronize());
    std::printf("PASS: %d full/chunk FP64/DF density and matrix cases; spins 1/2/4, "
                "orbitals 1/16/17/32/41/48, repeated original indices, padding, accumulated chunks, "
                "heterogeneous pairs, empty overlaps and 96 guarded output-range checks; "
                "max scaled error %.3e\n", cases, worst);
    const char *benchmark = std::getenv("SETPOINT_BENCHMARK");
    if (benchmark != nullptr && std::atoi(benchmark) > 0) {
        const int repeats = std::min(1000, std::atoi(benchmark));
        for (int no : {13, 26, 48})
            for (int spins : {1, 2, 4}) benchmark_density(no, spins, repeats);
    }
}

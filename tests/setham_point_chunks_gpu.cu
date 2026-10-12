/* Production CUDA quadrature with full atom rows and bounded overlap tiles.
   The oracle is independent long-double arithmetic.  In particular, a
   chunk's local orbital row must never replace the original density index. */
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

extern "C" int Set_Hamiltonian_Cuda_Density(
    int, int, int, int, const int *, const int *, const int *,
    const std::size_t *, const std::size_t *, const std::size_t *, const std::size_t *,
    const std::size_t *, const int *, const float *, const float *,
    const float *, const float *, double *, std::size_t, int);
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
    Device<double> d_density(density);
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
    return std::max({compare("density", density, expected_density),
                     compare("FP64 matrix", h_fp64, expected_matrix),
                     compare("DF matrix", h_df, expected_matrix)});
}

int main()
{
    check(cudaSetDevice(0));
    double worst = 0;
    int cases = 0;
    for (int no : {1, 16, 17, 32, 41, 48})
        for (int spins : {1, 2, 4}) {
            worst = std::max(worst, exercise(no, std::max(1, no - 3), spins, 273, false)); ++cases;
            for (int chunk : {31, 128, 273}) {
                worst = std::max(worst, exercise(no, std::max(1, no - 3), spins, chunk, true)); ++cases;
            }
        }
    check(cudaDeviceSynchronize());
    std::printf("PASS: %d full/chunk density and FP64/DF matrix cases; spins 1/2/4, "
                "orbitals 1/16/17/32/41/48, repeated original indices, padding, accumulated chunks; "
                "max scaled error %.3e\n", cases, worst);
}

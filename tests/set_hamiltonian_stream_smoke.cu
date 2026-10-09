/* CUDA streaming API regression.  The CPU oracle evaluates the quadrature
   in long double, independently of the tiled CUDA kernel.  Run through
   run_set_hamiltonian_stream_smoke.sh; no OpenMX input or MPI is needed. */
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" void *Set_Hamiltonian_Cuda_StreamCreate(std::size_t, std::size_t, const double *);
extern "C" void Set_Hamiltonian_Cuda_StreamDestroy(void *);
extern "C" int Set_Hamiltonian_Cuda_StreamRun(
    void *, int, int, std::size_t, double, int, int,
    std::size_t, std::size_t, std::size_t, std::size_t,
    const int *, const int *, const int *, const int *, const int *,
    const std::size_t *, const std::size_t *, const std::size_t *, const std::size_t *,
    const float *, const float *, double *, int);

namespace {

constexpr std::size_t kPotentialLength = 257;
constexpr std::size_t kArenaBytes = 2 * 1024 * 1024;
constexpr int kPairs = 7;
constexpr double kGridVolume = 0.071238947;

void fail(const char *message)
{
    std::fprintf(stderr, "FAIL: %s\n", message);
    std::exit(1);
}

void check(cudaError_t status)
{
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(status));
        std::exit(2);
    }
}

double value(std::uint32_t seed)
{
    seed ^= seed >> 16;
    seed *= 0x7feb352dU;
    seed ^= seed >> 15;
    seed *= 0x846ca68bU;
    seed ^= seed >> 16;
    return (static_cast<double>(seed & 0xffffU) - 32768.0) / 8192.0;
}

struct Batch {
    int spins;
    int max_no = 0;
    int max_output = 0;
    std::vector<int> no0, no1, nolg, mn, nc;
    std::vector<std::size_t> h_offset, nolg_offset, orbs0_offset, orbs1_offset;
    std::vector<float> orbs0, orbs1;
    std::vector<double> h;
    std::vector<unsigned char> active;
};

Batch make_batch(int spins, int seed)
{
    const int left_sizes[kPairs] = {1, 7, 16, 17, 23, 33, 73};
    const int right_sizes[kPairs] = {3, 19, 1, 31, 17, 9, 5};
    const int grid_sizes[kPairs] = {0, 1, 31, 32, 33, 67, 129};
    Batch batch;
    batch.spins = spins;
    // Nonzero prefixes, inter-pair gaps and tails exercise every offset.
    batch.h.resize(9);
    batch.mn.resize(5, 0);
    batch.nc.resize(5, 0);
    batch.orbs0.resize(11, 2.75f);
    batch.orbs1.resize(13, -3.25f);
    for (int p = 0; p < kPairs; p++) {
        const int no0 = left_sizes[(p + seed) % kPairs];
        const int no1 = right_sizes[(p * 3 + seed) % kPairs];
        const int nolg = grid_sizes[(p * 5 + seed) % kPairs];
        const int rows = nolg + 11;
        batch.no0.push_back(no0);
        batch.no1.push_back(no1);
        batch.nolg.push_back(nolg);
        batch.h_offset.push_back(batch.h.size());
        batch.nolg_offset.push_back(batch.mn.size());
        batch.orbs0_offset.push_back(batch.orbs0.size());
        batch.orbs1_offset.push_back(batch.orbs1.size());
        batch.max_no = std::max(batch.max_no, std::max(no0, no1));
        batch.max_output = std::max(batch.max_output, spins * no0 * no1);
        batch.h.resize(batch.h.size() + static_cast<std::size_t>(spins) * no0 * no1 + 7);
        for (int g = 0; g < nolg; g++) {
            batch.mn.push_back((g * 47 + p * 31 + seed * 7) % kPotentialLength);
            // Gather order differs from orbital row order, with repeats.
            batch.nc.push_back((g * 13 + p * 5 + seed) % rows);
        }
        batch.mn.resize(batch.mn.size() + 3, 0);
        batch.nc.resize(batch.nc.size() + 3, 0);
        for (int r = 0; r < rows * no0; r++)
            batch.orbs0.push_back(static_cast<float>(value(113U * seed + 97U * p + r + 1009U)));
        for (int r = 0; r < nolg * no1; r++)
            batch.orbs1.push_back(static_cast<float>(value(31U * seed + 71U * p + r + 65537U)));
        batch.orbs0.resize(batch.orbs0.size() + 5, 2.75f);
        batch.orbs1.resize(batch.orbs1.size() + 7, -3.25f);
    }
    batch.h.resize(batch.h.size() + 5);
    batch.active.resize(batch.h.size(), 0);
    for (std::size_t i = 0; i < batch.h.size(); i++) batch.h[i] = 77777.25 + static_cast<double>(i);
    for (int p = 0; p < kPairs; p++) {
        const std::size_t count = static_cast<std::size_t>(spins) * batch.no0[p] * batch.no1[p];
        for (std::size_t e = 0; e < count; e++) {
            const auto index = batch.h_offset[p] + e;
            batch.h[index] = value(static_cast<std::uint32_t>(e) + 3001U * p + 197U * seed) * 0.37 + 0.13;
            batch.active[index] = 1;
        }
    }
    return batch;
}

std::vector<long double> reference(const Batch &batch, const std::vector<double> &potential)
{
    std::vector<long double> result(batch.h.begin(), batch.h.end());
    for (int p = 0; p < kPairs; p++) {
        for (int s = 0; s < batch.spins; s++) {
            for (int i = 0; i < batch.no0[p]; i++) {
                for (int j = 0; j < batch.no1[p]; j++) {
                    const auto hindex = batch.h_offset[p] +
                        (static_cast<std::size_t>(s) * batch.no0[p] + i) * batch.no1[p] + j;
                    long double sum = result[hindex];
                    for (int g = 0; g < batch.nolg[p]; g++) {
                        const auto ng = batch.nolg_offset[p] + g;
                        const auto left = batch.orbs0_offset[p] +
                            static_cast<std::size_t>(batch.nc[ng]) * batch.no0[p] + i;
                        const auto right = batch.orbs1_offset[p] + static_cast<std::size_t>(g) * batch.no1[p] + j;
                        sum += static_cast<long double>(kGridVolume) *
                               potential[static_cast<std::size_t>(s) * kPotentialLength + batch.mn[ng]] *
                               batch.orbs0[left] * batch.orbs1[right];
                    }
                    result[hindex] = sum;
                }
            }
        }
    }
    return result;
}

int run(void *workspace, Batch &batch, int mode = 0)
{
    return Set_Hamiltonian_Cuda_StreamRun(workspace, kPairs, batch.spins, kPotentialLength,
        kGridVolume, batch.max_no, batch.max_output, batch.h.size(), batch.mn.size(),
        batch.orbs0.size(), batch.orbs1.size(), batch.no0.data(), batch.no1.data(), batch.nolg.data(),
        batch.mn.data(), batch.nc.data(), batch.h_offset.data(), batch.nolg_offset.data(),
        batch.orbs0_offset.data(), batch.orbs1_offset.data(), batch.orbs0.data(), batch.orbs1.data(), batch.h.data(), mode);
}

double verify(void *workspace, Batch &batch, const std::vector<double> &potential, int mode = 0)
{
    const auto initial = batch.h;
    const auto expected = reference(batch, potential);
    if (run(workspace, batch, mode) != 0) fail("valid batch failed");
    double worst = 0;
    for (std::size_t i = 0; i < batch.h.size(); i++) {
        if (!batch.active[i]) {
            if (std::memcmp(&batch.h[i], &initial[i], sizeof(double)) != 0) fail("H padding overwritten");
            continue;
        }
        const long double scaled_error = std::fabs(static_cast<long double>(batch.h[i]) - expected[i]) /
                                        std::max(1.0L, std::fabs(expected[i]));
        if (!std::isfinite(batch.h[i]) || scaled_error > 2e-12L) {
            std::fprintf(stderr, "Mismatch: spins=%d H[%zu] GPU=%.17g CPU=%.21Lg scaled_error=%.3Le\n",
                         batch.spins, i, batch.h[i], expected[i], scaled_error);
            fail("quadrature disagrees with CPU reference");
        }
        worst = std::max(worst, static_cast<double>(scaled_error));
    }
    return worst;
}

void reject_without_output(void *workspace, Batch &batch)
{
    const auto initial = batch.h;
    if (run(workspace, batch) <= 0) fail("arena overflow was not a recoverable failure");
    if (std::memcmp(batch.h.data(), initial.data(), initial.size() * sizeof(double)) != 0)
        fail("failed batch changed host H");
    check(cudaDeviceSynchronize());
    check(cudaGetLastError());
}

} // namespace

int main()
{
    check(cudaSetDevice(0));
    int runtime_version = 0;
    check(cudaRuntimeGetVersion(&runtime_version));
    std::printf("CUDA runtime %d\n", runtime_version);
    std::vector<double> potential(4 * kPotentialLength);
    for (std::size_t i = 0; i < potential.size(); i++) potential[i] = value(static_cast<std::uint32_t>(i) + 919U);
    void *workspace = Set_Hamiltonian_Cuda_StreamCreate(kArenaBytes, potential.size(), potential.data());
    if (!workspace) fail("cannot create streaming workspace");
    double worst = 0;
    int cases = 0;
    for (int mode : {0, 2}) {
      for (int spins : {1, 2, 4}) {
        for (int seed = 0; seed < kPairs; seed++) {
            auto batch = make_batch(spins, seed);
            worst = std::max(worst, verify(workspace, batch, potential, mode));
            cases++;
            // Reusing the same batch must add its contribution to the updated
            // H, not an arena's previous H or a freshly zeroed accumulator.
            worst = std::max(worst, verify(workspace, batch, potential, mode));
            cases++;
        }
    }
    }
    {
        auto oversized = make_batch(4, 3);
        oversized.orbs1.resize(kArenaBytes / sizeof(float) + 512, 0.5f);
        reject_without_output(workspace, oversized);
        // An overflow after earlier arrays were uploaded must not poison the
        // context or the next batch's different arena layout.
        auto after_failure = make_batch(2, 6);
        worst = std::max(worst, verify(workspace, after_failure, potential));
        cases++;
    }
    {
        void *tiny = Set_Hamiltonian_Cuda_StreamCreate(64, potential.size(), potential.data());
        if (!tiny) fail("cannot create tiny streaming workspace");
        auto batch = make_batch(1, 2);
        reject_without_output(tiny, batch);
        Set_Hamiltonian_Cuda_StreamDestroy(tiny);
        worst = std::max(worst, verify(workspace, batch, potential));
        cases++;
    }
    Set_Hamiltonian_Cuda_StreamDestroy(workspace);
    Set_Hamiltonian_Cuda_StreamDestroy(nullptr);
    check(cudaDeviceSynchronize());
    check(cudaGetLastError());
    std::printf("PASS: %d FP64/DF quadrature cases, spins 1/2/4, irregular orbitals/grids, arena reuse, "
                "H accumulation/padding, 2 recoverable overflows; max scaled error %.3e\n", cases, worst);
    return 0;
}

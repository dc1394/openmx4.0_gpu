// Numerical check of the double-float density kernels of
// source/set_hamiltonian_hip.cu: the local mode's CSR gather
// (Set_Density_Grid_Hip_LocalDensity, called by Set_Density_Grid_GPU.c on the
// ranks whose Set_Hamiltonian tables are device-resident) and the on-the-fly
// density (Set_Hamiltonian_Hip_OTF_Density), including its per-spin LDS split
// for the pairs whose S x NO0 x NO1 (hi, lo) floats do not fit in LDS.
//
// Synthetic tables in the layout of Set_Hamiltonian's matrix-element tables:
// per local atom the central orbitals over its whole sphere (GridN x NO0
// floats), per pair the neighbour orbitals at the overlap points (NOLG x NO1
// floats) and the overlap points' sphere indices (nolg_Nc), the density
// matrix blocks (S x NO0 x NO1 doubles per pair, pair order, spin major), and
// the CSR over the sphere points as SDG_local_build makes it.  Every result is
// compared with a long double evaluation of the same sums; the FP64 sum in the
// FP64 kernel's order is shown for scale.  Also checked: the CSR kernel's
// register and memory variants and two runs give the same bits, the
// on-the-fly kernel (atomics) agrees with the gather, and a pair whose DM
// blocks exceed the LDS takes the split kernel instead of FP64.
//
// Build and run from the repository root (no OpenMX objects are needed: the
// kernels' translation unit is included):
//   /opt/rocm-7.2.0/bin/hipcc -std=c++20 -O3 --offload-arch=gfx1200 -ffp-contract=off \
//       -DOCML_BASIC_ROUNDED_OPERATIONS -Isource tools/check_density_df_hip.cu -o check_density_df_hip
//   ./check_density_df_hip            (exit status 0 when every check passes)
#include "set_hamiltonian_hip.cu"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

/* the FP32:FP64 throughput ratio of eigen_refine_gpu.cu: a GeForce/RDNA
   value, so that the default precision is double-float */
extern "C" int openmx_gpu_fp32_fp64_ratio(void) { return 16; }

#define CK(x)                                                                                        \
    do {                                                                                             \
        hipError_t _s = (x);                                                                         \
        if (_s != hipSuccess) {                                                                      \
            std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(_s));          \
            std::exit(2);                                                                            \
        }                                                                                            \
    } while (0)

namespace check {

struct Problem {
    int S = 1;
    std::vector<int> grid_n, no_atom;                  /* per local atom */
    std::vector<int> pair_atom, no0, no1, nolg;        /* per pair */
    std::vector<std::size_t> hoff, noff, o0off, o1off, denoff;
    std::vector<int> nc;                               /* per overlap point */
    std::vector<float> orbs0, orbs1;
    std::vector<double> dm;
    std::vector<std::uint32_t> atom_out_base, out_ptr, term_pt, pt_pair;
    std::size_t output_count = 0, total_nolg = 0;
    int max_no1 = 0, max_no = 0, max_nolg = 0;
};

/* atoms with the given orbital counts; every atom pairs with itself (its
   whole sphere) and with `neighbours` others over random subsets of its
   sphere, as Set_Hamiltonian lists (Mc_AN, h_AN) */
Problem make_problem(int S, const std::vector<int> &species_no, int atoms, int neighbours, int grid_lo, int grid_hi,
                     unsigned seed)
{
    Problem P;
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    P.S = S;
    std::size_t out = 0, o0 = 0, o1 = 0, h = 0, pts = 0;
    P.atom_out_base.push_back(0);
    for (int a = 0; a < atoms; ++a) {
        const int n = grid_lo + static_cast<int>(rng() % static_cast<unsigned>(grid_hi - grid_lo + 1));
        const int no = species_no[rng() % species_no.size()];
        P.grid_n.push_back(n);
        P.no_atom.push_back(no);
        /* central orbitals: radially decaying values of both signs */
        for (int g = 0; g < n; ++g) {
            const double r = static_cast<double>(g) / n;
            for (int i = 0; i < no; ++i) P.orbs0.push_back(static_cast<float>(u(rng) * std::exp(-2.0 * r)));
        }
        for (int hh = 0; hh <= neighbours; ++hh) {
            const int nb_no = species_no[rng() % species_no.size()];
            std::vector<int> pts_of_pair;
            if (hh == 0) {
                for (int g = 0; g < n; ++g) pts_of_pair.push_back(g);
            }
            else {
                const double frac = 0.05 + 0.6 * (0.5 + 0.5 * u(rng));
                for (int g = 0; g < n; ++g)
                    if (0.5 + 0.5 * u(rng) < frac) pts_of_pair.push_back(g);
            }
            const int pno1 = (hh == 0) ? no : nb_no;
            P.pair_atom.push_back(a);
            P.no0.push_back(no);
            P.no1.push_back(pno1);
            P.nolg.push_back(static_cast<int>(pts_of_pair.size()));
            P.hoff.push_back(h);
            P.noff.push_back(pts);
            P.o0off.push_back(o0);
            P.o1off.push_back(o1);
            P.denoff.push_back(out);
            for (int g : pts_of_pair) {
                P.nc.push_back(g);
                for (int j = 0; j < pno1; ++j) P.orbs1.push_back(static_cast<float>(u(rng)));
            }
            /* DM blocks: doubles that floats do not represent, a dominant
               diagonal on the on-site pair */
            for (int s = 0; s < S; ++s)
                for (int i = 0; i < no; ++i)
                    for (int j = 0; j < pno1; ++j)
                        P.dm.push_back(0.5 * u(rng) + ((hh == 0 && i == j && s < 2) ? 1.0 : 0.0));
            h += static_cast<std::size_t>(S) * no * pno1;
            pts += pts_of_pair.size();
            o1 += pts_of_pair.size() * static_cast<std::size_t>(pno1);
            P.max_no1 = std::max(P.max_no1, pno1);
            P.max_no = std::max(P.max_no, std::max(no, pno1));
            P.max_nolg = std::max(P.max_nolg, static_cast<int>(pts_of_pair.size()));
        }
        o0 += static_cast<std::size_t>(n) * no;
        out += static_cast<std::size_t>(n);
        P.atom_out_base.push_back(static_cast<std::uint32_t>(out));
    }
    P.output_count = out;
    P.total_nolg = pts;

    /* the CSR of SDG_local_build: per output point its overlap points in
       pair order, then in the pair's point order */
    std::vector<std::uint32_t> degree(out, 0);
    P.pt_pair.assign(pts, 0);
    for (std::size_t p = 0; p < P.no0.size(); ++p)
        for (int k = 0; k < P.nolg[p]; ++k) {
            degree[P.atom_out_base[P.pair_atom[p]] + P.nc[P.noff[p] + k]]++;
            P.pt_pair[P.noff[p] + k] = static_cast<std::uint32_t>(p);
        }
    P.out_ptr.assign(out + 1, 0);
    for (std::size_t o = 0; o < out; ++o) {
        P.out_ptr[o + 1] = P.out_ptr[o] + degree[o];
        degree[o] = P.out_ptr[o];
    }
    P.term_pt.assign(pts, 0);
    for (std::size_t p = 0; p < P.no0.size(); ++p)
        for (int k = 0; k < P.nolg[p]; ++k)
            P.term_pt[degree[P.atom_out_base[P.pair_atom[p]] + P.nc[P.noff[p] + k]]++] =
                static_cast<std::uint32_t>(P.noff[p] + k);
    return P;
}

/* the reference (long double) and the FP64 kernel's sum order (double),
   and per point the sum of the magnitudes of the terms */
void reference(const Problem &P, std::vector<long double> &ref, std::vector<double> &fp64, std::vector<double> &mag)
{
    const std::size_t N = P.output_count;
    ref.assign(static_cast<std::size_t>(P.S) * N, 0.0L);
    fp64.assign(static_cast<std::size_t>(P.S) * N, 0.0);
    mag.assign(static_cast<std::size_t>(P.S) * N, 0.0);
    for (std::size_t o = 0; o < N; ++o) {
        for (std::uint32_t q = P.out_ptr[o]; q < P.out_ptr[o + 1]; ++q) {
            const std::uint32_t pt = P.term_pt[q], p = P.pt_pair[pt];
            const int n0 = P.no0[p], n1 = P.no1[p];
            const float *phi0 = &P.orbs0[P.o0off[p] + static_cast<std::size_t>(P.nc[pt]) * n0];
            const float *phi1 = &P.orbs1[P.o1off[p] + (pt - P.noff[p]) * static_cast<std::size_t>(n1)];
            for (int s = 0; s < P.S; ++s) {
                const double *d = &P.dm[P.hoff[p] + static_cast<std::size_t>(s) * n0 * n1];
                long double e = 0.0L;
                double e64 = 0.0, m = 0.0;
                for (int i = 0; i < n0; ++i) {
                    long double t = 0.0L;
                    double t64 = 0.0;
                    for (int j = 0; j < n1; ++j) {
                        t += static_cast<long double>(d[i * n1 + j]) * static_cast<long double>(phi1[j]);
                        t64 += static_cast<double>(phi1[j]) * d[i * n1 + j];
                        m += std::fabs(d[i * n1 + j] * phi1[j] * phi0[i]);
                    }
                    e += static_cast<long double>(phi0[i]) * t;
                    e64 += static_cast<double>(phi0[i]) * t64;
                }
                ref[s * N + o] += e;
                fp64[s * N + o] += e64;
                mag[s * N + o] += m;
            }
        }
    }
}

struct Errors {
    double max_mag = 0.0, rms_mag = 0.0, max_rel = 0.0;
};

/* errors relative to the point's sum of magnitudes, and relative to the
   value where the value is not a cancellation remnant */
Errors errors(const std::vector<double> &x, const std::vector<long double> &ref, const std::vector<double> &mag)
{
    Errors E;
    double sum2 = 0.0;
    for (std::size_t k = 0; k < x.size(); ++k) {
        const double err = static_cast<double>(std::fabs(static_cast<long double>(x[k]) - ref[k]));
        const double rm = (0.0 < mag[k]) ? err / mag[k] : err;
        E.max_mag = std::max(E.max_mag, rm);
        sum2 += rm * rm;
        if (1.0e-3 * mag[k] < std::fabs(static_cast<double>(ref[k])))
            E.max_rel = std::max(E.max_rel, err / std::fabs(static_cast<double>(ref[k])));
    }
    E.rms_mag = std::sqrt(sum2 / static_cast<double>(x.empty() ? 1 : x.size()));
    return E;
}

template <typename T>
T *upload(const std::vector<T> &v)
{
    T *d = nullptr;
    CK(hipMalloc(reinterpret_cast<void **>(&d), sizeof(T) * (v.empty() ? 1 : v.size())));
    if (!v.empty()) CK(hipMemcpy(d, v.data(), sizeof(T) * v.size(), hipMemcpyHostToDevice));
    return d;
}

struct Device {
    std::uint32_t *out_ptr, *term_pt, *pt_pair;
    int *no0, *no1, *nolg, *nc;
    std::size_t *hoff, *noff, *o0off, *o1off, *denoff;
    float *o0, *o1;
    double *dm, *den;
};

Device to_device(const Problem &P)
{
    Device D;
    D.out_ptr = upload(P.out_ptr);
    D.term_pt = upload(P.term_pt);
    D.pt_pair = upload(P.pt_pair);
    D.no0 = upload(P.no0);
    D.no1 = upload(P.no1);
    D.nolg = upload(P.nolg);
    D.nc = upload(P.nc);
    D.hoff = upload(P.hoff);
    D.noff = upload(P.noff);
    D.o0off = upload(P.o0off);
    D.o1off = upload(P.o1off);
    D.denoff = upload(P.denoff);
    D.o0 = upload(P.orbs0);
    D.o1 = upload(P.orbs1);
    D.dm = upload(P.dm);
    CK(hipMalloc(reinterpret_cast<void **>(&D.den), sizeof(double) * P.S * P.output_count));
    return D;
}

void release(Device &D)
{
    CK(hipFree(D.out_ptr)); CK(hipFree(D.term_pt)); CK(hipFree(D.pt_pair));
    CK(hipFree(D.no0)); CK(hipFree(D.no1)); CK(hipFree(D.nolg)); CK(hipFree(D.nc));
    CK(hipFree(D.hoff)); CK(hipFree(D.noff)); CK(hipFree(D.o0off)); CK(hipFree(D.o1off)); CK(hipFree(D.denoff));
    CK(hipFree(D.o0)); CK(hipFree(D.o1)); CK(hipFree(D.dm)); CK(hipFree(D.den));
}

/* the CSR kernel on fresh DM (its split is in place); max_no selects the
   variant (an upper bound of NO1: 100 forces the memory variant) */
std::vector<double> run_csr(const Problem &P, Device &D, int max_no, double *ms)
{
    std::vector<double> den(static_cast<std::size_t>(P.S) * P.output_count);
    hipEvent_t a, b;
    CK(hipMemcpy(D.dm, P.dm.data(), sizeof(double) * P.dm.size(), hipMemcpyHostToDevice));
    CK(hipEventCreate(&a));
    CK(hipEventCreate(&b));
    CK(hipEventRecord(a, 0));
    const int status = Set_Density_Grid_Hip_LocalDensity(P.output_count, P.S, max_no, P.dm.size(), D.out_ptr,
                                                          D.term_pt, D.pt_pair, D.no0, D.no1, D.hoff, D.noff, D.o0off,
                                                          D.o1off, D.nc, D.o0, D.o1, D.dm, D.den);
    CK(hipEventRecord(b, 0));
    CK(hipEventSynchronize(b));
    float t = 0.0f;
    CK(hipEventElapsedTime(&t, a, b));
    if (ms != nullptr) *ms = t;
    CK(hipEventDestroy(a));
    CK(hipEventDestroy(b));
    if (status != 0) {
        std::fprintf(stderr, "Set_Density_Grid_Hip_LocalDensity returned %d\n", status);
        std::exit(1);
    }
    CK(hipMemcpy(den.data(), D.den, sizeof(double) * den.size(), hipMemcpyDeviceToHost));
    return den;
}

/* the on-the-fly kernel over the same pairs (one batch) */
std::vector<double> run_otf(const Problem &P, Device &D, double *ms)
{
    std::vector<double> den(static_cast<std::size_t>(P.S) * P.output_count);
    hipEvent_t a, b;
    CK(hipMemcpy(D.dm, P.dm.data(), sizeof(double) * P.dm.size(), hipMemcpyHostToDevice));
    CK(hipMemset(D.den, 0, sizeof(double) * den.size()));
    CK(hipEventCreate(&a));
    CK(hipEventCreate(&b));
    CK(hipEventRecord(a, 0));
    const int status = Set_Hamiltonian_Hip_OTF_Density(static_cast<int>(P.no0.size()), P.max_nolg, P.S, P.max_no,
                                                       D.no0, D.no1, D.nolg, D.hoff, D.noff, D.o0off, D.o1off,
                                                       D.denoff, D.nc, D.dm, D.o0, D.o1, P.output_count, D.den);
    CK(hipEventRecord(b, 0));
    CK(hipEventSynchronize(b));
    float t = 0.0f;
    CK(hipEventElapsedTime(&t, a, b));
    if (ms != nullptr) *ms = t;
    CK(hipEventDestroy(a));
    CK(hipEventDestroy(b));
    if (status != 0) {
        std::fprintf(stderr, "Set_Hamiltonian_Hip_OTF_Density returned %d\n", status);
        std::exit(1);
    }
    CK(hipMemcpy(den.data(), D.den, sizeof(double) * den.size(), hipMemcpyDeviceToHost));
    return den;
}

/* FP64 timing reference: the loops of the local mode's FP64 OpenMP kernel
   (Set_Density_Grid_GPU.c) as a HIP kernel */
__global__ void csr_density_fp64_kernel(std::size_t N, int S, const std::uint32_t *out_ptr, const std::uint32_t *term_pt,
                                        const std::uint32_t *pt_pair, const int *no0, const int *no1,
                                        const std::size_t *hoff, const std::size_t *noff, const std::size_t *o0off,
                                        const std::size_t *o1off, const int *nc, const float *o0, const float *o1,
                                        const double *dm, double *den)
{
    const std::size_t out = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (out >= N) return;
    double g[4] = {0.0, 0.0, 0.0, 0.0};
    for (std::uint32_t q = out_ptr[out]; q < out_ptr[out + 1]; ++q) {
        const std::uint32_t pt = term_pt[q], p = pt_pair[pt];
        const int n0 = no0[p], n1 = no1[p];
        const std::size_t mat = static_cast<std::size_t>(n0) * n1;
        const float *phi0 = o0 + o0off[p] + static_cast<std::size_t>(nc[pt]) * n0;
        const float *phi1 = o1 + o1off[p] + (pt - noff[p]) * static_cast<std::size_t>(n1);
        double e[4] = {0.0, 0.0, 0.0, 0.0};
        for (int i = 0; i < n0; ++i) {
            double t[4] = {0.0, 0.0, 0.0, 0.0};
            for (int j = 0; j < n1; ++j) {
                const double f = phi1[j];
                for (int s = 0; s < S; ++s) t[s] += f * dm[hoff[p] + s * mat + static_cast<std::size_t>(i) * n1 + j];
            }
            for (int s = 0; s < S; ++s) e[s] += static_cast<double>(phi0[i]) * t[s];
        }
        for (int s = 0; s < S; ++s) g[s] += e[s];
    }
    for (int s = 0; s < S; ++s) den[s * N + out] = g[s];
}

double time_fp64(const Problem &P, Device &D)
{
    hipEvent_t a, b;
    CK(hipMemcpy(D.dm, P.dm.data(), sizeof(double) * P.dm.size(), hipMemcpyHostToDevice));
    CK(hipEventCreate(&a));
    CK(hipEventCreate(&b));
    CK(hipEventRecord(a, 0));
    hipLaunchKernelGGL(csr_density_fp64_kernel, dim3(static_cast<unsigned>((P.output_count + 127) / 128)), dim3(128),
                       0, 0, P.output_count, P.S, D.out_ptr, D.term_pt, D.pt_pair, D.no0, D.no1, D.hoff, D.noff,
                       D.o0off, D.o1off, D.nc, D.o0, D.o1, D.dm, D.den);
    CK(hipGetLastError());
    CK(hipEventRecord(b, 0));
    CK(hipEventSynchronize(b));
    float t = 0.0f;
    CK(hipEventElapsedTime(&t, a, b));
    CK(hipEventDestroy(a));
    CK(hipEventDestroy(b));
    return t;
}

int failures = 0;

void expect(bool ok, const char *what)
{
    if (!ok) {
        std::printf("  FAIL: %s\n", what);
        ++failures;
    }
}

void run_case(const char *name, int S, const std::vector<int> &species_no, int atoms, int neighbours, int grid_lo,
              int grid_hi, unsigned seed, bool expect_split, double tol)
{
    Problem P = make_problem(S, species_no, atoms, neighbours, grid_lo, grid_hi, seed);
    std::vector<long double> ref;
    std::vector<double> fp64, mag;
    reference(P, ref, fp64, mag);
    Device D = to_device(P);
    const std::size_t lds = static_cast<std::size_t>(S) * P.max_no * P.max_no * sizeof(float2);
    const bool split = static_cast<int>(lds) > device_shared_limit();

    std::printf("%s: S=%d atoms=%d pairs=%zu points=%zu terms=%zu max NO=%d (OTF LDS %zu B: %s)\n", name, S, atoms,
                P.no0.size(), P.output_count, P.total_nolg, P.max_no, lds, split ? "per-spin split" : "one block");
    expect(split == expect_split, "the expected on-the-fly kernel variant");

    double t_reg = 0.0, t_mem = 0.0, t_otf = 0.0;
    std::vector<double> csr = run_csr(P, D, P.max_no1, nullptr);
    csr = run_csr(P, D, P.max_no1, &t_reg);
    std::vector<double> csr2 = run_csr(P, D, P.max_no1, nullptr);
    std::vector<double> csr_mem = run_csr(P, D, 100, &t_mem);
    std::vector<double> otf = run_otf(P, D, nullptr);
    otf = run_otf(P, D, &t_otf);
    const double t_fp64 = time_fp64(P, D);

    const Errors e_csr = errors(csr, ref, mag), e_otf = errors(otf, ref, mag), e_64 = errors(fp64, ref, mag);
    std::printf("  CSR double-float : max %.2e rms %.2e of the magnitudes, max %.2e of the value\n", e_csr.max_mag,
                e_csr.rms_mag, e_csr.max_rel);
    std::printf("  OTF double-float : max %.2e rms %.2e of the magnitudes, max %.2e of the value\n", e_otf.max_mag,
                e_otf.rms_mag, e_otf.max_rel);
    std::printf("  FP64 (host, order of the FP64 kernel): max %.2e rms %.2e of the magnitudes, max %.2e of the value\n",
                e_64.max_mag, e_64.rms_mag, e_64.max_rel);
    std::printf("  time: CSR df %.3f ms (registers%s), %.3f ms (memory); OTF df %.3f ms; CSR FP64 (HIP) %.3f ms\n",
                t_reg, P.max_no1 <= 64 ? "" : " = memory", t_mem, t_otf, t_fp64);
    expect(e_csr.max_mag < tol, "CSR double-float accuracy");
    expect(e_otf.max_mag < tol, "OTF double-float accuracy");
    expect(std::equal(csr.begin(), csr.end(), csr2.begin()), "CSR: the same bits on a second run");
    expect(std::equal(csr.begin(), csr.end(), csr_mem.begin()), "CSR: register and memory variants give the same bits");

    /* the FP64 mode: nothing runs and the DM stays as it is */
    {
        std::vector<double> back(P.dm.size());
        CK(hipMemcpy(D.dm, P.dm.data(), sizeof(double) * P.dm.size(), hipMemcpyHostToDevice));
        grid_precision_mode = 0;
        const int status = Set_Density_Grid_Hip_LocalDensity(P.output_count, P.S, P.max_no1, P.dm.size(), D.out_ptr,
                                                             D.term_pt, D.pt_pair, D.no0, D.no1, D.hoff, D.noff,
                                                             D.o0off, D.o1off, D.nc, D.o0, D.o1, D.dm, D.den);
        grid_precision_mode = 1;
        CK(hipMemcpy(back.data(), D.dm, sizeof(double) * back.size(), hipMemcpyDeviceToHost));
        expect(status == 1 && std::equal(back.begin(), back.end(), P.dm.begin()),
               "FP64 mode: no kernel, the density matrix untouched");
    }
    release(D);
}

} // namespace check

int main()
{
    using check::run_case;
    setenv("OPENMX_GRID_PRECISION", "df", 1);
    if (!grid_precision_df()) {
        std::printf("FAIL: OPENMX_GRID_PRECISION=df not taken\n");
        return 1;
    }
    std::printf("device LDS per block: %d bytes\n", device_shared_limit());
    /* species: H s2p1 (5), C/O s2p2d1 (13), a transition metal s3p2d2 (19),
       an f element s3p3d3f2 (41); NC with 48 orbitals (s3p3d3f3) exceeds
       the LDS of one block for S = 4, and 80 orbitals the register rows */
    run_case("collinear, unpolarized", 1, {5, 13}, 16, 24, 1500, 3500, 1u, false, 5.0e-14);
    run_case("collinear, polarized", 2, {13, 19}, 12, 20, 1500, 3500, 2u, false, 5.0e-14);
    run_case("non-collinear", 4, {5, 13, 19}, 8, 16, 1200, 3000, 3u, false, 5.0e-14);
    run_case("non-collinear, 48 orbitals", 4, {48, 13}, 4, 6, 800, 1600, 4u, true, 5.0e-14);
    run_case("polarized, 41 and 48 orbitals", 2, {41, 48}, 4, 6, 800, 1600, 5u, false, 5.0e-14);
    /* beyond 64 orbitals: the memory variants (no register row) */
    run_case("non-collinear, 80 orbitals", 4, {80, 13}, 3, 4, 600, 1200, 6u, true, 5.0e-14);
    if (check::failures != 0) {
        std::printf("FAILED: %d check(s)\n", check::failures);
        return 1;
    }
    std::printf("PASS: double-float CSR and on-the-fly density kernels\n");
    return 0;
}

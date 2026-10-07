/* Precision probe of the dense eigensolve of OpenMX's GPU cluster solvers:
   cusolverDnXsyevdx for the MaxN lowest eigenpairs of a sampled matrix, in
   several precisions, each compared with a reference (the first mode given if
   it solves in FP64, else native FP64).

     eigen_precision_probe real    <H.bin> <X.bin> <n>  <maxn> <electrons> <occupancy> <kT> [reps] [modes]
     eigen_precision_probe complex <Hs2.bin> -     <n2> <maxn> <electrons> <occupancy> <kT> [reps] [modes]

   real: the Hamiltonian H and the transformed overlap X of the collinear
   cluster solver as sampled by Cluster_DFT_Col.c (OPENMX_GEMM_SAMPLE_DIR,
   column-major doubles); the probe forms C = X^T H X in FP64, the matrix the
   solver diagonalizes.  complex: Hs2 as sampled by Cluster_DFT_NonCol.c, the
   transformed non-collinear Hamiltonian (column-major, real and imaginary
   parts interleaved).  Both come from GEMMs and are symmetric or Hermitian
   only to rounding; the probe copies the lower triangle, which the
   eigensolver reads, onto the upper one (and zeroes the imaginary part of the
   diagonal) so that every mode, the refinement products included, works on
   one Hermitian matrix.  electrons, the occupancy of a state (2 for a
   spin-unpolarized collinear block, 1 otherwise) and kT (Hartree) set Fermi
   occupations; the chemical potential is found per mode from that mode's
   eigenvalues, as the solver does.

   modes (comma separated, default all available):
     fp64    native FP64 (the reference)
     emu     FP64 emulation with dynamic mantissa control, as OpenMX sets every
             solver handle with cuSOLVER >= 12.2 (CUDA 13)
     emuFb   FP64 emulation with b fixed mantissa bits (e.g. emuF32)
     emuOk   FP64 emulation, dynamic, mantissa bit offset k (e.g. emuO-16)
     fp32    an FP32 copy of the matrix (CHEEVDX / SSYEVDX)
     fp32d   the same through cusolverDnXsyevd (all eigenpairs; a full-range
             fp32 solve goes through Xsyevdx, which OpenMX's compatibility
             layer routes to Xsyevd only for FP64); fp32d0 and fp32d2 with
             the algorithm selector CUSOLVER_ALG_0 (automatic) and ALG_2
             (two-stage), cuSOLVER >= 12.3.4
     fp32oaK  FP32 solve of all n eigenpairs, then K Ogita-Aishima refinement
             steps of all of them (RefSyEv: R = I - X^H X, S = X^H H X,
             lambda_i = s_ii / (1 - r_ii), E_ij = (s_ij + lambda_j r_ij) /
             (lambda_j - lambda_i), or r_ij / 2 for |lambda_j - lambda_i| <=
             delta, X <- X + X E)
     fp32soaK the same for the maxn wanted eigenpairs only, with the other FP32
             eigenvectors as the rest of the correction basis (their
             eigenvalues taken from the FP32 solve): products of n x n x maxn
     fp64oaK, fp64soaK  the same from an FP64 solve, as a reference better than
             the FP64 solve itself (e.g. fp64soa2:fp64:d1e4 first in the list:
             without a large delta scale, pairs split only by rounding, with
             gaps just above delta, get corrections of order one and spoil it)
   The emulation modes need cuSOLVER >= 12.2 and are skipped otherwise.  The
   products of the refinement go through the GEMMul8 bridge
   (openmx_gemmul8Zgemm / Dgemm, settings from OPENMX_GEMMUL8_NUM_MOD_Z/D and
   _FASTMODE_Z/D), or plain cuBLAS FP64 with PROBE_OA_GEMM=fp64; delta is
   PROBE_OA_DELTA_SCALE (default 2) times (largest off-diagonal |s_ij| +
   spectral radius times largest |r_ij|).  With PROBE_OA_OCC=1 a pair whose
   Fermi occupations (from the eigenvalues of the solve) differ by less than
   PROBE_OA_OCC_TOL (default 1e-12) is only orthonormalized, as a cluster: the
   density matrix needs the occupied subspace, not the rotations inside it.
   A refinement mode may override these per mode with suffixes: :fp64 or
   :gemmul8 (products), :occ or :noocc, :d<scale> (e.g. fp32soa1:occ:d10),
   and :rr, which ends the refinement with a Rayleigh-Ritz step inside every
   cluster of the wanted states (consecutive estimates closer than the last
   delta) that holds a partially occupied state: there the refinement only
   orthonormalizes, the vectors stay mixtures, and different occupations
   make the density matrix see it.  The cluster's H and overlap are formed in
   cuBLAS FP64 and the small generalized problem is solved by cuSOLVER.
   :rr<c> chains the clusters with c times the last delta instead (pairs just
   above delta start with mixings of about 0.1, which two Newton steps leave
   at 1e-4), :rf<c> with the larger of c times the first delta and the last
   delta (the last one swings with the overlaps of the unrefined states).
   The report of a refinement step counts the pairs treated as a cluster and
   the corrected pairs with |E_ij| > 0.01 (large_E), for which a Newton step is
   inaccurate.

   Per mode: the median wall time of the eigensolver call over REPS calls
   after one warm-up (the matrix is copied in before each call, untimed; the
   refinement modes add the median time of their refinement), the
   number of eigenpairs, and against the reference the largest eigenvalue
   difference over the MaxN states, the band energy difference sum_i f_i e_i,
   and the largest element (|Re| + |Im| for complex) and the Frobenius norm of
   the difference of the density matrices V diag(f) V^H (dDM) and of the
   energy density matrices V diag(f e) V^H (dEDM), the latter also formed as
   (V diag(f) (A V)^H + A V diag(f) V^H) / 2 (dEDMh), which needs the occupied
   subspace only, not the eigenvectors inside it; both against the reference's
   V diag(f e) V^H (for the reference itself: self_dEDMh).  A first-step solve
   (maxn = n) uses the full range here, whereas OpenMX's compatibility layer
   routes full spectra to cusolverDnXsyevd with cuSOLVER >= 12.3.4. */
#include <cublas_v2.h>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

extern "C" {
cublasStatus_t openmx_gemmul8Dgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m,
                                   int n, int k, const double *alpha, const double *A, int lda, const double *B,
                                   int ldb, const double *beta, double *C, int ldc);
cublasStatus_t openmx_gemmul8Zgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m,
                                   int n, int k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda,
                                   const cuDoubleComplex *B, int ldb, const cuDoubleComplex *beta, cuDoubleComplex *C,
                                   int ldc);
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

void check(cusolverStatus_t status, const char *what)
{
    if (status != CUSOLVER_STATUS_SUCCESS) {
        std::fprintf(stderr, "%s: cuSOLVER status %d\n", what, static_cast<int>(status));
        std::exit(1);
    }
}

/* element-wise precision conversions; a complex matrix is handled as 2 n^2 reals */
__global__ void to_float(const double *in, float *out, size_t count)
{
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < count;
         i += static_cast<size_t>(gridDim.x) * blockDim.x)
        out[i] = static_cast<float>(in[i]);
}

__global__ void to_double(const float *in, double *out, size_t count)
{
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < count;
         i += static_cast<size_t>(gridDim.x) * blockDim.x)
        out[i] = static_cast<double>(in[i]);
}

/* column j of the n x maxn panel (width doubles per element) scaled by s[j] */
__global__ void scale_columns(double *v, const double *s, size_t rows, int cols)
{
    const size_t count = rows * static_cast<size_t>(cols);
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < count;
         i += static_cast<size_t>(gridDim.x) * blockDim.x)
        v[i] *= s[i / rows];
}

/* Make the matrix Hermitian from its lower triangle, the part the eigensolver
   reads (uplo = lower), so that the refinement products see the same matrix:
   the sampled matrices come from GEMMs and are Hermitian only to rounding.
   out receives the largest change as the bit pattern of a non-negative double. */
__global__ void hermitize_from_lower(double *H, int n, bool cplx, unsigned long long *out)
{
    double change = 0.0;
    const size_t count = static_cast<size_t>(n) * n;
    for (size_t at = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; at < count;
         at += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const size_t i = at % n, j = at / n;
        if (i > j) continue;
        if (i == j) {
            if (cplx) {
                change = fmax(change, fabs(H[2 * at + 1]));
                H[2 * at + 1] = 0.0;
            }
            continue;
        }
        const size_t low = j + i * static_cast<size_t>(n); /* element (j, i) */
        if (cplx) {
            change = fmax(change, hypot(H[2 * at] - H[2 * low], H[2 * at + 1] + H[2 * low + 1]));
            H[2 * at] = H[2 * low];
            H[2 * at + 1] = -H[2 * low + 1];
        }
        else {
            change = fmax(change, fabs(H[at] - H[low]));
            H[at] = H[low];
        }
    }
    atomicMax(out, static_cast<unsigned long long>(__double_as_longlong(change)));
}

/* Ogita-Aishima refinement on the n x kc block of the wanted columns.  Matrices
   are column-major with leading dimension n; a complex element is two
   doubles.  G = X^H X(:, 0:kc), S = X^H H X(:, 0:kc). */
__device__ inline double magnitude(const double *m, size_t at, bool cplx)
{
    return cplx ? hypot(m[2 * at], m[2 * at + 1]) : fabs(m[at]);
}

/* lambda_j = Re s_jj / Re g_jj (= s_jj / (1 - r_jj)) for the wanted j < kc */
__global__ void oa_lambda(const double *S, const double *G, int n, int kc, bool cplx, double *lam)
{
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < kc; j += gridDim.x * blockDim.x) {
        const size_t at = static_cast<size_t>(j) + static_cast<size_t>(j) * n;
        lam[j] = cplx ? S[2 * at] / G[2 * at] : S[at] / G[at];
    }
}

/* largest off-diagonal |s_ij| and largest |r_ij| = |delta_ij - g_ij| over the
   block, as the bit patterns of non-negative doubles (ordered like integers) */
__global__ void oa_maxima(const double *S, const double *G, int n, int kc, bool cplx, unsigned long long *out)
{
    double ms = 0.0, mr = 0.0;
    const size_t count = static_cast<size_t>(n) * kc;
    for (size_t at = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; at < count;
         at += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const size_t i = at % n, j = at / n;
        if (i != j) ms = fmax(ms, magnitude(S, at, cplx));
        const double gr = (cplx ? G[2 * at] : G[at]) - (i == j ? 1.0 : 0.0);
        const double gi = cplx ? G[2 * at + 1] : 0.0;
        mr = fmax(mr, hypot(gr, gi));
    }
    atomicMax(&out[0], static_cast<unsigned long long>(__double_as_longlong(ms)));
    atomicMax(&out[1], static_cast<unsigned long long>(__double_as_longlong(mr)));
}

/* E_ij = (s_ij + lambda_j r_ij) / (lambda_j - lambda_i), or r_ij / 2 when the
   two eigenvalues are within delta (and on the diagonal); counts the pairs
   treated as a cluster and the corrected pairs with |E_ij| > 0.01, for which
   one Newton step is not accurate (counters[0], counters[1]) */
__global__ void oa_correction(const double *S, const double *G, const double *lam, double delta, int n, int kc,
                              bool cplx, const double *occ, double occ_tol, double *E, unsigned long long *counters)
{
    unsigned long long local = 0, large = 0;
    const size_t count = static_cast<size_t>(n) * kc;
    for (size_t at = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; at < count;
         at += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const size_t i = at % n, j = at / n;
        const double rr = (i == j ? 1.0 : 0.0) - (cplx ? G[2 * at] : G[at]);
        const double ri = cplx ? -G[2 * at + 1] : 0.0;
        double er = 0.5 * rr, ei = 0.5 * ri;
        if (i != j) {
            const double d = lam[j] - lam[i];
            if (fabs(d) > delta && (occ == nullptr || fabs(occ[i] - occ[j]) >= occ_tol)) {
                er = ((cplx ? S[2 * at] : S[at]) + lam[j] * rr) / d;
                ei = cplx ? (S[2 * at + 1] + lam[j] * ri) / d : 0.0;
                if (hypot(er, ei) > 0.01) large++;
            }
            else {
                local++;
            }
        }
        if (cplx) {
            E[2 * at] = er;
            E[2 * at + 1] = ei;
        }
        else {
            E[at] = er;
        }
    }
    if (local) atomicAdd(&counters[0], local);
    if (large) atomicAdd(&counters[1], large);
}

struct Mode {
    std::string name;
    bool fp32 = false;
    bool emulated = false;
    int fixed_bits = 0;      /* > 0: fixed mantissa control */
    int offset = 0;          /* dynamic control with this offset */
    bool has_offset = false;
    int syevd = -1;          /* >= 0: fp32d: Xsyevd for all pairs, with this algorithm (9: default params) */
    int oa_iters = 0;        /* > 0: solve of all pairs and this many refinement steps */
    bool oa_subset = false;  /* refine the maxn wanted pairs only */
    int oa_gemm = -1;        /* refinement products: 1 cuBLAS FP64, 0 GEMMul8, -1 PROBE_OA_GEMM */
    int oa_occ = -1;         /* occupation clusters: 1 on, 0 off, -1 PROBE_OA_OCC */
    double oa_scale = -1.0;  /* delta scale, < 0: PROBE_OA_DELTA_SCALE */
    bool oa_rr = false;      /* Rayleigh-Ritz inside partially occupied clusters at the end */
    double oa_rr_scale = 1.0; /* their chaining threshold in units of the last delta */
    double oa_rf_scale = 0.0; /* > 0: max(this times the first delta, the last delta) instead */
};

bool parse_mode(const std::string &full, Mode *mode)
{
    mode->name = full;
    const size_t colon = full.find(':');
    const std::string text = full.substr(0, colon);
    if (colon != std::string::npos) {
        /* per-mode options of a refinement mode */
        for (size_t at = colon; at != std::string::npos;) {
            const size_t next = full.find(':', at + 1);
            const std::string option =
                full.substr(at + 1, next == std::string::npos ? std::string::npos : next - at - 1);
            if (option == "fp64") mode->oa_gemm = 1;
            else if (option == "gemmul8") mode->oa_gemm = 0;
            else if (option == "occ") mode->oa_occ = 1;
            else if (option == "noocc") mode->oa_occ = 0;
            else if (option == "rr") mode->oa_rr = true;
            else if (option.size() > 2 && option.compare(0, 2, "rr") == 0 && std::atof(option.c_str() + 2) >= 1.0) {
                mode->oa_rr = true;
                mode->oa_rr_scale = std::atof(option.c_str() + 2);
            }
            else if (option.size() > 2 && option.compare(0, 2, "rf") == 0 && std::atof(option.c_str() + 2) >= 1.0) {
                mode->oa_rr = true;
                mode->oa_rf_scale = std::atof(option.c_str() + 2);
            }
            else if (option.size() > 1 && option[0] == 'd' && std::atof(option.c_str() + 1) >= 0.0)
                mode->oa_scale = std::atof(option.c_str() + 1);
            else return false;
            at = next;
        }
    }
    if (text.size() > 6 && (text.compare(0, 4, "fp32") == 0 || text.compare(0, 4, "fp64") == 0) &&
        (text.compare(4, 2, "oa") == 0 || (text.size() > 7 && text.compare(4, 3, "soa") == 0))) {
        mode->fp32 = text[2] == '3';
        mode->oa_subset = text[4] == 's';
        mode->oa_iters = std::atoi(text.c_str() + (mode->oa_subset ? 7 : 6));
        return mode->oa_iters > 0;
    }
    if (colon != std::string::npos) return false;
    if (text == "fp64") return true;
    if (text == "fp32") { mode->fp32 = true; return true; }
    if (text == "fp32d") { mode->fp32 = true; mode->syevd = 9; return true; }
    if (text == "fp32d0") { mode->fp32 = true; mode->syevd = 0; return true; }
    if (text == "fp32d2") { mode->fp32 = true; mode->syevd = 2; return true; }
    if (text == "emu") { mode->emulated = true; return true; }
    if (text.rfind("emuF", 0) == 0 && text.size() > 4) {
        mode->emulated = true;
        mode->fixed_bits = std::atoi(text.c_str() + 4);
        return mode->fixed_bits > 0;
    }
    if (text.rfind("emuO", 0) == 0 && text.size() > 4) {
        mode->emulated = true;
        mode->has_offset = true;
        mode->offset = std::atoi(text.c_str() + 4);
        return true;
    }
    return false;
}

void configure(cusolverDnHandle_t handle, const Mode &mode)
{
    if (!mode.emulated) return;
#if CUSOLVER_VERSION >= 12200
    check(cusolverDnSetMathMode(handle, CUSOLVER_FP64_EMULATED_FIXEDPOINT_MATH), "cusolverDnSetMathMode");
    check(cusolverDnSetEmulationStrategy(handle, CUDA_EMULATION_STRATEGY_PERFORMANT), "cusolverDnSetEmulationStrategy");
    if (mode.fixed_bits > 0) {
        check(cusolverDnSetFixedPointEmulationMantissaControl(handle, CUDA_EMULATION_MANTISSA_CONTROL_FIXED),
              "cusolverDnSetFixedPointEmulationMantissaControl");
        check(cusolverDnSetFixedPointEmulationMaxMantissaBitCount(handle, mode.fixed_bits),
              "cusolverDnSetFixedPointEmulationMaxMantissaBitCount");
    }
    else {
        check(cusolverDnSetFixedPointEmulationMantissaControl(handle, CUDA_EMULATION_MANTISSA_CONTROL_DYNAMIC),
              "cusolverDnSetFixedPointEmulationMantissaControl");
        if (mode.has_offset)
            check(cusolverDnSetFixedPointEmulationMantissaBitOffset(handle, mode.offset),
                  "cusolverDnSetFixedPointEmulationMantissaBitOffset");
    }
#else
    (void)handle;
#endif
}

std::vector<double> read_matrix(const char *path, size_t count)
{
    std::vector<double> data(count);
    FILE *fp = std::fopen(path, "rb");
    if (fp == nullptr || std::fread(data.data(), sizeof(double), count, fp) != count) {
        std::fprintf(stderr, "cannot read %zu doubles from %s\n", count, path);
        std::exit(1);
    }
    if (std::fgetc(fp) != EOF) {
        std::fprintf(stderr, "%s is larger than %zu doubles: wrong dimension?\n", path, count);
        std::exit(1);
    }
    std::fclose(fp);
    return data;
}

/* Fermi occupations of the first maxn eigenvalues for the given electron
   count; the chemical potential by bisection */
std::vector<double> occupations(const std::vector<double> &e, double electrons, double occupancy, double kT,
                                double *mu_out)
{
    double lo = e.front() - 1.0, hi = e.back() + 1.0, mu = 0.0;
    std::vector<double> f(e.size());
    for (int it = 0; it < 200; it++) {
        mu = 0.5 * (lo + hi);
        double count = 0.0;
        for (size_t i = 0; i < e.size(); i++) {
            const double x = std::max(-700.0, std::min(700.0, (e[i] - mu) / kT));
            count += occupancy / (1.0 + std::exp(x));
        }
        if (count < electrons) lo = mu;
        else hi = mu;
    }
    for (size_t i = 0; i < e.size(); i++) {
        const double x = std::max(-700.0, std::min(700.0, (e[i] - mu) / kT));
        f[i] = occupancy / (1.0 + std::exp(x));
    }
    *mu_out = mu;
    return f;
}

struct Result {
    double ms = 0.0;
    long long meig = 0;
    std::vector<double> eig;
    std::vector<double> occ;
    double mu = 0.0;
    double band = 0.0;
};

} // namespace

int main(int argc, char **argv)
{
    if (argc < 9) {
        std::fprintf(stderr,
                     "usage: %s real|complex <H> <X|-> <n> <maxn> <electrons> <occupancy> <kT> [reps] [modes]\n",
                     argv[0]);
        return 2;
    }
    const bool cplx = std::strcmp(argv[1], "complex") == 0;
    if (!cplx && std::strcmp(argv[1], "real") != 0) {
        std::fprintf(stderr, "the first argument must be real or complex\n");
        return 2;
    }
    const int n = std::atoi(argv[4]), maxn = std::atoi(argv[5]);
    const double electrons = std::atof(argv[6]), occupancy = std::atof(argv[7]), kT = std::atof(argv[8]);
    const int reps = argc > 9 ? std::max(1, std::atoi(argv[9])) : 5;
    std::string mode_list = argc > 10 ? argv[10] : "";
    if (mode_list.empty()) {
#if CUSOLVER_VERSION >= 12200
        mode_list = "fp64,emu,emuO-8,emuO-16,emuF40,emuF32,fp32";
#else
        mode_list = "fp64,fp32";
#endif
    }
    if (n < 1 || maxn < 1 || n < maxn || !(kT > 0.0) || !(occupancy > 0.0) || electrons > occupancy * maxn) {
        std::fprintf(stderr, "invalid sizes or occupation parameters\n");
        return 2;
    }

    std::vector<Mode> modes;
    for (size_t start = 0; start < mode_list.size();) {
        size_t end = mode_list.find(',', start);
        if (end == std::string::npos) end = mode_list.size();
        Mode mode;
        if (!parse_mode(mode_list.substr(start, end - start), &mode)) {
            std::fprintf(stderr, "unknown mode %s\n", mode_list.substr(start, end - start).c_str());
            return 2;
        }
#if CUSOLVER_VERSION < 12200
        if (mode.emulated) {
            std::printf("# %s skipped: cuSOLVER %d has no FP64 emulation\n", mode.name.c_str(), CUSOLVER_VERSION);
            start = end + 1;
            continue;
        }
#endif
        /* the reference: the first mode if it solves in FP64, else native FP64 */
        if (modes.empty() && (mode.fp32 || mode.emulated)) {
            Mode ref;
            parse_mode("fp64", &ref);
            modes.push_back(ref);
        }
        if (modes.empty() || mode.name != modes.front().name) modes.push_back(mode);
        start = end + 1;
    }

    const int width = cplx ? 2 : 1;
    const size_t nn = static_cast<size_t>(n) * n, reals = nn * width;
    const cudaDataType type64 = cplx ? CUDA_C_64F : CUDA_R_64F, type32 = cplx ? CUDA_C_32F : CUDA_R_32F;

    cublasHandle_t blas = nullptr;
    check(cublasCreate(&blas), "cublasCreate");
    double *d_ref = nullptr, *d_a = nullptr, *d_dmref = nullptr, *d_edmref = nullptr, *d_dm = nullptr;
    double *d_panel = nullptr, *d_av = nullptr, *d_scale = nullptr;
    float *d_a32 = nullptr;
    check(cudaMalloc(&d_ref, reals * sizeof(double)), "cudaMalloc matrix");
    check(cudaMalloc(&d_a, reals * sizeof(double)), "cudaMalloc work matrix");
    for (double **b : {&d_dmref, &d_edmref, &d_dm, &d_panel})
        check(cudaMalloc(b, reals * sizeof(double)), "cudaMalloc density matrices");
    check(cudaMalloc(&d_av, static_cast<size_t>(n) * maxn * width * sizeof(double)), "cudaMalloc A V");
    check(cudaMalloc(&d_scale, static_cast<size_t>(maxn) * sizeof(double)), "cudaMalloc column scales");

    {
        std::vector<double> h = read_matrix(argv[2], reals);
        check(cudaMemcpy(d_ref, h.data(), reals * sizeof(double), cudaMemcpyHostToDevice), "upload H");
        if (!cplx) {
            /* C = X^T (H X) in FP64, the matrix the collinear solver diagonalizes */
            std::vector<double> x = read_matrix(argv[3], nn);
            double *d_x = nullptr, *d_t = nullptr;
            const double one = 1.0, zero = 0.0;
            check(cudaMalloc(&d_x, nn * sizeof(double)), "cudaMalloc X");
            check(cudaMalloc(&d_t, nn * sizeof(double)), "cudaMalloc HX");
            check(cudaMemcpy(d_x, x.data(), nn * sizeof(double), cudaMemcpyHostToDevice), "upload X");
            check(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, d_ref, n, d_x, n, &zero, d_t, n), "H X");
            check(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, n, n, n, &one, d_x, n, d_t, n, &zero, d_ref, n), "X^T HX");
            check(cudaFree(d_x), "cudaFree X");
            check(cudaFree(d_t), "cudaFree HX");
        }
    }
    double asymmetry = 0.0;
    {
        unsigned long long *d_change = nullptr, bits = 0;
        check(cudaMalloc(&d_change, sizeof(bits)), "cudaMalloc change");
        check(cudaMemset(d_change, 0, sizeof(bits)), "clear change");
        hermitize_from_lower<<<1024, 256>>>(d_ref, n, cplx, d_change);
        check(cudaMemcpy(&bits, d_change, sizeof(bits), cudaMemcpyDeviceToHost), "download change");
        std::memcpy(&asymmetry, &bits, sizeof(double));
        check(cudaFree(d_change), "cudaFree change");
    }

    int *d_info = nullptr;
    double *d_w = nullptr;
    check(cudaMalloc(&d_info, sizeof(int)), "cudaMalloc info");
    check(cudaMalloc(&d_w, (static_cast<size_t>(n) + 1) * sizeof(double)), "cudaMalloc eigenvalues");
    double vl = 0.0, vu = 0.0;
    float vl32 = 0.0f, vu32 = 0.0f;
    const char *oa_gemm_env = std::getenv("PROBE_OA_GEMM");
    const bool oa_fp64 = oa_gemm_env != nullptr && std::strcmp(oa_gemm_env, "fp64") == 0;
    const char *oa_scale_env = std::getenv("PROBE_OA_DELTA_SCALE");
    const double oa_scale = oa_scale_env != nullptr ? std::atof(oa_scale_env) : 2.0;
    const char *oa_occ_env = std::getenv("PROBE_OA_OCC");
    const bool oa_occ = oa_occ_env != nullptr && std::atoi(oa_occ_env) != 0;
    const char *oa_occ_tol_env = std::getenv("PROBE_OA_OCC_TOL");
    const double oa_occ_tol = oa_occ_tol_env != nullptr ? std::atof(oa_occ_tol_env) : 1e-12;

    std::printf("# %s %s n=%d maxn=%d electrons=%g occupancy=%g kT=%g reps=%d cuSOLVER %d; made Hermitian from the "
                "lower triangle, largest change %.2e\n",
                argv[1], argv[2], n, maxn, electrons, occupancy, kT, reps, CUSOLVER_VERSION, asymmetry);

    Result ref;
    for (const Mode &mode : modes) {
        cusolverDnHandle_t handle = nullptr;
        check(cusolverDnCreate(&handle), "cusolverDnCreate");
        configure(handle, mode);
        /* the refinement modes solve for all n pairs: the rest serve as correction basis */
        const int solved = (mode.oa_iters > 0 || mode.syevd >= 0) ? n : maxn;
        cusolverDnParams_t params = nullptr;
        if (mode.syevd >= 0 && mode.syevd != 9) {
#if CUSOLVER_VERSION >= 12304
            check(cusolverDnCreateParams(&params), "cusolverDnCreateParams");
            check(cusolverDnSetAdvOptions(params, CUSOLVERDN_SYEVD, static_cast<cusolverAlgMode_t>(mode.syevd)),
                  "cusolverDnSetAdvOptions");
#else
            std::printf("# %s skipped: cuSOLVER %d has no algorithm selector\n", mode.name.c_str(), CUSOLVER_VERSION);
            cusolverDnDestroy(handle);
            continue;
#endif
        }
        const int64_t il = 1, iu = solved;
        const cusolverEigRange_t range = (solved == n) ? CUSOLVER_EIG_RANGE_ALL : CUSOLVER_EIG_RANGE_I;
        const cudaDataType type = mode.fp32 ? type32 : type64;
        const cudaDataType wtype = mode.fp32 ? CUDA_R_32F : CUDA_R_64F;
        void *a = mode.fp32 ? static_cast<void *>(d_a32) : static_cast<void *>(d_a);
        if (mode.fp32 && d_a32 == nullptr) {
            check(cudaMalloc(&d_a32, reals * sizeof(float)), "cudaMalloc FP32 matrix");
            a = d_a32;
        }
        void *w = d_w;
        float *d_w32 = nullptr;
        if (mode.fp32) {
            check(cudaMalloc(&d_w32, (static_cast<size_t>(n) + 1) * sizeof(float)), "cudaMalloc FP32 eigenvalues");
            w = d_w32;
        }
        void *pvl = mode.fp32 ? static_cast<void *>(&vl32) : static_cast<void *>(&vl);
        void *pvu = mode.fp32 ? static_cast<void *>(&vu32) : static_cast<void *>(&vu);
        size_t dbytes = 0, hbytes = 0;
        int64_t meig = 0;
        if (mode.syevd >= 0)
            check(cusolverDnXsyevd_bufferSize(handle, params, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, n, type,
                                              a, n, wtype, w, type, &dbytes, &hbytes),
                  "cusolverDnXsyevd_bufferSize");
        else
            check(cusolverDnXsyevdx_bufferSize(handle, nullptr, CUSOLVER_EIG_MODE_VECTOR, range, CUBLAS_FILL_MODE_LOWER,
                                               n, type, a, n, pvl, pvu, il, iu, &meig, wtype, w, type, &dbytes,
                                               &hbytes),
                  "cusolverDnXsyevdx_bufferSize");
        void *d_work = nullptr;
        std::vector<char> h_work(std::max<size_t>(hbytes, 1));
        check(cudaMalloc(&d_work, std::max<size_t>(dbytes, 1)), "cudaMalloc workspace");

        auto load = [&] {
            if (mode.fp32) to_float<<<1024, 256>>>(d_ref, d_a32, reals);
            else check(cudaMemcpy(d_a, d_ref, reals * sizeof(double), cudaMemcpyDeviceToDevice), "copy matrix");
            check(cudaDeviceSynchronize(), "synchronize");
        };
        auto solve = [&] {
            if (mode.syevd >= 0) {
                check(cusolverDnXsyevd(handle, params, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, n, type, a, n,
                                       wtype, w, type, d_work, dbytes, h_work.data(), hbytes, d_info),
                      "cusolverDnXsyevd");
                meig = n;
            }
            else {
                check(cusolverDnXsyevdx(handle, nullptr, CUSOLVER_EIG_MODE_VECTOR, range, CUBLAS_FILL_MODE_LOWER, n,
                                        type, a, n, pvl, pvu, il, iu, &meig, wtype, w, type, d_work, dbytes,
                                        h_work.data(), hbytes, d_info),
                      "cusolverDnXsyevdx");
            }
        };
        std::vector<double> times;
        for (int r = 0; r <= reps; r++) {
            load();
            auto t0 = std::chrono::steady_clock::now();
            solve();
            check(cudaDeviceSynchronize(), "synchronize");
            if (r > 0)
                times.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        }
        int info = 0;
        check(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost), "download info");
        if (info != 0 || meig < solved) {
            std::printf("%s n=%d maxn=%d mode=%s FAILED info=%d meig=%lld\n", argv[2], n, maxn, mode.name.c_str(), info,
                        static_cast<long long>(meig));
            if (&mode == &modes.front()) {
                std::fprintf(stderr, "the FP64 reference failed\n");
                return 1;
            }
            check(cudaFree(d_work), "cudaFree workspace");
            if (d_w32) check(cudaFree(d_w32), "cudaFree FP32 eigenvalues");
            cusolverDnDestroy(handle);
            continue;
        }
        std::sort(times.begin(), times.end());

        Result res;
        res.ms = times[times.size() / 2];
        res.meig = meig;
        res.eig.resize(maxn);
        std::string oa_report;
        if (mode.oa_iters > 0) {
            /* Ogita-Aishima refinement of the FP32 pairs (all n, or the maxn wanted) */
            const int kc = mode.oa_subset ? maxn : n;
            const size_t block = static_cast<size_t>(n) * kc * width;
            double *d_x0 = nullptr, *d_y = nullptr, *d_g = nullptr, *d_s = nullptr, *d_e = nullptr, *d_xn = nullptr;
            double *d_lam = nullptr, *d_occ = nullptr;
            unsigned long long *d_counters = nullptr;
            check(cudaMalloc(&d_x0, reals * sizeof(double)), "cudaMalloc X0");
            for (double **b : {&d_y, &d_g, &d_s, &d_e, &d_xn})
                check(cudaMalloc(b, block * sizeof(double)), "cudaMalloc refinement block");
            check(cudaMalloc(&d_lam, static_cast<size_t>(n) * sizeof(double)), "cudaMalloc lambda");
            check(cudaMalloc(&d_counters, 4 * sizeof(unsigned long long)), "cudaMalloc counters");
            const bool use_fp64 = mode.oa_gemm >= 0 ? mode.oa_gemm == 1 : oa_fp64;
            const bool use_occ = mode.oa_occ >= 0 ? mode.oa_occ == 1 : oa_occ;
            const double scale = mode.oa_scale >= 0.0 ? mode.oa_scale : oa_scale;
            /* the pairs of the solve, all n, in FP64 */
            std::vector<double> lam0(n);
            if (mode.fp32) {
                std::vector<float> w32(n);
                check(cudaMemcpy(w32.data(), d_w32, n * sizeof(float), cudaMemcpyDeviceToHost),
                      "download FP32 eigenvalues");
                lam0.assign(w32.begin(), w32.end());
                to_double<<<1024, 256>>>(d_a32, d_x0, reals);
            }
            else {
                check(cudaMemcpy(lam0.data(), d_w, n * sizeof(double), cudaMemcpyDeviceToHost), "download eigenvalues");
                check(cudaMemcpy(d_x0, d_a, reals * sizeof(double), cudaMemcpyDeviceToDevice), "copy eigenvectors");
            }
            const double anorm = std::max(std::fabs(lam0.front()), std::fabs(lam0.back()));
            if (use_occ) {
                /* occupations of all n states of the solve, for the occupation clusters */
                double mu0 = 0.0;
                const std::vector<double> f0 = occupations(lam0, electrons, occupancy, kT, &mu0);
                check(cudaMalloc(&d_occ, static_cast<size_t>(n) * sizeof(double)), "cudaMalloc occupations");
                check(cudaMemcpy(d_occ, f0.data(), n * sizeof(double), cudaMemcpyHostToDevice), "upload occupations");
            }
            check(cudaDeviceSynchronize(), "synchronize");

            const cublasOperation_t herm = cplx ? CUBLAS_OP_C : CUBLAS_OP_T;
            auto gemm = [&](cublasOperation_t op_a, int m, int nc, int k, const double *A, const double *B, bool add,
                            double *C) {
                if (cplx) {
                    const cuDoubleComplex one = make_cuDoubleComplex(1.0, 0.0);
                    const cuDoubleComplex beta = make_cuDoubleComplex(add ? 1.0 : 0.0, 0.0);
                    check(openmx_gemmul8Zgemm(blas, op_a, CUBLAS_OP_N, m, nc, k, &one,
                                              reinterpret_cast<const cuDoubleComplex *>(A), n,
                                              reinterpret_cast<const cuDoubleComplex *>(B), n, &beta,
                                              reinterpret_cast<cuDoubleComplex *>(C), n), "refinement GEMM");
                }
                else {
                    const double one = 1.0, beta = add ? 1.0 : 0.0;
                    check(openmx_gemmul8Dgemm(blas, op_a, CUBLAS_OP_N, m, nc, k, &one, A, n, B, n, &beta, C, n),
                          "refinement GEMM");
                }
            };
            std::vector<double> oa_times;
            for (int r = 0; r <= reps; r++) {
                check(cudaMemcpy(d_a, d_x0, reals * sizeof(double), cudaMemcpyDeviceToDevice), "restore X");
                check(cudaMemcpy(d_lam, lam0.data(), n * sizeof(double), cudaMemcpyHostToDevice), "upload lambda");
                check(cudaDeviceSynchronize(), "synchronize");
                if (use_fp64) openmx_gemmul8SetEnabled(0);
                auto t0 = std::chrono::steady_clock::now();
                std::string report;
                double last_delta = 0.0, first_delta = 0.0;
                for (int it = 0; it < mode.oa_iters; it++) {
                    gemm(CUBLAS_OP_N, n, kc, n, d_ref, d_a, false, d_y);    /* Y = H X1 */
                    gemm(herm, n, kc, n, d_a, d_a, false, d_g);             /* G = X^H X1 */
                    gemm(herm, n, kc, n, d_a, d_y, false, d_s);             /* S = X^H Y */
                    oa_lambda<<<256, 256>>>(d_s, d_g, n, kc, cplx, d_lam);
                    check(cudaMemset(d_counters, 0, 4 * sizeof(unsigned long long)), "clear counters");
                    oa_maxima<<<1024, 256>>>(d_s, d_g, n, kc, cplx, d_counters);
                    unsigned long long bits[2];
                    check(cudaMemcpy(bits, d_counters, sizeof(bits), cudaMemcpyDeviceToHost), "download maxima");
                    double max_s, max_r;
                    std::memcpy(&max_s, &bits[0], sizeof(double));
                    std::memcpy(&max_r, &bits[1], sizeof(double));
                    const double delta = scale * (max_s + anorm * max_r);
                    last_delta = delta;
                    if (it == 0) first_delta = delta;
                    oa_correction<<<1024, 256>>>(d_s, d_g, d_lam, delta, n, kc, cplx, d_occ, oa_occ_tol, d_e,
                                                 d_counters + 2);
                    check(cudaMemcpy(d_xn, d_a, block * sizeof(double), cudaMemcpyDeviceToDevice), "copy X1");
                    gemm(CUBLAS_OP_N, n, kc, n, d_a, d_e, true, d_xn);      /* X1 += X E */
                    check(cudaMemcpy(d_a, d_xn, block * sizeof(double), cudaMemcpyDeviceToDevice), "update X1");
                    unsigned long long pairs[2];
                    check(cudaMemcpy(pairs, d_counters + 2, sizeof(pairs), cudaMemcpyDeviceToHost),
                          "download pair counts");
                    char buf[192];
                    std::snprintf(buf, sizeof(buf), " it%d:max_r=%.1e,max_s=%.1e,delta=%.1e,cluster_pairs=%llu,large_E=%llu",
                                  it + 1, max_r, max_s, delta, pairs[0], pairs[1]);
                    report += buf;
                }
                if (mode.oa_rr) {
                    /* Rayleigh-Ritz inside the clusters with a partially occupied state */
                    std::vector<double> lam_h(kc);
                    check(cudaMemcpy(lam_h.data(), d_lam, kc * sizeof(double), cudaMemcpyDeviceToHost),
                          "download lambda");
                    double mu_rr = 0.0;
                    const std::vector<double> f_rr = occupations(lam_h, electrons, occupancy, kT, &mu_rr);
                    int clusters = 0, largest = 0;
                    const double chain = mode.oa_rf_scale > 0.0 ? std::max(mode.oa_rf_scale * first_delta, last_delta)
                                                                : mode.oa_rr_scale * last_delta;
                    for (int c0 = 0; c0 < kc;) {
                        int c1 = c0;
                        while (c1 + 1 < kc && std::fabs(lam_h[c1 + 1] - lam_h[c1]) <= chain) c1++;
                        const int m = c1 - c0 + 1;
                        bool partial = false;
                        for (int j = c0; j <= c1; j++)
                            partial = partial || (f_rr[j] > oa_occ_tol * occupancy &&
                                                  f_rr[j] < (1.0 - oa_occ_tol) * occupancy);
                        if (m >= 2 && partial) {
                            const size_t col = static_cast<size_t>(c0) * n * width;
                            double *xc = d_a + col;                 /* the cluster's columns of X1 */
                            double *z = d_y;                        /* H X_C, then X_C Y */
                            double *hm = d_g, *nm = d_g + static_cast<size_t>(m) * m * width;
                            double *w = d_e;
                            int *d_rr_info = nullptr;
                            check(cudaMalloc(&d_rr_info, sizeof(int)), "cudaMalloc info");
                            if (cplx) {
                                const cuDoubleComplex one = make_cuDoubleComplex(1.0, 0.0);
                                const cuDoubleComplex zero = make_cuDoubleComplex(0.0, 0.0);
                                auto zc = [](double *p) { return reinterpret_cast<cuDoubleComplex *>(p); };
                                check(cublasZgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, n, &one, zc(d_ref), n, zc(xc), n,
                                                  &zero, zc(z), n), "H X_C");
                                check(cublasZgemm(blas, CUBLAS_OP_C, CUBLAS_OP_N, m, m, n, &one, zc(xc), n, zc(z), n,
                                                  &zero, zc(hm), m), "X_C^H H X_C");
                                check(cublasZgemm(blas, CUBLAS_OP_C, CUBLAS_OP_N, m, m, n, &one, zc(xc), n, zc(xc), n,
                                                  &zero, zc(nm), m), "X_C^H X_C");
                                int lwork = 0;
                                check(cusolverDnZhegvd_bufferSize(handle, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR,
                                                                  CUBLAS_FILL_MODE_LOWER, m, zc(hm), m, zc(nm), m, w,
                                                                  &lwork), "zhegvd buffer");
                                cuDoubleComplex *work = nullptr;
                                check(cudaMalloc(&work, std::max(lwork, 1) * sizeof(cuDoubleComplex)), "cudaMalloc zhegvd");
                                check(cusolverDnZhegvd(handle, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR,
                                                       CUBLAS_FILL_MODE_LOWER, m, zc(hm), m, zc(nm), m, w, work, lwork,
                                                       d_rr_info), "zhegvd");
                                check(cudaFree(work), "cudaFree zhegvd");
                                check(cublasZgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, m, &one, zc(xc), n, zc(hm), m,
                                                  &zero, zc(z), n), "X_C Y");
                            }
                            else {
                                const double one = 1.0, zero = 0.0;
                                check(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, n, &one, d_ref, n, xc, n, &zero,
                                                  z, n), "H X_C");
                                check(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, m, m, n, &one, xc, n, z, n, &zero, hm,
                                                  m), "X_C^T H X_C");
                                check(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, m, m, n, &one, xc, n, xc, n, &zero, nm,
                                                  m), "X_C^T X_C");
                                int lwork = 0;
                                check(cusolverDnDsygvd_bufferSize(handle, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR,
                                                                  CUBLAS_FILL_MODE_LOWER, m, hm, m, nm, m, w, &lwork),
                                      "dsygvd buffer");
                                double *work = nullptr;
                                check(cudaMalloc(&work, std::max(lwork, 1) * sizeof(double)), "cudaMalloc dsygvd");
                                check(cusolverDnDsygvd(handle, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR,
                                                       CUBLAS_FILL_MODE_LOWER, m, hm, m, nm, m, w, work, lwork,
                                                       d_rr_info), "dsygvd");
                                check(cudaFree(work), "cudaFree dsygvd");
                                check(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, m, &one, xc, n, hm, m, &zero, z,
                                                  n), "X_C Y");
                            }
                            int rr_info = 0;
                            check(cudaMemcpy(&rr_info, d_rr_info, sizeof(int), cudaMemcpyDeviceToHost), "download info");
                            check(cudaFree(d_rr_info), "cudaFree info");
                            if (rr_info != 0) {
                                std::fprintf(stderr, "cluster Rayleigh-Ritz failed: info %d (m=%d)\n", rr_info, m);
                                std::exit(1);
                            }
                            check(cudaMemcpy(xc, z, static_cast<size_t>(n) * m * width * sizeof(double),
                                             cudaMemcpyDeviceToDevice), "rotate X_C");
                            check(cudaMemcpy(lam_h.data() + c0, w, m * sizeof(double), cudaMemcpyDeviceToHost),
                                  "download cluster eigenvalues");
                            clusters++;
                            largest = std::max(largest, m);
                        }
                        c0 = c1 + 1;
                    }
                    check(cudaMemcpy(d_lam, lam_h.data(), kc * sizeof(double), cudaMemcpyHostToDevice), "upload lambda");
                    char buf[96];
                    std::snprintf(buf, sizeof(buf), " rr:clusters=%d,largest=%d,chain=%.1e", clusters, largest, chain);
                    report += buf;
                }
                check(cudaDeviceSynchronize(), "synchronize");
                if (use_fp64) openmx_gemmul8SetEnabled(1);
                if (r > 0)
                    oa_times.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
                oa_report = report;
            }
            std::sort(oa_times.begin(), oa_times.end());
            const double t_oa = oa_times[oa_times.size() / 2];
            char buf[160];
            std::snprintf(buf, sizeof(buf), " solve_ms=%.1f refine_ms=%.1f gemm=%s delta_scale=%g occ=%s", res.ms,
                          t_oa, use_fp64 ? "fp64" : "gemmul8", scale, use_occ ? "on" : "off");
            oa_report = std::string(buf) + oa_report;
            res.ms += t_oa;
            check(cudaMemcpy(res.eig.data(), d_lam, maxn * sizeof(double), cudaMemcpyDeviceToHost), "download lambda");
            for (double *b : {d_x0, d_y, d_g, d_s, d_e, d_xn, d_lam}) check(cudaFree(b), "cudaFree refinement");
            if (d_occ) check(cudaFree(d_occ), "cudaFree occupations");
            check(cudaFree(d_counters), "cudaFree counters");
        }
        else if (mode.fp32) {
            std::vector<float> e32(maxn);
            check(cudaMemcpy(e32.data(), d_w32, maxn * sizeof(float), cudaMemcpyDeviceToHost), "download eigenvalues");
            for (int i = 0; i < maxn; i++) res.eig[i] = e32[i];
        }
        else {
            check(cudaMemcpy(res.eig.data(), d_w, maxn * sizeof(double), cudaMemcpyDeviceToHost), "download eigenvalues");
        }
        res.occ = occupations(res.eig, electrons, occupancy, kT, &res.mu);
        for (int i = 0; i < maxn; i++) res.band += res.occ[i] * res.eig[i];

        /* the eigenvectors in FP64, the first maxn columns */
        double *vec = d_a;
        if (mode.fp32 && mode.oa_iters == 0) to_double<<<1024, 256>>>(d_a32, d_a, reals);
        check(cudaDeviceSynchronize(), "synchronize");

        /* V diag(c) into the panel */
        auto scaled_panel = [&](const std::vector<double> &c) {
            check(cudaMemcpy(d_scale, c.data(), maxn * sizeof(double), cudaMemcpyHostToDevice), "upload scales");
            check(cudaMemcpy(d_panel, vec, static_cast<size_t>(maxn) * n * width * sizeof(double),
                             cudaMemcpyDeviceToDevice), "copy panel");
            scale_columns<<<1024, 256>>>(d_panel, d_scale, static_cast<size_t>(n) * width, maxn);
        };
        /* C = alpha A op(B) + beta C in cuBLAS FP64, n x n x maxn unless stated */
        auto product = [&](cublasOperation_t opb, int m, int nc, int k, double alpha, const double *A,
                           const double *B, double beta, double *C) {
            if (cplx) {
                const cuDoubleComplex a = make_cuDoubleComplex(alpha, 0.0), b = make_cuDoubleComplex(beta, 0.0);
                check(cublasZgemm(blas, CUBLAS_OP_N, opb == CUBLAS_OP_T ? CUBLAS_OP_C : opb, m, nc, k, &a,
                                  reinterpret_cast<const cuDoubleComplex *>(A), n,
                                  reinterpret_cast<const cuDoubleComplex *>(B), n, &b,
                                  reinterpret_cast<cuDoubleComplex *>(C), n), "density matrix product");
            }
            else {
                check(cublasDgemm(blas, CUBLAS_OP_N, opb, m, nc, k, &alpha, A, n, B, n, &beta, C, n),
                      "density matrix product");
            }
        };
        /* m - reference in place, then its largest element (|Re| + |Im|) and Frobenius norm */
        auto difference = [&](double *m, const double *reference, double *largest, double *norm) {
            const double minus = -1.0;
            int at = 0;
            check(cublasDaxpy(blas, static_cast<int>(reals), &minus, reference, 1, m, 1), "difference");
            if (cplx) {
                check(cublasDznrm2(blas, static_cast<int>(nn), reinterpret_cast<cuDoubleComplex *>(m), 1, norm),
                      "difference norm");
                check(cublasIzamax(blas, static_cast<int>(nn), reinterpret_cast<cuDoubleComplex *>(m), 1, &at),
                      "difference max");
                double z[2];
                check(cudaMemcpy(z, m + 2 * (static_cast<size_t>(at) - 1), sizeof(z), cudaMemcpyDeviceToHost),
                      "download difference max");
                *largest = std::fabs(z[0]) + std::fabs(z[1]);
            }
            else {
                check(cublasDnrm2(blas, static_cast<int>(nn), m, 1, norm), "difference norm");
                check(cublasIdamax(blas, static_cast<int>(nn), m, 1, &at), "difference max");
                check(cudaMemcpy(largest, m + (static_cast<size_t>(at) - 1), sizeof(double), cudaMemcpyDeviceToHost),
                      "download difference max");
                *largest = std::fabs(*largest);
            }
        };
        const bool is_ref = &mode == &modes.front();
        std::vector<double> c(maxn);

        /* density matrix V diag(f) V^H = W W^H with W = V diag(sqrt f) */
        for (int i = 0; i < maxn; i++) c[i] = std::sqrt(std::max(0.0, res.occ[i]));
        scaled_panel(c);
        double *dm = is_ref ? d_dmref : d_dm;
        product(CUBLAS_OP_T, n, n, maxn, 1.0, d_panel, d_panel, 0.0, dm);
        double dm_max = 0.0, dm_norm = 0.0;
        if (!is_ref) difference(dm, d_dmref, &dm_max, &dm_norm);

        /* energy density matrix V diag(f e) V^H */
        for (int i = 0; i < maxn; i++) c[i] = res.occ[i] * res.eig[i];
        scaled_panel(c);
        double *edm = is_ref ? d_edmref : d_dm;
        product(CUBLAS_OP_T, n, n, maxn, 1.0, d_panel, vec, 0.0, edm);
        double edm_max = 0.0, edm_norm = 0.0;
        if (!is_ref) difference(edm, d_edmref, &edm_max, &edm_norm);

        /* the same from the subspace: (V diag(f) (A V)^H + A V diag(f) V^H) / 2 */
        for (int i = 0; i < maxn; i++) c[i] = res.occ[i];
        scaled_panel(c);
        product(CUBLAS_OP_N, n, maxn, n, 1.0, d_ref, vec, 0.0, d_av);
        product(CUBLAS_OP_T, n, n, maxn, 0.5, d_panel, d_av, 0.0, d_dm);
        product(CUBLAS_OP_T, n, n, maxn, 0.5, d_av, d_panel, 1.0, d_dm);
        double edmh_max = 0.0, edmh_norm = 0.0;
        difference(d_dm, d_edmref, &edmh_max, &edmh_norm);

        if (is_ref) {
            ref = res;
            std::printf("%s n=%d maxn=%d mode=%s ms=%.1f meig=%lld mu=%.10f band=%.12f self_dEDMh_max=%.2e "
                        "self_dEDMh_F=%.2e%s\n",
                        argv[2], n, maxn, mode.name.c_str(), res.ms, res.meig, res.mu, res.band, edmh_max, edmh_norm,
                        oa_report.c_str());
        }
        else {
            double de = 0.0;
            for (int i = 0; i < maxn; i++) de = std::max(de, std::fabs(res.eig[i] - ref.eig[i]));
            std::printf("%s n=%d maxn=%d mode=%s ms=%.1f speedup=%.2f meig=%lld max_dE=%.2e dmu=%.2e dband=%.2e "
                        "dDM_max=%.2e dDM_F=%.2e dEDM_max=%.2e dEDM_F=%.2e dEDMh_max=%.2e dEDMh_F=%.2e%s\n",
                        argv[2], n, maxn, mode.name.c_str(), res.ms, ref.ms / res.ms, res.meig, de, res.mu - ref.mu,
                        res.band - ref.band, dm_max, dm_norm, edm_max, edm_norm, edmh_max, edmh_norm,
                        oa_report.c_str());
        }
        std::fflush(stdout);
        check(cudaFree(d_work), "cudaFree workspace");
        if (d_w32) check(cudaFree(d_w32), "cudaFree FP32 eigenvalues");
        if (params) cusolverDnDestroyParams(params);
        cusolverDnDestroy(handle);
    }

    cublasDestroy(blas);
    cudaFree(d_ref);
    cudaFree(d_a);
    cudaFree(d_a32);
    cudaFree(d_dmref);
    cudaFree(d_edmref);
    cudaFree(d_dm);
    cudaFree(d_panel);
    cudaFree(d_av);
    cudaFree(d_scale);
    cudaFree(d_info);
    cudaFree(d_w);
    return 0;
}

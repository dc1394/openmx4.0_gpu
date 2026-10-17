/* The device stages of the non-collinear cluster solver's root dense path
   (source/Cluster_DFT_NonCol_dm_hip.cu), in isolation, on random matrices of
   the solver's shapes.  HIP port and extension of the NVIDIA version's
   tests/nc_back_transform_probe.cu.

   back    The back transform: the solver's device form (transpose plus two
           real GEMMs, ClusterNonCol_BackTransformDevice_HIP, unblocked
           GEMMul8), the host-staged form (ClusterNonCol_BackTransformHost_HIP,
           capped GEMMul8) and, with --old (n2 x n2 buffers of its own), the
           former complex GEMM against diag(S, S); each compared with a
           long-double reference on a sample of states (err: largest absolute
           difference over the largest absolute reference value).
   forward The forward stage: the gathered entries scattered into a dense
           matrix (ClusterNonCol_ScatterDense_HIP) and Hs2 assembled from six
           real blocks (ClusterNonCol_AssembleHs2_HIP) with the blocks of the
           right half in the left half of the destination, as the solver
           places them, both compared bit for bit with the host builds; and
           the timings of the six forward-transform products the solver
           runs on the device against the former host-staged transform (a
           handle, three buffers, and an upload, two products and a download
           per block).
   dm      The density matrix (6 components, 10 with --edm) of a sparse
           pattern of atom pairs (13 orbitals per atom, a window of
           neighbours): the per-entry sums (ClusterNonCol_CalcDMRootDenseDevice_HIP)
           and the products (ClusterNonCol_CalcDMRootDenseGemm_HIP,
           OPENMX_CLUSTER_GPU_DM_GEMM), timed (median of REPS after one
           warm-up) and compared with a long-double reference on a sample of
           entries.

     nc_back_transform_probe_hip <n> <maxn> [nk] [reps] [--fp64] [--old] [--edm]
                                 [--only back|forward|dm] [--neighbors K] [--label NAME]

   n2 = 2n; nk (default maxn/2) occupied states; --fp64 sends every product
   to plain hipBLAS FP64 (scf.gemmul8.enable off); otherwise the GEMMul8
   settings come from the bridge's environment variables.  The device
   memory it takes is printed first; keep it below what the GPU has free.

   Build: tools/run_nc_back_transform_probe_hip.sh. */
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

typedef struct {
    double r;
    double i;
} dcomplex;

extern "C" {
hipblasStatus_t openmx_gemmul8Zgemm(hipblasHandle_t, hipblasOperation_t, hipblasOperation_t, int, int, int,
                                    const hipDoubleComplex *, const hipDoubleComplex *, int, const hipDoubleComplex *,
                                    int, const hipDoubleComplex *, hipDoubleComplex *, int);
hipblasStatus_t openmx_gemmul8DgemmSolve(hipblasHandle_t, hipblasOperation_t, hipblasOperation_t, int, int, int,
                                         const double *, const double *, int, const double *, int, const double *,
                                         double *, int);
hipblasStatus_t openmx_gemmul8Dgemm(hipblasHandle_t, hipblasOperation_t, hipblasOperation_t, int, int, int,
                                    const double *, const double *, int, const double *, int, const double *, double *,
                                    int);
void openmx_gemmul8ReleaseWorkspaces(void);
void openmx_gemmul8SetEnabled(int enabled);
long long openmx_gemmul8NativeCalls(int is_complex);

int ClusterNonCol_BackTransformDevice_HIP(hipblasHandle_t handle, int n, int n2, int maxn, void *d_A, void *d_vt,
                                          const double *d_S);
int ClusterNonCol_BackTransformHost_HIP(hipblasHandle_t handle, int n, int n2, int maxn, const dcomplex *V,
                                        const double *S, dcomplex *evec, int ld);
int ClusterNonCol_ScatterDense_HIP(hipblasHandle_t handle, int tnum, int n, const double *values, const int *index,
                                   double *d_values, int *d_index, double *d_H);
int ClusterNonCol_AssembleHs2_HIP(hipblasHandle_t handle, int n, const double *r11, const double *i11,
                                  const double *r12, const double *i12, const double *r22, const double *i22,
                                  void *d_A);
int ClusterNonCol_CalcDMRootDenseDevice_HIP(int entry_count, int size_H1, int n, int n2, int nk_occ, int calc_edm,
                                            const int *basis0, const int *basis1, const double *occ,
                                            const double *occ_e, const dcomplex *d_dense_evec, int ld,
                                            void *d_scratch, size_t scratch_bytes, double *dm_buffer);
int ClusterNonCol_CalcDMRootDenseGemm_HIP(hipblasHandle_t handle, int entry_count, int size_H1, int n, int n2,
                                          int nk_occ, int calc_edm, const int *basis0, const int *basis1,
                                          const double *occ, const double *occ_e, const dcomplex *d_V, int ld,
                                          void *d_scratch, size_t scratch_bytes, double *dm_buffer);
}

namespace {

void check(hipError_t status, const char *what)
{
    if (status != hipSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(status));
        std::exit(1);
    }
}

void check(hipblasStatus_t status, const char *what)
{
    if (status != HIPBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "%s: hipBLAS status %d\n", what, static_cast<int>(status));
        std::exit(1);
    }
}

void check_rc(int rc, const char *what)
{
    if (rc != 0) {
        std::fprintf(stderr, "%s failed (%d)\n", what, rc);
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
double median_ms(int reps, Body body)
{
    std::vector<double> times;

    body(); /* warm-up */
    for (int r = 0; r < reps; r++) {
        check(hipDeviceSynchronize(), "synchronize");
        auto t0 = std::chrono::steady_clock::now();
        body();
        check(hipDeviceSynchronize(), "synchronize");
        times.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}

void *device_alloc(size_t bytes, const char *what)
{
    void *ptr = nullptr;
    check(hipMalloc(&ptr, bytes), what);
    return ptr;
}

/* ------------------------------------------------------------------ */

int run_back(hipblasHandle_t handle, int n, int maxn, int reps, bool old_form, const char *label)
{
    const int n2 = 2 * n;
    const size_t nn = static_cast<size_t>(n) * n, n2n2 = static_cast<size_t>(n2) * n2;
    const size_t panel = static_cast<size_t>(maxn) * n2;
    std::vector<double> h_s(nn);
    std::vector<dcomplex> h_v(n2n2), h_dev(panel), h_host(panel);
    uint64_t state = 0x9E3779B97F4A7C15ULL ^ (static_cast<uint64_t>(n) << 20) ^ static_cast<uint64_t>(maxn);

    for (auto &x : h_s) x = uniform(state);
    for (auto &z : h_v) {
        z.r = uniform(state);
        z.i = uniform(state);
    }

    dcomplex *d_a = static_cast<dcomplex *>(device_alloc(n2n2 * sizeof(dcomplex), "hipMalloc A"));
    dcomplex *d_vt = static_cast<dcomplex *>(device_alloc(panel * sizeof(dcomplex), "hipMalloc vt"));
    double *d_s = static_cast<double *>(device_alloc(nn * sizeof(double), "hipMalloc S"));
    check(hipMemcpy(d_s, h_s.data(), nn * sizeof(double), hipMemcpyHostToDevice), "upload S");

    auto device_form = [&] {
        check(hipMemcpy(d_a, h_v.data(), n2n2 * sizeof(dcomplex), hipMemcpyHostToDevice), "upload V");
        check_rc(ClusterNonCol_BackTransformDevice_HIP(handle, n, n2, maxn, d_a, d_vt, d_s), "device back transform");
    };
    auto host_form = [&] {
        check_rc(ClusterNonCol_BackTransformHost_HIP(handle, n, n2, maxn, h_v.data(), h_s.data(), h_host.data(), maxn),
                 "host-staged back transform");
    };
    /* the timings include the upload of V for the device form and every
       transfer for the host-staged one */
    const double t_dev = median_ms(reps, device_form);
    check(hipMemcpy2D(h_dev.data(), maxn * sizeof(dcomplex), d_a, n2 * sizeof(dcomplex), maxn * sizeof(dcomplex), n2,
                      hipMemcpyDeviceToHost), "download device form");
    const double t_host = median_ms(reps, host_form);

    double t_old = -1.0;
    std::vector<dcomplex> h_old;
    if (old_form) {
        std::vector<dcomplex> h_ss2(n2n2, dcomplex{0.0, 0.0});
        for (int col = 0; col < n; col++) {
            for (int row = 0; row < n; row++) {
                const dcomplex s = {h_s[row + static_cast<size_t>(col) * n], 0.0};
                h_ss2[row + static_cast<size_t>(col) * n2] = s;
                h_ss2[(row + n) + static_cast<size_t>(col + n) * n2] = s;
            }
        }
        dcomplex *d_ss2 = static_cast<dcomplex *>(device_alloc(n2n2 * sizeof(dcomplex), "hipMalloc Ss2"));
        dcomplex *d_old = static_cast<dcomplex *>(device_alloc(n2n2 * sizeof(dcomplex), "hipMalloc C old"));
        check(hipMemcpy(d_ss2, h_ss2.data(), n2n2 * sizeof(dcomplex), hipMemcpyHostToDevice), "upload Ss2");
        check(hipMemcpy(d_a, h_v.data(), n2n2 * sizeof(dcomplex), hipMemcpyHostToDevice), "upload V");
        const hipDoubleComplex one = make_hipDoubleComplex(1.0, 0.0), zero = make_hipDoubleComplex(0.0, 0.0);
        t_old = median_ms(reps, [&] {
            check(openmx_gemmul8Zgemm(handle, HIPBLAS_OP_T, HIPBLAS_OP_T, maxn, n2, n2, &one,
                                      reinterpret_cast<const hipDoubleComplex *>(d_a), n2,
                                      reinterpret_cast<const hipDoubleComplex *>(d_ss2), n2, &zero,
                                      reinterpret_cast<hipDoubleComplex *>(d_old), n2), "old back transform");
        });
        h_old.resize(panel);
        check(hipMemcpy2D(h_old.data(), maxn * sizeof(dcomplex), d_old, n2 * sizeof(dcomplex), maxn * sizeof(dcomplex),
                          n2, hipMemcpyDeviceToHost), "download old form");
        (void)hipFree(d_old);
        (void)hipFree(d_ss2);
    }

    /* the reference on a sample of states, in long double */
    const int samples = std::min(maxn, 48);
    double max_ref = 0.0, err_dev = 0.0, err_host = 0.0, err_old = 0.0;
    for (int t = 0; t < samples; t++) {
        const int l = static_cast<int>((static_cast<long long>(t) * 7919) % maxn);
        for (int i = 0; i < n2; i++) {
            const int half = i / n, ii = i % n;
            long double re = 0.0L, im = 0.0L;
            for (int jj = 0; jj < n; jj++) {
                const dcomplex v = h_v[(half * static_cast<size_t>(n) + jj) + static_cast<size_t>(l) * n2];
                const long double s = h_s[ii + static_cast<size_t>(jj) * n];
                re += s * v.r;
                im += s * v.i;
            }
            const size_t at = l + static_cast<size_t>(i) * maxn;
            max_ref = std::max(max_ref, static_cast<double>(std::hypot(re, im)));
            err_dev = std::max(err_dev, static_cast<double>(std::hypot(h_dev[at].r - re, h_dev[at].i - im)));
            err_host = std::max(err_host, static_cast<double>(std::hypot(h_host[at].r - re, h_host[at].i - im)));
            if (old_form) err_old = std::max(err_old, static_cast<double>(std::hypot(h_old[at].r - re, h_old[at].i - im)));
        }
    }
    const double scale = max_ref > 0.0 ? 1.0 / max_ref : 1.0;
    std::printf("back n=%d n2=%d maxn=%d %s device_ms=%.3f host_staged_ms=%.3f old_ms=%.3f err_device=%.2e "
                "err_host_staged=%.2e err_old=%.2e (%d sampled states)\n",
                n, n2, maxn, label, t_dev, t_host, t_old, err_dev * scale, err_host * scale,
                old_form ? err_old * scale : -1.0, samples);
    (void)hipFree(d_s);
    (void)hipFree(d_vt);
    (void)hipFree(d_a);
    return (err_dev * scale < 1e-12 && err_host * scale < 1e-12) ? 0 : 1;
}

/* ------------------------------------------------------------------ */

int run_forward(hipblasHandle_t handle, int n, int reps, const char *label)
{
    const int n2 = 2 * n;
    const size_t nn = static_cast<size_t>(n) * n, n2n2 = static_cast<size_t>(n2) * n2;
    uint64_t state = 0xD1B54A32D192ED03ULL ^ static_cast<uint64_t>(n);
    int failures = 0;

    /* the scatter: a sparse pattern of 13 x 13 atom blocks, each position once */
    const int orbitals = 13, atoms = n / orbitals, window = 20;
    std::vector<int> index;
    std::vector<double> values;
    for (int a = 0; a < atoms; a++) {
        for (int b = std::max(0, a - window); b <= std::min(atoms - 1, a + window); b++) {
            for (int i = 0; i < orbitals; i++) {
                for (int j = 0; j < orbitals; j++) {
                    index.push_back((a * orbitals + i) + (b * orbitals + j) * n);
                    values.push_back(uniform(state));
                }
            }
        }
    }
    const int tnum = static_cast<int>(index.size());
    std::vector<double> h_ref(nn, 0.0), h_got(nn);
    for (int p = 0; p < tnum; p++) h_ref[index[p]] += values[p];

    /* the solver's layout: eight n x n slots in the n2 x n2 complex buffer,
       the staging in the last two, the H11 blocks outside */
    double *d_a = static_cast<double *>(device_alloc(n2n2 * sizeof(dcomplex), "hipMalloc A"));
    double *d_out = static_cast<double *>(device_alloc(2 * nn * sizeof(double), "hipMalloc H11 blocks"));
    double *slot[8];
    for (int k = 0; k < 8; k++) slot[k] = d_a + static_cast<size_t>(k) * nn;
    if (static_cast<size_t>(tnum) > nn) {
        std::fprintf(stderr, "forward: the pattern exceeds the staging\n");
        return 1;
    }
    const double t_scatter = median_ms(reps, [&] {
        check_rc(ClusterNonCol_ScatterDense_HIP(handle, tnum, n, values.data(), index.data(), slot[6],
                                                reinterpret_cast<int *>(slot[7]), slot[4]), "scatter");
    });
    check(hipMemcpy(h_got.data(), slot[4], nn * sizeof(double), hipMemcpyDeviceToHost), "download scatter");
    const bool scatter_same = (std::memcmp(h_got.data(), h_ref.data(), nn * sizeof(double)) == 0);
    failures += !scatter_same;

    /* the assembly from six random blocks, compared with the host loop */
    std::vector<double> blocks(6 * nn);
    for (auto &x : blocks) x = uniform(state);
    const double *r11 = &blocks[0], *r12 = &blocks[nn], *r22 = &blocks[2 * nn];
    const double *i11 = &blocks[3 * nn], *i12 = &blocks[4 * nn], *i22 = &blocks[5 * nn];
    std::vector<dcomplex> hs2(n2n2), got(n2n2);
    for (int col = 0; col < n2; col++) {
        for (int row = 0; row < n2; row++) {
            dcomplex &h = hs2[row + static_cast<size_t>(col) * n2];
            if (row < n && col < n) {
                h.r = r11[row + static_cast<size_t>(col) * n];
                h.i = i11[row + static_cast<size_t>(col) * n];
            }
            else if (row < n) {
                h.r = r12[row + static_cast<size_t>(col - n) * n];
                h.i = i12[row + static_cast<size_t>(col - n) * n];
            }
            else if (col < n) {
                h.r = r12[col + static_cast<size_t>(row - n) * n];
                h.i = -i12[col + static_cast<size_t>(row - n) * n];
            }
            else {
                h.r = r22[(row - n) + static_cast<size_t>(col - n) * n];
                h.i = i22[(row - n) + static_cast<size_t>(col - n) * n];
            }
        }
    }
    auto upload_blocks = [&] {
        check(hipMemcpy(d_out, r11, nn * sizeof(double), hipMemcpyHostToDevice), "upload r11");
        check(hipMemcpy(d_out + nn, i11, nn * sizeof(double), hipMemcpyHostToDevice), "upload i11");
        check(hipMemcpy(slot[0], r12, nn * sizeof(double), hipMemcpyHostToDevice), "upload r12");
        check(hipMemcpy(slot[1], i12, nn * sizeof(double), hipMemcpyHostToDevice), "upload i12");
        check(hipMemcpy(slot[2], r22, nn * sizeof(double), hipMemcpyHostToDevice), "upload r22");
        check(hipMemcpy(slot[3], i22, nn * sizeof(double), hipMemcpyHostToDevice), "upload i22");
    };
    upload_blocks();
    check_rc(ClusterNonCol_AssembleHs2_HIP(handle, n, d_out, d_out + nn, slot[0], slot[1], slot[2], slot[3], d_a),
             "assembly");
    check(hipMemcpy(got.data(), d_a, n2n2 * sizeof(dcomplex), hipMemcpyDeviceToHost), "download Hs2");
    const bool assembly_same = (std::memcmp(got.data(), hs2.data(), n2n2 * sizeof(dcomplex)) == 0);
    failures += !assembly_same;
    double t_assembly = 0.0;
    {
        std::vector<double> times;
        for (int r = 0; r < reps + 1; r++) {
            upload_blocks();
            check(hipDeviceSynchronize(), "synchronize");
            auto t0 = std::chrono::steady_clock::now();
            check_rc(ClusterNonCol_AssembleHs2_HIP(handle, n, d_out, d_out + nn, slot[0], slot[1], slot[2], slot[3],
                                                   d_a), "assembly");
            check(hipDeviceSynchronize(), "synchronize");
            if (r) times.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        }
        std::sort(times.begin(), times.end());
        t_assembly = times[times.size() / 2];
    }
    /* the former host loop and the upload of Hs2 */
    const double t_host_assembly = median_ms(reps, [&] {
        std::vector<dcomplex> &h = got;
        for (int col = 0; col < n2; col++) {
            for (int row = 0; row < n2; row++) {
                dcomplex &z = h[row + static_cast<size_t>(col) * n2];
                if (row < n && col < n) {
                    z.r = r11[row + static_cast<size_t>(col) * n];
                    z.i = i11[row + static_cast<size_t>(col) * n];
                }
                else if (row < n) {
                    z.r = r12[row + static_cast<size_t>(col - n) * n];
                    z.i = i12[row + static_cast<size_t>(col - n) * n];
                }
                else if (col < n) {
                    z.r = r12[col + static_cast<size_t>(row - n) * n];
                    z.i = -i12[col + static_cast<size_t>(row - n) * n];
                }
                else {
                    z.r = r22[(row - n) + static_cast<size_t>(col - n) * n];
                    z.i = i22[(row - n) + static_cast<size_t>(col - n) * n];
                }
            }
        }
        check(hipMemcpy(d_a, h.data(), n2n2 * sizeof(dcomplex), hipMemcpyHostToDevice), "upload Hs2");
    });

    /* the six forward-transform products: on the device (the solver's
       products, capped GEMMul8 with the solve reserve), against the former
       host-staged transform */
    std::vector<double> h_s(nn);
    for (auto &x : h_s) x = uniform(state);
    double *d_s = d_out;   /* the H11 blocks are done with */
    check(hipMemcpy(d_s, h_s.data(), nn * sizeof(double), hipMemcpyHostToDevice), "upload S");
    const double one = 1.0, zero = 0.0;
    for (int k = 0; k < 6; k++) {
        check(hipMemcpy(slot[k], &blocks[k * nn], nn * sizeof(double), hipMemcpyHostToDevice), "upload block");
    }
    const double t_device_forward = median_ms(reps, [&] {
        for (int k = 0; k < 6; k++) {
            check(openmx_gemmul8DgemmSolve(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, n, n, n, &one, slot[k], n, d_s, n, &zero,
                                           slot[6], n), "H S");
            check(openmx_gemmul8DgemmSolve(handle, HIPBLAS_OP_T, HIPBLAS_OP_N, n, n, n, &one, d_s, n, slot[6], n, &zero,
                                           slot[7], n), "S^T (H S)");
        }
    });
    std::vector<double> host_block(nn);
    const double t_host_forward = median_ms(reps, [&] {
        hipblasHandle_t own = nullptr;
        double *b_s = nullptr, *b_a = nullptr, *b_c = nullptr;
        check(hipblasCreate(&own), "hipblasCreate");
        check(hipMalloc(&b_s, nn * sizeof(double)), "hipMalloc");
        check(hipMalloc(&b_a, nn * sizeof(double)), "hipMalloc");
        check(hipMalloc(&b_c, nn * sizeof(double)), "hipMalloc");
        check(hipMemcpy(b_s, h_s.data(), nn * sizeof(double), hipMemcpyHostToDevice), "upload");
        for (int k = 0; k < 6; k++) {
            check(hipMemcpy(b_a, &blocks[k * nn], nn * sizeof(double), hipMemcpyHostToDevice), "upload");
            check(openmx_gemmul8Dgemm(own, HIPBLAS_OP_N, HIPBLAS_OP_N, n, n, n, &one, b_a, n, b_s, n, &zero, b_c, n), "H S");
            check(openmx_gemmul8Dgemm(own, HIPBLAS_OP_T, HIPBLAS_OP_N, n, n, n, &one, b_s, n, b_c, n, &zero, b_a, n),
                  "S^T (H S)");
            check(hipMemcpy(host_block.data(), b_a, nn * sizeof(double), hipMemcpyDeviceToHost), "download");
        }
        (void)hipFree(b_c);
        (void)hipFree(b_a);
        (void)hipFree(b_s);
        (void)hipblasDestroy(own);
    });
    /* the same products give the same block */
    std::vector<double> dev_block(nn);
    check(hipMemcpy(dev_block.data(), slot[7], nn * sizeof(double), hipMemcpyDeviceToHost), "download block");
    const bool forward_same = (std::memcmp(dev_block.data(), host_block.data(), nn * sizeof(double)) == 0);

    std::printf("forward n=%d n2=%d %s entries=%d scatter_ms=%.3f scatter_bitwise=%s assembly_ms=%.3f "
                "host_assembly_upload_ms=%.3f assembly_bitwise=%s transform_device_ms=%.3f "
                "transform_host_staged_ms=%.3f transform_bitwise=%s\n",
                n, n2, label, tnum, t_scatter, scatter_same ? "yes" : "NO", t_assembly, t_host_assembly,
                assembly_same ? "yes" : "NO", t_device_forward, t_host_forward, forward_same ? "yes" : "no");
    (void)hipFree(d_out);
    (void)hipFree(d_a);
    return failures;
}

/* ------------------------------------------------------------------ */

int run_dm(hipblasHandle_t handle, int n, int maxn, int nk, int reps, bool edm, int neighbors, const char *label)
{
    const int n2 = 2 * n, orbitals = 13, atoms = n / orbitals;
    const size_t n2n2 = static_cast<size_t>(n2) * n2;
    const int comps = edm ? 10 : 6;
    uint64_t state = 0x8CB92BA72F3D8DD7ULL ^ static_cast<uint64_t>(nk);

    if (atoms * orbitals != n) {
        std::fprintf(stderr, "dm: n must be a multiple of %d\n", orbitals);
        return 1;
    }
    /* the entry order of the solver's DM entry cache: atom A, its
       neighbours B, the orbitals of A, those of B */
    std::vector<int> basis0, basis1;
    const int half_window = neighbors / 2;
    for (int a = 0; a < atoms; a++) {
        for (int b = std::max(0, a - half_window); b <= std::min(atoms - 1, a + half_window); b++) {
            for (int i = 0; i < orbitals; i++) {
                for (int j = 0; j < orbitals; j++) {
                    basis0.push_back(a * orbitals + i);
                    basis1.push_back(b * orbitals + j);
                }
            }
        }
    }
    const int entries = static_cast<int>(basis0.size());

    /* occupations: occupied, then a Fermi tail over the last 40 states */
    std::vector<double> occ(nk), occ_e(nk), energy(nk);
    for (int k = 0; k < nk; k++) {
        energy[k] = -1.0 + 1.5 * k / nk + 1e-3 * uniform(state);
        const double x = (k - (nk - 20)) / 4.0;
        occ[k] = 1.0 / (1.0 + std::exp(x));
        occ_e[k] = occ[k] * energy[k];
    }
    /* state-major vectors (state k of basis i at k + i n2), rows maxn.. unused */
    std::vector<dcomplex> h_v(n2n2);
    const double amp = 1.0 / std::sqrt(static_cast<double>(n2));
    for (auto &z : h_v) {
        z.r = amp * uniform(state);
        z.i = amp * uniform(state);
    }
    dcomplex *d_v = static_cast<dcomplex *>(device_alloc(n2n2 * sizeof(dcomplex), "hipMalloc V"));
    check(hipMemcpy(d_v, h_v.data(), n2n2 * sizeof(dcomplex), hipMemcpyHostToDevice), "upload V");
    /* the solve's arena after the back transform: 16 maxn n2 bytes */
    const size_t arena_bytes = static_cast<size_t>(16) * maxn * n2;
    void *d_arena = device_alloc(arena_bytes, "hipMalloc arena");

    std::vector<double> dm_sum(static_cast<size_t>(entries) * comps), dm_gemm(static_cast<size_t>(entries) * comps);
    const long long native0 = openmx_gemmul8NativeCalls(1);
    const double t_sum = median_ms(reps, [&] {
        check_rc(ClusterNonCol_CalcDMRootDenseDevice_HIP(entries, entries, n, n2, nk, edm, basis0.data(), basis1.data(),
                                                         occ.data(), occ_e.data(), d_v, n2, d_arena, arena_bytes,
                                                         dm_sum.data()), "per-entry sums");
    });
    const double t_gemm = median_ms(reps, [&] {
        check_rc(ClusterNonCol_CalcDMRootDenseGemm_HIP(handle, entries, entries, n, n2, nk, edm, basis0.data(),
                                                       basis1.data(), occ.data(), occ_e.data(), d_v, n2, d_arena,
                                                       arena_bytes, dm_gemm.data()), "products");
    });
    const long long native = openmx_gemmul8NativeCalls(1) - native0;

    /* the reference on a sample of entries, in long double */
    const int samples = std::min(entries, 4000);
    double max_ref[10] = {0}, err_sum[10] = {0}, err_gemm[10] = {0}, max_diff = 0.0;
    for (int t = 0; t < samples; t++) {
        const int p = static_cast<int>((static_cast<long long>(t) * 104729) % entries);
        const int ia = basis0[p], ib = basis1[p];
        long double acc[10] = {0};
        for (int k = 0; k < nk; k++) {
            const dcomplex au = h_v[k + static_cast<size_t>(ia) * n2], bu = h_v[k + static_cast<size_t>(ib) * n2];
            const dcomplex ad = h_v[k + static_cast<size_t>(ia + n) * n2], bd = h_v[k + static_cast<size_t>(ib + n) * n2];
            const long double re11 = (long double)au.r * bu.r + (long double)au.i * bu.i;
            const long double im11 = (long double)au.r * bu.i - (long double)au.i * bu.r;
            const long double re22 = (long double)ad.r * bd.r + (long double)ad.i * bd.i;
            const long double im22 = (long double)ad.r * bd.i - (long double)ad.i * bd.r;
            const long double re12 = (long double)au.r * bd.r + (long double)au.i * bd.i;
            const long double im12 = (long double)au.r * bd.i - (long double)au.i * bd.r;
            acc[0] += occ[k] * re11;
            acc[1] += occ[k] * re22;
            acc[2] += occ[k] * re12;
            acc[3] += occ[k] * im12;
            acc[4] += occ[k] * im11;
            acc[5] += occ[k] * im22;
            acc[6] += occ_e[k] * re11;
            acc[7] += occ_e[k] * re22;
            acc[8] += occ_e[k] * re12;
            acc[9] += occ_e[k] * im12;
        }
        for (int c = 0; c < comps; c++) {
            const size_t at = static_cast<size_t>(c) * entries + p;
            max_ref[c] = std::max(max_ref[c], static_cast<double>(std::fabs(acc[c])));
            err_sum[c] = std::max(err_sum[c], static_cast<double>(std::fabs(dm_sum[at] - acc[c])));
            err_gemm[c] = std::max(err_gemm[c], static_cast<double>(std::fabs(dm_gemm[at] - acc[c])));
        }
    }
    for (size_t at = 0; at < dm_sum.size(); at++) max_diff = std::max(max_diff, std::fabs(dm_sum[at] - dm_gemm[at]));
    double rel_sum = 0.0, rel_gemm = 0.0, ref_all = 0.0;
    for (int c = 0; c < comps; c++) {
        ref_all = std::max(ref_all, max_ref[c]);
        rel_sum = std::max(rel_sum, err_sum[c] / std::max(max_ref[c], 1e-300));
        rel_gemm = std::max(rel_gemm, err_gemm[c] / std::max(max_ref[c], 1e-300));
    }
    std::printf("dm n=%d n2=%d nk=%d %s%s entries=%d sums_ms=%.3f products_ms=%.3f native_fallbacks=%lld "
                "err_sums=%.2e err_products=%.2e max|sums-products|=%.2e (of max|DM| %.2e; %d sampled entries)\n",
                n, n2, nk, label, edm ? " edm" : "", entries, t_sum, t_gemm, native, rel_sum, rel_gemm, max_diff,
                ref_all, samples);
    (void)hipFree(d_arena);
    (void)hipFree(d_v);
    return (rel_gemm < 1e-11 && rel_sum < 1e-11) ? 0 : 1;
}

} // namespace

int main(int argc, char **argv)
{
    int positional[4] = {0, 0, 0, 5};
    int count = 0, neighbors = 87;
    bool fp64 = false, old_form = false, edm = false, bad = false;
    const char *only = nullptr, *label = nullptr;

    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--fp64") == 0) fp64 = true;
        else if (std::strcmp(argv[i], "--old") == 0) old_form = true;
        else if (std::strcmp(argv[i], "--edm") == 0) edm = true;
        else if (std::strcmp(argv[i], "--only") == 0 && i + 1 < argc) only = argv[++i];
        else if (std::strcmp(argv[i], "--neighbors") == 0 && i + 1 < argc) neighbors = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--label") == 0 && i + 1 < argc) label = argv[++i];
        else if (count < 4) positional[count++] = std::atoi(argv[i]);
        else bad = true;
    }
    const int n = positional[0], maxn = positional[1];
    const int nk = positional[2] > 0 ? positional[2] : std::max(1, maxn / 2);
    const int reps = std::max(1, positional[3]);
    if (bad || count < 2 || n < 13 || maxn < 1 || 2 * n < maxn || maxn < nk || neighbors < 1) {
        std::fprintf(stderr, "usage: %s <n> <maxn> [nk] [reps] [--fp64] [--old] [--edm] [--only back|forward|dm] "
                     "[--neighbors K] [--label NAME]   (nk <= maxn <= 2n)\n", argv[0]);
        return 2;
    }
    if (label == nullptr) label = fp64 ? "fp64" : "gemmul8";
    if (fp64) openmx_gemmul8SetEnabled(0);

    size_t free_bytes = 0, total_bytes = 0;
    check(hipMemGetInfo(&free_bytes, &total_bytes), "hipMemGetInfo");
    const double n2n2_mib = 16.0 * 4.0 * n * static_cast<double>(n) / 1048576.0;
    std::printf("device: %.0f MiB free of %.0f; the largest stage holds about %.0f MiB plus the GEMMul8 workspace\n",
                free_bytes / 1048576.0, total_bytes / 1048576.0,
                n2n2_mib * (old_form ? 3.0 : 1.0) + 16.0 * maxn * 2.0 * n / 1048576.0 + 8.0 * n * (double)n / 1048576.0);

    hipblasHandle_t handle = nullptr;
    check(hipblasCreate(&handle), "hipblasCreate");
    int failures = 0;
    if (only == nullptr || std::strcmp(only, "forward") == 0) failures += run_forward(handle, n, reps, label);
    openmx_gemmul8ReleaseWorkspaces();
    if (only == nullptr || std::strcmp(only, "back") == 0) failures += run_back(handle, n, maxn, reps, old_form, label);
    openmx_gemmul8ReleaseWorkspaces();
    if (only == nullptr || std::strcmp(only, "dm") == 0) failures += run_dm(handle, n, maxn, nk, reps, edm, neighbors, label);
    openmx_gemmul8ReleaseWorkspaces();
    (void)hipblasDestroy(handle);
    std::printf("%s\n", failures ? "FAILED" : "OK");
    return failures ? 1 : 0;
}

// Numerical check of the refined FP32 eigensolver (source/eigen_refine_gpu.cu)
// against the FP64 rocSOLVER divide and conquer, on synthetic matrices with
// the structure of the OpenMX dense problems: a Hermitian (or real symmetric)
// "Hamiltonian" whose levels come in degenerate pairs (the spin degeneracy of
// a non-magnetic non-collinear system), a bandwidth of about 2 Ha and an
// electron count that puts the Fermi level inside a degenerate pair, so that
// the Rayleigh-Ritz step of partially occupied clusters is exercised.
//
// For each case: a cold refined solve, then a warm solve of the matrix
// perturbed by 1e-6 (the next SCF step), each compared with FP64 in the
// eigenvalues and in the density matrix P = V diag(f) V^H of the wanted
// states, f the Fermi occupations at 300 K with the chemical potential from
// bisection.
//
// Then the product blocks in the layout of the collinear cluster solver
// (FP32 scratch and block region = one n x n FP64 panel, d_tmp): with few
// wanted states the blocks fit the region and no buffer of their own is
// allocated, with more (or all n) the solve allocates exactly
// openmx_eigen_refine_product_bytes(), the size its memory admission
// budgets; each solve still matches FP64.
//
// Build (from the repository root, after building the OpenMX objects):
//   MPI=/path/to/mpi        # the MPI installation of the OpenMX build
//   /opt/rocm-7.2.0/bin/hipcc -std=c++20 -O2 --offload-arch=gfx1200 -Isource $($MPI/bin/mpicc --showme:compile) \
//       -c tools/check_eigen_refine_hip.cu -o check_eigen_refine_hip.o
//   /opt/rocm-7.2.0/bin/hipcc --offload-arch=gfx1200 check_eigen_refine_hip.o source/eigen_refine_gpu.o \
//       source/gemmul8_bridge.o source/magma_openmx.o source/third_party/magma/lib/libmagma.a \
//       source/third_party/GEMMul8/lib/libgemmul8.a $($MPI/bin/mpicc --showme:link) \
//       -L/opt/rocm-7.2.0/lib -lhipsolver -lhipblas -lhipblaslt -lrocblas \
//       -L$AOCL/lib -lflame -lblis -o check_eigen_refine_hip
// Run: ./check_eigen_refine_hip [n_complex] [n_real] [n_blocks]   (defaults 2808 2808 1200; 0 skips)
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <hipsolver/hipsolver.h>
#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "eigen_refine_gpu.h"

extern "C" int MYID_MPI_COMM_WORLD;
int MYID_MPI_COMM_WORLD = 0;

#define CK(x)                                                                                    \
    do {                                                                                         \
        auto _s = (x);                                                                           \
        if ((int)_s != 0) {                                                                      \
            fprintf(stderr, "%s:%d: error %d\n", __FILE__, __LINE__, (int)_s);                   \
            exit(2);                                                                             \
        }                                                                                        \
    } while (0)

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

/* the sizes the refinement asked for, while recording */
static std::vector<size_t> g_requests;
static bool                g_recording = false;
static hipError_t try_malloc(void **ptr, size_t bytes)
{
    if (g_recording) g_requests.push_back(bytes);
    return hipMalloc(ptr, bytes);
}

extern "C" size_t openmx_eigen_refine_product_bytes(int cplx, int n, int maxn);
static void *host_malloc(size_t count, size_t elem, const char *label)
{
    void *p = calloc(count ? count : 1, elem);
    if (p == nullptr) {
        fprintf(stderr, "host allocation of %s failed\n", label);
        exit(2);
    }
    return p;
}

/* Fermi occupations at 300 K, chemical potential by bisection over the
   lowest maxn states for nel electrons */
static const double kBeta = 1.0 / (300.0 * 3.166811563e-6);
struct Fermi {
    double mu;
};
static double fermi(double e, int index, void *ctx)
{
    (void)index;
    double x = (e - ((Fermi *)ctx)->mu) * kBeta;
    if (x > 60.0) x = 60.0;
    if (x < -60.0) x = -60.0;
    return 1.0 / (1.0 + exp(x));
}
static double find_mu(const double *e, int maxn, double nel)
{
    double lo = e[0] - 1.0, hi = e[maxn - 1] + 1.0;
    for (int it = 0; it < 200; it++) {
        Fermi  f = {0.5 * (lo + hi)};
        double count = 0.0;
        for (int i = 0; i < maxn; i++) count += fermi(e[i], i + 1, &f);
        if (count < nel) lo = f.mu;
        else hi = f.mu;
    }
    return 0.5 * (lo + hi);
}

/* P = V diag(f) V^H over the first maxn columns of V (n x n device, width
   doubles per entry), on the host for the comparison */
static std::vector<double> density(hipblasHandle_t h, int cplx, int n, int maxn, const double *dV, const double *e,
                                   double nel)
{
    const int    width = cplx ? 2 : 1;
    Fermi        f = {find_mu(e, maxn, nel)};
    std::vector<double> hv((size_t)width * n * maxn), hw((size_t)width * n * maxn);
    CK(hipMemcpy(hv.data(), dV, hv.size() * sizeof(double), hipMemcpyDeviceToHost));
    for (int k = 0; k < maxn; k++) {
        double s = sqrt(fermi(e[k], k + 1, &f));
        for (size_t i = 0; i < (size_t)width * n; i++) hw[(size_t)k * width * n + i] = s * hv[(size_t)k * width * n + i];
    }
    double *dW, *dP;
    CK(hipMalloc(&dW, hw.size() * sizeof(double)));
    CK(hipMalloc(&dP, (size_t)width * n * n * sizeof(double)));
    CK(hipMemcpy(dW, hw.data(), hw.size() * sizeof(double), hipMemcpyHostToDevice));
    if (cplx) {
        hipDoubleComplex one = {1, 0}, zero = {0, 0};
        CK(hipblasZgemm(h, HIPBLAS_OP_N, HIPBLAS_OP_C, n, n, maxn, &one, (hipDoubleComplex *)dW, n,
                        (hipDoubleComplex *)dW, n, &zero, (hipDoubleComplex *)dP, n));
    }
    else {
        double one = 1, zero = 0;
        CK(hipblasDgemm(h, HIPBLAS_OP_N, HIPBLAS_OP_T, n, n, maxn, &one, dW, n, dW, n, &zero, dP, n));
    }
    std::vector<double> P((size_t)width * n * n);
    CK(hipMemcpy(P.data(), dP, P.size() * sizeof(double), hipMemcpyDeviceToHost));
    CK(hipFree(dW));
    CK(hipFree(dP));
    return P;
}

/* a "Hamiltonian" with doubly degenerate levels, H = kron(I2, H0): H0
   random symmetric/Hermitian with spectrum width ~2 Ha (scale), every level
   twice, as the spin degeneracy of a non-magnetic non-collinear system */
static std::vector<double> make_matrix(int cplx, int n, unsigned seed, double scale)
{
    const int width = cplx ? 2 : 1;
    const int h = n / 2;
    std::vector<double> H0((size_t)width * h * h);
    unsigned s = seed;
    auto rnd = [&]() { s = s * 1664525u + 1013904223u; return (double)(s >> 8) / (double)(1u << 24) - 0.5; };
    for (int j = 0; j < h; j++)
        for (int i = j; i < h; i++) {
            double re = rnd() * scale, im = (cplx && i != j) ? rnd() * scale : 0.0;
            size_t ij = (size_t)i + (size_t)j * h, ji = (size_t)j + (size_t)i * h;
            H0[width * ij] = re;
            H0[width * ji] = re;
            if (cplx) {
                H0[width * ij + 1] = im;
                H0[width * ji + 1] = -im;
            }
        }
    std::vector<double> H((size_t)width * n * n, 0.0);
    for (int b = 0; b < 2; b++)
        for (int j = 0; j < h; j++)
            for (int i = 0; i < h; i++) {
                size_t src = (size_t)i + (size_t)j * h;
                size_t dst = (size_t)(i + b * h) + (size_t)(j + b * h) * n;
                for (int w = 0; w < width; w++) H[width * dst + w] = H0[width * src + w];
            }
    return H;
}

struct Case {
    int    cplx, n;
    double nel;     /* electrons (occupancy 1 per state) */
};

static int run_case(const Case &cs, hipblasHandle_t h)
{
    const int    cplx = cs.cplx, n = cs.n, width = cplx ? 2 : 1;
    const int    maxn = std::min(n, (int)(cs.nel * 1.3) + 10);
    const size_t nn = (size_t)n * n;
    std::vector<double> H = make_matrix(cplx, n, 12345u + n + cplx, 2.0 / sqrt((double)n));
    int failures = 0;

    printf("=== %s n=%d maxn=%d electrons=%.1f\n", cplx ? "complex Hermitian" : "real symmetric", n, maxn, cs.nel);

    /* device buffers */
    double *dA, *dRef, *dX, *dW, *dWork = nullptr;
    void   *fp32;
    size_t  work_bytes = 0;
    int32_t *dInfo;
    const size_t fp32_bytes = ((cplx ? 2 : 1) * nn + n + 1) * sizeof(float);
    const size_t region = std::max(fp32_bytes, (size_t)3 * width * n * maxn * sizeof(double) + 4096);
    CK(hipMalloc(&dA, width * nn * sizeof(double)));
    CK(hipMalloc(&dRef, width * nn * sizeof(double)));
    CK(hipMalloc(&dX, width * nn * sizeof(double)));
    CK(hipMalloc(&dW, n * sizeof(double)));
    CK(hipMalloc(&fp32, region));
    CK(hipMalloc(&dInfo, sizeof(int32_t)));

    hipsolverDnHandle_t sol;
    CK(hipsolverDnCreate(&sol));

    EigenRefineDevice dev;
    memset(&dev, 0, sizeof(dev));
    dev.hipblas = h;
    dev.d_work = (void **)&dWork;
    dev.d_work_bytes = &work_bytes;
    dev.d_info = dInfo;
    dev.try_malloc = try_malloc;
    dev.host_malloc = host_malloc;
    EigenRefineState st;
    memset(&st, 0, sizeof(st));

    for (int step = 0; step < 3; step++) {
        /* the matrix of this "SCF step": H + 1e-6 step R */
        std::vector<double> Hs = H;
        if (step > 0) {
            unsigned s = 777u * step;
            auto rnd = [&]() { s = s * 1664525u + 1013904223u; return (double)(s >> 8) / (double)(1u << 24) - 0.5; };
            for (int j = 0; j < n; j++)
                for (int i = j; i < n; i++) {
                    double re = 1e-6 * rnd(), im = (cplx && i != j) ? 1e-6 * rnd() : 0.0;
                    size_t ij = (size_t)i + (size_t)j * n, ji = (size_t)j + (size_t)i * n;
                    Hs[width * ij] += re;
                    if (i != j) Hs[width * ji] += re;
                    if (cplx) {
                        Hs[width * ij + 1] += im;
                        Hs[width * ji + 1] -= im;
                    }
                }
        }
        /* FP64 reference; the exact zeros dithered by 1e-200, since
           rocSOLVER's Hermitian divide and conquer returns NaN vectors for
           exactly block-diagonal matrices */
        {
            std::vector<double> Hr = Hs;
            for (int j = 0; j < n; j++)
                for (int i = 0; i < n; i++) {
                    size_t at = (size_t)width * ((size_t)i + (size_t)j * n);
                    if (i != j && Hr[at] == 0.0 && (!cplx || Hr[at + 1] == 0.0)) Hr[at] = 1e-200 * (1 + (i * 7 + j * 13) % 17);
                }
            CK(hipMemcpy(dRef, Hr.data(), width * nn * sizeof(double), hipMemcpyHostToDevice));
        }
        int lwork = 0;
        double t0 = now();
        if (cplx) CK(hipsolverDnZheevd_bufferSize(sol, HIPSOLVER_EIG_MODE_VECTOR, HIPBLAS_FILL_MODE_LOWER, n, (hipDoubleComplex *)dRef, n, dW, &lwork));
        else CK(hipsolverDnDsyevd_bufferSize(sol, HIPSOLVER_EIG_MODE_VECTOR, HIPBLAS_FILL_MODE_LOWER, n, dRef, n, dW, &lwork));
        void *rwork;
        CK(hipMalloc(&rwork, (size_t)lwork * width * sizeof(double)));
        if (cplx) CK(hipsolverDnZheevd(sol, HIPSOLVER_EIG_MODE_VECTOR, HIPBLAS_FILL_MODE_LOWER, n, (hipDoubleComplex *)dRef, n, dW, (hipDoubleComplex *)rwork, lwork, dInfo));
        else CK(hipsolverDnDsyevd(sol, HIPSOLVER_EIG_MODE_VECTOR, HIPBLAS_FILL_MODE_LOWER, n, dRef, n, dW, (double *)rwork, lwork, dInfo));
        CK(hipDeviceSynchronize());
        double t_ref = now() - t0;
        CK(hipFree(rwork));
        std::vector<double> eref(n);
        CK(hipMemcpy(eref.data(), dW, n * sizeof(double), hipMemcpyDeviceToHost));
        std::vector<double> Pref = density(h, cplx, n, maxn, dRef, eref.data(), cs.nel);

        /* refined */
        CK(hipMemcpy(dA, Hs.data(), width * nn * sizeof(double), hipMemcpyHostToDevice));
        EigenRefineProblem pb;
        memset(&pb, 0, sizeof(pb));
        Fermi occ_ctx = {0.0};
        pb.cplx = cplx;
        pb.n = n;
        pb.maxn = maxn;
        pb.iterations = 2;
        pb.a = dA;
        pb.w = dW;
        pb.fp32 = fp32;
        pb.region = region;
        pb.x = dX;
        pb.transient = 1;
        pb.warm = (step > 0 && st.basis_valid && st.basis_n == n);
        pb.defaulted = 0;
        pb.occupation = fermi;
        pb.occupation_ctx = &occ_ctx;
        EigenRefineReport rep;
        std::vector<double> e0(n), f0(n);
        t0 = now();
        int rc = openmx_eigen_refine_prepare(&st, &dev, &pb, e0.data(), &rep);
        if (rc == -1) {
            pb.warm = 0;
            rc = openmx_eigen_refine_prepare(&st, &dev, &pb, e0.data(), &rep);
        }
        double t_prep = now() - t0;
        if (rc != 1) {
            printf("  step %d: prepare failed (rc %d)\n", step, rc);
            failures++;
            break;
        }
        occ_ctx.mu = find_mu(e0.data(), maxn, cs.nel);
        for (int i = 0; i < n; i++) f0[i] = fermi(e0[i], i + 1, &occ_ctx);
        double t1 = now();
        rc = openmx_eigen_refine_finish(&st, &dev, &pb, f0.data(), &rep);
        double t_fin = now() - t1;
        if (rc != 1) {
            printf("  step %d: finish failed\n", step);
            failures++;
            break;
        }
        std::vector<double> ew(maxn);
        CK(hipMemcpy(ew.data(), dW, maxn * sizeof(double), hipMemcpyDeviceToHost));
        double de = 0.0;
        long   nonfinite = 0;
        for (int i = 0; i < maxn; i++) {
            if (!std::isfinite(ew[i]) || !std::isfinite(eref[i])) nonfinite++;
            else de = std::max(de, fabs(ew[i] - eref[i]));
        }
        std::vector<double> P = density(h, cplx, n, maxn, dA, ew.data(), cs.nel);
        double dp = 0.0, pmax = 0.0;
        for (size_t i = 0; i < P.size(); i++) {
            if (!std::isfinite(P[i]) || !std::isfinite(Pref[i])) {
                nonfinite++;
                continue;
            }
            dp = std::max(dp, fabs(P[i] - Pref[i]));
            pmax = std::max(pmax, fabs(Pref[i]));
        }
        /* band energy sum f e over the wanted states, each with its own
           chemical potential */
        double eb = 0.0, eb_ref = 0.0;
        {
            Fermi fa = {find_mu(ew.data(), maxn, cs.nel)}, fb = {find_mu(eref.data(), maxn, cs.nel)};
            for (int i = 0; i < maxn; i++) {
                eb += fermi(ew[i], i + 1, &fa) * ew[i];
                eb_ref += fermi(eref[i], i + 1, &fb) * eref[i];
            }
        }
        Fermi fr = {find_mu(eref.data(), maxn, cs.nel)};
        int partial = 0;
        for (int i = 0; i < maxn; i++) {
            double f = fermi(eref[i], i + 1, &fr);
            if (f > 1e-12 && f < 1.0 - 1e-12) partial++;
        }
        printf("  step %d %s: FP64 %.3f s | refined prepare %.3f s + finish %.3f s | max|dE| %.2e  |dEband| %.2e  "
               "max|dP| %.2e (|P| %.2e)  rr clusters %d (largest %d)  partial states %d  delta %.1e%s\n",
               step, rep.fp32_solved ? "cold" : "warm", t_ref, t_prep, t_fin, de, fabs(eb - eb_ref), dp, pmax,
               rep.rr_clusters, rep.rr_largest, partial, rep.delta, nonfinite ? "  NON-FINITE VALUES" : "");
        if (nonfinite || dp > 1e-10 || fabs(eb - eb_ref) > 1e-9) {
            printf("  ** accuracy check FAILED (density matrix 1e-10, band energy 1e-9 Ha, all finite)\n");
            failures++;
        }
    }
    openmx_eigen_refine_state_release(&st);
    openmx_eigen_refine_device_release(&dev);
    CK(hipsolverDnDestroy(sol));
    CK(hipFree(dA));
    CK(hipFree(dRef));
    CK(hipFree(dX));
    CK(hipFree(dW));
    CK(hipFree(fp32));
    CK(hipFree(dInfo));
    if (dWork) CK(hipFree(dWork));
    return failures;
}

/* The product blocks in the collinear cluster solver's layout: the FP32
   scratch and the region of the blocks are one n x n FP64 panel (d_tmp).
   A cold refined solve of maxn wanted states of a real symmetric n x n
   matrix: the blocks take the region when they fit it, else the solve
   allocates exactly openmx_eigen_refine_product_bytes(0, n, maxn) (what the
   solver's admission budgets); the eigenvalues match FP64 either way. */
static int check_product_blocks(int n, hipblasHandle_t h)
{
    const size_t nn = (size_t)n * n, panel = nn * sizeof(double);
    const double nel = (double)(n / 6) + 1.0;
    /* the first step's states of the cluster solver (n/6 occupied + a
       margin, within the region), a third of n and more (beyond it), all */
    const int maxns[3] = {n / 6 + n / 10, n / 2, n};
    std::vector<double> H = make_matrix(0, n, 4242u + n, 2.0 / sqrt((double)n));
    double *dA, *dRef, *dX, *dW, *dT, *dWork = nullptr;
    size_t work_bytes = 0;
    int32_t *dInfo;
    int failures = 0;
    hipsolverDnHandle_t sol;

    printf("=== product blocks in the collinear cluster layout: real symmetric n=%d, region %.1f MiB\n", n,
           (double)panel / 1048576.0);
    CK(hipMalloc(&dA, panel));
    CK(hipMalloc(&dRef, panel));
    CK(hipMalloc(&dX, panel));
    CK(hipMalloc(&dW, n * sizeof(double)));
    CK(hipMalloc(&dT, panel));
    CK(hipMalloc(&dInfo, sizeof(int32_t)));
    CK(hipsolverDnCreate(&sol));

    /* FP64 reference eigenvalues */
    std::vector<double> eref(n);
    {
        int lwork = 0;
        double *rwork;
        CK(hipMemcpy(dRef, H.data(), panel, hipMemcpyHostToDevice));
        CK(hipsolverDnDsyevd_bufferSize(sol, HIPSOLVER_EIG_MODE_VECTOR, HIPBLAS_FILL_MODE_LOWER, n, dRef, n, dW, &lwork));
        CK(hipMalloc(&rwork, (size_t)lwork * sizeof(double)));
        CK(hipsolverDnDsyevd(sol, HIPSOLVER_EIG_MODE_VECTOR, HIPBLAS_FILL_MODE_LOWER, n, dRef, n, dW, rwork, lwork, dInfo));
        CK(hipDeviceSynchronize());
        CK(hipFree(rwork));
        CK(hipMemcpy(eref.data(), dW, n * sizeof(double), hipMemcpyDeviceToHost));
    }

    for (int c = 0; c < 3; c++) {
        const int    maxn = maxns[c];
        const size_t blocks = openmx_eigen_refine_product_bytes(0, n, maxn);
        const bool   own = panel < blocks;
        EigenRefineDevice dev;
        EigenRefineState st;
        EigenRefineProblem pb;
        EigenRefineReport rep;
        Fermi occ_ctx = {0.0};
        std::vector<double> e0(n), f0(n), ew(maxn);
        int rc;
        bool allocated = false;
        double de = 0.0;

        memset(&dev, 0, sizeof(dev));
        dev.hipblas = h;
        dev.d_work = (void **)&dWork;
        dev.d_work_bytes = &work_bytes;
        dev.d_info = dInfo;
        dev.try_malloc = try_malloc;
        dev.host_malloc = host_malloc;
        memset(&st, 0, sizeof(st));
        memset(&pb, 0, sizeof(pb));
        pb.cplx = 0;
        pb.n = n;
        pb.maxn = maxn;
        pb.iterations = 2;
        pb.a = dA;
        pb.w = dW;
        pb.fp32 = dT;
        pb.region = panel;
        pb.x = dX;
        pb.transient = 1;
        pb.occupation = fermi;
        pb.occupation_ctx = &occ_ctx;

        CK(hipMemcpy(dA, H.data(), panel, hipMemcpyHostToDevice));
        g_requests.clear();
        g_recording = true;
        rc = openmx_eigen_refine_prepare(&st, &dev, &pb, e0.data(), &rep);
        if (rc == 1) {
            occ_ctx.mu = find_mu(e0.data(), maxn, nel);
            for (int i = 0; i < n; i++) f0[i] = fermi(e0[i], i + 1, &occ_ctx);
            rc = openmx_eigen_refine_finish(&st, &dev, &pb, f0.data(), &rep);
        }
        g_recording = false;
        for (size_t r : g_requests) allocated = allocated || (r == blocks);
        if (rc == 1) {
            CK(hipMemcpy(ew.data(), dW, maxn * sizeof(double), hipMemcpyDeviceToHost));
            for (int i = 0; i < maxn; i++) de = std::isfinite(ew[i]) ? std::max(de, fabs(ew[i] - eref[i])) : INFINITY;
        }
        printf("  maxn %5d: product blocks %.1f MiB, %s; %s; max|dE| %.2e\n", maxn, (double)blocks / 1048576.0,
               own ? "beyond the region" : "within the region",
               allocated ? "allocated by the solve" : "no allocation of that size", de);
        if (rc != 1 || allocated != own || !(de <= 1e-9)) {
            printf("  ** product block check FAILED (rc %d)\n", rc);
            failures++;
        }
        openmx_eigen_refine_state_release(&st);
        openmx_eigen_refine_device_release(&dev);
    }
    CK(hipsolverDnDestroy(sol));
    CK(hipFree(dA));
    CK(hipFree(dRef));
    CK(hipFree(dX));
    CK(hipFree(dW));
    CK(hipFree(dT));
    CK(hipFree(dInfo));
    if (dWork) CK(hipFree(dWork));
    return failures;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int nc = argc > 1 ? atoi(argv[1]) : 2808;
    int nr = argc > 2 ? atoi(argv[2]) : 2808;
    int nb = argc > 3 ? atoi(argv[3]) : 1200;
    hipblasHandle_t h;
    CK(hipblasCreate(&h));
    int failures = 0;
    /* electron counts that put the Fermi level inside a degenerate pair:
       an odd count fills one state of the pair at the Fermi level */
    if (nc > 0) failures += run_case({1, nc, (double)(nc / 3) + 1.0}, h);
    if (nr > 0) failures += run_case({0, nr, (double)(nr / 3) + 1.0}, h);
    if (nb > 0) failures += check_product_blocks(nb, h);
    CK(hipblasDestroy(h));
    printf("%s (%d failure%s)\n", failures ? "FAILED" : "PASSED", failures, failures == 1 ? "" : "s");
    MPI_Finalize();
    return failures ? 1 : 0;
}

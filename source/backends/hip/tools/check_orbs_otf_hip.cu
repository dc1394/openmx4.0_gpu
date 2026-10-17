// Device check of the orbital evaluation of source/orbs_otf_hip.cu on synthetic
// species with the structure of the OpenMX PAO tables (logarithmic meshes of
// 500 points from 5.5e-4 bohr to the cutoff, L <= 3) on a 0.2 bohr grid.
//
// Every point of each atom's sphere (the central-atom tiles, no index list)
// and the overlap points of atom pairs (the neighbour tiles, through sphere
// indices) are evaluated
//   DF    OTF_Orbitals_Eval on a double-float upload (the default on gfx1200)
//   FP64f OTF_Orbitals_Eval_FP64 on that upload (what builds Orbs_Grid and
//         Orbs_Grid_FNAN in Set_Orbitals_Grid.c)
//   FP64  OTF_Orbitals_Eval on an FP64 upload (the FP64 tiles)
// and compared: FP64f must equal FP64 bit for bit; DF and FP64 must agree to
// a float (counts and worst ulps reported); both against a long double host
// reference with the end substitutes of the spline on every point of the
// end intervals (m = 1, m = mesh-1), inside the first mesh point (atoms on
// and next to grid points) and on a sample of the others.  Kernel times
// (best of the repetitions, hipEvent) per evaluation.
//
// With -DOTF_BASELINE a second build of orbs_otf_hip.cu (for instance the
// version before a change) whose entry points are renamed baseline_* is run
// on the same items: its double-float values must equal DF bitwise, and its
// FP64 values must equal FP64 except at points of the end intervals (the
// values of the host loops there read outside the radial table).
//
// Build (from the repository root):
//   /opt/rocm-7.2.0/bin/hipcc -std=c++20 -O3 --offload-arch=gfx1200 -ffp-contract=off \
//       -DOCML_BASIC_ROUNDED_OPERATIONS -Isource tools/check_orbs_otf_hip.cu source/orbs_otf_hip.cu \
//       source/openmx_precision.c -o check_orbs_otf_hip
// Baseline (B: a directory with the other orbs_otf_hip.cu/.h):
//   R="-DOTF_Orbitals_Upload=baseline_OTF_Orbitals_Upload -DOTF_Orbitals_Eval=baseline_OTF_Orbitals_Eval
//      -DOTF_Orbitals_Release=baseline_OTF_Orbitals_Release -DOTF_Orbitals_Ready=baseline_OTF_Orbitals_Ready
//      -DOTF_Orbitals_DoubleFloat=baseline_OTF_Orbitals_DoubleFloat -DOTF_Orbitals_Bytes=baseline_OTF_Orbitals_Bytes
//      -DOTF_Orbitals_Eval_FP64=baseline_OTF_Orbitals_Eval_FP64"
//   hipcc <flags above> $R -I$B -c $B/orbs_otf_hip.cu -o baseline.o
//   hipcc <flags above> -DOTF_BASELINE -Isource -c tools/check_orbs_otf_hip.cu -o check.o
//   hipcc <flags above> -Isource -c source/orbs_otf_hip.cu -o orbs_otf_hip.o
//   hipcc --offload-arch=gfx1200 check.o orbs_otf_hip.o baseline.o source/openmx_precision.c -o check_orbs_otf_hip
// Run: ./check_orbs_otf_hip [atoms] [spacing] [repetitions]   (defaults 6 0.20 3; < 1 GiB of device memory)
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "orbs_otf_hip.h"

#define CK(x)                                                                                    \
    do {                                                                                         \
        hipError_t _e = (x);                                                                     \
        if (_e != hipSuccess) {                                                                  \
            fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(_e));           \
            exit(2);                                                                             \
        }                                                                                        \
    } while (0)
#define CHECK(c)                                                                                 \
    do {                                                                                         \
        if (!(c)) {                                                                              \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);                         \
            exit(2);                                                                             \
        }                                                                                        \
    } while (0)

typedef int (*EvalFn)(void *, int, int, const OTF_Atom *, const int *, const size_t *, const size_t *,
                      const size_t *, const int *, const int *, const int *, float *);

#ifdef OTF_BASELINE
extern "C" {
int baseline_OTF_Orbitals_Upload(const OTF_HostTables *t, int double_float);
void baseline_OTF_Orbitals_Release(void);
int baseline_OTF_Orbitals_Eval(void *stream, int count, int max_npts, const OTF_Atom *atoms, const int *npts,
                               const size_t *idx_off, const size_t *out_off, const size_t *sph_off, const int *idx,
                               const int *gla, const int *cla, float *out);
}
#endif

typedef long double ld;

struct Species {
    int mesh, maxl, nb[4], no;
    std::vector<double> rv;
    std::vector<std::vector<double>> rw;   // rows L0 -> Mul0
};

static std::vector<Species> SP;
static int NG[3];
static double GT[3][3], ORG[3];
static const double ATV[3] = {0.0, 0.0, 0.0};

static void make_species()
{
    // mesh exp(x), x from -7.5 to xmax in 500 points (C6.0.pao: xmax 1.927, r 5.531e-4 .. 6.869)
    static const double xmax[3] = {1.927, 1.80, 2.0};
    static const int maxl[3] = {2, 1, 3};
    static const int nb[3][4] = {{2, 2, 1, 0}, {2, 1, 0, 0}, {2, 2, 2, 1}};

    for (int w = 0; w < 3; w++) {
        Species s;
        s.mesh = 500;
        s.maxl = maxl[w];
        s.no = 0;
        for (int i = 0; i < s.mesh; i++) s.rv.push_back(exp(-7.5 + (xmax[w] + 7.5) * i / (s.mesh - 1)));
        const double rc = s.rv[s.mesh - 1];
        for (int L0 = 0; L0 <= 3; L0++) {
            s.nb[L0] = (L0 <= s.maxl) ? nb[w][L0] : 0;
            s.no += (2 * L0 + 1) * s.nb[L0];
            for (int mul = 0; mul < s.nb[L0]; mul++) {
                std::vector<double> row(s.mesh);
                const double a = 0.35 + 0.25 * mul + 0.05 * w, b = 0.4 - 0.3 * mul;
                for (int i = 0; i < s.mesh; i++) {
                    const double r = s.rv[i], c = 1.0 - (r / rc) * (r / rc);
                    row[i] = pow(r, L0) * exp(-a * r * r) * (1.0 + b * r) * c * c * c * (1.5 - 0.3 * L0);
                }
                s.rw.push_back(row);
            }
        }
        SP.push_back(s);
    }
}

// ---- the long double reference (the host loops with the end substitutes) ----

static int interval(const Species &s, ld R)
{
    int m = 1;
    while (m < s.mesh - 1 && (ld)s.rv[m] < R) m++;
    return m;
}

static ld spline(const Species &s, const std::vector<double> &rw, int m, ld y1, ld *deriv)
{
    const std::vector<double> &rv = s.rv;
    const ld h2 = (ld)rv[m] - (ld)rv[m - 1];
    ld h1, h3;
    if (m == 1) {
        h3 = (ld)rv[m + 1] - (ld)rv[m];
        h1 = -(h2 + h3);
    }
    else if (m == s.mesh - 1) {
        h1 = (ld)rv[m - 1] - (ld)rv[m - 2];
        h3 = -(h1 + h2);
    }
    else {
        h1 = (ld)rv[m - 1] - (ld)rv[m - 2];
        h3 = (ld)rv[m + 1] - (ld)rv[m];
    }
    const ld f1 = rw[m == 1 ? m + 1 : m - 2], f2 = rw[m - 1], f3 = rw[m], f4 = rw[m == s.mesh - 1 ? m - 2 : m + 1];
    const ld g1 = ((f3 - f2) * h1 / h2 + (f2 - f1) * h2 / h1) / (h1 + h2);
    const ld g2 = ((f4 - f3) * h2 / h3 + (f3 - f2) * h3 / h2) / (h2 + h3);
    const ld y2 = y1 - 1, p = h2 * g1, q = h2 * g2;
    if (deriv != nullptr)
        *deriv = 2 * y2 / h2 * (3 * f2 + p + (2 * f2 + p) * y2) + y2 * y2 * (2 * f2 + p) / h2 +
                 2 * y1 / h2 * (3 * f3 - q - (2 * f3 - q) * y1) - y1 * y1 * (2 * f3 - q) / h2;
    return y2 * y2 * (3 * f2 + p + (2 * f2 + p) * y2) + y1 * y1 * (3 * f3 - q - (2 * f3 - q) * y1);
}

static ld radial(const Species &s, int L0, int row, ld R)
{
    const std::vector<double> &rv = s.rv, &rw = s.rw[row];
    if ((ld)rv[s.mesh - 1] < R) return 0;
    if (R < (ld)rv[0]) {
        const ld rm = rv[4];
        ld d1, a, b, c, d;
        const ld f = spline(s, rw, 4, 1.0L, &d1);
        if (L0 == 0) { a = 0; b = 0.5L * d1 / rm; c = 0; d = f - b * rm * rm; }
        else if (L0 == 1) { a = (rm * d1 - f) / (2 * rm * rm * rm); b = 0; c = d1 - 3 * a * rm * rm; d = 0; }
        else { b = (3 * f - rm * d1) / (rm * rm); a = (f - b * rm * rm) / (rm * rm * rm); c = 0; d = 0; }
        return a * R * R * R + b * R * R + c * R + d;
    }
    const int m = interval(s, R);
    return spline(s, rw, m, (R - (ld)rv[m - 1]) / ((ld)rv[m] - (ld)rv[m - 1]), nullptr);
}

static void reference(const Species &s, double x, double y, double z, ld *out)
{
    const ld X = x, Y = y, Z = z, r = sqrtl(X * X + Y * Y + Z * Z), r1 = sqrtl(X * X + Y * Y);
    ld cq = 0, sq = 1, cp = 1, sp = 0;
    int i = 0, row = 0;
    if (10e-15L <= r) {
        cq = Z / r;
        sq = r1 / r;
        if (10e-15L <= r1) { sp = Y / r1; cp = X / r1; }
    }
    for (int L0 = 0; L0 <= s.maxl; L0++) {
        ld A[7];
        if (L0 == 0) A[0] = 0.282094791773878;
        else if (L0 == 1) {
            A[0] = 0.48860251190292 * sq * cp; A[1] = 0.48860251190292 * sq * sp; A[2] = 0.48860251190292 * cq;
        }
        else if (L0 == 2) {
            A[0] = 0.94617469575756 * cq * cq - 0.31539156525252;
            A[1] = 0.54627421529604 * sq * sq * (1 - 2 * sp * sp);
            A[2] = 1.09254843059208 * sq * sq * sp * cp;
            A[3] = 1.09254843059208 * sq * cq * cp;
            A[4] = 1.09254843059208 * sq * cq * sp;
        }
        else {
            A[0] = 0.373176332590116 * (5 * cq * cq * cq - 3 * cq);
            A[1] = 0.457045799464466 * cp * sq * (5 * cq * cq - 1);
            A[2] = 0.457045799464466 * sp * sq * (5 * cq * cq - 1);
            A[3] = 1.44530572132028 * sq * sq * cq * (cp * cp - sp * sp);
            A[4] = 2.89061144264055 * sq * sq * cq * sp * cp;
            A[5] = 0.590043589926644 * sq * sq * sq * (4 * cp * cp * cp - 3 * cp);
            A[6] = 0.590043589926644 * sq * sq * sq * (3 * sp - 4 * sp * sp * sp);
        }
        for (int mul = 0; mul < s.nb[L0]; mul++, row++) {
            const ld rf = radial(s, L0, row, r);
            for (int M0 = 0; M0 <= 2 * L0; M0++) out[i++] = rf * A[M0];
        }
    }
}

// the device's coordinates and radius (the same FP64 expressions)
static void point(const OTF_Atom &a, int GNc, double *x, double *y, double *z, double *R)
{
    const int ng23 = NG[1] * NG[2], ng3 = NG[2];
    const int n1 = GNc / ng23, n2 = (GNc - n1 * ng23) / ng3, n3 = GNc - n1 * ng23 - n2 * ng3;
    *x = ((double)n1 * GT[0][0] + (double)n2 * GT[1][0] + (double)n3 * GT[2][0] + ORG[0]) + ATV[0] - a.gx - a.ax;
    *y = ((double)n1 * GT[0][1] + (double)n2 * GT[1][1] + (double)n3 * GT[2][1] + ORG[1]) + ATV[1] - a.gy - a.ay;
    *z = ((double)n1 * GT[0][2] + (double)n2 * GT[1][2] + (double)n3 * GT[2][2] + ORG[2]) + ATV[2] - a.gz - a.az;
    const double dum = *x * *x + *y * *y;
    *R = sqrt(dum + *z * *z);
}

static double ulpf(ld v)
{
    const float f = (float)fabsl(v);
    return (double)(nextafterf(f, INFINITY) - f);
}

// ---- items ----

struct Items {
    std::vector<OTF_Atom> atoms;
    std::vector<int> npts, idx;
    std::vector<size_t> idx_off, out_off, sph_off;
    size_t out_size = 0, points = 0;
    int max_npts = 1;
    bool use_idx = false;
};

struct Device {
    OTF_Atom *atoms;
    int *npts, *idx, *gla, *cla;
    size_t *idx_off, *out_off, *sph_off;
    float *out;
};

template <class T> static T *upload(const std::vector<T> &v)
{
    T *d = nullptr;
    CK(hipMalloc((void **)&d, sizeof(T) * (v.empty() ? 1 : v.size())));
    if (!v.empty()) CK(hipMemcpy(d, v.data(), sizeof(T) * v.size(), hipMemcpyHostToDevice));
    return d;
}

static Device to_device(const Items &it, const std::vector<int> &gla, const std::vector<int> &cla)
{
    Device d;
    d.atoms = upload(it.atoms);
    d.npts = upload(it.npts);
    d.idx = upload(it.idx);
    d.idx_off = upload(it.idx_off);
    d.out_off = upload(it.out_off);
    d.sph_off = upload(it.sph_off);
    d.gla = upload(gla);
    d.cla = upload(cla);
    CK(hipMalloc((void **)&d.out, sizeof(float) * it.out_size));
    return d;
}

static void release(Device &d)
{
    CK(hipFree(d.atoms)); CK(hipFree(d.npts)); CK(hipFree(d.idx)); CK(hipFree(d.idx_off));
    CK(hipFree(d.out_off)); CK(hipFree(d.sph_off)); CK(hipFree(d.gla)); CK(hipFree(d.cla)); CK(hipFree(d.out));
}

// one evaluation: its items, the evaluator, its values and best kernel time
struct Job {
    const Items *it;
    const Device *d;
    EvalFn fn;
    std::vector<float> *out;
    double ms;
};

// the jobs in turn, reps rounds (a shared or throttled GPU slows them
// alike); the best time of each, its values from the last round
static void run(std::vector<Job> &jobs, int reps)
{
    hipEvent_t e0, e1;
    CK(hipEventCreate(&e0));
    CK(hipEventCreate(&e1));
    for (Job &j : jobs) j.ms = 1e30;
    for (int r = 0; r < reps; r++) {
        for (Job &j : jobs) {
            const Items &it = *j.it;
            const Device &d = *j.d;
            float ms;
            CK(hipMemset(d.out, 0xff, sizeof(float) * it.out_size));   // NaN
            CK(hipEventRecord(e0, nullptr));
            CHECK(j.fn(nullptr, (int)it.atoms.size(), it.max_npts, d.atoms, d.npts, d.idx_off, d.out_off, d.sph_off,
                       it.use_idx ? d.idx : nullptr, d.gla, d.cla, d.out) == 0);
            CK(hipEventRecord(e1, nullptr));
            CK(hipEventSynchronize(e1));
            CK(hipEventElapsedTime(&ms, e0, e1));
            j.ms = std::min(j.ms, (double)ms);
            if (r == reps - 1) {
                j.out->assign(it.out_size, 0.0f);
                CK(hipMemcpy(j.out->data(), d.out, sizeof(float) * it.out_size, hipMemcpyDeviceToHost));
            }
        }
    }
    CK(hipEventDestroy(e0));
    CK(hipEventDestroy(e1));
}

struct Stats {
    size_t values = 0, ref_points = 0, ref_values = 0, end_points = 0, below_points = 0;
    size_t df_ulp = 0, df_tiny = 0, base_df = 0, base64 = 0, base64_end = 0;
    double worst_df = 0, worst64 = 0, worst_df_vs64 = 0, base64_diff = 0, max_value = 0;
};

// every value of the items: DF against FP64 (and the baselines); the
// reference on the special points and every sample-th point
static void compare(const Items &it, const std::vector<int> &gla, const std::vector<float> &df,
                    const std::vector<float> &f64, const std::vector<float> *bdf, const std::vector<float> *b64,
                    int sample, Stats &st)
{
    std::vector<ld> ref(64);
    size_t seen = 0;
    for (size_t p = 0; p < it.atoms.size(); p++) {
        const OTF_Atom &a = it.atoms[p];
        const Species &s = SP[a.wan];
        for (int k = 0; k < it.npts[p]; k++, seen++) {
            const int Nc = it.use_idx ? it.idx[it.idx_off[p] + k] : k;
            const int GNc = gla[it.sph_off[p] + Nc];
            const size_t o = it.out_off[p] + (size_t)k * a.no;
            double x, y, z, R;
            point(a, GNc, &x, &y, &z, &R);
            const bool below = R < s.rv[0], beyond = s.rv[s.mesh - 1] < R;
            const int m = (below || beyond) ? 0 : interval(s, R);
            const bool end = (m == 1 || m == s.mesh - 1);
            const bool check = end || below || seen % sample == 0;
            st.end_points += end;
            st.below_points += below;
            if (check) {
                reference(s, x, y, z, ref.data());
                st.ref_points++;
            }
            for (int i = 0; i < a.no; i++) {
                const float vd = df[o + i], v6 = f64[o + i];
                CHECK(!std::isnan(vd) && !std::isnan(v6));
                if (vd != v6) {
                    const double d = fabs((double)vd - (double)v6), u = ulpf(v6);
                    if (fabs(v6) < 1e-12) {
                        CHECK(d <= 2e-12);
                        st.df_tiny++;
                    }
                    else {
                        st.df_ulp++;
                        st.worst_df_vs64 = std::max(st.worst_df_vs64, d / u);
                    }
                }
                if (bdf != nullptr && memcmp(&(*bdf)[o + i], &vd, sizeof(float)) != 0) st.base_df++;
                if (b64 != nullptr && memcmp(&(*b64)[o + i], &v6, sizeof(float)) != 0) {
                    st.base64++;
                    st.base64_end += end;
                    st.base64_diff = std::max(st.base64_diff, fabs((double)(*b64)[o + i] - (double)v6));
                }
                st.max_value = std::max(st.max_value, fabs((double)v6));
                if (check) {
                    const double u = ulpf(ref[i]);
                    const double e6 = (double)fabsl((ld)v6 - ref[i]), ed = (double)fabsl((ld)vd - ref[i]);
                    if (!(e6 <= u + 1e-14) || !(ed <= u + 1e-12)) {
                        fprintf(stderr, "point %d of item %zu (species %d, R %.17g, m %d) orbital %d: DF %.9g FP64 "
                                "%.9g reference %.12Lg\n", k, p, a.wan, R, m, i, vd, v6, ref[i]);
                        exit(2);
                    }
                    if (fabsl(ref[i]) >= 1e-12L) {
                        st.worst64 = std::max(st.worst64, e6 / u);
                        st.worst_df = std::max(st.worst_df, ed / u);
                    }
                    st.ref_values++;
                }
                st.values++;
            }
        }
    }
}

static void tables(OTF_HostTables *t, std::vector<size_t> &rvo, std::vector<size_t> &rwb, std::vector<int> &spm,
                   std::vector<int> &spx, std::vector<int> &spb, std::vector<double> &rv, std::vector<double> &rwf)
{
    for (const Species &s : SP) {
        rvo.push_back(rv.size());
        rwb.push_back(rwf.size());
        spm.push_back(s.mesh);
        spx.push_back(s.maxl);
        for (int L0 = 0; L0 < 4; L0++) spb.push_back(s.nb[L0]);
        rv.insert(rv.end(), s.rv.begin(), s.rv.end());
        for (const std::vector<double> &row : s.rw) rwf.insert(rwf.end(), row.begin(), row.end());
    }
    rvo.push_back(rv.size());
    rwb.push_back(rwf.size());
    memset(t, 0, sizeof(*t));
    t->species = (int)SP.size();
    t->rv_off = rvo.data();
    t->rwf_base = rwb.data();
    t->mesh = spm.data();
    t->maxl = spx.data();
    t->nb = spb.data();
    t->rv = rv.data();
    t->rwf = rwf.data();
    t->rv_count = rv.size();
    t->rwf_count = rwf.size();
    t->atv = ATV;
    t->atv_rows = 1;
    t->ng23 = NG[1] * NG[2];
    t->ng3 = NG[2];
    for (int i = 0; i < 3; i++) {
        t->org[i] = ORG[i];
        for (int j = 0; j < 3; j++) t->g[i][j] = GT[i][j];
    }
}

int main(int argc, char **argv)
{
    const int natoms = (argc > 1) ? atoi(argv[1]) : 6;
    const double spacing = (argc > 2) ? atof(argv[2]) : 0.20;
    const int reps = (argc > 3) ? atoi(argv[3]) : 3;
    std::vector<OTF_Atom> atoms;
    std::vector<std::vector<int>> sphere;
    std::vector<int> gla, cla;
    std::vector<size_t> sph_start;
    Items S, P;

    make_species();
    {
        const double box = 2.0 * 7.5 + 6.0;   // cutoffs up to 7.39 around atoms within 3 bohr of the centre
        const int n = (int)ceil(box / spacing);
        NG[0] = NG[1] = NG[2] = n;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) GT[i][j] = (i == j) ? spacing : 0.01 * spacing * (i - j);
            ORG[i] = -0.5 * n * spacing;
        }
    }
    // the atoms: the first on a grid point (R = 0), the second 3e-4 bohr off one (inside rv[0]),
    // the third 5.6e-4 off one (between rv[0] and rv[1]: the interval m = 1), then random
    srand(12345);
    for (int a = 0; a < natoms; a++) {
        OTF_Atom at;
        memset(&at, 0, sizeof(at));
        at.wan = a % 3;
        at.no = SP[at.wan].no;
        const int GNc = (NG[0] / 2 + (a % 3) - 1) * NG[1] * NG[2] + (NG[1] / 2 + a % 2) * NG[2] + NG[2] / 2 - a % 4;
        OTF_Atom zero;
        memset(&zero, 0, sizeof(zero));
        double x, y, z, R;
        point(zero, GNc, &x, &y, &z, &R);
        static const double off[3] = {0.0, 3e-4, 5.6e-4};
        at.gx = x + (a < 3 ? off[a] : 0.37 * spacing * ((double)rand() / RAND_MAX - 0.5) * 8);
        at.gy = y + (a < 3 ? 0.0 : 0.37 * spacing * ((double)rand() / RAND_MAX - 0.5) * 8);
        at.gz = z + (a < 3 ? 0.0 : 0.37 * spacing * ((double)rand() / RAND_MAX - 0.5) * 8);
        atoms.push_back(at);
    }
    // the spheres: the grid points within each atom's cutoff
    for (const OTF_Atom &at : atoms) {
        const Species &s = SP[at.wan];
        std::vector<int> list;
        sph_start.push_back(gla.size());
        for (int GNc = 0; GNc < NG[0] * NG[1] * NG[2]; GNc++) {
            double x, y, z, R;
            point(at, GNc, &x, &y, &z, &R);
            if (R <= s.rv[s.mesh - 1] * 1.0001) {
                list.push_back(GNc);
                gla.push_back(GNc);
                cla.push_back(0);
            }
        }
        sphere.push_back(list);
    }
    // central-atom tiles: every sphere point, no index list
    for (int a = 0; a < natoms; a++) {
        S.atoms.push_back(atoms[a]);
        S.npts.push_back((int)sphere[a].size());
        S.idx_off.push_back(0);
        S.sph_off.push_back(sph_start[a]);
        S.out_off.push_back(S.out_size);
        S.out_size += sphere[a].size() * (size_t)atoms[a].no;
        S.points += sphere[a].size();
        S.max_npts = std::max(S.max_npts, (int)sphere[a].size());
    }
    // neighbour tiles: atom b at the points of centre a's sphere within b's cutoff (centres 0..2)
    P.use_idx = true;
    for (int a = 0; a < std::min(natoms, 3); a++) {
        for (int b = 0; b < natoms; b++) {
            if (b == a) continue;
            const OTF_Atom &nb = atoms[b];
            const size_t first = P.idx.size();
            for (int Nc = 0; Nc < (int)sphere[a].size(); Nc++) {
                double x, y, z, R;
                point(nb, sphere[a][Nc], &x, &y, &z, &R);
                if (R <= SP[nb.wan].rv[SP[nb.wan].mesh - 1]) P.idx.push_back(Nc);
            }
            const int n = (int)(P.idx.size() - first);
            P.atoms.push_back(nb);
            P.npts.push_back(n);
            P.idx_off.push_back(first);
            P.sph_off.push_back(sph_start[a]);
            P.out_off.push_back(P.out_size);
            P.out_size += (size_t)n * (size_t)nb.no;
            P.points += n;
            P.max_npts = std::max(P.max_npts, n);
        }
    }

    OTF_HostTables t;
    std::vector<size_t> rvo, rwb;
    std::vector<int> spm, spx, spb;
    std::vector<double> rv, rwf;
    tables(&t, rvo, rwb, spm, spx, spb, rv, rwf);

    Device dS = to_device(S, gla, cla), dP = to_device(P, gla, cla);
    std::vector<float> sdf, s64f, s64, pdf, p64f, p64;
    double ms[2][3];
    std::vector<float> *bsdf = nullptr, *bs64 = nullptr, *bpdf = nullptr, *bp64 = nullptr;
#ifdef OTF_BASELINE
    std::vector<float> b[4];
    double bms[2][2];
    bsdf = &b[0];
    bs64 = &b[1];
    bpdf = &b[2];
    bp64 = &b[3];
#endif

    // double-float upload: DF and forced FP64 (and the baseline's DF), in turn
    {
        std::vector<Job> jobs = {{&S, &dS, OTF_Orbitals_Eval, &sdf, 0}, {&S, &dS, OTF_Orbitals_Eval_FP64, &s64f, 0},
                                 {&P, &dP, OTF_Orbitals_Eval, &pdf, 0}, {&P, &dP, OTF_Orbitals_Eval_FP64, &p64f, 0}};
        CHECK(OTF_Orbitals_Upload(&t, 1) == 1);
#ifdef OTF_BASELINE
        CHECK(baseline_OTF_Orbitals_Upload(&t, 1) == 1);
        jobs.push_back({&S, &dS, baseline_OTF_Orbitals_Eval, &b[0], 0});
        jobs.push_back({&P, &dP, baseline_OTF_Orbitals_Eval, &b[2], 0});
#endif
        run(jobs, reps);
        ms[0][0] = jobs[0].ms;
        ms[0][1] = jobs[1].ms;
        ms[1][0] = jobs[2].ms;
        ms[1][1] = jobs[3].ms;
#ifdef OTF_BASELINE
        bms[0][0] = jobs[4].ms;
        bms[1][0] = jobs[5].ms;
#endif
        // FP64 upload: the FP64 tiles (and the baseline's FP64)
        jobs = {{&S, &dS, OTF_Orbitals_Eval, &s64, 0}, {&P, &dP, OTF_Orbitals_Eval, &p64, 0}};
        CHECK(OTF_Orbitals_Upload(&t, 0) == 1);
#ifdef OTF_BASELINE
        CHECK(baseline_OTF_Orbitals_Upload(&t, 0) == 1);
        jobs.push_back({&S, &dS, baseline_OTF_Orbitals_Eval, &b[1], 0});
        jobs.push_back({&P, &dP, baseline_OTF_Orbitals_Eval, &b[3], 0});
#endif
        run(jobs, reps);
        ms[0][2] = jobs[0].ms;
        ms[1][2] = jobs[1].ms;
        OTF_Orbitals_Release();
#ifdef OTF_BASELINE
        bms[0][1] = jobs[2].ms;
        bms[1][1] = jobs[3].ms;
        baseline_OTF_Orbitals_Release();
#endif
    }
    CHECK(memcmp(s64.data(), s64f.data(), sizeof(float) * S.out_size) == 0);
    CHECK(memcmp(p64.data(), p64f.data(), sizeof(float) * P.out_size) == 0);
    Stats ss, ps;
    compare(S, gla, sdf, s64, bsdf, bs64, 97, ss);
    compare(P, gla, pdf, p64, bpdf, bp64, 97, ps);

    const char *name[2] = {"central-atom tiles", "neighbour tiles"};
    const Items *it[2] = {&S, &P};
    const Stats *st[2] = {&ss, &ps};
    printf("%d atoms (species: 13, 5 and 25 orbitals; meshes of 500 points), grid %d^3 at %.3f bohr\n", natoms, NG[0],
           spacing);
    for (int k = 0; k < 2; k++) {
        const double mp = (double)it[k]->points * 1e-6;
        printf("%s: %zu items, %zu points, %zu values\n", name[k], it[k]->atoms.size(), it[k]->points, st[k]->values);
        printf("  kernel ms: double-float %.3f (%.1f Mpoints/s), FP64 forced %.3f, FP64 upload %.3f (%.1f Mpoints/s)\n",
               ms[k][0], mp / (ms[k][0] * 1e-3), ms[k][1], ms[k][2], mp / (ms[k][2] * 1e-3));
        printf("  FP64 forced = FP64 upload bitwise; double-float vs FP64: %zu values a float apart (worst %.2f ulp), "
               "%zu below 1e-12 apart by < 2e-12\n", st[k]->df_ulp, st[k]->worst_df_vs64, st[k]->df_tiny);
        printf("  reference (long double) on %zu points / %zu values (%zu in the end intervals, %zu inside rv[0]): "
               "worst FP64 %.2f ulp, double-float %.2f ulp\n", st[k]->ref_points, st[k]->ref_values,
               st[k]->end_points, st[k]->below_points, st[k]->worst64, st[k]->worst_df);
#ifdef OTF_BASELINE
        printf("  baseline: double-float %.3f ms, FP64 %.3f ms; values differing from it: double-float %zu, FP64 %zu "
               "(%zu at points of the end intervals; largest difference %.3e, largest value %.3e)\n", bms[k][0],
               bms[k][1], st[k]->base_df, st[k]->base64, st[k]->base64_end, st[k]->base64_diff, st[k]->max_value);
        CHECK(st[k]->base_df == 0 && st[k]->base64 == st[k]->base64_end);
#endif
    }
    release(dS);
    release(dP);
    printf("PASS\n");
    return 0;
}

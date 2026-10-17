#!/usr/bin/env python3
"""Check the control flow of the non-collinear band refined eigensolver on the CPU.

The stage machine and the refined-solve helpers of source/Band_DFT_NonCol.c
(BandNonCol_Refine*, BandNonCol_KRef*, BandNonCol_RootDenseRefinedFirstHalf
and ...SecondHalf, Band_DFT_NonCol_Refine*) are cut out of the production
file and built against host stubs: MPI as a single rank, hipMalloc/hipFree
on host memory with failure injection and a leak count, and a scripted mock
of openmx_eigen_refine_prepare/finish (eigen_refine_gpu.h); the transformed
Hamiltonian build and the FP64 completion of a k-point are stubs that
record their calls.  No device code, MPI or eigensolver runs.

Checked, on the resident path (every k-world owner in one turn group,
refined from the first SCF step on):
  - the first step is transient and cold, its estimates go to ko and EIGEN,
    the solve stays pending with its device buffers, the second half takes
    the occupations of the estimates at the given chemical potential,
    returns the refined eigenvalues, keeps the basis in the k-point's host
    slot (OPENMX_EIGEN_REFINE_WARM_BAND), frees every device buffer and
    completes the k-point as refined;
  - the next step starts warm from the slot (the slot's basis uploaded),
    a warm start too far from the new matrix is redone cold, a warm period
    step is cold, WARM_BAND=0 keeps no slot;
  - a failed prepare, finish or allocation falls back to FP64 with every
    device buffer freed; outside the first step a failure condemns the
    cycle (EndStep sets the final stage, the next BeginStep stays in FP64),
    in the first step it does not; persistent 2 turns the refinement off
    for the run;
on the two-pass path: the solve in one go, an allocation failure does not
condemn the cycle; the stage machine: the paths' gates (resident from step
1, two-pass from step 2, force_fp64, the last allowed step, XANES, emptied
states), the run-boundary reset (CL-V7: the next input reads the
configuration again), the configuration's OPENMX_EIGEN_REFINE_UNTIL from
the root (BD-V8), the memory test's base folding, its one collective per
call and its turn count, EndCycle, and the run-boundary release (no device
buffer left).  --cc (or CC) and CFLAGS select the compiler, --sanitize adds
ASan/UBSan.
"""

import argparse
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def function(source, name):
    match = re.search(r"^[A-Za-z][^;{}\n]*\b" + re.escape(name) + r"\([^;{]*?\)\s*\{", source, re.M)
    if match is None:
        raise ValueError(f"Cannot find production function {name}")
    pos, depth = source.index("{", match.start()) + 1, 1
    while depth:
        depth += (source[pos] == "{") - (source[pos] == "}")
        pos += 1
    return source[match.start():pos]


def block(source, first, last):
    start = source.index(first)
    end = source.index(last, start) + len(last)
    return source[start:end]


PREFIX = r'''
static int BandNonCol_mgpu_host_dense=0;
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "openmx_precision.h"

#define CHECK(condition) do { if (!(condition)) { \
    fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #condition); exit(2); } } while (0)

typedef struct { double r, i; } dcomplex;
typedef int hipError_t;
typedef void *hipblasHandle_t;
enum { hipSuccess = 0, hipErrorOutOfMemory = 2, hipMemcpyHostToDevice = 1, hipMemcpyDeviceToHost = 2 };
#define wait_hipfunc(call) CHECK((int)(call) == hipSuccess)

/* ---- the device: host memory with a live count and failure injection */
#define MAX_LIVE 64
static void *live[MAX_LIVE];
static size_t live_bytes[MAX_LIVE];
static int malloc_calls;
static size_t fail_malloc_bytes;
static int live_count(void) { int c = 0; for (int i = 0; i < MAX_LIVE; i++) c += (live[i] != NULL); return c; }
static hipError_t hipMalloc(void **ptr, size_t bytes)
{
    malloc_calls++;
    if (fail_malloc_bytes != 0 && bytes == fail_malloc_bytes) { *ptr = NULL; return hipErrorOutOfMemory; }
    *ptr = malloc(bytes);
    CHECK(*ptr != NULL);
    for (int i = 0; i < MAX_LIVE; i++) if (live[i] == NULL) { live[i] = *ptr; live_bytes[i] = bytes; return hipSuccess; }
    CHECK(0);
    return hipSuccess;
}
static hipError_t hipFree(void *ptr)
{
    for (int i = 0; i < MAX_LIVE; i++) if (live[i] == ptr) { live[i] = NULL; free(ptr); return hipSuccess; }
    CHECK(0);
    return hipSuccess;
}
static hipError_t hipGetLastError(void) { return hipSuccess; }
static int h2d_bytes, d2h_bytes;
static hipError_t hipMemcpy(void *dst, const void *src, size_t bytes, int kind)
{
    if (kind == hipMemcpyHostToDevice) h2d_bytes += (int)bytes; else d2h_bytes += (int)bytes;
    memcpy(dst, src, bytes);
    return hipSuccess;
}

/* ---- MPI as one rank */
typedef int MPI_Comm;
enum { MPI_INT = 1, MPI_DOUBLE = 2, MPI_MIN = 3, MPI_MAX = 4 };
static MPI_Comm mpi_comm_level1 = 0;
static int Host_ID = 0, allreduce_calls, bcast_calls;
static int MPI_Comm_rank(MPI_Comm comm, int *rank) { (void)comm; *rank = 0; return 0; }
static int MPI_Bcast(void *buf, int count, int type, int root, MPI_Comm comm)
{ (void)buf; (void)count; (void)type; (void)root; (void)comm; bcast_calls++; return 0; }
static int MPI_Allreduce(const void *send, void *recv, int count, int type, int op, MPI_Comm comm)
{
    (void)op; (void)comm;
    allreduce_calls++;
    memcpy(recv, send, (size_t)count * (type == MPI_DOUBLE ? sizeof(double) : sizeof(int)));
    return 0;
}

/* ---- the SCF's globals */
static int level_stdout = 1, Cnt_switch = 0, DFTSCF_loop = 40, xanes_calc = 0, empty_states_flag = 0;
static double Beta = 50.0, system_charge = 0.0;
static double FermiFunc_NC(double x, int index) { (void)index; return 1.0 / (1.0 + exp(x)); }
static int local_ranks = 4;
static int openmx_gpu_local_size_noncollective(void) { return local_ranks; }
static size_t free_device = (size_t)8 << 30;
static int OpenMX_GpuMemGetInfo(size_t *free_bytes, size_t *total_bytes)
{ *free_bytes = free_device; *total_bytes = (size_t)16 << 30; return 0; }
static int fp32_work_queries;

static void BandNonCol_AbortWithMessage(const char *msg) { fprintf(stderr, "%s\n", msg); exit(3); }

/* ---- the rest of the module the helpers touch */
typedef struct { hipblasHandle_t hipblas; dcomplex *d_H2; } BandNonColDenseGpuWorkspace;
static BandNonColDenseGpuWorkspace BandNonCol_dense_gpu_workspace;
static int handle_ensures;
static void BandNonCol_DenseGpu_EnsureHipblas(void)
{ handle_ensures++; BandNonCol_dense_gpu_workspace.hipblas = (hipblasHandle_t)&handle_ensures; }
static int BandNonCol_GemmWorkspaceTurnRelease(void) { return 1; }
static int gemmul8_releases, gemmul8_trims;
static void openmx_gemmul8ReleaseWorkspaces(void) { gemmul8_releases++; }
static void openmx_gemmul8TrimWorkspaces(void) { gemmul8_trims++; }
typedef struct { int dummy; } BandNonColRootDenseWorkspace;
'''

MOCK = r'''
/* ---- eigen_refine_gpu, scripted */
static int config_iterations = 2, config_ratio = 16, configure_calls;
static double config_until = 3.0e-9;
static int warm_band = 1, warm_period = 0;
static int script_prepare[16], script_prepare_persistent[16], n_prepare, prepare_calls;
static int script_finish[16], script_finish_persistent[16], n_finish, finish_calls;
static int last_warm, last_transient;
static double last_f0[64];
static dcomplex expected_warm_x[64 * 64];
static int check_warm_x;

/* exactly representable values, so that no contraction or folding of the
   compiler can make two evaluations differ */
static double estimate(int i) { return -1.0 + 0.125 * (double)i; }
static double refined(int i) { return -1.0 + 0.125 * (double)i + 0x1p-20; }
static double fp64(int i) { return -1.0 + 0.125 * (double)i + 0x1p-19; }

int openmx_eigen_refine_configure(int cplx, int verbose, int *ratio)
{ (void)cplx; (void)verbose; configure_calls++; *ratio = config_ratio; return config_iterations; }
double openmx_eigen_refine_until(void) { return config_until; }
int openmx_eigen_refine_warm_band_enabled(void) { return warm_band; }
int openmx_eigen_refine_warm_band_step(int scf_iter) { return warm_band && !(0 < warm_period && scf_iter % warm_period == 0); }
size_t openmx_eigen_refine_fp32_work_bytes(int cplx, int n) { (void)cplx; (void)n; fp32_work_queries++; return 4096; }
void openmx_eigen_refine_state_release(EigenRefineState *st)
{
    if (st->lam != NULL) CHECK(hipFree(st->lam) == hipSuccess);
    if (st->occ != NULL) CHECK(hipFree(st->occ) == hipSuccess);
    st->lam = st->occ = NULL;
    st->capacity = 0;
    st->basis_valid = 0;
}
void openmx_eigen_refine_device_release(EigenRefineDevice *dev) { (void)dev; }

int openmx_eigen_refine_prepare(EigenRefineState *st, EigenRefineDevice *dev, const EigenRefineProblem *pb,
                                double *e0, EigenRefineReport *rep)
{
    const size_t nn = (size_t)pb->n * (size_t)pb->n;
    int r = (prepare_calls < n_prepare) ? script_prepare[prepare_calls] : 1;
    int p = (prepare_calls < n_prepare) ? script_prepare_persistent[prepare_calls] : 0;

    memset(rep, 0, sizeof(*rep));
    prepare_calls++;
    last_warm = pb->warm;
    last_transient = pb->transient;
    CHECK(pb->cplx == 1 && pb->a == BandNonCol_dense_gpu_workspace.d_H2 && pb->x != NULL && pb->w != NULL);
    CHECK(pb->fp32 != NULL && pb->region >= 3 * (((size_t)pb->n * (size_t)pb->maxn * 16 + 511) / 512 * 512));
    CHECK(dev->hipblas != NULL && dev->shared == 1 && dev->try_malloc != NULL && dev->d_info != NULL);
    CHECK(pb->iterations == config_iterations && pb->occupation != NULL);
    if (st->capacity < pb->n) {
        CHECK(dev->try_malloc((void **)&st->lam, (size_t)pb->n * sizeof(double)) == hipSuccess);
        CHECK(dev->try_malloc((void **)&st->occ, (size_t)pb->n * sizeof(double)) == hipSuccess);
        st->capacity = pb->n;
    }
    if (pb->warm) {
        CHECK(st->basis_valid == 1 && st->basis_n == pb->n);
        if (check_warm_x) CHECK(memcmp(pb->x, expected_warm_x, nn * sizeof(dcomplex)) == 0);
        if (r < 0) return -1;
    }
    else {
        CHECK(r >= 0);
        st->basis_valid = 0;
    }
    if (r == 0) {
        /* as the module reports: a failure of a transient solve does not
           condemn the cycle, a native fallback of the products (2) ends
           the refinement whatever the step */
        rep->persistent = (p == 2) ? 2 : (pb->transient ? 0 : p);
        return 0;
    }
    for (int i = 0; i < pb->n; i++) e0[i] = estimate(i);
    if (!pb->warm) {
        dcomplex *x = (dcomplex *)pb->x;
        for (size_t i = 0; i < nn; i++) { x[i].r = 1.0 + (double)i; x[i].i = 0.0; }
    }
    memcpy(st->lam, e0, (size_t)pb->n * sizeof(double));
    return 1;
}

int openmx_eigen_refine_finish(EigenRefineState *st, EigenRefineDevice *dev, const EigenRefineProblem *pb,
                               const double *f0, EigenRefineReport *rep)
{
    const size_t nn = (size_t)pb->n * (size_t)pb->n;
    int r = (finish_calls < n_finish) ? script_finish[finish_calls] : 1;
    int p = (finish_calls < n_finish) ? script_finish_persistent[finish_calls] : 0;
    dcomplex *x = (dcomplex *)pb->x;

    (void)dev;
    finish_calls++;
    memcpy(last_f0, f0, (size_t)pb->n * sizeof(double));
    if (r == 0) {
        rep->persistent = p;
        st->basis_valid = 0;
        return 0;
    }
    for (int i = 0; i < pb->maxn; i++) pb->w[i] = refined(i);
    for (int i = 0; i < pb->n; i++) st->lam[i] = refined(i);
    for (size_t i = 0; i < nn; i++) { x[i].r = 100.0 + (double)i + 1000.0 * (double)finish_calls; x[i].i = -1.0; }
    st->basis_valid = 1;
    st->basis_n = pb->n;
    rep->columns = pb->maxn;
    return 1;
}

/* ---- the build and the completion of a k-point */
static int builds, completes, complete_refined = -1;
static void BandNonCol_RootDenseBuildH2(int rebuild_overlap, int n, int n2, int MaxN, double k1, double k2, double k3,
                                        const double *m_olp, const double *m_h11, const double *m_h22,
                                        const double *m_h12, const double *m_h12i, const double *m_i11,
                                        const double *m_i22, const double *m_i12, int *packed_order_GA, int *MP,
                                        double *ko, BandNonColRootDenseWorkspace *rdw)
{
    (void)rebuild_overlap; (void)n; (void)MaxN; (void)k1; (void)k2; (void)k3; (void)m_olp; (void)m_h11;
    (void)m_h22; (void)m_h12; (void)m_h12i; (void)m_i11; (void)m_i22; (void)m_i12; (void)packed_order_GA;
    (void)MP; (void)ko; (void)rdw;
    builds++;
    if (BandNonCol_dense_gpu_workspace.d_H2 == NULL)
        BandNonCol_dense_gpu_workspace.d_H2 = calloc((size_t)n2 * (size_t)n2, sizeof(dcomplex));
}
static void BandNonCol_RootDenseCompleteOneK(int n, int n2, int MaxN, int kloop, int refined_solve, double *ko,
                                             double ***EIGEN, int need_evec, int copy_evec_to_host,
                                             int defer_backtransform, BandNonColRootDenseWorkspace *rdw,
                                             dcomplex *evec_panel)
{
    (void)n; (void)n2; (void)need_evec; (void)copy_evec_to_host; (void)defer_backtransform; (void)rdw;
    (void)evec_panel;
    completes++;
    complete_refined = refined_solve;
    if (!refined_solve) for (int l = 1; l <= MaxN; l++) ko[l] = fp64(l - 1);
    for (int l = 1; l <= MaxN; l++) EIGEN[0][kloop][l] = ko[l];
}
'''

HARNESS = r'''
enum { N = 6, N2 = 12, MAXN = 7, TKNUM = 3 };
static double ko[N2 + 1];
static double eig_k[TKNUM][MAXN + 1];
static double *eig_rows[TKNUM] = {eig_k[0], eig_k[1], eig_k[2]};
static double **eig_spin[1] = {eig_rows};
static double ***EIGEN = eig_spin;
static BandNonColRootDenseWorkspace rdw;
static unsigned long checks;

static void script(int *results, int *persistent, int *count, int *calls, const int *r, const int *p, int m)
{
    for (int i = 0; i < m; i++) { results[i] = r[i]; persistent[i] = p[i]; }
    *count = m;
    *calls = 0;
}

/* one call's refined solve on the resident path; returns the first half's verdict */
static int resident_step(int scf_iter, int kloop, double mu)
{
    int pending;

    BandNonCol_RefineBeginStep(scf_iter, 1, 0, 0);
    BandNonCol_KRefEnsure(TKNUM, N2);
    pending = BandNonCol_RootDenseRefinedFirstHalf(scf_iter == 1, N, N2, MAXN, kloop, 0.0, 0.0, 0.0,
                                                   NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
                                                   ko, EIGEN, 1, 0, &rdw);
    CHECK(BandNonCol_refine.pending == pending);
    if (pending) {
        for (int l = 1; l <= MAXN; l++) CHECK(ko[l] == estimate(l - 1) && EIGEN[0][kloop][l] == estimate(l - 1));
        /* pending: the device buffers stay until the second half */
        CHECK(BandNonCol_refine_x != NULL && BandNonCol_refine_buf.d_scratch != NULL);
        BandNonCol_RootDenseRefinedSecondHalf(N, N2, MAXN, kloop, ko, EIGEN, mu, 1, 0, &rdw);
        CHECK(!BandNonCol_refine.pending);
        for (int i = 0; i < N2; i++) {
            double x = (estimate(i) - mu) * Beta;
            if (x <= -60.0) x = -60.0;
            if (60.0 <= x) x = 60.0;
            CHECK(last_f0[i] == FermiFunc_NC(x, i + 1));
        }
    }
    /* whatever happened, only the refinement state's own arrays stay */
    CHECK(BandNonCol_refine_x == NULL && BandNonCol_refine_buf.d_scratch == NULL);
    CHECK(BandNonCol_refine_buf.d_w == NULL && BandNonCol_refine_buf.d_info == NULL);
    CHECK(live_count() == ((BandNonCol_refine_state.lam != NULL) ? 2 : 0));
    BandNonCol_RefineEndStep();
    checks += 8;
    return pending;
}

int main(void)
{
    const int ok1[] = {1}, zero1[] = {0};

    setenv("OPENMX_BAND_REFINE_BASIS_MB", "64", 1);

    /* ---- the configuration: once, from the root, until with it */
    CHECK(BandNonCol_RefineIterations() == 0);
    openmx_set_fp32_enabled(0);
    BandNonCol_RefineConfigure();
    CHECK(configure_calls == 0 && bcast_calls == 0 && BandNonCol_RefineIterations() == 0);
    openmx_set_fp32_enabled(1);
    BandNonCol_RefineConfigure();
    CHECK(configure_calls == 1 && bcast_calls == 1 && BandNonCol_RefineIterations() == 2);
    CHECK(BandNonCol_refine.defaulted == 1 && BandNonCol_refine.until == 3.0e-9);
    BandNonCol_RefineConfigure();
    CHECK(configure_calls == 1 && bcast_calls == 1);
    openmx_set_fp32_enabled(0);
    CHECK(BandNonCol_RefineIterations() == 0);
    BandNonCol_RefineConfigure();
    CHECK(configure_calls == 1 && bcast_calls == 1);
    openmx_set_fp32_enabled(1);
    CHECK(BandNonCol_RefineIterations() == 2);

    /* ---- the memory test: one collective whatever the base, owners only ask */
    allreduce_calls = 0;
    CHECK(BandNonCol_RefineFits(0, N2, MAXN, 2, 1) == 0 && allreduce_calls == 1 && fp32_work_queries == 0);
    CHECK(BandNonCol_RefineFits(1, N2, MAXN, 2, 0) == 1 && allreduce_calls == 2 && fp32_work_queries == 0);
    CHECK(BandNonCol_RefineFits(1, N2, MAXN, 2, 1) == 1 && allreduce_calls == 3 && fp32_work_queries == 1);
    {
        /* the turns multiply the need: (extra + 3 n2^2 16) turns + 512 MiB */
        const size_t extra = sizeof(dcomplex) * N2 * N2 + BandNonCol_RefineScratchBytes(N2, MAXN) + 4096;
        const size_t one = extra + 3 * sizeof(dcomplex) * N2 * N2;
        free_device = one * 3 + ((size_t)512 << 20);
        CHECK(BandNonCol_RefineFits(1, N2, MAXN, 3, 1) == 1);
        CHECK(BandNonCol_RefineFits(1, N2, MAXN, 4, 1) == 0);
        CHECK(BandNonCol_RefineFits(1, N2, MAXN, 0, 1) == 1);
        free_device = (size_t)8 << 30;
    }
    checks += 12;

    /* ---- gates: resident from step 1, two-pass from step 2 */
    BandNonCol_RefineBeginStep(1, 1, 1, 1);
    CHECK(BandNonCol_refine.last_refined && !BandNonCol_refine.kdense && !BandNonCol_refine.staged);
    CHECK(BandNonCol_refine.transient);
    BandNonCol_RefineBeginStep(2, 1, 1, 1);
    CHECK(BandNonCol_refine.last_refined && BandNonCol_refine.kdense && BandNonCol_refine.staged);
    CHECK(!BandNonCol_refine.transient);
    BandNonCol_RefineBeginStep(DFTSCF_loop, 1, 1, 1);
    CHECK(!BandNonCol_refine.last_refined && !BandNonCol_refine.kdense && !BandNonCol_refine.staged);
    Band_DFT_NonCol_RefineForceFP64(1);
    BandNonCol_RefineBeginStep(3, 1, 1, 1);
    CHECK(!BandNonCol_refine.last_refined && !BandNonCol_refine.kdense);
    Band_DFT_NonCol_RefineForceFP64(0);
    xanes_calc = 1;
    BandNonCol_RefineBeginStep(3, 1, 1, 1);
    CHECK(!BandNonCol_refine.last_refined);
    xanes_calc = 0;
    empty_states_flag = 1;
    BandNonCol_RefineBeginStep(3, 1, 0, 0);
    CHECK(!BandNonCol_refine.last_refined);
    empty_states_flag = 0;
    BandNonCol_RefineBeginStep(3, 0, 0, 0);
    CHECK(!BandNonCol_refine.last_refined);
    checks += 10;

    /* ---- resident: step 1, transient and cold */
    script(script_prepare, script_prepare_persistent, &n_prepare, &prepare_calls, ok1, zero1, 1);
    script(script_finish, script_finish_persistent, &n_finish, &finish_calls, ok1, zero1, 1);
    CHECK(resident_step(1, 1, 0.05) == 1);
    CHECK(prepare_calls == 1 && last_warm == 0 && last_transient == 1 && finish_calls == 1);
    CHECK(complete_refined == 1 && completes == 1 && builds == 1);
    for (int l = 1; l <= MAXN; l++) CHECK(ko[l] == refined(l - 1) && EIGEN[0][1][l] == refined(l - 1));
    CHECK(BandNonCol_kref[1].valid && BandNonCol_kref[1].basis != NULL);
    CHECK(Band_DFT_NonCol_RefineLastRefined() == 1 && BandNonCol_refine.cycle_refined);
    /* the slot holds the refined basis of the finish */
    for (int i = 0; i < N2 * N2; i++) {
        expected_warm_x[i].r = 100.0 + (double)i + 1000.0 * 1.0;
        expected_warm_x[i].i = -1.0;
    }
    CHECK(memcmp(BandNonCol_kref[1].basis, expected_warm_x, sizeof(dcomplex) * N2 * N2) == 0);
    checks += 8;

    /* ---- step 2: warm from the slot */
    check_warm_x = 1;
    script(script_prepare, script_prepare_persistent, &n_prepare, &prepare_calls, ok1, zero1, 1);
    script(script_finish, script_finish_persistent, &n_finish, &finish_calls, ok1, zero1, 1);
    CHECK(resident_step(2, 1, 0.05) == 1);
    CHECK(prepare_calls == 1 && last_warm == 1 && last_transient == 0 && complete_refined == 1);
    check_warm_x = 0;

    /* ---- step 3: the warm start too far, redone cold */
    {
        const int r[] = {-1, 1}, p[] = {0, 0};
        script(script_prepare, script_prepare_persistent, &n_prepare, &prepare_calls, r, p, 2);
    }
    script(script_finish, script_finish_persistent, &n_finish, &finish_calls, ok1, zero1, 1);
    CHECK(resident_step(3, 1, 0.05) == 1);
    CHECK(prepare_calls == 2 && last_warm == 0 && complete_refined == 1);

    /* ---- a warm-period step is cold; WARM_BAND off keeps no slot */
    warm_period = 4;
    script(script_prepare, script_prepare_persistent, &n_prepare, &prepare_calls, ok1, zero1, 1);
    script(script_finish, script_finish_persistent, &n_finish, &finish_calls, ok1, zero1, 1);
    CHECK(resident_step(4, 1, 0.05) == 1 && last_warm == 0);
    warm_period = 0;
    warm_band = 0;
    script(script_prepare, script_prepare_persistent, &n_prepare, &prepare_calls, ok1, zero1, 1);
    script(script_finish, script_finish_persistent, &n_finish, &finish_calls, ok1, zero1, 1);
    CHECK(resident_step(5, 1, 0.05) == 1 && last_warm == 0);
    CHECK(!BandNonCol_kref[1].valid && BandNonCol_kref[1].basis == NULL);
    warm_band = 1;
    checks += 8;

    /* ---- failures in step 1 do not condemn the cycle */
    {
        const int r[] = {0}, p[] = {1};
        script(script_prepare, script_prepare_persistent, &n_prepare, &prepare_calls, r, p, 1);
    }
    completes = 0;
    CHECK(resident_step(1, 0, 0.0) == 0);
    CHECK(completes == 1 && complete_refined == 0);
    for (int l = 1; l <= MAXN; l++) CHECK(ko[l] == fp64(l - 1) && EIGEN[0][0][l] == fp64(l - 1));
    CHECK(BandNonCol_refine.persistent == 0 && !BandNonCol_refine.final_stage);
    /* an allocation failure of the first step does not condemn it either */
    fail_malloc_bytes = sizeof(dcomplex) * N2 * N2;
    script(script_prepare, script_prepare_persistent, &n_prepare, &prepare_calls, ok1, zero1, 1);
    CHECK(resident_step(1, 0, 0.0) == 0);
    CHECK(prepare_calls == 0 && BandNonCol_refine.persistent == 0 && !BandNonCol_refine.final_stage);
    fail_malloc_bytes = 0;
    checks += 6;

    /* ---- outside step 1 an allocation failure condemns the cycle */
    fail_malloc_bytes = sizeof(dcomplex) * N2 * N2;
    CHECK(resident_step(2, 0, 0.0) == 0);
    CHECK(BandNonCol_refine.final_stage == 1 && complete_refined == 0);
    fail_malloc_bytes = 0;
    BandNonCol_RefineBeginStep(3, 1, 1, 1);
    CHECK(!BandNonCol_refine.last_refined && !BandNonCol_refine.kdense);
    /* a new cycle starts over */
    BandNonCol_RefineBeginStep(1, 1, 0, 0);
    CHECK(BandNonCol_refine.last_refined && !BandNonCol_refine.final_stage && BandNonCol_kref == NULL);
    checks += 5;

    /* ---- a failed finish: FP64 on the preserved matrix, condemned (persistent 1) */
    script(script_prepare, script_prepare_persistent, &n_prepare, &prepare_calls, ok1, zero1, 1);
    {
        const int r[] = {0}, p[] = {1};
        script(script_finish, script_finish_persistent, &n_finish, &finish_calls, r, p, 1);
    }
    CHECK(resident_step(2, 2, 0.0) == 1);
    CHECK(complete_refined == 0 && BandNonCol_refine.final_stage == 1 && !BandNonCol_kref[2].valid);
    for (int l = 1; l <= MAXN; l++) CHECK(ko[l] == fp64(l - 1) && EIGEN[0][2][l] == fp64(l - 1));
    CHECK(Band_DFT_NonCol_RefineLastRefined() == 0);
    checks += 4;

    /* ---- persistent 2 ends the refinement for the run, until the run boundary */
    BandNonCol_RefineBeginStep(1, 1, 0, 0);
    BandNonCol_KRefEnsure(TKNUM, N2);
    {
        const int r[] = {0}, p[] = {2};
        script(script_prepare, script_prepare_persistent, &n_prepare, &prepare_calls, r, p, 1);
    }
    CHECK(BandNonCol_RootDenseRefinedFirstHalf(1, N, N2, MAXN, 0, 0, 0, 0, NULL, NULL, NULL, NULL, NULL, NULL,
                                               NULL, NULL, NULL, NULL, ko, EIGEN, 1, 0, &rdw) == 0);
    BandNonCol_RefineEndStep();
    CHECK(BandNonCol_RefineIterations() == 0 && BandNonCol_refine.defaulted == 0);
    BandNonCol_RefineBeginStep(1, 1, 1, 1);
    CHECK(!BandNonCol_refine.last_refined);
    /* the run boundary: released, and the stage read again by the next input */
    BandNonCol_RefineRelease();
    BandNonCol_RefineResetStage();
    CHECK(live_count() == 0);
    CHECK(BandNonCol_refine.iterations == -1 && BandNonCol_refine.final_stage == 0);
    BandNonCol_RefineBeginStep(1, 1, 0, 0);
    CHECK(configure_calls == 2 && BandNonCol_RefineIterations() == 2 && BandNonCol_refine.last_refined);
    checks += 6;

    /* ---- the two-pass solve: in one go, kept in the slot; no condemnation */
    BandNonCol_RefineBeginStep(2, 0, 1, 0);
    BandNonCol_KRefEnsure(TKNUM, N2);
    script(script_prepare, script_prepare_persistent, &n_prepare, &prepare_calls, ok1, zero1, 1);
    script(script_finish, script_finish_persistent, &n_finish, &finish_calls, ok1, zero1, 1);
    BandNonCol_DenseGpu_EnsureHipblas();
    CHECK(BandNonCol_RefinedSolveH2(N2, MAXN, 0, ko, &BandNonCol_kref[0], 1, 0.0, 1) == 1);
    CHECK(prepare_calls == 1 && last_transient == 0 && BandNonCol_kref[0].valid);
    for (int l = 1; l <= MAXN; l++) CHECK(ko[l] == refined(l - 1));
    BandNonCol_RefineDropDevice();
    fail_malloc_bytes = sizeof(dcomplex) * N2 * N2;
    CHECK(BandNonCol_RefinedSolveH2(N2, MAXN, 1, ko, &BandNonCol_kref[1], 1, 0.0, 1) == 0);
    CHECK(BandNonCol_refine.persistent == 0 && live_count() == 2);
    fail_malloc_bytes = 0;
    BandNonCol_RefineDropDevice();
    BandNonCol_RefineEndStep();
    CHECK(!BandNonCol_refine.final_stage && Band_DFT_NonCol_RefineLastRefined() == 1);
    checks += 6;

    /* ---- the end of the cycle: the slots and device buffers go */
    Band_DFT_NonCol_RefineEndCycle();
    CHECK(BandNonCol_kref == NULL && BandNonCol_kref_used == 0 && BandNonCol_refine_x == NULL);
    /* AfterMixing: below until the rest of the cycle is FP64 */
    Band_DFT_NonCol_RefineAfterMixing(1.0e-8);
    CHECK(!BandNonCol_refine.final_stage);
    Band_DFT_NonCol_RefineAfterMixing(1.0e-9);
    CHECK(BandNonCol_refine.final_stage);
    BandNonCol_RefineRelease();
    CHECK(live_count() == 0);
    free(BandNonCol_dense_gpu_workspace.d_H2);
    checks += 5;

    printf("PASS: %lu checks; resident two halves (transient first step, warm from the slot, cold redo, "
           "FP64 fallbacks, condemnation), two-pass solve, gates, run-boundary reset, configuration, "
           "memory test, end of cycle, no device buffer left\n", checks);
    return 0;
}
'''


def make_harness():
    nc = (ROOT / "source/Band_DFT_NonCol.c").read_text()
    header = (ROOT / "source/eigen_refine_gpu.h").read_text()
    types = block(header, "typedef struct {\n    hipblasHandle_t hipblas;", "} EigenRefineReport;")
    statics = block(nc, "typedef struct\n{\n    int     iterations;",
                    "static int                     BandNonCol_refine_e0_n = 0;")
    names = ["BandNonCol_TryDeviceMalloc", "BandNonCol_MallocArray", "BandNonCol_RefineFreeDevice",
             "BandNonCol_RefineDropDevice", "BandNonCol_KRefReset", "BandNonCol_RefineRelease",
             "BandNonCol_RefineResetStage", "BandNonCol_RefineConfigure", "BandNonCol_RefineIterations",
             "BandNonCol_RefineScratchBytes", "BandNonCol_RefineFits", "BandNonCol_RefineBeginStep",
             "BandNonCol_RefineEndStep", "Band_DFT_NonCol_RefineLastRefined", "Band_DFT_NonCol_RefineForceFP64",
             "Band_DFT_NonCol_RefineEndCycle", "Band_DFT_NonCol_RefineAfterMixing", "BandNonCol_RefineFermi",
             "BandNonCol_RefineChemP", "BandNonCol_KRefEnsure", "BandNonCol_RefineBasisLimit",
             "BandNonCol_KRefSlotAllocate", "BandNonCol_KRefSlotFree", "BandNonCol_RefineEnsureBuffers",
             "BandNonCol_RefinePrepareH2", "BandNonCol_RefineFinishH2", "BandNonCol_RefinedSolveH2"]
    halves = ["BandNonCol_RootDenseRefinedFirstHalf", "BandNonCol_RootDenseRefinedSecondHalf"]
    return "\n\n".join([PREFIX, types, statics, MOCK, *(function(nc, name) for name in names),
                        *(function(nc, name) for name in halves), HARNESS])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cc", default=os.environ.get("CC", "cc"))
    parser.add_argument("--sanitize", action="store_true")
    args = parser.parse_args()
    flags = shlex.split(os.environ.get("CFLAGS", "-O2"))
    if args.sanitize:
        flags += ["-g", "-fsanitize=address,undefined", "-fno-omit-frame-pointer"]
    with tempfile.TemporaryDirectory(prefix="openmx-band-refine-") as directory:
        source = Path(directory) / "refine.c"
        binary = Path(directory) / "refine"
        source.write_text(make_harness())
        subprocess.run([*shlex.split(args.cc), *flags, "-std=c11", "-D_DEFAULT_SOURCE", str(source),
                        "-I", str(ROOT / "source"), str(ROOT / "source/openmx_precision.c"),
                        "-o", str(binary), "-lm"], check=True)
        subprocess.run([str(binary)], check=True, stdout=subprocess.DEVNULL if os.environ.get("QUIET") else None)


if __name__ == "__main__":
    main()

/**********************************************************************
  Set_Hamiltonian.c:

     Set_Hamiltonian.c is a subroutine to make Hamiltonian matrix
     within LDA or GGA.

  Log of Set_Hamiltonian.c:

     24/April/2002  Released by T. Ozaki
     17/April/2013  Modified by A.M. Ito

***********************************************************************/

#include "mpi.h"
#include "openmx_common.h"
#include "openmx_gpu_task.h"
#include "openmx_gpu_workers.h"
#include "openmx_host_memory.h"
#include "lapack_prototypes.h"
#include "set_hip_default_device_from_local_rank.h"
#include <limits.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <time.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>


#define measure_time 0

/* Runtime phase profiling gated by OPENMX_BAND_PROFILE=1, in the NVIDIA
   version's format: one SETHPROF line per rank per call, and SETHOTFPROF /
   SETHOTFDENPROF lines of the on-the-fly paths, on stderr (no overhead when
   the variable is unset). */
typedef struct
{
    double memok;
    double base;
    double vpot;
    double matel;
    double barrier;
} SetHamiltonianProfileCounters;

static SetHamiltonianProfileCounters SetH_prof = {0};

static int SetH_ProfileEnabled(void)
{
    static int enabled = -1;

    if (enabled < 0) {
        const char *value = getenv("OPENMX_BAND_PROFILE");
        enabled = (value != NULL && atoi(value) != 0);
    }
    return enabled;
}

#define SETH_PROF_T0(t)                                                                                                \
    do {                                                                                                               \
        if (SetH_ProfileEnabled())                                                                                     \
            dtime(&(t));                                                                                               \
    } while (0)

#define SETH_PROF_ADD(field, t)                                                                                        \
    do {                                                                                                               \
        if (SetH_ProfileEnabled()) {                                                                                   \
            double _prof_t1;                                                                                           \
            dtime(&_prof_t1);                                                                                          \
            SetH_prof.field += _prof_t1 - (t);                                                                         \
        }                                                                                                              \
    } while (0)

enum {
    SETH_SIMD_BASELINE = 0,
    SETH_SIMD_AVX2 = 1,
    SETH_SIMD_AVX512 = 2
};

#if defined(__x86_64__) && !defined(__AMDGCN__) && \
    (defined(__clang__) || defined(__GNUC__))
#define SETH_X86_RUNTIME_DISPATCH 1
#define SETH_NOINLINE __attribute__((noinline))
#define SETH_ALWAYS_INLINE __attribute__((always_inline)) inline
#define SETH_TARGET_AVX2 \
    __attribute__((target("avx2,fma"),noinline))
#define SETH_TARGET_AVX512 \
    __attribute__((target("avx512f,fma"),noinline))
#else
#define SETH_X86_RUNTIME_DISPATCH 0
#define SETH_NOINLINE
#define SETH_ALWAYS_INLINE inline
#define SETH_TARGET_AVX2
#define SETH_TARGET_AVX512
#endif

static int Set_Hamiltonian_Simd_Mode(void)
{
    const char *value = getenv("OPENMX_SETHAM_SIMD");
    int allow_avx512 = 1;

    if (value != NULL &&
        (strcmp(value,"0") == 0 || strcasecmp(value,"scalar") == 0)) {
        return SETH_SIMD_BASELINE;
    }
    if (value != NULL && strcasecmp(value,"avx2") == 0) {
        allow_avx512 = 0;
    }

#if SETH_X86_RUNTIME_DISPATCH
    /* These builtins require both CPU support and enabled OS XSAVE/XCR0
       state, so an AVX-512 CPU with disabled ZMM state safely falls back. */
    __builtin_cpu_init();
    if (allow_avx512 && __builtin_cpu_supports("avx512f") &&
        __builtin_cpu_supports("fma")) {
        return SETH_SIMD_AVX512;
    }
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
        return SETH_SIMD_AVX2;
    }
#endif

    return SETH_SIMD_BASELINE;
}

static const char *Set_Hamiltonian_Simd_Mode_Name(int mode)
{
    if (mode == SETH_SIMD_AVX512) return "AVX-512";
    if (mode == SETH_SIMD_AVX2) return "AVX2";
    return "baseline";
}

void Calc_MatrixElements_dVH_Vxc_VNA(int Cnt_kind);
static void Calc_MatrixElements_dVH_Vxc_VNA_CPU(int Cnt_kind);
static void Calc_MatrixElements_dVH_Vxc_VNA_OpenMP(int Cnt_kind);
static void Set_Hamiltonian_Base_OpenMP(int SCF_iter, double *****H0, double *****HNL, double *****H);
static size_t Set_Hamiltonian_Base_OpenMP_DeviceBytes(int SCF_iter, int myid);
static size_t Set_Hamiltonian_MatrixElements_OpenMP_DeviceBytes(int Cnt_kind, int myid);
int Set_Hamiltonian_Hip_MatrixElements(
    int pair_count, int spin_count, int max_output_count, int max_no, int max_nolg,
    size_t total_h, size_t total_nolg, size_t total_orbs0, size_t total_orbs1,
    const int *pair_NO0, const int *pair_NO1, const int *pair_NOLG,
    const size_t *pair_h_offset, const size_t *pair_nolg_offset,
    const size_t *pair_orbs0_offset, const size_t *pair_orbs1_offset, const int *nolg_Nc,
    const double *vpotbuf, const float *orbs0buf, const float *orbs1buf, double *hbuf);
int Set_Hamiltonian_Hip_MatrixElements_Resident(
    int pair_count, int spin_count, int max_output_count, int max_no, int max_nolg,
    size_t total_h, size_t total_nolg,
    const int *d_pair_NO0, const int *d_pair_NO1, const int *d_pair_NOLG,
    const size_t *d_pair_h_offset, const size_t *d_pair_nolg_offset,
    const size_t *d_pair_orbs0_offset, const size_t *d_pair_orbs1_offset, const int *d_nolg_Nc,
    const float *d_orbs0buf, const float *d_orbs1buf,
    const double *vpotbuf, double *hbuf);

static int Set_Hamiltonian_OpenMP_Rank_Selected = 1;
static int Set_Hamiltonian_MatrixElements_GPU_Used = 0;
static size_t Set_Hamiltonian_ME_ResidentBytes(void);
void Set_Hamiltonian_Invalidate_OpenMP_MatrixElements_Cache(void);

#include "orbs_otf_hip.h"
int Set_Orbitals_Grid_OTF_Prepare(void);
void Set_Orbitals_Grid_OTF_Release(void);
void Set_Hamiltonian_OTF_Release(void);
static void Set_Hamiltonian_OTF_ReleaseDevice(void);
static size_t Set_Hamiltonian_OTF_DeviceBytes(void);

/* OPENMX_OTF_CHECK=1: the first table build of a geometry also evaluates
   every table entry with the device orbital evaluator (orbs_otf_hip.cu)
   and reports the largest difference on each rank */
static void Set_Hamiltonian_OTF_Check(int pair_count, const int *pair_Mc_AN, const int *pair_h_AN,
                                      const int *pair_NO0, const int *pair_NO1, const int *pair_NOLG,
                                      const size_t *pair_nolg_offset, const size_t *pair_orbs1_offset,
                                      const int *nolg_Nc, size_t total_nolg,
                                      const Type_Orbs_Grid *orbs0buf, size_t total_orbs0,
                                      const Type_Orbs_Grid *orbs1buf, size_t total_orbs1, int myid)
{
    OTF_Atom *atoms = NULL;
    int *npts = NULL, *gla = NULL, *cla = NULL;
    size_t *idx_off = NULL, *out_off = NULL, *sph_off = NULL, *atom_sph = NULL, *atom_o0 = NULL;
    size_t sph_total = 0, o0 = 0;
    float *h0 = NULL, *h1 = NULL;
    void *d_atoms = NULL, *d_npts = NULL, *d_idx_off = NULL, *d_out_off = NULL, *d_sph_off = NULL;
    void *d_gla = NULL, *d_cla = NULL, *d_nc = NULL, *d_o0 = NULL, *d_o1 = NULL;
    int Mc_AN, pair, max_atom = 1, max_pair = 1, ok = 0;
    double max0 = 0.0, max1 = 0.0, ref0 = 0.0, ref1 = 0.0;
    size_t diff0 = 0, diff1 = 0, k;

    if (!Set_Orbitals_Grid_OTF_Prepare()) {
        printf("<OTF check> rank %d: no device orbital tables\n", myid);
        fflush(stdout);
        return;
    }
    atom_sph = (size_t *)malloc(sizeof(size_t) * (size_t)(Matomnum + 2));
    atom_o0 = (size_t *)malloc(sizeof(size_t) * (size_t)(Matomnum + 2));
    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
        const int Gc_AN = M2G[Mc_AN];
        atom_sph[Mc_AN] = sph_total;
        atom_o0[Mc_AN] = o0;
        sph_total += (size_t)GridN_Atom[Gc_AN];
        o0 += (size_t)GridN_Atom[Gc_AN] * (size_t)Spe_Total_NO[WhatSpecies[Gc_AN]];
        if (max_atom < GridN_Atom[Gc_AN]) max_atom = GridN_Atom[Gc_AN];
    }
    gla = (int *)malloc(sizeof(int) * (sph_total ? sph_total : 1));
    cla = (int *)malloc(sizeof(int) * (sph_total ? sph_total : 1));
    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
        const int Gc_AN = M2G[Mc_AN];
        int Nc;
        for (Nc = 0; Nc < GridN_Atom[Gc_AN]; Nc++) {
            gla[atom_sph[Mc_AN] + (size_t)Nc] = GridListAtom[Mc_AN][Nc];
            cla[atom_sph[Mc_AN] + (size_t)Nc] = CellListAtom[Mc_AN][Nc];
        }
    }
    {
        const int items = (Matomnum > pair_count) ? Matomnum : pair_count;
        atoms = (OTF_Atom *)calloc((size_t)(items ? items : 1), sizeof(OTF_Atom));
        npts = (int *)malloc(sizeof(int) * (size_t)(items ? items : 1));
        idx_off = (size_t *)malloc(sizeof(size_t) * (size_t)(items ? items : 1));
        out_off = (size_t *)malloc(sizeof(size_t) * (size_t)(items ? items : 1));
        sph_off = (size_t *)malloc(sizeof(size_t) * (size_t)(items ? items : 1));
    }
    h0 = (float *)malloc(sizeof(float) * (total_orbs0 ? total_orbs0 : 1));
    h1 = (float *)malloc(sizeof(float) * (total_orbs1 ? total_orbs1 : 1));

#define OTF_CK(call) do { if ((call) != hipSuccess) goto cleanup; } while (0)
    OTF_CK(hipMalloc(&d_atoms, sizeof(OTF_Atom) * (size_t)((Matomnum > pair_count) ? Matomnum : pair_count) + 64));
    OTF_CK(hipMalloc(&d_npts, sizeof(int) * (size_t)((Matomnum > pair_count) ? Matomnum : pair_count) + 64));
    OTF_CK(hipMalloc(&d_idx_off, sizeof(size_t) * (size_t)((Matomnum > pair_count) ? Matomnum : pair_count) + 64));
    OTF_CK(hipMalloc(&d_out_off, sizeof(size_t) * (size_t)((Matomnum > pair_count) ? Matomnum : pair_count) + 64));
    OTF_CK(hipMalloc(&d_sph_off, sizeof(size_t) * (size_t)((Matomnum > pair_count) ? Matomnum : pair_count) + 64));
    OTF_CK(hipMalloc(&d_gla, sizeof(int) * (sph_total ? sph_total : 1)));
    OTF_CK(hipMalloc(&d_cla, sizeof(int) * (sph_total ? sph_total : 1)));
    OTF_CK(hipMalloc(&d_nc, sizeof(int) * (total_nolg ? total_nolg : 1)));
    OTF_CK(hipMalloc(&d_o0, sizeof(float) * (total_orbs0 ? total_orbs0 : 1)));
    OTF_CK(hipMalloc(&d_o1, sizeof(float) * (total_orbs1 ? total_orbs1 : 1)));
    OTF_CK(hipMemcpy(d_gla, gla, sizeof(int) * sph_total, hipMemcpyHostToDevice));
    OTF_CK(hipMemcpy(d_cla, cla, sizeof(int) * sph_total, hipMemcpyHostToDevice));
    OTF_CK(hipMemcpy(d_nc, nolg_Nc, sizeof(int) * total_nolg, hipMemcpyHostToDevice));

    /* the central atoms over their spheres */
    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
        const int Gc_AN = M2G[Mc_AN];
        const int Cwan = WhatSpecies[Gc_AN];
        OTF_Atom *a = &atoms[Mc_AN - 1];
        a->wan = Cwan;
        a->no = Spe_Total_NO[Cwan];
        a->gx = Gxyz[Gc_AN][1]; a->gy = Gxyz[Gc_AN][2]; a->gz = Gxyz[Gc_AN][3];
        a->ax = a->ay = a->az = 0.0;
        npts[Mc_AN - 1] = GridN_Atom[Gc_AN];
        idx_off[Mc_AN - 1] = 0;
        out_off[Mc_AN - 1] = atom_o0[Mc_AN];
        sph_off[Mc_AN - 1] = atom_sph[Mc_AN];
    }
    OTF_CK(hipMemcpy(d_atoms, atoms, sizeof(OTF_Atom) * (size_t)Matomnum, hipMemcpyHostToDevice));
    OTF_CK(hipMemcpy(d_npts, npts, sizeof(int) * (size_t)Matomnum, hipMemcpyHostToDevice));
    OTF_CK(hipMemcpy(d_idx_off, idx_off, sizeof(size_t) * (size_t)Matomnum, hipMemcpyHostToDevice));
    OTF_CK(hipMemcpy(d_out_off, out_off, sizeof(size_t) * (size_t)Matomnum, hipMemcpyHostToDevice));
    OTF_CK(hipMemcpy(d_sph_off, sph_off, sizeof(size_t) * (size_t)Matomnum, hipMemcpyHostToDevice));
    if (OTF_Orbitals_Eval(NULL, Matomnum, max_atom, (const OTF_Atom *)d_atoms, (const int *)d_npts,
                          (const size_t *)d_idx_off, (const size_t *)d_out_off, (const size_t *)d_sph_off, NULL,
                          (const int *)d_gla, (const int *)d_cla, (float *)d_o0) != 0) goto cleanup;
    OTF_CK(hipDeviceSynchronize());

    /* the neighbours at the pairs' overlap points */
    for (pair = 0; pair < pair_count; pair++) {
        const int Mc = pair_Mc_AN[pair], h_AN = pair_h_AN[pair];
        const int Gc_AN = M2G[Mc];
        const int Gh_AN = natn[Gc_AN][h_AN];
        const int Rnh = ncn[Gc_AN][h_AN];
        OTF_Atom *a = &atoms[pair];
        a->wan = WhatSpecies[Gh_AN];
        a->no = pair_NO1[pair];
        a->gx = Gxyz[Gh_AN][1]; a->gy = Gxyz[Gh_AN][2]; a->gz = Gxyz[Gh_AN][3];
        a->ax = atv[Rnh][1]; a->ay = atv[Rnh][2]; a->az = atv[Rnh][3];
        npts[pair] = pair_NOLG[pair];
        idx_off[pair] = pair_nolg_offset[pair];
        out_off[pair] = pair_orbs1_offset[pair];
        sph_off[pair] = atom_sph[Mc];
        if (max_pair < pair_NOLG[pair]) max_pair = pair_NOLG[pair];
    }
    OTF_CK(hipMemcpy(d_atoms, atoms, sizeof(OTF_Atom) * (size_t)pair_count, hipMemcpyHostToDevice));
    OTF_CK(hipMemcpy(d_npts, npts, sizeof(int) * (size_t)pair_count, hipMemcpyHostToDevice));
    OTF_CK(hipMemcpy(d_idx_off, idx_off, sizeof(size_t) * (size_t)pair_count, hipMemcpyHostToDevice));
    OTF_CK(hipMemcpy(d_out_off, out_off, sizeof(size_t) * (size_t)pair_count, hipMemcpyHostToDevice));
    OTF_CK(hipMemcpy(d_sph_off, sph_off, sizeof(size_t) * (size_t)pair_count, hipMemcpyHostToDevice));
    if (OTF_Orbitals_Eval(NULL, pair_count, max_pair, (const OTF_Atom *)d_atoms, (const int *)d_npts,
                          (const size_t *)d_idx_off, (const size_t *)d_out_off, (const size_t *)d_sph_off,
                          (const int *)d_nc, (const int *)d_gla, (const int *)d_cla, (float *)d_o1) != 0) goto cleanup;
    OTF_CK(hipDeviceSynchronize());
    OTF_CK(hipMemcpy(h0, d_o0, sizeof(float) * total_orbs0, hipMemcpyDeviceToHost));
    OTF_CK(hipMemcpy(h1, d_o1, sizeof(float) * total_orbs1, hipMemcpyDeviceToHost));
#undef OTF_CK
    for (k = 0; k < total_orbs0; k++) {
        const double d = fabs((double)h0[k] - (double)orbs0buf[k]);
        if (d > 0.0) diff0++;
        if (max0 < d) max0 = d;
        if (ref0 < fabs((double)orbs0buf[k])) ref0 = fabs((double)orbs0buf[k]);
    }
    for (k = 0; k < total_orbs1; k++) {
        const double d = fabs((double)h1[k] - (double)orbs1buf[k]);
        if (d > 0.0) diff1++;
        if (max1 < d) max1 = d;
        if (ref1 < fabs((double)orbs1buf[k])) ref1 = fabs((double)orbs1buf[k]);
    }
    ok = 1;
    printf("<OTF check> rank %d (%s): central atoms %zu values, %zu differ, max |diff| %.3e (max |value| %.3e); "
           "neighbours %zu values, %zu differ, max |diff| %.3e (max |value| %.3e)\n",
           myid, OTF_Orbitals_DoubleFloat() ? "double-float" : "FP64", total_orbs0, diff0, max0, ref0,
           total_orbs1, diff1, max1, ref1);
    fflush(stdout);

cleanup:
    if (!ok) {
        printf("<OTF check> rank %d: device evaluation failed\n", myid);
        fflush(stdout);
    }
    hipFree(d_atoms); hipFree(d_npts); hipFree(d_idx_off); hipFree(d_out_off); hipFree(d_sph_off);
    hipFree(d_gla); hipFree(d_cla); hipFree(d_nc); hipFree(d_o0); hipFree(d_o1);
    (void)hipGetLastError();
    free(atoms); free(npts); free(idx_off); free(out_off); free(sph_off); free(atom_sph); free(atom_o0);
    free(gla); free(cla); free(h0); free(h1);
}

typedef struct {
    int ready, cnt_kind;
    SetHamiltonianMETables t;
    int *pair_Mc_AN, *pair_h_AN, *pair_NO0, *pair_NO1, *pair_NOLG;
    int *nolg_MN, *nolg_Nc;
    size_t *pair_h_offset, *pair_nolg_offset, *pair_orbs0_offset, *pair_orbs1_offset;
    Type_Orbs_Grid *orbs0buf, *orbs1buf;
} SetHamiltonianMETableCache;

static SetHamiltonianMETableCache Set_Hamiltonian_ME_Tables = {0};

static int Set_Hamiltonian_ME_Associate(void *host, size_t bytes)
{
    void *dev = NULL;
    int ompdev = omp_get_default_device();
    if (bytes == 0) return 1;
    if (hipMalloc(&dev, bytes) != hipSuccess) return 0;
    if (hipMemcpy(dev, host, bytes, hipMemcpyHostToDevice) != hipSuccess ||
        omp_target_associate_ptr(host, dev, bytes, 0, ompdev) != 0) {
        hipFree(dev);
        return 0;
    }
    return 1;
}

static void Set_Hamiltonian_ME_Disassociate(void *host)
{
    int ompdev = omp_get_default_device();
    void *dev;
    if (host == NULL) return;
    dev = omp_get_mapped_ptr(host, ompdev);
    if (dev != NULL) {
        omp_target_disassociate_ptr(host, ompdev);
        hipFree(dev);
    }
}

int Band_DFT_NonCol_GpuSwitchNum(void);

/* mirrors the noncollinear band path's own GPU/CPU decision so the demote
   choice below tracks the solver that actually runs */
static int Set_Hamiltonian_NonColGpuEigensolverActive(void)
{
    int basis_count = 0;

    if (scf_eigen_lib_flag != GPUSOLVER || Solver != 3 || SpinP_switch != 3) {
        return 0;
    }

    for (int GA_AN = 1; GA_AN <= atomnum; GA_AN++) {
        basis_count += Spe_Total_CNO[WhatSpecies[GA_AN]];
    }
    return Band_DFT_NonCol_GpuSwitchNum() <= 2*basis_count;
}

/* set when a dense eigensolver found the device too full while the
   orbital tables were resident (Set_Hamiltonian_DemoteResidency): from then
   on the tables are uploaded per call and freed after it, so that the
   eigensolver keeps the room (OPENMX_SETHAM_DEMOTE_ON_PRESSURE=0 keeps
   them resident) */
static int Set_Hamiltonian_ME_residency_demoted = 0;

static int Set_Hamiltonian_ME_KeepDeviceResident(void)
{
    if (Set_Hamiltonian_ME_residency_demoted) return 0;
    if (Set_Hamiltonian_NonColGpuEigensolverActive()) {
        const char *demote = getenv("OPENMX_NC_EIGEN_DEMOTE_SETHAM");
        if (demote == NULL || atoi(demote) != 0) return 0;
    }
    return 1;
}

/* every rank of a dense eigensolver whose memory test failed: the resident
   orbital tables go now and stay off the device for the rest of the run;
   rank-local, returns the device bytes released on this rank */
size_t Set_Hamiltonian_DemoteResidency(void)
{
    static int checked = 0, enabled = 1;
    size_t released;

    if (!checked) {
        const char *value = getenv("OPENMX_SETHAM_DEMOTE_ON_PRESSURE");
        enabled = (value == NULL || value[0] == '\0' || atoi(value) != 0);
        checked = 1;
    }
    if (!enabled) return 0;
    released = Set_Hamiltonian_ME_ResidentBytes();
    Set_Hamiltonian_ME_residency_demoted = 1;
    if (0 < released) Set_Hamiltonian_Invalidate_OpenMP_MatrixElements_Cache();
    /* the on-the-fly state too (the sphere indices of the local atoms and
       the PAO tables): idle during the eigensolver, rebuilt lazily by the
       next Set_Hamiltonian or Set_Density_Grid; the batch plans are host
       memory of this geometry and stay */
    released += Set_Hamiltonian_OTF_DeviceBytes();
    Set_Hamiltonian_OTF_ReleaseDevice();
    Set_Orbitals_Grid_OTF_Release();
    return released;
}

void Set_Hamiltonian_Invalidate_OpenMP_MatrixElements_Cache(void)
{
    SetHamiltonianMETableCache *c = &Set_Hamiltonian_ME_Tables;
    Set_Hamiltonian_ME_Disassociate(c->pair_NO0);
    Set_Hamiltonian_ME_Disassociate(c->pair_NO1);
    Set_Hamiltonian_ME_Disassociate(c->pair_NOLG);
    Set_Hamiltonian_ME_Disassociate(c->nolg_MN);
    Set_Hamiltonian_ME_Disassociate(c->nolg_Nc);
    Set_Hamiltonian_ME_Disassociate(c->pair_h_offset);
    Set_Hamiltonian_ME_Disassociate(c->pair_nolg_offset);
    Set_Hamiltonian_ME_Disassociate(c->pair_orbs0_offset);
    Set_Hamiltonian_ME_Disassociate(c->pair_orbs1_offset);
    Set_Hamiltonian_ME_Disassociate(c->orbs0buf);
    Set_Hamiltonian_ME_Disassociate(c->orbs1buf);
    free(c->pair_Mc_AN); free(c->pair_h_AN); free(c->pair_NO0); free(c->pair_NO1);
    free(c->pair_NOLG); free(c->nolg_MN); free(c->nolg_Nc);
    free(c->pair_h_offset); free(c->pair_nolg_offset);
    free(c->pair_orbs0_offset); free(c->pair_orbs1_offset);
    free(c->orbs0buf); free(c->orbs1buf);
    memset(c, 0, sizeof(*c));
}

int Set_Hamiltonian_MatrixElementsTables_Ready(int Cnt_kind)
{ return Set_Hamiltonian_ME_Tables.ready && Set_Hamiltonian_ME_Tables.cnt_kind==Cnt_kind; }

int Set_Hamiltonian_GetMatrixElementsTables(int Cnt_kind, SetHamiltonianMETables *tables)
{
    if (tables==NULL) return 0;
    memset(tables,0,sizeof(*tables));
    if (!Set_Hamiltonian_MatrixElementsTables_Ready(Cnt_kind)) return 0;
    *tables = Set_Hamiltonian_ME_Tables.t;
    return 1;
}

void Set_Hamiltonian_Set_OpenMP_Rank_Selected(int selected)
{
    Set_Hamiltonian_OpenMP_Rank_Selected = selected ? 1 : 0;
}

int Set_Hamiltonian_OpenMP_Rank_Is_Selected(void)
{
    return Set_Hamiltonian_OpenMP_Rank_Selected;
}

enum {
    SET_HAMILTONIAN_PACK_ORDER_COL = 0,
    SET_HAMILTONIAN_PACK_ORDER_NONCOL = 1
};

typedef struct {
    int ready;
    int owns;
    int use_contracted;
    int order_mode;
    int size;
    int spin_count;
    int atom_count;
    int *order_GA;
    double *overlap;
    double *h[4];
    double *imnl[3];
} SetHamiltonianGpuSolverPackedCache;

static SetHamiltonianGpuSolverPackedCache Set_Hamiltonian_GpuSolver_Cache = {0};

static void Set_Hamiltonian_abort(const char *where, const char *message, int myid)
{
    if (myid == Host_ID) {
        fprintf(stderr, "Set_Hamiltonian: %s: %s\n", where, message);
        fflush(stderr);
    }
    MPI_Abort(mpi_comm_level1, 1);
    exit(1);
}

static size_t Set_Hamiltonian_checked_add(size_t a, size_t b, const char *label, int myid)
{
    if (b > ((size_t)-1) - a) {
        char message[256];
        snprintf(message, sizeof(message), "size overflow while estimating %s", label);
        Set_Hamiltonian_abort("OpenMP memory check", message, myid);
    }

    return a + b;
}

static size_t Set_Hamiltonian_checked_mul(size_t a, size_t b, const char *label, int myid)
{
    if (a != 0 && b > ((size_t)-1) / a) {
        char message[256];
        snprintf(message, sizeof(message), "size overflow while estimating %s", label);
        Set_Hamiltonian_abort("OpenMP memory check", message, myid);
    }

    return a * b;
}

static size_t Set_Hamiltonian_array_bytes(size_t count, size_t elem_size, const char *label, int myid)
{
    return Set_Hamiltonian_checked_mul(count, elem_size, label, myid);
}

static void Set_Hamiltonian_add_array_bytes(size_t *total, size_t count, size_t elem_size, const char *label, int myid)
{
    size_t bytes = Set_Hamiltonian_array_bytes(count, elem_size, label, myid);
    *total = Set_Hamiltonian_checked_add(*total, bytes, label, myid);
}

static int Set_Hamiltonian_OpenMP_Enabled(void)
{
    const char *gpu=getenv("OPENMX_GPU"),*value=getenv("OPENMX_SETHAM_BASE_GPU");
    return scf_eigen_lib_flag==GPUSOLVER && !(gpu && atoi(gpu)==0) &&
           (value ? atoi(value)!=0 : !openmx_gpu_is_apu());
}

static int Set_Hamiltonian_MatrixElements_OpenMP_Enabled(void)
{
    /*
     * The matrix-elements HIP path keeps its packed orbital tables resident
     * on the device across the SCF loop and re-uploads only the potential,
     * so it is enabled by default on every GPU; the device-memory preflight
     * still sends it to the host path when the tables do not fit.
    */
    const char *value = getenv("OPENMX_SETHAM_GPU");
    /* On by default, APUs included (OPENMX_SETHAM_GPU=0 opts out) -- except
       when the GPU is shared by more ranks than the dense band eigensolvers
       accept (openmx_band_gpu_dense_oversubscribed): the many small
       matrix-element kernels of all ranks then serialize on the device and
       the resident tables feed a solver path that is demoted anyway.  On the
       40-atom noncollinear GGFF at 24 ranks on one MI300A this path took
       0.51 s/SCF against 0.27 s/SCF for the AVX-512 host path.
       OPENMX_SETHAM_GPU=1 still forces it on. */
    int default_on = !openmx_band_gpu_dense_oversubscribed();
    int requested = (value == NULL || value[0] == '\0') ? default_on : (atoi(value) != 0);

    return scf_eigen_lib_flag == GPUSOLVER && requested &&
           gpu_rank_device_usable();
}

static int Set_Hamiltonian_DeviceMemoryOK(size_t required_bytes, const char *where, int myid, int use_device)
{
    MPI_Comm node_comm, device_comm;
    int local_rank, local_size, hip_device_count, device_rank, device_ranks;
    int hip_device;
    size_t free_bytes, total_bytes;
    hipError_t hip_err;
    unsigned long long local_required, group_required, local_free, group_free;
    int hip_ok;

    MPI_Comm_split_type(mpi_comm_level1, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node_comm);
    MPI_Comm_rank(node_comm, &local_rank);
    MPI_Comm_size(node_comm, &local_size);

    hip_device = -1;
    free_bytes = 0;
    total_bytes = 0;
    hip_ok = 0;

    if (use_device) {
        hip_err = hipGetDeviceCount(&hip_device_count);
        if (hip_err != hipSuccess || hip_device_count <= 0) {
            if (myid == Host_ID || hip_err != hipSuccess) {
                fprintf(stderr,
                        "Set_Hamiltonian: rank %d %s: failed to get HIP device count (%s); switching to CPU path.\n",
                        myid, where, hip_err == hipSuccess ? "no HIP device" : hipGetErrorString(hip_err));
                fflush(stderr);
            }
        }
        else {
            int omp_device_count = omp_get_num_devices();
            if (0 < omp_device_count && omp_device_count < hip_device_count) {
                hip_device_count = omp_device_count;
            }
            if (0 < SCF_Gpu_Num && SCF_Gpu_Num < hip_device_count) {
                hip_device_count = SCF_Gpu_Num;
            }
            hip_device = openmx_gpu_map_rank_to_device(local_rank, local_size,
                                                       hip_device_count);

            hip_err = hipSetDevice(hip_device);
            if (hip_err != hipSuccess) {
                fprintf(stderr,
                        "Set_Hamiltonian: rank %d %s: failed to set HIP device %d (%s); switching to CPU path.\n",
                        myid, where, hip_device, hipGetErrorString(hip_err));
                fflush(stderr);
            }
            else {
                hip_err = hipMemGetInfo(&free_bytes, &total_bytes);
                if (hip_err != hipSuccess) {
                    fprintf(stderr,
                            "Set_Hamiltonian: rank %d %s: failed to query HIP memory on device %d (%s); "
                            "switching to CPU path.\n",
                            myid, where, hip_device, hipGetErrorString(hip_err));
                    fflush(stderr);
                }
                else {
                    hip_ok = 1;
                }
            }
        }
    }

    MPI_Comm_split(node_comm, hip_ok ? hip_device : MPI_UNDEFINED, 0, &device_comm);
    MPI_Comm_free(&node_comm);

    if (!use_device) {
        return 1;
    }

    if (!hip_ok) {
        return 0;
    }

    if (required_bytes > (size_t)ULLONG_MAX || free_bytes > (size_t)ULLONG_MAX) {
        Set_Hamiltonian_abort("OpenMP memory check", "memory size does not fit unsigned long long", myid);
    }

    local_required = (unsigned long long)required_bytes;
    local_free = (unsigned long long)free_bytes;
    MPI_Allreduce(&local_required, &group_required, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, device_comm);
    MPI_Allreduce(&local_free, &group_free, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN, device_comm);
    MPI_Comm_rank(device_comm, &device_rank);
    MPI_Comm_size(device_comm, &device_ranks);
    MPI_Comm_free(&device_comm);

    if (group_free < group_required) {
        if (device_rank == 0) {
            if (2 <= level_stdout) {
                fprintf(stderr,
                        "Set_Hamiltonian: %s: GPU device %d is shared by %d rank(s), has %.3f MiB free "
                        "(%.3f MiB total on rank %d), but %.3f MiB is required in total; switching to CPU path.\n",
                        where, hip_device, device_ranks, (double)group_free / (1024.0 * 1024.0),
                        (double)total_bytes / (1024.0 * 1024.0), myid,
                        (double)group_required / (1024.0 * 1024.0));
                fflush(stderr);
            }
        }
        return 0;
    }

    return 1;
}

/* Device bytes held by the resident copies of the matrix-elements tables. */
static size_t Set_Hamiltonian_ME_ResidentBytes(void)
{
    const SetHamiltonianMETables *t = &Set_Hamiltonian_ME_Tables.t;
    size_t bytes = 0;

    if (!Set_Hamiltonian_ME_Tables.ready) return 0;
    if (t->meta_resident) bytes += (size_t)t->pair_count * (3 * sizeof(int) + 4 * sizeof(size_t));
    if (t->nolg_resident) bytes += 2 * sizeof(int) * t->total_nolg;
    if (t->orbs0_resident) bytes += sizeof(Type_Orbs_Grid) * t->total_orbs0;
    if (t->orbs1_resident) bytes += sizeof(Type_Orbs_Grid) * t->total_orbs1;
    return bytes;
}

/* The tables of the previous OpenMP call serve the next one unchanged when
   nothing invalidated them since -- every change of the geometry, of
   Orbs_Grid or of Cnt_kind goes through
   Set_Hamiltonian_Invalidate_OpenMP_MatrixElements_Cache -- and all of them
   still have their device copies; only the potential and the output then
   need to travel. */
static int Set_Hamiltonian_ME_CacheReusable(int Cnt_kind)
{
    const SetHamiltonianMETableCache *c = &Set_Hamiltonian_ME_Tables;
    const void *host[9];
    int ompdev, k;

    if (!c->ready || c->cnt_kind != Cnt_kind || c->t.pair_count <= 0) return 0;
    if (!(c->t.meta_resident && c->t.nolg_resident && c->t.orbs0_resident && c->t.orbs1_resident)) return 0;

    ompdev = omp_get_default_device();
    host[0] = c->pair_NO0;         host[1] = c->pair_NO1;          host[2] = c->pair_NOLG;
    host[3] = c->pair_h_offset;    host[4] = c->pair_nolg_offset;
    host[5] = c->pair_orbs0_offset; host[6] = c->pair_orbs1_offset;
    host[7] = c->orbs0buf;         host[8] = c->orbs1buf;
    for (k = 0; k < 9; k++) {
        if (host[k] == NULL || omp_get_mapped_ptr((void *)host[k], ompdev) == NULL) return 0;
    }
    return 1;
}

static int Set_Hamiltonian_Base_Use_OpenMP(int SCF_iter, int myid)
{
    size_t required_bytes;
    int memory_ok;

    if (!Set_Hamiltonian_OpenMP_Enabled()) {
        return 0;
    }

    required_bytes = Set_Hamiltonian_OpenMP_Rank_Selected ?
        Set_Hamiltonian_Base_OpenMP_DeviceBytes(SCF_iter, myid) : 0;
    memory_ok = Set_Hamiltonian_DeviceMemoryOK(required_bytes, "base OpenMP path", myid,
                                               Set_Hamiltonian_OpenMP_Rank_Selected);

    return Set_Hamiltonian_OpenMP_Rank_Selected && memory_ok;
}

static int Set_Hamiltonian_MatrixElements_Use_OpenMP(int Cnt_kind, int myid)
{
    size_t required_bytes;
    int local_enabled, any_enabled, local_use, memory_ok;

    local_enabled = Set_Hamiltonian_MatrixElements_OpenMP_Enabled();
    MPI_Allreduce(&local_enabled, &any_enabled, 1, MPI_INT, MPI_MAX,
                  mpi_comm_level1);
    if (!any_enabled) {
        return 0;
    }

    /* Every rank must enter Set_Hamiltonian_DeviceMemoryOK because it creates
       node/device communicators.  A failed local probe or a rank-specific
       override may still select the host implementation for that rank. */
    local_use = local_enabled && Set_Hamiltonian_OpenMP_Rank_Selected;
    required_bytes = local_use ?
        Set_Hamiltonian_MatrixElements_OpenMP_DeviceBytes(Cnt_kind, myid) : 0;
    memory_ok = Set_Hamiltonian_DeviceMemoryOK(required_bytes, "matrix-elements OpenMP path", myid,
                                               local_use);

    /* Tables left resident by an earlier call cannot serve the CPU path and
       would only pin memory the other stages need; the next call that fits
       rebuilds them. */
    if (local_use && !memory_ok) {
        size_t resident = Set_Hamiltonian_ME_ResidentBytes();

        if (0 < resident) {
            Set_Hamiltonian_Invalidate_OpenMP_MatrixElements_Cache();
            if (myid == Host_ID) {
                fprintf(stderr,
                        "Set_Hamiltonian: matrix-elements OpenMP path: released the resident orbital tables "
                        "(%.3f MiB on rank %d) that the CPU path cannot use.\n",
                        (double)resident / (1024.0 * 1024.0), myid);
                fflush(stderr);
            }
        }
    }

    return local_use && memory_ok;
}

static void *Set_Hamiltonian_malloc(size_t bytes, const char *name, int myid)
{
    void *p;

    if (bytes == 0) {
        bytes = 1;
    }

    p = malloc(bytes);
    if (p == NULL) {
        char message[256];
        snprintf(message, sizeof(message), "failed to allocate %s", name);
        Set_Hamiltonian_abort("OpenMP", message, myid);
    }

    return p;
}

static void Set_Hamiltonian_GpuSolver_Free_Cache(void)
{
    SetHamiltonianGpuSolverPackedCache *cache = &Set_Hamiltonian_GpuSolver_Cache;
    int i;

    free(cache->order_GA);
    free(cache->overlap);
    for (i = 0; i < 4; i++) {
        free(cache->h[i]);
    }
    for (i = 0; i < 3; i++) {
        free(cache->imnl[i]);
    }

    memset(cache, 0, sizeof(*cache));
}

void Set_Hamiltonian_Invalidate_GpuSolver_HS_Cache(void)
{
    Set_Hamiltonian_GpuSolver_Free_Cache();
}

static int Set_Hamiltonian_GpuSolver_LocalPackedSize(int myid)
{
    size_t total = 0;

    for (int MA_AN = 1; MA_AN <= Matomnum; MA_AN++) {
        int GA_AN = M2G[MA_AN];
        int wanA = WhatSpecies[GA_AN];
        int tnoA = Spe_Total_CNO[wanA];
        size_t neighbor_orbitals = 0;

        for (int LB_AN = 0; LB_AN <= FNAN[GA_AN]; LB_AN++) {
            int GB_AN = natn[GA_AN][LB_AN];
            int wanB = WhatSpecies[GB_AN];
            int tnoB = Spe_Total_CNO[wanB];

            neighbor_orbitals = Set_Hamiltonian_checked_add(neighbor_orbitals, (size_t)tnoB,
                                                            "GPU solver packed neighbor orbitals", myid);
        }

        total = Set_Hamiltonian_checked_add(
            total,
            Set_Hamiltonian_checked_mul((size_t)tnoA, neighbor_orbitals,
                                        "GPU solver packed local matrix segment", myid),
            "GPU solver packed local matrix segment", myid);
    }

    if ((size_t)INT_MAX < total) {
        Set_Hamiltonian_abort("GPU solver packed cache", "local packed matrix segment exceeds INT_MAX", myid);
    }

    return (int)total;
}

static void Set_Hamiltonian_GpuSolver_PackLocalMatrix(double ****mat, double *local, int local_size, int order_mode,
                                                     int myid)
{
    int k = 0;

    if (mat == NULL) {
        Set_Hamiltonian_abort("GPU solver packed cache", "NULL sparse matrix", myid);
    }

    for (int MA_AN = 1; MA_AN <= Matomnum; MA_AN++) {
        int GA_AN = M2G[MA_AN];
        int wanA = WhatSpecies[GA_AN];
        int tnoA = Spe_Total_CNO[wanA];

        if (order_mode == SET_HAMILTONIAN_PACK_ORDER_COL) {
            for (int LB_AN = 0; LB_AN <= FNAN[GA_AN]; LB_AN++) {
                int GB_AN = natn[GA_AN][LB_AN];
                int wanB = WhatSpecies[GB_AN];
                int tnoB = Spe_Total_CNO[wanB];

                for (int i = 0; i < tnoA; i++) {
                    for (int j = 0; j < tnoB; j++) {
                        local[k++] = mat[MA_AN][LB_AN][i][j];
                    }
                }
            }
        }
        else {
            for (int i = 0; i < tnoA; i++) {
                for (int LB_AN = 0; LB_AN <= FNAN[GA_AN]; LB_AN++) {
                    int GB_AN = natn[GA_AN][LB_AN];
                    int wanB = WhatSpecies[GB_AN];
                    int tnoB = Spe_Total_CNO[wanB];

                    for (int j = 0; j < tnoB; j++) {
                        local[k++] = mat[MA_AN][LB_AN][i][j];
                    }
                }
            }
        }
    }

    if (k != local_size) {
        Set_Hamiltonian_abort("GPU solver packed cache", "packed local matrix size mismatch", myid);
    }
}

static void Set_Hamiltonian_GpuSolver_GatherMatrixToSelected(double ****mat, double **target, int order_mode,
                                                             int local_size, int total_size, const int *counts,
                                                            const int *displs, const int *selected_roots,
                                                            MPI_Comm selected_comm, int first_selected_root,
                                                            int myid)
{
    double *local;
    double *recvbuf = NULL;

    *target = NULL;
    local = (double *)Set_Hamiltonian_malloc(sizeof(double) * (size_t)(0 < local_size ? local_size : 1),
                                             "GPU solver packed local matrix", myid);
    Set_Hamiltonian_GpuSolver_PackLocalMatrix(mat, local, local_size, order_mode, myid);

    if (myid == first_selected_root) {
        recvbuf = (double *)Set_Hamiltonian_malloc(sizeof(double) * (size_t)(0 < total_size ? total_size : 1),
                                                   "GPU solver packed matrix cache", myid);
    }

    MPI_Gatherv(local, local_size, MPI_DOUBLE, recvbuf, (int *)counts, (int *)displs, MPI_DOUBLE, first_selected_root,
                mpi_comm_level1);

    if (selected_roots[myid]) {
        if (myid == first_selected_root) {
            *target = recvbuf;
        }
        else {
            *target = (double *)Set_Hamiltonian_malloc(sizeof(double) * (size_t)(0 < total_size ? total_size : 1),
                                                       "GPU solver packed matrix cache", myid);
        }
        MPI_Bcast(*target, total_size, MPI_DOUBLE, 0, selected_comm);
    }

    free(local);
}

void Set_Hamiltonian_GpuSolver_SetMP(int *MP)
{
    int Anum = 1;

    if (MP == NULL) {
        return;
    }

    for (int GA_AN = 1; GA_AN <= atomnum; GA_AN++) {
        int wanA = WhatSpecies[GA_AN];
        MP[GA_AN] = Anum;
        Anum += Spe_Total_CNO[wanA];
    }
}

void Set_Hamiltonian_Build_GpuSolver_HS_Cache(int use_contracted, int refresh_static)
{
    SetHamiltonianGpuSolverPackedCache *cache = &Set_Hamiltonian_GpuSolver_Cache;
    int myid, numprocs;
    int local_selected;
    int selected_count = 0;
    int first_selected_root = -1;
    int local_size, total_size = 0;
    int local_matomnum;
    int *selected_roots = NULL;
    int *counts = NULL;
    int *displs = NULL;
    int *atom_counts = NULL;
    int *atom_displs = NULL;
    int dummy_atom = 0;
    int order_mode;
    int spin_count;
    int local_rebuild_static;
    int rebuild_static;
    MPI_Comm selected_comm = MPI_COMM_NULL;
    double ****overlap_src;
    double *****h_src;
    double *****imnl_src;

    MPI_Comm_size(mpi_comm_level1, &numprocs);
    MPI_Comm_rank(mpi_comm_level1, &myid);

    if (scf_eigen_lib_flag != GPUSOLVER) {
        Set_Hamiltonian_GpuSolver_Free_Cache();
        return;
    }

    order_mode = (SpinP_switch == 3) ? SET_HAMILTONIAN_PACK_ORDER_NONCOL : SET_HAMILTONIAN_PACK_ORDER_COL;
    spin_count = (SpinP_switch == 3) ? 4 : (SpinP_switch + 1);
    local_selected = Set_Hamiltonian_OpenMP_Rank_Selected ? 1 : 0;
    local_size = Set_Hamiltonian_GpuSolver_LocalPackedSize(myid);
    local_matomnum = Matomnum;

    selected_roots = (int *)Set_Hamiltonian_malloc(sizeof(int) * (size_t)numprocs, "GPU solver selected roots", myid);
    counts = (int *)Set_Hamiltonian_malloc(sizeof(int) * (size_t)numprocs, "GPU solver packed counts", myid);
    displs = (int *)Set_Hamiltonian_malloc(sizeof(int) * (size_t)numprocs, "GPU solver packed displacements", myid);
    atom_counts = (int *)Set_Hamiltonian_malloc(sizeof(int) * (size_t)numprocs, "GPU solver atom counts", myid);
    atom_displs = (int *)Set_Hamiltonian_malloc(sizeof(int) * (size_t)numprocs, "GPU solver atom displacements", myid);

    MPI_Allgather(&local_selected, 1, MPI_INT, selected_roots, 1, MPI_INT, mpi_comm_level1);
    MPI_Allgather(&local_size, 1, MPI_INT, counts, 1, MPI_INT, mpi_comm_level1);
    MPI_Allgather(&local_matomnum, 1, MPI_INT, atom_counts, 1, MPI_INT, mpi_comm_level1);

    for (int ID = 0; ID < numprocs; ID++) {
        if (selected_roots[ID]) {
            if (first_selected_root < 0) {
                first_selected_root = ID;
            }
            selected_count++;
        }
        if (counts[ID] < 0 || atom_counts[ID] < 0) {
            Set_Hamiltonian_abort("GPU solver packed cache", "negative gather count", myid);
        }
        if ((size_t)INT_MAX < (size_t)total_size + (size_t)counts[ID]) {
            Set_Hamiltonian_abort("GPU solver packed cache", "packed matrix size exceeds INT_MAX", myid);
        }
        displs[ID] = total_size;
        total_size += counts[ID];
    }

    {
        int atom_total = 0;
        for (int ID = 0; ID < numprocs; ID++) {
            if ((size_t)INT_MAX < (size_t)atom_total + (size_t)atom_counts[ID]) {
                Set_Hamiltonian_abort("GPU solver packed cache", "atom-order size exceeds INT_MAX", myid);
            }
            atom_displs[ID] = atom_total;
            atom_total += atom_counts[ID];
        }
        if (atom_total != atomnum) {
            Set_Hamiltonian_abort("GPU solver packed cache", "atom-order length mismatch", myid);
        }
    }

    if (selected_count == 0) {
        Set_Hamiltonian_GpuSolver_Free_Cache();
        free(atom_displs);
        free(atom_counts);
        free(displs);
        free(counts);
        free(selected_roots);
        return;
    }

    /* All ranks must make the same decision before the optional order/overlap
       collectives.  A full refresh is mandatory at SCF 1 (new MD geometry),
       while later uncontracted SCFs can retain the invariant part. */
    local_rebuild_static =
        refresh_static || !cache->ready || cache->use_contracted != (use_contracted ? 1 : 0) ||
        cache->order_mode != order_mode || cache->spin_count != spin_count || cache->atom_count != atomnum ||
        cache->size != total_size || cache->owns != local_selected ||
        (local_selected && (cache->order_GA == NULL || cache->overlap == NULL));
    MPI_Allreduce(&local_rebuild_static, &rebuild_static, 1, MPI_INT, MPI_MAX, mpi_comm_level1);

    if (rebuild_static) {
        Set_Hamiltonian_GpuSolver_Free_Cache();
    }
    else {
        for (int spin = 0; spin < 4; spin++) {
            free(cache->h[spin]);
            cache->h[spin] = NULL;
        }
        for (int comp = 0; comp < 3; comp++) {
            free(cache->imnl[comp]);
            cache->imnl[comp] = NULL;
        }
        cache->ready = 0;
    }

    MPI_Comm_split(mpi_comm_level1, local_selected ? 1 : MPI_UNDEFINED, myid, &selected_comm);

    cache->owns = local_selected;
    cache->use_contracted = use_contracted ? 1 : 0;
    cache->order_mode = order_mode;
    cache->size = total_size;
    cache->spin_count = spin_count;
    cache->atom_count = atomnum;

    if (rebuild_static) {
        if (myid == first_selected_root) {
            cache->order_GA = (int *)Set_Hamiltonian_malloc(sizeof(int) * (size_t)(atomnum + 2),
                                                            "GPU solver atom order cache", myid);
        }

        MPI_Gatherv((0 < Matomnum) ? &M2G[1] : &dummy_atom, Matomnum, MPI_INT,
                    (myid == first_selected_root) ? &cache->order_GA[1] : NULL,
                    atom_counts, atom_displs, MPI_INT, first_selected_root, mpi_comm_level1);

        if (local_selected) {
            if (myid != first_selected_root) {
                cache->order_GA = (int *)Set_Hamiltonian_malloc(sizeof(int) * (size_t)(atomnum + 2),
                                                                "GPU solver atom order cache", myid);
            }
            cache->order_GA[0] = 0;
            cache->order_GA[atomnum + 1] = 0;
            MPI_Bcast(&cache->order_GA[1], atomnum, MPI_INT, 0, selected_comm);
        }
    }

    overlap_src = use_contracted ? CntOLP[0] : OLP[0];
    h_src = use_contracted ? CntH : H;
    imnl_src = use_contracted ? iCntHNL : iHNL;

    if (rebuild_static) {
        Set_Hamiltonian_GpuSolver_GatherMatrixToSelected(overlap_src, &cache->overlap, order_mode, local_size,
                                                        total_size, counts, displs, selected_roots, selected_comm,
                                                        first_selected_root, myid);
    }

    for (int spin = 0; spin < spin_count; spin++) {
        Set_Hamiltonian_GpuSolver_GatherMatrixToSelected(h_src[spin], &cache->h[spin], order_mode, local_size,
                                                        total_size, counts, displs, selected_roots, selected_comm,
                                                        first_selected_root, myid);
    }

    if (SpinP_switch == 3 && imnl_src != NULL) {
        for (int comp = 0; comp < 3; comp++) {
            Set_Hamiltonian_GpuSolver_GatherMatrixToSelected(imnl_src[comp], &cache->imnl[comp], order_mode,
                                                            local_size, total_size, counts, displs, selected_roots,
                                                            selected_comm, first_selected_root, myid);
        }
    }

    cache->ready = 1;

    {
        const char *trace = getenv("OPENMX_BAND_CACHE_TRACE");
        if (myid == Host_ID && trace != NULL && atoi(trace) != 0) {
            printf("<DFT> packed HS cache: overlap/order %s, Hamiltonian refreshed\n",
                   rebuild_static ? "rebuilt" : "reused");
            fflush(stdout);
        }
    }

    if (selected_comm != MPI_COMM_NULL) {
        MPI_Comm_free(&selected_comm);
    }

    free(atom_displs);
    free(atom_counts);
    free(displs);
    free(counts);
    free(selected_roots);
}

int Set_Hamiltonian_GpuSolver_Packed_CacheReady(void)
{
    return Set_Hamiltonian_GpuSolver_Cache.ready;
}

int Set_Hamiltonian_GpuSolver_Packed_OwnsCache(void)
{
    SetHamiltonianGpuSolverPackedCache *cache = &Set_Hamiltonian_GpuSolver_Cache;
    return (cache->ready && cache->owns);
}

int Set_Hamiltonian_GpuSolver_Packed_OrderMode(void)
{
    return Set_Hamiltonian_GpuSolver_Cache.order_mode;
}

int Set_Hamiltonian_GpuSolver_Packed_Size(void)
{
    return Set_Hamiltonian_GpuSolver_Cache.size;
}

int *Set_Hamiltonian_GpuSolver_Packed_OrderGA(void)
{
    return Set_Hamiltonian_GpuSolver_Cache.order_GA;
}

double *Set_Hamiltonian_GpuSolver_Packed_Overlap(void)
{
    return Set_Hamiltonian_GpuSolver_Cache.overlap;
}

double *Set_Hamiltonian_GpuSolver_Packed_H(int spin)
{
    if (spin < 0 || 4 <= spin) {
        return NULL;
    }
    return Set_Hamiltonian_GpuSolver_Cache.h[spin];
}

double *Set_Hamiltonian_GpuSolver_Packed_ImNL(int comp)
{
    if (comp < 0 || 3 <= comp) {
        return NULL;
    }
    return Set_Hamiltonian_GpuSolver_Cache.imnl[comp];
}

double Set_Hamiltonian(char * mode, int MD_iter, int SCF_iter, int SCF_iter0, int TRAN_Poisson_flag2,
                       int SucceedReadingDMfile, int Cnt_kind, double ***** H0, double ***** HNL, double ***** CDM,
                       double ***** H)
{
    /***************************************************************
      Cnt_kind
        0:  Uncontracted Hamiltonian
        1:  Contracted Hamiltonian
  ***************************************************************/

    int    Mc_AN, Gc_AN, Mh_AN, h_AN, Gh_AN;
    int    i, j, k, Cwan, Hwan, NO0, NO1, spin, N, NOLG;
    int    Nc, Ncs, GNc, GRc, Nog, Nh, MN, XC_P_switch;
    double TStime, TEtime;
    double trace_base_end = 0.0, trace_vpot_end = 0.0, trace_matrix_end = 0.0;
    int    numprocs, myid;
    double time0, time1, time2, mflops;
    long   Num_C0, Num_C1;
    int    use_base_openmp;
    const char *trace_env;
    int trace_timing;
    double prof_t0 = 0.0;

    MPI_Comm_size(mpi_comm_level1, &numprocs);
    MPI_Comm_rank(mpi_comm_level1, &myid);
    MPI_Barrier(mpi_comm_level1);
    dtime(&TStime);
    trace_env = getenv("OPENMX_SETHAM_TIMING");
    trace_timing = (trace_env != NULL && atoi(trace_env) != 0);

    if (Cnt_kind != 0 && Cnt_kind != 1) {
        Set_Hamiltonian_abort("Set_Hamiltonian", "Cnt_kind must be 0 or 1", myid);
    }

    if (SpinP_switch != 0 && SpinP_switch != 1 && SpinP_switch != 3) {
        Set_Hamiltonian_abort("Set_Hamiltonian", "SpinP_switch must be 0, 1, or 3", myid);
    }

    SETH_PROF_T0(prof_t0);
    use_base_openmp = Set_Hamiltonian_OpenMP_Enabled();
    MPI_Allreduce(MPI_IN_PLACE,&use_base_openmp,1,MPI_INT,MPI_MAX,mpi_comm_level1);
    SETH_PROF_ADD(memok, prof_t0);

    /*****************************************************
                adding H0+HNL+(HCH) to H
  *****************************************************/

    if (measure_time)
        dtime(&time1);

    SETH_PROF_T0(prof_t0);
    if (use_base_openmp) {
        Set_Hamiltonian_Base_OpenMP(SCF_iter, H0, HNL, H);
    }

    /* spin non-collinear */

    else if (SpinP_switch == 3) {
#pragma omp parallel for if (omp_get_max_threads() > 1) private(Mc_AN, Gc_AN, Cwan, h_AN, Gh_AN, Hwan, i, j) schedule(static)
        for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
            Gc_AN = M2G[Mc_AN];
            Cwan  = WhatSpecies[Gc_AN];
            for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
                Gh_AN = natn[Gc_AN][h_AN];

                Hwan = WhatSpecies[Gh_AN];
                for (i = 0; i < Spe_Total_NO[Cwan]; i++) {
                    for (j = 0; j < Spe_Total_NO[Hwan]; j++) {

                        if (ProExpn_VNA == 0) {
                            H[0][Mc_AN][h_AN][i][j] =
                                F_Kin_flag * H0[0][Mc_AN][h_AN][i][j] + F_NL_flag * HNL[0][Mc_AN][h_AN][i][j];
                            H[1][Mc_AN][h_AN][i][j] =
                                F_Kin_flag * H0[0][Mc_AN][h_AN][i][j] + F_NL_flag * HNL[1][Mc_AN][h_AN][i][j];
                            H[2][Mc_AN][h_AN][i][j] = F_NL_flag * HNL[2][Mc_AN][h_AN][i][j];
                            H[3][Mc_AN][h_AN][i][j] = 0.0;
                        } else {
                            H[0][Mc_AN][h_AN][i][j] = F_Kin_flag * H0[0][Mc_AN][h_AN][i][j] +
                                                      F_VNA_flag * HVNA[Mc_AN][h_AN][i][j] +
                                                      F_NL_flag * HNL[0][Mc_AN][h_AN][i][j];
                            H[1][Mc_AN][h_AN][i][j] = F_Kin_flag * H0[0][Mc_AN][h_AN][i][j] +
                                                      F_VNA_flag * HVNA[Mc_AN][h_AN][i][j] +
                                                      F_NL_flag * HNL[1][Mc_AN][h_AN][i][j];
                            H[2][Mc_AN][h_AN][i][j] = F_NL_flag * HNL[2][Mc_AN][h_AN][i][j];
                            H[3][Mc_AN][h_AN][i][j] = 0.0;
                        }

                        /* Effective Hubbard Hamiltonain --- added by MJ */

                        if ((Hub_U_switch == 1 || 1 <= Constraint_NCS_switch) && F_U_flag == 1 && 2 <= SCF_iter) {
                            H[0][Mc_AN][h_AN][i][j] += H_Hub[0][Mc_AN][h_AN][i][j];
                            H[1][Mc_AN][h_AN][i][j] += H_Hub[1][Mc_AN][h_AN][i][j];
                            H[2][Mc_AN][h_AN][i][j] += H_Hub[2][Mc_AN][h_AN][i][j];
                        }

                        /* core hole Hamiltonain */

                        if (core_hole_state_flag == 1) {
                            H[0][Mc_AN][h_AN][i][j] += HCH[0][Mc_AN][h_AN][i][j];
                            H[1][Mc_AN][h_AN][i][j] += HCH[1][Mc_AN][h_AN][i][j];
                            H[2][Mc_AN][h_AN][i][j] += HCH[2][Mc_AN][h_AN][i][j];
                        }
                    }
                }
            }
        }
    }

    /* spin collinear */

    else {

#pragma omp parallel for if (omp_get_max_threads() > 1) private(Mc_AN, Gc_AN, Cwan, h_AN, Gh_AN, Hwan, i, j, spin) schedule(static)
        for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
            Gc_AN = M2G[Mc_AN];
            Cwan  = WhatSpecies[Gc_AN];
            for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
                Gh_AN = natn[Gc_AN][h_AN];
                Hwan  = WhatSpecies[Gh_AN];
                for (i = 0; i < Spe_Total_NO[Cwan]; i++) {
                    for (j = 0; j < Spe_Total_NO[Hwan]; j++) {
                        for (spin = 0; spin <= SpinP_switch; spin++) {

                            if (ProExpn_VNA == 0) {
                                H[spin][Mc_AN][h_AN][i][j] =
                                    F_Kin_flag * H0[0][Mc_AN][h_AN][i][j] + F_NL_flag * HNL[spin][Mc_AN][h_AN][i][j];
                            } else {
                                H[spin][Mc_AN][h_AN][i][j] = F_Kin_flag * H0[0][Mc_AN][h_AN][i][j] +
                                                             F_VNA_flag * HVNA[Mc_AN][h_AN][i][j] +
                                                             F_NL_flag * HNL[spin][Mc_AN][h_AN][i][j];
                            }

                            /* Effective Hubbard Hamiltonain --- added by MJ */
                            if ((Hub_U_switch == 1 || 1 <= Constraint_NCS_switch) && F_U_flag == 1 && 2 <= SCF_iter) {
                                H[spin][Mc_AN][h_AN][i][j] += H_Hub[spin][Mc_AN][h_AN][i][j];
                            }

                            /* core hole Hamiltonain */
                            if (core_hole_state_flag == 1) {
                                H[spin][Mc_AN][h_AN][i][j] += HCH[spin][Mc_AN][h_AN][i][j];
                            }
                        }
                    }
                }
            }
        }
    }

    if (measure_time) {
        dtime(&time2);
        printf("myid=%4d Time1=%18.10f\n", myid, time2 - time1);
        fflush(stdout);
    }
    if (trace_timing) dtime(&trace_base_end);

    if (Cnt_kind == 1) {
        Contract_Hamiltonian(H, CntH, OLP, CntOLP);
        if (SO_switch == 1)
            Contract_iHNL(iHNL, iCntHNL);
    }
    SETH_PROF_ADD(base, prof_t0);

    /*****************************************************
   calculation of Vpot;
  *****************************************************/

    if (myid == 0 && measure_time)
        dtime(&time1);

    XC_P_switch = 1;
    SETH_PROF_T0(prof_t0);
    Set_Vpot(MD_iter, SCF_iter, SCF_iter0, TRAN_Poisson_flag2, XC_P_switch);
    SETH_PROF_ADD(vpot, prof_t0);
    if (trace_timing) dtime(&trace_vpot_end);

    if (measure_time) {
        dtime(&time2);
        printf("myid=%4d Time2=%18.10f\n", myid, time2 - time1);
        fflush(stdout);
    }

    /*****************************************************
   calculation of matrix elements for dVH + Vxc (+ VNA)
  *****************************************************/

    SETH_PROF_T0(prof_t0);
    Calc_MatrixElements_dVH_Vxc_VNA(Cnt_kind);
    SETH_PROF_ADD(matel, prof_t0);
    if (trace_timing) dtime(&trace_matrix_end);

    /* for time */
    if (measure_time)
        dtime(&time1);
    SETH_PROF_T0(prof_t0);
    MPI_Barrier(mpi_comm_level1);
    SETH_PROF_ADD(barrier, prof_t0);
    {
        int local_gpu = Set_Hamiltonian_MatrixElements_GPU_Used;
        int any_gpu = 0;
        if (scf_eigen_lib_flag == GPUSOLVER)
            MPI_Allreduce(&local_gpu, &any_gpu, 1, MPI_INT, MPI_MAX, mpi_comm_level1);
        if (myid == Host_ID && mode != NULL && strcasecmp(mode, "stdout") == 0 && 0 < level_stdout) {
            printf("<Set_Hamiltonian>  Hamiltonian matrix for VNA+dVH+Vxc%s...\n",
                   any_gpu ? " (GPU-accelerated)" : "");
            fflush(stdout);
        }
    }
    if (measure_time) {
        dtime(&time2);
        printf("myid=%4d Time Barrier=%18.10f\n", myid, time2 - time1);
        fflush(stdout);
    }

    if (SetH_ProfileEnabled()) {
        fprintf(stderr, "SETHPROF id=%d it=%d memok=%.3f base=%.3f vpot=%.3f matel=%.3f bar=%.3f\n", myid, SCF_iter,
                SetH_prof.memok, SetH_prof.base, SetH_prof.vpot, SetH_prof.matel, SetH_prof.barrier);
        memset(&SetH_prof, 0, sizeof(SetH_prof));
    }

    dtime(&TEtime);
    time0 = TEtime - TStime;
    if (trace_timing) {
        double local_phase[5], max_phase[5];

        local_phase[0] = trace_base_end - TStime;
        local_phase[1] = trace_vpot_end - trace_base_end;
        local_phase[2] = trace_matrix_end - trace_vpot_end;
        local_phase[3] = TEtime - trace_matrix_end;
        local_phase[4] = time0;
        MPI_Reduce(local_phase, max_phase, 5, MPI_DOUBLE, MPI_MAX,
                   Host_ID, mpi_comm_level1);
        if (myid == Host_ID) {
            printf("<Set_Hamiltonian> timing SCF=%d max: base=%.6f Vpot/XC=%.6f matrix=%.6f barrier=%.6f total=%.6f s\n",
                   SCF_iter, max_phase[0], max_phase[1], max_phase[2],
                   max_phase[3], max_phase[4]);
            fflush(stdout);
        }
    }
    return time0;
}

/* ---------------------------------------------------------------------
   On-the-fly matrix elements (port of the NVIDIA version's
   Set_Hamiltonian_OnTheFly_MatrixElements): a rank whose orbital tables
   are not resident on the device evaluates, batch by batch of its atoms,
   the central atoms' orbitals over their spheres and the neighbours'
   orbitals at the overlap points on the device (orbs_otf_hip.cu), gathers
   the potential there from the grid uploaded once per call, and runs the
   same matrix-element kernel; only the potential and the H blocks travel.
   The device holds the atoms' sphere lists (a few MiB) and one batch
   arena of OPENMX_SETHAM_ONTHEFLY_MB (default 256).
   OPENMX_SETHAM_ONTHEFLY=0 turns it off.
   --------------------------------------------------------------------- */
int Set_Hamiltonian_Hip_OTF_VpotGather(int pair_count, int max_nolg, int spin_count, size_t batch_nolg,
                                       const int *d_nolg, const size_t *d_noff, const size_t *d_sph,
                                       const int *d_nc, const int *d_mgla, const double *d_vgrid,
                                       size_t vgrid_len, double gridvol, double *d_vb);
int Set_Hamiltonian_Hip_MatrixElements_Device(
    int pair_count, int spin_count, int max_output_count, int max_no, int max_nolg, size_t total_h,
    size_t total_nolg,
    const int *d_no0, const int *d_no1, const int *d_nolg,
    const size_t *d_hoff, const size_t *d_noff,
    const size_t *d_o0off, const size_t *d_o1off, const int *d_nc,
    const double *d_vpot, const float *d_o0, const float *d_o1, double *d_h);

typedef struct {
    int ready, matomnum;
    int *d_gla, *d_cla, *d_mgla;   /* every local atom's sphere: grid index, cell, grid-C index */
    size_t *atom_sph;              /* host: first sphere entry of each local atom (Matomnum + 2) */
    size_t sph_total;
} SetHamOTFState;

static SetHamOTFState Set_Hamiltonian_OTF_State;

static void SetH_OTF_ReleasePlans(void);

/* the device side of the on-the-fly state (the sphere indices); the batch
   plans, host memory of this geometry, stay (DemoteResidency) */
static void Set_Hamiltonian_OTF_ReleaseDevice(void)
{
    SetHamOTFState *st = &Set_Hamiltonian_OTF_State;

    if (st->d_gla != NULL) hipFree(st->d_gla);
    if (st->d_cla != NULL) hipFree(st->d_cla);
    if (st->d_mgla != NULL) hipFree(st->d_mgla);
    (void)hipGetLastError();
    free(st->atom_sph);
    memset(st, 0, sizeof(*st));
}

/* the whole on-the-fly state with the plans and their page-locked host
   memory: at a new geometry (Set_Orbitals_Grid), at the end of the SCF
   cycle and at the run boundary (Free_Arrays) */
void Set_Hamiltonian_OTF_Release(void)
{
    SetH_OTF_ReleasePlans();
    Set_Hamiltonian_OTF_ReleaseDevice();
}

/* the end of an SCF cycle (the NVIDIA version's hook of the same name):
   the on-the-fly state and the PAO tables of the tiles go, so that the
   forces find the device memory free; a later Set_Density_Grid or the next
   geometry rebuilds them */
void Set_Hamiltonian_OnTheFly_EndCycle(void)
{
    Set_Hamiltonian_OTF_Release();
    Set_Orbitals_Grid_OTF_Release();
}

/* the device bytes of the on-the-fly state of this rank */
static size_t Set_Hamiltonian_OTF_DeviceBytes(void)
{
    const SetHamOTFState *st = &Set_Hamiltonian_OTF_State;
    size_t bytes = OTF_Orbitals_Bytes();

    if (st->d_gla != NULL) bytes += 3 * sizeof(int) * st->sph_total;
    return bytes;
}

/* The on-the-fly tiles hold no tables (a few MiB of sphere indices and a
   transient arena), so every rank may take them, as every rank enters the
   NVIDIA version's GPU scheduler; OPENMX_SETHAM_GPU_KOWNER_ONLY=1 keeps
   them to the ranks selected for the table path (one per k-point group). */
static int Set_Hamiltonian_OTF_Rank_Selected(void)
{
    static int kowner_only = -1;

    if (kowner_only < 0) {
        const char *value = getenv("OPENMX_SETHAM_GPU_KOWNER_ONLY");
        kowner_only = (value != NULL && value[0] != '\0' && atoi(value) != 0) ? 1 : 0;
    }
    return kowner_only ? Set_Hamiltonian_OpenMP_Rank_Selected : 1;
}

static int Set_Hamiltonian_OTF_Enabled(void)
{
    static int enabled = -1;

    if (enabled < 0) {
        const char *value = getenv("OPENMX_SETHAM_ONTHEFLY");
        enabled = (value == NULL || value[0] == '\0' || atoi(value) != 0) ? 1 : 0;
    }
    return enabled;
}

/* the arena budget of a batch: OPENMX_SETHAM_ONTHEFLY_MB (default 256) for
   the matrix elements (kind 0); the density (kind 1) takes
   OPENMX_SETDEN_ONTHEFLY_MB when it is set, else the same value */
static size_t Set_Hamiltonian_OTF_Budget(int kind)
{
    static size_t budget[2] = {0, 0};

    if (budget[0] == 0) {
        const char *value = getenv("OPENMX_SETHAM_ONTHEFLY_MB");
        const char *den = getenv("OPENMX_SETDEN_ONTHEFLY_MB");
        long mib = (value != NULL && value[0] != '\0') ? atol(value) : 256;
        long den_mib;

        if (mib < 32) mib = 32;
        den_mib = (den != NULL && den[0] != '\0') ? atol(den) : mib;
        if (den_mib < 32) den_mib = 32;
        budget[0] = (size_t)mib * 1024 * 1024;
        budget[1] = (size_t)den_mib * 1024 * 1024;
    }
    return budget[kind == 1 ? 1 : 0];
}

/* the spheres of the local atoms on the device, once per geometry */
static int Set_Hamiltonian_OTF_Ensure(void)
{
    SetHamOTFState *st = &Set_Hamiltonian_OTF_State;
    int *gla = NULL, *cla = NULL, *mgla = NULL;
    int Mc_AN, ok = 0;

    if (st->ready && st->matomnum == Matomnum) return 1;
    Set_Hamiltonian_OTF_ReleaseDevice();
    st->matomnum = Matomnum;
    st->atom_sph = (size_t *)malloc(sizeof(size_t) * (size_t)(Matomnum + 2));
    if (st->atom_sph == NULL) return 0;
    st->sph_total = 0;
    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
        st->atom_sph[Mc_AN] = st->sph_total;
        st->sph_total += (size_t)GridN_Atom[M2G[Mc_AN]];
    }
    gla = (int *)malloc(sizeof(int) * (st->sph_total ? st->sph_total : 1));
    cla = (int *)malloc(sizeof(int) * (st->sph_total ? st->sph_total : 1));
    mgla = (int *)malloc(sizeof(int) * (st->sph_total ? st->sph_total : 1));
    if (gla == NULL || cla == NULL || mgla == NULL) goto done;
    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
        const int Gc_AN = M2G[Mc_AN];
        const size_t base = st->atom_sph[Mc_AN];
        int Nc;

        for (Nc = 0; Nc < GridN_Atom[Gc_AN]; Nc++) {
            gla[base + (size_t)Nc] = GridListAtom[Mc_AN][Nc];
            cla[base + (size_t)Nc] = CellListAtom[Mc_AN][Nc];
            mgla[base + (size_t)Nc] = MGridListAtom[Mc_AN][Nc];
        }
    }
    if (hipMalloc((void **)&st->d_gla, sizeof(int) * (st->sph_total ? st->sph_total : 1)) != hipSuccess ||
        hipMalloc((void **)&st->d_cla, sizeof(int) * (st->sph_total ? st->sph_total : 1)) != hipSuccess ||
        hipMalloc((void **)&st->d_mgla, sizeof(int) * (st->sph_total ? st->sph_total : 1)) != hipSuccess ||
        hipMemcpy(st->d_gla, gla, sizeof(int) * st->sph_total, hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(st->d_cla, cla, sizeof(int) * st->sph_total, hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(st->d_mgla, mgla, sizeof(int) * st->sph_total, hipMemcpyHostToDevice) != hipSuccess)
        goto done;
    st->ready = 1;
    ok = 1;

done:
    free(gla);
    free(cla);
    free(mgla);
    if (!ok) Set_Hamiltonian_OTF_ReleaseDevice();
    return ok;
}

#define SETH_OTF_ALIGN(x) (((x) + (size_t)255) & ~(size_t)255)

/* The free-memory test and the allocation of an on-the-fly arena as one
   step among the processes of the node (under the lock of the device
   probe, set_hip_default_device_from_local_rank.c): ranks that allocate at
   the same moment then each see the others' arenas and all keep the
   reserve.  NULL when the device has no room (or the allocation fails). */
static void *SetH_OTF_LockedAlloc(size_t bytes, size_t reserve, size_t *free_out)
{
    char path[64];
    int fd;
    void *p = NULL;
    size_t free_bytes = 0, total_bytes = 0;

    snprintf(path, sizeof(path), "/tmp/.openmx_gpu_probe.%ld", (long)getuid());
    fd = open(path, O_CREAT | O_RDWR, 0600);   /* closed right after the lock */
    if (0 <= fd) {
        while (flock(fd, LOCK_EX) != 0 && errno == EINTR);
    }
    if (hipMemGetInfo(&free_bytes, &total_bytes) == hipSuccess && bytes + reserve <= free_bytes) {
        if (hipMalloc(&p, bytes) != hipSuccess) p = NULL;
    }
    (void)hipGetLastError();
    if (0 <= fd) {
        (void)flock(fd, LOCK_UN);
        (void)close(fd);
    }
    if (free_out != NULL) *free_out = free_bytes;
    return p;
}

/* the device bytes of a batch of atoms [a0, a1): the meta image, the H
   blocks, the potential at the overlap points and the two orbital tiles */
static void Set_Hamiltonian_OTF_BatchSizes(int a0, int a1, int spin_count, size_t *image_bytes, size_t *h_count,
                                           size_t *nolg_count, size_t *o0_count, size_t *o1_count, int *pair_count,
                                           int *max_nolg, int *max_sphere, int *max_output)
{
    size_t na = 0, np = 0, bn = 0, bo0 = 0, bo1 = 0, bh = 0;
    int Mc_AN;

    *max_nolg = 0;
    *max_sphere = 0;
    *max_output = 0;
    for (Mc_AN = a0; Mc_AN < a1; Mc_AN++) {
        const int Gc_AN = M2G[Mc_AN];
        const int NO0 = Spe_Total_NO[WhatSpecies[Gc_AN]];
        int h_AN;

        na++;
        bo0 += (size_t)GridN_Atom[Gc_AN] * (size_t)NO0;
        if (*max_sphere < GridN_Atom[Gc_AN]) *max_sphere = GridN_Atom[Gc_AN];
        for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
            const int NO1 = Spe_Total_NO[WhatSpecies[natn[Gc_AN][h_AN]]];
            const int NOLG = NumOLG[Mc_AN][h_AN];

            np++;
            bn += (size_t)NOLG;
            bo1 += (size_t)NOLG * (size_t)NO1;
            bh += (size_t)spin_count * (size_t)NO0 * (size_t)NO1;
            if (*max_nolg < NOLG) *max_nolg = NOLG;
            if (*max_output < spin_count * NO0 * NO1) *max_output = spin_count * NO0 * NO1;
        }
    }
    *image_bytes = SETH_OTF_ALIGN(na * sizeof(OTF_Atom)) + SETH_OTF_ALIGN(na * sizeof(int)) +
                   3 * SETH_OTF_ALIGN(na * sizeof(size_t)) + SETH_OTF_ALIGN(np * sizeof(OTF_Atom)) +
                   SETH_OTF_ALIGN(np * sizeof(int)) + 3 * SETH_OTF_ALIGN(np * sizeof(size_t)) +
                   2 * SETH_OTF_ALIGN(np * sizeof(int)) + 2 * SETH_OTF_ALIGN(np * sizeof(size_t)) +
                   SETH_OTF_ALIGN(bn * sizeof(int)) + SETH_OTF_ALIGN(bh * sizeof(double));
    *h_count = bh;
    *nolg_count = bn;
    *o0_count = bo0;
    *o1_count = bo1;
    *pair_count = (int)np;
}

/* ---------------------------------------------------------------------
   Cached batch plans and page-locked transfers of the on-the-fly tiles
   (the NVIDIA version's SETH_OTF_GetPlan and its pinned staging).  The
   batches, their descriptors (the atoms and pairs, their point counts and
   offsets) and the overlap-point lists depend on the geometry, the species
   and the budget only, not on the SCF step: they are built once per
   geometry, the point lists of all the rank's pairs in one array.  Per call
   only the H (or DM) blocks are packed, in one page-locked array for the
   whole rank, and every transfer runs from page-locked memory: per batch
   the descriptors, the batch's slice of the point lists and of the H/DM
   blocks go up, and the results come back into the same H array (or
   through a page-locked bounce buffer into Tmp_Den_Grid).
   OPENMX_SETHAM_ONTHEFLY_PLAN_CACHE=0 rebuilds the plan in every call;
   OPENMX_SETHAM_ONTHEFLY_PINNED_MB (default 1024) caps the page-locked
   bytes of a rank, beyond which pageable memory is used.
   --------------------------------------------------------------------- */
typedef struct {
    int a0, a1, na, np;
    int max_nolg, max_sphere, max_output, max_no;
    size_t bh, bn, bo0, bo1, sph;      /* H/DM entries, overlap points, orbital values, sphere points */
    size_t dyn_first;                  /* the batch's first H/DM entry in the rank's packing */
    size_t nc_first;                   /* the batch's first overlap point in the rank's point lists */
    size_t sph_first;                  /* the batch's first sphere point (Set_Hamiltonian_OTF_State.atom_sph) */
    /* the arena: the descriptors [0, o_nc), the point lists, the H or DM
       blocks, then the device-only sections */
    size_t o_a_items, o_a_npts, o_a_ioff, o_a_ooff, o_a_soff;
    size_t o_p_items, o_p_npts, o_p_ioff, o_p_ooff, o_p_soff;
    size_t o_no0, o_no1, o_hoff, o_o0off, o_denoff;
    size_t o_nc, o_dyn, o_vb, o_den, o_o0, o_o1, end;
    unsigned char *desc;               /* host image of [0, o_nc) */
    int desc_pinned;
} SetHOTFBatch;

typedef struct {
    int valid, kind, matomnum, spin_count;
    size_t budget;
    int batch_count;
    SetHOTFBatch *batch;
    size_t arena_bytes;
    double *dyn;                       /* the H (kind 0) or DM (kind 1) blocks of the whole rank */
    size_t dyn_count;
    int dyn_pinned;
    double *bounce;                    /* kind 1: the density of one batch, all spins */
    size_t bounce_count;
    int bounce_pinned;
} SetHOTFPlan;

static SetHOTFPlan SetH_OTF_Plan[2];

/* the overlap points of all the rank's pairs (GListTAtoms1, atom and
   neighbour order) and the first point of every atom */
static struct {
    int ready, matomnum;
    int *nc;
    size_t nc_count;
    int nc_pinned;
    size_t *atom_nc;                   /* Matomnum + 2 */
    double *vstage;                    /* the potential grid on its way to the device */
    size_t vstage_count;
    int vstage_pinned;
} SetH_OTF_Static;

static size_t SetH_OTF_PinnedBytes = 0;

static int SetH_OTF_PlanCacheEnabled(void)
{
    static int enabled = -1;

    if (enabled < 0) {
        const char *value = getenv("OPENMX_SETHAM_ONTHEFLY_PLAN_CACHE");
        enabled = (value == NULL || value[0] == '\0' || atoi(value) != 0) ? 1 : 0;
    }
    return enabled;
}

/* page-locked while the rank's page-locked bytes stay within the cap, else
   pageable; NULL only when both fail */
static void *SetH_OTF_HostAlloc(size_t bytes, int *pinned)
{
    static long cap_mib = -1;
    void *p = NULL;

    if (cap_mib < 0) {
        const char *value = getenv("OPENMX_SETHAM_ONTHEFLY_PINNED_MB");
        cap_mib = (value != NULL && value[0] != '\0') ? atol(value) : 1024;
        if (cap_mib < 0) cap_mib = 0;
    }
    if (bytes == 0) bytes = 1;
    *pinned = 0;
    if (SetH_OTF_PinnedBytes + bytes <= (size_t)cap_mib * 1024 * 1024) {
        if (hipHostMalloc(&p, bytes, 0) == hipSuccess && p != NULL) {
            *pinned = 1;
            SetH_OTF_PinnedBytes += bytes;
            return p;
        }
        (void)hipGetLastError();
        p = NULL;
    }
    return malloc(bytes);
}

static void SetH_OTF_HostFree(void *p, int pinned, size_t bytes)
{
    if (p == NULL) return;
    if (bytes == 0) bytes = 1;
    if (pinned) {
        (void)hipHostFree(p);
        (void)hipGetLastError();
        SetH_OTF_PinnedBytes = (bytes <= SetH_OTF_PinnedBytes) ? SetH_OTF_PinnedBytes - bytes : 0;
    }
    else free(p);
}

static void SetH_OTF_FreePlan(SetHOTFPlan *pl)
{
    int b;

    if (pl->batch != NULL) {
        for (b = 0; b < pl->batch_count; b++)
            SetH_OTF_HostFree(pl->batch[b].desc, pl->batch[b].desc_pinned, pl->batch[b].o_nc);
        free(pl->batch);
    }
    SetH_OTF_HostFree(pl->dyn, pl->dyn_pinned, sizeof(double) * pl->dyn_count);
    SetH_OTF_HostFree(pl->bounce, pl->bounce_pinned, sizeof(double) * pl->bounce_count);
    memset(pl, 0, sizeof(*pl));
}

/* the plans and the point lists: with a new geometry (Set_Orbitals_Grid)
   and whenever the on-the-fly state goes */
static void SetH_OTF_ReleasePlans(void)
{
    SetH_OTF_FreePlan(&SetH_OTF_Plan[0]);
    SetH_OTF_FreePlan(&SetH_OTF_Plan[1]);
    SetH_OTF_HostFree(SetH_OTF_Static.nc, SetH_OTF_Static.nc_pinned, sizeof(int) * SetH_OTF_Static.nc_count);
    free(SetH_OTF_Static.atom_nc);
    SetH_OTF_HostFree(SetH_OTF_Static.vstage, SetH_OTF_Static.vstage_pinned,
                      sizeof(double) * SetH_OTF_Static.vstage_count);
    memset(&SetH_OTF_Static, 0, sizeof(SetH_OTF_Static));
}

static int SetH_OTF_EnsureStatic(void)
{
    size_t count = 0;
    int Mc_AN;

    if (SetH_OTF_Static.ready && SetH_OTF_Static.matomnum == Matomnum) return 1;
    SetH_OTF_ReleasePlans();
    SetH_OTF_Static.atom_nc = (size_t *)malloc(sizeof(size_t) * (size_t)(Matomnum + 2));
    if (SetH_OTF_Static.atom_nc == NULL) return 0;
    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
        const int Gc_AN = M2G[Mc_AN];
        int h_AN;

        SetH_OTF_Static.atom_nc[Mc_AN] = count;
        for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) count += (size_t)NumOLG[Mc_AN][h_AN];
    }
    SetH_OTF_Static.atom_nc[Matomnum + 1] = count;
    SetH_OTF_Static.nc = (int *)SetH_OTF_HostAlloc(sizeof(int) * count, &SetH_OTF_Static.nc_pinned);
    if (SetH_OTF_Static.nc == NULL) {
        SetH_OTF_ReleasePlans();
        return 0;
    }
    SetH_OTF_Static.nc_count = count;
    count = 0;
    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
        const int Gc_AN = M2G[Mc_AN];
        int h_AN;

        for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
            const int NOLG = NumOLG[Mc_AN][h_AN];
            if (0 < NOLG) memcpy(SetH_OTF_Static.nc + count, GListTAtoms1[Mc_AN][h_AN], sizeof(int) * (size_t)NOLG);
            count += (size_t)NOLG;
        }
    }
    SetH_OTF_Static.matomnum = Matomnum;
    SetH_OTF_Static.ready = 1;
    return 1;
}

/* the device bytes of a batch of kind 0 (matrix elements: the potential at
   the overlap points) or 1 (density: the density of the atoms' spheres) */
static size_t SetH_OTF_KindBatchBytes(int kind, int a0, int a1, int spin_count)
{
    size_t image, bh, bn, bo0, bo1, sph = 0;
    int np, mn, ms, mo, a;

    Set_Hamiltonian_OTF_BatchSizes(a0, a1, spin_count, &image, &bh, &bn, &bo0, &bo1, &np, &mn, &ms, &mo);
    if (kind == 0)
        return image + SETH_OTF_ALIGN((size_t)spin_count * bn * sizeof(double)) + SETH_OTF_ALIGN(bo0 * sizeof(float)) +
               SETH_OTF_ALIGN(bo1 * sizeof(float));
    for (a = a0; a < a1; a++) sph += (size_t)GridN_Atom[M2G[a]];
    return image + SETH_OTF_ALIGN((size_t)np * sizeof(size_t)) + SETH_OTF_ALIGN((size_t)spin_count * sph * sizeof(double)) +
           SETH_OTF_ALIGN(bo0 * sizeof(float)) + SETH_OTF_ALIGN(bo1 * sizeof(float));
}

/* the plan of kind 0 or 1 for this geometry, NULL when the host memory for
   it cannot be had */
static SetHOTFPlan *SetH_OTF_GetPlan(int kind, int spin_count)
{
    SetHOTFPlan *pl = &SetH_OTF_Plan[kind];
    const SetHamOTFState *st = &Set_Hamiltonian_OTF_State;
    const size_t budget = Set_Hamiltonian_OTF_Budget(kind);
    int *first = NULL;
    int count = 0, Mc_AN, b;
    size_t dyn_at = 0, max_bounce = 0;

    if (pl->valid && pl->matomnum == Matomnum && pl->spin_count == spin_count && pl->budget == budget) return pl;
    SetH_OTF_FreePlan(pl);
    if (!SetH_OTF_EnsureStatic()) return NULL;

    /* the batches: consecutive atoms while their arena fits the budget,
       at least one atom each */
    first = (int *)malloc(sizeof(int) * (size_t)(Matomnum + 2));
    if (first == NULL) return NULL;
    Mc_AN = 1;
    while (Mc_AN <= Matomnum) {
        int a1 = Mc_AN + 1;

        while (a1 <= Matomnum && SetH_OTF_KindBatchBytes(kind, Mc_AN, a1 + 1, spin_count) <= budget) a1++;
        first[count++] = Mc_AN;
        Mc_AN = a1;
    }
    first[count] = Matomnum + 1;
    pl->batch = (SetHOTFBatch *)calloc((size_t)count, sizeof(SetHOTFBatch));
    if (pl->batch == NULL) {
        free(first);
        return NULL;
    }
    pl->batch_count = count;
    pl->kind = kind;
    pl->matomnum = Matomnum;
    pl->spin_count = spin_count;
    pl->budget = budget;

    for (b = 0; b < count; b++) {
        SetHOTFBatch *bt = &pl->batch[b];
        size_t image_bytes, pos = 0, h_at = 0, n_at = 0, o0_at = 0, o1_at = 0, s_at = 0;
        int ia = 0, ip = 0;
        OTF_Atom *a_items, *p_items;
        int *a_npts, *p_npts, *no0, *no1;
        size_t *a_ioff, *a_ooff, *a_soff, *p_ioff, *p_ooff, *p_soff, *hoff, *o0off, *denoff = NULL;

        bt->a0 = first[b];
        bt->a1 = first[b + 1];
        bt->na = bt->a1 - bt->a0;
        Set_Hamiltonian_OTF_BatchSizes(bt->a0, bt->a1, spin_count, &image_bytes, &bt->bh, &bt->bn, &bt->bo0,
                                       &bt->bo1, &bt->np, &bt->max_nolg, &bt->max_sphere, &bt->max_output);
        for (Mc_AN = bt->a0; Mc_AN < bt->a1; Mc_AN++) bt->sph += (size_t)GridN_Atom[M2G[Mc_AN]];
        bt->dyn_first = dyn_at;
        bt->nc_first = SetH_OTF_Static.atom_nc[bt->a0];
        bt->sph_first = st->atom_sph[bt->a0];
#define SETH_OTF_SEC(off, bytes) do { (off) = pos; pos += SETH_OTF_ALIGN(bytes); } while (0)
        SETH_OTF_SEC(bt->o_a_items, (size_t)bt->na * sizeof(OTF_Atom));
        SETH_OTF_SEC(bt->o_a_npts, (size_t)bt->na * sizeof(int));
        SETH_OTF_SEC(bt->o_a_ioff, (size_t)bt->na * sizeof(size_t));
        SETH_OTF_SEC(bt->o_a_ooff, (size_t)bt->na * sizeof(size_t));
        SETH_OTF_SEC(bt->o_a_soff, (size_t)bt->na * sizeof(size_t));
        SETH_OTF_SEC(bt->o_p_items, (size_t)bt->np * sizeof(OTF_Atom));
        SETH_OTF_SEC(bt->o_p_npts, (size_t)bt->np * sizeof(int));
        SETH_OTF_SEC(bt->o_p_ioff, (size_t)bt->np * sizeof(size_t));
        SETH_OTF_SEC(bt->o_p_ooff, (size_t)bt->np * sizeof(size_t));
        SETH_OTF_SEC(bt->o_p_soff, (size_t)bt->np * sizeof(size_t));
        SETH_OTF_SEC(bt->o_no0, (size_t)bt->np * sizeof(int));
        SETH_OTF_SEC(bt->o_no1, (size_t)bt->np * sizeof(int));
        SETH_OTF_SEC(bt->o_hoff, (size_t)bt->np * sizeof(size_t));
        SETH_OTF_SEC(bt->o_o0off, (size_t)bt->np * sizeof(size_t));
        if (kind == 1) SETH_OTF_SEC(bt->o_denoff, (size_t)bt->np * sizeof(size_t));
        SETH_OTF_SEC(bt->o_nc, bt->bn * sizeof(int));
        SETH_OTF_SEC(bt->o_dyn, bt->bh * sizeof(double));
        /* device only */
        if (kind == 0) SETH_OTF_SEC(bt->o_vb, (size_t)spin_count * bt->bn * sizeof(double));
        else SETH_OTF_SEC(bt->o_den, (size_t)spin_count * bt->sph * sizeof(double));
        SETH_OTF_SEC(bt->o_o0, bt->bo0 * sizeof(float));
        SETH_OTF_SEC(bt->o_o1, bt->bo1 * sizeof(float));
#undef SETH_OTF_SEC
        bt->end = pos;
        if (pl->arena_bytes < pos) pl->arena_bytes = pos;
        if (kind == 1 && max_bounce < (size_t)spin_count * bt->sph) max_bounce = (size_t)spin_count * bt->sph;
        dyn_at += bt->bh;

        bt->desc = (unsigned char *)SetH_OTF_HostAlloc(bt->o_nc, &bt->desc_pinned);
        if (bt->desc == NULL) goto fail;
        a_items = (OTF_Atom *)(void *)(bt->desc + bt->o_a_items);
        a_npts = (int *)(void *)(bt->desc + bt->o_a_npts);
        a_ioff = (size_t *)(void *)(bt->desc + bt->o_a_ioff);
        a_ooff = (size_t *)(void *)(bt->desc + bt->o_a_ooff);
        a_soff = (size_t *)(void *)(bt->desc + bt->o_a_soff);
        p_items = (OTF_Atom *)(void *)(bt->desc + bt->o_p_items);
        p_npts = (int *)(void *)(bt->desc + bt->o_p_npts);
        p_ioff = (size_t *)(void *)(bt->desc + bt->o_p_ioff);
        p_ooff = (size_t *)(void *)(bt->desc + bt->o_p_ooff);
        p_soff = (size_t *)(void *)(bt->desc + bt->o_p_soff);
        no0 = (int *)(void *)(bt->desc + bt->o_no0);
        no1 = (int *)(void *)(bt->desc + bt->o_no1);
        hoff = (size_t *)(void *)(bt->desc + bt->o_hoff);
        o0off = (size_t *)(void *)(bt->desc + bt->o_o0off);
        if (kind == 1) denoff = (size_t *)(void *)(bt->desc + bt->o_denoff);

        for (Mc_AN = bt->a0; Mc_AN < bt->a1; Mc_AN++, ia++) {
            const int Gc_AN = M2G[Mc_AN];
            const int Cwan = WhatSpecies[Gc_AN];
            const int NO0 = Spe_Total_NO[Cwan];
            const size_t atom_o0 = o0_at, atom_den = s_at;
            int h_AN;

            if (bt->max_no < NO0) bt->max_no = NO0;
            memset(&a_items[ia], 0, sizeof(OTF_Atom));
            a_items[ia].wan = Cwan;
            a_items[ia].no = NO0;
            a_items[ia].gx = Gxyz[Gc_AN][1];
            a_items[ia].gy = Gxyz[Gc_AN][2];
            a_items[ia].gz = Gxyz[Gc_AN][3];
            a_npts[ia] = GridN_Atom[Gc_AN];
            a_ioff[ia] = 0;
            a_ooff[ia] = atom_o0;
            a_soff[ia] = st->atom_sph[Mc_AN];
            o0_at += (size_t)GridN_Atom[Gc_AN] * (size_t)NO0;
            s_at += (size_t)GridN_Atom[Gc_AN];

            for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++, ip++) {
                const int Gh_AN = natn[Gc_AN][h_AN];
                const int Rnh = ncn[Gc_AN][h_AN];
                const int Hwan = WhatSpecies[Gh_AN];
                const int NO1 = Spe_Total_NO[Hwan];
                const int NOLG = NumOLG[Mc_AN][h_AN];

                if (bt->max_no < NO1) bt->max_no = NO1;
                memset(&p_items[ip], 0, sizeof(OTF_Atom));
                p_items[ip].wan = Hwan;
                p_items[ip].no = NO1;
                p_items[ip].gx = Gxyz[Gh_AN][1];
                p_items[ip].gy = Gxyz[Gh_AN][2];
                p_items[ip].gz = Gxyz[Gh_AN][3];
                p_items[ip].ax = atv[Rnh][1];
                p_items[ip].ay = atv[Rnh][2];
                p_items[ip].az = atv[Rnh][3];
                p_npts[ip] = NOLG;
                p_ioff[ip] = n_at;
                p_ooff[ip] = o1_at;
                p_soff[ip] = st->atom_sph[Mc_AN];
                no0[ip] = NO0;
                no1[ip] = NO1;
                hoff[ip] = h_at;
                o0off[ip] = atom_o0;
                if (kind == 1) denoff[ip] = atom_den;
                n_at += (size_t)NOLG;
                o1_at += (size_t)NOLG * (size_t)NO1;
                h_at += (size_t)spin_count * (size_t)NO0 * (size_t)NO1;
            }
        }
    }
    free(first);
    first = NULL;

    pl->dyn_count = dyn_at;
    pl->dyn = (double *)SetH_OTF_HostAlloc(sizeof(double) * dyn_at, &pl->dyn_pinned);
    if (pl->dyn == NULL) goto fail;
    if (kind == 1) {
        pl->bounce_count = max_bounce;
        pl->bounce = (double *)SetH_OTF_HostAlloc(sizeof(double) * max_bounce, &pl->bounce_pinned);
        if (pl->bounce == NULL) goto fail;
    }
    pl->valid = 1;
    return pl;

fail:
    free(first);
    SetH_OTF_FreePlan(pl);
    return NULL;
}

/* the H (kind 0, from H) or DM (kind 1, from CDM) blocks of every pair of
   the rank into the plan, in the order of its batches */
static void SetH_OTF_PackBlocks(SetHOTFPlan *pl, double *****src)
{
    const int spin_count = pl->spin_count;
    size_t at = 0;
    int Mc_AN;

    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
        const int Gc_AN = M2G[Mc_AN];
        const int NO0 = Spe_Total_NO[WhatSpecies[Gc_AN]];
        int h_AN;

        for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
            const int NO1 = Spe_Total_NO[WhatSpecies[natn[Gc_AN][h_AN]]];
            int spin, i;

            for (spin = 0; spin < spin_count; spin++) {
                for (i = 0; i < NO0; i++) {
                    memcpy(pl->dyn + at, src[spin][Mc_AN][h_AN][i], sizeof(double) * (size_t)NO1);
                    at += (size_t)NO1;
                }
            }
        }
    }
}

/* the batch's descriptors, point lists and H/DM blocks into the arena */
static int SetH_OTF_Upload(const SetHOTFPlan *pl, const SetHOTFBatch *bt, unsigned char *arena)
{
    if (hipMemcpy(arena, bt->desc, bt->o_nc, hipMemcpyHostToDevice) != hipSuccess) return 0;
    if (0 < bt->bn && hipMemcpy(arena + bt->o_nc, SetH_OTF_Static.nc + bt->nc_first, sizeof(int) * bt->bn,
                                hipMemcpyHostToDevice) != hipSuccess) return 0;
    if (0 < bt->bh && hipMemcpy(arena + bt->o_dyn, pl->dyn + bt->dyn_first, sizeof(double) * bt->bh,
                                hipMemcpyHostToDevice) != hipSuccess) return 0;
    return 1;
}

/* the rank's potential grid, through a page-locked staging array */
static int SetH_OTF_UploadPotential(double *d_vgrid, int spin_count, size_t vgrid_len)
{
    const size_t count = (size_t)spin_count * vgrid_len;
    int spin;

    if (count == 0) return 1;
    if (SetH_OTF_Static.vstage_count < count) {
        SetH_OTF_HostFree(SetH_OTF_Static.vstage, SetH_OTF_Static.vstage_pinned,
                          sizeof(double) * SetH_OTF_Static.vstage_count);
        SetH_OTF_Static.vstage_count = 0;
        SetH_OTF_Static.vstage = (double *)SetH_OTF_HostAlloc(sizeof(double) * count, &SetH_OTF_Static.vstage_pinned);
        if (SetH_OTF_Static.vstage == NULL) return 0;
        SetH_OTF_Static.vstage_count = count;
    }
    for (spin = 0; spin < spin_count; spin++)
        memcpy(SetH_OTF_Static.vstage + (size_t)spin * vgrid_len, Vpot_Grid[spin], sizeof(double) * vgrid_len);
    return hipMemcpy(d_vgrid, SetH_OTF_Static.vstage, sizeof(double) * count, hipMemcpyHostToDevice) == hipSuccess;
}

/* 1 when this rank's matrix elements were computed on the device; 0 leaves
   them untouched for the table or CPU path.  Rank-local. */
static int Calc_MatrixElements_dVH_Vxc_VNA_OnTheFly(int Cnt_kind, int myid)
{
    const int spin_count = (SpinP_switch == 3) ? 4 : (SpinP_switch + 1);
    const size_t vgrid_len = (size_t)My_NumGridC;
    SetHamOTFState *st = &Set_Hamiltonian_OTF_State;
    SetHOTFPlan *pl;
    int b = 0, batch_count = 0, Mc_AN;
    size_t free_bytes = 0, hall_at = 0;
    unsigned char *arena = NULL;
    double *d_vgrid = NULL;
    int ok = 0;
    /* OPENMX_SETHAM_TIMING=1: a stdout line per call; OPENMX_BAND_PROFILE=1:
       the NVIDIA version's SETHOTFPROF line on stderr; either synchronizes
       the device at every stage so that its time lands in its own stage
       (plan: the plan lookup and the packing of the H blocks; alloc: the
       arena under the node's lock, including the wait for it) */
    const char *timing_env = getenv("OPENMX_SETHAM_TIMING");
    const int stdout_timing = (timing_env != NULL && atoi(timing_env) != 0);
    const int timing = stdout_timing || SetH_ProfileEnabled();
    double t_plan = 0.0, t_alloc = 0.0, t_pack = 0.0, t_eval = 0.0, t_kernel = 0.0, t_unpack = 0.0;
    double t_mark = 0.0, t_now = 0.0, t_start = 0.0;
#define SETH_OTF_TIME(acc) do { if (timing) { hipDeviceSynchronize(); dtime(&t_now); (acc) += t_now - t_mark; t_mark = t_now; } } while (0)

    if (Cnt_kind != 0 || Matomnum <= 0 || !Set_Hamiltonian_OTF_Enabled() ||
        !Set_Hamiltonian_MatrixElements_OpenMP_Enabled() || !Set_Hamiltonian_OTF_Rank_Selected()) return 0;
    if (!Set_Orbitals_Grid_OTF_Prepare() || !Set_Hamiltonian_OTF_Ensure()) return 0;
    if (timing) { dtime(&t_mark); t_start = t_mark; }
    pl = SetH_OTF_GetPlan(0, spin_count);
    if (pl == NULL) return 0;
    batch_count = pl->batch_count;

    /* the H blocks of every pair: the input of the batches and, in place,
       their results, written back to H only once all of them have
       completed (a failure leaves H untouched for the caller's path) */
    SetH_OTF_PackBlocks(pl, H);
    SETH_OTF_TIME(t_plan);

    /* the arena and the rank's potential grid behind it, one allocation */
    arena = (unsigned char *)SetH_OTF_LockedAlloc(
        pl->arena_bytes + SETH_OTF_ALIGN(sizeof(double) * (size_t)spin_count * (vgrid_len ? vgrid_len : 1)),
        (size_t)256 * 1024 * 1024, &free_bytes);
    if (arena == NULL) goto done;
    SETH_OTF_TIME(t_alloc);
    d_vgrid = (double *)(void *)(arena + pl->arena_bytes);
    if (!SetH_OTF_UploadPotential(d_vgrid, spin_count, vgrid_len)) goto done;
    SETH_OTF_TIME(t_pack);

    for (b = 0; b < batch_count; b++) {
        const SetHOTFBatch *bt = &pl->batch[b];

        if (!SetH_OTF_Upload(pl, bt, arena)) goto done;
        SETH_OTF_TIME(t_pack);
        if (OTF_Orbitals_Eval(NULL, bt->na, bt->max_sphere, (const OTF_Atom *)(void *)(arena + bt->o_a_items),
                              (const int *)(void *)(arena + bt->o_a_npts),
                              (const size_t *)(void *)(arena + bt->o_a_ioff),
                              (const size_t *)(void *)(arena + bt->o_a_ooff),
                              (const size_t *)(void *)(arena + bt->o_a_soff), NULL, st->d_gla, st->d_cla,
                              (float *)(void *)(arena + bt->o_o0)) != 0) goto done;
        if (OTF_Orbitals_Eval(NULL, bt->np, bt->max_nolg, (const OTF_Atom *)(void *)(arena + bt->o_p_items),
                              (const int *)(void *)(arena + bt->o_p_npts),
                              (const size_t *)(void *)(arena + bt->o_p_ioff),
                              (const size_t *)(void *)(arena + bt->o_p_ooff),
                              (const size_t *)(void *)(arena + bt->o_p_soff),
                              (const int *)(void *)(arena + bt->o_nc), st->d_gla, st->d_cla,
                              (float *)(void *)(arena + bt->o_o1)) != 0) goto done;
        SETH_OTF_TIME(t_eval);
        if (Set_Hamiltonian_Hip_OTF_VpotGather(bt->np, bt->max_nolg, spin_count, bt->bn,
                                               (const int *)(void *)(arena + bt->o_p_npts),
                                               (const size_t *)(void *)(arena + bt->o_p_ioff),
                                               (const size_t *)(void *)(arena + bt->o_p_soff),
                                               (const int *)(void *)(arena + bt->o_nc), st->d_mgla, d_vgrid, vgrid_len,
                                               GridVol, (double *)(void *)(arena + bt->o_vb)) != 0) goto done;
        if (Set_Hamiltonian_Hip_MatrixElements_Device(
                bt->np, spin_count, bt->max_output, bt->max_no, bt->max_nolg, bt->bh, bt->bn,
                (const int *)(void *)(arena + bt->o_no0), (const int *)(void *)(arena + bt->o_no1),
                (const int *)(void *)(arena + bt->o_p_npts), (const size_t *)(void *)(arena + bt->o_hoff),
                (const size_t *)(void *)(arena + bt->o_p_ioff), (const size_t *)(void *)(arena + bt->o_o0off),
                (const size_t *)(void *)(arena + bt->o_p_ooff), (const int *)(void *)(arena + bt->o_nc),
                (const double *)(void *)(arena + bt->o_vb), (const float *)(void *)(arena + bt->o_o0),
                (const float *)(void *)(arena + bt->o_o1), (double *)(void *)(arena + bt->o_dyn)) != 0) goto done;
        SETH_OTF_TIME(t_kernel);
        if (0 < bt->bh && hipMemcpy(pl->dyn + bt->dyn_first, arena + bt->o_dyn, sizeof(double) * bt->bh,
                                    hipMemcpyDeviceToHost) != hipSuccess) goto done;
        SETH_OTF_TIME(t_unpack);
    }

    /* every batch completed: the H blocks back */
    hall_at = 0;
    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
        const int Gc_AN = M2G[Mc_AN];
        const int NO0 = Spe_Total_NO[WhatSpecies[Gc_AN]];
        int h_AN;

        for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
            const int NO1 = Spe_Total_NO[WhatSpecies[natn[Gc_AN][h_AN]]];
            int spin, i;

            for (spin = 0; spin < spin_count; spin++) {
                for (i = 0; i < NO0; i++) {
                    memcpy(H[spin][Mc_AN][h_AN][i], pl->dyn + hall_at, sizeof(double) * (size_t)NO1);
                    hall_at += (size_t)NO1;
                }
            }
        }
    }
    SETH_OTF_TIME(t_unpack);
    ok = 1;
    if (stdout_timing) {
        printf("<Set_Hamiltonian> on-the-fly rank %d: plan %.3f alloc %.3f pack %.3f eval %.3f kernel %.3f "
               "unpack %.3f total %.3f s (%d batches)\n", myid, t_plan, t_alloc, t_pack, t_eval, t_kernel, t_unpack,
               t_now - t_start, batch_count);
        fflush(stdout);
    }
    if (SetH_ProfileEnabled()) {
        fprintf(stderr, "SETHOTFPROF id=%d batches=%d cpu_pairs=0 workspace_mib=%.1f "
                        "pack=%.3f eval=%.3f device=%.3f unpack=%.3f cpu=0.000 total=%.3f\n",
                myid, batch_count, (double)pl->arena_bytes / (1024.0 * 1024.0), t_plan + t_alloc + t_pack, t_eval,
                t_kernel, t_unpack, t_now - t_start);
    }
#undef SETH_OTF_TIME
    if (myid == Host_ID && 2 <= level_stdout) {
        static size_t last_workspace = 0;
        static int last_batches = -1;
        const size_t drift = last_workspace < pl->arena_bytes ? pl->arena_bytes - last_workspace
                                                              : last_workspace - pl->arena_bytes;
        if (last_batches != batch_count || last_workspace / 16 < drift || (3 <= level_stdout && drift != 0)) {
            printf("<Set_Hamiltonian> GPU on-the-fly orbital tiles: %d batch(es) of %.1f MiB on rank %d, %s "
                   "evaluation, %s plan\n", batch_count, (double)pl->arena_bytes / (1024.0 * 1024.0), myid,
                   OTF_Orbitals_DoubleFloat() ? "double-float" : "FP64",
                   SetH_OTF_PlanCacheEnabled() ? "cached" : "per-call");
            fflush(stdout);
            last_workspace = pl->arena_bytes;
            last_batches = batch_count;
        }
    }

done:
    if (!ok && myid == Host_ID) {
        if (2 <= level_stdout) {
            fprintf(stderr, "Set_Hamiltonian: rank %d: the on-the-fly matrix elements failed in batch %d of %d; "
                    "the table or CPU path takes over\n", myid, b + 1, batch_count);
            fflush(stderr);
        }
    }
    if (arena != NULL) hipFree(arena);   /* d_vgrid lies inside */
    (void)hipGetLastError();
    if (!SetH_OTF_PlanCacheEnabled()) SetH_OTF_FreePlan(&SetH_OTF_Plan[0]);
    return ok;
}

int Set_Hamiltonian_Hip_OTF_Density(int pair_count, int max_nolg, int spin_count, int max_no, const int *d_no0,
                                    const int *d_no1,
                                    const int *d_nolg, const size_t *d_dmoff, const size_t *d_noff,
                                    const size_t *d_o0off, const size_t *d_o1off, const size_t *d_denoff,
                                    const int *d_nc, const double *d_dm, const float *d_o0, const float *d_o1,
                                    size_t den_stride, double *d_den);

/* The density of Set_Density_Grid on the fly: Tmp_Den_Grid[s][Mc_AN][Nc]
   += sum over the atom's pairs and their overlap points of
   CDM[s][Mc_AN][h_AN] phi0 phi1, with the orbitals evaluated on the device
   batch by batch as for the matrix elements.  Returns 1 when this rank's
   density was computed (Tmp_Den_Grid holds it), 0 when the caller's CPU
   loop is to compute it (Tmp_Den_Grid untouched).  Rank-local.
   OPENMX_DENSITY_GRID_ONTHEFLY=0 turns it off. */
/* rank-local: whether this rank can take the on-the-fly density (the
   caller agrees on it across the ranks before the single-owner service) */
int Set_Density_Grid_OTF_Possible(int Cnt_kind)
{
    const char *enabled = getenv("OPENMX_DENSITY_GRID_ONTHEFLY");

    /* without contraction (Cnt_switch 0) the contracted orbitals are the
       primitive ones, so Cnt_kind 1 (the call before the forces) is the
       same density */
    if ((enabled != NULL && enabled[0] != '\0' && atoi(enabled) == 0) || (Cnt_kind != 0 && Cnt_kind != 1) ||
        Cnt_switch != 0 || !Set_Hamiltonian_OTF_Enabled() || !Set_Hamiltonian_MatrixElements_OpenMP_Enabled() ||
        !Set_Hamiltonian_OTF_Rank_Selected()) return 0;
    /* a rank without atoms has nothing to compute and must not hold the
       others back (the caller takes the minimum over the ranks) */
    if (Matomnum <= 0) return 1;
    return Set_Orbitals_Grid_OTF_Prepare() && Set_Hamiltonian_OTF_Ensure();
}

/* Plan before the density dispatch, as in the NVIDIA implementation.  This
   port batches whole atoms and already allows one atom to exceed the default
   workspace bound.  Use the very same cached arena size and allocation
   reserve as Run, rather than the NVIDIA pair-workspace accounting.  This
   check is rank-local; the locked allocation in Run remains authoritative
   when peer ranks allocate after this snapshot. */
int Set_Density_Grid_OTF_Prepare(int Cnt_kind, size_t *need_bytes, size_t *available_bytes)
{
    SetHOTFPlan *pl;
    size_t free_bytes = 0, total_bytes = 0;
    const size_t reserve = (size_t)256 * 1024 * 1024;

    *need_bytes = *available_bytes = 0;
    if (!Set_Density_Grid_OTF_Possible(Cnt_kind)) return 0;
    if (Matomnum <= 0) return 1;
    pl = SetH_OTF_GetPlan(1, SpinP_switch + 1);
    if (pl == NULL) return 0;
    *need_bytes = pl->arena_bytes;
    if (OpenMX_GpuMemGetInfo(&free_bytes, &total_bytes) != hipSuccess) return 0;
    *available_bytes = reserve < free_bytes ? free_bytes - reserve : 0;
    return pl->arena_bytes <= *available_bytes;
}

int Set_Density_Grid_OTF_Run(int Cnt_kind, double *****CDM, double ***Tmp_Den_Grid)
{
    const int spin_count = SpinP_switch + 1;
    SetHamOTFState *st = &Set_Hamiltonian_OTF_State;
    const char *enabled = getenv("OPENMX_DENSITY_GRID_ONTHEFLY");
    SetHOTFPlan *pl = NULL;
    int batch_count = 0, Mc_AN, b = 0, written = 0, myid, ok = 0;
    size_t free_bytes = 0;
    unsigned char *arena = NULL;
    const char *timing_env = getenv("OPENMX_SETHAM_TIMING");
    const int stdout_timing = (timing_env != NULL && atoi(timing_env) != 0);
    const int timing = stdout_timing || SetH_ProfileEnabled();
    double t_plan = 0.0, t_alloc = 0.0, t_pack = 0.0, t_eval = 0.0, t_kernel = 0.0, t_unpack = 0.0;
    double t_mark = 0.0, t_now = 0.0, t_start = 0.0;
#define SDG_OTF_TIME(acc) do { if (timing) { hipDeviceSynchronize(); dtime(&t_now); (acc) += t_now - t_mark; t_mark = t_now; } } while (0)

    MPI_Comm_rank(mpi_comm_level1, &myid);
    if (Matomnum <= 0) return 1;   /* nothing to add on this rank */
    if ((enabled != NULL && enabled[0] != '\0' && atoi(enabled) == 0) || (Cnt_kind != 0 && Cnt_kind != 1) ||
        Cnt_switch != 0 || !Set_Hamiltonian_OTF_Enabled() || !Set_Hamiltonian_MatrixElements_OpenMP_Enabled() ||
        !Set_Hamiltonian_OTF_Rank_Selected()) {
        static int noted = 0;
        if (!noted && myid == Host_ID && getenv("OPENMX_GPU_VERBOSE") != NULL) {
            printf("<Set_Density_Grid> on-the-fly density not enabled here (Cnt_kind %d, Cnt_switch %d, selected %d)\n",
                   Cnt_kind, Cnt_switch, Set_Hamiltonian_OTF_Rank_Selected());
            fflush(stdout);
            noted = 1;
        }
        return 0;
    }
    if (!Set_Orbitals_Grid_OTF_Prepare() || !Set_Hamiltonian_OTF_Ensure()) return 0;
    if (timing) { dtime(&t_mark); t_start = t_mark; }
    pl = SetH_OTF_GetPlan(1, spin_count);
    if (pl == NULL) return 0;
    batch_count = pl->batch_count;

    /* the DM blocks of every pair, as the H blocks of the matrix elements */
    SetH_OTF_PackBlocks(pl, CDM);
    SDG_OTF_TIME(t_plan);

    arena = (unsigned char *)SetH_OTF_LockedAlloc(pl->arena_bytes, (size_t)256 * 1024 * 1024, &free_bytes);
    if (arena == NULL) {
        static int warned = 0;

        if (!warned && 3 <= level_stdout) {
            printf("<Set_Density_Grid> rank %d: no room for the on-the-fly density (%.1f MiB free, %.1f MiB "
                   "needed with the reserve); CPU loop\n", myid, (double)free_bytes / 1048576.0,
                   (double)(pl->arena_bytes + (size_t)256 * 1024 * 1024) / 1048576.0);
            fflush(stdout);
            warned = 1;
        }
        goto done;
    }
    SDG_OTF_TIME(t_alloc);

    for (b = 0; b < batch_count; b++) {
        const SetHOTFBatch *bt = &pl->batch[b];

        if (!SetH_OTF_Upload(pl, bt, arena) ||
            hipMemset(arena + bt->o_den, 0, (size_t)spin_count * bt->sph * sizeof(double)) != hipSuccess) goto done;
        SDG_OTF_TIME(t_pack);
        if (OTF_Orbitals_Eval(NULL, bt->na, bt->max_sphere, (const OTF_Atom *)(void *)(arena + bt->o_a_items),
                              (const int *)(void *)(arena + bt->o_a_npts),
                              (const size_t *)(void *)(arena + bt->o_a_ioff),
                              (const size_t *)(void *)(arena + bt->o_a_ooff),
                              (const size_t *)(void *)(arena + bt->o_a_soff), NULL, st->d_gla, st->d_cla,
                              (float *)(void *)(arena + bt->o_o0)) != 0) goto done;
        if (OTF_Orbitals_Eval(NULL, bt->np, bt->max_nolg, (const OTF_Atom *)(void *)(arena + bt->o_p_items),
                              (const int *)(void *)(arena + bt->o_p_npts),
                              (const size_t *)(void *)(arena + bt->o_p_ioff),
                              (const size_t *)(void *)(arena + bt->o_p_ooff),
                              (const size_t *)(void *)(arena + bt->o_p_soff),
                              (const int *)(void *)(arena + bt->o_nc), st->d_gla, st->d_cla,
                              (float *)(void *)(arena + bt->o_o1)) != 0) goto done;
        SDG_OTF_TIME(t_eval);
        if (Set_Hamiltonian_Hip_OTF_Density(bt->np, bt->max_nolg, spin_count, bt->max_no,
                                            (const int *)(void *)(arena + bt->o_no0),
                                            (const int *)(void *)(arena + bt->o_no1),
                                            (const int *)(void *)(arena + bt->o_p_npts),
                                            (const size_t *)(void *)(arena + bt->o_hoff),
                                            (const size_t *)(void *)(arena + bt->o_p_ioff),
                                            (const size_t *)(void *)(arena + bt->o_o0off),
                                            (const size_t *)(void *)(arena + bt->o_p_ooff),
                                            (const size_t *)(void *)(arena + bt->o_denoff),
                                            (const int *)(void *)(arena + bt->o_nc),
                                            (const double *)(void *)(arena + bt->o_dyn),
                                            (const float *)(void *)(arena + bt->o_o0),
                                            (const float *)(void *)(arena + bt->o_o1), bt->sph,
                                            (double *)(void *)(arena + bt->o_den)) != 0) goto done;
        SDG_OTF_TIME(t_kernel);
        if (hipMemcpy(pl->bounce, arena + bt->o_den, sizeof(double) * (size_t)spin_count * bt->sph,
                      hipMemcpyDeviceToHost) != hipSuccess) goto done;
        /* Set_Density_Grid allocates Tmp_Den_Grid cleared and nothing else
           writes it before this path, so the batch's density goes in as it
           is (a failure clears what was written) */
        written = b + 1;
        for (int sp = 0; sp < spin_count; sp++) {
            for (Mc_AN = bt->a0; Mc_AN < bt->a1; Mc_AN++) {
                memcpy(Tmp_Den_Grid[sp][Mc_AN],
                       pl->bounce + (size_t)sp * bt->sph + (st->atom_sph[Mc_AN] - bt->sph_first),
                       sizeof(double) * (size_t)GridN_Atom[M2G[Mc_AN]]);
            }
        }
        SDG_OTF_TIME(t_unpack);
    }
    ok = 1;
    if (stdout_timing) {
        dtime(&t_now);
        printf("<Set_Density_Grid> on-the-fly rank %d: plan %.3f alloc %.3f pack %.3f eval %.3f kernel %.3f "
               "unpack %.3f total %.3f s (%d batches)\n", myid, t_plan, t_alloc, t_pack, t_eval, t_kernel, t_unpack,
               t_now - t_start, batch_count);
        fflush(stdout);
    }
    if (SetH_ProfileEnabled()) {
        dtime(&t_now);
        /* mode as the NVIDIA version prints it: 2 double-float, 0 FP64 */
        fprintf(stderr, "SETHOTFDENPROF id=%d batches=%d workspace_mib=%.1f mode=%d pack=%.3f eval=%.3f device=%.3f "
                        "total=%.3f\n", myid, batch_count, (double)pl->arena_bytes / (1024.0 * 1024.0),
                OTF_Orbitals_DoubleFloat() ? 2 : 0, t_plan + t_alloc + t_pack, t_eval, t_kernel + t_unpack,
                t_now - t_start);
    }
#undef SDG_OTF_TIME
    if (myid == Host_ID && 2 <= level_stdout) {
        static size_t last_workspace = 0;
        static int last_batches = -1;
        const size_t drift = last_workspace < pl->arena_bytes ? pl->arena_bytes - last_workspace
                                                              : last_workspace - pl->arena_bytes;
        if (last_batches != batch_count || last_workspace / 16 < drift || (3 <= level_stdout && drift != 0)) {
            printf("<Set_Density_Grid> GPU on-the-fly orbital tiles: %d batch(es) of %.1f MiB on rank %d\n",
                   batch_count, (double)pl->arena_bytes / (1024.0 * 1024.0), myid);
            fflush(stdout);
            last_workspace = pl->arena_bytes;
            last_batches = batch_count;
        }
    }

done:
    if (!ok) {
        int bw;

        for (bw = 0; bw < written; bw++) {
            const SetHOTFBatch *bt = &pl->batch[bw];
            for (int sp = 0; sp < spin_count; sp++)
                for (Mc_AN = bt->a0; Mc_AN < bt->a1; Mc_AN++)
                    memset(Tmp_Den_Grid[sp][Mc_AN], 0, sizeof(double) * (size_t)GridN_Atom[M2G[Mc_AN]]);
        }
        if (arena != NULL && myid == Host_ID) {
            if (3 <= level_stdout) {
                fprintf(stderr, "Set_Density_Grid: rank %d: the on-the-fly density failed in batch %d of %d; CPU loop\n",
                        myid, b + 1, batch_count);
                fflush(stderr);
            }
        }
    }
    if (arena != NULL) hipFree(arena);
    (void)hipGetLastError();
    if (!SetH_OTF_PlanCacheEnabled()) SetH_OTF_FreePlan(&SetH_OTF_Plan[1]);
    return ok;
}

static void Set_Hamiltonian_Stream_PairCPU(int cnt_kind,int mc,int h,int rank)
{
    const int gc=M2G[mc],gh=natn[gc][h],mh=F_G2M[gh];
    const int n0=cnt_kind?Spe_Total_CNO[WhatSpecies[gc]]:Spe_Total_NO[WhatSpecies[gc]];
    const int n1=cnt_kind?Spe_Total_CNO[WhatSpecies[gh]]:Spe_Total_NO[WhatSpecies[gh]];
    double *****target=cnt_kind?CntH:H;
    for(int spin=0;spin<=SpinP_switch;spin++)for(int i=0;i<n0;i++)for(int j=0;j<n1;j++) {
        double sum=0;
        for(int k=0;k<NumOLG[mc][h];k++) {
            int nc=GListTAtoms1[mc][h][k],nh=GListTAtoms2[mc][h][k],mn=MGridListAtom[mc][nc];
            const Type_Orbs_Grid *right=G2ID[gh]==rank?Orbs_Grid[mh][nh]:Orbs_Grid_FNAN[mc][h][k];
            sum+=((GridVol*Vpot_Grid[spin][mn])*(double)Orbs_Grid[mc][nc][i])*(double)right[j];
        }
        target[spin][mc][h][i][j]+=sum;
    }
}
typedef struct {int n0,n1,points;} SetHCapacityPair;
typedef struct {int spins,max_no;double volume;} SetHCapacityParameters;
static void SetH_capacity_kernel(size_t first,size_t n,const void *parameters,void *const v[],int gpu)
{
    (void)first;const SetHCapacityParameters p=*(const SetHCapacityParameters*)parameters;
    const int spins=p.spins,max_no=p.max_no;const double volume=p.volume;
    const SetHCapacityPair *pairs=v[0];const size_t *pa=v[1],*pb=v[2],*pw=v[3];
    const double *a=openmx_gpu_task_ragged_data(v[1],n),*b=openmx_gpu_task_ragged_data(v[2],n),*w=openmx_gpu_task_ragged_data(v[3],n);
    double *out=v[4];
#pragma omp target teams distribute parallel for is_device_ptr(pairs,pa,pb,pw,a,b,w,out) if(gpu)
    for(size_t pair=0;pair<n;pair++) {
        const int n0=pairs[pair].n0,n1=pairs[pair].n1,points=pairs[pair].points;

        for(int e=0;e<spins*max_no*max_no;e++) {
            int spin=e/(max_no*max_no),ij=e%(max_no*max_no),i=ij/max_no,j=ij%max_no;double sum=0;
            if(i<n0 && j<n1) {

                for(int k=0;k<points;k++) {
                    double weight=volume*w[pw[pair]+(size_t)k*spins+spin];
                    sum+=(weight*a[pa[pair]+(size_t)k*n0+i])*b[pb[pair]+(size_t)k*n1+j];
                }
            }
            out[pair*(size_t)spins*max_no*max_no+e]=sum;
        }
    }
}

/* Enter once per quadrature phase on all ranks. Only ranks whose existing
   owner/streamed path declined pack work; completed ranks can lend GPUs.
   Bounded host tiles avoid duplicating a full orbital cache on the host. */
static int SetH_capacity_recover(int cnt_kind,int local_done,int rank)
{
    const char *gpu=getenv("OPENMX_GPU");
    int enabled=scf_eigen_lib_flag==GPUSOLVER && !(gpu && atoi(gpu)==0) && Set_Hamiltonian_MatrixElements_OpenMP_Enabled();
    int pending=enabled && !local_done,any=0;
    MPI_Allreduce(&pending,&any,1,MPI_INT,MPI_MAX,mpi_comm_level1);
    if(!any)return local_done;
    int count=0,*mc_table=NULL,*h_table=NULL,ready=1;
    if(pending) {
        for(int mc=1;mc<=Matomnum;mc++)count+=FNAN[M2G[mc]]+1;
        mc_table=malloc((size_t)(count?count:1)*sizeof(int));h_table=malloc((size_t)(count?count:1)*sizeof(int));
        ready=mc_table && h_table;
        if(ready){int k=0;for(int mc=1;mc<=Matomnum;mc++)for(int h=0;h<=FNAN[M2G[mc]];h++){mc_table[k]=mc;h_table[k++]=h;}}
        else count=0;
    }
    SetHCapacityParameters parameters={SpinP_switch+1,0,GridVol};
    for(int s=0;s<SpeciesNum;s++){int no=cnt_kind?Spe_Total_CNO[s]:Spe_Total_NO[s];if(parameters.max_no<no)parameters.max_no=no;}
    const int spins=parameters.spins,max_no=parameters.max_no;
    const size_t out_stride=(size_t)spins*max_no*max_no;
    int first=0;
    for(;;) {
        int remaining=first<count,work_left=0;
        MPI_Allreduce(&remaining,&work_left,1,MPI_INT,MPI_MAX,mpi_comm_level1);if(!work_left)break;
        enum {MAX_PAIRS=256};SetHCapacityPair pair[MAX_PAIRS];
        size_t offsets[3][MAX_PAIRS+1]={{0}};
        size_t budget=67108864,bytes=0;int n=0;
        unsigned long long available=openmx_host_mem_available();int sharing=openmx_gpu_local_size_noncollective();if(sharing<1)sharing=1;
        if(available && budget>available/(4ULL*sharing))budget=(size_t)(available/(4ULL*sharing));
        while(first+n<count && n<MAX_PAIRS) {
            int mc=mc_table[first+n],h=h_table[first+n],gc=M2G[mc],gh=natn[gc][h];
            int n0=cnt_kind?Spe_Total_CNO[WhatSpecies[gc]]:Spe_Total_NO[WhatSpecies[gc]];
            int n1=cnt_kind?Spe_Total_CNO[WhatSpecies[gh]]:Spe_Total_NO[WhatSpecies[gh]],points=NumOLG[mc][h];
            size_t added=((size_t)points*((size_t)n0+n1+spins)+out_stride)*sizeof(double);
            if(n && (added>budget || bytes>budget-added))break;
            pair[n]=(SetHCapacityPair){n0,n1,points};
            offsets[0][n+1]=offsets[0][n]+(size_t)points*n0;
            offsets[1][n+1]=offsets[1][n]+(size_t)points*n1;
            offsets[2][n+1]=offsets[2][n]+(size_t)points*spins;
            n++;bytes+=added;
        }
        double *a=NULL,*b=NULL,*weights=NULL,*out=NULL;
        if(n) {
            a=malloc((offsets[0][n]?offsets[0][n]:1)*sizeof(double));
            b=malloc((offsets[1][n]?offsets[1][n]:1)*sizeof(double));
            weights=malloc((offsets[2][n]?offsets[2][n]:1)*sizeof(double));out=malloc((size_t)n*out_stride*sizeof(double));
            if(!a || !b || !weights || !out) {
                /* Earlier tiles have already been added: retry just these
                   pairs on the CPU, never add the full Hamiltonian twice. */
                for(int i=0;i<n;i++)Set_Hamiltonian_Stream_PairCPU(cnt_kind,mc_table[first+i],h_table[first+i],rank);
                first+=n;n=0;
            }
        }
        for(int i=0;i<n;i++) {
            int mc=mc_table[first+i],h=h_table[first+i],gh=natn[M2G[mc]][h],mh=F_G2M[gh];
            for(int k=0;k<pair[i].points;k++) {
                int nc=GListTAtoms1[mc][h][k],nh=GListTAtoms2[mc][h][k],mn=MGridListAtom[mc][nc];
                const Type_Orbs_Grid *left=Orbs_Grid[mc][nc],*right=G2ID[gh]==rank?Orbs_Grid[mh][nh]:Orbs_Grid_FNAN[mc][h][k];
                for(int j=0;j<pair[i].n0;j++)a[offsets[0][i]+(size_t)k*pair[i].n0+j]=left[j];
                for(int j=0;j<pair[i].n1;j++)b[offsets[1][i]+(size_t)k*pair[i].n1+j]=right[j];
                for(int s=0;s<spins;s++)weights[offsets[2][i]+(size_t)k*spins+s]=Vpot_Grid[s][mn];
            }
        }
        OpenMXGpuTask task={0};task.count=n;task.inputs=4;task.outputs=1;task.kernel=SetH_capacity_kernel;
        task.parameters=&parameters;task.parameter_bytes=sizeof(parameters);
        task.view[0]=(OpenMXGpuTaskView){pair,0,sizeof(pair[0])};
        task.view[1]=(OpenMXGpuTaskView){a,0,0,offsets[0],sizeof(double)};
        task.view[2]=(OpenMXGpuTaskView){b,0,0,offsets[1],sizeof(double)};
        task.view[3]=(OpenMXGpuTaskView){weights,0,0,offsets[2],sizeof(double)};
        task.view[4]=(OpenMXGpuTaskView){out,0,out_stride*sizeof(double)};
        openmx_gpu_task_execute(mpi_comm_level1,&task,enabled,"Hamiltonian-quadrature-capacity");
        for(int i=0;i<n;i++)for(int s=0;s<spins;s++)for(int row=0;row<pair[i].n0;row++)for(int col=0;col<pair[i].n1;col++) {
            double *****target=cnt_kind?CntH:H;
            target[s][mc_table[first+i]][h_table[first+i]][row][col]+=out[(size_t)i*out_stride+((size_t)s*max_no+row)*max_no+col];
        }
        free(a);free(b);free(weights);free(out);first+=n;
    }
    free(mc_table);free(h_table);return local_done || (pending && ready);
}

void Calc_MatrixElements_dVH_Vxc_VNA(int Cnt_kind)
{
    int myid,use_tables=0,local_done=0;
    Set_Hamiltonian_MatrixElements_GPU_Used=0;
    MPI_Comm_rank(mpi_comm_level1,&myid);
    if(!(openmx_gpu_capacity_test_recovery() || openmx_gpu_workers_requested())) {
        use_tables=Set_Hamiltonian_MatrixElements_Use_OpenMP(Cnt_kind,myid);
        if(use_tables && (Set_Hamiltonian_ME_CacheReusable(Cnt_kind) || Set_Hamiltonian_ME_KeepDeviceResident())) {
            Calc_MatrixElements_dVH_Vxc_VNA_OpenMP(Cnt_kind);local_done=1;
        } else if(Calc_MatrixElements_dVH_Vxc_VNA_OnTheFly(Cnt_kind,myid)) {
            if(Matomnum>0)Set_Hamiltonian_MatrixElements_GPU_Used=1;
            local_done=1;
        } else if(use_tables) {
            Calc_MatrixElements_dVH_Vxc_VNA_OpenMP(Cnt_kind);local_done=1;
        }
    }
    /* All ranks rendezvous after the existing MI300A/Radeon fast paths.
       Completed ranks can lend capacity without adding their matrix twice. */
    if(!SetH_capacity_recover(Cnt_kind,local_done,myid))
        Calc_MatrixElements_dVH_Vxc_VNA_CPU(Cnt_kind);
}

static size_t Set_Hamiltonian_Base_OpenMP_DeviceBytes(int SCF_iter, int myid)
{
    int Mc_AN, h_AN;
    int spin_count, pair_count;
    int add_hub, add_hch, use_vna;
    size_t total_mat, total_h, bytes;

    spin_count = (SpinP_switch == 3) ? 4 : (SpinP_switch + 1);
    add_hub = ((Hub_U_switch == 1 || 1 <= Constraint_NCS_switch) && F_U_flag == 1 && 2 <= SCF_iter);
    add_hch = (core_hole_state_flag == 1);
    use_vna = (ProExpn_VNA != 0);

    pair_count = 0;
    total_mat = 0;
    total_h = 0;
    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
        int Gc_AN = M2G[Mc_AN];
        int Cwan = WhatSpecies[Gc_AN];

        for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
            int Gh_AN = natn[Gc_AN][h_AN];
            int Hwan = WhatSpecies[Gh_AN];
            size_t mat_size = Set_Hamiltonian_checked_mul((size_t)Spe_Total_NO[Cwan],
                                                          (size_t)Spe_Total_NO[Hwan],
                                                          "base matrix size", myid);
            size_t h_size = Set_Hamiltonian_checked_mul((size_t)spin_count, mat_size,
                                                        "base Hamiltonian size", myid);

            total_mat = Set_Hamiltonian_checked_add(total_mat, mat_size, "base total matrix", myid);
            total_h = Set_Hamiltonian_checked_add(total_h, h_size, "base total Hamiltonian", myid);
            pair_count++;
        }
    }

    bytes = 0;
    Set_Hamiltonian_add_array_bytes(&bytes, (size_t)pair_count, sizeof(int), "base pair_NO0", myid);
    Set_Hamiltonian_add_array_bytes(&bytes, (size_t)pair_count, sizeof(int), "base pair_NO1", myid);
    Set_Hamiltonian_add_array_bytes(&bytes, (size_t)pair_count, sizeof(size_t), "base pair_mat_offset", myid);
    Set_Hamiltonian_add_array_bytes(&bytes, (size_t)pair_count, sizeof(size_t), "base pair_h_offset", myid);
    Set_Hamiltonian_add_array_bytes(&bytes, total_h, sizeof(double), "base hbuf", myid);
    Set_Hamiltonian_add_array_bytes(&bytes, total_mat, sizeof(double), "base h0buf", myid);
    Set_Hamiltonian_add_array_bytes(&bytes, total_h, sizeof(double), "base hnlbuf", myid);
    Set_Hamiltonian_add_array_bytes(&bytes, use_vna ? total_mat : 1u, sizeof(double), "base hvnabuf", myid);
    Set_Hamiltonian_add_array_bytes(&bytes, add_hub ? total_h : 1u, sizeof(double), "base hhubbuf", myid);
    Set_Hamiltonian_add_array_bytes(&bytes, add_hch ? total_h : 1u, sizeof(double), "base hchbuf", myid);

    return bytes;
}

static size_t Set_Hamiltonian_MatrixElements_OpenMP_DeviceBytes(int Cnt_kind, int myid)
{
    int Mc_AN, h_AN;
    int spin_count, pair_count;
    size_t total_h, total_nolg, total_orbs0, total_orbs1, bytes;

    if (Cnt_kind != 0 && Cnt_kind != 1) {
        Set_Hamiltonian_abort("Calc_MatrixElements_dVH_Vxc_VNA_OpenMP", "Cnt_kind must be 0 or 1", myid);
    }

    spin_count = (SpinP_switch == 3) ? 4 : (SpinP_switch + 1);

    /* with the previous call's tables still resident only the potential and
       the output are uploaded */
    if (Set_Hamiltonian_ME_CacheReusable(Cnt_kind)) {
        const SetHamiltonianMETables *t = &Set_Hamiltonian_ME_Tables.t;

        bytes = 0;
        Set_Hamiltonian_add_array_bytes(&bytes, t->total_h, sizeof(double), "matrix-elements hbuf", myid);
        Set_Hamiltonian_add_array_bytes(&bytes,
                                        Set_Hamiltonian_checked_mul((size_t)spin_count, t->total_nolg,
                                                                    "matrix-elements vpotbuf", myid),
                                        sizeof(double), "matrix-elements vpotbuf", myid);
        return bytes;
    }

    pair_count = 0;
    total_h = 0;
    total_nolg = 0;
    total_orbs0 = 0;
    total_orbs1 = 0;
    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
        int Gc_AN = M2G[Mc_AN];
        int Cwan = WhatSpecies[Gc_AN];

        for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
            int Gh_AN = natn[Gc_AN][h_AN];
            int Hwan = WhatSpecies[Gh_AN];
            int NOLG = NumOLG[Mc_AN][h_AN];
            int NO0, NO1;
            size_t mat_size, h_size, orbs0_size, orbs1_size;

            if (Cnt_kind == 0) {
                NO0 = Spe_Total_NO[Cwan];
                NO1 = Spe_Total_NO[Hwan];
            } else {
                NO0 = Spe_Total_CNO[Cwan];
                NO1 = Spe_Total_CNO[Hwan];
            }

            mat_size = Set_Hamiltonian_checked_mul((size_t)NO0, (size_t)NO1,
                                                   "matrix-elements matrix size", myid);
            h_size = Set_Hamiltonian_checked_mul((size_t)spin_count, mat_size,
                                                 "matrix-elements Hamiltonian size", myid);
            orbs0_size = Set_Hamiltonian_checked_mul((size_t)NOLG, (size_t)NO0,
                                                     "matrix-elements orbs0 size", myid);
            orbs1_size = Set_Hamiltonian_checked_mul((size_t)NOLG, (size_t)NO1,
                                                     "matrix-elements orbs1 size", myid);

            total_h = Set_Hamiltonian_checked_add(total_h, h_size, "matrix-elements total Hamiltonian", myid);
            total_nolg = Set_Hamiltonian_checked_add(total_nolg, (size_t)NOLG,
                                                     "matrix-elements total NOLG", myid);
            total_orbs1 = Set_Hamiltonian_checked_add(total_orbs1, orbs1_size,
                                                      "matrix-elements total orbs1", myid);
            (void)orbs0_size;
            pair_count++;
        }
        /* orbs0: the central atom's sphere once */
        total_orbs0 = Set_Hamiltonian_checked_add(
            total_orbs0,
            Set_Hamiltonian_checked_mul((size_t)GridN_Atom[Gc_AN],
                                        (size_t)((Cnt_kind == 0) ? Spe_Total_NO[Cwan] : Spe_Total_CNO[Cwan]),
                                        "matrix-elements orbs0 size", myid),
            "matrix-elements total orbs0", myid);
    }

    bytes = 0;
    Set_Hamiltonian_add_array_bytes(&bytes, (size_t)pair_count, sizeof(int), "matrix-elements pair_NO0", myid);
    Set_Hamiltonian_add_array_bytes(&bytes, (size_t)pair_count, sizeof(int), "matrix-elements pair_NO1", myid);
    Set_Hamiltonian_add_array_bytes(&bytes, (size_t)pair_count, sizeof(int), "matrix-elements pair_NOLG", myid);
    Set_Hamiltonian_add_array_bytes(&bytes, (size_t)pair_count, sizeof(size_t),
                                    "matrix-elements pair_h_offset", myid);
    Set_Hamiltonian_add_array_bytes(&bytes, (size_t)pair_count, sizeof(size_t),
                                    "matrix-elements pair_nolg_offset", myid);
    Set_Hamiltonian_add_array_bytes(&bytes, (size_t)pair_count, sizeof(size_t),
                                    "matrix-elements pair_orbs0_offset", myid);
    Set_Hamiltonian_add_array_bytes(&bytes, (size_t)pair_count, sizeof(size_t),
                                    "matrix-elements pair_orbs1_offset", myid);
    Set_Hamiltonian_add_array_bytes(&bytes, total_h, sizeof(double), "matrix-elements hbuf", myid);
    Set_Hamiltonian_add_array_bytes(&bytes,
                                    Set_Hamiltonian_checked_mul((size_t)spin_count, total_nolg,
                                                                "matrix-elements vpotbuf", myid),
                                    sizeof(double), "matrix-elements vpotbuf", myid);
    Set_Hamiltonian_add_array_bytes(&bytes, total_nolg, sizeof(int), "matrix-elements nolg_Nc", myid);
    Set_Hamiltonian_add_array_bytes(&bytes, total_orbs0, sizeof(Type_Orbs_Grid), "matrix-elements orbs0buf", myid);
    Set_Hamiltonian_add_array_bytes(&bytes, total_orbs1, sizeof(Type_Orbs_Grid), "matrix-elements orbs1buf", myid);

    return bytes;
}

typedef struct {int spins,nc,hub,hch,vna;double fkin,fnl,fvna;} SetHBaseCapacityParameters;
static void SetH_base_capacity_kernel(size_t first,size_t n,const void *parameters,void *const v[],int gpu)
{
    (void)first;const SetHBaseCapacityParameters p=*(const SetHBaseCapacityParameters*)parameters;
    const int spins=p.spins,nc=p.nc,hub=p.hub,hch=p.hch,vna=p.vna;
    const double fkin=p.fkin,fnl=p.fnl,fvna=p.fvna;
    const double *h0=v[0],*hnl=v[1],*hvna=v[2],*hhub=v[3],*hchv=v[4];double *out=v[5];
#pragma omp target teams distribute parallel for is_device_ptr(h0,hnl,hvna,hhub,hchv,out) if(gpu)
    for(size_t i=0;i<n;i++)for(int spin=0;spin<spins;spin++) {
        size_t at=i*spins+spin;double value=0;
        if(!nc || spin<2)value=fkin*h0[i]+(vna?fvna*hvna[i]:0)+fnl*hnl[at];
        else if(spin==2)value=fnl*hnl[at];
        if(!nc || spin<3){if(hub)value+=hhub[at];if(hch)value+=hchv[at];}
        out[at]=value;
    }
}

static void Set_Hamiltonian_Base_OpenMP(int SCF_iter, double *****H0, double *****HNL, double *****H)
{
    int Mc_AN, h_AN, Gc_AN, Gh_AN, Cwan, Hwan;
    int numprocs, myid;
    int spin_count, pair_count, pair;
    int add_hub, add_hch, use_vna;
    double f_kin, f_nl, f_vna;
    int *pair_Mc_AN, *pair_h_AN, *pair_NO0, *pair_NO1;
    size_t *pair_mat_offset, *pair_h_offset;
    size_t total_mat, total_h;
    double *hbuf, *h0buf, *hnlbuf, *hvnabuf, *hhubbuf, *hchbuf;

    MPI_Comm_size(mpi_comm_level1, &numprocs);
    MPI_Comm_rank(mpi_comm_level1, &myid);

    spin_count = (SpinP_switch == 3) ? 4 : (SpinP_switch + 1);
    add_hub = ((Hub_U_switch == 1 || 1 <= Constraint_NCS_switch) && F_U_flag == 1 && 2 <= SCF_iter);
    add_hch = (core_hole_state_flag == 1);
    use_vna = (ProExpn_VNA != 0);
    f_kin = (double)F_Kin_flag;
    f_nl = (double)F_NL_flag;
    f_vna = (double)F_VNA_flag;

    pair_count = 0;
    total_mat = 0;
    total_h = 0;
    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
        Gc_AN = M2G[Mc_AN];
        Cwan = WhatSpecies[Gc_AN];

        for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
            int NO0, NO1;

            Gh_AN = natn[Gc_AN][h_AN];
            Hwan = WhatSpecies[Gh_AN];
            NO0 = Spe_Total_NO[Cwan];
            NO1 = Spe_Total_NO[Hwan];
            total_mat += (size_t)NO0 * (size_t)NO1;
            total_h += (size_t)spin_count * (size_t)NO0 * (size_t)NO1;
            pair_count++;
        }
    }

    pair_Mc_AN = (int *)Set_Hamiltonian_malloc(sizeof(int) * (size_t)pair_count, "openmp pair_Mc_AN", myid);
    pair_h_AN = (int *)Set_Hamiltonian_malloc(sizeof(int) * (size_t)pair_count, "openmp pair_h_AN", myid);
    pair_NO0 = (int *)Set_Hamiltonian_malloc(sizeof(int) * (size_t)pair_count, "openmp pair_NO0", myid);
    pair_NO1 = (int *)Set_Hamiltonian_malloc(sizeof(int) * (size_t)pair_count, "openmp pair_NO1", myid);
    pair_mat_offset =
        (size_t *)Set_Hamiltonian_malloc(sizeof(size_t) * (size_t)pair_count, "openmp pair_mat_offset", myid);
    pair_h_offset =
        (size_t *)Set_Hamiltonian_malloc(sizeof(size_t) * (size_t)pair_count, "openmp pair_h_offset", myid);
    hbuf = (double *)Set_Hamiltonian_malloc(sizeof(double) * total_h, "openmp base hbuf", myid);
    h0buf = (double *)Set_Hamiltonian_malloc(sizeof(double) * total_mat, "openmp base h0buf", myid);
    hnlbuf = (double *)Set_Hamiltonian_malloc(sizeof(double) * total_h, "openmp base hnlbuf", myid);
    hvnabuf =
        (double *)Set_Hamiltonian_malloc(sizeof(double) * (use_vna ? total_mat : 1), "openmp base hvnabuf", myid);
    hhubbuf = (double *)Set_Hamiltonian_malloc(sizeof(double) * (add_hub ? total_h : 1), "openmp base hhubbuf", myid);
    hchbuf = (double *)Set_Hamiltonian_malloc(sizeof(double) * (add_hch ? total_h : 1), "openmp base hchbuf", myid);

    if (!use_vna) hvnabuf[0]=0;
    if (!add_hub) hhubbuf[0]=0;
    if (!add_hch) hchbuf[0]=0;
    pair = 0;
    total_mat = 0;
    total_h = 0;
    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
        Gc_AN = M2G[Mc_AN];
        Cwan = WhatSpecies[Gc_AN];

        for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
            int NO0, NO1, spin, i, j;
            size_t mat_size;

            Gh_AN = natn[Gc_AN][h_AN];
            Hwan = WhatSpecies[Gh_AN];
            NO0 = Spe_Total_NO[Cwan];
            NO1 = Spe_Total_NO[Hwan];
            mat_size = (size_t)NO0 * (size_t)NO1;

            pair_Mc_AN[pair] = Mc_AN;
            pair_h_AN[pair] = h_AN;
            pair_NO0[pair] = NO0;
            pair_NO1[pair] = NO1;
            pair_mat_offset[pair] = total_mat;
            pair_h_offset[pair] = total_h;

            for (i = 0; i < NO0; i++) {
                for (j = 0; j < NO1; j++) {
                    size_t ij = (size_t)i * (size_t)NO1 + (size_t)j;
                    size_t mij = total_mat + ij;
                    h0buf[mij] = H0[0][Mc_AN][h_AN][i][j];
                    if (use_vna) {
                        hvnabuf[mij] = HVNA[Mc_AN][h_AN][i][j];
                    }
                    for (spin = 0; spin < spin_count; spin++) {
                        size_t idx = (total_mat + ij) * spin_count + spin;
                        hnlbuf[idx] = (SpinP_switch == 3 && spin == 3) ? 0.0 : HNL[spin][Mc_AN][h_AN][i][j];
                        if (add_hub) {
                            hhubbuf[idx] = (SpinP_switch == 3 && spin == 3)?0:H_Hub[spin][Mc_AN][h_AN][i][j];
                        }
                        if (add_hch) {
                            hchbuf[idx] = (SpinP_switch == 3 && spin == 3)?0:HCH[spin][Mc_AN][h_AN][i][j];
                        }
                    }
                }
            }

            total_mat += mat_size;
            total_h += (size_t)spin_count * mat_size;
            pair++;
        }
    }

    SetHBaseCapacityParameters parameters={spin_count,SpinP_switch==3,add_hub,add_hch,use_vna,f_kin,f_nl,f_vna};
    OpenMXGpuTask task={0};task.count=total_mat;task.inputs=5;task.outputs=1;
    task.kernel=SetH_base_capacity_kernel;task.parameters=&parameters;task.parameter_bytes=sizeof(parameters);
    task.view[0]=(OpenMXGpuTaskView){h0buf,0,sizeof(double)};
    task.view[1]=(OpenMXGpuTaskView){hnlbuf,0,(size_t)spin_count*sizeof(double)};
    task.view[2]=(OpenMXGpuTaskView){hvnabuf,sizeof(double),use_vna?sizeof(double):0};
    task.view[3]=(OpenMXGpuTaskView){hhubbuf,sizeof(double),add_hub?(size_t)spin_count*sizeof(double):0};
    task.view[4]=(OpenMXGpuTaskView){hchbuf,sizeof(double),add_hch?(size_t)spin_count*sizeof(double):0};
    task.view[5]=(OpenMXGpuTaskView){hbuf,0,(size_t)spin_count*sizeof(double)};
    int enabled=scf_eigen_lib_flag==GPUSOLVER && Set_Hamiltonian_OpenMP_Enabled();
    openmx_gpu_task_execute(mpi_comm_level1,&task,enabled,"Hamiltonian-base");

    for (pair = 0; pair < pair_count; pair++) {
        int NO0 = pair_NO0[pair];
        int NO1 = pair_NO1[pair];
        int spin, i, j;
        size_t mat_size = (size_t)NO0 * (size_t)NO1;
        size_t h_off = pair_h_offset[pair];
        Mc_AN = pair_Mc_AN[pair];
        h_AN = pair_h_AN[pair];

        for (spin = 0; spin < spin_count; spin++) {
            for (i = 0; i < NO0; i++) {
                for (j = 0; j < NO1; j++) {
                    size_t ij = (size_t)i * (size_t)NO1 + (size_t)j;
                    H[spin][Mc_AN][h_AN][i][j] = hbuf[h_off + ij * spin_count + spin];
                }
            }
        }
    }

    free(hchbuf);
    free(hhubbuf);
    free(hvnabuf);
    free(hnlbuf);
    free(h0buf);
    free(hbuf);
    free(pair_h_offset);
    free(pair_mat_offset);
    free(pair_NO1);
    free(pair_NO0);
    free(pair_h_AN);
    free(pair_Mc_AN);
}

/* Calc_MatrixElements_dVH_Vxc_VNA_OpenMP on the tables the previous call left
   resident on the device (Set_Hamiltonian_ME_CacheReusable): nothing is
   rebuilt or re-uploaded except the potential and the output, so an
   iteration moves the vpotbuf/hbuf arrays instead of the multi-GB orbital
   tables, and the resident copies earn their keep instead of merely
   waiting for Force3. */
static void Calc_MatrixElements_dVH_Vxc_VNA_OpenMP_Resident(int Cnt_kind)
{
    const SetHamiltonianMETableCache *c = &Set_Hamiltonian_ME_Tables;
    const int ompdev = omp_get_default_device();
    const int spin_count = (SpinP_switch == 3) ? 4 : (SpinP_switch + 1);
    const int pair_count = c->t.pair_count;
    const size_t total_h = c->t.total_h;
    const size_t total_nolg = c->t.total_nolg;
    const int *pair_Mc_AN = c->pair_Mc_AN;
    const int *pair_h_AN = c->pair_h_AN;
    const int *pair_NO0 = c->pair_NO0;
    const int *pair_NO1 = c->pair_NO1;
    const int *pair_NOLG = c->pair_NOLG;
    const int *nolg_MN = c->nolg_MN;
    const size_t *pair_h_offset = c->pair_h_offset;
    const size_t *pair_nolg_offset = c->pair_nolg_offset;
    const size_t *pair_orbs0_offset = c->pair_orbs0_offset;
    const size_t *pair_orbs1_offset = c->pair_orbs1_offset;
    const Type_Orbs_Grid *orbs0buf = c->orbs0buf;
    const Type_Orbs_Grid *orbs1buf = c->orbs1buf;
    int max_output_count = 0, max_no = 0, max_nolg = 0;
    int myid, pair, spin;
    size_t k;
    double *hbuf, *vpotbuf;

    MPI_Comm_rank(mpi_comm_level1, &myid);

    for (pair = 0; pair < pair_count; pair++) {
        int count = spin_count * pair_NO0[pair] * pair_NO1[pair];
        if (max_output_count < count) max_output_count = count;
        if (max_no < pair_NO0[pair]) max_no = pair_NO0[pair];
        if (max_no < pair_NO1[pair]) max_no = pair_NO1[pair];
        if (max_nolg < pair_NOLG[pair]) max_nolg = pair_NOLG[pair];
    }

    hbuf = (double *)Set_Hamiltonian_malloc(sizeof(double) * total_h, "openmp hbuf", myid);
    vpotbuf = (double *)Set_Hamiltonian_malloc(sizeof(double) * (size_t)spin_count * total_nolg,
                                               "openmp vpotbuf", myid);

    for (pair = 0; pair < pair_count; pair++) {
        const int Mc_AN = pair_Mc_AN[pair];
        const int h_AN = pair_h_AN[pair];
        const int NO0 = pair_NO0[pair];
        const int NO1 = pair_NO1[pair];
        const size_t mat_size = (size_t)NO0 * (size_t)NO1;
        const size_t h_off = pair_h_offset[pair];
        int i, j;

        for (spin = 0; spin < spin_count; spin++) {
            for (i = 0; i < NO0; i++) {
                for (j = 0; j < NO1; j++) {
                    size_t idx = h_off + (size_t)spin * mat_size + (size_t)i * (size_t)NO1 + (size_t)j;
                    hbuf[idx] = (Cnt_kind == 0) ? H[spin][Mc_AN][h_AN][i][j] : CntH[spin][Mc_AN][h_AN][i][j];
                }
            }
        }
    }

    for (spin = 0; spin < spin_count; spin++) {
        const double *vpot = Vpot_Grid[spin];
        double *dst = vpotbuf + (size_t)spin * total_nolg;

        for (k = 0; k < total_nolg; k++) {
            dst[k] = GridVol * vpot[nolg_MN[k]];
        }
    }

    if (Set_Hamiltonian_Hip_MatrixElements_Resident(
            pair_count, spin_count, max_output_count, max_no, max_nolg, total_h, total_nolg,
            (const int *)omp_get_mapped_ptr((void *)pair_NO0, ompdev),
            (const int *)omp_get_mapped_ptr((void *)pair_NO1, ompdev),
            (const int *)omp_get_mapped_ptr((void *)pair_NOLG, ompdev),
            (const size_t *)omp_get_mapped_ptr((void *)pair_h_offset, ompdev),
            (const size_t *)omp_get_mapped_ptr((void *)pair_nolg_offset, ompdev),
            (const size_t *)omp_get_mapped_ptr((void *)pair_orbs0_offset, ompdev),
            (const size_t *)omp_get_mapped_ptr((void *)pair_orbs1_offset, ompdev),
            (const int *)omp_get_mapped_ptr((void *)c->nolg_Nc, ompdev),
            (const float *)omp_get_mapped_ptr((void *)orbs0buf, ompdev),
            (const float *)omp_get_mapped_ptr((void *)orbs1buf, ompdev),
            vpotbuf, hbuf) != 0) {
        /* the device call failed (no room while other ranks hold the
           device): the host path, whose blocked loops are an order of
           magnitude faster than a scan of the packed tables; it adds to H
           itself, so the packed results are not written back */
        Calc_MatrixElements_dVH_Vxc_VNA_CPU(Cnt_kind);
        free(vpotbuf);
        free(hbuf);
        return;
    }

    if (total_h > 0) Set_Hamiltonian_MatrixElements_GPU_Used = 1;

    for (pair = 0; pair < pair_count; pair++) {
        const int Mc_AN = pair_Mc_AN[pair];
        const int h_AN = pair_h_AN[pair];
        const int NO0 = pair_NO0[pair];
        const int NO1 = pair_NO1[pair];
        const size_t mat_size = (size_t)NO0 * (size_t)NO1;
        const size_t h_off = pair_h_offset[pair];
        int i, j;

        for (spin = 0; spin < spin_count; spin++) {
            for (i = 0; i < NO0; i++) {
                for (j = 0; j < NO1; j++) {
                    size_t idx = h_off + (size_t)spin * mat_size + (size_t)i * (size_t)NO1 + (size_t)j;
                    if (Cnt_kind == 0) {
                        H[spin][Mc_AN][h_AN][i][j] = hbuf[idx];
                    } else {
                        CntH[spin][Mc_AN][h_AN][i][j] = hbuf[idx];
                    }
                }
            }
        }
    }

    free(vpotbuf);
    free(hbuf);
}

static void Calc_MatrixElements_dVH_Vxc_VNA_OpenMP(int Cnt_kind)
{
    int Mc_AN, Gc_AN, h_AN, Gh_AN, Mh_AN, Cwan, Hwan;
    int numprocs, myid;
    int spin_count, pair_count, pair;
    int *pair_Mc_AN, *pair_h_AN, *pair_NO0, *pair_NO1, *pair_NOLG;
    int *nolg_MN, *nolg_Nc;
    size_t *pair_h_offset, *pair_nolg_offset, *pair_orbs0_offset, *pair_orbs1_offset;
    size_t total_h, total_nolg, total_nolg_all, total_orbs0, total_orbs1;
    size_t packed_total_h, packed_total_orbs0, packed_total_orbs1;
    int max_output_count = 0, max_no = 0, max_nolg = 0, host_done = 0;
    double *hbuf, *vpotbuf;
    Type_Orbs_Grid *orbs0buf, *orbs1buf;

    MPI_Comm_size(mpi_comm_level1, &numprocs);
    MPI_Comm_rank(mpi_comm_level1, &myid);
    (void)numprocs;

    if (Set_Hamiltonian_ME_CacheReusable(Cnt_kind)) {
        Calc_MatrixElements_dVH_Vxc_VNA_OpenMP_Resident(Cnt_kind);
        return;
    }
    Set_Hamiltonian_Invalidate_OpenMP_MatrixElements_Cache();

    if (Cnt_kind != 0 && Cnt_kind != 1) {
        Set_Hamiltonian_abort("Calc_MatrixElements_dVH_Vxc_VNA_OpenMP", "Cnt_kind must be 0 or 1", myid);
    }

    if (SpinP_switch != 0 && SpinP_switch != 1 && SpinP_switch != 3) {
        Set_Hamiltonian_abort("Calc_MatrixElements_dVH_Vxc_VNA_OpenMP", "SpinP_switch must be 0, 1, or 3", myid);
    }

    spin_count = (SpinP_switch == 3) ? 4 : (SpinP_switch + 1);

    pair_count = 0;
    total_h = 0;
    total_nolg = 0;
    total_orbs0 = 0;
    total_orbs1 = 0;
    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
        Gc_AN = M2G[Mc_AN];
        Cwan = WhatSpecies[Gc_AN];

        for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
            int NO0, NO1, NOLG;

            Gh_AN = natn[Gc_AN][h_AN];
            Hwan = WhatSpecies[Gh_AN];
            NOLG = NumOLG[Mc_AN][h_AN];

            if (Cnt_kind == 0) {
                NO0 = Spe_Total_NO[Cwan];
                NO1 = Spe_Total_NO[Hwan];
            } else {
                NO0 = Spe_Total_CNO[Cwan];
                NO1 = Spe_Total_CNO[Hwan];
            }

            total_h += (size_t)spin_count * (size_t)NO0 * (size_t)NO1;
            if (max_output_count < spin_count * NO0 * NO1) max_output_count = spin_count * NO0 * NO1;
            if (max_no < NO0) max_no = NO0;
            if (max_no < NO1) max_no = NO1;
            if (max_nolg < NOLG) max_nolg = NOLG;
            total_nolg += (size_t)NOLG;
            total_orbs1 += (size_t)NOLG * (size_t)NO1;
            pair_count++;
        }
        /* the central atom's orbitals once per point of its sphere (the
           pairs select their overlap points through nolg_Nc) */
        total_orbs0 += (size_t)GridN_Atom[Gc_AN] *
                       (size_t)((Cnt_kind == 0) ? Spe_Total_NO[Cwan] : Spe_Total_CNO[Cwan]);
    }

    pair_Mc_AN = (int *)Set_Hamiltonian_malloc(sizeof(int) * (size_t)pair_count, "openmp pair_Mc_AN", myid);
    pair_h_AN = (int *)Set_Hamiltonian_malloc(sizeof(int) * (size_t)pair_count, "openmp pair_h_AN", myid);
    pair_NO0 = (int *)Set_Hamiltonian_malloc(sizeof(int) * (size_t)pair_count, "openmp pair_NO0", myid);
    pair_NO1 = (int *)Set_Hamiltonian_malloc(sizeof(int) * (size_t)pair_count, "openmp pair_NO1", myid);
    pair_NOLG = (int *)Set_Hamiltonian_malloc(sizeof(int) * (size_t)pair_count, "openmp pair_NOLG", myid);
    pair_h_offset =
        (size_t *)Set_Hamiltonian_malloc(sizeof(size_t) * (size_t)pair_count, "openmp pair_h_offset", myid);
    pair_nolg_offset =
        (size_t *)Set_Hamiltonian_malloc(sizeof(size_t) * (size_t)pair_count, "openmp pair_nolg_offset", myid);
    pair_orbs0_offset =
        (size_t *)Set_Hamiltonian_malloc(sizeof(size_t) * (size_t)pair_count, "openmp pair_orbs0_offset", myid);
    pair_orbs1_offset =
        (size_t *)Set_Hamiltonian_malloc(sizeof(size_t) * (size_t)pair_count, "openmp pair_orbs1_offset", myid);
    hbuf = (double *)Set_Hamiltonian_malloc(sizeof(double) * total_h, "openmp hbuf", myid);
    vpotbuf = (double *)Set_Hamiltonian_malloc(sizeof(double) * (size_t)spin_count * total_nolg, "openmp vpotbuf", myid);
    orbs0buf =
        (Type_Orbs_Grid *)Set_Hamiltonian_malloc(sizeof(Type_Orbs_Grid) * total_orbs0, "openmp orbs0buf", myid);
    orbs1buf =
        (Type_Orbs_Grid *)Set_Hamiltonian_malloc(sizeof(Type_Orbs_Grid) * total_orbs1, "openmp orbs1buf", myid);
    nolg_MN = (int *)Set_Hamiltonian_malloc(sizeof(int) * total_nolg, "openmp nolg_MN", myid);
    nolg_Nc = (int *)Set_Hamiltonian_malloc(sizeof(int) * total_nolg, "openmp nolg_Nc", myid);

    total_nolg_all = total_nolg;
    packed_total_h = total_h;
    packed_total_orbs0 = total_orbs0;
    packed_total_orbs1 = total_orbs1;

    pair = 0;
    total_h = 0;
    total_nolg = 0;
    total_orbs0 = 0;
    total_orbs1 = 0;
    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
        const size_t atom_orbs0_offset = total_orbs0;

        Gc_AN = M2G[Mc_AN];
        Cwan = WhatSpecies[Gc_AN];

        {
            const int NO0 = (Cnt_kind == 0) ? Spe_Total_NO[Cwan] : Spe_Total_CNO[Cwan];
            int Nc, i;

            for (Nc = 0; Nc < GridN_Atom[Gc_AN]; Nc++) {
                for (i = 0; i < NO0; i++) {
                    orbs0buf[atom_orbs0_offset + (size_t)Nc * (size_t)NO0 + (size_t)i] = Orbs_Grid[Mc_AN][Nc][i];
                }
            }
            total_orbs0 += (size_t)GridN_Atom[Gc_AN] * (size_t)NO0;
        }

        for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
            int NO0, NO1, NOLG;
            size_t mat_size;
            int spin, i, j, Nog;

            Gh_AN = natn[Gc_AN][h_AN];
            Mh_AN = F_G2M[Gh_AN];
            Hwan = WhatSpecies[Gh_AN];
            NOLG = NumOLG[Mc_AN][h_AN];

            if (Cnt_kind == 0) {
                NO0 = Spe_Total_NO[Cwan];
                NO1 = Spe_Total_NO[Hwan];
            } else {
                NO0 = Spe_Total_CNO[Cwan];
                NO1 = Spe_Total_CNO[Hwan];
            }

            mat_size = (size_t)NO0 * (size_t)NO1;

            pair_Mc_AN[pair] = Mc_AN;
            pair_h_AN[pair] = h_AN;
            pair_NO0[pair] = NO0;
            pair_NO1[pair] = NO1;
            pair_NOLG[pair] = NOLG;
            pair_h_offset[pair] = total_h;
            pair_nolg_offset[pair] = total_nolg;
            pair_orbs0_offset[pair] = atom_orbs0_offset;
            pair_orbs1_offset[pair] = total_orbs1;

            for (spin = 0; spin < spin_count; spin++) {
                for (i = 0; i < NO0; i++) {
                    for (j = 0; j < NO1; j++) {
                        size_t idx = total_h + (size_t)spin * mat_size + (size_t)i * (size_t)NO1 + (size_t)j;
                        hbuf[idx] = (Cnt_kind == 0) ? H[spin][Mc_AN][h_AN][i][j] : CntH[spin][Mc_AN][h_AN][i][j];
                    }
                }
            }

            for (Nog = 0; Nog < NOLG; Nog++) {
                int Nc = GListTAtoms1[Mc_AN][h_AN][Nog];
                int MN = MGridListAtom[Mc_AN][Nc];
                int Nh = GListTAtoms2[Mc_AN][h_AN][Nog];
                Type_Orbs_Grid *orbs1 = (G2ID[Gh_AN] == myid) ? Orbs_Grid[Mh_AN][Nh] : Orbs_Grid_FNAN[Mc_AN][h_AN][Nog];

                nolg_MN[total_nolg + (size_t)Nog] = MN;
                nolg_Nc[total_nolg + (size_t)Nog] = Nc;

                for (j = 0; j < NO1; j++) {
                    orbs1buf[total_orbs1 + (size_t)Nog * (size_t)NO1 + (size_t)j] = orbs1[j];
                }
                for (spin = 0; spin < spin_count; spin++) {
                    vpotbuf[(size_t)spin * total_nolg_all + total_nolg + (size_t)Nog] =
                        GridVol * Vpot_Grid[spin][MN];
                }
            }

            total_h += (size_t)spin_count * mat_size;
            total_nolg += (size_t)NOLG;
            total_orbs1 += (size_t)NOLG * (size_t)NO1;
            pair++;
        }
    }

    {
        const char *check = getenv("OPENMX_OTF_CHECK");
        if (check != NULL && atoi(check) != 0 && Cnt_kind == 0) {
            Set_Hamiltonian_OTF_Check(pair_count, pair_Mc_AN, pair_h_AN, pair_NO0, pair_NO1, pair_NOLG,
                                      pair_nolg_offset, pair_orbs1_offset, nolg_Nc, total_nolg_all,
                                      orbs0buf, packed_total_orbs0, orbs1buf, packed_total_orbs1, myid);
        }
    }

    if (Set_Hamiltonian_Hip_MatrixElements(
            pair_count, spin_count, max_output_count, max_no, max_nolg,
            packed_total_h, total_nolg_all, packed_total_orbs0, packed_total_orbs1,
            pair_NO0, pair_NO1, pair_NOLG, pair_h_offset, pair_nolg_offset,
            pair_orbs0_offset, pair_orbs1_offset, nolg_Nc, vpotbuf,
            (const float *)orbs0buf, (const float *)orbs1buf, hbuf) != 0) {
        /* the device call failed: the host path (blocked loops, it adds to
           H itself) instead of a scan of the packed tables; the tables
           are still cached below */
        Calc_MatrixElements_dVH_Vxc_VNA_CPU(Cnt_kind);
        host_done = 1;
    }

    if (!host_done && packed_total_h > 0) Set_Hamiltonian_MatrixElements_GPU_Used = 1;

    for (pair = 0; pair < pair_count && !host_done; pair++) {
        int NO0 = pair_NO0[pair];
        int NO1 = pair_NO1[pair];
        int spin, i, j;
        size_t mat_size = (size_t)NO0 * (size_t)NO1;
        size_t h_off = pair_h_offset[pair];
        Mc_AN = pair_Mc_AN[pair];
        h_AN = pair_h_AN[pair];

        for (spin = 0; spin < spin_count; spin++) {
            for (i = 0; i < NO0; i++) {
                for (j = 0; j < NO1; j++) {
                    size_t idx = h_off + (size_t)spin * mat_size + (size_t)i * (size_t)NO1 + (size_t)j;
                    if (Cnt_kind == 0) {
                        H[spin][Mc_AN][h_AN][i][j] = hbuf[idx];
                    } else {
                        CntH[spin][Mc_AN][h_AN][i][j] = hbuf[idx];
                    }
                }
            }
        }
    }

    {
        SetHamiltonianMETableCache *c = &Set_Hamiltonian_ME_Tables;
        const int keep_device_resident = Set_Hamiltonian_ME_KeepDeviceResident();
        int resident_ok = 0;

        c->ready = 1; c->cnt_kind = Cnt_kind;
        c->pair_Mc_AN=pair_Mc_AN; c->pair_h_AN=pair_h_AN;
        c->pair_NO0=pair_NO0; c->pair_NO1=pair_NO1; c->pair_NOLG=pair_NOLG;
        c->nolg_MN=nolg_MN; c->nolg_Nc=nolg_Nc;
        c->pair_h_offset=pair_h_offset; c->pair_nolg_offset=pair_nolg_offset;
        c->pair_orbs0_offset=pair_orbs0_offset; c->pair_orbs1_offset=pair_orbs1_offset;
        c->orbs0buf=orbs0buf; c->orbs1buf=orbs1buf;

        c->t.pair_count=pair_count; c->t.total_h=packed_total_h;
        c->t.total_nolg=total_nolg_all; c->t.total_orbs0=packed_total_orbs0;
        c->t.total_orbs1=packed_total_orbs1;
        c->t.pair_Mc_AN=pair_Mc_AN; c->t.pair_h_AN=pair_h_AN;
        c->t.pair_NO0=pair_NO0; c->t.pair_NO1=pair_NO1; c->t.pair_NOLG=pair_NOLG;
        c->t.nolg_MN=nolg_MN; c->t.nolg_Nc=nolg_Nc;
        c->t.pair_h_offset=pair_h_offset; c->t.pair_nolg_offset=pair_nolg_offset;
        c->t.pair_orbs0_offset=pair_orbs0_offset; c->t.pair_orbs1_offset=pair_orbs1_offset;
        c->t.orbs0buf=orbs0buf; c->t.orbs1buf=orbs1buf;

        if (keep_device_resident) {
            resident_ok = 1;
            resident_ok &= Set_Hamiltonian_ME_Associate(pair_NO0,sizeof(int)*(size_t)pair_count);
            resident_ok &= Set_Hamiltonian_ME_Associate(pair_NO1,sizeof(int)*(size_t)pair_count);
            resident_ok &= Set_Hamiltonian_ME_Associate(pair_NOLG,sizeof(int)*(size_t)pair_count);
            resident_ok &= Set_Hamiltonian_ME_Associate(nolg_MN,sizeof(int)*total_nolg_all);
            resident_ok &= Set_Hamiltonian_ME_Associate(nolg_Nc,sizeof(int)*total_nolg_all);
            resident_ok &= Set_Hamiltonian_ME_Associate(pair_h_offset,sizeof(size_t)*(size_t)pair_count);
            resident_ok &= Set_Hamiltonian_ME_Associate(pair_nolg_offset,sizeof(size_t)*(size_t)pair_count);
            resident_ok &= Set_Hamiltonian_ME_Associate(pair_orbs0_offset,sizeof(size_t)*(size_t)pair_count);
            resident_ok &= Set_Hamiltonian_ME_Associate(pair_orbs1_offset,sizeof(size_t)*(size_t)pair_count);
            resident_ok &= Set_Hamiltonian_ME_Associate(orbs0buf,sizeof(Type_Orbs_Grid)*packed_total_orbs0);
            resident_ok &= Set_Hamiltonian_ME_Associate(orbs1buf,sizeof(Type_Orbs_Grid)*packed_total_orbs1);
        }
        c->t.meta_resident=resident_ok; c->t.nolg_resident=resident_ok;
        c->t.orbs0_resident=resident_ok; c->t.orbs1_resident=resident_ok;
        if (keep_device_resident && !resident_ok) {
            /* no room for the resident tables: from the next call on this
               rank takes the on-the-fly tiles rather than building and
               uploading the tables for every call */
            Set_Hamiltonian_ME_residency_demoted = 1;
        }
        if (!resident_ok) {
            /* An allocation or association may fail after earlier entries
               succeeded.  Do not leave those partial mappings live merely
               because the aggregate resident flags are false. */
            Set_Hamiltonian_ME_Disassociate(pair_NO0);
            Set_Hamiltonian_ME_Disassociate(pair_NO1);
            Set_Hamiltonian_ME_Disassociate(pair_NOLG);
            Set_Hamiltonian_ME_Disassociate(nolg_MN);
            Set_Hamiltonian_ME_Disassociate(nolg_Nc);
            Set_Hamiltonian_ME_Disassociate(pair_h_offset);
            Set_Hamiltonian_ME_Disassociate(pair_nolg_offset);
            Set_Hamiltonian_ME_Disassociate(pair_orbs0_offset);
            Set_Hamiltonian_ME_Disassociate(pair_orbs1_offset);
            Set_Hamiltonian_ME_Disassociate(orbs0buf);
            Set_Hamiltonian_ME_Disassociate(orbs1buf);
            c->t.meta_resident=c->t.nolg_resident=0;
            c->t.orbs0_resident=c->t.orbs1_resident=0;
        }
        if (!keep_device_resident && myid == Host_ID) {
            const char *trace = getenv("OPENMX_BAND_CACHE_TRACE");
            if (trace != NULL && atoi(trace) != 0) {
                printf("<DFT> Set_Hamiltonian orbital-table residency skipped for NC eigensolver\n");
                fflush(stdout);
            }
        }
    }
    free(vpotbuf);
    free(hbuf);
}

#define SETH_ME_BLK 16
#define SETH_ME_MAXNO 16

/* Blocked spin-unpolarized quadrature for one (Mc_AN,h_AN) pair: grid points
   are staged in panels of SETH_ME_BLK and the NO0xNO1 accumulator block stays
   L1-resident for the whole pair, instead of re-storing all NO0*NO1 partial
   sums for every grid point as the generic loop below does. The j dimension
   is padded to SETH_ME_MAXNO so the inner update vectorizes with full-width
   FMAs. Only the summation order differs from the generic path. */
static SETH_ALWAYS_INLINE void Set_Hamiltonian_ME_Pair_Blocked_Spin0_Impl(
                                                  int Mc_AN, int h_AN, int Mh_AN,
                                                  int Gh_AN_is_local, int NO0, int NO1,
                                                  int NOLG, double *restrict acc)
{
    const int *restrict nc_list = GListTAtoms1[Mc_AN][h_AN];
    const int *restrict nh_list = GListTAtoms2[Mc_AN][h_AN];
    const int *restrict mgrid   = MGridListAtom[Mc_AN];
    const double *restrict vpot = Vpot_Grid[0];
    Type_Orbs_Grid **restrict og_c = Orbs_Grid[Mc_AN];
    Type_Orbs_Grid **restrict og_h = Gh_AN_is_local ? Orbs_Grid[Mh_AN] : NULL;
    Type_Orbs_Grid **restrict og_f = Gh_AN_is_local ? NULL : Orbs_Grid_FNAN[Mc_AN][h_AN];
    double w[SETH_ME_BLK][SETH_ME_MAXNO] __attribute__((aligned(64)));
    double p1[SETH_ME_BLK][SETH_ME_MAXNO] __attribute__((aligned(64)));
    int    Nog0, g, i, j;

    for (i = 0; i < NO0 * SETH_ME_MAXNO; i++) {
        acc[i] = 0.0;
    }

    for (g = 0; g < SETH_ME_BLK; g++) {
        for (j = NO1; j < SETH_ME_MAXNO; j++) {
            p1[g][j] = 0.0;
        }
    }

    for (Nog0 = 0; Nog0 < NOLG; Nog0 += SETH_ME_BLK) {
        const int blk = (NOLG - Nog0 < SETH_ME_BLK) ? (NOLG - Nog0) : SETH_ME_BLK;

        for (g = 0; g < blk; g++) {
            const int Nog = Nog0 + g;
            const int Nc  = nc_list[Nog];
            const Type_Orbs_Grid *restrict orbs0 = og_c[Nc];
            const Type_Orbs_Grid *restrict orbs1 = og_h ? og_h[nh_list[Nog]] : og_f[Nog];
            const double v = GridVol * vpot[mgrid[Nc]];

            for (i = 0; i < NO0; i++) {
                w[g][i] = v * (double)orbs0[i];
            }
            for (j = 0; j < NO1; j++) {
                p1[g][j] = (double)orbs1[j];
            }
        }

        /* Consume four staged grid points while each accumulator vector is
           live.  Every element still sees g in ascending order, but the
           accumulator vectors are loaded/stored once per four rank-1
           updates instead of once per update. */
        g = 0;
        for (; g + 3 < blk; g += 4) {
            const double *restrict p10 = p1[g + 0];
            const double *restrict p11 = p1[g + 1];
            const double *restrict p12 = p1[g + 2];
            const double *restrict p13 = p1[g + 3];

#if defined(__clang__)
#pragma clang loop vectorize(disable)
#endif
            for (i = 0; i < NO0; i++) {
                const double a0 = w[g + 0][i];
                const double a1 = w[g + 1][i];
                const double a2 = w[g + 2][i];
                const double a3 = w[g + 3][i];
                double *restrict accrow = &acc[i * SETH_ME_MAXNO];

#pragma omp simd aligned(accrow,p10,p11,p12,p13:64)
                for (j = 0; j < SETH_ME_MAXNO; j++) {
                    double h = accrow[j];
                    h += a0*p10[j];
                    h += a1*p11[j];
                    h += a2*p12[j];
                    h += a3*p13[j];
                    accrow[j] = h;
                }
            }
        }
        for (; g < blk; g++) {
            const double *restrict p1g = p1[g];
            const double *restrict wg  = w[g];

#if defined(__clang__)
#pragma clang loop vectorize(disable)
#endif
            for (i = 0; i < NO0; i++) {
                const double a = wg[i];
                double *restrict accrow = &acc[i * SETH_ME_MAXNO];

#pragma omp simd aligned(accrow,p1g:64)
                for (j = 0; j < SETH_ME_MAXNO; j++) {
                    accrow[j] += a*p1g[j];
                }
            }
        }
    }
}

typedef void (*Set_Hamiltonian_Spin0_Pair_Function)(
                                                  int Mc_AN, int h_AN, int Mh_AN,
                                                  int Gh_AN_is_local, int NO0, int NO1,
                                                  int NOLG, double *restrict acc);

static SETH_NOINLINE void Set_Hamiltonian_ME_Pair_Blocked_Spin0_Baseline(
                                                  int Mc_AN, int h_AN, int Mh_AN,
                                                  int Gh_AN_is_local, int NO0, int NO1,
                                                  int NOLG, double *restrict acc)
{
    Set_Hamiltonian_ME_Pair_Blocked_Spin0_Impl(
        Mc_AN,h_AN,Mh_AN,Gh_AN_is_local,NO0,NO1,NOLG,acc);
}

static SETH_TARGET_AVX2 void Set_Hamiltonian_ME_Pair_Blocked_Spin0_AVX2(
                                                  int Mc_AN, int h_AN, int Mh_AN,
                                                  int Gh_AN_is_local, int NO0, int NO1,
                                                  int NOLG, double *restrict acc)
{
    Set_Hamiltonian_ME_Pair_Blocked_Spin0_Impl(
        Mc_AN,h_AN,Mh_AN,Gh_AN_is_local,NO0,NO1,NOLG,acc);
}

static SETH_TARGET_AVX512 void Set_Hamiltonian_ME_Pair_Blocked_Spin0_AVX512(
                                                  int Mc_AN, int h_AN, int Mh_AN,
                                                  int Gh_AN_is_local, int NO0, int NO1,
                                                  int NOLG, double *restrict acc)
{
    Set_Hamiltonian_ME_Pair_Blocked_Spin0_Impl(
        Mc_AN,h_AN,Mh_AN,Gh_AN_is_local,NO0,NO1,NOLG,acc);
}

/* Accumulate four overlap-grid points at a time for the non-collinear CPU
   path.  Each matrix element stays in a scalar/vector register across the
   four updates, reducing accumulator traffic without regrouping the floating
   point sum: for every spin and (i,j), contributions are still added in
   increasing Nog order. */
static SETH_ALWAYS_INLINE void Set_Hamiltonian_ME_Pair_NC4_Impl(
                                       int Mc_AN, int h_AN, int Mh_AN,
                                       int Gh_AN_is_local, int NO0, int NO1,
                                       int NOLG, double **restrict tmpH0,
                                       double **restrict tmpH1,
                                       double **restrict tmpH2,
                                       double **restrict tmpH3)
{
    const int *restrict nc_list = GListTAtoms1[Mc_AN][h_AN];
    const int *restrict nh_list = GListTAtoms2[Mc_AN][h_AN];
    const int *restrict mgrid = MGridListAtom[Mc_AN];
    const double *restrict vpot0 = Vpot_Grid[0];
    const double *restrict vpot1 = Vpot_Grid[1];
    const double *restrict vpot2 = Vpot_Grid[2];
    const double *restrict vpot3 = Vpot_Grid[3];
    Type_Orbs_Grid **restrict og_c = Orbs_Grid[Mc_AN];
    Type_Orbs_Grid **restrict og_h = Gh_AN_is_local ? Orbs_Grid[Mh_AN] : NULL;
    Type_Orbs_Grid **restrict og_f = Gh_AN_is_local ? NULL : Orbs_Grid_FNAN[Mc_AN][h_AN];
    int Nog, i, j;

    for (Nog = 0; Nog <= NOLG - 4; Nog += 4) {
        const int Nc0 = nc_list[Nog];
        const int Nc1 = nc_list[Nog + 1];
        const int Nc2 = nc_list[Nog + 2];
        const int Nc3 = nc_list[Nog + 3];
        const int MN0 = mgrid[Nc0];
        const int MN1 = mgrid[Nc1];
        const int MN2 = mgrid[Nc2];
        const int MN3 = mgrid[Nc3];
        const Type_Orbs_Grid *restrict orbs0_0 = og_c[Nc0];
        const Type_Orbs_Grid *restrict orbs0_1 = og_c[Nc1];
        const Type_Orbs_Grid *restrict orbs0_2 = og_c[Nc2];
        const Type_Orbs_Grid *restrict orbs0_3 = og_c[Nc3];
        const Type_Orbs_Grid *restrict orbs1_0 = og_h ? og_h[nh_list[Nog]] : og_f[Nog];
        const Type_Orbs_Grid *restrict orbs1_1 = og_h ? og_h[nh_list[Nog + 1]] : og_f[Nog + 1];
        const Type_Orbs_Grid *restrict orbs1_2 = og_h ? og_h[nh_list[Nog + 2]] : og_f[Nog + 2];
        const Type_Orbs_Grid *restrict orbs1_3 = og_h ? og_h[nh_list[Nog + 3]] : og_f[Nog + 3];
        const double v0_0 = GridVol * vpot0[MN0];
        const double v1_0 = GridVol * vpot1[MN0];
        const double v2_0 = GridVol * vpot2[MN0];
        const double v3_0 = GridVol * vpot3[MN0];
        const double v0_1 = GridVol * vpot0[MN1];
        const double v1_1 = GridVol * vpot1[MN1];
        const double v2_1 = GridVol * vpot2[MN1];
        const double v3_1 = GridVol * vpot3[MN1];
        const double v0_2 = GridVol * vpot0[MN2];
        const double v1_2 = GridVol * vpot1[MN2];
        const double v2_2 = GridVol * vpot2[MN2];
        const double v3_2 = GridVol * vpot3[MN2];
        const double v0_3 = GridVol * vpot0[MN3];
        const double v1_3 = GridVol * vpot1[MN3];
        const double v2_3 = GridVol * vpot2[MN3];
        const double v3_3 = GridVol * vpot3[MN3];

        for (i = 0; i < NO0; i++) {
            const double orb0_0 = (double)orbs0_0[i];
            const double orb0_1 = (double)orbs0_1[i];
            const double orb0_2 = (double)orbs0_2[i];
            const double orb0_3 = (double)orbs0_3[i];
            const double a0_0 = v0_0 * orb0_0;
            const double a1_0 = v1_0 * orb0_0;
            const double a2_0 = v2_0 * orb0_0;
            const double a3_0 = v3_0 * orb0_0;
            const double a0_1 = v0_1 * orb0_1;
            const double a1_1 = v1_1 * orb0_1;
            const double a2_1 = v2_1 * orb0_1;
            const double a3_1 = v3_1 * orb0_1;
            const double a0_2 = v0_2 * orb0_2;
            const double a1_2 = v1_2 * orb0_2;
            const double a2_2 = v2_2 * orb0_2;
            const double a3_2 = v3_2 * orb0_2;
            const double a0_3 = v0_3 * orb0_3;
            const double a1_3 = v1_3 * orb0_3;
            const double a2_3 = v2_3 * orb0_3;
            const double a3_3 = v3_3 * orb0_3;
            double *restrict row0 = tmpH0[i];
            double *restrict row1 = tmpH1[i];
            double *restrict row2 = tmpH2[i];
            double *restrict row3 = tmpH3[i];

            for (j = 0; j < NO1; j++) {
                double h0 = row0[j];
                double h1 = row1[j];
                double h2 = row2[j];
                double h3 = row3[j];
                double orb1 = (double)orbs1_0[j];

                h0 += a0_0 * orb1;
                h1 += a1_0 * orb1;
                h2 += a2_0 * orb1;
                h3 += a3_0 * orb1;

                orb1 = (double)orbs1_1[j];
                h0 += a0_1 * orb1;
                h1 += a1_1 * orb1;
                h2 += a2_1 * orb1;
                h3 += a3_1 * orb1;

                orb1 = (double)orbs1_2[j];
                h0 += a0_2 * orb1;
                h1 += a1_2 * orb1;
                h2 += a2_2 * orb1;
                h3 += a3_2 * orb1;

                orb1 = (double)orbs1_3[j];
                h0 += a0_3 * orb1;
                h1 += a1_3 * orb1;
                h2 += a2_3 * orb1;
                h3 += a3_3 * orb1;

                row0[j] = h0;
                row1[j] = h1;
                row2[j] = h2;
                row3[j] = h3;
            }
        }
    }

    /* One to three points may remain.  Keep the original single-Nog kernel
       for this tail, including its arithmetic order. */
    for (; Nog < NOLG; Nog++) {
        const int Nc = nc_list[Nog];
        const int MN = mgrid[Nc];
        const Type_Orbs_Grid *restrict orbs0 = og_c[Nc];
        const Type_Orbs_Grid *restrict orbs1 = og_h ? og_h[nh_list[Nog]] : og_f[Nog];
        const double v0 = GridVol * vpot0[MN];
        const double v1 = GridVol * vpot1[MN];
        const double v2 = GridVol * vpot2[MN];
        const double v3 = GridVol * vpot3[MN];

        for (i = 0; i < NO0; i++) {
            const double orb0 = (double)orbs0[i];
            const double a0 = v0 * orb0;
            const double a1 = v1 * orb0;
            const double a2 = v2 * orb0;
            const double a3 = v3 * orb0;
            double *restrict row0 = tmpH0[i];
            double *restrict row1 = tmpH1[i];
            double *restrict row2 = tmpH2[i];
            double *restrict row3 = tmpH3[i];

            for (j = 0; j < NO1; j++) {
                const double orb1 = (double)orbs1[j];
                row0[j] += a0 * orb1;
                row1[j] += a1 * orb1;
                row2[j] += a2 * orb1;
                row3[j] += a3 * orb1;
            }
        }
    }
}

typedef void (*Set_Hamiltonian_NC4_Pair_Function)(
                                       int Mc_AN, int h_AN, int Mh_AN,
                                       int Gh_AN_is_local, int NO0, int NO1,
                                       int NOLG, double **restrict tmpH0,
                                       double **restrict tmpH1,
                                       double **restrict tmpH2,
                                       double **restrict tmpH3);

static SETH_NOINLINE void Set_Hamiltonian_ME_Pair_NC4_Baseline(
                                       int Mc_AN, int h_AN, int Mh_AN,
                                       int Gh_AN_is_local, int NO0, int NO1,
                                       int NOLG, double **restrict tmpH0,
                                       double **restrict tmpH1,
                                       double **restrict tmpH2,
                                       double **restrict tmpH3)
{
    Set_Hamiltonian_ME_Pair_NC4_Impl(
        Mc_AN,h_AN,Mh_AN,Gh_AN_is_local,NO0,NO1,NOLG,
        tmpH0,tmpH1,tmpH2,tmpH3);
}

static SETH_TARGET_AVX2 void Set_Hamiltonian_ME_Pair_NC4_AVX2(
                                       int Mc_AN, int h_AN, int Mh_AN,
                                       int Gh_AN_is_local, int NO0, int NO1,
                                       int NOLG, double **restrict tmpH0,
                                       double **restrict tmpH1,
                                       double **restrict tmpH2,
                                       double **restrict tmpH3)
{
    Set_Hamiltonian_ME_Pair_NC4_Impl(
        Mc_AN,h_AN,Mh_AN,Gh_AN_is_local,NO0,NO1,NOLG,
        tmpH0,tmpH1,tmpH2,tmpH3);
}

static SETH_TARGET_AVX512 void Set_Hamiltonian_ME_Pair_NC4_AVX512(
                                       int Mc_AN, int h_AN, int Mh_AN,
                                       int Gh_AN_is_local, int NO0, int NO1,
                                       int NOLG, double **restrict tmpH0,
                                       double **restrict tmpH1,
                                       double **restrict tmpH2,
                                       double **restrict tmpH3)
{
    Set_Hamiltonian_ME_Pair_NC4_Impl(
        Mc_AN,h_AN,Mh_AN,Gh_AN_is_local,NO0,NO1,NOLG,
        tmpH0,tmpH1,tmpH2,tmpH3);
}

/* Spin-polarized collinear integration uses the same runtime ISA policy.
   Keeping the long Nog loop inside a targeted wrapper avoids silently
   reducing this path to the translation unit's non-AVX baseline. */
static SETH_ALWAYS_INLINE void Set_Hamiltonian_ME_Pair_Spin1_Impl(
                                       int Mc_AN, int h_AN, int Mh_AN,
                                       int Gh_AN_is_local, int NO0, int NO1,
                                       int NOLG, double **restrict tmpH0,
                                       double **restrict tmpH1)
{
    const int *restrict nc_list = GListTAtoms1[Mc_AN][h_AN];
    const int *restrict nh_list = GListTAtoms2[Mc_AN][h_AN];
    const int *restrict mgrid = MGridListAtom[Mc_AN];
    const double *restrict vpot0 = Vpot_Grid[0];
    const double *restrict vpot1 = Vpot_Grid[1];
    Type_Orbs_Grid **restrict og_c = Orbs_Grid[Mc_AN];
    Type_Orbs_Grid **restrict og_h = Gh_AN_is_local ? Orbs_Grid[Mh_AN] : NULL;
    Type_Orbs_Grid **restrict og_f = Gh_AN_is_local ? NULL : Orbs_Grid_FNAN[Mc_AN][h_AN];
    int Nog, i, j;

    for (Nog = 0; Nog < NOLG; Nog++) {
        const int Nc = nc_list[Nog];
        const int MN = mgrid[Nc];
        const Type_Orbs_Grid *restrict orbs0 = og_c[Nc];
        const Type_Orbs_Grid *restrict orbs1 =
            og_h ? og_h[nh_list[Nog]] : og_f[Nog];
        const double v0 = GridVol*vpot0[MN];
        const double v1 = GridVol*vpot1[MN];

        for (i = 0; i < NO0; i++) {
            const double orb0 = (double)orbs0[i];
            const double a0 = v0*orb0;
            const double a1 = v1*orb0;
            double *restrict row0 = tmpH0[i];
            double *restrict row1 = tmpH1[i];

            for (j = 0; j < NO1; j++) {
                const double orb1 = (double)orbs1[j];
                row0[j] += a0*orb1;
                row1[j] += a1*orb1;
            }
        }
    }
}

typedef void (*Set_Hamiltonian_Spin1_Pair_Function)(
                                       int Mc_AN, int h_AN, int Mh_AN,
                                       int Gh_AN_is_local, int NO0, int NO1,
                                       int NOLG, double **restrict tmpH0,
                                       double **restrict tmpH1);

#define SETH_DEFINE_SPIN1_WRAPPER(NAME,ATTR) \
static ATTR void NAME(int Mc_AN, int h_AN, int Mh_AN, \
                      int Gh_AN_is_local, int NO0, int NO1, int NOLG, \
                      double **restrict tmpH0, double **restrict tmpH1) \
{ \
    Set_Hamiltonian_ME_Pair_Spin1_Impl( \
        Mc_AN,h_AN,Mh_AN,Gh_AN_is_local,NO0,NO1,NOLG,tmpH0,tmpH1); \
}

SETH_DEFINE_SPIN1_WRAPPER(Set_Hamiltonian_ME_Pair_Spin1_Baseline,SETH_NOINLINE)
SETH_DEFINE_SPIN1_WRAPPER(Set_Hamiltonian_ME_Pair_Spin1_AVX2,SETH_TARGET_AVX2)
SETH_DEFINE_SPIN1_WRAPPER(Set_Hamiltonian_ME_Pair_Spin1_AVX512,SETH_TARGET_AVX512)

#undef SETH_DEFINE_SPIN1_WRAPPER

static void Calc_MatrixElements_dVH_Vxc_VNA_CPU(int Cnt_kind)
{
    int    Mc_AN, Gc_AN, Mh_AN, h_AN, Gh_AN;
    int    Nh0, Nh1, Nh2, Nh3;
    int    Nc0, Nc1, Nc2, Nc3;
    int    MN0, MN1, MN2, MN3;
    int    Nloop, OneD_Nloop;
    int    spin0_simd_mode;
    int *  OneD2spin, *OneD2Mc_AN, *OneD2h_AN;
    int    numprocs, myid;
    double time0, time1, time2, mflops;
    Set_Hamiltonian_Spin0_Pair_Function spin0_pair_function;
    Set_Hamiltonian_NC4_Pair_Function nc4_pair_function;
    Set_Hamiltonian_Spin1_Pair_Function spin1_pair_function;

    if (measure_time)
        dtime(&time1);

    MPI_Comm_size(mpi_comm_level1, &numprocs);
    MPI_Comm_rank(mpi_comm_level1, &myid);

    spin0_simd_mode = Set_Hamiltonian_Simd_Mode();
    if (spin0_simd_mode == SETH_SIMD_AVX512) {
        spin0_pair_function = Set_Hamiltonian_ME_Pair_Blocked_Spin0_AVX512;
        nc4_pair_function = Set_Hamiltonian_ME_Pair_NC4_AVX512;
        spin1_pair_function = Set_Hamiltonian_ME_Pair_Spin1_AVX512;
    }
    else if (spin0_simd_mode == SETH_SIMD_AVX2) {
        spin0_pair_function = Set_Hamiltonian_ME_Pair_Blocked_Spin0_AVX2;
        nc4_pair_function = Set_Hamiltonian_ME_Pair_NC4_AVX2;
        spin1_pair_function = Set_Hamiltonian_ME_Pair_Spin1_AVX2;
    }
    else {
        spin0_pair_function = Set_Hamiltonian_ME_Pair_Blocked_Spin0_Baseline;
        nc4_pair_function = Set_Hamiltonian_ME_Pair_NC4_Baseline;
        spin1_pair_function = Set_Hamiltonian_ME_Pair_Spin1_Baseline;
    }
    {
        const char *trace = getenv("OPENMX_SETHAM_SIMD_TRACE");
        if (myid == Host_ID && trace != NULL && atoi(trace) != 0) {
            printf("<Set_Hamiltonian> matrix-elements SIMD backend=%s\n",
                   Set_Hamiltonian_Simd_Mode_Name(spin0_simd_mode));
            fflush(stdout);
        }
    }

    if (Cnt_kind != 0 && Cnt_kind != 1) {
        Set_Hamiltonian_abort("Calc_MatrixElements_dVH_Vxc_VNA", "Cnt_kind must be 0 or 1", myid);
    }

    if (SpinP_switch != 0 && SpinP_switch != 1 && SpinP_switch != 3) {
        Set_Hamiltonian_abort("Calc_MatrixElements_dVH_Vxc_VNA", "SpinP_switch must be 0, 1, or 3", myid);
    }

    /* one-dimensionalization of loops */

    Nloop = 0;
    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
        Gc_AN = M2G[Mc_AN];
        for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
            Nloop++;
        }
    }

    OneD2Mc_AN = NULL;
    OneD2h_AN  = NULL;

    if (0 < Nloop) {
        if ((size_t)Nloop > ((size_t)-1) / sizeof(int)) {
            Set_Hamiltonian_abort("Calc_MatrixElements_dVH_Vxc_VNA", "loop table size overflow", myid);
        }

        OneD2Mc_AN = (int *)malloc(sizeof(int) * (size_t)Nloop);
        OneD2h_AN  = (int *)malloc(sizeof(int) * (size_t)Nloop);

        if (OneD2Mc_AN == NULL || OneD2h_AN == NULL) {
            free(OneD2Mc_AN);
            free(OneD2h_AN);
            Set_Hamiltonian_abort("Calc_MatrixElements_dVH_Vxc_VNA", "failed to allocate loop tables", myid);
        }
    }

    Nloop = 0;
    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
        Gc_AN = M2G[Mc_AN];
        for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {

            OneD2Mc_AN[Nloop] = Mc_AN;
            OneD2h_AN[Nloop]  = h_AN;
            Nloop++;
        }
    }

    OneD_Nloop = Nloop;

    if (measure_time) {
        dtime(&time2);
        printf("myid=%4d Time3=%18.10f\n", myid, time2 - time1);
        fflush(stdout);
    }

    /* numerical integration */

    if (measure_time)
        dtime(&time1);

#pragma omp parallel if (omp_get_max_threads() > 1)
    {
        int     Nloop, spin, Mc_AN, h_AN, Gh_AN, Mh_AN, Hwan, NOLG;
        int     Gc_AN, Cwan, NO0, NO1, spin0 = -1, Mc_AN0 = 0;
        int     i, j, Nc, MN, GNA, Nog, Nh, OMPID, Nthrds;
        int     M, N, K, lda, ldb, ldc, ii, jj;
        double  alpha, beta, Vpot;
        double  sum0, sum1, sum2, sum3, sum4;
        double *ChiV0, *Chi1, *ChiV0_2, *C;

        /* allocation of arrays */

        /* AITUNE */
        double ** AI_tmpH[4];
        {
            /* get size of temporary buffer */
            int AI_MaxNO = 0;
            if (Cnt_kind == 0) {
                int spe;
                for (spe = 0; spe < SpeciesNum; spe++) {
                    if (AI_MaxNO < Spe_Total_NO[spe]) {
                        AI_MaxNO = Spe_Total_NO[spe];
                    }
                }
            } else {
                int spe;
                for (spe = 0; spe < SpeciesNum; spe++) {
                    if (AI_MaxNO < Spe_Total_CNO[spe]) {
                        AI_MaxNO = Spe_Total_CNO[spe];
                    }
                }
            }

            int spin;
            for (spin = 0; spin <= SpinP_switch; spin++) {
                size_t ai_maxno = (size_t)AI_MaxNO;
                size_t elems;

                if (AI_MaxNO <= 0) {
                    Set_Hamiltonian_abort("Calc_MatrixElements_dVH_Vxc_VNA", "non-positive orbital buffer size",
                                          myid);
                }

                if (ai_maxno > ((size_t)-1) / sizeof(double *)) {
                    Set_Hamiltonian_abort("Calc_MatrixElements_dVH_Vxc_VNA", "row pointer size overflow", myid);
                }

                if (ai_maxno > ((size_t)-1) / sizeof(double) / ai_maxno) {
                    Set_Hamiltonian_abort("Calc_MatrixElements_dVH_Vxc_VNA", "matrix buffer size overflow", myid);
                }

                elems = ai_maxno * ai_maxno;

                AI_tmpH[spin] = (double **)malloc(sizeof(double *) * ai_maxno);

                int      i;
                double * p = (double *)malloc(sizeof(double) * elems);

                if (AI_tmpH[spin] == NULL || p == NULL) {
                    free(AI_tmpH[spin]);
                    free(p);
                    Set_Hamiltonian_abort("Calc_MatrixElements_dVH_Vxc_VNA", "failed to allocate matrix buffer",
                                          myid);
                }

                for (i = 0; i < AI_MaxNO; i++) {
                    AI_tmpH[spin][i] = p;
                    p += AI_MaxNO;
                }
            }
        }
        /* AITUNE */

        /* starting of one-dimensionalized loop */

#pragma omp for schedule(static, 1) /* guided */       /* AITUNE */
        for (Nloop = 0; Nloop < OneD_Nloop; Nloop++) { /* AITUNE */

            int Mc_AN = OneD2Mc_AN[Nloop];
            int h_AN  = OneD2h_AN[Nloop];
            int Gc_AN = M2G[Mc_AN];
            int Gh_AN = natn[Gc_AN][h_AN];
            int Mh_AN = F_G2M[Gh_AN];
            int Cwan  = WhatSpecies[Gc_AN];
            int Hwan  = WhatSpecies[Gh_AN];
            int GNA   = GridN_Atom[Gc_AN];
            int NOLG  = NumOLG[Mc_AN][h_AN];
            int Gh_AN_is_local = (G2ID[Gh_AN] == myid);

            int NO0, NO1;
            if (Cnt_kind == 0) {
                NO0 = Spe_Total_NO[Cwan];
                NO1 = Spe_Total_NO[Hwan];
            } else {
                NO0 = Spe_Total_CNO[Cwan];
                NO1 = Spe_Total_CNO[Hwan];
            }

            /* quadrature for Hij  */

            /* AITUNE change order of loop */
            if (SpinP_switch == 0 && NO0 <= SETH_ME_MAXNO && NO1 <= SETH_ME_MAXNO) {
                double acc[SETH_ME_MAXNO * SETH_ME_MAXNO] __attribute__((aligned(64)));
                int    i, j;

                spin0_pair_function(
                    Mc_AN,h_AN,Mh_AN,Gh_AN_is_local,NO0,NO1,NOLG,acc);

                if (Cnt_kind == 0) {
                    for (i = 0; i < NO0; i++) {
                        for (j = 0; j < NO1; j++) {
                            H[0][Mc_AN][h_AN][i][j] += acc[i * SETH_ME_MAXNO + j];
                        }
                    }
                } else {
                    for (i = 0; i < NO0; i++) {
                        for (j = 0; j < NO1; j++) {
                            CntH[0][Mc_AN][h_AN][i][j] += acc[i * SETH_ME_MAXNO + j];
                        }
                    }
                }

            } else if (SpinP_switch == 0) {
                /* AITUNE temporary buffer for "unroll-Jammed" HLO optimization by Intel */

                if (Cnt_kind == 0) {
                    int i;
                    for (i = 0; i < NO0; i++) {
                        int j;
                        for (j = 0; j < NO1; j++) {
                            AI_tmpH[0][i][j] = H[0][Mc_AN][h_AN][i][j];
                        }
                    }
                } else {
                    int i;
                    for (i = 0; i < NO0; i++) {
                        int j;
                        for (j = 0; j < NO1; j++) {
                            AI_tmpH[0][i][j] = CntH[0][Mc_AN][h_AN][i][j];
                        }
                    }
                }

                int Nog;
                for (Nog = 0; Nog < NOLG; Nog++) {

                    int Nc = GListTAtoms1[Mc_AN][h_AN][Nog];
                    int MN = MGridListAtom[Mc_AN][Nc];
                    int Nh = GListTAtoms2[Mc_AN][h_AN][Nog];
                    Type_Orbs_Grid *orbs1 =
                        Gh_AN_is_local ? Orbs_Grid[Mh_AN][Nh] : Orbs_Grid_FNAN[Mc_AN][h_AN][Nog];
                    Type_Orbs_Grid *orbs0 = Orbs_Grid[Mc_AN][Nc];

                    double AI_tmp_GVVG = GridVol * Vpot_Grid[0][MN];

                    int i;
                    for (i = 0; i < NO0; i++) {

                        double AI_tmp_i = AI_tmp_GVVG * orbs0[i];
                        double *tmp0 = AI_tmpH[0][i];
                        int    j;

                        for (j = 0; j < NO1; j++) {
                            tmp0[j] += AI_tmp_i * orbs1[j];
                        }
                    }

                } /* Nog */

                if (Cnt_kind == 0) {
                    int i;
                    for (i = 0; i < NO0; i++) {
                        int j;
                        for (j = 0; j < NO1; j++) {
                            H[0][Mc_AN][h_AN][i][j] = AI_tmpH[0][i][j];
                        }
                    }
                } else {
                    int i;
                    for (i = 0; i < NO0; i++) {
                        int j;
                        for (j = 0; j < NO1; j++) {
                            CntH[0][Mc_AN][h_AN][i][j] = AI_tmpH[0][i][j];
                        }
                    }
                }

            } else if (SpinP_switch == 1) {

                /* AITUNE temporary buffer for "unroll-Jammed" HLO optimization by Intel */

                if (Cnt_kind == 0) {
                    int i;
                    for (i = 0; i < NO0; i++) {
                        int j;
                        for (j = 0; j < NO1; j++) {
                            AI_tmpH[0][i][j] = H[0][Mc_AN][h_AN][i][j];
                            AI_tmpH[1][i][j] = H[1][Mc_AN][h_AN][i][j];
                        }
                    }
                } else {
                    int i;
                    for (i = 0; i < NO0; i++) {
                        int j;
                        for (j = 0; j < NO1; j++) {
                            AI_tmpH[0][i][j] = CntH[0][Mc_AN][h_AN][i][j];
                            AI_tmpH[1][i][j] = CntH[1][Mc_AN][h_AN][i][j];
                        }
                    }
                }

                spin1_pair_function(Mc_AN,h_AN,Mh_AN,Gh_AN_is_local,
                                    NO0,NO1,NOLG,AI_tmpH[0],AI_tmpH[1]);

                /* AITUNE copy from temporary buffer */

                if (Cnt_kind == 0) {
                    int i;
                    for (i = 0; i < NO0; i++) {
                        int j;
                        for (j = 0; j < NO1; j++) {
                            H[0][Mc_AN][h_AN][i][j] = AI_tmpH[0][i][j];
                            H[1][Mc_AN][h_AN][i][j] = AI_tmpH[1][i][j];
                        }
                    }
                } else {
                    int i;
                    for (i = 0; i < NO0; i++) {
                        int j;
                        for (j = 0; j < NO1; j++) {
                            CntH[0][Mc_AN][h_AN][i][j] = AI_tmpH[0][i][j];
                            CntH[1][Mc_AN][h_AN][i][j] = AI_tmpH[1][i][j];
                        }
                    }
                }

            }

            else { /* SpinP_switch==3 */

                /* AITUNE temporary buffer for "unroll-Jammed" HLO optimization by Intel */

                if (Cnt_kind == 0) {
                    int i;
                    for (i = 0; i < NO0; i++) {
                        int j;
                        for (j = 0; j < NO1; j++) {
                            AI_tmpH[0][i][j] = H[0][Mc_AN][h_AN][i][j];
                            AI_tmpH[1][i][j] = H[1][Mc_AN][h_AN][i][j];
                            AI_tmpH[2][i][j] = H[2][Mc_AN][h_AN][i][j];
                            AI_tmpH[3][i][j] = H[3][Mc_AN][h_AN][i][j];
                        }
                    }
                } else {
                    int i;
                    for (i = 0; i < NO0; i++) {
                        int j;
                        for (j = 0; j < NO1; j++) {
                            AI_tmpH[0][i][j] = CntH[0][Mc_AN][h_AN][i][j];
                            AI_tmpH[1][i][j] = CntH[1][Mc_AN][h_AN][i][j];
                            AI_tmpH[2][i][j] = CntH[2][Mc_AN][h_AN][i][j];
                            AI_tmpH[3][i][j] = CntH[3][Mc_AN][h_AN][i][j];
                        }
                    }
                }

                nc4_pair_function(Mc_AN,h_AN,Mh_AN,Gh_AN_is_local,
                                  NO0,NO1,NOLG,AI_tmpH[0],AI_tmpH[1],
                                  AI_tmpH[2],AI_tmpH[3]);

                /* AITUNE copy from temporary buffer */

                if (Cnt_kind == 0) {
                    int i;
                    for (i = 0; i < NO0; i++) {
                        int j;
                        for (j = 0; j < NO1; j++) {
                            H[0][Mc_AN][h_AN][i][j] = AI_tmpH[0][i][j];
                            H[1][Mc_AN][h_AN][i][j] = AI_tmpH[1][i][j];
                            H[2][Mc_AN][h_AN][i][j] = AI_tmpH[2][i][j];
                            H[3][Mc_AN][h_AN][i][j] = AI_tmpH[3][i][j];
                        }
                    }
                } else {
                    int i;
                    for (i = 0; i < NO0; i++) {
                        int j;
                        for (j = 0; j < NO1; j++) {
                            CntH[0][Mc_AN][h_AN][i][j] = AI_tmpH[0][i][j];
                            CntH[1][Mc_AN][h_AN][i][j] = AI_tmpH[1][i][j];
                            CntH[2][Mc_AN][h_AN][i][j] = AI_tmpH[2][i][j];
                            CntH[3][Mc_AN][h_AN][i][j] = AI_tmpH[3][i][j];
                        }
                    }
                }
            }
            /* AITUNE change order of loop */

        } /* Nloop */

        /* freeing of arrays */
        {
            int spin;
            for (spin = 0; spin <= SpinP_switch; spin++) {
                free(AI_tmpH[spin][0]);
                free(AI_tmpH[spin]);
            }
        }
    } /* pragma omp parallel */

    /* freeing of arrays */

    free(OneD2Mc_AN);
    free(OneD2h_AN);

    if (measure_time) {
        dtime(&time2);
        printf("myid=%4d Time4=%18.10f\n", myid, time2 - time1);
        fflush(stdout);
    }
}

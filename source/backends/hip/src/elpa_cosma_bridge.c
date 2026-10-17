#include "openmx_stdout_level.h"
/**********************************************************************
  elpa_cosma_bridge.c

  Implementation of the OpenMX <-> ELPA (2026.02, AMD GPU kernels)
  bridge used by scf.eigen.lib=gpusolver2.  See elpa_cosma_bridge.h.
***********************************************************************/

#include "elpa_cosma_bridge.h"

#include <elpa/elpa.h>

#include "hip_runtime_compat.h"
#include "set_hip_default_device_from_local_rank.h"
#include "openmx_stdout_level.h"
#include "openmx_host_memory.h"
#include "openmx_mgpu_runtime.h"
#include <mpi.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern void Cblacs_gridinfo(int ictxt, int *nprow, int *npcol, int *myrow, int *mycol);
extern void Cblacs_get(int ictxt, int what, int *val);
extern MPI_Comm Cblacs2sys_handle(int ictxt);
extern int numroc_(int *n, int *nb, int *iproc, int *isrcproc, int *nprocs);
extern int SCF_Gpu_Num;

static int gs2_elpa_inited = 0;

/* set while the cluster solvers use this bridge as the automatic capacity
   fallback of scf.eigen.lib=gpusolver: a solve or transform that the GPU
   cannot hold (and so runs on the CPU side) is then reported at
   level.of.stdout 1 instead of 2 */
static int gs2_fallback_report = 0;

/* one cached, fully set-up ELPA handle per (BLACS context, n, nev, type, nblk) */
#define GS2_MAX_HANDLES 16
typedef struct {
    int used, ictxt, n, nev, is_complex, nblk;
    int gpu;          /* 1 = AMD GPU kernels, 0 = CPU kernels */
    int ranks_on_dev; /* ranks of comm that share this rank's device */
    size_t need;      /* estimated per-rank device bytes of one solve */
    size_t retained;  /* device bytes the handle keeps between solves */
    MPI_Comm comm;    /* the grid's parent communicator */
    elpa_t h;
} gs2_handle_slot;
static gs2_handle_slot gs2_slots[GS2_MAX_HANDLES];

static long gs2_env_long(const char *name, long defval)
{
    const char *env = getenv(name);
    if (env != NULL && env[0] != '\0') {
        long v = atol(env);
        if (v >= 0) return v;
    }
    return defval;
}

/* ---------------------------------------------------------------------
   GPU-memory preflight.

   ELPA kills the whole run ("stop 1") when a device allocation fails
   inside a solve, so the GPU/CPU choice has to be made before the solver
   is entered: estimate what the solve will allocate on the device, and
   fall back to ELPA's CPU kernels when the GPU cannot hold it.  The
   verdict is re-evaluated at every solve (the resident caches of other
   OpenMX GPU stages grow during the first SCF iterations) and reduced
   with MPI_MIN so that every rank of the grid takes the same branch.
   --------------------------------------------------------------------- */

typedef struct {
    unsigned long long need_mb, free_mb, reserve_mb;
    int ranks, forced;
    /* host side of a COSMA transform (gs2_host_verdict): what COSMA would
       keep per rank, the ranks of the busiest node, the reserve and the
       least available memory of the grid's nodes; host_limited when that
       verdict, not the GPU, sent the transform to ScaLAPACK (2: COSMA
       cannot build a strategy within COSMA_CPU_MAX_MEMORY) */
    unsigned long long host_need_mb, host_avail_mb, host_reserve_mb;
    int host_ranks, host_limited;
} gs2_mem_details;

/* -1 = no forced mode, 0 = forced CPU, 1 = forced GPU */
static int gs2_forced_mode(const char *envname)
{
    const char *env = getenv(envname);
    if (env == NULL || env[0] == '\0') return -1;
    return (atoi(env) != 0);
}

/* A local override must not let one rank skip the admission collectives
   while its peers enter them, or select different ELPA/COSMA kernels. */
static int gs2_forced_mode_collective(const char *envname, MPI_Comm comm)
{
    const int mode = gs2_forced_mode(envname);
    int bounds[2] = {mode, -mode};

    MPI_Allreduce(MPI_IN_PLACE, bounds, 2, MPI_INT, MPI_MIN, comm);
    if (bounds[0] != -bounds[1]) {
        int rank;
        MPI_Comm_rank(comm, &rank);
        if (rank == 0) {
            fprintf(stderr, "openmx gpusolver2: inconsistent %s across ranks of the grid;"
                            " set the same kernel mode on every rank.\n", envname);
            fflush(stderr);
        }
        MPI_Abort(comm, 1);
    }
    return mode;
}

/* Local handle allocation/parameter errors must reach every peer before
   anybody enters ELPA's collective setup or eigenvector routines. Preserve
   the first failing rank's error code, including negative ELPA codes. */
static int gs2_collective_error(int error, MPI_Comm comm)
{
    int rank, first;
    MPI_Comm_rank(comm, &rank);
    first = error == ELPA_OK ? INT_MAX : rank;
    MPI_Allreduce(MPI_IN_PLACE, &first, 1, MPI_INT, MPI_MIN, comm);
    if (first == INT_MAX) return ELPA_OK;
    MPI_Bcast(&error, 1, MPI_INT, first, comm);
    return error;
}

/* Saturation makes an unrepresentable memory demand fail admission. */
static size_t gs2_bytes_add(size_t a, size_t b)
{
    return b > SIZE_MAX - a ? SIZE_MAX : a + b;
}

static size_t gs2_bytes_mul(size_t a, size_t b)
{
    return b && a > SIZE_MAX / b ? SIZE_MAX : a * b;
}

static void gs2_det_set_forced(gs2_mem_details *det)
{
    det->need_mb = det->free_mb = det->reserve_mb = 0;
    det->ranks = 0;
    det->forced = 1;
    det->host_limited = 0;
}

/* per-rank device bytes one ELPA solve is expected to allocate: the
   dominant terms are a few copies of the local block-cyclic matrix
   (a_dev/q_dev plus bandred/back-transform panels), modelled as
   FACTOR x local matrix, plus a fixed head (cuBLAS/cuSOLVER handles,
   n*nbw broadcast buffers, allocator slack).  Calibrated against the
   measured 2-stage peak of ~2 x local + ~110 MB per rank (sidia333
   n=5616 complex, np8, RTX 5080). */
static size_t gs2_need_bytes(int na_rows, int na_cols, int is_complex)
{
    size_t elem   = is_complex ? 16 : 8;
    size_t local  = gs2_bytes_mul(gs2_bytes_mul((size_t)na_rows, (size_t)na_cols), elem);
    size_t factor = (size_t)gs2_env_long("OPENMX_GS2_GPU_FACTOR", 3);
    size_t fixed  = gs2_bytes_mul((size_t)gs2_env_long("OPENMX_GS2_GPU_FIXED_MB", 128), 1048576);
    return gs2_bytes_add(gs2_bytes_mul(factor, local), fixed);
}

/* how many ranks end up on this rank's HIP device (they all run the
   same solve concurrently, so the free memory has to cover all of them).
   Collective over comm. */
static int gs2_ranks_on_device(MPI_Comm comm)
{
    int mydev = -1, ndev = 0, cnt = 1, est = 1, i, nl = 1;
    MPI_Comm shm;

    if (hipGetDevice(&mydev) != hipSuccess) mydev = -1;

    /* CUDA ordinals are process-local; use the physical UUID even when
       ranks see the allocated GPUs in different orders. */
    if (MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shm)
        == MPI_SUCCESS) {
        unsigned char uuid[16], *devices;
        int census_ok;
        MPI_Comm_size(shm, &nl);
        devices = (unsigned char *)malloc((size_t)nl * sizeof(uuid));
        census_ok = devices != NULL && mydev >= 0 &&
                    openmx_gpu_physical_uuid(mydev, uuid);
        if (!census_ok) (void)hipGetLastError();
        /* All peers must either gather or use the conservative bound. A
           malloc/query failure on just one rank must not strand the rest. */
        MPI_Allreduce(MPI_IN_PLACE, &census_ok, 1, MPI_INT, MPI_MIN, shm);
        cnt = nl;
        if (census_ok) {
            MPI_Allgather(uuid, sizeof(uuid), MPI_BYTE, devices, sizeof(uuid), MPI_BYTE, shm);
            for (cnt = 0, i = 0; i < nl; i++) {
                if (memcmp(devices + (size_t)i * sizeof(uuid), uuid, sizeof(uuid)) == 0) cnt++;
            }
        }
        free(devices);
        MPI_Comm_free(&shm);
    }

    /* comm may span only part of the node (the spin-split worlds of the
       collinear cluster solver run concurrently), so also estimate from
       the launcher's node-local size and keep the larger count */
    {
        const int ls = openmx_gpu_local_size_noncollective();
        if (mydev >= 0 &&
            hipGetDeviceCount(&ndev) == hipSuccess && ndev > 0) {
            if (SCF_Gpu_Num > 0 && SCF_Gpu_Num < ndev) ndev = SCF_Gpu_Num;
            for (est = 0, i = 0; i < ls; i++) {
                if (openmx_gpu_map_rank_to_device(i, ls, ndev) == mydev) est++;
            }
        }
        else {
            (void)hipGetLastError();
            est = ls;
        }
    }

    if (est > cnt) cnt = est;
    return (cnt > 0) ? cnt : 1;
}

/* uniform verdict: 1 when every rank of comm has room to run the solve
   on the GPU, 0 when at least one has not (-> CPU kernels everywhere;
   mixed settings are not allowed by ELPA).  current_mode damps
   oscillation: promoting a CPU handle back to the GPU needs 25% extra
   headroom.  Collective over comm unless forced by environment. */
static int gs2_gpu_verdict(size_t need, int ranks_on_dev, MPI_Comm comm,
                           int current_mode, size_t retained, gs2_mem_details *det)
{
    unsigned long long buf[2];
    size_t free_b = 0, total_b = 0, reserve, want_free;
    int ok;

    det->need_mb    = (unsigned long long)(need >> 20);
    det->ranks      = ranks_on_dev;
    det->reserve_mb = 0;
    det->free_mb    = 0;
    det->forced     = 0;
    det->host_limited = 0;

    reserve = gs2_bytes_mul((size_t)gs2_env_long("OPENMX_GS2_GPU_RESERVE_MB", 512), 1048576);
    det->reserve_mb = (unsigned long long)(reserve >> 20);

    want_free = gs2_bytes_add(gs2_bytes_mul(need, (size_t)ranks_on_dev), reserve);
    if (current_mode == 0) want_free = gs2_bytes_add(want_free, want_free / 4);

    if (hipMemGetInfo(&free_b, &total_b) != hipSuccess) {
        free_b = 0;
        ok = 0;
    }
    else {
        /* memory a live GPU handle keeps between solves is reused by the
           next solve, so count it as available rather than as consumed */
        free_b = gs2_bytes_add(free_b, retained);
        if (free_b > total_b) free_b = total_b;
        ok = (want_free != SIZE_MAX && want_free <= free_b);
    }

    buf[0] = (unsigned long long)ok;
    buf[1] = (unsigned long long)free_b;
    MPI_Allreduce(MPI_IN_PLACE, buf, 2, MPI_UNSIGNED_LONG_LONG, MPI_MIN, comm);

    det->free_mb = buf[1] >> 20;
    return (buf[0] != 0ULL);
}

static void gs2_report_mode(MPI_Comm comm, int n, int nev, int gpu,
                            const gs2_mem_details *det, const char *how)
{
    int r = 0;
    MPI_Comm_rank(comm, &r);
    if (r != 0 || !(OPENMX_STDOUT_LEVEL(2) ||
                    (gs2_fallback_report && !gpu && OPENMX_STDOUT_LEVEL(1)))) return;

    if (det->forced) {
        printf("<DFT> gpusolver2: ELPA n=%d nev=%d %s %s kernels (OPENMX_GS2_ELPA_GPU)\n",
               n, nev, how, gpu ? "GPU" : "CPU");
    }
    else if (n < 256 && !gpu) {
        printf("<DFT> gpusolver2: ELPA n=%d nev=%d %s CPU kernels (small matrix, n < 256)\n",
               n, nev, how);
    }
    else {
        printf("<DFT> %sgpusolver2: ELPA n=%d nev=%d %s %s kernels "
               "(GPU memory: need ~%llu MB/rank x %d ranks/GPU + reserve %llu MB %s free %llu MB)\n",
               (gs2_fallback_report && !gpu) ? "Warning: " : "", n, nev, how, gpu ? "GPU" : "CPU",
               det->need_mb, det->ranks, det->reserve_mb,
               gpu ? "<=" : ">", det->free_mb);
    }
    fflush(stdout);
}

static int gs2_elpa_init_once(void)
{
    if (gs2_elpa_inited) return 0;

    if (elpa_init(ELPA_API_VERSION) != ELPA_OK) {
        fprintf(stderr, "openmx gpusolver2: elpa_init failed (API %d unsupported)\n",
                ELPA_API_VERSION);
        return -1;
    }
    gs2_elpa_inited = 1;

    {
        int myid = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &myid);
        if (myid == 0 && OPENMX_STDOUT_LEVEL(2)) {
            printf("<DFT> gpusolver2: ELPA (API %d) with AMD GPU kernels + COSMA initialized\n",
                   ELPA_API_VERSION);
            fflush(stdout);
        }
    }
    return 0;
}

/* build (or fetch) the handle slot for this solve configuration */
static gs2_handle_slot *gs2_get_handle(int ictxt, int n, int nev, int is_complex,
                                       int nblk, int *perr)
{
    int i, err = ELPA_OK;
    int nprow, npcol, myrow, mycol, ZERO = 0;
    int na_rows, na_cols, sysctxt;
    int want_gpu = 1, have_verdict = 0, ranks_on_dev = 1;
    int forced;
    size_t need = 0;
    gs2_mem_details det = {0};
    MPI_Comm comm;
    elpa_t h;

    *perr = 0;
    Cblacs_get(ictxt, 10, &sysctxt);
    comm = Cblacs2sys_handle(sysctxt);
    forced = gs2_forced_mode_collective("OPENMX_GS2_ELPA_GPU", comm);

    for (i = 0; i < GS2_MAX_HANDLES; i++) {
        if (gs2_slots[i].used && gs2_slots[i].ictxt == ictxt && gs2_slots[i].n == n &&
            gs2_slots[i].nev == nev && gs2_slots[i].is_complex == is_complex &&
            gs2_slots[i].nblk == nblk) {
            break;
        }
    }
    {
        int cached = i < GS2_MAX_HANDLES;
        if (cached) {
            int relation = MPI_UNEQUAL;
            MPI_Comm_compare(gs2_slots[i].comm, comm, &relation);
            cached = relation == MPI_IDENT || relation == MPI_CONGRUENT;
        }
        /* Intervening subgroup solves may evict only some ranks' handles.
           Either every rank reuses its handle or every rank rebuilds it. */
        MPI_Allreduce(MPI_IN_PLACE, &cached, 1, MPI_INT, MPI_MIN, comm);
        if (cached) {
            /* the memory situation changes while a handle is cached (the
               resident caches of other GPU stages keep growing), so
               re-evaluate the GPU/CPU verdict before every solve and
               rebuild the handle on the other side when it flips */
            if (forced >= 0) {
                want_gpu = forced;
                gs2_det_set_forced(&det);
            }
            else if (n < 256) {
                /* same small-solve floor as the first verdict below */
                want_gpu = 0;
            }
            else {
                want_gpu = gs2_gpu_verdict(gs2_slots[i].need, gs2_slots[i].ranks_on_dev,
                                           comm, gs2_slots[i].gpu,
                                           gs2_slots[i].retained, &det);
            }
            int reuse = want_gpu == gs2_slots[i].gpu;
            MPI_Allreduce(MPI_IN_PLACE, &reuse, 1, MPI_INT, MPI_MIN, comm);
            if (reuse) return &gs2_slots[i];

            gs2_report_mode(comm, n, nev, want_gpu, &det, "switching to");
            have_verdict = 1;
        }
    }

    /* a different solve configuration is starting: release every cached
       handle first, so that only one ELPA handle holds GPU memory at a time
       (the S-stage handle would otherwise stay resident for the whole run
       and can exhaust small GPUs in the non-collinear 2n solves) */
    for (i = 0; i < GS2_MAX_HANDLES; i++) {
        if (gs2_slots[i].used) {
            int derr;
            elpa_deallocate(gs2_slots[i].h, &derr);
            gs2_slots[i].used = 0;
        }
    }

    err = gs2_collective_error(gs2_elpa_init_once(), comm);
    if (err != ELPA_OK) { *perr = err; return NULL; }

    Cblacs_gridinfo(ictxt, &nprow, &npcol, &myrow, &mycol);
    Cblacs_get(ictxt, 10, &sysctxt); /* SGET_BLACSCONTXT: system context */
    comm = Cblacs2sys_handle(sysctxt);

    na_rows = numroc_(&n, &nblk, &myrow, &ZERO, &nprow);
    na_cols = numroc_(&n, &nblk, &mycol, &ZERO, &npcol);

    ranks_on_dev = gs2_ranks_on_device(comm);
    need = gs2_need_bytes(na_rows, na_cols, is_complex);
    if (!have_verdict) {
        if (forced >= 0) {
            want_gpu = forced;
            gs2_det_set_forced(&det);
        }
        else if (n < 256) {
            /* tiny solves stay on the CPU kernels: the GPU 2-stage solver
               cannot win at this size, and ELPA's ROCm layer pins zero-length
               buffers for them, which hipHostMalloc rejects (CUDA tolerates
               the same call).  n is global, so every rank agrees. */
            want_gpu = 0;
        }
        else {
            want_gpu = gs2_gpu_verdict(need, ranks_on_dev, comm, -1, 0, &det);
        }
        if (!want_gpu || gs2_env_long("OPENMX_GS2_GPU_VERBOSE", 0) != 0) {
            gs2_report_mode(comm, n, nev, want_gpu, &det, "using");
        }
    }

    h = elpa_allocate(&err);
    if (err == ELPA_OK && h == NULL) err = -1;
    err = gs2_collective_error(err, comm);
    if (err != ELPA_OK) {
        *perr = err;
        if (h != NULL) { int derr; elpa_deallocate(h, &derr); }
        return NULL;
    }

    elpa_set_integer(h, "na", n, &err);
    if (err == ELPA_OK) elpa_set_integer(h, "nev", nev, &err);
    if (err == ELPA_OK) elpa_set_integer(h, "local_nrows", na_rows, &err);
    if (err == ELPA_OK) elpa_set_integer(h, "local_ncols", na_cols, &err);
    if (err == ELPA_OK) elpa_set_integer(h, "nblk", nblk, &err);
    if (err == ELPA_OK) elpa_set_integer(h, "mpi_comm_parent", (int)MPI_Comm_c2f(comm), &err);
    if (err == ELPA_OK) elpa_set_integer(h, "process_row", myrow, &err);
    if (err == ELPA_OK) elpa_set_integer(h, "process_col", mycol, &err);
    /* GPU selection: must be requested before elpa_setup so that the setup
       binds the device.  The device itself was assigned rank-wise by the
       central GPUSOLVER initialization (shared rank map, scf.Gpu.Num).
       When the preflight above found too little free device memory, run
       this configuration on ELPA's CPU kernels instead. */
    if (err == ELPA_OK) elpa_set_integer(h, "amd-gpu", want_gpu ? 1 : 0, &err);
    if (err == ELPA_OK && want_gpu) {
        int dev = 0, seterr = ELPA_OK;
        if (hipGetDevice(&dev) == hipSuccess) {
            elpa_set_integer(h, "use_gpu_id", dev, &seterr); /* optional */
        }
    }
    err = gs2_collective_error(err, comm);
    if (err != ELPA_OK) goto setup_failed;

    err = gs2_collective_error(elpa_setup(h), comm);
    if (err != ELPA_OK) goto setup_failed;

    {
        int solver = ELPA_SOLVER_2STAGE;
        const char *env = getenv("OPENMX_GS2_ELPA_SOLVER");
        if (env != NULL && atoi(env) == 1) solver = ELPA_SOLVER_1STAGE;
        elpa_set_integer(h, "solver", solver, &err);
        err = gs2_collective_error(err, comm);
        if (err != ELPA_OK) goto setup_failed;
    }

    for (i = 0; i < GS2_MAX_HANDLES; i++) {
        if (!gs2_slots[i].used) break;
    }
    if (i == GS2_MAX_HANDLES) { /* should not happen; recycle the first slot */
        int derr;
        elpa_deallocate(gs2_slots[0].h, &derr);
        i = 0;
    }
    gs2_slots[i].used = 1;
    gs2_slots[i].ictxt = ictxt;
    gs2_slots[i].n = n;
    gs2_slots[i].nev = nev;
    gs2_slots[i].is_complex = is_complex;
    gs2_slots[i].nblk = nblk;
    gs2_slots[i].gpu = want_gpu;
    gs2_slots[i].ranks_on_dev = ranks_on_dev;
    gs2_slots[i].need = need;
    gs2_slots[i].retained = 0;
    gs2_slots[i].comm = comm;
    gs2_slots[i].h = h;

    return &gs2_slots[i];

setup_failed:
    *perr = err;
    { int derr; elpa_deallocate(h, &derr); }
    return NULL;
}

/* update the slot's estimate of how much device memory its handle kept
   allocated across the solve (sampled as the free-memory drop over the
   solve; the diagonalization phase runs exclusively, so the drop is
   attributable to ELPA) */
static size_t gs2_free_now(void)
{
    size_t free_b = 0, total_b = 0;
    if (hipMemGetInfo(&free_b, &total_b) != hipSuccess) return 0;
    return free_b;
}

static void gs2_update_retained(gs2_handle_slot *s, size_t free_before, size_t free_after)
{
    if (!s->gpu) { s->retained = 0; return; }
    if (free_before > free_after && free_before - free_after > s->retained) {
        s->retained = free_before - free_after;
    }
}

int openmx_gs2_eigen_real(int n, int nev, double *a, int *desca, double *w, double *z,
                          int *descz)
{
    int err = 0;
    size_t free_before;
    gs2_handle_slot *s = gs2_get_handle(desca[1], n, nev, 0, desca[4], &err);

    (void)descz;
    if (s == NULL) return (err != 0) ? err : -1;

    free_before = gs2_free_now();
    elpa_eigenvectors_double(s->h, a, w, z, &err);
    if (err != ELPA_OK) {
        fprintf(stderr, "openmx gpusolver2: elpa_eigenvectors_double failed: %s\n",
                elpa_strerr(err));
        return err;
    }
    gs2_update_retained(s, free_before, gs2_free_now());
    return 0;
}

int openmx_gs2_eigen_complex(int n, int nev, void *a, int *desca, double *w, void *z,
                             int *descz)
{
    int err = 0;
    size_t free_before;
    gs2_handle_slot *s = gs2_get_handle(desca[1], n, nev, 1, desca[4], &err);

    (void)descz;
    if (s == NULL) return (err != 0) ? err : -1;

    free_before = gs2_free_now();
    elpa_eigenvectors_double_complex(s->h, (double complex *)a, w, (double complex *)z, &err);
    if (err != ELPA_OK) {
        fprintf(stderr, "openmx gpusolver2: elpa_eigenvectors_double_complex failed: %s\n",
                elpa_strerr(err));
        return err;
    }
    gs2_update_retained(s, free_before, gs2_free_now());
    return 0;
}

/* ---------------------------------------------------------------------
   COSMA <-> ScaLAPACK p?gemm dispatch.

   COSMA's GPU backend (Tiled-MM) allocates three device buffers x
   streams per scalar type, sized by its GEMM tiles, and keeps them for
   the rest of the run.  With many ranks per GPU those buffers are what
   actually overflows the card (observed at np18: every rank printing
   "error: GPU API call : out of memory" inside the first n2 x n2
   back-transform), so the transform GEMMs get the same treatment as
   the eigensolves: bound what Tiled-MM may allocate and, when the
   device cannot hold it, run the ScaLAPACK p?gemm (CPU) instead.

   To keep that bound tight the bridge caps COSMA's tile sizes (default
   1024) before the first COSMA context is created; the transforms are
   PCIe-transfer-bound at any tile size, so smaller tiles cost little.
   --------------------------------------------------------------------- */

extern void pdgemm_(const char *transa, const char *transb, const int *m, const int *n,
                    const int *k, const double *alpha, const double *a, const int *ia,
                    const int *ja, const int *desca, const double *b, const int *ib,
                    const int *jb, const int *descb, const double *beta, double *c,
                    const int *ic, const int *jc, const int *descc);
extern void pzgemm_(const char *transa, const char *transb, const int *m, const int *n,
                    const int *k, const double *alpha, const double *a, const int *ia,
                    const int *ja, const int *desca, const double *b, const int *ib,
                    const int *jb, const int *descb, const double *beta, double *c,
                    const int *ic, const int *jc, const int *descc);

/* host bytes per rank that cosma::pxgemm keeps for this product, -1 when
   COSMA has no strategy within COSMA_CPU_MAX_MEMORY (elpa_cosma_memory.cpp) */
extern double openmx_gs2_cosma_host_bytes(char transa, char transb, int m, int n, int k,
                                          const int *desca, const int *descb,
                                          const int *descc, int ictxt, int is_complex);

static long gs2_tile_cap = 5000;    /* effective COSMA max tile dimension */
static long gs2_gemm_streams = 2;   /* effective COSMA GPU stream count */
static int gs2_gemm_ranks = 0;      /* ranks sharing this rank's device */
static int gs2_gemm_ictxt = -1;     /* grid whose sharing count is cached */
static size_t gs2_gemm_retained[2]; /* persistent Tiled-MM bytes: [real, cplx] */
static int gs2_gemm_mode[2] = {-1, -1};

static void gs2_gemm_init_once(void)
{
    static int done = 0;
    char buf[32];
    long cap;

    if (done) return;
    done = 1;

    cap = gs2_env_long("OPENMX_GS2_GPU_TILE", 1024);
    if (cap > 0) {
        snprintf(buf, sizeof(buf), "%ld", cap);
        setenv("COSMA_GPU_MAX_TILE_M", buf, 0);
        setenv("COSMA_GPU_MAX_TILE_N", buf, 0);
        setenv("COSMA_GPU_MAX_TILE_K", buf, 0);
    }

    /* the caps actually in effect (a user-set environment wins above) */
    gs2_tile_cap = gs2_env_long("COSMA_GPU_MAX_TILE_M", 5000);
    {
        long capn = gs2_env_long("COSMA_GPU_MAX_TILE_N", 5000);
        long capk = gs2_env_long("COSMA_GPU_MAX_TILE_K", 5000);
        if (capn > gs2_tile_cap) gs2_tile_cap = capn;
        if (capk > gs2_tile_cap) gs2_tile_cap = capk;
    }
    gs2_gemm_streams = gs2_env_long("COSMA_GPU_STREAMS", 2);
    if (gs2_gemm_streams < 1) gs2_gemm_streams = 1;
}

/* upper bound of the device bytes Tiled-MM may hold after one GEMM */
static size_t gs2_gemm_bound(int m, int n, int k, int is_complex)
{
    size_t elem = is_complex ? 16 : 8;
    size_t tm = (m < gs2_tile_cap) ? (size_t)m : (size_t)gs2_tile_cap;
    size_t tn = (n < gs2_tile_cap) ? (size_t)n : (size_t)gs2_tile_cap;
    size_t tk = (k < gs2_tile_cap) ? (size_t)k : (size_t)gs2_tile_cap;
    size_t tiles = gs2_bytes_add(gs2_bytes_add(gs2_bytes_mul(tm, tk), gs2_bytes_mul(tk, tn)),
                                gs2_bytes_mul(tm, tn));
    return gs2_bytes_mul(gs2_bytes_mul((size_t)gs2_gemm_streams, elem), tiles);
}

static MPI_Comm gs2_ictxt_comm(int ictxt)
{
    int sysctxt;
    Cblacs_get(ictxt, 10, &sysctxt); /* SGET_BLACSCONTXT: system context */
    return Cblacs2sys_handle(sysctxt);
}

static void gs2_report_gemm(MPI_Comm comm, int m, int n, int k, int is_complex,
                            int gpu, const gs2_mem_details *det)
{
    const char *name = is_complex ? "pzgemm" : "pdgemm";
    const char *how  = gpu ? "COSMA (GPU)" : "ScaLAPACK (CPU)";
    int r = 0;
    MPI_Comm_rank(comm, &r);
    if (r != 0 || !(OPENMX_STDOUT_LEVEL(2) ||
                    (gs2_fallback_report && !gpu && OPENMX_STDOUT_LEVEL(1)))) return;

    if (det->forced) {
        printf("<DFT> gpusolver2: %s m=%d n=%d k=%d using %s (OPENMX_GS2_GEMM_GPU)\n",
               name, m, n, k, how);
    }
    else if (det->host_limited == 2) {
        printf("<DFT> %sgpusolver2: %s m=%d n=%d k=%d using %s "
               "(COSMA has no strategy within COSMA_CPU_MAX_MEMORY)\n",
               gs2_fallback_report ? "Warning: " : "", name, m, n, k, how);
    }
    else if (det->host_limited) {
        printf("<DFT> %sgpusolver2: %s m=%d n=%d k=%d using %s "
               "(host memory: COSMA keeps ~%llu MB/rank x %d ranks/node + reserve %llu MB"
               " > available %llu MB)\n",
               gs2_fallback_report ? "Warning: " : "", name, m, n, k, how,
               det->host_need_mb, det->host_ranks, det->host_reserve_mb, det->host_avail_mb);
    }
    else {
        printf("<DFT> gpusolver2: %s m=%d n=%d k=%d using %s "
               "(GPU memory: need ~%llu MB/rank x %d ranks/GPU + reserve %llu MB %s free %llu MB)\n",
               name, m, n, k, how,
               det->need_mb, det->ranks, det->reserve_mb,
               gpu ? "<=" : ">", det->free_mb);
    }
    fflush(stdout);
}

/* ranks of this node that hold COSMA's host buffers at the same time: the
   ranks of comm on the node, or every rank the launcher put there when
   that is more (the collinear spin worlds run their grids side by side).
   Collective over comm. */
static int gs2_ranks_on_node(MPI_Comm comm)
{
    int nl = 1, ls;
    const char *env = getenv("OMPI_COMM_WORLD_LOCAL_SIZE");
    MPI_Comm shm;

    if (MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shm)
        == MPI_SUCCESS) {
        MPI_Comm_size(shm, &nl);
        MPI_Comm_free(&shm);
    }
    if (env != NULL && (ls = atoi(env)) > nl) nl = ls;
    return nl;
}

/* Host side of a COSMA transform.  COSMA keeps its memory pool and COSTA's
   send/receive buffers (pinned under the GPU backend) for the rest of the
   run; at the large orders of the capacity fallback that is ~10 GB per
   rank (complex order 26000 on 16 ranks), several times the matrices
   themselves, and a node whose ranks all keep it runs out of memory, in
   the transform or later (the force calculation of Si1000 took another
   ~4.6 GB per rank).  The transform takes COSMA only when what it adds
   (the bytes this scalar type already holds are reused) for every rank of
   the node leaves a reserve free on every node of the grid: a quarter of
   the node's memory, at least 4 GiB (OPENMX_GS2_HOST_RESERVE_MB);
   otherwise ScaLAPACK.  The estimate (elpa_cosma_memory.cpp) follows
   COSMA's own strategy and is made on one rank.  Collective over comm,
   uniform. */
static double gs2_gemm_host_retained[2]; /* COSMA host bytes per rank: [real, cplx] */

static int gs2_host_verdict(const char *transa, const char *transb, int m, int n, int k,
                            const int *desca, const int *descb, const int *descc,
                            int is_complex, MPI_Comm comm, gs2_mem_details *det)
{
    const int t = is_complex ? 1 : 0;
    double est = 0.0, add;
    unsigned long long avail, total, reserve, need, buf[3];
    int myid = 0, nloc, ok;

    MPI_Comm_rank(comm, &myid);
    if (myid == 0) {
        est = openmx_gs2_cosma_host_bytes(*transa, *transb, m, n, k, desca, descb, descc,
                                          desca[1], is_complex);
    }
    MPI_Bcast(&est, 1, MPI_DOUBLE, 0, comm);
    nloc = gs2_ranks_on_node(comm);

    avail = openmx_host_mem_available();
    total = openmx_host_mem_total();
    reserve = total / 4;
    if (reserve < (4ULL << 30)) reserve = 4ULL << 30;
    {
        const char *v = getenv("OPENMX_GS2_HOST_RESERVE_MB");
        if (v != NULL && v[0] != '\0') reserve = strtoull(v, NULL, 10) << 20;
    }
    add = est - gs2_gemm_host_retained[t];
    need = (add > 0.0) ? (unsigned long long)(add * (double)nloc) : 0ULL;

    if (est < 0.0) ok = 0;
    else if (need == 0ULL || avail == 0ULL) ok = 1; /* nothing new, or unknown */
    else ok = (need + reserve <= avail);

    buf[0] = (unsigned long long)ok;
    buf[1] = (avail == 0ULL) ? ~0ULL : avail;
    buf[2] = ~0ULL - (unsigned long long)nloc;      /* MIN of this = MAX of nloc */
    MPI_Allreduce(MPI_IN_PLACE, buf, 3, MPI_UNSIGNED_LONG_LONG, MPI_MIN, comm);
    ok = (buf[0] != 0ULL);

    det->host_need_mb    = (est > 0.0) ? (unsigned long long)(est / 1048576.0) : 0ULL;
    det->host_avail_mb   = (buf[1] == ~0ULL) ? 0ULL : buf[1] >> 20;
    det->host_reserve_mb = reserve >> 20;
    det->host_ranks      = (int)(~0ULL - buf[2]);
    det->host_limited    = ok ? 0 : (est < 0.0 ? 2 : 1);

    if (ok && est > gs2_gemm_host_retained[t]) gs2_gemm_host_retained[t] = est;
    return ok;
}

/* uniform per-call decision; collective over the grid's parent comm */
static int gs2_gemm_use_gpu(const char *transa, const char *transb, int m, int n, int k,
                            const int *desca, const int *descb, const int *descc,
                            int is_complex)
{
    gs2_mem_details det;
    int t = is_complex ? 1 : 0;
    int forced, want;
    MPI_Comm comm;

    gs2_gemm_init_once();
    comm = gs2_ictxt_comm(desca[1]);

    forced = gs2_forced_mode_collective("OPENMX_GS2_GEMM_GPU", comm);
    if (forced >= 0) {
        want = forced;
        gs2_det_set_forced(&det);
    }
    else {
        int refresh = gs2_gemm_ranks <= 0 || gs2_gemm_ictxt != desca[1];
        /* A subgroup may have used a different grid since the last call;
           every member of this grid must join a refreshed census. */
        MPI_Allreduce(MPI_IN_PLACE, &refresh, 1, MPI_INT, MPI_MAX, comm);
        if (refresh) {
            gs2_gemm_ranks = gs2_ranks_on_device(comm);
            gs2_gemm_ictxt = desca[1];
        }
        want = gs2_gpu_verdict(gs2_gemm_bound(m, n, k, is_complex), gs2_gemm_ranks,
                               comm, gs2_gemm_mode[t], gs2_gemm_retained[t], &det);
        if (want) {
            want = gs2_host_verdict(transa, transb, m, n, k, desca, descb, descc,
                                    is_complex, comm, &det);
        }
    }

    /* report transitions (and, with the verbose knob, the first verdict);
       an initial GPU verdict is the normal case and stays silent */
    if (want != gs2_gemm_mode[t] &&
        (want == 0 || gs2_gemm_mode[t] == 0 ||
         gs2_env_long("OPENMX_GS2_GPU_VERBOSE", 0) != 0)) {
        gs2_report_gemm(comm, m, n, k, is_complex, want, &det);
    }
    gs2_gemm_mode[t] = want;
    return want;
}

void openmx_gs2_pdgemm_(const char *transa, const char *transb, const int *m, const int *n,
                        const int *k, const double *alpha, const double *a, const int *ia,
                        const int *ja, const int *desca, const double *b, const int *ib,
                        const int *jb, const int *descb, const double *beta, double *c,
                        const int *ic, const int *jc, const int *descc)
{
    if (gs2_gemm_use_gpu(transa, transb, *m, *n, *k, desca, descb, descc, 0)) {
        size_t before = gs2_free_now(), after;
        cosma_pdgemm_(transa, transb, m, n, k, alpha, a, ia, ja, desca,
                      b, ib, jb, descb, beta, c, ic, jc, descc);
        after = gs2_free_now();
        if (before > after) gs2_gemm_retained[0] += before - after;
    }
    else {
        pdgemm_(transa, transb, m, n, k, alpha, a, ia, ja, desca,
                b, ib, jb, descb, beta, c, ic, jc, descc);
    }
}

void openmx_gs2_pzgemm_(const char *transa, const char *transb, const int *m, const int *n,
                        const int *k, const double *alpha, const double *a, const int *ia,
                        const int *ja, const int *desca, const double *b, const int *ib,
                        const int *jb, const int *descb, const double *beta, double *c,
                        const int *ic, const int *jc, const int *descc)
{
    if (gs2_gemm_use_gpu(transa, transb, *m, *n, *k, desca, descb, descc, 1)) {
        size_t before = gs2_free_now(), after;
        cosma_pzgemm_(transa, transb, m, n, k, alpha, a, ia, ja, desca,
                      b, ib, jb, descb, beta, c, ic, jc, descc);
        after = gs2_free_now();
        if (before > after) gs2_gemm_retained[1] += before - after;
    }
    else {
        pzgemm_(transa, transb, m, n, k, alpha, a, ia, ja, desca,
                b, ib, jb, descb, beta, c, ic, jc, descc);
    }
}

/* Admission of the distributed solve for the automatic capacity fallback
   of the cluster solvers.  Collective over the grid of ictxt: the verdict
   of the per-solve preflight above, made for an eigensolve of order n on
   this grid plus the transform buffers COSMA keeps afterwards, before any
   handle exists.  Returns 1 (fits on the GPU kernels) or 0 on every rank
   of the grid; -1 when OPENMX_GS2_ELPA_GPU fixes the kernel side on some
   rank (no admission then).  need_mb is this rank's estimate, free_mb the
   smallest free device memory of the grid, ranks the ranks on this rank's
   device. */
int openmx_gs2_gpu_fits(int ictxt, int n, int nblk, int is_complex,
                        unsigned long long *need_mb, unsigned long long *free_mb,
                        int *ranks)
{
    int nprow, npcol, myrow, mycol, ZERO = 0;
    int na_rows, na_cols, ranks_on_dev, forced_any, ok, i;
    size_t need, retained;
    gs2_mem_details det;
    MPI_Comm comm;

    *need_mb = 0;
    *free_mb = 0;
    *ranks = 0;

    Cblacs_gridinfo(ictxt, &nprow, &npcol, &myrow, &mycol);
    comm = gs2_ictxt_comm(ictxt);

    forced_any = (gs2_forced_mode_collective("OPENMX_GS2_ELPA_GPU", comm) >= 0);
    if (forced_any) return -1;

    na_rows = numroc_(&n, &nblk, &myrow, &ZERO, &nprow);
    na_cols = numroc_(&n, &nblk, &mycol, &ZERO, &npcol);

    gs2_gemm_init_once();
    ranks_on_dev = gs2_ranks_on_device(comm);
    need = gs2_need_bytes(na_rows, na_cols, is_complex);
    /* COSMA's Tiled-MM buffers exist only when the transforms may run on
       COSMA (OPENMX_GS2_GEMM_GPU=0 keeps them on ScaLAPACK) */
    if (gs2_forced_mode("OPENMX_GS2_GEMM_GPU") != 0)
        need = gs2_bytes_add(need, gs2_gemm_bound(n, n, n, is_complex));
    /* on a re-admission (next MD step, orbital-optimization restart, post
       call) the device memory this bridge already holds is reused: COSMA's
       persistent buffers of this scalar type and any live ELPA handle, as
       in the per-call verdicts of gs2_gemm_use_gpu and gs2_get_handle */
    retained = gs2_gemm_retained[is_complex ? 1 : 0];
    for (i = 0; i < GS2_MAX_HANDLES; i++) {
        if (gs2_slots[i].used) retained = gs2_bytes_add(retained, gs2_slots[i].retained);
    }
    ok = gs2_gpu_verdict(need, ranks_on_dev, comm, -1, retained, &det);

    *need_mb = det.need_mb;
    *free_mb = det.free_mb;
    *ranks = ranks_on_dev;
    return ok;
}

void openmx_gs2_set_fallback_report(int on)
{
    gs2_fallback_report = (on != 0);
}

void openmx_gs2_grid_free(int ictxt)
{
    int i, err;

    if (gs2_gemm_ictxt == ictxt) {
        gs2_gemm_ranks = 0;
        gs2_gemm_ictxt = -1;
    }

    for (i = 0; i < GS2_MAX_HANDLES; i++) {
        if (gs2_slots[i].used && gs2_slots[i].ictxt == ictxt) {
            elpa_deallocate(gs2_slots[i].h, &err);
            gs2_slots[i].used = 0;
        }
    }
}

void openmx_gs2_finalize(void)
{
    int i, err;

    if (!gs2_elpa_inited) return;
    for (i = 0; i < GS2_MAX_HANDLES; i++) {
        if (gs2_slots[i].used) {
            elpa_deallocate(gs2_slots[i].h, &err);
            gs2_slots[i].used = 0;
        }
    }
    elpa_uninit(&err);
    gs2_elpa_inited = 0;
}

/**********************************************************************
  elpa_cosma_bridge.c

  Implementation of the OpenMX <-> ELPA (2026.02, NVIDIA GPU kernels)
  bridge used by scf.eigen.lib=gpusolver2.  See elpa_cosma_bridge.h.
***********************************************************************/

#include "elpa_cosma_bridge.h"

#include <elpa/elpa.h>

#include <cuda_runtime.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

extern void Cblacs_gridinfo(int ictxt, int *nprow, int *npcol, int *myrow, int *mycol);
extern void Cblacs_get(int ictxt, int what, int *val);
extern MPI_Comm Cblacs2sys_handle(int ictxt);
extern int numroc_(int *n, int *nb, int *iproc, int *isrcproc, int *nprocs);

static int gs2_elpa_inited = 0;

/* one cached, fully set-up ELPA handle per (BLACS context, n, nev, type) */
#define GS2_MAX_HANDLES 16
typedef struct {
    int used, ictxt, n, nev, is_complex;
    int gpu;          /* 1 = NVIDIA GPU kernels, 0 = CPU kernels */
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
} gs2_mem_details;

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
    size_t local  = (size_t)na_rows * (size_t)na_cols * elem;
    size_t factor = (size_t)gs2_env_long("OPENMX_GS2_GPU_FACTOR", 3);
    size_t fixed  = (size_t)gs2_env_long("OPENMX_GS2_GPU_FIXED_MB", 128) << 20;
    return factor * local + fixed;
}

/* how many ranks end up on this rank's CUDA device (they all run the
   same solve concurrently, so the free memory has to cover all of them).
   Collective over comm. */
static int gs2_ranks_on_device(MPI_Comm comm)
{
    int mydev = -1, ndev = 0, cnt = 1, est = 1, i, nl = 1;
    MPI_Comm shm;

    if (cudaGetDevice(&mydev) != cudaSuccess) mydev = -1;

    /* exact count within comm: ranks on my node that picked my device */
    if (MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shm)
        == MPI_SUCCESS) {
        int *devs;
        MPI_Comm_size(shm, &nl);
        devs = (int *)malloc((size_t)nl * sizeof(int));
        if (devs != NULL) {
            MPI_Allgather(&mydev, 1, MPI_INT, devs, 1, MPI_INT, shm);
            for (cnt = 0, i = 0; i < nl; i++) {
                if (devs[i] == mydev) cnt++;
            }
            free(devs);
        }
        MPI_Comm_free(&shm);
    }

    /* comm may span only part of the node (the spin-split worlds of the
       collinear cluster solver run concurrently), so also estimate from
       the launcher's node-local size and keep the larger count */
    {
        const char *env = getenv("OMPI_COMM_WORLD_LOCAL_SIZE");
        if (env != NULL && mydev >= 0 &&
            cudaGetDeviceCount(&ndev) == cudaSuccess && ndev > 0) {
            int ls = atoi(env);
            for (est = 0, i = 0; i < ls; i++) {
                if (i % ndev == mydev) est++;
            }
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

    {
        const char *env = getenv("OPENMX_GS2_ELPA_GPU");
        if (env != NULL && env[0] != '\0') {
            det->forced = 1;
            return (atoi(env) != 0);
        }
    }

    reserve = (size_t)gs2_env_long("OPENMX_GS2_GPU_RESERVE_MB", 512) << 20;
    det->reserve_mb = (unsigned long long)(reserve >> 20);

    want_free = need * (size_t)ranks_on_dev + reserve;
    if (current_mode == 0) want_free += want_free / 4;

    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) {
        free_b = 0;
        ok = 0;
    }
    else {
        /* memory a live GPU handle keeps between solves is reused by the
           next solve, so count it as available rather than as consumed */
        free_b += retained;
        ok = (want_free <= free_b);
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
    if (r != 0) return;

    if (det->forced) {
        printf("<DFT> gpusolver2: ELPA n=%d nev=%d %s %s kernels (OPENMX_GS2_ELPA_GPU)\n",
               n, nev, how, gpu ? "GPU" : "CPU");
    }
    else {
        printf("<DFT> gpusolver2: ELPA n=%d nev=%d %s %s kernels "
               "(GPU memory: need ~%llu MB/rank x %d ranks/GPU + reserve %llu MB %s free %llu MB)\n",
               n, nev, how, gpu ? "GPU" : "CPU",
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
        if (myid == 0) {
            printf("<DFT> gpusolver2: ELPA (API %d) with NVIDIA GPU kernels + COSMA initialized\n",
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
    size_t need = 0;
    gs2_mem_details det;
    MPI_Comm comm;
    elpa_t h;

    *perr = 0;

    for (i = 0; i < GS2_MAX_HANDLES; i++) {
        if (gs2_slots[i].used && gs2_slots[i].ictxt == ictxt && gs2_slots[i].n == n &&
            gs2_slots[i].nev == nev && gs2_slots[i].is_complex == is_complex) {

            /* the memory situation changes while a handle is cached (the
               resident caches of other GPU stages keep growing), so
               re-evaluate the GPU/CPU verdict before every solve and
               rebuild the handle on the other side when it flips */
            want_gpu = gs2_gpu_verdict(gs2_slots[i].need, gs2_slots[i].ranks_on_dev,
                                       gs2_slots[i].comm, gs2_slots[i].gpu,
                                       gs2_slots[i].retained, &det);
            if (want_gpu == gs2_slots[i].gpu) return &gs2_slots[i];

            gs2_report_mode(gs2_slots[i].comm, n, nev, want_gpu, &det, "switching to");
            have_verdict = 1;
            break;
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

    if (gs2_elpa_init_once() != 0) { *perr = -1; return NULL; }

    Cblacs_gridinfo(ictxt, &nprow, &npcol, &myrow, &mycol);
    Cblacs_get(ictxt, 10, &sysctxt); /* SGET_BLACSCONTXT: system context */
    comm = Cblacs2sys_handle(sysctxt);

    na_rows = numroc_(&n, &nblk, &myrow, &ZERO, &nprow);
    na_cols = numroc_(&n, &nblk, &mycol, &ZERO, &npcol);

    ranks_on_dev = gs2_ranks_on_device(comm);
    need = gs2_need_bytes(na_rows, na_cols, is_complex);
    if (!have_verdict) {
        want_gpu = gs2_gpu_verdict(need, ranks_on_dev, comm, -1, 0, &det);
        if (!want_gpu || gs2_env_long("OPENMX_GS2_GPU_VERBOSE", 0) != 0) {
            gs2_report_mode(comm, n, nev, want_gpu, &det, "using");
        }
    }

    h = elpa_allocate(&err);
    if (err != ELPA_OK) { *perr = err; return NULL; }

    elpa_set_integer(h, "na", n, &err);
    if (err == ELPA_OK) elpa_set_integer(h, "nev", nev, &err);
    if (err == ELPA_OK) elpa_set_integer(h, "local_nrows", na_rows, &err);
    if (err == ELPA_OK) elpa_set_integer(h, "local_ncols", na_cols, &err);
    if (err == ELPA_OK) elpa_set_integer(h, "nblk", nblk, &err);
    if (err == ELPA_OK) elpa_set_integer(h, "mpi_comm_parent", (int)MPI_Comm_c2f(comm), &err);
    if (err == ELPA_OK) elpa_set_integer(h, "process_row", myrow, &err);
    if (err == ELPA_OK) elpa_set_integer(h, "process_col", mycol, &err);
    if (err != ELPA_OK) { *perr = err; return NULL; }

    /* GPU selection: must be requested before elpa_setup so that the setup
       binds the device.  The device itself was assigned rank-wise by the
       central GPUSOLVER initialization (local_rank % ndev, scf.Gpu.Num).
       When the preflight above found too little free device memory, run
       this configuration on ELPA's CPU kernels instead. */
    elpa_set_integer(h, "nvidia-gpu", want_gpu ? 1 : 0, &err);
    if (err != ELPA_OK) { *perr = err; return NULL; }
    if (want_gpu) {
        int dev = 0, seterr = ELPA_OK;
        if (cudaGetDevice(&dev) == cudaSuccess) {
            elpa_set_integer(h, "use_gpu_id", dev, &seterr); /* optional */
        }
    }

    err = elpa_setup(h);
    if (err != ELPA_OK) { *perr = err; return NULL; }

    {
        int solver = ELPA_SOLVER_2STAGE;
        const char *env = getenv("OPENMX_GS2_ELPA_SOLVER");
        if (env != NULL && atoi(env) == 1) solver = ELPA_SOLVER_1STAGE;
        elpa_set_integer(h, "solver", solver, &err);
        if (err != ELPA_OK) { *perr = err; return NULL; }
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
    gs2_slots[i].gpu = want_gpu;
    gs2_slots[i].ranks_on_dev = ranks_on_dev;
    gs2_slots[i].need = need;
    gs2_slots[i].retained = 0;
    gs2_slots[i].comm = comm;
    gs2_slots[i].h = h;

    return &gs2_slots[i];
}

/* update the slot's estimate of how much device memory its handle kept
   allocated across the solve (sampled as the free-memory drop over the
   solve; the diagonalization phase runs exclusively, so the drop is
   attributable to ELPA) */
static size_t gs2_free_now(void)
{
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) return 0;
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

void openmx_gs2_grid_free(int ictxt)
{
    int i, err;

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

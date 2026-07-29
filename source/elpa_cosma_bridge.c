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
    elpa_t h;
} gs2_handle_slot;
static gs2_handle_slot gs2_slots[GS2_MAX_HANDLES];

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

/* build (or fetch) the handle for this solve configuration */
static elpa_t gs2_get_handle(int ictxt, int n, int nev, int is_complex, int nblk, int *perr)
{
    int i, err = ELPA_OK;
    int nprow, npcol, myrow, mycol, ZERO = 0;
    int na_rows, na_cols, sysctxt;
    MPI_Comm comm;
    elpa_t h;

    *perr = 0;

    for (i = 0; i < GS2_MAX_HANDLES; i++) {
        if (gs2_slots[i].used && gs2_slots[i].ictxt == ictxt && gs2_slots[i].n == n &&
            gs2_slots[i].nev == nev && gs2_slots[i].is_complex == is_complex) {
            return gs2_slots[i].h;
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
       central GPUSOLVER initialization (local_rank % ndev, scf.Gpu.Num). */
    elpa_set_integer(h, "nvidia-gpu", 1, &err);
    if (err != ELPA_OK) { *perr = err; return NULL; }
    {
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
    gs2_slots[i].h = h;

    return h;
}

int openmx_gs2_eigen_real(int n, int nev, double *a, int *desca, double *w, double *z,
                          int *descz)
{
    int err = 0;
    elpa_t h = gs2_get_handle(desca[1], n, nev, 0, desca[4], &err);

    (void)descz;
    if (h == NULL) return (err != 0) ? err : -1;

    elpa_eigenvectors_double(h, a, w, z, &err);
    if (err != ELPA_OK) {
        fprintf(stderr, "openmx gpusolver2: elpa_eigenvectors_double failed: %s\n",
                elpa_strerr(err));
        return err;
    }
    return 0;
}

int openmx_gs2_eigen_complex(int n, int nev, void *a, int *desca, double *w, void *z,
                             int *descz)
{
    int err = 0;
    elpa_t h = gs2_get_handle(desca[1], n, nev, 1, desca[4], &err);

    (void)descz;
    if (h == NULL) return (err != 0) ? err : -1;

    elpa_eigenvectors_double_complex(h, (double complex *)a, w, (double complex *)z, &err);
    if (err != ELPA_OK) {
        fprintf(stderr, "openmx gpusolver2: elpa_eigenvectors_double_complex failed: %s\n",
                elpa_strerr(err));
        return err;
    }
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

/**********************************************************************
  dlaf_cosma_bridge.c

  Implementation of the OpenMX <-> DLA-Future (0.10, CUDA backend)
  bridge used by scf.eigen.lib=gpusolver2.  See dlaf_cosma_bridge.h.
***********************************************************************/

#include "dlaf_cosma_bridge.h"

#include <complex.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <dlaf_c/eigensolver/eigensolver.h>
#include <dlaf_c/grid.h>
#include <dlaf_c/init.h>

static int gs2_dlaf_inited = 0;

/* BLACS contexts already registered with DLA-Future */
#define GS2_MAX_GRIDS 16
typedef struct {
    int used, ictxt;
} gs2_grid_slot;
static gs2_grid_slot gs2_grids[GS2_MAX_GRIDS];

/* pika worker threads per rank: share the cores of the node evenly among
   the local ranks unless OPENMX_GS2_PIKA_THREADS overrides it. */
static int gs2_pika_threads(void)
{
    const char *env = getenv("OPENMX_GS2_PIKA_THREADS");
    long ncore;
    int nlocal = 1, nthr;
    MPI_Comm shm;

    if (env != NULL && atoi(env) > 0) return atoi(env);

    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shm);
    MPI_Comm_size(shm, &nlocal);
    MPI_Comm_free(&shm);

    ncore = sysconf(_SC_NPROCESSORS_ONLN);
    if (ncore < 1) ncore = 1;
    if (nlocal < 1) nlocal = 1;

    nthr = (int)(ncore / nlocal);
    if (nthr < 2) nthr = 2; /* keep some task/MPI-polling overlap */
    return nthr;
}

static int gs2_dlaf_init_once(void)
{
    static char thr_arg[64];
    static char bind_arg[128];
    const char *argv_pika[5];
    const char *argv_dlaf[1] = { "openmx" };
    int argc_pika = 0, provided = 0, myid = 0;
    const char *bind, *pool;

    if (gs2_dlaf_inited) return 0;

    MPI_Query_thread(&provided);
    if (provided < MPI_THREAD_MULTIPLE) {
        fprintf(stderr,
                "openmx gpusolver2: DLA-Future needs MPI_THREAD_MULTIPLE, but MPI provides %d\n",
                provided);
        return -1;
    }

    /* The DLA-Future umpire pools default to 1 GiB blocks per rank, which
       exhausts a small GPU when several ranks share one device; preset
       256 MiB blocks (they grow on demand).  Exported DLAF_* variables
       win because setenv does not overwrite here. */
    setenv("DLAF_UMPIRE_DEVICE_MEMORY_POOL_INITIAL_BLOCK_BYTES", "268435456", 0);
    setenv("DLAF_UMPIRE_DEVICE_MEMORY_POOL_NEXT_BLOCK_BYTES", "268435456", 0);
    setenv("DLAF_UMPIRE_HOST_MEMORY_POOL_INITIAL_BLOCK_BYTES", "268435456", 0);
    setenv("DLAF_UMPIRE_HOST_MEMORY_POOL_NEXT_BLOCK_BYTES", "268435456", 0);

    /* DLA-Future opens 32 normal- plus 32 high-priority CUDA streams per
       rank and gives each its own cuBLAS/cuSOLVER handle pair; the device
       footprint of the handles then multiplies into gigabytes when several
       ranks share one GPU.  Trim the streams to 8+8 (honored with
       pika >= 0.31) and keep the cuBLAS handle workspace at the minimal
       16 KiB x 8 configuration; exported variables win. */
    setenv("DLAF_NUM_NP_GPU_STREAMS", "8", 0);
    setenv("DLAF_NUM_HP_GPU_STREAMS", "8", 0);
    setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8", 0);

    argv_pika[argc_pika++] = "openmx";
    snprintf(thr_arg, sizeof(thr_arg), "--pika:threads=%d", gs2_pika_threads());
    argv_pika[argc_pika++] = thr_arg;

    /* Several ranks on one node would all bind their pika workers to the
       same first cores of the process mask; leave the workers unbound
       unless the user asks for a specific binding. */
    bind = getenv("OPENMX_GS2_PIKA_BIND");
    if (bind == NULL) bind = "none";
    if (strcmp(bind, "default") != 0) {
        snprintf(bind_arg, sizeof(bind_arg), "--pika:bind=%s", bind);
        argv_pika[argc_pika++] = bind_arg;
    }

    /* Serialize every MPI call of pika (submission and polling) onto a
       dedicated one-thread pool.  Without it any worker thread issues
       and progresses the nonblocking collectives of DLA-Future, and the
       libnbc progression of Open MPI is not thread safe (concurrent
       schedules end in MPI_ERR_TRUNCATE aborts).  OPENMX_GS2_PIKA_MPI_POOL=0
       returns to the pika default for MPI libraries without this issue. */
    pool = getenv("OPENMX_GS2_PIKA_MPI_POOL");
    if (pool == NULL || atoi(pool) != 0) {
        argv_pika[argc_pika++] = "--pika:ini=pika.mpi.enable_pool=1";
    }

    dlaf_initialize(argc_pika, argv_pika, 1, argv_dlaf);
    gs2_dlaf_inited = 1;

    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    if (myid == 0) {
        printf("<DFT> gpusolver2: DLA-Future 0.10 (pika, CUDA) + COSMA initialized\n");
        fflush(stdout);
    }
    return 0;
}

/* register the BLACS context with DLA-Future once */
static int gs2_get_grid(int ictxt)
{
    int i;

    for (i = 0; i < GS2_MAX_GRIDS; i++) {
        if (gs2_grids[i].used && gs2_grids[i].ictxt == ictxt) return 0;
    }

    if (gs2_dlaf_init_once() != 0) return -1;

    for (i = 0; i < GS2_MAX_GRIDS; i++) {
        if (!gs2_grids[i].used) break;
    }
    if (i == GS2_MAX_GRIDS) {
        fprintf(stderr, "openmx gpusolver2: too many BLACS grids registered with DLA-Future\n");
        return -1;
    }

    dlaf_create_grid_from_blacs(ictxt);
    gs2_grids[i].used = 1;
    gs2_grids[i].ictxt = ictxt;
    return 0;
}

int openmx_gs2_eigen_real(int n, int nev, double *a, int *desca, double *w, double *z,
                          int *descz)
{
    int info = -1;

    (void)nev; /* DLA-Future computes all eigenpairs */
    if (gs2_get_grid(desca[1]) != 0) return -1;

    dlaf_pdsyevd('L', n, a, 1, 1, desca, w, z, 1, 1, descz, &info);
    if (info != 0) {
        fprintf(stderr, "openmx gpusolver2: dlaf_pdsyevd failed (info=%d)\n", info);
        return (info != 0) ? info : -1;
    }
    return 0;
}

int openmx_gs2_eigen_complex(int n, int nev, void *a, int *desca, double *w, void *z,
                             int *descz)
{
    int info = -1;

    (void)nev; /* DLA-Future computes all eigenpairs */
    if (gs2_get_grid(desca[1]) != 0) return -1;

    dlaf_pzheevd('L', n, (dlaf_complex_z *)a, 1, 1, desca, w, (dlaf_complex_z *)z, 1, 1,
                 descz, &info);
    if (info != 0) {
        fprintf(stderr, "openmx gpusolver2: dlaf_pzheevd failed (info=%d)\n", info);
        return (info != 0) ? info : -1;
    }
    return 0;
}

void openmx_gs2_grid_free(int ictxt)
{
    int i;

    for (i = 0; i < GS2_MAX_GRIDS; i++) {
        if (gs2_grids[i].used && gs2_grids[i].ictxt == ictxt) {
            dlaf_free_grid(ictxt);
            gs2_grids[i].used = 0;
        }
    }
}

void openmx_gs2_finalize(void)
{
    int i;

    if (!gs2_dlaf_inited) return;
    dlaf_free_all_grids();
    for (i = 0; i < GS2_MAX_GRIDS; i++) gs2_grids[i].used = 0;
    dlaf_finalize();
    gs2_dlaf_inited = 0;
}

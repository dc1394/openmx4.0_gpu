/**********************************************************************
  slate_bridge.cc

  Bridge between the ScaLAPACK-style distributed matrices of the
  mainline collinear/non-collinear cluster calculations and the SLATE
  library, used when scf.eigen.lib=gpusolver2 is selected.

  The eigensolvers replace ELPA/pdsyevx/pzheevx: slate::heev runs the
  two-stage tridiagonalization and (by default) the divide-and-conquer
  back-transformation distributed over every rank with the NVIDIA GPU
  as the execution target, so a cluster calculation finally scales
  over multiple ranks/GPUs.  The matrix products replace pdgemm_ and
  pzgemm_ with slate::gemm on the same GPUs.

  The local block-cyclic arrays of OpenMX are wrapped without copying
  (slate::fromScaLAPACK); the MPI communicator and the process grid
  are recovered from the BLACS context found in the descriptors, so
  the spin-split sub-communicators of the collinear path work
  unchanged.

  Environment knobs:
    OPENMX_GS2_TARGET=host      run SLATE on the CPUs (debugging aid)
    OPENMX_GS2_EIG_METHOD=qr    QR iteration instead of the default
                                divide-and-conquer eigensolver

  SLATE schedules its internal tasks with OpenMP and issues MPI calls
  from those tasks, so MPI must be initialized with at least
  MPI_THREAD_SERIALIZED (openmx.c requests MPI_THREAD_MULTIPLE).

  The NVHPC OpenMP runtime (libnvomp), which the rest of OpenMX runs
  on, serializes nested parallel regions regardless of
  omp_set_max_active_levels.  SLATE's band-to-tridiagonal stage
  (hb2st) launches a nested parallel-for whose iterations spin-wait
  on each other, which deadlocks when that region is serialized, so
  every SLATE call here runs with the OpenMP thread count forced to
  OPENMX_GS2_THREADS (default 1, which is always safe); the
  parallelism of gpusolver2 comes from the MPI ranks and the GPUs.
***********************************************************************/

#include "slate_bridge.h"

#include <slate/slate.hh>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <strings.h>
#include <vector>

extern "C" {
  void Cblacs_gridinfo(int ictxt, int *nprow, int *npcol, int *myrow, int *mycol);
  void Cblacs_get(int ictxt, int what, int *val);
  void Cblacs_pcoord(int ictxt, int pnum, int *prow, int *pcol);
  MPI_Comm Cblacs2sys_handle(int bhandle);
}

namespace {

struct Gs2Grid {
  MPI_Comm comm;
  int nprow, npcol, myrow, mycol;
  slate::GridOrder order;
};

/* Recover the MPI communicator and the process grid of a BLACS
   context.  The grid order is detected from the rank layout, which
   also validates that the communicator really matches the grid. */
int gs2_get_grid(int ictxt, Gs2Grid *g)
{
  int bhandle, rank;

  Cblacs_gridinfo(ictxt, &g->nprow, &g->npcol, &g->myrow, &g->mycol);
  if (g->nprow <= 0 || g->npcol <= 0 || g->myrow < 0 || g->mycol < 0) {
    std::fprintf(stderr, "slate_bridge: the calling rank is outside the BLACS grid\n");
    return -1;
  }

  Cblacs_get(ictxt, 10, &bhandle);
  g->comm = Cblacs2sys_handle(bhandle);
  MPI_Comm_rank(g->comm, &rank);

  /* The grid order must be decided identically on every rank, so it
     is read off the coordinates of grid process number 1 (the ranks
     on the grid diagonal fit both orders, which rules out a purely
     local test). */
  if (g->nprow * g->npcol == 1) {
    g->order = slate::GridOrder::Col;
  }
  else {
    int prow1, pcol1;
    Cblacs_pcoord(ictxt, 1, &prow1, &pcol1);
    if (prow1 == 0 && pcol1 == 1) {
      g->order = slate::GridOrder::Row;   /* ranks advance along a row */
    }
    else if (prow1 == 1 && pcol1 == 0) {
      g->order = slate::GridOrder::Col;   /* ranks advance down a column */
    }
    else {
      std::fprintf(stderr, "slate_bridge: the BLACS grid layout is neither row- nor column-major\n");
      return -2;
    }
  }

  /* validate that the communicator recovered from the context really
     carries this grid layout */
  int expect = (g->order == slate::GridOrder::Row)
             ? g->myrow * g->npcol + g->mycol
             : g->mycol * g->nprow + g->myrow;
  if (rank != expect) {
    std::fprintf(stderr, "slate_bridge: the BLACS grid does not match its MPI communicator\n");
    return -3;
  }

  return 0;
}

/* number of OpenMP threads a SLATE call may use (see the header
   comment; >1 deadlocks in slate::heev on libnvomp) */
int gs2_threads()
{
  static int nthr = 0;
  if (nthr == 0) {
    const char *e = std::getenv("OPENMX_GS2_THREADS");
    nthr = (e != NULL) ? atoi(e) : 1;
    if (nthr < 1) nthr = 1;
  }
  return nthr;
}

/* scope guard that pins the OpenMP thread count during a SLATE call
   and restores the surrounding setting afterwards.  The constructor
   also drains every pending device operation: the OpenACC kernels of
   the surrounding SCF machinery run asynchronously, and a device
   buffer released with writes still in flight can be handed to a
   SLATE allocation and clobbered afterwards. */
struct Gs2ThreadScope {
  int saved;
  Gs2ThreadScope() : saved(omp_get_max_threads())
  {
    omp_set_num_threads(gs2_threads());
    cudaDeviceSynchronize();
  }
  ~Gs2ThreadScope()
  { omp_set_num_threads(saved); }
};

slate::Target gs2_target()
{
  const char *t = std::getenv("OPENMX_GS2_TARGET");
  if (t != NULL && strcasecmp(t, "host") == 0) return slate::Target::HostTask;
  return slate::Target::Devices;
}

slate::MethodEig gs2_eig_method()
{
  const char *m = std::getenv("OPENMX_GS2_EIG_METHOD");
  if (m != NULL && strcasecmp(m, "qr") == 0) return slate::MethodEig::QR;
  return slate::MethodEig::DC;
}

/* One-time initialization: verify the MPI thread support SLATE relies
   on and announce the library on the first gpusolver2 operation. */
int gs2_init_check()
{
  static int state = 0;   /* 0: not checked, 1: ok, -1: failed */

  if (state != 0) return (state > 0) ? 0 : 1;

  int level = MPI_THREAD_SINGLE;
  MPI_Query_thread(&level);
  if (level < MPI_THREAD_SERIALIZED) {
    std::fprintf(stderr,
                 "slate_bridge: gpusolver2 needs MPI_THREAD_SERIALIZED or higher\n"
                 "but the MPI library provided a lower thread support level (%d).\n",
                 level);
    state = -1;
    return 1;
  }

  /* cuBLAS reserves a large per-handle device workspace by default,
     and with many ranks per GPU the handle creation itself runs out
     of memory (seen with 8 ranks on a 16 GB card).  SLATE creates its
     handles after this point, so ask cuBLAS for a small workspace
     unless the user chose one. */
  setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8", 0);

  /* Mixing CUDA library generations corrupts the SLATE GPU path with
     asynchronous illegal-address faults (observed with a binary built
     against CUDA 13.1 picking up 13.2 libraries through
     LD_LIBRARY_PATH, which overrides the RUNPATH of the binary), so
     refuse to run on a runtime that differs from the build. */
  if (gs2_target() == slate::Target::Devices) {
    int rt = 0;
    cudaRuntimeGetVersion(&rt);
    if (rt != CUDART_VERSION) {
      std::fprintf(stderr,
                   "slate_bridge: the CUDA runtime loaded at run time (%d) differs from the\n"
                   "one the gpusolver2 stack was built with (%d).  LD_LIBRARY_PATH probably\n"
                   "points at another CUDA installation; put the build-time CUDA library\n"
                   "directories first (see the RUNPATH of the openmx binary).\n",
                   rt, (int)CUDART_VERSION);
      state = -1;
      return 1;
    }
  }

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    std::printf("<DFT> gpusolver2: SLATE %d (%s target) initialized\n",
                slate::version(),
                (gs2_target() == slate::Target::Devices) ? "GPU devices" : "host");
    std::fflush(stdout);
  }

  state = 1;
  return 0;
}

enum { GS2_DESC_CTXT = 1, GS2_DESC_MB = 4, GS2_DESC_NB = 5, GS2_DESC_LLD = 8 };

template <typename scalar_t>
int gs2_eigen(int n, scalar_t *a, int *desca, double *w, scalar_t *z, int *descz)
{
  if (gs2_init_check() != 0) return 1;

  Gs2Grid g;
  int rc = gs2_get_grid(desca[GS2_DESC_CTXT], &g);
  if (rc != 0) return rc;

  int nb = desca[GS2_DESC_NB];

  try {
    Gs2ThreadScope thread_scope;

    auto A = slate::Matrix<scalar_t>::fromScaLAPACK(
        n, n, a, desca[GS2_DESC_LLD], desca[GS2_DESC_MB], nb,
        g.order, g.nprow, g.npcol, g.comm);
    auto Z = slate::Matrix<scalar_t>::fromScaLAPACK(
        n, n, z, descz[GS2_DESC_LLD], descz[GS2_DESC_MB], descz[GS2_DESC_NB],
        g.order, g.nprow, g.npcol, g.comm);

    std::vector<double> lambda(n);
    slate::Options opts = {
      { slate::Option::Target,    gs2_target() },
      { slate::Option::MethodEig, gs2_eig_method() },
    };

    if (g.order == slate::GridOrder::Col && g.nprow == g.npcol) {

      /* the layout already satisfies slate::heev: solve in place */
      auto Ah = slate::HermitianMatrix<scalar_t>(slate::Uplo::Lower, A);
      slate::heev(Ah, lambda, Z, opts);
    }
    else {

      /* slate::heev requires a column-major square process grid, but
         OpenMX sets up row-major (and usually non-square) grids.
         Redistribute the matrix onto the largest square subgrid of
         the same communicator, solve there, and redistribute the
         eigenvectors back; the ranks beyond psq*psq only assist the
         communication.  The two extra n*n copies live only inside
         this call. */
      int np, psq;
      MPI_Comm_size(g.comm, &np);
      psq = (int)std::sqrt((double)np + 0.5);
      while (psq*psq > np) psq--;

      auto Asq = slate::Matrix<scalar_t>(n, n, nb, psq, psq, g.comm);
      Asq.insertLocalTiles();
      auto Zsq = slate::Matrix<scalar_t>(n, n, nb, psq, psq, g.comm);
      Zsq.insertLocalTiles();

      slate::redistribute(A, Asq);
      auto Ah = slate::HermitianMatrix<scalar_t>(slate::Uplo::Lower, Asq);
      slate::heev(Ah, lambda, Zsq, opts);
      slate::redistribute(Zsq, Z);
    }

    std::memcpy(w, lambda.data(), (size_t)n * sizeof(double));
    return 0;
  }
  catch (std::exception const &e) {
    std::fprintf(stderr, "slate_bridge: slate::heev failed: %s\n", e.what());
    return 1;
  }
}

template <typename scalar_t>
void gs2_gemm(const char *transa, const char *transb,
              int m, int n, int k, scalar_t alpha,
              scalar_t *a, int *desca, scalar_t *b, int *descb,
              scalar_t beta, scalar_t *c, int *descc,
              int whole_matrix)
{
  if (gs2_init_check() != 0 || whole_matrix == 0) {
    std::fprintf(stderr, "slate_bridge: unsupported gpusolver2 matrix-product call\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  Gs2Grid g;
  if (gs2_get_grid(desca[GS2_DESC_CTXT], &g) != 0) MPI_Abort(MPI_COMM_WORLD, 1);

  int nota = (transa[0] == 'N' || transa[0] == 'n');
  int notb = (transb[0] == 'N' || transb[0] == 'n');

  try {
    Gs2ThreadScope thread_scope;
    auto A = slate::Matrix<scalar_t>::fromScaLAPACK(
        nota ? m : k, nota ? k : m, a, desca[GS2_DESC_LLD],
        desca[GS2_DESC_MB], desca[GS2_DESC_NB],
        g.order, g.nprow, g.npcol, g.comm);
    auto B = slate::Matrix<scalar_t>::fromScaLAPACK(
        notb ? k : n, notb ? n : k, b, descb[GS2_DESC_LLD],
        descb[GS2_DESC_MB], descb[GS2_DESC_NB],
        g.order, g.nprow, g.npcol, g.comm);
    auto C = slate::Matrix<scalar_t>::fromScaLAPACK(
        m, n, c, descc[GS2_DESC_LLD],
        descc[GS2_DESC_MB], descc[GS2_DESC_NB],
        g.order, g.nprow, g.npcol, g.comm);

    if (!nota) A = (transa[0] == 'C' || transa[0] == 'c') ? slate::conj_transpose(A)
                                                          : slate::transpose(A);
    if (!notb) B = (transb[0] == 'C' || transb[0] == 'c') ? slate::conj_transpose(B)
                                                          : slate::transpose(B);

    slate::gemm(alpha, A, B, beta, C, {
      { slate::Option::Target, gs2_target() },
    });
  }
  catch (std::exception const &e) {
    std::fprintf(stderr, "slate_bridge: slate::gemm failed: %s\n", e.what());
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
}

} /* anonymous namespace */

int openmx_gs2_eigen_real(int n, int nev, double *a, int *desca,
                          double *w, double *z, int *descz)
{
  (void)nev;   /* slate::heev computes every eigenpair */
  return gs2_eigen<double>(n, a, desca, w, z, descz);
}

int openmx_gs2_eigen_complex(int n, int nev, void *a, int *desca,
                             double *w, void *z, int *descz)
{
  (void)nev;
  return gs2_eigen<std::complex<double> >(n, (std::complex<double> *)a, desca,
                                          w, (std::complex<double> *)z, descz);
}

void openmx_gs2_pdgemm(const char *transa, const char *transb,
                       const int *m, const int *n, const int *k,
                       const double *alpha,
                       const double *a, const int *ia, const int *ja, const int *desca,
                       const double *b, const int *ib, const int *jb, const int *descb,
                       const double *beta,
                       double *c, const int *ic, const int *jc, const int *descc)
{
  int whole = (*ia == 1 && *ja == 1 && *ib == 1 && *jb == 1 && *ic == 1 && *jc == 1);
  gs2_gemm<double>(transa, transb, *m, *n, *k, *alpha,
                   (double *)a, (int *)desca, (double *)b, (int *)descb,
                   *beta, c, (int *)descc, whole);
}

void openmx_gs2_pzgemm(const char *transa, const char *transb,
                       const int *m, const int *n, const int *k,
                       const double *alpha,
                       const double *a, const int *ia, const int *ja, const int *desca,
                       const double *b, const int *ib, const int *jb, const int *descb,
                       const double *beta,
                       double *c, const int *ic, const int *jc, const int *descc)
{
  int whole = (*ia == 1 && *ja == 1 && *ib == 1 && *jb == 1 && *ic == 1 && *jc == 1);
  gs2_gemm<std::complex<double> >(transa, transb, *m, *n, *k,
                                  std::complex<double>(alpha[0], alpha[1]),
                                  (std::complex<double> *)a, (int *)desca,
                                  (std::complex<double> *)b, (int *)descb,
                                  std::complex<double>(beta[0], beta[1]),
                                  (std::complex<double> *)c, (int *)descc, whole);
}

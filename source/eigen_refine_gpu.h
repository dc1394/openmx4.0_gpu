/* Refined FP32 eigensolver of the dense GPU paths: an FP32 solve of all
   eigenpairs of a real symmetric or complex Hermitian matrix, Ogita-Aishima
   refinement of the wanted ones with the products through the GEMMul8
   bridge, a Rayleigh-Ritz step inside the clusters of nearly degenerate,
   partially occupied states, and a warm start from the vectors of the
   previous solve.  Shared by the cluster and band solvers; the stage control
   (which solve is refined, the FP64 re-solve at the stop) stays with each
   solver.  Every matrix is column-major on the device; a complex element is
   two doubles, real and imaginary part. */
#ifndef OPENMX_EIGEN_REFINE_GPU_H
#define OPENMX_EIGEN_REFINE_GPU_H

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <stddef.h>
#include <stdint.h>

/* The cuSOLVER and cuBLAS state of a solver: the refinement borrows its
   handles, its stream and its device workspace (grown through try_malloc,
   which returns cudaSuccess or an error without aborting). */
typedef struct {
    cublasHandle_t     cublas;          /* on the default stream */
    cusolverDnHandle_t cusolver;        /* on stream */
    cudaStream_t       stream;
    void             **d_work;
    size_t            *d_work_bytes;
    void             **h_work;
    size_t            *h_work_bytes;
    int32_t           *d_info;
    cudaError_t (*try_malloc)(void **ptr, size_t bytes);
    void *(*host_malloc)(size_t count, size_t elem, const char *label);
    cusolverDnParams_t params32;        /* two-stage algorithm of the FP32 full solve, once tried */
    int                params32_tried;
} EigenRefineDevice;

/* One eigenproblem instance (a spin, a k-point): what persists between its
   solves. */
typedef struct {
    int     cplx;
    int     capacity;      /* elements of lam and occ */
    double *lam;           /* eigenvalue estimates of all n states, device */
    double *occ;           /* their occupations, device (scratch) */
    int     basis_valid;   /* x holds the n vectors of the latest refined solve */
    int     basis_n;
    double *b1, *b2, *b3;  /* the product blocks of the solve in progress (prepare to finish); b3 holds
                              Y = A X1 while the unrefined columns are rotated against the refined ones
                              (NULL when every column is refined) */
    void   *own;           /* their allocation when the FP32 scratch has no room */
    int     products_ready; /* a warm prepare left the first step's products G in b1 and S in b2 */
    double  delta_cold;    /* delta of the first refinement step after the latest FP32 solve: a warm start's floor */
} EigenRefineState;

/* Fermi occupation of a state with estimate e and 1-based index (for
   orbitals emptied by index); ctx holds the chemical potential. */
typedef double (*EigenRefineOccupation)(double e, int index, void *ctx);

typedef struct {
    int     cplx;
    int     n;             /* dimension (2n of the non-collinear solvers) */
    int     maxn;          /* wanted states */
    int     iterations;    /* Newton steps */
    void   *a;             /* n x n, device: on return its first maxn columns hold the refined vectors */
    double *w;             /* maxn doubles, device: the refined eigenvalues */
    void   *fp32;          /* device scratch of at least (cplx ? 2 : 1) n^2 floats + (n + 1) floats */
    size_t  region;        /* bytes from fp32 on that the product blocks may take once the FP32 solve is done */
    void   *x;             /* n x n, device: the basis (all n vectors); persists for the warm start */
    int     transient;     /* failures of this solve do not condemn the rest of the SCF cycle */
    int     warm;          /* start from x and lam instead of an FP32 solve */
    int     defaulted;     /* the refinement is only on by default: a native fallback of its products ends it */
    EigenRefineOccupation occupation;
    void   *occupation_ctx;
} EigenRefineProblem;

typedef struct {
    int    persistent;     /* 0 no, 1 FP64 for the rest of the cycle, 2 FP64 for the rest of the run */
    int    columns;
    double max_r, max_s, delta, chain;
    int    rr_clusters, rr_largest;
    int    fp32_solved;    /* an FP32 solve ran (not a warm start) */
} EigenRefineReport;

/* K of OPENMX_EIGEN_REFINE, or the device default (2 on GPUs whose FP32:FP64
   throughput ratio is at least 8 when the complex GEMMul8 products are on);
   *ratio receives the ratio when the default applied, else 0; reports the
   decision on stdout when verbose.  Called on the root rank only. */
int openmx_eigen_refine_configure(int cplx, int verbose, int *ratio);

/* OPENMX_EIGEN_REFINE_UNTIL (printed NormRD, 0 = off); OPENMX_EIGEN_REFINE_WARM
   (default 1); OPENMX_EIGEN_REFINE_WARM_BAND (the band solvers, default 1);
   OPENMX_EIGEN_REFINE_WARM_PERIOD (the band solvers solve in FP32 again every
   so many SCF steps instead of warm-starting; default 0 = never: with the
   unrefined columns rotated against the refined ones a chain of warm starts
   agrees with FP64 to 2e-12 Ha on sidia333, where it used to settle 2e-10 Ha
   away) */
double openmx_eigen_refine_until(void);
int    openmx_eigen_refine_warm_enabled(void);
int    openmx_eigen_refine_warm_band_enabled(void);
int    openmx_eigen_refine_warm_band_period(void);
/* a band solve of SCF step scf_iter may start warm */
int    openmx_eigen_refine_warm_band_step(int scf_iter);

/* The first half of a refined solve: a is made symmetric or Hermitian from
   its lower triangle, then either (warm) the previous estimates are read
   from lam and the first refinement step's products are formed on the old
   vectors, whose maxima say whether those are still within 2e-5 of the new
   matrix, or all n eigenpairs are solved in FP32 into fp32 and copied to x
   as the basis.
   e0 (n doubles, host) receives the estimates, from which the caller
   computes occupations.  Returns 1 on success, 0 on failure with
   report->persistent set, -1 when a warm start found the old vectors too
   far from a (nothing changed: the caller calls again without warm). */
int openmx_eigen_refine_prepare(EigenRefineState *st, EigenRefineDevice *dev, const EigenRefineProblem *pb,
                                double *e0, EigenRefineReport *report);

/* The second half: f0 (n doubles, host) are the occupations of the states of
   e0.  Returns 1 on success, 0 on failure (report->persistent). */
int openmx_eigen_refine_finish(EigenRefineState *st, EigenRefineDevice *dev, const EigenRefineProblem *pb,
                               const double *f0, EigenRefineReport *report);

/* A prepare whose finish will not follow (the solve goes to FP64 after
   all): releases what prepare left behind. */
void openmx_eigen_refine_abandon(EigenRefineState *st);

void openmx_eigen_refine_state_release(EigenRefineState *st);
void openmx_eigen_refine_device_release(EigenRefineDevice *dev);

#endif

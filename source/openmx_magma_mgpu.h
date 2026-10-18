#ifndef OPENMX_MAGMA_MGPU_H
#define OPENMX_MAGMA_MGPU_H
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    OPENMX_MAGMA_SUCCESS = 0,
    OPENMX_MAGMA_UNAVAILABLE = 1,
    OPENMX_MAGMA_INVALID = 2,
    OPENMX_MAGMA_RESOURCES = 3,
    OPENMX_MAGMA_LIBRARY_ERROR = 4,
    OPENMX_MAGMA_VALIDATION_ERROR = 5,
    OPENMX_MAGMA_CUDA_ERROR = 6
};

/* Compile-time availability; this does not create any CUDA contexts. */
int openmx_magma_mgpu_available(void);
/* Non-prefix groups require the versioned device-map extension in the pinned
   private dependency. Unpatched external MAGMA still supports prefix groups. */
int openmx_magma_mgpu_available_devices(int ngpu, const int *devices);
/* Conservative per-device live scratch bound, including the reserve.
   SIZE_MAX denotes an unsupported order/group or arithmetic overflow. */
size_t openmx_magma_mgpu_device_bytes(int ngpu, int n, int is_complex);
const char *openmx_magma_mgpu_last_error(void);
/* Release retained host scratch at an input boundary. No CUDA allocations
   are retained; successful MAGMA initialization remains until process exit. */
void openmx_magma_mgpu_release_cache(void);

/* Host column-major symmetric/Hermitian solve using visible GPUs 0..ngpu-1.
   The caller must provide exclusive node-wide admission for those GPUs; this
   layer serializes threads in this process only. 2 <= ngpu <= 8. The calling
   thread's current CUDA device is restored before returning.

   Optional OPENMX_MAGMA_FP32 refinement is admitted only when every
   CUDA-visible GPU reports a known FP32:FP64 throughput ratio >= 8.
   Strong/unknown FP64 capability anywhere in the visible allocation
   retains the native FP64 driver, including outside a selected group.

   first/last are inclusive, 1-based eigenvalue indices; 1,n requests all.
   is_complex selects interleaved {real,imag} doubles. lower selects the input
   triangle. The other triangle need not be initialized. W has last-first+1
   entries; successful vectors occupy that many leading columns of A, with
   the original lda. If want_vectors==0 A is unchanged, and eigenvalue sanity
   rather than a residual is checked. On every nonzero return A and W remain
   unchanged, allowing the caller's ordinary solver fallback. Inputs/outputs
   must not overlap. n==0 accepts first=1,last=0 and is a no-op.

   Uses native FP64 MAGMA one-stage divide-and-conquer by default. Optional
   OPENMX_MAGMA_FP32=1 uses a complete FP32 basis with FP64 correction when
   vector output and the application FP32 policy are enabled; an unsuccessful
   correction retries native FP64 with the original input. Both paths use
   privately owned host workspaces, finite/sorted eigenvalue checks and sampled independent CPU
   residual/orthogonality checks (32 spread rows and four vectors by default;
   OPENMX_MAGMA_VALIDATE_FULL=1 checks all rows of those vectors). Every
   output entry is checked for finiteness. One host workspace is retained up
   to OPENMX_MAGMA_HOST_CACHE_MB (64 MiB total per process, 0 disables); larger
   solves use transient storage. Device scratch is never retained here.
   Build without OPENMX_HAVE_MAGMA for a dependency-free unavailable stub. */
int openmx_magma_mgpu_solve_host(int ngpu, int is_complex, int want_vectors,
    int lower, int n, int first, int last, void *A, int lda, double *W);

/* Same transaction contract, using distinct visible CUDA device ordinals in
   the supplied order. Does not modify CUDA_VISIBLE_DEVICES or rank binding.
   Its process-wide MAGMA logical-device map is scoped under the adapter's
   mutex and removed before restoring the caller's physical CUDA device. */
int openmx_magma_mgpu_solve_host_devices(int ngpu, const int *devices,
    int is_complex, int want_vectors, int lower, int n, int first, int last,
    void *A, int lda, double *W);

/* Leading eigenpairs, vectors requested, lower triangle supplied. */
int openmx_magma_mgpu_dsyevdx_host(int ngpu, int n, int maxn,
    double *A, int lda, double *W);
int openmx_magma_mgpu_zheevdx_host(int ngpu, int n, int maxn,
    void *A, int lda, double *W);

#ifdef __cplusplus
}
#endif
#endif

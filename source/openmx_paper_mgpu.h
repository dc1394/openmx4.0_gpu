#ifndef OPENMX_PAPER_MGPU_H
#define OPENMX_PAPER_MGPU_H

#ifdef __cplusplus
extern "C" {
#endif

/* Visible CUDA device count (zero when the runtime is unavailable). */
int openmx_paper_mgpu_available(void);

/* Real symmetric, column-major host interface. Reads the lower triangle.
 * On success, W contains nev ascending eigenvalues and the first nev
 * columns of A contain their eigenvectors. Devices [0,ngpu) are used.
 * Returns zero on success. A and W are unchanged on any failure.
 * The caller must serialize calls with other node-wide multi-GPU solvers.
 */
int openmx_paper_mgpu_dsyevdx_host(int ngpu, int n, int nev,
                                  double *A, int lda, double *W);
/* Hermitian extension: interleaved {real,imag} doubles, lower triangle;
 * imaginary diagonal entries are ignored. The paper evaluates real matrices;
 * this extension uses conjugate Householders and tridiagonal phase reduction.
 * The output/transactional contract is the same as the real entry point. */
int openmx_paper_mgpu_zheevdx_host(int ngpu, int n, int nev,
                                 void *A, int lda, double *W);
const char *openmx_paper_mgpu_last_error(void);

#ifdef __cplusplus
}
#endif
#endif

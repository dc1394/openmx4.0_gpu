#ifndef OPENMX_EXACT_EXCHANGE_HOST_H
#define OPENMX_EXACT_EXCHANGE_HOST_H

/* Matrix products of the exchange build on the host
   (docs/EXX_IMPLEMENTATION_GUIDE.md, section 11).

   The threads of the build are those of OpenMP of the main program (-nt).
   modules/exx runs its loops over the ABFs of a centre atom on them, through
   the loop of its table of functions (section 7.2), and
   Exact_Exchange_Host_Dgemm divides a large product among them. Every call
   of the BLAS then computes on the thread that makes it; the build does not
   depend on a BLAS with threads. The BLAS must allow calls of DGEMM from
   several threads at the same time.

   A BLAS that has threads of its own must not use them meanwhile: they
   would share the cores with the threads of OpenMP, and such a library
   serves one caller at a time. OpenBLAS can be built that way (its pthread
   variant). Exact_Exchange_Host_Blas_Single asks it to compute on the
   calling thread, through functions that a program linked with another BLAS
   does not have. A BLAS that computes on the threads of OpenMP, or on one
   thread, is left as it is: inside a parallel region it stays on the
   calling thread.

   Only OpenBLAS is recognized, and only when the program has its functions
   openblas_get_parallel, openblas_get_num_threads and
   openblas_set_num_threads. The references to them are weak, so a library
   that is linked statically gives them only when something else in the
   program uses them. Any other BLAS with threads of its own keeps them.
   The builds of this repository link MKL and AOCL as sequential libraries
   or on the threads of OpenMP (cmake/OpenMXHostLinalg.cmake). */

#ifdef __cplusplus
extern "C" {
#endif

/* The number of calls of the BLAS into which Exact_Exchange_Host_Dgemm
   divides a product: at most threads and at most the longer side of c, and
   so few that every call has at least 2.5e5 multiplications m*n*k. A
   smaller share is not worth the start of a parallel region. 1 means one
   call for the whole product. */
int Exact_Exchange_Host_Pieces(int threads, int m, int n, int k);

/* c = alpha*op(a)*op(b) + beta*c, column-major, with the arguments of DGEMM,
   by the BLAS of the main program. With more than one piece the longer side
   of c is divided into blocks of nearly equal length: the rows of c when it
   has at least as many rows as columns, its columns otherwise. The threads
   of OpenMP compute the blocks, each block with one call of the BLAS on the
   thread that has it, and the calls run at the same time. For one number of
   pieces the result does not depend on the timing of the threads; it can
   differ by rounding from the result of one call, as far as the BLAS rounds
   a block differently. Without OpenMP the blocks are computed one after
   another. */
void Exact_Exchange_Host_Dgemm(int threads, char transa, char transb, int m, int n, int k, double alpha,
                               const double *a, int lda, const double *b, int ldb, double beta, double *c,
                               int ldc);

/* Tells a BLAS with threads of its own to compute on the calling thread and
   returns the number of threads it had; returns 0 for every other BLAS,
   which is left as it is. Exact_Exchange_Host_Blas_Restore takes the
   returned number and gives the threads back; with 0 it does nothing. */
int Exact_Exchange_Host_Blas_Single(void);
void Exact_Exchange_Host_Blas_Restore(int threads);

#ifdef __cplusplus
}
#endif

#endif /* OPENMX_EXACT_EXCHANGE_HOST_H */

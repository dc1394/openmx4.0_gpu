/**********************************************************************
  Exact_Exchange_Host.c:

     Matrix products of the exchange build on the host, on the threads
     of OpenMP of the main program.

     See docs/EXX_IMPLEMENTATION_GUIDE.md, section 11, and
     openmx_exact_exchange_host.h for the interface. The functions take
     every input as an argument and read no global variable.
***********************************************************************/

#include <stddef.h>
#include "f77func.h"
#include "openmx_exact_exchange_host.h"

/* DGEMM of the BLAS of the main program, as lapack_prototypes.h declares
   it. That header needs mpi.h, which this file does not use. */
int F77_NAME(dgemm,DGEMM)(char *transa, char *transb, int *m, int *n, int *k, double *alpha, double *a, int *lda,
                          double *b, int *ldb, double *beta, double *c, int *ldc);

/* The functions of OpenBLAS that tell how it was built and set its number
   of threads. A weak reference is null in a program without them. */
#if defined(__GNUC__) || defined(__clang__) || defined(__NVCOMPILER) || defined(__PGI)
extern int openblas_get_parallel(void) __attribute__((weak));
extern int openblas_get_num_threads(void) __attribute__((weak));
extern void openblas_set_num_threads(int) __attribute__((weak));
#define OPENMX_EXX_OPENBLAS_FUNCTIONS 1
#endif

int Exact_Exchange_Host_Blas_Single(void)
{
#ifdef OPENMX_EXX_OPENBLAS_FUNCTIONS
  /* openblas_get_parallel: 0 one thread, 1 threads of its own, 2 OpenMP */
  if (openblas_get_parallel!=NULL && openblas_get_num_threads!=NULL && openblas_set_num_threads!=NULL
      && openblas_get_parallel()==1) {
    const int threads=openblas_get_num_threads();
    if (1<threads) {
      openblas_set_num_threads(1);
      return threads;
    }
  }
#endif
  return 0;
}

void Exact_Exchange_Host_Blas_Restore(int threads)
{
#ifdef OPENMX_EXX_OPENBLAS_FUNCTIONS
  if (0<threads && openblas_set_num_threads!=NULL) openblas_set_num_threads(threads);
#else
  (void)threads;
#endif
}

/* The multiplications m*n*k that every call of a divided product has at
   least. */
#define OPENMX_EXX_HOST_PIECE_WORK 2.5e5

int Exact_Exchange_Host_Pieces(int threads, int m, int n, int k)
{
  if (threads<2 || m<1 || n<1 || k<1) return 1;
  const double work=(double)m*(double)n*(double)k;
  const int length=m<n ? n : m;
  int pieces=threads<length ? threads : length;
  if (work<(double)pieces*OPENMX_EXX_HOST_PIECE_WORK) pieces = (int)(work/OPENMX_EXX_HOST_PIECE_WORK);
  return pieces<2 ? 1 : pieces;
}

void Exact_Exchange_Host_Dgemm(int threads, char transa, char transb, int m, int n, int k, double alpha,
                               const double *a, int lda, const double *b, int ldb, double beta, double *c,
                               int ldc)
{
  const int pieces=Exact_Exchange_Host_Pieces(threads,m,n,k);
  if (pieces<2) {
    F77_NAME(dgemm,DGEMM)(&transa,&transb,&m,&n,&k,&alpha,(double*)a,&lda,(double*)b,&ldb,&beta,c,&ldc);
    return;
  }
  /* op(a) is m x k and op(b) is k x n: a block of rows of c takes the same
     rows of op(a), a block of columns of c the same columns of op(b). The
     rows of op(a) are the columns of a when a is transposed, and the
     columns of op(b) are the rows of b when b is. */
  const int rows=n<=m;
  const int length=rows ? m : n;
  const int plain_a=transa=='N' || transa=='n',plain_b=transb=='N' || transb=='n';
#ifdef _OPENMP
#pragma omp parallel for schedule(static,1) num_threads(pieces)
#endif
  for (int piece=0; piece<pieces; piece++) {
    const int first=(int)((long long)piece*length/pieces);
    /* The BLAS takes its arguments by address: every call has its own. */
    char ta=transa,tb=transb;
    int count=(int)((long long)(piece+1)*length/pieces)-first;
    int mm=m,nn=n,kk=k,la=lda,lb=ldb,lc=ldc;
    double scale=alpha,keep=beta;
    if (rows) {
      const double *pa=plain_a ? a+first : a+(size_t)first*(size_t)lda;
      F77_NAME(dgemm,DGEMM)(&ta,&tb,&count,&nn,&kk,&scale,(double*)pa,&la,(double*)b,&lb,&keep,c+first,&lc);
    }
    else {
      const double *pb=plain_b ? b+(size_t)first*(size_t)ldb : b+first;
      F77_NAME(dgemm,DGEMM)(&ta,&tb,&mm,&count,&kk,&scale,(double*)a,&la,(double*)pb,&lb,&keep,
                            c+(size_t)first*(size_t)ldc,&lc);
    }
  }
}

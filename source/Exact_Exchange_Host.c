/**********************************************************************
  Exact_Exchange_Host.c:

     Matrix products of the exchange build on the host, on the threads
     of OpenMP of the main program.

     See docs/EXX_IMPLEMENTATION_GUIDE.md, section 8 and item 56 of
     section 13, and openmx_exact_exchange_host.h for the interface. The
     functions take every input as an argument and read no global
     variable.
***********************************************************************/

#include <stddef.h>
#include <string.h>
#include "f77func.h"
#include "openmx_exact_exchange_host.h"
#ifdef _OPENMP
#include <omp.h>
#endif

/* DGEMM of the BLAS of the main program, as lapack_prototypes.h declares
   it. That header needs mpi.h, which this file does not use. */
int F77_NAME(dgemm,DGEMM)(char *transa, char *transb, int *m, int *n, int *k, double *alpha, double *a, int *lda,
                          double *b, int *ldb, double *beta, double *c, int *ldc);

/* The functions of OpenBLAS that tell how it was built and set its number
   of threads. OPENMX_EXX_HOST_OPENBLAS() tells whether the program has
   them.

   A build that knows its BLAS to be OpenBLAS defines
   OPENMX_EXX_HOST_BLAS_OPENBLAS. The functions are then referenced as any
   other function, and the linker takes them from a static library as well.

   Every other build asks the linker with weak references: such a reference
   is null in a program without the functions. Two limits follow.
   - The null tests are right in position-independent code only. In other
     code the linker settles them when it finds the functions in a shared
     library, and a program that later runs with another library of that
     name would call a null pointer. The builds compile this file with
     -fPIC.
   - A static OpenBLAS is not recognized: openblas_get_parallel is alone in
     its member of the archive, and a weak reference does not bring a
     member into the program.
   Mach-O wants a weak reference to be found when the program is linked, so
   on macOS only the first form is used. */
#if defined(OPENMX_EXX_HOST_BLAS_OPENBLAS)
int openblas_get_parallel(void);
int openblas_get_num_threads(void);
void openblas_set_num_threads(int);
#define OPENMX_EXX_HOST_OPENBLAS() 1
#elif !defined(__APPLE__) && (defined(__GNUC__) || defined(__clang__) || defined(__NVCOMPILER) || defined(__PGI))
extern int openblas_get_parallel(void) __attribute__((weak));
extern int openblas_get_num_threads(void) __attribute__((weak));
extern void openblas_set_num_threads(int) __attribute__((weak));
#define OPENMX_EXX_HOST_OPENBLAS() \
  (openblas_get_parallel!=NULL && openblas_get_num_threads!=NULL && openblas_set_num_threads!=NULL)
#endif

int Exact_Exchange_Host_Blas_Single(void)
{
#ifdef OPENMX_EXX_HOST_OPENBLAS
  /* openblas_get_parallel: 0 one thread, 1 threads of its own, 2 OpenMP */
  if (OPENMX_EXX_HOST_OPENBLAS() && openblas_get_parallel()==1) {
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
#ifdef OPENMX_EXX_HOST_OPENBLAS
  if (0<threads && OPENMX_EXX_HOST_OPENBLAS()) openblas_set_num_threads(threads);
#else
  (void)threads;
#endif
}

/* The multiplications m*n*k that the calls of a divided product have on
   average at least. The blocks differ in length by one at most, so a call
   has more than half of it. */
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
  /* The team has the size of the team of the loops below, also for fewer
     pieces: block i is then the work of thread i, and the runtime keeps its
     threads from one region of the build to the next. */
#ifdef _OPENMP
#pragma omp parallel for schedule(static,1) num_threads(threads)
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

static void Exact_Exchange_Host_Note(long long *count, double *largest_work, int *largest, int m, int n, int k)
{
  const double work=(double)m*(double)n*(double)k;
  (*count)++;
  if (*largest_work<work) {
    *largest_work = work;
    largest[0] = m;
    largest[1] = n;
    largest[2] = k;
  }
}

void Exact_Exchange_Host_Note_Product(OpenMXExactExchangeHostProducts *products, int m, int n, int k)
{
  Exact_Exchange_Host_Note(&products->count,&products->largest_work,products->largest,m,n,k);
}

void Exact_Exchange_Host_Loop_Dgemm(OpenMXExactExchangeHostProducts *products, char transa, char transb, int m,
                                    int n, int k, double alpha, const double *a, int lda, const double *b,
                                    int ldb, double beta, double *c, int ldc)
{
  int thread=0;
#ifdef _OPENMP
  thread = omp_get_thread_num();
#endif
  OpenMXExactExchangeLoopProducts *mine=&products->loop[thread];
  Exact_Exchange_Host_Note(&mine->count,&mine->largest_work,mine->largest,m,n,k);
  F77_NAME(dgemm,DGEMM)(&transa,&transb,&m,&n,&k,&alpha,(double*)a,&lda,(double*)b,&ldb,&beta,c,&ldc);
}

int Exact_Exchange_Host_Parallel_For(OpenMXExactExchangeHostProducts *products, int count,
                                     void (*body)(void *argument, int index), void *argument)
{
  /* No loops, or a call from a body of the loop that runs: the calls in
     order on the calling thread. The counters and in_loop stay as they
     are, so the products of these calls are counted where the caller's
     are. */
  if (products->threads<2 || products->loop==NULL || products->in_loop) {
    for (int index=0; index<count; index++) body(argument,index);
    return 0;
  }
  for (int thread=0; thread<products->threads; thread++) {
    products->loop[thread].count = 0;
    products->loop[thread].largest_work = products->largest_work;
  }
  products->in_loop = 1;
#ifdef _OPENMP
#pragma omp parallel for schedule(static,1) num_threads(products->threads)
#endif
  for (int index=0; index<count; index++) body(argument,index);
  products->in_loop = 0;
  for (int thread=0; thread<products->threads; thread++) {
    const OpenMXExactExchangeLoopProducts *part=&products->loop[thread];
    products->count += part->count;
    if (products->largest_work<part->largest_work) {
      products->largest_work = part->largest_work;
      memcpy(products->largest,part->largest,sizeof(products->largest));
    }
  }
  return 0;
}

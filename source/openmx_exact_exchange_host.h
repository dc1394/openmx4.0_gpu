#ifndef OPENMX_EXACT_EXCHANGE_HOST_H
#define OPENMX_EXACT_EXCHANGE_HOST_H

/* Matrix products of the exchange build on the host
   (docs/EXX_IMPLEMENTATION_GUIDE.md, section 8 and item 56 of section 13).

   The threads of the build are those of OpenMP of the main program (-nt),
   whatever number of threads the BLAS has. modules/exx runs its loops over
   the ABFs of a centre atom on them, through the loop of its table of
   functions (section 7.2: Exact_Exchange_Host_Parallel_For), and
   Exact_Exchange_Host_Dgemm divides a large product among them. The calls
   of the BLAS in the loops and the calls of a divided product compute on
   the thread that makes them; the build does not depend on a BLAS with
   threads. A product that is not divided is one call from the thread that
   called the build, outside any parallel region: a BLAS on the threads of
   OpenMP may compute it with its team.

   This asks two things of the BLAS, and nothing checks them when the
   program runs. DGEMM must allow calls from several threads at the same
   time. And for the result of a build to be the same in every run with one
   number of threads, DGEMM must give the same bits for the same arguments
   on every thread and wherever its arrays lie in memory: every piece of a
   loop has work arrays of its own.

   A BLAS that has threads of its own must not use them meanwhile: they
   would share the cores with the threads of OpenMP, and such a library
   serves one caller at a time. OpenBLAS can be built that way (its pthread
   variant). Exact_Exchange_Host_Blas_Single asks it to compute on the
   calling thread, through functions that a program linked with another BLAS
   does not have. A BLAS that computes on the threads of OpenMP, or on one
   thread, is left as it is: it is expected to stay on the calling thread
   inside a parallel region, as a library on the OpenMP runtime of the
   program does while nested regions are off.

   Only OpenBLAS is recognized, by its functions openblas_get_parallel,
   openblas_get_num_threads and openblas_set_num_threads. The CPU build
   that has chosen OpenBLAS itself (OPENMX_CPU_LINALG of OPENBLAS, or AUTO
   that found it) compiles the unit with OPENMX_EXX_HOST_BLAS_OPENBLAS and
   so finds a shared and a static library. Every other build looks for the
   functions through weak references: it finds a shared OpenBLAS, and not a
   static one, in which the linker leaves openblas_get_parallel out. Any
   other BLAS with threads of its own keeps them. The builds of this
   repository link MKL and AOCL as sequential libraries or on the threads of
   OpenMP (cmake/OpenMXHostLinalg.cmake). */

#ifdef __cplusplus
extern "C" {
#endif

/* The number of calls of the BLAS into which Exact_Exchange_Host_Dgemm
   divides a product: at most threads and at most the longer side of c, and
   so few that the calls have on average at least 2.5e5 multiplications
   m*n*k (every call then has more than half of that). A smaller share is
   not worth the start of a parallel region. 1 means one call for the whole
   product. */
int Exact_Exchange_Host_Pieces(int threads, int m, int n, int k);

/* c = alpha*op(a)*op(b) + beta*c, column-major, with the arguments of DGEMM,
   by the BLAS of the main program. With more than one piece the longer side
   of c is divided into blocks of nearly equal length: the rows of c when it
   has at least as many rows as columns, its columns otherwise. A team of
   OpenMP threads, as many as the argument threads says, computes the
   blocks: thread i has block i, with one call of the BLAS, and the calls
   run at the same time. The team has the size of the team of
   Exact_Exchange_Host_Parallel_For also when the product has fewer pieces:
   a runtime that ends the threads of a team that shrinks would otherwise
   start them again for the next loop. For one number of pieces the result
   does not depend on the timing of the threads; it can differ by rounding
   from the result of one call, as far as the BLAS rounds a block
   differently. Without OpenMP the blocks are computed one after another. */
void Exact_Exchange_Host_Dgemm(int threads, char transa, char transb, int m, int n, int k, double alpha,
                               const double *a, int lda, const double *b, int ldb, double beta, double *c,
                               int ldc);

/* The matrix products that one thread computes in a loop of the build. The
   padding keeps the numbers of two threads in different cache lines. */
typedef struct {
  long long count;
  double largest_work;
  int largest[3];
  char padding[128-sizeof(long long)-sizeof(double)-3*sizeof(int)];
} OpenMXExactExchangeLoopProducts;

/* The matrix products of one build on this process and the state of its
   loops. The caller sets count, largest_work, largest and in_loop to zero,
   threads to the number of threads of the loops, and loop to an array of
   threads entries; with threads below 2 there are no loops and loop may be
   NULL. */
typedef struct {
  long long count;       /* products of the build */
  double largest_work;   /* the most multiplications m*n*k that one of them has */
  int largest[3];        /* m, n and k of that product */
  int threads;           /* threads of the loops of the build; below 2: no loops */
  OpenMXExactExchangeLoopProducts *loop; /* the products of every thread in the loop that runs */
  int in_loop;           /* a loop runs: its products are counted by thread */
} OpenMXExactExchangeHostProducts;

/* Counts a product that is called outside the loops, on the thread that
   called the build. */
void Exact_Exchange_Host_Note_Product(OpenMXExactExchangeHostProducts *products, int m, int n, int k);

/* A product that a body of Exact_Exchange_Host_Parallel_For calls: counted
   for the calling thread and computed by one call of the BLAS on that
   thread. Bodies call it at the same time. */
void Exact_Exchange_Host_Loop_Dgemm(OpenMXExactExchangeHostProducts *products, char transa, char transb, int m,
                                    int n, int k, double alpha, const double *a, int lda, const double *b,
                                    int ldb, double beta, double *c, int ldc);

/* The loop of the table of functions of modules/exx: the calls
   body(argument,0) to body(argument,count-1) on a team of products->threads
   threads of OpenMP, thread i with the indices i, i+threads, ... one after
   the other. Returns 0 when every call has returned. The products of the
   bodies (Exact_Exchange_Host_Loop_Dgemm) are counted by thread and added to
   those of the build afterwards; of two products with the same number of
   multiplications the one of the lower thread is kept as the largest.

   One loop runs at a time: modules/exx calls its loops from the thread that
   called the build and never from a body. A call from a body does not start
   a second team and does not touch the counters of the other threads: its
   bodies run on the calling thread, in order, and their products count for
   that thread. Without OpenMP the bodies run in order on the calling
   thread. With threads below 2, or without the array loop, the function
   only calls the bodies in order: their products are then products outside
   loops. */
int Exact_Exchange_Host_Parallel_For(OpenMXExactExchangeHostProducts *products, int count,
                                     void (*body)(void *argument, int index), void *argument);

/* Tells a BLAS with threads of its own to compute on the calling thread and
   returns the number of threads it had. Returns 0 when it changed nothing:
   for every other BLAS, and for a BLAS with threads of its own that already
   computes on one thread. Exact_Exchange_Host_Blas_Restore takes the
   returned number and gives the threads back; with 0 it does nothing. */
int Exact_Exchange_Host_Blas_Single(void);
void Exact_Exchange_Host_Blas_Restore(int threads);

#ifdef __cplusplus
}
#endif

#endif /* OPENMX_EXACT_EXCHANGE_HOST_H */

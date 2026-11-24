/* Matrix products of the exchange build on the host
   (source/Exact_Exchange_Host.c), run against a model of the BLAS: this file
   defines DGEMM, and the functions of OpenBLAS that tell how the library
   was built and set its number of threads.

   The model of DGEMM refuses the arguments that DGEMM refuses, notes every
   call with the thread that made it, and counts how often it writes every
   element of the result. It sums the terms of an element in one order, so
   the result of a divided product has the bits of the result of one call.
   The arrays have the size that DGEMM needs and no more; the rows of an
   array beyond the matrix hold NaN in the factors and a mark in the result.

   The test asks four things of the products of the unit: the number of
   calls is the one of the rule in openmx_exact_exchange_host.h, a divided
   product equals the product computed element by element and writes every
   element once and nothing else, its calls take blocks of nearly equal
   length of the longer side of the result, block i on thread i of a team
   of all threads when the program has OpenMP, and only a BLAS with threads
   of its own has its number of threads changed and given back.

   Of the loop of the build it asks: every index is called once, index i on
   thread i modulo the team; the products of the bodies are one call of
   DGEMM each on the thread of the body and give the product computed
   element by element; their number and the largest of them are those of a
   count in order, whatever the team is; nothing is written beside the
   counters of the threads; and a loop called from a body runs on the
   thread of that body.

   For products whose arrays have more elements than an int counts, the
   model only notes its arguments: the arrays are address space without
   memory, and the blocks and addresses of the calls are compared with the
   division of the header.

   The functions of OpenBLAS come in four forms. As the program is compiled
   by default it has all three, as one that is linked with a shared
   OpenBLAS. With OPENMX_TEST_WITHOUT_OPENBLAS it has none, as one linked
   with another BLAS. With OPENMX_TEST_WITHOUT_OPENBLAS_PARALLEL it lacks
   openblas_get_parallel only, as one linked with a static OpenBLAS. With
   OPENMX_EXX_HOST_BLAS_OPENBLAS, which the unit reads as well, the unit
   refers to the three functions directly, as in a build that has chosen
   OpenBLAS itself.

   On macOS the unit looks for the functions in the last form only. In the
   other forms it leaves the BLAS of the program as it is, and the test
   expects that. */

#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(__unix__) || defined(__APPLE__)
#include <sys/mman.h>
#if !defined(MAP_ANONYMOUS) && defined(MAP_ANON)
#define MAP_ANONYMOUS MAP_ANON
#endif
#endif
#ifdef _OPENMP
#include <omp.h>
#endif
#include "f77func.h"
#include "openmx_exact_exchange_host.h"

/* The unit finds the functions of OpenBLAS in this program: it has all
   three, and the unit looks for them. */
#if !defined(OPENMX_TEST_WITHOUT_OPENBLAS) && !defined(OPENMX_TEST_WITHOUT_OPENBLAS_PARALLEL) \
    && (defined(OPENMX_EXX_HOST_BLAS_OPENBLAS) || !defined(__APPLE__))
#define OPENMX_TEST_OPENBLAS_FOUND 1
#endif

int F77_NAME(dgemm,DGEMM)(char *transa, char *transb, int *m, int *n, int *k, double *alpha, double *a, int *lda,
                          double *b, int *ldb, double *beta, double *c, int *ldc);

enum { MAX_CALLS = 128 };

typedef struct {
  char transa,transb;
  int m,n,k,lda,ldb,ldc;
  double alpha,beta;
  const double *a,*b;
  double *c;
  int thread,team;           /* the thread that called, and the size of its team */
} Call;

typedef struct {
  int calls;
  Call call[MAX_CALLS];
  int wrong;                 /* calls with arguments that DGEMM refuses */
  int notes_only;            /* the arrays have no memory: no element is read or written */
  const double *c;           /* the result of the case, with its dimensions */
  int ldc,n;
  unsigned char *written;    /* how often every element of the result was written */
} Model;

static Model model;

static void check(int ok, const char *message)
{
  if (!ok) {
    fprintf(stderr,"exx host product: %s\n",message);
    exit(1);
  }
}

static void *allocate(size_t count, size_t size)
{
  void *memory=calloc(count<1 ? 1 : count,size);
  check(memory!=NULL,"the test has no memory");
  return memory;
}

static int plain(char letter)
{
  return letter=='N' || letter=='n';
}

static int transposed(char letter)
{
  return letter=='T' || letter=='t' || letter=='C' || letter=='c';
}

int F77_NAME(dgemm,DGEMM)(char *transa, char *transb, int *m, int *n, int *k, double *alpha, double *a, int *lda,
                          double *b, int *ldb, double *beta, double *c, int *ldc)
{
  const int plain_a=plain(*transa),plain_b=plain(*transb);
  const int rows_a=plain_a ? *m : *k,rows_b=plain_b ? *k : *n;
  const int valid=(plain_a || transposed(*transa)) && (plain_b || transposed(*transb))
                  && 0<=*m && 0<=*n && 0<=*k && (rows_a<1 ? 1 : rows_a)<=*lda && (rows_b<1 ? 1 : rows_b)<=*ldb
                  && (*m<1 ? 1 : *m)<=*ldc;
#ifdef _OPENMP
#pragma omp critical(openmx_test_model)
#endif
  {
    if (model.calls<MAX_CALLS) {
      Call *call=&model.call[model.calls];
      call->transa = *transa;
      call->transb = *transb;
      call->m = *m;
      call->n = *n;
      call->k = *k;
      call->lda = *lda;
      call->ldb = *ldb;
      call->ldc = *ldc;
      call->alpha = *alpha;
      call->beta = *beta;
      call->a = a;
      call->b = b;
      call->c = c;
#ifdef _OPENMP
      call->thread = omp_get_thread_num();
      call->team = omp_get_num_threads();
#else
      call->thread = 0;
      call->team = 1;
#endif
    }
    model.calls++;
  }
  if (!valid) {
    /* counted as the writes outside the result below: the two can meet */
#ifdef _OPENMP
#pragma omp atomic
#endif
    model.wrong++;
    return 0;
  }
  if (model.notes_only) return 0;
  for (int j=0; j<*n; j++) {
    for (int i=0; i<*m; i++) {
      double sum=0.0;
      for (int l=0; l<*k; l++) {
        const double x=plain_a ? a[(size_t)i+(size_t)l*(size_t)*lda] : a[(size_t)l+(size_t)i*(size_t)*lda];
        const double y=plain_b ? b[(size_t)l+(size_t)j*(size_t)*ldb] : b[(size_t)j+(size_t)l*(size_t)*ldb];
        sum += x*y;
      }
      double *element=c+(size_t)i+(size_t)j*(size_t)*ldc;
      *element = *beta==0.0 ? *alpha*sum : *alpha*sum+*beta*(*element);
      if (model.c==NULL) continue; /* a case with a result of its own for every call */
      if (model.c<=element && element<model.c+(size_t)model.ldc*(size_t)model.n) {
        unsigned char *count=model.written+(element-model.c);
#ifdef _OPENMP
#pragma omp atomic
#endif
        (*count)++;
      }
      else {
#ifdef _OPENMP
#pragma omp atomic
#endif
        model.wrong++;
      }
    }
  }
  return 0;
}

#ifndef OPENMX_TEST_WITHOUT_OPENBLAS
/* The functions of OpenBLAS. */
typedef struct {
  int parallel;              /* answer of openblas_get_parallel */
  int threads;
  int sets;                  /* calls of openblas_set_num_threads */
  int last_set;
} Blas;

static Blas blas;

int openblas_get_parallel(void);
int openblas_get_num_threads(void);
void openblas_set_num_threads(int threads);

#ifndef OPENMX_TEST_WITHOUT_OPENBLAS_PARALLEL
int openblas_get_parallel(void)
{
  return blas.parallel;
}
#endif

int openblas_get_num_threads(void)
{
  return blas.threads;
}

void openblas_set_num_threads(int threads)
{
  blas.sets++;
  blas.last_set = threads;
  blas.threads = threads;
}
#endif

static unsigned long long seed=0x2545F4914F6CDD1DULL;

static double random_value(void)
{
  seed = seed*6364136223846793005ULL+1442695040888963407ULL;
  return (double)(seed>>11)/9007199254740992.0*2.0-1.0;
}

/* A matrix of rows x columns in an array with leading dimension ld: random
   numbers in the matrix, outside in the rows beyond it. */
static double *matrix(int rows, int columns, int ld, double outside)
{
  double *values=allocate((size_t)ld*(size_t)columns,sizeof(double));
  for (int j=0; j<columns; j++)
    for (int i=0; i<ld; i++) values[(size_t)i+(size_t)j*(size_t)ld] = i<rows ? random_value() : outside;
  return values;
}

/* The pieces of the rule of the header: at most threads, at most the longer
   side of the result, and on average at least 2.5e5 multiplications in a
   call. */
static void pieces_follow_the_rule(void)
{
  static const struct { int threads,m,n,k,pieces; } rule[]={
    {1,1000,1000,1000,1}, {0,1000,1000,1000,1}, {-3,1000,1000,1000,1},
    {8,0,100,100,1}, {8,100,0,100,1}, {8,100,100,0,1}, {8,-1,100,100,1}, {8,100,-1,100,1}, {8,100,100,-1,1},
    {8,10,10,10,1}, {8,100,50,99,1}, {8,100,50,100,2}, {8,50,100,100,2}, {8,100,100,100,4},
    {3,100,100,100,3}, {8,200,100,100,8}, {8,199,100,100,7}, {2,200,100,100,2},
    {64,3,2,400000,3}, {64,2,5,400000,5}, {64,1,1,4000000,1}, {64,400,200,200,64}, {64,400,200,199,63},
    {18,8229,897,1131,18}, {18,20163,897,2171,18}, {64,46340,46340,46340,64}, {INT_MAX,1000,1000,1000,1000}};
  for (size_t i=0; i<sizeof(rule)/sizeof(rule[0]); i++)
    check(Exact_Exchange_Host_Pieces(rule[i].threads,rule[i].m,rule[i].n,rule[i].k)==rule[i].pieces,
          "the number of pieces of a product is not the one of the rule");
}

/* One product: the result against the product computed element by element,
   and the calls of the model against the division of the header. Returns
   the number of calls. */
static int product(int threads, char transa, char transb, int m, int n, int k, int pad_a, int pad_b, int pad_c,
                   double alpha, double beta)
{
  const int plain_a=plain(transa),plain_b=plain(transb);
  const int rows_a=plain_a ? m : k,columns_a=plain_a ? k : m,rows_b=plain_b ? k : n,columns_b=plain_b ? n : k;
  const int lda=(rows_a<1 ? 1 : rows_a)+pad_a,ldb=(rows_b<1 ? 1 : rows_b)+pad_b,ldc=(m<1 ? 1 : m)+pad_c;
  const size_t size_c=(size_t)ldc*(size_t)n;
  const double mark=-12345.6789;
  double *a=matrix(rows_a,columns_a,lda,NAN),*b=matrix(rows_b,columns_b,ldb,NAN);
  /* A product with beta = 0 does not read the result: NaN must not come
     through. */
  double *c=matrix(beta==0.0 ? 0 : m,n,ldc,beta==0.0 ? NAN : mark);
  double *expected=allocate(size_c,sizeof(double));
  if (0<size_c) memcpy(expected,c,size_c*sizeof(double));
  for (int j=0; j<n; j++) {
    for (int i=0; i<m; i++) {
      double sum=0.0;
      for (int l=0; l<k; l++) {
        const double x=plain_a ? a[(size_t)i+(size_t)l*(size_t)lda] : a[(size_t)l+(size_t)i*(size_t)lda];
        const double y=plain_b ? b[(size_t)l+(size_t)j*(size_t)ldb] : b[(size_t)j+(size_t)l*(size_t)ldb];
        sum += x*y;
      }
      double *element=expected+(size_t)i+(size_t)j*(size_t)ldc;
      *element = beta==0.0 ? alpha*sum : alpha*sum+beta*(*element);
    }
  }

  memset(&model,0,sizeof(model));
  model.c = c;
  model.ldc = ldc;
  model.n = n;
  model.written = allocate(size_c,1);
  Exact_Exchange_Host_Dgemm(threads,transa,transb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc);

  const int calls=model.calls;
  check(model.wrong==0,"a call of DGEMM has arguments that DGEMM refuses, or writes outside the result");
  check(calls==Exact_Exchange_Host_Pieces(threads,m,n,k),"a product is not divided into the pieces of the rule");
  check(calls<=MAX_CALLS,"a product has more calls than the test can note");
  check(size_c==0 || memcmp(c,expected,size_c*sizeof(double))==0,
        "a product differs from the product computed element by element");
  for (int j=0; j<n; j++)
    for (int i=0; i<ldc; i++)
      check(model.written[(size_t)i+(size_t)j*(size_t)ldc]==(i<m ? 1 : 0),
            "an element of the result is not written once, or a row beyond the matrix is written");

  if (calls==1) {
    const Call *call=&model.call[0];
    check(call->transa==transa && call->transb==transb && call->m==m && call->n==n && call->k==k
          && call->a==a && call->b==b && call->c==c && call->lda==lda && call->ldb==ldb && call->ldc==ldc
          && call->alpha==alpha && call->beta==beta,"an undivided product does not pass its arguments on");
    check(call->team==1,"an undivided product is called from a team of threads");
  }
  else {
    /* The blocks are those of the longer side of the result. That the calls
       cover the result once is checked above; here every call takes the
       factors of its block, and the lengths differ by one at most. */
    const int rows=n<=m;
    int shortest=INT_MAX,longest=0;
    for (int index=0; index<calls; index++) {
      const Call *call=&model.call[index];
      check(call->transa==transa && call->transb==transb && call->k==k && call->lda==lda && call->ldb==ldb
            && call->ldc==ldc && call->alpha==alpha && call->beta==beta,
            "a call of a divided product changes an argument that the division leaves alone");
      const ptrdiff_t offset=call->c-c;
      int length;
      if (rows) {
        check(0<=offset && offset<m && call->n==n && call->b==b,"a block of rows is not a block of rows of the result");
        check(call->a==(plain_a ? a+offset : a+(size_t)offset*(size_t)lda),
              "a block of rows does not take its rows of the first factor");
        length = call->m;
      }
      else {
        check(0<=offset && offset%ldc==0 && offset/ldc<n && call->m==m && call->a==a,
              "a block of columns is not a block of columns of the result");
        check(call->b==(plain_b ? b+(size_t)(offset/ldc)*(size_t)ldb : b+offset/ldc),
              "a block of columns does not take its columns of the second factor");
        length = call->n;
      }
      check(0<length,"a call of a divided product is empty");
      if (length<shortest) shortest = length;
      if (longest<length) longest = length;
#ifdef _OPENMP
      /* The team has all threads of the build, also for fewer blocks, and
         block i is the work of thread i; unless the runtime may not start
         that many. The blocks are not empty, so the first row or column of
         a block tells its number. */
      if (threads<=omp_get_thread_limit()) {
        const ptrdiff_t first=rows ? offset : offset/ldc;
        int piece=0;
        while (piece<calls && (long long)piece*(rows ? m : n)/calls!=first) piece++;
        check(call->team==threads && piece<calls && call->thread==piece,
              "block i of a divided product is not the work of thread i of a team of all threads");
      }
#else
      check(call->team==1 && call->thread==0,"a program without OpenMP has more than one thread");
#endif
    }
    check(longest-shortest<=1,"the blocks of a divided product differ in length by more than one");
  }

  free(model.written);
  memset(&model,0,sizeof(model));
  free(expected);
  free(c);
  free(b);
  free(a);
  return calls;
}

static void products_are_divided(void)
{
  /* Rows of the result (more rows than columns, and as many), and columns;
     lengths that the pieces do not divide. */
  static const int shape[][3]={{131,90,170}, {90,131,170}, {113,113,157}};
  static const int threads[]={2,3,8};
  static const char letter[]={'N','T','C'};
  static const char small[]={'n','t','c'};
  for (size_t s=0; s<3; s++) {
    for (size_t t=0; t<3; t++) {
      for (size_t x=0; x<3; x++) {
        for (size_t y=0; y<3; y++) {
          const int calls=product(threads[t],letter[x],letter[y],shape[s][0],shape[s][1],shape[s][2],
                                  (int)x,(int)y+1,(int)t,-0.7,x==y ? 0.0 : 0.3);
          check(calls==threads[t],"a product large enough is not divided among all threads");
        }
      }
    }
  }
  for (size_t s=0; s<2; s++)
    for (size_t x=0; x<3; x++)
      for (size_t y=0; y<3; y++)
        check(product(3,small[x],small[y],shape[s][0],shape[s][1],shape[s][2],0,0,0,1.0,1.0)==3,
              "a product with small letters is not divided");

  /* The forms of the exchange build, without rows beyond the matrices. */
  check(product(4,'N','N',240,60,80,0,0,0,1.0,0.0)==4,"the form of G is not divided");
  check(product(4,'N','T',90,160,70,0,0,0,1.0,0.0)==4,"the form of W is not divided");
  check(product(4,'T','N',60,150,240,0,0,0,1.0,0.0)==4,"the form of the last term is not divided");

  /* As many pieces as threads, and no more than the multiplications or the
     length of the result allow. */
  check(product(64,'N','T',400,200,200,0,0,0,1.0,0.0)==64,"a product is not divided among 64 threads");
  check(product(64,'T','N',200,400,199,2,0,1,0.5,-0.25)==63,"the pieces do not follow the multiplications");
  check(product(8,'N','N',3,2,400000,0,0,0,1.0,0.0)==3,"a result of three rows is not divided into its rows");
  check(product(8,'T','T',2,5,400000,1,1,1,1.0,1.0)==5,"a result of five columns is not divided into its columns");

  /* Products that stay one call. */
  check(product(8,'N','N',10,10,10,1,2,3,-0.7,0.3)==1,"a small product is divided");
  check(product(8,'T','N',100,50,99,0,0,0,1.0,0.0)==1,"a product below two pieces is divided");
  check(product(1,'N','N',131,90,170,0,0,0,1.0,0.0)==1,"a product is divided with one thread");
  check(product(0,'N','T',131,90,170,0,0,0,1.0,0.0)==1,"a product is divided without threads");
  check(product(-2,'T','T',131,90,170,0,0,0,1.0,0.0)==1,"a product is divided with a negative number of threads");
  check(product(8,'N','N',0,90,170,0,0,0,1.0,0.3)==1,"a product without rows is divided");
  check(product(8,'N','N',131,0,170,0,0,0,1.0,0.3)==1,"a product without columns is divided");
  check(product(8,'N','N',131,90,0,0,0,2,1.0,0.3)==1,"a product without terms is divided");
  check(product(8,'T','T',131,90,0,0,0,0,1.0,0.0)==1,"a product without terms is divided");
}

static void blas_threads_are_set_and_given_back(void)
{
#if !defined(OPENMX_TEST_WITHOUT_OPENBLAS) && !defined(OPENMX_TEST_OPENBLAS_FOUND)
  /* A static OpenBLAS: the linker brings the two functions of the threads
     into the program and leaves out the one that tells how the library was
     built. The unit cannot know that the library has threads of its own and
     leaves it as it is. The same holds for every OpenBLAS on macOS that the
     build has not named. */
  memset(&blas,0,sizeof(blas));
  blas.parallel = 1;
  blas.threads = 8;
  check(Exact_Exchange_Host_Blas_Single()==0 && blas.sets==0 && blas.threads==8,
        "a BLAS that does not tell how it was built is changed");
  Exact_Exchange_Host_Blas_Restore(0);
  Exact_Exchange_Host_Blas_Restore(7);
  check(blas.sets==0 && blas.threads==8,"a BLAS that does not tell how it was built is given threads");
  check(product(4,'N','T',90,160,70,0,0,0,1.0,0.0)==4 && blas.sets==0,
        "a product changes a BLAS that does not tell how it was built");
#elif !defined(OPENMX_TEST_WITHOUT_OPENBLAS)
  /* A library with threads of its own computes on the calling thread until
     the threads are given back. */
  memset(&blas,0,sizeof(blas));
  blas.parallel = 1;
  blas.threads = 8;
  int kept=Exact_Exchange_Host_Blas_Single();
  check(kept==8 && blas.threads==1 && blas.sets==1 && blas.last_set==1,
        "a BLAS with threads of its own is not set to one thread");
  /* asked again meanwhile, the unit has nothing to keep */
  const int inner=Exact_Exchange_Host_Blas_Single();
  check(inner==0 && blas.sets==1,"a BLAS on one thread is set again");
  Exact_Exchange_Host_Blas_Restore(inner);
  check(blas.threads==1 && blas.sets==1,"nothing to give back changes the threads of the BLAS");
  Exact_Exchange_Host_Blas_Restore(kept);
  check(blas.threads==8 && blas.sets==2 && blas.last_set==8,"the threads of the BLAS are not given back");

  /* A library on one thread, one without threads, and one on the threads of
     OpenMP are left as they are. */
  static const int left[][2]={{1,1}, {0,1}, {0,8}, {2,1}, {2,8}, {3,8}, {-1,8}};
  for (size_t i=0; i<sizeof(left)/sizeof(left[0]); i++) {
    memset(&blas,0,sizeof(blas));
    blas.parallel = left[i][0];
    blas.threads = left[i][1];
    kept = Exact_Exchange_Host_Blas_Single();
    check(kept==0 && blas.sets==0 && blas.threads==left[i][1],"a BLAS that needs no change is changed");
    Exact_Exchange_Host_Blas_Restore(kept);
    Exact_Exchange_Host_Blas_Restore(-4);
    check(blas.sets==0 && blas.threads==left[i][1],"nothing to give back changes the threads of the BLAS");
  }

  /* The products of a build between the two calls. */
  memset(&blas,0,sizeof(blas));
  blas.parallel = 1;
  blas.threads = 4;
  kept = Exact_Exchange_Host_Blas_Single();
  check(product(4,'N','T',90,160,70,0,0,0,1.0,0.0)==4 && blas.threads==1 && blas.sets==1,
        "a product changes the threads of the BLAS");
  Exact_Exchange_Host_Blas_Restore(kept);
  check(blas.threads==4 && blas.sets==2,"the threads of the BLAS are not given back after a product");
#else
  /* Without the functions of OpenBLAS there is nothing to change. The
     references of the unit are null here: a call through one of them would
     end the program. */
  check(Exact_Exchange_Host_Blas_Single()==0,"a BLAS without the functions of OpenBLAS is taken for OpenBLAS");
  Exact_Exchange_Host_Blas_Restore(0);
  Exact_Exchange_Host_Blas_Restore(7);
  check(product(4,'N','T',90,160,70,0,0,0,1.0,0.0)==4,"a product is not divided without the functions of OpenBLAS");
#endif
}


/* ---- products whose arrays an int does not count --------------------------- */

#ifdef MAP_ANONYMOUS
/* Address space for a number of elements, without memory. NULL when the
   system does not give it. */
static double *space(uint64_t elements)
{
  void *memory=mmap(NULL,(size_t)(elements*sizeof(double)),PROT_NONE,MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE,-1,0);
  return memory==MAP_FAILED ? NULL : (double*)memory;
}

/* One product in the mode of the model that only notes the arguments.
   Returns 0 when the system has no address space for the arrays. */
static int large_product(int threads, char transa, char transb, int m, int n, int k, int pad_a, int pad_b, int pad_c)
{
  const int plain_a=plain(transa),plain_b=plain(transb);
  const int rows_a=plain_a ? m : k,columns_a=plain_a ? k : m,rows_b=plain_b ? k : n,columns_b=plain_b ? n : k;
  const int lda=rows_a+pad_a,ldb=rows_b+pad_b,ldc=m+pad_c;
  const uint64_t size_a=(uint64_t)lda*(uint64_t)columns_a,size_b=(uint64_t)ldb*(uint64_t)columns_b;
  const uint64_t size_c=(uint64_t)ldc*(uint64_t)n;
  double *a=space(size_a),*b=space(size_b),*c=space(size_c);
  const int available=a!=NULL && b!=NULL && c!=NULL;
  if (available) {
    const int pieces=Exact_Exchange_Host_Pieces(threads,m,n,k);
    const int rows=n<=m,length=rows ? m : n;
    check(2<=pieces && pieces<=MAX_CALLS,"a case of the large dimensions is not divided, or has too many pieces");
    memset(&model,0,sizeof(model));
    model.notes_only = 1;
    Exact_Exchange_Host_Dgemm(threads,transa,transb,m,n,k,1.0,a,lda,b,ldb,0.0,c,ldc);
    check(model.calls==pieces && model.wrong==0,"a product with large arrays is not divided into valid calls");
    for (int piece=0; piece<pieces; piece++) {
      /* the block of the piece, and the call that has it */
      const int64_t first=(int64_t)piece*length/pieces,count=(int64_t)(piece+1)*length/pieces-first;
      const uint64_t offset_c=rows ? (uint64_t)first : (uint64_t)first*(uint64_t)ldc;
      const Call *call=NULL;
      for (int index=0; index<pieces; index++)
        if ((uint64_t)(model.call[index].c-c)==offset_c) call = &model.call[index];
      check(call!=NULL,"no call of a product with large arrays starts at the block of a piece");
      const uint64_t offset_a=(uint64_t)(call->a-a),offset_b=(uint64_t)(call->b-b);
      check(call->transa==transa && call->transb==transb && call->k==k && call->lda==lda && call->ldb==ldb
            && call->ldc==ldc,"a call of a product with large arrays changes an argument");
      if (rows)
        check(call->m==count && call->n==n && offset_b==0
              && offset_a==(plain_a ? (uint64_t)first : (uint64_t)first*(uint64_t)lda),
              "a block of rows of a product with large arrays has the wrong length or address");
      else
        check(call->m==m && call->n==count && offset_a==0
              && offset_b==(plain_b ? (uint64_t)first*(uint64_t)ldb : (uint64_t)first),
              "a block of columns of a product with large arrays has the wrong length or address");
      /* the last elements that the call may use are inside the arrays */
      const uint64_t call_rows_a=plain_a ? (uint64_t)call->m : (uint64_t)k;
      const uint64_t call_columns_a=plain_a ? (uint64_t)k : (uint64_t)call->m;
      const uint64_t call_rows_b=plain_b ? (uint64_t)k : (uint64_t)call->n;
      const uint64_t call_columns_b=plain_b ? (uint64_t)call->n : (uint64_t)k;
      check(offset_a+(call_columns_a-1)*(uint64_t)lda+call_rows_a<=size_a
            && offset_b+(call_columns_b-1)*(uint64_t)ldb+call_rows_b<=size_b
            && offset_c+((uint64_t)call->n-1)*(uint64_t)ldc+(uint64_t)call->m<=size_c,
            "a call of a product with large arrays reaches beyond an array");
    }
    memset(&model,0,sizeof(model));
  }
  if (a!=NULL) munmap(a,(size_t)(size_a*sizeof(double)));
  if (b!=NULL) munmap(b,(size_t)(size_b*sizeof(double)));
  if (c!=NULL) munmap(c,(size_t)(size_c*sizeof(double)));
  return available;
}
#endif

/* Blocks whose first element lies beyond the range of an int: the bounds
   of the blocks and the addresses of the calls need wider numbers. Returns
   0 when the cases could not run. */
static int bounds_hold_for_large_arrays(void)
{
#ifdef MAP_ANONYMOUS
  if (sizeof(size_t)<8) return 0;
  static const struct { char transa,transb; int m,n,k,pad_a,pad_b,pad_c; } cases[]={
    {'N','N',INT_MAX,1,1,0,0,0},        /* rows of a plain first factor */
    {'T','N',INT_MAX,1,1,0,0,0},        /* rows, the first factor transposed */
    {'N','N',1,INT_MAX,1,0,0,0},        /* columns of a plain second factor */
    {'N','T',1,INT_MAX,1,0,0,0},        /* columns, the second factor transposed */
    {'N','N',46341,46342,3,0,2,2000},   /* columns: the result has more than 2^31 elements */
    {'T','N',70000,3,40000,7,0,1},      /* rows: the transposed first factor has more */
    {'N','N',3,70000,40000,0,7,1},      /* columns: the second factor has more */
    {'T','T',200,400,199,2,3,1}};       /* 63 pieces for 64 threads */
  for (size_t i=0; i<sizeof(cases)/sizeof(cases[0]); i++)
    if (!large_product(64,cases[i].transa,cases[i].transb,cases[i].m,cases[i].n,cases[i].k,cases[i].pad_a,
                       cases[i].pad_b,cases[i].pad_c)) return 0;
  return 1;
#else
  return 0;
#endif
}


/* ---- the loop of the build ------------------------------------------------ */

enum { LOOP_M = 7, LOOP_N = 6, LOOP_K = 11, LOOP_RESULT = 128, LOOP_GUARD = 2 };

/* One loop of the test. The body of an index makes 1 + index%3 products of
   (2 + index%4) x 3 with inner dimension 5 + the number of the product; the
   index "special" makes one of LOOP_M x LOOP_N x LOOP_K more, the largest
   of the loop; the index "nested" runs a loop of its own after its first
   product. Every array has one entry per index, written by the call of
   that index only. */
typedef struct {
  OpenMXExactExchangeHostProducts *products;
  int special,nested,inner;
  int *calls,*thread,*team;
  double (*result)[LOOP_RESULT];
  int inner_calls[8],inner_thread[8],inner_next,inner_disorder,inner_status;
} Loop;

static int current_thread(void)
{
#ifdef _OPENMP
  return omp_get_thread_num();
#else
  return 0;
#endif
}

static int current_team(void)
{
#ifdef _OPENMP
  return omp_get_num_threads();
#else
  return 1;
#endif
}

static int loop_products(const Loop *loop, int index)
{
  return 1+index%3+(index==loop->special ? 1 : 0);
}

static void loop_dimensions(const Loop *loop, int index, int product, int *m, int *n, int *k)
{
  if (index==loop->special && product==loop_products(loop,index)-1) {
    *m = LOOP_M;
    *n = LOOP_N;
    *k = LOOP_K;
  }
  else {
    *m = 2+index%4;
    *n = 3;
    *k = 5+product;
  }
}

/* A number between -1 and 1 that depends on all four arguments. */
static double loop_value(int index, int product, int factor, int position)
{
  const unsigned long long x=((unsigned long long)index*1000003ULL+(unsigned long long)product*10007ULL
                              +(unsigned long long)factor*101ULL+(unsigned long long)position)*2654435761ULL;
  return (double)(x%2001ULL)/1000.0-1.0;
}

/* What the main program does with a product of modules/exx, without the
   device: inside a loop it is the product of a body, outside it is counted
   and goes to the divided product. */
static void build_product(OpenMXExactExchangeHostProducts *products, int m, int n, int k, const double *a,
                          const double *b, double *c)
{
  if (products->in_loop) Exact_Exchange_Host_Loop_Dgemm(products,'N','N',m,n,k,1.0,a,m,b,k,0.0,c,m);
  else {
    Exact_Exchange_Host_Note_Product(products,m,n,k);
    Exact_Exchange_Host_Dgemm(products->threads,'N','N',m,n,k,1.0,a,m,b,k,0.0,c,m);
  }
}

/* A product of a body into c: by the unit, or element by element. */
static void loop_product(const Loop *loop, int index, int product, double *c, int by_the_unit)
{
  int m,n,k;
  double a[LOOP_M*LOOP_K],b[LOOP_K*LOOP_N];
  loop_dimensions(loop,index,product,&m,&n,&k);
  for (int e=0; e<m*k; e++) a[e] = loop_value(index,product,0,e);
  for (int e=0; e<k*n; e++) b[e] = loop_value(index,product,1,e);
  if (by_the_unit) {
    build_product(loop->products,m,n,k,a,b,c);
    return;
  }
  for (int j=0; j<n; j++) {
    for (int i=0; i<m; i++) {
      double sum=0.0;
      for (int l=0; l<k; l++) sum += a[i+l*m]*b[l+j*k];
      c[i+j*m] = sum;
    }
  }
}

static void loop_inner_body(void *argument, int index)
{
  Loop *loop=(Loop*)argument;
  static const double a[3*4]={0.5,-1.0,2.0,1.5,0.25,-0.75,1.0,3.0,-2.0,0.125,4.0,-0.5};
  static const double b[4*3]={1.0,2.0,-1.0,0.5,-0.25,3.0,1.5,-2.0,0.75,-1.5,2.5,1.25};
  double c[3*3];
  loop->inner_calls[index]++;
  loop->inner_thread[index] = current_thread();
  if (index!=loop->inner_next) loop->inner_disorder = 1;
  loop->inner_next++;
  build_product(loop->products,3,3,4,a,b,c);
}

static void loop_body(void *argument, int index)
{
  Loop *loop=(Loop*)argument;
  double *c=loop->result[index];
  loop->calls[index]++;
  loop->thread[index] = current_thread();
  loop->team[index] = current_team();
  for (int product=0; product<loop_products(loop,index); product++) {
    int m,n,k;
    loop_dimensions(loop,index,product,&m,&n,&k);
    loop_product(loop,index,product,c,1);
    c += m*n;
    if (index==loop->nested && product==0)
      loop->inner_status = Exact_Exchange_Host_Parallel_For(loop->products,loop->inner,loop_inner_body,loop);
  }
}

/* A loop of count indices for a build with threads threads. with_array: the
   build has the counters of its threads; without them, and below two
   threads, the unit has no loops and only calls the bodies. large_before:
   a product outside the loops is larger than every product of the loop. */
static void loop_case(int threads, int count, int special, int nested, int with_array, int large_before)
{
  const int entries=(threads<1 ? 1 : threads)+2*LOOP_GUARD;
  OpenMXExactExchangeLoopProducts *array=allocate((size_t)entries,sizeof(*array));
  memset(array,0xA5,(size_t)entries*sizeof(*array));
  OpenMXExactExchangeHostProducts products={0,0.0,{0,0,0},threads,with_array ? array+LOOP_GUARD : NULL,0};
  Loop loop;
  memset(&loop,0,sizeof(loop));
  loop.products = &products;
  loop.special = special;
  loop.nested = nested;
  loop.inner = 3;
  loop.calls = allocate((size_t)count,sizeof(int));
  loop.thread = allocate((size_t)count,sizeof(int));
  loop.team = allocate((size_t)count,sizeof(int));
  loop.result = allocate((size_t)count,sizeof(*loop.result));
  loop.inner_status = -1;

  /* What a count in order gives: the products of the bodies, the largest
     of them, and one product outside the loops before and after. */
  const int before[3]={large_before ? 9 : 4,large_before ? 9 : 4,large_before ? 9 : 5};
  long long expected=1;
  double largest_work=(double)before[0]*(double)before[1]*(double)before[2];
  int largest[3]={before[0],before[1],before[2]};
  long long in_bodies=0;
  for (int index=0; index<count; index++) {
    for (int product=0; product<loop_products(&loop,index); product++) {
      int m,n,k;
      loop_dimensions(&loop,index,product,&m,&n,&k);
      in_bodies++;
      if (largest_work<(double)m*(double)n*(double)k) {
        largest_work = (double)m*(double)n*(double)k;
        largest[0] = m;
        largest[1] = n;
        largest[2] = k;
      }
    }
  }
  const int runs_nested=0<=nested && nested<count;
  if (runs_nested) in_bodies += loop.inner;
  expected += in_bodies;

  memset(&model,0,sizeof(model));
  Exact_Exchange_Host_Note_Product(&products,before[0],before[1],before[2]);
  const int status=Exact_Exchange_Host_Parallel_For(&products,count,loop_body,&loop);
  check(status==0,"the loop of the build reports a failure");
  check(products.in_loop==0,"the loop of the build is still marked as running after it returned");
  check(products.count==expected,"the products of a loop are not counted as a count in order counts them");
  check(products.largest_work==largest_work && memcmp(products.largest,largest,sizeof(largest))==0,
        "the largest product of a build with a loop is not the one of a count in order");
  check(model.calls==in_bodies && model.wrong==0,"a product of a body is not one valid call of DGEMM");

  /* The team: all threads of the build, unless the runtime may not start
     that many; one thread without loops and without OpenMP. */
  const int looped=with_array && 2<=threads;
  int team=1;
#ifdef _OPENMP
  if (looped && 0<count) team = threads<=omp_get_thread_limit() ? threads : loop.team[0];
#endif
  check(1<=team && team<=(threads<1 ? 1 : threads),"the team of a loop is larger than the threads of the build");
  for (int index=0; index<count; index++) {
    check(loop.calls[index]==1,"an index of a loop is not called once");
    check(loop.team[index]==team && loop.thread[index]==(looped ? index%team : 0),
          "an index of a loop does not run on its thread of the team");
    double wanted[LOOP_RESULT];
    double *c=wanted;
    size_t size=0;
    for (int product=0; product<loop_products(&loop,index); product++) {
      int m,n,k;
      loop_dimensions(&loop,index,product,&m,&n,&k);
      loop_product(&loop,index,product,c,0);
      c += m*n;
      size += (size_t)m*(size_t)n;
    }
    check(memcmp(loop.result[index],wanted,size*sizeof(double))==0,
          "a product of a body differs from the product computed element by element");
  }
  for (int index=0; index<model.calls && index<MAX_CALLS; index++)
    check(model.call[index].team==team,"a product of a body is called from another team than its body");

  /* A loop that a body calls: its bodies in order on the thread of that
     body, and no second team. */
  if (runs_nested) {
    check(loop.inner_status==0 && loop.inner_next==loop.inner && !loop.inner_disorder,
          "a loop called from a body does not run its bodies in order");
    for (int index=0; index<loop.inner; index++)
      check(loop.inner_calls[index]==1 && loop.inner_thread[index]==loop.thread[nested],
            "a loop called from a body does not run on the thread of that body");
  }

  /* The counters of the threads hold the products of the bodies; the
     threads beyond the team have none; nothing is written beside them. */
  if (looped) {
    long long counted=0;
    for (int thread=0; thread<threads; thread++) {
      const OpenMXExactExchangeLoopProducts *part=&products.loop[thread];
      check(0<=part->count && (thread<team || part->count==0),"a thread outside the team has products");
      counted += part->count;
    }
    check(counted==in_bodies,"the counters of the threads do not hold the products of the bodies");
  }
  for (int entry=0; entry<entries; entry++) {
    if (looped && LOOP_GUARD<=entry && entry<LOOP_GUARD+threads) continue;
    const unsigned char *bytes=(const unsigned char*)&array[entry];
    for (size_t byte=0; byte<sizeof(*array); byte++)
      check(bytes[byte]==0xA5,"a loop writes beside the counters of its threads");
  }

  /* A product outside the loops afterwards is counted with the build. */
  Exact_Exchange_Host_Note_Product(&products,2,2,2);
  check(products.count==expected+1 && products.largest_work==largest_work
        && memcmp(products.largest,largest,sizeof(largest))==0,"a product after a loop is not counted with the build");

  memset(&model,0,sizeof(model));
  free(loop.result);
  free(loop.team);
  free(loop.thread);
  free(loop.calls);
  free(array);
}

/* Two bodies with products of the same number of multiplications. */
static void tie_body(void *argument, int index)
{
  OpenMXExactExchangeHostProducts *products=(OpenMXExactExchangeHostProducts*)argument;
  double a[6*5],b[5*6],c[6*6];
  for (int e=0; e<30; e++) {
    a[e] = loop_value(index,0,0,e);
    b[e] = loop_value(index,0,1,e);
  }
  if (index==0) build_product(products,4,6,5,a,b,c);
  else          build_product(products,6,4,5,a,b,c);
}

static void loops_count_their_products(void)
{
  static const int threads[]={2,3,4,8};
  for (size_t t=0; t<sizeof(threads)/sizeof(threads[0]); t++) {
    const int team=threads[t];
    const int counts[]={0,1,team-1,team,team+1,4*team+3};
    for (size_t c=0; c<sizeof(counts)/sizeof(counts[0]); c++) loop_case(team,counts[c],counts[c]/2,-1,1,0);
    loop_case(team,2*team+1,-1,-1,1,0);    /* the largest product is one that several bodies make */
    loop_case(team,4*team+3,team,-1,1,1);  /* the largest product of the build is outside the loop */
    loop_case(team,team,0,team-1,1,0);     /* a loop called from a body, on the last thread */
    loop_case(team,team+2,1,1,1,0);        /* and on a thread with two indices */
  }
  /* No loops: below two threads, or without the counters of the threads. */
  loop_case(1,5,2,-1,1,0);
  loop_case(0,5,2,-1,1,0);
  loop_case(-3,5,2,3,1,0);
  loop_case(4,5,2,-1,0,0);
  loop_case(4,5,2,3,0,1);

  /* Of two products with the same number of multiplications the one of the
     lower thread stays the largest; in one thread, the first. */
  OpenMXExactExchangeLoopProducts array[4];
  OpenMXExactExchangeHostProducts products={0,0.0,{0,0,0},4,array,0};
  memset(&model,0,sizeof(model));
  check(Exact_Exchange_Host_Parallel_For(&products,2,tie_body,&products)==0 && products.count==2
        && products.largest_work==120.0 && products.largest[0]==4 && products.largest[1]==6
        && products.largest[2]==5,"of two products of the same size the one of the lower thread is not kept");
  Exact_Exchange_Host_Note_Product(&products,5,6,4);
  check(products.count==3 && products.largest[0]==4,"a later product of the same size replaces the largest");
  Exact_Exchange_Host_Note_Product(&products,5,5,5);
  check(products.count==4 && products.largest_work==125.0 && products.largest[0]==5 && products.largest[1]==5
        && products.largest[2]==5,"a larger product does not become the largest");
  memset(&model,0,sizeof(model));
}

int main(void)
{
#ifdef _OPENMP
  omp_set_dynamic(0);
#endif
  pieces_follow_the_rule();
  products_are_divided();
  const int large=bounds_hold_for_large_arrays();
  blas_threads_are_set_and_given_back();
  loops_count_their_products();
  printf("exx host product: ok (%s, %s%s)\n",
#ifdef _OPENMP
         "OpenMP",
#else
         "no OpenMP",
#endif
#if defined(OPENMX_TEST_WITHOUT_OPENBLAS)
         "no functions of OpenBLAS",
#elif defined(OPENMX_TEST_WITHOUT_OPENBLAS_PARALLEL)
         "OpenBLAS without openblas_get_parallel",
#elif defined(OPENMX_EXX_HOST_BLAS_OPENBLAS)
         "OpenBLAS named by the build",
#elif !defined(OPENMX_TEST_OPENBLAS_FOUND)
         "functions of OpenBLAS that the unit does not look for",
#else
         "functions of OpenBLAS",
#endif
         large ? "" : "; the arrays beyond the range of int were not checked: no address space");
  return 0;
}

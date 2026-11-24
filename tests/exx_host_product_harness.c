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

   The test asks four things of the unit: the number of calls is the one of
   the rule in openmx_exact_exchange_host.h, a divided product equals the
   product computed element by element and writes every element once and
   nothing else, its calls take blocks of nearly equal length of the longer
   side of the result, each on a thread of its own when the program has
   OpenMP, and only a BLAS with threads of its own has its number of threads
   changed and given back.

   Compiled with OPENMX_TEST_WITHOUT_OPENBLAS, the program has no functions
   of OpenBLAS, as one linked with another BLAS. */

#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "f77func.h"
#include "openmx_exact_exchange_host.h"

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
    if (!valid) model.wrong++;
  }
  if (!valid) return 0;
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
      if (model.c!=NULL && model.c<=element && element<model.c+(size_t)model.ldc*(size_t)model.n) {
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

/* The functions of OpenBLAS. */
typedef struct {
  int parallel;              /* answer of openblas_get_parallel */
  int threads;
  int sets;                  /* calls of openblas_set_num_threads */
  int last_set;
} Blas;

static Blas blas;

#ifndef OPENMX_TEST_WITHOUT_OPENBLAS
int openblas_get_parallel(void);
int openblas_get_num_threads(void);
void openblas_set_num_threads(int threads);

int openblas_get_parallel(void)
{
  return blas.parallel;
}

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
   side of the result, and at least 2.5e5 multiplications in every call. */
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
    unsigned char *thread_seen=allocate((size_t)calls,1);
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
      /* Every call runs on a thread of its own, unless the runtime may not
         start that many. */
      if (calls<=omp_get_thread_limit()) {
        check(call->team==calls && 0<=call->thread && call->thread<calls && !thread_seen[call->thread],
              "the calls of a divided product do not run on a thread each");
        thread_seen[call->thread] = 1;
      }
#else
      check(call->team==1 && call->thread==0,"a program without OpenMP has more than one thread");
#endif
    }
    check(longest-shortest<=1,"the blocks of a divided product differ in length by more than one");
    free(thread_seen);
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
#ifndef OPENMX_TEST_WITHOUT_OPENBLAS
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
  /* Without the functions of OpenBLAS there is nothing to change. */
  check(Exact_Exchange_Host_Blas_Single()==0,"a BLAS without the functions of OpenBLAS is taken for OpenBLAS");
  Exact_Exchange_Host_Blas_Restore(0);
  Exact_Exchange_Host_Blas_Restore(7);
  check(blas.sets==0,"a BLAS without the functions of OpenBLAS is changed");
  check(product(4,'N','T',90,160,70,0,0,0,1.0,0.0)==4,"a product is not divided without the functions of OpenBLAS");
#endif
}

int main(void)
{
#ifdef _OPENMP
  omp_set_dynamic(0);
#endif
  pieces_follow_the_rule();
  products_are_divided();
  blas_threads_are_set_and_given_back();
  printf("exx host product: ok (%s, %s)\n",
#ifdef _OPENMP
         "OpenMP",
#else
         "no OpenMP",
#endif
#ifndef OPENMX_TEST_WITHOUT_OPENBLAS
         "functions of OpenBLAS"
#else
         "no functions of OpenBLAS"
#endif
         );
  return 0;
}

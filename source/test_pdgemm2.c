#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "mpi.h"

int main(void) {
  int i, j, k;
  /************  MPI ***************************/
  int myrank_mpi, nprocs_mpi;
  MPI_Init( NULL, NULL);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank_mpi);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs_mpi);
  /************  BLACS ***************************/
  int ictxt, nprow, npcol, myrow, mycol,nb;
  int info,itemp,bhandle;
  int _ZERO=0,_ONE=1;
  int M=2000;
  int K=2000;
  int N=2000;
  nprow = 2; npcol = 2; 
  nb=200;

  //Cblacs_pinfo( &myrank_mpi, &nprocs_mpi ) ;
  //Cblacs_get( -1, 0, &ictxt );
  //Cblacs_gridinit( &ictxt, "Row", nprow, npcol );
  //Cblacs_gridinfo( ictxt, &nprow, &npcol, &myrow, &mycol );
  printf("myrank = %d\n",myrank_mpi);

  bhandle = Csys2blacs_handle(MPI_COMM_WORLD);
  ictxt = bhandle;
  Cblacs_gridinit( &ictxt, "Row", nprow, npcol );

  Cfree_blacs_system_handle(bhandle);
  Cblacs_gridexit( ictxt );
  MPI_Finalize();
  return 0;


  int rA = numroc_( &M, &nb, &myrow, &_ZERO, &nprow );
  int cA = numroc_( &K, &nb, &mycol, &_ZERO, &npcol );
  int rB = numroc_( &K, &nb, &myrow, &_ZERO, &nprow );
  int cB = numroc_( &N, &nb, &mycol, &_ZERO, &npcol );
  int rC = numroc_( &M, &nb, &myrow, &_ZERO, &nprow );
  int cC = numroc_( &N, &nb, &mycol, &_ZERO, &npcol );

  double *A = (double*) malloc(rA*cA*sizeof(double));
  double *B = (double*) malloc(rB*cB*sizeof(double));
  double *C = (double*) malloc(rC*cC*sizeof(double));

  int descA[9],descB[9],descC[9];

  descinit_(descA, &M,   &K,   &nb,  &nb,  &_ZERO, &_ZERO, &ictxt, &rA,  &info);
  descinit_(descB, &K,   &N,   &nb,  &nb,  &_ZERO, &_ZERO, &ictxt, &rB,  &info);
  descinit_(descC, &M,   &N,   &nb,  &nb,  &_ZERO, &_ZERO, &ictxt, &rC,  &info);

  double alpha = 1.0; double beta = 1.0;   
  double start, end, flops;
  srand(time(NULL)*myrow+mycol);
     #pragma simd
  for (j=0; j<rA*cA; j++)
    {
      A[j]=((double)rand()-(double)(RAND_MAX)*0.5)/(double)(RAND_MAX);
      //   printf("A in myrank: %d\n",myrank_mpi);
    }
  //   printf("A: %d\n",myrank_mpi);
     #pragma simd
  for (j=0; j<rB*cB; j++)
    {
      B[j]=((double)rand()-(double)(RAND_MAX)*0.5)/(double)(RAND_MAX);
    }
     #pragma simd
  for (j=0; j<rC*cC; j++)
    {
      C[j]=((double)rand()-(double)(RAND_MAX)*0.5)/(double)(RAND_MAX);
    }
  MPI_Barrier(MPI_COMM_WORLD);

  start=MPI_Wtime();



  pdgemm_ ("N", "N", &M , &N , &K , &alpha, A , &_ONE, &_ONE , descA , B , &_ONE, &_ONE , descB , &beta , C , &_ONE, &_ONE , descC );
  MPI_Barrier(MPI_COMM_WORLD);
  end=MPI_Wtime();

  if (myrow==0 && mycol==0)
    {
      flops = 2 * (double) M * (double) N * (double) K / (end-start) / 1e9;
      /*   printf("This is value: %d\t%d\t%d\t%d\t%d\t%d\t\n",rA,cA,rB,cB,rC,cC);
	   printf("%f\t%f\t%f\n", A[4], B[6], C[3]);*/
      printf("%f Gflops\n", flops);
    }
  Cblacs_gridexit( ictxt );


  MPI_Finalize();
  free(A);
  free(B);
  free(C);
  return 0;
}

#include <mpi.h>
#include <omp.h>
#include <mkl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
int main(int argc,char **argv) {
 int rank,provided,team=0;MPI_Init_thread(&argc,&argv,MPI_THREAD_SERIALIZED,&provided);MPI_Comm_rank(MPI_COMM_WORLD,&rank);
 #pragma omp parallel
 {
  #pragma omp single
  team=omp_get_num_threads();
 }
 const MKL_INT n=1024;size_t count=(size_t)n*n;
 MKL_Complex16 *a=malloc(count*sizeof(*a)),*b=malloc(count*sizeof(*b)),*c=malloc(count*sizeof(*c));
 if(!a||!b||!c)MPI_Abort(MPI_COMM_WORLD,1);
 for(size_t i=0;i<count;i++){a[i].real=b[i].real=1.;a[i].imag=b[i].imag=0.;}
 MKL_Complex16 alpha={1.,0.},beta={0.,0.};
 cblas_zgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,n,n,n,&alpha,a,n,b,n,&beta,c,n);
 int bad=0;for(size_t i=0;i<count;i++)if(fabs(c[i].real-n)>1e-10||fabs(c[i].imag)>1e-10)bad=1;
 printf("PROBE {\"rank\":%d,\"omp_team\":%d,\"omp_limit\":%d,\"mkl_limit\":%d,\"mpi_thread\":%d,\"bad\":%d}\n",rank,team,omp_get_max_threads(),mkl_get_max_threads(),provided,bad);
 free(c);free(b);free(a);MPI_Finalize();return bad;
}

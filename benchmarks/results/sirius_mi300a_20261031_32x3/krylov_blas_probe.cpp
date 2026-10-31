#include <hip/hip_runtime_api.h>
#include <hipblas/hipblas.h>
#include <rocblas/rocblas.h>
#include <mpi.h>
#include <omp.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define HIP(call) do { auto s=(call); if(s!=hipSuccess) { fprintf(stderr,"probe HIP %s=%d\n",#call,int(s)); MPI_Abort(MPI_COMM_WORLD,2); } } while(0)
#define BLAS(call) do { auto s=(call); if(s!=HIPBLAS_STATUS_SUCCESS) { fprintf(stderr,"probe BLAS %s=%d\n",#call,int(s)); MPI_Abort(MPI_COMM_WORLD,3); } } while(0)
int main(int argc,char **argv) {
 int provided=0,rank=0,local=0;
 MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);
 MPI_Comm_rank(MPI_COMM_WORLD,&rank);
 local=atoi(getenv("OMPI_COMM_WORLD_LOCAL_RANK"));
 const int device=local/8;
 HIP(hipSetDevice(device));
 if(argc>1) {rocblas_initialize(); fprintf(stderr,"PROBE_INIT rank=%d pending=%d\n",rank,int(hipGetLastError()));}
 int failures=0;
 #pragma omp parallel num_threads(3) reduction(+:failures)
 {
  const int tid=omp_get_thread_num();
  HIP(hipSetDevice(device));
  hipStream_t stream{};hipblasHandle_t blas{};
  HIP(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking));
  BLAS(hipblasCreate(&blas));BLAS(hipblasSetStream(blas,stream));
  double *a=nullptr,*b=nullptr,*c=nullptr;
  HIP(hipMalloc((void**)&a,sizeof(double)*492*492));
  HIP(hipMalloc((void**)&b,sizeof(double)*492*320));
  HIP(hipMalloc((void**)&c,sizeof(double)*492*320));
  std::vector<double> ha(492*492,1.0),hb(492*320,1.0),hc(492*320);
  HIP(hipMemcpyAsync(a,ha.data(),ha.size()*sizeof(double),hipMemcpyHostToDevice,stream));
  HIP(hipMemcpyAsync(b,hb.data(),hb.size()*sizeof(double),hipMemcpyHostToDevice,stream));
  HIP(hipStreamSynchronize(stream));
  const auto init_error=hipGetLastError();
  if(init_error!=hipSuccess)fprintf(stderr,"PROBE_THREAD_INIT rank=%d thread=%d pending=%d\n",rank,tid,int(init_error));
  const double one=1,zero=0;
  for(int op=0;op<3;op++) {
   hipblasStatus_t status;
   if(op==0)status=hipblasDgemm(blas,HIPBLAS_OP_N,HIPBLAS_OP_N,492,320,492,&one,a,492,b,492,&zero,c,492);
   else if(op==1)status=hipblasDgemm(blas,HIPBLAS_OP_T,HIPBLAS_OP_N,320,320,492,&one,b,492,c,492,&zero,a,320);
   else status=hipblasDgemm(blas,HIPBLAS_OP_T,HIPBLAS_OP_T,320,492,320,&one,a,320,b,492,&zero,c,320);
   const auto pending=hipPeekAtLastError();
   const auto sync=hipStreamSynchronize(stream);
   const auto clear=hipGetLastError();
   const int count=op==1?320*320:492*320;
   const auto copy=hipMemcpy(hc.data(),op==1?a:c,sizeof(double)*count,hipMemcpyDeviceToHost);
   const double expected=op==0?492.0:op==1?492.0*492:492.0*492*320;
   double error=0;
   for(int i=0;i<count;i++) error=std::max(error,std::abs(hc[i]-expected));
   if(status || sync || copy || !std::isfinite(error) || error!=0)failures++;
   fprintf(stderr,"PROBE rank=%d thread=%d device=%d op=%d blas=%d sync=%d copy=%d pending=%d,%d error=%.17g\n",rank,tid,device,op,int(status),int(sync),int(copy),int(pending),int(clear),error);
  }
  HIP(hipFree(a));HIP(hipFree(b));HIP(hipFree(c));
  BLAS(hipblasDestroy(blas));HIP(hipStreamDestroy(stream));
 }
 int all=0;MPI_Allreduce(&failures,&all,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
 if(rank==0)fprintf(stderr,"PROBE_NUMERICAL_FAILURES=%d\n",all);
 MPI_Finalize();return all?1:0;
}

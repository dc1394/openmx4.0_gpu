#include <cusolverDn.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cmath>
#include <cstdio>
#include <vector>
int main(int argc,char**) {
    cusolverDnHandle_t h=nullptr;
    if(cusolverDnCreate(&h)!=CUSOLVER_STATUS_SUCCESS)return 2;
    if(argc>1 && cusolverDnSetMathMode(h,CUSOLVER_FP64_EMULATED_FIXEDPOINT_MATH)!=CUSOLVER_STATUS_SUCCESS)return 3;
    // Exercise both forwarded APIs on the diagonal Hermitian matrix diag(2,3,4).
    for(int api=0;api<2;++api) {
        cuDoubleComplex matrix[9]{};
        for(int i=0;i<3;++i)matrix[4*i]=make_cuDoubleComplex(2+i,0);
        void *A=nullptr,*W=nullptr,*device=nullptr;int *info=nullptr;
        if(cudaMalloc(&A,sizeof(matrix))!=cudaSuccess || cudaMalloc(&W,3*sizeof(double))!=cudaSuccess ||
           cudaMalloc((void**)&info,sizeof(int))!=cudaSuccess)return 4;
        if(cudaMemcpy(A,matrix,sizeof(matrix),cudaMemcpyHostToDevice)!=cudaSuccess)return 5;
        size_t db=0,hb=0;int64_t count=0;double lo=0,hi=0;cusolverStatus_t status;
        if(api==0)status=cusolverDnXsyevdx_bufferSize(h,nullptr,CUSOLVER_EIG_MODE_VECTOR,CUSOLVER_EIG_RANGE_ALL,
            CUBLAS_FILL_MODE_LOWER,3,CUDA_C_64F,A,3,&lo,&hi,1,3,&count,CUDA_R_64F,W,CUDA_C_64F,&db,&hb);
        else status=cusolverDnXsyevd_bufferSize(h,nullptr,CUSOLVER_EIG_MODE_VECTOR,CUBLAS_FILL_MODE_LOWER,
            3,CUDA_C_64F,A,3,CUDA_R_64F,W,CUDA_C_64F,&db,&hb);
        if(status!=CUSOLVER_STATUS_SUCCESS)return 6;
        std::vector<unsigned char> host(hb);
        if(db && cudaMalloc(&device,db)!=cudaSuccess)return 7;
        if(api==0)status=cusolverDnXsyevdx(h,nullptr,CUSOLVER_EIG_MODE_VECTOR,CUSOLVER_EIG_RANGE_ALL,
            CUBLAS_FILL_MODE_LOWER,3,CUDA_C_64F,A,3,&lo,&hi,1,3,&count,CUDA_R_64F,W,CUDA_C_64F,
            device,db,host.data(),hb,info);
        else status=cusolverDnXsyevd(h,nullptr,CUSOLVER_EIG_MODE_VECTOR,CUBLAS_FILL_MODE_LOWER,
            3,CUDA_C_64F,A,3,CUDA_R_64F,W,CUDA_C_64F,device,db,host.data(),hb,info);
        if(status!=CUSOLVER_STATUS_SUCCESS)return 8;
        int value=-1;double eigen[3]{};
        if(cudaMemcpy(&value,info,sizeof(value),cudaMemcpyDeviceToHost)!=cudaSuccess ||
           cudaMemcpy(eigen,W,sizeof(eigen),cudaMemcpyDeviceToHost)!=cudaSuccess)return 9;
        if(value || (api==0 && count!=3))return 10;
        for(int i=0;i<3;++i)if(std::abs(eigen[i]-(2+i))>1e-12)return 11;
        std::printf("api=%s info=%d eigenvalues=%.15g,%.15g,%.15g\n",api==0?"Xsyevdx":"Xsyevd",value,eigen[0],eigen[1],eigen[2]);
        cudaFree(device);cudaFree(info);cudaFree(W);cudaFree(A);
    }
    return cusolverDnDestroy(h)==CUSOLVER_STATUS_SUCCESS?0:12;
}

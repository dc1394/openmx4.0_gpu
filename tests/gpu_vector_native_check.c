/* Native worker accuracy: independent long-double references, no MPI startup. */
#include "gpu_vector_native_probe.h"
#include "openmx_gpu_runtime.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <openacc.h>

static int outstanding;
cudaError_t __real_cudaMalloc(void**,size_t);
cudaError_t __real_cudaFree(void*);
cudaError_t __wrap_cudaMalloc(void **out,size_t bytes){
    cudaError_t status=__real_cudaMalloc(out,bytes);
    if(status==cudaSuccess && *out)++outstanding;
    return status;
}
cudaError_t __wrap_cudaFree(void *pointer){
    cudaError_t status=__real_cudaFree(pointer);
    if(status==cudaSuccess && pointer)--outstanding;
    return status;
}
/* The probe calls the actual worker phases directly on one GPU. */
int openmx_gpu_workers_capacity(OpenMXGpuWorkers *p,size_t n,int kind,OpenMXGpuWorkerCost f,void *context){
    (void)p;(void)n;(void)kind;(void)f;(void)context;abort();
}
int openmx_gpu_workers_run_prepared(const OpenMXGpuWorkers *p,int(*f)(void*,int,int),void *context,const char *name){
    (void)p;(void)f;(void)context;(void)name;abort();
}
int main(void){
    assert(cudaSetDevice(0)==cudaSuccess);acc_set_device_num(0,acc_device_nvidia);
    const size_t sizes[]={0,1,33,257,8195,65539};
    int comparisons=0;long double worst=0;
    for(int columns=1;columns<=8;columns*=2)for(int shape=0;shape<6;++shape)for(int mode=0;mode<2;++mode){
        const size_t n=sizes[shape],ld=n+7;const int pairs=columns*(columns+1)/2;
        const size_t outputs=mode?n:(size_t)pairs;
        double *data=malloc((size_t)columns*ld*sizeof(double)),*weight=malloc((n?n:1)*sizeof(double));
        double coefficients[8];int left[36],right[36],slot[8];
        for(int c=0;c<columns;++c){
            slot[c]=(3*c+1)%columns;coefficients[c]=.17*(c-3);
            for(size_t i=0;i<ld;++i)data[(size_t)c*ld+i]=sin((i%101)*.071+c*.19)*(i%2?1:-1);
        }
        for(size_t i=0;i<n;++i)weight[i]=(i%5?1.0:-.3)+(i%17)*.01;
        int p=0;for(int a=0;a<columns;++a)for(int b=0;b<=a;++b){left[p]=slot[a];right[p++]=slot[b];}
        OpenMXGpuVector job={.n=n,.ld=ld,.mode=mode,.columns=columns,.pairs=pairs,
            .data=data,.weight=weight,.coeff=coefficients,.left=left,.right=right};
        void *work=openmx_test_vector_create(&job);assert(work);
        assert(openmx_test_vector_phase(work,OPENMX_GPU_WORKER_PREPARE));
        assert(openmx_test_vector_phase(work,OPENMX_GPU_WORKER_EXECUTE));
        assert(openmx_test_vector_phase(work,OPENMX_GPU_WORKER_COMMIT));
        const double *actual=openmx_test_vector_result(work);
        for(size_t out=0;out<outputs;++out){
            long double expected=0;
            if(mode)for(int c=0;c<columns;++c)expected+=(long double)coefficients[c]*data[(size_t)c*ld+out];
            else for(size_t i=0;i<n;++i)expected+=(long double)weight[i]*data[(size_t)left[out]*ld+i]*data[(size_t)right[out]*ld+i];
            long double relative=fabsl(actual[out]-expected)/fmaxl(1,fabsl(expected));
            if(!isfinite(actual[out]) || relative>5e-12L){fprintf(stderr,"mode=%d n=%zu c=%d out=%zu error=%.6Lg\n",mode,n,columns,out,relative);return 1;}
            if(relative>worst)worst=relative;++comparisons;
        }
        assert(openmx_test_vector_phase(work,OPENMX_GPU_WORKER_RELEASE));
        openmx_test_vector_destroy(work);assert(outstanding==0);free(weight);free(data);
    }
    acc_shutdown(acc_device_nvidia);
    printf("PASS: 48 native vector configurations, %d long-double comparisons, empty/tail/padded/permuted/cancellation cases; max relative error %.6Lg; owned allocations balanced\n",comparisons,worst);
}

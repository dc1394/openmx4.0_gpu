/* Exercise the retained single-GPU fixture and explicit forward precision
   on cooperative tiles, including repeated buffer addresses/new contents. */
#define main retained_forward_main
#include "gemmul8_forward_smoke.cu"
#undef main
#include "openmx_mgpu_gemm.h"
static int enabled=0,ngpu=4,products=0;
extern "C" int openmx_mgpu_dispatch_dgemm(cublasHandle_t h,cublasOperation_t ta,cublasOperation_t tb,
    int m,int n,int k,const double *alpha,const double *a,int lda,const double *b,int ldb,
    const double *beta,double *c,int ldc,cublasStatus_t *status){
    if(!enabled || openmx_mgpuGemmInWorker())return 0;
    const int ids[4]={0,1,2,3};
    int used=openmx_mgpuDgemmTry(h,ngpu,ids,ta,tb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc,status);
    if(used && *status==CUBLAS_STATUS_SUCCESS){openmx_mgpu_gemm_report r{};openmx_mgpuGemmLastReport(&r);if(r.active_gpus!=ngpu)std::exit(3);products++;}
    return used;
}
extern "C" int openmx_mgpu_dispatch_zgemm(cublasHandle_t h,cublasOperation_t ta,cublasOperation_t tb,
    int m,int n,int k,const cuDoubleComplex *alpha,const cuDoubleComplex *a,int lda,const cuDoubleComplex *b,int ldb,
    const cuDoubleComplex *beta,cuDoubleComplex *c,int ldc,cublasStatus_t *status){
    if(!enabled || openmx_mgpuGemmInWorker())return 0;
    const int ids[4]={0,1,2,3};
    int used=openmx_mgpuZgemmTry(h,ngpu,ids,ta,tb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc,status);
    if(used && *status==CUBLAS_STATUS_SUCCESS){openmx_mgpu_gemm_report r{};openmx_mgpuGemmLastReport(&r);if(r.active_gpus!=ngpu)std::exit(3);products++;}
    return used;
}
template<class T> void cooperative(cublasHandle_t h,const Shape &s,const char *name){
    Bench<T> b(h,s);
    for(int stage:{FP64,GEMMUL8})for(int unblocked:{0,1})for(int reuse:{0,1})for(int change=0;change<2;change++){
        auto x=fill<T>(size_t(s.x_rows())*s.x_cols(),.31+change);
        auto v=fill<T>(size_t(s.v_rows())*s.v_cols(),.73+2*change);
        b.upload(x,v);openmx_gemmul8SetForwardStage(stage,15,0,reuse,unblocked);
        const int before=products;auto c=b.fixed(0,1);
        expect(products==before+1,"forward product did not distribute",name,s.name);
        expect(error(s,x,v,c)<2e-11,"cooperative precision/stale tile",name,s.name);
    }
}
int main(){
    int count=0;check(cudaGetDeviceCount(&count));if(count<4)return 77;
    // The original cache/version tests must still pass when cooperation is off.
    if(retained_forward_main())return 1;
    enabled=1;
    setenv("OPENMX_MGPU_GEMM_WORKSPACE_MB","1",1);
    setenv("OPENMX_MGPU_GEMM_HOST_MB","1",1);
    cublasHandle_t h;check(cublasCreate(&h));
    for(int g:{2,4}){
        ngpu=g;
        for(bool left:{false,true})for(auto op:{CUBLAS_OP_N,CUBLAS_OP_T,CUBLAS_OP_C}){
            Shape s{"tiled",left,op,137,83,113};
            cooperative<double>(h,s,"D");cooperative<cuDoubleComplex>(h,s,"Z");
        }
    }
    check(cublasDestroy(h));openmx_mgpuGemmRelease();openmx_gemmul8ReleasePrepared();openmx_gemmul8ReleaseWorkspaces();
    std::printf("%s: %d checks, %d cooperative forward products\n",failures?"FAIL":"PASS",checks,products);
    return failures?1:0;
}

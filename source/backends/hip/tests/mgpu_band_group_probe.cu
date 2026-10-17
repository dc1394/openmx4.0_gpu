/* Physical-device Band group dispatch, including an origin outside its group. */
#include "openmx_mgpu_runtime.h"
#include "openmx_mgpu_gemm.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
extern "C" {int level_stdout=0;}
#define HIP_OK(x) do {hipError_t e=(x);if(e!=hipSuccess){std::fprintf(stderr,"%s: %s\n",#x,hipGetErrorString(e));return 1;}}while(0)
#define REQUIRE(x) do {if(!(x)){std::fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#x);return 1;}}while(0)
int main(int argc,char **argv)
{
    int gpus=4,kpoints=2,owner=0;
    for(int i=1;i<argc;++i){
        if(!std::strcmp(argv[i],"--force-host")){setenv("OPENMX_MGPU_GEMM_FORCE_HOST","1",1);continue;}
        if(i+1==argc){std::fprintf(stderr,"Missing option value\n");return 2;}
        const char *key=argv[i];const int value=std::atoi(argv[++i]);
        if(!std::strcmp(key,"--gpus"))gpus=value;
        else if(!std::strcmp(key,"--kpoints"))kpoints=value;
        else if(!std::strcmp(key,"--owner"))owner=value;
        else return 2;
    }
    REQUIRE(gpus>=2 && gpus<=8 && kpoints>0);
    int visible=0;HIP_OK(hipGetDeviceCount(&visible));REQUIRE(visible>=gpus);
    setenv("OPENMX_MGPU","1",1);setenv("OPENMX_MGPU_GEMM","1",1);
    setenv("OPENMX_MGPU_VERBOSE","1",1);unsetenv("OPENMX_MGPU_NUM_GPUS");
    openmx_mgpu_configure(1,gpus,1);
    const int width=gpus/kpoints,used=openmx_mgpu_band_used_gpus(kpoints);
    REQUIRE(used==(width>=2?width*kpoints:0));
    // Input setup removes the remainder from rank binding before allocation.
    if(used)openmx_mgpu_configure(1,used,1);
    REQUIRE(owner>=0 && owner<(used?used:gpus));
    HIP_OK(hipSetDevice(owner));
    constexpr int n=192;
    std::vector<double> a(n*n,1),b(n*n,2),out(n*n,-7);
    std::vector<hipDoubleComplex> za(n*n,make_hipDoubleComplex(1,1));
    std::vector<hipDoubleComplex> zb(n*n,make_hipDoubleComplex(2,-1)),zout(n*n,make_hipDoubleComplex(-7,3));
    double *da=nullptr,*db=nullptr,*dc=nullptr;
    hipDoubleComplex *dza=nullptr,*dzb=nullptr,*dzc=nullptr;
    HIP_OK(hipMalloc(&da,n*n*sizeof(double)));HIP_OK(hipMalloc(&db,n*n*sizeof(double)));
    HIP_OK(hipMalloc(&dc,n*n*sizeof(double)));HIP_OK(hipMalloc(&dza,n*n*sizeof(hipDoubleComplex)));
    HIP_OK(hipMalloc(&dzb,n*n*sizeof(hipDoubleComplex)));HIP_OK(hipMalloc(&dzc,n*n*sizeof(hipDoubleComplex)));
    HIP_OK(hipMemcpy(da,a.data(),n*n*sizeof(double),hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(db,b.data(),n*n*sizeof(double),hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(dc,out.data(),n*n*sizeof(double),hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(dza,za.data(),n*n*sizeof(hipDoubleComplex),hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(dzb,zb.data(),n*n*sizeof(hipDoubleComplex),hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(dzc,zout.data(),n*n*sizeof(hipDoubleComplex),hipMemcpyHostToDevice));
    hipStream_t stream;HIP_OK(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking));
    hipblasHandle_t handle;REQUIRE(hipblasCreate(&handle)==HIPBLAS_STATUS_SUCCESS);
    REQUIRE(hipblasSetStream(handle,stream)==HIPBLAS_STATUS_SUCCESS);
    const double one=1,zero=0;const hipDoubleComplex zone=make_hipDoubleComplex(1,0),zzero=make_hipDoubleComplex(0,0);
    hipblasStatus_t status=HIPBLAS_STATUS_SUCCESS;
    REQUIRE(openmx_mgpu_dispatch_dgemm(handle,HIPBLAS_OP_N,HIPBLAS_OP_N,n,n,n,&one,da,n,db,n,&zero,dc,n,&status)==0);
    for(int repeat=0;repeat<2;++repeat)for(int point=0;point<kpoints;++point){
        auto previous=openmx_mgpu_band_scope_enter(1,kpoints,point);
        int devices[8];const int count=openmx_mgpu_band_group(kpoints,point,devices,8);
        REQUIRE(count==(used?width:0));
        for(int j=0;j<count;++j)REQUIRE(devices[j]==point*width+j && devices[j]<used);
        const int real=openmx_mgpu_dispatch_dgemm(handle,HIPBLAS_OP_N,HIPBLAS_OP_N,n,n,n,&one,da,n,db,n,&zero,dc,n,&status);
        REQUIRE(real==(used?1:0));if(real)REQUIRE(status==HIPBLAS_STATUS_SUCCESS);
        openmx_mgpu_gemm_report report{};
        if(real){openmx_mgpuGemmLastReport(&report);REQUIRE(report.active_gpus==width && !std::strcmp(report.reason,"success"));}
        const int complex=openmx_mgpu_dispatch_zgemm(handle,HIPBLAS_OP_N,HIPBLAS_OP_N,n,n,n,&zone,dza,n,dzb,n,&zzero,dzc,n,&status);
        REQUIRE(complex==(used?1:0));if(complex)REQUIRE(status==HIPBLAS_STATUS_SUCCESS);
        if(complex){openmx_mgpuGemmLastReport(&report);REQUIRE(report.active_gpus==width && !std::strcmp(report.reason,"success"));}
        int current=-1;HIP_OK(hipGetDevice(&current));REQUIRE(current==owner);
        HIP_OK(hipMemcpy(out.data(),dc,n*n*sizeof(double),hipMemcpyDeviceToHost));
        HIP_OK(hipMemcpy(zout.data(),dzc,n*n*sizeof(hipDoubleComplex),hipMemcpyDeviceToHost));
        for(auto v:out)REQUIRE(v==(used?2*n:-7));
        for(auto v:zout)REQUIRE(v.x==(used?3*n:-7) && v.y==(used?n:3));
        openmx_mgpu_band_scope_leave(previous);REQUIRE(!openmx_mgpu_scope_active());
    }
    REQUIRE(hipblasDestroy(handle)==HIPBLAS_STATUS_SUCCESS);HIP_OK(hipStreamDestroy(stream));
    HIP_OK(hipFree(da));HIP_OK(hipFree(db));HIP_OK(hipFree(dc));
    HIP_OK(hipFree(dza));HIP_OK(hipFree(dzb));HIP_OK(hipFree(dzc));
    openmx_mgpuGemmRelease();
    std::printf("PASS: physical Band groups G=%d K=%d q=%d used=%d owner=%d; real/complex repeated products, exact lists, device restoration and untouched decline\n",gpus,kpoints,width,used,owner);
}

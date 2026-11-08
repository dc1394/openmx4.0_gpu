#include <cusolverDn.h>
#include <cstdio>
int main(int argc,char**) {
    cusolverDnHandle_t h=nullptr;
    if(cusolverDnCreate(&h)!=CUSOLVER_STATUS_SUCCESS)return 2;
    if(argc>1 && cusolverDnSetMathMode(h,CUSOLVER_FP64_EMULATED_FIXEDPOINT_MATH)!=CUSOLVER_STATUS_SUCCESS)return 3;
    cusolverMathMode_t mode=CUSOLVER_DEFAULT_MATH;
    if(cusolverDnGetMathMode(h,&mode)!=CUSOLVER_STATUS_SUCCESS)return 4;
    std::printf("mode=%d native=%d emulated=%d\n",int(mode),int(CUSOLVER_DEFAULT_MATH),int(CUSOLVER_FP64_EMULATED_FIXEDPOINT_MATH));
    return cusolverDnDestroy(h)==CUSOLVER_STATUS_SUCCESS?0:5;
}

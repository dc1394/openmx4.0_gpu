#!/usr/bin/env python3
"""Independently verify production Band grouping and GEMM dispatch on fake GPUs.

Only CUDA type/function mocks are reused from the runtime fixture; this test
has its own backend spies, cases, reference partitions and assertions.
"""
import ast
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
mock_source = ast.parse((root / 'tests/run_mgpu_runtime_cpu.py').read_text())
headers = {}
for node in mock_source.body:
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        name = node.targets[0].id
        if name in ('cuda', 'complex_header', 'cublas', 'cusolver', 'nvml'):
            headers[name] = ast.literal_eval(node.value)

test = r'''
#include "openmx_mgpu_runtime.cu"
#include "openmx_mgpu_band_plan.h"
extern "C" {int level_stdout=0;}
static int checks=0,products=0;
static std::vector<int> actual;
extern "C" int openmx_dense_cpu_eigen(int,int,int,void*,int,double*) {std::abort();}
extern "C" int openmx_dense_cpu_gemm(char,char,int,int,int,int,const void*,int,const void*,int,void*,int) {std::abort();}
#define CHECK(x) do {++checks;if(!(x)){std::fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#x);std::exit(1);}}while(0)
extern "C" int openmx_magma_mgpu_available() {return 0;}
extern "C" int openmx_magma_mgpu_available_devices(int,const int*) {return 0;}
extern "C" void openmx_magma_mgpu_release_cache() {}
extern "C" size_t openmx_magma_mgpu_device_bytes(int,int,int) {return SIZE_MAX;}
extern "C" const char *openmx_magma_mgpu_last_error() {return "unused";}
extern "C" int openmx_magma_mgpu_solve_host_devices(int,const int*,int,int,int,int,int,int,void*,int,double*) {return 1;}
extern "C" int openmx_mgpuGemmInWorker() {return 0;}
extern "C" void openmx_mgpuGemmLastReport(openmx_mgpu_gemm_report *r) {
 *r={};r->active_gpus=int(actual.size());r->reason="success";
}
extern "C" int openmx_mgpuDgemmTry(cublasHandle_t,int count,const int *devices,cublasOperation_t,cublasOperation_t,
 int,int,int,const double*,const double*,int,const double*,int,const double*,double*,int,cublasStatus_t *status) {
 ++products;actual.assign(devices,devices+count);*status=CUBLAS_STATUS_SUCCESS;return 1;
}
extern "C" int openmx_mgpuZgemmTry(cublasHandle_t,int count,const int *devices,cublasOperation_t,cublasOperation_t,
 int,int,int,const cuDoubleComplex*,const cuDoubleComplex*,int,const cuDoubleComplex*,int,
 const cuDoubleComplex*,cuDoubleComplex*,int,cublasStatus_t *status) {
 ++products;actual.assign(devices,devices+count);*status=CUBLAS_STATUS_SUCCESS;return 1;
}
extern "C" int openmx_mgpuDgemmHostTry(int ngpu,const int *devices,cublasOperation_t ta,cublasOperation_t tb,int m,int n,int k,
 const double *alpha,const double *a,int lda,const double *b,int ldb,const double *beta,double *c,int ldc,cublasStatus_t *status) {
 return openmx_mgpuDgemmTry(nullptr,ngpu,devices,ta,tb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc,status);
}
extern "C" int openmx_mgpuZgemmHostTry(int ngpu,const int *devices,cublasOperation_t ta,cublasOperation_t tb,int m,int n,int k,
 const cuDoubleComplex *alpha,const cuDoubleComplex *a,int lda,const cuDoubleComplex *b,int ldb,const cuDoubleComplex *beta,cuDoubleComplex *c,int ldc,cublasStatus_t *status) {
 return openmx_mgpuZgemmTry(nullptr,ngpu,devices,ta,tb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc,status);
}
static int dispatch(bool complex=false) {
 int status=-1;
 double a=1,b=2,c=3,alpha=1,beta=0;
 cuDoubleComplex za{1,0},zb{2,0},zc{3,0},zone{1,0},zzero{0,0};
 const int handled=complex?
   openmx_mgpu_dispatch_zgemm((void*)1,0,0,1,1,1,&zone,&za,1,&zb,1,&zzero,&zc,1,&status):
   openmx_mgpu_dispatch_dgemm((void*)1,0,0,1,1,1,&alpha,&a,1,&b,1,&beta,&c,1,&status);
 if(handled)CHECK(status==CUBLAS_STATUS_SUCCESS);
 return handled;
}
static void configure(int g,int limit=-1) {
 visible=g;device=0;ratio=2;actual.clear();products=0;
 openmx_mgpu_configure(1,limit<0?g:limit,1);
}
int main() {
 unsetenv("OPENMX_MGPU_NUM_GPUS");setenv("OPENMX_MGPU","1",1);
 setenv("OPENMX_MGPU_GEMM","1",1);setenv("OPENMX_MGPU_VERBOSE","0",1);
 CHECK(!openmx_mgpu_scope_active());
 for(int g=1;g<=16;++g)for(int k=1;k<=9;++k) {
   configure(g);
   const int q=g/k,expected=q>=2 && q<=8?q:0;
   CHECK(openmx_mgpu_band_used_gpus(k)==expected*k);
   for(int point=0;point<k;++point) {
     int devices[8];std::fill_n(devices,8,-123);
     CHECK(openmx_mgpu_band_group(k,point,devices,8)==expected);
     CHECK(openmx_mgpu_band_group(k,point,nullptr,0)==expected);
     for(int j=0;j<8;++j)CHECK(devices[j]==(j<expected?point*q+j:-123));
     CHECK(device==0 && !openmx_mgpu_scope_active());
     if(expected) {
       std::fill_n(devices,8,-123);
       CHECK(openmx_mgpu_band_group(k,point,devices,expected-1)==0);
       for(int j=0;j<8;++j)CHECK(devices[j]==-123);
     }
   }
   CHECK(openmx_mgpu_band_group(k,-1,nullptr,0)==0);
   CHECK(openmx_mgpu_band_group(k,k,nullptr,0)==0);
 }
 configure(4);
 const int colK=openmx_mgpu_band_kpoint_count(0,1,6,1,1,0);
 const int spinK=openmx_mgpu_band_kpoint_count(1,1,6,1,1,0);
 const int ncK=openmx_mgpu_band_kpoint_count(3,1,3,1,1,0);
 CHECK(colK==3 && spinK==3 && ncK==3);
 CHECK(openmx_mgpu_band_used_gpus(colK)==0);
 for(int point=0;point<3;++point) {
   auto old=openmx_mgpu_band_scope_enter(1,3,point);
   CHECK(dispatch()==0 && dispatch(true)==0 && products==0);
   openmx_mgpu_band_scope_leave(old);
 }
 configure(8);const int used=openmx_mgpu_band_used_gpus(spinK);CHECK(used==6);
 // Model Input_std's cap before rank binding; the tail devices stay visible
 // to CUDA but are excluded by scf.Gpu.Num for every Band operation.
 configure(8,used);
 for(int spin=0;spin<2;++spin)for(int point=0;point<3;++point) {
   auto old=openmx_mgpu_band_scope_enter(1,3,point);
   CHECK(dispatch(spin!=0)==1);
   CHECK(actual==std::vector<int>({2*point,2*point+1}));
   CHECK(device==0); // owner outside group is a supported transfer origin
   openmx_mgpu_band_scope_leave(old);
 }
 CHECK(products==6 && dispatch()==0 && !openmx_mgpu_scope_active());
 configure(4);
 auto outer=openmx_mgpu_band_scope_enter(1,2,0);
 CHECK(dispatch()==1 && actual==std::vector<int>({0,1}));
 auto inner=openmx_mgpu_band_scope_enter(1,2,1);
 CHECK(dispatch(true)==1 && actual==std::vector<int>({2,3}));
 auto blocked=openmx_mgpu_band_scope_enter(0,2,1);
 CHECK(dispatch()==0);
 openmx_mgpu_band_scope_leave(blocked);
 CHECK(dispatch()==1 && actual==std::vector<int>({2,3}));
 openmx_mgpu_band_scope_leave(inner);
 CHECK(dispatch()==1 && actual==std::vector<int>({0,1}));
 openmx_mgpu_band_scope_leave(outer);CHECK(dispatch()==0);
 configure(8);setenv("OPENMX_MGPU_NUM_GPUS","4",1);
 CHECK(openmx_mgpu_band_used_gpus(3)==0);unsetenv("OPENMX_MGPU_NUM_GPUS");
 openmx_mgpu_configure(0,8,1);CHECK(openmx_mgpu_band_used_gpus(3)==0);
 openmx_mgpu_configure(1,8,0);CHECK(openmx_mgpu_band_used_gpus(3)==0);
 configure(8);CHECK(openmx_mgpu_band_used_gpus(0)==0 && openmx_mgpu_band_used_gpus(INT_MAX)==0);
 // Only this process's fake-device lock inodes are removed, after admission.
 for(int d=0;d<16;++d) {
   cudaDeviceProp prop{};cudaGetDeviceProperties(&prop,d);char path[128];
   int n=std::snprintf(path,sizeof(path),"/tmp/.openmx-mgpu-%lu-",(unsigned long)getuid());
   for(unsigned char byte:prop.uuid.bytes)n+=std::snprintf(path+n,sizeof(path)-n,"%02x",unsigned(byte));
   unlink(path);
 }
 // A full 2x2x2 noncollinear mesh can share two GPUs. Check the actual
 // dispatch list, cap, scopes, and unchanged current device for all points.
 setenv("OPENMX_MGPU_BAND_SHARED","1",1);
 configure(2);
 CHECK(openmx_mgpu_band_used_gpus(8)==2);
 for(int point=0;point<8;++point) {
   auto old=openmx_mgpu_band_scope_enter(1,8,point);
   CHECK(dispatch()==1 && dispatch(true)==1);
   CHECK(actual==std::vector<int>({0,1}) && device==0);
   openmx_mgpu_band_scope_leave(old);
 }
 configure(4);
 CHECK(openmx_mgpu_band_used_gpus(8)==4);
 for(int point=0;point<8;++point) {
   auto old=openmx_mgpu_band_scope_enter(1,8,point);
   CHECK(dispatch()==1);
   CHECK(actual==std::vector<int>({(point%2)*2,(point%2)*2+1}));
   openmx_mgpu_band_scope_leave(old);
 }
 configure(2,1);
 CHECK(openmx_mgpu_band_used_gpus(8)==0);
 unsetenv("OPENMX_MGPU_BAND_SHARED");
 std::printf("PASS: %d independent Band group/dispatch checks, including shared 8k/2GPU and 8k/4GPU groups\n",checks);
}
'''
with tempfile.TemporaryDirectory(prefix='openmx-band-group-') as folder:
    tmp = Path(folder)
    (tmp / 'cuda_runtime.h').write_text('#include "cuda_runtime_api.h"\n')
    for name, key in [('cuda_runtime_api.h', 'cuda'), ('cuComplex.h', 'complex_header'),
                      ('cublas_v2.h', 'cublas'), ('cusolverDn.h', 'cusolver'), ('nvml.h', 'nvml')]:
        (tmp / name).write_text(headers[key])
    (tmp / 'test.cpp').write_text(test)
    command = shlex.split(os.environ.get('CXX', 'c++')) + [
        '-std=c++17', '-O1', '-Wall', '-Wextra', '-Wno-address',
        '-fsanitize=undefined', '-fno-sanitize-recover=all', '-pthread',
        '-I' + str(tmp), '-I' + str(root / 'source'), str(tmp / 'test.cpp'),
        '-ldl', '-o', str(tmp / 'test')]
    subprocess.run(command, check=True)
    subprocess.run([str(tmp / 'test')], check=True, timeout=30)

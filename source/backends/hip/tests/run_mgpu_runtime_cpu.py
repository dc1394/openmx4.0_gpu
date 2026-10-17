#!/usr/bin/env python3
"""Compile the production runtime with HIP/backends mocked in host memory.

Exercises cluster/topology gates, explicit controls, scalar types/triangles,
autotuning and cached publication, transactional decline/error handling, and
real process admission locks. The real GPU numerical tests remain separate.
"""
from pathlib import Path
import os
import shlex
import subprocess
import tempfile
from mock_hip_sources import stage

root = Path(__file__).resolve().parents[1]
hip = r'''
#pragma once
#include <cstddef>
#include <cstring>
#include <unistd.h>
#include <cstdint>
#include <cstdlib>
using hipStream_t=void*;using hipError_t=int;
using hipDataType=int;
constexpr int HIP_R_64F=0,HIP_C_64F=1,hipSuccess=0;
constexpr int hipMemcpyDeviceToHost=0,hipMemcpyHostToDevice=1;
constexpr int hipDeviceAttributeSingleToDoublePrecisionPerfRatio=0;
struct hipUUID_t {char bytes[16];};
struct hipDeviceProp_t {hipUUID_t uuid;int pciDomainID,pciBusID,pciDeviceID;char gcnArchName[256];};
inline int device=1,visible=2,ratio=2,queries=0,duplicate_uuid=0,property_queries=0,property_fail=0;
inline int uuid_namespace=int(getpid());
inline int peer_support=1;
inline size_t memory_free=size_t(1)<<40;
inline int largest_memory_device=-1;
inline int hipMemGetInfo(size_t *f,size_t *t) {*f=memory_free;*t=size_t(1)<<40;if(largest_memory_device<device)largest_memory_device=device;return 0;}

inline int copy2d_h2d=0,copy_h2d=0,fail_h2d=0,sync_error=0;
inline int hipGetDevice(int *p) {*p=device;return 0;}
inline int hipSetDevice(int d) {device=d;return 0;}
inline int hipGetDeviceCount(int *p) {++queries;*p=visible;return 0;}
inline int hipDeviceGetAttribute(int *p,int,int) {*p=ratio;return 0;}
inline int hipGetDeviceProperties(hipDeviceProp_t *p,int d) {
 ++property_queries;if(property_fail) return 1;
 std::memset(p,0,sizeof(*p));
 std::memcpy(p->uuid.bytes,&uuid_namespace,sizeof(uuid_namespace));p->uuid.bytes[8]=duplicate_uuid?0:char(d);
 p->pciBusID=16+d;std::strcpy(p->gcnArchName,ratio<=2?"gfx942":"gfx1200");return 0;
}
inline int hipGetLastError(){return 0;}
inline int hipMalloc(void **p,size_t n){*p=std::malloc(n);return *p?0:1;}
inline int hipFree(void *p){std::free(p);return 0;}
inline int hipExtGetLinkTypeAndHopCount(int a,int b,uint32_t *t,uint32_t *h){
 const char *v=std::getenv("OPENMX_TEST_XGMI_MODE");int mode=v?std::atoi(v):0;
 *t=(mode==1 || (mode==2 && a==0))?4:2;*h=mode==3?2:1;return mode==4?1:0;
}
inline int hipDeviceCanAccessPeer(int *p,int,int) {*p=peer_support;return 0;}
inline int hipStreamSynchronize(hipStream_t) {return sync_error;}
inline int hipMemcpy2D(void *dst,size_t dp,const void *src,size_t sp,size_t width,size_t height,int kind) {
 if(kind==hipMemcpyHostToDevice) {++copy2d_h2d;if(fail_h2d) return 1;}
 for(size_t j=0;j<height;++j) std::memcpy((char*)dst+j*dp,(const char*)src+j*sp,width);
 return 0;
}
inline int hipMemcpy2DAsync(void *d,size_t dp,const void *s,size_t sp,size_t w,size_t h,int k,hipStream_t) {
 return hipMemcpy2D(d,dp,s,sp,w,h,k);
}
inline int hipMemcpy(void *d,const void *s,size_t bytes,int kind) {
 if(kind==hipMemcpyHostToDevice) {++copy_h2d;if(fail_h2d) return 1;}
 std::memcpy(d,s,bytes);return 0;
}
inline int hipMemcpyAsync(void *d,const void *s,size_t b,int k,hipStream_t) {return hipMemcpy(d,s,b,k);}
'''
complex_header = r'''
#pragma once
struct hipDoubleComplex {double x,y;};
inline hipDoubleComplex make_hipDoubleComplex(double x,double y) {return {x,y};}
'''
hipblas = r'''
#pragma once
#include "hip_complex_compat.h"
using hipblasHandle_t=void*;using hipblasStatus_t=int;using hipblasOperation_t=int;using hipblasFillMode_t=int;
constexpr int HIPBLAS_OP_N=0,HIPBLAS_OP_T=1,HIPBLAS_OP_C=2;
constexpr int HIPBLAS_FILL_MODE_LOWER=0,HIPBLAS_FILL_MODE_UPPER=1,HIPBLAS_STATUS_SUCCESS=0,HIPBLAS_STATUS_ALLOC_FAILED=3;
'''
hipsolver = r'''
#pragma once
#include "hip_runtime_api.h"
using hipsolverDnHandle_t=void*;using hipsolverStatus_t=int;using hipsolverEigMode_t=int;
constexpr int HIPSOLVER_STATUS_INTERNAL_ERROR=7,HIPSOLVER_STATUS_SUCCESS=0,HIPSOLVER_STATUS_EXECUTION_FAILED=6,HIPSOLVER_STATUS_INVALID_VALUE=3,HIPSOLVER_EIG_MODE_VECTOR=1;
inline int hipsolverDnCreate(hipsolverDnHandle_t *h){*h=(void*)1;return 0;}
inline int hipsolverDnDestroy(hipsolverDnHandle_t){return 0;}
inline int hipsolverDnGetStream(hipsolverDnHandle_t,hipStream_t *p) {*p=nullptr;return 0;}
'''
main = r'''
#include "openmx_mgpu_runtime.cu"
#include <sys/wait.h>
#include <thread>
extern "C" {int level_stdout=0;}
static int checks=0,native_calls=0,magma_calls=0,gemm_calls=0;
static int native_delay=0,magma_delay=0,native_fail=0,backend_fail=0,corrupt=0;
static int magma_available=1,worker=0;
static int cpu_eigen_calls=0,cpu_gemm_calls=0,host_gemm_failure=0;
static std::vector<int> last_devices;
#define CHECK(x) do {++checks;if(!(x)){std::fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#x);std::exit(1);}}while(0)
static void delay(int ms) {if(ms) usleep(ms*1000);}
static void result(void *a,int lda,int n,int nev,double *w,bool z) {
 for(int j=0;j<nev;++j) {
  w[j]=j+1;
  for(int i=0;i<n;++i) {
   if(z) ((hipDoubleComplex*)a)[size_t(j)*lda+i]={double(i==j),0};
   else ((double*)a)[size_t(j)*lda+i]=double(i==j);
  }
 }
}
extern "C" int openmx_dense_cpu_eigen(int n,int nev,int z,void *a,int lda,double *w) {
 ++cpu_eigen_calls; result(a,lda,n,nev,w,z); return 1;
}
extern "C" int openmx_dense_cpu_gemm(char,char,int,int,int,int,const void*,int,const void*,int,void*,int) {
 ++cpu_gemm_calls; return 1;
}
extern "C" int openmx_magma_mgpu_available() {return magma_available;}
extern "C" int openmx_magma_mgpu_available_devices(int,const int*) {return magma_available;}
extern "C" void openmx_magma_mgpu_release_cache() {}
extern "C" size_t openmx_magma_mgpu_device_bytes(int count,int n,int) {return n>32766?SIZE_MAX:(size_t(16)<<30)/count;}

extern "C" const char *openmx_magma_mgpu_last_error() {return "mock failure";}
extern "C" int openmx_magma_mgpu_solve_host(int ngpu,int z,int vec,int lower,int n,int first,int last,void *a,int lda,double *w) {
 ++magma_calls;CHECK(ngpu>=2 && vec && lower && first==1);delay(magma_delay);
 result(a,lda,n,last,w,z);if(corrupt) ((double*)a)[0]=99;
 device=0;return backend_fail;
}
extern "C" int openmx_magma_mgpu_solve_host_devices(int ngpu,const int *devices,int z,int vec,int lower,int n,int first,int last,void *a,int lda,double *w) {
 last_devices.assign(devices,devices+ngpu);
 return openmx_magma_mgpu_solve_host(ngpu,z,vec,lower,n,first,last,a,lda,w);
}
extern "C" int openmx_mgpuGemmInWorker() {return worker;}
extern "C" void openmx_mgpuGemmLastReport(openmx_mgpu_gemm_report *r) {*r={};}
extern "C" int openmx_mgpuDgemmTry(hipblasHandle_t,int ngpu,const int *devices,hipblasOperation_t,hipblasOperation_t,int,int,int,const double*,const double*,int,const double*,int,const double*,double*,int,hipblasStatus_t *s) {
 last_devices.assign(devices,devices+ngpu);++gemm_calls;*s=0;return 1;
}
extern "C" int openmx_mgpuZgemmTry(hipblasHandle_t,int ngpu,const int *devices,hipblasOperation_t,hipblasOperation_t,int,int,int,const hipDoubleComplex*,const hipDoubleComplex*,int,const hipDoubleComplex*,int,const hipDoubleComplex*,hipDoubleComplex*,int,hipblasStatus_t *s) {
 last_devices.assign(devices,devices+ngpu);++gemm_calls;*s=0;return 1;
}
extern "C" int openmx_mgpuDgemmHostTry(int ngpu,const int *devices,hipblasOperation_t ta,hipblasOperation_t tb,int m,int n,int k,
 const double *alpha,const double *a,int lda,const double *b,int ldb,const double *beta,double *c,int ldc,hipblasStatus_t *status) {
 if(host_gemm_failure) {*status=HIPBLAS_STATUS_ALLOC_FAILED;return host_gemm_failure==2;}
 return openmx_mgpuDgemmTry(nullptr,ngpu,devices,ta,tb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc,status);
}
extern "C" int openmx_mgpuZgemmHostTry(int ngpu,const int *devices,hipblasOperation_t ta,hipblasOperation_t tb,int m,int n,int k,
 const hipDoubleComplex *alpha,const hipDoubleComplex *a,int lda,const hipDoubleComplex *b,int ldb,const hipDoubleComplex *beta,hipDoubleComplex *c,int ldc,hipblasStatus_t *status) {
 if(host_gemm_failure) {*status=HIPBLAS_STATUS_ALLOC_FAILED;return host_gemm_failure==2;}
 return openmx_mgpuZgemmTry(nullptr,ngpu,devices,ta,tb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc,status);
}
struct Context {void *a;double *w;int *info;int64_t *meig;int n,lda,nev;bool z;};
static int native(void *p) {
 ++native_calls;delay(native_delay);Context &c=*(Context*)p;
 result(c.a,c.lda,c.n,c.nev,c.w,c.z);*c.info=native_fail;*c.meig=c.nev;
 return native_fail?HIPSOLVER_STATUS_EXECUTION_FAILED:HIPSOLVER_STATUS_SUCCESS;
}
struct Matrix {
 static constexpr int n=17,lda=21,nev=7;
 std::vector<double> data,original,w;
 int info=-5;int64_t found=-6;int status=-7;bool z;int uplo;
 Matrix(bool z_=false,int uplo_=0):data(size_t(lda)*n*(z_?2:1),-777),w(nev+2,-555),z(z_),uplo(uplo_) {
  for(int j=0;j<n;++j) for(int i=0;i<n;++i) {
   const bool stored=i==j || (i>j)==(uplo==0);
   const double value=stored?(i==j?i+1:0):std::numeric_limits<double>::quiet_NaN();
   if(z) ((hipDoubleComplex*)data.data())[size_t(j)*lda+i]={value,i==j?19.0:0.0};
   else data[size_t(j)*lda+i]=value;
  }
  original=data;
 }
 int attempt() {
  Context ctx{data.data(),w.data(),&info,&found,n,lda,nev,z};
  return openmx_mgpu_solve_device_try((void*)1,HIPSOLVER_EIG_MODE_VECTOR,uplo,n,z?HIP_C_64F:HIP_R_64F,
   data.data(),lda,nev,w.data(),&found,&info,native,&ctx,&status);
 }
 bool unchanged() {return std::memcmp(data.data(),original.data(),data.size()*sizeof(double))==0;}
 void check_result() {
  CHECK(status==0 && info==0 && found==nev && device==1);
  for(int j=0;j<nev;++j) {
   CHECK(w[j]==j+1);
   for(int i=0;i<n;++i) {
    if(z) {auto q=((hipDoubleComplex*)data.data())[size_t(j)*lda+i];CHECK(q.x==double(i==j)&&q.y==0);}
    else CHECK(data[size_t(j)*lda+i]==double(i==j));
   }
  }
  CHECK(w[nev]==-555 && w[nev+1]==-555);
  for(int j=0;j<n;++j) for(int i=0;i<lda;++i) if(i>=n || j>=nev) {
   const size_t pos=(size_t(j)*lda+i)*(z?2:1);
   CHECK(std::memcmp(data.data()+pos,original.data()+pos,(z?2:1)*sizeof(double))==0);
  }
 }
};
static void reset() {
 unsetenv("OPENMX_MGPU");unsetenv("OPENMX_MGPU_NUM_GPUS");unsetenv("OPENMX_MGPU_GEMM");
 unsetenv("OPENMX_MGPU_TUNE_TRIALS");
 unsetenv("OPENMX_MGPU_GEMM_MIN_N");setenv("OPENMX_TEST_XGMI_MODE","0",1);peer_support=1;
 setenv("OPENMX_MGPU_SOLVER_MIN_N","0",1);setenv("OPENMX_MGPU_SOLVER","magma",1);
 native_calls=magma_calls=gemm_calls=copy2d_h2d=copy_h2d=0;
 native_delay=magma_delay=native_fail=backend_fail=corrupt=0;
 device=1;visible=2;ratio=2;queries=property_queries=property_fail=0;duplicate_uuid=fail_h2d=sync_error=worker=0;
 magma_available=1;memory_free=size_t(1)<<40;largest_memory_device=-1;openmx_mgpu_force_magma(0);unsetenv("OPENMX_MGPU_MEMORY");
 openmx_mgpu_configure(-1,2,1);cluster_scope=0;band_kpoints=0;band_kpoint=-1;last_devices.clear();
}

// AMD's original MAGMA wrapper publishes eigenvalues to host memory.
// The cooperative adapter must give native calibration a private W and
// leave caller guards unchanged on a decline or an execution failure.
struct HostResultContext {void *a;double *public_w;int n,nev;bool z;};
static int native_host_result(void *opaque,double *private_w) {
 HostResultContext *c=(HostResultContext*)opaque;
 CHECK(private_w!=c->public_w);++native_calls;delay(native_delay);
 result(c->a,c->n,c->n,c->nev,private_w,c->z);
 return native_fail;
}
static void host_result_cases() {
 for(bool z:{false,true}) for(int mode=0;mode<5;mode++) {
  reset();openmx_mgpu_scope_enter(1);
  const int n=19+mode,nev=5;
  std::vector<double> a(size_t(n)*n*(z?2:1)),w(n+2,-555);
  for(int i=0;i<n;i++)a[(size_t(i)*n+i)*(z?2:1)]=i+1;
  auto original=a;int found=-21,status=-22;
  HostResultContext context{a.data(),w.data(),n,nev,z};
  if(mode==0)unsetenv("OPENMX_MGPU_SOLVER");
  if(mode==1)visible=1;
  if(mode==3){setenv("OPENMX_MGPU_SOLVER","auto",1);magma_delay=20;}
  if(mode==4)fail_h2d=1;
  int handled=openmx_mgpu_solve_host_result_try(z,n,nev,a.data(),w.data(),&found,native_host_result,&context,&status);
  CHECK(device==1);
  if(mode<2) {
   CHECK(!handled && found==-21 && status==-22 && a==original && native_calls==0);
   for(double x:w)CHECK(x==-555);
  } else if(mode==4) {
   CHECK(handled && status!=0 && found==0);
   for(double x:w)CHECK(x==-555);
  } else {
   CHECK(handled && status==0 && found==nev);
   CHECK(mode==3?native_calls>=2:native_calls==0);
   for(int j=0;j<nev;j++){
    CHECK(w[j]==j+1);
    for(int i=0;i<n;i++){
     size_t pos=(size_t(j)*n+i)*(z?2:1);
     CHECK(a[pos]==double(i==j));if(z)CHECK(a[pos+1]==0);
    }
   }
   for(int i=nev;i<n+2;i++)CHECK(w[i]==-555);
  }
 }
 reset();
}

int main() {
 host_result_cases();
 // Auxiliary work shares visibility/input caps, but is not filtered by an
 // eigensolver performance heuristic. Scoped Band lists remain exact.
 reset();visible=8;ratio=64;openmx_mgpu_configure(-1,6,1);
 int auxiliary[64];std::fill_n(auxiliary,64,-17);
 CHECK(openmx_mgpu_pool_devices(auxiliary,5)==0);
 CHECK(openmx_mgpu_pool_devices(auxiliary,64)==6);
 for(int i=0;i<6;i++)CHECK(auxiliary[i]==i);
 CHECK(openmx_mgpu_scope_devices(auxiliary,64)==0);
 auto auxiliary_scope=openmx_mgpu_band_scope_enter(1,3,2);
 CHECK(openmx_mgpu_scope_devices(auxiliary,64)==2 && auxiliary[0]==4 && auxiliary[1]==5);
 void *admission_token=openmx_mgpu_admit_devices(auxiliary,2);CHECK(admission_token!=nullptr);openmx_mgpu_release_devices(admission_token);
 auxiliary[0]=6;CHECK(openmx_mgpu_admit_devices(auxiliary,1)==nullptr);
 openmx_mgpu_band_scope_leave(auxiliary_scope);
 setenv("OPENMX_MGPU_NUM_GPUS","4",1);
 CHECK(openmx_mgpu_pool_devices(auxiliary,64)==4);
 auxiliary_scope=openmx_mgpu_band_scope_enter(1,3,0);
 CHECK(openmx_mgpu_scope_devices(auxiliary,64)==0);openmx_mgpu_band_scope_leave(auxiliary_scope);
 openmx_mgpu_configure(0,4,1);auxiliary[0]=device;
 admission_token=openmx_mgpu_admit_devices(auxiliary,1);CHECK(admission_token!=nullptr);openmx_mgpu_release_devices(admission_token);
 auxiliary[0]=(device+1)%4;CHECK(openmx_mgpu_admit_devices(auxiliary,1)==nullptr);
 unsetenv("OPENMX_MGPU_NUM_GPUS");
 // ELPA/COSMA and a failed Band group retain a separate auxiliary pool.
 openmx_mgpu_configure(0,3,0);openmx_mgpu_configure_auxiliary(-1);
 CHECK(openmx_mgpu_pool_devices(auxiliary,64)==3);
 int aux_dense_scope=openmx_mgpu_scope_enter(1);
 CHECK(openmx_mgpu_scope_devices(auxiliary,64)==0);
 CHECK(openmx_mgpu_host_plan(3,26000,1)==0);
 admission_token=openmx_mgpu_admit_devices(auxiliary,3);CHECK(admission_token!=nullptr);openmx_mgpu_release_devices(admission_token);
 auxiliary[0]=3;CHECK(openmx_mgpu_admit_devices(auxiliary,1)==nullptr);
 openmx_mgpu_scope_leave(aux_dense_scope);
 setenv("OPENMX_GPU","0",1);CHECK(openmx_mgpu_pool_devices(auxiliary,64)==0);unsetenv("OPENMX_GPU");
 openmx_mgpu_configure_auxiliary(0);CHECK(openmx_mgpu_pool_devices(auxiliary,64)==0);
 // Capacity planning never visits a GPU beyond either configured limit.
 reset();visible=8;openmx_mgpu_configure(-1,6,1);
 CHECK(openmx_mgpu_host_plan(3,26000,1)==2 && largest_memory_device==5 && device==1);
 openmx_mgpu_configure(-1,4,1);largest_memory_device=-1;
 CHECK(openmx_mgpu_host_plan(3,26000,1)==0 && largest_memory_device==-1);
 openmx_mgpu_configure(-1,1,1);
 CHECK(openmx_mgpu_host_plan(0,26000,1)==0 && std::strstr(openmx_mgpu_host_error(),"limit=1"));
 openmx_mgpu_configure(-1,2,1);memory_free=1;
 CHECK(openmx_mgpu_host_plan(0,26000,1)==0 && std::strstr(openmx_mgpu_host_error(),"cannot fit") && device==1);
 memory_free=size_t(1)<<40;magma_available=0;CHECK(openmx_mgpu_host_plan(0,26000,1)==0);
 magma_available=1;CHECK(openmx_mgpu_host_plan(0,32767,1)==0);
 // The GPU-free limits of a plan (pool, LP64 order) query no GPU and give
 // the same reasons, so the capacity decision can test them first.
 largest_memory_device=-1;openmx_mgpu_configure(-1,1,1);
 CHECK(openmx_mgpu_host_precheck(0,26000,1)==0 && std::strstr(openmx_mgpu_host_error(),"limit=1"));
 openmx_mgpu_configure(-1,2,1);
 CHECK(openmx_mgpu_host_precheck(0,32767,1)==0 && std::strstr(openmx_mgpu_host_error(),"LP64"));
 memory_free=1;CHECK(openmx_mgpu_host_precheck(0,26000,1)==2 && openmx_mgpu_host_error()[0]=='\0');
 CHECK(largest_memory_device==-1 && device==1);memory_free=size_t(1)<<40;
 // A multi-node run plans and runs the Cluster capacity path on the
 // planning rank's node; Band groups and device-resident cooperation stay
 // single-node, and the input/visibility limits still apply.
 reset();visible=2;openmx_mgpu_configure(-1,2,0);
 CHECK(openmx_mgpu_multi_node()==1 && openmx_mgpu_gpu_limit()==2);
 CHECK(openmx_mgpu_host_plan(0,26000,1)==2);
 CHECK(openmx_mgpu_host_plan(2,26000,1)==0 && openmx_mgpu_band_used_gpus(2)==0);
 // The GPU workers keep the node's whole pool in a multi-node run, and so
 // does a cluster owner's cooperative product; a Band group scope stays
 // single-node (band_used_gpus above).
 {int pool[8];CHECK(openmx_mgpu_pool_devices(pool,8)==2 && pool[0]==0 && pool[1]==1);}
 {int pool[8];int d[2]={0,1};openmx_mgpu_scope_enter(1);
  CHECK(openmx_mgpu_scope_devices(pool,8)==2);
  void *a=openmx_mgpu_admit_devices(d,2);CHECK(a!=nullptr);openmx_mgpu_release_devices(a);
  setenv("OPENMX_MGPU_GEMM","1",1);gemm_calls=0;
  double alpha=1,beta=0,ga=2,gb=3,gc=0;int gstatus=-1;
  CHECK(openmx_mgpu_dispatch_dgemm((void*)1,0,0,1,1,1,&alpha,&ga,1,&gb,1,&beta,&gc,1,&gstatus)==1 && gemm_calls==1);
  unsetenv("OPENMX_MGPU_GEMM");openmx_mgpu_scope_leave(0);}
 {int pool[8];visible=4;openmx_mgpu_configure(-1,4,0);setenv("OPENMX_MGPU_NUM_GPUS","3",1);
  CHECK(openmx_mgpu_pool_devices(pool,8)==3);unsetenv("OPENMX_MGPU_NUM_GPUS");
  openmx_mgpu_configure(0,4,0);CHECK(openmx_mgpu_pool_devices(pool,8)==0);
  visible=2;openmx_mgpu_configure(-1,2,0);}
 openmx_mgpu_scope_enter(1);
 {std::vector<double> a(16,0.0),w(4,0.0);
  for(int i=0;i<4;i++) a[size_t(i)*5]=double(i+1);
  CHECK(openmx_mgpu_host_solve(4,4,0,a.data(),4,w.data())==1 && magma_calls==1 && last_devices.size()==2);}
 // Late resource failures use CPU, but a numerical failure or a failed
 // output publication must never be reported as a capacity recovery.
 {std::vector<double> a(32,0.0),w(4,0.0);
  for (int z=0;z<2;z++) {
   int before=cpu_eigen_calls;
   backend_fail=OPENMX_MAGMA_RESOURCES;
   CHECK(openmx_mgpu_host_solve(4,4,z,a.data(),4,w.data())==1 && cpu_eigen_calls==before+1);
   backend_fail=OPENMX_MAGMA_VALIDATION_ERROR;
   CHECK(openmx_mgpu_host_solve(4,4,z,a.data(),4,w.data())==0 && cpu_eigen_calls==before+1);
  }
  backend_fail=0;
  host_gemm_failure=1;int before=cpu_gemm_calls;
  CHECK(openmx_mgpu_host_dgemm(HIPBLAS_OP_N,HIPBLAS_OP_N,2,2,2,a.data(),2,a.data(),2,a.data()+8,2)==1);
  CHECK(openmx_mgpu_host_zgemm(HIPBLAS_OP_N,HIPBLAS_OP_N,2,2,2,a.data(),2,a.data(),2,a.data()+16,2)==1);
  CHECK(cpu_gemm_calls==before+2);
  host_gemm_failure=2;
  CHECK(openmx_mgpu_host_dgemm(HIPBLAS_OP_N,HIPBLAS_OP_N,2,2,2,a.data(),2,a.data(),2,a.data()+8,2)==0);
  CHECK(cpu_gemm_calls==before+2);
  host_gemm_failure=0;
 }
 {Matrix m;CHECK(m.attempt()==0 && m.unchanged());} // device-resident MAGMA stays single-node
 visible=1;CHECK(openmx_mgpu_host_plan(0,26000,1)==0 && std::strstr(openmx_mgpu_host_error(),"pool=1"));
 visible=2;openmx_mgpu_configure(0,2,0);CHECK(openmx_mgpu_host_plan(0,26000,1)==0);
 visible=4;openmx_mgpu_configure(-1,2,0);setenv("OPENMX_MGPU_NUM_GPUS","3",1);
 CHECK(openmx_mgpu_host_plan(0,26000,1)==2);unsetenv("OPENMX_MGPU_NUM_GPUS");
 reset();CHECK(openmx_mgpu_multi_node()==0);
 reset();unsetenv("OPENMX_MGPU_SOLVER");openmx_mgpu_scope_enter(1);
 {Matrix m;CHECK(m.attempt()==0 && m.unchanged() && native_calls==0 && magma_calls==0);}
 CHECK(!openmx_mgpu_host_required());openmx_mgpu_force_magma(1);CHECK(openmx_mgpu_host_required());
 openmx_mgpu_force_magma(0);setenv("OPENMX_MGPU_MEMORY","host",1);CHECK(openmx_mgpu_host_required());
 reset();{Matrix m;CHECK(m.attempt()==0 && m.unchanged() && queries==0);}
 int previous=openmx_mgpu_scope_enter(1);CHECK(previous==0 && openmx_mgpu_scope_active());
 int nested=openmx_mgpu_scope_enter(0);CHECK(!openmx_mgpu_scope_active());
 openmx_mgpu_scope_leave(nested);CHECK(openmx_mgpu_scope_active());openmx_mgpu_scope_leave(previous);
 for(int setting=0;setting<6;++setting) {
  reset();openmx_mgpu_scope_enter(1);
  if(setting==0) openmx_mgpu_configure(-1,2,0);
  if(setting==1) openmx_mgpu_configure(0,2,1);
  if(setting==2) setenv("OPENMX_MGPU","0",1);
  if(setting==3) visible=1;
  if(setting==4) ratio=32;
  if(setting==5) setenv("OPENMX_MGPU_NUM_GPUS","1",1);
  Matrix m;CHECK(m.attempt()==0 && m.unchanged() && magma_calls==0 && native_calls==0);
 }
 for(bool z:{false,true}) for(int uplo:{0,1}) {
  reset();openmx_mgpu_scope_enter(1);Matrix m(z,uplo);
  CHECK(m.attempt()==1 && magma_calls==1 && native_calls==0);m.check_result();
  CHECK(copy2d_h2d==1 && copy_h2d==2); // exactly one publication
 }
 reset();openmx_mgpu_scope_enter(1);ratio=32;setenv("OPENMX_MGPU","1",1);
 {Matrix m;CHECK(m.attempt()==1);m.check_result();}
 reset();openmx_mgpu_scope_enter(1);duplicate_uuid=1;
 {Matrix m;CHECK(m.attempt()==1);m.check_result();} // no flock self-deadlock
 for(int failure=0;failure<2;++failure) {
  reset();openmx_mgpu_scope_enter(1);if(failure) corrupt=1;else backend_fail=4;
  Matrix m;CHECK(m.attempt()==0 && m.unchanged() && copy2d_h2d==0 && device==1);
 }
 reset();openmx_mgpu_scope_enter(1);setenv("OPENMX_MGPU_SOLVER","auto",1);
 {Matrix m;m.data[1]=std::numeric_limits<double>::quiet_NaN();m.original=m.data;
  CHECK(m.attempt()==1 && m.status==HIPSOLVER_STATUS_INVALID_VALUE && m.unchanged() && native_calls==0 && magma_calls==0);}
 reset();openmx_mgpu_scope_enter(1);fail_h2d=1;
 {Matrix m;CHECK(m.attempt()==1 && m.status==HIPSOLVER_STATUS_EXECUTION_FAILED);}
 // Fast MAGMA is selected once, then the cached path has a single commit.
 reset();openmx_mgpu_scope_enter(1);setenv("OPENMX_MGPU_SOLVER","auto",1);native_delay=30;
 {Matrix m;CHECK(m.attempt()==1);m.check_result();CHECK(native_calls==2 && magma_calls==2);}
 copy2d_h2d=copy_h2d=0;
 {Matrix m;CHECK(m.attempt()==1);m.check_result();CHECK(native_calls==2 && magma_calls==3 && copy2d_h2d==1 && copy_h2d==2);}
 // Native selection declines cached calls before host copies/backend calls.
 reset();openmx_mgpu_scope_enter(1);setenv("OPENMX_MGPU_SOLVER","auto",1);magma_delay=30;
 {Matrix m;CHECK(m.attempt()==1);m.check_result();}
 property_queries=0;property_fail=1;
 {Matrix m;CHECK(m.attempt()==0 && m.unchanged() && native_calls==2 && magma_calls==2);}
 CHECK(property_queries<=1); // HIP architecture query may reject before UUID admission
 reset();openmx_mgpu_scope_enter(1);setenv("OPENMX_MGPU_SOLVER","auto",1);setenv("OPENMX_MGPU_TUNE_TRIALS","1",1);
 {Matrix m;CHECK(m.attempt()==1 && native_calls==1 && magma_calls==1);m.check_result();}
 // Failed native calibration and failed trials restore the exact input,
 // including unused NaN triangle bytes, for the ordinary solver fallback.
 reset();openmx_mgpu_scope_enter(1);setenv("OPENMX_MGPU_SOLVER","auto",1);native_fail=backend_fail=1;
 {Matrix m(false,1);CHECK(m.attempt()==0 && m.unchanged() && device==1);}
 // Removed or unknown solver names cannot invoke a cooperative solve.
 for(const char *mode:{"paper","unknown"}) {
  reset();openmx_mgpu_scope_enter(1);setenv("OPENMX_MGPU_SOLVER",mode,1);
  Matrix m;CHECK(m.attempt()==0 && m.unchanged() && magma_calls==0 && native_calls==0);
 }
 reset();openmx_mgpu_scope_enter(1);setenv("OPENMX_MGPU_GEMM","1",1);
 double alpha=1,beta=0,a=2,b=3,c=0;int status=-1;
 CHECK(openmx_mgpu_dispatch_dgemm((void*)1,0,0,1,1,1,&alpha,&a,1,&b,1,&beta,&c,1,&status)==1 && gemm_calls==1);
 worker=1;CHECK(openmx_mgpu_dispatch_dgemm((void*)1,0,0,1,1,1,&alpha,&a,1,&b,1,&beta,&c,1,&status)==0 && gemm_calls==1);
 // A device-resident product stays on its owner's GPU by default, like the
 // eigensolver; cooperation is opt-in. Unknown values keep it off.
 for(const char *mode:{(const char*)nullptr,"0","off","yes"}) {
  reset();openmx_mgpu_scope_enter(1);
  if(mode) setenv("OPENMX_MGPU_GEMM",mode,1);
  CHECK(openmx_mgpu_dispatch_dgemm((void*)1,0,0,8192,8192,8192,&alpha,&a,8192,&b,8192,&beta,&c,8192,&status)==0 && gemm_calls==0);
 }
 // Opted-in cooperation (on) is portable to PCIe and skinny matrices. The
 // optional legacy auto policy retains its NVLink and size filters.
 for(int topology=0;topology<=5;++topology) {
  reset();openmx_mgpu_scope_enter(1);setenv("OPENMX_MGPU_GEMM","on",1);
  setenv("OPENMX_TEST_XGMI_MODE",std::to_string(topology).c_str(),1);
  peer_support=0;ratio=64;
  CHECK(openmx_mgpu_dispatch_dgemm((void*)1,0,0,17,7,9,&alpha,&a,17,&b,9,&beta,&c,17,&status)==1 && gemm_calls==1);
 }
 for(int topology=0;topology<=5;++topology) {
  reset();openmx_mgpu_scope_enter(1);setenv("OPENMX_MGPU_GEMM_MIN_N","1",1);
  setenv("OPENMX_MGPU_GEMM","auto",1);
  const std::string mode=std::to_string(topology==5?1:topology);
  setenv("OPENMX_TEST_XGMI_MODE",mode.c_str(),1);if(topology==5) peer_support=0;
  const int accepted=openmx_mgpu_dispatch_dgemm((void*)1,0,0,8192,4096,8192,&alpha,&a,8192,&b,8192,&beta,&c,8192,&status);
  CHECK(accepted==(topology==1) && gemm_calls==(topology==1));
 }
 reset();openmx_mgpu_scope_enter(1);setenv("OPENMX_TEST_XGMI_MODE","1",1);
 setenv("OPENMX_MGPU_GEMM","auto",1);
 CHECK(openmx_mgpu_dispatch_dgemm((void*)1,0,0,8192,512,8192,&alpha,&a,8192,&b,8192,&beta,&c,8192,&status)==0);
 // Planning is independent of scope. Divide the full pool before imposing
 // the per-backend limit; G=16,K=3 has three valid five-GPU groups.
 for(int g=1;g<=24;++g) for(int total=1;total<=16;++total) {
  reset();visible=g;openmx_mgpu_configure(1,g,1);
  const int q=g/total;const bool eligible=q>=2 && q<=8;
  CHECK(openmx_mgpu_band_used_gpus(total)==(eligible?q*total:0));
  CHECK(!openmx_mgpu_scope_active() && device==1);
  for(int kp=0;kp<total;++kp) {
   int ids[8];std::fill_n(ids,8,-17);
   CHECK(openmx_mgpu_band_group(total,kp,ids,8)==(eligible?q:0));
   for(int j=0;j<8;++j) CHECK(ids[j]==(eligible && j<q?kp*q+j:-17));
  }
 }
 reset();visible=8;openmx_mgpu_configure(1,8,1);
 int ids[8];std::fill_n(ids,8,-17);
 CHECK(openmx_mgpu_band_group(3,1,ids,1)==0 && ids[0]==-17);
 CHECK(openmx_mgpu_band_group(3,1,nullptr,0)==2);
 CHECK(openmx_mgpu_band_used_gpus(0)==0 && openmx_mgpu_band_used_gpus(-1)==0);
 CHECK(openmx_mgpu_band_group(3,-1,ids,8)==0 && openmx_mgpu_band_group(3,3,ids,8)==0);
 setenv("OPENMX_MGPU_NUM_GPUS","4",1);CHECK(openmx_mgpu_band_used_gpus(3)==0);
 unsetenv("OPENMX_MGPU_NUM_GPUS");setenv("OPENMX_MGPU","0",1);CHECK(openmx_mgpu_band_used_gpus(3)==0);
 unsetenv("OPENMX_MGPU");openmx_mgpu_configure(-1,8,1);ratio=32;CHECK(openmx_mgpu_band_used_gpus(3)==6);
 setenv("OPENMX_MGPU","1",1);CHECK(openmx_mgpu_band_used_gpus(3)==6);
 openmx_mgpu_configure(0,8,1);CHECK(openmx_mgpu_band_used_gpus(3)==0);
 openmx_mgpu_configure(1,8,0);CHECK(openmx_mgpu_band_used_gpus(3)==0);
 // Nesting and temporary worker suppression restore the complete group.
 reset();visible=8;openmx_mgpu_configure(1,8,1);openmx_mgpu_scope_enter(1);
 auto outer=openmx_mgpu_band_scope_enter(1,3,1);
 CHECK(devices_for_operation(ids)==2 && ids[0]==2 && ids[1]==3);
 auto inner=openmx_mgpu_band_scope_enter(1,3,2);
 CHECK(devices_for_operation(ids)==2 && ids[0]==4 && ids[1]==5);
 int suppressed=openmx_mgpu_scope_enter(0);CHECK(devices_for_operation(ids)==0);
 openmx_mgpu_scope_leave(suppressed);openmx_mgpu_band_scope_leave(inner);
 CHECK(devices_for_operation(ids)==2 && ids[0]==2 && ids[1]==3);
 openmx_mgpu_band_scope_leave(outer);CHECK(devices_for_operation(ids)==8 && ids[0]==0 && ids[7]==7);
 // Every host backend and both GEMM types receive the exact non-prefix list,
 // even when the immutable data owner belongs to another worker group.
 for(bool z:{false,true}) {
  reset();visible=8;openmx_mgpu_configure(1,8,1);
  auto state=openmx_mgpu_band_scope_enter(1,3,1);setenv("OPENMX_MGPU_SOLVER","magma",1);
  Matrix m(z);CHECK(m.attempt()==1);m.check_result();CHECK(last_devices==std::vector<int>({2,3}));
  setenv("OPENMX_MGPU_GEMM","1",1);
  CHECK(openmx_mgpu_dispatch_dgemm((void*)1,0,0,1,1,1,&alpha,&a,1,&b,1,&beta,&c,1,&status)==1);
  CHECK(last_devices==std::vector<int>({2,3}) && device==1);
  hipDoubleComplex za{1,0},zb{0,0};
  CHECK(openmx_mgpu_dispatch_zgemm((void*)1,0,0,1,1,1,&za,&za,1,&za,1,&zb,&za,1,&status)==1);
  CHECK(last_devices==std::vector<int>({2,3}) && device==1);openmx_mgpu_band_scope_leave(state);
 }
 // Force-on cannot turn an undersized Band allocation into a cluster group.
 reset();visible=4;openmx_mgpu_configure(1,4,1);openmx_mgpu_band_scope_enter(1,3,0);
 setenv("OPENMX_MGPU_GEMM","1",1);
 {Matrix m;CHECK(m.attempt()==0 && m.unchanged() && magma_calls==0);}
 CHECK(openmx_mgpu_dispatch_dgemm((void*)1,0,0,1,1,1,&alpha,&a,1,&b,1,&beta,&c,1,&status)==0);
 // Equal-size groups sharing the same origin require separate calibration.
 reset();visible=4;openmx_mgpu_configure(1,4,1);setenv("OPENMX_MGPU_SOLVER","auto",1);native_delay=30;
 auto s0=openmx_mgpu_band_scope_enter(1,2,0);
 {Matrix m;CHECK(m.attempt()==1);m.check_result();}
 openmx_mgpu_band_scope_leave(s0);auto s1=openmx_mgpu_band_scope_enter(1,2,1);
 {Matrix m;CHECK(m.attempt()==1);m.check_result();}
 CHECK(native_calls==4 && magma_calls==4 && last_devices==std::vector<int>({2,3}));
 {Matrix m;CHECK(m.attempt()==1);m.check_result();}
 CHECK(native_calls==4 && magma_calls==5);openmx_mgpu_band_scope_leave(s1);
 // Distinct owner processes serialize overlapping device subsets. Fork
 // before the mutex is locked, and preserve the mocked physical UUIDs.
 reset();int ready[2],go[2];CHECK(pipe(ready)==0 && pipe(go)==0);
 const pid_t child=fork();CHECK(child>=0);
 if(child==0) {
  close(ready[0]);close(go[1]);char token='r';
  if(write(ready[1],&token,1)!=1 || read(go[0],&token,1)!=1) _exit(2);
  const int devs[2]={0,1};Admission admission(devs,2);_exit(admission.ok?0:3);
 }
 close(ready[1]);close(go[0]);char token;CHECK(read(ready[0],&token,1)==1);
 int child_status=-1;
 {
  const int devs[2]={1,0};Admission admission(devs,2);CHECK(admission.ok);
  CHECK(write(go[1],"g",1)==1);usleep(60000);
  CHECK(waitpid(child,&child_status,WNOHANG)==0);
 }
 CHECK(waitpid(child,&child_status,0)==child && WIFEXITED(child_status) && WEXITSTATUS(child_status)==0);
 close(ready[0]);close(go[1]);
 std::printf("PASS: multi-GPU runtime scope/topology/cache/rollback/publication (%d assertions)\n",checks);
 // Tests create PID-specific UUID lock files; remove only after all users
 // have exited their admission scope. Production intentionally never unlinks.
 for(int d=0;d<8;++d) {hipDeviceProp_t p;hipGetDeviceProperties(&p,d);char name[128];int used=std::snprintf(name,sizeof(name),"/tmp/.openmx-mgpu-%lu-",(unsigned long)getuid());for(unsigned char b:p.uuid.bytes)used+=std::snprintf(name+used,sizeof(name)-used,"%02x",unsigned(b));unlink(name);}
}
'''
with tempfile.TemporaryDirectory(prefix="openmx-mgpu-runtime-") as folder:
    tmp = Path(folder)
    for name, text in (("hip_runtime_api.h", hip), ("hip_complex_compat.h", complex_header),
                       ("hipblas_compat.h", hipblas), ("hipsolver_compat.h", hipsolver),
                       ("test.cpp", main)):
        (tmp / name).write_text(text)
    stage(root,tmp,"openmx_mgpu_runtime.cu")
    (tmp/'hip_runtime_compat.h').write_text('#include "hip_runtime_api.h"\n')
    (tmp/'hsa').mkdir()
    (tmp/'hsa/hsa_ext_amd.h').write_text('#define HSA_AMD_LINK_INFO_TYPE_XGMI 4\n')
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    command = compiler + ["-std=c++17", "-O1", "-Wall", "-Wextra",
        "-Wno-address", "-fsanitize=undefined", "-fno-sanitize-recover=all", "-pthread", "-I"+str(tmp),
        "-I"+str(root/"source"), str(tmp/"test.cpp"), "-ldl", "-o", str(tmp/"test")]
    subprocess.run(command, check=True)
    environment = os.environ.copy()
    environment["LD_LIBRARY_PATH"] = str(tmp) + ":" + environment.get("LD_LIBRARY_PATH", "")
    subprocess.run([str(tmp/"test")], check=True, timeout=30, env=environment)

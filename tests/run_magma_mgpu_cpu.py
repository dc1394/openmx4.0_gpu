#!/usr/bin/env python3
"""Compile the production MAGMA adapter against deterministic CUDA/MAGMA mocks.

Checks dispatch/ownership, triangle and partial-output contracts, sampled
validation, resource/errors, rollback and the no-MAGMA build. Numerical
accuracy of the real library requires a separate multi-GPU smoke test.
"""
from pathlib import Path
import os
import shlex
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
cuda = r'''
#pragma once
#include <cstddef>
constexpr int cudaSuccess=0;
inline int device=3, visible=4, sync_fail=0, query_fail=0;
inline size_t free_bytes=size_t(40)<<30;
inline int cudaGetDevice(int *p) {*p=device;return query_fail;}
inline int cudaGetDeviceCount(int *p) {*p=visible;return query_fail;}
inline int cudaSetDevice(int p) {device=p;return 0;}
inline int cudaMemGetInfo(size_t *f,size_t *t) {*f=free_bytes;*t=size_t(40)<<30;return query_fail;}
inline int cudaDeviceSynchronize() {return sync_fail;}
'''
magma = r'''
#pragma once
#include <complex>
#include <limits>
using magma_int_t=int;
using magmaDoubleComplex=std::complex<double>;
using magma_vec_t=int;
using magma_range_t=int;
constexpr int MagmaMaxGPUs=8,MagmaVec=1,MagmaNoVec=0,MagmaLower=0,MagmaRangeAll=0,MagmaRangeI=1,MAGMA_SUCCESS=0;
#define MAGMA_Z_REAL(x) (x).real()
inline int init_count=0,init_calls=0,init_fail=0,solve_fail=0,corrupt=0,query_bad=0,calls=0,last_ngpu=0,last_first=0;
inline int magma_init() {++init_count;++init_calls;return init_fail;}
inline int magma_finalize() {--init_count;return 0;}
template<class T> int mock(int ngpu,int job,int uplo,int n,T *a,int lda,int first,int last,int *m,double *w,int *info) {
 ++calls;last_ngpu=ngpu;last_first=first;
 if(uplo!=MagmaLower || lda!=n) return *info=-10;
 // Upstream scalar quick return deliberately does not set m.
 if(n!=1) *m=last-first+1;
 for(int j=0;j<last-first+1;++j) w[j]=first-1+j;
 if(job) for(int j=0;j<last-first+1;++j) for(int i=0;i<n;++i) a[j*n+i]=T(i==first-1+j?1:0);
 if(corrupt && job) a[0]=T(99);
 return *info=solve_fail;
}
inline int magma_dsyevdx_m(int ngpu,int job,int,int uplo,int n,double *a,int lda,double,double,
 int first,int last,int *m,double *w,double *work,int lwork,int *iw,int,int *info) {
 if(lwork==-1) {*work=query_bad?std::numeric_limits<double>::infinity():16;*iw=8;return *info=0;}
 return mock(ngpu,job,uplo,n,a,lda,first,last,m,w,info);
}
inline int magma_zheevdx_m(int ngpu,int job,int,int uplo,int n,magmaDoubleComplex *a,int lda,double,double,
 int first,int last,int *m,double *w,magmaDoubleComplex *work,int lwork,double *rw,int,int *iw,int,int *info) {
 if(lwork==-1) {*work=16;*rw=16;*iw=8;return *info=0;}
 return mock(ngpu,job,uplo,n,a,lda,first,last,m,w,info);
}
'''
main = r'''
#include "openmx_magma_mgpu.cpp"
#include <cstdlib>
#include <new>
static int allocation_failure=-1;
__attribute__((noinline)) void *operator new(std::size_t size) {
 if(allocation_failure==0) {allocation_failure=-1;throw std::bad_alloc();}
 if(allocation_failure>0) --allocation_failure;
 if(void *p=std::malloc(size?size:1)) return p;
 throw std::bad_alloc();
}
__attribute__((noinline)) void operator delete(void *p) noexcept {std::free(p);}
__attribute__((noinline)) void operator delete(void *p,std::size_t) noexcept {std::free(p);}
static int checks=0;
#define CHECK(x) do {++checks;if(!(x)){std::fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#x);std::exit(1);}}while(0)
#ifdef OPENMX_HAVE_MAGMA
template<class T> void success(int lower,int vectors,int n=257) {
 const int lda=n+7;std::vector<T>a(size_t(lda)*n,T(-777)),waste;
 std::vector<double>w(8,-55);
 for(int j=0;j<n;++j) for(int i=0;i<n;++i)
  a[size_t(j)*lda+i]=((i>=j)==bool(lower)||i==j)?T(i==j?i:0):T(std::numeric_limits<double>::quiet_NaN());
 const auto before=a;
 CHECK(openmx_magma_mgpu_solve_host(2,sizeof(T)!=8,vectors,lower,n,3,6,a.data(),lda,w.data())==0);
 CHECK(device==3 && init_count==1 && last_ngpu==2 && last_first==3);
 for(int j=0;j<4;++j) CHECK(w[j]==j+2);
 CHECK(w[4]==-55);
 for(int j=0;j<n;++j) for(int i=0;i<lda;++i) {
  if(vectors && j<4 && i<n) CHECK(a[size_t(j)*lda+i]==T(i==j+2?1:0));
  else CHECK(std::memcmp(&a[size_t(j)*lda+i],&before[size_t(j)*lda+i],sizeof(T))==0);
 }
}
int main() {
 CHECK(openmx_magma_mgpu_available()==1);
 for(int lower=0;lower<2;++lower) for(int vectors=0;vectors<2;++vectors) {
  success<double>(lower,vectors);success<std::complex<double>>(lower,vectors);
 }
 CHECK(init_calls==1); // one process-wide initialization across D/Z solves
 success<double>(1,1);const auto *cached=real_workspace.matrix.data();
 success<double>(1,1);CHECK(real_workspace.matrix.data()==cached);
 CHECK(complex_workspace.matrix.capacity()==0);
 openmx_magma_mgpu_release_cache();CHECK(real_workspace.matrix.capacity()==0 && init_count==1);
 setenv("OPENMX_MAGMA_HOST_CACHE_MB","0",1);success<double>(1,1);
 CHECK(real_workspace.matrix.capacity()==0 && complex_workspace.matrix.capacity()==0);
 setenv("OPENMX_MAGMA_HOST_CACHE_MB","1",1);success<double>(1,1,400);
 CHECK(real_workspace.matrix.capacity()==0 && complex_workspace.matrix.capacity()==0);
 unsetenv("OPENMX_MAGMA_HOST_CACHE_MB");
 const int n=257;std::vector<double>a(n*n),w(4,-55);
 for(int i=0;i<n;++i) a[i*n+i]=i;
 const auto before=a;
 auto attempt=[&](int status) {
  CHECK(openmx_magma_mgpu_dsyevdx_host(2,n,4,a.data(),n,w.data())==status);
  CHECK(a==before && w==std::vector<double>(4,-55) && device==3 && init_count<=1);
  CHECK(openmx_magma_mgpu_last_error()[0]!='\0');
 };
 solve_fail=7;attempt(OPENMX_MAGMA_LIBRARY_ERROR);solve_fail=0;
 corrupt=1;attempt(OPENMX_MAGMA_VALIDATION_ERROR);corrupt=0;
 magma_library.release();CHECK(init_count==0);
 init_fail=1;attempt(OPENMX_MAGMA_LIBRARY_ERROR);CHECK(init_count==0);init_fail=0;
 sync_fail=1;attempt(OPENMX_MAGMA_CUDA_ERROR);sync_fail=0;
 query_fail=1;attempt(OPENMX_MAGMA_CUDA_ERROR);query_fail=0;
 visible=1;attempt(OPENMX_MAGMA_UNAVAILABLE);visible=4;
 free_bytes=1;attempt(OPENMX_MAGMA_RESOURCES);free_bytes=size_t(40)<<30;
 query_bad=1;attempt(OPENMX_MAGMA_RESOURCES);query_bad=0;
 openmx_magma_mgpu_release_cache();allocation_failure=0;attempt(OPENMX_MAGMA_RESOURCES);
 const int calls_before=calls;
 a[1]=std::numeric_limits<double>::quiet_NaN();const auto bad_input=a;
 CHECK(openmx_magma_mgpu_dsyevdx_host(2,n,4,a.data(),n,w.data())==OPENMX_MAGMA_INVALID);
 CHECK(std::memcmp(a.data(),bad_input.data(),a.size()*sizeof(double))==0 && calls==calls_before);
 a=before;
 CHECK(openmx_magma_mgpu_dsyevdx_host(9,n,4,a.data(),n,w.data())==OPENMX_MAGMA_INVALID);
 CHECK(openmx_magma_mgpu_dsyevdx_host(2,2147483647,4,a.data(),2147483647,w.data())==OPENMX_MAGMA_RESOURCES);
 CHECK(openmx_magma_mgpu_dsyevdx_host(2,0,0,nullptr,0,nullptr)==0);
 double scalar=0,eigenvalue=-55;
 CHECK(openmx_magma_mgpu_dsyevdx_host(2,1,1,&scalar,1,&eigenvalue)==0);
 CHECK(scalar==1 && eigenvalue==0 && device==3);
 CHECK(init_count==1);
 openmx_magma_mgpu_release_cache();magma_library.release();CHECK(init_count==0);
 std::printf("PASS: MAGMA host ownership/error/policy checks (%d assertions)\n",checks);
}
#else
int main() {
 double a=9,w=-8;
 CHECK(openmx_magma_mgpu_available()==0);
 CHECK(openmx_magma_mgpu_dsyevdx_host(2,1,1,&a,1,&w)==OPENMX_MAGMA_UNAVAILABLE);
 CHECK(a==9 && w==-8 && openmx_magma_mgpu_last_error()[0]);
 CHECK(openmx_magma_mgpu_dsyevdx_host(2,0,0,nullptr,0,nullptr)==0);
 std::puts("PASS: dependency-free MAGMA unavailable stub");
}
#endif
'''
with tempfile.TemporaryDirectory(prefix="openmx-magma-policy-") as folder:
    tmp = Path(folder)
    (tmp / "cuda_runtime_api.h").write_text(cuda)
    (tmp / "magma_v2.h").write_text(magma)
    (tmp / "test.cpp").write_text(main)
    for enabled in (False, True):
        command = shlex.split(os.environ.get("CXX", "c++")) + ["-std=c++17", "-O1", "-Wall", "-Wextra",
            "-fsanitize=undefined", "-fno-sanitize-recover=all", "-I"+str(tmp), "-I"+str(root/"source")]
        if enabled:
            command.append("-DOPENMX_HAVE_MAGMA")
        binary = tmp / ("enabled" if enabled else "stub")
        subprocess.run(command + [str(tmp/"test.cpp"), "-o", str(binary)], check=True)
        subprocess.run([str(binary)], check=True)

#pragma once
#include "openmx_core_native_resources.h"
#include "../../allocation_faults.hpp"
#include <array>
#include <cstdlib>
#include <stdexcept>

// Compile the production adapters against an independent SDK model. The
// portable library still owns the actual registration and lifetime logic.
#define OPENMX_GPU_BACKEND_H
#define OPENMX_GPU_RUNTIME_H
#define OPENMX_GPU_BLAS_H
#define OPENMX_GPU_SOLVER_H
#ifndef OPENMX_GPU_HIP
#define OPENMX_GPU_HIP 0
#endif
#define OPENMX_GPU_CUDA (!OPENMX_GPU_HIP)
using OpenMXGpuError=int;
using OpenMXGpuStream=void*;
using OpenMXBlasStatus=int;
using OpenMXBlasHandle=void*;
using OpenMXSolverStatus=int;
using OpenMXSolverHandle=void*;
constexpr int OPENMX_GPU_SUCCESS=0,OPENMX_GPU_OUT_OF_MEMORY=2;
constexpr int OPENMX_BLAS_SUCCESS=0,OPENMX_BLAS_ALLOC_FAILED=7,OPENMX_BLAS_EXECUTION_FAILED=13;
constexpr int OPENMX_SOLVER_SUCCESS=0,OPENMX_SOLVER_ALLOC_FAILED=23,OPENMX_SOLVER_EXECUTION_FAILED=31;
namespace {
void check(bool ok,const char* message) { if(!ok)throw std::runtime_error(message); }
struct Allocation { void* pointer=nullptr;int device=0,kind=0; };
std::array<Allocation,128> records;
std::array<int,512> events;
int current_device=0,acc_device=0,queries=0,live=0,event_count=0,fail_status=0;
void record(int event) { check(event_count<int(events.size()),"SDK event log overflow");events[event_count++]=event; }
int create(void** out,size_t bytes,int kind) {
  if(fail_status) { *out=nullptr;return fail_status; }
  for(auto& a:records)if(!a.pointer) {
    a={std::malloc(bytes?bytes:1),current_device,kind};if(!a.pointer)return 2;
    *out=a.pointer;live++;record(100+kind);return 0;
  }
  throw std::runtime_error("SDK model exhausted");
}
int release(void* pointer,int kind) {
  if(!pointer) { record(0);return 0; }
  for(auto& a:records)if(a.pointer==pointer) {
    check(a.device==current_device,"resource freed on a different device");
    check(a.kind==kind,"incorrect SDK deallocator");
    std::free(a.pointer);a.pointer=nullptr;live--;record(200+kind);return 0;
  }
  throw std::runtime_error("SDK object released twice");
}
}
static int OpenMXGpuGetDevice(int* device) { queries++;*device=current_device;return 0; }
static int OpenMXGpuSetDevice(int device) { current_device=device;record(300+device);return 0; }
static int OpenMXGpuMalloc(void** out,size_t bytes) { return create(out,bytes,1); }
static int OpenMXGpuFree(void* pointer) { return release(pointer,1); }
static int OpenMXGpuHostAlloc(void** out,size_t bytes,unsigned) { return create(out,bytes,2); }
static int OpenMXGpuFreeHost(void* pointer) { return release(pointer,2); }
static int OpenMXGpuStreamCreateWithFlags(void** out,unsigned) { return create(out,1,6); }
static int OpenMXGpuStreamSynchronize(void*) { record(400);return 0; }
static int OpenMXGpuStreamDestroy(void* pointer) { return release(pointer,6); }
static int OpenMXBlasCreate(void** out) { return create(out,1,5); }
static int OpenMXBlasDestroy(void* pointer) { return release(pointer,5); }
static int OpenMXSolverCreate(void** out) { return create(out,1,4); }
static int OpenMXSolverDestroy(void* pointer) { return release(pointer,4); }
static int openmx_test_acc_device() { return acc_device; }
static void openmx_test_acc_set(int device) { acc_device=device;current_device=device; }
static void* openmx_test_acc_malloc(size_t bytes) { void* result=nullptr;(void)create(&result,bytes,3);return result; }
static void openmx_test_acc_free(void* pointer) { check(!release(pointer,3),"ACC release"); }
#include "openmx_native_resources.h"

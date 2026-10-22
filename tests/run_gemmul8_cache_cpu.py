#!/usr/bin/env python3
"""Test the actual CUDA/HIP retained-cache code with independent SDK storage."""
from pathlib import Path
import os
import resource
import shlex
import subprocess
import tempfile

ROOT=Path(__file__).resolve().parents[1]
from fixtures.gemmul8_model import HEADERS as headers
runtime=r'''
#pragma once
#include "fixtures/gpu_views_model.hpp"
using cudaError_t=int;
using cudaStream_t=void*;
using OpenMXGpuStream=void*;
using OpenMXGpuDataType=int;
constexpr int cudaSuccess=0,cudaErrorInvalidValue=11,cudaErrorMemoryAllocation=2;
constexpr int cudaDevAttrSingleToDoublePrecisionPerfRatio=1,cudaDevAttrComputeCapabilityMajor=2,
    cudaDevAttrComputeCapabilityMinor=3,cudaDevAttrMultiProcessorCount=4;
inline void *mock_stream=nullptr;
inline size_t mock_free=size_t(40)<<30,mock_total=size_t(40)<<30;
inline int cudaGetDevice(int *out){return OpenMXGpuGetDevice(out);}
inline int cudaSetDevice(int device){return OpenMXGpuSetDevice(device);}
inline int cudaMalloc(void **out,size_t bytes){return OpenMXGpuMalloc(out,bytes);}
inline int cudaFree(void *pointer){return OpenMXGpuFree(pointer);}
inline int cudaGetLastError(){return OpenMXGpuGetLastError();}
inline const char *cudaGetErrorString(int){return "SDK cache model error";}
inline int cudaGetDeviceCount(int *out){*out=4;return 0;}
inline int cudaDeviceGetAttribute(int *out,int attr,int){*out=attr==1?64:attr==2?8:attr==3?9:128;return 0;}
inline int cudaMemGetInfo(size_t *free,size_t *total){*free=mock_free;*total=mock_total;return 0;}
inline int cudaStreamSynchronize(void*){return 0;}
'''

def bounded_stack():
    _,hard=resource.getrlimit(resource.RLIMIT_STACK)
    resource.setrlimit(resource.RLIMIT_STACK,(8*1024*1024,hard))

with tempfile.TemporaryDirectory(prefix='openmx-gemmul8-cache-') as directory:
    base=Path(directory)
    for backend in ['CUDA','HIP']:
        folder=base/backend.lower();folder.mkdir()
        # Put the real common headers beside the fake SDK, including C ABI
        # forwarding headers whose definitions live in core/include.
        for header in [*(ROOT/'source').glob('*.h'),*(ROOT/'source').glob('*.hpp')]:
            canonical=ROOT/'core/include'/header.name
            (folder/header.name).write_bytes((canonical if canonical.is_file() else header).read_bytes())
        if backend=='CUDA':
            for name,content in headers.items():(folder/name).write_text(content)
            (folder/'cuda_runtime.h').write_text(runtime)
            text=headers['cublas_v2.h'].replace('*s=nullptr','*s=mock_stream')
            (folder/'cublas_v2.h').write_text(text)
            bridge=ROOT/'source/gemmul8_bridge.cu'
        else:
            (folder/'hip_runtime_compat.h').write_text(runtime.replace('cuda','hip'))
            (folder/'hipblas_compat.h').write_text(headers['cublas_v2.h'].replace('cuda_runtime.h','hip_runtime_compat.h')
                .replace('cuComplex.h','hip_complex_compat.h').replace('cuDoubleComplex','hipDoubleComplex')
                .replace('cublas','hipblas').replace('CUBLAS','HIPBLAS').replace('*s=nullptr','*s=mock_stream'))
            (folder/'hip_complex_compat.h').write_text(headers['cuComplex.h'].replace('cuDoubleComplex','hipDoubleComplex'))
            (folder/'hipsolver_compat.h').write_text(headers['cusolverDn.h'].replace('cusolver','hipsolver')
                .replace('cudaDataType','hipDataType').replace('cublasFillMode_t','hipblasFillMode_t'))
            (folder/'openmx_gpu_arch_policy.h').write_text('#pragma once\ninline bool openmx_gpu_device_is_mi300a(){return false;}\n')
            (folder/'gemmul8.hpp').write_text(headers['gemmul8.hpp'])
            bridge=ROOT/'source/backends/hip/src/gemmul8_bridge.cu'
        (folder/'bridge_under_test.cu').write_bytes(bridge.read_bytes())
        binary=folder/'check'
        command=shlex.split(os.environ.get('CXX','c++'))+['-std=c++17','-O1','-g','-pthread',
            '-fsanitize=address,undefined','-fno-sanitize-recover=all','-fno-pie','-no-pie',
            '-I'+str(folder),'-I'+str(ROOT/'tests'),'-I'+str(ROOT/'core/tests')]
        if backend=='HIP':command+=['-D__HIP_PLATFORM_AMD__=1']
        command+=[str(ROOT/'tests/gemmul8_cache_harness.cpp'),str(ROOT/'source/openmx_env.c'),
                  '-I'+str(ROOT/'source'),str(ROOT/'core/tests/allocation_faults.cpp'),'-o',str(binary)]
        subprocess.run(command,check=True)
        environment={key:value for key,value in os.environ.items() if not key.startswith(('OPENMX_','GEMMUL8_'))}
        print('Backend:',backend,flush=True)
        subprocess.run([str(binary)],env=environment,check=True,timeout=120,preexec_fn=bounded_stack)

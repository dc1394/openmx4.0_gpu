#!/usr/bin/env python3
"""Check production dense-kernel indexing against the long-double oracle.

CUDA grid-stride loops run as one emulated thread; transfers/views and all
callback phases remain production code. ASan/UBSan cover padded/offset
storage and rollback. This does not validate GPU streams or OpenACC binding;
run_gpu_dense_workers_smoke.sh covers those on four physical GPUs.
"""
from pathlib import Path
import ast
import os
import re
import shlex
import subprocess
import tempfile

root=Path(__file__).resolve().parents[1]
# Reuse declarations without importing/running the GEMM test module.
tree=ast.parse((root/'tests/run_mgpu_gemm_cpu.py').read_text())
constants={node.targets[0].id:ast.literal_eval(node.value) for node in tree.body
           if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name)
           and node.targets[0].id in ('cuda','complex_header')}
extra=r'''
#define __host__
#define __device__
#define __global__
struct MockDim {unsigned x;};
inline constexpr MockDim blockIdx{0},threadIdx{0},blockDim{1},gridDim{1};
inline double atomicAdd(double *p,double x){double old=*p;*p+=x;return old;}
inline int cudaMemset(void *p,int x,size_t n){std::memset(p,x,n);return 0;}
inline int cudaMemcpyPeer(void *d,int,const void *s,int,size_t n){return cudaMemcpy(d,s,n,cudaMemcpyDeviceToDevice);}
'''
acc=r'''
#pragma once
#include "cuda_runtime.h"
constexpr int acc_device_nvidia=5;
inline int mock_acc_device=0;
inline void acc_set_device_num(int d,int){mock_acc_device=d;cudaSetDevice(d);}
inline int acc_get_device_num(int){return mock_acc_device;}
'''
workers=r'''
#include "openmx_gpu_workers.h"
#include "cuda_runtime.h"
#include "openacc.h"
#include <algorithm>
extern "C" {
int probe_gpus=4,probe_first=0,probe_count=4,probe_remote_copies=0,probe_capacity_error=0;
void openmx_gpu_workers_configure(int,int,int){}
void openmx_gpu_workers_scope(OpenMXGpuWorkers *p,size_t n){
    p->items=n;p->count=int(std::min(n,size_t(probe_count)));
    for(int i=0;i<p->count;i++)p->devices[i]=probe_first+i;
}
int openmx_gpu_workers_capacity(OpenMXGpuWorkers *p,size_t n,int kind,OpenMXGpuWorkerCost cost,void *ctx){
    openmx_gpu_workers_scope(p,n);for(int i=0;i<p->count;i++)if(cost(ctx,i,p->count)==SIZE_MAX)return 0;return 1;
}
int openmx_gpu_workers_capacity_device(OpenMXGpuWorkers *p,size_t n,int,OpenMXGpuWorkerDeviceCost cost,void *ctx){
    if(probe_capacity_error){mock_pending_error=probe_capacity_error;return 0;}
    openmx_gpu_workers_scope(p,n);
    for(int i=0;i<p->count;i++) {
        const size_t bytes=cost(ctx,i,p->count,p->devices[i]);
        if(bytes==SIZE_MAX)return 0;
        if(p->count==1 && p->devices[i]!=mock_device && bytes>cost(ctx,i,p->count,mock_device))
            ++probe_remote_copies;
    }
    return 1;
}
int openmx_gpu_workers_run(const OpenMXGpuWorkers *p,int (*callback)(void*,int,int),void *data,const char*){
    const int saved=mock_device,saved_acc=mock_acc_device;int success=1,started[64]={};
    for(int phase=0;phase<=3;phase++){
        if(!success && phase!=3)continue;
        int ok=1;
        for(int i=0;i<p->count;i++){
            if(phase==3 && !started[i])continue;
            cudaSetDevice(p->devices[i]);mock_acc_device=p->devices[i];
            if(phase==0)started[i]=1;
            ok &= callback(data,i,phase);
        }
        success &= ok;
    }
    cudaSetDevice(saved);mock_acc_device=saved_acc;return success;
}
int openmx_gpu_workers_run_prepared(const OpenMXGpuWorkers *pool,int (*callback)(void*,int,int),void *context,const char *label){
    return openmx_gpu_workers_run(pool,callback,context,label);
}
}
'''
with tempfile.TemporaryDirectory(prefix='openmx-dense-index-') as directory:
    tmp=Path(directory)
    (tmp/'cuda_runtime.h').write_text(constants['cuda']+extra)
    (tmp/'cuda_runtime_api.h').write_text('#include "cuda_runtime.h"\n')
    (tmp/'cuComplex.h').write_text(constants['complex_header'])
    (tmp/'openacc.h').write_text(acc)
    (tmp/'workers.cpp').write_text(workers)
    sources=[]
    for name in ('openmx_gpu_transform','openmx_gpu_dm'):
        text=(root/f'source/{name}.cu').read_text()
        text,count=re.subn(r'<<<.*?>>>','',text,flags=re.S)
        if not count:raise RuntimeError(f'No kernel launches found in {name}')
        target=tmp/(name+'.cpp');target.write_text(text);sources.append(target)
    fixture_text=(root/'tests/gpu_dense_workers_smoke.cu').read_text()
    fixture_text=fixture_text.replace('probe_gpus,probe_first,probe_count;',
        'probe_gpus,probe_first,probe_count,probe_remote_copies,probe_capacity_error;')
    marker='    std::printf("PASS: %d dense worker comparisons;'
    remote='''    probe_count=1;probe_first=2;
    check(cudaSetDevice(0));acc_set_device_num(0,acc_device_nvidia);
    setenv("OPENMX_GPU_WORKER_COPY","1",1);
    density(OPENMX_GPU_DM_REAL,false,false,false,false);
    require(probe_remote_copies>0,"remote single worker must budget the full vector copy");
    const double vector=2,occupation=0.5;
    const int basis=0;
    double output=19,*outputs[10]={&output};
    auto dm=[&](bool cpu) {
        auto function=cpu?openmx_gpu_dm_entries:openmx_gpu_dm_try_entries;
        return function(OPENMX_GPU_DM_REAL,1,1,1,1,&vector,1,&basis,&basis,
            nullptr,0,nullptr,nullptr,&occupation,nullptr,nullptr,outputs,1);
    };
    for(bool cpu:{false,true})for(int error:{cudaErrorMemoryAllocation,700}) {
        mock_pending_error=error;
        require(!dm(cpu),"DM must reject a pending caller error");
        require(mock_pending_error==error && output==19,
            "DM must preserve the caller error and output without CPU recovery");
        mock_pending_error=0;
    }
    probe_capacity_error=700;
    require(!dm(true) && mock_pending_error==700 && output==19,
        "DM must preserve a fatal preparation error without CPU recovery");
    mock_pending_error=0;
    probe_capacity_error=cudaErrorMemoryAllocation;
    require(dm(true) && mock_pending_error==0 && output==21,
        "DM may consume its own allocation error and recover on the CPU");
    probe_capacity_error=0;
'''
    assert marker in fixture_text
    fixture_text=fixture_text.replace(marker,remote+marker)
    fixture=tmp/'fixture.cpp';fixture.write_text(fixture_text)
    sources.extend((root/'source/openmx_gpu_views.c',tmp/'workers.cpp',fixture))
    command=shlex.split(os.environ.get('CXX','c++'))+['-std=c++17','-O1','-g','-fsanitize=address,undefined',
        '-fno-omit-frame-pointer','-I'+str(tmp),'-I'+str(root/'source'),'-x','c++']
    subprocess.run(command+list(map(str,sources))+[str(root/'source/openmx_env.c'),'-o',str(tmp/'check')],check=True)
    subprocess.run([str(tmp/'check')],check=True)

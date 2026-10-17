#!/usr/bin/env python3
"""Check production dense-kernel indexing against the long-double oracle.

HIP grid-stride loops run as one emulated thread; transfers/views and all
callback phases remain production code. ASan/UBSan cover padded/offset
storage and rollback. This does not validate GPU streams or OpenMP binding;
run_gpu_dense_workers_smoke.sh covers those on four physical GPUs.
"""
from pathlib import Path
import ast
import os
import re
import shlex
import subprocess
import tempfile
from mock_hip_sources import stage

root=Path(__file__).resolve().parents[1]
# Reuse declarations without importing/running the GEMM test module.
tree=ast.parse((root/'tests/run_mgpu_gemm_cpu.py').read_text())
constants={node.targets[0].id:ast.literal_eval(node.value) for node in tree.body
           if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name)
           and node.targets[0].id in ('hip','complex_header')}
extra=r'''
#define __host__
#define __device__
#define __global__
struct MockDim {unsigned x;};
inline constexpr MockDim blockIdx{0},threadIdx{0},blockDim{1},gridDim{1};
inline double atomicAdd(double *p,double x){double old=*p;*p+=x;return old;}
inline int hipMemset(void *p,int x,size_t n){std::memset(p,x,n);return 0;}
inline int hipMemcpyPeer(void *d,int,const void *s,int,size_t n){return hipMemcpy(d,s,n,hipMemcpyDeviceToDevice);}
'''
omp=r'''
#pragma once
#include "hip_runtime_compat.h"
inline int mock_omp_device=0;
inline int omp_get_default_device(){return mock_omp_device;}
inline void omp_set_default_device(int d){mock_omp_device=d;}
'''
workers=r'''
#include "openmx_gpu_workers.h"
#include "hip_runtime_compat.h"
#include <omp.h>
#include <algorithm>
extern "C" {
int probe_gpus=4,probe_first=0,probe_count=4;
void openmx_gpu_workers_configure(int,int,int){}
void openmx_gpu_workers_scope(OpenMXGpuWorkers *p,size_t n){
    p->items=n;p->count=int(std::min(n,size_t(probe_count)));
    for(int i=0;i<p->count;i++)p->devices[i]=probe_first+i;
}
int openmx_gpu_workers_capacity(OpenMXGpuWorkers *p,size_t n,int kind,OpenMXGpuWorkerCost cost,void *ctx){
    openmx_gpu_workers_scope(p,n);for(int i=0;i<p->count;i++)if(cost(ctx,i,p->count)==SIZE_MAX)return 0;return 1;
}
int openmx_gpu_workers_run(const OpenMXGpuWorkers *p,int (*callback)(void*,int,int),void *data,const char*){
    const int saved=mock_device,saved_omp=mock_omp_device;int success=1,started[64]={};
    for(int phase=0;phase<=3;phase++){
        if(!success && phase!=3)continue;
        int ok=1;
        for(int i=0;i<p->count;i++){
            if(phase==3 && !started[i])continue;
            hipSetDevice(p->devices[i]);mock_omp_device=p->devices[i];
            if(phase==0)started[i]=1;
            ok &= callback(data,i,phase);
        }
        success &= ok;
    }
    hipSetDevice(saved);mock_omp_device=saved_omp;return success;
}
}
'''
with tempfile.TemporaryDirectory(prefix='openmx-dense-index-') as directory:
    tmp=Path(directory)
    (tmp/'hip_runtime_compat.h').write_text(constants['hip']+extra)
    (tmp/'hip_runtime_api.h').write_text('#include "hip_runtime_compat.h"\n')
    (tmp/'hip_complex_compat.h').write_text(constants['complex_header'])
    (tmp/'omp.h').write_text(omp)
    (tmp/'workers.cpp').write_text(workers)
    stage(root,tmp,"openmx_gpu_views.c")
    sources=[]
    for name in ('openmx_gpu_transform','openmx_gpu_dm'):
        text=(root/f'source/{name}.cu').read_text()
        text,count=re.subn(r'<<<.*?>>>','',text,flags=re.S)
        if not count:raise RuntimeError(f'No kernel launches found in {name}')
        target=tmp/(name+'.cpp');target.write_text(text);sources.append(target)
    fixture=tmp/'fixture.cpp';fixture.write_text((root/'tests/gpu_dense_workers_smoke.cu').read_text())
    sources.extend((tmp/'openmx_gpu_views.c',tmp/'workers.cpp',fixture))
    command=shlex.split(os.environ.get('CXX','c++'))+['-std=c++17','-O1','-g','-fsanitize=address,undefined',
        '-fno-omit-frame-pointer','-I'+str(tmp),'-I'+str(root/'source'),'-x','c++']
    subprocess.run(command+list(map(str,sources))+['-o',str(tmp/'check')],check=True)
    subprocess.run([str(tmp/'check')],check=True)

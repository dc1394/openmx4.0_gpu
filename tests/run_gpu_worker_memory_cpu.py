#!/usr/bin/env python3
"""Exercise native GPU owners and actual worker failure paths with ASan/UBSan."""
from pathlib import Path
import os
import re
import resource
import shlex
import subprocess
import tempfile

ROOT=Path(__file__).resolve().parents[1]

def bounded_stack():
    _,hard=resource.getrlimit(resource.RLIMIT_STACK)
    resource.setrlimit(resource.RLIMIT_STACK,(8*1024*1024,hard))

with tempfile.TemporaryDirectory(prefix='openmx-worker-memory-') as directory:
    folder=Path(directory);sources=[]
    for name in ['openmx_gpu_dm','openmx_gpu_transform']:
        code,n=re.subn(r'<<<.*?>>>','',(ROOT/'source'/f'{name}.cu').read_text(),flags=re.S)
        if not n:raise RuntimeError('No production kernel launches in '+name)
        path=folder/(name+'.cpp');path.write_text(code);sources.append(str(path))
    for backend in ['CUDA','HIP']:
        binary=folder/backend.lower()
        command=shlex.split(os.environ.get('CXX','c++'))+['-std=c++17','-O1','-g','-fopenmp',
            '-fsanitize=address,undefined','-fno-sanitize-recover=all','-fno-pie','-no-pie',
            '-I'+str(ROOT/'source'),'-I'+str(ROOT/'tests'),'-I'+str(ROOT/'core/tests'),
            '-include',str(ROOT/'tests/fixtures/gpu_worker_model.hpp')]
        if backend=='HIP':command+=['-D__HIP_PLATFORM_AMD__=1']
        command+=sources+[str(ROOT/'tests/gpu_worker_memory_harness.cpp'),str(ROOT/'source/openmx_gpu_views.cpp'),
            str(ROOT/'source/openmx_env.c'),str(ROOT/'core/tests/allocation_faults.cpp'),'-o',str(binary)]
        subprocess.run(command,check=True)
        print('Backend:',backend,flush=True)
        subprocess.run([str(binary)],check=True,timeout=120,preexec_fn=bounded_stack)

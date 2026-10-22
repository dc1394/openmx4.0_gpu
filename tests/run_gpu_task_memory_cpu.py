#!/usr/bin/env python3
"""Exercise the actual task worker and owners with independent SDK storage."""
from pathlib import Path
import os
import resource
import shlex
import subprocess
import tempfile

ROOT=Path(__file__).resolve().parents[1]

def bounded_stack():
    _,hard=resource.getrlimit(resource.RLIMIT_STACK)
    resource.setrlimit(resource.RLIMIT_STACK,(8*1024*1024,hard))

with tempfile.TemporaryDirectory(prefix='openmx-task-memory-') as directory:
    folder=Path(directory)
    (folder/'mpi.h').write_text('typedef int MPI_Comm;\n#define MPI_COMM_NULL (-1)\n')
    for backend in ['CUDA','HIP']:
        binary=folder/backend.lower()
        command=shlex.split(os.environ.get('CXX','c++'))+['-std=c++17','-O1','-g',
            '-fsanitize=address,undefined','-fno-sanitize-recover=all','-fno-pie','-no-pie',
            '-I'+str(folder),'-I'+str(ROOT/'source'),'-I'+str(ROOT/'tests'),
            '-Wl,--wrap=malloc,--wrap=free']
        if backend=='HIP':command+=['-D__HIP_PLATFORM_AMD__=1']
        command+=[str(ROOT/'tests/gpu_task_memory_harness.cpp'),'-o',str(binary)]
        subprocess.run(command,check=True)
        print('Backend:',backend,flush=True)
        subprocess.run([str(binary)],check=True,timeout=120,preexec_fn=bounded_stack)

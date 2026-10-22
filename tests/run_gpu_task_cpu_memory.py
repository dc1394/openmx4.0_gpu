#!/usr/bin/env python3
"""Verify actual SDK-free CPU recovery, its allocation policy and RAII."""
from pathlib import Path
import os
import resource
import shlex
import signal
import subprocess
import tempfile

ROOT=Path(__file__).resolve().parents[1]

def bounded_process():
    _,hard=resource.getrlimit(resource.RLIMIT_STACK)
    resource.setrlimit(resource.RLIMIT_STACK,(8*1024*1024,hard))
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))

with tempfile.TemporaryDirectory(prefix='openmx-task-cpu-memory-') as directory:
    folder=Path(directory);binary=folder/'check'
    (folder/'mpi.h').write_text('typedef int MPI_Comm;\n#define MPI_COMM_NULL (-1)\n')
    command=shlex.split(os.environ.get('CXX','c++'))+['-std=c++17','-O1','-g',
        '-fsanitize=address,undefined','-fno-sanitize-recover=all','-fno-pie','-no-pie',
        '-I'+str(folder),'-I'+str(ROOT/'source'),'-Wl,--wrap=malloc,--wrap=free',
        str(ROOT/'source/openmx_gpu_task_cpu.cpp'),str(ROOT/'tests/gpu_task_cpu_memory_harness.cpp'),
        '-o',str(binary)]
    subprocess.run(command,check=True)
    subprocess.run([str(binary)],check=True,timeout=120,preexec_fn=bounded_process)
    for mode in ['overflow','oom']:
        result=subprocess.run([str(binary),mode],text=True,capture_output=True,timeout=30,preexec_fn=bounded_process)
        assert result.returncode==-signal.SIGABRT,(mode,result.returncode,result.stderr)
        assert result.stderr=='OpenMX: cannot allocate a single CPU range workspace.\n',result.stderr
    print('PASS: saturated CPU workspace makes no allocation; single-item OOM preserves the fatal diagnostic')

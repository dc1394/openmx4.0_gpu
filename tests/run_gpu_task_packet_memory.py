#!/usr/bin/env python3
"""Check real C++ task packets against a C wire fixture under ASan/UBSan."""
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

with tempfile.TemporaryDirectory(prefix='openmx-task-packet-memory-') as directory:
    folder=Path(directory);binary=folder/'check';wire=folder/'wire.o'
    (folder/'mpi.h').write_text('typedef int MPI_Comm;\n#define MPI_COMM_NULL (-1)\n')
    flags=['-O1','-g','-fsanitize=address,undefined','-fno-sanitize-recover=all','-fno-pie']
    subprocess.run(shlex.split(os.environ.get('CC','cc'))+['-std=c11',*flags,'-c',
        str(ROOT/'tests/gpu_task_packet_wire.c'),'-o',str(wire)],check=True)
    subprocess.run(shlex.split(os.environ.get('CXX','c++'))+['-std=c++17',*flags,'-no-pie',
        '-I'+str(folder),'-I'+str(ROOT/'source'),'-Wl,--wrap=calloc,--wrap=free',
        str(ROOT/'source/openmx_gpu_task_packet.cpp'),str(ROOT/'source/openmx_gpu_task_cpu.cpp'),
        str(ROOT/'tests/gpu_task_packet_memory_harness.cpp'),str(wire),'-o',str(binary)],check=True)
    subprocess.run([str(binary)],check=True,timeout=120,preexec_fn=bounded_stack)

#!/usr/bin/env python3
"""Verify C++ vector packet ownership and independent C wire fixtures."""
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

with tempfile.TemporaryDirectory(prefix='openmx-vector-packet-memory-') as directory:
    folder=Path(directory);wire=folder/'wire.o';binary=folder/'check'
    (folder/'mpi.h').write_text('typedef int MPI_Comm;\n')
    flags=['-O1','-g','-fsanitize=address,undefined','-fno-sanitize-recover=all','-fno-pie']
    subprocess.run(shlex.split(os.environ.get('CC','cc'))+['-std=c11',*flags,'-c',
        str(ROOT/'tests/gpu_vector_packet_wire.c'),'-o',str(wire)],check=True)
    subprocess.run(shlex.split(os.environ.get('CXX','c++'))+['-std=c++17',*flags,'-no-pie',
        '-I'+str(folder),'-I'+str(ROOT/'source'),'-Wl,--wrap=malloc,--wrap=calloc,--wrap=free',
        str(ROOT/'source/openmx_gpu_vector_packet.cpp'),str(ROOT/'tests/gpu_vector_packet_memory_harness.cpp'),
        str(wire),'-o',str(binary)],check=True)
    subprocess.run([str(binary)],check=True,timeout=120,preexec_fn=bounded_stack)

#!/usr/bin/env python3
"""Real six-rank MPI checks of C/C++ packet ownership and CPU recovery."""
from pathlib import Path
import argparse
import os
import shlex
import subprocess
import tempfile

ROOT=Path(__file__).resolve().parents[1]

def run_packet_mpi(kind):
    assert kind in ("task","vector")
    parser=argparse.ArgumentParser(description="Real six-rank "+kind+" packet ownership and recovery")
    parser.add_argument('--sanitize',action='store_true',help='enable UBSan with a supporting MPI compiler')
    args=parser.parse_args()
    cc=shlex.split(os.environ.get('MPICC','mpicc'))
    cxx=shlex.split(os.environ.get('MPICXX','mpicxx'))
    mpi=shlex.split(os.environ.get('MPIEXEC','mpirun'))
    version=subprocess.check_output(mpi+['--version'],text=True,stderr=subprocess.STDOUT)
    if 'Open MPI' in version or 'OpenRTE' in version:mpi+=['--oversubscribe']
    with tempfile.TemporaryDirectory(prefix="openmx-"+kind+"-packet-mpi-") as directory:
        folder=Path(directory);objects=[]
        flags=['-O1','-g','-I'+str(ROOT/'source')]
        if args.sanitize:flags+=['-fsanitize=undefined','-fno-sanitize-recover=all']
        for compiler,standard,source,name in [
            (cc,'c11',ROOT/('tests/gpu_'+kind+'_packet_mpi.c'),'driver'),
            *([(cxx,'c++17',ROOT/'source/openmx_gpu_task_cpu.cpp','cpu')] if kind=='task' else []),
            (cxx,'c++17',ROOT/('source/openmx_gpu_'+kind+'_packet.cpp'),'packet')]:
            obj=folder/(name+'.o');objects.append(str(obj))
            subprocess.run(compiler+['-std='+standard,*flags,'-c',str(source),'-o',str(obj)],check=True)
        binary=folder/'check'
        subprocess.run(cxx+flags+objects+['-Wl,--wrap=malloc,--wrap=calloc,--wrap=free,--wrap=openmx_gpu_'+kind+'_pack,--wrap=openmx_gpu_'+kind+'_remote',
            '-o',str(binary)],check=True)
        subprocess.run(mpi+['-np','6',str(binary)],check=True,timeout=90)

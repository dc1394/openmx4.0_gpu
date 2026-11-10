#!/usr/bin/env python3
"""Audit each MPI rank, inherit SVE128, and exec the official OpenMX binary."""
import ctypes
import os
import sys

rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
if int(os.environ['OMPI_COMM_WORLD_SIZE']) != 8:
    raise SystemExit('Expected 8 MPI ranks')
cores = sorted(os.sched_getaffinity(0))
if len(cores) != 1:
    raise SystemExit(f'Rank {rank}: expected one CPU core, got {cores}')
for key in ('OMP_NUM_THREADS', 'OMP_THREAD_LIMIT', 'OPENBLAS_NUM_THREADS'):
    if os.environ.get(key) != '1':
        raise SystemExit(f'Rank {rank}: {key} must be 1')
libc = ctypes.CDLL(None, use_errno=True)
if libc.prctl(50, 16 | (1 << 17), 0, 0, 0) < 0:  # PR_SVE_SET_VL | INHERIT
    raise SystemExit(f'Rank {rank}: SVE128 setup failed, errno={ctypes.get_errno()}')
vl = libc.prctl(51, 0, 0, 0, 0)  # PR_SVE_GET_VL
if vl < 0 or (vl & 65535) != 16:
    raise SystemExit(f'Rank {rank}: SVE vector length is not 128 bits')
print(f'[openmx-8x1] rank={rank} cpu={cores[0]} OMP=1 BLAS=1 SVE_VL=128',
      file=sys.stderr, flush=True)
binary = '/home/dc1394/openmx4.0/work/openmx'
os.execv(binary, [binary, sys.argv[1], '-nt', '1'])

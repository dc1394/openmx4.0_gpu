#!/usr/bin/env python3
"""Compile the complete SDK-free H/S cache with its real calculation owner.

The default single-rank fixture models MPI only. With --mpi-cxx/--mpiexec,
packing and lifetime checks run on real reordered/split MPI communicators.
"""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile
from native_state_test_support import compiler_command, bounded_stack, allocation_fault_source, allocation_fault_environment

ROOT = Path(__file__).resolve().parents[1]
MPI_MODEL = r'''
#ifndef PACKED_TEST_MPI_H
#define PACKED_TEST_MPI_H
#define PACKING_MPI_MODEL 1
#include <string.h>
#include <stdlib.h>
typedef int MPI_Comm;
#define MPI_COMM_WORLD 1
#define MPI_COMM_NULL 0
#define MPI_UNDEFINED -32766
#define MPI_INT 1
#define MPI_DOUBLE 2
#define MPI_MAX 1
static inline int MPI_Init(int *argc,char ***argv){(void)argc;(void)argv;return 0;}
static inline int MPI_Finalize(void){return 0;}
static inline int MPI_Comm_rank(MPI_Comm c,int *r){(void)c;*r=0;return 0;}
static inline int MPI_Comm_size(MPI_Comm c,int *n){(void)c;*n=1;return 0;}
static inline int MPI_Comm_split(MPI_Comm c,int color,int key,MPI_Comm *out){(void)c;(void)key;*out=color==MPI_UNDEFINED?MPI_COMM_NULL:2;return 0;}
static inline int MPI_Comm_free(MPI_Comm *c){*c=MPI_COMM_NULL;return 0;}
#ifdef __cplusplus
extern "C" {
#endif
void packed_test_abort(int);
#ifdef __cplusplus
}
#endif
static inline int MPI_Abort(MPI_Comm c,int status){(void)c;packed_test_abort(status);return 0;}
static inline size_t mpi_bytes(int type){return type==MPI_INT?sizeof(int):sizeof(double);}
static inline int MPI_Allgather(const void *s,int n,int t,void *d,int m,int u,MPI_Comm c){(void)m;(void)u;(void)c;if(n)memcpy(d,s,n*mpi_bytes(t));return 0;}
static inline int MPI_Allreduce(const void *s,void *d,int n,int t,int op,MPI_Comm c){(void)op;(void)c;if(n)memcpy(d,s,n*mpi_bytes(t));return 0;}
static inline int MPI_Gatherv(const void *s,int n,int t,void *d,const int *counts,const int *displs,int u,int root,MPI_Comm c){(void)counts;(void)root;(void)c;if(n)memcpy((char*)d+displs[0]*mpi_bytes(u),s,n*mpi_bytes(t));return 0;}
static inline int MPI_Bcast(void *p,int n,int t,int root,MPI_Comm c){(void)p;(void)n;(void)t;(void)root;(void)c;return 0;}
#endif
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core-library', type=Path, required=True)
    parser.add_argument('--mpi-cxx')
    parser.add_argument('--mpiexec')
    parser.add_argument('--ranks', type=int, default=4)
    parser.add_argument('--negative-control', action='store_true')
    args = parser.parse_args()
    if bool(args.mpi_cxx) != bool(args.mpiexec):
        parser.error('--mpi-cxx and --mpiexec must be provided together')
    library = args.core_library.resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix='openmx-hamiltonian-packing-') as directory:
        folder = Path(directory)
        if not args.mpi_cxx:
            (folder/'mpi.h').write_text(MPI_MODEL)
        include = ['-I'+str(folder), '-I'+str(ROOT/'source'), '-I'+str(ROOT/'core/include'), '-I'+str(ROOT/'core/tests')]
        source = ROOT/'source/openmx_hamiltonian_cache.c'
        env = dict(os.environ)
        if args.mpi_cxx:
            env['ASAN_OPTIONS'] = 'detect_leaks=0'  # MPI retains runtime allocations.
        for negative in (False, True) if args.negative_control else (False,):
            variant = 'wrong-order' if negative else 'production'
            if negative:
                # Force the historical collinear order on the noncollinear
                # path; the independent coordinate oracle must reject it.
                source = folder/'wrong-order.c'
                text = (ROOT/'source/openmx_hamiltonian_cache.c').read_text()
                old = 'if (order_mode == SET_HAMILTONIAN_PACK_ORDER_COL)'
                assert text.count(old) == 1
                source.write_text(text.replace(old, 'if (1)'))
            obj, exe = folder/(variant+'.o'), folder/variant
            subprocess.run(compiler_command('c', args.mpi_cxx) + include +
                           ['-x', 'c', '-c', str(source), '-o', str(obj)], check=True)
            subprocess.run(compiler_command('c++', args.mpi_cxx) + include +
                           [str(ROOT/'tests/hamiltonian_packing_harness.cpp'), allocation_fault_source(folder), str(obj),
                            str(library), '-Wl,-rpath,'+str(library.parent), '-o', str(exe)], check=True)
            env.update(allocation_fault_environment(folder))
            command = ([args.mpiexec, '-np', str(args.ranks)] if args.mpiexec else []) + [str(exe)]
            run = subprocess.run(command, env=env, text=True, capture_output=True,
                                 timeout=90, preexec_fn=bounded_stack)
            if not negative:
                print(run.stdout, end='')
                if run.returncode:
                    raise RuntimeError(run.stderr)
            elif run.returncode == 0 or 'packed matrix differs from independent coordinate oracle' not in run.stderr:
                raise RuntimeError('incorrect packing order was not rejected: '+run.stdout+run.stderr)
            else:
                print('PASS: independent oracle rejects noncollinear packing-order regression')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Exercise the production retry/MPI failure boundary without a GPU SDK."""
from pathlib import Path
import os
import shlex
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
runtime = r'''
#pragma once
typedef int OpenMXGpuError;
#define OPENMX_GPU_SUCCESS 0
#define OPENMX_GPU_OUT_OF_MEMORY 2
#define OPENMX_GPU_RUNTIME_NAME "test runtime"
int OpenMXGpuPeekAtLastError(void);
int OpenMXGpuGetLastError(void);
const char *OpenMXGpuGetErrorString(int);
'''
mpi = r'''
#pragma once
#define MPI_COMM_WORLD 0
int MPI_Initialized(int*);
int MPI_Finalized(int*);
int MPI_Abort(int,int);
'''
test = r'''
#define _POSIX_C_SOURCE 200809L
#include "openmx_gpu_error.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static int attempts, waits, cleared, pending, initialized=1, finalized, scenario;
static long milliseconds;
int OpenMXGpuPeekAtLastError(void) { return pending; }
int OpenMXGpuGetLastError(void) { int old=pending;pending=0;++cleared;return old; }
const char *OpenMXGpuGetErrorString(int s) { (void)s;return "injected"; }
int MPI_Initialized(int *v) { *v=initialized;return 0; }
int MPI_Finalized(int *v) { *v=finalized;return 0; }
static void report(void) {
 fprintf(stderr,"counts %d %d %ld %d %d\n",attempts,waits,milliseconds,cleared,pending);
}
int MPI_Abort(int comm,int error) { (void)comm;(void)error;exit(42); }
int test_nanosleep(const struct timespec *v,struct timespec *remaining) {
 (void)remaining;++waits;milliseconds+=v->tv_nsec/1000000;return 0;
}
static int operation(int allocation_status) {
 ++attempts;
 if (scenario==0) return 0;
 if (scenario==1 && attempts>2) return 0;
 if (scenario==3) return 99; /* Fatal status must not be retried. */
 pending=2;return allocation_status;
}
int main(int argc,char **argv) {
 if(argc!=3)return 2;
 atexit(report);scenario=atoi(argv[1]);int domain=atoi(argv[2]);
 if(scenario==4)pending=700;
 if(scenario==5)initialized=0;
 if(scenario==6)finalized=1;
 if(domain==0)OpenMXGpuWait(operation(2));
 else if(domain==1)OpenMXBlasWait(operation(BLAS_ALLOC));
 else OpenMXSolverWait(operation(SOLVER_ALLOC));
 return 0;
}
'''
with tempfile.TemporaryDirectory(prefix="openmx-gpu-error-") as directory:
    tmp = Path(directory)
    for name in ("openmx_gpu_error.h", "openmx_gpu_error.c"):
        (tmp / name).write_bytes((root / "source" / name).read_bytes())
    (tmp / "openmx_gpu_runtime.h").write_text(runtime)
    (tmp / "openmx_gpu_blas.h").write_text("#define OPENMX_BLAS_ALLOC_FAILED BLAS_ALLOC\n")
    (tmp / "openmx_gpu_solver.h").write_text("#define OPENMX_SOLVER_ALLOC_FAILED SOLVER_ALLOC\n")
    (tmp / "mpi.h").write_text(mpi)
    (tmp / "test.c").write_text(test)
    compiler = shlex.split(os.environ.get("CC", "cc"))
    for backend, blas_alloc, solver_alloc in (("CUDA", 3, 2), ("HIP", 2, 2)):
        subprocess.run(compiler + ["-std=c11", "-Wall", "-Wextra", "-Werror", "-I" + str(tmp),
            "-DBLAS_ALLOC=" + str(blas_alloc), "-DSOLVER_ALLOC=" + str(solver_alloc),
            "-Dnanosleep=test_nanosleep", str(tmp / "openmx_gpu_error.c"), str(tmp / "test.c"),
            "-o", str(tmp / "check")], check=True)
        # Last OOM is deliberately preserved on terminal failure.
        cases = [(0, 0, "1 0 0 0 0"), (1, 0, "3 2 3 2 0"),
                 (2, 42, "9 8 127 8 2"), (3, 42, "1 0 0 0 0"),
                 (4, 42, "0 0 0 0 700"), (5, 1, "9 8 127 8 2"),
                 (6, 1, "9 8 127 8 2")]
        for domain in range(3):
            for scenario, code, counts in cases:
                result = subprocess.run([str(tmp / "check"), str(scenario), str(domain)],
                    text=True, capture_output=True, timeout=5)
                assert result.returncode == code, (backend, scenario, domain, result)
                assert "counts " + counts in result.stderr, result.stderr
    print("PASS: runtime/BLAS/solver retries, fatal/pending errors, bounded backoff and MPI lifetime")

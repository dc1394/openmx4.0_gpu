#!/usr/bin/env python3
"""Check how Eigen_lapack.c and EigenBand_lapack.c handle a failed GPU
eigensolve, without a GPU or an MPI library.

The production files (Eigen_lapack.c, EigenBand_lapack.c and the LAPACK
helpers they call) are compiled for the host and linked with
tests/eigen_gpu_failure_harness.c, whose stubs replace the gpusolver wrappers
(solving with LAPACK, or failing with info 3, -2 = non-finite eigenpairs, or
-1007 = a library error after scribbling over their matrix) and MPI_Abort.
EigenBand_lapack must fall back to Eigen_HH on the original matrix and agree
with its CPU path; every other entry point must call MPI_Abort before storing
anything.  Uses the system reference LAPACK; about half a minute.
"""

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

PRODUCTION = ["Eigen_lapack.c", "EigenBand_lapack.c", "lapack_dstedc1.c", "lapack_dstegr1.c",
              "lapack_dstevx1.c", "lapack_dstedc2.c", "lapack_dstegr2.c", "lapack_dstevx2.c"]


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cc", default="gcc", help="host C compiler (default gcc)")
    parser.add_argument("--mpi-include", default=os.environ.get("MPI_HOME", "/home/kawai/opt/openmpi-rocmclang")
                        + "/include", help="directory of mpi.h (openmx_common.h includes it)")
    parser.add_argument("--libs", default="-llapack -lblas -lm", help="LAPACK/BLAS link flags")
    args = parser.parse_args()

    source = root / "source"
    with tempfile.TemporaryDirectory(prefix="openmx-eigen-failure-") as tmp:
        objects = []
        for name in PRODUCTION + ["../tests/eigen_gpu_failure_harness.c"]:
            path = (source / name).resolve()
            obj = Path(tmp) / (path.stem + ".o")
            command = [args.cc, "-O2", "-std=c11", "-w", "-fcommon", "-fopenmp", "-I" + str(source),
                       "-I" + args.mpi_include, "-c", str(path), "-o", str(obj)]
            subprocess.run(command, check=True)
            objects.append(str(obj))
        binary = Path(tmp) / "eigen_gpu_failure"
        subprocess.run([args.cc, "-fopenmp", *objects, *shlex.split(args.libs), "-o", str(binary)], check=True)
        env = dict(os.environ, OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "4"),
                   OPENBLAS_NUM_THREADS=os.environ.get("OPENBLAS_NUM_THREADS", "4"))
        subprocess.run([str(binary)], check=True, env=env)


if __name__ == "__main__":
    main()

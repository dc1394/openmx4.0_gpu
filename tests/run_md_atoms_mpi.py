#!/usr/bin/env python3
"""Check the production MD integrator, periodic images and MPI atom records."""
import os
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
CORE = Path(os.environ["OPENMX_CORE_LIBRARY"]).resolve(strict=True)
environment = os.environ.copy()
environment.setdefault("OMPI_CC", "cc")
environment.setdefault("OMPI_MCA_rmaps_base_oversubscribe", "1")

with tempfile.TemporaryDirectory(prefix="openmx-md-mpi-") as temp:
    exe = Path(temp) / "md_atoms"
    subprocess.run(
        [os.environ["MPICC"], "-std=c11", "-O2", "-DOPENMX_GPU_BACKEND=0",
         "-I" + str(ROOT / "source"), "-I" + str(ROOT / "source/backends/cpu/include"),
         "-I" + str(ROOT / "core/include"), str(ROOT / "tests/md_atoms_mpi.c"),
         str(ROOT / "source/openmx_md_step.c"), str(CORE),
         "-Wl,-rpath," + str(CORE.parent), "-lm", "-o", str(exe)],
        env=environment, check=True, timeout=30,
    )
    for ranks in (1, 4):
        subprocess.run([os.environ["MPIEXEC"], "-np", str(ranks), str(exe)],
                       env=environment, check=True, timeout=30)

#!/usr/bin/env python3
"""Exercise real CUDA/HIP archive preparation and resource adapters without an SDK."""
from pathlib import Path
import os
import resource
import shlex
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
library = Path(sys.argv[1]).resolve(strict=True)
adapter = (ROOT / "source/openmx_solver_workspace.h").read_text()
begin = adapter.index("static inline OpenMXNativeResources *OpenMX_Solver_Resources(")
end = adapter.index("\n}", begin) + 2
harness = (ROOT / "tests/force_hnl_gpu_harness.cpp").read_text().replace(
    "/* OPENMX_GPU_RESOURCE_ADAPTER */", adapter[begin:end])

with tempfile.TemporaryDirectory(prefix="openmx-force-hnl-archives-") as directory:
    folder = Path(directory)
    program = folder / "check.cpp"
    program.write_text(harness)
    for hip in (0, 1):
        exe = folder / ("check" + str(hip))
        command = shlex.split(os.environ.get("CXX", "c++")) + shlex.split(os.environ.get("CXXFLAGS", "")) + [
            "-std=c++20", "-O1", "-g", "-fopenmp", "-fno-pie", "-no-pie",
            "-fsanitize=address,undefined", "-fno-sanitize-recover=all", f"-DOPENMX_GPU_HIP={hip}",
            "-I" + str(ROOT / "core/tests"), "-I" + str(ROOT / "core/tests/fixtures/native_resources"),
            "-I" + str(ROOT / "source"), "-I" + str(ROOT / "core/include"),
            str(program), str(ROOT / "source/openmx_env.c"), str(ROOT / "core/tests/allocation_faults.cpp"), str(library),
            "-Wl,-rpath," + str(library.parent), "-o", str(exe)]
        subprocess.run(command, check=True)

        def bounded_stack():
            _, hard = resource.getrlimit(resource.RLIMIT_STACK)
            resource.setrlimit(resource.RLIMIT_STACK, (8 * 1024 * 1024, hard))

        subprocess.run([str(exe)], check=True, timeout=120, preexec_fn=bounded_stack)

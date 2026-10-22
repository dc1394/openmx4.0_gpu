#!/usr/bin/env python3
"""Check actual CUDA/HIP Force4B preparation with the C++ owner and SDK model.

GPU arithmetic and real MPI progress are covered by separate integration tests;
this test checks ownership, packing, admission, partial allocation and teardown.
"""
from pathlib import Path
import os
import resource
import shlex
import subprocess
import sys
import tempfile

from run_force4b_stream_cpu import extract_function, extract_struct

ROOT = Path(__file__).resolve().parents[1]
library = Path(sys.argv[1]).resolve(strict=True)
adapter = (ROOT / "source/openmx_solver_workspace.h").read_text()
a = adapter.index("static inline OpenMXNativeResources *OpenMX_Solver_Resources(")
b = adapter.index("\n}", a) + 2
harness = (ROOT / "tests/force_vna_gpu_harness.cpp").read_text().replace(
    "/* OPENMX_GPU_RESOURCE_ADAPTER */", adapter[a:b])
functions = [
    "Force_gpu_node_reserve", "Force_gpu_arena_off", "Force4B_GpuWorkspaceSizes",
    "Force4B_ArchiveGpuPreferred", "Force4B_GpuBegin",
    "Force4B_GpuCase2Begin", "Force4B_GpuArchiveHalo", "Force4B_GpuArchiveCase2",
    "Force4B_GpuCase1SizeAdd", "Force4B_GpuCase1SizeMul",
    "Force4B_GpuCase1ArenaOff", "Force4B_GpuCase1HaloCount", "Force4B_GpuCase1Layout",
    "Force4B_GpuCase1PlanAdd", "Force4B_GpuCase1StreamBegin",
    "Force4B_GpuCase1HostLimit",
    "Force4B_GpuCase1TurnsPlan", "Force4B_GpuCase1TurnsAllocate", "Force4B_GpuCase1TurnsBegin",
    "Force4B_GpuCase2Plan", "Force4B_GpuCase2StreamBegin",
]

with tempfile.TemporaryDirectory(prefix="openmx-force-vna-resources-") as directory:
    folder = Path(directory)
    for hip in (0, 1):
        source = (ROOT / ("source/backends/hip/src/Force.c" if hip else "source/Force.c")).read_text()
        # The host-availability query is independent of the owner. Use a fixed
        # node budget so CI RAM and unrelated jobs do not change admission.
        bodies = extract_struct(source, "Force4BCase1Device")
        bodies += "static size_t Force4B_GpuCase1HostAvailable(){return host_room;}\n"
        bodies += "\n".join(extract_function(source, name) for name in functions
                             if not hip or name != "Force4B_ArchiveGpuPreferred")
        # The production translation unit is C. These three allocations need
        # explicit casts only when those same functions are compiled as C++.
        for field in ("flat_dev", "halo_dev", "c2_dev"):
            bodies = bodies.replace("g->" + field + "=ForceGpu_Arena(", "g->" + field + "=(float*)ForceGpu_Arena(")
        program = folder / f"check{hip}.cpp"
        program.write_text(harness.replace("/* OPENMX_FORCE4B_PRODUCTION */", bodies))
        exe = folder / f"check{hip}"
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

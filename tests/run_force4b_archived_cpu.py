#!/usr/bin/env python3
"""Check CUDA archive leases against the real C++ owner and an SDK model."""
from pathlib import Path
import os
import resource
import shlex
import subprocess
import tempfile

from force_gpu_fixture import core_library_flags
from run_force4b_stream_cpu import extract_function, extract_struct

ROOT = Path(__file__).resolve().parents[1]
source = (ROOT / "source/Force.c").read_text()
body = extract_struct(source, "Force4BGpuContext")
for name in ("Force_gpu_node_reserve", "ForceGpu_DeviceSync", "Force4B_ArchiveGpuPreferred"):
    body += extract_function(source if name != "ForceGpu_DeviceSync" else
                             (ROOT / "source/openmx_force_gpu_resources.h").read_text(), name)
body += (ROOT / "source/openmx_force_archive_lease.h").read_text()
harness = (ROOT / "tests/force_archive_lease_harness.cpp").read_text().replace(
    "/* OPENMX_FORCE_ARCHIVE_PRODUCTION */", body)
with tempfile.TemporaryDirectory(prefix="openmx-force-archive-") as directory:
    folder = Path(directory)
    (folder / "check.cpp").write_text(harness)
    subprocess.run(shlex.split(os.environ.get("CXX", "c++")) + [
        "-std=c++20", "-O1", "-g", "-fsanitize=address,undefined", "-fno-sanitize-recover=all",
        "-fno-pie", "-no-pie", "-I" + str(ROOT / "core/include"),
        "-I" + str(ROOT / "core/tests"), "-I" + str(ROOT / "core/tests/fixtures/native_resources"),
        "-I" + str(ROOT / "source"), str(folder / "check.cpp"), str(ROOT / "source/openmx_env.c"),
        str(ROOT / "core/tests/allocation_faults.cpp"), *core_library_flags(),
        "-o", str(folder / "check")], check=True)

    def bounded_stack():
        _, hard = resource.getrlimit(resource.RLIMIT_STACK)
        resource.setrlimit(resource.RLIMIT_STACK, (8 * 1024 * 1024, hard))

    subprocess.run([str(folder / "check")], check=True, timeout=60, preexec_fn=bounded_stack)

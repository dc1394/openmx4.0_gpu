#!/usr/bin/env python3
"""Exercise the actual C++ GPU transport and C ABI with an independent SDK model."""
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


with tempfile.TemporaryDirectory(prefix="openmx-gpu-views-") as directory:
    folder=Path(directory);c=folder/"c_api.o";exe=folder/"check"
    subprocess.run(shlex.split(os.environ.get("CC","cc"))+["-std=c11","-O1","-g","-fsanitize=address,undefined",
        "-fno-sanitize-recover=all","-fno-pie","-I"+str(ROOT/"source"),"-c",str(ROOT/"tests/gpu_views_c_api.c"),"-o",str(c)],check=True)
    command=shlex.split(os.environ.get("CXX","c++"))+["-std=c++17","-O1","-g","-fopenmp","-fsanitize=address,undefined",
        "-fno-sanitize-recover=all","-fno-pie","-no-pie","-I"+str(ROOT/"source"),"-I"+str(ROOT/"tests"),
        "-I"+str(ROOT/"core/tests"),"-include",str(ROOT/"tests/fixtures/gpu_views_model.hpp"),
        str(ROOT/"tests/gpu_views_harness.cpp"),str(ROOT/"source/openmx_gpu_views.cpp"),
        str(ROOT/"source/openmx_env.c"),str(ROOT/"core/tests/allocation_faults.cpp"),str(c),"-o",str(exe)]
    subprocess.run(command,check=True)
    subprocess.run([str(exe)],check=True,timeout=120,preexec_fn=bounded_stack)

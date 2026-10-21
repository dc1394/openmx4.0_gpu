#!/usr/bin/env python3
"""Compare actual HNL/core-hole call sites in both backends with frozen traces."""
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
CORE = Path(os.environ["OPENMX_CORE_LIBRARY"]).resolve(strict=True)


def production_function(source, name):
    # Comments contain braces, including old commented-out numerical loops.
    clean = re.sub(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\x27(?:\\.|[^\x27\\])*\x27',
                   lambda m: re.sub(r'[^\n]', ' ', m[0]), source, flags=re.S)
    match = re.search(r'^(?:static\s+)?void ' + name + r'\([^;{}]*\)\s*\{', clean, re.M)
    assert match, name
    begin = match.end()
    end, depth = begin, 1
    while depth:
        depth += (clean[end] == '{') - (clean[end] == '}')
        end += 1
    return source[begin:end-1]


fixture = (ROOT / "tests/fixtures/force_traces_before_flat.c").read_text()
harness = (ROOT / "tests/force_traces_harness.c").read_text()
with tempfile.TemporaryDirectory(prefix="openmx-force-traces-") as directory:
    tmp = Path(directory)
    (tmp / "mpi.h").write_text("#pragma once\nstatic inline int MPI_Abort(int comm,int code) "
                               "{(void)comm;(void)code;abort();}\n")
    for backend, path in (("cuda", "source/Force.c"), ("hip", "source/backends/hip/src/Force.c")):
        for name in ("Force_HNL", "Force_CoreHole"):
            body = production_function((ROOT / path).read_text(), name)
            calls = re.findall(r'OpenMX_ForceSpinTrace\([^;]*;', body)
            assert len(calls) == 13, (backend, name, len(calls))
            production = "static void production_trace(int number) {switch(number) {\n"
            production += "\n".join(f"case {i}: {call} break;" for i, call in enumerate(calls))
            production += "\ndefault: abort();}}\n"
            source = tmp / f"{backend}-{name}.c"
            source.write_text(harness.replace("/* OPENMX_FORCE_REFERENCE */", fixture)
                              .replace("/* OPENMX_FORCE_PRODUCTION */", production))
            exe = source.with_suffix("")
            command = shlex.split(os.environ.get("CC", "cc")) + shlex.split(os.environ.get("CFLAGS", "")) + [
                "-std=c11", "-O2", "-fopenmp", "-ffp-contract=off", "-I" + str(tmp),
                "-I" + str(ROOT / "source"), str(source), str(CORE),
                "-Wl,-rpath," + str(CORE.parent), "-lm", "-o", str(exe)]
            subprocess.run(command, check=True)
            subprocess.run([str(exe)], check=True, timeout=90)

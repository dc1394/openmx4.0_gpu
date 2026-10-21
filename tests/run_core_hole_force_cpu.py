#!/usr/bin/env python3
"""Compare the real native bridge and C++ angular kernel with frozen code."""
import os
from pathlib import Path
import re
import resource
import shlex
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
library = Path(sys.argv[1]).resolve(strict=True)
source = (ROOT / "source/openmx_force_projectors.h").read_text()
begin = source.index("void dHCH_SO(")
production = source[begin:source.rindex("#endif")]
reference = (ROOT / "tests/fixtures/core_hole_spinor_before_views.c").read_text()
harness = (ROOT / "tests/core_hole_force_harness.c").read_text()
with tempfile.TemporaryDirectory(prefix="openmx-core-hole-force-") as directory:
    folder = Path(directory)
    program = folder / "check.c"
    program.write_text(harness.replace("/* OPENMX_CORE_HOLE_REFERENCE */", reference)
                       .replace("/* OPENMX_CORE_HOLE_PRODUCTION */", production))
    command = shlex.split(os.environ.get("CC", "cc")) + shlex.split(os.environ.get("CFLAGS", "")) + [
        "-std=c11", "-O1", "-g", "-fno-pie", "-no-pie", "-ffp-contract=off",
        "-fsanitize=address,undefined", "-fno-sanitize-recover=all",
        "-I" + str(ROOT / "source"), str(program), str(library),
        "-Wl,-rpath," + str(library.parent), "-lm", "-o", str(folder / "check")]
    subprocess.run(command, check=True)

    def bounded_stack():
        _, hard = resource.getrlimit(resource.RLIMIT_STACK)
        resource.setrlimit(resource.RLIMIT_STACK, (8 * 1024 * 1024, hard))

    subprocess.run([str(folder / "check")], check=True, timeout=120, preexec_fn=bounded_stack)

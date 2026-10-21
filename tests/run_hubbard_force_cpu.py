#!/usr/bin/env python3
"""Compare actual shared Hubbard derivatives with the frozen native functions."""
import os
from pathlib import Path
import resource
import shlex
import subprocess
import sys
import tempfile

root = Path(__file__).resolve().parents[1]
library = Path(sys.argv[1]).resolve(strict=True)
reference = (root / "tests/fixtures/hubbard_force_before_flat.c").read_text()
harness = (root / "tests/hubbard_force_harness.c").read_text()
with tempfile.TemporaryDirectory(prefix="openmx-hubbard-force-") as directory:
    folder = Path(directory)
    (folder / "check.c").write_text(harness.replace("/* OPENMX_HUBBARD_REFERENCE */", reference))
    command = shlex.split(os.environ.get("CC", "cc")) + [
        "-std=c11", "-O1", "-g", "-fno-pie", "-no-pie", "-ffp-contract=off",
        "-fsanitize=address,undefined", "-fno-sanitize-recover=all",
        "-I" + str(root / "source"), str(folder / "check.c"), str(library),
        "-Wl,-rpath," + str(library.parent), "-lm", "-o", str(folder / "check")]
    subprocess.run(command, check=True)

    def bounded_stack():
        soft, hard = resource.getrlimit(resource.RLIMIT_STACK)
        if soft == resource.RLIM_INFINITY:
            resource.setrlimit(resource.RLIMIT_STACK, (8 * 1024 * 1024, hard))

    subprocess.run([str(folder / "check")], check=True, timeout=90, preexec_fn=bounded_stack)

#!/usr/bin/env python3
"""Check actual native projector row selections against frozen scalar code."""
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


def function(name):
    clean = re.sub(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\x27(?:\\.|[^\x27\\])*\x27',
                   lambda m: re.sub(r'[^\n]', ' ', m[0]), source, flags=re.S)
    match = re.search(r'^(?:static\s+)?void ' + name + r'\([^;{}]*\)\s*\{', clean, re.M)
    assert match, name
    end, depth = match.end(), 1
    while depth:
        depth += (clean[end] == '{') - (clean[end] == '}')
        end += 1
    return source[match.start():end]


body = function("dHNL")
clean = re.sub(r'/\*.*?\*/|//[^\n]*', lambda m: " " * len(m[0]), body, flags=re.S)
blocks, fallbacks, relative = [], [], []
for match in re.finditer(r'if\(electronic->fields\[OPENMX_ELECTRONIC_PROJECTORS\].values\) \{', clean):
    end, depth = match.end(), 1
    while depth:
        depth += (clean[end] == '{') - (clean[end] == '}')
        end += 1
    blocks.append(body[match.end():end-1])
    assert clean[end:].lstrip().startswith("else")
    fallback = clean.index("{", end)
    stop, depth = fallback + 1, 1
    while depth:
        depth += (clean[stop] == '{') - (clean[stop] == '}')
        stop += 1
    fallbacks.append(body[fallback+1:stop-1])
    relative.append("1" if "dHNL_SO_rows(" in body[fallback:stop] else "0")
assert len(blocks) == 9 and relative.count("1") == 5
production = function("dHNL_SO_rows") + "\n" + function("dHNL_block_term") + "\n" + function("dHNL_block")
production += "\nstatic const int relative_cases[9]={" + ",".join(relative) + "};\n"
production += ("static void production_block_call(int number,int flat) {\n"
               "int l,l1,l2,l3;double ene,sumx,sumy,sumz;\n"
               "dcomplex sumx0,sumy0,sumz0,sumx1,sumy1,sumz1,sumx2,sumy2,sumz2;\n"
               "switch(number) {\n")
production += "\n".join(f"case {i}: {{if(flat) {{{block}}}else {{{fallbacks[i]}}} break;}}" for i, block in enumerate(blocks))
production += "\ndefault: abort();}}\n"
reference = (ROOT / "tests/fixtures/projector_force_before_views.c").read_text()
harness = (ROOT / "tests/projector_force_harness.c").read_text()
with tempfile.TemporaryDirectory(prefix="openmx-projector-force-") as directory:
    folder = Path(directory)
    program = folder / "check.c"
    program.write_text(harness.replace("/* OPENMX_PROJECTOR_REFERENCE */", reference)
                       .replace("/* OPENMX_PROJECTOR_PRODUCTION */", production))
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

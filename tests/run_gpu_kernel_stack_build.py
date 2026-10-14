#!/usr/bin/env python3
"""Exercise each production stack-scan link recipe with CPU-only tool doubles.

Failed compilation, relinking and cuobjdump calls must preserve the committed
stack limit and remove the incomplete binary. A plain make retry must recover.
The simulated resource report includes CPU-only objects, multiple architectures,
and a linked library with a larger frame that must not affect OpenMX's limit.
"""

import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
MAKEFILES = ("Makefile", "Makefile.pegasus", "Makefile.bunshiken")

COMPILER = r'''#!/usr/bin/env python3
import os
from pathlib import Path
import sys
args = sys.argv[1:]
phase = os.environ.get("FAIL_PHASE", "")
if "-c" in args:
    if phase == "compile":
        sys.exit(11)
    limit = next(a.split("=", 1)[1] for a in args if a.startswith("-DOPENMX_GPU_KERNEL_STACK_BYTES="))
    Path("set_cuda_default_device_from_local_rank.o").write_text(limit)
else:
    counter = Path("links")
    count = int(counter.read_text()) + 1 if counter.exists() else 1
    counter.write_text(str(count))
    Path("openmx").write_text(Path("set_cuda_default_device_from_local_rank.o").read_text())
    if phase == "relink" and count > 1:
        sys.exit(12)
'''

DUMP = r'''#!/usr/bin/env python3
import os
from pathlib import Path
import sys
name = Path(sys.argv[-1]).name
phase = os.environ.get("FAIL_PHASE", "")
if (phase == "object-scan" and name == "kernel.o") or (phase == "binary-scan" and name == "openmx"):
    print("cuobjdump fatal: simulated scan error", file=sys.stderr)
    sys.exit(13)
if name != "openmx" and name != "kernel.o":
    print("cuobjdump info: File '%s' does not contain device code" % name, file=sys.stderr)
    sys.exit(255)
if name == "kernel.o":
    print("\tFunction own_kernel:\n  REG:16 STACK:0 SHARED:0")
else:
    print("Function library_kernel:\n  REG:64 STACK:65536")
    print("arch = sm_80\n Function own_kernel:\n  REG:32 STACK:1536")
    print("arch = sm_120\n Function own_kernel:\n  REG:32 STACK:4096")
'''


def recipe(source: str) -> str:
    lines = source.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("openmx:"))
    end = start + 1
    while end < len(lines) and lines[end].startswith("\t"):
        end += 1
    # Use the actual error-deletion policy, not one injected by this test.
    deletion = "\n".join(line for line in lines if line.startswith(".DELETE_ON_ERROR:"))
    return deletion + "\n" + "\n".join(lines[start:end]) + "\n"


def run_case(makefile: str, phase: str, source: str) -> None:
    with tempfile.TemporaryDirectory(prefix="openmx-stack-build-") as tmp:
        work = Path(tmp)
        for name, text in (("cc", COMPILER), ("dump", DUMP), ("force_rpath", "#!/bin/sh\nexit 0\n")):
            path = work / name
            path.write_text(text)
            path.chmod(0o755)
        for name in ("kernel.o", "host.o"):
            (work / name).touch()
        (work / "set_cuda_default_device_from_local_rank.o").write_text("1024")
        (work / "gpu_kernel_stack.max").write_text("1024\n")
        (work / "Makefile").write_text(
            "CC=./cc\nCUOBJDUMP=./dump\nOBJS=kernel.o host.o set_cuda_default_device_from_local_rank.o\n"
            + recipe(source)
        )
        env = dict(os.environ, FAIL_PHASE=phase)
        failed = subprocess.run(["make", "openmx"], cwd=work, env=env, text=True, capture_output=True)
        assert failed.returncode != 0, (makefile, phase, failed.stdout, failed.stderr)
        assert not (work / "openmx").exists(), (makefile, phase, "failed binary retained")
        assert (work / "gpu_kernel_stack.max").read_text() == "1024\n", (makefile, phase, "stamp committed too soon")
        assert not list(work.glob("gpu_kernel_*.tmp")), (makefile, phase, "temporary files retained")
        env.pop("FAIL_PHASE")
        retry = subprocess.run(["make", "openmx"], cwd=work, env=env, text=True, capture_output=True)
        assert retry.returncode == 0, (makefile, phase, retry.stdout, retry.stderr)
        assert (work / "openmx").read_text() == "4096", (makefile, phase, "retry used stale limit")
        assert (work / "gpu_kernel_stack.max").read_text() == "4096\n"
        assert not list(work.glob("gpu_kernel_*.tmp"))
        links = (work / "links").read_text()
        subprocess.run(["make", "-s", "openmx"], cwd=work, env=env, check=True)
        assert (work / "links").read_text() == links, (makefile, "incremental build relinked")


def main() -> None:
    revision = os.environ.get("OPENMX_TEST_REVISION")
    for name in MAKEFILES:
        if revision:
            source = subprocess.check_output(["git", "show", f"{revision}:source/{name}"], cwd=ROOT, text=True)
        else:
            source = (ROOT / "source" / name).read_text()
        for phase in ("compile", "relink", "object-scan", "binary-scan"):
            run_case(name, phase, source)
    print("PASS: all 3 stack-scan link recipes recover from 4 failure stages and preserve incremental builds")


if __name__ == "__main__":
    main()

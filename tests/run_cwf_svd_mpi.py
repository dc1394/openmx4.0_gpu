#!/usr/bin/env python3
"""Run real/complex distributed SVD invariants with a real MPI/ScaLAPACK.

No GPU device is used. Supply the matching LP64 libraries, C++ compiler and
Fortran runtime flags for the native toolchain. With --costa-prefix, force
the prefixed compatibility archive into the positive executable to ensure
it cannot replace vendor symbols. --negative-control instead forces the
unprefixed archive and requires detection at four ranks.

Example (paths depend on the SDK):
  python3 tests/run_cwf_svd_mpi.py --cxx nvc++ --mpicc mpicc \
    --mpiexec mpirun --library /mpi/lib/libscalapack_lp64.a \
    --library /compiler/lib/liblapack_lp64.a \
    --library /compiler/lib/libblas_lp64.a \
    --link-flags='-pgf90libs -mp -lmpi_mpifh' \
    --costa-prefix source/third_party/gpusolver2-install \
    --negative-control --output build/cwf-svd-mpi
"""
from pathlib import Path
import argparse
import json
import os
import shlex
import signal
import subprocess
import sys


def mpi_flags(wrapper):
    try:
        compile_flags = subprocess.check_output(wrapper + ["--showme:compile"], text=True)
        link_flags = subprocess.check_output(wrapper + ["--showme:link"], text=True)
        return shlex.split(compile_flags), shlex.split(link_flags)
    except subprocess.CalledProcessError:
        # MPICH/Intel MPI's -show emits a complete compiler command. These
        # options also work after the input in our single compile/link call.
        flags = shlex.split(subprocess.check_output(wrapper + ["-show"], text=True))
        if not flags:
            raise RuntimeError("empty MPI wrapper configuration")
        return [], flags[1:]


def run(command, output, environment, timeout):
    with output.open("w") as log:
        process = subprocess.Popen(command, env=environment, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return process.wait(timeout=timeout)
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            except ProcessLookupError:
                pass
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", default=os.environ.get("CXX", "c++"))
    parser.add_argument("--mpicc", default="mpicc")
    parser.add_argument("--mpiexec", default="mpiexec")
    parser.add_argument("--mpi-args", default="")
    parser.add_argument("--library", type=Path, action="append", required=True)
    parser.add_argument("--link-flags", default="")
    parser.add_argument("--costa-prefix", type=Path)
    parser.add_argument("--negative-control", action="store_true")
    parser.add_argument("--ranks", default="1,2,4")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ranks = [int(value) for value in args.ranks.split(",")]
    if not ranks or any(value < 1 for value in ranks) or args.timeout < 1:
        parser.error("positive ranks and timeout are required")
    if args.negative_control and (not args.costa_prefix or 4 not in ranks):
        parser.error("negative control needs --costa-prefix and four ranks")
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).with_name("cwf_svd_mpi.cpp")
    compile_flags, link_flags = mpi_flags(shlex.split(args.mpicc))
    base = (shlex.split(args.cxx) + shlex.split(os.environ.get("CXXFLAGS", "")) +
            ["-std=c++17", "-O0", *compile_flags, str(source), "-o"])
    vendor_libraries = [str(path.resolve(strict=True)) for path in args.library]
    runtime_flags = link_flags + shlex.split(args.link_flags)
    environment = {key: value for key, value in os.environ.items()
                   if key not in ("LD_PRELOAD", "LD_LIBRARY_PATH")}
    environment.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    results, commands = [], []
    implementations = ["namespaced"] + (["interposed"] if args.negative_control else [])
    for implementation in implementations:
        executable = directory / implementation
        prefix_libraries = []
        if args.costa_prefix:
            lib = args.costa_prefix.resolve() / "lib"
            name = "costa_prefixed_scalapack" if implementation == "namespaced" else "costa_scalapack"
            archive = lib / ("lib" + name + ".a")
            core = lib / "libcosta.a"
            archive.resolve(strict=True); core.resolve(strict=True)
            prefix_libraries = (["-Wl,-force_load," + str(archive), str(core)]
                                if sys.platform == "darwin" else
                                ["-Wl,--whole-archive", str(archive),
                                 "-Wl,--no-whole-archive", str(core)])
        # Keep COSTA's core in the same archive group as the vendor libraries.
        libraries = prefix_libraries + vendor_libraries
        if sys.platform != "darwin":
            libraries = ["-Wl,--start-group", *libraries, "-Wl,--end-group"]
        command = base + [str(executable), *libraries, *runtime_flags]
        commands.append(command)
        code = run(command, directory / (implementation + "-build.log"), environment, args.timeout)
        if code:
            raise RuntimeError("SVD test compilation failed: " + implementation)
        for count in (ranks if implementation == "namespaced" else [4]):
            command = (shlex.split(args.mpiexec) + shlex.split(args.mpi_args) +
                       ["-np", str(count), str(executable)])
            log = directory / (implementation + "-np" + str(count) + ".log")
            code = run(command, log, environment, args.timeout)
            rows = [json.loads(line) for line in log.read_text().splitlines() if line.startswith("{")]
            if len(rows) != 8 or any(row["ranks"] != count for row in rows):
                raise RuntimeError("incomplete distributed SVD results: " + str(log))
            passed = sum(row["passed"] for row in rows)
            if implementation == "namespaced" and (code or passed != 8):
                raise RuntimeError("vendor SVD invariant failed: " + str(log))
            if implementation == "interposed" and (code != 1 or passed == 8):
                raise RuntimeError("interposition negative control did not fail: " + str(log))
            results.append(dict(implementation=implementation, ranks=count,
                                returncode=code, passed=passed, cases=rows))
            (directory / "results.json").write_text(json.dumps(results, indent=2) + "\n")
            print(implementation, count, "ranks", passed, "/8 invariant checks", flush=True)
    (directory / "compile-commands.json").write_text(json.dumps(commands, indent=2) + "\n")
    print("PASS: real/complex SVD orthogonality, reconstruction and polar products")


if __name__ == "__main__":
    main()

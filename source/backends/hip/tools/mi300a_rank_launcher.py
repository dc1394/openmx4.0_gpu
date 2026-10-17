#!/usr/bin/env python3
r"""Preserve each MPI rank's CPU placement before loading ROCm/OpenMP.

Example (OPENMX_AFFINITY_DIR must name this run's identity directory):
  mpirun -np 8 --map-by ppr:8:node:PE=3 --bind-to core \
    python3 tools/mi300a_rank_launcher.py ./openmx input.dat -nt 3

The inherited affinity must contain exactly the requested number of threads.
Use explicit OpenMP places so subsequent runtime initialization cannot infer
places from a helper thread's broader affinity.  Each rank records only its
identity and placement settings, then execs the target with the same PID.
No GPU runtime or OpenMX build is needed by this launcher.
"""

import datetime
import json
import os
from pathlib import Path
import re
import sys
import tempfile


def _nonnegative_integer(value, name):
    if value is None or not re.fullmatch(r"[0-9]+", value):
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _thread_count(command):
    positions = [i for i, value in enumerate(command[1:], 1) if value == "-nt"]
    if len(positions) != 1 or positions[0] + 1 >= len(command):
        raise ValueError("target command must contain exactly one '-nt N'")
    value = command[positions[0] + 1]
    if not re.fullmatch(r"[1-9][0-9]*", value):
        raise ValueError("-nt must be a positive integer")
    return int(value)


def _write_identity(directory, identity):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"rank.{identity['rank']}.json"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=directory,
            prefix=f".rank.{identity['rank']}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(identity, handle, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv=None):
    command = list(sys.argv[1:] if argv is None else argv)
    if command == ["--help"]:
        print(__doc__)
        return 0
    try:
        if not command:
            raise ValueError("usage: mi300a_rank_launcher.py BINARY INPUT -nt N")
        threads = _thread_count(command)
        rank = _nonnegative_integer(os.environ.get("OMPI_COMM_WORLD_RANK"),
                                    "OMPI_COMM_WORLD_RANK")
        local_rank = _nonnegative_integer(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK"),
                                          "OMPI_COMM_WORLD_LOCAL_RANK")
        directory = os.environ.get("OPENMX_AFFINITY_DIR")
        if not directory:
            raise ValueError("OPENMX_AFFINITY_DIR must name this run's identity directory")
        cpus = sorted(os.sched_getaffinity(0))
        if len(cpus) != threads:
            raise ValueError(
                f"rank {rank}: inherited affinity has {len(cpus)} CPUs, "
                f"but -nt requests {threads}; check mpirun --map-by and --bind-to"
            )

        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(threads)
        env["OMP_PLACES"] = ",".join("{" + str(cpu) + "}" for cpu in cpus)
        env["OMP_PROC_BIND"] = "close"
        # These runtime-specific masks can override the explicit OpenMP places.
        env.pop("KMP_AFFINITY", None)
        env.pop("GOMP_CPU_AFFINITY", None)
        env.setdefault("HSA_OVERRIDE_CPU_AFFINITY_DEBUG", "0")
        identity = {
            "kind": "mi300a_rank_identity",
            "schema_version": 1,
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "rank": rank,
            "local_rank": local_rank,
            "pid": os.getpid(),
            "inherited_cpus": cpus,
            "threads": threads,
            "omp_places": env["OMP_PLACES"],
            "omp_proc_bind": env["OMP_PROC_BIND"],
            "hsa_override_cpu_affinity_debug": env["HSA_OVERRIDE_CPU_AFFINITY_DEBUG"],
        }
        _write_identity(Path(directory), identity)
    except (OSError, ValueError) as error:
        print(f"mi300a_rank_launcher: {error}", file=sys.stderr)
        return 64

    try:
        os.execvpe(command[0], command, env)
    except OSError as error:
        print(f"mi300a_rank_launcher: exec failed: {error}", file=sys.stderr)
        return 127 if isinstance(error, FileNotFoundError) else 126


if __name__ == "__main__":
    sys.exit(main())

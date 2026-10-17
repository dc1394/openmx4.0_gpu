#!/usr/bin/env python3
"""Exercise the real MPI rank launcher with short CPU-only subprocesses.

Checks inherited (including noncontiguous) placement, exec PID/argv, recorded
identity, environment overrides, and rejection before executing invalid runs.
Run: python3 tests/run_mi300a_rank_launcher.py
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "tools" / "mi300a_rank_launcher.py"
PROBE = r'''
import json, os, sys
from pathlib import Path
print(json.dumps({
    'pid': os.getpid(), 'cpus': sorted(os.sched_getaffinity(0)),
    'argv': sys.argv[1:],
    'env': {name: os.environ.get(name) for name in (
        'OMP_NUM_THREADS', 'OMP_PLACES', 'OMP_PROC_BIND',
        'HSA_OVERRIDE_CPU_AFFINITY_DEBUG', 'KMP_AFFINITY',
        'GOMP_CPU_AFFINITY', 'LAUNCHER_UNRELATED_VALUE')},
    'identity': json.loads((Path(os.environ['OPENMX_AFFINITY_DIR']) /
        ('rank.' + os.environ['OMPI_COMM_WORLD_RANK'] + '.json')).read_text()),
}))
'''


def main():
    available = sorted(os.sched_getaffinity(0))
    assert available, "host has no allowed CPUs"
    count = 0
    with tempfile.TemporaryDirectory(prefix="openmx_rank_launcher_") as temporary:
        base = Path(temporary)
        probe = base / "probe with spaces.py"
        probe.write_text(PROBE)

        def launch(cpus, target_args, updates=None, expected=0, executable=None):
            nonlocal count
            count += 1
            identity_dir = base / f"identity_{count}"
            env = os.environ.copy()
            env.update(
                OMPI_COMM_WORLD_RANK="5", OMPI_COMM_WORLD_LOCAL_RANK="5",
                OPENMX_AFFINITY_DIR=str(identity_dir),
                OMP_NUM_THREADS="999", OMP_PLACES="cores", OMP_PROC_BIND="spread",
                KMP_AFFINITY="disabled", GOMP_CPU_AFFINITY="0",
                LAUNCHER_UNRELATED_VALUE="unchanged",
                LAUNCHER_PRIVATE_VALUE="must not be written into the identity",
            )
            env.pop("HSA_OVERRIDE_CPU_AFFINITY_DEBUG", None)
            for key, value in (updates or {}).items():
                if value is None:
                    env.pop(key, None)
                else:
                    env[key] = value
            target = [sys.executable, str(probe)] if executable is None else [executable]
            command = [sys.executable, str(LAUNCHER), *target, *target_args]
            proc = subprocess.Popen(
                command, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                preexec_fn=lambda: os.sched_setaffinity(0, cpus),
            )
            stdout, stderr = proc.communicate(timeout=10)
            assert proc.returncode == expected, (command, proc.returncode, stderr)
            if expected:
                assert not stdout, stdout
                assert stderr.startswith("mi300a_rank_launcher:"), stderr
                if expected == 64:
                    assert not identity_dir.exists(), "invalid configuration wrote an identity"
                return
            assert not stderr, stderr
            actual = json.loads(stdout)
            assert actual["pid"] == proc.pid, "launcher did not preserve PID across exec"
            assert actual["cpus"] == sorted(cpus)
            assert actual["argv"] == target_args, "target argv was changed"
            identity = actual["identity"]
            assert identity["kind"] == "mi300a_rank_identity"
            assert identity["schema_version"] == 1
            assert identity["rank"] == 5 and identity["local_rank"] == 5
            assert identity["pid"] == proc.pid
            assert identity["inherited_cpus"] == sorted(cpus)
            assert identity["threads"] == len(cpus)
            places = ",".join("{" + str(cpu) + "}" for cpu in sorted(cpus))
            assert identity["omp_places"] == places
            assert identity["omp_proc_bind"] == "close"
            actual_env = actual["env"]
            assert actual_env["OMP_NUM_THREADS"] == str(len(cpus))
            assert actual_env["OMP_PLACES"] == places
            assert actual_env["OMP_PROC_BIND"] == "close"
            expected_hsa = (updates or {}).get("HSA_OVERRIDE_CPU_AFFINITY_DEBUG", "0")
            assert actual_env["HSA_OVERRIDE_CPU_AFFINITY_DEBUG"] == expected_hsa
            assert identity["hsa_override_cpu_affinity_debug"] == expected_hsa
            assert actual_env["KMP_AFFINITY"] is None
            assert actual_env["GOMP_CPU_AFFINITY"] is None
            assert actual_env["LAUNCHER_UNRELATED_VALUE"] == "unchanged"
            assert list(identity_dir.iterdir()) == [identity_dir / "rank.5.json"]
            assert "PRIVATE" not in json.dumps(identity) and "must not" not in json.dumps(identity)

        masks = [available[:1]]
        if len(available) >= 2:
            masks.append([available[0], available[-1]])
        if len(available) >= 3:
            masks.append(available[:3])
        for cpus in masks:
            args = ["input with spaces.dat", "-nt", str(len(cpus)), "--literal=$()"]
            launch(cpus, args)
        for value in ("0", "1", ""):
            launch(masks[0], ["input.dat", "-nt", "1"],
                   {"HSA_OVERRIDE_CPU_AFFINITY_DEBUG": value})
        path_probe = base / "rank-probe"
        path_probe.write_text(f"#!{sys.executable}\n" + PROBE)
        path_probe.chmod(0o755)
        launch(masks[0], ["input.dat", "-nt", "1"], {"PATH": str(base)},
               executable="rank-probe")
        launch(masks[0], ["input.dat", "-nt", "2"], expected=64)
        for args in (["input.dat"], ["input.dat", "-nt"], ["input.dat", "-nt", "0"],
                     ["input.dat", "-nt", "-1"], ["input.dat", "-nt", "1.0"],
                     ["input.dat", "-nt", "1", "-nt", "1"]):
            launch(masks[0], args, expected=64)
        for key, value in (("OMPI_COMM_WORLD_RANK", None), ("OMPI_COMM_WORLD_RANK", "-1"),
                           ("OMPI_COMM_WORLD_RANK", "1.0"),
                           ("OMPI_COMM_WORLD_LOCAL_RANK", None),
                           ("OMPI_COMM_WORLD_LOCAL_RANK", "-1"),
                           ("OPENMX_AFFINITY_DIR", None), ("OPENMX_AFFINITY_DIR", "")):
            launch(masks[0], ["input.dat", "-nt", "1"], {key: value}, expected=64)
        blocking_file = base / "not-a-directory"
        blocking_file.write_text("unchanged")
        launch(masks[0], ["input.dat", "-nt", "1"],
               {"OPENMX_AFFINITY_DIR": str(blocking_file / "identities")}, expected=64)
        assert blocking_file.read_text() == "unchanged"
        launch(masks[0], ["input.dat", "-nt", "1"], expected=127,
               executable=str(base / "missing executable"))

    print(f"PASS: {count} CPU-only rank launcher subprocess cases")


if __name__ == "__main__":
    main()

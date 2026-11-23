#!/usr/bin/env python3
"""
exx_gpu_checks.py -- the runs of the acceptance table of milestone M6
(docs/EXX_IMPLEMENTATION_GUIDE.md, section 9): HSE06 in the CUDA build.

The script needs a machine with a GPU and two executables of the same commit,
the CPU build and the CUDA build.  It runs every system three times and
compares the total energies:

  NAME-cpu    the CPU executable
  NAME-gpu    the CUDA executable
  NAME-host   the CUDA executable with OPENMX_GPU_CAPACITY_TEST_RECOVERY=1, the
              fault injection of the GPU paths: the matrix products of the
              exchange matrix are left to the host, as they are when the
              memory of the device is short

  python3 tools/exx_gpu_checks.py list
  python3 tools/exx_gpu_checks.py run --cpu build/exx-cpu/openmx --gpu build/cuda/openmx --out build/exx-m6 [NAME ...]
  python3 tools/exx_gpu_checks.py table --out build/exx-m6

"run" without a name executes every run that results.json does not hold yet;
a name is a system (its three runs) or one run.  "table" ends with status 1
when a difference exceeds its limit, a run failed or did not converge, or the
device was not used as the row expects.  Only the standard library is used;
the inputs and the parser are those of tools/exx_kpoint_checks.py and
tools/exx_scf_checks.py.

The guide asks for three things, and the table has a block for each:
  1. the total energies of the CPU build and of the CUDA build agree within
     1e-7 Hartree;
  2. with too little device memory the CPU path takes over and gives the same
     result: the runs NAME-host, within 1e-7 Hartree of NAME-cpu, with no
     matrix product on the device;
  3. the time of every stage is recorded: the four stages of the exact
     exchange on process 0, the time in the matrix products on the device,
     and the whole calculation.

The device takes a matrix product of the exchange build only when the product
is large (source/openmx_exact_exchange_device.h): the systems marked "device"
below have such products, the others run the same code with every product on
the host.  Counted with the CPU build, one build of siz-p444 has 994 matrix
products and one of diaz-p222 has 2934; in both, 4 are large enough for the
device, two for each centre atom, and they hold 58% and 53% of the
multiplications.  The table expects exactly these products on the device, or
left to the host for lack of device memory.

The systems
  ch4             methane, the cluster solver, the matrix of the Gamma point
  dia-p222        diamond, C5.0-s1p1, 2 x 2 x 2 mesh, the band solver
  siz-p444        silicon, Si7.0-s2p2d1, 4 x 4 x 4 mesh              (device)
  siz-p444-spin   the same with scf.SpinPolarization on             (device)
  siz-p444-np2    the same with 2 MPI processes on the device       (device)
  diaz-p222       diamond, C6.0-s2p2d1, 2 x 2 x 2 mesh: the largest matrices,
                  0.5 GB on the device for one product               (device)
"""
import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exx_kpoint_checks as kpoint  # noqa: E402  (inputs of the band solver)
import exx_scf_checks as gamma      # noqa: E402  (inputs of the cluster solver, parser of the output)

REPO = Path(__file__).resolve().parents[1]
LIMIT = 1.0e-7  # Hartree, section 9 of the guide
VARIANTS = ("cpu", "gpu", "host")

SYSTEMS = {}


def add(name, text, ranks=1, threads=2, large=0):
    """A system: its input, and the number of matrix products of one build that are large enough for the device."""
    SYSTEMS[name] = dict(text=text, ranks=ranks, threads=threads, large=large, device=0 < large)


add("ch4", gamma.RUNS["ch4-hse"]["text"])
add("dia-p222", kpoint.make_input("dia_p222", "dia", (1, 1, 1), (2, 2, 2), "HSE06"))
# two large products for each of the 2 centre atoms, and as many again for the second spin channel
add("siz-p444", kpoint.make_input("siz_p444", "siz", (1, 1, 1), (4, 4, 4), "HSE06"), large=4)
add("siz-p444-spin", kpoint.make_input("siz_p444_spin", "siz", (1, 1, 1), (4, 4, 4), "HSE06", spin="on"), large=8)
add("siz-p444-np2", kpoint.make_input("siz_p444_np2", "siz", (1, 1, 1), (4, 4, 4), "HSE06"), ranks=2, large=4)
add("diaz-p222", kpoint.make_input("diaz_p222", "diaz", (1, 1, 1), (2, 2, 2), "HSE06"), large=4)

RUNS = [f"{system}-{variant}" for system in SYSTEMS for variant in VARIANTS]


def split(run):
    system, variant = run.rsplit("-", 1)
    return system, variant


def device_use(std):
    """What the standard output says about the device and the matrix products of the exchange matrix."""
    use = re.search(r"<EXX>  GPU: (\d+) matrix products of the exchange matrix on the device, in (\d+) processes\n"
                    r"<EXX>  \(at most (\S+) s in one process\); (\d+) left to the host for lack of device memory\.",
                    std)
    off = re.search(r"<EXX>  WARNING: (\d+) processes stopped using their device", std)
    # of one build on process 0: the number of products, and m, n and the inner dimension of the largest
    build = re.search(r"<EXX>  matrix products of process 0: (\d+), the largest (\d+) x (\d+) with inner dimension (\d+)",
                      std)
    memory = [float(mb) for mb in re.findall(r"<EXX>  on the device: \d+ in \S+ s, at most (\S+) MB for one product", std)]
    return dict(products=int(use.group(1)) if use else 0, processes=int(use.group(2)) if use else 0,
                seconds=float(use.group(3)) if use else 0.0, declined=int(use.group(4)) if use else 0,
                switched_off=int(off.group(1)) if off else 0,
                build=[int(v) for v in build.groups()] if build else None, megabytes=max(memory, default=0.0),
                failures=[f"process {p}: {what} (status {status})" for p, what, status in re.findall(
                    r"<EXX>  WARNING: process (\d+): (.+) failed with status (-?\d+)\.", std)])


def execute(args, run):
    system, variant = split(run)
    spec = SYSTEMS[system]
    run_dir = args.out / run
    run_dir.mkdir(parents=True, exist_ok=True)
    title = re.search(r"System.Name\s+(\S+)", spec["text"]).group(1)
    (run_dir / "input.dat").write_text(spec["text"].replace("@DATA_PATH@", str(args.data_path)))
    threads = str(args.threads or spec["threads"])
    env = dict(os.environ, OMP_NUM_THREADS=threads, OPENBLAS_NUM_THREADS=threads)
    env.pop("OPENMX_GPU_CAPACITY_TEST_RECOVERY", None)
    if variant == "host":
        env["OPENMX_GPU_CAPACITY_TEST_RECOVERY"] = "1"
    executable = args.cpu if variant == "cpu" else args.gpu
    start = time.time()
    with open(run_dir / "std.txt", "w") as handle:
        status = subprocess.run([*shlex.split(args.mpirun), "-np", str(spec["ranks"]), str(executable.resolve()),
                                 "input.dat"], cwd=run_dir, stdout=handle, stderr=subprocess.STDOUT,
                                env=env).returncode
    result = gamma.parse(run_dir, title)
    result["device"] = device_use((run_dir / "std.txt").read_text(errors="replace"))
    result.update(status=status, wall=round(time.time() - start, 2), ranks=spec["ranks"], threads=int(threads))
    return result


def machine():
    """The devices of the machine as nvidia-smi names them; nothing when the command is missing."""
    try:
        done = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return []
    return [line.strip() for line in done.stdout.splitlines() if line.strip()] if done.returncode == 0 else []


def run(args):
    args.out.mkdir(parents=True, exist_ok=True)
    store = args.out / "results.json"
    results = json.loads(store.read_text()) if store.exists() else {}
    names = []
    for name in args.names:
        if name in SYSTEMS:
            names += [f"{name}-{variant}" for variant in VARIANTS]
        elif name in RUNS:
            names.append(name)
        else:
            sys.exit(f"unknown run or system: {name}")
    results["_machine"] = machine()
    for name in names or [n for n in RUNS if n not in results]:
        r = results[name] = execute(args, name)
        store.write_text(json.dumps(results, indent=1))
        d = r["device"]
        print(f"{name:20s} exit {r['status']} {r['wall']:7.1f}s Utot {r['Utot']} Uexx {r['Uexx']} builds {r['builds']} "
              f"converged {r['achieved']} device products {d['products']} declined {d['declined']}", flush=True)
    return 0

# ---- table ----------------------------------------------------------------


def table(args):
    results = json.loads((args.out / "results.json").read_text())
    failures, missing = [], []

    def converged(name):
        r = results.get(name)
        return bool(r) and r["status"] == 0 and r["achieved"] and r["outer_converged"]

    def expect(label, ok, detail):
        print(f"  {label:50s} {detail}  {'ok' if ok else 'FAIL'}")
        if not ok:
            failures.append(label)

    def difference(system, variant, key="Utot"):
        a, b = results[f"{system}-{variant}"][key], results[f"{system}-cpu"][key]
        return None if a is None or b is None else a - b

    print("devices:", "; ".join(results.get("_machine") or ["not recorded (nvidia-smi gave nothing)"]))
    for system, spec in SYSTEMS.items():
        present = [v for v in VARIANTS if f"{system}-{v}" in results]
        if not present:
            continue
        print(f"== {system}" + ("  (device)" if spec["device"] else ""))
        absent = [f"{system}-{v}" for v in VARIANTS if v not in present]
        if absent:
            missing += absent
            print("  not run: " + " ".join(absent))
        for variant in present:
            r = results[f"{system}-{variant}"]
            d = r["device"]
            print(f"  {variant:4s} Utot {r['Utot']}  Uexx {r['Uexx']}  builds {r['builds']}  SCF steps {r['scf_steps']}  "
                  f"processes {r['ranks']} x {r['threads']} threads")
            for failure in d["failures"]:
                print(f"       device failure: {failure}")
        if "cpu" in present:
            for variant, label in (("gpu", "CUDA build minus CPU build"), ("host", "CUDA build, products on the host, minus CPU")):
                if variant not in present:
                    continue
                name = f"{system}-{variant}"
                if not (converged(name) and converged(f"{system}-cpu")):
                    expect(f"{label}: Utot", False, "a run failed or did not converge")
                    continue
                value = difference(system, variant)
                expect(f"{label}: Utot", abs(value) <= LIMIT, f"{value:+.3e}  limit {LIMIT:.0e}")
                print(f"  {label + ': Uexx':50s} {difference(system, variant, 'Uexx'):+.3e}")
        # where the matrix products ran
        build = results[f"{system}-{present[0]}"]["device"].get("build")
        if build:
            print(f"  one build on process 0: {build[0]} matrix products, the largest {build[1]} x {build[2]} "
                  f"with inner dimension {build[3]}")
        if "gpu" in present:
            r = results[f"{system}-gpu"]
            d = r["device"]
            large = spec["large"] * (r["builds"] or 0)
            detail = (f"{d['products']} on the device in {d['processes']} processes, {d['declined']} left to the host, "
                      f"{d['switched_off']} processes switched off; large products in {r['builds']} builds: {large}")
            if d.get("megabytes"):
                print(f"  CUDA build: at most {d['megabytes']:.1f} MB of device memory for one product on process 0")
            if spec["device"]:
                expect("CUDA build: products on the device", 0 < d["products"] and d["switched_off"] == 0
                       and d["processes"] == r["ranks"] and d["products"] + d["declined"] == large, detail)
            else:
                expect("CUDA build: no device failure", d["switched_off"] == 0 and d["products"] + d["declined"] == 0,
                       detail)
        if "host" in present:
            r = results[f"{system}-host"]
            d = r["device"]
            large = spec["large"] * (r["builds"] or 0)
            detail = (f"{d['products']} on the device, {d['declined']} left to the host, "
                      f"{d['switched_off']} processes switched off; large products in {r['builds']} builds: {large}")
            expect("fault injection: no product on the device", d["products"] == 0 and d["switched_off"] == 0
                   and d["declined"] == large, detail)
        if "cpu" in present:
            d = results[f"{system}-cpu"]["device"]
            expect("CPU build: no device", d["products"] == 0 and d["declined"] == 0 and d["switched_off"] == 0,
                   f"{d['products']} on the device, {d['declined']} left to the host")
        # the times of the stages
        print("  time (s)  preparation  density matrices  exchange matrix  of it on the device  addition  DFT in total  wall")
        for variant in present:
            r = results[f"{system}-{variant}"]
            t = r.get("exx_times") or [float("nan")] * 4
            print(f"  {variant:8s}  {t[0]:11.2f}  {t[1]:16.2f}  {t[2]:15.2f}  {r['device']['seconds']:19.2f}  {t[3]:8.2f}  "
                  f"{r.get('dft_time') or float('nan'):12.2f}  {r['wall']:.1f}")
    bad = [n for n in results if not n.startswith("_") and not converged(n)]
    print("runs that failed or did not converge:", bad if bad else "none")
    print("runs of a listed system that are missing:", missing if missing else "none")
    print("rows that failed:", failures if failures else "none")
    return 1 if bad or missing or failures else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="names of the runs")
    runner = commands.add_parser("run", help="execute runs and store their numbers")
    runner.add_argument("--cpu", type=Path, required=True, help="the executable of the CPU build")
    runner.add_argument("--gpu", type=Path, required=True, help="the executable of the CUDA build")
    runner.add_argument("--out", type=Path, required=True, help="directory of the runs and of results.json")
    runner.add_argument("--mpirun", default="mpirun", help="launcher with its options (default: mpirun)")
    runner.add_argument("--data-path", type=Path, default=REPO / "DFT_DATA19", help="DATA.PATH of the inputs")
    runner.add_argument("--threads", type=int, default=0, help="OpenMP threads of a process (default: 2)")
    runner.add_argument("names", nargs="*", help="systems or runs to execute (default: every run not done yet)")
    summary = commands.add_parser("table", help="print the acceptance table from results.json")
    summary.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "list":
        print("\n".join(RUNS))
        return 0
    return run(args) if args.command == "run" else table(args)


if __name__ == "__main__":
    raise SystemExit(main())

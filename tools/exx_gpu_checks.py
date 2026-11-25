#!/usr/bin/env python3
"""
exx_gpu_checks.py -- the runs of the acceptance table of milestone M6
(docs/EXX_IMPLEMENTATION_GUIDE.md, section 9): HSE06 in the CUDA build.

The script needs a machine with a GPU and two executables of the same commit,
the CPU build and the CUDA build.  It runs a system three times and
compares the total energies:

  NAME-cpu    the CPU executable
  NAME-gpu    the CUDA executable
  NAME-host   the CUDA executable with OPENMX_GPU_CAPACITY_TEST_RECOVERY=1, the
              fault injection of the GPU paths: the matrix products of the
              exchange matrix are left to the host, as they are when the
              memory of the device is short

  python3 tools/exx_gpu_checks.py list
  python3 tools/exx_gpu_checks.py run --cpu build/exx-cpu/openmx --gpu build/exx-cuda/openmx --out build/exx-m6 [NAME ...]
  python3 tools/exx_gpu_checks.py table --out build/exx-m6

"run" without a name executes every run that results.json does not hold yet;
a name is a system (all its runs) or one run.  "table" ends with status 1
when a difference exceeds its limit, a run failed, did not converge or is
missing, or the device was not used as the row expects.  Only the standard
library is used; the inputs and the parser are those of
tools/exx_kpoint_checks.py and tools/exx_scf_checks.py.

Threads.  A run uses one thread unless --threads asks for more.  The number
goes to the program with -nt (the program does not read OMP_NUM_THREADS) and
to OPENBLAS_NUM_THREADS.  The table prints the number that the program
reports.  A launcher may bind a single process to one core;
--mpirun "mpirun --bind-to none" lifts that for timings with more threads.

The guide asks for three things, and the table has a block for each:
  1. the total energies of the CPU build and of the CUDA build agree within
     1e-7 Hartree;
  2. with too little device memory the CPU path takes over and gives the same
     result: the runs NAME-host, within 1e-7 Hartree of NAME-cpu, with no
     matrix product and no call on the device;
  3. the time of every stage is recorded: the four stages of the exact
     exchange on process 0, the time on the device, and the whole
     calculation.

Where the build runs (source/openmx_exact_exchange_device.h).  A process
computes the contributions of its share of the centre atoms, a range of their
ABFs, with one call of modules/exx for every spin channel.  In the systems
below with products on the device a share is one or all of the atoms.  The
CUDA build offers the device the whole call first:
the device takes a call whose largest matrix product is large, and
computes all products of the call there, with the arrays in device memory.  A
call that the device does not take for lack of memory goes to the host, and
the device then takes the large products of that call one at a time, as it
did before whole calls existed.  The systems marked "device" below have large
products, the others run the same code with every call and product on the
host.  Counted with the CPU build, one build of siz-p444 has 54850 matrix
products and one of diaz-p222 has 162326; in both, 4 are large enough for the
device, two for each centre atom, and they hold 58% and 53% of the
multiplications.  A call on the device makes 2 products for each centre atom
and 5 for each of its V neighbours, the same 2 of them large.  The table
expects one call for each spin channel, process and build, every call on the
device or left to the host for lack of device memory, and the large products
of the calls left to the host on the device, or left to the host as well.

The systems
  ch4                methane, the cluster solver, the matrix of the Gamma point
  dia-p222           diamond, C5.0-s1p1, 2 x 2 x 2 mesh, the band solver
  siz-p444           silicon, Si7.0-s2p2d1, 4 x 4 x 4 mesh              (device)
  siz-p444-spin      the same with scf.SpinPolarization on             (device)
  siz-p444-np2       the same with 2 MPI processes on the device       (device)
  diaz-p222          diamond, C6.0-s2p2d1, 2 x 2 x 2 mesh: the largest matrices,
                     0.5 GB on the device for one product               (device)
  ch4-np4            methane with 4 MPI processes: one of them holds the dense
                     matrix, all of them get their share of the vectors
  o2                 the triplet of O2, the cluster solver, one process for both spins
  o2-np2             the same with 2 MPI processes, one for each spin
  dia-p222-spin-np2  diamond, C5.0-s1p1, 2 x 2 x 2 mesh, scf.SpinPolarization on,
                     2 MPI processes: one for each spin, with all its k points
  dia-k211-np2       diamond, C5.0-s1p1, 2 x 1 x 1 mesh, 2 MPI processes: one k point
                     for each process
  dia-k211-np4       the same with 4 MPI processes: two share a k point, and one
                     of the two holds its dense matrix
The last six exist for the dense eigensolver below: they have no run
NAME-host, and none of their matrix products is large enough for the device.

The dense eigensolver of the device (section 6.5 of the guide).  The CUDA
build solves a matrix on the device when its dimension reaches a threshold
(800 in the band solver, 400 in the cluster solver).  The systems above are
far smaller, so the runs NAME-gpu and NAME-host solve on the host.  The runs
below lower the threshold to 1 with OPENMX_BAND_GPU_SWITCH_NUM and
OPENMX_CLUSTER_GPU_SWITCH_NUM, and the exchange matrix is then added to the
Hamiltonian on the device:

  NAME-dense  the dense path of the device
  NAME-dfall  the same, and the memory admission of the dense path is made to
              fail (OPENMX_BAND_GPU_RESERVE_MB, OPENMX_CLUSTER_GPU_DIAG_RESERVE_MB):
              the eigensolver of the host takes over
  NAME-dpers  the dense path with OPENMX_BAND_GPU_PERSISTENT=1: the vectors of a
              process with one k point stay on the device
  NAME-dser   the dense path with OPENMX_CLUSTER_GPU_DIAG_SERIAL=2: one process
              solves both spins of the cluster solver in turn
  NAME-dfp64  the dense path with OPENMX_GEMMUL8_DISABLE=1: the matrix products
              of the eigensolver in FP64 of cuBLAS instead of GEMMul8

Each of them has to agree with NAME-cpu within 1e-7 Hartree.  The standard
output says to how many dense Hamiltonians the exchange matrix was added on
the device: the number has to be positive, and 0 in NAME-dfall.

GEMMul8 in the matrix products of the exchange build (section 11 of the
guide).  The device computes them in FP64 of cuBLAS unless the input says
scf.gemmul8.enable on.  The systems marked "device" have one more run:

  NAME-g8     the CUDA executable, and the input gets the line
              scf.gemmul8.enable on: the large products go through the
              GEMMul8 bridge

It has to agree with NAME-cpu within 1e-7 Hartree, and the standard output
has to say that GEMMul8 computed every large product on the device, in the
calls and one at a time; the other products of a call are FP64.  The table
also prints its difference from NAME-gpu, the same products in FP64.  No
other run may report GEMMul8 in these products.
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

# The runs of the dense eigensolver of the device: the variables that steer the CUDA build.
DENSE = {"OPENMX_BAND_GPU_SWITCH_NUM": "1", "OPENMX_CLUSTER_GPU_SWITCH_NUM": "1"}
STEERED = {
    "dense": DENSE,
    "dfall": dict(DENSE, OPENMX_BAND_GPU_RESERVE_MB="1000000", OPENMX_CLUSTER_GPU_DIAG_RESERVE_MB="1000000"),
    "dpers": dict(DENSE, OPENMX_BAND_GPU_PERSISTENT="1"),
    "dser": dict(DENSE, OPENMX_CLUSTER_GPU_DIAG_SERIAL="2"),
    "dfp64": dict(DENSE, OPENMX_GEMMUL8_DISABLE="1"),
}
STEERED_LABEL = {
    "dense": "dense path of the device",
    "dfall": "dense path, memory admission declined",
    "dpers": "dense path, vectors kept on the device",
    "dser": "dense path, both spins on one process",
    "dfp64": "dense path, products in FP64 of cuBLAS",
}
ENVIRONMENT = {"cpu": {}, "gpu": {}, "host": {"OPENMX_GPU_CAPACITY_TEST_RECOVERY": "1"}, "g8": {}, **STEERED}
# no run inherits one of these from the caller
STEERING = sorted({key for variables in ENVIRONMENT.values() for key in variables})
# lines that a run adds to the input of its system
INPUT_LINES = {"g8": "scf.gemmul8.enable        on\n"}

SYSTEMS = {}


def add(name, text, ranks=1, large=0, calls=0, base=VARIANTS, dense=("dense",)):
    """A system: its input, the numbers of matrix products and of calls of one build that are large enough for the
    device, and its runs."""
    # only a system with products on the device can compare GEMMul8 with FP64 in them
    SYSTEMS[name] = dict(text=text, ranks=ranks, large=large, calls=calls, device=0 < large,
                         variants=tuple(base) + tuple(dense) + (("g8",) if 0 < large else ()))


add("ch4", gamma.RUNS["ch4-hse"]["text"], dense=("dense", "dfall", "dfp64"))
add("dia-p222", kpoint.make_input("dia_p222", "dia", (1, 1, 1), (2, 2, 2), "HSE06"))
# two large products for each of the 2 centre atoms, and as many again for the second spin channel; one call for
# each spin channel and process
add("siz-p444", kpoint.make_input("siz_p444", "siz", (1, 1, 1), (4, 4, 4), "HSE06"), large=4, calls=1,
    dense=("dense", "dfall", "dfp64"))
add("siz-p444-spin", kpoint.make_input("siz_p444_spin", "siz", (1, 1, 1), (4, 4, 4), "HSE06", spin="on"), large=8,
    calls=2)
add("siz-p444-np2", kpoint.make_input("siz_p444_np2", "siz", (1, 1, 1), (4, 4, 4), "HSE06"), ranks=2, large=4,
    calls=2)
add("diaz-p222", kpoint.make_input("diaz_p222", "diaz", (1, 1, 1), (2, 2, 2), "HSE06"), large=4, calls=1)
# the systems of the dense eigensolver only.  The cluster solver: processes without the dense matrix, and the
# spin channels on one process and on two.  The band solver: a process for each spin channel, a process for
# each k point, and processes that share a k point.
add("ch4-np4", gamma.RUNS["ch4-hse-np4"]["text"], ranks=4, base=("cpu", "gpu"))
add("o2", gamma.RUNS["o2-hse"]["text"], base=("cpu", "gpu"))
add("o2-np2", gamma.RUNS["o2-hse-np2"]["text"], ranks=2, base=("cpu", "gpu"), dense=("dense", "dser"))
add("dia-p222-spin-np2", kpoint.make_input("dia_p222_spin_np2", "dia", (1, 1, 1), (2, 2, 2), "HSE06", spin="on"),
    ranks=2, base=("cpu", "gpu"))
add("dia-k211-np2", kpoint.make_input("dia_k211_np2", "dia", (1, 1, 1), (2, 1, 1), "HSE06"), ranks=2,
    base=("cpu", "gpu"), dense=("dense", "dpers", "dfall"))
add("dia-k211-np4", kpoint.make_input("dia_k211_np4", "dia", (1, 1, 1), (2, 1, 1), "HSE06"), ranks=4,
    base=("cpu", "gpu"))

RUNS = [f"{system}-{variant}" for system, spec in SYSTEMS.items() for variant in spec["variants"]]


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
    # the dense eigensolver of the device: the Hamiltonians that got the exchange matrix there, whether the memory
    # admission of the dense path was declined, and whether one process solved both spins of the cluster solver
    added = re.search(r"<EXX>  GPU: the exchange matrix was added to (\d+) dense Hamiltonians on the device\.", std)
    declined = ("GPU dense memory admission declined" in std
                or bool(re.search(r"<EXX>  Cluster collinear n=\d+: the device cannot hold the dense matrices", std)))
    # the products on the device that GEMMul8 computed and those that the bridge computed in FP64; None without the line
    bridge = re.search(r"<EXX>  GPU: GEMMul8 computed (\d+) of the matrix products on the device, FP64 the other (\d+) "
                       r"\(scf\.gemmul8\.enable on\)\.", std)
    # the whole calls of modules/exx on the device: the calls, the processes, the matrix products in them, the longest
    # time of a process, the calls left to the host; GEMMul8 in the large products of the calls; the most memory of one
    # call on process 0
    calls = re.search(r"<EXX>  GPU: (\d+) calls of the exchange build on the device, in (\d+) processes, with (\d+) "
                      r"matrix products, (\d+) of them large\n<EXX>  \(at most (\S+) s in one process\); (\d+) calls left "
                      r"to the host for lack of device memory\.", std)
    call_bridge = re.search(r"<EXX>  GPU: GEMMul8 computed (\d+) of the (\d+) large matrix products of these calls, FP64 "
                            r"the other (\d+) products \(scf\.gemmul8\.enable on\)\.", std)
    call_memory = [float(mb) for mb in re.findall(
        r"<EXX>  device calls of process 0: \d+ in \S+ s with \d+ matrix products, at most (\S+) MB for one call", std)]
    return dict(additions=int(added.group(1)) if added else 0, dense_declined=declined,
                gemmul8=int(bridge.group(1)) if bridge else None, gemmul8_fp64=int(bridge.group(2)) if bridge else None,
                serial="solves both spins serially on the GPU" in std,
                products=int(use.group(1)) if use else 0, processes=int(use.group(2)) if use else 0,
                seconds=float(use.group(3)) if use else 0.0, declined=int(use.group(4)) if use else 0,
                calls=int(calls.group(1)) if calls else 0, call_processes=int(calls.group(2)) if calls else 0,
                call_products=int(calls.group(3)) if calls else 0, call_large=int(calls.group(4)) if calls else 0,
                call_seconds=float(calls.group(5)) if calls else 0.0,
                declined_calls=int(calls.group(6)) if calls else 0,
                call_gemmul8=int(call_bridge.group(1)) if call_bridge else None,
                call_gemmul8_large=int(call_bridge.group(2)) if call_bridge else None,
                call_megabytes=max(call_memory, default=0.0),
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
    (run_dir / "input.dat").write_text(spec["text"].replace("@DATA_PATH@", str(args.data_path))
                                       + INPUT_LINES.get(variant, ""))
    # one thread unless asked: the numbers of a run are then the same every time on one machine
    threads = str(args.threads or 1)
    env = dict(os.environ, OMP_NUM_THREADS=threads, OPENBLAS_NUM_THREADS=threads)
    for key in STEERING:
        env.pop(key, None)
    env.update(ENVIRONMENT[variant])
    executable = args.cpu if variant == "cpu" else args.gpu
    start = time.time()
    with open(run_dir / "std.txt", "w") as handle:
        # the program takes its number of OpenMP threads from -nt, not from the environment
        status = subprocess.run([*shlex.split(args.mpirun), "-np", str(spec["ranks"]), str(executable.resolve()),
                                 "input.dat", "-nt", threads], cwd=run_dir, stdout=handle, stderr=subprocess.STDOUT,
                                env=env).returncode
    result = gamma.parse(run_dir, title)
    std = (run_dir / "std.txt").read_text(errors="replace")
    result["device"] = device_use(std)
    # the number that the program reports; the table prints this one
    reported = re.search(r"The number of threads in each node for OpenMP parallelization is (\d+)\.", std)
    result.update(status=status, wall=round(time.time() - start, 2), ranks=spec["ranks"],
                  threads=int(reported.group(1)) if reported else None, threads_asked=int(threads))
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
            names += [f"{name}-{variant}" for variant in SYSTEMS[name]["variants"]]
        elif name in RUNS:
            names.append(name)
        else:
            sys.exit(f"unknown run or system: {name}")
    results["_machine"] = machine()
    for name in names or [n for n in RUNS if n not in results]:
        r = results[name] = execute(args, name)
        store.write_text(json.dumps(results, indent=1))
        d = r["device"]
        print(f"{name:24s} exit {r['status']} {r['wall']:7.1f}s Utot {r['Utot']} Uexx {r['Uexx']} builds {r['builds']} "
              f"converged {r['achieved']} device calls {d['calls']} declined {d['declined_calls']} "
              f"device products {d['products']} declined {d['declined']} dense additions {d['additions']}"
              + ("" if d["gemmul8"] is None else f" GEMMul8 {d['gemmul8']} FP64 {d['gemmul8_fp64']}")
              + ("" if d["call_gemmul8"] is None
                 else f" GEMMul8 in calls {d['call_gemmul8']} of {d['call_gemmul8_large']}"),
              flush=True)
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
            failures.append(f"{system}: {label}")

    def difference(system, variant, key="Utot"):
        a, b = results[f"{system}-{variant}"][key], results[f"{system}-cpu"][key]
        return None if a is None or b is None else a - b

    def use(r):
        """Where the builds of a run ran, with 0 for the numbers that a run of an older version did not record."""
        d = dict(calls=0, call_processes=0, call_products=0, call_large=0, call_seconds=0.0, declined_calls=0,
                 call_gemmul8=None, call_gemmul8_large=None, call_megabytes=0.0)
        d.update(r["device"])
        return d

    def where(r, spec):
        """The text of a row about the device: what the run did and what its builds call for."""
        d, builds = use(r), r["builds"] or 0
        return (f"calls: {d['calls']} on the device in {d['call_processes']} processes with {d['call_products']} "
                f"products, {d['call_large']} of them large, {d['declined_calls']} left to the host; products: "
                f"{d['products']} on the device, {d['declined']} left to the host; {d['switched_off']} processes "
                f"switched off; in {builds} builds: {spec['calls'] * builds} calls, {spec['large'] * builds} large "
                f"products")

    def calls_add_up(r, spec):
        """Every call on the device or left to the host, the 2 large products of a centre atom in the calls on the
        device, and those of the calls left to the host on the device or left to the host one at a time."""
        d, builds = use(r), r["builds"] or 0
        return (d["switched_off"] == 0 and d["calls"] + d["declined_calls"] == spec["calls"] * builds
                and d["call_large"] * spec["calls"] == spec["large"] * d["calls"]
                and (d["products"] + d["declined"]) * spec["calls"] == spec["large"] * d["declined_calls"])

    print("devices:", "; ".join(results.get("_machine") or ["not recorded (nvidia-smi gave nothing)"]))
    untouched = []
    for system, spec in SYSTEMS.items():
        present = [v for v in spec["variants"] if f"{system}-{v}" in results]
        if not present:
            untouched.append(system)
            continue
        print(f"== {system}" + ("  (device)" if spec["device"] else ""))
        absent = [f"{system}-{v}" for v in spec["variants"] if v not in present]
        if absent:
            missing += absent
            print("  not run: " + " ".join(absent))
        for variant in present:
            r = results[f"{system}-{variant}"]
            d = r["device"]
            print(f"  {variant:5s} Utot {r['Utot']}  Uexx {r['Uexx']}  builds {r['builds']}  SCF steps {r['scf_steps']}  "
                  f"processes {r['ranks']} x {r['threads']} threads")
            for failure in d["failures"]:
                print(f"       device failure: {failure}")
        # a run of a version before -nt was passed has no threads_asked: its number is the one that was asked for
        wrong = [f"{v} {results[f'{system}-{v}']['threads']} of {results[f'{system}-{v}']['threads_asked']}"
                 for v in present if "threads_asked" in results[f"{system}-{v}"]
                 and results[f"{system}-{v}"]["threads"] != results[f"{system}-{v}"]["threads_asked"]]
        if wrong:
            expect("OpenMP threads as asked", False, "the program reports " + ", ".join(wrong))
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
            d = use(r)
            if d["call_megabytes"]:
                print(f"  CUDA build: at most {d['call_megabytes']:.1f} MB of device memory for one call on process 0")
            if d.get("megabytes"):
                print(f"  CUDA build: at most {d['megabytes']:.1f} MB of device memory for one product on process 0")
            if spec["device"]:
                expect("CUDA build: calls on the device", 0 < d["calls"] and d["call_processes"] == r["ranks"]
                       and calls_add_up(r, spec), where(r, spec))
            else:
                expect("CUDA build: no device failure", d["switched_off"] == 0 and d["products"] + d["declined"] == 0
                       and d["calls"] + d["declined_calls"] == 0, where(r, spec))
        if "host" in present:
            r = results[f"{system}-host"]
            d = use(r)
            builds = r["builds"] or 0
            expect("fault injection: nothing on the device", d["calls"] == 0 and d["products"] == 0
                   and d["switched_off"] == 0 and d["declined_calls"] == spec["calls"] * builds
                   and d["declined"] == spec["large"] * builds, where(r, spec))
        if "cpu" in present:
            d = use(results[f"{system}-cpu"])
            expect("CPU build: no device", d["products"] == 0 and d["declined"] == 0 and d["switched_off"] == 0
                   and d["calls"] == 0 and d["declined_calls"] == 0 and d.get("additions", 0) == 0,
                   f"{d['calls']} calls and {d['products']} products on the device, {d['declined_calls']} calls and "
                   f"{d['declined']} products left to the host")
        # GEMMul8 in the matrix products on the device: only where the input asks for it
        others = [v for v in present if v != "g8" and (use(results[f"{system}-{v}"])["gemmul8"] is not None
                                                       or use(results[f"{system}-{v}"])["call_gemmul8"] is not None)]
        if spec["device"] and [v for v in present if v != "g8"]:
            expect("without scf.gemmul8.enable: products in FP64", not others,
                   "GEMMul8 reported by " + (" ".join(others) if others else "no run"))
        if "g8" in present:
            name, label = f"{system}-g8", "GEMMul8 in the products"
            r = results[name]
            d = use(r)
            if "cpu" in present:
                if converged(name) and converged(f"{system}-cpu"):
                    value = difference(system, "g8")
                    expect(f"{label}: Utot minus CPU", abs(value) <= LIMIT, f"{value:+.3e}  limit {LIMIT:.0e}")
                    print(f"  {label + ': Uexx minus CPU':50s} {difference(system, 'g8', 'Uexx'):+.3e}")
                else:
                    expect(f"{label}: Utot minus CPU", False, "a run failed or did not converge")
            if "gpu" in present and converged(name) and converged(f"{system}-gpu"):
                # the same products in FP64 of cuBLAS: a number to record, without a limit of its own
                fp64 = results[f"{system}-gpu"]
                print(f"  {label + ': Utot minus the FP64 run':50s} {r['Utot'] - fp64['Utot']:+.3e}")
                print(f"  {label + ': Uexx minus the FP64 run':50s} {r['Uexx'] - fp64['Uexx']:+.3e}")
                fp64_use = use(fp64)
                print(f"  {label + ': seconds on the device':50s} {d['seconds'] + d['call_seconds']:.2f}  "
                      f"(FP64: {fp64_use['seconds'] + fp64_use['call_seconds']:.2f})")
            counted = ("no line about GEMMul8 in the calls" if d["call_gemmul8"] is None
                       else f"{d['call_gemmul8']} of the {d['call_gemmul8_large']} large products of the calls by "
                            f"GEMMul8")
            if d["products"]:
                counted += ("; no line about GEMMul8 in the products" if d.get("gemmul8") is None
                            else f"; {d['gemmul8']} by GEMMul8 and {d['gemmul8_fp64']} in FP64 of the products")
            # the large products of the calls, and those of the calls left to the host on the device
            expect(f"{label}: every large product by GEMMul8", 0 < d["calls"] and calls_add_up(r, spec)
                   and d["call_gemmul8"] == d["call_large"] and d["call_gemmul8_large"] == d["call_large"]
                   and (d["products"] == 0 or d.get("gemmul8") == d["products"]),
                   counted + "; " + where(r, spec))
        # the dense eigensolver of the device, and where the exchange matrix was added to the Hamiltonian
        for variant in ("gpu", "host"):
            if variant in present:
                additions = results[f"{system}-{variant}"]["device"].get("additions", 0)
                expect(f"{variant}: the eigensolver of the host", additions == 0,
                       f"exchange matrix added to {additions} Hamiltonians on the device")
        for variant in [v for v in present if v in STEERED]:
            name, label = f"{system}-{variant}", STEERED_LABEL[variant]
            r = results[name]
            d = r["device"]
            if "cpu" in present:
                if converged(name) and converged(f"{system}-cpu"):
                    value = difference(system, variant)
                    expect(f"{label}: Utot minus CPU", abs(value) <= LIMIT, f"{value:+.3e}  limit {LIMIT:.0e}")
                    print(f"  {label + ': Uexx minus CPU':50s} {difference(system, variant, 'Uexx'):+.3e}")
                    first, reference = r["outer"], results[f"{system}-cpu"]["outer"]
                    if first and reference:
                        # the first build takes the density matrix of GGA-PBE: no exchange matrix was added yet
                        print(f"  {label + ': Uexx of build 1 minus CPU':50s} {first[0][2] - reference[0][2]:+.3e}")
                else:
                    expect(f"{label}: Utot minus CPU", False, "a run failed or did not converge")
            detail = (f"exchange matrix added to {d.get('additions', 0)} Hamiltonians on the device; admission declined: "
                      f"{'yes' if d.get('dense_declined') else 'no'}; largest deviation of the collected density "
                      f"matrices {r.get('dm_deviation')}")
            if variant == "dfall":
                expect(f"{label}: host eigensolver", d.get("additions", 0) == 0 and bool(d.get("dense_declined")), detail)
            else:
                expect(f"{label}: addition on the device", 0 < d.get("additions", 0) and not d.get("dense_declined"),
                       detail)
            if variant == "dser":
                expect(f"{label}: serialized mode", bool(d.get("serial")),
                       "the message of the serialized mode " + ("is" if d.get("serial") else "is not") + " in the output")
        # the times of the stages
        print("  time (s)  preparation  density matrices  exchange matrix  of it on the device  addition  DFT in total  wall")
        for variant in present:
            r = results[f"{system}-{variant}"]
            t = r.get("exx_times") or [float("nan")] * 4
            d = use(r)
            on_device = d["seconds"] + d["call_seconds"]
            print(f"  {variant:8s}  {t[0]:11.2f}  {t[1]:16.2f}  {t[2]:15.2f}  {on_device:19.2f}  "
                  f"{t[3]:8.2f}  {r.get('dft_time') or float('nan'):12.2f}  {r['wall']:.1f}")
    bad = [n for n in results if not n.startswith("_") and not converged(n)]
    print("runs that failed or did not converge:", bad if bad else "none")
    print("runs of a listed system that are missing:", missing if missing else "none")
    print("systems without a run:", untouched if untouched else "none")
    print("rows that failed:", failures if failures else "none")
    return 1 if bad or missing or untouched or failures else 0


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
    runner.add_argument("--threads", type=int, default=0,
                        help="threads of a process: passed to the program with -nt and to the BLAS (default: 1)")
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

#!/usr/bin/env python3
"""
exx_force_checks.py -- the runs of milestone M7 (docs/EXX_IMPLEMENTATION_GUIDE.md,
section 9): the forces of HSE06 against the central differences of the total
energy of SCF calculations.

For every system the script runs the structure and, for a few components of
the position of an atom, the structures with that component moved by +h and
-h (h = 0.0005 Angstrom).  The force that the main program prints for the
structure is compared with the central difference -(Utot(+h) - Utot(-h))/(2h).
The displaced structures keep the real-space grid of the structure, as the
force tester of the main program (-forcetest) and the steps of MD.Type do:
without scf.fixed.grid the main program puts the origin of the grid at the
mean position of the atoms, which moves by h/(number of atoms).  The script
runs the structure first and gives the displaced runs its origin (the line
Grid_Origin of its standard output) as scf.fixed.grid.  The same comparison
is made with GGA-PBE, whose forces have no exact exchange: its difference is
the part that the real-space grid, the step h and the stopping of the SCF
leave.  That part reaches 1e-5 Hartree/bohr at 600 Ry and changes with the
position of the atoms relative to the points of the grid, so condition 2 of
M7 bounds the difference of HSE06 minus that of GGA-PBE, on the same
structures and grid, by 1e-5 Hartree/bohr.  Both differences are printed.
Condition 3: the sum over the atoms of the forces of the exact exchange,
which <EXX> prints, vanishes (bound 1e-12 Hartree/bohr).  The displaced
structures must have the neighbour lists of the structure (the numbers of C
and V sites that <EXX> prints): E_x jumps where a list changes.
The forces of 2 MPI processes must be those of 1 within 1e-6 Hartree/bohr: a
mistake in the share of the atoms among the processes changes a force by its
part from the exact exchange (1e-2 here), and 1e-6 is a tenth of the bound of
condition 2.

  python3 tools/exx_force_checks.py list
  python3 tools/exx_force_checks.py run --openmx build/exx-cpu/openmx --out build/exx-m7 [NAME ...]
  python3 tools/exx_force_checks.py table --out build/exx-m7

"run" without a name executes every run that results.json does not hold yet,
--jobs of them at a time (default 1), a displaced run after its structure
(which it adds when results.json does not hold it).  As root in a container,
pass --mpirun "mpirun --allow-run-as-root --oversubscribe".  "table" ends
with status 1 when a bounded difference exceeds its limit or a run that a
bound needs failed or did not converge.  Every run uses one OpenMP thread
(-nt 1).  Only the standard library is used; the parser of the output is the
one of tools/exx_scf_checks.py.

The systems
  ch4      methane with the atoms moved off the symmetric structure, in a cubic
           box of 10 Angstrom, C5.0-s1p1 and H5.0-s1, cluster solver, 600 Ry
  ch3      the methyl radical, bent, with spin polarization, cluster solver,
           600 Ry
  ch4c300, ch3c300
           the same with scf.energycutoff 300 Ry: recorded, not bounded
  dia      diamond with the lattice constant 3.62 Angstrom, the primitive cell
           with the second atom moved, C5.0-s1p1, the band solver with a
           2 x 2 x 2 mesh of k points, a grid of 40 x 40 x 40 points.  The
           lattice constant and the move keep every distance of a C or V site
           0.028 Angstrom or more from the radius of its list.
The stop criteria are scf.criterion 1e-12 and scf.exx.outer.criterion 1e-11:
the central difference divides the energies by 2h = 0.0019 bohr.
The names
  SYSTEM-hse, SYSTEM-pbe   the structure with HSE06 and GGA-PBE
  ...-AX+, ...-AX-         atom A moved along the axis X by +h and -h
  SYSTEM-hse-np2           the structure with 2 MPI processes
  ch4-hse-default, ch4-pbe-default
                           the default stop criteria (scf.criterion 1e-6,
                           scf.exx.outer.criterion 1e-6): their forces
                           minus those of the tight criteria are recorded
"""
import argparse
import concurrent.futures
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exx_scf_checks import parse  # noqa: E402  (the numbers of std.txt and of the .out file)

REPO = Path(__file__).resolve().parents[1]
BOHR = 0.529177210903  # Angstrom
STEP = 0.0005          # Angstrom

# ---- systems --------------------------------------------------------------

BOX = [(10.0, 0.0, 0.0), (0.0, 10.0, 0.0), (0.0, 0.0, 10.0)]
CLUSTER = ["scf.EigenvalueSolver        cluster", "scf.Kgrid                  1 1 1"]
A_DIAMOND = 3.62
SYSTEMS = {
    # species (symbol, basis, pseudopotential), atoms (symbol, x, y, z in
    # Angstrom, initial up and down occupations), cell vectors (Angstrom),
    # input lines of the grid and the solver, the components (atom, axis) to
    # move, and whether condition 2 bounds the differences of HSE06
    "ch4": dict(species=[("C", "C5.0-s1p1", "C_PBE19"), ("H", "H5.0-s1", "H_PBE19")],
                atoms=[("C", 0.03, -0.02, 0.01, 2.0, 2.0), ("H", -0.889981, -0.629312, 0.05, 0.5, 0.5),
                       ("H", 0.04, 0.629312, -0.889981, 0.5, 0.5), ("H", 0.0, 0.709312, 0.889981, 0.5, 0.5),
                       ("H", 0.939981, -0.629312, -0.03, 0.5, 0.5)],
                vectors=BOX, spin="off", solver=["scf.energycutoff           600.0"] + CLUSTER,
                moves=[(5, 0), (1, 1), (3, 2)], bounded=True),
    "ch3": dict(species=[("C", "C5.0-s1p1", "C_PBE19"), ("H", "H5.0-s1", "H_PBE19")],
                atoms=[("C", 0.02, 0.01, 0.12, 2.5, 1.5), ("H", 1.09, 0.03, -0.05, 0.5, 0.5),
                       ("H", -0.55, 0.93, -0.02, 0.5, 0.5), ("H", -0.52, -0.96, 0.04, 0.5, 0.5)],
                vectors=BOX, spin="on", solver=["scf.energycutoff           600.0"] + CLUSTER,
                moves=[(1, 2), (2, 0), (3, 1)], bounded=True),
    "dia": dict(species=[("C", "C5.0-s1p1", "C_PBE19")],
                atoms=[("C", 0.0, 0.0, 0.0, 2.0, 2.0),
                       ("C", A_DIAMOND / 4 + 0.02, A_DIAMOND / 4 - 0.03, A_DIAMOND / 4 + 0.01, 2.0, 2.0)],
                vectors=[(0.0, A_DIAMOND / 2, A_DIAMOND / 2), (A_DIAMOND / 2, 0.0, A_DIAMOND / 2),
                         (A_DIAMOND / 2, A_DIAMOND / 2, 0.0)], spin="off",
                solver=["scf.Ngrid                  40 40 40", "scf.fixed.grid             0.0 0.0 0.0",
                        "scf.EigenvalueSolver        band", "scf.Kgrid                  2 2 2"],
                moves=[(2, 0), (2, 1), (1, 2)], bounded=True),
}
for system in ("ch4", "ch3"):
    SYSTEMS[f"{system}c300"] = dict(SYSTEMS[system], solver=["scf.energycutoff           300.0"] + CLUSTER,
                                    bounded=False)
# the order in which "run" starts the runs that are ready: the slow systems first
PRIORITY = ["ch3", "ch4", "ch3c300", "ch4c300", "dia"]
AXES = "xyz"


def make_input(name, system, xc, move=None, criteria=True):
    s = SYSTEMS[system]
    lines = [
        "System.CurrrentDirectory         ./",
        f"System.Name                     {name}",
        "level.of.stdout                   1",
        "level.of.fileout                  0",
        "DATA.PATH @DATA_PATH@",
        f"Species.Number       {len(s['species'])}",
        "<Definition.of.Atomic.Species"]
    lines += [f" {symbol}   {basis}   {vps}" for symbol, basis, vps in s["species"]]
    lines += ["Definition.of.Atomic.Species>",
              f"Atoms.Number         {len(s['atoms'])}",
              "Atoms.SpeciesAndCoordinates.Unit   Ang",
              "<Atoms.SpeciesAndCoordinates"]
    for i, (symbol, x, y, z, up, down) in enumerate(s["atoms"], 1):
        position = [x, y, z]
        if move and move[0] == i:
            position[move[1]] += move[2]
        lines.append(f" {i}  {symbol}  {position[0]:.10f}  {position[1]:.10f}  {position[2]:.10f}  {up}  {down}")
    lines += ["Atoms.SpeciesAndCoordinates>",
              "Atoms.UnitVectors.Unit             Ang",
              "<Atoms.UnitVectors"]
    lines += [f"  {v[0]:.10f}  {v[1]:.10f}  {v[2]:.10f}" for v in s["vectors"]]
    lines += ["Atoms.UnitVectors>",
              f"scf.XcType                 {xc}",
              f"scf.SpinPolarization        {s['spin']}",
              "scf.ElectronicTemperature  300.0",
              "scf.maxIter                300",
              "scf.Mixing.Type           rmm-diis",
              "scf.Init.Mixing.Weight     0.10",
              "scf.Min.Mixing.Weight      0.001",
              "scf.Max.Mixing.Weight      0.400",
              "scf.Mixing.History          7",
              "scf.Mixing.StartPulay       5",
              "MD.Type                     nomd",
              "MD.maxIter                    1"]
    if criteria:
        lines += ["scf.criterion             1.0e-12",
                  "scf.exx.outer.criterion   1.0e-11"]
    lines += s["solver"]
    return "\n".join(lines) + "\n"


RUNS = {}
COMPARISONS = []  # (system, xc label, atom, axis, base run, run +h, run -h)
PROCESSES = []    # (run with 2 processes, run with 1)
CRITERIA = []     # (run with the default stop criteria, run with the tight ones)


def add(name, system, xc, move=None, ranks=1, criteria=True, base=None):
    # System.Name: ch4-hse-5x+ is ch4_hse_5xp, ch4-hse-5x- is ch4_hse_5xm.
    # base: the run whose grid origin a displaced run keeps (None: the input
    # fixes the grid, or the run is not displaced)
    label = re.sub(r"-$", "m", name.replace("+", "p")).replace("-", "_")
    RUNS[name] = dict(text=make_input(label, system, xc, move, criteria), ranks=ranks, system=system, base=base)


for system, spec in SYSTEMS.items():
    fixed = any(line.startswith("scf.fixed.grid") for line in spec["solver"])
    for label, xc in (("hse", "HSE06"), ("pbe", "GGA-PBE")):
        base = f"{system}-{label}"
        add(base, system, xc)
        for atom, axis in spec["moves"]:
            plus, minus = f"{base}-{atom}{AXES[axis]}+", f"{base}-{atom}{AXES[axis]}-"
            add(plus, system, xc, (atom, axis, STEP), base=None if fixed else base)
            add(minus, system, xc, (atom, axis, -STEP), base=None if fixed else base)
            COMPARISONS.append((system, label, atom, axis, base, plus, minus))
for system in ("ch4", "ch3", "dia"):
    add(f"{system}-hse-np2", system, "HSE06", ranks=2)
    PROCESSES.append((f"{system}-hse-np2", f"{system}-hse"))
for label, xc in (("hse", "HSE06"), ("pbe", "GGA-PBE")):
    add(f"ch4-{label}-default", "ch4", xc, criteria=False)
    CRITERIA.append((f"ch4-{label}-default", f"ch4-{label}"))

# ---- running --------------------------------------------------------------


def forces_of(run_dir, system):
    out = (run_dir / f"{system}.out")
    if not out.exists():
        return None
    text = out.read_text(errors="replace")
    block = re.search(r"<coordinates\.forces\n\s*\d+\n((?:.*\n)*?)coordinates\.forces>", text)
    if not block:
        return None
    return [[float(v) for v in line.split()[5:8]] for line in block.group(1).splitlines() if line.split()]


def grid_origin(run_dir):
    """The origin of the real-space grid (bohr) as the standard output of a run prints it."""
    std = run_dir / "std.txt"
    found = re.search(r"^Grid_Origin\s+(\S+)\s+(\S+)\s+(\S+)", std.read_text(errors="replace"), re.M) \
        if std.exists() else None
    return list(found.groups()) if found else None


def execute(args, name):
    spec = RUNS[name]
    run_dir = args.out / name
    run_dir.mkdir(parents=True, exist_ok=True)
    system = re.search(r"System.Name\s+(\S+)", spec["text"]).group(1)
    text = spec["text"].replace("@DATA_PATH@", str(args.data_path))
    if spec["base"]:
        origin = grid_origin(args.out / spec["base"])
        if origin is None:
            return name, dict(status=-1, wall=0.0, ranks=spec["ranks"], error=f"no grid origin of {spec['base']}")
        text += "scf.fixed.grid             " + "  ".join(origin) + "\n"
    (run_dir / "input.dat").write_text(text)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    start = time.time()
    with open(run_dir / "std.txt", "w") as handle:
        # one OpenMP thread, said in so many words: the program reads -nt and not the environment
        status = subprocess.run([*shlex.split(args.mpirun), "-np", str(spec["ranks"]), str(args.openmx.resolve()),
                                 "input.dat", "-nt", "1"], cwd=run_dir, stdout=handle, stderr=subprocess.STDOUT,
                                env=env).returncode
    result = parse(run_dir, system)
    std = (run_dir / "std.txt").read_text(errors="replace")
    result["forces"] = forces_of(run_dir, system)
    result["grid_origin"] = grid_origin(run_dir)
    exx = re.search(r"<EXX>  forces of the exact exchange in (\S+) s: largest component (\S+), sum (\S+) (\S+) (\S+)",
                    std)
    if exx:
        result["exx_forces"] = dict(seconds=float(exx.group(1)), largest=float(exx.group(2).rstrip(",")),
                                    sum=[float(v) for v in exx.groups()[2:]])
    result.update(status=status, wall=round(time.time() - start, 2), ranks=spec["ranks"])
    return name, result


def run(args):
    args.out.mkdir(parents=True, exist_ok=True)
    store = args.out / "results.json"
    results = json.loads(store.read_text()) if store.exists() else {}
    unknown = [n for n in args.names if n not in RUNS]
    if unknown:
        sys.exit("unknown runs: " + " ".join(unknown))
    names = list(dict.fromkeys(args.names or [n for n in RUNS if n not in results]))
    # a displaced run starts after the run whose grid it keeps
    for name in list(names):
        base = RUNS[name]["base"]
        if base and base not in results and base not in names:
            names.insert(0, base)
    order = {n: (PRIORITY.index(RUNS[n]["system"]), i) for i, n in enumerate(RUNS)}
    waiting = {}
    ready = []
    for name in names:
        base = RUNS[name]["base"]
        if base in names:
            waiting.setdefault(base, []).append(name)
        else:
            ready.append(name)
    running = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        while ready or running:
            ready.sort(key=order.get)
            while ready and len(running) < args.jobs:
                name = ready.pop(0)
                running[pool.submit(execute, args, name)] = name
            done, _ = concurrent.futures.wait(running, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                del running[future]
                name, r = future.result()
                results[name] = r
                store.write_text(json.dumps(results, indent=1))
                if "error" in r:
                    print(f"{name:22s} not started: {r['error']}", flush=True)
                else:
                    print(f"{name:22s} exit {r['status']} {r['wall']:7.1f}s Utot {r['Utot']} Uexx {r['Uexx']} "
                          f"builds {r['builds']} steps {r['scf_steps']} converged {r['achieved']}", flush=True)
                ready += waiting.pop(name, [])
    return 0

# ---- table ----------------------------------------------------------------


def table(args):
    results = json.loads((args.out / "results.json").read_text())
    failures, missing = [], []

    def hybrid(name):
        return name.split("-")[1] == "hse"

    def converged(name):
        r = results.get(name)
        return (bool(r) and r["status"] == 0 and r.get("achieved") and r.get("Utot") is not None
                and r.get("forces") is not None and (not hybrid(name) or r["outer_converged"]))

    def show(label, value, limit):
        if limit is None:
            print(f"  {label:58s} {value:+.3e}  record")
            return
        ok = abs(value) <= limit
        print(f"  {label:58s} {value:+.3e}  limit {limit:.0e}  {'ok' if ok else 'FAIL'}")
        if not ok:
            failures.append(label)

    def largest(a, b):
        return max(abs(x - y) for fa, fb in zip(a, b) for x, y in zip(fa, fb))

    print(f"== forces against the central differences of Utot (Hartree/bohr; h = {STEP} Angstrom)")
    differences = {}
    for system, label, atom, axis, base, plus, minus in COMPARISONS:
        names = (base, plus, minus)
        bad = [n for n in names if not converged(n)]
        if bad:
            if SYSTEMS[system]["bounded"]:
                missing += bad
            print(f"  {base} atom {atom} {AXES[axis]}: runs missing or not converged {bad}")
            continue
        b, p, m = (results[n] for n in names)
        analytic = b["forces"][atom - 1][axis]
        numeric = -(p["Utot"] - m["Utot"]) / (2.0 * STEP / BOHR)
        differences[(system, label, atom, axis)] = analytic - numeric
        print(f"  {base:12s} atom {atom} {AXES[axis]}: force {analytic:+.9f}, "
              f"difference of the energies {numeric:+.9f}")
        show(f"{base} atom {atom} {AXES[axis]}: force minus difference", analytic - numeric, None)
        for n in (plus, minus):
            if results[n].get("sizes") != b.get("sizes"):
                show(f"{n}: neighbour lists {results[n].get('sizes')}, not {b.get('sizes')}", 1.0, 0.0)
            if results[n].get("grid_origin") != b.get("grid_origin"):
                show(f"{n}: grid origin {results[n].get('grid_origin')}, not {b.get('grid_origin')}", 1.0, 0.0)

    # Condition 2 compares HSE06 with GGA-PBE on the same grid: the difference
    # of GGA-PBE, whose forces have no exact exchange, is the part that the
    # real-space grid leaves, and it changes with the position of the atoms
    # relative to the points of the grid.
    print("== condition 2: the difference of HSE06 minus that of GGA-PBE, same structures and grid (Hartree/bohr)")
    for system, label, atom, axis, base, plus, minus in COMPARISONS:
        if label != "hse" or not SYSTEMS[system]["bounded"]:
            continue
        hse, pbe = differences.get((system, "hse", atom, axis)), differences.get((system, "pbe", atom, axis))
        if hse is None or pbe is None:
            print(f"  {system} atom {atom} {AXES[axis]}: a difference is missing")
            continue
        show(f"{system} atom {atom} {AXES[axis]}: {hse:+.2e} minus {pbe:+.2e}", hse - pbe, 1e-5)

    print("== sums over the atoms of the forces of the exact exchange (Hartree/bohr)")
    for name in RUNS:
        if not hybrid(name) or not converged(name):
            continue
        e = results[name].get("exx_forces")
        if not e:
            show(f"{name}: no line of the forces of the exact exchange", 1.0, 0.0)
            continue
        show(f"{name}: largest component of the sum", max(abs(v) for v in e["sum"]), 1e-12)

    print("== forces of 2 MPI processes against those of 1 (Hartree/bohr)")
    for name, base in PROCESSES:
        if not converged(name) or not converged(base):
            missing += [n for n in (name, base) if not converged(n)]
            print(f"  {name}: runs missing or not converged")
            continue
        show(f"{name}: largest difference", largest(results[name]["forces"], results[base]["forces"]), 1e-6)

    print("== records (Hartree/bohr; no limit)")
    for name, base in CRITERIA:
        if converged(name) and converged(base):
            show(f"{name}: largest difference from the tight criteria",
                 largest(results[name]["forces"], results[base]["forces"]), None)
    # the real-space grid breaks the invariance under a translation of all atoms
    for name in RUNS:
        if name.count("-") == 1 and converged(name):
            total = [sum(f[x] for f in results[name]["forces"]) for x in range(3)]
            show(f"{name}: largest component of the sum of all forces", max(total, key=abs), None)

    print("== sizes and times")
    for name in (n for n in RUNS if n in results):
        r = results[name]
        e = r.get("exx_forces") or {}
        if "error" in r:
            print(f"  {name:22s} not started: {r['error']}")
            continue
        print(f"  {name:22s} np{r['ranks']} {r['wall']:7.1f} s  {r['builds']} builds, {r['scf_steps']} SCF steps  "
              f"atoms, orbitals, ABFs, C sites, V sites {r.get('sizes')}  EXX forces {e.get('seconds')} s")

    print("runs that a bound needs and that failed or did not converge:", sorted(set(missing)) or "none")
    print("bounded differences beyond their limit:", failures if failures else "none")
    return 1 if missing or failures else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="names of the runs")
    runner = commands.add_parser("run", help="execute runs and store their numbers")
    runner.add_argument("--openmx", type=Path, required=True, help="the executable of the main program")
    runner.add_argument("--out", type=Path, required=True, help="directory of the runs and of results.json")
    runner.add_argument("--mpirun", default="mpirun", help="launcher with its options (default: mpirun)")
    runner.add_argument("--data-path", type=Path, default=REPO / "DFT_DATA19", help="DATA.PATH of the inputs")
    runner.add_argument("--jobs", type=int, default=1, help="runs at the same time (default 1)")
    runner.add_argument("names", nargs="*", help="runs to execute (default: those not in results.json)")
    tabler = commands.add_parser("table", help="compare the stored numbers")
    tabler.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "list":
        for name, spec in RUNS.items():
            print(f"{name:22s} np{spec['ranks']}")
        return 0
    if args.command == "run":
        return run(args)
    return table(args)


if __name__ == "__main__":
    sys.exit(main())

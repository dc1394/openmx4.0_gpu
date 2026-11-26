#!/usr/bin/env python3
"""
exx_output_checks.py -- the runs of milestone M8 for the outputs after the SCF
(docs/EXX_IMPLEMENTATION_GUIDE.md, section 9): the band dispersion and the DOS
of the band solver with HSE06 and PBE0. They build H(k) from the Hamiltonian
of the neighbour list at k points of their own and add K(k) to it (section
6.5 of the guide).

  python3 tools/exx_output_checks.py list
  python3 tools/exx_output_checks.py run --openmx build/exx-cpu/openmx --out build/exx-m8out [NAME ...]
  python3 tools/exx_output_checks.py table --out build/exx-m8out

"run" without a name executes every run that results.json does not hold yet.
As root in a container, pass --mpirun "mpirun --allow-run-as-root --oversubscribe".
"table" ends with status 1 when a difference exceeds its limit or a run of a
comparison failed or did not converge.  The inputs are those of
tools/exx_kpoint_checks.py, with its stop criteria per primitive cell, and
every run uses one OpenMP thread (-nt 1).  Only the standard library is used.

The checks
  supercell  The bands of a supercell of c1 x c2 x c3 cells at the k point K
             are the bands of the primitive cell at the points (K + m)/c,
             0 <= m_i < c_i.  The path of the supercell runs between two
             points of no mesh, and the primitive cell takes one path for
             every m.  At every point the sorted eigenvalues of the supercell
             are compared with the sorted union of those of the primitive
             cell.  The two SCFs sample the same k points, so the difference
             is that of two converged SCFs; GGA-PBE shows it without the
             exact exchange.  The supercell folds its V sites by its own mesh,
             2 x 2 x 2, and the primitive cell by 4 x 4 x 4.
  mesh       A path through points of the SCF mesh: the bands there are
             compared with the eigenvalues of the last SCF step in the .out
             file.  The last build of K comes after that step, so the two
             differ by one outer iteration, below the outer criterion.  At
             level.of.stdout 2 the program compares K(k) of the outputs with
             K(k) of the SCF at the points of the mesh; the table takes the
             largest difference relative to the largest element.  The k
             points of OpenMX lie off the mesh by Shift_K_Point (1e-12), so
             D of a block from another cell of its class carries the phase
             2 pi delta.nT (section 6.5 of the guide); the limit 1e-9 is
             ten times its bound 1.3e-10.  Without the shift the difference
             is 2e-15.
  processes  The bands with 2 and 4 MPI processes against 1.
  dos        Dos.fileout and DosGauss.fileout with Dos.Kgrid 3 x 3 x 3, off
             the mesh 4 x 4 x 4 of the SCF: the eigenvalues of the .Dos.val
             file (Dos.fileout only, six decimals in Hartree) and of the
             .FermiSurf0.bxsf file (five decimals in eV) against the bands at
             the same 27 points, which the run computes as well.  The DOS
             files thus add the K(k) of the band dispersion; the supercell
             check shows that this K(k) holds off the mesh.  The tetrahedron
             DOS writes to the .bxsf file only the bands near the chemical
             potential, each under its number.
  cluster    The DOS of the cluster solver with Dos.Kgrid 1 1 1, which adds K
             as the SCF does, against the eigenvalues of the .out file.
  refusals   Inputs that the program has to refuse with a hybrid functional:
             the message has to name the keyword.

The names
  SYSTEM-pABC-XC-band     the primitive cell with the mesh A x B x C and the
                          paths of the supercell check
  SYSTEM-sABCkDEF-XC-band the supercell A x B x C with the mesh D x E x F
  -reg                    scf.Generation.Kpoint REGULAR (no Gamma point)
  XC                      pbe (GGA-PBE), hse (HSE06), pbe0 (PBE0)
  -mesh, -dos, -dosgauss, -cluster-dos, -npN, reject-...  the checks above
"""
import argparse
import itertools
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exx_kpoint_checks import PBE0_INNER, SYSTEMS, make_input  # noqa: E402
from exx_scf_checks import parse  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
EV = 27.2113845  # eV2Hartree of source/openmx_common.h: the program writes eV with it
XC = {"pbe": "GGA-PBE", "hse": "HSE06", "pbe0": "PBE0"}

# two points of no mesh for the path of a supercell, in units of its reciprocal vectors
PATH_START = (0.13, 0.27, 0.41)
PATH_END = (0.37, 0.11, 0.29)
POINTS = 4


def band_input(segments):
    """The keywords of a band dispersion through the segments (start, end, points)."""
    lines = ["Band.dispersion on", f"Band.Nkpath {len(segments)}", "<Band.kpath"]
    for n, (start, end, points) in enumerate(segments, 1):
        lines.append(f"  {points}  {start[0]:.15f} {start[1]:.15f} {start[2]:.15f}"
                     f"  {end[0]:.15f} {end[1]:.15f} {end[2]:.15f}  P{n} Q{n}")
    lines.append("Band.kpath>")
    return lines


def unfolded_segments(cells):
    """The paths of the primitive cell: (K + m)/c for every m, from PATH_START to PATH_END."""
    segments = []
    for m in itertools.product(*(range(c) for c in cells)):
        start = tuple((PATH_START[i] + m[i]) / cells[i] for i in range(3))
        end = tuple((PATH_END[i] + m[i]) / cells[i] for i in range(3))
        segments.append((start, end, POINTS))
    return segments


RUNS = {}


def add(name, system, cells, kgrid, xc, ranks=1, extra=(), **options):
    if xc == "PBE0":
        options.setdefault("inner", PBE0_INNER)
    text = make_input(name.replace("-", "_"), system, cells, kgrid, xc, extra=list(extra), **options)
    RUNS[name] = dict(text=text, ranks=ranks, cells=list(cells), kgrid=list(kgrid), system=system)


# (label, supercell run, primitive run, cells)
SUPERCELL = []
for system, primitive, supercell, cells, generation, xcs in (
        ("dia", (4, 4, 4), (2, 2, 2), (2, 2, 2), None, ("pbe", "hse", "pbe0")),
        ("dia", (4, 4, 4), (2, 2, 2), (2, 2, 2), "REGULAR", ("hse",)),
        ("hfm", (4, 1, 1), (2, 1, 1), (2, 1, 1), "REGULAR", ("hse",))):
    tag = "-reg" if generation else ""
    for xc in xcs:
        p = f"{system}-p{''.join(map(str, primitive))}{tag}-{xc}-band"
        s = f"{system}-s{''.join(map(str, cells))}k{''.join(map(str, supercell))}{tag}-{xc}-band"
        add(p, system, (1, 1, 1), primitive, XC[xc], generation=generation,
            extra=band_input(unfolded_segments(cells)))
        add(s, system, cells, supercell, XC[xc], generation=generation,
            extra=band_input([(PATH_START, PATH_END, POINTS)]))
        SUPERCELL.append((f"{s} against {p}", s, p, cells))

# paths through the points of the mesh 3 x 3 x 3
MESH_PATHS = [((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 4), ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 4),
              ((0.0, 0.0, 0.0), (1.0, 2.0, 0.0), 4)]
MESH = []
for xc in ("hse", "pbe0"):
    name = f"dia-p333-{xc}-mesh"
    add(name, "dia", (1, 1, 1), (3, 3, 3), XC[xc], extra=band_input(MESH_PATHS))
    MESH.append(name)

# the band dispersion of the supercell check with more processes
PROCESSES = []
for ranks in (2, 4):
    name = f"dia-p444-hse-band-np{ranks}"
    add(name, "dia", (1, 1, 1), (4, 4, 4), "HSE06", ranks=ranks, extra=band_input(unfolded_segments((2, 2, 2))))
    PROCESSES.append((name, "dia-p444-hse-band"))

# the DOS on a mesh 3 x 3 x 3, and the bands at its 27 points
DOS_GRID = (3, 3, 3)
DOS_PATHS = [((0.0, j / 3.0, k / 3.0), (2.0 / 3.0, j / 3.0, k / 3.0), 3) for j in range(3) for k in range(3)]
DOS = []
for kind, keyword in (("dos", "Dos.fileout on"), ("dosgauss", "DosGauss.fileout on")):
    name = f"dia-p444-hse-{kind}"
    add(name, "dia", (1, 1, 1), (4, 4, 4), "HSE06",
        extra=[keyword, "Dos.Kgrid 3 3 3", "Dos.Erange -40.0 40.0"] + band_input(DOS_PATHS))
    DOS.append(name)

# the DOS of the cluster solver
add("dia-s222-hse-cluster-dos", "dia", (2, 2, 2), (1, 1, 1), "HSE06",
    extra=["Dos.fileout on", "Dos.Kgrid 1 1 1", "Dos.Erange -40.0 40.0"])

# inputs that the program has to refuse with a hybrid functional, and the keyword of the message
REFUSALS = []
for name, cells, kgrid, extra, message in (
        ("reject-band-cluster", (1, 1, 1), (1, 1, 1), band_input(MESH_PATHS[:1]),
         "Band.dispersion with the cluster solver"),
        ("reject-band-lno", (1, 1, 1), (2, 2, 2), ["LNO.flag on"] + band_input(MESH_PATHS[:1]),
         "Band.dispersion with LNO.flag"),
        ("reject-dos-cluster", (1, 1, 1), (1, 1, 1), ["Dos.fileout on", "Dos.Kgrid 2 2 2"],
         "Dos.fileout or DosGauss.fileout with the cluster solver and Dos.Kgrid other than 1 1 1"),
        ("reject-cohp", (1, 1, 1), (2, 2, 2),
         ["Dos.fileout on", "COHP.calc.flag on", "COHP.num.pairs 1", "<COHP.pairs", "  1  1 2  0 0 0", "COHP.pairs>"],
         "COHP.calc.flag")):
    add(name, "dia", cells, kgrid, "HSE06", extra=extra)
    REFUSALS.append((name, message))


# ---- output files ---------------------------------------------------------

def read_bands(path):
    """The .Band file: a list of (k, [eigenvalues of every spin channel]), Hartree."""
    if not path.exists():
        return None
    lines = path.read_text().split("\n")
    n, spin, _ = lines[0].split()
    channels = int(spin) + 1
    paths = int(lines[2])
    position = 3 + paths
    points = []
    while position < len(lines) and lines[position].strip():
        k, values = None, []
        for _ in range(channels):
            head = lines[position].split()
            k = tuple(float(v) for v in head[1:4])
            values.append([float(v) for v in lines[position + 1].split()])
            position += 2
        points.append((k, values))
    return points


def read_out_eigenvalues(path):
    """The eigenvalues of the last SCF step at the k points of the .out file: (k, up, down)."""
    if not path.exists():
        return []
    text = path.read_text(errors="replace")
    found = re.findall(r"kloop=\d+\s*\n\s*k1=\s*(\S+)\s+k2=\s*(\S+)\s+k3=\s*(\S+)\s*\n\s*\n"
                       r"((?:\s*\d+\s+\S+\s+\S+[ \t]*\n)+)", text)
    points = []
    for k1, k2, k3, rows in found:
        values = [line.split() for line in rows.strip().splitlines()]
        points.append(((float(k1), float(k2), float(k3)), [float(v[1]) for v in values],
                       [float(v[2]) for v in values]))
    return points


def read_dos_values(path):
    """The .Dos.val file: grid, the first state of the window, and {(i, j, k): [eigenvalues per spin]}."""
    if not path.exists():
        return None
    text = path.read_text()
    grid = tuple(int(v) for v in re.search(r"^Kgrid\s+(\d+)\s+(\d+)\s+(\d+)", text, flags=re.M).groups())
    first = int(re.search(r"^irange\s+(-?\d+)\s+(-?\d+)", text, flags=re.M).group(1))
    spins = int(re.search(r"^Nspin\s+(\d+)", text, flags=re.M).group(1)) + 1
    body = text.split("<Eigenvalues\n", 1)[1].split("Eigenvalues>", 1)[0].strip().splitlines()
    values = {}
    for row, line in enumerate(body):
        fields = line.split()
        key = tuple(int(v) for v in fields[:3])
        values.setdefault(key, [None] * spins)[row % spins] = [float(v) for v in fields[3:]]
    return dict(grid=grid, first=first, values=values)


def read_bxsf(path):
    """The .FermiSurf0.bxsf file: the grid, the numbers of its bands and
    {(i, j, k): [eigenvalues]}, eV. The tetrahedron DOS writes only the bands
    near the chemical potential, each under its own number (BAND:  i1)."""
    if not path.exists():
        return None
    lines = path.read_text().split("\n")
    start = next(i for i, line in enumerate(lines) if line.startswith("BEGIN_BANDGRID_3D_simple_example"))
    bands = int(lines[start + 1])
    size = tuple(int(v) for v in lines[start + 2].split())
    labels, numbers = [], []
    position = start + 7
    for _ in range(bands):
        assert lines[position].startswith("BAND:")
        labels.append(int(lines[position].split()[1]))
        position += 1
        band = []
        while len(band) < size[0] * size[1] * size[2]:
            band += [float(v) for v in lines[position].split()]
            position += 1
        numbers.append(band)
    grid = tuple(s - 1 for s in size)
    values = {}
    for i, j, k in itertools.product(range(grid[0]), range(grid[1]), range(grid[2])):
        index = (i * size[1] + j) * size[2] + k
        values[(i, j, k)] = [numbers[b][index] for b in range(bands)]
    return dict(grid=grid, labels=labels, values=values)


# ---- runs -----------------------------------------------------------------

def execute(args, name):
    spec = RUNS[name]
    run_dir = args.out / name
    run_dir.mkdir(parents=True, exist_ok=True)
    system = re.search(r"System.Name\s+(\S+)", spec["text"]).group(1)
    (run_dir / "input.dat").write_text(spec["text"].replace("@DATA_PATH@", str(args.data_path)))
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    start = time.time()
    with open(run_dir / "std.txt", "w") as handle:
        # one OpenMP thread, said in so many words: the program reads -nt and not the environment
        status = subprocess.run([*shlex.split(args.mpirun), "-np", str(spec["ranks"]), str(args.openmx.resolve()),
                                 "input.dat", "-nt", "1"], cwd=run_dir, stdout=handle, stderr=subprocess.STDOUT,
                                env=env).returncode
    result = parse(run_dir, system)
    std = (run_dir / "std.txt").read_text(errors="replace")
    refused = re.search(r"^scf\.XcType=\S+ does not support\n((?:  \S.*\n)+)", std, flags=re.M)
    result["refused"] = [line.strip() for line in refused.group(1).splitlines()] if refused else None
    check = re.search(r"<EXX>  K\(k\) of the outputs at the (\d+) points of the mesh: largest difference from "
                      r"K\(k\) of the SCF (\S+), largest element (\S+)", std)
    if check:
        result["mesh_check"] = [int(check.group(1)), float(check.group(2).rstrip(",")), float(check.group(3))]
    outputs = re.search(r"<EXX>  outputs after the SCF: (.*)", std)
    result["outputs"] = outputs.group(1) if outputs else None
    result["bands"] = read_bands(run_dir / f"{system}.Band")
    result["out_eigenvalues"] = read_out_eigenvalues(run_dir / f"{system}.out")
    result["dos_values"] = None
    dos = read_dos_values(run_dir / f"{system}.Dos.val")
    if dos and dos["values"] and dos["first"] >= 1:
        result["dos_values"] = dict(grid=dos["grid"], first=dos["first"],
                                    values=[[list(k), v] for k, v in dos["values"].items()])
    bxsf = read_bxsf(run_dir / f"{system}.FermiSurf0.bxsf")
    result["bxsf"] = dict(grid=bxsf["grid"], labels=bxsf["labels"],
                          values=[[list(k), v] for k, v in bxsf["values"].items()]) if bxsf else None
    result.update(status=status, wall=round(time.time() - start, 2), ranks=spec["ranks"])
    return result


def run(args):
    args.out.mkdir(parents=True, exist_ok=True)
    store = args.out / "results.json"
    results = json.loads(store.read_text()) if store.exists() else {}
    unknown = [n for n in args.names if n not in RUNS]
    if unknown:
        sys.exit("unknown runs: " + " ".join(unknown))
    for name in args.names or [n for n in RUNS if n not in results]:
        r = results[name] = execute(args, name)
        store.write_text(json.dumps(results))
        print(f"{name:32s} exit {r['status']} {r['wall']:7.1f}s Utot {r['Utot']} Uexx {r['Uexx']} "
              f"builds {r['builds']} converged {r['achieved']} outputs {r['outputs']}", flush=True)
    return 0


# ---- table ----------------------------------------------------------------

def same_point(a, b, tolerance=1.0e-6):
    """k points that differ by a vector of the reciprocal lattice, up to a sign (k and -k)."""
    for sign in (1.0, -1.0):
        if all(abs((sign * x - y) - round(sign * x - y)) < tolerance for x, y in zip(a, b)):
            return True
    return False


def table(args):
    results = json.loads((args.out / "results.json").read_text())
    failures, bad = [], []
    missing = [n for n in RUNS if n not in results]

    def hybrid(name):
        return "-hse" in name or "-pbe0" in name

    def usable(name):
        r = results.get(name)
        if r is None:
            return False
        ok = r["status"] == 0 and r["achieved"] and (not hybrid(name) or r["outer_converged"])
        if not ok and name not in bad:
            bad.append(name)
        return ok

    def show(label, value, limit):
        ok = value is not None and abs(value) <= limit
        shown = f"{value:.3e}" if value is not None else "none"
        print(f"  {label:78s} {shown}  limit {limit:.0e}  {'ok' if ok else 'FAIL'}")
        if not ok:
            failures.append(label)

    print("== supercell: the bands of the supercell at K against those of the primitive cell at (K + m)/c "
          "(Hartree)")
    for label, s, p, cells in SUPERCELL:
        if not (usable(s) and usable(p)) or not results[s]["bands"] or not results[p]["bands"]:
            continue
        sb, pb = results[s]["bands"], results[p]["bands"]
        count = cells[0] * cells[1] * cells[2]
        largest = 0.0
        for j, (k, values) in enumerate(sb):
            for spin, energies in enumerate(values):
                union = sorted(e for m in range(count) for e in pb[m * POINTS + j][1][spin])
                largest = max(largest, max(abs(a - b) for a, b in zip(sorted(energies), union)))
        limit = 1e-6
        show(f"{label} ({len(sb)} points)", largest, limit)

    print("== mesh: the bands at points of the SCF mesh against the eigenvalues of the .out file (Hartree), and")
    print("   K(k) of the outputs against K(k) of the SCF on the mesh (relative to the largest element)")
    for name in MESH:
        if not usable(name) or not results[name]["bands"]:
            continue
        r = results[name]
        largest, compared = 0.0, 0
        for k, values in r["bands"]:
            for q, up, down in r["out_eigenvalues"]:
                if same_point(k, q, 1.0e-4):
                    for spin, energies in enumerate(values):
                        reference = up if spin == 0 else down
                        largest = max(largest, max(abs(a - b) for a, b in zip(energies, reference)))
                    compared += 1
                    break
        show(f"{name}: {compared} of {len(r['bands'])} points of the path on the listed mesh",
             largest if compared else None, 1e-8)
        check = r.get("mesh_check")
        # Shift_K_Point: see the docstring
        show(f"{name}: K(k) at the {check[0] if check else 0} points of the mesh",
             check[1] / check[2] if check else None, 1e-9)

    print("== processes: the bands with more MPI processes against one (Hartree)")
    for name, base in PROCESSES:
        if not (usable(name) and usable(base)) or not results[name]["bands"] or not results[base]["bands"]:
            continue
        largest = max(abs(a - b) for (_, x), (_, y) in zip(results[name]["bands"], results[base]["bands"])
                      for u, v in zip(x, y) for a, b in zip(u, v))
        show(f"{name} against {base}", largest, 1e-8)

    print("== dos: the eigenvalues of the DOS files against the bands at the same points")
    for name in DOS:
        if not usable(name) or not results[name]["bands"]:
            continue
        r = results[name]
        bands = r["bands"]

        def band_at(k):
            for q, values in bands:
                if same_point(k, q):
                    return values
            return None

        if r.get("dos_values"):
            d = r["dos_values"]
            largest, compared = 0.0, 0
            for key, values in d["values"]:
                reference = band_at(tuple(key[i] / d["grid"][i] for i in range(3)))
                if reference is None:
                    continue
                for spin, energies in enumerate(values):
                    window = reference[spin][d["first"] - 1:d["first"] - 1 + len(energies)]
                    largest = max(largest, max(abs(a - b) for a, b in zip(energies, window)))
                compared += 1
            show(f"{name}: .Dos.val at {compared} points (Hartree, six decimals)", largest if compared else None,
                 2e-6)
        if r.get("bxsf"):
            b = r["bxsf"]
            largest, compared = 0.0, 0
            for key, energies in b["values"]:
                reference = band_at(tuple(key[i] / b["grid"][i] for i in range(3)))
                if reference is None:
                    continue
                largest = max(largest, max(abs(a - reference[0][band - 1] * EV)
                                           for a, band in zip(energies, b["labels"])))
                compared += 1
            show(f"{name}: .FermiSurf0.bxsf at {compared} points (eV, five decimals)", largest if compared else None,
                 2e-5)
        else:
            show(f"{name}: .FermiSurf0.bxsf", None, 2e-5)

    print("== cluster: the DOS of the cluster solver against the eigenvalues of the .out file (Hartree)")
    name = "dia-s222-hse-cluster-dos"
    if usable(name):
        r = results[name]
        d = r.get("dos_values")
        g = (r.get("eigen_up"), r.get("eigen_down"))
        if d and g[0]:
            largest = 0.0
            for key, values in d["values"]:
                for spin, energies in enumerate(values):
                    window = g[spin][d["first"] - 1:d["first"] - 1 + len(energies)]
                    largest = max(largest, max(abs(a - b) for a, b in zip(energies, window)))
            show(f"{name}: .Dos.val (six decimals)", largest, 2e-6)
        else:
            show(f"{name}: .Dos.val and the eigenvalues of the .out file", None, 2e-6)

    print("== refusals")
    for name, message in REFUSALS:
        r = results.get(name)
        if r is None:
            continue
        ok = r["status"] == 0 and r["scf_steps"] is None and message in (r.get("refused") or [])
        print(f"  {name:24s} message names {r.get('refused') if r else None}  {'ok' if ok else 'FAIL'}")
        if not ok:
            failures.append(name)

    print("== times (s)")
    for name, r in results.items():
        print(f"  {name:32s} np{r['ranks']} {r['wall']:8.1f}  outputs: {r.get('outputs')}")

    print("runs not in results.json:", missing if missing else "none")
    print("runs that failed or did not converge:", bad if bad else "none")
    print("differences beyond their limit:", failures if failures else "none")
    return 1 if missing or bad or failures else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="names of the runs")
    runner = commands.add_parser("run", help="execute runs and store their numbers")
    runner.add_argument("--openmx", type=Path, required=True, help="the executable of the main program")
    runner.add_argument("--out", type=Path, required=True, help="directory of the runs and of results.json")
    runner.add_argument("--mpirun", default="mpirun", help="launcher with its options (default: mpirun)")
    runner.add_argument("--data-path", type=Path, default=REPO / "DFT_DATA19", help="DATA.PATH of the inputs")
    runner.add_argument("names", nargs="*", help="runs to execute (default: every run not done yet)")
    summary = commands.add_parser("table", help="print the table of the checks from results.json")
    summary.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "list":
        print("\n".join(RUNS))
        return 0
    return run(args) if args.command == "run" else table(args)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
exx_scf_checks.py -- the runs of the acceptance table of milestone M4
(docs/EXX_IMPLEMENTATION_GUIDE.md, section 9): HSE06 at the Gamma point with the
cluster solver, and the same table for PBE0 with the truncated Coulomb
interaction (M8).

The script writes the inputs, runs the main program and collects the numbers of
the standard output and of the .out file into results.json.  A second command
prints the table: the differences that the guide bounds are compared with their
limits, the others are listed for the progress file.

  python3 tools/exx_scf_checks.py list
  python3 tools/exx_scf_checks.py run --openmx build/exx-cpu/openmx --out build/exx-m4 [NAME ...]
  python3 tools/exx_scf_checks.py table --out build/exx-m4

"run" without a name executes every run that results.json does not hold yet.
As root in a container, pass --mpirun "mpirun --allow-run-as-root --oversubscribe".
"table" ends with status 1 when a bounded difference exceeds its limit or a
run did not converge.  The runs of the two rows that the guide only records
(scf.exx.abf.pca and scf.exx.v.cutoff away from their defaults) are exempt:
one that does not converge is listed, and the status stays 0.  The same
holds for the PBE0 runs that change scf.exx.abf.pca or
scf.exx.coulomb.cutoff, set scf.exx.v.cutoff to 8 or 12 bohr, or put the
molecule in the cell of 20 Angstrom; -pbe0-v24 has a limit and is not
exempt.  The energies of the runs with the default scf.criterion are listed
without a limit.  Only the standard library is used.

Threads.  Every run uses one OpenMP thread: the program takes that number
from the option -nt, not from OMP_NUM_THREADS, and the runs pass -nt 1.  The
runs with one MPI process set OPENBLAS_NUM_THREADS to 2, the others to 1; that
is the number "threads" in results.json.  With one OpenMP thread a run gives
the same numbers every time on the machine of the progress file, which is
what the comparison of two executables needs.

The systems
  ch4, h2o       the molecules of work/input_example/Methane.dat and H2O.dat with the
                 basis sets of those files, in a cubic cell of 10 Angstrom
  ch4dzp, h2odzp the same molecules with C6.0-s2p2d1, O6.0-s2p2d1 and H6.0-s2p1
  o2, o2dzp      the triplet of the oxygen molecule
  dia8, dia8dzp  the cubic cell of diamond with 8 atoms: the cell is smaller than
                 the range of the neighbour lists, so every atom meets its own
                 periodic images
The runs of a family
  -pbe                 GGA-PBE
  -hse                 HSE06 with the default keywords
  -hse-alpha0          scf.exx.alpha 0: has to give the energy of GGA-PBE
  -hse-shift           the molecule moved by (3, 5, -7) spacings of the real-space grid
  -hse-spin            scf.SpinPolarization on for the closed shell
  -hse-np2, -hse-np4   2 and 4 MPI processes (also -hse-spin-np2, -hse-spin-np4)
  -hse-pca3, 5, 6      scf.exx.abf.pca 1e-3, 1e-5, 1e-6
  -hse-v6, 12, 16      scf.exx.v.cutoff 6, 12, 16 bohr
  -pbe-c6, -hse-c6     the default scf.criterion (1e-6) instead of 1e-10
  -big-hse-v6 ... v16  the molecule in a cell of 20 Angstrom, where no periodic
                       image is within the range of the Coulomb matrix
  -hse-mix-M           scf.Mixing.Type M instead of rmm-diis (ch4, h2o, dia8)
  -hse-mix-M-p10       M with scf.Mixing.StartPulay 10 instead of 5: the other
                       way in which the mixing starts again after a build
  -hse-mix-M-c6        M with the default scf.criterion
  -pbe-mix-M-c6        GGA-PBE with Simple or Kerker mixing: these two mixers end
                       at a residual of about 1e-5, with and without the exact
                       exchange, and their rows are compared with each other
The PBE0 runs of a family (M8)
  -pbe0                PBE0 with the default keywords: R_c from the volume of
                       the cell (scf.exx.coulomb.cutoff absent) and the V
                       neighbours up to the reach of the kernel
                       (scf.exx.v.cutoff absent).  Every PBE0 run has
                       scf.criterion 1e-12 and scf.exx.outer.criterion 1e-9
                       (PBE0_CRITERION)
  -pbe-c12             GGA-PBE with scf.criterion 1e-12, for -pbe0-alpha0
  -pbe0-alpha0, -pbe0-shift, -pbe0-spin, -pbe0-np2, -pbe0-np4,
  -pbe0-spin-np2, -pbe0-spin-np4, -pbe0-pca3, 5, 6
                       as for HSE06
  -pbe0-v8, -pbe0-v12  scf.exx.v.cutoff 8 and 12 bohr, shorter than the reach
  -pbe0-v24            scf.exx.v.cutoff 24 bohr, beyond the reach: the blocks
                       that it adds vanish, so the energy is that of -pbe0
  -pbe0-rc8, -pbe0-rc16  scf.exx.coulomb.cutoff 8 and 16 bohr
  -big-pbe0            the molecule in a cell of 20 Angstrom: R_c doubles
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

REPO = Path(__file__).resolve().parents[1]
HARTREE_EV = 27.211386

# ---- systems --------------------------------------------------------------

CH4_ATOMS = [("C", 0.0, 0.0, 0.0, 2.0, 2.0),
             ("H", -0.889981, -0.629312, 0.0, 0.5, 0.5),
             ("H", 0.0, 0.629312, -0.889981, 0.5, 0.5),
             ("H", 0.0, 0.629312, 0.889981, 0.5, 0.5),
             ("H", 0.889981, -0.629312, 0.0, 0.5, 0.5)]
H2O_ATOMS = [("O", 0.0, 0.0, 0.0, 3.0, 3.0),
             ("H", 0.757, 0.586, 0.0, 0.5, 0.5),
             ("H", -0.757, 0.586, 0.0, 0.5, 0.5)]
O2_ATOMS = [("O", 0.0, 0.0, 0.0, 4.0, 2.0),
            ("O", 0.0, 0.0, 1.2075, 4.0, 2.0)]
A_DIAMOND = 3.567
DIAMOND_ATOMS = [("C", A_DIAMOND * x, A_DIAMOND * y, A_DIAMOND * z, 2.0, 2.0) for x, y, z in
                 [(0, 0, 0), (0, .5, .5), (.5, 0, .5), (.5, .5, 0),
                  (.25, .25, .25), (.25, .75, .75), (.75, .25, .75), (.75, .75, .25)]]

# the SCF settings of the two examples; grids is the number of points of the
# real-space grid along one axis of the 10 Angstrom cell
CH4_SCF = dict(temperature=300.0, cutoff=120.0, init=0.30, cell=10.0, grids=64)
H2O_SCF = dict(temperature=100.0, cutoff=150.0, init=0.10, cell=10.0, grids=75)
DIAMOND_SCF = dict(temperature=300.0, cutoff=150.0, init=0.10, cell=A_DIAMOND)

BASIS = {
    "ch4-min": {"H": "H5.0-s1", "C": "C5.0-s1p1"},
    "h2o-min": {"H": "H6.0-s1p1", "O": "O6.0-s1p1"},
    "dzp": {"H": "H6.0-s2p1", "C": "C6.0-s2p2d1", "O": "O6.0-s2p2d1"},
}
VPS = {"H": "H_PBE19", "C": "C_PBE19", "O": "O_PBE19"}


def make_input(name, atoms, scf, basis, xc="HSE06", spin="off", shift=(0.0, 0.0, 0.0), extra=(), cell=None,
               criterion="1.0e-10", mixing="rmm-diis", pulay=5, steps=100):
    species = sorted({a[0] for a in atoms})
    cell = scf["cell"] if cell is None else cell
    lines = [
        "System.CurrrentDirectory         ./",
        f"System.Name                     {name}",
        "level.of.stdout                   2",
        "level.of.fileout                  1",
        "DATA.PATH @DATA_PATH@",
        f"Species.Number       {len(species)}",
        "<Definition.of.Atomic.Species",
    ]
    lines += [f" {s}   {BASIS[basis][s]}   {VPS[s]}" for s in species]
    lines += ["Definition.of.Atomic.Species>",
              f"Atoms.Number         {len(atoms)}",
              "Atoms.SpeciesAndCoordinates.Unit   Ang",
              "<Atoms.SpeciesAndCoordinates"]
    for i, (s, x, y, z, up, dn) in enumerate(atoms, 1):
        lines.append(f" {i}  {s}  {x + shift[0]:.10f}  {y + shift[1]:.10f}  {z + shift[2]:.10f}  {up}  {dn}")
    lines += ["Atoms.SpeciesAndCoordinates>",
              "Atoms.UnitVectors.Unit             Ang",
              "<Atoms.UnitVectors",
              f"  {cell}   0.0   0.0", f"   0.0  {cell}   0.0", f"   0.0   0.0  {cell}",
              "Atoms.UnitVectors>",
              f"scf.XcType                 {xc}",
              f"scf.SpinPolarization        {spin}",
              f"scf.ElectronicTemperature  {scf['temperature']}",
              f"scf.energycutoff           {scf['cutoff']}",
              f"scf.maxIter                {steps}",
              "scf.EigenvalueSolver        cluster",
              "scf.Kgrid                  1 1 1",
              f"scf.Mixing.Type           {mixing}",
              f"scf.Init.Mixing.Weight     {scf['init']}",
              "scf.Min.Mixing.Weight      0.001",
              "scf.Max.Mixing.Weight      0.400",
              "scf.Mixing.History          7",
              f"scf.Mixing.StartPulay       {pulay}",
              f"scf.criterion             {criterion}",
              "MD.Type                     nomd",
              "MD.maxIter                    1"]
    lines += list(extra)
    return "\n".join(lines) + "\n"


RUNS = {}

# The PBE0 runs stop at tighter thresholds than those of HSE06 (scf.criterion
# 1e-10 and the default scf.exx.outer.criterion 1e-6).  With those, the PBE0
# runs of methane ended their last inner SCF at energies 6e-10 Hartree apart,
# by the number of steps alone: 1 and 2 processes differed by 1.2e-9, beyond
# the limit 1e-9 of the table, and agreed within 1e-12 with scf.criterion
# 1e-13 and scf.exx.outer.criterion 1e-10.  With the thresholds below, every
# bounded row of the PBE0 table stays within 2e-11 Hartree (M8).
PBE0_CRITERION = "1.0e-12"
PBE0_OUTER = "scf.exx.outer.criterion 1.0e-9"


def pbe0_input(tag, atoms, scf, basis, extra=(), **keywords):
    return make_input(tag, atoms, scf, basis, xc="PBE0", criterion=PBE0_CRITERION, extra=[PBE0_OUTER, *extra],
                      **keywords)


def add(name, text, ranks=1, threads=2):
    RUNS[name] = dict(text=text, ranks=ranks, threads=threads)


def molecule(tag, atoms, scf, basis):
    step = scf["cell"] / scf["grids"]
    add(f"{tag}-pbe", make_input(tag, atoms, scf, basis, xc="GGA-PBE"))
    add(f"{tag}-hse", make_input(tag, atoms, scf, basis))
    add(f"{tag}-hse-alpha0", make_input(tag, atoms, scf, basis, extra=["scf.exx.alpha 0.0"]))
    add(f"{tag}-hse-shift", make_input(tag, atoms, scf, basis, shift=(3 * step, 5 * step, -7 * step)))
    add(f"{tag}-hse-spin", make_input(tag, atoms, scf, basis, spin="on"))
    add(f"{tag}-hse-np2", make_input(tag, atoms, scf, basis), ranks=2, threads=1)
    add(f"{tag}-hse-np4", make_input(tag, atoms, scf, basis), ranks=4, threads=1)
    add(f"{tag}-hse-spin-np2", make_input(tag, atoms, scf, basis, spin="on"), ranks=2, threads=1)
    add(f"{tag}-hse-spin-np4", make_input(tag, atoms, scf, basis, spin="on"), ranks=4, threads=1)
    for pca in ("1.0e-3", "1.0e-5", "1.0e-6"):
        add(f"{tag}-hse-pca{pca[-1]}", make_input(tag, atoms, scf, basis, extra=[f"scf.exx.abf.pca {pca}"]))
    for cut in (6, 12, 16):
        add(f"{tag}-hse-v{cut}", make_input(tag, atoms, scf, basis, extra=[f"scf.exx.v.cutoff {cut}.0"]))
    add(f"{tag}-pbe-c6", make_input(tag, atoms, scf, basis, xc="GGA-PBE", criterion="1.0e-6"))
    add(f"{tag}-hse-c6", make_input(tag, atoms, scf, basis, criterion="1.0e-6"))
    for cut in (6, 8, 12, 16):
        add(f"{tag}-big-hse-v{cut}", make_input(tag, atoms, scf, basis, cell=20.0,
                                                extra=[f"scf.exx.v.cutoff {cut}.0"]))
    pbe0(tag, atoms, scf, basis)
    add(f"{tag}-pbe0-shift", pbe0_input(tag, atoms, scf, basis, shift=(3 * step, 5 * step, -7 * step)))
    add(f"{tag}-pbe0-spin", pbe0_input(tag, atoms, scf, basis, spin="on"))
    add(f"{tag}-pbe0-np2", pbe0_input(tag, atoms, scf, basis), ranks=2, threads=1)
    add(f"{tag}-pbe0-spin-np2", pbe0_input(tag, atoms, scf, basis, spin="on"), ranks=2, threads=1)
    add(f"{tag}-pbe0-spin-np4", pbe0_input(tag, atoms, scf, basis, spin="on"), ranks=4, threads=1)
    for cut in (8, 16):
        add(f"{tag}-pbe0-rc{cut}", pbe0_input(tag, atoms, scf, basis,
                                              extra=[f"scf.exx.coulomb.cutoff {cut}.0"]))
    add(f"{tag}-big-pbe0", pbe0_input(tag, atoms, scf, basis, cell=20.0))


def pbe0(tag, atoms, scf, basis):
    """The PBE0 runs that the molecules and the cell of diamond share."""
    add(f"{tag}-pbe-c12", make_input(tag, atoms, scf, basis, xc="GGA-PBE", criterion=PBE0_CRITERION))
    add(f"{tag}-pbe0", pbe0_input(tag, atoms, scf, basis))
    add(f"{tag}-pbe0-alpha0", pbe0_input(tag, atoms, scf, basis, extra=["scf.exx.alpha 0.0"]))
    add(f"{tag}-pbe0-np4", pbe0_input(tag, atoms, scf, basis), ranks=4, threads=1)
    for pca in ("1.0e-3", "1.0e-5", "1.0e-6"):
        add(f"{tag}-pbe0-pca{pca[-1]}", pbe0_input(tag, atoms, scf, basis,
                                                  extra=[f"scf.exx.abf.pca {pca}"]))
    for cut in (8, 12, 24):
        add(f"{tag}-pbe0-v{cut}", pbe0_input(tag, atoms, scf, basis,
                                             extra=[f"scf.exx.v.cutoff {cut}.0"]))


def triplet(tag, basis):
    add(f"{tag}-pbe", make_input(tag, O2_ATOMS, H2O_SCF, basis, xc="GGA-PBE", spin="on"))
    add(f"{tag}-hse", make_input(tag, O2_ATOMS, H2O_SCF, basis, spin="on"))
    add(f"{tag}-hse-np2", make_input(tag, O2_ATOMS, H2O_SCF, basis, spin="on"), ranks=2, threads=1)
    add(f"{tag}-hse-np4", make_input(tag, O2_ATOMS, H2O_SCF, basis, spin="on"), ranks=4, threads=1)
    add(f"{tag}-pbe0", pbe0_input(tag, O2_ATOMS, H2O_SCF, basis, spin="on"))
    add(f"{tag}-pbe0-np2", pbe0_input(tag, O2_ATOMS, H2O_SCF, basis, spin="on"), ranks=2, threads=1)
    add(f"{tag}-pbe0-np4", pbe0_input(tag, O2_ATOMS, H2O_SCF, basis, spin="on"), ranks=4, threads=1)


def diamond(tag, basis):
    add(f"{tag}-pbe", make_input(tag, DIAMOND_ATOMS, DIAMOND_SCF, basis, xc="GGA-PBE"))
    add(f"{tag}-hse", make_input(tag, DIAMOND_ATOMS, DIAMOND_SCF, basis))
    add(f"{tag}-hse-np4", make_input(tag, DIAMOND_ATOMS, DIAMOND_SCF, basis), ranks=4, threads=1)
    add(f"{tag}-hse-alpha0", make_input(tag, DIAMOND_ATOMS, DIAMOND_SCF, basis, extra=["scf.exx.alpha 0.0"]))
    for pca in ("1.0e-3", "1.0e-5", "1.0e-6"):
        add(f"{tag}-hse-pca{pca[-1]}", make_input(tag, DIAMOND_ATOMS, DIAMOND_SCF, basis,
                                                 extra=[f"scf.exx.abf.pca {pca}"]))
    for cut in (6, 12, 16):
        add(f"{tag}-hse-v{cut}", make_input(tag, DIAMOND_ATOMS, DIAMOND_SCF, basis,
                                            extra=[f"scf.exx.v.cutoff {cut}.0"]))
    pbe0(tag, DIAMOND_ATOMS, DIAMOND_SCF, basis)


# The mixers besides RMM-DIIS. Each of them starts again after a build in its
# own way (Exact_Exchange_Mixing_Offset), and scf.Mixing.StartPulay 5 and 10
# take the two branches of RMM-DIISK and RMM-DIISV.
MIXERS = ("gr-pulay", "rmm-diisk", "rmm-diish", "rmm-diisv")
# Simple and Kerker mixing keep no residuals and do not start again. Their
# weight falls to scf.Min.Mixing.Weight once the residual norm is small, so
# they are run with the default scf.criterion only, next to GGA-PBE.
PLAIN_MIXERS = ("simple", "kerker")


def mixing(tag, atoms, scf, basis):
    for m in MIXERS:
        add(f"{tag}-hse-mix-{m}", make_input(tag, atoms, scf, basis, mixing=m, steps=300))
    for m in ("rmm-diis",) + MIXERS:
        add(f"{tag}-hse-mix-{m}-p10", make_input(tag, atoms, scf, basis, mixing=m, pulay=10, steps=300))
    add(f"{tag}-hse-mix-rmm-diisk-c6", make_input(tag, atoms, scf, basis, mixing="rmm-diisk", steps=300,
                                                    criterion="1.0e-6"))
    for m in PLAIN_MIXERS:
        add(f"{tag}-pbe-mix-{m}-c6", make_input(tag, atoms, scf, basis, xc="GGA-PBE", mixing=m, steps=300,
                                                criterion="1.0e-6"))
        add(f"{tag}-hse-mix-{m}-c6", make_input(tag, atoms, scf, basis, mixing=m, steps=300, criterion="1.0e-6"))


molecule("ch4", CH4_ATOMS, CH4_SCF, "ch4-min")
molecule("h2o", H2O_ATOMS, H2O_SCF, "h2o-min")
molecule("ch4dzp", CH4_ATOMS, CH4_SCF, "dzp")
molecule("h2odzp", H2O_ATOMS, H2O_SCF, "dzp")
triplet("o2", "h2o-min")
triplet("o2dzp", "dzp")
diamond("dia8", "ch4-min")
diamond("dia8dzp", "dzp")
mixing("ch4", CH4_ATOMS, CH4_SCF, "ch4-min")
mixing("h2o", H2O_ATOMS, H2O_SCF, "h2o-min")
mixing("dia8", DIAMOND_ATOMS, DIAMOND_SCF, "ch4-min")

# ---- running and parsing --------------------------------------------------


def number(pattern, text, cast=float):
    found = re.findall(pattern, text, flags=re.M)
    return cast(found[-1]) if found else None


def parse(run_dir, system):
    result = {}
    std = (run_dir / "std.txt").read_text(errors="replace")
    out_path = run_dir / f"{system}.out"
    out = out_path.read_text(errors="replace") if out_path.exists() else ""
    for key in ("Utot", "Uexx", "Uele"):
        result[key] = number(rf"^\s+{key}\.\s+(-?\d+\.\d+)", out)
    result["scf_steps"] = number(r"SCF=\s*(\d+) \*+", std, int)
    result["achieved"] = "The SCF was achieved" in std
    result["outer_converged"] = "<EXX>  The outer iteration converged" in std
    result["outer"] = [(int(a), int(b), float(c), float(d)) for a, b, c, d in re.findall(
        r"<EXX>  outer iteration\s+(\d+) at SCF=\s*(\d+)\s+Uexx=\s*(\S+)\s+delta=\s*(\S+)", std)]
    result["builds"] = len(result["outer"]) or None
    deviations = [float(v) for v in re.findall(r"largest deviation from the solver (\S+)", std)]
    result["dm_deviation"] = max(deviations) if deviations else None
    sizes = re.search(r"<EXX>  (\d+) atoms, (\d+) orbitals, (\d+) ABFs, (\d+) C sites, (\d+) V sites", std)
    if sizes:
        result["sizes"] = [int(v) for v in sizes.groups()]
    truncated = re.search(r"<EXX>  truncated Coulomb interaction: R_c (\S+) bohr; V sites up to the larger ABF "
                          r"radius plus (\S+) bohr(?:; (\d+) reciprocal intervals)?", std)
    if truncated:
        result["cutoff"], result["v_reach"] = float(truncated.group(1)), float(truncated.group(2))
        if truncated.group(3):
            result["reciprocal_intervals"] = int(truncated.group(3))
    times = re.search(r"<EXX>  time \(s\): preparation (\S+), density matrices (\S+), exchange matrix (\S+), "
                      r"addition (\S+)", std)
    if times:
        result["exx_times"] = [float(v.rstrip(",")) for v in times.groups()]
    result["dft_time"] = number(r"DFT in total\s+=\s+(\S+)", std)
    block = re.search(r"HOMO =\s*(\d+)\s*\n\s*Eigenvalues\s*\n\s*Up-spin\s+Down-spin\s*\n((?:\s+\d+\s+\S+\s+\S+\s*\n)+)", out)
    if block:
        rows = [line.split() for line in block.group(2).strip().splitlines()]
        result["homo_index"] = int(block.group(1))
        result["eigen_up"] = [float(r[1]) for r in rows]
        result["eigen_down"] = [float(r[2]) for r in rows]
    return result


def execute(args, name):
    spec = RUNS[name]
    run_dir = args.out / name
    run_dir.mkdir(parents=True, exist_ok=True)
    system = re.search(r"System.Name\s+(\S+)", spec["text"]).group(1)
    (run_dir / "input.dat").write_text(spec["text"].replace("@DATA_PATH@", str(args.data_path)))
    env = dict(os.environ, OMP_NUM_THREADS=str(spec["threads"]), OPENBLAS_NUM_THREADS=str(spec["threads"]))
    start = time.time()
    with open(run_dir / "std.txt", "w") as handle:
        # one OpenMP thread, said in so many words: the program reads -nt and not the environment
        status = subprocess.run([*shlex.split(args.mpirun), "-np", str(spec["ranks"]), str(args.openmx.resolve()),
                                 "input.dat", "-nt", "1"], cwd=run_dir, stdout=handle, stderr=subprocess.STDOUT,
                                env=env).returncode
    result = parse(run_dir, system)
    result.update(status=status, wall=round(time.time() - start, 2), ranks=spec["ranks"], threads=spec["threads"])
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
        store.write_text(json.dumps(results, indent=1))
        print(f"{name:22s} exit {r['status']} {r['wall']:7.1f}s Utot {r['Utot']} Uexx {r['Uexx']} "
              f"builds {r['builds']} steps {r['scf_steps']} converged {r['achieved']}", flush=True)
    return 0

# ---- table ----------------------------------------------------------------


def table(args):
    results = json.loads((args.out / "results.json").read_text())
    failures = []

    def diff(a, b, key="Utot"):
        x, y = results.get(a, {}).get(key), results.get(b, {}).get(key)
        return None if x is None or y is None else x - y

    def show(label, value, limit=None):
        if value is None:
            return
        if limit is None:
            print(f"  {label:46s} {value:+.3e}")
            return
        ok = abs(value) <= limit
        print(f"  {label:46s} {value:+.3e}  limit {limit:.0e}  {'ok' if ok else 'FAIL'}")
        if not ok:
            failures.append(label)

    def converged(name):
        r = results.get(name)
        hybrid = "-hse" in name or "-pbe0" in name
        return bool(r) and r["status"] == 0 and r["achieved"] and (not hybrid or r["outer_converged"])

    def record(label, name, base, key="Utot"):
        """A row that the guide records without a limit: the difference, or what stopped the run."""
        r = results.get(name)
        if r is None:
            return
        if converged(name) and converged(base):
            show(label, diff(name, base, key))
        else:
            print(f"  {label:46s} not converged ({r['builds']} builds, {r['scf_steps']} steps)")

    def gap(name):
        r = results.get(name)
        if not r or "eigen_up" not in r or r["eigen_up"] != r["eigen_down"]:
            return None
        homo = r["homo_index"]
        return r["eigen_up"][homo - 1], r["eigen_up"][homo]

    def size(name, index):
        return results.get(name, {}).get("sizes", [None] * 5)[index]

    def pbe0_rows(tag):
        """The table of M4 for PBE0 (M8)."""
        r = results.get(f"{tag}-pbe0")
        if r is None:
            return
        print(f"== {tag}, PBE0")
        print(f"  atoms, orbitals, ABFs, C sites, V sites: {r.get('sizes')}  R_c {r.get('cutoff')} bohr")
        print(f"  PBE0 Utot {r['Utot']}  Uexx {r['Uexx']}  builds {r['builds']}  SCF steps {r['scf_steps']}")
        print("  outer iteration: " + ", ".join(f"{n} at SCF {s} delta {d:.2e}" for n, s, e, d in r["outer"]))
        g0, g1 = gap(f"{tag}-pbe-c12"), gap(f"{tag}-pbe0")
        if g0 and g1:
            print(f"  HOMO, LUMO (Hartree): PBE {g0[0]:.6f} {g0[1]:.6f}  PBE0 {g1[0]:.6f} {g1[1]:.6f}  "
                  f"gap {HARTREE_EV * (g0[1] - g0[0]):.3f} -> {HARTREE_EV * (g1[1] - g1[0]):.3f} eV")
        show("PBE0: alpha=0 minus PBE", diff(f"{tag}-pbe0-alpha0", f"{tag}-pbe-c12"), 1e-9)
        show("PBE0: shift: Utot", diff(f"{tag}-pbe0-shift", f"{tag}-pbe0"), 1e-8)
        show("PBE0: shift: Uexx", diff(f"{tag}-pbe0-shift", f"{tag}-pbe0", "Uexx"), 1e-8)
        show("PBE0: spin on minus off", diff(f"{tag}-pbe0-spin", f"{tag}-pbe0"), 1e-8)
        for n in ("np2", "np4"):
            show(f"PBE0: {n} minus np1", diff(f"{tag}-pbe0-{n}", f"{tag}-pbe0"), 1e-9)
            show(f"PBE0: spin {n} minus spin np1", diff(f"{tag}-pbe0-spin-{n}", f"{tag}-pbe0-spin"), 1e-9)
        show("PBE0: v24 (beyond the reach) minus automatic", diff(f"{tag}-pbe0-v24", f"{tag}-pbe0"), 1e-9)
        for p in ("pca3", "pca5", "pca6"):
            record(f"PBE0: {p} minus pca4 (ABFs {size(f'{tag}-pbe0-{p}', 2)})", f"{tag}-pbe0-{p}", f"{tag}-pbe0")
        for v in ("v8", "v12"):
            record(f"PBE0: {v} minus automatic (V sites {size(f'{tag}-pbe0-{v}', 4)})", f"{tag}-pbe0-{v}",
                   f"{tag}-pbe0")
        for c in ("rc8", "rc16"):
            record(f"PBE0: {c} minus automatic (V sites {size(f'{tag}-pbe0-{c}', 4)})", f"{tag}-pbe0-{c}",
                   f"{tag}-pbe0")
            record(f"PBE0: {c} minus automatic, Uexx", f"{tag}-pbe0-{c}", f"{tag}-pbe0", "Uexx")
        big = results.get(f"{tag}-big-pbe0")
        if big:
            print(f"  20 A cell: R_c {big.get('cutoff')} bohr, V sites {size(f'{tag}-big-pbe0', 4)}")
            record("PBE0: 20 A cell minus 10 A cell, Uexx", f"{tag}-big-pbe0", f"{tag}-pbe0", "Uexx")
        family = [x for n, x in results.items() if n.split("-")[0] == tag and "-pbe0" in n]
        deviations = [x["dm_deviation"] for x in family if x.get("dm_deviation") is not None]
        if deviations:
            show(f"PBE0: density matrices against the solver ({len(deviations)} runs)", max(deviations), 1e-10)
        if r.get("exx_times"):
            t = r["exx_times"]
            print(f"  time (s): preparation {t[0]}, density matrices {t[1]}, exchange matrix {t[2]}, addition {t[3]}; "
                  f"DFT in total {r.get('dft_time')}")

    for tag in sorted({n.split("-")[0] for n in results}):
        pbe0_rows(tag)
        r = results.get(f"{tag}-hse")
        if r is None:
            continue
        print(f"== {tag}")
        print(f"  atoms, orbitals, ABFs, C sites, V sites: {r.get('sizes')}")
        print(f"  HSE06 Utot {r['Utot']}  Uexx {r['Uexx']}  builds {r['builds']}  SCF steps {r['scf_steps']}")
        print("  outer iteration: " + ", ".join(f"{n} at SCF {s} delta {d:.2e}" for n, s, e, d in r["outer"]))
        pbe = results.get(f"{tag}-pbe")
        if pbe:
            print(f"  PBE   Utot {pbe['Utot']}  SCF steps {pbe['scf_steps']}")
            g0, g1 = gap(f"{tag}-pbe"), gap(f"{tag}-hse")
            if g0 and g1:
                print(f"  HOMO, LUMO (Hartree): PBE {g0[0]:.6f} {g0[1]:.6f}  HSE06 {g1[0]:.6f} {g1[1]:.6f}  "
                      f"gap {HARTREE_EV * (g0[1] - g0[0]):.3f} -> {HARTREE_EV * (g1[1] - g1[0]):.3f} eV")
        show("alpha=0 minus PBE", diff(f"{tag}-hse-alpha0", f"{tag}-pbe"), 1e-9)
        show("shift: Utot", diff(f"{tag}-hse-shift", f"{tag}-hse"), 1e-8)
        show("shift: Uexx", diff(f"{tag}-hse-shift", f"{tag}-hse", "Uexx"), 1e-8)
        show("spin on minus off", diff(f"{tag}-hse-spin", f"{tag}-hse"), 1e-8)
        for n in ("np2", "np4"):
            show(f"{n} minus np1", diff(f"{tag}-hse-{n}", f"{tag}-hse"), 1e-9)
            show(f"spin {n} minus spin np1", diff(f"{tag}-hse-spin-{n}", f"{tag}-hse-spin"), 1e-9)
        for p in ("pca3", "pca5", "pca6"):
            record(f"{p} minus pca4 (ABFs {size(f'{tag}-hse-{p}', 2)})", f"{tag}-hse-{p}", f"{tag}-hse")
        for v in ("v6", "v12", "v16"):
            record(f"{v} minus v8 (V sites {size(f'{tag}-hse-{v}', 4)})", f"{tag}-hse-{v}", f"{tag}-hse")
        for v in ("v6", "v12", "v16"):
            record(f"20 A cell: {v} minus v8 (V sites {size(f'{tag}-big-hse-{v}', 4)})",
                   f"{tag}-big-hse-{v}", f"{tag}-big-hse-v8")
        for n in sorted(x for x in results if x.startswith(f"{tag}-hse-mix-")):
            m = results[n]
            label = n[len(tag) + 9:]
            if n.endswith("-c6"):
                record(f"mixing {label} minus rmm-diis ({m['builds']} builds, {m['scf_steps']} steps)", n, f"{tag}-hse")
            else:
                show(f"mixing {label} minus rmm-diis ({m['builds']} builds, {m['scf_steps']} steps)",
                     diff(n, f"{tag}-hse"), 1e-8)
        for n in sorted(x for x in results if x.startswith(f"{tag}-pbe-mix-")):
            record(f"GGA-PBE: mixing {n[len(tag) + 9:]} minus rmm-diis ({results[n]['scf_steps']} steps)", n, f"{tag}-pbe")
        c6 = results.get(f"{tag}-hse-c6")
        if c6:
            pbe_steps = results.get(f"{tag}-pbe-c6", {}).get("scf_steps")
            print(f"  scf.criterion 1e-6: builds {c6['builds']}  SCF steps {c6['scf_steps']} (PBE alone {pbe_steps})")
            show("scf.criterion 1e-6 minus 1e-10", diff(f"{tag}-hse-c6", f"{tag}-hse"))
        family = [x for n, x in results.items() if n.split("-")[0] == tag and "-pbe0" not in n]
        deviations = [x["dm_deviation"] for x in family if x.get("dm_deviation") is not None]
        if deviations:
            show(f"density matrices against the solver ({len(deviations)} runs)", max(deviations), 1e-10)
        if r.get("exx_times"):
            t = r["exx_times"]
            print(f"  time (s): preparation {t[0]}, density matrices {t[1]}, exchange matrix {t[2]}, addition {t[3]}; "
                  f"DFT in total {r.get('dft_time')}")
    # -pbe0-v24 has a limit, so it is not exempt.
    recorded = re.compile(r"-(hse-(pca\d|v\d+)|pbe0-(pca\d|v8|v12|rc\d+)|big-pbe0)$")
    bad = [n for n in results if not converged(n)]
    exempt = [n for n in bad if recorded.search(n) and results[n]["status"] == 0]
    bad = [n for n in bad if n not in exempt]
    print("runs that failed or did not converge:", bad if bad else "none")
    print("recorded rows (scf.exx.abf.pca, scf.exx.v.cutoff, scf.exx.coulomb.cutoff) that did not converge:",
          exempt if exempt else "none")
    print("bounded differences beyond their limit:", failures if failures else "none")
    return 1 if bad or failures else 0


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
    summary = commands.add_parser("table", help="print the acceptance table from results.json")
    summary.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "list":
        print("\n".join(RUNS))
        return 0
    return run(args) if args.command == "run" else table(args)


if __name__ == "__main__":
    raise SystemExit(main())

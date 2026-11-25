#!/usr/bin/env python3
"""
exx_kpoint_checks.py -- the runs of the acceptance table of milestone M5
(docs/EXX_IMPLEMENTATION_GUIDE.md, section 9): HSE06 with k points and the
band solver, and the same table for PBE0 (milestone M8).

A calculation with an n1 x n2 x n3 mesh of k points that contains the Gamma
point is the calculation of the n1 x n2 x n3 supercell at the Gamma point: the
two have the same orbitals, density matrix and energy per primitive cell.  The
script runs such pairs, the first member with the band solver and the blocks
K(R) of the site form, the second with the cluster solver and the matrix of
the Gamma point (milestone M4), and compares their energies.  Further pairs
compare the band solver with itself on two cells, also on meshes that do not
contain the Gamma point.

  python3 tools/exx_kpoint_checks.py list
  python3 tools/exx_kpoint_checks.py run --openmx build/exx-cpu/openmx --out build/exx-m5 [NAME ...]
  python3 tools/exx_kpoint_checks.py table --out build/exx-m5

"run" without a name executes every run that results.json does not hold yet.
As root in a container, pass --mpirun "mpirun --allow-run-as-root --oversubscribe".
"table" ends with status 1 when a bounded difference exceeds its limit or a
run of a bounded comparison did not converge; a run that is only recorded and
does not converge is listed and leaves the status at 0.  Only the standard
library is used; the parser of the output is the one of tools/exx_scf_checks.py.

Threads.  Every run uses one OpenMP thread: the program takes that number
from the option -nt, not from OMP_NUM_THREADS, and the runs pass -nt 1.
"threads" in results.json is the value of OPENBLAS_NUM_THREADS: 2 in the runs
with one MPI process and in diaz-p222-hse-np2, 1 in the others.

The two members of a pair take the same path through the SCF.  For that the
real-space grid is given (scf.Ngrid and scf.fixed.grid), so that its points
are the same in both cells, and the two stop criteria, which OpenMX applies to
energies of the whole cell, are multiplied by the number of primitive cells
(scf.criterion 1e-10 and scf.exx.outer.criterion 1e-9 per primitive cell).

PBE0.  The radius of the truncated Coulomb interaction is by default
R_c = (3 N_k Omega/(4 pi))^(1/3), with N_k the number of k points of the mesh
and Omega the volume of the cell, so the two members of a pair have the same
R_c; the table compares the two values.  The PBE0 runs stop at scf.criterion
1e-12 and scf.exx.outer.criterion 1e-9 per primitive cell (PBE0_CRITERION of
tools/exx_scf_checks.py).  Where R_c + 3 r_max + (the reach of V) exceeds
2 pi/(step of kappa), 94.2 bohr with the default 1DFFT settings, the program
takes more reciprocal intervals for PBE0 (section 13, item 19 of the guide):
silicon with the 9 x 9 x 9 mesh does.

The systems
  dia   diamond, the primitive cell with 2 atoms and C5.0-s1p1
  diaz  the same with C6.0-s2p2d1
  siz   silicon, the primitive cell with 2 atoms and Si7.0-s2p2d1
  sim   the same with Si7.0-s1p1
  hfm   a chain of hydrogen atoms, 2.5 Angstrom apart, H5.0-s2p1, with the
        spins aligned: a spin-polarized system whose two spin channels differ
  hmet  the chain with 1.5 Angstrom between the atoms and no spin polarization:
        a half-filled band, with fractional occupations at the Fermi level
The chains have five orbitals per atom.  With one orbital per atom the density
matrix would follow from the symmetry alone, and the exchange matrix could not
change it.
The names
  SYSTEM-pABC          the primitive cell with the mesh A x B x C
  SYSTEM-sABC          the supercell A x B x C at the Gamma point (cluster solver)
  SYSTEM-sABCkDEF      the supercell A x B x C with the mesh D x E x F
  -reg                 scf.Generation.Kpoint REGULAR instead of GCenter: the
                       mesh of an even division does not contain the Gamma point
  -pbe, -hse, -pbe0    GGA-PBE, HSE06, PBE0
  -pbe-c12             GGA-PBE with scf.criterion 1e-12 per primitive cell, the
                       partner of -pbe0-alpha0
  -hse-alpha0          scf.exx.alpha 0: has to give the energy of GGA-PBE
  -hse-spin            scf.SpinPolarization on for the closed shell
  -pbe0-alpha0, -pbe0-spin  the same for PBE0
  -band                scf.EigenvalueSolver band for a supercell at the Gamma point
  -npN                 N MPI processes
  -vN                  scf.exx.v.cutoff N (bohr) instead of the default: 8
                       for HSE06, R_c + r_max for PBE0
  -default             the default stop criteria (scf.criterion 1e-6,
                       scf.exx.outer.criterion 1e-6)
  reject-...           an input that the program has to refuse
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
from exx_scf_checks import parse  # noqa: E402  (the numbers of std.txt and of the .out file)

REPO = Path(__file__).resolve().parents[1]

# ---- systems --------------------------------------------------------------

A_DIAMOND = 3.567
SYSTEMS = {
    # vectors and atoms of the primitive cell (Angstrom), points of the real-space
    # grid along the three vectors, initial occupations of an atom
    "dia": dict(species="C", basis="C5.0-s1p1", vps="C_PBE19",
                vectors=[(0.0, A_DIAMOND / 2, A_DIAMOND / 2), (A_DIAMOND / 2, 0.0, A_DIAMOND / 2),
                         (A_DIAMOND / 2, A_DIAMOND / 2, 0.0)],
                atoms=[(0.0, 0.0, 0.0), (A_DIAMOND / 4, A_DIAMOND / 4, A_DIAMOND / 4)],
                grid=(20, 20, 20), start=(2.0, 2.0), spin="off"),
    "hfm": dict(species="H", basis="H5.0-s2p1", vps="H_PBE19",
                vectors=[(2.5, 0.0, 0.0), (0.0, 8.0, 0.0), (0.0, 0.0, 8.0)], atoms=[(0.0, 0.0, 0.0)],
                grid=(20, 64, 64), start=(1.0, 0.0), spin="on"),
    "hmet": dict(species="H", basis="H5.0-s2p1", vps="H_PBE19",
                 vectors=[(1.5, 0.0, 0.0), (0.0, 8.0, 0.0), (0.0, 0.0, 8.0)], atoms=[(0.0, 0.0, 0.0)],
                 grid=(12, 64, 64), start=(0.5, 0.5), spin="off"),
}
SYSTEMS["diaz"] = dict(SYSTEMS["dia"], basis="C6.0-s2p2d1")
A_SILICON = 5.43
SYSTEMS["siz"] = dict(species="Si", basis="Si7.0-s2p2d1", vps="Si_PBE19",
                      vectors=[(0.0, A_SILICON / 2, A_SILICON / 2), (A_SILICON / 2, 0.0, A_SILICON / 2),
                               (A_SILICON / 2, A_SILICON / 2, 0.0)],
                      atoms=[(0.0, 0.0, 0.0), (A_SILICON / 4, A_SILICON / 4, A_SILICON / 4)],
                      grid=(32, 32, 32), start=(2.0, 2.0), spin="off")
SYSTEMS["sim"] = dict(SYSTEMS["siz"], basis="Si7.0-s1p1")
# the primitive cells of these are face-centred cubic and have a gap: their
# band edges on the mesh are recorded
FCC_INSULATORS = ("dia", "diaz", "siz", "sim")
# scf.criterion per primitive cell of the PBE0 runs and of their GGA-PBE partner
PBE0_INNER = 1.0e-12
SUFFIX = {"GGA-PBE": "pbe", "HSE06": "hse", "PBE0": "pbe0"}


def make_input(name, system, cells, kgrid, xc, spin=None, generation=None, extra=(), criteria=True, solver=None,
               inner=1.0e-10):
    s = SYSTEMS[system]
    count = cells[0] * cells[1] * cells[2]
    atoms = [tuple(a[x] + i * s["vectors"][0][x] + j * s["vectors"][1][x] + k * s["vectors"][2][x] for x in range(3))
             for i in range(cells[0]) for j in range(cells[1]) for k in range(cells[2]) for a in s["atoms"]]
    gamma_only = tuple(kgrid) == (1, 1, 1)
    lines = [
        "System.CurrrentDirectory         ./",
        f"System.Name                     {name}",
        "level.of.stdout                   2",
        "level.of.fileout                  1",
        "DATA.PATH @DATA_PATH@",
        "Species.Number       1",
        "<Definition.of.Atomic.Species",
        f" {s['species']}   {s['basis']}   {s['vps']}",
        "Definition.of.Atomic.Species>",
        f"Atoms.Number         {len(atoms)}",
        "Atoms.SpeciesAndCoordinates.Unit   Ang",
        "<Atoms.SpeciesAndCoordinates"]
    lines += [f" {i}  {s['species']}  {x:.10f}  {y:.10f}  {z:.10f}  {s['start'][0]}  {s['start'][1]}"
              for i, (x, y, z) in enumerate(atoms, 1)]
    lines += ["Atoms.SpeciesAndCoordinates>",
              "Atoms.UnitVectors.Unit             Ang",
              "<Atoms.UnitVectors"]
    lines += [f"  {v[0] * m:.10f}  {v[1] * m:.10f}  {v[2] * m:.10f}" for v, m in zip(s["vectors"], cells)]
    lines += ["Atoms.UnitVectors>",
              f"scf.XcType                 {xc}",
              f"scf.SpinPolarization        {spin or s['spin']}",
              "scf.ElectronicTemperature  300.0",
              f"scf.Ngrid                  {s['grid'][0] * cells[0]} {s['grid'][1] * cells[1]} {s['grid'][2] * cells[2]}",
              "scf.fixed.grid             0.0 0.0 0.0",
              "scf.maxIter                200",
              f"scf.EigenvalueSolver        {solver or ('cluster' if gamma_only else 'band')}",
              f"scf.Kgrid                  {kgrid[0]} {kgrid[1]} {kgrid[2]}",
              "scf.Mixing.Type           rmm-diis",
              "scf.Init.Mixing.Weight     0.10",
              "scf.Min.Mixing.Weight      0.001",
              "scf.Max.Mixing.Weight      0.400",
              "scf.Mixing.History          7",
              "scf.Mixing.StartPulay       5",
              "MD.Type                     nomd",
              "MD.maxIter                    1"]
    if criteria:
        lines += [f"scf.criterion             {inner * count:.1e}",
                  f"scf.exx.outer.criterion   {1.0e-9 * count:.1e}"]
    if generation:
        lines.append(f"scf.Generation.Kpoint       {generation}")
    lines += list(extra)
    return "\n".join(lines) + "\n"


RUNS = {}


def add(name, system, cells, kgrid, xc="HSE06", ranks=1, threads=2, **options):
    if xc == "PBE0":
        options.setdefault("inner", PBE0_INNER)
    text = make_input(name.replace("-", "_"), system, cells, kgrid, xc, **options)
    RUNS[name] = dict(text=text, ranks=ranks, threads=threads, cells=cells[0] * cells[1] * cells[2],
                      atoms=len(SYSTEMS[system]["atoms"]) * cells[0] * cells[1] * cells[2],
                      edges=(system in FCC_INSULATORS and tuple(cells) == (1, 1, 1) and tuple(kgrid) != (1, 1, 1)
                             and (options.get("spin") or SYSTEMS[system]["spin"]) == "off"))


def digits(triple):
    return "".join(str(v) for v in triple)


# (label, run, reference): the two describe the same crystal
PAIRS = []


def pair(system, cells, kgrid, total, xc="HSE06", **options):
    """The cell `cells` with the mesh `kgrid` against the cell `total` with the mesh that samples the same points."""
    rest = tuple(c * k // t for c, k, t in zip(cells, kgrid, total))
    tag = "-reg" if options.get("generation") == "REGULAR" else ""
    suffix = "-" + SUFFIX[xc]

    def name(c, k):
        cell = "p" if c == (1, 1, 1) else "s" + digits(c)
        mesh = digits(k) if c == (1, 1, 1) else ("" if k == (1, 1, 1) else "k" + digits(k))
        return f"{system}-{cell}{mesh}{tag}{suffix}"

    first, second = name(cells, kgrid), name(total, rest)
    for n, c, k in ((first, cells, kgrid), (second, total, rest)):
        if n not in RUNS:
            add(n, system, c, k, xc, **options)
    PAIRS.append((f"{first} against {second}", first, second))
    return first, second


# the acceptance of the guide: a mesh that contains the Gamma point against the supercell at the Gamma point
pair("dia", (1, 1, 1), (2, 2, 2), (2, 2, 2), xc="GGA-PBE")
pair("dia", (1, 1, 1), (2, 2, 2), (2, 2, 2))
pair("dia", (1, 1, 1), (3, 3, 3), (3, 3, 3))          # 13 of the 14 k points stand for a pair k, -k
pair("dia", (1, 1, 1), (3, 2, 1), (3, 2, 1))          # the three directions differ
pair("hfm", (1, 1, 1), (4, 1, 1), (4, 1, 1))          # two spin channels that differ
pair("hmet", (1, 1, 1), (4, 1, 1), (4, 1, 1))         # fractional occupations
pair("hmet", (1, 1, 1), (6, 1, 1), (6, 1, 1))
# the band solver on two cells
pair("dia", (1, 1, 1), (4, 4, 4), (2, 2, 2))
pair("dia", (1, 1, 1), (4, 4, 4), (2, 2, 2), generation="REGULAR")   # no Gamma point
pair("dia", (1, 1, 1), (4, 1, 1), (2, 1, 1), generation="REGULAR")   # the supercell has one k point, 1/4, for the pair
pair("hfm", (1, 1, 1), (4, 1, 1), (2, 1, 1), generation="REGULAR")
# a basis set with polarization functions
pair("diaz", (1, 1, 1), (2, 2, 2), (2, 2, 2))
# PBE0: the same pairs; R_c is 4.8 to 10.5 bohr for diamond, 8.5 and 10.1 bohr for the chains
for args, options in ((("dia", (1, 1, 1), (2, 2, 2), (2, 2, 2)), {}),
                      (("dia", (1, 1, 1), (3, 3, 3), (3, 3, 3)), {}),
                      (("dia", (1, 1, 1), (3, 2, 1), (3, 2, 1)), {}),
                      (("hfm", (1, 1, 1), (4, 1, 1), (4, 1, 1)), {}),
                      (("hmet", (1, 1, 1), (4, 1, 1), (4, 1, 1)), {}),
                      (("dia", (1, 1, 1), (4, 4, 4), (2, 2, 2)), {}),
                      (("dia", (1, 1, 1), (4, 4, 4), (2, 2, 2)), {"generation": "REGULAR"}),
                      (("diaz", (1, 1, 1), (2, 2, 2), (2, 2, 2)), {})):
    pair(*args, xc="PBE0", **options)

# one run against another: (label, run, base, quantity, limit)
SINGLE = []


def variant(label, name, base, system, cells, kgrid, limit, key="Utot", **options):
    add(name, system, cells, kgrid, **options)
    SINGLE.append((label, name, base, key, limit))


variant("alpha=0 minus PBE", "dia-p222-hse-alpha0", "dia-p222-pbe", "dia", (1, 1, 1), (2, 2, 2), 1e-9,
        extra=["scf.exx.alpha 0.0"])
variant("spin on minus off", "dia-p222-hse-spin", "dia-p222-hse", "dia", (1, 1, 1), (2, 2, 2), 1e-8, spin="on")
# scf.EigenvalueSolver band with the Gamma point alone: the program takes the cluster solver
variant("band in the input, Gamma point only, minus cluster", "dia-s222-hse-band", "dia-s222-hse", "dia",
        (2, 2, 2), (1, 1, 1), 1e-9, solver="band")
# MPI.  With fewer processes than k points every process solves several k
# points alone; otherwise the processes share the matrix of one k point.
for ranks in (2, 4):
    variant(f"np{ranks} minus np1, 14 k points", f"dia-p333-hse-np{ranks}", "dia-p333-hse", "dia", (1, 1, 1), (3, 3, 3),
            1e-9, ranks=ranks, threads=1)
    variant(f"np{ranks} minus np1, 1 k point", f"dia-s211k211-reg-hse-np{ranks}", "dia-s211k211-reg-hse", "dia",
            (2, 1, 1), (2, 1, 1), 1e-9, ranks=ranks, threads=1, generation="REGULAR")
    variant(f"np{ranks} minus np1, spin, 8 k points", f"dia-p222-hse-spin-np{ranks}", "dia-p222-hse-spin", "dia",
            (1, 1, 1), (2, 2, 2), 1e-9, ranks=ranks, threads=1, spin="on")
    variant(f"np{ranks} minus np1, spin, 3 k points", f"hfm-p411-hse-np{ranks}", "hfm-p411-hse", "hfm",
            (1, 1, 1), (4, 1, 1), 1e-9, ranks=ranks, threads=1)
variant("np4 minus np1, spin, 1 k point", "hfm-s211k211-reg-hse-np4", "hfm-s211k211-reg-hse", "hfm",
        (2, 1, 1), (2, 1, 1), 1e-9, ranks=4, threads=1, generation="REGULAR")
variant("np4 minus np1, 36 k points", "dia-p444-hse-np4", "dia-p444-hse", "dia", (1, 1, 1), (4, 4, 4), 1e-9,
        ranks=4, threads=1)
variant("np2 minus np1, 8 k points, s2p2d1", "diaz-p222-hse-np2", "diaz-p222-hse", "diaz", (1, 1, 1), (2, 2, 2), 1e-9,
        ranks=2, threads=2)
# PBE0
add("dia-p222-pbe-c12", "dia", (1, 1, 1), (2, 2, 2), xc="GGA-PBE", inner=PBE0_INNER)
variant("alpha=0 minus PBE", "dia-p222-pbe0-alpha0", "dia-p222-pbe-c12", "dia", (1, 1, 1), (2, 2, 2), 1e-9,
        xc="PBE0", extra=["scf.exx.alpha 0.0"])
variant("spin on minus off", "dia-p222-pbe0-spin", "dia-p222-pbe0", "dia", (1, 1, 1), (2, 2, 2), 1e-8, xc="PBE0",
        spin="on")
for ranks in (2, 4):
    variant(f"np{ranks} minus np1, 14 k points", f"dia-p333-pbe0-np{ranks}", "dia-p333-pbe0", "dia", (1, 1, 1),
            (3, 3, 3), 1e-9, xc="PBE0", ranks=ranks, threads=1)
    variant(f"np{ranks} minus np1, spin, 3 k points", f"hfm-p411-pbe0-np{ranks}", "hfm-p411-pbe0", "hfm", (1, 1, 1),
            (4, 1, 1), 1e-9, xc="PBE0", ranks=ranks, threads=1)
variant("np4 minus np1, 36 k points", "dia-p444-pbe0-np4", "dia-p444-pbe0", "dia", (1, 1, 1), (4, 4, 4), 1e-9,
        xc="PBE0", ranks=4, threads=1)
# 48 orbitals at one k point: with 4 processes the matrix has 3 x 3 blocks of 16
# rows and columns, and two of the blocks in each direction are on one process
add("dia-s321k211-reg-hse", "dia", (3, 2, 1), (2, 1, 1), generation="REGULAR")
variant("np4 minus np1, 1 k point, 48 orbitals", "dia-s321k211-reg-hse-np4", "dia-s321k211-reg-hse", "dia",
        (3, 2, 1), (2, 1, 1), 1e-9, ranks=4, threads=1, generation="REGULAR")

# Records without a limit: (label, runs).  The table prints their energies.
RECORDS = []


def record(label, names, system, cells, kgrid, **options):
    for name, extra in names:
        if name not in RUNS:
            add(name, system, cells, kgrid, extra=list(extra), **options)
    RECORDS.append((label, [name for name, _ in names]))


# The density matrix of a mesh repeats with the period of the supercell that
# the mesh stands for.  Where the exchange reaches further than that period it
# meets the repetitions, and the energy depends on the cutoff of V.
for mesh in (4, 6, 8):
    record(f"scf.exx.v.cutoff 6, 8, 12 and 16 bohr, mesh {mesh} x {mesh} x {mesh}",
           [(f"dia-p{mesh}{mesh}{mesh}-hse-v6", ["scf.exx.v.cutoff 6.0"]), (f"dia-p{mesh}{mesh}{mesh}-hse", []),
            (f"dia-p{mesh}{mesh}{mesh}-hse-v12", ["scf.exx.v.cutoff 12.0"]),
            (f"dia-p{mesh}{mesh}{mesh}-hse-v16", ["scf.exx.v.cutoff 16.0"])],
           "dia", (1, 1, 1), (mesh, mesh, mesh))
RECORDS.append(("meshes 2 x 2 x 2, 3 x 3 x 3, 4 x 4 x 4, 6 x 6 x 6 and 8 x 8 x 8",
                ["dia-p222-hse", "dia-p333-hse", "dia-p444-hse", "dia-p666-hse", "dia-p888-hse"]))
add("dia-p444-hse-default", "dia", (1, 1, 1), (4, 4, 4), criteria=False)
RECORDS.append(("the default stop criteria against the tight ones, mesh 4 x 4 x 4",
                ["dia-p444-hse-default", "dia-p444-hse"]))

# Basis sets with polarization functions on the meshes 4 x 4 x 4 and 8 x 8 x 8,
# with GGA-PBE beside HSE06: the table prints the band edges on the mesh.
for system in ("diaz", "siz"):
    for mesh in (4, 8):
        add(f"{system}-p{mesh}{mesh}{mesh}-pbe", system, (1, 1, 1), (mesh, mesh, mesh), xc="GGA-PBE")
        add(f"{system}-p{mesh}{mesh}{mesh}-hse", system, (1, 1, 1), (mesh, mesh, mesh))
RECORDS.append(("meshes 2 x 2 x 2, 4 x 4 x 4 and 8 x 8 x 8, C6.0-s2p2d1",
                ["diaz-p222-hse", "diaz-p444-hse", "diaz-p888-hse"]))
RECORDS.append(("meshes 4 x 4 x 4 and 8 x 8 x 8, silicon", ["siz-p444-hse", "siz-p888-hse"]))
# The cutoff of V with a basis that has polarization functions: silicon, the
# crystal with the smaller gap of the two, on the mesh 8 x 8 x 8.  The period
# of that mesh (58 bohr) is longer than the reach of the exchange with every
# one of the four cutoffs (41 to 51 bohr).
record("scf.exx.v.cutoff 6, 8, 12 and 16 bohr, silicon, mesh 8 x 8 x 8",
       [("siz-p888-hse-v6", ["scf.exx.v.cutoff 6.0"]), ("siz-p888-hse", []),
        ("siz-p888-hse-v12", ["scf.exx.v.cutoff 12.0"]), ("siz-p888-hse-v16", ["scf.exx.v.cutoff 16.0"])],
       "siz", (1, 1, 1), (8, 8, 8))

# PBE0 on the meshes 2 x 2 x 2 to 8 x 8 x 8: R_c grows with the mesh, from 5.3
# to 21.1 bohr, and the V list with it.
for mesh in (6, 8):
    add(f"dia-p{mesh}{mesh}{mesh}-pbe0", "dia", (1, 1, 1), (mesh, mesh, mesh), xc="PBE0")
RECORDS.append(("PBE0, meshes 2 x 2 x 2, 3 x 3 x 3, 4 x 4 x 4, 6 x 6 x 6 and 8 x 8 x 8",
                ["dia-p222-pbe0", "dia-p333-pbe0", "dia-p444-pbe0", "dia-p666-pbe0", "dia-p888-pbe0"]))
record("PBE0, scf.exx.v.cutoff 8 and 12 bohr against the automatic list (15.5 bohr), mesh 4 x 4 x 4",
       [("dia-p444-pbe0", []), ("dia-p444-pbe0-v8", ["scf.exx.v.cutoff 8.0"]),
        ("dia-p444-pbe0-v12", ["scf.exx.v.cutoff 12.0"])], "dia", (1, 1, 1), (4, 4, 4), xc="PBE0")
# Silicon with the minimal basis: R_c is 32.1 bohr with the mesh 8 x 8 x 8 and
# 36.1 bohr with 9 x 9 x 9.  2 R_c + 4 r_max is 92.1 and 100.2 bohr, so the
# second run takes 957 reciprocal intervals instead of 900.
for mesh in (8, 9):
    add(f"sim-p{mesh}{mesh}{mesh}-pbe0", "sim", (1, 1, 1), (mesh, mesh, mesh), xc="PBE0")
RECORDS.append(("PBE0, silicon with Si7.0-s1p1, meshes 8 x 8 x 8 and 9 x 9 x 9",
                ["sim-p888-pbe0", "sim-p999-pbe0"]))

# Inputs that the program has to refuse with a hybrid functional: the keyword
# that its message has to name.
for name, system, kgrid, extra, keyword in (
        ("reject-dm-range", "dia", (2, 2, 2), ["DM.export on", "DM.specify.energy.range on"], "DM.specify.energy.range"),
        ("reject-hks", "hmet", (4, 1, 1), ["NEGF.Output_HKS on"], "NEGF.Output_HKS"),
        ("reject-xanes", "dia", (2, 2, 2), ["XANES.gs.fileout on"], "XANES.gs.fileout"),
        ("reject-dos", "dia", (2, 2, 2), ["Dos.fileout on"], "Dos.fileout"),
        # the cluster solver stops this keyword by an older check; with the band solver it reaches the list
        ("reject-dosgauss", "dia", (2, 2, 2), ["DosGauss.fileout on"], "DosGauss.fileout")):
    add(name, system, (1, 1, 1), kgrid, extra=extra)
    RUNS[name]["rejected"] = keyword


# ---- runs -----------------------------------------------------------------

# A child of its own runs the program, so that the peak memory is that of one run.
WRAPPER = ("import resource,subprocess,sys\n"
           "status=subprocess.run(sys.argv[2:],stdout=open(sys.argv[1],'w'),stderr=subprocess.STDOUT).returncode\n"
           "print(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)\n"
           "sys.exit(status)\n")


HARTREE_EV = 27.211386245988


def band_edges(run_dir, system):
    """The band edges over the k points of the mesh, from the eigenvalues of the .out file (eV).

    The file lists one k point of every pair k, -k, in units of the reciprocal
    lattice vectors.  Half of the "Number of States" (the number of electrons)
    are the occupied bands.  X and L are the points of the face-centred cubic
    lattice: two coordinates 1/2 and one 0, and one or three coordinates 1/2.
    """
    out = run_dir / f"{system}.out"
    if not out.exists():
        return None
    text = out.read_text(errors="replace")
    electrons = re.search(r"Number of States\s+=\s+(\S+)", text)
    points = re.findall(r"kloop=\d+\s*\n\s*k1=\s*(\S+)\s+k2=\s*(\S+)\s+k3=\s*(\S+)\s*\n\s*\n"
                        r"((?:\s*\d+\s+\S+\s+\S+[ \t]*\n)+)", text)
    if not electrons or not points:
        return None
    occupied = int(round(float(electrons.group(1)) / 2.0))
    top, bottom, special = None, None, {}
    for k1, k2, k3, lines in points:
        values = [float(line.split()[1]) for line in lines.strip().splitlines()]
        if len(values) <= occupied:
            return None
        k = tuple(float(v) for v in (k1, k2, k3))
        halves = sorted(round(abs(v - round(v)), 4) for v in k)
        kind = {(0.0, 0.0, 0.0): "gamma", (0.0, 0.5, 0.5): "x", (0.0, 0.0, 0.5): "l", (0.5, 0.5, 0.5): "l"}.get(tuple(halves))
        if top is None or top[0] < values[occupied - 1]:
            top = (values[occupied - 1], k)
        if bottom is None or values[occupied] < bottom[0]:
            bottom = (values[occupied], k)
        if kind:
            old = special.get(kind)
            special[kind] = (max(values[occupied - 1], old[0]) if old else values[occupied - 1],
                             min(values[occupied], old[1]) if old else values[occupied])
    edges = dict(points=len(points), occupied=occupied, gap=round((bottom[0] - top[0]) * HARTREE_EV, 4),
                 top_k=list(top[1]), bottom_k=list(bottom[1]))
    if "gamma" in special:
        valence = special["gamma"][0]
        edges["top_at_gamma"] = abs(top[0] - valence) < 1e-9
        for kind, (_, conduction) in special.items():
            edges[kind] = round((conduction - valence) * HARTREE_EV, 4)
    return edges


def execute(args, name):
    spec = RUNS[name]
    run_dir = args.out / name
    run_dir.mkdir(parents=True, exist_ok=True)
    system = re.search(r"System.Name\s+(\S+)", spec["text"]).group(1)
    (run_dir / "input.dat").write_text(spec["text"].replace("@DATA_PATH@", str(args.data_path)))
    env = dict(os.environ, OMP_NUM_THREADS=str(spec["threads"]), OPENBLAS_NUM_THREADS=str(spec["threads"]))
    start = time.time()
    # one OpenMP thread, said in so many words: the program reads -nt and not the environment
    done = subprocess.run([sys.executable, "-c", WRAPPER, "std.txt", *shlex.split(args.mpirun), "-np", str(spec["ranks"]),
                           str(args.openmx.resolve()), "input.dat", "-nt", "1"], cwd=run_dir, env=env, capture_output=True,
                          text=True)
    result = parse(run_dir, system)
    std = (run_dir / "std.txt").read_text(errors="replace")
    asymmetry = [float(v) for v in re.findall(r"exchange matrix from its transpose (\S+)", std)]
    result["asymmetry"] = max(asymmetry) if asymmetry else None
    refused = re.search(r"^scf\.XcType=\S+ does not support\n((?:  \S.*\n)+)", std, flags=re.M)
    result["refused"] = [line.strip() for line in refused.group(1).splitlines()] if refused else None
    blocks = re.search(r"site form: (\d+) blocks of the density and exchange matrices, (\S+) MB", std)
    if blocks:
        result["blocks"] = [int(blocks.group(1)), float(blocks.group(2))]
    if spec["edges"]:
        result["edges"] = band_edges(run_dir, system)
    peak = done.stdout.strip().splitlines()
    result.update(status=done.returncode, wall=round(time.time() - start, 2), ranks=spec["ranks"],
                  threads=spec["threads"], cells=spec["cells"], atoms=spec["atoms"],
                  peak_mb=round(int(peak[-1]) / 1024.0, 1) if peak and peak[-1].isdigit() else None)
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
        print(f"{name:28s} exit {r['status']} {r['wall']:7.1f}s Utot {r['Utot']} Uexx {r['Uexx']} "
              f"builds {r['builds']} steps {r['scf_steps']} converged {r['achieved']}", flush=True)
    return 0

# ---- table ----------------------------------------------------------------


def table(args):
    results = json.loads((args.out / "results.json").read_text())
    failures = []

    def hybrid(name):
        return "-hse" in name or "-pbe0" in name

    def converged(name):
        r = results.get(name)
        return bool(r) and r["status"] == 0 and r["achieved"] and (not hybrid(name) or r["outer_converged"])

    def is_refusal(name):
        return name in RUNS and "rejected" in RUNS[name]

    def show(label, value, limit):
        ok = abs(value) <= limit
        print(f"  {label:58s} {value:+.3e}  limit {limit:.0e}  {'ok' if ok else 'FAIL'}")
        if not ok:
            failures.append(label)

    print("== pairs: the same crystal in two cells (Hartree; the limits are those of the guide)")
    for label, first, second in PAIRS:
        a, b = results.get(first), results.get(second)
        if not a or not b or a.get("Utot") is None or b.get("Utot") is None:
            continue
        ratio = b["cells"] / a["cells"]
        print(f" {label}: {a['builds']} and {b['builds']} builds, {a['scf_steps']} and {b['scf_steps']} SCF steps")
        show(f"Utot per atom ({a['atoms']} and {b['atoms']} atoms)", (a["Utot"] - b["Utot"] / ratio) / a["atoms"], 1e-7)
        if a.get("Uexx") is not None and b.get("Uexx") is not None:
            show(f"Uexx per cell of the first ({a['Uexx']:.9f})", a["Uexx"] - b["Uexx"] / ratio, 1e-9)
        if a.get("cutoff") is not None or b.get("cutoff") is not None:
            # both print R_c with six decimals
            show(f"R_c of the two (bohr; {a.get('cutoff')} and {b.get('cutoff')})",
                 (a.get("cutoff") or 0.0) - (b.get("cutoff") or 0.0), 1e-6)

    print("== one run against another (Hartree)")
    for label, name, base, key, limit in SINGLE:
        a, b = results.get(name), results.get(base)
        if a and b and a.get(key) is not None and b.get(key) is not None:
            show(f"{name}: {label}", a[key] - b[key], limit)
            if key == "Utot" and a.get("Uexx") is not None and b.get("Uexx") is not None:
                show(f"{name}: the same for Uexx", a["Uexx"] - b["Uexx"], limit)

    for title, tag in (("HSE06", "-hse"), ("PBE0", "-pbe0")):
        family = {n: r for n, r in results.items() if tag in n}
        if not family:
            continue
        print(f"== all runs with {title}")
        values = [r["asymmetry"] for r in family.values() if r.get("asymmetry") is not None]
        if values:
            show(f"K_xy(R) minus the transpose of K_yx(-R) ({len(values)} runs)", max(values), 1e-12)
        values = [r["dm_deviation"] for r in family.values() if r.get("dm_deviation") is not None]
        if values:
            show(f"density matrices against the solver ({len(values)} runs)", max(values), 1e-10)

    print("== records (Hartree per cell; no limit)")
    for label, names in RECORDS:
        known = [n for n in names if results.get(n) and results[n].get("Utot") is not None]
        if not known:
            continue
        print(f" {label}")
        first = results[known[0]]
        for n in known:
            r = results[n]
            print(f"  {n:24s} Utot {r['Utot']:.9f} ({r['Utot'] - first['Utot']:+.2e})  Uexx {r['Uexx']:.9f} "
                  f"({r['Uexx'] - first['Uexx']:+.2e})  {r['builds']} builds, {r['scf_steps']} SCF steps")

    print("== band edges on the k points of the mesh (eV; no limit): the gap between the highest occupied and the")
    print("   lowest empty state of the mesh, and the lowest empty state at Gamma, X and L above the highest")
    print("   occupied state at Gamma")
    for name, r in results.items():
        e = r.get("edges")
        if not e:
            continue
        special = "  ".join(f"{label} {e[key]:.3f}" for key, label in (("gamma", "Gamma"), ("x", "X"), ("l", "L")) if key in e)
        print(f"  {name:24s} {e['points']:4d} k points  gap {e['gap']:.3f}  top at {tuple(e['top_k'])}  "
              f"bottom at {tuple(e['bottom_k'])}  {special}")

    print("== inputs that have to be refused")
    for name, r in results.items():
        if not is_refusal(name):
            continue
        keyword = RUNS[name]["rejected"]
        ok = r["status"] == 0 and r["scf_steps"] is None and keyword in (r.get("refused") or [])
        print(f"  {name:24s} {keyword:28s} message names {r.get('refused')}  {'ok' if ok else 'FAIL'}")
        if not ok:
            failures.append(name)

    print("== sizes and times")
    for name, r in results.items():
        if is_refusal(name):
            continue
        t = r.get("exx_times")
        sizes, blocks = r.get("sizes"), r.get("blocks")
        truncated = (f"  R_c {r['cutoff']} bohr, V reach {r.get('v_reach')} bohr, reciprocal intervals "
                     f"{r.get('reciprocal_intervals')}" if r.get("cutoff") is not None else "")
        print(f"  {name:28s} np{r['ranks']} {r['wall']:7.1f} s  peak {r.get('peak_mb')} MB  "
              f"atoms, orbitals, ABFs, C sites, V sites {sizes}  blocks, MB {blocks}  "
              f"EXX time (preparation, density, build, addition) {t}{truncated}")

    # a run that only a record names may fail to converge: it is listed and leaves the status
    bounded = {n for _, first, second in PAIRS for n in (first, second)} | {n for _, name, base, _, _ in SINGLE
                                                                           for n in (name, base)}
    unconverged = [n for n in results if not is_refusal(n) and not converged(n)]
    bad = [n for n in unconverged if n in bounded]
    recorded = [n for n in unconverged if n not in bounded]
    print("runs that failed or did not converge:", bad if bad else "none")
    print("recorded runs that failed or did not converge:", recorded if recorded else "none")
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

#!/usr/bin/env python3
"""Offline analysis of the P6 accuracy dumps (plan v2.6 sec. 13.2-C).

Reads <case>/<case>.accdump.scf<N>.spin<S>.{Htilde,Ytilde,ko}.bin written
by the Cluster_DFT_Col dense owner (orthonormal basis, column-major FP64)
and reports, for a pair of runs (e.g. GEMMul8 vs cuBLAS):

  eps_K  = ||K_A - K_B||_F / ||K_B||_F          (K = Y f Y^T, f = 2 step)
  max|dK|, max|dH~|, max|d eps_i|
  r_HK   = ||H~K~ - K~H~||_F / (||H~||_F ||K~||_F)   per run

Usage:  ./accdump_analysis.py <caseA> <caseB> --nocc 128 [--scf 3 --spin 0]
        (nocc = occupied states; Si 64-atom cluster: 64*4/2 = 128)
"""
import argparse, sys
from pathlib import Path
import numpy as np


def load(case, scf, spin):
    base = Path(case) / f"{case}.accdump.scf{scf}.spin{spin}"
    meta = {}
    for line in (Path(f"{base}.meta.txt")).read_text().splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].lstrip("-").isdigit():
            meta[parts[0]] = int(parts[1])
    n, maxn = meta["n"], meta["maxn"]
    H = np.fromfile(f"{base}.Htilde.bin", dtype=np.float64)
    Y = np.fromfile(f"{base}.Ytilde.bin", dtype=np.float64)
    ko = np.fromfile(f"{base}.ko.bin", dtype=np.float64)
    if H.size != n * n or Y.size != n * maxn or ko.size != maxn:
        sys.exit(f"{case}: unexpected dump sizes (n={n}, maxn={maxn}, "
                 f"H={H.size}, Y={Y.size}, ko={ko.size})")
    H = H.reshape((n, n), order="F")
    Y = Y.reshape((n, maxn), order="F")
    return H, Y, ko, n, maxn


def density_matrix(Y, nocc, f=2.0):
    Yo = Y[:, :nocc]
    return f * (Yo @ Yo.T)


def fro(x):
    return float(np.linalg.norm(x, "fro"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("caseA")
    ap.add_argument("caseB", help="reference (FP64 baseline)")
    ap.add_argument("--nocc", type=int, required=True)
    ap.add_argument("--scf", type=int, default=3)
    ap.add_argument("--spin", type=int, default=0)
    a = ap.parse_args()

    HA, YA, koA, n, maxn = load(a.caseA, a.scf, a.spin)
    HB, YB, koB, n2, maxn2 = load(a.caseB, a.scf, a.spin)
    if (n, maxn) != (n2, maxn2):
        sys.exit("dimension mismatch between the two runs")
    if not (0 < a.nocc <= maxn):
        sys.exit(f"nocc {a.nocc} outside 1..{maxn}")

    KA = density_matrix(YA, a.nocc)
    KB = density_matrix(YB, a.nocc)

    # gauge check: K is invariant under eigenvector sign/rotation only within
    # degenerate blocks; warn if the occupied edge is nearly degenerate.
    gapB = koB[a.nocc] - koB[a.nocc - 1] if a.nocc < maxn else float("inf")

    print(f"# accdump comparison  scf={a.scf} spin={a.spin} n={n} maxn={maxn} nocc={a.nocc}")
    print(f"# A = {a.caseA}   B(reference) = {a.caseB}")
    print(f"HOMO-LUMO gap (B): {gapB:.6e} Ha "
          f"{'(WARNING: near-degenerate edge, K comparison may be gauge-limited)' if gapB < 1e-6 else ''}")
    print(f"eps_K              = {fro(KA-KB)/fro(KB):.6e}")
    print(f"max|dK|            = {float(np.max(np.abs(KA-KB))):.6e}")
    print(f"max|dHtilde|       = {float(np.max(np.abs(HA-HB))):.6e}   "
          f"(rel {float(np.max(np.abs(HA-HB)))/float(np.max(np.abs(HB))):.3e})")
    print(f"max|d eigenvalue|  = {float(np.max(np.abs(koA-koB))):.6e} Ha")
    for tag, H, K in (("A", HA, KA), ("B", HB, KB)):
        r = fro(H @ K - K @ H) / (fro(H) * fro(K))
        print(f"r_HK({tag})            = {r:.6e}")
    # orthonormality of the dumped eigenvectors (basis sanity)
    for tag, Y in (("A", YA), ("B", YB)):
        dev = float(np.max(np.abs(Y.T @ Y - np.eye(maxn))))
        print(f"max|Y^T Y - I| ({tag})  = {dev:.6e}")


if __name__ == "__main__":
    main()

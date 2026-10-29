#!/usr/bin/env python3
"""The non-collinear 2n x 2n block-cyclic index map (source/openmx_nc_indices.h).

Compiles the shared helper next to the former per-file copy (the version of
Hamiltonian_Band_NC_Hs2.c in HEAD recounted the basis size over every atom for
every matrix element) with stubbed globals, and compares the twelve outputs of
both over several process grids, block sizes, atom lists and every local
element of every process.  The map is pure integer arithmetic, so the two
must agree exactly.
"""
from pathlib import Path
import os
import shlex
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "source" / "openmx_nc_indices.h"


def legacy_helper():
    """The former static helper, from the committed Hamiltonian_Band_NC_Hs2.c."""
    text = subprocess.run(["git", "-C", str(ROOT), "show", "HEAD:source/Hamiltonian_Band_NC_Hs2.c"],
                          capture_output=True, text=True, check=True).stdout
    start = text.index("\nvoid trans_indices_n_to_n2(") + 1
    brace = text.index("{", start)
    depth, at = 0, brace
    while True:
        if text[at] == "{":
            depth += 1
        elif text[at] == "}":
            depth -= 1
            if depth == 0:
                break
        at += 1
    return text[start:at + 1].replace("void trans_indices_n_to_n2(", "static void legacy_trans_indices(", 1)


def shared_helper():
    text = HEADER.read_text()
    start = text.index("static inline int openmx_nc_basis_size(void)")
    end = text.index("#endif", start)
    return text[start:end]


DRIVER = r"""
#include <stdio.h>
#include <stdlib.h>
/* the globals both helpers read */
int atomnum, *WhatSpecies, *Spe_Total_CNO;
int np_rows, np_cols, nblk, nblk2, np_rows2, np_cols2;
%(legacy)s
%(shared)s
static long checks;
int main(void){
  static const int grids[][2] = {{1,1},{1,2},{2,1},{2,3},{3,3},{4,5},{6,3}};
  static const int blocks[] = {1, 2, 4, 7, 16};
  static const int species[] = {13, 9, 26, 4, 1};
  for (int atoms = 1; atoms <= 9; atoms += 4) {
    WhatSpecies = calloc(atoms + 1, sizeof(int)); Spe_Total_CNO = calloc(5, sizeof(int));
    atomnum = atoms;
    for (int s = 0; s < 5; s++) Spe_Total_CNO[s] = species[s];
    for (int a = 1; a <= atoms; a++) WhatSpecies[a] = (a * 3) %% 5;
    for (size_t g = 0; g < sizeof grids / sizeof grids[0]; g++) {
      np_rows = grids[g][0]; np_cols = grids[g][1];
      for (size_t g2 = 0; g2 < sizeof grids / sizeof grids[0]; g2++) {
        np_rows2 = grids[g2][0]; np_cols2 = grids[g2][1];
        for (size_t b = 0; b < sizeof blocks / sizeof blocks[0]; b++) {
          nblk = blocks[b];
          for (size_t b2 = 0; b2 < sizeof blocks / sizeof blocks[0]; b2++) {
            nblk2 = blocks[b2];
            const int n = openmx_nc_basis_size();
            /* every process of the n x n grid, every local element up to the n-th global row/column */
            for (int id1 = 0; id1 < np_rows * np_cols; id1++) {
              for (int i1 = 0; i1 < n; i1++) for (int j1 = 0; j1 < n; j1++) {
                int a[12], c[12];
                legacy_trans_indices(id1, i1, j1, &a[0],&a[1],&a[2],&a[3],&a[4],&a[5],&a[6],&a[7],&a[8],&a[9],&a[10],&a[11]);
                openmx_nc_trans_indices_n_to_n2(n, id1, i1, j1, &c[0],&c[1],&c[2],&c[3],&c[4],&c[5],&c[6],&c[7],&c[8],&c[9],&c[10],&c[11]);
                for (int q = 0; q < 12; q++) if (a[q] != c[q]) {
                  fprintf(stderr, "FAIL atoms=%%d grid=%%dx%%d grid2=%%dx%%d nblk=%%d nblk2=%%d id1=%%d i1=%%d j1=%%d output %%d: %%d vs %%d\n",
                          atoms, np_rows, np_cols, np_rows2, np_cols2, nblk, nblk2, id1, i1, j1, q, a[q], c[q]);
                  return 1;
                }
                checks++;
              }
            }
          }
        }
      }
    }
    free(WhatSpecies); free(Spe_Total_CNO);
  }
  printf("PASS: %%ld element maps of the non-collinear 2n x 2n layout agree with the former per-file helper\n", checks);
  return 0;
}
"""


def main():
    with tempfile.TemporaryDirectory(prefix="openmx-nc-indices-") as folder:
        p = Path(folder)
        (p / "check.c").write_text(DRIVER % {"legacy": legacy_helper(), "shared": shared_helper()})
        cc = shlex.split(os.environ.get("CC", "cc"))
        subprocess.run([*cc, "-std=c11", "-O2", "-Wall", "-o", str(p / "check"), str(p / "check.c")], check=True)
        subprocess.run([str(p / "check")], check=True, timeout=600)


if __name__ == "__main__":
    main()

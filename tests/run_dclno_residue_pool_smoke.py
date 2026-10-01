#!/usr/bin/env python3
"""Check the production DC-LNO residue allocator with a portable CPU harness.

No OpenMX build, MPI runtime or GPU is needed. --sanitize enables ASan/UBSan.
The allocator and its size checks are extracted from the current source so
the test exercises the same pointer view used by the scientific calculation.
"""

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import tempfile


def function(source, name):
    start = source.index("static ", source.index(name) - 32)
    brace = source.index("{", source.index(name, start))
    depth, end = 1, brace + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


PREFIX = r"""
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <limits.h>
static int Spe_Total_CNO[32], WhatSpecies[32], FNAN[32], *natn[32];
static int allocation_count;
static void DCLNO_AbortWithMessage(const char *message) {
    fprintf(stderr, "%s\n", message);
    exit(77);
}
static void *smoke_malloc(size_t bytes) {
    allocation_count++;
    return malloc(bytes);
}
#define malloc smoke_malloc
"""

HARNESS = r"""
#undef malloc
static int neighbors[32];

static void layout(int no0, int pairs) {
    natn[1] = neighbors;
    FNAN[1] = pairs - 1;
    for (int h = 0; h < pairs; h++) {
        neighbors[h] = h + 1;
        WhatSpecies[h + 1] = h + 1;
        Spe_Total_CNO[h + 1] = h == 0 ? no0
            : 1 + (int)(((size_t)no0 + 3u*(size_t)h) % 23u);
    }
}

int main(int argc, char **argv) {
    if (argc > 1) {
        int pairs = strcmp(argv[1], "row-overflow") == 0 ? 8 : 1;
        int states = strcmp(argv[1], "value-overflow") == 0 ? 8 : 2;
        int invalid = strcmp(argv[1], "invalid") == 0 || strcmp(argv[1], "invalid-one") == 0;
        /* Keep each individual row product representable on 32/64-bit hosts.
           Invalid-state cases stay small so another overflow cannot mask a
           missing states>=2 check. */
        int no0 = invalid ? 2 : (SIZE_MAX <= UINT32_MAX ? 32768 : INT_MAX);
        layout(no0, pairs);
        for (int h = 0; h < pairs; h++) Spe_Total_CNO[h + 1] = no0;
        if (invalid) states = strcmp(argv[1], "invalid") == 0 ? 0 : 1;
        DCLNO_AllocateResidueValues(NULL, 1, states);
        return 2; /* Expected dimension rejection before touching rows. */
    }
    const int sizes[] = {1, 2, 4, 9, 17};
    const int pairs_set[] = {1, 3, 8};
    const int windows[] = {2, 3, 17, 127};
    int cases = 0;
    for (int a = 0; a < 5; a++) for (int b = 0; b < 3; b++) for (int c = 0; c < 4; c++) {
        const int no0 = sizes[a], pairs = pairs_set[b], states = windows[c];
        double ****pooled = calloc(pairs, sizeof(*pooled));
        double ****legacy = calloc(pairs, sizeof(*legacy));
        layout(no0, pairs);
        for (int h = 0; h < pairs; h++) {
            int no1 = Spe_Total_CNO[h + 1];
            pooled[h] = calloc(no0, sizeof(*pooled[h]));
            legacy[h] = calloc(no0, sizeof(*legacy[h]));
            for (int i = 0; i < no0; i++) {
                pooled[h][i] = calloc(no1, sizeof(*pooled[h][i]));
                legacy[h][i] = calloc(no1, sizeof(*legacy[h][i]));
                for (int j = 0; j < no1; j++) legacy[h][i][j] = calloc(states, sizeof(double));
            }
        }
        allocation_count = 0;
        DCLNO_AllocateResidueValues(pooled, 1, states);
        if (allocation_count != 1) return 3;
        double *base = pooled[0][0][0];
        size_t row = 0;
        for (int h = 0; h < pairs; h++) for (int i = 0; i < no0; i++) {
            for (int j = 0; j < Spe_Total_CNO[h + 1]; j++, row++) {
                if (pooled[h][i][j] != base + row * states) return 4;
                for (int k = 0; k < states; k++) {
                    double value = ((double)row + 3) * (k + 7) / 31.0;
                    pooled[h][i][j][k] = value;
                    legacy[h][i][j][k] = value;
                }
            }
        }
        /* Read all rows after every write to expose aliasing between them;
           use the residue contraction's original per-state sum order. */
        for (int h = 0; h < pairs; h++) for (int i = 0; i < no0; i++) {
            for (int j = 0; j < Spe_Total_CNO[h + 1]; j++) {
                double new_sum = pooled[h][i][j][0], old_sum = legacy[h][i][j][0];
                if (memcmp(pooled[h][i][j], legacy[h][i][j], states * sizeof(double))) return 5;
                for (int k = 2; k < states; k++) {
                    new_sum += pooled[h][i][j][k] / (k + 1.0);
                    old_sum += legacy[h][i][j][k] / (k + 1.0);
                }
                if (memcmp(&new_sum, &old_sum, sizeof(double))) return 6;
            }
        }
        free(base);
        for (int h = 0; h < pairs; h++) {
            for (int i = 0; i < no0; i++) {
                for (int j = 0; j < Spe_Total_CNO[h + 1]; j++) free(legacy[h][i][j]);
                free(legacy[h][i]); free(pooled[h][i]);
            }
            free(legacy[h]); free(pooled[h]);
        }
        free(legacy); free(pooled);
        cases++;
    }
    printf("PASS %d residue layouts: one allocation, distinct rows, exact values/contractions, cleanup\n", cases);
    return 0;
}
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cc", default=os.environ.get("CC", "cc"))
    parser.add_argument("--sanitize", action="store_true")
    args = parser.parse_args()
    source = (Path(__file__).resolve().parents[1] / "source/Divide_Conquer_LNO.c").read_text()
    names = ["DCLNO_CheckedArrayBytes", "DCLNO_CheckedMulCount", "DCLNO_MallocArray",
             "DCLNO_AllocateResidueValues"]
    extracted = "\n".join(function(source, name) for name in names)
    with tempfile.TemporaryDirectory(prefix="openmx-dclno-residues-") as directory:
        harness = Path(directory) / "smoke.c"
        binary = Path(directory) / "smoke"
        harness.write_text(PREFIX + extracted + HARNESS)
        flags = ["-std=c11", "-O2", "-Wall", "-Wextra"]
        if args.sanitize:
            flags += ["-fsanitize=address,undefined", "-fno-sanitize-recover=all", "-fno-omit-frame-pointer"]
        subprocess.run(shlex.split(args.cc) + flags + [str(harness), "-o", str(binary)], check=True)
        subprocess.run([str(binary)], check=True)
        rejection = {
            "invalid": "Invalid residue dimensions in DC-LNO.",
            "invalid-one": "Invalid residue dimensions in DC-LNO.",
            "row-overflow": "Residue row count overflow in DC-LNO.",
            "value-overflow": "Dimension overflow in Divide_Conquer_LNO.c: residue value block",
            "byte-overflow": "Allocation size overflow in Divide_Conquer_LNO.c: residue value block",
        }
        for mode, expected in rejection.items():
            result = subprocess.run([str(binary), mode], capture_output=True, text=True)
            if result.returncode != 77 or expected not in result.stderr:
                raise RuntimeError(f"{mode}: expected exit 77 and {expected!r}, "
                                   f"got {result.returncode}: {result.stderr}")
        print("PASS invalid dimensions and row/value/byte overflow rejection")


if __name__ == "__main__":
    main()

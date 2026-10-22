#!/usr/bin/env python3
"""Check noncollinear sparse-to-dense orientation without MPI or a GPU.

Both assembly helpers are extracted from source/Cluster_DFT_NonCol.c. Literal
oracles cover asymmetric, antisymmetric and symmetric matrices with reversed
gather order and unequal atom orbital counts. CC and CFLAGS select the compiler.
"""

import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def extract_helper(source: str, name: str) -> str:
    match = re.search(r"^static void " + name + r"\(", source, re.M)
    if match is None:
        raise ValueError(f"missing production helper: {name}")
    opening = source.index("{", match.end())
    if ";" in source[match.end():opening]:
        raise ValueError(f"expected a definition of {name}")
    tokens = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|/\*.*?\*/|//[^\n]*|[{}]', re.S)
    depth = 0
    for token in tokens.finditer(source, opening):
        if token.group() == "{":
            depth += 1
        elif token.group() == "}":
            depth -= 1
            if depth == 0:
                return source[match.start():token.end()]
    raise ValueError(f"unterminated production helper: {name}")


PREFIX = r'''
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #condition); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

/* Atom 1 has two orbitals, atom 2 one; self is the first neighbor. */
static int atomnum = 2;
static int WhatSpecies[] = {0, 0, 1};
static int Spe_Total_CNO[] = {2, 1};
static int FNAN[] = {0, 1, 1};
static int atom1_neighbors[] = {1, 2}, atom2_neighbors[] = {2, 1};
static int *natn[] = {NULL, atom1_neighbors, atom2_neighbors};

static void ClusterNonCol_AbortWithMessage(const char *message)
{
    fprintf(stderr, "%s\n", message);
    exit(EXIT_FAILURE);
}

static size_t ClusterNonCol_CheckedAddCount(size_t a, size_t b, const char *message)
{
    if (a > SIZE_MAX - b) ClusterNonCol_AbortWithMessage(message);
    return a + b;
}

static size_t ClusterNonCol_CheckedMulCount(size_t a, size_t b, const char *message)
{
    if (b && a > SIZE_MAX / b) ClusterNonCol_AbortWithMessage(message);
    return a * b;
}

typedef struct { void *resources; } ClusterNonColState;
static ClusterNonColState fixture_state = {NULL};
static void openmx_core_native_free(void *owner, void *pointer) { (void)owner; free(pointer); }
static void *ClusterNonCol_MallocArray(ClusterNonColState *state, size_t count, size_t size, const char *message)
{
    (void)state;
    void *p = malloc(ClusterNonCol_CheckedMulCount(count, size, message));
    if (p == NULL) ClusterNonCol_AbortWithMessage(message);
    return p;
}
'''


HARNESS = r'''
static void check_matrix(const char *case_name, const char *path,
                         const double *actual, const double *expected)
{
    for (int col = 0; col < 3; col++) {
        for (int row = 0; row < 3; row++) {
            int p = row + col * 3;
            if (actual[p] != expected[p]) {
                fprintf(stderr, "FAIL %s %s row=%d col=%d: expected %g, got %g\n",
                        case_name, path, row, col, expected[p], actual[p]);
                exit(EXIT_FAILURE);
            }
        }
    }
}

int main(void)
{
    /* Global rows/columns: [atom1-orb0, atom1-orb1, atom2-orb0].
       Gather records in this explicitly chosen (row atom/orb; col atom/orb) order:
       (2,0;2,0), (2,0;1,0), (2,0;1,1),
       (1,0;1,0), (1,0;1,1), (1,0;2,0),
       (1,1;1,0), (1,1;1,1), (1,1;2,0).
       Expected matrices are literal column-major arrays, independent of the
       production index formula. Symmetric input alone cannot detect transpose. */
    const struct {
        const char *name;
        double gathered[9], expected[9];
    } cases[] = {
        {"asymmetric", {33,31,32,11,12,13,21,22,23},
                       {11,21,31,12,22,32,13,23,33}},
        {"antisymmetric", {0,3,-5,0,2,-3,-2,0,5},
                          {0,-2,3,2,0,-5,-3,5,0}},
        {"symmetric", {33,13,23,11,12,13,12,22,23},
                      {11,12,13,12,22,23,13,23,33}}
    };
    const int order[] = {0, 2, 1};
    int mp[] = {0, 1, 3};

    for (size_t c = 0; c < sizeof(cases) / sizeof(cases[0]); c++) {
        double host[9], scattered[9] = {0};
        int index[9];
        /* Poison outputs to detect omissions instead of relying on stack data. */
        for (int k = 0; k < 9; k++) { host[k] = -999; index[k] = -1; }
        ClusterNonCol_BuildDenseFromGathered(&fixture_state, cases[c].gathered, order, mp, host, 3, 9);
        ClusterNonCol_BuildDenseIndexFromGathered(order, mp, 3, 9, index);
        for (int k = 0; k < 9; k++) {
            CHECK(0 <= index[k] && index[k] < 9);
            scattered[index[k]] += cases[c].gathered[k];
        }
        check_matrix(cases[c].name, "host assembly", host, cases[c].expected);
        check_matrix(cases[c].name, "GPU index scatter", scattered, cases[c].expected);
    }
    puts("PASS: 3 noncollinear dense-index orientation cases (host and GPU mapping)");
    return 0;
}
'''


def main() -> None:
    source = (ROOT / "source/Cluster_DFT_NonCol.c").read_text()
    helpers = "\n".join(extract_helper(source, name) for name in (
        "ClusterNonCol_BuildDenseFromGathered",
        "ClusterNonCol_BuildDenseIndexFromGathered",
    ))
    with tempfile.TemporaryDirectory(prefix="openmx-noncol-dense-index-") as tmp:
        cfile, binary = Path(tmp) / "check.c", Path(tmp) / "check"
        cfile.write_text(PREFIX + helpers + HARNESS)
        subprocess.run(
            shlex.split(os.environ.get("CC", "cc"))
            + ["-std=c11", "-O2", "-Wall", "-Wextra", "-Werror"]
            + shlex.split(os.environ.get("CFLAGS", ""))
            + [str(cfile), "-o", str(binary)], check=True,
        )
        subprocess.run([str(binary)], check=True)


if __name__ == "__main__":
    main()

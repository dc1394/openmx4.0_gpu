#!/usr/bin/env python3
"""Check the production threaded cluster DM fallback without HIP or MPI.

Compare against state-outer density construction for mixed species, duplicate
neighbor images, partial charges, multiple rank slices and OpenMP thread counts.
A sparse virtual mapping checks row offsets beyond INT_MAX without 16 GiB RAM.
"""

import os
import re
from pathlib import Path
import shlex
import subprocess
import tempfile

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


ROOT = Path(__file__).resolve().parents[1]

PREFIX = r'''
#define _GNU_SOURCE
#include <limits.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #c); exit(1); } } while (0)
#define ATOMS 26
#define STATES 17
#define CAPACITY 2048
static int atomnum = ATOMS;
static int WhatSpecies[ATOMS+1], Spe_Total_CNO[] = {1, 2, 3};
static int FNAN[ATOMS+1], neighbors[ATOMS+1][4], *natn[ATOMS+1];
static void ClusterCol_AbortWithMessage(const char *message)
{
    fprintf(stderr, "%s\n", message);
    exit(1);
}
static void *ClusterCol_MallocArray(size_t count, size_t bytes, const char *label)
{
    CHECK(bytes == 0 || count <= SIZE_MAX / bytes);
    void *p = malloc(count * bytes);
    if (!p) ClusterCol_AbortWithMessage(label);
    return p;
}
'''

HARNESS = r'''
static int fixture(int *mp, int *row0, int *row1)
{
    int n = 0, entries = 0;
    for (int a = 1; a <= atomnum; a++) {
        WhatSpecies[a] = a % 3;
        mp[a] = n + 1;
        n += Spe_Total_CNO[WhatSpecies[a]];
        FNAN[a] = 3;
        natn[a] = neighbors[a];
        neighbors[a][0] = a;
        neighbors[a][1] = a % atomnum + 1;
        neighbors[a][2] = (a + 7) % atomnum + 1;
        neighbors[a][3] = neighbors[a][1];  /* distinct periodic image */
    }
    for (int a = 1; a <= atomnum; a++)
        for (int nb = 0; nb <= FNAN[a]; nb++) {
            const int b = natn[a][nb];
            for (int i = 0; i < Spe_Total_CNO[WhatSpecies[a]]; i++)
                for (int j = 0; j < Spe_Total_CNO[WhatSpecies[b]]; j++) {
                    CHECK(entries < CAPACITY);
                    row0[entries] = mp[a] - 1 + i;
                    row1[entries++] = mp[b] - 1 + j;
                }
        }
    return entries;
}

static void check_slices(void)
{
    int mp[ATOMS+1], row0[CAPACITY], row1[CAPACITY];
    const int entries = fixture(mp, row0, row1);
    const int n = mp[atomnum] - 1 + Spe_Total_CNO[WhatSpecies[atomnum]];
    double evec[3*ATOMS*STATES], packed[3*ATOMS*STATES];
    double occ[STATES], eig[STATES], docc[STATES];
    double reference[3][CAPACITY] = {{0}}, thread_reference[3][CAPACITY] = {{0}};
    const int teams[] = {1, 2, 4}, parts[] = {1, 3, 19};

    for (int b = 0; b < n; b++)
        for (int k = 0; k < STATES; k++)
            evec[b*STATES+k] = sin(0.19*(b+1)*(k+2));
    for (int k = 0; k < STATES; k++) {
        occ[k] = 1.0/(k+1);
        eig[k] = 0.13*k - 1.0;
        docc[k] = (k % 4)/7.0;
        /* Independent state-outer weighted outer-product reference. */
        for (int p = 0; p < entries; p++) {
            const double a = evec[row0[p]*STATES+k], b = evec[row1[p]*STATES+k];
            const double v = occ[k]*a*b;
            reference[0][p] += v;
            reference[1][p] += v*eig[k];
            reference[2][p] += docc[k]*a*b;
        }
    }
    omp_set_dynamic(0);
    for (int partial = 0; partial < 2; partial++)
        for (size_t t = 0; t < sizeof(teams)/sizeof(teams[0]); t++)
            for (size_t split = 0; split < sizeof(parts)/sizeof(parts[0]); split++) {
                double sum[3][CAPACITY] = {{0}};
                omp_set_num_threads(teams[t]);
                for (int rank = 0; rank < parts[split]; rank++) {
                    const int start = STATES*rank/parts[split];
                    const int nk = STATES*(rank+1)/parts[split] - start;
                    double local[3][CAPACITY] = {{0}};
                    for (int b = 0; b < n; b++)
                        memcpy(packed+b*nk, evec+b*STATES+start, sizeof(double)*(size_t)nk);
                    ClusterCol_AccumulateDM_CPU(entries, nk, mp, occ+start, eig+start,
                                               partial ? docc+start : NULL, packed,
                                               local[0], local[1], partial ? local[2] : NULL);
                    for (int q = 0; q < 2+partial; q++)
                        for (int p = 0; p < entries; p++) sum[q][p] += local[q][p];
                }
                for (int q = 0; q < 2+partial; q++)
                    for (int p = 0; p < entries; p++)
                        CHECK(fabs(sum[q][p]-reference[q][p]) <= 3e-14*(1+fabs(reference[q][p])));
                /* Thread scheduling must not change each entry's result.
                   Compare the same kernel: the independent outer-product
                   reference can have different compiler FMA contractions. */
                if (parts[split] == 1) {
                    if (teams[t] == 1) memcpy(thread_reference, sum, sizeof(sum));
                    for (int q = 0; q < 2+partial; q++)
                        CHECK(memcmp(sum[q], thread_reference[q], sizeof(double)*(size_t)entries) == 0);
                }
            }
    /* A rank with no states need not touch even the topology or input arrays. */
    ClusterCol_AccumulateDM_CPU(0, 0, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
    puts("PASS: CPU cluster DM, 18 combinations of threads/partitions/partial charge; empty rank");
}

static void check_large_offset(void)
{
    const int nk = 32768, mp[] = {0, 65538};
    const size_t row = (size_t)(mp[1]-1)*(size_t)nk;
    const size_t bytes = (row+(size_t)nk)*sizeof(double);
    double *evec, *occ, *eig;
    double dm = 0, edm = 0, expected = 0, expected_e = 0;
    CHECK(row > INT_MAX);
    evec = mmap(NULL, bytes, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE, -1, 0);
    CHECK(evec != MAP_FAILED);
    occ = malloc(sizeof(double)*(size_t)nk);
    eig = malloc(sizeof(double)*(size_t)nk);
    CHECK(occ && eig);
    atomnum = 1;
    WhatSpecies[1] = 0;
    FNAN[1] = 0;
    natn[1][0] = 1;
    for (int k = 0; k < nk; k++) {
        const double a = (k % 17 - 8)/9.0;
        evec[row+(size_t)k] = a;
        occ[k] = 0.25;
        eig[k] = (k % 11)/13.0;
        expected += occ[k]*a*a;
        expected_e += occ[k]*a*a*eig[k];
    }
    ClusterCol_AccumulateDM_CPU(1, nk, (int*)mp, occ, eig, NULL, evec, &dm, &edm, NULL);
    CHECK(fabs(dm-expected) <= 3e-13*(1+fabs(expected)));
    CHECK(fabs(edm-expected_e) <= 3e-13*(1+fabs(expected_e)));
    free(eig);
    free(occ);
    CHECK(munmap(evec, bytes) == 0);
    puts("PASS: CPU cluster DM row offset beyond INT_MAX");
}

int main(void)
{
    check_slices();
    check_large_offset();
    return 0;
}
'''


def main() -> None:
    source = (ROOT / "source/Cluster_DFT_Col.c").read_text()
    helper = extract_helper(source, "ClusterCol_AccumulateDM_CPU")
    with tempfile.TemporaryDirectory(prefix="openmx-cluster-dm-") as tmp:
        cfile, binary = Path(tmp) / "check.c", Path(tmp) / "check"
        cfile.write_text(PREFIX + helper + HARNESS)
        subprocess.run(
            shlex.split(os.environ.get("CC", "cc"))
            + ["-std=c11", "-O2", "-fopenmp", "-Wall", "-Wextra", "-Werror"]
            + shlex.split(os.environ.get("CFLAGS", ""))
            + [str(cfile), "-lm", "-o", str(binary)], check=True,
        )
        subprocess.run([str(binary)], check=True)


if __name__ == "__main__":
    main()

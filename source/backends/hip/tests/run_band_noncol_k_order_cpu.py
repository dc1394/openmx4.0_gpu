#!/usr/bin/env python3
"""Check the production noncollinear k-point schedule without MPI or a GPU.

The helper BandNonCol_InterleaveKOrder is extracted verbatim from
source/Band_DFT_NonCol.c (the AMD port of the NVIDIA test of the same name)
and built against tracking stubs of malloc/free and of
BandNonCol_AbortWithMessage.  Tests cover permutations, stable per-owner
order, uneven/empty owner queues, the order of the former inline AMD
schedule, and the invalid-owner and allocation-failure rejections (the
helper must allocate head, then next, with malloc).  Distinct-owner counts
describe scheduling opportunities, not measured speedups.
Run from any directory; --cc (or CC) and CFLAGS select the C compiler and
options, --sanitize adds ASan/UBSan.
"""

import argparse
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def extract_helper(source: str) -> str:
    name = "BandNonCol_InterleaveKOrder"
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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #condition); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

static void* tracked[2];
static int allocation_calls, fail_allocation;

/* The two helper allocations are tracked so error-path subprocesses can
   terminate cleanly even when leak checking is enabled. malloc(0) is given
   the legal non-NULL result, allowing a zero-point schedule to be tested. */
static void* schedule_malloc(size_t bytes)
{
    void* ptr;
    allocation_calls++;
    if (allocation_calls == fail_allocation) return NULL;
    ptr = malloc(bytes == 0 ? 1 : bytes);
    CHECK(ptr != NULL);
    for (int i = 0; i < 2; i++) {
        if (tracked[i] == NULL) {
            tracked[i] = ptr;
            return ptr;
        }
    }
    CHECK(0);
    return NULL;
}

static void schedule_free(void* ptr)
{
    if (ptr == NULL) return;
    for (int i = 0; i < 2; i++) {
        if (tracked[i] == ptr) {
            tracked[i] = NULL;
            free(ptr);
            return;
        }
    }
    CHECK(0);
}

static void BandNonCol_AbortWithMessage(const char* message)
{
    fprintf(stderr, "%s\n", message);
    for (int i = 0; i < 2; i++) {
        free(tracked[i]);
        tracked[i] = NULL;
    }
    exit(77);
}

#define malloc schedule_malloc
#define free schedule_free
'''


HARNESS = r'''
#undef malloc
#undef free

static unsigned long cases;

/* The inline schedule the AMD version used before the helper (owner
   cursors scanning the owner table round-robin): the helper must
   reproduce its order exactly, so the turn groups do not change. */
static void former_inline_order(int points, int ranks, const int* owner, int* order)
{
    int* next = calloc((size_t)(ranks == 0 ? 1 : ranks), sizeof(int));
    int turn = 0;
    CHECK(next != NULL);
    while (turn < points) {
        for (int rank = 0; rank < ranks; ++rank) {
            while (next[rank] < points && owner[next[rank]] != rank) ++next[rank];
            if (next[rank] < points) order[turn++] = next[rank]++;
        }
    }
    free(next);
}

static void check_layout(int points, int ranks, const int* owner, int* order)
{
    int* seen = calloc((size_t)(points == 0 ? 1 : points), sizeof(int));
    int* last = malloc(sizeof(int) * (size_t)ranks);
    int* original = malloc(sizeof(int) * (size_t)(points == 0 ? 1 : points));
    int* former = malloc(sizeof(int) * (size_t)(points == 0 ? 1 : points));
    CHECK(seen != NULL && last != NULL && original != NULL && former != NULL);
    if (points != 0) memcpy(original, owner, sizeof(int) * (size_t)points);
    for (int rank = 0; rank < ranks; rank++) last[rank] = -1;
    for (int point = 0; point < points; point++) order[point] = -1;
    allocation_calls = 0;
    BandNonCol_InterleaveKOrder(points, ranks, owner, order);
    CHECK(tracked[0] == NULL && tracked[1] == NULL);
    CHECK(allocation_calls == 2);
    for (int position = 0; position < points; position++) {
        const int point = order[position];
        CHECK(0 <= point && point < points);
        CHECK(seen[point] == 0);
        seen[point] = 1;
        CHECK(0 <= owner[point] && owner[point] < ranks);
        CHECK(last[owner[point]] < point);
        last[owner[point]] = point;
    }
    for (int point = 0; point < points; point++) CHECK(seen[point] == 1);
    CHECK(points == 0 || memcmp(original, owner, sizeof(int) * (size_t)points) == 0);
    former_inline_order(points, ranks, owner, former);
    CHECK(points == 0 || memcmp(former, order, sizeof(int) * (size_t)points) == 0);
    free(former); free(original); free(last); free(seen);
    cases++;
}

static int distinct_owners(int points, int ranks, const int* owner, const int* order, int group)
{
    int total = 0;
    int* seen = calloc((size_t)ranks, sizeof(int));
    CHECK(seen != NULL);
    for (int first = 0; first < points; first += group) {
        const int end = first + group < points ? first + group : points;
        memset(seen, 0, sizeof(int) * (size_t)ranks);
        for (int position = first; position < end; position++) {
            const int rank = owner[order[position]];
            if (!seen[rank]) { seen[rank] = 1; total++; }
        }
    }
    free(seen);
    return total;
}

int main(int argc, char** argv)
{
    if (argc > 1) {
        int owner[] = {0, 1, 2, 0}, order[4];
        if (strcmp(argv[1], "negative-owner") == 0) owner[2] = -1;
        else if (strcmp(argv[1], "upper-owner") == 0) owner[2] = 3;
        else if (strcmp(argv[1], "huge-owner") == 0) owner[2] = INT_MAX;
        else if (strcmp(argv[1], "oom-head") == 0) fail_allocation = 1;
        else if (strcmp(argv[1], "oom-next") == 0) fail_allocation = 2;
        else return 2;
        BandNonCol_InterleaveKOrder(4, 3, owner, order);
        return 3; /* Every mode above must reach the abort stub. */
    }

    /* Exhaustive small layouts include interspersed owners, ragged queues,
       all points on one owner, empty owners, and more ranks than points. */
    for (int ranks = 1; ranks <= 4; ranks++) {
        unsigned long layouts = 1;
        for (int points = 0; points <= 7; points++) {
            int owner[7], order[7];
            if (points != 0) layouts *= (unsigned long)ranks;
            for (unsigned long code = 0; code < layouts; code++) {
                unsigned long digits = code;
                for (int point = 0; point < points; point++) {
                    owner[point] = (int)(digits % (unsigned long)ranks);
                    digits /= (unsigned long)ranks;
                }
                check_layout(points, ranks, owner, order);
            }
        }
    }
    {
        const int owner[] = {17, 17, 0, 9, 17};
        const int expected[] = {2, 3, 0, 1, 4};
        int order[5];
        check_layout(5, 32, owner, order);
        CHECK(memcmp(order, expected, sizeof(expected)) == 0);
    }
    {
        int owner[997], order[997];
        /* Contiguous skew: one long queue, singleton queues and empty ranks. */
        for (int point = 0; point < 997; point++)
            owner[point] = point < 990 ? 6 : 2 * (point - 990);
        check_layout(997, 32, owner, order);
    }
    {
        enum { POINTS = 1000, RANKS = 18 };
        int owner[POINTS], original[POINTS], order[POINTS];
        const int groups[] = {2, 4, 8, 18};
        /* Same contiguous 1000/18 ownership boundaries used by OpenMX. */
        for (int rank = 0; rank < RANKS; rank++) {
            const int first = (int)((double)rank * ((double)POINTS / RANKS + 1.0e-12));
            const int end = rank == RANKS - 1 ? POINTS
                : (int)((double)(rank + 1) * ((double)POINTS / RANKS + 1.0e-12));
            for (int point = first; point < end; point++) owner[point] = rank;
        }
        for (int point = 0; point < POINTS; point++) original[point] = point;
        check_layout(POINTS, RANKS, owner, order);
        for (int g = 0; g < 4; g++) {
            const int group = groups[g];
            const int count = (POINTS + group - 1) / group;
            const int before = distinct_owners(POINTS, RANKS, owner, original, group);
            const int after = distinct_owners(POINTS, RANKS, owner, order, group);
            CHECK(after > before);
            CHECK(after >= POINTS - RANKS); /* Allow the uneven final owner round. */
            printf("1000 k points / 18 owners, group %d: distinct owners/group %.3f -> %.3f\n",
                group, (double)before / count, (double)after / count);
        }
    }
    printf("PASS: %lu schedule layouts; permutation, stable per-owner order, unchanged ownership, "
           "the former inline order, two allocations\n", cases);
    return 0;
}
'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cc", default=os.environ.get("CC", "cc"), help="C compiler (default: $CC or cc)")
    parser.add_argument("--sanitize", action="store_true", help="build with ASan/UBSan")
    args = parser.parse_args()
    try:
        helper = extract_helper((ROOT / "source" / "Band_DFT_NonCol.c").read_text())
        compiler = shlex.split(args.cc)
        if not compiler:
            raise ValueError("--cc must name a C compiler")
        flags = shlex.split(os.environ.get("CFLAGS", "-O2"))
        if args.sanitize:
            flags += ["-g", "-fsanitize=address,undefined", "-fno-omit-frame-pointer"]
        with tempfile.TemporaryDirectory(prefix="openmx-k-order-cpu-") as directory:
            source = Path(directory) / "k_order.c"
            binary = Path(directory) / "k_order"
            source.write_text(PREFIX + "\n" + helper + "\n" + HARNESS)
            subprocess.run([*compiler, *flags, "-std=c11", str(source), "-o", str(binary)], check=True)
            env = os.environ.copy()
            if args.sanitize:
                # the error paths exit through the stub on purpose
                env.setdefault("ASAN_OPTIONS", "detect_leaks=1")
            subprocess.run([str(binary)], check=True, env=env)
            for mode in ("negative-owner", "upper-owner", "huge-owner", "oom-head", "oom-next"):
                result = subprocess.run([str(binary), mode], capture_output=True, text=True, env=env)
                expected = "Failed to allocate" if mode.startswith("oom-") else "Invalid owner"
                if result.returncode != 77 or expected not in result.stderr:
                    raise ValueError(f"{mode}: expected rejection, got {result.returncode}: {result.stderr}")
            print("PASS: invalid-owner and allocation-failure rejection (5 cases)")
        return 0
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"Noncollinear k-order CPU test failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

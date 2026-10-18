#!/usr/bin/env python3
"""Stress production integral caches and per-atom timing updates with pthreads.

No MPI, CUDA or PAO data is required. Build the C++ core first and set
OPENMX_CORE_LIBRARY to its libopenmx_core.so when using a nondefault build. The actual cache helpers and accumulation
statements are extracted from Set_OLP_Kin and Set_Nonlocal. All workers update
one atom to expose lost timing updates at pair-partition boundaries, while
private Bessel tables must reproduce uncached values for different cutoffs.
CC/CFLAGS are honored; --helgrind additionally checks thread data races.
"""
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
CORE_LIBRARY = Path(os.environ.get("OPENMX_CORE_LIBRARY", ROOT / "build/core-serial/libopenmx_core.so")).resolve()
if not CORE_LIBRARY.is_file():
    raise SystemExit("Build the C++ core and set OPENMX_CORE_LIBRARY to libopenmx_core.so")


def function(source, declaration):
    start = source.index(declaration)
    opening = source.index("{", start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


olp = (ROOT / "source/Set_OLP_Kin.c").read_text()
nonlocal_source = (ROOT / "source/Set_Nonlocal.c").read_text()
spherical = (ROOT / "source/Spherical_Bessel.c").read_text()
cache = olp[olp.index("#define OLPKIN_BESSEL_CACHE_DEFAULT"):
            olp.index("/* Runtime phase profiling")]
profile = function(olp, "static int SetOLP_ProfileEnabled(void)")
bessel = function(spherical, "void Spherical_Bessel(")

# Extract the actual update, including its synchronization, in each pair loop.
updates = []
for name, source in (("olp", olp), ("nonlocal", nonlocal_source)):
    update = re.search(
        r"dtime\(&Etime_atom\);\s*\n"
        r"(?P<update>(?:#pragma omp atomic[^\n]*\n)?"
        r"\s*time_per_atom\[Gc_AN\] \+= Etime_atom - Stime_atom;)", source
    ).group("update")
    updates.append(f"static void update_{name}(void) {{\n"
                   "  const int Gc_AN = 0;\n"
                   "  const double Stime_atom = 1.0, Etime_atom = 2.0;\n"
                   + update + "\n}\n")

HARNESS = r'''
#define THREADS 8
#define UPDATES 50000
static pthread_barrier_t start_gate;

static void *worker(void *argument)
{
    const long tid = (long)argument;
    pthread_barrier_wait(&start_gate);
    const int cache_max = OLPKin_Bessel_CacheMax();
    assert(cache_max == 8);
    assert(SetOLP_ProfileEnabled() == 1);
    for (int call = 0; call < 3; ++call) {
        OLPKinBesselCache c = {0};
        c.lmax = 2 + call;
        c.gdim = 5 + (int)tid;
        for (int pair = 0; pair < 96; ++pair) {
            const double r = 0.5 + 0.25 * (pair % 3);
            const int lmax = 1 + pair % 2;
            const double *hit = OLPKin_Bessel_Lookup(&c, r, lmax);
            double *slot = hit ? NULL : OLPKin_Bessel_Insert(&c, r, lmax, cache_max);
            for (int i = 0; i < c.gdim; ++i) {
                double sb[40], dsb[40];
                Spherical_Bessel((0.1 + i * 0.5) * r, lmax, sb, dsb);
                for (int l = 0; l <= lmax; ++l) {
                    const size_t a = (size_t)l * c.gdim + i;
                    const size_t b = (size_t)(c.lmax + 1 + l) * c.gdim + i;
                    if (hit) {
                        assert(hit[a] == sb[l]);
                        assert(hit[b] == dsb[l]);
                    } else if (slot) {
                        slot[a] = sb[l];
                        slot[b] = dsb[l];
                    }
                }
            }
        }
        assert(c.count == 6 && c.hits == 90 && c.misses == 6);
        OLPKin_Bessel_Free(&c);
        assert(c.keys == NULL && c.tab == NULL && c.count == 0);
        assert(OLPKin_Bessel_Insert(&c, 1.0, 1, 0) == NULL);
    }
    pthread_barrier_wait(&start_gate);
    for (int i = 0; i < UPDATES; ++i) {
        update_olp();
        update_nonlocal();
    }
    return NULL;
}

int main(void)
{
    pthread_t threads[THREADS];
    setenv("OPENMX_OLPKIN_BESSEL_CACHE_MAX", "8", 1);
    setenv("OPENMX_BAND_PROFILE", "1", 1);
    assert(pthread_barrier_init(&start_gate, NULL, THREADS) == 0);
    for (long t = 0; t < THREADS; ++t)
        assert(pthread_create(&threads[t], NULL, worker, (void*)t) == 0);
    for (int t = 0; t < THREADS; ++t)
        assert(pthread_join(threads[t], NULL) == 0);
    pthread_barrier_destroy(&start_gate);
    assert(time_per_atom[0] == 2.0 * THREADS * UPDATES);
    puts("PASS: integral thread caches and shared timing accumulations");
    return 0;
}
'''

with tempfile.TemporaryDirectory(prefix="openmx-integral-threads-") as tmp:
    src, exe = Path(tmp) / "check.c", Path(tmp) / "check"
    src.write_text("#define _POSIX_C_SOURCE 200809L\n"
                   "#include <assert.h>\n#include <math.h>\n#include <pthread.h>\n"
                   "#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n"
                   '#include "openmx_core.h"\n'
                   "static double time_per_atom[1];\n"
                   + cache + profile + bessel + "\n".join(updates) + HARNESS)
    compiler = shlex.split(os.environ.get("CC", "cc"))
    flags = shlex.split(os.environ.get("CFLAGS", "-O2 -g"))
    subprocess.run(compiler + flags + ["-std=c11", "-fopenmp", "-pthread",
                   "-I" + str(ROOT / "source"), str(src), str(CORE_LIBRARY),
                   "-Wl,-rpath," + str(CORE_LIBRARY.parent), "-lm", "-o", str(exe)], check=True)
    runner = (["valgrind", "--tool=helgrind", "--error-exitcode=99", "--quiet"]
              if "--helgrind" in sys.argv[1:] else [])
    subprocess.run(runner + [str(exe)], check=True)

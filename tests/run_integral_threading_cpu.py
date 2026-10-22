#!/usr/bin/env python3
"""Stress production integral caches and per-atom timing updates with pthreads.

No MPI, CUDA or PAO data is required. Build the C++ core first and set
OPENMX_CORE_LIBRARY to its libopenmx_core.so when using a nondefault build.
The production C ABI owns each cache; runtime-option helpers and accumulation
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
nonlocal_profile = function(nonlocal_source, "static int SetNonlocal_ProfileEnabled(void)")
bessel = function(spherical, "void Spherical_Bessel(")

# Extract the actual update, including its synchronization, in each pair loop.
updates = []
for name, source in (("olp", olp), ("nonlocal", (ROOT / "source/openmx_nonlocal_cpu.h").read_text())):
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
        const int grid_points = 5 + (int)tid;
        double wavevectors[THREADS+5];
        char error[128];
        for (int i = 0; i < grid_points; ++i) wavevectors[i] = 0.1 + i * 0.5;
        OpenMXPairBesselWorkspace *workspace = openmx_core_pair_bessel_create(
            2 + call,grid_points,wavevectors,(size_t)cache_max,error,sizeof(error));
        assert(workspace != NULL);
        const OpenMXOverlapShape shape = {2+call,2,grid_points,0.1,
            0.1+(grid_points-1)*0.5,(size_t)cache_max};
        OpenMXOverlapBuffers buffers;
        OpenMXOverlapWorkspace *overlap = openmx_core_overlap_create(&shape,&buffers,error,sizeof(error));
        assert(overlap != NULL);
        const int projector_counts[2] = {2,0};
        int projector_rows[2][3] = {{0,0,2},{0,0,0}};
        const OpenMXNonlocalShape nonlocal_shape = {2+call,2,2,2,projector_counts,
            &projector_rows[0][0],3,grid_points,wavevectors,(size_t)cache_max};
        OpenMXNonlocalBuffers nonlocal_buffers;
        OpenMXNonlocalWorkspace *nonlocal = openmx_core_nonlocal_create(
            &nonlocal_shape,&nonlocal_buffers,error,sizeof(error));
        assert(nonlocal != NULL);
        const size_t last = openmx_core_overlap_pair(&buffers,2+call,1,2*(2+call),2+call,1,2*(2+call));
        const double marker = 1.0+tid+call*THREADS;
        assert(buffers.angular[7][last].r == 0.0);
        buffers.angular[7][last].r = marker;
        for (int pair = 0; pair < 96; ++pair) {
            const double r = 0.5 + 0.25 * (pair % 3);
            const int lmax = 1 + pair % 2;
            OpenMXPairBesselTable table;
            assert(openmx_core_pair_bessel_sample(workspace,r,lmax,&table,error,sizeof(error)));
            OpenMXPairBesselTable overlap_table;
            assert(openmx_core_overlap_sample(overlap,r,lmax,&overlap_table,error,sizeof(error)));
            assert(buffers.angular[7][last].r == marker);
            OpenMXPairBesselTable nonlocal_table;
            assert(openmx_core_nonlocal_sample(nonlocal,r,lmax,&nonlocal_table,error,sizeof(error)));
            for (int i = 0; i < grid_points; ++i) {
                double sb[40], dsb[40];
                Spherical_Bessel((0.1 + i * 0.5) * r, lmax, sb, dsb);
                double overlap_sb[40], overlap_dsb[40];
                Spherical_Bessel(buffers.wavevectors[i]*r,lmax,overlap_sb,overlap_dsb);
                for (int l = 0; l <= lmax; ++l) {
                    const size_t offset = (size_t)l * grid_points + i;
                    assert(table.values[offset] == sb[l]);
                    assert(table.derivatives[offset] == dsb[l]);
                    assert(nonlocal_table.values[offset] == sb[l]);
                    assert(nonlocal_table.derivatives[offset] == dsb[l]);
                    assert(overlap_table.values[offset] == overlap_sb[l]);
                    assert(overlap_table.derivatives[offset] == overlap_dsb[l]);
                }
            }
        }
        long hits, misses;
        openmx_core_pair_bessel_statistics(workspace,&hits,&misses);
        assert(hits == 90 && misses == 6);
        openmx_core_pair_bessel_destroy(workspace);
        openmx_core_overlap_statistics(overlap,&hits,&misses);
        assert(hits == 90 && misses == 6);
        openmx_core_overlap_destroy(overlap);
        openmx_core_nonlocal_destroy(nonlocal);
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
    assert(SetNonlocal_ProfileEnabled() == 1);
    setenv("OPENMX_BAND_PROFILE", "0", 1);
    assert(SetNonlocal_ProfileEnabled() == 0);
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
                   '#include "openmx_core_pair_bessel.h"\n'
                   '#include "openmx_core_overlap.h"\n'
                   '#include "openmx_core_nonlocal.h"\n'
                   "static double time_per_atom[1];\n"
                   + cache + profile + nonlocal_profile + bessel + "\n".join(updates) + HARNESS)
    compiler = shlex.split(os.environ.get("CC", "cc"))
    flags = shlex.split(os.environ.get("CFLAGS", "-O2 -g"))
    subprocess.run(compiler + flags + ["-std=c11", "-D_POSIX_C_SOURCE=200809L", "-fopenmp", "-pthread",
                   "-I" + str(ROOT / "source"), "-include", str(ROOT / "source/openmx_env.h"), "-include", "limits.h", str(ROOT / "source/openmx_env.c"), str(src), str(CORE_LIBRARY),
                   "-Wl,-rpath," + str(CORE_LIBRARY.parent), "-lm", "-o", str(exe)], check=True)
    runner = (["valgrind", "--tool=helgrind", "--error-exitcode=99", "--quiet"]
              if "--helgrind" in sys.argv[1:] else [])
    subprocess.run(runner + [str(exe)], check=True)

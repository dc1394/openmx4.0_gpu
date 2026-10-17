#!/usr/bin/env python3
"""Host-memory estimate of COSMA's transforms (source/elpa_cosma_memory.cpp).

The gpusolver2 bridge runs a transform on COSMA only when the host memory
that COSMA keeps afterwards (its memory pool and COSTA's send/receive
buffers) fits the node.  This compiles the production estimate against the
installed COSMA (third_party/gpusolver2-install) with the BLACS grid
queries and COSMA's grid adaptation replaced by recording stubs, and checks
it against COSMA's own Strategy for the orders of the capacity fallback
(Si1000 noncollinear: complex 26000 on 16 ranks; Si1440: real 18720,
complex 37440): the unlimited strategy, COSMA_CPU_MAX_MEMORY, a limit that
COSMA cannot meet (-1, without COSMA's diagnosis on stdout), the forwarding
of the grid adaptation and of the transpose flags.

  python3 tests/run_cosma_host_memory_cpu.py
  (CXX: the C++ compiler of the COSMA build, default /opt/gcc/12.2.0/bin/g++
   or g++; MPI_INCLUDE: directory of mpi.h, default from mpicc --showme)
"""
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile

root = Path(__file__).resolve().parents[1]
prefix = Path(os.environ.get("GPUSOLVER2_PREFIX", root / "source/third_party/gpusolver2-install"))
if not (prefix / "lib/libcosma.a").exists():
    print(f"SKIP: no COSMA installation under {prefix}")
    sys.exit(0)

cxx = os.environ.get("CXX") or ("/opt/gcc/12.2.0/bin/g++" if Path("/opt/gcc/12.2.0/bin/g++").exists() else "g++")
mpi_include = os.environ.get("MPI_INCLUDE")
if not mpi_include:
    try:
        mpi_include = subprocess.run(["mpicc", "--showme:incdirs"], check=True, capture_output=True,
                                     text=True).stdout.split()[0]
    except (OSError, subprocess.CalledProcessError, IndexError):
        print("SKIP: set MPI_INCLUDE to the directory of mpi.h")
        sys.exit(0)

driver = r'''
#include <cosma/cosma_pxgemm.hpp>
#include <cosma/scalapack.hpp>
#include <cosma/strategy.hpp>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

extern "C" double openmx_gs2_cosma_host_bytes(char, char, int, int, int, const int *,
                                              const int *, const int *, int, int);

/* BLACS grid of the test: GRID_ROWS x GRID_COLS, row-major */
static int rows() { return std::atoi(std::getenv("GRID_ROWS")); }
static int cols() { return std::atoi(std::getenv("GRID_COLS")); }
extern "C" void Cblacs_gridinfo(int, int *r, int *c, int *mr, int *mc) { *r = rows(); *c = cols(); *mr = *mc = 0; }

/* COSMA's grid adaptation: record the call, return the prefix of
   PREFIX_STEP ("s<dim><div>" or empty) */
static int adapt_calls = 0;
static char adapt_ta = 0, adapt_tb = 0, adapt_order = 0;
namespace cosma {
namespace scalapack {
costa::scalapack::ordering rank_ordering(int, int) { return costa::scalapack::ordering::row_major; }
}
void adapt_strategy_to_block_cyclic_grid(std::vector<int> &divisors, std::string &dimensions,
                                         std::string &step_type, int, int, int, int,
                                         scalapack::global_matrix_size &, scalapack::global_matrix_size &,
                                         scalapack::global_matrix_size &, scalapack::block_size &,
                                         scalapack::block_size &, scalapack::block_size &,
                                         int, int, int, int, int, int, char ta, char tb,
                                         int, int, char order)
{
    ++adapt_calls; adapt_ta = ta; adapt_tb = tb; adapt_order = order;
    const char *p = std::getenv("PREFIX_STEP");
    if (p && *p) { step_type += p[0]; dimensions += p[1]; divisors.push_back(std::atoi(p + 2)); }
}
}

int main(int argc, char **argv)
{
    const int m = std::atoi(argv[1]), n = std::atoi(argv[2]), k = std::atoi(argv[3]);
    const int z = std::atoi(argv[4]);
    const char ta = argv[5][0], tb = argv[6][0];
    int desc[9] = {1, 0, m, n, 128, 128, 0, 0, 1};
    const double got = openmx_gs2_cosma_host_bytes(ta, tb, m, n, k, desc, desc, desc, 0, z);

    /* COSMA's own strategy for the same problem (the reference) */
    const int P = rows() * cols();
    long long limit = std::numeric_limits<long long>::max();
    const char *v = std::getenv("COSMA_CPU_MAX_MEMORY");
    const long long elem = z ? 16 : 8;
    if (v) limit = std::atoll(v) * 1024LL * 1024LL / elem;
    std::vector<int> divs; std::string dims, types;
    const char *p = std::getenv("PREFIX_STEP");
    if (p && *p && P > 1) { types += p[0]; dims += p[1]; divs.push_back(std::atoi(p + 2)); }
    long long used = -1;
    {
        /* keep COSMA's own diagnosis of this reference off stdout, so that
           anything printed comes from the estimate under test */
        std::streambuf *out = std::cout.rdbuf(nullptr);
        try {
            cosma::Strategy s(m, n, k, P, divs, dims, types, limit);
            used = s.memory_used;
        } catch (...) {}
        std::cout.rdbuf(out);
        std::cout.clear();
    }
    std::printf("RESULT %.0f %lld %d %c%c%c\n", got, used, adapt_calls,
                adapt_ta ? adapt_ta : '-', adapt_tb ? adapt_tb : '-', adapt_order ? adapt_order : '-');
    return 0;
}
'''

failures = 0


def check(cond, what):
    global failures
    if not cond:
        failures += 1
        print("FAIL:", what)


with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    (tmp / "driver.cpp").write_text(driver)
    exe = tmp / "probe"
    flags = ["-std=c++17", "-O1", f"-I{prefix}/include", f"-I{mpi_include}"]
    cmd = [cxx, *flags, str(root / "source/elpa_cosma_memory.cpp"), str(tmp / "driver.cpp"),
           str(prefix / "lib/libcosma.a"), "-o", str(exe)]
    subprocess.run(cmd, check=True)
    gcc_lib = Path(cxx).resolve().parent.parent / "snos/lib64"
    env_base = dict(os.environ)
    if gcc_lib.is_dir():
        env_base["LD_LIBRARY_PATH"] = f"{gcc_lib}:{env_base.get('LD_LIBRARY_PATH', '')}"
    for key in ("COSMA_CPU_MAX_MEMORY", "COSMA_ADAPT_STRATEGY", "PREFIX_STEP"):
        env_base.pop(key, None)

    def run(m, n, k, z, ta="N", tb="N", rows=4, cols=4, **env):
        e = dict(env_base, GRID_ROWS=str(rows), GRID_COLS=str(cols), **env)
        out = subprocess.run([str(exe), str(m), str(n), str(k), str(z), ta, tb], env=e, check=True,
                             capture_output=True, text=True).stdout
        lines = out.splitlines()
        check(len(lines) == 1, f"only the result line on stdout, got {lines}")
        _, got, used, calls, flags_ = lines[-1].split()
        return float(got), int(used), int(calls), flags_

    def expected(m, n, k, z, used, P=16):
        elem = 16 if z else 8
        ab = (m * k + k * n) / P
        c = m * n / P
        return used * elem + 2.0 * 1.1 * max(ab, c) * elem

    cases = [
        (26000, 26000, 26000, 1, 464750000),   # spinor order of Si1000 NC
        (13000, 13000, 13000, 0, 116187500),   # its overlap order
        (37440, 37440, 37440, 1, 963705600),   # Si1440 NC
        (18720, 18720, 18720, 0, 240926400),   # Si1440 collinear
    ]
    for m, n, k, z, used_ref in cases:
        got, used, calls, flags_ = run(m, n, k, z)
        check(used == used_ref, f"n={m}: COSMA strategy memory {used} (expected {used_ref})")
        check(abs(got - expected(m, n, k, z, used)) < 1.0, f"n={m} z={z}: estimate {got:.0f}")
        check(calls == 1 and flags_ == "NNR", f"n={m}: adaptation called once with N,N,R ({calls}, {flags_})")
        print(f"n={m} {'complex' if z else 'real'} on 16 ranks: {got / 1e9:.2f} GB per rank "
              f"(pool {used * (16 if z else 8) / 1e9:.2f} GB)")

    # the transpose flags reach the adaptation as given (upper case)
    got, used, calls, flags_ = run(26000, 26000, 26000, 1, "t", "T")
    check(flags_ == "TTR", f"transpose flags forwarded: {flags_}")

    # COSMA_CPU_MAX_MEMORY (MiB per rank) bounds the pool; the estimate follows it
    got, used, calls, flags_ = run(26000, 26000, 26000, 1, COSMA_CPU_MAX_MEMORY="4000")
    check(0 < used <= 4000 * 1048576 // 16 and used < 464750000, f"limited strategy memory {used}")
    check(abs(got - expected(26000, 26000, 26000, 1, used)) < 1.0, "limited estimate")

    # a limit COSMA cannot meet: -1, and COSMA's diagnosis stays off stdout
    got, used, calls, flags_ = run(26000, 26000, 26000, 1, COSMA_CPU_MAX_MEMORY="1500")
    check(got == -1.0 and used == -1, f"infeasible limit gives -1 ({got}, {used})")

    # COSMA_ADAPT_STRATEGY=0: no adaptation call
    got, used, calls, flags_ = run(26000, 26000, 26000, 1, COSMA_ADAPT_STRATEGY="0")
    check(calls == 0, f"adaptation skipped when disabled ({calls})")

    # an adapted prefix enters the strategy (a sequential split of m by 2)
    got0, used0, _, _ = run(16384, 16384, 16384, 1)
    got1, used1, calls, _ = run(16384, 16384, 16384, 1, PREFIX_STEP="sm2")
    check(calls == 1 and used1 != used0, f"adapted prefix used ({used0} -> {used1})")
    check(abs(got1 - expected(16384, 16384, 16384, 1, used1)) < 1.0, "adapted estimate")

    # one rank: no adaptation, the whole problem
    got, used, calls, flags_ = run(2000, 2000, 2000, 0, rows=1, cols=1)
    check(calls == 0 and used > 0 and abs(got - expected(2000, 2000, 2000, 0, used, P=1)) < 1.0,
          f"single rank ({calls}, {used})")

if failures:
    print(f"{failures} check(s) failed")
    sys.exit(1)
print("all COSMA host-memory checks passed")

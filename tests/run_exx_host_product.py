#!/usr/bin/env python3
"""Run the host products of the exchange build against a model of the BLAS.

source/Exact_Exchange_Host.c is linked with
tests/exx_host_product_harness.c, which defines DGEMM and the functions of
OpenBLAS that set its number of threads. The pair is compiled five times:
with OpenMP, without OpenMP, and with OpenMP in three further forms of the
functions of OpenBLAS: none, as in a program linked with another BLAS;
without openblas_get_parallel, as with a static OpenBLAS; and referred to
directly, as in a build that has chosen OpenBLAS itself. The first program
runs a second time with a limit of two threads, so that the teams are
smaller than the unit asks for. No BLAS and no MPI are needed.

With --thread-sanitizer the pair is compiled once, with OpenMP and with
ThreadSanitizer in place of the other sanitizers, and run with Archer, the
tool of the OpenMP runtime of LLVM that tells ThreadSanitizer how OpenMP
synchronizes its threads. This needs Clang and its OpenMP runtime (CC names
the compiler). Archer is looked for beside the libraries of the compiler;
OMP_TOOL_LIBRARIES names it otherwise.
"""
from pathlib import Path
import argparse
import os
import shlex
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
# Seconds for one run of a program. The runs together stay below the timeout
# of the ctest exx-host-product (CMakeLists.txt), so that a program that
# hangs is named here.
RUN_TIMEOUT = 120
# (compiler options, the environments of the runs)
VARIANTS = (
    (['-fopenmp'], ({}, {'OMP_THREAD_LIMIT': '2', 'KMP_WARNINGS': '0'})),
    ([], ({},)),
    (['-fopenmp', '-DOPENMX_TEST_WITHOUT_OPENBLAS'], ({},)),
    (['-fopenmp', '-DOPENMX_TEST_WITHOUT_OPENBLAS_PARALLEL'], ({},)),
    (['-fopenmp', '-DOPENMX_EXX_HOST_BLAS_OPENBLAS'], ({},)),
)

parser = argparse.ArgumentParser(description=__doc__.split('\n', 1)[0])
parser.add_argument('--thread-sanitizer', action='store_true',
                    help='one program with OpenMP and ThreadSanitizer, run with Archer')
arguments = parser.parse_args()
compiler = shlex.split(os.environ.get('CC', 'cc'))


def joined(name, *options):
    """The options of a sanitizer: those of the environment, then ours."""
    return ':'.join([option for option in (os.environ.get(name),) if option] + list(options))


def build(binary, sanitizers, variant):
    subprocess.run(compiler + [
        '-std=gnu11', '-O1', '-g', '-Wall', '-Wextra', '-Werror', '-ffp-contract=off',
        *sanitizers, *variant, '-I'+str(ROOT/'source'),
        str(ROOT/'source/Exact_Exchange_Host.c'), str(ROOT/'tests/exx_host_product_harness.c'),
        '-lm', '-o', str(binary)], check=True)


with tempfile.TemporaryDirectory(prefix='openmx-exx-host-') as directory:
    if arguments.thread_sanitizer:
        # Without Archer, ThreadSanitizer does not see the barriers and locks
        # of the OpenMP runtime and reports races that are none.
        archer = os.environ.get('OMP_TOOL_LIBRARIES')
        if not archer:
            def answer(option):
                return subprocess.run(compiler + [option], capture_output=True, text=True).stdout.strip()
            # the library path of the compiler, or lib/ above lib/clang/<version>
            places = [Path(answer('-print-file-name=libarcher.so'))]
            resources = Path(answer('-print-resource-dir'))
            if len(resources.parents) > 1:
                places.append(resources.parents[1]/'libarcher.so')
            found = [place for place in places if place.is_absolute() and place.is_file()]
            if not found:
                sys.exit('run_exx_host_product.py: no libarcher.so beside the libraries of the compiler '
                         '(Clang with its OpenMP runtime is needed; OMP_TOOL_LIBRARIES can name the library)')
            archer = str(found[0])
        binary = Path(directory)/'harness-threads'
        build(binary, ['-fsanitize=thread'], ['-fopenmp'])
        environment = dict(os.environ)
        environment['OMP_TOOL_LIBRARIES'] = archer
        environment['TSAN_OPTIONS'] = joined('TSAN_OPTIONS', 'ignore_noninstrumented_modules=1', 'halt_on_error=1')
        subprocess.run([str(binary)], check=True, timeout=5*RUN_TIMEOUT, env=environment)
        sys.exit(0)
    # LeakSanitizer reports memory of the OpenMP runtime of LLVM itself when
    # the program ends: the runtime loses blocks of its allocator when a team
    # becomes larger than the earlier teams allowed for (Clang 18.1.3 with
    # libomp 18.1.3 on 4 cores: a program that only runs a team of 8 threads
    # and then one of 9 gets the report). Memory from that allocator is left
    # out of the leak check, and the log lists what was left out. The pattern
    # names the allocator and not the library: every allocation made inside a
    # parallel region has frames of the library below it, and a leak of the
    # test or of the unit there must stay an error. The runtime of GCC gives
    # no such report.
    suppressions = Path(directory)/'leaks'
    suppressions.write_text('leak:___kmp_allocate\n', encoding='utf-8')
    for number, (variant, runs) in enumerate(VARIANTS):
        binary = Path(directory)/f'harness{number}'
        build(binary, ['-fsanitize=address,undefined', '-fno-sanitize-recover=all'], variant)
        for settings in runs:
            environment = dict(os.environ)
            environment['LSAN_OPTIONS'] = joined('LSAN_OPTIONS', f'suppressions={suppressions}', 'print_suppressions=1')
            if 'OMP_THREAD_LIMIT' in settings:
                # Under a limit of threads the runtime of LLVM also loses
                # memory that it takes from malloc in functions without a
                # name. No pattern tells that memory from a leak of the unit
                # inside a parallel region, so this run does not check leaks.
                environment['ASAN_OPTIONS'] = joined('ASAN_OPTIONS', 'detect_leaks=0')
            environment.update(settings)
            subprocess.run([str(binary)], check=True, timeout=RUN_TIMEOUT, env=environment)

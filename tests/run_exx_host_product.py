#!/usr/bin/env python3
"""Run the host products of the exchange build against a model of the BLAS.

source/Exact_Exchange_Host.c is linked with
tests/exx_host_product_harness.c, which defines DGEMM and the functions of
OpenBLAS that set its number of threads. The pair is compiled three times:
with OpenMP, without OpenMP, and with OpenMP but without the functions of
OpenBLAS, as in a program linked with another BLAS. No BLAS and no MPI are
needed.
"""
from pathlib import Path
import os
import shlex
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = (['-fopenmp'], [], ['-fopenmp', '-DOPENMX_TEST_WITHOUT_OPENBLAS'])

with tempfile.TemporaryDirectory(prefix='openmx-exx-host-') as directory:
    # The OpenMP runtime of LLVM keeps no pointer to some of its own memory
    # after a team of threads has shrunk, and LeakSanitizer reports that
    # memory when the program ends (Clang 18.1.3 with libomp 18.1.3: a
    # program that only changes the size of its teams gets the same report).
    # Memory that the runtime allocated is left out of the leak check; a
    # leak of the test or of the unit stays an error.
    suppressions = Path(directory)/'leaks'
    suppressions.write_text('leak:libomp\n', encoding='utf-8')
    environment = dict(os.environ)
    environment['LSAN_OPTIONS'] = ':'.join(
        [option for option in (os.environ.get('LSAN_OPTIONS'),) if option]
        + [f'suppressions={suppressions}', 'print_suppressions=0'])
    for number, variant in enumerate(VARIANTS):
        binary = Path(directory)/f'harness{number}'
        command = shlex.split(os.environ.get('CC', 'cc')) + [
            '-std=gnu11', '-O1', '-g', '-Wall', '-Wextra', '-Werror', '-ffp-contract=off',
            '-fsanitize=address,undefined', '-fno-sanitize-recover=all', *variant,
            '-I'+str(ROOT/'source'),
            str(ROOT/'source/Exact_Exchange_Host.c'), str(ROOT/'tests/exx_host_product_harness.c'),
            '-lm', '-o', str(binary)]
        subprocess.run(command, check=True)
        subprocess.run([str(binary)], check=True, timeout=300, env=environment)

#!/usr/bin/env python3
"""Run the actual simple-mixing algorithm with calculation-owned history.

Only MPI reduction is modeled (one rank). The full production mixing and its
sign/maximum helpers run unchanged against the actual C++ shared owner.
"""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile
from native_state_test_support import bounded_stack, compiler_command
from run_cluster_col_state_cpu import function

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core-library', type=Path, required=True)
    args = parser.parse_args()
    library = args.core_library.resolve(strict=True)
    helpers = (ROOT / 'source/openmx_common.c').read_text()
    mixing = (ROOT / 'source/Simple_Mixing_DM.c').read_text()
    generated = '\n'.join([function(helpers, 'sgn'), function(helpers, 'largest'),
                            function(mixing, 'Simple_Mixing_DM')])
    with tempfile.TemporaryDirectory(prefix='openmx-mixing-history-') as directory:
        folder = Path(directory)
        (folder / 'mixing_history.inc').write_text(generated)
        executable = folder / 'check'
        command = compiler_command('c++') + [
            '-I' + str(folder), '-I' + str(ROOT / 'core/include'),
            str(ROOT / 'tests/mixing_history_harness.cpp'), str(library),
            '-Wl,-rpath,' + str(library.parent), '-o', str(executable)]
        subprocess.run(command, check=True)
        subprocess.run([str(executable)], env=os.environ.copy(), check=True,
                       timeout=120, preexec_fn=bounded_stack)


if __name__ == '__main__':
    main()

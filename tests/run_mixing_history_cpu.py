#!/usr/bin/env python3
"""Run the actual simple-mixing algorithm with calculation-owned history.

MPI reduction and scalar input parsing are modeled. Production mixing,
sign/maximum helpers, input binding and SCF output use the actual C++ owner.
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
    input_source = (ROOT / 'source/Input_std.c').read_text()
    binding_start = input_source.index('  OpenMXMixingHistory *mixing_history=',
                                      input_source.index('void Input_std('))
    binding_end = input_source.index('  OpenMXElectronicViews electronic_storage;', binding_start)
    weight_line = next(line for line in input_source.splitlines()
                       if 'input_double("scf.Init.Mixing.Weight"' in line)
    generated = '\n'.join([function(helpers, 'sgn'), function(helpers, 'largest'),
                            function(mixing, 'Simple_Mixing_DM'),
                            function((ROOT / 'source/outputfile1.c').read_text(), 'outputfile1')])
    generated += '\nstatic void parse_weight(OpenMXCoreContext *core) {\n'
    generated += input_source[binding_start:binding_end] + weight_line + '\n}\n'
    with tempfile.TemporaryDirectory(prefix='openmx-mixing-history-') as directory:
        folder = Path(directory)
        (folder / 'mixing_history.inc').write_text(generated)
        executable = folder / 'check'
        command = compiler_command('c++') + [
            '-I' + str(folder), '-I' + str(ROOT / 'core/include'),
            str(ROOT / 'tests/mixing_history_harness.cpp'), str(library),
            '-Wl,-rpath,' + str(library.parent), '-o', str(executable)]
        subprocess.run(command, check=True)
        subprocess.run([str(executable), str(folder / 'scf.out')], env=os.environ.copy(), check=True,
                       timeout=120, preexec_fn=bounded_stack)


if __name__ == '__main__':
    main()

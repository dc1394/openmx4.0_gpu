#!/usr/bin/env python3
"""Check the shared neutral-atom pair/cutoff module with independent moments.

Both production pair wrappers are compiled as C11 and C++17. Radial
interpolation is prescribed; closed polynomial moments and coordinate finite
differences check energy/gradient, images, species mapping and distance guards.
The caller's cutoff partition/initial values are tested without MPI or GPUs.
"""
import argparse
from pathlib import Path
import subprocess
import tempfile
from native_state_test_support import bounded_stack, compiler_command
from run_cwf_band_helpers_cpu import function

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--negative-control', action='store_true')
    args = parser.parse_args()
    header = (ROOT/'source/openmx_neutral_atom.h').read_text()
    grid_header=(ROOT/'core/include/openmx_core_neutral_grid.h').read_text()
    native = '\n'.join(function((ROOT/'source'/source).read_text(), 'EH0_TwoCenter')
        .replace('EH0_TwoCenter', entry, 1) for source, entry in
        [('Total_Energy.c', 'EnergyNeutralPair'), ('Stress.c', 'StressNeutralPair')])
    with tempfile.TemporaryDirectory(prefix='openmx-neutral-pair-') as directory:
        folder = Path(directory)
        (folder/'neutral_pair_native.inc').write_text(native)

        def build(name, language, code=header, grid=grid_header, reference64=False):
            (folder/'openmx_neutral_atom.h').write_text(code)
            (folder/'openmx_core_neutral_grid.h').write_text(grid)
            executable = folder/name
            flags = ['-DOPENMX_TEST_REFERENCE_BINARY64=1'] if reference64 else []
            subprocess.run(compiler_command(language)+flags+['-I'+str(folder),
                '-I'+str(ROOT/'core/include'), '-x', language,
                str(ROOT/'tests/neutral_pair_harness.c'), '-lm', '-o', str(executable)], check=True)
            return executable

        for language in ['c', 'c++']:
            subprocess.run([str(build('neutral-pair-'+language, language))], cwd=folder,
                           check=True, preexec_fn=bounded_stack)
            subprocess.run([str(build('neutral-pair-binary64-'+language, language, reference64=True))],
                           cwd=folder, check=True, preexec_fn=bounded_stack)
        if args.negative_control:
            for name, original, replacement in [
                ('sign', 'sumr -= wt*rho0*dr_va0*z2/r;', 'sumr += wt*rho0*dr_va0*z2/r;'),
                ('component', 'sumz = sumr*z/r;', 'sumz = sumr*y/r;'),
                ('radius', '    if (r<1.0e-10) r = 1.0e-10;\n    va0', '    va0'),
                ('distance', 'r = separation;\n    if (r<1.0e-10) r = 1.0e-10;', 'r = separation;'),
                ('cutoff', 'z2 = z - rcut;', 'z2 = z - 0.5*rcut;')]:
                assert header.count(original) == 1
                failed = subprocess.run([str(build(name, 'c', header.replace(original, replacement)))],
                    cwd=folder, capture_output=True, text=True, preexec_fn=bounded_stack)
                assert failed.returncode == 2 and 'FAIL line' in failed.stderr, failed
            stride='(component*tables->extent[1]+species)*tables->extent[2]'
            assert grid_header.count(stride)==1
            failed=subprocess.run([str(build('flat-row-layout','c',grid=grid_header.replace(
                stride,'(species*5+component)*tables->extent[2]')))],cwd=folder,
                capture_output=True,text=True,preexec_fn=bounded_stack)
            assert failed.returncode==2 and 'FAIL line'in failed.stderr,failed
    print('PASS: shared neutral C11/C++17 kernels and '+str(5 if args.negative_control else 0)+
          ' arithmetic negative controls and a flat-layout negative control')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Check production CWF energy decomposition and invocation-owned flat scratch.

The actual C++ owner is linked. MPI, XC and accelerated grid calls are modeled;
analytic inputs independently check atom/orbital tables and spin totals. Both
native backend conditionals run without a device SDK or an MPI installation.
"""
import argparse
from pathlib import Path
import re
import subprocess
import tempfile
from native_state_test_support import bounded_stack, compiler_command
from run_cwf_band_helpers_cpu import function, MPI

ROOT = Path(__file__).resolve().parents[1]


def check_output(path, spin, base):
    atoms = 1 if base == 1 else 2
    counts = [1] if atoms == 1 else [1, base - 1]
    counts = [n * (2 if spin == 3 else 1) for n in counts]
    channels = 2 if spin == 1 else 1
    charges = [[.2 + .1*s + .02*i for i in range(sum(counts))] for s in range(channels)]
    bands = [[-.3 - .07*s + .04*i for i in range(sum(counts))] for s in range(channels)]
    double_count = []
    for s in range(2 if spin else 1):
        h = -.5*.25*sum((.2+.1*s+.03*b + .05+.01*b)*(.3-.04*b) for b in range(4))
        x = -.25*sum((.2+.1*s+.03*b)*(-.4+.05*s+.02*b) for b in range(4))
        double_count.append(h + x + [-.31, -.27][s])
    populations = [.9, .7] if atoms == 1 else [2.2, 1.8]
    coefficients = ([sum(double_count)/sum(populations)] if spin == 3 else
                    [double_count[s]/populations[s] for s in range(channels)])
    corrections = [[coefficients[s]*q for q in charges[s]] for s in range(channels)]
    start = 0
    for atom, count in enumerate(counts):
        local = sum(sum(row[start:start+count]) for row in charges)*(2 if spin == 0 else 1)
        for s in range(channels):
            for i in range(start, start+count):
                corrections[s][i] += [ .8, 1.4 ][atom]*charges[s][i]/local
        start += count

    def columns(b, d):
        if spin == 3:
            return [b[0], d[0], b[0]+d[0]]
        if spin == 0:
            b, d = b*2, d*2
        return [b[0], b[1], sum(b), b[0]-b[1], d[0], d[1], sum(d), d[0]-d[1],
                sum(b)+sum(d), b[0]+d[0]-b[1]-d[1]]

    lines = path.read_text().splitlines()
    atom_rows = [list(map(float, m[2].split())) for line in lines
                 if (m := re.match(r'^\s*(\d+)\s+[AB]\s+([-.\d\s]+)$', line))]
    orbital_rows = [list(map(float, m[2].split())) for line in lines
                    if (m := re.match(r'^\s*(\d+)\s+([-.\d\s]+)$', line))]
    assert len(atom_rows) == atoms and len(orbital_rows) == sum(counts)
    expected_atoms, expected_orbitals = [], []
    start = 0
    for count in counts:
        expected_atoms.append(columns([sum(row[start:start+count]) for row in bands],
                                      [sum(row[start:start+count]) for row in corrections]))
        for i in range(start, start+count):
            expected_orbitals.append(columns([row[i] for row in bands], [row[i] for row in corrections]))
        start += count
    for actual, expected in zip(atom_rows+orbital_rows, expected_atoms+expected_orbitals):
        assert len(actual) == len(expected) and max(abs(a-b) for a,b in zip(actual, expected)) < 5.1e-7, (actual, expected)
    total = re.search(r'Total Sum\s*=\s*([-\d.]+)', '\n'.join(lines))
    expected = sum(row[-1 if spin == 3 else -2] for row in expected_atoms)
    assert total and abs(float(total[1])-expected) < 5.1e-9, ('footer', total, expected)
    if spin != 3:
        totals = re.search(r'up\s*=\s*([-\d.]+)\s+down\s*=\s*([-\d.]+)', '\n'.join(lines))
        expected_up = sum(row[0]+row[4] for row in expected_atoms)
        expected_down = sum(row[1]+row[5] for row in expected_atoms)
        assert totals and abs(float(totals[1])-expected_up)<5.1e-9 and abs(float(totals[2])-expected_down)<5.1e-9, ('spin footer', totals)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core-library', type=Path, required=True)
    parser.add_argument('--negative-control', action='store_true')
    args = parser.parse_args()
    library = args.core_library.resolve(strict=True)
    native = function((ROOT/'source/Total_Energy.c').read_text(), 'Energy_Decomposition_CWF')
    with tempfile.TemporaryDirectory(prefix='openmx-energy-workspace-') as directory:
        folder = Path(directory)
        (folder/'mpi.h').write_text(MPI)
        (folder/'omp.h').write_text('static inline int omp_get_max_threads(void){return 1;}\n'
                                  'static inline int omp_get_thread_num(void){return 0;}\n')

        def build(name, backend=0, code=native):
            (folder/'energy_native.inc').write_text(code)
            executable = folder/name
            subprocess.run(compiler_command('c') + ['-DOPENMX_GPU_HIP='+str(backend),
                '-I'+str(folder), '-I'+str(ROOT/'source'), '-I'+str(ROOT/'core/include'),
                '-x', 'c', str(ROOT/'tests/energy_workspace_harness.c'), '-x', 'none', str(library),
                '-Wl,-rpath,'+str(library.parent), '-lm', '-o', str(executable)], check=True)
            return executable

        for backend in [0, 1]:
            executable = build('energy-'+str(backend), backend)
            subprocess.run([str(executable), 'layout'], cwd=folder, check=True, preexec_fn=bounded_stack)
            for mode in ['bad-count', 'bad-eh0-count', 'bad-cwf']:
                failed = subprocess.run([str(executable), mode], cwd=folder, capture_output=True,
                                        text=True, preexec_fn=bounded_stack)
                assert failed.returncode == 77 and 'OpenMX solver workspace:' in failed.stderr
            for spin in [0, 1, 3]:
                for base in [1, 2, 4]:
                    for dispatch in [1, 7]:
                        subprocess.run([str(executable), str(spin), str(base), str(dispatch)],
                                       cwd=folder, check=True, preexec_fn=bounded_stack)
                        check_output(folder/'energy.CWF_Energy', spin, base)
        if args.negative_control:
            old = re.sub(r'for \(spin=0; spin<2; spin\+\+\)\{(\s*TsumE\[spin\])',
                         r'for (spin=0; spin<=SpinP_switch; spin++){\1', native)
            assert old != native
            executable = build('missing-nonmagnetic-down-total', code=old)
            subprocess.run([str(executable), '0', '1', '1'], cwd=folder, check=True, preexec_fn=bounded_stack)
            try:
                check_output(folder/'energy.CWF_Energy', 0, 1)
            except AssertionError as error:
                assert error.args[0][0] == 'footer', error
            else:
                raise AssertionError('old nonmagnetic footer was accepted')
    print('PASS: 36 CWF energy cases, flat scratch reuse/isolation, small-CWF bounds and footer negative control')


if __name__ == '__main__':
    main()

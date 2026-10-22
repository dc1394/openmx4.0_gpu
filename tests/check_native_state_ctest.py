#!/usr/bin/env python3
"""Require all native solver lifetime tests to use this build's real core.

This guards the SDK-free CI contract: the application may be disabled, but all
seven solver states, Hamiltonian mixing and the energy bridge must be tested.
The MPI CI configuration additionally requires both real-communicator tests.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess


def check_inventory(inventory, build, require_mpi=False):
    solvers = ['cluster-col', 'cluster-noncol', 'band-col', 'band-noncol', 'dc', 'dclno', 'krylov', 'mixing-h']
    expected = {name + '-calculation-state-lifecycle': 'native-state' for name in solvers}
    expected['solver-noncollinear-energy-output'] = 'native-state'
    if require_mpi:
        expected.update({name + '-calculation-state-mpi': 'native-state-mpi'
                         for name in ['dclno', 'krylov']})
    tests = inventory.get('tests', [])
    counts = Counter(test['name'] for test in tests)
    by_name = {test['name']: test for test in tests}
    for name, label in expected.items():
        if counts[name] != 1:
            raise ValueError(f'{name}: expected one registered test, found {counts[name]}')
        test = by_name[name]
        properties = {p['name']: p['value'] for p in test.get('properties', [])}
        if label not in properties.get('LABELS', []):
            raise ValueError(f'{name}: missing CI label {label}')
        command = test.get('command', [])
        if '--core-library' not in command:
            raise ValueError(f'{name}: no explicit core library')
        index = command.index('--core-library') + 1
        if index == len(command):
            raise ValueError(f'{name}: missing core library argument')
        library = Path(command[index]).resolve(strict=True)
        if build not in library.parents or not library.name.startswith('libopenmx_core.'):
            raise ValueError(f'{name}: library does not belong to this build: {library}')
        # Resolving through the platform loader must never replace execution of
        # the real state/entry fixture with a stub or a diagnostic-only command.
        script = Path(command[1]).resolve(strict=True)
        expected_scripts = {
            'cluster-col': 'run_cluster_col_state_cpu.py',
            'cluster-noncol': 'run_cluster_noncol_state_cpu.py',
            'band-col': 'run_band_state_cpu.py', 'band-noncol': 'run_band_state_cpu.py',
            'dc': 'run_local_solver_state_cpu.py', 'dclno': 'run_local_solver_state_cpu.py',
            'krylov': 'run_local_solver_state_cpu.py', 'mixing-h': 'run_mixing_h_state_cpu.py'}
        prefix = name.split('-calculation-state-')[0]
        expected_script = (expected_scripts[prefix] if prefix in expected_scripts else
                           'run_solver_energy_output_cpu.py')
        if script != Path(__file__).resolve().parent / expected_script:
            raise ValueError(f'{name}: incorrect production fixture: {script}')
        kinds = {'band-col': 'col', 'band-noncol': 'noncol',
                 'dc': 'dc', 'dclno': 'dclno', 'krylov': 'krylov'}
        if prefix in kinds:
            if '--kind' not in command or command[command.index('--kind') + 1] != kinds[prefix]:
                raise ValueError(f'{name}: incorrect solver variant')
        if name == 'solver-noncollinear-energy-output' and '--negative-control' not in command:
            raise ValueError(f'{name}: former-entry regression control is disabled')
        if label == 'native-state-mpi' and not all(flag in command for flag in ['--mpi-cxx', '--mpiexec']):
            raise ValueError(f'{name}: real MPI execution is missing')
    return len(expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--ctest', default='ctest')
    parser.add_argument('--require-mpi', action='store_true')
    args = parser.parse_args()
    build = args.build.resolve(strict=True)
    inventory = json.loads(subprocess.check_output(
        [args.ctest, '--test-dir', str(build), '--show-only=json-v1'], text=True))
    count = check_inventory(inventory, build, args.require_mpi)
    print(f'PASS: {count} native state tests registered once with this build\'s real core')


if __name__ == '__main__':
    main()

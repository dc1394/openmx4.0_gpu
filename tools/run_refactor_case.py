#!/usr/bin/env python3
"""Measure one unchanged OpenMX case and retain its numerical results and timings."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

from compare_gpu_suites import read_output
from run_gpu_suites import stop_job


def digest(path):
    with path.open('rb') as stream:
        checksum = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(block)
        return checksum.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mpiexec', type=Path, required=True)
    parser.add_argument('--mpi', choices=('openmpi', 'intel'), default='openmpi')
    parser.add_argument('--mode', choices=('cpu', 'gpu'), required=True)
    parser.add_argument('--ranks', type=int, default=18)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--timeout', type=int, default=1200)
    parser.add_argument('--core-library', type=Path)
    parser.add_argument('--library-path')
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--env', action='append', default=[], metavar='NAME=VALUE',
                        help='additional OPENMX_/GEMMUL8_ controls, recorded in result.json')
    args = parser.parse_args()
    if args.ranks < 1 or args.threads < 1 or args.timeout < 1:
        parser.error('ranks, threads and timeout must be positive')
    extra_controls = {}
    for item in args.env:
        name, separator, value = item.partition('=')
        if not separator or not name.startswith(('OPENMX_', 'GEMMUL8_')) or name == 'OPENMX_GPU':
            parser.error('--env requires an OPENMX_/GEMMUL8_ control other than OPENMX_GPU')
        extra_controls[name] = value
    args.binary = args.binary.resolve(strict=True)
    args.input = args.input.resolve(strict=True)
    args.mpiexec = args.mpiexec.resolve(strict=True)
    args.output = args.output.resolve()
    source = args.input.read_text()
    data_path = re.search(r'^\s*DATA.PATH\s+(\S+)', source, re.M | re.I)[1]
    data_path = (args.input.parent / data_path).resolve(strict=True)
    source = re.sub(r'^(\s*DATA.PATH\s+)\S+', lambda match: match[1] + str(data_path),
                    source, flags=re.M | re.I)
    source = re.sub(r'^(\s*System.CurrrentDirectory\s+)\S+', r'\g<1>./',
                    source, flags=re.M | re.I)
    if args.mode == 'cpu':
        source = re.sub(r'^(\s*scf.eigen.lib\s+)\S+', r'\g<1>elpa2',
                        source, flags=re.M | re.I)
    args.output.mkdir(parents=True, exist_ok=False)
    input_path = args.output / 'input.dat'
    input_path.write_text(source)
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(('OPENMX_', 'GEMMUL8_'))
                   and key not in ('LD_LIBRARY_PATH', 'LD_PRELOAD')}
    controls = dict(OPENMX_GPU='1' if args.mode == 'gpu' else '0',
                    OMP_NUM_THREADS=str(args.threads), OPENBLAS_NUM_THREADS=str(args.threads),
                    MKL_NUM_THREADS=str(args.threads), OMP_DYNAMIC='FALSE',
                    OPENMX_CLUSTER_PROFILE='1', OPENMX_BAND_PROFILE='1',
                    OPENMX_FORCE_PROFILE='1')
    if args.mpi == 'intel':
        controls.update(I_MPI_PIN='1', I_MPI_PIN_DOMAIN='core')
    if args.core_library:
        args.core_library = args.core_library.resolve(strict=True)
        controls['LD_PRELOAD'] = str(args.core_library)
    if args.library_path:
        controls['LD_LIBRARY_PATH'] = args.library_path
    controls.update(extra_controls)
    environment.update(controls)
    command = [str(args.mpiexec)]
    if args.mpi == 'openmpi':
        command += ['--bind-to', 'core']
    command += ['-np', str(args.ranks), str(args.binary), 'input.dat', '-nt', str(args.threads)]
    result = dict(status='running', command=command, controls=controls,
                  original_input_sha256=digest(args.input), input_sha256=digest(input_path),
                  binary_sha256=digest(args.binary), mode=args.mode,
                  ranks=args.ranks, threads=args.threads)
    if args.core_library:
        result['core_sha256'] = digest(args.core_library)
    result_path = args.output / 'result.json'
    result_path.write_text(json.dumps(result, indent=2) + '\n')
    begin = time.monotonic()
    try:
        with (args.output / 'run.log').open('w') as log:
            process = subprocess.Popen(command, cwd=args.output, env=environment, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            try:
                returncode = process.wait(timeout=args.timeout)
            except BaseException:
                stop_job(process)
                raise
            if returncode:
                raise subprocess.CalledProcessError(returncode, command)
        result['wall_seconds'] = time.monotonic() - begin
        result['returncode'] = process.returncode
        log = (args.output / 'run.log').read_text()
        if 'The calculation was normally finished.' not in log:
            raise ValueError('OpenMX did not finish normally')
        system = re.search(r'^\s*System.Name\s+(\S+)', source, re.M | re.I)[1]
        energy, atoms, forces, grids = read_output(args.output / (system + '.out'))
        phases = {}
        for match in re.finditer(r'^\s*(.+?)\s+=\s+\d+\s+([\d.]+)\s+\d+\s+([\d.]+)\s*$', log, re.M):
            phases[match[1]] = dict(min_seconds=float(match[2]), max_seconds=float(match[3]))
        result.update(status='passed', energy_hartree=energy, atoms=[list(atom) for atom in atoms],
                      forces_hartree_per_bohr=forces, grids=list(grids), phases=phases,
                      scf_iterations=len(re.findall(r'MD=\s*1\s+SCF=', log)),
                      profiles=[line for line in log.splitlines() if 'PROF' in line])
        if args.reference:
            baseline = json.loads(args.reference.read_text())
            for key in ('original_input_sha256', 'ranks', 'threads', 'scf_iterations', 'grids'):
                if baseline.get(key) != result[key]:
                    raise ValueError('reference differs in ' + key)
            if baseline.get('status') != 'passed':
                raise ValueError('reference case did not pass')
            previous_atoms = baseline['atoms']
            if len(previous_atoms) != len(atoms) or any(old[:2] != list(new[:2])
                                                       for old, new in zip(previous_atoms, atoms)):
                raise ValueError('reference differs in atom identities')
            # OpenMX prints coordinates to five decimal places. Intel and
            # NVHPC can round the last digit differently for the same input.
            position_difference = max(abs(old-new) for old_atom, new_atom in zip(previous_atoms, atoms)
                                      for old, new in zip(old_atom[2:], new_atom[2:]))
            if position_difference > 1.000001e-5:
                raise ValueError('reference differs in atom positions')
            energy_difference = abs(energy-baseline['energy_hartree'])
            previous_forces = baseline['forces_hartree_per_bohr']
            if len(previous_forces) != len(forces):
                raise ValueError('reference differs in force count')
            force_difference = max(abs(old-new) for old, new in zip(previous_forces, forces))
            result['comparison'] = dict(reference=str(args.reference.resolve()),
                                        max_printed_coordinate_difference=position_difference,
                                        energy_difference_hartree=energy_difference,
                                        max_force_difference_hartree_per_bohr=force_difference,
                                        wall_speedup=baseline['wall_seconds']/result['wall_seconds'])
            if energy_difference > 1e-7 or force_difference > 1e-7:
                raise ValueError('energy or force differs from reference by more than 1e-7')
    except Exception as error:
        result.update(status='failed', wall_seconds=time.monotonic() - begin, error=str(error))
    result_path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ('status', 'mode', 'wall_seconds')}, indent=2))
    return int(result['status'] != 'passed')


if __name__ == '__main__':
    raise SystemExit(main())

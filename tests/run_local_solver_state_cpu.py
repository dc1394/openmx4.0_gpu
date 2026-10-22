#!/usr/bin/env python3
"""Run actual DC/DC-LNO/Krylov state construction and cleanup with the real core.

SDK operations are modeled. The optional MPI compiler/launcher arguments use
real four-rank MPI; otherwise communicator descriptors are modeled too.
Production metadata, initialization, allocation adapters and transitive cleanup
are compiled as-is. This checks ownership, not physical GPU concurrency.
"""
import argparse
import os
from pathlib import Path
import re
import subprocess
import tempfile
from native_state_test_support import bounded_stack, compiler_command, allocation_fault_source, allocation_fault_environment
from native_resource_test_bridge import calculation_state_declaration
from run_cluster_col_state_cpu import function

ROOT = Path(__file__).resolve().parents[1]
MODULES = {'dc': ('Divide_Conquer', 'DC'), 'dclno': ('Divide_Conquer_LNO', 'DCLNO'),
           'krylov': ('Krylov', 'Krylov')}


def cleanup_functions(source, module, prefix):
    candidates = set(re.findall(r'\b((?:' + prefix + '|' + module + r')\w*)\s*\(', source))
    names = {module + '_Bind_Resources', prefix + '_Reset', module + '_Release_GPU_Caches'}
    bodies = {}
    while True:
        more = set(names)
        for name in names:
            if name not in bodies:
                bodies[name] = function(source, name)
            clean = re.sub(r'/\*.*?\*/|//[^\n]*', '', bodies[name], flags=re.S)
            for candidate in candidates - more:
                if not re.search(r'\b' + candidate + r'\s*\(', clean):
                    continue
                try:
                    bodies[candidate] = function(source, candidate)
                except ValueError:
                    continue
                more.add(candidate)
        if names == more:
            return [bodies[name] for name in sorted(names)]
        names = more


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core-library', type=Path, required=True)
    parser.add_argument('--kind', choices=MODULES, required=True)
    parser.add_argument('--mpi-cxx', help='Compile the same cleanup with a real MPI C++ wrapper')
    parser.add_argument('--mpiexec', help='MPI launcher for a four-rank communicator lifetime check')
    args = parser.parse_args()
    if bool(args.mpi_cxx) != bool(args.mpiexec):
        parser.error('--mpi-cxx and --mpiexec must be supplied together')
    library = args.core_library.resolve(strict=True)
    module, prefix = MODULES[args.kind]
    with tempfile.TemporaryDirectory(prefix='openmx-local-solver-state-') as directory:
        folder = Path(directory)
        for backend in ['cuda', 'hip']:
            path = ROOT / ('source/' + ('' if backend == 'cuda' else 'backends/hip/src/') + module + '.c')
            text = path.read_text()
            bodies = cleanup_functions(text, module, prefix)
            prototypes = '\n'.join(body[:body.index('{')].rstrip() + ';' for body in bodies)
            generated = calculation_state_declaration(text, prefix + 'State') + '\n' + prototypes + '\n' + '\n'.join(bodies)
            generated += '\nusing LocalState = ' + prefix + 'State;\n'
            generated += '#define local_bind ' + module + '_Bind_Resources\n'
            generated += '#define local_release ' + module + '_Release_GPU_Caches\n'
            (folder / 'local_solver_state.inc').write_text(generated)
            executable = folder / backend
            command = compiler_command('c++', args.mpi_cxx) + [
                '-Wno-unknown-pragmas',
                '-DOPENMX_GPU_HIP=' + str(int(backend == 'hip')), '-DLOCAL_' + args.kind.upper() + '=1',
                '-DLOCAL_MPI_REAL=' + str(int(bool(args.mpi_cxx))),
                '-I' + str(folder), '-I' + str(ROOT / 'source'), '-I' + str(ROOT / 'core/include'),
                '-I' + str(ROOT / 'core/tests/fixtures/native_resources'),
                '-I' + str(ROOT / 'core/tests'),
                str(ROOT / 'tests/local_solver_state_harness.cpp'),
                allocation_fault_source(folder), str(library),
                '-Wl,-rpath,' + str(library.parent), '-o', str(executable)]
            subprocess.run(command, check=True)
            environment = {k: v for k, v in os.environ.items() if not k.startswith(('OPENMX_', 'GEMMUL8_'))}
            environment.update(allocation_fault_environment(folder))
            if args.mpi_cxx:
                environment['ASAN_OPTIONS'] = 'detect_leaks=0'  # MPI retains process runtime allocations.
            print('Local solver state:', args.kind, backend, 'production cleanup functions:', len(bodies), flush=True)
            run = [str(executable)] if not args.mpiexec else [args.mpiexec, '-np', '4', str(executable)]
            subprocess.run(run, check=True, env=environment, timeout=120, preexec_fn=bounded_stack)


if __name__ == '__main__':
    main()

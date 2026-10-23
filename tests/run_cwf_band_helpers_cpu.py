#!/usr/bin/env python3
"""Exercise production CWF band helpers with the real C++ buffer owner.

MPI descriptors/collectives and dense eigensolvers are modeled. Independent
analytic Hermitian Hamiltonians check complex Fourier phases, both spin
channels, uneven atom ownership (including an empty rank), ragged band paths,
single-point paths and solver failure guards without a GPU SDK or MPI runtime.
"""
import argparse
import cmath
import math
from pathlib import Path
import re
import subprocess
import tempfile
from native_state_test_support import bounded_stack, compiler_command

ROOT = Path(__file__).resolve().parents[1]
MPI = r'''
#ifndef FIXTURE_MPI_H
#define FIXTURE_MPI_H
typedef int MPI_Comm;
typedef int MPI_Status;
typedef int MPI_Request;
#define MPI_INT 1
#define MPI_DOUBLE 2
#define MPI_SUM 3
#define MPI_IN_PLACE ((void*)1)
int MPI_Abort(MPI_Comm,int);
int MPI_Comm_size(MPI_Comm,int*);
int MPI_Comm_rank(MPI_Comm,int*);
int MPI_Barrier(MPI_Comm);
int MPI_Comm_split(MPI_Comm,int,int,MPI_Comm*);
int MPI_Comm_free(MPI_Comm*);
int MPI_Comm_c2f(MPI_Comm);
int MPI_Bcast(void*,int,int,int,MPI_Comm);
int MPI_Allreduce(const void*,void*,int,int,int,MPI_Comm);
#endif
'''


def function(source, name):
    tokens = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|/\*.*?\*/|//[^\n]*|[{}]', re.S)
    for match in re.finditer(r'^(?:static )?void ' + name + r'\s*\(', source, re.M):
        opening = source.index('{', match.end())
        if ';' in source[match.end():opening]:
            continue
        depth = 0
        for token in tokens.finditer(source, opening):
            if token[0] == '{':
                depth += 1
            elif token[0] == '}':
                depth -= 1
                if depth == 0:
                    return source[match.start():token.end()]
    raise ValueError('missing production definition: ' + name)


def output(path, spins, complex_values):
    lines = path.read_text().splitlines()
    assert [int(x) for x in lines[0].split()[:2]] == [2, spins - 1]
    assert int(lines[2]) == 3
    assert [int(line.split()[0]) for line in lines[3:6]] == [3, 1, 4]
    expected = [0, .25, .5, .17, -.25, -1 / 12, 1 / 12, .25]
    cursor = 6
    for coordinate in expected:
        for spin in range(spins):
            point = lines[cursor].split()
            assert int(point[0]) == 2 and abs(float(point[1]) - coordinate) < 2e-14
            assert all(abs(float(x)) < 1e-14 for x in point[2:])
            eigen = [float(x) for x in lines[cursor + 1].split()]
            assert len(eigen) == 2 and eigen[0] < eigen[1]
            a = [2, .25 - .2j, .25 + .2j, 3] if complex_values else [2, .25, .25, 3]
            t = [.1 + .03j, -.04 + .02j, .07 - .05j, .12 - .04j] if complex_values else [.1, -.04, .07, .12]
            phase = cmath.exp(2j * math.pi * coordinate)
            matrix = [complex(a[2*j+i]) + (5*spin if i == j else 0) +
                      (spin+1) * (complex(t[2*j+i])*phase + complex(t[2*i+j]).conjugate()*phase.conjugate())
                      for j in range(2) for i in range(2)]
            center = (matrix[0].real + matrix[3].real)/2
            radius = math.sqrt(((matrix[0].real-matrix[3].real)/2)**2 + abs(matrix[1])**2)
            assert max(abs(x-y) for x,y in zip(eigen,[center-radius,center+radius])) < 3e-13
            cursor += 2
    assert cursor == len(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core-library', type=Path, required=True)
    parser.add_argument('--negative-control', action='store_true')
    args = parser.parse_args()
    library = args.core_library.resolve(strict=True)
    col = (ROOT / 'source/Band_DFT_Col_CWF.c').read_text()
    nc = (ROOT / 'source/Band_DFT_NonCol_CWF.c').read_text()
    native = '\n'.join([function(col, 'Band_Dispersion_Col_CWF'),
                        function(nc, 'Band_Dispersion_NonCol_CWF'), function(nc, 'Construct_Band_Ms')])
    header = (ROOT / 'source/openmx_cwf_workspace.h').read_text()
    fixture = (ROOT / 'tests/cwf_band_helpers_harness.c').read_text()
    with tempfile.TemporaryDirectory(prefix='openmx-cwf-band-helpers-') as directory:
        folder = Path(directory)
        (folder / 'mpi.h').write_text(MPI)
        (folder / 'omp.h').write_text('static inline int omp_get_max_threads(void){return 1;}\n'
                                     'static inline int omp_get_thread_num(void){return 0;}\n')
        (folder / 'fixture.c').write_text(fixture)

        def build(name, code=native, included=header, backend=0):
            (folder / 'cwf_native.inc').write_text(code)
            (folder / 'openmx_cwf_workspace.h').write_text(included)
            executable = folder / name
            command = compiler_command('c') + ['-DOPENMX_GPU_HIP=' + str(backend),
                '-I' + str(folder), '-I' + str(ROOT / 'source'), '-I' + str(ROOT / 'core/include'),
                '-x', 'c', str(folder / 'fixture.c'), '-x', 'none', str(library),
                '-Wl,-rpath,' + str(library.parent), '-lm', '-o', str(executable)]
            subprocess.run(command, check=True)
            return executable

        for backend in [0, 1]:
            executable = build('helpers-' + str(backend), backend=backend)
            for mode in ['layout', 'matrix']:
                subprocess.run([str(executable), mode, '1'], cwd=folder, check=True, preexec_fn=bounded_stack)
            for mode, spins in [('col', 2), ('nc', 1)]:
                for solver in [1, 2, 7]:
                    subprocess.run([str(executable), mode, str(solver)], cwd=folder, check=True, preexec_fn=bounded_stack)
                    output(folder / (mode + '-' + str(solver) + '.CWF.Band'), spins, mode == 'nc')
            for mode in ['info', 'convergence', 'point', 'path', 'zero', 'nan', 'infinite', 'range', 'fraction']:
                failure = subprocess.run([str(executable), 'bad-' + mode, '1'], cwd=folder,
                    capture_output=True, text=True, preexec_fn=bounded_stack)
                assert failure.returncode == 77 and 'OpenMX solver workspace:' in failure.stderr, (mode, failure)
        if args.negative_control:
            old = 'return count==1 ? start : start+'
            assert header.count(old) == 1
            executable = build('missing-single-point-guard', included=header.replace(old, 'return start+'))
            failed = subprocess.run([str(executable), 'nc', '1'], cwd=folder, capture_output=True,
                                    text=True, preexec_fn=bounded_stack)
            assert failed.returncode != 0 and 'isfinite' in failed.stderr, failed
            padded = native.replace('for (ik=1; ik<=nkpath; ik++)', 'for (ik=0; ik<=nkpath; ik++)')
            padded = padded.replace('for (i_perk=1; i_perk<=n_perk[ik]; i_perk++)',
                                    'for (i_perk=0; i_perk<=n_perk[ik]; i_perk++)')
            assert padded != native
            executable = build('padded-kpoint-control', code=padded)
            failed = subprocess.run([str(executable), 'col', '1'], cwd=folder, capture_output=True,
                                    text=True, preexec_fn=bounded_stack)
            assert failed.returncode == 77 and 'invalid CWF band path point' in failed.stderr, failed
    print('PASS: ragged paths, both spin channels, complex Fourier/packing algebra, failure guards and negative controls')


if __name__ == '__main__':
    main()

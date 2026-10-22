#!/usr/bin/env python3
"""Exercise actual band state ownership with the real core and modeled SDK/MPI.

Compile the complete production POD, constructor, callback adapters, refinement
controls and the transitive deep cleanup functions without rewriting them.
OpenACC/OpenMP transfers and other modules' workspace pools are not executed.
"""
import argparse
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
from run_cluster_col_state_cpu import function, pod, bounded_stack
from native_resource_test_bridge import calculation_state_declaration

ROOT = Path(__file__).resolve().parents[1]


def cleanup_functions(source, prefix, module):
    candidates = set(re.findall(r'\b((?:' + prefix + '|' + module + r')_\w+)\s*\(', source))
    bodies = {}
    names = {module + '_Bind_Resources', prefix + '_Reset', prefix + '_ReleaseCaches',
             prefix + '_RefineDeviceMalloc', prefix + '_RefineHostAllocate',
             prefix + '_RefineConfigure', prefix + '_RefineIterations',
             prefix + '_RefineBeginStep', prefix + '_RefineEndStep',
             *(module + '_Refine' + suffix for suffix in
               ['LastRefined', 'ForceFP64', 'AfterMixing', 'EndCycle'])}
    while True:
        more = set(names)
        for name in names:
            if name not in bodies:
                bodies[name] = function(source, name)
            text = re.sub(r'/\*.*?\*/|//[^\n]*', '', bodies[name], flags=re.S)
            for candidate in candidates:
                if candidate in more or not re.search(r'\b' + candidate + r'\s*\(', text):
                    continue
                try:
                    bodies[candidate] = function(source, candidate)
                except ValueError:
                    continue  # An externally declared SDK kernel.
                more.add(candidate)
        if more == names:
            break
        names = more
    return [bodies[name] for name in sorted(names)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core-library', type=Path, required=True)
    parser.add_argument('--kind', choices=['col', 'noncol'], required=True)
    args = parser.parse_args()
    library = args.core_library.resolve(strict=True)
    kind = 'Col' if args.kind == 'col' else 'NonCol'
    prefix, module = 'Band' + kind, 'Band_DFT_' + kind
    with tempfile.TemporaryDirectory(prefix='openmx-band-state-') as directory:
        folder = Path(directory)
        for backend in ['cuda', 'hip']:
            base = ROOT / ('source' if backend == 'cuda' else 'source/backends/hip')
            source = base / (module + '.c' if backend == 'cuda' else 'src/' + module + '.c')
            header = base / ('eigen_refine_gpu.h' if backend == 'cuda' else 'include/eigen_refine_gpu.h')
            refine = base / ('eigen_refine_gpu.c' if backend == 'cuda' else 'src/eigen_refine_gpu.cu')
            text, sdk_header = source.read_text(), header.read_text()
            sdk_types = '\n'.join(pod(sdk_header, name) for name in
                ['EigenRefineDevice', 'EigenRefineState', 'EigenRefineProblem', 'EigenRefineReport'])
            bodies = cleanup_functions(text, prefix, module)
            prototypes = '\n'.join(body[:body.index('{')].rstrip() + ';' for body in bodies)
            allocation = '\n'.join(function(sdk_header, name) for name in
                ['openmx_eigen_refine_malloc', 'openmx_eigen_refine_host_malloc'])
            release = '\n'.join(function(refine.read_text(), name) for name in
                ['refine_blocks_release', 'openmx_eigen_refine_state_release'])
            generated = (sdk_types + '\n' + calculation_state_declaration(text, prefix + 'State') +
                '\n' + prototypes + '\n' + allocation + '\n' + release +
                '\nstatic void openmx_eigen_refine_device_release(EigenRefineDevice*) { refine_device_drains++; }\n' +
                '\n'.join(bodies))
            # Only names are aliased for one common test driver; production
            # definitions, fields and control-flow bodies stay unchanged.
            generated += '\nusing BandState = ' + prefix + 'State;\n'
            for target, name in [('band_bind', module + '_Bind_Resources'),
                ('band_release', prefix + '_ReleaseCaches'),
                ('band_configure', prefix + '_RefineConfigure'),
                ('band_iterations', prefix + '_RefineIterations'),
                ('band_begin', prefix + '_RefineBeginStep'), ('band_end', prefix + '_RefineEndStep'),
                ('band_last', module + '_RefineLastRefined'),
                ('band_force_fp64', module + '_RefineForceFP64'),
                ('band_mixing', module + '_RefineAfterMixing'),
                ('band_end_cycle', module + '_RefineEndCycle'),
                ('band_malloc', prefix + '_RefineDeviceMalloc'),
                ('band_host_malloc', prefix + '_RefineHostAllocate')]:
                generated += '#define ' + target + ' ' + name + '\n'
            (folder / 'band_state.inc').write_text(generated)
            executable = folder / backend
            command = shlex.split(os.environ.get('CXX', 'c++')) + [
                '-std=c++17', '-O1', '-g', '-fsanitize=address,undefined', '-fno-sanitize-recover=all',
                '-fno-pie', '-no-pie', '-Wno-unknown-pragmas',
                '-DOPENMX_GPU_HIP=' + str(int(backend == 'hip')),
                '-DBAND_NONCOL=' + str(int(kind == 'NonCol')),
                '-I' + str(folder), '-I' + str(ROOT / 'source'), '-I' + str(ROOT / 'core/include'),
                '-I' + str(ROOT / 'core/tests/fixtures/native_resources'), '-I' + str(ROOT / 'core/tests'),
                str(ROOT / 'tests/band_state_harness.cpp'), str(ROOT / 'core/tests/allocation_faults.cpp'),
                str(ROOT / 'source/openmx_env.c'), str(ROOT / 'source/openmx_precision.c'), str(library),
                '-Wl,-rpath,' + str(library.parent), '-o', str(executable)]
            subprocess.run(command, check=True)
            environment = {k: v for k, v in os.environ.items() if not k.startswith(('OPENMX_', 'GEMMUL8_'))}
            print('Band state:', args.kind, backend, 'production cleanup/control functions:', len(bodies), flush=True)
            subprocess.run([str(executable)], check=True, env=environment, timeout=120, preexec_fn=bounded_stack)


if __name__ == '__main__':
    main()

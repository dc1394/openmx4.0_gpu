#!/usr/bin/env python3
"""Project the legacy device-only C entry points to an unavailable CPU adapter.

Signatures come from the same canonical .cu files as the GPU build. No numerical
operation reports success and no output buffer is published. The application
uses its existing CPU algorithms; no GPU source or vendor SDK is compiled.
"""
import argparse
import json
from pathlib import Path
import re


def prepare(paths):
    result = ['// Generated from the canonical native GPU C interfaces.\n',
              '#include <cstddef>\n#include <cstdint>\n#include <cstring>\n',
              '#include "openmx_gpu_runtime.h"\n#include "openmx_gpu_blas.h"\n',
              '#include "openmx_gpu_solver.h"\n#include "openmx_mgpu_runtime.h"\n',
              '#include "openmx_mgpu_gemm.h"\n#include "openmx_gpu_transform.h"\n',
              '#include "openmx_gpu_dm.h"\n']
    seen = set()
    for path in paths:
        source = path.read_text()
        # Export declarations are flat C interfaces; argument lists may contain
        # callback types, so balance parentheses instead of stopping at the first.
        for match in re.finditer(r'extern\s+"C"\s+(?!__attribute__)([\w :]+?)(\s*\*\s*|\s+)(\w+)\s*\(', source):
            kind = match.group(1).strip() + ('*' if '*' in match.group(2) else '')
            name = match.group(3)
            start = match.end()
            end, level = start, 1
            while level:
                if source[end] == '(':
                    level += 1
                elif source[end] == ')':
                    level -= 1
                end += 1
            # A backend conditional may select the implementation body after
            # the common signature. Only the signature is needed by the host.
            body_start = re.match(r'(?:\s|\#[^\n]*\n|//[^\n]*\n|/\*[\s\S]*?\*/)*\{', source[end:])
            if not body_start or name in seen:
                continue
            seen.add(name)
            signature = source[match.start():end]
            if kind == 'void':
                body = ''
            elif '*' in kind:
                body = 'return "CPU-only build";' if 'char' in kind else 'return nullptr;'
            elif kind in ('cublasStatus_t', 'OpenMXBlasStatus'):
                body = 'return CUBLAS_STATUS_NOT_SUPPORTED;'
            elif kind in ('cusolverStatus_t', 'OpenMXSolverStatus'):
                body = 'return CUSOLVER_STATUS_NOT_SUPPORTED;'
            elif kind in ('cudaError_t', 'OpenMXGpuError'):
                body = 'return cudaErrorNoDevice;'
            elif kind == 'openmx_mgpu_scope_state':
                body = 'return {};'
            elif name.startswith('Set_Hamiltonian_Cuda_'):
                # These CUDA launchers use zero for success; one declines.
                body = 'return 1;'
            elif kind in ('int', 'long long', 'size_t', 'std::size_t'):
                body = 'return 0;'
            else:
                raise ValueError(f'unsupported accelerator return type: {kind} {name}')
            result.append(signature + ' { ' + body + ' }\n')
    if len(seen) < 40:
        raise ValueError(f'incomplete CPU accelerator boundary: {len(seen)} functions')
    return ''.join(result)


def application_sources(root):
    sources, kernels, objects = [], [], []
    pending = ''
    for line in (root / 'Makefile').read_text().splitlines():
        if pending:
            pending += ' ' + line
        elif re.match(r'^OBJS\s*(?:\+?=)', line):
            pending = line
        else:
            continue
        if line.rstrip().endswith('\\'):
            continue
        objects.extend(re.findall(r'\b(\w+)\.o\b', pending))
        pending = ''
    for name in dict.fromkeys(objects):
        if name in ('set_cuda_default_device_from_local_rank',
                    'set_openacc_device_from_local_rank', 'elpa_cosma_memory'):
            continue
        candidates = [root / (name + suffix) for suffix in ('.c', '.cpp', '.cu')]
        source = next((p for p in candidates if p.is_file()), None)
        if source is None:
            raise ValueError(f'missing application translation unit: {name}')
        (kernels if source.suffix == '.cu' else sources).append(str(source))
    if len(sources) < 200:
        raise ValueError(f'incomplete CPU application source list: {len(sources)} units')
    return dict(sources=sources, kernels=kernels)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    manifest = application_sources(args.source_root)
    args.manifest.write_text(json.dumps(manifest, indent=2) + '\n')
    output = prepare([Path(p) for p in manifest['kernels']])
    if not args.output.exists() or args.output.read_text() != output:
        args.output.write_text(output)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Freeze the six requested Sirius workloads; no scientific input changes."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

B = Path(__file__).resolve().parent
ROOT = B.parents[1]
SUITES = dict(S='input_example', L='large_example', L2='large2_example', L3='large3_example')
ROUTES = ('upstream_cpu', 'fork_cpu', 'fork_gpu')

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def keyword(text, name, value):
    pattern = r'^(\s*' + re.escape(name) + r'\s+)\S+'
    if re.search(pattern, text, re.M | re.I):
        return re.sub(pattern, lambda m: m[1] + str(value), text, flags=re.M | re.I)
    return name + '  ' + str(value) + '\n' + text

def system(text):
    return re.search(r'^\s*System.Name\s+(\S+)', text, re.M | re.I)[1]

def main():
    manifest = dict(source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    ranks_per_node=32, threads=3, gpu_solver='gpusolver', cpu_solver='elpa2',
                    workloads=dict(S=1, L=1, L2=2, L3=2, sidia333_nc_cluster=1, cghelgeo=1),
                    inputs=[], data=[], binaries={})
    inputs = []
    for suite, directory in SUITES.items():
        for original in sorted((ROOT/'work'/directory).glob('*.dat')):
            inputs.append((suite, directory, original))
    for name in ('sidia333_nc_cluster', 'cghelgeo'):
        inputs.append((name, 'named', ROOT/'benchmarks/inputs'/(name+'.dat')))
    pao, vps = set(), set()
    for workload, directory, original in inputs:
        text = original.read_text()
        species = re.search(r'<Definition.of.Atomic.Species\s*\n(.*?)^\s*Definition.of.Atomic.Species>', text, re.M | re.S | re.I)[1]
        for line in species.splitlines():
            fields = line.split('#', 1)[0].split()
            if fields:
                pao.add(fields[1].split('-', 1)[0] + '.pao')
                vps.add(fields[2] + '.vps')
        raw = B/'raw'/directory/original.name
        raw.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, raw)
        reference = original.parent/(system(text)+'.out')
        record = dict(workload=workload, name=original.stem, source=str(original.relative_to(ROOT)),
                      raw=str(raw.relative_to(B)), sha256=sha(raw), system=system(text), routes={})
        if directory != 'named':
            shutil.copy2(reference, raw.parent/reference.name)
            record['reference_sha256'] = sha(reference)
        for route in ROUTES:
            target = B/('harness_'+route)/'work'/directory/original.name
            target.parent.mkdir(parents=True, exist_ok=True)
            prepared = keyword(text, 'DATA.PATH', str(B/'data'))
            for spelling in ('System.CurrentDirectory', 'System.CurrrentDirectory'):
                if re.search(r'^\s*'+re.escape(spelling)+r'\s+', prepared, re.M | re.I):
                    prepared = keyword(prepared, spelling, './')
            prepared = keyword(prepared, 'scf.eigen.lib', 'gpusolver' if route=='fork_gpu' else 'elpa2')
            target.write_text(prepared)
            record['routes'][route] = dict(path=str(target.relative_to(B)), sha256=sha(target))
            if directory != 'named':
                shutil.copy2(reference, target.parent/reference.name)
        manifest['inputs'].append(record)
    for directory, names in (('PAO', pao), ('VPS', vps)):
        (B/'data'/directory).mkdir(parents=True, exist_ok=True)
        for name in sorted(names):
            target = B/'data'/directory/name
            shutil.copy2(ROOT/'DFT_DATA19'/directory/name, target)
            manifest['data'].append(dict(path=str(target.relative_to(B)), sha256=sha(target)))
    for route in ROUTES:
        harness = B/('harness_'+route)
        (harness/'tools').mkdir(exist_ok=True)
        shutil.copy2(ROOT/'tools/run_gpu_suites.py', harness/'tools/run_gpu_suites.py')
        (harness/'DFT_DATA19').symlink_to(B/'data', target_is_directory=True)
        frozen = B/'binaries'/route
        frozen.mkdir(parents=True)
        if route == 'upstream_cpu':
            shutil.copy2(ROOT/'build/sirius-upstream-4.0.1/openmx4.0/source/openmx', frozen/'openmx')
            shutil.copy2(ROOT/'build/sirius-upstream-4.0.1/build-manifest.json', frozen/'build-manifest.json')
        else:
            for name in ('openmx', 'libopenmx_core.so', 'libopenmx_host_simd.so'):
                shutil.copy2(ROOT/'work/codex_sirius_8x3_20261031/current'/name, frozen/name)
        manifest['binaries'][route] = {p.name:sha(p) for p in frozen.iterdir() if p.is_file()}
    shutil.copy2(ROOT/'tools/compare_gpu_suites.py', B/'compare_gpu_suites.py')
    shutil.copy2(ROOT/'work/codex_sirius_8x3_20261031/suite-mpirun', B/'suite-mpirun')
    (B/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(dict(inputs=len(inputs), pao=len(pao), vps=len(vps), routes=ROUTES)))

if __name__ == '__main__':
    main()

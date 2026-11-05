#!/usr/bin/env python3
"""Validate the published campaign without MPI, installed binaries, or build files."""
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[2] / 'tools'))
from analyze_sve256_local import read
from compare_gpu_suites import read_output

BASELINE = ['baseline-02', 'baseline-03', 'baseline-04']
OPTIMIZED = ['own-final-01', 'own-final-02', 'own-final-03']

def main():
    for entry in (ROOT / 'SHA256SUMS').read_text().splitlines():
        digest, name = entry.split('  ', 1)
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    runs = {}
    with tempfile.TemporaryDirectory(prefix='openmx-record-check-') as temp:
        for name in BASELINE + OPTIMIZED + ['optimized-01']:
            run = read(name)
            c, r = run['command'], run['result']
            ranks = json.loads((ROOT / name / 'rank-audit.json').read_text())
            assert [v['rank'] for v in ranks] == list(range(c['ranks'])), name
            cores = []
            for rank in ranks:
                assert rank['sve_bits'] == 256 and rank['selected_simd'] == 'SVE256', name
                assert rank['threads'] == '1' and rank['blas_threads'] == 1, name
                assert rank['blas_core'].casefold() == c['expected_blas_core'].casefold(), name
                assert rank['zgemm_library'] == c['blas_library'], name
                assert len(rank['affinity']) == 1, name
                assert rank['affinity_before_exec'] == rank['affinity'], name
                cores.extend(rank['affinity'])
            assert len(set(cores)) == 32, name
            for filename, key in [('run.log', 'log_sha256'),
                                  ('sidia333_nc_cluster.out', 'output_sha256')]:
                data = gzip.decompress((ROOT / name / (filename + '.gz')).read_bytes())
                assert hashlib.sha256(data).hexdigest() == r[key], name
                if filename.endswith('.out'):
                    output = Path(temp) / 'result.out'
                    output.write_bytes(data)
                    energy, atoms, forces, grids = read_output(output)
                    assert [energy, [list(atom) for atom in atoms], forces, list(grids)] == [r['energy'], r['atoms'], r['forces'], r['grids']], name
            runs[name] = run
    for group in (BASELINE, OPTIMIZED):
        ref = runs[group[0]]['command']
        for name in group[1:]:
            for key in ('binary_sha256', 'libraries', 'blas_sha256', 'environment', 'ranks', 'threads'):
                assert runs[name]['command'][key] == ref[key], (name, key)
    reference = runs[BASELINE[0]]
    for name, run in runs.items():
        a, b = reference['result'], run['result']
        assert run['command']['effective_sha256'] == reference['command']['effective_sha256'], name
        assert a['atoms'] == b['atoms'] and a['grids'] == b['grids'], name
        assert a['scf_not_converged'] and b['scf_not_converged'], name
        assert abs(a['energy'] - b['energy']) <= 1e-7, name
        assert max(abs(x - y) for x, y in zip(a['forces'], b['forces'])) <= 1e-7, name
    summary = json.loads((ROOT / 'analysis.json').read_text())
    for key, names in [('baseline', BASELINE), ('optimized', OPTIMIZED)]:
        samples = [runs[name]['result']['wall_seconds'] for name in names]
        assert summary[key]['samples'] == samples and summary[key]['median'] == statistics.median(samples)
    assert summary['speedup'] == summary['baseline']['median'] / summary['optimized']['median']
    assert summary['all_checks_passed'] and summary['complete_blas_audits']
    checkpoint = runs['optimized-01']
    for name in OPTIMIZED:
        run = runs[name]
        for key in ('binary_sha256', 'blas_sha256', 'environment', 'ranks', 'threads'):
            assert run['command'][key] == checkpoint['command'][key], (name, key)
        for key in ('energy', 'forces', 'scf_history'):
            assert run['result'][key] == checkpoint['result'][key], (name, key)
    print('PASS: archive hashes, seven full outputs, rank audits, numerical agreement, medians and kernel-only comparison')

if __name__ == '__main__':
    main()

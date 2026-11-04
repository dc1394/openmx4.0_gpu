#!/usr/bin/env python3
"""Validate the published Neoverse N3 campaign without starting calculations."""
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re
import statistics

import audit_forces


def main():
    root = Path(__file__).resolve().parent
    for line in (root / 'SHA256SUMS').read_text().splitlines():
        digest, name = line.split('  ', 1)
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
    report = json.loads(gzip.decompress((root / 'results.json.gz').read_bytes()))
    analysis = json.loads((root / 'analysis.json').read_text())
    manifest = json.loads((root / 'manifest.json').read_text())
    runtime_input = gzip.decompress((root / 'input.dat.gz').read_bytes())
    input_hash = hashlib.sha256(runtime_input).hexdigest()
    assert all(r['input_sha256'] == input_hash for r in manifest['runs'])
    runs = report['runs']
    assert report['status'] == 'matched' and len(runs) == 8
    assert report['all_numerically_matched'] and not report['all_scf_converged']
    reference = runs[0]
    stdout_hashes = {r['name']: r['stdout_sha256'] for r in manifest['runs']}
    for run in runs:
        assert run['status'] == 'completed' and run['returncode'] == 0
        assert run['scf_steps'] == 25 and not run['scf_converged']
        assert len(run['forces_Ha_bohr']) == 648 and len(run['atoms']) == 216
        assert run['grids'] == [140, 140, 140]
        for key in ('energy_Ha', 'forces_Ha_bohr', 'atoms', 'grids', 'scf_history'):
            assert run[key] == reference[key], (run['directory'], key)
        audits = run['rank_audit']
        assert len(audits) == 8
        assert [a['rank'] for a in audits] == list(range(8))
        assert sorted(a['cpu_affinity'][0] for a in audits) == list(range(8))
        expected = 'scalar' if run['mode'] == 'scalar' else 'SVE128'
        assert all(a['ranks'] == 8 and a['omp_threads'] == 1 and a['sve_bits'] == 128
                   and a['mode'] == expected and len(a['cpu_affinity']) == 1 for a in audits)
        name = Path(run['directory']).name
        raw = gzip.decompress((root / 'logs' / (name + '.log.gz')).read_bytes())
        assert hashlib.sha256(raw).hexdigest() == stdout_hashes[name]
        log = raw.decode()
        assert 'The calculation was normally finished.' in log
        assert re.findall(r'The SCF was not fully achieved at MD=\s*(\d+)', log) == ['1']
        timing = re.search(r'^\s*Total Computational Time\s*=\s*\d+\s+\S+\s+\d+\s+(\S+)', log, re.M)
        assert timing and float(timing[1]) == run['native_seconds']
    csv_rows = list(csv.DictReader((root / 'timings.csv').open()))
    assert len(csv_rows) == 8
    for row, run in zip(csv_rows, runs):
        assert row['phase'] == run['phase'] and row['mode'] == run['mode']
        assert float(row['native_seconds']) == run['native_seconds']
    for mode in ('scalar', 'sve128'):
        measured = [r['native_seconds'] for r in runs if r['phase'] == 'timed' and r['mode'] == mode]
        assert len(measured) == 3
        median = statistics.median(measured)
        assert median == report['timings'][mode]['native_median_seconds']
        assert median == analysis[mode + '_median_seconds']
    assert analysis['speedup'] == analysis['scalar_median_seconds'] / analysis['sve128_median_seconds']
    audit_forces.verify(root)
    print('PASS: file hashes, 8 runs, rank audits, energies, 648 forces, SCF histories, native times and medians.')
    print('SCF did not converge in any run; agreement concerns the original 25-iteration workload.')


if __name__ == '__main__':
    main()

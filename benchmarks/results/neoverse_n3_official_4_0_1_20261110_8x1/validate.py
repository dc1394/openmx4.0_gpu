#!/usr/bin/env python3
"""Re-parse committed official-CPU benchmark evidence without running MPI."""
import csv
from decimal import Decimal
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import re


def sha(data):
    return hashlib.sha256(data).hexdigest()


def number(token):
    value = float(token.replace('D', 'E').replace('d', 'e'))
    assert math.isfinite(value), token
    return value


def main():
    root = Path(__file__).resolve().parent
    entries = (root / 'SHA256SUMS').read_text().splitlines()
    for line in entries:
        digest, name = line.split('  ', 1)
        assert sha((root / name).read_bytes()) == digest, name
    listed = {line.split('  ', 1)[1] for line in entries}
    actual = {str(p.relative_to(root)) for p in root.rglob('*')
              if p.is_file() and p.name != 'SHA256SUMS' and '__pycache__' not in p.parts}
    assert listed == actual, (listed - actual, actual - listed)
    report = json.loads(gzip.decompress((root / 'results.json.gz').read_bytes()))
    summary = json.loads((root / 'summary.json').read_text())
    manifest = json.loads((root / 'manifest.json').read_text())
    makefile_hash = next(v for k, v in manifest['recorded_hashes'].items()
                         if k.endswith('/source/makefile'))
    assert sha(gzip.decompress((root / 'build/makefile.gz').read_bytes())) == makefile_hash
    runtime = gzip.decompress((root / 'input.dat.gz').read_bytes())
    original = gzip.decompress((root / 'original.dat.gz').read_bytes())
    assert sha(runtime) == manifest['runtime_input_sha256'] == report['input_sha256']
    assert sha(original) == manifest['original_input_sha256']
    assert sha(original) == '9c5466abf560c6481b83ec9a51976f4b218856a1da330e5a209b9041239decd0'
    # Independently reconstruct the published driver's input preparation.
    overrides = json.loads((root / 'input-overrides.json').read_text())
    replaced = {key.lower() for key in overrides}
    lines = [line for line in original.decode().splitlines()
             if not line.split() or line.split()[0].lower() not in replaced]
    lines.extend(f'{key} {value}' for key, value in overrides.items())
    assert ('\n'.join(lines) + '\n').encode() == runtime
    assert report['status'] == 'completed'
    assert report['warmups'] == manifest['warmups'] == 1
    assert report['timed_repetitions'] == manifest['timed_runs'] == summary['timed_repetitions'] == 1
    runs = report['runs']
    assert [r['name'] for r in runs] == ['warmup-01', 'timed-01']
    assert report['mpi_ranks'] == 8 and report['omp_threads_per_rank'] == 1
    parsed = []
    for run, artifact in zip(runs, manifest['runs']):
        name = run['name']
        assert run['status'] == 'completed' and run['returncode'] == 0
        assert name == artifact['name']
        log_bytes = gzip.decompress((root / 'logs' / (name + '.log.gz')).read_bytes())
        output_bytes = gzip.decompress((root / 'outputs' / (name + '.out.gz')).read_bytes())
        assert sha(log_bytes) == artifact['stdout_sha256'] == run['artifacts']['run.log']['sha256']
        assert sha(output_bytes) == artifact['output_sha256'] == run['artifacts']['result.out']['sha256']
        assert sha(runtime) == artifact['input_sha256'] == run['artifacts']['case.dat']['sha256']
        log, output = log_bytes.decode(), output_bytes.decode()
        assert 'The calculation was normally finished.' in log
        assert re.findall(r'The SCF was not fully achieved at MD=\s*(\d+)', log) == ['1']
        assert not re.search(r'The SCF was achieved at MD=', log)
        audits = re.findall(r'\[openmx-8x1\] rank=(\d+) cpu=(\d+) OMP=1 BLAS=1 SVE_VL=128', log)
        assert len(audits) == 8
        assert {int(rank) for rank, _ in audits} == set(range(8))
        assert {int(cpu) for _, cpu in audits} == set(range(8))
        assert {rank: int(cpu) for rank, cpu in audits} == run['rank_audit']
        energy = number(re.findall(r'^\s*Utot\.\s+(\S+)', output, re.M)[-1])
        assert energy == run['energy_Ha']
        block = re.findall(r'<coordinates\.forces\s*\n(.*?)coordinates\.forces>', output, re.S)
        assert len(block) == 1
        rows = block[0].splitlines()
        assert int(rows[0]) == 216
        rows = [line.split() for line in rows[1:] if line.strip()]
        assert len(rows) == 216 and all(len(row) == 8 for row in rows)
        atoms = [[int(row[0]), row[1], *map(number, row[2:5])] for row in rows]
        forces = [number(token) for row in rows for token in row[5:8]]
        decimals = [Decimal(token) for row in rows for token in row[5:8]]
        assert [a[0] for a in atoms] == list(range(1, 217))
        assert atoms == run['atoms'] and forces == run['forces_Ha_bohr']
        assert len(forces) == 648
        grid = [int(re.search(rf'^\s*Num.Grid{i}\.\s+(\d+)', output, re.M)[1]) for i in (1, 2, 3)]
        assert grid == run['grids'] == [140, 140, 140]
        history = [[int(step), number(residual), number(e)] for step, residual, e in
                   re.findall(r'SCF=\s*(\d+)\s+NormRD=\s*(\S+)\s+Uele=\s*(\S+)', output)]
        assert history == run['scf_history'] and [s[0] for s in history] == list(range(1, 26))
        assert run['scf_steps'] == 25 and not run['scf_converged']
        elapsed = number(re.search(r'^\s*Elapsed\.Time\.\s+(\S+)', output, re.M)[1])
        assert elapsed == run['native_seconds']
        table = output.split('Elapsed.Time.', 1)[1]
        phases = {name.strip(): number(value) for name, value in re.findall(
            r'^\s*(.*?)\s*=\s*\d+\s+\S+\s+\d+\s+(\S+)\s*$', table, re.M)}
        assert phases == run['phases_seconds']
        assert phases['Total Computational Time'] == elapsed and all(v >= 0 for v in phases.values())
        parsed.append((energy, atoms, forces, decimals, history))
    assert parsed[0] == parsed[1], 'Warmup and timed numerical records differ'
    assert summary['native_seconds'] == report['median_native_seconds'] == runs[1]['native_seconds'] == 1721.589
    assert summary['wall_seconds'] == runs[1]['wall_seconds']
    assert runs[0]['native_seconds'] == 1714.709
    csv_rows = list(csv.DictReader((root / 'timings.csv').open()))
    assert len(csv_rows) == 2
    for row, run in zip(csv_rows, runs):
        assert row['name'] == run['name']
        for key in ('native_seconds', 'wall_seconds', 'energy_Ha'):
            assert float(row[key]) == run[key]
    phase_rows = list(csv.DictReader((root / 'phase-timings.csv').open()))
    assert {r['phase']: float(r['official_seconds']) for r in phase_rows} == runs[1]['phases_seconds']
    reference = root.parent / manifest['reference_directory']
    analysis_bytes = (reference / 'analysis.json').read_bytes()
    forces_bytes = (reference / 'force-components.csv').read_bytes()
    assert sha(analysis_bytes) == manifest['reference_analysis_sha256']
    assert sha(forces_bytes) == manifest['reference_force_components_sha256']
    analysis = json.loads(analysis_bytes)
    reference_forces = list(csv.DictReader(io.StringIO(forces_bytes.decode())))
    assert len(reference_forces) == 648
    maximum_energy_delta = 0.0
    maximum_force_delta = 0.0
    for run, (_, atoms, forces, decimals, _) in zip(runs, parsed):
        for index, row in enumerate(reference_forces):
            assert (int(row['atom']), row['species'], row['axis']) == (atoms[index // 3][0], atoms[index // 3][1], 'xyz'[index % 3])
        de = abs(run['energy_Ha'] - analysis['energy_Ha'])
        df = max(abs(a - float(b['reference_force_Ha_bohr'])) for a, b in zip(forces, reference_forces))
        decimal_df = max(abs(a - Decimal(b['reference_force_Ha_bohr'])) for a, b in zip(decimals, reference_forces))
        assert de <= 1e-7 and df <= 1e-7 and decimal_df <= Decimal('1e-7')
        assert de == run['published_comparison']['energy_delta_Ha']
        assert df == run['published_comparison']['max_force_component_delta_Ha_bohr']
        maximum_energy_delta = max(maximum_energy_delta, de)
        maximum_force_delta = max(maximum_force_delta, df)
    assert maximum_energy_delta == summary['max_delta_energy_vs_published_Ha']
    assert maximum_force_delta == summary['max_delta_force_vs_published_Ha_bohr']
    assert not summary['all_runs_scf_converged']
    print('PASS: hashes, exact runtime input, 2 completed launches, 8 rank/core audits,')
    print('      native times, 216 atoms / 648 forces, grids, 25-step histories and reference tolerances.')
    print('Timed run: 1721.589 s; warmup excluded. Both runs reached 25 SCF steps without convergence.')


if __name__ == '__main__':
    main()

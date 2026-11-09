"""Validate numerical outputs, precision modes, GPU execution and placement."""
import csv
from decimal import Decimal
import gzip
import itertools
import json
from pathlib import Path
import re

from run import A, ROOT, VARIANTS, save, sha

REFERENCE = ROOT / 'benchmarks/results/rtx5080_cghelgeo_strict_20261108/upstream'
TOLERANCE = Decimal('1e-7')


def physical(text):
    excluded = {'data.path', 'scf.criterion', 'scf.eigen.lib', 'scf.fp32.enable',
                'scf.gemmul8.enable', 'level.of.stdout', 'scf.restart'}
    return [line.strip() for line in text.splitlines()
            if line.split() and line.split()[0].lower() not in excluded]


def exact_output(folder):
    path = folder / 'cghelgeo.out'
    text = path.read_text() if path.exists() else gzip.decompress(path.with_suffix('.out.gz').read_bytes()).decode()
    energy = Decimal(re.search(r'^\s*Utot\.\s+(\S+)', text, re.M)[1])
    marker = re.search(r'<coordinates.forces\s*\n\s*(\d+)\s*\n', text)
    forces = [Decimal(v) for line in text[marker.end():].splitlines()[:int(marker[1])]
              for v in line.split()[5:8]]
    assert energy.is_finite() and len(forces) == 1950 and all(f.is_finite() for f in forces)
    return energy, forces


def quantum(number):
    return Decimal(1).scaleb(number.as_tuple().exponent)


def cpu_set(mask):
    result = set()
    for part in mask.split(','):
        if '-' in part:
            low, high = map(int, part.split('-'))
            result.update(range(low, high + 1))
        else:
            result.add(int(part))
    return result


def main():
    assert json.loads((A / 'progress.json').read_text()) == dict(status='completed', runs=4)
    provenance = json.loads((A / 'provenance.json').read_text())
    assert sha(A / 'bin/openmx') == provenance['binaries']['snapshot_sha256']
    for name, digest in provenance['libraries'].items():
        assert sha(A / 'lib' / name) == digest
    for data in provenance['datasets']:
        assert sha(A / data['path']) == data['sha256']
    for data in provenance['seed_files']:
        assert sha(A / 'seed/cghelgeo_rst' / data['path']) == data['sha256']
    original = (A / 'original-cghelgeo.dat').read_text()
    records = {'upstream': json.loads((REFERENCE / 'result.json').read_text())}
    folders = {'upstream': REFERENCE}
    runtime = []
    for variant in VARIANTS:
        folder = A / 'runs' / variant
        folders[variant] = folder
        result = json.loads((folder / 'result.json').read_text())
        records[variant] = result
        log = (folder / 'run.log').read_text()
        text = (folder / 'input.dat').read_text()
        assert result['status'] == 'completed' and result['converged'] and result['returncode'] == 0
        assert result['criterion'] == 1e-10 and result['scf_steps'] < 200
        assert physical(text) == physical(original)
        assert sha(folder / 'input.dat') == result['input_sha256']
        assert sha(folder / 'cghelgeo.out') == result['output_sha256']
        assert re.search(r'^scf\.criterion\s+1e-10\s*$', text, re.M)
        assert '<Restart>  Found restart files' in log and '<Restart>  Could not find restart files' not in log
        assert log.count('The calculation was normally finished.') == 1
        assert log.count('The SCF was achieved at MD=') == 1 and 'The SCF was not fully achieved' not in log
        trace = [dict(SCF_step=index + 1, mixing_weight=float(weight), band_energy_hartree=float(energy),
                      band_energy_change_hartree=float(delta), displayed_sqrt_NormRD=float(norm),
                      criterion=float(criterion))
                 for index, (weight, energy, delta, norm, criterion) in enumerate(re.findall(
                     r'<DFT>\s+Mixing_weight=\s*(\S+)\s*\n'
                     r'<DFT>\s+Uele\s*=\s*(\S+)\s+dUele\s*=\s*(\S+)\s*\n'
                     r'<DFT>\s+NormRD\s*=\s*(\S+)\s+Criterion\s*=\s*(\S+)', log))]
        assert len(trace) == result['scf_steps']
        # DFT.c compares the unrooted mixing norm to 10*criterion, while
        # its diagnostic displays sqrt(abs(norm)). Allow only display rounding
        # of the band-energy change; the native convergence marker is required.
        final = trace[-1]
        assert final['band_energy_change_hartree'] <= 1e-10 + 0.5e-12
        assert final['displayed_sqrt_NormRD'] ** 2 < 10 * 1e-10
        save(folder / 'scf-trace.json', trace)
        ranks = [json.loads(p.read_text()) for p in sorted(folder.glob('rank-??.json'))]
        assert [r['rank'] for r in ranks] == list(range(18))
        libraries = []
        for rank in ranks:
            assert rank['cpus'] == [rank['rank'], rank['rank'] + 18]
            threads = json.loads((folder / ('rank-%02d-threads.json' % rank['rank'])).read_text())
            expected = set(rank['cpus'])
            main = next(t for t in threads if t['tid'] == rank['pid'])
            assert cpu_set(main['cpus']) <= expected
            pinned = [cpu_set(t['cpus']) for t in threads if cpu_set(t['cpus']) <= expected]
            assert set().union(*pinned) == expected
            maps = (folder / ('rank-%02d-maps.txt' % rank['rank'])).read_text()
            assert str(A / 'lib/libopenmx_core.so') in maps
            assert str(A / 'lib/libopenmx_host_simd.so') in maps
            assert 'libnvomp.so' in maps and 'libiomp5' not in maps and 'libgomp' not in maps
            assert '/hpcx-2.50/ompi5/' in maps
            libraries.append(dict(rank=rank['rank'],
                libraries=sorted({line.split()[-1] for line in maps.splitlines()
                                  if len(line.split()) >= 6 and '/' in line.split()[-1]})))
        save(folder / 'runtime-libraries.json', libraries)
        save(folder / 'rank-placement.json', ranks)
        detail = dict(variant=variant, all_18_ranks_verified=True,
                      all_18_runtime_maps_verified=True, main_and_OpenMP_affinity_verified=True,
                      restart_found=True)
        if variant == 'cpu_avx512_off':
            assert re.search(r'host SIMD\s*=\s*AVX-512', log)
            assert 'GPU-accelerated' not in log
            detail.update(AVX512_runtime_selected=True, GPU_disabled=True)
        else:
            assert 'GPU initialization failed' not in log
            assert 'GPU-accelerated' in log
            assert 'GPU dense memory admission declined' not in log
            assert 'Calculation of the VNA projector matrix (GPU-accelerated)' in log
            for number in [3, 4, 6, 7]:
                assert 'Force calculation #%d (GPU-accelerated)' % number in log
            modes = [json.loads(line) for p in sorted(folder.glob('rank-??-cusolver.jsonl'))
                     for line in p.read_text().splitlines()]
            assert modes and all(m['status'] == m['read_status'] == 0 for m in modes)
            solves = [m for m in modes if m['event'] in ['Xsyevd', 'Xsyevdx']]
            assert solves and max(m['n'] for m in solves) == 6670
            emulated = [m for m in solves if m['mode'] == m['emulated_mode'] and m['arithmetic'].endswith('fp64')]
            if variant != 'gpu_default':
                assert all(m['arithmetic'].endswith('fp64') for m in solves)
            else:
                assert any(m['arithmetic'].endswith('fp32') for m in solves)
                assert any(m['arithmetic'].endswith('fp64') for m in solves)
            if variant == 'gpu_off_native':
                assert all(m['mode'] == m['native_mode'] for m in modes)
                assert all(m['arithmetic'].endswith('fp64') for m in solves)
                assert 'scf.gemmul8.enable=off' in log
            else:
                assert emulated
                assert 'scf.gemmul8.enable=off' not in log
            gemms = re.findall(r'real (\d+) \+ (\d+) calls,.*?complex (\d+) \+ (\d+) calls', log)
            assert gemms
            emulated_calls = sum(int(a) + int(c) for a, b, c, d in gemms)
            native_calls = sum(int(b) + int(d) for a, b, c, d in gemms)
            if variant == 'gpu_off_native':
                assert emulated_calls == 0 and native_calls > 0
            else:
                assert emulated_calls > 0
            detail.update(cusolver_records=len(modes), cusolver_solves=len(solves),
                emulated_fp64_solves=len(emulated), largest_eigensystem=max(m['n'] for m in solves),
                solver_arithmetic=sorted({m['arithmetic'] for m in solves}),
                observed_math_modes=sorted({m['mode'] for m in solves}),
                gemmul8_calls=emulated_calls, native_cublas_calls=native_calls)
            save(folder / 'cusolver-modes.json', modes)
        if variant != 'gpu_default':
            assert 'scf.fp32.enable=off' in log and 'upstream float projector products are retained' in log
        else:
            assert 'scf.fp32.enable=off' not in log
        runtime.append(detail)
    assert physical((REFERENCE / 'input.dat').read_text()) == physical(original)
    for result in records.values():
        assert result['status'] == 'completed' and result['converged'] and result['criterion'] == 1e-10
        assert len(result['atoms']) == 650 and len(result['forces_hartree_per_bohr']) == 1950
    comparisons = []
    for left, right in itertools.combinations(records, 2):
        x, y = records[left], records[right]
        assert x['FFT_grid'] == y['FFT_grid']
        assert all(a[:2] == b[:2] for a, b in zip(x['atoms'], y['atoms']))
        coordinates = max(abs(u - v) for a, b in zip(x['atoms'], y['atoms']) for u, v in zip(a[2:], b[2:]))
        e1, f1 = exact_output(folders[left])
        e2, f2 = exact_output(folders[right])
        de = abs(e1 - e2)
        diffs = [abs(u - v) for u, v in zip(f1, f2)]
        df = max(diffs)
        index = diffs.index(df)
        e_upper = de + (quantum(e1) + quantum(e2)) / 2
        f_upper = max(d + (quantum(u) + quantum(v)) / 2 for d, u, v in zip(diffs, f1, f2))
        comparisons.append(dict(baseline=left, candidate=right, energy_difference_hartree=float(de),
            max_force_component_difference_hartree_per_bohr=float(df),
            energy_difference_rounding_upper_bound=float(e_upper),
            force_difference_rounding_upper_bound=float(f_upper), max_force_atom=index // 3 + 1,
            max_force_axis='xyz'[index % 3], max_coordinate_difference=coordinates,
            passed=e_upper <= TOLERANCE and f_upper <= TOLERANCE and coordinates <= 1.000001e-5))
    analysis = dict(all_passed=len(comparisons) == 10 and all(c['passed'] for c in comparisons),
        comparison=comparisons, criterion=1e-10, inherited_float_products_retained=True,
        tolerance=dict(energy_hartree=1e-7, force_component_hartree_per_bohr=1e-7),
        printed_rounding_included=True, protocol='Same-seed continuation regression; not cold-start timing.',
        upstream_reference_reused=True,
        runs=[dict(variant=name, energy_hartree=r['energy_hartree'], SCF_steps=r['scf_steps'])
              for name, r in records.items()])
    save(A / 'analysis.json', analysis)
    with (A / 'comparisons.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparisons[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(comparisons)
    mps = json.loads((A / 'mps-server.json').read_text())
    assert mps['returncode'] == 0 and re.search(r'\d+', mps['stdout'])
    assert json.loads((A / 'mps-stop.json').read_text())['returncode'] == 0
    save(A / 'runtime-validation.json', dict(passed=True, variants=runtime,
        frozen_binary_and_library_hashes_verified=True, common_seed_hashes_verified=True,
        dataset_hashes_verified=True, MPS_observed=True, private_MPS_stopped=True))
    print(json.dumps(analysis, indent=2))
    assert analysis['all_passed']


if __name__ == '__main__':
    main()

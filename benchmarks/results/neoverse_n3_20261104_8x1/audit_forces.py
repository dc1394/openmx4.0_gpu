#!/usr/bin/env python3
"""Audit exact printed force components; --collect re-extracts original outputs."""
import argparse
import csv
from decimal import Decimal, localcontext
import gzip
import hashlib
import io
import itertools
import json
from pathlib import Path
import re


BLOCK = re.compile(rb'^<coordinates\.forces[ \t]*\n[ \t]*(\d+)[ \t]*\n(.*?)^coordinates\.forces>', re.M | re.S)
AXES = 'xyz'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    raw = path.read_bytes()
    if path.suffix == '.gz':
        raw = gzip.decompress(raw)
    return json.loads(raw, parse_float=Decimal)


def parse_block(raw):
    matches = list(BLOCK.finditer(raw))
    require(len(matches) == 1, 'expected exactly one coordinates.forces block')
    match = matches[0]
    count = int(match[1])
    rows = [line.split() for line in match[2].splitlines() if line.strip()]
    require(count == 216 and len(rows) == count, 'expected 216 complete atom rows')
    atoms, coordinates, tokens, forces = [], [], [], []
    for i, row in enumerate(rows, 1):
        require(len(row) == 8 and int(row[0]) == i, 'invalid row width or atom ordering')
        atoms.append((i, row[1].decode('ascii')))
        values = [Decimal(x.decode('ascii')) for x in row[2:]]
        require(all(x.is_finite() for x in values), 'nonfinite coordinate or force')
        coordinates.append(tuple(values[:3]))
        tokens.extend(x.decode('ascii') for x in row[5:8])
        forces.extend(values[3:])
    return dict(atoms=atoms, coordinates=coordinates, forces=forces, tokens=tokens,
                raw_block=match[0])


def root_mean_square(values):
    with localcontext() as context:
        context.prec = 50
        return (sum(x*x for x in values) / Decimal(len(values))).sqrt()


def norm(values):
    with localcontext() as context:
        context.prec = 50
        return sum(x*x for x in values).sqrt()


def metric(value):
    return '0' if value == 0 else format(value, '.12E')


def compare(left, right):
    require(left['atoms'] == right['atoms'], 'atom identities/order differ')
    require(left['coordinates'] == right['coordinates'], 'printed coordinates differ')
    differences = [a-b for a, b in zip(left['forces'], right['forces'])]
    vector_differences = [norm(differences[i:i+3]) for i in range(0, 648, 3)]
    return dict(max_abs_component_difference_Ha_bohr=metric(max(map(abs, differences))),
                rms_component_difference_Ha_bohr=metric(root_mean_square(differences)),
                max_atom_vector_difference_Ha_bohr=metric(max(vector_differences)),
                differing_components=sum(x != 0 for x in differences),
                differing_atoms=sum(x != 0 for x in vector_differences),
                differing_printed_tokens=sum(a != b for a, b in zip(left['tokens'], right['tokens'])),
                force_block_bytes_equal=left['raw_block'] == right['raw_block'],
                per_axis_max_abs_difference_Ha_bohr={axis: metric(max(map(abs, differences[k::3])))
                                                    for k, axis in enumerate(AXES)})


def csv_text(rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def collect(root, source):
    manifest = read_json(root / 'manifest.json')
    archive = read_json(root / 'results.json.gz')
    original = {r['name']: r for r in manifest['runs']}
    records = []
    for run in archive['runs']:
        name = Path(run['directory']).name
        raw = (source / name / 'result.out').read_bytes()
        require(sha(raw) == original[name]['output_sha256'], f'{name}: original output hash changed')
        require(b'xyz-coordinates (Ang) and forces (Hartree/Bohr)' in raw, 'force units not confirmed')
        block = parse_block(raw)['raw_block']
        records.append(dict(name=name, source_output_sha256=sha(raw),
                            block_sha256=sha(block), block=block.decode('ascii')))
    payload = dict(source_commit=manifest['measured_source_commit'],
                   units='Hartree/Bohr', extraction='one complete coordinates.forces block per result.out',
                   original_full_output_hashes_verified=True, runs=records)
    data = (json.dumps(payload, indent=2) + '\n').encode()
    (root / 'force-blocks.json.gz').write_bytes(gzip.compress(data, mtime=0))


def analyze(root):
    payload = read_json(root / 'force-blocks.json.gz')
    manifest = read_json(root / 'manifest.json')
    archive = read_json(root / 'results.json.gz')
    require(payload['source_commit'] == manifest['measured_source_commit'], 'source commit differs')
    require(payload['units'] == 'Hartree/Bohr', 'unexpected force units')
    records = {r['name']: r for r in payload['runs']}
    require(len(records) == len(payload['runs']) == len(archive['runs']) == 8, 'need all eight unique runs')
    originals = {r['name']: r for r in manifest['runs']}
    frames, run_reports = {}, []
    for run in archive['runs']:
        name = Path(run['directory']).name
        record = records[name]
        raw = record['block'].encode('ascii')
        require(sha(raw) == record['block_sha256'], f'{name}: force block hash differs')
        require(record['source_output_sha256'] == originals[name]['output_sha256'], 'output provenance differs')
        frame = parse_block(raw)
        require(frame['forces'] == [Decimal(str(x)) for x in run['forces_Ha_bohr']], 'force archive differs')
        require(frame['atoms'] == [(a[0], a[1]) for a in run['atoms']], 'atom archive differs')
        require(frame['coordinates'] == [tuple(Decimal(str(x)) for x in a[2:]) for a in run['atoms']],
                'coordinate archive differs')
        require(all(re.fullmatch(r'-?\d+\.\d{12}', t) for t in frame['tokens']), 'expected 12 fractional digits')
        frames[name] = frame
        run_reports.append(dict(name=name, phase=run['phase'], mode=run['mode'],
            repeat=run['repeat'], block_sha256=sha(raw), force_components=648,
            nonzero_components=sum(x != 0 for x in frame['forces']),
            negative_zero_tokens=sum(Decimal(t) == 0 and t.startswith('-') for t in frame['tokens']),
            scf_steps=run['scf_steps'], scf_converged=run['scf_converged']))
    reference_name = 'timed-01-scalar'
    reference = frames[reference_name]
    pairs, pair_csv = [], []
    for left, right in itertools.combinations(run_reports, 2):
        values = compare(frames[left['name']], frames[right['name']])
        group = 'cross_mode' if left['mode'] != right['mode'] else left['mode'] + '_repeat'
        pairs.append(dict(left=left['name'], right=right['name'], group=group,
                          both_timed=left['phase'] == right['phase'] == 'timed', **values))
        pair_csv.append(dict(left=left['name'], right=right['name'], group=group,
            max_abs_component_difference_Ha_bohr=values['max_abs_component_difference_Ha_bohr'],
            rms_component_difference_Ha_bohr=values['rms_component_difference_Ha_bohr'],
            max_atom_vector_difference_Ha_bohr=values['max_atom_vector_difference_Ha_bohr'],
            differing_components=values['differing_components'], differing_atoms=values['differing_atoms'],
            differing_printed_tokens=values['differing_printed_tokens'],
            force_block_bytes_equal=values['force_block_bytes_equal']))
    force_values = reference['forces']
    norms = [norm(force_values[i:i+3]) for i in range(0, 648, 3)]
    component_rows = []
    for i, (atom, species) in enumerate(reference['atoms']):
        for axis_index, axis in enumerate(AXES):
            k = 3*i + axis_index
            values = [frame['forces'][k] for frame in frames.values()]
            component_rows.append(dict(atom=atom, species=species, axis=axis,
                reference_force_Ha_bohr=reference['tokens'][k],
                minimum_force_Ha_bohr=str(min(values)), maximum_force_Ha_bohr=str(max(values)),
                spread_Ha_bohr=metric(max(values)-min(values)),
                distinct_printed_tokens=len({frame['tokens'][k] for frame in frames.values()})))
    numeric_equal = all(p['differing_components'] == 0 for p in pairs)
    text_equal = all(p['force_block_bytes_equal'] for p in pairs)
    audit = dict(status='matched' if numeric_equal and text_equal else 'mismatch',
        measured_source_commit=payload['source_commit'], reference=reference_name,
        runs=8, atoms_per_run=216, components_per_run=648, parsed_force_values=8*648,
        all_pairs=len(pairs), cross_mode_pairs=sum(p['group'] == 'cross_mode' for p in pairs),
        within_mode_pairs=sum(p['group'] != 'cross_mode' for p in pairs),
        timed_only_pairs=sum(p['both_timed'] for p in pairs),
        units='Hartree/Bohr', decimal_fractional_digits=12, printed_quantum_Ha_bohr='1E-12',
        output_format=dict(source='source/MD_pac.c', force_format='%15.12f',
                           values='-Gxyz[k][17], -Gxyz[k][18], -Gxyz[k][19]'),
        all_components_decimal_equal=numeric_equal, all_force_blocks_byte_equal=text_equal,
        common_force_block_sha256=run_reports[0]['block_sha256'] if text_equal else None,
        maximum_pairwise_component_difference_Ha_bohr=metric(max(Decimal(p['max_abs_component_difference_Ha_bohr']) for p in pairs)),
        maximum_pairwise_rms_difference_Ha_bohr=metric(max(Decimal(p['rms_component_difference_Ha_bohr']) for p in pairs)),
        maximum_pairwise_atom_vector_difference_Ha_bohr=metric(max(Decimal(p['max_atom_vector_difference_Ha_bohr']) for p in pairs)),
        reference_magnitudes=dict(nonzero_components=sum(x != 0 for x in force_values),
            zero_components=sum(x == 0 for x in force_values),
            max_abs_component_Ha_bohr=str(max(map(abs, force_values))),
            rms_component_Ha_bohr=metric(root_mean_square(force_values)),
            min_atom_force_norm_Ha_bohr=metric(min(norms)), max_atom_force_norm_Ha_bohr=metric(max(norms)),
            max_norm_atom=norms.index(max(norms))+1,
            net_force_Ha_bohr={axis: str(sum(force_values[k::3])) for k, axis in enumerate(AXES)},
            net_force_norm_Ha_bohr=metric(norm([sum(force_values[k::3]) for k in range(3)]))),
        all_scf_converged=all(run['scf_converged'] for run in archive['runs']),
        limitations=['Forces of the original 25-SCF unconverged workload only.',
                     'Equality below 1e-12 Ha/bohr output resolution and in-memory bit identity are not established.',
                     'Vector norms and net forces are computed from the rounded printed components.'],
        run_records=run_reports, pairwise=pairs)
    summary = (
        'Force agreement audit: Neoverse N3, 8 MPI x 1 OpenMP\n'
        f"Status: {audit['status']}; 216 atoms x 3 components x 8 runs = 5184 force values.\n"
        'All 28 run pairs: 16 scalar/SVE128 pairs and 12 same-mode repeat pairs.\n'
        'There are 15 pairs among the six timed runs; warmups are included in the full audit.\n'
        f"Maximum component difference: {audit['maximum_pairwise_component_difference_Ha_bohr']} Ha/bohr\n"
        f"Maximum RMS component difference: {audit['maximum_pairwise_rms_difference_Ha_bohr']} Ha/bohr\n"
        f"Maximum atom-vector difference: {audit['maximum_pairwise_atom_vector_difference_Ha_bohr']} Ha/bohr\n"
        f'All force blocks byte-identical, including signed zeros: {text_equal}\n'
        f"Common block SHA256: {audit['common_force_block_sha256']}\n"
        f"Reference: {audit['reference_magnitudes']['nonzero_components']} nonzero force components, "
        f"{audit['reference_magnitudes']['zero_components']} zero components.\n"
        f"Largest absolute component: {audit['reference_magnitudes']['max_abs_component_Ha_bohr']} Ha/bohr\n"
        f"Largest atom-force norm: {metric(max(norms))} Ha/bohr (atom {norms.index(max(norms))+1})\n"
        'These are nonzero forces, not a trivial all-zero match.\n'
        'source/MD_pac.c writes -Gxyz[k][17], -Gxyz[k][18], -Gxyz[k][19]\n'
        'using %15.12f in Hartree/Bohr.\n'
        'Decimal comparisons preserve all 12 fractional digits and signed-zero tokens.\n'
        'Differences below the 1e-12 Ha/bohr output resolution cannot be excluded.\n'
        'This audit does not establish in-memory bit identity or converged-force accuracy.\n'
        'All eight calculations reached the original 25-SCF limit without convergence.\n')
    return {'force-audit.json': json.dumps(audit, indent=2) + '\n',
            'force-audit.txt': summary, 'force-components.csv': csv_text(component_rows),
            'force-pairs.csv': csv_text(pair_csv)}


def verify(root):
    outputs = analyze(root)
    for name, value in outputs.items():
        require((root / name).read_text() == value, f'{name}: audit is not reproducible')
    audit = json.loads(outputs['force-audit.json'])
    require(audit['status'] == 'matched', 'force agreement check failed')
    print('PASS: 5184 printed force values, 28 pairs, Decimal/component/vector/RMS checks and byte-identical blocks.')
    print('Output precision: 1e-12 Ha/bohr. All eight runs remain SCF-unconverged.')
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collect', type=Path, help='original completed campaign directory')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if args.collect:
        collect(root, args.collect.resolve())
        for name, value in analyze(root).items():
            (root / name).write_text(value)
    verify(root)


if __name__ == '__main__':
    main()

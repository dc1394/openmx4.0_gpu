#!/usr/bin/env python3
"""Regression checks for force extraction and comparisons at printed precision."""
from decimal import Decimal
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import audit_forces as audit


class ForceAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parent
        cls.payload = json.loads(gzip.decompress((cls.root / 'force-blocks.json.gz').read_bytes()))
        cls.raw = cls.payload['runs'][0]['block'].encode()
        cls.frame = audit.parse_block(cls.raw)

    def changed_row(self, atom, changes, raw=None):
        lines = (self.raw if raw is None else raw).splitlines(keepends=True)
        fields = lines[atom+1].split()
        for index, value in changes.items():
            fields[index] = value.encode()
        lines[atom+1] = b' '.join(fields) + b'\n'
        return b''.join(lines)

    def test_actual_block_is_nontrivial_and_has_signed_zeros(self):
        self.assertEqual(len(self.frame['forces']), 648)
        self.assertEqual(sum(x != 0 for x in self.frame['forces']), 646)
        self.assertEqual(sum(Decimal(t) == 0 and t.startswith('-') for t in self.frame['tokens']), 2)

    def test_one_quantum_difference_is_detected_without_float_tolerance(self):
        value = self.frame['forces'][0] + Decimal('1e-12')
        other = audit.parse_block(self.changed_row(1, {5: format(value, '.12f')}))
        result = audit.compare(self.frame, other)
        self.assertEqual(result['differing_components'], 1)
        self.assertEqual(result['differing_atoms'], 1)
        self.assertEqual(Decimal(result['max_abs_component_difference_Ha_bohr']), Decimal('1e-12'))
        self.assertEqual(Decimal(result['max_atom_vector_difference_Ha_bohr']), Decimal('1e-12'))
        self.assertGreater(Decimal(result['rms_component_difference_Ha_bohr']), 0)

    def test_cancelling_errors_are_not_hidden_by_net_force(self):
        raw = self.changed_row(1, {5: format(self.frame['forces'][0]+Decimal('1e-12'), '.12f')})
        other = audit.parse_block(self.changed_row(2,
            {5: format(self.frame['forces'][3]-Decimal('1e-12'), '.12f')}, raw=raw))
        self.assertEqual([sum(self.frame['forces'][k::3]) for k in range(3)],
                         [sum(other['forces'][k::3]) for k in range(3)])
        result = audit.compare(self.frame, other)
        self.assertEqual(result['differing_components'], 2)
        self.assertEqual(result['differing_atoms'], 2)
        self.assertEqual(Decimal(result['max_atom_vector_difference_Ha_bohr']), Decimal('1e-12'))

    def test_atom_vector_difference_uses_all_axes(self):
        other = audit.parse_block(self.changed_row(1, {
            5: format(self.frame['forces'][0]+Decimal('1e-12'), '.12f'),
            6: format(self.frame['forces'][1]+Decimal('1e-12'), '.12f')}))
        result = audit.compare(self.frame, other)
        self.assertEqual(result['differing_atoms'], 1)
        self.assertGreater(Decimal(result['max_atom_vector_difference_Ha_bohr']), Decimal('1e-12'))
        self.assertLess(Decimal(result['max_atom_vector_difference_Ha_bohr']), Decimal('1.5e-12'))

    def test_signed_zero_token_difference_is_not_numeric_difference(self):
        i = next(i for i, t in enumerate(self.frame['tokens']) if Decimal(t) == 0)
        other = audit.parse_block(self.changed_row(i//3+1, {5+i%3: '0.000000000000'}))
        result = audit.compare(self.frame, other)
        self.assertEqual(result['differing_components'], 0)
        self.assertEqual(result['differing_printed_tokens'], 1)
        self.assertFalse(result['force_block_bytes_equal'])

    def test_wrong_atom_order_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'atom ordering'):
            audit.parse_block(self.changed_row(2, {0: '1'}))

    def test_truncated_force_table_is_rejected(self):
        lines = self.raw.splitlines(keepends=True)
        del lines[-2]
        with self.assertRaisesRegex(ValueError, 'complete atom rows'):
            audit.parse_block(b''.join(lines))

    def test_duplicate_force_blocks_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            audit.parse_block(self.raw+b'\n'+self.raw)

    def test_nonfinite_component_is_rejected(self):
        for value in ('NaN', 'Infinity', '-Infinity'):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'nonfinite'):
                audit.parse_block(self.changed_row(1, {5: value}))

    def test_coordinate_change_is_rejected(self):
        other = audit.parse_block(self.changed_row(1, {2: '0.67889'}))
        with self.assertRaisesRegex(ValueError, 'coordinates differ'):
            audit.compare(self.frame, other)

    def test_block_mutation_cannot_bypass_original_force_archive(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ('manifest.json', 'results.json.gz'):
                shutil.copyfile(self.root / name, root / name)
            payload = json.loads(json.dumps(self.payload))
            value = self.frame['forces'][0] + Decimal('1e-12')
            raw = self.changed_row(1, {5: format(value, '.12f')})
            payload['runs'][0]['block'] = raw.decode()
            payload['runs'][0]['block_sha256'] = hashlib.sha256(raw).hexdigest()
            (root/'force-blocks.json.gz').write_bytes(gzip.compress(json.dumps(payload).encode()))
            with self.assertRaisesRegex(ValueError, 'force archive differs'):
                audit.analyze(root)


if __name__ == '__main__':
    unittest.main()

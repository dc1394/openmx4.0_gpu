#!/usr/bin/env python3
"""Regression checks for the ARM numerical validation's pass/fail criteria."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from validate_arm_cpu import check_scf, compare, test_environment


class ArmValidationTests(unittest.TestCase):
    # Representative Si8 stop: the printed residual is sqrt(abs(norm[0]));
    # it is larger than scf.criterion even though the energy has converged.
    history = ('SCF= 16 NormRD= 0.000000072583 Uele= -9.676351961777\n'
               'SCF= 17 NormRD= 0.000000020419 Uele= -9.676351961842\n')

    def test_converged_energy_with_larger_printed_residual(self):
        result = check_scf('The SCF was achieved at MD=   1\n', self.history)
        self.assertEqual(result['scf_steps'], 17)
        self.assertGreater(result['scf_residual'], 1e-10)
        self.assertLess(result['scf_band_energy_change'], 1e-10)

    def test_normal_exit_without_convergence_is_rejected(self):
        for log in ('The calculation was normally finished.',
                    'The SCF was not fully achieved at MD=   1\n'
                    'The calculation was normally finished.',
                    'The SCF was achieved at MD=   2\n'):
            with self.subTest(log=log), self.assertRaises(ValueError):
                check_scf(log, self.history)

    def test_incomplete_or_nonfinite_history_is_rejected(self):
        for history in ('', self.history.splitlines()[0],
                        self.history.replace('0.000000020419', 'nan'),
                        self.history.replace('-9.676351961842', 'inf')):
            with self.subTest(history=history), self.assertRaises(ValueError):
                check_scf('The SCF was achieved at MD=   1\n', history)

    def test_ctest_does_not_inherit_cpu_force_or_leak_workaround(self):
        env = test_environment({'PATH': '/usr/bin', 'OPENMX_GPU': '0',
                                'OPENMX_HOST_SIMD': 'sve', 'ASAN_OPTIONS': 'detect_leaks=0',
                                'LSAN_OPTIONS': 'exitcode=0'})
        self.assertEqual(env['PATH'], '/usr/bin')
        for key in ('OPENMX_GPU', 'OPENMX_HOST_SIMD', 'ASAN_OPTIONS', 'LSAN_OPTIONS'):
            self.assertNotIn(key, env)

    def test_force_difference_cannot_be_hidden_by_small_energy_difference(self):
        reference = dict(energy=-1.0, atoms=[(1, 'C', 0., 0., 0.)],
                         forces=[0., 0., 0.], grids=(8, 8, 8))
        candidate = dict(reference, energy=-1.0+1e-10, forces=[0., 8e-7, 0.])
        result = compare(reference, candidate, 1e-7)
        self.assertFalse(result['passed'])
        self.assertGreater(result['max_force_difference'], 1e-7)


if __name__ == '__main__':
    unittest.main()

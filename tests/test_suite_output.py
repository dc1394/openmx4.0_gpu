#!/usr/bin/env python3
"""Final-energy parsing must also cover Cluster-LNO and multi-step outputs."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from compare_gpu_suites import read_output


class SuiteOutputTest(unittest.TestCase):
    def read(self, energies):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "example.out"
            path.write_text(energies + """
Num.Grid1. 32
Num.Grid2. 40
Num.Grid3. 48
<coordinates.forces
1
1 Si 0.1 0.2 0.3 -0.4 0.5 -0.6
coordinates.forces>
""")
            return read_output(path)

    def test_summary_wins_over_iteration_energy(self):
        energy, atoms, forces, grids = self.read("Utot = -1.0\nUtot. -2.5\nUtot = -3.0\n")
        self.assertEqual(energy, -2.5)
        self.assertEqual(atoms, [(1, "Si", 0.1, 0.2, 0.3)])
        self.assertEqual(forces, [-0.4, 0.5, -0.6])
        self.assertEqual(grids, (32, 40, 48))

    def test_last_cluster_lno_md_step(self):
        self.assertEqual(self.read("  Utot = -1.0\n  Utot    = -2.75e1\n")[0], -27.5)

    def test_missing_energy(self):
        with self.assertRaisesRegex(ValueError, "Missing total energy"):
            self.read("Uele = -4.0\n")

    def test_nonfinite_energy_is_rejected(self):
        for energy in ("Utot. nan\n", "Utot = inf\n"):
            with self.subTest(energy=energy), self.assertRaisesRegex(ValueError, "nonfinite"):
                self.read(energy)


if __name__ == "__main__":
    unittest.main()

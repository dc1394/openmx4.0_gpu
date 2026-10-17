#!/usr/bin/env python3
"""Exercise suite input discovery without launching OpenMX or MPI."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


TOOL = Path(__file__).resolve().parents[1] / 'tools' / 'run_runtest_l3.py'


class RuntestPaths(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='openmx-runtest-paths-')
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.repo = self.directory / 'checkout'
        for name in ('source', 'work/input_example', 'DFT_DATA19'):
            (self.repo / name).mkdir(parents=True)
        self.inputs = self.repo / 'work' / 'input_example'
        (self.inputs / 'sample.dat').write_text('System.Name sample\n')
        (self.inputs / 'sample.out').write_text('reference\n')
        self.binary = self.directory / 'openmx'
        self.binary.write_text('mock binary; the launcher does not execute it\n')
        self.launcher = self.directory / 'mock-mpirun'
        self.launcher.write_text(
            '#!' + sys.executable + '\n'
            'from pathlib import Path\n'
            'Path("runtest.result").write_text('
            '"Elapsed time(s)= 0.01 diff Utot= 0.0 diff Force= 0.0\\n")\n')
        self.launcher.chmod(0o755)
        self.output = self.directory / 'results'

    def stage(self, external):
        if external:
            build = self.directory / 'outside-checkout' / 'hip'
            build.mkdir(parents=True)
            (build / '.openmx-backend.json').write_text(json.dumps(
                {'backend': 'HIP', 'source_root': str(self.repo)}))
            tool = build / 'native' / 'tools' / TOOL.name
        else:
            tool = self.repo / 'source' / 'backends' / 'hip' / 'tools' / TOOL.name
        tool.parent.mkdir(parents=True)
        shutil.copy2(TOOL, tool)
        return tool

    def run_tool(self, tool):
        return subprocess.run(
            [sys.executable, str(tool), '--binary', str(self.binary),
             '--output', str(self.output), '--suite', 'S', '--ranks', '1',
             '--mpirun', str(self.launcher), '--min-available-gib', '0'],
            capture_output=True, text=True, check=False)

    def check_completed(self, tool):
        result = self.run_tool(tool)
        self.assertEqual(result.returncode, 0, result.stderr)
        records = json.loads((self.output / 'results.json').read_text())
        self.assertEqual([(r['case'], r['status']) for r in records], [('sample', 'completed')])
        self.assertEqual((self.output / 'DFT_DATA19').resolve(), self.repo / 'DFT_DATA19')
        for name in ('sample.dat', 'sample.out'):
            self.assertEqual((self.output / 'sample' / 'input_example' / name).resolve(),
                             self.inputs / name)

    def test_backend_source_tool_finds_checkout_inputs(self):
        self.check_completed(self.stage(external=False))

    def test_external_native_tool_uses_build_source_marker(self):
        self.check_completed(self.stage(external=True))

    def test_empty_suite_fails_before_creating_output(self):
        tool = self.stage(external=True)
        (self.inputs / 'sample.dat').unlink()
        result = self.run_tool(tool)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn('No test inputs found for suite S', result.stderr)
        self.assertFalse(self.output.exists())

    def test_missing_source_tree_fails_before_creating_output(self):
        tool = self.stage(external=True)
        shutil.rmtree(self.repo / 'DFT_DATA19')
        result = self.run_tool(tool)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn('lacks source/, work/, or DFT_DATA19/', result.stderr)
        self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)

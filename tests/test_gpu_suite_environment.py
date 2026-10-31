#!/usr/bin/env python3
"""Check the environment received by a suite's MPI launcher, without MPI/GPU."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class SuiteEnvironmentTest(unittest.TestCase):
    def launch(self, inherited, overrides=()):
        with tempfile.TemporaryDirectory(prefix='openmx-suite-env-') as tmp:
            repo = Path(tmp)
            (repo/'tools').mkdir()
            shutil.copy2(ROOT/'tools/run_gpu_suites.py', repo/'tools/run_gpu_suites.py')
            inputs = repo/'work/input_example'
            inputs.mkdir(parents=True)
            (inputs/'Probe.dat').write_text('System.Name Probe\n')
            (inputs/'Probe.out').write_text('Reference unused by the launch stub.\n')
            binary = repo/'binary'
            binary.write_text('stub, never executed\n')
            capture = repo/'received.json'
            launcher = repo/'mpi-stub'
            launcher.write_text('#!'+sys.executable+'\nimport json,os,sys\n'
                'from pathlib import Path\n'
                'Path(os.environ["OMX_ENV_CAPTURE"]).write_text(json.dumps({k:os.environ.get(k) '
                'for k in ("OMP_NUM_THREADS","BLIS_NUM_THREADS","OPENBLAS_NUM_THREADS")}))\n'
                'sys.exit(17)\n')
            launcher.chmod(0o755)
            env = os.environ.copy()
            for key in ('OMP_NUM_THREADS', 'BLIS_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
                env.pop(key, None)
            env.update(inherited, OMX_ENV_CAPTURE=str(capture))
            command = [sys.executable, str(repo/'tools/run_gpu_suites.py'),
                '--binary', str(binary), '--output', str(repo/'result'),
                '--suite', 'S', '--cases', 'Probe', '--together', '--threads', '3',
                '--ranks', '1', '--mpirun', str(launcher), '--min-available-gib', '0']
            for assignment in overrides:
                command += ['--env', assignment]
            result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 1, result.stdout+result.stderr)
            return json.loads(capture.read_text())

    def test_preserves_explicit_blas_settings(self):
        self.assertEqual(self.launch(dict(OMP_NUM_THREADS='8', BLIS_NUM_THREADS='3',
                                          OPENBLAS_NUM_THREADS='2')),
            dict(OMP_NUM_THREADS='3', BLIS_NUM_THREADS='3', OPENBLAS_NUM_THREADS='2'))

    def test_explicit_cli_override_wins(self):
        self.assertEqual(self.launch(dict(BLIS_NUM_THREADS='3'), ['BLIS_NUM_THREADS=2'])['BLIS_NUM_THREADS'], '2')

    def test_default_blas_setting_remains_one(self):
        self.assertEqual(self.launch({}),
            dict(OMP_NUM_THREADS='3', BLIS_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'))


if __name__ == '__main__':
    unittest.main()

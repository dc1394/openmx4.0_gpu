#!/usr/bin/env python3
"""Exercise the OTF runner on CPU with a synthetic launcher and real output fixtures.

No MPI or GPU job is launched.  Checks preparation/reuse, all three environment
variants, saved-metadata collection, and preservation of previous run data.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / 'tools/run_density_otf_ab.py'


def invoke(*args, expected=0, env=None):
    result = subprocess.run([sys.executable, str(RUNNER), *map(str, args)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            universal_newlines=True, env=env)
    if result.returncode != expected:
        raise AssertionError('Unexpected runner exit:\n' + result.stdout + result.stderr)
    return result


with tempfile.TemporaryDirectory(prefix='openmx-otf-runner-') as tmp:
    folder = Path(tmp)
    output = folder / 'run'
    launcher = folder / 'synthetic-launcher'
    launcher.write_text('#!' + sys.executable + '\n' + r'''
import os
from pathlib import Path
import sys
variant = Path.cwd().name
expected = 'fp64' if variant == 'openacc' else None
assert os.getenv('OPENMX_ORBS_EVAL_PRECISION') == expected, 'orbital environment not isolated'
expected = None if variant == 'baseline_auto' else ('0' if variant == 'openacc' else '1')
assert os.getenv('OPENMX_SETDEN_FP64_CUDA') == expected, 'density environment not isolated'
assert os.getenv('OPENMX_GRID_PRECISION') == 'fp64'
Path('Pt63.out').write_bytes(Path(REFERENCE).read_bytes())
precision = 'double-float' if variant == 'baseline_auto' else 'FP64'
print('<Set_Hamiltonian> GPU on-the-fly orbital tiles: 2 rank(s) without resident tables, '
      '128.0 MiB workspace/rank on device 0, ' + precision + ' evaluation')
ranks = int(sys.argv[sys.argv.index('-np') + 1])
for step in range(1, 9):
    print(f'******************* MD=1 SCF={step} *******************')
    for rank in reversed(range(ranks)):
        print(f'SETHOTFDENPROF id={rank} batches=3 workspace_mib=128.0 mode=0 '
              'pack=0.001 eval=0.002 device=0.003 total=0.006')
print('The calculation was normally finished.')
'''.replace('REFERENCE', repr(str(ROOT / 'work/large_example/Pt63.out'))))
    launcher.chmod(0o755)
    invoke('--prepare-only', '--output', output)
    assert len(list((output / 'inputs').glob('*.dat'))) == 3
    invoke('--prepare-only', '--output', output)
    env = os.environ.copy()
    env['OPENMX_ORBS_EVAL_PRECISION'] = 'inherited-must-be-removed'
    run_args = ['--binary', '/bin/true', '--baseline', '/bin/true', '--mpirun', launcher,
                '--ranks', '2', '--cases', 'Pt63', '--output', output]
    invoke(*run_args, env=env)
    result = json.loads((output / 'results.json').read_text())
    assert result['passed'] and len(result['comparisons']) == 2
    assert set(result['runs']['Pt63']) == {'baseline_auto', 'openacc', 'cuda'}
    metadata = json.loads((output / 'command.json').read_text())
    for variant in ['cuda', 'baseline_auto']:
        assert 'OPENMX_ORBS_EVAL_PRECISION' not in metadata['variants'][variant]['environment']
    assert metadata['variants']['openacc']['environment']['OPENMX_ORBS_EVAL_PRECISION'] == 'fp64'
    # Infer the two ranks and selected single case from saved metadata.
    invoke('--collect-only', '--output', output)
    before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in output.rglob('*') if path.is_file()}
    invoke(*run_args, expected=2, env=env)
    after = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in output.rglob('*') if path.is_file()}
    assert before == after, 'refused rerun changed previous results'
    # Missing MPI profile coverage must fail numerical/performance validation.
    log = output / 'Pt63/cuda/run.log'
    log.write_text(log.read_text().replace('SETHOTFDENPROF id=1', 'SETHOTFDENPROF id=9'))
    invoke('--collect-only', '--output', output, expected=1)
    assert not json.loads((output / 'results.json').read_text())['passed']
    # A manually modified prepared input is also retained, not overwritten.
    prepared = folder / 'modified'
    invoke('--prepare-only', '--output', prepared)
    input_path = prepared / 'inputs/Pt63.dat'
    input_path.write_text('user-edited input\n')
    invoke('--prepare-only', '--output', prepared, expected=2)
    assert input_path.read_text() == 'user-edited input\n'
print('PASS: CPU synthetic OTF runner, three variants, prepare/run/collect, metadata and output preservation')

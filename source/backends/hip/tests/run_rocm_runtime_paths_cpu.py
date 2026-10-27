#!/usr/bin/env python3
"""Check HIP Makefile runtime selection without running GPU code or MPI."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
MAKEFILE = ROOT / 'source' / 'Makefile' if (ROOT / 'source' / 'Makefile').is_file() else ROOT / 'Makefile'
ROCM = Path(os.environ.get('ROCM', '/opt/rocm-7.2.0'))
LLVM = Path(os.environ.get('ROCM_LLVM', str(ROCM / 'llvm')))
CLANG = Path(os.environ.get('CLANG', str(LLVM / 'bin' / 'clang')))
LLVM_LIB = Path(os.environ.get('ROCM_LLVM_LIB', str(LLVM / 'lib')))
MAKE = shutil.which('make')
GCC = shutil.which('gcc')
AVAILABLE = (MAKE and GCC and CLANG.is_file() and (LLVM_LIB / 'libomptarget.so').is_file())


@unittest.skipUnless(AVAILABLE, 'ROCm clang/OpenMP runtime, GNU make and GCC are required')
class RocmRuntimePaths(unittest.TestCase):
    def test_compute_node_runtime_layout(self):
        # The legacy llvm/lib path exists on the login node but can be
        # absent on a compute node. Its lib/llvm/lib runtime must still win
        # over a host-library directory containing the same SONAME.
        with tempfile.TemporaryDirectory(prefix='openmx-rocm-layout-') as name:
            directory = Path(name)
            rocm = directory / 'rocm'
            runtime = rocm / 'lib/llvm/lib'
            host = directory / 'host'
            for folder, value in ((runtime, 0), (host, 9)):
                folder.mkdir(parents=True)
                source = folder / 'runtime.c'
                source.write_text(f'int runtime_version(void) {{ return {value}; }}\n')
                subprocess.run([GCC, '-shared', '-fPIC', str(source),
                                '-Wl,-soname,libopenmx_runtime_probe.so',
                                '-o', str(folder / 'libopenmx_runtime_probe.so')], check=True)
            (directory / 'probe.c').write_text(
                'extern int runtime_version(void);\n'
                'int main(void) { return runtime_version(); }\n')
            rule = ('runtime-layout-probe:\n'
                    f'\t{GCC} probe.c -L"{runtime}" -lopenmx_runtime_probe '
                    '$(RPATHS) -o probe')
            linked = subprocess.run(
                [MAKE, '--no-print-directory', '-C', str(directory),
                 '-I', str(MAKEFILE.parent), '-f', str(MAKEFILE),
                 '--eval', rule, 'runtime-layout-probe', 'ARCH=gfx942',
                 f'ROCM={rocm}', f'ROCM_LLVM_LIB={rocm / "llvm/lib"}',
                 f'AOCL_LIB={host}', f'MPI_LIBDIR={host}', f'AOCC_LIB={host}',
                 f'CLANG={CLANG}', 'MPI_CFLAGS=', 'MPI_FCFLAGS='],
                capture_output=True, text=True, check=False)
            self.assertEqual(linked.returncode, 0, linked.stdout + linked.stderr)
            executed = subprocess.run([str(directory / 'probe')],
                                      capture_output=True, text=True, check=False)
            self.assertEqual(executed.returncode, 0, executed.stderr)

    def test_ambient_runtime_does_not_shadow_rocm(self):
        with tempfile.TemporaryDirectory(prefix='openmx-rocm-runtime-') as name:
            directory = Path(name)
            old_runtime = directory / 'old-runtime'
            old_runtime.mkdir()
            stub = old_runtime / 'stub.c'
            stub.write_text('int old_runtime_without_ompx_api(void) { return 0; }\n')
            built = subprocess.run(
                [GCC, '-shared', '-fPIC', str(stub), '-o', str(old_runtime / 'libomptarget.so')],
                capture_output=True, text=True, check=False)
            self.assertEqual(built.returncode, 0, built.stderr)
            # The link must resolve this ROCm-specific extension. The probe is
            # never executed, so even its conditional call cannot start a GPU.
            (directory / 'probe.c').write_text(
                'extern int ompx_get_team_procs(int);\n'
                'int main(int argc, char **argv) {\n'
                '  return argc > 100 ? ompx_get_team_procs(0) : 0;\n'
                '}\n')
            rule = ('runtime-path-probe:\n'
                    '\t$(CLANG) -fopenmp -L"$(ROCM_LLVM_LIB)" probe.c '
                    '-lomptarget -Wl,--trace -o probe')
            # An explicit architecture prevents Makefile GPU autodetection.
            # Import the actual production configuration, but build only the
            # host probe rather than any OpenMX or dependency target.
            linked = subprocess.run(
                [MAKE, '--no-print-directory', '-C', str(directory),
                 '-I', str(MAKEFILE.parent), '-f', str(MAKEFILE),
                 '--eval', rule, 'runtime-path-probe', 'ARCH=gfx942',
                 f'ROCM={ROCM}', f'ROCM_LLVM={LLVM}', f'ROCM_LLVM_LIB={LLVM_LIB}',
                 f'CLANG={CLANG}', 'MPI_CFLAGS=', 'MPI_FCFLAGS='],
                env={**os.environ, 'LIBRARY_PATH': str(old_runtime)},
                capture_output=True, text=True, check=False)
            self.assertEqual(linked.returncode, 0, linked.stdout + linked.stderr)
            self.assertTrue((directory / 'probe').is_file())
            self.assertIn(str(LLVM_LIB / 'libomptarget.so'), linked.stdout)
            self.assertNotIn(str(old_runtime / 'libomptarget.so'), linked.stdout)


if __name__ == '__main__':
    unittest.main(verbosity=2)

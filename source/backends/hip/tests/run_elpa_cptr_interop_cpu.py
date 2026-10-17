#!/usr/bin/env python3
"""Check the builder's ELPA C_PTR workaround with real Flang and C mocks.

Only the production Fortran allocation wrappers and their interfaces are
compiled. C malloc supplies the buffers; HIP, GPU kernels and MPI are not used.
"""
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile


ROOT = Path(__file__).resolve().parents[1]
DEPS = ROOT / 'source' / 'third_party'
if not (DEPS / 'build_gpusolver2_stack.sh').is_file():
    DEPS = ROOT.parents[1] / 'third_party' / 'hip'
DIST = DEPS / 'dist' if (DEPS / 'dist').is_dir() else DEPS.parent / 'dist'
BUILDER = DEPS / 'build_gpusolver2_stack.sh'
ROCM = Path(os.environ.get('ROCM', '/opt/rocm-7.2.0'))
LLVM = Path(os.environ.get('ROCM_LLVM', str(ROCM / 'llvm')))
FLANG = shlex.split(os.environ.get('FLANG', str(LLVM / 'bin' / 'flang')))
CC = shlex.split(os.environ.get('CC', 'cc'))
AVAILABLE = FLANG and CC and shutil.which(FLANG[0]) and shutil.which(CC[0])


def routine(text, name):
    match = re.search(r'^\s*function\s+' + re.escape(name) +
                      r'\s*\(.*?^\s*end\s+function[^\n]*', text, re.M | re.S | re.I)
    if match is None:
        raise ValueError('Missing production routine ' + name)
    return match[0]


def fortran_source(template):
    interfaces = '\n'.join(routine(template, name) for name in (
        'hip_malloc_intptr_c', 'hip_malloc_cptr_c',
        'hip_malloc_host_intptr_c', 'hip_malloc_host_cptr_c'))
    # Keep intptr wrappers before cptr wrappers, as in production. Their
    # shared C binding is what exposes the affected Flang lowering.
    wrappers = '\n'.join(routine(template, name) for name in (
        'hip_malloc_intptr', 'hip_malloc_cptr', 'hip_malloc_host_intptr', 'hip_malloc_host_cptr'))
    return ('module elpa_allocation_fixture\n'
            '  use, intrinsic :: iso_c_binding\n'
            '  implicit none\n  interface\n' + interfaces + '\n'
            '    function fixture_release(p) bind(C) result(ok)\n'
            '      import c_ptr, c_int\n'
            '      type(c_ptr), value :: p\n'
            '      integer(c_int) :: ok\n'
            '    end function\n'
            '    function fixture_complete() bind(C) result(ok)\n'
            '      import c_int\n'
            '      integer(c_int) :: ok\n'
            '    end function\n'
            '  end interface\ncontains\n' + wrappers + '\nend module\n')


PROGRAM = r'''
program check_elpa_allocations
  use elpa_allocation_fixture
  implicit none
  type(c_ptr) :: p
  real(c_double), pointer :: buffer(:)
  integer(c_intptr_t) :: bytes
  integer :: kind, scale
  logical :: success
  do scale = 1, 2
    bytes = int(scale, c_intptr_t) * 262144_c_intptr_t
    do kind = 1, 2
      p = c_null_ptr
      if (kind == 1) then
        success = hip_malloc_cptr(p, bytes)
      else
        success = hip_malloc_host_cptr(p, bytes)
      end if
      if (.not. success) stop 1
      if (.not. c_associated(p)) stop 2
      call c_f_pointer(p, buffer, [int(bytes / 8)])
      if (buffer(1) /= real(kind, c_double)) stop 3
      if (buffer(size(buffer)) /= real(bytes, c_double)) stop 4
      buffer(1) = 42_c_double
      if (fixture_release(p) /= 1) stop 5
      nullify(buffer)
    end do
  end do
  if (fixture_complete() /= 1) stop 6
  print *, 'PASS: ELPA host/device C_PTR outputs, sizes and buffer contents'
end program
'''

C_MOCK = r'''
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
static void *active;
static int count[2];
static int allocate(intptr_t *output, size_t bytes, int kind) {
    if (!output) {
        fputs("ELPA_INTEROP_NULL_OUTPUT_SLOT\n", stderr);
        return 0;
    }
    if (active || (bytes != 262144 && bytes != 524288)) return 0;
    double *data = malloc(bytes);
    if (!data) return 0;
    data[0] = kind;
    data[bytes / sizeof(double) - 1] = (double)bytes;
    *output = (intptr_t)data;
    active = data;
    ++count[kind - 1];
    return 1;
}
int hipMallocFromC(intptr_t *output, size_t bytes) { return allocate(output, bytes, 1); }
int hipMallocHostFromC(intptr_t *output, size_t bytes) { return allocate(output, bytes, 2); }
int fixture_release(void *pointer) {
    if (pointer != active || !pointer || *(double *)pointer != 42.0) return 0;
    free(active); active = NULL; return 1;
}
int fixture_complete(void) { return !active && count[0] == 2 && count[1] == 2; }
'''


@unittest.skipUnless(AVAILABLE, 'ROCm Flang and a host C compiler are required')
class ElpaCptrInterop(unittest.TestCase):
    def test_real_builder_patch_repairs_host_and_device_output_slots(self):
        builder = BUILDER.read_text()
        block = re.search(r"^\s*python3 - [^\n]*<<'OPENMX_ELPA_CPTR_PATCH'\n(.*?)^OPENMX_ELPA_CPTR_PATCH\s*$",
                          builder, re.M | re.S)
        self.assertIsNotNone(block, 'Builder must expose the production ELPA pointer patch')
        version = re.search(r'GPUSOLVER2_ELPA_VER:=[\"]?([0-9.]+)', builder)[1].replace('.', '_')
        archive = DIST / ('elpa-new_release_' + version + '.zip')
        with zipfile.ZipFile(archive) as bundled:
            names = [name for name in bundled.namelist() if name.endswith('/src/GPU/ROCm/hip_template.F90')]
            self.assertEqual(len(names), 1)
            original = bundled.read(names[0]).decode()
        with tempfile.TemporaryDirectory(prefix='openmx-elpa-cptr-') as name:
            folder = Path(name)
            template = folder / 'hip_template.F90'
            template.write_text(original)
            patch = folder / 'apply-builder-patch.py'
            patch.write_text(block[1])
            env = {key:value for key,value in os.environ.items() if key not in (
                'CPATH','C_INCLUDE_PATH','CPLUS_INCLUDE_PATH','LIBRARY_PATH','LD_LIBRARY_PATH')}
            env['LD_LIBRARY_PATH'] = str(LLVM / 'lib')
            for _ in range(2):
                applied = subprocess.run([sys.executable, str(patch), str(template)],
                                         capture_output=True, text=True, env=env, check=False)
                self.assertEqual(applied.returncode, 0, applied.stdout + applied.stderr)
                if _ == 0: patched = template.read_text()
                else: self.assertEqual(template.read_text(), patched, 'Patch must be idempotent')
            self.assertNotEqual(patched, original)
            (folder / 'mock.c').write_text(C_MOCK)
            built = subprocess.run(CC + ['-std=c11','-O2','-c',str(folder/'mock.c'),'-o',str(folder/'mock.o')],
                                   capture_output=True, text=True, env=env, check=False)
            self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
            # Observe the unfixed ABI too. Old affected Flang reports the null
            # slot; newer compilers may already implement that binding correctly.
            for label, source in (('original', original), ('patched', patched)):
                (folder / 'fixture.F90').write_text(fortran_source(source))
                # ELPA compiles its module separately from callers. Preserve
                # that boundary: same-unit optimization can hide the bad ABI.
                (folder / 'main.F90').write_text(PROGRAM)
                for command in (
                    FLANG + ['-O2','-DWITH_AMD_GPU_VERSION','-c',str(folder/'fixture.F90'),'-o',str(folder/'fixture.o')],
                    FLANG + ['-O2',str(folder/'main.F90'),str(folder/'fixture.o'),str(folder/'mock.o'),'-o',str(folder/label)],
                ):
                    built = subprocess.run(command,cwd=folder,capture_output=True,text=True,env=env,check=False)
                    self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
                ran = subprocess.run([str(folder/label)],cwd=folder,capture_output=True,text=True,env=env,check=False)
                if label == 'original':
                    print('Unpatched Flang C_PTR diagnostic:',
                          'null output slot reproduced' if 'ELPA_INTEROP_NULL_OUTPUT_SLOT' in ran.stderr else
                          f'returncode={ran.returncode}', flush=True)
                else:
                    self.assertEqual(ran.returncode, 0, ran.stdout + ran.stderr)
                    self.assertIn('PASS: ELPA host/device C_PTR outputs', ran.stdout)
                    print(ran.stdout.strip(), flush=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)

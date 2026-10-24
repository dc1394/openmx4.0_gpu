#!/usr/bin/env python3
"""Check SDK-free CPU source selection and the real accelerator ABI boundary."""
import importlib.util
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('cpu_adapter', ROOT / 'tools/prepare_cpu_adapter.py')
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


class CPUBuildTest(unittest.TestCase):
    def test_sources_share_the_application_and_exclude_utilities(self):
        manifest = adapter.application_sources(ROOT / 'source')
        sources = {Path(p).name for p in manifest['sources']}
        self.assertTrue({'openmx.c', 'Input_std.c', 'Set_Hamiltonian.c', 'Set_Density_Grid.c',
                         'Force.c', 'openmx_grid_assembly.c', 'openmx_elpa_cpu.c'} <= sources)
        self.assertFalse({'cohp.c', 'check_lead.c', 'kSpin.c', 'DosMain.c',
                          'set_cuda_default_device_from_local_rank.c', 'elpa_cosma_memory.cpp'} & sources)
        self.assertFalse(any(p.endswith('.cu') for p in manifest['sources']))
        self.assertEqual(len(manifest['sources']), len(set(manifest['sources'])))

    @unittest.skipUnless(shutil.which('c++'), 'ordinary C++ compiler required')
    def test_device_operations_decline_without_changing_cpu_output(self):
        manifest = adapter.application_sources(ROOT / 'source')
        hamiltonian = (ROOT / 'source/Set_Hamiltonian.c').read_text()
        start = hamiltonian.index('int Set_Hamiltonian_Cuda_MatrixElements(')
        declaration = hamiltonian[start:hamiltonian.index(';', start) + 1]
        program = '''
#include "openmx_gpu_transform.h"
#include "openmx_gpu_dm.h"
#include "openmx_mgpu_runtime.h"
#include "openmx_gpu_blas.h"
#include <cassert>
typedef float Type_Orbs_Grid;
extern "C" {
''' + declaration + '''
void *Set_Hamiltonian_Cuda_StreamCreate(size_t,size_t,const double*);
}
template<class... A> int decline(int(*function)(A...)) { return function(A{}...); }
int main() {
  int devices=7;assert(cudaGetDeviceCount(&devices)==cudaSuccess && devices==0);
  void *p=reinterpret_cast<void*>(1);
  assert(cudaMalloc(&p,1024)!=cudaSuccess && p==nullptr);
  assert(decline(&Set_Hamiltonian_Cuda_MatrixElements)==1);
  assert(Set_Hamiltonian_Cuda_StreamCreate(1024,0,nullptr)==nullptr);
  double output[]={123,456,789};
  assert(openmx_gpu_transform(OPENMX_GPU_ZERO,0,1,1,nullptr,1,nullptr,1,output,1,0,0)==0);
  assert(output[0]==123 && output[1]==456 && output[2]==789);
  assert(openmx_mgpu_pool_devices(nullptr,0)==0);
  assert(openmx_mgpu_band_used_gpus(7)==0);
  assert(openmx_mgpu_admit_devices(nullptr,0)==nullptr);
  cublasHandle_t handle=nullptr;assert(cublasCreate(&handle)!=CUBLAS_STATUS_SUCCESS);
}
'''
        with tempfile.TemporaryDirectory(prefix='openmx-cpu-abi-') as name:
            folder = Path(name)
            (folder / 'adapter.cpp').write_text(adapter.prepare([Path(p) for p in manifest['kernels']]))
            (folder / 'check.cpp').write_text(program)
            command = [shutil.which('c++'), '-std=c++20', '-O2', '-DOPENMX_GPU_BACKEND=0',
                       '-I' + str(ROOT / 'source/backends/cpu/include'), '-I' + str(ROOT / 'source'),
                       str(folder / 'adapter.cpp'), str(folder / 'check.cpp'), '-o', str(folder / 'check')]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            subprocess.run([str(folder / 'check')], check=True)


if __name__ == '__main__':
    unittest.main()

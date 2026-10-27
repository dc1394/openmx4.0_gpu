#!/usr/bin/env python3
"""Exercise complete orbital cache lifetimes with the real C++ calculation owner.

CUDA control/upload code is extracted unchanged; HIP upload and kernel code is
compiled whole by the existing spline/reference fixture. SDK calls are modeled.
"""
import argparse
import os
from pathlib import Path
import re
import subprocess
import tempfile
from run_cluster_col_state_cpu import function
from native_state_test_support import (compiler_command, bounded_stack,
    allocation_fault_source, allocation_fault_environment)
ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core-library',type=Path,required=True)
    parser.add_argument('--backend',choices=['cuda','hip','both'],default='both')
    args=parser.parse_args();library=args.core_library.resolve(strict=True)
    if args.backend in ['cuda','both']:
        source=(ROOT/'source/Set_Orbitals_Grid.c').read_text()
        struct=source[source.index('struct OpenMXOrbitalGpuState {'):]
        struct=struct[:struct.index('\n};')+3]
        helpers=source[source.index('typedef struct { float hi, lo; } SOG_df;'):
            source.index('#pragma acc routine seq\nstatic void SOG_point_eval_df')]
        names=['SOG_cache_device','SOG_eval_double_float','SOG_arena_try','SOG_Device_Release',
            'SOG_Device_ReleaseAll','SOG_Device_Prepare','SOG_Device_Tables','SOG_Device_Bytes',
            'SOG_Device_EvalPrecisionName','Set_Orbitals_Grid_GPU_Release','SOG_Reset',
            'Set_Orbitals_Grid_GPU_Bind_Resources']
        bodies=[function(source,n)for n in names]
        prototypes='\n'.join(b[:b.index('{')].strip()+';'for b in bodies)
        maximum=re.search(r'^#define OPENMX_GPU_WORKERS_MAX \d+',
            (ROOT/'source/openmx_gpu_workers.h').read_text(),re.M)[0]
        metadata=maximum+'\n#define SOG_L0MAX 3\n'+struct+'\n'+prototypes+'''
#define SOG_dev orbital_state->devices[SOG_cache_device()]
#define SOG_dev_fp32_enabled orbital_state->fp32_modes[SOG_cache_device()]
'''
        with tempfile.TemporaryDirectory(prefix='openmx-orbital-state-')as directory:
            folder=Path(directory);chunks=[]
            for n,b in zip(names,bodies):
                if n=='SOG_Device_Prepare':b='#define malloc host_malloc\n#define free host_free\n'+b+'\n#undef malloc\n#undef free\n'
                chunks.append(b)
            (folder/'orbital_state.inc').write_text(metadata+'\n'+helpers+'\n'+'\n'.join(chunks))
            exe=folder/'check'
            command=compiler_command('c++')+[
                '-DOPENMX_GPU_HIP=0','-Wno-unknown-pragmas','-I'+str(folder),
                '-I'+str(ROOT/'source'),'-I'+str(ROOT/'core/include'),
                '-I'+str(ROOT/'core/tests'),'-I'+str(ROOT/'core/tests/fixtures/native_resources'),
                str(ROOT/'tests/orbital_state_harness.cpp'),allocation_fault_source(folder),
                str(ROOT/'source/openmx_env.c'),str(ROOT/'source/openmx_precision.c'),
                str(library),'-Wl,-rpath,'+str(library.parent),'-o',str(exe)]
            subprocess.run(command,check=True)
            environment=dict(os.environ,**allocation_fault_environment(folder))
            subprocess.run([str(exe)],check=True,env=environment,timeout=90,preexec_fn=bounded_stack)
    if args.backend in ['hip','both']:
        subprocess.run(['python3',str(ROOT/'source/backends/hip/tests/run_orbs_grid_boundaries.py'),
            '--core-library',str(library)],check=True,timeout=120,preexec_fn=bounded_stack)


if __name__=='__main__':main()

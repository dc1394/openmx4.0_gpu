#!/usr/bin/env python3
"""Exercise production potential-mixing ownership with the actual shared core.

Compile the actual state/cache, logging, entry, allocation and inverse bodies.
The large numerical callback and MPI/LAPACK/eigensolver calls are modeled;
independent inverse identities verify the native layouts and library contract.
"""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile
from native_state_test_support import (bounded_stack, compiler_command,
                                       allocation_fault_source, allocation_fault_environment)
from run_cluster_col_state_cpu import function
from run_mixing_rhok_state_cpu import block

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core-library', type=Path, required=True)
    args = parser.parse_args()
    library = args.core_library.resolve(strict=True)
    text = (ROOT / 'source/Mixing_V.c').read_text()
    names = ['MixV_Fast_Enabled', 'MixVKSk_Fast_Invalidate', 'MixVKSk_FastResize',
             'MixVKSk_RotateSlots', 'MixV_Reset', 'Mixing_V_Bind_Resources',
             'Inverse', 'Complex_Inverse', 'MixV_ScratchOperation', 'Mixing_V']
    bodies = [function(text, name) for name in names]
    generated = '\n'.join(body[:body.index('{')].rstrip() + ';' for body in bodies)
    generated += '\n' + '\n'.join(bodies)
    for namespace, method in [('unpivoted', 0), ('regularized', 2)]:
        inverse = function(text, 'Inverse').replace('int method_flag=1;', 'int method_flag=' + str(method) + ';')
        generated += '\nnamespace ' + namespace + ' {\n' + inverse + '\n}\n'
    diis = function(text, 'DIIS_Mixing_VKSk')
    begin = diis.index('int use_fast = MixV_Fast_Enabled();')
    end = diis.index('if (incremental){', begin)
    generated += '\nstatic int preflight(MixingVState *state,int SCF_iter,int NumMix,int spinmax) {\n'
    generated += diis[begin:end] + '\nreturn incremental;\n}\n'
    for name, field in [('DIIS_Mixing_VKSk', 'diis_first'), ('Kerker_Mixing_VKSk', 'kerker_first')]:
        body = function(text, name)
        begin = body.index('if (state->' + field + ')')
        if field == 'diis_first':
            selected = block(body, begin)
        else:
            end = body.index('state->' + field + '=0;', begin) + len('state->' + field + '=0;')
            selected = body[begin:end]
        generated += '\nstatic void log_' + field + '(MixingVState *state) {\n'
        generated += 'const int size_Re_OptVKSk=8,size_Kerker_weight=4;\n' + selected + '\n}\n'
    begin = diis.index('alden = OpenMX_Solver_OverwriteReal(')
    end = diis.index(';', diis.index('Kerker_weight = OpenMX_Solver_OverwriteReal(', begin)) + 1
    generated += '\nstatic int buffers(OpenMXSolverInvocation *scratch,void*,double *result) {\n'
    generated += 'const int spinmax=3,My_Max_NumGridB=17,My_NumGridB_CB=2;\n'
    generated += 'double *alden,*alden0,*gradF,*My_NMat,*NMat,*Kerker_weight;\n'
    generated += 'int size_Re_OptVKSk,size_Kerker_weight;\n'
    generated += 'OpenMXSolverRealTensor A,IA,Re_OptVKSk,Im_OptVKSk;\n'
    generated += diis[begin:end] + '\n'
    generated += '''check(A.extent[0]==7 && A.extent[1]==7 && IA.extent[0]==7,"Gram layout changed");
check(Re_OptVKSk.extent[0]==3 && Re_OptVKSk.extent[1]==17 && Im_OptVKSk.extent[1]==17,"inverse FFT lost real-space capacity");
for(int spin=0;spin<3;++spin)for(int k=0;k<17;++k){
openmx_solver_real_matrix_row(&Re_OptVKSk,spin)[k]=100*spin+k+.25;
openmx_solver_real_matrix_row(&Im_OptVKSk,spin)[k]=-100*spin-k-.5;}
alden[6]=1;alden0[6]=2;gradF[6]=3;My_NMat[48]=4;NMat[48]=5;Kerker_weight[1]=6;
check(openmx_solver_real_matrix_row(&Re_OptVKSk,2)[16]==216.25 && openmx_solver_real_matrix_row(&Im_OptVKSk,2)[16]==-216.5,"FFT row stride changed");
*result=23.5;return 1;}\n'''
    with tempfile.TemporaryDirectory(prefix='openmx-mixing-v-workspace-') as directory:
        folder = Path(directory)
        (folder / 'mpi.h').write_text('#pragma once\ntypedef int MPI_Comm;\n')
        (folder / 'omp.h').write_text('#pragma once\nstatic int omp_get_thread_num(){return 0;}\nstatic int omp_get_max_threads(){return 2;}\n')
        (folder / 'mixing_v_workspace.inc').write_text(generated)
        executable = folder / 'check'
        command = compiler_command('c++') + [
            '-I' + str(folder), '-I' + str(ROOT / 'source'), '-I' + str(ROOT / 'core/include'),
            '-I' + str(ROOT / 'core/tests'), str(ROOT / 'tests/mixing_v_workspace_harness.cpp'),
            allocation_fault_source(folder), str(ROOT / 'source/openmx_env.c'), str(library),
            '-Wl,-rpath,' + str(library.parent), '-o', str(executable)]
        subprocess.run(command, check=True)
        environment = {k: v for k, v in os.environ.items() if not k.startswith(('OPENMX_', 'GEMMUL8_'))}
        environment.update(allocation_fault_environment(folder))
        subprocess.run([str(executable)], env=environment, check=True,
                       timeout=120, preexec_fn=bounded_stack)


if __name__ == '__main__':
    main()

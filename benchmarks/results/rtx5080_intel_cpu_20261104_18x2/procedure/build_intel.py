#!/usr/bin/env python3
import hashlib,json,os,shutil,subprocess,time
from pathlib import Path
A=Path(__file__).resolve().parent
ROOT=Path('/home/dc1394/openmx4.0_gpu')
INTEL=Path('/home/dc1394/intel/oneapi')
COMP=INTEL/'compiler/2026.0'
MPI=INTEL/'mpi/2021.18'
MKL=INTEL/'mkl/2026.0'
def save(name,data):
    (A/name).write_text(json.dumps(data,indent=2)+'\n')
def main():
    env={k:v for k,v in os.environ.items() if not k.startswith(('OPENMX_','GEMMUL8_','OMP_','KMP_','I_MPI_','MKL_','CUDA_','OMPI_','PMIX_','PRTE_'))
         and k not in ['CC','CXX','FC','F77','CFLAGS','CXXFLAGS','FCFLAGS','FFLAGS','CPPFLAGS','LDFLAGS','LIBS','CPATH','C_INCLUDE_PATH','CPLUS_INCLUDE_PATH','LIBRARY_PATH','LD_LIBRARY_PATH','LD_PRELOAD','LD_AUDIT']}
    env.update(PATH=str(COMP/'bin')+':'+str(MPI/'bin')+':'+env['PATH'],
               LD_LIBRARY_PATH=':'.join(str(p/'lib') for p in [COMP,MPI,MKL]),
               I_MPI_ROOT=str(MPI),MKLROOT=str(MKL),I_MPI_CC=str(COMP/'bin/icx'),
               I_MPI_CXX=str(COMP/'bin/icpx'),I_MPI_F90=str(COMP/'bin/ifx'),I_MPI_FC=str(COMP/'bin/ifx'))
    cmake=shutil.which('cmake')
    assert cmake
    kokkos=A/'kokkos-src'
    if not kokkos.exists():shutil.copytree(ROOT/'build/cuda/_deps/kokkos-src',kokkos)
    configure=[cmake,'-S',str(ROOT),'-B',str(A/'build'),'-DOPENMX_GPU_BACKEND=CPU',
               '-DOPENMX_CPU_COMPILER=ICX','-DOPENMX_CPU_MPI=INTEL','-DOPENMX_CPU_LINALG=MKL',
               '-DOPENMX_CPU_MKL_THREADING=INTEL','-DOPENMX_CPU_MPI_ROOT='+str(MPI),
               '-DOPENMX_CPU_MKL_ROOT='+str(MKL),'-DCMAKE_C_COMPILER='+str(COMP/'bin/icx'),
               '-DCMAKE_CXX_COMPILER='+str(COMP/'bin/icpx'),'-DCMAKE_Fortran_COMPILER='+str(COMP/'bin/ifx'),
               '-DOPENMX_HOST_SIMD_CXX_COMPILER='+str(COMP/'bin/icpx'),
               '-DOPENMX_CPU_FFTW_INCLUDE_DIR='+str(MKL/'include/fftw'),
               '-DOPENMX_CPU_FFTW_LIBRARY='+str(MKL/'lib/libmkl_intel_lp64.so'),
               '-DCMAKE_BUILD_TYPE=Release','-DCMAKE_C_FLAGS_RELEASE=-O3 -DNDEBUG -xHost',
               '-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG -xHost','-DCMAKE_Fortran_FLAGS_RELEASE=-O3 -DNDEBUG -xHost',
               '-DOPENMX_CORE_BACKEND=SERIAL','-DOPENMX_CORE_HOST_SIMD=ON','-DOPENMX_COPY_TO_WORK=OFF',
               '-DOPENMX_BUILD_JOBS=8','-DBUILD_TESTING=ON','-DFETCHCONTENT_SOURCE_DIR_KOKKOS='+str(kokkos),
               '-DCMAKE_INSTALL_PREFIX='+str(A/'install'),'-DCMAKE_EXPORT_COMPILE_COMMANDS=ON']
    commands=[('configure',configure),('build',[cmake,'--build',str(A/'build'),'--target','openmx','openmx_cpu_numerics','--parallel','8']),
              ('install',[cmake,'--install',str(A/'build')])]
    save('build-commands.json',dict(commands=dict(commands),environment={k:env[k] for k in ['PATH','LD_LIBRARY_PATH','I_MPI_ROOT','MKLROOT','I_MPI_CC','I_MPI_CXX','I_MPI_F90']}))
    state=dict(status='running',phases=[])
    for name,cmd in commands:
        state['active_phase']=name;save('build-state.json',state)
        start=time.monotonic()
        with (A/(name+'.log')).open('w') as output:
            r=subprocess.run(cmd,env=env,stdout=output,stderr=subprocess.STDOUT)
        state['phases'].append(dict(name=name,returncode=r.returncode,elapsed_seconds=time.monotonic()-start))
        if r.returncode:
            state['status']='failed';save('build-state.json',state);raise SystemExit(r.returncode)
    state.update(status='completed',active_phase=None);save('build-state.json',state)
    print(json.dumps(state),flush=True)
if __name__=='__main__':main()

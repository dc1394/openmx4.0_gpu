"""Link real C++ GPU workers to the C extraction harnesses' SDK stubs.

The C controller, numerical callbacks, packets and failure controls remain
in their original language. Only the SDK calls cross this model boundary;
the production worker and its owners are compiled unchanged as C++17.
"""
from pathlib import Path
import os
import shlex
import subprocess


def compile_task_test(command, **kwargs):
    return _compile_worker_test(command, 'openmx_gpu_task_worker', **kwargs)


def compile_vector_test(command, **kwargs):
    return _compile_worker_test(command, 'openmx_gpu_vector_worker', **kwargs)


def _compile_worker_test(command, worker, **kwargs):
    source=next(Path(arg) for arg in command if str(arg).endswith('.c'))
    folder=source.parent
    root=Path(__file__).resolve().parents[1]
    source.write_text(source.read_text()+r'''
int openmx_task_model_allocate(void **out,size_t bytes){return OpenMXGpuMalloc(out,bytes);}
int openmx_task_model_release(void *pointer){return OpenMXGpuFree(pointer);}
int openmx_task_model_copy(void *out,const void *in,size_t bytes,int kind){return OpenMXGpuMemcpy(out,in,bytes,kind);}
''')
    shim=folder/'task_sdk_model.hpp'
    shim.write_text(r'''
#pragma once
#define OPENMX_GPU_RUNTIME_H
#include <cstddef>
#include <cstdlib>
using OpenMXGpuError=int;
enum {OPENMX_GPU_SUCCESS=0,OPENMX_GPU_OUT_OF_MEMORY=2,OPENMX_GPU_INVALID_VALUE=11,
      OPENMX_GPU_COPY_HOST_TO_DEVICE=1,OPENMX_GPU_COPY_DEVICE_TO_HOST=2};
extern "C" int openmx_task_model_allocate(void**,size_t);
extern "C" int openmx_task_model_release(void*);
extern "C" int openmx_task_model_copy(void*,const void*,size_t,int);
inline int OpenMXGpuGetDevice(int *device){*device=0;return 0;}
inline int OpenMXGpuSetDevice(int){return 0;}
inline int OpenMXGpuMalloc(void **out,size_t n){return openmx_task_model_allocate(out,n);}
inline int OpenMXGpuFree(void *pointer){return openmx_task_model_release(pointer);}
inline int OpenMXGpuMemcpy(void *out,const void *in,size_t n,int kind){return openmx_task_model_copy(out,in,n,kind);}
''')
    c_object=folder/'task_controller_test.o'
    worker_object=folder/'task_worker_test.o'
    compile_c=list(command)
    compile_c[compile_c.index('-o')+1]=str(c_object)
    compile_c+=['-c']
    subprocess.run(compile_c,**kwargs)
    cxx=shlex.split(os.environ.get('CXX','c++'))
    sanitizer=[arg for arg in command if str(arg).startswith(('-fsanitize=','-fno-sanitize','-fno-pie','-no-pie'))]
    subprocess.run(cxx+['-std=c++17','-O1','-g',*sanitizer,'-I'+str(folder),'-I'+str(root/'source'),
        '-Wno-unknown-pragmas','-include',str(shim),'-c',str(root/'source'/(worker+'.cpp')),'-o',str(worker_object)],**kwargs)
    cc=shlex.split(os.environ.get('CC','cc'))
    env_object=folder/'task_env.o'
    env_archive=folder/'libtask_env.a'
    subprocess.run(cc+['-std=c11','-O1','-g',*sanitizer,'-I'+str(root/'source'),
        '-c',str(root/'source/openmx_env.c'),'-o',str(env_object)],**kwargs)
    # Extract the real readers only when needed; some harnesses supply their
    # own environment model. No duplicate definitions or weak overrides.
    subprocess.run(['ar','rcs',str(env_archive),str(env_object)],**kwargs)
    link=cxx+[str(arg) for arg in command[len(cc):] if not str(arg).startswith('-std=')]
    link[link.index(str(source))]=str(c_object)
    link+=[str(worker_object),str(env_archive)]
    subprocess.run(link,**kwargs)

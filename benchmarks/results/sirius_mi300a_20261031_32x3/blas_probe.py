#!/usr/bin/env python3
"""Record the loaded BLAS provider and OpenMP settings without computing."""
import ctypes
import json
import os

lib=ctypes.CDLL(None)
class Info(ctypes.Structure):
    _fields_=[('path',ctypes.c_char_p),('base',ctypes.c_void_p),
              ('symbol',ctypes.c_char_p),('address',ctypes.c_void_p)]

def provider(name):
    info=Info()
    lib.dladdr.argtypes=[ctypes.c_void_p,ctypes.POINTER(Info)]
    if not lib.dladdr(ctypes.cast(getattr(lib,name),ctypes.c_void_p),ctypes.byref(info)):
        raise RuntimeError('Cannot locate '+name)
    return os.path.realpath(info.path.decode())

result={}
for name in ['bli_info_get_enable_threading','bli_info_get_enable_openmp',
             'bli_info_get_enable_pthreads','bli_thread_get_num_threads']:
    f=getattr(lib,name);f.restype=ctypes.c_int64
    result[name]=f()
for name in ['omp_get_max_threads','omp_get_max_active_levels']:
    f=getattr(lib,name);f.restype=ctypes.c_int
    result[name]=f()
result.update(dgemm_provider=provider('dgemm_'),openmp_provider=provider('omp_get_max_threads'))
assert result['bli_info_get_enable_openmp']==1
assert result['bli_info_get_enable_pthreads']==0
assert result['bli_thread_get_num_threads']==3
assert result['omp_get_max_threads']==3
assert result['omp_get_max_active_levels']==1
assert 'libblis-mt.so' in result['dgemm_provider']
assert '/opt/rocm-7.2.0/' in result['openmp_provider']
print(json.dumps(result))

#!/usr/bin/env python3
"""Apply documented platform integration to official OpenMX + patch4.0.1."""
from pathlib import Path
import re

root = Path('/home/ec2-user/openmx-build')
src = Path('/home/ec2-user/openmx4.0/source')
prefix = Path('/home/ec2-user/.local/opt/openmx-4.0.1-sve256')
adapter = (root/'reference/source_openmx_elpa_cpu.c').read_text()
adapter = adapter.replace('#include "openmx_env.h"\n#include "openmx_stdout_level.h"',
    '/* Standalone integration with the official OpenMX stdout setting. */\n'
    'extern int level_stdout;\n#define OPENMX_STDOUT_LEVEL(n) (level_stdout >= (n))')
adapter = adapter.replace('openmx_env_at(is_complex ? OPENMX_ENV_EXT_ELPA_DEFAULT_complex_kernel\n                                 : OPENMX_ENV_EXT_ELPA_DEFAULT_real_kernel)',
    'getenv(is_complex ? "ELPA_DEFAULT_complex_kernel" : "ELPA_DEFAULT_real_kernel")')
adapter = adapter.replace('openmx_env_flag_at(OPENMX_ENV_GPU_VERBOSE, 0)',
    '(getenv("OPENMX_ELPA_VERBOSE") != NULL && strcmp(getenv("OPENMX_ELPA_VERBOSE"), "0") != 0)')
assert 'openmx_env_' not in adapter
(src/'openmx_elpa_cpu.c').write_text(adapter)
(src/'openmx_elpa_cpu.h').write_bytes((root/'reference/source_openmx_elpa_cpu.h').read_bytes())

header = (root/'reference/openmx_common.h.official').read_bytes()
assert header.count(b'static int NBLK=128;') == 1
(src/'openmx_common.h').write_bytes(header.replace(b'static int NBLK=128;', b'static int NBLK=32;'))
main = src/'openmx.c'
text = main.read_text()
old = '  MPI_Init(&argc,&argv);'
new = ('  /* ELPA OpenMP requires at least serialized MPI calls. */\n'
       '  int mpi_thread_provided;\n'
       '  MPI_Init_thread(&argc,&argv,MPI_THREAD_SERIALIZED,&mpi_thread_provided);\n'
       '  if (mpi_thread_provided < MPI_THREAD_SERIALIZED) {\n'
       '    fprintf(stderr,"MPI_THREAD_SERIALIZED is required by this ELPA build.\\n");\n'
       '    MPI_Abort(MPI_COMM_WORLD,1);\n  }')
if old in text:
    (root/'reference/openmx.c.official').write_text(text)
    main.write_text(text.replace(old,new))
inp=src/'Input_std.c'
text=inp.read_text()
old='input_string("DATA.PATH",DFT_DATA_PATH,"../DFT_DATA19");'
if old in text:
    (root/'reference/Input_std.c.official').write_text(text)
    inp.write_text(text.replace(old,f'input_string("DATA.PATH",DFT_DATA_PATH,"{prefix}/share/openmx/DFT_DATA19");'))

mk=(root/'reference/makefile.official').read_text()
start=mk.index('MKLROOT = ')
end=mk.index('\n \n',start)
flags='-O3 -mcpu=neoverse-v1 -msve-vector-bits=scalable -fno-fast-math -ffp-contract=off'
elpa=Path('/home/ec2-user/.local/opt/elpa-2026.02.002-sve256')
blas=Path('/home/ec2-user/.local/opt/openblas-0.3.34-sve256/lib')
scalapack=Path('/home/ec2-user/openmx-build/scalapack/lib/libscalapack.a')
config=(f'CC = /usr/lib64/openmpi/bin/mpicc {flags} -std=gnu11 -Dnosse -fopenmp -fcommon -Wno-implicit-function-declaration -I{elpa}/include/elpa_openmp-2026.02.002\n'
        f'FC = /usr/lib64/openmpi/bin/mpifort {flags} -fopenmp -fallow-argument-mismatch\n'
        f'LIB = {elpa}/lib/libelpa_openmp.a {scalapack} -L{blas} -lopenblas -lfftw3 -lmpi_mpifh -lgfortran -lpthread -lm -ldl -Wl,--disable-new-dtags -Wl,-rpath,{blas}:/usr/lib64/openmpi/lib')
mk=mk[:start]+config+mk[end:]
mk=mk.replace('elpa1.o solve_evp_real.o solve_evp_complex.o \\\n','openmx_elpa_cpu.o \\\n',1)
start=mk.index('LIBELPADIR = ')
end=mk.index('\n#\n# OpenMX',start)
mk=mk[:start]+'# Modern external ELPA replaces the bundled legacy objects.\n'+mk[end:]
mk+='\nopenmx_elpa_cpu.o: openmx_elpa_cpu.c openmx_elpa_cpu.h\n\t$(CC) -c openmx_elpa_cpu.c\n'
(src/'makefile').write_text(mk)
print('Prepared official OpenMX 4.0.1 for modern ELPA, NBLK=32, GCC14/SVE256.')

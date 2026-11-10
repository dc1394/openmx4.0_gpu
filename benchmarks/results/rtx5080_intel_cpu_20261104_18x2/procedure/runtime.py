import hashlib,json,os,re,time
from pathlib import Path
A=Path(__file__).resolve().parent
ROOT=Path('/home/dc1394/openmx4.0_gpu')
ONEAPI=Path('/home/dc1394/intel/oneapi')
COMP=ONEAPI/'compiler/2026.0';MPI=ONEAPI/'mpi/2021.18';MKL=ONEAPI/'mkl/2026.0'
MASKS=[format((1<<i)|(1<<(i+18)),'x') for i in range(18)]
def save(path,data):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def environment(variant='intel'):
    env={k:v for k,v in os.environ.items() if not k.startswith(('OPENMX_','GEMMUL8_','OMP_','KMP_','I_MPI_','MKL_','CUDA_','OMPI_','PMIX_','PRTE_','BLIS_','OPENBLAS_'))
         and k not in ['LD_LIBRARY_PATH','LD_PRELOAD','LD_AUDIT','CPATH','C_INCLUDE_PATH','CPLUS_INCLUDE_PATH','LIBRARY_PATH']}
    dirs=([A/'install/lib'] if variant=='intel' else [])+[COMP/'lib',MPI/'lib',MKL/'lib']
    env.update(PATH=str(MPI/'bin')+':'+str(COMP/'bin')+':'+env['PATH'],LD_LIBRARY_PATH=':'.join(map(str,dirs)),
      I_MPI_ROOT=str(MPI),I_MPI_CC=str(COMP/'bin/icx'),I_MPI_CXX=str(COMP/'bin/icpx'),I_MPI_F90=str(COMP/'bin/ifx'),I_MPI_FC=str(COMP/'bin/ifx'),
      OMP_NUM_THREADS='2',OMP_DYNAMIC='FALSE',OMP_PLACES='threads',OMP_PROC_BIND='close',OMP_MAX_ACTIVE_LEVELS='1',OMP_STACKSIZE='64M',
      MKL_NUM_THREADS='2',MKL_DYNAMIC='FALSE',MKL_THREADING_LAYER='INTEL',MKL_INTERFACE_LAYER='LP64',
      OPENBLAS_NUM_THREADS='2',BLIS_NUM_THREADS='2',OPENMX_GPU='0',CUDA_VISIBLE_DEVICES='',
      I_MPI_PIN='1',I_MPI_PIN_DOMAIN='['+','.join(MASKS)+']',I_MPI_DEBUG='4')
    return env
def binary(variant):return A/('install/bin/openmx' if variant=='intel' else 'reference/openmx')
def launch(variant,args):return [str(MPI/'bin/mpiexec'),'-np','18',str(binary(variant))]+args+['-nt','2']
def adapted(text):
    text=re.sub(r'^\s*DATA.PATH\s+[^\n]*\n?','',text,flags=re.M|re.I)
    text='DATA.PATH '+str(A/'DFT_DATA19')+'\n'+text
    return re.sub(r'^(\s*scf.eigen.lib\s+)\S+',r'\g<1>elpa2',text,flags=re.M|re.I)
def physical(text):return re.sub(r'^\s*(DATA.PATH|scf.eigen.lib)\s+[^\n]*\n?','',text,flags=re.M|re.I)

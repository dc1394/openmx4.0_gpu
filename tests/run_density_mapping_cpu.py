#!/usr/bin/env python3
"""Inject CUDA allocation/transfer failures into the retained density cache."""
import ast
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
source = (root / 'source/Set_Density_Grid_GPU.c').read_text()
helper = source[source.index('static void SDG_acc_delete_if_present('):source.index('static void SDG_delete_device(')]
tree = ast.parse((root / 'tests/run_hamiltonian_mapping_cpu.py').read_text())
prefix = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
              and any(getattr(t, 'id', '') == 'prefix' for t in n.targets))
prefix = prefix.replace('static int cudaMemcpy(', 'static int copy_fail=-1, copies=0;\nstatic int cudaMemcpy(')
prefix = prefix.replace('{memcpy(to,from,n);return 0;}', '{if(copies++==copy_fail)return 1;memcpy(to,from,n);return 0;}')
prefix += '\nstatic int mpi_comm_level1;\nstatic void MPI_Abort(int comm,int status){abort();}\n'
main = r'''
int main(void) {
 int data[15][17],cases=0;
 for(int i=0;i<15;i++)for(int j=0;j<17;j++)data[i][j]=i*91+j;
 for(int retained=0;retained<15;retained++)for(int kind=0;kind<2;kind++)for(int fail=0;fail<=14;fail++) {
  fail_after=copy_fail=-1;allocations=copies=0;
  assert(SDG_copyin(data[retained],sizeof(data[retained])));
  void *saved=acc_deviceptr(data[retained]);
  allocations=copies=0;if(kind)copy_fail=fail;else fail_after=fail;
  int created[15]={0},ok=1;
  for(int i=0;i<15;i++) {
   if(acc_deviceptr(data[i]))continue;
   if(!SDG_copyin(data[i],sizeof(data[i]))){ok=0;break;}
   created[i]=1;
  }
  assert(ok==(fail==14));
  if(ok)for(int i=0;i<15;i++)assert(!memcmp(acc_deviceptr(data[i]),data[i],sizeof(data[i])));
  for(int i=0;i<15;i++)if(created[i])SDG_acc_delete_if_present(data[i],sizeof(data[i]));
  assert(live==1 && acc_deviceptr(data[retained])==saved);
  assert(!memcmp(saved,data[retained],sizeof(data[retained])));
  SDG_acc_delete_if_present(data[retained],sizeof(data[retained]));assert(live==0);cases++;
 }
 fail_after=copy_fail=-1;assert(SDG_create(data[0],sizeof(data[0])));
 SDG_acc_delete_if_present(data[0],sizeof(data[0]));assert(live==0);
 assert(SDG_create(NULL,0));SDG_acc_delete_if_present(NULL,0);
 printf("PASS: %d density-cache allocation/transfer rollback cases plus empty/direct mappings\n",cases);
}
'''
with tempfile.TemporaryDirectory(prefix='openmx-density-map-') as folder:
    p = Path(folder)
    (p / 'check.c').write_text(prefix + helper + main)
    subprocess.run(shlex.split(os.environ.get('CC', 'cc')) + [
        '-std=c11', '-O1', '-g', '-fsanitize=address,undefined', '-fno-sanitize-recover=all',
        str(p / 'check.c'), '-o', str(p / 'check')], check=True)
    subprocess.run([str(p / 'check')], check=True, timeout=30)

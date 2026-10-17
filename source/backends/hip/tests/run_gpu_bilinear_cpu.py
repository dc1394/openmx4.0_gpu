#!/usr/bin/env python3
"""Check sparse density packing, deduplication, slicing and rollback."""
import ast
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

root=Path(__file__).resolve().parents[1]
tree=ast.parse((root/"tests/run_gpu_vector_cpu.py").read_text())
prefix=next(ast.literal_eval(n.value) for n in tree.body
            if isinstance(n,ast.Assign) and any(getattr(t,"id","")=="prefix" for t in n.targets))
body=(root/"source/openmx_gpu_bilinear.c").read_text()
body=body.replace('#include "hip_runtime_compat.h"','').replace('#include "set_hip_default_device_from_local_rank.h"','')
main=r'''
unsigned long long openmx_host_mem_available(void){return 1ULL<<36;}
int openmx_gpu_local_size_noncollective(void){return 2;}

enum {ROWS=13};
static int spins,negative,scaled;
static double jagged[3][ROWS],*jrows[3],flat[7],mat[4][3][2],*mrows[4][3],dm[12];
static float floats[5];
static double result[ROWS*4+2],expected[ROWS*4];
static int visit(void *context,size_t row,OpenMXBilinearEmit emit,void *target){
 (void)context;
 for(size_t k=0;k<row%4;k++){
  OpenMXBilinearTerm t={0};
  if(k%2==0){
   t.left=(OpenMXBilinearSpan){NULL,jrows,row,0,3,8};
   t.right=(OpenMXBilinearSpan){floats,NULL,1,2,2,4};
   for(int s=0;s<spins;s++)t.matrix_rows[s]=mrows[s];
  }else{
   t.left=(OpenMXBilinearSpan){flat,NULL,row%7,1,1,8};
   t.right=(OpenMXBilinearSpan){flat,NULL,0,2,3,8};t.matrix=dm;
  }
  t.scaled=scaled;for(int s=0;s<spins;s++)t.scale[s]=.11*(1+(int)row-s)-(double)k*.2;
  if(!emit(target,&t))return 0;
 }
 return 1;
}
static void commit(void *context,size_t first,size_t last,const double *values){
 (void)context;memcpy(result+1+first*spins,values,(last-first)*spins*sizeof(double));
}
int main(void){
 for(int i=0;i<3;i++){
  jrows[i]=jagged[i];for(int r=0;r<ROWS;r++)jagged[i][r]=.1*(1+i+r);
  for(int s=0;s<4;s++){mrows[s][i]=mat[s][i];for(int j=0;j<2;j++)mat[s][i][j]=.4*(s+1)-.2*i+.3*j;}
 }
 for(int i=0;i<7;i++)flat[i]=.07*(i-2);
 for(int i=0;i<5;i++)floats[i]=.13f*(i+1);
 for(int i=0;i<12;i++)dm[i]=.02*i-.1;
 int cases=0;
 for(spins=1;spins<=4;spins*=2)for(negative=0;negative<2;negative++)for(scaled=0;scaled<2;scaled++){
  for(int row=0;row<ROWS;row++)for(int s=0;s<spins;s++){
   double value=0;
   for(int k=0;k<row%4;k++){
    double term=0,scale=scaled?.11*(1+row-s)-(double)k*.2:1;
    if(k%2==0)for(int i=0;i<3;i++)for(int j=0;j<2;j++)term+=jagged[i][row]*(double)floats[1+j*2]*mat[s][i][j];
    else for(int j=0;j<3;j++)term+=flat[row%7]*flat[j*2]*dm[s*3+j];
    value+=scale*term;
   }
   expected[row*spins+s]=negative&&s==3?-value:value;
  }
  OpenMXBilinearSource source={ROWS,spins,negative,NULL,visit,commit};
  void *packet=NULL;size_t nb=0;assert(openmx_gpu_bilinear_pack(&source,0,ROWS,&packet,&nb));
  PackedSource p={0};assert(packed_open(packet,nb,&p));
  assert(p.h.matrices==(uint64_t)spins*9 && p.h.orbitals<p.h.terms*5);
  assert(!packed_open(packet,nb-1,&p));free(packet);
  for(workers=1;workers<=4;workers++)for(int tier=0;tier<4;tier++){
   local_fail=tier>0;remote_fail=tier==2;fail_alloc=tier==3;
   for(int i=0;i<ROWS*4+2;i++)result[i]=1234;
   int ok=openmx_gpu_bilinear_execute(0,&source,0,1,"test");
   assert(ok==(tier<2));
   for(int i=0;i<ROWS*spins;i++){
    if(ok)assert(fabs(result[i+1]-expected[i])<1e-12);
    else assert(result[i+1]==1234);
   }
   assert(result[0]==1234 && result[1+ROWS*spins]==1234);cases++;
  }
 }
 spins=4;negative=scaled=1;workers=4;local_fail=remote_fail=fail_alloc=remote_calls=0;fail_once=1;
 OpenMXBilinearSource retry={ROWS,spins,negative,NULL,visit,commit};
 assert(openmx_gpu_bilinear_execute(0,&retry,0,1,"late allocation"));
 assert(remote_calls==0 && fail_once==0);
 for(int i=0;i<ROWS*spins;i++)assert(fabs(result[i+1]-expected[i])<1e-12);
 printf("PASS: %d sparse density cases (FP32/FP64/jagged orbitals, spin, remotes and rollback)\n",cases);
}
'''
with tempfile.TemporaryDirectory(prefix="openmx-bilinear-") as folder:
    p=Path(folder);(p/"mpi.h").write_text("typedef int MPI_Comm;\n")
    (p/"check.c").write_text(prefix+body+main)
    cc=shlex.split(os.environ.get("CC","cc"));cxx=shlex.split(os.environ.get("CXX","c++"))
    flags=["-O1","-g","-fsanitize=address,undefined","-fno-sanitize-recover=all","-I"+str(p),"-I"+str(root/"source")]
    subprocess.run(cc+["-std=c11","-Wno-unknown-pragmas"]+flags+["-c",str(p/"check.c"),"-o",str(p/"check.o")],check=True)
    subprocess.run(cxx+["-std=c++17"]+flags+["-c",str(root/"source/openmx_gpu_bilinear_pack.cpp"),"-o",str(p/"pack.o")],check=True)
    subprocess.run(cxx+flags+[str(p/"check.o"),str(p/"pack.o"),"-lm","-o",str(p/"check")],check=True)
    subprocess.run([str(p/"check")],check=True,timeout=60)

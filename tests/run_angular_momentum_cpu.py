#!/usr/bin/env python3
"""Check real Voronoi angular-momentum assembly with the calculation-owned table.

Compare frozen C outputs through l=14 and independently verify antisymmetry,
commutators and L^2=l(l+1). No MPI/GPU initialization is needed. The obsolete
coefficient array is poisoned before every production call.
"""
from pathlib import Path
import os
import re
import resource
import shlex
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
library=Path(sys.argv[1]).resolve(strict=True)
source=(ROOT/'source/Voronoi_Orbital_Moment.c').read_text()
hit=re.search(r'^void Generate_L_Matrix\([^;]*?\)\s*\{',source,re.M)
end=source.index('{',hit.start())+1;depth=1
while depth:
    depth+=(source[end]=='{')-(source[end]=='}');end+=1
native=source[hit.start():end]
prefix=r'''
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "openmx_core_real_harmonics.h"
#define YOUSO36 14
#define W 29
static dcomplex Comp2Real[15][31][31];
static dcomplex Complex(double r,double i){return (dcomplex){r,i};}
static dcomplex Conjg(dcomplex x){return (dcomplex){x.r,-x.i};}
static void near(double a,double b){if(!isfinite(a)||!isfinite(b)||fabs(a-b)>4e-12*(1+fabs(b))){fprintf(stderr,"angular momentum %.17g != %.17g\n",a,b);abort();}}
'''
main=r'''
int main(void)
{
  char error[160];OpenMXRealHarmonicData conversion;
  OpenMXCoreContext *owner=openmx_core_create_host(error,sizeof error);
  if(!owner || !openmx_core_real_harmonics_initialize(owner,14,&conversion,error,sizeof error)){fprintf(stderr,"%s\n",error);return 1;}
  double old[3][15][W][W]={{{{0}}}},now[3][15][W][W]={{{{0}}}};
  double *old_rows[3][15][W],*new_rows[3][15][W],**old_degrees[3][15],**new_degrees[3][15];
  for(int k=0;k<3;++k)for(int l=0;l<=14;++l){
    old_degrees[k][l]=old_rows[k][l];new_degrees[k][l]=new_rows[k][l];
    for(int r=0;r<W;++r){old_rows[k][l][r]=old[k][l][r];new_rows[k][l][r]=now[k][l][r];}
  }
  Set_Comp2Real();legacy_Generate_L_Matrix(14,old_degrees[0],old_degrees[1],old_degrees[2]);
  for(int l=0;l<15;++l)for(int a=0;a<31;++a)for(int b=0;b<31;++b)Comp2Real[l][a][b]=Complex(NAN,NAN);
  Generate_L_Matrix(&conversion,14,new_degrees[0],new_degrees[1],new_degrees[2]);
  int values=0,identities=0;
  for(int l=0;l<=14;++l)for(int r=0;r<=2*l;++r)for(int c=0;c<=2*l;++c){
    double squared=0.;
    for(int axis=0;axis<3;++axis){
      if(memcmp(&old[axis][l][r][c],&now[axis][l][r][c],sizeof(double))){fputs("legacy angular matrix differs\n",stderr);return 2;}
      ++values;near(now[axis][l][r][c],-now[axis][l][c][r]);++identities;
      for(int j=0;j<=2*l;++j)squared-=now[axis][l][r][j]*now[axis][l][j][c];
      const int next=(axis+1)%3,last=(axis+2)%3;double commutator=0.;
      for(int j=0;j<=2*l;++j)commutator+=now[axis][l][r][j]*now[next][l][j][c]-now[next][l][r][j]*now[axis][l][j][c];
      near(commutator,now[last][l][r][c]);++identities;
    }
    near(squared,r==c?l*(l+1.):0.);++identities;
  }
  openmx_core_destroy(owner);
  printf("PASS: %d bitwise angular momentum values; %d independent algebra identities; legacy table poisoned\n",values,identities);
  return 0;
}
'''
with tempfile.TemporaryDirectory(prefix='openmx-angular-momentum-') as directory:
    folder=Path(directory);path=folder/'check.c'
    path.write_text(prefix+(ROOT/'modules/basis/tests/fixtures/real_harmonics_before_owner.inc').read_text()+
                    (ROOT/'tests/fixtures/angular_momentum_before_owner.c').read_text()+native+main)
    command=shlex.split(os.environ.get('CC','cc'))+['-std=c11','-O1','-g','-ffp-contract=off',
        '-fno-pie','-no-pie','-fsanitize=address,undefined','-fno-sanitize-recover=all',
        '-I'+str(ROOT/'source'),str(path),str(library),'-Wl,-rpath,'+str(library.parent),'-lm','-o',str(folder/'check')]
    subprocess.run(command,check=True)
    def bounded_stack():
        soft,hard=resource.getrlimit(resource.RLIMIT_STACK)
        if soft==resource.RLIM_INFINITY:resource.setrlimit(resource.RLIMIT_STACK,(8*1024*1024,hard))
    subprocess.run([str(folder/'check')],check=True,timeout=60,preexec_fn=bounded_stack)

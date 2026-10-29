#!/usr/bin/env python3
"""Compare the real native harmonic adapters with frozen pre-library kernels.

Separate C translation units prevent a same-function comparison from being
folded away. CC/CFLAGS select the host compiler; --sanitize enables ASan/UBSan.
No MPI, Kokkos, application globals, or GPU runtime is required.
"""
import argparse
import os
from pathlib import Path
import re
import resource
import shlex
import subprocess
import tempfile

ROOT=Path(__file__).resolve().parents[1]


def function(source,name):
    hit=re.search(r'^void '+name+r'\([^;]*?\)\s*\{',source,re.M)
    start=source.index('{',hit.start());end=start+1;depth=1
    while depth:
        depth+=(source[end]=='{')-(source[end]=='}');end+=1
    return source[hit.start():end]


HARNESS=r'''
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define PI 3.1415926535897932384626
void Associated_Legendre(int,int,double,double*);
void legacy_Associated_Legendre(int,int,double,double*);
void ComplexSH(int,int,double,double,double*,double*,double*);
void legacy_ComplexSH(int,int,double,double,double*,double*,double*);
void current_orbital(int,int,double,double,double,double*,double*,double*);
void legacy_openmx_orbital_complex_sh(int,int,double,double,double,double*,double*,double*);
static long compared;
static void equal(const double *a,const double *b,int n,int l,int m,double theta)
{
  for(int i=0;i<n;++i) {
    if(memcmp(a+i,b+i,sizeof(double))) {
      fprintf(stderr,"harmonics differ l=%d m=%d theta=%.17g component=%d %.17g != %.17g\n",l,m,theta,i,a[i],b[i]);
      exit(1);
    }
    ++compared;
  }
}
int main(void)
{
  const double angles[]={0,1e-20,1e-12,1e-8,.03,.7,PI/2,2.5,PI-1e-8,PI};
  const double phis[]={-3.1,-0.,.9,PI,5.3};
  for(int l=0;l<=64;++l)for(int m=-l;m<=l;++m)
    for(unsigned t=0;t<sizeof(angles)/sizeof(*angles);++t)
      for(unsigned p=0;p<sizeof(phis)/sizeof(*phis);++p) {
        double a[6],b[6];const double theta=angles[t],phi=phis[p];
        ComplexSH(l,m,theta,phi,a,a+2,a+4);
        legacy_ComplexSH(l,m,theta,phi,b,b+2,b+4);
        equal(a,b,6,l,m,theta);
        current_orbital(l,m,sin(theta),cos(theta),phi,a,a+2,a+4);
        legacy_openmx_orbital_complex_sh(l,m,sin(theta),cos(theta),phi,b,b+2,b+4);
        equal(a,b,6,l,m,theta);
        if(m>=0) {
          Associated_Legendre(l,m,cos(theta),a);
          legacy_Associated_Legendre(l,m,cos(theta),b);
          equal(a,b,2,l,m,theta);
        }
      }
  printf("PASS native harmonics: %ld bitwise-equal double results, degrees 0..64\n",compared);
  return 0;
}
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sanitize',action='store_true')
    args=parser.parse_args()
    source=(ROOT/'source/openmx_common.c').read_text()
    production='''#include <stdio.h>
#include <stdlib.h>
#include "openmx_basis_harmonics.h"
#include "openmx_orbital_harmonics.h"
'''+function(source,'Associated_Legendre')+'\n'+function(source,'ComplexSH')+'''
void current_orbital(int l,int m,double st,double ct,double phi,double* v,double* dt,double* dp)
{ openmx_orbital_complex_sh(l,m,st,ct,phi,v,dt,dp); }
'''
    with tempfile.TemporaryDirectory(prefix='openmx-harmonics-')as temporary:
        folder=Path(temporary)
        (folder/'production.c').write_text(production)
        (folder/'check.c').write_text(HARNESS)
        flags=shlex.split(os.environ.get('CFLAGS','-O3 -ffp-contract=off'))
        if args.sanitize:
            flags+=['-g','-fno-pie','-no-pie','-fsanitize=address,undefined','-fno-sanitize-recover=all']
        command=shlex.split(os.environ.get('CC','cc'))+['-std=c11',*flags,'-I'+str(ROOT/'source'),
            str(ROOT/'tests/fixtures/harmonics_before_library.c'),str(folder/'production.c'),
            str(folder/'check.c'),'-lm','-o',str(folder/'check')]
        subprocess.run(command,check=True)
        resource.setrlimit(resource.RLIMIT_STACK,(8*1024*1024,resource.getrlimit(resource.RLIMIT_STACK)[1]))
        subprocess.run([str(folder/'check')],check=True,timeout=60)


if __name__=='__main__':main()

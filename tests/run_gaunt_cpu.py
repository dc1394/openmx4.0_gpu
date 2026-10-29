#!/usr/bin/env python3
"""Compare the actual native Gaunt boundary against the frozen original C code.

Includes selection-rule zeros used by rectangular tables, signed zero and
factorial-domain boundaries. No MPI, Core, Kokkos, or GPU runtime is required.
"""
import argparse
import os
from pathlib import Path
import resource
import shlex
import subprocess
import tempfile

ROOT=Path(__file__).resolve().parents[1]
HARNESS=r'''
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
double Gaunt(int,int,int,int,int,int);
double legacy_Gaunt(int,int,int,int,int,int);
int main(void)
{
  long count=0;
  for(int l=0;l<=19;++l)for(int l1=0;l1<=19;++l1)for(int l2=0;l2<=19;++l2) {
    if(l+l1+l2>38)continue;
    for(int m=-l;m<=l;++m)for(int m1=-l1;m1<=l1;++m1) {
      const int m2=m-m1;
      const double now=Gaunt(l,m,l1,m1,l2,m2),old=legacy_Gaunt(l,m,l1,m1,l2,m2);
      if(!isfinite(now) || memcmp(&now,&old,sizeof(double))) {
        fprintf(stderr,"Gaunt differs (%d,%d,%d,%d,%d,%d): %.17g != %.17g\n",l,m,l1,m1,l2,m2,now,old);return 1;
      }
      ++count;
    }
  }
  const int padding[][3]={{4,12,24},{24,4,12},{38,0,0},{0,0,38}};
  for(unsigned p=0;p<sizeof(padding)/sizeof(padding[0]);++p) {
    const int l=padding[p][0],l1=padding[p][1],l2=padding[p][2];
    for(int m=-l;m<=l;++m)for(int m1=-l1;m1<=l1;++m1) {
      const double now=Gaunt(l,m,l1,m1,l2,m-m1),old=legacy_Gaunt(l,m,l1,m1,l2,m-m1);
      if(!isfinite(now) || memcmp(&now,&old,sizeof(double)))return 3;
      ++count;
    }
  }
  if(!isnan(Gaunt(-1,0,0,0,0,0)) || !isnan(Gaunt(19,0,19,0,1,0)) ||
     !isnan(Gaunt(INT_MAX,0,0,0,0,0)) || !isnan(Gaunt(0,INT_MIN,0,0,0,0)))return 2;
  printf("PASS native Gaunt: %ld bitwise comparisons, signed selection zeros and safe factorial boundaries\n",count);
  return 0;
}
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sanitize',action='store_true')
    args=parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='openmx-gaunt-') as temporary:
        folder=Path(temporary);(folder/'check.c').write_text(HARNESS)
        flags=shlex.split(os.environ.get('CFLAGS','-O3 -ffp-contract=off'))
        if args.sanitize:
            flags+=['-g','-fno-pie','-no-pie','-fsanitize=address,undefined','-fno-sanitize-recover=all']
        command=shlex.split(os.environ.get('CC','cc'))+['-std=c11',*flags,
            '-I'+str(ROOT/'source'),str(ROOT/'source/Gaunt.c'),
            str(ROOT/'modules/basis/tests/fixtures/gaunt_before_library.c'),str(folder/'check.c'),
            '-lm','-o',str(folder/'check')]
        subprocess.run(command,check=True)
        resource.setrlimit(resource.RLIMIT_STACK,(8*1024*1024,resource.getrlimit(resource.RLIMIT_STACK)[1]))
        subprocess.run([str(folder/'check')],check=True,timeout=60)


if __name__=='__main__':main()

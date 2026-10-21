#!/usr/bin/env python3
"""Check production stateless Hartree interpolation and radial derivatives.

Use different caller-owned rows and charges, both extrapolation branches,
interior points, mesh endpoints and the logarithmic derivative's chain rule.
Compile the actual C functions under ASan/UBSan without application globals.
"""
import os
from pathlib import Path
import shlex
import resource
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
MAIN = r'''
static void check(double got,double expected) {
  if (!isfinite(got) || fabs(got-expected)>2e-12*(1+fabs(expected))) {
    fprintf(stderr,"Hartree interpolation: %.17g != %.17g\n",got,expected);abort();
  }
}
int main(void) {
  enum {N=16};double xv[N],rv[N],yv[N+2];int cases=0;
  for (int mesh=0;mesh<2;++mesh) {
    double xmin=-4.0+mesh*.7,step=.3+mesh*.05;
    for (int i=0;i<N;++i) {xv[i]=xmin+i*step;rv[i]=exp(xv[i]);}
    for (int i=0;i<N+2;++i) yv[i]=1.75;
    for (int k=0;k<4;++k) {
      double r=rv[0]*(.1+.2*k),x=log(r);
      check(VH_AtomF(7,N,x,r,xv,rv,yv),1.75);
      check(Dr_VH_AtomF(7,N,x,r,xv,rv,yv),0);cases+=2;
    }
    for (int i=0;i<N+2;++i) {
      double x=xmin+(i-1)*step;yv[i]=.2*x*x-.7*x+1.3;
    }
    for (int k=0;k<=90;++k) {
      double x=xmin+(N-1)*step*k/91.0,r=exp(x);
      check(VH_AtomF(3,N,x,r,xv,rv,yv),.2*x*x-.7*x+1.3);
      check(Dr_VH_AtomF(3,N,x,r,xv,rv,yv),(.4*x-.7)/r);cases+=2;
    }
    for (int charge=1;charge<=7;charge+=3) for (int k=0;k<4;++k) {
      double x=xv[N-1]+k*.2,r=exp(x),q=charge+.125;
      check(VH_AtomF(q,N,x,r,xv,rv,yv),q/r);
      check(Dr_VH_AtomF(q,N,x,r,xv,rv,yv),-q/r/r);cases+=2;
    }
  }
  printf("PASS: %d stateless Hartree value/derivative comparisons\n",cases);return 0;
}
'''
with tempfile.TemporaryDirectory(prefix='openmx-atomic-hartree-') as directory:
    folder=Path(directory)
    functions='\n'.join((ROOT/'source'/name).read_text().replace('#include "openmx_common.h"','')
                        for name in ('VH_AtomF.c','Dr_VH_AtomF.c'))
    source=folder/'check.c';source.write_text('#include <stdlib.h>\n'+functions+MAIN)
    command=shlex.split(os.environ.get('CC','cc'))+['-std=c11','-O1','-g',
        '-fsanitize=address,undefined','-fno-sanitize-recover=all',str(source),'-lm','-o',str(folder/'check')]
    subprocess.run(command,check=True)
    def bounded_stack():
        soft,hard=resource.getrlimit(resource.RLIMIT_STACK)
        if soft==resource.RLIM_INFINITY:
            resource.setrlimit(resource.RLIMIT_STACK,(8*1024*1024,hard))
    subprocess.run([str(folder/'check')],check=True,timeout=30,preexec_fn=bounded_stack)

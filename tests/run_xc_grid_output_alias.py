#!/usr/bin/env python3
"""Check GPU XC result downloads preserve the host's aliased-output contract.

Extract the production transfer block; only device-to-host copies are stubbed.
No CUDA kernels or functional arithmetic are simulated by this test.
"""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
source = (root / "source/Set_XC_Grid.c").read_text()
start = source.index("  /* Match the host rotation's store order:")
end = source.index("  if (dDen_Grid != NULL)", start)
code = r'''
#include <assert.h>
#include <stdio.h>
#include <string.h>
static void acc_memcpy_from_device(void *out,const void *in,size_t bytes) {
  assert(out && in); memcpy(out,in,bytes);
}
static void download(int rotate,double *Vxc0,double *Vxc1,double *Vxc2,double *Vxc3,
                     const double *d_vxc0,const double *d_vxc1,const double *d_vxc2,const double *d_vxc3) {
  const size_t nb=3*sizeof(double);
'''
code += source[start:end]
code += r'''
}
int main(void) {
  const double up[3]={-0.5,-0.25,-0.75},down[3]={-0.3,-0.15,-0.65};
  const double re[3]={0.01,0.02,0.03},im[3]={0.04,0.05,0.06},zero[3]={0,0,0};
  double a[3],b[3],c[3],d[3];
  download(1,a,b,c,d,up,down,re,im);
  assert(!memcmp(a,up,sizeof(a)) && !memcmp(b,down,sizeof(b)));
  assert(!memcmp(c,re,sizeof(c)) && !memcmp(d,im,sizeof(d)));
  /* The documented scalar-reference call aliases all four outputs;
     equal spin potentials produce zero off-diagonal components. */
  download(1,a,a,a,a,up,up,zero,zero);
  assert(!memcmp(a,up,sizeof(a)));
  /* Aliasing only off-diagonal and diagonal output pairs keeps the
     diagonal values, as does the host's non-collinear rotation. */
  download(1,a,b,a,b,up,down,re,im);
  assert(!memcmp(a,up,sizeof(a)) && !memcmp(b,down,sizeof(b)));
  /* Collinear/energy-density paths never access the optional outputs. */
  download(0,a,b,NULL,NULL,up,down,NULL,NULL);
  assert(!memcmp(a,up,sizeof(a)) && !memcmp(b,down,sizeof(b)));
  puts("PASS: XC distinct outputs, scalar alias, pair aliases, optional outputs");
  return 0;
}
'''
with tempfile.TemporaryDirectory(prefix="openmx-xc-alias-") as tmp:
    cfile, exe = Path(tmp) / "check.c", Path(tmp) / "check"
    cfile.write_text(code)
    subprocess.run(shlex.split(os.environ.get("CC", "cc")) +
                   ["-std=c11", "-O1", "-Wall", "-Wextra", str(cfile), "-o", str(exe)], check=True)
    subprocess.run([str(exe)], check=True)

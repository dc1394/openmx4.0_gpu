#!/usr/bin/env python3
"""Check production pre-context Band counts against explicitly enumerated orbits."""
from pathlib import Path
import os
import re
import shlex
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
source = (root / 'source/openmx_common.c').read_text()
start = source.index('void k_inversion(')
begin = source.index('{', start)
depth, end = 1, begin + 1
while depth:
    depth += (source[end] == '{') - (source[end] == '}')
    end += 1
inversion = source[start:end]
test = r'''
#include "openmx_mgpu_band_plan.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
INVERSION
static int checks=0;
#define CHECK(x) do { ++checks; if(!(x)) {fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#x);exit(1);} } while(0)
static int orbits(int generation,int nx,int ny,int nz) {
  unsigned char seen[12*12*7];
  memset(seen,0,sizeof(seen));
  int count=0;
  for(int x=0;x<nx;++x)for(int y=0;y<ny;++y)for(int z=0;z<nz;++z) {
    const int index=(x*ny+y)*nz+z;
    if(seen[index])continue;
    int ix,iy,iz;
    if(generation==1)k_inversion(x,y,z,nx,ny,nz,&ix,&iy,&iz);
    else {ix=(nx-x)%nx;iy=(ny-y)%ny;iz=(nz-z)%nz;}
    seen[index]=seen[(ix*ny+iy)*nz+iz]=1;
    ++count;
  }
  return count;
}
int main(void) {
  for(int nx=1;nx<=12;++nx)for(int ny=1;ny<=12;++ny)for(int nz=1;nz<=7;++nz)
    for(int generation=1;generation<=3;generation+=2) {
      const int actual=orbits(generation,nx,ny,nz);
      CHECK(openmx_mgpu_band_kpoint_count(0,generation,nx,ny,nz,0)==actual);
      CHECK(openmx_mgpu_band_kpoint_count(1,generation,nx,ny,nz,0)==actual);
      CHECK(openmx_mgpu_band_kpoint_count(3,generation,nx,ny,nz,0)==nx*ny*nz);
    }
  CHECK(openmx_mgpu_band_kpoint_count(0,1,2,1,1,0)==1);
  CHECK(openmx_mgpu_band_kpoint_count(1,1,4,1,1,0)==2);
  CHECK(openmx_mgpu_band_kpoint_count(1,1,6,1,1,0)==3);
  CHECK(openmx_mgpu_band_kpoint_count(3,1,3,1,1,0)==3);
  CHECK(openmx_mgpu_band_kpoint_count(0,2,8,8,8,17)==17);
  CHECK(openmx_mgpu_band_kpoint_count(1,2,8,8,8,17)==17);
  CHECK(openmx_mgpu_band_kpoint_count(0,2,8,8,8,0)==0);
  CHECK(openmx_mgpu_band_kpoint_count(0,2,8,8,8,-1)==0);
  CHECK(openmx_mgpu_band_kpoint_count(0,0,1,1,1,0)==0);
  CHECK(openmx_mgpu_band_kpoint_count(2,1,1,1,1,0)==0);
  CHECK(openmx_mgpu_band_kpoint_count(-1,1,1,1,1,0)==0);
  CHECK(openmx_mgpu_band_kpoint_count(0,1,0,1,1,0)==0);
  CHECK(openmx_mgpu_band_kpoint_count(3,1,1,-1,1,0)==0);
  CHECK(openmx_mgpu_band_kpoint_count(1,3,INT_MAX,INT_MAX,INT_MAX,0)==0);
  CHECK(openmx_mgpu_band_kpoint_count(3,1,INT_MAX,2,1,0)==0);
  CHECK(openmx_mgpu_band_kpoint_count(0,1,INT_MAX,1,1,0)==INT_MAX/2+1);
  CHECK(openmx_mgpu_band_kpoint_count(3,1,INT_MAX,1,1,0)==INT_MAX);
  printf("PASS: %d Band k-point plan checks against enumerated inversion orbits, spin, special grids and overflow\n",checks);
  return 0;
}
'''.replace('INVERSION', inversion)
with tempfile.TemporaryDirectory(prefix='openmx-band-plan-') as folder:
    tmp = Path(folder)
    (tmp / 'test.c').write_text(test)
    command = shlex.split(os.environ.get('CC', 'cc')) + [
        '-std=c11', '-O1', '-Wall', '-Wextra', '-fsanitize=undefined',
        '-fno-sanitize-recover=all', '-I' + str(root / 'source'),
        str(tmp / 'test.c'), '-o', str(tmp / 'test')]
    subprocess.run(command, check=True)
    subprocess.run([str(tmp / 'test')], check=True)

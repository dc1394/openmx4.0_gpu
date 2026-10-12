#!/usr/bin/env python3
"""Check production OpenACC density fallbacks with CUDA deliberately declined.

OpenACC pragmas are ignored on the CPU.  This exercises full-pair and compact
point-group indexing against an independent long-double quadrature, including
spin channels, repeated output indices, empty overlaps and group boundaries.
"""
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
source = (ROOT / "source/Set_Hamiltonian.c").read_text()
start = source.index("static void Set_Hamiltonian_OTF_DensityKernel(")
end = source.index("\n/* The density of the rank", start)
kernel = source[start:end]
prefix = r'''
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static int mpi_comm_level1;
static int MPI_Comm_rank(int comm,int *rank) { (void)comm; *rank=0; return 0; }
static void Set_Hamiltonian_abort(const char *a,const char *b,int rank) {
  (void)rank; fprintf(stderr,"%s: %s\n",a,b); exit(1);
}
/* Force the production double-float OpenACC fallback. */
#define Set_Hamiltonian_Cuda_Density(...) 1
static void check(int ok,const char *s) { if(!ok) { fprintf(stderr,"FAIL: %s\n",s); exit(1); } }
'''
main = r'''
int main(void) {
  enum { ROWS=521, OUT=550, MAXPTS=769, GUARD=7, MAXDM=92 };
  const int sizes[]={0,1,255,256,257,511,513,769};
  int cases=0;
  for(int mode=0;mode<3;mode++) for(int spins=1;spins<=4;spins*=2)
  for(size_t si=0;si<sizeof(sizes)/sizeof(sizes[0]);si++) {
    const int n=sizes[si];
    int no0[2]={3,4}, no1[2]={5,2}, nolg[2]={n,n/2+7};
    size_t hoff[2]={0,(size_t)spins*15}, noff[2]={0,(size_t)n};
    size_t o0[2]={0,ROWS*3}, o1[2]={0,(size_t)n*5}, outbase[2]={7,17};
    int nc[2*MAXPTS];
    float phi0[ROWS*7], phi1[MAXPTS*7], compact[MAXPTS*4];
    double dm[MAXDM], full[4*OUT+2*GUARD], chunks[4*OUT+2*GUARD];
    long double expect[4*OUT];
    float dmh[MAXDM], dml[MAXDM];
    const size_t dmcount=(size_t)spins*23;
    for(size_t j=0;j<dmcount;j++) {
      dm[j]=sin((double)(j+1)*0.57)*0.19;
      dmh[j]=(float)dm[j]; dml[j]=(float)(dm[j]-(double)dmh[j]);
    }
    for(int p=0;p<2;p++) {
      for(int row=0;row<ROWS;row++) for(int i=0;i<no0[p];i++)
        phi0[o0[p]+(size_t)row*no0[p]+i]=(float)(cos(row*0.079+i*0.71+p)*0.27);
      for(int k=0;k<nolg[p];k++) {
        nc[noff[p]+k]=(k*17+31*p)%ROWS;
        for(int j=0;j<no1[p];j++)
          phi1[o1[p]+(size_t)k*no1[p]+j]=(float)(sin(k*0.123+j*0.67+p)*0.23);
      }
    }
    for(int k=0;k<4*OUT+2*GUARD;k++) full[k]=chunks[k]=0.03125;
    for(int k=0;k<4*OUT;k++) expect[k]=0.03125L;
    for(int p=0;p<2;p++) for(int k=0;k<nolg[p];k++) for(int sp=0;sp<spins;sp++) {
      long double val=0;
      const int row=nc[noff[p]+k];
      for(int i=0;i<no0[p];i++) for(int j=0;j<no1[p];j++) {
        size_t ij=hoff[p]+(size_t)sp*no0[p]*no1[p]+(size_t)i*no1[p]+j;
        long double d=mode==1 ? (long double)dmh[ij] : (long double)dm[ij];
        val+=(long double)phi0[o0[p]+(size_t)row*no0[p]+i]*d*
             (long double)phi1[o1[p]+(size_t)k*no1[p]+j];
      }
      expect[(size_t)sp*OUT+outbase[p]+row]+=val;
    }
    Set_Hamiltonian_OTF_DensityKernel(2,spins,5,nolg[0]>nolg[1]?nolg[0]:nolg[1],
      no0,no1,nolg,hoff,noff,o0,o1,outbase,nc,phi0,phi1,dm,dmh,dml,full+GUARD,OUT,mode,0);
    for(int p=0;p<2;p++) {
      /* The 513-point chunks cross multiple 256-point gang boundaries. */
      for(int first=0;first<nolg[p] || (first==0 && nolg[p]==0);first+=513) {
        int count=nolg[p]-first; if(count>513) count=513;
        size_t zero=0;
        for(int k=0;k<count;k++) for(int i=0;i<no0[p];i++)
          compact[(size_t)k*no0[p]+i]=phi0[o0[p]+(size_t)nc[noff[p]+first+k]*no0[p]+i];
        Set_Hamiltonian_OTF_DensityKernel(1,spins,5,count,no0+p,no1+p,&count,hoff+p,&zero,
          &zero,&zero,outbase+p,nc+noff[p]+first,compact,phi1+o1[p]+(size_t)first*no1[p],
          dm,dmh,dml,chunks+GUARD,OUT,mode,1);
        if(!count) break;
      }
    }
    for(int k=0;k<GUARD;k++) {
      check(full[k]==0.03125 && chunks[k]==0.03125,"leading output guard");
      check(full[GUARD+4*OUT+k]==0.03125 && chunks[GUARD+4*OUT+k]==0.03125,"trailing output guard");
    }
    for(int k=0;k<4*OUT;k++) {
      const long double tol=mode==1?2e-7L:2e-12L;
      check(full[GUARD+k]==chunks[GUARD+k],"full and point-group results agree exactly");
      check(fabsl((long double)chunks[GUARD+k]-expect[k])<tol,"independent quadrature");
    }
    cases++;
  }
  printf("PASS: %d production density fallback cases, FP64/FP32/DF, 1/2/4 spins, point-group boundaries and output guards\n",cases);
  return 0;
}
'''
with tempfile.TemporaryDirectory(prefix="openmx-density-fallback-") as tmp:
    cfile, binary = Path(tmp)/"check.c", Path(tmp)/"check"
    cfile.write_text(prefix + kernel + main)
    subprocess.run(shlex.split(os.environ.get("CC", "cc")) + ["-std=c11", "-O1", "-g", "-Wall", "-Wextra", "-Wno-unknown-pragmas", "-Wno-unused-parameter"] + shlex.split(os.environ.get("CFLAGS", "")) + [str(cfile), "-lm", "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True, timeout=30)

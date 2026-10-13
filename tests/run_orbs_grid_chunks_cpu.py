#!/usr/bin/env python3
"""Check production compact orbital gathers against the per-point evaluators.

Only allocation/transfers are replaced with host operations. ASan/UBSan cover
valid, repeated sphere indices, signed coordinates, translated neighbors,
partial chunks and output guards in both FP64 and double-float modes.
"""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
source = (root / "source/Set_Orbitals_Grid.c").read_text()
point = source[source.index("#pragma acc routine seq\nstatic void SOG_point_eval("):
               source.index("/* One launch evaluates every grid point of one atom")]
prepare = source[source.index("int SOG_Device_Prepare(void)"):
                 source.index("\nconst char *SOG_Device_EvalPrecisionName")]
chunk = source[source.index("void SOG_Device_EvalPairChunk("):
               source.index("\ndouble Set_Orbitals_Grid(int Cnt_kind)")]
code = r'''
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "orbs_grid_gpu.h"
#include "openmx_precision.h"
#define SOG_L0MAX 3
#define SOG_MULMAX 8
static SOG_DeviceTables SOG_dev;
static int SOG_dev_fp32_enabled = -1;
void SOG_Device_Release(void) {
  free(SOG_dev.arena); memset(&SOG_dev,0,sizeof(SOG_dev)); SOG_dev_fp32_enabled=-1;
}
static int SpeciesNum=1, TCpyCell=2, Ngrid2=5, Ngrid3=7;
static int Spe_Num_Mesh_PAO[]={8}, Spe_MaxL_Basis[]={3};
static int nb[]={2,1,1,1}, *Spe_Num_Basis[]={nb};
static double rv[]={0.08,0.16,0.32,0.64,1.28,2.56,5.12,10.24};
static double radial[5][8], *Spe_PAO_RV[]={rv};
static double *s[]={radial[0],radial[1]}, *p[]={radial[2]}, *d[]={radial[3]}, *f[]={radial[4]};
static double **angular[]={s,p,d,f}, ***Spe_PAO_RWF[]={angular};
static double atv[3][4]={{0,0,0,0},{0,0.21,-0.17,0.08},{0,-0.30,0.19,0.12}};
static double gtv[4][4]={{0,0,0,0},{0,0.031,0.002,-0.007},
                        {0,-0.003,0.041,0.006},{0,0.008,-0.005,0.053}};
static double Grid_Origin[4]={0,-0.31,0.17,-0.23};
static int SOG_gpu_eligible(int kind) { (void)kind; return 1; }
static int SOG_eval_double_float(void) { return openmx_fp32_enabled(); }
static size_t SOG_arena_off(size_t *p,size_t n) { size_t o=*p; *p=(o+n+511)&~(size_t)511; return o; }
static void *SOG_arena_try(size_t n) { return malloc(n); }
static void acc_memcpy_to_device(void *d,const void *s,size_t n) { memcpy(d,s,n); }
'''
code += point + prepare + chunk
code += r'''
static void oracle(const SOG_GpuPair *p, int gn, int cell, float *out)
{
  const SOG_DeviceTables *t=&SOG_dev;
#define GEOMETRY t->ng23,t->ng3,t->g[0][0],t->g[0][1],t->g[0][2], \
                 t->g[1][0],t->g[1][1],t->g[1][2],t->g[2][0],t->g[2][1],t->g[2][2], \
                 t->org[0],t->org[1],t->org[2],out
  if(t->eval_df && openmx_fp32_enabled())
    SOG_point_eval_df(p->wan,p->no,p->gx,p->gy,p->gz,p->ax,p->ay,p->az,gn,cell,
      t->atv,t->rv,t->rwf,t->rwd,t->mcd,t->spl,t->rvo,t->rwb,t->spm,t->spx,t->spb,GEOMETRY);
  else
    SOG_point_eval(p->wan,p->no,p->gx,p->gy,p->gz,p->ax,p->ay,p->az,gn,cell,
      t->atv,t->rv,t->rwf,t->rvo,t->rwb,t->spm,t->spx,t->spb,GEOMETRY);
#undef GEOMETRY
}

int main(void)
{
  enum {ROWS=719, POINTS=273, NO=17, GPREFIX=11, NPREFIX=5, GUARD=19};
  int gla[GPREFIX+ROWS+7],cla[GPREFIX+ROWS+7],nc[NPREFIX+POINTS+3];
  const int lengths[]={-1,0,1,127,128,129,POINTS};
  float actual[GUARD+POINTS*NO+GUARD], expected[GUARD+POINTS*NO+GUARD];
  int cases=0,shift_changes=0;
  for(int r=0;r<5;r++) for(int m=0;m<8;m++)
    radial[r][m]=(r+1)*exp(-rv[m]*(0.31+0.07*r))*(1+0.1*sin(rv[m]+r));
  assert(SOG_Device_Prepare());
  for(int i=0;i<GPREFIX+ROWS+7;i++) { gla[i]=0; cla[i]=0; }
  for(int i=0;i<ROWS;i++) { gla[GPREFIX+i]=(i*53)%557-278; cla[GPREFIX+i]=i%3; }
  for(int i=0;i<NPREFIX+POINTS+3;i++) nc[i]=0;
  for(int i=0;i<POINTS;i++) nc[NPREFIX+i]=((i/2)*37+317)%ROWS;
  for(int enabled=0;enabled<2;enabled++) for(int mode=0;mode<2;mode++) for(int shift=0;shift<3;shift++) {
    openmx_set_fp32_enabled(enabled);
    SOG_GpuPair pair={0}; pair.no=NO;
    pair.gx=0.17; pair.gy=-0.23; pair.gz=0.31;
    pair.ax=atv[shift][1]; pair.ay=atv[shift][2]; pair.az=atv[shift][3];
    SOG_dev.eval_df=mode;
    for(size_t l=0;l<sizeof(lengths)/sizeof(lengths[0]);l++) {
      const int length=lengths[l];
      for(size_t i=0;i<sizeof(actual)/sizeof(actual[0]);i++) actual[i]=expected[i]=111.25f;
      for(int k=0;k<length;k++) {
        const int row=nc[NPREFIX+k];
        oracle(&pair,gla[GPREFIX+row],cla[GPREFIX+row],expected+GUARD+k*NO);
      }
      SOG_Device_EvalPairChunk(&pair,length,nc+NPREFIX,gla+GPREFIX,cla+GPREFIX,actual+GUARD);
      for(size_t i=0;i<sizeof(actual)/sizeof(actual[0]);i++) {
        assert(isfinite(actual[i]));
        assert(memcmp(actual+i,expected+i,sizeof(float))==0);
      }
      /* Every later chunk receives offset input/output pointers, while its
         Nc values remain original sphere indices. */
      for(size_t i=0;i<sizeof(actual)/sizeof(actual[0]);i++) actual[i]=111.25f;
      for(int first=0;first<length;first+=37) {
        const int count=length-first<37?length-first:37;
        SOG_Device_EvalPairChunk(&pair,count,nc+NPREFIX+first,gla+GPREFIX,cla+GPREFIX,actual+GUARD+first*NO);
      }
      assert(memcmp(actual,expected,sizeof(actual))==0);
      cases++;
    }
    if(shift) {
      SOG_GpuPair center=pair; float a[NO],b[NO];
      center.ax=center.ay=center.az=0;
      oracle(&pair,gla[GPREFIX+317],cla[GPREFIX+317],a);
      oracle(&center,gla[GPREFIX+317],cla[GPREFIX+317],b);
      if(memcmp(a,b,sizeof(a))) shift_changes++;
    }
  }
  assert(shift_changes==8);
  /* A setting change rebuilds the cached tables, dropping DF allocations
     while disabled and restoring them when a later input enables FP32. */
  for(int enabled=0;enabled<3;enabled++) {
    openmx_set_fp32_enabled(enabled%2);
    assert(SOG_Device_Prepare());
    assert(SOG_dev.eval_df==enabled%2);
    assert((SOG_dev.rwd!=NULL)==enabled%2);
    assert((SOG_dev.mcd!=NULL)==enabled%2);
    assert((SOG_dev.spl!=NULL)==enabled%2);
  }
  SOG_dev.ready=0;
  {
    SOG_GpuPair pair={0}; pair.no=NO; float guard=913.5f;
    SOG_Device_EvalPairChunk(&pair,POINTS,nc+NPREFIX,gla+GPREFIX,cla+GPREFIX,&guard);
    assert(guard==913.5f);
  }
  SOG_Device_Release();
  printf("PASS: %d FP64/DF compact orbital gathers; signed coordinates, cell/neighbor shifts, "
         "s/p/d/f orbitals, repeated sphere indices, chunk offsets, empty/unready calls, output guards\n",cases);
}
'''
with tempfile.TemporaryDirectory(prefix="openmx-orbs-chunks-") as tmp:
    test, exe = Path(tmp) / "check.c", Path(tmp) / "check"
    test.write_text(code)
    subprocess.run(shlex.split(os.environ.get("CC", "cc")) + [
        "-std=c11", "-O1", "-g", "-ffp-contract=off",
        "-fsanitize=address,undefined", "-fno-omit-frame-pointer",
        "-fno-pie", "-no-pie", "-I" + str(root / "source"),
        str(test), str(root / "source/openmx_precision.c"), "-lm", "-o", str(exe)], check=True)
    subprocess.run([str(exe)], check=True, timeout=30)

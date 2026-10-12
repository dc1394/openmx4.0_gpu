#!/usr/bin/env python3
"""Validate production OTF cached batch images and invalidation on the CPU.

CUDA allocation is replaced by host allocation; numerical kernels are not
run. The oracle checks each packed pair against its original atom geometry,
including cumulative DM offsets across multiple batches and both cache slots.
"""
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
source = (ROOT / "source/Set_Hamiltonian.c").read_text()


def function(name):
    match = re.search(r"^[^\n;]*\b" + re.escape(name) + r"\([^;]*?\)\n\{", source, re.M)
    if not match:
        raise RuntimeError(name)
    pos, depth = match.end(), 1
    while depth:
        if source[pos] == "{":
            depth += 1
        elif source[pos] == "}":
            depth -= 1
        pos += 1
    return source[match.start():pos]


def typedef(name):
    end = source.index("} " + name + ";") + len("} " + name + ";")
    return source[source.rfind("typedef struct {", 0, end):end]


prefix = r'''
#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef struct { int wan,no,pt0,pad; size_t out; double gx,gy,gz,ax,ay,az; } SOG_GpuPair;
static int Matomnum=3;
static int M2G[4]={0,1,2,3}, Spe_Total_NO[4]={0,3,5,7}, WhatSpecies[4]={0,1,2,3};
static int GridN_Atom[4]={0,31,47,59}, FNAN[4]={0,2,2,2};
static int natn[4][3], ncn[4][3], NumOLG[4][3], glist[4][3][60], *GListTAtoms1[4][3];
static int mglist[4][60], *MGridListAtom[4];
static double Gxyz[4][4], atv[3][4];
static int alloc_count, live_alloc, force_pageable;
static size_t gpu_free=16ULL*1024*1024*1024;
static int gpu_query_error;
static int Grid_Precision_Kernel(void) { return 2; }
static int OpenMX_GpuMemGetInfo(size_t *free_bytes,size_t *total_bytes) {
  *free_bytes=gpu_free; *total_bytes=16ULL*1024*1024*1024; return gpu_query_error;
}
static void check(int ok,const char *msg) { if(!ok) { fprintf(stderr,"FAIL: %s\n",msg); exit(1); } }
static void *alloc(size_t n) { void *p=calloc(1,n?n:1); check(p!=NULL,"host allocation"); alloc_count++; live_alloc++; return p; }
static void release(void *p) { if(p) { live_alloc--; free(p); } }
#define cudaSuccess 0
static int cudaMallocHost(void **p,size_t n) { if(force_pageable) return 1; *p=alloc(n); return 0; }
static int cudaFreeHost(void *p) { release(p); return 0; }
static int cudaGetLastError(void) { return 0; }
static void acc_free(void *p) { release(p); }
static void Set_Hamiltonian_abort(const char *a,const char *b,int r) { (void)r; fprintf(stderr,"%s: %s\n",a,b); exit(1); }
static void *Set_Hamiltonian_malloc(size_t n,const char *s,int r) { (void)s; (void)r; return alloc(n); }
static size_t Set_Hamiltonian_checked_mul(size_t a,size_t b,const char *s,int r) {
  if(a && b>SIZE_MAX/a) Set_Hamiltonian_abort(s,"overflow",r);
  return a*b;
}
static void Set_Hamiltonian_add_array_bytes(size_t *a,size_t n,size_t z,const char *s,int r) {
  size_t v=Set_Hamiltonian_checked_mul(n,z,s,r); if(v>SIZE_MAX-*a) Set_Hamiltonian_abort(s,"overflow",r); *a+=v;
}
static void Set_Hamiltonian_MatrixElements_CountTotals(int k,int r,int *n,size_t *h,size_t *g,size_t *o0,size_t *o1) {
  (void)k; (void)r; *n=9; *h=*g=*o0=*o1=0;
}
static void Set_Hamiltonian_Free_OpenACC_MatrixElements_Cache(void) {}
'''
parts = [prefix, typedef("SetHamiltonianOnTheFlyCache"), typedef("SetHOTFBatch"), typedef("SetHOTFPlan"), typedef("SetHOTFPairCost"), typedef("SetHOTFDensityPlan"),
         "static SetHamiltonianOnTheFlyCache Set_Hamiltonian_OTF;\nstatic SetHOTFPlan Set_Hamiltonian_OTF_Plan[2];\nstatic SetHOTFDensityPlan Set_Hamiltonian_OTF_DenPlan;"]
for name in ["SETH_OTF_DEN_RESERVE_BYTES", "SETH_OTF_DEN_RUNTIME_SLACK", "SETH_OTF_DEN_ALIGNMENT_SLACK", "SETH_OTF_DEN_GROW_QUANTUM"]:
    parts.append(re.search(r"^#define " + name + r"[^\n]*", source, re.M).group(0))
for name in ["SETH_OTF_PairCost", "SETH_OTF_CompactCost", "SETH_OTF_DensityLimitBytes", "SETH_OTF_DensityFixedBytes", "SETH_OTF_DensityBoundWorkspace", "SETH_OTF_DensityPairBytes", "Set_Hamiltonian_OnTheFly_DensityBudget", "Set_Hamiltonian_OnTheFly_DensityGrow", "SETH_OTF_FreePlan", "SETH_OTF_FreePlans", "SETH_OTF_GetPlan", "Set_Hamiltonian_OTF_Release", "Set_Hamiltonian_Invalidate_OpenACC_MatrixElements_Cache"]:
    body = function(name).replace("free(", "release(")
    # Device-release stub already accounts for its allocation.
    body = body.replace("acc_release(", "acc_free(")
    parts.append(body)
parts.append(r'''
int main(void) {
  unsigned int out_base[4]={0,0,31,78};
  int cases=0;
  for(int a=1;a<=3;a++) {
    MGridListAtom[a]=mglist[a];
    for(int k=0;k<GridN_Atom[a];k++) mglist[a][k]=a*100+k;
    for(int j=1;j<=3;j++) Gxyz[a][j]=0.25*a*j;
    for(int p=0;p<3;p++) {
      natn[a][p]=1+(a+p)%3; ncn[a][p]=p;
      NumOLG[a][p]=(a*13+p*7)%GridN_Atom[a];
      GListTAtoms1[a][p]=glist[a][p];
      for(int k=0;k<NumOLG[a][p];k++) glist[a][p][k]=(k*11+p)%GridN_Atom[a];
    }
  }
  for(int spins=1;spins<=4;spins*=2) for(int pageable=0;pageable<2;pageable++) {
    force_pageable=pageable;
    Set_Hamiltonian_OTF.atom_gla_off=alloc(4*sizeof(size_t));
    Set_Hamiltonian_OTF.atom_gla_off[1]=0; Set_Hamiltonian_OTF.atom_gla_off[2]=31; Set_Hamiltonian_OTF.atom_gla_off[3]=78;
    for(int kind=0;kind<2;kind++) {
      SetHOTFPlan *p=SETH_OTF_GetPlan(kind,0,spins,20000,out_base,0);
      check(p && p->ready && p->batch_count>1,"multibatch plan");
      size_t prefix=0; int seen=0;
      for(int b=0;b<p->batch_count;b++) {
        SetHOTFBatch *bt=&p->batch[b];
        const int *no0=(const int*)(bt->image+bt->off_NO0), *no1=(const int*)(bt->image+bt->off_NO1);
        const size_t *hoff=(const size_t*)(bt->image+bt->off_h_off), *noff=(const size_t*)(bt->image+bt->off_nolg_off);
        const int *nc=(const int*)(bt->image+bt->off_Nc);
        size_t local=0;
        for(int q=0;q<bt->pair_count;q++) {
          int a=bt->pair_Mc_AN[q], h=bt->pair_h_AN[q], a1=natn[a][h];
          check(no0[q]==Spe_Total_NO[a] && no1[q]==Spe_Total_NO[a1],"orbital shape");
          check(hoff[q]==(kind?prefix:local),"matrix offset across batches");
          if(kind) check(((size_t*)(bt->image+bt->off_out_base))[q]==out_base[a],"output base");
          for(int k=0;k<NumOLG[a][h];k++) {
            check(nc[noff[q]+k]==glist[a][h][k],"point gather");
            if(!kind) check(((int*)(bt->image+bt->off_MN))[noff[q]+k]==mglist[a][glist[a][h][k]],"potential gather");
          }
          size_t n=(size_t)spins*Spe_Total_NO[a]*Spe_Total_NO[a1]; prefix+=n; local+=n; seen++;
        }
      }
      check(seen==9,"pair coverage");
      int before=alloc_count;
      check(SETH_OTF_GetPlan(kind,0,spins,20000,out_base,0)==p,"cache identity");
      check(alloc_count==before,"cache rebuild without changes");
      cases++;
    }
    Set_Hamiltonian_Invalidate_OpenACC_MatrixElements_Cache();
    check(!Set_Hamiltonian_OTF_Plan[0].ready && !Set_Hamiltonian_OTF_Plan[1].ready,"geometry invalidation");
    check(live_alloc==0,"cached host allocations leaked");
  }
  /* Eu-sized central spheres cannot fit 128 MiB.  Overlap lists deliberately
     contain gaps, repeats, and reverse order; chunking must preserve them. */
  for(int a=1;a<=3;a++) {
    GridN_Atom[a]=1691481;
    Spe_Total_NO[a]=41;
    for(int h=0;h<3;h++) {
      NumOLG[a][h]=53-h;
      for(int k=0;k<NumOLG[a][h];k++) glist[a][h][k]=(59-k*7+a+h+420)%60;
    }
  }
  NumOLG[2][1]=0; /* zero-overlap pair still advances the packed DM offset */
  for(int spins=1;spins<=4;spins*=2) for(int pageable=0;pageable<2;pageable++) {
    force_pageable=pageable;
    Set_Hamiltonian_OTF.atom_gla_off=alloc(4*sizeof(size_t));
    Set_Hamiltonian_OTF.atom_gla_off[1]=0;
    Set_Hamiltonian_OTF.atom_gla_off[2]=1691481;
    Set_Hamiltonian_OTF.atom_gla_off[3]=3382962;
    for(int kind=0;kind<2;kind++) {
      SetHOTFPairCost chunkcost;
      SETH_OTF_CompactCost(kind,spins,1,0,13,0,&chunkcost);
      const size_t payload=chunkcost.bytes;
      SetHOTFPlan *plan=SETH_OTF_GetPlan(kind,0,spins,payload,out_base,0);
      check(plan && plan->batch_count>9,"large spheres use bounded chunks");
      int nextpair=0, nextpoint=0, zero_seen=0;
      size_t dm_prefix=0;
      double sums[9]={0};
      for(int b=0;b<plan->batch_count;b++) {
        const SetHOTFBatch *bt=&plan->batch[b];
        check(bt->compact_orbs0 && bt->pair_count==1,"large pair stays on device");
        check(bt->first==nextpair && bt->last==nextpair+1,"original pair order");
        check(bt->first_nolg==nextpoint,"contiguous overlap chunk order");
        const int a=bt->pair_Mc_AN[0], h=bt->pair_h_AN[0];
        const int n=((int*)(bt->image+bt->off_NOLG))[0];
        const int *nc=(const int*)(bt->image+bt->off_Nc);
        const SOG_GpuPair *central=(const SOG_GpuPair*)(bt->image+bt->off_central_pairs);
        check(n<=13 && (n>0 || NumOLG[a][h]==0),"bounded chunk size");
        check(bt->total_orbs0_tiles==(size_t)n*41 && bt->total_orbs1==(size_t)n*41,"compact tile sizes");
        check(((size_t*)(bt->image+bt->off_h_off))[0]==(kind?dm_prefix:0),"same DM block across chunks");
        check(central->wan==a && central->no==41 && central->gx==Gxyz[a][1] &&
              central->ax==0 && central->ay==0 && central->az==0,"central descriptor");
        if(kind) check(((size_t*)(bt->image+bt->off_out_base))[0]==out_base[a],"absolute density output base");
        for(int k=0;k<n;k++) {
          check(nc[k]==glist[a][h][nextpoint+k],"absolute original overlap gather");
          if(!kind) {
            check(((int*)(bt->image+bt->off_orb_Nc))[k]==k,"compact orbital gather");
            check(((int*)(bt->image+bt->off_MN))[k]==mglist[a][nc[k]],"absolute potential gather");
          }
          sums[nextpair]+=(double)(nc[k]+1)*(double)(nextpoint+k+1);
        }
        /* Recreate the arena layout independently of cost estimates. */
        size_t pos=bt->image_bytes;
        pos=(pos+511)&~(size_t)511; pos+=sizeof(float)*bt->total_orbs0_tiles;
        pos=(pos+511)&~(size_t)511; pos+=sizeof(float)*bt->total_orbs1;
        if(!kind) { pos=(pos+511)&~(size_t)511; pos+=sizeof(double)*bt->total_h; }
        check(pos<=payload,"chunk arena within requested payload");
        nextpoint+=n;
        if(nextpoint==NumOLG[a][h]) {
          double expect=0;
          for(int k=0;k<NumOLG[a][h];k++) expect+=(double)(glist[a][h][k]+1)*(double)(k+1);
          check(sums[nextpair]==expect,"every overlap exactly once");
          if(!n) zero_seen++;
          dm_prefix+=(size_t)spins*41*41;
          nextpair++; nextpoint=0;
        }
      }
      check(nextpair==9 && zero_seen==1,"all large and empty pairs covered");
      cases++;
    }
    Set_Hamiltonian_Invalidate_OpenACC_MatrixElements_Cache();
    check(live_alloc==0,"large-pair cleanup");
  }
  {
    const size_t mib=1024ULL*1024ULL, dm_count=9*41*41, output_count=3*1691481;
    size_t need=0, avail=0, take=0;
    int grow=0;
    unsetenv("OPENMX_SETDEN_ONTHEFLY_MB");
    gpu_free=16ULL*1024*mib;
    take=Set_Hamiltonian_OnTheFly_DensityBudget(1,dm_count,output_count,1,&grow,&need,&avail,0);
    check(take==0 && grow==1 && Set_Hamiltonian_OTF_DenPlan.largest>128*mib,"large atom prefers growth");
    check(Set_Hamiltonian_OnTheFly_DensityGrow(0,1,&avail),"abundant-memory growth admitted");
    check(avail==Set_Hamiltonian_OTF_DenPlan.all,"full-batch growth retained");
    gpu_free=4ULL*1024*mib;
    take=Set_Hamiltonian_OnTheFly_DensityBudget(1,dm_count,output_count,18,&grow,&need,&avail,0);
    check(take==0 && grow==1,"18 large ranks request growth");
    check(Set_Hamiltonian_OnTheFly_DensityGrow(0,18,&avail),"18 large ranks all admit compact chunks");
    check(avail<Set_Hamiltonian_OTF_DenPlan.largest && avail>=need,"compact chunk below full atom size");
    check(Set_Hamiltonian_OTF_DenPlan.grown+Set_Hamiltonian_OTF_DenPlan.fixed+
          SETH_OTF_DEN_RUNTIME_SLACK <= (gpu_free-SETH_OTF_DEN_RESERVE_BYTES)/18,"grown workspace within physical share");
    setenv("OPENMX_SETDEN_ONTHEFLY_MB","32",1);
    take=Set_Hamiltonian_OnTheFly_DensityBudget(1,dm_count,output_count,18,&grow,&need,&avail,0);
    check(take>0 && grow==0 && Set_Hamiltonian_OTF_DenPlan.default_workspace<=32*mib,"explicit cap admits bounded chunks");
    gpu_free=512*mib+18*(SETH_OTF_DEN_RUNTIME_SLACK+Set_Hamiltonian_OTF_DenPlan.fixed+16*mib);
    take=Set_Hamiltonian_OnTheFly_DensityBudget(1,dm_count,output_count,18,&grow,&need,&avail,0);
    check(take>0 && grow==0 && Set_Hamiltonian_OTF_DenPlan.default_workspace==16*mib,"preferred floor does not exceed physical share");
    gpu_free=1024*mib;
    take=Set_Hamiltonian_OnTheFly_DensityBudget(1,dm_count,output_count,18,&grow,&need,&avail,0);
    check(take==0 && grow==-1 && avail==0,"insufficient physical share declines without unsigned underflow");
    gpu_query_error=1;
    take=Set_Hamiltonian_OnTheFly_DensityBudget(1,dm_count,output_count,18,&grow,&need,&avail,0);
    check(take==0 && grow==-1 && !Set_Hamiltonian_OTF_DenPlan.valid,"failed physical query declines");
    gpu_query_error=0;
    unsetenv("OPENMX_SETDEN_ONTHEFLY_MB");
  }
  SetHOTFPlan *p=SETH_OTF_GetPlan(0,0,1,1,out_base,0);
  check(p && p->batch_count==9,"oversized Hamiltonian pairs");
  for(int b=0;b<p->batch_count;b++) check(p->batch[b].pair_count==0 && !p->batch[b].image,"CPU fallback placeholder");
  check(SETH_OTF_GetPlan(1,0,1,1,out_base,0)==NULL,"oversized density rejection");
  Set_Hamiltonian_Invalidate_OpenACC_MatrixElements_Cache(); check(live_alloc==0,"fallback cleanup");
  printf("PASS: %d cached OTF plans, multibatch DM offsets, images, geometry invalidation, pinned/pageable cleanup, Eu-sized compact chunks, 18-rank physical budgets/growth, impossible-budget fallback\n",cases);
  return 0;
}
''')
with tempfile.TemporaryDirectory(prefix="openmx-otf-plan-") as tmp:
    cfile, binary = Path(tmp)/"check.c", Path(tmp)/"check"
    cfile.write_text("\n\n".join(parts))
    subprocess.run(shlex.split(os.environ.get("CC", "cc")) + ["-std=c11", "-O1", "-g", "-Wall", "-Wextra"] + shlex.split(os.environ.get("CFLAGS", "")) + [str(cfile), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True, timeout=30)

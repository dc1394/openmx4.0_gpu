#!/usr/bin/env python3
"""Compare shared CPU HVNA contraction with frozen local and halo assembly.

Check both product precisions, s/p/d and empty bases, l=0/2/14 and empty
projectors, rank partitions, poisoned receive slots, input ownership, threads,
and intersections whose first/all terms are absent.
"""
import os
from pathlib import Path
import resource
import shlex
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
CORE = Path(os.environ.get("OPENMX_CORE_LIBRARY", ROOT / "build/core-serial/libopenmx_core.so")).resolve()
if not CORE.is_file():
    raise SystemExit("Build the C++ core and set OPENMX_CORE_LIBRARY")
source = Path(os.environ.get("OPENMX_VNA_SOURCE", ROOT / "source/Set_ProExpn_VNA.c")).read_text()
assert '#include "openmx_hvna_cpu.h"' in source
assert "SetPro_HVNA_CPU_Halo(&hvna_geometry" in source and "SetPro_HVNA_CPU_Local(&hvna_geometry" in source
REFERENCE = (ROOT / "tests/fixtures/hvna_cpu_before_raii.c").read_text()

PREFIX = r'''
#include <assert.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "openmx_core_hvna_workspace.h"
#define NS 2
#define NG 6
#define NB 4
#define NO 16
#define NC 225
#define NL 15
#define NR 2
typedef OPENMX_TEST_DS_TYPE Type_DS_VNA;
static int Matomnum,fp32,Num_RVNA,List_YOUSO[40],VNA_List[NL*NR+2],VNA_List2[NL*NR+2];
static int M2G[NG+1],F_G2M[NG+1],WhatSpecies[NG+1],FNAN[NG+1],Spe_Total_NO[NS];
static int neighbor_storage[NG+1][NB],*natn[NG+1];
static int intersection_storage[NG+1][NB][NB],*intersection_rows[NG+1][NB],**RMI1[NG+1];
static int receive_global[1],*Rcv_GAN[1]={receive_global},F_Rcv_Num_WK[1]={0};
static double Spe_Atom_Cut1[NS]={3,4},distance_storage[NG+1][NB],*Dis[NG+1],time_per_atom[NG+1];
static double energy_storage[NS][NL][NR],*energy_rows[NS][NL],**VNA_proj_ene[NS];
static Type_DS_VNA values[NG+2][NB][NO][NC];
static Type_DS_VNA *ds_rows[NG+2][NB][NO],**ds_neighbors[NG+2][NB],***ds_atoms[NG+2],****ds_channels[1]={ds_atoms};
static double output[NG+1][NB][NO][NO],expected[NG+1][NB][NO][NO];
static double *out_rows[NG+1][NB][NO],**out_neighbors[NG+1][NB],***hvna[NG+1];
static void dtime(double *time){static _Thread_local double tick;*time=tick++;}
static double dampingF(double cutoff,double distance){return 1.0-distance/(2.0*cutoff);}
static void SetPro_gpu_abort(const char *error){fprintf(stderr,"%s\n",error);abort();}
#include "openmx_hvna_cpu.h"
'''

MAIN = r'''
static void reset_output(void){
  for(size_t i=0;i<sizeof(output)/sizeof(double);++i)((double*)output)[i]=-91;
  memset(time_per_atom,0,sizeof(time_per_atom));
}
static void receive(int global){memcpy(values[NG+1],values[global],sizeof(values[0]));}
static void poison(void){
  for(size_t i=0;i<sizeof(values[NG+1])/sizeof(Type_DS_VNA);++i)((Type_DS_VNA*)values[NG+1])[i]=NAN;
}
static void configure(int local){
  const int order[NG+1]={0,3,1,5,4,2,6};
  Matomnum=local;
  for(int a=1;a<=NG;++a){M2G[a]=order[a];F_G2M[order[a]]=a;}
  for(int g=0;g<=NG;++g){
    WhatSpecies[g]=(g%3==0?0:1);FNAN[g]=1+g%3;
    natn[g]=neighbor_storage[g];Dis[g]=distance_storage[g];
    for(int j=0;j<NB;++j){
      natn[g][j]=g?1+(g-1+j)%NG:0;Dis[g][j]=0.17*(g+j);
    }
  }
  for(int a=0;a<=NG;++a){
    hvna[a]=out_neighbors[a];RMI1[a]=intersection_rows[a];
    for(int j=0;j<NB;++j){
      out_neighbors[a][j]=out_rows[a][j];intersection_rows[a][j]=intersection_storage[a][j];
      for(int m=0;m<NO;++m)out_rows[a][j][m]=output[a][j][m];
      const int other=natn[M2G[a]][j];
      for(int k=0;k<NB;++k)RMI1[a][j][k]=k && (a+j+k)%4==0?-1:(j+k)%(FNAN[other]+1);
    }
  }
  for(int a=0;a<NG+2;++a){
    ds_atoms[a]=ds_neighbors[a];
    const int global=a==Matomnum+1?NG+1:(a<=NG?M2G[a]:0);
    for(int k=0;k<NB;++k){
      ds_neighbors[a][k]=ds_rows[a][k];
      for(int m=0;m<NO;++m)ds_rows[a][k][m]=values[global][k][m];
    }
  }
}
static void scalar_reference(const OpenMXHvnaProjectorBuffers *table){
  for(size_t i=0;i<sizeof(expected)/sizeof(double);++i)((double*)expected)[i]=-91;
  for(int mc=1;mc<=Matomnum;++mc){
    const int gc=M2G[mc],rows=table->orbitals[WhatSpecies[gc]];
    for(int j=0;j<=FNAN[gc];++j){
      const int other=natn[gc][j],columns=table->orbitals[WhatSpecies[other]];
      for(int m=0;m<rows;++m)for(int n=0;n<columns;++n){
        double total=0;
        for(int k=0;k<=FNAN[gc];++k){
          const int kl=RMI1[mc][j][k];if(kl<0)continue;
          const int species=WhatSpecies[natn[gc][k]];
          double sum=0;int component=0;
          for(int p=0;p<table->groups;++p){
            double dot=0;
            for(int q=0;q<table->lengths[p];++q,++component){
              Type_DS_VNA left=values[gc][k][m][component],right=values[other][kl][n][component];
              dot+=fp32?(double)(left*right):(double)left*(double)right;
            }
            sum+=table->energies[species*table->groups+p]*dot;
          }
          total+=sum;
        }
        expected[mc][j][m][n]=dampingF(Spe_Atom_Cut1[WhatSpecies[gc]]+Spe_Atom_Cut1[WhatSpecies[other]],Dis[gc][j])*total;
      }
    }
  }
}
static void execute(const SetProHvnaGeometry *geometry,const OpenMXHvnaProjectors *owner,
    const OpenMXHvnaProjectorBuffers *table,const OpenMXHvnaWorkspaceBuffers *work){
  for(int remote=NG;remote>Matomnum;--remote){
    receive(M2G[remote]);
    SetPro_HVNA_CPU_Halo(geometry,table,work,ds_channels,hvna,remote,M2G[remote],fp32);
    poison();
  }
  SetPro_HVNA_CPU_Local(geometry,owner,table,ds_channels,hvna,fp32);
}
int main(void){
  omp_set_dynamic(0);
  for(int a=0;a<NG+1;++a)for(int k=0;k<NB;++k)for(int m=0;m<NO;++m)for(int p=0;p<NC;++p)
    values[a][k][m][p]=(Type_DS_VNA)(0.11*sin(.7*a+.3*k+.17*m+.013*p)+.023*cos(.071*p));
  for(int s=0;s<NS;++s){
    VNA_proj_ene[s]=energy_rows[s];
    for(int l=0;l<NL;++l){energy_rows[s][l]=energy_storage[s][l];for(int r=0;r<NR;++r)energy_storage[s][l][r]=.13*s-.07*l+.031*r+.11;}
  }
  const int bases[4][NS][3]={{{2,0,0},{1,0,0}},{{2,1,2},{0,2,0}},{{0,0,0},{1,1,1}},{{0,0,0},{0,0,0}}};
  const int projects[4][2]={{0,1},{2,2},{14,1},{0,0}},threads[3]={1,2,8},partitions[4]={0,1,3,6};
  int cases=0,missing_cases=0,precision_sensitive=0;
  for(int basis=0;basis<4;++basis)for(int projection=0;projection<4;++projection)
  for(int split=0;split<3;++split)for(int partition=0;partition<4;++partition)for(fp32=0;fp32<2;++fp32){
    configure(partitions[partition]);
    int angular[NS]={2,2},counts[NS][3];int *count_rows[NS]={counts[0],counts[1]};
    for(int s=0;s<NS;++s){
      Spe_Total_NO[s]=0;for(int l=0;l<3;++l){counts[s][l]=bases[basis][s][l];Spe_Total_NO[s]+=counts[s][l]*(2*l+1);}
    }
    List_YOUSO[7]=NO+2;List_YOUSO[34]=projects[projection][1];List_YOUSO[35]=projects[projection][0];
    Num_RVNA=(List_YOUSO[35]+1)*List_YOUSO[34];
    for(int l=0,p=0;l<=List_YOUSO[35];++l)for(int r=0;r<List_YOUSO[34];++r,++p){VNA_List[p]=l;VNA_List2[p]=r;}
    const OpenMXVnaPreparationShape shape={2,2,List_YOUSO[35],List_YOUSO[34],1,NS,angular,count_rows};
    OpenMXVnaPreparationBuffers preparation;char error[256];
    OpenMXVnaPreparation *prep=openmx_core_vna_preparation_create(&shape,&preparation,error,sizeof(error));assert(prep);
    OpenMXHvnaProjectorBuffers table;OpenMXHvnaProjectors *owner=openmx_core_hvna_projectors_create(prep,VNA_proj_ene,&table,error,sizeof(error));assert(owner);
    openmx_core_vna_preparation_destroy(prep);
    const SetProHvnaGeometry geometry={Matomnum,M2G,F_G2M,WhatSpecies,FNAN,natn,RMI1,Spe_Atom_Cut1,Dis,time_per_atom};
    OpenMXHvnaWorkspaceBuffers work;OpenMXHvnaWorkspace *worker=SetPro_HVNA_CPU_Workspace(owner,&work);
    reset_output();omp_set_num_threads(1);
    for(int remote=NG;remote>Matomnum;--remote){receive(M2G[remote]);reference_hvna_halo(ds_channels,hvna,remote,M2G[remote]);poison();}
    reference_hvna_local(ds_channels,hvna);memcpy(expected,output,sizeof(expected));
    const double saved_energy=energy_storage[0][0][0];energy_storage[0][0][0]=999;
    List_YOUSO[7]=1;Spe_Total_NO[0]=Spe_Total_NO[1]=0;
    for(int p=0;p<Num_RVNA;++p)VNA_List[p]=VNA_List2[p]=-999;
    reset_output();omp_set_num_threads(threads[split]);
    execute(&geometry,owner,&table,&work);
    if(memcmp(output,expected,sizeof(output))){fprintf(stderr,"legacy mismatch basis=%d projectors=%d threads=%d local=%d fp32=%d\n",basis,projection,threads[split],Matomnum,fp32);abort();}
    for(int mc=1;mc<=Matomnum;++mc)assert(time_per_atom[M2G[mc]]==NG-Matomnum+1);
    energy_storage[0][0][0]=saved_energy;
    cases++;
    if(basis==1 && projection==1 && split==0 && partition==2 && sizeof(Type_DS_VNA)==sizeof(float)){
      scalar_reference(&table);assert(!memcmp(output,expected,sizeof(output)));
      double sample=output[1][0][0][0];
      int previous=fp32;fp32=!fp32;scalar_reference(&table);fp32=previous;
      if(sample!=expected[1][0][0][0])precision_sensitive=1;
    }
    /* Missing k=0 must not add to a previous pair; an entirely empty
       intersection has zero contraction, even with NaN-filled scratch. */
    if(split==0 && partition==2){
      for(int mc=1;mc<=Matomnum;++mc)for(int j=0;j<=FNAN[M2G[mc]];++j){
        RMI1[mc][j][0]=-1;
        if(j==0)for(int k=0;k<NB;++k)RMI1[mc][j][k]=-1;
      }
      for(size_t i=0;i<work.orbital_capacity*work.orbital_capacity;++i)work.matrix[i]=NAN;
      scalar_reference(&table);reset_output();execute(&geometry,owner,&table,&work);
      for(size_t i=0;i<sizeof(output)/sizeof(double);++i)assert(isfinite(((double*)output)[i]) && fabs(((double*)output)[i]-((double*)expected)[i])<1e-12);
      missing_cases++;
    }
    openmx_core_hvna_workspace_destroy(worker);openmx_core_hvna_projectors_destroy(owner);
  }
  assert(sizeof(Type_DS_VNA)!=sizeof(float) || precision_sensitive);
  printf("PASS: %d exact legacy HVNA cases and %d missing-intersection cases (%zu-byte DS storage); ownership, padding, empty ranks and timing\n",cases,missing_cases,sizeof(Type_DS_VNA));
}
'''

with tempfile.TemporaryDirectory(prefix="openmx-hvna-cpu-") as directory:
    folder = Path(directory)
    src = folder / "check.c"
    src.write_text(PREFIX + REFERENCE + MAIN)
    for storage in ("float", "double"):
        executable = folder / storage
        command = shlex.split(os.environ.get("CC", "cc")) + [
            "-std=c11", "-O1", "-g", "-fopenmp", "-ffp-contract=off",
            "-fsanitize=address,undefined", "-fno-sanitize-recover=all",
            "-DOPENMX_TEST_DS_TYPE=" + storage, "-I" + str(ROOT / "source"),
            str(src), str(CORE), "-Wl,-rpath," + str(CORE.parent), "-lm", "-o", str(executable)]
        subprocess.run(command, check=True)
        def sanitizer_stack():
            soft, hard = resource.getrlimit(resource.RLIMIT_STACK)
            if soft == resource.RLIM_INFINITY:
                resource.setrlimit(resource.RLIMIT_STACK, (8 * 1024 * 1024, hard))
        subprocess.run([str(executable)], check=True, timeout=100, preexec_fn=sanitizer_stack)

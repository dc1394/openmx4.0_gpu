#!/usr/bin/env python3
"""Exercise production density preparation with real MPI and host CUDA stubs.

No GPU is used. Four ranks take different resident/OTF/empty/failure paths;
the extracted production builders must complete matching collectives. This
does not validate CUDA kernels or physical density values.
"""
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def function(source, name):
    match = re.search(r"^[^\n;]*\b" + re.escape(name) + r"\([^;]*?\)\n\{", source, re.M)
    if not match:
        raise RuntimeError(f"function not found: {name}")
    start = match.start()
    pos = match.end()
    depth = 1
    while depth:
        if source[pos] == "{":
            depth += 1
        elif source[pos] == "}":
            depth -= 1
        pos += 1
    return source[start:pos]


def typedef(source, name):
    end = source.index("} " + name + ";") + len("} " + name + ";")
    return source[source.rfind("typedef struct {", 0, end):end]


source = (ROOT / "source/Set_Density_Grid_GPU.c").read_text()
header = (ROOT / "source/openmx_common.h").read_text()
prefix = r'''
#define _POSIX_C_SOURCE 200809L
#include <mpi.h>
#include <stdint.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define GPUSOLVER 1
#define cudaSuccess 0
#define acc_device_nvidia 0
typedef float Type_Orbs_Grid;
static MPI_Comm mpi_comm_level1;
static int SpinP_switch=0, Solver=3, Cnt_switch=0, scf_eigen_lib_flag=GPUSOLVER;
static int Matomnum=1, M2G[5]={0,1,2,3,4}, GridN_Atom[5]={0,2,2,2,2};
static int rank_id, resident=1, fail_alloc, density_ranks;
static size_t registered_need;
static int split_calls, reduce_calls;
static int test_split_type(MPI_Comm comm,int split,int key,MPI_Info info,MPI_Comm *out) {
  split_calls++;
  return PMPI_Comm_split_type(comm,split,key,info,out);
}
static int test_allreduce(const void *send,void *recv,int n,MPI_Datatype type,MPI_Op op,MPI_Comm comm) {
  reduce_calls++;
  return PMPI_Allreduce(send,recv,n,type,op,comm);
}
static int cudaGetDeviceCount(int *p) { *p=1; return 0; }
static int acc_get_num_devices(int x) { (void)x; return 1; }
static int acc_get_device_num(int x) { (void)x; return 0; }
static int gpu_rank_device_usable(void) { return 1; }
static int OpenMX_GpuMemGetInfo(size_t *f,size_t *t) { *f=*t=(size_t)16*1024*1024*1024; return 0; }
static void OpenMX_GpuPhaseNeed_Register(const char *n,size_t b) { (void)n; registered_need=b; }
static int Set_Hamiltonian_MatrixElementsTables_Ready(int k) { (void)k; return resident; }
static int Set_Hamiltonian_OnTheFly_DensityPossible(int k,int r) { (void)k; (void)r; return !resident; }
static size_t Set_Hamiltonian_MatrixElements_TotalH(int k,int r) { (void)k; (void)r; return 1; }
static void Set_Hamiltonian_OnTheFly_SetDensityRanks(int r) { density_ranks=r; }
static void check(int ok,const char *what) {
  if (!ok) { fprintf(stderr,"rank %d: %s\n",rank_id,what); MPI_Abort(mpi_comm_level1,1); }
}
'''
stubs = r'''
static SDGLocalContext SDG_local;
static void SDG_local_delete_device(SDGLocalContext *c) { c->device_resident=0; }
static int Set_Hamiltonian_GetMatrixElementsTables(int k,SetHamiltonianMETables *t) {
  static int one[1]={1}, two[1]={2}, nc[2]={0,1};
  static size_t zero[1]={0};
  static float orbs[2]={1,2};
  (void)k;
  memset(t,0,sizeof(*t));
  t->pair_count=1; t->total_h=1; t->total_nolg=2; t->total_orbs0=t->total_orbs1=2;
  t->pair_Mc_AN=t->pair_NO0=t->pair_NO1=one; t->pair_NOLG=two; t->nolg_Nc=nc;
  t->pair_h_offset=t->pair_nolg_offset=t->pair_orbs0_offset=t->pair_orbs1_offset=zero;
  t->orbs0buf=t->orbs1buf=orbs;
  t->orbs0_resident=t->orbs1_resident=t->nolg_resident=t->meta_resident=resident;
  return 1;
}
static void *test_malloc(size_t n) { return fail_alloc ? NULL : malloc(n); }
'''
names = ["SDG_env_bool", "SDG_env_mib", "SDG_add_bytes", "SDG_malloc", "SDG_solver_supported",
         "SDG_local_free", "SDG_local_build", "SDG_local_call_bytes", "SDG_local_prepare",
         "Set_Density_Grid_GPU_Local_Prepare"]
parts = [prefix, typedef(header, "SetHamiltonianMETables"), typedef(source, "SDGLocalContext"), stubs]
for name in names:
    text = function(source, name)
    if name == "SDG_malloc":
        text = text.replace("return malloc(", "return test_malloc(")
    text = text.replace("MPI_Comm_split_type(", "test_split_type(")
    text = text.replace("MPI_Allreduce(", "test_allreduce(")
    parts.append(text)
parts.append(r'''
int main(int argc,char **argv) {
  int n, ok, all_ok, rounds=0;
  MPI_Init(&argc,&argv); mpi_comm_level1=MPI_COMM_WORLD;
  MPI_Comm_rank(mpi_comm_level1,&rank_id); MPI_Comm_size(mpi_comm_level1,&n);
  check(n==4,"run with 4 ranks");
  for(int step=0;step<7;step++) {
    scf_eigen_lib_flag=GPUSOLVER;
    setenv("OPENMX_DENSITY_GRID_GPU_LOCAL","1",1);
    Matomnum=rank_id==2 ? 0:1;
    resident=rank_id!=1; fail_alloc=0;
    if(step==1 && rank_id==3) SDG_local_free(); /* one fresh CSR among caches */
    if(step==2 && rank_id==3) { SDG_local_free(); fail_alloc=1; }
    if(step==3) {
      if(rank_id==3) SDG_local_free();
      if(rank_id==1) setenv("OPENMX_DENSITY_GRID_GPU_LOCAL","0",1);
    }
    if(step==4) setenv("OPENMX_DENSITY_GRID_GPU_LOCAL","2",1); /* OTF -> CSR */
    if(step==5 && rank_id==1) {
      SDG_local_free(); Matomnum=3;
      GridN_Atom[1]=GridN_Atom[2]=GridN_Atom[3]=INT_MAX;
    }
    if(step==6) { scf_eigen_lib_flag=0; density_ranks=7; registered_need=1234; }
    const int splits_before=split_calls, reductions_before=reduce_calls;
    ok=Set_Density_Grid_GPU_Local_Prepare(0,0);
    if(step==6) {
      check(split_calls==splits_before && reduce_calls==reductions_before,"CPU path entered collectives");
      check(density_ranks==0 && registered_need==0,"CPU path retained GPU budget hints");
    }
    else {
      check(split_calls==splits_before+1 && reduce_calls==reductions_before+2,"GPU collective participation");
    }
    MPI_Allreduce(&ok,&all_ok,1,MPI_INT,MPI_MIN,mpi_comm_level1);
    check(all_ok==(step==0 || step==1 || step==4),"wrong collective eligibility");
    check(ok==!((step==2 && rank_id==3) || (step==3 && rank_id==1) ||
                 (step==5 && rank_id==1) || step==6), "unexpected local preparation");
    if(step==0 || step==1) {
      check(density_ranks==1,"wrong OTF rank count");
      check(registered_need>0,"CSR phase need missing");
      if(rank_id==0 || rank_id==3) {
        check(SDG_local.ready && SDG_local.output_count==2,"CSR not ready");
        check(SDG_local.out_ptr[0]==0 && SDG_local.out_ptr[1]==1 && SDG_local.out_ptr[2]==2,"CSR offsets");
        check(SDG_local.term_pt[0]==0 && SDG_local.term_pt[1]==1,"CSR payload");
      }
    }
    if(step==5 && rank_id==1) check(!SDG_local.otf,"overflow retained OTF state");
    rounds++;
  }
  SDG_local_free();
  if(rank_id==0) printf("PASS: %d real-MPI rounds, mixed CSR/OTF, empty/cached ranks, allocation failure, mode switch, offset overflow and CPU collective bypass\n",rounds);
  MPI_Finalize(); return 0;
}
''')
env = os.environ.copy()
env.pop("LD_LIBRARY_PATH", None)
env.pop("OPAL_PREFIX", None)
env.setdefault("OMPI_CC", "gcc")
with tempfile.TemporaryDirectory(prefix="openmx-density-mpi-") as tmp:
    cfile, binary = Path(tmp)/"check.c", Path(tmp)/"check"
    cfile.write_text("\n\n".join(parts))
    subprocess.run(shlex.split(env.get("MPICC", "mpicc")) + ["-std=c11", "-O1", "-g", "-Wall", "-Wextra", str(cfile), "-o", str(binary)], env=env, check=True)
    subprocess.run(shlex.split(env.get("MPIEXEC", "mpirun")) + ["--oversubscribe", "--bind-to", "none", "-np", "4", str(binary)], env=env, timeout=30, check=True)

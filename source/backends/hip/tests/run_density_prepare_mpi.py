#!/usr/bin/env python3
"""Exercise the density's local-mode preparation with real MPI and host stubs.

The AMD port of the NVIDIA test of the same name.  The production functions
of source/Set_Density_Grid_GPU.c (SDG_local_build, SDG_local_prepare, the
public Set_Density_Grid_GPU_Local_Prepare and their helpers) are extracted
verbatim and built against host stubs of HIP, OpenMP, the device helpers
and Set_Hamiltonian (its matrix-element tables and the on-the-fly density
gate).  No GPU is used.  Four ranks take different paths in every step:
ranks 0 and 3 the CSR kernel (resident tables), rank 1 the on-the-fly
tiles, rank 2 an empty partition whose device is not usable (it must not
veto the others, as the NVIDIA version's Matomnum == 0 gate allows).  The
steps add a fresh CSR among cached ones, an allocation failure, a rank
with the local mode off, the forced-CSR mode (OTF -> CSR switch, and a
rank without resident tables declining), an offset overflow, a CPU
eigensolver, an on-the-fly-ineligible rank, a workspace shortage, final density
table reuse (fresh and cached CSR/OTF), and optimized-orbital rejection.
Checked: every rank enters
exactly one collective (an MPI_Allreduce) per GPU step and none in the CPU
step, never MPI_Comm_split_type (the budget's denominator is the ranks per
device, without communication), the per-rank results, the agreed result,
the CSR contents, the published phase need, the workspace admission census
and reporting without collectives at stdout levels 0..3.  This does not validate
kernels or density values.

Run: python3 tests/run_density_prepare_mpi.py [--mpicc MPICC] [--mpirun MPIRUN]
(defaults: $MPICC/$MPIEXEC, else the Open MPI of the OpenMX build when it
exists, else mpicc/mpirun from PATH; OMPI_CC=gcc unless set).
"""
import argparse
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
MPI_HOME = Path(os.environ.get("MPI_HOME", "/home/kawai/opt/openmpi-rocmclang"))


def function(source, name):
    match = re.search(r"^[^\n;]*\b" + re.escape(name) + r"\([^;]*?\)\n\{", source, re.M)
    if not match:
        raise RuntimeError(f"function not found: {name}")
    pos = match.end()
    depth = 1
    while depth:
        if source[pos] == "{":
            depth += 1
        elif source[pos] == "}":
            depth -= 1
        pos += 1
    return source[match.start():pos]


def typedef(source, name):
    end = source.index("} " + name + ";") + len("} " + name + ";")
    return source[source.rfind("typedef struct {", 0, end):end]


PREFIX = r'''
#define _POSIX_C_SOURCE 200809L
#include <mpi.h>
#include <stdint.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define GPUSOLVER 1
#define hipSuccess 0
typedef float Type_Orbs_Grid;
static MPI_Comm mpi_comm_level1;
static int SpinP_switch=0, Solver=3, Cnt_switch=0, scf_eigen_lib_flag=GPUSOLVER;
static int Matomnum=1, M2G[5]={0,1,2,3,4}, GridN_Atom[5]={0,2,2,2,2};
static int rank_id, tables=1, resident=1, otf_possible=1, device_usable=1, fail_alloc;
static int otf_calls;
static int level_stdout=3, Host_ID=0, workspace_ok=1;
static int Set_Density_Grid_OTF_Prepare(int k,size_t *need,size_t *avail) {
  *need=1024; *avail=workspace_ok ? 2048 : 512; return k==0 && workspace_ok;
}
static size_t registered_need = 12345;
static int split_calls, reduce_calls;
static int test_split_type(MPI_Comm comm,int split,int key,MPI_Info info,MPI_Comm *out) {
  split_calls++;
  return PMPI_Comm_split_type(comm,split,key,info,out);
}
static int test_allreduce(const void *send,void *recv,int n,MPI_Datatype type,MPI_Op op,MPI_Comm comm) {
  reduce_calls++;
  return PMPI_Allreduce(send,recv,n,type,op,comm);
}
static int hipGetDeviceCount(int *p) { *p=1; return 0; }
static int hipGetDevice(int *p) { *p=0; return 0; }
static int omp_get_num_devices(void) { return 1; }
static int openmx_gpu_is_apu(void) { return 0; }
static int gpu_rank_device_usable(void) { return device_usable; }
static int openmx_gpu_ranks_per_device_noncollective(void) { return 4; }
static int OpenMX_GpuMemGetInfo(size_t *f,size_t *t) { *f=*t=(size_t)16*1024*1024*1024; return 0; }
static void OpenMX_GpuPhaseNeed_Register(const char *n,size_t b) { (void)n; registered_need=b; }
static int Set_Hamiltonian_MatrixElementsTables_Ready(int k) { return k==0 && tables; }
static int Set_Density_Grid_OTF_Possible(int k) { otf_calls++; return k==0 && otf_possible; }
static void check(int ok,const char *what) {
  if (!ok) { fprintf(stderr,"rank %d: %s\n",rank_id,what); MPI_Abort(mpi_comm_level1,1); }
}
'''

STUBS = r'''
static SDGLocalContext SDG_local;
static void SDG_local_delete_device(SDGLocalContext *c) {
  c->d_out_ptr=c->d_term_pt=c->d_pt_pair=NULL; c->d_dm=c->d_tmpden=NULL;
}
static int Set_Hamiltonian_GetMatrixElementsTables(int k,SetHamiltonianMETables *t) {
  static int one[1]={1}, two[1]={2}, nc[2]={0,1};
  static size_t zero[1]={0};
  static float orbs[2]={1,2};
  check(k==0,"final density requested different orbital tables");
  memset(t,0,sizeof(*t));
  if (!tables) return 0;
  t->pair_count=1; t->total_h=1; t->total_nolg=2; t->total_orbs0=t->total_orbs1=2;
  t->pair_Mc_AN=t->pair_NO0=t->pair_NO1=one; t->pair_NOLG=two; t->nolg_Nc=nc;
  t->pair_h_offset=t->pair_nolg_offset=t->pair_orbs0_offset=t->pair_orbs1_offset=zero;
  t->orbs0buf=t->orbs1buf=orbs;
  t->orbs0_resident=t->orbs1_resident=t->nolg_resident=t->meta_resident=resident;
  return 1;
}
static void *test_malloc(size_t n) { return fail_alloc ? NULL : malloc(n); }
'''

NAMES = ["SDG_env_bool", "SDG_env_mib", "SDG_gpu_default_enabled", "SDG_add_bytes", "SDG_malloc",
         "SDG_solver_supported", "SDG_local_free", "SDG_local_build", "SDG_local_call_bytes",
         "SDG_local_prepare", "Set_Density_Grid_GPU_Local_Prepare", "Set_Density_Grid_GPU_Local_OnTheFly", "Set_Density_Grid_GPU_Local_Report"]

MAIN = r'''
/* the role of every rank in the ordinary steps */
static void roles(void) {
  Matomnum = rank_id==2 ? 0 : 1;
  device_usable = rank_id!=2;            /* the empty partition cannot use its device */
  tables = 1;
  resident = rank_id!=1;                 /* rank 1 has host tables only -> on the fly */
  otf_possible = 1;
  fail_alloc = 0;
}

int main(int argc,char **argv) {
  int n, ok, all_ok, rounds=0;
  /* expected agreed result and this rank's result per step */
  const int expect_all[16]   = {1,1,0,0,1,0,0,0,0,0,1,1,1,0,0,0};
  MPI_Init(&argc,&argv); mpi_comm_level1=MPI_COMM_WORLD;
  MPI_Comm_rank(mpi_comm_level1,&rank_id); MPI_Comm_size(mpi_comm_level1,&n);
  check(n==4,"run with 4 ranks");
  for(int step=0;step<16;step++) {
    int expect_ok = 1, cnt_kind = 0;
    Cnt_switch=0; workspace_ok=1;
    const int otf_before = otf_calls;
    scf_eigen_lib_flag=GPUSOLVER;
    setenv("OPENMX_DENSITY_GRID_GPU_LOCAL","1",1);
    roles();
    if(step==1 && rank_id==3) SDG_local_free();                        /* one fresh CSR among caches */
    if(step==2 && rank_id==3) { SDG_local_free(); fail_alloc=1; expect_ok=0; }
    if(step==3) {
      if(rank_id==3) SDG_local_free();                                  /* clears the failure */
      if(rank_id==1) { setenv("OPENMX_DENSITY_GRID_GPU_LOCAL","0",1); expect_ok=0; }
    }
    if(step==4) {                                                       /* forced CSR, OTF -> CSR */
      setenv("OPENMX_DENSITY_GRID_GPU_LOCAL","2",1);
      if(rank_id==1) resident=1;
    }
    if(step==5) {                                                       /* forced CSR declines */
      setenv("OPENMX_DENSITY_GRID_GPU_LOCAL","2",1);
      if(rank_id==1) { SDG_local_free(); expect_ok=0; }
    }
    if(step==6 && rank_id==3) {                                         /* offset overflow */
      SDG_local_free(); Matomnum=3;
      GridN_Atom[1]=GridN_Atom[2]=GridN_Atom[3]=INT_MAX;
      expect_ok=0;
    }
    if(step==7) { scf_eigen_lib_flag=0; registered_need=1234; expect_ok=0; }
    if(step==8) {
      GridN_Atom[1]=GridN_Atom[2]=GridN_Atom[3]=2;
      if(rank_id==3) SDG_local_free();
      if(rank_id==1) { otf_possible=0; expect_ok=0; }                   /* neither tables nor OTF */
    }
    if(step==9 && rank_id==1) { workspace_ok=0; expect_ok=0; }
    if(step==10) { cnt_kind=1; if(rank_id==3) SDG_local_free(); }
    if(step==11) { cnt_kind=1; resident=1; }
    if(step==12) { cnt_kind=1; resident=1; fail_alloc=1; }
    if(step==13) { cnt_kind=1; Cnt_switch=1; expect_ok=0; }
    if(step==14) { cnt_kind=2; expect_ok=0; }
    if(step==15) { cnt_kind=1; scf_eigen_lib_flag=0; expect_ok=0; }
    level_stdout=step%4;
    const int splits_before=split_calls, reductions_before=reduce_calls;
    ok=Set_Density_Grid_GPU_Local_Prepare(cnt_kind,0);
    check(split_calls==splits_before,"no MPI_Comm_split_type");
    if(step==7 || step==15) {
      check(reduce_calls==reductions_before,"CPU path entered a collective");
      check(registered_need==0,"CPU path kept a GPU phase need");
    }
    else {
      check(reduce_calls==reductions_before+1,"one collective per GPU step on every rank");
    }
    MPI_Allreduce(&ok,&all_ok,1,MPI_INT,MPI_MIN,mpi_comm_level1);
    if(ok!=expect_ok) { fprintf(stderr,"rank %d step %d: ok %d, expected %d\n",rank_id,step,ok,expect_ok); }
    check(ok==expect_ok,"unexpected local preparation");
    check(all_ok==expect_all[step],"wrong agreed eligibility");
    if(step==9) {
      check(SDG_census.atom_ranks==3 && SDG_census.declined==1 && SDG_census.workspace_declined==1,"workspace census differs across ranks");
      const int before=reduce_calls;
      /* Hidden reports at 0/1 must not consume the once-on-change notice.
         Level 2 prints it once; unchanged level 3 must not repeat it. */
      for(level_stdout=0;level_stdout<=3;level_stdout++) {
        Set_Density_Grid_GPU_Local_Report(2,1);
        Set_Density_Grid_GPU_Local_Report(2,1);
      }
      level_stdout=3;
      Set_Density_Grid_GPU_Local_Report(1,0);
      Set_Density_Grid_GPU_Local_Report(1,0);
      Set_Density_Grid_GPU_Local_Report(2,0);
      Set_Density_Grid_GPU_Local_Report(2,0);
      check(reduce_calls==before,"reporting entered a collective");
    }
    if(rank_id==2) check(ok==(step!=7 && step<13),"wrong empty-partition eligibility");
    if(ok && Matomnum>0) check(SDG_local.cnt_kind==0,"wrong cached orbital kind");
    if(step>=10 && step<=12 && Matomnum>0) {
      if(rank_id==1 && step==10) check(SDG_local.otf,"final OTF density missing");
      else check(SDG_local.ready && !SDG_local.otf,"final CSR density missing");
    }
    if(step==0 || step==1 || step==4) {
      const int csr = !(rank_id==1 && step!=4) && rank_id!=2;
      if(csr) {
        check(SDG_local.ready && !SDG_local.otf && !Set_Density_Grid_GPU_Local_OnTheFly(),"CSR not ready");
        check(SDG_local.output_count==2 && SDG_local.node_ranks==4,"CSR size or budget denominator");
        check(SDG_local.out_ptr[0]==0 && SDG_local.out_ptr[1]==1 && SDG_local.out_ptr[2]==2,"CSR offsets");
        check(SDG_local.term_pt[0]==0 && SDG_local.term_pt[1]==1 && SDG_local.pt_pair[1]==0,"CSR payload");
      }
      if(rank_id==1 && step!=4) check(SDG_local.otf && !SDG_local.ready && Set_Density_Grid_GPU_Local_OnTheFly(),"OTF state");
      /* ranks 0 and 3 (and 1 in step 4) publish their CSR bytes */
      {
        unsigned long long mine = (SDG_local.ready && !SDG_local.otf) ? SDG_local.extra_bytes : 0, sum = 0;
        MPI_Allreduce(&mine,&sum,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,mpi_comm_level1);
        check(registered_need==sum && sum>0,"published CSR need");
      }
    }
    if(step==2 && rank_id==3) check(SDG_local.unavailable && !SDG_local.ready,"allocation failure marks the rank");
    if(step==4 && rank_id==1) check(otf_calls==otf_before,"forced CSR asked for the on-the-fly tiles");
    if(step==5 && rank_id==1) check(otf_calls==otf_before && !SDG_local.ready,"forced CSR took a non-resident rank");
    if(step==6 && rank_id==3) check(SDG_local.unavailable && !SDG_local.otf,"overflow marks the rank");
    rounds++;
  }
  SDG_local_free();
  if(rank_id==0) printf("PASS: %d real-MPI rounds: mixed CSR/OTF, empty rank without a device, cached and fresh CSR, allocation failure, local mode off, forced CSR (switch and decline), offset overflow, CPU collective bypass, ineligible rank, workspace admission and census at stdout levels 0..3, final density table reuse and optimized-orbital rejection\n",rounds);
  MPI_Finalize(); return 0;
}
'''


def tool(env_name, name):
    if os.environ.get(env_name):
        return os.environ[env_name]
    local = MPI_HOME / "bin" / name
    return str(local) if local.exists() else (shutil.which(name) or name)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--mpicc", default=tool("MPICC", "mpicc"))
    parser.add_argument("--mpirun", default=tool("MPIEXEC", "mpirun"))
    args = parser.parse_args()

    source = (ROOT / "source/Set_Density_Grid_GPU.c").read_text()
    header = (ROOT / "source/openmx_common.h").read_text()
    census = re.search(r"static struct \{\n  int atom_ranks, declined, workspace_declined;\n\} SDG_census = \{0\};", source).group(0)
    parts = [PREFIX, typedef(header, "SetHamiltonianMETables"), typedef(source, "SDGLocalContext"), STUBS, census]
    for name in NAMES:
        text = function(source, name)
        if name == "SDG_malloc":
            text = text.replace("return malloc(", "return test_malloc(")
        text = text.replace("MPI_Comm_split_type(", "test_split_type(")
        text = text.replace("MPI_Allreduce(", "test_allreduce(")
        parts.append(text)
    parts.append(MAIN)

    env = os.environ.copy()
    env.setdefault("OMPI_CC", "gcc")
    with tempfile.TemporaryDirectory(prefix="openmx-density-mpi-") as tmp:
        cfile, binary = Path(tmp) / "check.c", Path(tmp) / "check"
        cfile.write_text("\n\n".join(parts))
        subprocess.run(shlex.split(args.mpicc) + ["-std=c11", "-O1", "-g", "-Wall", "-Wextra",
                                                  "-Wno-unused-function", str(cfile), "-o", str(binary)],
                       env=env, check=True)
        run = subprocess.run(shlex.split(args.mpirun) + ["--oversubscribe", "--bind-to", "none", "-np", "4", str(binary)],
                             env=env, timeout=60, check=True, capture_output=True, text=True)
        expected = [
            "<Set_Density_Grid> CPU loop on 2 of 3 rank(s) with atoms (1 declined local GPU preparation, 1 lacked on-the-fly workspace, 1 failed on the device)",
            "<Set_Density_Grid> 1 of 3 rank(s) with atoms lack on-the-fly workspace; the one-rank GPU service integrates the density",
            "<Set_Density_Grid> CPU loop on 1 of 3 rank(s) with atoms (1 declined local GPU preparation, 1 lacked on-the-fly workspace, 0 failed on the device)",
        ]
        reports = [line for line in run.stdout.splitlines() if line.startswith("<Set_Density_Grid>")]
        if reports != expected:
            raise AssertionError(f"Unexpected fallback reports: {reports!r}\n{run.stdout}\n{run.stderr}")
        print(run.stdout, end="")
        print(run.stderr, end="")


if __name__ == "__main__":
    main()

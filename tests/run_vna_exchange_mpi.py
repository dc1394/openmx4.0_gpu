#!/usr/bin/env python3
"""Compare the production VNA ring against the frozen legacy protocol.

Use real MPI, asymmetric atom ownership, empty ranks and empty projector/basis
blocks. Compare every transfer event and halo value, poisoning the receive
slot after each consumption. Both archive callbacks and CPU consumption run
with float/double storage and one/four ranks, without GPU initialization.
"""
import os
from pathlib import Path
import resource
import shlex
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
CORE = Path(os.environ["OPENMX_CORE_LIBRARY"]).resolve(strict=True)
compiler = shlex.split(os.environ.get("MPICC", shutil.which("mpicc") or "mpicc"))
launcher = os.environ.get("MPIEXEC") or str(Path(shutil.which(compiler[0]) or compiler[0]).parent / "mpirun")
assembly = (ROOT / "source/openmx_vna_assembly.h").read_text()
start = assembly.index("  int pending;\n  do {")
finish = assembly.index("  openmx_core_hvna_workspace_destroy(halo_owner);", start)
production = assembly[start:finish]
reference = (ROOT / "tests/fixtures/vna_exchange_before_raii.c").read_text()
# The frozen loop keeps its pointer-row callback; production now passes flat
# views. Both callbacks independently inspect the same received payload.
reference = reference.replace("SetPro_HVNA_CPU_Halo(", "reference_consume_halo(")
PREFIX = r'''
#include "openmx_projector_exchange.h"
#include <assert.h>
#include <limits.h>
#include <math.h>
#include <string.h>
#define NG 12
#define NA 4
#define NH 4
#define NO 3
#define NP 8
typedef TEST_DS_TYPE Type_DS_VNA;
#include "openmx_projector_matrix.h"
static MPI_Comm mpi_comm_level1;
static MPI_Datatype MPI_Type_DS_VNA;
static int Matomnum,M2G[NA+1],WhatSpecies[NG+1],FNAN[NG+1],Spe_Total_NO[2],List_YOUSO[40];
static int F_Snd_Num[4],F_Rcv_Num[4],F_TopMAN[4],F_Snd_Num_WK[4],F_Rcv_Num_WK[4];
static int send_local[4][NA],send_global[4][NA],receive_global[4][NA];
static int *Snd_MAN[4],*Snd_GAN[4],*Rcv_GAN[4],natn[NG+1][NH];
static Type_DS_VNA values[NA+2][NH][NO][NP],*rows[NA+2][NH][NO],**neighbors[NA+2][NH],***atoms[NA+2],****channels[1]={atoms};
static size_t matrix_atoms[NA+3],matrix_blocks[(NA+2)*NH+1];
static int matrix_shapes[2*(NA+2)*NH];
static OpenMXProjectorMatrix ds_matrix;
static OpenMXVnaMatrixViews matrix_views;
static int trace[NG+1],trace_size,events[1024][3],event_size,active_components;
static void event(int kind,int peer,int count) {
  assert(event_size<1024);events[event_size][0]=kind;events[event_size][1]=peer;events[event_size++][2]=count;
}
static int tracked_send(const void *p,int n,MPI_Datatype type,int peer,int tag,MPI_Comm comm,MPI_Request *request) {
  event(type==MPI_INT?1:2,peer,n);return MPI_Isend(p,n,type,peer,tag,comm,request);
}
static int tracked_receive(void *p,int n,MPI_Datatype type,int peer,int tag,MPI_Comm comm,MPI_Status *status) {
  event(type==MPI_INT?3:4,peer,n);return MPI_Recv(p,n,type,peer,tag,comm,status);
}
static int tracked_wait(MPI_Request *request,MPI_Status *status) {
  event(5,0,0);return MPI_Wait(request,status);
}
#define MPI_Isend tracked_send
#define MPI_Recv tracked_receive
#define MPI_Wait tracked_wait
static Type_DS_VNA expected(int g,int h,int row,int column) { return (Type_DS_VNA)(g*1000+h*100+row*10+column); }
static void consume(int original,int gc) {
  (void)original;assert(trace_size<NG+1);trace[trace_size++]=gc;
  for(int h=0;h<=FNAN[gc];++h)
    for(int i=0;i<Spe_Total_NO[WhatSpecies[gc]];++i)
      for(int j=0;j<active_components;++j)assert(values[Matomnum+1][h][i][j]==expected(gc,h,i,j));
  for(size_t i=0;i<sizeof(values[0])/sizeof(Type_DS_VNA);++i)((Type_DS_VNA*)values[Matomnum+1])[i]=NAN;
}
static void SetPro_HVNA_GpuArchiveHalo(const void *g,Type_DS_VNA *****ds,int original,int gc) {(void)g;(void)ds;consume(original,gc);}
static void SetPro_HVNA_DispatchArchive(const void *g,Type_DS_VNA *****ds,int original,int gc) {SetPro_HVNA_GpuArchiveHalo(g,ds,original,gc);}
static void reference_consume_halo(const void *geometry,const void *projectors,const void *workspace,
    Type_DS_VNA *****ds,double ****hvna,int original,int gc,int fp32) {
  (void)geometry;(void)projectors;(void)workspace;(void)ds;(void)hvna;(void)fp32;consume(original,gc);
}
'''
FLAT_CALLBACK = r'''
static void SetPro_HVNA_CPU_Halo(const void *geometry,const void *projectors,const void *workspace,
    const OpenMXProjectorMatrix *ds,const OpenMXDoubleBlockMatrix *hvna,int original,int gc,int fp32) {
  (void)geometry;(void)projectors;(void)workspace;(void)hvna;(void)fp32;
  for(int h=0;h<=FNAN[gc];++h)
    for(int i=0;i<Spe_Total_NO[WhatSpecies[gc]];++i)
      for(int j=0;j<active_components;++j)
        assert(openmx_projector_block_row(ds,0,Matomnum+1,h,i)[j]==expected(gc,h,i,j));
  consume(original,gc);
}
'''
LEGACY = r'''
static void legacy(int gpu) {
  int ID,IDS,IDR,numprocs,myid,size1,size2,Mc_AN,Gc_AN,Cwan,h_AN,Gh_AN,Hwan,tno1,tno2;
  int n,num,i,j,Original_Mc_AN,po,tag=999,setpro_gpu=gpu,fp32=1,hvna=0,hvna_geometry=0,hvna_projectors=0,halo_workspace=0;
  Type_DS_VNA *tmp_array,*tmp_array2,*****DS_VNA=channels;double ****HVNA=NULL;
  MPI_Comm_size(mpi_comm_level1,&numprocs);MPI_Comm_rank(mpi_comm_level1,&myid);
  int Snd_DS_VNA_Size[4],Rcv_DS_VNA_Size[4];MPI_Request request;MPI_Status stat;
'''
PRODUCTION = r'''
static void candidate(int gpu) {
  int peers,rank;MPI_Comm_size(mpi_comm_level1,&peers);MPI_Comm_rank(mpi_comm_level1,&rank);
  OpenMXProjectorTransfer transfer=OpenMX_TransferBegin(mpi_comm_level1,peers,sizeof(Type_DS_VNA));
  int *sent=transfer.buffers.sent,*received=transfer.buffers.received;
  int *send_sizes=transfer.buffers.send_sizes,*receive_sizes=transfer.buffers.receive_sizes;
  const int components=active_components,fp32=1,hvna=0,hvna_geometry=0,projectors=0,halo_workspace=0;
  Type_DS_VNA *****DS_VNA=channels;double ****HVNA=NULL;
  const OpenMXProjectorMatrix projectors_matrix=ds_matrix;
  const OpenMXVnaMatrixViews *matrices=&matrix_views;
  struct {int *orbitals;} preparation={Spe_Total_NO};
'''
MAIN = r'''
static void configure(int rank,int peers,int scenario) {
  const int ownership[4]={2,1,3,0};
  int offset[4],global=0;
  for(int p=0;p<peers;++p){offset[p]=global;global+=ownership[p];}
  Matomnum=ownership[rank];for(int mc=1;mc<=Matomnum;++mc)M2G[mc]=offset[rank]+mc;
  Spe_Total_NO[0]=3;Spe_Total_NO[1]=scenario==1?0:2;
  List_YOUSO[35]=1;List_YOUSO[34]=scenario==2?0:2;active_components=4*List_YOUSO[34];
  for(int g=1;g<=global;++g){WhatSpecies[g]=g%2;FNAN[g]=g%NH;for(int h=0;h<NH;++h)natn[g][h]=1+(g+h)%global;}
  int original=Matomnum+1;
  for(int peer=0;peer<peers;++peer){
    Snd_MAN[peer]=send_local[peer];Snd_GAN[peer]=send_global[peer];Rcv_GAN[peer]=receive_global[peer];
    F_Snd_Num[peer]=Matomnum;F_Rcv_Num[peer]=ownership[peer];F_TopMAN[peer]=original;original+=ownership[peer];
    for(int i=0;i<Matomnum;++i){Snd_MAN[peer][i]=i+1;Snd_GAN[peer][i]=M2G[i+1];}
    for(int i=0;i<ownership[peer];++i)Rcv_GAN[peer][i]=offset[peer]+i+1;
  }
  for(int mc=0;mc<NA+2;++mc){
    atoms[mc]=neighbors[mc];
    for(int h=0;h<NH;++h){neighbors[mc][h]=rows[mc][h];for(int i=0;i<NO;++i)rows[mc][h][i]=values[mc][h][i];}
  }
  for(int mc=1;mc<=Matomnum;++mc)for(int h=0;h<NH;++h)for(int i=0;i<NO;++i)for(int j=0;j<NP;++j)
    values[mc][h][i][j]=expected(M2G[mc],h,i,j);
  for(int a=0;a<=NA+2;++a)matrix_atoms[a]=(size_t)a*NH;
  for(int b=0;b<=(NA+2)*NH;++b){
    matrix_blocks[b]=(size_t)b*NO*NP;
    if(b<(NA+2)*NH){matrix_shapes[2*b]=NO;matrix_shapes[2*b+1]=NP;}
  }
  ds_matrix=(OpenMXProjectorMatrix){{NA+2,1,(NA+2)*NH,(NA+2)*NH*NO*NP,(NA+2)*NH*NO*NP,
      matrix_atoms,matrix_blocks,matrix_shapes},&values[0][0][0][0]};
  event_size=trace_size=0;memset(events,0,sizeof(events));memset(trace,0,sizeof(trace));
}
int main(int argc,char** argv) {
  MPI_Init(&argc,&argv);mpi_comm_level1=MPI_COMM_WORLD;
  MPI_Type_DS_VNA=sizeof(Type_DS_VNA)==sizeof(float)?MPI_FLOAT:MPI_DOUBLE;
  int rank,peers,cases=0;MPI_Comm_rank(mpi_comm_level1,&rank);MPI_Comm_size(mpi_comm_level1,&peers);assert(peers==1||peers==4);
  for(int repeat=0;repeat<2;++repeat)for(int scenario=0;scenario<3;++scenario)for(int gpu=0;gpu<2;++gpu){
    configure(rank,peers,scenario);legacy(gpu);
    int old_events[1024][3],old_trace[NG+1],old_count=event_size,old_size=trace_size;
    memcpy(old_events,events,sizeof(events));memcpy(old_trace,trace,sizeof(trace));
    MPI_Barrier(mpi_comm_level1);configure(rank,peers,scenario);candidate(gpu);
    assert(old_count==event_size && old_size==trace_size);
    assert(!memcmp(old_events,events,sizeof(events)) && !memcmp(old_trace,trace,sizeof(trace)));cases++;
  }
  int total;MPI_Reduce(&cases,&total,1,MPI_INT,MPI_SUM,0,mpi_comm_level1);
  if(!rank)printf("PASS: %d legacy-identical VNA MPI cases on %d ranks (%zu-byte storage); all transfer events and poisoned halo slots\n",total,peers,sizeof(Type_DS_VNA));
  MPI_Finalize();return 0;
}
'''

with tempfile.TemporaryDirectory(prefix="openmx-vna-exchange-") as directory:
    folder = Path(directory)
    src, exe = folder / "check.c", folder / "check"
    src.write_text(PREFIX + FLAT_CALLBACK + LEGACY + reference + "\n}\n" + PRODUCTION + production
                   + "\nOpenMX_TransferEnd(&transfer);\n}\n" + MAIN)
    environment = os.environ.copy()
    environment.setdefault("OMPI_CC", "cc")
    def finite_stack():
        soft, hard = resource.getrlimit(resource.RLIMIT_STACK)
        if soft == resource.RLIM_INFINITY:
            resource.setrlimit(resource.RLIMIT_STACK, (8 * 1024 * 1024, hard))
    for storage in ("float", "double"):
        subprocess.run(compiler + ["-std=c11", "-O2", "-DTEST_DS_TYPE=" + storage,
                                  "-I" + str(ROOT / "source"), str(src), str(CORE),
                                  "-Wl,-rpath," + str(CORE.parent), "-o", str(exe)], env=environment, check=True)
        for ranks in (1, 4):
            subprocess.run([launcher, "-np", str(ranks), str(exe)], env=environment,
                           check=True, timeout=90, preexec_fn=finite_stack)

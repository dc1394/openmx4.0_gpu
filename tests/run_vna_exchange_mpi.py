#!/usr/bin/env python3
"""Compare the production VNA ring against the frozen legacy protocol.

Use real MPI, asymmetric atom ownership, empty ranks and empty projector/basis
blocks. Compare payload events and halo values, checking that the new loop
sends directly, skips size messages, and reuses staging only after consumption.
Model host and device archives as well as CPU consumption on one/four ranks.
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
def helper(source, declaration, terminator="\n}\n"):
    first = source.index(declaration)
    return source[first:source.index(terminator, first) + len(terminator)]

gpu_context = (ROOT / "source/openmx_vna_gpu_context.h").read_text()
ring_helpers = (helper(assembly, "typedef struct { double send,", "\n")
                + helper(assembly, "static inline void SetPro_RingTick(")
                + helper(gpu_context, "static inline int SetPro_HVNA_HaloSpan("))
reference = (ROOT / "tests/fixtures/vna_exchange_before_raii.c").read_text()
# The frozen loop keeps its pointer-row callback and the production loop uses
# compact views. They must yield the same payload order and values.
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
typedef struct {
  size_t halo_count;
  int halo_atoms,*halo_base;
  size_t *halo_off;
  Type_DS_VNA *halo_host,*halo_dev;
} SetProGpuContext;
static Type_DS_VNA local_values[1+NA*NH*NO*NP],archive_values[1+NG*NH*NO*NP];
static int halo_base[NG+2],halo_world[NG+1],sent_now[4],received_now[4];
static size_t halo_offsets[NG*NH];
static SetProGpuContext gpu_archive;
static OpenMXProjectorMatrix ds_matrix;
static OpenMXVnaMatrixViews matrix_views;
static int trace[NG+1],trace_size,events[1024][3],event_size,active_components;
static int route,payload_send,size_messages,last_global,pending_consumption;
static Type_DS_VNA expected(int g,int h,int row,int column) { return (Type_DS_VNA)(g*1000+h*100+row*10+column); }
static void event(int kind,int peer,int count) {
  assert(event_size<1024);events[event_size][0]=kind;events[event_size][1]=peer;events[event_size++][2]=count;
}
static int tracked_send(const void *p,int n,MPI_Datatype type,int peer,int tag,MPI_Comm comm,MPI_Request *request) {
  payload_send=type!=MPI_INT;
  if(payload_send){
    event(2,peer,n);
    if(route>=0){
      const int mc=send_local[peer][sent_now[peer]++];
      assert(p==local_values+matrix_blocks[matrix_atoms[mc]]);
    }
  }else ++size_messages;
  return MPI_Isend(p,n,type,peer,tag,comm,request);
}
static int tracked_receive(void *p,int n,MPI_Datatype type,int peer,int tag,MPI_Comm comm,MPI_Status *status) {
  if(type==MPI_INT)++size_messages;
  else{
    event(4,peer,n);
    if(route>=0){
      assert(!pending_consumption);
      const int index=received_now[peer]++,halo=F_TopMAN[peer]+index-Matomnum;
      assert((p==archive_values+halo_offsets[halo_base[halo]])==(route==1));
      last_global=receive_global[peer][index];
      assert(trace_size<NG+1);trace[trace_size++]=last_global;
      pending_consumption=route==0 || (route==2 && n);
    }
  }
  return MPI_Recv(p,n,type,peer,tag,comm,status);
}
static int tracked_wait(MPI_Request *request,MPI_Status *status) {
  if(payload_send)event(5,0,0);
  return MPI_Wait(request,status);
}
#define MPI_Isend tracked_send
#define MPI_Recv tracked_receive
#define MPI_Wait tracked_wait
static double clock_ticks;
static void dtime(double *t) { *t=++clock_ticks; }
static void consume(int original,int gc) {
  (void)original;assert(trace_size<NG+1);trace[trace_size++]=gc;
  for(int h=0;h<=FNAN[gc];++h)
    for(int i=0;i<Spe_Total_NO[WhatSpecies[gc]];++i)
      for(int j=0;j<active_components;++j)assert(values[Matomnum+1][h][i][j]==expected(gc,h,i,j));
  for(size_t i=0;i<sizeof(values[0])/sizeof(Type_DS_VNA);++i)((Type_DS_VNA*)values[Matomnum+1])[i]=NAN;
}
static void SetPro_HVNA_GpuArchiveHalo(const void *g,Type_DS_VNA *****ds,int original,int gc) {(void)g;(void)ds;consume(original,gc);}
static void reference_consume_halo(const void *geometry,const void *projectors,const void *workspace,
    Type_DS_VNA *****ds,double ****hvna,int original,int gc,int fp32) {
  (void)geometry;(void)projectors;(void)workspace;(void)ds;(void)hvna;(void)fp32;consume(original,gc);
}
'''
FLAT_CALLBACK = r'''
static int check_payload(const Type_DS_VNA *p,int gc) {
  int n=0;
  for(int h=0;h<=FNAN[gc];++h)
    for(int i=0;i<Spe_Total_NO[WhatSpecies[gc]];++i)
      for(int j=0;j<active_components;++j)assert(p[n++]==expected(gc,h,i,j));
  return n;
}
static void SetPro_HVNA_Upload(Type_DS_VNA *device,Type_DS_VNA *host,size_t count) {
  assert(route==2 && pending_consumption);
  pending_consumption=0;
  assert((size_t)check_payload(host,last_global)==count);
  memcpy(device,host,count*sizeof(*host));
  for(size_t i=0;i<count;++i)host[i]=NAN;
}
static void SetPro_HVNA_CPU_Halo(const void *geometry,const void *projectors,const void *workspace,
    const OpenMXProjectorMatrix *ds,const OpenMXDoubleBlockMatrix *hvna,int original,int gc,
    Type_DS_VNA *received,int fp32,const void *halo_cache) {
  (void)geometry;(void)projectors;(void)workspace;(void)ds;(void)hvna;(void)fp32;assert(halo_cache);
  assert(route==0 && pending_consumption);
  pending_consumption=0;
  assert(gc==last_global && gc==halo_world[original-Matomnum]);
  const int n=check_payload(received,gc);
  for(int i=0;i<n;++i)received[i]=NAN;
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
  const int components=active_components,fp32=1,hvna_geometry=0,projectors=0,halo_workspace=0;
  gpu_archive.halo_host=gpu==1 ? archive_values : NULL;
  gpu_archive.halo_dev=gpu==2 ? archive_values : NULL;
  const SetProGpuContext *archives=gpu ? &gpu_archive : NULL;
  const OpenMXProjectorMatrix projectors_matrix=ds_matrix;
  const OpenMXVnaMatrixViews *matrices=&matrix_views;
  struct {int *orbitals;} preparation={Spe_Total_NO};
  /* the verbose phase clock runs: it must not disturb the protocol */
  const int ring_timed=1;SetProRingTimes ring;double ring_clock=0;int halo_cache=0;
  memset(&ring,0,sizeof(ring));dtime(&ring_clock);
'''
MAIN = r'''
static void configure(int rank,int peers,int scenario,int candidate_route) {
  const int ownership[4]={2,1,3,0};
  int offset[4],global=0;
  for(int p=0;p<peers;++p){offset[p]=global;global+=ownership[p];}
  Matomnum=ownership[rank];for(int mc=1;mc<=Matomnum;++mc)M2G[mc]=offset[rank]+mc;
  Spe_Total_NO[0]=3;Spe_Total_NO[1]=scenario==1?0:2;
  List_YOUSO[35]=1;List_YOUSO[34]=scenario==2?0:2;active_components=4*List_YOUSO[34];
  for(int g=1;g<=global;++g){WhatSpecies[g]=g%2;FNAN[g]=g%NH;for(int h=0;h<NH;++h)natn[g][h]=1+(g+h)%global;}
  int original=Matomnum+1,halo_blocks=0;
  size_t halo_count=0;
  for(int peer=0;peer<peers;++peer){
    Snd_MAN[peer]=send_local[peer];Snd_GAN[peer]=send_global[peer];Rcv_GAN[peer]=receive_global[peer];
    F_Snd_Num[peer]=Matomnum;F_Rcv_Num[peer]=ownership[peer];F_TopMAN[peer]=original;original+=ownership[peer];
    for(int i=0;i<Matomnum;++i){Snd_MAN[peer][i]=i+1;Snd_GAN[peer][i]=M2G[i+1];}
    for(int i=0;i<ownership[peer];++i){
      const int gc=offset[peer]+i+1,halo=F_TopMAN[peer]+i-Matomnum;
      Rcv_GAN[peer][i]=halo_world[halo]=gc;
      halo_base[halo]=halo_blocks;
      for(int h=0;h<=FNAN[gc];++h){
        halo_offsets[halo_blocks++]=halo_count;
        halo_count+=(size_t)Spe_Total_NO[WhatSpecies[gc]]*active_components;
      }
    }
  }
  for(int mc=0;mc<NA+2;++mc){
    atoms[mc]=neighbors[mc];
    for(int h=0;h<NH;++h){neighbors[mc][h]=rows[mc][h];for(int i=0;i<NO;++i)rows[mc][h][i]=values[mc][h][i];}
  }
  for(int mc=1;mc<=Matomnum;++mc)for(int h=0;h<NH;++h)for(int i=0;i<NO;++i)for(int j=0;j<NP;++j)
    values[mc][h][i][j]=expected(M2G[mc],h,i,j);
  matrix_atoms[0]=matrix_blocks[0]=0;
  matrix_shapes[0]=matrix_shapes[1]=1;
  size_t block=1,count=1;
  for(int mc=1;mc<=Matomnum;++mc){
    const int gc=M2G[mc],nr=Spe_Total_NO[WhatSpecies[gc]];
    matrix_atoms[mc]=block;
    for(int h=0;h<=FNAN[gc];++h){
      matrix_blocks[block]=count;
      matrix_shapes[2*block]=nr;matrix_shapes[2*block+1]=active_components;
      ++block;
      for(int i=0;i<nr;++i)for(int j=0;j<active_components;++j)
        local_values[count++]=values[mc][h][i][j];
    }
  }
  matrix_atoms[Matomnum+1]=block;matrix_blocks[block]=count;
  ds_matrix=(OpenMXProjectorMatrix){{Matomnum+1,1,block,count,count,
      matrix_atoms,matrix_blocks,matrix_shapes},local_values};
  gpu_archive=(SetProGpuContext){.halo_count=halo_count,.halo_atoms=global,.halo_base=halo_base,.halo_off=halo_offsets};
  for(size_t i=0;i<halo_count;++i)archive_values[i]=NAN;
  route=candidate_route;
  event_size=trace_size=size_messages=pending_consumption=0;
  memset(events,0,sizeof(events));memset(trace,0,sizeof(trace));
  memset(sent_now,0,sizeof(sent_now));memset(received_now,0,sizeof(received_now));
}
int main(int argc,char** argv) {
  MPI_Init(&argc,&argv);mpi_comm_level1=MPI_COMM_WORLD;
  MPI_Type_DS_VNA=sizeof(Type_DS_VNA)==sizeof(float)?MPI_FLOAT:MPI_DOUBLE;
  int rank,peers,cases=0;MPI_Comm_rank(mpi_comm_level1,&rank);MPI_Comm_size(mpi_comm_level1,&peers);assert(peers==1||peers==4);
  for(int repeat=0;repeat<2;++repeat)for(int scenario=0;scenario<3;++scenario)for(int gpu=0;gpu<3;++gpu){
    configure(rank,peers,scenario,-1);legacy(gpu);
    int old_events[1024][3],old_trace[NG+1],old_count=event_size,old_size=trace_size;
    memcpy(old_events,events,sizeof(events));memcpy(old_trace,trace,sizeof(trace));
    MPI_Barrier(mpi_comm_level1);configure(rank,peers,scenario,gpu);candidate(gpu);
    assert(size_messages==0 && pending_consumption==0);
    assert(old_count==event_size && old_size==trace_size);
    assert(!memcmp(old_events,events,sizeof(events)) && !memcmp(old_trace,trace,sizeof(trace)));
    if(gpu)for(int halo=1;halo<=gpu_archive.halo_atoms;++halo)
      check_payload(archive_values+halo_offsets[halo_base[halo]],halo_world[halo]);
    cases++;
  }
  int total;MPI_Reduce(&cases,&total,1,MPI_INT,MPI_SUM,0,mpi_comm_level1);
  if(!rank)printf("PASS: %d VNA MPI cases on %d ranks (%zu-byte storage); same payloads, direct archives and poisoned staging\n",total,peers,sizeof(Type_DS_VNA));
  MPI_Finalize();return 0;
}
'''

with tempfile.TemporaryDirectory(prefix="openmx-vna-exchange-") as directory:
    folder = Path(directory)
    src, exe = folder / "check.c", folder / "check"
    src.write_text(PREFIX + ring_helpers + FLAT_CALLBACK + LEGACY + reference + "\n}\n" + PRODUCTION + production
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

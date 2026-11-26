#!/usr/bin/env python3
"""Exercise the production transfer owner with in-flight MPI sends.

The receive direction grows while a rendezvous-sized send remains live. Float
and double owners, zero payloads, split communicators, repeated invocations
and buffer reuse run with one and four real ranks, without a GPU runtime.
The VNA force fixture also checks sparse routing, unequal centre counts and
reverse periodic images after removing the case-2 control messages.
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
SOURCE = r'''
#include "openmx_projector_exchange.h"
#include "openmx_core_solver_workspace.h"
#include <assert.h>
#include <string.h>

static int *OpenMX_Solver_Int(OpenMXSolverInvocation *owner,const char *name,size_t count)
{
  int *values;
  char error[256];
  assert(openmx_core_solver_int(owner,0,name,count,&values,error,sizeof(error)));
  return values;
}
#include "openmx_force_vna_case2.h"

static double value(int rank,int phase,int i) { return (rank+1)*1000000.0+phase*40000.0+i; }
static void check(MPI_Comm comm,int *calls)
{
  int rank,peers;MPI_Comm_rank(comm,&rank);MPI_Comm_size(comm,&peers);
  const int destination=(rank+1)%peers,source=(rank+peers-1)%peers;
  OpenMXProjectorTransfer single=OpenMX_TransferBegin(comm,peers,sizeof(float));
  OpenMXProjectorTransfer dual=OpenMX_TransferBegin(comm,peers,sizeof(double));
  single.buffers.sent[0]=41;dual.buffers.sent[0]=17;
  assert(single.buffers.sent!=dual.buffers.sent);
  for (int phase=0;phase<8;++phase) {
    const int send_count=phase==0?0:(phase%2?32768:5)+rank;
    const int receive_count=phase==0?0:(phase%2?32768:5)+source;
    float *send_float=OpenMX_TransferBuffer(&single,OPENMX_EXCHANGE_SEND,send_count);
    double *send_double=OpenMX_TransferBuffer(&dual,OPENMX_EXCHANGE_SEND,send_count);
    for (int i=0;i<send_count;++i) {send_float[i]=(float)value(rank,phase,i);send_double[i]=value(rank,phase,i);}
    MPI_Request requests[2];
    MPI_Isend(send_float,send_count,MPI_FLOAT,destination,101,comm,requests);
    MPI_Isend(send_double,send_count,MPI_DOUBLE,destination,102,comm,requests+1);
    /* Independent growth must not change the in-flight send pointer/data. */
    float *receive_float=OpenMX_TransferBuffer(&single,OPENMX_EXCHANGE_RECEIVE,receive_count+phase*8192);
    double *receive_double=OpenMX_TransferBuffer(&dual,OPENMX_EXCHANGE_RECEIVE,receive_count+phase*8192);
    assert(receive_float!=send_float && receive_double!=send_double);
    MPI_Recv(receive_double,receive_count,MPI_DOUBLE,source,102,comm,MPI_STATUS_IGNORE);
    MPI_Recv(receive_float,receive_count,MPI_FLOAT,source,101,comm,MPI_STATUS_IGNORE);
    MPI_Waitall(2,requests,MPI_STATUSES_IGNORE);
    for (int i=0;i<receive_count;++i) {
      assert(receive_float[i]==(float)value(source,phase,i));
      assert(receive_double[i]==value(source,phase,i));
    }
    assert(single.buffers.sent[0]==41 && dual.buffers.sent[0]==17);
    assert(OpenMX_TransferBuffer(&single,OPENMX_EXCHANGE_SEND,send_count)==send_float);
    (*calls)++;
  }
  OpenMX_TransferEnd(&single);
  assert(dual.buffers.sent[0]==17);
  OpenMX_TransferEnd(&dual);
}

static float projector_value(int atom,int neighbour,int dir,int row,int p)
{
  return (float)((atom*1000+neighbour*100+dir*20+row)*1024+p);
}

static int check_vna_case2(OpenMXSolverInvocation *scratch,void *argument,double *result)
{
  MPI_Comm comm=*(MPI_Comm*)argument;
  int rank,peers;
  MPI_Comm_rank(comm,&rank);
  MPI_Comm_size(comm,&peers);
  enum { atoms=4, max_peers=4, max_neighbours=5, max_orbitals=4, projectors=1024 };
  assert(peers==1 || peers==4);
  const int count[]={0,3,2,4,1}, species[]={0,0,1,2,3}, orbitals[]={2,3,1,4};
  int n1[]={1,3,3,2}, n2[]={2,4,1}, n3[]={3,1,1,3,3}, n4[]={4,2};
  int *neighbours[]={NULL,n1,n2,n3,n4};
  const int reverse[][max_neighbours]={{0},{0,2,1,2},{0,1,3},{0,2,1,4,3},{0,1}};
  const int owner[]={-1,0,0,peers==1?0:1,peers==1?0:2};
  int local_atom[atoms+1]={0}, force_index[atoms+1]={0}, local_count=0;
  int needs[max_peers][atoms+1]={0};
  for (int g=1;g<=atoms;++g) {
    if (owner[g]==rank) {
      local_atom[++local_count]=g;
      force_index[g]=local_count;
    }
    for (int k=0;k<=count[g];++k) needs[owner[g]][neighbours[g][k]]=1;
  }
  int sent_atoms[max_peers]={0}, received_atoms[max_peers]={0}, receive_start[max_peers]={0};
  int send_storage[max_peers][atoms], *sent_local[max_peers];
  int halo_count=0;
  for (int peer=0;peer<peers;++peer) {
    sent_local[peer]=send_storage[peer];
    if (peer==rank) continue;
    receive_start[peer]=local_count+halo_count+1;
    for (int g=1;g<=atoms;++g) {
      if (owner[g]==rank && needs[peer][g]) sent_local[peer][sent_atoms[peer]++]=force_index[g];
      if (owner[g]==peer && needs[rank][g]) {
        force_index[g]=local_count+(++halo_count);
        ++received_atoms[peer];
      }
    }
  }
  enum { max_entries=atoms*max_neighbours+1 };
  int targets[max_peers][max_entries]={0}, sources[max_peers][max_entries];
  int source_slots[max_peers][max_entries], target_slots[max_peers][max_entries];
  int *target[max_peers], *source[max_peers], *source_slot[max_peers], *target_slot[max_peers];
  for (int peer=0;peer<peers;++peer) {
    target[peer]=targets[peer];
    source[peer]=sources[peer];
    source_slot[peer]=source_slots[peer];
    target_slot[peer]=target_slots[peer];
    int p=0;
    for (int g=1;g<=atoms;++g) {
      if (owner[g]!=peer) continue;
      /* Input rows of the same centre deliberately come in reverse order. */
      for (int k=count[g];k>=0;--k) {
        const int from=neighbours[g][k];
        if (owner[from]!=rank) continue;
        target[peer][p]=g;
        source[peer][p]=force_index[from];
        source_slot[peer][p]=reverse[g][k];
        target_slot[peer][p++]=k;
      }
    }
  }
  enum { blocks=(atoms+2)*max_neighbours, stride=blocks*max_orbitals*projectors };
  float storage[4*stride];
  size_t atom_offsets[atoms+3], block_offsets[blocks+1];
  int shapes[2*blocks];
  for (int i=0;i<=atoms+2;++i) atom_offsets[i]=(size_t)i*max_neighbours;
  for (int i=0;i<=blocks;++i) block_offsets[i]=(size_t)i*max_orbitals*projectors;
  for (int i=0;i<blocks;++i) {
    shapes[2*i]=max_orbitals;
    shapes[2*i+1]=projectors;
  }
  OpenMXFloatBlockMatrix data={
    .layout={atoms+2,4,blocks,stride,4*stride,atom_offsets,block_offsets,shapes},
    .values=storage
  };
  for (int d=0;d<4;++d)
    for (int mc=1;mc<=local_count;++mc)
      for (int k=0;k<max_neighbours;++k)
        for (int i=0;i<max_orbitals;++i)
          for (int l=0;l<projectors;++l)
            openmx_float_block_row(&data,d,mc,k,i)[l]=projector_value(local_atom[mc],k,d,i,l);
  OpenMXProjectorTransfer transfer=OpenMX_TransferBegin(comm,peers,sizeof(float));
  OpenMXVnaCase2Exchange exchange={
    .topology={local_count,max_neighbours,local_atom,force_index,
      species,orbitals,count,neighbours,sent_atoms,received_atoms,receive_start,sent_local,
      target,source,source_slot,target_slot},
    .transfer=&transfer,.datatype=MPI_FLOAT,.rank=rank,.peers=peers,.tag=103,.projectors=projectors
  };
  OpenMX_VnaCase2Begin(&exchange,scratch);
  const int turns=peers==1?atoms:2, staging=local_count+1;
  for (int turn=1;turn<=turns;++turn) {
    for (int d=0;d<4;++d) memset(openmx_float_block_row(&data,d,staging,0,0),0,
        (size_t)max_neighbours*max_orbitals*projectors*sizeof(float));
    OpenMX_VnaCase2Stage(&exchange,turn,&data);
    if (turn>local_count) continue;
    const int g=local_atom[turn];
    for (int k=0;k<=count[g];++k) {
      const int from=neighbours[g][k];
      for (int d=0;d<4;++d)
        for (int i=0;i<orbitals[species[from]];++i)
          for (int p=0;p<projectors;++p)
            assert(openmx_float_block_row(&data,d,staging,k,i)[p]==projector_value(from,reverse[g][k],d,i,p));
    }
  }
  OpenMX_TransferEnd(&transfer);
  *result=1.0;
  return 1;
}

int main(int argc,char** argv)
{
  MPI_Init(&argc,&argv);
  int rank,peers,calls=0;MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&peers);
  for (int repeat=0;repeat<3;++repeat) {
    check(MPI_COMM_WORLD,&calls);
    MPI_Comm separate;MPI_Comm_split(MPI_COMM_WORLD,rank%2,rank,&separate);
    check(separate,&calls);MPI_Comm_free(&separate);
  }
  MPI_Comm comm=MPI_COMM_WORLD;
  double result;
  char error[256];
  assert(openmx_core_solver_run(1,check_vna_case2,&comm,&result,error,sizeof(error)) && result==1.0);
  int total;MPI_Reduce(&calls,&total,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
  if (!rank) printf("PASS: %d real MPI transfer phases on %d ranks; VNA rows, live sends, zero counts and independent communicators\n",total,peers);
  MPI_Finalize();return 0;
}
'''

with tempfile.TemporaryDirectory(prefix="openmx-projector-mpi-") as directory:
    folder = Path(directory)
    src, exe = folder / "check.c", folder / "check"
    src.write_text(SOURCE)
    environment = os.environ.copy()
    environment.setdefault("OMPI_CC", "cc")
    subprocess.run(compiler + ["-std=c11", "-O2", "-I" + str(ROOT / "source"), "-I" + str(ROOT / "core/include"),
                              str(src), str(CORE),
                              "-Wl,-rpath," + str(CORE.parent), "-o", str(exe)], env=environment, check=True)
    def finite_stack():
        soft, hard = resource.getrlimit(resource.RLIMIT_STACK)
        if soft == resource.RLIM_INFINITY:
            resource.setrlimit(resource.RLIMIT_STACK, (8 * 1024 * 1024, hard))
    for ranks in (1, 4):
        subprocess.run([launcher, "-np", str(ranks), str(exe)], env=environment,
                       check=True, timeout=90, preexec_fn=finite_stack)

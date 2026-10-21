#!/usr/bin/env python3
"""Exercise the production transfer owner with in-flight MPI sends.

The receive direction grows while a rendezvous-sized send remains live. Float
and double owners, zero payloads, split communicators, repeated invocations
and buffer reuse run with one and four real ranks, without a GPU runtime.
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
#include <assert.h>
#include <string.h>

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
int main(int argc,char** argv)
{
  MPI_Init(&argc,&argv);
  int rank,peers,calls=0;MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&peers);
  for (int repeat=0;repeat<3;++repeat) {
    check(MPI_COMM_WORLD,&calls);
    MPI_Comm separate;MPI_Comm_split(MPI_COMM_WORLD,rank%2,rank,&separate);
    check(separate,&calls);MPI_Comm_free(&separate);
  }
  int total;MPI_Reduce(&calls,&total,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
  if (!rank) printf("PASS: %d real MPI transfer phases on %d ranks; float/double, live sends, zero counts and independent communicators\n",total,peers);
  MPI_Finalize();return 0;
}
'''

with tempfile.TemporaryDirectory(prefix="openmx-projector-mpi-") as directory:
    folder = Path(directory)
    src, exe = folder / "check.c", folder / "check"
    src.write_text(SOURCE)
    environment = os.environ.copy()
    environment.setdefault("OMPI_CC", "cc")
    subprocess.run(compiler + ["-std=c11", "-O2", "-I" + str(ROOT / "source"), str(src), str(CORE),
                              "-Wl,-rpath," + str(CORE.parent), "-o", str(exe)], env=environment, check=True)
    def finite_stack():
        soft, hard = resource.getrlimit(resource.RLIMIT_STACK)
        if soft == resource.RLIM_INFINITY:
            resource.setrlimit(resource.RLIMIT_STACK, (8 * 1024 * 1024, hard))
    for ranks in (1, 4):
        subprocess.run([launcher, "-np", str(ranks), str(exe)], env=environment,
                       check=True, timeout=90, preexec_fn=finite_stack)

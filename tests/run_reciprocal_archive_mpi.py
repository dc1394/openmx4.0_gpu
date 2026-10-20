#!/usr/bin/env python3
"""Exercise production Fourier restart agreement/distribution with real MPI.

Use all five row plans, asymmetric species, empty ranks, existing/missing/short
restart files, including a read failure on only the final MPI rank. Compare
every broadcast's root, pointer and count, and the legacy barrier positions.
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


def block(source, marker):
    start = source.index(marker)
    end = source.index("{", start) + 1
    depth = 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


PREFIX = r'''
#include <mpi.h>
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include "openmx_core_reciprocal_archive.h"
static MPI_Comm mpi_comm_level1;
static int Scf_RestartFromFile,Num_Procs2,Species_Top[4],Species_End[4];
static const int Host_ID=0;
static char filepath[4096],filename[]="tables",restart_filename[]="tables";
static int events[1024][3],event_count,barriers,reductions;
static const OpenMXReciprocalRows *active;
static int tracked_bcast(void *values,int count,MPI_Datatype type,int root,MPI_Comm comm) {
  assert(type==MPI_DOUBLE && event_count<1024);
  size_t row=0;while(row<active->rows && active->row[row].values!=values)++row;
  assert(row<active->rows && count==active->row[row].count);
  events[event_count][0]=root;events[event_count][1]=(int)row;events[event_count++][2]=barriers;
  return MPI_Bcast(values,count,type,root,comm);
}
static int tracked_barrier(MPI_Comm comm) {++barriers;return MPI_Barrier(comm);}
static int tracked_reduce(const void *s,void *d,int n,MPI_Datatype t,MPI_Op op,MPI_Comm c) {
  assert(s==MPI_IN_PLACE && n==1 && t==MPI_INT && op==MPI_MIN);++reductions;
  return MPI_Allreduce(s,d,n,t,op,c);
}
static double value(int kind,size_t row,int column) {return kind*10000+row*100+column+.125;}
static void generate(int kind,int rank,const OpenMXReciprocalRows *rows) {
  for(int species=Species_Top[rank];species<=Species_End[rank];++species)
    for(size_t row=rows->species_offsets[species];row<rows->species_offsets[species+1];++row)
      for(int k=0;k<rows->row[row].count;++k)rows->row[row].values[k]=value(kind,row,k);
}
#define MPI_Bcast tracked_bcast
#define MPI_Barrier tracked_barrier
#define MPI_Allreduce tracked_reduce
'''

MAIN = r'''
#undef MPI_Bcast
#undef MPI_Barrier
#undef MPI_Allreduce
int main(int argc,char **argv) {
  MPI_Init(&argc,&argv);mpi_comm_level1=MPI_COMM_WORLD;
  int peers,rank;MPI_Comm_size(mpi_comm_level1,&peers);MPI_Comm_rank(mpi_comm_level1,&rank);
  assert(argc==2 && (peers==1 || peers==4));
  const int maximum[3]={2,1,2},n0[3]={2,0,1},n1[2]={1,2},n2[3]={0,0,0};
  int *counts[3]={(int*)n0,(int*)n1,(int*)n2};
  const int projectors[3]={2,0,3},spins[3]={0,1,1};
  char error[256];OpenMXCoreContext *core=openmx_core_create_host(error,sizeof(error));assert(core);
  OpenMXPaoViews pao;OpenMXVpsViews vps;OpenMXProductViews products;
  const OpenMXPaoShape pshape={3,3,2,5,7};const OpenMXVpsShape vshape={3,2,4,5,7};
  const OpenMXNeutralProjectorShape neutral={3,2,9};
  const OpenMXProductShape product={3,9,maximum,counts};
  assert(openmx_core_pao_allocate(core,&pshape,&pao,error,sizeof(error)));
  assert(openmx_core_vps_allocate(core,&vshape,&neutral,&vps,error,sizeof(error)));
  assert(openmx_core_product_allocate(core,&product,&products,error,sizeof(error)));
  Num_Procs2=peers<3?peers:3;
  for(int id=0;id<peers;++id) {
    Species_Top[id]=id<Num_Procs2?id*3/Num_Procs2:3;
    Species_End[id]=id<Num_Procs2?(id+1)*3/Num_Procs2-1:2;
  }
  const char *extensions[]={"ftpao","ftnlp","ftPEvna","ftCvna","ftProPAO"};
  int cases=0;
  for(int kind=0;kind<5;++kind) {
    OpenMXReciprocalArchiveInput in={kind,3,4,5,maximum,counts,projectors,spins};
    OpenMXReciprocalRows rows;
    OpenMXReciprocalArchive *archive=openmx_core_reciprocal_archive_create(core,&in,&rows,error,sizeof(error));assert(archive);
    active=&rows;
    snprintf(filepath,sizeof(filepath),"%s/",argv[1]);
    save_candidate(kind,rank,archive);
    MPI_Barrier(mpi_comm_level1);
    char saved[4096];snprintf(saved,sizeof(saved),"%s/tables_rst/tables.%s",argv[1],extensions[kind]);
    assert(!access(saved,R_OK));
    for(int scenario=0;scenario<4;++scenario) {
      char path[4096];snprintf(path,sizeof(path),"%s/tables_rst/tables.%s",argv[1],extensions[kind]);
      if(rank==0) {
        for(size_t row=0;row<rows.rows;++row)for(int k=0;k<rows.row[row].count;++k)
          rows.row[row].values[k]=value(kind,row,k);
        assert(openmx_core_reciprocal_archive_write(archive,path,error,sizeof(error)));
        if(scenario==3) {FILE *file=fopen(path,"wb");assert(file);fputc(1,file);assert(!fclose(file));}
      }
      MPI_Barrier(mpi_comm_level1);
      for(size_t row=0;row<rows.rows;++row)for(int k=0;k<rows.row[row].count;++k)rows.row[row].values[k]=NAN;
      snprintf(filepath,sizeof(filepath),"%s/%s",argv[1],scenario==2 && rank==peers-1?"missing/":"");
      Scf_RestartFromFile=scenario!=0;event_count=barriers=reductions=0;
      const int loaded=candidate(kind,rank,archive,rows);
      assert(loaded==(scenario==1) && reductions==(scenario!=0));
      for(size_t row=0;row<rows.rows;++row)for(int k=0;k<rows.row[row].count;++k)
        assert(rows.row[row].values[k]==value(kind,row,k));
      int count=0,expected_barriers=0;
      if(!loaded)for(int id=0;id<Num_Procs2;++id)
        for(int s=Species_Top[id];s<=Species_End[id];++s)
          for(size_t row=rows.species_offsets[s];row<rows.species_offsets[s+1];++row) {
            if(kind==OPENMX_ARCHIVE_PRODUCT)++expected_barriers;
            assert(events[count][0]==id && events[count][1]==(int)row && events[count][2]==expected_barriers);
            if(kind==OPENMX_ARCHIVE_PAO)++expected_barriers;
            ++count;
          }
      assert(event_count==count && barriers==expected_barriers);
      // Receiving active rows must not overwrite reserved projector/padding.
      assert(vps.transforms[0][0][0][0]==0 && vps.neutral_transforms[0][0][0][9]==0);
      MPI_Barrier(mpi_comm_level1);++cases;
    }
    openmx_core_reciprocal_archive_destroy(archive);
  }
  openmx_core_destroy(core);
  if(rank==0)printf("PASS %d production Fourier restart/distribution cases with %d MPI ranks\n",cases,peers);
  MPI_Finalize();return 0;
}
'''

functions = []
for kind, name in enumerate(("FT_PAO", "FT_NLP", "FT_ProExpn_VNA", "FT_VNA", "FT_ProductPAO")):
    source = (ROOT / "source" / (name + ".c")).read_text()
    restart = block(source, "if (Scf_RestartFromFile==1)")
    transfer = block(source, "for (ID=0; ID<Num_Procs2; ID++)")
    save = block(source, "if (myid==Host_ID)")
    functions.append(f"""
static int candidate_{kind}(int rank,OpenMXReciprocalArchive *archive,OpenMXReciprocalRows archive_rows) {{
  int RestartRead_Succeed=0,ID,NumSpe,Lspe,spe;
  char fileFT[8192],archive_error[256];
  {restart}
  if(!RestartRead_Succeed) {{ generate({kind},rank,&archive_rows);{transfer} }}
  return RestartRead_Succeed;
}}
static void save_{kind}(int myid,const OpenMXReciprocalArchive *archive) {{
  char fileFT[8192],archive_error[256];
  {save}
}}
""")
dispatch = "static int candidate(int kind,int rank,OpenMXReciprocalArchive *archive,OpenMXReciprocalRows rows) { switch(kind) {"
dispatch += "".join(f"case {k}:return candidate_{k}(rank,archive,rows);" for k in range(5))
dispatch += "} abort();}\n"
dispatch += "static void save_candidate(int kind,int rank,const OpenMXReciprocalArchive *archive) {switch(kind) {"
dispatch += "".join(f"case {k}:save_{k}(rank,archive);return;" for k in range(5))
dispatch += "} abort();}\n"

with tempfile.TemporaryDirectory(prefix="openmx-reciprocal-mpi-") as directory:
    folder = Path(directory)
    source, binary = folder / "test.c", folder / "test"
    source.write_text(PREFIX + "\n".join(functions) + dispatch + MAIN)
    flags = shlex.split(os.environ.get("CFLAGS", "-O2"))
    command = compiler + flags + ["-std=gnu11", "-I" + str(ROOT / "source"), str(source), str(CORE),
                                 "-Wl,-rpath," + str(CORE.parent), "-lm", "-o", str(binary)]
    subprocess.run(command, check=True)
    environment = {k: v for k, v in os.environ.items() if k not in ("LD_PRELOAD", "LD_LIBRARY_PATH")}
    environment.update(OMP_NUM_THREADS="1", OMPI_ALLOW_RUN_AS_ROOT="1", OMPI_ALLOW_RUN_AS_ROOT_CONFIRM="1")
    soft, hard = resource.getrlimit(resource.RLIMIT_STACK)
    resource.setrlimit(resource.RLIMIT_STACK, (min(8*1024*1024, hard) if hard > 0 else 8*1024*1024, hard))
    for ranks in (1, 4):
        shutil.rmtree(folder / "tables_rst", ignore_errors=True)
        subprocess.run([launcher, "--oversubscribe", "--bind-to", "none", "-np", str(ranks),
                        str(binary), str(folder)], check=True, env=environment, timeout=90)

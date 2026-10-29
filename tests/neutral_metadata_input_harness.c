/* Actual PAO/VPS adapters over independent Core metadata owners. Files are
   parsed by the real C++ readers; only atomic-view publication is modeled. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "openmx_core_neutral_grid_metadata.h"
#include "openmx_core_atomic.h"
#include "openmx_core_pao_file.h"
#include "openmx_core_vps_file.h"
#include "openmx_core_quadrature.h"
#define CHECK(c) do { if (!(c)) { fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#c); exit(2); } } while (0)
typedef int MPI_Comm;
static int core_hole_state_flag,Core_Hole_Atom,WhatSpecies[2],level_stdout,List_YOUSO[40],SO_switch;
static char Core_Hole_Orbital[2]="s";
static double core_hole_state_ev;
static OpenMXCoreContext *atomic_context;
static OpenMXAtomicViews atomic;
static OpenMXAtomicViews OpenMX_AtomicTables(OpenMXCoreContext *core) {
  CHECK(core==atomic_context);return atomic;
}
static int MPI_Abort(MPI_Comm communicator,int code) {
  (void)communicator;(void)code;_Exit(77);
}
typedef struct {
  OpenMXCoreContext *core;
  const OpenMXNeutralGridMetadataView *metadata;
  int species;
} OpenMXVpsReadInput;
#include "neutral_metadata_input.inc"

static void read_pao(OpenMXCoreContext *core,const OpenMXNeutralGridMetadataView *metadata,int spe,double value) {
  FILE *file=fopen("metadata.pao","w");CHECK(file);
  CHECK(fprintf(file,"AtomSpecies %.17g\ngrid.num.output 3\nPAO.Lmax 0\nPAO.Mul 1\n",value)>0);
  CHECK(fclose(file)==0);
  Read_PAO(core,metadata,spe,1,3,"metadata.pao");
}
static int read_vps(OpenMXCoreContext *core,const OpenMXNeutralGridMetadataView *metadata,int spe,
                    double value,char *error,size_t capacity) {
  FILE *file=fopen("metadata.vps","w");CHECK(file);
  CHECK(fprintf(file,"AtomSpecies %.17g\ngrid.num.output 6\n<project.energies\n0\nproject.energies>\n",value)>0);
  CHECK(fclose(file)==0);
  OpenMXVpsReadInput input={core,metadata,spe};
  return openmx_core_with_vps_file("metadata.vps",1,Read_VPS,&input,error,capacity);
}

int main(int argc,char **argv) {
  char error[512];
  OpenMXCoreContext *owner=openmx_core_create_host(error,sizeof(error));
  OpenMXCoreContext *other=openmx_core_create_host(error,sizeof(error));
  OpenMXNeutralGridMetadataShape shape={2};OpenMXNeutralGridMetadataView metadata,decoy;
  CHECK(owner&&other);
  CHECK(openmx_core_neutral_grid_metadata_prepare(owner,&shape,&metadata,error,sizeof(error)));
  CHECK(openmx_core_neutral_grid_metadata_prepare(other,&shape,&decoy,error,sizeof(error)));
  decoy.counts[0]=31;decoy.spacing[0]=.75;decoy.spacing[1]=.5;
  int numbers[2],meshes[2]={3,3},lmax[2],multiplicity[2];
  double cutoff[2],density[2][5],backup[2][5];
  atomic_context=owner;atomic.species.species=2;
  atomic.species.atomic_number=numbers;atomic.species.pao_mesh=meshes;
  atomic.species.pao_lmax=lmax;atomic.species.pao_multiplicity=multiplicity;atomic.species.cutoff=cutoff;
  atomic.density.values=&density[0][0];atomic.density_backup.values=&backup[0][0];
  atomic.density.extent[0]=atomic.density_backup.extent[0]=2;
  atomic.density.extent[1]=atomic.density_backup.extent[1]=5;
  if (argc>1&&!strncmp(argv[1],"boundary-",9)) {
    if (!strcmp(argv[1],"boundary-empty"))openmx_core_neutral_grid_metadata_release(owner);
    CHECK(!metadata_input_boundary(!strcmp(argv[1],"boundary-context")?NULL:owner,
          !strcmp(argv[1],"boundary-shape")?3:2,error,sizeof(error)));
    CHECK(strstr(error,"neutral grid metadata unavailable or shape mismatch"));puts(error);
  } else if (argc>3) {
    read_pao(owner,&metadata,1,strtod(argv[2],NULL));
    CHECK(!read_vps(owner,&metadata,1,strtod(argv[3],NULL),error,sizeof(error)));
    CHECK(strstr(error,"Not the same atom in PAO and VPS"));puts(error);
  } else {
    /* Expected integer species and ghost resets are independent of the
       production raw-value comparison and integer conversion. */
    static const struct { double pao,vps,raw;int number,zero_density; } cases[]={
      {.25,.25,.25,1,0},{.75,.75,.75,1,0},{1e-16,1e-16,1e-16,0,0},{1e-310,1e-310,1e-310,0,0},
      {1,1,1,1,0},{14,14,14,14,0},{0,0,0,0,0},{-0.,0,-0.,0,0},
      {14,0,0,0,1},{.25,0,0,0,1}
    };
    for (int k=0;k<(int)(sizeof(cases)/sizeof(cases[0]));++k) {
      const int spe=k%2;metadata.counts[spe]=73;numbers[0]=numbers[1]=-9;
      for(int s=0;s<2;++s)for(int i=0;i<5;++i){density[s][i]=10*s+i+.125;backup[s][i]=20*s+i+.25;}
      read_pao(owner,&metadata,spe,cases[k].pao);
      CHECK(!memcmp(&metadata.spacing[spe],&cases[k].pao,sizeof(double)));
      CHECK(read_vps(owner,&metadata,spe,cases[k].vps,error,sizeof(error)));
      CHECK(!memcmp(&metadata.spacing[spe],&cases[k].raw,sizeof(double)));
      CHECK(numbers[spe]==cases[k].number&&numbers[1-spe]==-9&&metadata.counts[spe]==73);
      for(int s=0;s<2;++s)for(int i=0;i<5;++i){
        const int zero=cases[k].zero_density&&s==spe&&i>=1&&i<=3;
        CHECK(density[s][i]==(zero?0:10*s+i+.125));CHECK(backup[s][i]==(zero?0:20*s+i+.25));
      }
      CHECK(decoy.counts[0]==31&&decoy.spacing[0]==.75&&decoy.spacing[1]==.5);
      CHECK(metadata_input_boundary(owner,2,error,sizeof(error)));
    }
    /* The reusable VPS reader rejects negative species before the callback. */
    read_pao(owner,&metadata,1,.25);
    CHECK(!read_vps(owner,&metadata,1,-.25,error,sizeof(error)));
    CHECK(metadata.spacing[1]==.25&&numbers[1]==1);
    puts("PASS: actual PAO/VPS readers, fractional/ghost/signed-zero species and independent owners");
  }
  openmx_core_destroy(other);openmx_core_destroy(owner);
  return 0;
}

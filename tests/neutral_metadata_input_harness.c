/* Exercise the actual PAO/VPS AtomSpecies blocks over two independent Core
   owners. File-token retrieval and MPI termination are modeled. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "openmx_core_neutral_grid_metadata.h"
#include "openmx_core_atomic.h"
#define CHECK(c) do { if (!(c)) { fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#c); exit(2); } } while (0)
static double supplied_value;
static int expect_mismatch, finalized;
static int mpi_comm_level1=3;
static void input_double(const char *key,double *value,double fallback) {
  CHECK(!strcmp(key,"AtomSpecies")&&fallback==0);*value=supplied_value;
}
static int MPI_Finalize(void) { finalized=1;return 0; }
static int MPI_Abort(int communicator,int code) {
  CHECK(communicator==3&&code==1);_Exit(77);
}
static void exit_check(void) {
  if (expect_mismatch) {
    if (!finalized) { fputs("FAIL: mismatch did not finalize MPI\n",stderr);_Exit(2); }
    fputs("PASS: exact raw AtomSpecies mismatch rejected\n",stderr);
  }
}
#include "neutral_metadata_input.inc"

int main(int argc,char **argv) {
  char error[256];
  OpenMXCoreContext *owner=openmx_core_create_host(error,sizeof(error));
  OpenMXCoreContext *other=openmx_core_create_host(error,sizeof(error));
  OpenMXNeutralGridMetadataShape shape={2};OpenMXNeutralGridMetadataView metadata,decoy;
  CHECK(owner&&other);
  CHECK(openmx_core_neutral_grid_metadata_prepare(owner,&shape,&metadata,error,sizeof(error)));
  CHECK(openmx_core_neutral_grid_metadata_prepare(other,&shape,&decoy,error,sizeof(error)));
  decoy.counts[0]=31;decoy.spacing[0]=.75;decoy.spacing[1]=.5;
  int numbers[2],meshes[2]={3,3};double density[2][5],backup[2][5];
  OpenMXAtomicViews atomic;memset(&atomic,0,sizeof(atomic));
  atomic.species.atomic_number=numbers;atomic.species.pao_mesh=meshes;
  atomic.density.values=&density[0][0];atomic.density_backup.values=&backup[0][0];
  atomic.density.extent[0]=atomic.density_backup.extent[0]=2;
  atomic.density.extent[1]=atomic.density_backup.extent[1]=5;
  if (argc>1&&!strncmp(argv[1],"boundary-",9)) {
    if (!strcmp(argv[1],"boundary-empty"))openmx_core_neutral_grid_metadata_release(owner);
    metadata_input_boundary(!strcmp(argv[1],"boundary-context")?NULL:owner,
                            !strcmp(argv[1],"boundary-shape")?3:2);
    CHECK(0);
  }
  atexit(exit_check);
  if (argc>3) {
    expect_mismatch=1;supplied_value=strtod(argv[2],NULL);metadata_read_pao(&metadata,&atomic,1);
    supplied_value=strtod(argv[3],NULL);metadata_read_vps(&metadata,&atomic,1);CHECK(0);
  }
  /* Explicit expected atom number and density-reset behavior are independent
     of the production comparison and its integer/fractional conversions. */
  static const struct { double pao,vps,raw;int number,zero_density; } cases[]={
    {.25,.25,.25,1,0},{.75,.75,.75,1,0},{1e-16,1e-16,1e-16,0,0},
    {1,1,1,1,0},{14,14,14,14,0},{0,0,0,0,0},{-0.,0,-0.,0,0},
    {14,0,0,0,1},{.25,0,0,0,1},{14,-.25,0,0,1}
  };
  for (int k=0;k<(int)(sizeof(cases)/sizeof(cases[0]));++k) {
    const int spe=k%2;metadata.counts[spe]=73;
    numbers[0]=numbers[1]=-9;
    for(int s=0;s<2;++s)for(int i=0;i<5;++i){density[s][i]=10*s+i+.125;backup[s][i]=20*s+i+.25;}
    supplied_value=cases[k].pao;metadata_read_pao(&metadata,&atomic,spe);
    CHECK(!memcmp(&metadata.spacing[spe],&cases[k].pao,sizeof(double)));
    supplied_value=cases[k].vps;metadata_read_vps(&metadata,&atomic,spe);
    CHECK(!memcmp(&metadata.spacing[spe],&cases[k].raw,sizeof(double)));
    CHECK(numbers[spe]==cases[k].number&&numbers[1-spe]==-9&&metadata.counts[spe]==73);
    for(int s=0;s<2;++s)for(int i=0;i<5;++i){
      const int zero=cases[k].zero_density&&s==spe&&i>=1&&i<=3;
      CHECK(density[s][i]==(zero?0:10*s+i+.125));CHECK(backup[s][i]==(zero?0:20*s+i+.25));
    }
    CHECK(decoy.counts[0]==31&&decoy.spacing[0]==.75&&decoy.spacing[1]==.5);
    metadata_input_boundary(owner,2);
  }
  openmx_core_destroy(other);openmx_core_destroy(owner);
  puts("PASS: actual PAO/VPS fractional, ghost, signed-zero and independent-owner input blocks");return 0;
}

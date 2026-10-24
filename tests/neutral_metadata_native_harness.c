/* Actual case-2 metadata allocation and final-release blocks over the real
   C++ owner. Other case-2 allocations and MPI abort are outside this fixture. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "openmx_core_neutral_grid_metadata.h"
#include "openmx_core_neutral_grid.h"
#define CHECK(c) do{if(!(c)){fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#c);exit(2);}}while(0)
static int SpeciesNum;
static int mpi_comm_level1=3;
static OpenMXCoreContext *abort_context;
static OpenMXNeutralGridMetadataView abort_snapshot;
int MPI_Abort(int communicator,int code){
  CHECK(communicator==3&&code==1&&abort_context);OpenMXNeutralGridMetadataView view;char error[256];
  CHECK(openmx_core_neutral_grid_metadata_get(abort_context,&view,error,sizeof(error)));
  CHECK(view.counts==abort_snapshot.counts&&view.spacing==abort_snapshot.spacing&&view.species==abort_snapshot.species);
  CHECK(view.counts[0]==17&&view.spacing[0]==.25);_Exit(77);
}
#include "neutral_metadata_native.inc"
static OpenMXNeutralGridMetadataView get(OpenMXCoreContext *context){
  OpenMXNeutralGridMetadataView view;char error[256];CHECK(openmx_core_neutral_grid_metadata_get(context,&view,error,sizeof(error)));return view;
}
static void check_metadata(OpenMXCoreContext *context,int seed){
  const OpenMXNeutralGridMetadataView view=get(context);
  CHECK(view.species==(size_t)SpeciesNum&&view.counts&&view.spacing);
  for(int s=0;s<SpeciesNum;s++){view.counts[s]=seed+31*s;view.spacing[s]=seed*.125+s*.0625;}
  for(int s=0;s<SpeciesNum;s++)CHECK(view.counts[s]==seed+31*s&&view.spacing[s]==seed*.125+s*.0625);
  const double raw_species[4]={.25,.75,-0.0,14};
  for(int i=0;i<4;i++){view.spacing[SpeciesNum-1]=raw_species[i];CHECK(!memcmp(&view.spacing[SpeciesNum-1],&raw_species[i],sizeof(double)));}
}
static void check_empty(OpenMXCoreContext *context){
  const OpenMXNeutralGridMetadataView view=get(context);CHECK(!view.counts&&!view.spacing&&!view.species);
}
int main(int argc,char **argv){
  char error[256];OpenMXCoreContext *context=openmx_core_create_host(error,sizeof(error)),*other=openmx_core_create_host(error,sizeof(error));CHECK(context&&other);
  const OpenMXNeutralGridMetadataShape other_shape={5};OpenMXNeutralGridMetadataView other_view;
  CHECK(openmx_core_neutral_grid_metadata_prepare(other,&other_shape,&other_view,error,sizeof(error)));other_view.counts[4]=83;other_view.spacing[4]=.75;
  const OpenMXNeutralGridShape grid_shape={2,13};OpenMXNeutralGridTables grid;
  CHECK(openmx_core_neutral_grid_prepare(context,&grid_shape,&grid,error,sizeof(error)));grid.values[129]=91;
  if(argc>1){
    SpeciesNum=3;native_allocate(context);OpenMXNeutralGridMetadataView initial=get(context);initial.counts[0]=17;initial.spacing[0]=.25;abort_context=context;abort_snapshot=get(context);
    if(!strcmp(argv[1],"shape"))SpeciesNum=0;else if(!strcmp(argv[1],"range"))SpeciesNum=-1;
    native_allocate(!strcmp(argv[1],"context")?NULL:context);CHECK(0);
  }
  int cases=0;const int species[4]={1,7,3,19};
  for(int shape=0;shape<4;shape++){
    SpeciesNum=species[shape];
    for(int repeat=0;repeat<3;repeat++){
      native_allocate(context);check_metadata(context,shape*10+repeat);CHECK(grid.values[129]==91&&other_view.counts[4]==83&&other_view.spacing[4]==.75);++cases;
    }
    native_free(context);check_empty(context);native_free(context);check_empty(context);
  }
  SpeciesNum=5;native_allocate(context);check_metadata(context,19);openmx_core_neutral_grid_release(context);CHECK(get(context).counts[0]==19);
  native_free(context);check_empty(context);CHECK(other_view.counts[4]==83);openmx_core_destroy(context);openmx_core_destroy(other);
  printf("PASS: %d actual metadata allocation cases, raw fractional/ghost AtomSpecies, release/clear, independent contexts and main-grid lifetime\n",cases);return 0;
}

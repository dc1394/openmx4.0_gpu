/* Real production case-4 allocation/free adapters over the actual C++ owner.
   MPI abort is modeled. Optional legacy orbital storage stays separately owned. */
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "openmx_core_neutral_grid.h"
#define CHECK(c) do{if(!(c)){fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#c);exit(2);}}while(0)
static int SpeciesNum,Max_TGN_EH0,Energy_Decomposition_flag,remake_headfile,alloc_first[64];
static int Spe_MaxL_Basis[3]={1,0,2},basis[3][3]={{1,2,0},{3,0,0},{2,1,2}},*Spe_Num_Basis[3]={basis[0],basis[1],basis[2]};
static double ****Arho_EH0_Orb;
static int mpi_comm_level1=3;
static OpenMXCoreContext *abort_context;
static OpenMXNeutralGridTables abort_snapshot;
int MPI_Abort(int communicator,int code){
  CHECK(communicator==3&&code==1&&abort_context);OpenMXNeutralGridTables current;char error[256];
  CHECK(openmx_core_neutral_grid_get(abort_context,&current,error,sizeof(error)));
  CHECK(current.values==abort_snapshot.values&&current.values[0]==17);
  _Exit(77);
}
#include "neutral_grid_native.inc"
static OpenMXNeutralGridTables get(OpenMXCoreContext *context){
  OpenMXNeutralGridTables view;char error[256];CHECK(openmx_core_neutral_grid_get(context,&view,error,sizeof(error)));return view;
}
static void check_grid(OpenMXCoreContext *context,double seed){
  const OpenMXNeutralGridTables view=get(context);CHECK(view.extent[0]==5&&view.extent[1]==(size_t)SpeciesNum&&view.extent[2]==(size_t)Max_TGN_EH0);
  for(int c=0;c<5;c++){
    for(int s=0;s<SpeciesNum;s++){
      CHECK(openmx_core_neutral_grid_row(&view,c,s)==view.values+(c*SpeciesNum+s)*Max_TGN_EH0);
      for(int p=0;p<Max_TGN_EH0;p++)openmx_core_neutral_grid_row(&view,c,s)[p]=seed+10000*c+100*s+p;
    }
  }
  for(int c=0;c<5;c++)for(int s=0;s<SpeciesNum;s++)for(int p=0;p<Max_TGN_EH0;p++)CHECK(view.values[(c*SpeciesNum+s)*Max_TGN_EH0+p]==seed+10000*c+100*s+p);
  if(Energy_Decomposition_flag){
    CHECK(Arho_EH0_Orb);for(int s=0;s<SpeciesNum;s++)for(int p=0;p<Max_TGN_EH0;p++)for(int l=0;l<=Spe_MaxL_Basis[s];l++)for(int m=0;m<Spe_Num_Basis[s][l];m++)Arho_EH0_Orb[s][p][l][m]=seed+p+l+m;
  }
}
static void check_empty(OpenMXCoreContext *context){
  const OpenMXNeutralGridTables view=get(context);CHECK(!view.values&&!view.extent[0]);
}
int main(int argc,char **argv){
  char error[256];OpenMXCoreContext *context=openmx_core_create_host(error,sizeof(error)),*other=openmx_core_create_host(error,sizeof(error));CHECK(context&&other);
  const OpenMXNeutralGridShape unrelated={2,11};OpenMXNeutralGridTables other_grid;
  CHECK(openmx_core_neutral_grid_prepare(other,&unrelated,&other_grid,error,sizeof(error)));other_grid.values[109]=91;
  if(argc>1){
    SpeciesNum=2;Max_TGN_EH0=7;Energy_Decomposition_flag=0;alloc_first[32]=1;native_allocate(context);check_grid(context,17);
    abort_context=context;abort_snapshot=get(context);
    if(!strcmp(argv[1],"shape"))SpeciesNum=0;else if(!strcmp(argv[1],"overflow"))SpeciesNum=Max_TGN_EH0=INT_MAX;
    native_allocate(!strcmp(argv[1],"context")?NULL:context);CHECK(0);
  }
  int cases=0;const int species[3]={1,3,2},points[3]={7,23,5};
  for(int orbital=0;orbital<2;orbital++)for(int shape=0;shape<3;shape++){
    SpeciesNum=species[shape];Max_TGN_EH0=points[shape];Energy_Decomposition_flag=orbital;alloc_first[32]=1;Arho_EH0_Orb=NULL;
    for(int repeat=0;repeat<3;repeat++){
      native_allocate(context);CHECK(alloc_first[32]==0);check_grid(context,shape*10+repeat);CHECK(other_grid.values[109]==91);++cases;
    }
    native_free(context);check_empty(context);
    /* Model the next input's allocation-flag initialization before a no-op release. */
    alloc_first[32]=1;Arho_EH0_Orb=NULL;native_free(context);check_empty(context);
    CHECK(other_grid.values[109]==91);
  }
  openmx_core_destroy(context);CHECK(other_grid.values[109]==91);openmx_core_destroy(other);
  printf("PASS: %d actual native allocation cases, five flat row mappings, optional orbital lifecycle, release, independent contexts\n",cases);return 0;
}

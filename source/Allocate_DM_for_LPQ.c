
/**********************************************************************
  Allocate_DM_for_LPQ.c:

     Allocate_DM_for_LPQ.c is a subroutine to allocate DMmu amd DMkmu.

  Log of Allocate_DM_for_LPQ.c:

     20/May/2019  Released by M.Fukuda

***********************************************************************/


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "openmx_common.h" 
#include "mpi.h"
#include "omp.h"
#include "flpq_dm.h"


void Allocate_DM_for_LPQ()
{ 
  int size_DMmu,size_iDMmu;
  int i,j,k,m,h_AN,Gh_AN,Mc_AN,Gc_AN;
  int tno0,tno1,Cwan,Hwan,N,so,spin;

        
  //  n = 0;
  //  for (i=1; i<=atomnum; i++){
  //    wanA  = WhatSpecies[i];
  //    n += Spe_Total_CNO[wanA];
  //  }
  //  n2 = n + 2;

  /* DMmu */

  size_DMmu = 0;
  DMmu = (double*****)malloc(sizeof(double****)*(SpinP_switch+1)); 
  for (k=0; k<=SpinP_switch; k++){
    DMmu[k] = (double****)malloc(sizeof(double***)*(Matomnum+1)); 
    FNAN[0] = 0;
    for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

      if (Mc_AN==0){
        Gc_AN = 0;
        tno0 = 1;
      }
      else{
        Gc_AN = M2G[Mc_AN];
        Cwan = WhatSpecies[Gc_AN];
        tno0 = Spe_Total_NO[Cwan];  
      }    

      DMmu[k][Mc_AN] = (double***)malloc(sizeof(double**)*(FNAN[Gc_AN]+1));
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

        if (Mc_AN==0){
          tno1 = 1;  
        }
        else{
          Gh_AN = natn[Gc_AN][h_AN];
          Hwan = WhatSpecies[Gh_AN];
          tno1 = Spe_Total_NO[Hwan];
        } 

        DMmu[k][Mc_AN][h_AN] = (double**)malloc(sizeof(double*)*tno0); 
        for (i=0; i<tno0; i++){
          DMmu[k][Mc_AN][h_AN][i] = (double*)malloc(sizeof(double)*tno1); 
          for (j=0; j<tno1; j++) DMmu[k][Mc_AN][h_AN][i][j] = 0.0; 
        }
        size_DMmu += tno0*tno1;
      }
    }
  }


  /* iDMmu */  

  size_iDMmu = 0;
  iDMmu = (double*****)malloc(sizeof(double****)*2); 
  for (k=0; k<2; k++){
    iDMmu[k] = (double****)malloc(sizeof(double***)*(Matomnum+1)); 
    FNAN[0] = 0;
    for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

      if (Mc_AN==0){
        Gc_AN = 0;
        tno0 = 1;
      }
      else{
        Gc_AN = M2G[Mc_AN];
        Cwan = WhatSpecies[Gc_AN];
        tno0 = Spe_Total_NO[Cwan];  
      }    

      iDMmu[k][Mc_AN] = (double***)malloc(sizeof(double**)*(FNAN[Gc_AN]+1)); 
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

        if (Mc_AN==0){
          tno1 = 1;  
        }
        else{
          Gh_AN = natn[Gc_AN][h_AN];
          Hwan = WhatSpecies[Gh_AN];
          tno1 = Spe_Total_NO[Hwan];
        } 

        iDMmu[k][Mc_AN][h_AN] = (double**)malloc(sizeof(double*)*tno0); 
        for (i=0; i<tno0; i++){
          iDMmu[k][Mc_AN][h_AN][i] = (double*)malloc(sizeof(double)*tno1); 
          for (j=0; j<tno1; j++)  iDMmu[k][Mc_AN][h_AN][i][j] = 0.0; 
        }
        size_iDMmu += tno0*tno1;
      }
    }
  }
}


  /* PrintMemory */
  //PrintMemory("Allocate_DMmu_for_LPQ: DMmu",     sizeof(double)*size_DMmu,     NULL);
  //PrintMemory("Allocate_DMmu_for_LPQ: iDMmu",     sizeof(double)*size_iDMmu,     NULL);

void Free_Arrays_DM_for_LPQ()
{ 
  int size_DMmu,size_iDMmu;
  int i,j,k,m,h_AN,Gh_AN,Mc_AN,Gc_AN;
  int tno0,tno1,Cwan,Hwan,N,so,spin;

        
  //  n = 0;
  //  for (i=1; i<=atomnum; i++){
  //    wanA  = WhatSpecies[i];
  //    n += Spe_Total_CNO[wanA];
  //  }
  //  n2 = n + 2;

  /* DMmu */

  for (k=0; k<=SpinP_switch; k++){
    for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

      if (Mc_AN==0){
        Gc_AN = 0;
        tno0 = 1;
      }
      else{
        Gc_AN = M2G[Mc_AN];
        Cwan = WhatSpecies[Gc_AN];
        tno0 = Spe_Total_NO[Cwan];  
      }    

      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
        for (i=0; i<tno0; i++){
          free(DMmu[k][Mc_AN][h_AN][i]); 
        }
        free(DMmu[k][Mc_AN][h_AN]); 
      }
      free(DMmu[k][Mc_AN]);
    }
    free(DMmu[k]); 
  }
  free(DMmu); 

  for (k=0; k<2; k++){
    for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

      if (Mc_AN==0){
        Gc_AN = 0;
        tno0 = 1;
      }
      else{
        Gc_AN = M2G[Mc_AN];
        Cwan = WhatSpecies[Gc_AN];
        tno0 = Spe_Total_NO[Cwan];  
      }    

      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
        for (i=0; i<tno0; i++){
          free(iDMmu[k][Mc_AN][h_AN][i]); 
        }
        free(iDMmu[k][Mc_AN][h_AN]); 
      }
      free(iDMmu[k][Mc_AN]);
    }
    free(iDMmu[k]); 
  }
  free(iDMmu); 

}

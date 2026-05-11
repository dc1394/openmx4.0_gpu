/**********************************************************************
  Calc_OLPexp.c:

   Calc_OLPexp.c is a subroutine to calculate overlap integrals 
   with exp(-ib dot r), where b is one of reciprocal vectors
   divided by Kspace_grid1, Kspace_grid2, or Kspace_grid3.  
   The weighted overlap integrals are used to calculate Mmn. 

  Log of Calc_OLPexp.c:

   25/Jan./2026  Released by T.Ozaki 

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include "openmx_common.h"
#include "mpi.h"


void Calc_OLPexp( FILE *fp )
{
  double time0;
  int Mc_AN,Gc_AN,Mh_AN,h_AN,Gh_AN,b,tno0,tno1;
  int i,j,Cwan,Hwan,NO0,NO1,tot_bvector;
  int TNO1,TNO2,wan1,wan2,num;
  int Rnh,spin,N,NumC[4];
  int Nc,GNc,GRc,Nog,Nh,MN,XC_P_switch;
  double x,y,z,bvector[3][3],mbdotr,co,si;
  double Cxyz[4],*Tmp_Vec;
  double *tmp_Orbs_Grid;
  dcomplex ctmp;
  dcomplex **ChiV;
  dcomplex *tmp_ChiV;
  dcomplex **tmp_OLPexp;
  dcomplex *****OLPexp;
  int numprocs,myid,ID,tag=999;
  MPI_Status stat;
  MPI_Request request;

  /* MPI */

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /*****************************************************
                       set bvector
  *****************************************************/

  tot_bvector = 3;

  bvector[0][0] = rtv[1][1]/(double)Kspace_grid1; 
  bvector[0][1] = rtv[1][2]/(double)Kspace_grid1; 
  bvector[0][2] = rtv[1][3]/(double)Kspace_grid1; 

  bvector[1][0] = rtv[2][1]/(double)Kspace_grid2; 
  bvector[1][1] = rtv[2][2]/(double)Kspace_grid2; 
  bvector[1][2] = rtv[2][3]/(double)Kspace_grid2; 

  bvector[2][0] = rtv[3][1]/(double)Kspace_grid3; 
  bvector[2][1] = rtv[3][2]/(double)Kspace_grid3; 
  bvector[2][2] = rtv[3][3]/(double)Kspace_grid3; 
  
  /****************************************************
                     allocation of arrays:
  ****************************************************/

  Tmp_Vec = (double*)malloc(sizeof(double)*2*List_YOUSO[8]*List_YOUSO[7]*List_YOUSO[7]);

  ChiV = (dcomplex**)malloc(sizeof(dcomplex*)*List_YOUSO[7]);
  for (i=0; i<List_YOUSO[7]; i++){
    ChiV[i] = (dcomplex*)malloc(sizeof(dcomplex)*List_YOUSO[11]);
  }

  tmp_ChiV = (dcomplex*)malloc(sizeof(dcomplex)*List_YOUSO[7]);
  tmp_Orbs_Grid = (double*)malloc(sizeof(double)*List_YOUSO[7]);

  tmp_OLPexp = (dcomplex**)malloc(sizeof(dcomplex*)*List_YOUSO[7]);
  for (i=0; i<List_YOUSO[7]; i++){
    tmp_OLPexp[i] = (dcomplex*)malloc(sizeof(dcomplex)*List_YOUSO[7]);
  }

  OLPexp=(dcomplex*****)malloc(sizeof(dcomplex****)*tot_bvector);
  for (b=0; b<tot_bvector; b++){

    OLPexp[b]=(dcomplex****)malloc(sizeof(dcomplex***)*(Matomnum+1));
    FNAN[0] = 0;
    for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

      if (Mc_AN==0){
	Gc_AN = 0;
	tno0 = 1;
      }
      else{
	Gc_AN = M2G[Mc_AN];
	Cwan = WhatSpecies[Gc_AN];
	tno0 = Spe_Total_CNO[Cwan];
      }

      OLPexp[b][Mc_AN] = (dcomplex***)malloc(sizeof(dcomplex**)*(FNAN[Gc_AN]+1));
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	if (Mc_AN==0){
	  tno1 = 1;
	}
	else{
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  tno1 = Spe_Total_CNO[Hwan];
	}

	OLPexp[b][Mc_AN][h_AN] = (dcomplex**)malloc(sizeof(dcomplex*)*tno0);
	for (i=0; i<tno0; i++){
	  OLPexp[b][Mc_AN][h_AN][i] = (dcomplex*)malloc(sizeof(dcomplex)*tno1);
	}
      }
    }
  }

  /*****************************************************
               matrix elements for OLPexp
  *****************************************************/

  for (b=0; b<tot_bvector; b++){
    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

      Gc_AN = M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];
      NO0 = Spe_Total_CNO[Cwan];

      for (i=0; i<NO0; i++){
	for (Nc=0; Nc<GridN_Atom[Gc_AN]; Nc++){

	  GNc = GridListAtom[Mc_AN][Nc];
	  GRc = CellListAtom[Mc_AN][Nc];

	  Get_Grid_XYZ(GNc,Cxyz);
	  x = Cxyz[1] + atv[GRc][1];
	  y = Cxyz[2] + atv[GRc][2];
	  z = Cxyz[3] + atv[GRc][3];

	  mbdotr = -(bvector[b][0]*x + bvector[b][1]*y + bvector[b][2]*z);
	  co = cos(mbdotr);
	  si = sin(mbdotr);

	  ChiV[i][Nc].r = co*Orbs_Grid[Mc_AN][Nc][i];
	  ChiV[i][Nc].i = si*Orbs_Grid[Mc_AN][Nc][i];
	}
      }

      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	Gh_AN = natn[Gc_AN][h_AN];
	Mh_AN = F_G2M[Gh_AN];

	Rnh = ncn[Gc_AN][h_AN];
	Hwan = WhatSpecies[Gh_AN];
	NO1 = Spe_Total_CNO[Hwan];

	/* initialize */

	for (i=0; i<NO0; i++){
	  for (j=0; j<NO1; j++){
	    tmp_OLPexp[i][j] = Complex(0.0,0.0);
	  }
	}

	/* summation of non-zero elements */

	for (Nog=0; Nog<NumOLG[Mc_AN][h_AN]; Nog++){

	  Nc = GListTAtoms1[Mc_AN][h_AN][Nog];
	  Nh = GListTAtoms2[Mc_AN][h_AN][Nog];

	  /* store ChiV in tmp_ChiV */

	  for (i=0; i<NO0; i++){
	    tmp_ChiV[i] = ChiV[i][Nc];
	  }

	  /* store Orbs_Grid in tmp_Orbs_Grid */

	  if (G2ID[Gh_AN]==myid){
	    for (j=0; j<NO1; j++){
	      tmp_Orbs_Grid[j] = Orbs_Grid[Mh_AN][Nh][j];
	    }
	  }
	  else{
	    for (j=0; j<NO1; j++){
	      tmp_Orbs_Grid[j] = Orbs_Grid_FNAN[Mc_AN][h_AN][Nog][j];
	    }
	  }

	  /* integration */

	  for (i=0; i<NO0; i++){
	    ctmp = tmp_ChiV[i];
	    for (j=0; j<NO1; j++){
	      tmp_OLPexp[i][j].r += ctmp.r*tmp_Orbs_Grid[j];
	      tmp_OLPexp[i][j].i += ctmp.i*tmp_Orbs_Grid[j];
	    }
	  }

	} // Nog

	/* OLPexp */

	for (i=0; i<NO0; i++){
	  for (j=0; j<NO1; j++){
	    OLPexp[b][Mc_AN][h_AN][i][j].r = tmp_OLPexp[i][j].r*GridVol;
	    OLPexp[b][Mc_AN][h_AN][i][j].i = tmp_OLPexp[i][j].i*GridVol;
	  }
	}

      } // h_AN
    } // Mc_AN
  } // b

  /****************************************************
                  save OLPexp to fp
  ****************************************************/

  for (b=0; b<tot_bvector; b++){
    for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
      ID = G2ID[Gc_AN];

      if (myid==ID){

	num = 0;
	Mc_AN = F_G2M[Gc_AN];
	wan1 = WhatSpecies[Gc_AN];
	TNO1 = Spe_Total_CNO[wan1];

	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  wan2 = WhatSpecies[Gh_AN];
	  TNO2 = Spe_Total_CNO[wan2];

	  for (i=0; i<TNO1; i++){
	    for (j=0; j<TNO2; j++){
	      Tmp_Vec[num] = OLPexp[b][Mc_AN][h_AN][i][j].r; num++;
	      Tmp_Vec[num] = OLPexp[b][Mc_AN][h_AN][i][j].i; num++;
	    }
	  }
	}

	if (myid!=Host_ID){
	  MPI_Isend(&num, 1, MPI_INT, Host_ID, tag, mpi_comm_level1, &request);
	  MPI_Wait(&request,&stat);
	  MPI_Isend(Tmp_Vec, num, MPI_DOUBLE, Host_ID, tag, mpi_comm_level1, &request);
	  MPI_Wait(&request,&stat);
	}
	else{
	  fwrite(Tmp_Vec, sizeof(double), num, fp);
	}
      }

      else if (ID!=myid && myid==Host_ID){
	MPI_Recv(&num, 1, MPI_INT, ID, tag, mpi_comm_level1, &stat);
	MPI_Recv(Tmp_Vec, num, MPI_DOUBLE, ID, tag, mpi_comm_level1, &stat);
	fwrite(Tmp_Vec, sizeof(double), num, fp);
      }
    }
  }

  /****************************************************
                    freeing of arrays:
  ****************************************************/

  free(Tmp_Vec);

  for (i=0; i<List_YOUSO[7]; i++){
    free(ChiV[i]);
  }
  free(ChiV);

  free(tmp_ChiV);
  free(tmp_Orbs_Grid);

  for (i=0; i<List_YOUSO[7]; i++){
    free(tmp_OLPexp[i]);
  }
  free(tmp_OLPexp);

  for (b=0; b<tot_bvector; b++){

    FNAN[0] = 0;
    for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

      if (Mc_AN==0){
	Gc_AN = 0;
	tno0 = 1;
      }
      else{
	Gc_AN = M2G[Mc_AN];
	Cwan = WhatSpecies[Gc_AN];
	tno0 = Spe_Total_CNO[Cwan];
      }

      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	if (Mc_AN==0){
	  tno1 = 1;
	}
	else{
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  tno1 = Spe_Total_CNO[Hwan];
	}

	for (i=0; i<tno0; i++){
	  free(OLPexp[b][Mc_AN][h_AN][i]);
	}
	free(OLPexp[b][Mc_AN][h_AN]);
      }
      free(OLPexp[b][Mc_AN]);
    }
    free(OLPexp[b]);
  }
  free(OLPexp);

}



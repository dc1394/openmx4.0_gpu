/**********************************************************************
  out_openmx_lpq.c:

     out_openmx_lpq.c is the routine to make output for flpq.

  Log of out_openmx_lpq.c:

     27/Jun/2016  Released by M.Fukuda

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#include "openmx_common.h"
#include "flpq_dm.h"

void out_openmx_lpq(int Cnt_kind, int Calc_CntOrbital_ON, double *****CDM, double *****iCDM)
{

  int i,j,k;
  int numprocs,myid,tag=999,ID;
  char fname[500];
  FILE *fp,*fp2;
  int iv[100];
  double *Tmp_Vec;

  MPI_Status stat;
  MPI_Request request;

  /* MPI */

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  if(myid==Host_ID){

/**********************************/
/* input data from OpenMX_LPQ.dat */
/**********************************/
  sprintf(fname,"OpenMX_LPQ.bin");
  fp = fopen(fname,"wb");
  //sprintf(fname,"OpenMX_LPQ.dat");
  //fp2 = fopen(fname,"w");

    //if ((fp = fopen(fname,"wb")) == NULL){
    //  printf("cannot open %s \n", fname);
    //  exit(0);
    //}

  fwrite(&SpinP_switch      ,sizeof(int),1,fp);
  fwrite(&Solver            ,sizeof(int),1,fp);
  fwrite(&Cnt_switch        ,sizeof(int),1,fp);
  fwrite(&Cnt_kind          ,sizeof(int),1,fp);
  fwrite(&Calc_CntOrbital_ON,sizeof(int),1,fp);
  fwrite(&MD_switch         ,sizeof(int),1,fp);
  fwrite(&level_stdout      ,sizeof(int),1,fp);
  fwrite(&ESM_switch        ,sizeof(int),1,fp);
  fwrite(&TCpyCell          ,sizeof(int),1,fp);
  fwrite(&CpyCell           ,sizeof(int),1,fp);
  fwrite(&Ngrid1,sizeof(int),1,fp);
  fwrite(&Ngrid2,sizeof(int),1,fp);
  fwrite(&Ngrid3,sizeof(int),1,fp);
  fwrite(List_YOUSO,sizeof(int),NYOUSO,fp);
  fwrite(&SpeciesNum,sizeof(int),1,fp);
  fwrite(&Max_FSNAN ,sizeof(int),1,fp);
  //fwrite(&Max_NumOLG,sizeof(int),1,fp);
  //fwrite(&Matomnum      ,sizeof(int),1,fp);
  //fwrite(&MatomnumF     ,sizeof(int),1,fp);
  fwrite(&atomnum       ,sizeof(int),1,fp);
  //fwrite(&NN_A2B_S      ,sizeof(int),1,fp);
  //fwrite(&NN_A2B_R      ,sizeof(int),1,fp);
  fwrite(&ScaleSize,sizeof(double),1,fp);
  fwrite(tv,sizeof(double),16,fp);
  fwrite(rtv,sizeof(double),16,fp);

  for (i=0; i<(atomnum+1); i++){
    fwrite(Gxyz[i],sizeof(double),YOUSO26,fp);
  }

  fwrite(Spe_Total_NO,    sizeof(int),SpeciesNum,fp);
  fwrite(Spe_Total_CNO,   sizeof(int),SpeciesNum,fp);
  fwrite(Spe_Num_Mesh_PAO,sizeof(int),SpeciesNum,fp);
  fwrite(Spe_MaxL_Basis,  sizeof(int),SpeciesNum,fp);


  for (i=0; i<SpeciesNum; i++){
    fwrite(Spe_Num_Basis[i],sizeof(int),Supported_MaxL+1,fp);
  }  
   
  for (i=0; i<SpeciesNum; i++){
    fwrite(Spe_Num_CBasis[i],sizeof(int),Supported_MaxL+1,fp);
  }  
  
  for (i=0; i<List_YOUSO[18]; i++){
    fwrite(Spe_PAO_RV[i],sizeof(double),List_YOUSO[21],fp);
  }

  for (i=0; i<List_YOUSO[18]; i++){
    for (j=0; j<=List_YOUSO[25]; j++){
      for (k=0; k<List_YOUSO[24]; k++){
        fwrite(Spe_PAO_RWF[i][j][k],sizeof(double),List_YOUSO[21],fp);
      }
    }
  }

  fwrite(Spe_Atom_Cut1, sizeof(double),SpeciesNum,fp);

  for (i=0; i<SpeciesNum; i++){
    fwrite(Spe_Specified_Num[i],sizeof(int),Spe_Total_NO[i],fp);
  }

  for (i=0; i<SpeciesNum; i++){
    for (j=0; j<Spe_Total_NO[i]; j++){
      fwrite(Spe_Trans_Orbital[i][j],sizeof(int),List_YOUSO[24],fp);
    }
  }

  fwrite(Spe_WhatAtom, sizeof(int),SpeciesNum,fp);
  fwrite(Spe_Core_Charge, sizeof(double),SpeciesNum,fp);
  fwrite(InitN_USpin,sizeof(double),atomnum+1,fp);
  fwrite(InitN_DSpin,sizeof(double),atomnum+1,fp);
  
  //fwrite(M2G,sizeof(int),Matomnum+2,fp);

  //fwrite(F_M2G,sizeof(int),Matomnum+MatomnumF+1,fp);

  //fwrite(F_G2M,sizeof(int),atomnum+1,fp);
  
  fwrite(WhatSpecies,sizeof(int),atomnum+1,fp);
  
  fwrite(GridN_Atom,sizeof(int),atomnum+1,fp);
  
  fwrite(FNAN,sizeof(int),atomnum+1,fp);

  for (i=0; i<=atomnum; i++){
    fwrite(ncn[i],sizeof(int),(int)(Max_FSNAN*ScaleSize)+1,fp);
  }

  //fwrite(G2ID,sizeof(int),atomnum+1,fp);

  //fwrite(F_Snd_Num,sizeof(int),Num_Procs,fp);

  //fwrite(S_Snd_Num,sizeof(int),Num_Procs,fp);

  //fwrite(F_Rcv_Num,sizeof(int),Num_Procs,fp);

  //fwrite(S_Rcv_Num,sizeof(int),Num_Procs,fp);

  //fwrite(F_TopMAN,sizeof(int),Num_Procs,fp);

  //for (i=0; i<numprocs; i++){
  //  fwrite(Snd_MAN[i],sizeof(int),(F_Snd_Num[i]+S_Snd_Num[i]+1),fp);
  //}

  //for (i=0; i<numprocs; i++){
  //  fwrite(Snd_GAN[i],sizeof(int),(F_Snd_Num[i]+S_Snd_Num[i]+1),fp);
  //}

  //for (i=0; i<numprocs; i++){
  //  fwrite(Rcv_GAN[i],sizeof(int),(F_Rcv_Num[i]+S_Rcv_Num[i]+1),fp);
  //}

  //fwrite(Num_Snd_Grid_A2B,sizeof(int),Num_Procs,fp);

  //fwrite(Num_Rcv_Grid_A2B,sizeof(int),Num_Procs,fp);

  //for (ID=0; ID<numprocs; ID++){
  //  fwrite(Index_Snd_Grid_A2B[ID],sizeof(int),3*Num_Snd_Grid_A2B[ID],fp);
  //}  
  //
  //for (ID=0; ID<numprocs; ID++){
  //  fwrite(Index_Rcv_Grid_A2B[ID],sizeof(int),3*Num_Rcv_Grid_A2B[ID],fp);
  //}  

  {
  int n,TN;
  TN = (2*CpyCell+1)*(2*CpyCell+1)*(2*CpyCell+1) - 1;

    for (i=0; i<(TN+1); i++){
      fwrite(atv[i],sizeof(double),4,fp);
    }

    n = 2*CpyCell + 4;
    for (i=0; i<n; i++){
      for (j=0; j<n; j++){
        fwrite(ratv[i][j],sizeof(int),n,fp);
      }
    }

    for (i=0; i<(TN+1); i++){
      fwrite(atv_ijk[i],sizeof(int),4,fp);
    }
  }

  for (i=0; i<=atomnum; i++){
    fwrite(natn[i],sizeof(int),((int)(Max_FSNAN*ScaleSize)+1),fp);
  }

  fwrite(&flag_energy_range_DM,sizeof(int),1,fp);
  if(flag_energy_range_DM==1){
    fwrite(DM_energy_range,sizeof(double),2,fp);
  }

  }/* if(myid==Host_ID) */

    /**********************************************
                    CntCoes
    ***********************************************/
  if (Cnt_switch==1){
    int num;
    int L0,Mul0,M0,al,p;
    int spin,Mc_AN,Gc_AN,wan;

    Tmp_Vec = (double*)malloc(sizeof(double)*List_YOUSO[8]*List_YOUSO[7]*List_YOUSO[7]);

    for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){

      wan = WhatSpecies[Gc_AN];
      ID = G2ID[Gc_AN];

      if (myid==ID){

        Mc_AN = F_G2M[Gc_AN];

        al = -1;
        num = 0;
        for (L0=0; L0<=Spe_MaxL_Basis[wan]; L0++){
	  for (Mul0=0; Mul0<Spe_Num_CBasis[wan][L0]; Mul0++){
	    for (M0=0; M0<=2*L0; M0++){
	      al++;
	      for (p=0; p<Spe_Specified_Num[wan][al]; p++){
	        Tmp_Vec[num] = CntCoes[Mc_AN][al][p];
                num++;
	      }
	    }
	  }
        }

        if (myid!=Host_ID){
          MPI_Isend(&num, 1, MPI_INT, Host_ID, tag, mpi_comm_level1, &request);
          MPI_Wait(&request,&stat);
          MPI_Isend(&Tmp_Vec[0], num, MPI_DOUBLE, Host_ID, tag, mpi_comm_level1, &request);
          MPI_Wait(&request,&stat);
        }
        else{
          fwrite(Tmp_Vec, sizeof(double), num, fp);
        }
       
      }

      else if (ID!=myid && myid==Host_ID){
        MPI_Recv(&num, 1, MPI_INT, ID, tag, mpi_comm_level1, &stat);
        MPI_Recv(&Tmp_Vec[0], num, MPI_DOUBLE, ID, tag, mpi_comm_level1, &stat);
          fwrite(Tmp_Vec, sizeof(double), num, fp);
      }

      ///**********************************************
      //                   write 
      //***********************************************/
      //
      //if (myid==Host_ID){

      //  //fprintf(fp,"\nAtom=%2d\n",Gc_AN);
      //  //fprintf(fp,"Basis specification  %s\n",SpeBasis[wan]);

      //  //fprintf(fp,"Contraction coefficients  p=");
      //  //for (i=0; i<List_YOUSO[24]; i++){
      //  //  fprintf(fp,"       %i  ",i);
      //  //}
      //  //fprintf(fp,"\n");

      //  al = -1;
      //  num = 0;

      //  for (L0=0; L0<=Spe_MaxL_Basis[wan]; L0++){
      //    for (Mul0=0; Mul0<Spe_Num_CBasis[wan][L0]; Mul0++){
      //      for (M0=0; M0<=2*L0; M0++){
      //        al++;

      //        //fprintf(fp,"Atom=%3d  L=%2d  Mul=%2d  M=%2d  ",Gc_AN,L0,Mul0,M0);
      //        for (p=0; p<Spe_Specified_Num[wan][al]; p++){
      //          fprintf(fp,"%9.5f ",Tmp_Vec[num]);
      //          num++;
      //        }
      //        for (p=Spe_Specified_Num[wan][al]; p<List_YOUSO[24]; p++){
      //          fprintf(fp,"  0.00000 ");
      //        }
      //        fprintf(fp,"\n");
      //      }
      //    }
      //  }
      //} /* if (myid==Host_ID) */
    }   /* Gc_AN */

    /* freeing of Tmp_Vec */
    free(Tmp_Vec);

    ///* CntCoes */

    //if (Cnt_switch==1){
    //  for (i=0; i<=(Matomnum+MatomnumF); i++){
    //    for (j=0; j<List_YOUSO[7]; j++){
    //      fwrite(CntCoes[i][j],sizeof(double),List_YOUSO[24],fp);
    //    }
    //  }
    //}

  }


  MPI_Barrier(mpi_comm_level1);

  if(myid==Host_ID){
    fclose(fp);
  }


  Tmp_Vec = (double*)malloc(sizeof(double)*List_YOUSO[8]*List_YOUSO[7]*List_YOUSO[7]);

  /***************************************************************
                        density matrix DM
  ****************************************************************/
  {
    int num;
    int spin,Mc_AN,Gc_AN,wan1,TNO1;
    int h_AN,Gh_AN,wan2,TNO2;

    for (spin=0; spin<=SpinP_switch; spin++){
      if(myid==Host_ID){
        sprintf(fname,"OpenMX_LPQ_DM/OpenMX_LPQ_DM_%d.bin",spin);
        //printf("%s\n",fname);
        fp = fopen(fname,"wb");
      }
      MPI_Barrier(mpi_comm_level1);

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
                Tmp_Vec[num] = CDM[spin][Mc_AN][h_AN][i][j];
                num++;
              }
            }
          }

          if (myid!=Host_ID){
            MPI_Isend(&num, 1, MPI_INT, Host_ID, tag, mpi_comm_level1, &request);
            MPI_Wait(&request,&stat);
            MPI_Isend(&Tmp_Vec[0], num, MPI_DOUBLE, Host_ID, tag, mpi_comm_level1, &request);
            MPI_Wait(&request,&stat);
          }
          else{
            fwrite(Tmp_Vec, sizeof(double), num, fp);
          }
        }

        else if (ID!=myid && myid==Host_ID){
          MPI_Recv(&num, 1, MPI_INT, ID, tag, mpi_comm_level1, &stat);
          MPI_Recv(&Tmp_Vec[0], num, MPI_DOUBLE, ID, tag, mpi_comm_level1, &stat);
          fwrite(Tmp_Vec, sizeof(double), num, fp);
        }

      }  

      MPI_Barrier(mpi_comm_level1);

      if(myid==Host_ID){
        fclose(fp);
      }
      MPI_Barrier(mpi_comm_level1);
    }
  }

  //if(Solver==4){
  if((Solver==4)||(SpinP_switch>1)){
  /***************************************************************
                        density matrix iDM
  ****************************************************************/
  {
    int num;
    int spin,Mc_AN,Gc_AN,wan1,TNO1;
    int h_AN,Gh_AN,wan2,TNO2;

    for (spin=0; spin<2; spin++){
      if(myid==Host_ID){
        sprintf(fname,"OpenMX_LPQ_DM/OpenMX_LPQ_DM_%d.bin",spin+SpinP_switch+1);
        fp = fopen(fname,"wb");
      }
      MPI_Barrier(mpi_comm_level1);

      for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
        ID = G2ID[Gc_AN];

        if (myid==ID){
          //printf("ID=%d\n",ID);

          num = 0;

          Mc_AN = F_G2M[Gc_AN];
          //printf("Mc_AN=%d, Gc_AN=%d\n",Mc_AN,Gc_AN);
          wan1 = WhatSpecies[Gc_AN];
          TNO1 = Spe_Total_CNO[wan1];
          for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
            Gh_AN = natn[Gc_AN][h_AN];
            wan2 = WhatSpecies[Gh_AN];
            TNO2 = Spe_Total_CNO[wan2];

            for (i=0; i<TNO1; i++){
              for (j=0; j<TNO2; j++){
                Tmp_Vec[num] = iCDM[spin][Mc_AN][h_AN][i][j];
                num++;
              }
            }
          }

          if (myid!=Host_ID){
            MPI_Isend(&num, 1, MPI_INT, Host_ID, tag, mpi_comm_level1, &request);
            MPI_Wait(&request,&stat);
            MPI_Isend(&Tmp_Vec[0], num, MPI_DOUBLE, Host_ID, tag, mpi_comm_level1, &request);
            MPI_Wait(&request,&stat);
          }
          else{
            fwrite(Tmp_Vec, sizeof(double), num, fp);
            //for (i=0; i<num; i++){
            //  printf("%4d %18.8E\n",i,Tmp_Vec[i]);
            //}
          }
        }

        else if (ID!=myid && myid==Host_ID){
          MPI_Recv(&num, 1, MPI_INT, ID, tag, mpi_comm_level1, &stat);
          MPI_Recv(&Tmp_Vec[0], num, MPI_DOUBLE, ID, tag, mpi_comm_level1, &stat);
          fwrite(Tmp_Vec, sizeof(double), num, fp);
          //for (i=0; i<num; i++){
          //  printf("%4d %18.8E\n",i,Tmp_Vec[i]);
          //}
        }

      }  

      MPI_Barrier(mpi_comm_level1);
      if(myid==Host_ID){
        fclose(fp);
      }
    }
  }
  } /*if((Solver==4)||(SpinP_switch>1)) */

  /* freeing of Tmp_Vec */
  free(Tmp_Vec);

//  /* write Density Matrix */
//  {
//    int spin,MA_AN,GA_AN,wanA,tnoA;
//    int LB_AN,GB_AN,wanB,tnoB;
//
//    for (spin=0; spin<=SpinP_switch; spin++) {
//      for (MA_AN=1; MA_AN<=Matomnum; MA_AN++) {
//        GA_AN = M2G[MA_AN];
//        wanA = WhatSpecies[GA_AN];
//        tnoA = Spe_Total_CNO[wanA];
//        for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
//          GB_AN = natn[GA_AN][LB_AN];
//          wanB = WhatSpecies[GB_AN];
//          tnoB = Spe_Total_CNO[wanB];
//          for (i=0; i<tnoA; i++){
//            fwrite(CDM[spin][MA_AN][LB_AN][i],sizeof(double),tnoB,fp);
//          }
//        }
//      }
//    }
//  }
//
//  /* write imaginary parts of Density Matrix */
//  {
//    int spin,MA_AN,GA_AN,wanA,tnoA;
//    int LB_AN,GB_AN,wanB,tnoB;
//
//    for (spin=0; spin<2; spin++) {
//      for (MA_AN=1; MA_AN<=Matomnum; MA_AN++) {
//        GA_AN = M2G[MA_AN];
//        wanA = WhatSpecies[GA_AN];
//        tnoA = Spe_Total_CNO[wanA];
//        for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
//          GB_AN = natn[GA_AN][LB_AN];
//          wanB = WhatSpecies[GB_AN];
//          tnoB = Spe_Total_CNO[wanB];
//          for (i=0; i<tnoA; i++){
//            fwrite(iCDM[spin][MA_AN][LB_AN][i],sizeof(double),tnoB,fp);
//          }
//        }
//      }
//    }
//  }

  //fclose(fp2);


}


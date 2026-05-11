/**********************************************************************
  Stress.c:
  
     Stress.c is a subroutine to calculate stress on a supercell.
     This subroutine is a modification of Force.c

  Log of Stress.c:

     150326 Kinetic and overlap terms are implemented by yshiihara.
***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "openmx_common.h"
#include "mpi.h"
#include <omp.h>
#include <fftw3.h>

#define measure_time 0

/*** subroutine */
static void MPI_OLP(double *****OLP1);
static void Stress3();
static void Stress4();
static void Stress4B(double *****CDM0);
static void Stress_core();
static void Stress_H0();
static void Stress_H1();
static void Stress_HNL(double *****CDM0, double *****iDM0);
static void Stress_plus_U_dual(double *****CDM0);
static void EH0_TwoCenter(int Gc_AN, int h_AN, double VH0ij[4]); /*from Total_Energy.c*/
static void EH0_TwoCenter_at_Cutoff(int wan1, int wan2, double VH0ij[4]); /*from Total_Energy.c*/

static void dHVNA(int where_flag, int Mc_AN, int h_AN, int q_AN,
		  Type_DS_VNA *****DS_VNA1, 
		  double *****TmpHVNA2, double *****TmpHVNA3, 
		  double **Hx, double **Hy, double **Hz, 
		  double ****H_sts);

static void dHNL(int where_flag,
		 int Mc_AN, int h_AN, int q_AN,
		 double ******DS_NL1,
		 dcomplex ***Hx, dcomplex ***Hy, dcomplex ***Hz,
		 dcomplex *****H_sts);

static void dHNL_SO(
	     double *sumx0r,
	     double *sumy0r, 
	     double *sumz0r, 
	     double *sumx1r,
	     double *sumy1r, 
	     double *sumz1r, 
	     double *sumx2r,
	     double *sumy2r, 
	     double *sumz2r, 
	     double *sumx0i,
	     double *sumy0i, 
	     double *sumz0i, 
	     double *sumx1i,
	     double *sumy1i, 
	     double *sumz1i, 
	     double *sumx2i,
	     double *sumy2i, 
	     double *sumz2i, 
             double fugou,
	     double PFp,
	     double PFm,
	     double ene_p,
	     double ene_m,
	     int l2, int *l,
	     int Mc_AN, int k,  int m,
	     int Mj_AN, int kl, int n,
	     double ******DS_NL1);







double Stress(double *****H0,
	      double ******DS_NL,
	      double *****OLP,
	      double *****CDM,
	      double *****EDM)
{
  /*** variables **/
  static int firsttime=1;
  int Nc,GNc,GRc,Cwan,s1,s2,BN_AB;
  int Mc_AN,Gc_AN,MNc,start_q_AN;
  double x,y,z,dx,dy,dz,tmp0,tmp1,tmp2,tmp3;
  double xx,r2,tot_den;
  double sumx,sumy,sumz,r,dege,pref;
  int i,j,k,l,Hwan,Qwan,so,p0,q,q0;
  int h_AN,Gh_AN,q_AN,Gq_AN;
  int ian,jan,kl,spin,spinmax,al,be,p,size_CDM0,size_iDM0;
  int tno0,tno1,tno2,Mh_AN,Mq_AN,n,num,size1,size2;
  int wanA,wanB,Gc_BN;
  int XC_P_switch;
  double time0;
  double dum,dge;
  double dEx,dEy,dEz;
  double Cxyz[4];
  double *Fx,*Fy,*Fz;
  dcomplex ***Hx;
  dcomplex ***Hy;
  dcomplex ***Hz;
  double ***HUx;
  double ***HUy;
  double ***HUz;
  dcomplex ****NC_HUx;
  dcomplex ****NC_HUy;
  dcomplex ****NC_HUz;
  double **HVNAx;
  double **HVNAy;
  double **HVNAz;
  double *****CDM0;
  double *****iDM0;
  double *tmp_array;
  double *tmp_array2;
  double Re00x,Re00y,Re00z;
  double Re11x,Re11y,Re11z;
  double Re01x,Re01y,Re01z;
  double Im00x,Im00y,Im00z;
  double Im11x,Im11y,Im11z;
  double Im01x,Im01y,Im01z;
  int *Snd_CDM0_Size,*Rcv_CDM0_Size;
  int *Snd_iDM0_Size,*Rcv_iDM0_Size;
  double TStime,TEtime;
  int numprocs,myid,tag=999,ID,IDS,IDR;
  double Stime_atom, Etime_atom;
  /* for OpenMP */
  int OMPID,Nthrds,Nthrds0,Nprocs;
  double stime,etime;

  MPI_Status stat;
  MPI_Request request;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);
  MPI_Barrier(mpi_comm_level1);
  dtime(&TStime);

  /* Stress */
  double **My_Stress_threads;
  double Stress[9],My_Stress[9];
  double sumv1[4],sumv2[4],sumv3[4];
  double sumt1[9],sumt2[9],sumt3[9],sumt4[9],sumt5[9];
  int    Rn1,Rn2;
  double lx1,ly1,lz1,lx2,ly2,lz2,lx,ly,lz;
  
  /*allocation of array*/
  
  /* CDM0 */
  size_CDM0 = 0;
  CDM0 = (double*****)malloc(sizeof(double****)*(SpinP_switch+1)); 
  for (k=0; k<=SpinP_switch; k++){
    CDM0[k] = (double****)malloc(sizeof(double***)*(Matomnum+MatomnumF+1)); 
    FNAN[0] = 0;
    for (Mc_AN=0; Mc_AN<=(Matomnum+MatomnumF); Mc_AN++){
      
      if (Mc_AN==0){
        Gc_AN = 0;
        tno0 = 1;
      }
      else{
        Gc_AN = F_M2G[Mc_AN];
        Cwan = WhatSpecies[Gc_AN];
        tno0 = Spe_Total_CNO[Cwan];  
      }    

      CDM0[k][Mc_AN] = (double***)malloc(sizeof(double**)*(FNAN[Gc_AN]+1)); 
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

        if (Mc_AN==0){
          tno1 = 1;  
        }
        else{
          Gh_AN = natn[Gc_AN][h_AN];
          Hwan = WhatSpecies[Gh_AN];
          tno1 = Spe_Total_CNO[Hwan];
        } 

        CDM0[k][Mc_AN][h_AN] = (double**)malloc(sizeof(double*)*tno0); 
        for (i=0; i<tno0; i++){
          CDM0[k][Mc_AN][h_AN][i] = (double*)malloc(sizeof(double)*tno1); 
          size_CDM0 += tno1;  
        }
      }
    }
  }

  Snd_CDM0_Size = (int*)malloc(sizeof(int)*numprocs);
  Rcv_CDM0_Size = (int*)malloc(sizeof(int)*numprocs);

  if ( SO_switch==1 || (Hub_U_switch==1 && SpinP_switch==3) || Constraint_NCS_switch==1 
       || Zeeman_NCS_switch==1 || Zeeman_NCO_switch==1 ){

    size_iDM0 = 0;
    iDM0 = (double*****)malloc(sizeof(double****)*2); 
    for (k=0; k<2; k++){
      iDM0[k] = (double****)malloc(sizeof(double***)*(Matomnum+MatomnumF+1)); 
      FNAN[0] = 0;
      for (Mc_AN=0; Mc_AN<=(Matomnum+MatomnumF); Mc_AN++){

	if (Mc_AN==0){
	  Gc_AN = 0;
	  tno0 = 1;
	}
	else{
	  Gc_AN = F_M2G[Mc_AN];
	  Cwan = WhatSpecies[Gc_AN];
	  tno0 = Spe_Total_CNO[Cwan];  
	}    

	iDM0[k][Mc_AN] = (double***)malloc(sizeof(double**)*(FNAN[Gc_AN]+1)); 
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	  if (Mc_AN==0){
	    tno1 = 1;  
	  }
	  else{
	    Gh_AN = natn[Gc_AN][h_AN];
	    Hwan = WhatSpecies[Gh_AN];
	    tno1 = Spe_Total_CNO[Hwan];
	  } 

	  iDM0[k][Mc_AN][h_AN] = (double**)malloc(sizeof(double*)*tno0); 
	  for (i=0; i<tno0; i++){
	    iDM0[k][Mc_AN][h_AN][i] = (double*)malloc(sizeof(double)*tno1); 
	    size_iDM0 += tno1;  
	  }
	}
      }
    }

    Snd_iDM0_Size = (int*)malloc(sizeof(int)*numprocs);
    Rcv_iDM0_Size = (int*)malloc(sizeof(int)*numprocs);
  }

  /****************************************************
                      PrintMemory
  ****************************************************/

  /* work later */

  /****************************************************
    CDM to CDM0
  ****************************************************/

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
  
    Gc_AN = M2G[Mc_AN];
    Cwan = WhatSpecies[Gc_AN];
    tno1 = Spe_Total_CNO[Cwan];
  
    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
  
      Gh_AN = natn[Gc_AN][h_AN];
      Hwan = WhatSpecies[Gh_AN];
      tno2 = Spe_Total_CNO[Hwan];
  
      for (spin=0; spin<=SpinP_switch; spin++){
        for (i=0; i<tno1; i++){
	  for (j=0; j<tno2; j++){
            CDM0[spin][Mc_AN][h_AN][i][j] = CDM[spin][Mc_AN][h_AN][i][j]; 
	  }
        }
      }
    }
  }

  /****************************************************
    iDM to iDM0
  ****************************************************/

  if ( SO_switch==1 || (Hub_U_switch==1 && F_U_flag==1 && SpinP_switch==3) || Constraint_NCS_switch==1
       || Zeeman_NCS_switch==1 || Zeeman_NCO_switch==1){

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

      Gc_AN = M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];
      tno1 = Spe_Total_CNO[Cwan];

      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	Gh_AN = natn[Gc_AN][h_AN];
	Hwan = WhatSpecies[Gh_AN];
	tno2 = Spe_Total_CNO[Hwan];

	for (i=0; i<tno1; i++){
	  for (j=0; j<tno2; j++){
	    iDM0[0][Mc_AN][h_AN][i][j] = iDM[0][0][Mc_AN][h_AN][i][j]; 
	    iDM0[1][Mc_AN][h_AN][i][j] = iDM[0][1][Mc_AN][h_AN][i][j]; 
	  }
	}
      }
    }
  }

  /****************************************************
   MPI:

   CDM0
  ****************************************************/

  /***********************************
             set data size
  ************************************/

  for (ID=0; ID<numprocs; ID++){

    IDS = (myid + ID) % numprocs;
    IDR = (myid - ID + numprocs) % numprocs;

    if (ID!=0){
      tag = 999;

      /* find data size to send block data */
      if (F_Snd_Num[IDS]!=0){

        size1 = 0;
        for (spin=0; spin<=SpinP_switch; spin++){
          for (n=0; n<F_Snd_Num[IDS]; n++){
            Mc_AN = Snd_MAN[IDS][n];
            Gc_AN = Snd_GAN[IDS][n];
            Cwan = WhatSpecies[Gc_AN]; 
            tno1 = Spe_Total_CNO[Cwan];
            for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
              Gh_AN = natn[Gc_AN][h_AN];        
              Hwan = WhatSpecies[Gh_AN];
              tno2 = Spe_Total_CNO[Hwan];
              for (i=0; i<tno1; i++){
                for (j=0; j<tno2; j++){
                  size1++; 
                } 
              } 
	    }
          }
	}
 
        Snd_CDM0_Size[IDS] = size1;
        MPI_Isend(&size1, 1, MPI_INT, IDS, tag, mpi_comm_level1, &request);
      }
      else{
        Snd_CDM0_Size[IDS] = 0;
      }

      /* receiving of size of data */

      if (F_Rcv_Num[IDR]!=0){
        MPI_Recv(&size2, 1, MPI_INT, IDR, tag, mpi_comm_level1, &stat);
        Rcv_CDM0_Size[IDR] = size2;
      }
      else{
        Rcv_CDM0_Size[IDR] = 0;
      }

      if (F_Snd_Num[IDS]!=0) MPI_Wait(&request,&stat);

    }
    else{
      Snd_CDM0_Size[IDS] = 0;
      Rcv_CDM0_Size[IDR] = 0;
    }
  }

  /***********************************
             data transfer
  ************************************/

  tag = 999;
  for (ID=0; ID<numprocs; ID++){

    IDS = (myid + ID) % numprocs;
    IDR = (myid - ID + numprocs) % numprocs;

    if (ID!=0){

      /*****************************
              sending of data 
      *****************************/

      if (F_Snd_Num[IDS]!=0){

        size1 = Snd_CDM0_Size[IDS];

        /* allocation of array */

        tmp_array = (double*)malloc(sizeof(double)*size1);

        /* multidimentional array to vector array */

        num = 0;
        for (spin=0; spin<=SpinP_switch; spin++){
          for (n=0; n<F_Snd_Num[IDS]; n++){
            Mc_AN = Snd_MAN[IDS][n];
            Gc_AN = Snd_GAN[IDS][n];
            Cwan = WhatSpecies[Gc_AN]; 
            tno1 = Spe_Total_CNO[Cwan];
            for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
              Gh_AN = natn[Gc_AN][h_AN];
              Hwan = WhatSpecies[Gh_AN];
              tno2 = Spe_Total_CNO[Hwan];
              for (i=0; i<tno1; i++){
                for (j=0; j<tno2; j++){
                  tmp_array[num] = CDM[spin][Mc_AN][h_AN][i][j];
                  num++;
                } 
              } 
	    }
          }
	}

        MPI_Isend(&tmp_array[0], size1, MPI_DOUBLE, IDS, tag, mpi_comm_level1, &request);

      }

      /*****************************
         receiving of block data
      *****************************/

      if (F_Rcv_Num[IDR]!=0){

        size2 = Rcv_CDM0_Size[IDR];
        
        /* allocation of array */
        tmp_array2 = (double*)malloc(sizeof(double)*size2);
        
        MPI_Recv(&tmp_array2[0], size2, MPI_DOUBLE, IDR, tag, mpi_comm_level1, &stat);

        num = 0;
        for (spin=0; spin<=SpinP_switch; spin++){
          Mc_AN = F_TopMAN[IDR] - 1; 
          for (n=0; n<F_Rcv_Num[IDR]; n++){
            Mc_AN++;
            Gc_AN = Rcv_GAN[IDR][n];
            Cwan = WhatSpecies[Gc_AN]; 
            tno1 = Spe_Total_CNO[Cwan];

            for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
              Gh_AN = natn[Gc_AN][h_AN];        
              Hwan = WhatSpecies[Gh_AN];
              tno2 = Spe_Total_CNO[Hwan];
              for (i=0; i<tno1; i++){
                for (j=0; j<tno2; j++){
                  CDM0[spin][Mc_AN][h_AN][i][j] = tmp_array2[num];
                  num++;
		}
	      }
	    }
	  }        
	}

        /* freeing of array */
        free(tmp_array2);
      }

      if (F_Snd_Num[IDS]!=0){
        MPI_Wait(&request,&stat);
        free(tmp_array); /* freeing of array */
      }

    }
  }

  /****************************************************
   MPI:

   iDM0
  ****************************************************/

  if ( SO_switch==1 || (Hub_U_switch==1 && F_U_flag==1 && SpinP_switch==3) || Constraint_NCS_switch==1
       || Zeeman_NCS_switch==1 || Zeeman_NCO_switch==1){
    /* not avairable so far */

    /***********************************
                set data size
    ************************************/

    for (ID=0; ID<numprocs; ID++){

      IDS = (myid + ID) % numprocs;
      IDR = (myid - ID + numprocs) % numprocs;

      if (ID!=0){
	tag = 999;

	/* find data size to send block data */
	if (F_Snd_Num[IDS]!=0){

	  size1 = 0;
	  for (so=0; so<2; so++){
	    for (n=0; n<F_Snd_Num[IDS]; n++){
	      Mc_AN = Snd_MAN[IDS][n];
	      Gc_AN = Snd_GAN[IDS][n];
	      Cwan = WhatSpecies[Gc_AN]; 
	      tno1 = Spe_Total_CNO[Cwan];
	      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
		Gh_AN = natn[Gc_AN][h_AN];        
		Hwan = WhatSpecies[Gh_AN];
		tno2 = Spe_Total_CNO[Hwan];
		for (i=0; i<tno1; i++){
		  for (j=0; j<tno2; j++){
		    size1++; 
		  } 
		} 
	      }
	    }
	  }
 
	  Snd_iDM0_Size[IDS] = size1;
	  MPI_Isend(&size1, 1, MPI_INT, IDS, tag, mpi_comm_level1, &request);

	}
	else{
	  Snd_iDM0_Size[IDS] = 0;
	}

	/* receiving of size of data */

	if (F_Rcv_Num[IDR]!=0){
	  MPI_Recv(&size2, 1, MPI_INT, IDR, tag, mpi_comm_level1, &stat);
	  Rcv_iDM0_Size[IDR] = size2;
	}
	else{
	  Rcv_iDM0_Size[IDR] = 0;
	}

	if (F_Snd_Num[IDS]!=0) MPI_Wait(&request,&stat);

      }
      else{
	Snd_iDM0_Size[IDS] = 0;
	Rcv_iDM0_Size[IDR] = 0;
      }
    }

    /***********************************
               data transfer
    ************************************/

    tag = 999;
    for (ID=0; ID<numprocs; ID++){

      IDS = (myid + ID) % numprocs;
      IDR = (myid - ID + numprocs) % numprocs;

      if (ID!=0){

	/*****************************
	      sending of data 
	*****************************/

	if (F_Snd_Num[IDS]!=0){

	  size1 = Snd_iDM0_Size[IDS];

	  /* allocation of array */

	  tmp_array = (double*)malloc(sizeof(double)*size1);

	  /* multidimentional array to vector array */

	  num = 0;
	  for (so=0; so<2; so++){
	    for (n=0; n<F_Snd_Num[IDS]; n++){
	      Mc_AN = Snd_MAN[IDS][n];
	      Gc_AN = Snd_GAN[IDS][n];
	      Cwan = WhatSpecies[Gc_AN]; 
	      tno1 = Spe_Total_CNO[Cwan];
	      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
		Gh_AN = natn[Gc_AN][h_AN];
		Hwan = WhatSpecies[Gh_AN];
		tno2 = Spe_Total_CNO[Hwan];
		for (i=0; i<tno1; i++){
		  for (j=0; j<tno2; j++){
		    tmp_array[num] = iDM[0][so][Mc_AN][h_AN][i][j];
		    num++;
		  } 
		} 
	      }
	    }
	  }

	  MPI_Isend(&tmp_array[0], size1, MPI_DOUBLE, IDS, tag, mpi_comm_level1, &request);

	}

	/*****************************
          receiving of block data
	*****************************/

	if (F_Rcv_Num[IDR]!=0){

	  size2 = Rcv_iDM0_Size[IDR];
        
	  /* allocation of array */
	  tmp_array2 = (double*)malloc(sizeof(double)*size2);
        
	  MPI_Recv(&tmp_array2[0], size2, MPI_DOUBLE, IDR, tag, mpi_comm_level1, &stat);

	  num = 0;
	  for (so=0; so<2; so++){
	    Mc_AN = F_TopMAN[IDR] - 1; 
	    for (n=0; n<F_Rcv_Num[IDR]; n++){
	      Mc_AN++;
	      Gc_AN = Rcv_GAN[IDR][n];
	      Cwan = WhatSpecies[Gc_AN]; 
	      tno1 = Spe_Total_CNO[Cwan];

	      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
		Gh_AN = natn[Gc_AN][h_AN];        
		Hwan = WhatSpecies[Gh_AN];
		tno2 = Spe_Total_CNO[Hwan];
		for (i=0; i<tno1; i++){
		  for (j=0; j<tno2; j++){
		    iDM0[so][Mc_AN][h_AN][i][j] = tmp_array2[num];
		    num++;
		  }
		}
	      }
	    }        
	  }

	  /* freeing of array */
	  free(tmp_array2);
	}

	if (F_Snd_Num[IDS]!=0){
	  MPI_Wait(&request,&stat);
	  free(tmp_array); /* freeing of array */
	}

      }
    }

  } /* if ( SO_switch==1 || (Hub_U_switch==1 && F_U_flag==1 && SpinP_switch==3) 
     || 1<=Constraint_NCS_switch || Zeeman_NCS_switch==1 || Zeeman_NCO_switch==1) */

  /* Initialization of Stress_Tensor */

  for (i=0; i<9; i++){
    Stress_Tensor[i] = 0.0;
  }

  /* Each stress component will be calculated below... */

  /****************************************************
                      #1 of Stress

              -\int \delta V_H drho_a/depsilon dr
                         and 
                force induced from PCC
              +\int V_XC drho_pcc/depsilon dr

  ****************************************************/

  /* begin(yshiihara) */
  {
    double aden,My_EXC;

    int spin;
    int Ng1,Ng2,Ng3,DN,BN;
    int XC_P_switch_stress;

    double **My_Stress_threads2;
    double Stress2[9];
    
    /* end(yshiihara) */
    
    if (myid==Host_ID && 0<level_stdout){
      printf("  Stress calculation #1\n");fflush(stdout);
    }

    dtime(&stime);

    /*********************************************************
      set RefVxc_Grid, where the CA-LDA exchange-correlation 
      functional is alway used.
    *********************************************************/

    XC_P_switch = 1;
    for (BN_AB=0; BN_AB<My_NumGridB_AB; BN_AB++){
      tot_den = ADensity_Grid_B[BN_AB] + ADensity_Grid_B[BN_AB];
      if (PCC_switch==1) {
	tot_den += PCCDensity_Grid_B[0][BN_AB] + PCCDensity_Grid_B[1][BN_AB];
      }
      RefVxc_Grid_B[BN_AB] = XC_Ceperly_Alder(tot_den,XC_P_switch);
    }
  
    Data_Grid_Copy_B2C_1( RefVxc_Grid_B,  RefVxc_Grid  );
    Data_Grid_Copy_B2C_1( dVHart_Grid_B,  dVHart_Grid  );
    Data_Grid_Copy_B2C_2( Vxc_Grid_B,     Vxc_Grid     );
    Data_Grid_Copy_B2C_2( Density_Grid_B, Density_Grid );

    My_EXC = 0.0;
    XC_P_switch = 0;
    for (BN=0; BN<My_NumGridB_AB; BN++){
      aden = 2.0*ADensity_Grid_B[BN];
      tmp1 = XC_Ceperly_Alder(aden,XC_P_switch);
      My_EXC += aden*tmp1;
    }
    
    /* begin(yshiihara) */ 

    /* get Nthrds0 */  
#pragma omp parallel shared(Nthrds0)
    {
      Nthrds0 = omp_get_num_threads();
    }

    My_Stress_threads=(double**)malloc(sizeof(double*)*(Nthrds0));
    for (i=0; i<Nthrds0; i++){
      My_Stress_threads[i] = (double*)malloc(sizeof(double)*9);
      for (j=0; j<9; j++) My_Stress_threads[i][j] = 0.0;
    }

    for (i=0; i<9; i++){
      Stress[i]    = 0.0;
      My_Stress[i] = 0.0;
    }
    /* end(yshiihara) */
  
#pragma omp parallel shared(myid,Spe_OpenCore_flag,Spe_Atomic_PCC,Spe_VPS_RV,Spe_VPS_XV,Spe_Num_Mesh_VPS,Spe_PAO_RV,Spe_Atomic_Den,Spe_PAO_XV,Spe_Num_Mesh_PAO,time_per_atom,level_stdout,GridVol,Vxc_Grid,RefVxc_Grid,SpinP_switch,F_Vxc_flag,PCC_switch,dVHart_Grid,F_dVHart_flag,Gxyz,atv,MGridListAtom,CellListAtom,GridListAtom,GridN_Atom,WhatSpecies,M2G,Matomnum,Stress,My_Stress,My_Stress_threads) private(OMPID,Nthrds,Nprocs,Mc_AN,Stime_atom,Etime_atom,Gc_AN,Cwan,sumx,sumy,sumz,Nc,GNc,GRc,MNc,Cxyz,x,y,z,dx,dy,dz,r,r2,tmp0,tmp1,tmp2,xx,sumt1,sumt2,sumt3,sumt4,sumt5)
    {
    
      /* get info. on OpenMP */ 

      OMPID = omp_get_thread_num();
      Nthrds = omp_get_num_threads();
      Nprocs = omp_get_num_procs();

      /* end(yshiihara)   */

      for (Mc_AN=(OMPID*Matomnum/Nthrds+1); Mc_AN<((OMPID+1)*Matomnum/Nthrds+1); Mc_AN++){

	dtime(&Stime_atom);

	Gc_AN = M2G[Mc_AN];
	Cwan = WhatSpecies[Gc_AN];

	sumx = 0.0;
	sumy = 0.0;
	sumz = 0.0;
	for (i=0; i<9; i++){
	  sumt1[i]    = 0.0;
	  sumt2[i]    = 0.0;
	  sumt3[i]    = 0.0;
	  sumt4[i]    = 0.0;
	  sumt5[i]    = 0.0;
	}

	for (Nc=0; Nc<GridN_Atom[Gc_AN]; Nc++){
	
	  GNc = GridListAtom[Mc_AN][Nc]; 
	  GRc = CellListAtom[Mc_AN][Nc];
	  MNc = MGridListAtom[Mc_AN][Nc]; 
	
	  Get_Grid_XYZ(GNc,Cxyz);
	  x = Cxyz[1] + atv[GRc][1]; 
	  y = Cxyz[2] + atv[GRc][2]; 
	  z = Cxyz[3] + atv[GRc][3];

	  dx = Gxyz[Gc_AN][1] - x;
	  dy = Gxyz[Gc_AN][2] - y;
	  dz = Gxyz[Gc_AN][3] - z;
	  r2 = dx*dx + dy*dy + dz*dz;
	  r = sqrt(r2);
	  xx = 0.5*log(r2);

	  /* for empty atoms */
	  if (r<1.0e-10) r = 1.0e-10;

	  if (1.0e-14<r){

	    tmp0 = Dr_KumoF( Spe_Num_Mesh_PAO[Cwan], xx, r, 
			     Spe_PAO_XV[Cwan], Spe_PAO_RV[Cwan], Spe_Atomic_Den[Cwan]);

	    tmp1 = dVHart_Grid[MNc]*tmp0/r*F_dVHart_flag;
	    /*begin(yshiihara) */ 
	    tmp2 = KumoF( Spe_Num_Mesh_PAO[Cwan], xx, 
			  Spe_PAO_XV[Cwan], Spe_PAO_RV[Cwan], Spe_Atomic_Den[Cwan]);
	    /*end(yshiihara) */ 
	    sumx += tmp1*dx;
	    sumy += tmp1*dy;
	    sumz += tmp1*dz;
	    /* begin(yshiihara) */ 
	    sumt1[0] += tmp1*dx*dx;
	    sumt1[1] += tmp1*dy*dx;
	    sumt1[2] += tmp1*dz*dx;
	    sumt1[3] += tmp1*dx*dy;
	    sumt1[4] += tmp1*dy*dy;
	    sumt1[5] += tmp1*dz*dy;
	    sumt1[6] += tmp1*dx*dz;
	    sumt1[7] += tmp1*dy*dz;
	    sumt1[8] += tmp1*dz*dz;
	    /* end(yshiihara) */

	    /* contribution of Exc^(0) */

	    /* begin(yshiihara) */

	    if (Exc0_correction_flag==1){

	      double exc0;
	      /*exc0 = XC_Ceperly_Alder(tmp2,0)*F_Vxc_flag;*/
	      /*tmp3 = RefVxc_Grid[MNc]*tmp2*F_Vxc_flag;*/
	      /*tmp3 = tmp2*exc0;*/

	      tmp1 = RefVxc_Grid[MNc]*tmp0/r*F_Vxc_flag;
	      sumx += tmp1*dx;
	      sumy += tmp1*dy;
	      sumz += tmp1*dz;
	      sumt2[0] += tmp1*dx*dx;
	      sumt2[1] += tmp1*dy*dx;
	      sumt2[2] += tmp1*dz*dx;
	      sumt2[3] += tmp1*dx*dy;
	      sumt2[4] += tmp1*dy*dy;
	      sumt2[5] += tmp1*dz*dy;
	      sumt2[6] += tmp1*dx*dz;
	      sumt2[7] += tmp1*dy*dz;
	      sumt2[8] += tmp1*dz*dz;
	    }
	    /* end(yshiihara) */

	    /* partial core correction */
	    if (PCC_switch==1){

	      tmp0 = 0.5*F_Vxc_flag*Dr_KumoF( Spe_Num_Mesh_VPS[Cwan], xx, r, 
					      Spe_VPS_XV[Cwan], Spe_VPS_RV[Cwan], Spe_Atomic_PCC[Cwan]);

	      if (SpinP_switch==0){
		tmp2 = 2.0*Vxc_Grid[0][MNc];
	      }
	      else {
		if (Spe_OpenCore_flag[Cwan]==0){
		  tmp2 = Vxc_Grid[0][MNc] + Vxc_Grid[1][MNc];
		}
		else if (Spe_OpenCore_flag[Cwan]==1){
		  tmp2 = 2.0*Vxc_Grid[0][MNc];
		}
		else if (Spe_OpenCore_flag[Cwan]==-1){
		  tmp2 = 2.0*Vxc_Grid[1][MNc];
		}
	      }

	      tmp1 = tmp2*tmp0/r;
	      sumx -= tmp1*dx;
	      sumy -= tmp1*dy;
	      sumz -= tmp1*dz;

	      /* begin(yshiihara) */
	      sumt3[0] -= tmp1*dx*dx;
	      sumt3[1] -= tmp1*dy*dx;
	      sumt3[2] -= tmp1*dz*dx;
	      sumt3[3] -= tmp1*dx*dy;
	      sumt3[4] -= tmp1*dy*dy;
	      sumt3[5] -= tmp1*dz*dy;
	      sumt3[6] -= tmp1*dx*dz;
	      sumt3[7] -= tmp1*dy*dz;
	      sumt3[8] -= tmp1*dz*dz;
	      /* end(yshiihara) */

	      /* contribution of Exc^(0) */
  
  	      if (Exc0_correction_flag==1){

		tmp2 = 2.0*RefVxc_Grid[MNc];
		tmp1 = tmp2*tmp0/r;
		sumx += tmp1*dx;
		sumy += tmp1*dy;
		sumz += tmp1*dz;

		/* begin(yshiihara) */
		sumt4[0] += tmp1*dx*dx;
		sumt4[1] += tmp1*dy*dx;
		sumt4[2] += tmp1*dz*dx;
		sumt4[3] += tmp1*dx*dy;
		sumt4[4] += tmp1*dy*dy;
		sumt4[5] += tmp1*dz*dy;
		sumt4[6] += tmp1*dx*dz;
		sumt4[7] += tmp1*dy*dz;
		sumt4[8] += tmp1*dz*dz;
		/* end(yshiihara) */
	      }

	    }
	  }
	} /* Nc */

	/* begin(yshiihara) */

	/*
	printf(" sumt1 : \n");fflush(stdout);
	for (i = 0; i<3; i++){
	  for (j = 0; j<3; j++){
	    printf("%16.8f ", sumt1[3*i+j]);
	  }
	  printf("\n");fflush(stdout);
	}
	printf(" sumt2 : \n");fflush(stdout);
	for (i = 0; i<3; i++){
	  for (j = 0; j<3; j++){
	    printf("%16.8f ", sumt2[3*i+j]);
	  }
	  printf("\n");fflush(stdout);
	}
	printf(" sumt3 : \n");fflush(stdout);
	for (i = 0; i<3; i++){
	  for (j = 0; j<3; j++){
	    printf("%16.8f ", sumt3[3*i+j]);
	  }
	  printf("\n");fflush(stdout);
	}
	printf(" sumt4 : \n");fflush(stdout);
	for (i = 0; i<3; i++){
	  for (j = 0; j<3; j++){
	    printf("%16.8f ", sumt4[3*i+j]);
	  }
	  printf("\n");fflush(stdout);
	}
	*/

	/* end(yshiihara) */

	/* begin(yshiihara) */
	My_Stress_threads[OMPID][0] -= (sumt1[0]+sumt2[0]+sumt3[0]+sumt4[0])*GridVol;
	My_Stress_threads[OMPID][1] -= (sumt1[1]+sumt2[1]+sumt3[1]+sumt4[1])*GridVol;
	My_Stress_threads[OMPID][2] -= (sumt1[2]+sumt2[2]+sumt3[2]+sumt4[2])*GridVol;
	My_Stress_threads[OMPID][3] -= (sumt1[3]+sumt2[3]+sumt3[3]+sumt4[3])*GridVol;
	My_Stress_threads[OMPID][4] -= (sumt1[4]+sumt2[4]+sumt3[4]+sumt4[4])*GridVol;
	My_Stress_threads[OMPID][5] -= (sumt1[5]+sumt2[5]+sumt3[5]+sumt4[5])*GridVol;
	My_Stress_threads[OMPID][6] -= (sumt1[6]+sumt2[6]+sumt3[6]+sumt4[6])*GridVol;
	My_Stress_threads[OMPID][7] -= (sumt1[7]+sumt2[7]+sumt3[7]+sumt4[7])*GridVol;
	My_Stress_threads[OMPID][8] -= (sumt1[8]+sumt2[8]+sumt3[8]+sumt4[8])*GridVol;
	/* end(yshiihara) */

	dtime(&Etime_atom);
	time_per_atom[Gc_AN] += Etime_atom - Stime_atom;
      
      } /* Mc_AN */

    } /* #pragma omp parallel */
  
    dtime(&etime);
    if(myid==0 && measure_time){
      printf("Time for stress#1=%18.5f\n",etime-stime);fflush(stdout);
    }
  
    /* begin(yshiihara) */
    for(i=0; i<Nthrds0; i++){
      for(j=0; j<9; j++){
	My_Stress[j] += My_Stress_threads[i][j];
      }
    }

    MPI_Allreduce(&My_Stress[0],&Stress[0],9,MPI_DOUBLE,MPI_SUM,mpi_comm_level1);

    /* add #1 of stress to Stress_Tensor */

    for (i=0; i<9; i++){
      Stress_Tensor[i] += Stress[i];
    }

    /* show #1 of stress */
  
    if (myid==Host_ID && 1<level_stdout){

      printf(" Components of stress #1: \n");fflush(stdout);
      for (i = 0; i<3; i++){
	for (j = 0; j<3; j++){
	  printf("%16.8f ", Stress[3*i+j]);
	}
	printf("\n");fflush(stdout);
      }
    }
    
    /* end(yshiihara) */
  
    if (ESM_switch!=0){
      if (myid==Host_ID && 0<level_stdout){
	printf("yshiihara: ESM is not currently supported in Stress.c \n");fflush(stdout);
      }
    }

    /* begin(yshiihara) */ 
    for (i=0; i<Nthrds0; i++){
      free(My_Stress_threads[i]);
    }
    free(My_Stress_threads);
    /* end(yshiihara) */

    /* begin(yshiihara) */ 
  }
  /* end(yshiihara) */

  /****************************************************
   contraction

   H0
   OLP
  ****************************************************/

  MPI_Barrier(mpi_comm_level1);
  
  if (Cnt_switch==1){

    Cont_Matrix0(H0[0],CntH0[0]);
    Cont_Matrix0(H0[1],CntH0[1]);
    Cont_Matrix0(H0[2],CntH0[2]);
    Cont_Matrix0(H0[3],CntH0[3]);

    Cont_Matrix0(OLP[0],CntOLP[0]);
    Cont_Matrix0(OLP[1],CntOLP[1]);
    Cont_Matrix0(OLP[2],CntOLP[2]);
    Cont_Matrix0(OLP[3],CntOLP[3]);
  }

  if ( Hub_U_switch==1 && Hub_U_occupation==1 ){
    if (myid==Host_ID && 0<level_stdout){
      printf(" LDA+U is not currently supported in Stress.c \n");fflush(stdout);
    }
    MPI_OLP(OLP);
  }

  MPI_Barrier(mpi_comm_level1);

  /* begin(yshiihara) */
  /* yshiihara: calculation on Lebedev grid performed in Total_energy.c */
  /********************************************************
    calculation of Exc^(0) and its contribution 
    to forces on the fine mesh
 
    Note that there is no stress coming from volume term.
  ********************************************************/

  if (Exc0_correction_flag==1){

    double **Leb_Grid_XYZW;

    Leb_Grid_XYZW = (double**)malloc(sizeof(double*)*Num_Leb_Grid); 
    for (i=0; i<Num_Leb_Grid; i++){
      Leb_Grid_XYZW[i] = (double*)malloc(sizeof(double)*4);
    }

    Set_Lebedev_Grid(Num_Leb_Grid,Leb_Grid_XYZW);

    /* get Nthrds0 */  
#pragma omp parallel shared(Nthrds0)
    {
      Nthrds0 = omp_get_num_threads();
    }

    /* initialize the temporal array storing the force contribution */

    for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
      Gxyz[Gc_AN][41] = 0.0;
      Gxyz[Gc_AN][42] = 0.0;
      Gxyz[Gc_AN][43] = 0.0;
    }

    /* begin(yshiihara) */
    {

      double exc0,vxc0,rs,sum,sumr,sumt,re,Sr,Dr,sumtx,sumty,sumtz,den,den0,gden0,w;
      double x0,y0,z0,x1,y1,z1,dx1,dy1,dz1,r1;
      double *My_sumr;
      int Nloop,Rn,ir,ia;

      /* start calc. */

      for (i=0; i<9; i++){
	My_Stress[i] = 0.0;
	Stress[i]    = 0.0;
      }

      rs = 0.0;
      sum = 0.0;

      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

	Gc_AN = M2G[Mc_AN];
	Cwan = WhatSpecies[Gc_AN];
	re = Spe_Atom_Cut1[Cwan];
	Sr = re + rs;
	Dr = re - rs;

	/* allocation of arrays */

	double *My_sumr,**My_sumrx,**My_sumry,**My_sumrz;

	My_sumr = (double*)malloc(sizeof(double)*Nthrds0);
	for (i=0; i<Nthrds0; i++) My_sumr[i]  = 0.0;

	My_sumrx = (double**)malloc(sizeof(double*)*Nthrds0);
	for (i=0; i<Nthrds0; i++){
	  My_sumrx[i] = (double*)malloc(sizeof(double)*(FNAN[Gc_AN]+1));
	  for (j=0; j<(FNAN[Gc_AN]+1); j++){
	    My_sumrx[i][j] = 0.0;
	  }
	}

	My_sumry = (double**)malloc(sizeof(double*)*Nthrds0);
	for (i=0; i<Nthrds0; i++){
	  My_sumry[i] = (double*)malloc(sizeof(double)*(FNAN[Gc_AN]+1));
	  for (j=0; j<(FNAN[Gc_AN]+1); j++){
	    My_sumry[i][j] = 0.0;
	  }
	}

	My_sumrz = (double**)malloc(sizeof(double*)*Nthrds0);
	for (i=0; i<Nthrds0; i++){
	  My_sumrz[i] = (double*)malloc(sizeof(double)*(FNAN[Gc_AN]+1));
	  for (j=0; j<(FNAN[Gc_AN]+1); j++){
	    My_sumrz[i][j] = 0.0;
	  }
	}

	My_Stress_threads=(double**)malloc(sizeof(double*)*(Nthrds0));
	for (Nloop=0; Nloop<Nthrds0; Nloop++){
	  My_Stress_threads[Nloop] = (double*)malloc(sizeof(double)*9);
	  for (i=0; i<9; i++){
	    My_Stress_threads[Nloop][i] = 0.0;
	  }
	}
	
#pragma omp parallel shared(Leb_Grid_XYZW,Num_Leb_Grid,Spe_Atomic_Den2,Spe_PAO_XV,Spe_Num_Mesh_PAO,My_sumr,My_sumrx,My_sumry,My_sumrz,My_Stress,My_Stress_threads,Dr,Sr,CoarseGL_Abscissae,CoarseGL_Weight,Gxyz,Gc_AN,FNAN,natn,ncn,WhatSpecies,atv,F_Vxc_flag,Cwan,PCC_switch) private(OMPID,Nthrds,Nprocs,ir,ia,r,w,sumt,sumtx,sumty,sumtz,x,x0,y0,z0,h_AN,Gh_AN,Rn,Hwan,x1,y1,z1,dx,dy,dz,r1,den,den0,gden0,dx1,dy1,dz1,exc0,vxc0,sumt1)
	{

	  double dx2,dy2,dz2;
	  double *gx,*gy,*gz,dexc0;
	  double *sum_gx,*sum_gy,*sum_gz;

	  gx = (double*)malloc(sizeof(double)*(FNAN[Gc_AN]+1));
	  gy = (double*)malloc(sizeof(double)*(FNAN[Gc_AN]+1));
	  gz = (double*)malloc(sizeof(double)*(FNAN[Gc_AN]+1));
	  sum_gx = (double*)malloc(sizeof(double)*(FNAN[Gc_AN]+1));
	  sum_gy = (double*)malloc(sizeof(double)*(FNAN[Gc_AN]+1));
	  sum_gz = (double*)malloc(sizeof(double)*(FNAN[Gc_AN]+1));

	  /* get info. on OpenMP */ 

	  OMPID = omp_get_thread_num();
	  Nthrds = omp_get_num_threads();
	  Nprocs = omp_get_num_procs();

	  for (ir=(OMPID*CoarseGL_Mesh/Nthrds); ir<((OMPID+1)*CoarseGL_Mesh/Nthrds); ir++){

	    r = 0.50*(Dr*CoarseGL_Abscissae[ir] + Sr);
	    sumt  = 0.0; 

	    for (i=0; i<(FNAN[Gc_AN]+1); i++){
	      sum_gx[i] = 0.0;
	      sum_gy[i] = 0.0;
	      sum_gz[i] = 0.0;
	    }

	    for (i=0; i<9; i++) sumt1[i] = 0.0;

	    for (ia=0; ia<Num_Leb_Grid; ia++){

	      x0 = r*Leb_Grid_XYZW[ia][0] + Gxyz[Gc_AN][1];
	      y0 = r*Leb_Grid_XYZW[ia][1] + Gxyz[Gc_AN][2];
	      z0 = r*Leb_Grid_XYZW[ia][2] + Gxyz[Gc_AN][3];

	      /* calculate rho_atom + rho_pcc */ 

	      den = 0.0;

	      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

		Gh_AN = natn[Gc_AN][h_AN];
		Rn = ncn[Gc_AN][h_AN]; 
		Hwan = WhatSpecies[Gh_AN];

		x1 = Gxyz[Gh_AN][1] + atv[Rn][1];
		y1 = Gxyz[Gh_AN][2] + atv[Rn][2];
		z1 = Gxyz[Gh_AN][3] + atv[Rn][3];
            
		dx = x1 - x0;
		dy = y1 - y0;
		dz = z1 - z0;

		x = 0.5*log(dx*dx + dy*dy + dz*dz);

		/* calculate density */

		den += KumoF( Spe_Num_Mesh_PAO[Hwan], x, 
			      Spe_PAO_XV[Hwan], Spe_PAO_RV[Hwan], Spe_Atomic_Den2[Hwan])*F_Vxc_flag;

		if (h_AN==0) den0 = den;

		/* calculate gradient of density */

		if (h_AN!=0){
		  r1 = sqrt(dx*dx + dy*dy + dz*dz);
		  gden0 = Dr_KumoF( Spe_Num_Mesh_PAO[Hwan], x, r1, 
				    Spe_PAO_XV[Hwan], Spe_PAO_RV[Hwan], Spe_Atomic_Den2[Hwan])*F_Vxc_flag;

		  gx[h_AN] = gden0/r1*dx;
		  gy[h_AN] = gden0/r1*dy;
		  gz[h_AN] = gden0/r1*dz;
		}

	      } /* h_AN */

	      /* calculate the CA-LDA exchange-correlation energy density */
	      exc0 = XC_Ceperly_Alder(den,0);

	      /* calculate the CA-LDA exchange-correlation potential */
	      dexc0 = XC_Ceperly_Alder(den,3);

	      /* Lebedev quadrature */

	      w = Leb_Grid_XYZW[ia][3];
	      sumt += w*den0*exc0;

	      for (h_AN=1; h_AN<=FNAN[Gc_AN]; h_AN++){
              
		dx1 = w*den0*dexc0*gx[h_AN]; 
		dy1 = w*den0*dexc0*gy[h_AN]; 
		dz1 = w*den0*dexc0*gz[h_AN]; 

		sum_gx[h_AN] += dx1;
		sum_gy[h_AN] += dy1;
		sum_gz[h_AN] += dz1;

		Gh_AN = natn[Gc_AN][h_AN];
		Rn = ncn[Gc_AN][h_AN]; 
              
		dx2 = Gxyz[Gh_AN][1] + atv[Rn][1] - Gxyz[Gc_AN][1];
		dy2 = Gxyz[Gh_AN][2] + atv[Rn][2] - Gxyz[Gc_AN][2];
		dz2 = Gxyz[Gh_AN][3] + atv[Rn][3] - Gxyz[Gc_AN][3];

		sumt1[0] += dx1*dx2;
		sumt1[1] += dx1*dy2;
		sumt1[2] += dx1*dz2;
		sumt1[3] += dy1*dx2;
		sumt1[4] += dy1*dy2;
		sumt1[5] += dy1*dz2;
		sumt1[6] += dz1*dx2;
		sumt1[7] += dz1*dy2;
		sumt1[8] += dz1*dz2;

	      } /* h_AN */

	    } /* ia */

	    /* r for Gauss-Legendre quadrature */

	    w = r*r*CoarseGL_Weight[ir]; 
	    My_sumr[OMPID]  += w*sumt;

	    for (h_AN=1; h_AN<=FNAN[Gc_AN]; h_AN++){
	      My_sumrx[OMPID][h_AN] += w*sum_gx[h_AN];
	      My_sumry[OMPID][h_AN] += w*sum_gy[h_AN];
	      My_sumrz[OMPID][h_AN] += w*sum_gz[h_AN];
	    }

	    for(i=0; i<9; i++){
	      My_Stress_threads[OMPID][i] += w*sumt1[i];
	    }

	  } /* ir */

	  free(gx);
	  free(gy);
	  free(gz);
	  free(sum_gx);
	  free(sum_gy);
	  free(sum_gz);

	} /* #pragma omp */

	sumr = 0.0;
	for (Nloop=0; Nloop<Nthrds0; Nloop++){
	  sumr += My_sumr[Nloop];
	}
	sum += 2.0*PI*Dr*sumr;

	/* add force */

	for (h_AN=1; h_AN<=FNAN[Gc_AN]; h_AN++){

	  double sumrx,sumry,sumrz;

	  sumrx = 0.0;
	  sumry = 0.0;
	  sumrz = 0.0;
	  for (Nloop=0; Nloop<Nthrds0; Nloop++){
	    sumrx += My_sumrx[Nloop][h_AN];
	    sumry += My_sumry[Nloop][h_AN];
	    sumrz += My_sumrz[Nloop][h_AN];
	  }

	  Gh_AN = natn[Gc_AN][h_AN];

	  Gxyz[Gh_AN][41] += 2.0*PI*Dr*sumrx;
	  Gxyz[Gh_AN][42] += 2.0*PI*Dr*sumry;
	  Gxyz[Gh_AN][43] += 2.0*PI*Dr*sumrz;

	  Gxyz[Gc_AN][41] -= 2.0*PI*Dr*sumrx;
	  Gxyz[Gc_AN][42] -= 2.0*PI*Dr*sumry;
	  Gxyz[Gc_AN][43] -= 2.0*PI*Dr*sumrz;
	}

	/* add stress */
      
	for (Nloop=0; Nloop<Nthrds0; Nloop++){
	  for(i=0; i<9; i++){
	    My_Stress[i] += 2.0*PI*Dr*My_Stress_threads[Nloop][i];
	  }
	}

	/* freeing of arrays */

	free(My_sumr);

	for (i=0; i<Nthrds0; i++){
	  free(My_sumrx[i]);
	}
	free(My_sumrx);

	for (i=0; i<Nthrds0; i++){
	  free(My_sumry[i]);
	}
	free(My_sumry);

	for (i=0; i<Nthrds0; i++){
	  free(My_sumrz[i]);
	}
	free(My_sumrz);

	for (i=0; i<Nthrds0; i++){
	  free(My_Stress_threads[i]);
	}
	free(My_Stress_threads);
	
      } /* Mc_AN */

      /****************************************************
                       MPI, My_Stress
      ****************************************************/

      MPI_Allreduce(&My_Stress[0],&Stress[0],9,MPI_DOUBLE,MPI_SUM,mpi_comm_level1);

      /* add components of stress (Lebedev) to Stress_Tensor */

      for (i=0; i<9; i++){
	Stress_Tensor[i] += Stress[i];
      }

      /* show stress (Lebedev) */

      if (myid==Host_ID && 1<level_stdout){
	printf(" Components of stress (Lebedev): \n");fflush(stdout);
	for (i = 0; i<3; i++){
	  for (j = 0; j<3; j++){
	    printf("%16.8f ", Stress[3*i+j]);
	  }
	  printf("\n");fflush(stdout);
	}
      }

    }
    /* end(yshiihara) */

    for (i=0; i<Num_Leb_Grid; i++){
      free(Leb_Grid_XYZW[i]);
    }
    free(Leb_Grid_XYZW);

  } /* end of if (Exc0_correction_flag==1) */

  /****************************************************
                      #2 of Stress

                    kinetic operator 
  ****************************************************/

  dtime(&stime);

  if (myid==Host_ID && 0<level_stdout){
    printf("  Stress calculation #2\n");fflush(stdout);
  }
  
#pragma omp parallel shared(Nthrds0)
  {
    Nthrds0 = omp_get_num_threads();
  }

  /* begin(yshiihara) */ 
  My_Stress_threads=(double**)malloc(sizeof(double*)*(Nthrds0));
  for (i=0; i<Nthrds0; i++){
    My_Stress_threads[i] = (double*)malloc(sizeof(double)*9);
    for (j=0; j<9; j++) My_Stress_threads[i][j] = 0.0;
  }
  /* end(yshiihara) */
  
#pragma omp parallel shared(time_per_atom,Gxyz,myid,level_stdout,iDM0,CDM0,CntH0,H0,F_Kin_flag,NC_v_eff,v_eff,OLP,Hub_U_occupation,Cnt_switch,F_NL_flag,List_YOUSO,RMI1,Zeeman_NCO_switch,Zeeman_NCS_switch,Constraint_NCS_switch,F_U_flag,Hub_U_switch,SO_switch,SpinP_switch,Spe_Total_CNO,F_G2M,natn,ncn,FNAN,WhatSpecies,M2G,Matomnum,My_Stress_threads) private(OMPID,Nthrds,Nprocs,Mc_AN,Stime_atom,Etime_atom,Gc_AN,Cwan,dEx,dEy,dEz,h_AN,Gh_AN,Mh_AN,Hwan,ian,start_q_AN,q_AN,Gq_AN,Mq_AN,Qwan,jan,kl,so,i,j,k,l,Hx,Hy,Hz,HUx,HUy,HUz,NC_HUx,NC_HUy,NC_HUz,s1,s2,pref,spinmax,spin,lx1,ly1,lz1,lx2,ly2,lz2,lx,ly,lz,Rn1,Rn2)
  {

    Hx = (dcomplex***)malloc(sizeof(dcomplex**)*3);
    for (i=0; i<3; i++){
      Hx[i] = (dcomplex**)malloc(sizeof(dcomplex*)*List_YOUSO[7]);
      for (j=0; j<List_YOUSO[7]; j++){
	Hx[i][j] = (dcomplex*)malloc(sizeof(dcomplex)*List_YOUSO[7]);
      }
    }

    Hy = (dcomplex***)malloc(sizeof(dcomplex**)*3);
    for (i=0; i<3; i++){
      Hy[i] = (dcomplex**)malloc(sizeof(dcomplex*)*List_YOUSO[7]);
      for (j=0; j<List_YOUSO[7]; j++){
	Hy[i][j] = (dcomplex*)malloc(sizeof(dcomplex)*List_YOUSO[7]);
      }
    }

    Hz = (dcomplex***)malloc(sizeof(dcomplex**)*3);
    for (i=0; i<3; i++){
      Hz[i] = (dcomplex**)malloc(sizeof(dcomplex*)*List_YOUSO[7]);
      for (j=0; j<List_YOUSO[7]; j++){
	Hz[i][j] = (dcomplex*)malloc(sizeof(dcomplex)*List_YOUSO[7]);
      }
    }
    
    /* get info. on OpenMP */ 

    OMPID = omp_get_thread_num();
    Nthrds = omp_get_num_threads();
    Nprocs = omp_get_num_procs();

    /*debug*/
    /*printf("OpenMP myid: %d, OPMID: %d, Nthrds: %d, Nprocs: %d \n",myid,OMPID,Nthrds,Nprocs);fflush(stdout);*/ 
    
    for (Mc_AN=(OMPID*Matomnum/Nthrds+1); Mc_AN<((OMPID+1)*Matomnum/Nthrds+1); Mc_AN++){
      
      dtime(&Stime_atom);
      
      Gc_AN = M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];
  
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	Gh_AN = natn[Gc_AN][h_AN];
	Mh_AN = F_G2M[Gh_AN];
	Hwan = WhatSpecies[Gh_AN];
	ian = Spe_Total_CNO[Hwan];
	/*stress*/
	Rn1 = ncn[Gc_AN][h_AN];
	lx1 = Gxyz[Gc_AN][1] - Gxyz[Gh_AN][1] - atv[Rn1][1];
	ly1 = Gxyz[Gc_AN][2] - Gxyz[Gh_AN][2] - atv[Rn1][2];
	lz1 = Gxyz[Gc_AN][3] - Gxyz[Gh_AN][3] - atv[Rn1][3];

	/*debug*/
	/*printf("Gc_AN: %d, Gh_AN: %d, lx1: %13.8f, ly1: %13.8f, lz1: %13.8f \n",Gc_AN,Gh_AN,lx1,ly1,lz1); */
	
	if ( SpinP_switch==3 && (SO_switch==1 || (Hub_U_switch==1 && F_U_flag==1)
	 || Constraint_NCS_switch==1 || Zeeman_NCS_switch==1 || Zeeman_NCO_switch==1) )
	  start_q_AN = 0;
	else 
	  start_q_AN = h_AN;

	for (q_AN=start_q_AN; q_AN<=FNAN[Gc_AN]; q_AN++){
	    
	  Gq_AN = natn[Gc_AN][q_AN];
	  Mq_AN = F_G2M[Gq_AN];
	  Qwan = WhatSpecies[Gq_AN];
	  jan = Spe_Total_CNO[Qwan];
	  kl = RMI1[Mc_AN][h_AN][q_AN];
	  /*stress*/
	  Rn2 = ncn[Gc_AN][q_AN];
	  lx2 = Gxyz[Gc_AN][1] - Gxyz[Gq_AN][1] - atv[Rn2][1];
	  ly2 = Gxyz[Gc_AN][2] - Gxyz[Gq_AN][2] - atv[Rn2][2];
	  lz2 = Gxyz[Gc_AN][3] - Gxyz[Gq_AN][3] - atv[Rn2][3];

	  /*debug*/
	  /*printf("Gc_AN: %d, Gq_AN: %d, lx2: %13.8f, ly2: %13.8f, lz2: %13.8f \n",Gc_AN,Gq_AN,lx2,ly2,lz2); */
	  	  
	  if (0<=kl){
	      
	    for (so=0; so<3; so++){
	      for (i=0; i<List_YOUSO[7]; i++){
		for (j=0; j<List_YOUSO[7]; j++){
		  Hx[so][i][j] = Complex(0.0,0.0);
		  Hy[so][i][j] = Complex(0.0,0.0);
		  Hz[so][i][j] = Complex(0.0,0.0);
		}
	      }
	    }
	      
	    /****************************************************
                               H0 = dKinetic
	    ****************************************************/

	    lx = 0.0;
	    ly = 0.0;
	    lz = 0.0;

	    if (F_Kin_flag==1){  

	      /* in case of no orbital optimization */
	      
	      if (Cnt_switch==0){
		if (h_AN==0){
		  for (i=0; i<ian; i++){
		    for (j=0; j<jan; j++){
		      Hx[0][i][j].r += H0[1][Mc_AN][q_AN][i][j];
		      Hy[0][i][j].r += H0[2][Mc_AN][q_AN][i][j];
		      Hz[0][i][j].r += H0[3][Mc_AN][q_AN][i][j];
		      
		      Hx[1][i][j].r += H0[1][Mc_AN][q_AN][i][j];
		      Hy[1][i][j].r += H0[2][Mc_AN][q_AN][i][j];
		      Hz[1][i][j].r += H0[3][Mc_AN][q_AN][i][j];
		    }
		  }
		  lx = lx2;
		  ly = ly2;
		  lz = lz2;
		}

		else if (h_AN!=0 && q_AN==0){
		  for (i=0; i<ian; i++){
		    for (j=0; j<jan; j++){
		      Hx[0][i][j].r += H0[1][Mc_AN][h_AN][j][i];
		      Hy[0][i][j].r += H0[2][Mc_AN][h_AN][j][i];
		      Hz[0][i][j].r += H0[3][Mc_AN][h_AN][j][i];

		      Hx[1][i][j].r += H0[1][Mc_AN][h_AN][j][i];
		      Hy[1][i][j].r += H0[2][Mc_AN][h_AN][j][i];
		      Hz[1][i][j].r += H0[3][Mc_AN][h_AN][j][i];
		    }
		  }
		  lx = lx1;
		  ly = ly1;
		  lz = lz1;
		}
	      }
	      
	      /* in case of orbital optimization */
	      
	      else{

		if (h_AN==0){
		  for (i=0; i<ian; i++){
		    for (j=0; j<jan; j++){

		      Hx[0][i][j].r += CntH0[1][Mc_AN][q_AN][i][j];
		      Hy[0][i][j].r += CntH0[2][Mc_AN][q_AN][i][j];
		      Hz[0][i][j].r += CntH0[3][Mc_AN][q_AN][i][j];

		      Hx[1][i][j].r += CntH0[1][Mc_AN][q_AN][i][j];
		      Hy[1][i][j].r += CntH0[2][Mc_AN][q_AN][i][j];
		      Hz[1][i][j].r += CntH0[3][Mc_AN][q_AN][i][j];

		    }
		  }
		  lx = lx2;
		  ly = ly2;
		  lz = lz2;
		} 

		else if (h_AN!=0 && q_AN==0){
		  for (i=0; i<ian; i++){
		    for (j=0; j<jan; j++){

		      Hx[0][i][j].r += CntH0[1][Mc_AN][h_AN][j][i];
		      Hy[0][i][j].r += CntH0[2][Mc_AN][h_AN][j][i];
		      Hz[0][i][j].r += CntH0[3][Mc_AN][h_AN][j][i];

		      Hx[1][i][j].r += CntH0[1][Mc_AN][h_AN][j][i];
		      Hy[1][i][j].r += CntH0[2][Mc_AN][h_AN][j][i];
		      Hz[1][i][j].r += CntH0[3][Mc_AN][h_AN][j][i];

		    }
		  }
		  lx = lx1;
		  ly = ly1;
		  lz = lz1;
		}
	      }
 
	    } /* if F_Kin_flag */

	    /****************************************************
                                \sum rho*dH
	    ****************************************************/

            /* non-spin polarization */

	    if (SpinP_switch==0){

	      if (q_AN==h_AN) pref = 2.0;
	      else            pref = 4.0; 

	      /* Stress */
	      dEx = 0.0;
	      dEy = 0.0;
	      dEz = 0.0;
	      
	      for (i=0; i<Spe_Total_CNO[Hwan]; i++){
		for (j=0; j<Spe_Total_CNO[Qwan]; j++){
		  dEx += pref*CDM0[0][Mh_AN][kl][i][j]*Hx[0][i][j].r;
		  dEy += pref*CDM0[0][Mh_AN][kl][i][j]*Hy[0][i][j].r;
		  dEz += pref*CDM0[0][Mh_AN][kl][i][j]*Hz[0][i][j].r;
		}
	      }

              /* debug */   

	      /*
	      if(fabs(dEx)>1.e-2 || fabs(dEy)>1.e-2 || fabs(dEz)>1.e-2) { 
		printf("Gh_AN: %d, Gc_AN: %d, Gq_AN: %d \n",Gh_AN,Gc_AN,Gq_AN); 
		printf("dEx: %13.8f, dEy: %13.8f, dEz: %13.8f, lx: %13.8f, ly: %13.8f, lz: %13.8f \n",
                        dEx,dEy,dEz,lx,ly,lz);fflush(stdout); 
		printf("lx1: %13.8f, ly1: %13.8f, lz1: %13.8f, lx2: %13.8f, ly2: %13.8f, lz2: %13.8f \n",
                        lx1,ly1,lz1,lx2,ly2,lz2);fflush(stdout); 
		printf("Gc_AN x: %13.8f, y: %13.8f, z: %13.8f  \n",Gxyz[Gc_AN][1],Gxyz[Gc_AN][2],Gxyz[Gc_AN][3]);
		printf("Gh_AN x: %13.8f, y: %13.8f, z: %13.8f, atv 1: %13.8f, 2: %13.8f, 3: %13.8f  \n",
                        Gxyz[Gh_AN][1],Gxyz[Gh_AN][2],Gxyz[Gh_AN][3],atv[Rn1][1],atv[Rn1][2],atv[Rn1][3]); 
		printf("Gq_AN x: %13.8f, y: %13.8f, z: %13.8f, atv 1: %13.8f, 2: %13.8f, 3: %13.8f  \n",
                        Gxyz[Gq_AN][1],Gxyz[Gq_AN][2],Gxyz[Gq_AN][3],atv[Rn2][1],atv[Rn2][2],atv[Rn2][3]); 
	      }
	      */

	      /*
	      if(fabs(dEx)>1.e-2 || fabs(dEy)>1.e-2 || fabs(dEz)>1.e-2) { 
		printf("ABC1 Gh_AN: %d, Gc_AN: %d, Gq_AN: %d \n",Gh_AN,Gc_AN,Gq_AN); 
		printf("dEx: %13.8f, dEy: %13.8f, dEz: %13.8f, lx: %13.8f, ly: %13.8f, lz: %13.8f \n",
                        dEx,dEy,dEz,lx,ly,lz);fflush(stdout); 

		printf("lx*dEz: %17.10f, lz*dEx: %17.10f\n",lx*dEz,lz*dEx);fflush(stdout); 
		printf("exz: %17.10f, ezx: %17.10f\n",
                        My_Stress_threads[0][2],My_Stress_threads[0][6]);fflush(stdout); 


		printf("lx1: %13.8f, ly1: %13.8f, lz1: %13.8f, lx2: %13.8f, ly2: %13.8f, lz2: %13.8f \n",
                        lx1,ly1,lz1,lx2,ly2,lz2);fflush(stdout); 
		printf("Gc_AN x: %13.8f, y: %13.8f, z: %13.8f  \n",Gxyz[Gc_AN][1],Gxyz[Gc_AN][2],Gxyz[Gc_AN][3]);
		printf("Gh_AN x: %13.8f, y: %13.8f, z: %13.8f, atv 1: %13.8f, 2: %13.8f, 3: %13.8f  \n",
                        Gxyz[Gh_AN][1],Gxyz[Gh_AN][2],Gxyz[Gh_AN][3],atv[Rn1][1],atv[Rn1][2],atv[Rn1][3]); 
		printf("Gq_AN x: %13.8f, y: %13.8f, z: %13.8f, atv 1: %13.8f, 2: %13.8f, 3: %13.8f  \n",
                        Gxyz[Gq_AN][1],Gxyz[Gq_AN][2],Gxyz[Gq_AN][3],atv[Rn2][1],atv[Rn2][2],atv[Rn2][3]); 
	      }
	      */

	      My_Stress_threads[OMPID][0] += 0.5*dEx*lx;
	      My_Stress_threads[OMPID][1] += 0.5*dEy*lx;
	      My_Stress_threads[OMPID][2] += 0.5*dEz*lx;
	      My_Stress_threads[OMPID][3] += 0.5*dEx*ly;
	      My_Stress_threads[OMPID][4] += 0.5*dEy*ly;
	      My_Stress_threads[OMPID][5] += 0.5*dEz*ly;
	      My_Stress_threads[OMPID][6] += 0.5*dEx*lz;
	      My_Stress_threads[OMPID][7] += 0.5*dEy*lz;
	      My_Stress_threads[OMPID][8] += 0.5*dEz*lz;

	    }

	    /* collinear spin polarized or non-colliear without SO and LDA+U */

	    else if (SpinP_switch==1 || (SpinP_switch==3 && SO_switch==0 && Hub_U_switch==0
		 && Constraint_NCS_switch==0 && Zeeman_NCS_switch==0 && Zeeman_NCO_switch==0)){ 

	      if (q_AN==h_AN) pref = 1.0;
	      else            pref = 2.0; 

	      /* Stress */

	      dEx = 0.0;
	      dEy = 0.0;
	      dEz = 0.0;
	      
	      for (i=0; i<Spe_Total_CNO[Hwan]; i++){
		for (j=0; j<Spe_Total_CNO[Qwan]; j++){

		  dEx += pref*(  CDM0[0][Mh_AN][kl][i][j]*Hx[0][i][j].r
	                       + CDM0[1][Mh_AN][kl][i][j]*Hx[1][i][j].r);
		  dEy += pref*(  CDM0[0][Mh_AN][kl][i][j]*Hy[0][i][j].r
	                       + CDM0[1][Mh_AN][kl][i][j]*Hy[1][i][j].r);
		  dEz += pref*(  CDM0[0][Mh_AN][kl][i][j]*Hz[0][i][j].r
		               + CDM0[1][Mh_AN][kl][i][j]*Hz[1][i][j].r);
		}
	      }

	      /*Stress*/

              /* debug */
	      /*
	      if(fabs(dEx)>1.e-5 || fabs(dEy)>1.e-5 || fabs(dEz)>1.e-5) { 
		printf("Gh_AN: %d, Gc_AN: %d, Gq_AN: %d \n",Gh_AN,Gc_AN,Gq_AN); 
		printf("dEx: %13.8f, dEy: %13.8f, dEz: %13.8f, lx: %13.8f, ly: %13.8f, lz: %13.8f \n",
                        dEx,dEy,dEz,lx,ly,lz);fflush(stdout); 
	      }
	      */

	      My_Stress_threads[OMPID][0] += 0.5*dEx*lx;
	      My_Stress_threads[OMPID][1] += 0.5*dEy*lx;
	      My_Stress_threads[OMPID][2] += 0.5*dEz*lx;
	      My_Stress_threads[OMPID][3] += 0.5*dEx*ly;
	      My_Stress_threads[OMPID][4] += 0.5*dEy*ly;
	      My_Stress_threads[OMPID][5] += 0.5*dEz*ly;
	      My_Stress_threads[OMPID][6] += 0.5*dEx*lz;
	      My_Stress_threads[OMPID][7] += 0.5*dEy*lz;
	      My_Stress_threads[OMPID][8] += 0.5*dEz*lz;
	    }
	    
	    /* spin collinear with spin-orbit coupling */
	    
	    else if ( SpinP_switch==1 && SO_switch==1 ){
	      printf("Spin-orbit coupling is not supported for collinear DFT calculations.\n");fflush(stdout);
	      MPI_Finalize();
	      exit(1);
	    }
	    
	    /* spin non-collinear with spin-orbit coupling or with LDA+U */
	    
	    else if ( SpinP_switch==3 && (SO_switch==1 || (Hub_U_switch==1 && F_U_flag==1) 
		  || Constraint_NCS_switch==1 || Zeeman_NCS_switch==1 || Zeeman_NCO_switch==1) ){
	      
	      /*Stress*/
	      printf("spin NC with SO coupling or with LDA+U: not supported for stress.\n");fflush(stdout);
	      MPI_Finalize();
	      exit(1);
	      
	    }
	    
	  } /* if(0<=kl) */
	} /* for q_AN */
      } /* for h_AN */

      dtime(&Etime_atom);
      time_per_atom[Gc_AN] += Etime_atom - Stime_atom;
      
    } /* for MC_AN */

    /* freeing of arrays */
    
    for (i=0; i<3; i++){
      for (j=0; j<List_YOUSO[7]; j++){
	free(Hx[i][j]);
      }
      free(Hx[i]);
    }
    free(Hx);
    
    for (i=0; i<3; i++){
      for (j=0; j<List_YOUSO[7]; j++){
	free(Hy[i][j]);
      }
      free(Hy[i]);
    }
    free(Hy);
    
    for (i=0; i<3; i++){
      for (j=0; j<List_YOUSO[7]; j++){
	free(Hz[i][j]);
      }
      free(Hz[i]);
    }
    free(Hz);

  } /* pragma */

  /* Stress */
  /* if (myid==Host_ID && 0<level_stdout) printf("debug1, Nthrds0: %d \n",Nthrds0);fflush(stdout); */ /*debug*/

  for(i=0; i<9; i++){
    My_Stress[i]=0.0;
  }

  for(i=0; i<Nthrds0; i++){
    for(j=0; j<9; j++){
      /* printf("myid:%d, i:%d, j:%d \n",myid,i,j);fflush(stdout); */ /*debug*/
      My_Stress[j] += My_Stress_threads[i][j];
    }
  }

  /* if (myid==Host_ID && 0<level_stdout) printf("debug2 \n");fflush(stdout); */ /*debug*/

  MPI_Allreduce(&My_Stress[0],&Stress[0],9,MPI_DOUBLE,MPI_SUM,mpi_comm_level1);

  /* add kinetic components of stress to Stress_Tensor */

  for (i=0; i<9; i++){
    Stress_Tensor[i] += Stress[i];
  }

  /* show Kinetic components of stress */
  
  if (myid==Host_ID && 1<level_stdout){
    printf(" Kinetic components of stress : \n");fflush(stdout);
    for (i = 0; i<3; i++){
      for (j = 0; j<3; j++){
	printf("%16.8f ", Stress[3*i+j]);
      }
      printf("\n");fflush(stdout);
    }
  }

  dtime(&etime);
  if(myid==0 && measure_time){
    printf("Time for stress#2=%18.5f\n",etime-stime);fflush(stdout);
  }

  /* begin(yshiihara) */ 
  for (i=0; i<Nthrds0; i++){
    free(My_Stress_threads[i]);
  }
  free(My_Stress_threads);
  /* end(yshiihara) */

 /****************************************************
                      #3 of Stress

               dn/depsilon * (VNA + dVH + Vxc)
            or 
               dn/depsilon * (dVH + Vxc)
  ****************************************************/

  dtime(&stime);
  
  if (myid==Host_ID && 0<level_stdout){
    printf("  Stress calculation #3\n");fflush(stdout);
  }

  Stress3();

  dtime(&etime);
  if(myid==0 && measure_time){
    printf("Time for Stress#3=%18.5f\n",etime-stime);fflush(stdout);
  } 

  /****************************************************
                    #4 of Stress

       Stress4:   n * dVNA/epsilon
       Stress4B:  from separable VNA projectors
  ****************************************************/

  dtime(&stime);

  if (myid==Host_ID && 0<level_stdout){
    printf("  Stress calculation #4\n");fflush(stdout);
  }

  if (ProExpn_VNA==0 && F_VNA_flag==1){
    Stress4();
  }
  else if (ProExpn_VNA==1 && F_VNA_flag==1){
    Stress4B(CDM0);
  }

  dtime(&etime);
  if(myid==0 && measure_time){
    printf("Time for Stress#4=%18.5f\n",etime-stime);fflush(stdout);
  } 

  /****************************************************
                      #5 of Stress

               Contribution from overlap
  ****************************************************/

  dtime(&stime);

  if (myid==Host_ID && 0<level_stdout){
    printf("  Stress calculation #5\n");fflush(stdout);
  }

  /* begin(yshiihara) */

#pragma omp parallel shared(Nthrds0)
  {
    Nthrds0 = omp_get_num_threads();
  }

  My_Stress_threads=(double**)malloc(sizeof(double*)*(Nthrds0));
  for (i=0; i<Nthrds0; i++){
    My_Stress_threads[i] = (double*)malloc(sizeof(double)*9);
    for (j=0; j<9; j++) My_Stress_threads[i][j] = 0.0;
  }
  /* end(yshiihara) */

#pragma omp parallel shared(time_per_atom,Fx,Fy,Fz,CntOLP,OLP,Cnt_switch,EDM,SpinP_switch,Spe_Total_CNO,natn,ncn,FNAN,WhatSpecies,M2G,Gxyz,atv,Matomnum,My_Stress_threads) private(OMPID,Nthrds,Nprocs,Mc_AN,Stime_atom,Etime_atom,Gc_AN,Cwan,h_AN,Gh_AN,Hwan,i,j,dum,dx,dy,dz,lx,ly,lz,Rn1)
  {

    /* get info. on OpenMP */ 

    OMPID = omp_get_thread_num();
    Nthrds = omp_get_num_threads();
    Nprocs = omp_get_num_procs();

    for (Mc_AN=(OMPID*Matomnum/Nthrds+1); Mc_AN<((OMPID+1)*Matomnum/Nthrds+1); Mc_AN++){

      dtime(&Stime_atom);
      Gc_AN = M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];

      for (h_AN=1; h_AN<=FNAN[Gc_AN]; h_AN++){

	Gh_AN = natn[Gc_AN][h_AN];
	Hwan = WhatSpecies[Gh_AN];

        /*stress*/
	Rn1 = ncn[Gc_AN][h_AN];
	lx = Gxyz[Gc_AN][1] - Gxyz[Gh_AN][1] - atv[Rn1][1];
	ly = Gxyz[Gc_AN][2] - Gxyz[Gh_AN][2] - atv[Rn1][2];
	lz = Gxyz[Gc_AN][3] - Gxyz[Gh_AN][3] - atv[Rn1][3];

	for (i=0; i<Spe_Total_CNO[Cwan]; i++){
	  for (j=0; j<Spe_Total_CNO[Hwan]; j++){

	    if (SpinP_switch==0){
	      dum = 2.0*EDM[0][Mc_AN][h_AN][i][j];
	    }
	    else if (SpinP_switch==1 || SpinP_switch==3){
	      dum = EDM[0][Mc_AN][h_AN][i][j] + EDM[1][Mc_AN][h_AN][i][j];
	    }

	    if (Cnt_switch==0){
	      dx = dum*OLP[1][Mc_AN][h_AN][i][j];
	      dy = dum*OLP[2][Mc_AN][h_AN][i][j];
	      dz = dum*OLP[3][Mc_AN][h_AN][i][j];
	    }
	    else{
	      dx = dum*CntOLP[1][Mc_AN][h_AN][i][j];
	      dy = dum*CntOLP[2][Mc_AN][h_AN][i][j];
	      dz = dum*CntOLP[3][Mc_AN][h_AN][i][j];
	    }
	    
	    My_Stress_threads[OMPID][0] -= dx*lx;
	    My_Stress_threads[OMPID][1] -= dy*lx;
	    My_Stress_threads[OMPID][2] -= dz*lx;
	    My_Stress_threads[OMPID][3] -= dx*ly;
	    My_Stress_threads[OMPID][4] -= dy*ly;
	    My_Stress_threads[OMPID][5] -= dz*ly;
	    My_Stress_threads[OMPID][6] -= dx*lz;
	    My_Stress_threads[OMPID][7] -= dy*lz;
	    My_Stress_threads[OMPID][8] -= dz*lz;
	    
	  } /* Hwan */
	} /*i<Spe_Total_CNO*/
      } /*h_AN*/
      
      dtime(&Etime_atom);
      time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

    } /*Mc_AN*/
  } /*pragma*/

  /*Stress*/

  /* if (myid==Host_ID && 0<level_stdout) printf("debug1, Nthrds0: %d \n",Nthrds0);fflush(stdout); */ /*debug*/

  for(i=0; i<9; i++){
    My_Stress[i]=0.0;
  }

  for(i=0; i<Nthrds0; i++){
    for(j=0; j<9; j++){
      /* printf("myid:%d, i:%d, j:%d \n",myid,i,j);fflush(stdout);*/ /*debug*/
      My_Stress[j] += My_Stress_threads[i][j];
    }
  }
  /* if (myid==Host_ID && 0<level_stdout) printf("debug2 \n");fflush(stdout); */ /*debug*/
  
  MPI_Allreduce(&My_Stress[0],&Stress[0],9,MPI_DOUBLE,MPI_SUM,mpi_comm_level1);

  /* add #5 of stress to Stress_Tensor */

  for (i=0; i<9; i++){
    Stress_Tensor[i] += Stress[i];
  }

  /* show #5 of stress */

  if (myid==Host_ID && 1<level_stdout){
    printf(" Overlap components of stress : \n");fflush(stdout);
    for (i = 0; i<3; i++){
      for (j = 0; j<3; j++){
	printf("%16.8f ", Stress[3*i+j]);
      }
      printf("\n");fflush(stdout);
    }
  }

  dtime(&etime);
  if(myid==0 && measure_time){
    printf("Time for Stress#5=%18.5f\n",etime-stime);fflush(stdout);
  } 

  /* begin(yshiihara) */ 
  for (i=0; i<Nthrds0; i++){
    free(My_Stress_threads[i]);
  }
  free(My_Stress_threads);
  /* end(yshiihara) */

  /****************************************************
            Stress from non-local potentials 
  ****************************************************/

  Stress_HNL(CDM0, iDM0); 

  /* begin(yshiihara) */
  /****************************************************
              Stress for hartree term H1
  ****************************************************/

  dtime(&stime);
  
  Stress_H1();

  dtime(&etime);
  if(myid==0 && measure_time){
    printf("Time for Stress(H1)=%18.5f\n",etime-stime);fflush(stdout);
  }
  
  /* end(yshiihara) */

  /* begin(yshiihara) */
  /****************************************************
               core-core repulsion Stress
  ****************************************************/
  dtime(&stime);
  
  Stress_core();

  dtime(&etime);
  if(myid==0 && measure_time){
    printf("Time for Stress(core)=%18.5f\n",etime-stime);fflush(stdout);
  }
  dtime(&stime);
  
  Stress_H0();

  dtime(&etime);
  if(myid==0 && measure_time){
    printf("Time for Stress(H0)=%18.5f\n",etime-stime);fflush(stdout);
  }
  /* end(yshiihara) */

  /****************************************************
                       DFT+U 
  ****************************************************/

  if (Hub_U_switch==1){
    if (Hub_U_occupation==2) Stress_plus_U_dual(CDM0); /* dual */
  }

  /****************************************************
          contribution of enthalpy term pV

      1 [GPa] = 0.3398827*0.0001 [Hartree/Bohr^3]
  ****************************************************/

  if ( MD_switch==17 || MD_switch==18){
    double coe = 0.3398827*0.0001;
    Stress_Tensor[0] += MD_applied_pressure*coe*Cell_Volume*(double)MD_applied_pressure_flag[0];
    Stress_Tensor[4] += MD_applied_pressure*coe*Cell_Volume*(double)MD_applied_pressure_flag[1];
    Stress_Tensor[8] += MD_applied_pressure*coe*Cell_Volume*(double)MD_applied_pressure_flag[2];

    UpV = MD_applied_pressure*coe*Cell_Volume;
  }

  /* Shutting down this subroutine */
  
  /* freeing of array */
  /* CDM0 */  
  for (k=0; k<=SpinP_switch; k++){
    FNAN[0] = 0;
    for (Mc_AN=0; Mc_AN<=(Matomnum+MatomnumF); Mc_AN++){
      
      if (Mc_AN==0){
	Gc_AN = 0;
	tno0 = 1;
      }
      else{
	Gc_AN = F_M2G[Mc_AN];
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
	  free(CDM0[k][Mc_AN][h_AN][i]);
	}
	free(CDM0[k][Mc_AN][h_AN]);
      }
      free(CDM0[k][Mc_AN]);
    }
    free(CDM0[k]);
  }
  free(CDM0);

  free(Snd_CDM0_Size);
  free(Rcv_CDM0_Size);
  
  /* iDM0 */  
  if ( SO_switch==1 || (Hub_U_switch==1 && SpinP_switch==3) || Constraint_NCS_switch==1 
       || Zeeman_NCS_switch==1 || Zeeman_NCO_switch==1 ){

    for (k=0; k<2; k++){

      FNAN[0] = 0;
      for (Mc_AN=0; Mc_AN<=(Matomnum+MatomnumF); Mc_AN++){

	if (Mc_AN==0){
	  Gc_AN = 0;
	  tno0 = 1;
	}
	else{
	  Gc_AN = F_M2G[Mc_AN];
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
	    free(iDM0[k][Mc_AN][h_AN][i]);
	  }
          free(iDM0[k][Mc_AN][h_AN]);
	}
        free(iDM0[k][Mc_AN]);
      }
      free(iDM0[k]);
    }
    free(iDM0);

    free(Snd_iDM0_Size);
    free(Rcv_iDM0_Size);
  }

  /* for time */

  MPI_Barrier(mpi_comm_level1);
  dtime(&TEtime);
  time0 = TEtime - TStime;

  return time0;
}


void Stress3()
{
  /****************************************************
  	#3 of Stress
 
	dn/depsilon * (VNA + dVH + Vxc)
	or
	dn/depsilon * (dVH + Vxc)
  ****************************************************/
  /* for OpenMP */

  /* MPI */
  int numprocs, myid;
  MPI_Comm_size(mpi_comm_level1, &numprocs);
  MPI_Comm_rank(mpi_comm_level1, &myid);

  /* begin(yshiihara) */
  int spin;
  int Ng1, Ng2, Ng3, DN, BN;
  int XC_P_switch_stress;

  /* Stress */	
  double Stress[9];
  double My_Stress[9];

  {
    int i;
    for (i = 0; i<9; i++) {
      Stress[i] = 0.0;
      My_Stress[i] = 0.0;
    }
  }

  /**********************************************************
            main loop for calculation of Stress #3
  **********************************************************/

  /* shared memory for force */

  double** Vpot_grid = (double**)malloc(sizeof(double*)*(SpinP_switch + 1));
  {
    double* p2 = (double*)malloc(sizeof(double)*(SpinP_switch + 1)*Max_GridN_Atom);

    int spin;
    for (spin = 0; spin<(SpinP_switch + 1); spin++) {
      Vpot_grid[spin] = p2;
      p2 += Max_GridN_Atom;
    }
  }

  double*** dChi0 = (double***)malloc(sizeof(double**)*Max_GridN_Atom);
  {
    double** p2 = (double**)malloc(sizeof(double*)*Max_GridN_Atom*List_YOUSO[7]);
    double* p = (double*)malloc(sizeof(double)*Max_GridN_Atom*List_YOUSO[7] * 3);
    int Nc;
    for (Nc = 0; Nc<Max_GridN_Atom; Nc++) {
      dChi0[Nc] = p2;
      p2 += List_YOUSO[7];
      int i;
      for (i = 0; i<List_YOUSO[7]; i++) {
	dChi0[Nc][i] = p;
	p += 3;
      }
    }
  }

  /* begin(yshiihara) */
  double** dxyz = (double**)malloc(sizeof(double*)*Max_GridN_Atom);
  {
    double* p = (double*)malloc(sizeof(double)*Max_GridN_Atom * 3);
    int Nc;
    for (Nc = 0; Nc<Max_GridN_Atom; Nc++) {
      dxyz[Nc] = p;
      p += 3;
    }
  }
  /* end(yshiihara)   */

  double sumx = 0.0; /* this must be defined out of parallel pragma */
  double sumy = 0.0;
  double sumz = 0.0;
  double sumt0 = 0.0; /* this must be defined out of parallel pragma */
  double sumt1 = 0.0;
  double sumt2 = 0.0;
  double sumt3 = 0.0;
  double sumt4 = 0.0;
  double sumt5 = 0.0;
  double sumt6 = 0.0;
  double sumt7 = 0.0;
  double sumt8 = 0.0;

#pragma omp parallel
  {

    /* allocation of arrays */

    double** dorbs0 = (double**)malloc(sizeof(double*) * 4);
    {
      int i;
      for (i = 0; i<4; i++) {
	dorbs0[i] = (double*)malloc(sizeof(double)*List_YOUSO[7]);
      }
    }
    double* orbs1 = (double*)malloc(sizeof(double)*List_YOUSO[7]);
		
    struct WORK_DORBITAL work_dObs;
    Get_dOrbitals_init(&work_dObs);

    double time1 = 0.0;
    double time2 = 0.0;
    double last_time;
    double current_time;

		
    int Mc_AN;
    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {

      int Gc_AN = M2G[Mc_AN];
      int Cwan = WhatSpecies[Gc_AN];
      int NO0 = Spe_Total_CNO[Cwan];

      /***********************************
   	          calc dOrb0
      ***********************************/
#pragma omp barrier      /* this barrier is necessary to wait (1)clearing sumx, sumy, sumz, and (2)initializing work_dObs */
      dtime(&last_time);

      int Nc;
#pragma omp for
      for (Nc = 0; Nc<GridN_Atom[Gc_AN]; Nc++) {

	int GNc = GridListAtom[Mc_AN][Nc];
	int GRc = CellListAtom[Mc_AN][Nc];
	int MNc = MGridListAtom[Mc_AN][Nc];

	double Cxyz[4];
	Get_Grid_XYZ(GNc, Cxyz);
	double x = Cxyz[1] + atv[GRc][1];
	double y = Cxyz[2] + atv[GRc][2];
	double z = Cxyz[3] + atv[GRc][3];
	double dx = x - Gxyz[Gc_AN][1];
	double dy = y - Gxyz[Gc_AN][2];
	double dz = z - Gxyz[Gc_AN][3];

	/* begin(yshiihara) */
	dxyz[Nc][0] = dx;
	dxyz[Nc][1] = dy;
	dxyz[Nc][2] = dz;
	/* end(yshiihara) */

	if (Cnt_switch == 0) {
	  /* AITUNE201704 Get_dOrbitals(Cwan, dx, dy, dz, dorbs0);*/
	  Get_dOrbitals_work(Cwan, dx, dy, dz, dorbs0, work_dObs);
	} else {
	  Get_Cnt_dOrbitals(Mc_AN, dx, dy, dz, dorbs0);
	}

	int k;
	for (k = 0; k<3; k++) {
	  int i;
	  for (i = 0; i<NO0; i++) {
	    dChi0[Nc][i][k] = dorbs0[k + 1][i];
	  }
	}


	if (SpinP_switch == 0 || SpinP_switch == 1) {

	  int spin;
	  for (spin = 0; spin <= SpinP_switch; spin++) {

	    double Vpt;
	    if (0 <= MNc) {
	      if (E_Field_switch == 1) {

		if (ProExpn_VNA == 0) {

		  Vpt = F_dVHart_flag*dVHart_Grid[MNc]
		    + F_Vxc_flag*Vxc_Grid[spin][MNc]
		    + F_VNA_flag*VNA_Grid[MNc]
		    + F_VEF_flag*VEF_Grid[MNc];

		} else {

		  Vpt = F_dVHart_flag*dVHart_Grid[MNc]
		    + F_Vxc_flag*Vxc_Grid[spin][MNc]
		    + F_VEF_flag*VEF_Grid[MNc];

		}

	      } else {
		if (ProExpn_VNA == 0) {

		  Vpt = F_dVHart_flag*dVHart_Grid[MNc]
		    + F_Vxc_flag*Vxc_Grid[spin][MNc]
		    + F_VNA_flag*VNA_Grid[MNc];

		} else {

		  Vpt = F_dVHart_flag*dVHart_Grid[MNc]
		    + F_Vxc_flag*Vxc_Grid[spin][MNc];

		}
	      }
	    } else {
	      Vpt = 0.0;
	    }

	    if (SpinP_switch == 0) {
	      Vpot_grid[0][Nc] = 4.0 * Vpt;
	    } else if (SpinP_switch == 1) {
	      Vpot_grid[spin][Nc] = 2.0 * Vpt;
	    }

	  }



	} else if (SpinP_switch == 3) {

	  /* spin non-collinear */


	  double ReVpt11;
	  double ReVpt22;
	  double ReVpt21;
	  double ImVpt21;

	  if (0 <= MNc) {
	    if (E_Field_switch == 1) {

	      if (ProExpn_VNA == 0) {

		ReVpt11 = F_dVHart_flag*dVHart_Grid[MNc]
		  + F_Vxc_flag*Vxc_Grid[0][MNc]
		  + F_VNA_flag*VNA_Grid[MNc]
		  + F_VEF_flag*VEF_Grid[MNc];

		ReVpt22 = F_dVHart_flag*dVHart_Grid[MNc]
		  + F_Vxc_flag*Vxc_Grid[1][MNc]
		  + F_VNA_flag*VNA_Grid[MNc]
		  + F_VEF_flag*VEF_Grid[MNc];

		ReVpt21 = F_Vxc_flag*Vxc_Grid[2][MNc];
		ImVpt21 = -F_Vxc_flag*Vxc_Grid[3][MNc];
	      } else {

		ReVpt11 = F_dVHart_flag*dVHart_Grid[MNc]
		  + F_Vxc_flag*Vxc_Grid[0][MNc]
		  + F_VEF_flag*VEF_Grid[MNc];

		ReVpt22 = F_dVHart_flag*dVHart_Grid[MNc]
		  + F_Vxc_flag*Vxc_Grid[1][MNc]
		  + F_VEF_flag*VEF_Grid[MNc];

		ReVpt21 = F_Vxc_flag*Vxc_Grid[2][MNc];
		ImVpt21 = -F_Vxc_flag*Vxc_Grid[3][MNc];
	      }

	    } else {

	      if (ProExpn_VNA == 0) {

		ReVpt11 = F_dVHart_flag*dVHart_Grid[MNc]
		  + F_Vxc_flag*Vxc_Grid[0][MNc]
		  + F_VNA_flag*VNA_Grid[MNc];

		ReVpt22 = F_dVHart_flag*dVHart_Grid[MNc]
		  + F_Vxc_flag*Vxc_Grid[1][MNc]
		  + F_VNA_flag*VNA_Grid[MNc];

		ReVpt21 = F_Vxc_flag*Vxc_Grid[2][MNc];
		ImVpt21 = -F_Vxc_flag*Vxc_Grid[3][MNc];

	      } else {

		ReVpt11 = F_dVHart_flag*dVHart_Grid[MNc] + F_Vxc_flag*Vxc_Grid[0][MNc];
		ReVpt22 = F_dVHart_flag*dVHart_Grid[MNc] + F_Vxc_flag*Vxc_Grid[1][MNc];

		ReVpt21 = F_Vxc_flag*Vxc_Grid[2][MNc];
		ImVpt21 = -F_Vxc_flag*Vxc_Grid[3][MNc];
	      }

	    }
	  } else {
	    ReVpt11 = 0.0;
	    ReVpt22 = 0.0;
	    ReVpt21 = 0.0;
	    ImVpt21 = 0.0;
	  }

	  Vpot_grid[0][Nc] = 2.0 * ReVpt11;
	  Vpot_grid[1][Nc] = 2.0 * ReVpt22;
	  Vpot_grid[2][Nc] = 4.0 * ReVpt21;
	  Vpot_grid[3][Nc] = 4.0 * ImVpt21;

					
	}

      }/* Nc, here omp barrier is called implicitly because of end of for loop */

      dtime(&current_time);
      time1 += current_time - last_time;
      last_time = current_time;

      /***********************************
                 calc Stress #3
      ***********************************/

      int h_AN;
      for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {

	int Gh_AN = natn[Gc_AN][h_AN];
	int Mh_AN = F_G2M[Gh_AN];
	int Rnh = ncn[Gc_AN][h_AN];
	int Hwan = WhatSpecies[Gh_AN];
	int NO1 = Spe_Total_CNO[Hwan];

	int Nog;
#pragma omp for reduction (+:sumx, sumy, sumz, sumt0, sumt1, sumt2, sumt3, sumt4, sumt5, sumt6, sumt7, sumt8)
	for (Nog = 0; Nog<NumOLG[Mc_AN][h_AN]; Nog++) {

	  int Nc = GListTAtoms1[Mc_AN][h_AN][Nog];
	  int Nh = GListTAtoms2[Mc_AN][h_AN][Nog];
					
	  double** const ai_dorbs0 = dChi0[Nc];

	  /* set orbs1 */

	  if (G2ID[Gh_AN] == myid) {
	    int j;
	    for (j = 0; j<NO1; j++) {
	      orbs1[j] = Orbs_Grid[Mh_AN][Nh][j];
	    }
	  } else {
	    int j;
	    for (j = 0; j<NO1; j++) {
	      orbs1[j] = Orbs_Grid_FNAN[Mc_AN][h_AN][Nog][j];
	    }
	  }

	  int spin;
	  for (spin = 0; spin <= SpinP_switch; spin++) {

	    double tmpx = 0.0;
	    double tmpy = 0.0;
	    double tmpz = 0.0;

	    int i;
	    for (i = 0; i<NO0; i++) {
	      double tmp0 = 0.0;
	      int j;
	      for (j = 0; j<NO1; j++) {
		tmp0 += orbs1[j] * DM[0][spin][Mc_AN][h_AN][i][j];
	      }

	      tmpx += ai_dorbs0[i][0] * tmp0;
	      tmpy += ai_dorbs0[i][1] * tmp0;
	      tmpz += ai_dorbs0[i][2] * tmp0;
	    }

	    /* due to difference in the definition between density matrix and density */
						
	    double Vpt = Vpot_grid[spin][Nc];
	    sumx += tmpx * Vpt;
	    sumy += tmpy * Vpt;
	    sumz += tmpz * Vpt;

	    /* begin(yshiihara) */
	    /* The minus sign is for switching the sign of dDen_Grid*/
	    sumt0 -= dxyz[Nc][0] * tmpx * Vpt;
	    sumt1 -= dxyz[Nc][1] * tmpx * Vpt;
	    sumt2 -= dxyz[Nc][2] * tmpx * Vpt;
	    sumt3 -= dxyz[Nc][0] * tmpy * Vpt;
	    sumt4 -= dxyz[Nc][1] * tmpy * Vpt;
	    sumt5 -= dxyz[Nc][2] * tmpy * Vpt;
	    sumt6 -= dxyz[Nc][0] * tmpz * Vpt;
	    sumt7 -= dxyz[Nc][1] * tmpz * Vpt;
	    sumt8 -= dxyz[Nc][2] * tmpz * Vpt;
	    /* end(yshiihara) */

	  }/* spin */

	}/* Nog */
      }/* h_AN, here omp barrier is called implicitly because of end of for loop  */

#pragma omp master
      {
	/**********************************
        only for forces
        Gxyz[Gc_AN][17] += sumx*GridVol;
        Gxyz[Gc_AN][18] += sumy*GridVol;
        Gxyz[Gc_AN][19] += sumz*GridVol;
	***********************************/

	sumx = 0.0;
	sumy = 0.0;
	sumz = 0.0;

	/* begin(yshiihara) */
	My_Stress[0] += sumt0 * GridVol;
	My_Stress[1] += sumt1 * GridVol;
	My_Stress[2] += sumt2 * GridVol;
	My_Stress[3] += sumt3 * GridVol;
	My_Stress[4] += sumt4 * GridVol;
	My_Stress[5] += sumt5 * GridVol;
	My_Stress[6] += sumt6 * GridVol;
	My_Stress[7] += sumt7 * GridVol;
	My_Stress[8] += sumt8 * GridVol;
	/* end(yshiihara)   */

	sumt0 = sumt1 = sumt2 = 0.0;
	sumt3 = sumt4 = sumt5 = 0.0;
	sumt6 = sumt7 = sumt8 = 0.0;
      }

      dtime(&current_time);
      time2 += current_time - last_time;
      last_time = current_time;

    } /* Mc_AN */

#if measure_time
#pragma omp master
    printf("<Stress>  stress(3) myid=%2d  time1, 2 = %lf [s], %lf [s]\n", myid, time1, time2);
#endif

    /* freeing of arrays */

    free(orbs1);

    int i;
    for (i = 0; i<4; i++) {
      free(dorbs0[i]);
    }
    free(dorbs0);

    Get_dOrbitals_free(work_dObs);

  } /* #pragma omp parallel */

  MPI_Allreduce(&My_Stress[0], &Stress[0], 9, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

  /* add #3 of stress to Stress_Tensor */

  {
    int i;
    for (i = 0; i<9; i++) {
      Stress_Tensor[i] += Stress[i];
    }
  }

  /* show #3 of stress */

  if (myid == Host_ID && 1<level_stdout) {
    printf(" Stress #3 : \n"); fflush(stdout);
    int i, j;
    for (i = 0; i<3; i++) {
      for (j = 0; j<3; j++) {
	printf("%16.8f ", Stress[3 * i + j]);
      }
      printf("\n"); fflush(stdout);
    }
  }

  /* end(yshiihara)   */

  /* free */

  free(dChi0[0][0]);
  free(dChi0[0]);
  free(dChi0);
	

  /* begin(yshiihara) */

  free(dxyz[0]);
  free(dxyz);

  /* AITUNE */
  free(Vpot_grid[0]);
  free(Vpot_grid);

  /**********************************************************
     In case of GGA, a stress term coming from
     dfxc/d|\nabra n| *|d|\nabra n|/d\nabra n * \nabra n
     is added.
  **********************************************************/

  /* for GGA stress */
  if (XC_switch == 4) {

    int i, j, k, n, spinmax;
    double My_Stress2[2][9], Stress2[2][9];
    double ***Axc_D;
    double ***Axc_B;
    double ***dDen_D;
    double ***dDen_B;

    /* allocation of arrays */

    Axc_D = (double***)malloc(sizeof(double**) * 2);
    for (spin = 0; spin <= 1; spin++) {
      Axc_D[spin] = (double**)malloc(sizeof(double*) * 3);
      for (i = 0; i<3; i++) {
	Axc_D[spin][i] = (double*)malloc(sizeof(double)*My_NumGridD);
      }
    }

    Axc_B = (double***)malloc(sizeof(double**) * 2);
    for (spin = 0; spin <= 1; spin++) {
      Axc_B[spin] = (double**)malloc(sizeof(double*) * 3);
      for (i = 0; i<3; i++) {
	Axc_B[spin][i] = (double*)malloc(sizeof(double)*My_NumGridB_AB);
      }
    }

    dDen_D = (double***)malloc(sizeof(double**) * 2);
    for (spin = 0; spin <= 1; spin++) {
      dDen_D[spin] = (double**)malloc(sizeof(double*) * 3);
      for (i = 0; i<3; i++) {
	dDen_D[spin][i] = (double*)malloc(sizeof(double)*My_NumGridD);
      }
    }

    dDen_B = (double***)malloc(sizeof(double**) * 2);
    for (spin = 0; spin <= 1; spin++) {
      dDen_B[spin] = (double**)malloc(sizeof(double*) * 3);
      for (i = 0; i<3; i++) {
	dDen_B[spin][i] = (double*)malloc(sizeof(double)*My_NumGridB_AB);
      }
    }

    XC_P_switch_stress = 3;
    /* first argument in Set_XC_Grid was added by S.Ryee */
    Set_XC_Grid(2,XC_P_switch_stress, XC_switch,
		Density_Grid_D[0], Density_Grid_D[1],
		Density_Grid_D[2], Density_Grid_D[3],
		NULL, NULL, NULL, NULL,
		Axc_D, dDen_D);

    /* change data structure: D -> B -> C */

    for (spin = 0; spin <= 1; spin++) {
      Ng1 = Max_Grid_Index_D[1] - Min_Grid_Index_D[1] + 1;
      Ng2 = Max_Grid_Index_D[2] - Min_Grid_Index_D[2] + 1;
      Ng3 = Max_Grid_Index_D[3] - Min_Grid_Index_D[3] + 1;

      for (n = 0; n<Num_Rcv_Grid_B2D[myid]; n++) {
	DN = Index_Rcv_Grid_B2D[myid][n];
	BN = Index_Snd_Grid_B2D[myid][n];

	i = DN / (Ng2*Ng3);
	j = (DN - i*Ng2*Ng3) / Ng3;
	k = DN - i*Ng2*Ng3 - j*Ng3;

	if (!(i <= 1 || (Ng1 - 2) <= i || j <= 1 || (Ng2 - 2) <= j || k <= 1 || (Ng3 - 2) <= k)) {

	  Axc_B[spin][0][BN] = Axc_D[spin][0][DN];
	  Axc_B[spin][1][BN] = Axc_D[spin][1][DN];
	  Axc_B[spin][2][BN] = Axc_D[spin][2][DN];

	  dDen_B[spin][0][BN] = dDen_D[spin][0][DN];
	  dDen_B[spin][1][BN] = dDen_D[spin][1][DN];
	  dDen_B[spin][2][BN] = dDen_D[spin][2][DN];
	}
      }
    }

    /* loop for BN */

    if (SpinP_switch == 0) spinmax = 0;
    else if (SpinP_switch == 1) spinmax = 1;
    else if (SpinP_switch == 3) spinmax = 1;

    for (spin = 0; spin <= spinmax; spin++) {

      for (i = 0; i<9; i++) My_Stress2[spin][i] = 0.0;

      for (BN = 0; BN<My_NumGridB_AB; BN++) {
	My_Stress2[spin][0] -= dDen_B[spin][0][BN] * Axc_B[spin][0][BN];
	My_Stress2[spin][1] -= dDen_B[spin][0][BN] * Axc_B[spin][1][BN];
	My_Stress2[spin][2] -= dDen_B[spin][0][BN] * Axc_B[spin][2][BN];
	My_Stress2[spin][3] -= dDen_B[spin][1][BN] * Axc_B[spin][0][BN];
	My_Stress2[spin][4] -= dDen_B[spin][1][BN] * Axc_B[spin][1][BN];
	My_Stress2[spin][5] -= dDen_B[spin][1][BN] * Axc_B[spin][2][BN];
	My_Stress2[spin][6] -= dDen_B[spin][2][BN] * Axc_B[spin][0][BN];
	My_Stress2[spin][7] -= dDen_B[spin][2][BN] * Axc_B[spin][1][BN];
	My_Stress2[spin][8] -= dDen_B[spin][2][BN] * Axc_B[spin][2][BN];
      }

      MPI_Allreduce(&My_Stress2[spin][0], &Stress2[spin][0], 9, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

      for (i = 0; i<9; i++) Stress2[spin][i] = Stress2[spin][i] * GridVol;

    } /*spin */

    if (SpinP_switch == 0) {
      for (i = 0; i<9; i++) Stress2[1][i] = Stress2[0][i];
    }

    /* add the GGA term in #3 of stress to Stress_Tensor */
    {
      int i;
      for (i = 0; i<9; i++) {
	Stress_Tensor[i] += Stress2[0][i] + Stress2[1][i];
      }
    }

    /* show the GGA term in #3 of stress */

    if (XC_switch == 4 && myid == Host_ID && 1<level_stdout) {
      printf(" Stress #3 (GGA term) : \n"); fflush(stdout);
      int i, j;
      for (i = 0; i<3; i++) {
	for (j = 0; j<3; j++) {
	  printf("%16.8f ", Stress2[0][3 * i + j] + Stress2[1][3 * i + j]);
	}
	printf("\n"); fflush(stdout);
      }
    }

    /* freeing of arrays */

    for (spin = 0; spin <= 1; spin++) {
      for (i = 0; i<3; i++) {
	free(Axc_D[spin][i]);
      }
      free(Axc_D[spin]);
    }
    free(Axc_D);

    for (spin = 0; spin <= 1; spin++) {
      for (i = 0; i<3; i++) {
	free(Axc_B[spin][i]);
      }
      free(Axc_B[spin]);
    }
    free(Axc_B);

    for (spin = 0; spin <= 1; spin++) {
      for (i = 0; i<3; i++) {
	free(dDen_D[spin][i]);
      }
      free(dDen_D[spin]);
    }
    free(dDen_D);

    for (spin = 0; spin <= 1; spin++) {
      for (i = 0; i<3; i++) {
	free(dDen_B[spin][i]);
      }
      free(dDen_B[spin]);
    }
    free(dDen_B);

  } /* if (XC_switch==4) */

  /* end(yshiihara) */
}


      
      


void Stress4()
{
  /****************************************************
                      #4 of Stress

                      n * dVNA/depsilon
  ****************************************************/

  int Mc_AN,Gc_AN,Cwan,Hwan,NO0,NO1;
  int i,j,k,Nc,Nh,GNc,GRc,MNc;
  int h_AN,Gh_AN,Mh_AN,Rnh,spin,Nog;
  double sum,tmp0,r,dx,dy,dz;
  double dvx,dvy,dvz;
  double sumx,sumy,sumz;
  double x,y,z,den;
  double Cxyz[4];

  /* begin(yshiihara) */
  int numprocs,myid;
  /* Stress */
  double **My_Stress_threads;
  double Stress[9];
  double My_Stress[9];
  double sumt[9];

  {
    int i;
    for(i=0; i<9; i++) {
      Stress[i]    = 0.0;
      My_Stress[i] = 0.0;
      sumt[i]      = 0.0;
    }
  }

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);
  MPI_Barrier(mpi_comm_level1);
  /* begin(yshiihara) */
    
  /**********************************************************
              main loop for calculation of Stress #4
  **********************************************************/

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    
    Gc_AN = M2G[Mc_AN];
    Cwan = WhatSpecies[Gc_AN];
    NO0 = Spe_Total_CNO[Cwan];

    /***********************************
                 summation 
    ***********************************/

    sumx = 0.0;
    sumy = 0.0;
    sumz = 0.0;
    /* begin(yshiihara) */
    for(i=0; i<9; i++){
      sumt[i] = 0.0;
    }
    /* begin(yshiihara) */

    for (Nc=0; Nc<GridN_Atom[Gc_AN]; Nc++){

      GNc = GridListAtom[Mc_AN][Nc]; 
      GRc = CellListAtom[Mc_AN][Nc];
      MNc = MGridListAtom[Mc_AN][Nc];

      Get_Grid_XYZ(GNc,Cxyz);
      x = Cxyz[1] + atv[GRc][1];
      y = Cxyz[2] + atv[GRc][2];
      z = Cxyz[3] + atv[GRc][3];
      dx = Gxyz[Gc_AN][1] - x;
      dy = Gxyz[Gc_AN][2] - y;
      dz = Gxyz[Gc_AN][3] - z;
      r = sqrt(dx*dx + dy*dy + dz*dz);

         /* for empty atoms or finite elemens basis */
      if (r<1.0e-10) r = 1.0e-10;

      if (1.0e-14<r){
        tmp0 = Dr_VNAF(Cwan,r);
        dvx = tmp0*dx/r;
        dvy = tmp0*dy/r;
        dvz = tmp0*dz/r;
      }
      else{
        dvx = 0.0;
        dvy = 0.0;
        dvz = 0.0;
      }

      den = Density_Grid[0][MNc] + Density_Grid[1][MNc];
      sumx += den*dvx;
      sumy += den*dvy;
      sumz += den*dvz;
      /* begin(yshiihara) */
      sumt[0] += den*dvx*dx;
      sumt[1] += den*dvy*dx;
      sumt[2] += den*dvz*dx;
      sumt[3] += den*dvx*dy;
      sumt[4] += den*dvy*dy;
      sumt[5] += den*dvz*dy;
      sumt[6] += den*dvx*dz;
      sumt[7] += den*dvy*dz;
      sumt[8] += den*dvz*dz;
      /* end(yshiihara) */
    }

    /* begin(yshiihara) */
    for(j=0; j<9; j++){
      My_Stress[j] += sumt[j]*GridVol*(double)F_VNA_flag;
    }
    /* end(yshiihara) */
    
  }

  /* begin(yshiihara) */
  MPI_Allreduce(&My_Stress[0],&Stress[0],9,MPI_DOUBLE,MPI_SUM,mpi_comm_level1);

  /* add #4 of stress to Stress_Tensor */

  for (i=0; i<9; i++){
    Stress_Tensor[i] += Stress[i];
  }

  /* show #4 of stress */
  
  if (myid==Host_ID && 1<level_stdout){
    printf("  Stress#4 : \n");fflush(stdout);
    for (i = 0; i<3; i++){
      for (j = 0; j<3; j++){
	printf("%16.8f ", Stress[3*i+j]);
      }
      printf("\n");fflush(stdout);
    }
  }
  /* end(yshiihara) */

}



void Stress4B(double *****CDM0)
{
  /****************************************************
                      #4 of Stress

            by the projector expansion of VNA
  ****************************************************/

  int Mc_AN,Gc_AN,Cwan,i,j,k,m,n,h_AN,q_AN,start_q_AN,Mq_AN;
  int jan,kl,Qwan,Gq_AN,Gh_AN,Mh_AN,Hwan,ian;
  int l1,l2,l3,l,LL,Mul1,Num_RVNA,tno0,ncp;
  int tno1,tno2,size1,size2,kk,num,po,po1,po2;
  int numprocs,myid,tag=999,ID,IDS,IDR;
  int **S_array,**R_array;
  int S_comm_flag,R_comm_flag;
  int SA_num,q,Sc_AN,GSc_AN;
  int Sc_wan,Sh_AN,GSh_AN,Sh_wan;
  int Sh_AN2,fan,jg,j0,jg0,Mj_AN0;
  int Original_Mc_AN;

  double rcutA,rcutB,rcut;
  double dEx,dEy,dEz,ene,pref;
  double Stime_atom, Etime_atom;
  double **HVNAx,**HVNAy,**HVNAz;
  double ****HVNA_sts;

  int *VNA_List;
  int *VNA_List2;
  int *Snd_DS_VNA_Size,*Rcv_DS_VNA_Size;  
  int *Indicator;
  Type_DS_VNA *tmp_array;
  Type_DS_VNA *tmp_array2;

  /* for OpenMP */
  int OMPID,Nthrds,Nthrds0,Nprocs,Nloop,ODNloop;
  int *OneD2h_AN,*OneD2q_AN;
  double *dEx_threads;
  double *dEy_threads;
  double *dEz_threads;
  double stime,etime;
  double stime1,etime1;

  /* Stress */
  double **My_Stress_threads;
  double Stress[9],My_Stress[9];

  MPI_Status stat;
  MPI_Request request;

  /* MPI */

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  dtime(&stime);

  /****************************
       allocation of arrays 
  *****************************/

  Indicator = (int*)malloc(sizeof(int)*numprocs);

  S_array = (int**)malloc(sizeof(int*)*numprocs);
  for (ID=0; ID<numprocs; ID++){
    S_array[ID] = (int*)malloc(sizeof(int)*3);
  }

  R_array = (int**)malloc(sizeof(int*)*numprocs);
  for (ID=0; ID<numprocs; ID++){
    R_array[ID] = (int*)malloc(sizeof(int)*3);
  }

  Snd_DS_VNA_Size = (int*)malloc(sizeof(int)*numprocs);
  Rcv_DS_VNA_Size = (int*)malloc(sizeof(int)*numprocs);

  VNA_List  = (int*)malloc(sizeof(int)*(List_YOUSO[34]*(List_YOUSO[35] + 1)+2) ); 
  VNA_List2 = (int*)malloc(sizeof(int)*(List_YOUSO[34]*(List_YOUSO[35] + 1)+2) );

  /* get Nthrds0 */  
#pragma omp parallel shared(Nthrds0)
  {
    Nthrds0 = omp_get_num_threads();
  }

  My_Stress_threads=(double**)malloc(sizeof(double*)*(Nthrds0));
  for (i=0; i<Nthrds0; i++){
    My_Stress_threads[i] = (double*)malloc(sizeof(double)*9);
    for (j=0; j<9; j++) My_Stress_threads[i][j] = 0.0;
  }

  /* initialization of Stress and My_Stress */

  for (i=0; i<9; i++){
    Stress[i]    = 0.0;
    My_Stress[i] = 0.0;
  }

  /*************************************************************
                 contraction of DS_VNA and HVNA2
  *************************************************************/

  if (Cnt_switch==1 && ProExpn_VNA==1){

    Cont_Matrix2(DS_VNA[0],CntDS_VNA[0]);
    Cont_Matrix2(DS_VNA[1],CntDS_VNA[1]);
    Cont_Matrix2(DS_VNA[2],CntDS_VNA[2]);
    Cont_Matrix2(DS_VNA[3],CntDS_VNA[3]);

    Cont_Matrix3(HVNA2[1],CntHVNA2[1]);
    Cont_Matrix3(HVNA2[2],CntHVNA2[2]);
    Cont_Matrix3(HVNA2[3],CntHVNA2[3]);

    Cont_Matrix4(HVNA3[1],CntHVNA3[1]);
    Cont_Matrix4(HVNA3[2],CntHVNA3[2]);
    Cont_Matrix4(HVNA3[3],CntHVNA3[3]);
  }

  /*************************************************************
                  make VNA_List and VNA_List2
  *************************************************************/

  l = 0;
  for (i=0; i<=List_YOUSO[35]; i++){     /* max L */
    for (j=0; j<List_YOUSO[34]; j++){    /* # of radial projectors */
      VNA_List[l]  = i;
      VNA_List2[l] = j;
      l++;
    }
  }

  Num_RVNA = List_YOUSO[34]*(List_YOUSO[35] + 1);

  dtime(&etime);
  if(myid==0 && measure_time){
    printf("Time for part1 of force#4=%18.5f\n",etime-stime);fflush(stdout);
  } 

  /*****************************************************
   if orbital optimization
   copy CntDS_VNA[0] into DS_VNA[0]
  *****************************************************/

  if (Cnt_switch==1){

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

      Gc_AN = F_M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];
      tno0 = Spe_Total_CNO[Cwan];  

      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	Gh_AN = natn[Gc_AN][h_AN];
	Hwan = WhatSpecies[Gh_AN];

	for (i=0; i<tno0; i++){

	  l = 0;
	  for (l1=0; l1<Num_RVNA; l1++){

	    l2 = 2*VNA_List[l1];
	    for (l3=0; l3<=l2; l3++){
	      DS_VNA[0][Mc_AN][h_AN][i][l] = CntDS_VNA[0][Mc_AN][h_AN][i][l];
	      l++;
	    }
	  }
	}
      }
    }
  }

  /*****************************************************
     (1) pre-multiplying DS_VNA[kk] with ene
     (2) copy DS_VNA[kk] or CntDS_VNA[kk] into DS_VNA[kk]
  *****************************************************/

  dtime(&stime);

  /* When Stress.c is integrated into Force.c and Total_Energy.c, the following note is very important.
     if Stress() is called after calling Force(), the following calculations should be skipped.
     Otherwise, DS_VNA[kk] is multiplied by "ene" twice.
  */

  if (0){

#pragma omp parallel shared(CntDS_VNA,DS_VNA,Cnt_switch,VNA_proj_ene,VNA_List2,VNA_List,Num_RVNA,natn,FNAN,Spe_Total_CNO,WhatSpecies,F_M2G,Matomnum) private(kk,OMPID,Nthrds,Nprocs,Gc_AN,Cwan,tno0,Mc_AN,h_AN,Gh_AN,Hwan,i,l,l1,LL,Mul1,ene,l2,l3)
  {

    /* get info. on OpenMP */ 

    OMPID = omp_get_thread_num();
    Nthrds = omp_get_num_threads();
    Nprocs = omp_get_num_procs();

    for (kk=1; kk<=3; kk++){
      for (Mc_AN=(OMPID*Matomnum/Nthrds+1); Mc_AN<((OMPID+1)*Matomnum/Nthrds+1); Mc_AN++){

	Gc_AN = F_M2G[Mc_AN];
	Cwan = WhatSpecies[Gc_AN];
	tno0 = Spe_Total_CNO[Cwan];  

	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];

	  for (i=0; i<tno0; i++){

	    l = 0;
	    for (l1=0; l1<Num_RVNA; l1++){

	      LL   = VNA_List[l1];
	      Mul1 = VNA_List2[l1];

	      ene = VNA_proj_ene[Hwan][LL][Mul1];
	      l2 = 2*VNA_List[l1];

	      if (Cnt_switch==0){
		for (l3=0; l3<=l2; l3++){
		  DS_VNA[kk][Mc_AN][h_AN][i][l] = ene*DS_VNA[kk][Mc_AN][h_AN][i][l];
		  l++;
		}
	      }
 
	      else{
		for (l3=0; l3<=l2; l3++){
		  DS_VNA[kk][Mc_AN][h_AN][i][l] = ene*CntDS_VNA[kk][Mc_AN][h_AN][i][l];
		  l++;
		}
	      }
	    }
	  } 

	} /* h_AN */
      } /* Mc_AN */
    } /* kk */

  } /* #pragma omp parallel */

  }

  dtime(&etime);
  if(myid==0 && measure_time){
    printf("Time for part2 of force#4=%18.5f\n",etime-stime);fflush(stdout);
  } 

  /*****************************************}********************** 
      THE FIRST CASE:
      In case of I=i or I=j 
      for d [ \sum_k <i|k>ek<k|j> ]/dRI  
  ****************************************************************/

  /*******************************************************
   *******************************************************
      multiplying overlap integrals WITH COMMUNICATION
   *******************************************************
   *******************************************************/

  MPI_Barrier(mpi_comm_level1);
  dtime(&stime);

  for (ID=0; ID<numprocs; ID++){
    F_Snd_Num_WK[ID] = 0;
    F_Rcv_Num_WK[ID] = 0;
  }

  do {

    /***********************************
             set the size of data
    ************************************/

    for (ID=0; ID<numprocs; ID++){

      IDS = (myid + ID) % numprocs;
      IDR = (myid - ID + numprocs) % numprocs;

      /* find the data size to send the block data */

      if ( 0<(F_Snd_Num[IDS]-F_Snd_Num_WK[IDS]) ){

	size1 = 0;
	n = F_Snd_Num_WK[IDS];

	Mc_AN = Snd_MAN[IDS][n];
	Gc_AN = Snd_GAN[IDS][n];
	Cwan = WhatSpecies[Gc_AN]; 
	tno1 = Spe_Total_NO[Cwan];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];        
	  Hwan = WhatSpecies[Gh_AN];
	  tno2 = (List_YOUSO[35]+1)*(List_YOUSO[35]+1)*List_YOUSO[34];
	  size1 += tno1*tno2; 
	}
 
	Snd_DS_VNA_Size[IDS] = size1;
	MPI_Isend(&size1, 1, MPI_INT, IDS, tag, mpi_comm_level1, &request);
      }
      else{
	Snd_DS_VNA_Size[IDS] = 0;
      }

      /* receiving of the size of the data */

      if ( 0<(F_Rcv_Num[IDR]-F_Rcv_Num_WK[IDR]) ){
	MPI_Recv(&size2, 1, MPI_INT, IDR, tag, mpi_comm_level1, &stat);
	Rcv_DS_VNA_Size[IDR] = size2;
      }
      else{
	Rcv_DS_VNA_Size[IDR] = 0;
      }

      if ( 0<(F_Snd_Num[IDS]-F_Snd_Num_WK[IDS]) )  MPI_Wait(&request,&stat);

    } /* ID */

      /***********************************
                data transfer
      ************************************/

    for (ID=0; ID<numprocs; ID++){

      IDS = (myid + ID) % numprocs;
      IDR = (myid - ID + numprocs) % numprocs;

      /******************************
             sending of the data 
      ******************************/

      if ( 0<(F_Snd_Num[IDS]-F_Snd_Num_WK[IDS]) ){

	size1 = Snd_DS_VNA_Size[IDS];

	/* allocation of the array */

	tmp_array = (Type_DS_VNA*)malloc(sizeof(Type_DS_VNA)*size1);

	/* multidimentional array to the vector array */

	num = 0;
	n = F_Snd_Num_WK[IDS];

	Mc_AN = Snd_MAN[IDS][n];
	Gc_AN = Snd_GAN[IDS][n];
	Cwan = WhatSpecies[Gc_AN]; 
	tno1 = Spe_Total_NO[Cwan];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];        
	  Hwan = WhatSpecies[Gh_AN];
	  tno2 = (List_YOUSO[35]+1)*(List_YOUSO[35]+1)*List_YOUSO[34];

	  for (i=0; i<tno1; i++){
	    for (j=0; j<tno2; j++){
	      tmp_array[num] = DS_VNA[0][Mc_AN][h_AN][i][j];
	      num++;
	    } 
	  } 
	}

	MPI_Isend(&tmp_array[0], size1, MPI_Type_DS_VNA, IDS, tag, mpi_comm_level1, &request);
      }

      /******************************
          receiving of the block data
      ******************************/

      if ( 0<(F_Rcv_Num[IDR]-F_Rcv_Num_WK[IDR]) ){
        
	size2 = Rcv_DS_VNA_Size[IDR];
	tmp_array2 = (Type_DS_VNA*)malloc(sizeof(Type_DS_VNA)*size2);
	MPI_Recv(&tmp_array2[0], size2, MPI_Type_DS_VNA, IDR, tag, mpi_comm_level1, &stat);

	/* store */

	num = 0;
	n = F_Rcv_Num_WK[IDR];
	Original_Mc_AN = F_TopMAN[IDR] + n;
	Gc_AN = Rcv_GAN[IDR][n];
	Cwan = WhatSpecies[Gc_AN];
	tno1 = Spe_Total_NO[Cwan];

	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  tno2 = (List_YOUSO[35]+1)*(List_YOUSO[35]+1)*List_YOUSO[34];

	  for (i=0; i<tno1; i++){
	    for (j=0; j<tno2; j++){
	      DS_VNA[0][Matomnum+1][h_AN][i][j] = tmp_array2[num];
	      num++;
	    }
	  }
	}

	/* free tmp_array2 */
	free(tmp_array2);

	/*****************************************
               multiplying overlap integrals
	*****************************************/

#pragma omp parallel shared(List_YOUSO,time_per_atom,Gxyz,CDM0,SpinP_switch,CntHVNA2,HVNA2,DS_VNA,Cnt_switch,RMI1,Original_Mc_AN,IDR,Rcv_GAN,F_Rcv_Num_WK,Spe_Total_CNO,F_G2M,natn,FNAN,WhatSpecies,M2G,Matomnum,My_Stress_threads) private(OMPID,Nthrds,Nprocs,Stime_atom,Etime_atom,dEx,dEy,dEz,Gc_AN,Mc_AN,Cwan,fan,h_AN,Gh_AN,Mh_AN,Hwan,ian,jg,j0,jg0,Mj_AN0,po2,q_AN,Gq_AN,Mq_AN,Qwan,jan,kl,HVNAx,HVNAy,HVNAz,HVNA_sts,i,j,k,m,n,pref)
	{

	  /* allocation of array */

	  HVNAx = (double**)malloc(sizeof(double*)*List_YOUSO[7]);
	  for (j=0; j<List_YOUSO[7]; j++){
	    HVNAx[j] = (double*)malloc(sizeof(double)*List_YOUSO[7]);
	  }

	  HVNAy = (double**)malloc(sizeof(double*)*List_YOUSO[7]);
	  for (j=0; j<List_YOUSO[7]; j++){
	    HVNAy[j] = (double*)malloc(sizeof(double)*List_YOUSO[7]);
	  }

	  HVNAz = (double**)malloc(sizeof(double*)*List_YOUSO[7]);
	  for (j=0; j<List_YOUSO[7]; j++){
	    HVNAz[j] = (double*)malloc(sizeof(double)*List_YOUSO[7]);
	  }

	  HVNA_sts = (double****)malloc(sizeof(double***)*4);
	  for (j=0; j<4; j++){
	    HVNA_sts[j] = (double***)malloc(sizeof(double**)*4);
	    for (k=0; k<4; k++){
	      HVNA_sts[j][k] = (double**)malloc(sizeof(double*)*List_YOUSO[7]);
	      for (m=0; m<List_YOUSO[7]; m++){
		HVNA_sts[j][k][m] = (double*)malloc(sizeof(double)*List_YOUSO[7]);
	      }
	    }
	  }

	  /* get info. on OpenMP */ 

	  OMPID = omp_get_thread_num();
	  Nthrds = omp_get_num_threads();
	  Nprocs = omp_get_num_procs();

	  for (Mc_AN=(OMPID*Matomnum/Nthrds+1); Mc_AN<((OMPID+1)*Matomnum/Nthrds+1); Mc_AN++){

	    dtime(&Stime_atom);

	    dEx = 0.0; 
	    dEy = 0.0; 
	    dEz = 0.0; 

	    Gc_AN = M2G[Mc_AN];
	    Cwan = WhatSpecies[Gc_AN];
	    fan = FNAN[Gc_AN];

	    h_AN = 0;
	    Gh_AN = natn[Gc_AN][h_AN];
	    Mh_AN = F_G2M[Gh_AN];
	    Hwan = WhatSpecies[Gh_AN];
	    ian = Spe_Total_CNO[Hwan];

	    n = F_Rcv_Num_WK[IDR];
	    jg = Rcv_GAN[IDR][n];

	    for (j0=0; j0<=fan; j0++){

	      jg0 = natn[Gc_AN][j0];
	      Mj_AN0 = F_G2M[jg0];

	      po2 = 0;
	      if (Original_Mc_AN==Mj_AN0){
		po2 = 1;
		q_AN = j0;
	      }

	      if (po2==1){

		Gq_AN = natn[Gc_AN][q_AN];
		Mq_AN = F_G2M[Gq_AN];
		Qwan = WhatSpecies[Gq_AN];
		jan = Spe_Total_CNO[Qwan];
		kl = RMI1[Mc_AN][h_AN][q_AN];

		if (Cnt_switch==0) {
		  dHVNA(0,Mc_AN,h_AN,q_AN,DS_VNA,HVNA2,HVNA3,HVNAx,HVNAy,HVNAz,HVNA_sts);
		}
		else { 
		  dHVNA(0,Mc_AN,h_AN,q_AN,DS_VNA,CntHVNA2,CntHVNA3,HVNAx,HVNAy,HVNAz,HVNA_sts);
		}

		/* contribution of force = Trace(CDM0*dH) */
		/* spin non-polarization */

		if (SpinP_switch==0){

                  if (q_AN==h_AN) pref = 2.0;
                  else            pref = 4.0; 

                  /* for forces */                 

		  for (i=0; i<ian; i++){
		    for (j=0; j<jan; j++){
		      dEx += pref*CDM0[0][Mh_AN][kl][i][j]*HVNAx[i][j];
		      dEy += pref*CDM0[0][Mh_AN][kl][i][j]*HVNAy[i][j];
		      dEz += pref*CDM0[0][Mh_AN][kl][i][j]*HVNAz[i][j];
		    }
		  }

		  /* for stress */                 

		  for (m=1; m<=3; m++){
		    for (n=1; n<=3; n++){
		      for (i=0; i<ian; i++){
			for (j=0; j<jan; j++){
			  My_Stress_threads[OMPID][(m-1)*3+(n-1)] += pref*CDM0[0][Mh_AN][kl][i][j]*HVNA_sts[m][n][i][j];
			}
		      }
		    }
		  }

		}

		/* else */

		else{

  		  if (q_AN==h_AN) pref = 1.0;
		  else            pref = 2.0; 

                  /* for forces */                 

		  for (i=0; i<ian; i++){
		    for (j=0; j<jan; j++){

		      dEx += pref*(  CDM0[0][Mh_AN][kl][i][j]
			           + CDM0[1][Mh_AN][kl][i][j] )*HVNAx[i][j];
		      dEy += pref*(  CDM0[0][Mh_AN][kl][i][j]
			           + CDM0[1][Mh_AN][kl][i][j] )*HVNAy[i][j];
		      dEz += pref*(  CDM0[0][Mh_AN][kl][i][j]
			           + CDM0[1][Mh_AN][kl][i][j] )*HVNAz[i][j];
		    }
		  }

                  /* for stress */                 

		  for (m=1; m<=3; m++){
		    for (n=1; n<=3; n++){
		      for (i=0; i<ian; i++){
			for (j=0; j<jan; j++){
			  My_Stress_threads[OMPID][(m-1)*3+(n-1)] += pref*( CDM0[0][Mh_AN][kl][i][j]*HVNA_sts[m][n][i][j]
			 	                                          + CDM0[1][Mh_AN][kl][i][j]*HVNA_sts[m][n][i][j]);
			}
		      }
		    }
		  }

		} /* else */

	      } /* if (po2==1) */
	    } /* j0 */             

	    /* force from #4B */

	    Gxyz[Gc_AN][41] += dEx;
	    Gxyz[Gc_AN][42] += dEy;
	    Gxyz[Gc_AN][43] += dEz;

	    /* timing */
	    dtime(&Etime_atom);
	    time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

	  } /* Mc_AN */

	  /* freeing of array */

	  for (j=0; j<List_YOUSO[7]; j++){
	    free(HVNAx[j]);
	  }
	  free(HVNAx);

	  for (j=0; j<List_YOUSO[7]; j++){
	    free(HVNAy[j]);
	  }
	  free(HVNAy);

	  for (j=0; j<List_YOUSO[7]; j++){
	    free(HVNAz[j]);
	  }
	  free(HVNAz);

	  for (j=0; j<4; j++){
	    for (k=0; k<4; k++){
	      for (m=0; m<List_YOUSO[7]; m++){
		free(HVNA_sts[j][k][m]);
	      }
              free(HVNA_sts[j][k]);
	    }
            free(HVNA_sts[j]);
	  }
          free(HVNA_sts);

	} /* #pragma omp parallel */

	/********************************************
              increment of F_Rcv_Num_WK[IDR] 
	********************************************/

	F_Rcv_Num_WK[IDR]++;

      } /* if ( 0<(F_Snd_Num[IDS]-F_Snd_Num_WK[IDS]) ) */

      if ( 0<(F_Snd_Num[IDS]-F_Snd_Num_WK[IDS]) ) {

	MPI_Wait(&request,&stat);
	free(tmp_array);  /* freeing of array */

	/********************************************
             increment of F_Snd_Num_WK[IDS]
	********************************************/

	F_Snd_Num_WK[IDS]++;
      } 

    } /* ID */

    /*****************************************************
      check whether all the communications have finished
    *****************************************************/

    po = 0;
    for (ID=0; ID<numprocs; ID++){

      IDS = (myid + ID) % numprocs;
      IDR = (myid - ID + numprocs) % numprocs;

      if ( 0<(F_Snd_Num[IDS]-F_Snd_Num_WK[IDS]) ) po += F_Snd_Num[IDS]-F_Snd_Num_WK[IDS];
      if ( 0<(F_Rcv_Num[IDR]-F_Rcv_Num_WK[IDR]) ) po += F_Rcv_Num[IDR]-F_Rcv_Num_WK[IDR];
    }

  } while (po!=0);

  dtime(&etime);
  if(myid==0 && measure_time){
    printf("Time for part3 of force#4=%18.5f\n",etime-stime);fflush(stdout);
  } 

  /*******************************************************
   *******************************************************
      THE FIRST CASE:
      multiplying overlap integrals WITHOUT COMMUNICATION
   *******************************************************
   *******************************************************/

  dtime(&stime);

#pragma omp parallel shared(time_per_atom,Gxyz,CDM0,SpinP_switch,CntHVNA2,HVNA2,DS_VNA,Cnt_switch,RMI1,FNAN,Spe_Total_CNO,WhatSpecies,F_G2M,natn,M2G,Matomnum,List_YOUSO,My_Stress_threads) private(HVNAx,HVNAy,HVNAz,HVNA_sts,OMPID,Nthrds,Nprocs,Mc_AN,Stime_atom,Etime_atom,dEx,dEy,dEz,Gc_AN,h_AN,Gh_AN,Mh_AN,Hwan,ian,q_AN,Gq_AN,Mq_AN,Qwan,jan,kl,i,j,k,m,n,kk,pref)
  {

    /* allocation of array */

    HVNAx = (double**)malloc(sizeof(double*)*List_YOUSO[7]);
    for (j=0; j<List_YOUSO[7]; j++){
      HVNAx[j] = (double*)malloc(sizeof(double)*List_YOUSO[7]);
    }

    HVNAy = (double**)malloc(sizeof(double*)*List_YOUSO[7]);
    for (j=0; j<List_YOUSO[7]; j++){
      HVNAy[j] = (double*)malloc(sizeof(double)*List_YOUSO[7]);
    }

    HVNAz = (double**)malloc(sizeof(double*)*List_YOUSO[7]);
    for (j=0; j<List_YOUSO[7]; j++){
      HVNAz[j] = (double*)malloc(sizeof(double)*List_YOUSO[7]);
    }

    HVNA_sts = (double****)malloc(sizeof(double***)*4);
    for (j=0; j<4; j++){
      HVNA_sts[j] = (double***)malloc(sizeof(double**)*4);
      for (k=0; k<4; k++){
	HVNA_sts[j][k] = (double**)malloc(sizeof(double*)*List_YOUSO[7]);
	for (m=0; m<List_YOUSO[7]; m++){
	  HVNA_sts[j][k][m] = (double*)malloc(sizeof(double)*List_YOUSO[7]);
	}
      }
    }

    /* get info. on OpenMP */ 

    OMPID = omp_get_thread_num();
    Nthrds = omp_get_num_threads();
    Nprocs = omp_get_num_procs();

    for (Mc_AN=(OMPID*Matomnum/Nthrds+1); Mc_AN<((OMPID+1)*Matomnum/Nthrds+1); Mc_AN++){

      dtime(&Stime_atom);

      dEx = 0.0; 
      dEy = 0.0; 
      dEz = 0.0; 

      Gc_AN = M2G[Mc_AN];
      h_AN = 0;
      Gh_AN = natn[Gc_AN][h_AN];
      Mh_AN = F_G2M[Gh_AN];
      Hwan = WhatSpecies[Gh_AN];
      ian = Spe_Total_CNO[Hwan];

      for (q_AN=h_AN; q_AN<=FNAN[Gc_AN]; q_AN++){

	Gq_AN = natn[Gc_AN][q_AN];
	Mq_AN = F_G2M[Gq_AN];

	if (Mq_AN<=Matomnum){

	  Qwan = WhatSpecies[Gq_AN];
	  jan = Spe_Total_CNO[Qwan];
	  kl = RMI1[Mc_AN][h_AN][q_AN];

	  if (Cnt_switch==0) {
	    dHVNA(0,Mc_AN,h_AN,q_AN,DS_VNA,HVNA2,HVNA3,HVNAx,HVNAy,HVNAz,HVNA_sts);
	  }
	  else { 
	    dHVNA(0,Mc_AN,h_AN,q_AN,DS_VNA,CntHVNA2,CntHVNA3,HVNAx,HVNAy,HVNAz,HVNA_sts);
	  }

	  if (SpinP_switch==0){

            if (q_AN==h_AN) pref = 2.0;
            else            pref = 4.0; 

            /* for forces */                 

	    for (i=0; i<ian; i++){
	      for (j=0; j<jan; j++){
		dEx += pref*CDM0[0][Mh_AN][kl][i][j]*HVNAx[i][j];
		dEy += pref*CDM0[0][Mh_AN][kl][i][j]*HVNAy[i][j];
		dEz += pref*CDM0[0][Mh_AN][kl][i][j]*HVNAz[i][j];
	      }
	    }

	    /* for stress */                 

	    for (m=1; m<=3; m++){
	      for (n=1; n<=3; n++){
		for (i=0; i<ian; i++){
		  for (j=0; j<jan; j++){
		    My_Stress_threads[OMPID][(m-1)*3+(n-1)] += pref*CDM0[0][Mh_AN][kl][i][j]*HVNA_sts[m][n][i][j];
		  }
		}
	      }
	    }

	  }

	  /* else */

	  else{

	    if (q_AN==h_AN) pref = 1.0;
	    else            pref = 2.0; 

            /* for forces */                 

	    for (i=0; i<ian; i++){
	      for (j=0; j<jan; j++){

		dEx += pref*(  CDM0[0][Mh_AN][kl][i][j]
			     + CDM0[1][Mh_AN][kl][i][j] )*HVNAx[i][j];
		dEy += pref*(  CDM0[0][Mh_AN][kl][i][j]
			     + CDM0[1][Mh_AN][kl][i][j] )*HVNAy[i][j];
		dEz += pref*(  CDM0[0][Mh_AN][kl][i][j]
			     + CDM0[1][Mh_AN][kl][i][j] )*HVNAz[i][j];
	      }
	    }

	    /* for stress */                 

	    for (m=1; m<=3; m++){
	      for (n=1; n<=3; n++){
		for (i=0; i<ian; i++){
		  for (j=0; j<jan; j++){
		    My_Stress_threads[OMPID][(m-1)*3+(n-1)] += pref*( CDM0[0][Mh_AN][kl][i][j]*HVNA_sts[m][n][i][j]
	 			                                    + CDM0[1][Mh_AN][kl][i][j]*HVNA_sts[m][n][i][j]);
		  }
		}
	      }
	    }

	  }
	}
      }

      /* force from #4B */

      Gxyz[Gc_AN][41] += dEx;
      Gxyz[Gc_AN][42] += dEy;
      Gxyz[Gc_AN][43] += dEz;

      /* timing */
      dtime(&Etime_atom);
      time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

    } /* Mc_AN */

      /* freeing of array */

    for (j=0; j<List_YOUSO[7]; j++){
      free(HVNAx[j]);
    }
    free(HVNAx);

    for (j=0; j<List_YOUSO[7]; j++){
      free(HVNAy[j]);
    }
    free(HVNAy);

    for (j=0; j<List_YOUSO[7]; j++){
      free(HVNAz[j]);
    }
    free(HVNAz);

    for (j=0; j<4; j++){
      for (k=0; k<4; k++){
	for (m=0; m<List_YOUSO[7]; m++){
	  free(HVNA_sts[j][k][m]);
	}
	free(HVNA_sts[j][k]);
      }
      free(HVNA_sts[j]);
    }
    free(HVNA_sts);

  } /* #pragma omp parallel */

  /* summing up results calculated by OpenMP threads */

  for(i=0; i<Nthrds0; i++){
    for(j=0; j<9; j++){
      My_Stress[j] += My_Stress_threads[i][j];
    }
  }

  /* MPI_Allreduce: My_Stress */

  MPI_Allreduce(My_Stress, Stress, 9, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

  for (i=0; i<9; i++){
    Stress[i] = Stress[i]*(double)F_VNA_flag;
  }

  /* add stress coming from the projector expansion of VNA to Stress_Tensor */

  for (i=0; i<9; i++){
    Stress_Tensor[i] += Stress[i];
  }

  /* show #4 of stress */
  
  if (myid==Host_ID && 1<level_stdout){
    printf("  Stress: projector expansion of VNA \n");fflush(stdout);
    for (i = 0; i<3; i++){
      for (j = 0; j<3; j++){
	printf("%16.8f ", Stress[3*i+j]);
      }
      printf("\n");fflush(stdout);
    }
  }

  dtime(&etime);
  if(myid==0 && measure_time){
    printf("Time for part4 of force#4=%18.5f\n",etime-stime);fflush(stdout);
  } 
        
  /***********************************
            freeing of arrays 
  ************************************/

  free(Indicator);

  for (ID=0; ID<numprocs; ID++){
    free(S_array[ID]);
  }
  free(S_array);

  for (ID=0; ID<numprocs; ID++){
    free(R_array[ID]);
  }
  free(R_array);

  free(Snd_DS_VNA_Size);
  free(Rcv_DS_VNA_Size);

  free(VNA_List);
  free(VNA_List2);

  for (i=0; i<Nthrds0; i++){
    free(My_Stress_threads[i]);
  }
  free(My_Stress_threads);

}



void Stress_core()
{
  /****************************************************
                       Ecore
  ****************************************************/

  int Mc_AN,Gc_AN,Cwan,Rn,h_AN,Gh_AN,Hwan;
  int i,spin,spinmax;
  double My_Ecore,Ecore,Zc,Zh,dum,dum2;
  double *My_Ecore_threads;
  double TmpEcore,dEx,dEy,dEz,r,lx,ly,lz;
  int numprocs,myid;
  double Stime_atom,Etime_atom;
  /* for OpenMP */
  int OMPID,Nthrds,Nthrds0,Nprocs,Nloop;

  /* begin(yshiihara) */
  /* Stress */
  int j;
  double **My_Stress_threads;
  double    Stress[9];
  double My_Stress[9];
  double      sumt[9];
  {
    int i;
    for(i=0; i<9; i++) {
      Stress[i]    = 0.0;
      My_Stress[i] = 0.0;
      sumt[i]      = 0.0;
    }
  }
  /* end(yshiihara) */

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /* get Nthrds0 */  
#pragma omp parallel shared(Nthrds0)
  {
    Nthrds0 = omp_get_num_threads();
  }
  
  /* allocation of array */
  /* begin(yshiihara) */ 

  My_Stress_threads=(double**)malloc(sizeof(double*)*(Nthrds0));
  for (i=0; i<Nthrds0; i++){
    My_Stress_threads[i] = (double*)malloc(sizeof(double)*9);
    for (j=0; j<9; j++) My_Stress_threads[i][j] = 0.0;
  }
  /* end(yshiihara) */
  
  /* begin(yshiihara) */ 
#pragma omp parallel shared(level_stdout,time_per_atom,atv,Gxyz,Dis,ncn,natn,FNAN,Spe_Core_Charge,WhatSpecies,M2G,Matomnum,My_Ecore_threads,DecEscc,My_Stress_threads) private(OMPID,Nthrds,Nprocs,Mc_AN,Stime_atom,Gc_AN,Cwan,Zc,dEx,dEy,dEz,h_AN,Gh_AN,Rn,Hwan,Zh,r,lx,ly,lz,dum,dum2,Etime_atom,TmpEcore,i,j,sumt)
  {
    /* end(yshiihara) */
    
    /* get info. on OpenMP */ 
    
    OMPID = omp_get_thread_num();
    Nthrds = omp_get_num_threads();
    Nprocs = omp_get_num_procs();
    
    for (Mc_AN=(OMPID*Matomnum/Nthrds+1); Mc_AN<((OMPID+1)*Matomnum/Nthrds+1); Mc_AN++){
      
      dtime(&Stime_atom);
      
      Gc_AN = M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];
      Zc = Spe_Core_Charge[Cwan];
      dEx = 0.0;
      dEy = 0.0;
      dEz = 0.0;
      TmpEcore = 0.0;
      for (i=0; i<9; i++){
	sumt[i] = 0.0;
      }
      
      for (h_AN=1; h_AN<=FNAN[Gc_AN]; h_AN++){
	
	Gh_AN = natn[Gc_AN][h_AN];
	Rn = ncn[Gc_AN][h_AN];
	Hwan = WhatSpecies[Gh_AN];
	Zh = Spe_Core_Charge[Hwan];
	r = Dis[Gc_AN][h_AN];
	
	/* for empty atoms or finite elemens basis */
	if (r<1.0e-10) r = 1.0e-10;
	
	lx = (Gxyz[Gc_AN][1] - Gxyz[Gh_AN][1] - atv[Rn][1])/r;
	ly = (Gxyz[Gc_AN][2] - Gxyz[Gh_AN][2] - atv[Rn][2])/r;
	lz = (Gxyz[Gc_AN][3] - Gxyz[Gh_AN][3] - atv[Rn][3])/r;
	dum = Zc*Zh/r;
	dum2 = dum/r;
        TmpEcore += dum; 
	dEx = dEx - lx*dum2;
	dEy = dEy - ly*dum2;
	dEz = dEz - lz*dum2;
	/*begin(yshiihara) */ 
	sumt[0] -= (Gxyz[Gc_AN][1] - Gxyz[Gh_AN][1] - atv[Rn][1])*lx*dum2;
	sumt[1] -= (Gxyz[Gc_AN][1] - Gxyz[Gh_AN][1] - atv[Rn][1])*ly*dum2;
	sumt[2] -= (Gxyz[Gc_AN][1] - Gxyz[Gh_AN][1] - atv[Rn][1])*lz*dum2;
	sumt[3] -= (Gxyz[Gc_AN][2] - Gxyz[Gh_AN][2] - atv[Rn][2])*lx*dum2;
	sumt[4] -= (Gxyz[Gc_AN][2] - Gxyz[Gh_AN][2] - atv[Rn][2])*ly*dum2;
	sumt[5] -= (Gxyz[Gc_AN][2] - Gxyz[Gh_AN][2] - atv[Rn][2])*lz*dum2;
	sumt[6] -= (Gxyz[Gc_AN][3] - Gxyz[Gh_AN][3] - atv[Rn][3])*lx*dum2;
	sumt[7] -= (Gxyz[Gc_AN][3] - Gxyz[Gh_AN][3] - atv[Rn][3])*ly*dum2;
	sumt[8] -= (Gxyz[Gc_AN][3] - Gxyz[Gh_AN][3] - atv[Rn][3])*lz*dum2;
	/* end(yshiihara) */
	
      } /* h_AN */
      
      /****************************************************
                        #6 of force
         Contribution from the core-core repulsions
      ****************************************************/
      /*begin(yshiihara) */ 
      for (i=0; i<9; i++){
	My_Stress_threads[OMPID][i] += sumt[i];
      }
      /*end(yshiihara) */

      dtime(&Etime_atom);
      time_per_atom[Gc_AN] += Etime_atom - Stime_atom;
    }

  } /* #pragma omp parallel */

  for (Nloop=0; Nloop<Nthrds0; Nloop++){
    for(i=0; i<9; i++){
      My_Stress[i] += 0.5*My_Stress_threads[Nloop][i];
    }
  }
  
  MPI_Allreduce(&My_Stress[0],&Stress[0],9,MPI_DOUBLE,MPI_SUM,mpi_comm_level1);

  /* add stress coming from core-core to Stress_Tensor */

  for (i=0; i<9; i++){
    Stress_Tensor[i] += Stress[i];
  }

  /* show stress coming from core-core */
    
  if (myid==Host_ID && 1<level_stdout){
    printf(" Components of stress (Core-core): \n");fflush(stdout);
    for (i = 0; i<3; i++){
      for (j = 0; j<3; j++){
	printf("%16.8f ", Stress[3*i+j]);
      }
      printf("\n");fflush(stdout);
    }
  }
  
  /* end(yshiihara) */

  /* freeing of array */
  
  /* begin(yshiihara) */
  /* free(My_Ecore_threads); */
  for (i=0; i<Nthrds0; i++){
    free(My_Stress_threads[i]);
  }
  free(My_Stress_threads);
  /* end(yshiihara) */
}




void Stress_H0()
{
  /****************************************************
              EH0 = -1/2\int n^a(r) V^a_H dr
  ****************************************************/

  int Mc_AN,Gc_AN,h_AN,Gh_AN,num,gnum,i;
  int wan,wan1,wan2,Nd,n1,n2,n3,spin,spinmax;
  double bc,dx,x,y,z,r1,r2,rho0,xx;
  double Scale_Grid_Ecut,TmpEH0;
  double EH0ij[4],My_EH0,EH0,tmp0;
  double *Fx,*Fy,*Fz,*g0;
  double dEx,dEy,dEz,Dx,Sx;
  double Z1,Z2,factor;
  double My_dEx,My_dEy,My_dEz;
  int numprocs,myid,ID;
  double stime,etime;
  double Stime_atom, Etime_atom;
  /* for OpenMP */
  int OMPID,Nthrds,Nthrds0,Nprocs,Nloop;
  /* begin(yshiihara) */
  /*double *My_EH0_threads;*/
  /* Stress */
  int j;
  double **My_Stress_threads;
  double    Stress[9];
  double My_Stress[9];
  double      sumt[9];
  int    Rn;
  double lx,ly,lz;
  
  int MD_iter;
  
  for(i=0; i<9; i++) {
    Stress[i]    = 0.0;
    My_Stress[i] = 0.0;
  }
  /* end(yshiihara) */
  
  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /****************************************************
             Set of atomic density on grids
  ****************************************************/

  /*(yshiihara) Following part can be neglected because it must be already performed in Total_Energy.c. */
  MD_iter = 1; 
  if (MD_iter==1){
    
    dtime(&stime);

    Scale_Grid_Ecut = 16.0*600.0;
    
    /* estimate the size of an array g0 */
    
    Max_Nd = 0;
    for (wan=0; wan<SpeciesNum; wan++){
      Spe_Spe2Ban[wan] = wan;
      bc = Spe_Atom_Cut1[wan];
      dx = PI/sqrt(Scale_Grid_Ecut);
      Nd = 2*(int)(bc/dx) + 1;
      if (Max_Nd<Nd) Max_Nd = Nd;
    }
    
    /* estimate sizes of arrays GridX,Y,Z_EH0, Arho_EH0, and Wt_EH0 */

    Max_TGN_EH0 = 0;
    for (wan=0; wan<SpeciesNum; wan++){
      
      Spe_Spe2Ban[wan] = wan;
      bc = Spe_Atom_Cut1[wan];
      dx = PI/sqrt(Scale_Grid_Ecut);

      Nd = 2*(int)(bc/dx) + 1;
      dx = 2.0*bc/(double)(Nd-1);
      gnum = Nd*CoarseGL_Mesh;

      if (Max_TGN_EH0<gnum) Max_TGN_EH0 = gnum;

      if (2<=level_stdout){
        printf("<Calc_EH0> A spe=%2d 1D-grids=%2d 3D-grids=%2d\n",wan,Nd,gnum);fflush(stdout);
      }
    }

    /* estimate sizes of arrays GridX,Y,Z_EH0, Arho_EH0, and Wt_EH0 */

    Max_TGN_EH0 = 0;
    for (wan=0; wan<SpeciesNum; wan++){

      Spe_Spe2Ban[wan] = wan;
      bc = Spe_Atom_Cut1[wan];
      dx = PI/sqrt(Scale_Grid_Ecut);

      Nd = 2*(int)(bc/dx) + 1;
      dx = 2.0*bc/(double)(Nd-1);
      gnum = Nd*CoarseGL_Mesh;

      if (Max_TGN_EH0<gnum) Max_TGN_EH0 = gnum;

      if (2<=level_stdout){
        printf("<Calc_EH0> A spe=%2d 1D-grids=%2d 3D-grids=%2d\n",wan,Nd,gnum);fflush(stdout);
      }
    }
    /* allocation of arrays GridX,Y,Z_EH0, Arho_EH0, and Wt_EH0 */

    Max_TGN_EH0 += 10;
    Allocate_Arrays(4);

    /* calculate GridX,Y,Z_EH0 and Wt_EH0 */

#pragma omp parallel shared(Spe_Num_Mesh_PAO,Spe_PAO_XV,Spe_Atomic_Den,Max_Nd,level_stdout,TGN_EH0,Wt_EH0,Arho_EH0,GridZ_EH0,GridY_EH0,GridX_EH0,Scale_Grid_Ecut,Spe_Atom_Cut1,dv_EH0,Spe_Spe2Ban,SpeciesNum,CoarseGL_Abscissae,CoarseGL_Weight) private(OMPID,Nthrds,Nprocs,wan,bc,dx,Nd,gnum,n1,n2,n3,x,y,z,tmp0,r1,rho0,g0,Sx,Dx,xx)
    {

      /* allocation of arrays g0 */

      g0 = (double*)malloc(sizeof(double)*Max_Nd);

      /* get info. on OpenMP */

      OMPID = omp_get_thread_num();
      Nthrds = omp_get_num_threads();
      Nprocs = omp_get_num_procs();

      for (wan=OMPID*SpeciesNum/Nthrds; wan<(OMPID+1)*SpeciesNum/Nthrds; wan++){

        Spe_Spe2Ban[wan] = wan;
        bc = Spe_Atom_Cut1[wan];
        dx = PI/sqrt(Scale_Grid_Ecut);
        Nd = 2*(int)(bc/dx) + 1;
        dx = 2.0*bc/(double)(Nd-1);
        dv_EH0[wan] = dx;

        for (n1=0; n1<Nd; n1++){
          g0[n1] = dx*(double)n1 - bc;
        }

        gnum = 0;
        y = 0.0;

        Sx = Spe_Atom_Cut1[wan] + 0.0;
        Dx = Spe_Atom_Cut1[wan] - 0.0;

        for (n3=0; n3<Nd; n3++){

          z = g0[n3];
          tmp0 = z*z;
	  for (n1=0; n1<CoarseGL_Mesh; n1++){

            x = 0.50*(Dx*CoarseGL_Abscissae[n1] + Sx);
            xx = 0.5*log(x*x + tmp0);

            GridX_EH0[wan][gnum] = x;
            GridY_EH0[wan][gnum] = y;
            GridZ_EH0[wan][gnum] = z;

            rho0 = KumoF( Spe_Num_Mesh_PAO[wan], xx,
                          Spe_PAO_XV[wan], Spe_PAO_RV[wan], Spe_Atomic_Den[wan]);

            Arho_EH0[wan][gnum] = rho0;
            Wt_EH0[wan][gnum] = PI*x*CoarseGL_Weight[n1]*Dx;

            gnum++;
          }

        } /* n3 */

        TGN_EH0[wan] = gnum;

        if (2<=level_stdout){
          printf("<Calc_EH0> B spe=%2d 1D-grids=%2d 3D-grids=%2d\n",wan,Nd,gnum);fflush(stdout);
        }

      } /* wan */

      /* free */
      free(g0);
	   
    } /* #pragma omp parallel */
    
    dtime(&etime);
    if(myid==0 && measure_time){
      printf("Time for part1 of EH0=%18.5f\n",etime-stime);fflush(stdout);
    }
    
  } /* if (MD_iter==1) */

  /****************************************************                                                                             
    calculation of scaling factors:                                                                                                 
  ****************************************************/

  if (MD_iter==1){

    for (wan1=0; wan1<SpeciesNum; wan1++){

      r1 = Spe_Atom_Cut1[wan1];
      Z1 = Spe_Core_Charge[wan1];

      for (wan2=0; wan2<SpeciesNum; wan2++){

        /* EH0_TwoCenter_at_Cutoff is parallelized by OpenMP */
        EH0_TwoCenter_at_Cutoff(wan1, wan2, EH0ij);

        r2 = Spe_Atom_Cut1[wan2];
        Z2 = Spe_Core_Charge[wan2];
        tmp0 = Z1*Z2/(r1+r2);


        if (1.0e-20<fabs(EH0ij[0])){
          EH0_scaling[wan1][wan2] = tmp0/EH0ij[0];
        }
        else{
          EH0_scaling[wan1][wan2] = 0.0;
        }

      }
    }
  }
  /*(yshiihara) above part can be neglected because it must be performed in force part */

  /****************************************************                                                                             
                -1/2\int n^a(r) V^a_H dr                                                                                            
  ****************************************************/

  dtime(&stime);

  /* get Nthrds0 */
#pragma omp parallel shared(Nthrds0)
  {
    Nthrds0 = omp_get_num_threads();
  }

  /* allocation of array */
  /* begin(yshiihara) */
  My_Stress_threads=(double**)malloc(sizeof(double*)*(Nthrds0));
  for (i=0; i<Nthrds0; i++){
    My_Stress_threads[i] = (double*)malloc(sizeof(double)*9);
    for (j=0; j<9; j++) My_Stress_threads[i][j] = 0.0;
  }
  /* end(yshiihara) */

    /* begin(yshiihara) */
#pragma omp parallel shared(Gxyz,atv,time_per_atom,RMI1,EH0_scaling,natn,ncn,FNAN,WhatSpecies,M2G,Matomnum,My_Stress_threads) private(OMPID,Nthrds,Nprocs,Mc_AN,Stime_atom,Gc_AN,wan1,h_AN,Gh_AN,wan2,factor,Etime_atom,TmpEH0,lx,ly,lz,Rn)
  {
    /* end(yshiihara) */

    double EH0ij[4];    

    /* get info. on OpenMP */ 

    OMPID = omp_get_thread_num();
    Nthrds = omp_get_num_threads();
    Nprocs = omp_get_num_procs();

    for (Mc_AN=(OMPID*Matomnum/Nthrds+1); Mc_AN<((OMPID+1)*Matomnum/Nthrds+1); Mc_AN++){

      dtime(&Stime_atom);

      Gc_AN = M2G[Mc_AN];
      wan1 = WhatSpecies[Gc_AN];
      
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	
	Gh_AN = natn[Gc_AN][h_AN];
	wan2 = WhatSpecies[Gh_AN];
	
	/* begin(yshiihara) */
	Rn = ncn[Gc_AN][h_AN];
	lx = Gxyz[Gc_AN][1] - Gxyz[Gh_AN][1] - atv[Rn][1];
	ly = Gxyz[Gc_AN][2] - Gxyz[Gh_AN][2] - atv[Rn][2];
	lz = Gxyz[Gc_AN][3] - Gxyz[Gh_AN][3] - atv[Rn][3];
	/* end(yshiihara) */
	
	if (h_AN==0) factor = 1.0;
	else         factor = EH0_scaling[wan1][wan2];

	EH0_TwoCenter(Gc_AN, h_AN, EH0ij);


	/* begin(yshiihara) */

	My_Stress_threads[OMPID][0] -= 0.25*factor*EH0ij[1]*lx;
	My_Stress_threads[OMPID][1] -= 0.25*factor*EH0ij[2]*lx;
	My_Stress_threads[OMPID][2] -= 0.25*factor*EH0ij[3]*lx;
	My_Stress_threads[OMPID][3] -= 0.25*factor*EH0ij[1]*ly;
	My_Stress_threads[OMPID][4] -= 0.25*factor*EH0ij[2]*ly;
	My_Stress_threads[OMPID][5] -= 0.25*factor*EH0ij[3]*ly;
	My_Stress_threads[OMPID][6] -= 0.25*factor*EH0ij[1]*lz;
	My_Stress_threads[OMPID][7] -= 0.25*factor*EH0ij[2]*lz;
	My_Stress_threads[OMPID][8] -= 0.25*factor*EH0ij[3]*lz;

	/* debug 
	   if (myid==Host_ID && 0<level_stdout){
	   printf(" My_Stress_threads1 : \n");fflush(stdout);
	   for (i = 0; i<3; i++){
	   for (j = 0; j<3; j++){
	   printf("%16.8f ", My_Stress_threads[OMPID][3*i+j]);
	   }
	   printf("\n");fflush(stdout);
	   }
	   }
	*/
	/* end(yshiihara) */

	if (h_AN==0) factor = 1.0;
	else         factor = EH0_scaling[wan2][wan1];

	EH0_TwoCenter(Gh_AN, RMI1[Mc_AN][h_AN][0], EH0ij);

	/* begin(yshiihara) */

	My_Stress_threads[OMPID][0] += 0.25*factor*EH0ij[1]*lx;
	My_Stress_threads[OMPID][1] += 0.25*factor*EH0ij[2]*lx;
	My_Stress_threads[OMPID][2] += 0.25*factor*EH0ij[3]*lx;
	My_Stress_threads[OMPID][3] += 0.25*factor*EH0ij[1]*ly;
	My_Stress_threads[OMPID][4] += 0.25*factor*EH0ij[2]*ly;
	My_Stress_threads[OMPID][5] += 0.25*factor*EH0ij[3]*ly;
	My_Stress_threads[OMPID][6] += 0.25*factor*EH0ij[1]*lz;
	My_Stress_threads[OMPID][7] += 0.25*factor*EH0ij[2]*lz;
	My_Stress_threads[OMPID][8] += 0.25*factor*EH0ij[3]*lz;

	/* debug
	   if (myid==Host_ID && 0<level_stdout){
	   printf(" My_Stress_threads2 : \n");fflush(stdout);
	   for (i = 0; i<3; i++){
	   for (j = 0; j<3; j++){
	   printf("%16.8f ", My_Stress_threads[OMPID][3*i+j]);
	   }
	   printf("\n");fflush(stdout);
	   }
	   }
	*/
	/* end(yshiihara) */

      } /* h_AN */

      dtime(&Etime_atom);
      time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

    } /* Mc_AN */
  } /* #pragma omp parallel */

  /* begin(yshiihara) */

  for(i=0; i<9; i++){
    My_Stress[i]=0.0;
  }
  for(i=0; i<Nthrds0; i++){
    for(j=0; j<9; j++){
      /* printf("myid:%d, i:%d, j:%d \n",myid,i,j);fflush(stdout); */ /*debug*/
      My_Stress[j] += My_Stress_threads[i][j];
    }
  }

  /* debug
  if (myid==Host_ID && 0<level_stdout){
    printf(" Components of My_Stress : \n");fflush(stdout);
    for (i = 0; i<3; i++){
      for (j = 0; j<3; j++){
	printf("%16.8f ", My_Stress[3*i+j]);
      }
      printf("\n");fflush(stdout);
    }
  }
  */
  
  MPI_Allreduce(&My_Stress[0],&Stress[0],9,MPI_DOUBLE,MPI_SUM,mpi_comm_level1);

  /* add stress coming from H0 to Stress_Tensor */

  for (i=0; i<9; i++){
    Stress_Tensor[i] += Stress[i];
  }

  /* show stress coming from H0 */
  
  if (myid==Host_ID && 1<level_stdout){
    printf(" Components of stress (H0) : \n");fflush(stdout);
    for (i = 0; i<3; i++){
      for (j = 0; j<3; j++){
	printf("%16.8f ", Stress[3*i+j]);
      }
      printf("\n");fflush(stdout);
    }
  }
  /* end(yshiihara) */

  /* begin(yshiihara) */

  /* freeing of array */
  /* begin(yshiihara) */

  for (i=0; i<Nthrds0; i++){
    free(My_Stress_threads[i]);
  }
  free(My_Stress_threads);

  /* measuring elapsed time */

  dtime(&etime);
  if(myid==0 && measure_time){
    printf("Time for part2 of EH0=%18.5f\n",etime-stime);fflush(stdout);
  }

}

void Stress_H1()
{ 
  int k1,k2,k3;
  int N2D,GNs,GN,BN_CB;
  int N3[4];
  double time0;
  double tmp0,sk1,sk2,sk3;
  double Gx,Gy,Gz,fac_invG2;
  double TStime,TEtime,etime;
  int numprocs,myid,tag=999,ID,IDS,IDR;
  MPI_Status stat;
  MPI_Request request;

  /* Stress */
  double **My_Stress_threads;
  double    Stress[9];
  double My_Stress[9];
  double      sumt[9];
  double      invG4,tmp1;
  int       i,j;

  double   *ReRhok,*ImRhok;

  /* allocation of arrays */  
  
  ReRhok = (double*)malloc(sizeof(double)*My_Max_NumGridB); 
  for (i=0; i<My_Max_NumGridB; i++) ReRhok[i] = 0.0;
  ImRhok = (double*)malloc(sizeof(double)*My_Max_NumGridB); 
  for (i=0; i<My_Max_NumGridB; i++) ImRhok[i] = 0.0;

  /* initialization of arrays */  
  
  for(i=0; i<9; i++) {
    Stress[i]    = 0.0;
    My_Stress[i] = 0.0;
  }

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /*if (myid==Host_ID && 0<level_stdout){
    printf("<Stress>  Poisson's equation using FFT...\n");
    }*/

  MPI_Barrier(mpi_comm_level1);
  dtime(&TStime);

  /****************************************************
            FFT of difference charge density 
  ****************************************************/

  etime = FFT_Density(0,ReRhok,ImRhok);

  /****************************************************
                       4*PI/G2/N^3
  ****************************************************/

  tmp0 = 1.0/(double)(Ngrid1*Ngrid2*Ngrid3);

  N2D = Ngrid3*Ngrid2;
  GNs = ((myid*N2D+numprocs-1)/numprocs)*Ngrid1;

  for (BN_CB=0; BN_CB<My_NumGridB_CB; BN_CB++){

    GN = BN_CB + GNs;     
    k3 = GN/(Ngrid2*Ngrid1);    
    k2 = (GN - k3*Ngrid2*Ngrid1)/Ngrid1;
    k1 = GN - k3*Ngrid2*Ngrid1 - k2*Ngrid1; 

    if (k1<Ngrid1/2) sk1 = (double)k1;
    else             sk1 = (double)(k1 - Ngrid1);

    if (k2<Ngrid2/2) sk2 = (double)k2;
    else             sk2 = (double)(k2 - Ngrid2);

    if (k3<Ngrid3/2) sk3 = (double)k3;
    else             sk3 = (double)(k3 - Ngrid3);

    Gx = sk1*rtv[1][1] + sk2*rtv[2][1] + sk3*rtv[3][1];
    Gy = sk1*rtv[1][2] + sk2*rtv[2][2] + sk3*rtv[3][2]; 
    Gz = sk1*rtv[1][3] + sk2*rtv[2][3] + sk3*rtv[3][3];
    invG4 =  tmp0/(Gx*Gx + Gy*Gy + Gz*Gz);
    invG4 = 4.0*PI*invG4*invG4*(double)F_dVHart_flag;
    tmp1  = ReRhok[BN_CB]*ReRhok[BN_CB]+ImRhok[BN_CB]*ImRhok[BN_CB];
    tmp1  = tmp1*Cell_Volume;
      
    if (k1!=0 || k2!=0 || k3!=0){
      My_Stress[0] += invG4*Gx*Gx*tmp1;
      My_Stress[1] += invG4*Gy*Gx*tmp1;
      My_Stress[2] += invG4*Gz*Gx*tmp1;
      My_Stress[3] += invG4*Gx*Gy*tmp1;
      My_Stress[4] += invG4*Gy*Gy*tmp1;
      My_Stress[5] += invG4*Gz*Gy*tmp1;
      My_Stress[6] += invG4*Gx*Gz*tmp1;
      My_Stress[7] += invG4*Gy*Gz*tmp1;
      My_Stress[8] += invG4*Gz*Gz*tmp1;
    }
  }

  MPI_Allreduce(&My_Stress[0],&Stress[0],9,MPI_DOUBLE,MPI_SUM,mpi_comm_level1);

  /* add stress coming from H1 to Stress_Tensor */

  for (i=0; i<9; i++){
    Stress_Tensor[i] += Stress[i];
  }

  /* show stress coming from H1 */

  if (myid==Host_ID && 1<level_stdout){
    printf(" Components of stress (H1_G) : \n");fflush(stdout);
    for (i = 0; i<3; i++){
      for (j = 0; j<3; j++){
	printf("%16.8f ", Stress[3*i+j]);
      }
      printf("\n");fflush(stdout);
    }
  }

  /* freeing of arrays */  

  free(ReRhok);
  free(ImRhok);
  
  /* for time */
  MPI_Barrier(mpi_comm_level1);
  dtime(&TEtime);
  time0 = TEtime - TStime;
  return;
}




void Stress_HNL(double *****CDM0, double *****iDM0)
{
  /****************************************************
                  Stress arising from HNL
  ****************************************************/

  int Mc_AN,Gc_AN,Cwan,i,j,k,m,h_AN,q_AN,Mq_AN,start_q_AN;
  int jan,kl,km,kl1,Qwan,Gq_AN,Gh_AN,Mh_AN,Hwan,ian;
  int l1,l2,l3,l,LL,Mul1,tno0,ncp,so;
  int tno1,tno2,size1,size2,n,kk,num,po,po1,po2;
  int numprocs,myid,tag=999,ID,IDS,IDR;
  int **S_array,**R_array;
  int S_comm_flag,R_comm_flag;
  int SA_num,q,Sc_AN,GSc_AN,smul;
  int Sc_wan,Sh_AN,GSh_AN,Sh_wan;
  int Sh_AN2,fan,jg,j0,jg0,Mj_AN0;
  int Original_Mc_AN;

  double rcutA,rcutB,rcut;
  double dEx,dEy,dEz,ene,pref;
  double Stime_atom, Etime_atom;
  dcomplex ***Hx,***Hy,***Hz;
  dcomplex ***Hx0,***Hy0,***Hz0;
  dcomplex ***Hx1,***Hy1,***Hz1;
  dcomplex *****H_sts;
  int *Snd_DS_NL_Size,*Rcv_DS_NL_Size;  
  int *Indicator;
  double *tmp_array;
  double *tmp_array2;

  /* for OpenMP */
  int OMPID,Nthrds,Nthrds0,Nprocs,Nloop,ODNloop;
  int *OneD2h_AN,*OneD2q_AN;
  double *dEx_threads;
  double *dEy_threads;
  double *dEz_threads;
  double stime,etime;
  double stime1,etime1;

  /* Stress */
  double **My_Stress_threads;
  double Stress[9],My_Stress[9];

  MPI_Status stat;
  MPI_Request request;

  /* MPI */

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  dtime(&stime);

  /****************************
       allocation of arrays 
  *****************************/

  Indicator = (int*)malloc(sizeof(int)*numprocs);

  S_array = (int**)malloc(sizeof(int*)*numprocs);
  for (ID=0; ID<numprocs; ID++){
    S_array[ID] = (int*)malloc(sizeof(int)*3);
  }

  R_array = (int**)malloc(sizeof(int*)*numprocs);
  for (ID=0; ID<numprocs; ID++){
    R_array[ID] = (int*)malloc(sizeof(int)*3);
  }

  Snd_DS_NL_Size = (int*)malloc(sizeof(int)*numprocs);
  Rcv_DS_NL_Size = (int*)malloc(sizeof(int)*numprocs);

  /* get Nthrds0 */  
#pragma omp parallel shared(Nthrds0)
  {
    Nthrds0 = omp_get_num_threads();
  }

  My_Stress_threads=(double**)malloc(sizeof(double*)*(Nthrds0));
  for (i=0; i<Nthrds0; i++){
    My_Stress_threads[i] = (double*)malloc(sizeof(double)*9);
    for (j=0; j<9; j++) My_Stress_threads[i][j] = 0.0;
  }

  /* initialization of Stress and My_Stress */

  for (i=0; i<9; i++){
    Stress[i]    = 0.0;
    My_Stress[i] = 0.0;
  }

  /*************************************************************
                    contraction of DS_NL
     Note: DS_NL is overwritten by CntDS_NL in Cont_Matrix1().
  *************************************************************/

  if (Cnt_switch==1){
    for (so=0; so<(SO_switch+1); so++){
      Cont_Matrix1(DS_NL[so][0],CntDS_NL[so][0]);
      Cont_Matrix1(DS_NL[so][1],CntDS_NL[so][1]);
      Cont_Matrix1(DS_NL[so][2],CntDS_NL[so][2]);
      Cont_Matrix1(DS_NL[so][3],CntDS_NL[so][3]);
    }
  }

  /*****************************************}********************** 
      THE FIRST CASE:
      In case of I=i or I=j 
      for d [ \sum_k <i|k>ek<k|j> ]/dRI  
  ****************************************************************/

  /*******************************************************
   *******************************************************
       multiplying overlap integrals WITH COMMUNICATION

       In case of I=i or I=j 
       for d [ \sum_k <i|k>ek<k|j> ]/dRI  
   *******************************************************
   *******************************************************/

  MPI_Barrier(mpi_comm_level1);
  dtime(&stime);

  Hx0 = (dcomplex***)malloc(sizeof(dcomplex**)*3);
  for (i=0; i<3; i++){
    Hx0[i] = (dcomplex**)malloc(sizeof(dcomplex*)*List_YOUSO[7]);
    for (j=0; j<List_YOUSO[7]; j++){
      Hx0[i][j] = (dcomplex*)malloc(sizeof(dcomplex)*List_YOUSO[7]);
    }
  }

  Hy0 = (dcomplex***)malloc(sizeof(dcomplex**)*3);
  for (i=0; i<3; i++){
    Hy0[i] = (dcomplex**)malloc(sizeof(dcomplex*)*List_YOUSO[7]);
    for (j=0; j<List_YOUSO[7]; j++){
      Hy0[i][j] = (dcomplex*)malloc(sizeof(dcomplex)*List_YOUSO[7]);
    }
  }

  Hz0 = (dcomplex***)malloc(sizeof(dcomplex**)*3);
  for (i=0; i<3; i++){
    Hz0[i] = (dcomplex**)malloc(sizeof(dcomplex*)*List_YOUSO[7]);
    for (j=0; j<List_YOUSO[7]; j++){
      Hz0[i][j] = (dcomplex*)malloc(sizeof(dcomplex)*List_YOUSO[7]);
    }
  }

  Hx1 = (dcomplex***)malloc(sizeof(dcomplex**)*3);
  for (i=0; i<3; i++){
    Hx1[i] = (dcomplex**)malloc(sizeof(dcomplex*)*List_YOUSO[7]);
    for (j=0; j<List_YOUSO[7]; j++){
      Hx1[i][j] = (dcomplex*)malloc(sizeof(dcomplex)*List_YOUSO[7]);
    }
  }

  Hy1 = (dcomplex***)malloc(sizeof(dcomplex**)*3);
  for (i=0; i<3; i++){
    Hy1[i] = (dcomplex**)malloc(sizeof(dcomplex*)*List_YOUSO[7]);
    for (j=0; j<List_YOUSO[7]; j++){
      Hy1[i][j] = (dcomplex*)malloc(sizeof(dcomplex)*List_YOUSO[7]);
    }
  }

  Hz1 = (dcomplex***)malloc(sizeof(dcomplex**)*3);
  for (i=0; i<3; i++){
    Hz1[i] = (dcomplex**)malloc(sizeof(dcomplex*)*List_YOUSO[7]);
    for (j=0; j<List_YOUSO[7]; j++){
      Hz1[i][j] = (dcomplex*)malloc(sizeof(dcomplex)*List_YOUSO[7]);
    }
  }

  H_sts = (dcomplex*****)malloc(sizeof(dcomplex****)*3);
  for (i=0; i<3; i++){
    H_sts[i] = (dcomplex****)malloc(sizeof(dcomplex***)*4);
    for (j=0; j<4; j++){
      H_sts[i][j] = (dcomplex***)malloc(sizeof(dcomplex**)*4);
      for (k=0; k<4; k++){
	H_sts[i][j][k] = (dcomplex**)malloc(sizeof(dcomplex*)*List_YOUSO[7]);
	for (m=0; m<List_YOUSO[7]; m++){
	  H_sts[i][j][k][m] = (dcomplex*)malloc(sizeof(dcomplex)*List_YOUSO[7]);
	}
      }
    }
  }

  /* initialization of F_Snd_Num_WK and F_Rcv_Num_WK */

  for (ID=0; ID<numprocs; ID++){
    F_Snd_Num_WK[ID] = 0;
    F_Rcv_Num_WK[ID] = 0;
  }

  /* start do loop */

  do {

    /***********************************                                                            
            set the size of data                                                                      
    ************************************/

    for (ID=0; ID<numprocs; ID++){

      IDS = (myid + ID) % numprocs;
      IDR = (myid - ID + numprocs) % numprocs;

      /* find the data size to send the block data */

      if ( 0<(F_Snd_Num[IDS]-F_Snd_Num_WK[IDS]) ){

	size1 = 0;
	n = F_Snd_Num_WK[IDS];

	Mc_AN = Snd_MAN[IDS][n];
	Gc_AN = Snd_GAN[IDS][n];
	Cwan = WhatSpecies[Gc_AN];
	tno1 = Spe_Total_NO[Cwan];

	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  tno2 = Spe_Total_VPS_Pro[Hwan];
	  size1 += (VPS_j_dependency[Hwan]+1)*tno1*tno2;
	}

	Snd_DS_NL_Size[IDS] = size1;
	MPI_Isend(&size1, 1, MPI_INT, IDS, tag, mpi_comm_level1, &request);
      }
      else{
	Snd_DS_NL_Size[IDS] = 0;
      }

      /* receiving of the size of the data */

      if ( 0<(F_Rcv_Num[IDR]-F_Rcv_Num_WK[IDR]) ){
	MPI_Recv(&size2, 1, MPI_INT, IDR, tag, mpi_comm_level1, &stat);
	Rcv_DS_NL_Size[IDR] = size2;
      }
      else{
	Rcv_DS_NL_Size[IDR] = 0;
      }

      if ( 0<(F_Snd_Num[IDS]-F_Snd_Num_WK[IDS]) )  MPI_Wait(&request,&stat);

    } /* ID */

    /***********************************
               data transfer
    ************************************/

    for (ID=0; ID<numprocs; ID++){

      IDS = (myid + ID) % numprocs;
      IDR = (myid - ID + numprocs) % numprocs;

      /******************************
            sending of the data 
      ******************************/

      if ( 0<(F_Snd_Num[IDS]-F_Snd_Num_WK[IDS]) ){

	size1 = Snd_DS_NL_Size[IDS];

	/* allocation of the array */

	tmp_array = (double*)malloc(sizeof(double)*size1);

	/* multidimentional array to the vector array */

	num = 0;
	n = F_Snd_Num_WK[IDS];

	Mc_AN = Snd_MAN[IDS][n];
	Gc_AN = Snd_GAN[IDS][n];
	Cwan = WhatSpecies[Gc_AN]; 
	tno1 = Spe_Total_NO[Cwan];

	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];        
	  Hwan = WhatSpecies[Gh_AN];
	  tno2 = Spe_Total_VPS_Pro[Hwan];

	  for (so=0; so<=VPS_j_dependency[Hwan]; so++){
	    for (i=0; i<tno1; i++){
	      for (j=0; j<tno2; j++){
		tmp_array[num] = DS_NL[so][0][Mc_AN][h_AN][i][j];
		num++;
	      } 
	    } 
	  }
	}

	MPI_Isend(&tmp_array[0], size1, MPI_DOUBLE, IDS, tag, mpi_comm_level1, &request);
      }

      /******************************
        receiving of the block data
      ******************************/

      if ( 0<(F_Rcv_Num[IDR]-F_Rcv_Num_WK[IDR]) ){
        
	size2 = Rcv_DS_NL_Size[IDR];
	tmp_array2 = (double*)malloc(sizeof(double)*size2);
	MPI_Recv(&tmp_array2[0], size2, MPI_DOUBLE, IDR, tag, mpi_comm_level1, &stat);

	/* store */

	num = 0;
	n = F_Rcv_Num_WK[IDR];
	Original_Mc_AN = F_TopMAN[IDR] + n;

	Gc_AN = Rcv_GAN[IDR][n];
	Cwan = WhatSpecies[Gc_AN];
	tno1 = Spe_Total_NO[Cwan];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  tno2 = Spe_Total_VPS_Pro[Hwan];

	  for (so=0; so<=VPS_j_dependency[Hwan]; so++){
	    for (i=0; i<tno1; i++){
	      for (j=0; j<tno2; j++){
		DS_NL[so][0][Matomnum+1][h_AN][i][j] = tmp_array2[num];
		num++;
	      }
	    }
	  }
	}

	/* free tmp_array2 */
	free(tmp_array2);

	/*****************************************************************
                           multiplying overlap integrals
	*****************************************************************/

	for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

	  dtime(&Stime_atom);

	  dEx = 0.0; 
	  dEy = 0.0; 
	  dEz = 0.0; 

	  Gc_AN = M2G[Mc_AN];
	  Cwan = WhatSpecies[Gc_AN];
	  fan = FNAN[Gc_AN];

	  h_AN = 0;
	  Gh_AN = natn[Gc_AN][h_AN];
	  Mh_AN = F_G2M[Gh_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  ian = Spe_Total_CNO[Hwan];

	  n = F_Rcv_Num_WK[IDR];
	  jg = Rcv_GAN[IDR][n];

	  for (j0=0; j0<=fan; j0++){

	    jg0 = natn[Gc_AN][j0];
	    Mj_AN0 = F_G2M[jg0];

	    po2 = 0;
	    if (Original_Mc_AN==Mj_AN0){
	      po2 = 1;
	      q_AN = j0;
	    }

	    if (po2==1){

	      Gq_AN = natn[Gc_AN][q_AN];
	      Mq_AN = F_G2M[Gq_AN];
	      Qwan = WhatSpecies[Gq_AN];
	      jan = Spe_Total_CNO[Qwan];
	      kl = RMI1[Mc_AN][h_AN][q_AN];

 	      dHNL(0,Mc_AN,h_AN,q_AN,DS_NL,Hx0,Hy0,Hz0,H_sts);

	      /* contribution of force = Trace(CDM0*dH) */
	      /* spin non-polarization */

	      if (SpinP_switch==0){

                if (q_AN==h_AN) pref = 2.0;
                else            pref = 4.0; 

                /* for forces */                 

		for (i=0; i<ian; i++){
		  for (j=0; j<jan; j++){

		    dEx += pref*CDM0[0][Mh_AN][kl][i][j]*Hx0[0][i][j].r;
		    dEy += pref*CDM0[0][Mh_AN][kl][i][j]*Hy0[0][i][j].r;
		    dEz += pref*CDM0[0][Mh_AN][kl][i][j]*Hz0[0][i][j].r;
		  }
		}

                /* for stress */                 

                for (m=1; m<=3; m++){
		  for (n=1; n<=3; n++){
		    for (i=0; i<ian; i++){
		      for (j=0; j<jan; j++){
			My_Stress[(m-1)*3+(n-1)] += pref*CDM0[0][Mh_AN][kl][i][j]*H_sts[0][m][n][i][j].r;
		      }
		    }
		  }
		}
	      }

	      /* collinear spin polarized or non-colliear without SO and LDA+U */

	      else if (SpinP_switch==1 || (SpinP_switch==3 && SO_switch==0 && Hub_U_switch==0
		   && Constraint_NCS_switch==0 && Zeeman_NCS_switch==0 && Zeeman_NCO_switch==0)){ 

		if (q_AN==h_AN) pref = 1.0;
		else            pref = 2.0; 

                /* for forces */                 

		for (i=0; i<Spe_Total_CNO[Hwan]; i++){
		  for (j=0; j<Spe_Total_CNO[Qwan]; j++){

		    dEx += pref*(  CDM0[0][Mh_AN][kl][i][j]*Hx0[0][i][j].r
			         + CDM0[1][Mh_AN][kl][i][j]*Hx0[1][i][j].r);
		    dEy += pref*(  CDM0[0][Mh_AN][kl][i][j]*Hy0[0][i][j].r
				 + CDM0[1][Mh_AN][kl][i][j]*Hy0[1][i][j].r);
		    dEz += pref*(  CDM0[0][Mh_AN][kl][i][j]*Hz0[0][i][j].r
				 + CDM0[1][Mh_AN][kl][i][j]*Hz0[1][i][j].r);
		  }
		}

                /* for stress */                 

                for (m=1; m<=3; m++){
		  for (n=1; n<=3; n++){
		    for (i=0; i<ian; i++){
		      for (j=0; j<jan; j++){
			My_Stress[(m-1)*3+(n-1)] += pref*( CDM0[0][Mh_AN][kl][i][j]*H_sts[0][m][n][i][j].r
	                                                 + CDM0[1][Mh_AN][kl][i][j]*H_sts[1][m][n][i][j].r);
		      }
		    }
		  }
		}

	      }

	      /* spin non-collinear with spin-orbit coupling or with LDA+U */

	      else if ( SpinP_switch==3 && (SO_switch==1 || (Hub_U_switch==1 && F_U_flag==1) 
                     || 1<=Constraint_NCS_switch || Zeeman_NCS_switch==1 || Zeeman_NCO_switch==1)){

                if (q_AN==h_AN){

                  /* for forces */ 

		  for (i=0; i<Spe_Total_CNO[Hwan]; i++){
		    for (j=0; j<Spe_Total_CNO[Qwan]; j++){

		      dEx +=
			  CDM0[0][Mh_AN][kl][i][j]*Hx0[0][i][j].r
			- iDM0[0][Mh_AN][kl][i][j]*Hx0[0][i][j].i
			+ CDM0[1][Mh_AN][kl][i][j]*Hx0[1][i][j].r
			- iDM0[1][Mh_AN][kl][i][j]*Hx0[1][i][j].i
		    + 2.0*CDM0[2][Mh_AN][kl][i][j]*Hx0[2][i][j].r
		    - 2.0*CDM0[3][Mh_AN][kl][i][j]*Hx0[2][i][j].i;

		      dEy += 
			  CDM0[0][Mh_AN][kl][i][j]*Hy0[0][i][j].r
			- iDM0[0][Mh_AN][kl][i][j]*Hy0[0][i][j].i
			+ CDM0[1][Mh_AN][kl][i][j]*Hy0[1][i][j].r
			- iDM0[1][Mh_AN][kl][i][j]*Hy0[1][i][j].i
		    + 2.0*CDM0[2][Mh_AN][kl][i][j]*Hy0[2][i][j].r
		    - 2.0*CDM0[3][Mh_AN][kl][i][j]*Hy0[2][i][j].i; 

		      dEz +=  
			  CDM0[0][Mh_AN][kl][i][j]*Hz0[0][i][j].r
			- iDM0[0][Mh_AN][kl][i][j]*Hz0[0][i][j].i
			+ CDM0[1][Mh_AN][kl][i][j]*Hz0[1][i][j].r
			- iDM0[1][Mh_AN][kl][i][j]*Hz0[1][i][j].i
		    + 2.0*CDM0[2][Mh_AN][kl][i][j]*Hz0[2][i][j].r
		    - 2.0*CDM0[3][Mh_AN][kl][i][j]*Hz0[2][i][j].i; 

		    }
		  }

                  /* for stress */ 

		  for (m=1; m<=3; m++){
		    for (n=1; n<=3; n++){
		      for (i=0; i<Spe_Total_CNO[Hwan]; i++){
			for (j=0; j<Spe_Total_CNO[Qwan]; j++){

			  My_Stress[(m-1)*3+(n-1)] +=
			          CDM0[0][Mh_AN][kl][i][j]*H_sts[0][m][n][i][j].r
			        - iDM0[0][Mh_AN][kl][i][j]*H_sts[0][m][n][i][j].i
			        + CDM0[1][Mh_AN][kl][i][j]*H_sts[1][m][n][i][j].r
			        - iDM0[1][Mh_AN][kl][i][j]*H_sts[1][m][n][i][j].i
			    + 2.0*CDM0[2][Mh_AN][kl][i][j]*H_sts[2][m][n][i][j].r
			    - 2.0*CDM0[3][Mh_AN][kl][i][j]*H_sts[2][m][n][i][j].i;
			}
		      }
		    }
		  }

		} /* if (q_AN==h_AN) */

                else {

                  /* for forces */ 

		  for (i=0; i<Spe_Total_CNO[Hwan]; i++){  /* Hwan */
		    for (j=0; j<Spe_Total_CNO[Qwan]; j++){ /* Qwan  */

		      dEx += 
			  CDM0[0][Mh_AN][kl ][i][j]*Hx0[0][i][j].r
			- iDM0[0][Mh_AN][kl ][i][j]*Hx0[0][i][j].i
			+ CDM0[1][Mh_AN][kl ][i][j]*Hx0[1][i][j].r
			- iDM0[1][Mh_AN][kl ][i][j]*Hx0[1][i][j].i
		    + 2.0*CDM0[2][Mh_AN][kl ][i][j]*Hx0[2][i][j].r
		    - 2.0*CDM0[3][Mh_AN][kl ][i][j]*Hx0[2][i][j].i;

		      dEy += 
			  CDM0[0][Mh_AN][kl ][i][j]*Hy0[0][i][j].r
			- iDM0[0][Mh_AN][kl ][i][j]*Hy0[0][i][j].i
			+ CDM0[1][Mh_AN][kl ][i][j]*Hy0[1][i][j].r
			- iDM0[1][Mh_AN][kl ][i][j]*Hy0[1][i][j].i
	 	    + 2.0*CDM0[2][Mh_AN][kl ][i][j]*Hy0[2][i][j].r
		    - 2.0*CDM0[3][Mh_AN][kl ][i][j]*Hy0[2][i][j].i; 

		      dEz +=
			  CDM0[0][Mh_AN][kl ][i][j]*Hz0[0][i][j].r
			- iDM0[0][Mh_AN][kl ][i][j]*Hz0[0][i][j].i
			+ CDM0[1][Mh_AN][kl ][i][j]*Hz0[1][i][j].r
			- iDM0[1][Mh_AN][kl ][i][j]*Hz0[1][i][j].i
	 	    + 2.0*CDM0[2][Mh_AN][kl ][i][j]*Hz0[2][i][j].r
		    - 2.0*CDM0[3][Mh_AN][kl ][i][j]*Hz0[2][i][j].i;

		    } /* j */
		  } /* i */

                  /* for stress */ 

		  for (m=1; m<=3; m++){
		    for (n=1; n<=3; n++){
		      for (i=0; i<Spe_Total_CNO[Hwan]; i++){
			for (j=0; j<Spe_Total_CNO[Qwan]; j++){

			  My_Stress[(m-1)*3+(n-1)] +=
			          CDM0[0][Mh_AN][kl][i][j]*H_sts[0][m][n][i][j].r
			        - iDM0[0][Mh_AN][kl][i][j]*H_sts[0][m][n][i][j].i
			        + CDM0[1][Mh_AN][kl][i][j]*H_sts[1][m][n][i][j].r
			        - iDM0[1][Mh_AN][kl][i][j]*H_sts[1][m][n][i][j].i
			    + 2.0*CDM0[2][Mh_AN][kl][i][j]*H_sts[2][m][n][i][j].r
			    - 2.0*CDM0[3][Mh_AN][kl][i][j]*H_sts[2][m][n][i][j].i;
			}
		      }
		    }
		  }

                  /* call dHNL */

		  dHNL(0,Mc_AN,q_AN,h_AN,DS_NL,Hx1,Hy1,Hz1,H_sts);
		  kl1 = RMI1[Mc_AN][q_AN][h_AN];

                  /* for forces */ 

		  for (i=0; i<Spe_Total_CNO[Qwan]; i++){ /* Qwan */
		    for (j=0; j<Spe_Total_CNO[Hwan]; j++){ /* Hwan */

		      dEx += 
			  CDM0[0][Mq_AN][kl1][i][j]*Hx1[0][i][j].r
			- iDM0[0][Mq_AN][kl1][i][j]*Hx1[0][i][j].i
			+ CDM0[1][Mq_AN][kl1][i][j]*Hx1[1][i][j].r
			- iDM0[1][Mq_AN][kl1][i][j]*Hx1[1][i][j].i
		    + 2.0*CDM0[2][Mq_AN][kl1][i][j]*Hx1[2][i][j].r
		    - 2.0*CDM0[3][Mq_AN][kl1][i][j]*Hx1[2][i][j].i;

		      dEy += 
			  CDM0[0][Mq_AN][kl1][i][j]*Hy1[0][i][j].r
			- iDM0[0][Mq_AN][kl1][i][j]*Hy1[0][i][j].i
			+ CDM0[1][Mq_AN][kl1][i][j]*Hy1[1][i][j].r
			- iDM0[1][Mq_AN][kl1][i][j]*Hy1[1][i][j].i
		    + 2.0*CDM0[2][Mq_AN][kl1][i][j]*Hy1[2][i][j].r
		    - 2.0*CDM0[3][Mq_AN][kl1][i][j]*Hy1[2][i][j].i; 

		      dEz +=
			  CDM0[0][Mq_AN][kl1][i][j]*Hz1[0][i][j].r
			- iDM0[0][Mq_AN][kl1][i][j]*Hz1[0][i][j].i
			+ CDM0[1][Mq_AN][kl1][i][j]*Hz1[1][i][j].r
			- iDM0[1][Mq_AN][kl1][i][j]*Hz1[1][i][j].i
		    + 2.0*CDM0[2][Mq_AN][kl1][i][j]*Hz1[2][i][j].r
		    - 2.0*CDM0[3][Mq_AN][kl1][i][j]*Hz1[2][i][j].i;

		    } /* j */
		  } /* i */

                  /* for stress */ 

		  for (m=1; m<=3; m++){
		    for (n=1; n<=3; n++){
		      for (i=0; i<Spe_Total_CNO[Qwan]; i++){ /* Qwan */
			for (j=0; j<Spe_Total_CNO[Hwan]; j++){ /* Hwan */

			  My_Stress[(m-1)*3+(n-1)] +=
			          CDM0[0][Mq_AN][kl1][i][j]*H_sts[0][m][n][i][j].r
			        - iDM0[0][Mq_AN][kl1][i][j]*H_sts[0][m][n][i][j].i
			        + CDM0[1][Mq_AN][kl1][i][j]*H_sts[1][m][n][i][j].r
			        - iDM0[1][Mq_AN][kl1][i][j]*H_sts[1][m][n][i][j].i
			    + 2.0*CDM0[2][Mq_AN][kl1][i][j]*H_sts[2][m][n][i][j].r
			    - 2.0*CDM0[3][Mq_AN][kl1][i][j]*H_sts[2][m][n][i][j].i;

			} /* j */
		      } /* i */
		    }
		  }

                }
	      }

	    } /* if (po2==1) */
	  } /* j0 */             

	  /* force from HNL */
	  /*
	  Gxyz[Gc_AN][41] += dEx;
	  Gxyz[Gc_AN][42] += dEy;
	  Gxyz[Gc_AN][43] += dEz;
	  */

	  /* timing */
	  dtime(&Etime_atom);
	  time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

	} /* Mc_AN */

	/********************************************
          increment of F_Rcv_Num_WK[IDR] 
	********************************************/

	F_Rcv_Num_WK[IDR]++;

      } /* if ( 0<(F_Snd_Num[IDS]-F_Snd_Num_WK[IDS]) ) */

      if ( 0<(F_Snd_Num[IDS]-F_Snd_Num_WK[IDS]) ) {

	MPI_Wait(&request,&stat);
	free(tmp_array);  /* freeing of array */

	/********************************************
             increment of F_Snd_Num_WK[IDS]
	********************************************/

	F_Snd_Num_WK[IDS]++;
      } 

    } /* ID */

    /*****************************************************
      check whether all the communications have finished
    *****************************************************/

    po = 0;
    for (ID=0; ID<numprocs; ID++){

      IDS = (myid + ID) % numprocs;
      IDR = (myid - ID + numprocs) % numprocs;

      if ( 0<(F_Snd_Num[IDS]-F_Snd_Num_WK[IDS]) ) po += F_Snd_Num[IDS]-F_Snd_Num_WK[IDS];
      if ( 0<(F_Rcv_Num[IDR]-F_Rcv_Num_WK[IDR]) ) po += F_Rcv_Num[IDR]-F_Rcv_Num_WK[IDR];
    }

  } while (po!=0);

  for (i=0; i<3; i++){
    for (j=0; j<List_YOUSO[7]; j++){
      free(Hx0[i][j]);
    }
    free(Hx0[i]);
  }
  free(Hx0);

  for (i=0; i<3; i++){
    for (j=0; j<List_YOUSO[7]; j++){
      free(Hy0[i][j]);
    }
    free(Hy0[i]);
  }
  free(Hy0);

  for (i=0; i<3; i++){
    for (j=0; j<List_YOUSO[7]; j++){
      free(Hz0[i][j]);
    }
    free(Hz0[i]);
  }
  free(Hz0);

  for (i=0; i<3; i++){
    for (j=0; j<List_YOUSO[7]; j++){
      free(Hx1[i][j]);
    }
    free(Hx1[i]);
  }
  free(Hx1);

  for (i=0; i<3; i++){
    for (j=0; j<List_YOUSO[7]; j++){
      free(Hy1[i][j]);
    }
    free(Hy1[i]);
  }
  free(Hy1);

  for (i=0; i<3; i++){
    for (j=0; j<List_YOUSO[7]; j++){
      free(Hz1[i][j]);
    }
    free(Hz1[i]);
  }
  free(Hz1);

  for (i=0; i<3; i++){
    for (j=0; j<4; j++){
      for (k=0; k<4; k++){
	for (m=0; m<List_YOUSO[7]; m++){
	  free(H_sts[i][j][k][m]);
	}
        free(H_sts[i][j][k]);
      }
      free(H_sts[i][j]);
    }
    free(H_sts[i]);
  }
  free(H_sts);


  dtime(&etime);
  if(myid==0 && measure_time){
    printf("Time for part1 of force_NL=%18.5f\n",etime-stime);fflush(stdout);
  } 

  /*
  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];

    if (2<=level_stdout){
      printf("<Force>  force(HNL1) myid=%2d  Mc_AN=%2d Gc_AN=%2d  %15.12f %15.12f %15.12f\n",
	     myid,Mc_AN,Gc_AN,Gxyz[Gc_AN][41],Gxyz[Gc_AN][42],Gxyz[Gc_AN][43]);fflush(stdout);
    }
  }
  */

  /*******************************************************
   *******************************************************
     THE FIRST CASE:
     multiplying overlap integrals WITHOUT COMMUNICATION

     In case of I=i or I=j 
     for d [ \sum_k <i|k>ek<k|j> ]/dRI  
   *******************************************************
   *******************************************************/

  dtime(&stime);

#pragma omp parallel shared(time_per_atom,Gxyz,CDM0,SpinP_switch,SO_switch,Hub_U_switch,F_U_flag,Constraint_NCS_switch,Zeeman_NCS_switch,Zeeman_NCO_switch,DS_NL,RMI1,FNAN,Spe_Total_CNO,WhatSpecies,F_G2M,natn,M2G,Matomnum,List_YOUSO,F_NL_flag,My_Stress_threads) private(Hx0,Hy0,Hz0,Hx1,Hy1,Hz1,H_sts,OMPID,Nthrds,Nprocs,Mc_AN,Stime_atom,Etime_atom,dEx,dEy,dEz,Gc_AN,h_AN,Gh_AN,Mh_AN,Hwan,ian,q_AN,Gq_AN,Mq_AN,Qwan,jan,kl,kl1,i,j,k,m,n,kk,pref)
  {

    /* allocation of array */

    Hx0 = (dcomplex***)malloc(sizeof(dcomplex**)*3);
    for (i=0; i<3; i++){
      Hx0[i] = (dcomplex**)malloc(sizeof(dcomplex*)*List_YOUSO[7]);
      for (j=0; j<List_YOUSO[7]; j++){
	Hx0[i][j] = (dcomplex*)malloc(sizeof(dcomplex)*List_YOUSO[7]);
      }
    }

    Hy0 = (dcomplex***)malloc(sizeof(dcomplex**)*3);
    for (i=0; i<3; i++){
      Hy0[i] = (dcomplex**)malloc(sizeof(dcomplex*)*List_YOUSO[7]);
      for (j=0; j<List_YOUSO[7]; j++){
	Hy0[i][j] = (dcomplex*)malloc(sizeof(dcomplex)*List_YOUSO[7]);
      }
    }

    Hz0 = (dcomplex***)malloc(sizeof(dcomplex**)*3);
    for (i=0; i<3; i++){
      Hz0[i] = (dcomplex**)malloc(sizeof(dcomplex*)*List_YOUSO[7]);
      for (j=0; j<List_YOUSO[7]; j++){
	Hz0[i][j] = (dcomplex*)malloc(sizeof(dcomplex)*List_YOUSO[7]);
      }
    }

    Hx1 = (dcomplex***)malloc(sizeof(dcomplex**)*3);
    for (i=0; i<3; i++){
      Hx1[i] = (dcomplex**)malloc(sizeof(dcomplex*)*List_YOUSO[7]);
      for (j=0; j<List_YOUSO[7]; j++){
	Hx1[i][j] = (dcomplex*)malloc(sizeof(dcomplex)*List_YOUSO[7]);
      }
    }

    Hy1 = (dcomplex***)malloc(sizeof(dcomplex**)*3);
    for (i=0; i<3; i++){
      Hy1[i] = (dcomplex**)malloc(sizeof(dcomplex*)*List_YOUSO[7]);
      for (j=0; j<List_YOUSO[7]; j++){
	Hy1[i][j] = (dcomplex*)malloc(sizeof(dcomplex)*List_YOUSO[7]);
      }
    }

    Hz1 = (dcomplex***)malloc(sizeof(dcomplex**)*3);
    for (i=0; i<3; i++){
      Hz1[i] = (dcomplex**)malloc(sizeof(dcomplex*)*List_YOUSO[7]);
      for (j=0; j<List_YOUSO[7]; j++){
	Hz1[i][j] = (dcomplex*)malloc(sizeof(dcomplex)*List_YOUSO[7]);
      }
    }

    H_sts = (dcomplex*****)malloc(sizeof(dcomplex****)*3);
    for (i=0; i<3; i++){
      H_sts[i] = (dcomplex****)malloc(sizeof(dcomplex***)*4);
      for (j=0; j<4; j++){
	H_sts[i][j] = (dcomplex***)malloc(sizeof(dcomplex**)*4);
	for (k=0; k<4; k++){
	  H_sts[i][j][k] = (dcomplex**)malloc(sizeof(dcomplex*)*List_YOUSO[7]);
	  for (m=0; m<List_YOUSO[7]; m++){
	    H_sts[i][j][k][m] = (dcomplex*)malloc(sizeof(dcomplex)*List_YOUSO[7]);
	  }
	}
      }
    }

    /* get info. on OpenMP */ 

    OMPID = omp_get_thread_num();
    Nthrds = omp_get_num_threads();
    Nprocs = omp_get_num_procs();

    for (Mc_AN=(OMPID*Matomnum/Nthrds+1); Mc_AN<((OMPID+1)*Matomnum/Nthrds+1); Mc_AN++){

      dtime(&Stime_atom);

      dEx = 0.0; 
      dEy = 0.0; 
      dEz = 0.0; 

      Gc_AN = M2G[Mc_AN];
      h_AN = 0;
      Gh_AN = natn[Gc_AN][h_AN];
      Mh_AN = F_G2M[Gh_AN];
      Hwan = WhatSpecies[Gh_AN];
      ian = Spe_Total_CNO[Hwan];

      for (q_AN=0; q_AN<=FNAN[Gc_AN]; q_AN++){

	Gq_AN = natn[Gc_AN][q_AN];
	Mq_AN = F_G2M[Gq_AN];

	if (Mq_AN<=Matomnum){

	  Qwan = WhatSpecies[Gq_AN];
	  jan = Spe_Total_CNO[Qwan];
	  kl = RMI1[Mc_AN][h_AN][q_AN];

          dHNL(0,Mc_AN,h_AN,q_AN,DS_NL,Hx0,Hy0,Hz0,H_sts);

	  if (SpinP_switch==0){

            if (q_AN==h_AN) pref = 2.0;
            else            pref = 4.0; 

            /* for forces */                 

	    for (i=0; i<ian; i++){
	      for (j=0; j<jan; j++){
		dEx += pref*CDM0[0][Mh_AN][kl][i][j]*Hx0[0][i][j].r;
		dEy += pref*CDM0[0][Mh_AN][kl][i][j]*Hy0[0][i][j].r;
		dEz += pref*CDM0[0][Mh_AN][kl][i][j]*Hz0[0][i][j].r;
	      }
	    }

	    /* for stress */                 

	    for (m=1; m<=3; m++){
	      for (n=1; n<=3; n++){
		for (i=0; i<ian; i++){
		  for (j=0; j<jan; j++){
		    My_Stress_threads[OMPID][(m-1)*3+(n-1)] += pref*CDM0[0][Mh_AN][kl][i][j]*H_sts[0][m][n][i][j].r;
		  }
		}
	      }
	    }

	  }

          /* collinear spin polarized or non-colliear without SO and LDA+U */

	  else if (SpinP_switch==1 || (SpinP_switch==3 && SO_switch==0 && Hub_U_switch==0
	        && Constraint_NCS_switch==0 && Zeeman_NCS_switch==0 && Zeeman_NCO_switch==0)){ 

	    if (q_AN==h_AN) pref = 1.0;
	    else            pref = 2.0; 

            /* for forces */                 

	    for (i=0; i<ian; i++){
	      for (j=0; j<jan; j++){

		dEx += pref*(  CDM0[0][Mh_AN][kl][i][j]*Hx0[0][i][j].r
			     + CDM0[1][Mh_AN][kl][i][j]*Hx0[1][i][j].r);
		dEy += pref*(  CDM0[0][Mh_AN][kl][i][j]*Hy0[0][i][j].r
			     + CDM0[1][Mh_AN][kl][i][j]*Hy0[1][i][j].r);
		dEz += pref*(  CDM0[0][Mh_AN][kl][i][j]*Hz0[0][i][j].r
			     + CDM0[1][Mh_AN][kl][i][j]*Hz0[1][i][j].r);
	      }
	    }

	    /* for stress */                 

	    for (m=1; m<=3; m++){
	      for (n=1; n<=3; n++){
		for (i=0; i<ian; i++){
		  for (j=0; j<jan; j++){
		    My_Stress_threads[OMPID][(m-1)*3+(n-1)] += pref*( CDM0[0][Mh_AN][kl][i][j]*H_sts[0][m][n][i][j].r
	 			                                    + CDM0[1][Mh_AN][kl][i][j]*H_sts[1][m][n][i][j].r);
		  }
		}
	      }
	    }
	  }

	  /* spin non-collinear with spin-orbit coupling or with LDA+U */

	  else if ( SpinP_switch==3 && (SO_switch==1 || (Hub_U_switch==1 && F_U_flag==1) 
		|| 1<=Constraint_NCS_switch || Zeeman_NCS_switch==1 || Zeeman_NCO_switch==1)){

            if (q_AN==h_AN){

              /* for forces */ 

	      for (i=0; i<Spe_Total_CNO[Hwan]; i++){
		for (j=0; j<Spe_Total_CNO[Qwan]; j++){

		  dEx += 
                      CDM0[0][Mh_AN][kl][i][j]*Hx0[0][i][j].r
		    - iDM0[0][Mh_AN][kl][i][j]*Hx0[0][i][j].i
		    + CDM0[1][Mh_AN][kl][i][j]*Hx0[1][i][j].r
		    - iDM0[1][Mh_AN][kl][i][j]*Hx0[1][i][j].i
	        + 2.0*CDM0[2][Mh_AN][kl][i][j]*Hx0[2][i][j].r
		- 2.0*CDM0[3][Mh_AN][kl][i][j]*Hx0[2][i][j].i;

		  dEy += 
                      CDM0[0][Mh_AN][kl][i][j]*Hy0[0][i][j].r
		    - iDM0[0][Mh_AN][kl][i][j]*Hy0[0][i][j].i
		    + CDM0[1][Mh_AN][kl][i][j]*Hy0[1][i][j].r
		    - iDM0[1][Mh_AN][kl][i][j]*Hy0[1][i][j].i
		+ 2.0*CDM0[2][Mh_AN][kl][i][j]*Hy0[2][i][j].r
	        - 2.0*CDM0[3][Mh_AN][kl][i][j]*Hy0[2][i][j].i; 

		  dEz +=
                      CDM0[0][Mh_AN][kl][i][j]*Hz0[0][i][j].r
		    - iDM0[0][Mh_AN][kl][i][j]*Hz0[0][i][j].i
		    + CDM0[1][Mh_AN][kl][i][j]*Hz0[1][i][j].r
		    - iDM0[1][Mh_AN][kl][i][j]*Hz0[1][i][j].i
	        + 2.0*CDM0[2][Mh_AN][kl][i][j]*Hz0[2][i][j].r
		- 2.0*CDM0[3][Mh_AN][kl][i][j]*Hz0[2][i][j].i;

		}
	      }

	      /* for stress */ 

	      for (m=1; m<=3; m++){
		for (n=1; n<=3; n++){
		  for (i=0; i<Spe_Total_CNO[Hwan]; i++){
		    for (j=0; j<Spe_Total_CNO[Qwan]; j++){

		      My_Stress_threads[OMPID][(m-1)*3+(n-1)] +=
		   	      CDM0[0][Mh_AN][kl][i][j]*H_sts[0][m][n][i][j].r
			    - iDM0[0][Mh_AN][kl][i][j]*H_sts[0][m][n][i][j].i
		  	    + CDM0[1][Mh_AN][kl][i][j]*H_sts[1][m][n][i][j].r
			    - iDM0[1][Mh_AN][kl][i][j]*H_sts[1][m][n][i][j].i
			+ 2.0*CDM0[2][Mh_AN][kl][i][j]*H_sts[2][m][n][i][j].r
			- 2.0*CDM0[3][Mh_AN][kl][i][j]*H_sts[2][m][n][i][j].i;
		    }
		  }
		}
	      }

            } /* if (q_AN==h_AN) */

            else{

              /* for forces */ 

              for (i=0; i<Spe_Total_CNO[Hwan]; i++){  /* Hwan */
		for (j=0; j<Spe_Total_CNO[Qwan]; j++){ /* Qwan  */

		  dEx += 
                      CDM0[0][Mh_AN][kl ][i][j]*Hx0[0][i][j].r
		    - iDM0[0][Mh_AN][kl ][i][j]*Hx0[0][i][j].i
		    + CDM0[1][Mh_AN][kl ][i][j]*Hx0[1][i][j].r
		    - iDM0[1][Mh_AN][kl ][i][j]*Hx0[1][i][j].i
		+ 2.0*CDM0[2][Mh_AN][kl ][i][j]*Hx0[2][i][j].r
                - 2.0*CDM0[3][Mh_AN][kl ][i][j]*Hx0[2][i][j].i;

		  dEy += 
                      CDM0[0][Mh_AN][kl ][i][j]*Hy0[0][i][j].r
		    - iDM0[0][Mh_AN][kl ][i][j]*Hy0[0][i][j].i
		    + CDM0[1][Mh_AN][kl ][i][j]*Hy0[1][i][j].r
		    - iDM0[1][Mh_AN][kl ][i][j]*Hy0[1][i][j].i
		+ 2.0*CDM0[2][Mh_AN][kl ][i][j]*Hy0[2][i][j].r
                - 2.0*CDM0[3][Mh_AN][kl ][i][j]*Hy0[2][i][j].i; 

		  dEz +=
                      CDM0[0][Mh_AN][kl ][i][j]*Hz0[0][i][j].r
		    - iDM0[0][Mh_AN][kl ][i][j]*Hz0[0][i][j].i
		    + CDM0[1][Mh_AN][kl ][i][j]*Hz0[1][i][j].r
		    - iDM0[1][Mh_AN][kl ][i][j]*Hz0[1][i][j].i
	        + 2.0*CDM0[2][Mh_AN][kl ][i][j]*Hz0[2][i][j].r
                - 2.0*CDM0[3][Mh_AN][kl ][i][j]*Hz0[2][i][j].i;

		} /* j */
	      } /* i */

	      /* for stress */ 

	      for (m=1; m<=3; m++){
		for (n=1; n<=3; n++){
		  for (i=0; i<Spe_Total_CNO[Hwan]; i++){
		    for (j=0; j<Spe_Total_CNO[Qwan]; j++){

		      My_Stress_threads[OMPID][(m-1)*3+(n-1)] +=
			      CDM0[0][Mh_AN][kl][i][j]*H_sts[0][m][n][i][j].r
		  	    - iDM0[0][Mh_AN][kl][i][j]*H_sts[0][m][n][i][j].i
		 	    + CDM0[1][Mh_AN][kl][i][j]*H_sts[1][m][n][i][j].r
			    - iDM0[1][Mh_AN][kl][i][j]*H_sts[1][m][n][i][j].i
			+ 2.0*CDM0[2][Mh_AN][kl][i][j]*H_sts[2][m][n][i][j].r
			- 2.0*CDM0[3][Mh_AN][kl][i][j]*H_sts[2][m][n][i][j].i;
		    }
		  }
		}
	      }

              /* call dHNL */

              dHNL(0,Mc_AN,q_AN,h_AN,DS_NL,Hx1,Hy1,Hz1,H_sts);
       	      kl1 = RMI1[Mc_AN][q_AN][h_AN];

              /* for forces */ 

	      for (i=0; i<Spe_Total_CNO[Qwan]; i++){ /* Qwan */
		for (j=0; j<Spe_Total_CNO[Hwan]; j++){ /* Hwan */

		  dEx += 
		      CDM0[0][Mq_AN][kl1][i][j]*Hx1[0][i][j].r
		    - iDM0[0][Mq_AN][kl1][i][j]*Hx1[0][i][j].i
		    + CDM0[1][Mq_AN][kl1][i][j]*Hx1[1][i][j].r
		    - iDM0[1][Mq_AN][kl1][i][j]*Hx1[1][i][j].i
		+ 2.0*CDM0[2][Mq_AN][kl1][i][j]*Hx1[2][i][j].r
		- 2.0*CDM0[3][Mq_AN][kl1][i][j]*Hx1[2][i][j].i;

		  dEy += 
                      CDM0[0][Mq_AN][kl1][i][j]*Hy1[0][i][j].r
		    - iDM0[0][Mq_AN][kl1][i][j]*Hy1[0][i][j].i
		    + CDM0[1][Mq_AN][kl1][i][j]*Hy1[1][i][j].r
		    - iDM0[1][Mq_AN][kl1][i][j]*Hy1[1][i][j].i
	        + 2.0*CDM0[2][Mq_AN][kl1][i][j]*Hy1[2][i][j].r
	        - 2.0*CDM0[3][Mq_AN][kl1][i][j]*Hy1[2][i][j].i; 

		  dEz +=
                      CDM0[0][Mq_AN][kl1][i][j]*Hz1[0][i][j].r
		    - iDM0[0][Mq_AN][kl1][i][j]*Hz1[0][i][j].i
		    + CDM0[1][Mq_AN][kl1][i][j]*Hz1[1][i][j].r
		    - iDM0[1][Mq_AN][kl1][i][j]*Hz1[1][i][j].i
	        + 2.0*CDM0[2][Mq_AN][kl1][i][j]*Hz1[2][i][j].r
		- 2.0*CDM0[3][Mq_AN][kl1][i][j]*Hz1[2][i][j].i;

		} /* j */
	      } /* i */

	      /* for stress */ 

	      for (m=1; m<=3; m++){
		for (n=1; n<=3; n++){
		  for (i=0; i<Spe_Total_CNO[Qwan]; i++){ /* Qwan */
		    for (j=0; j<Spe_Total_CNO[Hwan]; j++){ /* Hwan */

		      My_Stress_threads[OMPID][(m-1)*3+(n-1)] +=
			      CDM0[0][Mq_AN][kl1][i][j]*H_sts[0][m][n][i][j].r
			    - iDM0[0][Mq_AN][kl1][i][j]*H_sts[0][m][n][i][j].i
			    + CDM0[1][Mq_AN][kl1][i][j]*H_sts[1][m][n][i][j].r
			    - iDM0[1][Mq_AN][kl1][i][j]*H_sts[1][m][n][i][j].i
			+ 2.0*CDM0[2][Mq_AN][kl1][i][j]*H_sts[2][m][n][i][j].r
			- 2.0*CDM0[3][Mq_AN][kl1][i][j]*H_sts[2][m][n][i][j].i;

		    } /* j */
		  } /* i */
		}
	      }

	    } 
	  }
	}
      }

      /* force from #4B */
      /*
      if (F_NL_flag==1){
        Gxyz[Gc_AN][41] += dEx;
        Gxyz[Gc_AN][42] += dEy;
        Gxyz[Gc_AN][43] += dEz;
      }
      */

      /* timing */
      dtime(&Etime_atom);
      time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

    } /* Mc_AN */

    /* freeing of array */

    for (i=0; i<3; i++){
      for (j=0; j<List_YOUSO[7]; j++){
	free(Hx0[i][j]);
      }
      free(Hx0[i]);
    }
    free(Hx0);

    for (i=0; i<3; i++){
      for (j=0; j<List_YOUSO[7]; j++){
	free(Hy0[i][j]);
      }
      free(Hy0[i]);
    }
    free(Hy0);

    for (i=0; i<3; i++){
      for (j=0; j<List_YOUSO[7]; j++){
	free(Hz0[i][j]);
      }
      free(Hz0[i]);
    }
    free(Hz0);

    for (i=0; i<3; i++){
      for (j=0; j<List_YOUSO[7]; j++){
	free(Hx1[i][j]);
      }
      free(Hx1[i]);
    }
    free(Hx1);

    for (i=0; i<3; i++){
      for (j=0; j<List_YOUSO[7]; j++){
	free(Hy1[i][j]);
      }
      free(Hy1[i]);
    }
    free(Hy1);

    for (i=0; i<3; i++){
      for (j=0; j<List_YOUSO[7]; j++){
	free(Hz1[i][j]);
      }
      free(Hz1[i]);
    }
    free(Hz1);

    for (i=0; i<3; i++){
      for (j=0; j<4; j++){
	for (k=0; k<4; k++){
	  for (m=0; m<List_YOUSO[7]; m++){
	    free(H_sts[i][j][k][m]);
	  }
	  free(H_sts[i][j][k]);
	}
	free(H_sts[i][j]);
      }
      free(H_sts[i]);
    }
    free(H_sts);

  } /* #pragma omp parallel */

  /* summing up results calculated by OpenMP threads */

  for(i=0; i<Nthrds0; i++){
    for(j=0; j<9; j++){
      My_Stress[j] += My_Stress_threads[i][j];
    }
  }

  /* MPI_Allreduce: My_Stress */

  MPI_Allreduce(My_Stress, Stress, 9, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

  for (i=0; i<9; i++){
    Stress[i] = Stress[i]*(double)F_NL_flag; 
  }

  /* add stress coming from HNL to Stress_Tensor */

  for (i=0; i<9; i++){
    Stress_Tensor[i] += Stress[i];
  }

  /* show stress coming from HNL */

  if (myid==Host_ID && 1<level_stdout){

    if (myid==Host_ID){
      printf(" Components of stress HNL: \n");fflush(stdout);
      for (i=0; i<3; i++){
	for (j=0; j<3; j++){
	  printf("%16.8f ", Stress[3*i+j]);
	}
	printf("\n");fflush(stdout);
      }
    }
  }

  /* measuring time */

  dtime(&etime);
  if(myid==0 && measure_time){
    printf("Time for part2 of force_NL=%18.5f\n",etime-stime);fflush(stdout);
  } 

  /*
  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];

    if (2<=level_stdout){
      printf("<Force>  force(HNL2) myid=%2d  Mc_AN=%2d Gc_AN=%2d  %15.12f %15.12f %15.12f\n",
	     myid,Mc_AN,Gc_AN,Gxyz[Gc_AN][41],Gxyz[Gc_AN][42],Gxyz[Gc_AN][43]);fflush(stdout);
    }
  }
  */

  /***********************************
            freeing of arrays 
  ************************************/

  free(Indicator);

  for (ID=0; ID<numprocs; ID++){
    free(S_array[ID]);
  }
  free(S_array);

  for (ID=0; ID<numprocs; ID++){
    free(R_array[ID]);
  }
  free(R_array);

  free(Snd_DS_NL_Size);
  free(Rcv_DS_NL_Size);

  for (i=0; i<Nthrds0; i++){
    free(My_Stress_threads[i]);
  }
  free(My_Stress_threads);

}







void Stress_plus_U_dual(double *****CDM0)
{
  int i,j;
  int numprocs,myid;
  int Nthrds0;
  double **My_Stress_threads;
  double Stress[9],My_Stress[9];

  MPI_Status stat;
  MPI_Request request;

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  if (myid==Host_ID && 0<level_stdout){
    printf("  Stress calculation: DFT+U\n");fflush(stdout);
  }

  /* get Nthrds0 */

#pragma omp parallel shared(Nthrds0)
  {
    Nthrds0 = omp_get_num_threads();
  }

  /* allocation of array */

  My_Stress_threads=(double**)malloc(sizeof(double*)*(Nthrds0));
  for (i=0; i<Nthrds0; i++){
    My_Stress_threads[i] = (double*)malloc(sizeof(double)*9);
    for (j=0; j<9; j++) My_Stress_threads[i][j] = 0.0;
  }

  /* OpenMP threading for Mc_AN loop */

#pragma omp parallel shared(List_YOUSO,Matomnum,M2G,WhatSpecies,natn,ncn,F_G2M,Spe_Total_NO,Gxyz,atv,FNAN,RMI1,v_eff,OLP,SpinP_switch,CDM0,My_Stress_threads)
  {
    int i,j,k,Mc_AN,so,Rn,Gc_AN,Cwan;
    int spin,Gh_AN,Mh_AN,Hwan,h_AN,spinmax;
    int OMPID,Nthrds,Nprocs;
    double tmp,tmp1,tmp2,tmp3;
    double dEx,dEy,dEz,dege;
    double lx,ly,lz;

    /* get info. on OpenMP */ 

    OMPID = omp_get_thread_num();
    Nthrds = omp_get_num_threads();
    Nprocs = omp_get_num_procs();

    if (SpinP_switch==0){
      spinmax = 0;
      dege = 2.0;
    }
    else{
      spinmax = 1;
      dege = 1.0;
    }

    /* loop for Mc_AN */ 

    for (Mc_AN=(OMPID*Matomnum/Nthrds+1); Mc_AN<((OMPID+1)*Matomnum/Nthrds+1); Mc_AN++){

      Gc_AN = M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];

      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	Gh_AN = natn[Gc_AN][h_AN];
	Mh_AN = F_G2M[Gh_AN];
	Hwan = WhatSpecies[Gh_AN];

	Rn = ncn[Gc_AN][h_AN];
	lx = Gxyz[Gc_AN][1] - Gxyz[Gh_AN][1] - atv[Rn][1];
	ly = Gxyz[Gc_AN][2] - Gxyz[Gh_AN][2] - atv[Rn][2];
	lz = Gxyz[Gc_AN][3] - Gxyz[Gh_AN][3] - atv[Rn][3];

        dEx = 0.0;
        dEy = 0.0;
        dEz = 0.0;

        for (spin=0; spin<=spinmax; spin++){
	  for (i=0; i<Spe_Total_NO[Cwan]; i++){
	    for (j=0; j<Spe_Total_NO[Hwan]; j++){

	      tmp1 = 0.0; 
	      tmp2 = 0.0; 
	      tmp3 = 0.0; 

	      for (k=0; k<Spe_Total_NO[Cwan]; k++){
		tmp = v_eff[spin][Mc_AN][i][k];
		tmp1 += tmp*OLP[1][Mc_AN][h_AN][k][j];
		tmp2 += tmp*OLP[2][Mc_AN][h_AN][k][j];
		tmp3 += tmp*OLP[3][Mc_AN][h_AN][k][j];
	      }

	      for (k=0; k<Spe_Total_NO[Hwan]; k++){
		tmp = v_eff[spin][Mh_AN][k][j];
		tmp1 += tmp*OLP[1][Mc_AN][h_AN][i][k];
		tmp2 += tmp*OLP[2][Mc_AN][h_AN][i][k];
		tmp3 += tmp*OLP[3][Mc_AN][h_AN][i][k];
	      }

	      dEx += dege*tmp1*CDM0[spin][Mc_AN][h_AN][i][j];
	      dEy += dege*tmp2*CDM0[spin][Mc_AN][h_AN][i][j];
	      dEz += dege*tmp3*CDM0[spin][Mc_AN][h_AN][i][j];
	    }
	  }
	} /* spin */

	My_Stress_threads[OMPID][0] += 0.5*dEx*lx;
	My_Stress_threads[OMPID][1] += 0.5*dEy*lx;
	My_Stress_threads[OMPID][2] += 0.5*dEz*lx;
	My_Stress_threads[OMPID][3] += 0.5*dEx*ly;
	My_Stress_threads[OMPID][4] += 0.5*dEy*ly;
	My_Stress_threads[OMPID][5] += 0.5*dEz*ly;
	My_Stress_threads[OMPID][6] += 0.5*dEx*lz;
	My_Stress_threads[OMPID][7] += 0.5*dEy*lz;
	My_Stress_threads[OMPID][8] += 0.5*dEz*lz;

      } /* h_AN */
    } /* Mc_AN */

  } /* #pragma omp parallel */

  /* sum up stress */

  for(i=0; i<9; i++){
    My_Stress[i] = 0.0;
  }

  for(i=0; i<Nthrds0; i++){
    for(j=0; j<9; j++){
      My_Stress[j] += My_Stress_threads[i][j];
    }
  }

  MPI_Allreduce(&My_Stress[0],&Stress[0],9,MPI_DOUBLE,MPI_SUM,mpi_comm_level1);

  /* if (F_U_flag==0) */

  if (F_U_flag==0){
    for (i=0; i<9; i++) Stress[i] = 0.0;
  }

  /* add DFT-plus-U components of stress to Stress_Tensor */

  for (i=0; i<9; i++){
    Stress_Tensor[i] += Stress[i];
  }

  /* show Kinetic components of stress */
  
  if (myid==Host_ID && 1<level_stdout){
    printf(" Components of stress (DFT+U, dual) : \n");fflush(stdout);
    for (i = 0; i<3; i++){
      for (j = 0; j<3; j++){
	printf("%16.8f ", Stress[3*i+j]);
      }
      printf("\n");fflush(stdout);
    }
  }

  /* freeing of array */

  for (i=0; i<Nthrds0; i++){
    free(My_Stress_threads[i]);
  }
  free(My_Stress_threads);

}




void dHVNA(int where_flag, int Mc_AN, int h_AN, int q_AN,
           Type_DS_VNA *****DS_VNA1, 
           double *****TmpHVNA2, double *****TmpHVNA3, 
           double **Hx, double **Hy, double **Hz, 
           double ****H_sts)
{
  int i,j,k,m,n,l,kg,kan,so,deri_kind;
  int ig,ian,jg,jan,kl,kl1,kl2,Rni,Rnj,Rnk;
  int wakg,l1,l2,l3,Gc_AN,Mi_AN,Mj_AN,Mj_AN2,num_projectors;
  double sumx,sumy,sumz,ene,rcuti,rcutj,rcut;
  double tmpx,tmpy,tmpz,dmp,deri_dmp,tmp;
  double dx,dy,dz,x0,y0,z0,x1,y1,z1,r;
  double PFp,PFm,ene_p,ene_m;
  double sumx0,sumy0,sumz0;
  double sumx1,sumy1,sumz1;
  double sumx2,sumy2,sumz2;
  double tx,ty,tz;
  int L,LL,Mul1,Num_RVNA;

  Num_RVNA = List_YOUSO[34]*(List_YOUSO[35] + 1);
  num_projectors = (List_YOUSO[35]+1)*(List_YOUSO[35]+1)*List_YOUSO[34];

  /****************************************************
   start calc.
  ****************************************************/

  Gc_AN = M2G[Mc_AN];
  ig = natn[Gc_AN][h_AN];
  Rni = ncn[Gc_AN][h_AN];
  Mi_AN = F_G2M[ig];
  ian = Spe_Total_CNO[WhatSpecies[ig]];
  rcuti = Spe_Atom_Cut1[WhatSpecies[ig]];

  jg = natn[Gc_AN][q_AN];
  Rnj = ncn[Gc_AN][q_AN];
  Mj_AN = F_G2M[jg];
  jan = Spe_Total_CNO[WhatSpecies[jg]];
  rcutj = Spe_Atom_Cut1[WhatSpecies[jg]];

  rcut = rcuti + rcutj;
  kl = RMI1[Mc_AN][h_AN][q_AN];
  dmp = dampingF(rcut,Dis[ig][kl]);

  for (m=0; m<ian; m++){
    for (n=0; n<jan; n++){
      Hx[m][n] = 0.0;
      Hy[m][n] = 0.0;
      Hz[m][n] = 0.0;
    }
  }

  for (i=0; i<4; i++){
    for (j=0; j<4; j++){
      for (m=0; m<ian; m++){
	for (n=0; n<jan; n++){
	  H_sts[i][j][m][n] = 0.0;
	}
      }
    }
  }

  /****************************************************
    two-center integral with orbitals on one-center 
    
    in case of h_AN==0 && q_AN==0
  ****************************************************/

  if (h_AN==0 && q_AN==0 && where_flag==0){

    for (k=1; k<=FNAN[Gc_AN]; k++){

      kg = natn[Gc_AN][k];
      Rnk = ncn[Gc_AN][k];

      tx = Gxyz[Gc_AN][1] - Gxyz[kg][1] - atv[Rnk][1];
      ty = Gxyz[Gc_AN][2] - Gxyz[kg][2] - atv[Rnk][2];
      tz = Gxyz[Gc_AN][3] - Gxyz[kg][3] - atv[Rnk][3];

      /* for force */ 

      for (m=0; m<ian; m++){
        for (n=0; n<jan; n++){
	  Hx[m][n] += TmpHVNA2[1][Mc_AN][k][m][n];
	  Hy[m][n] += TmpHVNA2[2][Mc_AN][k][m][n];
	  Hz[m][n] += TmpHVNA2[3][Mc_AN][k][m][n];
        }
      }      

      /* for stress calculation */ 

      for (m=0; m<ian; m++){
        for (n=0; n<jan; n++){

	  H_sts[1][1][m][n] += TmpHVNA2[1][Mc_AN][k][m][n]*tx;
	  H_sts[1][2][m][n] += TmpHVNA2[1][Mc_AN][k][m][n]*ty;
	  H_sts[1][3][m][n] += TmpHVNA2[1][Mc_AN][k][m][n]*tz;

	  H_sts[2][1][m][n] += TmpHVNA2[2][Mc_AN][k][m][n]*tx;
	  H_sts[2][2][m][n] += TmpHVNA2[2][Mc_AN][k][m][n]*ty;
	  H_sts[2][3][m][n] += TmpHVNA2[2][Mc_AN][k][m][n]*tz;

	  H_sts[3][1][m][n] += TmpHVNA2[3][Mc_AN][k][m][n]*tx;
	  H_sts[3][2][m][n] += TmpHVNA2[3][Mc_AN][k][m][n]*ty;
	  H_sts[3][3][m][n] += TmpHVNA2[3][Mc_AN][k][m][n]*tz;
        }
      }    
    } /* k */
  }

  /****************************************************
    two-center integral with orbitals on one-center 
    
    in case of h_AN==q_AN && h_AN!=0 
  ****************************************************/

  else if (h_AN==q_AN && h_AN!=0){

    kl = RMI1[Mc_AN][h_AN][0];

    for (m=0; m<ian; m++){
      for (n=0; n<jan; n++){

	Hx[m][n] = -TmpHVNA3[1][Mc_AN][h_AN][m][n];
	Hy[m][n] = -TmpHVNA3[2][Mc_AN][h_AN][m][n];
	Hz[m][n] = -TmpHVNA3[3][Mc_AN][h_AN][m][n];
      }
    } 
  }

  /****************************************************
             two and three center integrals
             with orbitals on two-center
  ****************************************************/

  else{

    if (h_AN==0){

      /****************************************************
			   dH*ep*H
      ****************************************************/

      for (k=0; k<=FNAN[Gc_AN]; k++){

	kg = natn[Gc_AN][k];
        Rnk = ncn[Gc_AN][k];
	wakg = WhatSpecies[kg];
	kl = RMI1[Mc_AN][q_AN][k];

        tx = Gxyz[Gc_AN][1] - Gxyz[kg][1] - atv[Rnk][1];
        ty = Gxyz[Gc_AN][2] - Gxyz[kg][2] - atv[Rnk][2];
        tz = Gxyz[Gc_AN][3] - Gxyz[kg][3] - atv[Rnk][3];

	/****************************************************
			     non-local part
	****************************************************/

	if (0<=kl && where_flag==0){

          if (Mj_AN<=Matomnum) Mj_AN2 = Mj_AN;
          else                 Mj_AN2 = Matomnum+1; 

	  for (m=0; m<ian; m++){
	    for (n=0; n<jan; n++){

	      sumx = 0.0;
	      sumy = 0.0;
	      sumz = 0.0;

	      for (l=0; l<num_projectors; l++){
		sumx += DS_VNA1[1][Mc_AN][k][m][l]*DS_VNA1[0][Mj_AN2][kl][n][l];
		sumy += DS_VNA1[2][Mc_AN][k][m][l]*DS_VNA1[0][Mj_AN2][kl][n][l];
		sumz += DS_VNA1[3][Mc_AN][k][m][l]*DS_VNA1[0][Mj_AN2][kl][n][l];
	      }

              /* for forces */

	      Hx[m][n] += sumx;
	      Hy[m][n] += sumy;
	      Hz[m][n] += sumz;

              /* for stress calculation */ 

	      H_sts[1][1][m][n] += sumx*tx;
	      H_sts[1][2][m][n] += sumx*ty;
	      H_sts[1][3][m][n] += sumx*tz;

	      H_sts[2][1][m][n] += sumy*tx;
	      H_sts[2][2][m][n] += sumy*ty;
	      H_sts[2][3][m][n] += sumy*tz;

	      H_sts[3][1][m][n] += sumz*tx;
	      H_sts[3][2][m][n] += sumz*ty;
	      H_sts[3][3][m][n] += sumz*tz;

	    } /* n */
	  } /* m */

	} /* if */

      } /* k */

      /****************************************************
 		  	     H*ep*dH 
      ****************************************************/

      /* non-local part */

      if (q_AN==0){

        /* for forces */

	for (m=0; m<ian; m++){
	  for (n=m; n<jan; n++){
            
            tmpx = Hx[m][n] + Hx[n][m];
	    Hx[m][n] = tmpx;
	    Hx[n][m] = tmpx;

            tmpy = Hy[m][n] + Hy[n][m];
	    Hy[m][n] = tmpy;
	    Hy[n][m] = tmpy;

            tmpz = Hz[m][n] + Hz[n][m];
	    Hz[m][n] = tmpz;
	    Hz[n][m] = tmpz;
	  }
	}

        /* for stress */

	for (i=1; i<=3; i++){
	  for (j=1; j<=3; j++){
	    for (m=0; m<ian; m++){
	      for (n=m; n<jan; n++){

		tmp = H_sts[i][j][m][n] + H_sts[i][j][n][m];
		H_sts[i][j][m][n] = tmp;
		H_sts[i][j][n][m] = tmp;

		tmp = H_sts[i][j][m][n] + H_sts[i][j][n][m];
		H_sts[i][j][m][n] = tmp;
		H_sts[i][j][n][m] = tmp;
	      }
	    }
	  }
	}

      }

      else if (where_flag==1) {

	kg = natn[Gc_AN][0];
	wakg = WhatSpecies[kg];

	/****************************************************
			     non-local part
	****************************************************/

	for (m=0; m<ian; m++){
	  for (n=0; n<jan; n++){

	    sumx = 0.0;
	    sumy = 0.0;
	    sumz = 0.0;

            if (Mj_AN<=Matomnum){
              Mj_AN2 = Mj_AN;
 	      kl = RMI1[Mc_AN][q_AN][0];
	    }
            else{
              Mj_AN2 = Matomnum+1; 
	      kl = RMI1[Mc_AN][0][q_AN];
            }

	    for (l=0; l<num_projectors; l++){
	      sumx -= DS_VNA1[0][Mc_AN][0][m][l]*DS_VNA1[1][Mj_AN2][kl][n][l];
	      sumy -= DS_VNA1[0][Mc_AN][0][m][l]*DS_VNA1[2][Mj_AN2][kl][n][l];
	      sumz -= DS_VNA1[0][Mc_AN][0][m][l]*DS_VNA1[3][Mj_AN2][kl][n][l];
	    }

	    Hx[m][n] += sumx;
	    Hy[m][n] += sumy;
	    Hz[m][n] += sumz;

	  }
	}

      }

    } /* if (h_AN==0) */

    else {

      /****************************************************
			   dH*ep*H
      ****************************************************/

      kg = natn[Gc_AN][0];
      wakg = WhatSpecies[kg];
      kl1 = RMI1[Mc_AN][0][h_AN];
      kl2 = RMI1[Mc_AN][0][q_AN];

      /****************************************************
 		         non-local part
      ****************************************************/

      for (m=0; m<ian; m++){
	for (n=0; n<jan; n++){

	  sumx = 0.0;
	  sumy = 0.0;
	  sumz = 0.0;

          for (l=0; l<num_projectors; l++){
	    sumx -= DS_VNA1[1][Matomnum+1][kl1][m][l]*DS_VNA1[0][Matomnum+1][kl2][n][l];
	    sumy -= DS_VNA1[2][Matomnum+1][kl1][m][l]*DS_VNA1[0][Matomnum+1][kl2][n][l];
	    sumz -= DS_VNA1[3][Matomnum+1][kl1][m][l]*DS_VNA1[0][Matomnum+1][kl2][n][l];
	  }

	  Hx[m][n] = sumx;
	  Hy[m][n] = sumy;
	  Hz[m][n] = sumz;
	}
      }

      /****************************************************
			   H*ep*dH
      ****************************************************/

      if (q_AN!=0){

	kg = natn[Gc_AN][0];
	wakg = WhatSpecies[kg];
	kl1 = RMI1[Mc_AN][0][h_AN];
	kl2 = RMI1[Mc_AN][0][q_AN];

	/****************************************************
			    non-local part
	****************************************************/

	for (m=0; m<ian; m++){
	  for (n=0; n<jan; n++){

	    sumx = 0.0;
	    sumy = 0.0;
	    sumz = 0.0;

	    for (l=0; l<num_projectors; l++){
	      sumx -= DS_VNA1[0][Matomnum+1][kl1][m][l]*DS_VNA1[1][Matomnum+1][kl2][n][l];
	      sumy -= DS_VNA1[0][Matomnum+1][kl1][m][l]*DS_VNA1[2][Matomnum+1][kl2][n][l];
	      sumz -= DS_VNA1[0][Matomnum+1][kl1][m][l]*DS_VNA1[3][Matomnum+1][kl2][n][l];
	    }

	    Hx[m][n] += sumx;
	    Hy[m][n] += sumy;
	    Hz[m][n] += sumz;

	  }
	}
      }

    }
  }

  /****************************************************
                contribution by dampingF 
  ****************************************************/

  /* Qij * dH/d(x,y,z) for forces */

  for (m=0; m<ian; m++){
    for (n=0; n<jan; n++){
      Hx[m][n] = dmp*Hx[m][n];
      Hy[m][n] = dmp*Hy[m][n];
      Hz[m][n] = dmp*Hz[m][n];
    }
  }

  /* Qij * H_sts for stress */

  for (i=1; i<=3; i++){
    for (j=1; j<=3; j++){
      for (m=0; m<ian; m++){
	for (n=0; n<jan; n++){
	  H_sts[i][j][m][n] = dmp*H_sts[i][j][m][n];
	  H_sts[i][j][m][n] = dmp*H_sts[i][j][m][n];
	}
      }
    }
  }

  /* dQij/d(x,y,z) * H for forces and dQij/depsilon * H for stress */

  if ( (h_AN==0 && q_AN!=0) || (h_AN!=0 && q_AN==0) ){

    if      (h_AN==0)   kl = q_AN;
    else if (q_AN==0)   kl = h_AN;

    r = Dis[Gc_AN][kl];
 
    if (rcut<=r) {
      deri_dmp = 0.0;
      tmp = 0.0; 
    }
    else {
      deri_dmp = deri_dampingF(rcut,r);
      tmp = deri_dmp/dmp;
    }

    x0 = Gxyz[ig][1] + atv[Rni][1];
    x1 = Gxyz[jg][1] + atv[Rnj][1];

    y0 = Gxyz[ig][2] + atv[Rni][2];
    y1 = Gxyz[jg][2] + atv[Rnj][2];

    z0 = Gxyz[ig][3] + atv[Rni][3];
    z1 = Gxyz[jg][3] + atv[Rnj][3];

    tx = x0 - x1;
    ty = y0 - y1;
    tz = z0 - z1;

    /* for empty atoms or finite elemens basis */
    if (r<1.0e-10) r = 1.0e-10;

    if ( h_AN==0 ){
      dx = tmp*(x0-x1)/r;
      dy = tmp*(y0-y1)/r;
      dz = tmp*(z0-z1)/r;
    }

    else if ( q_AN==0 ){
      dx = tmp*(x1-x0)/r;
      dy = tmp*(y1-y0)/r;
      dz = tmp*(z1-z0)/r;
    }

    if (h_AN==0){ 

      /* for forces */

      for (m=0; m<ian; m++){
        for (n=0; n<jan; n++){
	  Hx[m][n] += HVNA[Mc_AN][kl][m][n]*dx;
	  Hy[m][n] += HVNA[Mc_AN][kl][m][n]*dy;
	  Hz[m][n] += HVNA[Mc_AN][kl][m][n]*dz;
        }
      }

      /* for stress */

      for (m=0; m<ian; m++){
	for (n=0; n<jan; n++){
	  H_sts[1][1][m][n] += HVNA[Mc_AN][kl][m][n]*dx*tx;
	  H_sts[1][2][m][n] += HVNA[Mc_AN][kl][m][n]*dx*ty;
	  H_sts[1][3][m][n] += HVNA[Mc_AN][kl][m][n]*dx*tz;
	  H_sts[2][1][m][n] += HVNA[Mc_AN][kl][m][n]*dy*tx;
	  H_sts[2][2][m][n] += HVNA[Mc_AN][kl][m][n]*dy*ty;
	  H_sts[2][3][m][n] += HVNA[Mc_AN][kl][m][n]*dy*tz;
	  H_sts[3][1][m][n] += HVNA[Mc_AN][kl][m][n]*dz*tx;
	  H_sts[3][2][m][n] += HVNA[Mc_AN][kl][m][n]*dz*ty;
	  H_sts[3][3][m][n] += HVNA[Mc_AN][kl][m][n]*dz*tz;
	}
      }

    } /* if (h_AN==0) */ 

    else if (q_AN==0){ 

      /* for forces */

      for (m=0; m<ian; m++){
        for (n=0; n<jan; n++){
	  Hx[m][n] += HVNA[Mc_AN][kl][n][m]*dx;
	  Hy[m][n] += HVNA[Mc_AN][kl][n][m]*dy;
	  Hz[m][n] += HVNA[Mc_AN][kl][n][m]*dz;
        }
      }

    }
  }

  /*
  printf("ABC1 Mc_AN=%2d h_AN=%2d q_AN=%2d\n",Mc_AN,h_AN,q_AN);
  for (m=0; m<ian; m++){
    for (n=0; n<jan; n++){
      printf("%10.5f ",H_sts[1][1][m][n]);
    }
    printf("\n");
  }
  */

}



void dHNL(int where_flag,
          int Mc_AN, int h_AN, int q_AN,
          double ******DS_NL1,
          dcomplex ***Hx, dcomplex ***Hy, dcomplex ***Hz,
          dcomplex *****H_sts)
{
  int i,j,k,m,n,l,kg,kan,so,deri_kind;
  int ig,ian,jg,jan,kl,kl1,kl2;
  int wakg,l1,l2,l3,Gc_AN,Mi_AN,Mi_AN2,Mj_AN,Mj_AN2;
  int Rni,Rnj,Rnk,somax;
  double PF[2],sumx,sumy,sumz,ene,dmp,deri_dmp;
  double tmpx,tmpy,tmpz,tmp,r;
  double x0,y0,z0,x1,y1,z1,dx,dy,dz;
  double tx,ty,tz;
  double rcuti,rcutj,rcut;
  double PFp,PFm,ene_p,ene_m;
  dcomplex sumx0,sumy0,sumz0;
  dcomplex sumx1,sumy1,sumz1;
  dcomplex sumx2,sumy2,sumz2;

  /****************************************************
   start calc.
  ****************************************************/

  Gc_AN = M2G[Mc_AN];
  ig = natn[Gc_AN][h_AN];
  Rni = ncn[Gc_AN][h_AN];
  Mi_AN = F_G2M[ig];
  ian = Spe_Total_CNO[WhatSpecies[ig]];
  rcuti = Spe_Atom_Cut1[WhatSpecies[ig]];

  jg = natn[Gc_AN][q_AN];
  Rnj = ncn[Gc_AN][q_AN];
  Mj_AN = F_G2M[jg];
  jan = Spe_Total_CNO[WhatSpecies[jg]];
  rcutj = Spe_Atom_Cut1[WhatSpecies[jg]];

  rcut = rcuti + rcutj;
  kl = RMI1[Mc_AN][h_AN][q_AN];
  dmp = dampingF(rcut,Dis[ig][kl]);

  for (so=0; so<3; so++){
    for (i=0; i<List_YOUSO[7]; i++){
      for (j=0; j<List_YOUSO[7]; j++){
	Hx[so][i][j] = Complex(0.0,0.0);
	Hy[so][i][j] = Complex(0.0,0.0);
	Hz[so][i][j] = Complex(0.0,0.0);
      }
    }
  }

  for (so=0; so<3; so++){
    for (i=0; i<4; i++){
      for (j=0; j<4; j++){
	for (m=0; m<List_YOUSO[7]; m++){
	  for (n=0; n<List_YOUSO[7]; n++){
	    H_sts[so][i][j][m][n] = Complex(0.0,0.0);
	  }
	}
      }
    }
  }

  if (h_AN==0){

    /****************************************************
                          dH*ep*H
    ****************************************************/

    for (k=0; k<=FNAN[Gc_AN]; k++){

      kg = natn[Gc_AN][k];
      Rnk = ncn[Gc_AN][k];
      wakg = WhatSpecies[kg];
      kan = Spe_Total_VPS_Pro[wakg];
      kl = RMI1[Mc_AN][q_AN][k];

      tx = Gxyz[Gc_AN][1] - Gxyz[kg][1] - atv[Rnk][1];
      ty = Gxyz[Gc_AN][2] - Gxyz[kg][2] - atv[Rnk][2];
      tz = Gxyz[Gc_AN][3] - Gxyz[kg][3] - atv[Rnk][3];

      /****************************************************
                   l-dependent non-local part
      ****************************************************/

      if (0<=kl && VPS_j_dependency[wakg]==0 && where_flag==0){

	for (m=0; m<ian; m++){
	  for (n=0; n<jan; n++){

	    sumx = 0.0;
	    sumy = 0.0;
	    sumz = 0.0;

	    l = 0;
	    for (l1=1; l1<=Spe_Num_RVPS[wakg]; l1++){

	      ene = Spe_VNLE[0][wakg][l1-1];
	      if      (Spe_VPS_List[wakg][l1]==0) l2 = 0;
	      else if (Spe_VPS_List[wakg][l1]==1) l2 = 2;
	      else if (Spe_VPS_List[wakg][l1]==2) l2 = 4;
	      else if (Spe_VPS_List[wakg][l1]==3) l2 = 6;

              if (Mj_AN<=Matomnum) Mj_AN2 = Mj_AN;
              else                 Mj_AN2 = Matomnum + 1; 

	      for (l3=0; l3<=l2; l3++){
		sumx += ene*DS_NL1[0][1][Mc_AN][k][m][l]*DS_NL1[0][0][Mj_AN2][kl][n][l];
		sumy += ene*DS_NL1[0][2][Mc_AN][k][m][l]*DS_NL1[0][0][Mj_AN2][kl][n][l];
		sumz += ene*DS_NL1[0][3][Mc_AN][k][m][l]*DS_NL1[0][0][Mj_AN2][kl][n][l];
		l++;
	      }
	    }

            /* for forces */

	    Hx[0][m][n].r += sumx;
	    Hy[0][m][n].r += sumy;
	    Hz[0][m][n].r += sumz;

	    Hx[1][m][n].r += sumx;
	    Hy[1][m][n].r += sumy;
	    Hz[1][m][n].r += sumz;

            /* for stress calculation */ 

	    H_sts[0][1][1][m][n].r += sumx*tx;
	    H_sts[0][1][2][m][n].r += sumx*ty;
	    H_sts[0][1][3][m][n].r += sumx*tz;

	    H_sts[0][2][1][m][n].r += sumy*tx;
	    H_sts[0][2][2][m][n].r += sumy*ty;
	    H_sts[0][2][3][m][n].r += sumy*tz;

	    H_sts[0][3][1][m][n].r += sumz*tx;
	    H_sts[0][3][2][m][n].r += sumz*ty;
	    H_sts[0][3][3][m][n].r += sumz*tz;

	    H_sts[1][1][1][m][n].r += sumx*tx;
	    H_sts[1][1][2][m][n].r += sumx*ty;
	    H_sts[1][1][3][m][n].r += sumx*tz;

	    H_sts[1][2][1][m][n].r += sumy*tx;
	    H_sts[1][2][2][m][n].r += sumy*ty;
	    H_sts[1][2][3][m][n].r += sumy*tz;

	    H_sts[1][3][1][m][n].r += sumz*tx;
	    H_sts[1][3][2][m][n].r += sumz*ty;
	    H_sts[1][3][3][m][n].r += sumz*tz;

	  } /* n */
	} /* m */

      } /* if */

      /****************************************************
                   j-dependent non-local part
      ****************************************************/

      else if ( 0<=kl && VPS_j_dependency[wakg]==1 && where_flag==0 ){

	for (m=0; m<ian; m++){
	  for (n=0; n<jan; n++){

	    sumx0 = Complex(0.0,0.0);
	    sumy0 = Complex(0.0,0.0);
	    sumz0 = Complex(0.0,0.0);

	    sumx1 = Complex(0.0,0.0);
	    sumy1 = Complex(0.0,0.0);
	    sumz1 = Complex(0.0,0.0);

	    sumx2 = Complex(0.0,0.0);
	    sumy2 = Complex(0.0,0.0);
	    sumz2 = Complex(0.0,0.0);

            if (Mj_AN<=Matomnum) Mj_AN2 = Mj_AN;
            else                 Mj_AN2 = Matomnum + 1; 

	    l = 0;
	    for (l1=1; l1<=Spe_Num_RVPS[wakg]; l1++){

	      ene_p = Spe_VNLE[0][wakg][l1-1];
	      ene_m = Spe_VNLE[1][wakg][l1-1];

	      if      (Spe_VPS_List[wakg][l1]==0) { l2=0; PFp=1.0;     PFm=0.0;     }  
	      else if (Spe_VPS_List[wakg][l1]==1) { l2=2; PFp=2.0/3.0; PFm=1.0/3.0; }
	      else if (Spe_VPS_List[wakg][l1]==2) { l2=4; PFp=3.0/5.0; PFm=2.0/5.0; }
	      else if (Spe_VPS_List[wakg][l1]==3) { l2=6; PFp=4.0/7.0; PFm=3.0/7.0; }

	      dHNL_SO(&sumx0.r,&sumy0.r,&sumz0.r,
                      &sumx1.r,&sumy1.r,&sumz1.r,
                      &sumx2.r,&sumy2.r,&sumz2.r,
                      &sumx0.i,&sumy0.i,&sumz0.i,
                      &sumx1.i,&sumy1.i,&sumz1.i,
                      &sumx2.i,&sumy2.i,&sumz2.i,
                      1.0,
                      PFp, PFm,
                      ene_p,ene_m,
                      l2, &l,
                      Mc_AN ,k, m,
                      Mj_AN2,kl,n,
                      DS_NL1);
	    }

            /* if (q_AN==0) */

            if (q_AN==0){

	      l = 0;
	      for (l1=1; l1<=Spe_Num_RVPS[wakg]; l1++){

		ene_p = Spe_VNLE[0][wakg][l1-1];
		ene_m = Spe_VNLE[1][wakg][l1-1];

		if      (Spe_VPS_List[wakg][l1]==0) { l2=0; PFp=1.0;     PFm=0.0;     }  
		else if (Spe_VPS_List[wakg][l1]==1) { l2=2; PFp=2.0/3.0; PFm=1.0/3.0; }
		else if (Spe_VPS_List[wakg][l1]==2) { l2=4; PFp=3.0/5.0; PFm=2.0/5.0; }
		else if (Spe_VPS_List[wakg][l1]==3) { l2=6; PFp=4.0/7.0; PFm=3.0/7.0; }

		dHNL_SO(&sumx0.r,&sumy0.r,&sumz0.r,
			&sumx1.r,&sumy1.r,&sumz1.r,
			&sumx2.r,&sumy2.r,&sumz2.r,
			&sumx0.i,&sumy0.i,&sumz0.i,
			&sumx1.i,&sumy1.i,&sumz1.i,
                        &sumx2.i,&sumy2.i,&sumz2.i,
			-1.0,
			PFp, PFm,
			ene_p,ene_m,
			l2, &l,
			Mj_AN2, kl, n,
			Mc_AN,  k,  m,
			DS_NL1);

	      }

	    } /* if (q_AN==0) */

            /* for forces */  

	    Hx[0][m][n].r += sumx0.r;     /* up-up */
	    Hy[0][m][n].r += sumy0.r;     /* up-up */
	    Hz[0][m][n].r += sumz0.r;     /* up-up */

	    Hx[1][m][n].r += sumx1.r;     /* dn-dn */
	    Hy[1][m][n].r += sumy1.r;     /* dn-dn */
	    Hz[1][m][n].r += sumz1.r;     /* dn-dn */

	    Hx[2][m][n].r += sumx2.r;     /* up-dn */
	    Hy[2][m][n].r += sumy2.r;     /* up-dn */
	    Hz[2][m][n].r += sumz2.r;     /* up-dn */

	    Hx[0][m][n].i += sumx0.i;     /* up-up */
	    Hy[0][m][n].i += sumy0.i;     /* up-up */
	    Hz[0][m][n].i += sumz0.i;     /* up-up */

	    Hx[1][m][n].i += sumx1.i;     /* dn-dn */
	    Hy[1][m][n].i += sumy1.i;     /* dn-dn */
	    Hz[1][m][n].i += sumz1.i;     /* dn-dn */

	    Hx[2][m][n].i += sumx2.i;     /* up-dn */
	    Hy[2][m][n].i += sumy2.i;     /* up-dn */
	    Hz[2][m][n].i += sumz2.i;     /* up-dn */

            /* for stress calculation */ 

            /* up-up, real */
	    H_sts[0][1][1][m][n].r += sumx0.r*tx;
	    H_sts[0][1][2][m][n].r += sumx0.r*ty;
	    H_sts[0][1][3][m][n].r += sumx0.r*tz;

	    H_sts[0][2][1][m][n].r += sumy0.r*tx;
	    H_sts[0][2][2][m][n].r += sumy0.r*ty;
	    H_sts[0][2][3][m][n].r += sumy0.r*tz;

	    H_sts[0][3][1][m][n].r += sumz0.r*tx;
	    H_sts[0][3][2][m][n].r += sumz0.r*ty;
	    H_sts[0][3][3][m][n].r += sumz0.r*tz;

            /* dn-dn, real */
	    H_sts[1][1][1][m][n].r += sumx1.r*tx;
	    H_sts[1][1][2][m][n].r += sumx1.r*ty;
	    H_sts[1][1][3][m][n].r += sumx1.r*tz;

	    H_sts[1][2][1][m][n].r += sumy1.r*tx;
	    H_sts[1][2][2][m][n].r += sumy1.r*ty;
	    H_sts[1][2][3][m][n].r += sumy1.r*tz;

	    H_sts[1][3][1][m][n].r += sumz1.r*tx;
	    H_sts[1][3][2][m][n].r += sumz1.r*ty;
	    H_sts[1][3][3][m][n].r += sumz1.r*tz;

            /* up-dn, real */
	    H_sts[2][1][1][m][n].r += sumx1.r*tx;
	    H_sts[2][1][2][m][n].r += sumx1.r*ty;
	    H_sts[2][1][3][m][n].r += sumx1.r*tz;

	    H_sts[2][2][1][m][n].r += sumy1.r*tx;
	    H_sts[2][2][2][m][n].r += sumy1.r*ty;
	    H_sts[2][2][3][m][n].r += sumy1.r*tz;

	    H_sts[2][3][1][m][n].r += sumz1.r*tx;
	    H_sts[2][3][2][m][n].r += sumz1.r*ty;
	    H_sts[2][3][3][m][n].r += sumz1.r*tz;

            /* up-up, imaginary */
	    H_sts[0][1][1][m][n].r += sumx0.i*tx;
	    H_sts[0][1][2][m][n].r += sumx0.i*ty;
	    H_sts[0][1][3][m][n].r += sumx0.i*tz;

	    H_sts[0][2][1][m][n].r += sumy0.i*tx;
	    H_sts[0][2][2][m][n].r += sumy0.i*ty;
	    H_sts[0][2][3][m][n].r += sumy0.i*tz;

	    H_sts[0][3][1][m][n].r += sumz0.i*tx;
	    H_sts[0][3][2][m][n].r += sumz0.i*ty;
	    H_sts[0][3][3][m][n].r += sumz0.i*tz;

            /* dn-dn, imaginary */
	    H_sts[1][1][1][m][n].r += sumx1.i*tx;
	    H_sts[1][1][2][m][n].r += sumx1.i*ty;
	    H_sts[1][1][3][m][n].r += sumx1.i*tz;

	    H_sts[1][2][1][m][n].r += sumy1.i*tx;
	    H_sts[1][2][2][m][n].r += sumy1.i*ty;
	    H_sts[1][2][3][m][n].r += sumy1.i*tz;

	    H_sts[1][3][1][m][n].r += sumz1.i*tx;
	    H_sts[1][3][2][m][n].r += sumz1.i*ty;
	    H_sts[1][3][3][m][n].r += sumz1.i*tz;

            /* up-dn, imaginary */
	    H_sts[2][1][1][m][n].r += sumx1.i*tx;
	    H_sts[2][1][2][m][n].r += sumx1.i*ty;
	    H_sts[2][1][3][m][n].r += sumx1.i*tz;

	    H_sts[2][2][1][m][n].r += sumy1.i*tx;
	    H_sts[2][2][2][m][n].r += sumy1.i*ty;
	    H_sts[2][2][3][m][n].r += sumy1.i*tz;

	    H_sts[2][3][1][m][n].r += sumz1.i*tx;
	    H_sts[2][3][2][m][n].r += sumz1.i*ty;
	    H_sts[2][3][3][m][n].r += sumz1.i*tz;

	  }
	}
      }

    } /* k */

    /****************************************************
                           H*ep*dH 
    ****************************************************/

    /* h_AN==0 && q_AN==0 */

    if (q_AN==0 && VPS_j_dependency[wakg]==0){

      /* for forces */

      for (m=0; m<ian; m++){
        for (n=m; n<jan; n++){

          tmpx = Hx[0][m][n].r + Hx[0][n][m].r;
          Hx[0][m][n].r = tmpx; 
          Hx[0][n][m].r = tmpx;
          Hx[1][m][n].r = tmpx; 
          Hx[1][n][m].r = tmpx;

          tmpy = Hy[0][m][n].r + Hy[0][n][m].r;
          Hy[0][m][n].r = tmpy; 
          Hy[0][n][m].r = tmpy;
          Hy[1][m][n].r = tmpy; 
          Hy[1][n][m].r = tmpy;

          tmpz = Hz[0][m][n].r + Hz[0][n][m].r;
          Hz[0][m][n].r = tmpz; 
          Hz[0][n][m].r = tmpz;
          Hz[1][m][n].r = tmpz; 
          Hz[1][n][m].r = tmpz;
        }
      }

      /* for stress */

      for (i=1; i<=3; i++){
        for (j=1; j<=3; j++){
	  for (m=0; m<ian; m++){
	    for (n=m; n<jan; n++){

              tmp = H_sts[0][i][j][m][n].r + H_sts[0][i][j][n][m].r;
              H_sts[0][i][j][m][n].r = tmp;
              H_sts[0][i][j][n][m].r = tmp;

              tmp = H_sts[1][i][j][m][n].r + H_sts[1][i][j][n][m].r;
              H_sts[1][i][j][m][n].r = tmp;
              H_sts[1][i][j][n][m].r = tmp;
	    }
	  }
        }
      }

    } /* if (q_AN==0 && VPS_j_dependency[wakg]==0) */

    /*******************************************************
      where_flag==1 means that dHNL is called from 
      THE SECOND CASE:
      In case of I=k with I!=i and I!=j
      d [ \sum_k <i|k>ek<k|j> ]/dRI 

      In this case, there is no contribution to the stress.
    ********************************************************/

    else if (where_flag==1){

      kg = natn[Gc_AN][0];
      wakg = WhatSpecies[kg];
      kan = Spe_Total_VPS_Pro[wakg];
      kl = RMI1[Mc_AN][q_AN][0];

      /****************************************************
                   l-dependent non-local part
      ****************************************************/

      if (VPS_j_dependency[wakg]==0){

	for (m=0; m<ian; m++){
	  for (n=0; n<jan; n++){

	    sumx = 0.0;
	    sumy = 0.0;
	    sumz = 0.0;

            if (Mj_AN<=Matomnum){
              Mj_AN2 = Mj_AN;
 	      kl2 = RMI1[Mc_AN][q_AN][0];
	    }
            else{
              Mj_AN2 = Matomnum + 1; 
	      kl2 = RMI1[Mc_AN][0][q_AN];
	    }

	    l = 0;
	    for (l1=1; l1<=Spe_Num_RVPS[wakg]; l1++){

	      ene = Spe_VNLE[0][wakg][l1-1];
	      if      (Spe_VPS_List[wakg][l1]==0) l2 = 0;
	      else if (Spe_VPS_List[wakg][l1]==1) l2 = 2;
	      else if (Spe_VPS_List[wakg][l1]==2) l2 = 4;
	      else if (Spe_VPS_List[wakg][l1]==3) l2 = 6;

	      for (l3=0; l3<=l2; l3++){

		sumx -= ene*DS_NL1[0][0][Mc_AN][0][m][l]*DS_NL1[0][1][Mj_AN2][kl2][n][l];
		sumy -= ene*DS_NL1[0][0][Mc_AN][0][m][l]*DS_NL1[0][2][Mj_AN2][kl2][n][l];
		sumz -= ene*DS_NL1[0][0][Mc_AN][0][m][l]*DS_NL1[0][3][Mj_AN2][kl2][n][l];
		l++;
	      }
	    }

            /* for forces */

	    Hx[0][m][n].r += sumx;
	    Hy[0][m][n].r += sumy;
	    Hz[0][m][n].r += sumz;

	    Hx[1][m][n].r += sumx;
	    Hy[1][m][n].r += sumy;
	    Hz[1][m][n].r += sumz;

            /* for stress */

	    H_sts[0][1][1][m][n].r += sumx*tx;
	    H_sts[0][1][2][m][n].r += sumx*ty;
	    H_sts[0][1][3][m][n].r += sumx*tz;

	    H_sts[0][2][1][m][n].r += sumy*tx;
	    H_sts[0][2][2][m][n].r += sumy*ty;
	    H_sts[0][2][3][m][n].r += sumy*tz;

	    H_sts[0][3][1][m][n].r += sumz*tx;
	    H_sts[0][3][2][m][n].r += sumz*ty;
	    H_sts[0][3][3][m][n].r += sumz*tz;

	    H_sts[1][1][1][m][n].r += sumx*tx;
	    H_sts[1][1][2][m][n].r += sumx*ty;
	    H_sts[1][1][3][m][n].r += sumx*tz;

	    H_sts[1][2][1][m][n].r += sumy*tx;
	    H_sts[1][2][2][m][n].r += sumy*ty;
	    H_sts[1][2][3][m][n].r += sumy*tz;

	    H_sts[1][3][1][m][n].r += sumz*tx;
	    H_sts[1][3][2][m][n].r += sumz*ty;
	    H_sts[1][3][3][m][n].r += sumz*tz;

	  } /* n */
	} /* m */
      } /* if (VPS_j_dependency[wakg]==0) */

      /****************************************************
                   j-dependent non-local part
      ****************************************************/

      else if ( VPS_j_dependency[wakg]==1 ){

	for (m=0; m<ian; m++){
	  for (n=0; n<jan; n++){

	    sumx0 = Complex(0.0,0.0);
	    sumy0 = Complex(0.0,0.0);
	    sumz0 = Complex(0.0,0.0);

	    sumx1 = Complex(0.0,0.0);
	    sumy1 = Complex(0.0,0.0);
	    sumz1 = Complex(0.0,0.0);

	    sumx2 = Complex(0.0,0.0);
	    sumy2 = Complex(0.0,0.0);
	    sumz2 = Complex(0.0,0.0);

            if (Mj_AN<=Matomnum){
              Mj_AN2 = Mj_AN;
 	      kl2 = RMI1[Mc_AN][q_AN][0];
	    }
            else{
              Mj_AN2 = Matomnum + 1; 
	      kl2 = RMI1[Mc_AN][0][q_AN];
	    }

	    l = 0;
	    for (l1=1; l1<=Spe_Num_RVPS[wakg]; l1++){

	      ene_p = Spe_VNLE[0][wakg][l1-1];
	      ene_m = Spe_VNLE[1][wakg][l1-1];

	      if      (Spe_VPS_List[wakg][l1]==0) { l2=0; PFp=1.0;     PFm=0.0;     }  
	      else if (Spe_VPS_List[wakg][l1]==1) { l2=2; PFp=2.0/3.0; PFm=1.0/3.0; }
	      else if (Spe_VPS_List[wakg][l1]==2) { l2=4; PFp=3.0/5.0; PFm=2.0/5.0; }
	      else if (Spe_VPS_List[wakg][l1]==3) { l2=6; PFp=4.0/7.0; PFm=3.0/7.0; }

       	      /* 1 */

	      dHNL_SO(&sumx0.r,&sumy0.r,&sumz0.r,
                      &sumx1.r,&sumy1.r,&sumz1.r,
                      &sumx2.r,&sumy2.r,&sumz2.r,
                      &sumx0.i,&sumy0.i,&sumz0.i,
                      &sumx1.i,&sumy1.i,&sumz1.i,
                      &sumx2.i,&sumy2.i,&sumz2.i,
                      -1.0,
                      PFp, PFm,
                      -ene_p,-ene_m,
                      l2, &l,
                      Mj_AN2,kl2,n,
                      Mc_AN, 0,  m,
                      DS_NL1);
	    }

	    Hx[0][m][n].r += sumx0.r;     /* up-up */
	    Hy[0][m][n].r += sumy0.r;     /* up-up */
	    Hz[0][m][n].r += sumz0.r;     /* up-up */

	    Hx[1][m][n].r += sumx1.r;     /* dn-dn */
	    Hy[1][m][n].r += sumy1.r;     /* dn-dn */
	    Hz[1][m][n].r += sumz1.r;     /* dn-dn */

	    Hx[2][m][n].r += sumx2.r;     /* up-dn */
	    Hy[2][m][n].r += sumy2.r;     /* up-dn */
	    Hz[2][m][n].r += sumz2.r;     /* up-dn */

	    Hx[0][m][n].i += sumx0.i;     /* up-up */
	    Hy[0][m][n].i += sumy0.i;     /* up-up */
	    Hz[0][m][n].i += sumz0.i;     /* up-up */

	    Hx[1][m][n].i += sumx1.i;     /* dn-dn */
	    Hy[1][m][n].i += sumy1.i;     /* dn-dn */
	    Hz[1][m][n].i += sumz1.i;     /* dn-dn */

	    Hx[2][m][n].i += sumx2.i;     /* up-dn */
	    Hy[2][m][n].i += sumy2.i;     /* up-dn */
	    Hz[2][m][n].i += sumz2.i;     /* up-dn */

	  }
	}
      }

    } /* else if (where_flag==1) */

  } /* if (h_AN==0) */

  else if (where_flag==0){

    /****************************************************
       H*ep*dH

       if (h_AN!=0 && where_flag==0)
       This happens 
       only if 
       ( SpinP_switch==3
         &&
         (SO_switch==1 || (Hub_U_switch==1 && F_U_flag==1) 
  	   || 1<=Constraint_NCS_switch || Zeeman_NCS_switch==1
           || Zeeman_NCO_switch==1)
         && 
         q_AN==0
       )
    ****************************************************/

    for (k=0; k<=FNAN[Gc_AN]; k++){

      kg = natn[Gc_AN][k];
      Rnk = ncn[Gc_AN][k];
      wakg = WhatSpecies[kg];
      kan = Spe_Total_VPS_Pro[wakg];
      kl = RMI1[Mc_AN][h_AN][k];

      tx = Gxyz[Gc_AN][1] - Gxyz[kg][1] - atv[Rnk][1];
      ty = Gxyz[Gc_AN][2] - Gxyz[kg][2] - atv[Rnk][2];
      tz = Gxyz[Gc_AN][3] - Gxyz[kg][3] - atv[Rnk][3];

      if (Mi_AN<=Matomnum) Mi_AN2 = Mi_AN;
      else                 Mi_AN2 = Matomnum + 1; 

      if (0<=kl && VPS_j_dependency[wakg]==1){

	for (m=0; m<ian; m++){
	  for (n=0; n<jan; n++){

	    sumx0 = Complex(0.0,0.0);
	    sumy0 = Complex(0.0,0.0);
	    sumz0 = Complex(0.0,0.0);

	    sumx1 = Complex(0.0,0.0);
	    sumy1 = Complex(0.0,0.0);
	    sumz1 = Complex(0.0,0.0);

	    sumx2 = Complex(0.0,0.0);
	    sumy2 = Complex(0.0,0.0);
	    sumz2 = Complex(0.0,0.0);

	    l = 0;
	    for (l1=1; l1<=Spe_Num_RVPS[wakg]; l1++){

	      ene_p = Spe_VNLE[0][wakg][l1-1];
	      ene_m = Spe_VNLE[1][wakg][l1-1];

	      if      (Spe_VPS_List[wakg][l1]==0) { l2=0; PFp=1.0;     PFm=0.0;     }  
	      else if (Spe_VPS_List[wakg][l1]==1) { l2=2; PFp=2.0/3.0; PFm=1.0/3.0; }
	      else if (Spe_VPS_List[wakg][l1]==2) { l2=4; PFp=3.0/5.0; PFm=2.0/5.0; }
	      else if (Spe_VPS_List[wakg][l1]==3) { l2=6; PFp=4.0/7.0; PFm=3.0/7.0; }

	      dHNL_SO(&sumx0.r,&sumy0.r,&sumz0.r,
                      &sumx1.r,&sumy1.r,&sumz1.r,
                      &sumx2.r,&sumy2.r,&sumz2.r,
                      &sumx0.i,&sumy0.i,&sumz0.i,
                      &sumx1.i,&sumy1.i,&sumz1.i,
                      &sumx2.i,&sumy2.i,&sumz2.i,
                      -1.0,
                      PFp, PFm,
                      ene_p, ene_m,
                      l2, &l,
                      Mj_AN,  k,  n,
                      Mi_AN2, kl, m,
                      DS_NL1);
	    }

            /* for forces */ 

	    Hx[0][m][n].r += sumx0.r;     /* up-up */
	    Hy[0][m][n].r += sumy0.r;     /* up-up */
	    Hz[0][m][n].r += sumz0.r;     /* up-up */

	    Hx[1][m][n].r += sumx1.r;     /* dn-dn */
	    Hy[1][m][n].r += sumy1.r;     /* dn-dn */
	    Hz[1][m][n].r += sumz1.r;     /* dn-dn */

	    Hx[2][m][n].r += sumx2.r;     /* up-dn */
	    Hy[2][m][n].r += sumy2.r;     /* up-dn */
	    Hz[2][m][n].r += sumz2.r;     /* up-dn */

	    Hx[0][m][n].i += sumx0.i;     /* up-up */
	    Hy[0][m][n].i += sumy0.i;     /* up-up */
	    Hz[0][m][n].i += sumz0.i;     /* up-up */

	    Hx[1][m][n].i += sumx1.i;     /* dn-dn */
	    Hy[1][m][n].i += sumy1.i;     /* dn-dn */
	    Hz[1][m][n].i += sumz1.i;     /* dn-dn */

	    Hx[2][m][n].i += sumx2.i;     /* up-dn */
	    Hy[2][m][n].i += sumy2.i;     /* up-dn */
	    Hz[2][m][n].i += sumz2.i;     /* up-dn */

            /* for stress calculation */ 

            /* up-up, real */
	    H_sts[0][1][1][m][n].r += sumx0.r*tx;
	    H_sts[0][1][2][m][n].r += sumx0.r*ty;
	    H_sts[0][1][3][m][n].r += sumx0.r*tz;

	    H_sts[0][2][1][m][n].r += sumy0.r*tx;
	    H_sts[0][2][2][m][n].r += sumy0.r*ty;
	    H_sts[0][2][3][m][n].r += sumy0.r*tz;

	    H_sts[0][3][1][m][n].r += sumz0.r*tx;
	    H_sts[0][3][2][m][n].r += sumz0.r*ty;
	    H_sts[0][3][3][m][n].r += sumz0.r*tz;

            /* dn-dn, real */
	    H_sts[1][1][1][m][n].r += sumx1.r*tx;
	    H_sts[1][1][2][m][n].r += sumx1.r*ty;
	    H_sts[1][1][3][m][n].r += sumx1.r*tz;

	    H_sts[1][2][1][m][n].r += sumy1.r*tx;
	    H_sts[1][2][2][m][n].r += sumy1.r*ty;
	    H_sts[1][2][3][m][n].r += sumy1.r*tz;

	    H_sts[1][3][1][m][n].r += sumz1.r*tx;
	    H_sts[1][3][2][m][n].r += sumz1.r*ty;
	    H_sts[1][3][3][m][n].r += sumz1.r*tz;

            /* up-dn, real */
	    H_sts[2][1][1][m][n].r += sumx1.r*tx;
	    H_sts[2][1][2][m][n].r += sumx1.r*ty;
	    H_sts[2][1][3][m][n].r += sumx1.r*tz;

	    H_sts[2][2][1][m][n].r += sumy1.r*tx;
	    H_sts[2][2][2][m][n].r += sumy1.r*ty;
	    H_sts[2][2][3][m][n].r += sumy1.r*tz;

	    H_sts[2][3][1][m][n].r += sumz1.r*tx;
	    H_sts[2][3][2][m][n].r += sumz1.r*ty;
	    H_sts[2][3][3][m][n].r += sumz1.r*tz;

            /* up-up, imaginary */
	    H_sts[0][1][1][m][n].r += sumx0.i*tx;
	    H_sts[0][1][2][m][n].r += sumx0.i*ty;
	    H_sts[0][1][3][m][n].r += sumx0.i*tz;

	    H_sts[0][2][1][m][n].r += sumy0.i*tx;
	    H_sts[0][2][2][m][n].r += sumy0.i*ty;
	    H_sts[0][2][3][m][n].r += sumy0.i*tz;

	    H_sts[0][3][1][m][n].r += sumz0.i*tx;
	    H_sts[0][3][2][m][n].r += sumz0.i*ty;
	    H_sts[0][3][3][m][n].r += sumz0.i*tz;

            /* dn-dn, imaginary */
	    H_sts[1][1][1][m][n].r += sumx1.i*tx;
	    H_sts[1][1][2][m][n].r += sumx1.i*ty;
	    H_sts[1][1][3][m][n].r += sumx1.i*tz;

	    H_sts[1][2][1][m][n].r += sumy1.i*tx;
	    H_sts[1][2][2][m][n].r += sumy1.i*ty;
	    H_sts[1][2][3][m][n].r += sumy1.i*tz;

	    H_sts[1][3][1][m][n].r += sumz1.i*tx;
	    H_sts[1][3][2][m][n].r += sumz1.i*ty;
	    H_sts[1][3][3][m][n].r += sumz1.i*tz;

            /* up-dn, imaginary */
	    H_sts[2][1][1][m][n].r += sumx1.i*tx;
	    H_sts[2][1][2][m][n].r += sumx1.i*ty;
	    H_sts[2][1][3][m][n].r += sumx1.i*tz;

	    H_sts[2][2][1][m][n].r += sumy1.i*tx;
	    H_sts[2][2][2][m][n].r += sumy1.i*ty;
	    H_sts[2][2][3][m][n].r += sumy1.i*tz;

	    H_sts[2][3][1][m][n].r += sumz1.i*tx;
	    H_sts[2][3][2][m][n].r += sumz1.i*ty;
	    H_sts[2][3][3][m][n].r += sumz1.i*tz;

	  } /* n */
	} /* m */
      } /* if (0<=kl && VPS_j_dependency[wakg]==1) */

    } /* k */     

  } /* else if (where_flag==0) */

  /* if (h_AN!=0 && where_flag==1) */

  /*******************************************************
    where_flag==1 means that dHNL is called from 
    THE SECOND CASE:
    In case of I=k with I!=i and I!=j
    d [ \sum_k <i|k>ek<k|j> ]/dRI 

    In this case, there is no contribution to the stress.
  ********************************************************/

  else {

    /****************************************************
                           dH*ep*H
    ****************************************************/

    kg = natn[Gc_AN][0];
    wakg = WhatSpecies[kg];
    kan = Spe_Total_VPS_Pro[wakg];
    kl1 = RMI1[Mc_AN][0][h_AN];
    kl2 = RMI1[Mc_AN][0][q_AN];

    /****************************************************
                   l-dependent non-local part
    ****************************************************/

    if (VPS_j_dependency[wakg]==0){

      for (m=0; m<ian; m++){
	for (n=0; n<jan; n++){

	  sumx = 0.0;
	  sumy = 0.0;
	  sumz = 0.0;

	  l = 0;
	  for (l1=1; l1<=Spe_Num_RVPS[wakg]; l1++){

	    ene = Spe_VNLE[0][wakg][l1-1];
	    if      (Spe_VPS_List[wakg][l1]==0) l2 = 0;
	    else if (Spe_VPS_List[wakg][l1]==1) l2 = 2;
	    else if (Spe_VPS_List[wakg][l1]==2) l2 = 4;
	    else if (Spe_VPS_List[wakg][l1]==3) l2 = 6;

	    for (l3=0; l3<=l2; l3++){
	      sumx -= ene*DS_NL1[0][1][Matomnum+1][kl1][m][l]*DS_NL1[0][0][Matomnum+1][kl2][n][l];
	      sumy -= ene*DS_NL1[0][2][Matomnum+1][kl1][m][l]*DS_NL1[0][0][Matomnum+1][kl2][n][l];
	      sumz -= ene*DS_NL1[0][3][Matomnum+1][kl1][m][l]*DS_NL1[0][0][Matomnum+1][kl2][n][l];
	      l++;
	    }
	  }

	  Hx[0][m][n].r = sumx;
	  Hy[0][m][n].r = sumy;
	  Hz[0][m][n].r = sumz;

	  Hx[1][m][n].r = sumx;
	  Hy[1][m][n].r = sumy;
	  Hz[1][m][n].r = sumz;

	  Hx[2][m][n].r = 0.0;         
	  Hy[2][m][n].r = 0.0;         
	  Hz[2][m][n].r = 0.0;         

	  Hx[0][m][n].i = 0.0;
	  Hy[0][m][n].i = 0.0;
	  Hz[0][m][n].i = 0.0;

	  Hx[1][m][n].i = 0.0;
	  Hy[1][m][n].i = 0.0;
	  Hz[1][m][n].i = 0.0;

	  Hx[2][m][n].i = 0.0;
	  Hy[2][m][n].i = 0.0;
	  Hz[2][m][n].i = 0.0;

	}
      }
    }

    /****************************************************
                 j-dependent non-local part
    ****************************************************/

    else if ( VPS_j_dependency[wakg]==1 ){

      for (m=0; m<ian; m++){
	for (n=0; n<jan; n++){

	  sumx0 = Complex(0.0,0.0);
	  sumy0 = Complex(0.0,0.0);
	  sumz0 = Complex(0.0,0.0);

	  sumx1 = Complex(0.0,0.0);
	  sumy1 = Complex(0.0,0.0);
	  sumz1 = Complex(0.0,0.0);

          sumx2 = Complex(0.0,0.0);
	  sumy2 = Complex(0.0,0.0);
	  sumz2 = Complex(0.0,0.0);

	  l = 0;
	  for (l1=1; l1<=Spe_Num_RVPS[wakg]; l1++){

	    ene_p = Spe_VNLE[0][wakg][l1-1];
	    ene_m = Spe_VNLE[1][wakg][l1-1];

	    if      (Spe_VPS_List[wakg][l1]==0) { l2=0; PFp=1.0;     PFm=0.0;     }  
	    else if (Spe_VPS_List[wakg][l1]==1) { l2=2; PFp=2.0/3.0; PFm=1.0/3.0; }
	    else if (Spe_VPS_List[wakg][l1]==2) { l2=4; PFp=3.0/5.0; PFm=2.0/5.0; }
	    else if (Spe_VPS_List[wakg][l1]==3) { l2=6; PFp=4.0/7.0; PFm=3.0/7.0; }

   	    /* 2 */

	    dHNL_SO(&sumx0.r,&sumy0.r,&sumz0.r,
		    &sumx1.r,&sumy1.r,&sumz1.r,
                    &sumx2.r,&sumy2.r,&sumz2.r,
		    &sumx0.i,&sumy0.i,&sumz0.i,
		    &sumx1.i,&sumy1.i,&sumz1.i,
                    &sumx2.i,&sumy2.i,&sumz2.i,
                    1.0, 
		    PFp, PFm,
		    -ene_p,-ene_m,
		    l2, &l,
		    Matomnum+1, kl1,m,
		    Matomnum+1, kl2,n,
		    DS_NL1);
	  }

	  Hx[0][m][n].r = sumx0.r;     /* up-up */
	  Hy[0][m][n].r = sumy0.r;     /* up-up */
	  Hz[0][m][n].r = sumz0.r;     /* up-up */

	  Hx[1][m][n].r = sumx1.r;     /* dn-dn */
	  Hy[1][m][n].r = sumy1.r;     /* dn-dn */
	  Hz[1][m][n].r = sumz1.r;     /* dn-dn */

	  Hx[2][m][n].r = sumx2.r;     /* up-dn */
	  Hy[2][m][n].r = sumy2.r;     /* up-dn */
	  Hz[2][m][n].r = sumz2.r;     /* up-dn */

	  Hx[0][m][n].i = sumx0.i;     /* up-up */
	  Hy[0][m][n].i = sumy0.i;     /* up-up */
	  Hz[0][m][n].i = sumz0.i;     /* up-up */

	  Hx[1][m][n].i = sumx1.i;     /* dn-dn */
	  Hy[1][m][n].i = sumy1.i;     /* dn-dn */
	  Hz[1][m][n].i = sumz1.i;     /* dn-dn */

	  Hx[2][m][n].i = sumx2.i;     /* up-dn */
	  Hy[2][m][n].i = sumy2.i;     /* up-dn */
	  Hz[2][m][n].i = sumz2.i;     /* up-dn */

	}
      }
    }

    /****************************************************
                           H*ep*dH
    ****************************************************/

    if (q_AN!=0) {

      kg = natn[Gc_AN][0];
      wakg = WhatSpecies[kg];
      kan = Spe_Total_VPS_Pro[wakg];
      kl1 = RMI1[Mc_AN][0][h_AN];
      kl2 = RMI1[Mc_AN][0][q_AN];

      /****************************************************
                     l-dependent non-local part
      ****************************************************/

      if (VPS_j_dependency[wakg]==0){

	for (m=0; m<ian; m++){
	  for (n=0; n<jan; n++){

	    sumx = 0.0;
	    sumy = 0.0;
	    sumz = 0.0;

	    l = 0;
	    for (l1=1; l1<=Spe_Num_RVPS[wakg]; l1++){

	      ene = Spe_VNLE[0][wakg][l1-1];
	      if      (Spe_VPS_List[wakg][l1]==0) l2 = 0;
	      else if (Spe_VPS_List[wakg][l1]==1) l2 = 2;
	      else if (Spe_VPS_List[wakg][l1]==2) l2 = 4;
	      else if (Spe_VPS_List[wakg][l1]==3) l2 = 6;

	      for (l3=0; l3<=l2; l3++){
		sumx -= ene*DS_NL1[0][0][Matomnum+1][kl1][m][l]*DS_NL1[0][1][Matomnum+1][kl2][n][l];
		sumy -= ene*DS_NL1[0][0][Matomnum+1][kl1][m][l]*DS_NL1[0][2][Matomnum+1][kl2][n][l];
		sumz -= ene*DS_NL1[0][0][Matomnum+1][kl1][m][l]*DS_NL1[0][3][Matomnum+1][kl2][n][l];
		l++;
	      }
	    }

	    Hx[0][m][n].r += sumx;
	    Hy[0][m][n].r += sumy;
	    Hz[0][m][n].r += sumz;

	    Hx[1][m][n].r += sumx;
	    Hy[1][m][n].r += sumy;
	    Hz[1][m][n].r += sumz;
	  }
	}
      }

      /****************************************************
                    j-dependent non-local part
      ****************************************************/

      else if ( VPS_j_dependency[wakg]==1 ){

	for (m=0; m<ian; m++){
	  for (n=0; n<jan; n++){

	    sumx0 = Complex(0.0,0.0);
	    sumy0 = Complex(0.0,0.0);
	    sumz0 = Complex(0.0,0.0);

	    sumx1 = Complex(0.0,0.0);
	    sumy1 = Complex(0.0,0.0);
	    sumz1 = Complex(0.0,0.0);

	    sumx2 = Complex(0.0,0.0);
	    sumy2 = Complex(0.0,0.0);
	    sumz2 = Complex(0.0,0.0);

	    l = 0;
	    for (l1=1; l1<=Spe_Num_RVPS[wakg]; l1++){

 	      ene_p = Spe_VNLE[0][wakg][l1-1];
	      ene_m = Spe_VNLE[1][wakg][l1-1];

	      if      (Spe_VPS_List[wakg][l1]==0) { l2=0; PFp=1.0;     PFm=0.0;     }  
	      else if (Spe_VPS_List[wakg][l1]==1) { l2=2; PFp=2.0/3.0; PFm=1.0/3.0; }
	      else if (Spe_VPS_List[wakg][l1]==2) { l2=4; PFp=3.0/5.0; PFm=2.0/5.0; }
	      else if (Spe_VPS_List[wakg][l1]==3) { l2=6; PFp=4.0/7.0; PFm=3.0/7.0; }

	      /* 4 */

	      dHNL_SO(&sumx0.r,&sumy0.r,&sumz0.r,
		      &sumx1.r,&sumy1.r,&sumz1.r,
                      &sumx2.r,&sumy2.r,&sumz2.r,
		      &sumx0.i,&sumy0.i,&sumz0.i,
		      &sumx1.i,&sumy1.i,&sumz1.i,
                      &sumx2.i,&sumy2.i,&sumz2.i,
                      -1.0,
		      PFp, PFm,
		      -ene_p,-ene_m,
		      l2, &l,
		      Matomnum+1, kl2,n,
		      Matomnum+1, kl1,m,
		      DS_NL1);
	    }

	    Hx[0][m][n].r += sumx0.r;     /* up-up */
	    Hy[0][m][n].r += sumy0.r;     /* up-up */
	    Hz[0][m][n].r += sumz0.r;     /* up-up */

	    Hx[1][m][n].r += sumx1.r;     /* dn-dn */
	    Hy[1][m][n].r += sumy1.r;     /* dn-dn */
	    Hz[1][m][n].r += sumz1.r;     /* dn-dn */

	    Hx[2][m][n].r += sumx2.r;     /* up-dn */
	    Hy[2][m][n].r += sumy2.r;     /* up-dn */
	    Hz[2][m][n].r += sumz2.r;     /* up-dn */

	    Hx[0][m][n].i += sumx0.i;     /* up-up */
	    Hy[0][m][n].i += sumy0.i;     /* up-up */
	    Hz[0][m][n].i += sumz0.i;     /* up-up */

	    Hx[1][m][n].i += sumx1.i;     /* dn-dn */
	    Hy[1][m][n].i += sumy1.i;     /* dn-dn */
	    Hz[1][m][n].i += sumz1.i;     /* dn-dn */

	    Hx[2][m][n].i += sumx2.i;     /* up-dn */
	    Hy[2][m][n].i += sumy2.i;     /* up-dn */
	    Hz[2][m][n].i += sumz2.i;     /* up-dn */

	  }
	}
      }

    }

  } /* else */

  /****************************************************
               contribution by dampingF 
  ****************************************************/

  /* Qij * dH/d(x,y,z) for forces */

  for (so=0; so<3; so++){
    for (m=0; m<ian; m++){
      for (n=0; n<jan; n++){

        Hx[so][m][n].r = dmp*Hx[so][m][n].r;
        Hy[so][m][n].r = dmp*Hy[so][m][n].r;
        Hz[so][m][n].r = dmp*Hz[so][m][n].r;

        Hx[so][m][n].i = dmp*Hx[so][m][n].i;
        Hy[so][m][n].i = dmp*Hy[so][m][n].i;
        Hz[so][m][n].i = dmp*Hz[so][m][n].i;
      }
    }
  }

  /* Qij * H_sts for stress */

  for (so=0; so<3; so++){
    for (i=1; i<=3; i++){
      for (j=1; j<=3; j++){
	for (m=0; m<ian; m++){
	  for (n=0; n<jan; n++){
	    H_sts[so][i][j][m][n].r = dmp*H_sts[so][i][j][m][n].r;
	    H_sts[so][i][j][m][n].i = dmp*H_sts[so][i][j][m][n].i;
	  }
	}
      }
    }
  }

  /* dQij/d(x,y,z) * H for forces and dQij/depsilon * H for stress */

  if ( (h_AN==0 && q_AN!=0) || (h_AN!=0 && q_AN==0) ){

    if      (h_AN==0)   kl = q_AN;
    else if (q_AN==0)   kl = h_AN;

    if      (SpinP_switch==0) somax = 0;
    else if (SpinP_switch==1) somax = 1;
    else if (SpinP_switch==3) somax = 2;

    r = Dis[Gc_AN][kl];

    if (rcut<=r) {
      deri_dmp = 0.0;
      tmp = 0.0; 
    }
    else {
      deri_dmp = deri_dampingF(rcut,r);
      tmp = deri_dmp/dmp;
    }

    x0 = Gxyz[ig][1] + atv[Rni][1];
    y0 = Gxyz[ig][2] + atv[Rni][2];
    z0 = Gxyz[ig][3] + atv[Rni][3];

    x1 = Gxyz[jg][1] + atv[Rnj][1];
    y1 = Gxyz[jg][2] + atv[Rnj][2];
    z1 = Gxyz[jg][3] + atv[Rnj][3];

    tx = x0 - x1;
    ty = y0 - y1;
    tz = z0 - z1;

    /* for empty atoms or finite elemens basis */
    if (r<1.0e-10) r = 1.0e-10;

    if (h_AN==0 && q_AN!=0){
      dx = tmp*(x0-x1)/r;
      dy = tmp*(y0-y1)/r;
      dz = tmp*(z0-z1)/r;
    }

    else if (h_AN!=0 && q_AN==0){
      dx = tmp*(x1-x0)/r;
      dy = tmp*(y1-y0)/r;
      dz = tmp*(z1-z0)/r;
    }

    if (SpinP_switch==0 || SpinP_switch==1){

      if (h_AN==0){ 

        /* for forces */

        for (so=0; so<=somax; so++){
	  for (m=0; m<ian; m++){
	    for (n=0; n<jan; n++){
	      Hx[so][m][n].r += HNL[so][Mc_AN][kl][m][n]*dx;
	      Hy[so][m][n].r += HNL[so][Mc_AN][kl][m][n]*dy;
	      Hz[so][m][n].r += HNL[so][Mc_AN][kl][m][n]*dz;
	    }
	  }
        }

        /* for stress */

        for (so=0; so<=somax; so++){
	  for (m=0; m<ian; m++){
	    for (n=0; n<jan; n++){
	      H_sts[so][1][1][m][n].r += HNL[so][Mc_AN][kl][m][n]*dx*tx;
	      H_sts[so][1][2][m][n].r += HNL[so][Mc_AN][kl][m][n]*dx*ty;
	      H_sts[so][1][3][m][n].r += HNL[so][Mc_AN][kl][m][n]*dx*tz;
	      H_sts[so][2][1][m][n].r += HNL[so][Mc_AN][kl][m][n]*dy*tx;
	      H_sts[so][2][2][m][n].r += HNL[so][Mc_AN][kl][m][n]*dy*ty;
	      H_sts[so][2][3][m][n].r += HNL[so][Mc_AN][kl][m][n]*dy*tz;
	      H_sts[so][3][1][m][n].r += HNL[so][Mc_AN][kl][m][n]*dz*tx;
	      H_sts[so][3][2][m][n].r += HNL[so][Mc_AN][kl][m][n]*dz*ty;
	      H_sts[so][3][3][m][n].r += HNL[so][Mc_AN][kl][m][n]*dz*tz;
	    }
	  }
        }

      }

      else if (q_AN==0){ 

        /* for forces */

        for (so=0; so<=somax; so++){
	  for (m=0; m<ian; m++){
	    for (n=0; n<jan; n++){
	      Hx[so][m][n].r += HNL[so][Mc_AN][kl][n][m]*dx;
	      Hy[so][m][n].r += HNL[so][Mc_AN][kl][n][m]*dy;
	      Hz[so][m][n].r += HNL[so][Mc_AN][kl][n][m]*dz;
	    }
	  }
        }
      }
    }

    else if (SpinP_switch==3){

      if (h_AN==0){ 

        /* for forces */

        for (so=0; so<=somax; so++){
	  for (m=0; m<ian; m++){
	    for (n=0; n<jan; n++){
	      Hx[so][m][n].r +=  HNL[so][Mc_AN][kl][m][n]*dx;
	      Hy[so][m][n].r +=  HNL[so][Mc_AN][kl][m][n]*dy;
	      Hz[so][m][n].r +=  HNL[so][Mc_AN][kl][m][n]*dz;
	    }
	  }
        }

        /* for stress */

        for (so=0; so<=somax; so++){
	  for (m=0; m<ian; m++){
	    for (n=0; n<jan; n++){
	      H_sts[so][1][1][m][n].r += HNL[so][Mc_AN][kl][m][n]*dx*tx;
	      H_sts[so][1][2][m][n].r += HNL[so][Mc_AN][kl][m][n]*dx*ty;
	      H_sts[so][1][3][m][n].r += HNL[so][Mc_AN][kl][m][n]*dx*tz;
	      H_sts[so][2][1][m][n].r += HNL[so][Mc_AN][kl][m][n]*dy*tx;
	      H_sts[so][2][2][m][n].r += HNL[so][Mc_AN][kl][m][n]*dy*ty;
	      H_sts[so][2][3][m][n].r += HNL[so][Mc_AN][kl][m][n]*dy*tz;
	      H_sts[so][3][1][m][n].r += HNL[so][Mc_AN][kl][m][n]*dz*tx;
	      H_sts[so][3][2][m][n].r += HNL[so][Mc_AN][kl][m][n]*dz*ty;
	      H_sts[so][3][3][m][n].r += HNL[so][Mc_AN][kl][m][n]*dz*tz;
	    }
	  }
        }
      }

      else if (q_AN==0){ 

        /* for forces */

        for (so=0; so<=somax; so++){
	  for (m=0; m<ian; m++){
	    for (n=0; n<jan; n++){
	      Hx[so][m][n].r +=  HNL[so][Mc_AN][kl][n][m]*dx;
	      Hy[so][m][n].r +=  HNL[so][Mc_AN][kl][n][m]*dy;
	      Hz[so][m][n].r +=  HNL[so][Mc_AN][kl][n][m]*dz;
	    }
	  }
        }
      }

      if (SO_switch==1){

        if (h_AN==0){ 

          /* for forces */

	  for (so=0; so<=somax; so++){
	    for (m=0; m<ian; m++){
	      for (n=0; n<jan; n++){
		Hx[so][m][n].i += iHNL[so][Mc_AN][kl][m][n]*dx;
		Hy[so][m][n].i += iHNL[so][Mc_AN][kl][m][n]*dy;
		Hz[so][m][n].i += iHNL[so][Mc_AN][kl][m][n]*dz;
	      }
	    }
	  }

          /* for stress */

	  for (so=0; so<=somax; so++){
	    for (m=0; m<ian; m++){
	      for (n=0; n<jan; n++){
		H_sts[so][1][1][m][n].i += iHNL[so][Mc_AN][kl][m][n]*dx*tx;
		H_sts[so][1][2][m][n].i += iHNL[so][Mc_AN][kl][m][n]*dx*ty;
		H_sts[so][1][3][m][n].i += iHNL[so][Mc_AN][kl][m][n]*dx*tz;
		H_sts[so][2][1][m][n].i += iHNL[so][Mc_AN][kl][m][n]*dy*tx;
		H_sts[so][2][2][m][n].i += iHNL[so][Mc_AN][kl][m][n]*dy*ty;
		H_sts[so][2][3][m][n].i += iHNL[so][Mc_AN][kl][m][n]*dy*tz;
		H_sts[so][3][1][m][n].i += iHNL[so][Mc_AN][kl][m][n]*dz*tx;
		H_sts[so][3][2][m][n].i += iHNL[so][Mc_AN][kl][m][n]*dz*ty;
		H_sts[so][3][3][m][n].i += iHNL[so][Mc_AN][kl][m][n]*dz*tz;
	      }
	    }
	  }
	}

        else if (q_AN==0){ 

          /* for forces */

	  for (so=0; so<=somax; so++){
	    for (m=0; m<ian; m++){
	      for (n=0; n<jan; n++){
		Hx[so][m][n].i += iHNL[so][Mc_AN][kl][n][m]*dx;
		Hy[so][m][n].i += iHNL[so][Mc_AN][kl][n][m]*dy;
		Hz[so][m][n].i += iHNL[so][Mc_AN][kl][n][m]*dz;
	      }
	    }
	  }
	}

      }
    }
  }

}


void dHNL_SO(
	     double *sumx0r,
	     double *sumy0r, 
	     double *sumz0r, 
	     double *sumx1r,
	     double *sumy1r, 
	     double *sumz1r, 
	     double *sumx2r,
	     double *sumy2r, 
	     double *sumz2r, 
	     double *sumx0i,
	     double *sumy0i, 
	     double *sumz0i, 
	     double *sumx1i,
	     double *sumy1i, 
	     double *sumz1i, 
	     double *sumx2i,
	     double *sumy2i, 
	     double *sumz2i, 
             double fugou,
	     double PFp,
	     double PFm,
	     double ene_p,
	     double ene_m,
	     int l2, int *l,
	     int Mc_AN, int k,  int m,
	     int Mj_AN, int kl, int n,
	     double ******DS_NL1)
{

  int l3,i;
  double tmpx,tmpy,tmpz;
  double tmp0,tmp1,tmp2;
  double tmp3,tmp4,tmp5,tmp6;
  double deri[4];

  /****************************************************
    off-diagonal contribution to up-dn matrix
    for spin non-collinear
  ****************************************************/

  if (SpinP_switch==3){

    /* p */ 
    if (l2==2){

      /* real contribution of l+1/2 to off diagonal up-down matrix */ 
      tmpx = 
        fugou*
        ( ene_p/3.0*DS_NL1[0][1][Mc_AN][k][m][*l  ]*DS_NL1[0][0][Mj_AN][kl][n][*l+2]
         -ene_p/3.0*DS_NL1[0][1][Mc_AN][k][m][*l+2]*DS_NL1[0][0][Mj_AN][kl][n][*l  ] );      

      tmpy = 
        fugou*
        ( ene_p/3.0*DS_NL1[0][2][Mc_AN][k][m][*l  ]*DS_NL1[0][0][Mj_AN][kl][n][*l+2]
         -ene_p/3.0*DS_NL1[0][2][Mc_AN][k][m][*l+2]*DS_NL1[0][0][Mj_AN][kl][n][*l  ] );      

      tmpz = 
        fugou*
        ( ene_p/3.0*DS_NL1[0][3][Mc_AN][k][m][*l  ]*DS_NL1[0][0][Mj_AN][kl][n][*l+2]
         -ene_p/3.0*DS_NL1[0][3][Mc_AN][k][m][*l+2]*DS_NL1[0][0][Mj_AN][kl][n][*l  ] );      

      *sumx2r += tmpx;
      *sumy2r += tmpy;
      *sumz2r += tmpz;

      /* imaginary contribution of l+1/2 to off diagonal up-down matrix */ 

      tmpx = 
        fugou*
        ( -ene_p/3.0*DS_NL1[0][1][Mc_AN][k][m][*l+1]*DS_NL1[0][0][Mj_AN][kl][n][*l+2]
          +ene_p/3.0*DS_NL1[0][1][Mc_AN][k][m][*l+2]*DS_NL1[0][0][Mj_AN][kl][n][*l+1] );

      tmpy = 
        fugou*
        ( -ene_p/3.0*DS_NL1[0][2][Mc_AN][k][m][*l+1]*DS_NL1[0][0][Mj_AN][kl][n][*l+2]
          +ene_p/3.0*DS_NL1[0][2][Mc_AN][k][m][*l+2]*DS_NL1[0][0][Mj_AN][kl][n][*l+1] );

      tmpz = 
        fugou*
        ( -ene_p/3.0*DS_NL1[0][3][Mc_AN][k][m][*l+1]*DS_NL1[0][0][Mj_AN][kl][n][*l+2]
          +ene_p/3.0*DS_NL1[0][3][Mc_AN][k][m][*l+2]*DS_NL1[0][0][Mj_AN][kl][n][*l+1] );

      *sumx2i += tmpx;
      *sumy2i += tmpy;
      *sumz2i += tmpz;

       /* real contribution of l-1/2 for to diagonal up-down matrix */ 

      tmpx = 
        fugou*
        ( ene_m/3.0*DS_NL1[1][1][Mc_AN][k][m][*l  ]*DS_NL1[1][0][Mj_AN][kl][n][*l+2]
         -ene_m/3.0*DS_NL1[1][1][Mc_AN][k][m][*l+2]*DS_NL1[1][0][Mj_AN][kl][n][*l  ] );

      tmpy = 
        fugou*
        ( ene_m/3.0*DS_NL1[1][2][Mc_AN][k][m][*l  ]*DS_NL1[1][0][Mj_AN][kl][n][*l+2]
         -ene_m/3.0*DS_NL1[1][2][Mc_AN][k][m][*l+2]*DS_NL1[1][0][Mj_AN][kl][n][*l  ] );

      tmpz = 
        fugou*
        ( ene_m/3.0*DS_NL1[1][3][Mc_AN][k][m][*l  ]*DS_NL1[1][0][Mj_AN][kl][n][*l+2]
         -ene_m/3.0*DS_NL1[1][3][Mc_AN][k][m][*l+2]*DS_NL1[1][0][Mj_AN][kl][n][*l  ] );

      *sumx2r -= tmpx;
      *sumy2r -= tmpy;
      *sumz2r -= tmpz;

       /* imaginary contribution of l-1/2 to off diagonal up-down matrix */ 

      tmpx = 
        fugou*
        ( -ene_m/3.0*DS_NL1[1][1][Mc_AN][k][m][*l+1]*DS_NL1[1][0][Mj_AN][kl][n][*l+2]
          +ene_m/3.0*DS_NL1[1][1][Mc_AN][k][m][*l+2]*DS_NL1[1][0][Mj_AN][kl][n][*l+1] );   

      tmpy = 
        fugou*
        ( -ene_m/3.0*DS_NL1[1][2][Mc_AN][k][m][*l+1]*DS_NL1[1][0][Mj_AN][kl][n][*l+2]
          +ene_m/3.0*DS_NL1[1][2][Mc_AN][k][m][*l+2]*DS_NL1[1][0][Mj_AN][kl][n][*l+1] );   

      tmpz = 
        fugou*
        ( -ene_m/3.0*DS_NL1[1][3][Mc_AN][k][m][*l+1]*DS_NL1[1][0][Mj_AN][kl][n][*l+2]
          +ene_m/3.0*DS_NL1[1][3][Mc_AN][k][m][*l+2]*DS_NL1[1][0][Mj_AN][kl][n][*l+1] );   

      *sumx2i -= tmpx;
      *sumy2i -= tmpy;
      *sumz2i -= tmpz;

    }

    /* d */ 
    if (l2==4){

       tmp0 = sqrt(3.0);
       tmp1 = ene_p/5.0; 
       tmp2 = tmp0*tmp1;

       /* real contribution of l+1/2 to off diagonal up-down matrix */ 

       for (i=1; i<=3; i++){
         deri[i] = 
            fugou* 
            ( -tmp2*DS_NL1[0][i][Mc_AN][k][m][*l  ]*DS_NL1[0][0][Mj_AN][kl][n][*l+3]
              +tmp2*DS_NL1[0][i][Mc_AN][k][m][*l+3]*DS_NL1[0][0][Mj_AN][kl][n][*l  ]
              +tmp1*DS_NL1[0][i][Mc_AN][k][m][*l+1]*DS_NL1[0][0][Mj_AN][kl][n][*l+3]
              -tmp1*DS_NL1[0][i][Mc_AN][k][m][*l+3]*DS_NL1[0][0][Mj_AN][kl][n][*l+1]
              +tmp1*DS_NL1[0][i][Mc_AN][k][m][*l+2]*DS_NL1[0][0][Mj_AN][kl][n][*l+4]
              -tmp1*DS_NL1[0][i][Mc_AN][k][m][*l+4]*DS_NL1[0][0][Mj_AN][kl][n][*l+2] );
       }
       *sumx2r += deri[1];
       *sumy2r += deri[2];
       *sumz2r += deri[3];

       /* imaginary contribution of l+1/2 to off diagonal up-down matrix */ 

       for (i=1; i<=3; i++){
         deri[i] = 
            fugou* 
            ( +tmp2*DS_NL1[0][i][Mc_AN][k][m][*l  ]*DS_NL1[0][0][Mj_AN][kl][n][*l+4]
              -tmp2*DS_NL1[0][i][Mc_AN][k][m][*l+4]*DS_NL1[0][0][Mj_AN][kl][n][*l  ]
              +tmp1*DS_NL1[0][i][Mc_AN][k][m][*l+1]*DS_NL1[0][0][Mj_AN][kl][n][*l+4]
              -tmp1*DS_NL1[0][i][Mc_AN][k][m][*l+4]*DS_NL1[0][0][Mj_AN][kl][n][*l+1]
              -tmp1*DS_NL1[0][i][Mc_AN][k][m][*l+2]*DS_NL1[0][0][Mj_AN][kl][n][*l+3]
              +tmp1*DS_NL1[0][i][Mc_AN][k][m][*l+3]*DS_NL1[0][0][Mj_AN][kl][n][*l+2] );
       }
       *sumx2i += deri[1];
       *sumy2i += deri[2];
       *sumz2i += deri[3];

       /* real contribution of l-1/2 for to diagonal up-down matrix */ 

       tmp1 = ene_m/5.0; 
       tmp2 = tmp0*tmp1;

       for (i=1; i<=3; i++){
         deri[i] = 
            fugou* 
            ( -tmp2*DS_NL1[1][i][Mc_AN][k][m][*l  ]*DS_NL1[1][0][Mj_AN][kl][n][*l+3]
              +tmp2*DS_NL1[1][i][Mc_AN][k][m][*l+3]*DS_NL1[1][0][Mj_AN][kl][n][*l  ]
              +tmp1*DS_NL1[1][i][Mc_AN][k][m][*l+1]*DS_NL1[1][0][Mj_AN][kl][n][*l+3]
              -tmp1*DS_NL1[1][i][Mc_AN][k][m][*l+3]*DS_NL1[1][0][Mj_AN][kl][n][*l+1]
              +tmp1*DS_NL1[1][i][Mc_AN][k][m][*l+2]*DS_NL1[1][0][Mj_AN][kl][n][*l+4]
              -tmp1*DS_NL1[1][i][Mc_AN][k][m][*l+4]*DS_NL1[1][0][Mj_AN][kl][n][*l+2] );
       }
       *sumx2r -= deri[1];
       *sumy2r -= deri[2];
       *sumz2r -= deri[3];

       /* imaginary contribution of l-1/2 to off diagonal up-down matrix */ 

       for (i=1; i<=3; i++){
         deri[i] = 
            fugou* 
	    ( +tmp2*DS_NL1[1][i][Mc_AN][k][m][*l  ]*DS_NL1[1][0][Mj_AN][kl][n][*l+4]
              -tmp2*DS_NL1[1][i][Mc_AN][k][m][*l+4]*DS_NL1[1][0][Mj_AN][kl][n][*l  ]
              +tmp1*DS_NL1[1][i][Mc_AN][k][m][*l+1]*DS_NL1[1][0][Mj_AN][kl][n][*l+4]
              -tmp1*DS_NL1[1][i][Mc_AN][k][m][*l+4]*DS_NL1[1][0][Mj_AN][kl][n][*l+1]
              -tmp1*DS_NL1[1][i][Mc_AN][k][m][*l+2]*DS_NL1[1][0][Mj_AN][kl][n][*l+3]
	      +tmp1*DS_NL1[1][i][Mc_AN][k][m][*l+3]*DS_NL1[1][0][Mj_AN][kl][n][*l+2] );
       }
       *sumx2i -= deri[1];
       *sumy2i -= deri[2];
       *sumz2i -= deri[3];
    }

    /* f */ 
    if (l2==6){

      /* real contribution of l+1/2 to off diagonal up-down matrix */ 

      tmp0 = sqrt(6.0);
      tmp1 = sqrt(3.0/2.0);
      tmp2 = sqrt(5.0/2.0);

      tmp3 = ene_p/7.0; 
      tmp4 = tmp1*tmp3; /* sqrt(3.0/2.0) */
      tmp5 = tmp2*tmp3; /* sqrt(5.0/2.0) */
      tmp6 = tmp0*tmp3; /* sqrt(6.0)     */

       for (i=1; i<=3; i++){
         deri[i] = 
	   fugou*
             ( -tmp6*DS_NL1[0][i][Mc_AN][k][m][*l  ]*DS_NL1[0][0][Mj_AN][kl][n][*l+1]
               +tmp6*DS_NL1[0][i][Mc_AN][k][m][*l+1]*DS_NL1[0][0][Mj_AN][kl][n][*l  ]
               -tmp5*DS_NL1[0][i][Mc_AN][k][m][*l+1]*DS_NL1[0][0][Mj_AN][kl][n][*l+3]
               +tmp5*DS_NL1[0][i][Mc_AN][k][m][*l+3]*DS_NL1[0][0][Mj_AN][kl][n][*l+1]
               -tmp5*DS_NL1[0][i][Mc_AN][k][m][*l+2]*DS_NL1[0][0][Mj_AN][kl][n][*l+4]
	       +tmp5*DS_NL1[0][i][Mc_AN][k][m][*l+4]*DS_NL1[0][0][Mj_AN][kl][n][*l+2]
               -tmp4*DS_NL1[0][i][Mc_AN][k][m][*l+3]*DS_NL1[0][0][Mj_AN][kl][n][*l+5]
	       +tmp4*DS_NL1[0][i][Mc_AN][k][m][*l+5]*DS_NL1[0][0][Mj_AN][kl][n][*l+3]
               -tmp4*DS_NL1[0][i][Mc_AN][k][m][*l+4]*DS_NL1[0][0][Mj_AN][kl][n][*l+6]
               +tmp4*DS_NL1[0][i][Mc_AN][k][m][*l+6]*DS_NL1[0][0][Mj_AN][kl][n][*l+4] ); 
       }
       *sumx2r += deri[1];
       *sumy2r += deri[2];
       *sumz2r += deri[3];

       /* imaginary contribution of l+1/2 to off diagonal up-down matrix */ 

       for (i=1; i<=3; i++){
         deri[i] = 
	   fugou*
	     ( +tmp6*DS_NL1[0][i][Mc_AN][k][m][*l  ]*DS_NL1[0][0][Mj_AN][kl][n][*l+2]
               -tmp6*DS_NL1[0][i][Mc_AN][k][m][*l+2]*DS_NL1[0][0][Mj_AN][kl][n][*l  ]
               +tmp5*DS_NL1[0][i][Mc_AN][k][m][*l+1]*DS_NL1[0][0][Mj_AN][kl][n][*l+4]
               -tmp5*DS_NL1[0][i][Mc_AN][k][m][*l+4]*DS_NL1[0][0][Mj_AN][kl][n][*l+1]
               -tmp5*DS_NL1[0][i][Mc_AN][k][m][*l+2]*DS_NL1[0][0][Mj_AN][kl][n][*l+3]
	       +tmp5*DS_NL1[0][i][Mc_AN][k][m][*l+3]*DS_NL1[0][0][Mj_AN][kl][n][*l+2]
               +tmp4*DS_NL1[0][i][Mc_AN][k][m][*l+3]*DS_NL1[0][0][Mj_AN][kl][n][*l+6]
	       -tmp4*DS_NL1[0][i][Mc_AN][k][m][*l+6]*DS_NL1[0][0][Mj_AN][kl][n][*l+3]
               -tmp4*DS_NL1[0][i][Mc_AN][k][m][*l+4]*DS_NL1[0][0][Mj_AN][kl][n][*l+5]
               +tmp4*DS_NL1[0][i][Mc_AN][k][m][*l+5]*DS_NL1[0][0][Mj_AN][kl][n][*l+4] );
       }
       *sumx2i += deri[1];
       *sumy2i += deri[2];
       *sumz2i += deri[3];

       /* real contribution of l-1/2 for to diagonal up-down matrix */ 

       tmp3 = ene_m/7.0; 
       tmp4 = tmp1*tmp3; /* sqrt(3.0/2.0) */
       tmp5 = tmp2*tmp3; /* sqrt(5.0/2.0) */
       tmp6 = tmp0*tmp3; /* sqrt(6.0)     */

       for (i=1; i<=3; i++){
         deri[i] = 
	   fugou*
	     ( -tmp6*DS_NL1[1][i][Mc_AN][k][m][*l  ]*DS_NL1[1][0][Mj_AN][kl][n][*l+1]
               +tmp6*DS_NL1[1][i][Mc_AN][k][m][*l+1]*DS_NL1[1][0][Mj_AN][kl][n][*l  ]
               -tmp5*DS_NL1[1][i][Mc_AN][k][m][*l+1]*DS_NL1[1][0][Mj_AN][kl][n][*l+3]
               +tmp5*DS_NL1[1][i][Mc_AN][k][m][*l+3]*DS_NL1[1][0][Mj_AN][kl][n][*l+1]
               -tmp5*DS_NL1[1][i][Mc_AN][k][m][*l+2]*DS_NL1[1][0][Mj_AN][kl][n][*l+4]
	       +tmp5*DS_NL1[1][i][Mc_AN][k][m][*l+4]*DS_NL1[1][0][Mj_AN][kl][n][*l+2]
               -tmp4*DS_NL1[1][i][Mc_AN][k][m][*l+3]*DS_NL1[1][0][Mj_AN][kl][n][*l+5]
	       +tmp4*DS_NL1[1][i][Mc_AN][k][m][*l+5]*DS_NL1[1][0][Mj_AN][kl][n][*l+3]
               -tmp4*DS_NL1[1][i][Mc_AN][k][m][*l+4]*DS_NL1[1][0][Mj_AN][kl][n][*l+6]
               +tmp4*DS_NL1[1][i][Mc_AN][k][m][*l+6]*DS_NL1[1][0][Mj_AN][kl][n][*l+4] );
       }
       *sumx2r -= deri[1];
       *sumy2r -= deri[2];
       *sumz2r -= deri[3];

       /* imaginary contribution of l-1/2 to off diagonal up-down matrix */ 

       for (i=1; i<=3; i++){
         deri[i] = 
            fugou* 
	      ( +tmp6*DS_NL1[1][i][Mc_AN][k][m][*l  ]*DS_NL1[1][0][Mj_AN][kl][n][*l+2]
                -tmp6*DS_NL1[1][i][Mc_AN][k][m][*l+2]*DS_NL1[1][0][Mj_AN][kl][n][*l  ]
                +tmp5*DS_NL1[1][i][Mc_AN][k][m][*l+1]*DS_NL1[1][0][Mj_AN][kl][n][*l+4]
                -tmp5*DS_NL1[1][i][Mc_AN][k][m][*l+4]*DS_NL1[1][0][Mj_AN][kl][n][*l+1]
                -tmp5*DS_NL1[1][i][Mc_AN][k][m][*l+2]*DS_NL1[1][0][Mj_AN][kl][n][*l+3]
	        +tmp5*DS_NL1[1][i][Mc_AN][k][m][*l+3]*DS_NL1[1][0][Mj_AN][kl][n][*l+2]
                +tmp4*DS_NL1[1][i][Mc_AN][k][m][*l+3]*DS_NL1[1][0][Mj_AN][kl][n][*l+6]
	        -tmp4*DS_NL1[1][i][Mc_AN][k][m][*l+6]*DS_NL1[1][0][Mj_AN][kl][n][*l+3]
                -tmp4*DS_NL1[1][i][Mc_AN][k][m][*l+4]*DS_NL1[1][0][Mj_AN][kl][n][*l+5]
                +tmp4*DS_NL1[1][i][Mc_AN][k][m][*l+5]*DS_NL1[1][0][Mj_AN][kl][n][*l+4] );
       }
       *sumx2i -= deri[1];
       *sumy2i -= deri[2];
       *sumz2i -= deri[3];
	   
    }

  }

  /****************************************************
      off-diagonal contribution on up-up and dn-dn
  ****************************************************/

  /* p */ 
  if (l2==2){
 
    tmpx =
       fugou*
       ( ene_p/3.0*DS_NL1[0][1][Mc_AN][k][m][*l  ]*DS_NL1[0][0][Mj_AN][kl][n][*l+1]
        -ene_p/3.0*DS_NL1[0][1][Mc_AN][k][m][*l+1]*DS_NL1[0][0][Mj_AN][kl][n][*l  ] ); 

    tmpy =
       fugou*
       ( ene_p/3.0*DS_NL1[0][2][Mc_AN][k][m][*l  ]*DS_NL1[0][0][Mj_AN][kl][n][*l+1]
        -ene_p/3.0*DS_NL1[0][2][Mc_AN][k][m][*l+1]*DS_NL1[0][0][Mj_AN][kl][n][*l  ] ); 

    tmpz =
       fugou*
       ( ene_p/3.0*DS_NL1[0][3][Mc_AN][k][m][*l  ]*DS_NL1[0][0][Mj_AN][kl][n][*l+1]
        -ene_p/3.0*DS_NL1[0][3][Mc_AN][k][m][*l+1]*DS_NL1[0][0][Mj_AN][kl][n][*l  ] ); 

    /* contribution of l+1/2 for up spin */ 
    *sumx0i += -tmpx;
    *sumy0i += -tmpy;
    *sumz0i += -tmpz;

    /* contribution of l+1/2 for down spin */ 
    *sumx1i += tmpx;
    *sumy1i += tmpy;
    *sumz1i += tmpz;

    tmpx =
       fugou*
       ( ene_m/3.0*DS_NL1[1][1][Mc_AN][k][m][*l  ]*DS_NL1[1][0][Mj_AN][kl][n][*l+1]
        -ene_m/3.0*DS_NL1[1][1][Mc_AN][k][m][*l+1]*DS_NL1[1][0][Mj_AN][kl][n][*l  ] ); 

    tmpy =
       fugou*
       ( ene_m/3.0*DS_NL1[1][2][Mc_AN][k][m][*l  ]*DS_NL1[1][0][Mj_AN][kl][n][*l+1]
        -ene_m/3.0*DS_NL1[1][2][Mc_AN][k][m][*l+1]*DS_NL1[1][0][Mj_AN][kl][n][*l  ] ); 

    tmpz =
       fugou*
       ( ene_m/3.0*DS_NL1[1][3][Mc_AN][k][m][*l  ]*DS_NL1[1][0][Mj_AN][kl][n][*l+1]
        -ene_m/3.0*DS_NL1[1][3][Mc_AN][k][m][*l+1]*DS_NL1[1][0][Mj_AN][kl][n][*l  ] ); 

    /* contribution of l-1/2 for up spin */
    *sumx0i += tmpx;
    *sumy0i += tmpy;
    *sumz0i += tmpz;

    /* contribution of l+1/2 for down spin */ 
    *sumx1i += -tmpx;
    *sumy1i += -tmpy;
    *sumz1i += -tmpz;
  }

  /* d */ 
  else if (l2==4){

    tmpx =
       fugou*
       (
       ene_p*2.0/5.0*DS_NL1[0][1][Mc_AN][k][m][*l+1]*DS_NL1[0][0][Mj_AN][kl][n][*l+2]
      -ene_p*2.0/5.0*DS_NL1[0][1][Mc_AN][k][m][*l+2]*DS_NL1[0][0][Mj_AN][kl][n][*l+1]
      +ene_p*1.0/5.0*DS_NL1[0][1][Mc_AN][k][m][*l+3]*DS_NL1[0][0][Mj_AN][kl][n][*l+4]
      -ene_p*1.0/5.0*DS_NL1[0][1][Mc_AN][k][m][*l+4]*DS_NL1[0][0][Mj_AN][kl][n][*l+3] ); 

    tmpy =
       fugou*
       (
       ene_p*2.0/5.0*DS_NL1[0][2][Mc_AN][k][m][*l+1]*DS_NL1[0][0][Mj_AN][kl][n][*l+2]
      -ene_p*2.0/5.0*DS_NL1[0][2][Mc_AN][k][m][*l+2]*DS_NL1[0][0][Mj_AN][kl][n][*l+1]
      +ene_p*1.0/5.0*DS_NL1[0][2][Mc_AN][k][m][*l+3]*DS_NL1[0][0][Mj_AN][kl][n][*l+4]
      -ene_p*1.0/5.0*DS_NL1[0][2][Mc_AN][k][m][*l+4]*DS_NL1[0][0][Mj_AN][kl][n][*l+3] ); 

    tmpz =
       fugou*
       (
       ene_p*2.0/5.0*DS_NL1[0][3][Mc_AN][k][m][*l+1]*DS_NL1[0][0][Mj_AN][kl][n][*l+2]
      -ene_p*2.0/5.0*DS_NL1[0][3][Mc_AN][k][m][*l+2]*DS_NL1[0][0][Mj_AN][kl][n][*l+1]
      +ene_p*1.0/5.0*DS_NL1[0][3][Mc_AN][k][m][*l+3]*DS_NL1[0][0][Mj_AN][kl][n][*l+4]
      -ene_p*1.0/5.0*DS_NL1[0][3][Mc_AN][k][m][*l+4]*DS_NL1[0][0][Mj_AN][kl][n][*l+3] ); 

    /* contribution of l+1/2 for up spin */ 
    *sumx0i += -tmpx;
    *sumy0i += -tmpy;
    *sumz0i += -tmpz;

    /* contribution of l+1/2 for down spin */ 
    *sumx1i += tmpx;
    *sumy1i += tmpy;
    *sumz1i += tmpz;

    tmpx =
       fugou*
       (
       ene_m*2.0/5.0*DS_NL1[1][1][Mc_AN][k][m][*l+1]*DS_NL1[1][0][Mj_AN][kl][n][*l+2]
      -ene_m*2.0/5.0*DS_NL1[1][1][Mc_AN][k][m][*l+2]*DS_NL1[1][0][Mj_AN][kl][n][*l+1]
      +ene_m*1.0/5.0*DS_NL1[1][1][Mc_AN][k][m][*l+3]*DS_NL1[1][0][Mj_AN][kl][n][*l+4]
      -ene_m*1.0/5.0*DS_NL1[1][1][Mc_AN][k][m][*l+4]*DS_NL1[1][0][Mj_AN][kl][n][*l+3] ); 

    tmpy = 
       fugou*
       (
       ene_m*2.0/5.0*DS_NL1[1][2][Mc_AN][k][m][*l+1]*DS_NL1[1][0][Mj_AN][kl][n][*l+2]
      -ene_m*2.0/5.0*DS_NL1[1][2][Mc_AN][k][m][*l+2]*DS_NL1[1][0][Mj_AN][kl][n][*l+1]
      +ene_m*1.0/5.0*DS_NL1[1][2][Mc_AN][k][m][*l+3]*DS_NL1[1][0][Mj_AN][kl][n][*l+4]
      -ene_m*1.0/5.0*DS_NL1[1][2][Mc_AN][k][m][*l+4]*DS_NL1[1][0][Mj_AN][kl][n][*l+3] ); 

    tmpz =
       fugou*
       (
       ene_m*2.0/5.0*DS_NL1[1][3][Mc_AN][k][m][*l+1]*DS_NL1[1][0][Mj_AN][kl][n][*l+2]
      -ene_m*2.0/5.0*DS_NL1[1][3][Mc_AN][k][m][*l+2]*DS_NL1[1][0][Mj_AN][kl][n][*l+1]
      +ene_m*1.0/5.0*DS_NL1[1][3][Mc_AN][k][m][*l+3]*DS_NL1[1][0][Mj_AN][kl][n][*l+4]
      -ene_m*1.0/5.0*DS_NL1[1][3][Mc_AN][k][m][*l+4]*DS_NL1[1][0][Mj_AN][kl][n][*l+3] ); 

    /* contribution of l-1/2 for up spin */ 
    *sumx0i += tmpx;
    *sumy0i += tmpy;
    *sumz0i += tmpz;

    /* contribution of l-1/2 for down spin */ 
    *sumx1i += -tmpx;
    *sumy1i += -tmpy;
    *sumz1i += -tmpz;

  }
  
  /* f */ 
  else if (l2==6){

    tmpx =
       fugou*
       (
       ene_p*1.0/7.0*DS_NL1[0][1][Mc_AN][k][m][*l+1]*DS_NL1[0][0][Mj_AN][kl][n][*l+2]
      -ene_p*1.0/7.0*DS_NL1[0][1][Mc_AN][k][m][*l+2]*DS_NL1[0][0][Mj_AN][kl][n][*l+1]
      +ene_p*2.0/7.0*DS_NL1[0][1][Mc_AN][k][m][*l+3]*DS_NL1[0][0][Mj_AN][kl][n][*l+4]
      -ene_p*2.0/7.0*DS_NL1[0][1][Mc_AN][k][m][*l+4]*DS_NL1[0][0][Mj_AN][kl][n][*l+3]
      +ene_p*3.0/7.0*DS_NL1[0][1][Mc_AN][k][m][*l+5]*DS_NL1[0][0][Mj_AN][kl][n][*l+6]
      -ene_p*3.0/7.0*DS_NL1[0][1][Mc_AN][k][m][*l+6]*DS_NL1[0][0][Mj_AN][kl][n][*l+5] );

    tmpy = 
       fugou*
       (
       ene_p*1.0/7.0*DS_NL1[0][2][Mc_AN][k][m][*l+1]*DS_NL1[0][0][Mj_AN][kl][n][*l+2]
      -ene_p*1.0/7.0*DS_NL1[0][2][Mc_AN][k][m][*l+2]*DS_NL1[0][0][Mj_AN][kl][n][*l+1]
      +ene_p*2.0/7.0*DS_NL1[0][2][Mc_AN][k][m][*l+3]*DS_NL1[0][0][Mj_AN][kl][n][*l+4]
      -ene_p*2.0/7.0*DS_NL1[0][2][Mc_AN][k][m][*l+4]*DS_NL1[0][0][Mj_AN][kl][n][*l+3]
      +ene_p*3.0/7.0*DS_NL1[0][2][Mc_AN][k][m][*l+5]*DS_NL1[0][0][Mj_AN][kl][n][*l+6]
      -ene_p*3.0/7.0*DS_NL1[0][2][Mc_AN][k][m][*l+6]*DS_NL1[0][0][Mj_AN][kl][n][*l+5] );

    tmpz =
       fugou*
       (
       ene_p*1.0/7.0*DS_NL1[0][3][Mc_AN][k][m][*l+1]*DS_NL1[0][0][Mj_AN][kl][n][*l+2]
      -ene_p*1.0/7.0*DS_NL1[0][3][Mc_AN][k][m][*l+2]*DS_NL1[0][0][Mj_AN][kl][n][*l+1]
      +ene_p*2.0/7.0*DS_NL1[0][3][Mc_AN][k][m][*l+3]*DS_NL1[0][0][Mj_AN][kl][n][*l+4]
      -ene_p*2.0/7.0*DS_NL1[0][3][Mc_AN][k][m][*l+4]*DS_NL1[0][0][Mj_AN][kl][n][*l+3]
      +ene_p*3.0/7.0*DS_NL1[0][3][Mc_AN][k][m][*l+5]*DS_NL1[0][0][Mj_AN][kl][n][*l+6]
      -ene_p*3.0/7.0*DS_NL1[0][3][Mc_AN][k][m][*l+6]*DS_NL1[0][0][Mj_AN][kl][n][*l+5] );

    /* contribution of l+1/2 for up spin */ 
    *sumx0i += -tmpx;
    *sumy0i += -tmpy;
    *sumz0i += -tmpz;

    /* contribution of l+1/2 for down spin */ 
    *sumx1i += tmpx;
    *sumy1i += tmpy;
    *sumz1i += tmpz;

    tmpx =
       fugou*
       (
       ene_m*1.0/7.0*DS_NL1[1][1][Mc_AN][k][m][*l+1]*DS_NL1[1][0][Mj_AN][kl][n][*l+2]
      -ene_m*1.0/7.0*DS_NL1[1][1][Mc_AN][k][m][*l+2]*DS_NL1[1][0][Mj_AN][kl][n][*l+1]
      +ene_m*2.0/7.0*DS_NL1[1][1][Mc_AN][k][m][*l+3]*DS_NL1[1][0][Mj_AN][kl][n][*l+4]
      -ene_m*2.0/7.0*DS_NL1[1][1][Mc_AN][k][m][*l+4]*DS_NL1[1][0][Mj_AN][kl][n][*l+3]
      +ene_m*3.0/7.0*DS_NL1[1][1][Mc_AN][k][m][*l+5]*DS_NL1[1][0][Mj_AN][kl][n][*l+6]
      -ene_m*3.0/7.0*DS_NL1[1][1][Mc_AN][k][m][*l+6]*DS_NL1[1][0][Mj_AN][kl][n][*l+5] );

    tmpy =
       fugou*
       (
       ene_m*1.0/7.0*DS_NL1[1][2][Mc_AN][k][m][*l+1]*DS_NL1[1][0][Mj_AN][kl][n][*l+2]
      -ene_m*1.0/7.0*DS_NL1[1][2][Mc_AN][k][m][*l+2]*DS_NL1[1][0][Mj_AN][kl][n][*l+1]
      +ene_m*2.0/7.0*DS_NL1[1][2][Mc_AN][k][m][*l+3]*DS_NL1[1][0][Mj_AN][kl][n][*l+4]
      -ene_m*2.0/7.0*DS_NL1[1][2][Mc_AN][k][m][*l+4]*DS_NL1[1][0][Mj_AN][kl][n][*l+3]
      +ene_m*3.0/7.0*DS_NL1[1][2][Mc_AN][k][m][*l+5]*DS_NL1[1][0][Mj_AN][kl][n][*l+6]
      -ene_m*3.0/7.0*DS_NL1[1][2][Mc_AN][k][m][*l+6]*DS_NL1[1][0][Mj_AN][kl][n][*l+5] );

    tmpz =
       fugou*
       (
       ene_m*1.0/7.0*DS_NL1[1][3][Mc_AN][k][m][*l+1]*DS_NL1[1][0][Mj_AN][kl][n][*l+2]
      -ene_m*1.0/7.0*DS_NL1[1][3][Mc_AN][k][m][*l+2]*DS_NL1[1][0][Mj_AN][kl][n][*l+1]
      +ene_m*2.0/7.0*DS_NL1[1][3][Mc_AN][k][m][*l+3]*DS_NL1[1][0][Mj_AN][kl][n][*l+4]
      -ene_m*2.0/7.0*DS_NL1[1][3][Mc_AN][k][m][*l+4]*DS_NL1[1][0][Mj_AN][kl][n][*l+3]
      +ene_m*3.0/7.0*DS_NL1[1][3][Mc_AN][k][m][*l+5]*DS_NL1[1][0][Mj_AN][kl][n][*l+6]
      -ene_m*3.0/7.0*DS_NL1[1][3][Mc_AN][k][m][*l+6]*DS_NL1[1][0][Mj_AN][kl][n][*l+5] );

    /* contribution of l-1/2 for up spin */ 
    *sumx0i += tmpx;
    *sumy0i += tmpy;
    *sumz0i += tmpz;

    /* contribution of l-1/2 for down spin */ 
    *sumx1i += -tmpx;
    *sumy1i += -tmpy;
    *sumz1i += -tmpz;
  }

  /****************************************************
      diagonal contribution on up-up and dn-dn
  ****************************************************/

  for (l3=0; l3<=l2; l3++){

    /* VNL for j=l+1/2 */
              
    tmpx = PFp*ene_p*DS_NL1[0][1][Mc_AN][k][m][*l]*DS_NL1[0][0][Mj_AN][kl][n][*l];
    tmpy = PFp*ene_p*DS_NL1[0][2][Mc_AN][k][m][*l]*DS_NL1[0][0][Mj_AN][kl][n][*l];
    tmpz = PFp*ene_p*DS_NL1[0][3][Mc_AN][k][m][*l]*DS_NL1[0][0][Mj_AN][kl][n][*l];

    *sumx0r += tmpx;
    *sumy0r += tmpy;
    *sumz0r += tmpz;

    *sumx1r += tmpx;
    *sumy1r += tmpy;
    *sumz1r += tmpz;

    /* VNL for j=l-1/2 */
              
    tmpx = PFm*ene_m*DS_NL1[1][1][Mc_AN][k][m][*l]*DS_NL1[1][0][Mj_AN][kl][n][*l];
    tmpy = PFm*ene_m*DS_NL1[1][2][Mc_AN][k][m][*l]*DS_NL1[1][0][Mj_AN][kl][n][*l];
    tmpz = PFm*ene_m*DS_NL1[1][3][Mc_AN][k][m][*l]*DS_NL1[1][0][Mj_AN][kl][n][*l];

    *sumx0r += tmpx;
    *sumy0r += tmpy;
    *sumz0r += tmpz;

    *sumx1r += tmpx;
    *sumy1r += tmpy;
    *sumz1r += tmpz;

    *l = *l + 1;
  }
}


  

void MPI_OLP(double *****OLP1)
{
  int i,j,h_AN,Gh_AN,Hwan,n;
  int tno1,tno2,Mc_AN,Gc_AN,Cwan;
  int num,k,size1,size2;
  double *tmp_array;
  double *tmp_array2;
  int *Snd_S_Size,*Rcv_S_Size;
  int numprocs,myid,ID,IDS,IDR,tag=999;

  MPI_Status stat;
  MPI_Request request;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /****************************************************
    allocation of arrays:
  ****************************************************/

  Snd_S_Size = (int*)malloc(sizeof(int)*numprocs);
  Rcv_S_Size = (int*)malloc(sizeof(int)*numprocs);

  /******************************************************************
   MPI

   OLP[1], OLP[2], and OLP[3]

   note:

   OLP is used in DC and GDC method, where overlap integrals
   of Matomnum+MatomnumF+MatomnumS+1 are stored.
   However, overlap integrals of Matomnum+MatomnumF+1 are
   stored in Force.c. So, F_TopMAN should be used to refer
   overlap integrals in Force.c, while S_TopMAN should be
   used in DC and GDC routines.

   Although OLP is used in Eff_Hub_Pot.c, the usage is 
   consistent with that of DC and GDC routines by the following 
   reason:

   DC or GDC:      OLP + Spe_Total_NO   if no orbital optimization 
                CntOLP + Spe_Total_CNO  if orbital optimization

   Eff_Hub_Pot:    OLP + Spe_Total_NO   always since the U-potential
                                        affects to primitive orbital 

   If no orbital optimization, both the usages are consistent.
   If orbital optimization, CntOLP and OLP are used in DC(GDC) and 
   Eff_Hub_Pot.c, respectively. Therefore, there is no conflict. 
  *******************************************************************/

  /***********************************
             set data size
  ************************************/

  for (ID=0; ID<numprocs; ID++){

    IDS = (myid + ID) % numprocs;
    IDR = (myid - ID + numprocs) % numprocs;

    if (ID!=0){
      tag = 999;

      /* find data size to send block data */
      if (F_Snd_Num[IDS]!=0){
	size1 = 0;
	for (n=0; n<F_Snd_Num[IDS]; n++){
	  Mc_AN = Snd_MAN[IDS][n];
	  Gc_AN = Snd_GAN[IDS][n];
	  Cwan = WhatSpecies[Gc_AN]; 
	  tno1 = Spe_Total_NO[Cwan];
	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	    Gh_AN = natn[Gc_AN][h_AN];        
	    Hwan = WhatSpecies[Gh_AN];
	    tno2 = Spe_Total_NO[Hwan];
            size1 += 4*tno1*tno2;
	  }
	}

	Snd_S_Size[IDS] = size1;
	MPI_Isend(&size1, 1, MPI_INT, IDS, tag, mpi_comm_level1, &request);
      }
      else{
	Snd_S_Size[IDS] = 0;
      }

      /* receiving of size of data */
 
      if (F_Rcv_Num[IDR]!=0){
	MPI_Recv(&size2, 1, MPI_INT, IDR, tag, mpi_comm_level1, &stat);
	Rcv_S_Size[IDR] = size2;
      }
      else{
	Rcv_S_Size[IDR] = 0;
      }

      if (F_Snd_Num[IDS]!=0) MPI_Wait(&request,&stat);
    }
    else{
      Snd_S_Size[IDS] = 0;
      Rcv_S_Size[IDR] = 0;
    }
  }

  /***********************************
               data transfer
  ************************************/

  tag = 999;
  for (ID=0; ID<numprocs; ID++){

    IDS = (myid + ID) % numprocs;
    IDR = (myid - ID + numprocs) % numprocs;

    if (ID!=0){

      /*****************************
                sending of data 
      *****************************/

      if (F_Snd_Num[IDS]!=0){

	size1 = Snd_S_Size[IDS];

	/* allocation of array */

	tmp_array = (double*)malloc(sizeof(double)*size1);

	/* multidimentional array to vector array */

	num = 0;

        for (k=0; k<=3; k++){
  	  for (n=0; n<F_Snd_Num[IDS]; n++){
	    Mc_AN = Snd_MAN[IDS][n];
	    Gc_AN = Snd_GAN[IDS][n];
	    Cwan = WhatSpecies[Gc_AN]; 
	    tno1 = Spe_Total_NO[Cwan];
	    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	      Gh_AN = natn[Gc_AN][h_AN];        
	      Hwan = WhatSpecies[Gh_AN];
	      tno2 = Spe_Total_NO[Hwan];
	      for (i=0; i<tno1; i++){
		for (j=0; j<tno2; j++){
		  tmp_array[num] = OLP1[k][Mc_AN][h_AN][i][j];
		  num++;
		} 
	      } 
	    }
	  }
	}

	MPI_Isend(&tmp_array[0], size1, MPI_DOUBLE, IDS, tag, mpi_comm_level1, &request);
      }

      /*****************************
           receiving of block data
      *****************************/

      if (F_Rcv_Num[IDR]!=0){
          
	size2 = Rcv_S_Size[IDR];
        
	/* allocation of array */
	tmp_array2 = (double*)malloc(sizeof(double)*size2);
         
	MPI_Recv(&tmp_array2[0], size2, MPI_DOUBLE, IDR, tag, mpi_comm_level1, &stat);

	num = 0;

        for (k=0; k<=3; k++){
	  Mc_AN = F_TopMAN[IDR] - 1; /* F_TopMAN should be used. */
  	  for (n=0; n<F_Rcv_Num[IDR]; n++){
	    Mc_AN++;
	    Gc_AN = Rcv_GAN[IDR][n];
	    Cwan = WhatSpecies[Gc_AN]; 
	    tno1 = Spe_Total_NO[Cwan];

	    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	      Gh_AN = natn[Gc_AN][h_AN];        
	      Hwan = WhatSpecies[Gh_AN];
	      tno2 = Spe_Total_NO[Hwan];
	      for (i=0; i<tno1; i++){
		for (j=0; j<tno2; j++){
		  OLP1[k][Mc_AN][h_AN][i][j] = tmp_array2[num];
		  num++;
		}
	      }
	    }
	  }        
	}

	/* freeing of array */
	free(tmp_array2);
      }

      if (F_Snd_Num[IDS]!=0){
	MPI_Wait(&request,&stat);
	free(tmp_array); /* freeing of array */
      }
    }
  }
 
  /****************************************************
    freeing of arrays:
  ****************************************************/

  free(Snd_S_Size);
  free(Rcv_S_Size);
}




void EH0_TwoCenter(int Gc_AN, int h_AN, double VH0ij[4])
{ 
  int n1,ban;
  int Gh_AN,Rn,wan1,wan2;
  double dv,x,y,z,r,r2,xx,va0,rho0,dr_va0;
  double z2,sum,sumr,sumx,sumy,sumz,wt;

  Gh_AN = natn[Gc_AN][h_AN];
  Rn = ncn[Gc_AN][h_AN];
  wan1 = WhatSpecies[Gc_AN];
  ban = Spe_Spe2Ban[wan1];
  wan2 = WhatSpecies[Gh_AN];
  dv = dv_EH0[ban];
  
  sum = 0.0;
  sumr = 0.0;

  for (n1=0; n1<TGN_EH0[ban]; n1++){
    x = GridX_EH0[ban][n1];
    y = GridY_EH0[ban][n1];
    z = GridZ_EH0[ban][n1];
    rho0 = Arho_EH0[ban][n1];
    wt = Wt_EH0[ban][n1];
    z2 = z - Dis[Gc_AN][h_AN];
    r2 = x*x + y*y + z2*z2;
    r = sqrt(r2);
    xx = 0.5*log(r2);

    /* for empty atoms or finite elemens basis */
    if (r<1.0e-10) r = 1.0e-10;

    va0 = VH_AtomF(wan2, 
                   Spe_Num_Mesh_VPS[wan2], xx, r, 
                   Spe_VPS_XV[wan2], Spe_VPS_RV[wan2], Spe_VH_Atom[wan2]);

    sum += wt*va0*rho0;

    if (h_AN!=0 && 1.0e-14<r){
      dr_va0 = Dr_VH_AtomF(wan2, 
                           Spe_Num_Mesh_VPS[wan2], xx, r, 
                           Spe_VPS_XV[wan2], Spe_VPS_RV[wan2], Spe_VH_Atom[wan2]);

      sumr -= wt*rho0*dr_va0*z2/r;
    }
  }

  sum  = sum*dv;

  if (h_AN!=0){

    /* for empty atoms or finite elemens basis */
    r = Dis[Gc_AN][h_AN];
    if (r<1.0e-10) r = 1.0e-10;

    x = Gxyz[Gc_AN][1] - (Gxyz[Gh_AN][1] + atv[Rn][1]);
    y = Gxyz[Gc_AN][2] - (Gxyz[Gh_AN][2] + atv[Rn][2]);
    z = Gxyz[Gc_AN][3] - (Gxyz[Gh_AN][3] + atv[Rn][3]);
    sumr = sumr*dv;
    sumx = sumr*x/r;
    sumy = sumr*y/r;
    sumz = sumr*z/r;
  }
  else{
    sumx = 0.0;
    sumy = 0.0;
    sumz = 0.0;
  }

  VH0ij[0] = sum;
  VH0ij[1] = sumx;
  VH0ij[2] = sumy;
  VH0ij[3] = sumz;
}



void EH0_TwoCenter_at_Cutoff(int wan1, int wan2, double VH0ij[4])
{ 
  int n1,ban;
  double dv,x,y,z,r1,r2,va0,rho0,dr_va0,rcut;
  double z2,sum,sumr,sumx,sumy,sumz,wt,r,xx;
  /* for OpenMP */
  int OMPID,Nthrds,Nthrds0,Nprocs,Nloop;
  double *my_sum_threads;

  ban = Spe_Spe2Ban[wan1];
  dv  = dv_EH0[ban];

  rcut = Spe_Atom_Cut1[wan1] + Spe_Atom_Cut1[wan2];

  /* get Nthrds0 */  
#pragma omp parallel shared(Nthrds0)
  {
    Nthrds0 = omp_get_num_threads();
  }

  /* allocation of array */
  my_sum_threads = (double*)malloc(sizeof(double)*Nthrds0);

  for (Nloop=0; Nloop<Nthrds0; Nloop++){
    my_sum_threads[Nloop] = 0.0;
  }

#pragma omp parallel shared(Spe_VH_Atom,Spe_VPS_XV,Spe_VPS_RV,Spe_Num_Mesh_VPS,wan2,Wt_EH0,my_sum_threads,rcut,Arho_EH0,GridZ_EH0,GridY_EH0,GridX_EH0,TGN_EH0,ban) private(n1,OMPID,Nthrds,Nprocs,x,y,z,rho0,wt,z2,r2,va0,r,xx)
  {
    /* get info. on OpenMP */

    OMPID = omp_get_thread_num();
    Nthrds = omp_get_num_threads();
    Nprocs = omp_get_num_procs();

    for (n1=OMPID*TGN_EH0[ban]/Nthrds; n1<(OMPID+1)*TGN_EH0[ban]/Nthrds; n1++){

      x = GridX_EH0[ban][n1];
      y = GridY_EH0[ban][n1];
      z = GridZ_EH0[ban][n1];
      rho0 = Arho_EH0[ban][n1];
      wt = Wt_EH0[ban][n1];
      z2 = z - rcut;
      r2 = x*x + y*y + z2*z2;
      r = sqrt(r2);
      xx = 0.5*log(r2);

      va0 = VH_AtomF(wan2, 
                     Spe_Num_Mesh_VPS[wan2], xx, r, 
                     Spe_VPS_XV[wan2], Spe_VPS_RV[wan2], Spe_VH_Atom[wan2]);

      my_sum_threads[OMPID] += wt*va0*rho0;
    }

  } /* #pragma omp parallel */

  sum  = 0.0;
  for (Nloop=0; Nloop<Nthrds0; Nloop++){
    sum += my_sum_threads[Nloop];
  }

  sum  = sum*dv;
  sumx = 0.0;
  sumy = 0.0;
  sumz = 0.0;

  VH0ij[0] = sum;
  VH0ij[1] = sumx;
  VH0ij[2] = sumy;
  VH0ij[3] = sumz;

  /* freeing of array */
  free(my_sum_threads);
}







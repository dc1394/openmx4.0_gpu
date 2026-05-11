/**********************************************************************
  Calc_LNBO.c:

   Calc_LNBO.c is a subroutine to calculate strictly localized non-orthogonal 
   natural bond orbitals on selected two atoms.

  Log of Calc_LNBO.c:

     09/Oct./2022  Released by T. Ozaki

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "mpi.h"
#include "openmx_common.h"
#include "lapack_prototypes.h"
#include <omp.h>

#define  measure_time   0

static double LNBO_Col_Diag(double ****OLP0, double *****Hks, double *****CDM);

double Calc_LNBO( double ****OLP0, double *****Hks, double *****CDM)
{
  double time0;

  /****************************************************
                    for collinear DFT
  ****************************************************/

  if ( SpinP_switch==0 || SpinP_switch==1){

    time0 = LNBO_Col_Diag(OLP0, Hks, CDM);
  }

  else if ( SpinP_switch==3 ){
    printf("The calculation of LNBO is not supported for the non-collinear calculation.\n");fflush(stdout);
    MPI_Finalize();
    exit(0);
  }

  return time0;
}






static double LNBO_Col_Diag(double ****OLP0, double *****Hks, double *****CDM)
{
  int i,j,k,l,n,Mc_AN,Gc_AN,h_AN,Mh_AN,Gh_AN,h_AN2,h_AN3;
  int Cwan,num,wan1,wan2,wan3,tno,tno0,tno1,tno2,tno3,spin;
  int l1,l2,l3,AN1,AN2,AN3,Rnh;
  int Nloop,po,p,q,size1,wan0,NO1,NO2,**RMI0;
  char *JOBVL,*JOBVR;
  int N,A,LDA,LDVL,LDVR,SDIM,LWORK,INFO,*IWORK;
  double ****Sbond,*****Hbond,*****DMbond;
  double ***DMS,*WR,*WI,*VL,*VR,*WORK,RCONDE,RCONDV;
  double *B,**S0,***H0,*Vec,sum,sum0,F;
  double *snd_array;
  double TStime,TEtime;
  int ID,IDS,IDR,myid,numprocs,tag=999;
  MPI_Status stat;
  MPI_Request request;

  dtime(&TStime);

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /********************************************
             allocation of arrays
  ********************************************/

  RMI0 = (int**)malloc(sizeof(int*)*List_YOUSO[2]);
  for (i=0; i<List_YOUSO[2]; i++){
    RMI0[i] = (int*)malloc(sizeof(int)*List_YOUSO[2]);
    for (j=0; j<List_YOUSO[2]; j++){
      RMI0[i][j] = 0;
    }    
  }

  Sbond = (double****)malloc(sizeof(double***)*2);
  for (k=0; k<2; k++){
    Sbond[k] = (double***)malloc(sizeof(double**)*List_YOUSO[2]);
    for (l=0; l<List_YOUSO[2]; l++){
      Sbond[k][l] = (double**)malloc(sizeof(double*)*List_YOUSO[7]);
      for (i=0; i<List_YOUSO[7]; i++){
        Sbond[k][l][i] = (double*)malloc(sizeof(double)*List_YOUSO[7]);
        for (j=0; j<List_YOUSO[7]; j++) Sbond[k][l][i][j] = 0.0;
      }
    }
  }

  Hbond = (double*****)malloc(sizeof(double****)*2);
  for (k=0; k<2; k++){
    Hbond[k] = (double****)malloc(sizeof(double***)*2);
    for (spin=0; spin<2; spin++){
      Hbond[k][spin] = (double***)malloc(sizeof(double**)*List_YOUSO[2]);
      for (l=0; l<List_YOUSO[2]; l++){
        Hbond[k][spin][l] = (double**)malloc(sizeof(double*)*List_YOUSO[7]);
	for (i=0; i<List_YOUSO[7]; i++){
	  Hbond[k][spin][l][i] = (double*)malloc(sizeof(double)*List_YOUSO[7]);
          for (j=0; j<List_YOUSO[7]; j++) Hbond[k][spin][l][i][j] = 0.0;
	}
      }
    }
  }

  DMbond = (double*****)malloc(sizeof(double****)*2);
  for (k=0; k<2; k++){
    DMbond[k] = (double****)malloc(sizeof(double***)*2);
    for (spin=0; spin<2; spin++){
      DMbond[k][spin] = (double***)malloc(sizeof(double**)*List_YOUSO[2]);
      for (l=0; l<List_YOUSO[2]; l++){
        DMbond[k][spin][l] = (double**)malloc(sizeof(double*)*List_YOUSO[7]);
	for (i=0; i<List_YOUSO[7]; i++){
	  DMbond[k][spin][l][i] = (double*)malloc(sizeof(double)*List_YOUSO[7]);
          for (j=0; j<List_YOUSO[7]; j++) DMbond[k][spin][l][i][j] = 0.0;
	}
      }
    }
  }

  DMS = (double***)malloc(sizeof(double**)*LNBO_num);
  for (p=0; p<LNBO_num; p++){
    DMS[p] = (double**)malloc(sizeof(double*)*(SpinP_switch+1));
    for (spin=0; spin<=SpinP_switch; spin++){
      DMS[p][spin] = (double*)malloc(sizeof(double)*4*List_YOUSO[7]*List_YOUSO[7]);
      for (i=0; i<4*List_YOUSO[7]*List_YOUSO[7]; i++) DMS[p][spin][i] = 0.0;
    }
  }

  S0 = (double**)malloc(sizeof(double*)*LNBO_num);
  for (p=0; p<LNBO_num; p++){
    S0[p] = (double*)malloc(sizeof(double)*4*List_YOUSO[7]*List_YOUSO[7]);
    for (i=0; i<4*List_YOUSO[7]*List_YOUSO[7]; i++) S0[p][i] = 0.0;
  }

  H0 = (double***)malloc(sizeof(double**)*LNBO_num);
  for (p=0; p<LNBO_num; p++){
    H0[p] = (double**)malloc(sizeof(double*)*(SpinP_switch+1));
    for (spin=0; spin<(SpinP_switch+1); spin++){
      H0[p][spin] = (double*)malloc(sizeof(double)*4*List_YOUSO[7]*List_YOUSO[7]);
      for (i=0; i<4*List_YOUSO[7]*List_YOUSO[7]; i++) H0[p][spin][i] = 0.0;
    }
  }

  Vec = (double*)malloc(sizeof(double)*2*List_YOUSO[7]);

  WR = (double*)malloc(sizeof(double)*2*List_YOUSO[7]);
  WI = (double*)malloc(sizeof(double)*2*List_YOUSO[7]);
  VL = (double*)malloc(sizeof(double)*List_YOUSO[7]*List_YOUSO[7]*8);
  VR = (double*)malloc(sizeof(double)*List_YOUSO[7]*List_YOUSO[7]*8);

  WORK = (double*)malloc(sizeof(double)*List_YOUSO[7]*20);
  IWORK = (int*)malloc(sizeof(int)*List_YOUSO[7]*2);

  B = (double*)malloc(sizeof(double)*4*List_YOUSO[7]*List_YOUSO[7]);

  if (alloc_first[39]==0){

    for (p=0; p<LNBO_num; p++){
      for (spin=0; spin<=SpinP_switch; spin++){
        free(LNBO_coes[p][spin]);
      }
      free(LNBO_coes[p]);
    }  
    free(LNBO_coes);

    for (p=0; p<LNBO_num; p++){
      for (spin=0; spin<=SpinP_switch; spin++){
        free(LNBO_pops[p][spin]);
      }
      free(LNBO_pops[p]);
    }  
    free(LNBO_pops);

    for (p=0; p<LNBO_num; p++){
      for (spin=0; spin<=SpinP_switch; spin++){
        free(LNBO_H[p][spin]);
      }
      free(LNBO_H[p]);
    }  
    free(LNBO_H);

    alloc_first[39] = 1;
  } 

  if (alloc_first[39]==1){

    LNBO_coes = (double***)malloc(sizeof(double**)*LNBO_num);
    for (p=0; p<LNBO_num; p++){
      LNBO_coes[p] = (double**)malloc(sizeof(double*)*(SpinP_switch+1));
      for (spin=0; spin<=SpinP_switch; spin++){
        LNBO_coes[p][spin] = (double*)malloc(sizeof(double)*List_YOUSO[7]*List_YOUSO[7]*4);
      }
    }  

    LNBO_pops = (double***)malloc(sizeof(double**)*LNBO_num);
    for (p=0; p<LNBO_num; p++){
      LNBO_pops[p] = (double**)malloc(sizeof(double*)*(SpinP_switch+1));
      for (spin=0; spin<=SpinP_switch; spin++){
        LNBO_pops[p][spin] = (double*)malloc(sizeof(double)*List_YOUSO[7]*2);
      }
    }  

    LNBO_H = (double***)malloc(sizeof(double**)*LNBO_num);
    for (p=0; p<LNBO_num; p++){
      LNBO_H[p] = (double**)malloc(sizeof(double*)*(SpinP_switch+1));
      for (spin=0; spin<=SpinP_switch; spin++){
        LNBO_H[p][spin] = (double*)malloc(sizeof(double)*List_YOUSO[7]*2);
      }
    }  
  
    alloc_first[39] = 0;
  }

  /********************************************
            calculations of LNBOs
  ********************************************/

  for (p=0; p<LNBO_num; p++){

    Gc_AN = LNBO_Atoms[p][0];

    wan0 = WhatSpecies[Gc_AN];
    NO1 = Spe_Total_CNO[wan0];
    Gc_AN = LNBO_Atoms[p][1];
    wan0 = WhatSpecies[Gc_AN];
    NO2 = Spe_Total_CNO[wan0];

    for (spin=0; spin<=SpinP_switch; spin++){
      for (i=0; i<(NO1+NO2); i++){
	for (j=0; j<(NO1+NO2); j++){
	  DMS[p][spin][(NO1+NO2)*j+i] = 0.0;
	} 
      }
    } 

    /***********************************************************
         MPI of information related to LNBO_Atoms[p][0,1]
    ***********************************************************/

    for (q=0; q<2; q++){

      Gc_AN = LNBO_Atoms[p][q];
      wan1 = WhatSpecies[Gc_AN];
      tno1 = Spe_Total_CNO[wan1];
      ID = G2ID[Gc_AN];

      size1 = 0;  
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	Gh_AN = natn[Gc_AN][h_AN];        
	wan2 = WhatSpecies[Gh_AN];
	tno2 = Spe_Total_CNO[wan2];
	size1 += tno1*tno2;
      }

      size1 = 3*size1*(SpinP_switch+1);
      snd_array = (double*)malloc(sizeof(double)*size1);

      if (myid==ID){

	Mc_AN = F_G2M[Gc_AN];
	k = 0; 

	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];        
	  wan2 = WhatSpecies[Gh_AN];
	  tno2 = Spe_Total_CNO[wan2];
	  for (i=0; i<tno1; i++){
	    for (j=0; j<tno2; j++){
	      snd_array[k] = OLP0[Mc_AN][h_AN][i][j];  
	      k++;
	    } 
	  } 
	}

	for (spin=0; spin<=SpinP_switch; spin++){
	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	    Gh_AN = natn[Gc_AN][h_AN];        
	    wan2 = WhatSpecies[Gh_AN];
	    tno2 = Spe_Total_CNO[wan2];
	    for (i=0; i<tno1; i++){
	      for (j=0; j<tno2; j++){
		snd_array[k] = Hks[spin][Mc_AN][h_AN][i][j];  
		k++;
	      } 
	    } 
	  }
	}

	for (spin=0; spin<=SpinP_switch; spin++){
	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	    Gh_AN = natn[Gc_AN][h_AN];        
	    wan2 = WhatSpecies[Gh_AN];
	    tno2 = Spe_Total_CNO[wan2];
	    for (i=0; i<tno1; i++){
	      for (j=0; j<tno2; j++){
		snd_array[k] = CDM[spin][Mc_AN][h_AN][i][j];  
		k++;
	      } 
	    } 
	  }
	}

      } /* if (myid==ID) */ 

      MPI_Bcast(snd_array, size1, MPI_DOUBLE, ID, mpi_comm_level1);

      /* store the data to Sbond, Hbond, and DMbond */

      k = 0; 
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	Gh_AN = natn[Gc_AN][h_AN];        
	wan2 = WhatSpecies[Gh_AN];
	tno2 = Spe_Total_CNO[wan2];
	for (i=0; i<tno1; i++){
	  for (j=0; j<tno2; j++){
	    Sbond[q][h_AN][i][j] = snd_array[k];
	    k++;
	  } 
	} 
      }

      for (spin=0; spin<=SpinP_switch; spin++){
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];        
	  wan2 = WhatSpecies[Gh_AN];
	  tno2 = Spe_Total_CNO[wan2];
	  for (i=0; i<tno1; i++){
	    for (j=0; j<tno2; j++){
  	      Hbond[q][spin][h_AN][i][j] = snd_array[k];
	      k++;
	    } 
	  } 
	}
      }

      for (spin=0; spin<=SpinP_switch; spin++){
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];        
	  wan2 = WhatSpecies[Gh_AN];
	  tno2 = Spe_Total_CNO[wan2];
	  for (i=0; i<tno1; i++){
	    for (j=0; j<tno2; j++){
  	      DMbond[q][spin][h_AN][i][j] = snd_array[k];
	      k++;
	    } 
	  } 
	}
      }

    } /* q */    

    free(snd_array);

    /***********************************************************
          calculation of on-site DMS defined by DM*S 
    ***********************************************************/

    if (myid==Host_ID){

      for (q=0; q<2; q++){

        Gc_AN = LNBO_Atoms[p][q];
        wan1 = WhatSpecies[Gc_AN];
        tno1 = Spe_Total_CNO[wan1];

	for (spin=0; spin<=SpinP_switch; spin++){

	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	    Gh_AN = natn[Gc_AN][h_AN];
	    wan2 = WhatSpecies[Gh_AN];
	    tno2 = Spe_Total_CNO[wan2];

	    for (i=0; i<tno1; i++){
	      for (j=0; j<tno1; j++){

		sum = 0.0;
		for (k=0; k<tno2; k++){
		  sum += DMbond[q][spin][h_AN][i][k]*Sbond[q][h_AN][j][k];
		}

		DMS[p][spin][(NO1+NO2)*(j+q*NO1)+(i+q*NO1)] += sum; 
	      }
	    }

	  } /* h_AN */

	  /* store the diagonal block element of H0 */

	  for (i=0; i<tno1; i++){
	    for (j=0; j<tno1; j++){
	      H0[p][spin][(NO1+NO2)*(j+q*NO1)+(i+q*NO1)] = Hbond[q][spin][0][i][j]; 
	    }
	  }

	} /* spin */

        /* store the diagonal block element of S0 */

	for (i=0; i<tno1; i++){
	  for (j=0; j<tno1; j++){
            S0[p][(NO1+NO2)*(j+q*NO1)+(i+q*NO1)] = Sbond[q][0][i][j]; 
	  }
	}

      } /* q */

      /*
      printf("ABC1 DMS\n");fflush(stdout);
      for (i=0; i<(NO1+NO2); i++){
        for (j=0; j<(NO1+NO2); j++){
          printf("%7.3f ",DMS[p][0][(NO1+NO2)*j+i]);fflush(stdout);
        } 
        printf("\n");fflush(stdout);
      } 
      */

    } /* if (myid==Host_ID) */ 

    /***********************************************************
          calculation of off-site DMS defined by DM*S 
    ***********************************************************/

    for (q=0; q<2; q++){

      /* MPI of RMI1 */    

      if (q==0){ 
        AN1 = LNBO_Atoms[p][0];
        AN2 = LNBO_Atoms[p][1];
      }
      else{
        AN1 = LNBO_Atoms[p][1];
        AN2 = LNBO_Atoms[p][0];
      }  

      ID = G2ID[AN1];

      if (myid==ID){
	Mc_AN = F_G2M[AN1];
	for (i=0; i<=FNAN[AN1]; i++){
	  for (j=0; j<=FNAN[AN1]; j++){
	    RMI0[i][j] = RMI1[Mc_AN][i][j];
	  }
	}
      }
      else {
	for (i=0; i<List_YOUSO[2]; i++){
	  for (j=0; j<List_YOUSO[2]; j++){
	    RMI0[i][j] = 0;
	  }
	}
      } 

      for (i=0; i<=FNAN[AN1]; i++){
	MPI_Allreduce(MPI_IN_PLACE, &RMI0[i][0], FNAN[AN1]+1, MPI_INT, MPI_SUM, mpi_comm_level1);
      }    

      /* find h_AN3 */ 

      h_AN3 = -1; 

      for (h_AN=0; h_AN<=FNAN[AN1]; h_AN++){
        Gh_AN = natn[AN1][h_AN];        
        Rnh = ncn[AN1][h_AN];

        if (q==0){
	  l1 = atv_ijk[Rnh][1];
	  l2 = atv_ijk[Rnh][2];
	  l3 = atv_ijk[Rnh][3];
        }
        else if (q==1){
	  l1 = -atv_ijk[Rnh][1];
	  l2 = -atv_ijk[Rnh][2];
	  l3 = -atv_ijk[Rnh][3];
        }

        if ( AN2==Gh_AN && LNBO_Atoms[p][2]==l1 && LNBO_Atoms[p][3]==l2 && LNBO_Atoms[p][4]==l3 ){
          h_AN3 = h_AN;
        }
      }             

      /* calculation of off-site DMS defined by DM*S */

      if (0<=h_AN3 && myid==Host_ID){

	wan1 = WhatSpecies[AN1];
	tno1 = Spe_Total_CNO[wan1];
	wan2 = WhatSpecies[AN2];
	tno2 = Spe_Total_CNO[wan2];

	for (spin=0; spin<=SpinP_switch; spin++){
	  for (h_AN=0; h_AN<=FNAN[AN1]; h_AN++){

	    AN3 = natn[AN1][h_AN];
	    wan3 = WhatSpecies[AN3];
	    tno3 = Spe_Total_CNO[wan3];
            h_AN2 = RMI0[h_AN3][h_AN]; 

            if (0<=h_AN2){

	      for (i=0; i<tno1; i++){
		for (j=0; j<tno2; j++){

		  if (q==0){ 

		    sum = 0.0;
		    for (k=0; k<tno3; k++){
		      sum += DMbond[0][spin][h_AN][i][k]*Sbond[1][h_AN2][j][k];
		    }

		    DMS[p][spin][(NO1+NO2)*(j+NO1)+i] += sum; 
		  }

		  else if (q==1){ 

		    sum = 0.0;
		    for (k=0; k<tno3; k++){
		      sum += DMbond[1][spin][h_AN][i][k]*Sbond[0][h_AN2][j][k];
		    }

		    DMS[p][spin][(NO1+NO2)*j+(i+NO1)] += sum; 
		  }

		} /* j */
	      } /* i */ 
            } /* if (0<=h_AN2) */

	  } /* h_AN */

	  /* store the off-diagonal block element of H0 */

	  if (q==0){ 
	    for (i=0; i<tno1; i++){
	      for (j=0; j<tno2; j++){
		H0[p][spin][(NO1+NO2)*(j+1*NO1)+(i+0*NO1)] = Hbond[q][spin][h_AN3][i][j]; 
	      }
	    }
	  }

	  else if (q==1){ 
	    for (i=0; i<tno1; i++){
	      for (j=0; j<tno2; j++){
		H0[p][spin][(NO1+NO2)*(j+0*NO1)+(i+1*NO1)] = Hbond[q][spin][h_AN3][i][j]; 
	      }
	    }
	  }

	} /* spin */

        /* store the off-diagonal block element of S0 */

        if (q==0){ 
	  for (i=0; i<tno1; i++){
	    for (j=0; j<tno2; j++){
              S0[p][(NO1+NO2)*(j+1*NO1)+(i+0*NO1)] = Sbond[q][h_AN3][i][j]; 
	    }
	  }
	}

        else if (q==1){ 
	  for (i=0; i<tno1; i++){
	    for (j=0; j<tno2; j++){
              S0[p][(NO1+NO2)*(j+0*NO1)+(i+1*NO1)] = Sbond[q][h_AN3][i][j]; 
	    }
	  }
        }

      } /* if (0<=h_AN3 && myid==Host_ID) */

    } /* q */

    /*
    if (myid==Host_ID){
      printf("ABC2 DMS\n");fflush(stdout);
      for (i=0; i<(NO1+NO2); i++){
	for (j=0; j<(NO1+NO2); j++){
	  printf("%7.3f ",DMS[p][0][(NO1+NO2)*j+i]);fflush(stdout);
	} 
	printf("\n");fflush(stdout);
      } 
    }
    */

  } /* p */

  /********************************************
            diagonalization of DMS
  ********************************************/

  for (p=0; p<LNBO_num; p++){

    AN1 = LNBO_Atoms[p][0];
    AN2 = LNBO_Atoms[p][1];
    wan1 = WhatSpecies[AN1];
    tno1 = Spe_Total_CNO[wan1];
    wan2 = WhatSpecies[AN2];
    tno2 = Spe_Total_CNO[wan2];

    tno = tno1 + tno2;

    for (spin=0; spin<=SpinP_switch; spin++){

      /* call the dgeev routine in lapack */

      JOBVL = "V";
      JOBVR = "V";
      N = tno;
      LDA = tno;
      LDVL = tno*2;
      LDVR = tno*2;
      LWORK = tno*10;

      for (i=0; i<tno*tno; i++) B[i] = DMS[p][spin][i];

      F77_NAME(dgeev,DGEEV)( JOBVL, JOBVR, &N, B, &LDA, WR, WI, VL, &LDVL, VR, &LDVR, 
                             WORK, &LWORK, &INFO );

      if (INFO!=0){
        printf("warning: INFO=%2d in calling dgeev in a function 'Calc_LNBO'\n",INFO);
      }

      /* ordering the eigenvalues and the orthogonal matrix */

      for (i=0; i<tno; i++) IWORK[i] = i;
      qsort_double_int2(tno,WR,IWORK);

      /* calculations of Frobenius norm */

      if (0 && myid==0){

	for (i=0; i<tno; i++){
	  for (j=0; j<tno; j++){
	    B[j*tno+i] = 0.0; 
	  }
	}     

        for (k=0; k<tno; k++){
          l = IWORK[k];

	  for (i=0; i<tno; i++){
	    for (j=0; j<tno; j++){
              B[j*tno+i] += VR[LDVR*l+i]*WR[k]*VL[LDVL*l+j];
	    }
	  }     
	}

        printf("S0 p=%2d spin=%2d Gc_AN=%2d\n",p,spin,Gc_AN);
	for (i=0; i<tno; i++){
	  for (j=0; j<tno; j++){
            printf("%10.6f ",S0[p][tno*j+i]);
	  }
          printf("\n");
	}

        printf("DMS p=%2d spin=%2d Gc_AN=%2d\n",p,spin,Gc_AN);
	for (i=0; i<tno; i++){
	  for (j=0; j<tno; j++){
            printf("%10.6f ",DMS[p][spin][tno*j+i]);
	  }
          printf("\n");
	}

        printf("B spin=%2d Gc_AN=%2d\n",spin,Gc_AN);
	for (i=0; i<tno; i++){
	  for (j=0; j<tno; j++){
            printf("%10.6f ",B[tno*j+i]);
	  }
          printf("\n");
	}
      }

      /* copy VR to LNBO_coes, where vectors in LNBO_coes are stored in column */

      for (j=0; j<tno; j++){

        k = IWORK[j];
	for (i=0; i<tno; i++){
	  LNBO_coes[p][spin][tno*j+i] = VR[LDVR*k+i];
	}

        /* normalization with S */

	for (i=0; i<tno; i++){

          sum = 0.0;
  	  for (k=0; k<tno; k++){
            sum += S0[p][tno*i+k]*LNBO_coes[p][spin][tno*j+k];
          }
          Vec[i] = sum;
        }

        sum = 0.0;
	for (k=0; k<tno; k++){
          sum += LNBO_coes[p][spin][tno*j+k]*Vec[k];
        } 
        sum = 1.0/sqrt(sum);

	for (i=0; i<tno; i++){
	  LNBO_coes[p][spin][tno*j+i] *= sum;
	}

        /* calculation of <LNBO|H|LNBO> */

	for (i=0; i<tno; i++){

          sum = 0.0;
  	  for (k=0; k<tno; k++){
            sum += H0[p][spin][tno*i+k]*LNBO_coes[p][spin][tno*j+k];
          }
          Vec[i] = sum;
        }

        sum = 0.0;
	for (k=0; k<tno; k++){
          sum += LNBO_coes[p][spin][tno*j+k]*Vec[k];
        } 

        LNBO_H[p][spin][j] = sum;        

      } /* j */

      /* store the eigenvalues */

      for (i=0; i<tno; i++){
        LNBO_pops[p][spin][i] = WR[i];
      }

      if (0 && myid==0){

        printf("ABC1 p=%2d\n",p);

	for (i=0; i<tno; i++){
	  printf("ABC myid=%2d spin=%2d Mc_AN=%2d i=%2d IWORK=%2d WR=%15.11f WI=%15.11f\n",
                      myid,spin,Mc_AN,i,IWORK[i],WR[i],WI[i]);fflush(stdout);
	}

        printf("QQQ myid=%2d p=%2d spin=%2d\n",myid,p,spin);fflush(stdout);

	printf("WWW1 myid=%2d LNBO_coes p=%2d spin=%2d\n",myid,p,spin);fflush(stdout);
	for (i=0; i<tno; i++){
	  for (j=0; j<tno; j++){
	    printf("%10.5f ",LNBO_coes[p][spin][tno*j+i]);fflush(stdout);
	  }
	  printf("\n");fflush(stdout);
	}

        printf("Check orthogonalization\n"); 
	for (i=0; i<tno; i++){
	  for (j=0; j<tno; j++){
 
            sum = 0.0;
	    for (k=0; k<tno; k++){
              sum += VL[LDVL*i+k]*VR[LDVR*j+k];
  	    }
            printf("%10.5f ",sum);
	  }
	  printf("\n");fflush(stdout);
	}
      }

      /* MPI of LNBO_coes */ 

      MPI_Bcast(&LNBO_coes[p][spin][0], tno*tno, MPI_DOUBLE, 0, mpi_comm_level1);

    } /* spin */
  } /* p */

  /********************************************
          save the result in *.lnao.
  ********************************************/

  if (myid==Host_ID){

    int m,mul,base;
    FILE *fp;
    char file[YOUSO10];
    char *Name_Angular[20][10];
    char *Name_Multiple[20];

    Name_Angular[0][0] = "s          ";
    Name_Angular[1][0] = "px         ";
    Name_Angular[1][1] = "py         ";
    Name_Angular[1][2] = "pz         ";
    Name_Angular[2][0] = "d3z^2-r^2  ";
    Name_Angular[2][1] = "dx^2-y^2   ";
    Name_Angular[2][2] = "dxy        ";
    Name_Angular[2][3] = "dxz        ";
    Name_Angular[2][4] = "dyz        ";
    Name_Angular[3][0] = "f5z^2-3r^2 ";
    Name_Angular[3][1] = "f5xz^2-xr^2";
    Name_Angular[3][2] = "f5yz^2-yr^2";
    Name_Angular[3][3] = "fzx^2-zy^2 ";
    Name_Angular[3][4] = "fxyz       ";
    Name_Angular[3][5] = "fx^3-3*xy^2";
    Name_Angular[3][6] = "f3yx^2-y^3 ";
    Name_Angular[4][0] = "g1         ";
    Name_Angular[4][1] = "g2         ";
    Name_Angular[4][2] = "g3         ";
    Name_Angular[4][3] = "g4         ";
    Name_Angular[4][4] = "g5         ";
    Name_Angular[4][5] = "g6         ";
    Name_Angular[4][6] = "g7         ";
    Name_Angular[4][7] = "g8         ";
    Name_Angular[4][8] = "g9         ";

    Name_Multiple[0] = " 0";
    Name_Multiple[1] = " 1";
    Name_Multiple[2] = " 2";
    Name_Multiple[3] = " 3";
    Name_Multiple[4] = " 4";
    Name_Multiple[5] = " 5";

    sprintf(file,"%s%s.LNBO",filepath,filename);

    if ((fp = fopen(file,"w")) != NULL){

      fprintf(fp,"\n");
      fprintf(fp,"***********************************************************\n");
      fprintf(fp,"***********************************************************\n");
      fprintf(fp,"          Localized Natural Bond Orbitals (LNBOs)          \n");
      fprintf(fp,"***********************************************************\n");
      fprintf(fp,"***********************************************************\n\n");

      for (p=0; p<LNBO_num; p++){
	for (spin=0; spin<=SpinP_switch; spin++){


	  AN1 = LNBO_Atoms[p][0];
	  AN2 = LNBO_Atoms[p][1];
	  wan1 = WhatSpecies[AN1];
	  tno1 = Spe_Total_CNO[wan1];
	  wan2 = WhatSpecies[AN2];
	  tno2 = Spe_Total_CNO[wan2];
	  tno = tno1 + tno2;

	  base = 8;

	  fprintf(fp,"%2d  %2s: Atomic Number=%2d  %2s: Atomic Number=%2d  spin=%2d\n",p+1,SpeName[wan1],AN1,SpeName[wan2],AN2,spin);

	  for (k=0; k<((tno-1)/base+1); k++){

	    fprintf(fp,"\n");
	    fprintf(fp,"                          ");
	    for (i=k*base; i<(k*base+base); i++){
	      if (i<tno){ 
		fprintf(fp," %3d     ",i+1);
	      }
	    }
	    fprintf(fp,"\n");
	    fprintf(fp,"        Population      ");

	    for (i=k*base; i<(k*base+base); i++){
	      if (i<tno){ 
		fprintf(fp,"%8.4f ",LNBO_pops[p][spin][i]);
	      }
	    }

	    fprintf(fp,"\n");
	    fprintf(fp,"      <LNBO|H|LNBO>     ");

	    for (i=k*base; i<(k*base+base); i++){
	      if (i<tno){ 
		fprintf(fp,"%8.4f ",LNBO_H[p][spin][i]);
	      }
	    }

	    fprintf(fp,"\n");
	    fprintf(fp,"       Angular     Radial\n");

            /* first atom */

            i = 0;
	    for (l=0; l<=Supported_MaxL; l++){
	      for (mul=0; mul<Spe_Num_Basis[wan1][l]; mul++){
		for (m=0; m<(2*l+1); m++){

                  if (i==0)
  		    fprintf(fp," %2d  %2s  %s%s  ",AN1,SpeName[wan1],Name_Angular[l][m],Name_Multiple[mul]);
                  else 
  		    fprintf(fp,"         %s%s  ",Name_Angular[l][m],Name_Multiple[mul]);

		  for (j=k*base; j<(k*base+base); j++){
		    if (j<tno){ 
		      fprintf(fp,"%8.4f ",LNBO_coes[p][spin][tno*j+i]);
		    }
		  }

		  fprintf(fp,"\n");
		  i++;
		}
	      }
	    }

            /* second atom */

	    for (l=0; l<=Supported_MaxL; l++){
	      for (mul=0; mul<Spe_Num_Basis[wan2][l]; mul++){
		for (m=0; m<(2*l+1); m++){

                  if (i==tno1)
  		    fprintf(fp," %2d  %2s  %s%s  ",AN2,SpeName[wan2],Name_Angular[l][m],Name_Multiple[mul]);
                  else 
  		    fprintf(fp,"         %s%s  ",Name_Angular[l][m],Name_Multiple[mul]);

		  for (j=k*base; j<(k*base+base); j++){
		    if (j<tno){ 
		      fprintf(fp,"%8.4f ",LNBO_coes[p][spin][tno*j+i]);
		    }
		  }

		  fprintf(fp,"\n");
		  i++;
		}
	      }
	    }

	  } 
          fprintf(fp,"\n");
	}
      }

      fclose(fp);
    }

    else{
      printf("Could not open %s\n",file);
    }

  } /* end of if (myid==Host_ID) */

/*
  printf("ABC100\n");
  MPI_Finalize();
  exit(0);
*/

  /********************************************
             freeing of arrays
  ********************************************/

  for (i=0; i<List_YOUSO[2]; i++){
    free(RMI0[i]);
  }
  free(RMI0);

  for (k=0; k<2; k++){
    for (l=0; l<List_YOUSO[2]; l++){
      for (i=0; i<List_YOUSO[7]; i++){
        free(Sbond[k][l][i]);
      }
      free(Sbond[k][l]);
    }
    free(Sbond[k]);
  }
  free(Sbond);

  for (k=0; k<2; k++){
    for (spin=0; spin<2; spin++){
      for (l=0; l<List_YOUSO[2]; l++){
	for (i=0; i<List_YOUSO[7]; i++){
	  free(Hbond[k][spin][l][i]);
	}
        free(Hbond[k][spin][l]);
      }
      free(Hbond[k][spin]);
    }
    free(Hbond[k]);
  }
  free(Hbond);

  for (k=0; k<2; k++){
    for (spin=0; spin<2; spin++){
      for (l=0; l<List_YOUSO[2]; l++){
	for (i=0; i<List_YOUSO[7]; i++){
	  free(DMbond[k][spin][l][i]);
	}
        free(DMbond[k][spin][l]);
      }
      free(DMbond[k][spin]);
    }
    free(DMbond[k]);
  }
  free(DMbond);

  for (p=0; p<LNBO_num; p++){
    for (spin=0; spin<=SpinP_switch; spin++){
      free(DMS[p][spin]);
    }
    free(DMS[p]);
  }
  free(DMS);

  for (p=0; p<LNBO_num; p++){
    free(S0[p]);
  }
  free(S0);

  for (p=0; p<LNBO_num; p++){
    for (spin=0; spin<(SpinP_switch+1); spin++){
      free(H0[p][spin]);
    }
    free(H0[p]);
  }
  free(H0);

  free(Vec);

  free(WR);
  free(WI);
  free(VL);
  free(VR);
  free(WORK);
  free(IWORK);
  free(B);

  /* for time */
  dtime(&TEtime);

  return (TEtime-TStime);
}

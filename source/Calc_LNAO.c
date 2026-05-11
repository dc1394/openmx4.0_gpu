/**********************************************************************
  Calc_LNAO.c:

   Calc_LNAO.c is a subroutine to calculate strictly localized non-orthogonal 
   natural atomic orbitals on selected atoms.

  Log of Calc_LNAO.c:

     06/Oct./2022  Released by T. Ozaki

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

static double LNAO_Col_Diag(double ****OLP0, double *****Hks, double *****CDM);

double Calc_LNAO( double ****OLP0, double *****Hks, double *****CDM)
{
  double time0;

  /****************************************************
                    for collinear DFT
  ****************************************************/

  if ( SpinP_switch==0 || SpinP_switch==1){

    time0 = LNAO_Col_Diag(OLP0, Hks, CDM);
  }

  else if ( SpinP_switch==3 ){
    printf("The calculation of LNAO is not supported for the non-collinear calculation.\n");fflush(stdout);
    MPI_Finalize();
    exit(0);
  }

  return time0;
}






static double LNAO_Col_Diag(double ****OLP0, double *****Hks, double *****CDM)
{
  int i,j,k,l,n,Mc_AN,Gc_AN,h_AN,Mh_AN,Gh_AN;
  int Cwan,num,wan1,wan2,tno0,tno1,tno2,spin;
  int Nloop,po,p,q;
  char *JOBVL,*JOBVR;
  int N,A,LDA,LDVL,LDVR,SDIM,LWORK,INFO,*IWORK;
  double ***DMS,*WR,*WI,*VL,*VR,*WORK,RCONDE,RCONDV;
  double *B,**S0,***H0,*Vec,sum,sum0,F;
  double TStime,TEtime;
  int ID,IDS,IDR,myid,numprocs,tag=999;
  int size1,size2;
  MPI_Status stat;
  MPI_Request request;

  dtime(&TStime);

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /********************************************
             allocation of arrays
  ********************************************/

  DMS = (double***)malloc(sizeof(double**)*LNAO_num);
  for (p=0; p<LNAO_num; p++){
    DMS[p] = (double**)malloc(sizeof(double*)*(SpinP_switch+1));
    for (spin=0; spin<=SpinP_switch; spin++){
      DMS[p][spin] = (double*)malloc(sizeof(double)*List_YOUSO[7]*List_YOUSO[7]);
      for (i=0; i<List_YOUSO[7]*List_YOUSO[7]; i++) DMS[p][spin][i] = 0.0;
    }
  }

  S0 = (double**)malloc(sizeof(double*)*LNAO_num);
  for (p=0; p<LNAO_num; p++){
    S0[p] = (double*)malloc(sizeof(double)*List_YOUSO[7]*List_YOUSO[7]);
    for (i=0; i<List_YOUSO[7]*List_YOUSO[7]; i++) S0[p][i] = 0.0;
  }

  H0 = (double***)malloc(sizeof(double**)*LNAO_num);
  for (p=0; p<LNAO_num; p++){
    H0[p] = (double**)malloc(sizeof(double*)*(SpinP_switch+1));
    for (spin=0; spin<(SpinP_switch+1); spin++){
      H0[p][spin] = (double*)malloc(sizeof(double)*List_YOUSO[7]*List_YOUSO[7]);
      for (i=0; i<List_YOUSO[7]*List_YOUSO[7]; i++) H0[p][spin][i] = 0.0;
    }
  }

  Vec = (double*)malloc(sizeof(double)*List_YOUSO[7]);

  WR = (double*)malloc(sizeof(double)*List_YOUSO[7]);
  WI = (double*)malloc(sizeof(double)*List_YOUSO[7]);
  VL = (double*)malloc(sizeof(double)*List_YOUSO[7]*List_YOUSO[7]*2);
  VR = (double*)malloc(sizeof(double)*List_YOUSO[7]*List_YOUSO[7]*2);

  WORK = (double*)malloc(sizeof(double)*List_YOUSO[7]*10);
  IWORK = (int*)malloc(sizeof(int)*List_YOUSO[7]);

  B = (double*)malloc(sizeof(double)*List_YOUSO[7]*List_YOUSO[7]);

  if (alloc_first[38]==0){

    for (p=0; p<LNAO_num; p++){
      for (spin=0; spin<=SpinP_switch; spin++){
        free(LNAO_coes[p][spin]);
      }
      free(LNAO_coes[p]);
    }  
    free(LNAO_coes);

    for (p=0; p<LNAO_num; p++){
      for (spin=0; spin<=SpinP_switch; spin++){
        free(LNAO_pops[p][spin]);
      }
      free(LNAO_pops[p]);
    }  
    free(LNAO_pops);

    for (p=0; p<LNAO_num; p++){
      for (spin=0; spin<=SpinP_switch; spin++){
        free(LNAO_H[p][spin]);
      }
      free(LNAO_H[p]);
    }  
    free(LNAO_H);

    alloc_first[38] = 1;
  } 

  if (alloc_first[38]==1){

    LNAO_coes = (double***)malloc(sizeof(double**)*LNAO_num);
    for (p=0; p<LNAO_num; p++){
      LNAO_coes[p] = (double**)malloc(sizeof(double*)*(SpinP_switch+1));
      for (spin=0; spin<=SpinP_switch; spin++){
        LNAO_coes[p][spin] = (double*)malloc(sizeof(double)*List_YOUSO[7]*List_YOUSO[7]);
      }
    }  

    LNAO_pops = (double***)malloc(sizeof(double**)*LNAO_num);
    for (p=0; p<LNAO_num; p++){
      LNAO_pops[p] = (double**)malloc(sizeof(double*)*(SpinP_switch+1));
      for (spin=0; spin<=SpinP_switch; spin++){
        LNAO_pops[p][spin] = (double*)malloc(sizeof(double)*List_YOUSO[7]);
      }
    }  

    LNAO_H = (double***)malloc(sizeof(double**)*LNAO_num);
    for (p=0; p<LNAO_num; p++){
      LNAO_H[p] = (double**)malloc(sizeof(double*)*(SpinP_switch+1));
      for (spin=0; spin<=SpinP_switch; spin++){
        LNAO_H[p][spin] = (double*)malloc(sizeof(double)*List_YOUSO[7]);
      }
    }  
  
    alloc_first[38] = 0;
  }

  /********************************************
       calculation of DMS defined by DM*S
  ********************************************/

  for (p=0; p<LNAO_num; p++){
    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

      Gc_AN = M2G[Mc_AN];
      wan1 = WhatSpecies[Gc_AN];
      tno1 = Spe_Total_CNO[wan1];
     
      if (LNAO_Atoms[p]==Gc_AN){ 

	for (spin=0; spin<=SpinP_switch; spin++){

	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	    Gh_AN = natn[Gc_AN][h_AN];
	    wan2 = WhatSpecies[Gh_AN];
	    tno2 = Spe_Total_CNO[wan2];

	    for (i=0; i<tno1; i++){
	      for (j=0; j<tno1; j++){

		sum = 0.0;
		for (k=0; k<tno2; k++){
		  sum += CDM[spin][Mc_AN][h_AN][i][k]*OLP0[Mc_AN][h_AN][j][k];
		}

		DMS[p][spin][tno1*j+i] += sum; 
	      }
	    }
	  }

	  /* store the onsite hamiltonian integrals */

	  for (i=0; i<tno1; i++){
	    for (j=0; j<tno1; j++){
	      H0[p][spin][tno1*j+i] = Hks[spin][Mc_AN][0][i][j]; 
	    } 
	  }

	} /* spin */

        /* store the onsite overlap integrals */

        for (i=0; i<tno1; i++){
          for (j=0; j<tno1; j++){
            S0[p][tno1*j+i] = OLP0[Mc_AN][0][i][j]; 
          } 
        }

      } /* if (LNAO_Atoms[p]==Gc_AN) */
    } /* Mc_AN */
  } /* p */

  /* MPI of DMS, H0, and S0 */

  for (p=0; p<LNAO_num; p++){
    for (spin=0; spin<=SpinP_switch; spin++){
      MPI_Allreduce(MPI_IN_PLACE, &DMS[p][spin][0], List_YOUSO[7]*List_YOUSO[7], MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
      MPI_Allreduce(MPI_IN_PLACE, &H0[p][spin][0], List_YOUSO[7]*List_YOUSO[7], MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
    }
    MPI_Allreduce(MPI_IN_PLACE, &S0[p][0], List_YOUSO[7]*List_YOUSO[7], MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
  }

  /********************************************
            diagonalization of DMS
  ********************************************/

  for (p=0; p<LNAO_num; p++){

    Gc_AN = LNAO_Atoms[p];
    wan1 = WhatSpecies[Gc_AN];
    tno1 = Spe_Total_CNO[wan1];

    for (spin=0; spin<=SpinP_switch; spin++){

      /* call the dgeev routine in lapack */

      JOBVL = "V";
      JOBVR = "V";
      N = tno1;
      LDA = tno1;
      LDVL = tno1*2;
      LDVR = tno1*2;
      LWORK = tno1*10;

      for (i=0; i<tno1*tno1; i++) B[i] = DMS[p][spin][i];

      F77_NAME(dgeev,DGEEV)( JOBVL, JOBVR, &N, B, &LDA, WR, WI, VL, &LDVL, VR, &LDVR, 
                             WORK, &LWORK, &INFO );

      if (INFO!=0){
        printf("warning: INFO=%2d in calling dgeev in a function 'Calc_LNAO'\n",INFO);
      }

      /* ordering the eigenvalues and the orthogonal matrix */

      for (i=0; i<tno1; i++) IWORK[i] = i;
      qsort_double_int2(tno1,WR,IWORK);

      /* calculations of Frobenius norm */

      if (0 && myid==0){

	for (i=0; i<tno1; i++){
	  for (j=0; j<tno1; j++){
	    B[j*tno1+i] = 0.0; 
	  }
	}     

        for (k=0; k<tno1; k++){
          l = IWORK[k];

	  for (i=0; i<tno1; i++){
	    for (j=0; j<tno1; j++){
              B[j*tno1+i] += VR[LDVR*l+i]*WR[k]*VL[LDVL*l+j];
	    }
	  }     
	}

        printf("S0 p=%2d spin=%2d Gc_AN=%2d\n",p,spin,Gc_AN);
	for (i=0; i<tno1; i++){
	  for (j=0; j<tno1; j++){
            printf("%10.6f ",S0[p][tno1*j+i]);
	  }
          printf("\n");
	}

        printf("DMS p=%2d spin=%2d Gc_AN=%2d\n",p,spin,Gc_AN);
	for (i=0; i<tno1; i++){
	  for (j=0; j<tno1; j++){
            printf("%10.6f ",DMS[p][spin][tno1*j+i]);
	  }
          printf("\n");
	}

        printf("B spin=%2d Gc_AN=%2d\n",spin,Gc_AN);
	for (i=0; i<tno1; i++){
	  for (j=0; j<tno1; j++){
            printf("%10.6f ",B[tno1*j+i]);
	  }
          printf("\n");
	}
      }

      /* copy VR to LNAO_coes, where vectors in LNAO_coes are stored in column */

      for (j=0; j<tno1; j++){

        k = IWORK[j];
	for (i=0; i<tno1; i++){
	  LNAO_coes[p][spin][tno1*j+i] = VR[LDVR*k+i];
	}

        /* normalization with S */

	for (i=0; i<tno1; i++){

          sum = 0.0;
  	  for (k=0; k<tno1; k++){
            sum += S0[p][tno1*i+k]*LNAO_coes[p][spin][tno1*j+k];
          }
          Vec[i] = sum;
        }

        sum = 0.0;
	for (k=0; k<tno1; k++){
          sum += LNAO_coes[p][spin][tno1*j+k]*Vec[k];
        } 
        sum = 1.0/sqrt(sum);

	for (i=0; i<tno1; i++){
	  LNAO_coes[p][spin][tno1*j+i] *= sum;
	}

        /* calculation of <LNAO|H|LNAO> */

	for (i=0; i<tno1; i++){

          sum = 0.0;
  	  for (k=0; k<tno1; k++){
            sum += H0[p][spin][tno1*i+k]*LNAO_coes[p][spin][tno1*j+k];
          }
          Vec[i] = sum;
        }

        sum = 0.0;
	for (k=0; k<tno1; k++){
          sum += LNAO_coes[p][spin][tno1*j+k]*Vec[k];
        } 

        LNAO_H[p][spin][j] = sum;        

      } /* j */

      /* store the eigenvalues */

      for (i=0; i<tno1; i++){
        LNAO_pops[p][spin][i] = WR[i];
      }

      if (0 && myid==0){

        printf("ABC1 p=%2d\n",p);

	for (i=0; i<tno1; i++){
	  printf("ABC myid=%2d spin=%2d Mc_AN=%2d i=%2d IWORK=%2d WR=%15.11f WI=%15.11f\n",
                      myid,spin,Mc_AN,i,IWORK[i],WR[i],WI[i]);fflush(stdout);
	}

        printf("QQQ myid=%2d p=%2d spin=%2d\n",myid,p,spin);fflush(stdout);

	printf("WWW1 myid=%2d LNAO_coes p=%2d spin=%2d\n",myid,p,spin);fflush(stdout);
	for (i=0; i<tno1; i++){
	  for (j=0; j<tno1; j++){
	    printf("%10.5f ",LNAO_coes[p][spin][tno1*j+i]);fflush(stdout);
	  }
	  printf("\n");fflush(stdout);
	}

        printf("Check orthogonalization\n"); 
	for (i=0; i<tno1; i++){
	  for (j=0; j<tno1; j++){
 
            sum = 0.0;
	    for (k=0; k<tno1; k++){
              sum += VL[LDVL*i+k]*VR[LDVR*j+k];
  	    }
            printf("%10.5f ",sum);
	  }
	  printf("\n");fflush(stdout);
	}

	/*
	printf("WWW1 myid=%2d VL spin=%2d Mc_AN=%2d\n",myid,spin,Mc_AN);fflush(stdout);
	for (i=0; i<tno1; i++){
	  for (j=0; j<tno1*2; j++){
	    printf("%10.5f ",VL[tno1*j+i]);fflush(stdout);
	  }
	  printf("\n");fflush(stdout);
	}

	printf("WWW1 myid=%2d VR spin=%2d Mc_AN=%2d\n",myid,spin,Mc_AN);fflush(stdout);
	for (i=0; i<tno1; i++){
	  for (j=0; j<tno1*2; j++){
	    printf("%10.5f ",VR[tno1*j+i]);fflush(stdout);
	  }
	  printf("\n");fflush(stdout);
	}
	*/

	/*
        MPI_Finalize();
        exit(0);
	*/

      }

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

    sprintf(file,"%s%s.LNAO",filepath,filename);

    if ((fp = fopen(file,"w")) != NULL){

      fprintf(fp,"\n");
      fprintf(fp,"***********************************************************\n");
      fprintf(fp,"***********************************************************\n");
      fprintf(fp,"         Localized Natural Atomic Orbitals (LNAOs)         \n");
      fprintf(fp,"***********************************************************\n");
      fprintf(fp,"***********************************************************\n\n");

      for (p=0; p<LNAO_num; p++){
	for (spin=0; spin<=SpinP_switch; spin++){

	  Gc_AN = LNAO_Atoms[p];
	  wan1 = WhatSpecies[Gc_AN];
	  tno1 = Spe_Total_CNO[wan1];

	  base = 8;

	  fprintf(fp,"%2d %2s  Atomic Number = %2d  spin=%2d\n",p+1,SpeName[wan1],Gc_AN,spin);

	  for (k=0; k<((tno1-1)/base+1); k++){

	    fprintf(fp,"\n");
	    fprintf(fp,"                 ");
	    for (i=k*base; i<(k*base+base); i++){
	      if (i<tno1){ 
		fprintf(fp," %3d     ",i+1);
	      }
	    }
	    fprintf(fp,"\n");
	    fprintf(fp,"   Population    ");

	    for (i=k*base; i<(k*base+base); i++){
	      if (i<tno1){ 
		fprintf(fp,"%8.4f ",LNAO_pops[p][spin][i]);
	      }
	    }

	    fprintf(fp,"\n");
	    fprintf(fp,"  <LNAO|H|LNAO>  ");

	    for (i=k*base; i<(k*base+base); i++){
	      if (i<tno1){ 
		fprintf(fp,"%8.4f ",LNAO_H[p][spin][i]);
	      }
	    }

	    fprintf(fp,"\n");
	    fprintf(fp," Angular   Radial\n");

            i = 0;
	    for (l=0; l<=Supported_MaxL; l++){
	      for (mul=0; mul<Spe_Num_Basis[wan1][l]; mul++){
		for (m=0; m<(2*l+1); m++){
		  fprintf(fp,"  %s%s  ",Name_Angular[l][m],Name_Multiple[mul]);

		  for (j=k*base; j<(k*base+base); j++){
		    if (j<tno1){ 
		      fprintf(fp,"%8.4f ",LNAO_coes[p][spin][tno1*j+i]);
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

  /********************************************
             freeing of arrays
  ********************************************/

  for (p=0; p<LNAO_num; p++){
    for (spin=0; spin<=SpinP_switch; spin++){
      free(DMS[p][spin]);
    }
    free(DMS[p]);
  }
  free(DMS);

  for (p=0; p<LNAO_num; p++){
    free(S0[p]);
  }
  free(S0);

  for (p=0; p<LNAO_num; p++){
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


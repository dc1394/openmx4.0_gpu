/**********************************************************************
  cohp.c:

    cohp.c is a program to post-process a .cohp file to calculate 
    COHP (Crystal Orbital Hamilton Population) and 
    COOP (Crystal Orbital Overlap Population).

    Usage:  ./cohp Cgra.cohp

  Log of cohp.c:

     02/Oct./2022  Released by T. Ozaki

***********************************************************************/
 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "Inputtools.h"

#define eV2Hartree 27.2113845
#define PI 3.1415926535897932384626
#define YOUSO10  500
#define show_flag            0

typedef struct DCOMPLEX{double r,i;} dcomplex;  

void tetrahedron(char *mode, char basename[YOUSO10], int SpinP_switch, double ChemP, int NumES,
                 double *****ko, double ********COHP, 
                 int COHP_num_pairs,  
                 int *GA_COHP, int *GB_COHP, int *tnoA_COHP, int *tnoB_COHP,
                 int *l1_COHP, int *l2_COHP, int *l3_COHP,
                 int Numk1, int Numk2, int Numk3);

void gaussian_broadening(char *mode, char basename[YOUSO10], int SpinP_switch, double ChemP, int NumES,
                         double *****ko, double ********COHP, 
                         int COHP_num_pairs,  
                         int *GA_COHP, int *GB_COHP, int *tnoA_COHP, int *tnoB_COHP,
                         int *l1_COHP, int *l2_COHP, int *l3_COHP,
                         int Numk1, int Numk2, int Numk3); 
 
 
char *Delete_path_extension(char *s0, char *r0);

int main(int argc, char **argv)
{
  char *file_cohp;
  int SpinP_switch,NumES;
  int COHP_num_pairs;
  int i_vec[20];
  double *****ko;
  double ********COHP;
  double ********COOP;
  int spin,tnoA,tnoB,i,j,k,m,p,method;
  int GA_AN,GB_AN,l1,l2,l3,k1,k2,k3,Numk1,Numk2,Numk3;
  int *GA_COHP,*GB_COHP,*tnoA_COHP,*tnoB_COHP;
  int *l1_COHP,*l2_COHP,*l3_COHP;
  double gaussian,tmp,sum0,sum; 
  char buf[2000],buf2[2000];
  double nfac,emin,emax,de,e0,e,d,coe;
  int Solver;
  char basename[YOUSO10];
  char file1[YOUSO10];
  FILE *fp_cohp,*fp;
  int Dos_N;
  double ChemP;

  /**************************************************
                read data from file_cohp
  ***************************************************/

  file_cohp = argv[1];

  //printf("file_cohp=%s\n",file_cohp);

  fp_cohp = fopen(file_cohp, "r");  

  if (fp_cohp==NULL){
    printf("%s cannot be found.\n",file_cohp);
  }

  else{

    fread(&Solver,sizeof(int),1,fp_cohp);
    if (show_flag) printf("Solver=%2d\n",Solver);

    fread(&SpinP_switch,sizeof(int),1,fp_cohp);
    if (show_flag) printf("SpinP_switch=%2d\n",SpinP_switch);
    if (SpinP_switch==3){
      printf("NC-DFT is not supported.\n"); 
      exit(0);
    } 

    fread(&Numk1,sizeof(int),1,fp_cohp);
    fread(&Numk2,sizeof(int),1,fp_cohp);
    fread(&Numk3,sizeof(int),1,fp_cohp);

    if (show_flag){
      printf("Numk1=%2d Numk2=%2d Numk3=%2d\n",Numk1,Numk2,Numk3);
    } 

    fread(&NumES,sizeof(int),1,fp_cohp);
    if (show_flag) printf("NumES=%2d\n",NumES);

    fread(&COHP_num_pairs,sizeof(int),1,fp_cohp);
    if (show_flag) printf("COHP_num_pairs=%2d\n",COHP_num_pairs);

    fread(&ChemP,sizeof(double),1,fp_cohp);
    if (show_flag) printf("ChemP=%15.12f\n",ChemP);

    GA_COHP = (int*)malloc(sizeof(int)*COHP_num_pairs);
    GB_COHP = (int*)malloc(sizeof(int)*COHP_num_pairs);
    tnoA_COHP = (int*)malloc(sizeof(int)*COHP_num_pairs);
    tnoB_COHP = (int*)malloc(sizeof(int)*COHP_num_pairs);
    l1_COHP = (int*)malloc(sizeof(int)*COHP_num_pairs);
    l2_COHP = (int*)malloc(sizeof(int)*COHP_num_pairs);
    l3_COHP = (int*)malloc(sizeof(int)*COHP_num_pairs);

    ko = (double*****)malloc(sizeof(double****)*Numk1);
    for (k1=0; k1<Numk1; k1++){
      ko[k1] = (double****)malloc(sizeof(double***)*Numk2);
      for (k2=0; k2<Numk2; k2++){
	ko[k1][k2] = (double***)malloc(sizeof(double**)*Numk3);
	for (k3=0; k3<Numk3; k3++){
	  ko[k1][k2][k3] = (double**)malloc(sizeof(double*)*(SpinP_switch+1));
	  for (spin=0; spin<(SpinP_switch+1); spin++){
	    ko[k1][k2][k3][spin] = (double*)malloc(sizeof(double)*NumES);
	  }
	}
      }
    }

    COHP = (double********)malloc(sizeof(double*******)*Numk1);
    COOP = (double********)malloc(sizeof(double*******)*Numk1);

    for (k1=0; k1<Numk1; k1++){

      COHP[k1] = (double*******)malloc(sizeof(double******)*Numk2);
      COOP[k1] = (double*******)malloc(sizeof(double******)*Numk2);

      for (k2=0; k2<Numk2; k2++){

	COHP[k1][k2] = (double******)malloc(sizeof(double*****)*Numk3);
	COOP[k1][k2] = (double******)malloc(sizeof(double*****)*Numk3);

	for (k3=0; k3<Numk3; k3++){

	  COHP[k1][k2][k3] = (double*****)malloc(sizeof(double****)*(SpinP_switch+1));
	  COOP[k1][k2][k3] = (double*****)malloc(sizeof(double****)*(SpinP_switch+1));

	  for (spin=0; spin<=SpinP_switch; spin++){

	    COHP[k1][k2][k3][spin] = (double****)malloc(sizeof(double***)*NumES);
	    COOP[k1][k2][k3][spin] = (double****)malloc(sizeof(double***)*NumES);

	    for (p=0; p<NumES; p++){

	      COHP[k1][k2][k3][spin][p] = (double***)malloc(sizeof(double**)*COHP_num_pairs);
	      COOP[k1][k2][k3][spin][p] = (double***)malloc(sizeof(double**)*COHP_num_pairs);

	      fread(&i,sizeof(int),1,fp_cohp);
	      if (i!=spin){
		printf("The data seems not to follow the regular format #1.\n"); 
		exit(0);
	      }

	      fread(i_vec,sizeof(int),3,fp_cohp);
              if (i_vec[0]!=k1 || i_vec[1]!=k2 || i_vec[2]!=k3){
		printf("The data seems not to follow the regular format #2.\n"); 
		exit(0);
	      }

	      fread(&ko[k1][k2][k3][spin][p],sizeof(double),1,fp_cohp);
	      if (show_flag) printf("k1=%2d k2=%2d k3=%2d p=%2d ko=%15.12f\n",k1,k2,k3,p,ko[k1][k2][k3][spin][p]);

	      for (i=0; i<COHP_num_pairs; i++){

		fread(&j,sizeof(int),1,fp_cohp);

		if (i!=j){
		  printf("The data seems not to follow the regular format #3.\n"); 
		  exit(0);
		}

		fread(&GA_AN,sizeof(int),1,fp_cohp);
		fread(&GB_AN,sizeof(int),1,fp_cohp);
		fread(&tnoA,sizeof(int),1,fp_cohp);
		fread(&tnoB,sizeof(int),1,fp_cohp);
		fread(&l1,sizeof(int),1,fp_cohp);
		fread(&l2,sizeof(int),1,fp_cohp);
		fread(&l3,sizeof(int),1,fp_cohp);

		GA_COHP[i] = GA_AN; 
		GB_COHP[i] = GB_AN; 
		tnoA_COHP[i] = tnoA;
		tnoB_COHP[i] = tnoB;
		l1_COHP[i] = l1;
		l2_COHP[i] = l2;
		l3_COHP[i] = l3;

		COHP[k1][k2][k3][spin][p][i] = (double**)malloc(sizeof(double*)*tnoA);
		COOP[k1][k2][k3][spin][p][i] = (double**)malloc(sizeof(double*)*tnoA);

		for (j=0; j<tnoA; j++){

		  COHP[k1][k2][k3][spin][p][i][j] = (double*)malloc(sizeof(double)*tnoB);
		  COOP[k1][k2][k3][spin][p][i][j] = (double*)malloc(sizeof(double)*tnoB);

		  fread(&COHP[k1][k2][k3][spin][p][i][j][0],sizeof(double),tnoB,fp_cohp);
		  fread(&COOP[k1][k2][k3][spin][p][i][j][0],sizeof(double),tnoB,fp_cohp);

		  for (k=0; k<tnoB; k++){
		    if (show_flag){ printf("spin=%2d p=%2d i=%2d j=%2d k=%2d COHP=%15.12f COOP=%15.12f\n",
		 			   spin,p,i,j,k,
                                           COHP[k1][k2][k3][spin][p][i][j][k],
					   COOP[k1][k2][k3][spin][p][i][j][k]); fflush(stdout); }
		  }

		} // j
	      } // i
	    } // p
	  } // spin
	}  // k3
      } // k2
    } // k1

  } // else 

  /* get basename */

  Delete_path_extension(file_cohp,basename);

  /************************************
      start the calculation of cohp
  ************************************/

  if (Numk1==1 && Numk2==1 && Numk3==1){ // the case with only the gamma point
    printf("Kgrid= 1 1 1 => Gaussian Broadening is employed\n");
    method = 2;  
  }
  else{
    printf("Which method do you use?, Tetrahedron(1), Gaussian Broadening(2)\n"); 
    fgets(buf,1000,stdin); sscanf(buf,"%d",&method);

    if (method!=1 && method!=2){
      printf("The method is not supported.\n");
      exit(0);
    } 
  }

  /* tetrahedron method */

  if (method==1){ 

    tetrahedron( "cohp",basename,SpinP_switch,ChemP,NumES,ko,COHP,
                 COHP_num_pairs,GA_COHP,GB_COHP,
                 tnoA_COHP,tnoB_COHP,l1_COHP,l2_COHP,l3_COHP,
                 Numk1,Numk2,Numk3 );

    tetrahedron( "coop",basename,SpinP_switch,ChemP,NumES,ko,COOP,
                 COHP_num_pairs,GA_COHP,GB_COHP,
                 tnoA_COHP,tnoB_COHP,l1_COHP,l2_COHP,l3_COHP,
                 Numk1,Numk2,Numk3 );
  }

  /* gaussian broadening method */

  else if (method==2){ 

    gaussian_broadening( "cohp",basename,SpinP_switch,ChemP,NumES,ko,COHP,
                         COHP_num_pairs,GA_COHP,GB_COHP,
                         tnoA_COHP,tnoB_COHP,l1_COHP,l2_COHP,l3_COHP,
                         Numk1,Numk2,Numk3 );

    gaussian_broadening( "coop",basename,SpinP_switch,ChemP,NumES,ko,COOP,
                         COHP_num_pairs,GA_COHP,GB_COHP,
                         tnoA_COHP,tnoB_COHP,l1_COHP,l2_COHP,l3_COHP,
                         Numk1,Numk2,Numk3 );
  }  
     
  /************************************
           freeing of arrays
  ************************************/

  for (k1=0; k1<Numk1; k1++){
    for (k2=0; k2<Numk2; k2++){
      for (k3=0; k3<Numk3; k3++){
	for (spin=0; spin<=SpinP_switch; spin++){
	  for (p=0; p<NumES; p++){
	    for (i=0; i<COHP_num_pairs; i++){
	      for (j=0; j<tnoA_COHP[i]; j++){
		free(COHP[k1][k2][k3][spin][p][i][j]);
		free(COOP[k1][k2][k3][spin][p][i][j]);
	      } // j
	      free(COHP[k1][k2][k3][spin][p][i]);
	      free(COOP[k1][k2][k3][spin][p][i]);
	    } // i
	    free(COHP[k1][k2][k3][spin][p]);
	    free(COOP[k1][k2][k3][spin][p]);
	  } // p
	  free(COHP[k1][k2][k3][spin]);
	  free(COOP[k1][k2][k3][spin]);
	} // spin
	free(COHP[k1][k2][k3]);
	free(COOP[k1][k2][k3]);
      }  // k3
      free(COHP[k1][k2]);
      free(COOP[k1][k2]);
    } // k2
    free(COHP[k1]);
    free(COOP[k1]);
  } // k1
  free(COHP);
  free(COOP);

  for (k1=0; k1<Numk1; k1++){
    for (k2=0; k2<Numk2; k2++){
      for (k3=0; k3<Numk3; k3++){
	for (spin=0; spin<(SpinP_switch+1); spin++){
	  free(ko[k1][k2][k3][spin]);
	}
	free(ko[k1][k2][k3]);
      }
      free(ko[k1][k2]);
    }
    free(ko[k1]);
  }
  free(ko);

  free(GA_COHP);
  free(GB_COHP);
  free(tnoA_COHP);
  free(tnoB_COHP);
  free(l1_COHP);
  free(l2_COHP);
  free(l3_COHP);

  exit(0);
}


void tetrahedron(char *mode, char basename[YOUSO10], int SpinP_switch, double ChemP, int NumES,
                 double *****ko, double ********COHP, 
                 int COHP_num_pairs,  
                 int *GA_COHP, int *GB_COHP, int *tnoA_COHP, int *tnoB_COHP,
                 int *l1_COHP, int *l2_COHP, int *l3_COHP,
                 int Numk1, int Numk2, int Numk3)
{
  int ie,i,j,k,k1,k2,k3,i_in,j_in,k_in,ieg,spin;
  int iemin,iemax,ic,itetra;
  int GA_AN,GB_AN,tnoA,tnoB,l1,l2,l3;
  int N_Dos,i_Dos[10];
  static int Dos_N,pair_index;
  double *EnergyMesh, **TCOHP,****COHP_ij,***COHP_i;
  double spindeg,emin,emax,de,x,result,nfac,sum,sum0;
  char buf[2000],buf2[2000];
  FILE *fp;
  char file1[YOUSO10],fext[YOUSO10];
  double cell_e[8], tetra_e[4];
  static int tetra_id[6][4]= { {0,1,2,5}, {1,2,3,5}, {2,3,5,7},
			       {0,2,4,5}, {2,4,5,6}, {2,5,6,7} };

  if      (SpinP_switch==0) spindeg = 2.0;
  else if (SpinP_switch==1) spindeg = 1.0;

  if      ( strcasecmp(mode,"cohp")==0 ) sprintf(fext,"cohp");
  else if ( strcasecmp(mode,"coop")==0 ) sprintf(fext,"coop");

  if  ( strcasecmp(mode,"cohp")==0 ){

    printf("Please input the number of energy meshes\n"); 
    fgets(buf,1000,stdin); sscanf(buf,"%i",&Dos_N); 
    if (Dos_N<1){
      printf("The number of energy meshes should be a positive integer.\n");
      exit(0);
    } 

    printf("Which pair for COHP : (1 to %d)\n",COHP_num_pairs);
    fgets(buf,1000,stdin); sscanf(buf,"%d",&pair_index);
    if (pair_index<1 || COHP_num_pairs<pair_index){
      printf("The index of COHP is not the range (1 to %d).\n",COHP_num_pairs);
      exit(0);
    } 
    pair_index--; 
    if (show_flag) printf("pair_index=%2d\n",pair_index); 
  }

  GA_AN = GA_COHP[pair_index];
  GB_AN = GB_COHP[pair_index];
  tnoA = tnoA_COHP[pair_index];
  tnoB = tnoB_COHP[pair_index];
  l1 = l1_COHP[pair_index];
  l2 = l2_COHP[pair_index];
  l3 = l3_COHP[pair_index];

  /* allocation of arrays */

  EnergyMesh = (double*)malloc(sizeof(double)*Dos_N);

  TCOHP = (double**)malloc(sizeof(double*)*(SpinP_switch+1));
  for (spin=0; spin<(SpinP_switch+1); spin++){
    TCOHP[spin] = (double*)malloc(sizeof(double)*Dos_N);
    for (k=0; k<Dos_N; k++) TCOHP[spin][k] = 0.0;
  }

  COHP_ij = (double****)malloc(sizeof(double***)*(SpinP_switch+1));
  for (spin=0; spin<(SpinP_switch+1); spin++){
    COHP_ij[spin] = (double***)malloc(sizeof(double**)*tnoA);
    for (i=0; i<tnoA; i++){
      COHP_ij[spin][i] = (double**)malloc(sizeof(double*)*tnoB);
      for (j=0; j<tnoB; j++){
        COHP_ij[spin][i][j] = (double*)malloc(sizeof(double)*Dos_N);
        for (k=0; k<Dos_N; k++) COHP_ij[spin][i][j][k] = 0.0;
      }
    } 
  }

  COHP_i = (double***)malloc(sizeof(double**)*(SpinP_switch+1));
  for (spin=0; spin<(SpinP_switch+1); spin++){
    COHP_i[spin] = (double**)malloc(sizeof(double*)*tnoA);
    for (i=0; i<tnoA; i++){
      COHP_i[spin][i] = (double*)malloc(sizeof(double)*Dos_N);
        for (k=0; k<Dos_N; k++) COHP_i[spin][i][k] = 0.0;
    } 
  }

  /* find emin and emax */

  emin =  10000.0;
  emax = -10000.0;

  for (spin=0; spin<=SpinP_switch; spin++){
    for (k1=0; k1<Numk1; k1++){
      for (k2=0; k2<Numk2; k2++){
	for (k3=0; k3<Numk3; k3++){
	  if (ko[k1][k2][k3][spin][0]<emin)       emin = ko[k1][k2][k3][spin][0];
	  if (emax<ko[k1][k2][k3][spin][NumES-1]) emax = ko[k1][k2][k3][spin][NumES-1];
	}
      }
    }
  }
 
  emin -= 0.2;
  emax += 0.2;
  de = (emax - emin)/(double)(Dos_N-1);
  for (ie=0; ie<Dos_N; ie++) EnergyMesh[ie] = emin + de*(double)ie;

  //printf("emin=%15.12f emax=%15.12f\n",emin,emax);

  /* tetrhedron method */

  for (spin=0; spin<=SpinP_switch; spin++) {
    for (ieg=0; ieg<NumES; ieg++) {
      for (k1=0; k1<Numk1; k1++){
	for (k2=0; k2<Numk2; k2++){
	  for (k3=0; k3<Numk3; k3++){

	    for (i_in=0; i_in<2; i_in++) {
	      for (j_in=0; j_in<2; j_in++) {
		for (k_in=0; k_in<2; k_in++) {
		  cell_e[i_in*4+j_in*2+k_in] = 
                    ko[ (k1+i_in)%Numk1 ][ (k2+j_in)%Numk2 ][ (k3+k_in)%Numk3 ][spin][ieg] ;
		}
	      }
	    }

	    for (itetra=0; itetra<6; itetra++) {
	      for (ic=0; ic<4; ic++) {
		tetra_e[ic]=cell_e[ tetra_id[itetra][ic] ];
	      }

	      OrderE0(tetra_e,4);

	      x = (tetra_e[0]-emin)/(emax-emin)*(Dos_N-1) - 1.0;
	      iemin = (int)x;
	      x = (tetra_e[3]-emin)/(emax-emin)*(Dos_N-1) + 1.0;
	      iemax = (int)x;

	      if (iemin<0) { iemin=0; }
	      if (iemax>=Dos_N) {iemax=Dos_N-1; }


	      if ( 0<=iemin && iemin<Dos_N && 0<=iemax && iemax<Dos_N ) {

		for (ie=iemin; ie<=iemax; ie++) {

		  ATM_Dos( tetra_e, &EnergyMesh[ie], &result);

                  for (i=0; i<tnoA; i++){
                    for (j=0; j<tnoB; j++){
                      COHP_ij[spin][i][j][ie] += result*COHP[k1][k2][k3][spin][ieg][pair_index][i][j];
		    }
		  }

		}
	      }

	    } /* itetra */

            //printf("ABC spin=%2d ieg=%2d k1=%2d k2=%2d k3=%2d\n",spin,ieg,k1,k2,k3); fflush(stdout);

	  } /* k3 */
	} /* k2 */
      } /* k1 */
    } /* ieg */
  } /* spin */

  printf("generating...\n");

  /* normalization of COHP_ij and save the results */

  nfac = 1.0/(double)(Numk1 * Numk2 * Numk3 * 6);

  for (spin=0; spin<=SpinP_switch; spin++) {
    for (i=0; i<tnoA; i++){
      for (j=0; j<tnoB; j++){

        /* normalization */

	for (ie=0; ie<Dos_N; ie++) {
	  COHP_ij[spin][i][j][ie] = COHP_ij[spin][i][j][ie] * nfac;
	}

        /* save the result */

        sprintf(file1,"%s_i%i_o%i_o%i_s%i_tetra.%s",basename,pair_index+1,i,j,spin,fext);

	if ((fp = fopen(file1,"w")) != NULL){

	  sum0 = 0.0;
	  for (ie=0; ie<Dos_N; ie++){
	    if (EnergyMesh[ie]<=ChemP) sum0 += COHP_ij[spin][i][j][ie]*de;
	  }

          if ( strcasecmp(mode,"cohp")==0 ){
	    fprintf(fp,"# ChemP=%18.15f (Hartree), %18.15f (eV), Partial Band Energy=%18.15f (Hartree), %18.15f (eV)\n",
		    ChemP,ChemP*eV2Hartree,sum0*spindeg,sum0*eV2Hartree*spindeg);
	  }

          else if ( strcasecmp(mode,"coop")==0 ){
	    fprintf(fp,"# ChemP=%18.15f (Hartree), %18.15f (eV), Partial occupation=%18.15f\n",
		    ChemP,ChemP*eV2Hartree,sum0*spindeg);
	  }

	  fprintf(fp,"# energy(eV), COHP (dimensionless), integral of COHP\n");

	  sum = 0.0;
	  for (ie=0; ie<Dos_N; ie++){
	    sum += COHP_ij[spin][i][j][ie]*de;
	    fprintf(fp,"%18.15f %18.15f %18.15f\n",
                     (EnergyMesh[ie]-ChemP)*eV2Hartree,
                     COHP_ij[spin][i][j][ie]*spindeg,
                     sum*eV2Hartree*spindeg);
	  }

	  printf(" %s\n",file1);
	  fclose(fp);
	}
	else{
	  printf("Failure of saving %s\n",file1);
	}

      }
    }
  }

  /* calculate COHP_i and save the result */

  for (spin=0; spin<=SpinP_switch; spin++) {
    for (i=0; i<tnoA; i++){

      /* calculate COHP_i */

      for (ie=0; ie<Dos_N; ie++) {

        sum = 0.0;  
        for (j=0; j<tnoB; j++){
	  sum += COHP_ij[spin][i][j][ie];
	}
        COHP_i[spin][i][ie] = sum;
      }

      /* save the result */

      sprintf(file1,"%s_i%i_o%i_s%i_tetra.%s",basename,pair_index+1,i,spin,fext);

      if ((fp = fopen(file1,"w")) != NULL){

	sum0 = 0.0;
	for (ie=0; ie<Dos_N; ie++){
	  if (EnergyMesh[ie]<=ChemP) sum0 += COHP_i[spin][i][ie]*de;
	}

        if ( strcasecmp(mode,"cohp")==0 ){
  	  fprintf(fp,"# ChemP=%18.15f (Hartree), %18.15f (eV), Partial Band Energy=%18.15f (Hartree), %18.15f (eV)\n",
	  	  ChemP,ChemP*eV2Hartree,sum0*spindeg,sum0*eV2Hartree*spindeg);
	}

        else if ( strcasecmp(mode,"coop")==0 ){
          fprintf(fp,"# ChemP=%18.15f (Hartree), %18.15f (eV), Partial occupation=%18.15f\n",
  	          ChemP,ChemP*eV2Hartree,sum0*spindeg);
	}

        fprintf(fp,"# energy(eV), COHP (dimensionless), integral of COHP\n");

	sum = 0.0;
	for (ie=0; ie<Dos_N; ie++){
	  sum += COHP_i[spin][i][ie]*de;
	  fprintf(fp,"%18.15f %18.15f %18.15f\n",
		  (EnergyMesh[ie]-ChemP)*eV2Hartree,
		  COHP_i[spin][i][ie]*spindeg,
		  sum*eV2Hartree*spindeg);
	}

	printf(" %s\n",file1);
	fclose(fp);
      }
      else{
	printf("Failure of saving %s\n",file1);
      }

    }
  }

  /* calculate TCOHP */

  for (spin=0; spin<=SpinP_switch; spin++) {

    /* calculate TCOHP */

    for (ie=0; ie<Dos_N; ie++) {

      sum = 0.0;  
      for (i=0; i<tnoA; i++){
	sum += COHP_i[spin][i][ie];
      }
      TCOHP[spin][ie] = sum;
    }

    /* save the result */

    sprintf(file1,"%s_i%i_s%i_tetra.%s",basename,pair_index+1,spin,fext);

    if ((fp = fopen(file1,"w")) != NULL){

      sum0 = 0.0;
      for (ie=0; ie<Dos_N; ie++){
	if (EnergyMesh[ie]<=ChemP) sum0 += TCOHP[spin][ie]*de;
      }

      if ( strcasecmp(mode,"cohp")==0 ){
        fprintf(fp,"# ChemP=%18.15f (Hartree), %18.15f (eV), Partial Band Energy=%18.15f (Hartree), %18.15f (eV)\n",
	        ChemP,ChemP*eV2Hartree,sum0*spindeg,sum0*eV2Hartree*spindeg);
      }
      else if ( strcasecmp(mode,"coop")==0 ){
	fprintf(fp,"# ChemP=%18.15f (Hartree), %18.15f (eV), Partial occupation=%18.15f\n",
		ChemP,ChemP*eV2Hartree,sum0*spindeg);
      }

      fprintf(fp,"# energy(eV), COHP (dimensionless), integral of COHP\n");

      sum = 0.0;
      for (ie=0; ie<Dos_N; ie++){
	sum += TCOHP[spin][ie]*de;
	fprintf(fp,"%18.15f %18.15f %18.15f\n",
		(EnergyMesh[ie]-ChemP)*eV2Hartree,
		TCOHP[spin][ie]*spindeg,
		sum*eV2Hartree*spindeg);
      }

      printf(" %s\n",file1);
      fclose(fp);
    }
    else{
      printf("Failure of saving %s\n",file1);
    }
  }

  /* freeing of arrays */

  free(EnergyMesh);

  for (spin=0; spin<(SpinP_switch+1); spin++){
    free(TCOHP[spin]);
  }
  free(TCOHP);

  for (spin=0; spin<(SpinP_switch+1); spin++){
    for (i=0; i<tnoA; i++){
      for (j=0; j<tnoB; j++){
        free(COHP_ij[spin][i][j]);
      }
      free(COHP_ij[spin][i]);
    } 
    free(COHP_ij[spin]);
  }
  free(COHP_ij);

  for (spin=0; spin<(SpinP_switch+1); spin++){
    for (i=0; i<tnoA; i++){
      free(COHP_i[spin][i]);
    } 
    free(COHP_i[spin]);
  }
  free(COHP_i);
}



void gaussian_broadening(char *mode, char basename[YOUSO10], int SpinP_switch, double ChemP, int NumES,
                         double *****ko, double ********COHP, 
                         int COHP_num_pairs,  
                         int *GA_COHP, int *GB_COHP, int *tnoA_COHP, int *tnoB_COHP,
                         int *l1_COHP, int *l2_COHP, int *l3_COHP,
                         int Numk1, int Numk2, int Numk3)
{
  int i,j,k,p,k1,k2,k3,spin,m,l1,l2,l3;
  int tnoA,tnoB,GA_AN,GB_AN;
  double sum,sum0,e,de,spindeg,d,tmp;
  double coe,e0,emin,emax;
  static int Dos_N,pair_index;
  static double gaussian,nfac;
  char buf[2000],buf2[2000];
  double *tweight,*pweight,*weight;
  FILE *fp;
  char file1[YOUSO10],fext[YOUSO10];

  if      (SpinP_switch==0) spindeg = 2.0;
  else if (SpinP_switch==1) spindeg = 1.0;

  if      ( strcasecmp(mode,"cohp")==0 ) sprintf(fext,"cohp");
  else if ( strcasecmp(mode,"coop")==0 ) sprintf(fext,"coop");

  if  ( strcasecmp(mode,"cohp")==0 ){

    printf("Please input a value of gaussian (double) (eV)\n"); 
    fgets(buf,1000,stdin); sscanf(buf,"%lf",&gaussian); gaussian = gaussian/eV2Hartree; 
    nfac = 1.0/(gaussian*sqrt(PI)*(double)(Numk1*Numk2*Numk3));
    if (gaussian<0.0){
      printf("The width of gaussian should be positive.\n");
      exit(0);
    } 

    printf("Please input the number of energy meshes\n"); 
    fgets(buf,1000,stdin); sscanf(buf,"%i",&Dos_N); 
    if (Dos_N<1){
      printf("The number of energy meshes should be a positive integer.\n");
      exit(0);
    } 

    printf("Which pair for COHP : (1 to %d)\n",COHP_num_pairs);
    fgets(buf,1000,stdin); sscanf(buf,"%d",&pair_index);
    if (pair_index<1 || COHP_num_pairs<pair_index){
      printf("The index of COHP is not the range (1 to %d).\n",COHP_num_pairs);
      exit(0);
    } 
    pair_index--; 
    if (show_flag) printf("pair_index=%2d\n",pair_index); 
  }

  tweight = (double*)malloc(sizeof(double)*Dos_N);
  pweight = (double*)malloc(sizeof(double)*Dos_N);
  weight  = (double*)malloc(sizeof(double)*Dos_N);

  GA_AN = GA_COHP[pair_index];
  GB_AN = GB_COHP[pair_index];
  tnoA = tnoA_COHP[pair_index];
  tnoB = tnoB_COHP[pair_index];
  l1 = l1_COHP[pair_index];
  l2 = l2_COHP[pair_index];
  l3 = l3_COHP[pair_index];

  printf("generating...\n");

  /* find emin and emax */

  emin =  10000.0;
  emax = -10000.0;

  for (spin=0; spin<=SpinP_switch; spin++){
    for (k1=0; k1<Numk1; k1++){
      for (k2=0; k2<Numk2; k2++){
	for (k3=0; k3<Numk3; k3++){
	  if (ko[k1][k2][k3][spin][0]<emin)       emin = ko[k1][k2][k3][spin][0];
	  if (emax<ko[k1][k2][k3][spin][NumES-1]) emax = ko[k1][k2][k3][spin][NumES-1];
	}
      }
    }
  }
 
  emin -= 10.0*gaussian;
  emax += 10.0*gaussian;
  de = (emax - emin)/(double)(Dos_N-1);

  /* calculate cohp */

  for (spin=0; spin<=SpinP_switch; spin++){

    for (m=0; m<Dos_N; m++) tweight[m] = 0.0;

    for (i=0; i<tnoA; i++){

      for (m=0; m<Dos_N; m++) pweight[m] = 0.0;

      for (j=0; j<tnoB; j++){

	for (m=0; m<Dos_N; m++) weight[m] = 0.0;

	for (k1=0; k1<Numk1; k1++){
	  for (k2=0; k2<Numk2; k2++){
	    for (k3=0; k3<Numk3; k3++){
	      for (p=0; p<NumES; p++){
          
		e0 = ko[k1][k2][k3][spin][p];
		coe = COHP[k1][k2][k3][spin][p][pair_index][i][j];

		for (k=0; k<Dos_N; k++){
		  e = emin + de*(double)k;
		  d = (e-e0)/gaussian; 
		  tmp = coe*exp(-d*d)*nfac;
		  weight[k]  += tmp;
		  pweight[k] += tmp;
		  tweight[k] += tmp;
		}
	      }
	    }
	  }
	}

	/* no sum */

	sprintf(file1,"%s_i%i_o%i_o%i_s%i_gauss.%s",basename,pair_index+1,i,j,spin,fext);

	if ((fp = fopen(file1,"w")) != NULL){

	  sum0 = 0.0;
	  for (k=0; k<Dos_N; k++){
	    e = emin + de*(double)k;
	    if (e<=ChemP) sum0 += weight[k]*de;
	  }

	  if ( strcasecmp(mode,"cohp")==0 ){
	    fprintf(fp,"# ChemP=%18.15f (Hartree), %18.15f (eV), Partial Band Energy=%18.15f (Hartree), %18.15f (eV)\n",
		    ChemP,ChemP*eV2Hartree,sum0*spindeg,sum0*eV2Hartree*spindeg);
	  }
	  else if ( strcasecmp(mode,"coop")==0 ){
	    fprintf(fp,"# ChemP=%18.15f (Hartree), %18.15f (eV), Partial occupation=%18.15f\n",
		    ChemP,ChemP*eV2Hartree,sum0*spindeg);
	  }

         fprintf(fp,"# energy(eV), COOP (states/eV), integral of COOP\n");

	  sum = 0.0;
	  for (k=0; k<Dos_N; k++){
	    e = emin + de*(double)k;
	    sum += weight[k]*de;
	    fprintf(fp,"%18.15f %18.15f %18.15f\n",(e-ChemP)*eV2Hartree,weight[k]*spindeg,sum*eV2Hartree*spindeg);
	  }

	  printf(" %s\n",file1);
	  fclose(fp);
	}
	else{
	  printf("Failure of saving %s\n",file1);
	}

      } // j

        /* sum over j */

      sprintf(file1,"%s_i%i_o%i_s%i_gauss.%s",basename,pair_index+1,i,spin,fext);

      if ((fp = fopen(file1,"w")) != NULL){

	sum0 = 0.0;
	for (k=0; k<Dos_N; k++){
	  e = emin + de*(double)k;
	  if (e<=ChemP) sum0 += pweight[k]*de;
	}

	if ( strcasecmp(mode,"cohp")==0 ){
	  fprintf(fp,"# ChemP=%18.15f (Hartree), %18.15f (eV), Partial Band Energy=%18.15f (Hartree), %18.15f (eV)\n",
		  ChemP,ChemP*eV2Hartree,sum0*spindeg,sum0*eV2Hartree*spindeg);
	}
	else if ( strcasecmp(mode,"coop")==0 ){
	  fprintf(fp,"# ChemP=%18.15f (Hartree), %18.15f (eV), Partial occupation=%18.15f\n",
		  ChemP,ChemP*eV2Hartree,sum0*spindeg);
	}

        fprintf(fp,"# energy(eV), COOP (states/eV), integral of COOP\n");

	sum = 0.0;
	for (k=0; k<Dos_N; k++){
	  e = emin + de*(double)k;
	  sum += pweight[k]*de;
	  fprintf(fp,"%18.15f %18.15f %18.15f\n",(e-ChemP)*eV2Hartree,pweight[k]*spindeg,sum*eV2Hartree*spindeg);
	}

	fclose(fp);
	printf(" %s\n",file1);

      }
      else{
	printf("Failure of saving %s\n",file1);
      }

    } // i     

    /* sum over i and j */

    sprintf(file1,"%s_i%i_s%i_gauss.%s",basename,pair_index+1,spin,fext);

    if ((fp = fopen(file1,"w")) != NULL){

      sum0 = 0.0;
      for (k=0; k<Dos_N; k++){
	e = emin + de*(double)k;
	if (e<=ChemP) sum0 += tweight[k]*de;
      }

      if ( strcasecmp(mode,"cohp")==0 ){
        fprintf(fp,"# ChemP=%18.15f (Hartree), %18.15f (eV), Partial Band Energy=%18.15f (Hartree), %18.15f (eV)\n",
	        ChemP,ChemP*eV2Hartree,sum0*spindeg,sum0*eV2Hartree*spindeg);
      }
      else if ( strcasecmp(mode,"coop")==0 ){
	fprintf(fp,"# ChemP=%18.15f (Hartree), %18.15f (eV), Partial occupation=%18.15f\n",
		ChemP,ChemP*eV2Hartree,sum0*spindeg);
      }

      fprintf(fp,"# energy(eV), COOP (states/eV), integral of COOP\n");

      sum = 0.0;
      for (k=0; k<Dos_N; k++){
	e = emin + de*(double)k;
	sum += tweight[k]*de;
	fprintf(fp,"%18.15f %18.15f %18.15f\n",(e-ChemP)*eV2Hartree,tweight[k]*spindeg,sum*eV2Hartree*spindeg);
      }

      fclose(fp);
      printf(" %s\n",file1);
    }
    else{
      printf("Failure of saving %s\n",file1);
    }

  } // spin

  /* freeing of arrays */

  free(tweight);
  free(pweight);
  free(weight);
}



/* 
   input s0
   output r0
   if input='../test.Dos.val' output='test'
*/

char *Delete_path_extension(char *s0, char *r0)
{
  char *c;
  /* char s[YOUSO10]; */

  /* find '/' */
  c=rindex(s0,'/');
  if (c)  {
    strcpy(r0,c+1);
  }
  else {
    strcpy(r0,s0);
  }
  if (show_flag) printf("<%s>\n",r0);

  if (strlen(r0)==0 ) { return  NULL; }
  
  c =index(r0,'.');
  if (c) {
    *c='\0';
  }
  if (show_flag) printf("<%s>\n",r0);
  return r0;

}





/*************************************************************************
  elpa_noncol.c:

  elpa_noncol.c is a program to read a file of *.scfout, and to diagonalize 
  the generalized eigenvalue problem within a non-collinear DFT calculation. 

  Usage: ./mpirun -np 10 ./elpa_noncol filename.scfout 

  Log of elpa_noncol.c:

     05/Dec./2023  Released by T. Ozaki 

**************************************************************************/

#include "f77func.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <mpi.h>

/* define struct, global variables, and functions */

typedef struct { double r,i; } dcomplex;

#define print_data  0
#define PI          3.1415926535897932384626
#define eV2Hartree  27.2113845                
#define kB          0.00008617251324000000   /* eV/K           */          
#define fp_bsize    1048576     /* buffer size for setvbuf */
#define SCFOUT_VERSION 3

#define LATEST_VERSION 3
#define FREAD(POINTER, SIZE, NUMBER, FILE_POINTER)\
  do {\
    fread(POINTER, SIZE, NUMBER, FILE_POINTER);\
    if (conversionSwitch){\
      int dATA;\
      for (dATA=0; dATA<NUMBER; dATA++){\
	char *out=(char*)(POINTER+dATA);\
	int bYTE;\
	for (bYTE=0; bYTE<SIZE/2; bYTE++){\
	  char tmp=out[bYTE];\
	  out[bYTE]=out[SIZE-bYTE-1];\
	  out[SIZE-bYTE-1]=tmp;\
	}\
      }\
    }\
  } while (0)
/* ***/

int atomnum,Catomnum,Latomnum,Ratomnum;
int TCpyCell,SpinP_switch,MatDim,MatDim2;
int **natn,**ncn,*Total_NumOrbs,*FNAN,**atv_ijk,*MP;
double **atv;
double tv[4][4],rtv[4][4];
double *eval;

/* scalapack */
static int NBLK=128;
/* for DimMat2 */
int nblk2,np_rows2,np_cols2,na_rows2,na_cols2,na_rows_max2,na_cols_max2;
int my_prow2,my_pcol2;
int bhandle1_2,ictxt1_2;
int descS2[9],descH2[9],descC2[9];
dcomplex *Hs2,*Ss2,*Cs2;


void Cblacs_barrier (int, char *);
void pzgemm_( char *TRANSA, char *TRANSB,
              int *M, int *N, int *K, 
              dcomplex *alpha, dcomplex *s, int *ONE1, int *ONE2,
              int descH[9], dcomplex *Ss, int *ONE3, int *ONE4,
              int descS[9], dcomplex *beta, dcomplex *Cs, int *ONE5, int *ONE6, int descC[9]);


void solve_evp_complex_( int *n1, int *n2, dcomplex *Cs, int *na_rows1, double *ko, dcomplex *Ss, 
                         int *na_rows2, int *nblk, int *mpi_comm_rows_int, int *mpi_comm_cols_int );

void elpa_solve_evp_complex_2stage_double_impl_
      ( int *n, int *MaxN, dcomplex *Hs, int *na_rows1, double *ko, dcomplex *Cs, 
        int *na_rows2, int *nblk, int *na_cols1,
        int *mpi_comm_rows_int, int *mpi_comm_cols_int, int *mpiworld );

int numroc_( int *n, int *nb, int *iproc, int *isrcproc, int *nprocs);
int Csys2blacs_handle(MPI_Comm comm);
void descinit_( int *desc, int *m, int *n, int *mb, int *nb, int *irsrc, int *icsrc,
                int *ictxt, int *lld, int *info);
int Cblacs_gridinit( int* context, char * order, int np_row, int np_col); 

void set_Ss2_Hs2(char *mode, char *argv[], double k1, double k2, double k3);

void zheevx_(char *JOBZ, char *RANGE, char *UPLO, int *N, dcomplex *A, int *LDA, double *VL, double *VU, int *IL, int *IU,
                        double *ABSTOL, int *M, double *W, dcomplex *Z, int *LDZ, dcomplex *WORK, int *LWORK, double *RWORK,
                        int *IWORK, int *IFAIL, int *INFO);



int main(int argc, char *argv[]) 
{
  int i,j,jg,po;
  double k1,k2,k3;
  int myid,numprocs;
  MPI_Comm mpi_comm_rows, mpi_comm_cols;
  int mpi_comm_rows_int,mpi_comm_cols_int;
  int scf_eigen_lib_flag=2;
  int ZERO=0, ONE=1;
  dcomplex alpha = {1.0,0.0}; dcomplex beta = {0.0,0.0};

  /* MPI initialize */

  MPI_Init(&argc,&argv);
  MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD,&myid);
  
  /* set k1, k2, and k3 */
  
  k1 = (double)atof(argv[2]);
  k2 = (double)atof(argv[3]);
  k3 = (double)atof(argv[4]);

  /* read the .scfout file and set Ss2 and Hs2 */
  
  set_Ss2_Hs2("set",argv,k1,k2,k3);

  /* diagonalize Ss */

  MPI_Comm_split(MPI_COMM_WORLD,my_pcol2,my_prow2,&mpi_comm_rows);
  MPI_Comm_split(MPI_COMM_WORLD,my_prow2,my_pcol2,&mpi_comm_cols);

  mpi_comm_rows_int = MPI_Comm_c2f(mpi_comm_rows);
  mpi_comm_cols_int = MPI_Comm_c2f(mpi_comm_cols);
  
  if (scf_eigen_lib_flag==1){

    F77_NAME(solve_evp_complex,SOLVE_EVP_COMPLEX)
      ( &MatDim2, &MatDim2, Ss2, &na_rows2, eval, Cs2, &na_rows2, &nblk2,
	&mpi_comm_rows_int, &mpi_comm_cols_int );
  }

  else if (scf_eigen_lib_flag==2){

    int mpiworld;
    mpiworld = MPI_Comm_c2f(MPI_COMM_WORLD);
    F77_NAME(elpa_solve_evp_complex_2stage_double_impl,ELPA_SOLVE_EVP_COMPLEX_2STAGE_DOUBLE_IMPL)
      ( &MatDim2, &MatDim2, Ss2, &na_rows2, eval, Cs2, &na_rows2, &nblk2, &na_cols2, 
	&mpi_comm_rows_int, &mpi_comm_cols_int, &mpiworld );

  }

  /*
  for (i=0; i<MatDim2; i++){
    printf("ABC1 %2d %15.12f\n",i,eval[i]);
  }
  */

  /* calculate (eigenvectors of S) times 1/sqrt(eval) */
  
  for (i=0; i<MatDim2; i++){
    eval[i] = 1.0/sqrt(eval[i]);
  }

  for (i=0; i<na_rows2; i++){
    for (j=0; j<na_cols2; j++){
      jg = np_cols2*nblk2*((j)/nblk2) + (j)%nblk2 + ((np_cols2+my_pcol2)%np_cols2)*nblk2;
      Ss2[j*na_rows2+i].r = Cs2[j*na_rows2+i].r*eval[jg];
      Ss2[j*na_rows2+i].i = Cs2[j*na_rows2+i].i*eval[jg];
    }
  }

  /* transformation of H by Ss */
  /* pzgemm:  H * U * 1.0/sqrt(eval) */

  for(i=0; i<na_rows_max2*na_cols_max2; i++){
    Cs2[i].r = 0.0;
    Cs2[i].i = 0.0;
  }

  Cblacs_barrier(ictxt1_2,"A");
  F77_NAME(pzgemm,PZGEMM)("N","N",&MatDim2,&MatDim2,&MatDim2,&alpha,Hs2,&ONE,&ONE,descH2,Ss2,
			  &ONE,&ONE,descS2,&beta,Cs2,&ONE,&ONE,descC2);

  /* 1.0/sqrt(ko[l]) * U^+ H * U * 1.0/sqrt(eval) */

  for(i=0; i<na_rows2*na_cols2; i++){
    Hs2[i].r = 0.0;
    Hs2[i].i = 0.0;
  }

  Cblacs_barrier(ictxt1_2,"C");
  F77_NAME(pzgemm,PZGEMM)("C","N",&MatDim2,&MatDim2,&MatDim2,&alpha,Ss2,&ONE,&ONE,descS2,Cs2,
			  &ONE,&ONE,descC2,&beta,Hs2,&ONE,&ONE,descH2);

  /* diagonalize H' */

  if (scf_eigen_lib_flag==1){

    F77_NAME(solve_evp_complex,SOLVE_EVP_COMPLEX)
      ( &MatDim2, &MatDim2, Hs2, &na_rows2, eval, Cs2,
	&na_rows2, &nblk2, &mpi_comm_rows_int, &mpi_comm_cols_int );
  }
  else if (scf_eigen_lib_flag==2){

    int mpiworld;
    mpiworld = MPI_Comm_c2f(MPI_COMM_WORLD);
    F77_NAME(elpa_solve_evp_complex_2stage_double_impl,ELPA_SOLVE_EVP_COMPLEX_2STAGE_DOUBLE_IMPL)
      ( &MatDim2, &MatDim2, Hs2, &na_rows2, eval, Cs2, &na_rows2, &nblk2, &na_cols2,
	&mpi_comm_rows_int, &mpi_comm_cols_int, &mpiworld );
  }

  if (myid==0){  
    for (i=0; i<MatDim2; i++){
      printf("i=%2d eigenvalues=%15.12f\n",i+1,eval[i]);
    }
  }

  MPI_Comm_free(&mpi_comm_rows);
  MPI_Comm_free(&mpi_comm_cols);

  /* free arrays */

  set_Ss2_Hs2("free",argv,k1,k2,k3);

  /* finalize MPI and exit */

  MPI_Finalize();
  exit(0);
}




void set_Ss2_Hs2(char *mode, char *argv[], double k1, double k2, double k3)
{
  static int first_flag=1;
  int ct_AN,Rn,version,order_max;
  int n1,n2,n3,i,j,p,TNO1,TNO2;
  int h_AN,Gh_AN,spin;
  int ig,jg,il,jl,l1,l2,l3;
  int brow,bcol,prow,pcol;
  int brow2,bcol2,prow2,pcol2;
  int i_vec[20];
  int *p_vec;
  double co,si,kRn,d_vec[20];
  FILE *fp;
  char buf[fp_bsize];          /* setvbuf */
  int myid,numprocs; 

  MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD,&myid);

  if (mode=="set"){
  
    if ((fp = fopen(argv[1],"r")) != NULL){

      setvbuf(fp,buf,_IOFBF,fp_bsize);  /* setvbuf */

      if (myid==0){
	printf("\nRead the scfout file (%s)\n\n",argv[1]);fflush(stdout);
      }
    
      fread(i_vec,sizeof(int),6,fp);

      int conversionSwitch;
      if (i_vec[1]==0 && i_vec[1]<0 || i_vec[1]>(LATEST_VERSION)*4+3){
	conversionSwitch=1;
	int i;
	for (i=0; i<6; i++){
	  int value=*(i_vec+i);
	  char *in=(char*)&value;
	  char *out=(char*)(i_vec+i);
	  int j;
	  for (j=0; j<sizeof(int); j++){
	    out[j]=in[sizeof(int)-j-1];
	  }
	}
	if (i_vec[1]==0 && i_vec[1]<0 || i_vec[1]>(LATEST_VERSION)*4+3){
	  puts("Error: Mismatch of the endianness");fflush(stdout);
	  MPI_Abort(MPI_COMM_WORLD, 1);
	}
      } else {
	conversionSwitch=0;
      }

      atomnum      = i_vec[0];
      SpinP_switch=i_vec[1]%4;
      version=i_vec[1]/4;
      char* openmxVersion;

      if (SpinP_switch==0 || SpinP_switch==1){
	if (myid==0){
	  printf("The code is valid for the non-collinear case.\n");    
	}
	MPI_Finalize();
	exit(0);
      }
    
      if (version==0){
	openmxVersion="3.7, 3.8 or an older distribution";
      } else if (version==1){
	openmxVersion="3.7.x (for development of HWC)";
      } else if (version==2){
	openmxVersion="3.7.x (for development of HWF)";
      } else if (version==3){
	openmxVersion="3.9";
      }

      if (version!=SCFOUT_VERSION){
	if (myid==0){
	  printf("The file format of the SCFOUT file:  %d\n", version);
	  printf("The vesion is not supported by the current read_scfout\n");
	} 
	MPI_Finalize();
	exit(0);
      }

      if (myid==0){
	//puts("***");
	//printf("The file format of the SCFOUT file:  %d\n", version);
      }

      Catomnum =     i_vec[2];
      Latomnum =     i_vec[3];
      Ratomnum =     i_vec[4];
      TCpyCell =     i_vec[5];

      /****************************************************
      order_max (added by N. Yamaguchi for HWC)
      ****************************************************/

      FREAD(i_vec, sizeof(int), 1, fp);
      order_max=i_vec[0];

      /****************************************************
      allocation of arrays:

      double atv[TCpyCell+1][4];
      ****************************************************/

      if (first_flag){
	atv = (double**)malloc(sizeof(double*)*(TCpyCell+1));
	for (Rn=0; Rn<=TCpyCell; Rn++){
	  atv[Rn] = (double*)malloc(sizeof(double)*4);
	}
      }

      /****************************************************
                   read atv[TCpyCell+1][4];
      ****************************************************/

      for (Rn=0; Rn<=TCpyCell; Rn++){
	FREAD(atv[Rn],sizeof(double),4,fp);
      }

      /****************************************************
        allocation of arrays:

        int atv_ijk[TCpyCell+1][4];
      ****************************************************/

      if (first_flag){
	atv_ijk = (int**)malloc(sizeof(int*)*(TCpyCell+1));
	for (Rn=0; Rn<=TCpyCell; Rn++){
	  atv_ijk[Rn] = (int*)malloc(sizeof(int)*4);
	}
      }
      
      /****************************************************
                  read atv_ijk[TCpyCell+1][4];
      ****************************************************/

      for (Rn=0; Rn<=TCpyCell; Rn++){
	FREAD(atv_ijk[Rn],sizeof(int),4,fp);
      }

      /****************************************************
        allocation of arrays:

        int Total_NumOrbs[atomnum+1];
        int FNAN[atomnum+1];
      ****************************************************/

      if (first_flag){
        Total_NumOrbs = (int*)malloc(sizeof(int)*(atomnum+1));
        FNAN = (int*)malloc(sizeof(int)*(atomnum+1));
      }
	
      /****************************************************
              the number of orbitals in each atom
      ****************************************************/

      p_vec = (int*)malloc(sizeof(int)*atomnum);
      FREAD(p_vec,sizeof(int),atomnum,fp);
      Total_NumOrbs[0] = 1;
      for (ct_AN=1; ct_AN<=atomnum; ct_AN++){
	Total_NumOrbs[ct_AN] = p_vec[ct_AN-1];
      }
      free(p_vec);

      /****************************************************
      FNAN[]:
      the number of first nearest neighbouring atoms
      ****************************************************/

      p_vec = (int*)malloc(sizeof(int)*atomnum);
      FREAD(p_vec,sizeof(int),atomnum,fp);
      FNAN[0] = 0;
      for (ct_AN=1; ct_AN<=atomnum; ct_AN++){
	FNAN[ct_AN] = p_vec[ct_AN-1];
      }
      free(p_vec);
    
      /****************************************************
      allocation of arrays:

      int natn[atomnum+1][FNAN[ct_AN]+1];
      int ncn[atomnum+1][FNAN[ct_AN]+1];
      ****************************************************/

      if (first_flag){
	natn = (int**)malloc(sizeof(int*)*(atomnum+1));
	for (ct_AN=0; ct_AN<=atomnum; ct_AN++){
	  natn[ct_AN] = (int*)malloc(sizeof(int)*(FNAN[ct_AN]+1));
	}

	ncn = (int**)malloc(sizeof(int*)*(atomnum+1));
	for (ct_AN=0; ct_AN<=atomnum; ct_AN++){
	  ncn[ct_AN] = (int*)malloc(sizeof(int)*(FNAN[ct_AN]+1));
	}
      }

      /****************************************************
      natn[][]:
      grobal index of neighboring atoms of an atom ct_AN
      ****************************************************/

      for (ct_AN=1; ct_AN<=atomnum; ct_AN++){
	FREAD(natn[ct_AN],sizeof(int),FNAN[ct_AN]+1,fp);
      }

      /****************************************************
      ncn[][]:
      grobal index for cell of neighboring atoms
      of an atom ct_AN
      ****************************************************/

      for (ct_AN=1; ct_AN<=atomnum; ct_AN++){
	FREAD(ncn[ct_AN],sizeof(int),FNAN[ct_AN]+1,fp);
      }

      /****************************************************
      tv[4][4]:
      unit cell vectors in Bohr
      ****************************************************/

      FREAD(tv[1],sizeof(double),4,fp);
      FREAD(tv[2],sizeof(double),4,fp);
      FREAD(tv[3],sizeof(double),4,fp);

      /****************************************************
      rtv[4][4]:
      unit cell vectors in Bohr
      ****************************************************/

      FREAD(rtv[1],sizeof(double),4,fp);
      FREAD(rtv[2],sizeof(double),4,fp);
      FREAD(rtv[3],sizeof(double),4,fp);

      /****************************************************
      Gxyz[][1-3]:
      atomic coordinates in Bohr
      ****************************************************/

      for (ct_AN=1; ct_AN<=atomnum; ct_AN++){
	FREAD(d_vec,sizeof(double),4,fp);
      }

      /*****************************************
        set parameters for BLACS and ScaLapack  
      *****************************************/

      int nblk_m,info,maxTNO,ZERO=0;

      if (first_flag){
        MP = (int*)malloc(sizeof(int)*(atomnum+1));
      }
	
      MatDim = 0;
      maxTNO = 0;
      for (ct_AN=1; ct_AN<=atomnum; ct_AN++){
	MP[ct_AN] = MatDim + 1;
	MatDim += Total_NumOrbs[ct_AN];
	if (maxTNO<Total_NumOrbs[ct_AN]) maxTNO = Total_NumOrbs[ct_AN]; 
      }
      MatDim2 = 2*MatDim;

      if (first_flag){
        eval = (double*)malloc(sizeof(double)*(MatDim2+3));
      }

      /* setting for BLACS in the matrix size of MatDim2 */

      int nblk_m2;

      np_cols2 = (int)(sqrt((float)numprocs));
      do{
	if((numprocs%np_cols2)==0) break;
	np_cols2--;
      } while (np_cols2>=2);
      np_rows2 = numprocs/np_cols2;

      nblk_m2 = NBLK;
      while((nblk_m2*np_rows2>MatDim2 || nblk_m2*np_cols2>MatDim2) && (nblk_m2 > 1)){
	nblk_m2 /= 2;
      }
      if(nblk_m2<1) nblk_m2 = 1;

      MPI_Allreduce(&nblk_m2,&nblk2,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD);
    
      my_prow2 = myid/np_cols2;
      my_pcol2 = myid%np_cols2;

      na_rows2 = numroc_(&MatDim2, &nblk2, &my_prow2, &ZERO, &np_rows2 );
      na_cols2 = numroc_(&MatDim2, &nblk2, &my_pcol2, &ZERO, &np_cols2 );

      bhandle1_2 = Csys2blacs_handle(MPI_COMM_WORLD);
      ictxt1_2 = bhandle1_2;

      Cblacs_gridinit(&ictxt1_2, "Row", np_rows2, np_cols2);

      if (first_flag){
        Ss2 = (dcomplex*)malloc(sizeof(dcomplex)*na_rows2*na_cols2);
        Hs2 = (dcomplex*)malloc(sizeof(dcomplex)*na_rows2*na_cols2);
        Cs2 = (dcomplex*)malloc(sizeof(dcomplex)*na_rows2*na_cols2);
      }

      for (i=0; i<na_rows2*na_cols2; i++){
	Ss2[i].r = 0.0; Ss2[i].i = 0.0;
	Hs2[i].r = 0.0; Hs2[i].i = 0.0;
	Cs2[i].r = 0.0; Cs2[i].i = 0.0;
      }
      
      MPI_Allreduce(&na_rows2,&na_rows_max2,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
      MPI_Allreduce(&na_cols2,&na_cols_max2,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
      Cs2 = (dcomplex*)malloc(sizeof(dcomplex)*na_rows_max2*na_cols_max2);

      for (i=0; i<na_rows_max2*na_cols_max2; i++){
	Cs2[i].r = 0.0; Cs2[i].i = 0.0;
      }
      
      descinit_(descH2, &MatDim2,  &MatDim2,  &nblk2,  &nblk2,  &ZERO, &ZERO, &ictxt1_2, &na_rows2,  &info);
      descinit_(descC2, &MatDim2,  &MatDim2,  &nblk2,  &nblk2,  &ZERO, &ZERO, &ictxt1_2, &na_rows2,  &info);
      descinit_(descS2, &MatDim2,  &MatDim2,  &nblk2,  &nblk2,  &ZERO, &ZERO, &ictxt1_2, &na_rows2,  &info);
      
      /*****************************************
                   set Hs with H
      *****************************************/

      int shift_i,shift_j;
      double *tmpHS; 
      tmpHS = (double*)malloc(sizeof(double)*maxTNO);

      for (spin=0; spin<=SpinP_switch; spin++){

	if      (spin==0) { shift_i = 0;      shift_j = 0;      } 
	else if (spin==1) { shift_i = MatDim; shift_j = MatDim; } 
	else if (spin==2) { shift_i = 0;      shift_j = MatDim; } 
	else if (spin==3) { shift_i = 0;      shift_j = MatDim; } 
	
	for (ct_AN=1; ct_AN<=atomnum; ct_AN++){
	  TNO1 = Total_NumOrbs[ct_AN];

	  for (h_AN=0; h_AN<=FNAN[ct_AN]; h_AN++){

	    Gh_AN = natn[ct_AN][h_AN];
	    Rn = ncn[ct_AN][h_AN];
	    TNO2 = Total_NumOrbs[Gh_AN];

	    l1 = atv_ijk[Rn][1];
	    l2 = atv_ijk[Rn][2];
	    l3 = atv_ijk[Rn][3];

	    kRn = k1*(double)l1 + k2*(double)l2 + k3*(double)l3;
	    si = sin(2.0*PI*kRn);
	    co = cos(2.0*PI*kRn);
	  
	    for (i=0; i<TNO1; i++){

	      ig = MP[ct_AN] + i + shift_i;
	      brow2 = (ig-1)/nblk2;
	      prow2 = brow2%np_rows2;

	      FREAD(tmpHS,sizeof(double),TNO2,fp);
	      
	      for (j=0; j<TNO2; j++){

		jg = MP[Gh_AN] + j + shift_j;
		bcol2 = (jg-1)/nblk2;
		pcol2 = bcol2%np_cols2;

		if (my_prow2==prow2 && my_pcol2==pcol2){

		  il = (brow2/np_rows2+1)*nblk2+1;
		  jl = (bcol2/np_cols2+1)*nblk2+1;

		  if (((my_prow2+np_rows2)%np_rows2) >= (brow2%np_rows2)){
		    if(my_prow2==prow2){
		      il = il+(ig-1)%nblk2;
		    }
		    il = il-nblk2;
		  }

		  if (((my_pcol2+np_cols2)%np_cols2) >= (bcol2%np_cols2)){
		    if(my_pcol2==pcol2){
		      jl = jl+(jg-1)%nblk2;
		    }
		    jl = jl-nblk2;
		  }

                  if (spin==0 || spin==1 || spin==2){
  		    Hs2[(jl-1)*na_rows2+il-1].r += tmpHS[j]*co;
		    Hs2[(jl-1)*na_rows2+il-1].i += tmpHS[j]*si;
		  }
		  else if (spin==3){
  		    Hs2[(jl-1)*na_rows2+il-1].r -= tmpHS[j]*si;
		    Hs2[(jl-1)*na_rows2+il-1].i += tmpHS[j]*co;
		  }
		  
		} // end of if (my_prow==prow && my_pcol==pcol)
		
	      } // j

              /* Consider the left bottom block */
	      
              if (spin==2 || spin==3){

                shift_i = 0;
		shift_j = MatDim;

  	        jg = MP[ct_AN] + i + shift_i;
	        bcol2 = (jg-1)/nblk2;
	        pcol2 = bcol2%np_cols2;

		for (j=0; j<TNO2; j++){

		  ig = MP[Gh_AN] + j + shift_j;
	          brow2 = (ig-1)/nblk2;
	          prow2 = brow2%np_rows2;

		  if (my_prow2==prow2 && my_pcol2==pcol2){

		    il = (brow2/np_rows2+1)*nblk2+1;
		    jl = (bcol2/np_cols2+1)*nblk2+1;

		    if (((my_prow2+np_rows2)%np_rows2) >= (brow2%np_rows2)){
		      if(my_prow2==prow2){
			il = il+(ig-1)%nblk2;
		      }
		      il = il-nblk2;
		    }

		    if (((my_pcol2+np_cols2)%np_cols2) >= (bcol2%np_cols2)){
		      if(my_pcol2==pcol2){
			jl = jl+(jg-1)%nblk2;
		      }
		      jl = jl-nblk2;
		    }

                    if (spin==2){
  		      Hs2[(jl-1)*na_rows2+il-1].r += tmpHS[j]*co;
		      Hs2[(jl-1)*na_rows2+il-1].i -= tmpHS[j]*si;
		    }
		    else if (spin==3){
		      Hs2[(jl-1)*na_rows2+il-1].r += tmpHS[j]*si;
		      Hs2[(jl-1)*na_rows2+il-1].i -= tmpHS[j]*co;
		    }
		      
		  } // end of if (my_prow==prow && my_pcol==pcol)
		} // j
	      } // end of if (spin==2 || spin==3)
	      
	    } // i
	  } // h_AN
	} // ct_AN
      } // spin
      
      /*****************************************
                   set Hs with iH
      *****************************************/

      for (spin=0; spin<=2; spin++){

	if      (spin==0) { shift_i = 0;      shift_j = 0;      } 
	else if (spin==1) { shift_i = MatDim; shift_j = MatDim; } 
	else if (spin==2) { shift_i = 0;      shift_j = MatDim; } 
	
	for (ct_AN=1; ct_AN<=atomnum; ct_AN++){
	  TNO1 = Total_NumOrbs[ct_AN];

	  for (h_AN=0; h_AN<=FNAN[ct_AN]; h_AN++){

	    Gh_AN = natn[ct_AN][h_AN];
	    Rn = ncn[ct_AN][h_AN];
	    TNO2 = Total_NumOrbs[Gh_AN];

	    l1 = atv_ijk[Rn][1];
	    l2 = atv_ijk[Rn][2];
	    l3 = atv_ijk[Rn][3];

	    kRn = k1*(double)l1 + k2*(double)l2 + k3*(double)l3;
	    si = sin(2.0*PI*kRn);
	    co = cos(2.0*PI*kRn);
	  
	    for (i=0; i<TNO1; i++){

	      ig = MP[ct_AN] + i + shift_i;
	      brow2 = (ig-1)/nblk2;
	      prow2 = brow2%np_rows2;

	      FREAD(tmpHS,sizeof(double),TNO2,fp);
	      
	      for (j=0; j<TNO2; j++){

		jg = MP[Gh_AN] + j + shift_j;
		bcol2 = (jg-1)/nblk2;
		pcol2 = bcol2%np_cols2;

		if (my_prow2==prow2 && my_pcol2==pcol2){

		  il = (brow2/np_rows2+1)*nblk2+1;
		  jl = (bcol2/np_cols2+1)*nblk2+1;

		  if (((my_prow2+np_rows2)%np_rows2) >= (brow2%np_rows2)){
		    if(my_prow2==prow2){
		      il = il+(ig-1)%nblk2;
		    }
		    il = il-nblk2;
		  }

		  if (((my_pcol2+np_cols2)%np_cols2) >= (bcol2%np_cols2)){
		    if(my_pcol2==pcol2){
		      jl = jl+(jg-1)%nblk2;
		    }
		    jl = jl-nblk2;
		  }

		  Hs2[(jl-1)*na_rows2+il-1].r -= tmpHS[j]*si;
		  Hs2[(jl-1)*na_rows2+il-1].i += tmpHS[j]*co;
		  
		} // end of if (my_prow==prow && my_pcol==pcol)
		
	      } // j

              /* Consider the left bottom block */
	      
              if (spin==2){

                shift_i = 0;
		shift_j = MatDim;

  	        jg = MP[ct_AN] + i + shift_i;
	        bcol2 = (jg-1)/nblk2;
	        pcol2 = bcol2%np_cols2;

		for (j=0; j<TNO2; j++){

		  ig = MP[Gh_AN] + j + shift_j;
	          brow2 = (ig-1)/nblk2;
	          prow2 = brow2%np_rows2;

		  if (my_prow2==prow2 && my_pcol2==pcol2){

		    il = (brow2/np_rows2+1)*nblk2+1;
		    jl = (bcol2/np_cols2+1)*nblk2+1;

		    if (((my_prow2+np_rows2)%np_rows2) >= (brow2%np_rows2)){
		      if(my_prow2==prow2){
			il = il+(ig-1)%nblk2;
		      }
		      il = il-nblk2;
		    }

		    if (((my_pcol2+np_cols2)%np_cols2) >= (bcol2%np_cols2)){
		      if(my_pcol2==pcol2){
			jl = jl+(jg-1)%nblk2;
		      }
		      jl = jl-nblk2;
		    }

		    Hs2[(jl-1)*na_rows2+il-1].r -= tmpHS[j]*si;
		    Hs2[(jl-1)*na_rows2+il-1].i -= tmpHS[j]*co;
		    
		  } // end of if (my_prow==prow && my_pcol==pcol)
		} // j
	      } // if (spin==2)
	      
	    } // i
	  } // h_AN
	} // ct_AN
      } // spin

      /*
      printf("Hs2.r\n");
      for (i=0; i<na_rows2; i++){
	for (j=0; j<na_cols2; j++){
	  printf("%8.4f ",Hs2[j*na_rows2+i].r);    
	}
	printf("\n");
      }

      printf("Hs2.i\n");
      for (i=0; i<na_rows2; i++){
	for (j=0; j<na_cols2; j++){
	  printf("%8.4f ",Hs2[j*na_rows2+i].i);    
	}
	printf("\n");
      }
      */

      /*****************************************
                      set Ss2
      *****************************************/
    
      for (ct_AN=1; ct_AN<=atomnum; ct_AN++){
	TNO1 = Total_NumOrbs[ct_AN];

	for (h_AN=0; h_AN<=FNAN[ct_AN]; h_AN++){

	  Gh_AN = natn[ct_AN][h_AN];
	  Rn = ncn[ct_AN][h_AN];
	  TNO2 = Total_NumOrbs[Gh_AN];

	  l1 = atv_ijk[Rn][1];
	  l2 = atv_ijk[Rn][2];
	  l3 = atv_ijk[Rn][3];

	  kRn = k1*(double)l1 + k2*(double)l2 + k3*(double)l3;
	  si = sin(2.0*PI*kRn);
	  co = cos(2.0*PI*kRn);
	  
	  for (i=0; i<TNO1; i++){

	    FREAD(tmpHS,sizeof(double),TNO2,fp);

            for (p=0; p<=1; p++){              
 
              if      (p==0) { shift_i = 0;      shift_j = 0;      } /* The left top block */       
              else if (p==1) { shift_i = MatDim; shift_j = MatDim; } /* The right bottom block */
	      
	      ig = MP[ct_AN] + i + shift_i;
	      brow2 = (ig-1)/nblk2;
	      prow2 = brow2%np_rows2;
	    
	      for (j=0; j<TNO2; j++){

		jg = MP[Gh_AN] + j + shift_j;
		bcol2 = (jg-1)/nblk2;
		pcol2 = bcol2%np_cols2;

		if (my_prow2==prow2 && my_pcol2==pcol2){

		  il = (brow2/np_rows2+1)*nblk2+1;
		  jl = (bcol2/np_cols2+1)*nblk2+1;

		  if (((my_prow2+np_rows2)%np_rows2) >= (brow2%np_rows2)){
		    if(my_prow2==prow2){
		      il = il+(ig-1)%nblk2;
		    }
		    il = il-nblk2;
		  }

		  if (((my_pcol2+np_cols2)%np_cols2) >= (bcol2%np_cols2)){
		    if(my_pcol2==pcol2){
		      jl = jl+(jg-1)%nblk2;
		    }
		    jl = jl-nblk2;
		  }

		  Ss2[(jl-1)*na_rows2+il-1].r += tmpHS[j]*co;
		  Ss2[(jl-1)*na_rows2+il-1].i += tmpHS[j]*si;
		  
		} // end of if (my_prow2==prow2 && my_pcol2==pcol2)
	      } // j
	    } // p
	    
	  } // i
	} // h_AN
      } // ct_AN

      /*
      printf("Ss2.r\n");
      for (i=0; i<na_rows2; i++){
	for (j=0; j<na_cols2; j++){
	  printf("%8.4f ",Ss2[j*na_rows2+i].r);    
	}
	printf("\n");
      }

      printf("Ss2.i\n");
      for (i=0; i<na_rows2; i++){
	for (j=0; j<na_cols2; j++){
	  printf("%8.4f ",Ss2[j*na_rows2+i].i);    
	}
	printf("\n");
      }
      */
      
      // freeing of arrays

      free(tmpHS);
    
      // fclose of fp
      fclose(fp);
    }

    else {
      printf("Failure of reading the scfout file (%s).\n",argv[1]);fflush(stdout);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    // first_flag = 0;
    first_flag = 0;

  } // end of if (mode=="set")

  else if (mode=="free"){

    free(Hs2);
    free(Cs2);
    free(Ss2);
    free(eval);
    free(MP);

    for (ct_AN=0; ct_AN<=atomnum; ct_AN++){
      free(ncn[ct_AN]);
    }
    free(ncn);
    
    for (ct_AN=0; ct_AN<=atomnum; ct_AN++){
      free(natn[ct_AN]);
    }
    free(natn);

    free(FNAN);
    free(Total_NumOrbs);

    for (Rn=0; Rn<=TCpyCell; Rn++){
      free(atv_ijk[Rn]);
    }
    free(atv_ijk);

    for (Rn=0; Rn<=TCpyCell; Rn++){
      free(atv[Rn]);
    }
    free(atv);
  }
}



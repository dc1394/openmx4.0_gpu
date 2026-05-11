/*************************************************************************
  elpa_col.c:

  elpa_col.c is a program to read a file of *.scfout, and to diagonalize 
  the generalized eigenvalue problem within a collinear DFT calculation. 

  Usage: ./mpirun -np 10 ./elpa_col filename.scfout 

  Log of elpa_col.c:

     15/Nov./2023  Released by T. Ozaki 

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
int TCpyCell,SpinP_switch,MatDim;
int **natn,**ncn,*Total_NumOrbs,*FNAN,**atv_ijk,*MP;
double **atv;
double tv[4][4],rtv[4][4];
double **eval;

/* scalapack */

static int NBLK=128;
/* for DimMat */
int nblk,np_rows,np_cols,na_rows,na_cols,na_rows_max,na_cols_max;
int my_prow,my_pcol;
int bhandle1,bhandle2,ictxt1,ictxt2;
int descS[9],descH[9],descC[9];
dcomplex *Ss,*Hs,*Cs;

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

void set_Ss_Hs(char *mode, char *argv[], int spin_index, double k1, double k2, double k3);

void zheevx_(char *JOBZ, char *RANGE, char *UPLO, int *N, dcomplex *A, int *LDA, double *VL, double *VU, int *IL, int *IU,
                        double *ABSTOL, int *M, double *W, dcomplex *Z, int *LDZ, dcomplex *WORK, int *LWORK, double *RWORK,
                        int *IWORK, int *IFAIL, int *INFO);



int main(int argc, char *argv[]) 
{
  int spin,i,j,jg,po;
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
  
  spin = 0;
  po = 0; 

  /* start do loop which is for the spin loop. */
  
  do {

    /* read the .scfout file and set Ss and Hs */
    
    set_Ss_Hs("set",argv,spin,k1,k2,k3);

    /* diagonalize Ss */

    MPI_Comm_split(MPI_COMM_WORLD,my_pcol,my_prow,&mpi_comm_rows);
    MPI_Comm_split(MPI_COMM_WORLD,my_prow,my_pcol,&mpi_comm_cols);

    mpi_comm_rows_int = MPI_Comm_c2f(mpi_comm_rows);
    mpi_comm_cols_int = MPI_Comm_c2f(mpi_comm_cols);
  
    if (scf_eigen_lib_flag==1){

      F77_NAME(solve_evp_complex,SOLVE_EVP_COMPLEX)
	( &MatDim, &MatDim, Ss, &na_rows, eval[spin], Cs, &na_rows, &nblk,
	  &mpi_comm_rows_int, &mpi_comm_cols_int );
    }

    else if (scf_eigen_lib_flag==2){

      int mpiworld;
      mpiworld = MPI_Comm_c2f(MPI_COMM_WORLD);
      F77_NAME(elpa_solve_evp_complex_2stage_double_impl,ELPA_SOLVE_EVP_COMPLEX_2STAGE_DOUBLE_IMPL)
	( &MatDim, &MatDim, Ss, &na_rows, eval[spin], Cs, &na_rows, &nblk, &na_cols, 
	  &mpi_comm_rows_int, &mpi_comm_cols_int, &mpiworld );

    }

    /*
    for (i=0; i<MatDim; i++){
      printf("ABC1 %2d %15.12f\n",i,eval[spin][i]);
    }
    */
  
    /* calculate (eigenvectors of S) times 1/sqrt(eval) */
  
    for (i=0; i<MatDim; i++){
      eval[spin][i] = 1.0/sqrt(eval[spin][i]);
    }

    for (i=0; i<na_rows; i++){
      for (j=0; j<na_cols; j++){
	jg = np_cols*nblk*((j)/nblk) + (j)%nblk + ((np_cols+my_pcol)%np_cols)*nblk;
	Ss[j*na_rows+i].r = Cs[j*na_rows+i].r*eval[spin][jg];
	Ss[j*na_rows+i].i = Cs[j*na_rows+i].i*eval[spin][jg];
      }
    }

    /* transformation of H by Ss */
    /* pzgemm:  H * U * 1.0/sqrt(eval) */

    for(i=0;i<na_rows_max*na_cols_max;i++){
      Cs[i].r = 0.0;
      Cs[i].i = 0.0;
    }
  
    Cblacs_barrier(ictxt2,"A");
    F77_NAME(pzgemm,PZGEMM)("N","N",&MatDim,&MatDim,&MatDim,&alpha,Hs,&ONE,&ONE,descH,Ss,
			    &ONE,&ONE,descS,&beta,Cs,&ONE,&ONE,descC);

    /* 1.0/sqrt(ko[l]) * U^+ H * U * 1.0/sqrt(eval) */

    for(i=0;i<na_rows*na_cols;i++){
      Hs[i].r = 0.0;
      Hs[i].i = 0.0;
    }

    Cblacs_barrier(ictxt2,"C");
    F77_NAME(pzgemm,PZGEMM)("C","N",&MatDim,&MatDim,&MatDim,&alpha,Ss,&ONE,&ONE,descS,Cs,
			    &ONE,&ONE,descC,&beta,Hs,&ONE,&ONE,descH);
  
    /* diagonalize H' */

    if (scf_eigen_lib_flag==1){

      F77_NAME(solve_evp_complex,SOLVE_EVP_COMPLEX)
	( &MatDim, &MatDim, Hs, &na_rows, eval[spin], Cs,
	  &na_rows, &nblk, &mpi_comm_rows_int, &mpi_comm_cols_int );
    }
    else if (scf_eigen_lib_flag==2){
      int mpiworld;
      mpiworld = MPI_Comm_c2f(MPI_COMM_WORLD);
      F77_NAME(elpa_solve_evp_complex_2stage_double_impl,ELPA_SOLVE_EVP_COMPLEX_2STAGE_DOUBLE_IMPL)
	( &MatDim, &MatDim, Hs, &na_rows, eval[spin], Cs, &na_rows, &nblk, &na_cols,
	  &mpi_comm_rows_int, &mpi_comm_cols_int, &mpiworld );
    }

    if (myid==0){  
      for (i=0; i<MatDim; i++){
        printf("spin=%2d i=%2d eigenvalues=%15.12f\n",spin,i+1,eval[spin][i]);
      }
    }

    MPI_Comm_free(&mpi_comm_rows);
    MPI_Comm_free(&mpi_comm_cols);

    /* control the do loop */
    
    if (SpinP_switch==1 && spin==0){
      spin++;
    }
    else{					    
      po = 1;
    }
    
  } while (po==0);
  
  /* free arrays */

  set_Ss_Hs("free",argv,spin,k1,k2,k3);

  /* finalize MPI and exit */

  MPI_Finalize();
  exit(0);
}




void set_Ss_Hs(char *mode, char *argv[], int spin_index, double k1, double k2, double k3)
{
  static int first_flag=1;
  int ct_AN,Rn,version,order_max;
  int n1,n2,n3,spin,i,j,TNO1,TNO2;
  int h_AN,Gh_AN;
  int ig,jg,il,jl,l1,l2,l3;
  int brow,bcol,prow,pcol;
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

      if (SpinP_switch==3){
	if (myid==0){
	  printf("The code is valid for the collinear case.\n");    
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

      if (first_flag){
	eval = (double**)malloc(sizeof(double*)*(SpinP_switch+1));
	for (i=0; i<(SpinP_switch+1); i++){
	  eval[i] = (double*)malloc(sizeof(double)*(MatDim+3));
	}
      }
	
      np_cols = (int)(sqrt((float)numprocs));
      do{
	if((numprocs%np_cols)==0) break;
	np_cols--;
      } while (np_cols>=2);
      np_rows = numprocs/np_cols;
    
      nblk_m = NBLK;
      while((nblk_m*np_rows>MatDim || nblk_m*np_cols>MatDim) && (nblk_m > 1)){
	nblk_m /= 2;
      }
      if(nblk_m<1) nblk_m = 1;

      MPI_Allreduce(&nblk_m,&nblk,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD);

      my_prow = myid/np_cols;
      my_pcol = myid%np_cols;

      na_rows = numroc_(&MatDim, &nblk, &my_prow, &ZERO, &np_rows);
      na_cols = numroc_(&MatDim, &nblk, &my_pcol, &ZERO, &np_cols );

      bhandle2 = Csys2blacs_handle(MPI_COMM_WORLD);
      ictxt2 = bhandle2;
      Cblacs_gridinit(&ictxt2, "Row", np_rows, np_cols);

      if (first_flag){
        Ss = (dcomplex*)malloc(sizeof(dcomplex)*na_rows*na_cols);
        Hs = (dcomplex*)malloc(sizeof(dcomplex)*na_rows*na_cols);
      }
      
      for (i=0; i<na_rows*na_cols; i++){
	Ss[i].r = 0.0; Ss[i].i = 0.0;
	Hs[i].r = 0.0; Hs[i].i = 0.0;
      }

      MPI_Allreduce(&na_rows,&na_rows_max,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
      MPI_Allreduce(&na_cols,&na_cols_max,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);

      if (first_flag){
        Cs = (dcomplex*)malloc(sizeof(dcomplex)*na_rows_max*na_cols_max);
      }
      
      for (i=0; i<na_rows_max*na_cols_max; i++){
	Cs[i].r = 0.0; Cs[i].i = 0.0;
      }

      descinit_(descS, &MatDim, &MatDim, &nblk, &nblk, &ZERO, &ZERO, &ictxt2, &na_rows, &info);
      descinit_(descH, &MatDim, &MatDim, &nblk, &nblk, &ZERO, &ZERO, &ictxt2, &na_rows, &info);
      descinit_(descC, &MatDim, &MatDim, &nblk, &nblk, &ZERO, &ZERO, &ictxt2, &na_rows, &info);

      /*****************************************
                     set Hs
      *****************************************/

      double *tmpHS; 
      tmpHS = (double*)malloc(sizeof(double)*maxTNO);

      for (spin=0; spin<=SpinP_switch; spin++){
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

	      ig = MP[ct_AN] + i;
	      brow = (ig-1)/nblk;
	      prow = brow%np_rows;

	      FREAD(tmpHS,sizeof(double),TNO2,fp);

	      if (spin==spin_index){

		for (j=0; j<TNO2; j++){
		  jg = MP[Gh_AN] + j;
		  bcol = (jg-1)/nblk;
		  pcol = bcol%np_cols;

		  if (my_prow==prow && my_pcol==pcol){

		    il = (brow/np_rows+1)*nblk+1;
		    jl = (bcol/np_cols+1)*nblk+1;

		    if (((my_prow+np_rows)%np_rows) >= (brow%np_rows)){
		      if(my_prow==prow){
			il = il+(ig-1)%nblk;
		      }
		      il = il-nblk;
		    }

		    if (((my_pcol+np_cols)%np_cols) >= (bcol%np_cols)){
		      if(my_pcol==pcol){
			jl = jl+(jg-1)%nblk;
		      }
		      jl = jl-nblk;
		    }

		    Hs[(jl-1)*na_rows+il-1].r += tmpHS[j]*co;
		    Hs[(jl-1)*na_rows+il-1].i += tmpHS[j]*si;
		  
		  } // end of if (my_prow==prow && my_pcol==pcol)
		
		} // j
	      } // end of if (spin==spin_index)
	    } // i
	  } // h_AN
	} // ct_AN
      } // spin

      /*
      printf("Hs.r\n");
      for (i=0; i<na_rows; i++){
	for (j=0; j<na_cols; j++){
	  printf("%8.4f ",Hs[j*na_rows+i].r);    
	}
	printf("\n");
      }
      */

      /*****************************************
                     set Ss
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

	    ig = MP[ct_AN] + i;
	    brow = (ig-1)/nblk;
	    prow = brow%np_rows;

	    FREAD(tmpHS,sizeof(double),TNO2,fp);

	    for (j=0; j<TNO2; j++){
	      jg = MP[Gh_AN] + j;
	      bcol = (jg-1)/nblk;
	      pcol = bcol%np_cols;

	      if (my_prow==prow && my_pcol==pcol){

		il = (brow/np_rows+1)*nblk+1;
		jl = (bcol/np_cols+1)*nblk+1;

		if (((my_prow+np_rows)%np_rows) >= (brow%np_rows)){
		  if(my_prow==prow){
		    il = il+(ig-1)%nblk;
		  }
		  il = il-nblk;
		}

		if (((my_pcol+np_cols)%np_cols) >= (bcol%np_cols)){
		  if(my_pcol==pcol){
		    jl = jl+(jg-1)%nblk;
		  }
		  jl = jl-nblk;
		}

		Ss[(jl-1)*na_rows+il-1].r += tmpHS[j]*co;
		Ss[(jl-1)*na_rows+il-1].i += tmpHS[j]*si;
		  
	      } // end of if (my_prow==prow && my_pcol==pcol)

	    } // j
	  } // i
	} // h_AN
      } // ct_AN

      /*
      printf("Ss.r\n");
      for (i=0; i<na_rows; i++){
	for (j=0; j<na_cols; j++){
	  printf("%8.4f ",Ss[j*na_rows+i].r);    
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

    free(Cs);
    free(Hs);
    free(Ss);

    for (i=0; i<(SpinP_switch+1); i++){
      free(eval[i]);
    }
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



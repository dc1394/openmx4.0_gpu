/**********************************************************************
  Electric_Polarization_BandNonCol.c:

   Electric_Polarization_BandCol.c is a subroutine to calculate polarization 
   using the Analytic Berry Connection (ABC) formula for the occupied space
   in the non-collinear band calculation. 

  Log of Electric_Polarization_BandNonCol.c:

    11/April/2025  Released by T. Ozaki

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <complex.h>
#include "mpi.h"
#include "openmx_common.h"
#include "lapack_prototypes.h"
#include "tran_variables.h"
#include <omp.h>

#define  measure_time  0

int nblk11,np_rows11,np_cols11,na_rows11,na_cols11;
int na_rows_max11,na_cols_max11;
int my_prow11,my_pcol11;
int bhandle11,ictxt11;
int desc11[9];
int nblk12,np_rows12,np_cols12,na_rows12,na_cols12;
int na_rows_max12,na_cols_max12;
int my_prow12,my_pcol12;
int bhandle12,ictxt12;
int desc12[9];
double ******OLPpo;

void Calc_OLPpo( double ******OLPpo );
void Allocate_Free_Electric_Polarization(int todo_flag);
static double EP1( double ****OLP0, double ******OLPpo, double *****CDM, 
   	           double EPol[3], double CPol[3], double EPol_BG[3] );

void solve_evp_complex_( int *n2, int *MaxN, dcomplex *Hs2, int *na_rows2_1, double *a, dcomplex *Cs2, int *na_rows2_2, 
                         int *nblk2, int *mpi_comm_rows_int, int *mpi_comm_cols_int );

void elpa_solve_evp_complex_2stage_double_impl_( int *n2, int *MaxN, dcomplex *Hs2, int *na_rows2_1, double *a, dcomplex *Cs2, 
                                                 int *na_rows2_2, int *nblk2, int *na_cols2, 
                                                 int *mpi_comm_rows_int, int *mpi_comm_cols_int, int *mpiworld );


static void Construct_Band_Ms( int cpx_flag, double ****Mat, double *M1, dcomplex *Ms, 
                               int *MP, double k1, double k2, double k3 );
static void Construct_Band_Mps( int cpx_flag, int p, 
				double ****Mat, double *M1, dcomplex *Ms, 
				int *MP, double k1, double k2, double k3 );
static void Construct_Band_DMs( int im_flag, double ****Mat, double *M1, dcomplex *DMs, int *MP );

static void Analytic_Berry_Connection( int Nocc, int add_flag,
				       int kloop, int n2, int MaxN, 
				       int *MP, int *T_k_op,
				       double k1, double k2, double k3,
				       double *H1, 
				       dcomplex *Ss2, dcomplex *Cs2, dcomplex *Hs2,  
				       dcomplex *Vs2, dcomplex *Qs2, 
				       dcomplex *Hs11, dcomplex *Hs22, dcomplex *Hs12,
				       double ***EIGEN,
				       double *****nh,double *****ImNL,double ****CntOLP,
				       dcomplex *ABC,
				       dcomplex *VDMs_ABC,
				       int *order_GA, 
				       int all_knum, 
				       int myworld1,
				       int myworld2,
				       MPI_Comm *MPI_CommWD1,
				       MPI_Comm *MPI_CommWD2 );



#pragma GCC optimize("O2")
double Electric_Polarization_BandNonCol(
                    int SCF_iter,
                    int knum_i, int knum_j, int knum_k,
		    int SpinP_switch,
		    double *****nh,
		    double *****ImNL,
		    double ****CntOLP,
		    double *****CDM,
		    double *****EDM,
		    double Eele0[2], double Eele1[2], 
		    int *MP,
		    int *order_GA,
		    double *ko,
		    double *koS,
		    double ***EIGEN,
		    double *H1,   
		    dcomplex *Hs11,   
		    dcomplex *Hs22,   
		    dcomplex *Hs12,   
		    dcomplex *Ss,
		    dcomplex *Cs,
                    dcomplex *Hs,
		    dcomplex *Ss2,
		    dcomplex *Cs2,
                    dcomplex *Hs2,
		    int ***k_op,
		    int *T_k_op,
		    int **T_k_ID,
		    double *T_KGrids1,
		    double *T_KGrids2,
		    double *T_KGrids3,
                    int myworld1,
		    int *NPROCS_ID1,
		    int *Comm_World1,
		    int *NPROCS_WD1,
		    int *Comm_World_StartID1,
		    MPI_Comm *MPI_CommWD1,
                    int myworld2,
		    int *NPROCS_ID2,
		    int *NPROCS_WD2,
		    int *Comm_World2,
		    int *Comm_World_StartID2,
		    MPI_Comm *MPI_CommWD2)
{
  static int firsttime=1;
  int i,j,k,l,m,n,n2,p,wan,MaxN,i0,ks;
  int i1,i1s,j1,ia,jb,lmax,po,po1,spin,s1,e1;
  int num2,RnB,l1,l2,l3,loop_num,ns,ne;
  int ct_AN,h_AN,wanA,tnoA,wanB,tnoB,m1,m2,HOMO;
  int MA_AN,GA_AN,Anum,num_kloop0,max_num_kloop0;
  int T_knum,S_knum,E_knum,kloop,kloop0;
  double av_num,lumos;
  double time0;
  int LB_AN,GB_AN,Bnum;
  double k1,k2,k3,Fkw;
  double sum,sumi,sum_weights;
  double Num_State;
  double My_Num_State;
  double FermiF,be[4],tmparray[3];
  double tmp,tmp1,eig,kw,EV_cut0;
  double x,Dnum,Dnum2,AcP,ChemP_MAX,ChemP_MIN;
  dcomplex *DMs11,*DMs22,*DMs12;
  dcomplex *ABC,*Vs2,*Qs2,*VDMs_ABC;
  int *is1,*ie1;
  int *is2,*ie2;
  int *My_NZeros;
  int *SP_NZeros;
  int *SP_Atoms;

  int all_knum; 
  dcomplex Ctmp1,Ctmp2;
  int ii,ij,ik;
  int BM,BN,BK;
  double u2,v2,uv,vu;
  double d1,d2;
  double My_Eele1[2]; 
  double TZ,dum,sumE,kRn,si,co;
  double Resum,ResumE,Redum,Redum2;
  double Imsum,ImsumE,Imdum,Imdum2;
  double TStime,TEtime,SiloopTime,EiloopTime;
  double Stime,Etime,Stime0,Etime0;
  double x_cut=60.0;
  double My_Eele0[2];

  char file_EV[YOUSO10];
  FILE *fp_EV;
  char buf[fp_bsize];          /* setvbuf */

  int AN,Rn;
  int parallel_mode;
  int numprocs0,myid0;
  int ID,ID0,ID1;
  int numprocs1,myid1;
  int numprocs2,myid2;
  int Num_Comm_World1;
  int Num_Comm_World2;

  int tag=999,IDS,IDR;
  MPI_Status stat;
  MPI_Request request;

  double time1,time2,time3;
  double time4,time5,time6;
  double time7,time8,time9;
  double time10,time11,time12,time13;

  /* for OpenMP */
  int OMPID,Nthrds,Nprocs;

  FILE* file;
  char* BUF[1000];

  MPI_Comm mpi_comm_rows, mpi_comm_cols;
  int mpi_comm_rows_int,mpi_comm_cols_int;
  int info,ig,jg,il,jl,prow,pcol,brow,bcol;
  int ZERO=0, ONE=1;
  dcomplex alpha = {1.0,0.0}; dcomplex beta = {0.0,0.0};
  int LOCr, LOCc, node, irow, icol;
  double mC_spin_i1,C_spin_i1;
  double *rDM11,*rDM22,*rDM12,*iDM12,*iDM11,*iDM22,*rEDM11,*rEDM22;
  double EPol[3],CPol[3],EPol_BG[3];

  /* for time */
  dtime(&TStime);

  time1 = 0.0;
  time2 = 0.0;
  time3 = 0.0;
  time4 = 0.0;
  time5 = 0.0;
  time6 = 0.0;
  time7 = 0.0;
  time8 = 0.0;
  time9 = 0.0;
  time10 = 0.0;
  time11 = 0.0;
  time12 = 0.0;
  time13 = 0.0;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs0);
  MPI_Comm_rank(mpi_comm_level1,&myid0);
  MPI_Barrier(mpi_comm_level1);

  Num_Comm_World1 = 1;

  /*********************************************** 
       for pallalel calculations in myworld1
  ***********************************************/

  MPI_Comm_size(MPI_CommWD1[myworld1],&numprocs1);
  MPI_Comm_rank(MPI_CommWD1[myworld1],&myid1);

  /****************************************************
   find the number of basis functions, n
  ****************************************************/

  n = 0;
  for (i=1; i<=atomnum; i++){
    wanA  = WhatSpecies[i];
    n += Spe_Total_CNO[wanA];
  }
  n2 = n*2; 

  /****************************************************
   find TZ
  ****************************************************/

  TZ = 0.0;
  for (i=1; i<=atomnum; i++){
    wan = WhatSpecies[i];
    TZ += Spe_Core_Charge[wan];
  }

  /***********************************************
     find the number of states to be solved 
  ***********************************************/

  MaxN = n2;
  HOMO = TZ - system_charge;

  /***********************************************
     allocation of arrays
  ***********************************************/

  My_NZeros = (int*)malloc(sizeof(int)*numprocs0);
  SP_NZeros = (int*)malloc(sizeof(int)*numprocs0);
  SP_Atoms = (int*)malloc(sizeof(int)*numprocs0);

  /***********************************************
              k-points by regular mesh 
  ***********************************************/

  for (i=0;i<knum_i;i++) {
    for (j=0;j<knum_j;j++) {
      for (k=0;k<knum_k;k++) {
	k_op[i][j][k] = 1;
      }
    }
  }

  /***********************************
       one-dimentionalize for MPI
  ************************************/

  T_knum = 0;
  for (i=0; i<knum_i; i++){
    for (j=0; j<knum_j; j++){
      for (k=0; k<knum_k; k++){
	if (0<k_op[i][j][k]){
	  T_knum++;
	}
      }
    }
  }

  /* set T_KGrids1,2,3 and T_k_op */

  /* Added by N. Yamaguchi ***/
  if (way_of_kpoint==1){
    /* ***/

    T_knum = 0;
    for (i=0; i<knum_i; i++){

      if (knum_i==1)  k1 = 0.0;
      else            k1 = -0.5 + (2.0*(double)i+1.0)/(2.0*(double)knum_i) + Shift_K_Point;

      for (j=0; j<knum_j; j++){

	if (knum_j==1)  k2 = 0.0;
	else            k2 = -0.5 + (2.0*(double)j+1.0)/(2.0*(double)knum_j) - Shift_K_Point;

	for (k=0; k<knum_k; k++){

	  if (knum_k==1)  k3 = 0.0;
	  else            k3 = -0.5 + (2.0*(double)k+1.0)/(2.0*(double)knum_k) + 2.0*Shift_K_Point;

	  if (0<k_op[i][j][k]){

	    T_KGrids1[T_knum] = k1;
	    T_KGrids2[T_knum] = k2;
	    T_KGrids3[T_knum] = k3;
	    T_k_op[T_knum]    = k_op[i][j][k];

	    T_knum++;
	  }
	}
      }
    }

    if (myid0==Host_ID && 0<level_stdout){

      printf(" KGrids1: ");fflush(stdout);
      for (i=0;i<=knum_i-1;i++){
	if (knum_i==1)  k1 = 0.0;
	else            k1 = -0.5 + (2.0*(double)i+1.0)/(2.0*(double)knum_i) + Shift_K_Point;
	printf("%9.5f ",k1);fflush(stdout);
      }
      printf("\n");fflush(stdout);

      printf(" KGrids2: ");fflush(stdout);

      for (i=0;i<=knum_j-1;i++){
	if (knum_j==1)  k2 = 0.0;
	else            k2 = -0.5 + (2.0*(double)i+1.0)/(2.0*(double)knum_j) - Shift_K_Point;
	printf("%9.5f ",k2);fflush(stdout);
      }
      printf("\n");fflush(stdout);

      printf(" KGrids3: ");fflush(stdout);
      for (i=0;i<=knum_k-1;i++){
	if (knum_k==1)  k3 = 0.0;
	else            k3 = -0.5 + (2.0*(double)i+1.0)/(2.0*(double)knum_k) + 2.0*Shift_K_Point;
	printf("%9.5f ",k3);fflush(stdout);
      }
      printf("\n");fflush(stdout);
    }
  } // end of if (way_of_kpoint==1)

  /* Added by N. Yamaguchi ***/
  /***********************************************
          k-points by a Gamma-centered mesh
   ***********************************************/
  
  else if (way_of_kpoint==3){
    
    T_knum = 0;
    for (i=0; i<knum_i; i++){

      if (knum_i==1)  k1 = 0.0;
      else            k1 = ((double)i)/((double)knum_i) + Shift_K_Point;

      for (j=0; j<knum_j; j++){

        if (knum_j==1)  k2 = 0.0;
        else            k2 = ((double)j)/((double)knum_j) - Shift_K_Point;

        for (k=0; k<knum_k; k++){

          if (knum_k==1)  k3 = 0.0;
          else            k3 = ((double)k)/((double)knum_k) + 2.0*Shift_K_Point;

          if (0<k_op[i][j][k]){

            T_KGrids1[T_knum] = k1;
            T_KGrids2[T_knum] = k2;
            T_KGrids3[T_knum] = k3;
            T_k_op[T_knum]    = k_op[i][j][k];

            T_knum++;
          }
        }
      }
    }

   if (myid0==Host_ID && 0<level_stdout){

      printf(" KGrids1: ");fflush(stdout);
      for (i=0;i<=knum_i-1;i++){
        if (knum_i==1)  k1 = 0.0;
        else            k1 = ((double)i)/((double)knum_i) + Shift_K_Point;
        printf("%9.5f ",k1);fflush(stdout);
      }
      printf("\n");fflush(stdout);

      printf(" KGrids2: ");fflush(stdout);

      for (i=0;i<=knum_j-1;i++){
        if (knum_j==1)  k2 = 0.0;
        else            k2 = ((double)i)/((double)knum_j) - Shift_K_Point;
        printf("%9.5f ",k2);fflush(stdout);
      }
      printf("\n");fflush(stdout);

      printf(" KGrids3: ");fflush(stdout);
      for (i=0;i<=knum_k-1;i++){
        if (knum_k==1)  k3 = 0.0;
        else            k3 = ((double)i)/((double)knum_k) + 2.0*Shift_K_Point;
        printf("%9.5f ",k3);fflush(stdout);
      }
      printf("\n");fflush(stdout);
    }
  } // end of else if (way_of_kpoint==3)
  /* ***/

  /***********************************************
            calculate the sum of weights
  ***********************************************/

  sum_weights = 0.0;
  for (k=0; k<T_knum; k++){
    sum_weights += (double)T_k_op[k];
  }

  /***********************************************
           allocate k-points into processors 
  ***********************************************/

  if (numprocs1<T_knum){

    /* set parallel_mode */
    parallel_mode = 0;

    /* allocation of kloop to ID */     

    for (ID=0; ID<numprocs1; ID++){
      tmp = (double)T_knum/(double)numprocs1;
      S_knum = (int)((double)ID*(tmp+1.0e-12)); 
      E_knum = (int)((double)(ID+1)*(tmp+1.0e-12)) - 1;
      if (ID==(numprocs1-1)) E_knum = T_knum - 1;
      if (E_knum<0)          E_knum = 0;

      for (k=S_knum; k<=E_knum; k++){
        /* ID in the first level world */
        T_k_ID[myworld1][k] = ID;
      }
    }

    /* find own informations */

    tmp = (double)T_knum/(double)numprocs1; 
    S_knum = (int)((double)myid1*(tmp+1.0e-12)); 
    E_knum = (int)((double)(myid1+1)*(tmp+1.0e-12)) - 1;
    if (myid1==(numprocs1-1)) E_knum = T_knum - 1;
    if (E_knum<0)             E_knum = 0;

    num_kloop0 = E_knum - S_knum + 1;

    MPI_Comm_size(MPI_CommWD2[myworld2],&numprocs2);
    MPI_Comm_rank(MPI_CommWD2[myworld2],&myid2);
  }

  else {

    /* set parallel_mode */
    parallel_mode = 1;
    num_kloop0 = 1;

    Num_Comm_World2 = T_knum;
    MPI_Comm_size(MPI_CommWD2[myworld2],&numprocs2);
    MPI_Comm_rank(MPI_CommWD2[myworld2],&myid2);

    S_knum = myworld2;

    /* allocate k-points into processors */
    
    for (k=0; k<T_knum; k++){
      /* ID in the first level world */
      T_k_ID[myworld1][k] = Comm_World_StartID2[k];
    }
  }

  /****************************************************
   find all_knum
   if (all_knum==1), all the calculation will be made 
   by the first diagonalization loop, and the second 
   diagonalization will be skipped. 
  ****************************************************/

  MPI_Allreduce(&num_kloop0, &all_knum, 1, MPI_INT, MPI_PROD, mpi_comm_level1);
  MPI_Allreduce(&num_kloop0, &max_num_kloop0, 1, MPI_INT, MPI_MAX, mpi_comm_level1);

  /****************************************************
                allocation of arrays
  ****************************************************/

  DMs11 = (dcomplex*)malloc(sizeof(dcomplex)*na_rows*na_cols);
  DMs22 = (dcomplex*)malloc(sizeof(dcomplex)*na_rows*na_cols);
  DMs12 = (dcomplex*)malloc(sizeof(dcomplex)*na_rows*na_cols);
  VDMs_ABC = (dcomplex*)malloc(sizeof(dcomplex)*na_rows2*na_cols2);

  ABC = (dcomplex*)malloc(sizeof(dcomplex)*3);
  for (i=0; i<3; i++){ ABC[i] = Complex(0.0,0.0); }

  Vs2 = (dcomplex*)malloc(sizeof(dcomplex)*na_rows_max2*na_cols_max2);
  Qs2 = (dcomplex*)malloc(sizeof(dcomplex)*na_rows_max2*na_cols_max2);

  /****************************************************
                      PrintMemory
  ****************************************************/

  if (firsttime && memoryusage_fileout) {
    PrintMemory("Electric_Polarization_BandNonCol: My_NZeros", sizeof(int)*numprocs0,NULL);
    PrintMemory("Electric_Polarization_BandNonCol: SP_NZeros", sizeof(int)*numprocs0,NULL);
    PrintMemory("Electric_Polarization_BandNonCol: SP_Atoms", sizeof(int)*numprocs0,NULL);
  }

  /****************************************************
   calculation of projection functions:
   projection functions must have different phase 
   factors in each myworld2. However, the phase factors
   are cancelled out in the ABC formula, which is a very
   nice aspect in the formula.  
  ****************************************************/

  /* set DMs */

  for (i=0; i<na_rows*na_cols; i++){
    DMs11[i] = Complex(0.0,0.0);
    DMs22[i] = Complex(0.0,0.0);
    DMs12[i] = Complex(0.0,0.0);
  }

  Construct_Band_DMs(0,CDM[0],   H1,DMs11,MP);
  Construct_Band_DMs(0,CDM[1],   H1,DMs22,MP);
  Construct_Band_DMs(0,CDM[2],   H1,DMs12,MP);
  Construct_Band_DMs(1,CDM[3],   H1,DMs12,MP);
  Construct_Band_DMs(1,iDM[0][0],H1,DMs11,MP);
  Construct_Band_DMs(1,iDM[0][1],H1,DMs22,MP);

  Hamiltonian_Band_NC_Hs2( DMs11, DMs22, DMs12, Hs2, MPI_CommWD2[myworld2] );

  /* diagonalize Hs2 */

  MPI_Comm_split(MPI_CommWD2[myworld2],my_pcol2,my_prow2,&mpi_comm_rows);
  MPI_Comm_split(MPI_CommWD2[myworld2],my_prow2,my_pcol2,&mpi_comm_cols);

  mpi_comm_rows_int = MPI_Comm_c2f(mpi_comm_rows);
  mpi_comm_cols_int = MPI_Comm_c2f(mpi_comm_cols);

  if (scf_eigen_lib_flag==1 || numprocs2<5){
    F77_NAME(solve_evp_complex,SOLVE_EVP_COMPLEX)
      ( &n2, &MaxN, Hs2, &na_rows2, &ko[1], VDMs_ABC, &na_rows2, &nblk2, &mpi_comm_rows_int, &mpi_comm_cols_int );
  }
      
  else if (scf_eigen_lib_flag==2){

#ifndef kcomp
    int mpiworld;
    mpiworld = MPI_Comm_c2f(MPI_CommWD2[myworld2]);
    F77_NAME(elpa_solve_evp_complex_2stage_double_impl,ELPA_SOLVE_EVP_COMPLEX_2STAGE_DOUBLE_IMPL)
      ( &n2, &MaxN, Hs2, &na_rows2, &ko[1], VDMs_ABC, &na_rows2, &nblk2, &na_cols2, 
	&mpi_comm_rows_int, &mpi_comm_cols_int, &mpiworld );
#else
    F77_NAME(solve_evp_complex,SOLVE_EVP_COMPLEX)
      ( &n2, &MaxN, Hs2, &na_rows2, &ko[1], VDMs_ABC, &na_rows2, &nblk2, &mpi_comm_rows_int, &mpi_comm_cols_int );
#endif

  }

  MPI_Comm_free(&mpi_comm_rows);
  MPI_Comm_free(&mpi_comm_cols);

  /********************************************************
                   set up BLACS_variables
  *********************************************************/

  /* A11: HOMO x HOMO */

  m1 = HOMO;
  m2 = HOMO;

  Set_BLACS_variables( MPI_CommWD2[myworld2],numprocs2,myid2,m1,m2,&np_rows11,&np_cols11,
		       &nblk11,&my_prow11,&my_pcol11,&na_rows11,&na_cols11,
		       &bhandle11,&ictxt11,&na_rows_max11,&na_cols_max11,desc11 );

  /* A12: HOMO x (n2-HOMO) */ 

  m1 = HOMO;
  m2 = n2 - HOMO;

  Set_BLACS_variables( MPI_CommWD2[myworld2],numprocs2,myid2,m1,m2,&np_rows12,&np_cols12,
		       &nblk12,&my_prow12,&my_pcol12,&na_rows12,&na_cols12,
		       &bhandle12,&ictxt12,&na_rows_max12,&na_cols_max12,desc12 );

  desc12[1] = desc11[1]; 

  /****************************************************
   ****************************************************
   if (all_knum==1) start kloop
   ****************************************************
  ****************************************************/

  if (all_knum==1){

    dtime(&SiloopTime);

    for (kloop0=0; kloop0<max_num_kloop0; kloop0++){

      /* get k1, k2, and k3 */

      if (kloop0<num_kloop0){

	kloop = S_knum + kloop0;
	k1 = T_KGrids1[kloop];
	k2 = T_KGrids2[kloop];
	k3 = T_KGrids3[kloop];
      }

      if (measure_time) dtime(&Stime);

      if (SCF_iter==1 || all_knum!=1){

	/* make Cs */

        for(i=0; i<na_rows*na_cols; i++){ Cs[i] = Complex(0.0,0.0);}
	Construct_Band_Ms(0,CntOLP,H1,Cs,MP,k1,k2,k3);

	/* diagonalize Cs */

	if (kloop0<num_kloop0){

	  MPI_Comm_split(MPI_CommWD2[myworld2],my_pcol,my_prow,&mpi_comm_rows);
	  MPI_Comm_split(MPI_CommWD2[myworld2],my_prow,my_pcol,&mpi_comm_cols);

	  mpi_comm_rows_int = MPI_Comm_c2f(mpi_comm_rows);
	  mpi_comm_cols_int = MPI_Comm_c2f(mpi_comm_cols);

	  if (scf_eigen_lib_flag==1 || numprocs2<5){
	    F77_NAME(solve_evp_complex,SOLVE_EVP_COMPLEX)
	      ( &n, &n, Cs, &na_rows, &ko[1], Ss, &na_rows, &nblk, &mpi_comm_rows_int, &mpi_comm_cols_int );
	  }
	
	  else if (scf_eigen_lib_flag==2){

#ifndef kcomp
	    int mpiworld;
	    mpiworld = MPI_Comm_c2f(MPI_CommWD2[myworld2]);
	    F77_NAME(elpa_solve_evp_complex_2stage_double_impl,ELPA_SOLVE_EVP_COMPLEX_2STAGE_DOUBLE_IMPL)
	      ( &n, &n, Cs, &na_rows, &ko[1], Ss, &na_rows, &nblk, &na_cols, 
		&mpi_comm_rows_int, &mpi_comm_cols_int, &mpiworld );
#else
	    F77_NAME(solve_evp_complex,SOLVE_EVP_COMPLEX)
	      ( &n, &n, Cs, &na_rows, &ko[1], Ss, &na_rows, &nblk, &mpi_comm_rows_int, &mpi_comm_cols_int );
#endif
	  
	  }

	  MPI_Comm_free(&mpi_comm_rows);
	  MPI_Comm_free(&mpi_comm_cols);

	  /* print to the standard output */

	  if (3<=level_stdout){
	    printf(" myid0=%2d kloop %2d  k1 k2 k3 %10.6f %10.6f %10.6f\n",
		   myid0,kloop,T_KGrids1[kloop],T_KGrids2[kloop],T_KGrids3[kloop]);
	    for (i=1; i<=n; i++){
	      printf("  Eigenvalues of OLP  %2d  %15.12f\n",i,ko[i]);
	    }
	  }

	  /* minus eigenvalues to 1.0e-10 */

	  for (l=1; l<=n; l++){
	    if (ko[l]<1.0e-10) ko[l] = 1.0e-10;
	    ko[l] = 1.0/sqrt(ko[l]);
	  }

	  /* calculate S*1/sqrt(ko) */

	  for(i=0; i<na_rows; i++){
	    for(j=0; j<na_cols; j++){
	      jg = np_cols*nblk*((j)/nblk) + (j)%nblk + ((np_cols+my_pcol)%np_cols)*nblk + 1;
	      Ss[j*na_rows+i].r = Ss[j*na_rows+i].r*ko[jg];
	      Ss[j*na_rows+i].i = Ss[j*na_rows+i].i*ko[jg];
	    }
	  }

	  /* make Ss2 */

	  Overlap_Band_NC_Ss2( Ss, Ss2, MPI_CommWD2[myworld2] );
	}
      }

      if (measure_time){
	dtime(&Etime);
	time1 += Etime - Stime;
      }

      /* ***************************************************
	 transformation of H with Ss

	 in case of SO_switch==0 && Hub_U_switch==0 && Constraint_NCS_switch==0 
	 && Zeeman_NCS_switch==0 && Zeeman_NCO_switch==0
 
	 H[i    ][j    ].r = RH[0];
	 H[i    ][j    ].i = 0.0;
	 H[i+NUM][j+NUM].r = RH[1];
	 H[i+NUM][j+NUM].i = 0.0;
	 H[i    ][j+NUM].r = RH[2];
	 H[i    ][j+NUM].i = RH[3];

	 in case of SO_switch==1 or Hub_U_switch==1 or 1<=Constraint_NCS_switch 
	 or Zeeman_NCS_switch==1 or Zeeman_NCO_switch==1 

	 H[i    ][j    ].r = RH[0];  
	 H[i    ][j    ].i = IH[0];
	 H[i+NUM][j+NUM].r = RH[1];
	 H[i+NUM][j+NUM].i = IH[1];
	 H[i    ][j+NUM].r = RH[2];
	 H[i    ][j+NUM].i = RH[3] + IH[2];
	 *************************************************** */

      if (measure_time) dtime(&Stime);
    
      /* set Hs */

      for(i=0; i<na_rows*na_cols; i++){
	Hs11[i] = Complex(0.0,0.0);
	Hs22[i] = Complex(0.0,0.0);
	Hs12[i] = Complex(0.0,0.0);
      }

      Construct_Band_Ms(0,nh[0],  H1, Hs11,MP,k1,k2,k3);
      Construct_Band_Ms(0,nh[1],  H1, Hs22,MP,k1,k2,k3);
      Construct_Band_Ms(0,nh[2],  H1, Hs12,MP,k1,k2,k3);
      Construct_Band_Ms(1,nh[3],  H1, Hs12,MP,k1,k2,k3);
      Construct_Band_Ms(1,ImNL[0],H1, Hs11,MP,k1,k2,k3);
      Construct_Band_Ms(1,ImNL[1],H1, Hs22,MP,k1,k2,k3);
      Construct_Band_Ms(1,ImNL[2],H1, Hs12,MP,k1,k2,k3);

      if (measure_time){
	dtime(&Etime);
	time2 += Etime - Stime;
	dtime(&Stime);
      }

      Hamiltonian_Band_NC_Hs2( Hs11, Hs22, Hs12, Hs2, MPI_CommWD2[myworld2] );
    
      if (measure_time){
	dtime(&Etime);
	time13 += Etime - Stime;
      }

      if (kloop0<num_kloop0){

	if (measure_time) dtime(&Stime);

	/* S^t x Hs11 x S */

	for (i=0; i<na_rows*na_cols; i++) Cs[i] = Complex(0.0,0.0);

	Cblacs_barrier(ictxt1,"A");
	F77_NAME(pzgemm,PZGEMM)("N","N",&n,&n,&n,&alpha,Hs11,&ONE,&ONE,descH,Ss,&ONE,&ONE,descS,&beta,Cs,&ONE,&ONE,descC);

	for (i=0; i<na_rows*na_cols; i++) Hs11[i] = Complex(0.0,0.0);

	Cblacs_barrier(ictxt1,"C");
	F77_NAME(pzgemm,PZGEMM)("C","N",&n,&n,&n,&alpha,Ss,&ONE,&ONE,descS,Cs,&ONE,&ONE,descC,&beta,Hs11,&ONE,&ONE,descH);

	/* S^t x Hs12 x S */

	for (i=0; i<na_rows*na_cols; i++) Cs[i] = Complex(0.0,0.0);

	Cblacs_barrier(ictxt1,"A");
	F77_NAME(pzgemm,PZGEMM)("N","N",&n,&n,&n,&alpha,Hs12,&ONE,&ONE,descH,Ss,&ONE,&ONE,descS,&beta,Cs,&ONE,&ONE,descC);

	for (i=0; i<na_rows*na_cols; i++) Hs12[i] = Complex(0.0,0.0);

	Cblacs_barrier(ictxt1,"C");
	F77_NAME(pzgemm,PZGEMM)("C","N",&n,&n,&n,&alpha,Ss,&ONE,&ONE,descS,Cs,&ONE,&ONE,descC,&beta,Hs12,&ONE,&ONE,descH);

	/* S^t x Hs22 x S */

	for (i=0; i<na_rows*na_cols; i++) Cs[i] = Complex(0.0,0.0);

	Cblacs_barrier(ictxt1,"A");
	F77_NAME(pzgemm,PZGEMM)("N","N",&n,&n,&n,&alpha,Hs22,&ONE,&ONE,descH,Ss,&ONE,&ONE,descS,&beta,Cs,&ONE,&ONE,descC);

	for (i=0; i<na_rows*na_cols; i++) Hs22[i] = Complex(0.0,0.0);

	Cblacs_barrier(ictxt1,"C");
	F77_NAME(pzgemm,PZGEMM)("C","N",&n,&n,&n,&alpha,Ss,&ONE,&ONE,descS,Cs,&ONE,&ONE,descC,&beta,Hs22,&ONE,&ONE,descH);

	if (measure_time){
	  dtime(&Etime);
	  time3 += Etime - Stime;
	}

	/****************************************************
	 diagonalize the transformed H
	****************************************************/

	if (measure_time){
	  dtime(&Stime);
	}

	Hamiltonian_Band_NC_Hs2( Hs11, Hs22, Hs12, Hs2, MPI_CommWD2[myworld2] );

	MPI_Comm_split(MPI_CommWD2[myworld2],my_pcol2,my_prow2,&mpi_comm_rows);
	MPI_Comm_split(MPI_CommWD2[myworld2],my_prow2,my_pcol2,&mpi_comm_cols);

	mpi_comm_rows_int = MPI_Comm_c2f(mpi_comm_rows);
	mpi_comm_cols_int = MPI_Comm_c2f(mpi_comm_cols);

	if (scf_eigen_lib_flag==1 || numprocs2<5){
	  F77_NAME(solve_evp_complex,SOLVE_EVP_COMPLEX)
	    ( &n2, &MaxN, Hs2, &na_rows2, &ko[1], Cs2, &na_rows2, &nblk2, &mpi_comm_rows_int, &mpi_comm_cols_int );
	}
      
	else if (scf_eigen_lib_flag==2){

#ifndef kcomp
	  int mpiworld;
	  mpiworld = MPI_Comm_c2f(MPI_CommWD2[myworld2]);
	  F77_NAME(elpa_solve_evp_complex_2stage_double_impl,ELPA_SOLVE_EVP_COMPLEX_2STAGE_DOUBLE_IMPL)
	    ( &n2, &MaxN, Hs2, &na_rows2, &ko[1], Cs2, &na_rows2, &nblk2, &na_cols2, 
	      &mpi_comm_rows_int, &mpi_comm_cols_int, &mpiworld );
#else
	  F77_NAME(solve_evp_complex,SOLVE_EVP_COMPLEX)
	    ( &n2, &MaxN, Hs2, &na_rows2, &ko[1], Cs2, &na_rows2, &nblk2, &mpi_comm_rows_int, &mpi_comm_cols_int );
#endif

	}

	MPI_Comm_free(&mpi_comm_rows);
	MPI_Comm_free(&mpi_comm_cols);

	if (2<=level_stdout){
	  for (i1=1; i1<=MaxN; i1++){
	    printf("  Eigenvalues of Kohn-Sham %2d  %15.12f\n", i1,ko[i1]);
	  }
	}

	for (l=1; l<=MaxN; l++){
	  EIGEN[0][kloop][l] = ko[l];
	}

	if (3<=level_stdout && 0<=kloop){
	  printf(" myid0=%2d  kloop %i, k1 k2 k3 %10.6f %10.6f %10.6f\n",
		 myid0,kloop,T_KGrids1[kloop],T_KGrids2[kloop],T_KGrids3[kloop]);
	  for (i1=1; i1<=n2; i1++){
	    printf("  Eigenvalues of Kohn-Sham %2d  %15.12f\n", i1, ko[i1]);
	  }
	}

      } /* end of if (kloop0<num_kloop0) */

      if (measure_time){
	dtime(&Etime);
	time4 += Etime - Stime;
      }

      /**************************************************
                calculation of wave functions
      **************************************************/

      if (measure_time) dtime(&Stime);

      for(k=0; k<na_rows2*na_cols2; k++){
	Hs2[k].r = 0.0;
	Hs2[k].i = 0.0;
      }

      Cblacs_barrier(ictxt1_2,"A");
      F77_NAME(pzgemm,PZGEMM)( "T","T",&n2,&n2,&n2,&alpha,Cs2,&ONE,&ONE,descC2,Ss2,
			       &ONE,&ONE,descS2,&beta,Hs2,&ONE,&ONE,descH2);

      if (measure_time){
	dtime(&Etime);
	time5 += Etime - Stime;
      }

    } /* kloop0 */

    /****************************************************
     MPI:

     EIGEN
    ****************************************************/

    if (measure_time){
      MPI_Barrier(mpi_comm_level1);
      dtime(&Stime);
    }

    for (kloop=0; kloop<T_knum; kloop++){
      /* get ID in the zeroth world */
      ID = Comm_World_StartID1[0] + T_k_ID[myworld1][kloop];
      MPI_Bcast(&EIGEN[0][kloop][0], MaxN+1, MPI_DOUBLE, ID, mpi_comm_level1);
    } 

    if (measure_time){
      dtime(&Etime);
      time6 += Etime - Stime;
    }

    /********************************************** 
     calculation of the analytic Berry connection
    **********************************************/ 

    kloop = S_knum;
    k1 = T_KGrids1[kloop];
    k2 = T_KGrids2[kloop];
    k3 = T_KGrids3[kloop];

    Analytic_Berry_Connection( HOMO, 1,
			       kloop,n2,MaxN,MP,T_k_op,k1,k2,k3,
			       H1,Ss2,Cs2,Hs2,Vs2,Qs2,Hs11,Hs22,Hs12,EIGEN,
                               nh,ImNL,CntOLP,  
			       ABC,VDMs_ABC,order_GA,all_knum,
			       myworld1,myworld2,MPI_CommWD1,MPI_CommWD2 );

    dtime(&EiloopTime);
    if (myid0==Host_ID && 0<level_stdout){
      printf("<Electric_Polarization_BandNonCol> time1=%lf\n", EiloopTime-SiloopTime);fflush(stdout);
    }

  } /* end of if (all_knum==1) */

  /****************************************************
   ****************************************************
   if (all_knum!=1) start kloop
   ****************************************************
  ****************************************************/

  if (all_knum!=1){

    dtime(&SiloopTime);

    /* for kloop */

    for (kloop0=0; kloop0<max_num_kloop0; kloop0++){

      /* get k1, k2, and k3 */

      if (kloop0<num_kloop0){

	kloop = S_knum + kloop0;
	k1 = T_KGrids1[kloop];
	k2 = T_KGrids2[kloop];
	k3 = T_KGrids3[kloop];
      }

      if (measure_time) dtime(&Stime);

      /* make Cs */

      for(i=0; i<na_rows*na_cols; i++){ Cs[i] = Complex(0.0,0.0);}
      Construct_Band_Ms(0,CntOLP,H1,Cs,MP,k1,k2,k3);

      /* diagonalize Cs */

      if (kloop0<num_kloop0){

	MPI_Comm_split(MPI_CommWD2[myworld2],my_pcol,my_prow,&mpi_comm_rows);
	MPI_Comm_split(MPI_CommWD2[myworld2],my_prow,my_pcol,&mpi_comm_cols);

	mpi_comm_rows_int = MPI_Comm_c2f(mpi_comm_rows);
	mpi_comm_cols_int = MPI_Comm_c2f(mpi_comm_cols);

        if (scf_eigen_lib_flag==1 || numprocs2<5){
  	  F77_NAME(solve_evp_complex,SOLVE_EVP_COMPLEX)
          ( &n, &n, Cs, &na_rows, &ko[1], Ss, &na_rows, &nblk, &mpi_comm_rows_int, &mpi_comm_cols_int );
	}

        else if (scf_eigen_lib_flag==2){

#ifndef kcomp
          int mpiworld;
          mpiworld = MPI_Comm_c2f(MPI_CommWD2[myworld2]);
          F77_NAME(elpa_solve_evp_complex_2stage_double_impl,ELPA_SOLVE_EVP_COMPLEX_2STAGE_DOUBLE_IMPL)
          ( &n, &n, Cs, &na_rows, &ko[1], Ss, &na_rows, &nblk, &na_cols, 
            &mpi_comm_rows_int, &mpi_comm_cols_int, &mpiworld );

#else
  	  F77_NAME(solve_evp_complex,SOLVE_EVP_COMPLEX)
          ( &n, &n, Cs, &na_rows, &ko[1], Ss, &na_rows, &nblk, &mpi_comm_rows_int, &mpi_comm_cols_int );
#endif
	}

	MPI_Comm_free(&mpi_comm_rows);
	MPI_Comm_free(&mpi_comm_cols);

	/* print to the standard output */

	if (3<=level_stdout){
	  printf(" myid0=%2d kloop %2d  k1 k2 k3 %10.6f %10.6f %10.6f\n",
		 myid0,kloop,T_KGrids1[kloop],T_KGrids2[kloop],T_KGrids3[kloop]);
	  for (i=1; i<=n; i++){
	    printf("  Eigenvalues of OLP  %2d  %15.12f\n",i,ko[i]);
	  }
	}

	/* minus eigenvalues to 1.0e-10 */

	for (l=1; l<=n; l++){
	  if (ko[l]<1.0e-10) ko[l] = 1.0e-10;
	  ko[l] = 1.0/sqrt(ko[l]);
	}

	/* calculate S*1/sqrt(ko) */

	for(i=0; i<na_rows; i++){
	  for(j=0; j<na_cols; j++){
	    jg = np_cols*nblk*((j)/nblk) + (j)%nblk + ((np_cols+my_pcol)%np_cols)*nblk + 1;
	    Ss[j*na_rows+i].r = Ss[j*na_rows+i].r*ko[jg];
	    Ss[j*na_rows+i].i = Ss[j*na_rows+i].i*ko[jg];
	  }
	}

	/* make Ss2 */

	Overlap_Band_NC_Ss2( Ss, Ss2, MPI_CommWD2[myworld2] );

      } /* end of if (kloop0<num_kloop0) */

      if (measure_time){
        dtime(&Etime);
        time9 += Etime - Stime;
      }

      /******************************************************************************
               transformation of H with Ss

        in case of SO_switch==0 && Hub_U_switch==0 && Constraint_NCS_switch==0 
                   && Zeeman_NCS_switch==0 && Zeeman_NCO_switch==0
 
        H[i    ][j    ].r = RH[0];
        H[i    ][j    ].i = 0.0;
        H[i+NUM][j+NUM].r = RH[1];
        H[i+NUM][j+NUM].i = 0.0;
        H[i    ][j+NUM].r = RH[2];
        H[i    ][j+NUM].i = RH[3];

        in case of SO_switch==1 or Hub_U_switch==1 or 1<=Constraint_NCS_switch 
                   or Zeeman_NCS_switch==1 or Zeeman_NCO_switch==1 

        H[i    ][j    ].r = RH[0];  
        H[i    ][j    ].i = IH[0];
        H[i+NUM][j+NUM].r = RH[1];
        H[i+NUM][j+NUM].i = IH[1];
        H[i    ][j+NUM].r = RH[2];
        H[i    ][j+NUM].i = RH[3] + IH[2];
      ******************************************************************************/
      
      if (measure_time) dtime(&Stime);
      
      /* set Hs */
      
      for(i=0; i<na_rows*na_cols; i++){
	Hs11[i] = Complex(0.0,0.0);
	Hs22[i] = Complex(0.0,0.0);
	Hs12[i] = Complex(0.0,0.0);
      }

      Construct_Band_Ms(0,nh[0],  H1, Hs11,MP,k1,k2,k3);
      Construct_Band_Ms(0,nh[1],  H1, Hs22,MP,k1,k2,k3);
      Construct_Band_Ms(0,nh[2],  H1, Hs12,MP,k1,k2,k3);
      Construct_Band_Ms(1,nh[3],  H1, Hs12,MP,k1,k2,k3);
      Construct_Band_Ms(1,ImNL[0],H1, Hs11,MP,k1,k2,k3);
      Construct_Band_Ms(1,ImNL[1],H1, Hs22,MP,k1,k2,k3);
      Construct_Band_Ms(1,ImNL[2],H1, Hs12,MP,k1,k2,k3);

      if (kloop0<num_kloop0){

	/* set Hs2 */

	Hamiltonian_Band_NC_Hs2( Hs11, Hs22, Hs12, Hs2, MPI_CommWD2[myworld2] );

	/* S^t x Hs11 x S */

	for (i=0; i<na_rows*na_cols; i++) Cs[i] = Complex(0.0,0.0);

	Cblacs_barrier(ictxt1,"A");
	F77_NAME(pzgemm,PZGEMM)("N","N",&n,&n,&n,&alpha,Hs11,&ONE,&ONE,descH,Ss,&ONE,&ONE,descS,&beta,Cs,&ONE,&ONE,descC);

	for (i=0; i<na_rows*na_cols; i++) Hs11[i] = Complex(0.0,0.0);

	Cblacs_barrier(ictxt1,"C");
	F77_NAME(pzgemm,PZGEMM)("C","N",&n,&n,&n,&alpha,Ss,&ONE,&ONE,descS,Cs,&ONE,&ONE,descC,&beta,Hs11,&ONE,&ONE,descH);

	/* S^t x Hs12 x S */

	for (i=0; i<na_rows*na_cols; i++) Cs[i] = Complex(0.0,0.0);

	Cblacs_barrier(ictxt1,"A");
	F77_NAME(pzgemm,PZGEMM)("N","N",&n,&n,&n,&alpha,Hs12,&ONE,&ONE,descH,Ss,&ONE,&ONE,descS,&beta,Cs,&ONE,&ONE,descC);

	for (i=0; i<na_rows*na_cols; i++) Hs12[i] = Complex(0.0,0.0);

	Cblacs_barrier(ictxt1,"C");
	F77_NAME(pzgemm,PZGEMM)("C","N",&n,&n,&n,&alpha,Ss,&ONE,&ONE,descS,Cs,&ONE,&ONE,descC,&beta,Hs12,&ONE,&ONE,descH);

	/* S^t x Hs22 x S */

	for (i=0; i<na_rows*na_cols; i++) Cs[i] = Complex(0.0,0.0);

	Cblacs_barrier(ictxt1,"A");
	F77_NAME(pzgemm,PZGEMM)("N","N",&n,&n,&n,&alpha,Hs22,&ONE,&ONE,descH,Ss,&ONE,&ONE,descS,&beta,Cs,&ONE,&ONE,descC);

	for (i=0; i<na_rows*na_cols; i++) Hs22[i] = Complex(0.0,0.0);

	Cblacs_barrier(ictxt1,"C");
	F77_NAME(pzgemm,PZGEMM)("C","N",&n,&n,&n,&alpha,Ss,&ONE,&ONE,descS,Cs,&ONE,&ONE,descC,&beta,Hs22,&ONE,&ONE,descH);

        if (measure_time){
  	  dtime(&Etime);
	  time10 += Etime - Stime;
	}

	/****************************************************
	   diagonalize the transformed H
	 ****************************************************/

	if (measure_time) dtime(&Stime);

	Hamiltonian_Band_NC_Hs2( Hs11, Hs22, Hs12, Hs2, MPI_CommWD2[myworld2] );

	MPI_Comm_split(MPI_CommWD2[myworld2],my_pcol2,my_prow2,&mpi_comm_rows);
	MPI_Comm_split(MPI_CommWD2[myworld2],my_prow2,my_pcol2,&mpi_comm_cols);

	mpi_comm_rows_int = MPI_Comm_c2f(mpi_comm_rows);
	mpi_comm_cols_int = MPI_Comm_c2f(mpi_comm_cols);
  
        if (scf_eigen_lib_flag==1 || numprocs2<5){
  	  F77_NAME(solve_evp_complex,SOLVE_EVP_COMPLEX)
          ( &n2, &MaxN, Hs2, &na_rows2, &ko[1], Cs2, &na_rows2, &nblk2, &mpi_comm_rows_int, &mpi_comm_cols_int );
	}
	
        else if (scf_eigen_lib_flag==2){

#ifndef kcomp
          int mpiworld;
          mpiworld = MPI_Comm_c2f(MPI_CommWD2[myworld2]);
          F77_NAME(elpa_solve_evp_complex_2stage_double_impl,ELPA_SOLVE_EVP_COMPLEX_2STAGE_DOUBLE_IMPL)
	  ( &n2, &MaxN, Hs2, &na_rows2, &ko[1], Cs2, &na_rows2, &nblk2, &na_cols2, 
            &mpi_comm_rows_int, &mpi_comm_cols_int, &mpiworld );

#else
  	  F77_NAME(solve_evp_complex,SOLVE_EVP_COMPLEX)
          ( &n2, &MaxN, Hs2, &na_rows2, &ko[1], Cs2, &na_rows2, &nblk2, &mpi_comm_rows_int, &mpi_comm_cols_int );
#endif
	}

	MPI_Comm_free(&mpi_comm_rows);
	MPI_Comm_free(&mpi_comm_cols);

	if (2<=level_stdout){
	  for (i1=1; i1<=MaxN; i1++){
	    printf("  Eigenvalues of Kohn-Sham %2d  %15.12f\n", i1,ko[i1]);
	  }
	}

	for (l=1; l<=MaxN; l++){
	  EIGEN[0][kloop][l] = ko[l];
	}

	if (3<=level_stdout && 0<=kloop && kloop0<num_kloop0){
	  printf(" myid0=%2d  kloop %i, k1 k2 k3 %10.6f %10.6f %10.6f\n",
		 myid0,kloop,T_KGrids1[kloop],T_KGrids2[kloop],T_KGrids3[kloop]);
	  for (i1=1; i1<=n2; i1++){
	    printf("  Eigenvalues of Kohn-Sham %2d  %15.12f\n", i1, ko[i1]);
	  }
	}

        if (measure_time){
          dtime(&Etime);
          time11 += Etime - Stime;
	}

        /**************************************************
                  calculation of wave functions
        **************************************************/

	for(k=0; k<na_rows2*na_cols2; k++){
	  Hs2[k].r = 0.0;
	  Hs2[k].i = 0.0;
	}

	Cblacs_barrier(ictxt1_2,"A");
	F77_NAME(pzgemm,PZGEMM)( "T","T",&n2,&n2,&n2,&alpha,Cs2,&ONE,&ONE,descC2,Ss2,
				 &ONE,&ONE,descS2,&beta,Hs2,&ONE,&ONE,descH2);

      } /* end of if (kloop0<num_kloop0) */

      /***************************************** 
           calculate the Berry connection
      *****************************************/ 

      Analytic_Berry_Connection( HOMO, (kloop0<num_kloop0),
				 kloop,n2,MaxN,MP,T_k_op,k1,k2,k3,
				 H1,Ss2,Cs2,Hs2,Vs2,Qs2,Hs11,Hs22,Hs12,EIGEN,
				 nh,ImNL,CntOLP,  
				 ABC,VDMs_ABC,order_GA,all_knum,
				 myworld1,myworld2,MPI_CommWD1,MPI_CommWD2 );

    } /* kloop0 */

    dtime(&EiloopTime);
    if (myid0==Host_ID && 0<level_stdout){
      printf("<Electric_Polarization_BandNonCol> time2=%lf\n", EiloopTime-SiloopTime);fflush(stdout);
    }

  } /* if (all_knum!=1) */

  /****************************************************
          electric polarization by Berry phase
  ****************************************************/

  tmparray[0] = ABC[0].i; tmparray[1] = ABC[1].i; tmparray[2] = ABC[2].i;  
  MPI_Allreduce(MPI_IN_PLACE, &tmparray[0], 3, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
  for (i=0; i<3; i++){ ABC[i].i = tmparray[i]/sum_weights; }

  /* taking account of the modulo 2pi */

  for (i=1; i<=3; i++){

    tmp = (ABC[0].i*rtv[i][1] + ABC[1].i*rtv[i][2] + ABC[2].i*rtv[i][3])/(2.0*PI);

    do{
      tmp += 1.0;
    } while (tmp<0.0);

    /* tmp is adjusted in between 0.0 and 1.0. */
    j = (int)tmp;
    be[i] = tmp - (double)j;
  }

  ABC[0].i = be[1]*tv[1][1] + be[2]*tv[2][1] + be[3]*tv[3][1];
  ABC[1].i = be[1]*tv[1][2] + be[2]*tv[2][2] + be[3]*tv[3][2];
  ABC[2].i = be[1]*tv[1][3] + be[2]*tv[2][3] + be[3]*tv[3][3];

  /****************************************************
          calculation of the other contribution
  ****************************************************/

  Allocate_Free_Electric_Polarization(1);
  time11 += EP1(CntOLP,OLPpo,CDM, EPol, CPol, EPol_BG);
  Allocate_Free_Electric_Polarization(2);

  /****************************************************
                show and save the results
  ****************************************************/

  double AbsD,E_dpx,E_dpy,E_dpz,C_dpx,C_dpy,C_dpz,E_dpx_BG,E_dpy_BG,E_dpz_BG;

  E_dpx = ABC[0].i - EPol[0];
  E_dpy = ABC[1].i - EPol[1];
  E_dpz = ABC[2].i - EPol[2];

  C_dpx = CPol[0];
  C_dpy = CPol[1];
  C_dpz = CPol[2];

  E_dpx_BG =-EPol_BG[0];
  E_dpy_BG =-EPol_BG[1];
  E_dpz_BG =-EPol_BG[2];

  dipole_moment[0][1] = AU2Debye*(C_dpx + E_dpx + E_dpx_BG);
  dipole_moment[0][2] = AU2Debye*(C_dpy + E_dpy + E_dpy_BG);
  dipole_moment[0][3] = AU2Debye*(C_dpz + E_dpz + E_dpz_BG);

  dipole_moment[1][1] = AU2Debye*C_dpx;
  dipole_moment[1][2] = AU2Debye*C_dpy;
  dipole_moment[1][3] = AU2Debye*C_dpz;

  dipole_moment[2][1] = AU2Debye*E_dpx;
  dipole_moment[2][2] = AU2Debye*E_dpy;
  dipole_moment[2][3] = AU2Debye*E_dpz;

  dipole_moment[3][1] = AU2Debye*E_dpx_BG;
  dipole_moment[3][2] = AU2Debye*E_dpy_BG;
  dipole_moment[3][3] = AU2Debye*E_dpz_BG;

  AbsD = sqrt( dipole_moment[0][1]*dipole_moment[0][1]
             + dipole_moment[0][2]*dipole_moment[0][2]
             + dipole_moment[0][3]*dipole_moment[0][3] );

  if (myid0==Host_ID){

    /* stdout */

    if (0<level_stdout){

      printf("\n*******************************************************\n"); fflush(stdout);
      printf("         Dipole moment (Debye) by the Berry phase        \n");  fflush(stdout);
      printf("*******************************************************\n\n"); fflush(stdout);

      printf(" Absolute D %17.8f\n\n",AbsD);
      printf("                      Dx                Dy                Dz\n"); fflush(stdout);
      printf(" Total       %17.8f %17.8f %17.8f\n",
	     dipole_moment[0][1],dipole_moment[0][2],dipole_moment[0][3]);fflush(stdout);
      printf(" Core        %17.8f %17.8f %17.8f\n",
	     dipole_moment[1][1],dipole_moment[1][2],dipole_moment[1][3]);fflush(stdout);
      printf(" Electron    %17.8f %17.8f %17.8f\n",
	     dipole_moment[2][1],dipole_moment[2][2],dipole_moment[2][3]);fflush(stdout);
      printf(" Back ground %17.8f %17.8f %17.8f\n\n",
	     dipole_moment[3][1],dipole_moment[3][2],dipole_moment[3][3]);fflush(stdout);
    }

    /* save the results to a file */

    FILE *fp_DPM;
    char file_DPM[YOUSO10] = ".dpm";

    fnjoint(filepath,filename,file_DPM);

    if ((fp_DPM = fopen(file_DPM,"w")) != NULL){

#ifdef xt3
      setvbuf(fp_DPM,buf,_IOFBF,fp_bsize);  /* setvbuf */
#endif

      fprintf(fp_DPM,"\n*******************************************************\n"); fflush(stdout);
      fprintf(fp_DPM,"         Dipole moment (Debye) by the Berry phase        \n");  fflush(stdout);
      fprintf(fp_DPM,"*******************************************************\n\n"); fflush(stdout);

      fprintf(fp_DPM," Absolute D %17.8f\n\n",AbsD);
      fprintf(fp_DPM,"                      Dx                Dy                Dz\n"); fflush(stdout);
      fprintf(fp_DPM," Total       %17.8f %17.8f %17.8f\n",
	      dipole_moment[0][1],dipole_moment[0][2],dipole_moment[0][3]);fflush(stdout);
      fprintf(fp_DPM," Core        %17.8f %17.8f %17.8f\n",
	      dipole_moment[1][1],dipole_moment[1][2],dipole_moment[1][3]);fflush(stdout);
      fprintf(fp_DPM," Electron    %17.8f %17.8f %17.8f\n",
	      dipole_moment[2][1],dipole_moment[2][2],dipole_moment[2][3]);fflush(stdout);
      fprintf(fp_DPM," Back ground %17.8f %17.8f %17.8f\n\n",
 	      dipole_moment[3][1],dipole_moment[3][2],dipole_moment[3][3]);fflush(stdout);

      fclose(fp_DPM);
    }
    else{
      printf("Failure of saving the DPM file.\n");fflush(stdout);
    }
  }    

  /****************************************************
                       free arrays
  ****************************************************/

  free(My_NZeros);
  free(SP_NZeros);
  free(SP_Atoms);

  free(DMs11);
  free(DMs22);
  free(DMs12);
  free(VDMs_ABC);
  free(ABC);
  free(Vs2);
  free(Qs2);

  /* for PrintMemory and allocation */
  firsttime=0;

  /* for elapsed time */

  if (measure_time){
    printf("myid0=%2d time1 =%9.4f\n",myid0,time1);fflush(stdout);
    printf("myid0=%2d time2 =%9.4f\n",myid0,time2);fflush(stdout);
    printf("myid0=%2d time3 =%9.4f\n",myid0,time3);fflush(stdout);
    printf("myid0=%2d time4 =%9.4f\n",myid0,time4);fflush(stdout);
    printf("myid0=%2d time5 =%9.4f\n",myid0,time5);fflush(stdout);
    printf("myid0=%2d time6 =%9.4f\n",myid0,time6);fflush(stdout);
    printf("myid0=%2d time7 =%9.4f\n",myid0,time7);fflush(stdout);
    printf("myid0=%2d time8 =%9.4f\n",myid0,time8);fflush(stdout);
    printf("myid0=%2d time9 =%9.4f\n",myid0,time9);fflush(stdout);
    printf("myid0=%2d time10=%9.4f\n",myid0,time10);fflush(stdout);
    printf("myid0=%2d time11=%9.4f\n",myid0,time11);fflush(stdout);
    printf("myid0=%2d time12=%9.4f\n",myid0,time12);fflush(stdout);
    printf("myid0=%2d time13=%9.4f\n",myid0,time12);fflush(stdout);
  }

  MPI_Barrier(mpi_comm_level1);
  dtime(&TEtime);
  time0 = TEtime - TStime;

  return time0;
}




void Construct_Band_Ms( int cpx_flag, double ****Mat, double *M1, dcomplex *Ms, 
                        int *MP, double k1, double k2, double k3)
{
  static int firsttime=1;
  int i,j,k;
  int MA_AN,GA_AN,LB_AN,GB_AN,AN,Rn,l1,l2,l3;
  int wanA,wanB,tnoA,tnoB,Anum,Bnum,NUM;
  int num,tnum,num_orbitals;
  int ID,myid,numprocs,tag=999;
  int *My_NZeros;
  int *is1,*ie1,*is2;
  int *My_Matomnum,*order_GA;
  double sum,kRn,si,co;
  double Stime,Etime;
  double AStime,AEtime;
  MPI_Status stat;
  MPI_Request request;
  int ig,jg,il,jl,prow,pcol,brow,bcol;

  if (measure_time){
    dtime(&AStime);
    dtime(&Stime);
  }

  /* MPI */

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);
  MPI_Barrier(mpi_comm_level1);

  /* allocation of arrays */

  My_NZeros = (int*)malloc(sizeof(int)*numprocs);
  My_Matomnum = (int*)malloc(sizeof(int)*numprocs);
  is1 = (int*)malloc(sizeof(int)*numprocs);
  ie1 = (int*)malloc(sizeof(int)*numprocs);
  is2 = (int*)malloc(sizeof(int)*numprocs);
  order_GA = (int*)malloc(sizeof(int)*(atomnum+2));

  if (firsttime && memoryusage_fileout) {
  PrintMemory("Electric_Polarization_BandNonCol: My_NZeros", sizeof(int)*numprocs,NULL);
  PrintMemory("Electric_Polarization_BandNonCol: SP_NZeros", sizeof(int)*numprocs,NULL);
  PrintMemory("Electric_Polarization_BandNonCol: SP_Atoms", sizeof(int)*numprocs,NULL);
  PrintMemory("Electric_Polarization_BandNonCol: is1", sizeof(int)*numprocs,NULL);
  PrintMemory("Electric_Polarization_BandNonCol: ie1", sizeof(int)*numprocs,NULL);
  PrintMemory("Electric_Polarization_BandNonCol: is2", sizeof(int)*numprocs,NULL);
  PrintMemory("Electric_Polarization_BandNonCol: order_GA", sizeof(int)*(atomnum+2),NULL);
  }
  firsttime = 1;

  /* find my total number of non-zero elements in myid */

  My_NZeros[myid] = 0;
  for (MA_AN=1; MA_AN<=Matomnum; MA_AN++){
    GA_AN = M2G[MA_AN];
    wanA = WhatSpecies[GA_AN];
    tnoA = Spe_Total_CNO[wanA];

    num = 0;      
    for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
      GB_AN = natn[GA_AN][LB_AN];
      wanB = WhatSpecies[GB_AN];
      tnoB = Spe_Total_CNO[wanB];
      num += tnoB;
    }

    My_NZeros[myid] += tnoA*num;
  }

  for (ID=0; ID<numprocs; ID++){
    MPI_Bcast(&My_NZeros[ID],1,MPI_INT,ID,mpi_comm_level1);
  }

  tnum = 0;
  for (ID=0; ID<numprocs; ID++){
    tnum += My_NZeros[ID];
  }  

  is1[0] = 0;
  ie1[0] = My_NZeros[0] - 1;

  for (ID=1; ID<numprocs; ID++){
    is1[ID] = ie1[ID-1] + 1;
    ie1[ID] = is1[ID] + My_NZeros[ID] - 1;
  }  

  /* set is2 and order_GA */

  My_Matomnum[myid] = Matomnum;
  for (ID=0; ID<numprocs; ID++){
    MPI_Bcast(&My_Matomnum[ID],1,MPI_INT,ID,mpi_comm_level1);
  }

  is2[0] = 1;
  for (ID=1; ID<numprocs; ID++){
    is2[ID] = is2[ID-1] + My_Matomnum[ID-1];
  }
  
  for (MA_AN=1; MA_AN<=Matomnum; MA_AN++){
    order_GA[is2[myid]+MA_AN-1] = M2G[MA_AN];
  }

  for (ID=0; ID<numprocs; ID++){
    MPI_Bcast(&order_GA[is2[ID]],My_Matomnum[ID],MPI_INT,ID,mpi_comm_level1);
  }

  /* set MP */

  Anum = 1;
  for (i=1; i<=atomnum; i++){
    MP[i] = Anum;
    wanA = WhatSpecies[i];
    Anum += Spe_Total_CNO[wanA];
  }
  NUM = Anum - 1;

  /* set M1 */

  for (i=0; i<tnum; i++) M1[i] = 0.0;

  k = is1[myid];
  for (MA_AN=1; MA_AN<=Matomnum; MA_AN++){
    GA_AN = M2G[MA_AN];
    wanA = WhatSpecies[GA_AN];
    tnoA = Spe_Total_CNO[wanA];
    for (i=0; i<tnoA; i++){
      for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
        GB_AN = natn[GA_AN][LB_AN];
        wanB = WhatSpecies[GB_AN];
        tnoB = Spe_Total_CNO[wanB];
        for (j=0; j<tnoB; j++){
          M1[k] = Mat[MA_AN][LB_AN][i][j]; 
          k++;
	}
      }
    }
  }

  if (measure_time){
    dtime(&Etime);
    printf("timeB1 myid=%2d %15.12f\n",myid,Etime-Stime);
  }

  if (measure_time){
    dtime(&Stime);
  }

  /* MPI M1 */

  MPI_Allreduce(MPI_IN_PLACE, &M1[0], tnum, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

  if (measure_time){
    dtime(&Etime);
    printf("timeB2 myid=%2d %15.12f\n",myid,Etime-Stime);
  }

  if (measure_time){
    dtime(&Stime);
  }

  /* M1 -> Ms */
  
  k = 0;
  for (AN=1; AN<=atomnum; AN++){
    GA_AN = order_GA[AN];
    wanA = WhatSpecies[GA_AN];
    tnoA = Spe_Total_CNO[wanA];
    Anum = MP[GA_AN];

    for (i=0; i<tnoA; i++){

      for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
	GB_AN = natn[GA_AN][LB_AN];
        Rn = ncn[GA_AN][LB_AN];
	wanB = WhatSpecies[GB_AN];
	tnoB = Spe_Total_CNO[wanB];
	Bnum = MP[GB_AN];

	l1 = atv_ijk[Rn][1];
	l2 = atv_ijk[Rn][2];
	l3 = atv_ijk[Rn][3];
	kRn = k1*(double)l1 + k2*(double)l2 + k3*(double)l3;

	si = sin(2.0*PI*kRn);
	co = cos(2.0*PI*kRn);

	for (j=0; j<tnoB; j++){
	  ig = Anum+i;
	  jg = Bnum+j;
	    
	  brow = (ig-1)/nblk;
	  bcol = (jg-1)/nblk;

	  prow = brow%np_rows;
	  pcol = bcol%np_cols;

	  if(my_prow==prow && my_pcol==pcol){

	    il = (brow/np_rows+1)*nblk+1;
	    jl = (bcol/np_cols+1)*nblk+1;

	    if(((my_prow+np_rows)%np_rows) >= (brow%np_rows)){
	      if(my_prow==prow){
		il = il+(ig-1)%nblk;
	      }
	      il = il-nblk;
	    }

	    if(((my_pcol+np_cols)%np_cols) >= (bcol%np_cols)){
	      if(my_pcol==pcol){
		jl = jl+(jg-1)%nblk;
	      }
	      jl = jl-nblk;
	    }

            if (cpx_flag==0){
	      Ms[(jl-1)*na_rows+il-1].r += M1[k]*co;
	      Ms[(jl-1)*na_rows+il-1].i += M1[k]*si;
	    }
            else if (cpx_flag==1){
	      Ms[(jl-1)*na_rows+il-1].r -= M1[k]*si;
	      Ms[(jl-1)*na_rows+il-1].i += M1[k]*co;
	    }
	  }
	    
	  k++;
	}
      }
    }
  }

  if (measure_time){
    dtime(&Etime);
    printf("timeB3 myid=%2d %15.12f\n",myid,Etime-Stime);
  }

  /* freeing of arrays */

  free(My_NZeros);
  free(My_Matomnum);
  free(is1);
  free(ie1);
  free(is2);
  free(order_GA);

  if (measure_time){
    dtime(&AEtime);
    printf("timeB_all myid=%2d %15.12f\n",myid,AEtime-AStime);
  }
}




void Construct_Band_DMs( int im_flag, double ****Mat, double *M1, dcomplex *DMs, int *MP )
{
  static int firsttime=1;
  int i,j,k;
  int MA_AN,GA_AN,LB_AN,GB_AN,AN,Rn,l1,l2,l3;
  int wanA,wanB,tnoA,tnoB,Anum,Bnum,NUM;
  int num,tnum,num_orbitals;
  int ID,myid,numprocs,tag=999;
  int *My_NZeros;
  int *is1,*ie1,*is2;
  int *My_Matomnum,*order_GA;
  double sum,kRn,si,co;
  double Stime,Etime;
  double AStime,AEtime;
  MPI_Status stat;
  MPI_Request request;
  int ig,jg,il,jl,prow,pcol,brow,bcol;

  if (measure_time){
    dtime(&AStime);
    dtime(&Stime);
  }

  /* MPI */

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);
  MPI_Barrier(mpi_comm_level1);

  /* allocation of arrays */

  My_NZeros = (int*)malloc(sizeof(int)*numprocs);
  My_Matomnum = (int*)malloc(sizeof(int)*numprocs);
  is1 = (int*)malloc(sizeof(int)*numprocs);
  ie1 = (int*)malloc(sizeof(int)*numprocs);
  is2 = (int*)malloc(sizeof(int)*numprocs);
  order_GA = (int*)malloc(sizeof(int)*(atomnum+2));

  if (firsttime && memoryusage_fileout) {
  PrintMemory("Electric_Polarization_BandNonCol: My_NZeros", sizeof(int)*numprocs,NULL);
  PrintMemory("Electric_Polarization_BandNonCol: SP_NZeros", sizeof(int)*numprocs,NULL);
  PrintMemory("Electric_Polarization_BandNonCol: SP_Atoms", sizeof(int)*numprocs,NULL);
  PrintMemory("Electric_Polarization_BandNonCol: is1", sizeof(int)*numprocs,NULL);
  PrintMemory("Electric_Polarization_BandNonCol: ie1", sizeof(int)*numprocs,NULL);
  PrintMemory("Electric_Polarization_BandNonCol: is2", sizeof(int)*numprocs,NULL);
  PrintMemory("Electric_Polarization_BandNonCol: order_GA", sizeof(int)*(atomnum+2),NULL);
  }
  firsttime = 1;

  /* find my total number of non-zero elements in myid */

  My_NZeros[myid] = 0;
  for (MA_AN=1; MA_AN<=Matomnum; MA_AN++){
    GA_AN = M2G[MA_AN];
    wanA = WhatSpecies[GA_AN];
    tnoA = Spe_Total_CNO[wanA];

    num = 0;      
    for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
      GB_AN = natn[GA_AN][LB_AN];
      wanB = WhatSpecies[GB_AN];
      tnoB = Spe_Total_CNO[wanB];
      num += tnoB;
    }

    My_NZeros[myid] += tnoA*num;
  }

  for (ID=0; ID<numprocs; ID++){
    MPI_Bcast(&My_NZeros[ID],1,MPI_INT,ID,mpi_comm_level1);
  }

  tnum = 0;
  for (ID=0; ID<numprocs; ID++){
    tnum += My_NZeros[ID];
  }  

  is1[0] = 0;
  ie1[0] = My_NZeros[0] - 1;

  for (ID=1; ID<numprocs; ID++){
    is1[ID] = ie1[ID-1] + 1;
    ie1[ID] = is1[ID] + My_NZeros[ID] - 1;
  }  

  /* set is2 and order_GA */

  My_Matomnum[myid] = Matomnum;
  for (ID=0; ID<numprocs; ID++){
    MPI_Bcast(&My_Matomnum[ID],1,MPI_INT,ID,mpi_comm_level1);
  }

  is2[0] = 1;
  for (ID=1; ID<numprocs; ID++){
    is2[ID] = is2[ID-1] + My_Matomnum[ID-1];
  }
  
  for (MA_AN=1; MA_AN<=Matomnum; MA_AN++){
    order_GA[is2[myid]+MA_AN-1] = M2G[MA_AN];
  }

  for (ID=0; ID<numprocs; ID++){
    MPI_Bcast(&order_GA[is2[ID]],My_Matomnum[ID],MPI_INT,ID,mpi_comm_level1);
  }

  /* set MP */

  Anum = 1;
  for (i=1; i<=atomnum; i++){
    MP[i] = Anum;
    wanA = WhatSpecies[i];
    Anum += Spe_Total_CNO[wanA];
  }
  NUM = Anum - 1;

  /* set M1 */

  for (i=0; i<tnum; i++) M1[i] = 0.0;

  k = is1[myid];
  for (MA_AN=1; MA_AN<=Matomnum; MA_AN++){
    GA_AN = M2G[MA_AN];
    wanA = WhatSpecies[GA_AN];
    tnoA = Spe_Total_CNO[wanA];
    for (i=0; i<tnoA; i++){
      for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
        GB_AN = natn[GA_AN][LB_AN];
        wanB = WhatSpecies[GB_AN];
        tnoB = Spe_Total_CNO[wanB];
        for (j=0; j<tnoB; j++){
          M1[k] = Mat[MA_AN][LB_AN][i][j]; 
          k++;
	}
      }
    }
  }

  if (measure_time){
    dtime(&Etime);
    printf("timeB1 myid=%2d %15.12f\n",myid,Etime-Stime);
  }

  if (measure_time){
    dtime(&Stime);
  }

  /* MPI M1 */

  MPI_Allreduce(MPI_IN_PLACE, &M1[0], tnum, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

  if (measure_time){
    dtime(&Etime);
    printf("timeB2 myid=%2d %15.12f\n",myid,Etime-Stime);
  }

  if (measure_time){
    dtime(&Stime);
  }

  /* M1 -> DMs */
  
  k = 0;
  for (AN=1; AN<=atomnum; AN++){
    GA_AN = order_GA[AN];
    wanA = WhatSpecies[GA_AN];
    tnoA = Spe_Total_CNO[wanA];
    Anum = MP[GA_AN];

    for (i=0; i<tnoA; i++){

      for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
	GB_AN = natn[GA_AN][LB_AN];
        Rn = ncn[GA_AN][LB_AN];
	wanB = WhatSpecies[GB_AN];
	tnoB = Spe_Total_CNO[wanB];
	Bnum = MP[GB_AN];

	for (j=0; j<tnoB; j++){
	  ig = Anum+i;
	  jg = Bnum+j;
	    
	  brow = (ig-1)/nblk;
	  bcol = (jg-1)/nblk;

	  prow = brow%np_rows;
	  pcol = bcol%np_cols;

	  if(Rn==0 && my_prow==prow && my_pcol==pcol){

	    il = (brow/np_rows+1)*nblk+1;
	    jl = (bcol/np_cols+1)*nblk+1;

	    if(((my_prow+np_rows)%np_rows) >= (brow%np_rows)){
	      if(my_prow==prow){
		il = il+(ig-1)%nblk;
	      }
	      il = il-nblk;
	    }

	    if(((my_pcol+np_cols)%np_cols) >= (bcol%np_cols)){
	      if(my_pcol==pcol){
		jl = jl+(jg-1)%nblk;
	      }
	      jl = jl-nblk;
	    }

            if (im_flag==0){
 	      DMs[(jl-1)*na_rows+il-1].r -= M1[k];
	    }
	    else if (im_flag==1){
	      DMs[(jl-1)*na_rows+il-1].i -= M1[k];
	    }

	  }
	    
	  k++;
	}
      }
    }
  }

  if (measure_time){
    dtime(&Etime);
    printf("timeB3 myid=%2d %15.12f\n",myid,Etime-Stime);
  }

  /* freeing of arrays */

  free(My_NZeros);
  free(My_Matomnum);
  free(is1);
  free(ie1);
  free(is2);
  free(order_GA);

  if (measure_time){
    dtime(&AEtime);
    printf("timeB_all myid=%2d %15.12f\n",myid,AEtime-AStime);
  }
}




void Construct_Band_Mps( int cpx_flag, int p, 
                         double ****Mat, double *M1, dcomplex *Ms, 
                         int *MP, double k1, double k2, double k3)
{
  static int firsttime=1;
  int i,j,k;
  int MA_AN,GA_AN,LB_AN,GB_AN,AN,Rn,l1,l2,l3;
  int wanA,wanB,tnoA,tnoB,Anum,Bnum,NUM;
  int num,tnum,num_orbitals;
  int ID,myid,numprocs,tag=999;
  int *My_NZeros;
  int *is1,*ie1,*is2;
  int *My_Matomnum,*order_GA;
  double sum,kRn,si,co;
  double Stime,Etime;
  double AStime,AEtime;
  MPI_Status stat;
  MPI_Request request;
  int ig,jg,il,jl,prow,pcol,brow,bcol;

  if (measure_time){
    dtime(&AStime);
    dtime(&Stime);
  }

  /* MPI */

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);
  MPI_Barrier(mpi_comm_level1);

  /* allocation of arrays */

  My_NZeros = (int*)malloc(sizeof(int)*numprocs);
  My_Matomnum = (int*)malloc(sizeof(int)*numprocs);
  is1 = (int*)malloc(sizeof(int)*numprocs);
  ie1 = (int*)malloc(sizeof(int)*numprocs);
  is2 = (int*)malloc(sizeof(int)*numprocs);
  order_GA = (int*)malloc(sizeof(int)*(atomnum+2));

  if (firsttime && memoryusage_fileout) {
  PrintMemory("Electric_Polarization_BandNonCol: My_NZeros", sizeof(int)*numprocs,NULL);
  PrintMemory("Electric_Polarization_BandNonCol: SP_NZeros", sizeof(int)*numprocs,NULL);
  PrintMemory("Electric_Polarization_BandNonCol: SP_Atoms", sizeof(int)*numprocs,NULL);
  PrintMemory("Electric_Polarization_BandNonCol: is1", sizeof(int)*numprocs,NULL);
  PrintMemory("Electric_Polarization_BandNonCol: ie1", sizeof(int)*numprocs,NULL);
  PrintMemory("Electric_Polarization_BandNonCol: is2", sizeof(int)*numprocs,NULL);
  PrintMemory("Electric_Polarization_BandNonCol: order_GA", sizeof(int)*(atomnum+2),NULL);
  }
  firsttime = 1;

  /* find my total number of non-zero elements in myid */

  My_NZeros[myid] = 0;
  for (MA_AN=1; MA_AN<=Matomnum; MA_AN++){
    GA_AN = M2G[MA_AN];
    wanA = WhatSpecies[GA_AN];
    tnoA = Spe_Total_CNO[wanA];

    num = 0;      
    for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
      GB_AN = natn[GA_AN][LB_AN];
      wanB = WhatSpecies[GB_AN];
      tnoB = Spe_Total_CNO[wanB];
      num += tnoB;
    }

    My_NZeros[myid] += tnoA*num;
  }

  for (ID=0; ID<numprocs; ID++){
    MPI_Bcast(&My_NZeros[ID],1,MPI_INT,ID,mpi_comm_level1);
  }

  tnum = 0;
  for (ID=0; ID<numprocs; ID++){
    tnum += My_NZeros[ID];
  }  

  is1[0] = 0;
  ie1[0] = My_NZeros[0] - 1;

  for (ID=1; ID<numprocs; ID++){
    is1[ID] = ie1[ID-1] + 1;
    ie1[ID] = is1[ID] + My_NZeros[ID] - 1;
  }  

  /* set is2 and order_GA */

  My_Matomnum[myid] = Matomnum;
  for (ID=0; ID<numprocs; ID++){
    MPI_Bcast(&My_Matomnum[ID],1,MPI_INT,ID,mpi_comm_level1);
  }

  is2[0] = 1;
  for (ID=1; ID<numprocs; ID++){
    is2[ID] = is2[ID-1] + My_Matomnum[ID-1];
  }
  
  for (MA_AN=1; MA_AN<=Matomnum; MA_AN++){
    order_GA[is2[myid]+MA_AN-1] = M2G[MA_AN];
  }

  for (ID=0; ID<numprocs; ID++){
    MPI_Bcast(&order_GA[is2[ID]],My_Matomnum[ID],MPI_INT,ID,mpi_comm_level1);
  }

  /* set MP */

  Anum = 1;
  for (i=1; i<=atomnum; i++){
    MP[i] = Anum;
    wanA = WhatSpecies[i];
    Anum += Spe_Total_CNO[wanA];
  }
  NUM = Anum - 1;

  /* set M1 */

  for (i=0; i<tnum; i++) M1[i] = 0.0;

  k = is1[myid];
  for (MA_AN=1; MA_AN<=Matomnum; MA_AN++){
    GA_AN = M2G[MA_AN];
    wanA = WhatSpecies[GA_AN];
    tnoA = Spe_Total_CNO[wanA];
    for (i=0; i<tnoA; i++){
      for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
        GB_AN = natn[GA_AN][LB_AN];
        wanB = WhatSpecies[GB_AN];
        tnoB = Spe_Total_CNO[wanB];
        for (j=0; j<tnoB; j++){
          M1[k] = Mat[MA_AN][LB_AN][i][j]; 
          k++;
	}
      }
    }
  }

  if (measure_time){
    dtime(&Etime);
    printf("timeB1 myid=%2d %15.12f\n",myid,Etime-Stime);
  }

  if (measure_time){
    dtime(&Stime);
  }

  /* MPI M1 */

  MPI_Allreduce(MPI_IN_PLACE, &M1[0], tnum, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

  if (measure_time){
    dtime(&Etime);
    printf("timeB2 myid=%2d %15.12f\n",myid,Etime-Stime);
  }

  if (measure_time){
    dtime(&Stime);
  }

  /* M1 -> Ms */
  
  k = 0;
  for (AN=1; AN<=atomnum; AN++){
    GA_AN = order_GA[AN];
    wanA = WhatSpecies[GA_AN];
    tnoA = Spe_Total_CNO[wanA];
    Anum = MP[GA_AN];

    for (i=0; i<tnoA; i++){

      for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
	GB_AN = natn[GA_AN][LB_AN];
        Rn = ncn[GA_AN][LB_AN];
	wanB = WhatSpecies[GB_AN];
	tnoB = Spe_Total_CNO[wanB];
	Bnum = MP[GB_AN];

	l1 = atv_ijk[Rn][1];
	l2 = atv_ijk[Rn][2];
	l3 = atv_ijk[Rn][3];
	kRn = k1*(double)l1 + k2*(double)l2 + k3*(double)l3;

	si = sin(2.0*PI*kRn);
	co = cos(2.0*PI*kRn);

	for (j=0; j<tnoB; j++){
	  ig = Anum+i;
	  jg = Bnum+j;
	    
	  brow = (ig-1)/nblk;
	  bcol = (jg-1)/nblk;

	  prow = brow%np_rows;
	  pcol = bcol%np_cols;

	  if(my_prow==prow && my_pcol==pcol){

	    il = (brow/np_rows+1)*nblk+1;
	    jl = (bcol/np_cols+1)*nblk+1;

	    if(((my_prow+np_rows)%np_rows) >= (brow%np_rows)){
	      if(my_prow==prow){
		il = il+(ig-1)%nblk;
	      }
	      il = il-nblk;
	    }

	    if(((my_pcol+np_cols)%np_cols) >= (bcol%np_cols)){
	      if(my_pcol==pcol){
		jl = jl+(jg-1)%nblk;
	      }
	      jl = jl-nblk;
	    }

            if (cpx_flag==0){
	      Ms[(jl-1)*na_rows+il-1].r +=-M1[k]*si*atv[Rn][p];
	      Ms[(jl-1)*na_rows+il-1].i += M1[k]*co*atv[Rn][p];;
	    }
            else if (cpx_flag==1){
	      Ms[(jl-1)*na_rows+il-1].r +=-M1[k]*co*atv[Rn][p];
	      Ms[(jl-1)*na_rows+il-1].i +=-M1[k]*si*atv[Rn][p];
	    }
	  }
	    
	  k++;
	}
      }
    }
  }

  if (measure_time){
    dtime(&Etime);
    printf("timeB3 myid=%2d %15.12f\n",myid,Etime-Stime);
  }

  /* freeing of arrays */

  free(My_NZeros);
  free(My_Matomnum);
  free(is1);
  free(ie1);
  free(is2);
  free(order_GA);

  if (measure_time){
    dtime(&AEtime);
    printf("timeB_all myid=%2d %15.12f\n",myid,AEtime-AStime);
  }
}



void Analytic_Berry_Connection( int Nocc, int add_flag,
				int kloop, int n2, int MaxN, 
				int *MP, int *T_k_op,
				double k1, double k2, double k3,
				double *H1, 
				dcomplex *Ss2, dcomplex *Cs2, dcomplex *Hs2,  
				dcomplex *Vs2, dcomplex *Qs2, 
                                dcomplex *Hs11, dcomplex *Hs22, dcomplex *Hs12,
				double ***EIGEN,
		                double *****nh,double *****ImNL,double ****CntOLP,
				dcomplex *ABC,
				dcomplex *VDMs_ABC,
				int *order_GA, 
				int all_knum, 
				int myworld1,
				int myworld2,
				MPI_Comm *MPI_CommWD1,
				MPI_Comm *MPI_CommWD2 )
{
  int LB_AN,i,j,i0,i1,l,m,GB_AN,Rn,wanB,tnoB,Bnum,mu,nu;
  int target_Atom,l1,l2,l3,po,p,q,g,LWORK,N1,N2,ia,ja,ib,jb,lwork;
  int j1,il,jl,bcol,pcol,brow,prow,Anum,wanA,tnoA,ig,jg,k,AN,GA_AN,info;
  double x,y,z,x0,y0,z0,kRn,si,co,omega,asum1[2];
  double absA2,absA,xf,FermiF,tmp1,d,del=1.0e-10;
  double x_cut=60.0,ko_ref,v,max_v,absB,*rwork;
  double ei,ej,tol=1.0e-5;
  int ZERO=0, ONE=1;
  dcomplex alpha = {1.0,0.0}; dcomplex beta = {0.0,0.0};
  dcomplex sumA,sumB,ctmp1,ctmp2,ctmp3,csum1,csum2,phase,xdiv,ctmp;
  dcomplex *A,*B,*mat,*mat2,*mat3,*mat4,*work;
  dcomplex *A11s,*A12s,*B12s,*C12s;
  double *sv;

  /* allocation of arrays */

  N1 = Nocc;
  N2 = n2 - Nocc;
  A11s = (dcomplex*)malloc(sizeof(dcomplex)*N1*N1);
  A12s = (dcomplex*)malloc(sizeof(dcomplex)*N1*N2);
  B12s = (dcomplex*)malloc(sizeof(dcomplex)*N1*N2);
  C12s = (dcomplex*)malloc(sizeof(dcomplex)*N1*N2);
  sv = (double*)malloc(sizeof(double)*N1);
  rwork = (double*)malloc(sizeof(double)*(1+4*N1));

  /* get size of work and allocate work */

  lwork = -1;
  F77_NAME(pzgesvd,PZGESVD)( "V", "V",
                             &N1, &N1,
			     A11s, &ONE, &ONE, desc11,
			     sv,
			     Qs2, &ONE, &ONE, desc11,
			     Vs2, &ONE, &ONE, desc11,
			     &ctmp,
			     &lwork,
                             rwork,
			     &info);

  lwork = (int)ctmp.r;
  work = (dcomplex*)malloc(sizeof(dcomplex)*lwork);

  /***********************************************************
         construction of A11s, A12s, and B12s matrices.
  ***********************************************************/

  /* calculate <projection Vec|eigen vectors>*/ 

  Cblacs_barrier(ictxt1_2,"A");
  F77_NAME(pzgemm,PZGEMM)( "T","T",&n2,&n2,&n2,&alpha,VDMs_ABC,&ONE,&ONE,descH2,Hs2,
			   &ONE,&ONE,descS2,&beta,Cs2,&ONE,&ONE,descC2 );

  /* redistribution of A to construct A11s */

  ia = 1; ja = 1; ib = 1; jb = 1;
  F77_NAME(pzgemr2d,PDGEMR2D)(&N1, &N1, Cs2, &ia, &ja, descC2, A11s, &ib, &jb, desc11, &ictxt11);

  /* redistribution of A to construct A12s */

  ia = 1; ja = N1+1; ib = 1; jb = 1;
  F77_NAME(pzgemr2d,PDGEMR2D)(&N1, &N2, Cs2, &ia, &ja, descC2, A12s, &ib, &jb, desc12, &ictxt12);

  /* singular value decomposition for A11 -> U Sigma V^dag */

  F77_NAME(pzgesvd,PZGESVD)( "V", "V",
			     &N1, &N1,
			     A11s, &ONE, &ONE, desc11,
			     sv,
			     Qs2, &ONE, &ONE, desc11,
			     Vs2, &ONE, &ONE, desc11,
			     work,
			     &lwork,
                             rwork,
			     &info);

  /* calculate V Sigma^-1 U^dag -> A11s */

  for (j=0; j<na_cols11; j++){

    jg = np_cols11*nblk11*((j)/nblk11) + (j)%nblk11 
         + ((np_cols11+my_pcol11)%np_cols11)*nblk11;

    for (i=0; i<na_rows11; i++){
      Qs2[j*na_rows11+i].r /= sv[jg]; 
      Qs2[j*na_rows11+i].i /= sv[jg]; 
    }
  }

  Cblacs_barrier(ictxt11,"A");
  F77_NAME(pzgemm,PZGEMM)( "C","C",
			   &N1, &N1, &N1, 
			   &alpha,
			   Vs2,&ONE,&ONE,desc11,
			   Qs2,&ONE,&ONE,desc11,
			   &beta,
			   A11s,&ONE,&ONE,desc11 );

  /* calculate A11s x A12s -> B12s */

  Cblacs_barrier(ictxt12,"A");
  F77_NAME(pzgemm,PZGEMM)( "N","N",
			   &N1, &N2, &N1, 
			   &alpha,
			   A11s,&ONE,&ONE,desc11,
			   A12s,&ONE,&ONE,desc12,
			   &beta,
			   B12s,&ONE,&ONE,desc12 );

  /************************************
         loop for x, y, and z
  ************************************/

  for (p=1; p<=3; p++){

    /********************************************
              calculations of S' and H'
    ********************************************/

    /* calculate S' and H' */

    for(i=0; i<na_rows2; i++){
      for(j=0; j<na_cols2; j++){
	Qs2[j*na_rows2+i] = Complex(0.0,0.0); // S'
	Vs2[j*na_rows2+i] = Complex(0.0,0.0); // H'
      }
    }

    for (i=0; i<na_rows*na_cols; i++){
      Hs11[i] = Complex(0.0,0.0);
      Hs22[i] = Complex(0.0,0.0);
      Hs12[i] = Complex(0.0,0.0);
    }
    
    Construct_Band_Mps(0,p,nh[0],  H1, Hs11,MP,k1,k2,k3);
    Construct_Band_Mps(0,p,nh[1],  H1, Hs22,MP,k1,k2,k3);
    Construct_Band_Mps(0,p,nh[2],  H1, Hs12,MP,k1,k2,k3);
    Construct_Band_Mps(1,p,nh[3],  H1, Hs12,MP,k1,k2,k3);
    Construct_Band_Mps(1,p,ImNL[0],H1, Hs11,MP,k1,k2,k3);
    Construct_Band_Mps(1,p,ImNL[1],H1, Hs22,MP,k1,k2,k3);
    Construct_Band_Mps(1,p,ImNL[2],H1, Hs12,MP,k1,k2,k3);
    Hamiltonian_Band_NC_Hs2( Hs11, Hs22, Hs12, Vs2, MPI_CommWD2[myworld2] );

    for (i=0; i<na_rows*na_cols; i++){
      Hs11[i] = Complex(0.0,0.0);
      Hs22[i] = Complex(0.0,0.0);
      Hs12[i] = Complex(0.0,0.0);
    }

    Construct_Band_Mps(0,p,CntOLP, H1, Hs11,MP,k1,k2,k3);
    Construct_Band_Mps(0,p,CntOLP, H1, Hs22,MP,k1,k2,k3);
    Hamiltonian_Band_NC_Hs2( Hs11, Hs22, Hs12, Qs2, MPI_CommWD2[myworld2] );

    /* calculate <c_i|S'|c_j> */

    for(i=0; i<na_rows2*na_cols2; i++){ Cs2[i] = Complex(0.0,0.0); }

    Cblacs_barrier(ictxt2,"A");

    F77_NAME(pzgemm,PZGEMM)( "N","T",&n2,&n2,&n2,&alpha,Qs2,&ONE,&ONE,descH2,Hs2,
			     &ONE,&ONE,descS2,&beta,Cs2,&ONE,&ONE,descC2 );

    for(i=0; i<na_rows2*na_cols2; i++){ Qs2[i] = Complex(0.0,0.0); }

    Cblacs_barrier(ictxt2,"A");

    for (i=0; i<na_rows2*na_cols2; i++){ Hs2[i].i = -Hs2[i].i; }

    F77_NAME(pzgemm,PZGEMM)( "N","N",&n2,&n2,&n2,&alpha,Hs2,&ONE,&ONE,descS2,Cs2,
			     &ONE,&ONE,descC2,&beta,Qs2,&ONE,&ONE,descH2 );

    for (i=0; i<na_rows2*na_cols2; i++){ Hs2[i].i = -Hs2[i].i; }

    /* calculate <c_i|H'|c_j> */

    for(i=0; i<na_rows2*na_cols2; i++){ Cs2[i] = Complex(0.0,0.0); }

    Cblacs_barrier(ictxt2,"A");

    F77_NAME(pzgemm,PZGEMM)( "N","T",&n2,&n2,&n2,&alpha,Vs2,&ONE,&ONE,descH2,Hs2,
			     &ONE,&ONE,descS2,&beta,Cs2,&ONE,&ONE,descC2 );

    for(i=0; i<na_rows2*na_cols2; i++){ Vs2[i] = Complex(0.0,0.0); }

    Cblacs_barrier(ictxt2,"A");

    for (i=0; i<na_rows2*na_cols2; i++){ Hs2[i].i = -Hs2[i].i; }

    F77_NAME(pzgemm,PZGEMM)( "N","N",&n2,&n2,&n2,&alpha,Hs2,&ONE,&ONE,descS2,Cs2,
			     &ONE,&ONE,descC2,&beta,Vs2,&ONE,&ONE,descH2 );

    for (i=0; i<na_rows2*na_cols2; i++){ Hs2[i].i = -Hs2[i].i; }

    /**********************************************************
                  calculate the Berry connection 
    **********************************************************/

    /* redistribution of Qs to construct A12s */

    ia = 1; ja = N1+1; ib = 1; jb = 1;
    F77_NAME(pzgemr2d,PDGEMR2D)(&N1, &N2, Qs2, &ia, &ja, descC2, A12s, &ib, &jb, desc12, &ictxt12);

    /* redistribution of Vs to construct C12s which is used as temporal array */

    ia = 1; ja = N1+1; ib = 1; jb = 1;
    F77_NAME(pzgemr2d,PDGEMR2D)(&N1, &N2, Vs2, &ia, &ja, descC2, C12s, &ib, &jb, desc12, &ictxt12);

    /* calculate C12s */    

    for (i=0; i<na_rows12; i++){

      ig = np_rows12*nblk12*((i)/nblk12) + (i)%nblk12 
           + ((np_rows12+my_prow12)%np_rows12)*nblk12 + 1;

      ei = EIGEN[0][kloop][ig]; 

      for (j=0; j<na_cols12; j++){

        jg = np_cols12*nblk12*((j)/nblk12) + (j)%nblk12
             + ((np_cols12+my_pcol12)%np_cols12)*nblk12 + N1 + 1;

        ej = EIGEN[0][kloop][jg]; 
        d = ei - ej;
        tmp1 = d/(d*d + del);

	C12s[j*na_rows12+i].r = (C12s[j*na_rows12+i].r - ei*A12s[j*na_rows12+i].r)*tmp1;
	C12s[j*na_rows12+i].i = (C12s[j*na_rows12+i].i - ei*A12s[j*na_rows12+i].i)*tmp1;
      }
    }

    /* calculate B12s x C12s^dag -> A11s */    

    Cblacs_barrier(ictxt12,"A");
    F77_NAME(pzgemm,PZGEMM)( "N","C",
			     &N1, &N1, &N2, 
			     &alpha,
			     B12s,&ONE,&ONE,desc12,
			     C12s,&ONE,&ONE,desc12,
			     &beta,
			     A11s,&ONE,&ONE,desc11 );

    /* calculate Tr[B12s x C12s^dag -> A11s] */

    asum1[0] = 0.0; asum1[1] = 0.0; 
    for (i=0; i<na_rows11; i++){

      ig = np_rows11*nblk11*((i)/nblk11) + (i)%nblk11 
	+ ((np_rows11+my_prow11)%np_rows11)*nblk11;

      for (j=0; j<na_cols11; j++){

	jg = np_cols11*nblk11*((j)/nblk11) + (j)%nblk11 
	  + ((np_cols11+my_pcol11)%np_cols11)*nblk11;

	if (ig==jg){
	  asum1[0] += A11s[j*na_rows11+i].r;
	  asum1[1] += A11s[j*na_rows11+i].i;
	} 
      }
    }

    /* Add the Berry connection to ABC.  
       MPI_Allreduce will be performed to gather all the contributions at end. */

    if (add_flag==1){
      ABC[p-1].r -= (double)T_k_op[kloop]*asum1[0];
      ABC[p-1].i -= (double)T_k_op[kloop]*asum1[1];

      /*
      if (p==1){
        printf("VVV kloop=%2d p=%2d asum1=%15.12f %15.12f\n",kloop,p,asum1[0],asum1[1]);fflush(stdout);
      }
      */
    }

  } // end of p 

  /* freeing of arrays */
  
  free(A11s);
  free(A12s);
  free(B12s);
  free(C12s);
  free(sv);
  free(rwork);
  free(work);
}


double EP1( double ****OLP0, double ******OLPpo, double *****CDM, 
            double EPol[3], double CPol[3], double EPol_BG[3] )
{
  int i,j,tno0,tno1,spin,Mc_AN,Gc_AN,h_AN,Gh_AN,Cwan,Hwan,Rn,l1,l2,l3;
  double My_EPol[3];
  double matx,maty,matz,Rx,Ry,Rz;
  double My_CPol[3],x,y,z,charge;
  int numprocs,myid;
  int N2D,GN,GNs,BN,n1,n2,n3;
  double My_E_dpx_BG,My_E_dpy_BG,My_E_dpz_BG;
  double cden_BG;

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  Calc_OLPpo(OLPpo);

  /* contribution from electronic wave functions */

  My_EPol[0] = 0.0; My_EPol[1] = 0.0; My_EPol[2] = 0.0;

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];    
    Cwan = WhatSpecies[Gc_AN];
    tno0 = Spe_Total_CNO[Cwan];

    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

      Gh_AN = natn[Gc_AN][h_AN];
      Rn = ncn[Gc_AN][h_AN];
      Hwan = WhatSpecies[Gh_AN];
      tno1 = Spe_Total_CNO[Hwan];

      l1 = atv_ijk[Rn][1];
      l2 = atv_ijk[Rn][2];
      l3 = atv_ijk[Rn][3];
      Rx = (double)l1*tv[1][1] + (double)l2*tv[2][1] + (double)l3*tv[3][1];
      Ry = (double)l1*tv[1][2] + (double)l2*tv[2][2] + (double)l3*tv[3][2];
      Rz = (double)l1*tv[1][3] + (double)l2*tv[2][3] + (double)l3*tv[3][3];

      for (i=0; i<tno0; i++){
	for (j=0; j<tno1; j++){

	  matx = OLPpo[0][0][Mc_AN][h_AN][i][j] - Rx*OLP0[Mc_AN][h_AN][i][j];
	  maty = OLPpo[1][0][Mc_AN][h_AN][i][j] - Ry*OLP0[Mc_AN][h_AN][i][j];
	  matz = OLPpo[2][0][Mc_AN][h_AN][i][j] - Rz*OLP0[Mc_AN][h_AN][i][j];

	  My_EPol[0] += (CDM[0][Mc_AN][h_AN][i][j]+CDM[1][Mc_AN][h_AN][i][j])*matx;
	  My_EPol[1] += (CDM[0][Mc_AN][h_AN][i][j]+CDM[1][Mc_AN][h_AN][i][j])*maty;
	  My_EPol[2] += (CDM[0][Mc_AN][h_AN][i][j]+CDM[1][Mc_AN][h_AN][i][j])*matz;
	}
      }

    } // h_AN
  } // Mc_AN

  MPI_Barrier(mpi_comm_level1);
  MPI_Allreduce(&My_EPol[0], &EPol[0], 3, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

  /* contribution from core charge */

  My_CPol[0] = 0.0; My_CPol[1] = 0.0; My_CPol[2] = 0.0;

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];
    x = Gxyz[Gc_AN][1];
    y = Gxyz[Gc_AN][2];
    z = Gxyz[Gc_AN][3];

    Cwan = WhatSpecies[Gc_AN];
    charge = Spe_Core_Charge[Cwan];
    My_CPol[0] += charge*x;
    My_CPol[1] += charge*y;
    My_CPol[2] += charge*z;
  }

  MPI_Allreduce(&My_CPol[0], &CPol[0], 3, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

  /* contribution from background charge */

  N2D = Ngrid1*Ngrid2;
  GNs = ((myid*N2D+numprocs-1)/numprocs)*Ngrid3;

  My_E_dpx_BG = 0.0;
  My_E_dpy_BG = 0.0;
  My_E_dpz_BG = 0.0; 

  for (BN=0; BN<My_NumGridB_AB; BN++){

    GN = BN + GNs;     
    n1 = GN/(Ngrid2*Ngrid3);    
    n2 = (GN - n1*Ngrid2*Ngrid3)/Ngrid3;
    n3 = GN - n1*Ngrid2*Ngrid3 - n2*Ngrid3; 

    x = (double)n1*gtv[1][1] + (double)n2*gtv[2][1]
      + (double)n3*gtv[3][1] + Grid_Origin[1];
    y = (double)n1*gtv[1][2] + (double)n2*gtv[2][2]
      + (double)n3*gtv[3][2] + Grid_Origin[2];
    z = (double)n1*gtv[1][3] + (double)n2*gtv[2][3]
      + (double)n3*gtv[3][3] + Grid_Origin[3];
   
    My_E_dpx_BG += x;
    My_E_dpy_BG += y;
    My_E_dpz_BG += z; 
    
  } /* BN */

  MPI_Allreduce(&My_E_dpx_BG, &EPol_BG[0], 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
  MPI_Allreduce(&My_E_dpy_BG, &EPol_BG[1], 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
  MPI_Allreduce(&My_E_dpz_BG, &EPol_BG[2], 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

  cden_BG = system_charge/Cell_Volume; 

  EPol_BG[0] = EPol_BG[0]*GridVol*cden_BG;
  EPol_BG[1] = EPol_BG[1]*GridVol*cden_BG;
  EPol_BG[2] = EPol_BG[2]*GridVol*cden_BG;

  return 0.0;
}

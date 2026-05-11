/**********************************************************************
  Electric_Polarization_BandCol.c:

   Electric_Polarization_BandCol.c is a subroutine to calculate polarization 
   using the Analytic Berry Connection (ABC) formula for the occupied space
   in the collinear band calculation. 

  Log of Electric_Polarization_BandCol.c:

    10/April/2025  Released by T. Ozaki

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "mpi.h"
#include "openmx_common.h"
#include "lapack_prototypes.h"
#include "tran_variables.h"
#include <omp.h>

#define  measure_time  0

void solve_evp_real_( int *n1, int *n2, double *Cs, int *na_rows1, double *a, double *Ss, int *na_rows2, int *nblk, 
                      int *mpi_comm_rows_int, int *mpi_comm_cols_int);

void elpa_solve_evp_real_2stage_double_impl_( int *n1, int *n2, double *Cs, int *na_rows1, double *a, double *Ss, int *na_rows2, 
                                              int *nblk, int *na_cols1, int *mpi_comm_rows_int, int *mpi_comm_cols_int, int *mpiworld);

 
void solve_evp_complex_( int *n1, int *n2, dcomplex *Cs, int *na_rows1, double *ko, dcomplex *Ss, 
                         int *na_rows2, int *nblk, int *mpi_comm_rows_int, int *mpi_comm_cols_int );

void elpa_solve_evp_complex_2stage_double_impl_
      ( int *n, int *MaxN, dcomplex *Hs, int *na_rows1, double *ko, dcomplex *Cs, 
        int *na_rows2, int *nblk, int *na_cols1,
        int *mpi_comm_rows_int, int *mpi_comm_cols_int, int *mpiworld );



static void Analytic_Berry_Connection(
                                int Nocc,
				int spin, int kloop, int n, int MaxN, 
				int *MP, int *T_k_op,
				double k1, double k2, double k3,
				double *S1, double *H1,
				dcomplex *Hs, dcomplex *Cs, 
				dcomplex *Vs, dcomplex *Qs, 
				double ***EIGEN, 
				dcomplex **ABC,
				dcomplex **VDMs_ABC,
				int *order_GA, 
	   		        int all_knum, 
				int myworld1,
				int myworld2,
				MPI_Comm *MPI_CommWD1,
				MPI_Comm *MPI_CommWD2 );


void Calc_OLPpo( double ******OLPpo );
void Allocate_Free_Electric_Polarization(int todo_flag);
static double EP1( double ****OLP0, double ******OLPpo, double *****CDM, 
    	           double EPol[2][3], double CPol[3], double EPol_BG[3] );

int nblk11[2],np_rows11[2],np_cols11[2],na_rows11[2],na_cols11[2];
int na_rows_max11[2],na_cols_max11[2];
int my_prow11[2],my_pcol11[2];
int bhandle11[2],ictxt11[2];
int desc11[2][9];
int nblk12[2],np_rows12[2],np_cols12[2],na_rows12[2],na_cols12[2];
int na_rows_max12[2],na_cols_max12[2];
int my_prow12[2],my_pcol12[2];
int bhandle12[2],ictxt12[2];
int desc12[2][9];
double ******OLPpo;


double Electric_Polarization_BandCol(
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
		    double *S1,   
		    double *CDM1,  
		    double *EDM1,
		    dcomplex *Ss,
		    dcomplex *Cs,
                    dcomplex *Hs,
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
  int i,j,k,l,m,n,p,q,wan,MaxN,i0,ks;
  int i1,i1s,j1,ia,jb,lmax,kmin,kmax,po,po1,spin,s1,e1;
  int num2,RnB,l1,l2,l3,loop_num,ns,ne,n1,n2;
  int ct_AN,h_AN,wanA,tnoA,wanB,tnoB;
  int MA_AN,GA_AN,Anum,num_kloop0;
  int T_knum,S_knum,E_knum,kloop,kloop0;
  double av_num,lumos;
  double time0,be[4];
  int LB_AN,GB_AN,Bnum;
  double k1,k2,k3,Fkw;
  double sum,sumi,sum_weights;
  double Num_State;
  double My_Num_State;
  double FermiF,tmp0,tmp1,tmp2;
  double tmp,eig,kw,EV_cut0;
  int *My_NZeros;
  int *SP_NZeros;
  int *SP_Atoms;

  int all_knum; 
  dcomplex ctmp1,ctmp2;
  int ii,ij,ik;
  int BM,BN,BK;
  double u2,v2,uv,vu,fac;
  double d1,d2,d3,d4,ReA,ImA;
  double My_Eele1[2]; 
  double TZ,dum,sumE,kRn,si,co;
  double Resum,ResumE,Redum,Redum2;
  double Imsum,ImsumE,Imdum,Imdum2;
  double TStime,TEtime,SiloopTime,EiloopTime;
  double Stime,Etime,Stime0,Etime0;
  double Stime1,Etime1;
  double FermiEps=1.0e-13;
  double x_cut=60.0;
  double My_Eele0[2];

  char file_EV[YOUSO10];
  FILE *fp_EV;
  char buf[fp_bsize];          /* setvbuf */
  int AN,Rn,size_H1;
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
  double time10,time11,time12;
  double time81,time82,time83;
  double time84,time85;
  double time51,time11A,time11B;

  MPI_Comm mpi_comm_rows, mpi_comm_cols;
  int mpi_comm_rows_int,mpi_comm_cols_int;
  int info,ig,jg,ig1,jg1,il,jl,prow,pcol,brow,bcol;
  int ZERO=0, ONE=1;
  dcomplex alpha = {1.0,0.0}; dcomplex beta = {0.0,0.0};
  int LOCr, LOCc, node, irow, icol;
  double mC_spin_i1,C_spin_i1;
  int Max_Num_Snd_EV,Max_Num_Rcv_EV;
  double tmparray[3];
  dcomplex **ABC,*Vs,*Qs,**VDMs_ABC;
  double **DMs,*VDMs;
  double EPol[2][3],CPol[3],EPol_BG[3];

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
  time81 = 0.0;
  time82 = 0.0;
  time83 = 0.0;
  time84 = 0.0;
  time85 = 0.0;
  time51 = 0.0;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs0);
  MPI_Comm_rank(mpi_comm_level1,&myid0);
  MPI_Barrier(mpi_comm_level1);

  Num_Comm_World1 = SpinP_switch + 1; 

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

  MaxN = n;

  /***********************************************
     allocation of arrays
  ***********************************************/

  My_NZeros = (int*)malloc(sizeof(int)*numprocs0);
  SP_NZeros = (int*)malloc(sizeof(int)*numprocs0);
  SP_Atoms = (int*)malloc(sizeof(int)*numprocs0);

  /***********************************************
              k-points by regular mesh 
  ***********************************************/
   
  if (way_of_kpoint==1){

    /**************************************************************
     k_op[i][j][k]: weight of DOS 
                 =0   no calc.
                 =1   G-point
                 =2   which has k<->-k point
        Now, only the relation, E(k)=E(-k), is used. 

    Future release: k_op will be used for symmetry operation 
    *************************************************************/

    for (i=0;i<knum_i;i++) {
      for (j=0;j<knum_j;j++) {
	for (k=0;k<knum_k;k++) {
	  k_op[i][j][k]=-999;
	}
      }
    }

    for (i=0;i<knum_i;i++) {
      for (j=0;j<knum_j;j++) {
	for (k=0;k<knum_k;k++) {

	  if ( k_op[i][j][k]==-999 ) {
	    k_inversion(i,j,k,knum_i,knum_j,knum_k,&ii,&ij,&ik);
	    if ( i==ii && j==ij && k==ik ) {
	      k_op[i][j][k]    = 1;
	    }

	    else {
	      k_op[i][j][k]    = 2;
	      k_op[ii][ij][ik] = 0;
	    }
	  }

	} /* k */
      } /* j */
    } /* i */

    /***********************************
       one-dimentionalize for MPI
    ************************************/

    /* set T_KGrids1,2,3 and T_k_op */

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

  }

  /***********************************************
                Monkhorst-Pack k-points 
  ***********************************************/

  else if (way_of_kpoint==2){

    T_knum = num_non_eq_kpt; 
   
    for (k=0; k<num_non_eq_kpt; k++){
      T_KGrids1[k] = NE_KGrids1[k];
      T_KGrids2[k] = NE_KGrids2[k];
      T_KGrids3[k] = NE_KGrids3[k];
      T_k_op[k]    = NE_T_k_op[k];
    }
  }

  /***********************************************
     k-points by a Gamma-centered mesh

     k_op[i][j][k]: weight of DOS 
                 =0  no calc.
                 =1  time reversal invariant momentum (TRIM)
                 =2  which has k<->-k point
        Now, only the relation, E(k)=E(-k), is used. 
  ***********************************************/
  
  else if (way_of_kpoint==3){

    for (i=0;i<knum_i;i++) {
      for (j=0;j<knum_j;j++) {
	for (k=0;k<knum_k;k++) {
	  k_op[i][j][k]=-999;
	}
      }
    }

    for (i=0;i<knum_i;i++) {
      for (j=0;j<knum_j;j++) {
	for (k=0;k<knum_k;k++) {
	  if ( k_op[i][j][k]==-999 ) {

	    if (i==0 || 2*i==knum_i){
              ii=i;
            } else {
              ii=knum_i-i;
	    }

	    if (j==0 || 2*j==knum_j){
              ij=j;
            } else {
              ij=knum_j-j;
            }

	    if (k==0 || 2*k==knum_k){
              ik=k;
            } else {
              ik=knum_k-k;
            }

	    if ((i==0 || 2*i==knum_i) && (j==0 || 2*j==knum_j) && (k==0 || 2*k==knum_k)){
	      k_op[i][j][k]    = 1;
	    } 
            else {
	      k_op[i][j][k]    = 2;
	      k_op[ii][ij][ik] = 0;
	    }
	  }

	} /* k */
      } /* j */
    } /* i */

    /***********************************
       one-dimentionalize for MPI
    ************************************/

    /* set T_KGrids1,2,3 and T_k_op */

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
  }

  /***********************************************
     allocation of arrays for the ABC formula
  ***********************************************/

  ABC = (dcomplex**)malloc(sizeof(dcomplex*)*2);
  for (spin=0; spin<2; spin++){
    ABC[spin] = (dcomplex*)malloc(sizeof(dcomplex)*3);
    for (i=0; i<3; i++){
      ABC[spin][i] = Complex(0.0,0.0); 
    }
  }

  Vs = (dcomplex*)malloc(sizeof(dcomplex)*na_rows_max*na_cols_max);
  Qs = (dcomplex*)malloc(sizeof(dcomplex)*na_rows_max*na_cols_max);

  DMs = (double**)malloc(sizeof(double*)*2);
  for (spin=0; spin<2; spin++){
    DMs[spin]= (double*)malloc(sizeof(double)*na_rows_max*na_cols_max);
    for (i=0; i<na_rows_max*na_cols_max; i++){ DMs[spin][i] = 0.0; }
  }

  VDMs = (double*)malloc(sizeof(double)*na_rows_max*na_cols_max);

  VDMs_ABC = (dcomplex**)malloc(sizeof(dcomplex*)*(SpinP_switch+1));
  for (spin=0; spin<=SpinP_switch; spin++){
    VDMs_ABC[spin] = (dcomplex*)malloc(sizeof(dcomplex)*na_rows_max*na_cols_max);
  }

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

  if (SpinP_switch==1 && numprocs0==1 && all_knum==1){
    all_knum = 0;
  }

  /****************************************************
     PrintMemory
  ****************************************************/

  if (firsttime && memoryusage_fileout) {
    PrintMemory("Electric_Polarization_BandCol: My_NZeros", sizeof(int)*numprocs0,NULL);
    PrintMemory("Electric_Polarization_BandCol: SP_NZeros", sizeof(int)*numprocs0,NULL);
    PrintMemory("Electric_Polarization_BandCol: SP_Atoms", sizeof(int)*numprocs0,NULL);
  }

  /****************************************************
     communicate T_k_ID
  ****************************************************/

  if (numprocs0==1 && SpinP_switch==1){
    for (k=0; k<T_knum; k++){
      T_k_ID[1][k] = T_k_ID[0][k];
    }
  }
  else{
    for (spin=0; spin<=SpinP_switch; spin++){
      ID = Comm_World_StartID1[spin];
      MPI_Bcast(&T_k_ID[spin][0], T_knum, MPI_INT, ID, mpi_comm_level1);
    }
  }

  /***********************************************
       calculation of projection functions
  ***********************************************/

  spin = myworld1;

  if (SpinP_switch==0){ 
    size_H1 = Get_OneD_HS_Col(1, CDM[0], S1, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);
  }
  else{
    size_H1 = Get_OneD_HS_Col(1, CDM[0], S1, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);
    size_H1 = Get_OneD_HS_Col(1, CDM[1], H1, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);
  }

Calc_PFs:

  /* set DMs */

  k = 0;
  for (AN=1; AN<=atomnum; AN++){

    GA_AN = order_GA[AN];
    wanA = WhatSpecies[GA_AN];
    tnoA = Spe_Total_CNO[wanA];
    Anum = MP[GA_AN];

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

      for (i=0; i<tnoA; i++){

	ig = Anum + i;
	brow = (ig-1)/nblk;
	prow = brow%np_rows;

	for (j=0; j<tnoB; j++){

	  jg = Bnum + j;
	  bcol = (jg-1)/nblk;
	  pcol = bcol%np_cols;

	  if (Rn==0 && my_prow==prow && my_pcol==pcol){

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

	    if (spin==0){ 
	      DMs[spin][(jl-1)*na_rows+il-1] = -S1[k]; 
	    }
	    else{
	      DMs[spin][(jl-1)*na_rows+il-1] = -H1[k]; 
	    }
	  }

	  k++;

	}
      }
    }
  }

  /* diagonalize DMs */

  MPI_Comm_split(MPI_CommWD2[myworld2],my_pcol,my_prow,&mpi_comm_rows);
  MPI_Comm_split(MPI_CommWD2[myworld2],my_prow,my_pcol,&mpi_comm_cols);

  mpi_comm_rows_int = MPI_Comm_c2f(mpi_comm_rows);
  mpi_comm_cols_int = MPI_Comm_c2f(mpi_comm_cols);

  if (scf_eigen_lib_flag==1){

    F77_NAME(solve_evp_real,SOLVE_EVP_REAL)
      ( &n, &n, DMs[spin], &na_rows, &ko[0], VDMs, &na_rows, 
	&nblk, &mpi_comm_rows_int, &mpi_comm_cols_int );

  }

  else if (scf_eigen_lib_flag==2){

#ifndef kcomp

    int mpiworld;
    mpiworld = MPI_Comm_c2f(MPI_CommWD2[myworld2]);
    F77_NAME(elpa_solve_evp_real_2stage_double_impl,ELPA_SOLVE_EVP_REAL_2STAGE_DOUBLE_IMPL)
      (&n, &n, DMs[spin], &na_rows, &ko[0], 
       VDMs, &na_rows, &nblk, &na_cols, &mpi_comm_rows_int, &mpi_comm_cols_int, &mpiworld);

#endif

  }

  MPI_Comm_free(&mpi_comm_rows);
  MPI_Comm_free(&mpi_comm_cols);

  /* VDMs is stored to VDMs_ABC */

  for (i=0; i<na_cols*na_rows; i++){
    VDMs_ABC[spin][i].r = VDMs[i];
    VDMs_ABC[spin][i].i = 0.0;
  }

  if (SpinP_switch==1 && numprocs0==1 && spin==0){
    spin++;  
    goto Calc_PFs; 
  }

  /********************************************************
                   set up BLACS_variables
  *********************************************************/

  if (SpinP_switch==0){

    if ( 1.0e-6<fabs(2.0*(double)Cluster_HOMO[0]-(TZ-system_charge)) ){

      Cluster_HOMO[0] = (int)(0.50*(TZ-system_charge)+1.0e-7);  
      //if (myid0==0) printf("potential error may exist in the ABC calculation.\n");fflush(stdout);
    }
  }
  else {
    if ( 1.0e-6<fabs((double)(Cluster_HOMO[0]+Cluster_HOMO[1])-(TZ-system_charge)) ){

      Cluster_HOMO[1] = TZ - system_charge - Cluster_HOMO[0];
      //if (myid0==0) printf("potential error may exist in the ABC calculation.\n");fflush(stdout);
    }
  }

  /* set up the BLACS configuration */

  for (spin=0; spin<=SpinP_switch; spin++){

    /* A11: Cluster_HOMO[spin] x Cluster_HOMO[spin] */

    n1 = Cluster_HOMO[spin];
    n2 = Cluster_HOMO[spin];

    Set_BLACS_variables( MPI_CommWD2[myworld2],numprocs2,myid2,n1,n2,&np_rows11[spin],&np_cols11[spin],
			 &nblk11[spin],&my_prow11[spin],&my_pcol11[spin],&na_rows11[spin],&na_cols11[spin],
			 &bhandle11[spin],&ictxt11[spin],&na_rows_max11[spin],&na_cols_max11[spin],desc11[spin] );
 
    /* A12: Cluster_HOMO[spin] x (n-Cluster_HOMO[spin]) */ 

    n1 = Cluster_HOMO[spin];
    n2 = n - Cluster_HOMO[spin];

    Set_BLACS_variables( MPI_CommWD2[myworld2],numprocs2,myid2,n1,n2,&np_rows12[spin],&np_cols12[spin],
			 &nblk12[spin],&my_prow12[spin],&my_pcol12[spin],&na_rows12[spin],&na_cols12[spin],
			 &bhandle12[spin],&ictxt12[spin],&na_rows_max12[spin],&na_cols_max12[spin],desc12[spin] );

    desc12[spin][1] = desc11[spin][1]; 

  } /* spin */

  /******************************************************************
  *******************************************************************
     if (all_knum==1)
  *******************************************************************
  ******************************************************************/

  if (measure_time) dtime(&Stime);

  if (all_knum==1){

    /* spin=myworld1 */

    spin = myworld1;

    /* set S1 */

    if (SCF_iter==1 || all_knum!=1){
      size_H1 = Get_OneD_HS_Col(1, CntOLP, S1, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);
    }

  diagonalize1:

    /* set H1 */

    if (SpinP_switch==0){ 
      size_H1 = Get_OneD_HS_Col(1, nh[0], H1, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);
    }
    else if (1<numprocs0){

      size_H1 = Get_OneD_HS_Col(1, nh[0], H1,   MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);
      size_H1 = Get_OneD_HS_Col(1, nh[1], CDM1, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);

      if (myworld1){
	for (i=0; i<size_H1; i++){
	  H1[i] = CDM1[i];
	}
      }
    }
    else{
      size_H1 = Get_OneD_HS_Col(1, nh[spin], H1, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);
    }

    if (measure_time){ 
      dtime(&Etime);
      time1 += Etime - Stime;
    }

    /****************************************************
                       start kloop
    ****************************************************/

    dtime(&SiloopTime);

    for (kloop0=0; kloop0<num_kloop0; kloop0++){

      kloop = S_knum + kloop0;

      k1 = T_KGrids1[kloop];
      k2 = T_KGrids2[kloop];
      k3 = T_KGrids3[kloop];

      /* make S and H */

      if (SCF_iter==1 || all_knum!=1){
	for (i=0; i<na_rows; i++){
	  for (j=0; j<na_cols; j++){
	    Cs[j*na_rows+i] = Complex(0.0,0.0);
	  }
	}
      }

      for(i=0;i<na_rows;i++){
	for(j=0;j<na_cols;j++){
	  Hs[j*na_rows+i] = Complex(0.0,0.0);
	}
      }

      k = 0;
      for (AN=1; AN<=atomnum; AN++){
	GA_AN = order_GA[AN];
	wanA = WhatSpecies[GA_AN];
	tnoA = Spe_Total_CNO[wanA];
	Anum = MP[GA_AN];

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

	  for (i=0; i<tnoA; i++){

	    ig = Anum + i;
	    brow = (ig-1)/nblk;
	    prow = brow%np_rows;

	    for (j=0; j<tnoB; j++){

	      jg = Bnum + j;
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

		if (SCF_iter==1 || all_knum!=1){
		  Cs[(jl-1)*na_rows+il-1].r += S1[k]*co;
		  Cs[(jl-1)*na_rows+il-1].i += S1[k]*si;
		}

		Hs[(jl-1)*na_rows+il-1].r += H1[k]*co;
		Hs[(jl-1)*na_rows+il-1].i += H1[k]*si;
	      }

	      k++;

	    }
	  }
	}
      }

      /* diagonalize S */

      if (measure_time) dtime(&Stime);

      if (parallel_mode==0 || (SCF_iter==1 || all_knum!=1)){
	MPI_Comm_split(MPI_CommWD2[myworld2],my_pcol,my_prow,&mpi_comm_rows);
	MPI_Comm_split(MPI_CommWD2[myworld2],my_prow,my_pcol,&mpi_comm_cols);

	mpi_comm_rows_int = MPI_Comm_c2f(mpi_comm_rows);
	mpi_comm_cols_int = MPI_Comm_c2f(mpi_comm_cols);

	if (scf_eigen_lib_flag==1){

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
#endif
	}

	MPI_Comm_free(&mpi_comm_rows);
	MPI_Comm_free(&mpi_comm_cols);
      }

      if (measure_time){
	dtime(&Etime);
	time2 += Etime - Stime;
      }

      if (SCF_iter==1 || all_knum!=1){

	if (3<=level_stdout){
	  printf(" myid0=%2d spin=%2d kloop %2d  k1 k2 k3 %10.6f %10.6f %10.6f\n",
		 myid0,spin,kloop,T_KGrids1[kloop],T_KGrids2[kloop],T_KGrids3[kloop]);
	  for (i1=1; i1<=n; i1++){
	    printf("  Eigenvalues of OLP  %2d  %15.12f\n",i1,ko[i1]);
	  }
	}

	/* minus eigenvalues to 1.0e-10 */

	for (l=1; l<=n; l++){
	  if (ko[l]<0.0) ko[l] = 1.0e-10;
	  koS[l] = ko[l];
	}

	/* calculate S*1/sqrt(ko) */

	for (l=1; l<=n; l++) ko[l] = 1.0/sqrt(ko[l]);

	/* S * 1.0/sqrt(ko[l]) */

	for(i=0; i<na_rows; i++){
	  for(j=0; j<na_cols; j++){
	    jg = np_cols*nblk*((j)/nblk) + (j)%nblk + ((np_cols+my_pcol)%np_cols)*nblk + 1;
	    Ss[j*na_rows+i].r = Ss[j*na_rows+i].r*ko[jg];
	    Ss[j*na_rows+i].i = Ss[j*na_rows+i].i*ko[jg];
	  }
	}
      }

      /****************************************************
         1.0/sqrt(ko[l]) * U^t * H * U * 1.0/sqrt(ko[l])
      ****************************************************/

      if (measure_time) dtime(&Stime);

      /* pzgemm */

      /* H * U * 1.0/sqrt(ko[l]) */

      for(i=0;i<na_rows_max*na_cols_max;i++){
	Cs[i].r = 0.0;
	Cs[i].i = 0.0;
      }

      Cblacs_barrier(ictxt2,"A");
      F77_NAME(pzgemm,PZGEMM)("N","N",&n,&n,&n,&alpha,Hs,&ONE,&ONE,descH,Ss,&ONE,&ONE,descS,&beta,Cs,&ONE,&ONE,descC);

      /* 1.0/sqrt(ko[l]) * U^+ H * U * 1.0/sqrt(ko[l]) */

      for(i=0;i<na_rows*na_cols;i++){
	Hs[i].r = 0.0;
	Hs[i].i = 0.0;
      }

      Cblacs_barrier(ictxt2,"C");
      F77_NAME(pzgemm,PZGEMM)("C","N",&n,&n,&n,&alpha,Ss,&ONE,&ONE,descS,Cs,&ONE,&ONE,descC,&beta,Hs,&ONE,&ONE,descH);

      if (measure_time){
	dtime(&Etime);
	time3 += Etime - Stime;
      }

      /* diagonalize H' */

      if (measure_time) dtime(&Stime);

      MPI_Comm_split(MPI_CommWD2[myworld2],my_pcol,my_prow,&mpi_comm_rows);
      MPI_Comm_split(MPI_CommWD2[myworld2],my_prow,my_pcol,&mpi_comm_cols);

      mpi_comm_rows_int = MPI_Comm_c2f(mpi_comm_rows);
      mpi_comm_cols_int = MPI_Comm_c2f(mpi_comm_cols);
	
      if (scf_eigen_lib_flag==1){

	F77_NAME(solve_evp_complex,SOLVE_EVP_COMPLEX)
	  ( &n, &MaxN, Hs, &na_rows, &ko[1], Cs, &na_rows, &nblk, &mpi_comm_rows_int, &mpi_comm_cols_int );
      }
      else if (scf_eigen_lib_flag==2){

#ifndef kcomp
	int mpiworld;
	mpiworld = MPI_Comm_c2f(MPI_CommWD2[myworld2]);
	F77_NAME(elpa_solve_evp_complex_2stage_double_impl,ELPA_SOLVE_EVP_COMPLEX_2STAGE_DOUBLE_IMPL)
	  ( &n, &MaxN, Hs, &na_rows, &ko[1], Cs, &na_rows, &nblk, &na_cols,
	    &mpi_comm_rows_int, &mpi_comm_cols_int, &mpiworld );
#endif
      }

      MPI_Comm_free(&mpi_comm_rows);
      MPI_Comm_free(&mpi_comm_cols);

      if (measure_time){
	dtime(&Etime);
	time4 += Etime - Stime;
      }

      for (l=1; l<=MaxN; l++){
	EIGEN[spin][kloop][l] = ko[l];
      }

      if (3<=level_stdout && 0<=kloop){
	printf(" myid0=%2d spin=%2d kloop %i, k1 k2 k3 %10.6f %10.6f %10.6f\n",
	       myid0,spin,kloop,T_KGrids1[kloop],T_KGrids2[kloop],T_KGrids3[kloop]);
	for (i1=1; i1<=n; i1++){
	  if (SpinP_switch==0)
	    printf("  Eigenvalues of Kohn-Sham %2d %15.12f %15.12f\n",
		   i1,EIGEN[0][kloop][i1],EIGEN[0][kloop][i1]);
	  else 
	    printf("  Eigenvalues of Kohn-Sham %2d %15.12f %15.12f\n",
		   i1,EIGEN[0][kloop][i1],EIGEN[1][kloop][i1]);
	}
      }

      /**************************************************
                calculation of wave functions 
      **************************************************/

      if (measure_time) dtime(&Stime);

      for(i=0; i<na_rows*na_cols; i++){
	Hs[i].r = 0.0;
	Hs[i].i = 0.0;
      }

      F77_NAME(pzgemm,PZGEMM)("T","T",&n,&n,&n,&alpha,Cs,&ONE,&ONE,descS,Ss,&ONE,&ONE,descC,&beta,Hs,&ONE,&ONE,descH);
      Cblacs_barrier(ictxt2,"A");

      if (measure_time){
	dtime(&Etime);
	time5 += Etime - Stime;
      }

    } /* kloop0 */

    if (measure_time) dtime(&EiloopTime);

    if (SpinP_switch==1 && numprocs0==1 && spin==0){
      spin++;  
      goto diagonalize1; 
    }

    /****************************************************
     MPI:

     EIGEN
    ****************************************************/

    if (measure_time){
      MPI_Barrier(mpi_comm_level1);
      dtime(&Stime);
    }

    for (spin=0; spin<=SpinP_switch; spin++){
      for (kloop=0; kloop<T_knum; kloop++){

	/* get ID in the zeroth world */
	ID = Comm_World_StartID1[spin] + T_k_ID[spin][kloop];
	MPI_Bcast(&EIGEN[spin][kloop][0], MaxN+1, MPI_DOUBLE, ID, mpi_comm_level1);
      } 
    }

    if (measure_time){
      dtime(&Etime);
      time7 += Etime - Stime;
    }

    /********************************************** 
     calculation of the analytic Berry connection
    **********************************************/ 

    spin = myworld1;
    kloop = S_knum;

    k1 = T_KGrids1[kloop];
    k2 = T_KGrids2[kloop];
    k3 = T_KGrids3[kloop];

    size_H1 = Get_OneD_HS_Col(1, CntOLP,   S1, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);
    size_H1 = Get_OneD_HS_Col(1, nh[spin], H1, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);

    Analytic_Berry_Connection( Cluster_HOMO[spin],
			       spin,kloop,n,MaxN,MP,T_k_op,k1,k2,k3,
			       S1,H1,Hs,Cs,Vs,Qs,EIGEN,
			       ABC,VDMs_ABC,order_GA,all_knum,
			       myworld1,myworld2,MPI_CommWD1,MPI_CommWD2 );

    if (measure_time){
      dtime(&Etime);
      time8 += Etime - Stime;
    }

    dtime(&EiloopTime);

    if (myid0==Host_ID && 0<level_stdout){
      printf("\n<Electric_Polarization_BandCol> time1=%lf\n", EiloopTime-SiloopTime);fflush(stdout);
    }

  } /* end of if (all_knum==1) */

  /******************************************************************
  *******************************************************************
     if (all_knum!=1)
  *******************************************************************
  ******************************************************************/

  dtime(&SiloopTime);

  if (all_knum!=1){

    /* spin=myworld1 */

    spin = myworld1;

  diagonalize2:

    /* set S1 */

    size_H1 = Get_OneD_HS_Col(1, CntOLP, S1, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);

    /* set H1 */

    if (SpinP_switch==0){ 
      size_H1 = Get_OneD_HS_Col(1, nh[0], H1,   MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);
    }
    else if (1<numprocs0){
      size_H1 = Get_OneD_HS_Col(1, nh[0], H1,   MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);
      size_H1 = Get_OneD_HS_Col(1, nh[1], CDM1, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);

      if (myworld1){
	for (i=0; i<size_H1; i++){
	  H1[i] = CDM1[i];
	}
      }
    }
    else{
      size_H1 = Get_OneD_HS_Col(1, nh[spin], H1, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);
    }

    /* for kloop */

    for (kloop0=0; kloop0<num_kloop0; kloop0++){

      kloop = kloop0 + S_knum;

      k1 = T_KGrids1[kloop];
      k2 = T_KGrids2[kloop];
      k3 = T_KGrids3[kloop];

      /* make S and H */

      for(i=0; i<na_rows; i++){
	for(j=0; j<na_cols; j++){
	  Cs[j*na_rows+i] = Complex(0.0,0.0);
	  Hs[j*na_rows+i] = Complex(0.0,0.0);
	}
      }

      k = 0;
      for (AN=1; AN<=atomnum; AN++){
	GA_AN = order_GA[AN];
	wanA = WhatSpecies[GA_AN];
	tnoA = Spe_Total_CNO[wanA];
	Anum = MP[GA_AN];

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

	  for (i=0; i<tnoA; i++){

	    ig = Anum + i;
	    brow = (ig-1)/nblk;
	    prow = brow%np_rows;

	    for (j=0; j<tnoB; j++){

	      jg = Bnum + j;
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

		if (SCF_iter==1 || all_knum!=1){
		  Cs[(jl-1)*na_rows+il-1].r += S1[k]*co;
		  Cs[(jl-1)*na_rows+il-1].i += S1[k]*si;
		}

		Hs[(jl-1)*na_rows+il-1].r += H1[k]*co;
		Hs[(jl-1)*na_rows+il-1].i += H1[k]*si;
	      }

	      k++;

	    }
	  }
	}
      }

      /* diagonalize S */

      if (measure_time) dtime(&Stime);

      MPI_Comm_split(MPI_CommWD2[myworld2],my_pcol,my_prow,&mpi_comm_rows);
      MPI_Comm_split(MPI_CommWD2[myworld2],my_prow,my_pcol,&mpi_comm_cols);

      mpi_comm_rows_int = MPI_Comm_c2f(mpi_comm_rows);
      mpi_comm_cols_int = MPI_Comm_c2f(mpi_comm_cols);

      if (scf_eigen_lib_flag==1){

        F77_NAME(solve_evp_complex,SOLVE_EVP_COMPLEX)
        ( &n, &n, Cs, &na_rows, &ko[1], Ss, &na_rows, &nblk, &mpi_comm_rows_int, &mpi_comm_cols_int);
      }
      else if (scf_eigen_lib_flag==2){

#ifndef kcomp
        int mpiworld;
        mpiworld = MPI_Comm_c2f(MPI_CommWD2[myworld2]);
        F77_NAME(elpa_solve_evp_complex_2stage_double_impl,ELPA_SOLVE_EVP_COMPLEX_2STAGE_DOUBLE_IMPL)
        ( &n, &n, Cs, &na_rows, &ko[1], Ss, &na_rows, &nblk, &na_cols,
          &mpi_comm_rows_int, &mpi_comm_cols_int, &mpiworld);
#endif
      }

      MPI_Comm_free(&mpi_comm_rows);
      MPI_Comm_free(&mpi_comm_cols);

      if (measure_time){
        dtime(&Etime);
        time9 += Etime - Stime;
      }

      if (3<=level_stdout){
	printf(" myid0=%2d kloop %2d  k1 k2 k3 %10.6f %10.6f %10.6f\n",
	       myid0,kloop,T_KGrids1[kloop],T_KGrids2[kloop],T_KGrids3[kloop]);
	for (i1=1; i1<=n; i1++){
	  printf("  Eigenvalues of OLP  %2d  %15.12f\n",i1,ko[i1]);
	}
      }

      /* minus eigenvalues to 1.0e-14 */

      for (l=1; l<=n; l++){
	if (ko[l]<0.0) ko[l] = 1.0e-10;
	koS[l] = ko[l];
      }

      /* calculate S*1/sqrt(ko) */

      for (l=1; l<=n; l++) ko[l] = 1.0/sqrt(ko[l]);

      /* S * 1.0/sqrt(ko[l])  */

      for(i=0;i<na_rows;i++){
	for(j=0;j<na_cols;j++){
	  jg = np_cols*nblk*((j)/nblk) + (j)%nblk + ((np_cols+my_pcol)%np_cols)*nblk + 1;
	  Ss[j*na_rows+i].r = Ss[j*na_rows+i].r*ko[jg];
	  Ss[j*na_rows+i].i = Ss[j*na_rows+i].i*ko[jg];
	}
      }

      /****************************************************
            1/sqrt(ko) * U^t * H * U * 1/sqrt(ko)
      ****************************************************/

      /* pzgemm */

      /* H * U * 1/sqrt(ko) */
    
      for(i=0;i<na_rows_max*na_cols_max;i++){
	Cs[i].r = 0.0;
	Cs[i].i = 0.0;
      }

      Cblacs_barrier(ictxt2,"A");

      F77_NAME(pzgemm,PZGEMM)("N","N",&n,&n,&n,&alpha,Hs,&ONE,&ONE,descH,Ss,&ONE,&ONE,descS,&beta,Cs,&ONE,&ONE,descC);

      /* 1/sqrt(ko) * U^+ H * U * 1/sqrt(ko) */

      for(i=0;i<na_rows*na_cols;i++){
	Hs[i].r = 0.0;
	Hs[i].i = 0.0;
      }

      Cblacs_barrier(ictxt2,"C");

      F77_NAME(pzgemm,PZGEMM)("C","N",&n,&n,&n,&alpha,Ss,&ONE,&ONE,descS,Cs,&ONE,&ONE,descC,&beta,Hs,&ONE,&ONE,descH);

      /* diagonalize H' */

      if (measure_time) dtime(&Stime);

      MPI_Comm_split(MPI_CommWD2[myworld2],my_pcol,my_prow,&mpi_comm_rows);
      MPI_Comm_split(MPI_CommWD2[myworld2],my_prow,my_pcol,&mpi_comm_cols);

      mpi_comm_rows_int = MPI_Comm_c2f(mpi_comm_rows);
      mpi_comm_cols_int = MPI_Comm_c2f(mpi_comm_cols);

      if (scf_eigen_lib_flag==1){
        F77_NAME(solve_evp_complex,SOLVE_EVP_COMPLEX)
        ( &n, &MaxN, Hs, &na_rows, &ko[1], Cs, &na_rows, &nblk, &mpi_comm_rows_int, &mpi_comm_cols_int );
      }
      else if (scf_eigen_lib_flag==2){

#ifndef kcomp
	int mpiworld;
	mpiworld = MPI_Comm_c2f(MPI_CommWD2[myworld2]);
	F77_NAME(elpa_solve_evp_complex_2stage_double_impl,ELPA_SOLVE_EVP_COMPLEX_2STAGE_DOUBLE_IMPL)
	  ( &n, &MaxN, Hs, &na_rows, &ko[1], Cs, &na_rows, &nblk, &na_cols, 
            &mpi_comm_rows_int, &mpi_comm_cols_int, &mpiworld );
#endif
      }

      MPI_Comm_free(&mpi_comm_rows);
      MPI_Comm_free(&mpi_comm_cols);

      if (measure_time){
        dtime(&Etime);
        time10 += Etime - Stime;
      }

      if (3<=level_stdout && 0<=kloop){
	printf("  kloop %i, k1 k2 k3 %10.6f %10.6f %10.6f\n",
	       kloop,T_KGrids1[kloop],T_KGrids2[kloop],T_KGrids3[kloop]);
	for (i1=1; i1<=n; i1++){
	  printf("  Eigenvalues of Kohn-Sham(DM) spin=%2d i1=%2d %15.12f\n",
		 spin,i1,ko[i1]);
	}
      }

      /****************************************************
        transformation to the original eigenvectors.
	     NOTE JRCAT-244p and JAIST-2122p 
      ****************************************************/

      for(i=0;i<na_rows*na_cols;i++){
	Hs[i].r = 0.0;
	Hs[i].i = 0.0;
      }

      F77_NAME(pzgemm,PZGEMM)( "T","T",&n,&n,&n,&alpha,Cs,&ONE,&ONE,descS,Ss,
                               &ONE,&ONE,descC,&beta,Hs,&ONE,&ONE,descH );
      Cblacs_barrier(ictxt2,"A");

      /***************************************** 
           calculate the Berry connection
      *****************************************/ 

      /* calculation of the analytic Berry connection */

      Analytic_Berry_Connection( Cluster_HOMO[spin],
				 spin,kloop,n,MaxN,MP,T_k_op,k1,k2,k3,
				 S1,H1,Hs,Cs,Vs,Qs,EIGEN,
				 ABC,VDMs_ABC,order_GA,all_knum,
				 myworld1,myworld2,MPI_CommWD1,MPI_CommWD2 );

    } /* kloop0 */

    if (measure_time){
      dtime(&Etime);
      time12 += Etime - Stime;
    }

    if (SpinP_switch==1 && numprocs0==1 && spin==0){
      spin++;  
      goto diagonalize2; 
    }

    dtime(&EiloopTime);

    if (myid0==Host_ID && 0<level_stdout){
      printf("\n<Electric_Polarization_BandCol> time2=%lf\n", EiloopTime-SiloopTime);fflush(stdout);
    }

  } /* if (all_knum!=1) */

  /****************************************************
          electric polarization by Berry phase
  ****************************************************/

  for (spin=0; spin<=SpinP_switch; spin++){

    tmparray[0] = ABC[spin][0].i; tmparray[1] = ABC[spin][1].i; tmparray[2] = ABC[spin][2].i;  
    MPI_Allreduce(MPI_IN_PLACE, &tmparray[0], 3, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
    for (i=0; i<3; i++){ ABC[spin][i].i = tmparray[i]/sum_weights; }

    /* taking account of the modulo 2pi */

    for (i=1; i<=3; i++){

      tmp = (ABC[spin][0].i*rtv[i][1] + ABC[spin][1].i*rtv[i][2] + ABC[spin][2].i*rtv[i][3])/(2.0*PI);

      do{
	tmp += 1.0;
      } while (tmp<0.0);

      /* tmp is adjusted in between 0.0 and 1.0. */
      j = (int)tmp;
      be[i] = tmp - (double)j;
    }

    ABC[spin][0].i = be[1]*tv[1][1] + be[2]*tv[2][1] + be[3]*tv[3][1];
    ABC[spin][1].i = be[1]*tv[1][2] + be[2]*tv[2][2] + be[3]*tv[3][2];
    ABC[spin][2].i = be[1]*tv[1][3] + be[2]*tv[2][3] + be[3]*tv[3][3];

  } // spin

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

  if (SpinP_switch==0){
    E_dpx = 2.0*(ABC[0][0].i - EPol[0][0]);
    E_dpy = 2.0*(ABC[0][1].i - EPol[0][1]);
    E_dpz = 2.0*(ABC[0][2].i - EPol[0][2]);
  }
  else{
    E_dpx = ABC[0][0].i + ABC[1][0].i - EPol[0][0] - EPol[1][0];
    E_dpy = ABC[0][1].i + ABC[1][1].i - EPol[0][1] - EPol[1][1];
    E_dpz = ABC[0][2].i + ABC[1][2].i - EPol[0][2] - EPol[1][2];
  }

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
      printf("         Dipole moment (Debye) by the Berry phase        \n"); fflush(stdout);
      printf("*******************************************************\n\n"); fflush(stdout);

      printf(" Absolute D %17.8f\n\n",AbsD);
      printf("                      Dx                Dy                Dz\n"); fflush(stdout);
      printf(" Total       %17.8f %17.8f %17.8f\n",
	     dipole_moment[0][1],dipole_moment[0][2],dipole_moment[0][3]);fflush(stdout);
      printf(" Core        %17.8f %17.8f %17.8f\n",
	     dipole_moment[1][1],dipole_moment[1][2],dipole_moment[1][3]);fflush(stdout);
      printf(" Electron    %17.8f %17.8f %17.8f\n",
	     dipole_moment[2][1],dipole_moment[2][2],dipole_moment[2][3]);fflush(stdout);

      if (SpinP_switch==0){
        printf("   up spin   %17.8f %17.8f %17.8f\n",
	       AU2Debye*(ABC[0][0].i - EPol[0][0]),
	       AU2Debye*(ABC[0][1].i - EPol[0][1]),
	       AU2Debye*(ABC[0][2].i - EPol[0][2]));fflush(stdout);

        printf("   down spin %17.8f %17.8f %17.8f\n",
	       AU2Debye*(ABC[0][0].i - EPol[0][0]),
	       AU2Debye*(ABC[0][1].i - EPol[0][1]),
	       AU2Debye*(ABC[0][2].i - EPol[0][2]));fflush(stdout);
      }
      else{
        printf("   up spin   %17.8f %17.8f %17.8f\n",
	       AU2Debye*(ABC[0][0].i - EPol[0][0]),
	       AU2Debye*(ABC[0][1].i - EPol[0][1]),
	       AU2Debye*(ABC[0][2].i - EPol[0][2]));fflush(stdout);

        printf("   down spin %17.8f %17.8f %17.8f\n",
	       AU2Debye*(ABC[1][0].i - EPol[1][0]),
	       AU2Debye*(ABC[1][1].i - EPol[1][1]),
	       AU2Debye*(ABC[1][2].i - EPol[1][2]));fflush(stdout);
      }

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

      if (SpinP_switch==0){
        fprintf(fp_DPM,"   up spin   %17.8f %17.8f %17.8f\n",
	        AU2Debye*(ABC[0][0].i - EPol[0][0]),
	        AU2Debye*(ABC[0][1].i - EPol[0][1]),
	        AU2Debye*(ABC[0][2].i - EPol[0][2]));fflush(stdout);

        fprintf(fp_DPM,"   down spin %17.8f %17.8f %17.8f\n",
	        AU2Debye*(ABC[0][0].i - EPol[0][0]),
	        AU2Debye*(ABC[0][1].i - EPol[0][1]),
	        AU2Debye*(ABC[0][2].i - EPol[0][2]));fflush(stdout);
      }
      else{
        fprintf(fp_DPM,"   up spin   %17.8f %17.8f %17.8f\n",
	        AU2Debye*(ABC[0][0].i - EPol[0][0]),
	        AU2Debye*(ABC[0][1].i - EPol[0][1]),
	        AU2Debye*(ABC[0][2].i - EPol[0][2]));fflush(stdout);

        fprintf(fp_DPM,"   down spin %17.8f %17.8f %17.8f\n",
	        AU2Debye*(ABC[1][0].i - EPol[1][0]),
	        AU2Debye*(ABC[1][1].i - EPol[1][1]),
	        AU2Debye*(ABC[1][2].i - EPol[1][2]));fflush(stdout);
      }

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

  free(SP_Atoms);
  free(SP_NZeros);
  free(My_NZeros);

  for (spin=0; spin<2; spin++){
    free(ABC[spin]);
  }
  free(ABC);

  free(Vs);
  free(Qs);

  for (spin=0; spin<2; spin++){
    free(DMs[spin]);
  }
  free(DMs);

  free(VDMs);

  for (spin=0; spin<=SpinP_switch; spin++){
    free(VDMs_ABC[spin]);
  }
  free(VDMs_ABC);

  for (spin=0; spin<=SpinP_switch; spin++){
    Cfree_blacs_system_handle(bhandle11[spin]);
    Cblacs_gridexit(ictxt11[spin]);
    Cfree_blacs_system_handle(bhandle12[spin]);
    Cblacs_gridexit(ictxt12[spin]);
  }

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
    printf("myid0=%2d time81=%9.4f\n",myid0,time81);fflush(stdout);
    printf("myid0=%2d time82=%9.4f\n",myid0,time82);fflush(stdout);
    printf("myid0=%2d time83=%9.4f\n",myid0,time83);fflush(stdout);
    printf("myid0=%2d time84=%9.4f\n",myid0,time84);fflush(stdout);
    printf("myid0=%2d time85=%9.4f\n",myid0,time85);fflush(stdout);
    printf("myid0=%2d time9 =%9.4f\n",myid0,time9);fflush(stdout);
    printf("myid0=%2d time10=%9.4f\n",myid0,time10);fflush(stdout);
    printf("myid0=%2d time11=%9.4f\n",myid0,time11);fflush(stdout);
    printf("myid0=%2d time11A=%9.4f\n",myid0,time11A);fflush(stdout);
    printf("myid0=%2d time11B=%9.4f\n",myid0,time11B);fflush(stdout);
    printf("myid0=%2d time12=%9.4f\n",myid0,time12);fflush(stdout);
  }

  MPI_Barrier(mpi_comm_level1);
  dtime(&TEtime);
  time0 = TEtime - TStime;
  return time0;
}







void Analytic_Berry_Connection( int Nocc,
				int spin, int kloop, int n, int MaxN, 
				int *MP, int *T_k_op,
				double k1, double k2, double k3,
				double *S1, double *H1,
				dcomplex *Hs, dcomplex *Cs, 
				dcomplex *Vs, dcomplex *Qs, 
				double ***EIGEN,
				dcomplex **ABC,
				dcomplex **VDMs_ABC,
				int *order_GA, 
				int all_knum, 
				int myworld1,
				int myworld2,
				MPI_Comm *MPI_CommWD1,
				MPI_Comm *MPI_CommWD2 )
{
  int LB_AN,i,j,i0,i1,l,m,GB_AN,Rn,wanB,tnoB,Bnum,mu,nu;
  int target_Atom,l1,l2,l3,po,p,q,g,LWORK,n1,n2,ia,ja,ib,jb,lwork;
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

  n1 = Nocc;
  n2 = n - Nocc;
  A11s = (dcomplex*)malloc(sizeof(dcomplex)*n1*n1);
  A12s = (dcomplex*)malloc(sizeof(dcomplex)*n1*n2);
  B12s = (dcomplex*)malloc(sizeof(dcomplex)*n1*n2);
  C12s = (dcomplex*)malloc(sizeof(dcomplex)*n1*n2);
  sv = (double*)malloc(sizeof(double)*n1);
  rwork = (double*)malloc(sizeof(double)*(1+4*n1));

  /* get size of work and allocate work */

  lwork = -1;
  F77_NAME(pzgesvd,PZGESVD)( "V", "V",
                             &n1, &n1,
			     A11s, &ONE, &ONE, desc11[spin],
			     sv,
			     Qs, &ONE, &ONE, desc11[spin],
			     Vs, &ONE, &ONE, desc11[spin],
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

  Cblacs_barrier(ictxt2,"A");

  F77_NAME(pzgemm,PZGEMM)( "T","T",&n,&n,&n,&alpha,VDMs_ABC[spin],&ONE,&ONE,descH,Hs,
			   &ONE,&ONE,descS,&beta,Cs,&ONE,&ONE,descC );

  /* redistribution of A to construct A11s */

  ia = 1; ja = 1; ib = 1; jb = 1;
  F77_NAME(pzgemr2d,PDGEMR2D)(&n1, &n1, Cs, &ia, &ja, descC, A11s, &ib, &jb, desc11[spin], &ictxt11[spin]);

  /* redistribution of A to construct A12s */

  ia = 1; ja = n1+1; ib = 1; jb = 1;
  F77_NAME(pzgemr2d,PDGEMR2D)(&n1, &n2, Cs, &ia, &ja, descC, A12s, &ib, &jb, desc12[spin], &ictxt12[spin]);

  /* singular value decomposition for A11 -> U Sigma V^dag */

  F77_NAME(pzgesvd,PZGESVD)( "V", "V",
			     &n1, &n1,
			     A11s, &ONE, &ONE, desc11[spin],
			     sv,
			     Qs, &ONE, &ONE, desc11[spin],
			     Vs, &ONE, &ONE, desc11[spin],
			     work,
			     &lwork,
                             rwork,
			     &info);

  /*
  for (i=0; i<n1; i++){
    printf("VVV1 i=%d sv=%15.12f\n",i,sv[i]);
  }
  
  MPI_Finalize();
  exit(0);
  */

  /* calculate V Sigma^-1 U^dag -> A11s */

  for (j=0; j<na_cols11[spin]; j++){

    jg = np_cols11[spin]*nblk11[spin]*((j)/nblk11[spin]) + (j)%nblk11[spin] 
         + ((np_cols11[spin]+my_pcol11[spin])%np_cols11[spin])*nblk11[spin];

    for (i=0; i<na_rows11[spin]; i++){
      Qs[j*na_rows11[spin]+i].r /= sv[jg]; 
      Qs[j*na_rows11[spin]+i].i /= sv[jg]; 
    }
  }

  Cblacs_barrier(ictxt11[spin],"A");
  F77_NAME(pzgemm,PZGEMM)( "C","C",
			   &n1, &n1, &n1, 
			   &alpha,
			   Vs,&ONE,&ONE,desc11[spin],
			   Qs,&ONE,&ONE,desc11[spin],
			   &beta,
			   A11s,&ONE,&ONE,desc11[spin] );

  /* calculate A11s x A12s -> B12s */

  Cblacs_barrier(ictxt12[spin],"A");
  F77_NAME(pzgemm,PZGEMM)( "N","N",
			   &n1, &n2, &n1, 
			   &alpha,
			   A11s,&ONE,&ONE,desc11[spin],
			   A12s,&ONE,&ONE,desc12[spin],
			   &beta,
			   B12s,&ONE,&ONE,desc12[spin] );

  /************************************
         loop for x, y, and z
  ************************************/

  for (p=1; p<=3; p++){

    /********************************************
              calculations of S' and H'
    ********************************************/

    /* calculate S' and H' */

    for(i=0; i<na_rows; i++){
      for(j=0; j<na_cols; j++){
	Qs[j*na_rows+i] = Complex(0.0,0.0); // S'
	Vs[j*na_rows+i] = Complex(0.0,0.0); // H'
      }
    }

    k = 0;
    for (AN=1; AN<=atomnum; AN++){
      GA_AN = order_GA[AN];
      wanA = WhatSpecies[GA_AN];
      tnoA = Spe_Total_CNO[wanA];
      Anum = MP[GA_AN];

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

	for (i=0; i<tnoA; i++){

	  ig = Anum + i;
	  brow = (ig-1)/nblk;
	  prow = brow%np_rows;

	  for (j=0; j<tnoB; j++){

	    jg = Bnum + j;
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

	      Qs[(jl-1)*na_rows+il-1].r +=-S1[k]*si*atv[Rn][p];
	      Qs[(jl-1)*na_rows+il-1].i += S1[k]*co*atv[Rn][p];

	      Vs[(jl-1)*na_rows+il-1].r +=-H1[k]*si*atv[Rn][p];
	      Vs[(jl-1)*na_rows+il-1].i += H1[k]*co*atv[Rn][p];

	    }

	    k++;

	  } // j
	} // i
      } // LB_AN

    } // end for the loop of AN

    /* calculate <c_i|S'|c_j> */

    for(i=0; i<na_rows*na_cols; i++){
      Cs[i].r = 0.0;
      Cs[i].i = 0.0;
    }

    Cblacs_barrier(ictxt2,"A");

    F77_NAME(pzgemm,PZGEMM)( "N","T",&n,&n,&n,&alpha,Qs,&ONE,&ONE,descH,Hs,
			     &ONE,&ONE,descS,&beta,Cs,&ONE,&ONE,descC );

    for(i=0; i<na_rows*na_cols; i++){
      Qs[i].r = 0.0;
      Qs[i].i = 0.0;
    }

    Cblacs_barrier(ictxt2,"A");

    for (i=0; i<na_rows*na_cols; i++){ Hs[i].i = -Hs[i].i; }

    F77_NAME(pzgemm,PZGEMM)( "N","N",&n,&n,&n,&alpha,Hs,&ONE,&ONE,descS,Cs,
			     &ONE,&ONE,descC,&beta,Qs,&ONE,&ONE,descH );

    for (i=0; i<na_rows*na_cols; i++){ Hs[i].i = -Hs[i].i; }

    /* calculate <c_i|H'|c_j> */

    for(i=0; i<na_rows*na_cols; i++){
      Cs[i].r = 0.0;
      Cs[i].i = 0.0;
    }
    
    Cblacs_barrier(ictxt2,"A");

    F77_NAME(pzgemm,PZGEMM)( "N","T",&n,&n,&n,&alpha,Vs,&ONE,&ONE,descH,Hs,
			     &ONE,&ONE,descS,&beta,Cs,&ONE,&ONE,descC );

    for(i=0; i<na_rows*na_cols; i++){
      Vs[i].r = 0.0;
      Vs[i].i = 0.0;
    }

    Cblacs_barrier(ictxt2,"A");

    for (i=0; i<na_rows*na_cols; i++){ Hs[i].i = -Hs[i].i; }

    F77_NAME(pzgemm,PZGEMM)( "N","N",&n,&n,&n,&alpha,Hs,&ONE,&ONE,descS,Cs,
			     &ONE,&ONE,descC,&beta,Vs,&ONE,&ONE,descH );

    for (i=0; i<na_rows*na_cols; i++){ Hs[i].i = -Hs[i].i; }

    /*
    printf("Vs=%15.12f %15.12f\n",Vs[0].r,Vs[0].i);
    MPI_Finalize();
    exit(0);
    */

    /**********************************************************
                  calculate the Berry connection 
    **********************************************************/

    /* redistribution of Qs to construct A12s */

    ia = 1; ja = n1+1; ib = 1; jb = 1;
    F77_NAME(pzgemr2d,PDGEMR2D)(&n1, &n2, Qs, &ia, &ja, descC, A12s, &ib, &jb, desc12[spin], &ictxt12[spin]);

    /* redistribution of Vs to construct C12s which is used as temporal array */

    ia = 1; ja = n1+1; ib = 1; jb = 1;
    F77_NAME(pzgemr2d,PDGEMR2D)( &n1, &n2, Vs, &ia, &ja, descC, C12s, 
                                 &ib, &jb, desc12[spin], &ictxt12[spin]);

    /* calculate C12s */    

    for (i=0; i<na_rows12[spin]; i++){

      ig = np_rows12[spin]*nblk12[spin]*((i)/nblk12[spin]) + (i)%nblk12[spin] 
           + ((np_rows12[spin]+my_prow12[spin])%np_rows12[spin])*nblk12[spin] + 1;

      ei = EIGEN[spin][kloop][ig]; 

      for (j=0; j<na_cols12[spin]; j++){

        jg = np_cols12[spin]*nblk12[spin]*((j)/nblk12[spin]) + (j)%nblk12[spin]
             + ((np_cols12[spin]+my_pcol12[spin])%np_cols12[spin])*nblk12[spin] + n1 + 1;

        ej = EIGEN[spin][kloop][jg]; 
        d = ei - ej;
        tmp1 = d/(d*d + del);

	C12s[j*na_rows12[spin]+i].r = (C12s[j*na_rows12[spin]+i].r - ei*A12s[j*na_rows12[spin]+i].r)*tmp1;
	C12s[j*na_rows12[spin]+i].i = (C12s[j*na_rows12[spin]+i].i - ei*A12s[j*na_rows12[spin]+i].i)*tmp1;

      }
    }

    /* calculate B12s x C12s^dag -> A11s */    

    Cblacs_barrier(ictxt12[spin],"A");
    F77_NAME(pzgemm,PZGEMM)( "N","C",
			     &n1, &n1, &n2, 
			     &alpha,
			     B12s,&ONE,&ONE,desc12[spin],
			     C12s,&ONE,&ONE,desc12[spin],
			     &beta,
			     A11s,&ONE,&ONE,desc11[spin] );

    /* calculate Tr[B12s x C12s^dag -> A11s] */

    asum1[0] = 0.0; asum1[1] = 0.0; 
    for (i=0; i<na_rows11[spin]; i++){

      ig = np_rows11[spin]*nblk11[spin]*((i)/nblk11[spin]) + (i)%nblk11[spin] 
	+ ((np_rows11[spin]+my_prow11[spin])%np_rows11[spin])*nblk11[spin];

      for (j=0; j<na_cols11[spin]; j++){

	jg = np_cols11[spin]*nblk11[spin]*((j)/nblk11[spin]) + (j)%nblk11[spin] 
	  + ((np_cols11[spin]+my_pcol11[spin])%np_cols11[spin])*nblk11[spin];

	if (ig==jg){
	  asum1[0] += A11s[j*na_rows11[spin]+i].r;
	  asum1[1] += A11s[j*na_rows11[spin]+i].i;
	} 
      }
    }

    /* Add the Berry connection to ABC.  
       MPI_Allreduce will be performed to gather all the contributions at end. */
   
    ABC[spin][p-1].r -= (double)T_k_op[kloop]*asum1[0];
    ABC[spin][p-1].i -= (double)T_k_op[kloop]*asum1[1];

    //printf("VVV spin=%2d kloop=%2d asum1=%15.12f %15.12f\n",spin,kloop,asum1[0],asum1[1]);
    //if (p==1) printf("%15.12f %15.12f\n",k1,asum1[1]);

  } // p 

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
            double EPol[2][3], double CPol[3], double EPol_BG[3] )
{
  int i,j,tno0,tno1,spin,Mc_AN,Gc_AN,h_AN,Gh_AN,Cwan,Hwan,Rn,l1,l2,l3;
  double My_EPol[2][3];
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

  for (spin=0; spin<=SpinP_switch; spin++){
    My_EPol[spin][0] = 0.0; My_EPol[spin][1] = 0.0; My_EPol[spin][2] = 0.0;
  }

  for (spin=0; spin<=SpinP_switch; spin++){
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

            matx = OLPpo[0][0][Mc_AN][h_AN][i][j] - (Rx-Gxyz[Gc_AN][1])*OLP0[Mc_AN][h_AN][i][j];
            maty = OLPpo[1][0][Mc_AN][h_AN][i][j] - (Ry-Gxyz[Gc_AN][2])*OLP0[Mc_AN][h_AN][i][j];
            matz = OLPpo[2][0][Mc_AN][h_AN][i][j] - (Rz-Gxyz[Gc_AN][3])*OLP0[Mc_AN][h_AN][i][j];

	    My_EPol[spin][0] += CDM[spin][Mc_AN][h_AN][i][j]*matx;
	    My_EPol[spin][1] += CDM[spin][Mc_AN][h_AN][i][j]*maty;
	    My_EPol[spin][2] += CDM[spin][Mc_AN][h_AN][i][j]*matz;
	  }
	}

      } // h_AN
    } // Mc_AN

  } // spin  

  MPI_Barrier(mpi_comm_level1);
  for (spin=0; spin<=SpinP_switch; spin++){
    MPI_Allreduce(&My_EPol[spin][0], &EPol[spin][0], 3, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
  }

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




void Calc_OLPpo( double ******OLPpo )
{
  double time0;
  int Mc_AN,Gc_AN,Mh_AN,h_AN,Gh_AN;
  int i,j,Cwan,Hwan,NO0,NO1,spinmax;
  int Rnh,Rnk,spin,N;
  int n1,n2,n3,L0,Mul0,M0,L1,Mul1,M1;
  int Nc,GNc,GRc,Nog,Nh,MN,XC_P_switch;
  double x,y,z,dx,dy,dz,tmpx,tmpy,tmpz;
  double bc,dv,r,theta,phi,sum,tmp0,tmp1;
  double Cxyz[4];
  double **ChiVx;
  double **ChiVy;
  double **ChiVz;
  double *tmp_ChiVx;
  double *tmp_ChiVy;
  double *tmp_ChiVz;
  double *tmp_Orbs_Grid;
  double **tmp_OLPpox;
  double **tmp_OLPpoy;
  double **tmp_OLPpoz;
  double TStime,TEtime;
  double TStime0,TEtime0;
  double TStime1,TEtime1;
  double TStime2,TEtime2;
  double TStime3,TEtime3;
  int numprocs,myid,tag=999,ID;
  double Stime_atom, Etime_atom;

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);
  MPI_Barrier(mpi_comm_level1);

  /****************************************************
    allocation of arrays:
  ****************************************************/

  ChiVx = (double**)malloc(sizeof(double*)*List_YOUSO[7]);
  for (i=0; i<List_YOUSO[7]; i++){
    ChiVx[i] = (double*)malloc(sizeof(double)*List_YOUSO[11]);
  }

  ChiVy = (double**)malloc(sizeof(double*)*List_YOUSO[7]);
  for (i=0; i<List_YOUSO[7]; i++){
    ChiVy[i] = (double*)malloc(sizeof(double)*List_YOUSO[11]);
  }

  ChiVz = (double**)malloc(sizeof(double*)*List_YOUSO[7]);
  for (i=0; i<List_YOUSO[7]; i++){
    ChiVz[i] = (double*)malloc(sizeof(double)*List_YOUSO[11]);
  }

  tmp_ChiVx = (double*)malloc(sizeof(double)*List_YOUSO[7]);
  tmp_ChiVy = (double*)malloc(sizeof(double)*List_YOUSO[7]);
  tmp_ChiVz = (double*)malloc(sizeof(double)*List_YOUSO[7]);

  tmp_Orbs_Grid = (double*)malloc(sizeof(double)*List_YOUSO[7]);

  tmp_OLPpox = (double**)malloc(sizeof(double*)*List_YOUSO[7]);
  for (i=0; i<List_YOUSO[7]; i++){
    tmp_OLPpox[i] = (double*)malloc(sizeof(double)*List_YOUSO[7]);
  }

  tmp_OLPpoy = (double**)malloc(sizeof(double*)*List_YOUSO[7]);
  for (i=0; i<List_YOUSO[7]; i++){
    tmp_OLPpoy[i] = (double*)malloc(sizeof(double)*List_YOUSO[7]);
  }

  tmp_OLPpoz = (double**)malloc(sizeof(double*)*List_YOUSO[7]);
  for (i=0; i<List_YOUSO[7]; i++){
    tmp_OLPpoz[i] = (double*)malloc(sizeof(double)*List_YOUSO[7]);
  }

  /*****************************************************
     matrix elements for OLPpo (added by N. Yamaguchi)
  *****************************************************/

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

    dtime(&Stime_atom);

    Gc_AN = M2G[Mc_AN];
    Cwan = WhatSpecies[Gc_AN];
    NO0 = Spe_Total_CNO[Cwan];

    for (i=0; i<NO0; i++){
      for (Nc=0; Nc<GridN_Atom[Gc_AN]; Nc++){

	GNc = GridListAtom[Mc_AN][Nc];
	GRc = CellListAtom[Mc_AN][Nc];

	Get_Grid_XYZ(GNc,Cxyz);
	x = Cxyz[1] + atv[GRc][1] - Gxyz[Gc_AN][1];
	y = Cxyz[2] + atv[GRc][2] - Gxyz[Gc_AN][2];
	z = Cxyz[3] + atv[GRc][3] - Gxyz[Gc_AN][3];

	ChiVx[i][Nc] = x*Orbs_Grid[Mc_AN][Nc][i];
	ChiVy[i][Nc] = y*Orbs_Grid[Mc_AN][Nc][i];
	ChiVz[i][Nc] = z*Orbs_Grid[Mc_AN][Nc][i];

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
	  tmp_OLPpox[i][j] = 0.0;
	  tmp_OLPpoy[i][j] = 0.0;
	  tmp_OLPpoz[i][j] = 0.0;
	}
      }

      /* summation of non-zero elements */

      for (Nog=0; Nog<NumOLG[Mc_AN][h_AN]; Nog++){

	Nc = GListTAtoms1[Mc_AN][h_AN][Nog];
	Nh = GListTAtoms2[Mc_AN][h_AN][Nog];

	/* store ChiVx,y,z in tmp_ChiVx,y,z */

	for (i=0; i<NO0; i++){
	  tmp_ChiVx[i] = ChiVx[i][Nc];
	  tmp_ChiVy[i] = ChiVy[i][Nc];
	  tmp_ChiVz[i] = ChiVz[i][Nc];
	}

	/* store Orbs_Grid in tmp_Orbs_Grid */

	if (G2ID[Gh_AN]==myid){
	  for (j=0; j<NO1; j++){
	    tmp_Orbs_Grid[j] = Orbs_Grid[Mh_AN][Nh][j];/* AITUNE */
	  }
	}
	else{
	  for (j=0; j<NO1; j++){
	    tmp_Orbs_Grid[j] = Orbs_Grid_FNAN[Mc_AN][h_AN][Nog][j];/* AITUNE */
	  }
	}

	/* integration */

	for (i=0; i<NO0; i++){
	  tmpx = tmp_ChiVx[i];
	  tmpy = tmp_ChiVy[i];
	  tmpz = tmp_ChiVz[i];
	  for (j=0; j<NO1; j++){
	    tmp_OLPpox[i][j] += tmpx*tmp_Orbs_Grid[j];
	    tmp_OLPpoy[i][j] += tmpy*tmp_Orbs_Grid[j];
	    tmp_OLPpoz[i][j] += tmpz*tmp_Orbs_Grid[j];
	  }
	}
      }

      /* OLPpox, OLPpoy, OLPpoz */

      for (i=0; i<NO0; i++){
	for (j=0; j<NO1; j++){
	  OLPpo[0][0][Mc_AN][h_AN][i][j] = tmp_OLPpox[i][j]*GridVol;
	  OLPpo[1][0][Mc_AN][h_AN][i][j] = tmp_OLPpoy[i][j]*GridVol;
	  OLPpo[2][0][Mc_AN][h_AN][i][j] = tmp_OLPpoz[i][j]*GridVol;
	}
      }

    } // h_AN

    dtime(&Etime_atom);
    time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

  } // Mc_AN

  /****************************************************
    freeing of arrays:
  ****************************************************/

  for (i=0; i<List_YOUSO[7]; i++){
    free(ChiVx[i]);
  }
  free(ChiVx);

  for (i=0; i<List_YOUSO[7]; i++){
    free(ChiVy[i]);
  }
  free(ChiVy);

  for (i=0; i<List_YOUSO[7]; i++){
    free(ChiVz[i]);
  }
  free(ChiVz);

  free(tmp_ChiVx);
  free(tmp_ChiVy);
  free(tmp_ChiVz);

  free(tmp_Orbs_Grid);

  for (i=0; i<List_YOUSO[7]; i++){
    free(tmp_OLPpox[i]);
  }
  free(tmp_OLPpox);

  for (i=0; i<List_YOUSO[7]; i++){
    free(tmp_OLPpoy[i]);
  }
  free(tmp_OLPpoy);

  for (i=0; i<List_YOUSO[7]; i++){
    free(tmp_OLPpoz[i]);
  }
  free(tmp_OLPpoz);
}


void Allocate_Free_Electric_Polarization(int todo_flag)
{
  static int firsttime=1;
  int tno0,tno1,i,j,Cwan,Hwan,h_AN,Gh_AN,Mc_AN,Gc_AN;
  int size_OLPpo=0;
  int direction,order,order_max;

  MPI_Barrier(mpi_comm_level1);

  order_max = 1;

  /********************************************
              allocation of arrays 
  ********************************************/

  if (todo_flag==1){

    OLPpo=(double******)malloc(sizeof(double*****)*3);
    for (direction=0; direction<3; direction++){
      OLPpo[direction]=(double*****)malloc(sizeof(double****)*order_max);
      for (order=0; order<order_max; order++){
	OLPpo[direction][order]=(double****)malloc(sizeof(double***)*(Matomnum+1));
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

	  OLPpo[direction][order][Mc_AN] = (double***)malloc(sizeof(double**)*(FNAN[Gc_AN]+1));
	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	    if (Mc_AN==0){
	      tno1 = 1;
	    }
	    else{
	      Gh_AN = natn[Gc_AN][h_AN];
	      Hwan = WhatSpecies[Gh_AN];
	      tno1 = Spe_Total_CNO[Hwan];
	    }

	    OLPpo[direction][order][Mc_AN][h_AN] = (double**)malloc(sizeof(double*)*tno0);
	    for (i=0; i<tno0; i++){
	      OLPpo[direction][order][Mc_AN][h_AN][i] = (double*)malloc(sizeof(double)*tno1);
	    }

            size_OLPpo += tno0*tno1; 
	  }
	}
      }
    }

    /* PrintMemory */

    if (firsttime && memoryusage_fileout) {
      PrintMemory("Allocate_Free_Electric_Polarization: OLPpo",sizeof(double)*size_OLPpo,NULL);
    }
    firsttime = 0;

  } /* end of if (todo_flag==1) */

  /********************************************
               freeing of arrays 
  ********************************************/

  if (todo_flag==2){

    for (direction=0; direction<3; direction++){
      for (order=0; order<order_max; order++){
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
	      free(OLPpo[direction][order][Mc_AN][h_AN][i]);
	    }
	    free(OLPpo[direction][order][Mc_AN][h_AN]);
	  }
	  free(OLPpo[direction][order][Mc_AN]);
	}
	free(OLPpo[direction][order]);
      }
      free(OLPpo[direction]);
    }
    free(OLPpo);
    OLPpo=NULL;

  } /* end of if (todo_flag==2) */

}

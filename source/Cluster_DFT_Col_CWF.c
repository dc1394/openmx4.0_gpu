/**********************************************************************************
  Cluster_DFT_Col_CWF.c:

     Cluster_DFT_Col_CWF.c is a subroutine to calculate closest Wannier 
     functions to a given set of orbials in the cluster calculation 
     based on the collinear DFT.

  Log of Cluster_DFT_Col_CWF.c:

     29/May/2023  Released by T. Ozaki

**********************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/stat.h>
#include "mpi.h"
#include "openmx_common.h"
#include "openmx_cusolver_dense_utils.h"
#include "lapack_prototypes.h"
#include "set_cuda_default_device_from_local_rank.h"
#include "set_openacc_device_from_local_rank.h"
#include <omp.h>
#include <openacc.h>

#define  measure_time   0

double DM_func;

/* GPU CuSolver context for Cluster_DFT_Col_CWF (added by H.Kawai, Phase D) */
typedef struct {
    int                device_id;
    cublasHandle_t     cublas;
    cusolverDnHandle_t cusolver;
} ClusterColCWFCuSolverCtx;

static ClusterColCWFCuSolverCtx ClusterColCWF_cusolver_ctx = {0};

static void ClusterColCWF_CuSolver_Destroy(void)
{
    ClusterColCWFCuSolverCtx *ctx = &ClusterColCWF_cusolver_ctx;
    if (ctx->cusolver != NULL) wait_cudafunc(cusolverDnDestroy(ctx->cusolver));
    if (ctx->cublas != NULL)   wait_cudafunc(cublasDestroy(ctx->cublas));
    memset(ctx, 0, sizeof(*ctx));
    ctx->device_id = -1;
}

static void ClusterColCWF_CuSolver_Init(void)
{
    ClusterColCWFCuSolverCtx *ctx = &ClusterColCWF_cusolver_ctx;
    int current_device;
    wait_cudafunc(cudaGetDevice(&current_device));
    if (ctx->device_id == current_device && ctx->cublas != NULL) return;
    if (ctx->cublas != NULL) ClusterColCWF_CuSolver_Destroy();
    ctx->device_id = current_device;
    wait_cudafunc(cublasCreate(&ctx->cublas));
    wait_cudafunc(cusolverDnCreate(&ctx->cusolver));
}

static void ClusterColCWF_GEMMul8Dgemm_OpenACC(cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
                                               double const * A, double const * B, double * C)
{
    ClusterColCWF_CuSolver_Init();
#pragma acc data      present(A[0 : m * k], B[0 : k * n], C[0 : m * n])
#pragma acc host_data use_device(A, B, C)
    {
        double const alpha = 1.0;
        double const beta  = 0.0;
        wait_cudafunc(openmx_gemmul8Dgemm(ClusterColCWF_cusolver_ctx.cublas, transa, transb, m, n, k, &alpha, A, m, B, k, &beta, C, m));
    }
}

void AllocateArrays_Col_QAO();
void FreeArrays_Col_QAO();
double Calc_Hybrid_AO_Col( double ****OLP0, double *****Hks, double *****CDM );
double Calc_MO_in_Bulk_Col( double ****OLP0, double *****Hks, double *****CDM );

void Save_EVec_CWF2W90( char *mode, char *fname, 
                        double *RealEVec, dcomplex *CompEVec, 
                        dcomplex *Full_EVec, 
                        int n, int MinN, int MaxN, int myid2, int numprocs2, 
                        int *is2, int *ie2, MPI_Comm comm );

void Save_EIGENVAL_CWF2W90( char *fname, double *EIGEN, int MinN, int MaxN, int myid2);

void Save_Projection_Matrix(char *mode,
			    char *fn,
			    double *RealA,
			    dcomplex *CompA,
			    dcomplex *Full_EVec, 
			    int MinN, int MaxN, 
			    int TNum_CWFs,
			    int nblk_CWF3, 
			    int na_rows_CWF3, int na_cols_CWF3,
			    int np_rows_CWF3, int np_cols_CWF3,
			    int my_prow_CWF3, int my_pcol_CWF3,
			    MPI_Comm comm);



static void lapack_dsyevx(int n0, int EVmax, double *A, double *Z, double *EVal);

static void Allocate_Free_Cluster_Col_CWF( int todo_flag, 
					   int myworld1, 
					   MPI_Comm *MPI_CommWD1,
					   int n,
					   int MaxN,
                                           int TNum_CWFs,
					   double **Cs,
					   double **Hs, 
					   double **Vs,
					   double **Ws,
					   double **EVs_PAO,
					   double ***WFs );

static double Calc_CWF_Cluster_Col(
    int myid0,
    int numprocs0,
    int myid1,
    int numprocs1,
    int myworld1,
    int size_H1,
    int *is2,
    int *ie2,
    int *MP,
    int n,
    int MinN,
    int MaxN,
    MPI_Comm *MPI_CommWD1,
    int *Comm_World_StartID1,
    double ****OLP0,
    double **ko,
    double **EVec1,
    dcomplex *Full_EVec );


void solve_evp_real_( int *n1, int *n2, double *Cs, int *na_rows1, double *a, double *Ss, int *na_rows2, int *nblk, 
                      int *mpi_comm_rows_int, int *mpi_comm_cols_int);

void elpa_solve_evp_real_2stage_double_impl_( int *n1, int *n2, double *Cs, int *na_rows1, double *a, double *Ss, int *na_rows2, 
                                              int *nblk, int *na_cols1, int *mpi_comm_rows_int, int *mpi_comm_cols_int, int *mpiworld);



double Cluster_DFT_Col_CWF(
                   int SCF_iter,
                   int SpinP_switch,
                   double **ko,
                   double *****nh, 
                   double ****CntOLP,
                   double *****CDM,
                   double *****EDM,
                   double Eele0[2], double Eele1[2],
		   int myworld1,
		   int *NPROCS_ID1,
		   int *Comm_World1,
		   int *NPROCS_WD1,
		   int *Comm_World_StartID1,
		   MPI_Comm *MPI_CommWD1,
                   int *MP,
		   int *is2,
		   int *ie2,
		   double *Ss,
		   double *Cs,
		   double *Hs,
		   double *CDM1,
		   double *EDM1,
		   double *PDM1,
		   int size_H1,
                   int *SP_NZeros,
                   int *SP_Atoms,
                   double **EVec1,
                   double *Work1)
{
  static int firsttime=1;
  int i,j,l,n,n2,n1,i1,i1s,j1,k1,l1;
  int wan,HOMO0,HOMO1;
  int spin,po,num0,num1,ires;
  int ct_AN,k,wanA,tnoA,wanB,tnoB;
  int GA_AN,Anum,loopN,Gc_AN;
  int MA_AN,LB_AN,GB_AN,Bnum,MinN,MaxN;
  int wan1,mul,m,bcast_flag;
  int *is1,*ie1;
  double time0,lumos,av_num;
  double *OneD_Mat1;
  double TZ,my_sum,sum,sumE,max_x=60.0;
  double sum0,sum1,sum2,sum3;
  double My_Eele1[2],tmp1,tmp2;
  double Num_State,x,FermiF,Dnum,Dnum2;
  double FermiF2,x2,diffF;
  double dum,ChemP_MAX,ChemP_MIN,spin_degeneracy;
  double TStime,TEtime;
  double FermiEps = 1.0e-13;
  double EV_cut0;
  double res;
  double e,weight,out0,out1,b0,b1,e0,e1;
  int numprocs0,myid0;
  int numprocs1,myid1;
  int ID,p,world_Snd,world_Rcv; 
  char *Name_Angular[Supported_MaxL+1][2*(Supported_MaxL+1)+1];
  char *Name_Multiple[20];
  double OLP_eigen_cut=Threshold_OLP_Eigen;
  char file_EV[YOUSO10] = ".EV";
  char buf[fp_bsize];          /* setvbuf */
  FILE *fp_EV;
  double stime, etime;
  double time1,time2,time3,time4,time5,time6,time7;

  /* for OpenMP */
  int OMPID,Nthrds,Nprocs;

  MPI_Comm mpi_comm_rows, mpi_comm_cols;
  int mpi_comm_rows_int,mpi_comm_cols_int;
  int info,ig,jg,il,jl,prow,pcol,brow,bcol;
  int ZERO=0, ONE=1;
  double alpha = 1.0; double beta = 0.0;
  int LOCr, LOCc, node, irow, icol;
  double C_spin_i1,mC_spin_i1;
  int sp;

  int ID0,IDS,IDR,Max_Num_Snd_EV,Max_Num_Rcv_EV;
  int *Num_Snd_EV,*Num_Rcv_EV;
  int *index_Snd_i,*index_Snd_j,*index_Rcv_i,*index_Rcv_j;
  double *EVec_Snd,*EVec_Rcv;
  MPI_Status stat;
  MPI_Request request;
  dcomplex *Full_EVec;

  /* initialize DM_func, CWF_Charge, and CWF_Energy */
 
  DM_func = 0.0;

  for (spin=0; spin<2; spin++){
    for (i=0; i<TNum_CWFs; i++){
      CWF_Charge[spin][i] = 0.0;
      CWF_Energy[spin][i] = 0.0;
    }
  }

  /* for time */
  MPI_Barrier(mpi_comm_level1);
  dtime(&TStime);

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs0);
  MPI_Comm_rank(mpi_comm_level1,&myid0);

  MPI_Comm_size(MPI_CommWD1[myworld1],&numprocs1);
  MPI_Comm_rank(MPI_CommWD1[myworld1],&myid1);

  /* show the message */

  if (myid0==0){
    printf("\n<Calculation of Closest Wannier Functions>\n");   
  }

  /*******************************************************
   calculation of atomic, hybrid, or molecular orbitals
  *******************************************************/

  if ( CWF_Guiding_Orbital==1 || CWF_Guiding_Orbital==2 ){
    AllocateArrays_Col_QAO();
    time1 = Calc_Hybrid_AO_Col(CntOLP, nh, CDM);
  }

  else if (CWF_Guiding_Orbital==3){
    time1 = Calc_MO_in_Bulk_Col(CntOLP, nh, ParDM);
  }

  /****************************************************
             calculation of the array size
  ****************************************************/

  n = 0;
  for (i=1; i<=atomnum; i++){
    wanA  = WhatSpecies[i];
    n = n + Spe_Total_CNO[wanA];
  }
  n2 = n + 2;

  /* GPU dispatch (added by H.Kawai): assign CUDA/OpenACC device when CuSOLVER is requested */
  if (scf_eigen_lib_flag == CuSOLVER && n >= GPU_CPU_SWITCH_NUM) {
      set_cuda_default_device_from_local_rank();
      set_openacc_nvidia_device_from_local_rank();
  }

  /****************************************************
                 Allocation of arrays
  ****************************************************/

  is1 = (int*)malloc(sizeof(int)*numprocs1);
  ie1 = (int*)malloc(sizeof(int)*numprocs1);

  Num_Snd_EV = (int*)malloc(sizeof(int)*numprocs1);
  Num_Rcv_EV = (int*)malloc(sizeof(int)*numprocs1);

  if (measure_time){
    time1 = 0.0;
    time2 = 0.0;
    time3 = 0.0;
    time4 = 0.0;
    time5 = 0.0;
    time6 = 0.0;
    time7 = 0.0;
  }

  if      (SpinP_switch==0) spin_degeneracy = 2.0;
  else if (SpinP_switch==1) spin_degeneracy = 1.0;

  /****************************************************
           calculate the total core charge
  ****************************************************/

  TZ = 0.0;
  for (i=1; i<=atomnum; i++){
    wan = WhatSpecies[i];
    TZ += Spe_Core_Charge[wan];
  }

  /****************************************************
         find the numbers of partions for MPI
  ****************************************************/

  if ( numprocs1<=n ){

    av_num = (double)n/(double)numprocs1;

    for (ID=0; ID<numprocs1; ID++){
      is1[ID] = (int)(av_num*(double)ID) + 1; 
      ie1[ID] = (int)(av_num*(double)(ID+1)); 
    }

    is1[0] = 1;
    ie1[numprocs1-1] = n; 

  }

  else{

    for (ID=0; ID<n; ID++){
      is1[ID] = ID + 1; 
      ie1[ID] = ID + 1;
    }
    for (ID=n; ID<numprocs1; ID++){
      is1[ID] =  1;
      ie1[ID] = -2;
    }
  }

  /****************************************************
         1. diagonalize the overlap matrix     
         2. search negative eigenvalues
  ****************************************************/

  MPI_Barrier(mpi_comm_level1);

  if (SCF_iter==1){
    Overlap_Cluster_Ss(CntOLP,Cs,MP,myworld1);
  }

  if (SpinP_switch==1 && numprocs0==1){
    Hamiltonian_Cluster_Hs(nh[0],Hs,MP,0,0);
  }
  else{
    for (spin=0; spin<=SpinP_switch; spin++){
      Hamiltonian_Cluster_Hs(nh[spin],Hs,MP,spin,myworld1);
    } 
  }

  if (SCF_iter==1){

    if (measure_time) dtime(&stime);

    MPI_Comm_split(MPI_CommWD1[myworld1],my_pcol,my_prow,&mpi_comm_rows);
    MPI_Comm_split(MPI_CommWD1[myworld1],my_prow,my_pcol,&mpi_comm_cols);

    mpi_comm_rows_int = MPI_Comm_c2f(mpi_comm_rows);
    mpi_comm_cols_int = MPI_Comm_c2f(mpi_comm_cols);

    if (scf_eigen_lib_flag==1){

      F77_NAME(solve_evp_real,SOLVE_EVP_REAL)(&n, &n, Cs, &na_rows, &ko[0][1], Ss, &na_rows, &nblk, &mpi_comm_rows_int, &mpi_comm_cols_int);
    }

    else if (scf_eigen_lib_flag==CuSOLVER && GPU_CPU_SWITCH_NUM<=n && na_rows==n && na_cols==n){
      OpenMX_CuSolver_DenseDsyevx_1based(Cs,Ss,ko[0],n,n,"Cluster_DFT_Col_CWF overlap CuSOLVER");
    }

    else if (scf_eigen_lib_flag==2 || scf_eigen_lib_flag==CuSOLVER){

#ifndef kcomp

      int mpiworld;
      mpiworld = MPI_Comm_c2f(MPI_CommWD1[myworld1]);

      F77_NAME(elpa_solve_evp_real_2stage_double_impl,ELPA_SOLVE_EVP_REAL_2STAGE_DOUBLE_IMPL)(&n, &n, Cs, &na_rows, &ko[0][1], 
                                            Ss, &na_rows, &nblk, &na_cols, &mpi_comm_rows_int, &mpi_comm_cols_int, &mpiworld);

#endif

    }

    MPI_Comm_free(&mpi_comm_rows);
    MPI_Comm_free(&mpi_comm_cols);

    /* print to the standard output */

    if (2<=level_stdout && myid0==Host_ID){
      for (l=1; l<=n; l++){
	printf("  Eigenvalues of OLP  %2d  %18.15f\n",l,ko[0][l]);fflush(stdout);
      }
    }

    /* minus eigenvalues to 1.0e-10 */

    for (l=1; l<=n; l++){
      if (ko[0][l]<0.0) ko[0][l] = 1.0e-10;
    }

    /* calculate S*1/sqrt(ko) */

    for (l=1; l<=n; l++){
      ko[0][l] = 1.0/sqrt(ko[0][l]);
    }

    for(i=0; i<na_rows; i++){
      for(j=0; j<na_cols; j++){
	jg = np_cols*nblk*((j)/nblk) + (j)%nblk + ((np_cols+my_pcol)%np_cols)*nblk + 1;
	Ss[j*na_rows+i] = Ss[j*na_rows+i]*ko[0][jg];
      }
    }

    if (measure_time){
      dtime(&etime);
      time1 += etime - stime; 
    }
  }

  /***********************************************
     find the number of states to be solved 
  ***********************************************/

  if (CWF_WindowF_switch==0){ // Poly

    /* find MaxN */
    for (i=n; 1<=i; i--){
      e = ko[0][i] - ChemP;
      out1 = CWF_disentangling_Erange[3];
      if (e<out1) break;
    }
    MaxN = i; 

    /* find MinN */
    for (i=1; i<=n; i++){
      e = ko[0][i] - ChemP;
      out0 = CWF_disentangling_Erange[0];
      if (out0<e) break;
    }
    MinN = i;

    TNum_PAOs = n;
    Num_KS_states_CWF = MaxN - MinN + 1;

    if (myid0==Host_ID){
      printf("  MinN=%2d MaxN=%2d (MaxN-MinN+1)=%2d TNum_CWFs=%2d\n",
	     MinN,MaxN,MaxN-MinN+1,TNum_CWFs);
    }

    if (MaxN<MinN){
      if (myid0==Host_ID){
	printf("Could not find any state to be included.\n"); 
	printf("Parameters for CWF disentangling must be improper.\n"); 
      }
      MPI_Finalize();
      exit(0);    
    }

    if (Num_KS_states_CWF<TNum_CWFs){
      if (myid0==Host_ID){
	printf("(MaxN-MinN+1) should be larger than TNum_CWFs.\n"); 
	printf("Parameters for CWF disentangling must be improper.\n"); 
      }
      MPI_Finalize();
      exit(0);    
    }

  }

  else if (CWF_WindowF_switch==1){ // Fermi
    TNum_PAOs = n;
    MinN = 1;
    MaxN = n;
    Num_KS_states_CWF = MaxN;
  }

  /****************************************************
    calculations of eigenvalues for up and down spins

     Note:
         MP indicates the starting position of
              atom i in arraies H and S
  ****************************************************/

  MPI_Barrier(mpi_comm_level1);

  if ( numprocs1<=MaxN ){
    
    av_num = (double)MaxN/(double)numprocs1;
    for (ID=0; ID<numprocs1; ID++){
      is2[ID] = (int)(av_num*(double)ID) + 1; 
      ie2[ID] = (int)(av_num*(double)(ID+1)); 
    }
    
    is2[0] = 1;
    ie2[numprocs1-1] = MaxN; 
  }

  else{

    for (ID=0; ID<MaxN; ID++){
      is2[ID] = ID + 1; 
      ie2[ID] = ID + 1;
    }

    for (ID=MaxN; ID<numprocs1; ID++){
      is2[ID] =  1;
      ie2[ID] =  0;
    }
  }

  /* making data structure of MPI communicaition for eigenvectors */

  for (ID=0; ID<numprocs1; ID++){
    Num_Snd_EV[ID] = 0;
    Num_Rcv_EV[ID] = 0;
  }

  for (i=0; i<na_rows; i++){

    ig = np_rows*nblk*((i)/nblk) + (i)%nblk + ((np_rows+my_prow)%np_rows)*nblk + 1;

    po = 0;
    for (ID=0; ID<numprocs1; ID++){
      if (is2[ID]<=ig && ig <=ie2[ID]){
        po = 1;
        ID0 = ID;
        break;
      }
    }

    if (po==1) Num_Snd_EV[ID0] += na_cols;
  }

  for (ID=0; ID<numprocs1; ID++){
    IDS = (myid1 + ID) % numprocs1;
    IDR = (myid1 - ID + numprocs1) % numprocs1;
    if (ID!=0){
      MPI_Isend(&Num_Snd_EV[IDS], 1, MPI_INT, IDS, 999, MPI_CommWD1[myworld1], &request);
      MPI_Recv(&Num_Rcv_EV[IDR], 1, MPI_INT, IDR, 999, MPI_CommWD1[myworld1], &stat);
      MPI_Wait(&request,&stat);
    }
    else{
      Num_Rcv_EV[IDR] = Num_Snd_EV[IDS];
    }
  }

  Max_Num_Snd_EV = 0;
  Max_Num_Rcv_EV = 0;
  for (ID=0; ID<numprocs1; ID++){
    if (Max_Num_Snd_EV<Num_Snd_EV[ID]) Max_Num_Snd_EV = Num_Snd_EV[ID];
    if (Max_Num_Rcv_EV<Num_Rcv_EV[ID]) Max_Num_Rcv_EV = Num_Rcv_EV[ID];
  }  

  Max_Num_Snd_EV++;
  Max_Num_Rcv_EV++;

  index_Snd_i = (int*)malloc(sizeof(int)*Max_Num_Snd_EV);
  index_Snd_j = (int*)malloc(sizeof(int)*Max_Num_Snd_EV);
  EVec_Snd = (double*)malloc(sizeof(double)*Max_Num_Snd_EV);
  index_Rcv_i = (int*)malloc(sizeof(int)*Max_Num_Rcv_EV);
  index_Rcv_j = (int*)malloc(sizeof(int)*Max_Num_Rcv_EV);
  EVec_Rcv = (double*)malloc(sizeof(double)*Max_Num_Rcv_EV);

  if (cwf2w90_output_flag){
    Full_EVec = (dcomplex*)malloc(sizeof(dcomplex)*MaxN*n);
  }  
  else {
    Full_EVec = (dcomplex*)malloc(sizeof(dcomplex)*1);
  }

  /* for PrintMemory */

  if (firsttime && memoryusage_fileout){
    PrintMemory("Cluster_DFT_Col: is1",sizeof(int)*numprocs1,NULL);
    PrintMemory("Cluster_DFT_Col: ie1",sizeof(int)*numprocs1,NULL);
    PrintMemory("Cluster_DFT_Col: Num_Snd_EV",sizeof(int)*numprocs1,NULL);
    PrintMemory("Cluster_DFT_Col: Num_Snd_EV",sizeof(int)*numprocs1,NULL);
    PrintMemory("Cluster_DFT_Col: index_Snd_i",sizeof(int)*Max_Num_Snd_EV,NULL);
    PrintMemory("Cluster_DFT_Col: index_Snd_j",sizeof(int)*Max_Num_Snd_EV,NULL);
    PrintMemory("Cluster_DFT_Col: index_Rcv_i",sizeof(int)*Max_Num_Rcv_EV,NULL);
    PrintMemory("Cluster_DFT_Col: index_Rcv_j",sizeof(int)*Max_Num_Rcv_EV,NULL);
    PrintMemory("Cluster_DFT_Col: EVec_Snd",sizeof(double)*Max_Num_Snd_EV,NULL);
    PrintMemory("Cluster_DFT_Col: EVec_Rcv",sizeof(double)*Max_Num_Rcv_EV,NULL);
  }
  firsttime=0;

  /*********************************************** 
       create a director to save EVec1 files
  ***********************************************/

  if (cwf2w90_output_flag){
    char operate[1000];
    sprintf(operate,"%s%s_evec",filepath,filename);
    mkdir(operate,0775); 
  }

  /*********************************************** 
       create a director to save EVec1 files
  ***********************************************/

  /* initialize ko */
  for (spin=0; spin<=SpinP_switch; spin++){
    for (i1=1; i1<=n; i1++){
      ko[spin][i1] = 10000.0;
    }
  }

  /* spin=myworld1 */

  spin = myworld1;

 diagonalize:

  if (measure_time) dtime(&stime);

  /* pdgemm */

  /* H * U * 1.0/sqrt(ko[l]) */

  for(i=0; i<na_rows_max*na_cols_max; i++){
    Cs[i] = 0.0;
  }

  Cblacs_barrier(ictxt1,"A");
  F77_NAME(pdgemm,PDGEMM)("N","N",&n,&n,&n,&alpha,Hs,&ONE,&ONE,descH,Ss,&ONE,&ONE,descS,&beta,Cs,&ONE,&ONE,descC);

  /* 1.0/sqrt(ko[l]) * U^+ H * U * 1.0/sqrt(ko[l]) */

  for(i=0; i<na_rows*na_cols; i++){
    Hs[i] = 0.0;
  }

  Cblacs_barrier(ictxt1,"C");
  F77_NAME(pdgemm,PDGEMM)("T","N",&n,&n,&n,&alpha,Ss,&ONE,&ONE,descS,Cs,&ONE,&ONE,descC,&beta,Hs,&ONE,&ONE,descH);

  if (measure_time){
    dtime(&etime);
    time2 += etime - stime;
  }

  /* The output C matrix is distributed by column. */

  if (measure_time) dtime(&stime);

  MPI_Comm_split(MPI_CommWD1[myworld1],my_pcol,my_prow,&mpi_comm_rows);
  MPI_Comm_split(MPI_CommWD1[myworld1],my_prow,my_pcol,&mpi_comm_cols);

  mpi_comm_rows_int = MPI_Comm_c2f(mpi_comm_rows);
  mpi_comm_cols_int = MPI_Comm_c2f(mpi_comm_cols);

  if (scf_eigen_lib_flag==1){
    F77_NAME(solve_evp_real,SOLVE_EVP_REAL)(&n, &MaxN, Hs, &na_rows, &ko[spin][1], Cs, 
                                            &na_rows, &nblk, &mpi_comm_rows_int, &mpi_comm_cols_int);
  }
  else if (scf_eigen_lib_flag==CuSOLVER && GPU_CPU_SWITCH_NUM<=n && na_rows==n && na_cols==n){
    OpenMX_CuSolver_DenseDsyevx_1based(Hs,Cs,ko[spin],n,MaxN,"Cluster_DFT_Col_CWF Hamiltonian CuSOLVER");
  }

  else if (scf_eigen_lib_flag==2 || scf_eigen_lib_flag==CuSOLVER){

#ifndef kcomp
    int mpiworld;
    mpiworld = MPI_Comm_c2f(MPI_CommWD1[myworld1]);

    F77_NAME(elpa_solve_evp_real_2stage_double_impl,ELPA_SOLVE_EVP_REAL_2STAGE_DOUBLE_IMPL)(&n, &MaxN, Hs, &na_rows, &ko[spin][1], 
									 		    Cs, &na_rows, &nblk, &na_cols, 
                                                                                            &mpi_comm_rows_int, &mpi_comm_cols_int, &mpiworld);
#endif

  }

  MPI_Comm_free(&mpi_comm_rows);
  MPI_Comm_free(&mpi_comm_cols);

  if (measure_time){
    dtime(&etime);
    time3 += etime - stime;
  }

  /****************************************************
      transformation to the original eigenvectors.
                       NOTE 244P
  ****************************************************/

  if (measure_time) dtime(&stime);

  for (i=0; i<na_rows*na_cols; i++) Hs[i] = 0.0;

  Cblacs_barrier(ictxt1,"A");
  F77_NAME(pdgemm,PDGEMM)("T","T",&n,&n,&n,&alpha,Cs,&ONE,&ONE,descC,Ss,&ONE,&ONE,descS,&beta,Hs,&ONE,&ONE,descH);

  /* MPI communications of Hs */

  for (ID=0; ID<numprocs1; ID++){
    
    IDS = (myid1 + ID) % numprocs1;
    IDR = (myid1 - ID + numprocs1) % numprocs1;

    k = 0;
    for(i=0; i<na_rows; i++){
      ig = np_rows*nblk*((i)/nblk) + (i)%nblk + ((np_rows+my_prow)%np_rows)*nblk + 1;
      if (is2[IDS]<=ig && ig <=ie2[IDS]){

        for (j=0; j<na_cols; j++){
          jg = np_cols*nblk*((j)/nblk) + (j)%nblk + ((np_cols+my_pcol)%np_cols)*nblk + 1;
 
          index_Snd_i[k] = ig;
          index_Snd_j[k] = jg;
          EVec_Snd[k] = Hs[j*na_rows+i];
          k++; 
	}
      }
    }

    if (ID!=0){

      if (Num_Snd_EV[IDS]!=0){
        MPI_Isend(index_Snd_i, Num_Snd_EV[IDS], MPI_INT, IDS, 999, MPI_CommWD1[myworld1], &request);
      }
      if (Num_Rcv_EV[IDR]!=0){
        MPI_Recv(index_Rcv_i, Num_Rcv_EV[IDR], MPI_INT, IDR, 999, MPI_CommWD1[myworld1], &stat);
      }
      if (Num_Snd_EV[IDS]!=0){
        MPI_Wait(&request,&stat);
      }

      if (Num_Snd_EV[IDS]!=0){
        MPI_Isend(index_Snd_j, Num_Snd_EV[IDS], MPI_INT, IDS, 999, MPI_CommWD1[myworld1], &request);
      }
      if (Num_Rcv_EV[IDR]!=0){
        MPI_Recv(index_Rcv_j, Num_Rcv_EV[IDR], MPI_INT, IDR, 999, MPI_CommWD1[myworld1], &stat);
      }
      if (Num_Snd_EV[IDS]!=0){
        MPI_Wait(&request,&stat);
      }

      if (Num_Snd_EV[IDS]!=0){
        MPI_Isend(EVec_Snd, Num_Snd_EV[IDS], MPI_DOUBLE, IDS, 999, MPI_CommWD1[myworld1], &request);
      }
      if (Num_Rcv_EV[IDR]!=0){
        MPI_Recv(EVec_Rcv, Num_Rcv_EV[IDR], MPI_DOUBLE, IDR, 999, MPI_CommWD1[myworld1], &stat);
      }
      if (Num_Snd_EV[IDS]!=0){
        MPI_Wait(&request,&stat);
      }
    }
    else{
      for(k=0; k<Num_Snd_EV[IDS]; k++){
        index_Rcv_i[k] = index_Snd_i[k];
        index_Rcv_j[k] = index_Snd_j[k];
        EVec_Rcv[k] = EVec_Snd[k];
      } 
    }

    for(k=0; k<Num_Rcv_EV[IDR]; k++){
      ig = index_Rcv_i[k];
      jg = index_Rcv_j[k];

      //m = (jg-1)*(ie2[myid1]-is2[myid1]+1)+ig-is2[myid1]; 

      /* Transpose */
      m = (ig-is2[myid1])*n + (jg-1);
      EVec1[spin][m] = EVec_Rcv[k];
    }
  } // ID

  /***********************************
      if cwf2w90_output_flag==1,  
      save EVec1 in binary format 
  ***********************************/

  if (cwf2w90_output_flag){

    char fname_EVec[YOUSO10];
    dcomplex ctmp;

    sprintf(fname_EVec,"%s%s_evec/evec_%d_%d",filepath,filename,spin,0);
    Save_EVec_CWF2W90( "real", fname_EVec, EVec1[spin], &ctmp, Full_EVec, n, MinN, MaxN, 
		       myid1, numprocs1, is2, ie2, MPI_CommWD1[myworld1] ); 

    char fname_EVal[YOUSO10];

    sprintf(fname_EVal,"%s%s_evec/eval_%d_%d",filepath,filename,spin,0);
    Save_EIGENVAL_CWF2W90(fname_EVal, ko[spin], MinN, MaxN, myid1);

  }  

  if (measure_time){
    dtime(&etime);
    time4 += etime - stime;
  }

  if (SpinP_switch==1 && numprocs0==1 && spin==0){
    spin++;
    Hamiltonian_Cluster_Hs(nh[spin],Hs,MP,spin,spin);
    goto diagonalize; 
  }

  /*********************************************** 
    MPI: ko
  ***********************************************/

  if (measure_time) dtime(&stime);

  for (sp=0; sp<=SpinP_switch; sp++){
    MPI_Bcast(&ko[sp][1],MaxN,MPI_DOUBLE,Comm_World_StartID1[sp],mpi_comm_level1);
  }

  /*********************************************** 
              calculation of CWFs
  ***********************************************/

  time6 += Calc_CWF_Cluster_Col( myid0,numprocs0,myid1,numprocs1,myworld1,
			         size_H1,is2,ie2,MP,n,MinN,MaxN,MPI_CommWD1,Comm_World_StartID1,
			         CntOLP,ko,EVec1,Full_EVec );

  if (measure_time){
    printf("Cluster_DFT myid=%2d time1=%7.3f time2=%7.3f time3=%7.3f time4=%7.3f time5=%7.3f time6=%7.3f time7=%7.3f\n",
            myid0,time1,time2,time3,time4,time5,time6,time7);fflush(stdout); 
  }

  /****************************************************
                          Free
  ****************************************************/

  free(Full_EVec);
  
  free(EVec_Rcv);
  free(index_Rcv_j);
  free(index_Rcv_i);
  free(EVec_Snd);
  free(index_Snd_j);
  free(index_Snd_i);

  free(Num_Rcv_EV);
  free(Num_Snd_EV);
  free(ie1);
  free(is1);

  /* freeing of arrays */

  if ( CWF_Guiding_Orbital==1 || CWF_Guiding_Orbital==2 ){
    FreeArrays_Col_QAO();
  }

  /* for elapsed time */

  MPI_Barrier(mpi_comm_level1);
  dtime(&TEtime);
  time0 = TEtime - TStime;
  return time0;
}

  

double Calc_CWF_Cluster_Col(
    int myid0,
    int numprocs0,
    int myid1,
    int numprocs1,
    int myworld1,
    int size_H1,
    int *is2,
    int *ie2,
    int *MP,
    int n,
    int MinN,
    int MaxN,
    MPI_Comm *MPI_CommWD1,
    int *Comm_World_StartID1,
    double ****OLP0,
    double **ko,
    double **EVec1,
    dcomplex *Full_EVec )
{
  int Gc_AN,GB_AN,Mc_AN,h_AN,Gh_AN,wanB,wan1,tno1,wan2,tno2,TNum_CWFs;
  int i,i1,j,l,k,k1,ig,jg,m1,NumCWFs,spin,num,p,q,ID,ID1,mmin,mmax,m;
  int EVec1_size,Max_EVec1_size,sp,*int_data;
  int gidx,Lidx,GA_AN,dim,i0,pnum,idx;
  int *MP2,*MP3,mpi_info[5],fp_Hop_ok=0;
  char fname[YOUSO10];
  FILE *fp_Hop;
  double sum,max_x=60.0,x,FermiF,tmp,w1=1.0e-10;
  double e,in0,in1,out0,out1,weight,dif,b0,b1,e0,e1;
  double *TmpEVec1,*InProd,*InProd_BasisFunc,*C2,**S1;
  double *Cs,*Hs,*Vs,*Ws,*sv,**WFs,*EVs_PAO;
  double *work,**Hop;
  int num_zero_sv=0,rank,lwork,info;
  double Sum_Charges[2],Sum_Energies[2];
  double stime,etime;
  int ZERO=0,ONE=1;
  double alpha=1.0,beta=0.0;
  MPI_Status stat;
  MPI_Request request;
  MPI_Comm mpi_comm_rows, mpi_comm_cols;
  int mpi_comm_rows_int,mpi_comm_cols_int;

  dtime(&stime);

  /* find the total number of CWFs */

  MP2 = (int*)malloc(sizeof(int)*(atomnum+1));

  if (CWF_Guiding_Orbital==1 || CWF_Guiding_Orbital==2){

    TNum_CWFs = 0;
    for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
      MP2[Gc_AN] = TNum_CWFs;
      wan1 = WhatSpecies[Gc_AN];
      TNum_CWFs += CWF_Num_predefined[wan1];
    }
  }

  else if (CWF_Guiding_Orbital==3){

    TNum_CWFs = 0;
    for (gidx=0; gidx<Num_CWF_Grouped_Atoms; gidx++){
      MP2[gidx] = TNum_CWFs;
      TNum_CWFs += Num_CWF_MOs_Group[gidx];
    }
  }

  /* setting of arrays for SCALAPCK */

  Allocate_Free_Cluster_Col_CWF( 1, myworld1, MPI_CommWD1, n, MaxN, TNum_CWFs, &Cs, &Hs, &Vs, &Ws, &EVs_PAO, &WFs );

  /* find the maximum size of EVec1 */

  EVec1_size = 0;
  for (ID=0; ID<numprocs1; ID++){
    if ( EVec1_size < (n*(ie2[ID]-is2[ID]+1)) ) EVec1_size = n*(ie2[ID]-is2[ID]+1);
  }
  MPI_Allreduce(&EVec1_size,&Max_EVec1_size,1,MPI_INT,MPI_MAX,mpi_comm_level1);

  /* allocation of arrays */

  TmpEVec1 = (double*)malloc(sizeof(double)*Max_EVec1_size);
  InProd = (double*)malloc(sizeof(double)*Max_EVec1_size);
  InProd_BasisFunc = (double*)malloc(sizeof(double)*Max_EVec1_size);
  C2 = (double*)malloc(sizeof(double)*List_YOUSO[7]);
  S1 = (double**)malloc(sizeof(double*)*List_YOUSO[7]);
  for (i=0; i<List_YOUSO[7]; i++) S1[i] = (double*)malloc(sizeof(double)*List_YOUSO[7]);

  sv = (double*)malloc(sizeof(double)*(MaxN+1));

  Hop = (double**)malloc(sizeof(double*)*(SpinP_switch+1));
  for (spin=0; spin<(SpinP_switch+1); spin++){ 
    Hop[spin] = (double*)malloc(sizeof(double)*(TNum_CWFs*TNum_CWFs));
    for (p=0; p<(TNum_CWFs*TNum_CWFs); p++) Hop[spin][p] = 0.0;
  }

  /* get size of work and allocate work */

  lwork = -1;
  F77_NAME(pdgesvd,PDGESVD)( "V", "V",
			     &MaxN, &TNum_CWFs,
			     Cs, &ONE, &ONE, desc_CWF3,
			     sv,
			     Ws, &ONE, &ONE, desc_CWF3,
			     Vs, &ONE, &ONE, desc_CWF1,
			     &tmp, 
			     &lwork,
			     &info);

  lwork = (int)tmp + 1;
  work = (double*)malloc(sizeof(double)*lwork);

  //printf("ABC1 Max_States %2d %2d\n",Max_States[0],Max_States[1]);

  /************************************************
       calculation of <LNAO|Bloch functions> 
  ************************************************/

  /* set spin=0 */

  spin = 0;

 spinloop:

  /* initialize Cs and Hs */

  for (i=0; i<na_rows_CWF4*na_cols_CWF4; i++){
    Cs[i] = 0.0;
    Hs[i] = 0.0;
    Ws[i] = 0.0;
    Vs[i] = 0.0;
  }

  /* In the loop of ID, the index of spin is changed regardless of myworld1. */

  for (ID=0; ID<numprocs0; ID++){

    if (ID==myid0){
      if (numprocs0!=1) spin = myworld1;
      ID1 = myid1;
      num = n*(ie2[ID1]-is2[ID1]+1);
      mpi_info[0] = spin;
      mpi_info[1] = ID1;
      mpi_info[2] = num;
      mpi_info[3] = is2[ID1];
      mpi_info[4] = ie2[ID1];
    }

    /* MPI_Bcast of num */

    MPI_Bcast( &mpi_info[0], 5, MPI_INT, ID, mpi_comm_level1 );

    /* set parameters */

    spin = mpi_info[0];
    ID1  = mpi_info[1];
    num  = mpi_info[2];
    mmin = mpi_info[3];
    mmax = mpi_info[4];

    /* initialize InProd and InProd_BasisFunc */

    for (p=0; p<Max_EVec1_size; p++){
      InProd[p] = 0.0;
      InProd_BasisFunc[p] = 0.0;
    }

    if ( num!=0 ){

      /* MPI_Bcast of EVec1 */

      if ( ID==myid0 ){
	for (i=0; i<num; i++){
	  TmpEVec1[i] = EVec1[spin][i];
	}
      }

      MPI_Bcast( &TmpEVec1[0], num, MPI_DOUBLE, ID, mpi_comm_level1 );

      /* store TmpEVec1 into EVs_PAO in the block cyclic form with n x MaxN.  */

      if ( (myworld1==spin && numprocs0!=1) || numprocs0==1 ){ 

	for (j=0; j<na_cols_CWF4; j++){  // state

	  m1 = np_cols_CWF4*nblk_CWF4*((j)/nblk_CWF4) + (j)%nblk_CWF4 
	    + ((np_cols_CWF4+my_pcol_CWF4)%np_cols_CWF4)*nblk_CWF4;

	  if ((mmin-1)<=m1 && m1<=(mmax-1)){ 

	    for (i=0; i<na_rows_CWF4; i++){ // PAO

	      ig = np_rows_CWF4*nblk_CWF4*((i)/nblk_CWF4) + (i)%nblk_CWF4
		+ ((np_rows_CWF4+my_prow_CWF4)%np_rows_CWF4)*nblk_CWF4;

	      //EVs_PAO[j*na_rows_CWF4+i] = TmpEVec1[ ig*(mmax-mmin+1)+m1-mmin+1 ];

              // Transpose
	      EVs_PAO[j*na_rows_CWF4+i] = TmpEVec1[ (m1-mmin+1)*n+ig ];

	    } // i

	  } // if ((mmin-1)<=m1 && m1<=(mmax-1))
	} // j
      } // if ( (myworld1==spin && numprocs0!=1) || numprocs0==1 )

      /* calculate <Bloch functions|basis functions> */

      for (m=0; m<=(mmax-mmin); m++){  // loop for KS index 
        
	for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

	  Gc_AN = M2G[Mc_AN];
	  wan1 = WhatSpecies[Gc_AN];
	  tno1 = Spe_Total_CNO[wan1];
      
	  for (i=0; i<tno1; i++){

	    sum = 0.0;

	    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	      Gh_AN = natn[Gc_AN][h_AN];
	      wan2 = WhatSpecies[Gh_AN];
	      tno2 = Spe_Total_CNO[wan2];

	      for (j=0; j<tno2; j++){

		//m1 = (MP[Gh_AN]-1+j)*(mmax-mmin+1) + m;

                // Transpose
		m1 = m*n + (MP[Gh_AN]-1+j);

		C2[j] = TmpEVec1[m1];
	      }      

	      for (j=0; j<tno2; j++){
		sum += OLP0[Mc_AN][h_AN][i][j]*C2[j];
	      } // j

	    } // h_AN

	    /* store <Bloch functions|basis functions> */

	    p = m*n + MP[Gc_AN] - 1 + i;
	    InProd_BasisFunc[p] = sum;

	  } // i
	} // Mc_AN 
      } // m      

      /* MPI_Allreduce of InProd_BasisFunc */       

      MPI_Allreduce( MPI_IN_PLACE, &InProd_BasisFunc[0], (mmax-mmin+1)*n, MPI_DOUBLE, MPI_SUM, mpi_comm_level1 );

      /* AO and HO case: calculate <Bloch functions|guiding functions> */

      if (CWF_Guiding_Orbital==1 || CWF_Guiding_Orbital==2){

        for (m=0; m<=(mmax-mmin); m++){  // loop for KS index 

	  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

	    Gc_AN = M2G[Mc_AN];
	    wan1 = WhatSpecies[Gc_AN];
	    tno1 = Spe_Total_CNO[wan1];
	    NumCWFs = CWF_Num_predefined[wan1];
      
	    for (i=0; i<NumCWFs; i++){

	      sum = 0.0;
	      for (l=0; l<tno1; l++){

		p = m*n + MP[Gc_AN] - 1 + l;  
		sum += InProd_BasisFunc[p]*QAO_coes[spin][Gc_AN][tno1*i+l];

	      } // l

   	      q = m*TNum_CWFs + MP2[Gc_AN] + i;
 	      InProd[q] = sum;

	    } // i
	  } // Mc_AN 
	} // m

      } /* end if (CWF_Guiding_Orbital==1 || CWF_Guiding_Orbital==2) */

      /* MO case: calculate <Bloch functions|guiding functions> */

      else if (CWF_Guiding_Orbital==3){

        for (m=0; m<=(mmax-mmin); m++){  // loop for KS index 
          
          for (gidx=0; gidx<Num_CWF_Grouped_Atoms; gidx++){

	    /* set MP3 */

            MP3 = (int*)malloc(sizeof(int)*CWF_Grouped_Atoms_EachNum[gidx]);

	    k = 0;
	    for (Lidx=0; Lidx<CWF_Grouped_Atoms_EachNum[gidx]; Lidx++){

	      MP3[Lidx] = k;

	      GA_AN = CWF_Grouped_Atoms[gidx][Lidx];
	      wan1 = WhatSpecies[GA_AN];
	      tno1 = Spe_Total_CNO[wan1];
	      k += tno1; 
	    }
 	    dim = k;

            for (k=0; k<Num_CWF_MOs_Group[gidx]; k++){

              sum = 0.0;

              for (Lidx=0; Lidx<CWF_Grouped_Atoms_EachNum[gidx]; Lidx++){

                i0 = MP3[Lidx];
                GA_AN = CWF_Grouped_Atoms[gidx][Lidx];
      	        wan1 = WhatSpecies[GA_AN];
	        tno1 = Spe_Total_CNO[wan1];
                
                for (i=0; i<tno1; i++){

  		  p = m*n + MP[GA_AN] - 1 + i;  
                  sum += InProd_BasisFunc[p]*CWF_Guiding_MOs[gidx][k][i0+i];

		} // i
	      } // Lidx

   	      q = m*TNum_CWFs + MP2[gidx] + k;
 	      InProd[q] = sum;

	    } // k

            /* freeing of MP3 */
 
            free(MP3); 

	  } // gidx
	} // m

      } /* end of else if (CWF_Guiding_Orbital==3) */

      /* MPI_Allreduce of InProd */       

      if (CWF_Guiding_Orbital==1 || CWF_Guiding_Orbital==2){
        MPI_Allreduce( MPI_IN_PLACE, &InProd[0], (mmax-mmin+1)*TNum_CWFs, MPI_DOUBLE, MPI_SUM, mpi_comm_level1 );
      }

      /* store InProd into an array: Cs in the SCALAPACK form */       

      if ( (myworld1==spin && numprocs0!=1) || numprocs0==1 ){ 

	for (i=0; i<na_rows_CWF3; i++){

	  m1 = np_rows_CWF3*nblk_CWF3*((i)/nblk_CWF3) + (i)%nblk_CWF3
	    + ((np_rows_CWF3+my_prow_CWF3)%np_rows_CWF3)*nblk_CWF3 + 1;

	  if ( mmin<=m1 && m1<=mmax ){ // the base is 1 for mmin and mmax.

	    m = m1 - mmin;  // local index

	    for (j=0; j<na_cols_CWF3; j++){

	      jg = np_cols_CWF3*nblk_CWF3*((j)/nblk_CWF3) + (j)%nblk_CWF3 
		+ ((np_cols_CWF3+my_pcol_CWF3)%np_cols_CWF3)*nblk_CWF3;

	      p = m*TNum_CWFs + jg;
	      Cs[j*na_rows_CWF3+i] = InProd[p];

	    } // j
	  } // if ( mmin<=m1 && m1<=mmax ) 
	} // i

      } // if ( (myworld1==spin && numprocs0!=1) || numprocs0==1 ) 
    } // if ( num!=0 )
  } // ID

  /*
  printf("Cs\n");
  for (i=0; i<na_rows_CWF3; i++){
    for (j=0; j<na_cols_CWF3; j++){
      printf("%8.4f ",Cs[j*na_rows_CWF3+i]);
    }
    printf("\n");
  }
  MPI_Finalize();
  exit(0);
  */

  /* set the index of sp */

  if (numprocs0==1) sp = spin;
  else              sp = myworld1; 

  /*********************************************************************
    Disentanling procedure:
    apply weighting based on the KS eigenvalue, 
    where the energy range is specified by CWF.disentangling.Erange. 
  *********************************************************************/

  out0 = CWF_disentangling_Erange[0];
  in0 = CWF_disentangling_Erange[1];
  in1 = CWF_disentangling_Erange[2];
  out1 = CWF_disentangling_Erange[3];
  b0 = 1.0/CWF_disentangling_smearing_kBT0;
  b1 = 1.0/CWF_disentangling_smearing_kBT1;
  e0 = CWF_disentangling_Erange[0] + ChemP; 
  e1 = CWF_disentangling_Erange[1] + ChemP; 

  for (i=0; i<na_rows_CWF3; i++){

    m1 = np_rows_CWF3*nblk_CWF3*((i)/nblk_CWF3) + (i)%nblk_CWF3
      + ((np_rows_CWF3+my_prow_CWF3)%np_rows_CWF3)*nblk_CWF3 + 1;

    e = ko[sp][m1];

    if (CWF_WindowF_switch==0){ // Poly
      weight = weight_CWF2(e,ChemP,in0,in1,out0,out1); 
    }
    else if (CWF_WindowF_switch==1){ // Fermi
      weight = (1.0/(exp(b0*(e0-e))+1.0))*(1.0/(exp(b1*(e-e1))+1.0)) + CWF_disentangling_smearing_bound;
    }

    for (j=0; j<na_cols_CWF3; j++){
      Cs[j*na_rows_CWF3+i] *= weight;
    }
  }

  /********************************************************
    if cwf2w90_output_flag==1,  
    save <Bloch functions|guiding functions>
    using MPI_File_write_at
  ********************************************************/

  if (cwf2w90_output_flag){

    char fname_ProM[YOUSO10];
    FILE *fp_ProM;
    dcomplex ctmp;

    sprintf(fname_ProM,"%s%s_evec/pm_%d_%d",filepath,filename,sp,0);

    Save_Projection_Matrix("real",fname_ProM,
			   Cs, &ctmp, Full_EVec, 
			   MinN, MaxN, TNum_CWFs,
			   nblk_CWF3, 
			   na_rows_CWF3, na_cols_CWF3,
			   np_rows_CWF3, np_cols_CWF3,
			   my_prow_CWF3, my_pcol_CWF3,
			   MPI_CommWD1[myworld1]);
  }

  /********************************************************
      Singular Value Decomposition (SVD) of Cs.
      As for how to set desc_CWF1, see also the comment 
      in Allocate_Free_Cluster_Col_CWF
  ********************************************************/

  /* As for the size of Cs, Ws, and Vs, see https://manpages.ubuntu.com/manpages/focal/man3/pdgesvd.3.html */

  F77_NAME(pdgesvd,PDGESVD)( "V", "V",
			     &MaxN, &TNum_CWFs,
			     Cs, &ONE, &ONE, desc_CWF3,   
			     sv,
			     Ws, &ONE, &ONE, desc_CWF3,                              
			     Vs, &ONE, &ONE, desc_CWF1,
			     work,
			     &lwork,
			     &info);

  /*
  for (i=0; i<TNum_CWFs; i++){
    printf("ZZZ1 of Cs myworld1=%2d myid0=%2d sp=%2d i=%2d sv=%18.15f\n",myworld1,myid0,sp,i,sv[i]);
  }
  */

  if (myid1==0){
    for (i=0; i<TNum_CWFs; i++){
      dif = sv[i] - 1.0;
      DM_func += dif*dif;
    }
  }

  /*******************************************
        Polar Decomposition (PD) of Cs
  *******************************************/

  Cblacs_barrier(ictxt1_CWF1,"A");
  F77_NAME(pdgemm,PDGEMM)( "N","N",
			   &MaxN, &TNum_CWFs, &TNum_CWFs, 
                           &alpha,
			   Ws,&ONE,&ONE,desc_CWF3,
			   Vs,&ONE,&ONE,desc_CWF1,
                           &beta,
			   Hs,&ONE,&ONE,desc_CWF3 );

  /**********************************************************
             calculation of <W_{i0}|H|W_{j0}>
  ***********************************************************/

  /* E x Hs -> Cs */

  for (i=0; i<na_rows_CWF3; i++){

    ig = np_rows_CWF3*nblk_CWF3*((i)/nblk_CWF3) + (i)%nblk_CWF3
      + ((np_rows_CWF3+my_prow_CWF3)%np_rows_CWF3)*nblk_CWF3 + 1;

    for (j=0; j<na_cols_CWF3; j++){

      x = ko[sp][ig];
      Cs[j*na_rows_CWF3+i] = x*Hs[j*na_rows_CWF3+i];
    }
  }

  /* Hs^dag x Cs -> Ws */

  Cblacs_barrier(ictxt1_CWF1,"A");
  F77_NAME(pdgemm,PDGEMM)( "T","N",
			   &TNum_CWFs, &TNum_CWFs, &MaxN,
			   &alpha,
			   Hs,&ONE,&ONE,desc_CWF3,
			   Cs,&ONE,&ONE,desc_CWF3,
			   &beta,
			   Ws,&ONE,&ONE,desc_CWF1 );

  for (i=0; i<na_rows_CWF1; i++){

    ig = np_rows_CWF1*nblk_CWF1*((i)/nblk_CWF1) + (i)%nblk_CWF1
      + ((np_rows_CWF1+my_prow_CWF1)%np_rows_CWF1)*nblk_CWF1;

    if (ig<TNum_CWFs){

      for (j=0; j<na_cols_CWF1; j++){

	jg = np_cols_CWF1*nblk_CWF1*((j)/nblk_CWF1) + (j)%nblk_CWF1 
	  + ((np_cols_CWF1+my_pcol_CWF1)%np_cols_CWF1)*nblk_CWF1;

	if (jg<TNum_CWFs){
	  Hop[sp][jg*TNum_CWFs+ig] = Ws[j*na_rows_CWF1+i]; 
	}
      }
    }
  }

  /**********************************************************
     calculate CWF w.r.t PAOs: 
                    EVs_PAO * Hs -> WFs
           (n x MaxN) * (MaxN x TNum_CWFs) -> (n x TNum_CWFs) 
             WF4               WF3               WF2
  ***********************************************************/

  Cblacs_barrier(ictxt1_CWF4,"A");
  F77_NAME(pdgemm,PDGEMM)( "N","N",
                           &n, &TNum_CWFs, &MaxN, 
                           &alpha,
                           EVs_PAO,   &ONE, &ONE, desc_CWF4,
                           Hs,        &ONE, &ONE, desc_CWF3,
                           &beta,
                           WFs[sp],   &ONE, &ONE, desc_CWF2 );

  /*************************************************************
    CWFs coefficients w.r.t PAOs are stored for calculating 
    values on CWFs on grids 
  **************************************************************/

  if (CWF_fileout_flag==1){

    for (k=0; k<CWF_fileout_Num; k++){

      /* AO or HO case */ 
      if (CWF_Guiding_Orbital==1 || CWF_Guiding_Orbital==2){
	Gc_AN = CWF_file_Atoms[k];
	wan1 = WhatSpecies[Gc_AN];
	pnum = CWF_Num_predefined[wan1];
	idx = Gc_AN;
      }
      /* MO case */
      else if (CWF_Guiding_Orbital==3){
	gidx = CWF_file_MOs[k];
	pnum = Num_CWF_MOs_Group[gidx]; 
	idx = gidx;
      }

      for (p=0; p<pnum; p++){

	q = MP2[idx] + p;

	for (i=0; i<n; i++) CWF_Coef[sp][k][p][0][0][0][i] = 0.0;

	for (j=0; j<na_cols_CWF2; j++){

	  jg = np_cols_CWF2*nblk_CWF2*((j)/nblk_CWF2) + (j)%nblk_CWF2 
	    + ((np_cols_CWF2+my_pcol_CWF2)%np_cols_CWF2)*nblk_CWF2;

	  if (q==jg){

	    for (i=0; i<na_rows_CWF2; i++){

	      ig = np_rows_CWF2*nblk_CWF2*((i)/nblk_CWF2) + (i)%nblk_CWF2
		+ ((np_rows_CWF2+my_prow_CWF2)%np_rows_CWF2)*nblk_CWF2;

	      CWF_Coef[sp][k][p][0][0][0][ig] = WFs[sp][j*na_rows_CWF2+i];

	    } // i
	  } // if (q==jg)
	} // j
      } // p
    } // k

    for (k=0; k<CWF_fileout_Num; k++){

      if (CWF_Guiding_Orbital==1 || CWF_Guiding_Orbital==2){
	Gc_AN = CWF_file_Atoms[k];
	wan1 = WhatSpecies[Gc_AN];
	pnum = CWF_Num_predefined[wan1];
      }
      else if (CWF_Guiding_Orbital==3){
	gidx = CWF_file_MOs[k];
	pnum = Num_CWF_MOs_Group[gidx]; 
      }

      for (p=0; p<pnum; p++){
	MPI_Allreduce( MPI_IN_PLACE, &CWF_Coef[sp][k][p][0][0][0][0], n, MPI_DOUBLE, MPI_SUM, MPI_CommWD1[myworld1]);
      }
    }

  } // end of if (CWF_fileout_flag==1)

  /***********************************************************
   calculations of effective charges and local band energies
  ***********************************************************/

  for (i=0; i<na_rows_CWF3; i++){

    ig = np_rows_CWF3*nblk_CWF3*((i)/nblk_CWF3) + (i)%nblk_CWF3
      + ((np_rows_CWF3+my_prow_CWF3)%np_rows_CWF3)*nblk_CWF3 + 1;

    for (j=0; j<na_cols_CWF3; j++){

      jg = np_cols_CWF3*nblk_CWF3*((j)/nblk_CWF3) + (j)%nblk_CWF3 
	+ ((np_cols_CWF3+my_pcol_CWF3)%np_cols_CWF3)*nblk_CWF3;

      x = (ko[sp][ig] - ChemP)*Beta;
      if (x<=-max_x) x = -max_x;
      if (max_x<=x)  x = max_x;
      FermiF = 1.0/(1.0 + exp(x));
      tmp = FermiF*Hs[j*na_rows_CWF3+i]*Hs[j*na_rows_CWF3+i];

      CWF_Charge[sp][jg] += tmp;
      CWF_Energy[sp][jg] += tmp*ko[sp][ig];
    }
  }

  MPI_Allreduce( MPI_IN_PLACE, &CWF_Charge[sp][0], TNum_CWFs, MPI_DOUBLE, MPI_SUM, MPI_CommWD1[myworld1] );
  MPI_Allreduce( MPI_IN_PLACE, &CWF_Energy[sp][0], TNum_CWFs, MPI_DOUBLE, MPI_SUM, MPI_CommWD1[myworld1] );

  /* goto spinloop if necessary. */

  if (SpinP_switch==1 && numprocs0==1 && spin==0){
    spin++;
    goto spinloop;
  }

  /* MPI: CWF_Charge, CWF_Energy, and CWF_Coef */

  if (SpinP_switch==1 && numprocs0!=1){

    MPI_Bcast(&CWF_Charge[0][0], TNum_CWFs, MPI_DOUBLE, Comm_World_StartID1[0], mpi_comm_level1);
    MPI_Bcast(&CWF_Energy[0][0], TNum_CWFs, MPI_DOUBLE, Comm_World_StartID1[0], mpi_comm_level1);

    MPI_Bcast(&CWF_Charge[1][0], TNum_CWFs, MPI_DOUBLE, Comm_World_StartID1[1], mpi_comm_level1);
    MPI_Bcast(&CWF_Energy[1][0], TNum_CWFs, MPI_DOUBLE, Comm_World_StartID1[1], mpi_comm_level1);

    if (CWF_fileout_flag==1){
      for (k=0; k<CWF_fileout_Num; k++){

	if (CWF_Guiding_Orbital==1 || CWF_Guiding_Orbital==2){

	  Gc_AN = CWF_file_Atoms[k];
	  wan1 = WhatSpecies[Gc_AN];
          pnum = CWF_Num_predefined[wan1];
	}

        else if (CWF_Guiding_Orbital==3){
  	  gidx = CWF_file_MOs[k];
          pnum = Num_CWF_MOs_Group[gidx]; 
	}

	for (p=0; p<pnum; p++){
	  MPI_Bcast(&CWF_Coef[0][k][p][0][0][0][0], n, MPI_DOUBLE, Comm_World_StartID1[0], mpi_comm_level1);
	  MPI_Bcast(&CWF_Coef[1][k][p][0][0][0][0], n, MPI_DOUBLE, Comm_World_StartID1[1], mpi_comm_level1);
	}
      }
    }
  }

  Sum_Charges[0] = 0.0;
  Sum_Charges[1] = 0.0;
  Sum_Energies[0] = 0.0;;
  Sum_Energies[1] = 0.0;;

  for (spin=0; spin<=SpinP_switch; spin++){
    for (i=0; i<TNum_CWFs; i++){
      Sum_Charges[spin]  += CWF_Charge[spin][i];
      Sum_Energies[spin] += CWF_Energy[spin][i];
    }
  }

  if (SpinP_switch==0){
    Sum_Charges[1]  = Sum_Charges[0];
    Sum_Energies[1] = Sum_Energies[0];
  }

  /* show DM_func */

  MPI_Allreduce( MPI_IN_PLACE, &DM_func, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

  if (SpinP_switch==0) DM_func = 2.0*DM_func;

  if (myid0==0) printf("DM_func=%15.12f  %15.12f per (CWF*spin)\n",
                        DM_func,DM_func/(double)(2*TNum_CWFs));

  /**********************************************************
             save the effective charges to a file
  ***********************************************************/

  if (CWF_Guiding_Orbital==1 || CWF_Guiding_Orbital==2){

    if ( myid0==Host_ID ){
        
      int i0;
      char file_CWF_Charge[YOUSO10];
      FILE *fp_CWF_Charge;
      char buf[fp_bsize];          /* setvbuf */
      double sumP[2],sumE[2],TZ;

      /* calculate TZ */

      TZ = 0.0;
      for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
	wan1 = WhatSpecies[Gc_AN];
	TZ += Spe_Core_Charge[wan1];
      }

      sprintf(file_CWF_Charge,"%s%s.CWF_Charge",filepath,filename);

      if ((fp_CWF_Charge = fopen(file_CWF_Charge,"w")) != NULL){

	setvbuf(fp_CWF_Charge,buf,_IOFBF,fp_bsize);  /* setvbuf */

	fprintf(fp_CWF_Charge,"\n");
	fprintf(fp_CWF_Charge,"***********************************************************\n");
	fprintf(fp_CWF_Charge,"***********************************************************\n");
	fprintf(fp_CWF_Charge,"              Populations evaluated by CWF                 \n");
	fprintf(fp_CWF_Charge,"***********************************************************\n");
	fprintf(fp_CWF_Charge,"***********************************************************\n\n");

        fprintf(fp_CWF_Charge,"  DM_func. %15.12f  %15.12f per (CWF*spin)\n\n",
		DM_func,DM_func/(double)(2*TNum_CWFs));

	fprintf(fp_CWF_Charge,"  Total spin moment (muB)  %12.9f\n\n",(Sum_Charges[0]-Sum_Charges[1]));
	fprintf(fp_CWF_Charge,"                    Up spin      Down spin     Sum           Diff\n");
 
	for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){

	  wan1 = WhatSpecies[Gc_AN];
	  i0 = MP2[Gc_AN];
	  sumP[0] = 0.0; sumP[1] = 0.0;

	  for (spin=0; spin<=SpinP_switch; spin++){
	    for (i=0; i<CWF_Num_predefined[wan1]; i++){
	      sumP[spin] += CWF_Charge[spin][i0+i];
	    }
	  }
       
	  if (SpinP_switch==0){
	    sumP[1] = sumP[0];
	  }             

	  fprintf(fp_CWF_Charge,"   %4d %4s     %12.9f %12.9f  %12.9f  %12.9f\n",
		  Gc_AN, SpeName[wan1], sumP[0], sumP[1], sumP[0]+sumP[1], sumP[0]-sumP[1]);

	} // Gc_AN

	fprintf(fp_CWF_Charge,"\n");
	fprintf(fp_CWF_Charge," Sum of populations evaluated by CWF\n");
	fprintf(fp_CWF_Charge,"     up   =%12.5f  down          =%12.5f\n",
		Sum_Charges[0],Sum_Charges[1]);
	fprintf(fp_CWF_Charge,"     total=%12.5f  ideal(neutral)=%12.5f\n",
		Sum_Charges[0]+Sum_Charges[1],TZ);     

	/* decomposed populations */

	fprintf(fp_CWF_Charge,"\n\n  Decomposed populations evaluated by CWF\n");

	for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){

	  wan1 = WhatSpecies[Gc_AN];
	  i0 = MP2[Gc_AN];

	  fprintf(fp_CWF_Charge,"\n %4d %4s          Up spin      Down spin     Sum           Diff\n",
		  Gc_AN,SpeName[wan1]);
	  fprintf(fp_CWF_Charge,"  orbital index\n");

	  wan1 = WhatSpecies[Gc_AN];
	  sumP[0] = 0.0; sumP[1] = 0.0;

	  if (SpinP_switch==0){
	    for (i=0; i<CWF_Num_predefined[wan1]; i++){
	      fprintf(fp_CWF_Charge,"      %2d         %12.9f %12.9f  %12.9f  %12.9f\n",
		      i,CWF_Charge[0][i0+i],CWF_Charge[0][i0+i],2.0*CWF_Charge[0][i0+i],0.0);
	    }
	  }
	  else if (SpinP_switch==1){
	    for (i=0; i<CWF_Num_predefined[wan1]; i++){
	      fprintf(fp_CWF_Charge,"      %2d         %12.9f %12.9f  %12.9f  %12.9f\n",
		      i,CWF_Charge[0][i0+i],CWF_Charge[1][i0+i],
		      CWF_Charge[0][i0+i]+CWF_Charge[1][i0+i],
		      CWF_Charge[0][i0+i]-CWF_Charge[1][i0+i]);
	    }
	  } 

	} // Gc_AN

	/* fclose fp_CWF_Charge */

	fclose(fp_CWF_Charge);

      } // if ((fp_CWF_Charge = fopen(file_CWF_Charge,"w")) != NULL)

      else{
	printf("Failure of saving the CWF_Charge file.\n");
      }

    } // if ( myid0==Host_ID )

  } /* end of if (CWF_Guiding_Orbital==1 || CWF_Guiding_Orbital==2) */

  else if (CWF_Guiding_Orbital==3){

    if ( myid0==Host_ID ){
        
      int i0;
      char file_CWF_Charge[YOUSO10];
      FILE *fp_CWF_Charge;
      char buf[fp_bsize];          /* setvbuf */
      double sumP[2],sumE[2],TZ;

      /* calculate TZ */

      TZ = 0.0;
      for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
	wan1 = WhatSpecies[Gc_AN];
	TZ += Spe_Core_Charge[wan1];
      }

      sprintf(file_CWF_Charge,"%s%s.CWF_Charge",filepath,filename);

      if ((fp_CWF_Charge = fopen(file_CWF_Charge,"w")) != NULL){

	setvbuf(fp_CWF_Charge,buf,_IOFBF,fp_bsize);  /* setvbuf */

	fprintf(fp_CWF_Charge,"\n");
	fprintf(fp_CWF_Charge,"***********************************************************\n");
	fprintf(fp_CWF_Charge,"***********************************************************\n");
	fprintf(fp_CWF_Charge,"              Populations evaluated by CWF                 \n");
	fprintf(fp_CWF_Charge,"***********************************************************\n");
	fprintf(fp_CWF_Charge,"***********************************************************\n\n");

        fprintf(fp_CWF_Charge,"  DM_func. %15.12f  %15.12f per CWF\n\n",
		DM_func,DM_func/(double)(2*TNum_CWFs));

	fprintf(fp_CWF_Charge,"  Total spin moment (muB)  %12.9f\n\n",(Sum_Charges[0]-Sum_Charges[1]));
	fprintf(fp_CWF_Charge,"   Group            Up spin      Down spin     Sum           Diff\n");

        for (gidx=0; gidx<Num_CWF_Grouped_Atoms; gidx++){

	  i0 = MP2[gidx];
	  sumP[0] = 0.0; sumP[1] = 0.0;

	  for (spin=0; spin<=SpinP_switch; spin++){
	    for (Lidx=0; Lidx<Num_CWF_MOs_Group[gidx]; Lidx++){
	      sumP[spin] += CWF_Charge[spin][i0+Lidx];
	    }
	  }

	  if (SpinP_switch==0){
	    sumP[1] = sumP[0];
	  }             

	  fprintf(fp_CWF_Charge,"   %4d          %12.9f %12.9f  %12.9f  %12.9f\n",
		  gidx+1, sumP[0], sumP[1], sumP[0]+sumP[1], sumP[0]-sumP[1]);

	} // gidx                 

	fprintf(fp_CWF_Charge,"\n");
	fprintf(fp_CWF_Charge," Sum of populations evaluated by CWF\n");
	fprintf(fp_CWF_Charge,"     up   =%12.5f  down          =%12.5f\n",
		Sum_Charges[0],Sum_Charges[1]);
	fprintf(fp_CWF_Charge,"     total=%12.5f  ideal(neutral)=%12.5f\n",
		Sum_Charges[0]+Sum_Charges[1],TZ);     

	/* decomposed populations */

	fprintf(fp_CWF_Charge,"\n\n  Orbitally decomposed populations evaluated by CWF\n");

        for (gidx=0; gidx<Num_CWF_Grouped_Atoms; gidx++){

	  wan1 = WhatSpecies[Gc_AN];
	  i0 = MP2[gidx];

	  fprintf(fp_CWF_Charge,"\n  Group %4d      Up spin      Down spin     Sum           Diff\n",gidx+1);
	  fprintf(fp_CWF_Charge,"  orbital index\n");

	  wan1 = WhatSpecies[Gc_AN];
	  sumP[0] = 0.0; sumP[1] = 0.0;

	  if (SpinP_switch==0){
	    for (i=0; i<Num_CWF_MOs_Group[gidx]; i++){
	      fprintf(fp_CWF_Charge,"      %2d         %12.9f %12.9f  %12.9f  %12.9f\n",
		      i+1,CWF_Charge[0][i0+i],CWF_Charge[0][i0+i],2.0*CWF_Charge[0][i0+i],0.0);
	    }
	  }
	  else if (SpinP_switch==1){
	    for (i=0; i<Num_CWF_MOs_Group[gidx]; i++){
	      fprintf(fp_CWF_Charge,"      %2d         %12.9f %12.9f  %12.9f  %12.9f\n",
		      i+1,CWF_Charge[0][i0+i],CWF_Charge[1][i0+i],
		      CWF_Charge[0][i0+i]+CWF_Charge[1][i0+i],
		      CWF_Charge[0][i0+i]-CWF_Charge[1][i0+i]);
	    }
	  } 

	} // gidx

	/* fclose fp_CWF_Charge */

	fclose(fp_CWF_Charge);

      } // if ((fp_CWF_Charge = fopen(file_CWF_Charge,"w")) != NULL)

      else{
	printf("Failure of saving the CWF_Charge file.\n");
      }

    } // if ( myid0==Host_ID )
  }

  /**********************************************************
             save Hop into a file of *.CWF.Hop
  ***********************************************************/

  for (spin=0; spin<(SpinP_switch+1); spin++){ 
    MPI_Allreduce( MPI_IN_PLACE, &Hop[spin][0], (TNum_CWFs*TNum_CWFs), MPI_DOUBLE, MPI_SUM, mpi_comm_level1 );
  }

  if (myid0==Host_ID){

    int_data = (int*)malloc(sizeof(int)*(atomnum+6));

    sprintf(fname,"%s%s.CWF.Hop",filepath,filename);
    if ((fp_Hop = fopen(fname,"w")) != NULL) fp_Hop_ok = 1;

    if ( CWF_Guiding_Orbital==1 || CWF_Guiding_Orbital==2 ){
      int_data[0] = atomnum;
    }
    else if (CWF_Guiding_Orbital==3){
      int_data[0] = Num_CWF_Grouped_Atoms;
    }    

    int_data[1] = SpinP_switch;
    int_data[2] = 1;
    int_data[3] = 1;
    int_data[4] = 1;
    int_data[5] = TNum_CWFs;

    if ( CWF_Guiding_Orbital==1 || CWF_Guiding_Orbital==2 ){
      for (i=1; i<=atomnum; i++){
        wan1 = WhatSpecies[i];
        int_data[5+i] = CWF_Num_predefined[wan1];
      }
    }
    else if (CWF_Guiding_Orbital==3){
      for (i=0; i<Num_CWF_Grouped_Atoms; i++){
        int_data[6+i] = Num_CWF_MOs_Group[i];
      }
    } 

    if (fp_Hop_ok==1){
      fwrite(&int_data[0],sizeof(int),(int_data[0]+6),fp_Hop);

      double tvecs[9];
      tvecs[0] = tv[1][1]; tvecs[1] = tv[1][2]; tvecs[2] = tv[1][3];               
      tvecs[3] = tv[2][1]; tvecs[4] = tv[2][2]; tvecs[5] = tv[2][3];               
      tvecs[6] = tv[3][1]; tvecs[7] = tv[3][2]; tvecs[8] = tv[3][3]; 
      fwrite(tvecs,sizeof(double),9,fp_Hop);
    }

    for (spin=0; spin<(SpinP_switch+1); spin++){ 

      if (fp_Hop_ok==1){
        fwrite(&Hop[spin][0],sizeof(double),(TNum_CWFs*TNum_CWFs),fp_Hop);
      }

      if (myid0==0 && 0){

	int GA,GB,wanA,wanB,p1,p2;
	double dx,dy,dz,dis; 

        printf("Hopping integrals with distance\n");

	p1 = 0;
	for (GA=1; GA<=atomnum; GA++){
	  wanA = WhatSpecies[GA];

	  p2 = 0;
	  for (GB=1; GB<=atomnum; GB++){
	    wanB = WhatSpecies[GB];

	    dx = Gxyz[GB][1] - Gxyz[GA][1]; 
	    dy = Gxyz[GB][2] - Gxyz[GA][2]; 
	    dz = Gxyz[GB][3] - Gxyz[GA][3]; 
	    dis = sqrt(dx*dx+dy*dy+dz*dz);

	    for (i=0; i<CWF_Num_predefined[wanA]; i++){
	      for (j=0; j<CWF_Num_predefined[wanB]; j++){
		printf("%15.12f %15.12f\n",dis*BohrR,Hop[spin][(p2+j)*TNum_CWFs+(p1+i)]); 
	      } 
	    }

	    p2 += CWF_Num_predefined[wanB]; 
	  } 
	  p1 += CWF_Num_predefined[wanA]; 
	} 
      }
    
    }

    free(int_data);
    if (fp_Hop_ok==1) fclose(fp_Hop);
  }  

  /* save Hop with the distance between two sites into the .CWF.Dis_vs_H file */

  if (myid0==0 && CWF_Dis_vs_H==1){

    int GA,GB,wanA,wanB,p1,p2,gidx1,gidx2,Lidx1,Lidx2;
    double dx,dy,dz,dis,xA,yA,zA,xB,yB,zB; 
    char fDis_vs_H[YOUSO10];
    FILE *fp_Dis_vs_H;
  
    sprintf(fDis_vs_H,"%s%s.CWF.Dis_vs_H",filepath,filename);

    if ((fp_Dis_vs_H = fopen(fDis_vs_H,"w")) != NULL){

      fprintf(fp_Dis_vs_H,"#\n"); 
      fprintf(fp_Dis_vs_H,"# spin, GA, GB, i, j, distance (Ang.), Hopping integral (eV)\n"); 
      fprintf(fp_Dis_vs_H,"#\n"); 

      if (CWF_Guiding_Orbital==1 || CWF_Guiding_Orbital==2){

        for (spin=0; spin<(SpinP_switch+1); spin++){ 

	  p1 = 0;
	  for (GA=1; GA<=atomnum; GA++){

	    wanA = WhatSpecies[GA];

	    p2 = 0;
	    for (GB=1; GB<=atomnum; GB++){

	      wanB = WhatSpecies[GB];

	      dx = Gxyz[GB][1] - Gxyz[GA][1]; 
	      dy = Gxyz[GB][2] - Gxyz[GA][2]; 
	      dz = Gxyz[GB][3] - Gxyz[GA][3]; 
	      dis = sqrt(dx*dx+dy*dy+dz*dz);

	      for (i=0; i<CWF_Num_predefined[wanA]; i++){
		for (j=0; j<CWF_Num_predefined[wanB]; j++){
		  fprintf(fp_Dis_vs_H,"%2d %2d %2d %2d %2d  %18.15f %18.15f\n",
			  spin,GA,GB,i,j,dis*BohrR,Hop[spin][(p2+j)*TNum_CWFs+(p1+i)]*eV2Hartree); 
		} 
	      }

	      p2 += CWF_Num_predefined[wanB]; 

	    } // GB 

	    p1 += CWF_Num_predefined[wanA]; 

	  } // GA
	} // spin

      } // end of if (myid0==0 && CWF_Dis_vs_H==1 && fp_Dis_vs_H_ok==1)

      else if (CWF_Guiding_Orbital==3){

        for (spin=0; spin<(SpinP_switch+1); spin++){ 

	  p1 = 0;

	  for (gidx1=0; gidx1<Num_CWF_Grouped_Atoms; gidx1++){
 
	    xA = 0.0; yA = 0.0; zA = 0.0; 

	    for (Lidx1=0; Lidx1<CWF_Grouped_Atoms_EachNum[gidx1]; Lidx1++){
	      GA = CWF_Grouped_Atoms[gidx1][Lidx1];
	      xA += Gxyz[GA][1];
	      yA += Gxyz[GA][2];
	      zA += Gxyz[GA][3];
	    }
	    xA /= (double)CWF_Grouped_Atoms_EachNum[gidx1];
	    yA /= (double)CWF_Grouped_Atoms_EachNum[gidx1];
	    zA /= (double)CWF_Grouped_Atoms_EachNum[gidx1];

	    p2 = 0;

	    for (gidx2=0; gidx2<Num_CWF_Grouped_Atoms; gidx2++){

	      xB = 0.0; yB = 0.0; zB = 0.0; 

	      for (Lidx2=0; Lidx2<CWF_Grouped_Atoms_EachNum[gidx2]; Lidx2++){
		GB = CWF_Grouped_Atoms[gidx2][Lidx2];
		xB += Gxyz[GB][1];
		yB += Gxyz[GB][2];
		zB += Gxyz[GB][3];
	      }
	      xB /= (double)CWF_Grouped_Atoms_EachNum[gidx2];
	      yB /= (double)CWF_Grouped_Atoms_EachNum[gidx2];
	      zB /= (double)CWF_Grouped_Atoms_EachNum[gidx2];

	      dx = xB - xA;
	      dy = yB - yA;
	      dz = zB - zA;
	      dis = sqrt(dx*dx+dy*dy+dz*dz);

	      for (i=0; i<Num_CWF_MOs_Group[gidx1]; i++){
		for (j=0; j<Num_CWF_MOs_Group[gidx2]; j++){
		  fprintf(fp_Dis_vs_H,"%2d %2d %2d %2d %2d  %18.15f %18.15f\n",
			  spin,gidx1+1,gidx2+1,i,j,dis*BohrR,Hop[spin][(p2+j)*TNum_CWFs+(p1+i)]*eV2Hartree); 
		}
	      }

	      p2 += Num_CWF_MOs_Group[gidx2];

	    } // gidx2

	    p1 += Num_CWF_MOs_Group[gidx1];
                 
	  } // gidx1
	} // spin

      } // end of else if (CWF_Guiding_Orbital==3)

      fclose(fp_Dis_vs_H);

    } /* end of if ((fp_Dis_vs_H = fopen(fDis_vs_H,"w")) != NULL) */

    else {
      printf("Could not open %s\n",fDis_vs_H);
    }

  } /* end of if (myid0==0 && CWF_Dis_vs_H==1) */

  /**********************************************************
                       freeing of arrays
  ***********************************************************/

  for (spin=0; spin<(SpinP_switch+1); spin++){ 
    free(Hop[spin]);
  }
  free(Hop);

  free(TmpEVec1);
  free(MP2);
  free(InProd);
  free(InProd_BasisFunc);
  free(C2);

  for (i=0; i<List_YOUSO[7]; i++){
    free(S1[i]);
  }
  free(S1);

  free(sv);
  free(work);

  /* freeing of arrays for SCALAPCK */

  Allocate_Free_Cluster_Col_CWF( 2, myworld1, MPI_CommWD1, n, MaxN, TNum_CWFs, &Cs, &Hs, &Vs, &Ws, &EVs_PAO, &WFs );

  /* mearuring elapsed time */

  dtime(&etime);
  return (etime-stime);
}



void Allocate_Free_Cluster_Col_CWF( int todo_flag, 
				    int myworld1, 
				    MPI_Comm *MPI_CommWD1,
				    int n,
				    int MaxN,
                                    int TNum_CWFs,
				    double **Cs,
				    double **Hs, 
				    double **Vs,
				    double **Ws,
				    double **EVs_PAO,
				    double ***WFs )
{
  static int firsttime=1;
  int ZERO=0, ONE=1,info,myid0,numprocs0,myid1,numprocs1;
  int i,k,nblk_m,nblk_m2,wanA,spin,size_EVec1;
  double tmp,tmp1;

  MPI_Barrier(mpi_comm_level1);
  MPI_Comm_size(mpi_comm_level1,&numprocs0);
  MPI_Comm_rank(mpi_comm_level1,&myid0);

  /********************************************
   allocation of arrays 
   
   CWF1: TNum_CWFs x TNum_CWFs
   CWF2: n x TNum_CWFs
   CWF3: MaxN x TNum_CWFs 
   CWF4: n x MaxN
  ********************************************/

  if (todo_flag==1){

    /* get numprocs1 and myid1 */

    MPI_Comm_size(MPI_CommWD1[myworld1],&numprocs1);
    MPI_Comm_rank(MPI_CommWD1[myworld1],&myid1);

    /* CWF1: setting of BLACS for matrices in size of TNum_CWFs x TNum_CWFs */

    np_cols_CWF1 = (int)(sqrt((float)numprocs1));
    do{
      if((numprocs1%np_cols_CWF1)==0) break;
      np_cols_CWF1--;
    } while (np_cols_CWF1>=2);
    np_rows_CWF1 = numprocs1/np_cols_CWF1;

    /*
     For pdgesvd, I noticed that the same 'nb' has to be used for Cs, Ws, and Vs.
     Otherwise, I encounterd an error. This is the reason why I used the same nb as shown below.  
    */

    nblk_m = NBLK;
    while((nblk_m*np_rows_CWF1>MaxN || nblk_m*np_cols_CWF1>TNum_CWFs) && (nblk_m > 1)){ // the same as for CWF3
    //while((nblk_m*np_rows_CWF1>TNum_CWFs || nblk_m*np_cols_CWF1>TNum_CWFs) && (nblk_m > 1)){
      nblk_m /= 2;
    }
    if(nblk_m<1) nblk_m = 1;

    MPI_Allreduce(&nblk_m,&nblk_CWF1,1,MPI_INT,MPI_MIN,mpi_comm_level1);

    my_prow_CWF1 = myid1/np_cols_CWF1;
    my_pcol_CWF1 = myid1%np_cols_CWF1;

    na_rows_CWF1 = numroc_(&MaxN, &nblk_CWF1, &my_prow_CWF1, &ZERO, &np_rows_CWF1 );  // the same as for CWF3
    na_cols_CWF1 = numroc_(&TNum_CWFs, &nblk_CWF1, &my_pcol_CWF1, &ZERO, &np_cols_CWF1 );

    bhandle1_CWF1 = Csys2blacs_handle(MPI_CommWD1[myworld1]);
    ictxt1_CWF1 = bhandle1_CWF1;

    Cblacs_gridinit(&ictxt1_CWF1, "Row", np_rows_CWF1, np_cols_CWF1);

    MPI_Allreduce(&na_rows_CWF1,&na_rows_max_CWF1,1,MPI_INT,MPI_MAX,MPI_CommWD1[myworld1]);
    MPI_Allreduce(&na_cols_CWF1,&na_cols_max_CWF1,1,MPI_INT,MPI_MAX,MPI_CommWD1[myworld1]);

    descinit_( desc_CWF1, &TNum_CWFs, &TNum_CWFs, &nblk_CWF1, &nblk_CWF1,  
               &ZERO, &ZERO, &ictxt1_CWF1, &na_rows_CWF1,  &info); 

    /* CWF2: setting of BLACS for matrices in size of n x TNum_CWFs */

    np_cols_CWF2 = (int)(sqrt((float)numprocs1));
    do{
      if((numprocs1%np_cols_CWF2)==0) break;
      np_cols_CWF2--;
    } while (np_cols_CWF2>=2);
    np_rows_CWF2 = numprocs1/np_cols_CWF2;

    nblk_m = NBLK;
    while((nblk_m*np_rows_CWF2>n || nblk_m*np_cols_CWF2>TNum_CWFs) && (nblk_m > 1)){
      nblk_m /= 2;
    }
    if(nblk_m<1) nblk_m = 1;

    MPI_Allreduce(&nblk_m,&nblk_CWF2,1,MPI_INT,MPI_MIN,mpi_comm_level1);

    ictxt1_CWF2 = bhandle1_CWF1;

    my_prow_CWF2 = myid1/np_cols_CWF2;
    my_pcol_CWF2 = myid1%np_cols_CWF2;

    Cblacs_gridinit(&ictxt1_CWF2, "Row", np_rows_CWF2, np_cols_CWF2);

    na_rows_CWF2 = numroc_(&n,         &nblk_CWF2, &my_prow_CWF2, &ZERO, &np_rows_CWF2 );
    na_cols_CWF2 = numroc_(&TNum_CWFs, &nblk_CWF2, &my_pcol_CWF2, &ZERO, &np_cols_CWF2 );

    MPI_Allreduce(&na_rows_CWF2, &na_rows_max_CWF2,1,MPI_INT,MPI_MAX,MPI_CommWD1[myworld1]);
    MPI_Allreduce(&na_cols_CWF2, &na_cols_max_CWF2,1,MPI_INT,MPI_MAX,MPI_CommWD1[myworld1]);

    descinit_( desc_CWF2, &n, &TNum_CWFs, &nblk_CWF2, &nblk_CWF2,  
               &ZERO, &ZERO, &ictxt1_CWF2, &na_rows_CWF2,  &info);

    desc_CWF2[1] = desc_CWF1[1];

    *WFs = (double**)malloc(sizeof(double*)*(SpinP_switch+1));
    for (spin=0; spin<(SpinP_switch+1); spin++){ 
      (*WFs)[spin] = (double*)malloc(sizeof(double)*na_rows_max_CWF2*na_cols_max_CWF2); 
    }

    /* CWF3: setting of BLACS for matrices in size of MaxN x TNum_CWFs */

    np_cols_CWF3 = (int)(sqrt((float)numprocs1));
    do{
      if((numprocs1%np_cols_CWF3)==0) break;
      np_cols_CWF3--;
    } while (np_cols_CWF3>=2);
    np_rows_CWF3 = numprocs1/np_cols_CWF3;

    nblk_m = NBLK;
    while((nblk_m*np_rows_CWF3>MaxN || nblk_m*np_cols_CWF3>TNum_CWFs) && (nblk_m > 1)){
      nblk_m /= 2;
    }
    if(nblk_m<1) nblk_m = 1;

    MPI_Allreduce(&nblk_m,&nblk_CWF3,1,MPI_INT,MPI_MIN,mpi_comm_level1);

    ictxt1_CWF3 = bhandle1_CWF1;

    my_prow_CWF3 = myid1/np_cols_CWF3;
    my_pcol_CWF3 = myid1%np_cols_CWF3;

    Cblacs_gridinit(&ictxt1_CWF3, "Row", np_rows_CWF3, np_cols_CWF3);

    na_rows_CWF3 = numroc_(&MaxN,      &nblk_CWF3, &my_prow_CWF3, &ZERO, &np_rows_CWF3 );
    na_cols_CWF3 = numroc_(&TNum_CWFs, &nblk_CWF3, &my_pcol_CWF3, &ZERO, &np_cols_CWF3 );

    MPI_Allreduce(&na_rows_CWF3, &na_rows_max_CWF3,1,MPI_INT,MPI_MAX,MPI_CommWD1[myworld1]);
    MPI_Allreduce(&na_cols_CWF3, &na_cols_max_CWF3,1,MPI_INT,MPI_MAX,MPI_CommWD1[myworld1]);

    descinit_( desc_CWF3, &MaxN, &TNum_CWFs, &nblk_CWF3, &nblk_CWF3,  
               &ZERO, &ZERO, &ictxt1_CWF3, &na_rows_CWF3,  &info);

    desc_CWF3[1] = desc_CWF1[1];

    /* CWF4: setting of BLACS for matrices in size of n x MaxN */

    np_cols_CWF4 = (int)(sqrt((float)numprocs1));
    do{
      if((numprocs1%np_cols_CWF4)==0) break;
      np_cols_CWF4--;
    } while (np_cols_CWF4>=2);
    np_rows_CWF4 = numprocs1/np_cols_CWF4;

    nblk_m = NBLK;
    while((nblk_m*np_rows_CWF4>n || nblk_m*np_cols_CWF4>MaxN) && (nblk_m > 1)){
      nblk_m /= 2;
    }
    if(nblk_m<1) nblk_m = 1;

    MPI_Allreduce(&nblk_m,&nblk_CWF4,1,MPI_INT,MPI_MIN,mpi_comm_level1);

    ictxt1_CWF4 = bhandle1_CWF1;

    my_prow_CWF4 = myid1/np_cols_CWF4;
    my_pcol_CWF4 = myid1%np_cols_CWF4;

    Cblacs_gridinit(&ictxt1_CWF4, "Row", np_rows_CWF4, np_cols_CWF4);

    na_rows_CWF4 = numroc_(&n,    &nblk_CWF4, &my_prow_CWF4, &ZERO, &np_rows_CWF4 );
    na_cols_CWF4 = numroc_(&MaxN, &nblk_CWF4, &my_pcol_CWF4, &ZERO, &np_cols_CWF4 );

    MPI_Allreduce(&na_rows_CWF4, &na_rows_max_CWF4,1,MPI_INT,MPI_MAX,MPI_CommWD1[myworld1]);
    MPI_Allreduce(&na_cols_CWF4, &na_cols_max_CWF4,1,MPI_INT,MPI_MAX,MPI_CommWD1[myworld1]);

    descinit_( desc_CWF4, &n, &MaxN, &nblk_CWF4, &nblk_CWF4,
               &ZERO, &ZERO, &ictxt1_CWF4, &na_rows_CWF4, &info);

    desc_CWF4[1] = desc_CWF1[1];

    *Cs = (double*)malloc(sizeof(double)*na_rows_max_CWF4*na_cols_max_CWF4);
    *Hs = (double*)malloc(sizeof(double)*na_rows_max_CWF4*na_cols_max_CWF4);
    *Vs = (double*)malloc(sizeof(double)*na_rows_max_CWF4*na_cols_max_CWF4);
    *Ws = (double*)malloc(sizeof(double)*na_rows_max_CWF4*na_cols_max_CWF4);
    *EVs_PAO = (double*)malloc(sizeof(double)*na_rows_max_CWF4*na_cols_max_CWF4); 

    /* save information of the memory usage */

    if (firsttime && memoryusage_fileout) {
      PrintMemory("Allocate_Free_Cluster_Col_CWF: Cs  ",sizeof(double)*na_rows_max_CWF4*na_cols_max_CWF4,NULL);
      PrintMemory("Allocate_Free_Cluster_Col_CWF: Hs  ",sizeof(double)*na_rows_max_CWF4*na_cols_max_CWF4,NULL);
      PrintMemory("Allocate_Free_Cluster_Col_CWF: Vs  ",sizeof(double)*na_rows_max_CWF4*na_cols_max_CWF4,NULL);
      PrintMemory("Allocate_Free_Cluster_Col_CWF: Ws  ",sizeof(double)*na_rows_max_CWF3*na_cols_max_CWF3,NULL);
      PrintMemory("Allocate_Free_Cluster_Col_CWF: WFs ",sizeof(double)*na_rows_max_CWF2*na_cols_max_CWF2,NULL);
      PrintMemory("Allocate_Free_Cluster_Col_CWF: EVs_PAO",sizeof(double)*na_rows_max_CWF4*na_cols_max_CWF4,NULL);
    }

    firsttime = 0;
  }

  /********************************************
               freeing of arrays 
  ********************************************/

  if (todo_flag==2){

    /* setting for BLACS */

    free(*Cs);
    free(*Hs);
    free(*Vs);
    free(*Ws);

    for (spin=0; spin<(SpinP_switch+1); spin++){ 
      free((*WFs)[spin]);
    }
    free(*WFs);

    free(*EVs_PAO);

    Cfree_blacs_system_handle(bhandle1_CWF1);
    Cblacs_gridexit(ictxt1_CWF1);

    Cfree_blacs_system_handle(bhandle1_CWF2);
    Cblacs_gridexit(ictxt1_CWF2);

    Cfree_blacs_system_handle(bhandle1_CWF3);
    Cblacs_gridexit(ictxt1_CWF3);

    Cfree_blacs_system_handle(bhandle1_CWF4);
    Cblacs_gridexit(ictxt1_CWF4);
  }
}



double Calc_Hybrid_AO_Col( double ****OLP0, double *****Hks, double *****CDM )
{
  int i,j,k,l,n,Mc_AN,Gc_AN,h_AN,Mh_AN,Gh_AN;
  int Cwan,num,wan1,wan2,tno0,tno1,tno2,spin;
  int Nloop,po,p,q,NumLNOs;
  double ***DMS;
  double *EVal,*EVec;
  double sum,sum0,F;

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

  DMS = (double***)malloc(sizeof(double**)*(Matomnum+1));
  for (p=0; p<(Matomnum+1); p++){
    DMS[p] = (double**)malloc(sizeof(double*)*(SpinP_switch+1));
    for (spin=0; spin<=SpinP_switch; spin++){
      DMS[p][spin] = (double*)malloc(sizeof(double)*List_YOUSO[7]*List_YOUSO[7]);
      for (i=0; i<List_YOUSO[7]*List_YOUSO[7]; i++) DMS[p][spin][i] = 0.0;
    }
  }

  EVal = (double*)malloc(sizeof(double)*List_YOUSO[7]);
  EVec = (double*)malloc(sizeof(double)*List_YOUSO[7]*List_YOUSO[7]);

  /********************************************
       calculation of DMS defined by DM*S
  ********************************************/

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

    Gc_AN = M2G[Mc_AN];
    wan1 = WhatSpecies[Gc_AN];
    tno1 = Spe_Total_CNO[wan1];

    for (spin=0; spin<=SpinP_switch; spin++){

      h_AN = 0;

      Gh_AN = natn[Gc_AN][h_AN];
      wan2 = WhatSpecies[Gh_AN];
      tno2 = Spe_Total_CNO[wan2];

      // select only the contributions of the minimal basis

      int mul,m,l2,mul2,m2; 

      if (h_AN==0){

	i = 0;
	for (l=0; l<=Spe_MaxL_Basis[wan1]; l++){
	  for (mul=0; mul<Spe_Num_Basis[wan1][l]; mul++){
	    for (m=0; m<(2*l+1); m++){
              
	      if (CWF_Guiding_AO[wan1][l][mul]==1){

		j = 0;   
		for (l2=0; l2<=Spe_MaxL_Basis[wan1]; l2++){
		  for (mul2=0; mul2<Spe_Num_Basis[wan1][l2]; mul2++){
		    for (m2=0; m2<(2*l2+1); m2++){

		      if (CWF_Guiding_AO[wan1][l2][mul2]==1){

			DMS[Mc_AN][spin][tno1*j+i] = -CDM[spin][Mc_AN][h_AN][i][j];
		      }
    
		      j++;

		    }
		  }
		}
	      }

	      i++;

	    }
	  }
	}
      }

    } /* spin */
  } /* Mc_AN */

  /********************************************
            diagonalization of DMS
  ********************************************/

  for (spin=0; spin<=SpinP_switch; spin++){

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

      Gc_AN = M2G[Mc_AN];
      wan1 = WhatSpecies[Gc_AN];
      tno1 = Spe_Total_CNO[wan1];
      NumLNOs = CWF_Num_predefined[wan1];

      /* call lapack_dsyevx */

      lapack_dsyevx( tno1, tno1, DMS[Mc_AN][spin], EVec, EVal );

      /* use hybrid orbitals */

      if (CWF_Guiding_Orbital==1){ 

	/* copy EVec to QAO_coes, where vectors in QAO_coes are stored in the column order. */

	for (j=0; j<NumLNOs; j++){

	  for (i=0; i<tno1; i++){
	    QAO_coes[spin][Gc_AN][tno1*j+i] = EVec[tno1*j+i];
	  }

	} /* j */
      }

      /* use atomic orbitals */

      else if (CWF_Guiding_Orbital==2){

        int mul,n0=0,n1=0; 

	for (l=0; l<=Spe_MaxL_Basis[wan1]; l++){
	  for (mul=0; mul<Spe_Num_Basis[wan1][l]; mul++){

            for (j=0; j<(2*l+1); j++){

              if (CWF_Guiding_AO[wan1][l][mul]==1){

                QAO_coes[spin][Gc_AN][tno1*n1+n0] = 1.0;

                n1++;
	      }

              n0++;
	    }  
	  }
	}
      }

      /* store the eigenvalues */

      for (i=0; i<NumLNOs; i++){
        LNAO1_pops[spin][Mc_AN][i] = sqrt(fabs(EVal[i]));
      }

      if (0 && myid==0){

        printf("QQQ1 Mc_AN=%2d\n",Mc_AN);

	for (i=0; i<tno1; i++){
	  printf("QQQ DMS Diago myid=%2d spin=%2d Mc_AN=%2d i=%2d LNAO1_pops=%15.11f\n",
                      myid,spin,Mc_AN,i,LNAO1_pops[spin][Mc_AN][i] );fflush(stdout);
	}
	printf("WWW1 myid=%2d QAO_coes Mc_AN=%2d spin=%2d\n",myid,Mc_AN,spin);fflush(stdout);
	for (i=0; i<tno1; i++){
	  for (j=0; j<NumLNOs; j++){
	    printf("%10.5f ",QAO_coes[spin][Gc_AN][tno1*j+i]);fflush(stdout);
	  }
	  printf("\n");fflush(stdout);
	}

        printf("Check orthogonalization\n"); 
	for (i=0; i<tno1; i++){
	  for (j=0; j<tno1; j++){
 
            sum = 0.0;
	    for (k=0; k<tno1; k++){
              sum += EVec[tno1*i+k]*EVec[tno1*j+k];
  	    }
            printf("%10.5f ",sum);
	  }
	  printf("\n");fflush(stdout);
	}

      } // if (0 && myid==0)

    } /* Mc_AN */

    for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){

      wan1 = WhatSpecies[Gc_AN];
      tno1 = Spe_Total_CNO[wan1];
      NumLNOs = CWF_Num_predefined[wan1];
      MPI_Allreduce(MPI_IN_PLACE, &QAO_coes[spin][Gc_AN][0],tno1*NumLNOs,MPI_DOUBLE,MPI_SUM,mpi_comm_level1);
    }

  } /* spin */

  /********************************************
             freeing of arrays
  ********************************************/

  for (p=0; p<(Matomnum+1); p++){
    for (spin=0; spin<=SpinP_switch; spin++){
      free(DMS[p][spin]);
    }
    free(DMS[p]);
  }
  free(DMS);

  free(EVal);
  free(EVec);

  /* elapsed time */
  dtime(&TEtime);
  return (TEtime-TStime);
}



double Calc_MO_in_Bulk_Col( double ****OLP0, double *****Hks, double *****CDM )
{
  int i,j,k,gidx,Lidx,GA_AN,GB_AN,GC_AN,tnum,wan1;
  int Lidx1,Lidx2,i0,j0,MA_AN,h_AN3,h_AN2,spin;
  int tno1,h_AN,Gh_AN,wan2,tno2,dim,l1,l2,l3,p,q;
  int wan3,tno3,RnC,il,jl,ig,jg,brow,bcol,prow,pcol;
  int numprocs,myid,po;
  int *MP3,**RMI0;
  double sum,TStime,TEtime;
  double *Hmo,*DMmo,*Smo,****DMmo2,****Hmo2,****Smo2,*DMS,*DMS2;
  double **OLPgidx,**Hgidx,*WI;
  double *EVal,*EVec;
  /* for scalpack */
  int ZERO=0, ONE=1, info;
  int nblk_MIB,np_rows_MIB,np_cols_MIB,na_rows_MIB,na_cols_MIB,na_rows_max_MIB,na_cols_max_MIB;
  int my_prow_MIB,my_pcol_MIB,nblk_m;
  int bhandle1_MIB,ictxt1_MIB;
  int desc_MIB[9];

  char file_CWF_MOs_info[YOUSO10];
  FILE *fp_MO_info;

  dtime(&TStime);

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /*****************************************************
                the main loop of gidx
  *****************************************************/

  /* open a file */
  if (myid==Host_ID){
    sprintf(file_CWF_MOs_info,"%s%s.CWF_MOs_info",filepath,filename);
    fp_MO_info = fopen(file_CWF_MOs_info,"w");
  }

  CWF_Guiding_MOs = (double***)malloc(sizeof(double**)*Num_CWF_Grouped_Atoms);

  for (gidx=0; gidx<Num_CWF_Grouped_Atoms; gidx++){

    /* allocation of MP3 */

    MP3 = (int*)malloc(sizeof(int)*CWF_Grouped_Atoms_EachNum[gidx]);

    /* set MP3 */

    k = 0;
    for (Lidx=0; Lidx<CWF_Grouped_Atoms_EachNum[gidx]; Lidx++){

      MP3[Lidx] = k;

      GA_AN = CWF_Grouped_Atoms[gidx][Lidx];
      wan1 = WhatSpecies[GA_AN];
      tno1 = Spe_Total_CNO[wan1];
      k += tno1; 
    }
 
    dim = k;

    /* setting for scalapack */

    /*
    np_cols_MIB = (int)(sqrt((float)numprocs));
    do{
      if((numprocs%np_cols_MIB)==0) break;
      np_cols_MIB--;
    } while (np_cols_MIB>=2);
    np_rows_MIB = numprocs/np_cols_MIB;

    nblk_m = NBLK;
    while((nblk_m*np_rows_MIB>dim || nblk_m*np_cols_MIB>dim) && (nblk_m > 1)){
      nblk_m /= 2;
    }
    if(nblk_m<1) nblk_m = 1;

    MPI_Allreduce(&nblk_m,&nblk_MIB,1,MPI_INT,MPI_MIN,mpi_comm_level1);

    my_prow_MIB = myid/np_cols_MIB;
    my_pcol_MIB = myid%np_cols_MIB;

    na_rows_MIB = numroc_(&dim, &nblk_MIB, &my_prow_MIB, &ZERO, &np_rows_MIB );
    na_cols_MIB = numroc_(&dim, &nblk_MIB, &my_pcol_MIB, &ZERO, &np_cols_MIB );

    bhandle1_MIB = Csys2blacs_handle(mpi_comm_level1);
    ictxt1_MIB = bhandle1_MIB;

    Cblacs_gridinit(&ictxt1_MIB, "Row", np_rows_MIB, np_cols_MIB);

    MPI_Allreduce(&na_rows_MIB, &na_rows_max_MIB,1,MPI_INT,MPI_MAX,mpi_comm_level1);
    MPI_Allreduce(&na_cols_MIB, &na_cols_max_MIB,1,MPI_INT,MPI_MAX,mpi_comm_level1);

    descinit_( desc_MIB, &dim, &dim, &nblk_MIB, &nblk_MIB,  
               &ZERO, &ZERO, &ictxt1_MIB, &na_rows_MIB,  &info);
    */

    /* count the number of non-zero elements */

    tnum = 0; 
    for (Lidx=0; Lidx<CWF_Grouped_Atoms_EachNum[gidx]; Lidx++){

      GA_AN = CWF_Grouped_Atoms[gidx][Lidx];
      wan1 = WhatSpecies[GA_AN];
      tno1 = Spe_Total_CNO[wan1];

      for (h_AN=0; h_AN<=FNAN[GA_AN]; h_AN++){

	Gh_AN = natn[GA_AN][h_AN];
	wan2 = WhatSpecies[Gh_AN];
	tno2 = Spe_Total_CNO[wan2];
        tnum += tno1*tno2;  
      }
    }

    /* allocation of arrays */ 

    DMS = (double*)malloc(sizeof(double)*dim*dim);
    for (i=0; i<dim*dim; i++) DMS[i] = 0.0;

    DMS2 = (double*)malloc(sizeof(double)*dim*dim);
    for (i=0; i<dim*dim; i++) DMS2[i] = 0.0;

    EVal = (double*)malloc(sizeof(double)*dim);
    EVec = (double*)malloc(sizeof(double)*dim*dim);

    WI = (double*)malloc(sizeof(double)*dim);

    OLPgidx = (double**)malloc(sizeof(double*)*dim);
    for (i=0; i<dim; i++){
      OLPgidx[i] = (double*)malloc(sizeof(double)*dim);
      for (j=0; j<dim; j++) OLPgidx[i][j] = 0.0;
    }

    Hgidx = (double**)malloc(sizeof(double*)*dim);
    for (i=0; i<dim; i++){
      Hgidx[i] = (double*)malloc(sizeof(double)*dim);
      for (j=0; j<dim; j++) Hgidx[i][j] = 0.0;
    }

    Smo = (double*)malloc(sizeof(double)*tnum);
    for (i=0; i<tnum; i++) Smo[i] = 0.0;

    DMmo = (double*)malloc(sizeof(double)*tnum);
    for (i=0; i<tnum; i++) DMmo[i] = 0.0;

    Hmo = (double*)malloc(sizeof(double)*tnum);
    for (i=0; i<tnum; i++) Hmo[i] = 0.0;

    Smo2 = (double****)malloc(sizeof(double***)*CWF_Grouped_Atoms_EachNum[gidx]);
    for (Lidx=0; Lidx<CWF_Grouped_Atoms_EachNum[gidx]; Lidx++){

      GA_AN = CWF_Grouped_Atoms[gidx][Lidx];
      wan1 = WhatSpecies[GA_AN];
      tno1 = Spe_Total_CNO[wan1];
      Smo2[Lidx] = (double***)malloc(sizeof(double**)*(FNAN[GA_AN]+1));

      for (h_AN=0; h_AN<=FNAN[GA_AN]; h_AN++){

	Gh_AN = natn[GA_AN][h_AN];
	wan2 = WhatSpecies[Gh_AN];
	tno2 = Spe_Total_CNO[wan2];
	Smo2[Lidx][h_AN] = (double**)malloc(sizeof(double*)*tno1);

	for (i=0; i<tno1; i++){
	  Smo2[Lidx][h_AN][i] = (double*)malloc(sizeof(double)*tno2);
          for (j=0; j<tno2; j++) Smo2[Lidx][h_AN][i][j] = 0.0;
	}
      }
    }

    DMmo2 = (double****)malloc(sizeof(double***)*CWF_Grouped_Atoms_EachNum[gidx]);
    for (Lidx=0; Lidx<CWF_Grouped_Atoms_EachNum[gidx]; Lidx++){

      GA_AN = CWF_Grouped_Atoms[gidx][Lidx];
      wan1 = WhatSpecies[GA_AN];
      tno1 = Spe_Total_CNO[wan1];
      DMmo2[Lidx] = (double***)malloc(sizeof(double**)*(FNAN[GA_AN]+1));

      for (h_AN=0; h_AN<=FNAN[GA_AN]; h_AN++){

	Gh_AN = natn[GA_AN][h_AN];
	wan2 = WhatSpecies[Gh_AN];
	tno2 = Spe_Total_CNO[wan2];
	DMmo2[Lidx][h_AN] = (double**)malloc(sizeof(double*)*tno1);

	for (i=0; i<tno1; i++){
	  DMmo2[Lidx][h_AN][i] = (double*)malloc(sizeof(double)*tno2);
	  for (j=0; j<tno2; j++) DMmo2[Lidx][h_AN][i][j] = 0.0;
	}
      }
    }

    Hmo2 = (double****)malloc(sizeof(double***)*CWF_Grouped_Atoms_EachNum[gidx]);
    for (Lidx=0; Lidx<CWF_Grouped_Atoms_EachNum[gidx]; Lidx++){

      GA_AN = CWF_Grouped_Atoms[gidx][Lidx];
      wan1 = WhatSpecies[GA_AN];
      tno1 = Spe_Total_CNO[wan1];

      Hmo2[Lidx] = (double***)malloc(sizeof(double**)*(FNAN[GA_AN]+1));
      for (h_AN=0; h_AN<=FNAN[GA_AN]; h_AN++){

	Gh_AN = natn[GA_AN][h_AN];
	wan2 = WhatSpecies[Gh_AN];
	tno2 = Spe_Total_CNO[wan2];

	Hmo2[Lidx][h_AN] = (double**)malloc(sizeof(double*)*tno1);
	for (i=0; i<tno1; i++){
	  Hmo2[Lidx][h_AN][i] = (double*)malloc(sizeof(double)*tno2);
	  for (j=0; j<tno2; j++) Hmo2[Lidx][h_AN][i][j] = 0.0;
	}
      }
    }

    RMI0 = (int**)malloc(sizeof(int*)*CWF_Grouped_Atoms_EachNum[gidx]);
    for (Lidx=0; Lidx<CWF_Grouped_Atoms_EachNum[gidx]; Lidx++){
      GA_AN = CWF_Grouped_Atoms[gidx][Lidx];
      RMI0[Lidx] = (int*)malloc(sizeof(int)*(FNAN[GA_AN]+1)*(FNAN[GA_AN]+1));
      for (i=0; i<(FNAN[GA_AN]+1)*(FNAN[GA_AN]+1); i++) RMI0[Lidx][i] = 0;
    }

    /* set Smo, DMmo, and Hmo */

    tnum = 0; 
    for (Lidx=0; Lidx<CWF_Grouped_Atoms_EachNum[gidx]; Lidx++){

      GA_AN = CWF_Grouped_Atoms[gidx][Lidx];
      MA_AN = F_G2M[GA_AN];

      wan1 = WhatSpecies[GA_AN];
      tno1 = Spe_Total_CNO[wan1];

      for (h_AN=0; h_AN<=FNAN[GA_AN]; h_AN++){

	Gh_AN = natn[GA_AN][h_AN];
	wan2 = WhatSpecies[Gh_AN];
	tno2 = Spe_Total_CNO[wan2];

	for (i=0; i<tno1; i++){
	  for (j=0; j<tno2; j++){

	    if (G2ID[GA_AN]==myid && 0<MA_AN){

	      Smo[tnum] = OLP0[MA_AN][h_AN][i][j];

              if (SpinP_switch==0){
	        DMmo[tnum] = CDM[0][MA_AN][h_AN][i][j];
	        Hmo[tnum]  = Hks[0][MA_AN][h_AN][i][j];
	      } 
              else if (SpinP_switch==1){
	        DMmo[tnum] = 0.5*(CDM[0][MA_AN][h_AN][i][j]+CDM[1][MA_AN][h_AN][i][j]);
	        Hmo[tnum]  = 0.5*(Hks[0][MA_AN][h_AN][i][j]+Hks[1][MA_AN][h_AN][i][j]);
	      } 
	    }

	    tnum++; 
	  }
	}
      }
    }    

    /* MPI_Allreduce of Smo and DMmo */

    MPI_Allreduce(MPI_IN_PLACE, &Smo[0],tnum,MPI_DOUBLE,MPI_SUM,mpi_comm_level1);
    MPI_Allreduce(MPI_IN_PLACE, &DMmo[0],tnum,MPI_DOUBLE,MPI_SUM,mpi_comm_level1);
    MPI_Allreduce(MPI_IN_PLACE, &Hmo[0], tnum,MPI_DOUBLE,MPI_SUM,mpi_comm_level1);

    /* copy Smo to Smo2, DMmo to DMmo2, and Hmo to Hmo2 */

    k = 0;
    for (Lidx=0; Lidx<CWF_Grouped_Atoms_EachNum[gidx]; Lidx++){
      GA_AN = CWF_Grouped_Atoms[gidx][Lidx];
      wan1 = WhatSpecies[GA_AN];
      tno1 = Spe_Total_CNO[wan1];

      for (h_AN=0; h_AN<=FNAN[GA_AN]; h_AN++){
	Gh_AN = natn[GA_AN][h_AN];
	wan2 = WhatSpecies[Gh_AN];
	tno2 = Spe_Total_CNO[wan2];

	for (i=0; i<tno1; i++){
	  for (j=0; j<tno2; j++){

            Smo2[Lidx][h_AN][i][j] = Smo[k];
            DMmo2[Lidx][h_AN][i][j] = DMmo[k];
            Hmo2[Lidx][h_AN][i][j] = Hmo[k];

            k++; 
	  }
	}
      }
    }    

    free(DMmo); free(Hmo); free(Smo);

    /* MPI of RMI1 */

    for (Lidx=0; Lidx<CWF_Grouped_Atoms_EachNum[gidx]; Lidx++){

      GA_AN = CWF_Grouped_Atoms[gidx][Lidx];
      MA_AN = F_G2M[GA_AN];

      if (G2ID[GA_AN]==myid && 0<MA_AN){
	for (i=0; i<=FNAN[GA_AN]; i++){
	  for (j=0; j<=FNAN[GA_AN]; j++){
            RMI0[Lidx][i*(FNAN[GA_AN]+1)+j] = RMI1[MA_AN][i][j];
	  }
	}
      }

      else{
	for (i=0; i<=FNAN[GA_AN]; i++){
	  for (j=0; j<=FNAN[GA_AN]; j++){
            RMI0[Lidx][i*(FNAN[GA_AN]+1)+j] = 0;
	  }
	}
      }
    }    

    for (Lidx=0; Lidx<CWF_Grouped_Atoms_EachNum[gidx]; Lidx++){
      GA_AN = CWF_Grouped_Atoms[gidx][Lidx];
      MPI_Allreduce(MPI_IN_PLACE, &RMI0[Lidx][0], (FNAN[GA_AN]+1)*(FNAN[GA_AN]+1), MPI_INT, MPI_SUM, mpi_comm_level1);
    }     

    /* multiplication of DMmos and Smo2 */
    
    for (Lidx1=0; Lidx1<CWF_Grouped_Atoms_EachNum[gidx]; Lidx1++){

      i0 = MP3[Lidx1];
      GA_AN = CWF_Grouped_Atoms[gidx][Lidx1];
      wan1 = WhatSpecies[GA_AN];
      tno1 = Spe_Total_CNO[wan1];

      for (Lidx2=0; Lidx2<CWF_Grouped_Atoms_EachNum[gidx]; Lidx2++){

	j0 = MP3[Lidx2];
	GB_AN = CWF_Grouped_Atoms[gidx][Lidx2];
	wan2 = WhatSpecies[GB_AN];
	tno2 = Spe_Total_CNO[wan2];

	/* find h_AN3 */ 
             
	h_AN3 = -1; 
	for (h_AN=0; h_AN<=FNAN[GA_AN]; h_AN++){
	  GC_AN = natn[GA_AN][h_AN];
	  RnC = ncn[GA_AN][h_AN];
	  l1 = atv_ijk[RnC][1];
	  l2 = atv_ijk[RnC][2];
	  l3 = atv_ijk[RnC][3];

	  if ( GB_AN==GC_AN && l1==0 && l2==0 && l3==0 ){
	    h_AN3 = h_AN;
	  }
	}

	if (0<=h_AN3){

	  for (h_AN=0; h_AN<=FNAN[GA_AN]; h_AN++){

	    GC_AN = natn[GA_AN][h_AN];
	    wan3 = WhatSpecies[GC_AN];
	    tno3 = Spe_Total_CNO[wan3];
	    h_AN2 = RMI0[Lidx1][h_AN3*(FNAN[GA_AN]+1)+h_AN]; 

	    if (0<=h_AN2){

	      for (i=0; i<tno1; i++){
		for (j=0; j<tno2; j++){

		  sum = 0.0;

		  for (k=0; k<tno3; k++){
		    sum += DMmo2[Lidx1][h_AN][i][k]*Smo2[Lidx2][h_AN2][j][k];
		  } /* k */  

		  DMS[(j0+j)*dim + (i0+i)] += sum; 

		} /* i */
	      } /* j */

	    } /* end of if (0<=h_AN2) */
	  } /* h_AN */          
	} /* if (0<=h_AN3) */
      } /* Lidx2 */
    } /* Lidx1 */

    /* calculation of DMS2 = DMS^t x DMS*/

    double alpha =-1.0, beta = 0.0; 
    F77_NAME(dgemm,DGEMM)( "T", "N", &dim, &dim, &dim, &alpha, 
			   DMS, &dim, DMS, &dim, &beta, DMS2, &dim);

    /* diagonalization of DMS2 */

    lapack_dsyevx( dim, dim, DMS2, EVec, EVal );
    for (i=0; i<dim; i++) EVal[i] = sqrt(fabs(EVal[i]));

    /* construct the hamiltonian and overlap matrices for the group of gidx */

    for (Lidx1=0; Lidx1<CWF_Grouped_Atoms_EachNum[gidx]; Lidx1++){

      i0 = MP3[Lidx1];
      GA_AN = CWF_Grouped_Atoms[gidx][Lidx1];
      wan1 = WhatSpecies[GA_AN];
      tno1 = Spe_Total_CNO[wan1];

      for (h_AN=0; h_AN<=FNAN[GA_AN]; h_AN++){

	GC_AN = natn[GA_AN][h_AN];
	RnC = ncn[GA_AN][h_AN];
	l1 = atv_ijk[RnC][1];
	l2 = atv_ijk[RnC][2];
	l3 = atv_ijk[RnC][3];
         
	if (l1==0 && l2==0 && l3==0 ){
            
	  for (Lidx2=0; Lidx2<CWF_Grouped_Atoms_EachNum[gidx]; Lidx2++){

	    j0 = MP3[Lidx2];
	    GB_AN = CWF_Grouped_Atoms[gidx][Lidx2];
	    wan2 = WhatSpecies[GB_AN];
	    tno2 = Spe_Total_CNO[wan2];

	    if (GB_AN==GC_AN){

	      for (i=0; i<tno1; i++){
		for (j=0; j<tno2; j++){

		  OLPgidx[i0+i][j0+j] = Smo2[Lidx1][h_AN][i][j];
		  Hgidx[i0+i][j0+j] = Hmo2[Lidx1][h_AN][i][j];
		}
	      }
	    } 

	  } /* Lidx2 */
	} /* end of if (l1==0 && l2==0 && l3==0 ) */
      } /* h_AN */   
    } /* Lidx1 */           

    for (p=0; p<dim; p++){

      /* calculate <EVec|S|EVec> */

      sum = 0.0;
      for (i=0; i<dim; i++){
	for (j=0; j<dim; j++){
	  sum += OLPgidx[i][j]*EVec[dim*p+i]*EVec[dim*p+j];
	}
      }

      /* normalization of EVec */

      sum = 1.0/sqrt(fabs(sum)); 
       
      for (i=0; i<dim; i++){
	EVec[dim*p+i] *= sum;  
      }        

      /* calculate <EVec|H|EVec> */

      sum = 0.0;
      for (i=0; i<dim; i++){
	for (j=0; j<dim; j++){
	  sum += Hgidx[i][j]*EVec[dim*p+i]*EVec[dim*p+j];
	}
      }

      WI[p] = sum; 

    } /* p */

    /* save the information of MOs into a file. */

    if (myid==Host_ID){

      fprintf(fp_MO_info,"# Group index=%2d\n",gidx+1);
      fprintf(fp_MO_info,"# 1st column: serial number\n");
      fprintf(fp_MO_info,"# 2nd column: on-site energy (eV) relative to chemical potential\n");
      fprintf(fp_MO_info,"# 3rd column: population\n");

      for (p=0; p<dim; p++){
        fprintf(fp_MO_info,"%2d %18.12f %18.12f\n",p+1,(WI[p]-ChemP)*eV2Hartree,EVal[p]);
      }
    }    

    /* select MOs to be used */

    CWF_Guiding_MOs[gidx] = (double**)malloc(sizeof(double*)*Num_CWF_MOs_Group[gidx]);
    for (i=0; i<Num_CWF_MOs_Group[gidx]; i++){
      CWF_Guiding_MOs[gidx][i] = (double*)malloc(sizeof(double)*dim);
    }             

    /* store EVec to select MOs to CWF_Guiding_MOs */

    for (p=0; p<Num_CWF_MOs_Group[gidx]; p++){
      for (j=0; j<dim; j++){
	CWF_Guiding_MOs[gidx][p][j] = EVec[dim*p+j];
      }
    }

    /* freeing of arrays */ 

    free(WI); 
    free(DMS);
    free(DMS2);

    free(EVal);
    free(EVec);

    for (i=0; i<dim; i++){
      free(OLPgidx[i]);
    }
    free(OLPgidx);

    for (i=0; i<dim; i++){
      free(Hgidx[i]);
    }
    free(Hgidx);

    for (Lidx=0; Lidx<CWF_Grouped_Atoms_EachNum[gidx]; Lidx++){
      free(RMI0[Lidx]);
    }
    free(RMI0);

    free(MP3);

    for (Lidx=0; Lidx<CWF_Grouped_Atoms_EachNum[gidx]; Lidx++){

      GA_AN = CWF_Grouped_Atoms[gidx][Lidx];
      wan1 = WhatSpecies[GA_AN];
      tno1 = Spe_Total_CNO[wan1];

      for (h_AN=0; h_AN<=FNAN[GA_AN]; h_AN++){
	for (i=0; i<tno1; i++){
	  free(Smo2[Lidx][h_AN][i]);
	}
        free(Smo2[Lidx][h_AN]);
      }
      free(Smo2[Lidx]);
    }
    free(Smo2);

    for (Lidx=0; Lidx<CWF_Grouped_Atoms_EachNum[gidx]; Lidx++){

      GA_AN = CWF_Grouped_Atoms[gidx][Lidx];
      wan1 = WhatSpecies[GA_AN];
      tno1 = Spe_Total_CNO[wan1];

      for (h_AN=0; h_AN<=FNAN[GA_AN]; h_AN++){
	for (i=0; i<tno1; i++){
	  free(DMmo2[Lidx][h_AN][i]);
	}
	free(DMmo2[Lidx][h_AN]);
      }
      free(DMmo2[Lidx]);
    }
    free(DMmo2);

    for (Lidx=0; Lidx<CWF_Grouped_Atoms_EachNum[gidx]; Lidx++){

      GA_AN = CWF_Grouped_Atoms[gidx][Lidx];
      wan1 = WhatSpecies[GA_AN];
      tno1 = Spe_Total_CNO[wan1];

      for (h_AN=0; h_AN<=FNAN[GA_AN]; h_AN++){
	for (i=0; i<tno1; i++){
	  free(Hmo2[Lidx][h_AN][i]);
	}
	free(Hmo2[Lidx][h_AN]);
      }
      free(Hmo2[Lidx]);
    }
    free(Hmo2);

    /* free the scalack setting */

    /*
    Cfree_blacs_system_handle(bhandle1_MIB);
    Cblacs_gridexit(ictxt1_MIB);
    */

  } /* gidx */  

  /* MPI_Bcast of CWF_Guiding_MOs  */

  for (gidx=0; gidx<Num_CWF_Grouped_Atoms; gidx++){

    dim = 0;
    for (Lidx=0; Lidx<CWF_Grouped_Atoms_EachNum[gidx]; Lidx++){
      GA_AN = CWF_Grouped_Atoms[gidx][Lidx];
      wan1 = WhatSpecies[GA_AN];
      dim += Spe_Total_CNO[wan1];
    }

    for (i=0; i<Num_CWF_MOs_Group[gidx]; i++){
      MPI_Bcast( &CWF_Guiding_MOs[gidx][i][0], dim, MPI_DOUBLE, Host_ID, mpi_comm_level1 );
    }
  }

  /* close a file */
  if (myid==Host_ID) fclose(fp_MO_info);

  /* elapsed time */
  dtime(&TEtime);
  return (TEtime-TStime);
}









void AllocateArrays_Col_QAO()
{
  int Mc_AN,Gc_AN,spin,wan1,i;

  /* allocation of arrays */

  QAO_coes = (double***)malloc(sizeof(double**)*(SpinP_switch+1));
  for (spin=0; spin<(SpinP_switch+1); spin++){
    QAO_coes[spin] = (double**)malloc(sizeof(double*)*(atomnum+1));
    for (Gc_AN=0; Gc_AN<(atomnum+1); Gc_AN++){
   
      if (Gc_AN==0){
        wan1 = 0;
      } 
      else{
        wan1 = WhatSpecies[Gc_AN];
      }

      QAO_coes[spin][Gc_AN] = (double*)malloc(sizeof(double)*(Spe_Total_CNO[wan1]*CWF_Num_predefined[wan1]));
      for (i=0; i<(Spe_Total_CNO[wan1]*CWF_Num_predefined[wan1]); i++){
        QAO_coes[spin][Gc_AN][i] = 0.0;
      }
    }  
  }

  LNAO1_pops = (double***)malloc(sizeof(double**)*(SpinP_switch+1));
  for (spin=0; spin<(SpinP_switch+1); spin++){
    LNAO1_pops[spin] = (double**)malloc(sizeof(double*)*(Matomnum+1));
    for (Mc_AN=0; Mc_AN<(Matomnum+1); Mc_AN++){

      if (Mc_AN==0){
        wan1 = 0;
      } 
      else{
        Gc_AN = M2G[Mc_AN];
        wan1 = WhatSpecies[Gc_AN];
      }

      LNAO1_pops[spin][Mc_AN] = (double*)malloc(sizeof(double)*CWF_Num_predefined[wan1]);
      for (i=0; i<CWF_Num_predefined[wan1]; i++){
        LNAO1_pops[spin][Mc_AN][i] = 0.0;
      }
    }
  }
}

void FreeArrays_Col_QAO()
{
  int Gc_AN,Mc_AN,spin;

  for (spin=0; spin<(SpinP_switch+1); spin++){
    for (Gc_AN=0; Gc_AN<(atomnum+1); Gc_AN++){
      free(QAO_coes[spin][Gc_AN]);
    }  
    free(QAO_coes[spin]);
  }
  free(QAO_coes);

  for (spin=0; spin<(SpinP_switch+1); spin++){
    for (Mc_AN=0; Mc_AN<(Matomnum+1); Mc_AN++){
      free(LNAO1_pops[spin][Mc_AN]);
    }
    free(LNAO1_pops[spin]);
  }
  free(LNAO1_pops);

}



void lapack_dsyevx(int n0, int EVmax, double *A, double *Z, double *EVal)
{
  char *name="lapack_dsyevx";

  char  *JOBZ="V";
  char  *RANGE="I";
  char  *UPLO="L";

  INTEGER n=n0;
  INTEGER LDA=n0;
  double VL,VU; /* dummy */
  INTEGER IL,IU; 
  double ABSTOL=LAPACK_ABSTOL;
  INTEGER M;

  INTEGER LDZ=n;
  INTEGER LWORK;
  double *WORK;
  INTEGER *IWORK;
  INTEGER *IFAIL, INFO;

  int i,j;

  LWORK=n*8;
  WORK=(double*)malloc(sizeof(double)*LWORK);
  IWORK=(INTEGER*)malloc(sizeof(INTEGER)*n*5);
  IFAIL=(INTEGER*)malloc(sizeof(INTEGER)*n);

  IL = 1;
  IU = EVmax;

  F77_NAME(dsyevx,DSYEVX)( JOBZ, RANGE, UPLO, &n, A, &LDA, &VL, &VU, &IL, &IU,
           &ABSTOL, &M, EVal, Z, &LDZ, WORK, &LWORK, IWORK,
           IFAIL, &INFO ); 

  if (INFO>0) {
    printf("\n%s: error in dsyevx_, info=%d\n\n",name,INFO);
  }

  if (INFO<0) {
     printf("%s: info=%d\n",name,INFO);
     exit(10);
  }

  free(IFAIL); 
  free(IWORK); 
  free(WORK); 
}

/**********************************************************************************
  Cluster_DFT_NonCol_CWF.c:

     Cluster_DFT_NonCol_CWF.c is a subroutine to calculate closest Wannier
     functions to a given set of orbials in the cluster calculation 
     based on the non-collinear DFT.

  Log of Cluster_DFT_NonCol.c:

     29/May/2023  Released by T. Ozaki

***********************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <complex.h>
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

/* GPU CuSolver context for Cluster_DFT_NonCol_CWF (added by H.Kawai, Phase D) */
typedef struct {
    int                device_id;
    cublasHandle_t     cublas;
    cusolverDnHandle_t cusolver;
} ClusterNonColCWFCuSolverCtx;

static ClusterNonColCWFCuSolverCtx ClusterNonColCWF_cusolver_ctx = {0};

static void ClusterNonColCWF_CuSolver_Destroy(void)
{
    ClusterNonColCWFCuSolverCtx *ctx = &ClusterNonColCWF_cusolver_ctx;
    if (ctx->cusolver != NULL) wait_cudafunc(cusolverDnDestroy(ctx->cusolver));
    if (ctx->cublas != NULL)   wait_cudafunc(cublasDestroy(ctx->cublas));
    memset(ctx, 0, sizeof(*ctx));
    ctx->device_id = -1;
}

static void ClusterNonColCWF_CuSolver_Init(void)
{
    ClusterNonColCWFCuSolverCtx *ctx = &ClusterNonColCWF_cusolver_ctx;
    int current_device;
    wait_cudafunc(cudaGetDevice(&current_device));
    if (ctx->device_id == current_device && ctx->cublas != NULL) return;
    if (ctx->cublas != NULL) ClusterNonColCWF_CuSolver_Destroy();
    ctx->device_id = current_device;
    wait_cudafunc(cublasCreate(&ctx->cublas));
    wait_cudafunc(cusolverDnCreate(&ctx->cusolver));
}

static void ClusterNonColCWF_GEMMul8Zgemm_OpenACC(cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
                                                  dcomplex const * A, dcomplex const * B, dcomplex * C)
{
    ClusterNonColCWF_CuSolver_Init();
#pragma acc data      present(A[0 : m * k], B[0 : k * n], C[0 : m * n])
#pragma acc host_data use_device(A, B, C)
    {
        cuDoubleComplex const alpha = make_cuDoubleComplex(1.0, 0.0);
        cuDoubleComplex const beta  = make_cuDoubleComplex(0.0, 0.0);
        wait_cudafunc(openmx_gemmul8Zgemm(ClusterNonColCWF_cusolver_ctx.cublas, transa, transb, m, n, k, &alpha,
                                          (cuDoubleComplex const *)A, m, (cuDoubleComplex const *)B, k, &beta,
                                          (cuDoubleComplex *)C, m));
    }
}

double complex ***CWF_Guiding_MOs_NC;
double complex ******CWF_Coef_NC;

void solve_evp_real_( int *n1, int *n2, double *Cs, int *na_rows1, double *a, double *Ss, int *na_rows2, int *nblk, 
                      int *mpi_comm_rows_int, int *mpi_comm_cols_int);

void elpa_solve_evp_real_2stage_double_impl_( int *n1, int *n2, double *Cs, int *na_rows1, double *a, double *Ss, int *na_rows2, 
                                              int *nblk, int *na_cols1, int *mpi_comm_rows_int, int *mpi_comm_cols_int, int *mpiworld);

void solve_evp_complex_( int *n2, int *MaxN, dcomplex *Hs2, int *na_rows2_1, double *a, dcomplex *Cs2, int *na_rows2_2, 
                         int *nblk2, int *mpi_comm_rows_int, int *mpi_comm_cols_int );

void elpa_solve_evp_complex_2stage_double_impl_( int *n2, int *MaxN, dcomplex *Hs2, int *na_rows2_1, double *a, dcomplex *Cs2, 
                                                 int *na_rows2_2, int *nblk2, int *na_cols2, 
                                                 int *mpi_comm_rows_int, int *mpi_comm_cols_int, int *mpiworld );

double Calc_Hybrid_AO_NonCol( double ****OLP0, double *****Hks, double *****CDM );
double Calc_MO_in_Bulk_NonCol( double *****CDM, double *****iCDM );

void lapack_zheev(int n0, dcomplex *A, double *W);
void AllocateArrays_NonCol_QAO2();
void FreeArrays_NonCol_QAO2();

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

void Save_EVec_CWF2W90( char *mode, char *fname, 
                        double *RealEVec, dcomplex *CompEVec, 
                        dcomplex *Full_EVec, 
                        int n, int MinN, int MaxN, int myid2, int numprocs2, 
                        int *is2, int *ie2, MPI_Comm comm );

void Save_EIGENVAL_CWF2W90( char *fname, double *EIGEN, int MinN, int MaxN, int myid2);


static double Calc_CWF_Cluster_NonCol(
				      int myid,
				      int numprocs,
				      int size_H1,
				      int *is2,
				      int *ie2,
				      int *MP,
				      int n2,
				      int MinN,
				      int MaxN,
				      double ****OLP0,
				      double *ko,
				      dcomplex *EVec1,
				      dcomplex *Full_EVec); 


static void Allocate_Free_Cluster_NonCol_CWF( int todo_flag, 
					      int n2,
					      int MaxN,
					      int TNum_CWFs,
					      dcomplex **Cs,
					      dcomplex **Hs, 
					      dcomplex **Vs,
					      dcomplex **Ws,
					      dcomplex **EVs_PAO,
					      dcomplex **WFs );






double Cluster_DFT_NonCol_CWF(
                   int SCF_iter,
                   int SpinP_switch,
                   double *ko,
                   double *****nh,
                   double *****ImNL,
                   double ****CntOLP,
                   double *****CDM,
                   double *****EDM,
                   double Eele0[2], double Eele1[2],
                   int *MP,
		   int *is2,
		   int *ie2,
		   double *Ss,
		   double *Cs,
		   double *rHs11,
		   double *rHs12,
		   double *rHs22,
		   double *iHs11,
		   double *iHs12,
		   double *iHs22,
                   dcomplex *Ss2,
                   dcomplex *Hs2,
                   dcomplex *Cs2,
		   double *DM1,
		   int size_H1, 
                   dcomplex *EVec1,
                   double *Work1)
{
  static int firsttime=1;
  int i,j,l,n,n2,n1,i1,i1s,j1,k1,l1;
  int ii1,jj1,jj2,ki,kj;
  int wan,HOMO0,HOMO1;
  int po,num0,num1;
  int mul,m,wan1,Gc_AN,bcast_flag;
  double time0,lumos;
  int ct_AN,k,wanA,tnoA,wanB,tnoB;
  int GA_AN,Anum,loopN;
  int MA_AN,LB_AN,GB_AN,Bnum,MinN,MaxN;
  double e,b0,b1,e0,e1,weight;
  double TZ,my_sum,sum,sumE,max_x=60.0;
  double My_Eele1[2];
  double Num_State,x,FermiF,Dnum,Dnum2;
  double FermiF2,x2,diffF,out0,out1;
  double dum,ChemP_MAX,ChemP_MIN;
  double TStime,TEtime;
  double FermiEps = 1.0e-13;
  char *Name_Angular[Supported_MaxL+1][2*(Supported_MaxL+1)+1];
  char *Name_Multiple[20];
  char file_EV[YOUSO10] = ".EV";
  FILE *fp_EV;
  char buf[fp_bsize];          /* setvbuf */
  double time1,time2,time3,time4,time5,time6,time7;
  double stime,etime;
  double av_num,tmp;
  int ig,jg,spin;
  int numprocs,myid,ID;
  int ke,ks,nblk_m,nblk_m2;
  int ID0,IDS,IDR,Max_Num_Snd_EV,Max_Num_Rcv_EV;
  int *Num_Snd_EV,*Num_Rcv_EV;
  int *index_Snd_i,*index_Snd_j,*index_Rcv_i,*index_Rcv_j;
  double *EVec_Snd,*EVec_Rcv;
  int ZERO=0,ONE=1,info;
  double Re_alpha = 1.0; double Re_beta = 0.0;
  dcomplex alpha = {1.0,0.0}; dcomplex beta = {0.0,0.0};
  dcomplex *Full_EVec;

  MPI_Comm mpi_comm_rows, mpi_comm_cols;
  int mpi_comm_rows_int,mpi_comm_cols_int;
  MPI_Status stat;
  MPI_Request request;

  /* initialize DM_func, CWF_Charge, and CWF_Energy */
 
  DM_func = 0.0;

  for (spin=0; spin<1; spin++){
    for (i=0; i<TNum_CWFs; i++){
      CWF_Charge[spin][i] = 0.0;
      CWF_Energy[spin][i] = 0.0;
    }
  }

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  MPI_Barrier(mpi_comm_level1);
  dtime(&TStime);

  /* show the message */

  if (myid==0){
    printf("\n<Calculation of Closest Wannier Functions>\n");   
  }

  /*******************************************************
   calculation of atomic, hybrid, or molecular orbitals
  *******************************************************/

  if ( CWF_Guiding_Orbital==1 || CWF_Guiding_Orbital==2 ){
    AllocateArrays_NonCol_QAO2();
    time1 = Calc_Hybrid_AO_NonCol(CntOLP, nh, CDM);
  }

  else if (CWF_Guiding_Orbital==3){
    time1 = Calc_MO_in_Bulk_NonCol(ParDM,iParDM);
  }

  /****************************************************
             calculation of the array size
  ****************************************************/

  n = 0;
  for (i=1; i<=atomnum; i++){
    wanA  = WhatSpecies[i];
    n  = n + Spe_Total_CNO[wanA];
  }
  n2 = 2*n;

  /* GPU dispatch (added by H.Kawai): assign CUDA/OpenACC device when CuSOLVER is requested */
  if (scf_eigen_lib_flag == CuSOLVER && n2 >= GPU_CPU_SWITCH_NUM) {
      set_cuda_default_device_from_local_rank();
      set_openacc_nvidia_device_from_local_rank();
  }

  /****************************************************
                  allocation of arrays
  ****************************************************/

  Num_Snd_EV = (int*)malloc(sizeof(int)*numprocs);
  Num_Rcv_EV = (int*)malloc(sizeof(int)*numprocs);

  /* initialize variables of measuring elapsed time */

  if (measure_time){
    time1 = 0.0;
    time2 = 0.0;
    time3 = 0.0;
    time4 = 0.0;
    time5 = 0.0;
    time6 = 0.0;
    time7 = 0.0;
  }

  /****************************************************
           calculate the total core charge
  ****************************************************/

  TZ = 0.0;
  for (i=1; i<=atomnum; i++){
    wan = WhatSpecies[i];
    TZ += Spe_Core_Charge[wan];
  }

  /***********************************************
     find the number of states to be solved 
  ***********************************************/

  if (CWF_WindowF_switch==0){ // Poly

    /* find MaxN */
    for (i=n2; 1<=i; i--){
      e = ko[i] - ChemP;
      out1 = CWF_disentangling_Erange[3];
      if (e<out1) break;
    }
    MaxN = i; 

    /* find MinN */
    for (i=1; i<=n2; i++){
      e = ko[i] - ChemP;
      out0 = CWF_disentangling_Erange[0];
      if (out0<e) break;
    }
    MinN = i;

    TNum_PAOs = n;
    TNum_PAOs2 = n2;
    Num_KS_states_CWF = MaxN - MinN + 1;

    if (myid==Host_ID){
      printf("  MinN=%2d MaxN=%2d (MaxN-MinN+1)=%2d TNum_CWFs=%2d\n",
	     MinN,MaxN,MaxN-MinN+1,TNum_CWFs);
    }

    if (MaxN<MinN){
      if (myid==Host_ID){
	printf("Could not find any state to be included.\n"); 
	printf("Parameters for CWF disentangling must be improper.\n"); 
      }
      MPI_Finalize();
      exit(0);    
    }

    if (Num_KS_states_CWF<TNum_CWFs){
      if (myid==Host_ID){
	printf("(MaxN-MinN+1) should be larger than TNum_CWFs.\n"); 
	printf("Parameters for CWF disentangling must be improper.\n"); 
      }
      MPI_Finalize();
      exit(0);    
    }
  }

  else if (CWF_WindowF_switch==1){ // Fermi
    TNum_PAOs = n;
    TNum_PAOs2 = n2;
    MinN = 1;
    MaxN = n2;
    Num_KS_states_CWF = MaxN;
  }

  /****************************************************
         find the numbers of partions for MPI
  ****************************************************/

  if ( numprocs<=MaxN ){

    av_num = (double)MaxN/(double)numprocs;

    for (ID=0; ID<numprocs; ID++){
      is2[ID] = (int)(av_num*(double)ID) + 1; 
      ie2[ID] = (int)(av_num*(double)(ID+1)); 
    }

    is2[0] = 1;
    ie2[numprocs-1] = MaxN; 
  }

  else{
    for (ID=0; ID<MaxN; ID++){
      is2[ID] = ID + 1; 
      ie2[ID] = ID + 1;
    }
    for (ID=MaxN; ID<numprocs; ID++){
      is2[ID] = 1;
      ie2[ID] = 0;
    }
  }

  /* making data structure of MPI communicaition for eigenvectors */

  for (ID=0; ID<numprocs; ID++){
    Num_Snd_EV[ID] = 0;
    Num_Rcv_EV[ID] = 0;
  }

  for (i=0; i<na_rows2; i++){

    ig = np_rows2*nblk2*((i)/nblk2) + (i)%nblk2 + ((np_rows2+my_prow2)%np_rows2)*nblk2 + 1;

    po = 0;
    for (ID=0; ID<numprocs; ID++){
      if (is2[ID]<=ig && ig <=ie2[ID]){
        po = 1;
        ID0 = ID;
        break;
      }
    }

    if (po==1) Num_Snd_EV[ID0] += na_cols2;
  }

  for (ID=0; ID<numprocs; ID++){
    IDS = (myid + ID) % numprocs;
    IDR = (myid - ID + numprocs) % numprocs;
    if (ID!=0){
      MPI_Isend(&Num_Snd_EV[IDS], 1, MPI_INT, IDS, 999, mpi_comm_level1, &request);
      MPI_Recv(&Num_Rcv_EV[IDR], 1, MPI_INT, IDR, 999, mpi_comm_level1, &stat);
      MPI_Wait(&request,&stat);
    }
    else{
      Num_Rcv_EV[IDR] = Num_Snd_EV[IDS];
    }
  }

  Max_Num_Snd_EV = 0;
  Max_Num_Rcv_EV = 0;
  for (ID=0; ID<numprocs; ID++){
    if (Max_Num_Snd_EV<Num_Snd_EV[ID]) Max_Num_Snd_EV = Num_Snd_EV[ID];
    if (Max_Num_Rcv_EV<Num_Rcv_EV[ID]) Max_Num_Rcv_EV = Num_Rcv_EV[ID];
  }  

  Max_Num_Snd_EV++;
  Max_Num_Rcv_EV++;

  index_Snd_i = (int*)malloc(sizeof(int)*Max_Num_Snd_EV);
  index_Snd_j = (int*)malloc(sizeof(int)*Max_Num_Snd_EV);
  EVec_Snd = (double*)malloc(sizeof(double)*Max_Num_Snd_EV*2);
  index_Rcv_i = (int*)malloc(sizeof(int)*Max_Num_Rcv_EV);
  index_Rcv_j = (int*)malloc(sizeof(int)*Max_Num_Rcv_EV);
  EVec_Rcv = (double*)malloc(sizeof(double)*Max_Num_Rcv_EV*2);

  if (cwf2w90_output_flag){
    Full_EVec = (dcomplex*)malloc(sizeof(dcomplex)*MaxN*n2);
  }  
  else {
    Full_EVec = (dcomplex*)malloc(sizeof(dcomplex)*1);
  }

  /* print memory size */

  if (firsttime && memoryusage_fileout){
    PrintMemory("Cluster_DFT_NonCol: Num_Snd_EV",sizeof(int)*numprocs,NULL);
    PrintMemory("Cluster_DFT_NonCol: Num_Ecv_EV",sizeof(int)*numprocs,NULL);
    PrintMemory("Cluster_DFT_NonCol: index_Snd_i",sizeof(int)*Max_Num_Snd_EV,NULL);
    PrintMemory("Cluster_DFT_NonCol: index_Snd_j",sizeof(int)*Max_Num_Snd_EV,NULL);
    PrintMemory("Cluster_DFT_NonCol: index_Rcv_i",sizeof(int)*Max_Num_Rcv_EV,NULL);
    PrintMemory("Cluster_DFT_NonCol: index_Rcv_j",sizeof(int)*Max_Num_Rcv_EV,NULL);
    PrintMemory("Cluster_DFT_NonCol: EVec_Snd",sizeof(double)*Max_Num_Snd_EV*2,NULL);
    PrintMemory("Cluster_DFT_NonCol: EVec_Rcv",sizeof(double)*Max_Num_Rcv_EV*2,NULL);
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

  /****************************************************
            diagonalize the overlap matrix     
  ****************************************************/

  if (SCF_iter==1){

    if (measure_time) dtime(&stime);

    Overlap_Cluster_Ss(CntOLP,Cs,MP,0);

    MPI_Comm_split(mpi_comm_level1,my_pcol,my_prow,&mpi_comm_rows);
    MPI_Comm_split(mpi_comm_level1,my_prow,my_pcol,&mpi_comm_cols);

    mpi_comm_rows_int = MPI_Comm_c2f(mpi_comm_rows);
    mpi_comm_cols_int = MPI_Comm_c2f(mpi_comm_cols);

    /* diagonalize Cs */

    if (scf_eigen_lib_flag==1){
      F77_NAME(solve_evp_real,SOLVE_EVP_REAL)( &n, &n, Cs, &na_rows, &ko[1], Ss, &na_rows, &nblk, 
                                               &mpi_comm_rows_int, &mpi_comm_cols_int );
    }
    else if (scf_eigen_lib_flag==CuSOLVER && GPU_CPU_SWITCH_NUM<=n2 && na_rows==n && na_cols==n){
      OpenMX_CuSolver_DenseDsyevx_1based(Cs,Ss,ko,n,n,"Cluster_DFT_NonCol_CWF overlap CuSOLVER");
    }

    else if (scf_eigen_lib_flag==2 || scf_eigen_lib_flag==CuSOLVER){

#ifndef kcomp

      int mpiworld;
      mpiworld = MPI_Comm_c2f(mpi_comm_level1);

      F77_NAME(elpa_solve_evp_real_2stage_double_impl,ELPA_SOLVE_EVP_REAL_2STAGE_DOUBLE_IMPL)
	( &n, &n, Cs, &na_rows, &ko[1], Ss, &na_rows, &nblk, &na_cols, 
          &mpi_comm_rows_int, &mpi_comm_cols_int, &mpiworld ); 
#endif
    }

    MPI_Comm_free(&mpi_comm_rows);
    MPI_Comm_free(&mpi_comm_cols);

    /* print to the standard output */

    if (2<=level_stdout){
      for (l=1; l<=n; l++){
	printf("  Eigenvalues of OLP  %2d  %18.15f\n",l,ko[l]);
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
	Ss[j*na_rows+i] = Ss[j*na_rows+i]*ko[jg];
      }
    }

    /* make Ss2 */

    Overlap_Cluster_NC_Ss2( Ss, Ss2 );

    if (measure_time){
      dtime(&etime);
      time1 += etime - stime; 
    }
  }

  /****************************************************************************
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
  *****************************************************************************/

  if (measure_time) dtime(&stime);

  /* set rHs and iHs */

  Hamiltonian_Cluster_Hs(nh[0],rHs11,MP,0,0);
  Hamiltonian_Cluster_Hs(nh[1],rHs22,MP,0,0);
  Hamiltonian_Cluster_Hs(nh[2],rHs12,MP,0,0);
  Hamiltonian_Cluster_Hs(nh[3],iHs12,MP,0,0);

  Hamiltonian_Cluster_Hs(ImNL[0],iHs11,MP,0,0);
  Hamiltonian_Cluster_Hs(ImNL[1],iHs22,MP,0,0);
  Hamiltonian_Cluster_Hs(ImNL[2],Cs,MP,0,0);

  for (i=0; i<na_rows*na_cols; i++) iHs12[i] += Cs[i];

  /* S^t x rHs11 x S */

  for (i=0; i<na_rows*na_cols; i++) Cs[i] = 0.0;

  Cblacs_barrier(ictxt1,"A");
  F77_NAME(pdgemm,PDGEMM)("N","N",&n,&n,&n,&Re_alpha,rHs11,&ONE,&ONE,descH,Ss,
                          &ONE,&ONE,descS,&Re_beta,Cs,&ONE,&ONE,descC);

  for (i=0; i<na_rows*na_cols; i++) rHs11[i] = 0.0;
  
  Cblacs_barrier(ictxt1,"C");
  F77_NAME(pdgemm,PDGEMM)("T","N",&n,&n,&n,&Re_alpha,Ss,&ONE,&ONE,descS,Cs,
                          &ONE,&ONE,descC,&Re_beta,rHs11,&ONE,&ONE,descH);

  /* S^t x rHs12 x S */

  for (i=0; i<na_rows*na_cols; i++) Cs[i] = 0.0;

  Cblacs_barrier(ictxt1,"A");
  F77_NAME(pdgemm,PDGEMM)("N","N",&n,&n,&n,&Re_alpha,rHs12,&ONE,&ONE,descH,Ss,
                          &ONE,&ONE,descS,&Re_beta,Cs,&ONE,&ONE,descC);

  for (i=0; i<na_rows*na_cols; i++) rHs12[i] = 0.0;

  Cblacs_barrier(ictxt1,"C");
  F77_NAME(pdgemm,PDGEMM)("T","N",&n,&n,&n,&Re_alpha,Ss,&ONE,&ONE,descS,Cs,
                          &ONE,&ONE,descC,&Re_beta,rHs12,&ONE,&ONE,descH);

  /* S^t x rHs22 x S */

  for (i=0; i<na_rows*na_cols; i++) Cs[i] = 0.0;

  Cblacs_barrier(ictxt1,"A");
  F77_NAME(pdgemm,PDGEMM)("N","N",&n,&n,&n,&Re_alpha,rHs22,&ONE,&ONE,descH,Ss,
                          &ONE,&ONE,descS,&Re_beta,Cs,&ONE,&ONE,descC);

  for (i=0; i<na_rows*na_cols; i++) rHs22[i] = 0.0;
  
  Cblacs_barrier(ictxt1,"C");
  F77_NAME(pdgemm,PDGEMM)("T","N",&n,&n,&n,&Re_alpha,Ss,&ONE,&ONE,descS,Cs,
                          &ONE,&ONE,descC,&Re_beta,rHs22,&ONE,&ONE,descH);

  /* S^t x iHs11 x S */

  for (i=0; i<na_rows*na_cols; i++) Cs[i] = 0.0;

  Cblacs_barrier(ictxt1,"A");
  F77_NAME(pdgemm,PDGEMM)("N","N",&n,&n,&n,&Re_alpha,iHs11,&ONE,&ONE,descH,Ss,
                          &ONE,&ONE,descS,&Re_beta,Cs,&ONE,&ONE,descC);

  for (i=0; i<na_rows*na_cols; i++) iHs11[i] = 0.0;

  Cblacs_barrier(ictxt1,"C");
  F77_NAME(pdgemm,PDGEMM)("T","N",&n,&n,&n,&Re_alpha,Ss,&ONE,&ONE,descS,Cs,
                          &ONE,&ONE,descC,&Re_beta,iHs11,&ONE,&ONE,descH);

  /* S^t x iHs12 x S */

  for (i=0; i<na_rows*na_cols; i++) Cs[i] = 0.0;

  Cblacs_barrier(ictxt1,"A");
  F77_NAME(pdgemm,PDGEMM)("N","N",&n,&n,&n,&Re_alpha,iHs12,&ONE,&ONE,descH,Ss,
                          &ONE,&ONE,descS,&Re_beta,Cs,&ONE,&ONE,descC);

  for (i=0; i<na_rows*na_cols; i++) iHs12[i] = 0.0;
  
  Cblacs_barrier(ictxt1,"C");
  F77_NAME(pdgemm,PDGEMM)("T","N",&n,&n,&n,&Re_alpha,Ss,&ONE,&ONE,descS,Cs,
                          &ONE,&ONE,descC,&Re_beta,iHs12,&ONE,&ONE,descH);

  /* S^t x iHs22 x S */

  for (i=0; i<na_rows*na_cols; i++) Cs[i] = 0.0;

  Cblacs_barrier(ictxt1,"A");
  F77_NAME(pdgemm,PDGEMM)("N","N",&n,&n,&n,&Re_alpha,iHs22,&ONE,&ONE,descH,Ss,
                          &ONE,&ONE,descS,&Re_beta,Cs,&ONE,&ONE,descC);

  for (i=0; i<na_rows*na_cols; i++) iHs22[i] = 0.0;
  
  Cblacs_barrier(ictxt1,"C");
  F77_NAME(pdgemm,PDGEMM)("T","N",&n,&n,&n,&Re_alpha,Ss,&ONE,&ONE,descS,Cs,
                          &ONE,&ONE,descC,&Re_beta,iHs22,&ONE,&ONE,descH);

  if (measure_time){
    dtime(&etime);
    time2 += etime - stime;
  }

  /****************************************************
             diagonalize the transformed H
  ****************************************************/

  if (measure_time) dtime(&stime);

  Hamiltonian_Cluster_NC_Hs2( rHs11, rHs22, rHs12, iHs11, iHs22, iHs12, Hs2 );

  MPI_Comm_split(mpi_comm_level1,my_pcol2,my_prow2,&mpi_comm_rows);
  MPI_Comm_split(mpi_comm_level1,my_prow2,my_pcol2,&mpi_comm_cols);

  mpi_comm_rows_int = MPI_Comm_c2f(mpi_comm_rows);
  mpi_comm_cols_int = MPI_Comm_c2f(mpi_comm_cols);

  if (scf_eigen_lib_flag==1){
    F77_NAME(solve_evp_complex,SOLVE_EVP_COMPLEX)( &n2, &MaxN, Hs2, &na_rows2, &ko[1], Cs2, &na_rows2, 
                                                   &nblk2, &mpi_comm_rows_int, &mpi_comm_cols_int );
  }
  else if (scf_eigen_lib_flag==CuSOLVER && GPU_CPU_SWITCH_NUM<=n2 && na_rows2==n2 && na_cols2==n2){
    OpenMX_CuSolver_DenseZheevx_1based(Hs2,Cs2,ko,n2,MaxN,"Cluster_DFT_NonCol_CWF Hamiltonian CuSOLVER");
  }

  else if (scf_eigen_lib_flag==2 || scf_eigen_lib_flag==CuSOLVER){

#ifndef kcomp

    int mpiworld;
    mpiworld = MPI_Comm_c2f(mpi_comm_level1);

    F77_NAME(elpa_solve_evp_complex_2stage_double_impl,ELPA_SOLVE_EVP_COMPLEX_2STAGE_DOUBLE_IMPL)
      ( &n2, &MaxN, Hs2, &na_rows2, &ko[1], Cs2, &na_rows2, &nblk2, &na_cols2, 
        &mpi_comm_rows_int, &mpi_comm_cols_int, &mpiworld );

#endif
  }

  MPI_Comm_free(&mpi_comm_rows);
  MPI_Comm_free(&mpi_comm_cols);
  
  if (2<=level_stdout){
    for (i1=1; i1<=MaxN; i1++){
      printf("  Eigenvalues of Kohn-Sham %2d  %15.12f\n", i1,ko[i1]);
    }
  }

  if (measure_time){
    dtime(&etime);
    time3 += etime - stime;
  }

  /****************************************************
      Transformation to the original eigenvectors.
      JRCAT NOTE 244P  C = U * lambda^{-1/2} * D
  ****************************************************/

  if (measure_time) dtime(&stime);

  for(k=0; k<na_rows2*na_cols2; k++){
    Hs2[k].r = 0.0;
    Hs2[k].i = 0.0;
  }

  Cblacs_barrier(ictxt1_2,"A");
  F77_NAME(pzgemm,PZGEMM)("T","T",&n2,&n2,&n2,&alpha,Cs2,&ONE,&ONE,descC2,Ss2,
                          &ONE,&ONE,descS2,&beta,Hs2,&ONE,&ONE,descH2);

  /* MPI communications of Hs2 */

  for (ID=0; ID<numprocs; ID++){
    
    IDS = (myid + ID) % numprocs;
    IDR = (myid - ID + numprocs) % numprocs;

    k = 0;
    for(i=0; i<na_rows2; i++){
      ig = np_rows2*nblk2*((i)/nblk2) + (i)%nblk2 + ((np_rows2+my_prow2)%np_rows2)*nblk2 + 1;

      if (is2[IDS]<=ig && ig <=ie2[IDS]){

        for (j=0; j<na_cols2; j++){
          jg = np_cols2*nblk2*((j)/nblk2) + (j)%nblk2 + ((np_cols2+my_pcol2)%np_cols2)*nblk2 + 1;
 
          index_Snd_i[k] = ig;
          index_Snd_j[k] = jg;
          EVec_Snd[2*k  ] = Hs2[j*na_rows2+i].r;
          EVec_Snd[2*k+1] = Hs2[j*na_rows2+i].i;
          k++; 
	}
      }
    }

    if (ID!=0){

      if (Num_Snd_EV[IDS]!=0){
        MPI_Isend(index_Snd_i, Num_Snd_EV[IDS], MPI_INT, IDS, 999, mpi_comm_level1, &request);
      }
      if (Num_Rcv_EV[IDR]!=0){
        MPI_Recv(index_Rcv_i, Num_Rcv_EV[IDR], MPI_INT, IDR, 999, mpi_comm_level1, &stat);
      }
      if (Num_Snd_EV[IDS]!=0){
        MPI_Wait(&request,&stat);
      }

      if (Num_Snd_EV[IDS]!=0){
        MPI_Isend(index_Snd_j, Num_Snd_EV[IDS], MPI_INT, IDS, 999, mpi_comm_level1, &request);
      }
      if (Num_Rcv_EV[IDR]!=0){
        MPI_Recv(index_Rcv_j, Num_Rcv_EV[IDR], MPI_INT, IDR, 999, mpi_comm_level1, &stat);
      }
      if (Num_Snd_EV[IDS]!=0){
        MPI_Wait(&request,&stat);
      }

      if (Num_Snd_EV[IDS]!=0){
        MPI_Isend(EVec_Snd, Num_Snd_EV[IDS]*2, MPI_DOUBLE, IDS, 999, mpi_comm_level1, &request);
      }
      if (Num_Rcv_EV[IDR]!=0){
        MPI_Recv(EVec_Rcv, Num_Rcv_EV[IDR]*2, MPI_DOUBLE, IDR, 999, mpi_comm_level1, &stat);
      }
      if (Num_Snd_EV[IDS]!=0){
        MPI_Wait(&request,&stat);
      }
    }
    else{
      for(k=0; k<Num_Snd_EV[IDS]; k++){
        index_Rcv_i[k] = index_Snd_i[k];
        index_Rcv_j[k] = index_Snd_j[k];
        EVec_Rcv[2*k  ] = EVec_Snd[2*k  ];
        EVec_Rcv[2*k+1] = EVec_Snd[2*k+1];
      } 
    }

    for(k=0; k<Num_Rcv_EV[IDR]; k++){

      ig = index_Rcv_i[k];
      jg = index_Rcv_j[k];

      m = (ig-is2[myid])*n2 + jg - 1;
      EVec1[m].r = EVec_Rcv[2*k  ];
      EVec1[m].i = EVec_Rcv[2*k+1];

    }
  } // ID

  /***********************************
          if cwf2w90_output_flag==1,  
          save EVec1 in binary format 
  ***********************************/

  if (cwf2w90_output_flag){

    char fname_EVec[YOUSO10];
    double tmp;

    sprintf(fname_EVec,"%s%s_evec/evec_%d",filepath,filename,0);
    Save_EVec_CWF2W90( "comp", fname_EVec, &tmp, EVec1, Full_EVec, n2, MinN, MaxN, 
		       myid, numprocs, is2, ie2, mpi_comm_level1 ); 

    char fname_EVal[YOUSO10];

    sprintf(fname_EVal,"%s%s_evec/eval_%d",filepath,filename,0);
    Save_EIGENVAL_CWF2W90(fname_EVal, ko, MinN, MaxN, myid);
  }  

  if (measure_time){
    dtime(&etime);
    time4 += etime - stime;
  }

  /*********************************************** 
              calculation of CWFs
  ***********************************************/

  time6 += Calc_CWF_Cluster_NonCol( myid,numprocs,size_H1,is2,ie2,MP,
                                    n2,MinN,MaxN,CntOLP,ko,EVec1,Full_EVec );

  if (measure_time){
    printf("Cluster_DFT_NonCol_CWF myid=%2d time1=%7.3f time2=%7.3f time3=%7.3f time4=%7.3f time5=%7.3f time6=%7.3f time7=%7.3f\n",
            myid,time1,time2,time3,time4,time5,time6,time7);fflush(stdout); 
  }

  /****************************************************
                          Free
  ****************************************************/

  free(Full_EVec);

  free(Num_Snd_EV);
  free(Num_Rcv_EV);

  free(index_Snd_i);
  free(index_Snd_j);
  free(EVec_Snd);
  free(index_Rcv_i);
  free(index_Rcv_j);
  free(EVec_Rcv);

  /* freeing of arrays */

  if ( CWF_Guiding_Orbital==1 || CWF_Guiding_Orbital==2 ){
    FreeArrays_NonCol_QAO2();
  }

  /* for elapsed time */

  MPI_Barrier(mpi_comm_level1);
  dtime(&TEtime);
  time0 = TEtime - TStime;
  return time0;
}



double Calc_CWF_Cluster_NonCol(
    int myid,
    int numprocs,
    int size_H1,
    int *is2,
    int *ie2,
    int *MP,
    int n2,
    int MinN,
    int MaxN,
    double ****OLP0,
    double *ko,
    dcomplex *EVec1,
    dcomplex *Full_EVec) 
{
  int Gc_AN,GB_AN,Mc_AN,h_AN,Gh_AN,wanB,wan1,tno1,wan2,tno2,TNum_CWFs;
  int i,i1,j,l,k,k1,ig,jg,m1,NumCWFs,num,p,q,ID,ID1,mmin,mmax,m;
  int EVec1_size,Max_EVec1_size,sp,*int_data;
  int gidx,Lidx,GA_AN,dim,i0,pnum,idx;
  int *MP2,*MP3,mpi_info[5],fp_Hop_ok=0;
  char fname[YOUSO10];
  FILE *fp_Hop;
  double sum,max_x=60.0,x,FermiF,tmp,w1=1.0e-10;
  double in0,in1,out0,out1,e,weight,dif,b0,b1,e0,e1; 
  double complex *InProd,*InProd_BasisFunc,*C2,**S1,*TmpEVec1;
  double complex *Hop;
  dcomplex *Cs,*Hs,*Vs,*Ws,*WFs,*EVs_PAO,ctmp0,*work;
  double complex ctmp,csum1,csum2;
  double *rwork,*sv;
  int num_zero_sv=0,rank,lwork,info;
  double Sum_Charges,Sum_Energies;
  double stime,etime;
  int ZERO=0,ONE=1;
  dcomplex alpha = {1.0,0.0}; dcomplex beta = {0.0,0.0};
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
      TNum_CWFs += 2*CWF_Num_predefined[wan1];
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

  Allocate_Free_Cluster_NonCol_CWF( 1, n2, MaxN, TNum_CWFs, &Cs, &Hs, &Vs, &Ws, &EVs_PAO, &WFs );

  /* find the maximum size of EVec1 */

  EVec1_size = 0;
  for (ID=0; ID<numprocs; ID++){
    if ( EVec1_size < (n2*(ie2[ID]-is2[ID]+1)) ) EVec1_size = n2*(ie2[ID]-is2[ID]+1);
  }
  MPI_Allreduce(&EVec1_size,&Max_EVec1_size,1,MPI_INT,MPI_MAX,mpi_comm_level1);

  /* allocation of arrays */

  TmpEVec1 = (double complex*)malloc(sizeof(double complex)*Max_EVec1_size);
  InProd = (double complex*)malloc(sizeof(double complex)*Max_EVec1_size);
  InProd_BasisFunc = (double complex*)malloc(sizeof(double complex)*Max_EVec1_size);

  C2 = (double complex*)malloc(sizeof(double complex)*2*List_YOUSO[7]);
  S1 = (double complex**)malloc(sizeof(double complex*)*2*List_YOUSO[7]);
  for (i=0; i<2*List_YOUSO[7]; i++) S1[i] = (double complex*)malloc(sizeof(double complex)*2*List_YOUSO[7]);

  Hop = (double complex*)malloc(sizeof(double complex)*(TNum_CWFs*TNum_CWFs));
  for (p=0; p<(TNum_CWFs*TNum_CWFs); p++) Hop[p] = 0.0 + 0.0*I;

  sv = (double*)malloc(sizeof(double)*(MaxN+1));
  rwork = (double*)malloc(sizeof(double)*(1+4*MaxN));

  /* get size of work and allocate work */

  lwork = -1;
  F77_NAME(pzgesvd,PZGESVD)( "V", "V",
			     &MaxN, &TNum_CWFs,
			     Cs, &ONE, &ONE, desc_CWF3,
			     sv,
			     Ws, &ONE, &ONE, desc_CWF3,
			     Vs, &ONE, &ONE, desc_CWF1,
			     &ctmp0, 
			     &lwork,
                             rwork,
			     &info);

  lwork = (int)ctmp0.r + 1;
  work = (dcomplex*)malloc(sizeof(dcomplex)*lwork);

  /***************************************************
   calculation of <Bloch function|localized orbital>
  ***************************************************/

  /*
  printf("WWW myid=%2d  %2d %2d  %2d %2d\n",
          myid,na_rows_CWF3,na_cols_CWF3,na_rows_CWF4,na_cols_CWF4);
  MPI_Finalize();
  exit(0);
  */

  /* initialize Cs and Hs */

  for (i=0; i<na_rows_CWF4*na_cols_CWF4; i++){
    Cs[i] = Complex(0.0,0.0);
    Hs[i] = Complex(0.0,0.0);
    Ws[i] = Complex(0.0,0.0);
    Vs[i] = Complex(0.0,0.0);
  }

  /*
  if (myid==0){
  for (i=0; i<na_rows_CWF3; i++){
    for (j=0; j<na_cols_CWF3; j++){
      printf("ABC0 myid=%2d i=%2d j=%2d  %15.12f %15.12f\n",
	     myid,i,j,Cs[j*na_rows_CWF3+i].r,Cs[j*na_rows_CWF3+i].i);fflush(stdout);  
    }
  }  
  }

  MPI_Finalize();
  exit(0);
  */

  /* In the loop of ID, the index of spin is changed regardless of myworld1. */

  for (ID=0; ID<numprocs; ID++){

    if (ID==myid){
      ID1 = myid;
      num = n2*(ie2[ID1]-is2[ID1]+1);
      mpi_info[0] = ID1;
      mpi_info[1] = num;
      mpi_info[2] = is2[ID1];
      mpi_info[3] = ie2[ID1];
    }

    /* MPI_Bcast of num */

    MPI_Bcast( &mpi_info[0], 4, MPI_INT, ID, mpi_comm_level1 );

    /* set parameters */

    ID1  = mpi_info[0];
    num  = mpi_info[1];
    mmin = mpi_info[2];
    mmax = mpi_info[3];

    /* initialize InProd and InProd_BasisFunc */

    for (p=0; p<Max_EVec1_size; p++){
      InProd[p] = 0.0 + I*0.0;
      InProd_BasisFunc[p] = 0.0 + I*0.0;
    }

    if ( num!=0 ){

      /* MPI_Bcast of EVec1 */

      if ( ID==myid ){
	for (i=0; i<num; i++){
	  TmpEVec1[i] = EVec1[i].r + I*EVec1[i].i;
	}
      }

      MPI_Bcast( &TmpEVec1[0], num, MPI_C_DOUBLE_COMPLEX, ID, mpi_comm_level1 );

      /* store TmpEVec1 into EVs_PAO in the block cyclic form with n2 x MaxN.  */

      for (j=0; j<na_cols_CWF4; j++){

	m1 = np_cols_CWF4*nblk_CWF4*((j)/nblk_CWF4) + (j)%nblk_CWF4 
	  + ((np_cols_CWF4+my_pcol_CWF4)%np_cols_CWF4)*nblk_CWF4;

	if ((mmin-1)<=m1 && m1<=(mmax-1)){ 

	  for (i=0; i<na_rows_CWF4; i++){

	    ig = np_rows_CWF4*nblk_CWF4*((i)/nblk_CWF4) + (i)%nblk_CWF4
	      + ((np_rows_CWF4+my_prow_CWF4)%np_rows_CWF4)*nblk_CWF4;

	    EVs_PAO[j*na_rows_CWF4+i].r = creal(TmpEVec1[ (m1-mmin+1)*n2 + ig ]);
	    EVs_PAO[j*na_rows_CWF4+i].i = cimag(TmpEVec1[ (m1-mmin+1)*n2 + ig ]);

	  } // i

	} // if ((mmin-1)<=m1 && m1<=(mmax-1))
      } // j

      /* calculate <Bloch functions|basis functions> */

      for (m=0; m<=(mmax-mmin); m++){  // loop for KS index 
        
	for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

	  Gc_AN = M2G[Mc_AN];
	  wan1 = WhatSpecies[Gc_AN];
	  tno1 = Spe_Total_CNO[wan1];
      
	  for (i=0; i<tno1; i++){

	    csum1 = 0.0 + I*0.0;
	    csum2 = 0.0 + I*0.0;

	    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	      Gh_AN = natn[Gc_AN][h_AN];
	      wan2 = WhatSpecies[Gh_AN];
	      tno2 = Spe_Total_CNO[wan2];

	      for (j=0; j<tno2; j++){

		m1 = m*n2 + MP[Gh_AN] - 1 + j;
		C2[j] = TmpEVec1[m1];
		
		m1 = m*n2 + MP[Gh_AN] - 1 + j + n2/2;
		C2[tno2+j] = TmpEVec1[m1];
	      }      

	      for (j=0; j<tno2; j++){
		csum1 += (OLP0[Mc_AN][h_AN][i][j]+0.0*I)*C2[j];
		csum2 += (OLP0[Mc_AN][h_AN][i][j]+0.0*I)*C2[tno2+j];

                /*
                if (Mc_AN==1 && i==0){
                  printf("WWW1 j=%2d h_AN=%2d OLP=%8.4f C2=%8.4f %8.4f\n",j,h_AN,OLP0[Mc_AN][h_AN][i][j],creal(C2[j]),creal(C2[tno2+j]));
		}
		*/

	      } // j

	    } // h_AN

	    /* store <Bloch functions|basis functions> */

	    p = m*n2 + MP[Gc_AN] - 1 + i;
	    InProd_BasisFunc[p] = csum1;

	    p = m*n2 + MP[Gc_AN] - 1 + i + n2/2;
	    InProd_BasisFunc[p] = csum2;

	  } // i
	} // Mc_AN 
      } // m      

      /* MPI_Allreduce of InProd_BasisFunc */       

      MPI_Allreduce( MPI_IN_PLACE, &InProd_BasisFunc[0], (mmax-mmin+1)*n2, MPI_C_DOUBLE_COMPLEX, MPI_SUM, mpi_comm_level1 );

      /*
      for (i=0; i<(mmax-mmin+1)*n2; i++){
        printf("VVV1 mmin=%2d mmax=%2d n2=%2d i=%3d %8.4f %8.4f\n",mmin,mmax,n2,i,creal(InProd_BasisFunc[i]),cimag(InProd_BasisFunc[i])); 
      } 
      */

      /* AO and HO case: calculate <Bloch functions|guiding functions> */

      if (CWF_Guiding_Orbital==1 || CWF_Guiding_Orbital==2){

        for (m=0; m<=(mmax-mmin); m++){  // loop for KS index 

	  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

	    Gc_AN = M2G[Mc_AN];
	    wan1 = WhatSpecies[Gc_AN];
	    tno1 = Spe_Total_CNO[wan1];
	    NumCWFs = CWF_Num_predefined[wan1];
      
	    for (i=0; i<2*NumCWFs; i++){

              csum1 = 0.0 + I*0.0;
              csum2 = 0.0 + I*0.0;

	      for (l=0; l<tno1; l++){

		p = m*n2 + MP[Gc_AN] - 1 + l;  
                ctmp = QAO2_coes[Gc_AN][2*tno1*i+l].r + I*QAO2_coes[Gc_AN][2*tno1*i+l].i;
		csum1 += InProd_BasisFunc[p]*ctmp;

		p = m*n2 + MP[Gc_AN] - 1 + l + n2/2;
                ctmp = QAO2_coes[Gc_AN][2*tno1*i+l+tno1].r + I*QAO2_coes[Gc_AN][2*tno1*i+l+tno1].i;
		csum2 += InProd_BasisFunc[p]*ctmp;

	      } // l

   	      q = m*TNum_CWFs + MP2[Gc_AN] + i;
 	      InProd[q] = csum1 + csum2;

	    } // i
	  } // Mc_AN 
	} // m

	/*
	for (i=0; i<(mmax-mmin+1)*TNum_CWFs; i++){
          printf("VVV2 mmin=%2d mmax=%2d n2=%2d i=%3d %8.4f %8.4f\n",mmin,mmax,n2,i,creal(InProd[i]),cimag(InProd[i])); 
        } 
	*/ 

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

  	      csum1 = 0.0 + I*0.0;
	      csum2 = 0.0 + I*0.0;

              for (Lidx=0; Lidx<CWF_Grouped_Atoms_EachNum[gidx]; Lidx++){

                i0 = MP3[Lidx];
                GA_AN = CWF_Grouped_Atoms[gidx][Lidx];
      	        wan1 = WhatSpecies[GA_AN];
	        tno1 = Spe_Total_CNO[wan1];

                for (i=0; i<tno1; i++){

  		  p = m*n2 + MP[GA_AN] - 1 + i;  
                  csum1 += InProd_BasisFunc[p]*CWF_Guiding_MOs_NC[gidx][k][i0+i];

  		  p = m*n2 + MP[GA_AN] - 1 + i + n2/2;
                  csum2 += InProd_BasisFunc[p]*CWF_Guiding_MOs_NC[gidx][k][i0+i+dim];

		} // i
	      } // Lidx

   	      q = m*TNum_CWFs + MP2[gidx] + k;
 	      InProd[q] = csum1 + csum2;

	    } // k

            /* freeing of MP3 */
 
            free(MP3); 

	  } // gidx
	} // m

      } /* end of else if (CWF_Guiding_Orbital==3) */

      /* MPI_Allreduce of InProd */       

      if (CWF_Guiding_Orbital==1 || CWF_Guiding_Orbital==2){
        MPI_Allreduce( MPI_IN_PLACE, &InProd[0], (mmax-mmin+1)*TNum_CWFs, MPI_C_DOUBLE_COMPLEX, MPI_SUM, mpi_comm_level1 );
      }

      /* store InProd into an array: Cs in the SCALAPACK form */       

      for (i=0; i<na_rows_CWF3; i++){

	m1 = np_rows_CWF3*nblk_CWF3*((i)/nblk_CWF3) + (i)%nblk_CWF3
	  + ((np_rows_CWF3+my_prow_CWF3)%np_rows_CWF3)*nblk_CWF3 + 1;

	if ( mmin<=m1 && m1<=mmax ){ // the base is 1 for mmin and mmax.

	  m = m1 - mmin;  // local index

	  for (j=0; j<na_cols_CWF3; j++){

	    jg = np_cols_CWF3*nblk_CWF3*((j)/nblk_CWF3) + (j)%nblk_CWF3 
	      + ((np_cols_CWF3+my_pcol_CWF3)%np_cols_CWF3)*nblk_CWF3;

	    p = m*TNum_CWFs + jg;
	    Cs[j*na_rows_CWF3+i].r = creal(InProd[p]);
	    Cs[j*na_rows_CWF3+i].i = cimag(InProd[p]);

	  } // j
	} // if ( mmin<=m1 && m1<=mmax ) 
      } // i

    } // if ( num!=0 )
  } // ID

  /*
  printf("Real Cs\n");
  for (i=0; i<MaxN; i++){
    for (j=0; j<TNum_CWFs; j++){
      printf("%8.4f ",creal(Cs[j*na_rows_CWF3+i])); 
    }
    printf("\n");
  }

  printf("Imag Cs\n");
  for (i=0; i<MaxN; i++){
    for (j=0; j<TNum_CWFs; j++){
      printf("%8.4f ",cimag(Cs[j*na_rows_CWF3+i])); 
    }
    printf("\n");
  }
  */

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

    e = ko[m1];

    if (CWF_WindowF_switch==0){ // Poly
      weight = weight_CWF2(e,ChemP,in0,in1,out0,out1); 
    }
    else if (CWF_WindowF_switch==1){ // Fermi
      weight = (1.0/(exp(b0*(e0-e))+1.0))*(1.0/(exp(b1*(e-e1))+1.0)) + CWF_disentangling_smearing_bound;
    }

    for (j=0; j<na_cols_CWF3; j++){
      Cs[j*na_rows_CWF3+i].r *= weight;
      Cs[j*na_rows_CWF3+i].i *= weight;
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
    double tmp;

    sprintf(fname_ProM,"%s%s_evec/pm_%d",filepath,filename,0);

    Save_Projection_Matrix("comp", fname_ProM,
			   &tmp, Cs, Full_EVec,
			   MinN, MaxN, TNum_CWFs,
			   nblk_CWF3, 
			   na_rows_CWF3, na_cols_CWF3,
			   np_rows_CWF3, np_cols_CWF3,
			   my_prow_CWF3, my_pcol_CWF3,
			   mpi_comm_level1);
  }

  /********************************************************
      Singular Value Decomposition (SVD) of Cs.
      As for how to set desc_CWF1, see also the comment 
      in Allocate_Free_Cluster_NonCol_CWF
  ********************************************************/

  /* As for the size of Cs, Ws, and Vs, 
     see https://manpages.ubuntu.com/manpages/focal/man3/pdgesvd.3.html */

  F77_NAME(pzgesvd,PZGESVD)( "V", "V",
			     &MaxN, &TNum_CWFs,
			     Cs, &ONE, &ONE, desc_CWF3,   
			     sv,
			     Ws, &ONE, &ONE, desc_CWF3,                              
			     Vs, &ONE, &ONE, desc_CWF1,
			     work,
			     &lwork,
                             rwork,
			     &info);

  /*
  if (myid==0){
    for (i=0; i<TNum_CWFs; i++){
      printf("ZZZ1 of Cs myid=%2d i=%2d sv=%18.15f\n",myid,i,sv[i]);
    }
  }
  */

  for (i=0; i<TNum_CWFs; i++){
    dif = sv[i] - 1.0;
    DM_func += dif*dif;
  }

  /*******************************************
        Polar Decomposition (PD) of Cs
  *******************************************/

  Cblacs_barrier(ictxt1_CWF1,"A");
  F77_NAME(pzgemm,PZGEMM)( "N","N",
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

      Cs[j*na_rows_CWF3+i].r = ko[ig]*Hs[j*na_rows_CWF3+i].r;
      Cs[j*na_rows_CWF3+i].i = ko[ig]*Hs[j*na_rows_CWF3+i].i;
    }
  }

  /* Hs^dag x Cs -> Ws */

  Cblacs_barrier(ictxt1_CWF1,"A");
  F77_NAME(pzgemm,PZGEMM)( "T","N",
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
	  Hop[jg*TNum_CWFs+ig] = Ws[j*na_rows_CWF1+i].r + I*Ws[j*na_rows_CWF1+i].i;
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
  F77_NAME(pzgemm,PZGEMM)( "N","N",
                           &n2, &TNum_CWFs, &MaxN, 
                           &alpha,
                           EVs_PAO,   &ONE, &ONE, desc_CWF4,
                           Hs,        &ONE, &ONE, desc_CWF3,
                           &beta,
                           WFs,       &ONE, &ONE, desc_CWF2 );

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
	pnum = 2*CWF_Num_predefined[wan1];
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

	for (i=0; i<n2; i++) CWF_Coef_NC[k][p][0][0][0][i] = 0.0 + 0.0*I;

	for (j=0; j<na_cols_CWF2; j++){

	  jg = np_cols_CWF2*nblk_CWF2*((j)/nblk_CWF2) + (j)%nblk_CWF2 
	    + ((np_cols_CWF2+my_pcol_CWF2)%np_cols_CWF2)*nblk_CWF2;

	  if (q==jg){

	    for (i=0; i<na_rows_CWF2; i++){

	      ig = np_rows_CWF2*nblk_CWF2*((i)/nblk_CWF2) + (i)%nblk_CWF2
		+ ((np_rows_CWF2+my_prow_CWF2)%np_rows_CWF2)*nblk_CWF2;

	      CWF_Coef_NC[k][p][0][0][0][ig] = WFs[j*na_rows_CWF2+i].r + I*WFs[j*na_rows_CWF2+i].i;

	    } // i
	  } // if (q==jg)
	} // j
      } // p
    } // k

    for (k=0; k<CWF_fileout_Num; k++){

      if (CWF_Guiding_Orbital==1 || CWF_Guiding_Orbital==2){
	Gc_AN = CWF_file_Atoms[k];
	wan1 = WhatSpecies[Gc_AN];
	pnum = 2*CWF_Num_predefined[wan1];
      }
      else if (CWF_Guiding_Orbital==3){
	gidx = CWF_file_MOs[k];
	pnum = Num_CWF_MOs_Group[gidx]; 
      }

      for (p=0; p<pnum; p++){
	MPI_Allreduce( MPI_IN_PLACE, &CWF_Coef_NC[k][p][0][0][0][0], n2, MPI_C_DOUBLE_COMPLEX, MPI_SUM, mpi_comm_level1 );
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

      x = (ko[ig] - ChemP)*Beta;
      if (x<=-max_x) x = -max_x;
      if (max_x<=x)  x = max_x;
      FermiF = 1.0/(1.0 + exp(x));

      tmp = FermiF*( Hs[j*na_rows_CWF3+i].r*Hs[j*na_rows_CWF3+i].r
                   + Hs[j*na_rows_CWF3+i].i*Hs[j*na_rows_CWF3+i].i );

      CWF_Charge[0][jg] += tmp;
      CWF_Energy[0][jg] += tmp*ko[ig];
    }
  }

  MPI_Allreduce( MPI_IN_PLACE, &CWF_Charge[0][0], TNum_CWFs, MPI_DOUBLE, MPI_SUM, mpi_comm_level1 );
  MPI_Allreduce( MPI_IN_PLACE, &CWF_Energy[0][0], TNum_CWFs, MPI_DOUBLE, MPI_SUM, mpi_comm_level1 );

  Sum_Charges = 0.0;
  Sum_Energies = 0.0;;

  for (i=0; i<TNum_CWFs; i++){
    Sum_Charges  += CWF_Charge[0][i];
    Sum_Energies += CWF_Energy[0][i];

    //if (myid==Host_ID) printf("BBB1 i=%2d %15.12f %15.12f\n",i,CWF_Charge[0][i],CWF_Energy[0][i]);
  }

  //if (myid==Host_ID) printf("BBB2 %15.12f %15.12f\n",Sum_Charges,Sum_Energies);

  /* show DM_func */ 

  if (myid==0) printf("DM_func=%15.12f  %15.12f per CWF\n",DM_func,DM_func/(double)TNum_CWFs);

  /**********************************************************
    save the effective charges and local energies to a file
  ***********************************************************/

  if (CWF_Guiding_Orbital==1 || CWF_Guiding_Orbital==2){

    if ( myid==Host_ID ){
        
      int i0;
      char file_CWF_Charge[YOUSO10];
      FILE *fp_CWF_Charge;
      char buf[fp_bsize];          /* setvbuf */
      double sumP,sumE,TZ;

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
                                 DM_func,DM_func/(double)TNum_CWFs);

	for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){

	  wan1 = WhatSpecies[Gc_AN];
	  i0 = MP2[Gc_AN];
	  sumP = 0.0; 

	  for (i=0; i<2*CWF_Num_predefined[wan1]; i++){
	    sumP += CWF_Charge[0][i0+i];
	  }
       
	  fprintf(fp_CWF_Charge,"   %4d %4s     %12.9f\n",Gc_AN, SpeName[wan1], sumP);

	} // Gc_AN

	fprintf(fp_CWF_Charge,"\n");
        fprintf(fp_CWF_Charge,"    total=       %12.9f    ideal(neutral)=%12.9f\n",Sum_Charges,TZ);     

	/* decomposed populations */

	fprintf(fp_CWF_Charge,"\n\n  Orbitally decomposed populations evaluated by CWF\n");

	for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){

	  wan1 = WhatSpecies[Gc_AN];
	  i0 = MP2[Gc_AN];

	  fprintf(fp_CWF_Charge,"\n %4d %4s\n",Gc_AN,SpeName[wan1]);
	  fprintf(fp_CWF_Charge,"  orbital index\n");

	  wan1 = WhatSpecies[Gc_AN];
	  sumP = 0.0; 

	  for (i=0; i<2*CWF_Num_predefined[wan1]; i++){
	    fprintf(fp_CWF_Charge,"      %2d         %12.9f\n", i,CWF_Charge[0][i0+i]);
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

    if ( myid==Host_ID ){
        
      int i0;
      char file_CWF_Charge[YOUSO10];
      FILE *fp_CWF_Charge;
      char buf[fp_bsize];          /* setvbuf */
      double sumP,sumE,TZ;

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
                                 DM_func,DM_func/(double)TNum_CWFs);

	fprintf(fp_CWF_Charge,"   Group\n");

        for (gidx=0; gidx<Num_CWF_Grouped_Atoms; gidx++){

	  i0 = MP2[gidx];
	  sumP = 0.0; 

	  for (Lidx=0; Lidx<Num_CWF_MOs_Group[gidx]; Lidx++){
	    sumP += CWF_Charge[0][i0+Lidx];
	  }

	  fprintf(fp_CWF_Charge,"   %4d          %12.9f\n", gidx+1, sumP);

	} // gidx                 

	fprintf(fp_CWF_Charge,"\n");
        fprintf(fp_CWF_Charge,"    total=       %12.9f    ideal(neutral)=%12.9f\n",Sum_Charges,TZ);     

	/* decomposed populations */

	fprintf(fp_CWF_Charge,"\n\n  Decomposed populations evaluated by CWF\n");

        for (gidx=0; gidx<Num_CWF_Grouped_Atoms; gidx++){

	  wan1 = WhatSpecies[Gc_AN];
	  i0 = MP2[gidx];

	  fprintf(fp_CWF_Charge,"\n  Group %4d\n",gidx+1);
	  fprintf(fp_CWF_Charge,"  orbital index\n");

	  wan1 = WhatSpecies[Gc_AN];
	  sumP = 0.0; 

	  for (i=0; i<Num_CWF_MOs_Group[gidx]; i++){
	    fprintf(fp_CWF_Charge,"      %2d         %12.9f\n", i+1,CWF_Charge[0][i0+i]);
	  }

	} // gidx

	/* fclose fp_CWF_Charge */

	fclose(fp_CWF_Charge);

      } // if ((fp_CWF_Charge = fopen(file_CWF_Charge,"w")) != NULL)

      else{
	printf("Failure of saving the CWF_Charge file.\n");
      }

    } // if ( myid==Host_ID )
  }

  /**********************************************************
             save Hop into a file of *.CWF.Hop
  ***********************************************************/

  MPI_Allreduce( MPI_IN_PLACE, &Hop[0], (TNum_CWFs*TNum_CWFs), MPI_C_DOUBLE_COMPLEX, MPI_SUM, mpi_comm_level1 );

  if (myid==Host_ID){

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
        int_data[5+i] = 2*CWF_Num_predefined[wan1];
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

    if (fp_Hop_ok==1){
      fwrite(&Hop[0],sizeof(double complex),(TNum_CWFs*TNum_CWFs),fp_Hop);
    }

    free(int_data);
    if (fp_Hop_ok==1) fclose(fp_Hop);
  }

  /* save Hop with the distance between two sites into the .CWF.Dis_vs_H file */

  if (myid==0 && CWF_Dis_vs_H==1){

    int GA,GB,wanA,wanB,p1,p2,gidx1,gidx2,Lidx1,Lidx2;
    double dx,dy,dz,dis,xA,yA,zA,xB,yB,zB,re,im,t; 
    char fDis_vs_H[YOUSO10];
    FILE *fp_Dis_vs_H;
  
    sprintf(fDis_vs_H,"%s%s.CWF.Dis_vs_H",filepath,filename);

    if ((fp_Dis_vs_H = fopen(fDis_vs_H,"w")) != NULL){

      fprintf(fp_Dis_vs_H,"#\n"); 
      fprintf(fp_Dis_vs_H,"# GA, GB, i, j, distance (Ang.), Hopping integral (eV)\n"); 
      fprintf(fp_Dis_vs_H,"#\n"); 

      if (CWF_Guiding_Orbital==1 || CWF_Guiding_Orbital==2){

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

		re = creal(Hop[(p2+j)*TNum_CWFs+(p1+i)]);                    
		im = cimag(Hop[(p2+j)*TNum_CWFs+(p1+i)]);
		t = sqrt(re*re+im*im)*eV2Hartree;

		fprintf(fp_Dis_vs_H,"%2d %2d %2d %2d  %18.15f %18.15f\n",
			GA,GB,i,j,dis*BohrR,t);
	      } 
	    }

	    p2 += CWF_Num_predefined[wanB]; 

	  } // GB 

	  p1 += CWF_Num_predefined[wanA]; 

	} // GA

      } // end of if (myid0==0 && CWF_Dis_vs_H==1 && fp_Dis_vs_H_ok==1)

      else if (CWF_Guiding_Orbital==3){

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

		re = creal(Hop[(p2+j)*TNum_CWFs+(p1+i)]);                    
		im = cimag(Hop[(p2+j)*TNum_CWFs+(p1+i)]);
		t = sqrt(re*re+im*im)*eV2Hartree;

		fprintf(fp_Dis_vs_H,"%2d %2d %2d %2d  %18.15f %18.15f\n",
			gidx1+1,gidx2+1,i,j,dis*BohrR,t);
	      }
	    }

	    p2 += Num_CWF_MOs_Group[gidx2];

	  } // gidx2

	  p1 += Num_CWF_MOs_Group[gidx1];
                 
	} // gidx1
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
  free(rwork);
  free(work);

  /* freeing of arrays for SCALAPCK */

  Allocate_Free_Cluster_NonCol_CWF( 2, n2, MaxN, TNum_CWFs, &Cs, &Hs, &Vs, &Ws, &EVs_PAO, &WFs );

  /* mearuring elapsed time */

  dtime(&etime);
  return (etime-stime);
}



void Allocate_Free_Cluster_NonCol_CWF( int todo_flag, 
				       int n2,
				       int MaxN,
				       int TNum_CWFs,
				       dcomplex **Cs,
				       dcomplex **Hs, 
				       dcomplex **Vs,
				       dcomplex **Ws,
				       dcomplex **EVs_PAO,
				       dcomplex **WFs )
{
  static int firsttime=1;
  int ZERO=0, ONE=1,info,myid,numprocs;
  int i,k,nblk_m,nblk_m2,wanA,spin,size_EVec1;
  double tmp,tmp1;

  MPI_Barrier(mpi_comm_level1);
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /********************************************
   allocation of arrays 
   
   CWF : TNum_CWFs x TNum_CWFs
   CWF2: n2 x TNum_CWFs
   CWF3: MaxN x TNum_CWFs 
   CWF4: n2 x MaxN
  ********************************************/

  if (todo_flag==1){

    /* CWF1: setting of BLACS for matrices in size of TNum_CWFs x TNum_CWFs */

    np_cols_CWF1 = (int)(sqrt((float)numprocs));
    do{
      if((numprocs%np_cols_CWF1)==0) break;
      np_cols_CWF1--;
    } while (np_cols_CWF1>=2);
    np_rows_CWF1 = numprocs/np_cols_CWF1;

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

    my_prow_CWF1 = myid/np_cols_CWF1;
    my_pcol_CWF1 = myid%np_cols_CWF1;

    na_rows_CWF1 = numroc_(&MaxN, &nblk_CWF1, &my_prow_CWF1, &ZERO, &np_rows_CWF1 );  // the same as for CWF3
    na_cols_CWF1 = numroc_(&TNum_CWFs, &nblk_CWF1, &my_pcol_CWF1, &ZERO, &np_cols_CWF1 );

    bhandle1_CWF1 = Csys2blacs_handle(mpi_comm_level1);
    ictxt1_CWF1 = bhandle1_CWF1;

    Cblacs_gridinit(&ictxt1_CWF1, "Row", np_rows_CWF1, np_cols_CWF1);

    MPI_Allreduce(&na_rows_CWF1,&na_rows_max_CWF1,1,MPI_INT,MPI_MAX,mpi_comm_level1);
    MPI_Allreduce(&na_cols_CWF1,&na_cols_max_CWF1,1,MPI_INT,MPI_MAX,mpi_comm_level1);

    descinit_( desc_CWF1, &TNum_CWFs, &TNum_CWFs, &nblk_CWF1, &nblk_CWF1,  
               &ZERO, &ZERO, &ictxt1_CWF1, &na_rows_CWF1,  &info); 

    /* CWF2: setting of BLACS for matrices in size of n2 x TNum_CWFs */

    np_cols_CWF2 = (int)(sqrt((float)numprocs));
    do{
      if((numprocs%np_cols_CWF2)==0) break;
      np_cols_CWF2--;
    } while (np_cols_CWF2>=2);
    np_rows_CWF2 = numprocs/np_cols_CWF2;

    nblk_m = NBLK;
    while((nblk_m*np_rows_CWF2>n2 || nblk_m*np_cols_CWF2>TNum_CWFs) && (nblk_m > 1)){
      nblk_m /= 2;
    }
    if(nblk_m<1) nblk_m = 1;

    MPI_Allreduce(&nblk_m,&nblk_CWF2,1,MPI_INT,MPI_MIN,mpi_comm_level1);

    ictxt1_CWF2 = bhandle1_CWF1;

    my_prow_CWF2 = myid/np_cols_CWF2;
    my_pcol_CWF2 = myid%np_cols_CWF2;

    Cblacs_gridinit(&ictxt1_CWF2, "Row", np_rows_CWF2, np_cols_CWF2);

    na_rows_CWF2 = numroc_(&n2,        &nblk_CWF2, &my_prow_CWF2, &ZERO, &np_rows_CWF2 );
    na_cols_CWF2 = numroc_(&TNum_CWFs, &nblk_CWF2, &my_pcol_CWF2, &ZERO, &np_cols_CWF2 );

    MPI_Allreduce(&na_rows_CWF2, &na_rows_max_CWF2,1,MPI_INT,MPI_MAX,mpi_comm_level1);
    MPI_Allreduce(&na_cols_CWF2, &na_cols_max_CWF2,1,MPI_INT,MPI_MAX,mpi_comm_level1);

    descinit_( desc_CWF2, &n2, &TNum_CWFs, &nblk_CWF2, &nblk_CWF2,  
               &ZERO, &ZERO, &ictxt1_CWF2, &na_rows_CWF2,  &info);

    desc_CWF2[1] = desc_CWF1[1];

    (*WFs) = (dcomplex*)malloc(sizeof(dcomplex)*na_rows_max_CWF2*na_cols_max_CWF2); 

    /* CWF3: setting of BLACS for matrices in size of MaxN x TNum_CWFs */

    np_cols_CWF3 = (int)(sqrt((float)numprocs));
    do{
      if((numprocs%np_cols_CWF3)==0) break;
      np_cols_CWF3--;
    } while (np_cols_CWF3>=2);
    np_rows_CWF3 = numprocs/np_cols_CWF3;

    nblk_m = NBLK;
    while((nblk_m*np_rows_CWF3>MaxN || nblk_m*np_cols_CWF3>TNum_CWFs) && (nblk_m > 1)){
      nblk_m /= 2;
    }
    if(nblk_m<1) nblk_m = 1;

    MPI_Allreduce(&nblk_m,&nblk_CWF3,1,MPI_INT,MPI_MIN,mpi_comm_level1);

    ictxt1_CWF3 = bhandle1_CWF1;

    my_prow_CWF3 = myid/np_cols_CWF3;
    my_pcol_CWF3 = myid%np_cols_CWF3;

    Cblacs_gridinit(&ictxt1_CWF3, "Row", np_rows_CWF3, np_cols_CWF3);

    na_rows_CWF3 = numroc_(&MaxN,      &nblk_CWF3, &my_prow_CWF3, &ZERO, &np_rows_CWF3 );
    na_cols_CWF3 = numroc_(&TNum_CWFs, &nblk_CWF3, &my_pcol_CWF3, &ZERO, &np_cols_CWF3 );

    MPI_Allreduce(&na_rows_CWF3, &na_rows_max_CWF3,1,MPI_INT,MPI_MAX,mpi_comm_level1);
    MPI_Allreduce(&na_cols_CWF3, &na_cols_max_CWF3,1,MPI_INT,MPI_MAX,mpi_comm_level1);

    descinit_( desc_CWF3, &MaxN, &TNum_CWFs, &nblk_CWF3, &nblk_CWF3,  
               &ZERO, &ZERO, &ictxt1_CWF3, &na_rows_CWF3,  &info);

    desc_CWF3[1] = desc_CWF1[1];

    /* CWF4: setting of BLACS for matrices in size of n2 x MaxN */

    np_cols_CWF4 = (int)(sqrt((float)numprocs));
    do{
      if((numprocs%np_cols_CWF4)==0) break;
      np_cols_CWF4--;
    } while (np_cols_CWF4>=2);
    np_rows_CWF4 = numprocs/np_cols_CWF4;

    nblk_m = NBLK;
    while((nblk_m*np_rows_CWF4>n2 || nblk_m*np_cols_CWF4>MaxN) && (nblk_m > 1)){
      nblk_m /= 2;
    }
    if(nblk_m<1) nblk_m = 1;

    MPI_Allreduce(&nblk_m,&nblk_CWF4,1,MPI_INT,MPI_MIN,mpi_comm_level1);

    ictxt1_CWF4 = bhandle1_CWF1;

    my_prow_CWF4 = myid/np_cols_CWF4;
    my_pcol_CWF4 = myid%np_cols_CWF4;

    Cblacs_gridinit(&ictxt1_CWF4, "Row", np_rows_CWF4, np_cols_CWF4);

    na_rows_CWF4 = numroc_(&n2,   &nblk_CWF4, &my_prow_CWF4, &ZERO, &np_rows_CWF4 );
    na_cols_CWF4 = numroc_(&MaxN, &nblk_CWF4, &my_pcol_CWF4, &ZERO, &np_cols_CWF4 );

    MPI_Allreduce(&na_rows_CWF4, &na_rows_max_CWF4,1,MPI_INT,MPI_MAX,mpi_comm_level1);
    MPI_Allreduce(&na_cols_CWF4, &na_cols_max_CWF4,1,MPI_INT,MPI_MAX,mpi_comm_level1);

    descinit_( desc_CWF4, &n2, &MaxN, &nblk_CWF4, &nblk_CWF4,
               &ZERO, &ZERO, &ictxt1_CWF4, &na_rows_CWF4, &info);

    desc_CWF4[1] = desc_CWF1[1];

    *Cs = (dcomplex*)malloc(sizeof(dcomplex)*na_rows_max_CWF4*na_cols_max_CWF4);
    *Hs = (dcomplex*)malloc(sizeof(dcomplex)*na_rows_max_CWF4*na_cols_max_CWF4);
    *Vs = (dcomplex*)malloc(sizeof(dcomplex)*na_rows_max_CWF4*na_cols_max_CWF4);
    *Ws = (dcomplex*)malloc(sizeof(dcomplex)*na_rows_max_CWF4*na_cols_max_CWF4);
    *EVs_PAO = (dcomplex*)malloc(sizeof(dcomplex)*na_rows_max_CWF4*na_cols_max_CWF4); 

    /* save information of the memory usage */

    if (firsttime && memoryusage_fileout) {
      PrintMemory("Allocate_Free_Cluster_Col_CWF: Cs  ",sizeof(dcomplex)*na_rows_max_CWF4*na_cols_max_CWF4,NULL);
      PrintMemory("Allocate_Free_Cluster_Col_CWF: Hs  ",sizeof(dcomplex)*na_rows_max_CWF4*na_cols_max_CWF4,NULL);
      PrintMemory("Allocate_Free_Cluster_Col_CWF: Vs  ",sizeof(dcomplex)*na_rows_max_CWF4*na_cols_max_CWF4,NULL);
      PrintMemory("Allocate_Free_Cluster_Col_CWF: Ws  ",sizeof(dcomplex)*na_rows_max_CWF3*na_cols_max_CWF3,NULL);
      PrintMemory("Allocate_Free_Cluster_Col_CWF: WFs ",sizeof(dcomplex)*na_rows_max_CWF2*na_cols_max_CWF2,NULL);
      PrintMemory("Allocate_Free_Cluster_Col_CWF: EVs_PAO",sizeof(dcomplex)*na_rows_max_CWF4*na_cols_max_CWF4,NULL);
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



double Calc_Hybrid_AO_NonCol( double ****OLP0, double *****Hks, double *****CDM )
{
  int i,j,k,l,n,Mc_AN,Gc_AN,h_AN,Mh_AN,Gh_AN;
  int Cwan,num,wan1,wan2,tno0,tno1,tno2,spin;
  int Nloop,po,p,q,NumLNOs;
  int mul,m,l2,mul2,m2; 
  dcomplex *DMS;
  double *EVal,*carray;
  double sum,sum0,F;
  double TStime,TEtime;
  int myid,numprocs,tag=999;
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

  carray = (double*)malloc(sizeof(double)*8*List_YOUSO[7]*List_YOUSO[7]);
  DMS = (dcomplex*)malloc(sizeof(dcomplex)*4*List_YOUSO[7]*List_YOUSO[7]);
  EVal = (double*)malloc(sizeof(double)*2*List_YOUSO[7]);

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

    for (i=0; i<(4*List_YOUSO[7]*List_YOUSO[7]); i++) DMS[i] = Complex(0.0,0.0);

    Gc_AN = M2G[Mc_AN];
    wan1 = WhatSpecies[Gc_AN];
    tno1 = Spe_Total_CNO[wan1];
    NumLNOs = CWF_Num_predefined[wan1];

    /* use hybrid orbitals */

    if (CWF_Guiding_Orbital==1){ 

      /* set DMS: select only the contributions of the minimal basis */

      int mul,m,l2,mul2,m2; 

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

		      DMS[2*tno1*(j     )+i     ].r = -CDM[0][Mc_AN][0][i][j];
		      DMS[2*tno1*(j+tno1)+i+tno1].r = -CDM[1][Mc_AN][0][i][j];
		      DMS[2*tno1*(j+tno1)+i     ].r = -CDM[2][Mc_AN][0][i][j];
		      DMS[2*tno1*(i     )+j+tno1].r = -CDM[2][Mc_AN][0][j][i];
		      DMS[2*tno1*(j+tno1)+i     ].i = -CDM[3][Mc_AN][0][i][j];
		      DMS[2*tno1*(i     )+j+tno1].i =  CDM[3][Mc_AN][0][j][i];
		      DMS[2*tno1*(j     )+i     ].i = -iDM[0][0][Mc_AN][0][i][j];
		      DMS[2*tno1*(j+tno1)+i+tno1].i = -iDM[0][1][Mc_AN][0][i][j];
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

      /* diagonalize the DMS */

      lapack_zheev( 2*tno1, DMS, EVal );

      /*
      for (i=0; i<2*tno1; i++){
        printf("ABC1 Mc_AN=%2d i=%2d %15.12f\n",Mc_AN,i,EVal[i]);
      }
      */

      /* store the eigenvectors to QAO2_coes */

      for (j=0; j<2*NumLNOs; j++){
	for (i=0; i<2*tno1; i++){
	  QAO2_coes[Gc_AN][2*tno1*j+i] = DMS[2*tno1*j+i];
	}
      } /* j */

    } /* end of if (CWF_Guiding_Orbital==1) */

    /* use atomic orbitals */

    else if (CWF_Guiding_Orbital==2){

      int mul,n0=0,n1=0; 

      for (l=0; l<=Spe_MaxL_Basis[wan1]; l++){
	for (mul=0; mul<Spe_Num_Basis[wan1][l]; mul++){

	  for (j=0; j<(2*l+1); j++){

	    if (CWF_Guiding_AO[wan1][l][mul]==1){

	      QAO2_coes[Gc_AN][2*tno1*n1+n0] = Complex(1.0,0.0);
	      QAO2_coes[Gc_AN][2*tno1*(NumLNOs+n1)+tno1+n0] = Complex(1.0,0.0);

	      n1++;
	    }

	    n0++;
	  }  
	}
      }

    }

  } /* Mc_AN */

  for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){

    wan1 = WhatSpecies[Gc_AN];
    tno1 = Spe_Total_CNO[wan1];
    NumLNOs = CWF_Num_predefined[wan1];

    for (i=0; i<4*tno1*NumLNOs; i++){
      carray[2*i  ] = QAO2_coes[Gc_AN][i].r; 
      carray[2*i+1] = QAO2_coes[Gc_AN][i].i; 
    }

    MPI_Allreduce(MPI_IN_PLACE, &carray[0], 8*tno1*NumLNOs,MPI_DOUBLE,MPI_SUM,mpi_comm_level1);

    for (i=0; i<4*tno1*NumLNOs; i++){
      QAO2_coes[Gc_AN][i].r = carray[2*i  ];
      QAO2_coes[Gc_AN][i].i = carray[2*i+1]; 
    }
  }

  /********************************************
             freeing of arrays
  ********************************************/

  free(carray);
  free(DMS);
  free(EVal);

  /* elapsed time */
  dtime(&TEtime);
  return (TEtime-TStime);
}




double Calc_MO_in_Bulk_NonCol( double *****CDM, double *****iCDM )
{
  int i,j,k,gidx,Lidx,GA_AN,GB_AN,GC_AN,tnum,wan1;
  int Lidx1,Lidx2,i0,j0,MA_AN,h_AN3,h_AN2,spin;
  int tno1,h_AN,Gh_AN,wan2,tno2,dim,dim2,l1,l2,l3,p,q;
  int wan3,tno3,RnC,il,jl,ig,jg,brow,bcol,prow,pcol;
  int numprocs,myid,po;
  int *MP3,**RMI0;
  double tmp,sum,TStime,TEtime;
  double complex csum0,csum1,csum2,ctmp;
  double complex *Smo,**DMmo,**Hmo;
  double complex ****Smo2,*****DMmo2,*****Hmo2;
  dcomplex *DMS,*DMS2;
  double complex **OLPgidx,**Hgidx,*EVec;
  double *WI;
  int *IWORK;
  double *EVal;
  /* for scalpack */
  int ZERO=0, ONE=1, info;
  int nblk_MIB,np_rows_MIB,np_cols_MIB,na_rows_MIB,na_cols_MIB;
  int na_rows_max_MIB,na_cols_max_MIB;
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

  CWF_Guiding_MOs_NC = (double complex***)malloc(sizeof(double complex**)*Num_CWF_Grouped_Atoms);

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
    dim2 = 2*k;

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

    DMS = (dcomplex*)malloc(sizeof(dcomplex)*dim2*dim2);
    for (i=0; i<dim2*dim2; i++) DMS[i] = Complex(0.0,0.0);

    DMS2 = (dcomplex*)malloc(sizeof(dcomplex)*dim2*dim2);
    for (i=0; i<dim2*dim2; i++) DMS2[i] = Complex(0.0,0.0);

    EVal = (double*)malloc(sizeof(double)*dim2);
    EVec = (double complex*)malloc(sizeof(double complex)*dim2*dim2);

    WI = (double*)malloc(sizeof(double)*dim2);
    IWORK = (int*)malloc(sizeof(int)*dim2);

    OLPgidx = (double complex**)malloc(sizeof(double complex*)*dim2);
    for (i=0; i<dim2; i++){
      OLPgidx[i] = (double complex*)malloc(sizeof(double complex)*dim2);
      for (j=0; j<dim2; j++) OLPgidx[i][j] = 0.0 + 0.0*I;
    }

    Hgidx = (double complex**)malloc(sizeof(double complex*)*dim2);
    for (i=0; i<dim2; i++){
      Hgidx[i] = (double complex*)malloc(sizeof(double complex)*dim2);
      for (j=0; j<dim2; j++) Hgidx[i][j] = 0.0 + 0.0*I;
    }

    Smo = (double complex*)malloc(sizeof(double complex)*tnum);
    for (i=0; i<tnum; i++) Smo[i] = 0.0 + 0.0*I;

    DMmo = (double complex**)malloc(sizeof(double complex*)*3);
    for (k=0; k<3; k++){
      DMmo[k] = (double complex*)malloc(sizeof(double complex)*tnum);
      for (i=0; i<tnum; i++) DMmo[k][i] = 0.0 + 0.0*I;
    }

    Hmo = (double complex**)malloc(sizeof(double complex*)*3);
    for (k=0; k<3; k++){
      Hmo[k] = (double complex*)malloc(sizeof(double complex)*tnum);
      for (i=0; i<tnum; i++) Hmo[k][i] = 0.0 + 0.0*I;
    }

    Smo2 = (double complex****)malloc(sizeof(double complex***)*CWF_Grouped_Atoms_EachNum[gidx]);
    for (Lidx=0; Lidx<CWF_Grouped_Atoms_EachNum[gidx]; Lidx++){

      GA_AN = CWF_Grouped_Atoms[gidx][Lidx];
      wan1 = WhatSpecies[GA_AN];
      tno1 = Spe_Total_CNO[wan1];
      Smo2[Lidx] = (double complex***)malloc(sizeof(double complex**)*(FNAN[GA_AN]+1));

      for (h_AN=0; h_AN<=FNAN[GA_AN]; h_AN++){

	Gh_AN = natn[GA_AN][h_AN];
	wan2 = WhatSpecies[Gh_AN];
	tno2 = Spe_Total_CNO[wan2];
	Smo2[Lidx][h_AN] = (double complex**)malloc(sizeof(double complex*)*tno1);

	for (i=0; i<tno1; i++){
	  Smo2[Lidx][h_AN][i] = (double complex*)malloc(sizeof(double complex)*tno2);
          for (j=0; j<tno2; j++) Smo2[Lidx][h_AN][i][j] = 0.0 + 0.0*I;
	}
      }
    }

    DMmo2 = (double complex*****)malloc(sizeof(double complex****)*3);
    for (k=0; k<3; k++){

      DMmo2[k] = (double complex****)malloc(sizeof(double complex***)*CWF_Grouped_Atoms_EachNum[gidx]);
      for (Lidx=0; Lidx<CWF_Grouped_Atoms_EachNum[gidx]; Lidx++){

	GA_AN = CWF_Grouped_Atoms[gidx][Lidx];
	wan1 = WhatSpecies[GA_AN];
	tno1 = Spe_Total_CNO[wan1];
	DMmo2[k][Lidx] = (double complex***)malloc(sizeof(double complex**)*(FNAN[GA_AN]+1));

	for (h_AN=0; h_AN<=FNAN[GA_AN]; h_AN++){

	  Gh_AN = natn[GA_AN][h_AN];
	  wan2 = WhatSpecies[Gh_AN];
	  tno2 = Spe_Total_CNO[wan2];
	  DMmo2[k][Lidx][h_AN] = (double complex**)malloc(sizeof(double complex*)*tno1);

	  for (i=0; i<tno1; i++){
	    DMmo2[k][Lidx][h_AN][i] = (double complex*)malloc(sizeof(double complex)*tno2);
	    for (j=0; j<tno2; j++) DMmo2[k][Lidx][h_AN][i][j] = 0.0 + 0.0*I;
	  }
	}
      }
    }

    Hmo2 = (double complex*****)malloc(sizeof(double complex****)*3);
    for (k=0; k<3; k++){
      Hmo2[k] = (double complex****)malloc(sizeof(double complex***)*CWF_Grouped_Atoms_EachNum[gidx]);
      for (Lidx=0; Lidx<CWF_Grouped_Atoms_EachNum[gidx]; Lidx++){

	GA_AN = CWF_Grouped_Atoms[gidx][Lidx];
	wan1 = WhatSpecies[GA_AN];
	tno1 = Spe_Total_CNO[wan1];

	Hmo2[k][Lidx] = (double complex***)malloc(sizeof(double complex**)*(FNAN[GA_AN]+1));
	for (h_AN=0; h_AN<=FNAN[GA_AN]; h_AN++){

	  Gh_AN = natn[GA_AN][h_AN];
	  wan2 = WhatSpecies[Gh_AN];
	  tno2 = Spe_Total_CNO[wan2];

	  Hmo2[k][Lidx][h_AN] = (double complex**)malloc(sizeof(double complex*)*tno1);
	  for (i=0; i<tno1; i++){
	    Hmo2[k][Lidx][h_AN][i] = (double complex*)malloc(sizeof(double complex)*tno2);
	    for (j=0; j<tno2; j++) Hmo2[k][Lidx][h_AN][i][j] = 0.0 + 0.0*I;
	  }
	}
      }
    }

    RMI0 = (int**)malloc(sizeof(int*)*CWF_Grouped_Atoms_EachNum[gidx]);
    for (Lidx=0; Lidx<CWF_Grouped_Atoms_EachNum[gidx]; Lidx++){
      GA_AN = CWF_Grouped_Atoms[gidx][Lidx];
      RMI0[Lidx] = (int*)malloc(sizeof(int)*(FNAN[GA_AN]+1)*(FNAN[GA_AN]+1));
      for (i=0; i<(FNAN[GA_AN]+1)*(FNAN[GA_AN]+1); i++) RMI0[Lidx][i] = 0;
    }

    /* set Smo, DMmo, iDMmo, Hmo */

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

	      Smo[tnum]      = OLP[0][MA_AN][h_AN][i][j] + 0.0*I;

              DMmo[0][tnum]  = CDM[0][MA_AN][h_AN][i][j] + iCDM[0][MA_AN][h_AN][i][j]*I;
              DMmo[1][tnum]  = CDM[1][MA_AN][h_AN][i][j] + iCDM[1][MA_AN][h_AN][i][j]*I;
              DMmo[2][tnum]  = CDM[2][MA_AN][h_AN][i][j] +  CDM[3][MA_AN][h_AN][i][j]*I;

              Hmo[0][tnum]   = H[0][MA_AN][h_AN][i][j] + iHNL[0][MA_AN][h_AN][i][j]*I;
              Hmo[1][tnum]   = H[1][MA_AN][h_AN][i][j] + iHNL[1][MA_AN][h_AN][i][j]*I;
              Hmo[2][tnum]   = H[2][MA_AN][h_AN][i][j] + H[3][MA_AN][h_AN][i][j]*I;
	    }

	    tnum++; 
	  }
	}
      }
    }    

    /* MPI_Allreduce of Smo, DMmo, and Hmo */

    MPI_Allreduce(MPI_IN_PLACE, &Smo[0],     tnum, MPI_C_DOUBLE_COMPLEX,MPI_SUM,mpi_comm_level1);
    MPI_Allreduce(MPI_IN_PLACE, &DMmo[0][0], tnum, MPI_C_DOUBLE_COMPLEX,MPI_SUM,mpi_comm_level1);
    MPI_Allreduce(MPI_IN_PLACE, &DMmo[1][0], tnum, MPI_C_DOUBLE_COMPLEX,MPI_SUM,mpi_comm_level1);
    MPI_Allreduce(MPI_IN_PLACE, &DMmo[2][0], tnum, MPI_C_DOUBLE_COMPLEX,MPI_SUM,mpi_comm_level1);
    MPI_Allreduce(MPI_IN_PLACE, &Hmo[0][0],  tnum, MPI_C_DOUBLE_COMPLEX,MPI_SUM,mpi_comm_level1);
    MPI_Allreduce(MPI_IN_PLACE, &Hmo[1][0],  tnum, MPI_C_DOUBLE_COMPLEX,MPI_SUM,mpi_comm_level1);
    MPI_Allreduce(MPI_IN_PLACE, &Hmo[2][0],  tnum, MPI_C_DOUBLE_COMPLEX,MPI_SUM,mpi_comm_level1);

    /* copy Smo->Smo2, DMmo->DMmo2, and Hmo->Hmo2 */

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

            Smo2[Lidx][h_AN][i][j]      = Smo[k];
            DMmo2[0][Lidx][h_AN][i][j]  = DMmo[0][k];
            DMmo2[1][Lidx][h_AN][i][j]  = DMmo[1][k];
            DMmo2[2][Lidx][h_AN][i][j]  = DMmo[2][k];
            Hmo2[0][Lidx][h_AN][i][j]   = Hmo[0][k];
            Hmo2[1][Lidx][h_AN][i][j]   = Hmo[1][k];
            Hmo2[2][Lidx][h_AN][i][j]   = Hmo[2][k];

            k++; 
	  }
	}
      }
    }    

    free(Smo);
    for (k=0; k<3; k++){ free(DMmo[k]); } free(DMmo);
    for (k=0; k<3; k++){ free(Hmo[k]);  } free(Hmo);

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

    /*********************************************************
     Multiplication of DMmos and Smo2

      DM = ( DMmo2[0]      DMmo2[2] ) 
           ( DMmo2[2]^dag  DMmo2[1] )

      S  = ( Smo2   0    )
           ( 0      Smo2 )

      DM x S = ( DMmo2[0] x Smo2      DMmo2[2] x Smo2 )        
               ( DMmo2[2]^dag x Smo2  DMmo2[1] x Smo2 )
    **********************************************************/ 
    
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

		  csum0 = 0.0 + 0.0*I;
		  csum1 = 0.0 + 0.0*I;
		  csum2 = 0.0 + 0.0*I;

		  for (k=0; k<tno3; k++){
		    csum0 += DMmo2[0][Lidx1][h_AN][i][k]*Smo2[Lidx2][h_AN2][j][k];
		    csum1 += DMmo2[1][Lidx1][h_AN][i][k]*Smo2[Lidx2][h_AN2][j][k];
		    csum2 += DMmo2[2][Lidx1][h_AN][i][k]*Smo2[Lidx2][h_AN2][j][k];
		  } /* k */  

		  DMS[(j0+j)*dim2 + (i0+i)].r         += creal(csum0); DMS[(j0+j)*dim2 + (i0+i)].i         += cimag(csum0);
		  DMS[(j0+j+dim)*dim2 + (i0+i+dim)].r += creal(csum1); DMS[(j0+j+dim)*dim2 + (i0+i+dim)].i += cimag(csum1);
		  DMS[(j0+j+dim)*dim2 + (i0+i)].r     += creal(csum2); DMS[(j0+j+dim)*dim2 + (i0+i)].i     += cimag(csum2);
		  DMS[(i0+i)*dim2 + (j0+j+dim)].r     += creal(csum2); DMS[(i0+i)*dim2 + (j0+j+dim)].i     -= cimag(csum2);;

		} /* i */
	      } /* j */

	    } /* end of if (0<=h_AN2) */
	  } /* h_AN */          
	} /* if (0<=h_AN3) */
      } /* Lidx2 */
    } /* Lidx1 */

    /* calculation of DMS2 = DMS^t x DMS*/

    dcomplex alpha = {-1.0,0.0}; dcomplex beta = {0.0,0.0};
    F77_NAME(zgemm,ZGEMM)( "C", "N", &dim2, &dim2, &dim2, &alpha, 
			   DMS, &dim2, DMS, &dim2, &beta, DMS2, &dim2);

    /* diagonalization of DMS2 */

    lapack_zheev( dim2, DMS2, EVal );
    for (i=0; i<dim2; i++) EVal[i] = sqrt(fabs(EVal[i]));
    for (i=0; i<dim2*dim2; i++) EVec[i] = DMS2[i].r + DMS2[i].i*I; 

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

		  OLPgidx[i0+i    ][j0+j    ] = Smo2[Lidx1][h_AN][i][j];
		  OLPgidx[i0+i+dim][j0+j+dim] = Smo2[Lidx1][h_AN][i][j];

		  Hgidx[i0+i    ][j0+j    ] = Hmo2[0][Lidx1][h_AN][i][j];
		  Hgidx[i0+i+dim][j0+j+dim] = Hmo2[1][Lidx1][h_AN][i][j];
		  Hgidx[i0+i    ][j0+j+dim] = Hmo2[2][Lidx1][h_AN][i][j];
		  Hgidx[j0+j+dim][i0+i]     = conj(Hmo2[2][Lidx1][h_AN][i][j]);
		}
	      }
	    } 

	  } /* Lidx2 */
	} /* end of if (l1==0 && l2==0 && l3==0 ) */
      } /* h_AN */   
    } /* Lidx1 */           

    for (p=0; p<dim2; p++){

      /* calculate <EVec|S|EVec> */

      csum0 = 0.0;
      for (i=0; i<dim2; i++){
	for (j=0; j<dim2; j++){
	  csum0 += OLPgidx[i][j]*conj(EVec[dim2*p+i])*EVec[dim2*p+j];
	}
      }

      /* normalization of EVec */

      ctmp = 1.0/sqrt(fabs(creal(csum0))) + 0.0*I; 
       
      for (i=0; i<dim2; i++){
	EVec[dim2*p+i] *= ctmp;  
      }        

      /* calculate <EVec|H|EVec> */

      csum0 = 0.0;
      for (i=0; i<dim2; i++){
	for (j=0; j<dim2; j++){
	  csum0 += Hgidx[i][j]*conj(EVec[dim2*p+i])*EVec[dim2*p+j];
	}
      }

      WI[p] = creal(csum0); 

    } /* p */

    /* save the information of MOs into a file. */

    if (myid==Host_ID){

      fprintf(fp_MO_info,"# Group index=%2d\n",gidx+1);
      fprintf(fp_MO_info,"# 1st column: serial number\n");
      fprintf(fp_MO_info,"# 2nd column: on-site energy (eV) relative to chemical potential\n");
      fprintf(fp_MO_info,"# 3rd column: population\n");

      for (p=0; p<dim2; p++){
        fprintf(fp_MO_info,"%2d %18.12f %18.12f\n",p+1,(WI[p]-ChemP)*eV2Hartree,EVal[p]);
      }
    }    

    /* select MOs to be used */

    CWF_Guiding_MOs_NC[gidx] = (double complex**)malloc(sizeof(double complex*)*Num_CWF_MOs_Group[gidx]);
    for (i=0; i<Num_CWF_MOs_Group[gidx]; i++){
      CWF_Guiding_MOs_NC[gidx][i] = (double complex*)malloc(sizeof(double complex)*dim2);
    }             

    /* store EVec to select MOs to CWF_Guiding_MOs_NC */

    for (p=0; p<Num_CWF_MOs_Group[gidx]; p++){
      for (j=0; j<dim2; j++){
	CWF_Guiding_MOs_NC[gidx][p][j] = EVec[dim2*p+j];
      }
    }

    /* freeing of arrays */ 

    free(WI); 
    free(IWORK);
    free(DMS);
    free(DMS2);

    free(EVal);
    free(EVec);

    for (i=0; i<dim2; i++){
      free(OLPgidx[i]);
    }
    free(OLPgidx);

    for (i=0; i<dim2; i++){
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

    for (k=0; k<3; k++){
      for (Lidx=0; Lidx<CWF_Grouped_Atoms_EachNum[gidx]; Lidx++){

	GA_AN = CWF_Grouped_Atoms[gidx][Lidx];
	wan1 = WhatSpecies[GA_AN];
	tno1 = Spe_Total_CNO[wan1];

	for (h_AN=0; h_AN<=FNAN[GA_AN]; h_AN++){
	  for (i=0; i<tno1; i++){
	    free(DMmo2[k][Lidx][h_AN][i]);
	  }
          free(DMmo2[k][Lidx][h_AN]);
	}
        free(DMmo2[k][Lidx]);
      }
      free(DMmo2[k]);
    }
    free(DMmo2);

    for (k=0; k<3; k++){
      for (Lidx=0; Lidx<CWF_Grouped_Atoms_EachNum[gidx]; Lidx++){

	GA_AN = CWF_Grouped_Atoms[gidx][Lidx];
	wan1 = WhatSpecies[GA_AN];
	tno1 = Spe_Total_CNO[wan1];

	for (h_AN=0; h_AN<=FNAN[GA_AN]; h_AN++){
	  for (i=0; i<tno1; i++){
	    free(Hmo2[k][Lidx][h_AN][i]);
	  }
          free(Hmo2[k][Lidx][h_AN]);
	}
        free(Hmo2[k][Lidx]);
      }
      free(Hmo2[k]);
    }
    free(Hmo2);

  } /* gidx */  

  /* MPI_Bcast of CWF_Guiding_MOs_NC  */

  for (gidx=0; gidx<Num_CWF_Grouped_Atoms; gidx++){

    dim2 = 0;
    for (Lidx=0; Lidx<CWF_Grouped_Atoms_EachNum[gidx]; Lidx++){
      GA_AN = CWF_Grouped_Atoms[gidx][Lidx];
      wan1 = WhatSpecies[GA_AN];
      dim2 += 2*Spe_Total_CNO[wan1];
    }

    for (i=0; i<Num_CWF_MOs_Group[gidx]; i++){
      MPI_Bcast( &CWF_Guiding_MOs_NC[gidx][i][0], dim2, MPI_C_DOUBLE_COMPLEX, Host_ID, mpi_comm_level1 );
    }
  }

  /* close a file */
  if (myid==Host_ID) fclose(fp_MO_info);

  /* elapsed time */
  dtime(&TEtime);
  return (TEtime-TStime);
}




void lapack_zheev(int n0, dcomplex *A, double *W)
{
  char *name="lapack_zheev";
  char *JOBZ="V";
  char *UPLO="L";
  INTEGER n=n0;
  INTEGER LDA=n0;
  INTEGER LWORK;
  dcomplex *WORK;
  double *RWORK;
  INTEGER INFO;

  LWORK = 3*n;
  WORK  = (dcomplex*)malloc(sizeof(dcomplex)*LWORK);
  RWORK = (double*)malloc(sizeof(double)*LWORK);

  F77_NAME(zheev,ZHEEV)( JOBZ, UPLO, &n, A, &LDA, W, WORK, &LWORK, RWORK, &INFO ); 

  if (INFO>0) {
    printf("\n%s: error in zheev, info=%d\n\n",name,INFO);
  }

  if (INFO<0) {
     printf("%s: info=%d\n",name,INFO);
     exit(10);
  }

  free(WORK); 
  free(RWORK); 
}


void AllocateArrays_NonCol_QAO2()
{
  int Mc_AN,Gc_AN,spin,wan1,i;

  /* allocation of array */

  QAO2_coes = (dcomplex**)malloc(sizeof(dcomplex*)*(atomnum+1));
  for (Gc_AN=0; Gc_AN<(atomnum+1); Gc_AN++){
   
    if (Gc_AN==0){
      wan1 = 0;
    } 
    else{
      wan1 = WhatSpecies[Gc_AN];
    }

    QAO2_coes[Gc_AN] = (dcomplex*)malloc(sizeof(dcomplex)*(4*Spe_Total_CNO[wan1]*CWF_Num_predefined[wan1]));
    for (i=0; i<(4*Spe_Total_CNO[wan1]*CWF_Num_predefined[wan1]); i++){
      QAO2_coes[Gc_AN][i] = Complex(0.0,0.0);
    }
  }  
}


void FreeArrays_NonCol_QAO2()
{
  int Gc_AN,Mc_AN;

  for (Gc_AN=0; Gc_AN<(atomnum+1); Gc_AN++){
    free(QAO2_coes[Gc_AN]);
  }  
  free(QAO2_coes);
}

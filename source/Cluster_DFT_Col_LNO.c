/**********************************************************************************
  Cluster_DFT_Col_LNO.c:

  Cluster_DFT_Col_LNO.c is a subroutine to perform a cluster collinear calculation,
  where overlap and Hamiltonian matrices are contracted by localized natural atomic
  orbitals (LNOs).

  Log of Cluster_DFT_Col_LNO.c:

  02/Nov./2023  Released by T. Ozaki

**********************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "mpi.h"
#include "openmx_common.h"
#include "openmx_cusolver_dense_utils.h"
#include "lapack_prototypes.h"
#include "set_cuda_default_device_from_local_rank.h"
#include "set_openacc_device_from_local_rank.h"
#include <omp.h>
#include <openacc.h>

#define  measure_time   0

/* GPU CuSolver context for Cluster_DFT_Col_LNO (added by H.Kawai, Phase D) */
typedef struct {
    int                device_id;
    cublasHandle_t     cublas;
    cusolverDnHandle_t cusolver;
} ClusterColLNOCuSolverCtx;

static ClusterColLNOCuSolverCtx ClusterColLNO_cusolver_ctx = {0};

static void ClusterColLNO_CuSolver_Destroy(void)
{
    ClusterColLNOCuSolverCtx *ctx = &ClusterColLNO_cusolver_ctx;
    if (ctx->cusolver != NULL) wait_cudafunc(cusolverDnDestroy(ctx->cusolver));
    if (ctx->cublas != NULL)   wait_cudafunc(cublasDestroy(ctx->cublas));
    memset(ctx, 0, sizeof(*ctx));
    ctx->device_id = -1;
}

static void ClusterColLNO_CuSolver_Init(void)
{
    ClusterColLNOCuSolverCtx *ctx = &ClusterColLNO_cusolver_ctx;
    int current_device;
    wait_cudafunc(cudaGetDevice(&current_device));
    if (ctx->device_id == current_device && ctx->cublas != NULL) return;
    if (ctx->cublas != NULL) ClusterColLNO_CuSolver_Destroy();
    ctx->device_id = current_device;
    wait_cudafunc(cublasCreate(&ctx->cublas));
    wait_cudafunc(cusolverDnCreate(&ctx->cusolver));
}

static void ClusterColLNO_GEMMul8Dgemm_OpenACC(cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
                                               double const * A, double const * B, double * C)
{
    ClusterColLNO_CuSolver_Init();
#pragma acc data      present(A[0 : m * k], B[0 : k * n], C[0 : m * n])
#pragma acc host_data use_device(A, B, C)
    {
        double const alpha = 1.0;
        double const beta  = 0.0;
        wait_cudafunc(openmx_gemmul8Dgemm(ClusterColLNO_cusolver_ctx.cublas, transa, transb, m, n, k, &alpha, A, m, B, k, &beta, C, m));
    }
}

void solve_evp_real_( int *n1, int *n2, double *Cs, int *na_rows1, double *a, double *Ss, int *na_rows2, int *nblk, 
                      int *mpi_comm_rows_int, int *mpi_comm_cols_int);

void elpa_solve_evp_real_2stage_double_impl_( int *n1, int *n2, double *Cs, int *na_rows1, double *a, double *Ss, int *na_rows2, 
                                              int *nblk, int *na_cols1, int *mpi_comm_rows_int, int *mpi_comm_cols_int, int *mpiworld);



static double Lapack_LU_Dinverse(int n, double *A);


static double Calc_DM_Cluster_collinear_LNO(int myid0,
					    int numprocs0,
					    int myid1,
					    int numprocs1,
					    int myworld1,
					    int size_H1,
					    int *is2,
					    int *ie2,
					    int *MP,
					    int n,
					    int TNum_LNOs,
					    MPI_Comm *MPI_CommWD1,
					    int *Comm_World_StartID1,
					    double *****CDM,
					    double *****EDM,
					    double **ko,
					    double *DM1,
					    double *EDM1,
					    double *PDM1,
					    double *Work1,
					    double **EVec1, 
					    int *SP_NZeros,
					    int *SP_Atoms );


double Cluster_DFT_Col_LNO(
                   char *mode,
                   int SCF_iter,
                   int SpinP_switch,
		   int n,
		   int TNum_LNOs,
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
		   double **Ss_LNO,
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
  int i,j,l,n1,i1,i1s,j1,k1,l1;
  int wan,HOMO0,HOMO1;
  int spin,po,num0,num1,ires;
  int ct_AN,k,wanA,tnoA,wanB,tnoB;
  int GA_AN,Anum,loopN,Gc_AN;
  int MA_AN,LB_AN,GB_AN,Bnum,MaxN;
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

  /* GPU dispatch (added by H.Kawai): assign CUDA/OpenACC device when CuSOLVER is requested */
  if (scf_eigen_lib_flag == CuSOLVER && n >= GPU_CPU_SWITCH_NUM) {
      set_cuda_default_device_from_local_rank();
      set_openacc_nvidia_device_from_local_rank();
  }

  /* for time */
  MPI_Barrier(mpi_comm_level1);
  dtime(&TStime);

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs0);
  MPI_Comm_rank(mpi_comm_level1,&myid0);

  MPI_Comm_size(MPI_CommWD1[myworld1],&numprocs1);
  MPI_Comm_rank(MPI_CommWD1[myworld1],&myid1);

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
                   total core charge
  ****************************************************/

  TZ = 0.0;
  for (i=1; i<=atomnum; i++){
    wan = WhatSpecies[i];
    TZ = TZ + Spe_Core_Charge[wan];
  }
 
  /****************************************************
         find the numbers of partions for MPI
  ****************************************************/

  if ( numprocs1<=TNum_LNOs ){

    av_num = (double)TNum_LNOs/(double)numprocs1;

    for (ID=0; ID<numprocs1; ID++){
      is1[ID] = (int)(av_num*(double)ID) + 1; 
      ie1[ID] = (int)(av_num*(double)(ID+1)); 
    }

    is1[0] = 1;
    ie1[numprocs1-1] = TNum_LNOs; 

  }

  else{

    for (ID=0; ID<TNum_LNOs; ID++){
      is1[ID] = ID + 1; 
      ie1[ID] = ID + 1;
    }
    for (ID=TNum_LNOs; ID<numprocs1; ID++){
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

    if (SpinP_switch==1 && numprocs0==1){
      Overlap_Cluster_LNO_Ss(CntOLP,Cs,MP,0,0);
    }
    else{
      for (spin=0; spin<=SpinP_switch; spin++){
	Overlap_Cluster_LNO_Ss(CntOLP,Cs,MP,spin,myworld1);
      } 
    }

    /*
    printf("S\n");
    for (i=0; i<TNum_LNOs; i++){
      for (j=0; j<TNum_LNOs; j++){
        printf("%8.4f ",Cs[i*TNum_LNOs+j]);
      }
      printf("\n");
    }
    */

    /* spin=myworld1 */

    spin = myworld1;
    
 S_diagonalize:

    if (measure_time) dtime(&stime);

    MPI_Comm_split(MPI_CommWD1[myworld1],my_pcol,my_prow,&mpi_comm_rows);
    MPI_Comm_split(MPI_CommWD1[myworld1],my_prow,my_pcol,&mpi_comm_cols);

    mpi_comm_rows_int = MPI_Comm_c2f(mpi_comm_rows);
    mpi_comm_cols_int = MPI_Comm_c2f(mpi_comm_cols);

    if (scf_eigen_lib_flag==1){

      F77_NAME(solve_evp_real,SOLVE_EVP_REAL)(&TNum_LNOs, &TNum_LNOs, Cs, &na_rows, &ko[0][1], Ss_LNO[spin], &na_rows,
					      &nblk, &mpi_comm_rows_int, &mpi_comm_cols_int);
    }

    else if (scf_eigen_lib_flag==CuSOLVER && GPU_CPU_SWITCH_NUM<=TNum_LNOs && na_rows==TNum_LNOs && na_cols==TNum_LNOs){
      OpenMX_CuSolver_DenseDsyevx_1based(Cs,Ss_LNO[spin],ko[0],TNum_LNOs,TNum_LNOs,
                                         "Cluster_DFT_Col_LNO overlap CuSOLVER");
    }

    else if (scf_eigen_lib_flag==2 || scf_eigen_lib_flag==CuSOLVER){

#ifndef kcomp

      int mpiworld;
      mpiworld = MPI_Comm_c2f(MPI_CommWD1[myworld1]);

      F77_NAME(elpa_solve_evp_real_2stage_double_impl,ELPA_SOLVE_EVP_REAL_2STAGE_DOUBLE_IMPL)(
                         &TNum_LNOs, &TNum_LNOs, Cs, &na_rows, &ko[0][1], Ss_LNO[spin],
			 &na_rows, &nblk, &na_cols, &mpi_comm_rows_int, &mpi_comm_cols_int, &mpiworld);

#endif
      
    }

    MPI_Comm_free(&mpi_comm_rows);
    MPI_Comm_free(&mpi_comm_cols);

    /* print to the standard output */

    if (2<=level_stdout && myid0==Host_ID){
      for (l=1; l<=TNum_LNOs; l++){
	printf("  Eigenvalues of OLP  %2d  %18.15f\n",l,ko[0][l]);fflush(stdout);
      }
    }

    /* minus eigenvalues to 1.0e-10 */

    for (l=1; l<=TNum_LNOs; l++){
      if (ko[0][l]<0.0) ko[0][l] = 1.0e-10;
    }

    /* calculate S*1/sqrt(ko) */

    for (l=1; l<=TNum_LNOs; l++){
      ko[0][l] = 1.0/sqrt(ko[0][l]);
    }

    for(i=0; i<na_rows; i++){
      for(j=0; j<na_cols; j++){
	jg = np_cols*nblk*((j)/nblk) + (j)%nblk + ((np_cols+my_pcol)%np_cols)*nblk + 1;
	Ss_LNO[spin][j*na_rows+i] = Ss_LNO[spin][j*na_rows+i]*ko[0][jg];
      }
    }

    if (SpinP_switch==1 && numprocs0==1 && spin==0){
      spin++;
      Overlap_Cluster_LNO_Ss(CntOLP,Cs,MP,spin,spin);
      goto S_diagonalize; 
    }

    if (measure_time){
      dtime(&etime);
      time1 += etime - stime; 
    }
    
  } // if (SCF_iter==1)

  /****************************************************
    calculations of eigenvalues for up and down spins

     Note:
         MP indicates the starting position of
              atom i in arraies H and S
  ****************************************************/

  if (SpinP_switch==1 && numprocs0==1){
    Overlap_Cluster_LNO_Ss(nh[0],Hs,MP,0,0);
  }
  else{
    for (spin=0; spin<=SpinP_switch; spin++){
      Overlap_Cluster_LNO_Ss(nh[spin],Hs,MP,spin,myworld1);
    } 
  }
  
  MPI_Barrier(mpi_comm_level1);

  /* find the maximum states in solved eigenvalues */

  if (SCF_iter==1){
    MaxN = TNum_LNOs;
  }
  else{

    lumos = (double)TNum_LNOs*0.5;

    if (lumos<400.0) lumos = 400.0;
    MaxN = (TZ-system_charge)/2 + (int)lumos;
    if (TNum_LNOs<MaxN) MaxN = TNum_LNOs;
  }

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
      is2[ID] = 1;
      ie2[ID] = 0;
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

  /* initialize ko */
  for (spin=0; spin<=SpinP_switch; spin++){
    for (i1=1; i1<=TNum_LNOs; i1++){
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
  F77_NAME(pdgemm,PDGEMM)("N","N",&TNum_LNOs,&TNum_LNOs,&TNum_LNOs,&alpha,Hs,&ONE,&ONE,
			  descH,Ss_LNO[spin],&ONE,&ONE,descS,&beta,Cs,&ONE,&ONE,descC);

  /* 1.0/sqrt(ko[l]) * U^+ H * U * 1.0/sqrt(ko[l]) */

  for(i=0; i<na_rows*na_cols; i++){
    Hs[i] = 0.0;
  }

  Cblacs_barrier(ictxt1,"C");
  F77_NAME(pdgemm,PDGEMM)("T","N",&TNum_LNOs,&TNum_LNOs,&TNum_LNOs,&alpha,Ss_LNO[spin],&ONE,&ONE,
			  descS,Cs,&ONE,&ONE,descC,&beta,Hs,&ONE,&ONE,descH);
  
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
    F77_NAME(solve_evp_real,SOLVE_EVP_REAL)(&TNum_LNOs, &MaxN, Hs, &na_rows, &ko[spin][1], Cs, 
                                            &na_rows, &nblk, &mpi_comm_rows_int, &mpi_comm_cols_int);
  }
  else if (scf_eigen_lib_flag==CuSOLVER && GPU_CPU_SWITCH_NUM<=TNum_LNOs && na_rows==TNum_LNOs && na_cols==TNum_LNOs){
    OpenMX_CuSolver_DenseDsyevx_1based(Hs,Cs,ko[spin],TNum_LNOs,MaxN,
                                       "Cluster_DFT_Col_LNO Hamiltonian CuSOLVER");
  }

  else if (scf_eigen_lib_flag==2 || scf_eigen_lib_flag==CuSOLVER){

#ifndef kcomp
    int mpiworld;
    mpiworld = MPI_Comm_c2f(MPI_CommWD1[myworld1]);

    F77_NAME(elpa_solve_evp_real_2stage_double_impl,
	     ELPA_SOLVE_EVP_REAL_2STAGE_DOUBLE_IMPL)(
	       &TNum_LNOs, &MaxN, Hs, &na_rows, &ko[spin][1], 
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
  F77_NAME(pdgemm,PDGEMM)( "T","T",&TNum_LNOs,&TNum_LNOs,&TNum_LNOs,&alpha,Cs,&ONE,&ONE,
			   descC,Ss_LNO[spin],&ONE,&ONE,descS,&beta,Hs,&ONE,&ONE,descH );

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
      m = (jg-1)*(ie2[myid1]-is2[myid1]+1)+ig-is2[myid1]; 
      EVec1[spin][m] = EVec_Rcv[k];
    }
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

  /*
  printf("EVec1\n");
  for (i=0; i<n; i++){
    for (j=0; j<8; j++){
      printf("%8.5f ",EVec1[0][i*MaxN+j]);
    } 
    printf("\n");
  } 
  */

  /*********************************************** 
    MPI: ko
  ***********************************************/

  if (measure_time) dtime(&stime);

  for (sp=0; sp<=SpinP_switch; sp++){
    MPI_Bcast(&ko[sp][1],MaxN,MPI_DOUBLE,Comm_World_StartID1[sp],mpi_comm_level1);
  }

  if ( strcasecmp(mode,"scf")==0 ){

    if (2<=level_stdout){
      for (i1=1; i1<=MaxN; i1++){
	if (SpinP_switch==0)
	  printf("  Eigenvalues of Kohn-Sham %2d %15.12f %15.12f\n",
		 i1,ko[0][i1],ko[0][i1]);
	else 
	  printf("  Eigenvalues of Kohn-Sham %2d %15.12f %15.12f\n",
		 i1,ko[0][i1],ko[1][i1]);
      }
    }

    /****************************************************
              searching of chemical potential
    ****************************************************/

    double Beta_trial1;

    /* first, find ChemP at 1200 K */

    Beta_trial1 = 1.0/kB/(1200.0/eV2Hartree);

    po = 0;
    loopN = 0;

    ChemP_MAX = 30.0;  
    ChemP_MIN =-30.0;
  
    do {

      ChemP = 0.50*(ChemP_MAX + ChemP_MIN);
      Num_State = 0.0;

      for (spin=0; spin<=SpinP_switch; spin++){
	for (i1=1; i1<=MaxN; i1++){
	  x = (ko[spin][i1] - ChemP)*Beta_trial1;
	  if (x<=-max_x) x = -max_x;
	  if (max_x<=x)  x = max_x;
	  FermiF = 1.0/(1.0 + exp(x));
	  Num_State = Num_State + spin_degeneracy*FermiF;
	  if (0.5<FermiF) Cluster_HOMO[spin] = i1;
	}
      }

      Dnum = (TZ - Num_State) - system_charge;
      if (0.0<=Dnum) ChemP_MIN = ChemP;
      else           ChemP_MAX = ChemP;
      if (fabs(Dnum)<1.0e-12) po = 1;

      if (myid1==Host_ID && 2<=level_stdout){
	printf("ChemP=%15.12f TZ=%15.12f Num_state=%15.12f\n",ChemP,TZ,Num_State); 
      }

      loopN++;

    } while (po==0 && loopN<1000); 

    /* second, find ChemP at the temperatue, starting from the previously found ChemP. */

    po = 0;
    loopN = 0;

    ChemP_MAX = 30.0;  
    ChemP_MIN =-30.0;
  
    do {

      if (loopN!=0){
	ChemP = 0.50*(ChemP_MAX + ChemP_MIN);
      }

      Num_State = 0.0;

      for (spin=0; spin<=SpinP_switch; spin++){
	for (i1=1; i1<=MaxN; i1++){
	  x = (ko[spin][i1] - ChemP)*Beta;
	  if (x<=-max_x) x = -max_x;
	  if (max_x<=x)  x = max_x;
	  FermiF = 1.0/(1.0 + exp(x));

	  Num_State = Num_State + spin_degeneracy*FermiF;
	  if (0.5<FermiF) Cluster_HOMO[spin] = i1;
	}
      }

      Dnum = (TZ - Num_State) - system_charge;
      if (0.0<=Dnum) ChemP_MIN = ChemP;
      else           ChemP_MAX = ChemP;
      if (fabs(Dnum)<1.0e-12) po = 1;

      if (myid1==Host_ID && 2<=level_stdout){
	printf("ChemP=%15.12f TZ=%15.12f Num_state=%15.12f\n",ChemP,TZ,Num_State); 
      }

      loopN++;

    } 
    while (po==0 && loopN<1000); 

    if (2<=level_stdout){
      printf("  ChemP=%15.12f\n",ChemP);
    }

    if (measure_time){
      dtime(&etime);
      time5 += etime - stime;
    }

    /****************************************************
          Energies by summing up eigenvalues
    ****************************************************/

    Eele0[0] = 0.0;
    Eele0[1] = 0.0;

    for (spin=0; spin<=SpinP_switch; spin++){
      for (i1=1; i1<=MaxN; i1++){

	if (xanes_calc==1) 
          x = (ko[spin][i1] - ChemP_XANES[spin])*Beta;
        else 
          x = (ko[spin][i1] - ChemP)*Beta;

	if (x<=-max_x) x = -max_x;
	if (max_x<=x)  x = max_x;
	FermiF = FermiFunc(x,spin,i1,&po,&x);

	Eele0[spin] += ko[spin][i1]*FermiF;

      }
    }

    if (SpinP_switch==0){
      Eele0[1] = Eele0[0];
    }

    /****************************************************
          density matrix and energy density matrix
                  for up and down spins
    ****************************************************/

    time6 += Calc_DM_Cluster_collinear_LNO( myid0,numprocs0,myid1,numprocs1,myworld1,
					    size_H1,is2,ie2,MP,n,TNum_LNOs,MPI_CommWD1,Comm_World_StartID1,
					    CDM,EDM,ko,CDM1,EDM1,PDM1,Work1,EVec1,SP_NZeros,SP_Atoms);

    /****************************************************
                        Bond Energies
    ****************************************************/
  
    My_Eele1[0] = 0.0;
    My_Eele1[1] = 0.0;

    for (spin=0; spin<=SpinP_switch; spin++){
      for (MA_AN=1; MA_AN<=Matomnum; MA_AN++){
	GA_AN = M2G[MA_AN];
	wanA = WhatSpecies[GA_AN];
	tnoA = Spe_Total_CNO[wanA];
	for (j=0; j<=FNAN[GA_AN]; j++){
	  wanB = WhatSpecies[natn[GA_AN][j]];
	  tnoB = Spe_Total_CNO[wanB];
	  for (k=0; k<tnoA; k++){
	    for (l=0; l<tnoB; l++){
	      My_Eele1[spin] += CDM[spin][MA_AN][j][k][l]*nh[spin][MA_AN][j][k][l];
	    }
	  }
	}
      }
    }
  
    /* MPI, My_Eele1 */
    for (spin=0; spin<=SpinP_switch; spin++){
      MPI_Allreduce(&My_Eele1[spin], &Eele1[spin], 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
    }

    if (SpinP_switch==0) Eele1[1] = Eele1[0];

    if (2<=level_stdout && myid0==Host_ID){
      printf("Eele0[0]=%15.12f Eele0[1]=%15.12f\n",Eele0[0],Eele0[1]);
      printf("Eele1[0]=%15.12f Eele1[1]=%15.12f\n",Eele1[0],Eele1[1]);
    }

    if (measure_time){
      dtime(&etime);
      time6 += etime - stime;
    }

    if (measure_time){
      dtime(&etime);
      time7 += etime - stime;
    }

  } /* if ( strcasecmp(mode,"scf")==0 ) */

  else if ( strcasecmp(mode,"diag")==0 ){
    /* nothing is done. */
  }

  if (measure_time){
    printf("Cluster_DFT myid=%2d time1=%7.3f time2=%7.3f time3=%7.3f time4=%7.3f time5=%7.3f time6=%7.3f time7=%7.3f\n",
            myid0,time1,time2,time3,time4,time5,time6,time7);fflush(stdout); 
  }

  /****************************************************
                          Free
  ****************************************************/

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

  /* for elapsed time */

  MPI_Barrier(mpi_comm_level1);
  dtime(&TEtime);
  time0 = TEtime - TStime;
  return time0;
}





double Calc_DM_Cluster_collinear_LNO(
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
    int TNum_LNOs,
    MPI_Comm *MPI_CommWD1,
    int *Comm_World_StartID1,
    double *****CDM,
    double *****EDM,
    double **ko,
    double *DM1,
    double *EDM1,
    double *PDM1,
    double *Work1,
    double **EVec1, 
    int *SP_NZeros,
    int *SP_Atoms )
{
  int spin,i,j,i0,j0,k,kmin,kmax,po,p,GA_AN,MA_AN,wanA,tnoA,Anum;
  int LB_AN,GB_AN,wanB,tnoB,Bnum,i1,j1,ID,tnoA0,tnoB0,MB_AN;
  double max_x=60.0,dum;
  double FermiF,FermiF2,x,x2,diffF,sum1,sum2;
  double FermiEps = 1.0e-13;
  double stime,etime,time,lumos;
  MPI_Status stat;
  MPI_Request request;
  double *FF,*dFF;
  double *CDM_LNO,*EDM_LNO,*Mat1,*Mat2;
  double **TmpEVec0,**TmpEVec1;

  dtime(&stime);

  kmin = is2[myid1];
  kmax = ie2[myid1];

  /* allocation of arrays */

  FF = (double*)malloc(sizeof(double)*(TNum_LNOs+1));
  dFF = (double*)malloc(sizeof(double)*(TNum_LNOs+1));

  TmpEVec0 = (double**)malloc(sizeof(double*)*List_YOUSO[7]);
  for (i=0; i<List_YOUSO[7]; i++){
    TmpEVec0[i] = (double*)malloc(sizeof(double)*(kmax-kmin+1));
  }

  TmpEVec1 = (double**)malloc(sizeof(double*)*List_YOUSO[7]);
  for (i=0; i<List_YOUSO[7]; i++){
    TmpEVec1[i] = (double*)malloc(sizeof(double)*(kmax-kmin+1));
  }

  CDM_LNO = (double*)malloc(sizeof(double)*List_YOUSO[7]*List_YOUSO[7]);
  EDM_LNO = (double*)malloc(sizeof(double)*List_YOUSO[7]*List_YOUSO[7]);
  Mat1 = (double*)malloc(sizeof(double)*List_YOUSO[7]*List_YOUSO[7]);
  Mat2 = (double*)malloc(sizeof(double)*List_YOUSO[7]*List_YOUSO[7]);
  
  /* spin=myworld1 */

  spin = myworld1;

 calc_dm_collinear:

  /* initialize DM1 */

  for (i=0; i<size_H1; i++){
    DM1[i] = 0.0;
    EDM1[i] = 0.0;
  }

  /* pre-calculation of Fermi Function */ 

  po = 0;

  for (k=is2[myid1]; k<=ie2[myid1]; k++){

    x = (ko[spin][k] - ChemP)*Beta;

    if (x<=-max_x) x = -max_x;
    if (max_x<=x)  x = max_x;
    FermiF = FermiFunc(x,spin,k,&po,&x);

    FF[k] = FermiF;
  }
  
  /* calculation of DM1 */

  p = 0;
  for (GA_AN=1; GA_AN<=atomnum; GA_AN++){

    wanA = WhatSpecies[GA_AN];
    tnoA = LNOs_Num_predefined[wanA];
    Anum = MP[GA_AN];

    /* store EVec1 to a temporal array */

    for (i=0; i<tnoA; i++){
      i0 = (Anum + i - 1)*(ie2[myid1]-is2[myid1]+1) - is2[myid1];
      for (k=kmin; k<=kmax; k++){
        TmpEVec0[i][k-kmin] = EVec1[spin][i0+k];
      }        
    }

    /* loop for LB_AN */

    for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){

      GB_AN = natn[GA_AN][LB_AN];
      wanB = WhatSpecies[GB_AN];
      tnoB = LNOs_Num_predefined[wanB];
      Bnum = MP[GB_AN];

      /* store EVec1 to a temporal array */

      for (j=0; j<tnoB; j++){
        j0 = (Bnum + j - 1)*(ie2[myid1]-is2[myid1]+1) - is2[myid1];
	for (k=kmin; k<=kmax; k++){
	  TmpEVec1[j][k-kmin] = EVec1[spin][j0+k];
	}        
      }

      /* loops for i and j */

      for (i=0; i<tnoA; i++){
	for (j=0; j<tnoB; j++){

          sum1 = 0.0;
          sum2 = 0.0;

	  for (k=kmin; k<=kmax; k++){
	    dum = FF[k]*TmpEVec0[i][k-kmin]*TmpEVec1[j][k-kmin];
	    sum1 += dum;
	    sum2 += dum*ko[spin][k];
	  }

	  DM1[p]  = sum1;
	  EDM1[p] = sum2;

	  /* increment of p */
	  p++;  

	}
      }
    }
  } /* GA_AN */
  
  /* MPI_Allreduce */

  MPI_Allreduce(DM1, Work1, size_H1, MPI_DOUBLE, MPI_SUM, MPI_CommWD1[myworld1]);
  for (i=0; i<size_H1; i++) DM1[i] = Work1[i];

  MPI_Allreduce(EDM1, Work1, size_H1, MPI_DOUBLE, MPI_SUM, MPI_CommWD1[myworld1]);
  for (i=0; i<size_H1; i++) EDM1[i] = Work1[i];
  
  /* store DM1 to a proper place */

  p = 0;
  for (GA_AN=1; GA_AN<=atomnum; GA_AN++){

    MA_AN = F_G2M[GA_AN];
    wanA = WhatSpecies[GA_AN];
    tnoA0 = Spe_Total_CNO[wanA];
    tnoA = LNOs_Num_predefined[wanA];
    Anum = MP[GA_AN];
    ID = G2ID[GA_AN];

    for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){

      GB_AN = natn[GA_AN][LB_AN];
      MB_AN = S_G2M[GB_AN];
      wanB = WhatSpecies[GB_AN];
      tnoB0 = Spe_Total_CNO[wanB];
      tnoB = LNOs_Num_predefined[wanB];
      Bnum = MP[GB_AN];

      if (myid0==ID){
	for (i=0; i<tnoA; i++){
	  for (j=0; j<tnoB; j++){

            CDM_LNO[i*tnoB+j] = DM1[p];
            EDM_LNO[i*tnoB+j] = EDM1[p];
	    
	    /* increment of p */
	    p++;  
	  }
	}

	/* transform CDM_LNO and EDM_LNO to those expressed by the original basis set */

	for (i=0; i<tnoA0; i++){
	  for (j=0; j<tnoB; j++){

	    sum1 = 0.0;
	    sum2 = 0.0;

	    for (k=0; k<tnoA; k++){
	      sum1 += LNO_coes[spin][MA_AN][tnoA0*k+i]*CDM_LNO[k*tnoB+j];
	      sum2 += LNO_coes[spin][MA_AN][tnoA0*k+i]*EDM_LNO[k*tnoB+j];
	    }

	    Mat1[i*tnoB+j] = sum1;
	    Mat2[i*tnoB+j] = sum2;
	  
	  } // j
	} // i
      
	for (i=0; i<tnoA0; i++){
	  for (j=0; j<tnoB0; j++){

	    sum1 = 0.0;
	    sum2 = 0.0;

	    for (k=0; k<tnoB; k++){
	      sum1 += Mat1[i*tnoB+k]*LNO_coes[spin][MB_AN][tnoB0*k+j];
	      sum2 += Mat2[i*tnoB+k]*LNO_coes[spin][MB_AN][tnoB0*k+j];
	    }

	    CDM[spin][MA_AN][LB_AN][i][j] = sum1;
	    EDM[spin][MA_AN][LB_AN][i][j] = sum2;
	  
	  } // j
	} // i

      }
      else{
	for (i=0; i<tnoA; i++){
	  for (j=0; j<tnoB; j++){
	    /* increment of p */
	    p++;  
	  }
	}
      }

    } /* LB_AN */
  } /* GA_AN */

  if (SpinP_switch==1 && numprocs0==1 && spin==0){
    spin++;  
    goto calc_dm_collinear;
  }
 
  else if (SpinP_switch==1 && numprocs0!=1){

    /* MPI communication of DM1 */

    if (Comm_World_StartID1[0]==myid0){
      MPI_Isend(DM1,size_H1,MPI_DOUBLE,Comm_World_StartID1[1],10,mpi_comm_level1,&request);
    }
    if (Comm_World_StartID1[1]==myid0){
      MPI_Isend(DM1,size_H1,MPI_DOUBLE,Comm_World_StartID1[0],20,mpi_comm_level1,&request);
    }
    if (Comm_World_StartID1[1]==myid0){
      MPI_Recv(Work1,size_H1,MPI_DOUBLE,Comm_World_StartID1[0],10,mpi_comm_level1,&stat);
      MPI_Wait(&request,&stat);
    }
    if (Comm_World_StartID1[0]==myid0){
      MPI_Recv(Work1,size_H1,MPI_DOUBLE,Comm_World_StartID1[1],20,mpi_comm_level1,&stat);
      MPI_Wait(&request,&stat);
    }

    MPI_Bcast(Work1, size_H1, MPI_DOUBLE, 0, MPI_CommWD1[myworld1]);  
    for (i=0; i<size_H1; i++) DM1[i] = Work1[i];

    /* MPI communication of EDM1 */

    if (Comm_World_StartID1[0]==myid0){
      MPI_Isend(EDM1,size_H1,MPI_DOUBLE,Comm_World_StartID1[1],10,mpi_comm_level1,&request);
    }
    if (Comm_World_StartID1[1]==myid0){
      MPI_Isend(EDM1,size_H1,MPI_DOUBLE,Comm_World_StartID1[0],20,mpi_comm_level1,&request);
    }
    if (Comm_World_StartID1[1]==myid0){
      MPI_Recv(Work1,size_H1,MPI_DOUBLE,Comm_World_StartID1[0],10,mpi_comm_level1,&stat);
      MPI_Wait(&request,&stat);
    }
    if (Comm_World_StartID1[0]==myid0){
      MPI_Recv(Work1,size_H1,MPI_DOUBLE,Comm_World_StartID1[1],20,mpi_comm_level1,&stat);
      MPI_Wait(&request,&stat);
    }

    MPI_Bcast(Work1, size_H1, MPI_DOUBLE, 0, MPI_CommWD1[myworld1]);  
    for (i=0; i<size_H1; i++) EDM1[i] = Work1[i];

    /* store DM1 to a proper place */

    if      (myworld1==0) spin = 1;
    else if (myworld1==1) spin = 0;

    p = 0;
    for (GA_AN=1; GA_AN<=atomnum; GA_AN++){

      MA_AN = F_G2M[GA_AN];
      wanA = WhatSpecies[GA_AN];
      tnoA0 = Spe_Total_CNO[wanA];
      tnoA = LNOs_Num_predefined[wanA];
      Anum = MP[GA_AN];
      ID = G2ID[GA_AN];

      for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
	
	GB_AN = natn[GA_AN][LB_AN];
        MB_AN = S_G2M[GB_AN];
	wanB = WhatSpecies[GB_AN];
        tnoB0 = Spe_Total_CNO[wanB];
        tnoB = LNOs_Num_predefined[wanB];
	Bnum = MP[GB_AN];

	if (myid0==ID){
         
	  for (i=0; i<tnoA; i++){
	    for (j=0; j<tnoB; j++){

              CDM_LNO[i*tnoB+j] = DM1[p];
              EDM_LNO[i*tnoB+j] = EDM1[p];
	      
	      /* increment of p */
	      p++;  
	    }
	  }
	}
	else{
	  for (i=0; i<tnoA; i++){
	    for (j=0; j<tnoB; j++){
	      /* increment of p */
	      p++;  
	    }
	  }
	}

        /* transform CDM_LNO and EDM_LNO to those expressed by the original basis set */

	for (i=0; i<tnoA0; i++){
	  for (j=0; j<tnoB; j++){

	    sum1 = 0.0;
	    sum2 = 0.0;

	    for (k=0; k<tnoA; k++){
	      sum1 += LNO_coes[spin][MA_AN][tnoA0*k+i]*CDM_LNO[k*tnoB+j];
	      sum2 += LNO_coes[spin][MA_AN][tnoA0*k+i]*EDM_LNO[k*tnoB+j];
	    }

	    Mat1[i*tnoB+j] = sum1;
	    Mat2[i*tnoB+j] = sum2;
	  
	  } // j
	} // i
      
	for (i=0; i<tnoA0; i++){
	  for (j=0; j<tnoB0; j++){

	    sum1 = 0.0;
	    sum2 = 0.0;

	    for (k=0; k<tnoB; k++){
	      sum1 += Mat1[i*tnoB+k]*LNO_coes[spin][MB_AN][tnoB0*k+j];
	      sum2 += Mat2[i*tnoB+k]*LNO_coes[spin][MB_AN][tnoB0*k+j];
	    }

	    CDM[spin][MA_AN][LB_AN][i][j] = sum1;
	    EDM[spin][MA_AN][LB_AN][i][j] = sum2;

	    //printf("%7.4f ",sum1);
	  } // j
	  //printf("\n");
	} // i
	
      } /* LB_AN */
    } /* GA_AN */
  } /* end of else if (SpinP_switch==1) */

  /* freeing of arrays */

  free(dFF);
  free(FF);

  for (i=0; i<List_YOUSO[7]; i++){
    free(TmpEVec0[i]);
  }
  free(TmpEVec0);

  for (i=0; i<List_YOUSO[7]; i++){
    free(TmpEVec1[i]);
  }
  free(TmpEVec1);

  free(CDM_LNO);
  free(EDM_LNO);
  free(Mat1);
  free(Mat2);

  dtime(&etime);
  return (etime-stime);

}


  




double Lapack_LU_Dinverse(int n, double *A)
{
    static char *thisprogram="Lapack_LU_inverse";
    int *ipiv;
    double *work,tmp,det;
    int lwork;
    int info,i,j;

    /* L*U factorization */

    ipiv = (int*) malloc(sizeof(int)*n);

    F77_NAME(dgetrf,DGETRF)(&n,&n,A,&n,ipiv,&info);

    if ( info !=0 ) {
      printf("dgetrf failed, info=%i, %s\n",info,thisprogram);
    }

    /* calculation of determinant */

    det = 1.0; 
    for (i=0; i<n; i++){
      tmp = det;
      det = tmp*A[n*(i)+(i)];
    }

    for (i=0; i<n; i++){
      if (ipiv[i] != i+1) { det = -det; }
    }

    /* 
    printf("det %15.12f\n",det);   

    for (i=0; i<n; i++){
      printf("i=%2d ipiv=%2d\n",i,ipiv[i]);
    }
    */

    /* inverse L*U factorization */

    lwork = 4*n;
    work = (double*)malloc(sizeof(double)*lwork);

    F77_NAME(dgetri,DGETRI)(&n, A, &n, ipiv, work, &lwork, &info);

    if ( info !=0 ) {
      printf("dgetrf failed, info=%i, %s\n",info,thisprogram);
    }

    free(work); free(ipiv);

    return det;
}

/**********************************************************************
  Cluster_DFT_NonCol_DMmu.c:

     Cluster_DFT_NonCol_DMmu.c is a subroutine to perform non-collinear 
     cluster calculations.

  Log of Cluster_DFT_NonCol_DMmu.c:

     21/Feb./2019  Released by T. Ozaki
     03/Jul./2023  Modified by M. Fukuda

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "openmx_common.h"
#include "openmx_cusolver_dense_utils.h"
#include "mpi.h"
#include "set_cuda_default_device_from_local_rank.h"
#include "set_openacc_device_from_local_rank.h"
#include <omp.h>
#include <openacc.h>
#include "flpq_dm.h"
#include "lapack_prototypes.h"

#define  measure_time   0

/* GPU CuSolver context for Cluster_DFT_NonCol_DMmu (added by H.Kawai, Phase D) */
typedef struct {
    int                device_id;
    cublasHandle_t     cublas;
    cusolverDnHandle_t cusolver;
} ClusterNonColDMmuCuSolverCtx;

static ClusterNonColDMmuCuSolverCtx ClusterNonColDMmu_cusolver_ctx = {0};

static void ClusterNonColDMmu_CuSolver_Destroy(void)
{
    ClusterNonColDMmuCuSolverCtx *ctx = &ClusterNonColDMmu_cusolver_ctx;
    if (ctx->cusolver != NULL) wait_cudafunc(cusolverDnDestroy(ctx->cusolver));
    if (ctx->cublas != NULL)   wait_cudafunc(cublasDestroy(ctx->cublas));
    memset(ctx, 0, sizeof(*ctx));
    ctx->device_id = -1;
}

static void ClusterNonColDMmu_CuSolver_Init(void)
{
    ClusterNonColDMmuCuSolverCtx *ctx = &ClusterNonColDMmu_cusolver_ctx;
    int current_device;
    wait_cudafunc(cudaGetDevice(&current_device));
    if (ctx->device_id == current_device && ctx->cublas != NULL) return;
    if (ctx->cublas != NULL) ClusterNonColDMmu_CuSolver_Destroy();
    ctx->device_id = current_device;
    wait_cudafunc(cublasCreate(&ctx->cublas));
    wait_cudafunc(cusolverDnCreate(&ctx->cusolver));
}

static void ClusterNonColDMmu_GEMMul8Zgemm_OpenACC(cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
                                                   dcomplex const * A, dcomplex const * B, dcomplex * C)
{
    ClusterNonColDMmu_CuSolver_Init();
#pragma acc data      present(A[0 : m * k], B[0 : k * n], C[0 : m * n])
#pragma acc host_data use_device(A, B, C)
    {
        cuDoubleComplex const alpha = make_cuDoubleComplex(1.0, 0.0);
        cuDoubleComplex const beta  = make_cuDoubleComplex(0.0, 0.0);
        wait_cudafunc(openmx_gemmul8Zgemm(ClusterNonColDMmu_cusolver_ctx.cublas, transa, transb, m, n, k, &alpha,
                                          (cuDoubleComplex const *)A, m, (cuDoubleComplex const *)B, k, &beta,
                                          (cuDoubleComplex *)C, m));
    }
}


void solve_evp_real_( int *n1, int *n2, double *Cs, int *na_rows1, double *a, double *Ss, int *na_rows2, int *nblk, 
                      int *mpi_comm_rows_int, int *mpi_comm_cols_int);

void elpa_solve_evp_real_2stage_double_impl_( int *n1, int *n2, double *Cs, int *na_rows1, double *a, double *Ss, int *na_rows2, 
                                              int *nblk, int *na_cols1, int *mpi_comm_rows_int, int *mpi_comm_cols_int, int *mpiworld);

void solve_evp_complex_( int *n2, int *MaxN, dcomplex *Hs2, int *na_rows2_1, double *a, dcomplex *Cs2, int *na_rows2_2, 
                         int *nblk2, int *mpi_comm_rows_int, int *mpi_comm_cols_int );

void elpa_solve_evp_complex_2stage_double_impl_( int *n2, int *MaxN, dcomplex *Hs2, int *na_rows2_1, double *a, dcomplex *Cs2, 
                                                 int *na_rows2_2, int *nblk2, int *na_cols2, 
                                                 int *mpi_comm_rows_int, int *mpi_comm_cols_int, int *mpiworld );

double Cluster_DFT_NonCol_DMmu(
                   char *mode,
                   int SCF_iter,
                   int SpinP_switch,
                   double *ko,
                   double *****nh,
                   double *****ImNL,
                   double ****CntOLP,
                   double *****CDM,
                   double *****iCDM,
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
  int MA_AN,LB_AN,GB_AN,Bnum,MaxN;
  int *is1,*ie1;
  double TZ,my_sum,sum,sumE,max_x=60.0;
  double My_Eele1[2];
  double Num_State,x,FermiF,Dnum,Dnum2;
  double FermiF2,x2,diffF;
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
  int ig,jg;
  int numprocs,myid,ID;
  int ke,ks,nblk_m,nblk_m2;
  int ID0,IDS,IDR,Max_Num_Snd_EV,Max_Num_Rcv_EV;
  int *Num_Snd_EV,*Num_Rcv_EV;
  int *index_Snd_i,*index_Snd_j,*index_Rcv_i,*index_Rcv_j;
  double *EVec_Snd,*EVec_Rcv;
  int ZERO=0,ONE=1,info;
  dcomplex alpha = {1.0,0.0}; dcomplex beta = {0.0,0.0};
  double Re_alpha=1.0;
  double Re_beta=0.0;
  
  MPI_Comm mpi_comm_rows, mpi_comm_cols;
  int mpi_comm_rows_int,mpi_comm_cols_int;
  MPI_Status stat;
  MPI_Request request;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  MPI_Barrier(mpi_comm_level1);
  dtime(&TStime);

  /* ***************************************************
             calculation of the array size
  *************************************************** */

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

  /* ***************************************************
                  allocation of arrays
  *************************************************** */

  is1 = (int*)malloc(sizeof(int)*numprocs);
  ie1 = (int*)malloc(sizeof(int)*numprocs);

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

  /* ***************************************************
                  total core charge
  *************************************************** */

  TZ = 0.0;
  for (i=1; i<=atomnum; i++){
    wan = WhatSpecies[i];
    TZ += Spe_Core_Charge[wan];
  }

  /* ***************************************************
         find the numbers of partions for MPI
  *************************************************** */

  if ( numprocs<=n ){

    av_num = (double)n/(double)numprocs;

    for (ID=0; ID<numprocs; ID++){
      is1[ID] = (int)(av_num*(double)ID) + 1; 
      ie1[ID] = (int)(av_num*(double)(ID+1)); 
    }

    is1[0] = 1;
    ie1[numprocs-1] = n; 
  }

  else{

    for (ID=0; ID<n; ID++){
      is1[ID] = ID + 1; 
      ie1[ID] = ID + 1;
    }

    for (ID=n; ID<numprocs; ID++){
      is1[ID] = 1;
      ie1[ID] = 0;
    }
  }

  /* find the maximum states in solved eigenvalues */
  
  if (SCF_iter==1){
    MaxN = n2; 
  }   
  else {

    if      ( strcasecmp(mode,"scf")==0 ) 
      lumos = (double)n2*0.20;      
    else if ( strcasecmp(mode,"dos")==0 )
      lumos = (double)n2*0.40;
    else if ( strcasecmp(mode,"lcaoout")==0 )
      lumos = (double)n2*0.40;
    else if ( strcasecmp(mode,"xanes")==0 )
      lumos = (double)n2*0.40;

    if (lumos<400.0) lumos = 400.0;
    MaxN = Cluster_HOMO[0] + (int)lumos;
    if (n2<MaxN) MaxN = n2;

    if (cal_partial_charge) MaxN = n2; 
  }

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

  /* print memory size */

  if (firsttime && memoryusage_fileout){
    PrintMemory("Cluster_DFT_NonCol: is1",sizeof(int)*numprocs,NULL);
    PrintMemory("Cluster_DFT_NonCol: ie1",sizeof(int)*numprocs,NULL);
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

  /* ***************************************************
            diagonalize the overlap matrix     
  *************************************************** */

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
      OpenMX_CuSolver_DenseDsyevx_1based(Cs,Ss,ko,n,n,"Cluster_DFT_NonCol_DMmu overlap CuSOLVER");
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
  F77_NAME(pdgemm,PDGEMM)("N","N",&n,&n,&n,&Re_alpha,rHs11,&ONE,&ONE,descH,Ss,&ONE,&ONE,descS,&Re_beta,Cs,&ONE,&ONE,descC);

  for (i=0; i<na_rows*na_cols; i++) rHs11[i] = 0.0;
  
  Cblacs_barrier(ictxt1,"C");
  F77_NAME(pdgemm,PDGEMM)("T","N",&n,&n,&n,&Re_alpha,Ss,&ONE,&ONE,descS,Cs,&ONE,&ONE,descC,&Re_beta,rHs11,&ONE,&ONE,descH);

  /* S^t x rHs12 x S */

  for (i=0; i<na_rows*na_cols; i++) Cs[i] = 0.0;

  Cblacs_barrier(ictxt1,"A");
  F77_NAME(pdgemm,PDGEMM)("N","N",&n,&n,&n,&Re_alpha,rHs12,&ONE,&ONE,descH,Ss,&ONE,&ONE,descS,&Re_beta,Cs,&ONE,&ONE,descC);

  for (i=0; i<na_rows*na_cols; i++) rHs12[i] = 0.0;

  Cblacs_barrier(ictxt1,"C");
  F77_NAME(pdgemm,PDGEMM)("T","N",&n,&n,&n,&Re_alpha,Ss,&ONE,&ONE,descS,Cs,&ONE,&ONE,descC,&Re_beta,rHs12,&ONE,&ONE,descH);

  /* S^t x rHs22 x S */

  for (i=0; i<na_rows*na_cols; i++) Cs[i] = 0.0;

  Cblacs_barrier(ictxt1,"A");
  F77_NAME(pdgemm,PDGEMM)("N","N",&n,&n,&n,&Re_alpha,rHs22,&ONE,&ONE,descH,Ss,&ONE,&ONE,descS,&Re_beta,Cs,&ONE,&ONE,descC);

  for (i=0; i<na_rows*na_cols; i++) rHs22[i] = 0.0;
  
  Cblacs_barrier(ictxt1,"C");
  F77_NAME(pdgemm,PDGEMM)("T","N",&n,&n,&n,&Re_alpha,Ss,&ONE,&ONE,descS,Cs,&ONE,&ONE,descC,&Re_beta,rHs22,&ONE,&ONE,descH);

  /* S^t x iHs11 x S */

  for (i=0; i<na_rows*na_cols; i++) Cs[i] = 0.0;

  Cblacs_barrier(ictxt1,"A");
  F77_NAME(pdgemm,PDGEMM)("N","N",&n,&n,&n,&Re_alpha,iHs11,&ONE,&ONE,descH,Ss,&ONE,&ONE,descS,&Re_beta,Cs,&ONE,&ONE,descC);

  for (i=0; i<na_rows*na_cols; i++) iHs11[i] = 0.0;

  Cblacs_barrier(ictxt1,"C");
  F77_NAME(pdgemm,PDGEMM)("T","N",&n,&n,&n,&Re_alpha,Ss,&ONE,&ONE,descS,Cs,&ONE,&ONE,descC,&Re_beta,iHs11,&ONE,&ONE,descH);

  /* S^t x iHs12 x S */

  for (i=0; i<na_rows*na_cols; i++) Cs[i] = 0.0;

  Cblacs_barrier(ictxt1,"A");
  F77_NAME(pdgemm,PDGEMM)("N","N",&n,&n,&n,&Re_alpha,iHs12,&ONE,&ONE,descH,Ss,&ONE,&ONE,descS,&Re_beta,Cs,&ONE,&ONE,descC);

  for (i=0; i<na_rows*na_cols; i++) iHs12[i] = 0.0;
  
  Cblacs_barrier(ictxt1,"C");
  F77_NAME(pdgemm,PDGEMM)("T","N",&n,&n,&n,&Re_alpha,Ss,&ONE,&ONE,descS,Cs,&ONE,&ONE,descC,&Re_beta,iHs12,&ONE,&ONE,descH);

  /* S^t x iHs22 x S */

  for (i=0; i<na_rows*na_cols; i++) Cs[i] = 0.0;

  Cblacs_barrier(ictxt1,"A");
  F77_NAME(pdgemm,PDGEMM)("N","N",&n,&n,&n,&Re_alpha,iHs22,&ONE,&ONE,descH,Ss,&ONE,&ONE,descS,&Re_beta,Cs,&ONE,&ONE,descC);

  for (i=0; i<na_rows*na_cols; i++) iHs22[i] = 0.0;
  
  Cblacs_barrier(ictxt1,"C");
  F77_NAME(pdgemm,PDGEMM)("T","N",&n,&n,&n,&Re_alpha,Ss,&ONE,&ONE,descS,Cs,&ONE,&ONE,descC,&Re_beta,iHs22,&ONE,&ONE,descH);

  if (measure_time){
    dtime(&etime);
    time2 += etime - stime;
  }

  /* ***************************************************
             diagonalize the transformed H
  *************************************************** */

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
    OpenMX_CuSolver_DenseZheevx_1based(Hs2,Cs2,ko,n2,MaxN,"Cluster_DFT_NonCol_DMmu Hamiltonian CuSOLVER");
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

  /* ***************************************************
      Transformation to the original eigenvectors.
      JRCAT NOTE 244P  C = U * lambda^{-1/2} * D
  *************************************************** */

  if (measure_time) dtime(&stime);

  for(k=0; k<na_rows2*na_cols2; k++){
    Hs2[k].r = 0.0;
    Hs2[k].i = 0.0;
  }

  Cblacs_barrier(ictxt1_2,"A");
  F77_NAME(pzgemm,PZGEMM)("T","T",&n2,&n2,&n2,&alpha,Cs2,&ONE,&ONE,descC2,Ss2,&ONE,&ONE,descS2,&beta,Hs2,&ONE,&ONE,descH2);

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
  }

  if (measure_time){
    dtime(&etime);
    time4 += etime - stime;
  }


  /****************************************************
                find chemical potential
  ****************************************************/

  if (measure_time) dtime(&stime);

  /* first, find ChemP at five times large temperatue */

  po = 0;
  loopN = 0;

  ChemP_MAX = 15.0;  
  ChemP_MIN =-15.0;

  do {
    ChemP = 0.50*(ChemP_MAX + ChemP_MIN);
    Num_State = 0.0;

    for (i1=1; i1<=MaxN; i1++){
      x = (ko[i1] - ChemP)*Beta*0.2;
      if (x<=-max_x) x = -max_x;
      if (max_x<=x)  x = max_x;
      FermiF = FermiFunc_NC(x,i1);
      Num_State = Num_State + FermiF;
      if (0.5<FermiF) Cluster_HOMO[0] = i1;
    }

    Dnum = (TZ - Num_State) - system_charge;
    if (0.0<=Dnum) ChemP_MIN = ChemP;
    else           ChemP_MAX = ChemP;
    if (fabs(Dnum)<1.0e-14) po = 1;
    loopN++;
  } 
  while (po==0 && loopN<1000); 

  /* second, find ChemP at the temperatue, starting from the previously found ChemP. */

  po = 0;
  loopN = 0;

  ChemP_MAX = 15.0;  
  ChemP_MIN =-15.0;

  do {

    if (loopN!=0){
      ChemP = 0.50*(ChemP_MAX + ChemP_MIN);
    }

    Num_State = 0.0;

    for (i1=1; i1<=MaxN; i1++){
      x = (ko[i1] - ChemP)*Beta;
      if (x<=-max_x) x = -max_x;
      if (max_x<=x)  x = max_x;
      FermiF = FermiFunc_NC(x,i1);
      Num_State = Num_State + FermiF;
      if (0.5<FermiF) Cluster_HOMO[0] = i1;
    }

    Dnum = (TZ - Num_State) - system_charge;
    if (0.0<=Dnum) ChemP_MIN = ChemP;
    else           ChemP_MAX = ChemP;
    if (fabs(Dnum)<1.0e-14) po = 1;
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
        density matrix and energy density matrix

        CDM[0]  Re alpha alpha density matrix
        CDM[1]  Re beta  beta  density matrix
        CDM[2]  Re alpha beta  density matrix
        CDM[3]  Im alpha beta  density matrix
        iDM[0][0]  Im alpha alpha density matrix
        iDM[0][1]  Im beta  beta  density matrix

    ****************************************************/

    if (measure_time) dtime(&stime);

    /* DM */

    time6 += Calc_DMmu_Cluster_non_collinear_ScaLAPACK( 1, myid, numprocs, size_H1, is2, ie2, MP, n, n2,
                                                      CDM, iCDM, ko, DM1, Work1, EVec1);

    time6 += Calc_DMmu_Cluster_non_collinear_ScaLAPACK( 2, myid, numprocs, size_H1, is2, ie2, MP, n, n2,
                                                      CDM, iCDM, ko, DM1, Work1, EVec1);

    time6 += Calc_DMmu_Cluster_non_collinear_ScaLAPACK( 3, myid, numprocs, size_H1, is2, ie2, MP, n, n2,
                                                      CDM, iCDM, ko, DM1, Work1, EVec1);

    time6 += Calc_DMmu_Cluster_non_collinear_ScaLAPACK( 4, myid, numprocs, size_H1, is2, ie2, MP, n, n2,
                                                      CDM, iCDM, ko, DM1, Work1, EVec1);

    time6 += Calc_DMmu_Cluster_non_collinear_ScaLAPACK( 5, myid, numprocs, size_H1, is2, ie2, MP, n, n2,
                                                      CDM, iCDM, ko, DM1, Work1, EVec1);

    time6 += Calc_DMmu_Cluster_non_collinear_ScaLAPACK( 6, myid, numprocs, size_H1, is2, ie2, MP, n, n2,
                                                      CDM, iCDM, ko, DM1, Work1, EVec1);

    if (measure_time){
      dtime(&etime);
      time7 += etime - stime;
    }



  /****************************************************
                          Free
  ****************************************************/

  free(is1);
  free(ie1);
  free(Num_Snd_EV);
  free(Num_Rcv_EV);

  free(index_Snd_i);
  free(index_Snd_j);
  free(EVec_Snd);
  free(index_Rcv_i);
  free(index_Rcv_j);
  free(EVec_Rcv);

  /* for elapsed time */

  MPI_Barrier(mpi_comm_level1);
  dtime(&TEtime);
  time0 = TEtime - TStime;
  return time0;
}





double Calc_DMmu_Cluster_non_collinear_ScaLAPACK(
    int calc_flag,
    int myid,
    int numprocs,
    int size_H1,
    int *is2,
    int *ie2,
    int *MP,
    int n,
    int n2,
    double *****CDM,
    double *****iDM0,
    double *ko,
    double *DM1,
    double *Work1,
    dcomplex *EVec1 )
{
  int i,j,k,po,p,GA_AN,MA_AN,wanA,tnoA,Anum;
  int LB_AN,GB_AN,wanB,tnoB,Bnum,i1,j1,ID;
  double max_x=60.0,dum;
  double FermiF,FermiF2,x,x2,diffF;
  double FermiF_lower, FermiF_upper, x_lower, x_upper;
  double FermiEps = 1.0e-13;
  double stime,etime,time,lumos;
  MPI_Status stat;
  MPI_Request request;

  dtime(&stime);

  /* initialize DM1 */

  for (i=0; i<size_H1; i++){
    DM1[i] = 0.0;
  }

  if (xanes_calc==1){ 
    if (myid==Host_ID) {
      printf("Cannot use XANES for Cluster_DFT_NonCol_DMmu.\n");
      MPI_Finalize();
      exit(0);
    }
  }

  /* calculation of DM1 */ 

  for (k=is2[myid]; k<=ie2[myid]; k++){

    if (flag_export_DM<2){
      x = (ko[k] - ChemP)*Beta;
      if (x<=-max_x) x = -max_x;
      if (max_x<=x)  x = max_x;
      FermiF = FermiFunc_NC(x,k);
    }

    /* Cutoff in the specified energy range */
    if ((flag_export_DM==1) && flag_energy_range_DM==1){
      if (xanes_calc==1){ 
        //if ( ((ko[k] - ChemP_XANES[spin]) < DM_energy_range[0]) || (DM_energy_range[1] < (ko[k] - ChemP_XANES[spin])) ){
        //  FermiF = 0;
        //}
      }
      else{
        if ( (DM_energy_range[0] < (ko[k] - ChemP)) && ((ko[k] - ChemP) < DM_energy_range[1]) ){
          FermiF = 0;
        }
      }
    }
    else if (flag_export_DM==2){ /* T-method */
      if (xanes_calc==1) {
        //x_lower = (ko[k] - (ChemP_XANES[spin] + DM_energy_range[0]))*DM_tilde_Beta_lower;
        //x_upper = (ko[k] - (ChemP_XANES[spin] + DM_energy_range[1]))*DM_tilde_Beta_upper;
      }
      else {
        x_lower = (ko[k] - (ChemP + DM_energy_range[0]))*DM_tilde_Beta_lower;
        x_upper = (ko[k] - (ChemP + DM_energy_range[1]))*DM_tilde_Beta_upper;
      }

      if (x_lower<=-max_x) x_lower = -max_x;
      if (x_upper<=-max_x) x_upper = -max_x;
      if (max_x<=x_lower)  x_lower = max_x;
      if (max_x<=x_upper)  x_upper = max_x;
      FermiF_upper = FermiFunc_NC(x_upper,k);
      FermiF_lower = FermiFunc_NC(x_lower,k);

      FermiF = FermiF_upper - FermiF_lower;
    }
    /* */

    if ( FermiEps<FermiF ) {

      p = 0;
      for (GA_AN=1; GA_AN<=atomnum; GA_AN++){

        wanA = WhatSpecies[GA_AN];
        tnoA = Spe_Total_CNO[wanA];
        Anum = MP[GA_AN];
        for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
          GB_AN = natn[GA_AN][LB_AN];
          wanB = WhatSpecies[GB_AN];
          tnoB = Spe_Total_CNO[wanB];
          Bnum = MP[GB_AN];
          for (i=0; i<tnoA; i++){
            for (j=0; j<tnoB; j++){

              i1 = (k-is2[myid])*n2 + Anum + i - 1; 
              j1 = (k-is2[myid])*n2 + Bnum + j - 1;

              switch (calc_flag){

                /* Re11 */
              case 1:
                DM1[p] += FermiF*(EVec1[i1].r*EVec1[j1].r + EVec1[i1].i*EVec1[j1].i);
                break;

                /* Re22 */
              case 2:
                DM1[p] += FermiF*(EVec1[i1+n].r*EVec1[j1+n].r + EVec1[i1+n].i*EVec1[j1+n].i);
                break;

                /* Re12 */
              case 3:
                DM1[p] += FermiF*(EVec1[i1].r*EVec1[j1+n].r + EVec1[i1].i*EVec1[j1+n].i);
                break;

                /* Im12 */
              case 4:
                DM1[p] += FermiF*(EVec1[i1].r*EVec1[j1+n].i - EVec1[i1].i*EVec1[j1+n].r);
                break;

                /* Im11 */
              case 5:
                DM1[p] += FermiF*(EVec1[i1].r*EVec1[j1].i - EVec1[i1].i*EVec1[j1].r);
                break;

                /* Im22 */
              case 6:
                DM1[p] += FermiF*(EVec1[i1+n].r*EVec1[j1+n].i - EVec1[i1+n].i*EVec1[j1+n].r);
                break;

              }

              /* increment of p */
              p++;  

            }
          }
        }
      } /* GA_AN */
    }
  }

  /* MPI_Allreduce */

  MPI_Allreduce(DM1, Work1, size_H1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
  for (i=0; i<size_H1; i++) DM1[i] = Work1[i];

  /* store DM1 to a proper place */

  p = 0;
  for (GA_AN=1; GA_AN<=atomnum; GA_AN++){

    MA_AN = F_G2M[GA_AN];
    wanA = WhatSpecies[GA_AN];
    tnoA = Spe_Total_CNO[wanA];
    Anum = MP[GA_AN];
    ID = G2ID[GA_AN];

    for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
      GB_AN = natn[GA_AN][LB_AN];
      wanB = WhatSpecies[GB_AN];
      tnoB = Spe_Total_CNO[wanB];
      Bnum = MP[GB_AN];

      if (myid==ID){
         
        for (i=0; i<tnoA; i++){
          for (j=0; j<tnoB; j++){

            switch (calc_flag){

              /* Re11 */ 
              case 1:
                CDM[0][MA_AN][LB_AN][i][j] = DM1[p];
                    break;

              /* Re22 */ 
              case 2:
                CDM[1][MA_AN][LB_AN][i][j] = DM1[p];
                    break;

              /* Re12 */ 
              case 3:
                CDM[2][MA_AN][LB_AN][i][j] = DM1[p];
                    break;

              /* Im12 */ 
              case 4:
                CDM[3][MA_AN][LB_AN][i][j] = DM1[p];
                    break;

              /* Im11 */ 
              case 5:
                iDM0[0][MA_AN][LB_AN][i][j] = DM1[p];
                    break;

              /* Im22 */ 
              case 6:
                iDM0[1][MA_AN][LB_AN][i][j] = DM1[p];
                    break;

            }

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

    } /* LB_AN */
  } /* GA_AN */

  dtime(&etime);
  return (etime-stime);
}



/**********************************************************************************
  Cluster_DFT_Col_DMmu.c:

     Cluster_DFT_Col_DMmu.c is a subroutine to perform cluster collinear calculations.

  Log of Cluster_DFT_Col_DMmu.c:

     21/Feb./2019  Released by T. Ozaki
     03/Jul./2023  Modified by M. Fukuda

**********************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "openmx_common.h"
#include "openmx_gpusolver_dense_utils.h"
#include "lapack_prototypes.h"
#include "mpi.h"
#include "set_hip_default_device_from_local_rank.h"
#include "set_openmp_device_from_local_rank.h"
#include <omp.h>
#include <omp.h>
#include "flpq_dm.h"
#include "f77func.h"

#define  measure_time   0

/* GPU GpuSolver context for Cluster_DFT_Col_DMmu (added by H.Kawai, Phase D) */
typedef struct {
    int                device_id;
    hipblasHandle_t     hipblas;
    hipsolverDnHandle_t gpusolver;
} ClusterColDMmuGpuSolverCtx;

static ClusterColDMmuGpuSolverCtx ClusterColDMmu_gpusolver_ctx = {0};

static void ClusterColDMmu_GpuSolver_Destroy(void)
{
    ClusterColDMmuGpuSolverCtx *ctx = &ClusterColDMmu_gpusolver_ctx;
    if (ctx->gpusolver != NULL) wait_hipfunc(hipsolverDnDestroy(ctx->gpusolver));
    if (ctx->hipblas != NULL)   wait_hipfunc(hipblasDestroy(ctx->hipblas));
    memset(ctx, 0, sizeof(*ctx));
    ctx->device_id = -1;
}

static void ClusterColDMmu_GpuSolver_Init(void)
{
    ClusterColDMmuGpuSolverCtx *ctx = &ClusterColDMmu_gpusolver_ctx;
    int current_device;
    wait_hipfunc(hipGetDevice(&current_device));
    if (ctx->device_id == current_device && ctx->hipblas != NULL) return;
    if (ctx->hipblas != NULL) ClusterColDMmu_GpuSolver_Destroy();
    ctx->device_id = current_device;
    wait_hipfunc(hipblasCreate(&ctx->hipblas));
    wait_hipfunc(hipsolverDnCreate(&ctx->gpusolver));
}

static void ClusterColDMmu_GEMMul8Dgemm_OpenMP(hipblasOperation_t transa, hipblasOperation_t transb, int m, int n, int k,
                                                double const * A, double const * B, double * C)
{
    ClusterColDMmu_GpuSolver_Init();
    {
        double const alpha = 1.0;
        double const beta  = 0.0;
        wait_hipfunc(openmx_gemmul8Dgemm(ClusterColDMmu_gpusolver_ctx.hipblas, transa, transb, m, n, k, &alpha, A, m, B, k, &beta, C, m));
    }
}

void solve_evp_real_( int *n1, int *n2, double *Cs, int *na_rows1, double *a, double *Ss, int *na_rows2, int *nblk, 
                      int *mpi_comm_rows_int, int *mpi_comm_cols_int);

void elpa_solve_evp_real_2stage_double_impl_( int *n1, int *n2, double *Cs, int *na_rows1, double *a, double *Ss, int *na_rows2, 
                                              int *nblk, int *na_cols1, int *mpi_comm_rows_int, int *mpi_comm_cols_int, int *mpiworld);

static double Calc_DMmu_Cluster_collinear(int myid0,
                                        int numprocs0,
                                        int myid1,
                                        int numprocs1,
                                        int myworld1,
                                        int size_H1,
                                        int *is2,
                                        int *ie2,
                                        int *MP,
                                        int n,
                                        MPI_Comm *MPI_CommWD1,
                                        int *Comm_World_StartID1,
                                        double *****CDM,
                                        double **ko,
                                        double *DM1,
                                        double *Work1,
                                        double **EVec1, 
                                        int *SP_NZeros,
                                        int *SP_Atoms );


double Cluster_DFT_Col_DMmu(
                   char *mode,
                   int SCF_iter,
                   int SpinP_switch,
                   double **ko,
                   double *****nh, 
                   double ****CntOLP,
                   double *****CDM,
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
  int MA_AN,LB_AN,GB_AN,Bnum,MaxN;
  int wan1,mul,m,bcast_flag;
  int *is1,*ie1;
  double time0,lumos,av_num;
  double *OneD_Mat1;
  double ***H;
  double TZ,my_sum,sum,sumE,max_x=60.0;
  double sum0,sum1,sum2,sum3;
  double tmp1,tmp2;
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

  /* for time */
  MPI_Barrier(mpi_comm_level1);
  dtime(&TStime);

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs0);
  MPI_Comm_rank(mpi_comm_level1,&myid0);

  MPI_Comm_size(MPI_CommWD1[myworld1],&numprocs1);
  MPI_Comm_rank(MPI_CommWD1[myworld1],&myid1);

  /****************************************************
             calculation of the array size
  ****************************************************/

  n = 0;
  for (i=1; i<=atomnum; i++){
    wanA  = WhatSpecies[i];
    n = n + Spe_Total_CNO[wanA];
  }
  n2 = n + 2;

  /* GPU dispatch (added by H.Kawai): assign CUDA/OpenMP device when GPUSOLVER is requested */
  if (scf_eigen_lib_flag == GPUSOLVER && n >= GPU_CPU_SWITCH_NUM) {
      set_hip_default_device_from_local_rank();
      set_openmp_nvidia_device_from_local_rank();
  }

  /****************************************************
   Allocation

   double  H[List_YOUSO[23]][n2][n2]  
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

    else if (scf_eigen_lib_flag==GPUSOLVER && GPU_CPU_SWITCH_NUM<=n && na_rows==n && na_cols==n){
      OpenMX_GpuSolver_DenseDsyevx_1based(Cs,Ss,ko[0],n,n,"Cluster_DFT_Col_DMmu overlap GPU solver");
    }

    else if (scf_eigen_lib_flag==2 || scf_eigen_lib_flag==GPUSOLVER){

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

  /****************************************************
    calculations of eigenvalues for up and down spins

     Note:
         MP indicates the starting position of
              atom i in arraies H and S
  ****************************************************/

  MPI_Barrier(mpi_comm_level1);

  /* find the maximum states in solved eigenvalues */

  if (SCF_iter==1){
    MaxN = n;
  }
  else{

    if      ( strcasecmp(mode,"scf")==0 ) 
      lumos = (double)n*0.100;
    else if ( strcasecmp(mode,"dos")==0 )
      lumos = (double)n*0.200;      
    else if ( strcasecmp(mode,"lcaoout")==0 )
      lumos = (double)n*0.200;      
    else if ( strcasecmp(mode,"xanes")==0 )
      lumos = (double)n*0.200;      
    else if ( strcasecmp(mode,"diag")==0 )
      lumos = (double)n*0.200;      

    if (lumos<400.0) lumos = 400.0;
    MaxN = (TZ-system_charge)/2 + (int)lumos;
    if (n<MaxN) MaxN = n;
    
    if (cal_partial_charge) MaxN = n; 
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
  else if (scf_eigen_lib_flag==GPUSOLVER && GPU_CPU_SWITCH_NUM<=n && na_rows==n && na_cols==n){
    OpenMX_GpuSolver_DenseDsyevx_1based(Hs,Cs,ko[spin],n,MaxN,"Cluster_DFT_Col_DMmu Hamiltonian GPU solver");
  }

  else if (scf_eigen_lib_flag==2 || scf_eigen_lib_flag==GPUSOLVER){

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

  for(i=0;i<na_rows*na_cols;i++){
    Hs[i] = 0.0;
  }

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

  /*********************************************** 
    MPI: ko
  ***********************************************/

  if (measure_time) dtime(&stime);

  for (sp=0; sp<=SpinP_switch; sp++){
    MPI_Bcast(&ko[sp][1],MaxN,MPI_DOUBLE,Comm_World_StartID1[sp],mpi_comm_level1);
  }


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

  /* for XANES */
  if (xanes_calc==1){

    /****************************************************
            searching of chemical potential
    ****************************************************/

    for (spin=0; spin<=SpinP_switch; spin++){

      po = 0;
      loopN = 0;

      ChemP_MAX = 30.0;  
      ChemP_MIN =-30.0;

      do {

        ChemP = 0.50*(ChemP_MAX + ChemP_MIN);
        Num_State = 0.0;

        for (i1=1; i1<=MaxN; i1++){
          x = (ko[spin][i1] - ChemP)*Beta;
          if (x<=-max_x) x = -max_x;
          if (max_x<=x)  x = max_x;
          FermiF = 1.0/(1.0 + exp(x));
          Num_State += FermiF;
        }

        Dnum = HOMO_XANES[spin] - Num_State;
        if (0.0<=Dnum) ChemP_MIN = ChemP;
        else           ChemP_MAX = ChemP;
        if (fabs(Dnum)<1.0e-14) po = 1;

        if (myid1==Host_ID && 2<=level_stdout){
          printf("spin=%2d ChemP=%15.12f HOMO_XANES=%2d Num_state=%15.12f\n",spin,ChemP,HOMO_XANES[spin],Num_State); 
        }

        loopN++;

      } while (po==0 && loopN<1000); 

      ChemP_XANES[spin] = ChemP;
      Cluster_HOMO[spin] = HOMO_XANES[spin];

    } /* spin */

    /* set ChemP */

    ChemP = 0.5*(ChemP_XANES[0] + ChemP_XANES[1]); 

  } /* end of if (xanes_calc==1) */

  /* start of else for if (xanes_calc==1) */

  else{

    /****************************************************
            searching of chemical potential
    ****************************************************/

    /* first, find ChemP at five times large temperatue */

    po = 0;
    loopN = 0;

    ChemP_MAX = 30.0;  
    ChemP_MIN =-30.0;
  
    do {

      ChemP = 0.50*(ChemP_MAX + ChemP_MIN);
      Num_State = 0.0;

      for (spin=0; spin<=SpinP_switch; spin++){
        for (i1=1; i1<=MaxN; i1++){
          x = (ko[spin][i1] - ChemP)*Beta*0.2;
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
      if (fabs(Dnum)<1.0e-14) po = 1;

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
      if (fabs(Dnum)<1.0e-14) po = 1;

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

  } /* end of else for if (xanes_calc==1) */



    /****************************************************
          density matrix and energy density matrix
                  for up and down spins
    ****************************************************/

    time6 += Calc_DMmu_Cluster_collinear( myid0,numprocs0,myid1,numprocs1,myworld1,
                                        size_H1,is2,ie2,MP,n,MPI_CommWD1,Comm_World_StartID1,
                                        CDM,ko,CDM1,Work1,EVec1,SP_NZeros,SP_Atoms);

    if (measure_time){
      dtime(&etime);
      time6 += etime - stime;
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






double Calc_DMmu_Cluster_collinear(
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
    MPI_Comm *MPI_CommWD1,
    int *Comm_World_StartID1,
    double *****CDM,
    double **ko,
    double *DM1,
    double *Work1,
    double **EVec1, 
    int *SP_NZeros,
    int *SP_Atoms )
{
  int spin,i,j,i0,j0,k,kmin,kmax,po,p,GA_AN,MA_AN,wanA,tnoA,Anum;
  int LB_AN,GB_AN,wanB,tnoB,Bnum,i1,j1,ID;
  double max_x=60.0,dum;
  double FermiF,FermiF2,x,x2,diffF,sum1,sum2;
  double FermiF_lower, FermiF_upper, x_lower, x_upper;
  double FermiEps = 1.0e-13;
  double stime,etime,time,lumos;
  MPI_Status stat;
  MPI_Request request;
  double *FF;

  dtime(&stime);

  /* allocation of arrays */

  FF = (double*)malloc(sizeof(double)*(n+1));

  /* spin=myworld1 */

  spin = myworld1;

 calc_dm_collinear:

  /* initialize DM1 */

  for (i=0; i<size_H1; i++){
    DM1[i] = 0.0;
  }

  /* pre-calculation of Fermi Function */ 

  po = 0;
  kmin = is2[myid1];
  kmax = ie2[myid1];
  
  for (k=is2[myid1]; k<=ie2[myid1]; k++){

    if (flag_export_DM<2){
      if (xanes_calc==1) 
        x = (ko[spin][k] - ChemP_XANES[spin])*Beta;
      else 
        x = (ko[spin][k] - ChemP)*Beta;

      if (x<=-max_x) x = -max_x;
      if (max_x<=x)  x = max_x;
      FermiF = FermiFunc(x,spin,k,&po,&x);

      FF[k] = FermiF;
    }


    /* Cutoff in the specified energy range */
    if ((flag_export_DM==1) && (flag_energy_range_DM==1) ){
      if (xanes_calc==1){ 
        if ( ((ko[spin][k] - ChemP_XANES[spin]) < DM_energy_range[0]) || (DM_energy_range[1] < (ko[spin][k] - ChemP_XANES[spin])) ){
          FF[k] = 0;
        }
      }
      else{
        if ( (DM_energy_range[0] < (ko[spin][k] - ChemP)) && ((ko[spin][k] - ChemP) < DM_energy_range[1]) ){
          FF[k] = 0;
        }
      }
    }
    else if (flag_export_DM==2){ /* T-method */
      if (xanes_calc==1) {
        x_lower = (ko[spin][k] - (ChemP_XANES[spin] + DM_energy_range[0]))*DM_tilde_Beta_lower;
        x_upper = (ko[spin][k] - (ChemP_XANES[spin] + DM_energy_range[1]))*DM_tilde_Beta_upper;
      }
      else {
        x_lower = (ko[spin][k] - (ChemP + DM_energy_range[0]))*DM_tilde_Beta_lower;
        x_upper = (ko[spin][k] - (ChemP + DM_energy_range[1]))*DM_tilde_Beta_upper;
      }

      if (x_lower<=-max_x) x_lower = -max_x;
      if (x_upper<=-max_x) x_upper = -max_x;
      if (max_x<=x_lower)  x_lower = max_x;
      if (max_x<=x_upper)  x_upper = max_x;
      FermiF_lower = FermiFunc(x_lower,spin,k,&po,&x_lower);
      FermiF_upper = FermiFunc(x_upper,spin,k,&po,&x_upper);

      FF[k] = FermiF_upper - FermiF_lower;
    }
    /* */

  }

  /* calculation of DM1 */ 

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

          i0 = (Anum + i - 1)*(ie2[myid1]-is2[myid1]+1) - is2[myid1];
          j0 = (Bnum + j - 1)*(ie2[myid1]-is2[myid1]+1) - is2[myid1];

          sum1 = 0.0;

          for (k=kmin; k<=kmax; k++){
            dum = FF[k]*EVec1[spin][i0+k]*EVec1[spin][j0+k];
            sum1 += dum;
          }

          DM1[p]  = sum1;

          /* increment of p */
          p++;  

        }
      }
    }
  } /* GA_AN */

  /* MPI_Allreduce */

  MPI_Allreduce(DM1, Work1, size_H1, MPI_DOUBLE, MPI_SUM, MPI_CommWD1[myworld1]);
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

      if (myid0==ID){
         
        for (i=0; i<tnoA; i++){
          for (j=0; j<tnoB; j++){

            CDM[spin][MA_AN][LB_AN][i][j] = DM1[p];

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

    /* store DM1 to a proper place */

    if      (myworld1==0) spin = 1;
    else if (myworld1==1) spin = 0;

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

        if (myid0==ID){
         
          for (i=0; i<tnoA; i++){
            for (j=0; j<tnoB; j++){

              CDM[spin][MA_AN][LB_AN][i][j] = DM1[p];

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
  } /* end of else if (SpinP_switch==1) */

  /* freeing of arrays */

  free(FF);

  dtime(&etime);
  return (etime-stime);

}


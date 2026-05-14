/**********************************************************************
  Band_DFT_NonCol_CWF.c:

     Band_DFT_NonCol_CWF.c is a subroutine to calculate closest Wannier 
     functions to a given set of orbials in the band calculation 
     based on the non-collinear DFT.

  Log of Band_DFT_NonCol_CWF.c:

     09/June/2023  Released by T. Ozaki

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <time.h>
#include <sys/stat.h>
#include <sys/types.h>
#include "mpi.h"
#include "openmx_common.h"
#include "openmx_cusolver_dense_utils.h"
#include "lapack_prototypes.h"
#include "tran_variables.h"
#include "set_cuda_default_device_from_local_rank.h"
#include "set_openacc_device_from_local_rank.h"
#include <omp.h>
#include <openacc.h>

#define  measure_time  0

/* GPU CuSolver context for Band_DFT_NonCol_CWF (added by H.Kawai, Phase D) */
typedef struct {
    int                device_id;
    cublasHandle_t     cublas;
    cusolverDnHandle_t cusolver;
} BandNonColCWFCuSolverCtx;

static BandNonColCWFCuSolverCtx BandNonColCWF_cusolver_ctx = {0};

static void BandNonColCWF_CuSolver_Destroy(void)
{
    BandNonColCWFCuSolverCtx *ctx = &BandNonColCWF_cusolver_ctx;
    if (ctx->cusolver != NULL) wait_cudafunc(cusolverDnDestroy(ctx->cusolver));
    if (ctx->cublas != NULL)   wait_cudafunc(cublasDestroy(ctx->cublas));
    memset(ctx, 0, sizeof(*ctx));
    ctx->device_id = -1;
}

static void BandNonColCWF_CuSolver_Init(void)
{
    BandNonColCWFCuSolverCtx *ctx = &BandNonColCWF_cusolver_ctx;
    int current_device;
    wait_cudafunc(cudaGetDevice(&current_device));
    if (ctx->device_id == current_device && ctx->cublas != NULL) return;
    if (ctx->cublas != NULL) BandNonColCWF_CuSolver_Destroy();
    ctx->device_id = current_device;
    wait_cudafunc(cublasCreate(&ctx->cublas));
    wait_cudafunc(cusolverDnCreate(&ctx->cusolver));
}

static void BandNonColCWF_GEMMul8Zgemm_OpenACC(cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
                                               dcomplex const * A, dcomplex const * B, dcomplex * C)
{
    BandNonColCWF_CuSolver_Init();
#pragma acc data      present(A[0 : m * k], B[0 : k * n], C[0 : m * n])
#pragma acc host_data use_device(A, B, C)
    {
        cuDoubleComplex const alpha = make_cuDoubleComplex(1.0, 0.0);
        cuDoubleComplex const beta  = make_cuDoubleComplex(0.0, 0.0);
        wait_cudafunc(openmx_gemmul8Zgemm(BandNonColCWF_cusolver_ctx.cublas, transa, transb, m, n, k, &alpha,
                                          (cuDoubleComplex const *)A, m, (cuDoubleComplex const *)B, k, &beta,
                                          (cuDoubleComplex *)C, m));
    }
}

double **ReEVec_i1,**ImEVec_i1,**ReEVec_i2,**ImEVec_i2;
double **ReEVec_j1,**ImEVec_j1,**ReEVec_j2,**ImEVec_j2;
double complex ***CWF_Guiding_MOs_NC;
double complex ******CWF_Coef_NC;
double DM_func;

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

void solve_evp_complex_( int *n2, int *MaxN, dcomplex *Hs2, int *na_rows2_1, double *a, dcomplex *Cs2, int *na_rows2_2, 
                         int *nblk2, int *mpi_comm_rows_int, int *mpi_comm_cols_int );

void elpa_solve_evp_complex_2stage_double_impl_( int *n2, int *MaxN, dcomplex *Hs2, int *na_rows2_1, double *a, dcomplex *Cs2, 
                                                 int *na_rows2_2, int *nblk2, int *na_cols2, 
                                                 int *mpi_comm_rows_int, int *mpi_comm_cols_int, int *mpiworld );


static void Construct_Band_Ms( int cpx_flag, double ****Mat, double *M1, dcomplex *Ms, 
                               int *MP, double k1, double k2, double k3);


void AllocateArrays_NonCol_QAO2();
void FreeArrays_NonCol_QAO2();
double Calc_Hybrid_AO_NonCol( double ****OLP0, double *****Hks, double *****CDM );
double Calc_MO_in_Bulk_NonCol( double *****CDM, double *****iCDM );

static void Band_Dispersion_NonCol_CWF( int nkpath, int *n_perk, double ***kpath, char ***kname, 
					int TNum_CWFs, double complex ****TB_Hopping, 
					int knum_i, int knum_j, int knum_k, MPI_Comm mpi_comm_level5 );

static double Calc_CWF_Band_NonCol(
				   char *mode,
				   int own_calc_flag,
				   int kloop,
				   int knum_i, int knum_j, int knum_k,
				   double *T_KGrids1,
				   double *T_KGrids2,
				   double *T_KGrids3,
				   double sum_weights,
				   double weight_kpoint,
				   int myid0,
				   int numprocs0,
				   int myid1,
				   int numprocs1,
				   int myid2,
				   int numprocs2,
				   int myworld1,
				   int myworld2,
				   int *is2,
				   int *ie2,
				   int *MP,
				   int n2,
				   int MinN,
				   int MaxN,
				   int TNum_CWFs,
				   MPI_Comm *MPI_CommWD1,
				   int *Comm_World_StartID1,
				   MPI_Comm *MPI_CommWD2,
				   int *Comm_World_StartID2,
				   double ****OLP0,
				   double ***EIGEN,
				   dcomplex **EVec1,
				   dcomplex *Full_EVec,
				   int nkpath, int *n_perk, double ***kpath, char ***kname,
				   double *S1,
				   int *order_GA );


static void Allocate_Free_Band_NonCol_CWF( int todo_flag, 
					   int myworld2, 
					   MPI_Comm *MPI_CommWD2,
					   int n2,
					   int MaxN,
					   int TNum_CWFs,
					   int knum_i, int knum_j, int knum_k,
					   dcomplex **Cs,
					   dcomplex **Hs, 
					   dcomplex **Vs,
					   dcomplex **Ws,
					   dcomplex **EVs_PAO,
					   dcomplex **WFs,
					   double complex *****Hop );


#pragma GCC optimize("O2")
double Band_DFT_NonCol_CWF(
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
		    dcomplex **EVec1,
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
		    MPI_Comm *MPI_CommWD2,
                    int nkpath, int *n_perk, 
                    double ***kpath, char ***kname)
{
  static int firsttime=1;
  int i,j,k,l,m,n,n2,p,wan,MaxN,MinN,i0,ks;
  int i1,i1s,j1,ia,jb,lmax,po,po1,spin;
  int num2,RnB,l1,l2,l3,loop_num,ns,ne;
  int ct_AN,h_AN,wanA,tnoA,wanB,tnoB;
  int MA_AN,GA_AN,Anum,num_kloop0,max_num_kloop0;
  int T_knum,S_knum,E_knum,kloop,kloop0;
  double av_num,lumos;
  double time0;
  int LB_AN,GB_AN,Bnum;
  double k1,k2,k3,Fkw;
  double sum,sumi,sum_weights;
  double Num_State;
  double My_Num_State;
  double FermiF;
  double tmp,tmp1,eig,kw,EV_cut0;
  double e,b0,b1,e0,e1,weight;
  double x,Dnum,Dnum2,AcP,ChemP_MAX,ChemP_MIN;
  int *is1,*ie1;
  int *is2,*ie2;
  int *My_NZeros;
  int *SP_NZeros;
  int *SP_Atoms;
  int size_H1;

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
  double x_cut=60.0,in0,in1,out0,out1;
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

  int Max_Num_Snd_EV,Max_Num_Rcv_EV;
  int *Num_Snd_EV,*Num_Rcv_EV;
  int *index_Snd_i,*index_Snd_j,*index_Rcv_i,*index_Rcv_j;
  double *EVec_Snd,*EVec_Rcv;
  dcomplex *Full_EVec;

  /* initialize DM_func, CWF_Charge, and CWF_Energy */
 
  DM_func = 0.0;

  for (spin=0; spin<1; spin++){
    for (i=0; i<TNum_CWFs; i++){
      CWF_Charge[spin][i] = 0.0;
      CWF_Energy[spin][i] = 0.0;
    }
  }

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

  /* show the message */

  if (myid0==0){
    printf("\n<Calculation of Closest Wannier Functions>\n");   
  }

  /*********************************************** 
       for pallalel calculations in myworld1
  ***********************************************/

  MPI_Comm_size(MPI_CommWD1[myworld1],&numprocs1);
  MPI_Comm_rank(MPI_CommWD1[myworld1],&myid1);

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
         find the number of basis functions, n
  ****************************************************/

  n = 0;
  for (i=1; i<=atomnum; i++){
    wanA  = WhatSpecies[i];
    n += Spe_Total_CNO[wanA];
  }
  n2 = n*2;

  /* GPU dispatch (added by H.Kawai): assign CUDA/OpenACC device when CuSOLVER is requested */
  if (scf_eigen_lib_flag == CuSOLVER && n2 >= GPU_CPU_SWITCH_NUM) {
      set_cuda_default_device_from_local_rank();
      set_openacc_nvidia_device_from_local_rank();
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
      e = EIGEN[0][0][i] - ChemP;
      out1 = CWF_disentangling_Erange[3];
      if (e<out1) break;
    }
    MaxN = i; 

    /* find MinN */
    for (i=1; i<=n2; i++){
      e = EIGEN[0][0][i] - ChemP;
      out0 = CWF_disentangling_Erange[0];
      if (out0<e) break;
    }
    MinN = i;

    TNum_PAOs = n;
    TNum_PAOs2 = n2;
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
    TNum_PAOs2 = n2;
    MinN = 1;
    MaxN = n2;
    Num_KS_states_CWF = MaxN;
  }

  /***********************************************
     allocation of arrays
  ***********************************************/

  My_NZeros = (int*)malloc(sizeof(int)*numprocs0);
  SP_NZeros = (int*)malloc(sizeof(int)*numprocs0);
  SP_Atoms = (int*)malloc(sizeof(int)*numprocs0);

  ReEVec_i1 = (double**)malloc(sizeof(double*)*List_YOUSO[7]); 
  for (i=0; i<List_YOUSO[7]; i++){
    ReEVec_i1[i] = (double*)malloc(sizeof(double)*(MaxN+1));
  }

  ImEVec_i1 = (double**)malloc(sizeof(double*)*List_YOUSO[7]); 
  for (i=0; i<List_YOUSO[7]; i++){
    ImEVec_i1[i] = (double*)malloc(sizeof(double)*(MaxN+1));
  }

  ReEVec_i2 = (double**)malloc(sizeof(double*)*List_YOUSO[7]); 
  for (i=0; i<List_YOUSO[7]; i++){
    ReEVec_i2[i] = (double*)malloc(sizeof(double)*(MaxN+1));
  }

  ImEVec_i2 = (double**)malloc(sizeof(double*)*List_YOUSO[7]); 
  for (i=0; i<List_YOUSO[7]; i++){
    ImEVec_i2[i] = (double*)malloc(sizeof(double)*(MaxN+1));
  }

  ReEVec_j1 = (double**)malloc(sizeof(double*)*List_YOUSO[7]); 
  for (i=0; i<List_YOUSO[7]; i++){
    ReEVec_j1[i] = (double*)malloc(sizeof(double)*(MaxN+1));
  }

  ImEVec_j1 = (double**)malloc(sizeof(double*)*List_YOUSO[7]); 
  for (i=0; i<List_YOUSO[7]; i++){
    ImEVec_j1[i] = (double*)malloc(sizeof(double)*(MaxN+1));
  }

  ReEVec_j2 = (double**)malloc(sizeof(double*)*List_YOUSO[7]); 
  for (i=0; i<List_YOUSO[7]; i++){
    ReEVec_j2[i] = (double*)malloc(sizeof(double)*(MaxN+1));
  }

  ImEVec_j2 = (double**)malloc(sizeof(double*)*List_YOUSO[7]); 
  for (i=0; i<List_YOUSO[7]; i++){
    ImEVec_j2[i] = (double*)malloc(sizeof(double)*(MaxN+1));
  }

  if (way_of_kpoint==1){

    T_knum = 0;
    for (i=0; i<knum_i; i++){

      if (knum_i==1)  k1 = 0.0;
      else            k1 = -0.5 + (2.0*(double)i+1.0)/(2.0*(double)knum_i);

      for (j=0; j<knum_j; j++){

	if (knum_j==1)  k2 = 0.0;
	else            k2 = -0.5 + (2.0*(double)j+1.0)/(2.0*(double)knum_j);

	for (k=0; k<knum_k; k++){

	  if (knum_k==1)  k3 = 0.0;
	  else            k3 = -0.5 + (2.0*(double)k+1.0)/(2.0*(double)knum_k);

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
	else            k1 = -0.5 + (2.0*(double)i+1.0)/(2.0*(double)knum_i);
	printf("%9.5f ",k1);fflush(stdout);
      }
      printf("\n");fflush(stdout);

      printf(" KGrids2: ");fflush(stdout);

      for (i=0;i<=knum_j-1;i++){
	if (knum_j==1)  k2 = 0.0;
	else            k2 = -0.5 + (2.0*(double)i+1.0)/(2.0*(double)knum_j);
	printf("%9.5f ",k2);fflush(stdout);
      }
      printf("\n");fflush(stdout);

      printf(" KGrids3: ");fflush(stdout);
      for (i=0;i<=knum_k-1;i++){
	if (knum_k==1)  k3 = 0.0;
	else            k3 = -0.5 + (2.0*(double)i+1.0)/(2.0*(double)knum_k);
	printf("%9.5f ",k3);fflush(stdout);
      }
      printf("\n");fflush(stdout);
    }
  } // end of if (way_of_kpoint==1)

  /***********************************************
         k-points by a Gamma-centered mesh
  ***********************************************/
  
  else if (way_of_kpoint==3){
    
    T_knum = 0;
    for (i=0; i<knum_i; i++){

      if (knum_i==1)  k1 = 0.0;
      else            k1 = ((double)i)/((double)knum_i);

      for (j=0; j<knum_j; j++){

        if (knum_j==1)  k2 = 0.0;
        else            k2 = ((double)j)/((double)knum_j);

        for (k=0; k<knum_k; k++){

          if (knum_k==1)  k3 = 0.0;
          else            k3 = ((double)k)/((double)knum_k);

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
        else            k1 = ((double)i)/((double)knum_i);
        printf("%9.5f ",k1);fflush(stdout);
      }
      printf("\n");fflush(stdout);

      printf(" KGrids2: ");fflush(stdout);

      for (i=0;i<=knum_j-1;i++){
        if (knum_j==1)  k2 = 0.0;
        else            k2 = ((double)i)/((double)knum_j);
        printf("%9.5f ",k2);fflush(stdout);
      }
      printf("\n");fflush(stdout);

      printf(" KGrids3: ");fflush(stdout);
      for (i=0;i<=knum_k-1;i++){
        if (knum_k==1)  k3 = 0.0;
        else            k3 = ((double)i)/((double)knum_k);
        printf("%9.5f ",k3);fflush(stdout);
      }
      printf("\n");fflush(stdout);
    }
  } // end of else if (way_of_kpoint==3)

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
                make is1, ie1, is2, ie2
  ****************************************************/

  /* allocation */

  is1 = (int*)malloc(sizeof(int)*numprocs2);
  ie1 = (int*)malloc(sizeof(int)*numprocs2);
  is2 = (int*)malloc(sizeof(int)*numprocs2);
  ie2 = (int*)malloc(sizeof(int)*numprocs2);

  Num_Snd_EV = (int*)malloc(sizeof(int)*numprocs2);
  Num_Rcv_EV = (int*)malloc(sizeof(int)*numprocs2);

  /* make is1 and ie1 */ 

  if ( numprocs2<=n ){

    av_num = (double)n/(double)numprocs2;

    for (ID=0; ID<numprocs2; ID++){
      is1[ID] = (int)(av_num*(double)ID) + 1; 
      ie1[ID] = (int)(av_num*(double)(ID+1)); 
    }

    is1[0] = 1;
    ie1[numprocs2-1] = n; 

  }

  else{

    for (ID=0; ID<n; ID++){
      is1[ID] = ID + 1; 
      ie1[ID] = ID + 1;
    }
    for (ID=n; ID<numprocs2; ID++){
      is1[ID] =  1;
      ie1[ID] =  0;
    }
  }

  /* make is2 and ie2 */ 

  if ( numprocs2<=MaxN ){

    av_num = (double)MaxN/(double)numprocs2;

    for (ID=0; ID<numprocs2; ID++){
      is2[ID] = (int)(av_num*(double)ID) + 1; 
      ie2[ID] = (int)(av_num*(double)(ID+1)); 
    }

    is2[0] = 1;
    ie2[numprocs2-1] = MaxN; 
  }

  else{
    for (ID=0; ID<MaxN; ID++){
      is2[ID] = ID + 1; 
      ie2[ID] = ID + 1;
    }
    for (ID=MaxN; ID<numprocs2; ID++){
      is2[ID] = 1;
      ie2[ID] = 0;
    }
  }

  /****************************************************************
    making data structure of MPI communicaition for eigenvectors 
  ****************************************************************/

  for (ID=0; ID<numprocs2; ID++){
    Num_Snd_EV[ID] = 0;
    Num_Rcv_EV[ID] = 0;
  }

  for(i=0; i<na_rows2; i++){

    ig = np_rows2*nblk2*((i)/nblk2) + (i)%nblk2 + ((np_rows2+my_prow2)%np_rows2)*nblk2 + 1;

    po = 0;
    for (ID=0; ID<numprocs2; ID++){
      if (is2[ID]<=ig && ig <=ie2[ID]){
	po = 1;
	ID0 = ID;
	break;
      }
    }

    if (po==1) Num_Snd_EV[ID0] += na_cols2;
  }

  for (ID=0; ID<numprocs2; ID++){
    IDS = (myid2 + ID) % numprocs2;
    IDR = (myid2 - ID + numprocs2) % numprocs2;
    if (ID!=0){
      MPI_Isend(&Num_Snd_EV[IDS], 1, MPI_INT, IDS, 999, MPI_CommWD2[myworld2], &request);
      MPI_Recv(&Num_Rcv_EV[IDR],  1, MPI_INT, IDR, 999, MPI_CommWD2[myworld2], &stat);
      MPI_Wait(&request,&stat);
    }
    else{
      Num_Rcv_EV[IDR] = Num_Snd_EV[IDS];
    }
  }

  Max_Num_Snd_EV = 0;
  Max_Num_Rcv_EV = 0;
  for (ID=0; ID<numprocs2; ID++){
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

  /****************************************************
                      PrintMemory
  ****************************************************/

  if (firsttime && memoryusage_fileout) {
  PrintMemory("Band_DFT_NonCol: My_NZeros", sizeof(int)*numprocs0,NULL);
  PrintMemory("Band_DFT_NonCol: SP_NZeros", sizeof(int)*numprocs0,NULL);
  PrintMemory("Band_DFT_NonCol: SP_Atoms", sizeof(int)*numprocs0,NULL);
  PrintMemory("Band_DFT_NonCol: is1", sizeof(int)*numprocs2,NULL);
  PrintMemory("Band_DFT_NonCol: ie1", sizeof(int)*numprocs2,NULL);
  PrintMemory("Band_DFT_NonCol: is2", sizeof(int)*numprocs2,NULL);
  PrintMemory("Band_DFT_NonCol: ie2", sizeof(int)*numprocs2,NULL);
  PrintMemory("Band_DFT_NonCol: Num_Snd_EV", sizeof(int)*numprocs2,NULL);
  PrintMemory("Band_DFT_NonCol: Num_Rcv_EV", sizeof(int)*numprocs2,NULL);
  PrintMemory("Band_DFT_NonCol: index_Snd_i", sizeof(int)*Max_Num_Snd_EV,NULL);
  PrintMemory("Band_DFT_NonCol: index_Snd_j", sizeof(int)*Max_Num_Snd_EV,NULL);
  PrintMemory("Band_DFT_NonCol: EVec_Snd", sizeof(double)*Max_Num_Snd_EV*2,NULL);
  PrintMemory("Band_DFT_NonCol: index_Rcv_i", sizeof(int)*Max_Num_Rcv_EV,NULL);
  PrintMemory("Band_DFT_NonCol: index_Rcv_j", sizeof(int)*Max_Num_Rcv_EV,NULL);
  PrintMemory("Band_DFT_NonCol: EVec_Rcv", sizeof(double)*Max_Num_Rcv_EV*2,NULL);
  }

  /*********************************************** 
       create a director to save EVec1 files
  ***********************************************/

  if (cwf2w90_output_flag){
    char operate[1000];
    sprintf(operate,"%s%s_evec",filepath,filename);
    mkdir(operate,0775); 
  }

  /****************************************************
     allocation of arrays for Calc_CWF_Band_NonCol
  ****************************************************/

  Calc_CWF_Band_NonCol( "allocate", 0, kloop,knum_i,knum_j,knum_k,
			T_KGrids1,T_KGrids2,T_KGrids3,sum_weights,0.0,
			myid0,numprocs0,myid1,numprocs1,myid2,numprocs2,myworld1,myworld2,
			is2,ie2,MP,n2,MinN,MaxN,TNum_CWFs,MPI_CommWD1,
                        Comm_World_StartID1,MPI_CommWD2,Comm_World_StartID2, 
			CntOLP,EIGEN,EVec1,Full_EVec,nkpath, n_perk, kpath, kname,
                        H1, order_GA );

  /****************************************************
                      start kloop
  ****************************************************/

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

    MPI_Comm_split(MPI_CommWD2[myworld2],my_pcol,my_prow,&mpi_comm_rows);
    MPI_Comm_split(MPI_CommWD2[myworld2],my_prow,my_pcol,&mpi_comm_cols);

    mpi_comm_rows_int = MPI_Comm_c2f(mpi_comm_rows);
    mpi_comm_cols_int = MPI_Comm_c2f(mpi_comm_cols);

    if (scf_eigen_lib_flag==1 || (numprocs2<5 && scf_eigen_lib_flag!=CuSOLVER)){
      F77_NAME(solve_evp_complex,SOLVE_EVP_COMPLEX)
	( &n, &n, Cs, &na_rows, &ko[1], Ss, &na_rows, &nblk, &mpi_comm_rows_int, &mpi_comm_cols_int );
    }

    else if (scf_eigen_lib_flag==CuSOLVER && GPU_CPU_SWITCH_NUM<=n2 && na_rows==n && na_cols==n){
      OpenMX_CuSolver_DenseZheevx_1based(Cs,Ss,ko,n,n,"Band_DFT_NonCol_CWF overlap CuSOLVER");
    }

    else if (scf_eigen_lib_flag==2 || scf_eigen_lib_flag==CuSOLVER){

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

    Hamiltonian_Band_NC_Hs2( Hs11, Hs22, Hs12, Hs2, MPI_CommWD2[myworld2] );

    /* set H1 and order_GA */

    size_H1 = Get_OneD_HS_Col(1, CntOLP, H1, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);

    /* if (kloop0<num_kloop0) perform the calculation */

    if (kloop0<num_kloop0){

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

      /****************************************************
	 diagonalize the transformed H
      ****************************************************/

      Hamiltonian_Band_NC_Hs2( Hs11, Hs22, Hs12, Hs2, MPI_CommWD2[myworld2] );

      MPI_Comm_split(MPI_CommWD2[myworld2],my_pcol2,my_prow2,&mpi_comm_rows);
      MPI_Comm_split(MPI_CommWD2[myworld2],my_prow2,my_pcol2,&mpi_comm_cols);

      mpi_comm_rows_int = MPI_Comm_c2f(mpi_comm_rows);
      mpi_comm_cols_int = MPI_Comm_c2f(mpi_comm_cols);

      if (scf_eigen_lib_flag==1 || (numprocs2<5 && scf_eigen_lib_flag!=CuSOLVER)){
	F77_NAME(solve_evp_complex,SOLVE_EVP_COMPLEX)
	  ( &n2, &MaxN, Hs2, &na_rows2, &ko[1], Cs2, &na_rows2, &nblk2, &mpi_comm_rows_int, &mpi_comm_cols_int );
      }
      else if (scf_eigen_lib_flag==CuSOLVER && GPU_CPU_SWITCH_NUM<=n2 && na_rows2==n2 && na_cols2==n2){
	OpenMX_CuSolver_DenseZheevx_1based(Hs2,Cs2,ko,n2,MaxN,"Band_DFT_NonCol_CWF Hamiltonian CuSOLVER");
      }
      else if (scf_eigen_lib_flag==2 || scf_eigen_lib_flag==CuSOLVER){

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

      /**************************************************
            calculations of KS wave functions. 
      **************************************************/

      for(k=0; k<na_rows2*na_cols2; k++){
	Hs2[k].r = 0.0;
	Hs2[k].i = 0.0;
      }

      Cblacs_barrier(ictxt1_2,"A");
      F77_NAME(pzgemm,PZGEMM)( "T","T",&n2,&n2,&n2,&alpha,Cs2,&ONE,&ONE,descC2,Ss2,
			       &ONE,&ONE,descS2,&beta,Hs2,&ONE,&ONE,descH2);

      /* MPI communications of Hs2 */

      for (ID=0; ID<numprocs2; ID++){
    
	IDS = (myid2 + ID) % numprocs2;
	IDR = (myid2 - ID + numprocs2) % numprocs2;

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
	    MPI_Isend(index_Snd_i, Num_Snd_EV[IDS], MPI_INT, IDS, 999, MPI_CommWD2[myworld2], &request);
	  }
	  if (Num_Rcv_EV[IDR]!=0){
	    MPI_Recv(index_Rcv_i, Num_Rcv_EV[IDR], MPI_INT, IDR, 999, MPI_CommWD2[myworld2], &stat);
	  }
	  if (Num_Snd_EV[IDS]!=0){
	    MPI_Wait(&request,&stat);
	  }

	  if (Num_Snd_EV[IDS]!=0){
	    MPI_Isend(index_Snd_j, Num_Snd_EV[IDS], MPI_INT, IDS, 999, MPI_CommWD2[myworld2], &request);
	  }
	  if (Num_Rcv_EV[IDR]!=0){
	    MPI_Recv(index_Rcv_j, Num_Rcv_EV[IDR], MPI_INT, IDR, 999, MPI_CommWD2[myworld2], &stat);
	  }
	  if (Num_Snd_EV[IDS]!=0){
	    MPI_Wait(&request,&stat);
	  }

	  if (Num_Snd_EV[IDS]!=0){
	    MPI_Isend(EVec_Snd, Num_Snd_EV[IDS]*2, MPI_DOUBLE, IDS, 999, MPI_CommWD2[myworld2], &request);
	  }
	  if (Num_Rcv_EV[IDR]!=0){
	    MPI_Recv(EVec_Rcv, Num_Rcv_EV[IDR]*2, MPI_DOUBLE, IDR, 999, MPI_CommWD2[myworld2], &stat);
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

          m = (ig-is2[myid2])*n2+(jg-1);
	  EVec1[0][m].r = EVec_Rcv[2*k  ];
	  EVec1[0][m].i = EVec_Rcv[2*k+1];

	}
      } // ID

      /***********************************
          if cwf2w90_output_flag==1,  
          save EVec1 in binary format 
      ***********************************/

      if (cwf2w90_output_flag){

        char fname_EVec[YOUSO10];
        double tmp;

	sprintf(fname_EVec,"%s%s_evec/evec_%d",filepath,filename,kloop);
        Save_EVec_CWF2W90( "comp", fname_EVec, &tmp, EVec1[0], Full_EVec, n2, MinN, MaxN, 
                           myid2, numprocs2, is2, ie2, MPI_CommWD2[myworld2] ); 

        char fname_EVal[YOUSO10];

	sprintf(fname_EVal,"%s%s_evec/eval_%d",filepath,filename,kloop);
        Save_EIGENVAL_CWF2W90(fname_EVal, EIGEN[0][kloop], MinN, MaxN, myid2);
      }  

    } /* end of if (kloop0<num_kloop0) */

    /************************************************************
     calculation of closest Wannier functions.
     In Calc_CWF_Band_Col, MPI communication will be performed 
     in mpi_comm_level1. Thus, kloop0 should be the same among
     all the processes.  
    ************************************************************/

    if (measure_time) dtime(&Stime);

    Calc_CWF_Band_NonCol( "calc",(kloop0<num_kloop0),kloop,knum_i,knum_j,knum_k,
			  T_KGrids1,T_KGrids2,T_KGrids3,sum_weights,(double)T_k_op[kloop],
			  myid0,numprocs0,myid1,numprocs1,myid2,numprocs2,myworld1,myworld2,
			  is2,ie2,MP,n2,MinN,MaxN,TNum_CWFs,MPI_CommWD1,
                          Comm_World_StartID1,MPI_CommWD2,Comm_World_StartID2, 
			  CntOLP,EIGEN,EVec1,Full_EVec,nkpath, n_perk, kpath, kname,
                          H1, order_GA );
                  
    if (measure_time){
      dtime(&Etime);
      time6 += Etime - Stime;
    }

  } /* kloop0 */

  /****************************************************
     MPI:

     EIGEN
  ****************************************************/

  for (kloop=0; kloop<T_knum; kloop++){
    /* get ID in the zeroth world */
    ID = Comm_World_StartID1[0] + T_k_ID[myworld1][kloop];
    MPI_Bcast(&EIGEN[0][kloop][0], MaxN+1, MPI_DOUBLE, ID, mpi_comm_level1);
  } 

  /****************************************************
      freeing of arrays for Calc_CWF_Band_NonCol
  ****************************************************/

  Calc_CWF_Band_NonCol( "free", 0, kloop,knum_i,knum_j,knum_k,
			T_KGrids1,T_KGrids2,T_KGrids3,sum_weights,0.0,
			myid0,numprocs0,myid1,numprocs1,myid2,numprocs2,myworld1,myworld2,
			is2,ie2,MP,n2,MinN,MaxN,TNum_CWFs,MPI_CommWD1,
                        Comm_World_StartID1,MPI_CommWD2,Comm_World_StartID2, 
			CntOLP,EIGEN,EVec1,Full_EVec,nkpath, n_perk, kpath, kname,
                        H1, order_GA );

  /****************************************************
                       free arrays
  ****************************************************/

  if (cwf2w90_output_flag){
    free(Full_EVec);
  }  

  free(My_NZeros);
  free(SP_NZeros);
  free(SP_Atoms);

  for (i=0; i<List_YOUSO[7]; i++){
    free(ReEVec_i1[i]);
  }
  free(ReEVec_i1);

  for (i=0; i<List_YOUSO[7]; i++){
    free(ImEVec_i1[i]);
  }
  free(ImEVec_i1);

  for (i=0; i<List_YOUSO[7]; i++){
    free(ReEVec_i2[i]);
  }
  free(ReEVec_i2);

  for (i=0; i<List_YOUSO[7]; i++){
    free(ImEVec_i2[i]);
  }
  free(ImEVec_i2);

  for (i=0; i<List_YOUSO[7]; i++){
    free(ReEVec_j1[i]);
  }
  free(ReEVec_j1);

  for (i=0; i<List_YOUSO[7]; i++){
    free(ImEVec_j1[i]);
  }
  free(ImEVec_j1);

  for (i=0; i<List_YOUSO[7]; i++){
    free(ReEVec_j2[i]);
  }
  free(ReEVec_j2);

  for (i=0; i<List_YOUSO[7]; i++){
    free(ImEVec_j2[i]);
  }
  free(ImEVec_j2);

  free(is1);
  free(ie1);

  free(is2);
  free(ie2);

  free(Num_Snd_EV);
  free(Num_Rcv_EV);

  free(index_Snd_i);
  free(index_Snd_j);
  free(EVec_Snd); 
  free(index_Rcv_i);
  free(index_Rcv_j);
  free(EVec_Rcv);

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

  /* freeing of arrays */

  if ( CWF_Guiding_Orbital==1 || CWF_Guiding_Orbital==2 ){
    FreeArrays_NonCol_QAO2();
  }

  MPI_Barrier(mpi_comm_level1);
  dtime(&TEtime);
  time0 = TEtime - TStime;
  return time0;
}




static double Calc_CWF_Band_NonCol(
    char *mode,
    int own_calc_flag,
    int kloop,
    int knum_i, int knum_j, int knum_k,
    double *T_KGrids1,
    double *T_KGrids2,
    double *T_KGrids3,
    double sum_weights,
    double weight_kpoint,
    int myid0,
    int numprocs0,
    int myid1,
    int numprocs1,
    int myid2,
    int numprocs2,
    int myworld1,
    int myworld2,
    int *is2,
    int *ie2,
    int *MP,
    int n2,
    int MinN,
    int MaxN,
    int TNum_CWFs,
    MPI_Comm *MPI_CommWD1,
    int *Comm_World_StartID1,
    MPI_Comm *MPI_CommWD2,
    int *Comm_World_StartID2,
    double ****OLP0,
    double ***EIGEN,
    dcomplex **EVec1,
    dcomplex *Full_EVec,
    int nkpath, int *n_perk, double ***kpath, char ***kname,
    double *S1,
    int *order_GA )
{
  int AN,Gc_AN,GB_AN,Mc_AN,h_AN,Gh_AN,Rnh,wanB,wan1,tno1,wan2,tno2;
  int i,i1,j,l,l1,l2,l3,ll1,ll2,ll3,k,ig,jg,m1,mm,NumCWFs,num,p,q,ID,ID2,mmin,mmax,m;
  int current_myworld1,current_myworld2,current_kloop,loopN,iter,po;
  int EVec1_size,Max_EVec1_size,gidx,Lidx,dim,GA_AN,i0,pnum;
  int *MP2,*MP3,mpi_info[8],non_zero_sv_okay;
  double mu,min_mu,max_mu,dif;
  double Stime,Etime,time61=0.0,time62=0.0,time63=0.0,time64=0.0;
  double time65=0.0,time66=0.0,time67=0.0,time68=0.0;;
  double sum0,sum1,sum,max_x=60.0,x,FermiF,kRn,tmp,rtmp,itmp,w1=1.0e-10;
  double *sv,*rwork,k1,k2,k3,co,si;
  dcomplex *work,ctmp0;
  double complex *TmpEVec1,*InProd,*InProd_BasisFunc,*C2;
  double complex csum1,csum2,*csum1_orb,*csum2_orb,ctmp;
  static double complex ****Hop;
  static dcomplex *Cs,*Hs,*Vs,*Ws,*EVs_PAO,*WFs;
  double complex phase; 
  int num_zero_sv=0,rank,lwork,info;
  double Sum_Charges,Sum_Energies;
  double stime,etime;
  int ZERO=0,ONE=1;
  dcomplex alpha = {1.0,0.0}; dcomplex beta = {0.0,0.0};
  MPI_Status stat;
  MPI_Request request;
  MPI_Comm mpi_comm_rows, mpi_comm_cols;
  int mpi_comm_rows_int,mpi_comm_cols_int;
  double complex *SumHop;
  double complex ****TB_Hopping;
  int nblk_m,numprocs5,myid5;
  MPI_Comm mpi_comm_level5;
  FILE *fp_Hop,*fp_Dis_vs_H;
  char fname[YOUSO10];
  char fDis_vs_H[YOUSO10];
  int fp_Hop_ok = 0;
  int fp_Dis_vs_H_ok = 0;
  int *int_data;

  /*********************************************************************************  
  **********************************************************************************
   mode: allocate
   allocation of arrays for SCALAPCK
  **********************************************************************************
  *********************************************************************************/

  if (strcasecmp(mode,"allocate")==0){

    Allocate_Free_Band_NonCol_CWF( 1, myworld2, MPI_CommWD2, n2, MaxN, TNum_CWFs, knum_i, knum_j, knum_k, 
                                   &Cs, &Hs, &Vs, &Ws, &EVs_PAO, &WFs, &Hop );

    return 0.0;
  }

  /*********************************************************************************  
  **********************************************************************************
   mode: free
   freeing of arrays for SCALAPCK
  **********************************************************************************
  *********************************************************************************/

  if (strcasecmp(mode,"free")==0){

    /* DM_func */

    MPI_Allreduce( MPI_IN_PLACE, &DM_func, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
    DM_func = DM_func/sum_weights; 

    if (myid0==0) printf("DM_func=%15.12f  %15.12f per CWF\n",DM_func,DM_func/(double)TNum_CWFs);

    /* MPI_Allreduce of CWF_Charge and CWF_Energy */

    MPI_Allreduce( MPI_IN_PLACE, &CWF_Charge[0][0], TNum_CWFs, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
    MPI_Allreduce( MPI_IN_PLACE, &CWF_Energy[0][0], TNum_CWFs, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

    tmp = 1.0/sum_weights; 

    for (i=0; i<TNum_CWFs; i++){
      CWF_Charge[0][i] *= tmp; 
      CWF_Energy[0][i] *= tmp; 
    }
   
    Sum_Charges = 0.0;
    Sum_Energies = 0.0;;
    for (i=0; i<TNum_CWFs; i++){
      Sum_Charges  += CWF_Charge[0][i];
      Sum_Energies += CWF_Energy[0][i];
    }

    //printf("ABC5 TNum_CWFs=%2d myworld1=%2d myworld2=%2d %15.12f %15.12f\n",TNum_CWFs,myworld1,myworld2,Sum_Charges,Sum_Energies);

    /**********************************************************
      save the effective charges and local energies to a file
    ***********************************************************/
    
    if (CWF_Guiding_Orbital==1 || CWF_Guiding_Orbital==2){

      if ( myid0==Host_ID ){
        
	int i0;
	char file_CWF_Charge[YOUSO10];
	FILE *fp_CWF_Charge;
	char buf[fp_bsize];          /* setvbuf */
	double sumP,sumE,TZ;

	MP2 = (int*)malloc(sizeof(int)*(atomnum+1));

	/* calculate TZ */

	TZ = 0.0;
        TNum_CWFs = 0;
	for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
	  MP2[Gc_AN] = TNum_CWFs;
	  wan1 = WhatSpecies[Gc_AN];
	  TZ += Spe_Core_Charge[wan1];
          TNum_CWFs += 2*CWF_Num_predefined[wan1];
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
  	  free(MP2);

	} // if ((fp_CWF_Charge = fopen(file_CWF_Charge,"w")) != NULL)

	else{
	  printf("Failure of saving the CWF_Charge file.\n");
	}

      } // if ( myid0==Host_ID )

    } /* end of if (CWF_Guiding_Orbital==1 || CWF_Guiding_Orbital==2) */

    else if (CWF_Guiding_Orbital==3){

      if ( myid0==Host_ID ){
        
	int i0,gidx;
	char file_CWF_Charge[YOUSO10];
	FILE *fp_CWF_Charge;
	char buf[fp_bsize];          /* setvbuf */
	double sumP,sumE,TZ;

	MP2 = (int*)malloc(sizeof(int)*Num_CWF_Grouped_Atoms);

	/* calculate TZ */

	TZ = 0.0;
	for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
	  wan1 = WhatSpecies[Gc_AN];
	  TZ += Spe_Core_Charge[wan1];
	}

        TNum_CWFs = 0;
        for (gidx=0; gidx<Num_CWF_Grouped_Atoms; gidx++){
          MP2[gidx] = TNum_CWFs;
          TNum_CWFs += Num_CWF_MOs_Group[gidx];
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
  	  free(MP2);

	} // if ((fp_CWF_Charge = fopen(file_CWF_Charge,"w")) != NULL)

	else{
	  printf("Failure of saving the CWF_Charge file.\n");
	}

      } // if ( myid0==Host_ID )
    }

    /* free CWF_Charge and CWF_Energy */

    MPI_Barrier(mpi_comm_level1);

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
	}
	/* MO case */
	else if (CWF_Guiding_Orbital==3){
	  gidx = CWF_file_MOs[k];
	  pnum = Num_CWF_MOs_Group[gidx]; 
	}

	for (p=0; p<pnum; p++){

	  for (l1=-CWF_Plot_SuperCells[0]; l1<=CWF_Plot_SuperCells[0]; l1++){

	    ll1 = l1 + CWF_Plot_SuperCells[0]; 

	    for (l2=-CWF_Plot_SuperCells[1]; l2<=CWF_Plot_SuperCells[1]; l2++){

	      ll2 = l2 + CWF_Plot_SuperCells[1]; 

	      for (l3=-CWF_Plot_SuperCells[2]; l3<=CWF_Plot_SuperCells[2]; l3++){

		ll3 = l3 + CWF_Plot_SuperCells[2]; 
      	         
		MPI_Allreduce( MPI_IN_PLACE, &CWF_Coef_NC[k][p][ll1][ll2][ll3][0], n2, MPI_C_DOUBLE_COMPLEX, MPI_SUM, mpi_comm_level1);
	      }
	    }
	  }
	}
      }

    } // if (CWF_fileout_flag==1)

    /*************************************************************
          summation of Hop and store them into TB_Hopping
    **************************************************************/

    /****************************************************************************************
      if ( (TNum_CWFs*TNum_CWFs)<numprocs0 ), make a new MPI world, called mpi_comm_level5, 
      which includes MPI processes of TNum_CWFs*TNum_CWFs (=numprocs5), and perform 
      the band structure calculation in the world of mpi_comm_level5.
    ****************************************************************************************/

    if ( (TNum_CWFs*TNum_CWFs)<numprocs0 ){

      numprocs5 = TNum_CWFs*TNum_CWFs;
      Make_Comm_Worlds3(mpi_comm_level1,myid0,numprocs0,numprocs5,&mpi_comm_level5);
      if (myid0<numprocs5) MPI_Comm_rank(mpi_comm_level5,&myid5);
    }

    else{
      numprocs5 = numprocs0;
      myid5 = myid0;
      mpi_comm_level5 = mpi_comm_level1;
    }

    /* if (myid0<numprocs5), set up variables related to CWF5  */

    if (myid0<numprocs5){ 

      /* CWF5: setting of BLACS for matrices in size of TNum_CWFs x TNum_CWFs */

      np_cols_CWF5 = (int)(sqrt((float)numprocs5));
      do{
	if((numprocs5%np_cols_CWF5)==0) break;
	np_cols_CWF5--;
      } while (np_cols_CWF5>=2);
      np_rows_CWF5 = numprocs5/np_cols_CWF5;

      nblk_m = NBLK;
      while((nblk_m*np_rows_CWF5>TNum_CWFs || nblk_m*np_cols_CWF5>TNum_CWFs) && (nblk_m > 1)){
	nblk_m /= 2;
      }
      if(nblk_m<1) nblk_m = 1;

      MPI_Allreduce(&nblk_m,&nblk_CWF5,1,MPI_INT,MPI_MIN,mpi_comm_level5);

      my_prow_CWF5 = myid5/np_cols_CWF5;
      my_pcol_CWF5 = myid5%np_cols_CWF5;

      na_rows_CWF5 = numroc_(&TNum_CWFs, &nblk_CWF5, &my_prow_CWF5, &ZERO, &np_rows_CWF5 ); 
      na_cols_CWF5 = numroc_(&TNum_CWFs, &nblk_CWF5, &my_pcol_CWF5, &ZERO, &np_cols_CWF5 );

      bhandle1_CWF5 = Csys2blacs_handle(mpi_comm_level5);
      ictxt1_CWF5 = bhandle1_CWF5;

      Cblacs_gridinit(&ictxt1_CWF5, "Row", np_rows_CWF5, np_cols_CWF5);

      MPI_Allreduce(&na_rows_CWF5,&na_rows_max_CWF5,1,MPI_INT,MPI_MAX,mpi_comm_level5);
      MPI_Allreduce(&na_cols_CWF5,&na_cols_max_CWF5,1,MPI_INT,MPI_MAX,mpi_comm_level5);

      descinit_( desc_CWF5, &TNum_CWFs, &TNum_CWFs, &nblk_CWF5, &nblk_CWF5,  
		 &ZERO, &ZERO, &ictxt1_CWF5, &na_rows_CWF5,  &info); 

      /* allocation of TB_Hopping */   

      if ( Band_disp_switch==1 && Band_Nkpath>0 ){
	TB_Hopping = (double complex****)malloc(sizeof(double complex***)*knum_i);
	for ( ll1=0; ll1<knum_i; ll1++ ){
	  TB_Hopping[ll1] = (double complex***)malloc(sizeof(double complex**)*knum_j);
	  for ( ll2=0; ll2<knum_j; ll2++ ){
	    TB_Hopping[ll1][ll2] = (double complex**)malloc(sizeof(double complex*)*knum_k);
	    for ( ll3=0; ll3<knum_k; ll3++ ){
	      TB_Hopping[ll1][ll2][ll3] = (double complex*)malloc(sizeof(double complex)*(na_rows_CWF5*na_cols_CWF5+1));
	    }
	  }
	}
      }

    } /* end of if (myid0<numprocs5) */

    /* allocation of SumHop */

    SumHop = (double complex*)malloc(sizeof(double complex)*TNum_CWFs*TNum_CWFs);

    /* fopen fp_Hop and save information into filename.CWF.Hop */

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
      int_data[2] = knum_i;
      int_data[3] = knum_j;
      int_data[4] = knum_k;
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

      if (fp_Hop_ok==1) fwrite(int_data,sizeof(int),(int_data[0]+6),fp_Hop);

      /* save tv into a file of *.CWF.Hop */

      if (fp_Hop_ok==1){

	double tvecs[9];
	tvecs[0] = tv[1][1]; tvecs[1] = tv[1][2]; tvecs[2] = tv[1][3];               
	tvecs[3] = tv[2][1]; tvecs[4] = tv[2][2]; tvecs[5] = tv[2][3];               
	tvecs[6] = tv[3][1]; tvecs[7] = tv[3][2]; tvecs[8] = tv[3][3]; 

	fwrite(tvecs,sizeof(double),9,fp_Hop);
      }
    }

    /* fopen fp_Dis_vs_H */
    
    if (myid0==0 && CWF_Dis_vs_H==1){
      sprintf(fDis_vs_H,"%s%s.CWF.Dis_vs_H",filepath,filename);
      if ((fp_Dis_vs_H = fopen(fDis_vs_H,"w")) != NULL) fp_Dis_vs_H_ok = 1;

      if (fp_Dis_vs_H_ok==1){
        fprintf(fp_Dis_vs_H,"#\n"); 
        fprintf(fp_Dis_vs_H,"# GA, GB, l1, l2, l3, i, j, distance (Ang.), Hopping integral (eV)\n"); 
        fprintf(fp_Dis_vs_H,"#\n"); 
      }
    }

    /* sum Hop */

    for ( ll1=0; ll1<knum_i; ll1++ ){
      for ( ll2=0; ll2<knum_j; ll2++ ){
	for ( ll3=0; ll3<knum_k; ll3++ ){
                        
	  for (p=0; p<(TNum_CWFs*TNum_CWFs); p++){
            SumHop[p] = 0.0;
	  }

	  for (i=0; i<na_rows_CWF1; i++){

	    ig = np_rows_CWF1*nblk_CWF1*((i)/nblk_CWF1) + (i)%nblk_CWF1
	      + ((np_rows_CWF1+my_prow_CWF1)%np_rows_CWF1)*nblk_CWF1;

	    if (ig<TNum_CWFs){

	      for (j=0; j<na_cols_CWF1; j++){

		jg = np_cols_CWF1*nblk_CWF1*((j)/nblk_CWF1) + (j)%nblk_CWF1 
		  + ((np_cols_CWF1+my_pcol_CWF1)%np_cols_CWF1)*nblk_CWF1;

		if (jg<TNum_CWFs){
		  SumHop[jg*TNum_CWFs+ig] += Hop[ll1][ll2][ll3][j*na_rows_CWF1+i]; 
		}
	      }
	    }
	  }

	  MPI_Allreduce( MPI_IN_PLACE, SumHop, TNum_CWFs*TNum_CWFs, MPI_C_DOUBLE_COMPLEX, MPI_SUM, mpi_comm_level1 );

	  /* save SumHop into a file of *.CWF.Hop */

	  if (myid0==Host_ID && fp_Hop_ok==1){
	    fwrite(SumHop,sizeof(double complex),(TNum_CWFs*TNum_CWFs),fp_Hop);
	  }

          /* save Hop with the distance between two sites. */

	  if (myid0==0 && CWF_Dis_vs_H==1 && fp_Dis_vs_H_ok==1){

            int GA,GB,wanA,wanB,p1,p2,l1,l2,l3,gidx1,gidx2,Lidx1,Lidx2;
            double dx,dy,dz,dis,xA,yA,zA,xB,yB,zB,t,re,im; 

	    l1 = ll1 - (knum_i-1)/2;
	    l2 = ll2 - (knum_j-1)/2;
	    l3 = ll3 - (knum_k-1)/2;

            if (CWF_Guiding_Orbital==1 || CWF_Guiding_Orbital==2){

	      p1 = 0;
	      for (GA=1; GA<=atomnum; GA++){
		wanA = WhatSpecies[GA];

		p2 = 0;
		for (GB=1; GB<=atomnum; GB++){
		  wanB = WhatSpecies[GB];

		  dx = Gxyz[GB][1] + (double)l1*tv[1][1] + (double)l2*tv[2][1] + (double)l3*tv[3][1] - Gxyz[GA][1]; 
		  dy = Gxyz[GB][2] + (double)l1*tv[1][2] + (double)l2*tv[2][2] + (double)l3*tv[3][2] - Gxyz[GA][2]; 
		  dz = Gxyz[GB][3] + (double)l1*tv[1][3] + (double)l2*tv[2][3] + (double)l3*tv[3][3] - Gxyz[GA][3]; 
		  dis = sqrt(dx*dx+dy*dy+dz*dz);

		  for (i=0; i<2*CWF_Num_predefined[wanA]; i++){
		    for (j=0; j<2*CWF_Num_predefined[wanB]; j++){

		      re = creal(SumHop[(p2+j)*TNum_CWFs+(p1+i)]);                    
		      im = cimag(SumHop[(p2+j)*TNum_CWFs+(p1+i)]);
		      t = sqrt(re*re+im*im)*eV2Hartree;

		      fprintf(fp_Dis_vs_H,"%2d %2d %2d %2d %2d %2d %2d  %18.15f %18.15f\n",
			      GA,GB,l1,l2,l3,i,j,dis*BohrR,t); 

		    } 
		  }

		  p2 += 2*CWF_Num_predefined[wanB]; 

		} // GB

		p1 += 2*CWF_Num_predefined[wanA]; 

	      } // GA 

	    } // end of if (CWF_Guiding_Orbital==1 || CWF_Guiding_Orbital==2)

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

		  dx = xB + (double)l1*tv[1][1] + (double)l2*tv[2][1] + (double)l3*tv[3][1] - xA;
		  dy = yB + (double)l1*tv[1][2] + (double)l2*tv[2][2] + (double)l3*tv[3][2] - yA;
		  dz = zB + (double)l1*tv[1][3] + (double)l2*tv[2][3] + (double)l3*tv[3][3] - zA;
		  dis = sqrt(dx*dx+dy*dy+dz*dz);

		  for (i=0; i<Num_CWF_MOs_Group[gidx1]; i++){
		    for (j=0; j<Num_CWF_MOs_Group[gidx2]; j++){

		      re = creal(SumHop[(p2+j)*TNum_CWFs+(p1+i)]);                    
		      im = cimag(SumHop[(p2+j)*TNum_CWFs+(p1+i)]);
		      t = sqrt(re*re+im*im)*eV2Hartree;

		      fprintf(fp_Dis_vs_H,"%2d %2d %2d %2d %2d %2d %2d  %18.15f %18.15f\n",
			      gidx1+1,gidx2+1,l1,l2,l3,i,j,dis*BohrR,t); 
		    }
		  }

		  p2 += Num_CWF_MOs_Group[gidx2];

		} // gidx2

		p1 += Num_CWF_MOs_Group[gidx1];
                 
	      } // gidx1

	    } // end of else if (CWF_Guiding_Orbital==3)

	  } // end of if (myid0==0 && CWF_Dis_vs_H==1 && fp_Dis_vs_H_ok==1)

	  /* store SumHop into TB_Hopping */

	  if ( Band_disp_switch==1 && Band_Nkpath>0 && myid0<numprocs5 ){

	    for (i=0; i<na_rows_CWF5; i++){

	      ig = np_rows_CWF5*nblk_CWF5*((i)/nblk_CWF5) + (i)%nblk_CWF5
		+ ((np_rows_CWF5+my_prow_CWF5)%np_rows_CWF5)*nblk_CWF5;

	      for (j=0; j<na_cols_CWF5; j++){

		jg = np_cols_CWF5*nblk_CWF5*((j)/nblk_CWF5) + (j)%nblk_CWF5 
		  + ((np_cols_CWF5+my_pcol_CWF5)%np_cols_CWF5)*nblk_CWF5;

		TB_Hopping[ll1][ll2][ll3][j*na_rows_CWF5+i] = SumHop[jg*TNum_CWFs+ig];
	      }
	    }
	  }

	} // ll1
      } // ll2 
    } // ll1

    /* fclose fp_Dis_vs_H */
    
    if (myid0==0 && CWF_Dis_vs_H==1 && fp_Dis_vs_H_ok==1){
      fclose(fp_Dis_vs_H);
    }

    /* fclose fp_Hop */

    if (myid0==Host_ID){
      free(int_data);
      if (fp_Hop_ok==1) fclose(fp_Hop);
    }

    /*************************************************************
                calculation of the band dispersion 
    **************************************************************/

    if ( Band_disp_switch==1 && Band_Nkpath>0 && myid0<numprocs5 ){
      Band_Dispersion_NonCol_CWF( nkpath, n_perk, kpath, kname, TNum_CWFs, 
                                  TB_Hopping, knum_i, knum_j, knum_k, mpi_comm_level5 );    
    } 

    /* freeing of arrays */   

    free(SumHop);

    if (myid0<numprocs5){ 
      Cfree_blacs_system_handle(bhandle1_CWF5);
      Cblacs_gridexit(ictxt1_CWF5);
    }

    if ( Band_disp_switch==1 && Band_Nkpath>0 && myid0<numprocs5 ){

      for ( ll1=0; ll1<knum_i; ll1++ ){
	for ( ll2=0; ll2<knum_j; ll2++ ){
	  for ( ll3=0; ll3<knum_k; ll3++ ){
	    free(TB_Hopping[ll1][ll2][ll3]);
	  }
	  free(TB_Hopping[ll1][ll2]);
	}
	free(TB_Hopping[ll1]);
      }
      free(TB_Hopping);
    }

    /*************************************************************
                        free arrays and return
    **************************************************************/

    Allocate_Free_Band_NonCol_CWF( 2, myworld2, MPI_CommWD2, n2, MaxN, TNum_CWFs, knum_i, knum_j, knum_k, 
                                   &Cs, &Hs, &Vs, &Ws, &EVs_PAO, &WFs, &Hop );

    return 0.0;
  }

  /*********************************************************************************  
  **********************************************************************************
   mode: calc
  **********************************************************************************
  *********************************************************************************/

  if (measure_time) dtime(&Stime);

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

  /* find the maximum size of EVec1 */

  EVec1_size = 0;
  for (ID=0; ID<numprocs2; ID++){
    if ( EVec1_size < (n2*(ie2[ID]-is2[ID]+1)) ) EVec1_size = n2*(ie2[ID]-is2[ID]+1);
  }

  MPI_Allreduce(&EVec1_size,&Max_EVec1_size,1,MPI_INT,MPI_MAX,mpi_comm_level1);

  /* allocation of arrays */

  csum1_orb = (double complex*)malloc(sizeof(double complex)*List_YOUSO[7]);
  csum2_orb = (double complex*)malloc(sizeof(double complex)*List_YOUSO[7]);
  TmpEVec1 = (double complex*)malloc(sizeof(double complex)*Max_EVec1_size);
  InProd = (double complex*)malloc(sizeof(double complex)*Max_EVec1_size);
  InProd_BasisFunc = (double complex*)malloc(sizeof(double complex)*Max_EVec1_size);
  C2 = (double complex*)malloc(sizeof(double complex)*2*List_YOUSO[7]);
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

  lwork = (int)ctmp0.r;
  work = (dcomplex*)malloc(sizeof(dcomplex)*lwork);

  if (measure_time){
    dtime(&Etime);
    time61 += Etime - Stime;
  }

  /***************************************************
   calculation of <Bloch function|localized orbital>
  ***************************************************/

  /* initialize Cs and Hs */

  for (i=0; i<na_rows_CWF4*na_cols_CWF4; i++){
    Cs[i] = Complex(0.0,0.0);
    Hs[i] = Complex(0.0,0.0);
    Ws[i] = Complex(0.0,0.0);
    Vs[i] = Complex(0.0,0.0);
  }

  for (ID=0; ID<numprocs2; ID++){

    if (ID==myid2){
      ID2 = myid2;
      num = n2*(ie2[ID2]-is2[ID2]+1);
      mpi_info[0] = ID2;
      mpi_info[1] = num;
      mpi_info[2] = is2[ID2];
      mpi_info[3] = ie2[ID2];
      mpi_info[4] = myworld1;
      mpi_info[5] = myworld2;
      mpi_info[6] = kloop;
    }

    /* MPI_Bcast of mpi_info */

    MPI_Bcast( &mpi_info[0], 7, MPI_INT, ID, MPI_CommWD2[myworld2] );

    /* set parameters */

    ID2              = mpi_info[0];
    num              = mpi_info[1];
    mmin             = mpi_info[2];
    mmax             = mpi_info[3];
    current_myworld1 = mpi_info[4];
    current_myworld2 = mpi_info[5];
    current_kloop    = mpi_info[6];
    k1 = T_KGrids1[current_kloop]; 
    k2 = T_KGrids2[current_kloop]; 
    k3 = T_KGrids3[current_kloop]; 

    /* initialize InProd and InProd_BasisFunc */

    for (p=0; p<Max_EVec1_size; p++){
      InProd[p] = 0.0 + I*0.0;
      InProd_BasisFunc[p] = 0.0 + I*0.0;
    }

    if ( num!=0 ){

      if (measure_time) dtime(&Stime);

      /* MPI_Bcast of EVec1 */

      if ( ID==myid2 ){
	for (i=0; i<num; i++){
	  TmpEVec1[i] = EVec1[0][i].r + I*EVec1[0][i].i;
	}
      }

      MPI_Bcast( &TmpEVec1[0], num, MPI_C_DOUBLE_COMPLEX, ID, MPI_CommWD2[myworld2] );

      /* store TmpEVec1 into EVs_PAO in the block cyclic form with n2 x MaxN.  */

      if ( myworld1==current_myworld1 && myworld2==current_myworld2 ){

	for (j=0; j<na_cols_CWF4; j++){

	  m1 = np_cols_CWF4*nblk_CWF4*((j)/nblk_CWF4) + (j)%nblk_CWF4 
	    + ((np_cols_CWF4+my_pcol_CWF4)%np_cols_CWF4)*nblk_CWF4;

	  if ((mmin-1)<=m1 && m1<=(mmax-1)){ 

	    for (i=0; i<na_rows_CWF4; i++){

	      ig = np_rows_CWF4*nblk_CWF4*((i)/nblk_CWF4) + (i)%nblk_CWF4
		+ ((np_rows_CWF4+my_prow_CWF4)%np_rows_CWF4)*nblk_CWF4;

	      EVs_PAO[j*na_rows_CWF4+i].r =creal(TmpEVec1[ (m1-mmin+1)*n2+ig ]);
	      EVs_PAO[j*na_rows_CWF4+i].i =cimag(TmpEVec1[ (m1-mmin+1)*n2+ig ]);

	    } // i

	  } // if ((mmin-1)<=m1 && m1<=(mmax-1))
	} // j
      } // if ( myworld1==current_myworld1 && myworld2==current_myworld2 )

      /* calculate <Bloch functions|basis functions> */

      for (m=0; m<=(mmax-mmin); m++){  // loop for KS index 

        k = 0;
        for (AN=1; AN<=atomnum; AN++){

  	  Gc_AN = order_GA[AN];
	  wan1 = WhatSpecies[Gc_AN];
	  tno1 = Spe_Total_CNO[wan1];

	  for (i=0; i<tno1; i++){
            csum1_orb[i] = 0.0 + I*0.0;
            csum2_orb[i] = 0.0 + I*0.0;
          }

	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	    Gh_AN = natn[Gc_AN][h_AN];
	    Rnh = ncn[Gc_AN][h_AN];
	    wan2 = WhatSpecies[Gh_AN];
	    tno2 = Spe_Total_CNO[wan2];

	    l1 = atv_ijk[Rnh][1];
	    l2 = atv_ijk[Rnh][2];
	    l3 = atv_ijk[Rnh][3];
	    kRn = k1*(double)l1 + k2*(double)l2 + k3*(double)l3;
	    phase = cos(2.0*PI*kRn) + I*sin(2.0*PI*kRn);

	    for (i=0; i<tno1; i++){

	      for (j=0; j<tno2; j++){

                m1 = m*n2 + (MP[Gh_AN]-1+j);
		C2[j] = phase*TmpEVec1[m1];

		m1 = m*n2 + (MP[Gh_AN]-1+j+n2/2);
		C2[tno2+j] = phase*TmpEVec1[m1];
	      }      

	      for (j=0; j<tno2; j++){

		csum1_orb[i] += (S1[k]+I*0.0)*C2[j];
		csum2_orb[i] += (S1[k]+I*0.0)*C2[tno2+j];

                k++;

	      } // j
	    } // i
	  } // h_AN

          /* store <Bloch functions|basis functions> */
          // take a conjugate complex by considering <Bloch functions|basis functions>

          for (i=0; i<tno1; i++){

	    p = m*n2 + MP[Gc_AN] - 1 + i;
	    InProd_BasisFunc[p] = creal(csum1_orb[i]) - I*cimag(csum1_orb[i]);  

	    p = m*n2 + MP[Gc_AN] - 1 + i + n2/2;
	    InProd_BasisFunc[p] = creal(csum2_orb[i]) - I*cimag(csum2_orb[i]);  

	  } // i

	} // AN
      } // m

      /* AO and HO case: calculate <Bloch functions|guiding functions> */

      if (CWF_Guiding_Orbital==1 || CWF_Guiding_Orbital==2){

	for (m=0; m<=(mmax-mmin); m++){  // loop for KS index 
	  for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){

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
	  } // Gc_AN 
	} // m

      } // end of if (CWF_Guiding_Orbital==1 || CWF_Guiding_Orbital==2)

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

      if (measure_time){
	dtime(&Etime);
	time62 += Etime - Stime;
      }

      /* store InProd into an array: Cs in the SCALAPACK form */       

      if (measure_time) dtime(&Stime);

      if ( myworld1==current_myworld1 && myworld2==current_myworld2 ){

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

      } // if ( myworld1==current_myworld1 && myworld2==current_myworld2 )

      if (measure_time){
	dtime(&Etime);
	time63 += Etime - Stime;
      }

    } // if ( num!=0 )
  } // ID

  /*********************************************************************
    Disentanling procedure:
    apply weighting based on the KS eigenvalue, 
    where the energy range is specified by CWF.disentangling.Erange. 
  *********************************************************************/

  if (own_calc_flag){

    double b0,b1,e,e0,e1,x;
    double weight,in0,in1,out0,out1; 

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

      e = EIGEN[0][kloop][m1];  

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
  }

  /********************************************************
    if cwf2w90_output_flag==1,  
    save <Bloch functions|guiding functions>
    using MPI_File_write_at
  ********************************************************/

  if (own_calc_flag && cwf2w90_output_flag){

    char fname_ProM[YOUSO10];
    FILE *fp_ProM;
    double tmp;

    sprintf(fname_ProM,"%s%s_evec/pm_%d",filepath,filename,kloop);

    Save_Projection_Matrix("comp", fname_ProM,
			   &tmp, Cs, Full_EVec, 
			   MinN, MaxN, TNum_CWFs,
			   nblk_CWF3, 
			   na_rows_CWF3, na_cols_CWF3,
			   np_rows_CWF3, np_cols_CWF3,
			   my_prow_CWF3, my_pcol_CWF3,
			   MPI_CommWD2[myworld2]);
  }

  /********************************************************
      Singular Value Decomposition (SVD) of Cs.
      As for how to set desc_CWF1, see also the comment 
      in Allocate_Free_Band_Col_CWF
  ********************************************************/

  /* As for the size of Cs, Ws, and Vs, see https://manpages.ubuntu.com/manpages/focal/man3/pdgesvd.3.html */

  if (own_calc_flag){

    if (measure_time) dtime(&Stime);

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
    if (myid0==0){
      for (i=0; i<TNum_CWFs; i++){
        printf("ZZZ1 of Cs myworld1=%2d myid0=%2d i=%2d sv=%18.15f\n",myworld1,myid0,i,sv[i]);fflush(stdout);
      }
    }
    */

    if (myid2==0){
      for (i=0; i<TNum_CWFs; i++){
        dif = sv[i] - 1.0;
        DM_func += weight_kpoint*dif*dif;
      }
    }

    if (measure_time){
      dtime(&Etime);
      time64 += Etime - Stime;
    }

  } // if (own_calc_flag)

  /**************************************************************
   if (own_calc_flag), the following calculations are performed:
   1. Polar Decomposition (PD) of Cs
   2. Calculate CWF w.r.t PAOs
   3. Store WFs with phase factors in CWF_Coef
   4. calculations of effective charges and local band energies
  ***************************************************************/

  if (own_calc_flag){

    /*******************************************
        Polar Decomposition (PD) of Cs
    *******************************************/

    if (measure_time) dtime(&Stime);

    Cblacs_barrier(ictxt1_CWF1,"A");
    F77_NAME(pzgemm,PZGEMM)( "N","N",
			     &MaxN, &TNum_CWFs, &TNum_CWFs, 
			     &alpha,
			     Ws,&ONE,&ONE,desc_CWF3,
			     Vs,&ONE,&ONE,desc_CWF1,
			     &beta,
			     Hs,&ONE,&ONE,desc_CWF3 );

    if (measure_time){
      dtime(&Etime);
      time65 += Etime - Stime;
    }

    /**********************************************************
     calculate CWF w.r.t PAOs: 
                    EVs_PAO * Hs -> WFs
           (n2 x MaxN) * (MaxN x TNum_CWFs) -> (n2 x TNum_CWFs) 
             WF4               WF3               WF2
      WFs is Fk.
    ***********************************************************/

    if (measure_time) dtime(&Stime);

    Cblacs_barrier(ictxt1_CWF4,"A");
    F77_NAME(pzgemm,PZGEMM)( "N","N",
			     &n2, &TNum_CWFs, &MaxN, 
			     &alpha,
			     EVs_PAO,   &ONE, &ONE, desc_CWF4,
			     Hs,        &ONE, &ONE, desc_CWF3,
			     &beta,
			     WFs,       &ONE, &ONE, desc_CWF2 );

    if (measure_time){
      dtime(&Etime);
      time66 += Etime - Stime;
    }

    /***********************************************************
           store WFs with phase factors in CWF_Coef_NC
    ***********************************************************/

    if (measure_time) dtime(&Stime);

    if (CWF_fileout_flag==1){
     
      int idx;

      k1 = T_KGrids1[kloop]; 
      k2 = T_KGrids2[kloop]; 
      k3 = T_KGrids3[kloop]; 
      tmp = weight_kpoint/sum_weights; 

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

	  for (j=0; j<na_cols_CWF2; j++){

	    jg = np_cols_CWF2*nblk_CWF2*((j)/nblk_CWF2) + (j)%nblk_CWF2 
	      + ((np_cols_CWF2+my_pcol_CWF2)%np_cols_CWF2)*nblk_CWF2;

	    if (q==jg){

	      for (l1=-CWF_Plot_SuperCells[0]; l1<=CWF_Plot_SuperCells[0]; l1++){

		ll1 = l1 + CWF_Plot_SuperCells[0]; 

		for (l2=-CWF_Plot_SuperCells[1]; l2<=CWF_Plot_SuperCells[1]; l2++){

		  ll2 = l2 + CWF_Plot_SuperCells[1]; 

		  for (l3=-CWF_Plot_SuperCells[2]; l3<=CWF_Plot_SuperCells[2]; l3++){

		    ll3 = l3 + CWF_Plot_SuperCells[2]; 

		    kRn = k1*(double)l1 + k2*(double)l2 + k3*(double)l3;
		    si = sin(2.0*PI*kRn);
		    co = cos(2.0*PI*kRn);

		    for (i=0; i<na_rows_CWF2; i++){

		      ig = np_rows_CWF2*nblk_CWF2*((i)/nblk_CWF2) + (i)%nblk_CWF2
			+ ((np_rows_CWF2+my_prow_CWF2)%np_rows_CWF2)*nblk_CWF2;

		      CWF_Coef_NC[k][p][ll1][ll2][ll3][ig] += tmp*(WFs[j*na_rows_CWF2+i].r*co - WFs[j*na_rows_CWF2+i].i*si)
                                                          + I*tmp*(WFs[j*na_rows_CWF2+i].r*si + WFs[j*na_rows_CWF2+i].i*co);

		    } // i
		  } // l3
		} // l2
	      } // l1
	    } // if (q==jg)
	  } // j
	} // p
      } // k 

    } // end of if (CWF_fileout_flag==1)

    if (measure_time){
      dtime(&Etime);
      time67 += Etime - Stime;
    }

    /***********************************************************
     calculations of effective charges and local band energies
    ***********************************************************/

    if (measure_time) dtime(&Stime);

    for (i=0; i<na_rows_CWF3; i++){

      ig = np_rows_CWF3*nblk_CWF3*((i)/nblk_CWF3) + (i)%nblk_CWF3
	+ ((np_rows_CWF3+my_prow_CWF3)%np_rows_CWF3)*nblk_CWF3 + 1;

      for (j=0; j<na_cols_CWF3; j++){

	jg = np_cols_CWF3*nblk_CWF3*((j)/nblk_CWF3) + (j)%nblk_CWF3 
	  + ((np_cols_CWF3+my_pcol_CWF3)%np_cols_CWF3)*nblk_CWF3;

	x = (EIGEN[0][kloop][ig] - ChemP)*Beta;
	if (x<=-max_x) x = -max_x;
	if (max_x<=x)  x = max_x;
	FermiF = 1.0/(1.0 + exp(x));

	rtmp = Hs[j*na_rows_CWF3+i].r;
	itmp = Hs[j*na_rows_CWF3+i].i;
	tmp = weight_kpoint*FermiF*(rtmp*rtmp+itmp*itmp);

	CWF_Charge[0][jg] += tmp;
	CWF_Energy[0][jg] += tmp*EIGEN[0][kloop][ig];
      }
    }

    /***********************************************************
                 calculation of <W_{i0}|H|W_{jRn}>
    ***********************************************************/

    /* E x Hs -> Cs */

    for (i=0; i<na_rows_CWF3; i++){

      ig = np_rows_CWF3*nblk_CWF3*((i)/nblk_CWF3) + (i)%nblk_CWF3
	+ ((np_rows_CWF3+my_prow_CWF3)%np_rows_CWF3)*nblk_CWF3 + 1;

      for (j=0; j<na_cols_CWF3; j++){

	x = EIGEN[0][kloop][ig];
        Cs[j*na_rows_CWF3+i].r = x*Hs[j*na_rows_CWF3+i].r;
        Cs[j*na_rows_CWF3+i].i = x*Hs[j*na_rows_CWF3+i].i;
      }
    }

    /* Hs^dag x Cs -> Ws */

    Cblacs_barrier(ictxt1_CWF1,"A");
    F77_NAME(pzgemm,PZGEMM)( "C","N",
			     &TNum_CWFs, &TNum_CWFs, &MaxN,
			     &alpha,
			     Hs,&ONE,&ONE,desc_CWF3,
			     Cs,&ONE,&ONE,desc_CWF3,
			     &beta,
			     Ws,&ONE,&ONE,desc_CWF1 );

    /*
       Hop += Ws*exp(-ikRn)
    */

    k1 = T_KGrids1[kloop]; 
    k2 = T_KGrids2[kloop]; 
    k3 = T_KGrids3[kloop]; 

    tmp = weight_kpoint/sum_weights; 

    for ( l1=-(knum_i-1)/2; l1<=(knum_i-1)/2; l1++ ){

      ll1 = l1 + (knum_i-1)/2;

      for ( l2=-(knum_j-1)/2; l2<=(knum_j-1)/2; l2++ ){

        ll2 = l2 + (knum_j-1)/2;

	for ( l3=-(knum_k-1)/2; l3<=(knum_k-1)/2; l3++ ){

          ll3 = l3 + (knum_k-1)/2;

          kRn = k1*(double)l1 + k2*(double)l2 + k3*(double)l3;
	  si = tmp*sin(-2.0*PI*kRn);
	  co = tmp*cos(-2.0*PI*kRn);
 
	  for (i=0; i<na_rows_CWF1; i++){
	    for (j=0; j<na_cols_CWF1; j++){
	      Hop[ll1][ll2][ll3][j*na_rows_CWF1+i] += (co*Ws[j*na_rows_CWF1+i].r - si*Ws[j*na_rows_CWF1+i].i)
     		                                  + I*(si*Ws[j*na_rows_CWF1+i].r + co*Ws[j*na_rows_CWF1+i].i);

	    }
	  }
	}      
      }      
    }      

    if (measure_time){
      dtime(&Etime);
      time68 += Etime - Stime;
    }

  } // if (own_calc_flag)

  /* freeing of arrays */

  free(csum1_orb);
  free(csum2_orb);
  free(TmpEVec1);
  free(MP2);
  free(InProd);
  free(InProd_BasisFunc);
  free(C2);

  free(sv);
  free(work);
  free(rwork);

  /* mearuring elapsed time */

  if (measure_time){
    printf("myid0=%2d time61=%15.12f\n",myid0,time61); 
    printf("myid0=%2d time62=%15.12f\n",myid0,time62); 
    printf("myid0=%2d time63=%15.12f\n",myid0,time63); 
    printf("myid0=%2d time64=%15.12f\n",myid0,time64); 
    printf("myid0=%2d time65=%15.12f\n",myid0,time65); 
    printf("myid0=%2d time66=%15.12f\n",myid0,time66); 
    printf("myid0=%2d time67=%15.12f\n",myid0,time67); 
    printf("myid0=%2d time68=%15.12f\n",myid0,time68); 
  }

  dtime(&etime);
  return (etime-stime);
}





void Allocate_Free_Band_NonCol_CWF( int todo_flag, 
				    int myworld2, 
				    MPI_Comm *MPI_CommWD2,
				    int n2,
				    int MaxN,
				    int TNum_CWFs,
				    int knum_i, int knum_j, int knum_k,
				    dcomplex **Cs,
				    dcomplex **Hs, 
				    dcomplex **Vs,
				    dcomplex **Ws,
				    dcomplex **EVs_PAO,
				    dcomplex **WFs,
				    double complex *****Hop )
{
  static int firsttime=1;
  int ZERO=0, ONE=1,info,myid2,numprocs2;
  int i,j,k,p,nblk_m,nblk_m2,wanA,size_EVec1;
  double tmp,tmp1;

  /********************************************
   allocation of arrays 
   
   CWF1: TNum_CWFs x TNum_CWFs
   CWF2: n2 x TNum_CWFs
   CWF3: MaxN x TNum_CWFs 
   CWF4: n2 x MaxN
  ********************************************/

  if (todo_flag==1){

    /* get numprocs2 and myid2 */

    MPI_Comm_size(MPI_CommWD2[myworld2],&numprocs2);
    MPI_Comm_rank(MPI_CommWD2[myworld2],&myid2);

    /* CWF1: setting of BLACS for matrices in size of TNum_CWFs x TNum_CWFs */

    np_cols_CWF1 = (int)(sqrt((float)numprocs2));
    do{
      if((numprocs2%np_cols_CWF1)==0) break;
      np_cols_CWF1--;
    } while (np_cols_CWF1>=2);
    np_rows_CWF1 = numprocs2/np_cols_CWF1;

    /*
     For pzgesvd, it was noticed that the same 'nb' has to be used for Cs, Ws, and Vs.
     Otherwise, I encounterd an error. This is the reason why I used the same Nb as shown below.  
    */

    nblk_m = NBLK;
    while((nblk_m*np_rows_CWF1>MaxN || nblk_m*np_cols_CWF1>TNum_CWFs) && (nblk_m > 1)){ // the same as for CWF3
    //while((nblk_m*np_rows_CWF1>TNum_CWFs || nblk_m*np_cols_CWF1>TNum_CWFs) && (nblk_m > 1)){
      nblk_m /= 2;
    }
    if(nblk_m<1) nblk_m = 1;

    MPI_Allreduce(&nblk_m,&nblk_CWF1,1,MPI_INT,MPI_MIN,mpi_comm_level1);

    my_prow_CWF1 = myid2/np_cols_CWF1;
    my_pcol_CWF1 = myid2%np_cols_CWF1;

    //na_rows_CWF1 = numroc_(&TNum_CWFs, &nblk_CWF1, &my_prow_CWF1, &ZERO, &np_rows_CWF1 ); 
    na_rows_CWF1 = numroc_(&MaxN, &nblk_CWF1, &my_prow_CWF1, &ZERO, &np_rows_CWF1 );  // the same as for CWF3
    na_cols_CWF1 = numroc_(&TNum_CWFs, &nblk_CWF1, &my_pcol_CWF1, &ZERO, &np_cols_CWF1 );

    bhandle1_CWF1 = Csys2blacs_handle(MPI_CommWD2[myworld2]);
    ictxt1_CWF1 = bhandle1_CWF1;

    Cblacs_gridinit(&ictxt1_CWF1, "Row", np_rows_CWF1, np_cols_CWF1);

    MPI_Allreduce(&na_rows_CWF1,&na_rows_max_CWF1,1,MPI_INT,MPI_MAX,MPI_CommWD2[myworld2]);
    MPI_Allreduce(&na_cols_CWF1,&na_cols_max_CWF1,1,MPI_INT,MPI_MAX,MPI_CommWD2[myworld2]);

    descinit_( desc_CWF1, &TNum_CWFs, &TNum_CWFs, &nblk_CWF1, &nblk_CWF1,  
               &ZERO, &ZERO, &ictxt1_CWF1, &na_rows_CWF1,  &info); 

    /* CWF2: setting of BLACS for matrices in size of n2 x TNum_CWFs */

    np_cols_CWF2 = (int)(sqrt((float)numprocs2));
    do{
      if((numprocs2%np_cols_CWF2)==0) break;
      np_cols_CWF2--;
    } while (np_cols_CWF2>=2);
    np_rows_CWF2 = numprocs2/np_cols_CWF2;

    nblk_m = NBLK;
    while((nblk_m*np_rows_CWF2>n2 || nblk_m*np_cols_CWF2>TNum_CWFs) && (nblk_m > 1)){
      nblk_m /= 2;
    }
    if(nblk_m<1) nblk_m = 1;

    MPI_Allreduce(&nblk_m,&nblk_CWF2,1,MPI_INT,MPI_MIN,mpi_comm_level1);

    ictxt1_CWF2 = bhandle1_CWF1;

    my_prow_CWF2 = myid2/np_cols_CWF2;
    my_pcol_CWF2 = myid2%np_cols_CWF2;

    Cblacs_gridinit(&ictxt1_CWF2, "Row", np_rows_CWF2, np_cols_CWF2);

    na_rows_CWF2 = numroc_(&n2,        &nblk_CWF2, &my_prow_CWF2, &ZERO, &np_rows_CWF2 );
    na_cols_CWF2 = numroc_(&TNum_CWFs, &nblk_CWF2, &my_pcol_CWF2, &ZERO, &np_cols_CWF2 );

    MPI_Allreduce(&na_rows_CWF2, &na_rows_max_CWF2,1,MPI_INT,MPI_MAX,MPI_CommWD2[myworld2]);
    MPI_Allreduce(&na_cols_CWF2, &na_cols_max_CWF2,1,MPI_INT,MPI_MAX,MPI_CommWD2[myworld2]);

    descinit_( desc_CWF2, &n2, &TNum_CWFs, &nblk_CWF2, &nblk_CWF2,  
               &ZERO, &ZERO, &ictxt1_CWF2, &na_rows_CWF2,  &info);

    desc_CWF2[1] = desc_CWF1[1];

    /* CWF3: setting of BLACS for matrices in size of MaxN x TNum_CWFs */

    np_cols_CWF3 = (int)(sqrt((float)numprocs2));
    do{
      if((numprocs2%np_cols_CWF3)==0) break;
      np_cols_CWF3--;
    } while (np_cols_CWF3>=2);
    np_rows_CWF3 = numprocs2/np_cols_CWF3;

    nblk_m = NBLK;
    while((nblk_m*np_rows_CWF3>MaxN || nblk_m*np_cols_CWF3>TNum_CWFs) && (nblk_m > 1)){
      nblk_m /= 2;
    }
    if(nblk_m<1) nblk_m = 1;

    MPI_Allreduce(&nblk_m,&nblk_CWF3,1,MPI_INT,MPI_MIN,mpi_comm_level1);

    ictxt1_CWF3 = bhandle1_CWF1;

    my_prow_CWF3 = myid2/np_cols_CWF3;
    my_pcol_CWF3 = myid2%np_cols_CWF3;

    Cblacs_gridinit(&ictxt1_CWF3, "Row", np_rows_CWF3, np_cols_CWF3);

    na_rows_CWF3 = numroc_(&MaxN,      &nblk_CWF3, &my_prow_CWF3, &ZERO, &np_rows_CWF3 );
    na_cols_CWF3 = numroc_(&TNum_CWFs, &nblk_CWF3, &my_pcol_CWF3, &ZERO, &np_cols_CWF3 );

    MPI_Allreduce(&na_rows_CWF3, &na_rows_max_CWF3,1,MPI_INT,MPI_MAX,MPI_CommWD2[myworld2]);
    MPI_Allreduce(&na_cols_CWF3, &na_cols_max_CWF3,1,MPI_INT,MPI_MAX,MPI_CommWD2[myworld2]);

    descinit_( desc_CWF3, &MaxN, &TNum_CWFs, &nblk_CWF3, &nblk_CWF3,  
               &ZERO, &ZERO, &ictxt1_CWF3, &na_rows_CWF3,  &info);

    desc_CWF3[1] = desc_CWF1[1];

    /*
    printf("GGG1 myid2=%2d MaxN=%2d TNum_CWFs=%2d nblk_CWF3=%2d na_rows_CWF3=%2d\n",
	   myid2,MaxN,TNum_CWFs,nblk_CWF3,na_rows_CWF3);
    */

    /* CWF4: setting of BLACS for matrices in size of n2 x MaxN */

    np_cols_CWF4 = (int)(sqrt((float)numprocs2));
    do{
      if((numprocs2%np_cols_CWF4)==0) break;
      np_cols_CWF4--;
    } while (np_cols_CWF4>=2);
    np_rows_CWF4 = numprocs2/np_cols_CWF4;

    nblk_m = NBLK;
    while((nblk_m*np_rows_CWF4>n2 || nblk_m*np_cols_CWF4>MaxN) && (nblk_m > 1)){
      nblk_m /= 2;
    }
    if(nblk_m<1) nblk_m = 1;

    MPI_Allreduce(&nblk_m,&nblk_CWF4,1,MPI_INT,MPI_MIN,mpi_comm_level1);

    ictxt1_CWF4 = bhandle1_CWF1;

    my_prow_CWF4 = myid2/np_cols_CWF4;
    my_pcol_CWF4 = myid2%np_cols_CWF4;

    Cblacs_gridinit(&ictxt1_CWF4, "Row", np_rows_CWF4, np_cols_CWF4);

    na_rows_CWF4 = numroc_(&n2,   &nblk_CWF4, &my_prow_CWF4, &ZERO, &np_rows_CWF4 );
    na_cols_CWF4 = numroc_(&MaxN, &nblk_CWF4, &my_pcol_CWF4, &ZERO, &np_cols_CWF4 );

    MPI_Allreduce(&na_rows_CWF4, &na_rows_max_CWF4,1,MPI_INT,MPI_MAX,MPI_CommWD2[myworld2]);
    MPI_Allreduce(&na_cols_CWF4, &na_cols_max_CWF4,1,MPI_INT,MPI_MAX,MPI_CommWD2[myworld2]);

    descinit_( desc_CWF4, &n2, &MaxN, &nblk_CWF4, &nblk_CWF4,
               &ZERO, &ZERO, &ictxt1_CWF4, &na_rows_CWF4, &info);

    desc_CWF4[1] = desc_CWF1[1];

    /* allocation of arrays */

    *WFs = (dcomplex*)malloc(sizeof(dcomplex)*(na_rows_max_CWF2*na_cols_max_CWF2+1)); 
    for (i=0; i<(na_rows_max_CWF2*na_cols_max_CWF2+1); i++){
      (*WFs)[i] = Complex(0.0,0.0);
    }

    *Cs = (dcomplex*)malloc(sizeof(dcomplex)*(na_rows_max_CWF4*na_cols_max_CWF4+1));
    *Hs = (dcomplex*)malloc(sizeof(dcomplex)*(na_rows_max_CWF4*na_cols_max_CWF4+1));
    *Vs = (dcomplex*)malloc(sizeof(dcomplex)*(na_rows_max_CWF4*na_cols_max_CWF4+1));
    *Ws = (dcomplex*)malloc(sizeof(dcomplex)*(na_rows_max_CWF4*na_cols_max_CWF4+1));
    *EVs_PAO = (dcomplex*)malloc(sizeof(dcomplex)*(na_rows_max_CWF4*na_cols_max_CWF4+1)); 

    *Hop = (double complex****)malloc(sizeof(double complex***)*knum_i);
    for (i=0; i<knum_i; i++){
      (*Hop)[i] = (double complex***)malloc(sizeof(double complex**)*knum_j);
      for (j=0; j<knum_j; j++){
	(*Hop)[i][j] = (double complex**)malloc(sizeof(double complex*)*knum_k);
	for (k=0; k<knum_k; k++){
	  (*Hop)[i][j][k] = (double complex*)malloc(sizeof(double complex)*(na_rows_max_CWF1*na_cols_max_CWF1+1));
	  for (p=0; p<(na_rows_max_CWF1*na_cols_max_CWF1+1); p++){
	    (*Hop)[i][j][k][p] = 0.0 + 0.0*I;
	  }
	}
      }
    }

    /* save information of the memory usage */

    if (firsttime && memoryusage_fileout) {
      PrintMemory("Allocate_Free_Band_NonCol_CWF: Cs  ",sizeof(dcomplex)*na_rows_max_CWF4*na_cols_max_CWF4,NULL);
      PrintMemory("Allocate_Free_Band_NonCol_CWF: Hs  ",sizeof(dcomplex)*na_rows_max_CWF4*na_cols_max_CWF4,NULL);
      PrintMemory("Allocate_Free_Band_NonCol_CWF: Vs  ",sizeof(dcomplex)*na_rows_max_CWF4*na_cols_max_CWF4,NULL);
      PrintMemory("Allocate_Free_Band_NonCol_CWF: Ws  ",sizeof(dcomplex)*na_rows_max_CWF3*na_cols_max_CWF3,NULL);
      PrintMemory("Allocate_Free_Band_NonCol_CWF: WFs ",sizeof(dcomplex)*na_rows_max_CWF2*na_cols_max_CWF2,NULL);
      PrintMemory("Allocate_Free_Band_NonCol_CWF: EVs_PAO",sizeof(dcomplex)*na_rows_max_CWF4*na_cols_max_CWF4,NULL);
      PrintMemory("Allocate_Free_Band_NonCol_CWF: Hop",sizeof(double)*knum_i*knum_j*knum_k*na_rows_max_CWF1*na_cols_max_CWF1,NULL);
    }

    firsttime = 0;
  }

  /********************************************
               freeing of arrays 
  ********************************************/

  if (todo_flag==2){

    /* freeing for BLACS */

    free(*Cs);
    free(*Hs);
    free(*Vs);
    free(*Ws);
    free(*WFs);

    free(*EVs_PAO);

    for (i=0; i<knum_i; i++){
      for (j=0; j<knum_j; j++){
	for (k=0; k<knum_k; k++){
	  free((*Hop)[i][j][k]);
	}
	free((*Hop)[i][j]);
      }
      free((*Hop)[i]);
    }
    free((*Hop));

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
  PrintMemory("Construct_Band_Ms: My_NZeros", sizeof(int)*numprocs,NULL);
  PrintMemory("Band_DFT_NonCol: SP_NZeros", sizeof(int)*numprocs,NULL);
  PrintMemory("Band_DFT_NonCol: SP_Atoms", sizeof(int)*numprocs,NULL);
  PrintMemory("Band_DFT_NonCol: is1", sizeof(int)*numprocs,NULL);
  PrintMemory("Band_DFT_NonCol: ie1", sizeof(int)*numprocs,NULL);
  PrintMemory("Band_DFT_NonCol: is2", sizeof(int)*numprocs,NULL);
  PrintMemory("Band_DFT_NonCol: order_GA", sizeof(int)*(atomnum+2),NULL);
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



void Band_Dispersion_NonCol_CWF( int nkpath, int *n_perk, double ***kpath, char ***kname, 
				 int TNum_CWFs, double complex ****TB_Hopping, 
				 int knum_i, int knum_j, int knum_k, MPI_Comm mpi_comm_level5 ) 
{
  static int firsttime=1;
  int i,j,k,ik,l,l1,l2,l3,ll1,ll2,ll3;
  int i_perk,id;
  double time0,k1,k2,k3;
  dcomplex *Hs,*Cs;
  double ***EigenVal;
  double kRn,si,co;
  double complex ctmp;
  double TStime,TEtime, SiloopTime,EiloopTime;
  MPI_Comm mpi_comm_rows, mpi_comm_cols;
  int mpi_comm_rows_int,mpi_comm_cols_int;
  int numprocs,myid;

  char file_Band[YOUSO10];
  FILE *fp_Band;
  char buf[fp_bsize];          /* setvbuf */

  /* MPI */
  MPI_Comm_size(mpi_comm_level5,&numprocs);
  MPI_Comm_rank(mpi_comm_level5,&myid);
  MPI_Barrier(mpi_comm_level5);
  
  /* d */
  if (myid==Host_ID && 0<level_stdout){
    printf("Band_Dispersion_Col_CWF starts.\n");fflush(stdout);
  }  
  
  dtime(&TStime);
  
  /****************************************************
                 allocation of arrays
  ****************************************************/

  EigenVal = (double***)malloc(sizeof(double**)*(nkpath+1));
  for (ik=0; ik<=nkpath; ik++) {
    EigenVal[ik] = (double**)malloc(sizeof(double*)*(n_perk[ik]+1));
    for (i_perk=0; i_perk<=n_perk[ik]; i_perk++) {
      EigenVal[ik][i_perk] = (double*)malloc(sizeof(double)*(TNum_CWFs+2));
    }
  }

  Hs = (dcomplex*)malloc(sizeof(dcomplex)*(na_rows_max_CWF5*na_cols_max_CWF5+1));
  Cs = (dcomplex*)malloc(sizeof(dcomplex)*(na_rows_max_CWF5*na_cols_max_CWF5+1));

  /* show the kpaths */

  if (myid==Host_ID && 0<level_stdout){
    printf("kpath\n");
    for (ik=1;ik<=nkpath; ik++) {
      printf("%d (%18.15f %18.15f %18.15f)->(%18.15f %18.15f %18.15f)\n", ik,
              kpath[ik][1][1],kpath[ik][1][2],kpath[ik][1][3],
              kpath[ik][2][1],kpath[ik][2][2],kpath[ik][2][3]);fflush(stdout);
    }
  }

  /****************************************************
       start the calculation of band dispersion
  ****************************************************/

  dtime(&SiloopTime);

  /* setting for ELPA */

  MPI_Comm_split(mpi_comm_level5,my_pcol_CWF5,my_prow_CWF5,&mpi_comm_rows);
  MPI_Comm_split(mpi_comm_level5,my_prow_CWF5,my_pcol_CWF5,&mpi_comm_cols);
        
  mpi_comm_rows_int = MPI_Comm_c2f(mpi_comm_rows);
  mpi_comm_cols_int = MPI_Comm_c2f(mpi_comm_cols);

  /* start of the loops of ik and i_perk */

  for (ik=0; ik<=nkpath; ik++) {
    for (i_perk=0; i_perk<=n_perk[ik]; i_perk++) {

      id=1;
      k1 = kpath[ik][1][id] + (kpath[ik][2][id]-kpath[ik][1][id])*(i_perk-1)/(n_perk[ik]-1);
      id=2;
      k2 = kpath[ik][1][id] + (kpath[ik][2][id]-kpath[ik][1][id])*(i_perk-1)/(n_perk[ik]-1);
      id=3;
      k3 = kpath[ik][1][id] + (kpath[ik][2][id]-kpath[ik][1][id])*(i_perk-1)/(n_perk[ik]-1);

      /* initialize Hs */

      for (i=0; i<na_rows_CWF5*na_cols_CWF5; i++){ Hs[i].r = 0.0; Hs[i].i = 0.0; }

      /* make the Hamiltonian */

      for ( l1=-(knum_i-1)/2; l1<=(knum_i-1)/2; l1++ ){
	ll1 = l1 + (knum_i-1)/2;

	for ( l2=-(knum_j-1)/2; l2<=(knum_j-1)/2; l2++ ){
	  ll2 = l2 + (knum_j-1)/2;

	  for ( l3=-(knum_k-1)/2; l3<=(knum_k-1)/2; l3++ ){
	    ll3 = l3 + (knum_k-1)/2;

	    kRn = k1*(double)l1 + k2*(double)l2 + k3*(double)l3;
	    si = sin(2.0*PI*kRn);
	    co = cos(2.0*PI*kRn);

	    for (i=0; i<na_rows_CWF5; i++){
	      for (j=0; j<na_cols_CWF5; j++){

		ctmp = (co+I*si)*TB_Hopping[ll1][ll2][ll3][j*na_rows_CWF5+i];; 

		Hs[j*na_rows_CWF5+i].r += creal(ctmp);
		Hs[j*na_rows_CWF5+i].i += cimag(ctmp);

	      } // j
	    } // i

	  } // l3
	} // l2
      } // l1

      /* diagonalize the Hamiltonian */

      if (scf_eigen_lib_flag==1){
  
	F77_NAME(solve_evp_complex,SOLVE_EVP_COMPLEX)
          ( &TNum_CWFs, &TNum_CWFs, Hs, &na_rows_CWF5, &EigenVal[ik][i_perk][0], 
            Cs, &na_rows_CWF5, &nblk_CWF5, &mpi_comm_rows_int, &mpi_comm_cols_int );
      }   

      else if (scf_eigen_lib_flag==CuSOLVER && GPU_CPU_SWITCH_NUM<=TNum_CWFs
               && na_rows_CWF5==TNum_CWFs && na_cols_CWF5==TNum_CWFs){
	OpenMX_CuSolver_DenseZheevx_0based(Hs,Cs,&EigenVal[ik][i_perk][0],
	                                   TNum_CWFs,TNum_CWFs,
	                                   "Band_DFT_NonCol_CWF CWF-band Hamiltonian CuSOLVER");
      }

      else if (scf_eigen_lib_flag==2 || scf_eigen_lib_flag==CuSOLVER){
  
#ifndef kcomp
	int mpiworld;
	mpiworld = MPI_Comm_c2f(mpi_comm_level5);
	F77_NAME(elpa_solve_evp_complex_2stage_double_impl,ELPA_SOLVE_EVP_COMPLEX_2STAGE_DOUBLE_IMPL)
	  ( &TNum_CWFs, &TNum_CWFs, Hs, &na_rows_CWF5, &EigenVal[ik][i_perk][0], Cs, 
	    &na_rows_CWF5, &nblk_CWF5, &na_cols_CWF5, 
	    &mpi_comm_rows_int, &mpi_comm_cols_int, &mpiworld );
#endif

      } 

    } // i_perk
  } // ik

  /* freeing of the setting for ELPA */

  MPI_Comm_free(&mpi_comm_rows);
  MPI_Comm_free(&mpi_comm_cols);

  /****************************************************
                    write a file 
  ****************************************************/

  if (myid==Host_ID) {

    strcpy(file_Band,".CWF.Band");
    fnjoint(filepath,filename,file_Band);  

    if ((fp_Band = fopen(file_Band,"w"))==NULL) {

#ifdef xt3
      setvbuf(fp_Band,buf,_IOFBF,fp_bsize);  /* setvbuf */
#endif

      printf("<Band_DFT_kpath> can not open a file (%s)\n",file_Band);
      return;
    }

    fprintf(fp_Band," %d  %d  %18.15f\n",TNum_CWFs,0,ChemP);  /* set SpinP_switch==0 */
    for (i=1;i<=3;i++) 
      for (j=1;j<=3;j++) {
	fprintf(fp_Band,"%18.15f ", rtv[i][j]);
      }
    fprintf(fp_Band,"\n");
    fprintf(fp_Band,"%d\n",nkpath);
    for (i=1;i<=nkpath;i++) {
      fprintf(fp_Band,"%d %18.15f %18.15f %18.15f  %18.15f %18.15f %18.15f  %s %s\n",
	      n_perk[i],
	      kpath[i][1][1], kpath[i][1][2], kpath[i][1][3],
	      kpath[i][2][1], kpath[i][2][2], kpath[i][2][3],
	      kname[i][1],kname[i][2]);
    }

    for (ik=1; ik<=nkpath; ik++) {
      for (i_perk=1; i_perk<=n_perk[ik]; i_perk++) {

        id=1;
        k1 = kpath[ik][1][id]+
          (kpath[ik][2][id]-kpath[ik][1][id])*(i_perk-1)/(n_perk[ik]-1);
        id=2;
        k2 = kpath[ik][1][id]+
          (kpath[ik][2][id]-kpath[ik][1][id])*(i_perk-1)/(n_perk[ik]-1);
        id=3;
        k3 = kpath[ik][1][id]+
          (kpath[ik][2][id]-kpath[ik][1][id])*(i_perk-1)/(n_perk[ik]-1);

	fprintf(fp_Band,"%d %18.15f %18.15f %18.15f\n", TNum_CWFs,k1,k2,k3);

	for (l=0; l<TNum_CWFs; l++) {
	  fprintf(fp_Band,"%18.15f ",EigenVal[ik][i_perk][l]);
	}
	fprintf(fp_Band  ,"\n");
      }
    }

    fclose(fp_Band);

  } /* if (myid==Host_ID) */
  
  /****************************************************
                  freeing of arrays
  ****************************************************/

  for (ik=0; ik<=nkpath; ik++) {
    for (i_perk=0; i_perk<=n_perk[ik]; i_perk++) {
      free(EigenVal[ik][i_perk]);
    }
    free(EigenVal[ik]);
  }
  free(EigenVal);

  free(Hs);
  free(Cs);

  /* elapsed time */
  dtime(&TEtime);
}

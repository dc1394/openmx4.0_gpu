#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "mpi.h"
#include "openmx_common.h"
#include "openmx_env.h"
#include "openmx_mixing_v_workspace.h"
#include "lapack_prototypes.h"
#include "tran_variables.h"
#include <omp.h>

#define  maxima_step  300000.0
#define  measure_time   0

#ifdef MIN
#undef MIN
#endif
#define MIN(a,b) ((a)<(b))?  (a):(b)

static void Kerker_Mixing_VKSk(OpenMXSolverInvocation *scratch, MixingVState *state,  int Change_switch, double Mix_wgt );
static void DIIS_Mixing_VKSk(OpenMXSolverInvocation *scratch, MixingVState *state, int SCF_iter, double Mix_wgt );
static void Inverse(OpenMXSolverInvocation *scratch, int n, const OpenMXSolverRealTensor *a, const OpenMXSolverRealTensor *ia);
static void Complex_Inverse(OpenMXSolverInvocation *scratch, int n, const OpenMXSolverRealTensor *a, const OpenMXSolverRealTensor *ia, const OpenMXSolverRealTensor *b, const OpenMXSolverRealTensor *ib);

void F77_NAME(dgemv,DGEMV)(const char *trans, const int *m, const int *n,
                           const double *alpha, const double *a, const int *lda,
                           const double *x, const int *incx, const double *beta,
                           double *y, const int *incy);

/* Fast path of the KS-potential DIIS: same construction as in
   DIIS_Mixing_Rhok.c (incremental Gram row + slot-pointer rotation +
   memmove residual shift), active under scf.eigen.lib gpusolver and
   disabled with OPENMX_MIX_FAST=0. */

static int MixV_Fast_Enabled(void)
{
  if (scf_eigen_lib_flag != GPUSOLVER) return 0;
  return openmx_env_flag_at(OPENMX_ENV_MIX_FAST, 1);
}

static void MixVKSk_Fast_Invalidate(MixingVState *state)
{
  state->fast.valid = 0;
  state->fast.last_scf_iter = -1;
}

/* Keep the existing free/allocate/layout-update order. The calculation
   resource group also owns partial initialization and final cleanup. */
static int MixVKSk_FastResize(MixingVState *state,int history,int spinmax)
{
  openmx_core_native_free(state->resources,state->fast.A);
  state->fast.A=(double*)openmx_core_native_malloc(state->resources,1,
      sizeof(double)*(size_t)history*(size_t)history,0);
  if (!state->fast.A) return 0;
  state->fast.l38=history;
  state->fast.spinmax=spinmax;
  MixVKSk_Fast_Invalidate(state);
  return 1;
}

static void MixVKSk_RotateSlots(double ***arr, int nslot)
{
  double **tmp = arr[nslot-1];
  int m;
  for (m=nslot-1; 0<m; m--) arr[m] = arr[m-1];
  arr[0] = tmp;
}

static double Mixing_V_Compute(OpenMXSolverInvocation *scratch, MixingVState *state,  int MD_iter,
                 int SCF_iter,
                 int SCF_iter0 );

static int MixV_ScratchOperation(OpenMXSolverInvocation *scratch,void *opaque,double *result)
{
  const MixV_ScratchArguments *arguments=(const MixV_ScratchArguments*)opaque;
  *result=Mixing_V_Compute(scratch,arguments->state,arguments->md_iteration,
      arguments->scf_iteration,arguments->first_iteration);
  return 1;
}

double Mixing_V(OpenMXMixingVState *state,int MD_iter,int SCF_iter,int SCF_iter0)
{
  int numprocs,myid;
  double TStime,TEtime;
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);
  MPI_Barrier(mpi_comm_level1);
  dtime(&TStime);
  MixV_ScratchArguments arguments={state,MD_iter,SCF_iter,SCF_iter0};
  (void)OpenMX_Solver_Run(MixV_ScratchOperation,&arguments);
  MPI_Barrier(mpi_comm_level1);
  dtime(&TEtime);
  return TEtime-TStime;
}

static double Mixing_V_Compute(OpenMXSolverInvocation *scratch, MixingVState *state,  int MD_iter,
		 int SCF_iter,
		 int SCF_iter0 )
{
  int pSCF_iter,NumMix,NumSlide;
  int spin,ct_AN,wan1,TNO1,h_AN,Gh_AN,wan2;
  int TNO2,i,j,ian,jan,m,n;
  int spinmax,k1,k2,k3;
  int Mc_AN,Gc_AN;
  double time0;
  double TStime,TEtime;
  int numprocs,myid,ID;

  /*********************************************************
   RMM-DIIS mixing with Kerker's weighting factor in k-space
   for the Kohn-Sham potential
  *********************************************************/

  if (SCF_iter==1){
    Kerker_Mixing_VKSk(scratch, state, 0,Mixing_weight);
  }

  else if (SCF_iter<Pulay_SCF){
    Kerker_Mixing_VKSk(scratch, state, 1,Mixing_weight);
  }

  else{

    if (6<SCF_iter){
      DIIS_Mixing_VKSk(scratch, state, SCF_iter-(Pulay_SCF-3),Mixing_weight);
    }
    else{
      DIIS_Mixing_VKSk(scratch, state, SCF_iter,Mixing_weight);
    }
  }

  /* if SCF_iter0==1, then NormRD[0]=1 */

  if (SCF_iter0==1) NormRD[0] = 1.0;

  return 0.0;
} 


void DIIS_Mixing_VKSk(OpenMXSolverInvocation *scratch, MixingVState *state, int SCF_iter, double Mix_wgt)
{
  int spin,Mc_AN,Gc_AN,wan1,TNO1,h_AN,Gh_AN;
  int wan2,TNO2,i,j,k,NumMix,NumSlide;
  int SCFi,SCFj,tno1,tno0,Cwan,Hwan,k1,k2,k3,kk2; 
  int pSCF_iter,size_Re_OptVKSk,size_Kerker_weight;
  int spinmax,MN,flag_nan,refiter,po3,po4,refiter2;
  int N2D,GN,GNs,N3[4],BN_AB,p0,p1,M,N,K;
  double time1,time2,time3,time4,time5; 
  double Stime1,Etime1,scale;
  double *alden,*alden0,*gradF;
  OpenMXSolverRealTensor A,IA;
  double *My_NMat,*NMat;
  double alpha,beta;
  double tmp0,itmp0,tmp1;
  double im1,im2,re1,re2; 
  double OptRNorm,My_OptRNorm,sum0,sum1;
  double dum1,dum2,bunsi,bunbo,sum,coef_OptVKS;
  double Gx,Gy,Gz,G2,G12,G22,G32,F,F0;
  double Min_Weight;
  double Max_Weight;
  double G0,G02,G02p,weight,wgt0,wgt1;
  double sk1,sk2,sk3;
  OpenMXSolverRealTensor Re_OptVKSk;
  OpenMXSolverRealTensor Im_OptVKSk;
  double *Kerker_weight;
  int numprocs,myid;
  char nanchar[300];
  /* for OpenMP */
  int OMPID,Nthrds,Nprocs,Nloop,Nthrds0;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /* find an optimum G0 */

  G12 = rtv[1][1]*rtv[1][1] + rtv[1][2]*rtv[1][2] + rtv[1][3]*rtv[1][3]; 
  G22 = rtv[2][1]*rtv[2][1] + rtv[2][2]*rtv[2][2] + rtv[2][3]*rtv[2][3]; 
  G32 = rtv[3][1]*rtv[3][1] + rtv[3][2]*rtv[3][2] + rtv[3][3]*rtv[3][3]; 

  if (G12<G22) G0 = G12;
  else         G0 = G22;
  if (G32<G0)  G0 = G32;

  G0 = sqrt(G0);
  G02 = Kerker_factor*Kerker_factor*G0*G0;
  G02p = (0.1*Kerker_factor*G0)*(0.1*Kerker_factor*G0);

  if      (SpinP_switch==0)  spinmax = 1;
  else if (SpinP_switch==1)  spinmax = 2;
  else if (SpinP_switch==3)  spinmax = 3;

  /****************************************************
    allocation of arrays:
  ****************************************************/
  
  alden = OpenMX_Solver_OverwriteReal(scratch,"mixv-DIIS_Mixing_VKSk-alden",List_YOUSO[38]);
  alden0 = OpenMX_Solver_OverwriteReal(scratch,"mixv-DIIS_Mixing_VKSk-alden0",List_YOUSO[38]);
  gradF = OpenMX_Solver_OverwriteReal(scratch,"mixv-DIIS_Mixing_VKSk-gradF",List_YOUSO[38]);
  
  My_NMat = OpenMX_Solver_OverwriteReal(scratch,"mixv-DIIS_Mixing_VKSk-My_NMat",(size_t)List_YOUSO[38]*(size_t)List_YOUSO[38]);
  NMat = OpenMX_Solver_OverwriteReal(scratch,"mixv-DIIS_Mixing_VKSk-NMat",(size_t)List_YOUSO[38]*(size_t)List_YOUSO[38]);

  A = OpenMX_Solver_OverwriteView(scratch,"mixv-DIIS_Mixing_VKSk-A",List_YOUSO[38],List_YOUSO[38]);
  
  IA = OpenMX_Solver_OverwriteView(scratch,"mixv-DIIS_Mixing_VKSk-IA",List_YOUSO[38],List_YOUSO[38]);

  /* Re_OptVKSk and Im_OptVKSk */

  size_Re_OptVKSk = spinmax*My_NumGridB_CB;

  /* The inverse FFT needs the full real-space capacity. */
  Re_OptVKSk = OpenMX_Solver_OverwriteView(scratch,"mixv-DIIS_Mixing_VKSk-Re_OptVKSk",spinmax,My_Max_NumGridB);

  /* The inverse FFT needs the full real-space capacity. */
  Im_OptVKSk = OpenMX_Solver_OverwriteView(scratch,"mixv-DIIS_Mixing_VKSk-Im_OptVKSk",spinmax,My_Max_NumGridB);

  size_Kerker_weight = My_NumGridB_CB;
  Kerker_weight = OpenMX_Solver_OverwriteReal(scratch,"mixv-DIIS_Mixing_VKSk-Kerker_weight",My_NumGridB_CB);

  /***********************************
            set Kerker_weight 
  ************************************/

  if (measure_time==1) dtime(&Stime1);

  N2D = Ngrid3*Ngrid2;
  GNs = ((myid*N2D+numprocs-1)/numprocs)*Ngrid1;

  for (k=0; k<My_NumGridB_CB; k++){

    /* get k3, k2, and k1 */

    GN = k + GNs;     
    k3 = GN/(Ngrid2*Ngrid1);    
    k2 = (GN - k3*Ngrid2*Ngrid1)/Ngrid1;
    k1 = GN - k3*Ngrid2*Ngrid1 - k2*Ngrid1; 

    if (k1<Ngrid1/2) sk1 = (double)k1;
    else             sk1 = (double)(k1 - Ngrid1);

    if (k2<Ngrid2/2) sk2 = (double)k2;
    else             sk2 = (double)(k2 - Ngrid2);

    if (k3<Ngrid3/2) sk3 = (double)k3;
    else             sk3 = (double)(k3 - Ngrid3);

    Gx = sk1*rtv[1][1] + sk2*rtv[2][1] + sk3*rtv[3][1];
    Gy = sk1*rtv[1][2] + sk2*rtv[2][2] + sk3*rtv[3][2]; 
    Gz = sk1*rtv[1][3] + sk2*rtv[2][3] + sk3*rtv[3][3];
    G2 = Gx*Gx + Gy*Gy + Gz*Gz;

    if (k1==0 && k2==0 && k3==0)  weight = 1.0;
    else                          weight = (G2 + G02)/(G2 + G02p);

    Kerker_weight[k] = sqrt(weight);

  } /* k */

  if (measure_time==1){ 
    dtime(&Etime1);
    time1 = Etime1 - Stime1;      
  }

  /****************************************************
     start calc.
  ****************************************************/
 
  if (SCF_iter==1){
    /* memory calc. done, only 1st iteration */
    if (state->diis_first) {
      PrintMemory("DIIS_Mixing_VKSk: Re_OptVKSk",sizeof(double)*size_Re_OptVKSk,NULL);
      PrintMemory("DIIS_Mixing_VKSk: Im_OptVKSk",sizeof(double)*size_Re_OptVKSk,NULL);
      PrintMemory("DIIS_Mixing_VKSk: Kerker_weight",sizeof(double)*size_Kerker_weight,NULL);
      state->diis_first=0;
    }

    Kerker_Mixing_VKSk(scratch, state, 0,1.0);
  } 

  else if (SCF_iter==2){

    if (measure_time==1) dtime(&Stime1);

    /* construct residual rho */

    for (spin=0; spin<spinmax; spin++){
      for (k=0; k<My_NumGridB_CB; k++){

	Residual_ReVKSk[spin][2*My_NumGridB_CB+k] = (ReVKSk[0][spin][k] - ReVKSk[1][spin][k])*Kerker_weight[k];
	Residual_ImVKSk[spin][2*My_NumGridB_CB+k] = (ImVKSk[0][spin][k] - ImVKSk[1][spin][k])*Kerker_weight[k];

      } /* k */
    } /* spin */

    if (measure_time==1){ 
      dtime(&Etime1);
      time2 = Etime1 - Stime1;      
    }

    /* rho1 to rho2 */

    for (spin=0; spin<spinmax; spin++){
      for (k=0; k<My_NumGridB_CB; k++){
	ReVKSk[2][spin][k] = ReVKSk[1][spin][k];
	ImVKSk[2][spin][k] = ImVKSk[1][spin][k];
      }
    }

    Kerker_Mixing_VKSk(scratch, state, 0,Mixing_weight);
  } 

  else {

    /****************************************************
             construct residual KS potential 
    ****************************************************/

    if (measure_time==1) dtime(&Stime1);

    for (spin=0; spin<spinmax; spin++){
      for (k=0; k<My_NumGridB_CB; k++){

	Residual_ReVKSk[spin][My_NumGridB_CB+k] = (ReVKSk[0][spin][k] - ReVKSk[1][spin][k])*Kerker_weight[k];
	Residual_ImVKSk[spin][My_NumGridB_CB+k] = (ImVKSk[0][spin][k] - ImVKSk[1][spin][k])*Kerker_weight[k];
      } /* k */
    } /* spin */

    if (measure_time==1){ 
      dtime(&Etime1);
      time3 = Etime1 - Stime1;      
    }

    if ((SCF_iter-1)<Num_Mixing_pDM){
      NumMix   = SCF_iter - 1;
      NumSlide = NumMix + 1;
    }
    else{
      NumMix   = Num_Mixing_pDM;
      NumSlide = NumMix;
    }

    /****************************************************
              alpha from residual KS potentials
    ****************************************************/

    if (measure_time==1) dtime(&Stime1);

    /* calculation of the norm matrix for the residual vectors */

    {
      int use_fast = MixV_Fast_Enabled();
      int incremental = 0;

      if (use_fast){
        if (state->fast.A == NULL || state->fast.l38 != List_YOUSO[38] ||
            state->fast.spinmax != spinmax){
          if (!MixVKSk_FastResize(state,List_YOUSO[38],spinmax)){
            printf("Mixing_V: failed to allocate the Gram cache.\n");
            MPI_Abort(mpi_comm_level1,1);
          }
        }
        incremental = (state->fast.valid &&
                       state->fast.last_scf_iter+1 == SCF_iter &&
                       (NumMix == state->fast.last_nummix ||
                        NumMix == state->fast.last_nummix+1));
      }

      if (incremental){

        int l38 = state->fast.l38;
        int inc1 = 1;
        double *C = state->fast.A;
        double *my_row = My_NMat;   /* scratch */
        double *row = NMat;

        for (i=NumMix-1; 1<=i; i--){
          for (j=NumMix-1; 1<=j; j--){
            C[i*l38+j] = C[(i-1)*l38+(j-1)];
          }
        }

        M = NumMix + 1;
        K = My_NumGridB_CB;
        alpha = 1.0;
        beta = 1.0;
        for (i=0; i<M; i++) my_row[i] = 0.0;

        for (spin=0; spin<spinmax; spin++){
          F77_NAME(dgemv,DGEMV)( "T", &K, &M, &alpha,
                                 Residual_ReVKSk[spin], &K,
                                 &Residual_ReVKSk[spin][My_NumGridB_CB], &inc1,
                                 &beta, my_row, &inc1);
          F77_NAME(dgemv,DGEMV)( "T", &K, &M, &alpha,
                                 Residual_ImVKSk[spin], &K,
                                 &Residual_ImVKSk[spin][My_NumGridB_CB], &inc1,
                                 &beta, my_row, &inc1);
        }

        MPI_Allreduce(my_row, row, M, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

        for (j=0; j<NumMix; j++){
          C[j] = row[j+1];
          C[j*l38] = row[j+1];
        }

        for (i=0; i<NumMix; i++){
          for (j=0; j<NumMix; j++){
            openmx_solver_real_matrix_row(&A,i)[j] = C[i*l38+j];
          }
        }
      }

      else{

        for (i=0; i<List_YOUSO[38]*List_YOUSO[38]; i++) My_NMat[i] = 0.0;

        for (spin=0; spin<spinmax; spin++){

          M = NumMix + 1;
          N = NumMix + 1;
          K = My_NumGridB_CB;
          alpha = 1.0;
          beta = 1.0;

          F77_NAME(dgemm,DGEMM)( "T","N", &M, &N, &K,
                                 &alpha,
                                 Residual_ReVKSk[spin], &K,
                                 Residual_ReVKSk[spin], &K,
                                 &beta,
                                 My_NMat,
                                 &M);

          F77_NAME(dgemm,DGEMM)( "T","N", &M, &N, &K,
                                 &alpha,
                                 Residual_ImVKSk[spin], &K,
                                 Residual_ImVKSk[spin], &K,
                                 &beta,
                                 My_NMat,
                                 &M);
        }

        MPI_Allreduce(My_NMat, NMat, (NumMix+1)*(NumMix+1),
                      MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

        for (i=0; i<NumMix; i++){
          for (j=0; j<NumMix; j++){
            openmx_solver_real_matrix_row(&A,i)[j] = NMat[(i+1)*(NumMix+1)+(j+1)];
          }
        }

        if (use_fast){
          int l38 = state->fast.l38;
          for (i=0; i<NumMix; i++){
            for (j=0; j<NumMix; j++){
              state->fast.A[i*l38+j] = openmx_solver_real_matrix_row(&A,i)[j];
            }
          }
        }
      }

      if (use_fast){
        state->fast.valid = 1;
        state->fast.last_scf_iter = SCF_iter;
        state->fast.last_nummix = NumMix;
      }
      else{
        MixVKSk_Fast_Invalidate(state);
      }
    }

    if (measure_time==1){ 
      dtime(&Etime1);
      time4 = Etime1 - Stime1;      
    }

    /* store NormRD */

    for (i=4; 1<=i; i--){
      NormRD[i] = NormRD[i-1];
      History_Uele[i] = History_Uele[i-1];
    }
    NormRD[0] = openmx_solver_real_matrix_row(&A,0)[0]/(double)(1.0e+5*Ngrid1*Ngrid2*Ngrid3*spinmax);
    History_Uele[0] = Uele;

    /* set matrix elements on lammda */

    for (SCFi=1; SCFi<=NumMix; SCFi++){
       openmx_solver_real_matrix_row(&A,SCFi-1)[NumMix] = -1.0;
       openmx_solver_real_matrix_row(&A,NumMix)[SCFi-1] = -1.0;
    }
    openmx_solver_real_matrix_row(&A,NumMix)[NumMix] = 0.0;

    /* solve the linear equation */

    Inverse(scratch, NumMix,&A,&IA);

    for (SCFi=1; SCFi<=NumMix; SCFi++){
       alden[SCFi] = -openmx_solver_real_matrix_row(&IA,SCFi-1)[NumMix];
    }

    /*************************************
      check "nan", "NaN", "inf" or "Inf"
    *************************************/

    flag_nan = 0;
    for (SCFi=1; SCFi<=NumMix; SCFi++){

      sprintf(nanchar,"%8.4f",alden[SCFi]);
      if (strstr(nanchar,"nan")!=NULL || strstr(nanchar,"NaN")!=NULL 
       || strstr(nanchar,"inf")!=NULL || strstr(nanchar,"Inf")!=NULL){

        flag_nan = 1;
      }
    }

    if (flag_nan==1){
      for (SCFi=1; SCFi<=NumMix; SCFi++){
        alden[SCFi] = 0.0;
      }
      alden[1] = 0.1;
      alden[2] = 0.9;
    }

    /****************************************************
              calculate optimized residual rho
    ****************************************************/

    if (measure_time==1) dtime(&Stime1);

    for (spin=0; spin<spinmax; spin++){
      for (k=0; k<My_NumGridB_CB; k++){
        openmx_solver_real_matrix_row(&Re_OptVKSk,spin)[k] = 0.0;
        openmx_solver_real_matrix_row(&Im_OptVKSk,spin)[k] = 0.0;
      }
    }   

    /* Pulay mixing for residual VKSk */

    for (pSCF_iter=1; pSCF_iter<=NumMix; pSCF_iter++){

      tmp0 = alden[pSCF_iter]; 
      p0 = pSCF_iter*My_NumGridB_CB;

#pragma omp parallel shared(spinmax,My_NumGridB_CB,Re_OptVKSk,Im_OptVKSk,tmp0,p0,Residual_ReVKSk,Residual_ImVKSk)
      {

        int OMPID,Nthrds,k; 

	OMPID = omp_get_thread_num();
	Nthrds = omp_get_num_threads();

	if (spinmax==1){
	  for (k=My_NumGridB_CB*OMPID/Nthrds; k<My_NumGridB_CB*(OMPID+1)/Nthrds; k++){
            openmx_solver_real_matrix_row(&Re_OptVKSk,0)[k] += tmp0*Residual_ReVKSk[0][p0+k];
            openmx_solver_real_matrix_row(&Im_OptVKSk,0)[k] += tmp0*Residual_ImVKSk[0][p0+k];
	  } /* k */
	}

	else if (spinmax==2){
	  for (k=My_NumGridB_CB*OMPID/Nthrds; k<My_NumGridB_CB*(OMPID+1)/Nthrds; k++){
            openmx_solver_real_matrix_row(&Re_OptVKSk,0)[k] += tmp0*Residual_ReVKSk[0][p0+k];
            openmx_solver_real_matrix_row(&Im_OptVKSk,0)[k] += tmp0*Residual_ImVKSk[0][p0+k];
            openmx_solver_real_matrix_row(&Re_OptVKSk,1)[k] += tmp0*Residual_ReVKSk[1][p0+k];
            openmx_solver_real_matrix_row(&Im_OptVKSk,1)[k] += tmp0*Residual_ImVKSk[1][p0+k];
	  } /* k */
	}

	else if (spinmax==3){
	  for (k=My_NumGridB_CB*OMPID/Nthrds; k<My_NumGridB_CB*(OMPID+1)/Nthrds; k++){
            openmx_solver_real_matrix_row(&Re_OptVKSk,0)[k] += tmp0*Residual_ReVKSk[0][p0+k];
            openmx_solver_real_matrix_row(&Im_OptVKSk,0)[k] += tmp0*Residual_ImVKSk[0][p0+k];
            openmx_solver_real_matrix_row(&Re_OptVKSk,1)[k] += tmp0*Residual_ReVKSk[1][p0+k];
            openmx_solver_real_matrix_row(&Im_OptVKSk,1)[k] += tmp0*Residual_ImVKSk[1][p0+k];
            openmx_solver_real_matrix_row(&Re_OptVKSk,2)[k] += tmp0*Residual_ReVKSk[2][p0+k];
            openmx_solver_real_matrix_row(&Im_OptVKSk,2)[k] += tmp0*Residual_ImVKSk[2][p0+k];
	  } /* k */
	}

      } /* #pragma omp parallel */

    } /* pSCF_iter */

    if      (1.0e-2<=NormRD[0])                        coef_OptVKS = 0.5;
    else if (1.0e-4<=NormRD[0]  && NormRD[0]<1.0e-2)   coef_OptVKS = 0.6;
    else if (1.0e-6<=NormRD[0]  && NormRD[0]<1.0e-4)   coef_OptVKS = 0.7;
    else if (1.0e-8<=NormRD[0]  && NormRD[0]<1.0e-6)   coef_OptVKS = 0.8;
    else if (1.0e-10<=NormRD[0] && NormRD[0]<1.0e-8)   coef_OptVKS = 0.9;
    else                                               coef_OptVKS = 1.0;

    /****************************************************
                       mixing of VKS
    ****************************************************/

    /****************************************************
       Pulay mixing
    ****************************************************/

    if ( SCF_iter%EveryPulay_SCF==0 ) {

      /* reduce Mixing_weight so that charge sloshing can be avoided at the next SCF */

      Mixing_weight = 0.1*Max_Mixing_weight;
      if (Mixing_weight<Min_Mixing_weight) Mixing_weight = Min_Mixing_weight;
      
      /* initial ReVKSk and ImVKSk */
      
      for (spin=0; spin<spinmax; spin++){
        for (k=0; k<My_NumGridB_CB; k++){
          ReVKSk[0][spin][k] = 0.0;
	  ImVKSk[0][spin][k] = 0.0;
	}
      }

      /* Pulay mixing */

      for (pSCF_iter=1; pSCF_iter<=NumMix; pSCF_iter++){

	tmp0 =  alden[pSCF_iter];

#pragma omp parallel shared(tmp0,pSCF_iter,spinmax,My_NumGridB_CB,ReVKSk,ImVKSk) 
	{

          int OMPID,Nthrds,k; 

	  OMPID = omp_get_thread_num();
	  Nthrds = omp_get_num_threads();

	  if (spinmax==1){
  	    for (k=My_NumGridB_CB*OMPID/Nthrds; k<My_NumGridB_CB*(OMPID+1)/Nthrds; k++){
	      ReVKSk[0][0][k] += tmp0*ReVKSk[pSCF_iter][0][k];
	      ImVKSk[0][0][k] += tmp0*ImVKSk[pSCF_iter][0][k];
	    } /* k */
	  }

	  else if (spinmax==2){
  	    for (k=My_NumGridB_CB*OMPID/Nthrds; k<My_NumGridB_CB*(OMPID+1)/Nthrds; k++){
	      ReVKSk[0][0][k] += tmp0*ReVKSk[pSCF_iter][0][k];
	      ImVKSk[0][0][k] += tmp0*ImVKSk[pSCF_iter][0][k];
	      ReVKSk[0][1][k] += tmp0*ReVKSk[pSCF_iter][1][k];
	      ImVKSk[0][1][k] += tmp0*ImVKSk[pSCF_iter][1][k];
	    } /* k */
	  }

	  else if (spinmax==3){
  	    for (k=My_NumGridB_CB*OMPID/Nthrds; k<My_NumGridB_CB*(OMPID+1)/Nthrds; k++){
	      ReVKSk[0][0][k] += tmp0*ReVKSk[pSCF_iter][0][k];
	      ImVKSk[0][0][k] += tmp0*ImVKSk[pSCF_iter][0][k];
	      ReVKSk[0][1][k] += tmp0*ReVKSk[pSCF_iter][1][k];
	      ImVKSk[0][1][k] += tmp0*ImVKSk[pSCF_iter][1][k];
	      ReVKSk[0][2][k] += tmp0*ReVKSk[pSCF_iter][2][k];
	      ImVKSk[0][2][k] += tmp0*ImVKSk[pSCF_iter][2][k];
	    } /* k */
	  }

	} /* #pragma omp parallel */

      } /* pSCF_iter */

      /* Correction by optimized residual VKS */

      for (spin=0; spin<spinmax; spin++){

        for (k=0; k<My_NumGridB_CB; k++){

 	  if (1.0<NormRD[0])
  	    weight = 1.0/(Kerker_weight[k]*Kerker_weight[k]);
          else 
            weight = 1.0/Kerker_weight[k];

          ReVKSk[0][spin][k] += coef_OptVKS*openmx_solver_real_matrix_row(&Re_OptVKSk,spin)[k]*weight;
          ImVKSk[0][spin][k] += coef_OptVKS*openmx_solver_real_matrix_row(&Im_OptVKSk,spin)[k]*weight;

	  /* correction to large changing components */

          tmp0 = ReVKSk[0][spin][k] - ReVKSk[1][spin][k];  
          tmp1 = ImVKSk[0][spin][k] - ImVKSk[1][spin][k];  

          if ( maxima_step<(fabs(tmp0)+fabs(tmp1)) ){
            ReVKSk[0][spin][k] = sgn(tmp0)*maxima_step + ReVKSk[1][spin][k]; 
            ImVKSk[0][spin][k] = sgn(tmp1)*maxima_step + ImVKSk[1][spin][k];
          }

	} /* k */
      } /* spin */

      if (measure_time==1){ 
        dtime(&Etime1);
        time5 = Etime1 - Stime1;      
      }

    } /* end of if ( SCF_iter%EveryPulay_SCF==0 ) */

    /****************************************************
       Kerker mixing
    ****************************************************/

    else {

      /* find an optimum mixing weight */

      Min_Weight = Min_Mixing_weight;
      Max_Weight = Max_Mixing_weight;

      if ((int)sgn(History_Uele[0]-History_Uele[1])
	  ==(int)sgn(History_Uele[1]-History_Uele[2])
	  && NormRD[0]<NormRD[1]){

	/* tmp0 = 1.6*Mixing_weight; */

	tmp0 = NormRD[1]/(largest(NormRD[1]-NormRD[0],10e-10))*Mixing_weight;

	if (tmp0<Max_Weight){
	  if (Min_Weight<tmp0){
	    Mixing_weight = tmp0;
	  }
	  else{ 
	    Mixing_weight = Min_Weight;
	  }
	}
	else{ 
	  Mixing_weight = Max_Weight;
	  SCF_RENZOKU++;  
	}
      }
   
      else if ((int)sgn(History_Uele[0]-History_Uele[1])
	       ==(int)sgn(History_Uele[1]-History_Uele[2])
	       && NormRD[1]<NormRD[0]){

	tmp0 = NormRD[1]/(largest(NormRD[1]+NormRD[0],10e-10))*Mixing_weight;

	/* tmp0 = Mixing_weight/1.6; */

	if (tmp0<Max_Weight){
	  if (Min_Weight<tmp0)
	    Mixing_weight = tmp0;
	  else 
	    Mixing_weight = Min_Weight;
	}
	else 
	  Mixing_weight = Max_Weight;

	SCF_RENZOKU = -1;  
      }

      else if ((int)sgn(History_Uele[0]-History_Uele[1])
	       !=(int)sgn(History_Uele[1]-History_Uele[2])
	       && NormRD[0]<NormRD[1]){

	/* tmp0 = Mixing_weight*1.2; */

	tmp0 = NormRD[1]/(largest(NormRD[1]-NormRD[0],10e-10))*Mixing_weight;

	if (tmp0<Max_Weight){
	  if (Min_Weight<tmp0)
	    Mixing_weight = tmp0;
	  else 
	    Mixing_weight = Min_Weight;
	}
	else{ 
	  Mixing_weight = Max_Weight;
	  SCF_RENZOKU++;
	}
      }

      else if ((int)sgn(History_Uele[0]-History_Uele[1])
	       !=(int)sgn(History_Uele[1]-History_Uele[2])
	       && NormRD[1]<NormRD[0]){

	/* tmp0 = Mixing_weight/2.0; */

	tmp0 = NormRD[1]/(largest(NormRD[1]+NormRD[0],10e-10))*Mixing_weight;

	if (tmp0<Max_Weight){
	  if (Min_Weight<tmp0)
	    Mixing_weight = tmp0;
	  else 
	    Mixing_weight = Min_Weight;
	}
	else 
	  Mixing_weight = Max_Weight;

	SCF_RENZOKU = -1;
      }

      Mix_wgt = Mixing_weight;

      for (spin=0; spin<spinmax; spin++){
        for (k=0; k<My_NumGridB_CB; k++){
         
	  weight = 1.0/(Kerker_weight[k]*Kerker_weight[k]);
	  wgt0  = Mix_wgt*weight;
	  wgt1 =  1.0 - wgt0;
        
	  ReVKSk[0][spin][k] = wgt0*ReVKSk[0][spin][k] + wgt1*ReVKSk[1][spin][k];
	  ImVKSk[0][spin][k] = wgt0*ImVKSk[0][spin][k] + wgt1*ImVKSk[1][spin][k];
        
	} /* k */
      } /* spin */
    } /* else */

    /****************************************************
                         shift of VKS
    ****************************************************/

    if (MixV_Fast_Enabled()){

      /* pointer rotation == the legacy content shift; the legacy shift
         leaves slot 0 in place (the tail below reads it), so restore its
         content from the rotated slot 1 with one slot copy */

      MixVKSk_RotateSlots(ReVKSk, List_YOUSO[38]);
      MixVKSk_RotateSlots(ImVKSk, List_YOUSO[38]);
      for (spin=0; spin<spinmax; spin++){
        memcpy(ReVKSk[0][spin], ReVKSk[1][spin],
               sizeof(double)*(size_t)My_NumGridB_CB);
        memcpy(ImVKSk[0][spin], ImVKSk[1][spin],
               sizeof(double)*(size_t)My_NumGridB_CB);
      }
    }
    else{

      for (pSCF_iter=(List_YOUSO[38]-1); 0<pSCF_iter; pSCF_iter--){

        for (spin=0; spin<spinmax; spin++){
          for (k=0; k<My_NumGridB_CB; k++){
            ReVKSk[pSCF_iter][spin][k] = ReVKSk[pSCF_iter-1][spin][k];
            ImVKSk[pSCF_iter][spin][k] = ImVKSk[pSCF_iter-1][spin][k];
          }
        }
      }
    }

    /****************************************************
                    shift of residual rho
    ****************************************************/

    if (MixV_Fast_Enabled()){

      size_t nb = sizeof(double)*(size_t)My_NumGridB_CB;

      for (spin=0; spin<spinmax; spin++){
        memmove(&Residual_ReVKSk[spin][My_NumGridB_CB], &Residual_ReVKSk[spin][0],
                nb*(size_t)(List_YOUSO[38]-1));
        memmove(&Residual_ImVKSk[spin][My_NumGridB_CB], &Residual_ImVKSk[spin][0],
                nb*(size_t)(List_YOUSO[38]-1));
      }
    }
    else{

      for (pSCF_iter=(List_YOUSO[38]-1); 0<pSCF_iter; pSCF_iter--){

        p0 = pSCF_iter*My_NumGridB_CB;
        p1 = (pSCF_iter-1)*My_NumGridB_CB;

        for (spin=0; spin<spinmax; spin++){
          for (k=0; k<My_NumGridB_CB; k++){
            Residual_ReVKSk[spin][p0+k] = Residual_ReVKSk[spin][p1+k];
            Residual_ImVKSk[spin][p0+k] = Residual_ImVKSk[spin][p1+k];
          }
        }
      }
    }

  } /* else */

  /****************************************************
                 find VKS in real space 
  ****************************************************/

  tmp0 = 1.0/(double)Ngrid1/(double)Ngrid2/(double)Ngrid3;

  for (spin=0; spin<spinmax; spin++){

    for (k=0; k<My_NumGridB_CB; k++){
      openmx_solver_real_matrix_row(&Re_OptVKSk,0)[k] = ReVKSk[0][spin][k];
      openmx_solver_real_matrix_row(&Im_OptVKSk,0)[k] = ImVKSk[0][spin][k];
    }

    if (spin==0 || spin==1){

      Get_Value_inReal(0,Vpot_Grid_B[spin],Vpot_Grid_B[spin],openmx_solver_real_matrix_row(&Re_OptVKSk,0),openmx_solver_real_matrix_row(&Im_OptVKSk,0));

      for (BN_AB=0; BN_AB<My_NumGridB_AB; BN_AB++){
	Vpot_Grid_B[spin][BN_AB] = Vpot_Grid_B[spin][BN_AB]*tmp0;
      }
    }

    else if (spin==2){

      Get_Value_inReal(1,Vpot_Grid_B[2],Vpot_Grid_B[3],openmx_solver_real_matrix_row(&Re_OptVKSk,0),openmx_solver_real_matrix_row(&Im_OptVKSk,0));

      for (BN_AB=0; BN_AB<My_NumGridB_AB; BN_AB++){
	Vpot_Grid_B[2][BN_AB] = Vpot_Grid_B[2][BN_AB]*tmp0;
	Vpot_Grid_B[3][BN_AB] = Vpot_Grid_B[3][BN_AB]*tmp0;
      }
    }
  }

  if (measure_time==1){ 
    printf("time1=%12.6f time2=%12.6f time3=%12.6f time4=%12.6f time5=%12.6f\n",
            time1,time2,time3,time4,time5);
  }

  /****************************************************
    freeing of arrays:
  ****************************************************/
  
















}




void Kerker_Mixing_VKSk(OpenMXSolverInvocation *scratch, MixingVState *state,  int Change_switch, double Mix_wgt )
{
  int ian,jan,Mc_AN,Gc_AN,spin,spinmax;
  int h_AN,Gh_AN,m,n,i,j,k,k1,k2,k3;
  int MN,pSCF_iter,p0,p1;
  int GN,GNs,N2D,BN_AB,N3[4];
  double Mix_wgt2,Norm,My_Norm,tmp0,tmp1;
  double Min_Weight,Max_Weight,wgt0,wgt1;
  double Gx,Gy,Gz,G2,size_Kerker_weight;
  double sk1,sk2,sk3,G12,G22,G32;
  double G0,G02,G02p,weight;
  int numprocs,myid,ID;
  double *Kerker_weight;
  double *ReVk,*ImVk;
  /* for OpenMP */
  int OMPID,Nthrds,Nprocs,Nloop,Nthrds0;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /* allocate arrays */

  size_Kerker_weight = My_NumGridB_CB;
  Kerker_weight = OpenMX_Solver_OverwriteReal(scratch,"mixv-Kerker_Mixing_VKSk-Kerker_weight",My_NumGridB_CB);
  ReVk = OpenMX_Solver_OverwriteReal(scratch,"mixv-Kerker_Mixing_VKSk-ReVk",My_Max_NumGridB);
  ImVk = OpenMX_Solver_OverwriteReal(scratch,"mixv-Kerker_Mixing_VKSk-ImVk",My_Max_NumGridB);

  if (state->kerker_first)
  PrintMemory("Kerker_Mixing_VKSk: Kerker_weight",sizeof(double)*size_Kerker_weight,NULL);
  state->kerker_first=0;

  /* find an optimum G0 */

  G12 = rtv[1][1]*rtv[1][1] + rtv[1][2]*rtv[1][2] + rtv[1][3]*rtv[1][3]; 
  G22 = rtv[2][1]*rtv[2][1] + rtv[2][2]*rtv[2][2] + rtv[2][3]*rtv[2][3]; 
  G32 = rtv[3][1]*rtv[3][1] + rtv[3][2]*rtv[3][2] + rtv[3][3]*rtv[3][3]; 

  if (G12<G22) G0 = G12;
  else         G0 = G22;
  if (G32<G0)  G0 = G32;

  if (Change_switch==0) G0 = 0.0;
  else                  G0 = sqrt(G0);

  G02 = Kerker_factor*Kerker_factor*G0*G0;
  G02p = (0.1*Kerker_factor*G0)*(0.1*Kerker_factor*G0);

  if      (SpinP_switch==0)  spinmax = 1;
  else if (SpinP_switch==1)  spinmax = 2;
  else if (SpinP_switch==3)  spinmax = 3;

  /***********************************
            set Kerker_weight 
  ************************************/

  N2D = Ngrid3*Ngrid2;
  GNs = ((myid*N2D+numprocs-1)/numprocs)*Ngrid1;

  for (k=0; k<My_NumGridB_CB; k++){

    /* get k3, k2, and k1 */

    GN = k + GNs;     
    k3 = GN/(Ngrid2*Ngrid1);    
    k2 = (GN - k3*Ngrid2*Ngrid1)/Ngrid1;
    k1 = GN - k3*Ngrid2*Ngrid1 - k2*Ngrid1;

    if (k1<Ngrid1/2)  sk1 = (double)k1;
    else              sk1 = (double)(k1 - Ngrid1);

    if (k2<Ngrid2/2)  sk2 = (double)k2;
    else              sk2 = (double)(k2 - Ngrid2);

    if (k3<Ngrid3/2)  sk3 = (double)k3;
    else              sk3 = (double)(k3 - Ngrid3);
 
    Gx = sk1*rtv[1][1] + sk2*rtv[2][1] + sk3*rtv[3][1];
    Gy = sk1*rtv[1][2] + sk2*rtv[2][2] + sk3*rtv[3][2]; 
    Gz = sk1*rtv[1][3] + sk2*rtv[2][3] + sk3*rtv[3][3];
    G2 = Gx*Gx + Gy*Gy + Gz*Gz;

    if (k1==0 && k2==0 && k3==0)  weight = 1.0;
    else                          weight = (G2+G02)/(G2+G02p);

    Kerker_weight[k] = sqrt(weight);

  } /* k */

  /* start... */

  Min_Weight = Min_Mixing_weight;
  if (SCF_RENZOKU==-1){
    Max_Weight = Max_Mixing_weight;
    Max_Mixing_weight2 = Max_Mixing_weight;
  }
  else if (SCF_RENZOKU==1000){  /* past 3 */
    Max_Mixing_weight2 = 2.0*Max_Mixing_weight2;
    if (0.7<Max_Mixing_weight2) Max_Mixing_weight2 = 0.7;
    Max_Weight = Max_Mixing_weight2;
    SCF_RENZOKU = 0;
  }
  else{
    Max_Weight = Max_Mixing_weight2;
  }

  /****************************************************
       norm of residual KS potential in k-space
  ****************************************************/

  MPI_Barrier(mpi_comm_level1);

  My_Norm = 0.0;
  for (spin=0; spin<spinmax; spin++){
    for (k=0; k<My_NumGridB_CB; k++){

      weight = Kerker_weight[k];
      tmp0 = (ReVKSk[0][spin][k] - ReVKSk[1][spin][k])*weight;
      tmp1 = (ImVKSk[0][spin][k] - ImVKSk[1][spin][k])*weight;
      Residual_ReVKSk[spin][My_NumGridB_CB+k] = tmp0;
      Residual_ImVKSk[spin][My_NumGridB_CB+k] = tmp1;

      My_Norm += tmp0*tmp0 + tmp1*tmp1;

    } /* k */
  } /* spin */

  /****************************************************
    MPI: 

    My_Norm
  ****************************************************/

  MPI_Allreduce(&My_Norm, &Norm, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

  /****************************************************
    find an optimum mixing weight
  ****************************************************/

  for (i=4; 1<=i; i--){
    NormRD[i] = NormRD[i-1];
    History_Uele[i] = History_Uele[i-1];
  }
  NormRD[0] = Norm/(double)(1.0e+5*Ngrid1*Ngrid2*Ngrid3*spinmax);
  History_Uele[0] = Uele;

  if (Change_switch==1){

    if ((int)sgn(History_Uele[0]-History_Uele[1])
	  ==(int)sgn(History_Uele[1]-History_Uele[2])
       && NormRD[0]<NormRD[1]){

      /* tmp0 = 1.6*Mixing_weight; */

      tmp0 = 1.6*Mixing_weight;

      if (tmp0<Max_Weight){
        if (Min_Weight<tmp0){
          Mixing_weight = tmp0;
	}
        else{ 
          Mixing_weight = Min_Weight;
	}
      }
      else{ 
        Mixing_weight = Max_Weight;
        SCF_RENZOKU++;  
      }
    }
   
    else if ((int)sgn(History_Uele[0]-History_Uele[1])
             ==(int)sgn(History_Uele[1]-History_Uele[2])
             && NormRD[1]<NormRD[0]){

      /* tmp0 = Mixing_weight/1.6; */
 
      tmp0 = Mixing_weight/1.6;

      if (tmp0<Max_Weight){
        if (Min_Weight<tmp0)
          Mixing_weight = tmp0;
        else 
          Mixing_weight = Min_Weight;
      }
      else 
        Mixing_weight = Max_Weight;

      SCF_RENZOKU = -1;  
    }

    else if ((int)sgn(History_Uele[0]-History_Uele[1])
	     !=(int)sgn(History_Uele[1]-History_Uele[2])
             && NormRD[0]<NormRD[1]){

      /* tmp0 = Mixing_weight*1.2; */

      tmp0 = 1.2*Mixing_weight;

      if (tmp0<Max_Weight){
        if (Min_Weight<tmp0)
          Mixing_weight = tmp0;
        else 
          Mixing_weight = Min_Weight;
      }
      else{ 
        Mixing_weight = Max_Weight;
        SCF_RENZOKU++;
      }
    }

    else if ((int)sgn(History_Uele[0]-History_Uele[1])
	     !=(int)sgn(History_Uele[1]-History_Uele[2])
             && NormRD[1]<NormRD[0]){

      /* tmp0 = Mixing_weight/2.0; */

      tmp0 = Mixing_weight/2.0;

      if (tmp0<Max_Weight){
        if (Min_Weight<tmp0)
          Mixing_weight = tmp0;
        else 
          Mixing_weight = Min_Weight;
      }
      else 
        Mixing_weight = Max_Weight;

      SCF_RENZOKU = -1;
    }

    Mix_wgt = Mixing_weight;
  }

  /****************************************************
                        Mixing
  ****************************************************/

  for (spin=0; spin<spinmax; spin++){
    for (k=0; k<My_NumGridB_CB; k++){

      weight = 1.0/Kerker_weight[k];

      if (Change_switch==0){
        wgt0 = 1.0;
        wgt1 = 0.0;
      }
      else{
        wgt0  = Mix_wgt*weight;
        wgt1 =  1.0 - wgt0;
      }

      ReVKSk[0][spin][k] = wgt0*ReVKSk[0][spin][k] + wgt1*ReVKSk[1][spin][k];
      ImVKSk[0][spin][k] = wgt0*ImVKSk[0][spin][k] + wgt1*ImVKSk[1][spin][k];

      /* correction to largely changing components */

      tmp0 = ReVKSk[0][spin][k] - ReVKSk[1][spin][k];  
      tmp1 = ImVKSk[0][spin][k] - ImVKSk[1][spin][k];  

      if ( maxima_step<(fabs(tmp0)+fabs(tmp1)) ){
	ReVKSk[0][spin][k] = sgn(tmp0)*maxima_step + ReVKSk[1][spin][k]; 
	ImVKSk[0][spin][k] = sgn(tmp1)*maxima_step + ImVKSk[1][spin][k]; 
      }

    } /* k */
  } /* spin */

  /****************************************************
                  shift of KS potentials
  ****************************************************/

  if (MixV_Fast_Enabled()){

    MixVKSk_RotateSlots(ReVKSk, List_YOUSO[38]);
    MixVKSk_RotateSlots(ImVKSk, List_YOUSO[38]);
    for (spin=0; spin<spinmax; spin++){
      memcpy(ReVKSk[0][spin], ReVKSk[1][spin],
             sizeof(double)*(size_t)My_NumGridB_CB);
      memcpy(ImVKSk[0][spin], ImVKSk[1][spin],
             sizeof(double)*(size_t)My_NumGridB_CB);
    }
  }
  else{

    for (pSCF_iter=(List_YOUSO[38]-1); 0<pSCF_iter; pSCF_iter--){
      for (spin=0; spin<spinmax; spin++){
        for (k=0; k<My_NumGridB_CB; k++){
          ReVKSk[pSCF_iter][spin][k] = ReVKSk[pSCF_iter-1][spin][k];
          ImVKSk[pSCF_iter][spin][k] = ImVKSk[pSCF_iter-1][spin][k];
        }
      }
    }
  }

  /****************************************************
              shift of residual KS potentials
  ****************************************************/

  if (MixV_Fast_Enabled()){

    size_t nb = sizeof(double)*(size_t)My_NumGridB_CB;

    for (spin=0; spin<spinmax; spin++){
      memmove(&Residual_ReVKSk[spin][My_NumGridB_CB], &Residual_ReVKSk[spin][0],
              nb*(size_t)(List_YOUSO[38]-1));
      memmove(&Residual_ImVKSk[spin][My_NumGridB_CB], &Residual_ImVKSk[spin][0],
              nb*(size_t)(List_YOUSO[38]-1));
    }
  }
  else{

    for (pSCF_iter=(List_YOUSO[38]-1); 0<pSCF_iter; pSCF_iter--){

      p0 = pSCF_iter*My_NumGridB_CB;
      p1 = (pSCF_iter-1)*My_NumGridB_CB;

      for (spin=0; spin<spinmax; spin++){
        for (k=0; k<My_NumGridB_CB; k++){
          Residual_ReVKSk[spin][p0+k] = Residual_ReVKSk[spin][p1+k];
          Residual_ImVKSk[spin][p0+k] = Residual_ImVKSk[spin][p1+k];
        }
      }
    }
  }

  /************************************************************
    find the KS potentials for the partition B in real space 
  ************************************************************/

  tmp0 = 1.0/(double)(Ngrid1*Ngrid2*Ngrid3);

  for (spin=0; spin<spinmax; spin++){

    for (k=0; k<My_NumGridB_CB; k++){
      ReVk[k] = ReVKSk[0][spin][k];
      ImVk[k] = ImVKSk[0][spin][k];
    }

    if (spin==0 || spin==1){

      Get_Value_inReal(0,Vpot_Grid_B[spin],Vpot_Grid_B[spin],ReVk,ImVk);

      for (BN_AB=0; BN_AB<My_NumGridB_AB; BN_AB++){
	Vpot_Grid_B[spin][BN_AB] = Vpot_Grid_B[spin][BN_AB]*tmp0;
      }
    }

    else if (spin==2){

      Get_Value_inReal(1,Vpot_Grid_B[2],Vpot_Grid_B[3],ReVk,ImVk);

      for (BN_AB=0; BN_AB<My_NumGridB_AB; BN_AB++){
	Vpot_Grid_B[2][BN_AB] = Vpot_Grid_B[2][BN_AB]*tmp0;
	Vpot_Grid_B[3][BN_AB] = Vpot_Grid_B[3][BN_AB]*tmp0;
      }
    }
  }

  /* freeing of arrays */




}





void Inverse(OpenMXSolverInvocation *scratch, int n, const OpenMXSolverRealTensor *a, const OpenMXSolverRealTensor *ia)
{
  int method_flag=1;

  if (method_flag==0){

  /****************************************************
                  LU decomposition
                      0 to n
   NOTE:
   This routine does not consider the reduction of rank
  ****************************************************/

  int i,j,k;
  double w;
  double *x,*y;
  OpenMXSolverRealTensor da;

  /***************************************************
    allocation of arrays: 

     x[List_YOUSO[38]]
     y[List_YOUSO[38]]
     da[List_YOUSO[38]][List_YOUSO[38]]
  ***************************************************/

  x = OpenMX_Solver_OverwriteReal(scratch,"mixv-Inverse-x",List_YOUSO[38]);
  y = OpenMX_Solver_OverwriteReal(scratch,"mixv-Inverse-y",List_YOUSO[38]);
  for (i=0; i<List_YOUSO[38]; i++){
    x[i] = 0.0;
    y[i] = 0.0;
  }

  da = OpenMX_Solver_OverwriteView(scratch,"mixv-Inverse-da",List_YOUSO[38],List_YOUSO[38]);
  for (i=0; i<List_YOUSO[38]; i++){
    for (j=0; j<List_YOUSO[38]; j++){
      openmx_solver_real_matrix_row(&da,i)[j] = 0.0;
    }
  }

  /* start calc. */

  if (n==-1){
    for (i=0; i<List_YOUSO[38]; i++){
      for (j=0; j<List_YOUSO[38]; j++){
        openmx_solver_real_matrix_row(a,i)[j] = 0.0;
      }
    }
  }
  else{
    for (i=0; i<=n; i++){
      for (j=0; j<=n; j++){
        openmx_solver_real_matrix_row(&da,i)[j] = openmx_solver_real_matrix_row(a,i)[j];
      }
    }

    /****************************************************
                     LU factorization
    ****************************************************/

    for (k=0; k<=n-1; k++){
      w = 1.0/openmx_solver_real_matrix_row(a,k)[k];
      for (i=k+1; i<=n; i++){
        openmx_solver_real_matrix_row(a,i)[k] = w*openmx_solver_real_matrix_row(a,i)[k];
	for (j=k+1; j<=n; j++){
          openmx_solver_real_matrix_row(a,i)[j] = openmx_solver_real_matrix_row(a,i)[j] - openmx_solver_real_matrix_row(a,i)[k]*openmx_solver_real_matrix_row(a,k)[j];
	}
      }
    }

    for (k=0; k<=n; k++){

      /****************************************************
                             Ly = b
      ****************************************************/

      for (i=0; i<=n; i++){
	if (i==k)
	  y[i] = 1.0;
	else
	  y[i] = 0.0;
	for (j=0; j<=i-1; j++){
          y[i] = y[i] - openmx_solver_real_matrix_row(a,i)[j]*y[j];
	}
      }

      /****************************************************
                             Ux = y 
      ****************************************************/

      for (i=n; 0<=i; i--){
	x[i] = y[i];
	for (j=n; (i+1)<=j; j--){
          x[i] = x[i] - openmx_solver_real_matrix_row(a,i)[j]*x[j];
	}
        x[i] = x[i]/openmx_solver_real_matrix_row(a,i)[i];
        openmx_solver_real_matrix_row(ia,i)[k] = x[i];
      }
    }

    for (i=0; i<=n; i++){
      for (j=0; j<=n; j++){
        openmx_solver_real_matrix_row(a,i)[j] = openmx_solver_real_matrix_row(&da,i)[j];
      }
    }
  }

  /***************************************************
    freeing of arrays: 

     x[List_YOUSO[38]]
     y[List_YOUSO[38]]
     da[List_YOUSO[38]][List_YOUSO[38]]
  ***************************************************/






  }
  
  else if (method_flag==1){

    int i,j,M,N,LDA,INFO;
    int *IPIV,LWORK;
    double *A,*WORK;

    A = OpenMX_Solver_OverwriteReal(scratch,"mixv-Inverse-lu-A",(size_t)(n+2)*(size_t)(n+2));
    WORK = OpenMX_Solver_OverwriteReal(scratch,"mixv-Inverse-lu-WORK",(n+2));
    IPIV = OpenMX_Solver_Int(scratch,"mixv-Inverse-lu-IPIV",(n+2));

    for (i=0; i<=n; i++){
      for (j=0; j<=n; j++){
        A[i*(n+1)+j] = openmx_solver_real_matrix_row(a,i)[j];
      }
    }

    M = n + 1;
    N = M;
    LDA = M;
    LWORK = M;

    F77_NAME(dgetrf,DGETRF)( &M, &N, A, &LDA, IPIV, &INFO);
    F77_NAME(dgetri,DGETRI)( &N, A, &LDA, IPIV, WORK, &LWORK, &INFO);

    for (i=0; i<=n; i++){
      for (j=0; j<=n; j++){
        openmx_solver_real_matrix_row(ia,i)[j] = A[i*(n+1)+j];
      }
    }




  }

  else if (method_flag==2){

    int N,i,j,k;
    double *A,*B,*ko;
    double sum;

    N = n + 1;

    A = OpenMX_Solver_OverwriteReal(scratch,"mixv-Inverse-eigen-A",(size_t)(N+2)*(size_t)(N+2));
    B = OpenMX_Solver_OverwriteReal(scratch,"mixv-Inverse-eigen-B",(size_t)(N+2)*(size_t)(N+2));
    ko = OpenMX_Solver_OverwriteReal(scratch,"mixv-Inverse-eigen-ko",(N+2));

    for (i=0; i<N; i++){
      for (j=0; j<N; j++){
        A[j*N+i] = openmx_solver_real_matrix_row(a,i)[j];
      }
    }

    Eigen_lapack3(A, ko, N, N); 

    for (i=0; i<N; i++){
      ko[i] = 1.0/(ko[i]+1.0e-13);
    } 

    for (i=0; i<N; i++){
      for (j=0; j<N; j++){
        B[i*N+j] = A[i*N+j]*ko[i];
      }
    }

    for (i=0; i<N; i++){
      for (j=0; j<N; j++){
        openmx_solver_real_matrix_row(ia,i)[j] = 0.0;
      }
    }

    for (i=0; i<N; i++){
      for (j=0; j<N; j++){
        sum = 0.0;
	for (k=0; k<N; k++){
	  sum += A[k*N+i]*B[k*N+j];
	}
        openmx_solver_real_matrix_row(ia,i)[j] = sum;
      }
    }




  }
}

void Complex_Inverse(OpenMXSolverInvocation *scratch, int n, const OpenMXSolverRealTensor *a, const OpenMXSolverRealTensor *ia, const OpenMXSolverRealTensor *b, const OpenMXSolverRealTensor *ib)
{
  /****************************************************
                  LU decomposition
                      0 to n
   NOTE:
   This routine does not consider the reduction of rank
  ****************************************************/

  int i,j,k;
  double w2,re,im;
  dcomplex w;
  dcomplex *x,*y;
  OpenMXSolverComplexTensor da;

  /***************************************************
    allocation of arrays: 

     x[List_YOUSO[38]]
     y[List_YOUSO[38]]
     da[List_YOUSO[38]][List_YOUSO[38]]
  ***************************************************/

  x = OpenMX_Solver_OverwriteComplex(scratch,"mixv-Complex_Inverse-x",List_YOUSO[38]);
  y = OpenMX_Solver_OverwriteComplex(scratch,"mixv-Complex_Inverse-y",List_YOUSO[38]);

  da = OpenMX_Solver_OverwriteComplexView(scratch,"mixv-Complex_Inverse-da",List_YOUSO[38],List_YOUSO[38]);


  /* start calc. */

  if (n==-1){
    for (i=0; i<List_YOUSO[38]; i++){
      for (j=0; j<List_YOUSO[38]; j++){
         openmx_solver_real_matrix_row(a,i)[j] = 0.0;
        openmx_solver_real_matrix_row(ia,i)[j] = 0.0;
      }
    }
  }

  else {

    for (i=0; i<=n; i++){
      for (j=0; j<=n; j++){
        openmx_solver_complex_matrix_row(&da,i)[j].r =  openmx_solver_real_matrix_row(a,i)[j];
        openmx_solver_complex_matrix_row(&da,i)[j].i = openmx_solver_real_matrix_row(ia,i)[j];
      }
    }

    /****************************************************
                     LU factorization
    ****************************************************/

    for (k=0; k<=n-1; k++){

      w.r =  openmx_solver_real_matrix_row(a,k)[k];
      w.i = openmx_solver_real_matrix_row(ia,k)[k];
      w2 = 1.0/(w.r*w.r + w.i*w.i); 

      for (i=k+1; i<=n; i++){

        re = w2*(w.r* openmx_solver_real_matrix_row(a,i)[k] + w.i*openmx_solver_real_matrix_row(ia,i)[k]);
        im = w2*(w.r*openmx_solver_real_matrix_row(ia,i)[k] - w.i* openmx_solver_real_matrix_row(a,i)[k]);
        openmx_solver_real_matrix_row(a,i)[k] = re;
       openmx_solver_real_matrix_row(ia,i)[k] = im;

	for (j=k+1; j<=n; j++){
           re =  openmx_solver_real_matrix_row(a,i)[j] - (openmx_solver_real_matrix_row(a,i)[k]* openmx_solver_real_matrix_row(a,k)[j] - openmx_solver_real_matrix_row(ia,i)[k]*openmx_solver_real_matrix_row(ia,k)[j]);
           im = openmx_solver_real_matrix_row(ia,i)[j] - (openmx_solver_real_matrix_row(a,i)[k]*openmx_solver_real_matrix_row(ia,k)[j] + openmx_solver_real_matrix_row(ia,i)[k]* openmx_solver_real_matrix_row(a,k)[j]);
           openmx_solver_real_matrix_row(a,i)[j] = re;
          openmx_solver_real_matrix_row(ia,i)[j] = im;
	}
      }
    }

    for (k=0; k<=n; k++){

      /****************************************************
                             Ly = b
      ****************************************************/

      for (i=0; i<=n; i++){
	if (i==k){
	  y[i].r = 1.0;
	  y[i].i = 0.0;
	}
	else{
	  y[i].r = 0.0;
	  y[i].i = 0.0;
	}

	for (j=0; j<=i-1; j++){
          y[i].r = y[i].r - (openmx_solver_real_matrix_row(a,i)[j]*y[j].r - openmx_solver_real_matrix_row(ia,i)[j]*y[j].i);
          y[i].i = y[i].i - (openmx_solver_real_matrix_row(a,i)[j]*y[j].i + openmx_solver_real_matrix_row(ia,i)[j]*y[j].r);
	}
      }

      /****************************************************
                             Ux = y 
      ****************************************************/

      for (i=n; 0<=i; i--){
	x[i] = y[i];
	for (j=n; (i+1)<=j; j--){
          x[i].r = x[i].r - (openmx_solver_real_matrix_row(a,i)[j]*x[j].r - openmx_solver_real_matrix_row(ia,i)[j]*x[j].i);
          x[i].i = x[i].i - (openmx_solver_real_matrix_row(a,i)[j]*x[j].i + openmx_solver_real_matrix_row(ia,i)[j]*x[j].r);
	}

        w.r =  openmx_solver_real_matrix_row(a,i)[i];
        w.i = openmx_solver_real_matrix_row(ia,i)[i];
        w2 = 1.0/(w.r*w.r + w.i*w.i); 

	re = w2*(w.r*x[i].r + w.i*x[i].i);
        im = w2*(w.r*x[i].i - w.i*x[i].r);
	x[i].r = re;
	x[i].i = im;

         openmx_solver_real_matrix_row(b,i)[k] = x[i].r;
        openmx_solver_real_matrix_row(ib,i)[k] = x[i].i;
      }
    }

    for (i=0; i<=n; i++){
      for (j=0; j<=n; j++){
         openmx_solver_real_matrix_row(a,i)[j] = openmx_solver_complex_matrix_row(&da,i)[j].r;
        openmx_solver_real_matrix_row(ia,i)[j] = openmx_solver_complex_matrix_row(&da,i)[j].i;
      }
    }
  }

  /***************************************************
    freeing of arrays: 

     x[List_YOUSO[38]]
     y[List_YOUSO[38]]
     da[List_YOUSO[38]][List_YOUSO[38]]
  ***************************************************/





}

static void MixV_Reset(OpenMXNativeResources *owner,void *metadata)
{
  (void)owner;
  memset(metadata,0,sizeof(MixingVState));
}

OpenMXMixingVState *Mixing_V_Bind_Resources(OpenMXCoreContext *core)
{
  OpenMXNativeResources *owner=NULL;void *metadata=NULL;char error[256];
  if (!openmx_core_native_resources_state(core,"Mixing_V",sizeof(MixingVState),
      MixV_Reset,&owner,&metadata,error,sizeof(error))) {
    fprintf(stderr,"OpenMX potential mixing state: %s\n",error);
    abort();
  }
  MixingVState *state=(MixingVState*)metadata;
  if (!state->initialized) {
    state->resources=owner;
    state->diis_first=state->kerker_first=1;
    state->fast.last_scf_iter=-1;
    state->initialized=1;
  }
  return state;
}

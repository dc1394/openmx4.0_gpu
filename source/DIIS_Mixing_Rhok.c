/**********************************************************************
  DIIS_Mixing_Rhok.c:

     DIIS_Mixing_Rhok.c is a subroutine to achieve self-consistent field
     using the direct inversion in the iterative subspace in k-space.

  Log of DIIS_Mixing_Rhok.c:

     3/Jan/2005  Released by T.Ozaki

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "mpi.h"
#include "openmx_common.h"
#include "openmx_mixing_policy.h"
#include "openmx_env.h"
#include "openmx_mixing_rhok_workspace.h"
#include "tran_prototypes.h"
#include "tran_variables.h"
#include "lapack_prototypes.h"
#include "openmx_mixing_lapack.h"
#include <omp.h>


#define  measure_time   0
#define  maxima_step  10000000.0

#ifdef MIN
#undef MIN
#endif
#define MIN(a,b) ((a)<(b))?  (a):(b)


static void Inverse(OpenMXSolverInvocation *scratch, int n, const OpenMXSolverRealTensor *a, const OpenMXSolverRealTensor *ia);
static void Complex_Inverse(OpenMXSolverInvocation *scratch, int n, const OpenMXSolverRealTensor *a, const OpenMXSolverRealTensor *ia, const OpenMXSolverRealTensor *b, const OpenMXSolverRealTensor *ib);

void F77_NAME(dgemv,DGEMV)(const char *trans, const int *m, const int *n,
                           const double *alpha, const double *a, const int *lda,
                           const double *x, const int *incx, const double *beta,
                           double *y, const int *incy);

/*******************************************************************************
  Fast path of the k-space DIIS (active under scf.eigen.lib gpusolver,
  disabled with OPENMX_MIX_FAST=0):

  - The stored residual columns are pre-weighted and only slide between
    iterations, so the Gram matrix entries of the older pairs stay valid.
    On a continuity iteration only the new row (fresh residual against all
    live columns) is computed with dgemv and reduced with one short
    MPI_Allreduce; the rest of A comes from the host cache of the previous
    (already Allreduced, hence rank-uniform) matrix.  Any break in the
    SCF-iteration continuity falls back to the legacy dgemm+harvest.
  - The O(List_YOUSO[38]*Ngrid) shifts become a pointer rotation of the
    top-level Rhok slots (plus one slot copy: the legacy shift leaves
    slot 0 in place and the tail reads it) and a single memmove per spin
    for the flat residual matrix.  Both are bitwise-identical to the
    legacy shifts.
*******************************************************************************/

static int Mix_Fast_Enabled(void)
{
  if (scf_eigen_lib_flag != GPUSOLVER) return 0;
  return openmx_env_flag_at(OPENMX_ENV_MIX_FAST, 1);
}

static void MixRhok_Fast_Invalidate(MixingRhokState *state)
{
  state->fast.valid = 0;
  state->fast.last_scf_iter = -1;
}

/* Preserve the legacy free/allocate/layout-update order. The C++ resource
   group also drains the cache on partial initialization and context teardown. */
static int MixRhok_FastResize(MixingRhokState *state, int history, int spinmax)
{
  openmx_core_native_free(state->resources,state->fast.A);
  state->fast.A=(double*)openmx_core_native_malloc(state->resources,1,
      sizeof(double)*(size_t)history*(size_t)history,0);
  if (state->fast.A==NULL) return 0;
  state->fast.l38=history;
  state->fast.spinmax=spinmax;
  MixRhok_Fast_Invalidate(state);
  return 1;
}

/* rotate the top slot pointers: afterwards slot m holds the former slot m-1
   (m=1..nslot-1) and slot 0 is a recycled buffer overwritten by the caller */
static void MixRhok_RotateSlots(double ***arr, int nslot)
{
  double **tmp = arr[nslot-1];
  int m;
  for (m=nslot-1; 0<m; m--) arr[m] = arr[m-1];
  arr[0] = tmp;
}


static void DIIS_Mixing_Rhok_Normal(OpenMXSolverInvocation *scratch, MixingRhokState *state, int SCF_iter,
			      double Mix_wgt,
			      double ***ReRhok,
			      double ***ImRhok,
			      double **Residual_ReRhok,
			      double **Residual_ImRhok,
			      double *ReVk,
			      double *ImVk,
			      double *ReRhoAtomk,
			      double *ImRhoAtomk);


static void DIIS_Mixing_Rhok_NEGF(OpenMXSolverInvocation *scratch, MixingRhokState *state, int SCF_iter,
				  double Mix_wgt,
				  double ***ReRhok,
				  double ***ImRhok,
				  double **Residual_ReRhok,
				  double **Residual_ImRhok,
				  double *ReVk,
				  double *ImVk,
				  double *ReRhoAtomk,
				  double *ImRhoAtomk);
	
void DIIS_Mixing_Rhok(MixingRhokState *state, int SCF_iter,
                      double Mix_wgt,
                      double ***ReRhok,
                      double ***ImRhok,
                      double **Residual_ReRhok,
                      double **Residual_ImRhok,
                      double *ReVk,
                      double *ImVk,
                      double *ReRhoAtomk,
                      double *ImRhoAtomk)
{

  if (Solver!=4 || TRAN_Poisson_flag==2){

    MixRhok_WithScratch(DIIS_Mixing_Rhok_Normal,state,
			    SCF_iter,
			    Mix_wgt,
			    ReRhok,
			    ImRhok,
			    Residual_ReRhok,
			    Residual_ImRhok,
			    ReVk,
			    ImVk,
			    ReRhoAtomk,
			    ImRhoAtomk);
  }

  else if (Solver==4){

    MixRhok_WithScratch(DIIS_Mixing_Rhok_NEGF,state,
			  SCF_iter,
			  Mix_wgt,
			  ReRhok,
			  ImRhok,
			  Residual_ReRhok,
			  Residual_ImRhok,
			  ReVk,
			  ImVk,
			  ReRhoAtomk,
			  ImRhoAtomk);
  }

}




void DIIS_Mixing_Rhok_Normal(OpenMXSolverInvocation *scratch, MixingRhokState *state, int SCF_iter,
			     double Mix_wgt,
			     double ***ReRhok,
			     double ***ImRhok,
			     double **Residual_ReRhok,
			     double **Residual_ImRhok,
			     double *ReVk,
			     double *ImVk,
			     double *ReRhoAtomk,
			     double *ImRhoAtomk)
{
  int spin,Mc_AN,Gc_AN,wan1,TNO1,h_AN,Gh_AN;
  int wan2,TNO2,i,j,k,NumMix,NumSlide;
  int SCFi,SCFj,tno1,tno0,Cwan,Hwan,k1,k2,k3,kk2; 
  int pSCF_iter,size_Re_OptRhok,size_Kerker_weight;
  int spinmax,MN,flag_nan,refiter,po3,po4,refiter2;
  int N2D,GN,GNs,N3[4],BN_AB,p0,p1,M,N,K;
  double time1,time2,time3,time4,time5; 
  double Stime1,Etime1,scale,be;
  double *alden,*alden0,*gradF;
  OpenMXSolverRealTensor A,IA;
  double *My_NMat,*NMat;
  double alpha,beta;
  double tmp0,itmp0,tmp1;
  double im1,im2,re1,re2; 
  double OptRNorm,My_OptRNorm,sum0,sum1;
  double dum1,dum2,bunsi,bunbo,sum,coef_OptRho;
  double Gx,Gy,Gz,G2,G12,G22,G32,F,F0;
  double Min_Weight;
  double Max_Weight;
  double G0,G02,G02p,weight,wgt0,wgt1;
  double sk1,sk2,sk3;
  OpenMXSolverRealTensor Re_OptRhok;
  OpenMXSolverRealTensor Im_OptRhok;
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
  
  alden = OpenMX_Solver_OverwriteReal(scratch,"rhok-DIIS_Mixing_Rhok_Normal-alden",List_YOUSO[38]);
  alden0 = OpenMX_Solver_OverwriteReal(scratch,"rhok-DIIS_Mixing_Rhok_Normal-alden0",List_YOUSO[38]);
  gradF = OpenMX_Solver_OverwriteReal(scratch,"rhok-DIIS_Mixing_Rhok_Normal-gradF",List_YOUSO[38]);
  
  My_NMat = OpenMX_Solver_OverwriteReal(scratch,"rhok-DIIS_Mixing_Rhok_Normal-My_NMat",(size_t)List_YOUSO[38]*(size_t)List_YOUSO[38]);
  NMat = OpenMX_Solver_OverwriteReal(scratch,"rhok-DIIS_Mixing_Rhok_Normal-NMat",(size_t)List_YOUSO[38]*(size_t)List_YOUSO[38]);

  A = OpenMX_Solver_OverwriteView(scratch,"rhok-DIIS_Mixing_Rhok_Normal-A",List_YOUSO[38],List_YOUSO[38]);
  
  IA = OpenMX_Solver_OverwriteView(scratch,"rhok-DIIS_Mixing_Rhok_Normal-IA",List_YOUSO[38],List_YOUSO[38]);

  /* Re_OptRhok and Im_OptRhok */

  size_Re_OptRhok = spinmax*My_NumGridB_CB;

  Re_OptRhok = OpenMX_Solver_OverwriteView(scratch,"rhok-DIIS_Mixing_Rhok_Normal-Re_OptRhok",spinmax,My_NumGridB_CB);


  Im_OptRhok = OpenMX_Solver_OverwriteView(scratch,"rhok-DIIS_Mixing_Rhok_Normal-Im_OptRhok",spinmax,My_NumGridB_CB);


  size_Kerker_weight = My_NumGridB_CB;
  Kerker_weight = OpenMX_Solver_OverwriteReal(scratch,"rhok-DIIS_Mixing_Rhok_Normal-Kerker_weight",My_NumGridB_CB);

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

    if (k1==0 && k2==0 && k3==0)  weight = 0.5;
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
    if (state->diis_first_normal) {
      PrintMemory("DIIS_Mixing_Rhok: Re_OptRhok",sizeof(double)*size_Re_OptRhok,NULL);
      PrintMemory("DIIS_Mixing_Rhok: Im_OptRhok",sizeof(double)*size_Re_OptRhok,NULL);
      PrintMemory("DIIS_Mixing_Rhok: Kerker_weight",sizeof(double)*size_Kerker_weight,NULL);
      state->diis_first_normal=0;
    }

    Kerker_Mixing_Rhok(state, 0,1.00,ReRhok,ImRhok,
                       Residual_ReRhok,Residual_ImRhok,
                       ReVk,ImVk,ReRhoAtomk,ImRhoAtomk);
  } 

  else if (SCF_iter==2){

    if (measure_time==1) dtime(&Stime1);

    /* construct residual rho */

    for (spin=0; spin<spinmax; spin++){
      for (k=0; k<My_NumGridB_CB; k++){

	Residual_ReRhok[spin][2*My_NumGridB_CB+k] = (ReRhok[0][spin][k] - ReRhok[1][spin][k])*Kerker_weight[k];
	Residual_ImRhok[spin][2*My_NumGridB_CB+k] = (ImRhok[0][spin][k] - ImRhok[1][spin][k])*Kerker_weight[k];

      } /* k */
    } /* spin */

    if (measure_time==1){ 
      dtime(&Etime1);
      time2 = Etime1 - Stime1;      
    }

    /* rho1 to rho2 */

    for (spin=0; spin<spinmax; spin++){
      for (k=0; k<My_NumGridB_CB; k++){
	ReRhok[2][spin][k] = ReRhok[1][spin][k];
	ImRhok[2][spin][k] = ImRhok[1][spin][k];
      }
    }

    Kerker_Mixing_Rhok(state, 0,state->history->weight,ReRhok,ImRhok,
                       Residual_ReRhok,Residual_ImRhok,
                       ReVk,ImVk,ReRhoAtomk,ImRhoAtomk);
  } 

  else {

    /****************************************************
                   construct residual rho1
    ****************************************************/

    if (measure_time==1) dtime(&Stime1);

    for (spin=0; spin<spinmax; spin++){
      for (k=0; k<My_NumGridB_CB; k++){

	Residual_ReRhok[spin][My_NumGridB_CB+k] = (ReRhok[0][spin][k] - ReRhok[1][spin][k])*Kerker_weight[k];
	Residual_ImRhok[spin][My_NumGridB_CB+k] = (ImRhok[0][spin][k] - ImRhok[1][spin][k])*Kerker_weight[k];

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
                    alpha from residual rho
    ****************************************************/

    if (measure_time==1) dtime(&Stime1);

    /* calculation of the norm matrix for the residual vectors */

    {
      int use_fast = Mix_Fast_Enabled();
      int incremental = 0;

      if (use_fast){
        if (state->fast.A == NULL || state->fast.l38 != List_YOUSO[38] ||
            state->fast.spinmax != spinmax){
          if (!MixRhok_FastResize(state,List_YOUSO[38],spinmax)){
            printf("DIIS_Mixing_Rhok: failed to allocate the Gram cache.\n");
            MPI_Abort(mpi_comm_level1,1);
          }
        }
        incremental = (state->fast.valid &&
                       state->fast.last_scf_iter+1 == SCF_iter &&
                       (NumMix == state->fast.last_nummix ||
                        NumMix == state->fast.last_nummix+1));
      }

      if (incremental){

        /* the stored residual columns only slid since the last call, so the
           older pairs of the (rank-uniform, Allreduced) Gram matrix stay
           valid; only the fresh-residual row is computed and reduced */

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
                                 Residual_ReRhok[spin], &K,
                                 &Residual_ReRhok[spin][My_NumGridB_CB], &inc1,
                                 &beta, my_row, &inc1);
          F77_NAME(dgemv,DGEMV)( "T", &K, &M, &alpha,
                                 Residual_ImRhok[spin], &K,
                                 &Residual_ImRhok[spin][My_NumGridB_CB], &inc1,
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
                                 Residual_ReRhok[spin], &K,
                                 Residual_ReRhok[spin], &K,
                                 &beta,
                                 My_NMat,
                                 &M);

          F77_NAME(dgemm,DGEMM)( "T","N", &M, &N, &K,
                                 &alpha,
                                 Residual_ImRhok[spin], &K,
                                 Residual_ImRhok[spin], &K,
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
        MixRhok_Fast_Invalidate(state);
      }
    }

    if (measure_time==1){ 
      dtime(&Etime1);
      time4 = Etime1 - Stime1;      
    }

    /* store NormRD */

    OpenMX_Mixing_Shift(state->history);
    state->history->norm[0] = openmx_solver_real_matrix_row(&A,0)[0]/(double)Ngrid1/(double)Ngrid2/(double)Ngrid3/(double)spinmax;
    state->history->energy[0] = state->energies->band;

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
        openmx_solver_real_matrix_row(&Re_OptRhok,spin)[k] = 0.0;
        openmx_solver_real_matrix_row(&Im_OptRhok,spin)[k] = 0.0;
      }
    }   

    /* Pulay mixing for residual Rhok */

    for (pSCF_iter=1; pSCF_iter<=NumMix; pSCF_iter++){

      tmp0 = alden[pSCF_iter]; 
      p0 = pSCF_iter*My_NumGridB_CB;

#pragma omp parallel shared(spinmax,My_NumGridB_CB,Re_OptRhok,Im_OptRhok,tmp0,p0,Residual_ReRhok,Residual_ImRhok)
      {

        int OMPID,Nthrds,k; 

	OMPID = omp_get_thread_num();
	Nthrds = omp_get_num_threads();

	if (spinmax==1){
	  for (k=My_NumGridB_CB*OMPID/Nthrds; k<My_NumGridB_CB*(OMPID+1)/Nthrds; k++){
            openmx_solver_real_matrix_row(&Re_OptRhok,0)[k] += tmp0*Residual_ReRhok[0][p0+k];
            openmx_solver_real_matrix_row(&Im_OptRhok,0)[k] += tmp0*Residual_ImRhok[0][p0+k];
	  } /* k */
	}

	else if (spinmax==2){
	  for (k=My_NumGridB_CB*OMPID/Nthrds; k<My_NumGridB_CB*(OMPID+1)/Nthrds; k++){
            openmx_solver_real_matrix_row(&Re_OptRhok,0)[k] += tmp0*Residual_ReRhok[0][p0+k];
            openmx_solver_real_matrix_row(&Im_OptRhok,0)[k] += tmp0*Residual_ImRhok[0][p0+k];
            openmx_solver_real_matrix_row(&Re_OptRhok,1)[k] += tmp0*Residual_ReRhok[1][p0+k];
            openmx_solver_real_matrix_row(&Im_OptRhok,1)[k] += tmp0*Residual_ImRhok[1][p0+k];
	  } /* k */
	}

	else if (spinmax==3){
	  for (k=My_NumGridB_CB*OMPID/Nthrds; k<My_NumGridB_CB*(OMPID+1)/Nthrds; k++){
            openmx_solver_real_matrix_row(&Re_OptRhok,0)[k] += tmp0*Residual_ReRhok[0][p0+k];
            openmx_solver_real_matrix_row(&Im_OptRhok,0)[k] += tmp0*Residual_ImRhok[0][p0+k];
            openmx_solver_real_matrix_row(&Re_OptRhok,1)[k] += tmp0*Residual_ReRhok[1][p0+k];
            openmx_solver_real_matrix_row(&Im_OptRhok,1)[k] += tmp0*Residual_ImRhok[1][p0+k];
            openmx_solver_real_matrix_row(&Re_OptRhok,2)[k] += tmp0*Residual_ReRhok[2][p0+k];
            openmx_solver_real_matrix_row(&Im_OptRhok,2)[k] += tmp0*Residual_ImRhok[2][p0+k];
	  } /* k */
	}

      } /* #pragma omp parallel */

    } /* pSCF_iter */

    if      (1.0e-2<=state->history->norm[0])                        coef_OptRho = 0.5;
    else if (1.0e-4<=state->history->norm[0]  && state->history->norm[0]<1.0e-2)   coef_OptRho = 0.6;
    else if (1.0e-6<=state->history->norm[0]  && state->history->norm[0]<1.0e-4)   coef_OptRho = 0.7;
    else if (1.0e-8<=state->history->norm[0]  && state->history->norm[0]<1.0e-6)   coef_OptRho = 0.8;
    else if (1.0e-10<=state->history->norm[0] && state->history->norm[0]<1.0e-8)   coef_OptRho = 0.9;
    else                                               coef_OptRho = 1.0;

    /****************************************************
                        mixing of rho 
    ****************************************************/

    /****************************************************
       Pulay mixing
    ****************************************************/

    if ( SCF_iter%EveryPulay_SCF==0 ) {

      /* reduce Mixing_weight so that charge sloshing can be avoided at the next SCF */

      state->history->weight = 0.1*Max_Mixing_weight;
      if (state->history->weight<Min_Mixing_weight) state->history->weight = Min_Mixing_weight;
      
      /* initial ReRhok and ImRhok */
      
      for (spin=0; spin<spinmax; spin++){
        for (k=0; k<My_NumGridB_CB; k++){
          ReRhok[0][spin][k] = 0.0;
	  ImRhok[0][spin][k] = 0.0;
	}
      }

      /* Pulay mixing */

      for (pSCF_iter=1; pSCF_iter<=NumMix; pSCF_iter++){

	tmp0 =  alden[pSCF_iter];

#pragma omp parallel shared(tmp0,pSCF_iter,spinmax,My_NumGridB_CB,ReRhok,ImRhok) 
	{

          int OMPID,Nthrds,k; 

	  OMPID = omp_get_thread_num();
	  Nthrds = omp_get_num_threads();

	  if (spinmax==1){
  	    for (k=My_NumGridB_CB*OMPID/Nthrds; k<My_NumGridB_CB*(OMPID+1)/Nthrds; k++){
	      ReRhok[0][0][k] += tmp0*ReRhok[pSCF_iter][0][k];
	      ImRhok[0][0][k] += tmp0*ImRhok[pSCF_iter][0][k];
	    } /* k */
	  }

	  else if (spinmax==2){
  	    for (k=My_NumGridB_CB*OMPID/Nthrds; k<My_NumGridB_CB*(OMPID+1)/Nthrds; k++){
	      ReRhok[0][0][k] += tmp0*ReRhok[pSCF_iter][0][k];
	      ImRhok[0][0][k] += tmp0*ImRhok[pSCF_iter][0][k];
	      ReRhok[0][1][k] += tmp0*ReRhok[pSCF_iter][1][k];
	      ImRhok[0][1][k] += tmp0*ImRhok[pSCF_iter][1][k];
	    } /* k */
	  }

	  else if (spinmax==3){
  	    for (k=My_NumGridB_CB*OMPID/Nthrds; k<My_NumGridB_CB*(OMPID+1)/Nthrds; k++){
	      ReRhok[0][0][k] += tmp0*ReRhok[pSCF_iter][0][k];
	      ImRhok[0][0][k] += tmp0*ImRhok[pSCF_iter][0][k];
	      ReRhok[0][1][k] += tmp0*ReRhok[pSCF_iter][1][k];
	      ImRhok[0][1][k] += tmp0*ImRhok[pSCF_iter][1][k];
	      ReRhok[0][2][k] += tmp0*ReRhok[pSCF_iter][2][k];
	      ImRhok[0][2][k] += tmp0*ImRhok[pSCF_iter][2][k];
	    } /* k */
	  }

	} /* #pragma omp parallel */

      } /* pSCF_iter */

      /* Correction by optimized residual rho */

      for (spin=0; spin<spinmax; spin++){

        for (k=0; k<My_NumGridB_CB; k++){

          if (1.0<state->history->norm[0])
  	    weight = 1.0/(Kerker_weight[k]*Kerker_weight[k]);
          else 
            weight = 1.0/Kerker_weight[k];

          ReRhok[0][spin][k] += coef_OptRho*openmx_solver_real_matrix_row(&Re_OptRhok,spin)[k]*weight;
          ImRhok[0][spin][k] += coef_OptRho*openmx_solver_real_matrix_row(&Im_OptRhok,spin)[k]*weight;

	  /* correction to large changing components */

          tmp0 = ReRhok[0][spin][k] - ReRhok[1][spin][k];  
          tmp1 = ImRhok[0][spin][k] - ImRhok[1][spin][k];  

          if ( maxima_step<(fabs(tmp0)+fabs(tmp1)) ){
            ReRhok[0][spin][k] = sgn(tmp0)*maxima_step + ReRhok[1][spin][k]; 
            ImRhok[0][spin][k] = sgn(tmp1)*maxima_step + ImRhok[1][spin][k];
          }

	} /* k */
      } /* spin */

      if (measure_time==1){ 
        dtime(&Etime1);
        time5 = Etime1 - Stime1;      
      }

    }

    /****************************************************
       Kerker mixing
    ****************************************************/

    else {

      /* find an optimum mixing weight */

      Min_Weight = Min_Mixing_weight;
      Max_Weight = Max_Mixing_weight;

      OpenMX_Mixing_Adapt(state->history, OPENMX_MIXING_DENSITY_RATIO, Min_Weight, Max_Weight);

      Mix_wgt = state->history->weight;

      for (spin=0; spin<spinmax; spin++){
        for (k=0; k<My_NumGridB_CB; k++){
         
	  weight = 1.0/(Kerker_weight[k]*Kerker_weight[k]);
	  wgt0  = Mix_wgt*weight;
	  wgt1 =  1.0 - wgt0;
        
	  ReRhok[0][spin][k] = wgt0*ReRhok[0][spin][k] + wgt1*ReRhok[1][spin][k];
	  ImRhok[0][spin][k] = wgt0*ImRhok[0][spin][k] + wgt1*ImRhok[1][spin][k];
        
	} /* k */
      } /* spin */
    } /* else */

    /****************************************************
                         shift of rho
    ****************************************************/

    if (Mix_Fast_Enabled()){

      /* pointer rotation == the legacy content shift; the legacy shift
         leaves slot 0 in place (the tail below reads it), so restore its
         content from the rotated slot 1 with one slot copy */

      MixRhok_RotateSlots(ReRhok, List_YOUSO[38]);
      MixRhok_RotateSlots(ImRhok, List_YOUSO[38]);
      for (spin=0; spin<spinmax; spin++){
        memcpy(ReRhok[0][spin], ReRhok[1][spin],
               sizeof(double)*(size_t)My_NumGridB_CB);
        memcpy(ImRhok[0][spin], ImRhok[1][spin],
               sizeof(double)*(size_t)My_NumGridB_CB);
      }
    }
    else{

      for (pSCF_iter=(List_YOUSO[38]-1); 0<pSCF_iter; pSCF_iter--){

        for (spin=0; spin<spinmax; spin++){
          for (k=0; k<My_NumGridB_CB; k++){
            ReRhok[pSCF_iter][spin][k] = ReRhok[pSCF_iter-1][spin][k];
            ImRhok[pSCF_iter][spin][k] = ImRhok[pSCF_iter-1][spin][k];
          }
        }
      }
    }

    /****************************************************
                    shift of residual rho
    ****************************************************/

    if (Mix_Fast_Enabled()){

      size_t nb = sizeof(double)*(size_t)My_NumGridB_CB;

      for (spin=0; spin<spinmax; spin++){
        memmove(&Residual_ReRhok[spin][My_NumGridB_CB], &Residual_ReRhok[spin][0],
                nb*(size_t)(List_YOUSO[38]-1));
        memmove(&Residual_ImRhok[spin][My_NumGridB_CB], &Residual_ImRhok[spin][0],
                nb*(size_t)(List_YOUSO[38]-1));
      }
    }
    else{

      for (pSCF_iter=(List_YOUSO[38]-1); 0<pSCF_iter; pSCF_iter--){

        p0 = pSCF_iter*My_NumGridB_CB;
        p1 = (pSCF_iter-1)*My_NumGridB_CB;

        for (spin=0; spin<spinmax; spin++){
          for (k=0; k<My_NumGridB_CB; k++){
            Residual_ReRhok[spin][p0+k] = Residual_ReRhok[spin][p1+k];
            Residual_ImRhok[spin][p0+k] = Residual_ImRhok[spin][p1+k];
          }
        }
      }
    }
  } /* else */

  /****************************************************
        find the charge density in real space
  ****************************************************/

  tmp0 = 1.0/(double)Ngrid1/(double)Ngrid2/(double)Ngrid3;

  for (spin=0; spin<spinmax; spin++){

    for (k=0; k<My_NumGridB_CB; k++){
      ReVk[k] = ReRhok[0][spin][k];
      ImVk[k] = ImRhok[0][spin][k];
    }

    if (spin==0 || spin==1){
      Get_Value_inReal(0,Density_Grid_B[spin],Density_Grid_B[spin],ReVk,ImVk);

      for (BN_AB=0; BN_AB<My_NumGridB_AB; BN_AB++){
	Density_Grid_B[spin][BN_AB] = Density_Grid_B[spin][BN_AB]*tmp0;
      }
    }

    else if (spin==2){

      Get_Value_inReal(1,Density_Grid_B[2],Density_Grid_B[3],ReVk,ImVk);

      for (BN_AB=0; BN_AB<My_NumGridB_AB; BN_AB++){
	Density_Grid_B[2][BN_AB] = Density_Grid_B[2][BN_AB]*tmp0;
	Density_Grid_B[3][BN_AB] = Density_Grid_B[3][BN_AB]*tmp0;
      }
    }
  }

  /******************************************************
             MPI: from the partitions B to D
  ******************************************************/

  Density_Grid_Copy_B2D(Density_Grid_B);

  /****************************************************
      set ReV2 and ImV2 which are used in Poisson.c
  ****************************************************/

  if (SpinP_switch==0){
    for (k=0; k<My_NumGridB_CB; k++){
      ReVk[k] = 2.0*ReRhok[0][0][k] - ReRhoAtomk[k];
      ImVk[k] = 2.0*ImRhok[0][0][k] - ImRhoAtomk[k];
    }
  }
  
  else {
    for (k=0; k<My_NumGridB_CB; k++){
      ReVk[k] = ReRhok[0][0][k] + ReRhok[0][1][k] - ReRhoAtomk[k];
      ImVk[k] = ImRhok[0][0][k] + ImRhok[0][1][k] - ImRhoAtomk[k];
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



void DIIS_Mixing_Rhok_NEGF(OpenMXSolverInvocation *scratch, MixingRhokState *state, int SCF_iter,
			   double Mix_wgt,
			   double ***ReRhok,
			   double ***ImRhok,
			   double **Residual_ReRhok,
			   double **Residual_ImRhok,
			   double *ReVk,
			   double *ImVk,
			   double *ReRhoAtomk,
			   double *ImRhoAtomk)
{
  int spin,Mc_AN,Gc_AN,wan1,TNO1,h_AN,Gh_AN;
  int wan2,TNO2,i,j,k,NumMix,NumSlide;
  int SCFi,SCFj,tno1,tno0,Cwan,Hwan,n1,k1,k2,k3,kk2; 
  int pSCF_iter,size_Re_OptRhok,size_Kerker_weight;
  int spinmax,MN,flag_nan,M,N,K;
  int N2D,GN,GNs,N3[4],BN_AB,p0,p1;
  double time1,time2,time3,time4,time5; 
  double Stime1,Etime1;
  double *alden,alpha,beta;
  OpenMXSolverRealTensor A,IA,My_A;
  double *My_A_thrds;
  double *My_NMat,*NMat;
  double tmp0,itmp0,tmp1,im1,im2,re1,re2;
  double OptRNorm,My_OptRNorm,x,wt;
  double dum1,dum2,bunsi,bunbo,sum,coef_OptRho;
  double Gx,Gy,Gz,G2,G12,G22,G32;
  double Min_Weight,Q,sumL,sumR;
  double Max_Weight;
  double G0,G02,G02p,weight,wgt0,wgt1;
  double sk1,sk2,sk3;
  OpenMXSolverRealTensor Re_OptRhok;
  OpenMXSolverRealTensor Im_OptRhok;
  double *Kerker_weight;
  int numprocs,myid;
  char nanchar[300];
  /* for OpenMP */
  int OMPID,Nthrds,Nprocs,Nloop,Nthrds0;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /* find an optimum G0 */

  G22 = rtv[2][1]*rtv[2][1] + rtv[2][2]*rtv[2][2] + rtv[2][3]*rtv[2][3]; 
  G32 = rtv[3][1]*rtv[3][1] + rtv[3][2]*rtv[3][2] + rtv[3][3]*rtv[3][3]; 

  if (G22<G32) G0 = G22;
  else         G0 = G32;

  G0 = sqrt(G0);
  G02 = Kerker_factor*Kerker_factor*G0*G0;
  G02p = (0.1*Kerker_factor*G0)*(0.1*Kerker_factor*G0);

  if      (SpinP_switch==0)  spinmax = 1;
  else if (SpinP_switch==1)  spinmax = 2;
  else if (SpinP_switch==3)  spinmax = 3;

  /****************************************************
    allocation of arrays:
  ****************************************************/
  
  alden = OpenMX_Solver_OverwriteReal(scratch,"rhok-DIIS_Mixing_Rhok_NEGF-alden",List_YOUSO[38]);

  My_NMat = OpenMX_Solver_OverwriteReal(scratch,"rhok-DIIS_Mixing_Rhok_NEGF-My_NMat",(size_t)List_YOUSO[38]*(size_t)List_YOUSO[38]);
  NMat = OpenMX_Solver_OverwriteReal(scratch,"rhok-DIIS_Mixing_Rhok_NEGF-NMat",(size_t)List_YOUSO[38]*(size_t)List_YOUSO[38]);

  A = OpenMX_Solver_OverwriteView(scratch,"rhok-DIIS_Mixing_Rhok_NEGF-A",List_YOUSO[38],List_YOUSO[38]);

  My_A = OpenMX_Solver_OverwriteView(scratch,"rhok-DIIS_Mixing_Rhok_NEGF-My_A",List_YOUSO[38],List_YOUSO[38]);

  IA = OpenMX_Solver_OverwriteView(scratch,"rhok-DIIS_Mixing_Rhok_NEGF-IA",List_YOUSO[38],List_YOUSO[38]);

  /* Re_OptRhok and Im_OptRhok */

  size_Re_OptRhok = spinmax*My_NumGridB_CB;

  Re_OptRhok = OpenMX_Solver_OverwriteView(scratch,"rhok-DIIS_Mixing_Rhok_NEGF-Re_OptRhok",spinmax,My_NumGridB_CB);
  for (spin=0; spin<spinmax; spin++){
    for (k=0; k<My_NumGridB_CB; k++) openmx_solver_real_matrix_row(&Re_OptRhok,spin)[k] = 0.0;
  }

  Im_OptRhok = OpenMX_Solver_OverwriteView(scratch,"rhok-DIIS_Mixing_Rhok_NEGF-Im_OptRhok",spinmax,My_NumGridB_CB);
  for (spin=0; spin<spinmax; spin++){
    for (k=0; k<My_NumGridB_CB; k++) openmx_solver_real_matrix_row(&Im_OptRhok,spin)[k] = 0.0;
  }

  size_Kerker_weight = My_NumGridB_CB;
  Kerker_weight = OpenMX_Solver_OverwriteReal(scratch,"rhok-DIIS_Mixing_Rhok_NEGF-Kerker_weight",My_NumGridB_CB);

  /***********************************
            set Kerker_weight 
  ************************************/

  N2D = Ngrid3*Ngrid2;
  GNs = ((myid*N2D+numprocs-1)/numprocs)*Ngrid1;

  if (measure_time==1) dtime(&Stime1);

  for (k=0; k<My_NumGridB_CB; k+=Ngrid1){

    /* get k3 and k2 */

    GN = k + GNs;     
    k3 = GN/(Ngrid2*Ngrid1);    
    k2 = (GN - k3*Ngrid2*Ngrid1)/Ngrid1;

    if (k2<Ngrid2/2)  sk2 = (double)k2;
    else              sk2 = (double)(k2 - Ngrid2);

    if (k3<Ngrid3/2)  sk3 = (double)k3;
    else              sk3 = (double)(k3 - Ngrid3);

    Gx = sk2*rtv[2][1] + sk3*rtv[3][1];
    Gy = sk2*rtv[2][2] + sk3*rtv[3][2]; 
    Gz = sk2*rtv[2][3] + sk3*rtv[3][3];
    G2 = Gx*Gx + Gy*Gy + Gz*Gz;

    for (k1=0; k1<Ngrid1; k1++){

      if (k2==0 && k3==0){

	sumL = 0.0;
	sumR = 0.0; 

	if (SpinP_switch==0){

	  for (n1=0; n1<k1; n1++){
	    sumL += 2.0*ReRhok[0][0][k+n1];
	  }

	  for (n1=k1+1; n1<Ngrid1; n1++){
	    sumR += 2.0*ReRhok[0][0][k+n1];
	  }
	}
        
	else if (SpinP_switch==1 || SpinP_switch==3){

	  for (n1=0; n1<k1; n1++){
	    sumL += ReRhok[0][0][k+n1] + ReRhok[0][1][k+n1];
	  }

	  for (n1=k1+1; n1<Ngrid1; n1++){
	    sumR += ReRhok[0][0][k+n1] + ReRhok[0][1][k+n1];
	  }
	}

	Q = 4.0*fabs(sumL - sumR)*GridVol + 1.0;

      }    
      else {
	Q = 1.0;
      }

      weight = (G2 + G02)/(G2 + G02p)*Q;
      Kerker_weight[k+k1] = sqrt(weight);

    } /* k1 */
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
    if (state->diis_first_negf) {
      PrintMemory("DIIS_Mixing_Rhok: Re_OptRhok",sizeof(double)*size_Re_OptRhok,NULL);
      PrintMemory("DIIS_Mixing_Rhok: Im_OptRhok",sizeof(double)*size_Re_OptRhok,NULL);
      PrintMemory("DIIS_Mixing_Rhok: Kerker_weight",sizeof(double)*size_Kerker_weight,NULL);
      state->diis_first_negf=0;
    }

    Kerker_Mixing_Rhok(state, 0, 1.0, ReRhok, ImRhok,
                       Residual_ReRhok, Residual_ImRhok,
                       ReVk,ImVk, ReRhoAtomk, ImRhoAtomk);
  } 

  else if (SCF_iter==2){

    if (measure_time==1) dtime(&Stime1);

    /* construct residual rho */

    for (spin=0; spin<spinmax; spin++){
      for (k=0; k<My_NumGridB_CB; k++){

	Residual_ReRhok[spin][2*My_NumGridB_CB+k] = (ReRhok[0][spin][k] - ReRhok[1][spin][k])*Kerker_weight[k];
	Residual_ImRhok[spin][2*My_NumGridB_CB+k] = (ImRhok[0][spin][k] - ImRhok[1][spin][k])*Kerker_weight[k];

      } /* k */
    }

    if (measure_time==1){ 
      dtime(&Etime1);
      time2 = Etime1 - Stime1;      
    }

    /* rho1 to rho2 */

    for (spin=0; spin<spinmax; spin++){
      for (k=0; k<My_NumGridB_CB; k++){
	ReRhok[2][spin][k] = ReRhok[1][spin][k];
	ImRhok[2][spin][k] = ImRhok[1][spin][k];
      }
    }

    Kerker_Mixing_Rhok(state, 0, state->history->weight, ReRhok, ImRhok,
                       Residual_ReRhok, Residual_ImRhok,
                       ReVk,ImVk, ReRhoAtomk, ImRhoAtomk);
  }

  else {

    /****************************************************
                   construct residual rho1
    ****************************************************/

    if (measure_time==1) dtime(&Stime1);

    for (spin=0; spin<spinmax; spin++){
      for (k=0; k<My_NumGridB_CB; k++){

	Residual_ReRhok[spin][My_NumGridB_CB+k] = (ReRhok[0][spin][k] - ReRhok[1][spin][k])*Kerker_weight[k];
	Residual_ImRhok[spin][My_NumGridB_CB+k] = (ImRhok[0][spin][k] - ImRhok[1][spin][k])*Kerker_weight[k];
      }
    }

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
                    alpha from residual rho
    ****************************************************/

    if (measure_time==1) dtime(&Stime1);

    /* calculation of the norm matrix for the residual vectors */

    for (i=0; i<List_YOUSO[38]*List_YOUSO[38]; i++) My_NMat[i] = 0.0;

    for (spin=0; spin<spinmax; spin++){

      M = NumMix + 1;
      N = NumMix + 1;
      K = My_NumGridB_CB;   
      alpha = 1.0;
      beta = 1.0;

      F77_NAME(dgemm,DGEMM)( "T","N", &M, &N, &K, 
                             &alpha, 
                             Residual_ReRhok[spin], &K, 
                             Residual_ReRhok[spin], &K, 
                             &beta, 
                             My_NMat,
                             &M);

      F77_NAME(dgemm,DGEMM)( "T","N", &M, &N, &K, 
                             &alpha, 
                             Residual_ImRhok[spin], &K, 
                             Residual_ImRhok[spin], &K, 
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

    /**********************************
              original code
    **********************************/

    /*
    for (SCFi=1; SCFi<=NumMix; SCFi++){
      for (SCFj=SCFi; SCFj<=NumMix; SCFj++){ 

        tmp0 = 0.0; 
	for (spin=0; spin<spinmax; spin++){
          for (k=0; k<My_NumGridB_CB; k++){

	    re1 = Residual_ReRhok[spin][SCFi*My_NumGridB_CB+k];
	    im1 = Residual_ImRhok[spin][SCFi*My_NumGridB_CB+k];

	    re2 = Residual_ReRhok[spin][SCFj*My_NumGridB_CB+k];
	    im2 = Residual_ImRhok[spin][SCFj*My_NumGridB_CB+k];

	    tmp0 += re1*re2 + im1*im2;
	  }
	}

	My_A[SCFi][SCFj] = tmp0;

      }
    }

    for (SCFi=1; SCFi<=NumMix; SCFi++){
      for (SCFj=SCFi; SCFj<=NumMix; SCFj++){
        MPI_Allreduce(&My_A[SCFi][SCFj], &A[SCFi-1][SCFj-1],
                      1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
        A[SCFj-1][SCFi-1] = A[SCFi-1][SCFj-1];
      }
    }
    */

    if (measure_time==1){ 
      dtime(&Etime1);
      time4 = Etime1 - Stime1;      
    }

    /* store NormRD */

    OpenMX_Mixing_Shift(state->history);
    state->history->norm[0] = openmx_solver_real_matrix_row(&A,0)[0];
    state->history->energy[0] = state->energies->band;

    /* solve the linear equation */

    for (SCFi=1; SCFi<=NumMix; SCFi++){
       openmx_solver_real_matrix_row(&A,SCFi-1)[NumMix] = -1.0;
       openmx_solver_real_matrix_row(&A,NumMix)[SCFi-1] = -1.0;
    }
    openmx_solver_real_matrix_row(&A,NumMix)[NumMix] = 0.0;

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

    for (pSCF_iter=1; pSCF_iter<=NumMix; pSCF_iter++){

      tmp0 =  alden[pSCF_iter]; 
      p0 = pSCF_iter*My_NumGridB_CB;

      for (spin=0; spin<spinmax; spin++){
	for (k=0; k<My_NumGridB_CB; k++){

	  /* Pulay mixing */

          openmx_solver_real_matrix_row(&Re_OptRhok,spin)[k] += tmp0*Residual_ReRhok[spin][p0+k];
          openmx_solver_real_matrix_row(&Im_OptRhok,spin)[k] += tmp0*Residual_ImRhok[spin][p0+k];

	} /* k */
      } /* spin */

    } /* pSCF_iter */

    if      (1.0e-2<=state->history->norm[0])                        coef_OptRho = 0.5;
    else if (1.0e-4<=state->history->norm[0]  && state->history->norm[0]<1.0e-2)   coef_OptRho = 0.6;
    else if (1.0e-6<=state->history->norm[0]  && state->history->norm[0]<1.0e-4)   coef_OptRho = 0.7;
    else if (1.0e-8<=state->history->norm[0]  && state->history->norm[0]<1.0e-6)   coef_OptRho = 0.8;
    else if (1.0e-10<=state->history->norm[0] && state->history->norm[0]<1.0e-8)   coef_OptRho = 0.9;
    else                                               coef_OptRho = 1.0;

    /****************************************************
                        mixing of rho 
    ****************************************************/

    /****************************************************
       Pulay mixing
    ****************************************************/

    if ( SCF_iter%EveryPulay_SCF==0 ) {

      /* reduce Mixing_weight so that charge sloshing can be avoided at the next SCF */

      state->history->weight = 0.1*Max_Mixing_weight;
      if (state->history->weight<Min_Mixing_weight) state->history->weight = Min_Mixing_weight;

      /* initial ReRhok and ImRhok */

      for (spin=0; spin<spinmax; spin++){
        for (k=0; k<My_NumGridB_CB; k++){
          ReRhok[0][spin][k] = 0.0;
          ImRhok[0][spin][k] = 0.0;
	}
      }

      /* Pulay mixing */

      for (pSCF_iter=1; pSCF_iter<=NumMix; pSCF_iter++){

	tmp0 =  alden[pSCF_iter];

        for (spin=0; spin<spinmax; spin++){
          for (k=0; k<My_NumGridB_CB; k++){

	    ReRhok[0][spin][k] += tmp0*ReRhok[pSCF_iter][spin][k];
	    ImRhok[0][spin][k] += tmp0*ImRhok[pSCF_iter][spin][k];

	  } /* k */
	} /* spin */
      } /* pSCF_iter */

      /* Correction by optimized residual rho */

      for (spin=0; spin<spinmax; spin++){
        for (k=0; k<My_NumGridB_CB; k++){
 
	  weight = 1.0/(Kerker_weight[k]*Kerker_weight[k]);

          ReRhok[0][spin][k] += coef_OptRho*openmx_solver_real_matrix_row(&Re_OptRhok,spin)[k]*weight;
          ImRhok[0][spin][k] += coef_OptRho*openmx_solver_real_matrix_row(&Im_OptRhok,spin)[k]*weight;

	  /* correction to large changing components */

          tmp0 = ReRhok[0][spin][k] - ReRhok[1][spin][k];  
          tmp1 = ImRhok[0][spin][k] - ImRhok[1][spin][k];  

          if ( maxima_step<(fabs(tmp0)+fabs(tmp1)) ){
            ReRhok[0][spin][k] = sgn(tmp0)*maxima_step + ReRhok[1][spin][k]; 
            ImRhok[0][spin][k] = sgn(tmp1)*maxima_step + ImRhok[1][spin][k]; 
          }

	} /* k */
      } /* spin */

      if (measure_time==1){ 
        dtime(&Etime1);
        time5 = Etime1 - Stime1;      
      }

    }

    /****************************************************
       Kerker mixing
    ****************************************************/

    else {

      /* find an optimum mixing weight */

      Min_Weight = Min_Mixing_weight;
      Max_Weight = Max_Mixing_weight;

      OpenMX_Mixing_Adapt(state->history, OPENMX_MIXING_DENSITY_RATIO, Min_Weight, Max_Weight);

      /* Kerer mixing */

      Mix_wgt = state->history->weight;

      for (spin=0; spin<spinmax; spin++){
        for (k=0; k<My_NumGridB_CB; k++){

	  weight = 1.0/(Kerker_weight[k]*Kerker_weight[k]);
	  wgt0  = Mix_wgt*weight;
	  wgt1 =  1.0 - wgt0;

	  ReRhok[0][spin][k] = wgt0*ReRhok[0][spin][k] + wgt1*ReRhok[1][spin][k];
	  ImRhok[0][spin][k] = wgt0*ImRhok[0][spin][k] + wgt1*ImRhok[1][spin][k];

	} /* k */
      } /* spin */
    } /* else */

    /****************************************************
                        shift of rho
    ****************************************************/

    for (pSCF_iter=(List_YOUSO[38]-1); 0<pSCF_iter; pSCF_iter--){
      for (spin=0; spin<spinmax; spin++){
        for (k=0; k<My_NumGridB_CB; k++){
	  ReRhok[pSCF_iter][spin][k] = ReRhok[pSCF_iter-1][spin][k]; 
	  ImRhok[pSCF_iter][spin][k] = ImRhok[pSCF_iter-1][spin][k]; 
	}
      }
    }

    /****************************************************
                    shift of residual rho
    ****************************************************/

    for (pSCF_iter=(List_YOUSO[38]-1); 0<pSCF_iter; pSCF_iter--){

      p0 = pSCF_iter*My_NumGridB_CB;
      p1 = (pSCF_iter-1)*My_NumGridB_CB; 

      for (spin=0; spin<spinmax; spin++){
        for (k=0; k<My_NumGridB_CB; k++){
	  Residual_ReRhok[spin][p0+k] = Residual_ReRhok[spin][p1+k];
	  Residual_ImRhok[spin][p0+k] = Residual_ImRhok[spin][p1+k]; 
	}
      }
    }

  } /* else */

  /****************************************************
        find the charge density in real space 
  ****************************************************/

  for (spin=0; spin<spinmax; spin++){

    for (k=0; k<My_NumGridB_CB; k++){
      ReVk[k] = ReRhok[0][spin][k];
      ImVk[k] = ImRhok[0][spin][k];
    }
/* revised by Y. Xiao for Noncollinear NEGF calculations */
  if (spin==0 || spin==1) {
    Get_Value_inReal2D(0, Density_Grid_B[spin], NULL, ReVk, ImVk);
  } else {
    Get_Value_inReal2D(1, Density_Grid_B[2], Density_Grid_B[3], ReVk, ImVk);
  }
/* until here by Y. Xiao for Noncollinear NEGF calculations */

/*    Get_Value_inReal2D(0, Density_Grid_B[spin], NULL, ReVk, ImVk); */
  }

  /******************************************************
             MPI: from the partitions B to D
  ******************************************************/

  Density_Grid_Copy_B2D(Density_Grid_B);

  if (measure_time==1){ 
    printf("time1=%12.6f time2=%12.6f time3=%12.6f time4=%12.6f time5=%12.6f\n",
            time1,time2,time3,time4,time5);
  }

  /****************************************************
    freeing of arrays:
  ****************************************************/
  
















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

    x = OpenMX_Solver_OverwriteReal(scratch,"rhok-Inverse-x",List_YOUSO[38]);
    y = OpenMX_Solver_OverwriteReal(scratch,"rhok-Inverse-y",List_YOUSO[38]);
    for (i=0; i<List_YOUSO[38]; i++){
      x[i] = 0.0;
      y[i] = 0.0;
    }

    da = OpenMX_Solver_OverwriteView(scratch,"rhok-Inverse-da",List_YOUSO[38],List_YOUSO[38]);
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
    OpenMXRealLuOperations operations;
    operations.context=NULL;
    operations.factor=OpenMX_Mixing_LuFactor;
    operations.invert=OpenMX_Mixing_LuInvert;
    OpenMX_Solver_InversePivoted(scratch,n,a,ia,&operations);
  }

  else if (method_flag==2){

    int N,i,j,k;
    double *A,*B,*ko;
    double sum;

    N = n + 1;

    A = OpenMX_Solver_OverwriteReal(scratch,"rhok-Inverse-eigen-A",(size_t)(N+2)*(size_t)(N+2));
    B = OpenMX_Solver_OverwriteReal(scratch,"rhok-Inverse-eigen-B",(size_t)(N+2)*(size_t)(N+2));
    ko = OpenMX_Solver_OverwriteReal(scratch,"rhok-Inverse-eigen-ko",(N+2));

    /*
    printf("N=%2d\n",N);

    N = 3;

    a[0][0] =-1.2; a[0][1] =-1.0; a[0][2] = -0.0;
    a[1][0] =-1.0; a[1][1] =-1.2; a[1][2] = -1.0;
    a[2][0] =-0.0; a[2][1] =-1.0; a[2][2] = -1.2;
    */

    for (i=0; i<N; i++){
      for (j=0; j<N; j++){
        A[j*N+i] = openmx_solver_real_matrix_row(a,i)[j];
      }
    }

    Eigen_lapack3(A, ko, N, N); 

    /*
    printf("ko\n");
    for (i=0; i<N; i++){
      printf("ko i=%2d %15.12f\n",i,ko[i]);
    } 

    printf("A\n");
    for (i=0; i<N; i++){
      for (j=0; j<N; j++){
        printf("%10.5f ",A[i*N+j]); 
      }
      printf("\n");
    }
    */

    for (i=0; i<N; i++){
      /*
      printf("i=%2d ko=%18.15f iko=%18.15f\n",i,ko[i],1.0/ko[i]);
      */

      ko[i] = 1.0/(ko[i]+1.0e-13);
    } 

    for (i=0; i<N; i++){
      for (j=0; j<N; j++){
        B[i*N+j] = A[i*N+j]*ko[i];
      }
    }

    /*
    printf("A*ko\n");
    for (i=0; i<N; i++){
      for (j=0; j<N; j++){
        printf("%10.5f ",B[i*N+j]); 
      }
      printf("\n");
    }
    */

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

    /*	
    printf("ia\n");
    for (i=0; i<N; i++){
      for (j=0; j<N; j++){
        printf("%10.5f ",ia[i][j]);
      }
      printf("\n");
    }

    MPI_Finalize();
    exit(0);
    */




  }
}


void Inverse2(OpenMXSolverInvocation *scratch, int n, const OpenMXSolverRealTensor *a, const OpenMXSolverRealTensor *ia)
{
  int N,i,j,k;
  double *A,*B,*ko;
  double sum;

  N = n + 1;

  A = OpenMX_Solver_OverwriteReal(scratch,"rhok-Inverse2-eigen-A",(size_t)(N+2)*(size_t)(N+2));
  B = OpenMX_Solver_OverwriteReal(scratch,"rhok-Inverse2-eigen-B",(size_t)(N+2)*(size_t)(N+2));
  ko = OpenMX_Solver_OverwriteReal(scratch,"rhok-Inverse2-eigen-ko",(N+2));

  /*
    printf("N=%2d\n",N);

    N = 3;

    a[0][0] =-1.2; a[0][1] =-1.0; a[0][2] = -0.0;
    a[1][0] =-1.0; a[1][1] =-1.2; a[1][2] = -1.0;
    a[2][0] =-0.0; a[2][1] =-1.0; a[2][2] = -1.2;
  */

  for (i=0; i<N; i++){
    for (j=0; j<N; j++){
      A[j*N+i] = openmx_solver_real_matrix_row(a,i)[j];
    }
  }

  Eigen_lapack3(A, ko, N, N); 

  /*
    printf("ko\n");
    for (i=0; i<N; i++){
    printf("ko i=%2d %15.12f\n",i,ko[i]);
    } 

    printf("A\n");
    for (i=0; i<N; i++){
    for (j=0; j<N; j++){
    printf("%10.5f ",A[i*N+j]); 
    }
    printf("\n");
    }
  */

  for (i=0; i<N; i++){

    printf("i=%2d ko=%18.15f iko=%18.15f\n",i,ko[i],1.0/ko[i]);

    ko[i] = 1.0/ko[i];
  } 
 
  for (i=0; i<N; i++){
    for (j=0; j<N; j++){
      B[i*N+j] = A[i*N+j]*ko[i];
    }
  }

  /*
    printf("A*ko\n");
    for (i=0; i<N; i++){
    for (j=0; j<N; j++){
    printf("%10.5f ",B[i*N+j]); 
    }
    printf("\n");
    }
  */

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

  /*	
    printf("ia\n");
    for (i=0; i<N; i++){
    for (j=0; j<N; j++){
    printf("%10.5f ",ia[i][j]);
    }
    printf("\n");
    }

    MPI_Finalize();
    exit(0);
  */





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

  x = OpenMX_Solver_OverwriteComplex(scratch,"rhok-Complex_Inverse-x",List_YOUSO[38]);
  y = OpenMX_Solver_OverwriteComplex(scratch,"rhok-Complex_Inverse-y",List_YOUSO[38]);

  da = OpenMX_Solver_OverwriteComplexView(scratch,"rhok-Complex_Inverse-da",List_YOUSO[38],List_YOUSO[38]);


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

static void MixRhok_Reset(OpenMXNativeResources *owner, void *metadata)
{
  MixingRhokState *state=(MixingRhokState*)metadata;
  (void)owner;
  /* The group releases every registered allocation after this callback. */
  memset(state,0,sizeof(*state));
}

OpenMXMixingRhokState *Mixing_Rhok_Bind_Resources(OpenMXCoreContext *core)
{
  OpenMXNativeResources *owner=NULL;
  void *metadata=NULL;
  char error[256];
  if (!openmx_core_native_resources_state(core,"Mixing_Rhok",sizeof(MixingRhokState),
      MixRhok_Reset,&owner,&metadata,error,sizeof(error))) {
    fprintf(stderr,"OpenMX density mixing state: %s\n",error);
    abort();
  }
  MixingRhokState *state=(MixingRhokState*)metadata;
  if (!state->initialized) {
    state->resources=owner;
    if(!openmx_core_mixing_history(core,&state->history,error,sizeof(error))) {
      fprintf(stderr,"OpenMX mixing history: %s\n",error);
      abort();
    }
    state->fast.last_scf_iter=-1;
    state->diis_first_normal=state->diis_first_negf=1;
    state->kerker_first_normal=state->kerker_first_negf=1;
    if(!openmx_core_band_energies(core,&state->energies,error,sizeof(error))) {
      fprintf(stderr,"OpenMX band energies: %s\n",error);
      abort();
    }
    state->initialized=1;
  }
  return state;
}

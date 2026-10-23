#include "openmx_gpu_backend.h"
#include "openmx_gpu_device.h"
#include "openmx_gpu_runtime.h"
#if OPENMX_GPU_HIP
#include "openmx_target.h"
#endif
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "mpi.h"
#include "openmx_common.h"
#include "openmx_env.h"
#include "openmx_gpu_workers.h"
#include "openmx_gpu_vector.h"
#include "openmx_host_memory.h"
#include "openmx_native_resources.h"
#include "openmx_mixing_h_state.h"
#include "openmx_solver_workspace.h"
#include <limits.h>

#include "lapack_prototypes.h"


/* Persistent metadata belongs to the explicit calculation owner. */
typedef struct {
  int      verdict;          /* -1 unknown, 0 CPU fallback, 1 GPU; sticky per fingerprint */
  int      valid;            /* device ring mirrors the host residual history */
  int      announced;
  int      fp_md_iter;       /* fingerprint of the sticky verdict/layout */
  int      fp_matomnum;
  int      fp_spinp;
  int      fp_nslot;
  size_t   fp_L;
  int      last_scf_iter;
  int      nslot;            /* ring slots == Num_Mixing_pDM */
  size_t   L;                /* flat doubles per history slot on this rank */
  size_t   L_plane;          /* collinear: doubles per spin plane; NC: == L */
  unsigned char *base;       /* single cudaMalloc arena */
  size_t   total_bytes;
  size_t   o_ring, o_w, o_r, o_pair;
  int     *perm;             /* logical slot -> physical ring slot */
  size_t  *atom_off;         /* [Matomnum+1] flat offset of each atom block within a plane */
  double  *h_flat;           /* host staging buffer (L doubles) */
  double  *h_r;              /* downloaded optimum residual (L doubles) */
} MixH_GpuState;

typedef struct {
  unsigned char *base;
  size_t first,L,o_ring,o_w,o_r,o_pair,bytes;
  size_t available,total;
} MixH_GpuShard;

typedef struct {
  int      valid;
  int      last_scf_iter;
  int      nslot;
  int      nblk;         /* sum over local atoms of Spe_Total_NO */
  int     *blk_base;     /* [Matomnum+1]: first block id of each atom */
  int     *perm;         /* logical slot -> physical slot */
  double  *S;            /* [nslot][nslot][nblk], physical slot indexing */
  double  *wblk;         /* [nblk] metric weights of this iteration */
} MixH_IncState;

struct OpenMXMixingHState {
  OpenMXNativeResources *resources; /* borrowed from this calculation */
  OpenMXMixingHistory *history; /* shared by this calculation's mixing methods */
  int initialized;
  MixH_GpuState gpu;
  MixH_GpuShard shard[OPENMX_GPU_WORKERS_MAX];
  OpenMXGpuWorkers workers;
  double *host_ring, *host_weight;
  size_t capacity_reserve;
  MixH_IncState inc;
};
typedef OpenMXMixingHState MixingHState;

static void Simple_Mixing_H(OpenMXMixingHistory *history, int MD_iter, int SCF_iter, int SCF_iter0 , const OpenMXElectronicViews *electronic);
static void Pulay_Mixing_H(OpenMXSolverInvocation *scratch, MixingHState *state, int MD_iter, int SCF_iter, int SCF_iter0 , const OpenMXElectronicViews *electronic);
static void Pulay_Mixing_H_MultiSecant(OpenMXSolverInvocation *scratch, int MD_iter, int SCF_iter, int SCF_iter0 , const OpenMXElectronicViews *electronic);
static void Pulay_Mixing_H_with_One_Shot_Hessian(OpenMXSolverInvocation *scratch, int MD_iter, int SCF_iter, int SCF_iter0 , const OpenMXElectronicViews *electronic);
static void Inverse(OpenMXSolverInvocation *scratch, int n, const OpenMXSolverRealTensor *a, const OpenMXSolverRealTensor *ia);


typedef struct {
  MixingHState *state;
  int MD_iter, SCF_iter, SCF_iter0;
  const OpenMXElectronicViews *electronic;
} MixH_PulayArguments;

static int MixH_PulayOperation(OpenMXSolverInvocation *scratch, void *opaque, double *result)
{
  const MixH_PulayArguments *a=(const MixH_PulayArguments*)opaque;
  Pulay_Mixing_H(scratch, a->state, a->MD_iter, a->SCF_iter, a->SCF_iter0, a->electronic);
  *result=0.0;
  return 1;
}

double Mixing_H(MixingHState *state,  int MD_iter, int SCF_iter, int SCF_iter0 , const OpenMXElectronicViews *electronic)
{
  double time0;
  double TStime,TEtime;
  int numprocs,myid,ID;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  MPI_Barrier(mpi_comm_level1);
  dtime(&TStime);

  /*******************************************************
    Simple mixing 
  *******************************************************/

  if ( SCF_iter<=(Pulay_SCF-1) ){
    Simple_Mixing_H(state->history, MD_iter, SCF_iter, SCF_iter0 , electronic);
  }

  /*******************************************************
    Pulay's method:
    Residual Minimazation Method (RMM) using
    Direct Inversion in the Iterative Subspace (DIIS)
  *******************************************************/

  else{

    MixH_PulayArguments arguments={state,MD_iter,SCF_iter,SCF_iter0,electronic};
    (void)OpenMX_Solver_Run(MixH_PulayOperation,&arguments);

    /*
    Pulay_Mixing_H_with_One_Shot_Hessian( MD_iter, SCF_iter, SCF_iter0 );
    */

    /*
    Pulay_Mixing_H_MultiSecant( MD_iter, SCF_iter, SCF_iter0 );
    */
  }

  /* if SCF_iter0==1, then NormRD[0]=1 */
  if (SCF_iter0==1) NormRD[0] = 1.0;

  MPI_Barrier(mpi_comm_level1);
  dtime(&TEtime);
  time0 = TEtime - TStime;
  return time0;
} 





void Pulay_Mixing_H_MultiSecant(OpenMXSolverInvocation *scratch, int MD_iter, int SCF_iter, int SCF_iter0 , const OpenMXElectronicViews *electronic)
{
  double *****H=electronic->H;
  double ******HisH1=electronic->HisH1;
  double ******HisH2=electronic->HisH2;
  double ******ResidualH1=electronic->ResidualH1;
  double ******ResidualH2=electronic->ResidualH2;
  double *****iHNL=electronic->iHNL;

  int Mc_AN,Gc_AN,Cwan,Hwan,h_AN,Gh_AN,i,j,spin;
  int dim,m,n,flag_nan;
  double sum,my_sum,tmp1,tmp2,alpha;
  double r,r10,r11,r12,r13,r20,r21,r22;
  double h,h10,h11,h12,h13,h20,h21,h22;
  double my_sy,my_yy,sy,yy,norm,s,y,or,al,be;
  OpenMXSolverRealTensor A,IA;
  double *coes,*coes2,*ror;
  char nanchar[300];

  /****************************************************
       determination of dimension of the subspace
  ****************************************************/

  if (SCF_iter<=Num_Mixing_pDM) dim = SCF_iter-1;
  else                          dim = Num_Mixing_pDM;

  /****************************************************
                allocation of arrays 
  ****************************************************/

  coes = OpenMX_Solver_OverwriteReal(scratch,"mixh-Pulay_Mixing_H_MultiSecant-coes",List_YOUSO[39]);
  coes2 = OpenMX_Solver_OverwriteReal(scratch,"mixh-Pulay_Mixing_H_MultiSecant-coes2",List_YOUSO[39]);
  ror = OpenMX_Solver_OverwriteReal(scratch,"mixh-Pulay_Mixing_H_MultiSecant-ror",List_YOUSO[39]);

  A = OpenMX_Solver_OverwriteView(scratch,"mixh-Pulay_Mixing_H_MultiSecant-A",List_YOUSO[39],List_YOUSO[39]);

  IA = OpenMX_Solver_OverwriteView(scratch,"mixh-Pulay_Mixing_H_MultiSecant-IA",List_YOUSO[39],List_YOUSO[39]);

  /****************************************************
                 shift the residual H
  ****************************************************/

  if (SpinP_switch==3){

    /* shift the residual Hamiltonian */

    for (m=dim; 0<m; m--){
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	Gc_AN = M2G[Mc_AN];    
	Cwan = WhatSpecies[Gc_AN];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  for (i=0; i<Spe_Total_NO[Cwan]; i++){
	    for (j=0; j<Spe_Total_NO[Hwan]; j++){

	      ResidualH1[m][0][Mc_AN][h_AN][i][j] = ResidualH1[m-1][0][Mc_AN][h_AN][i][j];
	      ResidualH1[m][1][Mc_AN][h_AN][i][j] = ResidualH1[m-1][1][Mc_AN][h_AN][i][j];
	      ResidualH1[m][2][Mc_AN][h_AN][i][j] = ResidualH1[m-1][2][Mc_AN][h_AN][i][j];
	      ResidualH1[m][3][Mc_AN][h_AN][i][j] = ResidualH1[m-1][3][Mc_AN][h_AN][i][j];

	      ResidualH2[m][0][Mc_AN][h_AN][i][j] = ResidualH2[m-1][0][Mc_AN][h_AN][i][j];
	      ResidualH2[m][1][Mc_AN][h_AN][i][j] = ResidualH2[m-1][1][Mc_AN][h_AN][i][j];
	      ResidualH2[m][2][Mc_AN][h_AN][i][j] = ResidualH2[m-1][2][Mc_AN][h_AN][i][j];
	    }
	  }
	}
      }
    }

    /* calculate the current residual Hamiltonian */

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];    
      Cwan = WhatSpecies[Gc_AN];
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	Gh_AN = natn[Gc_AN][h_AN];
	Hwan = WhatSpecies[Gh_AN];
	for (i=0; i<Spe_Total_NO[Cwan]; i++){
	  for (j=0; j<Spe_Total_NO[Hwan]; j++){

	    ResidualH1[0][0][Mc_AN][h_AN][i][j] = H[0][Mc_AN][h_AN][i][j] - HisH1[0][0][Mc_AN][h_AN][i][j];
	    ResidualH1[0][1][Mc_AN][h_AN][i][j] = H[1][Mc_AN][h_AN][i][j] - HisH1[0][1][Mc_AN][h_AN][i][j];
	    ResidualH1[0][2][Mc_AN][h_AN][i][j] = H[2][Mc_AN][h_AN][i][j] - HisH1[0][2][Mc_AN][h_AN][i][j];
	    ResidualH1[0][3][Mc_AN][h_AN][i][j] = H[3][Mc_AN][h_AN][i][j] - HisH1[0][3][Mc_AN][h_AN][i][j];

	    ResidualH2[0][0][Mc_AN][h_AN][i][j] = iHNL[0][Mc_AN][h_AN][i][j] - HisH2[0][0][Mc_AN][h_AN][i][j];
	    ResidualH2[0][1][Mc_AN][h_AN][i][j] = iHNL[1][Mc_AN][h_AN][i][j] - HisH2[0][1][Mc_AN][h_AN][i][j];
	    ResidualH2[0][2][Mc_AN][h_AN][i][j] = iHNL[2][Mc_AN][h_AN][i][j] - HisH2[0][2][Mc_AN][h_AN][i][j];
	  }
	}
      }
    }

  }

  else{

    /* shift the residual Hamiltonian */

    for (m=dim; 0<m; m--){
      for (spin=0; spin<=SpinP_switch; spin++){
	for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	  Gc_AN = M2G[Mc_AN];    
	  Cwan = WhatSpecies[Gc_AN];
	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	    Gh_AN = natn[Gc_AN][h_AN];
	    Hwan = WhatSpecies[Gh_AN];
	    for (i=0; i<Spe_Total_NO[Cwan]; i++){
	      for (j=0; j<Spe_Total_NO[Hwan]; j++){
		ResidualH1[m][spin][Mc_AN][h_AN][i][j] = ResidualH1[m-1][spin][Mc_AN][h_AN][i][j];
	      }
	    }
	  }
	}
      }
    }

    /* calculate the current residual Hamiltonian */

    for (spin=0; spin<=SpinP_switch; spin++){
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	Gc_AN = M2G[Mc_AN];    
	Cwan = WhatSpecies[Gc_AN];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  for (i=0; i<Spe_Total_NO[Cwan]; i++){
	    for (j=0; j<Spe_Total_NO[Hwan]; j++){
  	      ResidualH1[0][spin][Mc_AN][h_AN][i][j] = H[spin][Mc_AN][h_AN][i][j] - HisH1[0][spin][Mc_AN][h_AN][i][j];
	    }
	  }
	}
      }
    }

  } /* else */

  /****************************************************
          calculation of the residual matrix
  ****************************************************/

  for (m=0; m<dim; m++){
    for (n=0; n<dim; n++){

      my_sum = 0.0;

      if (SpinP_switch==3){

	for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	  Gc_AN = M2G[Mc_AN];    
	  Cwan = WhatSpecies[Gc_AN];
	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	    Gh_AN = natn[Gc_AN][h_AN];
	    Hwan = WhatSpecies[Gh_AN];
	    for (i=0; i<Spe_Total_NO[Cwan]; i++){
	      for (j=0; j<Spe_Total_NO[Hwan]; j++){

		tmp1 = ResidualH1[m][0][Mc_AN][h_AN][i][j];
		tmp2 = ResidualH1[n][0][Mc_AN][h_AN][i][j];
                my_sum += tmp1*tmp2; 

		tmp1 = ResidualH1[m][1][Mc_AN][h_AN][i][j];
		tmp2 = ResidualH1[n][1][Mc_AN][h_AN][i][j];
                my_sum += tmp1*tmp2; 

		tmp1 = ResidualH1[m][2][Mc_AN][h_AN][i][j];
		tmp2 = ResidualH1[n][2][Mc_AN][h_AN][i][j];
                my_sum += tmp1*tmp2; 

		tmp1 = ResidualH1[m][3][Mc_AN][h_AN][i][j];
		tmp2 = ResidualH1[n][3][Mc_AN][h_AN][i][j];
                my_sum += tmp1*tmp2; 

		tmp1 = ResidualH2[m][0][Mc_AN][h_AN][i][j];
		tmp2 = ResidualH2[n][0][Mc_AN][h_AN][i][j];
                my_sum += tmp1*tmp2; 

		tmp1 = ResidualH2[m][1][Mc_AN][h_AN][i][j];
		tmp2 = ResidualH2[n][1][Mc_AN][h_AN][i][j];
                my_sum += tmp1*tmp2; 

		tmp1 = ResidualH2[m][2][Mc_AN][h_AN][i][j];
		tmp2 = ResidualH2[n][2][Mc_AN][h_AN][i][j];
                my_sum += tmp1*tmp2; 
	      }
	    }
	  }
	}

      } /* if (SpinP_switch==3 */

      else{

	for (spin=0; spin<=SpinP_switch; spin++){
	  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	    Gc_AN = M2G[Mc_AN];    
	    Cwan = WhatSpecies[Gc_AN];
	    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	      Gh_AN = natn[Gc_AN][h_AN];
	      Hwan = WhatSpecies[Gh_AN];
	      for (i=0; i<Spe_Total_NO[Cwan]; i++){
		for (j=0; j<Spe_Total_NO[Hwan]; j++){
		  tmp1 = ResidualH1[m][spin][Mc_AN][h_AN][i][j];
		  tmp2 = ResidualH1[n][spin][Mc_AN][h_AN][i][j];
                  my_sum += tmp1*tmp2; 
		}
	      }
	    }
	  }
	}

      } /* else */

      MPI_Allreduce(&my_sum, &openmx_solver_real_matrix_row(&A,m)[n], 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
      openmx_solver_real_matrix_row(&A,n)[m] = openmx_solver_real_matrix_row(&A,m)[n];

    } /* n */
  } /* m */

  NormRD[0] = openmx_solver_real_matrix_row(&A,0)[0]/(double)atomnum;

  for (m=1; m<=dim; m++){
    openmx_solver_real_matrix_row(&A,m-1)[dim] = -1.0;
    openmx_solver_real_matrix_row(&A,dim)[m-1] = -1.0;
  }
  openmx_solver_real_matrix_row(&A,dim)[dim] = 0.0;

  Inverse(scratch, dim,&A,&IA);

  for (m=1; m<=dim; m++){
    coes[m] = -openmx_solver_real_matrix_row(&IA,m-1)[dim];
  }

  /****************************************************
            check "nan", "NaN", "inf" or "Inf"
  ****************************************************/

  flag_nan = 0;
  for (m=1; m<=dim; m++){

    sprintf(nanchar,"%8.4f",coes[m]);
    if (   strstr(nanchar,"nan")!=NULL || strstr(nanchar,"NaN")!=NULL 
	|| strstr(nanchar,"inf")!=NULL || strstr(nanchar,"Inf")!=NULL){

      flag_nan = 1;
    }
  }

  if (flag_nan==1){
    for (m=1; m<=dim; m++){
      coes[m] = 0.0;
    }
    coes[1] = 0.05;
    coes[2] = 0.95;
  }

  /****************************************************
      calculation of optimum residual Hamiltonian
  ****************************************************/

  if (SpinP_switch==3){

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];    
      Cwan = WhatSpecies[Gc_AN];
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	Gh_AN = natn[Gc_AN][h_AN];
	Hwan = WhatSpecies[Gh_AN];
	for (i=0; i<Spe_Total_NO[Cwan]; i++){
	  for (j=0; j<Spe_Total_NO[Hwan]; j++){

            r10 = 0.0; 
            r11 = 0.0;
            r12 = 0.0;
            r13 = 0.0;

            r20 = 0.0; 
            r21 = 0.0;
            r22 = 0.0;

	    for (m=0; m<dim; m++){

	      r10 += ResidualH1[m][0][Mc_AN][h_AN][i][j]*coes[m+1];
	      r11 += ResidualH1[m][1][Mc_AN][h_AN][i][j]*coes[m+1];
	      r12 += ResidualH1[m][2][Mc_AN][h_AN][i][j]*coes[m+1];
	      r13 += ResidualH1[m][3][Mc_AN][h_AN][i][j]*coes[m+1];

	      r20 += ResidualH2[m][0][Mc_AN][h_AN][i][j]*coes[m+1];
	      r21 += ResidualH2[m][1][Mc_AN][h_AN][i][j]*coes[m+1];
	      r22 += ResidualH2[m][2][Mc_AN][h_AN][i][j]*coes[m+1];
	    }

            /* optimum Residual H is stored in ResidualH1[dim+1] and ResidualH2[dim+1] */

	    ResidualH1[dim+1][0][Mc_AN][h_AN][i][j] = r10;
	    ResidualH1[dim+1][1][Mc_AN][h_AN][i][j] = r11;
	    ResidualH1[dim+1][2][Mc_AN][h_AN][i][j] = r12;
	    ResidualH1[dim+1][3][Mc_AN][h_AN][i][j] = r13;

	    ResidualH2[dim+1][0][Mc_AN][h_AN][i][j] = r20;
	    ResidualH2[dim+1][1][Mc_AN][h_AN][i][j] = r21;
	    ResidualH2[dim+1][2][Mc_AN][h_AN][i][j] = r22;

	  }
	}
      }
    }

  }

  else{

    for (spin=0; spin<=SpinP_switch; spin++){
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	Gc_AN = M2G[Mc_AN];    
	Cwan = WhatSpecies[Gc_AN];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  for (i=0; i<Spe_Total_NO[Cwan]; i++){
	    for (j=0; j<Spe_Total_NO[Hwan]; j++){

	      r = 0.0; 
	      for (m=0; m<dim; m++){
		r += ResidualH1[m][spin][Mc_AN][h_AN][i][j]*coes[m+1];
	      }

              /* optimum Residual H is stored in ResidualH1[dim+1] */

              ResidualH1[dim+1][spin][Mc_AN][h_AN][i][j] = r;

	    }
	  }
	}
      }
    }
  }

  /******************************************************
   calculations of inner products of <s0|y0> and <y0|y0>
   in order to estimate the parameter "al".
  ******************************************************/

  my_sy = 0.0;
  my_yy = 0.0;

  if (SpinP_switch==3){

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];    
      Cwan = WhatSpecies[Gc_AN];
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	Gh_AN = natn[Gc_AN][h_AN];
	Hwan = WhatSpecies[Gh_AN];
	for (i=0; i<Spe_Total_NO[Cwan]; i++){
	  for (j=0; j<Spe_Total_NO[Hwan]; j++){

	    tmp1 = HisH1[0][0][Mc_AN][h_AN][i][j] - HisH1[1][0][Mc_AN][h_AN][i][j];           /* s */
	    tmp2 = ResidualH1[0][0][Mc_AN][h_AN][i][j] - ResidualH1[1][0][Mc_AN][h_AN][i][j]; /* y */
	    my_sy += tmp1*tmp2; 
	    my_yy += tmp2*tmp2; 

	    tmp1 = HisH1[0][1][Mc_AN][h_AN][i][j] - HisH1[1][1][Mc_AN][h_AN][i][j];           /* s */
	    tmp2 = ResidualH1[0][1][Mc_AN][h_AN][i][j] - ResidualH1[1][1][Mc_AN][h_AN][i][j]; /* y */
	    my_sy += tmp1*tmp2; 
	    my_yy += tmp2*tmp2; 

	    tmp1 = HisH1[0][2][Mc_AN][h_AN][i][j] - HisH1[1][2][Mc_AN][h_AN][i][j];           /* s */
	    tmp2 = ResidualH1[0][2][Mc_AN][h_AN][i][j] - ResidualH1[1][2][Mc_AN][h_AN][i][j]; /* y */
	    my_sy += tmp1*tmp2; 
	    my_yy += tmp2*tmp2; 

	    tmp1 = HisH1[0][3][Mc_AN][h_AN][i][j] - HisH1[1][3][Mc_AN][h_AN][i][j];           /* s */
	    tmp2 = ResidualH1[0][3][Mc_AN][h_AN][i][j] - ResidualH1[1][3][Mc_AN][h_AN][i][j]; /* y */
	    my_sy += tmp1*tmp2; 
	    my_yy += tmp2*tmp2; 

	    tmp1 = HisH2[0][0][Mc_AN][h_AN][i][j] - HisH2[1][0][Mc_AN][h_AN][i][j];           /* s */
	    tmp2 = ResidualH2[0][0][Mc_AN][h_AN][i][j] - ResidualH2[1][0][Mc_AN][h_AN][i][j]; /* y */
	    my_sy += tmp1*tmp2; 
	    my_yy += tmp2*tmp2; 

	    tmp1 = HisH2[0][1][Mc_AN][h_AN][i][j] - HisH2[1][1][Mc_AN][h_AN][i][j];           /* s */
	    tmp2 = ResidualH2[0][1][Mc_AN][h_AN][i][j] - ResidualH2[1][1][Mc_AN][h_AN][i][j]; /* y */
	    my_sy += tmp1*tmp2; 
	    my_yy += tmp2*tmp2; 

	    tmp1 = HisH2[0][2][Mc_AN][h_AN][i][j] - HisH2[1][2][Mc_AN][h_AN][i][j];           /* s */
	    tmp2 = ResidualH2[0][2][Mc_AN][h_AN][i][j] - ResidualH2[1][2][Mc_AN][h_AN][i][j]; /* y */
	    my_sy += tmp1*tmp2; 
	    my_yy += tmp2*tmp2; 
	  }
	}
      }
    }

  } /* if (SpinP_switch==3 */

  else{

    for (spin=0; spin<=SpinP_switch; spin++){
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	Gc_AN = M2G[Mc_AN];    
	Cwan = WhatSpecies[Gc_AN];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  for (i=0; i<Spe_Total_NO[Cwan]; i++){
	    for (j=0; j<Spe_Total_NO[Hwan]; j++){

	      tmp1 = HisH1[0][spin][Mc_AN][h_AN][i][j] - HisH1[1][spin][Mc_AN][h_AN][i][j];           /* s */
	      tmp2 = ResidualH1[0][spin][Mc_AN][h_AN][i][j] - ResidualH1[1][spin][Mc_AN][h_AN][i][j]; /* y */
	      my_sy += tmp1*tmp2; 
	      my_yy += tmp2*tmp2; 
	    }
	  }
	}
      }
    }

  } /* else */

  MPI_Allreduce(&my_sy, &sy, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
  MPI_Allreduce(&my_yy, &yy, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

  /* al < sy/yy */

  al = sy/yy - 0.2;

  /****************************************************
         calculations of inner products of <r|y> 
  ****************************************************/

  for (m=0; m<dim; m++){
    for (n=0; n<dim; n++){

      my_sum = 0.0;

      if (SpinP_switch==3){

	for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	  Gc_AN = M2G[Mc_AN];    
	  Cwan = WhatSpecies[Gc_AN];
	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	    Gh_AN = natn[Gc_AN][h_AN];
	    Hwan = WhatSpecies[Gh_AN];
	    for (i=0; i<Spe_Total_NO[Cwan]; i++){
	      for (j=0; j<Spe_Total_NO[Hwan]; j++){

                /* m */
		s = HisH1[m][0][Mc_AN][h_AN][i][j] - HisH1[m+1][0][Mc_AN][h_AN][i][j];           /* s */
		y = ResidualH1[m][0][Mc_AN][h_AN][i][j] - ResidualH1[m+1][0][Mc_AN][h_AN][i][j]; /* y */
                r = s - al*y;                                                                    /* r */
                /* n */
		y = ResidualH1[n][0][Mc_AN][h_AN][i][j] - ResidualH1[n+1][0][Mc_AN][h_AN][i][j]; /* y */
		my_sum += r*y; 

                /* m */
		s = HisH1[m][1][Mc_AN][h_AN][i][j] - HisH1[m+1][1][Mc_AN][h_AN][i][j];           /* s */
		y = ResidualH1[m][1][Mc_AN][h_AN][i][j] - ResidualH1[m+1][1][Mc_AN][h_AN][i][j]; /* y */
                r = s - al*y;                                                                    /* r */
                /* n */
		y = ResidualH1[n][1][Mc_AN][h_AN][i][j] - ResidualH1[n+1][1][Mc_AN][h_AN][i][j]; /* y */
		my_sum += r*y; 

                /* m */
		s = HisH1[m][2][Mc_AN][h_AN][i][j] - HisH1[m+1][2][Mc_AN][h_AN][i][j];           /* s */
		y = ResidualH1[m][2][Mc_AN][h_AN][i][j] - ResidualH1[m+1][2][Mc_AN][h_AN][i][j]; /* y */
                r = s - al*y;                                                                    /* r */
                /* n */
		y = ResidualH1[n][2][Mc_AN][h_AN][i][j] - ResidualH1[n+1][2][Mc_AN][h_AN][i][j]; /* y */
		my_sum += r*y; 

                /* m */
		s = HisH1[m][3][Mc_AN][h_AN][i][j] - HisH1[m+1][3][Mc_AN][h_AN][i][j];           /* s */
		y = ResidualH1[m][3][Mc_AN][h_AN][i][j] - ResidualH1[m+1][3][Mc_AN][h_AN][i][j]; /* y */
                r = s - al*y;                                                                    /* r */
                /* n */
		y = ResidualH1[n][3][Mc_AN][h_AN][i][j] - ResidualH1[n+1][3][Mc_AN][h_AN][i][j]; /* y */
		my_sum += r*y; 

                /* m */
		s = HisH2[m][0][Mc_AN][h_AN][i][j] - HisH2[m+1][0][Mc_AN][h_AN][i][j];           /* s */
		y = ResidualH2[m][0][Mc_AN][h_AN][i][j] - ResidualH2[m+1][0][Mc_AN][h_AN][i][j]; /* y */
                r = s - al*y;                                                                    /* r */
                /* n */
		y = ResidualH2[n][0][Mc_AN][h_AN][i][j] - ResidualH2[n+1][0][Mc_AN][h_AN][i][j]; /* y */
		my_sum += r*y; 

                /* m */
		s = HisH2[m][1][Mc_AN][h_AN][i][j] - HisH2[m+1][1][Mc_AN][h_AN][i][j];           /* s */
		y = ResidualH2[m][1][Mc_AN][h_AN][i][j] - ResidualH2[m+1][1][Mc_AN][h_AN][i][j]; /* y */
                r = s - al*y;                                                                    /* r */
                /* n */
		y = ResidualH2[n][1][Mc_AN][h_AN][i][j] - ResidualH2[n+1][1][Mc_AN][h_AN][i][j]; /* y */
		my_sum += r*y; 

                /* m */
		s = HisH2[m][2][Mc_AN][h_AN][i][j] - HisH2[m+1][2][Mc_AN][h_AN][i][j];           /* s */
		y = ResidualH2[m][2][Mc_AN][h_AN][i][j] - ResidualH2[m+1][2][Mc_AN][h_AN][i][j]; /* y */
                r = s - al*y;                                                                    /* r */
                /* n */
		y = ResidualH2[n][2][Mc_AN][h_AN][i][j] - ResidualH2[n+1][2][Mc_AN][h_AN][i][j]; /* y */
		my_sum += r*y; 

	      }
	    }
	  }
	}

      } /* if (SpinP_switch==3 */

      else{

	for (spin=0; spin<=SpinP_switch; spin++){
	  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	    Gc_AN = M2G[Mc_AN];    
	    Cwan = WhatSpecies[Gc_AN];
	    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	      Gh_AN = natn[Gc_AN][h_AN];
	      Hwan = WhatSpecies[Gh_AN];
	      for (i=0; i<Spe_Total_NO[Cwan]; i++){
		for (j=0; j<Spe_Total_NO[Hwan]; j++){

		  /* m */
		  s = HisH1[m][spin][Mc_AN][h_AN][i][j] - HisH1[m+1][spin][Mc_AN][h_AN][i][j];           /* s */
		  y = ResidualH1[m][spin][Mc_AN][h_AN][i][j] - ResidualH1[m+1][spin][Mc_AN][h_AN][i][j]; /* y */
		  r = s - al*y;                                                                          /* r */
		  /* n */
		  y = ResidualH1[n][spin][Mc_AN][h_AN][i][j] - ResidualH1[n+1][spin][Mc_AN][h_AN][i][j]; /* y */
		  my_sum += r*y; 

		}
	      }
	    }
	  }
	}

      } /* else */

      MPI_Allreduce(&my_sum, &openmx_solver_real_matrix_row(&A,m)[n], 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

    } /* n */
  } /* m */    

  Inverse(scratch, dim-1,&A,&IA);

  /****************************************************
    calculations of inner products of <r|OptResidualH> 
  ****************************************************/

  for (m=0; m<dim; m++){

    my_sum = 0.0;

    if (SpinP_switch==3){

      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	Gc_AN = M2G[Mc_AN];    
	Cwan = WhatSpecies[Gc_AN];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  for (i=0; i<Spe_Total_NO[Cwan]; i++){
	    for (j=0; j<Spe_Total_NO[Hwan]; j++){

	      s = HisH1[m][0][Mc_AN][h_AN][i][j] - HisH1[m+1][0][Mc_AN][h_AN][i][j];           /* s */
	      y = ResidualH1[m][0][Mc_AN][h_AN][i][j] - ResidualH1[m+1][0][Mc_AN][h_AN][i][j]; /* y */
	      r = s - al*y;                                                                    /* r */
              or = ResidualH1[dim+1][0][Mc_AN][h_AN][i][j];                                    /* OptResidualH */
	      my_sum += r*or; 

	      s = HisH1[m][1][Mc_AN][h_AN][i][j] - HisH1[m+1][1][Mc_AN][h_AN][i][j];           /* s */
	      y = ResidualH1[m][1][Mc_AN][h_AN][i][j] - ResidualH1[m+1][1][Mc_AN][h_AN][i][j]; /* y */
	      r = s - al*y;                                                                    /* r */
              or = ResidualH1[dim+1][1][Mc_AN][h_AN][i][j];                                    /* OptResidualH */
	      my_sum += r*or; 

	      s = HisH1[m][2][Mc_AN][h_AN][i][j] - HisH1[m+1][2][Mc_AN][h_AN][i][j];           /* s */
	      y = ResidualH1[m][2][Mc_AN][h_AN][i][j] - ResidualH1[m+1][2][Mc_AN][h_AN][i][j]; /* y */
	      r = s - al*y;                                                                    /* r */
              or = ResidualH1[dim+1][2][Mc_AN][h_AN][i][j];                                    /* OptResidualH */
	      my_sum += r*or; 

	      s = HisH1[m][3][Mc_AN][h_AN][i][j] - HisH1[m+1][3][Mc_AN][h_AN][i][j];           /* s */
	      y = ResidualH1[m][3][Mc_AN][h_AN][i][j] - ResidualH1[m+1][3][Mc_AN][h_AN][i][j]; /* y */
	      r = s - al*y;                                                                    /* r */
              or = ResidualH1[dim+1][3][Mc_AN][h_AN][i][j];                                    /* OptResidualH */
	      my_sum += r*or; 

	      s = HisH2[m][0][Mc_AN][h_AN][i][j] - HisH2[m+1][0][Mc_AN][h_AN][i][j];           /* s */
	      y = ResidualH2[m][0][Mc_AN][h_AN][i][j] - ResidualH2[m+1][0][Mc_AN][h_AN][i][j]; /* y */
	      r = s - al*y;                                                                    /* r */
              or = ResidualH2[dim+1][0][Mc_AN][h_AN][i][j];                                    /* OptResidualH */
	      my_sum += r*or; 

	      s = HisH2[m][1][Mc_AN][h_AN][i][j] - HisH2[m+1][1][Mc_AN][h_AN][i][j];           /* s */
	      y = ResidualH2[m][1][Mc_AN][h_AN][i][j] - ResidualH2[m+1][1][Mc_AN][h_AN][i][j]; /* y */
	      r = s - al*y;                                                                    /* r */
              or = ResidualH2[dim+1][1][Mc_AN][h_AN][i][j];                                    /* OptResidualH */
	      my_sum += r*or; 

	      s = HisH2[m][2][Mc_AN][h_AN][i][j] - HisH2[m+1][2][Mc_AN][h_AN][i][j];           /* s */
	      y = ResidualH2[m][2][Mc_AN][h_AN][i][j] - ResidualH2[m+1][2][Mc_AN][h_AN][i][j]; /* y */
	      r = s - al*y;                                                                    /* r */
              or = ResidualH2[dim+1][2][Mc_AN][h_AN][i][j];                                    /* OptResidualH */
	      my_sum += r*or; 

	    }
	  }
	}
      }

    } /* if (SpinP_switch==3 */

    else{

      for (spin=0; spin<=SpinP_switch; spin++){
	for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	  Gc_AN = M2G[Mc_AN];    
	  Cwan = WhatSpecies[Gc_AN];
	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	    Gh_AN = natn[Gc_AN][h_AN];
	    Hwan = WhatSpecies[Gh_AN];
	    for (i=0; i<Spe_Total_NO[Cwan]; i++){
	      for (j=0; j<Spe_Total_NO[Hwan]; j++){

		s = HisH1[m][spin][Mc_AN][h_AN][i][j] - HisH1[m+1][spin][Mc_AN][h_AN][i][j];           
		y = ResidualH1[m][spin][Mc_AN][h_AN][i][j] - ResidualH1[m+1][spin][Mc_AN][h_AN][i][j]; 
		r = s - al*y;                                                                          
		or = ResidualH1[dim+1][spin][Mc_AN][h_AN][i][j];                                   
		my_sum += r*or; 
	      }
	    }
	  }
	}
      }

    } /* else */

    MPI_Allreduce(&my_sum, &ror[m], 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

  } /* m */    

  /****************************************************
     calculation of \sum_j b_{ij} * <r_j|OptResidualH> 
  ****************************************************/
    
  for (m=0; m<dim; m++){
    sum = 0.0;  
    for (n=0; n<dim; n++){
      sum += openmx_solver_real_matrix_row(&IA,m)[n]*ror[n];
    }

    coes2[m] = sum;
  }

  /****************************************************
                 mixing of Hamiltonian
  ****************************************************/

  if (1.0e-1<=NormRD[0])
    alpha = 0.5;
  else if (1.0e-2<=NormRD[0] && NormRD[0]<1.0e-1)
    alpha = 0.6;
  else if (1.0e-3<=NormRD[0] && NormRD[0]<1.0e-2)
    alpha = 0.7;
  else if (1.0e-4<=NormRD[0] && NormRD[0]<1.0e-3)
    alpha = 0.8;
  else
    alpha = 1.0;

  if (SpinP_switch==3){

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];    
      Cwan = WhatSpecies[Gc_AN];
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	Gh_AN = natn[Gc_AN][h_AN];
	Hwan = WhatSpecies[Gh_AN];
	for (i=0; i<Spe_Total_NO[Cwan]; i++){
	  for (j=0; j<Spe_Total_NO[Hwan]; j++){

            h10 = 0.0; 
            h11 = 0.0;
            h12 = 0.0;
            h13 = 0.0;

            h20 = 0.0; 
            h21 = 0.0;
            h22 = 0.0;

	    for (m=0; m<dim; m++){

	      h10 += HisH1[m][0][Mc_AN][h_AN][i][j]*coes[m+1];
	      h11 += HisH1[m][1][Mc_AN][h_AN][i][j]*coes[m+1];
	      h12 += HisH1[m][2][Mc_AN][h_AN][i][j]*coes[m+1];
	      h13 += HisH1[m][3][Mc_AN][h_AN][i][j]*coes[m+1];

	      h20 += HisH2[m][0][Mc_AN][h_AN][i][j]*coes[m+1];
	      h21 += HisH2[m][1][Mc_AN][h_AN][i][j]*coes[m+1];
	      h22 += HisH2[m][2][Mc_AN][h_AN][i][j]*coes[m+1];

	      s = HisH1[m][0][Mc_AN][h_AN][i][j] - HisH1[m+1][0][Mc_AN][h_AN][i][j];           /* s */
	      y = ResidualH1[m][0][Mc_AN][h_AN][i][j] - ResidualH1[m+1][0][Mc_AN][h_AN][i][j]; /* y */
	      r = s - al*y;                                                                    /* r */
	      h10 -= r*coes2[m];

	      s = HisH1[m][1][Mc_AN][h_AN][i][j] - HisH1[m+1][1][Mc_AN][h_AN][i][j];           /* s */
	      y = ResidualH1[m][1][Mc_AN][h_AN][i][j] - ResidualH1[m+1][1][Mc_AN][h_AN][i][j]; /* y */
	      r = s - al*y;                                                                    /* r */
	      h11 -= r*coes2[m];

	      s = HisH1[m][2][Mc_AN][h_AN][i][j] - HisH1[m+1][2][Mc_AN][h_AN][i][j];           /* s */
	      y = ResidualH1[m][2][Mc_AN][h_AN][i][j] - ResidualH1[m+1][2][Mc_AN][h_AN][i][j]; /* y */
	      r = s - al*y;                                                                    /* r */
	      h12 -= r*coes2[m];

	      s = HisH1[m][3][Mc_AN][h_AN][i][j] - HisH1[m+1][3][Mc_AN][h_AN][i][j];           /* s */
	      y = ResidualH1[m][3][Mc_AN][h_AN][i][j] - ResidualH1[m+1][3][Mc_AN][h_AN][i][j]; /* y */
	      r = s - al*y;                                                                    /* r */
	      h13 -= r*coes2[m];

	      s = HisH2[m][0][Mc_AN][h_AN][i][j] - HisH2[m+1][0][Mc_AN][h_AN][i][j];           /* s */
	      y = ResidualH2[m][0][Mc_AN][h_AN][i][j] - ResidualH2[m+1][0][Mc_AN][h_AN][i][j]; /* y */
	      r = s - al*y;                                                                    /* r */
	      h20 -= r*coes2[m];

	      s = HisH2[m][1][Mc_AN][h_AN][i][j] - HisH2[m+1][1][Mc_AN][h_AN][i][j];           /* s */
	      y = ResidualH2[m][1][Mc_AN][h_AN][i][j] - ResidualH2[m+1][1][Mc_AN][h_AN][i][j]; /* y */
	      r = s - al*y;                                                                    /* r */
	      h21 -= r*coes2[m];

	      s = HisH2[m][2][Mc_AN][h_AN][i][j] - HisH2[m+1][2][Mc_AN][h_AN][i][j];           /* s */
	      y = ResidualH2[m][2][Mc_AN][h_AN][i][j] - ResidualH2[m+1][2][Mc_AN][h_AN][i][j]; /* y */
	      r = s - al*y;                                                                    /* r */
	      h22 -= r*coes2[m];
	    }

	    H[0][Mc_AN][h_AN][i][j]    = h10 - al*ResidualH1[dim+1][0][Mc_AN][h_AN][i][j];
	    H[1][Mc_AN][h_AN][i][j]    = h11 - al*ResidualH1[dim+1][1][Mc_AN][h_AN][i][j];
	    H[2][Mc_AN][h_AN][i][j]    = h12 - al*ResidualH1[dim+1][2][Mc_AN][h_AN][i][j];
	    H[3][Mc_AN][h_AN][i][j]    = h13 - al*ResidualH1[dim+1][3][Mc_AN][h_AN][i][j];

            iHNL[0][Mc_AN][h_AN][i][j] = h20 - al*ResidualH2[dim+1][0][Mc_AN][h_AN][i][j];
            iHNL[1][Mc_AN][h_AN][i][j] = h21 - al*ResidualH2[dim+1][1][Mc_AN][h_AN][i][j];
            iHNL[2][Mc_AN][h_AN][i][j] = h22 - al*ResidualH2[dim+1][2][Mc_AN][h_AN][i][j];

	  }
	}
      }
    }

  }

  else{

    for (spin=0; spin<=SpinP_switch; spin++){
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	Gc_AN = M2G[Mc_AN];    
	Cwan = WhatSpecies[Gc_AN];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  for (i=0; i<Spe_Total_NO[Cwan]; i++){
	    for (j=0; j<Spe_Total_NO[Hwan]; j++){

	      h = 0.0;
	      for (m=0; m<dim; m++){

		h += HisH1[m][spin][Mc_AN][h_AN][i][j]*coes[m+1];
		s = HisH1[m][spin][Mc_AN][h_AN][i][j] - HisH1[m+1][spin][Mc_AN][h_AN][i][j];           
		y = ResidualH1[m][spin][Mc_AN][h_AN][i][j] - ResidualH1[m+1][spin][Mc_AN][h_AN][i][j]; 
		r = s - al*y;                                                                          
		h -= r*coes2[m];
	      }

	      H[spin][Mc_AN][h_AN][i][j] = h - al*ResidualH1[dim+1][spin][Mc_AN][h_AN][i][j];
	    }
	  }
	}
      }
    }
  }

  /****************************************************
                  shifting of Hamiltonian
  ****************************************************/

  if (SpinP_switch==3){

    /* shift the current Hamiltonian */

    for (m=dim; 0<m; m--){
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	Gc_AN = M2G[Mc_AN];    
	Cwan = WhatSpecies[Gc_AN];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  for (i=0; i<Spe_Total_NO[Cwan]; i++){
	    for (j=0; j<Spe_Total_NO[Hwan]; j++){

	      HisH1[m][0][Mc_AN][h_AN][i][j] = HisH1[m-1][0][Mc_AN][h_AN][i][j];
	      HisH1[m][1][Mc_AN][h_AN][i][j] = HisH1[m-1][1][Mc_AN][h_AN][i][j];
	      HisH1[m][2][Mc_AN][h_AN][i][j] = HisH1[m-1][2][Mc_AN][h_AN][i][j];
	      HisH1[m][3][Mc_AN][h_AN][i][j] = HisH1[m-1][3][Mc_AN][h_AN][i][j];

	      HisH2[m][0][Mc_AN][h_AN][i][j] = HisH2[m-1][0][Mc_AN][h_AN][i][j];
	      HisH2[m][1][Mc_AN][h_AN][i][j] = HisH2[m-1][1][Mc_AN][h_AN][i][j];
	      HisH2[m][2][Mc_AN][h_AN][i][j] = HisH2[m-1][2][Mc_AN][h_AN][i][j];
	    }
	  }
	}
      }
    }

    /* save the current Hamiltonian */

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];    
      Cwan = WhatSpecies[Gc_AN];
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	Gh_AN = natn[Gc_AN][h_AN];
	Hwan = WhatSpecies[Gh_AN];
	for (i=0; i<Spe_Total_NO[Cwan]; i++){
	  for (j=0; j<Spe_Total_NO[Hwan]; j++){

	    HisH1[0][0][Mc_AN][h_AN][i][j] = H[0][Mc_AN][h_AN][i][j];
	    HisH1[0][1][Mc_AN][h_AN][i][j] = H[1][Mc_AN][h_AN][i][j];
	    HisH1[0][2][Mc_AN][h_AN][i][j] = H[2][Mc_AN][h_AN][i][j];
	    HisH1[0][3][Mc_AN][h_AN][i][j] = H[3][Mc_AN][h_AN][i][j];

	    HisH2[0][0][Mc_AN][h_AN][i][j] = iHNL[0][Mc_AN][h_AN][i][j];
	    HisH2[0][1][Mc_AN][h_AN][i][j] = iHNL[1][Mc_AN][h_AN][i][j];
	    HisH2[0][2][Mc_AN][h_AN][i][j] = iHNL[2][Mc_AN][h_AN][i][j];
	  }
	}
      }
    }

  }

  else {

    /* shift the current Hamiltonian */

    for (m=dim; 0<m; m--){
      for (spin=0; spin<=SpinP_switch; spin++){
	for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	  Gc_AN = M2G[Mc_AN];    
	  Cwan = WhatSpecies[Gc_AN];
	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	    Gh_AN = natn[Gc_AN][h_AN];
	    Hwan = WhatSpecies[Gh_AN];
	    for (i=0; i<Spe_Total_NO[Cwan]; i++){
	      for (j=0; j<Spe_Total_NO[Hwan]; j++){
		HisH1[m][spin][Mc_AN][h_AN][i][j] = HisH1[m-1][spin][Mc_AN][h_AN][i][j];
	      }
	    }
	  }
	}
      }
    }

    /* save the current Hamiltonian */

    for (spin=0; spin<=SpinP_switch; spin++){
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	Gc_AN = M2G[Mc_AN];    
	Cwan = WhatSpecies[Gc_AN];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  for (i=0; i<Spe_Total_NO[Cwan]; i++){
	    for (j=0; j<Spe_Total_NO[Hwan]; j++){
	      HisH1[0][spin][Mc_AN][h_AN][i][j] = H[spin][Mc_AN][h_AN][i][j];
	    }
	  }
	}
      }
    }

  } /* else */

  /* Invocation owner releases numerical scratch. */



}










void Pulay_Mixing_H_with_One_Shot_Hessian(OpenMXSolverInvocation *scratch, int MD_iter, int SCF_iter, int SCF_iter0 , const OpenMXElectronicViews *electronic)
{
  double *****H=electronic->H;
  double ******HisH1=electronic->HisH1;
  double ******HisH2=electronic->HisH2;
  double ******ResidualH1=electronic->ResidualH1;
  double ******ResidualH2=electronic->ResidualH2;
  double *****iHNL=electronic->iHNL;

  int Mc_AN,Gc_AN,Cwan,Hwan,h_AN,Gh_AN,i,j,spin;
  int dim,m,n,flag_nan;
  double my_sum,tmp1,tmp2,alpha;
  double r,r10,r11,r12,r13,r20,r21,r22;
  double h,h10,h11,h12,h13,h20,h21,h22;
  double my_sy,my_yy,sy,yy,norm,s,y,or,al,be;
  OpenMXSolverRealTensor A,IA;
  double *coes;
  char nanchar[300];

  /****************************************************
       determination of dimension of the subspace
  ****************************************************/

  if (SCF_iter<=Num_Mixing_pDM) dim = SCF_iter-1;
  else                          dim = Num_Mixing_pDM;

  /****************************************************
                allocation of arrays 
  ****************************************************/

  coes = OpenMX_Solver_OverwriteReal(scratch,"mixh-Pulay_Mixing_H_with_One_Shot_Hessian-coes",List_YOUSO[39]);

  A = OpenMX_Solver_OverwriteView(scratch,"mixh-Pulay_Mixing_H_with_One_Shot_Hessian-A",List_YOUSO[39],List_YOUSO[39]);

  IA = OpenMX_Solver_OverwriteView(scratch,"mixh-Pulay_Mixing_H_with_One_Shot_Hessian-IA",List_YOUSO[39],List_YOUSO[39]);

  /****************************************************
                 shift the residual H
  ****************************************************/

  if (SpinP_switch==3){

    /* shift the residual Hamiltonian */

    for (m=dim; 0<m; m--){
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	Gc_AN = M2G[Mc_AN];    
	Cwan = WhatSpecies[Gc_AN];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  for (i=0; i<Spe_Total_NO[Cwan]; i++){
	    for (j=0; j<Spe_Total_NO[Hwan]; j++){

	      ResidualH1[m][0][Mc_AN][h_AN][i][j] = ResidualH1[m-1][0][Mc_AN][h_AN][i][j];
	      ResidualH1[m][1][Mc_AN][h_AN][i][j] = ResidualH1[m-1][1][Mc_AN][h_AN][i][j];
	      ResidualH1[m][2][Mc_AN][h_AN][i][j] = ResidualH1[m-1][2][Mc_AN][h_AN][i][j];
	      ResidualH1[m][3][Mc_AN][h_AN][i][j] = ResidualH1[m-1][3][Mc_AN][h_AN][i][j];

	      ResidualH2[m][0][Mc_AN][h_AN][i][j] = ResidualH2[m-1][0][Mc_AN][h_AN][i][j];
	      ResidualH2[m][1][Mc_AN][h_AN][i][j] = ResidualH2[m-1][1][Mc_AN][h_AN][i][j];
	      ResidualH2[m][2][Mc_AN][h_AN][i][j] = ResidualH2[m-1][2][Mc_AN][h_AN][i][j];
	    }
	  }
	}
      }
    }

    /* calculate the current residual Hamiltonian */

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];    
      Cwan = WhatSpecies[Gc_AN];
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	Gh_AN = natn[Gc_AN][h_AN];
	Hwan = WhatSpecies[Gh_AN];
	for (i=0; i<Spe_Total_NO[Cwan]; i++){
	  for (j=0; j<Spe_Total_NO[Hwan]; j++){

	    ResidualH1[0][0][Mc_AN][h_AN][i][j] = H[0][Mc_AN][h_AN][i][j] - HisH1[0][0][Mc_AN][h_AN][i][j];
	    ResidualH1[0][1][Mc_AN][h_AN][i][j] = H[1][Mc_AN][h_AN][i][j] - HisH1[0][1][Mc_AN][h_AN][i][j];
	    ResidualH1[0][2][Mc_AN][h_AN][i][j] = H[2][Mc_AN][h_AN][i][j] - HisH1[0][2][Mc_AN][h_AN][i][j];
	    ResidualH1[0][3][Mc_AN][h_AN][i][j] = H[3][Mc_AN][h_AN][i][j] - HisH1[0][3][Mc_AN][h_AN][i][j];

	    ResidualH2[0][0][Mc_AN][h_AN][i][j] = iHNL[0][Mc_AN][h_AN][i][j] - HisH2[0][0][Mc_AN][h_AN][i][j];
	    ResidualH2[0][1][Mc_AN][h_AN][i][j] = iHNL[1][Mc_AN][h_AN][i][j] - HisH2[0][1][Mc_AN][h_AN][i][j];
	    ResidualH2[0][2][Mc_AN][h_AN][i][j] = iHNL[2][Mc_AN][h_AN][i][j] - HisH2[0][2][Mc_AN][h_AN][i][j];
	  }
	}
      }
    }

  }

  else{

    /* shift the residual Hamiltonian */

    for (m=dim; 0<m; m--){
      for (spin=0; spin<=SpinP_switch; spin++){
	for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	  Gc_AN = M2G[Mc_AN];    
	  Cwan = WhatSpecies[Gc_AN];
	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	    Gh_AN = natn[Gc_AN][h_AN];
	    Hwan = WhatSpecies[Gh_AN];
	    for (i=0; i<Spe_Total_NO[Cwan]; i++){
	      for (j=0; j<Spe_Total_NO[Hwan]; j++){
		ResidualH1[m][spin][Mc_AN][h_AN][i][j] = ResidualH1[m-1][spin][Mc_AN][h_AN][i][j];
	      }
	    }
	  }
	}
      }
    }

    /* calculate the current residual Hamiltonian */

    for (spin=0; spin<=SpinP_switch; spin++){
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	Gc_AN = M2G[Mc_AN];    
	Cwan = WhatSpecies[Gc_AN];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  for (i=0; i<Spe_Total_NO[Cwan]; i++){
	    for (j=0; j<Spe_Total_NO[Hwan]; j++){
  	      ResidualH1[0][spin][Mc_AN][h_AN][i][j] = H[spin][Mc_AN][h_AN][i][j] - HisH1[0][spin][Mc_AN][h_AN][i][j];
	    }
	  }
	}
      }
    }

  } /* else */

  /****************************************************
          calculation of the residual matrix
  ****************************************************/

  for (m=0; m<dim; m++){
    for (n=0; n<dim; n++){

      my_sum = 0.0;

      if (SpinP_switch==3){

	for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	  Gc_AN = M2G[Mc_AN];    
	  Cwan = WhatSpecies[Gc_AN];
	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	    Gh_AN = natn[Gc_AN][h_AN];
	    Hwan = WhatSpecies[Gh_AN];
	    for (i=0; i<Spe_Total_NO[Cwan]; i++){
	      for (j=0; j<Spe_Total_NO[Hwan]; j++){

		tmp1 = ResidualH1[m][0][Mc_AN][h_AN][i][j];
		tmp2 = ResidualH1[n][0][Mc_AN][h_AN][i][j];
                my_sum += tmp1*tmp2; 

		tmp1 = ResidualH1[m][1][Mc_AN][h_AN][i][j];
		tmp2 = ResidualH1[n][1][Mc_AN][h_AN][i][j];
                my_sum += tmp1*tmp2; 

		tmp1 = ResidualH1[m][2][Mc_AN][h_AN][i][j];
		tmp2 = ResidualH1[n][2][Mc_AN][h_AN][i][j];
                my_sum += tmp1*tmp2; 

		tmp1 = ResidualH1[m][3][Mc_AN][h_AN][i][j];
		tmp2 = ResidualH1[n][3][Mc_AN][h_AN][i][j];
                my_sum += tmp1*tmp2; 

		tmp1 = ResidualH2[m][0][Mc_AN][h_AN][i][j];
		tmp2 = ResidualH2[n][0][Mc_AN][h_AN][i][j];
                my_sum += tmp1*tmp2; 

		tmp1 = ResidualH2[m][1][Mc_AN][h_AN][i][j];
		tmp2 = ResidualH2[n][1][Mc_AN][h_AN][i][j];
                my_sum += tmp1*tmp2; 

		tmp1 = ResidualH2[m][2][Mc_AN][h_AN][i][j];
		tmp2 = ResidualH2[n][2][Mc_AN][h_AN][i][j];
                my_sum += tmp1*tmp2; 
	      }
	    }
	  }
	}

      } /* if (SpinP_switch==3 */

      else{

	for (spin=0; spin<=SpinP_switch; spin++){
	  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	    Gc_AN = M2G[Mc_AN];    
	    Cwan = WhatSpecies[Gc_AN];
	    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	      Gh_AN = natn[Gc_AN][h_AN];
	      Hwan = WhatSpecies[Gh_AN];
	      for (i=0; i<Spe_Total_NO[Cwan]; i++){
		for (j=0; j<Spe_Total_NO[Hwan]; j++){
		  tmp1 = ResidualH1[m][spin][Mc_AN][h_AN][i][j];
		  tmp2 = ResidualH1[n][spin][Mc_AN][h_AN][i][j];
                  my_sum += tmp1*tmp2; 
		}
	      }
	    }
	  }
	}

      } /* else */

      MPI_Allreduce(&my_sum, &openmx_solver_real_matrix_row(&A,m)[n], 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
      openmx_solver_real_matrix_row(&A,n)[m] = openmx_solver_real_matrix_row(&A,m)[n];

    } /* n */
  } /* m */

  NormRD[0] = openmx_solver_real_matrix_row(&A,0)[0]/(double)atomnum;

  for (m=1; m<=dim; m++){
    openmx_solver_real_matrix_row(&A,m-1)[dim] = -1.0;
    openmx_solver_real_matrix_row(&A,dim)[m-1] = -1.0;
  }
  openmx_solver_real_matrix_row(&A,dim)[dim] = 0.0;

  Inverse(scratch, dim,&A,&IA);

  for (m=1; m<=dim; m++){
    coes[m] = -openmx_solver_real_matrix_row(&IA,m-1)[dim];
  }

  /****************************************************
            check "nan", "NaN", "inf" or "Inf"
  ****************************************************/

  flag_nan = 0;
  for (m=1; m<=dim; m++){

    sprintf(nanchar,"%8.4f",coes[m]);
    if (   strstr(nanchar,"nan")!=NULL || strstr(nanchar,"NaN")!=NULL 
	|| strstr(nanchar,"inf")!=NULL || strstr(nanchar,"Inf")!=NULL){

      flag_nan = 1;
    }
  }

  if (flag_nan==1){
    for (m=1; m<=dim; m++){
      coes[m] = 0.0;
    }
    coes[1] = 0.05;
    coes[2] = 0.95;
  }

  /****************************************************
      calculation of optimum residual Hamiltonian
  ****************************************************/

  if (SpinP_switch==3){

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];    
      Cwan = WhatSpecies[Gc_AN];
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	Gh_AN = natn[Gc_AN][h_AN];
	Hwan = WhatSpecies[Gh_AN];
	for (i=0; i<Spe_Total_NO[Cwan]; i++){
	  for (j=0; j<Spe_Total_NO[Hwan]; j++){

            r10 = 0.0; 
            r11 = 0.0;
            r12 = 0.0;
            r13 = 0.0;

            r20 = 0.0; 
            r21 = 0.0;
            r22 = 0.0;

	    for (m=0; m<dim; m++){

	      r10 += ResidualH1[m][0][Mc_AN][h_AN][i][j]*coes[m+1];
	      r11 += ResidualH1[m][1][Mc_AN][h_AN][i][j]*coes[m+1];
	      r12 += ResidualH1[m][2][Mc_AN][h_AN][i][j]*coes[m+1];
	      r13 += ResidualH1[m][3][Mc_AN][h_AN][i][j]*coes[m+1];

	      r20 += ResidualH2[m][0][Mc_AN][h_AN][i][j]*coes[m+1];
	      r21 += ResidualH2[m][1][Mc_AN][h_AN][i][j]*coes[m+1];
	      r22 += ResidualH2[m][2][Mc_AN][h_AN][i][j]*coes[m+1];
	    }

            /* optimum Residual H is stored in ResidualH1[dim] and ResidualH2[dim] */

	    ResidualH1[dim][0][Mc_AN][h_AN][i][j] = r10;
	    ResidualH1[dim][1][Mc_AN][h_AN][i][j] = r11;
	    ResidualH1[dim][2][Mc_AN][h_AN][i][j] = r12;
	    ResidualH1[dim][3][Mc_AN][h_AN][i][j] = r13;

	    ResidualH2[dim][0][Mc_AN][h_AN][i][j] = r20;
	    ResidualH2[dim][1][Mc_AN][h_AN][i][j] = r21;
	    ResidualH2[dim][2][Mc_AN][h_AN][i][j] = r22;

	  }
	}
      }
    }

  }

  else{

    for (spin=0; spin<=SpinP_switch; spin++){
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	Gc_AN = M2G[Mc_AN];    
	Cwan = WhatSpecies[Gc_AN];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  for (i=0; i<Spe_Total_NO[Cwan]; i++){
	    for (j=0; j<Spe_Total_NO[Hwan]; j++){

	      r = 0.0; 
	      for (m=0; m<dim; m++){
		r += ResidualH1[m][spin][Mc_AN][h_AN][i][j]*coes[m+1];
	      }

              /* optimum Residual H is stored in ResidualH1[dim] */

              ResidualH1[dim][spin][Mc_AN][h_AN][i][j] = r;

	    }
	  }
	}
      }
    }
  }

  /****************************************************
           innner products of <s|y> and <y|y>
  ****************************************************/

  my_sy = 0.0;
  my_yy = 0.0;

  if (SpinP_switch==3){

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];    
      Cwan = WhatSpecies[Gc_AN];
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	Gh_AN = natn[Gc_AN][h_AN];
	Hwan = WhatSpecies[Gh_AN];
	for (i=0; i<Spe_Total_NO[Cwan]; i++){
	  for (j=0; j<Spe_Total_NO[Hwan]; j++){

	    tmp1 = HisH1[0][0][Mc_AN][h_AN][i][j] - HisH1[1][0][Mc_AN][h_AN][i][j];           /* s */
	    tmp2 = ResidualH1[0][0][Mc_AN][h_AN][i][j] - ResidualH1[1][0][Mc_AN][h_AN][i][j]; /* y */
	    my_sy += tmp1*tmp2; 
	    my_yy += tmp2*tmp2; 

	    tmp1 = HisH1[0][1][Mc_AN][h_AN][i][j] - HisH1[1][1][Mc_AN][h_AN][i][j];           /* s */
	    tmp2 = ResidualH1[0][1][Mc_AN][h_AN][i][j] - ResidualH1[1][1][Mc_AN][h_AN][i][j]; /* y */
	    my_sy += tmp1*tmp2; 
	    my_yy += tmp2*tmp2; 

	    tmp1 = HisH1[0][2][Mc_AN][h_AN][i][j] - HisH1[1][2][Mc_AN][h_AN][i][j];           /* s */
	    tmp2 = ResidualH1[0][2][Mc_AN][h_AN][i][j] - ResidualH1[1][2][Mc_AN][h_AN][i][j]; /* y */
	    my_sy += tmp1*tmp2; 
	    my_yy += tmp2*tmp2; 

	    tmp1 = HisH1[0][3][Mc_AN][h_AN][i][j] - HisH1[1][3][Mc_AN][h_AN][i][j];           /* s */
	    tmp2 = ResidualH1[0][3][Mc_AN][h_AN][i][j] - ResidualH1[1][3][Mc_AN][h_AN][i][j]; /* y */
	    my_sy += tmp1*tmp2; 
	    my_yy += tmp2*tmp2; 

	    tmp1 = HisH2[0][0][Mc_AN][h_AN][i][j] - HisH2[1][0][Mc_AN][h_AN][i][j];           /* s */
	    tmp2 = ResidualH2[0][0][Mc_AN][h_AN][i][j] - ResidualH2[1][0][Mc_AN][h_AN][i][j]; /* y */
	    my_sy += tmp1*tmp2; 
	    my_yy += tmp2*tmp2; 

	    tmp1 = HisH2[0][1][Mc_AN][h_AN][i][j] - HisH2[1][1][Mc_AN][h_AN][i][j];           /* s */
	    tmp2 = ResidualH2[0][1][Mc_AN][h_AN][i][j] - ResidualH2[1][1][Mc_AN][h_AN][i][j]; /* y */
	    my_sy += tmp1*tmp2; 
	    my_yy += tmp2*tmp2; 

	    tmp1 = HisH2[0][2][Mc_AN][h_AN][i][j] - HisH2[1][2][Mc_AN][h_AN][i][j];           /* s */
	    tmp2 = ResidualH2[0][2][Mc_AN][h_AN][i][j] - ResidualH2[1][2][Mc_AN][h_AN][i][j]; /* y */
	    my_sy += tmp1*tmp2; 
	    my_yy += tmp2*tmp2; 
	  }
	}
      }
    }

  } /* if (SpinP_switch==3 */

  else{

    for (spin=0; spin<=SpinP_switch; spin++){
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	Gc_AN = M2G[Mc_AN];    
	Cwan = WhatSpecies[Gc_AN];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  for (i=0; i<Spe_Total_NO[Cwan]; i++){
	    for (j=0; j<Spe_Total_NO[Hwan]; j++){

	      tmp1 = HisH1[0][spin][Mc_AN][h_AN][i][j] - HisH1[1][spin][Mc_AN][h_AN][i][j];           /* s */
	      tmp2 = ResidualH1[0][spin][Mc_AN][h_AN][i][j] - ResidualH1[1][spin][Mc_AN][h_AN][i][j]; /* y */
	      my_sy += tmp1*tmp2; 
	      my_yy += tmp2*tmp2; 
	    }
	  }
	}
      }
    }

  } /* else */

  MPI_Allreduce(&my_sy, &sy, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
  MPI_Allreduce(&my_yy, &yy, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

  /* al < sy/yy */

  al = sy/yy - 0.2;

  /* be = 1/(<s|y>-al*<y|y>) */

  be = 1.0/(sy-al*yy);

  /****************************************************
      inner product of (<s|-al<y|)|OptResidualH>
  ****************************************************/
 
  my_sum = 0.0;

  if (SpinP_switch==3){

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];    
      Cwan = WhatSpecies[Gc_AN];
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	Gh_AN = natn[Gc_AN][h_AN];
	Hwan = WhatSpecies[Gh_AN];
	for (i=0; i<Spe_Total_NO[Cwan]; i++){
	  for (j=0; j<Spe_Total_NO[Hwan]; j++){

	    s = HisH1[0][0][Mc_AN][h_AN][i][j] - HisH1[1][0][Mc_AN][h_AN][i][j];           /* s */
	    y = ResidualH1[0][0][Mc_AN][h_AN][i][j] - ResidualH1[1][0][Mc_AN][h_AN][i][j]; /* y */
            or = ResidualH1[dim][0][Mc_AN][h_AN][i][j];                                    /* OptResidualH */
	    my_sum += (s-al*y)*or;

	    s = HisH1[0][1][Mc_AN][h_AN][i][j] - HisH1[1][1][Mc_AN][h_AN][i][j];           /* s */
	    y = ResidualH1[0][1][Mc_AN][h_AN][i][j] - ResidualH1[1][1][Mc_AN][h_AN][i][j]; /* y */
            or = ResidualH1[dim][1][Mc_AN][h_AN][i][j];                                    /* OptResidualH */
	    my_sum += (s-al*y)*or;

	    s = HisH1[0][2][Mc_AN][h_AN][i][j] - HisH1[1][2][Mc_AN][h_AN][i][j];           /* s */
	    y = ResidualH1[0][2][Mc_AN][h_AN][i][j] - ResidualH1[1][2][Mc_AN][h_AN][i][j]; /* y */
            or = ResidualH1[dim][2][Mc_AN][h_AN][i][j];                                    /* OptResidualH */
	    my_sum += (s-al*y)*or;

	    s = HisH1[0][3][Mc_AN][h_AN][i][j] - HisH1[1][3][Mc_AN][h_AN][i][j];           /* s */
	    y = ResidualH1[0][3][Mc_AN][h_AN][i][j] - ResidualH1[1][3][Mc_AN][h_AN][i][j]; /* y */
            or = ResidualH1[dim][3][Mc_AN][h_AN][i][j];                                    /* OptResidualH */
	    my_sum += (s-al*y)*or;

	    s = HisH2[0][0][Mc_AN][h_AN][i][j] - HisH2[1][0][Mc_AN][h_AN][i][j];           /* s */
	    y = ResidualH2[0][0][Mc_AN][h_AN][i][j] - ResidualH2[1][0][Mc_AN][h_AN][i][j]; /* y */
            or = ResidualH2[dim][0][Mc_AN][h_AN][i][j];                                    /* OptResidualH */
	    my_sum += (s-al*y)*or;

	    s = HisH2[0][1][Mc_AN][h_AN][i][j] - HisH2[1][1][Mc_AN][h_AN][i][j];           /* s */
	    y = ResidualH2[0][1][Mc_AN][h_AN][i][j] - ResidualH2[1][1][Mc_AN][h_AN][i][j]; /* y */
            or = ResidualH2[dim][1][Mc_AN][h_AN][i][j];                                    /* OptResidualH */
	    my_sum += (s-al*y)*or;

	    s = HisH2[0][2][Mc_AN][h_AN][i][j] - HisH2[1][2][Mc_AN][h_AN][i][j];           /* s */
	    y = ResidualH2[0][2][Mc_AN][h_AN][i][j] - ResidualH2[1][2][Mc_AN][h_AN][i][j]; /* y */
            or = ResidualH2[dim][2][Mc_AN][h_AN][i][j];                                    /* OptResidualH */
	    my_sum += (s-al*y)*or;
	  }
	}
      }
    }

  } /* if (SpinP_switch==3 */

  else{

    for (spin=0; spin<=SpinP_switch; spin++){
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	Gc_AN = M2G[Mc_AN];    
	Cwan = WhatSpecies[Gc_AN];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  for (i=0; i<Spe_Total_NO[Cwan]; i++){
	    for (j=0; j<Spe_Total_NO[Hwan]; j++){
	      s = HisH1[0][spin][Mc_AN][h_AN][i][j] - HisH1[1][spin][Mc_AN][h_AN][i][j];           /* s */
	      y = ResidualH1[0][spin][Mc_AN][h_AN][i][j] - ResidualH1[1][spin][Mc_AN][h_AN][i][j]; /* y */
	      or = ResidualH1[dim][spin][Mc_AN][h_AN][i][j];                                       /* OptResidualH */
	      my_sum += (s-al*y)*or;
	    }
	  }
	}
      }
    }

  } /* else */

  MPI_Allreduce(&my_sum, &norm, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
  be = norm*be;

  /****************************************************
                 mixing of Hamiltonian
  ****************************************************/

  if (1.0e-1<=NormRD[0])
    alpha = 0.5;
  else if (1.0e-2<=NormRD[0] && NormRD[0]<1.0e-1)
    alpha = 0.6;
  else if (1.0e-3<=NormRD[0] && NormRD[0]<1.0e-2)
    alpha = 0.7;
  else if (1.0e-4<=NormRD[0] && NormRD[0]<1.0e-3)
    alpha = 0.8;
  else
    alpha = 1.0;

  if (SpinP_switch==3){

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];    
      Cwan = WhatSpecies[Gc_AN];
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	Gh_AN = natn[Gc_AN][h_AN];
	Hwan = WhatSpecies[Gh_AN];
	for (i=0; i<Spe_Total_NO[Cwan]; i++){
	  for (j=0; j<Spe_Total_NO[Hwan]; j++){

            h10 = 0.0; 
            h11 = 0.0;
            h12 = 0.0;
            h13 = 0.0;

            h20 = 0.0; 
            h21 = 0.0;
            h22 = 0.0;

	    for (m=0; m<dim; m++){

	      h10 += HisH1[m][0][Mc_AN][h_AN][i][j]*coes[m+1];
	      h11 += HisH1[m][1][Mc_AN][h_AN][i][j]*coes[m+1];
	      h12 += HisH1[m][2][Mc_AN][h_AN][i][j]*coes[m+1];
	      h13 += HisH1[m][3][Mc_AN][h_AN][i][j]*coes[m+1];

	      h20 += HisH2[m][0][Mc_AN][h_AN][i][j]*coes[m+1];
	      h21 += HisH2[m][1][Mc_AN][h_AN][i][j]*coes[m+1];
	      h22 += HisH2[m][2][Mc_AN][h_AN][i][j]*coes[m+1];
	    }

	    s = HisH1[0][0][Mc_AN][h_AN][i][j] - HisH1[1][0][Mc_AN][h_AN][i][j];           /* s */
	    y = ResidualH1[0][0][Mc_AN][h_AN][i][j] - ResidualH1[1][0][Mc_AN][h_AN][i][j]; /* y */
            or = ResidualH1[dim][0][Mc_AN][h_AN][i][j];                                    /* OptResidualH */
	    H[0][Mc_AN][h_AN][i][j] = h10 - alpha*(al*or + (s-al*y)*be);

	    s = HisH1[0][1][Mc_AN][h_AN][i][j] - HisH1[1][1][Mc_AN][h_AN][i][j];           /* s */
	    y = ResidualH1[0][1][Mc_AN][h_AN][i][j] - ResidualH1[1][1][Mc_AN][h_AN][i][j]; /* y */
            or = ResidualH1[dim][1][Mc_AN][h_AN][i][j];                                    /* OptResidualH */
	    H[1][Mc_AN][h_AN][i][j] = h11 - alpha*(al*or + (s-al*y)*be);

	    s = HisH1[0][2][Mc_AN][h_AN][i][j] - HisH1[1][2][Mc_AN][h_AN][i][j];           /* s */
	    y = ResidualH1[0][2][Mc_AN][h_AN][i][j] - ResidualH1[1][2][Mc_AN][h_AN][i][j]; /* y */
            or = ResidualH1[dim][2][Mc_AN][h_AN][i][j];                                    /* OptResidualH */
	    H[2][Mc_AN][h_AN][i][j] = h12 - alpha*(al*or + (s-al*y)*be);

	    s = HisH1[0][3][Mc_AN][h_AN][i][j] - HisH1[1][3][Mc_AN][h_AN][i][j];           /* s */
	    y = ResidualH1[0][3][Mc_AN][h_AN][i][j] - ResidualH1[1][3][Mc_AN][h_AN][i][j]; /* y */
            or = ResidualH1[dim][3][Mc_AN][h_AN][i][j];                                    /* OptResidualH */
	    H[3][Mc_AN][h_AN][i][j] = h13 - alpha*(al*or + (s-al*y)*be);

	    s = HisH2[0][0][Mc_AN][h_AN][i][j] - HisH2[1][0][Mc_AN][h_AN][i][j];           /* s */
	    y = ResidualH2[0][0][Mc_AN][h_AN][i][j] - ResidualH2[1][0][Mc_AN][h_AN][i][j]; /* y */
            or = ResidualH2[dim][0][Mc_AN][h_AN][i][j];                                    /* OptResidualH */
	    iHNL[0][Mc_AN][h_AN][i][j] = h20 - alpha*(al*or + (s-al*y)*be);

	    s = HisH2[0][1][Mc_AN][h_AN][i][j] - HisH2[1][1][Mc_AN][h_AN][i][j];           /* s */
	    y = ResidualH2[0][1][Mc_AN][h_AN][i][j] - ResidualH2[1][1][Mc_AN][h_AN][i][j]; /* y */
            or = ResidualH2[dim][1][Mc_AN][h_AN][i][j];                                    /* OptResidualH */
	    iHNL[1][Mc_AN][h_AN][i][j] = h21 - alpha*(al*or + (s-al*y)*be);

	    s = HisH2[0][2][Mc_AN][h_AN][i][j] - HisH2[1][2][Mc_AN][h_AN][i][j];           /* s */
	    y = ResidualH2[0][2][Mc_AN][h_AN][i][j] - ResidualH2[1][2][Mc_AN][h_AN][i][j]; /* y */
            or = ResidualH2[dim][2][Mc_AN][h_AN][i][j];                                    /* OptResidualH */
	    iHNL[2][Mc_AN][h_AN][i][j] = h22 - alpha*(al*or + (s-al*y)*be);
	  }
	}
      }
    }

  }

  else{

    for (spin=0; spin<=SpinP_switch; spin++){
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	Gc_AN = M2G[Mc_AN];    
	Cwan = WhatSpecies[Gc_AN];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  for (i=0; i<Spe_Total_NO[Cwan]; i++){
	    for (j=0; j<Spe_Total_NO[Hwan]; j++){

	      h = 0.0;
	      for (m=0; m<dim; m++){
		h += HisH1[m][spin][Mc_AN][h_AN][i][j]*coes[m+1];
	      }

	      s = HisH1[0][spin][Mc_AN][h_AN][i][j] - HisH1[1][spin][Mc_AN][h_AN][i][j];           /* s */
	      y = ResidualH1[0][spin][Mc_AN][h_AN][i][j] - ResidualH1[1][spin][Mc_AN][h_AN][i][j]; /* y */
              or = ResidualH1[dim][spin][Mc_AN][h_AN][i][j];                                       /* OptResidualH */
	      H[spin][Mc_AN][h_AN][i][j] = h - alpha*(al*or + (s-al*y)*be);

	    }
	  }
	}
      }
    }
  }

  /****************************************************
                  shifting of Hamiltonian
  ****************************************************/

  if (SpinP_switch==3){

    /* shift the current Hamiltonian */

    for (m=dim; 0<m; m--){
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	Gc_AN = M2G[Mc_AN];    
	Cwan = WhatSpecies[Gc_AN];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  for (i=0; i<Spe_Total_NO[Cwan]; i++){
	    for (j=0; j<Spe_Total_NO[Hwan]; j++){

	      HisH1[m][0][Mc_AN][h_AN][i][j] = HisH1[m-1][0][Mc_AN][h_AN][i][j];
	      HisH1[m][1][Mc_AN][h_AN][i][j] = HisH1[m-1][1][Mc_AN][h_AN][i][j];
	      HisH1[m][2][Mc_AN][h_AN][i][j] = HisH1[m-1][2][Mc_AN][h_AN][i][j];
	      HisH1[m][3][Mc_AN][h_AN][i][j] = HisH1[m-1][3][Mc_AN][h_AN][i][j];

	      HisH2[m][0][Mc_AN][h_AN][i][j] = HisH2[m-1][0][Mc_AN][h_AN][i][j];
	      HisH2[m][1][Mc_AN][h_AN][i][j] = HisH2[m-1][1][Mc_AN][h_AN][i][j];
	      HisH2[m][2][Mc_AN][h_AN][i][j] = HisH2[m-1][2][Mc_AN][h_AN][i][j];
	    }
	  }
	}
      }
    }

    /* save the current Hamiltonian */

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];    
      Cwan = WhatSpecies[Gc_AN];
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	Gh_AN = natn[Gc_AN][h_AN];
	Hwan = WhatSpecies[Gh_AN];
	for (i=0; i<Spe_Total_NO[Cwan]; i++){
	  for (j=0; j<Spe_Total_NO[Hwan]; j++){

	    HisH1[0][0][Mc_AN][h_AN][i][j] = H[0][Mc_AN][h_AN][i][j];
	    HisH1[0][1][Mc_AN][h_AN][i][j] = H[1][Mc_AN][h_AN][i][j];
	    HisH1[0][2][Mc_AN][h_AN][i][j] = H[2][Mc_AN][h_AN][i][j];
	    HisH1[0][3][Mc_AN][h_AN][i][j] = H[3][Mc_AN][h_AN][i][j];

	    HisH2[0][0][Mc_AN][h_AN][i][j] = iHNL[0][Mc_AN][h_AN][i][j];
	    HisH2[0][1][Mc_AN][h_AN][i][j] = iHNL[1][Mc_AN][h_AN][i][j];
	    HisH2[0][2][Mc_AN][h_AN][i][j] = iHNL[2][Mc_AN][h_AN][i][j];
	  }
	}
      }
    }

  }

  else {

    /* shift the current Hamiltonian */

    for (m=dim; 0<m; m--){
      for (spin=0; spin<=SpinP_switch; spin++){
	for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	  Gc_AN = M2G[Mc_AN];    
	  Cwan = WhatSpecies[Gc_AN];
	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	    Gh_AN = natn[Gc_AN][h_AN];
	    Hwan = WhatSpecies[Gh_AN];
	    for (i=0; i<Spe_Total_NO[Cwan]; i++){
	      for (j=0; j<Spe_Total_NO[Hwan]; j++){
		HisH1[m][spin][Mc_AN][h_AN][i][j] = HisH1[m-1][spin][Mc_AN][h_AN][i][j];
	      }
	    }
	  }
	}
      }
    }

    /* save the current Hamiltonian */

    for (spin=0; spin<=SpinP_switch; spin++){
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	Gc_AN = M2G[Mc_AN];    
	Cwan = WhatSpecies[Gc_AN];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  for (i=0; i<Spe_Total_NO[Cwan]; i++){
	    for (j=0; j<Spe_Total_NO[Hwan]; j++){
	      HisH1[0][spin][Mc_AN][h_AN][i][j] = H[spin][Mc_AN][h_AN][i][j];
	    }
	  }
	}
      }
    }

  } /* else */

  /* Invocation owner releases numerical scratch. */

}





/*******************************************************************************
  GPU acceleration of Pulay_Mixing_H (RMM-DIISH).

  The CPU implementation sweeps the whole sparse Hamiltonian structure
  dim*dim times per SCF iteration for the residual Gram matrix (with one
  MPI_Allreduce per matrix element) and another ~4*dim times for the
  shift/mix/save loops.  With scf.Mixing.History around 50 this dominates
  the SCF step.  The GPU path keeps the residual history device-resident:

    - ONE cudaMalloc arena per rank holds a ring of Num_Mixing_pDM flat
      residual slots plus the metric-weight vector, the optimum-residual
      vector, and the Gram pair buffer.  The per-iteration "shift" is a
      rotation of a host-side logical-to-physical slot map (zero copies).
    - The dim*(dim+1)/2 weighted inner products run in one OpenACC kernel;
      the whole triangle is reduced with a single MPI_Allreduce.
    - The optimum residual (sum_m coes[m+1]*R[m]) is one device kernel;
      only that L-vector is downloaded per iteration.
    - HisH1/HisH2 stay host-resident (they are touched O(dim*L) once per
      iteration); their shift becomes a rotation of the m-level pointers,
      which yields bitwise-identical logical contents.
    - The host jagged ResidualH1/ResidualH2 arrays are ALSO kept logically
      current (pointer rotation + fresh slot 0 each iteration), so the
      E_Temp-controller SCF rewind (DFT.c sets SCF_iter_shift mid-cycle,
      Simple_Mixing_H briefly takes over, then Pulay re-engages with a
      large dim) can always re-seed the device ring from the host arrays.

  The preflight is MPI-collective over mpi_comm_level1 and accounts for the
  SUM of the arena sizes of all ranks sharing one device.  On a negative
  verdict the original CPU code below runs unchanged.  Env knobs:
  OPENMX_MIXH_GPU (=0 never / =1 force), OPENMX_MIXH_GPU_RESERVE_MB
  (default max(total/10, 1536 MiB)), OPENMX_MIXH_GPU_VERBOSE.
*******************************************************************************/

#if !OPENMX_GPU_HIP
#include <openacc.h>
#endif
#include <stdint.h>






/* The sparse residual dimension is partitioned, including its persistent
   circular history, weights and optimum residual. Only the small Gram
   triangle is summed on the host before the existing MPI reduction. */



/* Used only when the persistent device history cannot be admitted. The
   host ring lets contractions try remote node GPUs without publishing
   partial Gram matrices or changing the logical history permutation. */


static void MixH_Distributed_Free(MixingHState *state);

static void MixH_AbortWithMessage(const char *message)
{
  fprintf(stderr, "%s\n", message);
  fflush(stderr);
  MPI_Abort(mpi_comm_level1, 1);
}

/* Bounded device allocation (same policy as the cluster/band GPU paths):
   absorb short allocation races with a few retries, report persistent
   failure to the caller instead of retrying forever. */
static OpenMXGpuError MixH_TryDeviceMalloc(void **ptr, size_t bytes)
{
  OpenMXGpuError err = OPENMX_GPU_OUT_OF_MEMORY;

  for (int attempt = 0; attempt < 8; attempt++) {
    err = OpenMXGpuMalloc(ptr, bytes);
    if (err == OPENMX_GPU_SUCCESS) return OPENMX_GPU_SUCCESS;
    (void)OpenMXGpuGetLastError();

    double wait_time = drand48() * WAITTIME;
    double start_time = MPI_Wtime();
    double current_time = start_time;
    while ((current_time - start_time) < wait_time) {
      current_time = MPI_Wtime();
    }
  }

  *ptr = NULL;
  return err;
}

static size_t MixH_Align512(size_t bytes)
{
  return (bytes + (size_t)511) & ~((size_t)511);
}

/* Release everything and force a fresh preflight on the next Pulay call.
   Called before the force/energy phase of every SCF cycle (DFT.c) so the
   history ring never competes with the force-path device transients. */
static void MixH_IncFree(MixingHState *state);
static void MixH_IncBuild(MixingHState *state, int nslot);

void Mixing_H_Release_GPU(MixingHState *state)
{
  if (state == NULL) return;
  MixH_IncFree(state);
  MixH_Distributed_Free(state);
  if (state->gpu.base != NULL) OpenMX_Resource_GpuFree(state->resources,state->gpu.base);
  openmx_core_native_free(state->resources, state->gpu.perm);
  openmx_core_native_free(state->resources, state->gpu.atom_off);
  openmx_core_native_free(state->resources, state->gpu.h_flat);
  openmx_core_native_free(state->resources, state->gpu.h_r);
  openmx_core_native_free(state->resources, state->host_ring);
  openmx_core_native_free(state->resources, state->host_weight);
  state->host_ring = state->host_weight = NULL;
  state->gpu.base = NULL;
  state->gpu.perm = NULL;
  state->gpu.atom_off = NULL;
  state->gpu.h_flat = NULL;
  state->gpu.h_r = NULL;
  state->gpu.total_bytes = 0U;
  state->gpu.L = 0U;
  state->gpu.L_plane = 0U;
  state->gpu.nslot = 0;
  state->gpu.valid = 0;
  state->gpu.verdict = -1;
  state->gpu.announced = 0;
  state->gpu.fp_md_iter = -1;
  state->gpu.fp_matomnum = -1;
  state->gpu.fp_spinp = -1;
  state->gpu.fp_nslot = -1;
  state->gpu.fp_L = 0U;
  state->gpu.last_scf_iter = -1;
}

/* Flat layout of one history slot: collinear packs the planes
   spin-major with the CPU loop order (Mc_AN, h_AN, i, j) inside each
   plane; the noncollinear case packs the seven components (four of
   H/HisH1/ResidualH1, three of iHNL/HisH2/ResidualH2) consecutively per
   (Mc_AN,h_AN,i,j) element. */
static void MixH_ComputeLayout(MixingHState *state)
{
  int Mc_AN,Gc_AN,Cwan,h_AN,Gh_AN,Hwan;
  int ncomp = (SpinP_switch==3) ? 7 : 1;
  size_t acc = 0U;

  openmx_core_native_free(state->resources, state->gpu.atom_off);
  state->gpu.atom_off = (size_t*)openmx_core_native_malloc(state->resources, 1, sizeof(size_t)*(size_t)(Matomnum+1), 0);
  if (state->gpu.atom_off == NULL) {
    MixH_AbortWithMessage("Mixing_H.c: failed to allocate the GPU atom offset table.");
  }
  state->gpu.atom_off[0] = 0U;

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    size_t blk = 0U;
    Gc_AN = M2G[Mc_AN];
    Cwan = WhatSpecies[Gc_AN];
    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
      Gh_AN = natn[Gc_AN][h_AN];
      Hwan = WhatSpecies[Gh_AN];
      blk += (size_t)Spe_Total_NO[Cwan]*(size_t)Spe_Total_NO[Hwan];
    }
    state->gpu.atom_off[Mc_AN] = acc;
    acc += blk*(size_t)ncomp;
  }

  state->gpu.L_plane = acc;
  state->gpu.L = (SpinP_switch==3) ? acc : acc*(size_t)(SpinP_switch+1);
}

/* rotate the top-level (history) pointers of a six-star array over
   m = 0..hi; logically identical to the CPU content shift
   arr[m] = arr[m-1] (m = hi..1) followed by an overwrite of arr[0]. */
static void MixH_RotatePtr6(double ******arr, int hi)
{
  double *****tmp = arr[hi];
  int m;
  for (m=hi; 0<m; m--) arr[m] = arr[m-1];
  arr[0] = tmp;
}

/* Compute the fresh residual into the (already rotated) host slot 0 and,
   when dst is non-NULL, the packed flat image of that slot. */
static void MixH_Residual0Fused(MixingHState *state, double *dst, const OpenMXElectronicViews *electronic)
{
  double *****H=electronic->H;
  double ******HisH1=electronic->HisH1;
  double ******HisH2=electronic->HisH2;
  double ******ResidualH1=electronic->ResidualH1;
  double ******ResidualH2=electronic->ResidualH2;
  double *****iHNL=electronic->iHNL;

  int Mc_AN,Gc_AN,Cwan,h_AN,Gh_AN,Hwan,i,j,spin;

  if (SpinP_switch==3){

#pragma omp parallel for private(Mc_AN,Gc_AN,Cwan,h_AN,Gh_AN,Hwan,i,j) schedule(static)
    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      size_t k = state->gpu.atom_off[Mc_AN];
      Gc_AN = M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
        Gh_AN = natn[Gc_AN][h_AN];
        Hwan = WhatSpecies[Gh_AN];
        for (i=0; i<Spe_Total_NO[Cwan]; i++){
          for (j=0; j<Spe_Total_NO[Hwan]; j++){

            double r0 = H[0][Mc_AN][h_AN][i][j]    - HisH1[0][0][Mc_AN][h_AN][i][j];
            double r1 = H[1][Mc_AN][h_AN][i][j]    - HisH1[0][1][Mc_AN][h_AN][i][j];
            double r2 = H[2][Mc_AN][h_AN][i][j]    - HisH1[0][2][Mc_AN][h_AN][i][j];
            double r3 = H[3][Mc_AN][h_AN][i][j]    - HisH1[0][3][Mc_AN][h_AN][i][j];
            double s0 = iHNL[0][Mc_AN][h_AN][i][j] - HisH2[0][0][Mc_AN][h_AN][i][j];
            double s1 = iHNL[1][Mc_AN][h_AN][i][j] - HisH2[0][1][Mc_AN][h_AN][i][j];
            double s2 = iHNL[2][Mc_AN][h_AN][i][j] - HisH2[0][2][Mc_AN][h_AN][i][j];

            ResidualH1[0][0][Mc_AN][h_AN][i][j] = r0;
            ResidualH1[0][1][Mc_AN][h_AN][i][j] = r1;
            ResidualH1[0][2][Mc_AN][h_AN][i][j] = r2;
            ResidualH1[0][3][Mc_AN][h_AN][i][j] = r3;
            ResidualH2[0][0][Mc_AN][h_AN][i][j] = s0;
            ResidualH2[0][1][Mc_AN][h_AN][i][j] = s1;
            ResidualH2[0][2][Mc_AN][h_AN][i][j] = s2;

            if (dst != NULL){
              dst[k  ] = r0;
              dst[k+1] = r1;
              dst[k+2] = r2;
              dst[k+3] = r3;
              dst[k+4] = s0;
              dst[k+5] = s1;
              dst[k+6] = s2;
            }
            k += 7;
          }
        }
      }
    }

  }
  else{

    for (spin=0; spin<=SpinP_switch; spin++){
      size_t plane = (size_t)spin*state->gpu.L_plane;
#pragma omp parallel for private(Mc_AN,Gc_AN,Cwan,h_AN,Gh_AN,Hwan,i,j) schedule(static)
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
        size_t k = plane + state->gpu.atom_off[Mc_AN];
        Gc_AN = M2G[Mc_AN];
        Cwan = WhatSpecies[Gc_AN];
        for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
          Gh_AN = natn[Gc_AN][h_AN];
          Hwan = WhatSpecies[Gh_AN];
          for (i=0; i<Spe_Total_NO[Cwan]; i++){
            for (j=0; j<Spe_Total_NO[Hwan]; j++){
              double r = H[spin][Mc_AN][h_AN][i][j] - HisH1[0][spin][Mc_AN][h_AN][i][j];
              ResidualH1[0][spin][Mc_AN][h_AN][i][j] = r;
              if (dst != NULL) dst[k] = r;
              k++;
            }
          }
        }
      }
    }
  }
}

/* pack the host jagged residual slot m into a flat buffer (seeding) */
static void MixH_PackResidualSlot(MixingHState *state, int m, double *dst, const OpenMXElectronicViews *electronic)
{
  double ******ResidualH1=electronic->ResidualH1;
  double ******ResidualH2=electronic->ResidualH2;

  int Mc_AN,Gc_AN,Cwan,h_AN,Gh_AN,Hwan,i,j,spin;

  if (SpinP_switch==3){

#pragma omp parallel for private(Mc_AN,Gc_AN,Cwan,h_AN,Gh_AN,Hwan,i,j) schedule(static)
    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      size_t k = state->gpu.atom_off[Mc_AN];
      Gc_AN = M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
        Gh_AN = natn[Gc_AN][h_AN];
        Hwan = WhatSpecies[Gh_AN];
        for (i=0; i<Spe_Total_NO[Cwan]; i++){
          for (j=0; j<Spe_Total_NO[Hwan]; j++){
            dst[k  ] = ResidualH1[m][0][Mc_AN][h_AN][i][j];
            dst[k+1] = ResidualH1[m][1][Mc_AN][h_AN][i][j];
            dst[k+2] = ResidualH1[m][2][Mc_AN][h_AN][i][j];
            dst[k+3] = ResidualH1[m][3][Mc_AN][h_AN][i][j];
            dst[k+4] = ResidualH2[m][0][Mc_AN][h_AN][i][j];
            dst[k+5] = ResidualH2[m][1][Mc_AN][h_AN][i][j];
            dst[k+6] = ResidualH2[m][2][Mc_AN][h_AN][i][j];
            k += 7;
          }
        }
      }
    }

  }
  else{

    for (spin=0; spin<=SpinP_switch; spin++){
      size_t plane = (size_t)spin*state->gpu.L_plane;
#pragma omp parallel for private(Mc_AN,Gc_AN,Cwan,h_AN,Gh_AN,Hwan,i,j) schedule(static)
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
        size_t k = plane + state->gpu.atom_off[Mc_AN];
        Gc_AN = M2G[Mc_AN];
        Cwan = WhatSpecies[Gc_AN];
        for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
          Gh_AN = natn[Gc_AN][h_AN];
          Hwan = WhatSpecies[Gh_AN];
          for (i=0; i<Spe_Total_NO[Cwan]; i++){
            for (j=0; j<Spe_Total_NO[Hwan]; j++){
              dst[k] = ResidualH1[m][spin][Mc_AN][h_AN][i][j];
              k++;
            }
          }
        }
      }
    }
  }
}

/* pack the metric weights (metric[Mc_AN][i], identical for every spin
   plane / component) into the flat layout */
static void MixH_PackMetricW(MixingHState *state, OpenMXSolverRealFieldView metric, double *dst)
{
  int Mc_AN,Gc_AN,Cwan,h_AN,Gh_AN,Hwan,i,j,spin;
  int ncomp = (SpinP_switch==3) ? 7 : 1;
  int nsp = (SpinP_switch==3) ? 1 : SpinP_switch+1;

  for (spin=0; spin<nsp; spin++){
    size_t plane = (size_t)spin*state->gpu.L_plane;
#pragma omp parallel for private(Mc_AN,Gc_AN,Cwan,h_AN,Gh_AN,Hwan,i,j) schedule(static)
    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      size_t k = plane + state->gpu.atom_off[Mc_AN];
      Gc_AN = M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
        Gh_AN = natn[Gc_AN][h_AN];
        Hwan = WhatSpecies[Gh_AN];
        for (i=0; i<Spe_Total_NO[Cwan]; i++){
          double w = metric.blocks[Mc_AN].values[i];
          for (j=0; j<Spe_Total_NO[Hwan]; j++){
            int c;
            for (c=0; c<ncomp; c++) dst[k+c] = w;
            k += ncomp;
          }
        }
      }
    }
  }
}

/* H = sum_m coes[m+1]*HisH1[m] + alpha*r  (and iHNL analogously for NC) */
static void MixH_MixFromHistory(MixingHState *state, int dim, double alpha, const double *coes,
                                const double *r_flat, const OpenMXElectronicViews *electronic)
{
  double *****H=electronic->H;
  double ******HisH1=electronic->HisH1;
  double ******HisH2=electronic->HisH2;
  double *****iHNL=electronic->iHNL;

  int Mc_AN,Gc_AN,Cwan,h_AN,Gh_AN,Hwan,i,j,spin,m;

  if (SpinP_switch==3){

#pragma omp parallel for private(Mc_AN,Gc_AN,Cwan,h_AN,Gh_AN,Hwan,i,j,m) schedule(static)
    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      size_t k = state->gpu.atom_off[Mc_AN];
      Gc_AN = M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
        Gh_AN = natn[Gc_AN][h_AN];
        Hwan = WhatSpecies[Gh_AN];
        for (i=0; i<Spe_Total_NO[Cwan]; i++){
          for (j=0; j<Spe_Total_NO[Hwan]; j++){

            double h0=0.0,h1=0.0,h2=0.0,h3=0.0,g0=0.0,g1=0.0,g2=0.0;
            for (m=0; m<dim; m++){
              double c = coes[m+1];
              h0 += HisH1[m][0][Mc_AN][h_AN][i][j]*c;
              h1 += HisH1[m][1][Mc_AN][h_AN][i][j]*c;
              h2 += HisH1[m][2][Mc_AN][h_AN][i][j]*c;
              h3 += HisH1[m][3][Mc_AN][h_AN][i][j]*c;
              g0 += HisH2[m][0][Mc_AN][h_AN][i][j]*c;
              g1 += HisH2[m][1][Mc_AN][h_AN][i][j]*c;
              g2 += HisH2[m][2][Mc_AN][h_AN][i][j]*c;
            }

            H[0][Mc_AN][h_AN][i][j] = h0 + alpha*r_flat[k  ];
            H[1][Mc_AN][h_AN][i][j] = h1 + alpha*r_flat[k+1];
            H[2][Mc_AN][h_AN][i][j] = h2 + alpha*r_flat[k+2];
            H[3][Mc_AN][h_AN][i][j] = h3 + alpha*r_flat[k+3];
            iHNL[0][Mc_AN][h_AN][i][j] = g0 + alpha*r_flat[k+4];
            iHNL[1][Mc_AN][h_AN][i][j] = g1 + alpha*r_flat[k+5];
            iHNL[2][Mc_AN][h_AN][i][j] = g2 + alpha*r_flat[k+6];
            k += 7;
          }
        }
      }
    }

  }
  else{

    for (spin=0; spin<=SpinP_switch; spin++){
      size_t plane = (size_t)spin*state->gpu.L_plane;
#pragma omp parallel for private(Mc_AN,Gc_AN,Cwan,h_AN,Gh_AN,Hwan,i,j,m) schedule(static)
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
        size_t k = plane + state->gpu.atom_off[Mc_AN];
        Gc_AN = M2G[Mc_AN];
        Cwan = WhatSpecies[Gc_AN];
        for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
          Gh_AN = natn[Gc_AN][h_AN];
          Hwan = WhatSpecies[Gh_AN];
          for (i=0; i<Spe_Total_NO[Cwan]; i++){
            for (j=0; j<Spe_Total_NO[Hwan]; j++){
              double h = 0.0;
              for (m=0; m<dim; m++){
                h += HisH1[m][spin][Mc_AN][h_AN][i][j]*coes[m+1];
              }
              H[spin][Mc_AN][h_AN][i][j] = h + alpha*r_flat[k];
              k++;
            }
          }
        }
      }
    }
  }
}

/* save the mixed Hamiltonian into the (already rotated) history slot 0 */
static void MixH_SaveHistory0(const OpenMXElectronicViews *electronic)
{
  double *****H=electronic->H;
  double ******HisH1=electronic->HisH1;
  double ******HisH2=electronic->HisH2;
  double *****iHNL=electronic->iHNL;

  int Mc_AN,Gc_AN,Cwan,h_AN,Gh_AN,Hwan,i,j,spin;

  if (SpinP_switch==3){

#pragma omp parallel for private(Mc_AN,Gc_AN,Cwan,h_AN,Gh_AN,Hwan,i,j) schedule(static)
    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
        Gh_AN = natn[Gc_AN][h_AN];
        Hwan = WhatSpecies[Gh_AN];
        for (i=0; i<Spe_Total_NO[Cwan]; i++){
          for (j=0; j<Spe_Total_NO[Hwan]; j++){
            HisH1[0][0][Mc_AN][h_AN][i][j] = H[0][Mc_AN][h_AN][i][j];
            HisH1[0][1][Mc_AN][h_AN][i][j] = H[1][Mc_AN][h_AN][i][j];
            HisH1[0][2][Mc_AN][h_AN][i][j] = H[2][Mc_AN][h_AN][i][j];
            HisH1[0][3][Mc_AN][h_AN][i][j] = H[3][Mc_AN][h_AN][i][j];
            HisH2[0][0][Mc_AN][h_AN][i][j] = iHNL[0][Mc_AN][h_AN][i][j];
            HisH2[0][1][Mc_AN][h_AN][i][j] = iHNL[1][Mc_AN][h_AN][i][j];
            HisH2[0][2][Mc_AN][h_AN][i][j] = iHNL[2][Mc_AN][h_AN][i][j];
          }
        }
      }
    }

  }
  else{

    for (spin=0; spin<=SpinP_switch; spin++){
#pragma omp parallel for private(Mc_AN,Gc_AN,Cwan,h_AN,Gh_AN,Hwan,i,j) schedule(static)
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
        Gc_AN = M2G[Mc_AN];
        Cwan = WhatSpecies[Gc_AN];
        for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
          Gh_AN = natn[Gc_AN][h_AN];
          Hwan = WhatSpecies[Gh_AN];
          for (i=0; i<Spe_Total_NO[Cwan]; i++){
            for (j=0; j<Spe_Total_NO[Hwan]; j++){
              HisH1[0][spin][Mc_AN][h_AN][i][j] = H[spin][Mc_AN][h_AN][i][j];
            }
          }
        }
      }
    }
  }
}

/* Collective preflight: rebuild the layout and the device arena when the
   fingerprint changed, account for every rank sharing this device, and
   agree on one verdict over mpi_comm_level1. */
/* weighted Gram triangle of the logical residual slots 0..dim-1 */
static void MixH_GramPart(MixH_GpuShard *shard,int dim, int npair, const int *pm, const int *pn,
                            const int *pslot, double *tri_out)
{
  double *ring = (double*)(shard->base + shard->o_ring);
  double *w    = (double*)(shard->base + shard->o_w);
  double *ap   = (double*)(shard->base + shard->o_pair);
  const long Lk = (long)shard->L;
  const size_t Ls = shard->L;
  int p;

  if (0 < Lk){
#if OPENMX_GPU_HIP
    /* Match the host-backed vector worker: each pair gets a team-wide
       FP64 reduction rather than a single lane walking the history. */
#pragma omp target teams distribute thread_limit(256) is_device_ptr(ring,w,ap) map(to:pm[0:npair],pn[0:npair],pslot[0:dim])
#else
#pragma acc parallel loop gang deviceptr(ring,w,ap) copyin(pm[0:npair],pn[0:npair],pslot[0:dim])
#endif
    for (p=0; p<npair; p++){
      const double *Rm = ring + (size_t)pslot[pm[p]]*Ls;
      const double *Rn = ring + (size_t)pslot[pn[p]]*Ls;
      double s = 0.0;
      long k;
#if OPENMX_GPU_HIP
#pragma omp parallel for reduction(+:s)
#else
#pragma acc loop vector reduction(+:s)
#endif
      for (k=0; k<Lk; k++) s += w[k]*Rm[k]*Rn[k];
      ap[p] = s;
    }
#if OPENMX_GPU_HIP
    openmx_target_download(tri_out, ap, sizeof(double)*(size_t)npair);
#else
    acc_memcpy_from_device(tri_out, ap, sizeof(double)*(size_t)npair);
#endif
  }
  else{
    for (p=0; p<npair; p++) tri_out[p] = 0.0;
  }
}

/* optimum residual r = sum_m coes[m+1]*R[m] on the device */
static void MixH_OptResidualPart(MixingHState *state, MixH_GpuShard *shard,int dim, const int *pslot, const double *cshift)
{
  double *ring = (double*)(shard->base + shard->o_ring);
  double *r    = (double*)(shard->base + shard->o_r);
  const long Lk = (long)shard->L;
  const size_t Ls = shard->L;
  long k;

  if (Lk <= 0) return;

#if OPENMX_GPU_HIP
#pragma omp target teams distribute parallel for is_device_ptr(ring,r) map(to:pslot[0:dim],cshift[0:dim])
#else
#pragma acc parallel loop gang vector deviceptr(ring,r) copyin(pslot[0:dim],cshift[0:dim])
#endif
  for (k=0; k<Lk; k++){
    double s = 0.0;
    int m;
    for (m=0; m<dim; m++) s += cshift[m]*ring[(size_t)pslot[m]*Ls + (size_t)k];
    r[k] = s;
  }
#if OPENMX_GPU_HIP
  openmx_target_download(state->gpu.h_r+shard->first, r, sizeof(double)*(size_t)Lk);
#else
  acc_memcpy_from_device(state->gpu.h_r+shard->first, r, sizeof(double)*(size_t)Lk);
#endif
}


enum { MIXH_PROBE,MIXH_ALLOCATE,MIXH_FREE,MIXH_UPLOAD,MIXH_GRAM,MIXH_RESIDUAL };
typedef struct {
  MixingHState *state;
  int mode,slot,dim,npair;
  const int *pm,*pn,*pslot;
  const double *coeff;
  double *tri;
} MixH_Work;

static int MixH_Worker(void *opaque,int worker,int phase)
{
  MixH_Work *w=(MixH_Work*)opaque;
  MixingHState *state=w->state;
  MixH_GpuShard *s=state->shard+worker;
  if (w->mode==MIXH_ALLOCATE && phase==OPENMX_GPU_WORKER_PREPARE) {
    if (OpenMX_Resource_GpuMalloc(state->resources,(void**)&s->base,s->bytes)!=OPENMX_GPU_SUCCESS) {
      (void)OpenMXGpuGetLastError();s->base=NULL;return 0;
    }
  }
  if (phase!=OPENMX_GPU_WORKER_EXECUTE) return 1;
  if (w->mode==MIXH_PROBE) return OpenMX_GpuMemGetInfo(&s->available,&s->total)==OPENMX_GPU_SUCCESS;
  if (w->mode==MIXH_FREE) {
    int ok=!s->base || OpenMX_Resource_GpuFree(state->resources,s->base)==OPENMX_GPU_SUCCESS;s->base=NULL;return ok;
  }
  if (w->mode==MIXH_UPLOAD && s->L) {
    size_t offset=w->slot<0?s->o_w:s->o_ring+(size_t)w->slot*s->L*sizeof(double);
    return OpenMXGpuMemcpy(s->base+offset,state->gpu.h_flat+s->first,s->L*sizeof(double),OPENMX_GPU_COPY_HOST_TO_DEVICE)==OPENMX_GPU_SUCCESS;
  }
  if (w->mode==MIXH_GRAM) MixH_GramPart(s,w->dim,w->npair,w->pm,w->pn,w->pslot,w->tri+(size_t)worker*w->npair);
  if (w->mode==MIXH_RESIDUAL) MixH_OptResidualPart(state, s,w->dim,w->pslot,w->coeff);
  return 1;
}

static void MixH_Distributed_Free(MixingHState *state)
{
  int have_arena=0;
  for (int worker=0;worker<state->workers.count;worker++) if (state->shard[worker].base) have_arena=1;
  if (have_arena) {
    MixH_Work work={0};work.state=state;work.mode=MIXH_FREE;
    if (!openmx_gpu_workers_run(&state->workers,MixH_Worker,&work,"mixing-release"))
      MixH_AbortWithMessage("Mixing_H: failed to release distributed residual history.");
  }
  memset(state->shard,0,sizeof(state->shard));memset(&state->workers,0,sizeof(state->workers));
}

static size_t MixH_ShardCost(void *opaque,int worker,int count)
{
  MixingHState *state=(MixingHState*)opaque;
  size_t L=openmx_gpu_worker_cut(state->gpu.L,worker+1,count)-
           openmx_gpu_worker_cut(state->gpu.L,worker,count);
  size_t slots=(size_t)state->gpu.nslot,npair=slots*(slots+1)/2;
  /* Include four alignment paddings and the small OpenACC pair tables. */
  if (!slots || npair>(SIZE_MAX/sizeof(double)-256)/4 ||
      L>(SIZE_MAX/sizeof(double)-npair*4-256)/(slots+2)) return SIZE_MAX;
  size_t bytes=((slots+2)*L+npair*4+256)*sizeof(double);
  return bytes>SIZE_MAX-state->capacity_reserve ? SIZE_MAX : bytes+state->capacity_reserve;
}

static int MixH_Distributed_Allocate(MixingHState *state, int forced_on,int myid)
{
  (void)forced_on;
  int local_ok=gpu_rank_device_usable() &&
    openmx_gpu_workers_capacity(&state->workers,state->gpu.L?state->gpu.L:1,1,MixH_ShardCost,state);
  for (int worker=0;worker<state->workers.count;worker++) {
    MixH_GpuShard *s=state->shard+worker;
    s->first=openmx_gpu_worker_cut(state->gpu.L,worker,state->workers.count);
    s->L=openmx_gpu_worker_cut(state->gpu.L,worker+1,state->workers.count)-s->first;
    const size_t Lb=s->L*sizeof(double),npair=(size_t)state->gpu.nslot*(state->gpu.nslot+1)/2;
    s->o_ring=0;s->o_w=MixH_Align512((size_t)state->gpu.nslot*Lb);
    s->o_r=MixH_Align512(s->o_w+Lb);s->o_pair=MixH_Align512(s->o_r+Lb);
    s->bytes=MixH_Align512(s->o_pair+npair*sizeof(double));
    if (!s->bytes) s->bytes=512;
  }
  /* Allocation is serialized by physical GPU UUID, including ranks whose
     CUDA ordinals refer to different device visibility lists. cudaMalloc
     is the final admission check for simultaneously resident histories. */
  MixH_Work work={0};work.state=state;work.mode=MIXH_ALLOCATE;
  if (local_ok) local_ok=openmx_gpu_workers_run(&state->workers,MixH_Worker,&work,"mixing-history-allocation");
  int fit=0;
  MPI_Allreduce(&local_ok,&fit,1,MPI_INT,MPI_MIN,mpi_comm_level1);
  if (!fit) {
    MixH_Distributed_Free(state);
    if (myid==Host_ID && level_stdout>=2)
      fprintf(stderr,"<Mixing_H> Persistent GPU history does not fit; trying host-backed local/remote GPU contractions.\n");
  }
  return fit;
}

static int MixH_HostBacked_Allocate(MixingHState *state)
{
  const size_t slots=(size_t)state->gpu.nslot;
  int local_ok=slots>0 && state->gpu.L<=SIZE_MAX/sizeof(double)/(slots+3);
  unsigned long long need=local_ok ? (unsigned long long)((slots+3)*state->gpu.L*sizeof(double)) : ULLONG_MAX;
  unsigned long long sum=0,available=openmx_host_mem_available();
  MPI_Comm node;
  MPI_Comm_split_type(mpi_comm_level1,MPI_COMM_TYPE_SHARED,0,MPI_INFO_NULL,&node);
  int ranks=1;MPI_Comm_size(node,&ranks);
  /* Avoid overflowing the node sum when one rank has an impossible size. */
  if (need>ULLONG_MAX/(unsigned)ranks) local_ok=0;
  MPI_Allreduce(MPI_IN_PLACE,&local_ok,1,MPI_INT,MPI_MIN,node);
  if (local_ok) {
    MPI_Allreduce(&need,&sum,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,node);
    if (available && sum>available/2) local_ok=0;
  }
  MPI_Comm_free(&node);
  int fit=0;MPI_Allreduce(&local_ok,&fit,1,MPI_INT,MPI_MIN,mpi_comm_level1);
  if (fit) {
    state->host_ring=(double*)openmx_core_native_malloc(state->resources, state->gpu.L ? slots*state->gpu.L : 1, sizeof(double), 1);
    state->host_weight=(double*)openmx_core_native_malloc(state->resources, state->gpu.L?state->gpu.L:1, sizeof(double), 1);
    local_ok=state->host_ring && state->host_weight;
    MPI_Allreduce(&local_ok,&fit,1,MPI_INT,MPI_MIN,mpi_comm_level1);
  }
  if (!fit) {
    openmx_core_native_free(state->resources, state->host_ring);openmx_core_native_free(state->resources, state->host_weight);
    state->host_ring=state->host_weight=NULL;
  }
  return fit;
}

static void MixH_Upload(MixingHState *state, int slot)
{
  if (state->host_ring) {
    double *target=slot<0?state->host_weight:state->host_ring+(size_t)slot*state->gpu.L;
    memcpy(target,state->gpu.h_flat,state->gpu.L*sizeof(double));
    return;
  }
  MixH_Work work={0};work.state=state;work.mode=MIXH_UPLOAD;work.slot=slot;
  if (!openmx_gpu_workers_run(&state->workers,MixH_Worker,&work,"mixing-upload"))
    MixH_AbortWithMessage("Mixing_H: failed to update a distributed residual-history shard.");
}

static void MixH_GramDevice(OpenMXSolverInvocation *scratch, MixingHState *state, int dim,int npair,const int *pm,const int *pn,const int *pslot,double *tri_out)
{
  if (state->host_ring) {
    int *pairs=OpenMX_Solver_Int(scratch,"mixh-gram-pairs",(size_t)2*npair);
    if (!pairs) MixH_AbortWithMessage("Mixing_H: cannot allocate Gram pair indices.");
    for (int p=0;p<npair;p++) {pairs[p]=pslot[pm[p]];pairs[npair+p]=pslot[pn[p]];}
    OpenMXGpuVector job={0};job.n=job.ld=state->gpu.L;
    job.columns=state->gpu.nslot;job.pairs=npair;job.data=state->host_ring;
    job.weight=state->host_weight;job.left=pairs;job.right=pairs+npair;job.output=tri_out;
    openmx_gpu_vector_execute(mpi_comm_level1,&job,1,"mixing-Gram");
    return;
  }
  MixH_Work w={0};w.state=state;w.mode=MIXH_GRAM;w.dim=dim;w.npair=npair;w.pm=pm;w.pn=pn;w.pslot=pslot;
  const size_t count=(size_t)(npair?npair:1)*(size_t)state->workers.count;
  w.tri=OpenMX_Solver_OverwriteReal(scratch,"mixh-gram-workers",count);
  memset(w.tri,0,count*sizeof(double));
  if (!w.tri) MixH_AbortWithMessage("Mixing_H: cannot allocate partial Gram matrices.");
  if (!openmx_gpu_workers_run(&state->workers,MixH_Worker,&w,"mixing-Gram"))
    MixH_AbortWithMessage("Mixing_H: a distributed Gram kernel failed.");
  for (int p=0;p<npair;p++) {
    tri_out[p]=0.0;
    for (int worker=0;worker<state->workers.count;worker++) tri_out[p]+=w.tri[(size_t)worker*npair+p];
  }

}

static void MixH_OptResidualDevice(OpenMXSolverInvocation *scratch, MixingHState *state, int dim,const int *pslot,const double *cshift)
{
  if (state->host_ring) {
    double *coeff=OpenMX_Solver_OverwriteReal(scratch,"mixh-residual-coeff",(size_t)state->gpu.nslot);
    memset(coeff,0,(size_t)state->gpu.nslot*sizeof(double));
    if (!coeff) MixH_AbortWithMessage("Mixing_H: cannot allocate residual coefficients.");
    for (int i=0;i<dim;i++) coeff[pslot[i]]=cshift[i];
    OpenMXGpuVector job={0};job.n=job.ld=state->gpu.L;job.mode=1;
    job.columns=state->gpu.nslot;job.data=state->host_ring;job.coeff=coeff;job.output=state->gpu.h_r;
    openmx_gpu_vector_execute(mpi_comm_level1,&job,1,"mixing-optimum-residual");
    return;
  }
  MixH_Work w={0};w.state=state;w.mode=MIXH_RESIDUAL;w.dim=dim;w.pslot=pslot;w.coeff=cshift;
  if (!openmx_gpu_workers_run(&state->workers,MixH_Worker,&w,"mixing-optimum-residual"))
    MixH_AbortWithMessage("Mixing_H: a distributed optimum-residual kernel failed.");
}

static int MixH_GpuPreflight(MixingHState *state, int MD_iter, int myid)
{
  const long force_mode = openmx_env_long_at(OPENMX_ENV_MIXH_GPU, -1);   /* 0 off, 1 on, 2 incremental CPU */
  int forced_on = (force_mode == 1);
  int forced_off = (force_mode == 0);
  int forced_inc = (force_mode == 2);
  int nslot = Num_Mixing_pDM;
  int fit, my_fit;
  size_t free_b = 0U, total_b = 0U;

  if (state->gpu.verdict != -1 &&
      state->gpu.fp_md_iter == MD_iter &&
      state->gpu.fp_matomnum == Matomnum &&
      state->gpu.fp_spinp == SpinP_switch &&
      state->gpu.fp_nslot == nslot){
    return state->gpu.verdict;
  }

  Mixing_H_Release_GPU(state);
  MixH_ComputeLayout(state);
  state->gpu.nslot = nslot;
  state->gpu.fp_md_iter = MD_iter;
  state->gpu.fp_matomnum = Matomnum;
  state->gpu.fp_spinp = SpinP_switch;
  state->gpu.fp_nslot = nslot;
  state->gpu.fp_L = state->gpu.L;

  if (forced_off){
    state->gpu.verdict = 0;
    return 0;
  }

  if (forced_inc){
    MixH_IncBuild(state, nslot);
    if (openmx_env_flag_at(OPENMX_ENV_MIXH_GPU_VERBOSE, 0) && myid == Host_ID){
      printf("<Mixing_H> incremental CPU RMM-DIISH mixing forced by OPENMX_MIXH_GPU=2.\n");
      fflush(stdout);
    }
    state->gpu.verdict = 2;
    return 2;
  }

  {
    size_t Lb = state->gpu.L*sizeof(double);
    size_t npair_cap = (size_t)nslot*((size_t)nslot+1)/2;
    state->gpu.o_ring = 0U;
    state->gpu.o_w    = MixH_Align512((size_t)nslot*Lb);
    state->gpu.o_r    = MixH_Align512(state->gpu.o_w + Lb);
    state->gpu.o_pair = MixH_Align512(state->gpu.o_r + Lb);
    state->gpu.total_bytes = MixH_Align512(state->gpu.o_pair + npair_cap*sizeof(double));
  }

  /* Quiesce and measure, then account per shared device.  Deliberately NO
     acc_clear_freelists() here: this preflight runs mid-cycle (first Pulay
     iteration) and draining the OpenACC pool at that point changes the
     allocation dynamics of the other resident GPU modules on a nearly full
     device.  Pool blocks therefore show up as "used", which only makes the
     estimate conservative. */
#if OPENMX_GPU_HIP
  OpenMXGpuDeviceSynchronize();
#else
  acc_wait_all();
#endif
  OpenMXGpuDeviceSynchronize();
  MPI_Barrier(mpi_comm_level1);
  OpenMX_GpuMemGetInfo(&free_b, &total_b);
  size_t reserve=total_b/10;
  if (reserve<(size_t)1536*1024*1024) reserve=(size_t)1536*1024*1024;
  if (0<openmx_env_ulong_at(OPENMX_ENV_MIXH_GPU_RESERVE_MB,0)) reserve=openmx_env_mib_bytes_at(OPENMX_ENV_MIXH_GPU_RESERVE_MB,0);
  /* The common planner already reserves 256 MiB. */
  state->capacity_reserve=reserve>(size_t)256*1024*1024?reserve-(size_t)256*1024*1024:0;

  fit=MixH_Distributed_Allocate(state, forced_on,myid);
  if (!fit) fit=MixH_HostBacked_Allocate(state);

  if (fit){
    state->gpu.perm = (int*)openmx_core_native_malloc(state->resources, 1, sizeof(int)*(size_t)nslot, 0);
    state->gpu.h_flat = (double*)openmx_core_native_malloc(state->resources, 1, sizeof(double)*((state->gpu.L==0U)?1U:state->gpu.L), 0);
    state->gpu.h_r = (double*)openmx_core_native_malloc(state->resources, 1, sizeof(double)*((state->gpu.L==0U)?1U:state->gpu.L), 0);
    my_fit=state->gpu.perm && state->gpu.h_flat && state->gpu.h_r;
    MPI_Allreduce(&my_fit,&fit,1,MPI_INT,MPI_MIN,mpi_comm_level1);
  }
  if (fit){
    if (openmx_env_flag_at(OPENMX_ENV_MIXH_GPU_VERBOSE, 0) && myid == Host_ID){
      printf("<Mixing_H> GPU RMM-DIISH mixing engaged (%d slots x %.1f MiB per rank).\n",
             nslot, (double)state->gpu.L*sizeof(double)/(1024.0*1024.0));
      fflush(stdout);
    }
    state->gpu.verdict = 1;
    return 1;
  }

  MixH_Distributed_Free(state);
  openmx_core_native_free(state->resources, state->host_ring);openmx_core_native_free(state->resources, state->host_weight);state->host_ring=state->host_weight=NULL;
  openmx_core_native_free(state->resources, state->gpu.perm);openmx_core_native_free(state->resources, state->gpu.h_flat);openmx_core_native_free(state->resources, state->gpu.h_r);
  state->gpu.perm=NULL;state->gpu.h_flat=state->gpu.h_r=NULL;
  MixH_IncBuild(state, nslot);
  state->gpu.verdict = 2;
  return 2;
}

/*******************************************************************************
  Tier-B fallback: incremental block-Gram Pulay mixing on the CPU.

  When the device-resident history does not fit (or OPENMX_MIXH_GPU=2), the
  residual Gram matrix is decomposed exactly as

      A[m][n] = sum_blk  w_blk * S_blk[m][n],     blk = (Mc_AN, i)

  where S_blk[m][n] is the metric-independent partial Gram of the (Mc_AN,i)
  row block.  Because the history "shift" is a pointer rotation, the old
  S_blk entries stay bitwise valid across iterations and only the new row
  S_blk[0][n] (n = 0..dim-1) has to be computed: O(dim*L) work instead of
  the legacy O(dim^2*L), while the changing metric is re-applied exactly
  every iteration.  A broken continuity (SCF rewind, new cycle) triggers a
  full O(dim^2*L) re-seed of S.
*******************************************************************************/





static void MixH_IncFree(MixingHState *state)
{
  openmx_core_native_free(state->resources, state->inc.blk_base);
  openmx_core_native_free(state->resources, state->inc.perm);
  openmx_core_native_free(state->resources, state->inc.S);
  openmx_core_native_free(state->resources, state->inc.wblk);
  state->inc.blk_base = NULL;
  state->inc.perm = NULL;
  state->inc.S = NULL;
  state->inc.wblk = NULL;
  state->inc.valid = 0;
  state->inc.last_scf_iter = -1;
  state->inc.nslot = 0;
  state->inc.nblk = 0;
}

static void MixH_IncBuild(MixingHState *state, int nslot)
{
  int Mc_AN,Gc_AN,Cwan,nblk;

  MixH_IncFree(state);

  nblk = 0;
  state->inc.blk_base = (int*)openmx_core_native_malloc(state->resources, 1, sizeof(int)*(size_t)(Matomnum+1), 0);
  if (state->inc.blk_base == NULL){
    MixH_AbortWithMessage("Mixing_H.c: failed to allocate the block base table.");
  }
  state->inc.blk_base[0] = 0;
  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];
    Cwan = WhatSpecies[Gc_AN];
    state->inc.blk_base[Mc_AN] = nblk;
    nblk += Spe_Total_NO[Cwan];
  }
  state->inc.nblk = nblk;
  state->inc.nslot = nslot;

  state->inc.perm = (int*)openmx_core_native_malloc(state->resources, 1, sizeof(int)*(size_t)nslot, 0);
  state->inc.S = (double*)openmx_core_native_malloc(state->resources, 1, sizeof(double)*
               ((size_t)nslot*(size_t)nslot*(size_t)((nblk==0)?1:nblk)), 0);
  state->inc.wblk = (double*)openmx_core_native_malloc(state->resources, 1, sizeof(double)*(size_t)((nblk==0)?1:nblk), 0);
  if (state->inc.perm == NULL || state->inc.S == NULL || state->inc.wblk == NULL){
    MixH_AbortWithMessage("Mixing_H.c: failed to allocate the block Gram cache.");
  }
}

/* compute the block Gram row of logical slot mlog against logical slots
   0..nmax and store it (symmetrically) under physical indexing */
static void MixH_IncComputeRow(OpenMXSolverInvocation *scratch, MixingHState *state, int mlog, int nmax, const OpenMXElectronicViews *electronic)
{
  double ******ResidualH1=electronic->ResidualH1;
  double ******ResidualH2=electronic->ResidualH2;

  int Mc_AN,Gc_AN,Cwan,h_AN,Gh_AN,Hwan,i,j,n,spin;
  const int nslot = state->inc.nslot;
  const int nblk = state->inc.nblk;
  const int pm = state->inc.perm[mlog];

#pragma omp parallel private(Mc_AN,Gc_AN,Cwan,h_AN,Gh_AN,Hwan,i,j,n,spin)
  {
    double *acc = OpenMX_Solver_OverwriteReal(scratch,"mixh-inc-acc",(size_t)(nmax+1));

#pragma omp for schedule(static)
    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      int blk0 = state->inc.blk_base[Mc_AN];
      Gc_AN = M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];

      for (i=0; i<Spe_Total_NO[Cwan]; i++){
        int blk = blk0 + i;

        for (n=0; n<=nmax; n++) acc[n] = 0.0;

        if (SpinP_switch==3){

          for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
            int jan;
            Gh_AN = natn[Gc_AN][h_AN];
            Hwan = WhatSpecies[Gh_AN];
            jan = Spe_Total_NO[Hwan];

            {
              const double *p10 = ResidualH1[mlog][0][Mc_AN][h_AN][i];
              const double *p11 = ResidualH1[mlog][1][Mc_AN][h_AN][i];
              const double *p12 = ResidualH1[mlog][2][Mc_AN][h_AN][i];
              const double *p13 = ResidualH1[mlog][3][Mc_AN][h_AN][i];
              const double *p20 = ResidualH2[mlog][0][Mc_AN][h_AN][i];
              const double *p21 = ResidualH2[mlog][1][Mc_AN][h_AN][i];
              const double *p22 = ResidualH2[mlog][2][Mc_AN][h_AN][i];

              for (n=0; n<=nmax; n++){
                const double *q10 = ResidualH1[n][0][Mc_AN][h_AN][i];
                const double *q11 = ResidualH1[n][1][Mc_AN][h_AN][i];
                const double *q12 = ResidualH1[n][2][Mc_AN][h_AN][i];
                const double *q13 = ResidualH1[n][3][Mc_AN][h_AN][i];
                const double *q20 = ResidualH2[n][0][Mc_AN][h_AN][i];
                const double *q21 = ResidualH2[n][1][Mc_AN][h_AN][i];
                const double *q22 = ResidualH2[n][2][Mc_AN][h_AN][i];
                double s = 0.0;
                for (j=0; j<jan; j++){
                  s += p10[j]*q10[j] + p11[j]*q11[j] + p12[j]*q12[j] + p13[j]*q13[j]
                     + p20[j]*q20[j] + p21[j]*q21[j] + p22[j]*q22[j];
                }
                acc[n] += s;
              }
            }
          }

        }
        else{

          for (spin=0; spin<=SpinP_switch; spin++){
            for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
              int jan;
              Gh_AN = natn[Gc_AN][h_AN];
              Hwan = WhatSpecies[Gh_AN];
              jan = Spe_Total_NO[Hwan];

              {
                const double *p = ResidualH1[mlog][spin][Mc_AN][h_AN][i];
                for (n=0; n<=nmax; n++){
                  const double *q = ResidualH1[n][spin][Mc_AN][h_AN][i];
                  double s = 0.0;
                  for (j=0; j<jan; j++) s += p[j]*q[j];
                  acc[n] += s;
                }
              }
            }
          }
        }

        for (n=0; n<=nmax; n++){
          int pn = state->inc.perm[n];
          state->inc.S[((size_t)pm*(size_t)nslot + (size_t)pn)*(size_t)nblk + (size_t)blk] = acc[n];
          state->inc.S[((size_t)pn*(size_t)nslot + (size_t)pm)*(size_t)nblk + (size_t)blk] = acc[n];
        }
      }
    }


  }
}

/* fused optimum-residual + Hamiltonian mixing sweep:
   H = sum_m coes[m+1]*HisH1[m] + alpha * sum_m coes[m+1]*ResidualH1[m] */
static void MixH_IncMixSweep(int dim, double alpha, const double *coes, const OpenMXElectronicViews *electronic)
{
  double *****H=electronic->H;
  double ******HisH1=electronic->HisH1;
  double ******HisH2=electronic->HisH2;
  double ******ResidualH1=electronic->ResidualH1;
  double ******ResidualH2=electronic->ResidualH2;
  double *****iHNL=electronic->iHNL;

  int Mc_AN,Gc_AN,Cwan,h_AN,Gh_AN,Hwan,i,j,m,spin;

  if (SpinP_switch==3){

#pragma omp parallel for private(Mc_AN,Gc_AN,Cwan,h_AN,Gh_AN,Hwan,i,j,m) schedule(static)
    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
        Gh_AN = natn[Gc_AN][h_AN];
        Hwan = WhatSpecies[Gh_AN];
        for (i=0; i<Spe_Total_NO[Cwan]; i++){
          for (j=0; j<Spe_Total_NO[Hwan]; j++){

            double r10=0.0,r11=0.0,r12=0.0,r13=0.0,r20=0.0,r21=0.0,r22=0.0;
            double h10=0.0,h11=0.0,h12=0.0,h13=0.0,h20=0.0,h21=0.0,h22=0.0;

            for (m=0; m<dim; m++){
              double c = coes[m+1];
              r10 += ResidualH1[m][0][Mc_AN][h_AN][i][j]*c;
              r11 += ResidualH1[m][1][Mc_AN][h_AN][i][j]*c;
              r12 += ResidualH1[m][2][Mc_AN][h_AN][i][j]*c;
              r13 += ResidualH1[m][3][Mc_AN][h_AN][i][j]*c;
              r20 += ResidualH2[m][0][Mc_AN][h_AN][i][j]*c;
              r21 += ResidualH2[m][1][Mc_AN][h_AN][i][j]*c;
              r22 += ResidualH2[m][2][Mc_AN][h_AN][i][j]*c;
              h10 += HisH1[m][0][Mc_AN][h_AN][i][j]*c;
              h11 += HisH1[m][1][Mc_AN][h_AN][i][j]*c;
              h12 += HisH1[m][2][Mc_AN][h_AN][i][j]*c;
              h13 += HisH1[m][3][Mc_AN][h_AN][i][j]*c;
              h20 += HisH2[m][0][Mc_AN][h_AN][i][j]*c;
              h21 += HisH2[m][1][Mc_AN][h_AN][i][j]*c;
              h22 += HisH2[m][2][Mc_AN][h_AN][i][j]*c;
            }

            H[0][Mc_AN][h_AN][i][j] = h10 + alpha*r10;
            H[1][Mc_AN][h_AN][i][j] = h11 + alpha*r11;
            H[2][Mc_AN][h_AN][i][j] = h12 + alpha*r12;
            H[3][Mc_AN][h_AN][i][j] = h13 + alpha*r13;
            iHNL[0][Mc_AN][h_AN][i][j] = h20 + alpha*r20;
            iHNL[1][Mc_AN][h_AN][i][j] = h21 + alpha*r21;
            iHNL[2][Mc_AN][h_AN][i][j] = h22 + alpha*r22;
          }
        }
      }
    }

  }
  else{

    for (spin=0; spin<=SpinP_switch; spin++){
#pragma omp parallel for private(Mc_AN,Gc_AN,Cwan,h_AN,Gh_AN,Hwan,i,j,m) schedule(static)
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
        Gc_AN = M2G[Mc_AN];
        Cwan = WhatSpecies[Gc_AN];
        for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
          Gh_AN = natn[Gc_AN][h_AN];
          Hwan = WhatSpecies[Gh_AN];
          for (i=0; i<Spe_Total_NO[Cwan]; i++){
            for (j=0; j<Spe_Total_NO[Hwan]; j++){
              double r = 0.0;
              double h = 0.0;
              for (m=0; m<dim; m++){
                double c = coes[m+1];
                r += ResidualH1[m][spin][Mc_AN][h_AN][i][j]*c;
                h += HisH1[m][spin][Mc_AN][h_AN][i][j]*c;
              }
              H[spin][Mc_AN][h_AN][i][j] = h + alpha*r;
            }
          }
        }
      }
    }
  }
}

static void Pulay_Mixing_H_Inc(OpenMXSolverInvocation *scratch, MixingHState *state, int MD_iter, int SCF_iter, int SCF_iter0, int dim, const OpenMXElectronicViews *electronic)
{
  double ******HisH1=electronic->HisH1;
  double ******HisH2=electronic->HisH2;
  double ******ResidualH1=electronic->ResidualH1;
  double ******ResidualH2=electronic->ResidualH2;

  int Mc_AN,Gc_AN,Cwan,i,m,n,flag_nan;
  int seed;
  double alpha,d;
  OpenMXSolverRealTensor A,IA;
  double *coes;
  OpenMXSolverRealFieldView metric;
  char nanchar[300];

  /* small host work arrays (same shapes as the legacy path) */

  coes = OpenMX_Solver_OverwriteReal(scratch,"mixh-Pulay_Mixing_H_Inc-coes",List_YOUSO[39]);

  A = OpenMX_Solver_OverwriteView(scratch,"mixh-Pulay_Mixing_H_Inc-A",List_YOUSO[39],List_YOUSO[39]);

  IA = OpenMX_Solver_OverwriteView(scratch,"mixh-Pulay_Mixing_H_Inc-IA",List_YOUSO[39],List_YOUSO[39]);

  metric = OpenMX_Solver_Field(scratch,1,(size_t)Matomnum+1);
  for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){
    int tno;
    if (Mc_AN==0){
      tno = 1;
    }
    else{
      Gc_AN = M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];
      tno = Spe_Total_NO[Cwan];
    }
    OpenMX_Solver_FieldResize(metric,0,Mc_AN,tno);
  }

  /* metric used for the norm calculations (identical to the legacy path) */

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];
    Cwan = WhatSpecies[Gc_AN];
    for (i=0; i<Spe_Total_NO[Cwan]; i++){
      d = fabs(HisH1[0][0][Mc_AN][0][i][i]-ChemP);
      metric.blocks[Mc_AN].values[i] = 5.0/(d*d+5.0);
      state->inc.wblk[state->inc.blk_base[Mc_AN]+i] = metric.blocks[Mc_AN].values[i];
    }
  }

  /* residual shift (pointer rotation) + fresh slot 0, then the block-Gram
     ring update: only the new row on continuity, everything on a seed */

  seed = !(state->inc.valid && state->inc.last_scf_iter+1 == SCF_iter);

  MixH_RotatePtr6(ResidualH1, dim);
  if (SpinP_switch==3) MixH_RotatePtr6(ResidualH2, dim);
  MixH_Residual0Fused(state, NULL, electronic);

  if (seed){
    for (m=0; m<state->inc.nslot; m++) state->inc.perm[m] = m;
    for (m=0; m<dim; m++) MixH_IncComputeRow(scratch, state, m, m, electronic);
    state->inc.valid = 1;
  }
  else{
    int last = state->inc.perm[state->inc.nslot-1];
    for (m=state->inc.nslot-1; 0<m; m--) state->inc.perm[m] = state->inc.perm[m-1];
    state->inc.perm[0] = last;
    MixH_IncComputeRow(scratch, state, 0, dim-1, electronic);
  }
  state->inc.last_scf_iter = SCF_iter;

  /* assemble the metric-weighted Gram triangle and reduce it once */

  {
    int npair = dim*(dim+1)/2;
    int *pmv = OpenMX_Solver_Int(scratch,"mixh-Pulay_Mixing_H_Inc-pmv",(size_t)npair);
    int *pnv = OpenMX_Solver_Int(scratch,"mixh-Pulay_Mixing_H_Inc-pnv",(size_t)npair);
    double *tri_my = OpenMX_Solver_OverwriteReal(scratch,"mixh-Pulay_Mixing_H_Inc-tri_my",(size_t)npair);
    double *tri = OpenMX_Solver_OverwriteReal(scratch,"mixh-Pulay_Mixing_H_Inc-tri",(size_t)npair);
    int p = 0;

    if (pmv==NULL || pnv==NULL || tri_my==NULL || tri==NULL){
      MixH_AbortWithMessage("Mixing_H.c: failed to allocate the Gram pair tables.");
    }

    for (m=0; m<dim; m++){
      for (n=0; n<=m; n++){
        pmv[p] = m;
        pnv[p] = n;
        p++;
      }
    }

#pragma omp parallel for schedule(static)
    for (p=0; p<npair; p++){
      const int nblk = state->inc.nblk;
      const double *Sv = state->inc.S
        + ((size_t)state->inc.perm[pmv[p]]*(size_t)state->inc.nslot
           + (size_t)state->inc.perm[pnv[p]])*(size_t)nblk;
      double s = 0.0;
      int blk;
      for (blk=0; blk<nblk; blk++) s += state->inc.wblk[blk]*Sv[blk];
      tri_my[p] = s;
    }

    MPI_Allreduce(tri_my, tri, npair, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

    for (p=0; p<npair; p++){
      openmx_solver_real_matrix_row(&A,pmv[p])[pnv[p]] = tri[p];
      openmx_solver_real_matrix_row(&A,pnv[p])[pmv[p]] = tri[p];
    }




  }

  NormRD[0] = openmx_solver_real_matrix_row(&A,0)[0]/(double)atomnum;

  for (m=1; m<=dim; m++){
    openmx_solver_real_matrix_row(&A,m-1)[dim] = -1.0;
    openmx_solver_real_matrix_row(&A,dim)[m-1] = -1.0;
  }
  openmx_solver_real_matrix_row(&A,dim)[dim] = 0.0;

  Inverse(scratch, dim,&A,&IA);

  for (m=1; m<=dim; m++){
    coes[m] = -openmx_solver_real_matrix_row(&IA,m-1)[dim];
  }

  /* check "nan", "NaN", "inf" or "Inf" (identical to the legacy path) */

  flag_nan = 0;
  for (m=1; m<=dim; m++){

    sprintf(nanchar,"%8.4f",coes[m]);
    if (   strstr(nanchar,"nan")!=NULL || strstr(nanchar,"NaN")!=NULL
	|| strstr(nanchar,"inf")!=NULL || strstr(nanchar,"Inf")!=NULL){

      flag_nan = 1;
    }
  }

  if (flag_nan==1){
    for (m=1; m<=dim; m++){
      coes[m] = 0.0;
    }
    coes[1] = 0.05;
    coes[2] = 0.95;
  }

  if (1.0e-1<=NormRD[0])
    alpha = 0.5;
  else if (1.0e-2<=NormRD[0] && NormRD[0]<1.0e-1)
    alpha = 0.6;
  else if (1.0e-3<=NormRD[0] && NormRD[0]<1.0e-2)
    alpha = 0.7;
  else if (1.0e-4<=NormRD[0] && NormRD[0]<1.0e-3)
    alpha = 0.8;
  else
    alpha = 1.0;

  MixH_IncMixSweep(dim, alpha, coes, electronic);

  /* Hamiltonian history shift (pointer rotation) and save of the mixed H */

  MixH_RotatePtr6(HisH1, dim);
  if (SpinP_switch==3) MixH_RotatePtr6(HisH2, dim);
  MixH_SaveHistory0(electronic);

  /* Invocation owner releases the host work arrays. */

  /* Invocation owner releases metric rows. */

  (void)MD_iter;
  (void)SCF_iter0;
}

static void Pulay_Mixing_H_GPU(OpenMXSolverInvocation *scratch, MixingHState *state, int MD_iter, int SCF_iter, int SCF_iter0, int dim, const OpenMXElectronicViews *electronic)
{
  double ******HisH1=electronic->HisH1;
  double ******HisH2=electronic->HisH2;
  double ******ResidualH1=electronic->ResidualH1;
  double ******ResidualH2=electronic->ResidualH2;

  int Mc_AN,Gc_AN,Cwan,i,m,n,flag_nan;
  int seed;
  double alpha,d;
  OpenMXSolverRealTensor A,IA;
  double *coes;
  OpenMXSolverRealFieldView metric;
  char nanchar[300];

  /* allocation of the small host work arrays (same shapes as the CPU path) */

  coes = OpenMX_Solver_OverwriteReal(scratch,"mixh-Pulay_Mixing_H_GPU-coes",List_YOUSO[39]);

  A = OpenMX_Solver_OverwriteView(scratch,"mixh-Pulay_Mixing_H_GPU-A",List_YOUSO[39],List_YOUSO[39]);

  IA = OpenMX_Solver_OverwriteView(scratch,"mixh-Pulay_Mixing_H_GPU-IA",List_YOUSO[39],List_YOUSO[39]);

  metric = OpenMX_Solver_Field(scratch,1,(size_t)Matomnum+1);
  for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){
    int tno;
    if (Mc_AN==0){
      tno = 1;
    }
    else{
      Gc_AN = M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];
      tno = Spe_Total_NO[Cwan];
    }
    OpenMX_Solver_FieldResize(metric,0,Mc_AN,tno);
  }

  /* metric used for the norm calculations (identical to the CPU path) */

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];
    Cwan = WhatSpecies[Gc_AN];
    for (i=0; i<Spe_Total_NO[Cwan]; i++){
      d = fabs(HisH1[0][0][Mc_AN][0][i][i]-ChemP);
      metric.blocks[Mc_AN].values[i] = 5.0/(d*d+5.0);
    }
  }

  /* shift the residual history (pointer rotation == the CPU content shift)
     and compute the fresh residual slot 0 on the host */

  seed = !(state->gpu.valid && state->gpu.last_scf_iter+1 == SCF_iter);

  MixH_RotatePtr6(ResidualH1, dim);
  if (SpinP_switch==3) MixH_RotatePtr6(ResidualH2, dim);
  MixH_Residual0Fused(state, state->gpu.h_flat, electronic);

  /* bring the device ring up to date */

  if (seed){
    for (m=0; m<state->gpu.nslot; m++) state->gpu.perm[m] = m;
    if (0U < state->gpu.L){
      MixH_Upload(state, 0);
      for (m=1; m<dim; m++){
        MixH_PackResidualSlot(state, m, state->gpu.h_flat, electronic);
        MixH_Upload(state, m);
      }
    }
    state->gpu.valid = 1;
  }
  else{
    int last = state->gpu.perm[state->gpu.nslot-1];
    for (m=state->gpu.nslot-1; 0<m; m--) state->gpu.perm[m] = state->gpu.perm[m-1];
    state->gpu.perm[0] = last;
    if (0U < state->gpu.L){
      MixH_Upload(state, state->gpu.perm[0]);
    }
  }
  state->gpu.last_scf_iter = SCF_iter;

  /* metric weights */

  if (0U < state->gpu.L){
    MixH_PackMetricW(state, metric, state->gpu.h_flat);
    MixH_Upload(state, -1);
  }

  /* residual Gram matrix: one device kernel + one Allreduce */

  {
    int npair = dim*(dim+1)/2;
    int *pm = OpenMX_Solver_Int(scratch,"mixh-Pulay_Mixing_H_GPU-pm",(size_t)npair);
    int *pn = OpenMX_Solver_Int(scratch,"mixh-Pulay_Mixing_H_GPU-pn",(size_t)npair);
    double *tri_my = OpenMX_Solver_OverwriteReal(scratch,"mixh-Pulay_Mixing_H_GPU-tri_my",(size_t)npair);
    double *tri = OpenMX_Solver_OverwriteReal(scratch,"mixh-Pulay_Mixing_H_GPU-tri",(size_t)npair);
    int p = 0;

    if (pm==NULL || pn==NULL || tri_my==NULL || tri==NULL){
      MixH_AbortWithMessage("Mixing_H.c: failed to allocate the Gram pair tables.");
    }

    for (m=0; m<dim; m++){
      for (n=0; n<=m; n++){
        pm[p] = m;
        pn[p] = n;
        p++;
      }
    }

    MixH_GramDevice(scratch, state, dim, npair, pm, pn, state->gpu.perm, tri_my);
    MPI_Allreduce(tri_my, tri, npair, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

    for (p=0; p<npair; p++){
      openmx_solver_real_matrix_row(&A,pm[p])[pn[p]] = tri[p];
      openmx_solver_real_matrix_row(&A,pn[p])[pm[p]] = tri[p];
    }




  }

  NormRD[0] = openmx_solver_real_matrix_row(&A,0)[0]/(double)atomnum;

  for (m=1; m<=dim; m++){
    openmx_solver_real_matrix_row(&A,m-1)[dim] = -1.0;
    openmx_solver_real_matrix_row(&A,dim)[m-1] = -1.0;
  }
  openmx_solver_real_matrix_row(&A,dim)[dim] = 0.0;

  Inverse(scratch, dim,&A,&IA);

  for (m=1; m<=dim; m++){
    coes[m] = -openmx_solver_real_matrix_row(&IA,m-1)[dim];
  }

  /* check "nan", "NaN", "inf" or "Inf" (identical to the CPU path) */

  flag_nan = 0;
  for (m=1; m<=dim; m++){

    sprintf(nanchar,"%8.4f",coes[m]);
    if (   strstr(nanchar,"nan")!=NULL || strstr(nanchar,"NaN")!=NULL
	|| strstr(nanchar,"inf")!=NULL || strstr(nanchar,"Inf")!=NULL){

      flag_nan = 1;
    }
  }

  if (flag_nan==1){
    for (m=1; m<=dim; m++){
      coes[m] = 0.0;
    }
    coes[1] = 0.05;
    coes[2] = 0.95;
  }

  /* optimum residual on the device, then the mixing on the host */

  {
    double *cshift = OpenMX_Solver_OverwriteReal(scratch,"mixh-Pulay_Mixing_H_GPU-cshift",(size_t)dim);
    if (cshift==NULL){
      MixH_AbortWithMessage("Mixing_H.c: failed to allocate the coefficient buffer.");
    }
    for (m=0; m<dim; m++) cshift[m] = coes[m+1];
    MixH_OptResidualDevice(scratch, state, dim, state->gpu.perm, cshift);

  }

  if (1.0e-1<=NormRD[0])
    alpha = 0.5;
  else if (1.0e-2<=NormRD[0] && NormRD[0]<1.0e-1)
    alpha = 0.6;
  else if (1.0e-3<=NormRD[0] && NormRD[0]<1.0e-2)
    alpha = 0.7;
  else if (1.0e-4<=NormRD[0] && NormRD[0]<1.0e-3)
    alpha = 0.8;
  else
    alpha = 1.0;

  MixH_MixFromHistory(state, dim, alpha, coes, state->gpu.h_r, electronic);

  /* shift the Hamiltonian history (pointer rotation) and save the mixed H */

  MixH_RotatePtr6(HisH1, dim);
  if (SpinP_switch==3) MixH_RotatePtr6(HisH2, dim);
  MixH_SaveHistory0(electronic);

  /* Invocation owner releases the host work arrays. */

  /* Invocation owner releases metric rows. */

  (void)MD_iter;
  (void)SCF_iter0;
}

void Pulay_Mixing_H(OpenMXSolverInvocation *scratch, MixingHState *state, int MD_iter, int SCF_iter, int SCF_iter0 , const OpenMXElectronicViews *electronic)
{
  double *****H=electronic->H;
  double ******HisH1=electronic->HisH1;
  double ******HisH2=electronic->HisH2;
  double ******ResidualH1=electronic->ResidualH1;
  double ******ResidualH2=electronic->ResidualH2;
  double *****iHNL=electronic->iHNL;

  int Mc_AN,Gc_AN,Cwan,Hwan,h_AN,Gh_AN,i,j,spin;
  int dim,m,n,flag_nan,tno;
  double my_sum,tmp1,tmp2,alpha,max_diff,d;
  double r,r10,r11,r12,r13,r20,r21,r22;
  double h,h10,h11,h12,h13,h20,h21,h22;
  OpenMXSolverRealTensor A,IA;
  double *coes;
  OpenMXSolverRealFieldView metric;
  char nanchar[300];

  /****************************************************
       determination of dimension of the subspace
  ****************************************************/

  if (SCF_iter<=Num_Mixing_pDM) dim = SCF_iter-1;
  else                          dim = Num_Mixing_pDM;

  /* GPU fast path; MPI-collective on every rank.  On a negative device
     preflight the original CPU code below runs unchanged. */

  if (scf_eigen_lib_flag == GPUSOLVER && 1<=dim){
    int myid_mixh, tier;
    MPI_Comm_rank(mpi_comm_level1,&myid_mixh);
    tier = MixH_GpuPreflight(state, MD_iter, myid_mixh);
    if (tier == 1){
      Pulay_Mixing_H_GPU(scratch, state, MD_iter, SCF_iter, SCF_iter0, dim, electronic);
      return;
    }
    if (tier == 2){
      Pulay_Mixing_H_Inc(scratch, state, MD_iter, SCF_iter, SCF_iter0, dim, electronic);
      return;
    }
  }

  /****************************************************
                allocation of arrays 
  ****************************************************/

  coes = OpenMX_Solver_OverwriteReal(scratch,"mixh-Pulay_Mixing_H-coes",List_YOUSO[39]);

  A = OpenMX_Solver_OverwriteView(scratch,"mixh-Pulay_Mixing_H-A",List_YOUSO[39],List_YOUSO[39]);

  IA = OpenMX_Solver_OverwriteView(scratch,"mixh-Pulay_Mixing_H-IA",List_YOUSO[39],List_YOUSO[39]);

  metric = OpenMX_Solver_Field(scratch,1,(size_t)Matomnum+1);
  for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){
    if (Mc_AN==0){
      tno = 1; 
    }
    else{
      Gc_AN = M2G[Mc_AN];    
      Cwan = WhatSpecies[Gc_AN];
      tno = Spe_Total_NO[Cwan];
    }
    OpenMX_Solver_FieldResize(metric,0,Mc_AN,tno);
  }  

  /****************************************************
     determine metric used for calculations of norm
  ****************************************************/

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];    
    Cwan = WhatSpecies[Gc_AN];
    for (i=0; i<Spe_Total_NO[Cwan]; i++){
      d = fabs(HisH1[0][0][Mc_AN][0][i][i]-ChemP);
      metric.blocks[Mc_AN].values[i] = 5.0/(d*d+5.0);
    }
  }

  /****************************************************
                 shift the residual H
  ****************************************************/

  if (SpinP_switch==3){

    /* shift the residual Hamiltonian */

    for (m=dim; 0<m; m--){
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	Gc_AN = M2G[Mc_AN];    
	Cwan = WhatSpecies[Gc_AN];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  for (i=0; i<Spe_Total_NO[Cwan]; i++){
	    for (j=0; j<Spe_Total_NO[Hwan]; j++){

	      ResidualH1[m][0][Mc_AN][h_AN][i][j] = ResidualH1[m-1][0][Mc_AN][h_AN][i][j];
	      ResidualH1[m][1][Mc_AN][h_AN][i][j] = ResidualH1[m-1][1][Mc_AN][h_AN][i][j];
	      ResidualH1[m][2][Mc_AN][h_AN][i][j] = ResidualH1[m-1][2][Mc_AN][h_AN][i][j];
	      ResidualH1[m][3][Mc_AN][h_AN][i][j] = ResidualH1[m-1][3][Mc_AN][h_AN][i][j];

	      ResidualH2[m][0][Mc_AN][h_AN][i][j] = ResidualH2[m-1][0][Mc_AN][h_AN][i][j];
	      ResidualH2[m][1][Mc_AN][h_AN][i][j] = ResidualH2[m-1][1][Mc_AN][h_AN][i][j];
	      ResidualH2[m][2][Mc_AN][h_AN][i][j] = ResidualH2[m-1][2][Mc_AN][h_AN][i][j];
	    }
	  }
	}
      }
    }

    /* calculate the current residual Hamiltonian */

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];    
      Cwan = WhatSpecies[Gc_AN];
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	Gh_AN = natn[Gc_AN][h_AN];
	Hwan = WhatSpecies[Gh_AN];
	for (i=0; i<Spe_Total_NO[Cwan]; i++){
	  for (j=0; j<Spe_Total_NO[Hwan]; j++){

	    ResidualH1[0][0][Mc_AN][h_AN][i][j] = H[0][Mc_AN][h_AN][i][j] - HisH1[0][0][Mc_AN][h_AN][i][j];
	    ResidualH1[0][1][Mc_AN][h_AN][i][j] = H[1][Mc_AN][h_AN][i][j] - HisH1[0][1][Mc_AN][h_AN][i][j];
	    ResidualH1[0][2][Mc_AN][h_AN][i][j] = H[2][Mc_AN][h_AN][i][j] - HisH1[0][2][Mc_AN][h_AN][i][j];
	    ResidualH1[0][3][Mc_AN][h_AN][i][j] = H[3][Mc_AN][h_AN][i][j] - HisH1[0][3][Mc_AN][h_AN][i][j];

	    ResidualH2[0][0][Mc_AN][h_AN][i][j] = iHNL[0][Mc_AN][h_AN][i][j] - HisH2[0][0][Mc_AN][h_AN][i][j];
	    ResidualH2[0][1][Mc_AN][h_AN][i][j] = iHNL[1][Mc_AN][h_AN][i][j] - HisH2[0][1][Mc_AN][h_AN][i][j];
	    ResidualH2[0][2][Mc_AN][h_AN][i][j] = iHNL[2][Mc_AN][h_AN][i][j] - HisH2[0][2][Mc_AN][h_AN][i][j];
	  }
	}
      }
    }

  }

  else{

    /* shift the residual Hamiltonian */

    for (m=dim; 0<m; m--){
      for (spin=0; spin<=SpinP_switch; spin++){
	for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	  Gc_AN = M2G[Mc_AN];    
	  Cwan = WhatSpecies[Gc_AN];
	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	    Gh_AN = natn[Gc_AN][h_AN];
	    Hwan = WhatSpecies[Gh_AN];
	    for (i=0; i<Spe_Total_NO[Cwan]; i++){
	      for (j=0; j<Spe_Total_NO[Hwan]; j++){
		ResidualH1[m][spin][Mc_AN][h_AN][i][j] = ResidualH1[m-1][spin][Mc_AN][h_AN][i][j];
	      }
	    }
	  }
	}
      }
    }

    /* calculate the current residual Hamiltonian */

    for (spin=0; spin<=SpinP_switch; spin++){
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	Gc_AN = M2G[Mc_AN];    
	Cwan = WhatSpecies[Gc_AN];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  for (i=0; i<Spe_Total_NO[Cwan]; i++){
	    for (j=0; j<Spe_Total_NO[Hwan]; j++){
  	      ResidualH1[0][spin][Mc_AN][h_AN][i][j] = H[spin][Mc_AN][h_AN][i][j] - HisH1[0][spin][Mc_AN][h_AN][i][j];
	    }
	  }
	}
      }
    }

  } /* else */

  /****************************************************
          calculation of the residual matrix
  ****************************************************/

  for (m=0; m<dim; m++){
    for (n=0; n<dim; n++){

      my_sum = 0.0;

      if (SpinP_switch==3){

	for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	  Gc_AN = M2G[Mc_AN];    
	  Cwan = WhatSpecies[Gc_AN];
	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	    Gh_AN = natn[Gc_AN][h_AN];
	    Hwan = WhatSpecies[Gh_AN];
	    for (i=0; i<Spe_Total_NO[Cwan]; i++){
	      for (j=0; j<Spe_Total_NO[Hwan]; j++){

		tmp1 = ResidualH1[m][0][Mc_AN][h_AN][i][j];
		tmp2 = ResidualH1[n][0][Mc_AN][h_AN][i][j];
                my_sum += metric.blocks[Mc_AN].values[i]*tmp1*tmp2;

		tmp1 = ResidualH1[m][1][Mc_AN][h_AN][i][j];
		tmp2 = ResidualH1[n][1][Mc_AN][h_AN][i][j];
                my_sum += metric.blocks[Mc_AN].values[i]*tmp1*tmp2;

		tmp1 = ResidualH1[m][2][Mc_AN][h_AN][i][j];
		tmp2 = ResidualH1[n][2][Mc_AN][h_AN][i][j];
                my_sum += metric.blocks[Mc_AN].values[i]*tmp1*tmp2;

		tmp1 = ResidualH1[m][3][Mc_AN][h_AN][i][j];
		tmp2 = ResidualH1[n][3][Mc_AN][h_AN][i][j];
                my_sum += metric.blocks[Mc_AN].values[i]*tmp1*tmp2;

		tmp1 = ResidualH2[m][0][Mc_AN][h_AN][i][j];
		tmp2 = ResidualH2[n][0][Mc_AN][h_AN][i][j];
                my_sum += metric.blocks[Mc_AN].values[i]*tmp1*tmp2;

		tmp1 = ResidualH2[m][1][Mc_AN][h_AN][i][j];
		tmp2 = ResidualH2[n][1][Mc_AN][h_AN][i][j];
                my_sum += metric.blocks[Mc_AN].values[i]*tmp1*tmp2;

		tmp1 = ResidualH2[m][2][Mc_AN][h_AN][i][j];
		tmp2 = ResidualH2[n][2][Mc_AN][h_AN][i][j];
                my_sum += metric.blocks[Mc_AN].values[i]*tmp1*tmp2;
	      }
	    }
	  }
	}

      } /* if (SpinP_switch==3 */

      else{

	for (spin=0; spin<=SpinP_switch; spin++){
	  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	    Gc_AN = M2G[Mc_AN];    
	    Cwan = WhatSpecies[Gc_AN];
	    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	      Gh_AN = natn[Gc_AN][h_AN];
	      Hwan = WhatSpecies[Gh_AN];
	      for (i=0; i<Spe_Total_NO[Cwan]; i++){
		for (j=0; j<Spe_Total_NO[Hwan]; j++){
		  tmp1 = ResidualH1[m][spin][Mc_AN][h_AN][i][j];
		  tmp2 = ResidualH1[n][spin][Mc_AN][h_AN][i][j];
                  my_sum += metric.blocks[Mc_AN].values[i]*tmp1*tmp2;
		}
	      }
	    }
	  }
	}

      } /* else */

      MPI_Allreduce(&my_sum, &openmx_solver_real_matrix_row(&A,m)[n], 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
      openmx_solver_real_matrix_row(&A,n)[m] = openmx_solver_real_matrix_row(&A,m)[n];

    } /* n */
  } /* m */

  NormRD[0] = openmx_solver_real_matrix_row(&A,0)[0]/(double)atomnum;

  for (m=1; m<=dim; m++){
    openmx_solver_real_matrix_row(&A,m-1)[dim] = -1.0;
    openmx_solver_real_matrix_row(&A,dim)[m-1] = -1.0;
  }
  openmx_solver_real_matrix_row(&A,dim)[dim] = 0.0;

  Inverse(scratch, dim,&A,&IA);

  for (m=1; m<=dim; m++){
    coes[m] = -openmx_solver_real_matrix_row(&IA,m-1)[dim];
  }

  /****************************************************
            check "nan", "NaN", "inf" or "Inf"
  ****************************************************/

  flag_nan = 0;
  for (m=1; m<=dim; m++){

    sprintf(nanchar,"%8.4f",coes[m]);
    if (   strstr(nanchar,"nan")!=NULL || strstr(nanchar,"NaN")!=NULL 
	|| strstr(nanchar,"inf")!=NULL || strstr(nanchar,"Inf")!=NULL){

      flag_nan = 1;
    }
  }

  if (flag_nan==1){
    for (m=1; m<=dim; m++){
      coes[m] = 0.0;
    }
    coes[1] = 0.05;
    coes[2] = 0.95;
  }

  /****************************************************
      calculation of optimum residual Hamiltonian
  ****************************************************/

  if (SpinP_switch==3){

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];    
      Cwan = WhatSpecies[Gc_AN];
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	Gh_AN = natn[Gc_AN][h_AN];
	Hwan = WhatSpecies[Gh_AN];
	for (i=0; i<Spe_Total_NO[Cwan]; i++){
	  for (j=0; j<Spe_Total_NO[Hwan]; j++){

            r10 = 0.0; 
            r11 = 0.0;
            r12 = 0.0;
            r13 = 0.0;

            r20 = 0.0; 
            r21 = 0.0;
            r22 = 0.0;

	    for (m=0; m<dim; m++){

	      r10 += ResidualH1[m][0][Mc_AN][h_AN][i][j]*coes[m+1];
	      r11 += ResidualH1[m][1][Mc_AN][h_AN][i][j]*coes[m+1];
	      r12 += ResidualH1[m][2][Mc_AN][h_AN][i][j]*coes[m+1];
	      r13 += ResidualH1[m][3][Mc_AN][h_AN][i][j]*coes[m+1];

	      r20 += ResidualH2[m][0][Mc_AN][h_AN][i][j]*coes[m+1];
	      r21 += ResidualH2[m][1][Mc_AN][h_AN][i][j]*coes[m+1];
	      r22 += ResidualH2[m][2][Mc_AN][h_AN][i][j]*coes[m+1];
	    }

            /* optimum Residual H is stored in ResidualH1[dim] and ResidualH2[dim] */

	    ResidualH1[dim][0][Mc_AN][h_AN][i][j] = r10;
	    ResidualH1[dim][1][Mc_AN][h_AN][i][j] = r11;
	    ResidualH1[dim][2][Mc_AN][h_AN][i][j] = r12;
	    ResidualH1[dim][3][Mc_AN][h_AN][i][j] = r13;

	    ResidualH2[dim][0][Mc_AN][h_AN][i][j] = r20;
	    ResidualH2[dim][1][Mc_AN][h_AN][i][j] = r21;
	    ResidualH2[dim][2][Mc_AN][h_AN][i][j] = r22;

	  }
	}
      }
    }

  }

  else{

    for (spin=0; spin<=SpinP_switch; spin++){
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	Gc_AN = M2G[Mc_AN];    
	Cwan = WhatSpecies[Gc_AN];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  for (i=0; i<Spe_Total_NO[Cwan]; i++){
	    for (j=0; j<Spe_Total_NO[Hwan]; j++){

	      r = 0.0; 
	      for (m=0; m<dim; m++){
		r += ResidualH1[m][spin][Mc_AN][h_AN][i][j]*coes[m+1];
	      }

              /* optimum Residual H is stored in ResidualH1[dim] */

              ResidualH1[dim][spin][Mc_AN][h_AN][i][j] = r;

	    }
	  }
	}
      }
    }
  }

  /****************************************************
                   mixing of Hamiltonian
  ****************************************************/

  if (1.0e-1<=NormRD[0])
    alpha = 0.5;
  else if (1.0e-2<=NormRD[0] && NormRD[0]<1.0e-1)
    alpha = 0.6;
  else if (1.0e-3<=NormRD[0] && NormRD[0]<1.0e-2)
    alpha = 0.7;
  else if (1.0e-4<=NormRD[0] && NormRD[0]<1.0e-3)
    alpha = 0.8;
  else
    alpha = 1.0;

  if (SpinP_switch==3){

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];    
      Cwan = WhatSpecies[Gc_AN];
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	Gh_AN = natn[Gc_AN][h_AN];
	Hwan = WhatSpecies[Gh_AN];
	for (i=0; i<Spe_Total_NO[Cwan]; i++){
	  for (j=0; j<Spe_Total_NO[Hwan]; j++){

            h10 = 0.0; 
            h11 = 0.0;
            h12 = 0.0;
            h13 = 0.0;

            h20 = 0.0; 
            h21 = 0.0;
            h22 = 0.0;

	    for (m=0; m<dim; m++){

	      h10 += HisH1[m][0][Mc_AN][h_AN][i][j]*coes[m+1];
	      h11 += HisH1[m][1][Mc_AN][h_AN][i][j]*coes[m+1];
	      h12 += HisH1[m][2][Mc_AN][h_AN][i][j]*coes[m+1];
	      h13 += HisH1[m][3][Mc_AN][h_AN][i][j]*coes[m+1];

	      h20 += HisH2[m][0][Mc_AN][h_AN][i][j]*coes[m+1];
	      h21 += HisH2[m][1][Mc_AN][h_AN][i][j]*coes[m+1];
	      h22 += HisH2[m][2][Mc_AN][h_AN][i][j]*coes[m+1];
	    }

	    H[0][Mc_AN][h_AN][i][j] = h10 + alpha*ResidualH1[dim][0][Mc_AN][h_AN][i][j];
	    H[1][Mc_AN][h_AN][i][j] = h11 + alpha*ResidualH1[dim][1][Mc_AN][h_AN][i][j];
	    H[2][Mc_AN][h_AN][i][j] = h12 + alpha*ResidualH1[dim][2][Mc_AN][h_AN][i][j];
	    H[3][Mc_AN][h_AN][i][j] = h13 + alpha*ResidualH1[dim][3][Mc_AN][h_AN][i][j];
             
	    iHNL[0][Mc_AN][h_AN][i][j] = h20 + alpha*ResidualH2[dim][0][Mc_AN][h_AN][i][j];
	    iHNL[1][Mc_AN][h_AN][i][j] = h21 + alpha*ResidualH2[dim][1][Mc_AN][h_AN][i][j];
	    iHNL[2][Mc_AN][h_AN][i][j] = h22 + alpha*ResidualH2[dim][2][Mc_AN][h_AN][i][j];

	  }
	}
      }
    }

  }

  else{

    for (spin=0; spin<=SpinP_switch; spin++){
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	Gc_AN = M2G[Mc_AN];    
	Cwan = WhatSpecies[Gc_AN];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  for (i=0; i<Spe_Total_NO[Cwan]; i++){
	    for (j=0; j<Spe_Total_NO[Hwan]; j++){

	      r = 0.0; 
	      h = 0.0;

	      for (m=0; m<dim; m++){
		r += ResidualH1[m][spin][Mc_AN][h_AN][i][j]*coes[m+1];
		h += HisH1[m][spin][Mc_AN][h_AN][i][j]*coes[m+1];
	      }

	      H[spin][Mc_AN][h_AN][i][j] = h + alpha*r;
	    }
	  }
	}
      }
    }
  }

  /****************************************************
                  shifting of Hamiltonian
  ****************************************************/

  if (SpinP_switch==3){

    /* shift the current Hamiltonian */

    for (m=dim; 0<m; m--){
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	Gc_AN = M2G[Mc_AN];    
	Cwan = WhatSpecies[Gc_AN];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  for (i=0; i<Spe_Total_NO[Cwan]; i++){
	    for (j=0; j<Spe_Total_NO[Hwan]; j++){

	      HisH1[m][0][Mc_AN][h_AN][i][j] = HisH1[m-1][0][Mc_AN][h_AN][i][j];
	      HisH1[m][1][Mc_AN][h_AN][i][j] = HisH1[m-1][1][Mc_AN][h_AN][i][j];
	      HisH1[m][2][Mc_AN][h_AN][i][j] = HisH1[m-1][2][Mc_AN][h_AN][i][j];
	      HisH1[m][3][Mc_AN][h_AN][i][j] = HisH1[m-1][3][Mc_AN][h_AN][i][j];

	      HisH2[m][0][Mc_AN][h_AN][i][j] = HisH2[m-1][0][Mc_AN][h_AN][i][j];
	      HisH2[m][1][Mc_AN][h_AN][i][j] = HisH2[m-1][1][Mc_AN][h_AN][i][j];
	      HisH2[m][2][Mc_AN][h_AN][i][j] = HisH2[m-1][2][Mc_AN][h_AN][i][j];
	    }
	  }
	}
      }
    }

    /* save the current Hamiltonian */

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];    
      Cwan = WhatSpecies[Gc_AN];
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	Gh_AN = natn[Gc_AN][h_AN];
	Hwan = WhatSpecies[Gh_AN];
	for (i=0; i<Spe_Total_NO[Cwan]; i++){
	  for (j=0; j<Spe_Total_NO[Hwan]; j++){

	    HisH1[0][0][Mc_AN][h_AN][i][j] = H[0][Mc_AN][h_AN][i][j];
	    HisH1[0][1][Mc_AN][h_AN][i][j] = H[1][Mc_AN][h_AN][i][j];
	    HisH1[0][2][Mc_AN][h_AN][i][j] = H[2][Mc_AN][h_AN][i][j];
	    HisH1[0][3][Mc_AN][h_AN][i][j] = H[3][Mc_AN][h_AN][i][j];

	    HisH2[0][0][Mc_AN][h_AN][i][j] = iHNL[0][Mc_AN][h_AN][i][j];
	    HisH2[0][1][Mc_AN][h_AN][i][j] = iHNL[1][Mc_AN][h_AN][i][j];
	    HisH2[0][2][Mc_AN][h_AN][i][j] = iHNL[2][Mc_AN][h_AN][i][j];
	  }
	}
      }
    }

  }

  else {

    /* shift the current Hamiltonian */

    for (m=dim; 0<m; m--){
      for (spin=0; spin<=SpinP_switch; spin++){
	for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	  Gc_AN = M2G[Mc_AN];    
	  Cwan = WhatSpecies[Gc_AN];
	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	    Gh_AN = natn[Gc_AN][h_AN];
	    Hwan = WhatSpecies[Gh_AN];
	    for (i=0; i<Spe_Total_NO[Cwan]; i++){
	      for (j=0; j<Spe_Total_NO[Hwan]; j++){
		HisH1[m][spin][Mc_AN][h_AN][i][j] = HisH1[m-1][spin][Mc_AN][h_AN][i][j];
	      }
	    }
	  }
	}
      }
    }

    /* save the current Hamiltonian */

    for (spin=0; spin<=SpinP_switch; spin++){
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	Gc_AN = M2G[Mc_AN];    
	Cwan = WhatSpecies[Gc_AN];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  for (i=0; i<Spe_Total_NO[Cwan]; i++){
	    for (j=0; j<Spe_Total_NO[Hwan]; j++){
	      HisH1[0][spin][Mc_AN][h_AN][i][j] = H[spin][Mc_AN][h_AN][i][j];
	    }
	  }
	}
      }
    }

  } /* else */

  /* Invocation owner releases numerical scratch. */

  /* Invocation owner releases metric rows. */

}



void Simple_Mixing_H(OpenMXMixingHistory *history, int MD_iter, int SCF_iter, int SCF_iter0 , const OpenMXElectronicViews *electronic)
{
  double ******DM=electronic->DM;
  double *****H=electronic->H;
  double ******HisH1=electronic->HisH1;
  double ******HisH2=electronic->HisH2;
  double ******ResidualDM=electronic->ResidualDM;
  double ******ResidualH1=electronic->ResidualH1;
  double ******ResidualH2=electronic->ResidualH2;
  double ******iDM=electronic->iDM;
  double *****iHNL=electronic->iHNL;
  double ******iResidualDM=electronic->iResidualDM;

  int m,Mc_AN,Gc_AN,Cwan,spin,dim;
  int i,j,h_AN,Gh_AN,Hwan,ian,jan,n;
  double w1,w2,My_Norm,Norm;
  double d0,d1,d2,d3,d,tmp0;
  double Mix_wgt,Min_Weight,Max_Weight;
  int numprocs,myid,ID;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /* start... */

  Min_Weight = Min_Mixing_weight;
  if (history->consecutive==-1){
    Max_Weight = Max_Mixing_weight;
    history->adaptive_max_weight = Max_Mixing_weight;
  }
  else if (history->consecutive==1000){  /* past 3 */
    history->adaptive_max_weight = 2.0*history->adaptive_max_weight;
    if (0.7<history->adaptive_max_weight) history->adaptive_max_weight = 0.7;
    Max_Weight = history->adaptive_max_weight;
    history->consecutive = 0;
  }
  else{
    Max_Weight = history->adaptive_max_weight;
  }

  /* determination of dim */

  if (SCF_iter<Num_Mixing_pDM) dim = SCF_iter;
  else                         dim = Num_Mixing_pDM;

  /****************************************************
                  shift the residual H
  ****************************************************/

  if (Mixing_switch!=1 && Mixing_switch!=6){

    if (SpinP_switch==3){

      /* shift the residual Hamiltonian */

      for (m=(dim-1); 0<m; m--){
	for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	  Gc_AN = M2G[Mc_AN];    
	  Cwan = WhatSpecies[Gc_AN];
	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	    Gh_AN = natn[Gc_AN][h_AN];
	    Hwan = WhatSpecies[Gh_AN];
	    for (i=0; i<Spe_Total_NO[Cwan]; i++){
	      for (j=0; j<Spe_Total_NO[Hwan]; j++){

		ResidualH1[m][0][Mc_AN][h_AN][i][j] = ResidualH1[m-1][0][Mc_AN][h_AN][i][j];
		ResidualH1[m][1][Mc_AN][h_AN][i][j] = ResidualH1[m-1][1][Mc_AN][h_AN][i][j];
		ResidualH1[m][2][Mc_AN][h_AN][i][j] = ResidualH1[m-1][2][Mc_AN][h_AN][i][j];
		ResidualH1[m][3][Mc_AN][h_AN][i][j] = ResidualH1[m-1][3][Mc_AN][h_AN][i][j];

		ResidualH2[m][0][Mc_AN][h_AN][i][j] = ResidualH2[m-1][0][Mc_AN][h_AN][i][j];
		ResidualH2[m][1][Mc_AN][h_AN][i][j] = ResidualH2[m-1][1][Mc_AN][h_AN][i][j];
		ResidualH2[m][2][Mc_AN][h_AN][i][j] = ResidualH2[m-1][2][Mc_AN][h_AN][i][j];
	      }
	    }
	  }
	}
      }

      /* calculate the current residual Hamiltonian */

      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	Gc_AN = M2G[Mc_AN];    
	Cwan = WhatSpecies[Gc_AN];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  for (i=0; i<Spe_Total_NO[Cwan]; i++){
	    for (j=0; j<Spe_Total_NO[Hwan]; j++){

	      ResidualH1[0][0][Mc_AN][h_AN][i][j] = H[0][Mc_AN][h_AN][i][j] - HisH1[0][0][Mc_AN][h_AN][i][j];
	      ResidualH1[0][1][Mc_AN][h_AN][i][j] = H[1][Mc_AN][h_AN][i][j] - HisH1[0][1][Mc_AN][h_AN][i][j];
	      ResidualH1[0][2][Mc_AN][h_AN][i][j] = H[2][Mc_AN][h_AN][i][j] - HisH1[0][2][Mc_AN][h_AN][i][j];
	      ResidualH1[0][3][Mc_AN][h_AN][i][j] = H[3][Mc_AN][h_AN][i][j] - HisH1[0][3][Mc_AN][h_AN][i][j];

	      ResidualH2[0][0][Mc_AN][h_AN][i][j] = iHNL[0][Mc_AN][h_AN][i][j] - HisH2[0][0][Mc_AN][h_AN][i][j];
	      ResidualH2[0][1][Mc_AN][h_AN][i][j] = iHNL[1][Mc_AN][h_AN][i][j] - HisH2[0][1][Mc_AN][h_AN][i][j];
	      ResidualH2[0][2][Mc_AN][h_AN][i][j] = iHNL[2][Mc_AN][h_AN][i][j] - HisH2[0][2][Mc_AN][h_AN][i][j];
	    }
	  }
	}
      }

    }

    else{

      /* shift the residual Hamiltonian */

      for (m=(dim-1); 0<m; m--){
	for (spin=0; spin<=SpinP_switch; spin++){
	  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	    Gc_AN = M2G[Mc_AN];    
	    Cwan = WhatSpecies[Gc_AN];
	    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	      Gh_AN = natn[Gc_AN][h_AN];
	      Hwan = WhatSpecies[Gh_AN];
	      for (i=0; i<Spe_Total_NO[Cwan]; i++){
		for (j=0; j<Spe_Total_NO[Hwan]; j++){
		  ResidualH1[m][spin][Mc_AN][h_AN][i][j] = ResidualH1[m-1][spin][Mc_AN][h_AN][i][j];
		}
	      }
	    }
	  }
	}
      }

      /* calculate the current residual Hamiltonian */

      for (spin=0; spin<=SpinP_switch; spin++){
	for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	  Gc_AN = M2G[Mc_AN];    
	  Cwan = WhatSpecies[Gc_AN];
	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	    Gh_AN = natn[Gc_AN][h_AN];
	    Hwan = WhatSpecies[Gh_AN];
	    for (i=0; i<Spe_Total_NO[Cwan]; i++){
	      for (j=0; j<Spe_Total_NO[Hwan]; j++){
		ResidualH1[0][spin][Mc_AN][h_AN][i][j] = H[spin][Mc_AN][h_AN][i][j] - HisH1[0][spin][Mc_AN][h_AN][i][j];
	      }
	    }
	  }
	}
      }

    } /* else */

  } /* if (Mixing_switch!=1 && Mixing_switch!=6) */

  /****************************************************
              calculation of NormRH
  ****************************************************/

  My_Norm = 0.0;

  if (SpinP_switch==3){

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];    
      Cwan = WhatSpecies[Gc_AN];
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	Gh_AN = natn[Gc_AN][h_AN];
	Hwan = WhatSpecies[Gh_AN];
	for (i=0; i<Spe_Total_NO[Cwan]; i++){
	  for (j=0; j<Spe_Total_NO[Hwan]; j++){

	    d0 = HisH1[0][0][Mc_AN][h_AN][i][j] - H[0][Mc_AN][h_AN][i][j];  
	    d1 = HisH1[0][1][Mc_AN][h_AN][i][j] - H[1][Mc_AN][h_AN][i][j];  
	    d2 = HisH1[0][2][Mc_AN][h_AN][i][j] - H[2][Mc_AN][h_AN][i][j];  
	    d3 = HisH1[0][3][Mc_AN][h_AN][i][j] - H[3][Mc_AN][h_AN][i][j];  
            My_Norm += d0*d0 + d1*d1 + d2*d2 + d3*d3;

            d0 = HisH2[0][0][Mc_AN][h_AN][i][j] - iHNL[0][Mc_AN][h_AN][i][j];
            d1 = HisH2[0][1][Mc_AN][h_AN][i][j] - iHNL[1][Mc_AN][h_AN][i][j];
            d2 = HisH2[0][2][Mc_AN][h_AN][i][j] - iHNL[2][Mc_AN][h_AN][i][j];
            My_Norm += d0*d0 + d1*d1 + d2*d2;

	  }
	}
      }
    }

  }

  else{

    for (spin=0; spin<=SpinP_switch; spin++){
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	Gc_AN = M2G[Mc_AN];    
	Cwan = WhatSpecies[Gc_AN];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  for (i=0; i<Spe_Total_NO[Cwan]; i++){
	    for (j=0; j<Spe_Total_NO[Hwan]; j++){
  	      d = HisH1[0][spin][Mc_AN][h_AN][i][j] - H[spin][Mc_AN][h_AN][i][j];  
              My_Norm += d*d;
	    }
	  }
	}
      }
    }
  }

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
    history->energy[i] = history->energy[i-1];
  }
  NormRD[0] = Norm/(double)atomnum;;
  history->energy[0] = Uele;

  if (1<SCF_iter){

    if ( (int)sgn(history->energy[0]-history->energy[1])
	 ==(int)sgn(history->energy[1]-history->energy[2])
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
	history->consecutive++;
      }
    }
   
    else if ( (int)sgn(history->energy[0]-history->energy[1])
	      ==(int)sgn(history->energy[1]-history->energy[2])
	      && NormRD[1]<NormRD[0]){

      tmp0 = NormRD[1]/(largest(NormRD[1]+NormRD[0],10e-10))*Mixing_weight;

      /* tmp0 = Mixing_weight/1.6; */

      if (tmp0<Max_Weight){
	if (Min_Weight<tmp0)
	  Mixing_weight = tmp0;
	else 
	  Mixing_weight = Min_Weight;
      }
      else{
	Mixing_weight = Max_Weight;
      }

      history->consecutive = -1;
    }

    else if ( (int)sgn(history->energy[0]-history->energy[1])
	      !=(int)sgn(history->energy[1]-history->energy[2])
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
	history->consecutive++;
      }
    }

    else if ( (int)sgn(history->energy[0]-history->energy[1])
	      !=(int)sgn(history->energy[1]-history->energy[2])
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

      history->consecutive = -1;
    }
  }

  Mix_wgt = Mixing_weight;

  /****************************************************
            finding a proper mixing weight
  ****************************************************/

  if (SCF_iter==1){
    w1 = 1.0;
    w2 = 1.0 - w1;
  }
  else{
    w1 = Mix_wgt;
    w2 = 1.0 - w1;
  }

  /****************************************************
       performing the simple mixing for Hamiltonian 
  ****************************************************/

  if (SpinP_switch==3){

    /* shift the current Hamiltonian */

    for (m=dim; 0<m; m--){
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	Gc_AN = M2G[Mc_AN];    
	Cwan = WhatSpecies[Gc_AN];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  for (i=0; i<Spe_Total_NO[Cwan]; i++){
	    for (j=0; j<Spe_Total_NO[Hwan]; j++){

	      HisH1[m][0][Mc_AN][h_AN][i][j] = HisH1[m-1][0][Mc_AN][h_AN][i][j];
	      HisH1[m][1][Mc_AN][h_AN][i][j] = HisH1[m-1][1][Mc_AN][h_AN][i][j];
	      HisH1[m][2][Mc_AN][h_AN][i][j] = HisH1[m-1][2][Mc_AN][h_AN][i][j];
	      HisH1[m][3][Mc_AN][h_AN][i][j] = HisH1[m-1][3][Mc_AN][h_AN][i][j];

	      HisH2[m][0][Mc_AN][h_AN][i][j] = HisH2[m-1][0][Mc_AN][h_AN][i][j];
	      HisH2[m][1][Mc_AN][h_AN][i][j] = HisH2[m-1][1][Mc_AN][h_AN][i][j];
	      HisH2[m][2][Mc_AN][h_AN][i][j] = HisH2[m-1][2][Mc_AN][h_AN][i][j];
	    }
	  }
	}
      }
    }

    /* mix the current Hamiltonian and the last one */

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];    
      Cwan = WhatSpecies[Gc_AN];
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	Gh_AN = natn[Gc_AN][h_AN];
	Hwan = WhatSpecies[Gh_AN];
	for (i=0; i<Spe_Total_NO[Cwan]; i++){
	  for (j=0; j<Spe_Total_NO[Hwan]; j++){

	    H[0][Mc_AN][h_AN][i][j] = w2*HisH1[1][0][Mc_AN][h_AN][i][j] + w1*H[0][Mc_AN][h_AN][i][j];  
	    H[1][Mc_AN][h_AN][i][j] = w2*HisH1[1][1][Mc_AN][h_AN][i][j] + w1*H[1][Mc_AN][h_AN][i][j];  
	    H[2][Mc_AN][h_AN][i][j] = w2*HisH1[1][2][Mc_AN][h_AN][i][j] + w1*H[2][Mc_AN][h_AN][i][j];  
	    H[3][Mc_AN][h_AN][i][j] = w2*HisH1[1][3][Mc_AN][h_AN][i][j] + w1*H[3][Mc_AN][h_AN][i][j];  

	    HisH1[0][0][Mc_AN][h_AN][i][j] = H[0][Mc_AN][h_AN][i][j];
	    HisH1[0][1][Mc_AN][h_AN][i][j] = H[1][Mc_AN][h_AN][i][j];
	    HisH1[0][2][Mc_AN][h_AN][i][j] = H[2][Mc_AN][h_AN][i][j];
	    HisH1[0][3][Mc_AN][h_AN][i][j] = H[3][Mc_AN][h_AN][i][j];

	    iHNL[0][Mc_AN][h_AN][i][j] = w2*HisH2[1][0][Mc_AN][h_AN][i][j] + w1*iHNL[0][Mc_AN][h_AN][i][j];
	    iHNL[1][Mc_AN][h_AN][i][j] = w2*HisH2[1][1][Mc_AN][h_AN][i][j] + w1*iHNL[1][Mc_AN][h_AN][i][j];
	    iHNL[2][Mc_AN][h_AN][i][j] = w2*HisH2[1][2][Mc_AN][h_AN][i][j] + w1*iHNL[2][Mc_AN][h_AN][i][j];

	    HisH2[0][0][Mc_AN][h_AN][i][j] = iHNL[0][Mc_AN][h_AN][i][j];
	    HisH2[0][1][Mc_AN][h_AN][i][j] = iHNL[1][Mc_AN][h_AN][i][j];
	    HisH2[0][2][Mc_AN][h_AN][i][j] = iHNL[2][Mc_AN][h_AN][i][j];
	  }
	}
      }
    }

  }

  else {

    /* shift the current Hamiltonian */

    for (m=dim; 0<m; m--){
      for (spin=0; spin<=SpinP_switch; spin++){
	for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	  Gc_AN = M2G[Mc_AN];    
	  Cwan = WhatSpecies[Gc_AN];
	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	    Gh_AN = natn[Gc_AN][h_AN];
	    Hwan = WhatSpecies[Gh_AN];
	    for (i=0; i<Spe_Total_NO[Cwan]; i++){
	      for (j=0; j<Spe_Total_NO[Hwan]; j++){
		HisH1[m][spin][Mc_AN][h_AN][i][j] = HisH1[m-1][spin][Mc_AN][h_AN][i][j];
	      }
	    }
	  }
	}
      }
    }

    /* mix the current Hamiltonian and the last one */

    for (spin=0; spin<=SpinP_switch; spin++){
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	Gc_AN = M2G[Mc_AN];    
	Cwan = WhatSpecies[Gc_AN];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  for (i=0; i<Spe_Total_NO[Cwan]; i++){
	    for (j=0; j<Spe_Total_NO[Hwan]; j++){
	      H[spin][Mc_AN][h_AN][i][j] = w2*HisH1[1][spin][Mc_AN][h_AN][i][j] + w1*H[spin][Mc_AN][h_AN][i][j];
	      HisH1[0][spin][Mc_AN][h_AN][i][j] = H[spin][Mc_AN][h_AN][i][j];
	    }
	  }
	}
      }
    }

  } /* else */

  /* In case of RMM-DIIS */

  if (Mixing_switch==1 || Mixing_switch==6){

    for (spin=0; spin<=SpinP_switch; spin++){

      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	Gc_AN = M2G[Mc_AN];
	ian = Spe_Total_CNO[WhatSpecies[Gc_AN]];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];      
	  jan = Spe_Total_CNO[WhatSpecies[Gh_AN]];

	  for (m=0; m<ian; m++){
	    for (n=0; n<jan; n++){
  
              ResidualDM[2][spin][Mc_AN][h_AN][m][n] = DM[0][spin][Mc_AN][h_AN][m][n]
                                                      -DM[1][spin][Mc_AN][h_AN][m][n];

	      DM[2][spin][Mc_AN][h_AN][m][n] = DM[1][spin][Mc_AN][h_AN][m][n];
	      DM[1][spin][Mc_AN][h_AN][m][n] = DM[0][spin][Mc_AN][h_AN][m][n];
	    }
	  }

	  if ( (SpinP_switch==3 && ( SO_switch==1 || Hub_U_switch==1 || 1<=Constraint_NCS_switch
             || Zeeman_NCS_switch==1 || Zeeman_NCO_switch==1 )) && spin<=1 ){ 

	    for (m=0; m<ian; m++){
	      for (n=0; n<jan; n++){

	        iResidualDM[2][spin][Mc_AN][h_AN][m][n] = iDM[0][spin][Mc_AN][h_AN][m][n]
                                                         -iDM[1][spin][Mc_AN][h_AN][m][n];

		iDM[2][spin][Mc_AN][h_AN][m][n] = iDM[1][spin][Mc_AN][h_AN][m][n];
		iDM[1][spin][Mc_AN][h_AN][m][n] = iDM[0][spin][Mc_AN][h_AN][m][n];
	      }
	    }
	  }

	}
      }

    } /* spin */

  } /* if (Mixing_switch==1 || Mixing_switch==6) */

}





void Inverse(OpenMXSolverInvocation *scratch, int n, const OpenMXSolverRealTensor *a, const OpenMXSolverRealTensor *ia)
{
  int method_flag=2;

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

     x[List_YOUSO[39]]
     y[List_YOUSO[39]]
     da[List_YOUSO[39]][List_YOUSO[39]]
  ***************************************************/

  x = OpenMX_Solver_OverwriteReal(scratch,"mixh-Inverse-x",List_YOUSO[39]);
  y = OpenMX_Solver_OverwriteReal(scratch,"mixh-Inverse-y",List_YOUSO[39]);
  for (i=0; i<List_YOUSO[39]; i++){
    x[i] = 0.0;
    y[i] = 0.0;
  }

  da = OpenMX_Solver_OverwriteView(scratch,"mixh-Inverse-da",List_YOUSO[39],List_YOUSO[39]);
  for (i=0; i<List_YOUSO[39]; i++){
    for (j=0; j<List_YOUSO[39]; j++){
      openmx_solver_real_matrix_row(&da,i)[j] = 0.0;
    }
  }

  /* start calc. */

  if (n==-1){
    for (i=0; i<List_YOUSO[39]; i++){
      for (j=0; j<List_YOUSO[39]; j++){
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

     x[List_YOUSO[39]]
     y[List_YOUSO[39]]
     da[List_YOUSO[39]][List_YOUSO[39]]
  ***************************************************/


  }
  
  else if (method_flag==1){

    int i,j,M,N,LDA,INFO;
    int *IPIV,LWORK;
    double *A,*WORK;

    A = OpenMX_Solver_OverwriteReal(scratch,"mixh-Inverse-lu-A",(n+2)*(n+2));
    WORK = OpenMX_Solver_OverwriteReal(scratch,"mixh-Inverse-lu-WORK",(n+2));
    IPIV = OpenMX_Solver_Int(scratch,"mixh-Inverse-lu-IPIV",(n+2));

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

    A = OpenMX_Solver_OverwriteReal(scratch,"mixh-Inverse-eigen-A",(N+2)*(N+2));
    B = OpenMX_Solver_OverwriteReal(scratch,"mixh-Inverse-eigen-B",(N+2)*(N+2));
    ko = OpenMX_Solver_OverwriteReal(scratch,"mixh-Inverse-eigen-ko",(N+2));

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


void Inverse0(OpenMXSolverInvocation *scratch, int n, const OpenMXSolverRealTensor *a, const OpenMXSolverRealTensor *ia)
{
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

    x[List_YOUSO[39]]
    y[List_YOUSO[39]]
    da[List_YOUSO[39]][List_YOUSO[39]]
  ***************************************************/

  x = OpenMX_Solver_OverwriteReal(scratch,"mixh-Inverse0-x",List_YOUSO[39]);
  y = OpenMX_Solver_OverwriteReal(scratch,"mixh-Inverse0-y",List_YOUSO[39]);

  da = OpenMX_Solver_OverwriteView(scratch,"mixh-Inverse0-da",List_YOUSO[39],List_YOUSO[39]);


  /* start calc. */

  if (n==-1){
    for (i=0; i<List_YOUSO[39]; i++){
      for (j=0; j<List_YOUSO[39]; j++){
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

     x[List_YOUSO[39]]
     y[List_YOUSO[39]]
     da[List_YOUSO[39]][List_YOUSO[39]]
  ***************************************************/


}

static void MixH_Reset(OpenMXNativeResources *owner, void *metadata)
{
  MixingHState *state=(MixingHState*)metadata;
  (void)owner;
  if (state->initialized) Mixing_H_Release_GPU(state);
}

OpenMXMixingHState *Mixing_H_Bind_Resources(OpenMXCoreContext *core)
{
  OpenMXNativeResources *owner=NULL;
  void *metadata=NULL;
  char error[256];
  if (!openmx_core_native_resources_state(core, "Mixing_H", sizeof(MixingHState),
      MixH_Reset, &owner, &metadata, error, sizeof(error))) {
    fprintf(stderr,"OpenMX Hamiltonian mixing state: %s\n",error);
    abort();
  }
  MixingHState *state=(MixingHState*)metadata;
  if (!state->initialized) {
    state->resources=owner;
    if(!openmx_core_mixing_history(core,&state->history,error,sizeof(error))) {
      fprintf(stderr,"OpenMX mixing history: %s\n",error);
      abort();
    }
    const MixH_GpuState initial={
  -1, 0, 0, -1, -1, -1, -1, 0U, -1, 0, 0U, 0U,
  NULL, 0U, 0U, 0U, 0U, 0U, NULL, NULL, NULL, NULL
};
    state->gpu=initial;
    state->inc.last_scf_iter=-1;
    state->initialized=1;
  }
  return state;
}

/**********************************************************************
  DIIS_Mixing_DM.c:

     DIIS_Mixing_DM.c is a subroutine to achieve self-consistent field
     using the direct inversion in the iterative subspace.

  Log of DIIS_Mixing_DM.c:

     22/Nov/2001  Released by T.Ozaki

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "openmx_common.h"
#include "openmx_mixing_dm_workspace.h"
#include "openmx_env.h"
#include "mpi.h"
#include <omp.h>


static void Inverse(OpenMXSolverInvocation *scratch, int n, const OpenMXSolverRealTensor *a, const OpenMXSolverRealTensor *ia);

/*******************************************************************************
  Fast path (active under scf.eigen.lib gpusolver, disabled with
  OPENMX_MIX_FAST=0):

  - The residual Gram matrix is accumulated for ALL pairs in ONE sweep over
    the sparse structure and reduced with ONE MPI_Allreduce, instead of one
    full sweep and one Allreduce per pair.  With one thread the per-pair
    accumulation order is identical to the legacy loops.
  - The optimized-residual and mixing sweeps are fused into one pass.
  - The ResidualDM/iResidualDM history shift becomes a rotation of the
    top-level slot pointers (slot 0 stays in place; the recycled buffer
    lands in slot 1, which is always rewritten before it is read).  The
    DM/iDM history shifts keep the legacy content copies (other modules
    hold device-side state tied to the DM buffers) but get OpenMP.
*******************************************************************************/

static int MixDM_Fast_Enabled(void)
{
  if (scf_eigen_lib_flag != GPUSOLVER) return 0;
  return openmx_env_flag_at(OPENMX_ENV_MIX_FAST, 1);
}

/* rotate the slot pointers 1..nslot-1; slot 0 is left untouched */
static void MixDM_RotateSlots1(double ******arr, int nslot)
{
  double *****tmp = arr[nslot-1];
  int m;
  for (m=nslot-1; 2<=m; m--) arr[m] = arr[m-1];
  arr[1] = tmp;
}




static void DIIS_Mixing_DM_Compute(OpenMXSolverInvocation *scratch,const MixDM_ScratchArguments *arguments)
{
  int SCF_iter=arguments->iteration;
  double ******ResidualDM=arguments->residual;
  const OpenMXElectronicViews *electronic=arguments->electronic;
  double ******iResidualDM=arguments->imaginary_residual;
  OpenMXMixingDMState *state=arguments->state;
  double ******DM=electronic->DM;
  double ******iDM=electronic->iDM;

  int spin,Mc_AN,Gc_AN,wan1,TNO1,h_AN,Gh_AN;
  int wan2,TNO2,i,j,k,NumMix,NumSlide;
  int SCFi,SCFj,tno1,tno0,Cwan,Hwan; 
  int pSCF_iter,size_OptRDM,size_iOptRDM;
  int SpinP_num,flag_nan;
  double *alden;
  OpenMXSolverRealTensor A;
  double My_A;
  OpenMXSolverRealTensor IA;
  double Av_dia,IAv_dia,tmp0;
  double OptNorm_RDM,My_OptNorm_RDM;
  double dum1,dum2,bunsi,bunbo,sum,coef_OptRDM;
  OpenMXElectronicMatrixView OptRDM;
  OpenMXElectronicMatrixView iOptRDM;
  int numprocs,myid;
  char nanchar[300];

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  SpinP_num = SpinP_switch;

  /****************************************************
    allocation of arrays:

    double alden[List_YOUSO[16]];
    double A[List_YOUSO[16]][List_YOUSO[16]];
    double IA[List_YOUSO[16]][List_YOUSO[16]];
    double OptRDM[SpinP_num+1]
                        [Matomnum+1]
                        [FNAN[Gc_AN]+1]
                        [Spe_Total_NO[Cwan]]
                        [Spe_Total_NO[Hwan]];
  ****************************************************/

  alden = OpenMX_Solver_OverwriteReal(scratch,"DIIS_Mixing_DM-alden",List_YOUSO[16]);
  A = OpenMX_Solver_OverwriteView(scratch,"DIIS_Mixing_DM-A",List_YOUSO[16],List_YOUSO[16]);
  IA = OpenMX_Solver_OverwriteView(scratch,"DIIS_Mixing_DM-IA",List_YOUSO[16],List_YOUSO[16]);
  FNAN[0] = 0;
  OptRDM = OpenMX_Solver_MatrixScratch(scratch,&electronic->fields[OPENMX_ELECTRONIC_DENSITY],0);
  size_OptRDM = (int)OptRDM.elements;
  if (SpinP_switch==3 && ( SO_switch==1
                        || Hub_U_switch==1
                        || 1<=Constraint_NCS_switch
                        || Zeeman_NCS_switch==1
                        || Zeeman_NCO_switch==1 )){
    iOptRDM = OpenMX_Solver_MatrixScratch(scratch,&electronic->fields[OPENMX_ELECTRONIC_IMAGINARY_DENSITY],0);
    size_iOptRDM = (int)iOptRDM.elements;
  }

  /****************************************************
     start calc.
  ****************************************************/
 
  if (SCF_iter==1){
    /* memory calc. done, only 1st iteration */
    if (state->diis_first) {
      PrintMemory("DIIS_Mixing_DM: OptRDM",sizeof(double)*size_OptRDM,NULL);
      if (SpinP_switch==3 && ( SO_switch==1 
                            || Hub_U_switch==1
                            || 1<=Constraint_NCS_switch
                            || Zeeman_NCS_switch==1
                            || Zeeman_NCO_switch==1 )){ 
        PrintMemory("DIIS_Mixing_DM: iOptRDM",sizeof(double)*size_iOptRDM,NULL);
      }
      state->diis_first=0;
    }

    Simple_Mixing_DM(state->history,0,1.00,DM[0],DM[1],DM[2],iDM[0],iDM[1],iDM[2],ResidualDM[2],iResidualDM[2]);
  } 

  else if (SCF_iter==2){

    for (spin=0; spin<=SpinP_num; spin++){
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
        Gc_AN = M2G[Mc_AN];
	wan1 = WhatSpecies[Gc_AN];
	TNO1 = Spe_Total_CNO[wan1];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  wan2 = WhatSpecies[Gh_AN];
	  TNO2 = Spe_Total_CNO[wan2];

	  for (i=0; i<TNO1; i++){
	    for (j=0; j<TNO2; j++){
	      ResidualDM[2][spin][Mc_AN][h_AN][i][j] =
 		      DM[0][spin][Mc_AN][h_AN][i][j]
                     -DM[1][spin][Mc_AN][h_AN][i][j];
	    }
	  }

          if (SpinP_switch==3
              &&
             ( SO_switch==1 || Hub_U_switch==1 || 1<=Constraint_NCS_switch || Zeeman_NCS_switch==1 || Zeeman_NCO_switch==1 )
              && spin<2){ 

	    for (i=0; i<TNO1; i++){
	      for (j=0; j<TNO2; j++){
	        iResidualDM[2][spin][Mc_AN][h_AN][i][j] =
 		        iDM[0][spin][Mc_AN][h_AN][i][j]
                       -iDM[1][spin][Mc_AN][h_AN][i][j];
	      }
	    }
	  }        

	}
      }
    }

    for (spin=0; spin<=SpinP_num; spin++){
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
        Gc_AN = M2G[Mc_AN];
	wan1 = WhatSpecies[Gc_AN];
	TNO1 = Spe_Total_CNO[wan1];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  wan2 = WhatSpecies[Gh_AN];
	  TNO2 = Spe_Total_CNO[wan2];

	  for (i=0; i<TNO1; i++){
	    for (j=0; j<TNO2; j++){
	      DM[2][spin][Mc_AN][h_AN][i][j] = DM[1][spin][Mc_AN][h_AN][i][j];
	    }
	  }

          if (SpinP_switch==3
            && ( SO_switch==1 || Hub_U_switch==1 || 1<=Constraint_NCS_switch || Zeeman_NCS_switch==1 || Zeeman_NCO_switch==1)
            && spin<2){ 

	    for (i=0; i<TNO1; i++){
	      for (j=0; j<TNO2; j++){
	        iDM[2][spin][Mc_AN][h_AN][i][j] = iDM[1][spin][Mc_AN][h_AN][i][j];
	      }
	    }
	  }
	}
      }
    }

    Simple_Mixing_DM(state->history,0,state->history->weight,DM[0],DM[1],DM[2],
		     iDM[0],iDM[1],iDM[2],ResidualDM[2],iResidualDM[2]);

  } 
  else {

    /****************************************************
                        Calc of RDM1
    ****************************************************/

    for (spin=0; spin<=SpinP_num; spin++){
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
        Gc_AN = M2G[Mc_AN];
	wan1 = WhatSpecies[Gc_AN];
	TNO1 = Spe_Total_CNO[wan1];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  wan2 = WhatSpecies[Gh_AN];
	  TNO2 = Spe_Total_CNO[wan2];

	  for (i=0; i<TNO1; i++){
	    for (j=0; j<TNO2; j++){
	      ResidualDM[1][spin][Mc_AN][h_AN][i][j] =
		      DM[0][spin][Mc_AN][h_AN][i][j] - DM[1][spin][Mc_AN][h_AN][i][j];
	    }
	  }

          if (SpinP_switch==3
             && ( SO_switch==1 || Hub_U_switch==1 || 1<=Constraint_NCS_switch || Zeeman_NCS_switch==1 || Zeeman_NCO_switch==1)
             && spin<2){ 
	
            for (i=0; i<TNO1; i++){
	      for (j=0; j<TNO2; j++){
	        iResidualDM[1][spin][Mc_AN][h_AN][i][j] =
		        iDM[0][spin][Mc_AN][h_AN][i][j] - iDM[1][spin][Mc_AN][h_AN][i][j];
	      }
	    }
	  }
	}
      }
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
                        alpha from RDM
    ****************************************************/

    if (MixDM_Fast_Enabled()){

      /* every Gram pair is accumulated in ONE sweep over the sparse
         structure (per-pair order identical to the legacy loops with one
         thread) and the whole triangle is reduced with ONE MPI_Allreduce */

      int npair = NumMix*(NumMix+1)/2;
      int nth = omp_get_max_threads();
      int with_imag = (SpinP_switch==3 &&
                       ( SO_switch==1 || Hub_U_switch==1 || 1<=Constraint_NCS_switch
                      || Zeeman_NCS_switch==1 || Zeeman_NCO_switch==1 ));
      double *part = OpenMX_Solver_OverwriteReal(scratch,"DIIS_Mixing_DM-part",(size_t)nth*(size_t)npair);
      memset(part,0,((size_t)nth*(size_t)npair)*sizeof(double));
      double *my_tri = OpenMX_Solver_OverwriteReal(scratch,"DIIS_Mixing_DM-my_tri",(size_t)npair);
      double *tri = OpenMX_Solver_OverwriteReal(scratch,"DIIS_Mixing_DM-tri",(size_t)npair);

      if (part==NULL || my_tri==NULL || tri==NULL){
        printf("DIIS_Mixing_DM: failed to allocate the Gram buffers.\n");
        MPI_Abort(mpi_comm_level1,1);
      }

      for (spin=0; spin<=SpinP_num; spin++){

#pragma omp parallel private(Mc_AN,Gc_AN,wan1,TNO1,h_AN,Gh_AN,wan2,TNO2,i,j)
        {
          int tid = omp_get_thread_num();
          double *acc = &part[(size_t)tid*(size_t)npair];
          const double *rows[List_YOUSO[16]];
          double vals[List_YOUSO[16]];

#pragma omp for schedule(static)
          for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
            Gc_AN = M2G[Mc_AN];
            wan1 = WhatSpecies[Gc_AN];
            TNO1 = Spe_Total_CNO[wan1];
            for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
              Gh_AN = natn[Gc_AN][h_AN];
              wan2 = WhatSpecies[Gh_AN];
              TNO2 = Spe_Total_CNO[wan2];

              for (i=0; i<TNO1; i++){
                int m,a,b,p;
                for (m=1; m<=NumMix; m++) rows[m] = ResidualDM[m][spin][Mc_AN][h_AN][i];
                for (j=0; j<TNO2; j++){
                  for (m=1; m<=NumMix; m++) vals[m] = rows[m][j];
                  p = 0;
                  for (a=1; a<=NumMix; a++){
                    double va = vals[a];
                    for (b=a; b<=NumMix; b++){
                      acc[p] += va*vals[b];
                      p++;
                    }
                  }
                }
              }

              if (with_imag && spin<2){
                for (i=0; i<TNO1; i++){
                  int m,a,b,p;
                  for (m=1; m<=NumMix; m++) rows[m] = iResidualDM[m][spin][Mc_AN][h_AN][i];
                  for (j=0; j<TNO2; j++){
                    for (m=1; m<=NumMix; m++) vals[m] = rows[m][j];
                    p = 0;
                    for (a=1; a<=NumMix; a++){
                      double va = vals[a];
                      for (b=a; b<=NumMix; b++){
                        acc[p] += va*vals[b];
                        p++;
                      }
                    }
                  }
                }
              }

            }
          }
        }
      }

      {
        int p,t;
        for (p=0; p<npair; p++){
          double s = 0.0;
          for (t=0; t<nth; t++) s += part[(size_t)t*(size_t)npair+p];
          my_tri[p] = s;
        }
      }

      MPI_Allreduce(my_tri, tri, npair, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

      {
        int p = 0;
        for (SCFi=1; SCFi<=NumMix; SCFi++){
          for (SCFj=SCFi; SCFj<=NumMix; SCFj++){
            openmx_solver_real_matrix_row(&A,SCFi-1)[SCFj-1] = tri[p];
            openmx_solver_real_matrix_row(&A,SCFj-1)[SCFi-1] = tri[p];
            p++;
          }
        }
      }



    }

    else{

    for (SCFi=1; SCFi<=NumMix; SCFi++){
      for (SCFj=SCFi; SCFj<=NumMix; SCFj++){

        My_A = 0.0;

#pragma omp parallel for reduction(+:My_A) private(spin, Mc_AN, Gc_AN, wan1, TNO1, h_AN, Gh_AN, wan2, TNO2, i, j, dum1, dum2) if(SpinP_num > 1) schedule(static,1)
        for (spin=0; spin<=SpinP_num; spin++){
          for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
            Gc_AN = M2G[Mc_AN];
	    wan1 = WhatSpecies[Gc_AN];
	    TNO1 = Spe_Total_CNO[wan1];
            for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	      Gh_AN = natn[Gc_AN][h_AN];
	      wan2 = WhatSpecies[Gh_AN];
	      TNO2 = Spe_Total_CNO[wan2];

              for (i=0; i<TNO1; i++){
                for (j=0; j<TNO2; j++){
                  dum1 = ResidualDM[SCFi][spin][Mc_AN][h_AN][i][j];
                  dum2 = ResidualDM[SCFj][spin][Mc_AN][h_AN][i][j];
                  My_A += dum1*dum2;
                }
              }

              if (SpinP_switch==3
                  &&
                 ( SO_switch==1 || Hub_U_switch==1 || 1<=Constraint_NCS_switch || Zeeman_NCS_switch==1 || Zeeman_NCO_switch==1)
                  && spin<2){

                for (i=0; i<TNO1; i++){
                  for (j=0; j<TNO2; j++){
                    dum1 = iResidualDM[SCFi][spin][Mc_AN][h_AN][i][j];
                    dum2 = iResidualDM[SCFj][spin][Mc_AN][h_AN][i][j];
                    My_A += dum1*dum2;
                  }
                }
	      }

            }
          }
	}

        /* MPI My_A */

        MPI_Allreduce(&My_A, &openmx_solver_real_matrix_row(&A,SCFi-1)[SCFj-1], 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

        openmx_solver_real_matrix_row(&A,SCFj-1)[SCFi-1] = openmx_solver_real_matrix_row(&A,SCFi-1)[SCFj-1];
      }
    }

    } /* else (legacy Gram) */

    Av_dia = openmx_solver_real_matrix_row(&A,0)[0];
    state->history->norm[0] = openmx_solver_real_matrix_row(&A,0)[0];

    for (SCFi=1; SCFi<=NumMix; SCFi++){
      openmx_solver_real_matrix_row(&A,SCFi-1)[NumMix] = -1.0;
      openmx_solver_real_matrix_row(&A,NumMix)[SCFi-1] = -1.0;
    }
    openmx_solver_real_matrix_row(&A,NumMix)[NumMix] = 0.0;

#pragma forceinline
    Inverse(scratch,NumMix,&A,&IA);
    for (SCFi=1; SCFi<=NumMix; SCFi++){
      alden[SCFi] = -openmx_solver_real_matrix_row(&IA,SCFi-1)[NumMix];

      /*
      printf("AP Alpha_%i=%15.12f\n",SCFi,alden[SCFi]);
      */
    }

    /*
    printf("Lambda=%15.12f\n",-IA[NumMix][NumMix]);
    */

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
              Calculate optimized residual DM
    ****************************************************/

    if (MixDM_Fast_Enabled()){

      /* fused optimized-residual + mixing sweep: per element
         sum_r = sum_m alden[m]*ResidualDM[m], sum_d = sum_m alden[m]*DM[m],
         DM[0] = sum_d + coef*sum_r; identical per-element sums and (with
         one thread) identical norm accumulation order as the legacy code */

      int with_imag = (SpinP_switch==3 &&
                       ( SO_switch==1 || Hub_U_switch==1 || 1<=Constraint_NCS_switch
                      || Zeeman_NCS_switch==1 || Zeeman_NCO_switch==1 ));
      int nth = omp_get_max_threads();
      double *norm_part = OpenMX_Solver_OverwriteReal(scratch,"DIIS_Mixing_DM-norm_part",(size_t)nth);
      memset(norm_part,0,((size_t)nth)*sizeof(double));

      if (norm_part==NULL){
        printf("DIIS_Mixing_DM: failed to allocate the norm buffer.\n");
        MPI_Abort(mpi_comm_level1,1);
      }

      if (1.0e-1<=state->history->norm[0])
        coef_OptRDM = 0.5;
      else if (1.0e-2<=state->history->norm[0] && state->history->norm[0]<1.0e-1)
        coef_OptRDM = 0.6;
      else if (1.0e-3<=state->history->norm[0] && state->history->norm[0]<1.0e-2)
        coef_OptRDM = 0.7;
      else if (1.0e-4<=state->history->norm[0] && state->history->norm[0]<1.0e-3)
        coef_OptRDM = 0.8;
      else
        coef_OptRDM = 1.0;

      for (spin=0; spin<=SpinP_num; spin++){

#pragma omp parallel private(Mc_AN,Gc_AN,wan1,TNO1,h_AN,Gh_AN,wan2,TNO2,i,j)
        {
          int tid = omp_get_thread_num();
          double my_norm = 0.0;
          const double *rrows[List_YOUSO[16]];
          const double *drows[List_YOUSO[16]];

#pragma omp for schedule(static)
          for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
            Gc_AN = M2G[Mc_AN];
            wan1 = WhatSpecies[Gc_AN];
            TNO1 = Spe_Total_CNO[wan1];
            for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
              Gh_AN = natn[Gc_AN][h_AN];
              wan2 = WhatSpecies[Gh_AN];
              TNO2 = Spe_Total_CNO[wan2];

              for (i=0; i<TNO1; i++){
                int m;
                for (m=1; m<=NumMix; m++){
                  rrows[m] = ResidualDM[m][spin][Mc_AN][h_AN][i];
                  drows[m] = DM[m][spin][Mc_AN][h_AN][i];
                }
                for (j=0; j<TNO2; j++){
                  double sum_r = 0.0;
                  double sum_d = 0.0;
                  for (m=1; m<=NumMix; m++){
                    sum_r += alden[m]*rrows[m][j];
                    sum_d += alden[m]*drows[m][j];
                  }
                  my_norm += sum_r*sum_r;
                  DM[0][spin][Mc_AN][h_AN][i][j] = sum_d + coef_OptRDM*sum_r;
                }
              }

              if (with_imag && spin<2){
                for (i=0; i<TNO1; i++){
                  int m;
                  for (m=1; m<=NumMix; m++){
                    rrows[m] = iResidualDM[m][spin][Mc_AN][h_AN][i];
                    drows[m] = iDM[m][spin][Mc_AN][h_AN][i];
                  }
                  for (j=0; j<TNO2; j++){
                    double sum_r = 0.0;
                    double sum_d = 0.0;
                    for (m=1; m<=NumMix; m++){
                      sum_r += alden[m]*rrows[m][j];
                      sum_d += alden[m]*drows[m][j];
                    }
                    my_norm += sum_r*sum_r;
                    iDM[0][spin][Mc_AN][h_AN][i][j] = sum_d + coef_OptRDM*sum_r;
                  }
                }
              }

            }
          }

          norm_part[tid] += my_norm;
        }
      }

      {
        int t;
        My_OptNorm_RDM = 0.0;
        for (t=0; t<nth; t++) My_OptNorm_RDM += norm_part[t];
      }


      /* MPI My_OptNorm_RDM */
      MPI_Allreduce(&My_OptNorm_RDM, &OptNorm_RDM, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
    }

    else{

    My_OptNorm_RDM = 0.0;

    for (spin=0; spin<=SpinP_num; spin++){
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
        Gc_AN = M2G[Mc_AN];
	wan1 = WhatSpecies[Gc_AN];
	TNO1 = Spe_Total_CNO[wan1];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  wan2 = WhatSpecies[Gh_AN];
	  TNO2 = Spe_Total_CNO[wan2];

	  for (i=0; i<TNO1; i++){
	    for (j=0; j<TNO2; j++){

              /* Pulay mixing */

              sum = 0.0;
	      for (pSCF_iter=1; pSCF_iter<=NumMix; pSCF_iter++){
		sum += alden[pSCF_iter]*ResidualDM[pSCF_iter][spin][Mc_AN][h_AN][i][j];
	      }

              openmx_electronic_row(&OptRDM,spin,Mc_AN,h_AN,i)[j] = sum;
              My_OptNorm_RDM += sum*sum;

	    }
	  }

          if (SpinP_switch==3
             &&
            ( SO_switch==1 || Hub_U_switch==1 || 1<=Constraint_NCS_switch || Zeeman_NCS_switch==1 || Zeeman_NCO_switch==1)
             &&
             spin<2){
 
	    for (i=0; i<TNO1; i++){
	      for (j=0; j<TNO2; j++){

		/* Pulay mixing */

		sum = 0.0;
		for (pSCF_iter=1; pSCF_iter<=NumMix; pSCF_iter++){
		  sum += alden[pSCF_iter]*iResidualDM[pSCF_iter][spin][Mc_AN][h_AN][i][j];
		}

                openmx_electronic_row(&iOptRDM,spin,Mc_AN,h_AN,i)[j] = sum;
		My_OptNorm_RDM += sum*sum;

	      }
	    }
	  }

	}
      }
    }

    /* MPI My_OptNorm_RDM */
    MPI_Allreduce(&My_OptNorm_RDM, &OptNorm_RDM, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

    /*
    printf("OptNorm_RDM=%18.12f\n",OptNorm_RDM);
    */

    if (1.0e-1<=state->history->norm[0])
      coef_OptRDM = 0.5;
    else if (1.0e-2<=state->history->norm[0] && state->history->norm[0]<1.0e-1)
      coef_OptRDM = 0.6;
    else if (1.0e-3<=state->history->norm[0] && state->history->norm[0]<1.0e-2)
      coef_OptRDM = 0.7;
    else if (1.0e-4<=state->history->norm[0] && state->history->norm[0]<1.0e-3)
      coef_OptRDM = 0.8;
    else
      coef_OptRDM = 1.0;

    /****************************************************
                        Mixing of DM 
    ****************************************************/

    for (spin=0; spin<=SpinP_num; spin++){
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
        Gc_AN = M2G[Mc_AN];
	wan1 = WhatSpecies[Gc_AN];
	TNO1 = Spe_Total_CNO[wan1];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  wan2 = WhatSpecies[Gh_AN];
	  TNO2 = Spe_Total_CNO[wan2];

	  for (i=0; i<TNO1; i++){
	    for (j=0; j<TNO2; j++){

              /* Pulay mixing */

              sum = 0.0;
	      for (pSCF_iter=1; pSCF_iter<=NumMix; pSCF_iter++){
		sum += alden[pSCF_iter]*DM[pSCF_iter][spin][Mc_AN][h_AN][i][j];
	      }

              /* Correction by optimized RDM */

              DM[0][spin][Mc_AN][h_AN][i][j] = sum + coef_OptRDM*openmx_electronic_row(&OptRDM,spin,Mc_AN,h_AN,i)[j];
	    }
	  }

          if (SpinP_switch==3
            &&
           ( SO_switch==1 || Hub_U_switch==1 || 1<=Constraint_NCS_switch || Zeeman_NCS_switch==1 || Zeeman_NCO_switch==1)
            && spin<2){ 

	    for (i=0; i<TNO1; i++){
	      for (j=0; j<TNO2; j++){

		/* Pulay mixing */

		sum = 0.0;
		for (pSCF_iter=1; pSCF_iter<=NumMix; pSCF_iter++){
		  sum += alden[pSCF_iter]*iDM[pSCF_iter][spin][Mc_AN][h_AN][i][j];
		}

		/* Correction by optimized RDM */

                iDM[0][spin][Mc_AN][h_AN][i][j] = sum + coef_OptRDM*openmx_electronic_row(&iOptRDM,spin,Mc_AN,h_AN,i)[j];
	      }
	    }
	  }

	}
      }
    }

    } /* else (legacy Opt+Mix) */

    /****************************************************
                         Shift of DM
    ****************************************************/

    for (pSCF_iter=NumSlide; 0<pSCF_iter; pSCF_iter--){
      for (spin=0; spin<=SpinP_num; spin++){
        for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
          Gc_AN = M2G[Mc_AN];
	  wan1 = WhatSpecies[Gc_AN]; 
	  TNO1 = Spe_Total_CNO[wan1];
	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	    Gh_AN = natn[Gc_AN][h_AN];
	    wan2 = WhatSpecies[Gh_AN];          
	    TNO2 = Spe_Total_CNO[wan2]; 

	    for (i=0; i<TNO1; i++){
	      for (j=0; j<TNO2; j++){
		DM[pSCF_iter][spin][Mc_AN][h_AN][i][j]
                =
                DM[pSCF_iter-1][spin][Mc_AN][h_AN][i][j];
	      }
	    }

            if (SpinP_switch==3
                &&
               (SO_switch==1 || Hub_U_switch==1 || 1<=Constraint_NCS_switch || Zeeman_NCS_switch==1 || Zeeman_NCO_switch==1)
                && spin<2){
 
	      for (i=0; i<TNO1; i++){
		for (j=0; j<TNO2; j++){
		  iDM[pSCF_iter][spin][Mc_AN][h_AN][i][j]
		  =
		  iDM[pSCF_iter-1][spin][Mc_AN][h_AN][i][j];
		}
	      }
	    }

	  }
	}
      }
    }

    /****************************************************
                         Shift of RDM
    ****************************************************/

    if (MixDM_Fast_Enabled()){

      /* pointer rotation of the slots 1..List_YOUSO[16]-1; the recycled
         buffer lands in slot 1, which is rewritten before any read */

      MixDM_RotateSlots1(ResidualDM, List_YOUSO[16]);
      if (SpinP_switch==3 &&
          ( SO_switch==1 || Hub_U_switch==1 || 1<=Constraint_NCS_switch
         || Zeeman_NCS_switch==1 || Zeeman_NCO_switch==1 )){
        MixDM_RotateSlots1(iResidualDM, List_YOUSO[16]);
      }
    }
    else
    for (pSCF_iter=NumSlide; 1<pSCF_iter; pSCF_iter--){
      for (spin=0; spin<=SpinP_num; spin++){
        for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
          Gc_AN = M2G[Mc_AN];
	  wan1 = WhatSpecies[Gc_AN]; 
	  TNO1 = Spe_Total_CNO[wan1];
	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	    Gh_AN = natn[Gc_AN][h_AN];
	    wan2 = WhatSpecies[Gh_AN];          
	    TNO2 = Spe_Total_CNO[wan2]; 

	    for (i=0; i<TNO1; i++){
	      for (j=0; j<TNO2; j++){
		ResidualDM[pSCF_iter][spin][Mc_AN][h_AN][i][j]
                =
                ResidualDM[pSCF_iter-1][spin][Mc_AN][h_AN][i][j];
	      }
	    }

            if (SpinP_switch==3
               &&
               ( SO_switch==1 || Hub_U_switch==1 || 1<=Constraint_NCS_switch || Zeeman_NCS_switch==1 || Zeeman_NCO_switch==1)
               && spin<2){ 
	      for (i=0; i<TNO1; i++){
		for (j=0; j<TNO2; j++){
		  iResidualDM[pSCF_iter][spin][Mc_AN][h_AN][i][j]
		  =
		  iResidualDM[pSCF_iter-1][spin][Mc_AN][h_AN][i][j];
		}
	      }
	    }
	  }
	}
      }
    }
  }

  /* Invocation exit releases scratch without reading atom topology. */
  FNAN[0] = 0;
}





void Inverse(OpenMXSolverInvocation *scratch, int n, const OpenMXSolverRealTensor *a, const OpenMXSolverRealTensor *ia)
{
  OpenMX_Solver_InverseUnpivoted(scratch,n,a,ia);
}


void DIIS_Mixing_DM(OpenMXMixingDMState *state, int SCF_iter, double ******ResidualDM, double ******iResidualDM, const OpenMXElectronicViews *electronic)
{
  MixDM_WithScratch(DIIS_Mixing_DM_Compute,state,SCF_iter,ResidualDM,iResidualDM,electronic);
}

static void MixDM_Reset(OpenMXNativeResources *owner,void *metadata)
{
  (void)owner;
  memset(metadata,0,sizeof(OpenMXMixingDMState));
}

OpenMXMixingDMState *Mixing_DM_Bind_Resources(OpenMXCoreContext *core)
{
  OpenMXNativeResources *owner=NULL;
  void *metadata=NULL;
  char error[256];
  if(!openmx_core_native_resources_state(core,"Mixing_DM",sizeof(OpenMXMixingDMState),
      MixDM_Reset,&owner,&metadata,error,sizeof(error))) {
    fprintf(stderr,"OpenMX real-space density mixing state: %s\n",error);
    abort();
  }
  OpenMXMixingDMState *state=(OpenMXMixingDMState*)metadata;
  if(!state->initialized) {
    if(!openmx_core_mixing_history(core,&state->history,error,sizeof(error))) {
      fprintf(stderr,"OpenMX mixing history: %s\n",error);
      abort();
    }
    state->diis_first=state->gr_first=1;
    state->initialized=1;
  }
  return state;
}

/**********************************************************************
  ADIIS_Mixing_DM.c:

  ADIIS_Mixing_DM.c is a subroutine to achieve self-consistent field
  using Augmented Roothaan-Hall (ARH) energy function + 
  the direct inversion in the iterative subspace (DIIS).

  Log of ADIIS_Mixing_DM.c:

     03/Apr./2014  Released by T.Ozaki

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "openmx_common.h"
#include "openmx_mixing_dm_workspace.h"
#include "mpi.h"


static void Inverse(OpenMXSolverInvocation *scratch, int n, const OpenMXSolverRealTensor *a, const OpenMXSolverRealTensor *ia);




static void ADIIS_Mixing_DM_Compute(OpenMXSolverInvocation *scratch,const MixDM_ScratchArguments *arguments)
{
  int SCF_iter=arguments->iteration;
  double ******ResidualDM=arguments->residual;
  const OpenMXElectronicViews *electronic=arguments->electronic;
  double ******iResidualDM=arguments->imaginary_residual;
  double ******DM=electronic->DM;
  double *****H=electronic->H;
  double ******HisH1=electronic->HisH1;
  double ******iDM=electronic->iDM;

  int spin,Mc_AN,Gc_AN,wan1,TNO1,h_AN,Gh_AN;
  int wan2,TNO2,i,j,k,NumMix,NumSlide;
  int SCFi,SCFj,SCFk,tno1,tno0,Cwan,Hwan; 
  int pSCF_iter,size_OptRDM,size_iOptRDM;
  int SpinP_num,flag_nan,n,po;
  double My_sum,My_A,tt2,F;
  double *alden,*alR,*DiDn_Hn;
  OpenMXSolverRealTensor DiDn_HjHn;
  OpenMXSolverRealTensor A,IA,ct;
  double *c,*t,*dF;
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
  ****************************************************/

  alden = OpenMX_Solver_OverwriteReal(scratch,"ADIIS_Mixing_DM-alden",List_YOUSO[16]);
  alR = OpenMX_Solver_OverwriteReal(scratch,"ADIIS_Mixing_DM-alR",List_YOUSO[16]);
  t = OpenMX_Solver_OverwriteReal(scratch,"ADIIS_Mixing_DM-t",List_YOUSO[16]);
  c = OpenMX_Solver_OverwriteReal(scratch,"ADIIS_Mixing_DM-c",List_YOUSO[16]);
  dF = OpenMX_Solver_OverwriteReal(scratch,"ADIIS_Mixing_DM-dF",List_YOUSO[16]);
  DiDn_Hn = OpenMX_Solver_OverwriteReal(scratch,"ADIIS_Mixing_DM-DiDn_Hn",List_YOUSO[16]);
  A = OpenMX_Solver_OverwriteView(scratch,"ADIIS_Mixing_DM-A",List_YOUSO[16],List_YOUSO[16]);
  IA = OpenMX_Solver_OverwriteView(scratch,"ADIIS_Mixing_DM-IA",List_YOUSO[16],List_YOUSO[16]);
  ct = OpenMX_Solver_OverwriteView(scratch,"ADIIS_Mixing_DM-ct",List_YOUSO[16],List_YOUSO[16]);
  DiDn_HjHn = OpenMX_Solver_OverwriteView(scratch,"ADIIS_Mixing_DM-DiDn_HjHn",List_YOUSO[16],List_YOUSO[16]);
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
   determination of the number of previous informations
   to be mixed
  ****************************************************/

  if ((SCF_iter-1)<Num_Mixing_pDM){
    NumMix   = SCF_iter - 1;
    NumSlide = NumMix + 1;
  }
  else{
    NumMix   = Num_Mixing_pDM;
    NumSlide = NumMix;
  }

  /****************************************************
                     store HisH
  ****************************************************/

  for (SCFi=NumMix; 1<SCFi; SCFi--){
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
              HisH1[SCFi][spin][Mc_AN][h_AN][i][j] = HisH1[SCFi-1][spin][Mc_AN][h_AN][i][j];
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
	    HisH1[1][spin][Mc_AN][h_AN][i][j] = H[spin][Mc_AN][h_AN][i][j];
	  }
	}
      }
    }
  }
    
  /****************************************************
             calculation of <Di-Dn|H(Dn)> 
  ****************************************************/

  for (SCFi=1; SCFi<=NumMix; SCFi++){

    My_sum = 0.0;

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

              /* Di-Dn */
	      dum1 = DM[SCFi][spin][Mc_AN][h_AN][i][j] - DM[1][spin][Mc_AN][h_AN][i][j];

              /* H(Dn) */
	      dum2 = H[spin][Mc_AN][h_AN][i][j];

              /* <Di-Dn|H(Dn)>  */
	      My_sum += dum1*dum2;
	    }
	  }
	}
      }
    }

    /* MPI My_sum */

    MPI_Allreduce(&My_sum, &DiDn_Hn[SCFi], 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
    DiDn_Hn[SCFi] = DiDn_Hn[SCFi];

  }

  /****************************************************
           calculation of <Di-Dn|H(Dj)-H(Dn)> 
  ****************************************************/

  for (SCFi=1; SCFi<=NumMix; SCFi++){
    for (SCFj=1; SCFj<=NumMix; SCFj++){

      My_sum = 0.0;

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

		/* Di-Dn */
		dum1 = DM[SCFi][spin][Mc_AN][h_AN][i][j] - DM[1][spin][Mc_AN][h_AN][i][j];

		/* H(Dj)-H(Dn) */
		dum2 = HisH1[SCFj][spin][Mc_AN][h_AN][i][j] - HisH1[1][spin][Mc_AN][h_AN][i][j];

		/* <Di-Dn|H(Dj)-H(Dn)> */
		My_sum += dum1*dum2;
	      }
	    }
	  }
	}
      }

      /* MPI My_sum */
      MPI_Allreduce(&My_sum, &openmx_solver_real_matrix_row(&DiDn_HjHn,SCFi)[SCFj], 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
      openmx_solver_real_matrix_row(&DiDn_HjHn,SCFi)[SCFj] = openmx_solver_real_matrix_row(&DiDn_HjHn,SCFi)[SCFj];

    }
  }

  /****************************************************
                  optimization of F 
  ****************************************************/

  /* initialization of t and c */

  for (SCFi=1; SCFi<=NumMix; SCFi++){
    t[SCFi] = 1.0/(double)SCFi;
  }

  tt2 = 0.0; 
  for (SCFi=1; SCFi<=NumMix; SCFi++){
    tt2 += t[SCFi]*t[SCFi];
  }

  for (SCFi=1; SCFi<=NumMix; SCFi++){
    c[SCFi] = t[SCFi]*t[SCFi]/tt2;
  }
  
  n = 1;
  po = 0;

  do {

    /* calculaitons of dc/dt */

    for (SCFi=1; SCFi<=NumMix; SCFi++){
      for (SCFj=1; SCFj<=NumMix; SCFj++){

        if (SCFi==SCFj)
          openmx_solver_real_matrix_row(&ct,SCFi)[SCFj] = 2.0*(t[SCFi]*tt2-t[SCFi]*t[SCFi]*t[SCFj])/(tt2*tt2);
        else
          openmx_solver_real_matrix_row(&ct,SCFi)[SCFj] = -2.0*t[SCFi]*t[SCFi]*t[SCFj]/(tt2*tt2);
      }
    }    

    /* calculaitons of dF/dt */

    for (SCFk=1; SCFk<=NumMix; SCFk++){

      dF[SCFk] = 0.0;

      for (SCFi=1; SCFi<=NumMix; SCFi++){
        dF[SCFk] += openmx_solver_real_matrix_row(&ct,SCFi)[SCFk]*DiDn_Hn[SCFi];
      }

      for (SCFi=1; SCFi<=NumMix; SCFi++){
	for (SCFj=1; SCFj<=NumMix; SCFj++){
          dF[SCFk] += 0.5*(openmx_solver_real_matrix_row(&ct,SCFi)[SCFk]*c[SCFj] +c[SCFi]*openmx_solver_real_matrix_row(&ct,SCFj)[SCFk])*openmx_solver_real_matrix_row(&DiDn_HjHn,SCFi)[SCFj];
	}
      }
    }

    /* update t */

    for (SCFi=1; SCFi<=NumMix; SCFi++){
      t[SCFi] = t[SCFi] - 10.0*dF[SCFi];
    }

    tt2 = 0.0; 
    for (SCFi=1; SCFi<=NumMix; SCFi++){
      tt2 += t[SCFi]*t[SCFi];
    }

    for (SCFi=1; SCFi<=NumMix; SCFi++){
      c[SCFi] = t[SCFi]*t[SCFi]/tt2;
    }

    /* calculation of F */

    F = 0.0;

    for (SCFi=1; SCFi<=NumMix; SCFi++){
      F += c[SCFi]*DiDn_Hn[SCFi];
    }

    for (SCFi=1; SCFi<=NumMix; SCFi++){
      for (SCFj=1; SCFj<=NumMix; SCFj++){
        F += 0.5*c[SCFi]*c[SCFj]*openmx_solver_real_matrix_row(&DiDn_HjHn,SCFi)[SCFj];
      }
    }

    n++;

  } while (po==0 && n<300);

  for (SCFi=1; SCFi<=NumMix; SCFi++){
    alden[SCFi] = c[SCFi];
  }

  /****************************************************
         calculation of residual density matrix
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

	/*
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
	*/

      }
    }
  }

  /****************************************************
          calculation of optimum coefficients 
  ****************************************************/

  for (SCFi=1; SCFi<=NumMix; SCFi++){
    for (SCFj=1; SCFj<=NumMix; SCFj++){

      My_A = 0.0;

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

	    /*
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
	    */

	  }
	}
      }

      /* MPI My_A */
      MPI_Allreduce(&My_A, &openmx_solver_real_matrix_row(&A,SCFi-1)[SCFj-1], 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

      if (SCFi==1 && SCFj==1) NormRD[0] = openmx_solver_real_matrix_row(&A,0)[0];

      openmx_solver_real_matrix_row(&A,SCFi-1)[SCFj-1] = openmx_solver_real_matrix_row(&A,SCFi-1)[SCFj-1];

    } /* SCFj */
  } /* SCFi */

  for (SCFi=1; SCFi<=NumMix; SCFi++){
    openmx_solver_real_matrix_row(&A,SCFi-1)[NumMix] = 1.0;
    openmx_solver_real_matrix_row(&A,NumMix)[SCFi-1] = 1.0;
  }
  openmx_solver_real_matrix_row(&A,NumMix)[NumMix] = 0.0;

  Inverse(scratch,NumMix,&A,&IA);

  /* IA*(-<Di-Dn|H(Dn)>) */

  for (SCFi=1; SCFi<=NumMix; SCFi++){
    alR[SCFi] = openmx_solver_real_matrix_row(&IA,SCFi-1)[NumMix];
  }

  /*************************************
      check "nan", "NaN", "inf" or "Inf"
  *************************************/

  flag_nan = 0;
  for (SCFi=1; SCFi<=NumMix; SCFi++){

    sprintf(nanchar,"%8.4f",alR[SCFi]);
    if (strstr(nanchar,"nan")!=NULL || strstr(nanchar,"NaN")!=NULL 
     || strstr(nanchar,"inf")!=NULL || strstr(nanchar,"Inf")!=NULL){

      flag_nan = 1;
    }
  }

  if (flag_nan==1){
    for (SCFi=1; SCFi<=NumMix; SCFi++){
      alR[SCFi] = 0.0;
    }
    alR[1] = 0.1;
    alR[2] = 0.9;
  }

  /****************************************************
           calculation of optimized residual DM 
  ****************************************************/

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
	      sum += alR[pSCF_iter]*ResidualDM[pSCF_iter][spin][Mc_AN][h_AN][i][j];
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
		sum += alR[pSCF_iter]*iResidualDM[pSCF_iter][spin][Mc_AN][h_AN][i][j];
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

  if (1.0e-1<=NormRD[0])
    coef_OptRDM = 0.5;
  else if (1.0e-2<=NormRD[0] && NormRD[0]<1.0e-1)
    coef_OptRDM = 0.6;
  else if (1.0e-3<=NormRD[0] && NormRD[0]<1.0e-2)
    coef_OptRDM = 0.7;
  else if (1.0e-4<=NormRD[0] && NormRD[0]<1.0e-3)
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

  /* Invocation exit releases scratch without reading atom topology. */
  FNAN[0] = 0;
}





void Inverse(OpenMXSolverInvocation *scratch, int n, const OpenMXSolverRealTensor *a, const OpenMXSolverRealTensor *ia)
{
  OpenMX_Solver_InverseUnpivoted(scratch,n,a,ia);
}


void ADIIS_Mixing_DM(int SCF_iter, double ******ResidualDM, double ******iResidualDM, const OpenMXElectronicViews *electronic)
{
  MixDM_WithScratch(ADIIS_Mixing_DM_Compute,NULL,SCF_iter,ResidualDM,iResidualDM,electronic);
}

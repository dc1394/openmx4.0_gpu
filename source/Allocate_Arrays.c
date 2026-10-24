#include <math.h>
#include <limits.h>
#include "openmx_core_pao.h"
#include "openmx_core_vps.h"
#include "openmx_core_product.h"
#include "openmx_core_neutral_grid.h"
#include "openmx_core_neutral_grid_metadata.h"
#include <stdio.h>
#include <stdlib.h>
#include "openmx_common.h"

void Allocate_Arrays(int wherefrom)
{
  Allocate_Arrays_with_context(wherefrom,NULL);
}

void Allocate_Arrays_with_context(int wherefrom, OpenMXCoreContext *core)
{
  int i,j,k,ii,L,ct_AN,wan,p,l,al,so,spin;
  int Lmax,num,m,n,spe,Ndim;

  switch(wherefrom){  

    case 0: 
      /* call from Input_std.c */

      {
        const OpenMXSpeciesShape shape={SpeciesNum,Supported_MaxL+1,YOUSO10};
        OpenMXSpeciesViews views;
        char error[256];
        if (!openmx_core_species_allocate(core,&shape,&views,error,sizeof(error))) {
          fprintf(stderr,"Allocate_Arrays: species: %s\n",error);
          MPI_Abort(mpi_comm_level1,1);
          return;
        }
        SpeName=views.name;
        SpeBasis=views.basis_name;
        SpeBasisName=views.basis_file;
        SpeVPS=views.potential_file;
        Spe_AtomicMass=views.mass;
        Spe_MaxL_Basis=views.basis_lmax;
        Spe_Num_Basis=views.basis_rows;
        Spe_Num_CBasis=views.contracted_basis_rows;
        Spe_OpenCore_flag=views.open_core;
        Number_of_Electrons_Species=views.electrons;
        Spe_Spe2Ban=views.species_index;
        VPS_j_dependency=views.relativistic;
        Spe_Atom_Cut1=views.cutoff;
        Spe_Core_Charge=views.core_charge;
        Spe_Num_Mesh_VPS=views.vps_mesh;
        Spe_Num_Mesh_PAO=views.pao_mesh;
        Spe_Total_VPS_Pro=views.projector_components;
        Spe_Num_RVPS=views.radial_projectors;
        Spe_PAO_LMAX=views.pao_lmax;
        Spe_PAO_Mul=views.pao_multiplicity;
        Spe_WhatAtom=views.atomic_number;
        Spe_Total_NO=views.orbitals;
        Spe_Total_CNO=views.contracted_orbitals;
      }
      Species_Top = (int*)malloc(sizeof(int)*Num_Procs);
      Species_End = (int*)malloc(sizeof(int)*Num_Procs);
      F_Snd_Num = (int*)malloc(sizeof(int)*Num_Procs);
      S_Snd_Num = (int*)malloc(sizeof(int)*Num_Procs);
      F_Rcv_Num = (int*)malloc(sizeof(int)*Num_Procs);
      S_Rcv_Num = (int*)malloc(sizeof(int)*Num_Procs);
      F_TopMAN = (int*)malloc(sizeof(int)*Num_Procs);
      S_TopMAN = (int*)malloc(sizeof(int)*Num_Procs);
      Snd_HFS_Size = (int*)malloc(sizeof(int)*Num_Procs);
      Rcv_HFS_Size = (int*)malloc(sizeof(int)*Num_Procs);

      Num_Snd_Grid_A2B = (int*)malloc(sizeof(int)*Num_Procs);
      Num_Rcv_Grid_A2B = (int*)malloc(sizeof(int)*Num_Procs);
      Num_Snd_Grid_B2C = (int*)malloc(sizeof(int)*Num_Procs);
      Num_Rcv_Grid_B2C = (int*)malloc(sizeof(int)*Num_Procs);
      Num_Snd_Grid_B2D = (int*)malloc(sizeof(int)*Num_Procs);
      Num_Rcv_Grid_B2D = (int*)malloc(sizeof(int)*Num_Procs);
      Num_Snd_Grid_B_AB2CA = (int*)malloc(sizeof(int)*Num_Procs);
      Num_Rcv_Grid_B_AB2CA = (int*)malloc(sizeof(int)*Num_Procs);
      Num_Snd_Grid_B_CA2CB = (int*)malloc(sizeof(int)*Num_Procs);
      Num_Rcv_Grid_B_CA2CB = (int*)malloc(sizeof(int)*Num_Procs);
/* added by mari 05.12.2014 */
      Num_Snd_Grid_B_AB2C = (int*)malloc(sizeof(int)*Num_Procs);
      Num_Rcv_Grid_B_AB2C = (int*)malloc(sizeof(int)*Num_Procs);


      EH0_scaling = (double**)malloc(sizeof(double*)*SpeciesNum);
      for (i=0; i<SpeciesNum; i++){
        EH0_scaling[i] = (double*)malloc(sizeof(double)*SpeciesNum);
      }

      SO_factor = (double**)malloc(sizeof(double*)*SpeciesNum);
      for (i=0; i<SpeciesNum; i++){
        SO_factor[i] = (double*)malloc(sizeof(double)*4);
      }

    break;

    case 1:
      /* call from Input_std.c */

      Gxyz = (double**)malloc(sizeof(double*)*(atomnum+4));
      for (i=0; i<(atomnum+4); i++){
        Gxyz[i] = (double*)malloc(sizeof(double)*YOUSO26);
        for (j=0; j<YOUSO26; j++){
          Gxyz[i][j] = 0.0;
        }
      }

      num = M_GDIIS_HISTORY + 1;
      
      GxyzHistoryIn= (double***)malloc(sizeof(double**)*num);
      for(i=0; i<num; i++) {
        GxyzHistoryIn[i] = (double**)malloc(sizeof(double*)*(atomnum+4));
        for(j=0; j<(atomnum+4); j++) {
          GxyzHistoryIn[i][j] = (double*)malloc(sizeof(double)*4);
          for (k=0; k<4; k++){
            GxyzHistoryIn[i][j][k] = 0.0;
	  }
	}
      }

      GxyzHistoryR= (double***)malloc(sizeof(double**)*num);
      for(i=0; i<num; i++) {
        GxyzHistoryR[i] = (double**)malloc(sizeof(double*)*(atomnum+4));
        for(j=0; j<(atomnum+4); j++) {
          GxyzHistoryR[i][j] = (double*)malloc(sizeof(double)*4);
          for (k=0; k<4; k++){
            GxyzHistoryR[i][j][k] = 0.0;
          }
	}
      }

      His_Gxyz = (double**)malloc(sizeof(double*)*Extrapolated_Charge_History);
      for(i=0; i<Extrapolated_Charge_History; i++) {
        His_Gxyz[i] = (double*)malloc(sizeof(double)*(atomnum*3));
      }

      atom_Fixed_XYZ = (int**)malloc(sizeof(int*)*(atomnum+1));
      for(i=0; i<=atomnum; i++){
        atom_Fixed_XYZ[i] = (int*)malloc(sizeof(int)*4);
        /* default='relaxed' */
        atom_Fixed_XYZ[i][1] = 0;  
        atom_Fixed_XYZ[i][2] = 0;
        atom_Fixed_XYZ[i][3] = 0;
      }

      Cell_Gxyz = (double**)malloc(sizeof(double*)*(atomnum+4));
      for (i=0; i<(atomnum+4); i++){
        Cell_Gxyz[i] = (double*)malloc(sizeof(double)*4);
      }      

      InitN_USpin = (double*)malloc(sizeof(double)*(atomnum+1));
      InitN_DSpin = (double*)malloc(sizeof(double)*(atomnum+1));
      for (i=0; i<=atomnum; i++){
        InitN_USpin[i] = 0.0;
        InitN_DSpin[i] = 0.0;
      } 
      WhatSpecies = (int*)malloc(sizeof(int)*(atomnum+1));
      GridN_Atom = (int*)malloc(sizeof(int)*(atomnum+1));
      RNUM  = (int*)malloc(sizeof(int)*(atomnum+1));
      RNUM2 = (int*)malloc(sizeof(int)*(atomnum+1));
      G2ID = (int*)malloc(sizeof(int)*(atomnum+1));
      F_G2M = (int*)malloc(sizeof(int)*(atomnum+1));
      S_G2M = (int*)malloc(sizeof(int)*(atomnum+1));
      for (i=0; i<=atomnum; i++){
        F_G2M[i] = -1;
        S_G2M[i] = -1;
      } 

      time_per_atom = (double*)malloc(sizeof(double)*(atomnum+1));
      for (i=1; i<=atomnum; i++) time_per_atom[i] = 0.0;

      if (Solver==1 || Solver==5 || Solver==6 || Solver==8 || Solver==11){
        orderN_FNAN = (int*)malloc(sizeof(int)*(atomnum+1)); 
        orderN_SNAN = (int*)malloc(sizeof(int)*(atomnum+1)); 
      }

      /* spin non-collinear */
      if (SpinP_switch==3){
        Angle0_Spin = (double*)malloc(sizeof(double)*(atomnum+1));
        Angle1_Spin = (double*)malloc(sizeof(double)*(atomnum+1));
        InitAngle0_Spin = (double*)malloc(sizeof(double)*(atomnum+1));
        InitAngle1_Spin = (double*)malloc(sizeof(double)*(atomnum+1));
        Angle0_Orbital = (double*)malloc(sizeof(double)*(atomnum+1));
        Angle1_Orbital = (double*)malloc(sizeof(double)*(atomnum+1));
        InitAngle0_Orbital = (double*)malloc(sizeof(double)*(atomnum+1));
        InitAngle1_Orbital = (double*)malloc(sizeof(double)*(atomnum+1));
        OrbitalMoment = (double*)malloc(sizeof(double)*(atomnum+1));

        for (i=0; i<=atomnum; i++){
          Angle0_Spin[i] = 0.0;
          Angle1_Spin[i] = 0.0;
          InitAngle0_Spin[i] = 0.0;
          InitAngle1_Spin[i] = 0.0;
          Angle0_Orbital[i] = 0.0;
          Angle1_Orbital[i] = 0.0;
          InitAngle0_Orbital[i] = 0.0;
          InitAngle1_Orbital[i] = 0.0;
          OrbitalMoment[i] = 0.0;
        } 

        Constraint_SpinAngle = (int*)malloc(sizeof(int)*(atomnum+1));
        for(i=1; i<=atomnum; i++) Constraint_SpinAngle[i]=0; /* default='no constraint' */

        Orbital_Moment_XYZ = (double**)malloc(sizeof(double*)*(atomnum+1));
        for(i=0; i<(atomnum+1); i++){
          Orbital_Moment_XYZ[i] = (double*)malloc(sizeof(double)*3);
	}
        Constraint_OrbitalAngle = (int*)malloc(sizeof(int)*(atomnum+1));
        for(i=1; i<=atomnum; i++) Constraint_OrbitalAngle[i]=0; /* default='no constraint' */
      }

      /* arrays for LDA+U added by MJ */

      if (Hub_U_switch==1 || 1<=Constraint_NCS_switch || Zeeman_NCS_switch==1 || Zeeman_NCO_switch==1){

	Hub_U_Basis =  (double***)malloc(sizeof(double**)*SpeciesNum);
	for (i=0; i<SpeciesNum; i++){
	  Hub_U_Basis[i] = (double**)malloc(sizeof(double*)*(Spe_MaxL_Basis[i]+1));
	  for (l=0; l<(Spe_MaxL_Basis[i]+1); l++){
	    Hub_U_Basis[i][l] = (double*)malloc(sizeof(double)*Spe_Num_Basis[i][l]);
	  }
	}

        OrbPol_flag = (int*)malloc(sizeof(int)*(atomnum+1));
      }

      /* arrays for general LDA+U by S.Ryee */

      if (Hub_U_switch==1 && Hub_Type==2){

        /* Array for Hund J */
	Hund_J_Basis =  (double***)malloc(sizeof(double**)*SpeciesNum);
	for (i=0; i<SpeciesNum; i++){
	  Hund_J_Basis[i] = (double**)malloc(sizeof(double*)*(Spe_MaxL_Basis[i]+1));
	  for (l=0; l<(Spe_MaxL_Basis[i]+1); l++){
	    Hund_J_Basis[i][l] = (double*)malloc(sizeof(double)*Spe_Num_Basis[i][l]);
	  }
	}

        /* Array for generating index of orbitals having nonzero Hubbard U and Hund J */
	Nonzero_UJ =  (int***)malloc(sizeof(double**)*SpeciesNum);
	for (i=0; i<SpeciesNum; i++){
	  Nonzero_UJ[i] = (int**)malloc(sizeof(double*)*(Spe_MaxL_Basis[i]+1));
	  for (l=0; l<(Spe_MaxL_Basis[i]+1); l++){
	    Nonzero_UJ[i][l] = (int*)malloc(sizeof(double)*Spe_Num_Basis[i][l]);
	  }
	}
      }

      /* EF */

      if (MD_switch==4){

        Hessian = (double**)malloc(sizeof(double*)*(3*atomnum+2));
	for (i=0; i<(3*atomnum+2); i++){
          Hessian[i] = (double*)malloc(sizeof(double)*(3*atomnum+2));
  	  for (j=0; j<(3*atomnum+2); j++){
            Hessian[i][j] = 0.0;
	  }
	}
      }

      /* BFGS */

      if (MD_switch==5){

        InvHessian = (double**)malloc(sizeof(double*)*(3*atomnum+2));
	for (i=0; i<(3*atomnum+2); i++){
          InvHessian[i] = (double*)malloc(sizeof(double)*(3*atomnum+2));
  	  for (j=0; j<(3*atomnum+2); j++){
            InvHessian[i][j] = 0.0;
	  }
	}
      }

      /* RF added by hmweng */

      if (MD_switch==6){

        Hessian = (double**)malloc(sizeof(double*)*(3*atomnum+2));
	for (i=0; i<(3*atomnum+2); i++){
          Hessian[i] = (double*)malloc(sizeof(double)*(3*atomnum+2));
  	  for (j=0; j<(3*atomnum+2); j++){
            Hessian[i][j] = 0.0;
	  }
	}
      }

      /* RFC */

      if (MD_switch==18){

        Hessian = (double**)malloc(sizeof(double*)*(3*atomnum+11));
	for (i=0; i<(3*atomnum+11); i++){
          Hessian[i] = (double*)malloc(sizeof(double)*(3*atomnum+11));
  	  for (j=0; j<(3*atomnum+11); j++){
            Hessian[i][j] = 0.0;
	  }
	}
      }

      /* Constraint_NCS_switch==2 */
      if (Constraint_NCS_switch==2){
        InitMagneticMoment = (double*)malloc(sizeof(double)*(atomnum+1));
      }

      /* LNO_Num  */  

      if (LNO_flag==1){
        LNO_Num = (int*)malloc(sizeof(int)*(atomnum+1)); 
        LNOs_Num_predefined = (int*)malloc(sizeof(int)*SpeciesNum);
      }

      if (CWF_Calc==1){
	CWF_Guiding_AO =  (int***)malloc(sizeof(int**)*SpeciesNum);
	for (i=0; i<SpeciesNum; i++){
	  CWF_Guiding_AO[i] = (int**)malloc(sizeof(int*)*(Spe_MaxL_Basis[i]+1));
	  for (l=0; l<(Spe_MaxL_Basis[i]+1); l++){
	    CWF_Guiding_AO[i][l] = (int*)malloc(sizeof(int)*Spe_Num_Basis[i][l]);
            for (k=0; k<Spe_Num_Basis[i][l]; k++){
  	      CWF_Guiding_AO[i][l][k] = 0;
	    }
	  }
	}

        CWF_Num_predefined = (int*)malloc(sizeof(int)*SpeciesNum);
        for (k=0; k<SpeciesNum; k++){
          CWF_Num_predefined[k] = 1;
	}
      }

    break;

    case 2:
      /* call from readfile.c */

      {
        const OpenMXNeutralGridMetadataShape shape={(size_t)SpeciesNum};
        OpenMXNeutralGridMetadataView metadata;
        char error[256];
        if (!openmx_core_neutral_grid_metadata_prepare(core,&shape,&metadata,error,sizeof(error))) {
          fprintf(stderr,"Allocate_Arrays: neutral grid metadata: %s\n",error);
          MPI_Abort(mpi_comm_level1,1);
          return;
        }
        TGN_EH0=metadata.counts;
        dv_EH0=metadata.spacing;
      }
      FNAN = (int*)malloc(sizeof(int)*(atomnum+1));
      SNAN = (int*)malloc(sizeof(int)*(atomnum+1));
      ONAN = (int*)malloc(sizeof(int)*(atomnum+1));
      zp = (dcomplex*)malloc(sizeof(dcomplex)*POLES);
      Ep = (dcomplex*)malloc(sizeof(dcomplex)*POLES);
      Rp = (dcomplex*)malloc(sizeof(dcomplex)*POLES);

      if (Solver==11){
        FNAN_DCLNO = (int*)malloc(sizeof(int)*(atomnum+1));
        SNAN_DCLNO = (int*)malloc(sizeof(int)*(atomnum+1));
      }

      /* Cluster2 */

      if (Solver==9){

        ON2_zp = (dcomplex*)malloc(sizeof(dcomplex)*(ON2_Npoles+2));
        ON2_Rp = (dcomplex*)malloc(sizeof(dcomplex)*(ON2_Npoles+2));
        ON2_method = (int*)malloc(sizeof(int)*(ON2_Npoles+2));

        ON2_zp_f = (dcomplex*)malloc(sizeof(dcomplex)*(ON2_Npoles+2));
        ON2_Rp_f = (dcomplex*)malloc(sizeof(dcomplex)*(ON2_Npoles+2));
        ON2_method_f = (int*)malloc(sizeof(int)*(ON2_Npoles+2));

 	for (i=0; i<(ON2_Npoles+2); i++){
          ON2_zp[i] = Complex(0.0,0.0);
          ON2_Rp[i] = Complex(0.0,0.0);
          ON2_zp_f[i] = Complex(0.0,0.0);
          ON2_Rp_f[i] = Complex(0.0,0.0);
          ON2_method[i] = 0;
          ON2_method_f[i] = 0;
	}
      }

      /* EGAC */

      if (Solver==10){

        EGAC_Top = (int*)malloc(sizeof(int)*Num_Procs);
        EGAC_End = (int*)malloc(sizeof(int)*Num_Procs);
        Num_Rcv_HS_EGAC = (int*)malloc(sizeof(int)*Num_Procs);
        Num_Snd_HS_EGAC = (int*)malloc(sizeof(int)*Num_Procs);
        Num_Rcv_DM_EGAC = (int*)malloc(sizeof(int)*Num_Procs);
        Num_Snd_DM_EGAC = (int*)malloc(sizeof(int)*Num_Procs);
        Top_Index_HS_EGAC = (int*)malloc(sizeof(int)*Num_Procs);
        Num_Rcv_GA_EGAC = (int*)malloc(sizeof(int)*Num_Procs);
        Num_Snd_GA_EGAC = (int*)malloc(sizeof(int)*Num_Procs);
        Top_Index_GA_EGAC = (int*)malloc(sizeof(int)*Num_Procs);
        Snd_GA_EGAC_Size = (int*)malloc(sizeof(int)*Num_Procs);
        Rcv_GA_EGAC_Size = (int*)malloc(sizeof(int)*Num_Procs);
        Snd_OLP_EGAC_Size = (int*)malloc(sizeof(int)*Num_Procs);
        Rcv_OLP_EGAC_Size = (int*)malloc(sizeof(int)*Num_Procs);
        G2M_EGAC = (int*)malloc(sizeof(int)*(atomnum+1));
        G2M_DM_Snd_EGAC = (int*)malloc(sizeof(int)*(atomnum+1));
        Snd_DM_EGAC_Size = (int*)malloc(sizeof(int)*Num_Procs);
        Rcv_DM_EGAC_Size = (int*)malloc(sizeof(int)*Num_Procs);

        EGAC_zp = (dcomplex*)malloc(sizeof(dcomplex)*(EGAC_Npoles+202));
        EGAC_Rp = (dcomplex*)malloc(sizeof(dcomplex)*(EGAC_Npoles+202));
        EGAC_method = (int*)malloc(sizeof(int)*(EGAC_Npoles+202));

        EGAC_zp_f = (dcomplex*)malloc(sizeof(dcomplex)*(EGAC_Npoles+202));
        EGAC_Rp_f = (dcomplex*)malloc(sizeof(dcomplex)*(EGAC_Npoles+202));
        EGAC_method_f = (int*)malloc(sizeof(int)*(EGAC_Npoles+202));

 	for (i=0; i<(EGAC_Npoles+2); i++){
          EGAC_zp[i] = Complex(0.0,0.0);
          EGAC_Rp[i] = Complex(0.0,0.0);
          EGAC_zp_f[i] = Complex(0.0,0.0);
          EGAC_Rp_f[i] = Complex(0.0,0.0);
          EGAC_method[i] = 0;
          EGAC_method_f[i] = 0;
	}
      }

    break;

    case 3:

      /* call from truncation.c */

      if (alloc_first[8]==0){

	for (ct_AN=0; ct_AN<=atomnum; ct_AN++){
	  free(natn[ct_AN]);
	}
        free(natn);

	for (ct_AN=0; ct_AN<=atomnum; ct_AN++){
	  free(ncn[ct_AN]);
	}
        free(ncn);

	for (ct_AN=0; ct_AN<=atomnum; ct_AN++){
	  free(Dis[ct_AN]);
	}
        free(Dis);
      }

      natn = (int**)malloc(sizeof(int*)*(atomnum+1));
      for (ct_AN=0; ct_AN<=atomnum; ct_AN++){
        natn[ct_AN] = (int*)malloc(sizeof(int)*((int)(Max_FSNAN*ScaleSize)+1));
      }

      ncn  = (int**)malloc(sizeof(int*)*(atomnum+1));
      for (ct_AN=0; ct_AN<=atomnum; ct_AN++){
        ncn[ct_AN] = (int*)malloc(sizeof(int)*((int)(Max_FSNAN*ScaleSize)+1));
      }

      Dis = (double**)malloc(sizeof(double*)*(atomnum+1));
      for (ct_AN=0; ct_AN<=atomnum; ct_AN++){
        Dis[ct_AN] = (double*)malloc(sizeof(double)*((int)(Max_FSNAN*ScaleSize)+1));
      }

      alloc_first[8] = 0;

    break;

    case 4:

      if (alloc_first[32]==0){

        if (Energy_Decomposition_flag==1){

	  for (i=0; i<SpeciesNum; i++){
	    for (k=0; k<Max_TGN_EH0; k++){
	      for (l=0; l<(Spe_MaxL_Basis[i]+1); l++){
		free(Arho_EH0_Orb[i][k][l]);
	      }
   	      free(Arho_EH0_Orb[i][k]);
	    }
            free(Arho_EH0_Orb[i]);
	  }
          free(Arho_EH0_Orb);
	}
      }

      /* The calculation owns the flat numeric values. */
      {
        const OpenMXNeutralGridShape shape={SpeciesNum,Max_TGN_EH0};
        OpenMXNeutralGridTables tables;
        char error[256];
        if (!openmx_core_neutral_grid_prepare(core,&shape,&tables,error,sizeof(error))) {
          fprintf(stderr,"Allocate_Arrays: neutral grid: %s\n",error);
          MPI_Abort(mpi_comm_level1,1);
          return;
        }
      }


      if (Energy_Decomposition_flag==1){

        Arho_EH0_Orb = (double****)malloc(sizeof(double***)*SpeciesNum);
        for (i=0; i<SpeciesNum; i++){
          Arho_EH0_Orb[i] = (double***)malloc(sizeof(double**)*Max_TGN_EH0);
	  for (k=0; k<Max_TGN_EH0; k++){
            Arho_EH0_Orb[i][k] = (double**)malloc(sizeof(double*)*(Spe_MaxL_Basis[i]+1));
	    for (l=0; l<(Spe_MaxL_Basis[i]+1); l++){
              Arho_EH0_Orb[i][k][l] = (double*)malloc(sizeof(double)*Spe_Num_Basis[i][l]);
	    }
	  }
        }
      }

      alloc_first[32] = 0;

    break;

    case 5:

      /* call from Input_std.c */

      MO_kpoint = (double**)malloc(sizeof(double*)*(MO_Nkpoint+1));
      for (i=0; i<(MO_Nkpoint+1); i++){
        MO_kpoint[i] = (double*)malloc(sizeof(double)*4);
      }

    break;

    case 6:
      if (!core) {
        fprintf(stderr,"Calculation tables require an explicit OpenMX core context\n");
        MPI_Abort(mpi_comm_level1,1);
        return;
      }
      {
        char error[256];
        OpenMXPaoViews views;
        OpenMXPaoShape shape;
        if (List_YOUSO[25] < 0 || List_YOUSO[25] == INT_MAX) {
          fprintf(stderr,"Invalid PAO angular capacity\n");
          MPI_Abort(mpi_comm_level1,1);
          return;
        }
        shape = (OpenMXPaoShape){List_YOUSO[18],List_YOUSO[25]+1,List_YOUSO[24],
                                List_YOUSO[21],List_YOUSO[15]};
        if (!openmx_core_pao_allocate(core,&shape,&views,error,sizeof(error))) {
          fprintf(stderr,"Allocate_Arrays(PAO): %s\n",error);
          MPI_Abort(mpi_comm_level1,1);
          return;
        }



      }
      break;

    case 7:
      if (!core) {
        fprintf(stderr,"Calculation tables require an explicit OpenMX core context\n");
        MPI_Abort(mpi_comm_level1,1);
        return;
      }
      {
        char error[256];
        OpenMXVpsViews views;
        OpenMXVpsShape shape;
        OpenMXNeutralProjectorShape neutral_shape;
        const OpenMXNeutralProjectorShape *neutral = NULL;
        if (SO_switch < 0 || SO_switch > 1 ||
            (ProExpn_VNA==1 && (List_YOUSO[35] < 0 || List_YOUSO[35] == INT_MAX))) {
          fprintf(stderr,"Invalid VPS spin or VNA angular capacity\n");
          MPI_Abort(mpi_comm_level1,1);
          return;
        }
        shape = (OpenMXVpsShape){List_YOUSO[18],SO_switch+1,List_YOUSO[19],
                                List_YOUSO[22],List_YOUSO[15]};
        if (ProExpn_VNA==1) {
          neutral_shape = (OpenMXNeutralProjectorShape){List_YOUSO[35]+1,List_YOUSO[34],GL_Mesh};
          neutral = &neutral_shape;
        }
        if (!openmx_core_vps_allocate(core,&shape,neutral,&views,error,sizeof(error))) {
          fprintf(stderr,"Allocate_Arrays(VPS): %s\n",error);
          MPI_Abort(mpi_comm_level1,1);
          return;
        }


      }

      if (ProExpn_VNA==1) {
        const OpenMXProductShape shape = {List_YOUSO[18],GL_Mesh,Spe_MaxL_Basis,Spe_Num_Basis};
        OpenMXProductViews views;
        char error[256];
        if (!openmx_core_product_allocate(core,&shape,&views,error,sizeof(error))) {
          fprintf(stderr,"Allocate_Arrays(products): %s\n",error);
          MPI_Abort(mpi_comm_level1,1);
          return;
        }
      }
      break;

    case 8:  /* hmweng */

      Wannier_ProSpeName = (char**)malloc(sizeof(char*)*Wannier_Num_Kinds_Projectors);
      for (i=0; i<Wannier_Num_Kinds_Projectors; i++){
        Wannier_ProSpeName[i] = (char*)malloc(sizeof(char)*YOUSO10);
      }  

      Wannier_ProName = (char**)malloc(sizeof(char*)*Wannier_Num_Kinds_Projectors);
      for (i=0; i<Wannier_Num_Kinds_Projectors; i++){
        Wannier_ProName[i] = (char*)malloc(sizeof(char)*YOUSO10);
      }  

      Wannier_Pos = (double**)malloc(sizeof(double*)*Wannier_Num_Kinds_Projectors);
      for (i=0; i<Wannier_Num_Kinds_Projectors; i++){
        Wannier_Pos[i] = (double*)malloc(sizeof(double)*4);
      }

      Wannier_X_Direction = (double**)malloc(sizeof(double*)*Wannier_Num_Kinds_Projectors);
      for (i=0; i<Wannier_Num_Kinds_Projectors; i++){
        Wannier_X_Direction[i] = (double*)malloc(sizeof(double)*4);
      }

      Wannier_Z_Direction = (double**)malloc(sizeof(double*)*Wannier_Num_Kinds_Projectors);
      for (i=0; i<Wannier_Num_Kinds_Projectors; i++){
        Wannier_Z_Direction[i] = (double*)malloc(sizeof(double)*4);
      }

      Wannier_Num_Pro = (int*)malloc(sizeof(int)*Wannier_Num_Kinds_Projectors);

      Wannier_NumL_Pro = (int**)malloc(sizeof(int*)*Wannier_Num_Kinds_Projectors);
      for (i=0; i<Wannier_Num_Kinds_Projectors; i++){
        Wannier_NumL_Pro[i] = (int*)malloc(sizeof(int)*4);
      }

      WannierPro2SpeciesNum = (int*)malloc(sizeof(int)*Wannier_Num_Kinds_Projectors);
      Wannier_Guide=(double**)malloc(sizeof(double*)*3);
      for(i=0;i<3;i++){
        Wannier_Guide[i]=(double*)malloc(sizeof(double)*Wannier_Func_Num);
      }


      Wannier_Euler_Rotation_Angle = (double**)malloc(sizeof(double*)*Wannier_Num_Kinds_Projectors);
      for (i=0; i<Wannier_Num_Kinds_Projectors; i++){
        Wannier_Euler_Rotation_Angle[i] = (double*)malloc(sizeof(double)*3);
        for(j=0; j<3; j++){
          Wannier_Euler_Rotation_Angle[i][j] =0.0; 
        }
      } 

      Wannier_RotMat_for_Real_Func = (double****)malloc(sizeof(double***)*Wannier_Num_Kinds_Projectors);
      for (i=0; i<Wannier_Num_Kinds_Projectors; i++){
        Wannier_RotMat_for_Real_Func[i] = (double***)malloc(sizeof(double**)*4);
        for(j=0; j<4; j++){
          Wannier_RotMat_for_Real_Func[i][j] = (double**)malloc(sizeof(double*)*(2*j+1));
          for(l=0; l<2*j+1; l++){
            Wannier_RotMat_for_Real_Func[i][j][l] = (double*)malloc(sizeof(double)*(2*j+1));
          }
        }
      }
      
      Wannier_ProName2Num=(int*)malloc(sizeof(int)*Wannier_Num_Kinds_Projectors);

      alloc_first[24] = 0;

      break;

      /* hmweng */

      case 9:

      Wannier_Select_Matrix=(int**)malloc(sizeof(int*)*Wannier_Num_Kinds_Projectors);
      for(i=0; i<Wannier_Num_Kinds_Projectors; i++){
        Wannier_Select_Matrix[i]=(int*)malloc(sizeof(int)*Wannier_Num_Pro[i]);
        for(j=0;j<Wannier_Num_Pro[i];j++){
	  Wannier_Select_Matrix[i][j]=-1;
        } 
      }

      Wannier_Projector_Hybridize_Matrix=(double***)malloc(sizeof(double**)*Wannier_Num_Kinds_Projectors);
      for(i=0; i<Wannier_Num_Kinds_Projectors; i++){
        Wannier_Projector_Hybridize_Matrix[i]=(double**)malloc(sizeof(double*)*Wannier_Num_Pro[i]);
        for(j=0;j<Wannier_Num_Pro[i];j++){
	  Wannier_Projector_Hybridize_Matrix[i][j]=(double*)malloc(sizeof(double)*Wannier_Num_Pro[i]);
        }
      }

      alloc_first[25] = 0; 

      break;


  case 10:

    /* MD_VS4 */

    AtomGr = (int*)malloc(sizeof(int)*(atomnum+1));
    atnum_AtGr = (int*)malloc(sizeof(int)*(num_AtGr+1));
    Temp_AtGr = (double*)malloc(sizeof(double)*(num_AtGr+1));

    break;

  case 11: /* NBO by T.Ohwaki */

    NBO_FCenter = (int*)malloc(sizeof(int)*(Num_NBO_FCenter+1));

    Num_NHOs = (int*)malloc(sizeof(int)*(atomnum+1));

    NAO_kpoint = (double**)malloc(sizeof(double*)*(NAO_Nkpoint+1));
    for (i=0; i<(NAO_Nkpoint+1); i++){
      NAO_kpoint[i] = (double*)malloc(sizeof(double)*4);
    }

    rlmax_EC_NAO = (int*)malloc(sizeof(int)*(atomnum+1));
    for (i=0; i<=atomnum; i++){
      rlmax_EC_NAO[i] = 0;
    }

    rlmax_EC2_NAO = (int*)malloc(sizeof(int)*(atomnum+1));
    for (i=0; i<=atomnum; i++){
      rlmax_EC2_NAO[i] = 0;
    }

    EKC_core_size_NAO = (int*)malloc(sizeof(int)*(atomnum+1));
    for (i=0; i<=atomnum; i++){
      EKC_core_size_NAO[i] = 0;
    }

    F_Snd_Num_NAO = (int*)malloc(sizeof(int)*Num_Procs);
    S_Snd_Num_NAO = (int*)malloc(sizeof(int)*Num_Procs);
    F_Rcv_Num_NAO = (int*)malloc(sizeof(int)*Num_Procs);
    S_Rcv_Num_NAO = (int*)malloc(sizeof(int)*Num_Procs);

    F_TopMAN_NAO = (int*)malloc(sizeof(int)*Num_Procs);
    S_TopMAN_NAO = (int*)malloc(sizeof(int)*Num_Procs);

    F_G2M_NAO = (int*)malloc(sizeof(int)*(atomnum+1));
    S_G2M_NAO = (int*)malloc(sizeof(int)*(atomnum+1));

    Snd_HFS_Size_NAO = (int*)malloc(sizeof(int)*Num_Procs);
    Rcv_HFS_Size_NAO = (int*)malloc(sizeof(int)*Num_Procs);

    break;

    case 12: /* COHP */

    COHP_AtomA = (int*)malloc(sizeof(int)*COHP_num_pairs);
    COHP_AtomB = (int*)malloc(sizeof(int)*COHP_num_pairs);
    COHP_CellB1 = (int*)malloc(sizeof(int)*COHP_num_pairs);
    COHP_CellB2 = (int*)malloc(sizeof(int)*COHP_num_pairs);
    COHP_CellB3 = (int*)malloc(sizeof(int)*COHP_num_pairs);

    break;

    case 13: /* LNAO */

    LNAO_Atoms = (int*)malloc(sizeof(int)*LNAO_num);

    break;

    case 14: /* LNBO */

    LNBO_Atoms = (int**)malloc(sizeof(int*)*LNBO_num);
    for (i=0; i<LNBO_num; i++){
      LNBO_Atoms[i] = (int*)malloc(sizeof(int)*5);
    }

    break;

    /* efficient exchange methods */
    
    case 15:

      Ndim = Ng1_Rec_Coulomb*Ng1_Rec_Coulomb*Ng1_Rec_Coulomb;
      SVals_Coulomb = (double*)malloc(sizeof(double)*Ndim);
      SVecs_U_Coulomb = (double*)malloc(sizeof(double)*Ndim*Ndim);
      SVecs_V_Coulomb = (double*)malloc(sizeof(double)*Ndim*Ndim);
      qgrid_Rec_Coulomb = (double*)malloc(sizeof(double)*Ng1_Rec_Coulomb);
      
    break;

  }

}

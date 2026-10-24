/**********************************************************************
  Stress.c:
  
     Stress.c is a subroutine to calculate stress on a supercell.
     This subroutine is a modification of Force.c

  Log of Stress.c:

     150326 Kinetic and overlap terms are implemented by yshiihara.
***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "openmx_common.h"
#include "openmx_stress_workspace.h"
#include "openmx_neutral_atom.h"
#include "openmx_core_neutral_grid.h"
#include "openmx_orbitals.h"
#include "openmx_core_vna_matrix.h"
#include "openmx_precision.h"
#include "mpi.h"
#include "openmx_projector_exchange.h"
#include <omp.h>
#include <fftw3.h>

#define measure_time 0

/*** subroutine */
static void MPI_OLP(OpenMXSolverInvocation *scratch,double *****OLP1);
static void Stress3(OpenMXSolverInvocation *scratch,const OpenMXAtomicViews *atomic, const OpenMXElectronicViews *electronic, const OpenMXLocalBasisViews *local_basis);
static void Stress4(const OpenMXAtomicViews *atomic);
static void Stress4B(OpenMXSolverInvocation *scratch,const OpenMXVnaMatrixViews *vna, const OpenMXReciprocalViews *reciprocal, double *****CDM0, const OpenMXLocalBasisViews *local_basis);
static void Stress_core(OpenMXSolverInvocation *scratch);
static void Stress_H0(OpenMXSolverInvocation *scratch,OpenMXCoreContext *core,const OpenMXAtomicViews *atomic);
static void Stress_H1(OpenMXSolverInvocation *scratch);
static void Stress_HNL(OpenMXSolverInvocation *scratch,const OpenMXAtomicViews *atomic, double *****CDM0, double *****iDM0, const OpenMXElectronicViews *electronic, const OpenMXLocalBasisViews *local_basis);
static void Stress_plus_U_dual(OpenMXSolverInvocation *scratch,double *****CDM0, const OpenMXElectronicViews *electronic, const OpenMXLocalBasisViews *local_basis);
static void EH0_TwoCenter(const OpenMXNeutralGridTables *tables,const OpenMXAtomicViews *atomic, int Gc_AN, int h_AN, double VH0ij[4]); /*from Total_Energy.c*/
static void EH0_TwoCenter_at_Cutoff(OpenMXSolverInvocation *scratch,const OpenMXNeutralGridTables *tables,const OpenMXAtomicViews *atomic, int wan1, int wan2, double VH0ij[4]); /*from Total_Energy.c*/

static void dHVNA(const OpenMXVnaMatrixViews *vna, int where_flag, int Mc_AN, int h_AN, int q_AN,
		  Type_DS_VNA *****DS_VNA1, 
		  double *****TmpHVNA2, double *****TmpHVNA3, 
		  const OpenMXSolverRealTensor *Hx,const OpenMXSolverRealTensor *Hy,
           const OpenMXSolverRealTensor *Hz,const OpenMXSolverRealTensor *H_sts);

static void dHNL(const OpenMXAtomicViews *atomic, int where_flag,
		 int Mc_AN, int h_AN, int q_AN,
		 double ******DS_NL1,
		 const OpenMXSolverComplexTensor *Hx,const OpenMXSolverComplexTensor *Hy,
          const OpenMXSolverComplexTensor *Hz,const OpenMXSolverComplexTensor *H_sts, const OpenMXElectronicViews *electronic);

static void dHNL_SO(
	     double *sumx0r,
	     double *sumy0r, 
	     double *sumz0r, 
	     double *sumx1r,
	     double *sumy1r, 
	     double *sumz1r, 
	     double *sumx2r,
	     double *sumy2r, 
	     double *sumz2r, 
	     double *sumx0i,
	     double *sumy0i, 
	     double *sumz0i, 
	     double *sumx1i,
	     double *sumy1i, 
	     double *sumz1i, 
	     double *sumx2i,
	     double *sumy2i, 
	     double *sumz2i, 
             double fugou,
	     double PFp,
	     double PFm,
	     double ene_p,
	     double ene_m,
	     int l2, int *l,
	     int Mc_AN, int k,  int m,
	     int Mj_AN, int kl, int n,
	     double ******DS_NL1);







typedef struct {
  OpenMXCoreContext *core;
  OpenMXThermodynamicEnergies *thermo;
  const OpenMXAtomicViews *atomic;
  const OpenMXVnaMatrixViews *vna;
  const OpenMXReciprocalViews *reciprocal;
  double *****H0,******DS_NL,*****OLP,*****CDM,*****EDM;
  const OpenMXElectronicViews *electronic;
  const OpenMXLocalBasisViews *local_basis;
} OpenMXStressArguments;
static double Stress_owned(const OpenMXForceWorkspace *halo,OpenMXCoreContext *core,OpenMXThermodynamicEnergies *thermo, const OpenMXAtomicViews *atomic, const OpenMXVnaMatrixViews *vna, const OpenMXReciprocalViews *reciprocal, double *****H0,
	      double ******DS_NL,
	      double *****OLP,
	      double *****CDM,
	      double *****EDM, const OpenMXElectronicViews *electronic, const OpenMXLocalBasisViews *local_basis);
static int Stress_Operation(const OpenMXForceWorkspace *halo,void *opaque,double *result)
{
  const OpenMXStressArguments *a=opaque;
  *result=Stress_owned(halo,a->core,a->thermo,a->atomic,a->vna,a->reciprocal,a->H0,a->DS_NL,
      a->OLP,a->CDM,a->EDM,a->electronic,a->local_basis);
  return 1;
}
double Stress(OpenMXCoreContext *core,OpenMXThermodynamicEnergies *thermo, const OpenMXAtomicViews *atomic, const OpenMXVnaMatrixViews *vna, const OpenMXReciprocalViews *reciprocal, double *****H0,
	      double ******DS_NL,
	      double *****OLP,
	      double *****CDM,
	      double *****EDM, const OpenMXElectronicViews *electronic, const OpenMXLocalBasisViews *local_basis)
{
  OpenMXStressArguments args={core,thermo,atomic,vna,reciprocal,H0,DS_NL,OLP,CDM,EDM,electronic,local_basis};
  double begin,end;
  dtime(&begin);
  const int imaginary=SO_switch==1 || (Hub_U_switch==1 && SpinP_switch==3) || Constraint_NCS_switch==1
      || Zeeman_NCS_switch==1 || Zeeman_NCO_switch==1;
  const OpenMXForceInput input={atomnum,Matomnum,MatomnumF,SpeciesNum,SpinP_switch,imaginary,
      F_M2G,WhatSpecies,atomic->species.contracted_orbitals,FNAN,natn};
  (void)OpenMX_Mechanical_Run(&input,Stress_Operation,&args);
  dtime(&end);
  return end-begin;
}

static double Stress_owned(const OpenMXForceWorkspace *halo,OpenMXCoreContext *core,OpenMXThermodynamicEnergies *thermo, const OpenMXAtomicViews *atomic, const OpenMXVnaMatrixViews *vna, const OpenMXReciprocalViews *reciprocal, double *****H0,
	      double ******DS_NL,
	      double *****OLP,
	      double *****CDM,
	      double *****EDM, const OpenMXElectronicViews *electronic, const OpenMXLocalBasisViews *local_basis)
{
  OpenMXSolverInvocation *scratch=halo->scratch;

  double *****CntH0=electronic->CntH0;
  double *****CntOLP=electronic->CntOLP;
  double ******iDM=electronic->iDM;

  /*** variables **/

  int Nc,GNc,GRc,Cwan,s1,s2,BN_AB;
  int Mc_AN,Gc_AN,MNc,start_q_AN;
  double x,y,z,dx,dy,dz,tmp0,tmp1,tmp2,tmp3;
  double xx,r2,tot_den;
  double sumx,sumy,sumz,r,dege,pref;
  int i,j,k,l,Hwan,Qwan,so,p0,q,q0;
  int h_AN,Gh_AN,q_AN,Gq_AN;
  int ian,jan,kl,spin,spinmax,al,be,p;
  int tno0,tno1,tno2,Mh_AN,Mq_AN,n,num,size1,size2;
  int wanA,wanB,Gc_BN;
  int XC_P_switch;
  double time0;
  double dum,dge;
  double dEx,dEy,dEz;
  double Cxyz[4];
  double *Fx,*Fy,*Fz;
  OpenMXSolverComplexTensor Hx;
  OpenMXSolverComplexTensor Hy;
  OpenMXSolverComplexTensor Hz;
  double ***HUx;
  double ***HUy;
  double ***HUz;
  dcomplex ****NC_HUx;
  dcomplex ****NC_HUy;
  dcomplex ****NC_HUz;
  double **HVNAx;
  double **HVNAy;
  double **HVNAz;
  double *****CDM0;
  double *****iDM0;
  double *tmp_array;
  double *tmp_array2;
  double Re00x,Re00y,Re00z;
  double Re11x,Re11y,Re11z;
  double Re01x,Re01y,Re01z;
  double Im00x,Im00y,Im00z;
  double Im11x,Im11y,Im11z;
  double Im01x,Im01y,Im01z;
  int *Snd_CDM0_Size,*Rcv_CDM0_Size;
  int *Snd_iDM0_Size,*Rcv_iDM0_Size;
  double TStime,TEtime;
  int numprocs,myid,tag=999,ID,IDS,IDR;
  double Stime_atom, Etime_atom;
  /* for OpenMP */
  int OMPID,Nthrds,Nthrds0,Nprocs;
  double stime,etime;

  MPI_Status stat;
  MPI_Request request;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);
  MPI_Barrier(mpi_comm_level1);
  dtime(&TStime);

  /* Stress */
  OpenMXSolverRealTensor stress_threads;
  double Stress[9],My_Stress[9];
  double sumv1[4],sumv2[4],sumv3[4];
  double sumt1[9],sumt2[9],sumt3[9],sumt4[9],sumt5[9];
  int    Rn1,Rn2;
  double lx1,ly1,lz1,lx2,ly2,lz2,lx,ly,lz;
  
  /*allocation of array*/
  
  CDM0=halo->density_rows;iDM0=halo->imaginary_rows;
  Snd_CDM0_Size=OpenMX_Solver_Int(scratch,"stress:halo:send-counts",(size_t)numprocs);
  Rcv_CDM0_Size=OpenMX_Solver_Int(scratch,"stress:halo:receive-counts",(size_t)numprocs);
  if(halo->imaginary_rows){
    Snd_iDM0_Size=OpenMX_Solver_Int(scratch,"stress:halo:send-imaginary-counts",(size_t)numprocs);
    Rcv_iDM0_Size=OpenMX_Solver_Int(scratch,"stress:halo:receive-imaginary-counts",(size_t)numprocs);
  }

  /****************************************************
                      PrintMemory
  ****************************************************/

  /* work later */

  /****************************************************
    CDM to CDM0
  ****************************************************/

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
  
    Gc_AN = M2G[Mc_AN];
    Cwan = WhatSpecies[Gc_AN];
    tno1 = atomic->species.contracted_orbitals[Cwan];
  
    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
  
      Gh_AN = natn[Gc_AN][h_AN];
      Hwan = WhatSpecies[Gh_AN];
      tno2 = atomic->species.contracted_orbitals[Hwan];
  
      for (spin=0; spin<=SpinP_switch; spin++){
        for (i=0; i<tno1; i++){
	  for (j=0; j<tno2; j++){
            CDM0[spin][Mc_AN][h_AN][i][j] = CDM[spin][Mc_AN][h_AN][i][j]; 
	  }
        }
      }
    }
  }

  /****************************************************
    iDM to iDM0
  ****************************************************/

  if ( SO_switch==1 || (Hub_U_switch==1 && F_U_flag==1 && SpinP_switch==3) || Constraint_NCS_switch==1
       || Zeeman_NCS_switch==1 || Zeeman_NCO_switch==1){

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

      Gc_AN = M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];
      tno1 = atomic->species.contracted_orbitals[Cwan];

      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	Gh_AN = natn[Gc_AN][h_AN];
	Hwan = WhatSpecies[Gh_AN];
	tno2 = atomic->species.contracted_orbitals[Hwan];

	for (i=0; i<tno1; i++){
	  for (j=0; j<tno2; j++){
	    iDM0[0][Mc_AN][h_AN][i][j] = iDM[0][0][Mc_AN][h_AN][i][j]; 
	    iDM0[1][Mc_AN][h_AN][i][j] = iDM[0][1][Mc_AN][h_AN][i][j]; 
	  }
	}
      }
    }
  }

  /****************************************************
   MPI:

   CDM0
  ****************************************************/

  /***********************************
             set data size
  ************************************/

  for (ID=0; ID<numprocs; ID++){

    IDS = (myid + ID) % numprocs;
    IDR = (myid - ID + numprocs) % numprocs;

    if (ID!=0){
      tag = 999;

      /* find data size to send block data */
      if (F_Snd_Num[IDS]!=0){

        size1 = 0;
        for (spin=0; spin<=SpinP_switch; spin++){
          for (n=0; n<F_Snd_Num[IDS]; n++){
            Mc_AN = Snd_MAN[IDS][n];
            Gc_AN = Snd_GAN[IDS][n];
            Cwan = WhatSpecies[Gc_AN]; 
            tno1 = atomic->species.contracted_orbitals[Cwan];
            for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
              Gh_AN = natn[Gc_AN][h_AN];        
              Hwan = WhatSpecies[Gh_AN];
              tno2 = atomic->species.contracted_orbitals[Hwan];
              for (i=0; i<tno1; i++){
                for (j=0; j<tno2; j++){
                  size1++; 
                } 
              } 
	    }
          }
	}
 
        Snd_CDM0_Size[IDS] = size1;
        MPI_Isend(&size1, 1, MPI_INT, IDS, tag, mpi_comm_level1, &request);
      }
      else{
        Snd_CDM0_Size[IDS] = 0;
      }

      /* receiving of size of data */

      if (F_Rcv_Num[IDR]!=0){
        MPI_Recv(&size2, 1, MPI_INT, IDR, tag, mpi_comm_level1, &stat);
        Rcv_CDM0_Size[IDR] = size2;
      }
      else{
        Rcv_CDM0_Size[IDR] = 0;
      }

      if (F_Snd_Num[IDS]!=0) MPI_Wait(&request,&stat);

    }
    else{
      Snd_CDM0_Size[IDS] = 0;
      Rcv_CDM0_Size[IDR] = 0;
    }
  }

  /***********************************
             data transfer
  ************************************/

  tag = 999;
  for (ID=0; ID<numprocs; ID++){

    IDS = (myid + ID) % numprocs;
    IDR = (myid - ID + numprocs) % numprocs;

    if (ID!=0){

      /*****************************
              sending of data 
      *****************************/

      if (F_Snd_Num[IDS]!=0){

        size1 = Snd_CDM0_Size[IDS];

        /* allocation of array */

        tmp_array=OpenMX_Stress_HaloBuffer(scratch,"stress:halo:send",size1);

        /* multidimentional array to vector array */

        num = 0;
        for (spin=0; spin<=SpinP_switch; spin++){
          for (n=0; n<F_Snd_Num[IDS]; n++){
            Mc_AN = Snd_MAN[IDS][n];
            Gc_AN = Snd_GAN[IDS][n];
            Cwan = WhatSpecies[Gc_AN]; 
            tno1 = atomic->species.contracted_orbitals[Cwan];
            for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
              Gh_AN = natn[Gc_AN][h_AN];
              Hwan = WhatSpecies[Gh_AN];
              tno2 = atomic->species.contracted_orbitals[Hwan];
              for (i=0; i<tno1; i++){
                for (j=0; j<tno2; j++){
                  tmp_array[num] = CDM[spin][Mc_AN][h_AN][i][j];
                  num++;
                } 
              } 
	    }
          }
	}

        MPI_Isend(&tmp_array[0], size1, MPI_DOUBLE, IDS, tag, mpi_comm_level1, &request);

      }

      /*****************************
         receiving of block data
      *****************************/

      if (F_Rcv_Num[IDR]!=0){

        size2 = Rcv_CDM0_Size[IDR];
        
        /* allocation of array */
        tmp_array2=OpenMX_Stress_HaloBuffer(scratch,"stress:halo:receive",size2);
        
        MPI_Recv(&tmp_array2[0], size2, MPI_DOUBLE, IDR, tag, mpi_comm_level1, &stat);

        num = 0;
        for (spin=0; spin<=SpinP_switch; spin++){
          Mc_AN = F_TopMAN[IDR] - 1; 
          for (n=0; n<F_Rcv_Num[IDR]; n++){
            Mc_AN++;
            Gc_AN = Rcv_GAN[IDR][n];
            Cwan = WhatSpecies[Gc_AN]; 
            tno1 = atomic->species.contracted_orbitals[Cwan];

            for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
              Gh_AN = natn[Gc_AN][h_AN];        
              Hwan = WhatSpecies[Gh_AN];
              tno2 = atomic->species.contracted_orbitals[Hwan];
              for (i=0; i<tno1; i++){
                for (j=0; j<tno2; j++){
                  CDM0[spin][Mc_AN][h_AN][i][j] = tmp_array2[num];
                  num++;
		}
	      }
	    }
	  }        
	}

        /* freeing of array */

      }

      if (F_Snd_Num[IDS]!=0){
        MPI_Wait(&request,&stat);
         /* freeing of array */
      }

    }
  }

  /****************************************************
   MPI:

   iDM0
  ****************************************************/

  if ( SO_switch==1 || (Hub_U_switch==1 && F_U_flag==1 && SpinP_switch==3) || Constraint_NCS_switch==1
       || Zeeman_NCS_switch==1 || Zeeman_NCO_switch==1){
    /* not avairable so far */

    /***********************************
                set data size
    ************************************/

    for (ID=0; ID<numprocs; ID++){

      IDS = (myid + ID) % numprocs;
      IDR = (myid - ID + numprocs) % numprocs;

      if (ID!=0){
	tag = 999;

	/* find data size to send block data */
	if (F_Snd_Num[IDS]!=0){

	  size1 = 0;
	  for (so=0; so<2; so++){
	    for (n=0; n<F_Snd_Num[IDS]; n++){
	      Mc_AN = Snd_MAN[IDS][n];
	      Gc_AN = Snd_GAN[IDS][n];
	      Cwan = WhatSpecies[Gc_AN]; 
	      tno1 = atomic->species.contracted_orbitals[Cwan];
	      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
		Gh_AN = natn[Gc_AN][h_AN];        
		Hwan = WhatSpecies[Gh_AN];
		tno2 = atomic->species.contracted_orbitals[Hwan];
		for (i=0; i<tno1; i++){
		  for (j=0; j<tno2; j++){
		    size1++; 
		  } 
		} 
	      }
	    }
	  }
 
	  Snd_iDM0_Size[IDS] = size1;
	  MPI_Isend(&size1, 1, MPI_INT, IDS, tag, mpi_comm_level1, &request);

	}
	else{
	  Snd_iDM0_Size[IDS] = 0;
	}

	/* receiving of size of data */

	if (F_Rcv_Num[IDR]!=0){
	  MPI_Recv(&size2, 1, MPI_INT, IDR, tag, mpi_comm_level1, &stat);
	  Rcv_iDM0_Size[IDR] = size2;
	}
	else{
	  Rcv_iDM0_Size[IDR] = 0;
	}

	if (F_Snd_Num[IDS]!=0) MPI_Wait(&request,&stat);

      }
      else{
	Snd_iDM0_Size[IDS] = 0;
	Rcv_iDM0_Size[IDR] = 0;
      }
    }

    /***********************************
               data transfer
    ************************************/

    tag = 999;
    for (ID=0; ID<numprocs; ID++){

      IDS = (myid + ID) % numprocs;
      IDR = (myid - ID + numprocs) % numprocs;

      if (ID!=0){

	/*****************************
	      sending of data 
	*****************************/

	if (F_Snd_Num[IDS]!=0){

	  size1 = Snd_iDM0_Size[IDS];

	  /* allocation of array */

	  tmp_array=OpenMX_Stress_HaloBuffer(scratch,"stress:halo:send",size1);

	  /* multidimentional array to vector array */

	  num = 0;
	  for (so=0; so<2; so++){
	    for (n=0; n<F_Snd_Num[IDS]; n++){
	      Mc_AN = Snd_MAN[IDS][n];
	      Gc_AN = Snd_GAN[IDS][n];
	      Cwan = WhatSpecies[Gc_AN]; 
	      tno1 = atomic->species.contracted_orbitals[Cwan];
	      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
		Gh_AN = natn[Gc_AN][h_AN];
		Hwan = WhatSpecies[Gh_AN];
		tno2 = atomic->species.contracted_orbitals[Hwan];
		for (i=0; i<tno1; i++){
		  for (j=0; j<tno2; j++){
		    tmp_array[num] = iDM[0][so][Mc_AN][h_AN][i][j];
		    num++;
		  } 
		} 
	      }
	    }
	  }

	  MPI_Isend(&tmp_array[0], size1, MPI_DOUBLE, IDS, tag, mpi_comm_level1, &request);

	}

	/*****************************
          receiving of block data
	*****************************/

	if (F_Rcv_Num[IDR]!=0){

	  size2 = Rcv_iDM0_Size[IDR];
        
	  /* allocation of array */
	  tmp_array2=OpenMX_Stress_HaloBuffer(scratch,"stress:halo:receive",size2);
        
	  MPI_Recv(&tmp_array2[0], size2, MPI_DOUBLE, IDR, tag, mpi_comm_level1, &stat);

	  num = 0;
	  for (so=0; so<2; so++){
	    Mc_AN = F_TopMAN[IDR] - 1; 
	    for (n=0; n<F_Rcv_Num[IDR]; n++){
	      Mc_AN++;
	      Gc_AN = Rcv_GAN[IDR][n];
	      Cwan = WhatSpecies[Gc_AN]; 
	      tno1 = atomic->species.contracted_orbitals[Cwan];

	      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
		Gh_AN = natn[Gc_AN][h_AN];        
		Hwan = WhatSpecies[Gh_AN];
		tno2 = atomic->species.contracted_orbitals[Hwan];
		for (i=0; i<tno1; i++){
		  for (j=0; j<tno2; j++){
		    iDM0[so][Mc_AN][h_AN][i][j] = tmp_array2[num];
		    num++;
		  }
		}
	      }
	    }        
	  }

	  /* freeing of array */

	}

	if (F_Snd_Num[IDS]!=0){
	  MPI_Wait(&request,&stat);
	   /* freeing of array */
	}

      }
    }

  } /* if ( SO_switch==1 || (Hub_U_switch==1 && F_U_flag==1 && SpinP_switch==3) 
     || 1<=Constraint_NCS_switch || Zeeman_NCS_switch==1 || Zeeman_NCO_switch==1) */

  /* Initialization of Stress_Tensor */

  for (i=0; i<9; i++){
    Stress_Tensor[i] = 0.0;
  }

  /* Each stress component will be calculated below... */

  /****************************************************
                      #1 of Stress

              -\int \delta V_H drho_a/depsilon dr
                         and 
                force induced from PCC
              +\int V_XC drho_pcc/depsilon dr

  ****************************************************/

  /* begin(yshiihara) */
  {
    double aden,My_EXC;

    int spin;
    int Ng1,Ng2,Ng3,DN,BN;
    int XC_P_switch_stress;

    double **My_Stress_threads2;
    double Stress2[9];
    
    /* end(yshiihara) */
    
    if (myid==Host_ID && 0<level_stdout){
      printf("  Stress calculation #1\n");fflush(stdout);
    }

    dtime(&stime);

    /*********************************************************
      set RefVxc_Grid, where the CA-LDA exchange-correlation 
      functional is alway used.
    *********************************************************/

    XC_P_switch = 1;
    for (BN_AB=0; BN_AB<My_NumGridB_AB; BN_AB++){
      tot_den = ADensity_Grid_B[BN_AB] + ADensity_Grid_B[BN_AB];
      if (PCC_switch==1) {
	tot_den += PCCDensity_Grid_B[0][BN_AB] + PCCDensity_Grid_B[1][BN_AB];
      }
      RefVxc_Grid_B[BN_AB] = XC_Ceperly_Alder(tot_den,XC_P_switch);
    }
  
    Data_Grid_Copy_B2C_1( RefVxc_Grid_B,  RefVxc_Grid  );
    Data_Grid_Copy_B2C_1( dVHart_Grid_B,  dVHart_Grid  );
    Data_Grid_Copy_B2C_2( Vxc_Grid_B,     Vxc_Grid     );
    Data_Grid_Copy_B2C_2( Density_Grid_B, Density_Grid );

    My_EXC = 0.0;
    XC_P_switch = 0;
    for (BN=0; BN<My_NumGridB_AB; BN++){
      aden = 2.0*ADensity_Grid_B[BN];
      tmp1 = XC_Ceperly_Alder(aden,XC_P_switch);
      My_EXC += aden*tmp1;
    }
    
    /* begin(yshiihara) */ 

    /* get Nthrds0 */  
    /* Dynamic teams may grow after a separate probe region. */
    Nthrds0 = omp_get_max_threads();

    OpenMX_Stress_ThreadsPrepare(scratch,Nthrds0,&stress_threads);
    for (i=0; i<Nthrds0; i++){

      for (j=0; j<9; j++) openmx_solver_real_matrix_row(&stress_threads,i)[j] = 0.0;
    }

    for (i=0; i<9; i++){
      Stress[i]    = 0.0;
      My_Stress[i] = 0.0;
    }
    /* end(yshiihara) */
  
#pragma omp parallel shared(atomic,myid,time_per_atom,level_stdout,GridVol,Vxc_Grid,RefVxc_Grid,SpinP_switch,F_Vxc_flag,PCC_switch,dVHart_Grid,F_dVHart_flag,Gxyz,atv,MGridListAtom,CellListAtom,GridListAtom,GridN_Atom,WhatSpecies,M2G,Matomnum,Stress,My_Stress,stress_threads) private(OMPID,Nthrds,Nprocs,Mc_AN,Stime_atom,Etime_atom,Gc_AN,Cwan,sumx,sumy,sumz,Nc,GNc,GRc,MNc,Cxyz,x,y,z,dx,dy,dz,r,r2,tmp0,tmp1,tmp2,xx,sumt1,sumt2,sumt3,sumt4,sumt5,i)
    {
    
      /* get info. on OpenMP */ 

      OMPID = omp_get_thread_num();
      Nthrds = omp_get_num_threads();
      Nprocs = omp_get_num_procs();

      /* end(yshiihara)   */

      for (Mc_AN=(OMPID*Matomnum/Nthrds+1); Mc_AN<((OMPID+1)*Matomnum/Nthrds+1); Mc_AN++){

	dtime(&Stime_atom);

	Gc_AN = M2G[Mc_AN];
	Cwan = WhatSpecies[Gc_AN];

	sumx = 0.0;
	sumy = 0.0;
	sumz = 0.0;
	for (i=0; i<9; i++){
	  sumt1[i]    = 0.0;
	  sumt2[i]    = 0.0;
	  sumt3[i]    = 0.0;
	  sumt4[i]    = 0.0;
	  sumt5[i]    = 0.0;
	}

	for (Nc=0; Nc<GridN_Atom[Gc_AN]; Nc++){
	
	  GNc = GridListAtom[Mc_AN][Nc]; 
	  GRc = CellListAtom[Mc_AN][Nc];
	  MNc = MGridListAtom[Mc_AN][Nc]; 
	
	  Get_Grid_XYZ(GNc,Cxyz);
	  x = Cxyz[1] + atv[GRc][1]; 
	  y = Cxyz[2] + atv[GRc][2]; 
	  z = Cxyz[3] + atv[GRc][3];

	  dx = Gxyz[Gc_AN][1] - x;
	  dy = Gxyz[Gc_AN][2] - y;
	  dz = Gxyz[Gc_AN][3] - z;
	  r2 = dx*dx + dy*dy + dz*dz;
	  r = sqrt(r2);
	  xx = 0.5*log(r2);

	  /* for empty atoms */
	  if (r<1.0e-10) r = 1.0e-10;

	  if (1.0e-14<r){

	    tmp0 = Dr_KumoF( atomic->species.pao_mesh[Cwan], xx, r,
			     openmx_radial_row2(&atomic->pao_log_radii,Cwan), openmx_radial_row2(&atomic->pao_radii,Cwan), openmx_radial_row2(&atomic->density,Cwan));

	    tmp1 = dVHart_Grid[MNc]*tmp0/r*F_dVHart_flag;
	    /*begin(yshiihara) */ 
	    tmp2 = KumoF( atomic->species.pao_mesh[Cwan], xx,
			  openmx_radial_row2(&atomic->pao_log_radii,Cwan), openmx_radial_row2(&atomic->pao_radii,Cwan), openmx_radial_row2(&atomic->density,Cwan));
	    /*end(yshiihara) */ 
	    sumx += tmp1*dx;
	    sumy += tmp1*dy;
	    sumz += tmp1*dz;
	    /* begin(yshiihara) */ 
	    sumt1[0] += tmp1*dx*dx;
	    sumt1[1] += tmp1*dy*dx;
	    sumt1[2] += tmp1*dz*dx;
	    sumt1[3] += tmp1*dx*dy;
	    sumt1[4] += tmp1*dy*dy;
	    sumt1[5] += tmp1*dz*dy;
	    sumt1[6] += tmp1*dx*dz;
	    sumt1[7] += tmp1*dy*dz;
	    sumt1[8] += tmp1*dz*dz;
	    /* end(yshiihara) */

	    /* contribution of Exc^(0) */

	    /* begin(yshiihara) */

	    if (Exc0_correction_flag==1){

	      double exc0;
	      /*exc0 = XC_Ceperly_Alder(tmp2,0)*F_Vxc_flag;*/
	      /*tmp3 = RefVxc_Grid[MNc]*tmp2*F_Vxc_flag;*/
	      /*tmp3 = tmp2*exc0;*/

	      tmp1 = RefVxc_Grid[MNc]*tmp0/r*F_Vxc_flag;
	      sumx += tmp1*dx;
	      sumy += tmp1*dy;
	      sumz += tmp1*dz;
	      sumt2[0] += tmp1*dx*dx;
	      sumt2[1] += tmp1*dy*dx;
	      sumt2[2] += tmp1*dz*dx;
	      sumt2[3] += tmp1*dx*dy;
	      sumt2[4] += tmp1*dy*dy;
	      sumt2[5] += tmp1*dz*dy;
	      sumt2[6] += tmp1*dx*dz;
	      sumt2[7] += tmp1*dy*dz;
	      sumt2[8] += tmp1*dz*dz;
	    }
	    /* end(yshiihara) */

	    /* partial core correction */
	    if (PCC_switch==1){

	      tmp0 = 0.5*F_Vxc_flag*Dr_KumoF( atomic->species.vps_mesh[Cwan], xx, r,
					      openmx_radial_row2(&atomic->vps_log_radii,Cwan), openmx_radial_row2(&atomic->vps_radii,Cwan), openmx_radial_row2(&atomic->partial_core_density,Cwan));

	      if (SpinP_switch==0){
		tmp2 = 2.0*Vxc_Grid[0][MNc];
	      }
	      else {
		if (atomic->species.open_core[Cwan]==0){
		  tmp2 = Vxc_Grid[0][MNc] + Vxc_Grid[1][MNc];
		}
		else if (atomic->species.open_core[Cwan]==1){
		  tmp2 = 2.0*Vxc_Grid[0][MNc];
		}
		else if (atomic->species.open_core[Cwan]==-1){
		  tmp2 = 2.0*Vxc_Grid[1][MNc];
		}
	      }

	      tmp1 = tmp2*tmp0/r;
	      sumx -= tmp1*dx;
	      sumy -= tmp1*dy;
	      sumz -= tmp1*dz;

	      /* begin(yshiihara) */
	      sumt3[0] -= tmp1*dx*dx;
	      sumt3[1] -= tmp1*dy*dx;
	      sumt3[2] -= tmp1*dz*dx;
	      sumt3[3] -= tmp1*dx*dy;
	      sumt3[4] -= tmp1*dy*dy;
	      sumt3[5] -= tmp1*dz*dy;
	      sumt3[6] -= tmp1*dx*dz;
	      sumt3[7] -= tmp1*dy*dz;
	      sumt3[8] -= tmp1*dz*dz;
	      /* end(yshiihara) */

	      /* contribution of Exc^(0) */
  
  	      if (Exc0_correction_flag==1){

		tmp2 = 2.0*RefVxc_Grid[MNc];
		tmp1 = tmp2*tmp0/r;
		sumx += tmp1*dx;
		sumy += tmp1*dy;
		sumz += tmp1*dz;

		/* begin(yshiihara) */
		sumt4[0] += tmp1*dx*dx;
		sumt4[1] += tmp1*dy*dx;
		sumt4[2] += tmp1*dz*dx;
		sumt4[3] += tmp1*dx*dy;
		sumt4[4] += tmp1*dy*dy;
		sumt4[5] += tmp1*dz*dy;
		sumt4[6] += tmp1*dx*dz;
		sumt4[7] += tmp1*dy*dz;
		sumt4[8] += tmp1*dz*dz;
		/* end(yshiihara) */
	      }

	    }
	  }
	} /* Nc */

	/* begin(yshiihara) */

	/*
	printf(" sumt1 : \n");fflush(stdout);
	for (i = 0; i<3; i++){
	  for (j = 0; j<3; j++){
	    printf("%16.8f ", sumt1[3*i+j]);
	  }
	  printf("\n");fflush(stdout);
	}
	printf(" sumt2 : \n");fflush(stdout);
	for (i = 0; i<3; i++){
	  for (j = 0; j<3; j++){
	    printf("%16.8f ", sumt2[3*i+j]);
	  }
	  printf("\n");fflush(stdout);
	}
	printf(" sumt3 : \n");fflush(stdout);
	for (i = 0; i<3; i++){
	  for (j = 0; j<3; j++){
	    printf("%16.8f ", sumt3[3*i+j]);
	  }
	  printf("\n");fflush(stdout);
	}
	printf(" sumt4 : \n");fflush(stdout);
	for (i = 0; i<3; i++){
	  for (j = 0; j<3; j++){
	    printf("%16.8f ", sumt4[3*i+j]);
	  }
	  printf("\n");fflush(stdout);
	}
	*/

	/* end(yshiihara) */

	/* begin(yshiihara) */
	openmx_solver_real_matrix_row(&stress_threads,OMPID)[0] -= (sumt1[0]+sumt2[0]+sumt3[0]+sumt4[0])*GridVol;
	openmx_solver_real_matrix_row(&stress_threads,OMPID)[1] -= (sumt1[1]+sumt2[1]+sumt3[1]+sumt4[1])*GridVol;
	openmx_solver_real_matrix_row(&stress_threads,OMPID)[2] -= (sumt1[2]+sumt2[2]+sumt3[2]+sumt4[2])*GridVol;
	openmx_solver_real_matrix_row(&stress_threads,OMPID)[3] -= (sumt1[3]+sumt2[3]+sumt3[3]+sumt4[3])*GridVol;
	openmx_solver_real_matrix_row(&stress_threads,OMPID)[4] -= (sumt1[4]+sumt2[4]+sumt3[4]+sumt4[4])*GridVol;
	openmx_solver_real_matrix_row(&stress_threads,OMPID)[5] -= (sumt1[5]+sumt2[5]+sumt3[5]+sumt4[5])*GridVol;
	openmx_solver_real_matrix_row(&stress_threads,OMPID)[6] -= (sumt1[6]+sumt2[6]+sumt3[6]+sumt4[6])*GridVol;
	openmx_solver_real_matrix_row(&stress_threads,OMPID)[7] -= (sumt1[7]+sumt2[7]+sumt3[7]+sumt4[7])*GridVol;
	openmx_solver_real_matrix_row(&stress_threads,OMPID)[8] -= (sumt1[8]+sumt2[8]+sumt3[8]+sumt4[8])*GridVol;
	/* end(yshiihara) */

	dtime(&Etime_atom);
	time_per_atom[Gc_AN] += Etime_atom - Stime_atom;
      
      } /* Mc_AN */

    } /* #pragma omp parallel */
  
    dtime(&etime);
    if(myid==0 && measure_time){
      printf("Time for stress#1=%18.5f\n",etime-stime);fflush(stdout);
    }
  
    /* begin(yshiihara) */
    for(i=0; i<Nthrds0; i++){
      for(j=0; j<9; j++){
	My_Stress[j] += openmx_solver_real_matrix_row(&stress_threads,i)[j];
      }
    }

    MPI_Allreduce(&My_Stress[0],&Stress[0],9,MPI_DOUBLE,MPI_SUM,mpi_comm_level1);

    /* add #1 of stress to Stress_Tensor */

    for (i=0; i<9; i++){
      Stress_Tensor[i] += Stress[i];
    }

    /* show #1 of stress */
  
    if (myid==Host_ID && 1<level_stdout){

      printf(" Components of stress #1: \n");fflush(stdout);
      for (i = 0; i<3; i++){
	for (j = 0; j<3; j++){
	  printf("%16.8f ", Stress[3*i+j]);
	}
	printf("\n");fflush(stdout);
      }
    }
    
    /* end(yshiihara) */
  
    if (ESM_switch!=0){
      if (myid==Host_ID && 0<level_stdout){
	printf("yshiihara: ESM is not currently supported in Stress.c \n");fflush(stdout);
      }
    }

    /* begin(yshiihara) */ 

    /* end(yshiihara) */

    /* begin(yshiihara) */ 
  }
  /* end(yshiihara) */

  /****************************************************
   contraction

   H0
   OLP
  ****************************************************/

  MPI_Barrier(mpi_comm_level1);
  
  if (Cnt_switch==1){

    Cont_Matrix0(H0[0],CntH0[0], local_basis);
    Cont_Matrix0(H0[1],CntH0[1], local_basis);
    Cont_Matrix0(H0[2],CntH0[2], local_basis);
    Cont_Matrix0(H0[3],CntH0[3], local_basis);

    Cont_Matrix0(OLP[0],CntOLP[0], local_basis);
    Cont_Matrix0(OLP[1],CntOLP[1], local_basis);
    Cont_Matrix0(OLP[2],CntOLP[2], local_basis);
    Cont_Matrix0(OLP[3],CntOLP[3], local_basis);
  }

  if ( Hub_U_switch==1 && Hub_U_occupation==1 ){
    if (myid==Host_ID && 0<level_stdout){
      printf(" LDA+U is not currently supported in Stress.c \n");fflush(stdout);
    }
    MPI_OLP(scratch,OLP);
  }

  MPI_Barrier(mpi_comm_level1);

  /* begin(yshiihara) */
  /* yshiihara: calculation on Lebedev grid performed in Total_energy.c */
  /********************************************************
    calculation of Exc^(0) and its contribution 
    to forces on the fine mesh
 
    Note that there is no stress coming from volume term.
  ********************************************************/

  if (Exc0_correction_flag==1){

    double **Leb_Grid_XYZW;

    OpenMXSolverRealTensor lebedev;
    OpenMX_XC_LebedevPrepare(scratch,Num_Leb_Grid,&lebedev,&Leb_Grid_XYZW);

    Set_Lebedev_Grid(Num_Leb_Grid,Leb_Grid_XYZW);

    /* get Nthrds0 */  
    /* Dynamic teams may grow after a separate probe region. */
    Nthrds0 = omp_get_max_threads();

    /* initialize the temporal array storing the force contribution */

    for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
      Gxyz[Gc_AN][41] = 0.0;
      Gxyz[Gc_AN][42] = 0.0;
      Gxyz[Gc_AN][43] = 0.0;
    }

    /* begin(yshiihara) */
    {

      double exc0,vxc0,rs,sum,sumr,sumt,re,Sr,Dr,sumtx,sumty,sumtz,den,den0,gden0,w;
      double x0,y0,z0,x1,y1,z1,dx1,dy1,dz1,r1;
      double *My_sumr;
      OpenMXSolverRealTensor xc_partials;
      size_t max_neighbors=0;
      for(int atom=1;atom<=Matomnum;atom++){
        size_t count=(size_t)FNAN[M2G[atom]]+1;
        if(count>max_neighbors)max_neighbors=count;
      }
      max_neighbors=OpenMX_Solver_SimdColumns(max_neighbors);
      OpenMX_XC_PartialsPrepare(scratch,"stress:xc-partials",Nthrds0,max_neighbors,&xc_partials);
      My_sumr=OpenMX_Solver_Real(scratch,"stress:xc-sums",(size_t)Nthrds0);
      int Nloop,Rn,ir,ia;

      /* start calc. */

      for (i=0; i<9; i++){
	My_Stress[i] = 0.0;
	Stress[i]    = 0.0;
      }

      rs = 0.0;
      sum = 0.0;

      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

	Gc_AN = M2G[Mc_AN];
	Cwan = WhatSpecies[Gc_AN];
	re = atomic->species.cutoff[Cwan];
	Sr = re + rs;
	Dr = re - rs;

	/* allocation of arrays */

	for (i=0; i<Nthrds0; i++) My_sumr[i]  = 0.0;


	for (i=0; i<Nthrds0; i++){

	  for (j=0; j<(FNAN[Gc_AN]+1); j++){
	    OpenMX_XC_ForceRow(&xc_partials,0,i)[j] = 0.0;
	  }
	}


	for (i=0; i<Nthrds0; i++){

	  for (j=0; j<(FNAN[Gc_AN]+1); j++){
	    OpenMX_XC_ForceRow(&xc_partials,1,i)[j] = 0.0;
	  }
	}


	for (i=0; i<Nthrds0; i++){

	  for (j=0; j<(FNAN[Gc_AN]+1); j++){
	    OpenMX_XC_ForceRow(&xc_partials,2,i)[j] = 0.0;
	  }
	}

	OpenMX_Stress_ThreadsPrepare(scratch,Nthrds0,&stress_threads);
	for (Nloop=0; Nloop<Nthrds0; Nloop++){

	  for (i=0; i<9; i++){
	    openmx_solver_real_matrix_row(&stress_threads,Nloop)[i] = 0.0;
	  }
	}
	
#pragma omp parallel shared(scratch,max_neighbors,xc_partials,atomic,lebedev,Num_Leb_Grid,My_sumr,My_Stress,stress_threads,Dr,Sr,CoarseGL_Abscissae,CoarseGL_Weight,Gxyz,Gc_AN,FNAN,natn,ncn,WhatSpecies,atv,F_Vxc_flag,Cwan,PCC_switch) private(OMPID,Nthrds,Nprocs,ir,ia,r,w,sumt,sumtx,sumty,sumtz,x,x0,y0,z0,h_AN,Gh_AN,Rn,Hwan,x1,y1,z1,dx,dy,dz,r1,den,den0,gden0,dx1,dy1,dz1,exc0,vxc0,sumt1,i)
	{

	  double dx2,dy2,dz2;
	  double *gx,*gy,*gz,dexc0;
	  double *sum_gx,*sum_gy,*sum_gz;

	  /* get info. on OpenMP */ 

	  OMPID = omp_get_thread_num();
          OpenMXSolverRealTensor derivatives;
          OpenMX_XC_WorkerPrepare(scratch,"stress:xc-derivatives",max_neighbors,&derivatives);
          gx=openmx_solver_real_matrix_row(&derivatives,0);
          gy=openmx_solver_real_matrix_row(&derivatives,1);
          gz=openmx_solver_real_matrix_row(&derivatives,2);
          sum_gx=openmx_solver_real_matrix_row(&derivatives,3);
          sum_gy=openmx_solver_real_matrix_row(&derivatives,4);
          sum_gz=openmx_solver_real_matrix_row(&derivatives,5);
          double *partial_x=OpenMX_XC_ForceRow(&xc_partials,0,OMPID);
          double *partial_y=OpenMX_XC_ForceRow(&xc_partials,1,OMPID);
          double *partial_z=OpenMX_XC_ForceRow(&xc_partials,2,OMPID);

	  Nthrds = omp_get_num_threads();
	  Nprocs = omp_get_num_procs();

	  for (ir=(OMPID*CoarseGL_Mesh/Nthrds); ir<((OMPID+1)*CoarseGL_Mesh/Nthrds); ir++){

	    r = 0.50*(Dr*CoarseGL_Abscissae[ir] + Sr);
	    sumt  = 0.0; 

	    for (i=0; i<(FNAN[Gc_AN]+1); i++){
	      sum_gx[i] = 0.0;
	      sum_gy[i] = 0.0;
	      sum_gz[i] = 0.0;
	    }

	    for (i=0; i<9; i++) sumt1[i] = 0.0;

	    for (ia=0; ia<Num_Leb_Grid; ia++){

	      x0 = r*openmx_solver_real_matrix_row(&lebedev,ia)[0] + Gxyz[Gc_AN][1];
	      y0 = r*openmx_solver_real_matrix_row(&lebedev,ia)[1] + Gxyz[Gc_AN][2];
	      z0 = r*openmx_solver_real_matrix_row(&lebedev,ia)[2] + Gxyz[Gc_AN][3];

	      /* calculate rho_atom + rho_pcc */ 

	      den = 0.0;

	      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

		Gh_AN = natn[Gc_AN][h_AN];
		Rn = ncn[Gc_AN][h_AN]; 
		Hwan = WhatSpecies[Gh_AN];

		x1 = Gxyz[Gh_AN][1] + atv[Rn][1];
		y1 = Gxyz[Gh_AN][2] + atv[Rn][2];
		z1 = Gxyz[Gh_AN][3] + atv[Rn][3];
            
		dx = x1 - x0;
		dy = y1 - y0;
		dz = z1 - z0;

		x = 0.5*log(dx*dx + dy*dy + dz*dz);

		/* calculate density */

		den += KumoF( atomic->species.pao_mesh[Hwan], x,
			      openmx_radial_row2(&atomic->pao_log_radii,Hwan), openmx_radial_row2(&atomic->pao_radii,Hwan), openmx_radial_row2(&atomic->density_backup,Hwan))*F_Vxc_flag;

		if (h_AN==0) den0 = den;

		/* calculate gradient of density */

		if (h_AN!=0){
		  r1 = sqrt(dx*dx + dy*dy + dz*dz);
		  gden0 = Dr_KumoF( atomic->species.pao_mesh[Hwan], x, r1,
				    openmx_radial_row2(&atomic->pao_log_radii,Hwan), openmx_radial_row2(&atomic->pao_radii,Hwan), openmx_radial_row2(&atomic->density_backup,Hwan))*F_Vxc_flag;

		  gx[h_AN] = gden0/r1*dx;
		  gy[h_AN] = gden0/r1*dy;
		  gz[h_AN] = gden0/r1*dz;
		}

	      } /* h_AN */

	      /* calculate the CA-LDA exchange-correlation energy density */
	      exc0 = XC_Ceperly_Alder(den,0);

	      /* calculate the CA-LDA exchange-correlation potential */
	      dexc0 = XC_Ceperly_Alder(den,3);

	      /* Lebedev quadrature */

	      w = openmx_solver_real_matrix_row(&lebedev,ia)[3];
	      sumt += w*den0*exc0;

	      for (h_AN=1; h_AN<=FNAN[Gc_AN]; h_AN++){
              
		dx1 = w*den0*dexc0*gx[h_AN]; 
		dy1 = w*den0*dexc0*gy[h_AN]; 
		dz1 = w*den0*dexc0*gz[h_AN]; 

		sum_gx[h_AN] += dx1;
		sum_gy[h_AN] += dy1;
		sum_gz[h_AN] += dz1;

		Gh_AN = natn[Gc_AN][h_AN];
		Rn = ncn[Gc_AN][h_AN]; 
              
		dx2 = Gxyz[Gh_AN][1] + atv[Rn][1] - Gxyz[Gc_AN][1];
		dy2 = Gxyz[Gh_AN][2] + atv[Rn][2] - Gxyz[Gc_AN][2];
		dz2 = Gxyz[Gh_AN][3] + atv[Rn][3] - Gxyz[Gc_AN][3];

		sumt1[0] += dx1*dx2;
		sumt1[1] += dx1*dy2;
		sumt1[2] += dx1*dz2;
		sumt1[3] += dy1*dx2;
		sumt1[4] += dy1*dy2;
		sumt1[5] += dy1*dz2;
		sumt1[6] += dz1*dx2;
		sumt1[7] += dz1*dy2;
		sumt1[8] += dz1*dz2;

	      } /* h_AN */

	    } /* ia */

	    /* r for Gauss-Legendre quadrature */

	    w = r*r*CoarseGL_Weight[ir]; 
	    My_sumr[OMPID]  += w*sumt;

	    for (h_AN=1; h_AN<=FNAN[Gc_AN]; h_AN++){
	      partial_x[h_AN] += w*sum_gx[h_AN];
	      partial_y[h_AN] += w*sum_gy[h_AN];
	      partial_z[h_AN] += w*sum_gz[h_AN];
	    }

	    for(i=0; i<9; i++){
	      openmx_solver_real_matrix_row(&stress_threads,OMPID)[i] += w*sumt1[i];
	    }

	  } /* ir */

	} /* #pragma omp */

	sumr = 0.0;
	for (Nloop=0; Nloop<Nthrds0; Nloop++){
	  sumr += My_sumr[Nloop];
	}
	sum += 2.0*PI*Dr*sumr;

	/* add force */

	for (h_AN=1; h_AN<=FNAN[Gc_AN]; h_AN++){

	  double sumrx,sumry,sumrz;

	  sumrx = 0.0;
	  sumry = 0.0;
	  sumrz = 0.0;
	  for (Nloop=0; Nloop<Nthrds0; Nloop++){
	    sumrx += OpenMX_XC_ForceRow(&xc_partials,0,Nloop)[h_AN];
	    sumry += OpenMX_XC_ForceRow(&xc_partials,1,Nloop)[h_AN];
	    sumrz += OpenMX_XC_ForceRow(&xc_partials,2,Nloop)[h_AN];
	  }

	  Gh_AN = natn[Gc_AN][h_AN];

	  Gxyz[Gh_AN][41] += 2.0*PI*Dr*sumrx;
	  Gxyz[Gh_AN][42] += 2.0*PI*Dr*sumry;
	  Gxyz[Gh_AN][43] += 2.0*PI*Dr*sumrz;

	  Gxyz[Gc_AN][41] -= 2.0*PI*Dr*sumrx;
	  Gxyz[Gc_AN][42] -= 2.0*PI*Dr*sumry;
	  Gxyz[Gc_AN][43] -= 2.0*PI*Dr*sumrz;
	}

	/* add stress */
      
	for (Nloop=0; Nloop<Nthrds0; Nloop++){
	  for(i=0; i<9; i++){
	    My_Stress[i] += 2.0*PI*Dr*openmx_solver_real_matrix_row(&stress_threads,Nloop)[i];
	  }
	}

	/* freeing of arrays */

      } /* Mc_AN */

      /****************************************************
                       MPI, My_Stress
      ****************************************************/

      MPI_Allreduce(&My_Stress[0],&Stress[0],9,MPI_DOUBLE,MPI_SUM,mpi_comm_level1);

      /* add components of stress (Lebedev) to Stress_Tensor */

      for (i=0; i<9; i++){
	Stress_Tensor[i] += Stress[i];
      }

      /* show stress (Lebedev) */

      if (myid==Host_ID && 1<level_stdout){
	printf(" Components of stress (Lebedev): \n");fflush(stdout);
	for (i = 0; i<3; i++){
	  for (j = 0; j<3; j++){
	    printf("%16.8f ", Stress[3*i+j]);
	  }
	  printf("\n");fflush(stdout);
	}
      }

    }
    /* end(yshiihara) */

  } /* end of if (Exc0_correction_flag==1) */

  /****************************************************
                      #2 of Stress

                    kinetic operator 
  ****************************************************/

  dtime(&stime);

  if (myid==Host_ID && 0<level_stdout){
    printf("  Stress calculation #2\n");fflush(stdout);
  }
  
  /* Dynamic teams may grow after a separate probe region. */
  Nthrds0 = omp_get_max_threads();

  /* begin(yshiihara) */ 
  OpenMX_Stress_ThreadsPrepare(scratch,Nthrds0,&stress_threads);
  for (i=0; i<Nthrds0; i++){

    for (j=0; j<9; j++) openmx_solver_real_matrix_row(&stress_threads,i)[j] = 0.0;
  }
  /* end(yshiihara) */
  
#pragma omp parallel shared(scratch,atomic,time_per_atom,Gxyz,myid,level_stdout,iDM0,CDM0,CntH0,H0,F_Kin_flag,local_basis,OLP,Hub_U_occupation,Cnt_switch,F_NL_flag,List_YOUSO,RMI1,Zeeman_NCO_switch,Zeeman_NCS_switch,Constraint_NCS_switch,F_U_flag,Hub_U_switch,SO_switch,SpinP_switch,F_G2M,natn,ncn,FNAN,WhatSpecies,M2G,Matomnum,stress_threads) private(OMPID,Nthrds,Nprocs,Mc_AN,Stime_atom,Etime_atom,Gc_AN,Cwan,dEx,dEy,dEz,h_AN,Gh_AN,Mh_AN,Hwan,ian,start_q_AN,q_AN,Gq_AN,Mq_AN,Qwan,jan,kl,so,i,j,k,l,Hx,Hy,Hz,HUx,HUy,HUz,NC_HUx,NC_HUy,NC_HUz,s1,s2,pref,spinmax,spin,lx1,ly1,lz1,lx2,ly2,lz2,lx,ly,lz,Rn1,Rn2)
  {

    OpenMX_Stress_KineticPrepare(scratch,"stress:kinetic:x",List_YOUSO[7],&Hx);

    OpenMX_Stress_KineticPrepare(scratch,"stress:kinetic:y",List_YOUSO[7],&Hy);

    OpenMX_Stress_KineticPrepare(scratch,"stress:kinetic:z",List_YOUSO[7],&Hz);
    
    /* get info. on OpenMP */ 

    OMPID = omp_get_thread_num();
    Nthrds = omp_get_num_threads();
    Nprocs = omp_get_num_procs();

    /*debug*/
    /*printf("OpenMP myid: %d, OPMID: %d, Nthrds: %d, Nprocs: %d \n",myid,OMPID,Nthrds,Nprocs);fflush(stdout);*/ 
    
    for (Mc_AN=(OMPID*Matomnum/Nthrds+1); Mc_AN<((OMPID+1)*Matomnum/Nthrds+1); Mc_AN++){
      
      dtime(&Stime_atom);
      
      Gc_AN = M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];
  
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	Gh_AN = natn[Gc_AN][h_AN];
	Mh_AN = F_G2M[Gh_AN];
	Hwan = WhatSpecies[Gh_AN];
	ian = atomic->species.contracted_orbitals[Hwan];
	/*stress*/
	Rn1 = ncn[Gc_AN][h_AN];
	lx1 = Gxyz[Gc_AN][1] - Gxyz[Gh_AN][1] - atv[Rn1][1];
	ly1 = Gxyz[Gc_AN][2] - Gxyz[Gh_AN][2] - atv[Rn1][2];
	lz1 = Gxyz[Gc_AN][3] - Gxyz[Gh_AN][3] - atv[Rn1][3];

	/*debug*/
	/*printf("Gc_AN: %d, Gh_AN: %d, lx1: %13.8f, ly1: %13.8f, lz1: %13.8f \n",Gc_AN,Gh_AN,lx1,ly1,lz1); */
	
	if ( SpinP_switch==3 && (SO_switch==1 || (Hub_U_switch==1 && F_U_flag==1)
	 || Constraint_NCS_switch==1 || Zeeman_NCS_switch==1 || Zeeman_NCO_switch==1) )
	  start_q_AN = 0;
	else 
	  start_q_AN = h_AN;

	for (q_AN=start_q_AN; q_AN<=FNAN[Gc_AN]; q_AN++){
	    
	  Gq_AN = natn[Gc_AN][q_AN];
	  Mq_AN = F_G2M[Gq_AN];
	  Qwan = WhatSpecies[Gq_AN];
	  jan = atomic->species.contracted_orbitals[Qwan];
	  kl = RMI1[Mc_AN][h_AN][q_AN];
	  /*stress*/
	  Rn2 = ncn[Gc_AN][q_AN];
	  lx2 = Gxyz[Gc_AN][1] - Gxyz[Gq_AN][1] - atv[Rn2][1];
	  ly2 = Gxyz[Gc_AN][2] - Gxyz[Gq_AN][2] - atv[Rn2][2];
	  lz2 = Gxyz[Gc_AN][3] - Gxyz[Gq_AN][3] - atv[Rn2][3];

	  /*debug*/
	  /*printf("Gc_AN: %d, Gq_AN: %d, lx2: %13.8f, ly2: %13.8f, lz2: %13.8f \n",Gc_AN,Gq_AN,lx2,ly2,lz2); */
	  	  
	  if (0<=kl){
	      
	    for (so=0; so<3; so++){
	      for (i=0; i<List_YOUSO[7]; i++){
		for (j=0; j<List_YOUSO[7]; j++){
		  openmx_solver_complex_plane_row(&Hx,so,i)[j] = Complex(0.0,0.0);
		  openmx_solver_complex_plane_row(&Hy,so,i)[j] = Complex(0.0,0.0);
		  openmx_solver_complex_plane_row(&Hz,so,i)[j] = Complex(0.0,0.0);
		}
	      }
	    }
	      
	    /****************************************************
                               H0 = dKinetic
	    ****************************************************/

	    lx = 0.0;
	    ly = 0.0;
	    lz = 0.0;

	    if (F_Kin_flag==1){  

	      /* in case of no orbital optimization */
	      
	      if (Cnt_switch==0){
		if (h_AN==0){
		  for (i=0; i<ian; i++){
		    for (j=0; j<jan; j++){
		      openmx_solver_complex_plane_row(&Hx,0,i)[j].r += H0[1][Mc_AN][q_AN][i][j];
		      openmx_solver_complex_plane_row(&Hy,0,i)[j].r += H0[2][Mc_AN][q_AN][i][j];
		      openmx_solver_complex_plane_row(&Hz,0,i)[j].r += H0[3][Mc_AN][q_AN][i][j];
		      
		      openmx_solver_complex_plane_row(&Hx,1,i)[j].r += H0[1][Mc_AN][q_AN][i][j];
		      openmx_solver_complex_plane_row(&Hy,1,i)[j].r += H0[2][Mc_AN][q_AN][i][j];
		      openmx_solver_complex_plane_row(&Hz,1,i)[j].r += H0[3][Mc_AN][q_AN][i][j];
		    }
		  }
		  lx = lx2;
		  ly = ly2;
		  lz = lz2;
		}

		else if (h_AN!=0 && q_AN==0){
		  for (i=0; i<ian; i++){
		    for (j=0; j<jan; j++){
		      openmx_solver_complex_plane_row(&Hx,0,i)[j].r += H0[1][Mc_AN][h_AN][j][i];
		      openmx_solver_complex_plane_row(&Hy,0,i)[j].r += H0[2][Mc_AN][h_AN][j][i];
		      openmx_solver_complex_plane_row(&Hz,0,i)[j].r += H0[3][Mc_AN][h_AN][j][i];

		      openmx_solver_complex_plane_row(&Hx,1,i)[j].r += H0[1][Mc_AN][h_AN][j][i];
		      openmx_solver_complex_plane_row(&Hy,1,i)[j].r += H0[2][Mc_AN][h_AN][j][i];
		      openmx_solver_complex_plane_row(&Hz,1,i)[j].r += H0[3][Mc_AN][h_AN][j][i];
		    }
		  }
		  lx = lx1;
		  ly = ly1;
		  lz = lz1;
		}
	      }
	      
	      /* in case of orbital optimization */
	      
	      else{

		if (h_AN==0){
		  for (i=0; i<ian; i++){
		    for (j=0; j<jan; j++){

		      openmx_solver_complex_plane_row(&Hx,0,i)[j].r += CntH0[1][Mc_AN][q_AN][i][j];
		      openmx_solver_complex_plane_row(&Hy,0,i)[j].r += CntH0[2][Mc_AN][q_AN][i][j];
		      openmx_solver_complex_plane_row(&Hz,0,i)[j].r += CntH0[3][Mc_AN][q_AN][i][j];

		      openmx_solver_complex_plane_row(&Hx,1,i)[j].r += CntH0[1][Mc_AN][q_AN][i][j];
		      openmx_solver_complex_plane_row(&Hy,1,i)[j].r += CntH0[2][Mc_AN][q_AN][i][j];
		      openmx_solver_complex_plane_row(&Hz,1,i)[j].r += CntH0[3][Mc_AN][q_AN][i][j];

		    }
		  }
		  lx = lx2;
		  ly = ly2;
		  lz = lz2;
		} 

		else if (h_AN!=0 && q_AN==0){
		  for (i=0; i<ian; i++){
		    for (j=0; j<jan; j++){

		      openmx_solver_complex_plane_row(&Hx,0,i)[j].r += CntH0[1][Mc_AN][h_AN][j][i];
		      openmx_solver_complex_plane_row(&Hy,0,i)[j].r += CntH0[2][Mc_AN][h_AN][j][i];
		      openmx_solver_complex_plane_row(&Hz,0,i)[j].r += CntH0[3][Mc_AN][h_AN][j][i];

		      openmx_solver_complex_plane_row(&Hx,1,i)[j].r += CntH0[1][Mc_AN][h_AN][j][i];
		      openmx_solver_complex_plane_row(&Hy,1,i)[j].r += CntH0[2][Mc_AN][h_AN][j][i];
		      openmx_solver_complex_plane_row(&Hz,1,i)[j].r += CntH0[3][Mc_AN][h_AN][j][i];

		    }
		  }
		  lx = lx1;
		  ly = ly1;
		  lz = lz1;
		}
	      }
 
	    } /* if F_Kin_flag */

	    /****************************************************
                                \sum rho*dH
	    ****************************************************/

            /* non-spin polarization */

	    if (SpinP_switch==0){

	      if (q_AN==h_AN) pref = 2.0;
	      else            pref = 4.0; 

	      /* Stress */
	      dEx = 0.0;
	      dEy = 0.0;
	      dEz = 0.0;
	      
	      for (i=0; i<atomic->species.contracted_orbitals[Hwan]; i++){
		for (j=0; j<atomic->species.contracted_orbitals[Qwan]; j++){
		  dEx += pref*CDM0[0][Mh_AN][kl][i][j]*openmx_solver_complex_plane_row(&Hx,0,i)[j].r;
		  dEy += pref*CDM0[0][Mh_AN][kl][i][j]*openmx_solver_complex_plane_row(&Hy,0,i)[j].r;
		  dEz += pref*CDM0[0][Mh_AN][kl][i][j]*openmx_solver_complex_plane_row(&Hz,0,i)[j].r;
		}
	      }

              /* debug */   

	      /*
	      if(fabs(dEx)>1.e-2 || fabs(dEy)>1.e-2 || fabs(dEz)>1.e-2) { 
		printf("Gh_AN: %d, Gc_AN: %d, Gq_AN: %d \n",Gh_AN,Gc_AN,Gq_AN); 
		printf("dEx: %13.8f, dEy: %13.8f, dEz: %13.8f, lx: %13.8f, ly: %13.8f, lz: %13.8f \n",
                        dEx,dEy,dEz,lx,ly,lz);fflush(stdout); 
		printf("lx1: %13.8f, ly1: %13.8f, lz1: %13.8f, lx2: %13.8f, ly2: %13.8f, lz2: %13.8f \n",
                        lx1,ly1,lz1,lx2,ly2,lz2);fflush(stdout); 
		printf("Gc_AN x: %13.8f, y: %13.8f, z: %13.8f  \n",Gxyz[Gc_AN][1],Gxyz[Gc_AN][2],Gxyz[Gc_AN][3]);
		printf("Gh_AN x: %13.8f, y: %13.8f, z: %13.8f, atv 1: %13.8f, 2: %13.8f, 3: %13.8f  \n",
                        Gxyz[Gh_AN][1],Gxyz[Gh_AN][2],Gxyz[Gh_AN][3],atv[Rn1][1],atv[Rn1][2],atv[Rn1][3]); 
		printf("Gq_AN x: %13.8f, y: %13.8f, z: %13.8f, atv 1: %13.8f, 2: %13.8f, 3: %13.8f  \n",
                        Gxyz[Gq_AN][1],Gxyz[Gq_AN][2],Gxyz[Gq_AN][3],atv[Rn2][1],atv[Rn2][2],atv[Rn2][3]); 
	      }
	      */

	      /*
	      if(fabs(dEx)>1.e-2 || fabs(dEy)>1.e-2 || fabs(dEz)>1.e-2) { 
		printf("ABC1 Gh_AN: %d, Gc_AN: %d, Gq_AN: %d \n",Gh_AN,Gc_AN,Gq_AN); 
		printf("dEx: %13.8f, dEy: %13.8f, dEz: %13.8f, lx: %13.8f, ly: %13.8f, lz: %13.8f \n",
                        dEx,dEy,dEz,lx,ly,lz);fflush(stdout); 

		printf("lx*dEz: %17.10f, lz*dEx: %17.10f\n",lx*dEz,lz*dEx);fflush(stdout); 
		printf("exz: %17.10f, ezx: %17.10f\n",
                        openmx_solver_real_matrix_row(&stress_threads,0)[2],openmx_solver_real_matrix_row(&stress_threads,0)[6]);fflush(stdout);


		printf("lx1: %13.8f, ly1: %13.8f, lz1: %13.8f, lx2: %13.8f, ly2: %13.8f, lz2: %13.8f \n",
                        lx1,ly1,lz1,lx2,ly2,lz2);fflush(stdout); 
		printf("Gc_AN x: %13.8f, y: %13.8f, z: %13.8f  \n",Gxyz[Gc_AN][1],Gxyz[Gc_AN][2],Gxyz[Gc_AN][3]);
		printf("Gh_AN x: %13.8f, y: %13.8f, z: %13.8f, atv 1: %13.8f, 2: %13.8f, 3: %13.8f  \n",
                        Gxyz[Gh_AN][1],Gxyz[Gh_AN][2],Gxyz[Gh_AN][3],atv[Rn1][1],atv[Rn1][2],atv[Rn1][3]); 
		printf("Gq_AN x: %13.8f, y: %13.8f, z: %13.8f, atv 1: %13.8f, 2: %13.8f, 3: %13.8f  \n",
                        Gxyz[Gq_AN][1],Gxyz[Gq_AN][2],Gxyz[Gq_AN][3],atv[Rn2][1],atv[Rn2][2],atv[Rn2][3]); 
	      }
	      */

	      openmx_solver_real_matrix_row(&stress_threads,OMPID)[0] += 0.5*dEx*lx;
	      openmx_solver_real_matrix_row(&stress_threads,OMPID)[1] += 0.5*dEy*lx;
	      openmx_solver_real_matrix_row(&stress_threads,OMPID)[2] += 0.5*dEz*lx;
	      openmx_solver_real_matrix_row(&stress_threads,OMPID)[3] += 0.5*dEx*ly;
	      openmx_solver_real_matrix_row(&stress_threads,OMPID)[4] += 0.5*dEy*ly;
	      openmx_solver_real_matrix_row(&stress_threads,OMPID)[5] += 0.5*dEz*ly;
	      openmx_solver_real_matrix_row(&stress_threads,OMPID)[6] += 0.5*dEx*lz;
	      openmx_solver_real_matrix_row(&stress_threads,OMPID)[7] += 0.5*dEy*lz;
	      openmx_solver_real_matrix_row(&stress_threads,OMPID)[8] += 0.5*dEz*lz;

	    }

	    /* collinear spin polarized or non-colliear without SO and LDA+U */

	    else if (SpinP_switch==1 || (SpinP_switch==3 && SO_switch==0 && Hub_U_switch==0
		 && Constraint_NCS_switch==0 && Zeeman_NCS_switch==0 && Zeeman_NCO_switch==0)){ 

	      if (q_AN==h_AN) pref = 1.0;
	      else            pref = 2.0; 

	      /* Stress */

	      dEx = 0.0;
	      dEy = 0.0;
	      dEz = 0.0;
	      
	      for (i=0; i<atomic->species.contracted_orbitals[Hwan]; i++){
		for (j=0; j<atomic->species.contracted_orbitals[Qwan]; j++){

		  dEx += pref*(  CDM0[0][Mh_AN][kl][i][j]*openmx_solver_complex_plane_row(&Hx,0,i)[j].r
	                       + CDM0[1][Mh_AN][kl][i][j]*openmx_solver_complex_plane_row(&Hx,1,i)[j].r);
		  dEy += pref*(  CDM0[0][Mh_AN][kl][i][j]*openmx_solver_complex_plane_row(&Hy,0,i)[j].r
	                       + CDM0[1][Mh_AN][kl][i][j]*openmx_solver_complex_plane_row(&Hy,1,i)[j].r);
		  dEz += pref*(  CDM0[0][Mh_AN][kl][i][j]*openmx_solver_complex_plane_row(&Hz,0,i)[j].r
		               + CDM0[1][Mh_AN][kl][i][j]*openmx_solver_complex_plane_row(&Hz,1,i)[j].r);
		}
	      }

	      /*Stress*/

              /* debug */
	      /*
	      if(fabs(dEx)>1.e-5 || fabs(dEy)>1.e-5 || fabs(dEz)>1.e-5) { 
		printf("Gh_AN: %d, Gc_AN: %d, Gq_AN: %d \n",Gh_AN,Gc_AN,Gq_AN); 
		printf("dEx: %13.8f, dEy: %13.8f, dEz: %13.8f, lx: %13.8f, ly: %13.8f, lz: %13.8f \n",
                        dEx,dEy,dEz,lx,ly,lz);fflush(stdout); 
	      }
	      */

	      openmx_solver_real_matrix_row(&stress_threads,OMPID)[0] += 0.5*dEx*lx;
	      openmx_solver_real_matrix_row(&stress_threads,OMPID)[1] += 0.5*dEy*lx;
	      openmx_solver_real_matrix_row(&stress_threads,OMPID)[2] += 0.5*dEz*lx;
	      openmx_solver_real_matrix_row(&stress_threads,OMPID)[3] += 0.5*dEx*ly;
	      openmx_solver_real_matrix_row(&stress_threads,OMPID)[4] += 0.5*dEy*ly;
	      openmx_solver_real_matrix_row(&stress_threads,OMPID)[5] += 0.5*dEz*ly;
	      openmx_solver_real_matrix_row(&stress_threads,OMPID)[6] += 0.5*dEx*lz;
	      openmx_solver_real_matrix_row(&stress_threads,OMPID)[7] += 0.5*dEy*lz;
	      openmx_solver_real_matrix_row(&stress_threads,OMPID)[8] += 0.5*dEz*lz;
	    }
	    
	    /* spin collinear with spin-orbit coupling */
	    
	    else if ( SpinP_switch==1 && SO_switch==1 ){
	      printf("Spin-orbit coupling is not supported for collinear DFT calculations.\n");fflush(stdout);
	      MPI_Finalize();
	      exit(1);
	    }
	    
	    /* spin non-collinear with spin-orbit coupling or with LDA+U */
	    
	    else if ( SpinP_switch==3 && (SO_switch==1 || (Hub_U_switch==1 && F_U_flag==1) 
		  || Constraint_NCS_switch==1 || Zeeman_NCS_switch==1 || Zeeman_NCO_switch==1) ){
	      
	      /*Stress*/
	      printf("spin NC with SO coupling or with LDA+U: not supported for stress.\n");fflush(stdout);
	      MPI_Finalize();
	      exit(1);
	      
	    }
	    
	  } /* if(0<=kl) */
	} /* for q_AN */
      } /* for h_AN */

      dtime(&Etime_atom);
      time_per_atom[Gc_AN] += Etime_atom - Stime_atom;
      
    } /* for MC_AN */

    /* freeing of arrays */

  } /* pragma */

  /* Stress */
  /* if (myid==Host_ID && 0<level_stdout) printf("debug1, Nthrds0: %d \n",Nthrds0);fflush(stdout); */ /*debug*/

  for(i=0; i<9; i++){
    My_Stress[i]=0.0;
  }

  for(i=0; i<Nthrds0; i++){
    for(j=0; j<9; j++){
      /* printf("myid:%d, i:%d, j:%d \n",myid,i,j);fflush(stdout); */ /*debug*/
      My_Stress[j] += openmx_solver_real_matrix_row(&stress_threads,i)[j];
    }
  }

  /* if (myid==Host_ID && 0<level_stdout) printf("debug2 \n");fflush(stdout); */ /*debug*/

  MPI_Allreduce(&My_Stress[0],&Stress[0],9,MPI_DOUBLE,MPI_SUM,mpi_comm_level1);

  /* add kinetic components of stress to Stress_Tensor */

  for (i=0; i<9; i++){
    Stress_Tensor[i] += Stress[i];
  }

  /* show Kinetic components of stress */
  
  if (myid==Host_ID && 1<level_stdout){
    printf(" Kinetic components of stress : \n");fflush(stdout);
    for (i = 0; i<3; i++){
      for (j = 0; j<3; j++){
	printf("%16.8f ", Stress[3*i+j]);
      }
      printf("\n");fflush(stdout);
    }
  }

  dtime(&etime);
  if(myid==0 && measure_time){
    printf("Time for stress#2=%18.5f\n",etime-stime);fflush(stdout);
  }

  /* begin(yshiihara) */ 

  /* end(yshiihara) */

 /****************************************************
                      #3 of Stress

               dn/depsilon * (VNA + dVH + Vxc)
            or 
               dn/depsilon * (dVH + Vxc)
  ****************************************************/

  dtime(&stime);
  
  if (myid==Host_ID && 0<level_stdout){
    printf("  Stress calculation #3\n");fflush(stdout);
  }

  Stress3(scratch,atomic, electronic, local_basis);

  dtime(&etime);
  if(myid==0 && measure_time){
    printf("Time for Stress#3=%18.5f\n",etime-stime);fflush(stdout);
  } 

  /****************************************************
                    #4 of Stress

       Stress4:   n * dVNA/epsilon
       Stress4B:  from separable VNA projectors
  ****************************************************/

  dtime(&stime);

  if (myid==Host_ID && 0<level_stdout){
    printf("  Stress calculation #4\n");fflush(stdout);
  }

  if (ProExpn_VNA==0 && F_VNA_flag==1){
    Stress4(atomic);
  }
  else if (ProExpn_VNA==1 && F_VNA_flag==1){
    Stress4B(scratch,vna, reciprocal, CDM0, local_basis);
  }

  dtime(&etime);
  if(myid==0 && measure_time){
    printf("Time for Stress#4=%18.5f\n",etime-stime);fflush(stdout);
  } 

  /****************************************************
                      #5 of Stress

               Contribution from overlap
  ****************************************************/

  dtime(&stime);

  if (myid==Host_ID && 0<level_stdout){
    printf("  Stress calculation #5\n");fflush(stdout);
  }

  /* begin(yshiihara) */

  /* Dynamic teams may grow after a separate probe region. */
  Nthrds0 = omp_get_max_threads();

  OpenMX_Stress_ThreadsPrepare(scratch,Nthrds0,&stress_threads);
  for (i=0; i<Nthrds0; i++){

    for (j=0; j<9; j++) openmx_solver_real_matrix_row(&stress_threads,i)[j] = 0.0;
  }
  /* end(yshiihara) */

#pragma omp parallel shared(atomic,time_per_atom,Fx,Fy,Fz,CntOLP,OLP,Cnt_switch,EDM,SpinP_switch,natn,ncn,FNAN,WhatSpecies,M2G,Gxyz,atv,Matomnum,stress_threads) private(OMPID,Nthrds,Nprocs,Mc_AN,Stime_atom,Etime_atom,Gc_AN,Cwan,h_AN,Gh_AN,Hwan,i,j,dum,dx,dy,dz,lx,ly,lz,Rn1)
  {

    /* get info. on OpenMP */ 

    OMPID = omp_get_thread_num();
    Nthrds = omp_get_num_threads();
    Nprocs = omp_get_num_procs();

    for (Mc_AN=(OMPID*Matomnum/Nthrds+1); Mc_AN<((OMPID+1)*Matomnum/Nthrds+1); Mc_AN++){

      dtime(&Stime_atom);
      Gc_AN = M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];

      for (h_AN=1; h_AN<=FNAN[Gc_AN]; h_AN++){

	Gh_AN = natn[Gc_AN][h_AN];
	Hwan = WhatSpecies[Gh_AN];

        /*stress*/
	Rn1 = ncn[Gc_AN][h_AN];
	lx = Gxyz[Gc_AN][1] - Gxyz[Gh_AN][1] - atv[Rn1][1];
	ly = Gxyz[Gc_AN][2] - Gxyz[Gh_AN][2] - atv[Rn1][2];
	lz = Gxyz[Gc_AN][3] - Gxyz[Gh_AN][3] - atv[Rn1][3];

	for (i=0; i<atomic->species.contracted_orbitals[Cwan]; i++){
	  for (j=0; j<atomic->species.contracted_orbitals[Hwan]; j++){

	    if (SpinP_switch==0){
	      dum = 2.0*EDM[0][Mc_AN][h_AN][i][j];
	    }
	    else if (SpinP_switch==1 || SpinP_switch==3){
	      dum = EDM[0][Mc_AN][h_AN][i][j] + EDM[1][Mc_AN][h_AN][i][j];
	    }

	    if (Cnt_switch==0){
	      dx = dum*OLP[1][Mc_AN][h_AN][i][j];
	      dy = dum*OLP[2][Mc_AN][h_AN][i][j];
	      dz = dum*OLP[3][Mc_AN][h_AN][i][j];
	    }
	    else{
	      dx = dum*CntOLP[1][Mc_AN][h_AN][i][j];
	      dy = dum*CntOLP[2][Mc_AN][h_AN][i][j];
	      dz = dum*CntOLP[3][Mc_AN][h_AN][i][j];
	    }
	    
	    openmx_solver_real_matrix_row(&stress_threads,OMPID)[0] -= dx*lx;
	    openmx_solver_real_matrix_row(&stress_threads,OMPID)[1] -= dy*lx;
	    openmx_solver_real_matrix_row(&stress_threads,OMPID)[2] -= dz*lx;
	    openmx_solver_real_matrix_row(&stress_threads,OMPID)[3] -= dx*ly;
	    openmx_solver_real_matrix_row(&stress_threads,OMPID)[4] -= dy*ly;
	    openmx_solver_real_matrix_row(&stress_threads,OMPID)[5] -= dz*ly;
	    openmx_solver_real_matrix_row(&stress_threads,OMPID)[6] -= dx*lz;
	    openmx_solver_real_matrix_row(&stress_threads,OMPID)[7] -= dy*lz;
	    openmx_solver_real_matrix_row(&stress_threads,OMPID)[8] -= dz*lz;
	    
	  } /* Hwan */
	} /*i<Spe_Total_CNO*/
      } /*h_AN*/
      
      dtime(&Etime_atom);
      time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

    } /*Mc_AN*/
  } /*pragma*/

  /*Stress*/

  /* if (myid==Host_ID && 0<level_stdout) printf("debug1, Nthrds0: %d \n",Nthrds0);fflush(stdout); */ /*debug*/

  for(i=0; i<9; i++){
    My_Stress[i]=0.0;
  }

  for(i=0; i<Nthrds0; i++){
    for(j=0; j<9; j++){
      /* printf("myid:%d, i:%d, j:%d \n",myid,i,j);fflush(stdout);*/ /*debug*/
      My_Stress[j] += openmx_solver_real_matrix_row(&stress_threads,i)[j];
    }
  }
  /* if (myid==Host_ID && 0<level_stdout) printf("debug2 \n");fflush(stdout); */ /*debug*/
  
  MPI_Allreduce(&My_Stress[0],&Stress[0],9,MPI_DOUBLE,MPI_SUM,mpi_comm_level1);

  /* add #5 of stress to Stress_Tensor */

  for (i=0; i<9; i++){
    Stress_Tensor[i] += Stress[i];
  }

  /* show #5 of stress */

  if (myid==Host_ID && 1<level_stdout){
    printf(" Overlap components of stress : \n");fflush(stdout);
    for (i = 0; i<3; i++){
      for (j = 0; j<3; j++){
	printf("%16.8f ", Stress[3*i+j]);
      }
      printf("\n");fflush(stdout);
    }
  }

  dtime(&etime);
  if(myid==0 && measure_time){
    printf("Time for Stress#5=%18.5f\n",etime-stime);fflush(stdout);
  } 

  /* begin(yshiihara) */ 

  /* end(yshiihara) */

  /****************************************************
            Stress from non-local potentials 
  ****************************************************/

  Stress_HNL(scratch,atomic, CDM0, iDM0, electronic, local_basis);

  /* begin(yshiihara) */
  /****************************************************
              Stress for hartree term H1
  ****************************************************/

  dtime(&stime);
  
  Stress_H1(scratch);

  dtime(&etime);
  if(myid==0 && measure_time){
    printf("Time for Stress(H1)=%18.5f\n",etime-stime);fflush(stdout);
  }
  
  /* end(yshiihara) */

  /* begin(yshiihara) */
  /****************************************************
               core-core repulsion Stress
  ****************************************************/
  dtime(&stime);
  
  Stress_core(scratch);

  dtime(&etime);
  if(myid==0 && measure_time){
    printf("Time for Stress(core)=%18.5f\n",etime-stime);fflush(stdout);
  }
  dtime(&stime);
  
  Stress_H0(scratch,core,atomic);

  dtime(&etime);
  if(myid==0 && measure_time){
    printf("Time for Stress(H0)=%18.5f\n",etime-stime);fflush(stdout);
  }
  /* end(yshiihara) */

  /****************************************************
                       DFT+U 
  ****************************************************/

  if (Hub_U_switch==1){
    if (Hub_U_occupation==2) Stress_plus_U_dual(scratch,CDM0, electronic, local_basis); /* dual */
  }

  /****************************************************
          contribution of enthalpy term pV

      1 [GPa] = 0.3398827*0.0001 [Hartree/Bohr^3]
  ****************************************************/

  if ( MD_switch==17 || MD_switch==18){
    double coe = 0.3398827*0.0001;
    Stress_Tensor[0] += MD_applied_pressure*coe*Cell_Volume*(double)MD_applied_pressure_flag[0];
    Stress_Tensor[4] += MD_applied_pressure*coe*Cell_Volume*(double)MD_applied_pressure_flag[1];
    Stress_Tensor[8] += MD_applied_pressure*coe*Cell_Volume*(double)MD_applied_pressure_flag[2];

    thermo->pressure_volume = MD_applied_pressure*coe*Cell_Volume;
  }

  /* for time */

  MPI_Barrier(mpi_comm_level1);
  dtime(&TEtime);
  time0 = TEtime - TStime;

  return time0;
}


static void Stress3(OpenMXSolverInvocation *scratch,const OpenMXAtomicViews *atomic, const OpenMXElectronicViews *electronic, const OpenMXLocalBasisViews *local_basis)
{
  double ******DM=electronic->DM;

  /****************************************************
  	#3 of Stress
 
	dn/depsilon * (VNA + dVH + Vxc)
	or
	dn/depsilon * (dVH + Vxc)
  ****************************************************/
  /* for OpenMP */

  /* MPI */
  int numprocs, myid;
  MPI_Comm_size(mpi_comm_level1, &numprocs);
  MPI_Comm_rank(mpi_comm_level1, &myid);

  /* begin(yshiihara) */
  int spin;
  int Ng1, Ng2, Ng3, DN, BN;
  int XC_P_switch_stress;

  /* Stress */	
  double Stress[9];
  double My_Stress[9];

  {
    int i;
    for (i = 0; i<9; i++) {
      Stress[i] = 0.0;
      My_Stress[i] = 0.0;
    }
  }

  /**********************************************************
            main loop for calculation of Stress #3
  **********************************************************/

  /* shared memory for force */

  OpenMXStressGridScratch grid;
  OpenMX_Stress_GridPrepare(scratch,SpinP_switch,Max_GridN_Atom,List_YOUSO[7],&grid);

  double sumx = 0.0; /* this must be defined out of parallel pragma */
  double sumy = 0.0;
  double sumz = 0.0;
  double sumt0 = 0.0; /* this must be defined out of parallel pragma */
  double sumt1 = 0.0;
  double sumt2 = 0.0;
  double sumt3 = 0.0;
  double sumt4 = 0.0;
  double sumt5 = 0.0;
  double sumt6 = 0.0;
  double sumt7 = 0.0;
  double sumt8 = 0.0;

#pragma omp parallel
  {

    /* Borrow worker-owned producer rows and orbital sampler. */

    OpenMXSolverRealTensor orbital_rows;
    OpenMX_Stress_OrbitalPrepare(scratch,List_YOUSO[7],&orbital_rows);
    double *dorbs0[4];
    for(size_t row=0;row<4;row++)dorbs0[row]=openmx_solver_real_matrix_row(&orbital_rows,row);
    double *orbs1=OpenMX_Solver_OverwriteReal(scratch,"stress:grid:neighbor-orbitals",(size_t)List_YOUSO[7]);
    OpenMXOrbitalBuffers work_dObs;
    OpenMX_Solver_Orbitals(scratch,"stress:grid:orbital-sampler",
        (int)atomic->orbitals.extent[1]-1,(int)atomic->orbitals.extent[2],1,&work_dObs);

    double time1 = 0.0;
    double time2 = 0.0;
    double last_time;
    double current_time;

		
    int Mc_AN;
    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {

      int Gc_AN = M2G[Mc_AN];
      int Cwan = WhatSpecies[Gc_AN];
      int NO0 = atomic->species.contracted_orbitals[Cwan];

      /***********************************
   	          calc dOrb0
      ***********************************/
#pragma omp barrier      /* this barrier is necessary to wait (1)clearing sumx, sumy, sumz, and (2)initializing work_dObs */
      dtime(&last_time);

      int Nc;
#pragma omp for
      for (Nc = 0; Nc<GridN_Atom[Gc_AN]; Nc++) {

	int GNc = GridListAtom[Mc_AN][Nc];
	int GRc = CellListAtom[Mc_AN][Nc];
	int MNc = MGridListAtom[Mc_AN][Nc];

	double Cxyz[4];
	Get_Grid_XYZ(GNc, Cxyz);
	double x = Cxyz[1] + atv[GRc][1];
	double y = Cxyz[2] + atv[GRc][2];
	double z = Cxyz[3] + atv[GRc][3];
	double dx = x - Gxyz[Gc_AN][1];
	double dy = y - Gxyz[Gc_AN][2];
	double dz = z - Gxyz[Gc_AN][3];

	/* begin(yshiihara) */
	openmx_solver_real_matrix_row(&grid.displacements,Nc)[0] = dx;
	openmx_solver_real_matrix_row(&grid.displacements,Nc)[1] = dy;
	openmx_solver_real_matrix_row(&grid.displacements,Nc)[2] = dz;
	/* end(yshiihara) */

	if (Cnt_switch == 0) {
	  /* AITUNE201704 Get_dOrbitals(Cwan, dx, dy, dz, dorbs0);*/
	  Get_dOrbitals_work(atomic, Cwan, dx, dy, dz, dorbs0, work_dObs);
	} else {
	  Get_Cnt_dOrbitals(atomic, Mc_AN, dx, dy, dz, dorbs0, local_basis);
	}

	int k;
	for (k = 0; k<3; k++) {
	  int i;
	  for (i = 0; i<NO0; i++) {
	    OpenMX_Stress_DerivativeRow(&grid.derivatives,Nc,i)[k] = dorbs0[k + 1][i];
	  }
	}


	if (SpinP_switch == 0 || SpinP_switch == 1) {

	  int spin;
	  for (spin = 0; spin <= SpinP_switch; spin++) {

	    double Vpt;
	    if (0 <= MNc) {
	      if (E_Field_switch == 1) {

		if (ProExpn_VNA == 0) {

		  Vpt = F_dVHart_flag*dVHart_Grid[MNc]
		    + F_Vxc_flag*Vxc_Grid[spin][MNc]
		    + F_VNA_flag*VNA_Grid[MNc]
		    + F_VEF_flag*VEF_Grid[MNc];

		} else {

		  Vpt = F_dVHart_flag*dVHart_Grid[MNc]
		    + F_Vxc_flag*Vxc_Grid[spin][MNc]
		    + F_VEF_flag*VEF_Grid[MNc];

		}

	      } else {
		if (ProExpn_VNA == 0) {

		  Vpt = F_dVHart_flag*dVHart_Grid[MNc]
		    + F_Vxc_flag*Vxc_Grid[spin][MNc]
		    + F_VNA_flag*VNA_Grid[MNc];

		} else {

		  Vpt = F_dVHart_flag*dVHart_Grid[MNc]
		    + F_Vxc_flag*Vxc_Grid[spin][MNc];

		}
	      }
	    } else {
	      Vpt = 0.0;
	    }

	    if (SpinP_switch == 0) {
	      openmx_solver_real_matrix_row(&grid.potential,0)[Nc] = 4.0 * Vpt;
	    } else if (SpinP_switch == 1) {
	      openmx_solver_real_matrix_row(&grid.potential,spin)[Nc] = 2.0 * Vpt;
	    }

	  }

	} else if (SpinP_switch == 3) {

	  /* spin non-collinear */


	  double ReVpt11;
	  double ReVpt22;
	  double ReVpt21;
	  double ImVpt21;

	  if (0 <= MNc) {
	    if (E_Field_switch == 1) {

	      if (ProExpn_VNA == 0) {

		ReVpt11 = F_dVHart_flag*dVHart_Grid[MNc]
		  + F_Vxc_flag*Vxc_Grid[0][MNc]
		  + F_VNA_flag*VNA_Grid[MNc]
		  + F_VEF_flag*VEF_Grid[MNc];

		ReVpt22 = F_dVHart_flag*dVHart_Grid[MNc]
		  + F_Vxc_flag*Vxc_Grid[1][MNc]
		  + F_VNA_flag*VNA_Grid[MNc]
		  + F_VEF_flag*VEF_Grid[MNc];

		ReVpt21 = F_Vxc_flag*Vxc_Grid[2][MNc];
		ImVpt21 = -F_Vxc_flag*Vxc_Grid[3][MNc];
	      } else {

		ReVpt11 = F_dVHart_flag*dVHart_Grid[MNc]
		  + F_Vxc_flag*Vxc_Grid[0][MNc]
		  + F_VEF_flag*VEF_Grid[MNc];

		ReVpt22 = F_dVHart_flag*dVHart_Grid[MNc]
		  + F_Vxc_flag*Vxc_Grid[1][MNc]
		  + F_VEF_flag*VEF_Grid[MNc];

		ReVpt21 = F_Vxc_flag*Vxc_Grid[2][MNc];
		ImVpt21 = -F_Vxc_flag*Vxc_Grid[3][MNc];
	      }

	    } else {

	      if (ProExpn_VNA == 0) {

		ReVpt11 = F_dVHart_flag*dVHart_Grid[MNc]
		  + F_Vxc_flag*Vxc_Grid[0][MNc]
		  + F_VNA_flag*VNA_Grid[MNc];

		ReVpt22 = F_dVHart_flag*dVHart_Grid[MNc]
		  + F_Vxc_flag*Vxc_Grid[1][MNc]
		  + F_VNA_flag*VNA_Grid[MNc];

		ReVpt21 = F_Vxc_flag*Vxc_Grid[2][MNc];
		ImVpt21 = -F_Vxc_flag*Vxc_Grid[3][MNc];

	      } else {

		ReVpt11 = F_dVHart_flag*dVHart_Grid[MNc] + F_Vxc_flag*Vxc_Grid[0][MNc];
		ReVpt22 = F_dVHart_flag*dVHart_Grid[MNc] + F_Vxc_flag*Vxc_Grid[1][MNc];

		ReVpt21 = F_Vxc_flag*Vxc_Grid[2][MNc];
		ImVpt21 = -F_Vxc_flag*Vxc_Grid[3][MNc];
	      }

	    }
	  } else {
	    ReVpt11 = 0.0;
	    ReVpt22 = 0.0;
	    ReVpt21 = 0.0;
	    ImVpt21 = 0.0;
	  }

	  openmx_solver_real_matrix_row(&grid.potential,0)[Nc] = 2.0 * ReVpt11;
	  openmx_solver_real_matrix_row(&grid.potential,1)[Nc] = 2.0 * ReVpt22;
	  openmx_solver_real_matrix_row(&grid.potential,2)[Nc] = 4.0 * ReVpt21;
	  openmx_solver_real_matrix_row(&grid.potential,3)[Nc] = 4.0 * ImVpt21;

					
	}

      }/* Nc, here omp barrier is called implicitly because of end of for loop */

      dtime(&current_time);
      time1 += current_time - last_time;
      last_time = current_time;

      /***********************************
                 calc Stress #3
      ***********************************/

      int h_AN;
      for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {

	int Gh_AN = natn[Gc_AN][h_AN];
	int Mh_AN = F_G2M[Gh_AN];
	int Rnh = ncn[Gc_AN][h_AN];
	int Hwan = WhatSpecies[Gh_AN];
	int NO1 = atomic->species.contracted_orbitals[Hwan];

	int Nog;
#pragma omp for reduction (+:sumx, sumy, sumz, sumt0, sumt1, sumt2, sumt3, sumt4, sumt5, sumt6, sumt7, sumt8)
	for (Nog = 0; Nog<NumOLG[Mc_AN][h_AN]; Nog++) {

	  int Nc = GListTAtoms1[Mc_AN][h_AN][Nog];
	  int Nh = GListTAtoms2[Mc_AN][h_AN][Nog];
					
	  double* const ai_dorbs0 = OpenMX_Stress_DerivativeRow(&grid.derivatives,Nc,0);

	  /* set orbs1 */

	  if (G2ID[Gh_AN] == myid) {
	    int j;
	    for (j = 0; j<NO1; j++) {
	      orbs1[j] = Orbs_Grid[Mh_AN][Nh][j];
	    }
	  } else {
	    int j;
	    for (j = 0; j<NO1; j++) {
	      orbs1[j] = Orbs_Grid_FNAN[Mc_AN][h_AN][Nog][j];
	    }
	  }

	  int spin;
	  for (spin = 0; spin <= SpinP_switch; spin++) {

	    double tmpx = 0.0;
	    double tmpy = 0.0;
	    double tmpz = 0.0;

	    int i;
	    for (i = 0; i<NO0; i++) {
	      double tmp0 = 0.0;
	      int j;
	      for (j = 0; j<NO1; j++) {
		tmp0 += orbs1[j] * DM[0][spin][Mc_AN][h_AN][i][j];
	      }

	      tmpx += ai_dorbs0[3*i+0] * tmp0;
	      tmpy += ai_dorbs0[3*i+1] * tmp0;
	      tmpz += ai_dorbs0[3*i+2] * tmp0;
	    }

	    /* due to difference in the definition between density matrix and density */
						
	    double Vpt = openmx_solver_real_matrix_row(&grid.potential,spin)[Nc];
	    sumx += tmpx * Vpt;
	    sumy += tmpy * Vpt;
	    sumz += tmpz * Vpt;

	    /* begin(yshiihara) */
	    /* The minus sign is for switching the sign of dDen_Grid*/
	    sumt0 -= openmx_solver_real_matrix_row(&grid.displacements,Nc)[0] * tmpx * Vpt;
	    sumt1 -= openmx_solver_real_matrix_row(&grid.displacements,Nc)[1] * tmpx * Vpt;
	    sumt2 -= openmx_solver_real_matrix_row(&grid.displacements,Nc)[2] * tmpx * Vpt;
	    sumt3 -= openmx_solver_real_matrix_row(&grid.displacements,Nc)[0] * tmpy * Vpt;
	    sumt4 -= openmx_solver_real_matrix_row(&grid.displacements,Nc)[1] * tmpy * Vpt;
	    sumt5 -= openmx_solver_real_matrix_row(&grid.displacements,Nc)[2] * tmpy * Vpt;
	    sumt6 -= openmx_solver_real_matrix_row(&grid.displacements,Nc)[0] * tmpz * Vpt;
	    sumt7 -= openmx_solver_real_matrix_row(&grid.displacements,Nc)[1] * tmpz * Vpt;
	    sumt8 -= openmx_solver_real_matrix_row(&grid.displacements,Nc)[2] * tmpz * Vpt;
	    /* end(yshiihara) */

	  }/* spin */

	}/* Nog */
      }/* h_AN, here omp barrier is called implicitly because of end of for loop  */

#pragma omp master
      {
	/**********************************
        only for forces
        Gxyz[Gc_AN][17] += sumx*GridVol;
        Gxyz[Gc_AN][18] += sumy*GridVol;
        Gxyz[Gc_AN][19] += sumz*GridVol;
	***********************************/

	sumx = 0.0;
	sumy = 0.0;
	sumz = 0.0;

	/* begin(yshiihara) */
	My_Stress[0] += sumt0 * GridVol;
	My_Stress[1] += sumt1 * GridVol;
	My_Stress[2] += sumt2 * GridVol;
	My_Stress[3] += sumt3 * GridVol;
	My_Stress[4] += sumt4 * GridVol;
	My_Stress[5] += sumt5 * GridVol;
	My_Stress[6] += sumt6 * GridVol;
	My_Stress[7] += sumt7 * GridVol;
	My_Stress[8] += sumt8 * GridVol;
	/* end(yshiihara)   */

	sumt0 = sumt1 = sumt2 = 0.0;
	sumt3 = sumt4 = sumt5 = 0.0;
	sumt6 = sumt7 = sumt8 = 0.0;
      }

      dtime(&current_time);
      time2 += current_time - last_time;
      last_time = current_time;

    } /* Mc_AN */

#if measure_time
#pragma omp master
    printf("<Stress>  stress(3) myid=%2d  time1, 2 = %lf [s], %lf [s]\n", myid, time1, time2);
#endif

  } /* #pragma omp parallel */

  MPI_Allreduce(&My_Stress[0], &Stress[0], 9, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

  /* add #3 of stress to Stress_Tensor */

  {
    int i;
    for (i = 0; i<9; i++) {
      Stress_Tensor[i] += Stress[i];
    }
  }

  /* show #3 of stress */

  if (myid == Host_ID && 1<level_stdout) {
    printf(" Stress #3 : \n"); fflush(stdout);
    int i, j;
    for (i = 0; i<3; i++) {
      for (j = 0; j<3; j++) {
	printf("%16.8f ", Stress[3 * i + j]);
      }
      printf("\n"); fflush(stdout);
    }
  }

  /* end(yshiihara)   */

  /**********************************************************
     In case of GGA, a stress term coming from
     dfxc/d|\nabra n| *|d|\nabra n|/d\nabra n * \nabra n
     is added.
  **********************************************************/

  /* for GGA stress */
  if (XC_switch == 4) {

    int i, j, k, n, spinmax;
    double My_Stress2[2][9], Stress2[2][9];
    OpenMXSolverRealTensor gga_d,gga_b;
    OpenMX_Stress_GradientPrepare(scratch,"stress:grid:gga-d",My_NumGridD,&gga_d);
    OpenMX_Stress_GradientPrepare(scratch,"stress:grid:gga-b",My_NumGridB_AB,&gga_b);
    double *axc_d_rows[2][3],*density_d_rows[2][3];
    double **Axc_D[2]={axc_d_rows[0],axc_d_rows[1]};
    double **dDen_D[2]={density_d_rows[0],density_d_rows[1]};
    for(spin=0;spin<2;spin++)for(int axis=0;axis<3;axis++){
      axc_d_rows[spin][axis]=OpenMX_Stress_GradientRow(&gga_d,0,spin,axis);
      density_d_rows[spin][axis]=OpenMX_Stress_GradientRow(&gga_d,1,spin,axis);
    }

    XC_P_switch_stress = 3;
    /* first argument in Set_XC_Grid was added by S.Ryee */
    Set_XC_Grid(2,XC_P_switch_stress, XC_switch,
		Density_Grid_D[0], Density_Grid_D[1],
		Density_Grid_D[2], Density_Grid_D[3],
		NULL, NULL, NULL, NULL,
		Axc_D, dDen_D);

    /* change data structure: D -> B -> C */

    for (spin = 0; spin <= 1; spin++) {
      Ng1 = Max_Grid_Index_D[1] - Min_Grid_Index_D[1] + 1;
      Ng2 = Max_Grid_Index_D[2] - Min_Grid_Index_D[2] + 1;
      Ng3 = Max_Grid_Index_D[3] - Min_Grid_Index_D[3] + 1;

      for (n = 0; n<Num_Rcv_Grid_B2D[myid]; n++) {
	DN = Index_Rcv_Grid_B2D[myid][n];
	BN = Index_Snd_Grid_B2D[myid][n];

	i = DN / (Ng2*Ng3);
	j = (DN - i*Ng2*Ng3) / Ng3;
	k = DN - i*Ng2*Ng3 - j*Ng3;

	if (!(i <= 1 || (Ng1 - 2) <= i || j <= 1 || (Ng2 - 2) <= j || k <= 1 || (Ng3 - 2) <= k)) {

	  OpenMX_Stress_GradientRow(&gga_b,0,spin,0)[BN] = OpenMX_Stress_GradientRow(&gga_d,0,spin,0)[DN];
	  OpenMX_Stress_GradientRow(&gga_b,0,spin,1)[BN] = OpenMX_Stress_GradientRow(&gga_d,0,spin,1)[DN];
	  OpenMX_Stress_GradientRow(&gga_b,0,spin,2)[BN] = OpenMX_Stress_GradientRow(&gga_d,0,spin,2)[DN];

	  OpenMX_Stress_GradientRow(&gga_b,1,spin,0)[BN] = OpenMX_Stress_GradientRow(&gga_d,1,spin,0)[DN];
	  OpenMX_Stress_GradientRow(&gga_b,1,spin,1)[BN] = OpenMX_Stress_GradientRow(&gga_d,1,spin,1)[DN];
	  OpenMX_Stress_GradientRow(&gga_b,1,spin,2)[BN] = OpenMX_Stress_GradientRow(&gga_d,1,spin,2)[DN];
	}
      }
    }

    /* loop for BN */

    if (SpinP_switch == 0) spinmax = 0;
    else if (SpinP_switch == 1) spinmax = 1;
    else if (SpinP_switch == 3) spinmax = 1;

    for (spin = 0; spin <= spinmax; spin++) {

      for (i = 0; i<9; i++) My_Stress2[spin][i] = 0.0;

      for (BN = 0; BN<My_NumGridB_AB; BN++) {
	My_Stress2[spin][0] -= OpenMX_Stress_GradientRow(&gga_b,1,spin,0)[BN] * OpenMX_Stress_GradientRow(&gga_b,0,spin,0)[BN];
	My_Stress2[spin][1] -= OpenMX_Stress_GradientRow(&gga_b,1,spin,0)[BN] * OpenMX_Stress_GradientRow(&gga_b,0,spin,1)[BN];
	My_Stress2[spin][2] -= OpenMX_Stress_GradientRow(&gga_b,1,spin,0)[BN] * OpenMX_Stress_GradientRow(&gga_b,0,spin,2)[BN];
	My_Stress2[spin][3] -= OpenMX_Stress_GradientRow(&gga_b,1,spin,1)[BN] * OpenMX_Stress_GradientRow(&gga_b,0,spin,0)[BN];
	My_Stress2[spin][4] -= OpenMX_Stress_GradientRow(&gga_b,1,spin,1)[BN] * OpenMX_Stress_GradientRow(&gga_b,0,spin,1)[BN];
	My_Stress2[spin][5] -= OpenMX_Stress_GradientRow(&gga_b,1,spin,1)[BN] * OpenMX_Stress_GradientRow(&gga_b,0,spin,2)[BN];
	My_Stress2[spin][6] -= OpenMX_Stress_GradientRow(&gga_b,1,spin,2)[BN] * OpenMX_Stress_GradientRow(&gga_b,0,spin,0)[BN];
	My_Stress2[spin][7] -= OpenMX_Stress_GradientRow(&gga_b,1,spin,2)[BN] * OpenMX_Stress_GradientRow(&gga_b,0,spin,1)[BN];
	My_Stress2[spin][8] -= OpenMX_Stress_GradientRow(&gga_b,1,spin,2)[BN] * OpenMX_Stress_GradientRow(&gga_b,0,spin,2)[BN];
      }

      MPI_Allreduce(&My_Stress2[spin][0], &Stress2[spin][0], 9, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

      for (i = 0; i<9; i++) Stress2[spin][i] = Stress2[spin][i] * GridVol;

    } /*spin */

    if (SpinP_switch == 0) {
      for (i = 0; i<9; i++) Stress2[1][i] = Stress2[0][i];
    }

    /* add the GGA term in #3 of stress to Stress_Tensor */
    {
      int i;
      for (i = 0; i<9; i++) {
	Stress_Tensor[i] += Stress2[0][i] + Stress2[1][i];
      }
    }

    /* show the GGA term in #3 of stress */

    if (XC_switch == 4 && myid == Host_ID && 1<level_stdout) {
      printf(" Stress #3 (GGA term) : \n"); fflush(stdout);
      int i, j;
      for (i = 0; i<3; i++) {
	for (j = 0; j<3; j++) {
	  printf("%16.8f ", Stress2[0][3 * i + j] + Stress2[1][3 * i + j]);
	}
	printf("\n"); fflush(stdout);
      }
    }

  } /* if (XC_switch==4) */

  /* end(yshiihara) */
}


      
      


void Stress4(const OpenMXAtomicViews *atomic)
{
  /****************************************************
                      #4 of Stress

                      n * dVNA/depsilon
  ****************************************************/

  int Mc_AN,Gc_AN,Cwan,Hwan,NO0,NO1;
  int i,j,k,Nc,Nh,GNc,GRc,MNc;
  int h_AN,Gh_AN,Mh_AN,Rnh,spin,Nog;
  double sum,tmp0,r,dx,dy,dz;
  double dvx,dvy,dvz;
  double sumx,sumy,sumz;
  double x,y,z,den;
  double Cxyz[4];

  /* begin(yshiihara) */
  int numprocs,myid;
  /* Stress */

  double Stress[9];
  double My_Stress[9];
  double sumt[9];

  {
    int i;
    for(i=0; i<9; i++) {
      Stress[i]    = 0.0;
      My_Stress[i] = 0.0;
      sumt[i]      = 0.0;
    }
  }

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);
  MPI_Barrier(mpi_comm_level1);
  /* begin(yshiihara) */
    
  /**********************************************************
              main loop for calculation of Stress #4
  **********************************************************/

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    
    Gc_AN = M2G[Mc_AN];
    Cwan = WhatSpecies[Gc_AN];
    NO0 = atomic->species.contracted_orbitals[Cwan];

    /***********************************
                 summation 
    ***********************************/

    sumx = 0.0;
    sumy = 0.0;
    sumz = 0.0;
    /* begin(yshiihara) */
    for(i=0; i<9; i++){
      sumt[i] = 0.0;
    }
    /* begin(yshiihara) */

    for (Nc=0; Nc<GridN_Atom[Gc_AN]; Nc++){

      GNc = GridListAtom[Mc_AN][Nc]; 
      GRc = CellListAtom[Mc_AN][Nc];
      MNc = MGridListAtom[Mc_AN][Nc];

      Get_Grid_XYZ(GNc,Cxyz);
      x = Cxyz[1] + atv[GRc][1];
      y = Cxyz[2] + atv[GRc][2];
      z = Cxyz[3] + atv[GRc][3];
      dx = Gxyz[Gc_AN][1] - x;
      dy = Gxyz[Gc_AN][2] - y;
      dz = Gxyz[Gc_AN][3] - z;
      r = sqrt(dx*dx + dy*dy + dz*dz);

         /* for empty atoms or finite elemens basis */
      if (r<1.0e-10) r = 1.0e-10;

      if (1.0e-14<r){
        tmp0 = Dr_VNAF(atomic, Cwan,r);
        dvx = tmp0*dx/r;
        dvy = tmp0*dy/r;
        dvz = tmp0*dz/r;
      }
      else{
        dvx = 0.0;
        dvy = 0.0;
        dvz = 0.0;
      }

      den = Density_Grid[0][MNc] + Density_Grid[1][MNc];
      sumx += den*dvx;
      sumy += den*dvy;
      sumz += den*dvz;
      /* begin(yshiihara) */
      sumt[0] += den*dvx*dx;
      sumt[1] += den*dvy*dx;
      sumt[2] += den*dvz*dx;
      sumt[3] += den*dvx*dy;
      sumt[4] += den*dvy*dy;
      sumt[5] += den*dvz*dy;
      sumt[6] += den*dvx*dz;
      sumt[7] += den*dvy*dz;
      sumt[8] += den*dvz*dz;
      /* end(yshiihara) */
    }

    /* begin(yshiihara) */
    for(j=0; j<9; j++){
      My_Stress[j] += sumt[j]*GridVol*(double)F_VNA_flag;
    }
    /* end(yshiihara) */
    
  }

  /* begin(yshiihara) */
  MPI_Allreduce(&My_Stress[0],&Stress[0],9,MPI_DOUBLE,MPI_SUM,mpi_comm_level1);

  /* add #4 of stress to Stress_Tensor */

  for (i=0; i<9; i++){
    Stress_Tensor[i] += Stress[i];
  }

  /* show #4 of stress */
  
  if (myid==Host_ID && 1<level_stdout){
    printf("  Stress#4 : \n");fflush(stdout);
    for (i = 0; i<3; i++){
      for (j = 0; j<3; j++){
	printf("%16.8f ", Stress[3*i+j]);
      }
      printf("\n");fflush(stdout);
    }
  }
  /* end(yshiihara) */

}



void Stress4B(OpenMXSolverInvocation *scratch,const OpenMXVnaMatrixViews *vna, const OpenMXReciprocalViews *reciprocal, double *****CDM0, const OpenMXLocalBasisViews *local_basis)
{
  double *****HVNA2=vna->HVNA2;
  double *****HVNA3=vna->HVNA3;
  double *****CntHVNA2=vna->CntHVNA2;
  double *****CntHVNA3=vna->CntHVNA3;
  Type_DS_VNA *****DS_VNA=vna->DS_VNA;
  Type_DS_VNA *****CntDS_VNA=vna->CntDS_VNA;
  /****************************************************
                      #4 of Stress

            by the projector expansion of VNA
  ****************************************************/

  int Mc_AN,Gc_AN,Cwan,i,j,k,m,n,h_AN,q_AN,start_q_AN,Mq_AN;
  int jan,kl,Qwan,Gq_AN,Gh_AN,Mh_AN,Hwan,ian;
  int l1,l2,l3,l,LL,Mul1,Num_RVNA,tno0,ncp;
  int tno1,tno2,size1,size2,kk,num,po,po1,po2;
  int numprocs,myid,tag=999,ID,IDS,IDR;

  int S_comm_flag,R_comm_flag;
  int SA_num,q,Sc_AN,GSc_AN;
  int Sc_wan,Sh_AN,GSh_AN,Sh_wan;
  int Sh_AN2,fan,jg,j0,jg0,Mj_AN0;
  int Original_Mc_AN;

  double rcutA,rcutB,rcut;
  double dEx,dEy,dEz,ene,pref;
  double Stime_atom, Etime_atom;
  OpenMXSolverRealTensor HVNAx,HVNAy,HVNAz;
  OpenMXSolverRealTensor HVNA_sts;

  int *VNA_List;
  int *VNA_List2;
  int *Snd_DS_VNA_Size,*Rcv_DS_VNA_Size;  

  Type_DS_VNA *tmp_array;
  Type_DS_VNA *tmp_array2;

  /* for OpenMP */
  int OMPID,Nthrds,Nthrds0,Nprocs,Nloop,ODNloop;
  int *OneD2h_AN,*OneD2q_AN;
  double *dEx_threads;
  double *dEy_threads;
  double *dEz_threads;
  double stime,etime;
  double stime1,etime1;

  /* Stress */
  OpenMXSolverRealTensor stress_threads;
  double Stress[9],My_Stress[9];

  MPI_Status stat;
  MPI_Request request;

  /* MPI */

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);
  OpenMXProjectorTransfer projector_transfer;
  OpenMX_TransferBeginOwned(OpenMX_Solver_Resources(scratch,"stress:vna:projector-transfer"),
      mpi_comm_level1,numprocs,sizeof(Type_DS_VNA),&projector_transfer);
  int *send_progress=projector_transfer.buffers.sent;
  int *receive_progress=projector_transfer.buffers.received;


  dtime(&stime);

  /****************************
       allocation of arrays 
  *****************************/

  Snd_DS_VNA_Size = projector_transfer.buffers.send_sizes;
  Rcv_DS_VNA_Size = projector_transfer.buffers.receive_sizes;

  OpenMX_Stress_VnaListsPrepare(scratch,List_YOUSO[34],List_YOUSO[35],&VNA_List,&VNA_List2);

  /* get Nthrds0 */  
  /* Dynamic teams may grow after a separate probe region. */
  Nthrds0 = omp_get_max_threads();

  OpenMX_Stress_ThreadsPrepare(scratch,Nthrds0,&stress_threads);
  for (i=0; i<Nthrds0; i++){

    for (j=0; j<9; j++) openmx_solver_real_matrix_row(&stress_threads,i)[j] = 0.0;
  }

  /* initialization of Stress and My_Stress */

  for (i=0; i<9; i++){
    Stress[i]    = 0.0;
    My_Stress[i] = 0.0;
  }

  /*************************************************************
                 contraction of DS_VNA and HVNA2
  *************************************************************/

  if (Cnt_switch==1 && ProExpn_VNA==1){

    Cont_Matrix2(DS_VNA[0],CntDS_VNA[0], local_basis);
    Cont_Matrix2(DS_VNA[1],CntDS_VNA[1], local_basis);
    Cont_Matrix2(DS_VNA[2],CntDS_VNA[2], local_basis);
    Cont_Matrix2(DS_VNA[3],CntDS_VNA[3], local_basis);

    Cont_Matrix3(HVNA2[1],CntHVNA2[1], local_basis);
    Cont_Matrix3(HVNA2[2],CntHVNA2[2], local_basis);
    Cont_Matrix3(HVNA2[3],CntHVNA2[3], local_basis);

    Cont_Matrix4(HVNA3[1],CntHVNA3[1], local_basis);
    Cont_Matrix4(HVNA3[2],CntHVNA3[2], local_basis);
    Cont_Matrix4(HVNA3[3],CntHVNA3[3], local_basis);
  }

  /*************************************************************
                  make VNA_List and VNA_List2
  *************************************************************/

  l = 0;
  for (i=0; i<=List_YOUSO[35]; i++){     /* max L */
    for (j=0; j<List_YOUSO[34]; j++){    /* # of radial projectors */
      VNA_List[l]  = i;
      VNA_List2[l] = j;
      l++;
    }
  }

  Num_RVNA = List_YOUSO[34]*(List_YOUSO[35] + 1);

  dtime(&etime);
  if(myid==0 && measure_time){
    printf("Time for part1 of force#4=%18.5f\n",etime-stime);fflush(stdout);
  } 

  /*****************************************************
   if orbital optimization
   copy CntDS_VNA[0] into DS_VNA[0]
  *****************************************************/

  if (Cnt_switch==1){

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

      Gc_AN = F_M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];
      tno0 = Spe_Total_CNO[Cwan];  

      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	Gh_AN = natn[Gc_AN][h_AN];
	Hwan = WhatSpecies[Gh_AN];

	for (i=0; i<tno0; i++){

	  l = 0;
	  for (l1=0; l1<Num_RVNA; l1++){

	    l2 = 2*VNA_List[l1];
	    for (l3=0; l3<=l2; l3++){
	      DS_VNA[0][Mc_AN][h_AN][i][l] = CntDS_VNA[0][Mc_AN][h_AN][i][l];
	      l++;
	    }
	  }
	}
      }
    }
  }

  /*****************************************************
     (1) pre-multiplying DS_VNA[kk] with ene
     (2) copy DS_VNA[kk] or CntDS_VNA[kk] into DS_VNA[kk]
  *****************************************************/

  dtime(&stime);

  /* When Stress.c is integrated into Force.c and Total_Energy.c, the following note is very important.
     if Stress() is called after calling Force(), the following calculations should be skipped.
     Otherwise, DS_VNA[kk] is multiplied by "ene" twice.
  */

  if (0){

#pragma omp parallel shared(scratch,CntDS_VNA,DS_VNA,Cnt_switch,reciprocal,VNA_List2,VNA_List,Num_RVNA,natn,FNAN,Spe_Total_CNO,WhatSpecies,F_M2G,Matomnum) private(kk,OMPID,Nthrds,Nprocs,Gc_AN,Cwan,tno0,Mc_AN,h_AN,Gh_AN,Hwan,i,l,l1,LL,Mul1,ene,l2,l3)
  {

    /* get info. on OpenMP */ 

    OMPID = omp_get_thread_num();
    Nthrds = omp_get_num_threads();
    Nprocs = omp_get_num_procs();

    for (kk=1; kk<=3; kk++){
      for (Mc_AN=(OMPID*Matomnum/Nthrds+1); Mc_AN<((OMPID+1)*Matomnum/Nthrds+1); Mc_AN++){

	Gc_AN = F_M2G[Mc_AN];
	Cwan = WhatSpecies[Gc_AN];
	tno0 = Spe_Total_CNO[Cwan];  

	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];

	  for (i=0; i<tno0; i++){

	    l = 0;
	    for (l1=0; l1<Num_RVNA; l1++){

	      LL   = VNA_List[l1];
	      Mul1 = VNA_List2[l1];

	      ene = openmx_radial_value3(&reciprocal->neutral_energies,Hwan,LL,Mul1);
	      l2 = 2*VNA_List[l1];

	      if (Cnt_switch==0){
		for (l3=0; l3<=l2; l3++){
		  DS_VNA[kk][Mc_AN][h_AN][i][l] = ene*DS_VNA[kk][Mc_AN][h_AN][i][l];
		  l++;
		}
	      }
 
	      else{
		for (l3=0; l3<=l2; l3++){
		  DS_VNA[kk][Mc_AN][h_AN][i][l] = ene*CntDS_VNA[kk][Mc_AN][h_AN][i][l];
		  l++;
		}
	      }
	    }
	  } 

	} /* h_AN */
      } /* Mc_AN */
    } /* kk */

  } /* #pragma omp parallel */

  }

  dtime(&etime);
  if(myid==0 && measure_time){
    printf("Time for part2 of force#4=%18.5f\n",etime-stime);fflush(stdout);
  } 

  /*****************************************}********************** 
      THE FIRST CASE:
      In case of I=i or I=j 
      for d [ \sum_k <i|k>ek<k|j> ]/dRI  
  ****************************************************************/

  /*******************************************************
   *******************************************************
      multiplying overlap integrals WITH COMMUNICATION
   *******************************************************
   *******************************************************/

  MPI_Barrier(mpi_comm_level1);
  dtime(&stime);

  for (ID=0; ID<numprocs; ID++){
    send_progress[ID] = 0;
    receive_progress[ID] = 0;
  }

  do {

    /***********************************
             set the size of data
    ************************************/

    for (ID=0; ID<numprocs; ID++){

      IDS = (myid + ID) % numprocs;
      IDR = (myid - ID + numprocs) % numprocs;

      /* find the data size to send the block data */

      if ( 0<(F_Snd_Num[IDS]-send_progress[IDS]) ){

	size1 = 0;
	n = send_progress[IDS];

	Mc_AN = Snd_MAN[IDS][n];
	Gc_AN = Snd_GAN[IDS][n];
	Cwan = WhatSpecies[Gc_AN]; 
	tno1 = Spe_Total_NO[Cwan];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];        
	  Hwan = WhatSpecies[Gh_AN];
	  tno2 = (List_YOUSO[35]+1)*(List_YOUSO[35]+1)*List_YOUSO[34];
	  size1 += tno1*tno2; 
	}
 
	Snd_DS_VNA_Size[IDS] = size1;
	MPI_Isend(&size1, 1, MPI_INT, IDS, tag, mpi_comm_level1, &request);
      }
      else{
	Snd_DS_VNA_Size[IDS] = 0;
      }

      /* receiving of the size of the data */

      if ( 0<(F_Rcv_Num[IDR]-receive_progress[IDR]) ){
	MPI_Recv(&size2, 1, MPI_INT, IDR, tag, mpi_comm_level1, &stat);
	Rcv_DS_VNA_Size[IDR] = size2;
      }
      else{
	Rcv_DS_VNA_Size[IDR] = 0;
      }

      if ( 0<(F_Snd_Num[IDS]-send_progress[IDS]) )  MPI_Wait(&request,&stat);

    } /* ID */

      /***********************************
                data transfer
      ************************************/

    for (ID=0; ID<numprocs; ID++){

      IDS = (myid + ID) % numprocs;
      IDR = (myid - ID + numprocs) % numprocs;

      /******************************
             sending of the data 
      ******************************/

      if ( 0<(F_Snd_Num[IDS]-send_progress[IDS]) ){

	size1 = Snd_DS_VNA_Size[IDS];

	/* allocation of the array */

	tmp_array = OpenMX_TransferBuffer(&projector_transfer,OPENMX_EXCHANGE_SEND,size1);

	/* multidimentional array to the vector array */

	num = 0;
	n = send_progress[IDS];

	Mc_AN = Snd_MAN[IDS][n];
	Gc_AN = Snd_GAN[IDS][n];
	Cwan = WhatSpecies[Gc_AN]; 
	tno1 = Spe_Total_NO[Cwan];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];        
	  Hwan = WhatSpecies[Gh_AN];
	  tno2 = (List_YOUSO[35]+1)*(List_YOUSO[35]+1)*List_YOUSO[34];

	  for (i=0; i<tno1; i++){
	    for (j=0; j<tno2; j++){
	      tmp_array[num] = DS_VNA[0][Mc_AN][h_AN][i][j];
	      num++;
	    } 
	  } 
	}

	MPI_Isend(&tmp_array[0], size1, MPI_Type_DS_VNA, IDS, tag, mpi_comm_level1, &request);
      }

      /******************************
          receiving of the block data
      ******************************/

      if ( 0<(F_Rcv_Num[IDR]-receive_progress[IDR]) ){
        
	size2 = Rcv_DS_VNA_Size[IDR];
	tmp_array2 = OpenMX_TransferBuffer(&projector_transfer,OPENMX_EXCHANGE_RECEIVE,size2);
	MPI_Recv(&tmp_array2[0], size2, MPI_Type_DS_VNA, IDR, tag, mpi_comm_level1, &stat);

	/* store */

	num = 0;
	n = receive_progress[IDR];
	Original_Mc_AN = F_TopMAN[IDR] + n;
	Gc_AN = Rcv_GAN[IDR][n];
	Cwan = WhatSpecies[Gc_AN];
	tno1 = Spe_Total_NO[Cwan];

	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  tno2 = (List_YOUSO[35]+1)*(List_YOUSO[35]+1)*List_YOUSO[34];

	  for (i=0; i<tno1; i++){
	    for (j=0; j<tno2; j++){
	      DS_VNA[0][Matomnum+1][h_AN][i][j] = tmp_array2[num];
	      num++;
	    }
	  }
	}

	/* free tmp_array2 */


	/*****************************************
               multiplying overlap integrals
	*****************************************/

#pragma omp parallel shared(scratch,List_YOUSO,time_per_atom,Gxyz,CDM0,SpinP_switch,CntHVNA2,HVNA2,DS_VNA,Cnt_switch,RMI1,Original_Mc_AN,IDR,Rcv_GAN,receive_progress,Spe_Total_CNO,F_G2M,natn,FNAN,WhatSpecies,M2G,Matomnum,stress_threads) private(OMPID,Nthrds,Nprocs,Stime_atom,Etime_atom,dEx,dEy,dEz,Gc_AN,Mc_AN,Cwan,fan,h_AN,Gh_AN,Mh_AN,Hwan,ian,jg,j0,jg0,Mj_AN0,po2,q_AN,Gq_AN,Mq_AN,Qwan,jan,kl,HVNAx,HVNAy,HVNAz,HVNA_sts,i,j,k,m,n,pref)
	{

	  /* allocation of array */

	  OpenMX_Stress_VnaPrepare(scratch,List_YOUSO[7],&HVNAx,&HVNAy,&HVNAz,&HVNA_sts);

	  /* get info. on OpenMP */ 

	  OMPID = omp_get_thread_num();
	  Nthrds = omp_get_num_threads();
	  Nprocs = omp_get_num_procs();

	  for (Mc_AN=(OMPID*Matomnum/Nthrds+1); Mc_AN<((OMPID+1)*Matomnum/Nthrds+1); Mc_AN++){

	    dtime(&Stime_atom);

	    dEx = 0.0; 
	    dEy = 0.0; 
	    dEz = 0.0; 

	    Gc_AN = M2G[Mc_AN];
	    Cwan = WhatSpecies[Gc_AN];
	    fan = FNAN[Gc_AN];

	    h_AN = 0;
	    Gh_AN = natn[Gc_AN][h_AN];
	    Mh_AN = F_G2M[Gh_AN];
	    Hwan = WhatSpecies[Gh_AN];
	    ian = Spe_Total_CNO[Hwan];

	    n = receive_progress[IDR];
	    jg = Rcv_GAN[IDR][n];

	    for (j0=0; j0<=fan; j0++){

	      jg0 = natn[Gc_AN][j0];
	      Mj_AN0 = F_G2M[jg0];

	      po2 = 0;
	      if (Original_Mc_AN==Mj_AN0){
		po2 = 1;
		q_AN = j0;
	      }

	      if (po2==1){

		Gq_AN = natn[Gc_AN][q_AN];
		Mq_AN = F_G2M[Gq_AN];
		Qwan = WhatSpecies[Gq_AN];
		jan = Spe_Total_CNO[Qwan];
		kl = RMI1[Mc_AN][h_AN][q_AN];

		if (Cnt_switch==0) {
		  dHVNA(vna, 0,Mc_AN,h_AN,q_AN,DS_VNA,HVNA2,HVNA3,&HVNAx,&HVNAy,&HVNAz,&HVNA_sts);
		}
		else { 
		  dHVNA(vna, 0,Mc_AN,h_AN,q_AN,DS_VNA,CntHVNA2,CntHVNA3,&HVNAx,&HVNAy,&HVNAz,&HVNA_sts);
		}

		/* contribution of force = Trace(CDM0*dH) */
		/* spin non-polarization */

		if (SpinP_switch==0){

                  if (q_AN==h_AN) pref = 2.0;
                  else            pref = 4.0; 

                  /* for forces */                 

		  for (i=0; i<ian; i++){
		    for (j=0; j<jan; j++){
		      dEx += pref*CDM0[0][Mh_AN][kl][i][j]*openmx_solver_real_matrix_row(&HVNAx,i)[j];
		      dEy += pref*CDM0[0][Mh_AN][kl][i][j]*openmx_solver_real_matrix_row(&HVNAy,i)[j];
		      dEz += pref*CDM0[0][Mh_AN][kl][i][j]*openmx_solver_real_matrix_row(&HVNAz,i)[j];
		    }
		  }

		  /* for stress */                 

		  for (m=1; m<=3; m++){
		    for (n=1; n<=3; n++){
		      for (i=0; i<ian; i++){
			for (j=0; j<jan; j++){
			  openmx_solver_real_matrix_row(&stress_threads,OMPID)[(m-1)*3+(n-1)] += pref*CDM0[0][Mh_AN][kl][i][j]*OpenMX_Stress_VnaStrainRow(&HVNA_sts,m,n,i)[j];
			}
		      }
		    }
		  }

		}

		/* else */

		else{

  		  if (q_AN==h_AN) pref = 1.0;
		  else            pref = 2.0; 

                  /* for forces */                 

		  for (i=0; i<ian; i++){
		    for (j=0; j<jan; j++){

		      dEx += pref*(  CDM0[0][Mh_AN][kl][i][j]
			           + CDM0[1][Mh_AN][kl][i][j] )*openmx_solver_real_matrix_row(&HVNAx,i)[j];
		      dEy += pref*(  CDM0[0][Mh_AN][kl][i][j]
			           + CDM0[1][Mh_AN][kl][i][j] )*openmx_solver_real_matrix_row(&HVNAy,i)[j];
		      dEz += pref*(  CDM0[0][Mh_AN][kl][i][j]
			           + CDM0[1][Mh_AN][kl][i][j] )*openmx_solver_real_matrix_row(&HVNAz,i)[j];
		    }
		  }

                  /* for stress */                 

		  for (m=1; m<=3; m++){
		    for (n=1; n<=3; n++){
		      for (i=0; i<ian; i++){
			for (j=0; j<jan; j++){
			  openmx_solver_real_matrix_row(&stress_threads,OMPID)[(m-1)*3+(n-1)] += pref*( CDM0[0][Mh_AN][kl][i][j]*OpenMX_Stress_VnaStrainRow(&HVNA_sts,m,n,i)[j]
                                            + CDM0[1][Mh_AN][kl][i][j]*OpenMX_Stress_VnaStrainRow(&HVNA_sts,m,n,i)[j]);
			}
		      }
		    }
		  }

		} /* else */

	      } /* if (po2==1) */
	    } /* j0 */             

	    /* force from #4B */

	    Gxyz[Gc_AN][41] += dEx;
	    Gxyz[Gc_AN][42] += dEy;
	    Gxyz[Gc_AN][43] += dEz;

	    /* timing */
	    dtime(&Etime_atom);
	    time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

	  } /* Mc_AN */

	  /* freeing of array */

	} /* #pragma omp parallel */

	/********************************************
              increment of receive_progress[IDR]
	********************************************/

	receive_progress[IDR]++;

      } /* if ( 0<(F_Snd_Num[IDS]-send_progress[IDS]) ) */

      if ( 0<(F_Snd_Num[IDS]-send_progress[IDS]) ) {

	MPI_Wait(&request,&stat);
	  /* freeing of array */

	/********************************************
             increment of send_progress[IDS]
	********************************************/

	send_progress[IDS]++;
      } 

    } /* ID */

    /*****************************************************
      check whether all the communications have finished
    *****************************************************/

    po = 0;
    for (ID=0; ID<numprocs; ID++){

      IDS = (myid + ID) % numprocs;
      IDR = (myid - ID + numprocs) % numprocs;

      if ( 0<(F_Snd_Num[IDS]-send_progress[IDS]) ) po += F_Snd_Num[IDS]-send_progress[IDS];
      if ( 0<(F_Rcv_Num[IDR]-receive_progress[IDR]) ) po += F_Rcv_Num[IDR]-receive_progress[IDR];
    }

  } while (po!=0);

  dtime(&etime);
  if(myid==0 && measure_time){
    printf("Time for part3 of force#4=%18.5f\n",etime-stime);fflush(stdout);
  } 

  /*******************************************************
   *******************************************************
      THE FIRST CASE:
      multiplying overlap integrals WITHOUT COMMUNICATION
   *******************************************************
   *******************************************************/

  dtime(&stime);

#pragma omp parallel shared(scratch,time_per_atom,Gxyz,CDM0,SpinP_switch,CntHVNA2,HVNA2,DS_VNA,Cnt_switch,RMI1,FNAN,Spe_Total_CNO,WhatSpecies,F_G2M,natn,M2G,Matomnum,List_YOUSO,stress_threads) private(HVNAx,HVNAy,HVNAz,HVNA_sts,OMPID,Nthrds,Nprocs,Mc_AN,Stime_atom,Etime_atom,dEx,dEy,dEz,Gc_AN,h_AN,Gh_AN,Mh_AN,Hwan,ian,q_AN,Gq_AN,Mq_AN,Qwan,jan,kl,i,j,k,m,n,kk,pref)
  {

    /* allocation of array */

    OpenMX_Stress_VnaPrepare(scratch,List_YOUSO[7],&HVNAx,&HVNAy,&HVNAz,&HVNA_sts);

    /* get info. on OpenMP */ 

    OMPID = omp_get_thread_num();
    Nthrds = omp_get_num_threads();
    Nprocs = omp_get_num_procs();

    for (Mc_AN=(OMPID*Matomnum/Nthrds+1); Mc_AN<((OMPID+1)*Matomnum/Nthrds+1); Mc_AN++){

      dtime(&Stime_atom);

      dEx = 0.0; 
      dEy = 0.0; 
      dEz = 0.0; 

      Gc_AN = M2G[Mc_AN];
      h_AN = 0;
      Gh_AN = natn[Gc_AN][h_AN];
      Mh_AN = F_G2M[Gh_AN];
      Hwan = WhatSpecies[Gh_AN];
      ian = Spe_Total_CNO[Hwan];

      for (q_AN=h_AN; q_AN<=FNAN[Gc_AN]; q_AN++){

	Gq_AN = natn[Gc_AN][q_AN];
	Mq_AN = F_G2M[Gq_AN];

	if (Mq_AN<=Matomnum){

	  Qwan = WhatSpecies[Gq_AN];
	  jan = Spe_Total_CNO[Qwan];
	  kl = RMI1[Mc_AN][h_AN][q_AN];

	  if (Cnt_switch==0) {
	    dHVNA(vna, 0,Mc_AN,h_AN,q_AN,DS_VNA,HVNA2,HVNA3,&HVNAx,&HVNAy,&HVNAz,&HVNA_sts);
	  }
	  else { 
	    dHVNA(vna, 0,Mc_AN,h_AN,q_AN,DS_VNA,CntHVNA2,CntHVNA3,&HVNAx,&HVNAy,&HVNAz,&HVNA_sts);
	  }

	  if (SpinP_switch==0){

            if (q_AN==h_AN) pref = 2.0;
            else            pref = 4.0; 

            /* for forces */                 

	    for (i=0; i<ian; i++){
	      for (j=0; j<jan; j++){
		dEx += pref*CDM0[0][Mh_AN][kl][i][j]*openmx_solver_real_matrix_row(&HVNAx,i)[j];
		dEy += pref*CDM0[0][Mh_AN][kl][i][j]*openmx_solver_real_matrix_row(&HVNAy,i)[j];
		dEz += pref*CDM0[0][Mh_AN][kl][i][j]*openmx_solver_real_matrix_row(&HVNAz,i)[j];
	      }
	    }

	    /* for stress */                 

	    for (m=1; m<=3; m++){
	      for (n=1; n<=3; n++){
		for (i=0; i<ian; i++){
		  for (j=0; j<jan; j++){
		    openmx_solver_real_matrix_row(&stress_threads,OMPID)[(m-1)*3+(n-1)] += pref*CDM0[0][Mh_AN][kl][i][j]*OpenMX_Stress_VnaStrainRow(&HVNA_sts,m,n,i)[j];
		  }
		}
	      }
	    }

	  }

	  /* else */

	  else{

	    if (q_AN==h_AN) pref = 1.0;
	    else            pref = 2.0; 

            /* for forces */                 

	    for (i=0; i<ian; i++){
	      for (j=0; j<jan; j++){

		dEx += pref*(  CDM0[0][Mh_AN][kl][i][j]
			     + CDM0[1][Mh_AN][kl][i][j] )*openmx_solver_real_matrix_row(&HVNAx,i)[j];
		dEy += pref*(  CDM0[0][Mh_AN][kl][i][j]
			     + CDM0[1][Mh_AN][kl][i][j] )*openmx_solver_real_matrix_row(&HVNAy,i)[j];
		dEz += pref*(  CDM0[0][Mh_AN][kl][i][j]
			     + CDM0[1][Mh_AN][kl][i][j] )*openmx_solver_real_matrix_row(&HVNAz,i)[j];
	      }
	    }

	    /* for stress */                 

	    for (m=1; m<=3; m++){
	      for (n=1; n<=3; n++){
		for (i=0; i<ian; i++){
		  for (j=0; j<jan; j++){
		    openmx_solver_real_matrix_row(&stress_threads,OMPID)[(m-1)*3+(n-1)] += pref*( CDM0[0][Mh_AN][kl][i][j]*OpenMX_Stress_VnaStrainRow(&HVNA_sts,m,n,i)[j]
                                            + CDM0[1][Mh_AN][kl][i][j]*OpenMX_Stress_VnaStrainRow(&HVNA_sts,m,n,i)[j]);
		  }
		}
	      }
	    }

	  }
	}
      }

      /* force from #4B */

      Gxyz[Gc_AN][41] += dEx;
      Gxyz[Gc_AN][42] += dEy;
      Gxyz[Gc_AN][43] += dEz;

      /* timing */
      dtime(&Etime_atom);
      time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

    } /* Mc_AN */

      /* freeing of array */

  } /* #pragma omp parallel */

  /* summing up results calculated by OpenMP threads */

  for(i=0; i<Nthrds0; i++){
    for(j=0; j<9; j++){
      My_Stress[j] += openmx_solver_real_matrix_row(&stress_threads,i)[j];
    }
  }

  /* MPI_Allreduce: My_Stress */

  MPI_Allreduce(My_Stress, Stress, 9, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

  for (i=0; i<9; i++){
    Stress[i] = Stress[i]*(double)F_VNA_flag;
  }

  /* add stress coming from the projector expansion of VNA to Stress_Tensor */

  for (i=0; i<9; i++){
    Stress_Tensor[i] += Stress[i];
  }

  /* show #4 of stress */
  
  if (myid==Host_ID && 1<level_stdout){
    printf("  Stress: projector expansion of VNA \n");fflush(stdout);
    for (i = 0; i<3; i++){
      for (j = 0; j<3; j++){
	printf("%16.8f ", Stress[3*i+j]);
      }
      printf("\n");fflush(stdout);
    }
  }

  dtime(&etime);
  if(myid==0 && measure_time){
    printf("Time for part4 of force#4=%18.5f\n",etime-stime);fflush(stdout);
  } 
        
  /***********************************
            freeing of arrays 
  ************************************/


}



void Stress_core(OpenMXSolverInvocation *scratch)
{
  /****************************************************
                       Ecore
  ****************************************************/

  int Mc_AN,Gc_AN,Cwan,Rn,h_AN,Gh_AN,Hwan;
  int i,spin,spinmax;
  double My_Ecore,Ecore,Zc,Zh,dum,dum2;
  double *My_Ecore_threads;
  double TmpEcore,dEx,dEy,dEz,r,lx,ly,lz;
  int numprocs,myid;
  double Stime_atom,Etime_atom;
  /* for OpenMP */
  int OMPID,Nthrds,Nthrds0,Nprocs,Nloop;

  /* begin(yshiihara) */
  /* Stress */
  int j;
  OpenMXSolverRealTensor stress_threads;
  double    Stress[9];
  double My_Stress[9];
  double      sumt[9];
  {
    int i;
    for(i=0; i<9; i++) {
      Stress[i]    = 0.0;
      My_Stress[i] = 0.0;
      sumt[i]      = 0.0;
    }
  }
  /* end(yshiihara) */

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /* get Nthrds0 */  
  /* Dynamic teams may grow after a separate probe region. */
  Nthrds0 = omp_get_max_threads();
  
  /* allocation of array */
  /* begin(yshiihara) */ 

  OpenMX_Stress_ThreadsPrepare(scratch,Nthrds0,&stress_threads);
  for (i=0; i<Nthrds0; i++){

    for (j=0; j<9; j++) openmx_solver_real_matrix_row(&stress_threads,i)[j] = 0.0;
  }
  /* end(yshiihara) */
  
  /* begin(yshiihara) */ 
#pragma omp parallel shared(level_stdout,time_per_atom,atv,Gxyz,Dis,ncn,natn,FNAN,Spe_Core_Charge,WhatSpecies,M2G,Matomnum,My_Ecore_threads,DecEscc,stress_threads) private(OMPID,Nthrds,Nprocs,Mc_AN,Stime_atom,Gc_AN,Cwan,Zc,dEx,dEy,dEz,h_AN,Gh_AN,Rn,Hwan,Zh,r,lx,ly,lz,dum,dum2,Etime_atom,TmpEcore,i,j,sumt)
  {
    /* end(yshiihara) */
    
    /* get info. on OpenMP */ 
    
    OMPID = omp_get_thread_num();
    Nthrds = omp_get_num_threads();
    Nprocs = omp_get_num_procs();
    
    for (Mc_AN=(OMPID*Matomnum/Nthrds+1); Mc_AN<((OMPID+1)*Matomnum/Nthrds+1); Mc_AN++){
      
      dtime(&Stime_atom);
      
      Gc_AN = M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];
      Zc = Spe_Core_Charge[Cwan];
      dEx = 0.0;
      dEy = 0.0;
      dEz = 0.0;
      TmpEcore = 0.0;
      for (i=0; i<9; i++){
	sumt[i] = 0.0;
      }
      
      for (h_AN=1; h_AN<=FNAN[Gc_AN]; h_AN++){
	
	Gh_AN = natn[Gc_AN][h_AN];
	Rn = ncn[Gc_AN][h_AN];
	Hwan = WhatSpecies[Gh_AN];
	Zh = Spe_Core_Charge[Hwan];
	r = Dis[Gc_AN][h_AN];
	
	/* for empty atoms or finite elemens basis */
	if (r<1.0e-10) r = 1.0e-10;
	
	lx = (Gxyz[Gc_AN][1] - Gxyz[Gh_AN][1] - atv[Rn][1])/r;
	ly = (Gxyz[Gc_AN][2] - Gxyz[Gh_AN][2] - atv[Rn][2])/r;
	lz = (Gxyz[Gc_AN][3] - Gxyz[Gh_AN][3] - atv[Rn][3])/r;
	dum = Zc*Zh/r;
	dum2 = dum/r;
        TmpEcore += dum; 
	dEx = dEx - lx*dum2;
	dEy = dEy - ly*dum2;
	dEz = dEz - lz*dum2;
	/*begin(yshiihara) */ 
	sumt[0] -= (Gxyz[Gc_AN][1] - Gxyz[Gh_AN][1] - atv[Rn][1])*lx*dum2;
	sumt[1] -= (Gxyz[Gc_AN][1] - Gxyz[Gh_AN][1] - atv[Rn][1])*ly*dum2;
	sumt[2] -= (Gxyz[Gc_AN][1] - Gxyz[Gh_AN][1] - atv[Rn][1])*lz*dum2;
	sumt[3] -= (Gxyz[Gc_AN][2] - Gxyz[Gh_AN][2] - atv[Rn][2])*lx*dum2;
	sumt[4] -= (Gxyz[Gc_AN][2] - Gxyz[Gh_AN][2] - atv[Rn][2])*ly*dum2;
	sumt[5] -= (Gxyz[Gc_AN][2] - Gxyz[Gh_AN][2] - atv[Rn][2])*lz*dum2;
	sumt[6] -= (Gxyz[Gc_AN][3] - Gxyz[Gh_AN][3] - atv[Rn][3])*lx*dum2;
	sumt[7] -= (Gxyz[Gc_AN][3] - Gxyz[Gh_AN][3] - atv[Rn][3])*ly*dum2;
	sumt[8] -= (Gxyz[Gc_AN][3] - Gxyz[Gh_AN][3] - atv[Rn][3])*lz*dum2;
	/* end(yshiihara) */
	
      } /* h_AN */
      
      /****************************************************
                        #6 of force
         Contribution from the core-core repulsions
      ****************************************************/
      /*begin(yshiihara) */ 
      for (i=0; i<9; i++){
	openmx_solver_real_matrix_row(&stress_threads,OMPID)[i] += sumt[i];
      }
      /*end(yshiihara) */

      dtime(&Etime_atom);
      time_per_atom[Gc_AN] += Etime_atom - Stime_atom;
    }

  } /* #pragma omp parallel */

  for (Nloop=0; Nloop<Nthrds0; Nloop++){
    for(i=0; i<9; i++){
      My_Stress[i] += 0.5*openmx_solver_real_matrix_row(&stress_threads,Nloop)[i];
    }
  }
  
  MPI_Allreduce(&My_Stress[0],&Stress[0],9,MPI_DOUBLE,MPI_SUM,mpi_comm_level1);

  /* add stress coming from core-core to Stress_Tensor */

  for (i=0; i<9; i++){
    Stress_Tensor[i] += Stress[i];
  }

  /* show stress coming from core-core */
    
  if (myid==Host_ID && 1<level_stdout){
    printf(" Components of stress (Core-core): \n");fflush(stdout);
    for (i = 0; i<3; i++){
      for (j = 0; j<3; j++){
	printf("%16.8f ", Stress[3*i+j]);
      }
      printf("\n");fflush(stdout);
    }
  }
  
  /* end(yshiihara) */

  /* freeing of array */
  
  /* begin(yshiihara) */
  /* free(My_Ecore_threads); */

  /* end(yshiihara) */
}




void Stress_H0(OpenMXSolverInvocation *scratch,OpenMXCoreContext *core,const OpenMXAtomicViews *atomic)
{
  OpenMXNeutralGridTables tables;
  char grid_error[256]={0};

  /****************************************************
              EH0 = -1/2\int n^a(r) V^a_H dr
  ****************************************************/

  int Mc_AN,Gc_AN,h_AN,Gh_AN,num,gnum,i;
  int wan,wan1,wan2,Nd,n1,n2,n3,spin,spinmax;
  double bc,dx,x,y,z,r1,r2,rho0,xx;
  double Scale_Grid_Ecut,TmpEH0;
  double EH0ij[4],My_EH0,EH0,tmp0;
  double *Fx,*Fy,*Fz,*g0;
  double dEx,dEy,dEz,Dx,Sx;
  double Z1,Z2,factor;
  double My_dEx,My_dEy,My_dEz;
  int numprocs,myid,ID;
  double stime,etime;
  double Stime_atom, Etime_atom;
  /* for OpenMP */
  int OMPID,Nthrds,Nthrds0,Nprocs,Nloop;
  /* begin(yshiihara) */
  /*double *My_EH0_threads;*/
  /* Stress */
  int j;
  OpenMXSolverRealTensor stress_threads;
  double    Stress[9];
  double My_Stress[9];
  double      sumt[9];
  int    Rn;
  double lx,ly,lz;
  
  int MD_iter;
  
  for(i=0; i<9; i++) {
    Stress[i]    = 0.0;
    My_Stress[i] = 0.0;
  }
  /* end(yshiihara) */
  
  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /****************************************************
             Set of atomic density on grids
  ****************************************************/

  /*(yshiihara) Following part can be neglected because it must be already performed in Total_Energy.c. */
  MD_iter = 1; 
  if (MD_iter==1){
    
    dtime(&stime);

    Scale_Grid_Ecut = 16.0*600.0;
    
    /* estimate the size of an array g0 */
    
    Max_Nd = 0;
    for (wan=0; wan<SpeciesNum; wan++){
      atomic->species.species_index[wan] = wan;
      bc = atomic->species.cutoff[wan];
      dx = PI/sqrt(Scale_Grid_Ecut);
      Nd = 2*(int)(bc/dx) + 1;
      if (Max_Nd<Nd) Max_Nd = Nd;
    }
    
    /* estimate sizes of arrays GridX,Y,Z_EH0, Arho_EH0, and Wt_EH0 */

    Max_TGN_EH0 = 0;
    for (wan=0; wan<SpeciesNum; wan++){
      
      atomic->species.species_index[wan] = wan;
      bc = atomic->species.cutoff[wan];
      dx = PI/sqrt(Scale_Grid_Ecut);

      Nd = 2*(int)(bc/dx) + 1;
      dx = 2.0*bc/(double)(Nd-1);
      gnum = Nd*CoarseGL_Mesh;

      if (Max_TGN_EH0<gnum) Max_TGN_EH0 = gnum;

      if (2<=level_stdout){
        printf("<Calc_EH0> A spe=%2d 1D-grids=%2d 3D-grids=%2d\n",wan,Nd,gnum);fflush(stdout);
      }
    }

    /* estimate sizes of arrays GridX,Y,Z_EH0, Arho_EH0, and Wt_EH0 */

    Max_TGN_EH0 = 0;
    for (wan=0; wan<SpeciesNum; wan++){

      atomic->species.species_index[wan] = wan;
      bc = atomic->species.cutoff[wan];
      dx = PI/sqrt(Scale_Grid_Ecut);

      Nd = 2*(int)(bc/dx) + 1;
      dx = 2.0*bc/(double)(Nd-1);
      gnum = Nd*CoarseGL_Mesh;

      if (Max_TGN_EH0<gnum) Max_TGN_EH0 = gnum;

      if (2<=level_stdout){
        printf("<Calc_EH0> A spe=%2d 1D-grids=%2d 3D-grids=%2d\n",wan,Nd,gnum);fflush(stdout);
      }
    }
    /* allocation of arrays GridX,Y,Z_EH0, Arho_EH0, and Wt_EH0 */

    Max_TGN_EH0 += 10;
    Allocate_Arrays_with_context(4,core);
    if (!openmx_core_neutral_grid_get(core,&tables,grid_error,sizeof(grid_error)) ||
        !tables.values || tables.extent[1]!=(size_t)SpeciesNum ||
        tables.extent[2]!=(size_t)Max_TGN_EH0) {
      fprintf(stderr,"neutral grid view unavailable or shape mismatch: %s\n",grid_error);
      MPI_Abort(mpi_comm_level1,1);
      return;
    }

    /* calculate GridX,Y,Z_EH0 and Wt_EH0 */

#pragma omp parallel shared(scratch,atomic,Max_Nd,level_stdout,TGN_EH0,tables,Scale_Grid_Ecut,dv_EH0,SpeciesNum,CoarseGL_Abscissae,CoarseGL_Weight) private(OMPID,Nthrds,Nprocs,wan,bc,dx,Nd,gnum,n1,n2,n3,x,y,z,tmp0,r1,rho0,g0,Sx,Dx,xx)
    {

      /* allocation of arrays g0 */

      g0=OpenMX_Stress_AxisBuffer(scratch,Max_Nd);

      /* get info. on OpenMP */

      OMPID = omp_get_thread_num();
      Nthrds = omp_get_num_threads();
      Nprocs = omp_get_num_procs();

      for (wan=OMPID*SpeciesNum/Nthrds; wan<(OMPID+1)*SpeciesNum/Nthrds; wan++){

        atomic->species.species_index[wan] = wan;
        bc = atomic->species.cutoff[wan];
        dx = PI/sqrt(Scale_Grid_Ecut);
        Nd = 2*(int)(bc/dx) + 1;
        dx = 2.0*bc/(double)(Nd-1);
        dv_EH0[wan] = dx;

        const OpenMXNeutralQuadratureView quadrature={Nd,CoarseGL_Mesh,bc,dx,PI,
            CoarseGL_Abscissae,CoarseGL_Weight};
        gnum=OpenMX_Neutral_GridFill(atomic,wan,&quadrature,g0,
            openmx_core_neutral_grid_row(&tables,0,wan),
            openmx_core_neutral_grid_row(&tables,1,wan),
            openmx_core_neutral_grid_row(&tables,2,wan),
            openmx_core_neutral_grid_row(&tables,3,wan),
            openmx_core_neutral_grid_row(&tables,4,wan));

        TGN_EH0[wan] = gnum;

        if (2<=level_stdout){
          printf("<Calc_EH0> B spe=%2d 1D-grids=%2d 3D-grids=%2d\n",wan,Nd,gnum);fflush(stdout);
        }

      } /* wan */

      /* free */

	   
    } /* #pragma omp parallel */
    
    dtime(&etime);
    if(myid==0 && measure_time){
      printf("Time for part1 of EH0=%18.5f\n",etime-stime);fflush(stdout);
    }
    
  } /* if (MD_iter==1) */

  /****************************************************                                                                             
    calculation of scaling factors:                                                                                                 
  ****************************************************/

  if (MD_iter==1){

    for (wan1=0; wan1<SpeciesNum; wan1++){

      r1 = atomic->species.cutoff[wan1];
      Z1 = atomic->species.core_charge[wan1];

      for (wan2=0; wan2<SpeciesNum; wan2++){

        /* EH0_TwoCenter_at_Cutoff is parallelized by OpenMP */
        EH0_TwoCenter_at_Cutoff(scratch,&tables,atomic, wan1, wan2, EH0ij);

        r2 = atomic->species.cutoff[wan2];
        Z2 = atomic->species.core_charge[wan2];
        tmp0 = Z1*Z2/(r1+r2);


        if (1.0e-20<fabs(EH0ij[0])){
          EH0_scaling[wan1][wan2] = tmp0/EH0ij[0];
        }
        else{
          EH0_scaling[wan1][wan2] = 0.0;
        }

      }
    }
  }
  /*(yshiihara) above part can be neglected because it must be performed in force part */

  /****************************************************                                                                             
                -1/2\int n^a(r) V^a_H dr                                                                                            
  ****************************************************/

  dtime(&stime);

  /* get Nthrds0 */
  /* Dynamic teams may grow after a separate probe region. */
  Nthrds0 = omp_get_max_threads();

  /* allocation of array */
  /* begin(yshiihara) */
  OpenMX_Stress_ThreadsPrepare(scratch,Nthrds0,&stress_threads);
  for (i=0; i<Nthrds0; i++){

    for (j=0; j<9; j++) openmx_solver_real_matrix_row(&stress_threads,i)[j] = 0.0;
  }
  /* end(yshiihara) */

    /* begin(yshiihara) */
#pragma omp parallel shared(Gxyz,atv,time_per_atom,RMI1,EH0_scaling,natn,ncn,FNAN,WhatSpecies,M2G,Matomnum,stress_threads) private(OMPID,Nthrds,Nprocs,Mc_AN,Stime_atom,Gc_AN,wan1,h_AN,Gh_AN,wan2,factor,Etime_atom,TmpEH0,lx,ly,lz,Rn)
  {
    /* end(yshiihara) */

    double EH0ij[4];    

    /* get info. on OpenMP */ 

    OMPID = omp_get_thread_num();
    Nthrds = omp_get_num_threads();
    Nprocs = omp_get_num_procs();

    for (Mc_AN=(OMPID*Matomnum/Nthrds+1); Mc_AN<((OMPID+1)*Matomnum/Nthrds+1); Mc_AN++){

      dtime(&Stime_atom);

      Gc_AN = M2G[Mc_AN];
      wan1 = WhatSpecies[Gc_AN];
      
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	
	Gh_AN = natn[Gc_AN][h_AN];
	wan2 = WhatSpecies[Gh_AN];
	
	/* begin(yshiihara) */
	Rn = ncn[Gc_AN][h_AN];
	lx = Gxyz[Gc_AN][1] - Gxyz[Gh_AN][1] - atv[Rn][1];
	ly = Gxyz[Gc_AN][2] - Gxyz[Gh_AN][2] - atv[Rn][2];
	lz = Gxyz[Gc_AN][3] - Gxyz[Gh_AN][3] - atv[Rn][3];
	/* end(yshiihara) */
	
	if (h_AN==0) factor = 1.0;
	else         factor = EH0_scaling[wan1][wan2];

	EH0_TwoCenter(&tables,atomic, Gc_AN, h_AN, EH0ij);


	/* begin(yshiihara) */

	openmx_solver_real_matrix_row(&stress_threads,OMPID)[0] -= 0.25*factor*EH0ij[1]*lx;
	openmx_solver_real_matrix_row(&stress_threads,OMPID)[1] -= 0.25*factor*EH0ij[2]*lx;
	openmx_solver_real_matrix_row(&stress_threads,OMPID)[2] -= 0.25*factor*EH0ij[3]*lx;
	openmx_solver_real_matrix_row(&stress_threads,OMPID)[3] -= 0.25*factor*EH0ij[1]*ly;
	openmx_solver_real_matrix_row(&stress_threads,OMPID)[4] -= 0.25*factor*EH0ij[2]*ly;
	openmx_solver_real_matrix_row(&stress_threads,OMPID)[5] -= 0.25*factor*EH0ij[3]*ly;
	openmx_solver_real_matrix_row(&stress_threads,OMPID)[6] -= 0.25*factor*EH0ij[1]*lz;
	openmx_solver_real_matrix_row(&stress_threads,OMPID)[7] -= 0.25*factor*EH0ij[2]*lz;
	openmx_solver_real_matrix_row(&stress_threads,OMPID)[8] -= 0.25*factor*EH0ij[3]*lz;

	/* debug 
	   if (myid==Host_ID && 0<level_stdout){
	   printf(" My_Stress_threads1 : \n");fflush(stdout);
	   for (i = 0; i<3; i++){
	   for (j = 0; j<3; j++){
	   printf("%16.8f ", openmx_solver_real_matrix_row(&stress_threads,OMPID)[3*i+j]);
	   }
	   printf("\n");fflush(stdout);
	   }
	   }
	*/
	/* end(yshiihara) */

	if (h_AN==0) factor = 1.0;
	else         factor = EH0_scaling[wan2][wan1];

	EH0_TwoCenter(&tables,atomic, Gh_AN, RMI1[Mc_AN][h_AN][0], EH0ij);

	/* begin(yshiihara) */

	openmx_solver_real_matrix_row(&stress_threads,OMPID)[0] += 0.25*factor*EH0ij[1]*lx;
	openmx_solver_real_matrix_row(&stress_threads,OMPID)[1] += 0.25*factor*EH0ij[2]*lx;
	openmx_solver_real_matrix_row(&stress_threads,OMPID)[2] += 0.25*factor*EH0ij[3]*lx;
	openmx_solver_real_matrix_row(&stress_threads,OMPID)[3] += 0.25*factor*EH0ij[1]*ly;
	openmx_solver_real_matrix_row(&stress_threads,OMPID)[4] += 0.25*factor*EH0ij[2]*ly;
	openmx_solver_real_matrix_row(&stress_threads,OMPID)[5] += 0.25*factor*EH0ij[3]*ly;
	openmx_solver_real_matrix_row(&stress_threads,OMPID)[6] += 0.25*factor*EH0ij[1]*lz;
	openmx_solver_real_matrix_row(&stress_threads,OMPID)[7] += 0.25*factor*EH0ij[2]*lz;
	openmx_solver_real_matrix_row(&stress_threads,OMPID)[8] += 0.25*factor*EH0ij[3]*lz;

	/* debug
	   if (myid==Host_ID && 0<level_stdout){
	   printf(" My_Stress_threads2 : \n");fflush(stdout);
	   for (i = 0; i<3; i++){
	   for (j = 0; j<3; j++){
	   printf("%16.8f ", openmx_solver_real_matrix_row(&stress_threads,OMPID)[3*i+j]);
	   }
	   printf("\n");fflush(stdout);
	   }
	   }
	*/
	/* end(yshiihara) */

      } /* h_AN */

      dtime(&Etime_atom);
      time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

    } /* Mc_AN */
  } /* #pragma omp parallel */

  /* begin(yshiihara) */

  for(i=0; i<9; i++){
    My_Stress[i]=0.0;
  }
  for(i=0; i<Nthrds0; i++){
    for(j=0; j<9; j++){
      /* printf("myid:%d, i:%d, j:%d \n",myid,i,j);fflush(stdout); */ /*debug*/
      My_Stress[j] += openmx_solver_real_matrix_row(&stress_threads,i)[j];
    }
  }

  /* debug
  if (myid==Host_ID && 0<level_stdout){
    printf(" Components of My_Stress : \n");fflush(stdout);
    for (i = 0; i<3; i++){
      for (j = 0; j<3; j++){
	printf("%16.8f ", My_Stress[3*i+j]);
      }
      printf("\n");fflush(stdout);
    }
  }
  */
  
  MPI_Allreduce(&My_Stress[0],&Stress[0],9,MPI_DOUBLE,MPI_SUM,mpi_comm_level1);

  /* add stress coming from H0 to Stress_Tensor */

  for (i=0; i<9; i++){
    Stress_Tensor[i] += Stress[i];
  }

  /* show stress coming from H0 */
  
  if (myid==Host_ID && 1<level_stdout){
    printf(" Components of stress (H0) : \n");fflush(stdout);
    for (i = 0; i<3; i++){
      for (j = 0; j<3; j++){
	printf("%16.8f ", Stress[3*i+j]);
      }
      printf("\n");fflush(stdout);
    }
  }
  /* end(yshiihara) */

  /* begin(yshiihara) */

  /* freeing of array */
  /* begin(yshiihara) */



  /* measuring elapsed time */

  dtime(&etime);
  if(myid==0 && measure_time){
    printf("Time for part2 of EH0=%18.5f\n",etime-stime);fflush(stdout);
  }

}

void Stress_H1(OpenMXSolverInvocation *scratch)
{ 
  int k1,k2,k3;
  int N2D,GNs,GN,BN_CB;
  int N3[4];
  double time0;
  double tmp0,sk1,sk2,sk3;
  double Gx,Gy,Gz,fac_invG2;
  double TStime,TEtime,etime;
  int numprocs,myid,tag=999,ID,IDS,IDR;
  MPI_Status stat;
  MPI_Request request;

  /* Stress */

  double    Stress[9];
  double My_Stress[9];
  double      sumt[9];
  double      invG4,tmp1;
  int       i,j;

  double   *ReRhok,*ImRhok;

  /* allocation of arrays */  
  
  OpenMXSolverRealTensor reciprocal;
  OpenMX_Stress_ReciprocalPrepare(scratch,My_Max_NumGridB,&reciprocal);
  ReRhok = openmx_solver_real_matrix_row(&reciprocal,0);
  for (i=0; i<My_Max_NumGridB; i++) ReRhok[i] = 0.0;
  ImRhok = openmx_solver_real_matrix_row(&reciprocal,1);
  for (i=0; i<My_Max_NumGridB; i++) ImRhok[i] = 0.0;

  /* initialization of arrays */  
  
  for(i=0; i<9; i++) {
    Stress[i]    = 0.0;
    My_Stress[i] = 0.0;
  }

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /*if (myid==Host_ID && 0<level_stdout){
    printf("<Stress>  Poisson's equation using FFT...\n");
    }*/

  MPI_Barrier(mpi_comm_level1);
  dtime(&TStime);

  /****************************************************
            FFT of difference charge density 
  ****************************************************/

  etime = FFT_Density(0,ReRhok,ImRhok);

  /****************************************************
                       4*PI/G2/N^3
  ****************************************************/

  tmp0 = 1.0/(double)(Ngrid1*Ngrid2*Ngrid3);

  N2D = Ngrid3*Ngrid2;
  GNs = ((myid*N2D+numprocs-1)/numprocs)*Ngrid1;

  for (BN_CB=0; BN_CB<My_NumGridB_CB; BN_CB++){

    GN = BN_CB + GNs;     
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
    invG4 =  tmp0/(Gx*Gx + Gy*Gy + Gz*Gz);
    invG4 = 4.0*PI*invG4*invG4*(double)F_dVHart_flag;
    tmp1  = ReRhok[BN_CB]*ReRhok[BN_CB]+ImRhok[BN_CB]*ImRhok[BN_CB];
    tmp1  = tmp1*Cell_Volume;
      
    if (k1!=0 || k2!=0 || k3!=0){
      My_Stress[0] += invG4*Gx*Gx*tmp1;
      My_Stress[1] += invG4*Gy*Gx*tmp1;
      My_Stress[2] += invG4*Gz*Gx*tmp1;
      My_Stress[3] += invG4*Gx*Gy*tmp1;
      My_Stress[4] += invG4*Gy*Gy*tmp1;
      My_Stress[5] += invG4*Gz*Gy*tmp1;
      My_Stress[6] += invG4*Gx*Gz*tmp1;
      My_Stress[7] += invG4*Gy*Gz*tmp1;
      My_Stress[8] += invG4*Gz*Gz*tmp1;
    }
  }

  MPI_Allreduce(&My_Stress[0],&Stress[0],9,MPI_DOUBLE,MPI_SUM,mpi_comm_level1);

  /* add stress coming from H1 to Stress_Tensor */

  for (i=0; i<9; i++){
    Stress_Tensor[i] += Stress[i];
  }

  /* show stress coming from H1 */

  if (myid==Host_ID && 1<level_stdout){
    printf(" Components of stress (H1_G) : \n");fflush(stdout);
    for (i = 0; i<3; i++){
      for (j = 0; j<3; j++){
	printf("%16.8f ", Stress[3*i+j]);
      }
      printf("\n");fflush(stdout);
    }
  }

  /* freeing of arrays */  



  
  /* for time */
  MPI_Barrier(mpi_comm_level1);
  dtime(&TEtime);
  time0 = TEtime - TStime;
  return;
}




void Stress_HNL(OpenMXSolverInvocation *scratch,const OpenMXAtomicViews *atomic, double *****CDM0, double *****iDM0, const OpenMXElectronicViews *electronic, const OpenMXLocalBasisViews *local_basis)
{
  double ******CntDS_NL=electronic->CntDS_NL;
  double ******DS_NL=electronic->DS_NL;

  /****************************************************
                  Stress arising from HNL
  ****************************************************/

  int Mc_AN,Gc_AN,Cwan,i,j,k,m,h_AN,q_AN,Mq_AN,start_q_AN;
  int jan,kl,km,kl1,Qwan,Gq_AN,Gh_AN,Mh_AN,Hwan,ian;
  int l1,l2,l3,l,LL,Mul1,tno0,ncp,so;
  int tno1,tno2,size1,size2,n,kk,num,po,po1,po2;
  int numprocs,myid,tag=999,ID,IDS,IDR;

  int S_comm_flag,R_comm_flag;
  int SA_num,q,Sc_AN,GSc_AN,smul;
  int Sc_wan,Sh_AN,GSh_AN,Sh_wan;
  int Sh_AN2,fan,jg,j0,jg0,Mj_AN0;
  int Original_Mc_AN;

  double rcutA,rcutB,rcut;
  double dEx,dEy,dEz,ene,pref;
  double Stime_atom, Etime_atom;
  dcomplex ***Hx,***Hy,***Hz;
  OpenMXSolverComplexTensor Hx0,Hy0,Hz0;
  OpenMXSolverComplexTensor Hx1,Hy1,Hz1;
  OpenMXSolverComplexTensor H_sts;
  int *Snd_DS_NL_Size,*Rcv_DS_NL_Size;  

  double *tmp_array;
  double *tmp_array2;

  /* for OpenMP */
  int OMPID,Nthrds,Nthrds0,Nprocs,Nloop,ODNloop;
  int *OneD2h_AN,*OneD2q_AN;
  double *dEx_threads;
  double *dEy_threads;
  double *dEz_threads;
  double stime,etime;
  double stime1,etime1;

  /* Stress */
  OpenMXSolverRealTensor stress_threads;
  double Stress[9],My_Stress[9];

  MPI_Status stat;
  MPI_Request request;

  /* MPI */

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);
  OpenMXProjectorTransfer projector_transfer;
  OpenMX_TransferBeginOwned(OpenMX_Solver_Resources(scratch,"stress:nonlocal:projector-transfer"),
      mpi_comm_level1,numprocs,sizeof(double),&projector_transfer);
  int *send_progress=projector_transfer.buffers.sent;
  int *receive_progress=projector_transfer.buffers.received;


  dtime(&stime);

  /****************************
       allocation of arrays 
  *****************************/

  Snd_DS_NL_Size = projector_transfer.buffers.send_sizes;
  Rcv_DS_NL_Size = projector_transfer.buffers.receive_sizes;

  /* get Nthrds0 */  
  /* Dynamic teams may grow after a separate probe region. */
  Nthrds0 = omp_get_max_threads();

  OpenMX_Stress_ThreadsPrepare(scratch,Nthrds0,&stress_threads);
  for (i=0; i<Nthrds0; i++){

    for (j=0; j<9; j++) openmx_solver_real_matrix_row(&stress_threads,i)[j] = 0.0;
  }

  /* initialization of Stress and My_Stress */

  for (i=0; i<9; i++){
    Stress[i]    = 0.0;
    My_Stress[i] = 0.0;
  }

  /*************************************************************
                    contraction of DS_NL
     Note: DS_NL is overwritten by CntDS_NL in Cont_Matrix1().
  *************************************************************/

  if (Cnt_switch==1){
    for (so=0; so<(SO_switch+1); so++){
      Cont_Matrix1(atomic, DS_NL[so][0],CntDS_NL[so][0], local_basis);
      Cont_Matrix1(atomic, DS_NL[so][1],CntDS_NL[so][1], local_basis);
      Cont_Matrix1(atomic, DS_NL[so][2],CntDS_NL[so][2], local_basis);
      Cont_Matrix1(atomic, DS_NL[so][3],CntDS_NL[so][3], local_basis);
    }
  }

  /*****************************************}********************** 
      THE FIRST CASE:
      In case of I=i or I=j 
      for d [ \sum_k <i|k>ek<k|j> ]/dRI  
  ****************************************************************/

  /*******************************************************
   *******************************************************
       multiplying overlap integrals WITH COMMUNICATION

       In case of I=i or I=j 
       for d [ \sum_k <i|k>ek<k|j> ]/dRI  
   *******************************************************
   *******************************************************/

  MPI_Barrier(mpi_comm_level1);
  dtime(&stime);

  OpenMX_Stress_HnlPrepare(scratch,List_YOUSO[7],&Hx0,&Hy0,&Hz0,&Hx1,&Hy1,&Hz1,&H_sts);

  /* initialization of send_progress and receive_progress */

  for (ID=0; ID<numprocs; ID++){
    send_progress[ID] = 0;
    receive_progress[ID] = 0;
  }

  /* start do loop */

  do {

    /***********************************                                                            
            set the size of data                                                                      
    ************************************/

    for (ID=0; ID<numprocs; ID++){

      IDS = (myid + ID) % numprocs;
      IDR = (myid - ID + numprocs) % numprocs;

      /* find the data size to send the block data */

      if ( 0<(F_Snd_Num[IDS]-send_progress[IDS]) ){

	size1 = 0;
	n = send_progress[IDS];

	Mc_AN = Snd_MAN[IDS][n];
	Gc_AN = Snd_GAN[IDS][n];
	Cwan = WhatSpecies[Gc_AN];
	tno1 = atomic->species.orbitals[Cwan];

	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  tno2 = atomic->species.projector_components[Hwan];
	  size1 += (atomic->species.relativistic[Hwan]+1)*tno1*tno2;
	}

	Snd_DS_NL_Size[IDS] = size1;
	MPI_Isend(&size1, 1, MPI_INT, IDS, tag, mpi_comm_level1, &request);
      }
      else{
	Snd_DS_NL_Size[IDS] = 0;
      }

      /* receiving of the size of the data */

      if ( 0<(F_Rcv_Num[IDR]-receive_progress[IDR]) ){
	MPI_Recv(&size2, 1, MPI_INT, IDR, tag, mpi_comm_level1, &stat);
	Rcv_DS_NL_Size[IDR] = size2;
      }
      else{
	Rcv_DS_NL_Size[IDR] = 0;
      }

      if ( 0<(F_Snd_Num[IDS]-send_progress[IDS]) )  MPI_Wait(&request,&stat);

    } /* ID */

    /***********************************
               data transfer
    ************************************/

    for (ID=0; ID<numprocs; ID++){

      IDS = (myid + ID) % numprocs;
      IDR = (myid - ID + numprocs) % numprocs;

      /******************************
            sending of the data 
      ******************************/

      if ( 0<(F_Snd_Num[IDS]-send_progress[IDS]) ){

	size1 = Snd_DS_NL_Size[IDS];

	/* allocation of the array */

	tmp_array = OpenMX_TransferBuffer(&projector_transfer,OPENMX_EXCHANGE_SEND,size1);

	/* multidimentional array to the vector array */

	num = 0;
	n = send_progress[IDS];

	Mc_AN = Snd_MAN[IDS][n];
	Gc_AN = Snd_GAN[IDS][n];
	Cwan = WhatSpecies[Gc_AN]; 
	tno1 = atomic->species.orbitals[Cwan];

	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];        
	  Hwan = WhatSpecies[Gh_AN];
	  tno2 = atomic->species.projector_components[Hwan];

	  for (so=0; so<=atomic->species.relativistic[Hwan]; so++){
	    for (i=0; i<tno1; i++){
	      for (j=0; j<tno2; j++){
		tmp_array[num] = DS_NL[so][0][Mc_AN][h_AN][i][j];
		num++;
	      } 
	    } 
	  }
	}

	MPI_Isend(&tmp_array[0], size1, MPI_DOUBLE, IDS, tag, mpi_comm_level1, &request);
      }

      /******************************
        receiving of the block data
      ******************************/

      if ( 0<(F_Rcv_Num[IDR]-receive_progress[IDR]) ){
        
	size2 = Rcv_DS_NL_Size[IDR];
	tmp_array2 = OpenMX_TransferBuffer(&projector_transfer,OPENMX_EXCHANGE_RECEIVE,size2);
	MPI_Recv(&tmp_array2[0], size2, MPI_DOUBLE, IDR, tag, mpi_comm_level1, &stat);

	/* store */

	num = 0;
	n = receive_progress[IDR];
	Original_Mc_AN = F_TopMAN[IDR] + n;

	Gc_AN = Rcv_GAN[IDR][n];
	Cwan = WhatSpecies[Gc_AN];
	tno1 = atomic->species.orbitals[Cwan];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  tno2 = atomic->species.projector_components[Hwan];

	  for (so=0; so<=atomic->species.relativistic[Hwan]; so++){
	    for (i=0; i<tno1; i++){
	      for (j=0; j<tno2; j++){
		DS_NL[so][0][Matomnum+1][h_AN][i][j] = tmp_array2[num];
		num++;
	      }
	    }
	  }
	}

	/* free tmp_array2 */


	/*****************************************************************
                           multiplying overlap integrals
	*****************************************************************/

	for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

	  dtime(&Stime_atom);

	  dEx = 0.0; 
	  dEy = 0.0; 
	  dEz = 0.0; 

	  Gc_AN = M2G[Mc_AN];
	  Cwan = WhatSpecies[Gc_AN];
	  fan = FNAN[Gc_AN];

	  h_AN = 0;
	  Gh_AN = natn[Gc_AN][h_AN];
	  Mh_AN = F_G2M[Gh_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  ian = atomic->species.contracted_orbitals[Hwan];

	  n = receive_progress[IDR];
	  jg = Rcv_GAN[IDR][n];

	  for (j0=0; j0<=fan; j0++){

	    jg0 = natn[Gc_AN][j0];
	    Mj_AN0 = F_G2M[jg0];

	    po2 = 0;
	    if (Original_Mc_AN==Mj_AN0){
	      po2 = 1;
	      q_AN = j0;
	    }

	    if (po2==1){

	      Gq_AN = natn[Gc_AN][q_AN];
	      Mq_AN = F_G2M[Gq_AN];
	      Qwan = WhatSpecies[Gq_AN];
	      jan = atomic->species.contracted_orbitals[Qwan];
	      kl = RMI1[Mc_AN][h_AN][q_AN];

              dHNL(atomic, 0,Mc_AN,h_AN,q_AN,DS_NL,&Hx0,&Hy0,&Hz0,&H_sts, electronic);

	      /* contribution of force = Trace(CDM0*dH) */
	      /* spin non-polarization */

	      if (SpinP_switch==0){

                if (q_AN==h_AN) pref = 2.0;
                else            pref = 4.0; 

                /* for forces */                 

		for (i=0; i<ian; i++){
		  for (j=0; j<jan; j++){

		    dEx += pref*CDM0[0][Mh_AN][kl][i][j]*openmx_solver_complex_plane_row(&Hx0,0,i)[j].r;
		    dEy += pref*CDM0[0][Mh_AN][kl][i][j]*openmx_solver_complex_plane_row(&Hy0,0,i)[j].r;
		    dEz += pref*CDM0[0][Mh_AN][kl][i][j]*openmx_solver_complex_plane_row(&Hz0,0,i)[j].r;
		  }
		}

                /* for stress */                 

                for (m=1; m<=3; m++){
		  for (n=1; n<=3; n++){
		    for (i=0; i<ian; i++){
		      for (j=0; j<jan; j++){
			My_Stress[(m-1)*3+(n-1)] += pref*CDM0[0][Mh_AN][kl][i][j]*OpenMX_Stress_HnlStrainRow(&H_sts,0,m,n,i)[j].r;
		      }
		    }
		  }
		}
	      }

	      /* collinear spin polarized or non-colliear without SO and LDA+U */

	      else if (SpinP_switch==1 || (SpinP_switch==3 && SO_switch==0 && Hub_U_switch==0
		   && Constraint_NCS_switch==0 && Zeeman_NCS_switch==0 && Zeeman_NCO_switch==0)){ 

		if (q_AN==h_AN) pref = 1.0;
		else            pref = 2.0; 

                /* for forces */                 

		for (i=0; i<atomic->species.contracted_orbitals[Hwan]; i++){
		  for (j=0; j<atomic->species.contracted_orbitals[Qwan]; j++){

		    dEx += pref*(  CDM0[0][Mh_AN][kl][i][j]*openmx_solver_complex_plane_row(&Hx0,0,i)[j].r
			         + CDM0[1][Mh_AN][kl][i][j]*openmx_solver_complex_plane_row(&Hx0,1,i)[j].r);
		    dEy += pref*(  CDM0[0][Mh_AN][kl][i][j]*openmx_solver_complex_plane_row(&Hy0,0,i)[j].r
				 + CDM0[1][Mh_AN][kl][i][j]*openmx_solver_complex_plane_row(&Hy0,1,i)[j].r);
		    dEz += pref*(  CDM0[0][Mh_AN][kl][i][j]*openmx_solver_complex_plane_row(&Hz0,0,i)[j].r
				 + CDM0[1][Mh_AN][kl][i][j]*openmx_solver_complex_plane_row(&Hz0,1,i)[j].r);
		  }
		}

                /* for stress */                 

                for (m=1; m<=3; m++){
		  for (n=1; n<=3; n++){
		    for (i=0; i<ian; i++){
		      for (j=0; j<jan; j++){
			My_Stress[(m-1)*3+(n-1)] += pref*( CDM0[0][Mh_AN][kl][i][j]*OpenMX_Stress_HnlStrainRow(&H_sts,0,m,n,i)[j].r
	                                                 + CDM0[1][Mh_AN][kl][i][j]*OpenMX_Stress_HnlStrainRow(&H_sts,1,m,n,i)[j].r);
		      }
		    }
		  }
		}

	      }

	      /* spin non-collinear with spin-orbit coupling or with LDA+U */

	      else if ( SpinP_switch==3 && (SO_switch==1 || (Hub_U_switch==1 && F_U_flag==1) 
                     || 1<=Constraint_NCS_switch || Zeeman_NCS_switch==1 || Zeeman_NCO_switch==1)){

                if (q_AN==h_AN){

                  /* for forces */ 

		  for (i=0; i<atomic->species.contracted_orbitals[Hwan]; i++){
		    for (j=0; j<atomic->species.contracted_orbitals[Qwan]; j++){

		      dEx +=
			  CDM0[0][Mh_AN][kl][i][j]*openmx_solver_complex_plane_row(&Hx0,0,i)[j].r
			- iDM0[0][Mh_AN][kl][i][j]*openmx_solver_complex_plane_row(&Hx0,0,i)[j].i
			+ CDM0[1][Mh_AN][kl][i][j]*openmx_solver_complex_plane_row(&Hx0,1,i)[j].r
			- iDM0[1][Mh_AN][kl][i][j]*openmx_solver_complex_plane_row(&Hx0,1,i)[j].i
		    + 2.0*CDM0[2][Mh_AN][kl][i][j]*openmx_solver_complex_plane_row(&Hx0,2,i)[j].r
		    - 2.0*CDM0[3][Mh_AN][kl][i][j]*openmx_solver_complex_plane_row(&Hx0,2,i)[j].i;

		      dEy += 
			  CDM0[0][Mh_AN][kl][i][j]*openmx_solver_complex_plane_row(&Hy0,0,i)[j].r
			- iDM0[0][Mh_AN][kl][i][j]*openmx_solver_complex_plane_row(&Hy0,0,i)[j].i
			+ CDM0[1][Mh_AN][kl][i][j]*openmx_solver_complex_plane_row(&Hy0,1,i)[j].r
			- iDM0[1][Mh_AN][kl][i][j]*openmx_solver_complex_plane_row(&Hy0,1,i)[j].i
		    + 2.0*CDM0[2][Mh_AN][kl][i][j]*openmx_solver_complex_plane_row(&Hy0,2,i)[j].r
		    - 2.0*CDM0[3][Mh_AN][kl][i][j]*openmx_solver_complex_plane_row(&Hy0,2,i)[j].i;

		      dEz +=  
			  CDM0[0][Mh_AN][kl][i][j]*openmx_solver_complex_plane_row(&Hz0,0,i)[j].r
			- iDM0[0][Mh_AN][kl][i][j]*openmx_solver_complex_plane_row(&Hz0,0,i)[j].i
			+ CDM0[1][Mh_AN][kl][i][j]*openmx_solver_complex_plane_row(&Hz0,1,i)[j].r
			- iDM0[1][Mh_AN][kl][i][j]*openmx_solver_complex_plane_row(&Hz0,1,i)[j].i
		    + 2.0*CDM0[2][Mh_AN][kl][i][j]*openmx_solver_complex_plane_row(&Hz0,2,i)[j].r
		    - 2.0*CDM0[3][Mh_AN][kl][i][j]*openmx_solver_complex_plane_row(&Hz0,2,i)[j].i;

		    }
		  }

                  /* for stress */ 

		  for (m=1; m<=3; m++){
		    for (n=1; n<=3; n++){
		      for (i=0; i<atomic->species.contracted_orbitals[Hwan]; i++){
			for (j=0; j<atomic->species.contracted_orbitals[Qwan]; j++){

			  My_Stress[(m-1)*3+(n-1)] +=
			          CDM0[0][Mh_AN][kl][i][j]*OpenMX_Stress_HnlStrainRow(&H_sts,0,m,n,i)[j].r
			        - iDM0[0][Mh_AN][kl][i][j]*OpenMX_Stress_HnlStrainRow(&H_sts,0,m,n,i)[j].i
			        + CDM0[1][Mh_AN][kl][i][j]*OpenMX_Stress_HnlStrainRow(&H_sts,1,m,n,i)[j].r
			        - iDM0[1][Mh_AN][kl][i][j]*OpenMX_Stress_HnlStrainRow(&H_sts,1,m,n,i)[j].i
			    + 2.0*CDM0[2][Mh_AN][kl][i][j]*OpenMX_Stress_HnlStrainRow(&H_sts,2,m,n,i)[j].r
			    - 2.0*CDM0[3][Mh_AN][kl][i][j]*OpenMX_Stress_HnlStrainRow(&H_sts,2,m,n,i)[j].i;
			}
		      }
		    }
		  }

		} /* if (q_AN==h_AN) */

                else {

                  /* for forces */ 

		  for (i=0; i<atomic->species.contracted_orbitals[Hwan]; i++){  /* Hwan */
		    for (j=0; j<atomic->species.contracted_orbitals[Qwan]; j++){ /* Qwan  */

		      dEx += 
			  CDM0[0][Mh_AN][kl ][i][j]*openmx_solver_complex_plane_row(&Hx0,0,i)[j].r
			- iDM0[0][Mh_AN][kl ][i][j]*openmx_solver_complex_plane_row(&Hx0,0,i)[j].i
			+ CDM0[1][Mh_AN][kl ][i][j]*openmx_solver_complex_plane_row(&Hx0,1,i)[j].r
			- iDM0[1][Mh_AN][kl ][i][j]*openmx_solver_complex_plane_row(&Hx0,1,i)[j].i
		    + 2.0*CDM0[2][Mh_AN][kl ][i][j]*openmx_solver_complex_plane_row(&Hx0,2,i)[j].r
		    - 2.0*CDM0[3][Mh_AN][kl ][i][j]*openmx_solver_complex_plane_row(&Hx0,2,i)[j].i;

		      dEy += 
			  CDM0[0][Mh_AN][kl ][i][j]*openmx_solver_complex_plane_row(&Hy0,0,i)[j].r
			- iDM0[0][Mh_AN][kl ][i][j]*openmx_solver_complex_plane_row(&Hy0,0,i)[j].i
			+ CDM0[1][Mh_AN][kl ][i][j]*openmx_solver_complex_plane_row(&Hy0,1,i)[j].r
			- iDM0[1][Mh_AN][kl ][i][j]*openmx_solver_complex_plane_row(&Hy0,1,i)[j].i
                    + 2.0*CDM0[2][Mh_AN][kl ][i][j]*openmx_solver_complex_plane_row(&Hy0,2,i)[j].r
		    - 2.0*CDM0[3][Mh_AN][kl ][i][j]*openmx_solver_complex_plane_row(&Hy0,2,i)[j].i;

		      dEz +=
			  CDM0[0][Mh_AN][kl ][i][j]*openmx_solver_complex_plane_row(&Hz0,0,i)[j].r
			- iDM0[0][Mh_AN][kl ][i][j]*openmx_solver_complex_plane_row(&Hz0,0,i)[j].i
			+ CDM0[1][Mh_AN][kl ][i][j]*openmx_solver_complex_plane_row(&Hz0,1,i)[j].r
			- iDM0[1][Mh_AN][kl ][i][j]*openmx_solver_complex_plane_row(&Hz0,1,i)[j].i
                    + 2.0*CDM0[2][Mh_AN][kl ][i][j]*openmx_solver_complex_plane_row(&Hz0,2,i)[j].r
		    - 2.0*CDM0[3][Mh_AN][kl ][i][j]*openmx_solver_complex_plane_row(&Hz0,2,i)[j].i;

		    } /* j */
		  } /* i */

                  /* for stress */ 

		  for (m=1; m<=3; m++){
		    for (n=1; n<=3; n++){
		      for (i=0; i<atomic->species.contracted_orbitals[Hwan]; i++){
			for (j=0; j<atomic->species.contracted_orbitals[Qwan]; j++){

			  My_Stress[(m-1)*3+(n-1)] +=
			          CDM0[0][Mh_AN][kl][i][j]*OpenMX_Stress_HnlStrainRow(&H_sts,0,m,n,i)[j].r
			        - iDM0[0][Mh_AN][kl][i][j]*OpenMX_Stress_HnlStrainRow(&H_sts,0,m,n,i)[j].i
			        + CDM0[1][Mh_AN][kl][i][j]*OpenMX_Stress_HnlStrainRow(&H_sts,1,m,n,i)[j].r
			        - iDM0[1][Mh_AN][kl][i][j]*OpenMX_Stress_HnlStrainRow(&H_sts,1,m,n,i)[j].i
			    + 2.0*CDM0[2][Mh_AN][kl][i][j]*OpenMX_Stress_HnlStrainRow(&H_sts,2,m,n,i)[j].r
			    - 2.0*CDM0[3][Mh_AN][kl][i][j]*OpenMX_Stress_HnlStrainRow(&H_sts,2,m,n,i)[j].i;
			}
		      }
		    }
		  }

                  /* call dHNL */

		  dHNL(atomic, 0,Mc_AN,q_AN,h_AN,DS_NL,&Hx1,&Hy1,&Hz1,&H_sts, electronic);
		  kl1 = RMI1[Mc_AN][q_AN][h_AN];

                  /* for forces */ 

		  for (i=0; i<atomic->species.contracted_orbitals[Qwan]; i++){ /* Qwan */
		    for (j=0; j<atomic->species.contracted_orbitals[Hwan]; j++){ /* Hwan */

		      dEx += 
			  CDM0[0][Mq_AN][kl1][i][j]*openmx_solver_complex_plane_row(&Hx1,0,i)[j].r
			- iDM0[0][Mq_AN][kl1][i][j]*openmx_solver_complex_plane_row(&Hx1,0,i)[j].i
			+ CDM0[1][Mq_AN][kl1][i][j]*openmx_solver_complex_plane_row(&Hx1,1,i)[j].r
			- iDM0[1][Mq_AN][kl1][i][j]*openmx_solver_complex_plane_row(&Hx1,1,i)[j].i
		    + 2.0*CDM0[2][Mq_AN][kl1][i][j]*openmx_solver_complex_plane_row(&Hx1,2,i)[j].r
		    - 2.0*CDM0[3][Mq_AN][kl1][i][j]*openmx_solver_complex_plane_row(&Hx1,2,i)[j].i;

		      dEy += 
			  CDM0[0][Mq_AN][kl1][i][j]*openmx_solver_complex_plane_row(&Hy1,0,i)[j].r
			- iDM0[0][Mq_AN][kl1][i][j]*openmx_solver_complex_plane_row(&Hy1,0,i)[j].i
			+ CDM0[1][Mq_AN][kl1][i][j]*openmx_solver_complex_plane_row(&Hy1,1,i)[j].r
			- iDM0[1][Mq_AN][kl1][i][j]*openmx_solver_complex_plane_row(&Hy1,1,i)[j].i
		    + 2.0*CDM0[2][Mq_AN][kl1][i][j]*openmx_solver_complex_plane_row(&Hy1,2,i)[j].r
		    - 2.0*CDM0[3][Mq_AN][kl1][i][j]*openmx_solver_complex_plane_row(&Hy1,2,i)[j].i;

		      dEz +=
			  CDM0[0][Mq_AN][kl1][i][j]*openmx_solver_complex_plane_row(&Hz1,0,i)[j].r
			- iDM0[0][Mq_AN][kl1][i][j]*openmx_solver_complex_plane_row(&Hz1,0,i)[j].i
			+ CDM0[1][Mq_AN][kl1][i][j]*openmx_solver_complex_plane_row(&Hz1,1,i)[j].r
			- iDM0[1][Mq_AN][kl1][i][j]*openmx_solver_complex_plane_row(&Hz1,1,i)[j].i
		    + 2.0*CDM0[2][Mq_AN][kl1][i][j]*openmx_solver_complex_plane_row(&Hz1,2,i)[j].r
		    - 2.0*CDM0[3][Mq_AN][kl1][i][j]*openmx_solver_complex_plane_row(&Hz1,2,i)[j].i;

		    } /* j */
		  } /* i */

                  /* for stress */ 

		  for (m=1; m<=3; m++){
		    for (n=1; n<=3; n++){
		      for (i=0; i<atomic->species.contracted_orbitals[Qwan]; i++){ /* Qwan */
			for (j=0; j<atomic->species.contracted_orbitals[Hwan]; j++){ /* Hwan */

			  My_Stress[(m-1)*3+(n-1)] +=
			          CDM0[0][Mq_AN][kl1][i][j]*OpenMX_Stress_HnlStrainRow(&H_sts,0,m,n,i)[j].r
			        - iDM0[0][Mq_AN][kl1][i][j]*OpenMX_Stress_HnlStrainRow(&H_sts,0,m,n,i)[j].i
			        + CDM0[1][Mq_AN][kl1][i][j]*OpenMX_Stress_HnlStrainRow(&H_sts,1,m,n,i)[j].r
			        - iDM0[1][Mq_AN][kl1][i][j]*OpenMX_Stress_HnlStrainRow(&H_sts,1,m,n,i)[j].i
			    + 2.0*CDM0[2][Mq_AN][kl1][i][j]*OpenMX_Stress_HnlStrainRow(&H_sts,2,m,n,i)[j].r
			    - 2.0*CDM0[3][Mq_AN][kl1][i][j]*OpenMX_Stress_HnlStrainRow(&H_sts,2,m,n,i)[j].i;

			} /* j */
		      } /* i */
		    }
		  }

                }
	      }

	    } /* if (po2==1) */
	  } /* j0 */             

	  /* force from HNL */
	  /*
	  Gxyz[Gc_AN][41] += dEx;
	  Gxyz[Gc_AN][42] += dEy;
	  Gxyz[Gc_AN][43] += dEz;
	  */

	  /* timing */
	  dtime(&Etime_atom);
	  time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

	} /* Mc_AN */

	/********************************************
          increment of receive_progress[IDR]
	********************************************/

	receive_progress[IDR]++;

      } /* if ( 0<(F_Snd_Num[IDS]-send_progress[IDS]) ) */

      if ( 0<(F_Snd_Num[IDS]-send_progress[IDS]) ) {

	MPI_Wait(&request,&stat);
	  /* freeing of array */

	/********************************************
             increment of send_progress[IDS]
	********************************************/

	send_progress[IDS]++;
      } 

    } /* ID */

    /*****************************************************
      check whether all the communications have finished
    *****************************************************/

    po = 0;
    for (ID=0; ID<numprocs; ID++){

      IDS = (myid + ID) % numprocs;
      IDR = (myid - ID + numprocs) % numprocs;

      if ( 0<(F_Snd_Num[IDS]-send_progress[IDS]) ) po += F_Snd_Num[IDS]-send_progress[IDS];
      if ( 0<(F_Rcv_Num[IDR]-receive_progress[IDR]) ) po += F_Rcv_Num[IDR]-receive_progress[IDR];
    }

  } while (po!=0);

  dtime(&etime);
  if(myid==0 && measure_time){
    printf("Time for part1 of force_NL=%18.5f\n",etime-stime);fflush(stdout);
  } 

  /*
  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];

    if (2<=level_stdout){
      printf("<Force>  force(HNL1) myid=%2d  Mc_AN=%2d Gc_AN=%2d  %15.12f %15.12f %15.12f\n",
	     myid,Mc_AN,Gc_AN,Gxyz[Gc_AN][41],Gxyz[Gc_AN][42],Gxyz[Gc_AN][43]);fflush(stdout);
    }
  }
  */

  /*******************************************************
   *******************************************************
     THE FIRST CASE:
     multiplying overlap integrals WITHOUT COMMUNICATION

     In case of I=i or I=j 
     for d [ \sum_k <i|k>ek<k|j> ]/dRI  
   *******************************************************
   *******************************************************/

  dtime(&stime);

#pragma omp parallel shared(scratch,time_per_atom,Gxyz,CDM0,SpinP_switch,SO_switch,Hub_U_switch,F_U_flag,Constraint_NCS_switch,Zeeman_NCS_switch,Zeeman_NCO_switch,DS_NL,RMI1,FNAN,WhatSpecies,F_G2M,natn,M2G,Matomnum,List_YOUSO,F_NL_flag,stress_threads,atomic) private(Hx0,Hy0,Hz0,Hx1,Hy1,Hz1,H_sts,OMPID,Nthrds,Nprocs,Mc_AN,Stime_atom,Etime_atom,dEx,dEy,dEz,Gc_AN,h_AN,Gh_AN,Mh_AN,Hwan,ian,q_AN,Gq_AN,Mq_AN,Qwan,jan,kl,kl1,i,j,k,m,n,kk,pref)
  {

    /* allocation of array */

    OpenMX_Stress_HnlPrepare(scratch,List_YOUSO[7],&Hx0,&Hy0,&Hz0,&Hx1,&Hy1,&Hz1,&H_sts);

    /* get info. on OpenMP */ 

    OMPID = omp_get_thread_num();
    Nthrds = omp_get_num_threads();
    Nprocs = omp_get_num_procs();

    for (Mc_AN=(OMPID*Matomnum/Nthrds+1); Mc_AN<((OMPID+1)*Matomnum/Nthrds+1); Mc_AN++){

      dtime(&Stime_atom);

      dEx = 0.0; 
      dEy = 0.0; 
      dEz = 0.0; 

      Gc_AN = M2G[Mc_AN];
      h_AN = 0;
      Gh_AN = natn[Gc_AN][h_AN];
      Mh_AN = F_G2M[Gh_AN];
      Hwan = WhatSpecies[Gh_AN];
      ian = atomic->species.contracted_orbitals[Hwan];

      for (q_AN=0; q_AN<=FNAN[Gc_AN]; q_AN++){

	Gq_AN = natn[Gc_AN][q_AN];
	Mq_AN = F_G2M[Gq_AN];

	if (Mq_AN<=Matomnum){

	  Qwan = WhatSpecies[Gq_AN];
	  jan = atomic->species.contracted_orbitals[Qwan];
	  kl = RMI1[Mc_AN][h_AN][q_AN];

          dHNL(atomic, 0,Mc_AN,h_AN,q_AN,DS_NL,&Hx0,&Hy0,&Hz0,&H_sts, electronic);

	  if (SpinP_switch==0){

            if (q_AN==h_AN) pref = 2.0;
            else            pref = 4.0; 

            /* for forces */                 

	    for (i=0; i<ian; i++){
	      for (j=0; j<jan; j++){
		dEx += pref*CDM0[0][Mh_AN][kl][i][j]*openmx_solver_complex_plane_row(&Hx0,0,i)[j].r;
		dEy += pref*CDM0[0][Mh_AN][kl][i][j]*openmx_solver_complex_plane_row(&Hy0,0,i)[j].r;
		dEz += pref*CDM0[0][Mh_AN][kl][i][j]*openmx_solver_complex_plane_row(&Hz0,0,i)[j].r;
	      }
	    }

	    /* for stress */                 

	    for (m=1; m<=3; m++){
	      for (n=1; n<=3; n++){
		for (i=0; i<ian; i++){
		  for (j=0; j<jan; j++){
		    openmx_solver_real_matrix_row(&stress_threads,OMPID)[(m-1)*3+(n-1)] += pref*CDM0[0][Mh_AN][kl][i][j]*OpenMX_Stress_HnlStrainRow(&H_sts,0,m,n,i)[j].r;
		  }
		}
	      }
	    }

	  }

          /* collinear spin polarized or non-colliear without SO and LDA+U */

	  else if (SpinP_switch==1 || (SpinP_switch==3 && SO_switch==0 && Hub_U_switch==0
	        && Constraint_NCS_switch==0 && Zeeman_NCS_switch==0 && Zeeman_NCO_switch==0)){ 

	    if (q_AN==h_AN) pref = 1.0;
	    else            pref = 2.0; 

            /* for forces */                 

	    for (i=0; i<ian; i++){
	      for (j=0; j<jan; j++){

		dEx += pref*(  CDM0[0][Mh_AN][kl][i][j]*openmx_solver_complex_plane_row(&Hx0,0,i)[j].r
			     + CDM0[1][Mh_AN][kl][i][j]*openmx_solver_complex_plane_row(&Hx0,1,i)[j].r);
		dEy += pref*(  CDM0[0][Mh_AN][kl][i][j]*openmx_solver_complex_plane_row(&Hy0,0,i)[j].r
			     + CDM0[1][Mh_AN][kl][i][j]*openmx_solver_complex_plane_row(&Hy0,1,i)[j].r);
		dEz += pref*(  CDM0[0][Mh_AN][kl][i][j]*openmx_solver_complex_plane_row(&Hz0,0,i)[j].r
			     + CDM0[1][Mh_AN][kl][i][j]*openmx_solver_complex_plane_row(&Hz0,1,i)[j].r);
	      }
	    }

	    /* for stress */                 

	    for (m=1; m<=3; m++){
	      for (n=1; n<=3; n++){
		for (i=0; i<ian; i++){
		  for (j=0; j<jan; j++){
		    openmx_solver_real_matrix_row(&stress_threads,OMPID)[(m-1)*3+(n-1)] += pref*( CDM0[0][Mh_AN][kl][i][j]*OpenMX_Stress_HnlStrainRow(&H_sts,0,m,n,i)[j].r
                                                                    + CDM0[1][Mh_AN][kl][i][j]*OpenMX_Stress_HnlStrainRow(&H_sts,1,m,n,i)[j].r);
		  }
		}
	      }
	    }
	  }

	  /* spin non-collinear with spin-orbit coupling or with LDA+U */

	  else if ( SpinP_switch==3 && (SO_switch==1 || (Hub_U_switch==1 && F_U_flag==1) 
		|| 1<=Constraint_NCS_switch || Zeeman_NCS_switch==1 || Zeeman_NCO_switch==1)){

            if (q_AN==h_AN){

              /* for forces */ 

	      for (i=0; i<atomic->species.contracted_orbitals[Hwan]; i++){
		for (j=0; j<atomic->species.contracted_orbitals[Qwan]; j++){

		  dEx += 
                      CDM0[0][Mh_AN][kl][i][j]*openmx_solver_complex_plane_row(&Hx0,0,i)[j].r
		    - iDM0[0][Mh_AN][kl][i][j]*openmx_solver_complex_plane_row(&Hx0,0,i)[j].i
		    + CDM0[1][Mh_AN][kl][i][j]*openmx_solver_complex_plane_row(&Hx0,1,i)[j].r
		    - iDM0[1][Mh_AN][kl][i][j]*openmx_solver_complex_plane_row(&Hx0,1,i)[j].i
	        + 2.0*CDM0[2][Mh_AN][kl][i][j]*openmx_solver_complex_plane_row(&Hx0,2,i)[j].r
		- 2.0*CDM0[3][Mh_AN][kl][i][j]*openmx_solver_complex_plane_row(&Hx0,2,i)[j].i;

		  dEy += 
                      CDM0[0][Mh_AN][kl][i][j]*openmx_solver_complex_plane_row(&Hy0,0,i)[j].r
		    - iDM0[0][Mh_AN][kl][i][j]*openmx_solver_complex_plane_row(&Hy0,0,i)[j].i
		    + CDM0[1][Mh_AN][kl][i][j]*openmx_solver_complex_plane_row(&Hy0,1,i)[j].r
		    - iDM0[1][Mh_AN][kl][i][j]*openmx_solver_complex_plane_row(&Hy0,1,i)[j].i
		+ 2.0*CDM0[2][Mh_AN][kl][i][j]*openmx_solver_complex_plane_row(&Hy0,2,i)[j].r
	        - 2.0*CDM0[3][Mh_AN][kl][i][j]*openmx_solver_complex_plane_row(&Hy0,2,i)[j].i;

		  dEz +=
                      CDM0[0][Mh_AN][kl][i][j]*openmx_solver_complex_plane_row(&Hz0,0,i)[j].r
		    - iDM0[0][Mh_AN][kl][i][j]*openmx_solver_complex_plane_row(&Hz0,0,i)[j].i
		    + CDM0[1][Mh_AN][kl][i][j]*openmx_solver_complex_plane_row(&Hz0,1,i)[j].r
		    - iDM0[1][Mh_AN][kl][i][j]*openmx_solver_complex_plane_row(&Hz0,1,i)[j].i
	        + 2.0*CDM0[2][Mh_AN][kl][i][j]*openmx_solver_complex_plane_row(&Hz0,2,i)[j].r
		- 2.0*CDM0[3][Mh_AN][kl][i][j]*openmx_solver_complex_plane_row(&Hz0,2,i)[j].i;

		}
	      }

	      /* for stress */ 

	      for (m=1; m<=3; m++){
		for (n=1; n<=3; n++){
		  for (i=0; i<atomic->species.contracted_orbitals[Hwan]; i++){
		    for (j=0; j<atomic->species.contracted_orbitals[Qwan]; j++){

		      openmx_solver_real_matrix_row(&stress_threads,OMPID)[(m-1)*3+(n-1)] +=
                              CDM0[0][Mh_AN][kl][i][j]*OpenMX_Stress_HnlStrainRow(&H_sts,0,m,n,i)[j].r
			    - iDM0[0][Mh_AN][kl][i][j]*OpenMX_Stress_HnlStrainRow(&H_sts,0,m,n,i)[j].i
                            + CDM0[1][Mh_AN][kl][i][j]*OpenMX_Stress_HnlStrainRow(&H_sts,1,m,n,i)[j].r
			    - iDM0[1][Mh_AN][kl][i][j]*OpenMX_Stress_HnlStrainRow(&H_sts,1,m,n,i)[j].i
			+ 2.0*CDM0[2][Mh_AN][kl][i][j]*OpenMX_Stress_HnlStrainRow(&H_sts,2,m,n,i)[j].r
			- 2.0*CDM0[3][Mh_AN][kl][i][j]*OpenMX_Stress_HnlStrainRow(&H_sts,2,m,n,i)[j].i;
		    }
		  }
		}
	      }

            } /* if (q_AN==h_AN) */

            else{

              /* for forces */ 

              for (i=0; i<atomic->species.contracted_orbitals[Hwan]; i++){  /* Hwan */
		for (j=0; j<atomic->species.contracted_orbitals[Qwan]; j++){ /* Qwan  */

		  dEx += 
                      CDM0[0][Mh_AN][kl ][i][j]*openmx_solver_complex_plane_row(&Hx0,0,i)[j].r
		    - iDM0[0][Mh_AN][kl ][i][j]*openmx_solver_complex_plane_row(&Hx0,0,i)[j].i
		    + CDM0[1][Mh_AN][kl ][i][j]*openmx_solver_complex_plane_row(&Hx0,1,i)[j].r
		    - iDM0[1][Mh_AN][kl ][i][j]*openmx_solver_complex_plane_row(&Hx0,1,i)[j].i
		+ 2.0*CDM0[2][Mh_AN][kl ][i][j]*openmx_solver_complex_plane_row(&Hx0,2,i)[j].r
                - 2.0*CDM0[3][Mh_AN][kl ][i][j]*openmx_solver_complex_plane_row(&Hx0,2,i)[j].i;

		  dEy += 
                      CDM0[0][Mh_AN][kl ][i][j]*openmx_solver_complex_plane_row(&Hy0,0,i)[j].r
		    - iDM0[0][Mh_AN][kl ][i][j]*openmx_solver_complex_plane_row(&Hy0,0,i)[j].i
		    + CDM0[1][Mh_AN][kl ][i][j]*openmx_solver_complex_plane_row(&Hy0,1,i)[j].r
		    - iDM0[1][Mh_AN][kl ][i][j]*openmx_solver_complex_plane_row(&Hy0,1,i)[j].i
		+ 2.0*CDM0[2][Mh_AN][kl ][i][j]*openmx_solver_complex_plane_row(&Hy0,2,i)[j].r
                - 2.0*CDM0[3][Mh_AN][kl ][i][j]*openmx_solver_complex_plane_row(&Hy0,2,i)[j].i;

		  dEz +=
                      CDM0[0][Mh_AN][kl ][i][j]*openmx_solver_complex_plane_row(&Hz0,0,i)[j].r
		    - iDM0[0][Mh_AN][kl ][i][j]*openmx_solver_complex_plane_row(&Hz0,0,i)[j].i
		    + CDM0[1][Mh_AN][kl ][i][j]*openmx_solver_complex_plane_row(&Hz0,1,i)[j].r
		    - iDM0[1][Mh_AN][kl ][i][j]*openmx_solver_complex_plane_row(&Hz0,1,i)[j].i
	        + 2.0*CDM0[2][Mh_AN][kl ][i][j]*openmx_solver_complex_plane_row(&Hz0,2,i)[j].r
                - 2.0*CDM0[3][Mh_AN][kl ][i][j]*openmx_solver_complex_plane_row(&Hz0,2,i)[j].i;

		} /* j */
	      } /* i */

	      /* for stress */ 

	      for (m=1; m<=3; m++){
		for (n=1; n<=3; n++){
		  for (i=0; i<atomic->species.contracted_orbitals[Hwan]; i++){
		    for (j=0; j<atomic->species.contracted_orbitals[Qwan]; j++){

		      openmx_solver_real_matrix_row(&stress_threads,OMPID)[(m-1)*3+(n-1)] +=
			      CDM0[0][Mh_AN][kl][i][j]*OpenMX_Stress_HnlStrainRow(&H_sts,0,m,n,i)[j].r
                            - iDM0[0][Mh_AN][kl][i][j]*OpenMX_Stress_HnlStrainRow(&H_sts,0,m,n,i)[j].i
                            + CDM0[1][Mh_AN][kl][i][j]*OpenMX_Stress_HnlStrainRow(&H_sts,1,m,n,i)[j].r
			    - iDM0[1][Mh_AN][kl][i][j]*OpenMX_Stress_HnlStrainRow(&H_sts,1,m,n,i)[j].i
			+ 2.0*CDM0[2][Mh_AN][kl][i][j]*OpenMX_Stress_HnlStrainRow(&H_sts,2,m,n,i)[j].r
			- 2.0*CDM0[3][Mh_AN][kl][i][j]*OpenMX_Stress_HnlStrainRow(&H_sts,2,m,n,i)[j].i;
		    }
		  }
		}
	      }

              /* call dHNL */

              dHNL(atomic, 0,Mc_AN,q_AN,h_AN,DS_NL,&Hx1,&Hy1,&Hz1,&H_sts, electronic);
       	      kl1 = RMI1[Mc_AN][q_AN][h_AN];

              /* for forces */ 

	      for (i=0; i<atomic->species.contracted_orbitals[Qwan]; i++){ /* Qwan */
		for (j=0; j<atomic->species.contracted_orbitals[Hwan]; j++){ /* Hwan */

		  dEx += 
		      CDM0[0][Mq_AN][kl1][i][j]*openmx_solver_complex_plane_row(&Hx1,0,i)[j].r
		    - iDM0[0][Mq_AN][kl1][i][j]*openmx_solver_complex_plane_row(&Hx1,0,i)[j].i
		    + CDM0[1][Mq_AN][kl1][i][j]*openmx_solver_complex_plane_row(&Hx1,1,i)[j].r
		    - iDM0[1][Mq_AN][kl1][i][j]*openmx_solver_complex_plane_row(&Hx1,1,i)[j].i
		+ 2.0*CDM0[2][Mq_AN][kl1][i][j]*openmx_solver_complex_plane_row(&Hx1,2,i)[j].r
		- 2.0*CDM0[3][Mq_AN][kl1][i][j]*openmx_solver_complex_plane_row(&Hx1,2,i)[j].i;

		  dEy += 
                      CDM0[0][Mq_AN][kl1][i][j]*openmx_solver_complex_plane_row(&Hy1,0,i)[j].r
		    - iDM0[0][Mq_AN][kl1][i][j]*openmx_solver_complex_plane_row(&Hy1,0,i)[j].i
		    + CDM0[1][Mq_AN][kl1][i][j]*openmx_solver_complex_plane_row(&Hy1,1,i)[j].r
		    - iDM0[1][Mq_AN][kl1][i][j]*openmx_solver_complex_plane_row(&Hy1,1,i)[j].i
	        + 2.0*CDM0[2][Mq_AN][kl1][i][j]*openmx_solver_complex_plane_row(&Hy1,2,i)[j].r
	        - 2.0*CDM0[3][Mq_AN][kl1][i][j]*openmx_solver_complex_plane_row(&Hy1,2,i)[j].i;

		  dEz +=
                      CDM0[0][Mq_AN][kl1][i][j]*openmx_solver_complex_plane_row(&Hz1,0,i)[j].r
		    - iDM0[0][Mq_AN][kl1][i][j]*openmx_solver_complex_plane_row(&Hz1,0,i)[j].i
		    + CDM0[1][Mq_AN][kl1][i][j]*openmx_solver_complex_plane_row(&Hz1,1,i)[j].r
		    - iDM0[1][Mq_AN][kl1][i][j]*openmx_solver_complex_plane_row(&Hz1,1,i)[j].i
	        + 2.0*CDM0[2][Mq_AN][kl1][i][j]*openmx_solver_complex_plane_row(&Hz1,2,i)[j].r
		- 2.0*CDM0[3][Mq_AN][kl1][i][j]*openmx_solver_complex_plane_row(&Hz1,2,i)[j].i;

		} /* j */
	      } /* i */

	      /* for stress */ 

	      for (m=1; m<=3; m++){
		for (n=1; n<=3; n++){
		  for (i=0; i<atomic->species.contracted_orbitals[Qwan]; i++){ /* Qwan */
		    for (j=0; j<atomic->species.contracted_orbitals[Hwan]; j++){ /* Hwan */

		      openmx_solver_real_matrix_row(&stress_threads,OMPID)[(m-1)*3+(n-1)] +=
			      CDM0[0][Mq_AN][kl1][i][j]*OpenMX_Stress_HnlStrainRow(&H_sts,0,m,n,i)[j].r
			    - iDM0[0][Mq_AN][kl1][i][j]*OpenMX_Stress_HnlStrainRow(&H_sts,0,m,n,i)[j].i
			    + CDM0[1][Mq_AN][kl1][i][j]*OpenMX_Stress_HnlStrainRow(&H_sts,1,m,n,i)[j].r
			    - iDM0[1][Mq_AN][kl1][i][j]*OpenMX_Stress_HnlStrainRow(&H_sts,1,m,n,i)[j].i
			+ 2.0*CDM0[2][Mq_AN][kl1][i][j]*OpenMX_Stress_HnlStrainRow(&H_sts,2,m,n,i)[j].r
			- 2.0*CDM0[3][Mq_AN][kl1][i][j]*OpenMX_Stress_HnlStrainRow(&H_sts,2,m,n,i)[j].i;

		    } /* j */
		  } /* i */
		}
	      }

	    } 
	  }
	}
      }

      /* force from #4B */
      /*
      if (F_NL_flag==1){
        Gxyz[Gc_AN][41] += dEx;
        Gxyz[Gc_AN][42] += dEy;
        Gxyz[Gc_AN][43] += dEz;
      }
      */

      /* timing */
      dtime(&Etime_atom);
      time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

    } /* Mc_AN */

    /* freeing of array */

  } /* #pragma omp parallel */

  /* summing up results calculated by OpenMP threads */

  for(i=0; i<Nthrds0; i++){
    for(j=0; j<9; j++){
      My_Stress[j] += openmx_solver_real_matrix_row(&stress_threads,i)[j];
    }
  }

  /* MPI_Allreduce: My_Stress */

  MPI_Allreduce(My_Stress, Stress, 9, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

  for (i=0; i<9; i++){
    Stress[i] = Stress[i]*(double)F_NL_flag; 
  }

  /* add stress coming from HNL to Stress_Tensor */

  for (i=0; i<9; i++){
    Stress_Tensor[i] += Stress[i];
  }

  /* show stress coming from HNL */

  if (myid==Host_ID && 1<level_stdout){

    if (myid==Host_ID){
      printf(" Components of stress HNL: \n");fflush(stdout);
      for (i=0; i<3; i++){
	for (j=0; j<3; j++){
	  printf("%16.8f ", Stress[3*i+j]);
	}
	printf("\n");fflush(stdout);
      }
    }
  }

  /* measuring time */

  dtime(&etime);
  if(myid==0 && measure_time){
    printf("Time for part2 of force_NL=%18.5f\n",etime-stime);fflush(stdout);
  } 

  /*
  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];

    if (2<=level_stdout){
      printf("<Force>  force(HNL2) myid=%2d  Mc_AN=%2d Gc_AN=%2d  %15.12f %15.12f %15.12f\n",
	     myid,Mc_AN,Gc_AN,Gxyz[Gc_AN][41],Gxyz[Gc_AN][42],Gxyz[Gc_AN][43]);fflush(stdout);
    }
  }
  */

  /***********************************
            freeing of arrays 
  ************************************/


}







void Stress_plus_U_dual(OpenMXSolverInvocation *scratch,double *****CDM0, const OpenMXElectronicViews *electronic, const OpenMXLocalBasisViews *local_basis)
{

  double *****OLP=electronic->OLP;

  int i,j;
  int numprocs,myid;
  int Nthrds0;
  OpenMXSolverRealTensor stress_threads;
  double Stress[9],My_Stress[9];

  MPI_Status stat;
  MPI_Request request;

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  if (myid==Host_ID && 0<level_stdout){
    printf("  Stress calculation: DFT+U\n");fflush(stdout);
  }

  /* get Nthrds0 */

  /* Dynamic teams may grow after a separate probe region. */
  Nthrds0 = omp_get_max_threads();

  /* allocation of array */

  OpenMX_Stress_ThreadsPrepare(scratch,Nthrds0,&stress_threads);
  for (i=0; i<Nthrds0; i++){

    for (j=0; j<9; j++) openmx_solver_real_matrix_row(&stress_threads,i)[j] = 0.0;
  }

  /* OpenMP threading for Mc_AN loop */

#pragma omp parallel shared(List_YOUSO,Matomnum,M2G,WhatSpecies,natn,ncn,F_G2M,Spe_Total_NO,Gxyz,atv,FNAN,RMI1,local_basis,OLP,SpinP_switch,CDM0,stress_threads)
  {
    int i,j,k,Mc_AN,so,Rn,Gc_AN,Cwan;
    int spin,Gh_AN,Mh_AN,Hwan,h_AN,spinmax;
    int OMPID,Nthrds,Nprocs;
    double tmp,tmp1,tmp2,tmp3;
    double dEx,dEy,dEz,dege;
    double lx,ly,lz;

    /* get info. on OpenMP */ 

    OMPID = omp_get_thread_num();
    Nthrds = omp_get_num_threads();
    Nprocs = omp_get_num_procs();

    if (SpinP_switch==0){
      spinmax = 0;
      dege = 2.0;
    }
    else{
      spinmax = 1;
      dege = 1.0;
    }

    /* loop for Mc_AN */ 

    for (Mc_AN=(OMPID*Matomnum/Nthrds+1); Mc_AN<((OMPID+1)*Matomnum/Nthrds+1); Mc_AN++){

      Gc_AN = M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];

      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	Gh_AN = natn[Gc_AN][h_AN];
	Mh_AN = F_G2M[Gh_AN];
	Hwan = WhatSpecies[Gh_AN];

	Rn = ncn[Gc_AN][h_AN];
	lx = Gxyz[Gc_AN][1] - Gxyz[Gh_AN][1] - atv[Rn][1];
	ly = Gxyz[Gc_AN][2] - Gxyz[Gh_AN][2] - atv[Rn][2];
	lz = Gxyz[Gc_AN][3] - Gxyz[Gh_AN][3] - atv[Rn][3];

        dEx = 0.0;
        dEy = 0.0;
        dEz = 0.0;

        for (spin=0; spin<=spinmax; spin++){
	  for (i=0; i<Spe_Total_NO[Cwan]; i++){
	    for (j=0; j<Spe_Total_NO[Hwan]; j++){

	      tmp1 = 0.0; 
	      tmp2 = 0.0; 
	      tmp3 = 0.0; 

	      for (k=0; k<Spe_Total_NO[Cwan]; k++){
		tmp = openmx_local_potential_row(local_basis, spin, Mc_AN, i)[k];
		tmp1 += tmp*OLP[1][Mc_AN][h_AN][k][j];
		tmp2 += tmp*OLP[2][Mc_AN][h_AN][k][j];
		tmp3 += tmp*OLP[3][Mc_AN][h_AN][k][j];
	      }

	      for (k=0; k<Spe_Total_NO[Hwan]; k++){
		tmp = openmx_local_potential_row(local_basis, spin, Mh_AN, k)[j];
		tmp1 += tmp*OLP[1][Mc_AN][h_AN][i][k];
		tmp2 += tmp*OLP[2][Mc_AN][h_AN][i][k];
		tmp3 += tmp*OLP[3][Mc_AN][h_AN][i][k];
	      }

	      dEx += dege*tmp1*CDM0[spin][Mc_AN][h_AN][i][j];
	      dEy += dege*tmp2*CDM0[spin][Mc_AN][h_AN][i][j];
	      dEz += dege*tmp3*CDM0[spin][Mc_AN][h_AN][i][j];
	    }
	  }
	} /* spin */

	openmx_solver_real_matrix_row(&stress_threads,OMPID)[0] += 0.5*dEx*lx;
	openmx_solver_real_matrix_row(&stress_threads,OMPID)[1] += 0.5*dEy*lx;
	openmx_solver_real_matrix_row(&stress_threads,OMPID)[2] += 0.5*dEz*lx;
	openmx_solver_real_matrix_row(&stress_threads,OMPID)[3] += 0.5*dEx*ly;
	openmx_solver_real_matrix_row(&stress_threads,OMPID)[4] += 0.5*dEy*ly;
	openmx_solver_real_matrix_row(&stress_threads,OMPID)[5] += 0.5*dEz*ly;
	openmx_solver_real_matrix_row(&stress_threads,OMPID)[6] += 0.5*dEx*lz;
	openmx_solver_real_matrix_row(&stress_threads,OMPID)[7] += 0.5*dEy*lz;
	openmx_solver_real_matrix_row(&stress_threads,OMPID)[8] += 0.5*dEz*lz;

      } /* h_AN */
    } /* Mc_AN */

  } /* #pragma omp parallel */

  /* sum up stress */

  for(i=0; i<9; i++){
    My_Stress[i] = 0.0;
  }

  for(i=0; i<Nthrds0; i++){
    for(j=0; j<9; j++){
      My_Stress[j] += openmx_solver_real_matrix_row(&stress_threads,i)[j];
    }
  }

  MPI_Allreduce(&My_Stress[0],&Stress[0],9,MPI_DOUBLE,MPI_SUM,mpi_comm_level1);

  /* if (F_U_flag==0) */

  if (F_U_flag==0){
    for (i=0; i<9; i++) Stress[i] = 0.0;
  }

  /* add DFT-plus-U components of stress to Stress_Tensor */

  for (i=0; i<9; i++){
    Stress_Tensor[i] += Stress[i];
  }

  /* show Kinetic components of stress */
  
  if (myid==Host_ID && 1<level_stdout){
    printf(" Components of stress (DFT+U, dual) : \n");fflush(stdout);
    for (i = 0; i<3; i++){
      for (j = 0; j<3; j++){
	printf("%16.8f ", Stress[3*i+j]);
      }
      printf("\n");fflush(stdout);
    }
  }

  /* freeing of array */



}




void dHVNA(const OpenMXVnaMatrixViews *vna, int where_flag, int Mc_AN, int h_AN, int q_AN,
           Type_DS_VNA *****DS_VNA1, 
           double *****TmpHVNA2, double *****TmpHVNA3, 
           const OpenMXSolverRealTensor *Hx,const OpenMXSolverRealTensor *Hy,
           const OpenMXSolverRealTensor *Hz,const OpenMXSolverRealTensor *H_sts)
{
  const int fp32 = openmx_fp32_enabled();
  int i,j,k,m,n,l,kg,kan,so,deri_kind;
  int ig,ian,jg,jan,kl,kl1,kl2,Rni,Rnj,Rnk;
  int wakg,l1,l2,l3,Gc_AN,Mi_AN,Mj_AN,Mj_AN2,num_projectors;
  double sumx,sumy,sumz,ene,rcuti,rcutj,rcut;
  double tmpx,tmpy,tmpz,dmp,deri_dmp,tmp;
  double dx,dy,dz,x0,y0,z0,x1,y1,z1,r;
  double PFp,PFm,ene_p,ene_m;
  double sumx0,sumy0,sumz0;
  double sumx1,sumy1,sumz1;
  double sumx2,sumy2,sumz2;
  double tx,ty,tz;
  int L,LL,Mul1,Num_RVNA;

  Num_RVNA = List_YOUSO[34]*(List_YOUSO[35] + 1);
  num_projectors = (List_YOUSO[35]+1)*(List_YOUSO[35]+1)*List_YOUSO[34];

  /****************************************************
   start calc.
  ****************************************************/

  Gc_AN = M2G[Mc_AN];
  ig = natn[Gc_AN][h_AN];
  Rni = ncn[Gc_AN][h_AN];
  Mi_AN = F_G2M[ig];
  ian = Spe_Total_CNO[WhatSpecies[ig]];
  rcuti = Spe_Atom_Cut1[WhatSpecies[ig]];

  jg = natn[Gc_AN][q_AN];
  Rnj = ncn[Gc_AN][q_AN];
  Mj_AN = F_G2M[jg];
  jan = Spe_Total_CNO[WhatSpecies[jg]];
  rcutj = Spe_Atom_Cut1[WhatSpecies[jg]];

  rcut = rcuti + rcutj;
  kl = RMI1[Mc_AN][h_AN][q_AN];
  dmp = dampingF(rcut,Dis[ig][kl]);

  for (m=0; m<ian; m++){
    for (n=0; n<jan; n++){
      openmx_solver_real_matrix_row(Hx,m)[n] = 0.0;
      openmx_solver_real_matrix_row(Hy,m)[n] = 0.0;
      openmx_solver_real_matrix_row(Hz,m)[n] = 0.0;
    }
  }

  for (i=0; i<4; i++){
    for (j=0; j<4; j++){
      for (m=0; m<ian; m++){
	for (n=0; n<jan; n++){
	  OpenMX_Stress_VnaStrainRow(H_sts,i,j,m)[n] = 0.0;
	}
      }
    }
  }

  /****************************************************
    two-center integral with orbitals on one-center 
    
    in case of h_AN==0 && q_AN==0
  ****************************************************/

  if (h_AN==0 && q_AN==0 && where_flag==0){

    for (k=1; k<=FNAN[Gc_AN]; k++){

      kg = natn[Gc_AN][k];
      Rnk = ncn[Gc_AN][k];

      tx = Gxyz[Gc_AN][1] - Gxyz[kg][1] - atv[Rnk][1];
      ty = Gxyz[Gc_AN][2] - Gxyz[kg][2] - atv[Rnk][2];
      tz = Gxyz[Gc_AN][3] - Gxyz[kg][3] - atv[Rnk][3];

      /* for force */ 

      for (m=0; m<ian; m++){
        for (n=0; n<jan; n++){
	  openmx_solver_real_matrix_row(Hx,m)[n] += TmpHVNA2[1][Mc_AN][k][m][n];
	  openmx_solver_real_matrix_row(Hy,m)[n] += TmpHVNA2[2][Mc_AN][k][m][n];
	  openmx_solver_real_matrix_row(Hz,m)[n] += TmpHVNA2[3][Mc_AN][k][m][n];
        }
      }      

      /* for stress calculation */ 

      for (m=0; m<ian; m++){
        for (n=0; n<jan; n++){

	  OpenMX_Stress_VnaStrainRow(H_sts,1,1,m)[n] += TmpHVNA2[1][Mc_AN][k][m][n]*tx;
	  OpenMX_Stress_VnaStrainRow(H_sts,1,2,m)[n] += TmpHVNA2[1][Mc_AN][k][m][n]*ty;
	  OpenMX_Stress_VnaStrainRow(H_sts,1,3,m)[n] += TmpHVNA2[1][Mc_AN][k][m][n]*tz;

	  OpenMX_Stress_VnaStrainRow(H_sts,2,1,m)[n] += TmpHVNA2[2][Mc_AN][k][m][n]*tx;
	  OpenMX_Stress_VnaStrainRow(H_sts,2,2,m)[n] += TmpHVNA2[2][Mc_AN][k][m][n]*ty;
	  OpenMX_Stress_VnaStrainRow(H_sts,2,3,m)[n] += TmpHVNA2[2][Mc_AN][k][m][n]*tz;

	  OpenMX_Stress_VnaStrainRow(H_sts,3,1,m)[n] += TmpHVNA2[3][Mc_AN][k][m][n]*tx;
	  OpenMX_Stress_VnaStrainRow(H_sts,3,2,m)[n] += TmpHVNA2[3][Mc_AN][k][m][n]*ty;
	  OpenMX_Stress_VnaStrainRow(H_sts,3,3,m)[n] += TmpHVNA2[3][Mc_AN][k][m][n]*tz;
        }
      }    
    } /* k */
  }

  /****************************************************
    two-center integral with orbitals on one-center 
    
    in case of h_AN==q_AN && h_AN!=0 
  ****************************************************/

  else if (h_AN==q_AN && h_AN!=0){

    kl = RMI1[Mc_AN][h_AN][0];

    for (m=0; m<ian; m++){
      for (n=0; n<jan; n++){

	openmx_solver_real_matrix_row(Hx,m)[n] = -TmpHVNA3[1][Mc_AN][h_AN][m][n];
	openmx_solver_real_matrix_row(Hy,m)[n] = -TmpHVNA3[2][Mc_AN][h_AN][m][n];
	openmx_solver_real_matrix_row(Hz,m)[n] = -TmpHVNA3[3][Mc_AN][h_AN][m][n];
      }
    } 
  }

  /****************************************************
             two and three center integrals
             with orbitals on two-center
  ****************************************************/

  else{

    if (h_AN==0){

      /****************************************************
			   dH*ep*H
      ****************************************************/

      for (k=0; k<=FNAN[Gc_AN]; k++){

	kg = natn[Gc_AN][k];
        Rnk = ncn[Gc_AN][k];
	wakg = WhatSpecies[kg];
	kl = RMI1[Mc_AN][q_AN][k];

        tx = Gxyz[Gc_AN][1] - Gxyz[kg][1] - atv[Rnk][1];
        ty = Gxyz[Gc_AN][2] - Gxyz[kg][2] - atv[Rnk][2];
        tz = Gxyz[Gc_AN][3] - Gxyz[kg][3] - atv[Rnk][3];

	/****************************************************
			     non-local part
	****************************************************/

	if (0<=kl && where_flag==0){

          if (Mj_AN<=Matomnum) Mj_AN2 = Mj_AN;
          else                 Mj_AN2 = Matomnum+1; 

	  for (m=0; m<ian; m++){
	    for (n=0; n<jan; n++){

	      sumx = 0.0;
	      sumy = 0.0;
	      sumz = 0.0;

	      for (l=0; l<num_projectors; l++){
		sumx += fp32 ? (double)(DS_VNA1[1][Mc_AN][k][m][l] * DS_VNA1[0][Mj_AN2][kl][n][l])
		                 : (double)DS_VNA1[1][Mc_AN][k][m][l] * (double)DS_VNA1[0][Mj_AN2][kl][n][l];
		sumy += fp32 ? (double)(DS_VNA1[2][Mc_AN][k][m][l] * DS_VNA1[0][Mj_AN2][kl][n][l])
		                 : (double)DS_VNA1[2][Mc_AN][k][m][l] * (double)DS_VNA1[0][Mj_AN2][kl][n][l];
		sumz += fp32 ? (double)(DS_VNA1[3][Mc_AN][k][m][l] * DS_VNA1[0][Mj_AN2][kl][n][l])
		                 : (double)DS_VNA1[3][Mc_AN][k][m][l] * (double)DS_VNA1[0][Mj_AN2][kl][n][l];
	      }

              /* for forces */

	      openmx_solver_real_matrix_row(Hx,m)[n] += sumx;
	      openmx_solver_real_matrix_row(Hy,m)[n] += sumy;
	      openmx_solver_real_matrix_row(Hz,m)[n] += sumz;

              /* for stress calculation */ 

	      OpenMX_Stress_VnaStrainRow(H_sts,1,1,m)[n] += sumx*tx;
	      OpenMX_Stress_VnaStrainRow(H_sts,1,2,m)[n] += sumx*ty;
	      OpenMX_Stress_VnaStrainRow(H_sts,1,3,m)[n] += sumx*tz;

	      OpenMX_Stress_VnaStrainRow(H_sts,2,1,m)[n] += sumy*tx;
	      OpenMX_Stress_VnaStrainRow(H_sts,2,2,m)[n] += sumy*ty;
	      OpenMX_Stress_VnaStrainRow(H_sts,2,3,m)[n] += sumy*tz;

	      OpenMX_Stress_VnaStrainRow(H_sts,3,1,m)[n] += sumz*tx;
	      OpenMX_Stress_VnaStrainRow(H_sts,3,2,m)[n] += sumz*ty;
	      OpenMX_Stress_VnaStrainRow(H_sts,3,3,m)[n] += sumz*tz;

	    } /* n */
	  } /* m */

	} /* if */

      } /* k */

      /****************************************************
 		  	     H*ep*dH 
      ****************************************************/

      /* non-local part */

      if (q_AN==0){

        /* for forces */

	for (m=0; m<ian; m++){
	  for (n=m; n<jan; n++){
            
            tmpx = openmx_solver_real_matrix_row(Hx,m)[n] + openmx_solver_real_matrix_row(Hx,n)[m];
	    openmx_solver_real_matrix_row(Hx,m)[n] = tmpx;
	    openmx_solver_real_matrix_row(Hx,n)[m] = tmpx;

            tmpy = openmx_solver_real_matrix_row(Hy,m)[n] + openmx_solver_real_matrix_row(Hy,n)[m];
	    openmx_solver_real_matrix_row(Hy,m)[n] = tmpy;
	    openmx_solver_real_matrix_row(Hy,n)[m] = tmpy;

            tmpz = openmx_solver_real_matrix_row(Hz,m)[n] + openmx_solver_real_matrix_row(Hz,n)[m];
	    openmx_solver_real_matrix_row(Hz,m)[n] = tmpz;
	    openmx_solver_real_matrix_row(Hz,n)[m] = tmpz;
	  }
	}

        /* for stress */

	for (i=1; i<=3; i++){
	  for (j=1; j<=3; j++){
	    for (m=0; m<ian; m++){
	      for (n=m; n<jan; n++){

		tmp = OpenMX_Stress_VnaStrainRow(H_sts,i,j,m)[n] + OpenMX_Stress_VnaStrainRow(H_sts,i,j,n)[m];
		OpenMX_Stress_VnaStrainRow(H_sts,i,j,m)[n] = tmp;
		OpenMX_Stress_VnaStrainRow(H_sts,i,j,n)[m] = tmp;

		tmp = OpenMX_Stress_VnaStrainRow(H_sts,i,j,m)[n] + OpenMX_Stress_VnaStrainRow(H_sts,i,j,n)[m];
		OpenMX_Stress_VnaStrainRow(H_sts,i,j,m)[n] = tmp;
		OpenMX_Stress_VnaStrainRow(H_sts,i,j,n)[m] = tmp;
	      }
	    }
	  }
	}

      }

      else if (where_flag==1) {

	kg = natn[Gc_AN][0];
	wakg = WhatSpecies[kg];

	/****************************************************
			     non-local part
	****************************************************/

	for (m=0; m<ian; m++){
	  for (n=0; n<jan; n++){

	    sumx = 0.0;
	    sumy = 0.0;
	    sumz = 0.0;

            if (Mj_AN<=Matomnum){
              Mj_AN2 = Mj_AN;
 	      kl = RMI1[Mc_AN][q_AN][0];
	    }
            else{
              Mj_AN2 = Matomnum+1; 
	      kl = RMI1[Mc_AN][0][q_AN];
            }

	    for (l=0; l<num_projectors; l++){
	      sumx -= fp32 ? (double)(DS_VNA1[0][Mc_AN][0][m][l] * DS_VNA1[1][Mj_AN2][kl][n][l])
	                       : (double)DS_VNA1[0][Mc_AN][0][m][l] * (double)DS_VNA1[1][Mj_AN2][kl][n][l];
	      sumy -= fp32 ? (double)(DS_VNA1[0][Mc_AN][0][m][l] * DS_VNA1[2][Mj_AN2][kl][n][l])
	                       : (double)DS_VNA1[0][Mc_AN][0][m][l] * (double)DS_VNA1[2][Mj_AN2][kl][n][l];
	      sumz -= fp32 ? (double)(DS_VNA1[0][Mc_AN][0][m][l] * DS_VNA1[3][Mj_AN2][kl][n][l])
	                       : (double)DS_VNA1[0][Mc_AN][0][m][l] * (double)DS_VNA1[3][Mj_AN2][kl][n][l];
	    }

	    openmx_solver_real_matrix_row(Hx,m)[n] += sumx;
	    openmx_solver_real_matrix_row(Hy,m)[n] += sumy;
	    openmx_solver_real_matrix_row(Hz,m)[n] += sumz;

	  }
	}

      }

    } /* if (h_AN==0) */

    else {

      /****************************************************
			   dH*ep*H
      ****************************************************/

      kg = natn[Gc_AN][0];
      wakg = WhatSpecies[kg];
      kl1 = RMI1[Mc_AN][0][h_AN];
      kl2 = RMI1[Mc_AN][0][q_AN];

      /****************************************************
 		         non-local part
      ****************************************************/

      for (m=0; m<ian; m++){
	for (n=0; n<jan; n++){

	  sumx = 0.0;
	  sumy = 0.0;
	  sumz = 0.0;

          for (l=0; l<num_projectors; l++){
	    sumx -= fp32 ? (double)(DS_VNA1[1][Matomnum+1][kl1][m][l] * DS_VNA1[0][Matomnum+1][kl2][n][l])
	                     : (double)DS_VNA1[1][Matomnum+1][kl1][m][l] * (double)DS_VNA1[0][Matomnum+1][kl2][n][l];
	    sumy -= fp32 ? (double)(DS_VNA1[2][Matomnum+1][kl1][m][l] * DS_VNA1[0][Matomnum+1][kl2][n][l])
	                     : (double)DS_VNA1[2][Matomnum+1][kl1][m][l] * (double)DS_VNA1[0][Matomnum+1][kl2][n][l];
	    sumz -= fp32 ? (double)(DS_VNA1[3][Matomnum+1][kl1][m][l] * DS_VNA1[0][Matomnum+1][kl2][n][l])
	                     : (double)DS_VNA1[3][Matomnum+1][kl1][m][l] * (double)DS_VNA1[0][Matomnum+1][kl2][n][l];
	  }

	  openmx_solver_real_matrix_row(Hx,m)[n] = sumx;
	  openmx_solver_real_matrix_row(Hy,m)[n] = sumy;
	  openmx_solver_real_matrix_row(Hz,m)[n] = sumz;
	}
      }

      /****************************************************
			   H*ep*dH
      ****************************************************/

      if (q_AN!=0){

	kg = natn[Gc_AN][0];
	wakg = WhatSpecies[kg];
	kl1 = RMI1[Mc_AN][0][h_AN];
	kl2 = RMI1[Mc_AN][0][q_AN];

	/****************************************************
			    non-local part
	****************************************************/

	for (m=0; m<ian; m++){
	  for (n=0; n<jan; n++){

	    sumx = 0.0;
	    sumy = 0.0;
	    sumz = 0.0;

	    for (l=0; l<num_projectors; l++){
	      sumx -= fp32 ? (double)(DS_VNA1[0][Matomnum+1][kl1][m][l] * DS_VNA1[1][Matomnum+1][kl2][n][l])
	                       : (double)DS_VNA1[0][Matomnum+1][kl1][m][l] * (double)DS_VNA1[1][Matomnum+1][kl2][n][l];
	      sumy -= fp32 ? (double)(DS_VNA1[0][Matomnum+1][kl1][m][l] * DS_VNA1[2][Matomnum+1][kl2][n][l])
	                       : (double)DS_VNA1[0][Matomnum+1][kl1][m][l] * (double)DS_VNA1[2][Matomnum+1][kl2][n][l];
	      sumz -= fp32 ? (double)(DS_VNA1[0][Matomnum+1][kl1][m][l] * DS_VNA1[3][Matomnum+1][kl2][n][l])
	                       : (double)DS_VNA1[0][Matomnum+1][kl1][m][l] * (double)DS_VNA1[3][Matomnum+1][kl2][n][l];
	    }

	    openmx_solver_real_matrix_row(Hx,m)[n] += sumx;
	    openmx_solver_real_matrix_row(Hy,m)[n] += sumy;
	    openmx_solver_real_matrix_row(Hz,m)[n] += sumz;

	  }
	}
      }

    }
  }

  /****************************************************
                contribution by dampingF 
  ****************************************************/

  /* Qij * dH/d(x,y,z) for forces */

  for (m=0; m<ian; m++){
    for (n=0; n<jan; n++){
      openmx_solver_real_matrix_row(Hx,m)[n] = dmp*openmx_solver_real_matrix_row(Hx,m)[n];
      openmx_solver_real_matrix_row(Hy,m)[n] = dmp*openmx_solver_real_matrix_row(Hy,m)[n];
      openmx_solver_real_matrix_row(Hz,m)[n] = dmp*openmx_solver_real_matrix_row(Hz,m)[n];
    }
  }

  /* Qij * H_sts for stress */

  for (i=1; i<=3; i++){
    for (j=1; j<=3; j++){
      for (m=0; m<ian; m++){
	for (n=0; n<jan; n++){
	  OpenMX_Stress_VnaStrainRow(H_sts,i,j,m)[n] = dmp*OpenMX_Stress_VnaStrainRow(H_sts,i,j,m)[n];
	  OpenMX_Stress_VnaStrainRow(H_sts,i,j,m)[n] = dmp*OpenMX_Stress_VnaStrainRow(H_sts,i,j,m)[n];
	}
      }
    }
  }

  /* dQij/d(x,y,z) * H for forces and dQij/depsilon * H for stress */

  if ( (h_AN==0 && q_AN!=0) || (h_AN!=0 && q_AN==0) ){

    if      (h_AN==0)   kl = q_AN;
    else if (q_AN==0)   kl = h_AN;

    r = Dis[Gc_AN][kl];
 
    if (rcut<=r) {
      deri_dmp = 0.0;
      tmp = 0.0; 
    }
    else {
      deri_dmp = deri_dampingF(rcut,r);
      tmp = deri_dmp/dmp;
    }

    x0 = Gxyz[ig][1] + atv[Rni][1];
    x1 = Gxyz[jg][1] + atv[Rnj][1];

    y0 = Gxyz[ig][2] + atv[Rni][2];
    y1 = Gxyz[jg][2] + atv[Rnj][2];

    z0 = Gxyz[ig][3] + atv[Rni][3];
    z1 = Gxyz[jg][3] + atv[Rnj][3];

    tx = x0 - x1;
    ty = y0 - y1;
    tz = z0 - z1;

    /* for empty atoms or finite elemens basis */
    if (r<1.0e-10) r = 1.0e-10;

    if ( h_AN==0 ){
      dx = tmp*(x0-x1)/r;
      dy = tmp*(y0-y1)/r;
      dz = tmp*(z0-z1)/r;
    }

    else if ( q_AN==0 ){
      dx = tmp*(x1-x0)/r;
      dy = tmp*(y1-y0)/r;
      dz = tmp*(z1-z0)/r;
    }

    if (h_AN==0){ 

      /* for forces */

      for (m=0; m<ian; m++){
        for (n=0; n<jan; n++){
	  openmx_solver_real_matrix_row(Hx,m)[n] += openmx_double_block_row(&vna->potential,0,Mc_AN,kl,m)[n]*dx;
	  openmx_solver_real_matrix_row(Hy,m)[n] += openmx_double_block_row(&vna->potential,0,Mc_AN,kl,m)[n]*dy;
	  openmx_solver_real_matrix_row(Hz,m)[n] += openmx_double_block_row(&vna->potential,0,Mc_AN,kl,m)[n]*dz;
        }
      }

      /* for stress */

      for (m=0; m<ian; m++){
	for (n=0; n<jan; n++){
	  OpenMX_Stress_VnaStrainRow(H_sts,1,1,m)[n] += openmx_double_block_row(&vna->potential,0,Mc_AN,kl,m)[n]*dx*tx;
	  OpenMX_Stress_VnaStrainRow(H_sts,1,2,m)[n] += openmx_double_block_row(&vna->potential,0,Mc_AN,kl,m)[n]*dx*ty;
	  OpenMX_Stress_VnaStrainRow(H_sts,1,3,m)[n] += openmx_double_block_row(&vna->potential,0,Mc_AN,kl,m)[n]*dx*tz;
	  OpenMX_Stress_VnaStrainRow(H_sts,2,1,m)[n] += openmx_double_block_row(&vna->potential,0,Mc_AN,kl,m)[n]*dy*tx;
	  OpenMX_Stress_VnaStrainRow(H_sts,2,2,m)[n] += openmx_double_block_row(&vna->potential,0,Mc_AN,kl,m)[n]*dy*ty;
	  OpenMX_Stress_VnaStrainRow(H_sts,2,3,m)[n] += openmx_double_block_row(&vna->potential,0,Mc_AN,kl,m)[n]*dy*tz;
	  OpenMX_Stress_VnaStrainRow(H_sts,3,1,m)[n] += openmx_double_block_row(&vna->potential,0,Mc_AN,kl,m)[n]*dz*tx;
	  OpenMX_Stress_VnaStrainRow(H_sts,3,2,m)[n] += openmx_double_block_row(&vna->potential,0,Mc_AN,kl,m)[n]*dz*ty;
	  OpenMX_Stress_VnaStrainRow(H_sts,3,3,m)[n] += openmx_double_block_row(&vna->potential,0,Mc_AN,kl,m)[n]*dz*tz;
	}
      }

    } /* if (h_AN==0) */ 

    else if (q_AN==0){ 

      /* for forces */

      for (m=0; m<ian; m++){
        for (n=0; n<jan; n++){
	  openmx_solver_real_matrix_row(Hx,m)[n] += openmx_double_block_row(&vna->potential,0,Mc_AN,kl,n)[m]*dx;
	  openmx_solver_real_matrix_row(Hy,m)[n] += openmx_double_block_row(&vna->potential,0,Mc_AN,kl,n)[m]*dy;
	  openmx_solver_real_matrix_row(Hz,m)[n] += openmx_double_block_row(&vna->potential,0,Mc_AN,kl,n)[m]*dz;
        }
      }

    }
  }

  /*
  printf("ABC1 Mc_AN=%2d h_AN=%2d q_AN=%2d\n",Mc_AN,h_AN,q_AN);
  for (m=0; m<ian; m++){
    for (n=0; n<jan; n++){
      printf("%10.5f ",OpenMX_Stress_VnaStrainRow(H_sts,1,1,m)[n]);
    }
    printf("\n");
  }
  */

}



void dHNL(const OpenMXAtomicViews *atomic, int where_flag,
          int Mc_AN, int h_AN, int q_AN,
          double ******DS_NL1,
          const OpenMXSolverComplexTensor *Hx,const OpenMXSolverComplexTensor *Hy,
          const OpenMXSolverComplexTensor *Hz,const OpenMXSolverComplexTensor *H_sts, const OpenMXElectronicViews *electronic)
{
  double *****HNL=electronic->HNL;
  double *****iHNL=electronic->iHNL;

  int i,j,k,m,n,l,kg,kan,so,deri_kind;
  int ig,ian,jg,jan,kl,kl1,kl2;
  int wakg,l1,l2,l3,Gc_AN,Mi_AN,Mi_AN2,Mj_AN,Mj_AN2;
  int Rni,Rnj,Rnk,somax;
  double PF[2],sumx,sumy,sumz,ene,dmp,deri_dmp;
  double tmpx,tmpy,tmpz,tmp,r;
  double x0,y0,z0,x1,y1,z1,dx,dy,dz;
  double tx,ty,tz;
  double rcuti,rcutj,rcut;
  double PFp,PFm,ene_p,ene_m;
  dcomplex sumx0,sumy0,sumz0;
  dcomplex sumx1,sumy1,sumz1;
  dcomplex sumx2,sumy2,sumz2;

  /****************************************************
   start calc.
  ****************************************************/

  Gc_AN = M2G[Mc_AN];
  ig = natn[Gc_AN][h_AN];
  Rni = ncn[Gc_AN][h_AN];
  Mi_AN = F_G2M[ig];
  ian = atomic->species.contracted_orbitals[WhatSpecies[ig]];
  rcuti = atomic->species.cutoff[WhatSpecies[ig]];

  jg = natn[Gc_AN][q_AN];
  Rnj = ncn[Gc_AN][q_AN];
  Mj_AN = F_G2M[jg];
  jan = atomic->species.contracted_orbitals[WhatSpecies[jg]];
  rcutj = atomic->species.cutoff[WhatSpecies[jg]];

  rcut = rcuti + rcutj;
  kl = RMI1[Mc_AN][h_AN][q_AN];
  dmp = dampingF(rcut,Dis[ig][kl]);

  for (so=0; so<3; so++){
    for (i=0; i<List_YOUSO[7]; i++){
      for (j=0; j<List_YOUSO[7]; j++){
	openmx_solver_complex_plane_row(Hx,so,i)[j] = Complex(0.0,0.0);
	openmx_solver_complex_plane_row(Hy,so,i)[j] = Complex(0.0,0.0);
	openmx_solver_complex_plane_row(Hz,so,i)[j] = Complex(0.0,0.0);
      }
    }
  }

  for (so=0; so<3; so++){
    for (i=0; i<4; i++){
      for (j=0; j<4; j++){
	for (m=0; m<List_YOUSO[7]; m++){
	  for (n=0; n<List_YOUSO[7]; n++){
	    OpenMX_Stress_HnlStrainRow(H_sts,so,i,j,m)[n] = Complex(0.0,0.0);
	  }
	}
      }
    }
  }

  if (h_AN==0){

    /****************************************************
                          dH*ep*H
    ****************************************************/

    for (k=0; k<=FNAN[Gc_AN]; k++){

      kg = natn[Gc_AN][k];
      Rnk = ncn[Gc_AN][k];
      wakg = WhatSpecies[kg];
      kan = atomic->species.projector_components[wakg];
      kl = RMI1[Mc_AN][q_AN][k];

      tx = Gxyz[Gc_AN][1] - Gxyz[kg][1] - atv[Rnk][1];
      ty = Gxyz[Gc_AN][2] - Gxyz[kg][2] - atv[Rnk][2];
      tz = Gxyz[Gc_AN][3] - Gxyz[kg][3] - atv[Rnk][3];

      /****************************************************
                   l-dependent non-local part
      ****************************************************/

      if (0<=kl && atomic->species.relativistic[wakg]==0 && where_flag==0){

	for (m=0; m<ian; m++){
	  for (n=0; n<jan; n++){

	    sumx = 0.0;
	    sumy = 0.0;
	    sumz = 0.0;

	    l = 0;
	    for (l1=1; l1<=atomic->species.radial_projectors[wakg]; l1++){

	      ene = openmx_radial_row3(&atomic->projector_energies,0,wakg)[l1-1];
	      if      (openmx_atomic_projector_angular(atomic,wakg)[l1]==0) l2 = 0;
	      else if (openmx_atomic_projector_angular(atomic,wakg)[l1]==1) l2 = 2;
	      else if (openmx_atomic_projector_angular(atomic,wakg)[l1]==2) l2 = 4;
	      else if (openmx_atomic_projector_angular(atomic,wakg)[l1]==3) l2 = 6;

              if (Mj_AN<=Matomnum) Mj_AN2 = Mj_AN;
              else                 Mj_AN2 = Matomnum + 1; 

	      for (l3=0; l3<=l2; l3++){
		sumx += ene*DS_NL1[0][1][Mc_AN][k][m][l]*DS_NL1[0][0][Mj_AN2][kl][n][l];
		sumy += ene*DS_NL1[0][2][Mc_AN][k][m][l]*DS_NL1[0][0][Mj_AN2][kl][n][l];
		sumz += ene*DS_NL1[0][3][Mc_AN][k][m][l]*DS_NL1[0][0][Mj_AN2][kl][n][l];
		l++;
	      }
	    }

            /* for forces */

	    openmx_solver_complex_plane_row(Hx,0,m)[n].r += sumx;
	    openmx_solver_complex_plane_row(Hy,0,m)[n].r += sumy;
	    openmx_solver_complex_plane_row(Hz,0,m)[n].r += sumz;

	    openmx_solver_complex_plane_row(Hx,1,m)[n].r += sumx;
	    openmx_solver_complex_plane_row(Hy,1,m)[n].r += sumy;
	    openmx_solver_complex_plane_row(Hz,1,m)[n].r += sumz;

            /* for stress calculation */ 

	    OpenMX_Stress_HnlStrainRow(H_sts,0,1,1,m)[n].r += sumx*tx;
	    OpenMX_Stress_HnlStrainRow(H_sts,0,1,2,m)[n].r += sumx*ty;
	    OpenMX_Stress_HnlStrainRow(H_sts,0,1,3,m)[n].r += sumx*tz;

	    OpenMX_Stress_HnlStrainRow(H_sts,0,2,1,m)[n].r += sumy*tx;
	    OpenMX_Stress_HnlStrainRow(H_sts,0,2,2,m)[n].r += sumy*ty;
	    OpenMX_Stress_HnlStrainRow(H_sts,0,2,3,m)[n].r += sumy*tz;

	    OpenMX_Stress_HnlStrainRow(H_sts,0,3,1,m)[n].r += sumz*tx;
	    OpenMX_Stress_HnlStrainRow(H_sts,0,3,2,m)[n].r += sumz*ty;
	    OpenMX_Stress_HnlStrainRow(H_sts,0,3,3,m)[n].r += sumz*tz;

	    OpenMX_Stress_HnlStrainRow(H_sts,1,1,1,m)[n].r += sumx*tx;
	    OpenMX_Stress_HnlStrainRow(H_sts,1,1,2,m)[n].r += sumx*ty;
	    OpenMX_Stress_HnlStrainRow(H_sts,1,1,3,m)[n].r += sumx*tz;

	    OpenMX_Stress_HnlStrainRow(H_sts,1,2,1,m)[n].r += sumy*tx;
	    OpenMX_Stress_HnlStrainRow(H_sts,1,2,2,m)[n].r += sumy*ty;
	    OpenMX_Stress_HnlStrainRow(H_sts,1,2,3,m)[n].r += sumy*tz;

	    OpenMX_Stress_HnlStrainRow(H_sts,1,3,1,m)[n].r += sumz*tx;
	    OpenMX_Stress_HnlStrainRow(H_sts,1,3,2,m)[n].r += sumz*ty;
	    OpenMX_Stress_HnlStrainRow(H_sts,1,3,3,m)[n].r += sumz*tz;

	  } /* n */
	} /* m */

      } /* if */

      /****************************************************
                   j-dependent non-local part
      ****************************************************/

      else if ( 0<=kl && atomic->species.relativistic[wakg]==1 && where_flag==0 ){

	for (m=0; m<ian; m++){
	  for (n=0; n<jan; n++){

	    sumx0 = Complex(0.0,0.0);
	    sumy0 = Complex(0.0,0.0);
	    sumz0 = Complex(0.0,0.0);

	    sumx1 = Complex(0.0,0.0);
	    sumy1 = Complex(0.0,0.0);
	    sumz1 = Complex(0.0,0.0);

	    sumx2 = Complex(0.0,0.0);
	    sumy2 = Complex(0.0,0.0);
	    sumz2 = Complex(0.0,0.0);

            if (Mj_AN<=Matomnum) Mj_AN2 = Mj_AN;
            else                 Mj_AN2 = Matomnum + 1; 

	    l = 0;
	    for (l1=1; l1<=atomic->species.radial_projectors[wakg]; l1++){

	      ene_p = openmx_radial_row3(&atomic->projector_energies,0,wakg)[l1-1];
	      ene_m = openmx_radial_row3(&atomic->projector_energies,1,wakg)[l1-1];

	      if      (openmx_atomic_projector_angular(atomic,wakg)[l1]==0) { l2=0; PFp=1.0;     PFm=0.0;     }
	      else if (openmx_atomic_projector_angular(atomic,wakg)[l1]==1) { l2=2; PFp=2.0/3.0; PFm=1.0/3.0; }
	      else if (openmx_atomic_projector_angular(atomic,wakg)[l1]==2) { l2=4; PFp=3.0/5.0; PFm=2.0/5.0; }
	      else if (openmx_atomic_projector_angular(atomic,wakg)[l1]==3) { l2=6; PFp=4.0/7.0; PFm=3.0/7.0; }

	      dHNL_SO(&sumx0.r,&sumy0.r,&sumz0.r,
                      &sumx1.r,&sumy1.r,&sumz1.r,
                      &sumx2.r,&sumy2.r,&sumz2.r,
                      &sumx0.i,&sumy0.i,&sumz0.i,
                      &sumx1.i,&sumy1.i,&sumz1.i,
                      &sumx2.i,&sumy2.i,&sumz2.i,
                      1.0,
                      PFp, PFm,
                      ene_p,ene_m,
                      l2, &l,
                      Mc_AN ,k, m,
                      Mj_AN2,kl,n,
                      DS_NL1);
	    }

            /* if (q_AN==0) */

            if (q_AN==0){

	      l = 0;
	      for (l1=1; l1<=atomic->species.radial_projectors[wakg]; l1++){

		ene_p = openmx_radial_row3(&atomic->projector_energies,0,wakg)[l1-1];
		ene_m = openmx_radial_row3(&atomic->projector_energies,1,wakg)[l1-1];

		if      (openmx_atomic_projector_angular(atomic,wakg)[l1]==0) { l2=0; PFp=1.0;     PFm=0.0;     }
		else if (openmx_atomic_projector_angular(atomic,wakg)[l1]==1) { l2=2; PFp=2.0/3.0; PFm=1.0/3.0; }
		else if (openmx_atomic_projector_angular(atomic,wakg)[l1]==2) { l2=4; PFp=3.0/5.0; PFm=2.0/5.0; }
		else if (openmx_atomic_projector_angular(atomic,wakg)[l1]==3) { l2=6; PFp=4.0/7.0; PFm=3.0/7.0; }

		dHNL_SO(&sumx0.r,&sumy0.r,&sumz0.r,
			&sumx1.r,&sumy1.r,&sumz1.r,
			&sumx2.r,&sumy2.r,&sumz2.r,
			&sumx0.i,&sumy0.i,&sumz0.i,
			&sumx1.i,&sumy1.i,&sumz1.i,
                        &sumx2.i,&sumy2.i,&sumz2.i,
			-1.0,
			PFp, PFm,
			ene_p,ene_m,
			l2, &l,
			Mj_AN2, kl, n,
			Mc_AN,  k,  m,
			DS_NL1);

	      }

	    } /* if (q_AN==0) */

            /* for forces */  

	    openmx_solver_complex_plane_row(Hx,0,m)[n].r += sumx0.r;     /* up-up */
	    openmx_solver_complex_plane_row(Hy,0,m)[n].r += sumy0.r;     /* up-up */
	    openmx_solver_complex_plane_row(Hz,0,m)[n].r += sumz0.r;     /* up-up */

	    openmx_solver_complex_plane_row(Hx,1,m)[n].r += sumx1.r;     /* dn-dn */
	    openmx_solver_complex_plane_row(Hy,1,m)[n].r += sumy1.r;     /* dn-dn */
	    openmx_solver_complex_plane_row(Hz,1,m)[n].r += sumz1.r;     /* dn-dn */

	    openmx_solver_complex_plane_row(Hx,2,m)[n].r += sumx2.r;     /* up-dn */
	    openmx_solver_complex_plane_row(Hy,2,m)[n].r += sumy2.r;     /* up-dn */
	    openmx_solver_complex_plane_row(Hz,2,m)[n].r += sumz2.r;     /* up-dn */

	    openmx_solver_complex_plane_row(Hx,0,m)[n].i += sumx0.i;     /* up-up */
	    openmx_solver_complex_plane_row(Hy,0,m)[n].i += sumy0.i;     /* up-up */
	    openmx_solver_complex_plane_row(Hz,0,m)[n].i += sumz0.i;     /* up-up */

	    openmx_solver_complex_plane_row(Hx,1,m)[n].i += sumx1.i;     /* dn-dn */
	    openmx_solver_complex_plane_row(Hy,1,m)[n].i += sumy1.i;     /* dn-dn */
	    openmx_solver_complex_plane_row(Hz,1,m)[n].i += sumz1.i;     /* dn-dn */

	    openmx_solver_complex_plane_row(Hx,2,m)[n].i += sumx2.i;     /* up-dn */
	    openmx_solver_complex_plane_row(Hy,2,m)[n].i += sumy2.i;     /* up-dn */
	    openmx_solver_complex_plane_row(Hz,2,m)[n].i += sumz2.i;     /* up-dn */

            /* for stress calculation */ 

            /* up-up, real */
	    OpenMX_Stress_HnlStrainRow(H_sts,0,1,1,m)[n].r += sumx0.r*tx;
	    OpenMX_Stress_HnlStrainRow(H_sts,0,1,2,m)[n].r += sumx0.r*ty;
	    OpenMX_Stress_HnlStrainRow(H_sts,0,1,3,m)[n].r += sumx0.r*tz;

	    OpenMX_Stress_HnlStrainRow(H_sts,0,2,1,m)[n].r += sumy0.r*tx;
	    OpenMX_Stress_HnlStrainRow(H_sts,0,2,2,m)[n].r += sumy0.r*ty;
	    OpenMX_Stress_HnlStrainRow(H_sts,0,2,3,m)[n].r += sumy0.r*tz;

	    OpenMX_Stress_HnlStrainRow(H_sts,0,3,1,m)[n].r += sumz0.r*tx;
	    OpenMX_Stress_HnlStrainRow(H_sts,0,3,2,m)[n].r += sumz0.r*ty;
	    OpenMX_Stress_HnlStrainRow(H_sts,0,3,3,m)[n].r += sumz0.r*tz;

            /* dn-dn, real */
	    OpenMX_Stress_HnlStrainRow(H_sts,1,1,1,m)[n].r += sumx1.r*tx;
	    OpenMX_Stress_HnlStrainRow(H_sts,1,1,2,m)[n].r += sumx1.r*ty;
	    OpenMX_Stress_HnlStrainRow(H_sts,1,1,3,m)[n].r += sumx1.r*tz;

	    OpenMX_Stress_HnlStrainRow(H_sts,1,2,1,m)[n].r += sumy1.r*tx;
	    OpenMX_Stress_HnlStrainRow(H_sts,1,2,2,m)[n].r += sumy1.r*ty;
	    OpenMX_Stress_HnlStrainRow(H_sts,1,2,3,m)[n].r += sumy1.r*tz;

	    OpenMX_Stress_HnlStrainRow(H_sts,1,3,1,m)[n].r += sumz1.r*tx;
	    OpenMX_Stress_HnlStrainRow(H_sts,1,3,2,m)[n].r += sumz1.r*ty;
	    OpenMX_Stress_HnlStrainRow(H_sts,1,3,3,m)[n].r += sumz1.r*tz;

            /* up-dn, real */
	    OpenMX_Stress_HnlStrainRow(H_sts,2,1,1,m)[n].r += sumx1.r*tx;
	    OpenMX_Stress_HnlStrainRow(H_sts,2,1,2,m)[n].r += sumx1.r*ty;
	    OpenMX_Stress_HnlStrainRow(H_sts,2,1,3,m)[n].r += sumx1.r*tz;

	    OpenMX_Stress_HnlStrainRow(H_sts,2,2,1,m)[n].r += sumy1.r*tx;
	    OpenMX_Stress_HnlStrainRow(H_sts,2,2,2,m)[n].r += sumy1.r*ty;
	    OpenMX_Stress_HnlStrainRow(H_sts,2,2,3,m)[n].r += sumy1.r*tz;

	    OpenMX_Stress_HnlStrainRow(H_sts,2,3,1,m)[n].r += sumz1.r*tx;
	    OpenMX_Stress_HnlStrainRow(H_sts,2,3,2,m)[n].r += sumz1.r*ty;
	    OpenMX_Stress_HnlStrainRow(H_sts,2,3,3,m)[n].r += sumz1.r*tz;

            /* up-up, imaginary */
	    OpenMX_Stress_HnlStrainRow(H_sts,0,1,1,m)[n].r += sumx0.i*tx;
	    OpenMX_Stress_HnlStrainRow(H_sts,0,1,2,m)[n].r += sumx0.i*ty;
	    OpenMX_Stress_HnlStrainRow(H_sts,0,1,3,m)[n].r += sumx0.i*tz;

	    OpenMX_Stress_HnlStrainRow(H_sts,0,2,1,m)[n].r += sumy0.i*tx;
	    OpenMX_Stress_HnlStrainRow(H_sts,0,2,2,m)[n].r += sumy0.i*ty;
	    OpenMX_Stress_HnlStrainRow(H_sts,0,2,3,m)[n].r += sumy0.i*tz;

	    OpenMX_Stress_HnlStrainRow(H_sts,0,3,1,m)[n].r += sumz0.i*tx;
	    OpenMX_Stress_HnlStrainRow(H_sts,0,3,2,m)[n].r += sumz0.i*ty;
	    OpenMX_Stress_HnlStrainRow(H_sts,0,3,3,m)[n].r += sumz0.i*tz;

            /* dn-dn, imaginary */
	    OpenMX_Stress_HnlStrainRow(H_sts,1,1,1,m)[n].r += sumx1.i*tx;
	    OpenMX_Stress_HnlStrainRow(H_sts,1,1,2,m)[n].r += sumx1.i*ty;
	    OpenMX_Stress_HnlStrainRow(H_sts,1,1,3,m)[n].r += sumx1.i*tz;

	    OpenMX_Stress_HnlStrainRow(H_sts,1,2,1,m)[n].r += sumy1.i*tx;
	    OpenMX_Stress_HnlStrainRow(H_sts,1,2,2,m)[n].r += sumy1.i*ty;
	    OpenMX_Stress_HnlStrainRow(H_sts,1,2,3,m)[n].r += sumy1.i*tz;

	    OpenMX_Stress_HnlStrainRow(H_sts,1,3,1,m)[n].r += sumz1.i*tx;
	    OpenMX_Stress_HnlStrainRow(H_sts,1,3,2,m)[n].r += sumz1.i*ty;
	    OpenMX_Stress_HnlStrainRow(H_sts,1,3,3,m)[n].r += sumz1.i*tz;

            /* up-dn, imaginary */
	    OpenMX_Stress_HnlStrainRow(H_sts,2,1,1,m)[n].r += sumx1.i*tx;
	    OpenMX_Stress_HnlStrainRow(H_sts,2,1,2,m)[n].r += sumx1.i*ty;
	    OpenMX_Stress_HnlStrainRow(H_sts,2,1,3,m)[n].r += sumx1.i*tz;

	    OpenMX_Stress_HnlStrainRow(H_sts,2,2,1,m)[n].r += sumy1.i*tx;
	    OpenMX_Stress_HnlStrainRow(H_sts,2,2,2,m)[n].r += sumy1.i*ty;
	    OpenMX_Stress_HnlStrainRow(H_sts,2,2,3,m)[n].r += sumy1.i*tz;

	    OpenMX_Stress_HnlStrainRow(H_sts,2,3,1,m)[n].r += sumz1.i*tx;
	    OpenMX_Stress_HnlStrainRow(H_sts,2,3,2,m)[n].r += sumz1.i*ty;
	    OpenMX_Stress_HnlStrainRow(H_sts,2,3,3,m)[n].r += sumz1.i*tz;

	  }
	}
      }

    } /* k */

    /****************************************************
                           H*ep*dH 
    ****************************************************/

    /* h_AN==0 && q_AN==0 */

    if (q_AN==0 && atomic->species.relativistic[wakg]==0){

      /* for forces */

      for (m=0; m<ian; m++){
        for (n=m; n<jan; n++){

          tmpx = openmx_solver_complex_plane_row(Hx,0,m)[n].r + openmx_solver_complex_plane_row(Hx,0,n)[m].r;
          openmx_solver_complex_plane_row(Hx,0,m)[n].r = tmpx;
          openmx_solver_complex_plane_row(Hx,0,n)[m].r = tmpx;
          openmx_solver_complex_plane_row(Hx,1,m)[n].r = tmpx;
          openmx_solver_complex_plane_row(Hx,1,n)[m].r = tmpx;

          tmpy = openmx_solver_complex_plane_row(Hy,0,m)[n].r + openmx_solver_complex_plane_row(Hy,0,n)[m].r;
          openmx_solver_complex_plane_row(Hy,0,m)[n].r = tmpy;
          openmx_solver_complex_plane_row(Hy,0,n)[m].r = tmpy;
          openmx_solver_complex_plane_row(Hy,1,m)[n].r = tmpy;
          openmx_solver_complex_plane_row(Hy,1,n)[m].r = tmpy;

          tmpz = openmx_solver_complex_plane_row(Hz,0,m)[n].r + openmx_solver_complex_plane_row(Hz,0,n)[m].r;
          openmx_solver_complex_plane_row(Hz,0,m)[n].r = tmpz;
          openmx_solver_complex_plane_row(Hz,0,n)[m].r = tmpz;
          openmx_solver_complex_plane_row(Hz,1,m)[n].r = tmpz;
          openmx_solver_complex_plane_row(Hz,1,n)[m].r = tmpz;
        }
      }

      /* for stress */

      for (i=1; i<=3; i++){
        for (j=1; j<=3; j++){
	  for (m=0; m<ian; m++){
	    for (n=m; n<jan; n++){

              tmp = OpenMX_Stress_HnlStrainRow(H_sts,0,i,j,m)[n].r + OpenMX_Stress_HnlStrainRow(H_sts,0,i,j,n)[m].r;
              OpenMX_Stress_HnlStrainRow(H_sts,0,i,j,m)[n].r = tmp;
              OpenMX_Stress_HnlStrainRow(H_sts,0,i,j,n)[m].r = tmp;

              tmp = OpenMX_Stress_HnlStrainRow(H_sts,1,i,j,m)[n].r + OpenMX_Stress_HnlStrainRow(H_sts,1,i,j,n)[m].r;
              OpenMX_Stress_HnlStrainRow(H_sts,1,i,j,m)[n].r = tmp;
              OpenMX_Stress_HnlStrainRow(H_sts,1,i,j,n)[m].r = tmp;
	    }
	  }
        }
      }

    } /* if (q_AN==0 && VPS_j_dependency[wakg]==0) */

    /*******************************************************
      where_flag==1 means that dHNL is called from 
      THE SECOND CASE:
      In case of I=k with I!=i and I!=j
      d [ \sum_k <i|k>ek<k|j> ]/dRI 

      In this case, there is no contribution to the stress.
    ********************************************************/

    else if (where_flag==1){

      kg = natn[Gc_AN][0];
      wakg = WhatSpecies[kg];
      kan = atomic->species.projector_components[wakg];
      kl = RMI1[Mc_AN][q_AN][0];

      /****************************************************
                   l-dependent non-local part
      ****************************************************/

      if (atomic->species.relativistic[wakg]==0){

	for (m=0; m<ian; m++){
	  for (n=0; n<jan; n++){

	    sumx = 0.0;
	    sumy = 0.0;
	    sumz = 0.0;

            if (Mj_AN<=Matomnum){
              Mj_AN2 = Mj_AN;
 	      kl2 = RMI1[Mc_AN][q_AN][0];
	    }
            else{
              Mj_AN2 = Matomnum + 1; 
	      kl2 = RMI1[Mc_AN][0][q_AN];
	    }

	    l = 0;
	    for (l1=1; l1<=atomic->species.radial_projectors[wakg]; l1++){

	      ene = openmx_radial_row3(&atomic->projector_energies,0,wakg)[l1-1];
	      if      (openmx_atomic_projector_angular(atomic,wakg)[l1]==0) l2 = 0;
	      else if (openmx_atomic_projector_angular(atomic,wakg)[l1]==1) l2 = 2;
	      else if (openmx_atomic_projector_angular(atomic,wakg)[l1]==2) l2 = 4;
	      else if (openmx_atomic_projector_angular(atomic,wakg)[l1]==3) l2 = 6;

	      for (l3=0; l3<=l2; l3++){

		sumx -= ene*DS_NL1[0][0][Mc_AN][0][m][l]*DS_NL1[0][1][Mj_AN2][kl2][n][l];
		sumy -= ene*DS_NL1[0][0][Mc_AN][0][m][l]*DS_NL1[0][2][Mj_AN2][kl2][n][l];
		sumz -= ene*DS_NL1[0][0][Mc_AN][0][m][l]*DS_NL1[0][3][Mj_AN2][kl2][n][l];
		l++;
	      }
	    }

            /* for forces */

	    openmx_solver_complex_plane_row(Hx,0,m)[n].r += sumx;
	    openmx_solver_complex_plane_row(Hy,0,m)[n].r += sumy;
	    openmx_solver_complex_plane_row(Hz,0,m)[n].r += sumz;

	    openmx_solver_complex_plane_row(Hx,1,m)[n].r += sumx;
	    openmx_solver_complex_plane_row(Hy,1,m)[n].r += sumy;
	    openmx_solver_complex_plane_row(Hz,1,m)[n].r += sumz;

            /* for stress */

	    OpenMX_Stress_HnlStrainRow(H_sts,0,1,1,m)[n].r += sumx*tx;
	    OpenMX_Stress_HnlStrainRow(H_sts,0,1,2,m)[n].r += sumx*ty;
	    OpenMX_Stress_HnlStrainRow(H_sts,0,1,3,m)[n].r += sumx*tz;

	    OpenMX_Stress_HnlStrainRow(H_sts,0,2,1,m)[n].r += sumy*tx;
	    OpenMX_Stress_HnlStrainRow(H_sts,0,2,2,m)[n].r += sumy*ty;
	    OpenMX_Stress_HnlStrainRow(H_sts,0,2,3,m)[n].r += sumy*tz;

	    OpenMX_Stress_HnlStrainRow(H_sts,0,3,1,m)[n].r += sumz*tx;
	    OpenMX_Stress_HnlStrainRow(H_sts,0,3,2,m)[n].r += sumz*ty;
	    OpenMX_Stress_HnlStrainRow(H_sts,0,3,3,m)[n].r += sumz*tz;

	    OpenMX_Stress_HnlStrainRow(H_sts,1,1,1,m)[n].r += sumx*tx;
	    OpenMX_Stress_HnlStrainRow(H_sts,1,1,2,m)[n].r += sumx*ty;
	    OpenMX_Stress_HnlStrainRow(H_sts,1,1,3,m)[n].r += sumx*tz;

	    OpenMX_Stress_HnlStrainRow(H_sts,1,2,1,m)[n].r += sumy*tx;
	    OpenMX_Stress_HnlStrainRow(H_sts,1,2,2,m)[n].r += sumy*ty;
	    OpenMX_Stress_HnlStrainRow(H_sts,1,2,3,m)[n].r += sumy*tz;

	    OpenMX_Stress_HnlStrainRow(H_sts,1,3,1,m)[n].r += sumz*tx;
	    OpenMX_Stress_HnlStrainRow(H_sts,1,3,2,m)[n].r += sumz*ty;
	    OpenMX_Stress_HnlStrainRow(H_sts,1,3,3,m)[n].r += sumz*tz;

	  } /* n */
	} /* m */
      } /* if (VPS_j_dependency[wakg]==0) */

      /****************************************************
                   j-dependent non-local part
      ****************************************************/

      else if ( atomic->species.relativistic[wakg]==1 ){

	for (m=0; m<ian; m++){
	  for (n=0; n<jan; n++){

	    sumx0 = Complex(0.0,0.0);
	    sumy0 = Complex(0.0,0.0);
	    sumz0 = Complex(0.0,0.0);

	    sumx1 = Complex(0.0,0.0);
	    sumy1 = Complex(0.0,0.0);
	    sumz1 = Complex(0.0,0.0);

	    sumx2 = Complex(0.0,0.0);
	    sumy2 = Complex(0.0,0.0);
	    sumz2 = Complex(0.0,0.0);

            if (Mj_AN<=Matomnum){
              Mj_AN2 = Mj_AN;
 	      kl2 = RMI1[Mc_AN][q_AN][0];
	    }
            else{
              Mj_AN2 = Matomnum + 1; 
	      kl2 = RMI1[Mc_AN][0][q_AN];
	    }

	    l = 0;
	    for (l1=1; l1<=atomic->species.radial_projectors[wakg]; l1++){

	      ene_p = openmx_radial_row3(&atomic->projector_energies,0,wakg)[l1-1];
	      ene_m = openmx_radial_row3(&atomic->projector_energies,1,wakg)[l1-1];

	      if      (openmx_atomic_projector_angular(atomic,wakg)[l1]==0) { l2=0; PFp=1.0;     PFm=0.0;     }
	      else if (openmx_atomic_projector_angular(atomic,wakg)[l1]==1) { l2=2; PFp=2.0/3.0; PFm=1.0/3.0; }
	      else if (openmx_atomic_projector_angular(atomic,wakg)[l1]==2) { l2=4; PFp=3.0/5.0; PFm=2.0/5.0; }
	      else if (openmx_atomic_projector_angular(atomic,wakg)[l1]==3) { l2=6; PFp=4.0/7.0; PFm=3.0/7.0; }

       	      /* 1 */

	      dHNL_SO(&sumx0.r,&sumy0.r,&sumz0.r,
                      &sumx1.r,&sumy1.r,&sumz1.r,
                      &sumx2.r,&sumy2.r,&sumz2.r,
                      &sumx0.i,&sumy0.i,&sumz0.i,
                      &sumx1.i,&sumy1.i,&sumz1.i,
                      &sumx2.i,&sumy2.i,&sumz2.i,
                      -1.0,
                      PFp, PFm,
                      -ene_p,-ene_m,
                      l2, &l,
                      Mj_AN2,kl2,n,
                      Mc_AN, 0,  m,
                      DS_NL1);
	    }

	    openmx_solver_complex_plane_row(Hx,0,m)[n].r += sumx0.r;     /* up-up */
	    openmx_solver_complex_plane_row(Hy,0,m)[n].r += sumy0.r;     /* up-up */
	    openmx_solver_complex_plane_row(Hz,0,m)[n].r += sumz0.r;     /* up-up */

	    openmx_solver_complex_plane_row(Hx,1,m)[n].r += sumx1.r;     /* dn-dn */
	    openmx_solver_complex_plane_row(Hy,1,m)[n].r += sumy1.r;     /* dn-dn */
	    openmx_solver_complex_plane_row(Hz,1,m)[n].r += sumz1.r;     /* dn-dn */

	    openmx_solver_complex_plane_row(Hx,2,m)[n].r += sumx2.r;     /* up-dn */
	    openmx_solver_complex_plane_row(Hy,2,m)[n].r += sumy2.r;     /* up-dn */
	    openmx_solver_complex_plane_row(Hz,2,m)[n].r += sumz2.r;     /* up-dn */

	    openmx_solver_complex_plane_row(Hx,0,m)[n].i += sumx0.i;     /* up-up */
	    openmx_solver_complex_plane_row(Hy,0,m)[n].i += sumy0.i;     /* up-up */
	    openmx_solver_complex_plane_row(Hz,0,m)[n].i += sumz0.i;     /* up-up */

	    openmx_solver_complex_plane_row(Hx,1,m)[n].i += sumx1.i;     /* dn-dn */
	    openmx_solver_complex_plane_row(Hy,1,m)[n].i += sumy1.i;     /* dn-dn */
	    openmx_solver_complex_plane_row(Hz,1,m)[n].i += sumz1.i;     /* dn-dn */

	    openmx_solver_complex_plane_row(Hx,2,m)[n].i += sumx2.i;     /* up-dn */
	    openmx_solver_complex_plane_row(Hy,2,m)[n].i += sumy2.i;     /* up-dn */
	    openmx_solver_complex_plane_row(Hz,2,m)[n].i += sumz2.i;     /* up-dn */

	  }
	}
      }

    } /* else if (where_flag==1) */

  } /* if (h_AN==0) */

  else if (where_flag==0){

    /****************************************************
       H*ep*dH

       if (h_AN!=0 && where_flag==0)
       This happens 
       only if 
       ( SpinP_switch==3
         &&
         (SO_switch==1 || (Hub_U_switch==1 && F_U_flag==1) 
  	   || 1<=Constraint_NCS_switch || Zeeman_NCS_switch==1
           || Zeeman_NCO_switch==1)
         && 
         q_AN==0
       )
    ****************************************************/

    for (k=0; k<=FNAN[Gc_AN]; k++){

      kg = natn[Gc_AN][k];
      Rnk = ncn[Gc_AN][k];
      wakg = WhatSpecies[kg];
      kan = atomic->species.projector_components[wakg];
      kl = RMI1[Mc_AN][h_AN][k];

      tx = Gxyz[Gc_AN][1] - Gxyz[kg][1] - atv[Rnk][1];
      ty = Gxyz[Gc_AN][2] - Gxyz[kg][2] - atv[Rnk][2];
      tz = Gxyz[Gc_AN][3] - Gxyz[kg][3] - atv[Rnk][3];

      if (Mi_AN<=Matomnum) Mi_AN2 = Mi_AN;
      else                 Mi_AN2 = Matomnum + 1; 

      if (0<=kl && atomic->species.relativistic[wakg]==1){

	for (m=0; m<ian; m++){
	  for (n=0; n<jan; n++){

	    sumx0 = Complex(0.0,0.0);
	    sumy0 = Complex(0.0,0.0);
	    sumz0 = Complex(0.0,0.0);

	    sumx1 = Complex(0.0,0.0);
	    sumy1 = Complex(0.0,0.0);
	    sumz1 = Complex(0.0,0.0);

	    sumx2 = Complex(0.0,0.0);
	    sumy2 = Complex(0.0,0.0);
	    sumz2 = Complex(0.0,0.0);

	    l = 0;
	    for (l1=1; l1<=atomic->species.radial_projectors[wakg]; l1++){

	      ene_p = openmx_radial_row3(&atomic->projector_energies,0,wakg)[l1-1];
	      ene_m = openmx_radial_row3(&atomic->projector_energies,1,wakg)[l1-1];

	      if      (openmx_atomic_projector_angular(atomic,wakg)[l1]==0) { l2=0; PFp=1.0;     PFm=0.0;     }
	      else if (openmx_atomic_projector_angular(atomic,wakg)[l1]==1) { l2=2; PFp=2.0/3.0; PFm=1.0/3.0; }
	      else if (openmx_atomic_projector_angular(atomic,wakg)[l1]==2) { l2=4; PFp=3.0/5.0; PFm=2.0/5.0; }
	      else if (openmx_atomic_projector_angular(atomic,wakg)[l1]==3) { l2=6; PFp=4.0/7.0; PFm=3.0/7.0; }

	      dHNL_SO(&sumx0.r,&sumy0.r,&sumz0.r,
                      &sumx1.r,&sumy1.r,&sumz1.r,
                      &sumx2.r,&sumy2.r,&sumz2.r,
                      &sumx0.i,&sumy0.i,&sumz0.i,
                      &sumx1.i,&sumy1.i,&sumz1.i,
                      &sumx2.i,&sumy2.i,&sumz2.i,
                      -1.0,
                      PFp, PFm,
                      ene_p, ene_m,
                      l2, &l,
                      Mj_AN,  k,  n,
                      Mi_AN2, kl, m,
                      DS_NL1);
	    }

            /* for forces */ 

	    openmx_solver_complex_plane_row(Hx,0,m)[n].r += sumx0.r;     /* up-up */
	    openmx_solver_complex_plane_row(Hy,0,m)[n].r += sumy0.r;     /* up-up */
	    openmx_solver_complex_plane_row(Hz,0,m)[n].r += sumz0.r;     /* up-up */

	    openmx_solver_complex_plane_row(Hx,1,m)[n].r += sumx1.r;     /* dn-dn */
	    openmx_solver_complex_plane_row(Hy,1,m)[n].r += sumy1.r;     /* dn-dn */
	    openmx_solver_complex_plane_row(Hz,1,m)[n].r += sumz1.r;     /* dn-dn */

	    openmx_solver_complex_plane_row(Hx,2,m)[n].r += sumx2.r;     /* up-dn */
	    openmx_solver_complex_plane_row(Hy,2,m)[n].r += sumy2.r;     /* up-dn */
	    openmx_solver_complex_plane_row(Hz,2,m)[n].r += sumz2.r;     /* up-dn */

	    openmx_solver_complex_plane_row(Hx,0,m)[n].i += sumx0.i;     /* up-up */
	    openmx_solver_complex_plane_row(Hy,0,m)[n].i += sumy0.i;     /* up-up */
	    openmx_solver_complex_plane_row(Hz,0,m)[n].i += sumz0.i;     /* up-up */

	    openmx_solver_complex_plane_row(Hx,1,m)[n].i += sumx1.i;     /* dn-dn */
	    openmx_solver_complex_plane_row(Hy,1,m)[n].i += sumy1.i;     /* dn-dn */
	    openmx_solver_complex_plane_row(Hz,1,m)[n].i += sumz1.i;     /* dn-dn */

	    openmx_solver_complex_plane_row(Hx,2,m)[n].i += sumx2.i;     /* up-dn */
	    openmx_solver_complex_plane_row(Hy,2,m)[n].i += sumy2.i;     /* up-dn */
	    openmx_solver_complex_plane_row(Hz,2,m)[n].i += sumz2.i;     /* up-dn */

            /* for stress calculation */ 

            /* up-up, real */
	    OpenMX_Stress_HnlStrainRow(H_sts,0,1,1,m)[n].r += sumx0.r*tx;
	    OpenMX_Stress_HnlStrainRow(H_sts,0,1,2,m)[n].r += sumx0.r*ty;
	    OpenMX_Stress_HnlStrainRow(H_sts,0,1,3,m)[n].r += sumx0.r*tz;

	    OpenMX_Stress_HnlStrainRow(H_sts,0,2,1,m)[n].r += sumy0.r*tx;
	    OpenMX_Stress_HnlStrainRow(H_sts,0,2,2,m)[n].r += sumy0.r*ty;
	    OpenMX_Stress_HnlStrainRow(H_sts,0,2,3,m)[n].r += sumy0.r*tz;

	    OpenMX_Stress_HnlStrainRow(H_sts,0,3,1,m)[n].r += sumz0.r*tx;
	    OpenMX_Stress_HnlStrainRow(H_sts,0,3,2,m)[n].r += sumz0.r*ty;
	    OpenMX_Stress_HnlStrainRow(H_sts,0,3,3,m)[n].r += sumz0.r*tz;

            /* dn-dn, real */
	    OpenMX_Stress_HnlStrainRow(H_sts,1,1,1,m)[n].r += sumx1.r*tx;
	    OpenMX_Stress_HnlStrainRow(H_sts,1,1,2,m)[n].r += sumx1.r*ty;
	    OpenMX_Stress_HnlStrainRow(H_sts,1,1,3,m)[n].r += sumx1.r*tz;

	    OpenMX_Stress_HnlStrainRow(H_sts,1,2,1,m)[n].r += sumy1.r*tx;
	    OpenMX_Stress_HnlStrainRow(H_sts,1,2,2,m)[n].r += sumy1.r*ty;
	    OpenMX_Stress_HnlStrainRow(H_sts,1,2,3,m)[n].r += sumy1.r*tz;

	    OpenMX_Stress_HnlStrainRow(H_sts,1,3,1,m)[n].r += sumz1.r*tx;
	    OpenMX_Stress_HnlStrainRow(H_sts,1,3,2,m)[n].r += sumz1.r*ty;
	    OpenMX_Stress_HnlStrainRow(H_sts,1,3,3,m)[n].r += sumz1.r*tz;

            /* up-dn, real */
	    OpenMX_Stress_HnlStrainRow(H_sts,2,1,1,m)[n].r += sumx1.r*tx;
	    OpenMX_Stress_HnlStrainRow(H_sts,2,1,2,m)[n].r += sumx1.r*ty;
	    OpenMX_Stress_HnlStrainRow(H_sts,2,1,3,m)[n].r += sumx1.r*tz;

	    OpenMX_Stress_HnlStrainRow(H_sts,2,2,1,m)[n].r += sumy1.r*tx;
	    OpenMX_Stress_HnlStrainRow(H_sts,2,2,2,m)[n].r += sumy1.r*ty;
	    OpenMX_Stress_HnlStrainRow(H_sts,2,2,3,m)[n].r += sumy1.r*tz;

	    OpenMX_Stress_HnlStrainRow(H_sts,2,3,1,m)[n].r += sumz1.r*tx;
	    OpenMX_Stress_HnlStrainRow(H_sts,2,3,2,m)[n].r += sumz1.r*ty;
	    OpenMX_Stress_HnlStrainRow(H_sts,2,3,3,m)[n].r += sumz1.r*tz;

            /* up-up, imaginary */
	    OpenMX_Stress_HnlStrainRow(H_sts,0,1,1,m)[n].r += sumx0.i*tx;
	    OpenMX_Stress_HnlStrainRow(H_sts,0,1,2,m)[n].r += sumx0.i*ty;
	    OpenMX_Stress_HnlStrainRow(H_sts,0,1,3,m)[n].r += sumx0.i*tz;

	    OpenMX_Stress_HnlStrainRow(H_sts,0,2,1,m)[n].r += sumy0.i*tx;
	    OpenMX_Stress_HnlStrainRow(H_sts,0,2,2,m)[n].r += sumy0.i*ty;
	    OpenMX_Stress_HnlStrainRow(H_sts,0,2,3,m)[n].r += sumy0.i*tz;

	    OpenMX_Stress_HnlStrainRow(H_sts,0,3,1,m)[n].r += sumz0.i*tx;
	    OpenMX_Stress_HnlStrainRow(H_sts,0,3,2,m)[n].r += sumz0.i*ty;
	    OpenMX_Stress_HnlStrainRow(H_sts,0,3,3,m)[n].r += sumz0.i*tz;

            /* dn-dn, imaginary */
	    OpenMX_Stress_HnlStrainRow(H_sts,1,1,1,m)[n].r += sumx1.i*tx;
	    OpenMX_Stress_HnlStrainRow(H_sts,1,1,2,m)[n].r += sumx1.i*ty;
	    OpenMX_Stress_HnlStrainRow(H_sts,1,1,3,m)[n].r += sumx1.i*tz;

	    OpenMX_Stress_HnlStrainRow(H_sts,1,2,1,m)[n].r += sumy1.i*tx;
	    OpenMX_Stress_HnlStrainRow(H_sts,1,2,2,m)[n].r += sumy1.i*ty;
	    OpenMX_Stress_HnlStrainRow(H_sts,1,2,3,m)[n].r += sumy1.i*tz;

	    OpenMX_Stress_HnlStrainRow(H_sts,1,3,1,m)[n].r += sumz1.i*tx;
	    OpenMX_Stress_HnlStrainRow(H_sts,1,3,2,m)[n].r += sumz1.i*ty;
	    OpenMX_Stress_HnlStrainRow(H_sts,1,3,3,m)[n].r += sumz1.i*tz;

            /* up-dn, imaginary */
	    OpenMX_Stress_HnlStrainRow(H_sts,2,1,1,m)[n].r += sumx1.i*tx;
	    OpenMX_Stress_HnlStrainRow(H_sts,2,1,2,m)[n].r += sumx1.i*ty;
	    OpenMX_Stress_HnlStrainRow(H_sts,2,1,3,m)[n].r += sumx1.i*tz;

	    OpenMX_Stress_HnlStrainRow(H_sts,2,2,1,m)[n].r += sumy1.i*tx;
	    OpenMX_Stress_HnlStrainRow(H_sts,2,2,2,m)[n].r += sumy1.i*ty;
	    OpenMX_Stress_HnlStrainRow(H_sts,2,2,3,m)[n].r += sumy1.i*tz;

	    OpenMX_Stress_HnlStrainRow(H_sts,2,3,1,m)[n].r += sumz1.i*tx;
	    OpenMX_Stress_HnlStrainRow(H_sts,2,3,2,m)[n].r += sumz1.i*ty;
	    OpenMX_Stress_HnlStrainRow(H_sts,2,3,3,m)[n].r += sumz1.i*tz;

	  } /* n */
	} /* m */
      } /* if (0<=kl && VPS_j_dependency[wakg]==1) */

    } /* k */     

  } /* else if (where_flag==0) */

  /* if (h_AN!=0 && where_flag==1) */

  /*******************************************************
    where_flag==1 means that dHNL is called from 
    THE SECOND CASE:
    In case of I=k with I!=i and I!=j
    d [ \sum_k <i|k>ek<k|j> ]/dRI 

    In this case, there is no contribution to the stress.
  ********************************************************/

  else {

    /****************************************************
                           dH*ep*H
    ****************************************************/

    kg = natn[Gc_AN][0];
    wakg = WhatSpecies[kg];
    kan = atomic->species.projector_components[wakg];
    kl1 = RMI1[Mc_AN][0][h_AN];
    kl2 = RMI1[Mc_AN][0][q_AN];

    /****************************************************
                   l-dependent non-local part
    ****************************************************/

    if (atomic->species.relativistic[wakg]==0){

      for (m=0; m<ian; m++){
	for (n=0; n<jan; n++){

	  sumx = 0.0;
	  sumy = 0.0;
	  sumz = 0.0;

	  l = 0;
	  for (l1=1; l1<=atomic->species.radial_projectors[wakg]; l1++){

	    ene = openmx_radial_row3(&atomic->projector_energies,0,wakg)[l1-1];
	    if      (openmx_atomic_projector_angular(atomic,wakg)[l1]==0) l2 = 0;
	    else if (openmx_atomic_projector_angular(atomic,wakg)[l1]==1) l2 = 2;
	    else if (openmx_atomic_projector_angular(atomic,wakg)[l1]==2) l2 = 4;
	    else if (openmx_atomic_projector_angular(atomic,wakg)[l1]==3) l2 = 6;

	    for (l3=0; l3<=l2; l3++){
	      sumx -= ene*DS_NL1[0][1][Matomnum+1][kl1][m][l]*DS_NL1[0][0][Matomnum+1][kl2][n][l];
	      sumy -= ene*DS_NL1[0][2][Matomnum+1][kl1][m][l]*DS_NL1[0][0][Matomnum+1][kl2][n][l];
	      sumz -= ene*DS_NL1[0][3][Matomnum+1][kl1][m][l]*DS_NL1[0][0][Matomnum+1][kl2][n][l];
	      l++;
	    }
	  }

	  openmx_solver_complex_plane_row(Hx,0,m)[n].r = sumx;
	  openmx_solver_complex_plane_row(Hy,0,m)[n].r = sumy;
	  openmx_solver_complex_plane_row(Hz,0,m)[n].r = sumz;

	  openmx_solver_complex_plane_row(Hx,1,m)[n].r = sumx;
	  openmx_solver_complex_plane_row(Hy,1,m)[n].r = sumy;
	  openmx_solver_complex_plane_row(Hz,1,m)[n].r = sumz;

	  openmx_solver_complex_plane_row(Hx,2,m)[n].r = 0.0;
	  openmx_solver_complex_plane_row(Hy,2,m)[n].r = 0.0;
	  openmx_solver_complex_plane_row(Hz,2,m)[n].r = 0.0;

	  openmx_solver_complex_plane_row(Hx,0,m)[n].i = 0.0;
	  openmx_solver_complex_plane_row(Hy,0,m)[n].i = 0.0;
	  openmx_solver_complex_plane_row(Hz,0,m)[n].i = 0.0;

	  openmx_solver_complex_plane_row(Hx,1,m)[n].i = 0.0;
	  openmx_solver_complex_plane_row(Hy,1,m)[n].i = 0.0;
	  openmx_solver_complex_plane_row(Hz,1,m)[n].i = 0.0;

	  openmx_solver_complex_plane_row(Hx,2,m)[n].i = 0.0;
	  openmx_solver_complex_plane_row(Hy,2,m)[n].i = 0.0;
	  openmx_solver_complex_plane_row(Hz,2,m)[n].i = 0.0;

	}
      }
    }

    /****************************************************
                 j-dependent non-local part
    ****************************************************/

    else if ( atomic->species.relativistic[wakg]==1 ){

      for (m=0; m<ian; m++){
	for (n=0; n<jan; n++){

	  sumx0 = Complex(0.0,0.0);
	  sumy0 = Complex(0.0,0.0);
	  sumz0 = Complex(0.0,0.0);

	  sumx1 = Complex(0.0,0.0);
	  sumy1 = Complex(0.0,0.0);
	  sumz1 = Complex(0.0,0.0);

          sumx2 = Complex(0.0,0.0);
	  sumy2 = Complex(0.0,0.0);
	  sumz2 = Complex(0.0,0.0);

	  l = 0;
	  for (l1=1; l1<=atomic->species.radial_projectors[wakg]; l1++){

	    ene_p = openmx_radial_row3(&atomic->projector_energies,0,wakg)[l1-1];
	    ene_m = openmx_radial_row3(&atomic->projector_energies,1,wakg)[l1-1];

	    if      (openmx_atomic_projector_angular(atomic,wakg)[l1]==0) { l2=0; PFp=1.0;     PFm=0.0;     }
	    else if (openmx_atomic_projector_angular(atomic,wakg)[l1]==1) { l2=2; PFp=2.0/3.0; PFm=1.0/3.0; }
	    else if (openmx_atomic_projector_angular(atomic,wakg)[l1]==2) { l2=4; PFp=3.0/5.0; PFm=2.0/5.0; }
	    else if (openmx_atomic_projector_angular(atomic,wakg)[l1]==3) { l2=6; PFp=4.0/7.0; PFm=3.0/7.0; }

   	    /* 2 */

	    dHNL_SO(&sumx0.r,&sumy0.r,&sumz0.r,
		    &sumx1.r,&sumy1.r,&sumz1.r,
                    &sumx2.r,&sumy2.r,&sumz2.r,
		    &sumx0.i,&sumy0.i,&sumz0.i,
		    &sumx1.i,&sumy1.i,&sumz1.i,
                    &sumx2.i,&sumy2.i,&sumz2.i,
                    1.0, 
		    PFp, PFm,
		    -ene_p,-ene_m,
		    l2, &l,
		    Matomnum+1, kl1,m,
		    Matomnum+1, kl2,n,
		    DS_NL1);
	  }

	  openmx_solver_complex_plane_row(Hx,0,m)[n].r = sumx0.r;     /* up-up */
	  openmx_solver_complex_plane_row(Hy,0,m)[n].r = sumy0.r;     /* up-up */
	  openmx_solver_complex_plane_row(Hz,0,m)[n].r = sumz0.r;     /* up-up */

	  openmx_solver_complex_plane_row(Hx,1,m)[n].r = sumx1.r;     /* dn-dn */
	  openmx_solver_complex_plane_row(Hy,1,m)[n].r = sumy1.r;     /* dn-dn */
	  openmx_solver_complex_plane_row(Hz,1,m)[n].r = sumz1.r;     /* dn-dn */

	  openmx_solver_complex_plane_row(Hx,2,m)[n].r = sumx2.r;     /* up-dn */
	  openmx_solver_complex_plane_row(Hy,2,m)[n].r = sumy2.r;     /* up-dn */
	  openmx_solver_complex_plane_row(Hz,2,m)[n].r = sumz2.r;     /* up-dn */

	  openmx_solver_complex_plane_row(Hx,0,m)[n].i = sumx0.i;     /* up-up */
	  openmx_solver_complex_plane_row(Hy,0,m)[n].i = sumy0.i;     /* up-up */
	  openmx_solver_complex_plane_row(Hz,0,m)[n].i = sumz0.i;     /* up-up */

	  openmx_solver_complex_plane_row(Hx,1,m)[n].i = sumx1.i;     /* dn-dn */
	  openmx_solver_complex_plane_row(Hy,1,m)[n].i = sumy1.i;     /* dn-dn */
	  openmx_solver_complex_plane_row(Hz,1,m)[n].i = sumz1.i;     /* dn-dn */

	  openmx_solver_complex_plane_row(Hx,2,m)[n].i = sumx2.i;     /* up-dn */
	  openmx_solver_complex_plane_row(Hy,2,m)[n].i = sumy2.i;     /* up-dn */
	  openmx_solver_complex_plane_row(Hz,2,m)[n].i = sumz2.i;     /* up-dn */

	}
      }
    }

    /****************************************************
                           H*ep*dH
    ****************************************************/

    if (q_AN!=0) {

      kg = natn[Gc_AN][0];
      wakg = WhatSpecies[kg];
      kan = atomic->species.projector_components[wakg];
      kl1 = RMI1[Mc_AN][0][h_AN];
      kl2 = RMI1[Mc_AN][0][q_AN];

      /****************************************************
                     l-dependent non-local part
      ****************************************************/

      if (atomic->species.relativistic[wakg]==0){

	for (m=0; m<ian; m++){
	  for (n=0; n<jan; n++){

	    sumx = 0.0;
	    sumy = 0.0;
	    sumz = 0.0;

	    l = 0;
	    for (l1=1; l1<=atomic->species.radial_projectors[wakg]; l1++){

	      ene = openmx_radial_row3(&atomic->projector_energies,0,wakg)[l1-1];
	      if      (openmx_atomic_projector_angular(atomic,wakg)[l1]==0) l2 = 0;
	      else if (openmx_atomic_projector_angular(atomic,wakg)[l1]==1) l2 = 2;
	      else if (openmx_atomic_projector_angular(atomic,wakg)[l1]==2) l2 = 4;
	      else if (openmx_atomic_projector_angular(atomic,wakg)[l1]==3) l2 = 6;

	      for (l3=0; l3<=l2; l3++){
		sumx -= ene*DS_NL1[0][0][Matomnum+1][kl1][m][l]*DS_NL1[0][1][Matomnum+1][kl2][n][l];
		sumy -= ene*DS_NL1[0][0][Matomnum+1][kl1][m][l]*DS_NL1[0][2][Matomnum+1][kl2][n][l];
		sumz -= ene*DS_NL1[0][0][Matomnum+1][kl1][m][l]*DS_NL1[0][3][Matomnum+1][kl2][n][l];
		l++;
	      }
	    }

	    openmx_solver_complex_plane_row(Hx,0,m)[n].r += sumx;
	    openmx_solver_complex_plane_row(Hy,0,m)[n].r += sumy;
	    openmx_solver_complex_plane_row(Hz,0,m)[n].r += sumz;

	    openmx_solver_complex_plane_row(Hx,1,m)[n].r += sumx;
	    openmx_solver_complex_plane_row(Hy,1,m)[n].r += sumy;
	    openmx_solver_complex_plane_row(Hz,1,m)[n].r += sumz;
	  }
	}
      }

      /****************************************************
                    j-dependent non-local part
      ****************************************************/

      else if ( atomic->species.relativistic[wakg]==1 ){

	for (m=0; m<ian; m++){
	  for (n=0; n<jan; n++){

	    sumx0 = Complex(0.0,0.0);
	    sumy0 = Complex(0.0,0.0);
	    sumz0 = Complex(0.0,0.0);

	    sumx1 = Complex(0.0,0.0);
	    sumy1 = Complex(0.0,0.0);
	    sumz1 = Complex(0.0,0.0);

	    sumx2 = Complex(0.0,0.0);
	    sumy2 = Complex(0.0,0.0);
	    sumz2 = Complex(0.0,0.0);

	    l = 0;
	    for (l1=1; l1<=atomic->species.radial_projectors[wakg]; l1++){

              ene_p = openmx_radial_row3(&atomic->projector_energies,0,wakg)[l1-1];
	      ene_m = openmx_radial_row3(&atomic->projector_energies,1,wakg)[l1-1];

	      if      (openmx_atomic_projector_angular(atomic,wakg)[l1]==0) { l2=0; PFp=1.0;     PFm=0.0;     }
	      else if (openmx_atomic_projector_angular(atomic,wakg)[l1]==1) { l2=2; PFp=2.0/3.0; PFm=1.0/3.0; }
	      else if (openmx_atomic_projector_angular(atomic,wakg)[l1]==2) { l2=4; PFp=3.0/5.0; PFm=2.0/5.0; }
	      else if (openmx_atomic_projector_angular(atomic,wakg)[l1]==3) { l2=6; PFp=4.0/7.0; PFm=3.0/7.0; }

	      /* 4 */

	      dHNL_SO(&sumx0.r,&sumy0.r,&sumz0.r,
		      &sumx1.r,&sumy1.r,&sumz1.r,
                      &sumx2.r,&sumy2.r,&sumz2.r,
		      &sumx0.i,&sumy0.i,&sumz0.i,
		      &sumx1.i,&sumy1.i,&sumz1.i,
                      &sumx2.i,&sumy2.i,&sumz2.i,
                      -1.0,
		      PFp, PFm,
		      -ene_p,-ene_m,
		      l2, &l,
		      Matomnum+1, kl2,n,
		      Matomnum+1, kl1,m,
		      DS_NL1);
	    }

	    openmx_solver_complex_plane_row(Hx,0,m)[n].r += sumx0.r;     /* up-up */
	    openmx_solver_complex_plane_row(Hy,0,m)[n].r += sumy0.r;     /* up-up */
	    openmx_solver_complex_plane_row(Hz,0,m)[n].r += sumz0.r;     /* up-up */

	    openmx_solver_complex_plane_row(Hx,1,m)[n].r += sumx1.r;     /* dn-dn */
	    openmx_solver_complex_plane_row(Hy,1,m)[n].r += sumy1.r;     /* dn-dn */
	    openmx_solver_complex_plane_row(Hz,1,m)[n].r += sumz1.r;     /* dn-dn */

	    openmx_solver_complex_plane_row(Hx,2,m)[n].r += sumx2.r;     /* up-dn */
	    openmx_solver_complex_plane_row(Hy,2,m)[n].r += sumy2.r;     /* up-dn */
	    openmx_solver_complex_plane_row(Hz,2,m)[n].r += sumz2.r;     /* up-dn */

	    openmx_solver_complex_plane_row(Hx,0,m)[n].i += sumx0.i;     /* up-up */
	    openmx_solver_complex_plane_row(Hy,0,m)[n].i += sumy0.i;     /* up-up */
	    openmx_solver_complex_plane_row(Hz,0,m)[n].i += sumz0.i;     /* up-up */

	    openmx_solver_complex_plane_row(Hx,1,m)[n].i += sumx1.i;     /* dn-dn */
	    openmx_solver_complex_plane_row(Hy,1,m)[n].i += sumy1.i;     /* dn-dn */
	    openmx_solver_complex_plane_row(Hz,1,m)[n].i += sumz1.i;     /* dn-dn */

	    openmx_solver_complex_plane_row(Hx,2,m)[n].i += sumx2.i;     /* up-dn */
	    openmx_solver_complex_plane_row(Hy,2,m)[n].i += sumy2.i;     /* up-dn */
	    openmx_solver_complex_plane_row(Hz,2,m)[n].i += sumz2.i;     /* up-dn */

	  }
	}
      }

    }

  } /* else */

  /****************************************************
               contribution by dampingF 
  ****************************************************/

  /* Qij * dH/d(x,y,z) for forces */

  for (so=0; so<3; so++){
    for (m=0; m<ian; m++){
      for (n=0; n<jan; n++){

        openmx_solver_complex_plane_row(Hx,so,m)[n].r = dmp*openmx_solver_complex_plane_row(Hx,so,m)[n].r;
        openmx_solver_complex_plane_row(Hy,so,m)[n].r = dmp*openmx_solver_complex_plane_row(Hy,so,m)[n].r;
        openmx_solver_complex_plane_row(Hz,so,m)[n].r = dmp*openmx_solver_complex_plane_row(Hz,so,m)[n].r;

        openmx_solver_complex_plane_row(Hx,so,m)[n].i = dmp*openmx_solver_complex_plane_row(Hx,so,m)[n].i;
        openmx_solver_complex_plane_row(Hy,so,m)[n].i = dmp*openmx_solver_complex_plane_row(Hy,so,m)[n].i;
        openmx_solver_complex_plane_row(Hz,so,m)[n].i = dmp*openmx_solver_complex_plane_row(Hz,so,m)[n].i;
      }
    }
  }

  /* Qij * H_sts for stress */

  for (so=0; so<3; so++){
    for (i=1; i<=3; i++){
      for (j=1; j<=3; j++){
	for (m=0; m<ian; m++){
	  for (n=0; n<jan; n++){
	    OpenMX_Stress_HnlStrainRow(H_sts,so,i,j,m)[n].r = dmp*OpenMX_Stress_HnlStrainRow(H_sts,so,i,j,m)[n].r;
	    OpenMX_Stress_HnlStrainRow(H_sts,so,i,j,m)[n].i = dmp*OpenMX_Stress_HnlStrainRow(H_sts,so,i,j,m)[n].i;
	  }
	}
      }
    }
  }

  /* dQij/d(x,y,z) * H for forces and dQij/depsilon * H for stress */

  if ( (h_AN==0 && q_AN!=0) || (h_AN!=0 && q_AN==0) ){

    if      (h_AN==0)   kl = q_AN;
    else if (q_AN==0)   kl = h_AN;

    if      (SpinP_switch==0) somax = 0;
    else if (SpinP_switch==1) somax = 1;
    else if (SpinP_switch==3) somax = 2;

    r = Dis[Gc_AN][kl];

    if (rcut<=r) {
      deri_dmp = 0.0;
      tmp = 0.0; 
    }
    else {
      deri_dmp = deri_dampingF(rcut,r);
      tmp = deri_dmp/dmp;
    }

    x0 = Gxyz[ig][1] + atv[Rni][1];
    y0 = Gxyz[ig][2] + atv[Rni][2];
    z0 = Gxyz[ig][3] + atv[Rni][3];

    x1 = Gxyz[jg][1] + atv[Rnj][1];
    y1 = Gxyz[jg][2] + atv[Rnj][2];
    z1 = Gxyz[jg][3] + atv[Rnj][3];

    tx = x0 - x1;
    ty = y0 - y1;
    tz = z0 - z1;

    /* for empty atoms or finite elemens basis */
    if (r<1.0e-10) r = 1.0e-10;

    if (h_AN==0 && q_AN!=0){
      dx = tmp*(x0-x1)/r;
      dy = tmp*(y0-y1)/r;
      dz = tmp*(z0-z1)/r;
    }

    else if (h_AN!=0 && q_AN==0){
      dx = tmp*(x1-x0)/r;
      dy = tmp*(y1-y0)/r;
      dz = tmp*(z1-z0)/r;
    }

    if (SpinP_switch==0 || SpinP_switch==1){

      if (h_AN==0){ 

        /* for forces */

        for (so=0; so<=somax; so++){
	  for (m=0; m<ian; m++){
	    for (n=0; n<jan; n++){
	      openmx_solver_complex_plane_row(Hx,so,m)[n].r += HNL[so][Mc_AN][kl][m][n]*dx;
	      openmx_solver_complex_plane_row(Hy,so,m)[n].r += HNL[so][Mc_AN][kl][m][n]*dy;
	      openmx_solver_complex_plane_row(Hz,so,m)[n].r += HNL[so][Mc_AN][kl][m][n]*dz;
	    }
	  }
        }

        /* for stress */

        for (so=0; so<=somax; so++){
	  for (m=0; m<ian; m++){
	    for (n=0; n<jan; n++){
	      OpenMX_Stress_HnlStrainRow(H_sts,so,1,1,m)[n].r += HNL[so][Mc_AN][kl][m][n]*dx*tx;
	      OpenMX_Stress_HnlStrainRow(H_sts,so,1,2,m)[n].r += HNL[so][Mc_AN][kl][m][n]*dx*ty;
	      OpenMX_Stress_HnlStrainRow(H_sts,so,1,3,m)[n].r += HNL[so][Mc_AN][kl][m][n]*dx*tz;
	      OpenMX_Stress_HnlStrainRow(H_sts,so,2,1,m)[n].r += HNL[so][Mc_AN][kl][m][n]*dy*tx;
	      OpenMX_Stress_HnlStrainRow(H_sts,so,2,2,m)[n].r += HNL[so][Mc_AN][kl][m][n]*dy*ty;
	      OpenMX_Stress_HnlStrainRow(H_sts,so,2,3,m)[n].r += HNL[so][Mc_AN][kl][m][n]*dy*tz;
	      OpenMX_Stress_HnlStrainRow(H_sts,so,3,1,m)[n].r += HNL[so][Mc_AN][kl][m][n]*dz*tx;
	      OpenMX_Stress_HnlStrainRow(H_sts,so,3,2,m)[n].r += HNL[so][Mc_AN][kl][m][n]*dz*ty;
	      OpenMX_Stress_HnlStrainRow(H_sts,so,3,3,m)[n].r += HNL[so][Mc_AN][kl][m][n]*dz*tz;
	    }
	  }
        }

      }

      else if (q_AN==0){ 

        /* for forces */

        for (so=0; so<=somax; so++){
	  for (m=0; m<ian; m++){
	    for (n=0; n<jan; n++){
	      openmx_solver_complex_plane_row(Hx,so,m)[n].r += HNL[so][Mc_AN][kl][n][m]*dx;
	      openmx_solver_complex_plane_row(Hy,so,m)[n].r += HNL[so][Mc_AN][kl][n][m]*dy;
	      openmx_solver_complex_plane_row(Hz,so,m)[n].r += HNL[so][Mc_AN][kl][n][m]*dz;
	    }
	  }
        }
      }
    }

    else if (SpinP_switch==3){

      if (h_AN==0){ 

        /* for forces */

        for (so=0; so<=somax; so++){
	  for (m=0; m<ian; m++){
	    for (n=0; n<jan; n++){
	      openmx_solver_complex_plane_row(Hx,so,m)[n].r +=  HNL[so][Mc_AN][kl][m][n]*dx;
	      openmx_solver_complex_plane_row(Hy,so,m)[n].r +=  HNL[so][Mc_AN][kl][m][n]*dy;
	      openmx_solver_complex_plane_row(Hz,so,m)[n].r +=  HNL[so][Mc_AN][kl][m][n]*dz;
	    }
	  }
        }

        /* for stress */

        for (so=0; so<=somax; so++){
	  for (m=0; m<ian; m++){
	    for (n=0; n<jan; n++){
	      OpenMX_Stress_HnlStrainRow(H_sts,so,1,1,m)[n].r += HNL[so][Mc_AN][kl][m][n]*dx*tx;
	      OpenMX_Stress_HnlStrainRow(H_sts,so,1,2,m)[n].r += HNL[so][Mc_AN][kl][m][n]*dx*ty;
	      OpenMX_Stress_HnlStrainRow(H_sts,so,1,3,m)[n].r += HNL[so][Mc_AN][kl][m][n]*dx*tz;
	      OpenMX_Stress_HnlStrainRow(H_sts,so,2,1,m)[n].r += HNL[so][Mc_AN][kl][m][n]*dy*tx;
	      OpenMX_Stress_HnlStrainRow(H_sts,so,2,2,m)[n].r += HNL[so][Mc_AN][kl][m][n]*dy*ty;
	      OpenMX_Stress_HnlStrainRow(H_sts,so,2,3,m)[n].r += HNL[so][Mc_AN][kl][m][n]*dy*tz;
	      OpenMX_Stress_HnlStrainRow(H_sts,so,3,1,m)[n].r += HNL[so][Mc_AN][kl][m][n]*dz*tx;
	      OpenMX_Stress_HnlStrainRow(H_sts,so,3,2,m)[n].r += HNL[so][Mc_AN][kl][m][n]*dz*ty;
	      OpenMX_Stress_HnlStrainRow(H_sts,so,3,3,m)[n].r += HNL[so][Mc_AN][kl][m][n]*dz*tz;
	    }
	  }
        }
      }

      else if (q_AN==0){ 

        /* for forces */

        for (so=0; so<=somax; so++){
	  for (m=0; m<ian; m++){
	    for (n=0; n<jan; n++){
	      openmx_solver_complex_plane_row(Hx,so,m)[n].r +=  HNL[so][Mc_AN][kl][n][m]*dx;
	      openmx_solver_complex_plane_row(Hy,so,m)[n].r +=  HNL[so][Mc_AN][kl][n][m]*dy;
	      openmx_solver_complex_plane_row(Hz,so,m)[n].r +=  HNL[so][Mc_AN][kl][n][m]*dz;
	    }
	  }
        }
      }

      if (SO_switch==1){

        if (h_AN==0){ 

          /* for forces */

	  for (so=0; so<=somax; so++){
	    for (m=0; m<ian; m++){
	      for (n=0; n<jan; n++){
		openmx_solver_complex_plane_row(Hx,so,m)[n].i += iHNL[so][Mc_AN][kl][m][n]*dx;
		openmx_solver_complex_plane_row(Hy,so,m)[n].i += iHNL[so][Mc_AN][kl][m][n]*dy;
		openmx_solver_complex_plane_row(Hz,so,m)[n].i += iHNL[so][Mc_AN][kl][m][n]*dz;
	      }
	    }
	  }

          /* for stress */

	  for (so=0; so<=somax; so++){
	    for (m=0; m<ian; m++){
	      for (n=0; n<jan; n++){
		OpenMX_Stress_HnlStrainRow(H_sts,so,1,1,m)[n].i += iHNL[so][Mc_AN][kl][m][n]*dx*tx;
		OpenMX_Stress_HnlStrainRow(H_sts,so,1,2,m)[n].i += iHNL[so][Mc_AN][kl][m][n]*dx*ty;
		OpenMX_Stress_HnlStrainRow(H_sts,so,1,3,m)[n].i += iHNL[so][Mc_AN][kl][m][n]*dx*tz;
		OpenMX_Stress_HnlStrainRow(H_sts,so,2,1,m)[n].i += iHNL[so][Mc_AN][kl][m][n]*dy*tx;
		OpenMX_Stress_HnlStrainRow(H_sts,so,2,2,m)[n].i += iHNL[so][Mc_AN][kl][m][n]*dy*ty;
		OpenMX_Stress_HnlStrainRow(H_sts,so,2,3,m)[n].i += iHNL[so][Mc_AN][kl][m][n]*dy*tz;
		OpenMX_Stress_HnlStrainRow(H_sts,so,3,1,m)[n].i += iHNL[so][Mc_AN][kl][m][n]*dz*tx;
		OpenMX_Stress_HnlStrainRow(H_sts,so,3,2,m)[n].i += iHNL[so][Mc_AN][kl][m][n]*dz*ty;
		OpenMX_Stress_HnlStrainRow(H_sts,so,3,3,m)[n].i += iHNL[so][Mc_AN][kl][m][n]*dz*tz;
	      }
	    }
	  }
	}

        else if (q_AN==0){ 

          /* for forces */

	  for (so=0; so<=somax; so++){
	    for (m=0; m<ian; m++){
	      for (n=0; n<jan; n++){
		openmx_solver_complex_plane_row(Hx,so,m)[n].i += iHNL[so][Mc_AN][kl][n][m]*dx;
		openmx_solver_complex_plane_row(Hy,so,m)[n].i += iHNL[so][Mc_AN][kl][n][m]*dy;
		openmx_solver_complex_plane_row(Hz,so,m)[n].i += iHNL[so][Mc_AN][kl][n][m]*dz;
	      }
	    }
	  }
	}

      }
    }
  }

}


void dHNL_SO(
	     double *sumx0r,
	     double *sumy0r, 
	     double *sumz0r, 
	     double *sumx1r,
	     double *sumy1r, 
	     double *sumz1r, 
	     double *sumx2r,
	     double *sumy2r, 
	     double *sumz2r, 
	     double *sumx0i,
	     double *sumy0i, 
	     double *sumz0i, 
	     double *sumx1i,
	     double *sumy1i, 
	     double *sumz1i, 
	     double *sumx2i,
	     double *sumy2i, 
	     double *sumz2i, 
             double fugou,
	     double PFp,
	     double PFm,
	     double ene_p,
	     double ene_m,
	     int l2, int *l,
	     int Mc_AN, int k,  int m,
	     int Mj_AN, int kl, int n,
	     double ******DS_NL1)
{

  int l3,i;
  double tmpx,tmpy,tmpz;
  double tmp0,tmp1,tmp2;
  double tmp3,tmp4,tmp5,tmp6;
  double deri[4];

  /****************************************************
    off-diagonal contribution to up-dn matrix
    for spin non-collinear
  ****************************************************/

  if (SpinP_switch==3){

    /* p */ 
    if (l2==2){

      /* real contribution of l+1/2 to off diagonal up-down matrix */ 
      tmpx = 
        fugou*
        ( ene_p/3.0*DS_NL1[0][1][Mc_AN][k][m][*l  ]*DS_NL1[0][0][Mj_AN][kl][n][*l+2]
         -ene_p/3.0*DS_NL1[0][1][Mc_AN][k][m][*l+2]*DS_NL1[0][0][Mj_AN][kl][n][*l  ] );      

      tmpy = 
        fugou*
        ( ene_p/3.0*DS_NL1[0][2][Mc_AN][k][m][*l  ]*DS_NL1[0][0][Mj_AN][kl][n][*l+2]
         -ene_p/3.0*DS_NL1[0][2][Mc_AN][k][m][*l+2]*DS_NL1[0][0][Mj_AN][kl][n][*l  ] );      

      tmpz = 
        fugou*
        ( ene_p/3.0*DS_NL1[0][3][Mc_AN][k][m][*l  ]*DS_NL1[0][0][Mj_AN][kl][n][*l+2]
         -ene_p/3.0*DS_NL1[0][3][Mc_AN][k][m][*l+2]*DS_NL1[0][0][Mj_AN][kl][n][*l  ] );      

      *sumx2r += tmpx;
      *sumy2r += tmpy;
      *sumz2r += tmpz;

      /* imaginary contribution of l+1/2 to off diagonal up-down matrix */ 

      tmpx = 
        fugou*
        ( -ene_p/3.0*DS_NL1[0][1][Mc_AN][k][m][*l+1]*DS_NL1[0][0][Mj_AN][kl][n][*l+2]
          +ene_p/3.0*DS_NL1[0][1][Mc_AN][k][m][*l+2]*DS_NL1[0][0][Mj_AN][kl][n][*l+1] );

      tmpy = 
        fugou*
        ( -ene_p/3.0*DS_NL1[0][2][Mc_AN][k][m][*l+1]*DS_NL1[0][0][Mj_AN][kl][n][*l+2]
          +ene_p/3.0*DS_NL1[0][2][Mc_AN][k][m][*l+2]*DS_NL1[0][0][Mj_AN][kl][n][*l+1] );

      tmpz = 
        fugou*
        ( -ene_p/3.0*DS_NL1[0][3][Mc_AN][k][m][*l+1]*DS_NL1[0][0][Mj_AN][kl][n][*l+2]
          +ene_p/3.0*DS_NL1[0][3][Mc_AN][k][m][*l+2]*DS_NL1[0][0][Mj_AN][kl][n][*l+1] );

      *sumx2i += tmpx;
      *sumy2i += tmpy;
      *sumz2i += tmpz;

       /* real contribution of l-1/2 for to diagonal up-down matrix */ 

      tmpx = 
        fugou*
        ( ene_m/3.0*DS_NL1[1][1][Mc_AN][k][m][*l  ]*DS_NL1[1][0][Mj_AN][kl][n][*l+2]
         -ene_m/3.0*DS_NL1[1][1][Mc_AN][k][m][*l+2]*DS_NL1[1][0][Mj_AN][kl][n][*l  ] );

      tmpy = 
        fugou*
        ( ene_m/3.0*DS_NL1[1][2][Mc_AN][k][m][*l  ]*DS_NL1[1][0][Mj_AN][kl][n][*l+2]
         -ene_m/3.0*DS_NL1[1][2][Mc_AN][k][m][*l+2]*DS_NL1[1][0][Mj_AN][kl][n][*l  ] );

      tmpz = 
        fugou*
        ( ene_m/3.0*DS_NL1[1][3][Mc_AN][k][m][*l  ]*DS_NL1[1][0][Mj_AN][kl][n][*l+2]
         -ene_m/3.0*DS_NL1[1][3][Mc_AN][k][m][*l+2]*DS_NL1[1][0][Mj_AN][kl][n][*l  ] );

      *sumx2r -= tmpx;
      *sumy2r -= tmpy;
      *sumz2r -= tmpz;

       /* imaginary contribution of l-1/2 to off diagonal up-down matrix */ 

      tmpx = 
        fugou*
        ( -ene_m/3.0*DS_NL1[1][1][Mc_AN][k][m][*l+1]*DS_NL1[1][0][Mj_AN][kl][n][*l+2]
          +ene_m/3.0*DS_NL1[1][1][Mc_AN][k][m][*l+2]*DS_NL1[1][0][Mj_AN][kl][n][*l+1] );   

      tmpy = 
        fugou*
        ( -ene_m/3.0*DS_NL1[1][2][Mc_AN][k][m][*l+1]*DS_NL1[1][0][Mj_AN][kl][n][*l+2]
          +ene_m/3.0*DS_NL1[1][2][Mc_AN][k][m][*l+2]*DS_NL1[1][0][Mj_AN][kl][n][*l+1] );   

      tmpz = 
        fugou*
        ( -ene_m/3.0*DS_NL1[1][3][Mc_AN][k][m][*l+1]*DS_NL1[1][0][Mj_AN][kl][n][*l+2]
          +ene_m/3.0*DS_NL1[1][3][Mc_AN][k][m][*l+2]*DS_NL1[1][0][Mj_AN][kl][n][*l+1] );   

      *sumx2i -= tmpx;
      *sumy2i -= tmpy;
      *sumz2i -= tmpz;

    }

    /* d */ 
    if (l2==4){

       tmp0 = sqrt(3.0);
       tmp1 = ene_p/5.0; 
       tmp2 = tmp0*tmp1;

       /* real contribution of l+1/2 to off diagonal up-down matrix */ 

       for (i=1; i<=3; i++){
         deri[i] = 
            fugou* 
            ( -tmp2*DS_NL1[0][i][Mc_AN][k][m][*l  ]*DS_NL1[0][0][Mj_AN][kl][n][*l+3]
              +tmp2*DS_NL1[0][i][Mc_AN][k][m][*l+3]*DS_NL1[0][0][Mj_AN][kl][n][*l  ]
              +tmp1*DS_NL1[0][i][Mc_AN][k][m][*l+1]*DS_NL1[0][0][Mj_AN][kl][n][*l+3]
              -tmp1*DS_NL1[0][i][Mc_AN][k][m][*l+3]*DS_NL1[0][0][Mj_AN][kl][n][*l+1]
              +tmp1*DS_NL1[0][i][Mc_AN][k][m][*l+2]*DS_NL1[0][0][Mj_AN][kl][n][*l+4]
              -tmp1*DS_NL1[0][i][Mc_AN][k][m][*l+4]*DS_NL1[0][0][Mj_AN][kl][n][*l+2] );
       }
       *sumx2r += deri[1];
       *sumy2r += deri[2];
       *sumz2r += deri[3];

       /* imaginary contribution of l+1/2 to off diagonal up-down matrix */ 

       for (i=1; i<=3; i++){
         deri[i] = 
            fugou* 
            ( +tmp2*DS_NL1[0][i][Mc_AN][k][m][*l  ]*DS_NL1[0][0][Mj_AN][kl][n][*l+4]
              -tmp2*DS_NL1[0][i][Mc_AN][k][m][*l+4]*DS_NL1[0][0][Mj_AN][kl][n][*l  ]
              +tmp1*DS_NL1[0][i][Mc_AN][k][m][*l+1]*DS_NL1[0][0][Mj_AN][kl][n][*l+4]
              -tmp1*DS_NL1[0][i][Mc_AN][k][m][*l+4]*DS_NL1[0][0][Mj_AN][kl][n][*l+1]
              -tmp1*DS_NL1[0][i][Mc_AN][k][m][*l+2]*DS_NL1[0][0][Mj_AN][kl][n][*l+3]
              +tmp1*DS_NL1[0][i][Mc_AN][k][m][*l+3]*DS_NL1[0][0][Mj_AN][kl][n][*l+2] );
       }
       *sumx2i += deri[1];
       *sumy2i += deri[2];
       *sumz2i += deri[3];

       /* real contribution of l-1/2 for to diagonal up-down matrix */ 

       tmp1 = ene_m/5.0; 
       tmp2 = tmp0*tmp1;

       for (i=1; i<=3; i++){
         deri[i] = 
            fugou* 
            ( -tmp2*DS_NL1[1][i][Mc_AN][k][m][*l  ]*DS_NL1[1][0][Mj_AN][kl][n][*l+3]
              +tmp2*DS_NL1[1][i][Mc_AN][k][m][*l+3]*DS_NL1[1][0][Mj_AN][kl][n][*l  ]
              +tmp1*DS_NL1[1][i][Mc_AN][k][m][*l+1]*DS_NL1[1][0][Mj_AN][kl][n][*l+3]
              -tmp1*DS_NL1[1][i][Mc_AN][k][m][*l+3]*DS_NL1[1][0][Mj_AN][kl][n][*l+1]
              +tmp1*DS_NL1[1][i][Mc_AN][k][m][*l+2]*DS_NL1[1][0][Mj_AN][kl][n][*l+4]
              -tmp1*DS_NL1[1][i][Mc_AN][k][m][*l+4]*DS_NL1[1][0][Mj_AN][kl][n][*l+2] );
       }
       *sumx2r -= deri[1];
       *sumy2r -= deri[2];
       *sumz2r -= deri[3];

       /* imaginary contribution of l-1/2 to off diagonal up-down matrix */ 

       for (i=1; i<=3; i++){
         deri[i] = 
            fugou* 
	    ( +tmp2*DS_NL1[1][i][Mc_AN][k][m][*l  ]*DS_NL1[1][0][Mj_AN][kl][n][*l+4]
              -tmp2*DS_NL1[1][i][Mc_AN][k][m][*l+4]*DS_NL1[1][0][Mj_AN][kl][n][*l  ]
              +tmp1*DS_NL1[1][i][Mc_AN][k][m][*l+1]*DS_NL1[1][0][Mj_AN][kl][n][*l+4]
              -tmp1*DS_NL1[1][i][Mc_AN][k][m][*l+4]*DS_NL1[1][0][Mj_AN][kl][n][*l+1]
              -tmp1*DS_NL1[1][i][Mc_AN][k][m][*l+2]*DS_NL1[1][0][Mj_AN][kl][n][*l+3]
	      +tmp1*DS_NL1[1][i][Mc_AN][k][m][*l+3]*DS_NL1[1][0][Mj_AN][kl][n][*l+2] );
       }
       *sumx2i -= deri[1];
       *sumy2i -= deri[2];
       *sumz2i -= deri[3];
    }

    /* f */ 
    if (l2==6){

      /* real contribution of l+1/2 to off diagonal up-down matrix */ 

      tmp0 = sqrt(6.0);
      tmp1 = sqrt(3.0/2.0);
      tmp2 = sqrt(5.0/2.0);

      tmp3 = ene_p/7.0; 
      tmp4 = tmp1*tmp3; /* sqrt(3.0/2.0) */
      tmp5 = tmp2*tmp3; /* sqrt(5.0/2.0) */
      tmp6 = tmp0*tmp3; /* sqrt(6.0)     */

       for (i=1; i<=3; i++){
         deri[i] = 
	   fugou*
             ( -tmp6*DS_NL1[0][i][Mc_AN][k][m][*l  ]*DS_NL1[0][0][Mj_AN][kl][n][*l+1]
               +tmp6*DS_NL1[0][i][Mc_AN][k][m][*l+1]*DS_NL1[0][0][Mj_AN][kl][n][*l  ]
               -tmp5*DS_NL1[0][i][Mc_AN][k][m][*l+1]*DS_NL1[0][0][Mj_AN][kl][n][*l+3]
               +tmp5*DS_NL1[0][i][Mc_AN][k][m][*l+3]*DS_NL1[0][0][Mj_AN][kl][n][*l+1]
               -tmp5*DS_NL1[0][i][Mc_AN][k][m][*l+2]*DS_NL1[0][0][Mj_AN][kl][n][*l+4]
	       +tmp5*DS_NL1[0][i][Mc_AN][k][m][*l+4]*DS_NL1[0][0][Mj_AN][kl][n][*l+2]
               -tmp4*DS_NL1[0][i][Mc_AN][k][m][*l+3]*DS_NL1[0][0][Mj_AN][kl][n][*l+5]
	       +tmp4*DS_NL1[0][i][Mc_AN][k][m][*l+5]*DS_NL1[0][0][Mj_AN][kl][n][*l+3]
               -tmp4*DS_NL1[0][i][Mc_AN][k][m][*l+4]*DS_NL1[0][0][Mj_AN][kl][n][*l+6]
               +tmp4*DS_NL1[0][i][Mc_AN][k][m][*l+6]*DS_NL1[0][0][Mj_AN][kl][n][*l+4] ); 
       }
       *sumx2r += deri[1];
       *sumy2r += deri[2];
       *sumz2r += deri[3];

       /* imaginary contribution of l+1/2 to off diagonal up-down matrix */ 

       for (i=1; i<=3; i++){
         deri[i] = 
	   fugou*
	     ( +tmp6*DS_NL1[0][i][Mc_AN][k][m][*l  ]*DS_NL1[0][0][Mj_AN][kl][n][*l+2]
               -tmp6*DS_NL1[0][i][Mc_AN][k][m][*l+2]*DS_NL1[0][0][Mj_AN][kl][n][*l  ]
               +tmp5*DS_NL1[0][i][Mc_AN][k][m][*l+1]*DS_NL1[0][0][Mj_AN][kl][n][*l+4]
               -tmp5*DS_NL1[0][i][Mc_AN][k][m][*l+4]*DS_NL1[0][0][Mj_AN][kl][n][*l+1]
               -tmp5*DS_NL1[0][i][Mc_AN][k][m][*l+2]*DS_NL1[0][0][Mj_AN][kl][n][*l+3]
	       +tmp5*DS_NL1[0][i][Mc_AN][k][m][*l+3]*DS_NL1[0][0][Mj_AN][kl][n][*l+2]
               +tmp4*DS_NL1[0][i][Mc_AN][k][m][*l+3]*DS_NL1[0][0][Mj_AN][kl][n][*l+6]
	       -tmp4*DS_NL1[0][i][Mc_AN][k][m][*l+6]*DS_NL1[0][0][Mj_AN][kl][n][*l+3]
               -tmp4*DS_NL1[0][i][Mc_AN][k][m][*l+4]*DS_NL1[0][0][Mj_AN][kl][n][*l+5]
               +tmp4*DS_NL1[0][i][Mc_AN][k][m][*l+5]*DS_NL1[0][0][Mj_AN][kl][n][*l+4] );
       }
       *sumx2i += deri[1];
       *sumy2i += deri[2];
       *sumz2i += deri[3];

       /* real contribution of l-1/2 for to diagonal up-down matrix */ 

       tmp3 = ene_m/7.0; 
       tmp4 = tmp1*tmp3; /* sqrt(3.0/2.0) */
       tmp5 = tmp2*tmp3; /* sqrt(5.0/2.0) */
       tmp6 = tmp0*tmp3; /* sqrt(6.0)     */

       for (i=1; i<=3; i++){
         deri[i] = 
	   fugou*
	     ( -tmp6*DS_NL1[1][i][Mc_AN][k][m][*l  ]*DS_NL1[1][0][Mj_AN][kl][n][*l+1]
               +tmp6*DS_NL1[1][i][Mc_AN][k][m][*l+1]*DS_NL1[1][0][Mj_AN][kl][n][*l  ]
               -tmp5*DS_NL1[1][i][Mc_AN][k][m][*l+1]*DS_NL1[1][0][Mj_AN][kl][n][*l+3]
               +tmp5*DS_NL1[1][i][Mc_AN][k][m][*l+3]*DS_NL1[1][0][Mj_AN][kl][n][*l+1]
               -tmp5*DS_NL1[1][i][Mc_AN][k][m][*l+2]*DS_NL1[1][0][Mj_AN][kl][n][*l+4]
	       +tmp5*DS_NL1[1][i][Mc_AN][k][m][*l+4]*DS_NL1[1][0][Mj_AN][kl][n][*l+2]
               -tmp4*DS_NL1[1][i][Mc_AN][k][m][*l+3]*DS_NL1[1][0][Mj_AN][kl][n][*l+5]
	       +tmp4*DS_NL1[1][i][Mc_AN][k][m][*l+5]*DS_NL1[1][0][Mj_AN][kl][n][*l+3]
               -tmp4*DS_NL1[1][i][Mc_AN][k][m][*l+4]*DS_NL1[1][0][Mj_AN][kl][n][*l+6]
               +tmp4*DS_NL1[1][i][Mc_AN][k][m][*l+6]*DS_NL1[1][0][Mj_AN][kl][n][*l+4] );
       }
       *sumx2r -= deri[1];
       *sumy2r -= deri[2];
       *sumz2r -= deri[3];

       /* imaginary contribution of l-1/2 to off diagonal up-down matrix */ 

       for (i=1; i<=3; i++){
         deri[i] = 
            fugou* 
	      ( +tmp6*DS_NL1[1][i][Mc_AN][k][m][*l  ]*DS_NL1[1][0][Mj_AN][kl][n][*l+2]
                -tmp6*DS_NL1[1][i][Mc_AN][k][m][*l+2]*DS_NL1[1][0][Mj_AN][kl][n][*l  ]
                +tmp5*DS_NL1[1][i][Mc_AN][k][m][*l+1]*DS_NL1[1][0][Mj_AN][kl][n][*l+4]
                -tmp5*DS_NL1[1][i][Mc_AN][k][m][*l+4]*DS_NL1[1][0][Mj_AN][kl][n][*l+1]
                -tmp5*DS_NL1[1][i][Mc_AN][k][m][*l+2]*DS_NL1[1][0][Mj_AN][kl][n][*l+3]
	        +tmp5*DS_NL1[1][i][Mc_AN][k][m][*l+3]*DS_NL1[1][0][Mj_AN][kl][n][*l+2]
                +tmp4*DS_NL1[1][i][Mc_AN][k][m][*l+3]*DS_NL1[1][0][Mj_AN][kl][n][*l+6]
	        -tmp4*DS_NL1[1][i][Mc_AN][k][m][*l+6]*DS_NL1[1][0][Mj_AN][kl][n][*l+3]
                -tmp4*DS_NL1[1][i][Mc_AN][k][m][*l+4]*DS_NL1[1][0][Mj_AN][kl][n][*l+5]
                +tmp4*DS_NL1[1][i][Mc_AN][k][m][*l+5]*DS_NL1[1][0][Mj_AN][kl][n][*l+4] );
       }
       *sumx2i -= deri[1];
       *sumy2i -= deri[2];
       *sumz2i -= deri[3];
	   
    }

  }

  /****************************************************
      off-diagonal contribution on up-up and dn-dn
  ****************************************************/

  /* p */ 
  if (l2==2){
 
    tmpx =
       fugou*
       ( ene_p/3.0*DS_NL1[0][1][Mc_AN][k][m][*l  ]*DS_NL1[0][0][Mj_AN][kl][n][*l+1]
        -ene_p/3.0*DS_NL1[0][1][Mc_AN][k][m][*l+1]*DS_NL1[0][0][Mj_AN][kl][n][*l  ] ); 

    tmpy =
       fugou*
       ( ene_p/3.0*DS_NL1[0][2][Mc_AN][k][m][*l  ]*DS_NL1[0][0][Mj_AN][kl][n][*l+1]
        -ene_p/3.0*DS_NL1[0][2][Mc_AN][k][m][*l+1]*DS_NL1[0][0][Mj_AN][kl][n][*l  ] ); 

    tmpz =
       fugou*
       ( ene_p/3.0*DS_NL1[0][3][Mc_AN][k][m][*l  ]*DS_NL1[0][0][Mj_AN][kl][n][*l+1]
        -ene_p/3.0*DS_NL1[0][3][Mc_AN][k][m][*l+1]*DS_NL1[0][0][Mj_AN][kl][n][*l  ] ); 

    /* contribution of l+1/2 for up spin */ 
    *sumx0i += -tmpx;
    *sumy0i += -tmpy;
    *sumz0i += -tmpz;

    /* contribution of l+1/2 for down spin */ 
    *sumx1i += tmpx;
    *sumy1i += tmpy;
    *sumz1i += tmpz;

    tmpx =
       fugou*
       ( ene_m/3.0*DS_NL1[1][1][Mc_AN][k][m][*l  ]*DS_NL1[1][0][Mj_AN][kl][n][*l+1]
        -ene_m/3.0*DS_NL1[1][1][Mc_AN][k][m][*l+1]*DS_NL1[1][0][Mj_AN][kl][n][*l  ] ); 

    tmpy =
       fugou*
       ( ene_m/3.0*DS_NL1[1][2][Mc_AN][k][m][*l  ]*DS_NL1[1][0][Mj_AN][kl][n][*l+1]
        -ene_m/3.0*DS_NL1[1][2][Mc_AN][k][m][*l+1]*DS_NL1[1][0][Mj_AN][kl][n][*l  ] ); 

    tmpz =
       fugou*
       ( ene_m/3.0*DS_NL1[1][3][Mc_AN][k][m][*l  ]*DS_NL1[1][0][Mj_AN][kl][n][*l+1]
        -ene_m/3.0*DS_NL1[1][3][Mc_AN][k][m][*l+1]*DS_NL1[1][0][Mj_AN][kl][n][*l  ] ); 

    /* contribution of l-1/2 for up spin */
    *sumx0i += tmpx;
    *sumy0i += tmpy;
    *sumz0i += tmpz;

    /* contribution of l+1/2 for down spin */ 
    *sumx1i += -tmpx;
    *sumy1i += -tmpy;
    *sumz1i += -tmpz;
  }

  /* d */ 
  else if (l2==4){

    tmpx =
       fugou*
       (
       ene_p*2.0/5.0*DS_NL1[0][1][Mc_AN][k][m][*l+1]*DS_NL1[0][0][Mj_AN][kl][n][*l+2]
      -ene_p*2.0/5.0*DS_NL1[0][1][Mc_AN][k][m][*l+2]*DS_NL1[0][0][Mj_AN][kl][n][*l+1]
      +ene_p*1.0/5.0*DS_NL1[0][1][Mc_AN][k][m][*l+3]*DS_NL1[0][0][Mj_AN][kl][n][*l+4]
      -ene_p*1.0/5.0*DS_NL1[0][1][Mc_AN][k][m][*l+4]*DS_NL1[0][0][Mj_AN][kl][n][*l+3] ); 

    tmpy =
       fugou*
       (
       ene_p*2.0/5.0*DS_NL1[0][2][Mc_AN][k][m][*l+1]*DS_NL1[0][0][Mj_AN][kl][n][*l+2]
      -ene_p*2.0/5.0*DS_NL1[0][2][Mc_AN][k][m][*l+2]*DS_NL1[0][0][Mj_AN][kl][n][*l+1]
      +ene_p*1.0/5.0*DS_NL1[0][2][Mc_AN][k][m][*l+3]*DS_NL1[0][0][Mj_AN][kl][n][*l+4]
      -ene_p*1.0/5.0*DS_NL1[0][2][Mc_AN][k][m][*l+4]*DS_NL1[0][0][Mj_AN][kl][n][*l+3] ); 

    tmpz =
       fugou*
       (
       ene_p*2.0/5.0*DS_NL1[0][3][Mc_AN][k][m][*l+1]*DS_NL1[0][0][Mj_AN][kl][n][*l+2]
      -ene_p*2.0/5.0*DS_NL1[0][3][Mc_AN][k][m][*l+2]*DS_NL1[0][0][Mj_AN][kl][n][*l+1]
      +ene_p*1.0/5.0*DS_NL1[0][3][Mc_AN][k][m][*l+3]*DS_NL1[0][0][Mj_AN][kl][n][*l+4]
      -ene_p*1.0/5.0*DS_NL1[0][3][Mc_AN][k][m][*l+4]*DS_NL1[0][0][Mj_AN][kl][n][*l+3] ); 

    /* contribution of l+1/2 for up spin */ 
    *sumx0i += -tmpx;
    *sumy0i += -tmpy;
    *sumz0i += -tmpz;

    /* contribution of l+1/2 for down spin */ 
    *sumx1i += tmpx;
    *sumy1i += tmpy;
    *sumz1i += tmpz;

    tmpx =
       fugou*
       (
       ene_m*2.0/5.0*DS_NL1[1][1][Mc_AN][k][m][*l+1]*DS_NL1[1][0][Mj_AN][kl][n][*l+2]
      -ene_m*2.0/5.0*DS_NL1[1][1][Mc_AN][k][m][*l+2]*DS_NL1[1][0][Mj_AN][kl][n][*l+1]
      +ene_m*1.0/5.0*DS_NL1[1][1][Mc_AN][k][m][*l+3]*DS_NL1[1][0][Mj_AN][kl][n][*l+4]
      -ene_m*1.0/5.0*DS_NL1[1][1][Mc_AN][k][m][*l+4]*DS_NL1[1][0][Mj_AN][kl][n][*l+3] ); 

    tmpy = 
       fugou*
       (
       ene_m*2.0/5.0*DS_NL1[1][2][Mc_AN][k][m][*l+1]*DS_NL1[1][0][Mj_AN][kl][n][*l+2]
      -ene_m*2.0/5.0*DS_NL1[1][2][Mc_AN][k][m][*l+2]*DS_NL1[1][0][Mj_AN][kl][n][*l+1]
      +ene_m*1.0/5.0*DS_NL1[1][2][Mc_AN][k][m][*l+3]*DS_NL1[1][0][Mj_AN][kl][n][*l+4]
      -ene_m*1.0/5.0*DS_NL1[1][2][Mc_AN][k][m][*l+4]*DS_NL1[1][0][Mj_AN][kl][n][*l+3] ); 

    tmpz =
       fugou*
       (
       ene_m*2.0/5.0*DS_NL1[1][3][Mc_AN][k][m][*l+1]*DS_NL1[1][0][Mj_AN][kl][n][*l+2]
      -ene_m*2.0/5.0*DS_NL1[1][3][Mc_AN][k][m][*l+2]*DS_NL1[1][0][Mj_AN][kl][n][*l+1]
      +ene_m*1.0/5.0*DS_NL1[1][3][Mc_AN][k][m][*l+3]*DS_NL1[1][0][Mj_AN][kl][n][*l+4]
      -ene_m*1.0/5.0*DS_NL1[1][3][Mc_AN][k][m][*l+4]*DS_NL1[1][0][Mj_AN][kl][n][*l+3] ); 

    /* contribution of l-1/2 for up spin */ 
    *sumx0i += tmpx;
    *sumy0i += tmpy;
    *sumz0i += tmpz;

    /* contribution of l-1/2 for down spin */ 
    *sumx1i += -tmpx;
    *sumy1i += -tmpy;
    *sumz1i += -tmpz;

  }
  
  /* f */ 
  else if (l2==6){

    tmpx =
       fugou*
       (
       ene_p*1.0/7.0*DS_NL1[0][1][Mc_AN][k][m][*l+1]*DS_NL1[0][0][Mj_AN][kl][n][*l+2]
      -ene_p*1.0/7.0*DS_NL1[0][1][Mc_AN][k][m][*l+2]*DS_NL1[0][0][Mj_AN][kl][n][*l+1]
      +ene_p*2.0/7.0*DS_NL1[0][1][Mc_AN][k][m][*l+3]*DS_NL1[0][0][Mj_AN][kl][n][*l+4]
      -ene_p*2.0/7.0*DS_NL1[0][1][Mc_AN][k][m][*l+4]*DS_NL1[0][0][Mj_AN][kl][n][*l+3]
      +ene_p*3.0/7.0*DS_NL1[0][1][Mc_AN][k][m][*l+5]*DS_NL1[0][0][Mj_AN][kl][n][*l+6]
      -ene_p*3.0/7.0*DS_NL1[0][1][Mc_AN][k][m][*l+6]*DS_NL1[0][0][Mj_AN][kl][n][*l+5] );

    tmpy = 
       fugou*
       (
       ene_p*1.0/7.0*DS_NL1[0][2][Mc_AN][k][m][*l+1]*DS_NL1[0][0][Mj_AN][kl][n][*l+2]
      -ene_p*1.0/7.0*DS_NL1[0][2][Mc_AN][k][m][*l+2]*DS_NL1[0][0][Mj_AN][kl][n][*l+1]
      +ene_p*2.0/7.0*DS_NL1[0][2][Mc_AN][k][m][*l+3]*DS_NL1[0][0][Mj_AN][kl][n][*l+4]
      -ene_p*2.0/7.0*DS_NL1[0][2][Mc_AN][k][m][*l+4]*DS_NL1[0][0][Mj_AN][kl][n][*l+3]
      +ene_p*3.0/7.0*DS_NL1[0][2][Mc_AN][k][m][*l+5]*DS_NL1[0][0][Mj_AN][kl][n][*l+6]
      -ene_p*3.0/7.0*DS_NL1[0][2][Mc_AN][k][m][*l+6]*DS_NL1[0][0][Mj_AN][kl][n][*l+5] );

    tmpz =
       fugou*
       (
       ene_p*1.0/7.0*DS_NL1[0][3][Mc_AN][k][m][*l+1]*DS_NL1[0][0][Mj_AN][kl][n][*l+2]
      -ene_p*1.0/7.0*DS_NL1[0][3][Mc_AN][k][m][*l+2]*DS_NL1[0][0][Mj_AN][kl][n][*l+1]
      +ene_p*2.0/7.0*DS_NL1[0][3][Mc_AN][k][m][*l+3]*DS_NL1[0][0][Mj_AN][kl][n][*l+4]
      -ene_p*2.0/7.0*DS_NL1[0][3][Mc_AN][k][m][*l+4]*DS_NL1[0][0][Mj_AN][kl][n][*l+3]
      +ene_p*3.0/7.0*DS_NL1[0][3][Mc_AN][k][m][*l+5]*DS_NL1[0][0][Mj_AN][kl][n][*l+6]
      -ene_p*3.0/7.0*DS_NL1[0][3][Mc_AN][k][m][*l+6]*DS_NL1[0][0][Mj_AN][kl][n][*l+5] );

    /* contribution of l+1/2 for up spin */ 
    *sumx0i += -tmpx;
    *sumy0i += -tmpy;
    *sumz0i += -tmpz;

    /* contribution of l+1/2 for down spin */ 
    *sumx1i += tmpx;
    *sumy1i += tmpy;
    *sumz1i += tmpz;

    tmpx =
       fugou*
       (
       ene_m*1.0/7.0*DS_NL1[1][1][Mc_AN][k][m][*l+1]*DS_NL1[1][0][Mj_AN][kl][n][*l+2]
      -ene_m*1.0/7.0*DS_NL1[1][1][Mc_AN][k][m][*l+2]*DS_NL1[1][0][Mj_AN][kl][n][*l+1]
      +ene_m*2.0/7.0*DS_NL1[1][1][Mc_AN][k][m][*l+3]*DS_NL1[1][0][Mj_AN][kl][n][*l+4]
      -ene_m*2.0/7.0*DS_NL1[1][1][Mc_AN][k][m][*l+4]*DS_NL1[1][0][Mj_AN][kl][n][*l+3]
      +ene_m*3.0/7.0*DS_NL1[1][1][Mc_AN][k][m][*l+5]*DS_NL1[1][0][Mj_AN][kl][n][*l+6]
      -ene_m*3.0/7.0*DS_NL1[1][1][Mc_AN][k][m][*l+6]*DS_NL1[1][0][Mj_AN][kl][n][*l+5] );

    tmpy =
       fugou*
       (
       ene_m*1.0/7.0*DS_NL1[1][2][Mc_AN][k][m][*l+1]*DS_NL1[1][0][Mj_AN][kl][n][*l+2]
      -ene_m*1.0/7.0*DS_NL1[1][2][Mc_AN][k][m][*l+2]*DS_NL1[1][0][Mj_AN][kl][n][*l+1]
      +ene_m*2.0/7.0*DS_NL1[1][2][Mc_AN][k][m][*l+3]*DS_NL1[1][0][Mj_AN][kl][n][*l+4]
      -ene_m*2.0/7.0*DS_NL1[1][2][Mc_AN][k][m][*l+4]*DS_NL1[1][0][Mj_AN][kl][n][*l+3]
      +ene_m*3.0/7.0*DS_NL1[1][2][Mc_AN][k][m][*l+5]*DS_NL1[1][0][Mj_AN][kl][n][*l+6]
      -ene_m*3.0/7.0*DS_NL1[1][2][Mc_AN][k][m][*l+6]*DS_NL1[1][0][Mj_AN][kl][n][*l+5] );

    tmpz =
       fugou*
       (
       ene_m*1.0/7.0*DS_NL1[1][3][Mc_AN][k][m][*l+1]*DS_NL1[1][0][Mj_AN][kl][n][*l+2]
      -ene_m*1.0/7.0*DS_NL1[1][3][Mc_AN][k][m][*l+2]*DS_NL1[1][0][Mj_AN][kl][n][*l+1]
      +ene_m*2.0/7.0*DS_NL1[1][3][Mc_AN][k][m][*l+3]*DS_NL1[1][0][Mj_AN][kl][n][*l+4]
      -ene_m*2.0/7.0*DS_NL1[1][3][Mc_AN][k][m][*l+4]*DS_NL1[1][0][Mj_AN][kl][n][*l+3]
      +ene_m*3.0/7.0*DS_NL1[1][3][Mc_AN][k][m][*l+5]*DS_NL1[1][0][Mj_AN][kl][n][*l+6]
      -ene_m*3.0/7.0*DS_NL1[1][3][Mc_AN][k][m][*l+6]*DS_NL1[1][0][Mj_AN][kl][n][*l+5] );

    /* contribution of l-1/2 for up spin */ 
    *sumx0i += tmpx;
    *sumy0i += tmpy;
    *sumz0i += tmpz;

    /* contribution of l-1/2 for down spin */ 
    *sumx1i += -tmpx;
    *sumy1i += -tmpy;
    *sumz1i += -tmpz;
  }

  /****************************************************
      diagonal contribution on up-up and dn-dn
  ****************************************************/

  for (l3=0; l3<=l2; l3++){

    /* VNL for j=l+1/2 */
              
    tmpx = PFp*ene_p*DS_NL1[0][1][Mc_AN][k][m][*l]*DS_NL1[0][0][Mj_AN][kl][n][*l];
    tmpy = PFp*ene_p*DS_NL1[0][2][Mc_AN][k][m][*l]*DS_NL1[0][0][Mj_AN][kl][n][*l];
    tmpz = PFp*ene_p*DS_NL1[0][3][Mc_AN][k][m][*l]*DS_NL1[0][0][Mj_AN][kl][n][*l];

    *sumx0r += tmpx;
    *sumy0r += tmpy;
    *sumz0r += tmpz;

    *sumx1r += tmpx;
    *sumy1r += tmpy;
    *sumz1r += tmpz;

    /* VNL for j=l-1/2 */
              
    tmpx = PFm*ene_m*DS_NL1[1][1][Mc_AN][k][m][*l]*DS_NL1[1][0][Mj_AN][kl][n][*l];
    tmpy = PFm*ene_m*DS_NL1[1][2][Mc_AN][k][m][*l]*DS_NL1[1][0][Mj_AN][kl][n][*l];
    tmpz = PFm*ene_m*DS_NL1[1][3][Mc_AN][k][m][*l]*DS_NL1[1][0][Mj_AN][kl][n][*l];

    *sumx0r += tmpx;
    *sumy0r += tmpy;
    *sumz0r += tmpz;

    *sumx1r += tmpx;
    *sumy1r += tmpy;
    *sumz1r += tmpz;

    *l = *l + 1;
  }
}


  

void MPI_OLP(OpenMXSolverInvocation *scratch,double *****OLP1)
{
  int i,j,h_AN,Gh_AN,Hwan,n;
  int tno1,tno2,Mc_AN,Gc_AN,Cwan;
  int num,k,size1,size2;
  double *tmp_array;
  double *tmp_array2;
  int *Snd_S_Size,*Rcv_S_Size;
  int numprocs,myid,ID,IDS,IDR,tag=999;

  MPI_Status stat;
  MPI_Request request;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /****************************************************
    allocation of arrays:
  ****************************************************/

  OpenMX_Stress_OverlapCountsPrepare(scratch,numprocs,&Snd_S_Size,&Rcv_S_Size);

  /******************************************************************
   MPI

   OLP[1], OLP[2], and OLP[3]

   note:

   OLP is used in DC and GDC method, where overlap integrals
   of Matomnum+MatomnumF+MatomnumS+1 are stored.
   However, overlap integrals of Matomnum+MatomnumF+1 are
   stored in Force.c. So, F_TopMAN should be used to refer
   overlap integrals in Force.c, while S_TopMAN should be
   used in DC and GDC routines.

   Although OLP is used in Eff_Hub_Pot.c, the usage is 
   consistent with that of DC and GDC routines by the following 
   reason:

   DC or GDC:      OLP + Spe_Total_NO   if no orbital optimization 
                CntOLP + Spe_Total_CNO  if orbital optimization

   Eff_Hub_Pot:    OLP + Spe_Total_NO   always since the U-potential
                                        affects to primitive orbital 

   If no orbital optimization, both the usages are consistent.
   If orbital optimization, CntOLP and OLP are used in DC(GDC) and 
   Eff_Hub_Pot.c, respectively. Therefore, there is no conflict. 
  *******************************************************************/

  /***********************************
             set data size
  ************************************/

  for (ID=0; ID<numprocs; ID++){

    IDS = (myid + ID) % numprocs;
    IDR = (myid - ID + numprocs) % numprocs;

    if (ID!=0){
      tag = 999;

      /* find data size to send block data */
      if (F_Snd_Num[IDS]!=0){
	size1 = 0;
	for (n=0; n<F_Snd_Num[IDS]; n++){
	  Mc_AN = Snd_MAN[IDS][n];
	  Gc_AN = Snd_GAN[IDS][n];
	  Cwan = WhatSpecies[Gc_AN]; 
	  tno1 = Spe_Total_NO[Cwan];
	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	    Gh_AN = natn[Gc_AN][h_AN];        
	    Hwan = WhatSpecies[Gh_AN];
	    tno2 = Spe_Total_NO[Hwan];
            size1 += 4*tno1*tno2;
	  }
	}

	Snd_S_Size[IDS] = size1;
	MPI_Isend(&size1, 1, MPI_INT, IDS, tag, mpi_comm_level1, &request);
      }
      else{
	Snd_S_Size[IDS] = 0;
      }

      /* receiving of size of data */
 
      if (F_Rcv_Num[IDR]!=0){
	MPI_Recv(&size2, 1, MPI_INT, IDR, tag, mpi_comm_level1, &stat);
	Rcv_S_Size[IDR] = size2;
      }
      else{
	Rcv_S_Size[IDR] = 0;
      }

      if (F_Snd_Num[IDS]!=0) MPI_Wait(&request,&stat);
    }
    else{
      Snd_S_Size[IDS] = 0;
      Rcv_S_Size[IDR] = 0;
    }
  }

  /***********************************
               data transfer
  ************************************/

  tag = 999;
  for (ID=0; ID<numprocs; ID++){

    IDS = (myid + ID) % numprocs;
    IDR = (myid - ID + numprocs) % numprocs;

    if (ID!=0){

      /*****************************
                sending of data 
      *****************************/

      if (F_Snd_Num[IDS]!=0){

	size1 = Snd_S_Size[IDS];

	/* allocation of array */

	tmp_array=OpenMX_Stress_HaloBuffer(scratch,"stress:overlap:send",size1);

	/* multidimentional array to vector array */

	num = 0;

        for (k=0; k<=3; k++){
  	  for (n=0; n<F_Snd_Num[IDS]; n++){
	    Mc_AN = Snd_MAN[IDS][n];
	    Gc_AN = Snd_GAN[IDS][n];
	    Cwan = WhatSpecies[Gc_AN]; 
	    tno1 = Spe_Total_NO[Cwan];
	    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	      Gh_AN = natn[Gc_AN][h_AN];        
	      Hwan = WhatSpecies[Gh_AN];
	      tno2 = Spe_Total_NO[Hwan];
	      for (i=0; i<tno1; i++){
		for (j=0; j<tno2; j++){
		  tmp_array[num] = OLP1[k][Mc_AN][h_AN][i][j];
		  num++;
		} 
	      } 
	    }
	  }
	}

	MPI_Isend(&tmp_array[0], size1, MPI_DOUBLE, IDS, tag, mpi_comm_level1, &request);
      }

      /*****************************
           receiving of block data
      *****************************/

      if (F_Rcv_Num[IDR]!=0){
          
	size2 = Rcv_S_Size[IDR];
        
	/* allocation of array */
	tmp_array2=OpenMX_Stress_HaloBuffer(scratch,"stress:overlap:receive",size2);
         
	MPI_Recv(&tmp_array2[0], size2, MPI_DOUBLE, IDR, tag, mpi_comm_level1, &stat);

	num = 0;

        for (k=0; k<=3; k++){
	  Mc_AN = F_TopMAN[IDR] - 1; /* F_TopMAN should be used. */
  	  for (n=0; n<F_Rcv_Num[IDR]; n++){
	    Mc_AN++;
	    Gc_AN = Rcv_GAN[IDR][n];
	    Cwan = WhatSpecies[Gc_AN]; 
	    tno1 = Spe_Total_NO[Cwan];

	    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	      Gh_AN = natn[Gc_AN][h_AN];        
	      Hwan = WhatSpecies[Gh_AN];
	      tno2 = Spe_Total_NO[Hwan];
	      for (i=0; i<tno1; i++){
		for (j=0; j<tno2; j++){
		  OLP1[k][Mc_AN][h_AN][i][j] = tmp_array2[num];
		  num++;
		}
	      }
	    }
	  }        
	}

	/* freeing of array */

      }

      if (F_Snd_Num[IDS]!=0){
	MPI_Wait(&request,&stat);
	 /* freeing of array */
      }
    }
  }
 
  /****************************************************
    freeing of arrays:
  ****************************************************/



}




void EH0_TwoCenter(const OpenMXNeutralGridTables *tables,const OpenMXAtomicViews *atomic, int Gc_AN, int h_AN, double VH0ij[4])
{
  const int Gh_AN=natn[Gc_AN][h_AN],Rn=ncn[Gc_AN][h_AN];
  const int wan1=WhatSpecies[Gc_AN],wan2=WhatSpecies[Gh_AN];
  const int ban=atomic->species.species_index[wan1];
  const OpenMXNeutralGridView grid={TGN_EH0[ban],dv_EH0[ban],
      openmx_core_neutral_grid_row(tables,0,ban),
      openmx_core_neutral_grid_row(tables,1,ban),
      openmx_core_neutral_grid_row(tables,2,ban),
      openmx_core_neutral_grid_row(tables,3,ban),
      openmx_core_neutral_grid_row(tables,4,ban)};
  OpenMXNeutralPotentialView potential;
  OpenMX_Neutral_PotentialBorrow(atomic,wan2,&potential);
  double displacement[3]={0.0,0.0,0.0};
  if(h_AN!=0){
    displacement[0]=Gxyz[Gc_AN][1]-(Gxyz[Gh_AN][1]+atv[Rn][1]);
    displacement[1]=Gxyz[Gc_AN][2]-(Gxyz[Gh_AN][2]+atv[Rn][2]);
    displacement[2]=Gxyz[Gc_AN][3]-(Gxyz[Gh_AN][3]+atv[Rn][3]);
  }
  OpenMX_Neutral_Pair(&grid,&potential,h_AN,Dis[Gc_AN][h_AN],displacement,VH0ij);
}



void EH0_TwoCenter_at_Cutoff(OpenMXSolverInvocation *scratch,const OpenMXNeutralGridTables *tables,const OpenMXAtomicViews *atomic, int wan1, int wan2, double VH0ij[4])
{ 
  int ban;
  double dv,rcut;
  double sum,sumx,sumy,sumz;
  /* for OpenMP */
  int OMPID,Nthrds,Nthrds0,Nprocs,Nloop;
  OpenMXSolverRealTensor cutoff_sums;

  ban = atomic->species.species_index[wan1];
  dv  = dv_EH0[ban];
  const OpenMXNeutralGridView grid={TGN_EH0[ban],dv,
      openmx_core_neutral_grid_row(tables,0,ban),
      openmx_core_neutral_grid_row(tables,1,ban),
      openmx_core_neutral_grid_row(tables,2,ban),
      openmx_core_neutral_grid_row(tables,3,ban),
      openmx_core_neutral_grid_row(tables,4,ban)};
  OpenMXNeutralPotentialView potential;
  OpenMX_Neutral_PotentialBorrow(atomic,wan2,&potential);


  rcut = atomic->species.cutoff[wan1] + atomic->species.cutoff[wan2];

  /* get Nthrds0 */  
  /* Dynamic teams may grow after a separate probe region. */
  Nthrds0 = omp_get_max_threads();

  /* allocation of array */
  OpenMX_Stress_CutoffPrepare(scratch,Nthrds0,&cutoff_sums);

  for (Nloop=0; Nloop<Nthrds0; Nloop++){
    openmx_solver_real_matrix_row(&cutoff_sums,Nloop)[0] = 0.0;
  }

#pragma omp parallel shared(grid,potential,cutoff_sums,rcut) private(OMPID,Nthrds,Nprocs)
  {
    /* get info. on OpenMP */

    OMPID = omp_get_thread_num();
    Nthrds = omp_get_num_threads();
    Nprocs = omp_get_num_procs();

    OpenMX_Neutral_CutoffAccumulate(&grid,&potential,rcut,
        OMPID*grid.count/Nthrds,(OMPID+1)*grid.count/Nthrds,openmx_solver_real_matrix_row(&cutoff_sums,OMPID));

  } /* #pragma omp parallel */

  sum  = 0.0;
  for (Nloop=0; Nloop<Nthrds0; Nloop++){
    sum += openmx_solver_real_matrix_row(&cutoff_sums,Nloop)[0];
  }

  sum  = sum*dv;
  sumx = 0.0;
  sumy = 0.0;
  sumz = 0.0;

  VH0ij[0] = sum;
  VH0ij[1] = sumx;
  VH0ij[2] = sumy;
  VH0ij[3] = sumz;

  /* freeing of array */

}







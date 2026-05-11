/**********************************************************************
  Set_Hamiltonian.c:

     Set_Hamiltonian.c is a subroutine to make Hamiltonian matrix
     within LDA or GGA.

  Log of Set_Hamiltonian.c:

     24/April/2002  Released by T. Ozaki
     17/April/2013  Modified by A.M. Ito

***********************************************************************/

#include "mpi.h"
#include "openmx_common.h"
#include "lapack_prototypes.h"
#include <limits.h>
#include <math.h>
#include <omp.h>
#include <openacc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define  measure_time   0
#define  flag_Calc_MatrixElements_dVH_Vxc_VNA   0

/* GPU device assignment helper (added by H.Kawai, ported from 3.9.9 GPU)
 * Selects a CUDA device per node-local rank using MPI_COMM_TYPE_SHARED, and
 * returns 1 if memory budget required_bytes fits across the shared group. */
static int Set_Hamiltonian_DeviceMemoryOK(size_t required_bytes, const char *where, int myid)
{
    MPI_Comm node_comm, device_comm;
    int local_rank, cuda_device_count, acc_device_count, device_count;
    int cuda_device;
    size_t free_bytes = 0, total_bytes = 0;
    cudaError_t cuda_err;
    unsigned long long local_required, group_required, local_free, group_free;
    int cuda_ok = 0;

    MPI_Comm_split_type(mpi_comm_level1, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node_comm);
    MPI_Comm_rank(node_comm, &local_rank);

    cuda_err = cudaGetDeviceCount(&cuda_device_count);
    if (cuda_err == cudaSuccess && cuda_device_count > 0) {
        acc_device_count = acc_get_num_devices(acc_device_nvidia);
        if (acc_device_count > 0) {
            device_count = (cuda_device_count < acc_device_count) ? cuda_device_count : acc_device_count;
            cuda_device = local_rank % device_count;
            cuda_err = cudaSetDevice(cuda_device);
            if (cuda_err == cudaSuccess) {
                acc_set_device_num(cuda_device, acc_device_nvidia);
                cuda_err = cudaMemGetInfo(&free_bytes, &total_bytes);
                if (cuda_err == cudaSuccess) cuda_ok = 1;
            }
        }
    }

    MPI_Comm_split(node_comm, cuda_ok ? cuda_device : MPI_UNDEFINED, 0, &device_comm);
    MPI_Comm_free(&node_comm);
    if (!cuda_ok) return 0;

    local_required = (unsigned long long)required_bytes;
    local_free     = (unsigned long long)free_bytes;
    MPI_Allreduce(&local_required, &group_required, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, device_comm);
    MPI_Allreduce(&local_free,     &group_free,     1, MPI_UNSIGNED_LONG_LONG, MPI_MIN, device_comm);
    MPI_Comm_free(&device_comm);

    return (group_free >= group_required) ? 1 : 0;
}

void Calc_MatrixElements_dVH_Vxc_VNA_by_AI(int Cnt_kind);
void Calc_MatrixElements_dVH_Vxc_VNA_by_TO(int Cnt_kind);

double Set_Hamiltonian(char *mode,
                       int MD_iter,
                       int SCF_iter,
                       int SCF_iter0,
                       int TRAN_Poisson_flag2,
                       int SucceedReadingDMfile,
                       int Cnt_kind,
                       double *****H0,
                       double *****HNL,
                       double *****CDM,
		       double *****H)
{
  /***************************************************************
      Cnt_kind
        0:  Uncontracted Hamiltonian    
        1:  Contracted Hamiltonian    
  ***************************************************************/

  int Mc_AN,Gc_AN,Mh_AN,h_AN,Gh_AN;
  int i,j,k,Cwan,Hwan,NO0,NO1,spin,N,NOLG;
  int Nc,Ncs,GNc,GRc,Nog,Nh,MN,XC_P_switch;
  double TStime,TEtime;
  int numprocs,myid;
  double time0,time1,time2,mflops;
  long Num_C0,Num_C1;

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);
  MPI_Barrier(mpi_comm_level1);
  dtime(&TStime);

  if (MYID_MPI_COMM_WORLD==Host_ID && myid==Host_ID 
      && strcasecmp(mode,"stdout")==0 && 0<level_stdout ){
    printf("<Set_Hamiltonian>  Hamiltonian matrix for VNA+dVH+Vxc...\n");fflush(stdout);
  }

  /*****************************************************
                adding H0+HNL+(HCH) to H 
  *****************************************************/

  if(measure_time) dtime(&time1);

  /* spin non-collinear */

  if (SpinP_switch==3){
    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];    
      Cwan = WhatSpecies[Gc_AN];
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
        Gh_AN = natn[Gc_AN][h_AN];

        Hwan = WhatSpecies[Gh_AN];
        for (i=0; i<Spe_Total_NO[Cwan]; i++){
          for (j=0; j<Spe_Total_NO[Hwan]; j++){

            if (ProExpn_VNA==0){
              H[0][Mc_AN][h_AN][i][j] = F_Kin_flag*H0[0][Mc_AN][h_AN][i][j]
 	                              + F_NL_flag*HNL[0][Mc_AN][h_AN][i][j];
              H[1][Mc_AN][h_AN][i][j] = F_Kin_flag*H0[0][Mc_AN][h_AN][i][j]
		                      + F_NL_flag*HNL[1][Mc_AN][h_AN][i][j];
              H[2][Mc_AN][h_AN][i][j] = F_NL_flag*HNL[2][Mc_AN][h_AN][i][j];
              H[3][Mc_AN][h_AN][i][j] = 0.0;
	    }
            else{
              H[0][Mc_AN][h_AN][i][j] = F_Kin_flag*H0[0][Mc_AN][h_AN][i][j]
	 	                      + F_VNA_flag*HVNA[Mc_AN][h_AN][i][j]
		                      + F_NL_flag*HNL[0][Mc_AN][h_AN][i][j];
              H[1][Mc_AN][h_AN][i][j] = F_Kin_flag*H0[0][Mc_AN][h_AN][i][j]
		                      + F_VNA_flag*HVNA[Mc_AN][h_AN][i][j]
		                      + F_NL_flag*HNL[1][Mc_AN][h_AN][i][j];
              H[2][Mc_AN][h_AN][i][j] = F_NL_flag*HNL[2][Mc_AN][h_AN][i][j];
              H[3][Mc_AN][h_AN][i][j] = 0.0;
            }

            /* Effective Hubbard Hamiltonain --- added by MJ */

	    if (  (Hub_U_switch==1 || 1<=Constraint_NCS_switch || Zeeman_NCS_switch==1 || Zeeman_NCO_switch==1) 
                 && F_U_flag==1 && 2<=SCF_iter
              ){

	      H[0][Mc_AN][h_AN][i][j] += H_Hub[0][Mc_AN][h_AN][i][j];
	      H[1][Mc_AN][h_AN][i][j] += H_Hub[1][Mc_AN][h_AN][i][j];
	      H[2][Mc_AN][h_AN][i][j] += H_Hub[2][Mc_AN][h_AN][i][j];
	    }

	    /* core hole Hamiltonian */

	    if (core_hole_state_flag==1){
	      H[0][Mc_AN][h_AN][i][j] += HCH[0][Mc_AN][h_AN][i][j];
	      H[1][Mc_AN][h_AN][i][j] += HCH[1][Mc_AN][h_AN][i][j];
	      H[2][Mc_AN][h_AN][i][j] += HCH[2][Mc_AN][h_AN][i][j];
	    }

	    if (xmcd_calc==1){
	      H[0][Mc_AN][h_AN][i][j] += H_XMCD[0][Mc_AN][h_AN][i][j];
	      H[1][Mc_AN][h_AN][i][j] += H_XMCD[1][Mc_AN][h_AN][i][j];
	      H[2][Mc_AN][h_AN][i][j] += H_XMCD[2][Mc_AN][h_AN][i][j];
	      H[3][Mc_AN][h_AN][i][j] += H_XMCD[3][Mc_AN][h_AN][i][j];
	    }

          }
        }
      }
    }
  }

  /* spin collinear */

  else{

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];    
      Cwan = WhatSpecies[Gc_AN];
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
        Gh_AN = natn[Gc_AN][h_AN];
        Hwan = WhatSpecies[Gh_AN];
        for (i=0; i<Spe_Total_NO[Cwan]; i++){
          for (j=0; j<Spe_Total_NO[Hwan]; j++){
            for (spin=0; spin<=SpinP_switch; spin++){

              if (ProExpn_VNA==0){
                H[spin][Mc_AN][h_AN][i][j] = F_Kin_flag*H0[0][Mc_AN][h_AN][i][j]
		                           + F_NL_flag*HNL[spin][Mc_AN][h_AN][i][j];
	      }
              else{
                H[spin][Mc_AN][h_AN][i][j] = F_Kin_flag*H0[0][Mc_AN][h_AN][i][j]
		                           + F_VNA_flag*HVNA[Mc_AN][h_AN][i][j]
		                           + F_NL_flag*HNL[spin][Mc_AN][h_AN][i][j];
              }

	      /* Effective Hubbard Hamiltonian --- added by MJ */
	      if ( (Hub_U_switch==1 || 1<=Constraint_NCS_switch) && F_U_flag==1 && 2<=SCF_iter ){
		H[spin][Mc_AN][h_AN][i][j] += H_Hub[spin][Mc_AN][h_AN][i][j];
	      }

	      /* core hole Hamiltonian */
	      if (core_hole_state_flag==1){
		H[spin][Mc_AN][h_AN][i][j] += HCH[spin][Mc_AN][h_AN][i][j];
	      }
            }
          }
        }
      }
    }

  }

  if(measure_time){ 
    dtime(&time2);
    printf("myid=%4d Time1=%18.10f\n",myid,time2-time1);fflush(stdout);
  }

  if (Cnt_kind==1) {
    Contract_Hamiltonian( H, CntH, OLP, CntOLP );
    if (SO_switch==1) Contract_iHNL(iHNL,iCntHNL);
  }

  /*****************************************************
   calculation of Vpot;
  *****************************************************/

  if(myid==0 && measure_time)  dtime(&time1);

  XC_P_switch = 1;
  Set_Vpot(MD_iter,SCF_iter,SCF_iter0,TRAN_Poisson_flag2,XC_P_switch);

  if(measure_time){ 
    dtime(&time2);
    printf("myid=%4d Time2=%18.10f\n",myid,time2-time1);fflush(stdout);
  }

  /*****************************************************
   calculation of matrix elements for dVH + Vxc (+ VNA)
  *****************************************************/

  if      (flag_Calc_MatrixElements_dVH_Vxc_VNA==0){
    Calc_MatrixElements_dVH_Vxc_VNA_by_AI(Cnt_kind);
  }
  else if (flag_Calc_MatrixElements_dVH_Vxc_VNA==1){
    Calc_MatrixElements_dVH_Vxc_VNA_by_TO(Cnt_kind);
  }

  /* for time */
  if (measure_time) dtime(&time1);
  MPI_Barrier(mpi_comm_level1);
  if(measure_time) {
    dtime(&time2);
    printf("myid=%4d Time Barrier=%18.10f\n",myid,time2-time1);
    fflush(stdout);
  }

  dtime(&TEtime);
  time0 = TEtime - TStime;
  return time0;
}



void Calc_MatrixElements_dVH_Vxc_VNA_by_TO(int Cnt_kind)
{
  int i,j,spin,NO0,NO1,mm,Nog,Nc,Nh;
  int Nh_0,Nh_1,Nh_2,Nh_3,Nc_0,Nc_1,Nc_2,Nc_3;
  int Nprocs,Nthrds,Cwan,Ncs,Rnh,Hwan;
  int Mc_AN,Gc_AN,Mh_AN,h_AN,Gh_AN;
  int Nh0,Nh1,Nh2,Nh3;
  int Nc0,Nc1,Nc2,Nc3;
  int MN0,MN1,MN2,MN3;
  int Nloop,OneD_Nloop;
  int *OneD2spin,*OneD2Mc_AN,*OneD2h_AN;
  int numprocs,myid,OMPID;
  double tmp0,tmp0_0,tmp0_1,tmp0_2,tmp0_3;
  double time0,time1,time2,mflops;
  double **tmp_ChiV0_0,**tmp_ChiV0_1,**tmp_ChiV0_2,**tmp_ChiV0_3;
  double *tmp_Orbs_Grid_0,*tmp_Orbs_Grid_1,*tmp_Orbs_Grid_2,*tmp_Orbs_Grid_3;
  double *Chi0,***ChiV0,***tmp_H;

  if(measure_time) dtime(&time1);

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  dtime(&time1);

  /* one-dimensionalize the Mc_AN and h_AN loops */

  OneD_Nloop = 0;
  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];    
    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
      OneD_Nloop++;
    }
  }  

  OneD2Mc_AN = (int*)malloc(sizeof(int)*(OneD_Nloop+1));
  OneD2h_AN = (int*)malloc(sizeof(int)*(OneD_Nloop+1));

  OneD_Nloop = 0;
  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];    
    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
      OneD2Mc_AN[OneD_Nloop] = Mc_AN; 
      OneD2h_AN[OneD_Nloop] = h_AN; 
      OneD_Nloop++;
    }
  }

  /* OpenMP */

/*
#pragma omp parallel shared(gtv,Gxyz,atv,GridListAtom,CellListAtom,OneD2h_AN,OneD2Mc_AN,OneD_Nloop,GridN_Atom,H,CntH,List_YOUSO,GridVol,Orbs_Grid,Vpot_Grid,MGridListAtom,GListTAtoms2,GListTAtoms1,Matomnum,NumOLG,SpinP_switch,Spe_Total_CNO,Spe_Total_NO,Cnt_kind,WhatSpecies,ncn,natn,F_G2M,FNAN)  private(OMPID,Nthrds,Nprocs,h_AN,Gh_AN,Mh_AN,Rnh,Hwan,NO1,spin,i,j,tmp_H,Nog,Nc_0,Nc_1,Nc_2,Nc_3,Nh_0,Nh_1,Nh_2,Nh_3,tmp_ChiV0_0,tmp_ChiV0_1,tmp_ChiV0_2,tmp_ChiV0_3,tmp_Orbs_Grid_0,tmp_Orbs_Grid_1,tmp_Orbs_Grid_2,tmp_Orbs_Grid_3,mm,Nc,Nh,tmp0,tmp0_0,tmp0_1,tmp0_2,tmp0_3,Mc_AN,Gc_AN,Cwan,NO0,Ncs,ChiV0,Chi0,Nloop)
  {
*/
    /* allocation of arrays */ 

    Chi0 = (double*)malloc(sizeof(double)*List_YOUSO[7]);

    ChiV0 = (double***)malloc(sizeof(double**)*4);
    for (i=0; i<4; i++){
      ChiV0[i] = (double**)malloc(sizeof(double*)*List_YOUSO[7]);
      for (j=0; j<List_YOUSO[7]; j++){
	ChiV0[i][j] = (double*)malloc(sizeof(double)*List_YOUSO[11]);
      }
    }

    tmp_H = (double***)malloc(sizeof(double**)*4);
    for (i=0; i<4; i++){
      tmp_H[i] = (double**)malloc(sizeof(double*)*List_YOUSO[7]);
      for (j=0; j<List_YOUSO[7]; j++){
	tmp_H[i][j] = (double*)malloc(sizeof(double)*List_YOUSO[7]);
      }
    }

    tmp_ChiV0_0 = (double**)malloc(sizeof(double*)*4);
    for (i=0; i<4; i++){
      tmp_ChiV0_0[i] = (double*)malloc(sizeof(double)*List_YOUSO[7]);
    }

    tmp_ChiV0_1 = (double**)malloc(sizeof(double*)*4);
    for (i=0; i<4; i++){
      tmp_ChiV0_1[i] = (double*)malloc(sizeof(double)*List_YOUSO[7]);
    }
    
    tmp_ChiV0_2 = (double**)malloc(sizeof(double*)*4);
    for (i=0; i<4; i++){
      tmp_ChiV0_2[i] = (double*)malloc(sizeof(double)*List_YOUSO[7]);
    }
    
    tmp_ChiV0_3 = (double**)malloc(sizeof(double*)*4);
    for (i=0; i<4; i++){
      tmp_ChiV0_3[i] = (double*)malloc(sizeof(double)*List_YOUSO[7]);
    }
    
    tmp_Orbs_Grid_0 = (double*)malloc(sizeof(double)*List_YOUSO[7]);
    tmp_Orbs_Grid_1 = (double*)malloc(sizeof(double)*List_YOUSO[7]);
    tmp_Orbs_Grid_2 = (double*)malloc(sizeof(double)*List_YOUSO[7]);
    tmp_Orbs_Grid_3 = (double*)malloc(sizeof(double)*List_YOUSO[7]);
    
    /* get info. on OpenMP */ 

    /*
    OMPID = omp_get_thread_num();
    Nthrds = omp_get_num_threads();
    Nprocs = omp_get_num_procs();
    */
    
    /* one-dimensionalized loop */

    for (Nloop=0; Nloop<OneD_Nloop; Nloop++){

//    for (Nloop=OMPID*OneD_Nloop/Nthrds; Nloop<(OMPID+1)*OneD_Nloop/Nthrds; Nloop++){

      /* get Mc_AN and h_AN */

      Mc_AN = OneD2Mc_AN[Nloop];
      h_AN  = OneD2h_AN[Nloop];

      /* set data on Mc_AN */

      Gc_AN = M2G[Mc_AN];    
      Cwan = WhatSpecies[Gc_AN];

      if (Cnt_kind==0) NO0 = Spe_Total_NO[Cwan];
      else             NO0 = Spe_Total_CNO[Cwan];
    
      for (spin=0; spin<=SpinP_switch; spin++){
	for (i=0; i<NO0; i++){

	  for (Nc=0; Nc<GridN_Atom[Gc_AN]; Nc++){
	    ChiV0[spin][i][Nc] = Orbs_Grid[Mc_AN][Nc][i]*Vpot_Grid[spin][MGridListAtom[Mc_AN][Nc]];
	  }

	  Ncs = GridN_Atom[Gc_AN] - GridN_Atom[Gc_AN]%4; 
	  for (Nc=Ncs; Nc<GridN_Atom[Gc_AN]; Nc++){
	    ChiV0[spin][i][Nc] = Orbs_Grid[Mc_AN][Nc][i]*Vpot_Grid[spin][MGridListAtom[Mc_AN][Nc]];
	  }
	}
      }

      /* set data on h_AN */

      Gh_AN = natn[Gc_AN][h_AN];
      Mh_AN = F_G2M[Gh_AN];

      Rnh = ncn[Gc_AN][h_AN];
      Hwan = WhatSpecies[Gh_AN];

      if (Cnt_kind==0)
	NO1 = Spe_Total_NO[Hwan];
      else   
	NO1 = Spe_Total_CNO[Hwan];

      /* initialize tmp_H */

      for (spin=0; spin<=SpinP_switch; spin++){
	for (i=0; i<NO0; i++){
	  for (j=0; j<NO1; j++){
	    tmp_H[spin][i][j] = 0.0;                          
	  }
	}
      }
      
      /* summation of non-zero elements */

      for (Nog=0; Nog<NumOLG[Mc_AN][h_AN]-3; Nog+=4){

        Nc_0 = GListTAtoms1[Mc_AN][h_AN][Nog+0];
        Nc_1 = GListTAtoms1[Mc_AN][h_AN][Nog+1];
        Nc_2 = GListTAtoms1[Mc_AN][h_AN][Nog+2];
        Nc_3 = GListTAtoms1[Mc_AN][h_AN][Nog+3];

        Nh_0 = GListTAtoms2[Mc_AN][h_AN][Nog+0];
        Nh_1 = GListTAtoms2[Mc_AN][h_AN][Nog+1];
        Nh_2 = GListTAtoms2[Mc_AN][h_AN][Nog+2];
        Nh_3 = GListTAtoms2[Mc_AN][h_AN][Nog+3];

	/* store ChiV0 in tmp_ChiV0 */
 
	for (spin=0; spin<=SpinP_switch; spin++){
	  for (i=0; i<NO0; i++){
	    tmp_ChiV0_0[spin][i] = ChiV0[spin][i][Nc_0];
	    tmp_ChiV0_1[spin][i] = ChiV0[spin][i][Nc_1];
	    tmp_ChiV0_2[spin][i] = ChiV0[spin][i][Nc_2];
	    tmp_ChiV0_3[spin][i] = ChiV0[spin][i][Nc_3];
	  }
	}

	/* store Orbs_Grid in tmp_Orbs_Grid */

	for (j=0; j<NO1; j++){
	  tmp_Orbs_Grid_0[j] = Orbs_Grid[Mh_AN][Nh_0][j];
	  tmp_Orbs_Grid_1[j] = Orbs_Grid[Mh_AN][Nh_1][j];
	  tmp_Orbs_Grid_2[j] = Orbs_Grid[Mh_AN][Nh_2][j];
	  tmp_Orbs_Grid_3[j] = Orbs_Grid[Mh_AN][Nh_3][j];
	}

	/* integration */
 
	for (spin=0; spin<=SpinP_switch; spin++){
	  for (i=0; i<NO0; i++){

	    tmp0_0 = tmp_ChiV0_0[spin][i];
	    tmp0_1 = tmp_ChiV0_1[spin][i];
	    tmp0_2 = tmp_ChiV0_2[spin][i];
	    tmp0_3 = tmp_ChiV0_3[spin][i];

	    for (j=0; j<NO1; j++){
	      tmp_H[spin][i][j] += (  tmp0_0*tmp_Orbs_Grid_0[j]
			             +tmp0_1*tmp_Orbs_Grid_1[j]
				     +tmp0_2*tmp_Orbs_Grid_2[j]
				     +tmp0_3*tmp_Orbs_Grid_3[j]);
	    }

	  }
	}
      }

      mm = NumOLG[Mc_AN][h_AN]-(NumOLG[Mc_AN][h_AN]/4)*4;

      if(mm != 0){
 
	for (Nog=NumOLG[Mc_AN][h_AN]-mm; Nog<NumOLG[Mc_AN][h_AN]; Nog++){
 
	  Nc = GListTAtoms1[Mc_AN][h_AN][Nog];
	  Nh = GListTAtoms2[Mc_AN][h_AN][Nog];

	  /* store ChiV0 in tmp_ChiV0_0 */
 
	  for (spin=0; spin<=SpinP_switch; spin++){
	    for (i=0; i<NO0; i++){
	      tmp_ChiV0_0[spin][i] = ChiV0[spin][i][Nc];
	    }
	  }
 
	  /* store Orbs_Grid in tmp_Orbs_Grid */

	  for (j=0; j<NO1; j++){
	    tmp_Orbs_Grid_0[j] = Orbs_Grid[Mh_AN][Nh][j];
	  }

	  /* integration */

	  for (spin=0; spin<=SpinP_switch; spin++){

	    for (i=0; i<NO0; i++){
	      tmp0 = tmp_ChiV0_0[spin][i];
	      for (j=0; j<NO1; j++){
		tmp_H[spin][i][j] += tmp0*tmp_Orbs_Grid_0[j];
	      }
	    }
	  }
	}
      }
      
      /* add tmp_H to H or CntH */

      if (Cnt_kind==0){

	for (spin=0; spin<=SpinP_switch; spin++){
	  for (i=0; i<NO0; i++){
	    for (j=0; j<NO1; j++){
	      H[spin][Mc_AN][h_AN][i][j] += tmp_H[spin][i][j]*GridVol;
	    }
	  }
	}
      }
 
      else{
	for (spin=0; spin<=SpinP_switch; spin++){
	  for (i=0; i<NO0; i++){
	    for (j=0; j<NO1; j++){
	      CntH[spin][Mc_AN][h_AN][i][j] += tmp_H[spin][i][j]*GridVol;
	    }
	  }
	}
      }
      
    } /* Nloop */

    /* freeing of arrays */ 

    free(tmp_Orbs_Grid_3);
    free(tmp_Orbs_Grid_2);
    free(tmp_Orbs_Grid_1);
    free(tmp_Orbs_Grid_0);

    for (i=0; i<4; i++){
      free(tmp_ChiV0_3[i]);
    }
    free(tmp_ChiV0_3);

    for (i=0; i<4; i++){
      free(tmp_ChiV0_2[i]);
    }
    free(tmp_ChiV0_2);

    for (i=0; i<4; i++){
      free(tmp_ChiV0_1[i]);
    }
    free(tmp_ChiV0_1);

    for (i=0; i<4; i++){
      free(tmp_ChiV0_0[i]);
    }
    free(tmp_ChiV0_0);

    for (i=0; i<4; i++){
      for (j=0; j<List_YOUSO[7]; j++){
	free(tmp_H[i][j]);
      }
      free(tmp_H[i]);
    }
    free(tmp_H);

    for (i=0; i<4; i++){
      for (j=0; j<List_YOUSO[7]; j++){
	free(ChiV0[i][j]);
      }
      free(ChiV0[i]);
    }
    free(ChiV0);

    free(Chi0);

//#pragma omp flush(H,CntH)

  //} /* #pragma omp parallel */

  /* freeing of arrays */ 

  free(OneD2h_AN);
  free(OneD2Mc_AN);

  if(measure_time){ 
    dtime(&time2);
    printf("myid=%4d Time4=%18.10f\n",myid,time2-time1);fflush(stdout);
  }
}


void Calc_MatrixElements_dVH_Vxc_VNA_by_AI(int Cnt_kind)
{
  int Mc_AN,Gc_AN,Mh_AN,h_AN,Gh_AN;
  int Nh0,Nh1,Nh2,Nh3;
  int Nc0,Nc1,Nc2,Nc3;
  int MN0,MN1,MN2,MN3;
  int Nloop,OneD_Nloop;
  int *OneD2spin,*OneD2Mc_AN,*OneD2h_AN;
  int numprocs,myid;
  double time0,time1,time2,mflops;

  if(measure_time) dtime(&time1);

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /* one-dimensionalization of loops */

  Nloop = 0;
  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];    
    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
      Nloop++;
    }
  }

  OneD2Mc_AN = (int*)malloc(sizeof(int)*Nloop);
  OneD2h_AN = (int*)malloc(sizeof(int)*Nloop);

  Nloop = 0;
  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];    
    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

      OneD2Mc_AN[Nloop] = Mc_AN;
      OneD2h_AN[Nloop] = h_AN;
      Nloop++;
    }
  }

  OneD_Nloop = Nloop;

  if(measure_time){ 
    dtime(&time2);
    printf("myid=%4d Time3=%18.10f\n",myid,time2-time1);fflush(stdout);
  }

  /* numerical integration */

  if(measure_time) dtime(&time1);

#pragma omp parallel 
  {
    int Nloop,spin,Mc_AN,h_AN,Gh_AN,Mh_AN,Hwan,NOLG;
    int Gc_AN,Cwan,NO0,NO1,spin0=-1,Mc_AN0=0;
    int i,j,Nc,MN,GNA,Nog,Nh,OMPID,Nthrds;
    int M,N,K,lda,ldb,ldc,ii,jj;
    double alpha,beta,Vpot;
    double sum0,sum1,sum2,sum3,sum4;
    double *ChiV0,*Chi1,*ChiV0_2,*C;

    /* allocation of arrays */

    /* AITUNE */
    double **AI_tmpH[4];
    {
      /* get size of temporary buffer */
      int AI_MaxNO = 0;
      if (Cnt_kind==0){
	int spe;
	for(spe = 0; spe < SpeciesNum; spe++){
	  if(AI_MaxNO < Spe_Total_NO[spe]){ AI_MaxNO = Spe_Total_NO[spe];}
	}
      }else{
	int spe;
	for(spe = 0; spe < SpeciesNum; spe++){
	  if(AI_MaxNO < Spe_Total_CNO[spe]){ AI_MaxNO = Spe_Total_CNO[spe];}
	}
      }
			
      int spin;
      for (spin=0; spin<=SpinP_switch; spin++){
	AI_tmpH[spin] = (double**)malloc(sizeof(double*)*AI_MaxNO);

	int i;
	double *p = (double*)malloc(sizeof(double)*AI_MaxNO*AI_MaxNO);
	for(i = 0; i < AI_MaxNO; i++){
	  AI_tmpH[spin][i] = p;
	  p += AI_MaxNO;
	}
      }			
    }
    /* AITUNE */

    /* starting of one-dimensionalized loop */

#pragma omp for schedule(static,1)  /* guided */  /* AITUNE */
    for (Nloop = 0; Nloop < OneD_Nloop; Nloop++){ /* AITUNE */

      int Mc_AN = OneD2Mc_AN[Nloop];
      int h_AN = OneD2h_AN[Nloop];
      int Gc_AN = M2G[Mc_AN];    
      int Gh_AN = natn[Gc_AN][h_AN];
      int Mh_AN = F_G2M[Gh_AN];
      int Cwan = WhatSpecies[Gc_AN];
      int Hwan = WhatSpecies[Gh_AN];
      int GNA = GridN_Atom[Gc_AN];
      int NOLG = NumOLG[Mc_AN][h_AN]; 

      int NO0,NO1;
      if (Cnt_kind==0){
	NO0 = Spe_Total_NO[Cwan];
	NO1 = Spe_Total_NO[Hwan];
      }
      else{
	NO0 = Spe_Total_CNO[Cwan];
	NO1 = Spe_Total_CNO[Hwan];
      }

      /* quadrature for Hij  */

      /* AITUNE change order of loop */
      if(SpinP_switch==0){
	/* AITUNE temporary buffer for "unroll-Jammed" HLO optimization by Intel */
		
	if (Cnt_kind==0){
	  int i;
	  for (i=0; i<NO0; i++){
	    int j;
	    for (j=0; j<NO1; j++){
	      AI_tmpH[0][i][j] = H[0][Mc_AN][h_AN][i][j];
	    }
	  }
	}else{
	  int i;
	  for (i=0; i<NO0; i++){
	    int j;
	    for (j=0; j<NO1; j++){
	      AI_tmpH[0][i][j] = CntH[0][Mc_AN][h_AN][i][j];
	    }
	  }
	}

	int Nog;
	for (Nog=0; Nog<NOLG; Nog++){

	  int Nc = GListTAtoms1[Mc_AN][h_AN][Nog];
	  int MN = MGridListAtom[Mc_AN][Nc];
	  int Nh = GListTAtoms2[Mc_AN][h_AN][Nog];
		
	  double AI_tmp_GVVG = GridVol * Vpot_Grid[0][MN];
	  
	  if (G2ID[Gh_AN]==myid){
	    int i;
	    for (i=0; i<NO0; i++){

	      double AI_tmp_i = AI_tmp_GVVG * Orbs_Grid[Mc_AN][Nc][i];
	      int j;

	      for (j=0; j<NO1; j++){
		AI_tmpH[0][i][j] += AI_tmp_i * Orbs_Grid[Mh_AN][Nh][j];
	      }		
	    }
			
	  }else{
	    int i;
	    for (i=0; i<NO0; i++){
					
	      double AI_tmp_i = AI_tmp_GVVG * Orbs_Grid[Mc_AN][Nc][i];
	      int j;

	      for (j=0; j<NO1; j++){
		AI_tmpH[0][i][j] += AI_tmp_i * Orbs_Grid_FNAN[Mc_AN][h_AN][Nog][j];
	      }
	    }
	  }
		
	}/* Nog */
	
	if (Cnt_kind==0){
	  int i;
	  for (i=0; i<NO0; i++){
	    int j;
	    for (j=0; j<NO1; j++){
	      H[0][Mc_AN][h_AN][i][j] = AI_tmpH[0][i][j];
	    }
	  }
	}else{
	  int i;
	  for (i=0; i<NO0; i++){
	    int j;
	    for (j=0; j<NO1; j++){
	      CntH[0][Mc_AN][h_AN][i][j] = AI_tmpH[0][i][j];
	    }
	  }
	}

      }else if (SpinP_switch == 1){
	  
	/* AITUNE temporary buffer for "unroll-Jammed" HLO optimization by Intel */
		
	if (Cnt_kind==0){
	  int i;
	  for (i=0; i<NO0; i++){
	    int j;
	    for (j=0; j<NO1; j++){
	      AI_tmpH[0][i][j] = H[0][Mc_AN][h_AN][i][j];
	    }
	  }
	  for (i=0; i<NO0; i++){
	    int j;
	    for (j=0; j<NO1; j++){
	      AI_tmpH[1][i][j] = H[1][Mc_AN][h_AN][i][j];
	    }
	  }
	}else{
	  int i;
	  for (i=0; i<NO0; i++){
	    int j;
	    for (j=0; j<NO1; j++){
	      AI_tmpH[0][i][j] = CntH[0][Mc_AN][h_AN][i][j];
	    }
	  }
	  for (i=0; i<NO0; i++){
	    int j;
	    for (j=0; j<NO1; j++){
	      AI_tmpH[1][i][j] = CntH[1][Mc_AN][h_AN][i][j];
	    }
	  }
	}

	int Nog;
	for (Nog=0; Nog<NOLG; Nog++){

	  int Nc = GListTAtoms1[Mc_AN][h_AN][Nog];
	  int MN = MGridListAtom[Mc_AN][Nc];
	  int Nh = GListTAtoms2[Mc_AN][h_AN][Nog];
		
	  double AI_tmp_GVVG = GridVol * Vpot_Grid[0][MN];
	  double AI_tmp_GVVG1 = GridVol * Vpot_Grid[1][MN];

	  if (G2ID[Gh_AN]==myid){
				
	    int i;
	    for (i=0; i<NO0; i++){

	      double AI_tmp_i = AI_tmp_GVVG * Orbs_Grid[Mc_AN][Nc][i];
	      int j;
	      for (j=0; j<NO1; j++){
		AI_tmpH[0][i][j] += AI_tmp_i * Orbs_Grid[Mh_AN][Nh][j];
	      }		
	    }

	    for (i=0; i<NO0; i++){

	      double AI_tmp_i = AI_tmp_GVVG1 * Orbs_Grid[Mc_AN][Nc][i];

	      int j;
	      for (j=0; j<NO1; j++){
		AI_tmpH[1][i][j] += AI_tmp_i * Orbs_Grid[Mh_AN][Nh][j];
	      }		
	    }
			
	  }else{
	    int i;
	    for (i=0; i<NO0; i++){
					
	      double AI_tmp_i = AI_tmp_GVVG * Orbs_Grid[Mc_AN][Nc][i];
	      int j;
	      for (j=0; j<NO1; j++){
		AI_tmpH[0][i][j] += AI_tmp_i * Orbs_Grid_FNAN[Mc_AN][h_AN][Nog][j];
	      }
	    }
			
	    for (i=0; i<NO0; i++){
					
	      double AI_tmp_i = AI_tmp_GVVG1 * Orbs_Grid[Mc_AN][Nc][i];
	      int j;
	      for (j=0; j<NO1; j++){
		AI_tmpH[1][i][j] += AI_tmp_i * Orbs_Grid_FNAN[Mc_AN][h_AN][Nog][j];
	      }
	    }
	  }
		
	}/* Nog */
	
	/* AITUNE copy from temporary buffer */

	if (Cnt_kind==0){
	  int i;
	  for (i=0; i<NO0; i++){
	    int j;
	    for (j=0; j<NO1; j++){
	      H[0][Mc_AN][h_AN][i][j] = AI_tmpH[0][i][j];
	    }
	  }
	  for (i=0; i<NO0; i++){
	    int j;
	    for (j=0; j<NO1; j++){
	      H[1][Mc_AN][h_AN][i][j] = AI_tmpH[1][i][j];
	    }
	  }
	}else{
	  int i;
	  for (i=0; i<NO0; i++){
	    int j;
	    for (j=0; j<NO1; j++){
	      CntH[0][Mc_AN][h_AN][i][j] = AI_tmpH[0][i][j];
	    }
	  }
	  for (i=0; i<NO0; i++){
	    int j;
	    for (j=0; j<NO1; j++){
	      CntH[1][Mc_AN][h_AN][i][j] = AI_tmpH[1][i][j];
	    }
	  }
	}

      }

      else{ /* SpinP_switch==3 */
	 
	/* AITUNE temporary buffer for "unroll-Jammed" HLO optimization by Intel */

	if (Cnt_kind==0){
	  int i;
	  for (i=0; i<NO0; i++){
	    int j;
	    for (j=0; j<NO1; j++){
	      AI_tmpH[0][i][j] = H[0][Mc_AN][h_AN][i][j];
	      AI_tmpH[1][i][j] = H[1][Mc_AN][h_AN][i][j];
	      AI_tmpH[2][i][j] = H[2][Mc_AN][h_AN][i][j];
	      AI_tmpH[3][i][j] = H[3][Mc_AN][h_AN][i][j];
	    }
	  }
	}

        else{
	  int i;
	  for (i=0; i<NO0; i++){
	    int j;
	    for (j=0; j<NO1; j++){
	      AI_tmpH[0][i][j] = CntH[0][Mc_AN][h_AN][i][j];
	      AI_tmpH[1][i][j] = CntH[1][Mc_AN][h_AN][i][j];
	      AI_tmpH[2][i][j] = CntH[2][Mc_AN][h_AN][i][j];
	      AI_tmpH[3][i][j] = CntH[3][Mc_AN][h_AN][i][j];
	    }
	  }
	}

	if (Cnt_kind==0){
	  int i;
	  for (i=0; i<NO0; i++){
	    int j;
	    for (j=0; j<NO1; j++){
	      AI_tmpH[0][i][j] = H[0][Mc_AN][h_AN][i][j];
	    }
	  }
	  for (i=0; i<NO0; i++){
	    int j;
	    for (j=0; j<NO1; j++){
	      AI_tmpH[1][i][j] = H[1][Mc_AN][h_AN][i][j];
	    }
	  }
	  for (i=0; i<NO0; i++){
	    int j;
	    for (j=0; j<NO1; j++){
	      AI_tmpH[2][i][j] = H[2][Mc_AN][h_AN][i][j];
	    }
	  }
	  for (i=0; i<NO0; i++){
	    int j;
	    for (j=0; j<NO1; j++){
	      AI_tmpH[3][i][j] = H[3][Mc_AN][h_AN][i][j];
	    }
	  }
	}else{
	  int i;
	  for (i=0; i<NO0; i++){
	    int j;
	    for (j=0; j<NO1; j++){
	      AI_tmpH[0][i][j] = CntH[0][Mc_AN][h_AN][i][j];
	    }
	  }
	  for (i=0; i<NO0; i++){
	    int j;
	    for (j=0; j<NO1; j++){
	      AI_tmpH[1][i][j] = CntH[1][Mc_AN][h_AN][i][j];
	    }
	  }
	  for (i=0; i<NO0; i++){
	    int j;
	    for (j=0; j<NO1; j++){
	      AI_tmpH[2][i][j] = CntH[2][Mc_AN][h_AN][i][j];
	    }
	  }
	  for (i=0; i<NO0; i++){
	    int j;
	    for (j=0; j<NO1; j++){
	      AI_tmpH[3][i][j] = CntH[3][Mc_AN][h_AN][i][j];
	    }
	  }
			
	}

	int Nog;

	for (Nog=0; Nog<NOLG; Nog++){

	  int Nc = GListTAtoms1[Mc_AN][h_AN][Nog];
	  int MN = MGridListAtom[Mc_AN][Nc];
	  int Nh = GListTAtoms2[Mc_AN][h_AN][Nog];
		
	  double AI_tmp_GVVG = GridVol * Vpot_Grid[0][MN];
	  double AI_tmp_GVVG1 = GridVol * Vpot_Grid[1][MN];
	  double AI_tmp_GVVG2 = GridVol * Vpot_Grid[2][MN];
	  double AI_tmp_GVVG3 = GridVol * Vpot_Grid[3][MN];

	  if (G2ID[Gh_AN]==myid){

	    int i;
	    for (i=0; i<NO0; i++){

	      double AI_tmp_i = AI_tmp_GVVG * Orbs_Grid[Mc_AN][Nc][i];

	      for (j=0; j<NO1; j++){
		AI_tmpH[0][i][j] += AI_tmp_i * Orbs_Grid[Mh_AN][Nh][j];
	      }		
	    }
			
	    for (i=0; i<NO0; i++){

	      double AI_tmp_i = AI_tmp_GVVG1 * Orbs_Grid[Mc_AN][Nc][i];

	      for (j=0; j<NO1; j++){
		AI_tmpH[1][i][j] += AI_tmp_i * Orbs_Grid[Mh_AN][Nh][j];
	      }		
	    }
			
	    for (i=0; i<NO0; i++){

	      double AI_tmp_i = AI_tmp_GVVG2 * Orbs_Grid[Mc_AN][Nc][i];

	      for (j=0; j<NO1; j++){
		AI_tmpH[2][i][j] += AI_tmp_i * Orbs_Grid[Mh_AN][Nh][j];
	      }		
	    }
			
	    for (i=0; i<NO0; i++){

	      double AI_tmp_i = AI_tmp_GVVG3 * Orbs_Grid[Mc_AN][Nc][i];

	      for (j=0; j<NO1; j++){
		AI_tmpH[3][i][j] += AI_tmp_i * Orbs_Grid[Mh_AN][Nh][j];
	      }		
	    }
			
	  }

          else{

	    int i;
	    for (i=0; i<NO0; i++){
					
	      double AI_tmp_i = AI_tmp_GVVG * Orbs_Grid[Mc_AN][Nc][i];

	      for (j=0; j<NO1; j++){
		AI_tmpH[0][i][j] += AI_tmp_i * Orbs_Grid_FNAN[Mc_AN][h_AN][Nog][j];
	      }
	    }
			
	    for (i=0; i<NO0; i++){
					
	      double AI_tmp_i = AI_tmp_GVVG1 * Orbs_Grid[Mc_AN][Nc][i];

	      for (j=0; j<NO1; j++){
		AI_tmpH[1][i][j] += AI_tmp_i * Orbs_Grid_FNAN[Mc_AN][h_AN][Nog][j];
	      }
	    }
	    for (i=0; i<NO0; i++){
					
	      double AI_tmp_i = AI_tmp_GVVG2 * Orbs_Grid[Mc_AN][Nc][i];

	      for (j=0; j<NO1; j++){
		AI_tmpH[2][i][j] += AI_tmp_i * Orbs_Grid_FNAN[Mc_AN][h_AN][Nog][j];
		
	      }
	    }			
	    for (i=0; i<NO0; i++){
					
	      double AI_tmp_i = AI_tmp_GVVG3 * Orbs_Grid[Mc_AN][Nc][i];

	      for (j=0; j<NO1; j++){
		AI_tmpH[3][i][j] += AI_tmp_i * Orbs_Grid_FNAN[Mc_AN][h_AN][Nog][j];
	      }
	    }
	  }

		
	} /* Nog */
	
	/* AITUNE copy from temporary buffer */

	if (Cnt_kind==0){
	  int i;
	  for (i=0; i<NO0; i++){
	    int j;
	    for (j=0; j<NO1; j++){
	      H[0][Mc_AN][h_AN][i][j] = AI_tmpH[0][i][j];
	    }
	  }
	  for (i=0; i<NO0; i++){
	    int j;
	    for (j=0; j<NO1; j++){
	      H[1][Mc_AN][h_AN][i][j] = AI_tmpH[1][i][j];
	    }
	  }
	  for (i=0; i<NO0; i++){
	    int j;
	    for (j=0; j<NO1; j++){
	      H[2][Mc_AN][h_AN][i][j] = AI_tmpH[2][i][j];
	    }
	  }
	  for (i=0; i<NO0; i++){
	    int j;
	    for (j=0; j<NO1; j++){
	      H[3][Mc_AN][h_AN][i][j] = AI_tmpH[3][i][j];
	    }
	  }
	}else{
	  int i;
	  for (i=0; i<NO0; i++){
	    int j;
	    for (j=0; j<NO1; j++){
	      CntH[0][Mc_AN][h_AN][i][j] = AI_tmpH[0][i][j];
	    }
	  }
	  for (i=0; i<NO0; i++){
	    int j;
	    for (j=0; j<NO1; j++){
	      CntH[1][Mc_AN][h_AN][i][j] = AI_tmpH[1][i][j];
	    }
	  }
	  for (i=0; i<NO0; i++){
	    int j;
	    for (j=0; j<NO1; j++){
	      CntH[2][Mc_AN][h_AN][i][j] = AI_tmpH[2][i][j];
	    }
	  }
	  for (i=0; i<NO0; i++){
	    int j;
	    for (j=0; j<NO1; j++){
	      CntH[3][Mc_AN][h_AN][i][j] = AI_tmpH[3][i][j];
	    }
	  }
	}
      }
      /* AITUNE change order of loop */

    } /* Nloop */

    /* freeing of arrays */
    {
      int spin;
      for (spin=0; spin<=SpinP_switch; spin++){
	free(AI_tmpH[spin][0]);
	free(AI_tmpH[spin]);
      }
    }

  } /* pragma omp parallel */ 
  /* freeing of arrays */

  free(OneD2Mc_AN);
  free(OneD2h_AN);

  if(measure_time){ 
    dtime(&time2);
    printf("myid=%4d Time4=%18.10f\n",myid,time2-time1);fflush(stdout);
  }
}

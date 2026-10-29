#include "openmx_gpu_backend.h"
#include "openmx_gpu_device.h"
/**********************************************************************
  Total_Energy.c:

     Total_Energy.c is a subrutine to calculate the total energy

  Log of Total_Energy.c:

     22/Nov/2001  Released by T. Ozaki
     19/Feb/2006  The subroutine name 'Correction_Energy' was changed 
                  to 'Total_Energy'

***********************************************************************/

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "openmx_common.h"
#include "openmx_xc.h"
#include "openmx_env.h"
#include "openmx_core_vna_matrix.h"
#include "openmx_energy_workspace.h"
#include "openmx_quadrature.h"
#include "openmx_host_simd.h"
#include "openmx_core_energy_simd.h"
#include "openmx_neutral_atom.h"
#include "openmx_core_neutral_grid.h"
#include "openmx_core_neutral_grid_metadata.h"
#include "mpi.h"
#include <omp.h>

/* the section timers print when OPENMX_BAND_PROFILE=1 (formerly a
   compile-time switch) */
static int TotalEnergy_ProfileEnabled(void)
{
  return openmx_env_flag_at(OPENMX_ENV_BAND_PROFILE, 0);
}
#define measure_time (TotalEnergy_ProfileEnabled())

#ifdef MAX 
#undef MAX
#endif
#define MAX(a,b) ((a)>(b))?  (a):(b) 

#ifdef MIN
#undef MIN
#endif
#define MIN(a,b) (((a)<(b))?  (a):(b))

static double Calc_Ecore(OpenMXSolverInvocation *scratch);
static double Calc_EH0(const OpenMXEnergyWorkspace *workspace, OpenMXCoreContext *core, const OpenMXAtomicViews *atomic, int MD_iter);
static double Calc_Ekin(const OpenMXEnergyWorkspace *workspace, const OpenMXElectronicViews *electronic);
static double Calc_Ena(const OpenMXEnergyWorkspace *workspace, const OpenMXVnaMatrixViews *vna, const OpenMXElectronicViews *electronic, const OpenMXLocalBasisViews *local_basis);
static double Calc_Enl(const OpenMXEnergyWorkspace *workspace, const OpenMXElectronicViews *electronic, const OpenMXLocalBasisViews *local_basis);
static double Calc_ECH(const OpenMXEnergyWorkspace *workspace, const OpenMXElectronicViews *electronic);
static void Calc_EXC_EH1(OpenMXSolverInvocation *scratch, const OpenMXAtomicViews *atomic, const OpenMXGaussRuleData *radial, double ECE[]);
static void Calc_Edc(double ECE[]);
static double Calc_Ehub(const OpenMXLocalBasisViews *local_basis);   /* --- added by MJ  */
static double Calc_EdftD();  /* added by okuno */
static double Calc_EdftD3(OpenMXSolverInvocation *scratch); /* added by Ellner */
static void EH0_TwoCenter(const OpenMXNeutralGridTables *tables,const OpenMXNeutralGridMetadataView *metadata,const OpenMXAtomicViews *atomic, int Gc_AN, int h_AN, double VH0ij[4]);
static void EH0_TwoCenter_at_Cutoff(OpenMXSolverInvocation *scratch,const OpenMXNeutralGridTables *tables,const OpenMXNeutralGridMetadataView *metadata, const OpenMXAtomicViews *atomic, int wan1, int wan2, double VH0ij[4]);
static void Energy_Decomposition(const OpenMXBandEnergies *energies, double ECE[], const OpenMXElectronicViews *electronic);
static void Energy_Decomposition_CWF(OpenMXSolverInvocation *scratch, const OpenMXCWFObservables *cwf_values, double ECE[]);
#if OPENMX_GPU_HIP
static int TotalEnergyUseOpenMP(void);
void TotalEnergy_EXC_EH1_Grid_OpenMP(int spinmax, double *My_Ena, double *My_Eef,
#else
static int TotalEnergyUseOpenACC(void);
void TotalEnergy_EXC_EH1_Grid_OpenACC(int spinmax, double *My_Ena, double *My_Eef,
#endif
                                      double *My_EH1, double My_EXC[2]);
#if OPENMX_GPU_HIP
void TotalEnergy_Dipole_Grid_OpenMP(int GNs, double *My_E_dpx, double *My_E_dpy,
#else
void TotalEnergy_Dipole_Grid_OpenACC(int GNs, double *My_E_dpx, double *My_E_dpy,
#endif
                                     double *My_E_dpz, double *My_E_dpx_BG,
                                     double *My_E_dpy_BG, double *My_E_dpz_BG);
#if OPENMX_GPU_HIP
void TotalEnergy_CWF_Dc_Grid_OpenMP(int spinmax, double My_dcEH1[2], double My_dcEXC[2]);
void TotalEnergy_Exc0_Batch_OpenMP(const OpenMXAtomicViews *atomic, const OpenMXGaussRuleData *radial, const OpenMXLebedevRuleData *lebedev, double *sum_out);
void TotalEnergy_EH0_TwoCenter_Batch_OpenMP(const OpenMXNeutralGridTables *tables,const OpenMXNeutralGridMetadataView *metadata,const OpenMXAtomicViews *atomic, int pair_count, int *pair_ban, int *pair_wan2,
#else
void TotalEnergy_CWF_Dc_Grid_OpenACC(int spinmax, double My_dcEH1[2], double My_dcEXC[2]);
void TotalEnergy_Exc0_Batch_OpenACC(const OpenMXAtomicViews *atomic, const OpenMXGaussRuleData *radial, const OpenMXLebedevRuleData *lebedev, double *sum_out);
void TotalEnergy_EH0_TwoCenter_Batch_OpenACC(const OpenMXNeutralGridTables *tables,const OpenMXNeutralGridMetadataView *metadata,const OpenMXAtomicViews *atomic, int pair_count, int *pair_ban, int *pair_wan2,
#endif
                                             int *pair_has_deriv, double *pair_dis,
                                             double *pair_dirx, double *pair_diry, double *pair_dirz,
                                             double *out0, double *out1, double *out2, double *out3);

/* Atom-pair indices belong to the current energy invocation. */


#if OPENMX_GPU_HIP
static int TotalEnergyUseOpenMP(void)
#else
static int TotalEnergyUseOpenACC(void)
#endif
{
  /* The batched EH0 integrals are spin-independent and the grid
     reductions use the same spinmax<=1 expressions as the host loops
     for every SpinP_switch, so the non-collinear case runs on the
     device as well. */
  return openmx_env_flag_at(OPENMX_ENV_TOTALENERGY_GPU, 1) && (scf_eigen_lib_flag == GPUSOLVER) && gpu_rank_device_usable();
}

#ifdef OPENMX_HOST_SIMD_H
/* Device-or-host decisions of the GPU batches of this file
   (OpenMX_HostSimdStageOnHost, collective over mpi_comm_level1: every rank
   calls them at the same point, behind the global GPUSOLVER test). need =
   the device working set of the whole batch on this rank, laid out as the
   batch's task views (each view on a 16-byte boundary); budget = free device
   memory minus the 256 MiB reserve of the worker capacity check. On one
   device the task executor runs a batch in one piece or not at all, so a
   batch that does not fit ends in its task's host body (the host SIMD
   kernels in Total_Energy_GPU.c). OPENMX_TOTALENERGY_GPU=0 switches the
   batches off: the host branches below replace their task bodies. */
static size_t TotalEnergy_AddViews(size_t bytes,size_t views,size_t count,size_t item)
{
  if (bytes==SIZE_MAX || (item && count>(SIZE_MAX-15)/item)) return SIZE_MAX;
  const size_t one=(count*item+15)&~(size_t)15;
  if (views && one>(SIZE_MAX-bytes)/views) return SIZE_MAX;
  return bytes+views*one;
}
static int TotalEnergy_StageOnHost(const char *stage,size_t need)
{
  if (!openmx_env_flag_at(OPENMX_ENV_TOTALENERGY_GPU, 1)) need = SIZE_MAX;
  return OpenMX_HostSimdStageOnHost(stage,need,OpenMX_HostSimdDeviceBudget((size_t)256<<20));
}
/* EH0 two-centre pairs: 3 int + 4 double inputs and 4 double outputs per
   pair, the species tables of the neutral-atom grids and VPS meshes */
static size_t TotalEnergy_EH0Bytes(size_t pairs)
{
  const size_t species=(size_t)SpeciesNum,mesh=(size_t)List_YOUSO[22];
  size_t b=TotalEnergy_AddViews(0,3,pairs,sizeof(int));
  b=TotalEnergy_AddViews(b,8,pairs,sizeof(double));
  b=TotalEnergy_AddViews(b,2,species,sizeof(int));
  b=TotalEnergy_AddViews(b,2,species,sizeof(double));
  b=TotalEnergy_AddViews(b,5,species*(size_t)Max_TGN_EH0,sizeof(double));
  b=TotalEnergy_AddViews(b,2,species*mesh,sizeof(double));
  return TotalEnergy_AddViews(b,1,species*(mesh+2),sizeof(double));
}
/* Exc0 atoms: per local atom its centre and neighbour lists (up to the
   largest FNAN+1), the energy and pair-force outputs and the per-point
   prefactor scratch of all radial x Lebedev points, plus the PAO tables */
static size_t TotalEnergy_Exc0Bytes(int points)
{
  size_t atoms=(size_t)Matomnum,nb=1,species=(size_t)SpeciesNum,mesh=(size_t)List_YOUSO[21];
  for (int mc=1; mc<=Matomnum; mc++) if (nb<(size_t)FNAN[M2G[mc]]+1) nb=(size_t)FNAN[M2G[mc]]+1;
  size_t b=TotalEnergy_AddViews(0,1,atoms,sizeof(int));
  b=TotalEnergy_AddViews(b,4,atoms,sizeof(double));
  b=TotalEnergy_AddViews(b,1,atoms*nb,sizeof(int));
  b=TotalEnergy_AddViews(b,4,atoms*nb,sizeof(double));
  b=TotalEnergy_AddViews(b,1,species,sizeof(int));
  b=TotalEnergy_AddViews(b,2,species*mesh,sizeof(double));
  b=TotalEnergy_AddViews(b,1,species*(mesh+2),sizeof(double));
  b=TotalEnergy_AddViews(b,4,(size_t)points,sizeof(double));
  b=TotalEnergy_AddViews(b,2,(size_t)CoarseGL_Mesh,sizeof(double));
  b=TotalEnergy_AddViews(b,1,atoms,sizeof(double));
  b=TotalEnergy_AddViews(b,1,atoms*3*nb,sizeof(double));
  return TotalEnergy_AddViews(b,1,atoms*(size_t)CoarseGL_Mesh*(size_t)points,sizeof(double));
}
/* grid reductions: the coarse-grid arrays each kind reads (as
   TotalEnergy_Grid_Capacity selects them) and the six sums */
static size_t TotalEnergy_GridBytes(int kind)
{
  size_t arrays=kind==1 ? 2 : kind==2 ? 6 : 9+(ProExpn_VNA==0)+(E_Field_switch==1);
  return TotalEnergy_AddViews(TotalEnergy_AddViews(0,arrays,(size_t)My_NumGridB_AB,sizeof(double)),1,6,sizeof(double));
}
#endif




typedef struct {
  OpenMXCoreContext *core;
  const OpenMXBandEnergies *energies;
  const OpenMXCWFObservables *cwf_values;
  const OpenMXAtomicViews *atomic;
  const OpenMXVnaMatrixViews *vna;
  int MD_iter;
  double *ECE;
  const OpenMXElectronicViews *electronic;
  const OpenMXLocalBasisViews *local_basis;
} TotalEnergyArguments;

static void Total_Energy_owned(OpenMXSolverInvocation *scratch, OpenMXCoreContext *core, const OpenMXBandEnergies *energies, const OpenMXCWFObservables *cwf_values, const OpenMXAtomicViews *atomic, const OpenMXVnaMatrixViews *vna, int MD_iter, double ECE[], const OpenMXElectronicViews *electronic, const OpenMXLocalBasisViews *local_basis);
static int TotalEnergyOperation(OpenMXSolverInvocation *scratch,void *opaque,double *result)
{
  const TotalEnergyArguments *a=(const TotalEnergyArguments*)opaque;
  Total_Energy_owned(scratch,a->core,a->energies,a->cwf_values,a->atomic,a->vna,a->MD_iter,a->ECE,a->electronic,a->local_basis);
  *result=0.0;
  return 1;
}

double Total_Energy(OpenMXCoreContext *core, const OpenMXBandEnergies *energies, const OpenMXCWFObservables *cwf_values, const OpenMXAtomicViews *atomic, const OpenMXVnaMatrixViews *vna, int MD_iter, double ECE[], const OpenMXElectronicViews *electronic, const OpenMXLocalBasisViews *local_basis)
{
  TotalEnergyArguments arguments={core,energies,cwf_values,atomic,vna,MD_iter,ECE,electronic,local_basis};
  double start,end;
  dtime(&start);
  (void)OpenMX_Solver_Run(TotalEnergyOperation,&arguments);
  dtime(&end);
  return end-start;
}

static void Total_Energy_owned(OpenMXSolverInvocation *scratch, OpenMXCoreContext *core, const OpenMXBandEnergies *energies, const OpenMXCWFObservables *cwf_values, const OpenMXAtomicViews *atomic, const OpenMXVnaMatrixViews *vna, int MD_iter, double ECE[], const OpenMXElectronicViews *electronic, const OpenMXLocalBasisViews *local_basis)
{ 
  const OpenMXGaussRuleData radial=OpenMX_NeutralQuadrature(core,CoarseGL_Mesh,mpi_comm_level1);
  int numprocs,myid;
  int Mc_AN,Gc_AN,h_AN;
  double stime,etime;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);


  /****************************************************
   For OpenMP:
   making of arrays of the one-dimensionalized loop
  ****************************************************/

  size_t pair_count=0;
  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];
    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++) pair_count++;
  }
  OpenMXEnergyWorkspace workspace;
  OpenMX_Energy_PairsPrepare(scratch,pair_count,&workspace);
  int *OneD2Mc_AN=openmx_solver_int_matrix_row(&workspace.pair_indices,0);
  int *OneD2h_AN=openmx_solver_int_matrix_row(&workspace.pair_indices,1);
  int OneD_Nloop=0;
  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];
    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
      OneD2Mc_AN[OneD_Nloop] = Mc_AN;
      OneD2h_AN[OneD_Nloop] = h_AN;
      OneD_Nloop++;
    }
  }

  /****************************************************
               core-core repulsion energy
  ****************************************************/

  dtime(&stime);

  ECE[0] = Calc_Ecore(scratch);

  dtime(&etime);
  if(myid==0 && measure_time){
    printf("Time for Ecore=%18.5f\n",etime-stime);fflush(stdout);
  } 
  
  /****************************************************
              EH0 = -1/2\int n^a(r) V^a_H dr
  ****************************************************/

  dtime(&stime);

  ECE[1] = Calc_EH0(&workspace, core, atomic, MD_iter);

  dtime(&etime);
  if(myid==0 && measure_time){
    printf("Time for EH0=%18.5f\n",etime-stime);fflush(stdout);
  } 

  /****************************************************
                    kinetic energy
  ****************************************************/

  dtime(&stime);

  if (F_Kin_flag==1)  ECE[2] = Calc_Ekin(&workspace, electronic);

  dtime(&etime);
  if(myid==0 && measure_time){
    printf("Time for Ekin=%18.5f\n",etime-stime);fflush(stdout);
  } 

  /****************************************************
              neutral atom potential energy
  ****************************************************/

  dtime(&stime);

  if (F_VNA_flag==1 && ProExpn_VNA==1)  ECE[3] = Calc_Ena(&workspace, vna, electronic, local_basis);

  dtime(&etime);
  if(myid==0 && measure_time){
    printf("Time for Ena=%18.5f\n",etime-stime);fflush(stdout);
  } 

  /****************************************************
           non-local pseudo potential energy
  ****************************************************/

  dtime(&stime);

  if (F_NL_flag==1)  ECE[4] = Calc_Enl(&workspace, electronic, local_basis);

  dtime(&etime);
  if(myid==0 && measure_time){
    printf("Time for Enl=%18.5f\n",etime-stime);fflush(stdout);
  } 

  /****************************************************
        The penalty term to create a core hole 
  ****************************************************/

  dtime(&stime);

  if (core_hole_state_flag==1 && F_CH_flag==1){
    ECE[14] = Calc_ECH(&workspace, electronic);
  }

  dtime(&etime);
  if(myid==0 && measure_time){
    printf("Time for ECH=%18.5f\n",etime-stime);fflush(stdout);
  } 

  /****************************************************
     EXC = \sum_{\sigma} (n_{\sigma}(r)+n_pcc(r)\epsilon_{xc}
     EH1 = 1/2\int {n(r)-n_a(r)} \delta V_H(r) dr
     if (ProExpn_VNA==0) Ena = \int n(r) Vna(r) dr
  ****************************************************/

  dtime(&stime);

  Calc_EXC_EH1(scratch, atomic, &radial, ECE);
  if (Solver==12) Calc_Edc(ECE);

  dtime(&etime);
  if(myid==0 && measure_time){
    printf("Time for EXC_EH1=%18.5f\n",etime-stime);fflush(stdout);
  } 

  /****************************************************
    LDA+U energy   --- added by MJ
  ****************************************************/

  if (F_U_flag==1){
    if (Hub_U_switch==1)  ECE[8] = Calc_Ehub(local_basis);
    else                  ECE[8] = 0.0;
  }

  /*********************************************************
   DFT-D2 and D3 by Grimme (implemented by Okuno and Ellner)
  *********************************************************/

  if (F_dftD_flag==1){
    if(dftD_switch==1){
      if(version_dftD==2) ECE[13] = Calc_EdftD(); 
      if(version_dftD==3) ECE[13] = Calc_EdftD3(scratch);
    }
    else ECE[13] = 0.0;
  }

  /****************************************************
    Stress coming from volume terms
          
    ECE[3]; Una
    ECE[5]; UH1

    The volume term coming from Exc is added in 
    Calc_EXC_EH1 
  ****************************************************/

  if (scf_stress_flag){

    if (ProExpn_VNA==0){
      double s3,s5;

      s3 = (double)F_VNA_flag*ECE[3]; 
      s5 = (double)F_dVHart_flag*ECE[5];

      Stress_Tensor[0] += s3 + s5;
      Stress_Tensor[4] += s3 + s5;
      Stress_Tensor[8] += s3 + s5;
    }
    else{
      double s5;

      s5 = (double)F_dVHart_flag*ECE[5];
      Stress_Tensor[0] += s5;
      Stress_Tensor[4] += s5;
      Stress_Tensor[8] += s5;
    }

    {
      int i,j;
      double tmp;

      /* symmetrization of stress tensor */

      for (i=0; i<3; i++){
	for (j=(i+1); j<3; j++){
	  tmp = 0.5*(Stress_Tensor[3*i+j]+Stress_Tensor[3*j+i]);
	  Stress_Tensor[3*i+j] = tmp;
	  Stress_Tensor[3*j+i] = tmp;
	}
      }

      /* show the stress tensor including all the contributions */

      if (MYID_MPI_COMM_WORLD==Host_ID && 0<level_stdout){

	printf("\n*******************************************************\n"); fflush(stdout);
	printf("               Stress tensor (Hartree/bohr^3)            \n"); fflush(stdout);
	printf("*******************************************************\n\n"); fflush(stdout);

	for (i=0; i<3; i++){
	  for (j=0; j<3; j++){
	    printf("%17.8f ", Stress_Tensor[3*i+j]/Cell_Volume);
	  }
	  printf("\n");fflush(stdout);
	}
      }
    }
  }

  /*********************************************************
                decomposition of total energy 
  *********************************************************/

  if (Energy_Decomposition_flag==1){

    Energy_Decomposition(energies, ECE, electronic);
  }

  /*********************************************************
           decomposition of total energy by CWFs
  *********************************************************/

  if (CWF_Calc==1 && CWF_Energy_Decomposition==1){

    Energy_Decomposition_CWF(scratch, cwf_values, ECE);
  }

  /* The invocation owns the pair map and all helper scratch. */
}








double Calc_Ekin(const OpenMXEnergyWorkspace *workspace, const OpenMXElectronicViews *electronic)
{
  const int OneD_Nloop=workspace->pair_count;
  const int *OneD2Mc_AN=openmx_solver_int_matrix_row(&workspace->pair_indices,0);
  const int *OneD2h_AN=openmx_solver_int_matrix_row(&workspace->pair_indices,1);

  double *****CntH0=electronic->CntH0;
  double ******DM=electronic->DM;
  double *****H0=electronic->H0;

  /****************************************************
          calculate the kinetic energy, Ekin
  ****************************************************/

  int i,j,spin,spinmax;
  int Mc_AN,Gc_AN,Cwan,Rn,h_AN,Gh_AN,Hwan;
  double My_Ekin,Ekin,Zc,Zh,dum,dum2;
  int numprocs,myid;
  double Stime_atom, Etime_atom;
  double *My_Ekin_threads;
  /* for OpenMP */
  int OMPID,Nthrds,Nthrds0,Nprocs,Nloop;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /****************************************************
      conventional calculation of kinetic energy
  ****************************************************/

  /* get Nthrds0 */  
  /* Dynamic teams may grow after a separate probe region. */
  Nthrds0 = omp_get_max_threads();

  /* allocation of array */
  My_Ekin_threads = OpenMX_Solver_OverwriteReal(workspace->owner,"total-energy:thread-sums",(size_t)Nthrds0);
  for (Nloop=0; Nloop<Nthrds0; Nloop++) My_Ekin_threads[Nloop] = 0.0;

  if (SpinP_switch==0 || SpinP_switch==1){

    for (spin=0; spin<=SpinP_switch; spin++){

#pragma omp parallel shared(SpinP_switch,time_per_atom,spin,CntH0,H0,DM,My_Ekin_threads,Spe_Total_CNO,Cnt_switch,natn,WhatSpecies,M2G,OneD2h_AN,OneD2Mc_AN,OneD_Nloop) private(OMPID,Nthrds,Nprocs,Nloop,Stime_atom,Mc_AN,h_AN,Gc_AN,Cwan,Gh_AN,Hwan,i,j,Etime_atom)
      {

	/* get info. on OpenMP */ 

	OMPID = omp_get_thread_num();
	Nthrds = omp_get_num_threads();
	Nprocs = omp_get_num_procs();

	/* one-dimensionalized loop */

	for (Nloop=OMPID*OneD_Nloop/Nthrds; Nloop<(OMPID+1)*OneD_Nloop/Nthrds; Nloop++){

	  dtime(&Stime_atom);

	  /* get Mc_AN and h_AN */

	  Mc_AN = OneD2Mc_AN[Nloop];
	  h_AN  = OneD2h_AN[Nloop];

	  /* set data on Mc_AN */

	  Gc_AN = M2G[Mc_AN];
	  Cwan = WhatSpecies[Gc_AN];

	  /* set data on h_AN */

	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];

	  if (Cnt_switch==0){
	    for (i=0; i<Spe_Total_CNO[Cwan]; i++){
	      for (j=0; j<Spe_Total_CNO[Hwan]; j++){
		My_Ekin_threads[OMPID] += DM[0][spin][Mc_AN][h_AN][i][j]*H0[0][Mc_AN][h_AN][i][j];
	      }
	    }
	  }

	  else{

	    for (i=0; i<Spe_Total_CNO[Cwan]; i++){
	      for (j=0; j<Spe_Total_CNO[Hwan]; j++){
		My_Ekin_threads[OMPID] += DM[0][spin][Mc_AN][h_AN][i][j]*CntH0[0][Mc_AN][h_AN][i][j];
	      }
	    }
	  }

	  dtime(&Etime_atom);
#pragma omp atomic update
	  time_per_atom[Gc_AN] += Etime_atom - Stime_atom;
	}

        if (SpinP_switch==0) My_Ekin_threads[OMPID] = 2.0*My_Ekin_threads[OMPID];

      } /* #pragma omp parallel */
    } /* spin */ 

  }
  else if (SpinP_switch==3){

#pragma omp parallel shared(time_per_atom,H0,DM,My_Ekin_threads,Spe_Total_CNO,natn,WhatSpecies,M2G,OneD2h_AN,OneD2Mc_AN,OneD_Nloop) private(OMPID,Nthrds,Nprocs,Nloop,Stime_atom,Mc_AN,Gc_AN,Cwan,h_AN,Gh_AN,Hwan,i,j,Etime_atom)
    {

      /* get info. on OpenMP */ 

      OMPID = omp_get_thread_num();
      Nthrds = omp_get_num_threads();
      Nprocs = omp_get_num_procs();

      /* one-dimensionalized loop */

      for (Nloop=OMPID*OneD_Nloop/Nthrds; Nloop<(OMPID+1)*OneD_Nloop/Nthrds; Nloop++){

	dtime(&Stime_atom);

	/* get Mc_AN and h_AN */

	Mc_AN = OneD2Mc_AN[Nloop];
	h_AN  = OneD2h_AN[Nloop];

	/* set data on Mc_AN */

	Gc_AN = M2G[Mc_AN];
	Cwan = WhatSpecies[Gc_AN];

	/* set data on h_AN */

	Gh_AN = natn[Gc_AN][h_AN];
	Hwan = WhatSpecies[Gh_AN];

	for (i=0; i<Spe_Total_CNO[Cwan]; i++){
	  for (j=0; j<Spe_Total_CNO[Hwan]; j++){
	    My_Ekin_threads[OMPID] += (DM[0][0][Mc_AN][h_AN][i][j] + DM[0][1][Mc_AN][h_AN][i][j])*H0[0][Mc_AN][h_AN][i][j];
	  }
	}

	dtime(&Etime_atom);
#pragma omp atomic update
	time_per_atom[Gc_AN] += Etime_atom - Stime_atom;
      }

    } /* #pragma omp parallel */
  }

  /* sum of My_Ekin_threads */
  My_Ekin = 0.0;
  for (Nloop=0; Nloop<Nthrds0; Nloop++){
    My_Ekin += My_Ekin_threads[Nloop];
  }

  /* sum of My_Ekin */
  MPI_Allreduce(&My_Ekin, &Ekin, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

  /* The enclosing invocation owns the partial sums. */

  /****************************************************
                      return Ekin
  ****************************************************/

  return Ekin;  
}





double Calc_Ena(const OpenMXEnergyWorkspace *workspace, const OpenMXVnaMatrixViews *vna, const OpenMXElectronicViews *electronic, const OpenMXLocalBasisViews *local_basis)
{
  const int OneD_Nloop=workspace->pair_count;
  const int *OneD2Mc_AN=openmx_solver_int_matrix_row(&workspace->pair_indices,0);
  const int *OneD2h_AN=openmx_solver_int_matrix_row(&workspace->pair_indices,1);

  double *****CntH0=electronic->CntH0;
  double ******DM=electronic->DM;

  /****************************************************
     calculate the neutral atom potential energy, Ena
  ****************************************************/

  int i,j,spin,spinmax;
  int Mc_AN,Gc_AN,Cwan,Rn,h_AN,Gh_AN,Hwan;
  double My_Ena,Ena,Zc,Zh,dum,dum2;
  int numprocs,myid;
  double Stime_atom, Etime_atom;
  double *My_Ena_threads;
  /* for OpenMP */
  int OMPID,Nthrds,Nthrds0,Nprocs,Nloop;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /**********************************************************
   conventional calculation of neutral atom potential energy
  **********************************************************/

  /* get Nthrds0 */  
  /* Dynamic teams may grow after a separate probe region. */
  Nthrds0 = omp_get_max_threads();

  /* allocation of array */
  My_Ena_threads = OpenMX_Solver_OverwriteReal(workspace->owner,"total-energy:thread-sums",(size_t)Nthrds0);
  for (Nloop=0; Nloop<Nthrds0; Nloop++) My_Ena_threads[Nloop] = 0.0;

  if (SpinP_switch==0 || SpinP_switch==1){

    if (Cnt_switch==1){
      /* temporaly, we borrow the CntH0 matrix */
      Cont_Matrix0(vna->HVNA,CntH0[0], local_basis);
    }

    for (spin=0; spin<=SpinP_switch; spin++){

#pragma omp parallel shared(spin,SpinP_switch,time_per_atom,CntH0,vna,DM,My_Ena_threads,Spe_Total_CNO,Cnt_switch,natn,WhatSpecies,M2G,OneD2h_AN,OneD2Mc_AN,OneD_Nloop) private(OMPID,Nthrds,Nprocs,Nloop,Stime_atom,Mc_AN,h_AN,Gc_AN,Cwan,Gh_AN,Hwan,i,j,Etime_atom)
      {

	/* get info. on OpenMP */ 

	OMPID = omp_get_thread_num();
	Nthrds = omp_get_num_threads();
	Nprocs = omp_get_num_procs();

	/* one-dimensionalized loop */

	for (Nloop=OMPID*OneD_Nloop/Nthrds; Nloop<(OMPID+1)*OneD_Nloop/Nthrds; Nloop++){

	  dtime(&Stime_atom);

	  /* get Mc_AN and h_AN */

	  Mc_AN = OneD2Mc_AN[Nloop];
	  h_AN  = OneD2h_AN[Nloop];

	  /* set data on Mc_AN */

	  Gc_AN = M2G[Mc_AN];
	  Cwan = WhatSpecies[Gc_AN];

	  /* set data on h_AN */

	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];

	  if (Cnt_switch==0){
	    for (i=0; i<Spe_Total_CNO[Cwan]; i++){
	      for (j=0; j<Spe_Total_CNO[Hwan]; j++){
		My_Ena_threads[OMPID] += DM[0][spin][Mc_AN][h_AN][i][j]*openmx_double_block_row(&vna->potential,0,Mc_AN,h_AN,i)[j];
	      }
	    }
	  }

	  else {

	    for (i=0; i<Spe_Total_CNO[Cwan]; i++){
	      for (j=0; j<Spe_Total_CNO[Hwan]; j++){
		My_Ena_threads[OMPID] += DM[0][spin][Mc_AN][h_AN][i][j]*CntH0[0][Mc_AN][h_AN][i][j];
	      }
	    }
	  }

	  dtime(&Etime_atom);
#pragma omp atomic update
	  time_per_atom[Gc_AN] += Etime_atom - Stime_atom;
	}

        if (SpinP_switch==0) My_Ena_threads[OMPID] = 2.0*My_Ena_threads[OMPID];

      } /* #pragma omp parallel */
    } /* spin */

  }
  else if (SpinP_switch==3){

#pragma omp parallel shared(time_per_atom,vna,DM,Spe_Total_CNO,natn,WhatSpecies,M2G,OneD2h_AN,OneD2Mc_AN,OneD_Nloop) private(OMPID,Nthrds,Nprocs,Nloop,Stime_atom,Mc_AN,Gc_AN,h_AN,Cwan,Gh_AN,Hwan,i,j,Etime_atom)
    {

      /* get info. on OpenMP */ 

      OMPID  = omp_get_thread_num();
      Nthrds = omp_get_num_threads();
      Nprocs = omp_get_num_procs();

      /* one-dimensionalized loop */

      for (Nloop=OMPID*OneD_Nloop/Nthrds; Nloop<(OMPID+1)*OneD_Nloop/Nthrds; Nloop++){

	dtime(&Stime_atom);

	/* get Mc_AN and h_AN */

	Mc_AN = OneD2Mc_AN[Nloop];
	h_AN  = OneD2h_AN[Nloop];

	/* set data on Mc_AN */

	Gc_AN = M2G[Mc_AN];
	Cwan = WhatSpecies[Gc_AN];

	/* set data on h_AN */

	Gh_AN = natn[Gc_AN][h_AN];
	Hwan = WhatSpecies[Gh_AN];

	for (i=0; i<Spe_Total_CNO[Cwan]; i++){
	  for (j=0; j<Spe_Total_CNO[Hwan]; j++){
	    My_Ena_threads[OMPID] += (DM[0][0][Mc_AN][h_AN][i][j]+DM[0][1][Mc_AN][h_AN][i][j])*openmx_double_block_row(&vna->potential,0,Mc_AN,h_AN,i)[j];
	  }
	}

	dtime(&Etime_atom);
#pragma omp atomic update
	time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

      } /* Nloop */
    } /* #pragma omp parallel */ 
  } /* else if (SpinP_switch==3) */


  /* sum of My_Ena_threads */
  My_Ena = 0.0;
  for (Nloop=0; Nloop<Nthrds0; Nloop++){
    My_Ena += My_Ena_threads[Nloop];
  }

  /* sum of My_Ena */
  MPI_Allreduce(&My_Ena, &Ena, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

  /* The enclosing invocation owns the partial sums. */

  /****************************************************
                      return Ena
  ****************************************************/

  return Ena;  
}





double Calc_Enl(const OpenMXEnergyWorkspace *workspace, const OpenMXElectronicViews *electronic, const OpenMXLocalBasisViews *local_basis)
{
  const int OneD_Nloop=workspace->pair_count;
  const int *OneD2Mc_AN=openmx_solver_int_matrix_row(&workspace->pair_indices,0);
  const int *OneD2h_AN=openmx_solver_int_matrix_row(&workspace->pair_indices,1);

  double *****CntH0=electronic->CntH0;
  double ******DM=electronic->DM;
  double *****HNL=electronic->HNL;
  double ******iDM=electronic->iDM;
  double *****iHNL0=electronic->iHNL0;

  /****************************************************
     calculate the non-local pseudo potential energy
  ****************************************************/

  int i,j,spin,spinmax;
  int Mc_AN,Gc_AN,Cwan,Rn,h_AN,Gh_AN,Hwan;
  double My_Enl,Enl,Zc,Zh,dum,dum2;
  int numprocs,myid;
  double Stime_atom, Etime_atom;
  double *My_Enl_threads;
  /* for OpenMP */
  int OMPID,Nthrds,Nthrds0,Nprocs,Nloop;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /*******************************************************************
   conventional calculation of the non-local pseudo potential energy
  *******************************************************************/

  /* get Nthrds0 */  
  /* Dynamic teams may grow after a separate probe region. */
  Nthrds0 = omp_get_max_threads();

  /* allocation of array */
  My_Enl_threads = OpenMX_Solver_OverwriteReal(workspace->owner,"total-energy:thread-sums",(size_t)Nthrds0);
  for (Nloop=0; Nloop<Nthrds0; Nloop++) My_Enl_threads[Nloop] = 0.0;

  if (SpinP_switch==0 || SpinP_switch==1){

    for (spin=0; spin<=SpinP_switch; spin++){

      if (Cnt_switch==1){
        /* temporaly, borrow the CntH0 matrix */
        Cont_Matrix0(HNL[spin],CntH0[0], local_basis);
      }

#pragma omp parallel shared(spin,SpinP_switch,time_per_atom,CntH0,HNL,DM,My_Enl_threads,Spe_Total_CNO,Cnt_switch,natn,WhatSpecies,M2G,OneD2h_AN,OneD2Mc_AN,OneD_Nloop) private(OMPID,Nthrds,Nprocs,Nloop,Stime_atom,Mc_AN,h_AN,Gc_AN,Cwan,Gh_AN,Hwan,Etime_atom,i,j)
      {

	/* get info. on OpenMP */ 

	OMPID = omp_get_thread_num();
	Nthrds = omp_get_num_threads();
	Nprocs = omp_get_num_procs();

	/* one-dimensionalized loop */

	for (Nloop=OMPID*OneD_Nloop/Nthrds; Nloop<(OMPID+1)*OneD_Nloop/Nthrds; Nloop++){

	  dtime(&Stime_atom);

	  /* get Mc_AN and h_AN */

	  Mc_AN = OneD2Mc_AN[Nloop];
	  h_AN  = OneD2h_AN[Nloop];

	  /* set data on Mc_AN */

	  Gc_AN = M2G[Mc_AN];
	  Cwan = WhatSpecies[Gc_AN];

	  /* set data on h_AN */

	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];

	  if (Cnt_switch==0){
	    for (i=0; i<Spe_Total_CNO[Cwan]; i++){
	      for (j=0; j<Spe_Total_CNO[Hwan]; j++){
		My_Enl_threads[OMPID] += DM[0][spin][Mc_AN][h_AN][i][j]*HNL[spin][Mc_AN][h_AN][i][j];
	      }
	    }
	  }

	  else {
	    for (i=0; i<Spe_Total_CNO[Cwan]; i++){
	      for (j=0; j<Spe_Total_CNO[Hwan]; j++){
		My_Enl_threads[OMPID] += DM[0][spin][Mc_AN][h_AN][i][j]*CntH0[0][Mc_AN][h_AN][i][j];
	      }
	    }
	  }

	  dtime(&Etime_atom);
#pragma omp atomic update
	  time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

	} /* Nloop */

        if (SpinP_switch==0) My_Enl_threads[OMPID] = 2.0*My_Enl_threads[OMPID];

      } /* #pragma omp parallel */
    } /* spin */

  }
  else if (SpinP_switch==3){

#pragma omp parallel shared(time_per_atom,iHNL0,HNL,iDM,DM,My_Enl_threads,Spe_Total_CNO,natn,WhatSpecies,M2G,OneD2h_AN,OneD2Mc_AN,OneD_Nloop) private(OMPID,Nthrds,Nprocs,Nloop,Stime_atom,Etime_atom,Mc_AN,h_AN,Gc_AN,Cwan,Gh_AN,Hwan,i,j)
    {

      /* get info. on OpenMP */ 

      OMPID  = omp_get_thread_num();
      Nthrds = omp_get_num_threads();
      Nprocs = omp_get_num_procs();

      /* one-dimensionalized loop */

      for (Nloop=OMPID*OneD_Nloop/Nthrds; Nloop<(OMPID+1)*OneD_Nloop/Nthrds; Nloop++){

	dtime(&Stime_atom);

	/* get Mc_AN and h_AN */

	Mc_AN = OneD2Mc_AN[Nloop];
	h_AN  = OneD2h_AN[Nloop];

	/* set data on Mc_AN */

	Gc_AN = M2G[Mc_AN];
	Cwan = WhatSpecies[Gc_AN];

	/* set data on h_AN */

	Gh_AN = natn[Gc_AN][h_AN];
	Hwan = WhatSpecies[Gh_AN];

	for (i=0; i<Spe_Total_CNO[Cwan]; i++){
	  for (j=0; j<Spe_Total_CNO[Hwan]; j++){

	    My_Enl_threads[OMPID] += 
	         DM[0][0][Mc_AN][h_AN][i][j]*  HNL[0][Mc_AN][h_AN][i][j]
	      - iDM[0][0][Mc_AN][h_AN][i][j]*iHNL0[0][Mc_AN][h_AN][i][j]
	      +  DM[0][1][Mc_AN][h_AN][i][j]*  HNL[1][Mc_AN][h_AN][i][j]
	      - iDM[0][1][Mc_AN][h_AN][i][j]*iHNL0[1][Mc_AN][h_AN][i][j]
	   + 2.0*DM[0][2][Mc_AN][h_AN][i][j]*  HNL[2][Mc_AN][h_AN][i][j] 
	   - 2.0*DM[0][3][Mc_AN][h_AN][i][j]*iHNL0[2][Mc_AN][h_AN][i][j];
 
	  }
	}

	dtime(&Etime_atom);
#pragma omp atomic update
	time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

      } /* Nloop */
    } /* #pragma omp parallel */
  }

  /* sum of My_Enl_threads */
  My_Enl = 0.0;
  for (Nloop=0; Nloop<Nthrds0; Nloop++){
    My_Enl += My_Enl_threads[Nloop];
  }

  /* sum of My_Enl */
  MPI_Allreduce(&My_Enl, &Enl, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

  /* The enclosing invocation owns the partial sums. */

  /****************************************************
                      return Enl
  ****************************************************/

  return Enl;  
}


double Calc_ECH(const OpenMXEnergyWorkspace *workspace, const OpenMXElectronicViews *electronic)
{
  const int OneD_Nloop=workspace->pair_count;
  const int *OneD2Mc_AN=openmx_solver_int_matrix_row(&workspace->pair_indices,0);
  const int *OneD2h_AN=openmx_solver_int_matrix_row(&workspace->pair_indices,1);

  double ******DM=electronic->DM;
  double *****HCH=electronic->HCH;
  double ******iDM=electronic->iDM;
  double *****iHCH=electronic->iHCH;

  /****************************************************
    calculate the penalty term to create a core-hole 
  ****************************************************/

  int i,j,spin,spinmax;
  int Mc_AN,Gc_AN,Cwan,Rn,h_AN,Gh_AN,Hwan;
  double My_ECH,ECH,Zc,Zh,dum,dum2;
  int numprocs,myid;
  double Stime_atom, Etime_atom;
  double *My_ECH_threads;
  /* for OpenMP */
  int OMPID,Nthrds,Nthrds0,Nprocs,Nloop;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /*******************************************************************
            conventional calculation of the penalty term 
  *******************************************************************/

  /* get Nthrds0 */  
  /* Dynamic teams may grow after a separate probe region. */
  Nthrds0 = omp_get_max_threads();

  /* allocation of array */
  My_ECH_threads = OpenMX_Solver_OverwriteReal(workspace->owner,"total-energy:thread-sums",(size_t)Nthrds0);
  for (Nloop=0; Nloop<Nthrds0; Nloop++) My_ECH_threads[Nloop] = 0.0;

  if (SpinP_switch==0 || SpinP_switch==1){

    for (spin=0; spin<=SpinP_switch; spin++){

#pragma omp parallel shared(spin,SpinP_switch,time_per_atom,HCH,DM,My_ECH_threads,Spe_Total_CNO,Cnt_switch,natn,WhatSpecies,M2G,OneD2h_AN,OneD2Mc_AN,OneD_Nloop) private(OMPID,Nthrds,Nprocs,Nloop,Stime_atom,Mc_AN,h_AN,Gc_AN,Cwan,Gh_AN,Hwan,Etime_atom,i,j)
      {

	/* get info. on OpenMP */ 

	OMPID = omp_get_thread_num();
	Nthrds = omp_get_num_threads();
	Nprocs = omp_get_num_procs();

	/* one-dimensionalized loop */

	for (Nloop=OMPID*OneD_Nloop/Nthrds; Nloop<(OMPID+1)*OneD_Nloop/Nthrds; Nloop++){

	  dtime(&Stime_atom);

	  /* get Mc_AN and h_AN */

	  Mc_AN = OneD2Mc_AN[Nloop];
	  h_AN  = OneD2h_AN[Nloop];

	  /* set data on Mc_AN */

	  Gc_AN = M2G[Mc_AN];
	  Cwan = WhatSpecies[Gc_AN];

	  /* set data on h_AN */

	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];

	  for (i=0; i<Spe_Total_CNO[Cwan]; i++){
	    for (j=0; j<Spe_Total_CNO[Hwan]; j++){
	      My_ECH_threads[OMPID] += DM[0][spin][Mc_AN][h_AN][i][j]*HCH[spin][Mc_AN][h_AN][i][j];
	    }
	  }

	  dtime(&Etime_atom);
#pragma omp atomic update
	  time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

	} /* Nloop */

        if (SpinP_switch==0) My_ECH_threads[OMPID] = 2.0*My_ECH_threads[OMPID];

      } /* #pragma omp parallel */
    } /* spin */

  }
  else if (SpinP_switch==3){

#pragma omp parallel shared(time_per_atom,iHCH,HCH,iDM,DM,My_ECH_threads,Spe_Total_CNO,natn,WhatSpecies,M2G,OneD2h_AN,OneD2Mc_AN,OneD_Nloop) private(OMPID,Nthrds,Nprocs,Nloop,Stime_atom,Etime_atom,Mc_AN,h_AN,Gc_AN,Cwan,Gh_AN,Hwan,i,j)
    {

      /* get info. on OpenMP */ 

      OMPID  = omp_get_thread_num();
      Nthrds = omp_get_num_threads();
      Nprocs = omp_get_num_procs();

      /* one-dimensionalized loop */

      for (Nloop=OMPID*OneD_Nloop/Nthrds; Nloop<(OMPID+1)*OneD_Nloop/Nthrds; Nloop++){

	dtime(&Stime_atom);

	/* get Mc_AN and h_AN */

	Mc_AN = OneD2Mc_AN[Nloop];
	h_AN  = OneD2h_AN[Nloop];

	/* set data on Mc_AN */

	Gc_AN = M2G[Mc_AN];
	Cwan = WhatSpecies[Gc_AN];

	/* set data on h_AN */

	Gh_AN = natn[Gc_AN][h_AN];
	Hwan = WhatSpecies[Gh_AN];

	for (i=0; i<Spe_Total_CNO[Cwan]; i++){
	  for (j=0; j<Spe_Total_CNO[Hwan]; j++){

	    My_ECH_threads[OMPID] += 
	         DM[0][0][Mc_AN][h_AN][i][j]* HCH[0][Mc_AN][h_AN][i][j]
	      - iDM[0][0][Mc_AN][h_AN][i][j]*iHCH[0][Mc_AN][h_AN][i][j]
	      +  DM[0][1][Mc_AN][h_AN][i][j]* HCH[1][Mc_AN][h_AN][i][j]
	      - iDM[0][1][Mc_AN][h_AN][i][j]*iHCH[1][Mc_AN][h_AN][i][j]
	   + 2.0*DM[0][2][Mc_AN][h_AN][i][j]* HCH[2][Mc_AN][h_AN][i][j] 
	   - 2.0*DM[0][3][Mc_AN][h_AN][i][j]*iHCH[2][Mc_AN][h_AN][i][j];

	  }
	}

	dtime(&Etime_atom);
#pragma omp atomic update
	time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

      } /* Nloop */
    } /* #pragma omp parallel */
  }

  /* sum of My_ECH_threads */
  My_ECH = 0.0;
  for (Nloop=0; Nloop<Nthrds0; Nloop++){
    My_ECH += My_ECH_threads[Nloop];
  }

  /* sum of My_ECH */
  MPI_Allreduce(&My_ECH, &ECH, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

  /* The enclosing invocation owns the partial sums. */

  /****************************************************
                      return ECH
  ****************************************************/

  return ECH;  
}





double Calc_Ecore(OpenMXSolverInvocation *scratch)
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

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  if (MYID_MPI_COMM_WORLD==Host_ID && 0<level_stdout){
    printf("  Force calculation #6%s\n",
#if OPENMX_GPU_HIP
           TotalEnergyUseOpenMP() ? " (GPU-accelerated)" : "");fflush(stdout);
#else
           TotalEnergyUseOpenACC() ? " (GPU-accelerated)" : "");fflush(stdout);
#endif
  }

  /* get Nthrds0 */  
  /* Dynamic teams may grow after a separate probe region. */
  Nthrds0 = omp_get_max_threads();

  /* allocation of array */
  My_Ecore_threads = OpenMX_Solver_OverwriteReal(scratch,"total-energy:thread-sums",(size_t)Nthrds0);
  for (Nloop=0; Nloop<Nthrds0; Nloop++) My_Ecore_threads[Nloop] = 0.0;

#pragma omp parallel shared(level_stdout,time_per_atom,atv,Gxyz,Dis,ncn,natn,FNAN,Spe_Core_Charge,WhatSpecies,M2G,Matomnum,My_Ecore_threads,DecEscc,Energy_Decomposition_flag,SpinP_switch,Spe_MaxL_Basis,Spe_Num_Basis) private(OMPID,Nthrds,Nprocs,Mc_AN,Stime_atom,Gc_AN,Cwan,Zc,dEx,dEy,dEz,h_AN,Gh_AN,Rn,Hwan,Zh,r,lx,ly,lz,dum,dum2,Etime_atom,TmpEcore)
  {

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

      } /* h_AN */

      /****************************************************
                        #6 of force
         Contribution from the core-core repulsions
      ****************************************************/

      My_Ecore_threads[OMPID] += 0.50*TmpEcore;
      Gxyz[Gc_AN][17] += dEx;
      Gxyz[Gc_AN][18] += dEy;
      Gxyz[Gc_AN][19] += dEz;

      if (Energy_Decomposition_flag==1){

        DecEscc[0][Mc_AN][0] = 0.25*TmpEcore;
        DecEscc[1][Mc_AN][0] = 0.25*TmpEcore;
      } 

      if (2<=level_stdout){
	printf("<Total_Ene>  force(6) myid=%2d  Mc_AN=%2d Gc_AN=%2d  %15.12f %15.12f %15.12f\n",
	       myid,Mc_AN,Gc_AN,dEx,dEy,dEz);fflush(stdout);
      }

      dtime(&Etime_atom);
      time_per_atom[Gc_AN] += Etime_atom - Stime_atom;
    }

  } /* #pragma omp parallel */

  My_Ecore = 0.0;
  for (Nloop=0; Nloop<Nthrds0; Nloop++){
    My_Ecore += My_Ecore_threads[Nloop];
  }

  MPI_Allreduce(&My_Ecore, &Ecore, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

  /* The invocation retains the reusable partial sums. */

  return Ecore;  
}








double Calc_EH0(const OpenMXEnergyWorkspace *workspace, OpenMXCoreContext *core, const OpenMXAtomicViews *atomic, int MD_iter)
{
  const OpenMXGaussRuleData radial=OpenMX_NeutralQuadrature(core,CoarseGL_Mesh,mpi_comm_level1);
  OpenMXNeutralGridTables tables;
  char grid_error[256]={0};
  OpenMXNeutralGridMetadataView metadata;
  if (!openmx_core_neutral_grid_metadata_get(core,&metadata,grid_error,sizeof(grid_error)) ||
      !metadata.counts || !metadata.spacing || metadata.species!=(size_t)SpeciesNum) {
    fprintf(stderr,"neutral grid metadata unavailable or shape mismatch: %s\n",grid_error);
    MPI_Abort(mpi_comm_level1,1);
    return 0.0;
  }
  if (MD_iter!=1) {
      if (!openmx_core_neutral_grid_get(core,&tables,grid_error,sizeof(grid_error)) ||
          !tables.values || tables.extent[1]!=(size_t)SpeciesNum ||
          tables.extent[2]!=(size_t)Max_TGN_EH0) {
        fprintf(stderr,"neutral grid view unavailable or shape mismatch: %s\n",grid_error);
        MPI_Abort(mpi_comm_level1,1);
        return 0.0;
      }
  }

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
  double *My_EH0_threads;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /****************************************************
     allocation of arrays:

    double Fx[Matomnum+1];
    double Fy[Matomnum+1];
    doubel Fz[Matomnum+1];
  ****************************************************/

  const OpenMXSolverRealTensor forces=OpenMX_Solver_OverwriteView(workspace->owner,
      "total-energy:eh0-forces",3,OpenMX_Solver_SimdColumns((size_t)Matomnum+1));
  Fx=openmx_solver_real_matrix_row(&forces,0);
  Fy=openmx_solver_real_matrix_row(&forces,1);
  Fz=openmx_solver_real_matrix_row(&forces,2);

  /****************************************************
             Set of atomic density on grids
  ****************************************************/

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
    
    /* allocation of arrays GridX,Y,Z_EH0, Arho_EH0, and Wt_EH0 */

    Max_TGN_EH0 += 10; 

    Allocate_Arrays_with_context(4,core);
    if (!openmx_core_neutral_grid_get(core,&tables,grid_error,sizeof(grid_error)) ||
        !tables.values || tables.extent[1]!=(size_t)SpeciesNum ||
        tables.extent[2]!=(size_t)Max_TGN_EH0) {
      fprintf(stderr,"neutral grid view unavailable or shape mismatch: %s\n",grid_error);
      MPI_Abort(mpi_comm_level1,1);
      return 0.0;
    }

    /* calculate GridX,Y,Z_EH0 and Wt_EH0 */

#pragma omp parallel shared(workspace,atomic,Max_Nd,level_stdout,metadata,tables,Scale_Grid_Ecut,SpeciesNum,radial) private(OMPID,Nthrds,Nprocs,wan,bc,dx,Nd,gnum,n1,n2,n3,x,y,z,tmp0,r1,rho0,g0,Sx,Dx,xx)
    {

      int l,p;

      /* allocation of arrays g0 */

      g0 = OpenMX_Solver_OverwriteReal(workspace->owner,"total-energy:eh0-grid",(size_t)Max_Nd);

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
	metadata.spacing[wan] = dx;

	const OpenMXNeutralQuadratureView quadrature={Nd,CoarseGL_Mesh,bc,dx,PI,
            radial.nodes,radial.weights};
        gnum=OpenMX_Neutral_GridFill(atomic,wan,&quadrature,g0,
            openmx_core_neutral_grid_row(&tables,0,wan),
            openmx_core_neutral_grid_row(&tables,1,wan),
            openmx_core_neutral_grid_row(&tables,2,wan),
            openmx_core_neutral_grid_row(&tables,3,wan),
            openmx_core_neutral_grid_row(&tables,4,wan));

        metadata.counts[wan] = gnum;

	if (2<=level_stdout){
	  printf("<Calc_EH0> B spe=%2d 1D-grids=%2d 3D-grids=%2d\n",wan,Nd,gnum);fflush(stdout);
	}

      } /* wan */

      /* free */
      /* g0 belongs to this invocation and OpenMP worker. */

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
	EH0_TwoCenter_at_Cutoff(workspace->owner,&tables,&metadata, atomic, wan1, wan2, EH0ij);

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

  /****************************************************
                -1/2\int n^a(r) V^a_H dr
  ****************************************************/

  dtime(&stime);

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Fx[Mc_AN] = 0.0;
    Fy[Mc_AN] = 0.0;
    Fz[Mc_AN] = 0.0;
  }

  My_EH0 = 0.0;

  int eh0_device = scf_eigen_lib_flag == GPUSOLVER;
#ifdef OPENMX_HOST_SIMD_H
  if (eh0_device)
    eh0_device = !TotalEnergy_StageOnHost("Total_Energy EH0",TotalEnergy_EH0Bytes(2*(size_t)workspace->pair_count));
#endif
  if (eh0_device){
    int pair_count,pair_index,pair_has_d;
    int Gc2_AN,h2_AN,Gh2_AN,Rn;
    int *pair_ban,*pair_wan2,*pair_has_deriv;
    double gpu_eh0_time,atom_weight;
    double *pair_dis,*pair_dirx,*pair_diry,*pair_dirz;
    double *out0,*out1,*out2,*out3;

    OpenMXEnergyEH0Batch batch;
    OpenMX_Energy_EH0BatchPrepare(workspace->owner,2u*(size_t)workspace->pair_count,&batch);
    pair_count=batch.count;
    pair_ban=openmx_solver_int_matrix_row(&batch.indices,0);
    pair_wan2=openmx_solver_int_matrix_row(&batch.indices,1);
    pair_has_deriv=openmx_solver_int_matrix_row(&batch.indices,2);
    pair_dis=openmx_solver_real_matrix_row(&batch.values,0);
    pair_dirx=openmx_solver_real_matrix_row(&batch.values,1);
    pair_diry=openmx_solver_real_matrix_row(&batch.values,2);
    pair_dirz=openmx_solver_real_matrix_row(&batch.values,3);
    out0=openmx_solver_real_matrix_row(&batch.values,4);
    out1=openmx_solver_real_matrix_row(&batch.values,5);
    out2=openmx_solver_real_matrix_row(&batch.values,6);
    out3=openmx_solver_real_matrix_row(&batch.values,7);

    pair_index = 0;
    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];
      wan1 = WhatSpecies[Gc_AN];

      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

        Gh_AN = natn[Gc_AN][h_AN];
        wan2 = WhatSpecies[Gh_AN];

        pair_ban[pair_index] = atomic->species.species_index[wan1];
        pair_wan2[pair_index] = wan2;
        pair_has_d = (h_AN!=0);
        pair_has_deriv[pair_index] = pair_has_d;
        pair_dis[pair_index] = Dis[Gc_AN][h_AN];

        if (pair_has_d){
          Rn = ncn[Gc_AN][h_AN];
          r1 = Dis[Gc_AN][h_AN];
          if (r1<1.0e-10) r1 = 1.0e-10;
          x = Gxyz[Gc_AN][1] - (Gxyz[Gh_AN][1] + atv[Rn][1]);
          y = Gxyz[Gc_AN][2] - (Gxyz[Gh_AN][2] + atv[Rn][2]);
          z = Gxyz[Gc_AN][3] - (Gxyz[Gh_AN][3] + atv[Rn][3]);
          pair_dirx[pair_index] = x/r1;
          pair_diry[pair_index] = y/r1;
          pair_dirz[pair_index] = z/r1;
        }
        else{
          pair_dirx[pair_index] = 0.0;
          pair_diry[pair_index] = 0.0;
          pair_dirz[pair_index] = 0.0;
        }
        pair_index++;

        Gc2_AN = Gh_AN;
        h2_AN = RMI1[Mc_AN][h_AN][0];
        Gh2_AN = natn[Gc2_AN][h2_AN];
        wan1 = WhatSpecies[Gc2_AN];
        wan2 = WhatSpecies[Gh2_AN];

        pair_ban[pair_index] = atomic->species.species_index[wan1];
        pair_wan2[pair_index] = wan2;
        pair_has_d = (h2_AN!=0);
        pair_has_deriv[pair_index] = pair_has_d;
        pair_dis[pair_index] = Dis[Gc2_AN][h2_AN];

        if (pair_has_d){
          Rn = ncn[Gc2_AN][h2_AN];
          r1 = Dis[Gc2_AN][h2_AN];
          if (r1<1.0e-10) r1 = 1.0e-10;
          x = Gxyz[Gc2_AN][1] - (Gxyz[Gh2_AN][1] + atv[Rn][1]);
          y = Gxyz[Gc2_AN][2] - (Gxyz[Gh2_AN][2] + atv[Rn][2]);
          z = Gxyz[Gc2_AN][3] - (Gxyz[Gh2_AN][3] + atv[Rn][3]);
          pair_dirx[pair_index] = x/r1;
          pair_diry[pair_index] = y/r1;
          pair_dirz[pair_index] = z/r1;
        }
        else{
          pair_dirx[pair_index] = 0.0;
          pair_diry[pair_index] = 0.0;
          pair_dirz[pair_index] = 0.0;
        }
        pair_index++;

        wan1 = WhatSpecies[Gc_AN];
      }
    }

    dtime(&Stime_atom);
#if OPENMX_GPU_HIP
    TotalEnergy_EH0_TwoCenter_Batch_OpenMP(&tables,&metadata,atomic, pair_count, pair_ban, pair_wan2,
#else
    TotalEnergy_EH0_TwoCenter_Batch_OpenACC(&tables,&metadata,atomic, pair_count, pair_ban, pair_wan2,
#endif
                                            pair_has_deriv, pair_dis,
                                            pair_dirx, pair_diry, pair_dirz,
                                            out0, out1, out2, out3);
    dtime(&Etime_atom);
    gpu_eh0_time = Etime_atom - Stime_atom;

    pair_index = 0;
    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

      Gc_AN = M2G[Mc_AN];
      wan1 = WhatSpecies[Gc_AN];
      TmpEH0 = 0.0;

      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

        Gh_AN = natn[Gc_AN][h_AN];
        wan2 = WhatSpecies[Gh_AN];

        if (h_AN==0) factor = 1.0;
        else         factor = EH0_scaling[wan1][wan2];

        TmpEH0 -= 0.250*factor*out0[pair_index];
        Fx[Mc_AN] = Fx[Mc_AN] - 0.5*factor*out1[pair_index];
        Fy[Mc_AN] = Fy[Mc_AN] - 0.5*factor*out2[pair_index];
        Fz[Mc_AN] = Fz[Mc_AN] - 0.5*factor*out3[pair_index];
        pair_index++;

        if (h_AN==0) factor = 1.0;
        else         factor = EH0_scaling[wan2][wan1];

        TmpEH0 -= 0.250*factor*out0[pair_index];
        Fx[Mc_AN] = Fx[Mc_AN] + 0.5*factor*out1[pair_index];
        Fy[Mc_AN] = Fy[Mc_AN] + 0.5*factor*out2[pair_index];
        Fz[Mc_AN] = Fz[Mc_AN] + 0.5*factor*out3[pair_index];
        pair_index++;

      } /* h_AN */

      My_EH0 += TmpEH0;

      if (Energy_Decomposition_flag==1){
        DecEscc[0][Mc_AN][0] += 0.5*TmpEH0;
        DecEscc[1][Mc_AN][0] += 0.5*TmpEH0;
      }

      atom_weight = (double)(2*(FNAN[Gc_AN] + 1))/(double)pair_count;
      time_per_atom[Gc_AN] += gpu_eh0_time*atom_weight;
    }

    /* The invocation owns the packed batch. */
  }
  else{

    /* get Nthrds0 */
    /* Dynamic teams may grow after a separate probe region. */
    Nthrds0 = omp_get_max_threads();

    /* allocation of array */
    My_EH0_threads = OpenMX_Solver_OverwriteReal(workspace->owner,"total-energy:thread-sums",(size_t)Nthrds0);
    for (Nloop=0; Nloop<Nthrds0; Nloop++) My_EH0_threads[Nloop] = 0.0;

#pragma omp parallel shared(atomic,time_per_atom,RMI1,EH0_scaling,natn,FNAN,WhatSpecies,M2G,Matomnum,My_EH0_threads,DecEscc,Energy_Decomposition_flag,List_YOUSO,SpinP_switch) private(OMPID,Nthrds,Nprocs,Mc_AN,Stime_atom,Gc_AN,wan1,h_AN,Gh_AN,wan2,factor,Etime_atom,TmpEH0)
    {

      int l,p;
      double EH0ij[4];

      /* get info. on OpenMP */

      OMPID = omp_get_thread_num();
      Nthrds = omp_get_num_threads();
      Nprocs = omp_get_num_procs();

      for (Mc_AN=(OMPID*Matomnum/Nthrds+1); Mc_AN<((OMPID+1)*Matomnum/Nthrds+1); Mc_AN++){

        dtime(&Stime_atom);

        Gc_AN = M2G[Mc_AN];
        wan1 = WhatSpecies[Gc_AN];
        TmpEH0 = 0.0;

        for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

          Gh_AN = natn[Gc_AN][h_AN];
          wan2 = WhatSpecies[Gh_AN];

          if (h_AN==0) factor = 1.0;
          else         factor = EH0_scaling[wan1][wan2];

          EH0_TwoCenter(&tables,&metadata,atomic, Gc_AN, h_AN, EH0ij);
          TmpEH0 -= 0.250*factor*EH0ij[0];
          Fx[Mc_AN] = Fx[Mc_AN] - 0.5*factor*EH0ij[1];
          Fy[Mc_AN] = Fy[Mc_AN] - 0.5*factor*EH0ij[2];
          Fz[Mc_AN] = Fz[Mc_AN] - 0.5*factor*EH0ij[3];

          if (h_AN==0) factor = 1.0;
          else         factor = EH0_scaling[wan2][wan1];

          EH0_TwoCenter(&tables,&metadata,atomic, Gh_AN, RMI1[Mc_AN][h_AN][0], EH0ij);
          TmpEH0 -= 0.250*factor*EH0ij[0];
          Fx[Mc_AN] = Fx[Mc_AN] + 0.5*factor*EH0ij[1];
          Fy[Mc_AN] = Fy[Mc_AN] + 0.5*factor*EH0ij[2];
          Fz[Mc_AN] = Fz[Mc_AN] + 0.5*factor*EH0ij[3];

        } /* h_AN */

        My_EH0_threads[OMPID] += TmpEH0;

        if (Energy_Decomposition_flag==1){

          DecEscc[0][Mc_AN][0] += 0.5*TmpEH0;
          DecEscc[1][Mc_AN][0] += 0.5*TmpEH0;
        }

        dtime(&Etime_atom);
        time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

      } /* Mc_AN */

    } /* #pragma omp parallel */

    /* sum of My_EH0_threads */
    for (Nloop=0; Nloop<Nthrds0; Nloop++){
      My_EH0 += My_EH0_threads[Nloop];
    }

    /* The invocation retains the reusable partial sums. */
  }

  /* sum of My_EH0 */
  MPI_Allreduce(&My_EH0, &EH0, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

  dtime(&etime);
  if(myid==0 && measure_time){
    printf("Time for part2 of EH0=%18.5f\n",etime-stime);fflush(stdout);
  } 

  /*******************************************************
                      #7 of force
   contribution from the classical Coulomb energy between
   the neutral atomic charge and the neutral potential 
  *******************************************************/

  if (MYID_MPI_COMM_WORLD==Host_ID && 0<level_stdout){
    printf("  Force calculation #7%s\n",
#if OPENMX_GPU_HIP
           TotalEnergyUseOpenMP() ? " (GPU-accelerated)" : "");fflush(stdout);
#else
           TotalEnergyUseOpenACC() ? " (GPU-accelerated)" : "");fflush(stdout);
#endif
  }

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];

    if (2<=level_stdout){
      printf("<Total_Ene>  force(7) myid=%2d  Mc_AN=%2d Gc_AN=%2d  %15.12f %15.12f %15.12f\n",
              myid,Mc_AN,Gc_AN,Fx[Mc_AN],Fy[Mc_AN],Fz[Mc_AN]);fflush(stdout);
    }

    Gxyz[Gc_AN][17] += Fx[Mc_AN];
    Gxyz[Gc_AN][18] += Fy[Mc_AN];
    Gxyz[Gc_AN][19] += Fz[Mc_AN];
  }

  /****************************************************
   MPI, Gxyz[Gc_AN][17-19]
  ****************************************************/

  for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
    ID = G2ID[Gc_AN];
    MPI_Bcast(&Gxyz[Gc_AN][17], 3, MPI_DOUBLE, ID, mpi_comm_level1);
  }

  /* The invocation owns all three force rows. */

  /* return */

  return EH0;  
}


#pragma optimization_level 1
void Calc_Edc(double ECE[])
{
  /************************************************************

   This routine calculates two terms relavant to 
   the double counting terms, which are used to evaluate the 
   total energy in the contracted diagonalization method. 

   Based on the Harris functional, the total energy is given by 
   Utot = Uele + Udc,
   where 
   Udc = UH1
         - \int dr rho(r) delta VH(r) 
         Uxc0 + Uxc1
         - \sum_{\sigma} \int dr rho_{\sigma}(r) Vxc_{\sigma}(r) 
         + Ucore + UH0
         + other terms such as UvdW. 

   In this routine, 

         - \int dr rho(r) delta VH(r) 
   and 
         - \sum_{\sigma} \int dr rho_{\sigma}(r) Vxc_{\sigma}(r) 

   are calculated, and they are stored as 

     ECE[15] = EH1;
     ECE[16] = EXC[0];
     ECE[17] = EXC[1];

  ************************************************************/

  int i,spin,spinmax,XC_P_switch;
  int numS,numR,My_GNum,BN_AB;
  int n,n1,n2,n3,Ng1,Ng2,Ng3,j,k;
  int GNc,GRc,MNc;
  int GN,GNs,BN,DN,LN,N2D,n2D,N3[4];
  double EXC[2],EH1,sum,tot_den;
  double My_EXC[2],My_EH1;
  double My_EXC_VolumeTerm[2];
  double EXC_VolumeTerm[2];
  double My_Eef,Eef;
  double sum_charge,My_charge;
  int numprocs,myid,tag=999,ID,IDS,IDR;
  double Cxyz[4],gradxyz[4];
  double Stime_atom,Etime_atom;
  double time0,time1;
  double sden[2],tden,aden,pden[2];

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  if      (SpinP_switch==0) spinmax = 0;
  else if (SpinP_switch==1) spinmax = 1;
  else if (SpinP_switch==3) spinmax = 1;

  /****************************************************
   set Vxc_Grid
  ****************************************************/

  XC_P_switch = 1;

  Set_XC_Grid(2, XC_P_switch,XC_switch,
	      Density_Grid_D[0],Density_Grid_D[1],
	      Density_Grid_D[2],Density_Grid_D[3],
	      Vxc_Grid_D[0], Vxc_Grid_D[1],
	      Vxc_Grid_D[2], Vxc_Grid_D[3],
	      NULL,NULL);

  /* copy Vxc_Grid_D to Vxc_Grid_B */

  Ng1 = Max_Grid_Index_D[1] - Min_Grid_Index_D[1] + 1;
  Ng2 = Max_Grid_Index_D[2] - Min_Grid_Index_D[2] + 1;
  Ng3 = Max_Grid_Index_D[3] - Min_Grid_Index_D[3] + 1;

  for (n=0; n<Num_Rcv_Grid_B2D[myid]; n++){
    DN = Index_Rcv_Grid_B2D[myid][n];
    BN = Index_Snd_Grid_B2D[myid][n];

    i = DN/(Ng2*Ng3);
    j = (DN-i*Ng2*Ng3)/Ng3;
    k = DN - i*Ng2*Ng3 - j*Ng3; 

    if ( !(i<=1 || (Ng1-2)<=i || j<=1 || (Ng2-2)<=j || k<=1 || (Ng3-2)<=k)){
      for (spin=0; spin<=SpinP_switch; spin++){
	Vxc_Grid_B[spin][BN] = Vxc_Grid_D[spin][DN];
      }
    }
  }

  /****************************************************
    calculate contributions to the double counting terms 
    arising from EH1 and EXC
  ****************************************************/

  My_EH1 = 0.0;
  My_EXC[0] = 0.0;
  My_EXC[1] = 0.0;

  for (BN=0; BN<My_NumGridB_AB; BN++){

    sden[0] = Density_Grid_B[0][BN];
    sden[1] = Density_Grid_B[1][BN];
    tden = sden[0] + sden[1];

    /* The dc term for EH1 = -\int \delta n(r) \delta V_H dr */
    My_EH1 -= tden*dVHart_Grid_B[BN];

    /* -\sum_{\sigma} n_{\sigma}\v_{\sigma,xc} */

    for (spin=0; spin<=spinmax; spin++){
      My_EXC[spin] -= sden[spin]*Vxc_Grid_B[spin][BN];
    }

  } /* BN */

  /****************************************************
       multiplying GridVol and MPI communication
  ****************************************************/

  MPI_Barrier(mpi_comm_level1);

  /* The dc term EH1 = \int \delta n(r) \delta V_H dr */
  My_EH1 *= GridVol;
  My_EXC[0] *= GridVol;
  My_EXC[1] *= GridVol;

  /****************************************************
   MPI:

   EH1, EXC
  ****************************************************/

  MPI_Barrier(mpi_comm_level1);
  MPI_Allreduce(&My_EH1, &EH1, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
  for (spin=0; spin<=spinmax; spin++){
    MPI_Allreduce(&My_EXC[spin], &EXC[spin], 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
  }

  if (SpinP_switch==0) EXC[1] = EXC[0];

  ECE[15] = EH1;
  ECE[16] = EXC[0];
  ECE[17] = EXC[1];
  
  //printf("VVV1 dc %18.15f %18.15f %18.15f\n",EH1,EXC[0],EXC[1]);
}


#pragma optimization_level 1
void Calc_EXC_EH1(OpenMXSolverInvocation *scratch, const OpenMXAtomicViews *atomic, const OpenMXGaussRuleData *radial, double ECE[])
{
  /************************************************************
     EXC = \sum_{\sigma} (n_{\sigma}(r)+n_pcc(r)\epsilon_{xc}
     EH1 = 1/2\int {n(r)-n_a(r)} \delta V_H dr
  ************************************************************/

  int i,spin,spinmax,XC_P_switch;
  int numS,numR,My_GNum,BN_AB;
  int n,n1,n2,n3,Ng1,Ng2,Ng3,j,k;
  int GNc,GRc,MNc;
  int GN,GNs,BN,DN,LN,N2D,n2D,N3[4];
  double EXC[2],EH1,sum,tot_den;
  double My_EXC[2],My_EH1;
  double My_EXC_VolumeTerm[2];
  double EXC_VolumeTerm[2];
  double My_Eef,Eef;
  double My_Ena,Ena;
  double sum_charge,My_charge;
  int numprocs,myid,tag=999,ID,IDS,IDR;
  double Cxyz[4],gradxyz[4];
  double Stime_atom,Etime_atom;
  double time0,time1;
  double sden[2],tden,aden,pden[2];

  /* dipole moment */
  int Gc_AN,Mc_AN,spe;
  double x,y,z,den,charge,cden_BG;
  double E_dpx,E_dpy,E_dpz; 
  double E_dpx_BG,E_dpy_BG,E_dpz_BG; 
  double C_dpx,C_dpy,C_dpz;
  double My_E_dpx_BG,My_E_dpy_BG,My_E_dpz_BG; 
  double My_E_dpx,My_E_dpy,My_E_dpz; 
  double My_C_dpx,My_C_dpy,My_C_dpz;
  double AbsD;
  double x0,y0,z0,r;
  double rs,re,ts,te,ps,pe;
  double Sp,Dp,St,Dt,Sr,Dr;
  double r1,dx,dy,dz,dx1,dy1,dz1;
  double x1,y1,z1,den0,exc0,w;
  double sumr,sumt;
  double sumrx,sumtx;
  double sumry,sumty;
  double sumrz,sumtz;
  double gden0,vxc0;
  int ir,ia,Cwan,Rn,Hwan,Gh_AN,h_AN;
  char file_DPM[YOUSO10] = ".dpm";
  FILE *fp_DPM;
  char buf[fp_bsize];          /* setvbuf */
  MPI_Status stat;
  MPI_Request request;
  /* for OpenMP */
  int OMPID,Nthrds,Nthrds0,Nprocs,Nloop;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  if      (SpinP_switch==0) spinmax = 0;
  else if (SpinP_switch==1) spinmax = 1;
  else if (SpinP_switch==3) spinmax = 1;

  /****************************************************
   set Vxc_Grid
  ****************************************************/

  XC_P_switch = 0;

  dtime(&time0);
  Set_XC_Grid(2, XC_P_switch,XC_switch,
	      Density_Grid_D[0],Density_Grid_D[1],
	      Density_Grid_D[2],Density_Grid_D[3],
	      Vxc_Grid_D[0], Vxc_Grid_D[1],
	      Vxc_Grid_D[2], Vxc_Grid_D[3],
	      NULL,NULL);
  dtime(&time1);
  if (myid==0 && measure_time){
    printf("Time for EXC_EH1(Set_XC_Grid)=%18.5f\n",time1-time0);fflush(stdout);
  }

  /* copy Vxc_Grid_D to Vxc_Grid_B */

  Ng1 = Max_Grid_Index_D[1] - Min_Grid_Index_D[1] + 1;
  Ng2 = Max_Grid_Index_D[2] - Min_Grid_Index_D[2] + 1;
  Ng3 = Max_Grid_Index_D[3] - Min_Grid_Index_D[3] + 1;

  for (n=0; n<Num_Rcv_Grid_B2D[myid]; n++){
    DN = Index_Rcv_Grid_B2D[myid][n];
    BN = Index_Snd_Grid_B2D[myid][n];

    i = DN/(Ng2*Ng3);
    j = (DN-i*Ng2*Ng3)/Ng3;
    k = DN - i*Ng2*Ng3 - j*Ng3; 

    if ( !(i<=1 || (Ng1-2)<=i || j<=1 || (Ng2-2)<=j || k<=1 || (Ng3-2)<=k)){
      for (spin=0; spin<=SpinP_switch; spin++){
	Vxc_Grid_B[spin][BN] = Vxc_Grid_D[spin][DN];
      }
    }
  }

  /*********************************************************
   set RefVxc_Grid, where the CA-LDA exchange-correlation 
   functional is alway used.
  *********************************************************/

  XC_P_switch = 0;
  dtime(&time0);
  for (BN_AB=0; BN_AB<My_NumGridB_AB; BN_AB++){
    tot_den = ADensity_Grid_B[BN_AB] + ADensity_Grid_B[BN_AB];
    if (PCC_switch==1) {
      tot_den += PCCDensity_Grid_B[0][BN_AB] + PCCDensity_Grid_B[1][BN_AB];
    }
    RefVxc_Grid_B[BN_AB] = openmx_xc_ceperley_alder(tot_den,XC_P_switch);
  }
  dtime(&time1);
  if (myid==0 && measure_time){
    printf("Time for EXC_EH1(RefVxc)=%18.5f\n",time1-time0);fflush(stdout);
  }

  /****************************************************
        calculations of Ena, Eef, EH1, and EXC
  ****************************************************/

  My_Ena = 0.0;
  My_Eef = 0.0;
  My_EH1 = 0.0;
  My_EXC[0] = 0.0;
  My_EXC[1] = 0.0;

  dtime(&time0);
  int grid_device = scf_eigen_lib_flag == GPUSOLVER;
#ifdef OPENMX_HOST_SIMD_H
  if (grid_device)
    grid_device = !TotalEnergy_StageOnHost("Total_Energy XC/Hartree grid",TotalEnergy_GridBytes(0));
#endif
  if (grid_device){
#if OPENMX_GPU_HIP
    TotalEnergy_EXC_EH1_Grid_OpenMP(spinmax, &My_Ena, &My_Eef, &My_EH1, My_EXC);
#else
    TotalEnergy_EXC_EH1_Grid_OpenACC(spinmax, &My_Ena, &My_Eef, &My_EH1, My_EXC);
#endif
  }
  else{
    for (BN=0; BN<My_NumGridB_AB; BN++){

      sden[0] = Density_Grid_B[0][BN];
      sden[1] = Density_Grid_B[1][BN];
      tden = sden[0] + sden[1];
      aden = ADensity_Grid_B[BN];
      pden[0] = PCCDensity_Grid_B[0][BN];
      pden[1] = PCCDensity_Grid_B[1][BN];

      /* if (ProExpn_VNA==off), Ena is calculated here. */
      if (ProExpn_VNA==0) My_Ena += tden*VNA_Grid_B[BN];

      /* electric energy by electric field */
      if (E_Field_switch==1) My_Eef += tden*VEF_Grid_B[BN];

      /* EH1 = 1/2\int \delta n(r) \delta V_H dr */
      My_EH1 += (tden - 2.0*aden)*dVHart_Grid_B[BN];

      /*   EXC = \sum_{\sigma} (n_{\sigma}+n_pcc)\epsilon_{xc}
                -(n_{atom}+n_pcc)\epsilon_{xc}(n_{atom})

          calculation of the difference between the xc energies
          calculated by wave-function-charge and atomic charge
          on the coarse grid.  */

      if (Exc0_correction_flag==1){
        for (spin=0; spin<=spinmax; spin++){
          My_EXC[spin] += (sden[spin]+pden[spin])*Vxc_Grid_B[spin][BN] - (aden+pden[spin])*RefVxc_Grid_B[BN];
        }
      }
      else{
        for (spin=0; spin<=spinmax; spin++){
          My_EXC[spin] += (sden[spin]+pden[spin])*Vxc_Grid_B[spin][BN];
        }
      }

    } /* BN */
  }

  dtime(&time1);
  if (myid==0 && measure_time){
    printf("Time for EXC_EH1(grid reduction)=%18.5f\n",time1-time0);fflush(stdout);
  }

  /****************************************************
       multiplying GridVol and MPI communication
  ****************************************************/

  MPI_Barrier(mpi_comm_level1);

  /* if (ProExpn_VNA==off), Ena is calculated here. */

  if (ProExpn_VNA==0){

    if (F_VNA_flag==1){
      My_Ena *= GridVol;
      MPI_Allreduce(&My_Ena, &Ena, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
      ECE[3] = Ena;
    }
    else{
      ECE[3] = 0.0;
    }
  }

  /* electric energy by electric field */
  if (E_Field_switch==1){
    My_Eef *= GridVol;
    MPI_Allreduce(&My_Eef, &Eef, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
    ECE[12] = Eef;
  }
  else {
    ECE[12] = 0.0;
  }
  if (F_VEF_flag==0){
    ECE[12] = 0.0;
  }

  /* EH1 = 1/2\int \delta n(r) \delta V_H dr */
  My_EH1 *= (0.5*GridVol);

  /************************************************************
    EXC = \sum_{\sigma} n_{\sigma}\epsilon_{xc}
       - n_{atom}\epsilon_{xc}(n_{atom})

       calculation of the difference between the xc energies 
       calculated by wave-function-charge and atomic charge
       on the coarse grid.  

       My_EXC_VolumeTerm will be used to take account of 
       volume term for stress. 
  *************************************************************/

  My_EXC[0] *= GridVol;
  My_EXC[1] *= GridVol;

  My_EXC_VolumeTerm[0] = My_EXC[0];
  My_EXC_VolumeTerm[1] = My_EXC[1];

  /****************************************************
    calculation of Exc^(0) and its contribution 
    to forces on the fine mesh
  ****************************************************/

  dtime(&time0);
  if (Exc0_correction_flag==1){

    OpenMXLebedevRuleData lebedev;
    OpenMX_Energy_LebedevPrepare(scratch,Num_Leb_Grid,&lebedev);


    /* get Nthrds0 */  
    /* Dynamic teams may grow after a separate probe region. */
    Nthrds0 = omp_get_max_threads();

    /* initialize the temporal array storing the force contribution */

    for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
      Gxyz[Gc_AN][41] = 0.0;
      Gxyz[Gc_AN][42] = 0.0;
      Gxyz[Gc_AN][43] = 0.0;
    }

    /* start calc. */

    rs = 0.0;
    sum = 0.0;

    int exc0_device = scf_eigen_lib_flag == GPUSOLVER;
#ifdef OPENMX_HOST_SIMD_H
    if (exc0_device)
      exc0_device = !TotalEnergy_StageOnHost("Total_Energy Exc0",TotalEnergy_Exc0Bytes(Num_Leb_Grid));
#endif
    if (exc0_device){
#if OPENMX_GPU_HIP
      TotalEnergy_Exc0_Batch_OpenMP(atomic, radial, &lebedev, &sum);
#else
      TotalEnergy_Exc0_Batch_OpenACC(atomic, radial, &lebedev, &sum);
#endif
    }
    else{

    size_t max_neighbors=0;
    for(int local_atom=1;local_atom<=Matomnum;local_atom++){
      const size_t count=(size_t)FNAN[M2G[local_atom]]+1;
      if(max_neighbors<count)max_neighbors=count;
    }
    max_neighbors=OpenMX_Solver_SimdColumns(max_neighbors);
    OpenMXSolverRealTensor xc_partials;
    OpenMX_Energy_XCPartialsPrepare(scratch,(size_t)Nthrds0,max_neighbors,&xc_partials);
    double *My_sumr=OpenMX_Solver_OverwriteReal(scratch,"total-energy:thread-sums",(size_t)Nthrds0);

    /* The Core's Kokkos::simd shell kernel (OPENMX_HOST_SIMD) takes the
       Lebedev loop of every radial shell; legacy mode, a refusal or memory
       shortage keep the loop below. */
#ifdef OPENMX_HOST_SIMD_H
    const int xc_mode=OpenMX_HostSimdMode();
    const int xc_simd=(xc_mode!=OPENMX_HOST_SIMD_LEGACY);
    const size_t xc_scratch_count=xc_simd?openmx_core_energy_xc_shell_scratch((int)max_neighbors):0;
#else
    /* the standalone test harness compiles this block without the host SIMD glue */
    const int xc_simd=0;
    const size_t xc_scratch_count=0;
#endif

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
	  OpenMX_Energy_XCForceRow(&xc_partials,0,i)[j] = 0.0;
	}
      }

      for (i=0; i<Nthrds0; i++){
	for (j=0; j<(FNAN[Gc_AN]+1); j++){
	  OpenMX_Energy_XCForceRow(&xc_partials,1,i)[j] = 0.0;
	}
      }

      for (i=0; i<Nthrds0; i++){
	for (j=0; j<(FNAN[Gc_AN]+1); j++){
	  OpenMX_Energy_XCForceRow(&xc_partials,2,i)[j] = 0.0;
	}
      }

      /* the neighbours of this atom for the shell kernel: positions with
         the cell shift, cutoff squared and PAO density tables */
      double *xc_nx=NULL,*xc_ny=NULL,*xc_nz=NULL,*xc_cut2=NULL;
      int *xc_mesh=NULL;
      const double **xc_xv=NULL,**xc_rv=NULL,**xc_yv=NULL;
      if (xc_simd){
        const size_t count=(size_t)FNAN[Gc_AN]+1;
        xc_nx=(double*)malloc(sizeof(double)*count);
        xc_ny=(double*)malloc(sizeof(double)*count);
        xc_nz=(double*)malloc(sizeof(double)*count);
        xc_cut2=(double*)malloc(sizeof(double)*count);
        xc_mesh=(int*)malloc(sizeof(int)*count);
        xc_xv=(const double**)malloc(sizeof(double*)*count);
        xc_rv=(const double**)malloc(sizeof(double*)*count);
        xc_yv=(const double**)malloc(sizeof(double*)*count);
        if (xc_nx && xc_ny && xc_nz && xc_cut2 && xc_mesh && xc_xv && xc_rv && xc_yv){
          for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
            Gh_AN = natn[Gc_AN][h_AN];
            Rn = ncn[Gc_AN][h_AN];
            Hwan = WhatSpecies[Gh_AN];
            xc_nx[h_AN] = Gxyz[Gh_AN][1] + atv[Rn][1];
            xc_ny[h_AN] = Gxyz[Gh_AN][2] + atv[Rn][2];
            xc_nz[h_AN] = Gxyz[Gh_AN][3] + atv[Rn][3];
            xc_cut2[h_AN] = atomic->species.cutoff[Hwan]*atomic->species.cutoff[Hwan];
            xc_mesh[h_AN] = atomic->species.pao_mesh[Hwan];
            xc_xv[h_AN] = openmx_radial_row2(&atomic->pao_log_radii,Hwan);
            xc_rv[h_AN] = openmx_radial_row2(&atomic->pao_radii,Hwan);
            xc_yv[h_AN] = openmx_radial_row2(&atomic->density_backup,Hwan);
          }
        }
        else{
          free(xc_nx); free(xc_ny); free(xc_nz); free(xc_cut2); free(xc_mesh); free(xc_xv); free(xc_rv); free(xc_yv);
          xc_nx=xc_ny=xc_nz=xc_cut2=NULL; xc_mesh=NULL; xc_xv=xc_rv=xc_yv=NULL;
        }
      }

#pragma omp parallel shared(scratch,max_neighbors,xc_partials,atomic,lebedev,Num_Leb_Grid,My_sumr,Dr,Sr,radial,Gxyz,Gc_AN,FNAN,natn,ncn,WhatSpecies,atv,F_Vxc_flag,Cwan,PCC_switch) private(OMPID,Nthrds,Nprocs,ir,ia,r,w,sumt,sumtx,sumty,sumtz,x,x0,y0,z0,h_AN,Gh_AN,Rn,Hwan,x1,y1,z1,dx,dy,dz,r1,den,den0,gden0,dx1,dy1,dz1,exc0,vxc0,i)
      {

	double *gx,*gy,*gz,dexc0;
	double *sum_gx,*sum_gy,*sum_gz;
	double r2,rcutH2;
        OpenMXSolverRealTensor derivatives;
        OpenMX_Energy_XCWorkerPrepare(scratch,max_neighbors,&derivatives);
        gx=openmx_solver_real_matrix_row(&derivatives,0);
        gy=openmx_solver_real_matrix_row(&derivatives,1);
        gz=openmx_solver_real_matrix_row(&derivatives,2);
        sum_gx=openmx_solver_real_matrix_row(&derivatives,3);
        sum_gy=openmx_solver_real_matrix_row(&derivatives,4);
        sum_gz=openmx_solver_real_matrix_row(&derivatives,5);


	/* get info. on OpenMP */ 

	OMPID = omp_get_thread_num();
	Nthrds = omp_get_num_threads();
	Nprocs = omp_get_num_procs();
        double *partial_x=OpenMX_Energy_XCForceRow(&xc_partials,0,OMPID);
        double *partial_y=OpenMX_Energy_XCForceRow(&xc_partials,1,OMPID);
        double *partial_z=OpenMX_Energy_XCForceRow(&xc_partials,2,OMPID);
        double *xc_scratch=(xc_nx!=NULL)?(double*)malloc(sizeof(double)*(xc_scratch_count?xc_scratch_count:1)):NULL;
        double den0_carry=0.0;

	for (ir=(OMPID*CoarseGL_Mesh/Nthrds); ir<((OMPID+1)*CoarseGL_Mesh/Nthrds); ir++){

	  r = 0.50*(Dr*radial->nodes[ir] + Sr);
	  sumt  = 0.0; 

	  for (i=0; i<(FNAN[Gc_AN]+1); i++){
	    sum_gx[i] = 0.0;
	    sum_gy[i] = 0.0;
	    sum_gz[i] = 0.0;
	  }

	  int xc_done=0;
#ifdef OPENMX_HOST_SIMD_H
	  if (xc_scratch!=NULL){
	    OpenMXEnergyXcShell shell;
	    shell.points=Num_Leb_Grid;
	    shell.lebedev=lebedev.xyzw;
	    shell.lebedev_stride=4;
	    shell.radius=r;
	    shell.center[0]=Gxyz[Gc_AN][1];
	    shell.center[1]=Gxyz[Gc_AN][2];
	    shell.center[2]=Gxyz[Gc_AN][3];
	    shell.vxc=(double)F_Vxc_flag;
	    shell.neighbors=FNAN[Gc_AN]+1;
	    shell.nx=xc_nx; shell.ny=xc_ny; shell.nz=xc_nz; shell.cutoff2=xc_cut2;
	    shell.mesh=xc_mesh; shell.log_radii=xc_xv; shell.radii=xc_rv; shell.density=xc_yv;
	    shell.scratch=xc_scratch; shell.scratch_count=xc_scratch_count;
	    shell.den0=&den0_carry; shell.sumt=&sumt;
	    shell.sum_gx=sum_gx; shell.sum_gy=sum_gy; shell.sum_gz=sum_gz;
	    xc_done=openmx_core_energy_xc_shell(&shell,xc_mode);
	  }
#endif

	  if (!xc_done)
	  for (ia=0; ia<Num_Leb_Grid; ia++){

	    x0 = r*openmx_lebedev_point(&lebedev,ia)[0] + Gxyz[Gc_AN][1];
	    y0 = r*openmx_lebedev_point(&lebedev,ia)[1] + Gxyz[Gc_AN][2];
	    z0 = r*openmx_lebedev_point(&lebedev,ia)[2] + Gxyz[Gc_AN][3];

	    /* calculate rho_atom + rho_pcc */ 

	    den = 0.0;

	    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	      Gh_AN = natn[Gc_AN][h_AN];
	      Rn = ncn[Gc_AN][h_AN]; 
	      Hwan = WhatSpecies[Gh_AN];
	      rcutH2 = atomic->species.cutoff[Hwan]*atomic->species.cutoff[Hwan];

	      x1 = Gxyz[Gh_AN][1] + atv[Rn][1];
	      y1 = Gxyz[Gh_AN][2] + atv[Rn][2];
	      z1 = Gxyz[Gh_AN][3] + atv[Rn][3];
            
	      dx = x1 - x0;
	      dy = y1 - y0;
	      dz = z1 - z0;

	      r2 = dx*dx+dy*dy+dz*dz;

	      gx[h_AN] = 0.0;
	      gy[h_AN] = 0.0;
	      gz[h_AN] = 0.0;

	      if (r2<rcutH2){

		x = 0.5*log(r2);

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

	      } /* if (r2<rcutH2) */

	    } /* h_AN */

	    /* calculate the CA-LDA exchange-correlation energy density */
	    exc0 = openmx_xc_ceperley_alder(den,0);

	    /* calculate the CA-LDA exchange-correlation potential */
	    dexc0 = openmx_xc_ceperley_alder(den,3);

	    /* Lebedev quadrature */

	    w = openmx_lebedev_point(&lebedev,ia)[3];
	    sumt += w*den0*exc0;

	    for (h_AN=1; h_AN<=FNAN[Gc_AN]; h_AN++){
	      sum_gx[h_AN] += w*den0*dexc0*gx[h_AN]; 
	      sum_gy[h_AN] += w*den0*dexc0*gy[h_AN]; 
	      sum_gz[h_AN] += w*den0*dexc0*gz[h_AN]; 
	    }

	  } /* ia */

	  /* r for Gauss-Legendre quadrature */

	  w = r*r*radial->weights[ir];
	  My_sumr[OMPID]  += w*sumt;

	  for (h_AN=1; h_AN<=FNAN[Gc_AN]; h_AN++){
	    partial_x[h_AN] += w*sum_gx[h_AN];
	    partial_y[h_AN] += w*sum_gy[h_AN];
	    partial_z[h_AN] += w*sum_gz[h_AN];
	  }

	} /* ir */

        free(xc_scratch);

      } /* #pragma omp */

      free(xc_nx); free(xc_ny); free(xc_nz); free(xc_cut2); free(xc_mesh); free(xc_xv); free(xc_rv); free(xc_yv);

      sumr = 0.0;
      for (Nloop=0; Nloop<Nthrds0; Nloop++){
	sumr += My_sumr[Nloop];
      }
      sum += 2.0*PI*Dr*sumr;

      /* add force */

      for (h_AN=1; h_AN<=FNAN[Gc_AN]; h_AN++){

	sumrx = 0.0;
	sumry = 0.0;
	sumrz = 0.0;
	for (Nloop=0; Nloop<Nthrds0; Nloop++){
	  sumrx += OpenMX_Energy_XCForceRow(&xc_partials,0,Nloop)[h_AN];
	  sumry += OpenMX_Energy_XCForceRow(&xc_partials,1,Nloop)[h_AN];
	  sumrz += OpenMX_Energy_XCForceRow(&xc_partials,2,Nloop)[h_AN];
	}

	Gh_AN = natn[Gc_AN][h_AN];

	Gxyz[Gh_AN][41] += 2.0*PI*Dr*sumrx;
	Gxyz[Gh_AN][42] += 2.0*PI*Dr*sumry;
	Gxyz[Gh_AN][43] += 2.0*PI*Dr*sumrz;

	Gxyz[Gc_AN][41] -= 2.0*PI*Dr*sumrx;
	Gxyz[Gc_AN][42] -= 2.0*PI*Dr*sumry;
	Gxyz[Gc_AN][43] -= 2.0*PI*Dr*sumrz;
      }

      /* freeing of arrays */

      /* The invocation retains the partials for the next atom. */


    } /* Mc_AN */

    } /* else (host path) */

    /* add Exc^0 calculated on the fine mesh to My_EXC */

    My_EXC[0] += 0.5*sum;
    My_EXC[1] += 0.5*sum;

    /* Angular data and ABI rows expire with the invocation. */

    /* MPI: Gxyz[][41,42,43] */

    for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
      MPI_Allreduce(&Gxyz[Gc_AN][41], &gradxyz[0], 3, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

      Gxyz[Gc_AN][17] += gradxyz[0];
      Gxyz[Gc_AN][18] += gradxyz[1];
      Gxyz[Gc_AN][19] += gradxyz[2];

      if (2<=level_stdout){
	printf("<Total_Ene>  force(8) myid=%2d Gc_AN=%2d  %15.12f %15.12f %15.12f\n",
	       myid,Gc_AN,gradxyz[0],gradxyz[1],gradxyz[2]);
      }
    }

  } /* if (Exc0_correction_flag==1) */

  dtime(&time1);
  if (myid==0 && measure_time){
    printf("Time for EXC_EH1(Exc0 fine mesh)=%18.5f\n",time1-time0);fflush(stdout);
  }

  /****************************************************
   MPI, Gxyz[Gc_AN][17-19]
  ****************************************************/

  for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
    ID = G2ID[Gc_AN];
    MPI_Bcast(&Gxyz[Gc_AN][17], 3, MPI_DOUBLE, ID, mpi_comm_level1);

    if (2<=level_stdout && MYID_MPI_COMM_WORLD==Host_ID){
      printf("<Total_Ene>  force(t) myid=%2d Gc_AN=%2d  %15.12f %15.12f %15.12f\n",
              myid,Gc_AN,Gxyz[Gc_AN][17],Gxyz[Gc_AN][18],Gxyz[Gc_AN][19]);fflush(stdout);
    }
  }

  /****************************************************
   MPI:

   EH1, EXC
  ****************************************************/

  MPI_Barrier(mpi_comm_level1);
  MPI_Allreduce(&My_EH1, &EH1, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
  for (spin=0; spin<=spinmax; spin++){
    MPI_Allreduce(&My_EXC[spin], &EXC[spin], 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
    MPI_Allreduce(&My_EXC_VolumeTerm[spin], &EXC_VolumeTerm[spin], 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
  }

  if (SpinP_switch==0){
    ECE[5] = EH1;
    ECE[6] = EXC[0];
    ECE[7] = EXC[0];
    EXC_VolumeTerm[1] = EXC_VolumeTerm[0];
  }
  else if (SpinP_switch==1 || SpinP_switch==3) {
    ECE[5] = EH1;
    ECE[6] = EXC[0];
    ECE[7] = EXC[1];
  }

  if (F_dVHart_flag==0){
    ECE[5] = 0.0;
  }

  if (F_Vxc_flag==0){
    ECE[6] = 0.0;
    ECE[7] = 0.0;
  }

  if (F_Vxc_flag==1){
    Stress_Tensor[0] += EXC_VolumeTerm[0] + EXC_VolumeTerm[1];
    Stress_Tensor[4] += EXC_VolumeTerm[0] + EXC_VolumeTerm[1];
    Stress_Tensor[8] += EXC_VolumeTerm[0] + EXC_VolumeTerm[1];
  }

  /****************************************************
             calculation of dipole moment
  ****************************************************/

  if (Polarization_ABC_flag==0){

    /* contribution from electron density */

    N2D = Ngrid1*Ngrid2;
    GNs = ((myid*N2D+numprocs-1)/numprocs)*Ngrid3;

    My_E_dpx = 0.0;
    My_E_dpy = 0.0;
    My_E_dpz = 0.0;

    My_E_dpx_BG = 0.0;
    My_E_dpy_BG = 0.0;
    My_E_dpz_BG = 0.0; 

    int dipole_device = scf_eigen_lib_flag == GPUSOLVER;
#ifdef OPENMX_HOST_SIMD_H
    if (dipole_device)
      dipole_device = !TotalEnergy_StageOnHost("Total_Energy dipole grid",TotalEnergy_GridBytes(1));
#endif
    if (dipole_device){
#if OPENMX_GPU_HIP
      TotalEnergy_Dipole_Grid_OpenMP(GNs, &My_E_dpx, &My_E_dpy, &My_E_dpz,
#else
      TotalEnergy_Dipole_Grid_OpenACC(GNs, &My_E_dpx, &My_E_dpy, &My_E_dpz,
#endif
                                      &My_E_dpx_BG, &My_E_dpy_BG, &My_E_dpz_BG);
    }
    else{
      for (BN=0; BN<My_NumGridB_AB; BN++){

        GN = BN + GNs;
        n1 = GN/(Ngrid2*Ngrid3);
        n2 = (GN - n1*Ngrid2*Ngrid3)/Ngrid3;
        n3 = GN - n1*Ngrid2*Ngrid3 - n2*Ngrid3;

        x = (double)n1*gtv[1][1] + (double)n2*gtv[2][1]
	  + (double)n3*gtv[3][1] + Grid_Origin[1];
        y = (double)n1*gtv[1][2] + (double)n2*gtv[2][2]
	  + (double)n3*gtv[3][2] + Grid_Origin[2];
        z = (double)n1*gtv[1][3] + (double)n2*gtv[2][3]
	  + (double)n3*gtv[3][3] + Grid_Origin[3];

        den = Density_Grid_B[0][BN] + Density_Grid_B[1][BN];

        My_E_dpx += den*x;
        My_E_dpy += den*y;
        My_E_dpz += den*z;

        My_E_dpx_BG += x;
        My_E_dpy_BG += y;
        My_E_dpz_BG += z;

      } /* BN */
    }

    MPI_Allreduce(&My_E_dpx, &E_dpx, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
    MPI_Allreduce(&My_E_dpy, &E_dpy, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
    MPI_Allreduce(&My_E_dpz, &E_dpz, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

    MPI_Allreduce(&My_E_dpx_BG, &E_dpx_BG, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
    MPI_Allreduce(&My_E_dpy_BG, &E_dpy_BG, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
    MPI_Allreduce(&My_E_dpz_BG, &E_dpz_BG, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

    E_dpx = E_dpx*GridVol;
    E_dpy = E_dpy*GridVol;
    E_dpz = E_dpz*GridVol;

    cden_BG = system_charge/Cell_Volume; 

    E_dpx_BG = E_dpx_BG*GridVol*cden_BG;
    E_dpy_BG = E_dpy_BG*GridVol*cden_BG;
    E_dpz_BG = E_dpz_BG*GridVol*cden_BG;

    /* contribution from core charge */

    My_C_dpx = 0.0;
    My_C_dpy = 0.0;
    My_C_dpz = 0.0;

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];
      x = Gxyz[Gc_AN][1];
      y = Gxyz[Gc_AN][2];
      z = Gxyz[Gc_AN][3];

      spe = WhatSpecies[Gc_AN];
      charge = atomic->species.core_charge[spe];
      My_C_dpx += charge*x;
      My_C_dpy += charge*y;
      My_C_dpz += charge*z;
    }

    MPI_Allreduce(&My_C_dpx, &C_dpx, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
    MPI_Allreduce(&My_C_dpy, &C_dpy, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
    MPI_Allreduce(&My_C_dpz, &C_dpz, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

    dipole_moment[0][1] = AU2Debye*(C_dpx - E_dpx - E_dpx_BG);
    dipole_moment[0][2] = AU2Debye*(C_dpy - E_dpy - E_dpy_BG);
    dipole_moment[0][3] = AU2Debye*(C_dpz - E_dpz - E_dpz_BG);

    dipole_moment[1][1] = AU2Debye*C_dpx;
    dipole_moment[1][2] = AU2Debye*C_dpy;
    dipole_moment[1][3] = AU2Debye*C_dpz;

    dipole_moment[2][1] = -AU2Debye*E_dpx;
    dipole_moment[2][2] = -AU2Debye*E_dpy;
    dipole_moment[2][3] = -AU2Debye*E_dpz;

    dipole_moment[3][1] = -AU2Debye*E_dpx_BG;
    dipole_moment[3][2] = -AU2Debye*E_dpy_BG;
    dipole_moment[3][3] = -AU2Debye*E_dpz_BG;

    AbsD = sqrt( dipole_moment[0][1]*dipole_moment[0][1]
		 + dipole_moment[0][2]*dipole_moment[0][2]
		 + dipole_moment[0][3]*dipole_moment[0][3] );

    if (MYID_MPI_COMM_WORLD==Host_ID){

      if (0<level_stdout){
	printf("\n*******************************************************\n"); fflush(stdout);
	printf("                  Dipole moment (Debye)                 \n");  fflush(stdout);
	printf("*******************************************************\n\n"); fflush(stdout);

	printf(" Absolute D %17.8f\n\n",AbsD);
	printf("                      Dx                Dy                Dz\n"); fflush(stdout);
	printf(" Total       %17.8f %17.8f %17.8f\n",
	       dipole_moment[0][1],dipole_moment[0][2],dipole_moment[0][3]);fflush(stdout);
	printf(" Core        %17.8f %17.8f %17.8f\n",
	       dipole_moment[1][1],dipole_moment[1][2],dipole_moment[1][3]);fflush(stdout);
	printf(" Electron    %17.8f %17.8f %17.8f\n",
	       dipole_moment[2][1],dipole_moment[2][2],dipole_moment[2][3]);fflush(stdout);
	printf(" Back ground %17.8f %17.8f %17.8f\n",
	       dipole_moment[3][1],dipole_moment[3][2],dipole_moment[3][3]);fflush(stdout);
      }

      /********************************************************
             write the dipole moments to a file
      ********************************************************/

      fnjoint(filepath,filename,file_DPM);

      if ((fp_DPM = fopen(file_DPM,"w")) != NULL){

#ifdef xt3
	setvbuf(fp_DPM,buf,_IOFBF,fp_bsize);  /* setvbuf */
#endif

	fprintf(fp_DPM,"\n");
	fprintf(fp_DPM,"***********************************************************\n");
	fprintf(fp_DPM,"***********************************************************\n");
	fprintf(fp_DPM,"                    Dipole moment (Debye)                  \n");
	fprintf(fp_DPM,"***********************************************************\n");
	fprintf(fp_DPM,"***********************************************************\n\n");

	fprintf(fp_DPM," Absolute D %17.8f\n\n",AbsD);
	fprintf(fp_DPM,"                      Dx                Dy                Dz\n");
	fprintf(fp_DPM," Total       %17.8f %17.8f %17.8f\n",
                dipole_moment[0][1],dipole_moment[0][2],dipole_moment[0][3]);
	fprintf(fp_DPM," Core        %17.8f %17.8f %17.8f\n",
                dipole_moment[1][1],dipole_moment[1][2],dipole_moment[1][3]);
	fprintf(fp_DPM," Electron    %17.8f %17.8f %17.8f\n",
                dipole_moment[2][1],dipole_moment[2][2],dipole_moment[2][3]);
	fprintf(fp_DPM," Back ground %17.8f %17.8f %17.8f\n",
                dipole_moment[3][1],dipole_moment[3][2],dipole_moment[3][3]);

	fclose(fp_DPM);
      }
      else{
	printf("Failure of saving the DPM file.\n");fflush(stdout);
      }
    }
  }

}



void EH0_TwoCenter(const OpenMXNeutralGridTables *tables,const OpenMXNeutralGridMetadataView *metadata,const OpenMXAtomicViews *atomic, int Gc_AN, int h_AN, double VH0ij[4])
{
  const int Gh_AN=natn[Gc_AN][h_AN],Rn=ncn[Gc_AN][h_AN];
  const int wan1=WhatSpecies[Gc_AN],wan2=WhatSpecies[Gh_AN];
  const int ban=atomic->species.species_index[wan1];
  const OpenMXNeutralGridView grid={metadata->counts[ban],metadata->spacing[ban],
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














void EH0_TwoCenter_at_Cutoff(OpenMXSolverInvocation *scratch,const OpenMXNeutralGridTables *tables,const OpenMXNeutralGridMetadataView *metadata, const OpenMXAtomicViews *atomic, int wan1, int wan2, double VH0ij[4])
{ 
  int ban;
  double dv,rcut;
  double sum,sumx,sumy,sumz;
  /* for OpenMP */
  int OMPID,Nthrds,Nthrds0,Nprocs,Nloop;
  double *my_sum_threads;

  ban = atomic->species.species_index[wan1];
  dv  = metadata->spacing[ban];
  const OpenMXNeutralGridView grid={metadata->counts[ban],dv,
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
  my_sum_threads = OpenMX_Solver_OverwriteReal(scratch,"total-energy:thread-sums",(size_t)Nthrds0);

  for (Nloop=0; Nloop<Nthrds0; Nloop++){
    my_sum_threads[Nloop] = 0.0;
  }

#pragma omp parallel shared(grid,potential,my_sum_threads,rcut) private(OMPID,Nthrds,Nprocs)
  {
    /* get info. on OpenMP */

    OMPID = omp_get_thread_num();
    Nthrds = omp_get_num_threads();
    Nprocs = omp_get_num_procs();

    OpenMX_Neutral_CutoffAccumulate(&grid,&potential,rcut,
        OMPID*grid.count/Nthrds,(OMPID+1)*grid.count/Nthrds,my_sum_threads+OMPID);

  } /* #pragma omp parallel */

  sum  = 0.0;
  for (Nloop=0; Nloop<Nthrds0; Nloop++){
    sum += my_sum_threads[Nloop];
  }

  sum  = sum*dv;
  sumx = 0.0;
  sumy = 0.0;
  sumz = 0.0;

  VH0ij[0] = sum;
  VH0ij[1] = sumx;
  VH0ij[2] = sumy;
  VH0ij[3] = sumz;

  /* The invocation retains the reusable partial sums. */
}





double Calc_Ehub(const OpenMXLocalBasisViews *local_basis)
{

 /****************************************************
         LDA+U energy correction added by MJ
  ****************************************************/

  int Mc_AN,Gc_AN,wan1;
  int cnt1,cnt2,l1,mul1,m1,l2,mul2,m2;
  int spin,max_spin;
  double My_Ehub,Ehub,Uvalue,tmpv,sum;
  int numprocs,myid,ID;

  /* added by S.Ryee */
  int on_off,cnt_start,tmp_l1,ii,jj,kk,ll;	
  double Jvalue,tmpEhub1,tmpEhub2,tmpEhub3,tmpEhub4,trace_spin,trace_opp_spin;
  int dd;
  int NZUJ;
  dcomplex N_00_ac,N_11_ac,N_00_bd,N_11_bd,N_01_ac,N_10_ac,N_01_bd,N_10_bd;
  dcomplex AMF_00_ac,AMF_11_ac,AMF_00_bd,AMF_11_bd,AMF_01_ac,AMF_10_ac,AMF_01_bd,AMF_10_bd;
  dcomplex trace_N00,trace_N11,trace_N01,trace_N10;
  /*******************/ 


  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

 /****************************************************
                 caculation of My_Ehub
  ****************************************************/

  if      (SpinP_switch==0) max_spin = 0;
  else if (SpinP_switch==1) max_spin = 1;
  else if (SpinP_switch==3) max_spin = 1;

  My_Ehub = 0.0;
  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];
    wan1 = WhatSpecies[Gc_AN];

   /****************************************************
                     collinear case
    ****************************************************/
    if (SpinP_switch!=3){
      switch (Hub_Type){
      case 1:		/* Dudarev form */
        for (spin=0; spin<=max_spin; spin++){

          /* Hubbard term, 0.5*Tr(N) */

	  cnt1 = 0;
	  for(l1=0; l1<=Spe_MaxL_Basis[wan1]; l1++ ){
	    for(mul1=0; mul1<Spe_Num_Basis[wan1][l1]; mul1++){

	      Uvalue = Hub_U_Basis[wan1][l1][mul1];
	      for(m1=0; m1<(2*l1+1); m1++){

                tmpv = 0.5*Uvalue*openmx_local_density_row(local_basis, 0, spin, Mc_AN, cnt1)[cnt1];
	        My_Ehub += tmpv;

	        cnt1++;
	      }
	    }
	  }


          /* Hubbard term, -0.5*Tr(N*N) */

	  cnt1 = 0;
	  for(l1=0; l1<=Spe_MaxL_Basis[wan1]; l1++ ){
	    for(mul1=0; mul1<Spe_Num_Basis[wan1][l1]; mul1++){
	      for(m1=0; m1<(2*l1+1); m1++){

                sum = 0.0;  

	        cnt2 = 0;
	        for(l2=0; l2<=Spe_MaxL_Basis[wan1]; l2++ ){
		  for(mul2=0; mul2<Spe_Num_Basis[wan1][l2]; mul2++){
		    for(m2=0; m2<(2*l2+1); m2++){

		      if (l1==l2 && mul1==mul2){
                      
		        Uvalue = Hub_U_Basis[wan1][l1][mul1];
		        sum -= 0.5*Uvalue*openmx_local_density_row(local_basis, 0, spin, Mc_AN, cnt1)[cnt2]*
			                  openmx_local_density_row(local_basis, 0, spin, Mc_AN, cnt2)[cnt1];
		      }

		      cnt2++;
		    }
		  }
	        }

                My_Ehub += sum;

	        cnt1++;
	      }
	    }
	  }
	} /* spin */
      break;

      case 2:		/* general form by S.Ryee */
	/* U Energy */

	for(l1=0; l1<=Spe_MaxL_Basis[wan1]; l1++){
	  for(mul1=0; mul1<Spe_Num_Basis[wan1][l1]; mul1++){
	    Uvalue = Hub_U_Basis[wan1][l1][mul1];
	    Jvalue = Hund_J_Basis[wan1][l1][mul1];
            NZUJ = Nonzero_UJ[wan1][l1][mul1];
            if(NZUJ>0){
	      cnt_start = 0;
	      switch (mul1){
	      case 0:	/* mul1 = 0 */
	        if(l1 > 0){
		  for(tmp_l1=0; tmp_l1<l1; tmp_l1++){
		    cnt_start += (2*tmp_l1+1)*Spe_Num_Basis[wan1][tmp_l1];
		  }
	        }
	        else{	/* l1 = 0 */
		  cnt_start = 0;
	        }
	      break;
	
	      case 1:	/* mul1 = 1 */
	        if(l1 > 0){
		  for(tmp_l1=0; tmp_l1<l1; tmp_l1++){
		    cnt_start += (2*tmp_l1+1)*Spe_Num_Basis[wan1][tmp_l1];
		  }
		  cnt_start += (2*l1+1)*mul1;
	        }
	        else{	/* l1 = 0 */
		  cnt_start = (2*l1+1)*mul1;
	        }
	      break;
	      }	/* switch (mul1) */


	      trace_spin = 0.0;
	      trace_opp_spin = 0.0;
	      for(ii=0; ii<(2*l1+1); ii++){
	        trace_spin += openmx_local_density_row(local_basis, 0, 0, Mc_AN, cnt_start+ii)[cnt_start+ii];
	        trace_opp_spin += openmx_local_density_row(local_basis, 0, 1, Mc_AN, cnt_start+ii)[cnt_start+ii];
	      }
	      /* dc Energy */
	      if(dc_Type==1){  /* sFLL */
	        My_Ehub -= 0.5*(Uvalue)*(trace_spin+trace_opp_spin)*(trace_spin+trace_opp_spin-1.0) 
		          -0.5*(Jvalue)*(trace_spin*(trace_spin-1.0)+trace_opp_spin*(trace_opp_spin-1.0));
	      } /* sFLL */

              if(dc_Type==3){  /* cFLL */
	        My_Ehub -= 0.5*(Uvalue)*(trace_spin+trace_opp_spin)*(trace_spin+trace_opp_spin-1.0) 
		          -0.25*(Jvalue)*(trace_spin+trace_opp_spin)*(trace_spin+trace_opp_spin-2.0);
	      } /* cFLL */


             /*f(dc_Type==3){  
	        My_Ehub -= dc_alpha[count]*(0.5*(Uvalue)*(trace_spin+trace_opp_spin)*(trace_spin+trace_opp_spin-1.0) 
		                    -0.5*(Jvalue)*(trace_spin*(trace_spin-1.0)+trace_opp_spin*(trace_opp_spin-1.0)));
   	      } */

              /* loop start for interaction energy */
	      for(ii=0; ii<(2*l1+1); ii++){
                for(jj=0; jj<(2*l1+1); jj++){
	          for(kk=0; kk<(2*l1+1); kk++){
	            for(ll=0; ll<(2*l1+1); ll++){
                      switch(dc_Type){
                      case 1:  /* sFLL */
		        tmpEhub1 = 0.5*Coulomb_Array[NZUJ][ii][jj][kk][ll]*
                                      (openmx_local_density_row(local_basis, 0, 0, Mc_AN, cnt_start+ii)[cnt_start+kk]*
                                       openmx_local_density_row(local_basis, 0, 1, Mc_AN, cnt_start+jj)[cnt_start+ll]
                                      +openmx_local_density_row(local_basis, 0, 1, Mc_AN, cnt_start+ii)[cnt_start+kk]*
                                       openmx_local_density_row(local_basis, 0, 0, Mc_AN, cnt_start+jj)[cnt_start+ll]);
		        tmpEhub2 = 0.5*(Coulomb_Array[NZUJ][ii][jj][kk][ll]-Coulomb_Array[NZUJ][ii][jj][ll][kk])*
				      (openmx_local_density_row(local_basis, 0, 0, Mc_AN, cnt_start+ii)[cnt_start+kk]*
				       openmx_local_density_row(local_basis, 0, 0, Mc_AN, cnt_start+jj)[cnt_start+ll]
                                      +openmx_local_density_row(local_basis, 0, 1, Mc_AN, cnt_start+ii)[cnt_start+kk]*
                                       openmx_local_density_row(local_basis, 0, 1, Mc_AN, cnt_start+jj)[cnt_start+ll]);
	                My_Ehub += tmpEhub1 + tmpEhub2;
                      break;
              
                      case 2:  /* sAMF */
                        tmpEhub1 = 0.5*Coulomb_Array[NZUJ][ii][jj][kk][ll]*
                                      (AMF_Array[NZUJ][0][0][ii][kk]*
			    	       AMF_Array[NZUJ][1][0][jj][ll]
                                      +AMF_Array[NZUJ][1][0][ii][kk]*
                                       AMF_Array[NZUJ][0][0][jj][ll]);
		        tmpEhub2 = 0.5*(Coulomb_Array[NZUJ][ii][jj][kk][ll]-Coulomb_Array[NZUJ][ii][jj][ll][kk])*
				      (AMF_Array[NZUJ][0][0][ii][kk]*
				       AMF_Array[NZUJ][0][0][jj][ll]
                                      +AMF_Array[NZUJ][1][0][ii][kk]*
                                       AMF_Array[NZUJ][1][0][jj][ll]);
	                My_Ehub += tmpEhub1 + tmpEhub2;
                      break;

                      case 3:  /* cFLL */
		        tmpEhub1 = 0.5*Coulomb_Array[NZUJ][ii][jj][kk][ll]*
                                      (openmx_local_density_row(local_basis, 0, 0, Mc_AN, cnt_start+ii)[cnt_start+kk]*
                                       openmx_local_density_row(local_basis, 0, 1, Mc_AN, cnt_start+jj)[cnt_start+ll]
                                      +openmx_local_density_row(local_basis, 0, 1, Mc_AN, cnt_start+ii)[cnt_start+kk]*
                                       openmx_local_density_row(local_basis, 0, 0, Mc_AN, cnt_start+jj)[cnt_start+ll]);
		        tmpEhub2 = 0.5*(Coulomb_Array[NZUJ][ii][jj][kk][ll]-Coulomb_Array[NZUJ][ii][jj][ll][kk])*
				      (openmx_local_density_row(local_basis, 0, 0, Mc_AN, cnt_start+ii)[cnt_start+kk]*
				       openmx_local_density_row(local_basis, 0, 0, Mc_AN, cnt_start+jj)[cnt_start+ll]
                                      +openmx_local_density_row(local_basis, 0, 1, Mc_AN, cnt_start+ii)[cnt_start+kk]*
                                       openmx_local_density_row(local_basis, 0, 1, Mc_AN, cnt_start+jj)[cnt_start+ll]);
	                My_Ehub += tmpEhub1 + tmpEhub2;
                      break;

                      case 4:  /* cAMF */
                        tmpEhub1 = 0.5*Coulomb_Array[NZUJ][ii][jj][kk][ll]*
                                      (AMF_Array[NZUJ][0][0][ii][kk]*
			    	       AMF_Array[NZUJ][1][0][jj][ll]
                                      +AMF_Array[NZUJ][1][0][ii][kk]*
                                       AMF_Array[NZUJ][0][0][jj][ll]);
		        tmpEhub2 = 0.5*(Coulomb_Array[NZUJ][ii][jj][kk][ll]-Coulomb_Array[NZUJ][ii][jj][ll][kk])*
				      (AMF_Array[NZUJ][0][0][ii][kk]*
				       AMF_Array[NZUJ][0][0][jj][ll]
                                      +AMF_Array[NZUJ][1][0][ii][kk]*
                                       AMF_Array[NZUJ][1][0][jj][ll]);
	                My_Ehub += tmpEhub1 + tmpEhub2;
                      break;

                /*      case 3:  
		        tmpEhub1 = 0.5*dc_alpha[count]*(Coulomb_Array[count][ii][jj][kk][ll]*
                                      (DM_onsite[0][0][Mc_AN][cnt_start+ii][cnt_start+kk]*
			    	       DM_onsite[0][1][Mc_AN][cnt_start+jj][cnt_start+ll]
                                      +DM_onsite[0][1][Mc_AN][cnt_start+ii][cnt_start+kk]*
                                       DM_onsite[0][0][Mc_AN][cnt_start+jj][cnt_start+ll]));
		        tmpEhub2 = 0.5*dc_alpha[count]*((Coulomb_Array[count][ii][jj][kk][ll]-Coulomb_Array[count][ii][jj][ll][kk])*
				      (DM_onsite[0][0][Mc_AN][cnt_start+ii][cnt_start+kk]*
				       DM_onsite[0][0][Mc_AN][cnt_start+jj][cnt_start+ll]
                                      +DM_onsite[0][1][Mc_AN][cnt_start+ii][cnt_start+kk]*
                                       DM_onsite[0][1][Mc_AN][cnt_start+jj][cnt_start+ll]));
                        tmpEhub3 = 0.5*(1.0-dc_alpha[count])*(Coulomb_Array[count][ii][jj][kk][ll]*
                                      (AMF_Array[count][0][0][ii][kk]*
			    	       AMF_Array[count][1][0][jj][ll]
                                      +AMF_Array[count][1][0][ii][kk]*
                                       AMF_Array[count][0][0][jj][ll]));
		        tmpEhub4 = 0.5*(1.0-dc_alpha[count])*((Coulomb_Array[count][ii][jj][kk][ll]-Coulomb_Array[count][ii][jj][ll][kk])*
				      (AMF_Array[count][0][0][ii][kk]*
				       AMF_Array[count][0][0][jj][ll]
                                      +AMF_Array[count][1][0][ii][kk]*
                                       AMF_Array[count][1][0][jj][ll]));
                        My_Ehub += tmpEhub1 + tmpEhub2 + tmpEhub3 + tmpEhub4;
                      break; */
                      } /* switch dc_Type */
		    }
		  }
	        }
	      }      
	      
            }
	  }	/* mul1 */
	}	/* l1 */
      break;
 
      } /* Hub_Type */
    } /* SpinP_switch */ 

   /****************************************************
                     non-collinear case
    ****************************************************/

    else {
    
      switch (Hub_Type){ 
      case 1:	/* Dudarev form */
        /* Hubbard term, 0.5*Tr(N) */

        cnt1 = 0;
        for(l1=0; l1<=Spe_MaxL_Basis[wan1]; l1++ ){
  	  for(mul1=0; mul1<Spe_Num_Basis[wan1][l1]; mul1++){

	    Uvalue = Hub_U_Basis[wan1][l1][mul1];
	    for(m1=0; m1<(2*l1+1); m1++){

	      tmpv = 0.5*Uvalue*( openmx_local_occupation_row(local_basis, 0, 0, 0, Mc_AN, cnt1)[cnt1].r
			      + openmx_local_occupation_row(local_basis, 0, 1, 1, Mc_AN, cnt1)[cnt1].r);
              My_Ehub += tmpv;

	      cnt1++;
	    }
	  }
        }

        /* Hubbard term, -0.5*Tr(N*N) */

        cnt1 = 0;
        for(l1=0; l1<=Spe_MaxL_Basis[wan1]; l1++ ){
 	  for(mul1=0; mul1<Spe_Num_Basis[wan1][l1]; mul1++){
	    for(m1=0; m1<(2*l1+1); m1++){

              sum = 0.0;  

	      cnt2 = 0;
	      for(l2=0; l2<=Spe_MaxL_Basis[wan1]; l2++ ){
	        for(mul2=0; mul2<Spe_Num_Basis[wan1][l2]; mul2++){
		  for(m2=0; m2<(2*l2+1); m2++){

		    if (l1==l2 && mul1==mul2){

		      Uvalue = Hub_U_Basis[wan1][l1][mul1];

		      sum -= 0.5*Uvalue*( openmx_local_occupation_row(local_basis, 0, 0, 0, Mc_AN, cnt1)[cnt2].r*
			                openmx_local_occupation_row(local_basis, 0, 0, 0, Mc_AN, cnt1)[cnt2].r
					    +
				        openmx_local_occupation_row(local_basis, 0, 0, 0, Mc_AN, cnt1)[cnt2].i*
					openmx_local_occupation_row(local_basis, 0, 0, 0, Mc_AN, cnt1)[cnt2].i
					    +
				        openmx_local_occupation_row(local_basis, 0, 0, 1, Mc_AN, cnt1)[cnt2].r*
					openmx_local_occupation_row(local_basis, 0, 0, 1, Mc_AN, cnt1)[cnt2].r
					    +
					openmx_local_occupation_row(local_basis, 0, 0, 1, Mc_AN, cnt1)[cnt2].i*
					openmx_local_occupation_row(local_basis, 0, 0, 1, Mc_AN, cnt1)[cnt2].i
					    +
					openmx_local_occupation_row(local_basis, 0, 1, 0, Mc_AN, cnt1)[cnt2].r*
					openmx_local_occupation_row(local_basis, 0, 1, 0, Mc_AN, cnt1)[cnt2].r
					    +
					openmx_local_occupation_row(local_basis, 0, 1, 0, Mc_AN, cnt1)[cnt2].i*
					openmx_local_occupation_row(local_basis, 0, 1, 0, Mc_AN, cnt1)[cnt2].i
					    +
					openmx_local_occupation_row(local_basis, 0, 1, 1, Mc_AN, cnt1)[cnt2].r*
					openmx_local_occupation_row(local_basis, 0, 1, 1, Mc_AN, cnt1)[cnt2].r
					    +
					openmx_local_occupation_row(local_basis, 0, 1, 1, Mc_AN, cnt1)[cnt2].i*
					openmx_local_occupation_row(local_basis, 0, 1, 1, Mc_AN, cnt1)[cnt2].i );

		    }

		    cnt2++;
		  }
	        }
	      }

              My_Ehub += sum;

	      cnt1++;
	    } /* m1 */
	  } /* mul1 */
        } /* l1 */
      break;

      case 2:	/* general form by S.Ryee */

        /* U Energy */
        for(l1=0; l1<=Spe_MaxL_Basis[wan1]; l1++ ){
          for(mul1=0; mul1<Spe_Num_Basis[wan1][l1]; mul1++){
           Uvalue = Hub_U_Basis[wan1][l1][mul1];
           Jvalue = Hund_J_Basis[wan1][l1][mul1];
           NZUJ = Nonzero_UJ[wan1][l1][mul1];
           if(NZUJ>0){
            cnt_start = 0;
            switch (mul1){
            case 0:   /* mul1 = 0 */
              if(l1 > 0){
                for(tmp_l1=0; tmp_l1<l1; tmp_l1++){
                  cnt_start += (2*tmp_l1+1)*Spe_Num_Basis[wan1][tmp_l1];
                }
              }
              else{   /* l1 = 0 */
                cnt_start = 0;
              }
            break;

            case 1:   /* mul1 = 1 */
              if(l1 > 0){
                for(tmp_l1=0; tmp_l1<l1; tmp_l1++){
                  cnt_start += (2*tmp_l1+1)*Spe_Num_Basis[wan1][tmp_l1];
                }
                cnt_start += (2*l1+1)*mul1;
              }
              else{   /* l1 = 0 */
                cnt_start = (2*l1+1)*mul1;
              }
            break;
            } /* switch (mul1) */


	    trace_N00.r = 0.0;
            trace_N00.i = 0.0;
	    trace_N11.r = 0.0;
	    trace_N11.i = 0.0;
            trace_N01.r = 0.0;
            trace_N01.i = 0.0;
            trace_N10.r = 0.0;
            trace_N10.i = 0.0;

            for(dd=0; dd<(2*l1+1); dd++){
              trace_N00.r += openmx_local_occupation_row(local_basis, 0, 0, 0, Mc_AN, cnt_start+dd)[cnt_start+dd].r;
	      trace_N00.i += openmx_local_occupation_row(local_basis, 0, 0, 0, Mc_AN, cnt_start+dd)[cnt_start+dd].i;
	      trace_N11.r += openmx_local_occupation_row(local_basis, 0, 1, 1, Mc_AN, cnt_start+dd)[cnt_start+dd].r;
	      trace_N11.i += openmx_local_occupation_row(local_basis, 0, 1, 1, Mc_AN, cnt_start+dd)[cnt_start+dd].i;

              trace_N01.r += openmx_local_occupation_row(local_basis, 0, 0, 1, Mc_AN, cnt_start+dd)[cnt_start+dd].r;
              trace_N01.i += openmx_local_occupation_row(local_basis, 0, 0, 1, Mc_AN, cnt_start+dd)[cnt_start+dd].i;
              trace_N10.r += openmx_local_occupation_row(local_basis, 0, 1, 0, Mc_AN, cnt_start+dd)[cnt_start+dd].r;
              trace_N10.i += openmx_local_occupation_row(local_basis, 0, 1, 0, Mc_AN, cnt_start+dd)[cnt_start+dd].i;
            }
	    /* Double counting energy */
	    if(dc_Type==1){  /* sFLL */
	      My_Ehub -= 0.5*(Uvalue)*(trace_N00.r+trace_N11.r)*(trace_N00.r+trace_N11.r-1.0)
		        -0.5*(Jvalue)*(Cmul(trace_N00,Csub(trace_N00,Complex(1.0,0.0))).r
				    +Cmul(trace_N11,Csub(trace_N11,Complex(1.0,0.0))).r);
	      My_Ehub -=-0.5*(Jvalue)*(Cmul(trace_N01,trace_N10).r + Cmul(trace_N10,trace_N01).r);
	
	    }  /* sFLL */

            if(dc_Type==3){ /* cFLL */
              My_Ehub -= 0.5*(Uvalue)*(trace_N00.r+trace_N11.r)*(trace_N00.r+trace_N11.r-1.0)
		        -0.25*(Jvalue)*(trace_N00.r+trace_N11.r)*(trace_N00.r+trace_N11.r-2.0);
	    }  /* cFLL */


          /*  if(dc_Type==3){  
	      My_Ehub -= dc_alpha[count]*(0.5*(Uvalue)*(trace_N00.r+trace_N11.r)*(trace_N00.r+trace_N11.r-1.0)
		                  -0.5*(Jvalue)*(Cmul(trace_N00,Csub(trace_N00,Complex(1.0,0.0))).r
				  +Cmul(trace_N11,Csub(trace_N11,Complex(1.0,0.0))).r));
	      My_Ehub -=-dc_alpha[count]*(0.5*(Jvalue)*(Cmul(trace_N01,trace_N10).r + Cmul(trace_N10,trace_N01).r));
            }  */

            /* loop start for interaction energy */
            for(ii=0; ii<(2*l1+1); ii++){
    	      for(jj=0; jj<(2*l1+1); jj++){
	        for(kk=0; kk<(2*l1+1); kk++){
	          for(ll=0; ll<(2*l1+1); ll++){
                    switch(dc_Type){
                    case 1:  /* sFLL */
		      N_00_ac.r=openmx_local_occupation_row(local_basis, 0, 0, 0, Mc_AN, cnt_start+ii)[cnt_start+kk].r;
		      N_00_ac.i=openmx_local_occupation_row(local_basis, 0, 0, 0, Mc_AN, cnt_start+ii)[cnt_start+kk].i;
		      N_11_ac.r=openmx_local_occupation_row(local_basis, 0, 1, 1, Mc_AN, cnt_start+ii)[cnt_start+kk].r;
		      N_11_ac.i=openmx_local_occupation_row(local_basis, 0, 1, 1, Mc_AN, cnt_start+ii)[cnt_start+kk].i;

		      N_00_bd.r=openmx_local_occupation_row(local_basis, 0, 0, 0, Mc_AN, cnt_start+jj)[cnt_start+ll].r;
		      N_00_bd.i=openmx_local_occupation_row(local_basis, 0, 0, 0, Mc_AN, cnt_start+jj)[cnt_start+ll].i;
		      N_11_bd.r=openmx_local_occupation_row(local_basis, 0, 1, 1, Mc_AN, cnt_start+jj)[cnt_start+ll].r;
		      N_11_bd.i=openmx_local_occupation_row(local_basis, 0, 1, 1, Mc_AN, cnt_start+jj)[cnt_start+ll].i;

		      N_01_ac.r=openmx_local_occupation_row(local_basis, 0, 0, 1, Mc_AN, cnt_start+ii)[cnt_start+kk].r;
		      N_01_ac.i=openmx_local_occupation_row(local_basis, 0, 0, 1, Mc_AN, cnt_start+ii)[cnt_start+kk].i;
		      N_10_ac.r=openmx_local_occupation_row(local_basis, 0, 1, 0, Mc_AN, cnt_start+ii)[cnt_start+kk].r;
		      N_10_ac.i=openmx_local_occupation_row(local_basis, 0, 1, 0, Mc_AN, cnt_start+ii)[cnt_start+kk].i;

		      N_01_bd.r=openmx_local_occupation_row(local_basis, 0, 0, 1, Mc_AN, cnt_start+jj)[cnt_start+ll].r;
		      N_01_bd.i=openmx_local_occupation_row(local_basis, 0, 0, 1, Mc_AN, cnt_start+jj)[cnt_start+ll].i;
		      N_10_bd.r=openmx_local_occupation_row(local_basis, 0, 1, 0, Mc_AN, cnt_start+jj)[cnt_start+ll].r;
		      N_10_bd.i=openmx_local_occupation_row(local_basis, 0, 1, 0, Mc_AN, cnt_start+jj)[cnt_start+ll].i;

                      /* diagonal term */
   		      tmpEhub1 = 0.5*Coulomb_Array[NZUJ][ii][jj][kk][ll]*
                                    (Cmul(N_00_ac,N_00_bd).r + Cmul(N_11_ac,N_11_bd).r
			            +Cmul(N_00_ac,N_11_bd).r + Cmul(N_11_ac,N_00_bd).r)
			        -0.5*Coulomb_Array[NZUJ][ii][jj][ll][kk]*
                                    (Cmul(N_00_ac,N_00_bd).r + Cmul(N_11_ac,N_11_bd).r);
		      /* off-diagonal term */
		      tmpEhub2 = -0.5*Coulomb_Array[NZUJ][ii][jj][ll][kk]*
                                     (Cmul(N_01_ac,N_10_bd).r + Cmul(N_10_ac,N_01_bd).r);
		      /* LDA+U energy */
		      My_Ehub += tmpEhub1 + tmpEhub2;	
                    break;
                   
                    case 2:  /* sAMF */
                      N_00_ac.r=AMF_Array[NZUJ][0][0][ii][kk];
		      N_00_ac.i=AMF_Array[NZUJ][0][1][ii][kk];
		      N_11_ac.r=AMF_Array[NZUJ][1][0][ii][kk];
		      N_11_ac.i=AMF_Array[NZUJ][1][1][ii][kk];

		      N_00_bd.r=AMF_Array[NZUJ][0][0][jj][ll];
		      N_00_bd.i=AMF_Array[NZUJ][0][1][jj][ll];
		      N_11_bd.r=AMF_Array[NZUJ][1][0][jj][ll];
		      N_11_bd.i=AMF_Array[NZUJ][1][1][jj][ll];

		      N_01_ac.r=AMF_Array[NZUJ][2][0][ii][kk];
		      N_01_ac.i=AMF_Array[NZUJ][2][1][ii][kk];
		      N_10_ac.r=AMF_Array[NZUJ][3][0][ii][kk];
		      N_10_ac.i=AMF_Array[NZUJ][3][1][ii][kk];

		      N_01_bd.r=AMF_Array[NZUJ][2][0][jj][ll];
		      N_01_bd.i=AMF_Array[NZUJ][2][1][jj][ll];
		      N_10_bd.r=AMF_Array[NZUJ][3][0][jj][ll];
		      N_10_bd.i=AMF_Array[NZUJ][3][1][jj][ll];

                      /* diagonal term */
		      tmpEhub1 = 0.5*Coulomb_Array[NZUJ][ii][jj][kk][ll]*
                                    (Cmul(N_00_ac,N_00_bd).r + Cmul(N_11_ac,N_11_bd).r
			            +Cmul(N_00_ac,N_11_bd).r + Cmul(N_11_ac,N_00_bd).r)
			        -0.5*Coulomb_Array[NZUJ][ii][jj][ll][kk]*
                                    (Cmul(N_00_ac,N_00_bd).r + Cmul(N_11_ac,N_11_bd).r);
		      /* off-diagonal term */
		      tmpEhub2 = -0.5*Coulomb_Array[NZUJ][ii][jj][ll][kk]*
                                     (Cmul(N_01_ac,N_10_bd).r + Cmul(N_10_ac,N_01_bd).r);
		      /* LDA+U energy */
		      My_Ehub += tmpEhub1 + tmpEhub2;	
                    break;

                    case 3:  /* cFLL */
		      N_00_ac.r=openmx_local_occupation_row(local_basis, 0, 0, 0, Mc_AN, cnt_start+ii)[cnt_start+kk].r;
		      N_00_ac.i=openmx_local_occupation_row(local_basis, 0, 0, 0, Mc_AN, cnt_start+ii)[cnt_start+kk].i;
		      N_11_ac.r=openmx_local_occupation_row(local_basis, 0, 1, 1, Mc_AN, cnt_start+ii)[cnt_start+kk].r;
		      N_11_ac.i=openmx_local_occupation_row(local_basis, 0, 1, 1, Mc_AN, cnt_start+ii)[cnt_start+kk].i;

		      N_00_bd.r=openmx_local_occupation_row(local_basis, 0, 0, 0, Mc_AN, cnt_start+jj)[cnt_start+ll].r;
		      N_00_bd.i=openmx_local_occupation_row(local_basis, 0, 0, 0, Mc_AN, cnt_start+jj)[cnt_start+ll].i;
		      N_11_bd.r=openmx_local_occupation_row(local_basis, 0, 1, 1, Mc_AN, cnt_start+jj)[cnt_start+ll].r;
		      N_11_bd.i=openmx_local_occupation_row(local_basis, 0, 1, 1, Mc_AN, cnt_start+jj)[cnt_start+ll].i;

		      N_01_ac.r=openmx_local_occupation_row(local_basis, 0, 0, 1, Mc_AN, cnt_start+ii)[cnt_start+kk].r;
		      N_01_ac.i=openmx_local_occupation_row(local_basis, 0, 0, 1, Mc_AN, cnt_start+ii)[cnt_start+kk].i;
		      N_10_ac.r=openmx_local_occupation_row(local_basis, 0, 1, 0, Mc_AN, cnt_start+ii)[cnt_start+kk].r;
		      N_10_ac.i=openmx_local_occupation_row(local_basis, 0, 1, 0, Mc_AN, cnt_start+ii)[cnt_start+kk].i;

		      N_01_bd.r=openmx_local_occupation_row(local_basis, 0, 0, 1, Mc_AN, cnt_start+jj)[cnt_start+ll].r;
		      N_01_bd.i=openmx_local_occupation_row(local_basis, 0, 0, 1, Mc_AN, cnt_start+jj)[cnt_start+ll].i;
		      N_10_bd.r=openmx_local_occupation_row(local_basis, 0, 1, 0, Mc_AN, cnt_start+jj)[cnt_start+ll].r;
		      N_10_bd.i=openmx_local_occupation_row(local_basis, 0, 1, 0, Mc_AN, cnt_start+jj)[cnt_start+ll].i;

                      /* diagonal term */
   		      tmpEhub1 = 0.5*Coulomb_Array[NZUJ][ii][jj][kk][ll]*
                                    (Cmul(N_00_ac,N_00_bd).r + Cmul(N_11_ac,N_11_bd).r
			            +Cmul(N_00_ac,N_11_bd).r + Cmul(N_11_ac,N_00_bd).r)
			        -0.5*Coulomb_Array[NZUJ][ii][jj][ll][kk]*
                                    (Cmul(N_00_ac,N_00_bd).r + Cmul(N_11_ac,N_11_bd).r);
		      /* off-diagonal term */
		      tmpEhub2 = -0.5*Coulomb_Array[NZUJ][ii][jj][ll][kk]*
                                     (Cmul(N_01_ac,N_10_bd).r + Cmul(N_10_ac,N_01_bd).r);
		      /* LDA+U energy */
		      My_Ehub += tmpEhub1 + tmpEhub2;	
                    break;
 
                    case 4:  /* cAMF */
                      N_00_ac.r=AMF_Array[NZUJ][0][0][ii][kk];
		      N_00_ac.i=AMF_Array[NZUJ][0][1][ii][kk];
		      N_11_ac.r=AMF_Array[NZUJ][1][0][ii][kk];
		      N_11_ac.i=AMF_Array[NZUJ][1][1][ii][kk];

		      N_00_bd.r=AMF_Array[NZUJ][0][0][jj][ll];
		      N_00_bd.i=AMF_Array[NZUJ][0][1][jj][ll];
		      N_11_bd.r=AMF_Array[NZUJ][1][0][jj][ll];
		      N_11_bd.i=AMF_Array[NZUJ][1][1][jj][ll];

		      N_01_ac.r=AMF_Array[NZUJ][2][0][ii][kk];
		      N_01_ac.i=AMF_Array[NZUJ][2][1][ii][kk];
		      N_10_ac.r=AMF_Array[NZUJ][3][0][ii][kk];
		      N_10_ac.i=AMF_Array[NZUJ][3][1][ii][kk];

		      N_01_bd.r=AMF_Array[NZUJ][2][0][jj][ll];
		      N_01_bd.i=AMF_Array[NZUJ][2][1][jj][ll];
		      N_10_bd.r=AMF_Array[NZUJ][3][0][jj][ll];
		      N_10_bd.i=AMF_Array[NZUJ][3][1][jj][ll];

                      /* diagonal term */
		      tmpEhub1 = 0.5*Coulomb_Array[NZUJ][ii][jj][kk][ll]*
                                    (Cmul(N_00_ac,N_00_bd).r + Cmul(N_11_ac,N_11_bd).r
			            +Cmul(N_00_ac,N_11_bd).r + Cmul(N_11_ac,N_00_bd).r)
			        -0.5*Coulomb_Array[NZUJ][ii][jj][ll][kk]*
                                    (Cmul(N_00_ac,N_00_bd).r + Cmul(N_11_ac,N_11_bd).r);
		      /* off-diagonal term */
		      tmpEhub2 = -0.5*Coulomb_Array[NZUJ][ii][jj][ll][kk]*
                                     (Cmul(N_01_ac,N_10_bd).r + Cmul(N_10_ac,N_01_bd).r);
		      /* LDA+U energy */
		      My_Ehub += tmpEhub1 + tmpEhub2;	
                    break;
                /*    case 3:  
		      N_00_ac.r=NC_OcpN[0][0][0][Mc_AN][cnt_start+ii][cnt_start+kk].r;
		      N_00_ac.i=NC_OcpN[0][0][0][Mc_AN][cnt_start+ii][cnt_start+kk].i;
		      N_11_ac.r=NC_OcpN[0][1][1][Mc_AN][cnt_start+ii][cnt_start+kk].r;
		      N_11_ac.i=NC_OcpN[0][1][1][Mc_AN][cnt_start+ii][cnt_start+kk].i;

		      N_00_bd.r=NC_OcpN[0][0][0][Mc_AN][cnt_start+jj][cnt_start+ll].r;
		      N_00_bd.i=NC_OcpN[0][0][0][Mc_AN][cnt_start+jj][cnt_start+ll].i;
		      N_11_bd.r=NC_OcpN[0][1][1][Mc_AN][cnt_start+jj][cnt_start+ll].r;
		      N_11_bd.i=NC_OcpN[0][1][1][Mc_AN][cnt_start+jj][cnt_start+ll].i;

		      N_01_ac.r=NC_OcpN[0][0][1][Mc_AN][cnt_start+ii][cnt_start+kk].r;
		      N_01_ac.i=NC_OcpN[0][0][1][Mc_AN][cnt_start+ii][cnt_start+kk].i;
		      N_10_ac.r=NC_OcpN[0][1][0][Mc_AN][cnt_start+ii][cnt_start+kk].r;
		      N_10_ac.i=NC_OcpN[0][1][0][Mc_AN][cnt_start+ii][cnt_start+kk].i;

		      N_01_bd.r=NC_OcpN[0][0][1][Mc_AN][cnt_start+jj][cnt_start+ll].r;
		      N_01_bd.i=NC_OcpN[0][0][1][Mc_AN][cnt_start+jj][cnt_start+ll].i;
		      N_10_bd.r=NC_OcpN[0][1][0][Mc_AN][cnt_start+jj][cnt_start+ll].r;
		      N_10_bd.i=NC_OcpN[0][1][0][Mc_AN][cnt_start+jj][cnt_start+ll].i;

                      AMF_00_ac.r=AMF_Array[count][0][0][ii][kk];
		      AMF_00_ac.i=AMF_Array[count][0][1][ii][kk];
		      AMF_11_ac.r=AMF_Array[count][1][0][ii][kk];
		      AMF_11_ac.i=AMF_Array[count][1][1][ii][kk];

		      AMF_00_bd.r=AMF_Array[count][0][0][jj][ll];
		      AMF_00_bd.i=AMF_Array[count][0][1][jj][ll];
		      AMF_11_bd.r=AMF_Array[count][1][0][jj][ll];
		      AMF_11_bd.i=AMF_Array[count][1][1][jj][ll];

		      AMF_01_ac.r=AMF_Array[count][2][0][ii][kk];
		      AMF_01_ac.i=AMF_Array[count][2][1][ii][kk];
		      AMF_10_ac.r=AMF_Array[count][3][0][ii][kk];
		      AMF_10_ac.i=AMF_Array[count][3][1][ii][kk];

		      AMF_01_bd.r=AMF_Array[count][2][0][jj][ll];
		      AMF_01_bd.i=AMF_Array[count][2][1][jj][ll];
		      AMF_10_bd.r=AMF_Array[count][3][0][jj][ll];
		      AMF_10_bd.i=AMF_Array[count][3][1][jj][ll];

		      tmpEhub1 = 0.5*Coulomb_Array[count][ii][jj][kk][ll]*
                                    (Cmul(N_00_ac,N_00_bd).r + Cmul(N_11_ac,N_11_bd).r
			            +Cmul(N_00_ac,N_11_bd).r + Cmul(N_11_ac,N_00_bd).r)
			        -0.5*Coulomb_Array[count][ii][jj][ll][kk]*
                                    (Cmul(N_00_ac,N_00_bd).r + Cmul(N_11_ac,N_11_bd).r);
		      tmpEhub2 = -0.5*Coulomb_Array[count][ii][jj][ll][kk]*
                                     (Cmul(N_01_ac,N_10_bd).r + Cmul(N_10_ac,N_01_bd).r);

		      tmpEhub3 = 0.5*Coulomb_Array[count][ii][jj][kk][ll]*
                                    (Cmul(AMF_00_ac,AMF_00_bd).r + Cmul(AMF_11_ac,AMF_11_bd).r
			            +Cmul(AMF_00_ac,AMF_11_bd).r + Cmul(AMF_11_ac,AMF_00_bd).r)
			        -0.5*Coulomb_Array[count][ii][jj][ll][kk]*
                                    (Cmul(AMF_00_ac,AMF_00_bd).r + Cmul(AMF_11_ac,AMF_11_bd).r);
		      tmpEhub4 = -0.5*Coulomb_Array[count][ii][jj][ll][kk]*
                                     (Cmul(AMF_01_ac,AMF_10_bd).r + Cmul(AMF_10_ac,AMF_01_bd).r);

                      My_Ehub += dc_alpha[count]*(tmpEhub1+tmpEhub2) + (1.0-dc_alpha[count])*(tmpEhub3+tmpEhub4);
                    break; */
		    } /* dc switch */
		  }
		}
	      }
	    }
           } /* Uvalue != 0.0 || Jvalue != 0.0 */

          } /* mul1 */
        } /* l1 */

      break;
      } /* Hub_Type */

    } /* SpinP_switch */
  
  } /* Mc_AN */
 

  if (SpinP_switch==0) My_Ehub = 2.0*My_Ehub;

 /****************************************************
                      MPI My_Ehub
  ****************************************************/

  MPI_Allreduce(&My_Ehub, &Ehub, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

  /* if (F_U_flag==0) */
  if (F_U_flag==0) Ehub = 0.0;
 
  return Ehub;  
}







/* okuno */
double Calc_EdftD()
{
  /************************************************
     The subroutine calculates the semiemprical 
     vdW correction to DFT-GGA proposed by 
     S. Grimme, J. Comput. Chem. 27, 1787 (2006).
  *************************************************/

  double My_EdftD,EdftD;
  double rij[4],fdamp,fdamp2;
  double rij0[4],par;
  double dist,dist6,dist2;
  double exparg,expval;
  double rcut_dftD2;
  int numprocs,myid,ID;
  int Mc_AN,Gc_AN,wanA,wanB;
  int Gc_BN;
  int nrm,nr;
  int i,j,k;
  int n1,n2,n3;
  int per_flag1,per_flag2;
  int n1_max,n2_max,n3_max; 
  double test_ene;
  double dblcnt_factor;
  double E,dEx,dEy,dEz,dist7;
  double my_st[9],st[9];

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  My_EdftD = 0.0;
  EdftD    = 0.0;
  rcut_dftD2 = rcut_dftD*rcut_dftD;

  dblcnt_factor = 0.5;

  for (i=0; i<9; i++) my_st[i] = 0.0;

  /* here we calculate DFT-D dispersion energy */
  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

    E = 0.0;
    dEx = 0.0;
    dEy = 0.0;
    dEz = 0.0;

    Gc_AN = M2G[Mc_AN];
    wanA = WhatSpecies[Gc_AN];
    per_flag1 = (int)Gxyz[Gc_AN][60]; 

    for(Gc_BN=1; Gc_BN<=atomnum; Gc_BN++){

      wanB = WhatSpecies[Gc_BN];
      per_flag2 = (int)Gxyz[Gc_BN][60]; 

      rij0[1] = Gxyz[Gc_AN][1] - Gxyz[Gc_BN][1];
      rij0[2] = Gxyz[Gc_AN][2] - Gxyz[Gc_BN][2];
      rij0[3] = Gxyz[Gc_AN][3] - Gxyz[Gc_BN][3];

      par = beta_dftD/(Rsum_dftD[wanA][wanB]);

      if (per_flag1==0 && per_flag2==0){
        n1_max = 0;
        n2_max = 0;
        n3_max = 0;
      }
      else if (per_flag1==0 && per_flag2==1){
        n1_max = n1_DFT_D;
        n2_max = n2_DFT_D;
        n3_max = n3_DFT_D;
      }
      else if (per_flag1==1 && per_flag2==0){
        n1_max = 0;
        n2_max = 0;
        n3_max = 0;
      }
      else if (per_flag1==1 && per_flag2==1){
        n1_max = n1_DFT_D;
        n2_max = n2_DFT_D;
        n3_max = n3_DFT_D;
      }

      /*
      printf("Gc_AN=%2d Gc_BN=%2d %2d %2d %2d %2d %2d\n",Gc_AN,Gc_BN,per_flag1,per_flag2,n1_max,n2_max,n3_max);
      */

      for (n1=-n1_max; n1<=n1_max; n1++){
	for (n2=-n2_max; n2<=n2_max; n2++){
	  for (n3=-n3_max; n3<=n3_max; n3++){
            
            /* for double counting */
            if((!((abs(n1)+abs(n2)+abs(n3))==0)) && (per_flag1==0 && per_flag2==1) ){
	      dblcnt_factor = 1.0;
	    }
            else{
	      dblcnt_factor = 0.5;
            }

	    rij[1] = rij0[1] - ( (double)n1*tv[1][1]
	                       + (double)n2*tv[2][1] 
	                       + (double)n3*tv[3][1] ); 

	    rij[2] = rij0[2] - ( (double)n1*tv[1][2]
			       + (double)n2*tv[2][2] 
			       + (double)n3*tv[3][2] ); 

	    rij[3] = rij0[3] - ( (double)n1*tv[1][3]
			       + (double)n2*tv[2][3] 
			       + (double)n3*tv[3][3] ); 

            dist2 = rij[1]*rij[1] + rij[2]*rij[2] + rij[3]*rij[3];

            if (0.1<dist2 && dist2<=rcut_dftD2){

	      dist  = sqrt(dist2); 
	      dist6 = dist2*dist2*dist2;            

	      /* calculate the vdW energy */
	      exparg = -beta_dftD*((dist/Rsum_dftD[wanA][wanB])-1.0);
	      expval = exp(exparg);
	      fdamp = scal6_dftD/(1.0+expval);

	      E -= dblcnt_factor*C6ij_dftD[wanA][wanB]/dist6*fdamp;

	      /* calculate the gradient of the vdW energy */

              dist7 = dist6 * dist;
	      fdamp2 = C6ij_dftD[wanA][wanB]*fdamp/dist6*(expval*par/(1.0+expval) - 6.0/dist);
              dEx -= fdamp2*rij[1]/dist;
              dEy -= fdamp2*rij[2]/dist;
              dEz -= fdamp2*rij[3]/dist;

	      /* calculate stress tensor */

              k = 0;
              for (i=1; i<=3; i++){
                for (j=1; j<=3; j++){
                  my_st[k] -= 0.5*(fdamp2*rij[i]/dist)*rij[j];
                  k++;
  	        }
	      }
	    }

	  } /* n3 */
	} /* n2 */
      } /* n1 */
    } /* Gc_BN */

    My_EdftD += E;

    /* energy decomposition */

    if (Energy_Decomposition_flag==1){

      DecEvdw[0][Mc_AN][0] = E;
      DecEvdw[1][Mc_AN][0] = E;
    }

    /* gradients from two-body terms */

    Gxyz[Gc_AN][17] += dEx;
    Gxyz[Gc_AN][18] += dEy;
    Gxyz[Gc_AN][19] += dEz;

    /*
    printf("Gc_AN=%2d dEx=%15.12f dEy=%15.12f dEz=%15.12f\n",Gc_AN,dEx,dEy,dEz);
    */

  } /* Mc_AN */

  /* MPI_Allreduce */

  MPI_Allreduce(&My_EdftD, &EdftD, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
  MPI_Allreduce(&my_st[0], &st[0], 9, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

  /* MPI_Bcast, Gxyz[Gc_AN][17-19] */

  for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
    ID = G2ID[Gc_AN];
    MPI_Bcast(&Gxyz[Gc_AN][17], 3, MPI_DOUBLE, ID, mpi_comm_level1);
  }

  /* add the contribution to Stress_Tensor */

  for (i=0; i<9; i++) Stress_Tensor[i] += st[i];

  /*
  if (myid==0){
  printf("ABC stress\n"); 

  double detA,sum;
  double itv[5][5];

  detA = tv[1][1]*tv[2][2]*tv[3][3]+tv[2][1]*tv[3][2]*tv[1][3]+tv[3][1]*tv[1][2]*tv[2][3]
        -tv[1][1]*tv[3][2]*tv[2][3]-tv[3][1]*tv[2][2]*tv[1][3]-tv[2][1]*tv[1][2]*tv[3][3];

  itv[1][1] = (tv[2][2]*tv[3][3] - tv[2][3]*tv[3][2])/detA; 
  itv[1][2] = (tv[1][3]*tv[3][2] - tv[1][2]*tv[3][3])/detA;
  itv[1][3] = (tv[1][2]*tv[2][3] - tv[1][3]*tv[2][2])/detA;

  itv[2][1] = (tv[2][3]*tv[3][1] - tv[2][1]*tv[3][3])/detA; 
  itv[2][2] = (tv[1][1]*tv[3][3] - tv[1][3]*tv[3][1])/detA;
  itv[2][3] = (tv[1][3]*tv[2][1] - tv[1][1]*tv[2][3])/detA;

  itv[3][1] = (tv[2][1]*tv[3][2] - tv[2][2]*tv[3][1])/detA; 
  itv[3][2] = (tv[1][2]*tv[3][1] - tv[1][1]*tv[3][2])/detA;
  itv[3][3] = (tv[1][1]*tv[2][2] - tv[1][2]*tv[2][1])/detA;

  for (i=1; i<=3; i++){
    for (j=1; j<=3; j++){

      sum = 0.0;
      for (k=1; k<=3; k++){
        sum += itv[k][i]*st[3*(k-1)+(j-1)];
      }

      printf("%15.12f ",sum);
    }
    printf("\n"); 
  }
  }
  */


  return EdftD;
}
/* okuno */



/* Ellner */
double Calc_EdftD3(OpenMXSolverInvocation *scratch)
{
  /***********************************************************************
   The subroutine calculates the semiemprical DFTD3 vdW correction
   DFTD3 with zero damping:
    S. Grimme, et al., J. Chem. Phys. 132, 154104 (2010).
   DFTD3 with BJ damping
    A.D. Becke and E.R.Johnson, J. Chem. Phys. 122, 154101 (2005).
    E.R. Johnson and A.D. Becke, J. Chem. Phys. 123, 024101 (2005).
    E.R. Johnson and A D. Becke, J. Chem. Phys. 124, 174104 (2006).
  ************************************************************************/

  /* VARIABLES DECLARATOIN */
  double My_EdftD,EdftD; /* energy */
  double E; /* atomic energy */
  double rij[4],fdamp,fdamp6,fdamp8,t6,t62,t8,t82,dE6,dE8,dEC; /* interaction */
  double rij0[4]; /* positions */
  double dist,dist2,dist5,dist6,dist7,dist8; /**/
  double rcut2, cncut2; /* cutoff values */
  int numprocs,myid,ID; /* MPI */
  int Mc_AN,Gc_AN,Gc_BN,Gc_CN,wanA,wanB,iZ; /* atom counting and species */
  int i,j,k; /* dummy vars */
  int n1,n2,n3,n1_max,n2_max,n3_max; /* PBC */
  double per_flagA, per_flagB, dblcnt_factor; /* double counting */
  double dEx,dEy,dEz; /* gradients*/
  OpenMXDispersionView dispersion;
  double xn, *CN; /* Coordination number */
  double exparg,expval, powarg, powval; /**/
  double Z, W, C6_ref, dAi, dBj, Lij, C6, C8, dZi, dZj, dWi, dWj; /* Gaussian distance C6, C8 parameter */
  double C8C6;
  double dCN_AB; /* d(CN_A)/d(r_AB) of one image, recomputed in the three-body loop */
  double my_st[9],st[9];


  /* MPI AND INITIALIZATION */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);
  n1_max = n1_CN_DFT_D;
  n2_max = n2_CN_DFT_D;
  n3_max = n3_CN_DFT_D;
  My_EdftD = 0.0;
  EdftD    = 0.0;
  rcut2 = rcut_dftD*rcut_dftD;
  cncut2 = cncut_dftD*cncut_dftD;

  for (i=0; i<9; i++) my_st[i] = 0.0;

  /* [d3mem 2026-10-29] dCN_AB/dr_AB is only needed for the owned atom A and is recomputed where it is used (the
     three-body loop), so no (atom, atom, image) storage is requested: image radius 0. With the full radius every
     rank held (atomnum+1)^2 x images doubles, 0.92 GB per rank for 2064 atoms and 27 images. */
  OpenMX_Energy_DispersionPrepare(scratch,atomnum,0,0,0,&dispersion);
  CN=dispersion.coordination;

  /* Compute coordination numbers CN_A by adding an inverse damping function (dCN_AB/dr_AB: three-body loop) */
  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];
    wanA = WhatSpecies[Gc_AN];
    iZ = Spe_WhatAtom[wanA];
    if ( iZ>0 ) {
      xn=0.0;
      for(Gc_BN=1; Gc_BN<=atomnum; Gc_BN++){
	wanB = WhatSpecies[Gc_BN];
	iZ = Spe_WhatAtom[wanB];
	if ( iZ>0 ) {
	  per_flagB = (int)Gxyz[Gc_BN][60];
	  rij0[1] = Gxyz[Gc_AN][1] - Gxyz[Gc_BN][1];
	  rij0[2] = Gxyz[Gc_AN][2] - Gxyz[Gc_BN][2];
	  rij0[3] = Gxyz[Gc_AN][3] - Gxyz[Gc_BN][3];
	  if (per_flagB==0){
	    n1_max = 0;
	    n2_max = 0;
	    n3_max = 0;
	  }
	  else {
	    n1_max = n1_CN_DFT_D;
	    n2_max = n2_CN_DFT_D;
	    n3_max = n3_CN_DFT_D;
	  }
	  for (n1=-n1_max; n1<=n1_max; n1++){
	    for (n2=-n2_max; n2<=n2_max; n2++){
	      for (n3=-n3_max; n3<=n3_max; n3++){
		rij[1] = rij0[1] - ( (double)n1*tv[1][1]
				     + (double)n2*tv[2][1]
				     + (double)n3*tv[3][1] );
		rij[2] = rij0[2] - ( (double)n1*tv[1][2]
				     + (double)n2*tv[2][2]
				     + (double)n3*tv[3][2] );
		rij[3] = rij0[3] - ( (double)n1*tv[1][3]
				     + (double)n2*tv[2][3]
				     + (double)n3*tv[3][3] );

		dist2 = rij[1]*rij[1] + rij[2]*rij[2] + rij[3]*rij[3];

		if (dist2<cncut2 && dist2>0.1){
		  dist  = sqrt(dist2);
		  exparg = -k1_dftD*((rcovab_dftD[wanA][wanB]/dist)-1.0); /* Rsum is scaled by k2 */
		  expval = exp(exparg);
		  fdamp = 1.0/(1.0+expval);
		  xn+=fdamp;
		}
	      } /* n3 */
	    } /* n2 */
	  } /* n1 */
	}
      } /* Gc_BN */
      CN[Gc_AN] = xn;
    }
  } /* Mc_AN */

  /* [d3mpi 2026-10-29] Broadcast CN only (atomnum calls). dCN_AB is no longer stored, and was only read for the
     owned atom A; its element-wise MPI_Bcast took atomnum^2 x images calls (1.2e8 for 2064 atoms and 27 images,
     about 600 s per MD step on 256 ranks in OpenMX 4.0.1). */
  MPI_Barrier(mpi_comm_level1);
  for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
    wanA = WhatSpecies[Gc_AN];
    iZ = Spe_WhatAtom[wanA];
    if ( iZ>0 ) {
      ID = G2ID[Gc_AN];
      MPI_Bcast(&CN[Gc_AN], 1, MPI_DOUBLE, ID, mpi_comm_level1);
    }
  } /* Gc_AN */

  /* Calculate energy and collect gradients of two body terms C_ij*d(f_ij/r_ij)/dr_ij,
     also dCi_ij dCj_ij dEC0_ij needed in gradients of 3 body terms */

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];
    wanA = WhatSpecies[Gc_AN];
    iZ = Spe_WhatAtom[wanA];

    if ( iZ>0 ) {

      dblcnt_factor=0.5;
      dEx = 0.0;
      dEy = 0.0;
      dEz = 0.0;
      E = 0.0;

      per_flagA = (int)Gxyz[Gc_AN][60];
      for(Gc_BN=1; Gc_BN<=atomnum; Gc_BN++){
	wanB = WhatSpecies[Gc_BN];
	iZ = Spe_WhatAtom[wanB];
	if ( iZ>0 ) {
	  (*openmx_dispersion_energy(&dispersion,Gc_AN,Gc_BN))=0.0;
	  per_flagB = (int)Gxyz[Gc_BN][60];
	  rij0[1] = Gxyz[Gc_AN][1] - Gxyz[Gc_BN][1];
	  rij0[2] = Gxyz[Gc_AN][2] - Gxyz[Gc_BN][2];
	  rij0[3] = Gxyz[Gc_AN][3] - Gxyz[Gc_BN][3];
	  /* Calculate C6, C8 coefficient and derivatives with Gaussian-distance (L) */
	  Z = 0.0;
	  W = 0.0;
	  dZi=0.0;
	  dZj=0.0;
	  dWi=0.0;
	  dWj=0.0;
	  /* [d3fix 2026-10-14] log-sum-exp: shift the Gaussian exponents by their maximum so that W never
	     underflows (the original code set C6=0 when W<1e-12, which drops e.g. phosphate-P pairs whose CN
	     lies far above all P reference CNs). Ratios Z/W and dC6/dCN are unchanged by the common shift. */
	  double maxarg_d3fix = -1.0e300;
	  for (i=0; i<maxcn_dftD[wanA]; i++){
	    for (j=0; j<maxcn_dftD[wanB]; j++){
	      if (C6ab_dftD[wanA][wanB][i][j][0]>1.0e-12){
		dAi = CN[Gc_AN] - C6ab_dftD[wanA][wanB][i][j][1];
		dBj = CN[Gc_BN] - C6ab_dftD[wanA][wanB][i][j][2];
		exparg = -k3_dftD*( dAi*dAi + dBj*dBj );
		if (exparg>maxarg_d3fix) maxarg_d3fix = exparg;
	      }
	    }
	  }
	  for (i=0; i<maxcn_dftD[wanA]; i++){
	    for (j=0; j<maxcn_dftD[wanB]; j++){
	      C6_ref=C6ab_dftD[wanA][wanB][i][j][0];
	      if (C6_ref>1.0e-12){
		dAi = CN[Gc_AN] - C6ab_dftD[wanA][wanB][i][j][1];
		dBj = CN[Gc_BN] - C6ab_dftD[wanA][wanB][i][j][2];
		exparg = -k3_dftD*( dAi*dAi + dBj*dBj );
		Lij=exp(exparg - maxarg_d3fix);
		Z += C6_ref*Lij;
		W += Lij;
		dZi+=C6_ref*Lij*2.0*k3_dftD*dAi;
		dZj+=C6_ref*Lij*2.0*k3_dftD*dBj;
		dWi+=Lij*2.0*k3_dftD*dAi;
		dWj+=Lij*2.0*k3_dftD*dBj;
	      }
	    } /* CN_j */
	  } /* CN_i */

	  if (W>1.0e-12){

	    C6 = Z/W;
	    C8 = 3.0*C6*r2r4ab_dftD[wanA][wanB];
            C8C6 = 3.0*r2r4ab_dftD[wanA][wanB];
	    (*openmx_dispersion_c6(&dispersion,Gc_AN,Gc_BN))=((dZi*W)-(dWi*Z))/(W*W);
	  }
	  else{
	    C6 = 0.0;
	    C8 = 0.0;
            C8C6 = 3.0*r2r4ab_dftD[wanA][wanB];
	    (*openmx_dispersion_c6(&dispersion,Gc_AN,Gc_BN))=0.0;
	  }
	  /*  CALCULATE ENERGY AND TWO FIRST PART OF GRADIENTS*/
	  if (per_flagB==0){
	    n1_max = 0;
	    n2_max = 0;
	    n3_max = 0;
	  }
	  else {
	    n1_max = n1_DFT_D;
	    n2_max = n2_DFT_D;
	    n3_max = n3_DFT_D;
	  }
	  for (n1=-n1_max; n1<=n1_max; n1++){
	    for (n2=-n2_max; n2<=n2_max; n2++){
	      for (n3=-n3_max; n3<=n3_max; n3++){

		/* for double counting */

		if((!((abs(n1)+abs(n2)+abs(n3))==0)) && (per_flagA==0 && per_flagB==1) ){
		  dblcnt_factor = 1.0;
		}
		else{
		  dblcnt_factor = 0.5;
		}

		rij[1] = rij0[1] - ( (double)n1*tv[1][1]
				   + (double)n2*tv[2][1]
				   + (double)n3*tv[3][1] );
		rij[2] = rij0[2] - ( (double)n1*tv[1][2]
				   + (double)n2*tv[2][2]
				   + (double)n3*tv[3][2] );
		rij[3] = rij0[3] - ( (double)n1*tv[1][3]
				   + (double)n2*tv[2][3]
				   + (double)n3*tv[3][3] );

		dist2 = rij[1]*rij[1] + rij[2]*rij[2] + rij[3]*rij[3];

		if (0.1<dist2 && dist2<rcut2){

		  dist  = sqrt(dist2);
		  dist5 = dist2*dist2*dist;
		  dist6 = dist2*dist2*dist2;
		  dist7 = dist6*dist;
		  dist8 = dist6*dist2;

		  if (DFTD3_damp_dftD == 1){ /*DFTD3 ZERO DAMPING*/

		    /* calculate the vdW energy of E6 and grad of f6/r6 term*/
		    powarg = dist/(sr6_dftD*r0ab_dftD[wanB][wanA]);
		    powval = pow(powarg,-alp6_dftD);
		    fdamp6 = 1.0/(1.0+6.0*powval);
		    E -= dblcnt_factor*s6_dftD*C6*fdamp6/dist6;
		    dE6=(s6_dftD*C6*fdamp6/dist6)*(6.0/dist)*(-1.0+alp6_dftD*powval*fdamp6);

		    /* calculate the vdW energy of E8 and grad of f8/r8 term*/
		    powarg = dist/(sr8_dftD*r0ab_dftD[wanB][wanA]);
		    powval = pow(powarg,-alp8_dftD);
		    fdamp8 = 1.0/(1.0+6.0*powval);
		    E -= dblcnt_factor*s8_dftD*C8*fdamp8/dist8;
		    dE8=(s8_dftD*C8*fdamp8/dist8)*(2.0/dist)*(-4.0+3.0*alp8_dftD*powval*fdamp8);
		    (*openmx_dispersion_energy(&dispersion,Gc_AN,Gc_BN))+=s6_dftD*fdamp6/dist6+s8_dftD*3.0*r2r4ab_dftD[wanA][wanB]*fdamp8/dist8;

		  } /* END IF ZERO DAMPING */

		  if (DFTD3_damp_dftD == 2){ /*DFTD3 BJ DAMPING*/

		    fdamp = (a1_dftD*sqrt(C8C6)+a2_dftD);
		    fdamp6=fdamp*fdamp*fdamp*fdamp*fdamp*fdamp;
		    fdamp8=fdamp6*fdamp*fdamp;
		    t6=dist6 + fdamp6;
		    t62=t6*t6;
		    t8=dist8 + fdamp8;
		    t82=t8*t8;
		    E -= dblcnt_factor*s6_dftD*C6/t6;
		    dE6=-s6_dftD*C6*6.0*dist5/t62;
		    E -= dblcnt_factor*s8_dftD*C8/t8;
		    dE8=-s8_dftD*C8*8.0*dist7/t82;
		    (*openmx_dispersion_energy(&dispersion,Gc_AN,Gc_BN))+=s6_dftD/t6+s8_dftD*3.0*r2r4ab_dftD[wanA][wanB]/t8;
		  } /* IF BJ DAMPING */

		  dEx -= (dE6+dE8)*rij[1]/dist;
		  dEy -= (dE6+dE8)*rij[2]/dist;
		  dEz -= (dE6+dE8)*rij[3]/dist;

  	          /* calculate stress tensor */

		  k = 0;
		  for (i=1; i<=3; i++){
		    for (j=1; j<=3; j++){
		      my_st[k] -= 0.5*((dE6+dE8)*rij[i]/dist)*rij[j];
		      k++;
		    }
		  }

		} /* if dist2 < rcut */
	      } /* n3 */
	    } /* n2 */
	  } /* n1 */
	}
      } /* Gc_BN */

      My_EdftD += E;

      /* energy decomposition */

      if (Energy_Decomposition_flag==1){

        DecEvdw[0][Mc_AN][0] = E;
        DecEvdw[1][Mc_AN][0] = E;
      }

      /* gradients from two-body terms */

      Gxyz[Gc_AN][17] += dEx;
      Gxyz[Gc_AN][18] += dEy;
      Gxyz[Gc_AN][19] += dEz;

        }

  } /* Mc_AN */

  /*MPI BROADCAST GRADIENTS AND REDUCE ENERGIES - MPI_Barrier(mpi_comm_level1); */

  MPI_Allreduce(&My_EdftD, &EdftD, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

  /* [d3mpi 2026-10-29] Row A of the C6 and energy derivatives is written only by the owner of A and is zero on every
     other rank (the prepare zeroes it), so a row-wise MPI_SUM gives every rank exactly the owner's values. This
     replaces 2 x atomnum^2 element-wise MPI_Bcast calls. CN was broadcast above. */
  for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
    wanA = WhatSpecies[Gc_AN];
    iZ = Spe_WhatAtom[wanA];
    if ( iZ>0 ) {
      MPI_Allreduce(MPI_IN_PLACE, openmx_dispersion_c6(&dispersion,Gc_AN,0), atomnum+1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
      MPI_Allreduce(MPI_IN_PLACE, openmx_dispersion_energy(&dispersion,Gc_AN,0), atomnum+1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
    }
  } /* Gc_AN */

  MPI_Barrier(mpi_comm_level1); /* NOT SURE IF ITS NEEDED! */

  /* Calculate three body terms of gradients */

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

    Gc_AN = M2G[Mc_AN];
    wanA = WhatSpecies[Gc_AN];
    iZ = Spe_WhatAtom[wanA];

    if ( iZ>0 ) {

      dEx = 0.0;
      dEy = 0.0;
      dEz = 0.0;

      for(Gc_BN=1; Gc_BN<=atomnum; Gc_BN++){

	wanB = WhatSpecies[Gc_BN];
	iZ = Spe_WhatAtom[wanB];

	if ( iZ>0 ) {

	  per_flagB = (int)Gxyz[Gc_BN][60];
	  rij0[1] = Gxyz[Gc_AN][1] - Gxyz[Gc_BN][1];
	  rij0[2] = Gxyz[Gc_AN][2] - Gxyz[Gc_BN][2];
	  rij0[3] = Gxyz[Gc_AN][3] - Gxyz[Gc_BN][3];

	  if (per_flagB==0){
	    n1_max = 0;
	    n2_max = 0;
	    n3_max = 0;
	  }
	  else {
	    n1_max = n1_CN_DFT_D;
	    n2_max = n2_CN_DFT_D;
	    n3_max = n3_CN_DFT_D;
	  }

	  for (n1=-n1_max; n1<=n1_max; n1++){
	    for (n2=-n2_max; n2<=n2_max; n2++){
	      for (n3=-n3_max; n3<=n3_max; n3++){

		dEC=0.0;
		rij[1] = rij0[1] - ( (double)n1*tv[1][1]
				   + (double)n2*tv[2][1]
				   + (double)n3*tv[3][1] );
		rij[2] = rij0[2] - ( (double)n1*tv[1][2]
				   + (double)n2*tv[2][2]
				   + (double)n3*tv[3][2] );
		rij[3] = rij0[3] - ( (double)n1*tv[1][3]
				   + (double)n2*tv[2][3]
				   + (double)n3*tv[3][3] );

		dist2 = rij[1]*rij[1] + rij[2]*rij[2] + rij[3]*rij[3];

		if (0.1<dist2 && dist2<cncut2){

		  /* calculate grad of C6 term: dEC=dEC0_ik*dC6_ik*dCN_ij+dEC0_jk*dC6_jk*dCN_ij */

		  dist = sqrt(dist2);

		  /* [d3mem 2026-10-29] dCN_AB/dr_AB with the same expressions as in the CN loop */
		  exparg = -k1_dftD*((rcovab_dftD[wanA][wanB]/dist)-1.0);
		  expval = exp(exparg);
		  fdamp = 1.0/(1.0+expval);
		  dCN_AB=-fdamp*fdamp*expval*k1_dftD*rcovab_dftD[wanA][wanB]/dist2;

		  for(Gc_CN=1; Gc_CN<=atomnum; Gc_CN++){

		    dEC += (*openmx_dispersion_energy(&dispersion,Gc_AN,Gc_CN))*(*openmx_dispersion_c6(&dispersion,Gc_AN,Gc_CN))*dCN_AB;
		    dEC += (*openmx_dispersion_energy(&dispersion,Gc_BN,Gc_CN))*(*openmx_dispersion_c6(&dispersion,Gc_BN,Gc_CN))*dCN_AB;

		  } /* Gc_CN */

		  dEx += dEC*rij[1]/dist;
		  dEy += dEC*rij[2]/dist;
		  dEz += dEC*rij[3]/dist;

   	          /* calculate stress tensor */

		  k = 0;
		  for (i=1; i<=3; i++){
		    for (j=1; j<=3; j++){
		      my_st[k] += 0.5*dEC*rij[i]/dist*rij[j];
		      k++;
		    }
		  }

		} /* if dist2 < cn_thr */

	      } /* n3 */
	    } /* n2 */
	  } /* n1 */
	} /* if ( iZ>0 ) */
      } /* Gc_BN */

      Gxyz[Gc_AN][17] += dEx;
      Gxyz[Gc_AN][18] += dEy;
      Gxyz[Gc_AN][19] += dEz;


    } /* if ( iZ>0 ) */
  } /* Mc_AN */

  /* MPI_Allreduce */

  MPI_Allreduce(&my_st[0], &st[0], 9, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

  /* add the contribution to Stress_Tensor */

  for (i=0; i<9; i++) Stress_Tensor[i] += st[i];

  /* MPI_Bcast, Gxyz[Gc_AN][17-19] */

  for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
    ID = G2ID[Gc_AN];
    MPI_Bcast(&Gxyz[Gc_AN][17], 3, MPI_DOUBLE, ID, mpi_comm_level1);
  }


  /*
  if (myid==0){

    double detA,sum;
    double itv[5][5];

    detA = tv[1][1]*tv[2][2]*tv[3][3]+tv[2][1]*tv[3][2]*tv[1][3]+tv[3][1]*tv[1][2]*tv[2][3]
          -tv[1][1]*tv[3][2]*tv[2][3]-tv[3][1]*tv[2][2]*tv[1][3]-tv[2][1]*tv[1][2]*tv[3][3];

    itv[1][1] = (tv[2][2]*tv[3][3] - tv[2][3]*tv[3][2])/detA;
    itv[1][2] = (tv[1][3]*tv[3][2] - tv[1][2]*tv[3][3])/detA;
    itv[1][3] = (tv[1][2]*tv[2][3] - tv[1][3]*tv[2][2])/detA;

    itv[2][1] = (tv[2][3]*tv[3][1] - tv[2][1]*tv[3][3])/detA;
    itv[2][2] = (tv[1][1]*tv[3][3] - tv[1][3]*tv[3][1])/detA;
    itv[2][3] = (tv[1][3]*tv[2][1] - tv[1][1]*tv[2][3])/detA;

    itv[3][1] = (tv[2][1]*tv[3][2] - tv[2][2]*tv[3][1])/detA;
    itv[3][2] = (tv[1][2]*tv[3][1] - tv[1][1]*tv[3][2])/detA;
    itv[3][3] = (tv[1][1]*tv[2][2] - tv[1][2]*tv[2][1])/detA;

    printf("ABC stress\n");
    for (i=1; i<=3; i++){
      for (j=1; j<=3; j++){

	sum = 0.0;
	for (k=1; k<=3; k++){
	  sum += itv[k][i]*st[3*(k-1)+(j-1)];
	}

	printf("%15.12f ",sum);
      }
      printf("\n");
    }
  }
  */


  return EdftD;
}
/* Ellner */


void Energy_Decomposition(const OpenMXBandEnergies *energies, double ECE[], const OpenMXElectronicViews *electronic)
{
  double ******DM=electronic->DM;
  double *****H=electronic->H;
  double *****H0=electronic->H0;
  double *****OLP=electronic->OLP;
  double ******iDM=electronic->iDM;
  double *****iHNL=electronic->iHNL;

  int i,spin,spinmax,XC_P_switch;
  int numS,numR,My_GNum,BN_AB,max_ene;
  int n,n1,n2,n3,Ng1,Ng2,Ng3,j,k;
  int GN,GNs,BN,DN,LN,N2D,n2D,N3[4];
  double intVH1[2],intVxc[2];
  double My_intVH1[2],My_intVxc[2];
  double c[2],Etot;
  int numprocs,myid,tag=999,ID,IDS,IDR;
  double Cxyz[4],gradxyz[4];
  double Stime_atom,Etime_atom;
  double time0,time1;
  double sden[2],tden,aden,pden[2];
  int tnoA,tnoB,wanA,wanB,Mc_AN,Gc_AN,h_AN;
  double sum,tsum,Total_Mul_up,Total_Mul_dn;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /* calculation of total energy  */

  max_ene = 14;

  Etot = 0.0; 
  for (i=0; i<=max_ene; i++){
    Etot += ECE[i];
  }
  Etot = Etot - ECE[0] - ECE[1] - ECE[13]; 

  Total_Mul_up = 0.0;
  Total_Mul_dn = 0.0;

  for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
    Total_Mul_up += InitN_USpin[Gc_AN];
    Total_Mul_dn += InitN_DSpin[Gc_AN];
  }
  
  if      (SpinP_switch==0){
    c[0] = 0.5*(Etot - energies->band)/(Total_Mul_up+1.0e-15);
  }
  else if (SpinP_switch==1) {
    c[0] = (Etot - energies->band)/(Total_Mul_up+Total_Mul_dn+1.0e-15);
    c[1] = c[0];

  }
  else if (SpinP_switch==3) {
    c[0] = (Etot - energies->band)/(Total_Mul_up+Total_Mul_dn+1.0e-15);
    c[1] = c[0];
  }

  /****************************************************
       calculations of DecEkin, DecEv, and DecEcon
  ****************************************************/

  if (SpinP_switch==0 || SpinP_switch==1){

    for (spin=0; spin<=SpinP_switch; spin++){

      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

	Gc_AN = M2G[Mc_AN];
	wanA = WhatSpecies[Gc_AN];
	tnoA = Spe_Total_CNO[wanA];

        for (i=0; i<tnoA; i++){

          DecEkin[spin][Mc_AN][i] = 0.0;
          DecEv[spin][Mc_AN][i]   = 0.0;
          DecEcon[spin][Mc_AN][i] = 0.0;

	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	    wanB = WhatSpecies[natn[Gc_AN][h_AN]];
	    tnoB = Spe_Total_CNO[wanB];
	    for (j=0; j<tnoB; j++){

	      DecEkin[spin][Mc_AN][i] += DM[0][spin][Mc_AN][h_AN][i][j]*H0[0][Mc_AN][h_AN][i][j];

	      DecEv[spin][Mc_AN][i]   += DM[0][spin][Mc_AN][h_AN][i][j]*(H[spin][Mc_AN][h_AN][i][j]-H0[0][Mc_AN][h_AN][i][j]);
	      DecEcon[spin][Mc_AN][i] += DM[0][spin][Mc_AN][h_AN][i][j]*OLP[0][Mc_AN][h_AN][i][j]*c[spin];
	    }
	  }
	}
      }
    }
  }

  else if (SpinP_switch==3){

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

      Gc_AN = M2G[Mc_AN];
      wanA = WhatSpecies[Gc_AN];
      tnoA = Spe_Total_CNO[wanA];

      for (i=0; i<tnoA; i++){

	DecEkin[0][Mc_AN][i] = 0.0;
	DecEkin[1][Mc_AN][i] = 0.0;

	DecEv[0][Mc_AN][i]   = 0.0;
	DecEv[1][Mc_AN][i]   = 0.0;

	DecEcon[0][Mc_AN][i] = 0.0;
	DecEcon[1][Mc_AN][i] = 0.0;

	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	  wanB = WhatSpecies[natn[Gc_AN][h_AN]];
	  tnoB = Spe_Total_CNO[wanB];
	  for (j=0; j<tnoB; j++){

	    DecEkin[0][Mc_AN][i] += DM[0][0][Mc_AN][h_AN][i][j]*H0[0][Mc_AN][h_AN][i][j];
	    DecEkin[1][Mc_AN][i] += DM[0][1][Mc_AN][h_AN][i][j]*H0[0][Mc_AN][h_AN][i][j];

	    DecEcon[0][Mc_AN][i] += DM[0][0][Mc_AN][h_AN][i][j]*OLP[0][Mc_AN][h_AN][i][j]*c[0];
	    DecEcon[1][Mc_AN][i] += DM[0][1][Mc_AN][h_AN][i][j]*OLP[0][Mc_AN][h_AN][i][j]*c[1];

	    DecEv[0][Mc_AN][i] += 
                    DM[0][0][Mc_AN][h_AN][i][j]*(H[0][Mc_AN][h_AN][i][j]-H0[0][Mc_AN][h_AN][i][j])
	         - iDM[0][0][Mc_AN][h_AN][i][j]*iHNL[0][Mc_AN][h_AN][i][j]
                 +  DM[0][1][Mc_AN][h_AN][i][j]*(H[1][Mc_AN][h_AN][i][j]-H0[0][Mc_AN][h_AN][i][j])
	         - iDM[0][1][Mc_AN][h_AN][i][j]*iHNL[1][Mc_AN][h_AN][i][j]
	      + 2.0*DM[0][2][Mc_AN][h_AN][i][j]*H[2][Mc_AN][h_AN][i][j]
              - 2.0*DM[0][3][Mc_AN][h_AN][i][j]*(H[3][Mc_AN][h_AN][i][j]+iHNL[2][Mc_AN][h_AN][i][j]);

	    DecEv[1][Mc_AN][i] = 0.0; 

	  }
	}
      }
    }

  }

}



void Energy_Decomposition_CWF(OpenMXSolverInvocation *scratch, const OpenMXCWFObservables *cwf_values, double ECE[])
{
  int i,i0,spin,spinmax,spinmax1,XC_P_switch;
  int numS,numR,My_GNum,BN_AB,Gc_AN;
  int n,n1,n2,n3,Ng1,Ng2,Ng3,j,k,Ncwf;
  int GNc,GRc,MNc,h_AN,Gh_AN,Rn,Mc_AN,Cwan,Hwan;
  int GN,GNs,BN,DN,LN,N2D,n2D,N3[4];
  double My_dcEXC[2],My_dcEH1[2];
  double dcEXC[2],dcEH1[2],Edc[2];
  double Ne[2],c[2],LNe;
  OpenMXSolverRealTensor CWF_Energy2,CWF_Energy3;
  double sden[2],aden,w,sum,sum2,r,Zc,Zh;
  int numprocs,myid;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  if      (SpinP_switch==0) {spinmax = 0; spinmax1 = 0;}
  else if (SpinP_switch==1) {spinmax = 1; spinmax1 = 1;}
  else if (SpinP_switch==3) {spinmax = 1; spinmax1 = 0;}

  /****************************************************
                allocation of arrays
  ****************************************************/

  OpenMX_Energy_CWFScratch(scratch,TNum_CWFs,&CWF_Energy2,&CWF_Energy3);
  for (spin=0; spin<2; spin++){
    for (i=0; i<TNum_CWFs; i++) openmx_solver_real_matrix_row(&CWF_Energy2,spin)[i] = 0.0;
  }
  for (spin=0; spin<2; spin++){
    for (i=0; i<TNum_CWFs; i++) openmx_solver_real_matrix_row(&CWF_Energy3,spin)[i] = 0.0;
  }

  /****************************************************
   set Vxc_Grid
  ****************************************************/

  XC_P_switch = 1;

  Set_XC_Grid(2, XC_P_switch,XC_switch,
	      Density_Grid_D[0],Density_Grid_D[1],
	      Density_Grid_D[2],Density_Grid_D[3],
	      Vxc_Grid_D[0], Vxc_Grid_D[1],
	      Vxc_Grid_D[2], Vxc_Grid_D[3],
	      NULL,NULL);

  /* copy Vxc_Grid_D to Vxc_Grid_B */

  Ng1 = Max_Grid_Index_D[1] - Min_Grid_Index_D[1] + 1;
  Ng2 = Max_Grid_Index_D[2] - Min_Grid_Index_D[2] + 1;
  Ng3 = Max_Grid_Index_D[3] - Min_Grid_Index_D[3] + 1;

  for (n=0; n<Num_Rcv_Grid_B2D[myid]; n++){
    DN = Index_Rcv_Grid_B2D[myid][n];
    BN = Index_Snd_Grid_B2D[myid][n];

    i = DN/(Ng2*Ng3);
    j = (DN-i*Ng2*Ng3)/Ng3;
    k = DN - i*Ng2*Ng3 - j*Ng3; 

    if ( !(i<=1 || (Ng1-2)<=i || j<=1 || (Ng2-2)<=j || k<=1 || (Ng3-2)<=k)){
      for (spin=0; spin<=SpinP_switch; spin++){
	Vxc_Grid_B[spin][BN] = Vxc_Grid_D[spin][DN];
      }
    }
  }

  /*******************************************************************
  Calculations of dcEH1 and dcEXC

  dcEH1 = \sum_{\sigma} -1/2\int dr {n_{sigma}+0.5*n_a} \delta V_H
  dcEXC = \sum_{\sigma} -\int dr n_{\sigma}v_{xc}^{\sigma}
  ********************************************************************/

  My_dcEH1[0] = 0.0;
  My_dcEH1[1] = 0.0;
  My_dcEXC[0] = 0.0;
  My_dcEXC[1] = 0.0;

  int cwf_device = scf_eigen_lib_flag == GPUSOLVER;
#ifdef OPENMX_HOST_SIMD_H
  if (cwf_device)
    cwf_device = !TotalEnergy_StageOnHost("Total_Energy CWF double counting",TotalEnergy_GridBytes(2));
#endif
  if (cwf_device){
#if OPENMX_GPU_HIP
    TotalEnergy_CWF_Dc_Grid_OpenMP(spinmax, My_dcEH1, My_dcEXC);
#else
    TotalEnergy_CWF_Dc_Grid_OpenACC(spinmax, My_dcEH1, My_dcEXC);
#endif
  }
  else{
    for (BN=0; BN<My_NumGridB_AB; BN++){

      sden[0] = Density_Grid_B[0][BN];
      sden[1] = Density_Grid_B[1][BN];
      aden = ADensity_Grid_B[BN];

      for (spin=0; spin<=spinmax; spin++){
        My_dcEH1[spin] += (sden[spin] + aden)*dVHart_Grid_B[BN];
        My_dcEXC[spin] += sden[spin]*Vxc_Grid_B[spin][BN];
      }

    } /* BN */
  }

  /****************************************************
       multiplying GridVol and MPI communication
  ****************************************************/

  MPI_Barrier(mpi_comm_level1);

  My_dcEH1[0] *= -(0.5*GridVol);
  My_dcEH1[1] *= -(0.5*GridVol);

  My_dcEXC[0] *= -GridVol;
  My_dcEXC[1] *= -GridVol;

  MPI_Barrier(mpi_comm_level1);
  for (spin=0; spin<=spinmax; spin++){
    MPI_Allreduce(&My_dcEH1[spin], &dcEH1[spin], 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
    MPI_Allreduce(&My_dcEXC[spin], &dcEXC[spin], 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
  }

  if (SpinP_switch==0){
    dcEH1[1] = dcEH1[0];
    dcEXC[1] = dcEXC[0];
  }

  /****************************************************
                  calculate c_sigma       
  ****************************************************/

  Ne[0] = 0.0;
  Ne[1] = 0.0;

  for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
    Ne[0] += InitN_USpin[Gc_AN];
    Ne[1] += InitN_DSpin[Gc_AN];
  }

  if (SpinP_switch==3) Ne[0] = Ne[0] + Ne[1];   

  if (SpinP_switch==3){

    Edc[0] = dcEH1[0] + dcEH1[1] + dcEXC[0] + dcEXC[1] + ECE[6] + ECE[7];
    c[0] = Edc[0]/(Ne[0]+1.0e-15);
  }  
  else{
    for (spin=0; spin<=spinmax1; spin++){
      Edc[spin] = dcEH1[spin] + dcEXC[spin] + ECE[6+spin];
      c[spin] = Edc[spin]/(Ne[spin]+1.0e-15);
    }
  }

  /****************************************************
                calculate CWF_Energy2
  ****************************************************/

  for (spin=0; spin<=spinmax1; spin++){
    for (i=0; i<TNum_CWFs; i++){
      openmx_solver_real_matrix_row(&CWF_Energy2,spin)[i] = c[spin]*openmx_cwf_charge_row(cwf_values,spin)[i];
    }
  }

  /****************************************************
                calculate CWF_Energy3
  ****************************************************/

  i0 = 0;

  for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){

    Cwan = WhatSpecies[Gc_AN];

    if (SpinP_switch==3) Ncwf = 2*CWF_Num_predefined[Cwan];
    else                 Ncwf = CWF_Num_predefined[Cwan];

    if (G2ID[Gc_AN]==myid){

      Mc_AN = F_G2M[Gc_AN];
      sum = 2.0*DecEscc[0][Mc_AN][0];  

      LNe = 0.0;
      for (spin=0; spin<=spinmax1; spin++){
        for (i=0; i<Ncwf; i++){
          LNe += openmx_cwf_charge_row(cwf_values,spin)[i0+i];
        }
      }

      if (SpinP_switch==0) LNe *= 2.0; 

      for (spin=0; spin<=spinmax1; spin++){
        for (i=0; i<Ncwf; i++){
          w = openmx_cwf_charge_row(cwf_values,spin)[i0+i]/(LNe+1.0e-15);
          openmx_solver_real_matrix_row(&CWF_Energy3,spin)[i0+i] = sum*w;
        } 
      }

    } /* if (G2ID[Gc_AN]==myid) */

    i0 += Ncwf;    

  } /* Gc_AN */

  for (spin=0; spin<=spinmax1; spin++){
    MPI_Allreduce( MPI_IN_PLACE, &openmx_solver_real_matrix_row(&CWF_Energy3,spin)[0], TNum_CWFs, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
  }

  /****************************************************
                 save the information
  ****************************************************/

  if ( myid==Host_ID ){
        
    int i0,wan1;
    char file_CWF[YOUSO10];
    FILE *fp_CWF;
    char buf[fp_bsize];          /* setvbuf */
    double sumE[2],sumE2[2];
    double TsumE[2],TsumE2[2];



    sprintf(file_CWF,"%s%s.CWF_Energy",filepath,filename);

    if ((fp_CWF = fopen(file_CWF,"w")) != NULL){

      setvbuf(fp_CWF,buf,_IOFBF,fp_bsize);  /* setvbuf */

      fprintf(fp_CWF,"\n");
      fprintf(fp_CWF,"***********************************************************\n");
      fprintf(fp_CWF,"***********************************************************\n");
      fprintf(fp_CWF,"                Energy decomposition by CWF                \n");
      fprintf(fp_CWF,"                The unit is Hartree.                       \n");
      fprintf(fp_CWF,"***********************************************************\n");
      fprintf(fp_CWF,"***********************************************************\n\n");

      if (SpinP_switch==0 || SpinP_switch==1){

	fprintf(fp_CWF,"             Eband(up)  Eband(dn)  Eband(sum) Eband(dif)  Edc(up)     Edc(dn)     Edc(sum)   Edc(dif)    Etot(sum)  Etot(dif)\n");

	i0 = 0;
        TsumE[0] = 0.0; TsumE[1] = 0.0;
        TsumE2[0] = 0.0; TsumE2[1] = 0.0;
  
	for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){

	  wan1 = WhatSpecies[Gc_AN];
	  sumE[0]  = 0.0; sumE[1]  = 0.0;
	  sumE2[0] = 0.0; sumE2[1] = 0.0;

	  for (spin=0; spin<=SpinP_switch; spin++){
	    for (i=0; i<CWF_Num_predefined[wan1]; i++){
	      sumE[spin]  += openmx_cwf_energy_row(cwf_values,spin)[i0+i];
	      sumE2[spin] += openmx_solver_real_matrix_row(&CWF_Energy2,spin)[i0+i] + openmx_solver_real_matrix_row(&CWF_Energy3,spin)[i0+i];
	    }
	  }
       
	  if (SpinP_switch==0){
	    sumE[1] = sumE[0];
	    sumE2[1] = sumE2[0];
	  }             

	  fprintf(fp_CWF," %4d %4s  %10.6f %10.6f %10.6f %10.6f %10.6f %10.6f %10.6f %10.6f %10.6f %10.6f\n",
		  Gc_AN,SpeName[wan1],
		  sumE[0],sumE[1],sumE[0]+sumE[1],sumE[0]-sumE[1],
		  sumE2[0],sumE2[1],sumE2[0]+sumE2[1],sumE2[0]-sumE2[1],
		  sumE[0]+sumE[1]+sumE2[0]+sumE2[1],
		  sumE[0]+sumE2[0]-sumE[1]-sumE2[1] );

	  i0 += CWF_Num_predefined[wan1];

	  for (spin=0; spin<2; spin++){
            TsumE[spin] += sumE[spin]; 
            TsumE2[spin] += sumE2[spin]; 
	  }

	} // Gc_AN

	fprintf(fp_CWF,"\n");
	fprintf(fp_CWF," Sum of decomposed energies evaluated by CWF\n");
	fprintf(fp_CWF,"        up  = %12.8f  down = %12.8f\n",
		TsumE[0]+TsumE2[0],TsumE[1]+TsumE2[1]);
	fprintf(fp_CWF,"  Total Sum = %12.8f  Dif  = %12.8f\n",
		TsumE[0]+TsumE2[0]+TsumE[1]+TsumE2[1],
		TsumE[0]+TsumE2[0]-TsumE[1]-TsumE2[1]);

	/* orbitally decomposed energies */

	fprintf(fp_CWF,"\n\n  Orbitally decomposed energies (Hartree) evaluated by CWF\n");

	i0 = 0;
	for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){

	  wan1 = WhatSpecies[Gc_AN];

	  fprintf(fp_CWF,"\n %4d%4s  Eband(up)  Eband(dn)  Eband(sum) Eband(dif) Edc(up)    Edc(dn)    Edc(sum)   Edc(dif)   Etot(sum)  Etot(dif)\n",Gc_AN,SpeName[wan1]);
		
	  fprintf(fp_CWF,"  orbital index\n");

	  wan1 = WhatSpecies[Gc_AN];
	  sumE[0] = 0.0; sumE[1] = 0.0;

	  if (SpinP_switch==0){
	    for (i=0; i<CWF_Num_predefined[wan1]; i++){
	      fprintf(fp_CWF,"   %2d    %10.6f %10.6f %10.6f %10.6f %10.6f %10.6f %10.6f %10.6f %10.6f %10.6f\n",
		      i,
		      openmx_cwf_energy_row(cwf_values,0)[i0+i],openmx_cwf_energy_row(cwf_values,0)[i0+i],
		      openmx_cwf_energy_row(cwf_values,0)[i0+i]+openmx_cwf_energy_row(cwf_values,0)[i0+i],
		      openmx_cwf_energy_row(cwf_values,0)[i0+i]-openmx_cwf_energy_row(cwf_values,0)[i0+i],
		      openmx_solver_real_matrix_row(&CWF_Energy2,0)[i0+i]+openmx_solver_real_matrix_row(&CWF_Energy3,0)[i0+i],openmx_solver_real_matrix_row(&CWF_Energy2,0)[i0+i]+openmx_solver_real_matrix_row(&CWF_Energy3,0)[i0+i],
		      openmx_solver_real_matrix_row(&CWF_Energy2,0)[i0+i]+openmx_solver_real_matrix_row(&CWF_Energy3,0)[i0+i]+openmx_solver_real_matrix_row(&CWF_Energy2,0)[i0+i]+openmx_solver_real_matrix_row(&CWF_Energy3,0)[i0+i],
		      openmx_solver_real_matrix_row(&CWF_Energy2,0)[i0+i]+openmx_solver_real_matrix_row(&CWF_Energy3,0)[i0+i]-openmx_solver_real_matrix_row(&CWF_Energy2,0)[i0+i]-openmx_solver_real_matrix_row(&CWF_Energy3,0)[i0+i],
		      openmx_cwf_energy_row(cwf_values,0)[i0+i]+openmx_solver_real_matrix_row(&CWF_Energy2,0)[i0+i]+openmx_solver_real_matrix_row(&CWF_Energy3,0)[i0+i]+openmx_cwf_energy_row(cwf_values,0)[i0+i]+openmx_solver_real_matrix_row(&CWF_Energy2,0)[i0+i]+openmx_solver_real_matrix_row(&CWF_Energy3,0)[i0+i],
		      openmx_cwf_energy_row(cwf_values,0)[i0+i]+openmx_solver_real_matrix_row(&CWF_Energy2,0)[i0+i]+openmx_solver_real_matrix_row(&CWF_Energy3,0)[i0+i]-openmx_cwf_energy_row(cwf_values,0)[i0+i]-openmx_solver_real_matrix_row(&CWF_Energy2,0)[i0+i]-openmx_solver_real_matrix_row(&CWF_Energy3,0)[i0+i]);

	    }
	  }
	  else if (SpinP_switch==1){
	    for (i=0; i<CWF_Num_predefined[wan1]; i++){
	      fprintf(fp_CWF,"   %2d    %10.6f %10.6f %10.6f %10.6f %10.6f %10.6f %10.6f %10.6f %10.6f %10.6f\n",
		      i,
		      openmx_cwf_energy_row(cwf_values,0)[i0+i],openmx_cwf_energy_row(cwf_values,1)[i0+i],
		      openmx_cwf_energy_row(cwf_values,0)[i0+i]+openmx_cwf_energy_row(cwf_values,1)[i0+i],
		      openmx_cwf_energy_row(cwf_values,0)[i0+i]-openmx_cwf_energy_row(cwf_values,1)[i0+i],
		      openmx_solver_real_matrix_row(&CWF_Energy2,0)[i0+i]+openmx_solver_real_matrix_row(&CWF_Energy3,0)[i0+i],openmx_solver_real_matrix_row(&CWF_Energy2,1)[i0+i]+openmx_solver_real_matrix_row(&CWF_Energy3,1)[i0+i],
		      openmx_solver_real_matrix_row(&CWF_Energy2,0)[i0+i]+openmx_solver_real_matrix_row(&CWF_Energy3,0)[i0+i]+openmx_solver_real_matrix_row(&CWF_Energy2,1)[i0+i]+openmx_solver_real_matrix_row(&CWF_Energy3,1)[i0+i],
		      openmx_solver_real_matrix_row(&CWF_Energy2,0)[i0+i]+openmx_solver_real_matrix_row(&CWF_Energy3,0)[i0+i]-openmx_solver_real_matrix_row(&CWF_Energy2,1)[i0+i]-openmx_solver_real_matrix_row(&CWF_Energy3,1)[i0+i],
		      openmx_cwf_energy_row(cwf_values,0)[i0+i]+openmx_solver_real_matrix_row(&CWF_Energy2,0)[i0+i]+openmx_solver_real_matrix_row(&CWF_Energy3,0)[i0+i]+openmx_cwf_energy_row(cwf_values,1)[i0+i]+openmx_solver_real_matrix_row(&CWF_Energy2,1)[i0+i]+openmx_solver_real_matrix_row(&CWF_Energy3,1)[i0+i],
		      openmx_cwf_energy_row(cwf_values,0)[i0+i]+openmx_solver_real_matrix_row(&CWF_Energy2,0)[i0+i]+openmx_solver_real_matrix_row(&CWF_Energy3,0)[i0+i]-openmx_cwf_energy_row(cwf_values,1)[i0+i]-openmx_solver_real_matrix_row(&CWF_Energy2,1)[i0+i]-openmx_solver_real_matrix_row(&CWF_Energy3,1)[i0+i]);
	    }
	  } 

	  i0 += CWF_Num_predefined[wan1];

	} // Gc_AN

      } // if (SpinP_switch==0 || SpinP_switch==1)

      else if (SpinP_switch==3){

	fprintf(fp_CWF,"                Eband        Edc           Etot\n");

	i0 = 0;
        TsumE[0] = 0.0;
        TsumE2[0] = 0.0;

	for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){

	  wan1 = WhatSpecies[Gc_AN];
	  sumE[0]  = 0.0; 
	  sumE2[0] = 0.0; 

	  for (i=0; i<2*CWF_Num_predefined[wan1]; i++){
	    sumE[0]  += openmx_cwf_energy_row(cwf_values,0)[i0+i];
	    sumE2[0] += openmx_solver_real_matrix_row(&CWF_Energy2,0)[i0+i] + openmx_solver_real_matrix_row(&CWF_Energy3,0)[i0+i];
	  }
       
	  fprintf(fp_CWF," %4d %4s     %10.6f %10.6f %10.6f\n",
		  Gc_AN,SpeName[wan1],sumE[0],sumE2[0],sumE[0]+sumE2[0]);

	  i0 += 2*CWF_Num_predefined[wan1];

          TsumE[0] += sumE[0];
          TsumE2[0] += sumE2[0];

	} // Gc_AN

	fprintf(fp_CWF,"\n");
	fprintf(fp_CWF,"  Total Sum = %12.8f\n",TsumE[0]+TsumE2[0]);

	/* orbitally decomposed energies */

	fprintf(fp_CWF,"\n\n  Orbitally decomposed energies (Hartree) evaluated by CWF\n");

	i0 = 0;
	for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){

	  wan1 = WhatSpecies[Gc_AN];

	  fprintf(fp_CWF,"\n %4d%4s       Eband      Edc        Etot\n",Gc_AN,SpeName[wan1]);
		
	  fprintf(fp_CWF,"  orbital index\n");

	  wan1 = WhatSpecies[Gc_AN];
	  sumE[0] = 0.0; sumE[1] = 0.0;

	  for (i=0; i<2*CWF_Num_predefined[wan1]; i++){
	    fprintf(fp_CWF,"   %2d         %10.6f %10.6f %10.6f\n",
		    i,
		    openmx_cwf_energy_row(cwf_values,0)[i0+i],
		    openmx_solver_real_matrix_row(&CWF_Energy2,0)[i0+i]+openmx_solver_real_matrix_row(&CWF_Energy3,0)[i0+i],
		    openmx_cwf_energy_row(cwf_values,0)[i0+i]+openmx_solver_real_matrix_row(&CWF_Energy2,0)[i0+i]+openmx_solver_real_matrix_row(&CWF_Energy3,0)[i0+i]);
	  }

	  i0 += 2*CWF_Num_predefined[wan1];

	} // Gc_AN

      } // else if (SpinP_switch==3)

      /* fclose fp_CWF */

      fclose(fp_CWF);

    } // if ((fp_CWF = fopen(file_CWF,"w")) != NULL)

    else{
      printf("Failure of saving the CWF_Energy file.\n");
    }

  } // if ( myid==Host_ID )

  /* Scratch matrices are released with the enclosing invocation. */
}

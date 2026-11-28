/**********************************************************************
  MD_pac.c:

     MD_pac.c is a subroutine to perform molecular dynamics
     simulations and geometry optimization.

  Log of MD_pac.c:

     22/Nov/2001  Released by T. Ozaki
     15/Dec/2003  DIISs are added by H. Kino
     14/May/2004  NVT_VS is added by M. Ohfuti
     25/May/2004  Modified by T. Ozaki
     14/Jul/2007  RF added by H.M. Weng
     08/Jan/2010  NVT_VS2 added by T. Ohwaki 
     23/Dec/2012  RestartFiles4GeoOpt added by T. Ozaki
     08/Oct/2019  NPT-MD added by MIZUHO, T. Iitaka, and T. Ozaki

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "mpi.h"
#include "openmx_common.h"
#include "lapack_prototypes.h"
#include "openmx_geometry_opt.h"
#include "openmx_md_step.h"


#define Criterion_Max_Step            0.50

 
static void NoMD(const OpenMXChemicalPotentials *chemical, OpenMXDynamicsState *dynamics, OpenMXThermodynamicEnergies *thermo, const OpenMXBandEnergies *energies, const OpenMXEnergyComponents *components, int iter);
static void VerletXYZ(OpenMXMD *md, const OpenMXChemicalPotentials *chemical, OpenMXDynamicsState *dynamics, OpenMXThermodynamicEnergies *thermo, const OpenMXBandEnergies *energies, const OpenMXEnergyComponents *components, int iter);
static void NVT_VS(OpenMXMD *md, const OpenMXChemicalPotentials *chemical, OpenMXDynamicsState *dynamics, OpenMXThermodynamicEnergies *thermo, const OpenMXBandEnergies *energies, const OpenMXEnergyComponents *components, int iter);  /* added by mari */
static void NVT_VS2(OpenMXMD *md, const OpenMXChemicalPotentials *chemical, OpenMXDynamicsState *dynamics, OpenMXThermodynamicEnergies *thermo, const OpenMXBandEnergies *energies, const OpenMXEnergyComponents *components, int iter); /* added by Ohwaki */
static void NVT_VS4(OpenMXMD *md, const OpenMXChemicalPotentials *chemical, OpenMXDynamicsState *dynamics, OpenMXThermodynamicEnergies *thermo, const OpenMXBandEnergies *energies, const OpenMXEnergyComponents *components, int iter); /* added by Ohwaki */
static void NVT_NH(OpenMXMD *md, const OpenMXChemicalPotentials *chemical, OpenMXDynamicsState *dynamics, OpenMXThermodynamicEnergies *thermo, const OpenMXBandEnergies *energies, const OpenMXEnergyComponents *components, int iter);
static void NVT_Langevin(OpenMXMD *md, OpenMXDynamicsState *dynamics, OpenMXThermodynamicEnergies *thermo, int iter); /* added by Ohwaki */
static void Steepest_Descent(OpenMXGeometryOpt *geometry, const OpenMXChemicalPotentials *chemical, OpenMXDynamicsState *dynamics, OpenMXThermodynamicEnergies *thermo, const OpenMXBandEnergies *energies, const OpenMXEnergyComponents *components, int iter, int SD_scaling_flag);
static void GDIIS(OpenMXGeometryOpt *geometry, const OpenMXChemicalPotentials *chemical, OpenMXDynamicsState *dynamics, OpenMXThermodynamicEnergies *thermo, const OpenMXBandEnergies *energies, const OpenMXEnergyComponents *components, int iter, int iter0);
static void GDIIS_BFGS(OpenMXGeometryOpt *geometry, const OpenMXChemicalPotentials *chemical, OpenMXDynamicsState *dynamics, OpenMXThermodynamicEnergies *thermo, const OpenMXBandEnergies *energies, const OpenMXEnergyComponents *components, int iter, int iter0);
static void GDIIS_EF(OpenMXGeometryOpt *geometry, const OpenMXChemicalPotentials *chemical, OpenMXDynamicsState *dynamics, OpenMXThermodynamicEnergies *thermo, const OpenMXBandEnergies *energies, const OpenMXEnergyComponents *components, int iter, int iter0);
static void Geometry_Opt_DIIS(OpenMXGeometryOpt *geometry, const OpenMXChemicalPotentials *chemical, OpenMXDynamicsState *dynamics, OpenMXThermodynamicEnergies *thermo, const OpenMXBandEnergies *energies, const OpenMXEnergyComponents *components, int iter);
static void Geometry_Opt_DIIS_BFGS(OpenMXGeometryOpt *geometry, const OpenMXChemicalPotentials *chemical, OpenMXDynamicsState *dynamics, OpenMXThermodynamicEnergies *thermo, const OpenMXBandEnergies *energies, const OpenMXEnergyComponents *components, int iter);
static void Geometry_Opt_DIIS_EF(OpenMXGeometryOpt *geometry, const OpenMXChemicalPotentials *chemical, OpenMXDynamicsState *dynamics, OpenMXThermodynamicEnergies *thermo, const OpenMXBandEnergies *energies, const OpenMXEnergyComponents *components, int iter);
static void Geometry_Opt_RF(OpenMXGeometryOpt *geometry, const OpenMXChemicalPotentials *chemical, OpenMXDynamicsState *dynamics, OpenMXThermodynamicEnergies *thermo, const OpenMXBandEnergies *energies, const OpenMXEnergyComponents *components, int iter);
static void Estimate_Initial_Hessian(OpenMXGeometryOpt *geometry, int diis_iter, int CellOpt_flag, double itv[4][4]);
static void RF(OpenMXGeometryOpt *geometry, const OpenMXChemicalPotentials *chemical, OpenMXDynamicsState *dynamics, OpenMXThermodynamicEnergies *thermo, const OpenMXBandEnergies *energies, const OpenMXEnergyComponents *components, int iter, int iter0);
static void RFC5(OpenMXGeometryOpt *geometry, const OpenMXChemicalPotentials *chemical, OpenMXDynamicsState *dynamics, OpenMXThermodynamicEnergies *thermo, const OpenMXBandEnergies *energies, const OpenMXEnergyComponents *components, int iter, int iter0, double dE_da[4][4], double itv[4][4]);
static void EvsLC(const OpenMXChemicalPotentials *chemical, OpenMXDynamicsState *dynamics, OpenMXThermodynamicEnergies *thermo, const OpenMXBandEnergies *energies, const OpenMXEnergyComponents *components, int iter);
static void Delta_Factor(const OpenMXChemicalPotentials *chemical, OpenMXDynamicsState *dynamics, OpenMXThermodynamicEnergies *thermo, const OpenMXBandEnergies *energies, const OpenMXEnergyComponents *components, int iter);
static int RestartFiles4GeoOpt(char *mode);
static void Output_abc_file(OpenMXThermodynamicEnergies *thermo, int iter);
static void Cell_Opt_SD(const OpenMXChemicalPotentials *chemical, OpenMXDynamicsState *dynamics, OpenMXThermodynamicEnergies *thermo, const OpenMXBandEnergies *energies, const OpenMXEnergyComponents *components, int iter, int SD_scaling_flag);
static void Cell_Opt_RF(OpenMXGeometryOpt *geometry, const OpenMXChemicalPotentials *chemical, OpenMXDynamicsState *dynamics, OpenMXThermodynamicEnergies *thermo, const OpenMXBandEnergies *energies, const OpenMXEnergyComponents *components, int iter);
static void NPT_VS_PR(OpenMXMD *md, OpenMXDynamicsState *dynamics, OpenMXThermodynamicEnergies *thermo, int iter); /* added by MIZUHO for NPT-MD */
static void NPT_VS_WV(OpenMXMD *md, OpenMXDynamicsState *dynamics, OpenMXThermodynamicEnergies *thermo, int iter); /* added by MIZUHO for NPT-MD */
static void NPT_NH_PR(OpenMXMD *md, OpenMXDynamicsState *dynamics, OpenMXThermodynamicEnergies *thermo, int iter); /* added by MIZUHO for NPT-MD */
static void NPT_NH_WV(OpenMXMD *md, OpenMXDynamicsState *dynamics, OpenMXThermodynamicEnergies *thermo, int iter); /* added by MIZUHO for NPT-MD */

int SuccessReadingfiles;

double MD_pac(OpenMXCoreContext *core, int iter, char *fname_input)
{
  OpenMXChemicalPotentials *chemical=NULL;
  char chemical_error[256];
  if(!openmx_core_chemical_potentials(core,&chemical,chemical_error,sizeof(chemical_error))) {
    fprintf(stderr,"OpenMX chemical potentials: %s\n",chemical_error);
    MPI_Abort(mpi_comm_level1,1);
    abort();
  }

  double time0;
  double TStime,TEtime;
  int numprocs,myid;
  int i,j,k;

  dtime(&TStime);
  OpenMXBandEnergies *energies=NULL;
  char band_energy_error[256];
  if(!openmx_core_band_energies(core,&energies,band_energy_error,sizeof(band_energy_error))) {
    fprintf(stderr,"OpenMX band energies: %s\n",band_energy_error);
    abort();
  }

  OpenMXEnergyComponents *components=NULL;
  char component_error[256];
  if(!openmx_core_energy_components(core,&components,component_error,sizeof(component_error))) {
    fprintf(stderr,"OpenMX energy components: %s\n",component_error);
    abort();
  }

  OpenMXThermodynamicEnergies *thermo=NULL;
  char thermodynamic_error[256];
  if(!openmx_core_thermodynamic_energies(core,&thermo,thermodynamic_error,sizeof(thermodynamic_error))) {
    fprintf(stderr,"OpenMX thermodynamic energies: %s\n",thermodynamic_error);
    abort();
  }

  OpenMXDynamicsState *dynamics=NULL;
  char dynamics_error[256];
  if(!openmx_core_dynamics(core,&dynamics,dynamics_error,sizeof(dynamics_error))) {
    fprintf(stderr,"OpenMX dynamics state: %s\n",dynamics_error);
    abort();
  }

 
  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  if (myid==Host_ID && 0<level_stdout){
    printf("\n*******************************************************\n"); 
    printf("             MD or geometry opt. at MD =%2d              \n",iter);
    printf("*******************************************************\n\n"); 
  }

  if (myid==Host_ID && MD_OutABC==1) Output_abc_file(thermo, iter);

  /*********************************************** 
    read restart files for geometry optimization
    MD_switch==3: Steepest_Descent
    MD_switch==4: Geometry_Opt_DIIS_EF
    MD_switch==5: Geometry_Opt_DIIS_BFGS
    MD_switch==6: Geometry_Opt_RF
    MD_switch==7: Geometry_Opt_DIIS
  ***********************************************/

  if (iter==1 && GeoOpt_RestartFromFile){
    if ( MD_switch==3 || MD_switch==4 || MD_switch==5 || MD_switch==6 || MD_switch==7 ){
      if (RestartFiles4GeoOpt("read")){
        SuccessReadingfiles = 1;
        if (myid==Host_ID){
          printf("Found restart files for geometry optimization\n");
        }
      }
    }
  }
  else {
    SuccessReadingfiles = 0;
  }

  OpenMXMD *md=NULL;
  OpenMXMDRecords records=OPENMX_MD_RECORD_NPT;
  int molecular_dynamics=1;
  switch (MD_switch) {
    case  1: records=OPENMX_MD_RECORD_VERLET; break;
    case  2: case 11: case 14: records=OPENMX_MD_RECORD_SCALING; break;
    case  9: records=OPENMX_MD_RECORD_NOSE; break;
    case 15: records=OPENMX_MD_RECORD_LANGEVIN; break;
    case 27: case 28: case 29: case 30: break;
    default: molecular_dynamics=0; break;
  }
  if (molecular_dynamics) {
    const OpenMXMDAtoms atoms={atomnum,Matomnum,G2ID,M2G,Gxyz,atom_Fixed_XYZ};
    md=OpenMX_MD_Begin(core,mpi_comm_level1,&atoms,scf_eigen_lib_flag==GPUSOLVER,level_stdout);
    if (records!=OPENMX_MD_RECORD_NPT) OpenMX_MD_Center(md,17);
  }

  OpenMXGeometryOpt *geometry=NULL;
  if ((MD_switch>=3 && MD_switch<=7) || MD_switch==18)
    geometry=OpenMX_Geometry_Begin(core,mpi_comm_level1,Host_ID,atomnum,G2ID);

  /* Call a subroutine based on MD_switch */

  switch (MD_switch) {
    case  0: NoMD(chemical, dynamics, thermo, energies, components, iter+MD_Current_Iter);                    break;
    case  1: VerletXYZ(md, chemical, dynamics, thermo, energies, components, iter+MD_Current_Iter);               break;
    case  2: NVT_VS(md, chemical, dynamics, thermo, energies, components, iter+MD_Current_Iter);                  break;  /* added by mari */
    case  3: Steepest_Descent(geometry, chemical, dynamics, thermo, energies, components, iter+MD_Current_Iter,1);      break;
    case  4: Geometry_Opt_DIIS_EF(geometry, chemical, dynamics, thermo, energies, components, iter+MD_Current_Iter);    break;
    case  5: Geometry_Opt_DIIS_BFGS(geometry, chemical, dynamics, thermo, energies, components, iter+MD_Current_Iter);  break;
    case  6: Geometry_Opt_RF(geometry, chemical, dynamics, thermo, energies, components, iter+MD_Current_Iter);         break;  /* added by hmweng */
    case  7: Geometry_Opt_DIIS(geometry, chemical, dynamics, thermo, energies, components, iter+MD_Current_Iter);       break;
    case  8:                                                break;  /* not used */
    case  9: NVT_NH(md, chemical, dynamics, thermo, energies, components, iter+MD_Current_Iter);                  break;
    case 10:                                                break;  /* not used */
    case 11: NVT_VS2(md, chemical, dynamics, thermo, energies, components, iter+MD_Current_Iter);                 break;  /* added by Ohwaki */
    case 12: EvsLC(chemical, dynamics, thermo, energies, components, iter);                                   break;
    case 14: NVT_VS4(md, chemical, dynamics, thermo, energies, components, iter+MD_Current_Iter);                 break;  /* added by Ohwaki */
    case 15: NVT_Langevin(md, dynamics, thermo, iter+MD_Current_Iter);            break;  /* added by Ohwaki */
    case 16: Delta_Factor(chemical, dynamics, thermo, energies, components, iter);                            break;  /* delta-factor */
    case 17: Cell_Opt_SD(chemical, dynamics, thermo, energies, components, iter+MD_Current_Iter,1);           break;
    case 18: Cell_Opt_RF(geometry, chemical, dynamics, thermo, energies, components, iter+MD_Current_Iter);             break;
    /* added by MIZUHO for NPT-MD */
    case 27: NPT_VS_PR(md, dynamics, thermo, iter+MD_Current_Iter);               break;
    case 28: NPT_VS_WV(md, dynamics, thermo, iter+MD_Current_Iter);               break;
    case 29: NPT_NH_PR(md, dynamics, thermo, iter+MD_Current_Iter);               break;
    case 30: NPT_NH_WV(md, dynamics, thermo, iter+MD_Current_Iter);               break;
  }

  if (md) {
    OpenMXMDCell cell;
    cell.pi=PI;
    for (int i=0; i<3; ++i) {
      memcpy(cell.real[i],tv[i+1]+1,3*sizeof(double));
      memcpy(cell.reciprocal[i],rtv[i+1]+1,3*sizeof(double));
      cell.origin[i]=Grid_Origin[i+1];
    }
    const OpenMXMDHistory history={Cell_Gxyz,His_Gxyz,GxyzHistoryIn,
                                   Extrapolated_Charge_History,M_GDIIS_HISTORY+1};
    Correct_Position_flag=OpenMX_MD_Finish(md,records,&cell,&history);
    /* Langevin reports the force on atom 1 after the owners exchange it. */
    if (records==OPENMX_MD_RECORD_LANGEVIN && myid==Host_ID) {
      char fileE[YOUSO10];
      const int step=iter+MD_Current_Iter;
      sprintf(fileE,"%s%s.ene",filepath,filename);
      iterout_md(chemical,dynamics,thermo,energies,components,step,MD_TimeStep*(step-1),fileE);
    }
  }

  /* making of an input file with the final structure */

  if (Runtest_flag==0){

    Make_InputFile_with_FinalCoord(dynamics, fname_input,iter+MD_Current_Iter);
  }

  /*********************************************** 
    save restart files for geometry optimization
    MD_switch==4: Geometry_Opt_DIIS_EF
    MD_switch==5: Geometry_Opt_DIIS_BFGS
    MD_switch==6: Geometry_Opt_RF
    MD_switch==7: Geometry_Opt_DIIS
  ***********************************************/

  if ( MD_switch==3 || MD_switch==4 || MD_switch==5 || MD_switch==6 || MD_switch==7 ){
    RestartFiles4GeoOpt("write");
  }

  MPI_Bcast(&MD_Opt_OK, 1, MPI_INT, Host_ID, mpi_comm_level1);

  dtime(&TEtime);
  time0 = TEtime - TStime;
  return time0;
}


void NoMD(const OpenMXChemicalPotentials *chemical, OpenMXDynamicsState *dynamics, OpenMXThermodynamicEnergies *thermo, const OpenMXBandEnergies *energies, const OpenMXEnergyComponents *components, int iter)
{
  char fileCoord[YOUSO10];
  FILE *fp_crd,*fp_SD;
  int i,j,k;
  char buf[fp_bsize];          /* setvbuf */
  int numprocs,myid,ID;
  char fileE[YOUSO10];

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  MD_Opt_OK = 1;

  if (myid==Host_ID){ 

    if (MD_Opt_OK==1 || iter==MD_IterNumber){

      strcpy(fileCoord,".crd");
      fnjoint(filepath,filename,fileCoord);
      if ((fp_crd = fopen(fileCoord,"w")) != NULL){

        setvbuf(fp_crd,buf,_IOFBF,fp_bsize);  /* setvbuf */

        fprintf(fp_crd,"\n");
        fprintf(fp_crd,"***********************************************************\n");
        fprintf(fp_crd,"***********************************************************\n");
        fprintf(fp_crd,"       xyz-coordinates (Ang) and forces (Hartree/Bohr)  \n");
        fprintf(fp_crd,"***********************************************************\n");
        fprintf(fp_crd,"***********************************************************\n\n");

        fprintf(fp_crd,"<coordinates.forces\n");
        fprintf(fp_crd,"  %i\n",atomnum);
        for (k=1; k<=atomnum; k++){
          i = WhatSpecies[k];
          j = Spe_WhatAtom[i];
          fprintf(fp_crd," %4d  %4s   %9.5f %9.5f %9.5f  %15.12f %15.12f %15.12f\n",
                  k,Atom_Symbol[j],
	          Gxyz[k][1]*BohrR,Gxyz[k][2]*BohrR,Gxyz[k][3]*BohrR,
	    	  -Gxyz[k][17],-Gxyz[k][18],-Gxyz[k][19]);
        }
        fprintf(fp_crd,"coordinates.forces>\n");
        fclose(fp_crd);
      }
      else
        printf("error(1) in MD_pac.c\n");
    }

  } /* if (myid==Host_ID) */

  /****************************************************
   write informatins to *.ene
  ****************************************************/

  if (myid==Host_ID){  
    sprintf(fileE,"%s%s.ene",filepath,filename);
    iterout_md(chemical, dynamics, thermo, energies, components, iter,MD_TimeStep*(iter-1),fileE);
  }

}

    


void VerletXYZ(OpenMXMD *md, const OpenMXChemicalPotentials *chemical, OpenMXDynamicsState *dynamics, OpenMXThermodynamicEnergies *thermo, const OpenMXBandEnergies *energies, const OpenMXEnergyComponents *components, int iter)
{
  int myid;
  MPI_Comm_rank(mpi_comm_level1,&myid);
  const double dt=41.3411*MD_TimeStep;
  const double mass_scale=unified_atomic_mass_unit/electron_mass;
  MD_Opt_OK=0;

  if (iter!=1) {
    const OpenMXMDStep kick={OPENMX_MD_VERLET_KICK,dt,0.0,0.0};
    OpenMX_MD_Move(md,kick,mass_scale,1.0,NULL,NULL);
  }
  thermo->ionic_kinetic=OpenMX_MD_Kinetic(md,mass_scale,1);
  dynamics->temperature=thermo->ionic_kinetic/(1.5*kB*(double)atomnum)*eV2Hartree;
  if (myid==Host_ID) {
    char fileE[YOUSO10];
    sprintf(fileE,"%s%s.ene",filepath,filename);
    iterout_md(chemical,dynamics,thermo,energies,components,iter,MD_TimeStep*(iter-1),fileE);
  }

  if (iter==1 || iter!=MD_IterNumber) {
    const OpenMXMDStep drift={OPENMX_MD_VERLET_DRIFT,dt,0.0,0.0};
    OpenMX_MD_Move(md,drift,mass_scale,1.0,NULL,NULL);
  }
}




void Steepest_Descent(OpenMXGeometryOpt *geometry, const OpenMXChemicalPotentials *chemical, OpenMXDynamicsState *dynamics, OpenMXThermodynamicEnergies *thermo, const OpenMXBandEnergies *energies, const OpenMXEnergyComponents *components, int iter, int SD_scaling_flag)
{
  /* 1au=2.4189*10^-2 fs, 1fs=41.341105 au */
  int i,j,k;
  int header_flag;
  double dt,SD_max,SD_min,SD_init,scale,MaxStep;
  double My_Max_Force,Wscale;
  char fileCoord[YOUSO10];
  char fileSD[YOUSO10];
  FILE *fp_crd,*fp_SD;
  int myid;
  char buf[fp_bsize];          /* setvbuf */
  char fileE[YOUSO10];
  char file_name[YOUSO10];
  FILE *fp;

  /* MPI */
  MPI_Comm_rank(mpi_comm_level1,&myid);

  MD_Opt_OK = 0;
  Wscale = unified_atomic_mass_unit/electron_mass;

  /******************************************
              read *.rst4gopt.SD1
  ******************************************/

  if (SuccessReadingfiles){

    sprintf(file_name,"%s%s_rst/%s.rst4gopt.SD1",filepath,filename,filename);

    if ((fp = fopen(file_name,"rb")) != NULL){

      fread(&Correct_Position_flag, sizeof(int), 1, fp);
      fread(&SD_scaling,      sizeof(double), 1, fp);
      fread(&SD_scaling_user, sizeof(double), 1, fp);
      fread(&Past_Utot,       sizeof(double), 10, fp);

      fclose(fp);
    }    
    else{
      printf("Failure of saving %s\n",file_name);
    }
  }

  /****************************************************
   find the maximum value of force 
  ****************************************************/

  My_Max_Force = 0.0;
#pragma omp parallel for schedule(static) reduction(max:My_Max_Force) if(Matomnum>=128)
  for (int Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    const int Gc_AN = M2G[Mc_AN];
    double tmp0 = 0.0;
    for (int j=1; j<=3; j++){
      if (atom_Fixed_XYZ[Gc_AN][j]==0){
        tmp0 += Gxyz[Gc_AN][16+j]*Gxyz[Gc_AN][16+j];
      }
    }
    tmp0 = sqrt(tmp0); 
    if (My_Max_Force<tmp0) My_Max_Force = tmp0;
  }

  /****************************************************
   MPI, Max_Force
  ****************************************************/

  MPI_Allreduce(&My_Max_Force, &Max_Force, 1, MPI_DOUBLE, MPI_MAX, mpi_comm_level1);

  if (Max_Force<MD_Opt_criterion){

    MD_Opt_OK = 1;

    /* show a message if the geometry optimization was achieved or not. */
    if (myid==Host_ID) printf("\nThe geometry optimization was achieved.\n");
  }

  /****************************************************
   set up SD_scaling
  ****************************************************/

  dt = 41.3411*2.0;
  SD_init = dt*dt/Wscale;
  SD_max = SD_init*15.0;   /* default 15   */
  SD_min = SD_init*0.04;   /* default 0.02 */

  //SD_min = SD_init*0.01;   /* default 0.02 */


  if (iter==1 || SD_scaling_flag==0){

    SD_scaling_user = Max_Force/BohrR/1.5;
    SD_scaling = SD_scaling_user/(Max_Force+1.0e-10);

    if (SD_max<SD_scaling) SD_scaling = SD_max;
    if (SD_scaling<SD_min) SD_scaling = SD_min;
  }

  else{

    if (Past_Utot[1]<thermo->total){
      SD_scaling = SD_scaling/2.0;
    }
    else if (Past_Utot[1]<Past_Utot[2] && thermo->total<Past_Utot[1] && iter%4==1){
      SD_scaling = SD_scaling*2.5;
    }

    if (SD_max<SD_scaling) SD_scaling = SD_max;
    if (SD_scaling<SD_min) SD_scaling = SD_min;

    Past_Utot[5] = Past_Utot[4];
    Past_Utot[4] = Past_Utot[3];
    Past_Utot[3] = Past_Utot[2];
    Past_Utot[2] = Past_Utot[1];
    Past_Utot[1] = thermo->total;
  }
  
  if (myid==Host_ID && 0<level_stdout) printf("<Steepest_Descent>  SD_scaling=%15.12f\n",SD_scaling);

  /****************************************************
   write informatins to *.ene
  ****************************************************/

  if (myid==Host_ID){  

    sprintf(fileE,"%s%s.ene",filepath,filename);
    iterout_md(chemical, dynamics, thermo, energies, components, iter,MD_TimeStep*(iter-1),fileE);
  }

  /****************************************************
    move atoms
  ****************************************************/

  if (MD_Opt_OK!=1 && iter!=MD_IterNumber){

    /* avoid too large movement */
    if ( Criterion_Max_Step<(Max_Force*SD_scaling) )
      scale = Criterion_Max_Step/(Max_Force*SD_scaling);
    else 
      scale = 1.0; 

    /* update coordinates */
#pragma omp parallel for schedule(static) if(Matomnum>=128)
    for (int Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      const int Gc_AN = M2G[Mc_AN];
      for (int j=1; j<=3; j++){
	if (atom_Fixed_XYZ[Gc_AN][j]==0){
	  Gxyz[Gc_AN][j] = Gxyz[Gc_AN][j] - scale*SD_scaling*Gxyz[Gc_AN][16+j];
	}
      }
    }
  }

  /****************************************************
   MPI, Gxyz
  ****************************************************/

  OpenMX_Geometry_Share_Atoms(geometry,Gxyz);

  if (myid==Host_ID){ 

    if (0<level_stdout){ 

      printf("<Steepest_Descent>  |Maximum force| (Hartree/Bohr) =%15.12f\n",
             Max_Force);
      printf("<Steepest_Descent>  Criterion       (Hartree/Bohr) =%15.12f\n",
	     MD_Opt_criterion);

      printf("\n");
      for (i=1; i<=atomnum; i++){
	printf("  atom=%3d, XYZ(ang) Fxyz(a.u.)=%9.4f %9.4f %9.4f  %9.4f %9.4f %9.4f\n",
	       i,BohrR*Gxyz[i][1],BohrR*Gxyz[i][2],BohrR*Gxyz[i][3],
	       Gxyz[i][17],Gxyz[i][18],Gxyz[i][19] ); 
      }   
    }

    strcpy(fileSD,".SD");
    fnjoint(filepath,filename,fileSD);

    fp_SD = fopen(fileSD, "r");
    if (fp_SD==NULL){
      header_flag = 1;
    }
    else{
      header_flag = 0;
      fclose(fp_SD);
    }  

    if ((fp_SD = fopen(fileSD,"a")) != NULL){

      setvbuf(fp_SD,buf,_IOFBF,fp_bsize);  /* setvbuf */

      if ( Criterion_Max_Step<(Max_Force*SD_scaling) )
	MaxStep = Criterion_Max_Step;
      else 
	MaxStep = SD_scaling*Max_Force;

      if (header_flag==1){

        fprintf(fp_SD,"\n");
        fprintf(fp_SD,"***********************************************************\n");
        fprintf(fp_SD,"***********************************************************\n");
        fprintf(fp_SD,"              History of geometry optimization             \n");
        fprintf(fp_SD,"***********************************************************\n");
        fprintf(fp_SD,"***********************************************************\n\n");

        fprintf(fp_SD,"  MD_iter   SD_scaling     |Maximum force|   Maximum step        Utot\n");
        fprintf(fp_SD,"                           (Hartree/Bohr)        (Ang)         (Hartree)\n\n");
      }

      fprintf(fp_SD,"  %3d  %15.8f  %15.8f  %15.8f  %15.8f\n",
              iter,SD_scaling,Max_Force,MaxStep*BohrR, thermo->total);

      fclose(fp_SD);
    }
    else
      printf("Error(7) in MD_pac.c\n");

    if (MD_Opt_OK==1 || iter==MD_IterNumber){

      strcpy(fileCoord,".crd");
      fnjoint(filepath,filename,fileCoord);
      if ((fp_crd = fopen(fileCoord,"w")) != NULL){

        setvbuf(fp_crd,buf,_IOFBF,fp_bsize);  /* setvbuf */

        fprintf(fp_crd,"\n");
        fprintf(fp_crd,"***********************************************************\n");
        fprintf(fp_crd,"***********************************************************\n");
        fprintf(fp_crd,"       xyz-coordinates (Ang) and forces (Hartree/Bohr)  \n");
        fprintf(fp_crd,"***********************************************************\n");
        fprintf(fp_crd,"***********************************************************\n\n");

        fprintf(fp_crd,"<coordinates.forces\n");
        fprintf(fp_crd,"  %i\n",atomnum);
        for (k=1; k<=atomnum; k++){
          i = WhatSpecies[k];
          j = Spe_WhatAtom[i];
          fprintf(fp_crd," %4d  %4s   %9.5f %9.5f %9.5f  %15.12f %15.12f %15.12f\n",
                  k,Atom_Symbol[j],
	          Gxyz[k][1]*BohrR,Gxyz[k][2]*BohrR,Gxyz[k][3]*BohrR,
	    	  -Gxyz[k][17],-Gxyz[k][18],-Gxyz[k][19]);
        }
        fprintf(fp_crd,"coordinates.forces>\n");
        fclose(fp_crd);
      }
      else
        printf("error(1) in MD_pac.c\n");
    }

  } /* if (myid==Host_ID) */

  /******************************************
              save *.rst4gopt.SD1 
  ******************************************/

  if (myid==Host_ID){

    sprintf(file_name,"%s%s_rst/%s.rst4gopt.SD1",filepath,filename,filename);

    if ((fp = fopen(file_name,"wb")) != NULL){

      fwrite(&Correct_Position_flag, sizeof(int), 1, fp);
      fwrite(&SD_scaling,      sizeof(double), 1, fp);
      fwrite(&SD_scaling_user, sizeof(double), 1, fp);
      fwrite(&Past_Utot,       sizeof(double), 10, fp);

      fclose(fp);
    }    
    else{
      printf("Failure of saving %s\n",file_name);
    }
  }

}



void Cell_Opt_SD(const OpenMXChemicalPotentials *chemical, OpenMXDynamicsState *dynamics, OpenMXThermodynamicEnergies *thermo, const OpenMXBandEnergies *energies, const OpenMXEnergyComponents *components, int iter, int SD_scaling_flag)
{
  /* 1au=2.4189*10^-2 fs, 1fs=41.341105 au */

  int i,j,k;
  double itv[4][4],dE_da[4][4];
  double ntv[4][4],proj_dE_da[4];
  double sum,detA,Max_Gradient;
  double My_Max_Force,Max_Force;

  int header_flag;
  double dt,SD_max,SD_min,SD_init,scale,MaxStep;
  double Wscale;
  char fileCoord[YOUSO10];
  char fileSD[YOUSO10];
  FILE *fp_crd,*fp_SD;
  int myid;
  char buf[fp_bsize];          /* setvbuf */
  char fileE[YOUSO10];
  char file_name[YOUSO10];
  FILE *fp;

  /* MPI */
  MPI_Comm_rank(mpi_comm_level1,&myid);

  MD_Opt_OK = 0;
  Wscale = unified_atomic_mass_unit/electron_mass;

  /******************************************
              read *.rst4gopt.SD1
  ******************************************/

  if (SuccessReadingfiles){

    sprintf(file_name,"%s%s_rst/%s.rst4gopt.SD1",filepath,filename,filename);

    if ((fp = fopen(file_name,"rb")) != NULL){

      fread(&Correct_Position_flag, sizeof(int), 1, fp);
      fread(&SD_scaling,      sizeof(double), 1, fp);
      fread(&SD_scaling_user, sizeof(double), 1, fp);
      fread(&Past_Utot,       sizeof(double), 10, fp);

      fclose(fp);
    }    
    else{
      printf("Failure of saving %s\n",file_name);
    }
  }

  /****************************************************
     calculate the inverse of a matrix consisiting 
     of the cell vectors   
  ****************************************************/

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
        sum += itv[k][i]*Stress_Tensor[3*(k-1)+(j-1)];
      }

      dE_da[i][j] = sum;
      if (Cell_Fixed_XYZ[i][j]==1) dE_da[i][j] = 0.0;
    }
  }

  /****************************************************
          find the maximum gradient dE/daij
  ****************************************************/

  Max_Gradient = 0.0;

  /* no constraint for cell vectors */
  /* while keeping the fractional coordinates */

  if (cellopt_switch==1){

    for (i=1; i<=3; i++){
      for (j=1; j<=3; j++){

	if (Max_Gradient<fabs(dE_da[i][j]) && Cell_Fixed_XYZ[i][j]==0){
	  Max_Gradient = fabs(dE_da[i][j]); 
	}
      }
    }
  }

  /* angles fixed for cell vectors */
  /* while keeping the fractional coordinates */

  else if (cellopt_switch==2){

    for (i=1; i<=3; i++){

      if (   Cell_Fixed_XYZ[i][1]==0
          && Cell_Fixed_XYZ[i][2]==0 
	  && Cell_Fixed_XYZ[i][3]==0 ){ 

	sum = 0.0;
	for (j=1; j<=3; j++){
	  sum += tv[i][j]*dE_da[i][j]; 
	}

        sum = sum/sqrt(tv[i][1]*tv[i][1]+tv[i][2]*tv[i][2]+tv[i][3]*tv[i][3]);

        if (Max_Gradient<fabs(sum)) Max_Gradient = fabs(sum);

      }
    }
  }

  /* angles fixed and |a1|=|a2|=|a3| for cell vectors */
  /* while keeping the fractional coordinates */

  else if (cellopt_switch==3){

    for (i=1; i<=3; i++){

      sum = 0.0;
      for (j=1; j<=3; j++){
	sum += tv[i][j]*dE_da[i][j]; 
      }

      sum = sum/sqrt(tv[i][1]*tv[i][1]+tv[i][2]*tv[i][2]+tv[i][3]*tv[i][3]);

      if (Max_Gradient<fabs(sum)) Max_Gradient = fabs(sum);
    }
  }

  /* angles fixed and |a1|=|a2|!=|a3| for cell vectors */
  /* while keeping the fractional coordinates */

  else if (cellopt_switch==4){

    for (i=1; i<=3; i++){

      sum = 0.0;
      for (j=1; j<=3; j++){
	sum += tv[i][j]*dE_da[i][j]; 
      }

      sum = sum/sqrt(tv[i][1]*tv[i][1]+tv[i][2]*tv[i][2]+tv[i][3]*tv[i][3]);

      if (Max_Gradient<fabs(sum)) Max_Gradient = fabs(sum);
    }
  }

  /* no constraint for atomic coordinates */

  else if (cellopt_switch==5 || cellopt_switch==6 || cellopt_switch==7){

    for (i=1; i<=3; i++){
      for (j=1; j<=3; j++){

	if (Max_Gradient<fabs(dE_da[i][j]) && Cell_Fixed_XYZ[i][j]==0){
	  Max_Gradient = fabs(dE_da[i][j]); 
	}
      }
    }

    My_Max_Force = 0.0;
#pragma omp parallel for schedule(static) reduction(max:My_Max_Force) if(Matomnum>=128)
    for (int Mc_AN=1; Mc_AN<=Matomnum; ++Mc_AN) {
      const int Gc_AN=M2G[Mc_AN];
      for (int j=1; j<=3; ++j) {
        if (atom_Fixed_XYZ[Gc_AN][j]==0) {
          const double force=fabs(Gxyz[Gc_AN][16+j]);
          if (My_Max_Force<force) My_Max_Force=force;
        }
      }
    }

    MPI_Allreduce(&My_Max_Force, &Max_Force, 1, MPI_DOUBLE, MPI_MAX, mpi_comm_level1);

    /*
    if (myid==0) printf("ABC1 Max_dE=%15.12f Max_F=%15.12f\n",Max_Gradient,Max_Force);
    */

    if (Max_Gradient<Max_Force) Max_Gradient = Max_Force;
  }

  /****************************************************
   MD_Opt_OK
  ****************************************************/

  if (Max_Gradient<MD_Opt_criterion){

     MD_Opt_OK = 1;

    /* show a message if the geometry optimization was achieved or not. */
    if (myid==Host_ID) printf("\nThe geometry optimization was achieved.\n");
  }

  /****************************************************
   set up SD_scaling
  ****************************************************/

  dt = 41.3411*2.0;
  SD_init = dt*dt/Wscale;
  SD_max = SD_init*15.0;   /* default 15   */
  SD_min = SD_init*0.04;   /* default 0.02 */

  if (iter==1 || SD_scaling_flag==0){

    SD_scaling_user = Max_Gradient/BohrR/1.5;
    SD_scaling = SD_scaling_user/(Max_Gradient+1.0e-10);

    if (SD_max<SD_scaling) SD_scaling = SD_max;
    if (SD_scaling<SD_min) SD_scaling = SD_min;
  }

  else{
    if (Past_Utot[1]<(thermo->total+thermo->pressure_volume) && iter%3==1){
      SD_scaling = SD_scaling/5.0;
    }
    else if (Past_Utot[1]<Past_Utot[2] && (thermo->total+thermo->pressure_volume)<Past_Utot[1] && iter%4==1){
      SD_scaling = SD_scaling*2.5;
    }

    if (SD_max<SD_scaling) SD_scaling = SD_max;
    if (SD_scaling<SD_min) SD_scaling = SD_min;

    Past_Utot[5] = Past_Utot[4];
    Past_Utot[4] = Past_Utot[3];
    Past_Utot[3] = Past_Utot[2];
    Past_Utot[2] = Past_Utot[1];
    Past_Utot[1] = thermo->total + thermo->pressure_volume;
  }
  
  if (myid==Host_ID && 0<level_stdout) printf("<Steepest_Descent>  SD_scaling=%15.12f\n",SD_scaling);

  /****************************************************
   write informatins to *.ene
  ****************************************************/

  if (myid==Host_ID){  

    sprintf(fileE,"%s%s.ene",filepath,filename);
    iterout_md(chemical, dynamics, thermo, energies, components, iter,MD_TimeStep*(iter-1),fileE);
  }

  /****************************************************
    update lattice vectors and cartesian coordinates
  ****************************************************/

  if (MD_Opt_OK!=1 && iter!=MD_IterNumber){

    /* avoid too large movement */
    if ( Criterion_Max_Step<(Max_Gradient*SD_scaling) )
      scale = Criterion_Max_Step/(Max_Gradient*SD_scaling);
    else 
      scale = 1.0; 

    /***************************************************
                    update lattice vectors
    ***************************************************/

    /* no constraint for cell vectors */
    /* while keeping the fractional coordinates */

    if (cellopt_switch==1){
    
      for (i=1; i<=3; i++){
	for (j=1; j<=3; j++){
	  if (Cell_Fixed_XYZ[i][j]==0){
	    tv[i][j] = tv[i][j] - scale*SD_scaling*dE_da[i][j];
	  }
	}
      }
    }

    /* angles fixed for cell vectors */
    /* while keeping the fractional coordinates */

    else if (cellopt_switch==2){

      for (i=1; i<=3; i++){

        if (   Cell_Fixed_XYZ[i][1]==0
            && Cell_Fixed_XYZ[i][2]==0 
            && Cell_Fixed_XYZ[i][3]==0 ){ 

	  sum = 0.0;
	  for (j=1; j<=3; j++){
            sum += tv[i][j]*dE_da[i][j]; 
	  }

          sum = sum/sqrt(tv[i][1]*tv[i][1]+tv[i][2]*tv[i][2]+tv[i][3]*tv[i][3]);

          tv[i][1] = tv[i][1] - scale*SD_scaling*sum*tv[i][1];
          tv[i][2] = tv[i][2] - scale*SD_scaling*sum*tv[i][2];
          tv[i][3] = tv[i][3] - scale*SD_scaling*sum*tv[i][3];
	}
      }
    }

    /* angles fixed and |a1|=|a2|=|a3| for cell vectors */
    /* while keeping the fractional coordinates */

    else if (cellopt_switch==3){

      double length_a[4],av_a;

      for (i=1; i<=3; i++){

        sum = 0.0;  
	for (j=1; j<=3; j++){
	  sum += tv[i][j]*dE_da[i][j]; 
	}
	sum = sum/sqrt(tv[i][1]*tv[i][1]+tv[i][2]*tv[i][2]+tv[i][3]*tv[i][3]);

        tv[i][1] = tv[i][1] - scale*SD_scaling*sum*tv[i][1];
        tv[i][2] = tv[i][2] - scale*SD_scaling*sum*tv[i][2];
        tv[i][3] = tv[i][3] - scale*SD_scaling*sum*tv[i][3];
      }

      for (i=1; i<=3; i++){
        length_a[i] = 0.0;  
	for (j=1; j<=3; j++){
	  length_a[i] += tv[i][j]*tv[i][j];
	}
        length_a[i] = sqrt(length_a[i]);
      }

      av_a = (length_a[1]+length_a[2]+length_a[3])/3.0; 

      for (i=1; i<=3; i++){
	for (j=1; j<=3; j++){
          tv[i][j] = tv[i][j]/length_a[i]*av_a;
	}
      }
    }

    /* angles fixed and |a1|=|a2|!=|a3| for cell vectors */
    /* while keeping the fractional coordinates */

    else if (cellopt_switch==4){

      double length_a[4],av_a;

      for (i=1; i<=3; i++){

        sum = 0.0;  
	for (j=1; j<=3; j++){
	  sum += tv[i][j]*dE_da[i][j]; 
	}
	sum = sum/sqrt(tv[i][1]*tv[i][1]+tv[i][2]*tv[i][2]+tv[i][3]*tv[i][3]);

        tv[i][1] = tv[i][1] - scale*SD_scaling*sum*tv[i][1];
        tv[i][2] = tv[i][2] - scale*SD_scaling*sum*tv[i][2];
        tv[i][3] = tv[i][3] - scale*SD_scaling*sum*tv[i][3];
      }

      for (i=1; i<=3; i++){
        length_a[i] = 0.0;  
	for (j=1; j<=3; j++){
	  length_a[i] += tv[i][j]*tv[i][j];
	}
        length_a[i] = sqrt(length_a[i]);
      }

      av_a = (length_a[1]+length_a[2])/2.0;      

      for (i=1; i<=2; i++){
	for (j=1; j<=3; j++){
          tv[i][j] = tv[i][j]/length_a[i]*av_a;
	}
      }
    }

    /* no constraint for atomic coordinates */

    else if (cellopt_switch==5 || cellopt_switch==6 || cellopt_switch==7){

      /* calculate dE_dq */

#pragma omp parallel for schedule(static) if(atomnum>=128)
      for (int Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
        for (int i=1; i<=3; i++){
          double sum = 0.0;
          for (int j=1; j<=3; j++){
            sum += Gxyz[Gc_AN][16+j]*tv[i][j];
	  }

          if (atom_Fixed_XYZ[Gc_AN][i]==1){
            Gxyz[Gc_AN][13+i] = 0.0;
	  }
          else{
            Gxyz[Gc_AN][13+i] = sum;
	  }
	}
      }

      /* update tv */

      for (i=1; i<=3; i++){
	for (j=1; j<=3; j++){
	  if (Cell_Fixed_XYZ[i][j]==0){
	    tv[i][j] = tv[i][j] - scale*SD_scaling*dE_da[i][j];
	  }
	}
      }
    }

    /***************************************************
                update cartesian coordinates
    ***************************************************/

    if (1<=cellopt_switch && cellopt_switch<=7) {
      const int relax_atoms=cellopt_switch>=5;
#pragma omp parallel for schedule(static) if(atomnum>=128)
      for (int Gc_AN=1; Gc_AN<=atomnum; ++Gc_AN) {
        for (int j=1; j<=3; ++j) {
          Gxyz[Gc_AN][j] = Cell_Gxyz[Gc_AN][1]*tv[1][j]
                         + Cell_Gxyz[Gc_AN][2]*tv[2][j]
                         + Cell_Gxyz[Gc_AN][3]*tv[3][j] + Grid_Origin[j];
          if (relax_atoms && atom_Fixed_XYZ[Gc_AN][j]==0)
            Gxyz[Gc_AN][j] -= 0.2*scale*SD_scaling*Gxyz[Gc_AN][16+j];
        }
      }
    }
  }

  /****************************************************
                show and save information
  ****************************************************/

  if (myid==Host_ID){ 

    if (0<level_stdout){ 

      printf("<Steepest_Descent>  |Maximum dE/da| (Hartree/Bohr) =%15.12f\n",
             Max_Gradient);
      printf("<Steepest_Descent>  Criterion       (Hartree/Bohr) =%15.12f\n",
	     MD_Opt_criterion);

      printf("\n");
      printf(" Cell vectors and derivatives of total energy with respect to them\n");

      printf(" a1(Ang.) =%10.5f %10.5f %10.5f   dE/da1(a.u.) =%10.5f %10.5f %10.5f\n",
             tv[1][1]*BohrR,tv[1][2]*BohrR,tv[1][3]*BohrR,dE_da[1][1],dE_da[1][2],dE_da[1][3]);
      printf(" a2(Ang.) =%10.5f %10.5f %10.5f   dE/da2(a.u.) =%10.5f %10.5f %10.5f\n",
             tv[2][1]*BohrR,tv[2][2]*BohrR,tv[2][3]*BohrR,dE_da[2][1],dE_da[2][2],dE_da[2][3]);
      printf(" a3(Ang.) =%10.5f %10.5f %10.5f   dE/da3(a.u.) =%10.5f %10.5f %10.5f\n",
             tv[3][1]*BohrR,tv[3][2]*BohrR,tv[3][3]*BohrR,dE_da[3][1],dE_da[3][2],dE_da[3][3]);


      if (1<level_stdout){ 

	printf("\n");
	for (i=1; i<=atomnum; i++){
	  printf("     atom=%4d, XYZ(ang) Fxyz(a.u.)=%9.4f %9.4f %9.4f  %9.4f %9.4f %9.4f\n",
		 i,BohrR*Gxyz[i][1],BohrR*Gxyz[i][2],BohrR*Gxyz[i][3],
		 Gxyz[i][17],Gxyz[i][18],Gxyz[i][19] ); fflush(stdout);
	}   
      }
    }

    strcpy(fileSD,".SD");
    fnjoint(filepath,filename,fileSD);

    fp_SD = fopen(fileSD, "r");
    if (fp_SD==NULL){
      header_flag = 1;
    }
    else{
      header_flag = 0;
      fclose(fp_SD);
    }  

    if ((fp_SD = fopen(fileSD,"a")) != NULL){

      setvbuf(fp_SD,buf,_IOFBF,fp_bsize);  /* setvbuf */

      if ( Criterion_Max_Step<(Max_Gradient*SD_scaling) )
	MaxStep = Criterion_Max_Step;
      else 
	MaxStep = SD_scaling*Max_Gradient;

      if (header_flag==1){

        fprintf(fp_SD,"\n");
        fprintf(fp_SD,"***********************************************************\n");
        fprintf(fp_SD,"***********************************************************\n");
        fprintf(fp_SD,"                History of cell optimization               \n");
        fprintf(fp_SD,"***********************************************************\n");
        fprintf(fp_SD,"***********************************************************\n\n");

        fprintf(fp_SD,"  MD_iter   SD_scaling     |Maximum force|   Maximum step        Utot             Enpy           Volume\n");
        fprintf(fp_SD,"                           (Hartree/Bohr)        (Ang)         (Hartree)        (Hartree)        (Ang^3)\n\n");
      }
      fprintf(fp_SD,"  %3d  %15.8f  %15.8f  %15.8f  %15.8f  %15.8f  %15.8f\n",
              iter,SD_scaling,Max_Gradient,MaxStep*BohrR, thermo->total, thermo->total+thermo->pressure_volume,Cell_Volume*0.14818474347690476628);
      fclose(fp_SD);
    }
    else
      printf("Error(7) in MD_pac.c\n");

    if (MD_Opt_OK==1 || iter==MD_IterNumber){

      strcpy(fileCoord,".crd");
      fnjoint(filepath,filename,fileCoord);
      if ((fp_crd = fopen(fileCoord,"w")) != NULL){

        setvbuf(fp_crd,buf,_IOFBF,fp_bsize);  /* setvbuf */

        fprintf(fp_crd,"\n");
        fprintf(fp_crd,"***********************************************************\n");
        fprintf(fp_crd,"***********************************************************\n");
        fprintf(fp_crd,"    Cell vectors (Ang.) and derivatives of total energy    \n");
        fprintf(fp_crd,"             with respect to them (Hartree/Bohr)           \n");
        fprintf(fp_crd,"***********************************************************\n");
        fprintf(fp_crd,"***********************************************************\n\n");

        fprintf(fp_crd," a1 =%10.5f %10.5f %10.5f   dE/da1 =%10.5f %10.5f %10.5f\n",
                tv[1][1]*BohrR,tv[1][2]*BohrR,tv[1][3]*BohrR,dE_da[1][1],dE_da[1][2],dE_da[1][3]);
        fprintf(fp_crd," a2 =%10.5f %10.5f %10.5f   dE/da2 =%10.5f %10.5f %10.5f\n",
               tv[2][1]*BohrR,tv[2][2]*BohrR,tv[2][3]*BohrR,dE_da[2][1],dE_da[2][2],dE_da[2][3]);
        fprintf(fp_crd," a3 =%10.5f %10.5f %10.5f   dE/da3 =%10.5f %10.5f %10.5f\n",
               tv[3][1]*BohrR,tv[3][2]*BohrR,tv[3][3]*BohrR,dE_da[3][1],dE_da[3][2],dE_da[3][3]);

        fprintf(fp_crd,"\n");
        fprintf(fp_crd,"***********************************************************\n");
        fprintf(fp_crd,"***********************************************************\n");
        fprintf(fp_crd,"       xyz-coordinates (Ang.) and forces (Hartree/Bohr)    \n");
        fprintf(fp_crd,"***********************************************************\n");
        fprintf(fp_crd,"***********************************************************\n\n");

        fprintf(fp_crd,"<coordinates.forces\n");
        fprintf(fp_crd,"  %i\n",atomnum);
        for (k=1; k<=atomnum; k++){
          i = WhatSpecies[k];
          j = Spe_WhatAtom[i];
          fprintf(fp_crd," %4d  %4s   %9.5f %9.5f %9.5f  %15.12f %15.12f %15.12f\n",
                  k,Atom_Symbol[j],
	          Gxyz[k][1]*BohrR,Gxyz[k][2]*BohrR,Gxyz[k][3]*BohrR,
	    	  -Gxyz[k][17],-Gxyz[k][18],-Gxyz[k][19]);
        }
        fprintf(fp_crd,"coordinates.forces>\n");
        fclose(fp_crd);
      }
      else
        printf("error(1) in MD_pac.c\n");
    }

  } /* if (myid==Host_ID) */

  /******************************************
              save *.rst4gopt.SD1 
  ******************************************/

  if (myid==Host_ID){

    sprintf(file_name,"%s%s_rst/%s.rst4gopt.SD1",filepath,filename,filename);

    if ((fp = fopen(file_name,"wb")) != NULL){

      fwrite(&Correct_Position_flag, sizeof(int), 1, fp);
      fwrite(&SD_scaling,      sizeof(double), 1, fp);
      fwrite(&SD_scaling_user, sizeof(double), 1, fp);
      fwrite(&Past_Utot,       sizeof(double), 10, fp);

      fclose(fp);
    }    
    else{
      printf("Failure of saving %s\n",file_name);
    }
  }

}


static void Update_Cell_Opt_History(int count, int vectors, const double gradient[4][4])
{
#pragma omp parallel for schedule(static) if(atomnum>=128)
  for (int atom=1; atom<=atomnum; ++atom) {
    for (int step=count-2; step>=0; --step) {
      memcpy(GxyzHistoryIn[step+1][atom]+1,GxyzHistoryIn[step][atom]+1,3*sizeof(double));
      memcpy(GxyzHistoryR[step+1][atom]+1,GxyzHistoryR[step][atom]+1,3*sizeof(double));
    }
    memcpy(GxyzHistoryIn[0][atom]+1,Cell_Gxyz[atom]+1,3*sizeof(double));
    for (int i=1; i<=3; ++i) {
      double sum=0.0;
      if (atom_Fixed_XYZ[atom][i]!=1)
        for (int j=1; j<=3; ++j) sum+=Gxyz[atom][16+j]*tv[i][j];
      Gxyz[atom][13+i]=GxyzHistoryR[0][atom][i]=sum;
    }
  }

  for (int v=1; v<=vectors; ++v) {
    const int site=atomnum+v;
    for (int step=count-2; step>=0; --step) {
      memcpy(GxyzHistoryIn[step+1][site]+1,GxyzHistoryIn[step][site]+1,3*sizeof(double));
      memcpy(GxyzHistoryR[step+1][site]+1,GxyzHistoryR[step][site]+1,3*sizeof(double));
    }
    memcpy(GxyzHistoryIn[0][site]+1,tv[v]+1,3*sizeof(double));
    memcpy(GxyzHistoryR[0][site]+1,gradient[v]+1,3*sizeof(double));
  }
}


void Cell_Opt_RF(OpenMXGeometryOpt *geometry, const OpenMXChemicalPotentials *chemical, OpenMXDynamicsState *dynamics, OpenMXThermodynamicEnergies *thermo, const OpenMXBandEnergies *energies, const OpenMXEnergyComponents *components, int iter)
{
  int i,j,k,diis_iter;
  double dE_da[4][4],detA,itv[4][4],sum;
  double sMD_TimeStep;
  static int local_iter=1;
  static int SD_iter=0,GDIIS_iter=0;
  static int flag=0;
  static int Every_iter;
  char file_name[YOUSO10];
  FILE *fp;
  int tmp_array[10];
  int everyiter,buf_iter;
  int myid;

  MPI_Comm_rank(mpi_comm_level1,&myid);

  /******************************************
              read *.rst4gopt.RF1
  ******************************************/

  if (SuccessReadingfiles){

    sprintf(file_name,"%s%s_rst/%s.rst4gopt.RF1",filepath,filename,filename);

    if ((fp = fopen(file_name,"rb")) != NULL){

      fread(tmp_array, sizeof(int), 6, fp);

      local_iter            = tmp_array[0];
      SD_iter               = tmp_array[1];
      GDIIS_iter            = tmp_array[2];
      flag                  = tmp_array[3];
      Every_iter            = tmp_array[4];
      Correct_Position_flag = tmp_array[5];
 
      fread(&SD_scaling,      sizeof(double), 1, fp);
      fread(&SD_scaling_user, sizeof(double), 1, fp);
      fread(&Past_Utot,       sizeof(double), 10, fp);

      fclose(fp);
    }    
    else{
      printf("Failure of saving %s\n",file_name);
    }
  }

  /****************************************************
     calculate the inverse of a matrix consisiting 
     of the cell vectors
     and
     calculate dE_da   
  ****************************************************/

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

  /* calculate dE_d(a1,a2,a3) */

  for (i=1; i<=3; i++){
    for (j=1; j<=3; j++){

      sum = 0.0;
      for (k=1; k<=3; k++){
        sum += itv[k][i]*Stress_Tensor[3*(k-1)+(j-1)];
      }

      if (Cell_Fixed_XYZ[i][j]==1){
        dE_da[i][j] = 0.0;
      }
      else{
        dE_da[i][j] = sum;
      }
    }
  }

  /* set parameters */

  Every_iter = OptEveryDIIS;

  if (iter<M_GDIIS_HISTORY)
    diis_iter = iter;
  else   
    diis_iter = M_GDIIS_HISTORY;

  /* increament of iter */

  if (iter<OptStartDIIS){
    flag = 0;     
  }

  else if (iter==OptStartDIIS){
    flag = 1;     
    GDIIS_iter++;
  }

  else if (flag==0){
    SD_iter++; 
  }  
  else if (flag==1){
    GDIIS_iter++;
  }

  /* SD */

  if (flag==0){

    if (SD_iter==1)
      Cell_Opt_SD(chemical, dynamics, thermo, energies, components, iter,0);
    else 
      Cell_Opt_SD(chemical, dynamics, thermo, energies, components, iter,1);

    Update_Cell_Opt_History(diis_iter,3,dE_da);

    /* initialize local_iter */

    local_iter = 1;

  }

  /* RFC */

  else {
 
    if (   cellopt_switch==5 
        || cellopt_switch==6 
        || cellopt_switch==7 ){

       RFC5(geometry, chemical, dynamics, thermo, energies, components, local_iter,iter,dE_da,itv);

    }
    else {

      if (myid==Host_ID){
        printf("The optimizer is not supported.\n");
      }

      MPI_Finalize();
      exit(1);
    }

    local_iter++;
  }

  /* check the number of iterations */

  if (Every_iter<=SD_iter){
    flag = 1;
    SD_iter = 0;
    GDIIS_iter = 0;
  }
  
  else if (Every_iter<=GDIIS_iter){
    flag = 0;
    SD_iter = 0;
    GDIIS_iter = 0;
  }
  
  /******************************************
              save *.rst4gopt.RF1 
  ******************************************/

  if (myid==Host_ID){
  
    sprintf(file_name,"%s%s_rst/%s.rst4gopt.RF1",filepath,filename,filename);
  
    if ((fp = fopen(file_name,"wb")) != NULL){
  
      tmp_array[0] = local_iter;
      tmp_array[1] = SD_iter;
      tmp_array[2] = GDIIS_iter;
      tmp_array[3] = flag;
      tmp_array[4] = Every_iter;
      tmp_array[5] = Correct_Position_flag;
   
      fwrite(tmp_array, sizeof(int), 6, fp);
  
      fwrite(&SD_scaling,      sizeof(double), 1, fp);
      fwrite(&SD_scaling_user, sizeof(double), 1, fp);
      fwrite(&Past_Utot,       sizeof(double), 10, fp);
  
      fclose(fp);
    }    
    else{
      printf("Failure of saving %s\n",file_name);
    }
  }

}




/*BEGIN hmweng*/
void Geometry_Opt_RF(OpenMXGeometryOpt *geometry, const OpenMXChemicalPotentials *chemical, OpenMXDynamicsState *dynamics, OpenMXThermodynamicEnergies *thermo, const OpenMXBandEnergies *energies, const OpenMXEnergyComponents *components, int iter)
{
  int i,iatom,k,diis_iter;
  double sMD_TimeStep;
  static int local_iter=1;
  static int SD_iter=0,GDIIS_iter=0;
  static int flag=0;
  static int Every_iter;
  char file_name[YOUSO10];
  FILE *fp;
  int tmp_array[10];
  int everyiter,buf_iter;
  int numprocs,myid,ID;  

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);
  MPI_Barrier(mpi_comm_level1);

  /******************************************
              read *.rst4gopt.RF1
  ******************************************/

  if (SuccessReadingfiles){

    sprintf(file_name,"%s%s_rst/%s.rst4gopt.RF1",filepath,filename,filename);

    if ((fp = fopen(file_name,"rb")) != NULL){

      fread(tmp_array, sizeof(int), 6, fp);

      local_iter            = tmp_array[0];
      SD_iter               = tmp_array[1];
      GDIIS_iter            = tmp_array[2];
      flag                  = tmp_array[3];
      Every_iter            = tmp_array[4];
      Correct_Position_flag = tmp_array[5];
 
      fread(&SD_scaling,      sizeof(double), 1, fp);
      fread(&SD_scaling_user, sizeof(double), 1, fp);
      fread(&Past_Utot,       sizeof(double), 10, fp);

      fclose(fp);
    }    
    else{
      printf("Failure of saving %s\n",file_name);
    }
  }

  /* set parameters */

  Every_iter = OptEveryDIIS;

  if (iter<M_GDIIS_HISTORY)
    diis_iter = iter;
  else   
    diis_iter = M_GDIIS_HISTORY;

  /* increament of iter */

  if (iter<OptStartDIIS){
    flag = 0;     
  }

  else if (iter==OptStartDIIS){
    flag = 1;     
    GDIIS_iter++;
  }

  else if (flag==0){
    SD_iter++; 
  }  
  else if (flag==1){
    GDIIS_iter++;
  }

  /* SD */

  if (flag==0){

    if (SD_iter==1)
      Steepest_Descent(geometry, chemical, dynamics, thermo, energies, components, iter,0);
    else 
      Steepest_Descent(geometry, chemical, dynamics, thermo, energies, components, iter,1);

    /* shift one */

    for (i=(diis_iter-2); 0<=i; i--) {
      for (iatom=1; iatom<=atomnum; iatom++) {
	for (k=1; k<=3; k++) {
	  GxyzHistoryIn[i+1][iatom][k] = GxyzHistoryIn[i][iatom][k];
	  GxyzHistoryR[i+1][iatom][k]  = GxyzHistoryR[i][iatom][k];
	}
      }
    }

    /* add GxyzHistoryIn and GxyzHisotryR */

    for (iatom=1; iatom<=atomnum; iatom++) {
      for (k=1; k<=3; k++) {
        if (atom_Fixed_XYZ[iatom][k]==0){
	  GxyzHistoryIn[0][iatom][k] = Gxyz[iatom][k];
	  GxyzHistoryR[0][iatom][k]  = Gxyz[iatom][16+k];
	}
        else{
	  GxyzHistoryIn[0][iatom][k] = Gxyz[iatom][k];
	  GxyzHistoryR[0][iatom][k]  = 0.0;
        }
      }
    }

    /* initialize local_iter */

    local_iter = 1;

  }

  /* GDIIS */

  else {

    RF(geometry, chemical, dynamics, thermo, energies, components, local_iter,iter);
    local_iter++;
  }

  /* check the number of iterations */

  if (Every_iter<=SD_iter){
    flag = 1;
    SD_iter = 0;
    GDIIS_iter = 0;
  }

  else if (Every_iter<=GDIIS_iter){
    flag = 0;
    SD_iter = 0;
    GDIIS_iter = 0;
  }

  /******************************************
              save *.rst4gopt.RF1 
  ******************************************/

  if (myid==Host_ID){

    sprintf(file_name,"%s%s_rst/%s.rst4gopt.RF1",filepath,filename,filename);

    if ((fp = fopen(file_name,"wb")) != NULL){

      tmp_array[0] = local_iter;
      tmp_array[1] = SD_iter;
      tmp_array[2] = GDIIS_iter;
      tmp_array[3] = flag;
      tmp_array[4] = Every_iter;
      tmp_array[5] = Correct_Position_flag;
 
      fwrite(tmp_array, sizeof(int), 6, fp);

      fwrite(&SD_scaling,      sizeof(double), 1, fp);
      fwrite(&SD_scaling_user, sizeof(double), 1, fp);
      fwrite(&Past_Utot,       sizeof(double), 10, fp);

      fclose(fp);
    }    
    else{
      printf("Failure of saving %s\n",file_name);
    }
  }

}
/*END hmweng */


void Geometry_Opt_DIIS(OpenMXGeometryOpt *geometry, const OpenMXChemicalPotentials *chemical, OpenMXDynamicsState *dynamics, OpenMXThermodynamicEnergies *thermo, const OpenMXBandEnergies *energies, const OpenMXEnergyComponents *components, int iter)
{
  int i,iatom,k,diis_iter;
  double sMD_TimeStep;
  static int local_iter=1;
  static int SD_iter=0,GDIIS_iter=0;
  static int flag=0;
  static int Every_iter;
  char file_name[YOUSO10];
  FILE *fp;
  int tmp_array[10];
  int everyiter,buf_iter;
  int numprocs,myid,ID;  

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);
  MPI_Barrier(mpi_comm_level1);

  /******************************************
              read *.rst4gopt.DIIS1
  ******************************************/

  if (SuccessReadingfiles){

    sprintf(file_name,"%s%s_rst/%s.rst4gopt.DIIS1",filepath,filename,filename);

    if ((fp = fopen(file_name,"rb")) != NULL){

      fread(tmp_array, sizeof(int), 6, fp);

      local_iter            = tmp_array[0];
      SD_iter               = tmp_array[1];
      GDIIS_iter            = tmp_array[2];
      flag                  = tmp_array[3];
      Every_iter            = tmp_array[4];
      Correct_Position_flag = tmp_array[5];
 
      fread(&SD_scaling,      sizeof(double), 1, fp);
      fread(&SD_scaling_user, sizeof(double), 1, fp);
      fread(&Past_Utot,       sizeof(double), 10, fp);

      fclose(fp);
    }    
    else{
      printf("Failure of saving %s\n",file_name);
    }
  }

  /* set parameters */

  Every_iter = OptEveryDIIS;

  if (iter<M_GDIIS_HISTORY)
    diis_iter = iter;
  else   
    diis_iter = M_GDIIS_HISTORY;

  /* increament of iter */

  if (iter<OptStartDIIS){
    flag = 0;     
  }

  else if (iter==OptStartDIIS){
    flag = 1;     
    GDIIS_iter++;
  }

  else if (flag==0){
    SD_iter++; 
  }  
  else if (flag==1){
    GDIIS_iter++;
  }

  /* SD */

  if (flag==0){

    if (SD_iter==1)
      Steepest_Descent(geometry, chemical, dynamics, thermo, energies, components, iter,0);
    else 
      Steepest_Descent(geometry, chemical, dynamics, thermo, energies, components, iter,1);

    /* shift one */

    for (i=(diis_iter-2); 0<=i; i--) {
      for (iatom=1; iatom<=atomnum; iatom++) {
	for (k=1; k<=3; k++) {
	  GxyzHistoryIn[i+1][iatom][k] = GxyzHistoryIn[i][iatom][k];
	  GxyzHistoryR[i+1][iatom][k]  = GxyzHistoryR[i][iatom][k];
	}
      }
    }

    /* add GxyzHistoryIn and GxyzHisotryR */

    for (iatom=1; iatom<=atomnum; iatom++) {
      for (k=1; k<=3; k++) {
        if (atom_Fixed_XYZ[iatom][k]==0){
	  GxyzHistoryIn[0][iatom][k] = Gxyz[iatom][k];
	  GxyzHistoryR[0][iatom][k]  = Gxyz[iatom][16+k];
	}
        else{
	  GxyzHistoryIn[0][iatom][k] = Gxyz[iatom][k];
	  GxyzHistoryR[0][iatom][k]  = 0.0;
        }
      }
    }

    /* initialize local_iter */

    local_iter = 1;

  }

  /* GDIIS */

  else {

    GDIIS(geometry, chemical, dynamics, thermo, energies, components, local_iter,iter);
    local_iter++;
  }

  /* check the number of iterations */

  if (Every_iter<=SD_iter){
    flag = 1;
    SD_iter = 0;
    GDIIS_iter = 0;
  }

  else if (Every_iter<=GDIIS_iter){
    flag = 0;
    SD_iter = 0;
    GDIIS_iter = 0;
  }

  /******************************************
              save *.rst4gopt.DIIS1 
  ******************************************/

  if (myid==Host_ID){

    sprintf(file_name,"%s%s_rst/%s.rst4gopt.DIIS1",filepath,filename,filename);

    if ((fp = fopen(file_name,"wb")) != NULL){

      tmp_array[0] = local_iter;
      tmp_array[1] = SD_iter;
      tmp_array[2] = GDIIS_iter;
      tmp_array[3] = flag;
      tmp_array[4] = Every_iter;
      tmp_array[5] = Correct_Position_flag;
 
      fwrite(tmp_array, sizeof(int), 6, fp);

      fwrite(&SD_scaling,      sizeof(double), 1, fp);
      fwrite(&SD_scaling_user, sizeof(double), 1, fp);
      fwrite(&Past_Utot,       sizeof(double), 10, fp);

      fclose(fp);
    }    
    else{
      printf("Failure of saving %s\n",file_name);
    }
  }

}




void Geometry_Opt_DIIS_BFGS(OpenMXGeometryOpt *geometry, const OpenMXChemicalPotentials *chemical, OpenMXDynamicsState *dynamics, OpenMXThermodynamicEnergies *thermo, const OpenMXBandEnergies *energies, const OpenMXEnergyComponents *components, int iter)
{
  int i,iatom,k,diis_iter;
  double sMD_TimeStep;
  static int local_iter=1;
  static int SD_iter=0,GDIIS_iter=0;
  static int flag=0;
  static int Every_iter;
  char file_name[YOUSO10];
  FILE *fp;
  int tmp_array[10];
  int everyiter,buf_iter;
  int numprocs,myid,ID;  

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);
  MPI_Barrier(mpi_comm_level1);

  /******************************************
              read *.rst4gopt.BFGS1
  ******************************************/

  if (SuccessReadingfiles){

    sprintf(file_name,"%s%s_rst/%s.rst4gopt.BFGS1",filepath,filename,filename);

    if ((fp = fopen(file_name,"rb")) != NULL){

      fread(tmp_array, sizeof(int), 6, fp);

      local_iter            = tmp_array[0];
      SD_iter               = tmp_array[1];
      GDIIS_iter            = tmp_array[2];
      flag                  = tmp_array[3];
      Every_iter            = tmp_array[4];
      Correct_Position_flag = tmp_array[5];
 
      fread(&SD_scaling,      sizeof(double), 1, fp);
      fread(&SD_scaling_user, sizeof(double), 1, fp);
      fread(&Past_Utot,       sizeof(double), 10, fp);

      fclose(fp);
    }    
    else{
      printf("Failure of saving %s\n",file_name);
    }
  }

  /* set parameters */

  Every_iter = OptEveryDIIS;

  if (iter<M_GDIIS_HISTORY)
    diis_iter = iter;
  else   
    diis_iter = M_GDIIS_HISTORY;

  /* increament of iter */

  if (iter<OptStartDIIS){
    flag = 0;     
  }

  else if (iter==OptStartDIIS){
    flag = 1;     
    GDIIS_iter++;
  }

  else if (flag==0){
    SD_iter++; 
  }  
  else if (flag==1){
    GDIIS_iter++;
  }

  /* SD */

  if (flag==0){

    if (SD_iter==1)
      Steepest_Descent(geometry, chemical, dynamics, thermo, energies, components, iter,0);
    else 
      Steepest_Descent(geometry, chemical, dynamics, thermo, energies, components, iter,1);

    /* shift one */

    for (i=(diis_iter-2); 0<=i; i--) {
      for (iatom=1; iatom<=atomnum; iatom++) {
	for (k=1; k<=3; k++) {
	  GxyzHistoryIn[i+1][iatom][k] = GxyzHistoryIn[i][iatom][k];
	  GxyzHistoryR[i+1][iatom][k]  = GxyzHistoryR[i][iatom][k];
	}
      }
    }

    /* add GxyzHistoryIn and GxyzHisotryR */

    for (iatom=1; iatom<=atomnum; iatom++) {
      for (k=1; k<=3; k++) {
        if (atom_Fixed_XYZ[iatom][k]==0){
	  GxyzHistoryIn[0][iatom][k] = Gxyz[iatom][k];
	  GxyzHistoryR[0][iatom][k]  = Gxyz[iatom][16+k];
	}
        else{
	  GxyzHistoryIn[0][iatom][k] = Gxyz[iatom][k];
	  GxyzHistoryR[0][iatom][k]  = 0.0;
        }
      }
    }

    /* initialize local_iter */

    local_iter = 1;

  }

  /* GDIIS */

  else {
    GDIIS_BFGS(geometry, chemical, dynamics, thermo, energies, components, local_iter,iter);
    local_iter++;
  }

  /* check the number of iterations */

  if (Every_iter<=SD_iter){
    flag = 1;
    SD_iter = 0;
    GDIIS_iter = 0;
  }

  else if (Every_iter<=GDIIS_iter){
    flag = 0;
    SD_iter = 0;
    GDIIS_iter = 0;
  }

  /******************************************
              save *.rst4gopt.BFGS1 
  ******************************************/

  if (myid==Host_ID){

    sprintf(file_name,"%s%s_rst/%s.rst4gopt.BFGS1",filepath,filename,filename);

    if ((fp = fopen(file_name,"wb")) != NULL){

      tmp_array[0] = local_iter;
      tmp_array[1] = SD_iter;
      tmp_array[2] = GDIIS_iter;
      tmp_array[3] = flag;
      tmp_array[4] = Every_iter;
      tmp_array[5] = Correct_Position_flag;
 
      fwrite(tmp_array, sizeof(int), 6, fp);

      fwrite(&SD_scaling,      sizeof(double), 1, fp);
      fwrite(&SD_scaling_user, sizeof(double), 1, fp);
      fwrite(&Past_Utot,       sizeof(double), 10, fp);

      fclose(fp);
    }    
    else{
      printf("Failure of saving %s\n",file_name);
    }
  }
}


void Geometry_Opt_DIIS_EF(OpenMXGeometryOpt *geometry, const OpenMXChemicalPotentials *chemical, OpenMXDynamicsState *dynamics, OpenMXThermodynamicEnergies *thermo, const OpenMXBandEnergies *energies, const OpenMXEnergyComponents *components, int iter)
{
  int i,iatom,k,diis_iter;
  double sMD_TimeStep;
  static int local_iter=1;
  static int SD_iter=0,GDIIS_iter=0;
  static int flag=0;
  static int Every_iter;
  char file_name[YOUSO10];
  FILE *fp;
  int tmp_array[10];
  int everyiter,buf_iter;
  int numprocs,myid,ID;  

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);
  MPI_Barrier(mpi_comm_level1);

  /******************************************
              read *.rst4gopt.EF1
  ******************************************/

  if (SuccessReadingfiles){

    sprintf(file_name,"%s%s_rst/%s.rst4gopt.EF1",filepath,filename,filename);

    if ((fp = fopen(file_name,"rb")) != NULL){

      fread(tmp_array, sizeof(int), 6, fp);

      local_iter            = tmp_array[0];
      SD_iter               = tmp_array[1];
      GDIIS_iter            = tmp_array[2];
      flag                  = tmp_array[3];
      Every_iter            = tmp_array[4];
      Correct_Position_flag = tmp_array[5];
 
      fread(&SD_scaling,      sizeof(double), 1, fp);
      fread(&SD_scaling_user, sizeof(double), 1, fp);
      fread(&Past_Utot,       sizeof(double), 10, fp);

      fclose(fp);
    }    
    else{
      printf("Failure of saving %s\n",file_name);
    }
  }

  /* set parameters */

  Every_iter = OptEveryDIIS;

  if (iter<M_GDIIS_HISTORY)
    diis_iter = iter;
  else   
    diis_iter = M_GDIIS_HISTORY;

  /* increament of iter */

  if (iter<OptStartDIIS){
    flag = 0;     
  }

  else if (iter==OptStartDIIS){
    flag = 1;     
    GDIIS_iter++;
  }

  else if (flag==0){
    SD_iter++; 
  }  
  else if (flag==1){
    GDIIS_iter++;
  }

  /* SD */

  if (flag==0){

    if (SD_iter==1)
      Steepest_Descent(geometry, chemical, dynamics, thermo, energies, components, iter,0);
    else 
      Steepest_Descent(geometry, chemical, dynamics, thermo, energies, components, iter,1);

    /* shift one */

    for (i=(diis_iter-2); 0<=i; i--) {
      for (iatom=1; iatom<=atomnum; iatom++) {
	for (k=1; k<=3; k++) {
	  GxyzHistoryIn[i+1][iatom][k] = GxyzHistoryIn[i][iatom][k];
	  GxyzHistoryR[i+1][iatom][k]  = GxyzHistoryR[i][iatom][k];
	}
      }
    }

    /* add GxyzHistoryIn and GxyzHisotryR */

    for (iatom=1; iatom<=atomnum; iatom++) {
      for (k=1; k<=3; k++) {
        if (atom_Fixed_XYZ[iatom][k]==0){
	  GxyzHistoryIn[0][iatom][k] = Gxyz[iatom][k];
	  GxyzHistoryR[0][iatom][k]  = Gxyz[iatom][16+k];
	}
        else{
	  GxyzHistoryIn[0][iatom][k] = Gxyz[iatom][k];
	  GxyzHistoryR[0][iatom][k]  = 0.0;
        }
      }
    }

    /* initialize local_iter */

    local_iter = 1;

  }

  /* GDIIS */

  else {

    GDIIS_EF(geometry, chemical, dynamics, thermo, energies, components, local_iter,iter);
    local_iter++;
  }

  /* check the number of iterations */

  if (Every_iter<=SD_iter){
    flag = 1;
    SD_iter = 0;
    GDIIS_iter = 0;
  }

  else if (Every_iter<=GDIIS_iter){
    flag = 0;
    SD_iter = 0;
    GDIIS_iter = 0;
  }

  /******************************************
              save *.rst4gopt.EF1 
  ******************************************/

  if (myid==Host_ID){

    sprintf(file_name,"%s%s_rst/%s.rst4gopt.EF1",filepath,filename,filename);

    if ((fp = fopen(file_name,"wb")) != NULL){

      tmp_array[0] = local_iter;
      tmp_array[1] = SD_iter;
      tmp_array[2] = GDIIS_iter;
      tmp_array[3] = flag;
      tmp_array[4] = Every_iter;
      tmp_array[5] = Correct_Position_flag;
 
      fwrite(tmp_array, sizeof(int), 6, fp);

      fwrite(&SD_scaling,      sizeof(double), 1, fp);
      fwrite(&SD_scaling_user, sizeof(double), 1, fp);
      fwrite(&Past_Utot,       sizeof(double), 10, fp);

      fclose(fp);
    }    
    else{
      printf("Failure of saving %s\n",file_name);
    }
  }
  
}



void GDIIS(OpenMXGeometryOpt *geometry, const OpenMXChemicalPotentials *chemical, OpenMXDynamicsState *dynamics, OpenMXThermodynamicEnergies *thermo, const OpenMXBandEnergies *energies, const OpenMXEnergyComponents *components, int iter, int iter0)
{
  /* 1au=2.4189*10^-2 fs, 1fs=41.341105 au */

  char *func_name="DIIS";
  double *A,*B,sumB;
  double mixing,force_Max;
  static double sMD_TimeStep,dx_max=0.05; 
  double diff_dx,diff,Max_Step;
  int header_flag;
  INTEGER i,j,k,iatom;
  int diis_iter;
  char fileCoord[YOUSO10];
  char fileSD[YOUSO10];
  FILE *fp_crd,*fp_SD;
  char buf[fp_bsize];          /* setvbuf */
  char fileE[YOUSO10];

  /* variables for MPI */
  int myid;

  /* MPI myid */
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /* share Gxyz */
  OpenMX_Geometry_Share_Atoms(geometry,Gxyz);

  if (iter<M_GDIIS_HISTORY)
    diis_iter = iter;
  else   
    diis_iter = M_GDIIS_HISTORY;

  /* shift one */

  for (i=(diis_iter-2); 0<=i; i--) {
    for (iatom=1; iatom<=atomnum; iatom++) {
      for (k=1; k<=3; k++) {
	GxyzHistoryIn[i+1][iatom][k] = GxyzHistoryIn[i][iatom][k];
	GxyzHistoryR[i+1][iatom][k]  = GxyzHistoryR[i][iatom][k];
      }
    }
  }

  /* add GxyzHistoryIn and GxyzHisotryR */

  for (iatom=1; iatom<=atomnum; iatom++)   {

    for (k=1;k<=3;k++) {
      if (atom_Fixed_XYZ[iatom][k]==0){
	GxyzHistoryIn[0][iatom][k] = Gxyz[iatom][k];
	GxyzHistoryR[0][iatom][k]  = Gxyz[iatom][16+k];
      }
      else{
	GxyzHistoryIn[0][iatom][k] = Gxyz[iatom][k];
	GxyzHistoryR[0][iatom][k]  = 0.0;
      }
    }
  }

  if (myid!=Host_ID)   goto Last_Bcast; 

  /*********************** for myid==Host_ID **************************/

  /* allocation of arrays */

  /* Max of force */

  force_Max=0.0;
  for (iatom=1;iatom<=atomnum;iatom++)   {
    for (k=1;k<=3;k++) {
      if (atom_Fixed_XYZ[iatom][k]==0){
	if (force_Max< fabs(Gxyz[iatom][16+k]) ) force_Max = fabs(Gxyz[iatom][16+k]);
      }
    }
  }

  sMD_TimeStep = 0.05/(0.01*41.341105);

  if (2<=level_stdout){
    printf("<%s>  |Maximum force| (Hartree/Bohr) = %15.12f tuned_dt= %f\n",func_name,force_Max, sMD_TimeStep);
    printf("<%s>  Criterion      (Hartree/Bohr) = %15.12f\n", func_name, MD_Opt_criterion);
  }

  OpenMX_Geometry_Prepare_DIIS(geometry,GxyzHistoryR,atomnum,diis_iter,&A,&B);

  if (2<=level_stdout){
    printf("<%s>  DIIS matrix\n",func_name);
    for (i=0; i<(diis_iter+1); i++) {
      printf("<%s> ",func_name);
      for (j=0; j<(diis_iter+1); j++) {
        printf("%6.3f ",A[i*(diis_iter+1)+j]);
      }
      printf("\n");
    }
  }

  /* lapack routine */

  if (2<=level_stdout){
    printf("M_GDIIS_HISTORY=%2d diis_iter=%2d\n",M_GDIIS_HISTORY,diis_iter);
  }

  const int info=OpenMX_Geometry_Solve_DIIS(geometry,diis_iter,A,B);

  if (info!=0) {
    printf("<%s> dsysv_, info=%d\n",func_name,info);
    printf("<%s> \n",func_name);
    printf("<%s> ERROR, aborting\n",func_name);
    printf("<%s> \n",func_name);

    MD_Opt_OK =1; 
    /* no change */

    goto Last_Bcast ;
  }

  if (2<=level_stdout){
    printf("<%s> diis alpha=",func_name);
    sumB = 0;
    for (i=0; i<diis_iter; i++) {
      printf("%f ",B[i]);
      sumB += B[i];
    }
    printf("%lf\n",B[diis_iter]);
  }

  if (force_Max<MD_Opt_criterion ){

     MD_Opt_OK = 1;

    /* show a message if the geometry optimization was achieved or not. */
    if (myid==Host_ID) printf("\nThe geometry optimization was achieved.\n");
  }

  /****************************************************
   write informatins to *.ene
  ****************************************************/

  if (myid==Host_ID){  

    sprintf(fileE,"%s%s.ene",filepath,filename);
    iterout_md(chemical, dynamics, thermo, energies, components, iter,MD_TimeStep*(iter-1),fileE);
  }

  if (MD_Opt_OK!=1 && iter!=MD_IterNumber){

    /* initialize */

    for (iatom=1; iatom<=atomnum; iatom++) {
      for (j=1; j<=3; j++) {
	Gxyz[iatom][j] = 0.0;
      }
    }

    /* add tilde{R} */

    for (iatom=1; iatom<=atomnum; iatom++) {
      for (j=1; j<=3; j++) {
	if (atom_Fixed_XYZ[iatom][j]==0){
	  for (i=0; i<diis_iter; i++) {
	    Gxyz[iatom][j] += GxyzHistoryR[i][iatom][j]*B[i];
	  }
	}
      }
    }

    sumB = 0.0;
    for (iatom=1; iatom<=atomnum; iatom++) {
      for (j=1; j<=3; j++) {
	sumB += Gxyz[iatom][j]*Gxyz[iatom][j] ;
      }
    }

    sumB = sqrt(sumB)/(double)atomnum;

    if (2<=level_stdout){
      printf("<%s> |tilde{R}|=%E\n",func_name, sumB);
    }

    if      (1.0e-2<sumB)  mixing =-0.2;
    else if (1.0e-3<sumB)  mixing =-0.3;
    else if (1.0e-4<sumB)  mixing =-0.4;
    else if (1.0e-5<sumB)  mixing =-0.5;
    else if (1.0e-6<sumB)  mixing =-0.6;
    else                   mixing =-0.7;

    if (2<=level_stdout){
      printf("<%s> mixing=%15.12f\n",func_name,mixing);
    }

    for (iatom=1; iatom<=atomnum; iatom++) {
      for (j=1;j<=3;j++) {
	Gxyz[iatom][j] *= mixing;
      }
    }

    /* tilde{x} */

    for (iatom=1;iatom<=atomnum;iatom++) {
      for (j=1;j<=3;j++) {
	for (i=0; i<diis_iter; i++) {
	  Gxyz[iatom][j] += GxyzHistoryIn[i][iatom][j]*B[i]; 
	}
      }
    }

    /************************************************************
        In case of a too large updating, do a modest updating
    ************************************************************/  

    Max_Step = 0.0;
    for (iatom=1; iatom<=atomnum; iatom++) {
      for (j=1; j<=3; j++) {

        diff = fabs(Gxyz[iatom][j] - GxyzHistoryIn[0][iatom][j]);
        if (Max_Step<diff) Max_Step = diff;
      }
    }

    if (Criterion_Max_Step<Max_Step){

      for (iatom=1; iatom<=atomnum; iatom++) {
	for (j=1; j<=3; j++) {

	  diff = Gxyz[iatom][j] - GxyzHistoryIn[0][iatom][j];
	  Gxyz[iatom][j] = GxyzHistoryIn[0][iatom][j] + diff/Max_Step*Criterion_Max_Step;
	}
      }
    }

    /* find Max_Step */
    Max_Step = 0.0;
    for (iatom=1; iatom<=atomnum; iatom++) {
      for (j=1; j<=3; j++) {

        diff = fabs(Gxyz[iatom][j] - GxyzHistoryIn[0][iatom][j]);
        if (Max_Step<diff) Max_Step = diff;
      }
    }

    if (2<=level_stdout){

      printf("<%s> diff_x= %f , dE= %f\n",func_name, Max_Step, fabs(thermo->total-Past_Utot[1]) );

      /* print atomic positions */
      printf("<%s> atomnum= %d\n",func_name,atomnum);
      for (i=1; i<=atomnum; i++){
	j = Spe_WhatAtom[WhatSpecies[i]];
	printf("  %3d %s XYZ(ang) Fxyz(a.u.)= %9.4f %9.4f %9.4f  %9.4f %9.4f %9.4f\n",
	       i,Atom_Symbol[j],
	       BohrR*Gxyz[i][1],BohrR*Gxyz[i][2],BohrR*Gxyz[i][3],
	       Gxyz[i][17],Gxyz[i][18],Gxyz[i][19] ); 
      }   
    }

  } /* if (MD_Opt_OK!=1 && iter!=MD_IterNumber) */

  Past_Utot[1]=thermo->total;

  Max_Force = force_Max;
  SD_scaling_user = SD_scaling*Max_Force*0.2;


  /*********************** end of "myid==Host_ID" **************************/

 Last_Bcast: 

  MPI_Bcast(&MD_Opt_OK,1,MPI_INT, Host_ID, mpi_comm_level1);

  OpenMX_Geometry_Publish(geometry,Gxyz,NULL);


  if (myid==Host_ID){ 

    if (0<level_stdout){
      printf("<%s>  |Maximum force| (Hartree/Bohr) =%15.12f\n",
	     func_name,Max_Force);fflush(stdout);
      printf("<%s>  Criterion       (Hartree/Bohr) =%15.12f\n",
	     func_name,MD_Opt_criterion);fflush(stdout);

      printf("\n");
      for (i=1; i<=atomnum; i++){
	printf("     atom=%4d, XYZ(ang) Fxyz(a.u.)=%9.4f %9.4f %9.4f  %9.4f %9.4f %9.4f\n",
	       i,BohrR*Gxyz[i][1],BohrR*Gxyz[i][2],BohrR*Gxyz[i][3],
	       Gxyz[i][17],Gxyz[i][18],Gxyz[i][19] ); fflush(stdout);
      }   
    }

    strcpy(fileSD,".SD");
    fnjoint(filepath,filename,fileSD);

    fp_SD = fopen(fileSD, "r");
    if (fp_SD==NULL){
      header_flag = 1;
    }
    else{
      header_flag = 0;
      fclose(fp_SD);
    }  

    if ((fp_SD = fopen(fileSD,"a")) != NULL){

      setvbuf(fp_SD,buf,_IOFBF,fp_bsize);  /* setvbuf */

      if (header_flag==1){

        fprintf(fp_SD,"\n");
        fprintf(fp_SD,"***********************************************************\n");
        fprintf(fp_SD,"***********************************************************\n");
        fprintf(fp_SD,"              History of geometry optimization             \n");
        fprintf(fp_SD,"***********************************************************\n");
        fprintf(fp_SD,"***********************************************************\n\n");


        fprintf(fp_SD,"  MD_iter   SD_scaling     |Maximum force|   Maximum step        Utot\n");
        fprintf(fp_SD,"                           (Hartree/Bohr)        (Ang)         (Hartree)\n\n");
      }

      fprintf(fp_SD,"  %3d  %15.8f  %15.8f  %15.8f  %15.8f\n",
              iter0,SD_scaling,Max_Force,Max_Step*BohrR,thermo->total);
      fclose(fp_SD);
    }
    else{
      printf("Could not open a file in MD_pac.!\n");
    }

    if (MD_Opt_OK==1 || iter0==MD_IterNumber){

      strcpy(fileCoord,".crd");
      fnjoint(filepath,filename,fileCoord);
      if ((fp_crd = fopen(fileCoord,"w")) != NULL){

        setvbuf(fp_crd,buf,_IOFBF,fp_bsize);  /* setvbuf */

        fprintf(fp_crd,"\n");
        fprintf(fp_crd,"***********************************************************\n");
        fprintf(fp_crd,"***********************************************************\n");
        fprintf(fp_crd,"       xyz-coordinates (Ang) and forces (Hartree/Bohr)  \n");
        fprintf(fp_crd,"***********************************************************\n");
        fprintf(fp_crd,"***********************************************************\n\n");

        fprintf(fp_crd,"<coordinates.forces\n");
        fprintf(fp_crd,"  %i\n",atomnum);
        for (k=1; k<=atomnum; k++){
          i = WhatSpecies[k];
          j = Spe_WhatAtom[i];
          fprintf(fp_crd," %4d  %4s   %9.5f %9.5f %9.5f  %15.12f %15.12f %15.12f\n",
                  k,Atom_Symbol[j],
	          Gxyz[k][1]*BohrR,Gxyz[k][2]*BohrR,Gxyz[k][3]*BohrR,
	    	  -Gxyz[k][17],-Gxyz[k][18],-Gxyz[k][19]);
        }
        fprintf(fp_crd,"coordinates.forces>\n");
        fclose(fp_crd);
      }
      else
        printf("error(1) in MD_pac.c\n");
    }

  } /* if (myid==Host_ID) */

}




/*hmweng start RF method*/
void RF(OpenMXGeometryOpt *geometry, const OpenMXChemicalPotentials *chemical, OpenMXDynamicsState *dynamics, OpenMXThermodynamicEnergies *thermo, const OpenMXBandEnergies *energies, const OpenMXEnergyComponents *components, int iter, int iter0)
{
  /* 1au=2.4189*10^-2 fs, 1fs=41.341105 au */

  int m1,m2;
  int header_flag;
  char *func_name="RF";
  double dt,diff,Max_Step;
  double *A,*B,sumB;
  double mixing,force_Max;
  double itv[4][4];
  static double sMD_TimeStep,dx_max=0.05; 
  double diff_dx;
  INTEGER i,j,k,iatom;
  int diis_iter;
  char fileCoord[YOUSO10];
  char fileSD[YOUSO10];
  FILE *fp_crd,*fp_SD;
  char buf[fp_bsize];          /* setvbuf */
  char fileE[YOUSO10];

  /* variables for MPI */
  int myid;

  /* MPI myid */
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /* share Gxyz */
  OpenMX_Geometry_Share_Atoms(geometry,Gxyz);

  /* set diis_iter */

  if (iter<M_GDIIS_HISTORY)
    diis_iter = iter;
  else   
    diis_iter = M_GDIIS_HISTORY;

  /* shift one */

  for (i=(diis_iter-2); 0<=i; i--) {
    for (iatom=1; iatom<=atomnum; iatom++) {
      for (k=1; k<=3; k++) {
	GxyzHistoryIn[i+1][iatom][k] = GxyzHistoryIn[i][iatom][k];
	GxyzHistoryR[i+1][iatom][k]  = GxyzHistoryR[i][iatom][k];
      }
    }
  }

  /* add GxyzHistoryIn and GxyzHisotryR */

  for (iatom=1; iatom<=atomnum; iatom++)   {

    for (k=1;k<=3;k++) {
      if (atom_Fixed_XYZ[iatom][k]==0){
	GxyzHistoryIn[0][iatom][k] =  Gxyz[iatom][k];
	GxyzHistoryR[0][iatom][k]  =  Gxyz[iatom][16+k];
      }
      else{
	GxyzHistoryIn[0][iatom][k] = Gxyz[iatom][k];
	GxyzHistoryR[0][iatom][k]  = 0.0;
      }
    }
  }

  /* set the initial approximate Hessian */

  if (iter%30==1){

    if (iter0<M_GDIIS_HISTORY)
      Estimate_Initial_Hessian(geometry, iter0-1,0,itv);
    else   
      Estimate_Initial_Hessian(geometry, M_GDIIS_HISTORY-1,0,itv);
  }

  if (myid!=Host_ID) goto Last_Bcast; 

  /*********************** for myid==Host_ID **************************/

  /* allocation of arrays */

  /* Max of force */

  force_Max=0.0;
  for (iatom=1; iatom<=atomnum; iatom++)   {
    for (k=1;k<=3;k++) {
      if (atom_Fixed_XYZ[iatom][k]==0){
	if (force_Max< fabs(Gxyz[iatom][16+k]) ) force_Max = fabs(Gxyz[iatom][16+k]);
      }
    }
  }

  sMD_TimeStep = 0.05/(0.01*41.341105);

  if (2<=level_stdout){
    printf("<%s>  |Maximum force| (Hartree/Bohr) = %15.12f tuned_dt= %f\n",func_name,force_Max, sMD_TimeStep);
    printf("<%s>  Criterion      (Hartree/Bohr) = %15.12f\n", func_name, MD_Opt_criterion);
  }

  OpenMX_Geometry_Prepare_DIIS(geometry,GxyzHistoryR,atomnum,diis_iter,&A,&B);

  if (2<=level_stdout){
    printf("<%s>  DIIS matrix\n",func_name);
    for (i=0; i<(diis_iter+1); i++) {
      printf("<%s> ",func_name);
      for (j=0; j<(diis_iter+1); j++) {
        printf("%6.3f ",A[i*(diis_iter+1)+j]);
      }
      printf("\n");
    }
  }

  /* lapack routine */

  if (2<=level_stdout){
    printf("M_GDIIS_HISTORY=%2d diis_iter=%2d\n",M_GDIIS_HISTORY,diis_iter);
  }

  const int info=OpenMX_Geometry_Solve_DIIS(geometry,diis_iter,A,B);

  if (info!=0) {
    printf("<%s> dsysv_, info=%d\n",func_name,info);
    printf("<%s> \n",func_name);
    printf("<%s> ERROR, aborting\n",func_name);
    printf("<%s> \n",func_name);

    MD_Opt_OK =1; 
    /* no change */

    goto Last_Bcast ;
  }

  if (2<=level_stdout){
    printf("<%s> diis alpha=",func_name);
    sumB = 0;
    for (i=0; i<diis_iter; i++) {
      printf("%f ",B[i]);
      sumB += B[i];
    }
    printf("%lf\n",B[diis_iter]);
  }

  if (force_Max<MD_Opt_criterion ){

    MD_Opt_OK = 1;
    Max_Force = force_Max;

    /* show a message if the geometry optimization was achieved or not. */
    if (myid==Host_ID) printf("\nThe geometry optimization was achieved.\n");
  }

  /****************************************************
   write informatins to *.ene
  ****************************************************/

  if (myid==Host_ID){  

    sprintf(fileE,"%s%s.ene",filepath,filename);
    iterout_md(chemical, dynamics, thermo, energies, components, iter,MD_TimeStep*(iter-1),fileE);
  }

  if (MD_Opt_OK!=1 && iter!=MD_IterNumber){

    /* initialize */

    for (iatom=1; iatom<=atomnum; iatom++) {
      for (j=1; j<=3; j++) {
	Gxyz[iatom][j] = 0.0;
      }
    }

    /* add tilde{R} */

    for (iatom=1; iatom<=atomnum; iatom++) {
      for (j=1; j<=3; j++) {
	if (atom_Fixed_XYZ[iatom][j]==0){
	  for (i=0; i<diis_iter; i++) {
	    Gxyz[iatom][j] += GxyzHistoryR[i][iatom][j]*B[i];
	  }
	}
      }
    }

    sumB = 0.0;
    for (iatom=1; iatom<=atomnum; iatom++) {
      for (j=1; j<=3; j++) {
	sumB += Gxyz[iatom][j]*Gxyz[iatom][j];
      }
    }

    sumB = sqrt(sumB)/(double)atomnum;

    /*********************************************************************************
     update an approximate Hessian matrix 
   
     H_k = H_{k-1} + y*y^t/(s^t*y) - H_{k-1}*s * s^t * H_{k-1}^t /(s^t * H_{k-1} * s)
     y = GxyzHistoryR[0][iatom][k]  - GxyzHistoryR[1][iatom][k]
     s = GxyzHistoryIn[0][iatom][k] - GxyzHistoryIn[1][iatom][k]
    *********************************************************************************/

    if (iter!=1)
      OpenMX_Geometry_BFGS(geometry,Hessian,GxyzHistoryIn,GxyzHistoryR,atomnum,OPENMX_GEOMETRY_RF_HESSIAN);

    /*
    if (myid==Host_ID){  
      printf("Hessian\n");
      for (i=1; i<=3*atomnum; i++){     
	for (j=1; j<=3*atomnum; j++){     
	  printf("%8.4f ",Hessian[i][j]);
	}
	printf("\n");
      }
    }
    */

    /************************************************************
            Construct augmented Hessian matrix  
           | H    g | ( s )         ( s ) 
           |        |       = lamda             
           | g    0 | ( 1 )         ( 1 )
    ************************************************************/
  
    Hessian[3*atomnum+1][3*atomnum+1]=0.0;
  
    m2 = 0;
    for (i=1; i<=atomnum; i++){   
      for (k=1; k<=3; k++){   
	m2++;
	Hessian[3*atomnum+1][m2]=Gxyz[i][k];
	Hessian[m2][3*atomnum+1]=Gxyz[i][k];  
      }	
    }
  
    /************************************************************
     find the lowest eigenvalue and corresponding eigenvector
               of the augmented Hessian matrix
    ************************************************************/  

    OpenMX_Geometry_RF_Step(geometry,Hessian,3*atomnum+1,
                            scf_eigen_lib_flag==GPUSOLVER,level_stdout);

    if (2<=level_stdout){
      printf("<%s> |tilde{R}|=%E\n",func_name, sumB);
    }

    /* initialize */

    for (iatom=1; iatom<=atomnum; iatom++) {
      for (j=1;j<=3;j++) {
	Gxyz[iatom][j] = 0.0;
      }
    }

    /* calculate the DIIS coordinates tilde{x} */

    m1 = 0;
    for (iatom=1;iatom<=atomnum;iatom++) {
      for (j=1;j<=3;j++) {

	for (i=0; i<diis_iter; i++) {
	  Gxyz[iatom][j] += GxyzHistoryIn[i][iatom][j]*B[i]; 
	}

	/* a quasi Newton method */ 

	m1++;
	Gxyz[iatom][j] += Hessian[0][m1]; 

      }
    }

    /************************************************************
        In case of a too large updating, do a modest updating
    ************************************************************/  

    Max_Step = 0.0;
    for (iatom=1; iatom<=atomnum; iatom++) {
      for (j=1; j<=3; j++) {

        diff = fabs(Gxyz[iatom][j] - GxyzHistoryIn[0][iatom][j]);
        if (Max_Step<diff) Max_Step = diff;
      }
    }

    if (Criterion_Max_Step<Max_Step){

      for (iatom=1; iatom<=atomnum; iatom++) {
	for (j=1; j<=3; j++) {

	  diff = Gxyz[iatom][j] - GxyzHistoryIn[0][iatom][j];
	  Gxyz[iatom][j] = GxyzHistoryIn[0][iatom][j] + diff/Max_Step*Criterion_Max_Step;
	}
      }
    }

    /* find Max_Step */
    Max_Step = 0.0;
    for (iatom=1; iatom<=atomnum; iatom++) {
      for (j=1; j<=3; j++) {

        diff = fabs(Gxyz[iatom][j] - GxyzHistoryIn[0][iatom][j]);
        if (Max_Step<diff) Max_Step = diff;
      }
    }

    /************************************************************
                   show cooridinates and gradients
    ************************************************************/  

    if (2<=level_stdout){

      printf("<%s> diff_x= %f , dE= %f\n",func_name, Max_Step, fabs(thermo->total-Past_Utot[1]) );

      /* print atomic positions */
      printf("<%s> atomnum= %d\n",func_name,atomnum);
      for (i=1; i<=atomnum; i++){
	j = Spe_WhatAtom[WhatSpecies[i]];
	printf("  %3d %s XYZ(ang) Fxyz(a.u.)= %9.4f %9.4f %9.4f  %9.4f %9.4f %9.4f\n",
	       i,Atom_Symbol[j],
	       BohrR*Gxyz[i][1],BohrR*Gxyz[i][2],BohrR*Gxyz[i][3],
	       Gxyz[i][17],Gxyz[i][18],Gxyz[i][19] ); 
      }   
    }

    Past_Utot[1]=thermo->total;

    Max_Force = force_Max;
    SD_scaling_user = SD_scaling*Max_Force*0.2;
  } /* if (MD_Opt_OK!=1 && iter!=MD_IterNumber) */

  /*********************** end of "myid==Host_ID" **************************/

 Last_Bcast: 

  MPI_Bcast(&MD_Opt_OK,1,MPI_INT, Host_ID, mpi_comm_level1);

  OpenMX_Geometry_Publish(geometry,Gxyz,NULL);

  if (myid==Host_ID){ 

    if (0<level_stdout){ 

      printf("<%s>  |Maximum force| (Hartree/Bohr) =%15.12f\n",
	     func_name,Max_Force);fflush(stdout);
      printf("<%s>  Criterion       (Hartree/Bohr) =%15.12f\n",
	     func_name,MD_Opt_criterion);fflush(stdout);

      printf("\n");
      for (i=1; i<=atomnum; i++){
	printf("     atom=%4d, XYZ(ang) Fxyz(a.u.)=%9.4f %9.4f %9.4f  %9.4f %9.4f %9.4f\n",
	       i,BohrR*Gxyz[i][1],BohrR*Gxyz[i][2],BohrR*Gxyz[i][3],
	       Gxyz[i][17],Gxyz[i][18],Gxyz[i][19] ); fflush(stdout);
      }   
    }

    strcpy(fileSD,".SD");
    fnjoint(filepath,filename,fileSD);

    fp_SD = fopen(fileSD, "r");
    if (fp_SD==NULL){
      header_flag = 1;
    }
    else{
      header_flag = 0;
      fclose(fp_SD);
    }  

    if ((fp_SD = fopen(fileSD,"a")) != NULL){

      setvbuf(fp_SD,buf,_IOFBF,fp_bsize);  /* setvbuf */

      if (header_flag==1){

        fprintf(fp_SD,"\n");
        fprintf(fp_SD,"***********************************************************\n");
        fprintf(fp_SD,"***********************************************************\n");
        fprintf(fp_SD,"              History of geometry optimization             \n");
        fprintf(fp_SD,"***********************************************************\n");
        fprintf(fp_SD,"***********************************************************\n\n");


        fprintf(fp_SD,"  MD_iter   SD_scaling     |Maximum force|   Maximum step        Utot\n");
        fprintf(fp_SD,"                           (Hartree/Bohr)        (Ang)         (Hartree)\n\n");
      }

      fprintf(fp_SD,"  %3d  %15.8f  %15.8f  %15.8f  %15.8f\n",
              iter0,SD_scaling,Max_Force,Max_Step*BohrR,thermo->total);
      fclose(fp_SD);
    }
    else{
      printf("Could not open a file in MD_pac.!\n");
    }

    if (MD_Opt_OK==1 || iter0==MD_IterNumber){

      strcpy(fileCoord,".crd");
      fnjoint(filepath,filename,fileCoord);
      if ((fp_crd = fopen(fileCoord,"w")) != NULL){

        setvbuf(fp_crd,buf,_IOFBF,fp_bsize);  /* setvbuf */

        fprintf(fp_crd,"\n");
        fprintf(fp_crd,"***********************************************************\n");
        fprintf(fp_crd,"***********************************************************\n");
        fprintf(fp_crd,"       xyz-coordinates (Ang) and forces (Hartree/Bohr)  \n");
        fprintf(fp_crd,"***********************************************************\n");
        fprintf(fp_crd,"***********************************************************\n\n");

        fprintf(fp_crd,"<coordinates.forces\n");
        fprintf(fp_crd,"  %i\n",atomnum);
        for (k=1; k<=atomnum; k++){
          i = WhatSpecies[k];
          j = Spe_WhatAtom[i];
          fprintf(fp_crd," %4d  %4s   %9.5f %9.5f %9.5f  %15.12f %15.12f %15.12f\n",
                  k,Atom_Symbol[j],
	          Gxyz[k][1]*BohrR,Gxyz[k][2]*BohrR,Gxyz[k][3]*BohrR,
	    	  -Gxyz[k][17],-Gxyz[k][18],-Gxyz[k][19]);
        }
        fprintf(fp_crd,"coordinates.forces>\n");
        fclose(fp_crd);
      }
      else
        printf("error(1) in MD_pac.c\n");
    }

  } /* if (myid==Host_ID) */

}



void RFC5(OpenMXGeometryOpt *geometry, const OpenMXChemicalPotentials *chemical, OpenMXDynamicsState *dynamics, OpenMXThermodynamicEnergies *thermo, const OpenMXBandEnergies *energies, const OpenMXEnergyComponents *components, int iter, int iter0, double dE_da[4][4], double itv[4][4])
{
  /* 1au=2.4189*10^-2 fs, 1fs=41.341105 au */

  int m1,m2;
  int header_flag;
  char *func_name="RF";
  double dt,diff,Max_Step,sum;
  double *A,*B,sumB;
  double ntv[4][4];
  double mixing,force_Max,Max_Gradient;
  static double sMD_TimeStep,dx_max=0.05; 
  double diff_dx;
  INTEGER i,j,k,iatom,icell;
  int diis_iter;
  char fileCoord[YOUSO10];
  char fileSD[YOUSO10];
  FILE *fp_crd,*fp_SD;
  char buf[fp_bsize];          /* setvbuf */
  char fileE[YOUSO10];
  int Gc_AN,nvecs;
  int myid;

  /* MPI myid */
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /* share Gxyz */
  OpenMX_Geometry_Share_Atoms(geometry,Gxyz);

  /* set the number of vectors to be changed */

  if      (cellopt_switch==5) nvecs = 3;
  else if (cellopt_switch==6) nvecs = 2;
  else if (cellopt_switch==7) nvecs = 1;

  /* set diis_iter */

  if (iter<M_GDIIS_HISTORY)
    diis_iter = iter;
  else   
    diis_iter = M_GDIIS_HISTORY;

  Update_Cell_Opt_History(diis_iter,nvecs,dE_da);

  /* set the initial approximate Hessian */

  if (iter==1){

    if (iter0<M_GDIIS_HISTORY)
      Estimate_Initial_Hessian(geometry, iter0-1,1,itv);
    else
      Estimate_Initial_Hessian(geometry, M_GDIIS_HISTORY-1,1,itv);
  }

  if (myid!=Host_ID) goto Last_Bcast;

  /*********************** for myid==Host_ID **************************/

  /* The convergence test includes both cell and Cartesian gradients. */
  Max_Gradient = 0.0;
  for (i=1; i<=3; ++i)
    for (j=1; j<=3; ++j) Max_Gradient = fmax(Max_Gradient,fabs(dE_da[i][j]));

  force_Max = 0.0;
#pragma omp parallel for schedule(static) reduction(max:force_Max) if(atomnum>=128) private(k)
  for (iatom=1; iatom<=atomnum; ++iatom)
    for (k=1; k<=3; ++k) force_Max = fmax(force_Max,fabs(Gxyz[iatom][16+k]));
  Max_Gradient = fmax(Max_Gradient,force_Max);
  sMD_TimeStep = 0.05/(0.01*41.341105);

  if (2<=level_stdout){
    printf("<%s>  |Maximum gradient| (Hartree/Bohr) = %15.12f tuned_dt= %f\n",func_name,Max_Gradient, sMD_TimeStep);
    printf("<%s>  Criterion      (Hartree/Bohr) = %15.12f\n", func_name, MD_Opt_criterion);
  }

  OpenMX_Geometry_Prepare_DIIS(geometry,GxyzHistoryR,atomnum+nvecs,diis_iter,&A,&B);

  if (2<=level_stdout && myid==0){
    printf("<%s>  DIIS matrix\n",func_name);
    for (i=0; i<(diis_iter+1); i++) {
      printf("<%s> ",func_name);
      for (j=0; j<(diis_iter+1); j++) {
        printf("%6.3f ",A[i*(diis_iter+1)+j]);
      }
      printf("\n");
    }
  }

  /* lapack routine */

  if (2<=level_stdout){
    printf("<%s> M_GDIIS_HISTORY=%2d diis_iter=%2d\n",func_name,M_GDIIS_HISTORY,diis_iter);
  }

  const int info=OpenMX_Geometry_Solve_DIIS(geometry,diis_iter,A,B);

  if (info!=0) {
    printf("<%s> dsysv_, info=%d\n",func_name,info);
    printf("<%s> \n",func_name);
    printf("<%s> ERROR, aborting\n",func_name);
    printf("<%s> \n",func_name);

    MD_Opt_OK =1; 
    /* no change */

    goto Last_Bcast ;
  }

  if (2<=level_stdout){
    printf("<%s> diis alpha=",func_name);
    sumB = 0;
    for (i=0; i<diis_iter; i++) {
      printf("%f ",B[i]);
      sumB += B[i];
    }
    printf("%lf\n",B[diis_iter]);
  }

  if (Max_Gradient<MD_Opt_criterion ){
    MD_Opt_OK = 1;
    Max_Force = Max_Gradient;

    /* show a message if the geometry optimization was achieved or not. */
    if (myid==Host_ID) printf("\nThe geometry optimization was achieved.\n");
  }

  /****************************************************
               write informatins to *.ene
  ****************************************************/

  if (myid==Host_ID){  

    sprintf(fileE,"%s%s.ene",filepath,filename);
    iterout_md(chemical, dynamics, thermo, energies, components, iter,MD_TimeStep*(iter-1),fileE);
  }

  if (MD_Opt_OK!=1 && iter!=MD_IterNumber){

    /* initialize Gxyz */

    for (iatom=1; iatom<=(atomnum+nvecs); iatom++) {
      for (j=1; j<=3; j++) {

        /* store coordinates at the previous step */

        if (iatom<=atomnum) 
          Gxyz[iatom][20+j] = Gxyz[iatom][j]; 
        else 
          Gxyz[iatom][20+j] = tv[iatom-atomnum][j]; 

        /* initialize */
	Gxyz[iatom][j] = 0.0;
      }
    }

    /* add tilde{R} for Gxyz */

    for (iatom=1; iatom<=atomnum; iatom++) {
      for (j=1; j<=3; j++) {
	for (i=0; i<diis_iter; i++) {
	  Gxyz[iatom][j] += GxyzHistoryR[i][iatom][j]*B[i];
	}
      }
    }

    /* add tilde{R} for tv */

    for (icell=1; icell<=nvecs; icell++) {
      for (j=1; j<=3; j++) {
	for (i=0; i<diis_iter; i++) {
	  Gxyz[atomnum+icell][j] += GxyzHistoryR[i][atomnum+icell][j]*B[i];
	}
      }
    }

    /* calculate the norm of tilde{R} */

    sumB = 0.0;

    for (iatom=1; iatom<=(atomnum+nvecs); iatom++) {
      for (j=1; j<=3; j++) {
	sumB += Gxyz[iatom][j]*Gxyz[iatom][j];
      }
    }

    sumB = sqrt(sumB)/(double)(atomnum+nvecs);

    /*********************************************************************************
     update an approximate Hessian matrix 
   
     H_k = H_{k-1} + y*y^t/(s^t*y) - H_{k-1}*s * s^t * H_{k-1}^t /(s^t * H_{k-1} * s)
     y = GxyzHistoryR[0][iatom][k]  - GxyzHistoryR[1][iatom][k]
     s = GxyzHistoryIn[0][iatom][k] - GxyzHistoryIn[1][iatom][k]
    *********************************************************************************/

    if (iter!=1)
      OpenMX_Geometry_BFGS(geometry,Hessian,GxyzHistoryIn,GxyzHistoryR,atomnum+nvecs,OPENMX_GEOMETRY_RF_HESSIAN);

    /* taking account of constraints for lattice vectors */

    m1 = 0;
    for (i=1; i<=3; i++){     
      for (j=1; j<=3; j++){     

        m1++;

        if (Cell_Fixed_XYZ[i][j]==1){

	  for (k=1; k<=(3*atomnum+9); k++){     
            Hessian[3*atomnum+m1][k] = 0.0;
            Hessian[k][3*atomnum+m1] = 0.0;
	  }

          Hessian[3*atomnum+m1][3*atomnum+m1] = 100.0;
	} 
      }
    }

    /* taking account of constraints for fractional coordinates */

    m1 = 0;
    for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
      for (i=1; i<=3; i++){

        m1++;

        if (atom_Fixed_XYZ[Gc_AN][i]==1){

	  for (k=1; k<=(3*atomnum+9); k++){     
            Hessian[m1][k] = 0.0;
            Hessian[k][m1] = 0.0;
	  }

          Hessian[m1][m1] = 200.0;
	}
      }
    }

    /*
    if (myid==Host_ID){  
      printf("Hessian\n");
      for (i=1; i<=(3*atomnum+nvecs*3); i++){     
	for (j=1; j<=(3*atomnum+nvecs*3); j++){     
	  printf("%6.3f ",Hessian[i][j]);
	}
	printf("\n");
      }
    }
    */

    /************************************************************
            Construct augmented Hessian matrix  
           | H    g | ( s )         ( s ) 
           |        |       = lamda             
           | g    0 | ( 1 )         ( 1 )
    ************************************************************/
  
    Hessian[3*atomnum+nvecs*3+1][3*atomnum+nvecs*3+1] = 0.0;
    
    m2 = 0;
    for (i=1; i<=(atomnum+nvecs); i++){   
      for (k=1; k<=3; k++){   
	m2++;
	Hessian[3*atomnum+nvecs*3+1][m2] = Gxyz[i][k];
	Hessian[m2][3*atomnum+nvecs*3+1] = Gxyz[i][k];  
      }	
    }
    
    /************************************************************
     find the lowest eigenvalue and corresponding eigenvector
               of the augmented Hessian matrix
    ************************************************************/  

    OpenMX_Geometry_RF_Step(geometry,Hessian,3*(atomnum+nvecs)+1,
                            scf_eigen_lib_flag==GPUSOLVER,level_stdout);

    if (2<=level_stdout){
      printf("<%s> |tilde{R}|=%E\n",func_name, sumB);
    }

    /**************************************************
      update fractional coordinates and cell vectors 
    **************************************************/

    /* Mix fractional and lattice coordinates and apply the RF step. */
#pragma omp parallel for schedule(static) if(atomnum>=128)
    for (int site=1; site<=atomnum+nvecs; ++site) {
      for (int j=1; j<=3; ++j) {
        double mixed=0.0;
        for (int step=0; step<diis_iter; ++step)
          mixed+=GxyzHistoryIn[step][site][j]*B[step];
        Cell_Gxyz[site][j]=mixed+Hessian[0][3*(site-1)+j];
      }
    }

    /************************************
                    set tv 
    ************************************/

    for (i=1; i<=nvecs; i++){
      for (j=1; j<=3; j++){
	tv[i][j] = Cell_Gxyz[atomnum+i][j];
      }
    }

    /* Convert to Cartesian coordinates and measure the largest component step. */
    Max_Step=0.0;
#pragma omp parallel for schedule(static) reduction(max:Max_Step) if(atomnum>=128)
    for (int atom=1; atom<=atomnum; ++atom) {
      for (int j=1; j<=3; ++j) {
        Gxyz[atom][j]=Cell_Gxyz[atom][1]*tv[1][j]
                     +Cell_Gxyz[atom][2]*tv[2][j]
                     +Cell_Gxyz[atom][3]*tv[3][j]+Grid_Origin[j];
        Max_Step=fmax(Max_Step,fabs(Gxyz[atom][j]-Gxyz[atom][20+j]));
      }
    }

    /* cell vectors */
    
    for (i=1; i<=nvecs; i++) {
      for (j=1; j<=3; j++) {
    
        diff = fabs(tv[i][j] - Gxyz[atomnum+i][20+j]);
        if (Max_Step<diff) Max_Step = diff;
      }
    }
    
    /******************************************************
     if (Criterion_Max_Step<Max_Step), correct the update
    ******************************************************/

    if (Criterion_Max_Step<Max_Step){

      /* atomic coordinates */

      for (iatom=1; iatom<=atomnum; iatom++) {
	for (j=1; j<=3; j++) {

	  diff = Gxyz[iatom][j] - Gxyz[iatom][20+j];
	  Gxyz[iatom][j] = Gxyz[iatom][20+j] + diff/Max_Step*Criterion_Max_Step;
	}
      }

      /* cell vectors */

      for (i=1; i<=nvecs; i++) {
	for (j=1; j<=3; j++) {

	  diff = tv[i][j] - Gxyz[atomnum+i][20+j];
	  tv[i][j] = Gxyz[atomnum+i][20+j] + diff/Max_Step*Criterion_Max_Step;
	}
      }
    }

    /*************************
          find Max_Step 
    *************************/

    Max_Step = 0.0;

    /* atomic coordinates */

    for (iatom=1; iatom<=atomnum; iatom++) {
      for (j=1; j<=3; j++) {

        diff = fabs(Gxyz[iatom][j] - Gxyz[iatom][20+j]);
        if (Max_Step<diff) Max_Step = diff;
      }
    }

    /* cell vectors */

    for (i=1; i<=nvecs; i++) {
      for (j=1; j<=3; j++) {

        diff = fabs(tv[i][j] - Gxyz[atomnum+i][20+j]);
        if (Max_Step<diff) Max_Step = diff;
      }
    }

    /************************************************************
                   show cooridinates and gradients
    ************************************************************/  

    if (2<=level_stdout){

      printf("<%s> diff_x= %f , dE= %f\n",func_name, Max_Step, fabs(thermo->total-Past_Utot[1]) );

      /* print atomic positions */
      printf("<%s> atomnum= %d\n",func_name,atomnum);
      for (i=1; i<=atomnum; i++){
	j = Spe_WhatAtom[WhatSpecies[i]];
	printf("  %3d %s XYZ(ang) Fxyz(a.u.)= %9.4f %9.4f %9.4f  %9.4f %9.4f %9.4f\n",
	       i,Atom_Symbol[j],
	       BohrR*Gxyz[i][1],BohrR*Gxyz[i][2],BohrR*Gxyz[i][3],
	       Gxyz[i][17],Gxyz[i][18],Gxyz[i][19] ); 
      }   
    }

    Past_Utot[1]=thermo->total;

    Max_Force = Max_Gradient;
    SD_scaling_user = SD_scaling*Max_Force*0.2;

  } /* if (MD_Opt_OK!=1 && iter!=MD_IterNumber) */


  /*********************** end of "myid==Host_ID" **************************/

 Last_Bcast: 

  MPI_Bcast(&MD_Opt_OK,1,MPI_INT, Host_ID, mpi_comm_level1);

  OpenMX_Geometry_Publish(geometry,Gxyz,tv);

  /* print information */ 

  if (myid==Host_ID){ 

    if (0<level_stdout){ 

      printf("<%s>  |Maximum force| (Hartree/Bohr) =%15.12f\n",
	     func_name,Max_Force);fflush(stdout);
      printf("<%s>  Criterion       (Hartree/Bohr) =%15.12f\n",
	     func_name,MD_Opt_criterion);fflush(stdout);

      printf("\n");
      printf(" Cell vectors and derivatives of total energy with respect to them\n");

      printf(" a1(Ang.) =%10.5f %10.5f %10.5f   dE/da1(a.u.) =%10.5f %10.5f %10.5f\n",
             tv[1][1]*BohrR,tv[1][2]*BohrR,tv[1][3]*BohrR,dE_da[1][1],dE_da[1][2],dE_da[1][3]);
      printf(" a2(Ang.) =%10.5f %10.5f %10.5f   dE/da2(a.u.) =%10.5f %10.5f %10.5f\n",
             tv[2][1]*BohrR,tv[2][2]*BohrR,tv[2][3]*BohrR,dE_da[2][1],dE_da[2][2],dE_da[2][3]);
      printf(" a3(Ang.) =%10.5f %10.5f %10.5f   dE/da3(a.u.) =%10.5f %10.5f %10.5f\n",
             tv[3][1]*BohrR,tv[3][2]*BohrR,tv[3][3]*BohrR,dE_da[3][1],dE_da[3][2],dE_da[3][3]);

      if (1<level_stdout){ 

	printf("\n");
	for (i=1; i<=atomnum; i++){
	  printf("     atom=%4d, XYZ(Ang) Fxyz(a.u.)=%9.4f %9.4f %9.4f  %9.4f %9.4f %9.4f\n",
		 i,BohrR*Gxyz[i][1],BohrR*Gxyz[i][2],BohrR*Gxyz[i][3],
		 Gxyz[i][17],Gxyz[i][18],Gxyz[i][19] ); fflush(stdout);
	}   
      }
    }

    strcpy(fileSD,".SD");
    fnjoint(filepath,filename,fileSD);

    fp_SD = fopen(fileSD, "r");
    if (fp_SD==NULL){
      header_flag = 1;
    }
    else{
      header_flag = 0;
      fclose(fp_SD);
    }  

    if ((fp_SD = fopen(fileSD,"a")) != NULL){

      setvbuf(fp_SD,buf,_IOFBF,fp_bsize);  /* setvbuf */

      if (header_flag==1){

        fprintf(fp_SD,"\n");
        fprintf(fp_SD,"***********************************************************\n");
        fprintf(fp_SD,"***********************************************************\n");
        fprintf(fp_SD,"                History of cell optimization               \n");
        fprintf(fp_SD,"***********************************************************\n");
        fprintf(fp_SD,"***********************************************************\n\n");


        fprintf(fp_SD,"  MD_iter   SD_scaling     |Maximum force|   Maximum step        Utot             Enpy           Volume\n");
        fprintf(fp_SD,"                           (Hartree/Bohr)        (Ang)         (Hartree)        (Hartree)        (Ang^3)\n\n");
      }

      fprintf(fp_SD,"  %3d  %15.8f  %15.8f  %15.8f  %15.8f  %15.8f  %15.8f\n",
              iter0, SD_scaling, Max_Force, Max_Step*BohrR, thermo->total, thermo->total+thermo->pressure_volume,Cell_Volume*0.14818474347690476628);
      fclose(fp_SD);
    }
    else{
      printf("Could not open a file in MD_pac.!\n");
    }

    if (MD_Opt_OK==1 || iter0==MD_IterNumber){

      strcpy(fileCoord,".crd");
      fnjoint(filepath,filename,fileCoord);
      if ((fp_crd = fopen(fileCoord,"w")) != NULL){

        setvbuf(fp_crd,buf,_IOFBF,fp_bsize);  /* setvbuf */

        fprintf(fp_crd,"\n");
        fprintf(fp_crd,"***********************************************************\n");
        fprintf(fp_crd,"***********************************************************\n");
        fprintf(fp_crd,"    Cell vectors (Ang.) and derivatives of total energy    \n");
        fprintf(fp_crd,"             with respect to them (Hartree/Bohr)           \n");
        fprintf(fp_crd,"***********************************************************\n");
        fprintf(fp_crd,"***********************************************************\n\n");

        fprintf(fp_crd," a1 =%10.5f %10.5f %10.5f   dE/da1 =%10.5f %10.5f %10.5f\n",
                tv[1][1]*BohrR,tv[1][2]*BohrR,tv[1][3]*BohrR,dE_da[1][1],dE_da[1][2],dE_da[1][3]);
        fprintf(fp_crd," a2 =%10.5f %10.5f %10.5f   dE/da2 =%10.5f %10.5f %10.5f\n",
               tv[2][1]*BohrR,tv[2][2]*BohrR,tv[2][3]*BohrR,dE_da[2][1],dE_da[2][2],dE_da[2][3]);
        fprintf(fp_crd," a3 =%10.5f %10.5f %10.5f   dE/da3 =%10.5f %10.5f %10.5f\n",
               tv[3][1]*BohrR,tv[3][2]*BohrR,tv[3][3]*BohrR,dE_da[3][1],dE_da[3][2],dE_da[3][3]);

        fprintf(fp_crd,"\n");
        fprintf(fp_crd,"***********************************************************\n");
        fprintf(fp_crd,"***********************************************************\n");
        fprintf(fp_crd,"       xyz-coordinates (Ang) and forces (Hartree/Bohr)  \n");
        fprintf(fp_crd,"***********************************************************\n");
        fprintf(fp_crd,"***********************************************************\n\n");

        fprintf(fp_crd,"<coordinates.forces\n");
        fprintf(fp_crd,"  %i\n",atomnum);
        for (k=1; k<=atomnum; k++){
          i = WhatSpecies[k];
          j = Spe_WhatAtom[i];
          fprintf(fp_crd," %4d  %4s   %9.5f %9.5f %9.5f  %15.12f %15.12f %15.12f\n",
                  k,Atom_Symbol[j],
	          Gxyz[k][1]*BohrR,Gxyz[k][2]*BohrR,Gxyz[k][3]*BohrR,
	    	  -Gxyz[k][17],-Gxyz[k][18],-Gxyz[k][19]);
        }
        fprintf(fp_crd,"coordinates.forces>\n");
        fclose(fp_crd);
      }
      else
        printf("error(1) in MD_pac.c\n");
    }

  } /* if (myid==Host_ID) */

}










void GDIIS_BFGS(OpenMXGeometryOpt *geometry, const OpenMXChemicalPotentials *chemical, OpenMXDynamicsState *dynamics, OpenMXThermodynamicEnergies *thermo, const OpenMXBandEnergies *energies, const OpenMXEnergyComponents *components, int iter, int iter0)
{
  /* 1au=2.4189*10^-2 fs, 1fs=41.341105 au */

  int m1;
  int header_flag;
  char *func_name="BFGS";
  double dt;
  double *A,*B,sumB;
  double mixing,force_Max;
  static double sMD_TimeStep,dx_max=0.05; 
  double diff_dx,diff,Max_Step;
  INTEGER i,j,k,iatom;
  int diis_iter;
  char fileCoord[YOUSO10];
  char fileSD[YOUSO10];
  FILE *fp_crd,*fp_SD;
  char buf[fp_bsize];          /* setvbuf */
  char fileE[YOUSO10];

  /* variables for MPI */
  int myid;

  /* MPI myid */
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /* share Gxyz */
  OpenMX_Geometry_Share_Atoms(geometry,Gxyz);

  if (iter<M_GDIIS_HISTORY)
    diis_iter = iter;
  else   
    diis_iter = M_GDIIS_HISTORY;

  /* shift one */

  for (i=(diis_iter-2); 0<=i; i--) {
    for (iatom=1; iatom<=atomnum; iatom++) {
      for (k=1; k<=3; k++) {
	GxyzHistoryIn[i+1][iatom][k] = GxyzHistoryIn[i][iatom][k];
	GxyzHistoryR[i+1][iatom][k]  = GxyzHistoryR[i][iatom][k];
      }
    }
  }

  /* Max of force */

  force_Max=0.0;
  for (iatom=1; iatom<=atomnum; iatom++)   {
    for (k=1;k<=3;k++) {
      if (atom_Fixed_XYZ[iatom][k]==0){
	if (force_Max< fabs(Gxyz[iatom][16+k]) ) force_Max = fabs(Gxyz[iatom][16+k]);
      }
    }
  }

  sMD_TimeStep = 0.05/(0.01*41.341105);

  if (2<=level_stdout && myid==Host_ID){
    printf("<%s>  |Maximum force| (Hartree/Bohr) = %15.12f tuned_dt= %f\n",func_name,force_Max, sMD_TimeStep);
    printf("<%s>  Criterion      (Hartree/Bohr) = %15.12f\n", func_name, MD_Opt_criterion);
  }

  /* add GxyzHistoryIn and GxyzHisotryR */

  for (iatom=1; iatom<=atomnum; iatom++)   {

    for (k=1;k<=3;k++) {
      if (atom_Fixed_XYZ[iatom][k]==0){
	GxyzHistoryIn[0][iatom][k] = Gxyz[iatom][k];
	GxyzHistoryR[0][iatom][k]  = Gxyz[iatom][16+k];
      }
      else{
	GxyzHistoryIn[0][iatom][k] = Gxyz[iatom][k];
	GxyzHistoryR[0][iatom][k]  = 0.0;
      }
    }
  }

  if (myid!=Host_ID)  goto Last_Bcast; 

  /*********************** for myid==Host_ID **************************/

  /* allocation of arrays */

  OpenMX_Geometry_Prepare_DIIS(geometry,GxyzHistoryR,atomnum,diis_iter,&A,&B);

  if (2<=level_stdout){
    printf("<%s>  DIIS matrix\n",func_name);
    for (i=0; i<(diis_iter+1); i++) {
      printf("<%s> ",func_name);
      for (j=0; j<(diis_iter+1); j++) {
        printf("%10.5f ",A[i*(diis_iter+1)+j]);
      }
      printf("\n");
    }
  }

  /* lapack routine */

  if (2<=level_stdout){
    printf("M_GDIIS_HISTORY=%2d diis_iter=%2d\n",M_GDIIS_HISTORY,diis_iter);
  }

  const int info=OpenMX_Geometry_Solve_DIIS(geometry,diis_iter,A,B);

  if (info!=0) {
    printf("<%s> dsysv_, info=%d\n",func_name,info);
    printf("<%s> \n",func_name);
    printf("<%s> ERROR, aborting\n",func_name);
    printf("<%s> \n",func_name);

    MD_Opt_OK =1; 
    /* no change */

    goto Last_Bcast ;
  }

  if (2<=level_stdout){
    printf("<%s> diis alpha=",func_name);
    sumB = 0;
    for (i=0; i<diis_iter; i++) {
      printf("%10.5f ",B[i]);
      sumB += B[i];
    }
    printf("%lf\n",B[diis_iter]);
  }

  if (force_Max<MD_Opt_criterion ){

     MD_Opt_OK = 1;

    /* show a message if the geometry optimization was achieved or not. */
    if (myid==Host_ID) printf("\nThe geometry optimization was achieved.\n");
  }

  /****************************************************
   write informatins to *.ene
  ****************************************************/

  if (myid==Host_ID){  

    sprintf(fileE,"%s%s.ene",filepath,filename);
    iterout_md(chemical, dynamics, thermo, energies, components, iter,MD_TimeStep*(iter-1),fileE);
  }

  if (MD_Opt_OK!=1 && iter!=MD_IterNumber){

    /* initialize */

    for (iatom=1; iatom<=atomnum; iatom++) {
      for (j=1; j<=3; j++) {
	Gxyz[iatom][j] = 0.0;
      }
    }

    /* add tilde{R} */

    for (iatom=1; iatom<=atomnum; iatom++) {
      for (j=1; j<=3; j++) {
	if (atom_Fixed_XYZ[iatom][j]==0){
	  for (i=0; i<diis_iter; i++) {
	    Gxyz[iatom][j] += GxyzHistoryR[i][iatom][j]*B[i];
	  }
	}
      }
    }

    sumB = 0.0;
    for (iatom=1; iatom<=atomnum; iatom++) {
      for (j=1; j<=3; j++) {
	sumB += Gxyz[iatom][j]*Gxyz[iatom][j] ;
      }
    }

    sumB = sqrt(sumB)/(double)atomnum;

    /************************************************************
     update an approximate inverse of Hessian matrix 

     y = GxyzHistoryR[0][iatom][k]  - GxyzHistoryR[1][iatom][k]
     s = GxyzHistoryIn[0][iatom][k] - GxyzHistoryIn[1][iatom][k]
    ************************************************************/

    if (iter==1){
      for (i=1; i<=3*atomnum; i++){     
	for (j=1; j<=3*atomnum; j++){     
	  InvHessian[i][j] = 0.0;
	}
	InvHessian[i][i] = 1.0;
      }
    }

    else {
      OpenMX_Geometry_BFGS(geometry,InvHessian,GxyzHistoryIn,GxyzHistoryR,atomnum,
                          OPENMX_GEOMETRY_INVERSE_HESSIAN);
    }

    OpenMX_Geometry_Apply_Inverse(geometry,InvHessian,Gxyz);

    if (2<=level_stdout){
      printf("<%s> |tilde{R}|=%E\n",func_name, sumB);
    }

    /* initialize */

    for (iatom=1; iatom<=atomnum; iatom++) {
      for (j=1; j<=3; j++) {
	Gxyz[iatom][j] = 0.0;
      }
    }

    /* calculate the DIIS coordinates tilde{x} */

    m1 = 0;
    for (iatom=1; iatom<=atomnum; iatom++) {
      for (j=1; j<=3; j++) {

	for (i=0; i<diis_iter; i++) {
	  Gxyz[iatom][j] += GxyzHistoryIn[i][iatom][j]*B[i]; 
	}

	/* a quasi Newton method */ 

	m1++;
	Gxyz[iatom][j] -= InvHessian[0][m1]; 

      }
    }

    /************************************************************
        In case of a too large updating, do a modest updating
    ************************************************************/  

    Max_Step = 0.0;
    for (iatom=1; iatom<=atomnum; iatom++) {
      for (j=1; j<=3; j++) {

        diff = fabs(Gxyz[iatom][j] - GxyzHistoryIn[0][iatom][j]);
        if (Max_Step<diff) Max_Step = diff;
      }
    }

    if (Criterion_Max_Step<Max_Step){

      for (iatom=1; iatom<=atomnum; iatom++) {
	for (j=1; j<=3; j++) {

	  diff = Gxyz[iatom][j] - GxyzHistoryIn[0][iatom][j];
	  Gxyz[iatom][j] = GxyzHistoryIn[0][iatom][j] + diff/Max_Step*Criterion_Max_Step;
	}
      }
    }

    /* find Max_Step */
    Max_Step = 0.0;
    for (iatom=1; iatom<=atomnum; iatom++) {
      for (j=1; j<=3; j++) {

        diff = fabs(Gxyz[iatom][j] - GxyzHistoryIn[0][iatom][j]);
        if (Max_Step<diff) Max_Step = diff;
      }
    }

    /************************************************************
                   show cooridinates and gradients
    ************************************************************/  

    if (2<=level_stdout){

      printf("<%s> diff_x= %f , dE= %f\n",func_name, Max_Step, fabs(thermo->total-Past_Utot[1]) );

      /* print atomic positions */
      printf("<%s> atomnum= %d\n",func_name,atomnum);
      for (i=1; i<=atomnum; i++){
	j = Spe_WhatAtom[WhatSpecies[i]];
	printf("  %3d %s XYZ(Ang) Fxyz(a.u.)= %9.4f %9.4f %9.4f  %9.4f %9.4f %9.4f\n",
	       i,Atom_Symbol[j],
	       BohrR*Gxyz[i][1],BohrR*Gxyz[i][2],BohrR*Gxyz[i][3],
	       Gxyz[i][17],Gxyz[i][18],Gxyz[i][19] ); 
      }   
    }

  } /* if (MD_Opt_OK!=1 && iter!=MD_IterNumber) */

  Past_Utot[1]=thermo->total;

  Max_Force = force_Max;
  SD_scaling_user = SD_scaling*Max_Force*0.2;


  /*********************** end of "myid==Host_ID" **************************/

 Last_Bcast: 

  MPI_Bcast(&MD_Opt_OK,1,MPI_INT, Host_ID, mpi_comm_level1);

  OpenMX_Geometry_Publish(geometry,Gxyz,NULL);

  if (myid==Host_ID){ 

    if (0<level_stdout){ 

      printf("<%s>  |Maximum force| (Hartree/Bohr) =%15.12f\n",
	     func_name,Max_Force);fflush(stdout);
      printf("<%s>  Criterion       (Hartree/Bohr) =%15.12f\n",
	     func_name,MD_Opt_criterion);fflush(stdout);

      printf("\n");
      for (i=1; i<=atomnum; i++){
	printf("     atom=%4d, XYZ(ang) Fxyz(a.u.)=%9.4f %9.4f %9.4f  %9.4f %9.4f %9.4f\n",
	       i,BohrR*Gxyz[i][1],BohrR*Gxyz[i][2],BohrR*Gxyz[i][3],
	       Gxyz[i][17],Gxyz[i][18],Gxyz[i][19] ); fflush(stdout);
      }   
    }

    strcpy(fileSD,".SD");
    fnjoint(filepath,filename,fileSD);

    fp_SD = fopen(fileSD, "r");
    if (fp_SD==NULL){
      header_flag = 1;
    }
    else{
      header_flag = 0;
      fclose(fp_SD);
    }  

    if ((fp_SD = fopen(fileSD,"a")) != NULL){

      setvbuf(fp_SD,buf,_IOFBF,fp_bsize);  /* setvbuf */

      if (header_flag==1){

        fprintf(fp_SD,"\n");
        fprintf(fp_SD,"***********************************************************\n");
        fprintf(fp_SD,"***********************************************************\n");
        fprintf(fp_SD,"              History of geometry optimization             \n");
        fprintf(fp_SD,"***********************************************************\n");
        fprintf(fp_SD,"***********************************************************\n\n");


        fprintf(fp_SD,"  MD_iter   SD_scaling     |Maximum force|   Maximum step        Utot\n");
        fprintf(fp_SD,"                           (Hartree/Bohr)        (Ang)         (Hartree)\n\n");
      }

      fprintf(fp_SD,"  %3d  %15.8f  %15.8f  %15.8f  %15.8f\n",
              iter0,SD_scaling,Max_Force,Max_Step*BohrR,thermo->total);
      fclose(fp_SD);
    }
    else{
      printf("Could not open a file in MD_pac.!\n");
    }

    if (MD_Opt_OK==1 || iter0==MD_IterNumber){

      strcpy(fileCoord,".crd");
      fnjoint(filepath,filename,fileCoord);
      if ((fp_crd = fopen(fileCoord,"w")) != NULL){

        setvbuf(fp_crd,buf,_IOFBF,fp_bsize);  /* setvbuf */

        fprintf(fp_crd,"\n");
        fprintf(fp_crd,"***********************************************************\n");
        fprintf(fp_crd,"***********************************************************\n");
        fprintf(fp_crd,"       xyz-coordinates (Ang) and forces (Hartree/Bohr)  \n");
        fprintf(fp_crd,"***********************************************************\n");
        fprintf(fp_crd,"***********************************************************\n\n");

        fprintf(fp_crd,"<coordinates.forces\n");
        fprintf(fp_crd,"  %i\n",atomnum);
        for (k=1; k<=atomnum; k++){
          i = WhatSpecies[k];
          j = Spe_WhatAtom[i];
          fprintf(fp_crd," %4d  %4s   %9.5f %9.5f %9.5f  %15.12f %15.12f %15.12f\n",
                  k,Atom_Symbol[j],
	          Gxyz[k][1]*BohrR,Gxyz[k][2]*BohrR,Gxyz[k][3]*BohrR,
	    	  -Gxyz[k][17],-Gxyz[k][18],-Gxyz[k][19]);
        }
        fprintf(fp_crd,"coordinates.forces>\n");
        fclose(fp_crd);
      }
      else
        printf("error(1) in MD_pac.c\n");
    }

  } /* if (myid==Host_ID) */

}









void GDIIS_EF(OpenMXGeometryOpt *geometry, const OpenMXChemicalPotentials *chemical, OpenMXDynamicsState *dynamics, OpenMXThermodynamicEnergies *thermo, const OpenMXBandEnergies *energies, const OpenMXEnergyComponents *components, int iter, int iter0)
{
  /* 1au=2.4189*10^-2 fs, 1fs=41.341105 au */

  int m1,wan,wsp;
  int header_flag;
  static double Utot0,scaling_factor;
  char *func_name="EF";
  double dt;
  double itv[4][4];
  double *A,*B,sumB;
  double mixing,force_Max;
  static double sMD_TimeStep,dx_max=0.05; 
  double diff_dx,diff,Max_Step;
  INTEGER i,j,k,iatom;
  int diis_iter;
  char fileCoord[YOUSO10];
  char fileSD[YOUSO10];
  FILE *fp_crd,*fp_SD;
  char buf[fp_bsize];          /* setvbuf */
  char fileE[YOUSO10];
  double tmp_array[10];
  char file_name[YOUSO10];
  FILE *fp;

  /* variables for MPI */
  int myid;

  /* MPI myid */
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /******************************************
              read *.rst4gopt.EF2 
  ******************************************/

  if (SuccessReadingfiles){

    sprintf(file_name,"%s%s_rst/%s.rst4gopt.EF2",filepath,filename,filename);

    if ((fp = fopen(file_name,"rb")) != NULL){

      fread(tmp_array, sizeof(double), 2, fp);

      Utot0          = tmp_array[0];
      scaling_factor = tmp_array[1];
 
      fclose(fp);
    }    
    else{
      printf("Failure of saving %s\n",file_name);
    }
  }

  /* share Gxyz */
  OpenMX_Geometry_Share_Atoms(geometry,Gxyz);

  /* set diis_iter */

  if (iter<M_GDIIS_HISTORY)
    diis_iter = iter;
  else   
    diis_iter = M_GDIIS_HISTORY;

  /* shift one */

  for (i=(diis_iter-2); 0<=i; i--) {
    for (iatom=1; iatom<=atomnum; iatom++) {
      for (k=1; k<=3; k++) {
	GxyzHistoryIn[i+1][iatom][k] = GxyzHistoryIn[i][iatom][k];
	GxyzHistoryR[i+1][iatom][k]  = GxyzHistoryR[i][iatom][k];
      }
    }
  }

  /* add GxyzHistoryIn and GxyzHisotryR */

  for (iatom=1; iatom<=atomnum; iatom++)   {

    for (k=1; k<=3; k++) {
      if (atom_Fixed_XYZ[iatom][k]==0){
	GxyzHistoryIn[0][iatom][k] = Gxyz[iatom][k];
	GxyzHistoryR[0][iatom][k]  = Gxyz[iatom][16+k];
      }
      else{
	GxyzHistoryIn[0][iatom][k] = Gxyz[iatom][k];
	GxyzHistoryR[0][iatom][k]  = 0.0;
      }
    }
  }

  /* set the initial approximate Hessian */

  if (iter==1){
    scaling_factor = 2.0;
    Estimate_Initial_Hessian(geometry, diis_iter,0,itv);
  }
  
  if (myid!=Host_ID)   goto Last_Bcast; 

  /*********************** for myid==Host_ID **************************/

  /* allocation of arrays */

  /* Max of force */

  force_Max=0.0;
  for (iatom=1; iatom<=atomnum; iatom++)   {

    wsp = WhatSpecies[iatom];
    wan = Spe_WhatAtom[wsp];

    for (k=1;k<=3;k++) {

      if (atom_Fixed_XYZ[iatom][k]==0 && wan!=0){

	if (force_Max< fabs(Gxyz[iatom][16+k]) ) force_Max = fabs(Gxyz[iatom][16+k]);
      }

    }
  }

  sMD_TimeStep = 0.05/(0.01*41.341105);

  if (2<=level_stdout){
    printf("<%s>  |Maximum force| (Hartree/Bohr) = %15.12f tuned_dt= %f\n",func_name,force_Max, sMD_TimeStep);
    printf("<%s>  Criterion      (Hartree/Bohr) = %15.12f\n", func_name, MD_Opt_criterion);
  }

  OpenMX_Geometry_Prepare_DIIS(geometry,GxyzHistoryR,atomnum,diis_iter,&A,&B);

  if (2<=level_stdout){
    printf("<%s>  DIIS matrix\n",func_name);
    for (i=0; i<(diis_iter+1); i++) {
      printf("<%s> ",func_name);
      for (j=0; j<(diis_iter+1); j++) {
        printf("%6.3f ",A[i*(diis_iter+1)+j]);
      }
      printf("\n");
    }
  }

  /* lapack routine */

  if (2<=level_stdout){
    printf("M_GDIIS_HISTORY=%2d diis_iter=%2d\n",M_GDIIS_HISTORY,diis_iter);
  }

  const int info=OpenMX_Geometry_Solve_DIIS(geometry,diis_iter,A,B);

  if (info!=0) {
    printf("<%s> dsysv_, info=%d\n",func_name,info);
    printf("<%s> \n",func_name);
    printf("<%s> ERROR, aborting\n",func_name);
    printf("<%s> \n",func_name);

    MD_Opt_OK =1; 
    /* no change */

    goto Last_Bcast ;
  }

  if (2<=level_stdout){
    printf("<%s> diis alpha=",func_name);
    sumB = 0;
    for (i=0; i<diis_iter; i++) {
      printf("%f ",B[i]);
      sumB += B[i];
    }
    printf("%lf\n",B[diis_iter]);
  }

  if (force_Max<MD_Opt_criterion ){

     MD_Opt_OK = 1;

    /* show a message if the geometry optimization was achieved or not. */
    if (myid==Host_ID) printf("\nThe geometry optimization was achieved.\n");
  }
    
  /****************************************************
   write informatins to *.ene
  ****************************************************/
    
  if (myid==Host_ID){  
    
    sprintf(fileE,"%s%s.ene",filepath,filename);
    iterout_md(chemical, dynamics, thermo, energies, components, iter,MD_TimeStep*(iter-1),fileE);
  } 

  if (MD_Opt_OK!=1 && iter!=MD_IterNumber){

    /* initialize */

    for (iatom=1; iatom<=atomnum; iatom++) {
      for (j=1; j<=3; j++) {
	Gxyz[iatom][j] = 0.0;
      }
    }

    /* add tilde{R} */

    for (iatom=1; iatom<=atomnum; iatom++) {
      for (j=1; j<=3; j++) {
	if (atom_Fixed_XYZ[iatom][j]==0){
	  for (i=0; i<diis_iter; i++) {
	    Gxyz[iatom][j] += GxyzHistoryR[i][iatom][j]*B[i];
	  }
	}
      }
    }

    /* store tilde{R} into Hessian[][0] */

    m1 = 0;
    for (iatom=1; iatom<=atomnum; iatom++) {
      for (j=1; j<=3; j++) {
	m1++;
	Hessian[m1][0] = Gxyz[iatom][j];
      }
    }

    sumB = 0.0;
    for (iatom=1; iatom<=atomnum; iatom++) {
      for (j=1; j<=3; j++) {
	sumB += Gxyz[iatom][j]*Gxyz[iatom][j] ;
      }
    }

    sumB = sqrt(sumB)/(double)atomnum;

    if (2<=level_stdout){
      printf("<%s> |tilde{R}|=%E\n",func_name, sumB);
    }

    /***********************************************************************************
      update an approximate Hessian matrix

      H_k = H_{k-1} + y*y^t/(s^t*y) - H_{k-1}*s * s^t * H_{k-1}^t /(s^t * H_{k-1} * s)
      y = GxyzHistoryR[0][iatom][k]  - GxyzHistoryR[1][iatom][k]
      s = GxyzHistoryIn[0][iatom][k] - GxyzHistoryIn[1][iatom][k]
    ***********************************************************************************/

    if (iter!=1)
      OpenMX_Geometry_BFGS(geometry,Hessian,GxyzHistoryIn,GxyzHistoryR,atomnum,OPENMX_GEOMETRY_EF_HESSIAN);

    const double min_eigenvalue=atomnum<=4 ? 0.10 : 0.005;
    OpenMX_Geometry_EF_Step(geometry,Hessian,min_eigenvalue,scf_eigen_lib_flag==GPUSOLVER,level_stdout);

    /************************************************************
            calculate the DIIS coordinates tilde{x}
              and perform a quasi Newton method
    ************************************************************/

    /* update scaling_factor */

    if (Utot0<thermo->total && iter!=1) scaling_factor = 0.95*scaling_factor;

    /* initialize */

    for (iatom=1; iatom<=atomnum; iatom++) {
      for (j=1;j<=3;j++) {
	Gxyz[iatom][j] = 0.0;
      }
    }

    m1 = 0;
    for (iatom=1; iatom<=atomnum; iatom++) {
      for (j=1; j<=3; j++) {

	/* DIIS coordinate */

	for (i=0; i<diis_iter; i++) {
	  Gxyz[iatom][j] += GxyzHistoryIn[i][iatom][j]*B[i]; 
	}


	/* with a quasi Newton method */ 

	m1++;

	Gxyz[iatom][j] -= scaling_factor*Hessian[0][m1];
      }
    }

    /************************************************************
        In case of a too large updating, do a modest updating
    ************************************************************/  

    Max_Step = 0.0;
    for (iatom=1; iatom<=atomnum; iatom++) {
      for (j=1; j<=3; j++) {

        diff = fabs(Gxyz[iatom][j] - GxyzHistoryIn[0][iatom][j]);
        if (Max_Step<diff) Max_Step = diff;
      }
    }

    if (Criterion_Max_Step<Max_Step){

      for (iatom=1; iatom<=atomnum; iatom++) {
	for (j=1; j<=3; j++) {

	  diff = Gxyz[iatom][j] - GxyzHistoryIn[0][iatom][j];
	  Gxyz[iatom][j] = GxyzHistoryIn[0][iatom][j] + diff/Max_Step*Criterion_Max_Step;
	}
      }
    }

    /* find Max_Step */
    Max_Step = 0.0;
    for (iatom=1; iatom<=atomnum; iatom++) {
      for (j=1; j<=3; j++) {

        diff = fabs(Gxyz[iatom][j] - GxyzHistoryIn[0][iatom][j]);
        if (Max_Step<diff) Max_Step = diff;
      }
    }

    /************************************************************
                    show coordinates and gradients
    ************************************************************/  

    if (2<=level_stdout){

      printf("<%s> diff_x= %f , dE= %f\n",func_name, Max_Step, fabs(thermo->total-Past_Utot[1]) );

      /* print atomic positions */
      printf("<%s> atomnum= %d\n",func_name,atomnum);
      for (i=1; i<=atomnum; i++){
	j = Spe_WhatAtom[WhatSpecies[i]];
	printf("  %3d %s XYZ(ang) Fxyz(a.u.)= %9.4f %9.4f %9.4f  %9.4f %9.4f %9.4f\n",
	       i,Atom_Symbol[j],
	       BohrR*Gxyz[i][1],BohrR*Gxyz[i][2],BohrR*Gxyz[i][3],
	       Gxyz[i][17],Gxyz[i][18],Gxyz[i][19] ); 
      }   
    }

  } /* if (MD_Opt_OK!=1 && iter!=MD_IterNumber) */

  Past_Utot[1]=thermo->total;

  Max_Force = force_Max;
  SD_scaling_user = SD_scaling*Max_Force*0.2;

  /* save Utot */

  Utot0 = thermo->total;



  /*********************** end of "myid==Host_ID" **************************/

 Last_Bcast: 

  MPI_Bcast(&MD_Opt_OK,1,MPI_INT, Host_ID, mpi_comm_level1);

  OpenMX_Geometry_Publish(geometry,Gxyz,NULL);


  if (myid==Host_ID){ 

    if (0<level_stdout){ 

      printf("<%s>  |Maximum force| (Hartree/Bohr) =%15.12f\n",
	     func_name,Max_Force);fflush(stdout);
      printf("<%s>  Criterion       (Hartree/Bohr) =%15.12f\n",
	     func_name,MD_Opt_criterion);fflush(stdout);

      printf("\n");
      for (i=1; i<=atomnum; i++){
	printf("     atom=%4d, XYZ(ang) Fxyz(a.u.)=%9.4f %9.4f %9.4f  %9.4f %9.4f %9.4f\n",
	       i,BohrR*Gxyz[i][1],BohrR*Gxyz[i][2],BohrR*Gxyz[i][3],
	       Gxyz[i][17],Gxyz[i][18],Gxyz[i][19] ); fflush(stdout);
      }   
    }

    strcpy(fileSD,".SD");
    fnjoint(filepath,filename,fileSD);

    fp_SD = fopen(fileSD, "r");
    if (fp_SD==NULL){
      header_flag = 1;
    }
    else{
      header_flag = 0;
      fclose(fp_SD);
    }  

    if ((fp_SD = fopen(fileSD,"a")) != NULL){

      setvbuf(fp_SD,buf,_IOFBF,fp_bsize);  /* setvbuf */

      if (header_flag==1){

        fprintf(fp_SD,"\n");
        fprintf(fp_SD,"***********************************************************\n");
        fprintf(fp_SD,"***********************************************************\n");
        fprintf(fp_SD,"              History of geometry optimization             \n");
        fprintf(fp_SD,"***********************************************************\n");
        fprintf(fp_SD,"***********************************************************\n\n");


        fprintf(fp_SD,"  MD_iter   SD_scaling     |Maximum force|   Maximum step        Utot\n");
        fprintf(fp_SD,"                           (Hartree/Bohr)        (Ang)         (Hartree)\n\n");
      }

      fprintf(fp_SD,"  %3d  %15.8f  %15.8f  %15.8f  %15.8f\n",
              iter0,SD_scaling,Max_Force,Max_Step*BohrR,thermo->total);
      fclose(fp_SD);
    }
    else{
      printf("Could not open a file in MD_pac.!\n");
    }

    if (MD_Opt_OK==1 || iter0==MD_IterNumber){

      strcpy(fileCoord,".crd");
      fnjoint(filepath,filename,fileCoord);
      if ((fp_crd = fopen(fileCoord,"w")) != NULL){

        setvbuf(fp_crd,buf,_IOFBF,fp_bsize);  /* setvbuf */

        fprintf(fp_crd,"\n");
        fprintf(fp_crd,"***********************************************************\n");
        fprintf(fp_crd,"***********************************************************\n");
        fprintf(fp_crd,"       xyz-coordinates (Ang) and forces (Hartree/Bohr)  \n");
        fprintf(fp_crd,"***********************************************************\n");
        fprintf(fp_crd,"***********************************************************\n\n");

        fprintf(fp_crd,"<coordinates.forces\n");
        fprintf(fp_crd,"  %i\n",atomnum);
        for (k=1; k<=atomnum; k++){
          i = WhatSpecies[k];
          j = Spe_WhatAtom[i];
          fprintf(fp_crd," %4d  %4s   %9.5f %9.5f %9.5f  %15.12f %15.12f %15.12f\n",
                  k,Atom_Symbol[j],
	          Gxyz[k][1]*BohrR,Gxyz[k][2]*BohrR,Gxyz[k][3]*BohrR,
	    	  -Gxyz[k][17],-Gxyz[k][18],-Gxyz[k][19]);
        }
        fprintf(fp_crd,"coordinates.forces>\n");
        fclose(fp_crd);
      }
      else
        printf("error(1) in MD_pac.c\n");
    }

  } /* if (myid==Host_ID) */

  /******************************************
              save *.rst4gopt.EF2 
  ******************************************/

  if (myid==Host_ID){

    sprintf(file_name,"%s%s_rst/%s.rst4gopt.EF2",filepath,filename,filename);

    if ((fp = fopen(file_name,"wb")) != NULL){

      tmp_array[0] = Utot0;
      tmp_array[1] = scaling_factor;
 
      fwrite(tmp_array, sizeof(double), 2, fp);
      fclose(fp);
    }    
    else{
      printf("Failure of saving %s\n",file_name);
    }
  }

}



















void NVT_VS(OpenMXMD *md, const OpenMXChemicalPotentials *chemical, OpenMXDynamicsState *dynamics, OpenMXThermodynamicEnergies *thermo, const OpenMXBandEnergies *energies, const OpenMXEnergyComponents *components, int iter)
{
  /* added by mari */
  /********************************************************
   This routine is added by Mari Ohfuti (May 20004).                

   a constant temperature molecular dynamics by a velocity
   scaling method with velocity-Verlet integrator
  ********************************************************/
  /******************************************************* 
   1 a.u.=2.4189*10^-2 fs, 1fs=41.341105 a.u. 
  ********************************************************/

  /****************************************************
    Gxyz[][1] = x-coordinate at current step
    Gxyz[][2] = y-coordinate at current step
    Gxyz[][3] = z-coordinate at current step

    Gxyz[][14] = dEtot/dx at previous step
    Gxyz[][15] = dEtot/dy at previous step
    Gxyz[][16] = dEtot/dz at previous step

    Gxyz[][17] = dEtot/dx at current step
    Gxyz[][18] = dEtot/dy at current step
    Gxyz[][19] = dEtot/dz at current step

    Gxyz[][20] = atomic mass

    Gxyz[][21] = x-coordinate at previous step
    Gxyz[][22] = y-coordinate at previous step
    Gxyz[][23] = z-coordinate at previous step

    Gxyz[][24] = x-component of velocity at current step
    Gxyz[][25] = y-component of velocity at current step
    Gxyz[][26] = z-component of velocity at current step

    Gxyz[][27] = x-component of velocity at t+dt/2
    Gxyz[][28] = y-component of velocity at t+dt/2
    Gxyz[][29] = z-component of velocity at t+dt/2

    Gxyz[][30] = hx
    Gxyz[][31] = hy
    Gxyz[][32] = hz

  ****************************************************/

  double dt,dt2,x;
  double Wscale;
  int i;
  int myid;
  char fileE[YOUSO10];

  /* MPI */
  MPI_Comm_rank(mpi_comm_level1,&myid);

  MD_Opt_OK = 0;
  dt = 41.3411*MD_TimeStep;
  dt2 = dt*dt;
  Wscale = unified_atomic_mass_unit/electron_mass;

  /****************************************************
                    update velocity 
  ****************************************************/

  const OpenMXMDStep kick={iter==1 ? OPENMX_MD_SCALE_START : OPENMX_MD_SCALE_KICK,dt,dt2,0.0};
  OpenMX_MD_Move(md,kick,Wscale,1.0,NULL,NULL);

  /****************************************************
   correct so that the sum of velocities can be zero.                              
  ****************************************************/

  OpenMX_MD_Center(md,24);

  /****************************************************
                     Kinetic Energy 
  ****************************************************/

  thermo->ionic_kinetic=OpenMX_MD_Kinetic(md,Wscale,0);
  
  /* calculation of temperature (K) */
  dynamics->temperature = thermo->ionic_kinetic/(1.5*kB*(double)atomnum)*eV2Hartree;
  
  /* calculation of a given temperature (K) */
  for (i=1; i<=TempNum; i++) {
    if( (iter>NumScale[i-1]) && (iter<=NumScale[i]) ) {
      dynamics->target_temperature = TempPara[i-1][2] + (TempPara[i][2] - TempPara[i-1][2])*
          ((double)iter-(double)TempPara[i-1][1])/((double)TempPara[i][1]-(double)TempPara[i-1][1]);
    }
  }
  
  /****************************************************
   write informatins to *.ene
  ****************************************************/
  
  if (myid==Host_ID){  
  
    sprintf(fileE,"%s%s.ene",filepath,filename);

    /* corrected on 2016/1/19 by ADO */
    iterout_md(chemical, dynamics, thermo, energies, components, iter,MD_TimeStep*(iter-1),fileE);
  }
  
  /****************************************************
                    velocity scaling 
  ****************************************************/
  
  if(iter!=1) {

    x = 1.0;
    for (i=1; i<=TempNum; i++) {

      if( (iter>NumScale[i-1]) && (iter<=NumScale[i]) ) {

        /**************************************************
         find a scaling parameter, x, when MD step matches
         at the step where the temperature scaling is made.
         Otherwise, x = 1.0.
        **************************************************/

        if((iter-NumScale[i-1])%IntScale[i]==0) {

          dynamics->target_temperature = TempPara[i-1][2] + (TempPara[i][2] - TempPara[i-1][2])*
               ((double)iter-(double)TempPara[i-1][1])/((double)TempPara[i][1]-(double)TempPara[i-1][1]);
 
          x = dynamics->target_temperature + (dynamics->temperature-dynamics->target_temperature)*RatScale[i];
          x = sqrt(1.5*kB*x/(thermo->ionic_kinetic*eV2Hartree)*(double)atomnum);
        }
      }
    }

    /* do scaling */

    if (MD_Opt_OK!=1 && iter!=MD_IterNumber){

      const OpenMXMDStep drift={OPENMX_MD_SCALE_DRIFT,dt,dt2,0.0};
      OpenMX_MD_Move(md,drift,Wscale,x,NULL,NULL);

    }

  }
}




void NVT_Langevin(OpenMXMD *md, OpenMXDynamicsState *dynamics, OpenMXThermodynamicEnergies *thermo, int iter)
{
  /* added by T.Ohwaki */
  /********************************************************
   This routine is added by Tsuruku Ohwaki (May 2012).

   a constant temperature molecular dynamics by Langevin
   heat-bath method with velocity-Verlet integrator
  ********************************************************/
  /*******************************************************
   1 a.u.=2.4189*10^-2 fs, 1fs=41.341105 a.u.
  ********************************************************/

  /****************************************************
    Gxyz[][1] = x-coordinate at current step
    Gxyz[][2] = y-coordinate at current step
    Gxyz[][3] = z-coordinate at current step

    Gxyz[][14] = dEtot/dx at previous step
    Gxyz[][15] = dEtot/dy at previous step
    Gxyz[][16] = dEtot/dz at previous step

    Gxyz[][17] = dEtot/dx at current step
    Gxyz[][18] = dEtot/dy at current step
    Gxyz[][19] = dEtot/dz at current step

    Gxyz[][20] = atomic mass

    Gxyz[][21] = x-coordinate at previous step
    Gxyz[][22] = y-coordinate at previous step
    Gxyz[][23] = z-coordinate at previous step

    Gxyz[][24] = x-component of velocity at t
    Gxyz[][25] = y-component of velocity at t
    Gxyz[][26] = z-component of velocity at t

    Gxyz[][27] = x-component of tilde velocity at t+dt
    Gxyz[][28] = y-component of tilde velocity at t+dt
    Gxyz[][29] = z-component of tilde velocity at t+dt

    Gxyz[][30] = hx
    Gxyz[][31] = hy
    Gxyz[][32] = hz

  ****************************************************/

  double dt,dt2,Wscale;
  int Mc_AN,Gc_AN,i,j;

  MD_Opt_OK = 0;
  dt = 41.3411*MD_TimeStep;
  dt2 = dt*dt;
  Wscale = unified_atomic_mass_unit/electron_mass;

  /* calculation of a given temperature (K) */

  i = 1;
  do {
    if ( TempPara[i][1]<=iter && iter<TempPara[i+1][1] ){

      dynamics->target_temperature = TempPara[i][2] + (TempPara[i+1][2] - TempPara[i][2])*
                 ((double)iter - (double)TempPara[i][1])
                /((double)TempPara[i+1][1] - (double)TempPara[i][1]);

    }
    i++;
  } while (i<=(TempNum-1));

  /****************************************************
   add random forces on Gxyz[17-19]
  ****************************************************/

  /* Keep the native random draw order on each owning rank. */
  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

    Gc_AN = M2G[Mc_AN];
    const double Lang_sig = sqrt(2.0*Gxyz[Gc_AN][20]*Wscale*FricFac*kB*dynamics->target_temperature/dt/eV2Hartree);

    for (j=1; j<=3; j++){

      if (atom_Fixed_XYZ[Gc_AN][j]==0){

	const double rtmp1 = (double)rand()/RAND_MAX;
	const double rtmp2 = (double)rand()/RAND_MAX;

	const double RandomF = Lang_sig * sqrt(-2.0*log(rtmp1)) * cos(2.0*PI*rtmp2);
	Gxyz[Gc_AN][16+j] -= RandomF;

      }
    }
  }

  const OpenMXMDStep move={iter==1 ? OPENMX_MD_LANGEVIN_START : OPENMX_MD_LANGEVIN_DRIFT,
                           dt,dt2,FricFac};
  OpenMX_MD_Move(md,move,Wscale,1.0,NULL,NULL);
  thermo->ionic_kinetic=OpenMX_MD_Kinetic(md,Wscale,1);

  /* calculation of temperature (K) */
  dynamics->temperature = thermo->ionic_kinetic/(1.5*kB*(double)atomnum)*eV2Hartree;

}




void NVT_VS2(OpenMXMD *md, const OpenMXChemicalPotentials *chemical, OpenMXDynamicsState *dynamics, OpenMXThermodynamicEnergies *thermo, const OpenMXBandEnergies *energies, const OpenMXEnergyComponents *components, int iter)
{
  /* added by T.Ohwaki */
  /********************************************************
   This routine is added by T.Ohwaki (May 20004).                

   a constant temperature molecular dynamics by a velocity
   scaling method for each atom with velocity-Verlet integrator
  ********************************************************/
  /******************************************************* 
   1 a.u.=2.4189*10^-2 fs, 1fs=41.341105 a.u. 
  ********************************************************/

  /****************************************************
    Gxyz[][1] = x-coordinate at current step
    Gxyz[][2] = y-coordinate at current step
    Gxyz[][3] = z-coordinate at current step

    Gxyz[][14] = dEtot/dx at previous step
    Gxyz[][15] = dEtot/dy at previous step
    Gxyz[][16] = dEtot/dz at previous step

    Gxyz[][17] = dEtot/dx at current step
    Gxyz[][18] = dEtot/dy at current step
    Gxyz[][19] = dEtot/dz at current step

    Gxyz[][20] = atomic mass

    Gxyz[][21] = x-coordinate at previous step
    Gxyz[][22] = y-coordinate at previous step
    Gxyz[][23] = z-coordinate at previous step

    Gxyz[][24] = x-component of velocity at current step
    Gxyz[][25] = y-component of velocity at current step
    Gxyz[][26] = z-component of velocity at current step
  ****************************************************/

  double dt,dt2,sum;
  double Wscale;

  double *Atomic_Temp,*Atomic_Ukc,*Atomic_Scale;

  int Mc_AN,Gc_AN,i,j;
  int myid;

  char fileE[YOUSO10];

  /* MPI */
  MPI_Comm_rank(mpi_comm_level1,&myid);

  MD_Opt_OK = 0;
  dt = 41.3411*MD_TimeStep;
  dt2 = dt*dt;
  Wscale = unified_atomic_mass_unit/electron_mass;

  Atomic_Temp  = (double*)malloc(sizeof(double)*(atomnum+1));
  Atomic_Ukc   = (double*)malloc(sizeof(double)*(atomnum+1));
  Atomic_Scale = (double*)malloc(sizeof(double)*(atomnum+1));

  /****************************************************
                Velocity Verlet algorithm
  ****************************************************/

  const OpenMXMDStep kick={iter==1 ? OPENMX_MD_SCALE_START : OPENMX_MD_SCALE_KICK,dt,dt2,0.0};
  OpenMX_MD_Move(md,kick,Wscale,1.0,NULL,NULL);

  /****************************************************
      Kinetic Energy & Temperature (for each atom)
  ****************************************************/

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];

    sum = 0.0;
    for (j=1; j<=3; j++){
      sum = sum + Gxyz[Gc_AN][j+23]*Gxyz[Gc_AN][j+23];
    }

    Atomic_Ukc[Gc_AN]  = 0.5*Gxyz[Gc_AN][20]*Wscale*sum;
    Atomic_Temp[Gc_AN] = Atomic_Ukc[Gc_AN]/(1.5*kB)*eV2Hartree;
  }

  /****************************************************
   correct so that the sum of velocities can be zero.                              
  ****************************************************/

  OpenMX_MD_Center(md,24);

  /****************************************************
            Kinetic Energy (for whole system) 
  ****************************************************/

  thermo->ionic_kinetic=OpenMX_MD_Kinetic(md,Wscale,0);

  /* calculation of temperature (K) */
  dynamics->temperature = thermo->ionic_kinetic/(1.5*kB*(double)atomnum)*eV2Hartree;

  /****************************************************
   write informatins to *.ene
  ****************************************************/

  if (myid==Host_ID){  

    sprintf(fileE,"%s%s.ene",filepath,filename);
    iterout_md(chemical, dynamics, thermo, energies, components, iter,MD_TimeStep*(iter-1),fileE);
  }

  /****************************************************
   velocity scaling for each atom
  ****************************************************/

  if(iter!=1) {

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];
      Atomic_Scale[Gc_AN] = 1.0;
    }

    for (i=1; i<=TempNum; i++) {

      if( (iter>NumScale[i-1]) && (iter<=NumScale[i]) ) {

        /*************************************************************
         find a scaling parameter, Atomic_Scale, when MD step matches
         at the step where the temperature scaling is made.
         Otherwise, Atomic_Scale = 1.0.
        *************************************************************/

        if((iter-NumScale[i-1])%IntScale[i]==0) {

          dynamics->target_temperature = TempPara[i-1][2] + (TempPara[i][2] - TempPara[i-1][2])*
               ((double)iter-(double)TempPara[i-1][1])/((double)TempPara[i][1]-(double)TempPara[i-1][1]);
 
          for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
            Gc_AN = M2G[Mc_AN];

            Atomic_Scale[Gc_AN] = sqrt((dynamics->target_temperature + (Atomic_Temp[Gc_AN]-dynamics->target_temperature)*RatScale[i])/Atomic_Temp[Gc_AN]);

          }

        }
      }
    }

    /* do scaling */

    if (MD_Opt_OK!=1 && iter!=MD_IterNumber){

      const OpenMXMDStep drift={OPENMX_MD_SCALE_DRIFT,dt,dt2,0.0};
      OpenMX_MD_Move(md,drift,Wscale,1.0,Atomic_Scale,NULL);

    }

  }

  free(Atomic_Temp);
  free(Atomic_Ukc);
  free(Atomic_Scale);
}


void NVT_VS4(OpenMXMD *md, const OpenMXChemicalPotentials *chemical, OpenMXDynamicsState *dynamics, OpenMXThermodynamicEnergies *thermo, const OpenMXBandEnergies *energies, const OpenMXEnergyComponents *components, int iter)
{
  /* added by T. Ohwaki */
  /********************************************************
   This routine is added by T. Ohwaki (2011/11/11).

   a constant temperature molecular dynamics
   by a velocity scaling method for each atom group
   with velocity-Verlet integrator
  ********************************************************/
  /*******************************************************
   1 a.u.=2.4189*10^-2 fs, 1fs=41.341105 a.u.
  ********************************************************/

  /****************************************************
    Gxyz[][1] = x-coordinate at current step
    Gxyz[][2] = y-coordinate at current step
    Gxyz[][3] = z-coordinate at current step

    Gxyz[][14] = dEtot/dx at previous step
    Gxyz[][15] = dEtot/dy at previous step
    Gxyz[][16] = dEtot/dz at previous step

    Gxyz[][17] = dEtot/dx at current step
    Gxyz[][18] = dEtot/dy at current step
    Gxyz[][19] = dEtot/dz at current step

    Gxyz[][20] = atomic mass

    Gxyz[][21] = x-coordinate at previous step
    Gxyz[][22] = y-coordinate at previous step
    Gxyz[][23] = z-coordinate at previous step

    Gxyz[][24] = x-component of velocity at current step
    Gxyz[][25] = y-component of velocity at current step
    Gxyz[][26] = z-component of velocity at current step
  ****************************************************/

  double dt,dt2;
  double Wscale;

  int Mc_AN,Gc_AN,i,j,k;
  int myid;

  char fileE[YOUSO10];

  /* MPI */
  MPI_Comm_rank(mpi_comm_level1,&myid);

  MD_Opt_OK = 0;
  dt = 41.3411*MD_TimeStep;
  dt2 = dt*dt;
  Wscale = unified_atomic_mass_unit/electron_mass;

  double sum_AtGr[num_AtGr+1],Ukc_AtGr[num_AtGr+1];
  double AtomGr_Scale[num_AtGr+1];

  /****************************************************
                Velocity Verlet algorithm
  ****************************************************/

  const OpenMXMDStep kick={iter==1 ? OPENMX_MD_SCALE_START : OPENMX_MD_SCALE_KICK,dt,dt2,0.0};
  OpenMX_MD_Move(md,kick,Wscale,1.0,NULL,NULL);

  /****************************************************
   correct so that the sum of velocities can be zero.                              
  ****************************************************/

  OpenMX_MD_Center(md,24);

  /******************************************************
      Kinetic Energy & Temperature (for each atom group)
  ******************************************************/

  for (k=1; k<=num_AtGr; k++){ 
    sum_AtGr[k] = 0.0; 
  }

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];
    k = AtomGr[Gc_AN];

    for (j=1; j<=3; j++){
      sum_AtGr[k] += Gxyz[Gc_AN][j+23]*Gxyz[Gc_AN][j+23]*Gxyz[Gc_AN][20];
    }
  }

  for (k=1; k<=num_AtGr; k++){
    Ukc_AtGr[k] = 0.5*Wscale*sum_AtGr[k];
  }

  /****************************************************
            Kinetic Energy (for whole system)
  ****************************************************/

  thermo->ionic_kinetic=OpenMX_MD_Kinetic(md,Wscale,0);

  /* calculation of temperature (K) */
  dynamics->temperature = thermo->ionic_kinetic/(1.5*kB*(double)atomnum)*eV2Hartree;

  /****************************************************
   MPI: Ukc_AtGr
  ****************************************************/

  MPI_Allreduce(MPI_IN_PLACE,Ukc_AtGr+1,num_AtGr,MPI_DOUBLE,MPI_SUM,mpi_comm_level1);
  for (k=1; k<=num_AtGr; k++){

    /* calculation of temperature (K) */
    Temp_AtGr[k] = Ukc_AtGr[k]/(1.5*kB*(double)atnum_AtGr[k])*eV2Hartree;
  }

  /****************************************************
   write informatins to *.ene
  ****************************************************/

  if (myid==Host_ID){
    sprintf(fileE,"%s%s.ene",filepath,filename);
    iterout_md(chemical, dynamics, thermo, energies, components, iter,MD_TimeStep*(iter-1),fileE);
  }

  /****************************************************
   velocity scaling for each atom group
  ****************************************************/

  if(iter!=1) {

  for (k=1; k<=num_AtGr; k++){
    AtomGr_Scale[k] = 1.0;
  }

    for (i=1; i<=TempNum; i++) {

      if( (iter>NumScale[i-1]) && (iter<=NumScale[i]) ) {

        /*************************************************************
         find a scaling parameter, Atomic_Scale, when MD step matches
         at the step where the temperature scaling is made.
         Otherwise, Atomic_Scale = 1.0.
        *************************************************************/

        if((iter-NumScale[i-1])%IntScale[i]==0) {

          dynamics->target_temperature = TempPara[i-1][2] + (TempPara[i][2] - TempPara[i-1][2])*
                     ((double)iter - (double)TempPara[i-1][1])
                    /((double)TempPara[i][1] - (double)TempPara[i-1][1]);

          for (k=1; k<=num_AtGr; k++){

            AtomGr_Scale[k] = sqrt((dynamics->target_temperature + (Temp_AtGr[k]-dynamics->target_temperature)*RatScale[i])/Temp_AtGr[k]);

          }

        }
      }
    }

    /* do scaling */

    if (MD_Opt_OK!=1 && iter!=MD_IterNumber){

      const OpenMXMDStep drift={OPENMX_MD_SCALE_DRIFT,dt,dt2,0.0};
      OpenMX_MD_Move(md,drift,Wscale,1.0,AtomGr_Scale,AtomGr);

    }

  }
}




void NVT_NH(OpenMXMD *md, const OpenMXChemicalPotentials *chemical, OpenMXDynamicsState *dynamics, OpenMXThermodynamicEnergies *thermo, const OpenMXBandEnergies *energies, const OpenMXEnergyComponents *components, int iter)
{
  /***********************************************************
   a constant temperature molecular dynamics by a Nose-Hoover
   method with velocity-Verlet integrator
  ***********************************************************/
  /*********************************************************** 
   1 a.u.=2.4189*10^-2 fs, 1fs=41.341105 a.u. 
  ***********************************************************/
  /****************************************************
    Gxyz[][1] = x-coordinate at current step
    Gxyz[][2] = y-coordinate at current step
    Gxyz[][3] = z-coordinate at current step

    Gxyz[][14] = dEtot/dx at previous step
    Gxyz[][15] = dEtot/dy at previous step
    Gxyz[][16] = dEtot/dz at previous step

    Gxyz[][17] = dEtot/dx at current step
    Gxyz[][18] = dEtot/dy at current step
    Gxyz[][19] = dEtot/dz at current step

    Gxyz[][20] = atomic mass

    Gxyz[][21] = x-coordinate at previous step
    Gxyz[][22] = y-coordinate at previous step
    Gxyz[][23] = z-coordinate at previous step

    Gxyz[][24] = x-component of velocity at current step
    Gxyz[][25] = y-component of velocity at current step
    Gxyz[][26] = z-component of velocity at current step

    Gxyz[][27] = x-component of velocity at t+dt/2
    Gxyz[][28] = y-component of velocity at t+dt/2
    Gxyz[][29] = z-component of velocity at t+dt/2

    Gxyz[][30] = hx
    Gxyz[][31] = hy
    Gxyz[][32] = hz

  ****************************************************/

  int i,myid,NH_switch;
  double dt,Wscale;
  char fileE[YOUSO10];

  /* MPI */
  MPI_Comm_rank(mpi_comm_level1,&myid);

  MD_Opt_OK = 0;
  dt = 41.3411*MD_TimeStep;
  Wscale = unified_atomic_mass_unit/electron_mass;

  /* find a given temperature by a linear interpolation */

  NH_switch = 0;
  i = 1;
  do {

    if ( TempPara[i][1]<=iter && iter<TempPara[i+1][1] ){

      dynamics->target_temperature = TempPara[i][2] + (TempPara[i+1][2] - TempPara[i][2])*
            ((double)iter-(double)TempPara[i][1])/((double)TempPara[i+1][1]-(double)TempPara[i][1]);

      NH_switch = 1; 
    }

    i++;
  } while (NH_switch==0 && i<=(TempNum-1));  

  /****************************************************
                Velocity Verlet algorithm
  ****************************************************/

  if (iter==1){

    /****************************************************
                       Kinetic Energy 
    ****************************************************/

    thermo->ionic_kinetic=OpenMX_MD_Kinetic(md,Wscale,1);

    /* calculation of temperature (K) */
    dynamics->temperature = thermo->ionic_kinetic/(1.5*kB*(double)atomnum)*eV2Hartree;

    /****************************************************
     Nose-Hoover Hamiltonian which is a conserved quantity
    ****************************************************/

    dynamics->nose_energy = thermo->total + thermo->ionic_kinetic + 0.5*dynamics->nose_friction*dynamics->nose_friction*TempQ*Wscale
                        + 3.0*kB*(double)atomnum*dynamics->target_temperature*dynamics->nose_coordinate/eV2Hartree;

    /****************************************************
     write informatins to *.ene
    ****************************************************/

    if (myid==Host_ID){  

      sprintf(fileE,"%s%s.ene",filepath,filename);
      iterout_md(chemical, dynamics, thermo, energies, components, iter,MD_TimeStep*(iter-1),fileE);
    }

    /****************************************************
      first step in velocity Verlet 
    ****************************************************/

    dynamics->nose_friction = 0.0;
    dynamics->nose_coordinate = 0.0;

    const OpenMXMDStep step={OPENMX_MD_VERLET_DRIFT,dt,0.0,dynamics->nose_friction};
    OpenMX_MD_Move(md,step,Wscale,1.0,NULL,NULL);

    /* zeta( t+0.5*dt ) */

    dynamics->nose_next_friction = dynamics->nose_friction + (thermo->ionic_kinetic - 1.5*kB*(double)atomnum*dynamics->target_temperature/eV2Hartree)*dt/(TempQ*Wscale);
    dynamics->nose_friction = dynamics->nose_next_friction;

    /* R( r+dt ) */
    dynamics->nose_coordinate = dynamics->nose_coordinate + dynamics->nose_next_friction*dt;

  }

  else if (NH_switch==1) {

    /*****************************************************
     second step:

     refinement of v and zeta by a Newton-Raphson method
    *****************************************************/

    const double thermostat_mass=TempQ*Wscale;
    for (int iteration=0; iteration<21; ++iteration) {
      double moments[3]={0.0};
#pragma omp parallel for schedule(static) reduction(+:moments[:3]) if(Matomnum>=128)
      for (int local=1; local<=Matomnum; ++local) {
        const int atom=M2G[local];
        double *xyz=Gxyz[atom];
        const double mass=xyz[20]*Wscale;
        double speed2=0.0, residual_velocity=0.0;
        for (int d=0; d<3; ++d) {
          if (atom_Fixed_XYZ[atom][d+1]==0) {
            const double v=xyz[24+d], acceleration=-xyz[17+d]/mass;
            const double residual=xyz[27+d]+(acceleration-dynamics->nose_friction*v)*0.5*dt-v;
            xyz[30+d]=residual;
            speed2+=v*v;
            residual_velocity+=residual*v;
          }
        }
        moments[0]+=0.5*xyz[20]*Wscale*speed2;
        moments[1]+=mass*residual_velocity*dt/thermostat_mass;
        moments[2]-=0.5*xyz[20]*Wscale*speed2*dt*dt/thermostat_mass;
      }
      MPI_Allreduce(MPI_IN_PLACE,moments,3,MPI_DOUBLE,MPI_SUM,mpi_comm_level1);
      thermo->ionic_kinetic=moments[0];

      const double h=dynamics->nose_next_friction
          +(thermo->ionic_kinetic-1.5*kB*(double)atomnum*dynamics->target_temperature/eV2Hartree)
             *dt/thermostat_mass-dynamics->nose_friction;
      const double diagonal=dynamics->nose_friction*0.5*dt+1.0;
      const double dzeta=(-h*diagonal-moments[1])/(-diagonal+moments[2]);

      double correction2=0.0;
#pragma omp parallel for schedule(static) reduction(+:correction2) if(Matomnum>=128)
      for (int local=1; local<=Matomnum; ++local) {
        const int atom=M2G[local];
        double *xyz=Gxyz[atom];
        for (int d=0; d<3; ++d) {
          if (atom_Fixed_XYZ[atom][d+1]==0) {
            const double dv=(xyz[30+d]-0.5*xyz[24+d]*dt*dzeta)/diagonal;
            xyz[24+d]+=dv;
            correction2+=dv*dv;
          }
        }
      }
      dynamics->nose_friction+=dzeta;
      MPI_Allreduce(MPI_IN_PLACE,&correction2,1,MPI_DOUBLE,MPI_SUM,mpi_comm_level1);
      if (correction2+dzeta*dzeta<1.0e-12) break;
    }

    /****************************************************
     correct so that the sum of velocities can be zero.                              
    ****************************************************/

    OpenMX_MD_Center(md,24);

    /****************************************************
                       Kinetic Energy 
    ****************************************************/

    thermo->ionic_kinetic=OpenMX_MD_Kinetic(md,Wscale,1);

    /* calculation of temperature (K) */

    dynamics->temperature = thermo->ionic_kinetic/(1.5*kB*(double)atomnum)*eV2Hartree;

    /****************************************************
     Nose-Hoover Hamiltonian which is a conserved quantity
    ****************************************************/

    dynamics->nose_energy = thermo->total + thermo->ionic_kinetic + 0.5*dynamics->nose_friction*dynamics->nose_friction*TempQ*Wscale
                        + 3.0*kB*(double)atomnum*dynamics->target_temperature*dynamics->nose_coordinate/eV2Hartree;

    /****************************************************
     write informatins to *.ene
    ****************************************************/

    if (myid==Host_ID){  

      sprintf(fileE,"%s%s.ene",filepath,filename);
      iterout_md(chemical, dynamics, thermo, energies, components, iter,MD_TimeStep*(iter-1),fileE);
    }

    /*****************************************************
     first step:

       v(t)    ->  v(t+0.5*dt) 
       r(t)    ->  r(t+dt) 
       zeta(t) ->  zeta(t+0.5*dt)
       R(t)    ->  R(t+dt) 
    *****************************************************/

    if (MD_Opt_OK!=1 && iter!=MD_IterNumber){

      const OpenMXMDStep step={OPENMX_MD_VERLET_DRIFT,dt,0.0,dynamics->nose_friction};
      OpenMX_MD_Move(md,step,Wscale,1.0,NULL,NULL);

      /* zeta( t+0.5*dt ) */

      dynamics->nose_next_friction = dynamics->nose_friction + (thermo->ionic_kinetic - 1.5*kB*(double)atomnum*dynamics->target_temperature/eV2Hartree)*dt/(TempQ*Wscale);
      dynamics->nose_friction = dynamics->nose_next_friction;

      /* R( r+dt ) */
      dynamics->nose_coordinate = dynamics->nose_coordinate + dynamics->nose_next_friction*dt;

    }

  }

  else {

    /****************************************************
      second step in velocity Verlet 
    ****************************************************/

    const OpenMXMDStep step={OPENMX_MD_VERLET_KICK,dt,0.0,0.0};
    OpenMX_MD_Move(md,step,Wscale,1.0,NULL,NULL);

    /****************************************************
     correct so that the sum of velocities can be zero.                              
    ****************************************************/

    OpenMX_MD_Center(md,24);

    /****************************************************
                       Kinetic Energy 
    ****************************************************/

    thermo->ionic_kinetic=OpenMX_MD_Kinetic(md,Wscale,1);

    /* calculation of temperature (K) */
    dynamics->temperature = thermo->ionic_kinetic/(1.5*kB*(double)atomnum)*eV2Hartree;

    /****************************************************
     Nose-Hoover Hamiltonian which is a conserved quantity
    ****************************************************/

    dynamics->nose_energy = thermo->total + thermo->ionic_kinetic + 0.5*dynamics->nose_friction*dynamics->nose_friction*TempQ*Wscale
                        + 3.0*kB*(double)atomnum*dynamics->target_temperature*dynamics->nose_coordinate/eV2Hartree;

    /****************************************************
     write informatins to *.ene
    ****************************************************/

    if (myid==Host_ID){

      sprintf(fileE,"%s%s.ene",filepath,filename);
      iterout_md(chemical, dynamics, thermo, energies, components, iter,MD_TimeStep*(iter-1),fileE);
    }

    /****************************************************
      first step in velocity Verlet
    ****************************************************/

    if (MD_Opt_OK!=1 && iter!=MD_IterNumber){

      const OpenMXMDStep step={OPENMX_MD_VERLET_DRIFT,dt,0.0,0.0};
      OpenMX_MD_Move(md,step,Wscale,1.0,NULL,NULL);
    }

    /******************************************************
     Nose-Hoover Hamiltonian which is a conserved quantity
    ******************************************************/

    dynamics->nose_energy = thermo->total + thermo->ionic_kinetic + 0.5*dynamics->nose_friction*dynamics->nose_friction*TempQ*Wscale
                        + 3.0*kB*(double)atomnum*dynamics->target_temperature*dynamics->nose_coordinate/eV2Hartree;

  }
}


static void Delta_Factor(const OpenMXChemicalPotentials *chemical, OpenMXDynamicsState *dynamics, OpenMXThermodynamicEnergies *thermo, const OpenMXBandEnergies *energies, const OpenMXEnergyComponents *components, int iter)
{
  int i,j,Gc_AN,myid;
  char fileE[YOUSO10];
  double scaling_factor;
  static double tv0[4][4];
  static double Cell_Volume0;
  static int firsttime=1;

  MPI_Comm_rank(mpi_comm_level1,&myid);

  /****************************************************
   store tv into tv0 
  ****************************************************/

  if (firsttime){

    for (i=1; i<=3; i++){
      for (j=1; j<=3; j++){
	tv0[i][j] = tv[i][j];
      }
    }

    Cell_Volume0 = Cell_Volume;

    firsttime = 0;
  }

  /****************************************************
   write informatins to *.ene
  ****************************************************/

  if (myid==Host_ID){  
    sprintf(fileE,"%s%s.ene",filepath,filename);
    iterout_md(chemical, dynamics, thermo, energies, components, iter,MD_TimeStep*(iter-1),fileE);
  }

  /****************************************************
    scale tv 
  ****************************************************/

  if      (iter==1) scaling_factor = pow(0.940,1.0/3.0);
  else if (iter==2) scaling_factor = pow(0.960,1.0/3.0);
  else if (iter==3) scaling_factor = pow(0.980,1.0/3.0);
  else if (iter==4) scaling_factor = pow(1.020,1.0/3.0);
  else if (iter==5) scaling_factor = pow(1.040,1.0/3.0);
  else if (iter==6) scaling_factor = pow(1.060,1.0/3.0);

  if (scaling_factor<7){

    for (i=1; i<=3; i++){
      for (j=1; j<=3; j++){
	tv[i][j] = tv0[i][j]*scaling_factor;
      }
    }

    /****************************************************
                update cartesian coordinates
    ****************************************************/

    for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){

      Gxyz[Gc_AN][1] =  Cell_Gxyz[Gc_AN][1]*tv[1][1]
                      + Cell_Gxyz[Gc_AN][2]*tv[2][1]
                      + Cell_Gxyz[Gc_AN][3]*tv[3][1] + Grid_Origin[1];

      Gxyz[Gc_AN][2] =  Cell_Gxyz[Gc_AN][1]*tv[1][2]
                      + Cell_Gxyz[Gc_AN][2]*tv[2][2]
                      + Cell_Gxyz[Gc_AN][3]*tv[3][2] + Grid_Origin[2];

      Gxyz[Gc_AN][3] =  Cell_Gxyz[Gc_AN][1]*tv[1][3]
                      + Cell_Gxyz[Gc_AN][2]*tv[2][3]
                      + Cell_Gxyz[Gc_AN][3]*tv[3][3] + Grid_Origin[3];
    }
  }

}




static void EvsLC(const OpenMXChemicalPotentials *chemical, OpenMXDynamicsState *dynamics, OpenMXThermodynamicEnergies *thermo, const OpenMXBandEnergies *energies, const OpenMXEnergyComponents *components, int iter)
{
  int i,j,Gc_AN,myid;
  char fileE[YOUSO10];
  static double tv0[4][4];
  static int firsttime=1;

  MPI_Comm_rank(mpi_comm_level1,&myid);

  /****************************************************
   store tv into tv0 
  ****************************************************/

  if (firsttime){

    for (i=1; i<=3; i++){
      for (j=1; j<=3; j++){
	tv0[i][j] = tv[i][j];
      }
    }

    firsttime = 0;
  }

  /****************************************************
   write informatins to *.ene
  ****************************************************/

  if (myid==Host_ID){  
    sprintf(fileE,"%s%s.ene",filepath,filename);
    iterout_md(chemical, dynamics, thermo, energies, components, iter,MD_TimeStep*(iter-1),fileE);
  }

  /****************************************************
    scale tv 
  ****************************************************/

  for (i=1; i<=3; i++){
    if (MD_EvsLattice_flag[i-1]==1){
      for (j=1; j<=3; j++){
        tv[i][j] += tv0[i][j]*MD_EvsLattice_Step/100.0;
      }
    }
    else if (MD_EvsLattice_flag[i-1]==-1){
      for (j=1; j<=3; j++){
        tv[i][j] -= tv0[i][j]*MD_EvsLattice_Step/100.0;
      }
    }
  }

  /****************************************************
    update cartesian coordinates
  ****************************************************/

  for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){

    Gxyz[Gc_AN][1] =  Cell_Gxyz[Gc_AN][1]*tv[1][1]
                    + Cell_Gxyz[Gc_AN][2]*tv[2][1]
                    + Cell_Gxyz[Gc_AN][3]*tv[3][1] + Grid_Origin[1];

    Gxyz[Gc_AN][2] =  Cell_Gxyz[Gc_AN][1]*tv[1][2]
                    + Cell_Gxyz[Gc_AN][2]*tv[2][2]
                    + Cell_Gxyz[Gc_AN][3]*tv[3][2] + Grid_Origin[2];

    Gxyz[Gc_AN][3] =  Cell_Gxyz[Gc_AN][1]*tv[1][3]
                    + Cell_Gxyz[Gc_AN][2]*tv[2][3]
                    + Cell_Gxyz[Gc_AN][3]*tv[3][3] + Grid_Origin[3];
  }
}



void Calc_Temp_Atoms(OpenMXDynamicsState *dynamics, OpenMXThermodynamicEnergies *thermo, int iter)
{
  int i,Mc_AN,Gc_AN,j,My_Nr,Nr;
  double My_Ukc,sum,Wscale,dt,v;

  /* calculation of temperature (K) of the atomic system */

  dt = 41.3411*MD_TimeStep;
  Wscale = unified_atomic_mass_unit/electron_mass;
  thermo->ionic_kinetic = 0.0;
  My_Ukc = 0.0;
  My_Nr = 0;
  Nr = 0;

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];
    sum = 0.0;
    for (j=1; j<=3; j++){
      if (atom_Fixed_XYZ[Gc_AN][j]==0){
	sum += Gxyz[Gc_AN][j+23]*Gxyz[Gc_AN][j+23];
        My_Nr++;
      }
    }
    My_Ukc += 0.5*Gxyz[Gc_AN][20]*Wscale*sum;
  }

  MPI_Allreduce(&My_Ukc, &thermo->ionic_kinetic, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
  MPI_Allreduce(&My_Nr, &Nr, 1, MPI_INT, MPI_SUM, mpi_comm_level1);

  dynamics->temperature = thermo->ionic_kinetic/(0.5*kB*(double)Nr)*eV2Hartree;

  /* for old version */
  /*
  Temp = Ukc/(1.5*kB*(double)atomnum)*eV2Hartree;
  */

  /* calculation of given temperature (K) of the atomic system */

  for (i=1; i<=TempNum; i++) {
    if( (iter>TempPara[i-1][1]) && (iter<=TempPara[i][1]) ) {
      dynamics->target_temperature = TempPara[i-1][2] + (TempPara[i][2] - TempPara[i-1][2])*
          ((double)iter-(double)TempPara[i-1][1])/((double)TempPara[i][1]-(double)TempPara[i-1][1]);
    }
  }
}




int RestartFiles4GeoOpt(char *mode)
{
  int i,j,num;
  int success_flag;
  int numprocs,myid;
  char file_name[YOUSO10];
  FILE *fp;
  
  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);
   
  /* write */
  if ( strcasecmp(mode,"write")==0) {

    sprintf(file_name,"%s%s_rst/%s.rst4gopt",filepath,filename,filename);

    if (myid==Host_ID){

      if ( (fp = fopen(file_name,"wb")) != NULL ){

	num = M_GDIIS_HISTORY + 1;
      
	for(i=0; i<num; i++) {
	  for(j=0; j<(atomnum+1); j++) {
	    fwrite(GxyzHistoryIn[i][j], sizeof(double), 4, fp);
	  }
	}

	for(i=0; i<num; i++) {
	  for(j=0; j<(atomnum+1); j++) {
	    fwrite(GxyzHistoryR[i][j], sizeof(double), 4, fp);
	  }
	}

	for(i=0; i<Extrapolated_Charge_History; i++) {
	  fwrite(His_Gxyz[i], sizeof(double), atomnum*3, fp);
	}

	/* Geometry_Opt_DIIS_EF */
	if (MD_switch==4){
	  for (i=0; i<(3*atomnum+2); i++){
	    fwrite(Hessian[i], sizeof(double), 3*atomnum+2, fp);
	  }
	}

	/* BFGS */
	if (MD_switch==5){
	  for (i=0; i<(3*atomnum+2); i++){
	    fwrite(InvHessian[i], sizeof(double), 3*atomnum+2, fp);
	  }
	}

	/* RF */
	if (MD_switch==6){
	  for (i=0; i<(3*atomnum+2); i++){
	    fwrite(Hessian[i], sizeof(double), 3*atomnum+2, fp);
	  }
	}

	/* RFC */
	if (MD_switch==18){
	  for (i=0; i<(3*atomnum+10); i++){
	    fwrite(Hessian[i], sizeof(double), 3*atomnum+10, fp);
	  }
	}

	success_flag = 1;
	fclose(fp);
      }
      else{
        printf("Failure of saving %s\n",file_name);
        success_flag = 0;
      }
    }

    else{
      success_flag = 1;
    }

    /* MPI_Bcast */    
    MPI_Bcast(&success_flag, 1, MPI_INT, Host_ID, mpi_comm_level1);
  }

  /* read */
  else if (strcasecmp(mode,"read")==0) {

    sprintf(file_name,"%s%s_rst/%s.rst4gopt",filepath,filename,filename);

    if ((fp = fopen(file_name,"rb")) != NULL){

      num = M_GDIIS_HISTORY + 1;
      
      for(i=0; i<num; i++) {
	for(j=0; j<(atomnum+1); j++) {
	  fread(GxyzHistoryIn[i][j], sizeof(double), 4, fp);
	}
      }

      for(i=0; i<num; i++) {
	for(j=0; j<(atomnum+1); j++) {
	  fread(GxyzHistoryR[i][j], sizeof(double), 4, fp);
	}
      }

      for(i=0; i<Extrapolated_Charge_History; i++) {
	fread(His_Gxyz[i], sizeof(double), atomnum*3, fp);
      }

      /* Geometry_Opt_DIIS_EF */
      if (MD_switch==4){
	for (i=0; i<(3*atomnum+2); i++){
	  fread(Hessian[i], sizeof(double), 3*atomnum+2, fp);
	}
      }

      /* BFGS */
      if (MD_switch==5){
	for (i=0; i<(3*atomnum+2); i++){
	  fread(InvHessian[i], sizeof(double), 3*atomnum+2, fp);
	}
      }

      /* RF */
      if (MD_switch==6){
	for (i=0; i<(3*atomnum+2); i++){
	  fread(Hessian[i], sizeof(double), 3*atomnum+2, fp);
	}
      }

      /* RFC */
      if (MD_switch==18){
	for (i=0; i<(3*atomnum+10); i++){
	  fread(Hessian[i], sizeof(double), 3*atomnum+10, fp);
	}
      }

      success_flag = 1;
      fclose(fp);

    }
    else{
      printf("Failure of reading %s\n",file_name);
      success_flag = 0;
    }
  }

  return success_flag;
}


void Output_abc_file(OpenMXThermodynamicEnergies *thermo, int iter)
{
  int i,j,k;
  FILE *fp_abc;
  char buf[fp_bsize];          /* setvbuf */
  char fileabc[YOUSO10]="",dotabc[YOUSO10],iter_s[YOUSO10];

  printf("\nOutputting to abc file.\n");
  snprintf(iter_s,sizeof(iter_s),"%d",iter);
  strcpy(dotabc,".abc");
  fnjoint(iter_s,dotabc,fileabc);
  fnjoint(filepath,filename,fileabc);

  if ((fp_abc = fopen(fileabc,"w")) != NULL){

    setvbuf(fp_abc,buf,_IOFBF,fp_bsize);  /* setvbuf */

    fprintf(fp_abc,"# Total Energy (Hartree), abc-coordinates and Lattice Parameters (Ang) #\n");
    fprintf(fp_abc,"TotalEnergy   %18.15f\n",thermo->total);
    fprintf(fp_abc,"Atoms.Number  %i\n",atomnum);
    fprintf(fp_abc,"<Atoms.SpeciesAndCoordinates\n");

    /* atomic coordinates in case of colliear non-spin polarization */

    if (SpinP_switch==0){

      for (k=1; k<=atomnum; k++){
	i = WhatSpecies[k];
	j = Spe_WhatAtom[i];
	fprintf(fp_abc," %4d  %4s   %18.15f %18.15f %18.15f\n",
		k,Atom_Symbol[j],
		Gxyz[k][1]*BohrR,Gxyz[k][2]*BohrR,Gxyz[k][3]*BohrR);
      }
    }

    /* atomic coordinates in case of colliear spin polarization */

    else if (SpinP_switch==1){

      for (k=1; k<=atomnum; k++){
	i = WhatSpecies[k];
	j = Spe_WhatAtom[i];

        if ( 0.0<=(InitN_USpin[k] - InitN_DSpin[k])){
  	   fprintf(fp_abc," %4d  %4s   %18.15f %18.15f %18.15f %18.15f %18.15f %18.14f %18.14f\n",
		   k,Atom_Symbol[j],
		   Gxyz[k][1]*BohrR,Gxyz[k][2]*BohrR,Gxyz[k][3]*BohrR,
                   InitN_USpin[k]+InitN_DSpin[k],
                   fabs(InitN_USpin[k]-InitN_DSpin[k]),
                   0.0,0.0);
	}
        else {
  	   fprintf(fp_abc," %4d  %4s   %18.15f %18.15f %18.15f %18.15f %18.15f %18.14f %18.14f\n",
		   k,Atom_Symbol[j],
		   Gxyz[k][1]*BohrR,Gxyz[k][2]*BohrR,Gxyz[k][3]*BohrR,
                   InitN_USpin[k]+InitN_DSpin[k],
                   fabs(InitN_USpin[k]-InitN_DSpin[k]),
                   180.0,0.0);
        }

      }
    }

    /* atomic coordinates in case of non-colliear spin polarization */

    else if (SpinP_switch==3){

      for (k=1; k<=atomnum; k++){
	i = WhatSpecies[k];
	j = Spe_WhatAtom[i];

	fprintf(fp_abc," %4d  %4s   %18.15f %18.15f %18.15f %18.15f %18.15f %18.14f %18.14f\n",
		k,Atom_Symbol[j],
		Gxyz[k][1]*BohrR,Gxyz[k][2]*BohrR,Gxyz[k][3]*BohrR,
		InitN_USpin[k]+InitN_DSpin[k],
		InitN_USpin[k]-InitN_DSpin[k],
		Angle0_Spin[k]/PI*180.0, Angle1_Spin[k]/PI*180.0);

        /* check negativity of magnetic moment */

        if ( (InitN_USpin[k]-InitN_DSpin[k])<0.0 ){

          printf("found an error (negativity) in writing the abc file.\n");
          MPI_Finalize();
          exit(1);
        }

      }
    }

    /* cell vectors */

    fprintf(fp_abc,"Atoms.SpeciesAndCoordinates>\n");
    fprintf(fp_abc,"<Atoms.UnitVectors\n");
    fprintf(fp_abc,"%18.15f %18.15f %18.15f\n",tv[1][1]*BohrR,tv[1][2]*BohrR,tv[1][3]*BohrR);
    fprintf(fp_abc,"%18.15f %18.15f %18.15f\n",tv[2][1]*BohrR,tv[2][2]*BohrR,tv[2][3]*BohrR);
    fprintf(fp_abc,"%18.15f %18.15f %18.15f\n",tv[3][1]*BohrR,tv[3][2]*BohrR,tv[3][3]*BohrR);
    fprintf(fp_abc,"Atoms.UnitVectors>\n");
    fprintf(fp_abc,"# End #\n");
    fclose(fp_abc);
  }
  else
    printf("error(1) in MD_pac.c\n");

}


void Estimate_Initial_Hessian(OpenMXGeometryOpt *geometry, int diis_iter, int CellOpt_flag, double itv[4][4])
{
  int myid;

  /* MPI communication */

  MPI_Comm_rank(mpi_comm_level1,&myid);

  /*******************************************
               the identity matrix 
  *******************************************/

  if (Initial_Hessian_flag==0){
    if (myid!=Host_ID) return;
    const int n=3*atomnum+(CellOpt_flag==1 ? 9 : 0);
#pragma omp parallel for schedule(static) if(n>=256)
    for (int i=1; i<=n; ++i) {
      memset(Hessian[i]+1,0,(size_t)n*sizeof(double));
      Hessian[i][i]=1.0;
    }
  }

  /**************************************************************
      A model Hessian proposed by H.B. Schlegel

      Refs.
      H.B. Schlegel, Theo. Chim. Acta (Berl.) 66, 333 (1984).
      J.M.Wittbrodt and H.B. Schlegel, 
      J. Mol. Str. (Theochem) 398-399, 55 (1997).
  **************************************************************/

  else if (Initial_Hessian_flag==1){
    const OpenMXSchlegelModel model={
      .local_atoms=Matomnum,
      .local_to_global=M2G,
      .species=WhatSpecies,
      .elements=Spe_WhatAtom,
      .neighbors=FNAN,
      .neighbor_atoms=natn,
      .neighbor_images=ncn,
      .positions=Gxyz,
      .translations=atv,
      .distances=Dis
    };
    const double (*cell)[4]=CellOpt_flag==1 ? tv : NULL;
    OpenMX_Geometry_Schlegel(geometry,Hessian,&model,cell,itv);
    if (myid!=Host_ID) return;

    int I,J,i,j;
    /* check constraint and correct Hessian */

    for (I=1; I<=atomnum; I++){     
      for (i=1; i<=3; i++){     

        if (atom_Fixed_XYZ[I][i]==1){

	  for (J=1; J<=atomnum; J++){     
	    for (j=1; j<=3; j++){     
	      Hessian[(I-1)*3+i][(J-1)*3+j] = 0.0;
	      Hessian[(J-1)*3+j][(I-1)*3+i] = 0.0;
	    }
	  }

          Hessian[(I-1)*3+i][(I-1)*3+i] = 1.0;
	}
      }
    }

  }

  /*******************************************
    an approximate Hessian by force fitting
  *******************************************/

  else if (Initial_Hessian_flag==2){

    int Gc_AN,Mc_AN,i,j,p,Rn;
    int wsp1,wsp2,m1,m2;
    int h_AN,Gh_AN,po,loopN;
    double **ep,**sig;
    double **dQ_dep,**dQ_dsig;
    double ***GLJ,ELJ[20],my_ELJ[20],r,r2,r3,g[4];
    double norm_sig,norm_ep,my_norm_sig,my_norm_ep;
    double sr,sr2,sr4,sr5,sr6,sr11,sr12;
    double df_dr,d2f_dr2,d2f_depdr,d2f_dsigdr;
    double dgx,dgy,dgz,coe_ep,coe_sig;
    double Q=0.0,Qold,My_Q,dg_dep[4],dg_dsig[4];
    double d2r_dx1dx2[4][4],d[4];
    double ep0[104],sig0[104];

    /* set sig0 in a.u. */

    sig0[  0] = 0.8908987/BohrR*0.75;
    sig0[  1] = 0.8908987/BohrR*0.75;
    sig0[  2] = 0.8908987/BohrR*0.64;
    sig0[  3] = 0.8908987/BohrR*2.68;  
    sig0[  4] = 0.8908987/BohrR*1.80;  
    sig0[  5] = 0.8908987/BohrR*1.64; 
    sig0[  6] = 0.8908987/BohrR*1.54; 
    sig0[  7] = 0.8908987/BohrR*1.50; 
    sig0[  8] = 0.8908987/BohrR*1.46; 
    sig0[  9] = 0.8908987/BohrR*1.42; 
    sig0[ 10] = 0.8908987/BohrR*1.38; 
    sig0[ 11] = 0.8908987/BohrR*3.08; 
    sig0[ 12] = 0.8908987/BohrR*2.60; 
    sig0[ 13] = 0.8908987/BohrR*2.36; 
    sig0[ 14] = 0.8908987/BohrR*2.22; 
    sig0[ 15] = 0.8908987/BohrR*2.12; 
    sig0[ 16] = 0.8908987/BohrR*2.04;
    sig0[ 17] = 0.8908987/BohrR*1.98; 

    sig0[ 18] = 0.8908987/BohrR*2.68; 
    sig0[ 19] = 0.8908987/BohrR*2.68; 
    sig0[ 20] = 0.8908987/BohrR*2.68; 
    sig0[ 21] = 0.8908987/BohrR*2.68; 
    sig0[ 22] = 0.8908987/BohrR*2.68; 
    sig0[ 23] = 0.8908987/BohrR*2.68; 
    sig0[ 24] = 0.8908987/BohrR*2.68; 
    sig0[ 25] = 0.8908987/BohrR*2.68; 
    sig0[ 26] = 0.8908987/BohrR*2.68; 
    sig0[ 27] = 0.8908987/BohrR*2.68; 
    sig0[ 28] = 0.8908987/BohrR*2.68; 
    sig0[ 29] = 0.8908987/BohrR*2.68; 
    sig0[ 30] = 0.8908987/BohrR*2.68; 
    sig0[ 31] = 0.8908987/BohrR*2.68; 
    sig0[ 32] = 0.8908987/BohrR*2.68; 
    sig0[ 33] = 0.8908987/BohrR*2.68; 
    sig0[ 34] = 0.8908987/BohrR*2.68; 
    sig0[ 35] = 0.8908987/BohrR*2.68; 
    sig0[ 36] = 0.8908987/BohrR*2.68; 
    sig0[ 37] = 0.8908987/BohrR*2.68; 
    sig0[ 38] = 0.8908987/BohrR*2.68; 
    sig0[ 39] = 0.8908987/BohrR*2.68; 
    sig0[ 40] = 0.8908987/BohrR*2.68; 
    sig0[ 41] = 0.8908987/BohrR*2.68; 
    sig0[ 42] = 0.8908987/BohrR*2.68; 
    sig0[ 43] = 0.8908987/BohrR*2.68; 
    sig0[ 44] = 0.8908987/BohrR*2.68; 
    sig0[ 45] = 0.8908987/BohrR*2.68; 
    sig0[ 46] = 0.8908987/BohrR*2.68; 

    sig0[ 47] = 0.8908987/BohrR*3.06; 

    sig0[ 48] = 0.8908987/BohrR*2.68; 
    sig0[ 49] = 0.8908987/BohrR*2.68; 
    sig0[ 50] = 0.8908987/BohrR*2.68; 
    sig0[ 51] = 0.8908987/BohrR*2.68; 
    sig0[ 52] = 0.8908987/BohrR*2.68; 
    sig0[ 53] = 0.8908987/BohrR*2.68; 
    sig0[ 54] = 0.8908987/BohrR*2.68; 
    sig0[ 55] = 0.8908987/BohrR*2.68; 
    sig0[ 56] = 0.8908987/BohrR*2.68; 
    sig0[ 57] = 0.8908987/BohrR*2.68; 
    sig0[ 58] = 0.8908987/BohrR*2.68; 
    sig0[ 59] = 0.8908987/BohrR*2.68; 
    sig0[ 60] = 0.8908987/BohrR*2.68; 
    sig0[ 61] = 0.8908987/BohrR*2.68; 
    sig0[ 62] = 0.8908987/BohrR*2.68; 
    sig0[ 63] = 0.8908987/BohrR*2.68; 
    sig0[ 64] = 0.8908987/BohrR*2.68; 
    sig0[ 65] = 0.8908987/BohrR*2.68; 
    sig0[ 66] = 0.8908987/BohrR*2.68; 
    sig0[ 67] = 0.8908987/BohrR*2.68; 
    sig0[ 68] = 0.8908987/BohrR*2.68; 
    sig0[ 69] = 0.8908987/BohrR*2.68; 
    sig0[ 70] = 0.8908987/BohrR*2.68; 
    sig0[ 71] = 0.8908987/BohrR*2.68; 
    sig0[ 72] = 0.8908987/BohrR*2.68; 
    sig0[ 73] = 0.8908987/BohrR*2.68; 
    sig0[ 74] = 0.8908987/BohrR*2.68; 
    sig0[ 75] = 0.8908987/BohrR*2.68; 
    sig0[ 76] = 0.8908987/BohrR*2.68; 
    sig0[ 77] = 0.8908987/BohrR*2.68; 
    sig0[ 78] = 0.8908987/BohrR*2.68; 
    sig0[ 79] = 0.8908987/BohrR*2.68; 
    sig0[ 80] = 0.8908987/BohrR*2.68; 
    sig0[ 81] = 0.8908987/BohrR*2.68; 
    sig0[ 82] = 0.8908987/BohrR*2.68; 
    sig0[ 83] = 0.8908987/BohrR*2.68; 
    sig0[ 84] = 0.8908987/BohrR*2.68; 
    sig0[ 85] = 0.8908987/BohrR*2.68; 
    sig0[ 86] = 0.8908987/BohrR*2.68; 
    sig0[ 87] = 0.8908987/BohrR*2.68; 
    sig0[ 88] = 0.8908987/BohrR*2.68; 
    sig0[ 89] = 0.8908987/BohrR*2.68; 
    sig0[ 90] = 0.8908987/BohrR*2.68; 
    sig0[ 91] = 0.8908987/BohrR*2.68; 
    sig0[ 92] = 0.8908987/BohrR*2.68; 
    sig0[ 93] = 0.8908987/BohrR*2.68; 
    sig0[ 94] = 0.8908987/BohrR*2.68; 
    sig0[ 95] = 0.8908987/BohrR*2.68; 
    sig0[ 96] = 0.8908987/BohrR*2.68; 
    sig0[ 97] = 0.8908987/BohrR*2.68; 
    sig0[ 98] = 0.8908987/BohrR*2.68; 
    sig0[ 99] = 0.8908987/BohrR*2.68; 
    sig0[100] = 0.8908987/BohrR*2.68; 
    sig0[101] = 0.8908987/BohrR*2.68; 
    sig0[102] = 0.8908987/BohrR*2.68; 
    sig0[103] = 0.8908987/BohrR*2.68; 

    /* set ep0 */

    ep0[  0] = 0.01/eV2Hartree;
    ep0[  1] = 0.27/eV2Hartree;
    ep0[  2] = 0.01/eV2Hartree;  
    ep0[  3] = 0.19/eV2Hartree;
    ep0[  4] = 0.39/eV2Hartree;
    ep0[  5] = 0.69/eV2Hartree;
    ep0[  6] = 0.87/eV2Hartree;
    ep0[  7] = 0.58/eV2Hartree;
    ep0[  8] = 0.31/eV2Hartree;
    ep0[  9] = 0.10/eV2Hartree;
    ep0[ 10] = 0.01/eV2Hartree;
    ep0[ 11] = 0.13/eV2Hartree;
    ep0[ 12] = 0.18/eV2Hartree;
    ep0[ 13] = 0.40/eV2Hartree;
    ep0[ 14] = 0.55/eV2Hartree;
    ep0[ 15] = 0.41/eV2Hartree;
    ep0[ 16] = 0.34/eV2Hartree;
    ep0[ 17] = 0.17/eV2Hartree;
    ep0[ 18] = 0.01/eV2Hartree;
    ep0[ 19] = 0.11/eV2Hartree;
    ep0[ 20] = 0.22/eV2Hartree;
    ep0[ 21] = 0.46/eV2Hartree;

    ep0[ 22] = 0.01/eV2Hartree;
    ep0[ 23] = 0.01/eV2Hartree;
    ep0[ 24] = 0.01/eV2Hartree;
    ep0[ 25] = 0.01/eV2Hartree;
    ep0[ 26] = 0.01/eV2Hartree;
    ep0[ 27] = 0.01/eV2Hartree;
    ep0[ 28] = 0.01/eV2Hartree;
    ep0[ 29] = 0.01/eV2Hartree;
    ep0[ 30] = 0.01/eV2Hartree;
    ep0[ 31] = 0.01/eV2Hartree;
    ep0[ 32] = 0.01/eV2Hartree;
    ep0[ 33] = 0.01/eV2Hartree;
    ep0[ 34] = 0.01/eV2Hartree;
    ep0[ 35] = 0.01/eV2Hartree;
    ep0[ 36] = 0.01/eV2Hartree;
    ep0[ 37] = 0.01/eV2Hartree;
    ep0[ 38] = 0.01/eV2Hartree;
    ep0[ 39] = 0.01/eV2Hartree;
    ep0[ 40] = 0.01/eV2Hartree;
    ep0[ 41] = 0.01/eV2Hartree;
    ep0[ 42] = 0.01/eV2Hartree;
    ep0[ 43] = 0.01/eV2Hartree;
    ep0[ 44] = 0.01/eV2Hartree;
    ep0[ 45] = 0.01/eV2Hartree;
    ep0[ 46] = 0.01/eV2Hartree;
    ep0[ 47] = 0.35/eV2Hartree;
    ep0[ 48] = 0.01/eV2Hartree; 
    ep0[ 49] = 0.01/eV2Hartree;
    ep0[ 50] = 0.01/eV2Hartree;
    ep0[ 51] = 0.01/eV2Hartree;
    ep0[ 52] = 0.01/eV2Hartree;
    ep0[ 53] = 0.01/eV2Hartree;
    ep0[ 54] = 0.01/eV2Hartree;
    ep0[ 55] = 0.01/eV2Hartree;
    ep0[ 56] = 0.01/eV2Hartree;
    ep0[ 57] = 0.01/eV2Hartree;
    ep0[ 58] = 0.01/eV2Hartree;
    ep0[ 59] = 0.01/eV2Hartree;
    ep0[ 60] = 0.01/eV2Hartree;
    ep0[ 61] = 0.01/eV2Hartree;
    ep0[ 62] = 0.01/eV2Hartree;
    ep0[ 63] = 0.01/eV2Hartree;
    ep0[ 64] = 0.01/eV2Hartree;
    ep0[ 65] = 0.01/eV2Hartree;
    ep0[ 66] = 0.01/eV2Hartree;
    ep0[ 67] = 0.01/eV2Hartree;
    ep0[ 68] = 0.01/eV2Hartree;
    ep0[ 69] = 0.01/eV2Hartree;
    ep0[ 70] = 0.01/eV2Hartree;
    ep0[ 71] = 0.01/eV2Hartree;
    ep0[ 72] = 0.01/eV2Hartree;
    ep0[ 73] = 0.01/eV2Hartree;
    ep0[ 74] = 0.01/eV2Hartree;
    ep0[ 75] = 0.01/eV2Hartree;
    ep0[ 76] = 0.01/eV2Hartree;
    ep0[ 77] = 0.01/eV2Hartree;
    ep0[ 78] = 0.01/eV2Hartree;
    ep0[ 79] = 0.01/eV2Hartree;
    ep0[ 80] = 0.01/eV2Hartree;
    ep0[ 81] = 0.01/eV2Hartree;
    ep0[ 82] = 0.01/eV2Hartree;
    ep0[ 83] = 0.01/eV2Hartree;
    ep0[ 84] = 0.01/eV2Hartree;
    ep0[ 85] = 0.01/eV2Hartree;
    ep0[ 86] = 0.01/eV2Hartree;
    ep0[ 87] = 0.01/eV2Hartree;
    ep0[ 88] = 0.01/eV2Hartree;
    ep0[ 89] = 0.01/eV2Hartree;
    ep0[ 90] = 0.01/eV2Hartree;
    ep0[ 91] = 0.01/eV2Hartree;
    ep0[ 92] = 0.01/eV2Hartree;
    ep0[ 93] = 0.01/eV2Hartree;
    ep0[ 94] = 0.01/eV2Hartree;
    ep0[ 95] = 0.01/eV2Hartree;
    ep0[ 96] = 0.01/eV2Hartree;
    ep0[ 97] = 0.01/eV2Hartree;
    ep0[ 98] = 0.01/eV2Hartree;
    ep0[ 99] = 0.01/eV2Hartree;
    ep0[100] = 0.01/eV2Hartree;
    ep0[101] = 0.01/eV2Hartree;
    ep0[102] = 0.01/eV2Hartree;
    ep0[103] = 0.01/eV2Hartree;

    /* allocation of arrays */

    GLJ = (double***)malloc(sizeof(double**)*(M_GDIIS_HISTORY+1));
    for (p=0; p<(M_GDIIS_HISTORY+1); p++){
      GLJ[p] = (double**)malloc(sizeof(double*)*(Matomnum+1));
      for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){
	GLJ[p][Mc_AN] = (double*)malloc(sizeof(double)*4);
	for (i=0; i<4; i++) GLJ[p][Mc_AN][i] = 0.0;
      }
    }

    ep = (double**)malloc(sizeof(double*)*(Matomnum+1));
    for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){
      if (Mc_AN==0){
        ep[Mc_AN] = (double*)malloc(sizeof(double)*1);
        ep[Mc_AN][0] = 0.0;
      }
      else {
        Gc_AN = M2G[Mc_AN];    
        wsp1 = WhatSpecies[Gc_AN];
        m1 = Spe_WhatAtom[wsp1];

        ep[Mc_AN] = (double*)malloc(sizeof(double)*(FNAN[Gc_AN]+1));
        for (h_AN=1; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
          wsp2 = WhatSpecies[Gh_AN];
          m2 = Spe_WhatAtom[wsp2];
          ep[Mc_AN][h_AN] = sqrt(ep0[m1]*ep0[m2]);  /* set the initial value */
	}
      }
    }

    sig = (double**)malloc(sizeof(double*)*(Matomnum+1));
    for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){
      if (Mc_AN==0){
        sig[Mc_AN] = (double*)malloc(sizeof(double)*1);
        sig[Mc_AN][0] = 0.0;
      }
      else {
        Gc_AN = M2G[Mc_AN];    
        wsp1 = WhatSpecies[Gc_AN];
        m1 = Spe_WhatAtom[wsp1];

        sig[Mc_AN] = (double*)malloc(sizeof(double)*(FNAN[Gc_AN]+1));
        for (h_AN=1; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
          wsp2 = WhatSpecies[Gh_AN];
          m2 = Spe_WhatAtom[wsp2];
          sig[Mc_AN][h_AN] = 0.5*(sig0[m1]+sig0[m2]); /* set the initial value */
	}
      }
    }

    dQ_dep = (double**)malloc(sizeof(double*)*(Matomnum+1));
    for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){
      if (Mc_AN==0){
        dQ_dep[Mc_AN] = (double*)malloc(sizeof(double)*1);
      }
      else {
        Gc_AN = M2G[Mc_AN];    
        dQ_dep[Mc_AN] = (double*)malloc(sizeof(double)*(FNAN[Gc_AN]+1));
      }
    }

    dQ_dsig = (double**)malloc(sizeof(double*)*(Matomnum+1));
    for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){
      if (Mc_AN==0){
        dQ_dsig[Mc_AN] = (double*)malloc(sizeof(double)*1);
      }
      else {
        Gc_AN = M2G[Mc_AN];    
        dQ_dsig[Mc_AN] = (double*)malloc(sizeof(double)*(FNAN[Gc_AN]+1));
      }
    }

    /*************************************************************
      optimization of the obeject function for gradient fitting 
    *************************************************************/

    po = 0;
    loopN = 1;

    do {

      My_Q = 0.0;

      for (p=0; p<diis_iter; p++) {

	/* calculate the LJ total energy, the LJ gradients, and the object function Q */

	my_ELJ[p] = 0.0;

	for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	  Gc_AN = M2G[Mc_AN];    

	  GLJ[p][Mc_AN][1] = 0.0;
	  GLJ[p][Mc_AN][2] = 0.0;
	  GLJ[p][Mc_AN][3] = 0.0;

	  for (h_AN=1; h_AN<=FNAN[Gc_AN]; h_AN++){
	    Gh_AN = natn[Gc_AN][h_AN];
	    Rn = ncn[Gc_AN][h_AN];

            d[1] = GxyzHistoryIn[p][Gc_AN][1] - GxyzHistoryIn[p][Gh_AN][1] - atv[Rn][1];
            d[2] = GxyzHistoryIn[p][Gc_AN][2] - GxyzHistoryIn[p][Gh_AN][2] - atv[Rn][2];
            d[3] = GxyzHistoryIn[p][Gc_AN][3] - GxyzHistoryIn[p][Gh_AN][3] - atv[Rn][3];
	    r = sqrt(d[1]*d[1]+d[2]*d[2]+d[3]*d[3]);

	    g[1] = d[1]/r;
	    g[2] = d[2]/r;
	    g[3] = d[3]/r;

	    sr = sig[Mc_AN][h_AN]/r;
	    sr2 = sr*sr;
	    sr4 = sr2*sr2;
	    sr6 = sr2*sr4;
	    sr12 = sr6*sr6;

	    my_ELJ[p] += 2.0*ep[Mc_AN][h_AN]*(sr12 - sr6); /* The factor of 0.5 is considered. */
         
	    d2f_depdr = 24.0*(sr6 - 2.0*sr12)/r;
	    df_dr = d2f_depdr*ep[Mc_AN][h_AN];

	    GLJ[p][Mc_AN][1] += g[1]*df_dr;
	    GLJ[p][Mc_AN][2] += g[2]*df_dr;
	    GLJ[p][Mc_AN][3] += g[3]*df_dr;
	  }

	  dgx = GLJ[p][Mc_AN][1] - GxyzHistoryR[p][Gc_AN][1];
	  dgy = GLJ[p][Mc_AN][2] - GxyzHistoryR[p][Gc_AN][2];
	  dgz = GLJ[p][Mc_AN][3] - GxyzHistoryR[p][Gc_AN][3];
      
	  My_Q += 0.5*(dgx*dgx+dgy*dgy+dgz*dgz);  

	} /* Mc_AN */
      } /* p */

      /* MPI of Q */ 

      Qold = Q;
      MPI_Allreduce(&My_Q, &Q, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

      for (p=0; p<diis_iter; p++) {
        MPI_Allreduce(&my_ELJ[p], &ELJ[p], 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
      }

      /*
      for (p=0; p<diis_iter; p++) {
        printf("ELJ[%d]=%18.15f\n",p,ELJ[p]);
	for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	  printf("Mc_AN=%2d p=%2d GLJ=%18.15f %18.15f %18.15f\n",
                  Gc_AN,p,GLJ[p][Mc_AN][1],GLJ[p][Mc_AN][2],GLJ[p][Mc_AN][3]); 
	}    
      }
      */

      /* calculate the gradients of Q with respect to ep and sig */

      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	Gc_AN = M2G[Mc_AN];    
	for (h_AN=1; h_AN<=FNAN[Gc_AN]; h_AN++){
	  dQ_dep[Mc_AN][h_AN]  = 0.0;
	  dQ_dsig[Mc_AN][h_AN] = 0.0;
	}
      }

      for (p=0; p<diis_iter; p++) {
	for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	  Gc_AN = M2G[Mc_AN];    

	  for (h_AN=1; h_AN<=FNAN[Gc_AN]; h_AN++){

	    Gh_AN = natn[Gc_AN][h_AN];
	    Rn = ncn[Gc_AN][h_AN];

            d[1] = GxyzHistoryIn[p][Gc_AN][1] - GxyzHistoryIn[p][Gh_AN][1] - atv[Rn][1];
            d[2] = GxyzHistoryIn[p][Gc_AN][2] - GxyzHistoryIn[p][Gh_AN][2] - atv[Rn][2];
            d[3] = GxyzHistoryIn[p][Gc_AN][3] - GxyzHistoryIn[p][Gh_AN][3] - atv[Rn][3];
	    r = sqrt(d[1]*d[1]+d[2]*d[2]+d[3]*d[3]);

	    g[1] = d[1]/r;
	    g[2] = d[2]/r;
	    g[3] = d[3]/r;

	    sr = sig[Mc_AN][h_AN]/r;
	    sr2 = sr*sr;
	    sr4 = sr2*sr2;
	    sr5 = sr*sr4;
	    sr6 = sr2*sr4;
	    sr11 = sr5*sr6;
	    sr12 = sr6*sr6;

	    d2f_depdr = 24.0*(sr6 - 2.0*sr12)/r;
	    df_dr = d2f_depdr*ep[Mc_AN][h_AN];
	    d2f_dsigdr = 144.0*ep[Mc_AN][h_AN]*(sr5 - 4.0*sr11)/r/r;

	    dg_dep[1] = g[1]*d2f_depdr;
	    dg_dep[2] = g[2]*d2f_depdr;
	    dg_dep[3] = g[3]*d2f_depdr;

	    dg_dsig[1] = g[1]*d2f_dsigdr;
	    dg_dsig[2] = g[2]*d2f_dsigdr;
	    dg_dsig[3] = g[3]*d2f_dsigdr;

  	    dgx = GLJ[p][Mc_AN][1] - GxyzHistoryR[p][Gc_AN][1];
	    dgy = GLJ[p][Mc_AN][2] - GxyzHistoryR[p][Gc_AN][2];
	    dgz = GLJ[p][Mc_AN][3] - GxyzHistoryR[p][Gc_AN][3];

	    dQ_dep[Mc_AN][h_AN]  += dgx*dg_dep[1] + dgy*dg_dep[2] + dgz*dg_dep[3];
	    dQ_dsig[Mc_AN][h_AN] += dgx*dg_dsig[1] + dgy*dg_dsig[2] + dgz*dg_dsig[3];

	    /*
	      printf("Gc_AN=%2d h_AN=%2d dQ_dep=%15.12f dQ_dsig=%15.12f\n",
	      Gc_AN,h_AN,dQ_dep[Mc_AN][h_AN],dQ_dsig[Mc_AN][h_AN]);
	    */

	  } /* h_AN */

	  /*
          printf("DDD1 p=%2d Gc_AN=%2d %15.12f %15.12f\n",
                       p,Gc_AN,GLJ[p][Mc_AN][1],GxyzHistoryR[p][Gc_AN][1]);
	  */

	} /* Mc_AN */
      } /* p */

      /* calculate the norm of gradients */

      my_norm_ep  = 0.0; 
      my_norm_sig = 0.0; 
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	Gc_AN = M2G[Mc_AN];    
	for (h_AN=1; h_AN<=FNAN[Gc_AN]; h_AN++){
	  my_norm_ep  +=  dQ_dep[Mc_AN][h_AN]*dQ_dep[Mc_AN][h_AN];
	  my_norm_sig += dQ_dsig[Mc_AN][h_AN]*dQ_dsig[Mc_AN][h_AN];
	}
      }

      /* MPI of Q */ 

      if (loopN==1){

        MPI_Allreduce(&my_norm_ep,  &norm_ep,  1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
        MPI_Allreduce(&my_norm_sig, &norm_sig, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

	norm_ep  = norm_ep/(double)atomnum;
	norm_sig = norm_sig/(double)atomnum;

        coe_ep  = 0.001/norm_ep; 
        if (0.1<coe_ep) coe_ep = 0.1;

        coe_sig = 0.001/norm_sig; 
        if (0.1<coe_sig) coe_sig = 0.1;

      }
      else{

        if (Q<Qold){
          coe_ep  = 1.5*coe_ep;
          coe_sig = 1.5*coe_sig;
	}
        else{
          coe_ep  = coe_ep/3.0;
          coe_sig = coe_sig/3.0;
        }
      }

      /*
      printf("loopN=%4d Q=%18.15f coe_ep=%15.12f coe_sig=%15.12f\n",loopN,Q,coe_ep,coe_sig);
      */

      /* update ep and sig */

      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

	Gc_AN = M2G[Mc_AN];    

	for (h_AN=1; h_AN<=FNAN[Gc_AN]; h_AN++){

          ep[Mc_AN][h_AN]  -= coe_ep*dQ_dep[Mc_AN][h_AN]; 
          sig[Mc_AN][h_AN] -= coe_sig*dQ_dsig[Mc_AN][h_AN]; 
	}
      }

      if (Q<0.000000001) po = 1;

      loopN++;

    } while(po==0 && loopN<200);

    /* initialize Hessian */

    for (i=1; i<=3*atomnum; i++){     
      for (j=1; j<=3*atomnum; j++){     
	Hessian[i][j] = 0.0;
      }
      Hessian[i][i] = 0.1;
    }

    /* construct the Hessian */

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];    

      for (h_AN=1; h_AN<=FNAN[Gc_AN]; h_AN++){
	Gh_AN = natn[Gc_AN][h_AN];
	Rn = ncn[Gc_AN][h_AN];
	r = Dis[Gc_AN][h_AN];
        r2 = r*r;
        r3 = r2*r; 

	g[1] = (Gxyz[Gc_AN][1] - Gxyz[Gh_AN][1] - atv[Rn][1])/r;
	g[2] = (Gxyz[Gc_AN][2] - Gxyz[Gh_AN][2] - atv[Rn][2])/r;
	g[3] = (Gxyz[Gc_AN][3] - Gxyz[Gh_AN][3] - atv[Rn][3])/r;

	sr = sig[Mc_AN][h_AN]/r;
	sr2 = sr*sr;
	sr4 = sr2*sr2;
	sr5 = sr*sr4;
	sr6 = sr2*sr4;
	sr11 = sr5*sr6;
	sr12 = sr6*sr6;

	d2f_depdr = 24.0*(sr6 - 2.0*sr12)/r;
	df_dr = d2f_depdr*ep[Mc_AN][h_AN];
        d2f_dr2 = -ep[Mc_AN][h_AN]*(168.0*sr6-624.0*sr12)/r2;

        for (i=1; i<=3; i++){
          for (j=1; j<=3; j++){
            d2r_dx1dx2[i][j] = -g[i]*g[j]/r;
	  }
          d2r_dx1dx2[i][i] += 1.0/r;
        }
        
        /* diagonal block */

        for (i=1; i<=3; i++){
          for (j=1; j<=3; j++){
	    Hessian[(Gc_AN-1)*3+i][(Gc_AN-1)*3+j] += d2r_dx1dx2[i][j]*df_dr + g[i]*g[j]*d2f_dr2;      
	  }
	}
  
        /* off-diagonal block */

        for (i=1; i<=3; i++){
          for (j=1; j<=3; j++){
	    Hessian[(Gc_AN-1)*3+i][(Gh_AN-1)*3+j] += -d2r_dx1dx2[i][j]*df_dr - g[i]*g[j]*d2f_dr2;      
	  }
	}

      } /* h_AN */
    } /* Mc_AN */       

    OpenMX_Geometry_Gather_Hessian(geometry,Hessian,0);

    /* freeing of arrays */

    for (p=0; p<(M_GDIIS_HISTORY+1); p++){
      for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){
	free(GLJ[p][Mc_AN]);
      }
      free(GLJ[p]);
    }
    free(GLJ);

    for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){
      free(ep[Mc_AN]);
    }
    free(ep);

    for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){
      free(sig[Mc_AN]);
    }
    free(sig);

    for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){
      free(dQ_dep[Mc_AN]);
    }
    free(dQ_dep);

    for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){
      free(dQ_dsig[Mc_AN]);
    }
    free(dQ_dsig);

  }

  else{

    int i,j;

    for (i=1; i<=3*atomnum; i++){     
      for (j=1; j<=3*atomnum; j++){     
	Hessian[i][j] = 0.0;
      }
      Hessian[i][i] = 1.0;
    }
  }


  /*
  for (i=1; i<=3*atomnum; i++){     
    for (j=1; j<=3*atomnum; j++){     
      printf("%8.5f ",Hessian[i][j]);
    }
    printf("\n");
  }

  MPI_Finalize();
  exit(0);
  */

}



/* added by MIZUHO for NPT-MD */
// macros for Gxyz array
#define POSITION(Gxyz)    (&Gxyz[ 0])
#define POSITION_X(Gxyz)  (Gxyz[ 1])
#define POSITION_Y(Gxyz)  (Gxyz[ 2])
#define POSITION_Z(Gxyz)  (Gxyz[ 3])

#define VELOCITY(Gxyz)    (&Gxyz[23])
#define VELOCITY_X(Gxyz)  (Gxyz[24])
#define VELOCITY_Y(Gxyz)  (Gxyz[25])
#define VELOCITY_Z(Gxyz)  (Gxyz[26])

#define GRADIENT(Gxyz)    (&Gxyz[16])
#define GRADIENT_X(Gxyz)  (Gxyz[17])
#define GRADIENT_Y(Gxyz)  (Gxyz[18])
#define GRADIENT_Z(Gxyz)  (Gxyz[19])

#define ATOMIC_MASS(Gxyz) (Gxyz[20]*(unified_atomic_mass_unit/electron_mass))

#define FIXED(atom_Fixed_XYZ)   (&atom_Fixed_XYZ[0])
#define FIXED_X(atom_Fixed_XYZ) (atom_Fixed_XYZ[1])
#define FIXED_Y(atom_Fixed_XYZ) (atom_Fixed_XYZ[2])
#define FIXED_Z(atom_Fixed_XYZ) (atom_Fixed_XYZ[3])

double getStress( int al, int be, double Volume )
{
  return (-1.0)*Stress_Tensor[3*(al-1)+(be-1)]/Volume;
}

double defaultMassBarostat( void )
{
  double MassAtom = 0.0;
  int ai, a;
  int numprocs, myid;

  MPI_Comm_rank(mpi_comm_level1,&myid);

  for( ai=1; ai<=Matomnum; ai++ ){
    a = M2G[ai];
    MassAtom += ATOMIC_MASS(Gxyz[a]);

    //printf("ABC1 a=%2d mass=%15.12f\n",a,ATOMIC_MASS(Gxyz[a]));
  }
  MPI_Allreduce( MPI_IN_PLACE, &MassAtom, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

  double MassBarostat;

  switch (MD_switch) {
  case 27: // NPT_VS_PR
  case 29: // NPT_NH_PR
    MassBarostat = 0.75*MassAtom/(M_PI*M_PI);
    break;
  case 28: // NPT_VS_WV
  case 30: // NPT_NH_WV
    MassBarostat = 0.75*MassAtom/(M_PI*M_PI)/pow(Cell_Volume,2.0/3.0)*0.01;
    break;
  default:
    printf("default MassBarostat is not supported for MD_switch %d.\n", MD_switch );
    MPI_Finalize();
    exit(1);
  }

  if (myid==Host_ID){
    printf("A default NPT.Mass.Barostat is set to %15.12f (atomic mass unit).\n",
            MassBarostat/(unified_atomic_mass_unit/electron_mass));
  }

  return MassBarostat;
}



/* Fractional atomic update shared by the NPT integrators. */
static double NPT_Kick(OpenMXMD *md, double (*coords)[4], double (*vel)[4],
    double K[4][4], double coupling[4][4], double L[4][4],
    double dt, double Wscale, double friction)
{
  OpenMXMDNPTStep step;
  step.dt=dt;
  step.friction=friction;
  for (int i=0; i<3; ++i) {
    memcpy(step.reciprocal[i],K[i+1]+1,3*sizeof(double));
    memcpy(step.coupling[i],coupling[i+1]+1,3*sizeof(double));
    for (int j=0; j<3; ++j) step.lattice_columns[i][j]=L[j+1][i+1];
    step.fractional_origin[i]=Dot_Product(K[i+1],Grid_Origin);
  }
  return OpenMX_MD_NPT_Kick(md,&step,Wscale,coords,vel);
}

static void NPT_Drift(double (*coords)[4], double (*vel)[4], double dt,
    double Wscale, double scale, double Wa[4][4])
{
  double tensor[9]={0.0};
#pragma omp parallel for schedule(static) reduction(+:tensor[:9]) if(Matomnum>=128)
  for (int local=1; local<=Matomnum; ++local) {
    const double mass=Gxyz[M2G[local]][20]*Wscale;
    for (int i=1; i<=3; ++i) {
      if (scale!=1.0) vel[local][i]*=scale;
      coords[local][i]+=vel[local][i]*dt;
    }
    for (int i=0; i<3; ++i)
      for (int j=0; j<3; ++j) tensor[3*i+j]+=vel[local][i+1]*vel[local][j+1]*mass;
  }
  MPI_Allreduce(MPI_IN_PLACE,tensor,9,MPI_DOUBLE,MPI_SUM,mpi_comm_level1);
  for (int i=0; i<3; ++i) memcpy(Wa[i+1]+1,tensor+3*i,3*sizeof(double));
}


// Velocity-Scaling and Parrinello-Rahman method
void NPT_VS_PR(OpenMXMD *md, OpenMXDynamicsState *dynamics, OpenMXThermodynamicEnergies *thermo, int iter)
{
  /* Rank-local fractional positions and velocities. */
  OpenMXMDLocalVector *coords=OpenMX_MD_NPT_Workspace(md);
  OpenMXMDLocalVector *dcoords=coords+Matomnum+1;

  // lattice dynamics
  double d2L[4][4];       // acceleration
  double dL[4][4];        // velocity

  // lattice vectors
  double L[4][4]; // real lattice vectors
  double K[4][4]; // reciprocal lattice vectors

  // misc matrices for Parrinello-Rahman method
  double Sigma[4][4], G[4][4], dG[4][4], invG[4][4], invGdG[4][4];

  // work matrices
  double Wa[4][4], Wb[4][4];

  // indexes
  int i, j, k; // abc-axes
  int al, be;  // xyz-coordinates
  int ai, a;   // local and global atom indexes

  int myid;

  MPI_Comm_rank(mpi_comm_level1,&myid);

  double dt = 41.3411*MD_TimeStep;
  double Wscale = unified_atomic_mass_unit/electron_mass;

  //---- VS PR Step 0. set dL

  for( i=1; i<=3; i++ ){
    for( al=1; al<=3; al++ ){
      dL[i][al] = tv_velocity[i][al];
    }
  }

  if ( iter == 1 ){
    if ( PresW == 0.0 ) PresW = defaultMassBarostat();
  }

  //---- VS PR Step 1. setup L from tv

  // L = (tv[1],tv[2],tv[3])

  for( i=1; i<=3; i++ ){
    for( al=1; al<=3; al++ ){
      L[i][al] = tv[i][al];
    }
  }

  //---- VS PR Step 2. setup misc matrices

  // Sigma = (L2*L3,L3*L1,L1*L2)
  Cross_Product( L[2], L[3], Sigma[1] );
  Cross_Product( L[3], L[1], Sigma[2] );
  Cross_Product( L[1], L[2], Sigma[3] );

  // Omega, Cell_Volume
  double Omega = Dot_Product( Sigma[1], L[1] );

  // K = Sigma/Omega
  for( i=1; i<=3; i++ ){
    for( al=1; al<=3; al++ ){
      K[i][al] = Sigma[i][al]/Omega;
    }
  }

  // G[i][j] = sum_al L[i][al] * L[j][al]
  for( i=1; i<=3; i++ ){
    for( j=1; j<=3; j++ ){
      G[i][j] = Dot_Product( L[i], L[j] );
    }
  }

  // dG[i][j] = sum_al dL[i][al] * L[j][al] +  L[i][al] * dL[j][al]
  for( i=1; i<=3; i++ ){
    for( j=1; j<=3; j++ ){
      dG[i][j] = Dot_Product( dL[i], L[j] ) + Dot_Product( L[i], dL[j] );
    }
  }

  // (1/G)[i][j] = sum_al Sigma[i][al] * Sigma[j][al] /Omega^2
  for( i=1; i<=3; i++ ){
    for( j=1; j<=3; j++ ){
      invG[i][j] = Dot_Product( Sigma[i], Sigma[j] )/(Omega*Omega);
    }
  }

  //  (1/G*dG)[i][j] = sum_k invG[i][k]*dG[k][j] = sum_k invG[i][k]*dG[j][k]
  for( i=1; i<=3; i++ ){
    for( j=1; j<=3; j++ ){
      invGdG[i][j] = Dot_Product( invG[i], dG[j] ); // dG is symmetric
    }
  }

  //---- VS PR Step 3. setup coords from Gxyz

  thermo->ionic_kinetic=NPT_Kick(md,coords,dcoords,K,invGdG,L,dt,Wscale,0.0);

  // calc kinetic temperature
  dynamics->temperature = 2.0/(3.0*atomnum*(kB/eV2Hartree)) * thermo->ionic_kinetic;

  // calc given temperature and scaling factor
  double scale = 1.0;
  for( i=1; i<=TempNum; i++ ){
    int iter_begin    = TempPara[i-1][1];
    int iter_end      = TempPara[i][1];

    if( iter_begin<iter && iter<=iter_end ){
      double x = (double)(iter-iter_begin)/(iter_end-iter_begin);
      dynamics->target_temperature = TempPara[i-1][2] + (TempPara[i][2] - TempPara[i-1][2])*x;
      if( iter==1 || (iter-iter_begin) % IntScale[i] ==0 && fabs(dynamics->target_temperature-dynamics->temperature)>TempTol ){
	double s = RatScale[i];
	scale = dynamics->temperature<dynamics->target_temperature*1.0e-8 ? 1.0 :
	  sqrt( (dynamics->target_temperature + (dynamics->temperature-dynamics->target_temperature)*s)/dynamics->temperature );
      }
      break;
    }
  }

  //  printf("GivenTemp %f Temp %f\n", GivenTemp, Temp );

  // update GivenPressure
  double GivenPres;
  for( i=1; i<=PreNum; i++ ){
    int iter_begin    = PrePara[i-1][1];
    int iter_end      = PrePara[i  ][1];

    if( iter_begin<iter && iter<=iter_end ){
      double x = (double)(iter-iter_begin)/(iter_end-iter_begin);
      GivenPres = PrePara[i-1][2] + (PrePara[i][2] - PrePara[i-1][2])*x;
      break;
    }
  }

  NPT_Drift(coords,dcoords,dt,Wscale,scale,Wa);

  // Wb = (sum_abcabc L[abc][xyz]*Wa[abc][abc]*L[abc][xyz]/Omega + Stress[xyz][xyz]
  //    - Pressure[xyz][xyz])/PresW
  for( al=1; al<=3; al++ ){
    for( be=1; be<=3; be++ ){
      Wb[al][be] = 0.0;
      for( i=1; i<=3; i++ ){
	for( j=1; j<=3; j++ ){
	  Wb[al][be] += L[i][al] * Wa[i][j] * L[j][be];
	}
      }
      Wb[al][be] /= Omega;
      Wb[al][be] += getStress(al,be,Omega);
      Wb[al][be] -= GivenPres*(al==be ? 1.0 : 0.0)*(double)MD_applied_pressure_flag[al-1];
      Wb[al][be] /= PresW;
    }
  }

  // restrict lattice vector
  if( LatticeRestriction == 1 ){ // cubic
    Wb[1][2] = Wb[2][1] = 0.0;
    Wb[1][3] = Wb[3][1] = 0.0;
    Wb[2][3] = Wb[3][2] = 0.0;
    Wb[1][1] = Wb[2][2] = Wb[3][3] = (Wb[1][1]+Wb[2][2]+Wb[3][3])/3.0;
  }
  else if( LatticeRestriction == 2 ){ // ortho
    Wb[1][2] = Wb[2][1] = 0.0;
    Wb[1][3] = Wb[3][1] = 0.0;
    Wb[2][3] = Wb[3][2] = 0.0;
  }

  // d2L = Wb*Sigma
  for( i=1; i<=3; i++ ){
    for( al=1; al<=3; al++ ){
      d2L[i][al] = Dot_Product( Wb[al], Sigma[i] );
    }
  }

  //---- VS PR Step 10. evolve dL and L

  // evolve dL by dt
  for( i=1; i<=3; i++ ){
    for( al=1; al<=3; al++ ){
      dL[i][al] += d2L[i][al] * dt;
      tv_velocity[i][al] = dL[i][al];
    }
  }

  // evolve L by dt
  for( i=1; i<=3; i++ ){
    for( al=1; al<=3; al++ ){
      L[i][al] += dL[i][al] * dt;
    }
  }

  //---- VS PR Step 11. update Gxyz from coords

  // update position, velocity
#pragma omp parallel for schedule(static) if(Matomnum>=128) private(a,i,al)
  for( ai=1; ai<=Matomnum; ai++ ){
    a = M2G[ai];
    for( al=1; al<=3; al++ ){

      /* store the coordinates at the previous step */

      Gxyz[a][20+al] = Gxyz[a][al];

      /* update the coordinates */

      if( !FIXED(atom_Fixed_XYZ[a])[al] ){
	POSITION(Gxyz[a])[al] = Grid_Origin[al];
	for( i=1; i<=3; i++ ){
	  POSITION(Gxyz[a])[al] += coords[ai][i]*L[i][al];
	}
      }
    }

    for( al=1; al<=3; al++ ){

      /* update the velocity */
      
      if( !FIXED(atom_Fixed_XYZ[a])[al] ){
	VELOCITY(Gxyz[a])[al] = 0.0;
	for( i=1; i<=3; i++ ){
	  VELOCITY(Gxyz[a])[al] += dcoords[ai][i]*L[i][al];
	}
      }
    }
  }

  //---- VS PR Step 12. update tv and rtv from L

  // update tv

  for( i=1; i<=3; i++ ){
    for( al=1; al<=3; al++ ){
      tv[i][al] = L[i][al];
    }
  }

  // update rtv, Cell_Volume
  // rtv = (L2*L3,L3*L1,L1*L2)/Omega*2*Pi

  Cross_Product( tv[2], tv[3], rtv[1] );
  Cross_Product( tv[3], tv[1], rtv[2] );
  Cross_Product( tv[1], tv[2], rtv[3] );
  Cell_Volume = fabs(Dot_Product( rtv[1], tv[1] ));
  for( i=1; i<=3; i++ ){
    for( al=1; al<=3; al++ ){
      rtv[i][al] = rtv[i][al]/Cell_Volume*2*M_PI;
    }
  }

  //---- VS PR Step 13. calc Hamiltonian as the conserved quantity
  double TrdLdL = 0.0;
  for( i=1; i<=3; i++ ){
    TrdLdL += Dot_Product( dL[i], dL[i] );
  }
  double Ubaro = 0.5*PresW*TrdLdL;
  double Wbaro = GivenPres*Omega;
  double Ham = thermo->total + thermo->ionic_kinetic + Ubaro + Wbaro;

  if ( myid == 0 ){

    static int first = 1;
    static FILE* fptr;
    static char fileE[YOUSO10];

    if ( first ){
      first = 0;
      
      sprintf(fileE,"%s%s.npt.ene",filepath,filename);
      
      fptr = fopen(fileE,"r");

      if (fptr==NULL) fptr = fopen(fileE,"w");
      else            fptr = fopen(fileE,"a");

      fprintf( fptr, "# NPT_VS_PR TimeStep=%e (fs) MassBarostat=%e (atomic mass unit)\n", 
               MD_TimeStep, PresW/(unified_atomic_mass_unit/electron_mass) );
      fprintf( fptr, "#  1:iter 2:time  3:Volume 4:Temp    5:Stress   6:Utot   7:Ukc    8:Ubaro  9:Wbaro    10:Hamiltonian\n");
    }
    else{
      fptr = fopen(fileE,"a");
    }  

    fprintf( fptr, "%4d %f %f %f %f %f %f %f %f %f\n", iter, MD_TimeStep*(double)iter,
	     fabs(Omega), dynamics->temperature,
             (getStress(1,1,Omega)+getStress(2,2,Omega)+getStress(3,3,Omega))*29.42*1000/3.0, /* 1/(0.3398827*0.0001) */
	     thermo->total, thermo->ionic_kinetic, Ubaro, Wbaro, Ham );

    fflush(fptr);
    fclose(fptr);
  }

} // NPT_VS_PR





// Velocity-Scaling and Wentzcovitch method
void NPT_VS_WV(OpenMXMD *md, OpenMXDynamicsState *dynamics, OpenMXThermodynamicEnergies *thermo, int iter)
{
  /* Rank-local fractional positions and velocities. */
  OpenMXMDLocalVector *coords=OpenMX_MD_NPT_Workspace(md);
  OpenMXMDLocalVector *dcoords=coords+Matomnum+1;

  // lattice dynamics
  double d2L[4][4]; // acceleration
  double dL[4][4]; // velocity, preserved for next calls

  // lattice vectors
  double L[4][4]; // real lattice vectors
  double K[4][4]; // reciprocal lattice vectors

  // misc matrices for Wentzcovitch method
  double Sigma[4][4], G[4][4], dG[4][4], invG[4][4], invGdG[4][4];
  double E[4][4], F[4][4], invF[4][4], dF[4][4];
  double SigmaP[4][4][4][4], FP[4][4][4][4];
  double TrEFP[4][4], dSigma[4][4], dLdF[4][4];
  static double F0[4][4]; // F at the initial step

  // work matrices
  double Wa[4][4], Wb[4][4];

  // indexes
  int i, j, k; // abc-axes
  int al, be;  // xyz-coordinates
  int ai, a;   // local and global atom indexes

  int myid;

  MPI_Comm_rank(mpi_comm_level1,&myid);

  double dt = 41.3411*MD_TimeStep;
  double Wscale = unified_atomic_mass_unit/electron_mass;

  //---- VS WV Step 0. initialize dL[i][al] = 0

  for( i=1; i<=3; i++ ){
    for( al=1; al<=3; al++ ){
      dL[i][al] = tv_velocity[i][al];
    }
  }

  if ( iter == 1 ){
    if ( PresW == 0.0 ) PresW = defaultMassBarostat();
  }

  //---- VS WV Step 1. setup L from tv

  // L = (tv[1],tv[2],tv[3])
  for( i=1; i<=3; i++ ){
    for( al=1; al<=3; al++ ){
      L[i][al] = tv[i][al];
    }
  }

  //---- VS WV Step 2. setup misc matrices

  // Sigma = (L2*L3,L3*L1,L1*L2)
  Cross_Product( L[2], L[3], Sigma[1] );
  Cross_Product( L[3], L[1], Sigma[2] );
  Cross_Product( L[1], L[2], Sigma[3] );

  // Omega, Cell_Volume
  double Omega = Dot_Product( Sigma[1], L[1] );

  // K = Sigma/Omega
  for( i=1; i<=3; i++ ){
    for( al=1; al<=3; al++ ){
      K[i][al] = Sigma[i][al]/Omega;
    }
  }

  // G[i][j] = sum_al L[i][al] * L[j][al]
  for( i=1; i<=3; i++ ){
    for( j=1; j<=3; j++ ){
      G[i][j] = Dot_Product( L[i], L[j] );
    }
  }

  // dG[i][j] = sum_al dL[i][al] * L[j][al] +  L[i][al] * dL[j][al]
  for( i=1; i<=3; i++ ){
    for( j=1; j<=3; j++ ){
      dG[i][j] = Dot_Product( dL[i], L[j] ) + Dot_Product( L[i], dL[j] );
    }
  }

  // (1/G)[i][j] = sum_al Sigma[i][al] * Sigma[j][al] /Omega^2
  for( i=1; i<=3; i++ ){
    for( j=1; j<=3; j++ ){
      invG[i][j] = Dot_Product( Sigma[i], Sigma[j] )/(Omega*Omega);
    }
  }

  //  (1/G*dG)[i][j] = sum_k invG[i][k]*dG[k][j] = sum_k invG[i][k]*dG[j][k]
  for( i=1; i<=3; i++ ){
    for( j=1; j<=3; j++ ){
      invGdG[i][j] = Dot_Product( invG[i], dG[j] ); // dG is symmetric
    }
  }

  // E[i][j] = sum_al dL[i][al] * dL[j][al]
  for( i=1; i<=3; i++ ){
    for( j=1; j<=3; j++ ){
      E[i][j] = Dot_Product( dL[i], dL[j] );
    }
  }

  // F[i][j] = sum_al L[i][al] * L[j][al]
  for( i=1; i<=3; i++ ){
    for( j=1; j<=3; j++ ){
      F[i][j] = Dot_Product( Sigma[i], Sigma[j] );
      if( iter==1 ){
	F0[i][j] = F[i][j];
        NPT_WV_F0[i][j] = F0[i][j]; 
      }
      else{
	F0[i][j] = NPT_WV_F0[i][j];
      }
    }
  }

  // invF[i][j] = sum_al L[i][al] * L[j][al] / Omega^2
  for( i=1; i<=3; i++ ){
    for( j=1; j<=3; j++ ){
      invF[i][j] = Dot_Product( L[i], L[j] )/(Omega*Omega);
    }
  }

  // SigmaP
  for( i=1; i<=3; i++ ){
    for( al=1; al<=3; al++ ){
      for( j=1; j<=3; j++ ){
	for( be=1; be<=3; be++ ){
	  SigmaP[i][al][j][be] =
	    (Sigma[i][al]*Sigma[j][be] - Sigma[j][al]*Sigma[i][be])/Omega;
	}
      }
    }
  }

  // dSigma[i][al] = sum_j sum_be SigmaP[i][al][j][be] * dL[j][be]
  for( i=1; i<=3; i++ ){
    for( al=1; al<=3; al++ ){
      dSigma[i][al] = 0.0;
      for( j=1; j<=3; j++ ){
	for( be=1; be<=3; be++ ){
	  dSigma[i][al] += SigmaP[i][al][j][be] * dL[j][be];
	}
      }
    }
  }

  // dF[i][j] = sum_al dSigma[i][al] * Sigma[j][al] + Sigma[i][al] * dSigma[j][al]
  for( i=1; i<=3; i++ ){
    for( j=1; j<=3; j++ ){
      dF[i][j] = Dot_Product( dSigma[i], Sigma[j] )
	+ Dot_Product( dSigma[i], Sigma[j] );
    }
  }

  // FP[i][al][j][k] = sum_be SigmaP[i][al][j][be] * Sigma[k][be] + Sigma[j][be]*SigmaP[i][al][k][be]
  for( i=1; i<=3; i++ ){
    for( al=1; al<=3; al++ ){
      for( j=1; j<=3; j++ ){
	for( k=1; k<=3; k++ ){
	  FP[i][al][j][k] = Dot_Product( SigmaP[i][al][j], Sigma[k] )
	    + Dot_Product( Sigma[j], SigmaP[i][al][k] );
	}
      }
    }
  }

  // TrEFP[i][al] = sum_jk E[j][k] * FP[i][al][j][k]
  for( i=1; i<=3; i++ ){
    for( al=1; al<=3; al++ ){
      TrEFP[i][al] = 0.0;
      for( j=1; j<=3; j++ ){
	for( k=1; k<=3; k++ ){
	  TrEFP[i][al] += E[j][k] * FP[i][al][j][k];
	}
      }
    }
  }

  // dLdF[i][al] = sum_j dF[i][j] * dL[j][al]
  for( i=1; i<=3; i++ ){
    for( al=1; al<=3; al++ ){
      dLdF[i][al] = 0.0;
      for( j=1; j<=3; j++ ){
	dLdF[i][al] += dF[i][j] * dL[j][al];
      }
    }
  }

  //---- VS WV Step 3. setup coords from Gxyz

  thermo->ionic_kinetic=NPT_Kick(md,coords,dcoords,K,invGdG,L,dt,Wscale,0.0);

  // calc kinetic temperature
  dynamics->temperature = 2.0/(3.0*atomnum*(kB/eV2Hartree)) * thermo->ionic_kinetic;

  // calc given temperature and scaling factor
  double scale = 1.0;
  for( i=1; i<=TempNum; i++ ){
    int iter_begin    = TempPara[i-1][1];
    int iter_end      = TempPara[i][1];

    if( iter_begin<iter && iter<=iter_end ){
      double x = (double)(iter-iter_begin)/(iter_end-iter_begin);
      dynamics->target_temperature = TempPara[i-1][2] + (TempPara[i][2] - TempPara[i-1][2])*x;
      if( iter==1 || (iter-iter_begin) % IntScale[i] ==0 && fabs(dynamics->target_temperature-dynamics->temperature)>TempTol ){
	double s = RatScale[i];
	scale = dynamics->temperature<dynamics->target_temperature*1.0e-8 ? 1.0 :
	  sqrt( (dynamics->target_temperature + (dynamics->temperature-dynamics->target_temperature)*s)/dynamics->temperature );
      }
      break;
    }
  }

  //  printf("GivenTemp %f Temp %f\n", GivenTemp, Temp );

  // update GivenPressure
  double GivenPres;
  for( i=1; i<=PreNum; i++ ){
    int iter_begin    = PrePara[i-1][1];
    int iter_end      = PrePara[i  ][1];

    if( iter_begin<iter && iter<=iter_end ){
      double x = (double)(iter-iter_begin)/(iter_end-iter_begin);
      GivenPres = PrePara[i-1][2] + (PrePara[i][2] - PrePara[i-1][2])*x;
      break;
    }
  }

  NPT_Drift(coords,dcoords,dt,Wscale,scale,Wa);

  // Wb = (sum_abcabc L[abc][xyz]*Wa[abc][abc]*L[abc][xyz]/Omega + Stress[xyz][xyz]
  //    - Pressure[xyz][xyz])/PresW
  for( al=1; al<=3; al++ ){
    for( be=1; be<=3; be++ ){
      Wb[al][be] = 0.0;
      for( i=1; i<=3; i++ ){
	for( j=1; j<=3; j++ ){
	  Wb[al][be] += L[i][al] * Wa[i][j] * L[j][be];
	}
      }
      Wb[al][be] /= Omega;
      Wb[al][be] += getStress(al,be,Omega);
      Wb[al][be] -= GivenPres*(al==be ? 1.0 : 0.0)*(double)MD_applied_pressure_flag[al-1];
      Wb[al][be] /= PresW;
    }
  }

  // restrict lattice vector
  if( LatticeRestriction == 1 ){ // cubic
    Wb[1][2] = Wb[2][1] = 0.0;
    Wb[1][3] = Wb[3][1] = 0.0;
    Wb[2][3] = Wb[3][2] = 0.0;
    Wb[1][1] = Wb[2][2] = Wb[3][3] = (Wb[1][1]+Wb[2][2]+Wb[3][3])/3.0;
  }
  else if( LatticeRestriction == 2 ){ // ortho
    Wb[1][2] = Wb[2][1] = 0.0;
    Wb[1][3] = Wb[3][1] = 0.0;
    Wb[2][3] = Wb[3][2] = 0.0;
  }

  // d2L = Wb*Sigma
  for( i=1; i<=3; i++ ){
    for( al=1; al<=3; al++ ){
      d2L[i][al] = Dot_Product( Wb[al], Sigma[i] );
    }
  }

  // Wa = d2L + 0.5*TrEFP - dLdF
  for( i=1; i<=3; i++ ){
    for( al=1; al<=3; al++ ){
      Wa[i][al] = d2L[i][al] + 0.5*TrEFP[i][al] - dLdF[i][al];
    }
  }

  // d2L[i][al] = sum_k Wa[k][al] * invF[i][k], Wentzcovitch
  for( i=1; i<=3; i++ ){
    for( al=1; al<=3; al++ ){
      d2L[i][al] = 0.0;
      for( k=1; k<=3; k++ ){
	d2L[i][al] += Wa[k][al] * invF[i][k];
      }
    }
  }

  //---- VS WV Step 10. evolve dL and L

  // evolve dL by dt
  for( i=1; i<=3; i++ ){
    for( al=1; al<=3; al++ ){
      dL[i][al] += d2L[i][al] * dt;
      tv_velocity[i][al] = dL[i][al];
    }
  }

  // evolve L by dt
  for( i=1; i<=3; i++ ){
    for( al=1; al<=3; al++ ){
      L[i][al] += dL[i][al] * dt;
    }
  }

  //---- VS WV Step 11. update Gxyz from coords

  // update position, velocity
#pragma omp parallel for schedule(static) if(Matomnum>=128) private(a,i,al)
  for( ai=1; ai<=Matomnum; ai++ ){
    a = M2G[ai];
    for( al=1; al<=3; al++ ){

      /* store the coordinates at the previous step */

      Gxyz[a][20+al] = Gxyz[a][al];

      /* update the coordinates */

      if( !FIXED(atom_Fixed_XYZ[a])[al] ){
	POSITION(Gxyz[a])[al] = Grid_Origin[al];
	for( i=1; i<=3; i++ ){
	  POSITION(Gxyz[a])[al] += coords[ai][i]*L[i][al];
	}
      }
    }
    for( al=1; al<=3; al++ ){
      if( !FIXED(atom_Fixed_XYZ[a])[al] ){
	VELOCITY(Gxyz[a])[al] = 0.0;
	for( i=1; i<=3; i++ ){
	  VELOCITY(Gxyz[a])[al] += dcoords[ai][i]*L[i][al];
	}
      }
    }
  }

  //---- VS WV Step 12. update tv from L

  // update tv
  for( i=1; i<=3; i++ ){
    for( al=1; al<=3; al++ ){
      tv[i][al] = L[i][al];
    }
  }

  // update rtv, Cell_Volume
  // rtv = (L2*L3,L3*L1,L1*L2)/Omega*2*Pi

  Cross_Product( tv[2], tv[3], rtv[1] );
  Cross_Product( tv[3], tv[1], rtv[2] );
  Cross_Product( tv[1], tv[2], rtv[3] );
  Cell_Volume = fabs(Dot_Product( rtv[1], tv[1] ));
  for( i=1; i<=3; i++ ){
    for( al=1; al<=3; al++ ){
      rtv[i][al] = rtv[i][al]/Cell_Volume*2*M_PI;
    }
  }

  //---- VS WV Step 13. calc Hamiltonian as the conserved quantity
  double TrdLF0dL = 0.0;
  for( i=1; i<=3; i++ ){
    for( j=1; j<=3; j++ ){
      TrdLF0dL += Dot_Product( dL[i], dL[i] ) * F0[i][j];
    }
  }

  double Ubaro = 0.5*PresW*TrdLF0dL;
  double Wbaro = GivenPres*Omega;
  double Ham = thermo->total + thermo->ionic_kinetic + Ubaro + Wbaro;

  if ( myid == 0 ){

    static int first = 1;
    static FILE* fptr;
    static char fileE[YOUSO10];

    if ( first ){
      first = 0;
      
      sprintf(fileE,"%s%s.npt.ene",filepath,filename);
      
      fptr = fopen(fileE,"r");

      if (fptr==NULL) fptr = fopen(fileE,"w");
      else            fptr = fopen(fileE,"a");

      fprintf( fptr, "# NPT_VS_WV TimeStep=%e (fs) MassBarostat=%e (atomic mass unit)\n", 
               MD_TimeStep, PresW/(unified_atomic_mass_unit/electron_mass) );
      fprintf( fptr, "#  1:iter 2:time  3:Volume 4:Temp    5:Stress   6:Utot   7:Ukc    8:Ubaro  9:Wbaro    10:Hamiltonian\n");
    }
    else{
      fptr = fopen(fileE,"a");
    }  

    fprintf( fptr, "%4d %f %f %f %f %f %f %f %f %f\n", iter, MD_TimeStep*(double)iter,
	     fabs(Omega), dynamics->temperature,
             (getStress(1,1,Omega)+getStress(2,2,Omega)+getStress(3,3,Omega))*29.42*1000/3.0, /* 1/(0.3398827*0.0001) */
	     thermo->total, thermo->ionic_kinetic, Ubaro, Wbaro, Ham );

    fflush(fptr);
    fclose(fptr);
  }

} // NPT_VS_WV



// Nose-Hoover and Parrinello-Rahman method
void NPT_NH_PR(OpenMXMD *md, OpenMXDynamicsState *dynamics, OpenMXThermodynamicEnergies *thermo,  int iter )
{
  /* Rank-local fractional positions and velocities. */
  OpenMXMDLocalVector *coords=OpenMX_MD_NPT_Workspace(md);
  OpenMXMDLocalVector *dcoords=coords+Matomnum+1;

  // lattice dynamics
  double d2L[4][4]; // acceleration
  static double dL[4][4]; // velocity, preserved for next calls

  // lattice vectors
  double L[4][4]; // real lattice vectors
  double K[4][4]; // reciprocal lattice vectors

  // misc matrices for Parrinello-Rahman method
  double Sigma[4][4], G[4][4], dG[4][4], invG[4][4], invGdG[4][4];

  // work matrices
  double Wa[4][4], Wb[4][4];

  // indexes
  int i, j, k; // abc-axes
  int al, be; // xyz-coordinates
  int ai, a; // local and global atom indexes

  int myid;

  MPI_Comm_rank(mpi_comm_level1,&myid);

  double dt = 41.3411*MD_TimeStep;
  double Wscale = unified_atomic_mass_unit/electron_mass;

  //---- NH PR Step 0. initialize dL[i][al] = 0

  for( i=1; i<=3; i++ ){
    for( al=1; al<=3; al++ ){
      dL[i][al] = tv_velocity[i][al];
    }
  }

  if( iter == 1 ){
    if( PresW == 0.0 ) PresW = defaultMassBarostat();
  }

  // initialize thermostat

  if( iter == 1 ){
    dynamics->nose_next_friction = 0.0;
    dynamics->nose_friction = 0.0;
    dynamics->nose_coordinate = 0.0;
  }

  //---- NH PR Step 1. setup L from tv

  // L = (tv[1],tv[2],tv[3])
  for( i=1; i<=3; i++ ){
    for( al=1; al<=3; al++ ){
      L[i][al] = tv[i][al];
    }
  }

  //---- NH PR Step 2. setup misc matrices

  // Sigma = (L2*L3,L3*L1,L1*L2)
  Cross_Product( L[2], L[3], Sigma[1] );
  Cross_Product( L[3], L[1], Sigma[2] );
  Cross_Product( L[1], L[2], Sigma[3] );

  // Omega, Cell_Volume
  double Omega = Dot_Product( Sigma[1], L[1] );

  // K = Sigma/Omega
  for( i=1; i<=3; i++ ){
    for( al=1; al<=3; al++ ){
      K[i][al] = Sigma[i][al]/Omega;
    }
  }

  // G[i][j] = sum_al L[i][al] * L[j][al]
  for( i=1; i<=3; i++ ){
    for( j=1; j<=3; j++ ){
      G[i][j] = Dot_Product( L[i], L[j] );
    }
  }

  // dG[i][j] = sum_al dL[i][al] * L[j][al] +  L[i][al] * dL[j][al]
  for( i=1; i<=3; i++ ){
    for( j=1; j<=3; j++ ){
      dG[i][j] = Dot_Product( dL[i], L[j] ) + Dot_Product( L[i], dL[j] );
    }
  }

  // (1/G)[i][j] = sum_al Sigma[i][al] * Sigma[j][al] /Omega^2
  for( i=1; i<=3; i++ ){
    for( j=1; j<=3; j++ ){
      invG[i][j] = Dot_Product( Sigma[i], Sigma[j] )/(Omega*Omega);
    }
  }

  //  (1/G*dG)[i][j] = sum_k invG[i][k]*dG[k][j] = sum_k invG[i][k]*dG[j][k]
  for( i=1; i<=3; i++ ){
    for( j=1; j<=3; j++ ){
      invGdG[i][j] = Dot_Product( invG[i], dG[j] ); // dG is symmetric
    }
  }

  //---- NH PR Step 3. setup coords from Gxyz

  thermo->ionic_kinetic=NPT_Kick(md,coords,dcoords,K,invGdG,L,dt,Wscale,dynamics->nose_friction);

  // calc kinetic temperature
  dynamics->temperature = 2.0/(3.0*atomnum*(kB/eV2Hartree)) * thermo->ionic_kinetic;

  // calc given temperature
  for( i=1; i<=TempNum; i++ ){
    int iter_begin    = TempPara[i-1][1];
    int iter_end      = TempPara[i][1];

    if( iter_begin<iter && iter<=iter_end ){
      double x = (double)(iter-iter_begin)/(iter_end-iter_begin);
      dynamics->target_temperature = TempPara[i-1][2] + (TempPara[i][2] - TempPara[i-1][2])*x;
      break;
    }
  }

  //---- NH PR Step 7. evolve Nose-Hoover thermostat

  double NH_dzeta = (double)atomnum*(kB/eV2Hartree)*(dynamics->temperature-dynamics->target_temperature)/(TempQ*Wscale);

  dynamics->nose_next_friction = dynamics->nose_friction + NH_dzeta * dt;
  dynamics->nose_friction = dynamics->nose_next_friction;
  dynamics->nose_coordinate = dynamics->nose_coordinate + dynamics->nose_friction*dt;

  // update GivenPressure
  double GivenPres;
  for( i=1; i<=PreNum; i++ ){
    int iter_begin    = PrePara[i-1][1];
    int iter_end      = PrePara[i  ][1];

    if( iter_begin<iter && iter<=iter_end ){
      double x = (double)(iter-iter_begin)/(iter_end-iter_begin);
      GivenPres = PrePara[i-1][2] + (PrePara[i][2] - PrePara[i-1][2])*x;
      break;
    }
  }

  NPT_Drift(coords,dcoords,dt,Wscale,1.0,Wa);

  // Wb = (sum_abcabc L[abc][xyz]*Wa[abc][abc]*L[abc][xyz]/Omega + Stress[xyz][xyz]
  //    - Pressure[xyz][xyz])/PresW
  for( al=1; al<=3; al++ ){
    for( be=1; be<=3; be++ ){
      Wb[al][be] = 0.0;
      for( i=1; i<=3; i++ ){
	for( j=1; j<=3; j++ ){
	  Wb[al][be] += L[i][al] * Wa[i][j] * L[j][be];
	}
      }
      Wb[al][be] /= Omega;
      Wb[al][be] += getStress(al,be,Omega);
      Wb[al][be] -= GivenPres*(al==be ? 1.0 : 0.0)*(double)MD_applied_pressure_flag[al-1];
      Wb[al][be] /= PresW;
    }
  }

  // restrict lattice vector
  if( LatticeRestriction == 1 ){ // cubic
    Wb[1][2] = Wb[2][1] = 0.0;
    Wb[1][3] = Wb[3][1] = 0.0;
    Wb[2][3] = Wb[3][2] = 0.0;
    Wb[1][1] = Wb[2][2] = Wb[3][3] = (Wb[1][1]+Wb[2][2]+Wb[3][3])/3.0;
  }
  else if( LatticeRestriction == 2 ){ // ortho
    Wb[1][2] = Wb[2][1] = 0.0;
    Wb[1][3] = Wb[3][1] = 0.0;
    Wb[2][3] = Wb[3][2] = 0.0;
  }

  // d2L = Wb*Sigma
  for( i=1; i<=3; i++ ){
    for( al=1; al<=3; al++ ){
      d2L[i][al] = Dot_Product( Wb[al], Sigma[i] );
    }
  }

  //---- NH PR Step 10. evolve dL and L

  // evolve dL by dt
  for( i=1; i<=3; i++ ){
    for( al=1; al<=3; al++ ){
      dL[i][al] += d2L[i][al] * dt;
      tv_velocity[i][al] = dL[i][al];
    }
  }

  // evolve L by dt
  for( i=1; i<=3; i++ ){
    for( al=1; al<=3; al++ ){
      L[i][al] += dL[i][al] * dt;
    }
  }

  //---- NH PR Step 11. update Gxyz from coords

  // update position, velocity
#pragma omp parallel for schedule(static) if(Matomnum>=128) private(a,i,al)
  for( ai=1; ai<=Matomnum; ai++ ){
    a = M2G[ai];
    for( al=1; al<=3; al++ ){

      /* store the coordinates at the previous step */

      Gxyz[a][20+al] = Gxyz[a][al];

      /* update the coordinates */

      if( !FIXED(atom_Fixed_XYZ[a])[al] ){
	POSITION(Gxyz[a])[al] = Grid_Origin[al];
	for( i=1; i<=3; i++ ){
	  POSITION(Gxyz[a])[al] += coords[ai][i]*L[i][al];
	}
      }
    }

    for( al=1; al<=3; al++ ){

      /* update the velocity */

      if( !FIXED(atom_Fixed_XYZ[a])[al] ){
	VELOCITY(Gxyz[a])[al] = 0.0;
	for( i=1; i<=3; i++ ){
	  VELOCITY(Gxyz[a])[al] += dcoords[ai][i]*L[i][al];
	}
      }
    }
  }

  //---- NH PR Step 12. update tv and rtv from L

  // update tv

  for( i=1; i<=3; i++ ){
    for( al=1; al<=3; al++ ){
      tv[i][al] = L[i][al];
    }
  }

  // update rtv, Cell_Volume
  // rtv = (L2*L3,L3*L1,L1*L2)/Omega*2*Pi

  Cross_Product( tv[2], tv[3], rtv[1] );
  Cross_Product( tv[3], tv[1], rtv[2] );
  Cross_Product( tv[1], tv[2], rtv[3] );
  Cell_Volume = fabs(Dot_Product( rtv[1], tv[1] ));
  for( i=1; i<=3; i++ ){
    for( al=1; al<=3; al++ ){
      rtv[i][al] = rtv[i][al]/Cell_Volume*2*M_PI;
    }
  }

  //---- NH PR Step 13. calc Hamiltonian as the conserved quantity
  double TrdLdL = 0.0;
  for( i=1; i<=3; i++ ){
    TrdLdL += Dot_Product( dL[i], dL[i] );

  }

  double Uthermo = 0.5*dynamics->nose_friction*dynamics->nose_friction*TempQ*Wscale
                 + 3.0*(double)atomnum*(kB/eV2Hartree)*dynamics->target_temperature*dynamics->nose_coordinate;

  double Ubaro = 0.5*PresW*TrdLdL;
  double Wbaro = GivenPres*Omega;
  double Ham = thermo->total + thermo->ionic_kinetic + Uthermo + Ubaro + Wbaro;

  if ( myid == 0 ){

    static int first = 1;
    static FILE* fptr;
    static char fileE[YOUSO10];

    if ( first ){
      first = 0;
      
      sprintf(fileE,"%s%s.npt.ene",filepath,filename);
      
      fptr = fopen(fileE,"r");

      if (fptr==NULL) fptr = fopen(fileE,"w");
      else            fptr = fopen(fileE,"a");

      fprintf( fptr, "# NPT_NH_PR TimeStep=%e MassBarostat=%e\n", dt, PresW );
      fprintf( fptr, "#  1:iter 2:Volume 3:Temp    4:Stress   5:Utot   6:Ukc    7:Ubaro  8:Wbaro    9:Hamiltonian\n");
    }
    else{
      fptr = fopen(fileE,"a");
    }  

    fprintf( fptr, "%4d %f %f %+f %+f %f %f %f %+f\n", iter,
	     fabs(Omega), dynamics->temperature, (getStress(1,1,Omega)+getStress(2,2,Omega)+getStress(3,3,Omega))*29.42*1000/3.0,
	     thermo->total, thermo->ionic_kinetic, Ubaro, Wbaro, Ham );

    fflush(fptr);
    fclose(fptr);
  }

} // NPT_NH_PR




// Nose-Hoover and Wentzcovitch method
void NPT_NH_WV(OpenMXMD *md, OpenMXDynamicsState *dynamics, OpenMXThermodynamicEnergies *thermo,  int iter )
{
  /* Rank-local fractional positions and velocities. */
  OpenMXMDLocalVector *coords=OpenMX_MD_NPT_Workspace(md);
  OpenMXMDLocalVector *dcoords=coords+Matomnum+1;

  // lattice dynamics
  double d2L[4][4]; // acceleration
  static double dL[4][4]; // velocity, preserved for next calls

  // lattice vectors
  double L[4][4]; // real lattice vectors
  double K[4][4]; // reciprocal lattice vectors

  // misc matrices for Wentzcovitch method
  double Sigma[4][4], G[4][4], dG[4][4], invG[4][4], invGdG[4][4];
  double E[4][4], F[4][4], invF[4][4], dF[4][4];
  double SigmaP[4][4][4][4], FP[4][4][4][4];
  double TrEFP[4][4], dSigma[4][4], dLdF[4][4];
  static double F0[4][4]; // F at the initial step

  // work matrices
  double Wa[4][4], Wb[4][4];

  // indexes
  int i, j, k; // abc-axes
  int al, be; // xyz-coordinates
  int ai, a; // local and global atom indexes

  int myid;

  MPI_Comm_rank(mpi_comm_level1,&myid);

  double dt = 41.3411*MD_TimeStep;
  double Wscale = unified_atomic_mass_unit/electron_mass;

  //---- NH WV Step 0. initialize dL[i][al] = 0

  for( i=1; i<=3; i++ ){
    for( al=1; al<=3; al++ ){
      dL[i][al] = tv_velocity[i][al];
    }
  }

  if( iter == 1 ){
    if( PresW == 0.0 ) PresW = defaultMassBarostat();
  }

  // initialize thermostat

  if( iter == 1 ){
    dynamics->nose_next_friction = 0.0;
    dynamics->nose_friction = 0.0;
    dynamics->nose_coordinate = 0.0;
  }

  //---- NH WV Step 1. setup L from tv

  // L = (tv[1],tv[2],tv[3])
  for( i=1; i<=3; i++ ){
    for( al=1; al<=3; al++ ){
      L[i][al] = tv[i][al];
    }
  }

  //---- NH WV Step 2. setup misc matrices

  // Sigma = (L2*L3,L3*L1,L1*L2)
  Cross_Product( L[2], L[3], Sigma[1] );
  Cross_Product( L[3], L[1], Sigma[2] );
  Cross_Product( L[1], L[2], Sigma[3] );

  // Omega, Cell_Volume
  double Omega = Dot_Product( Sigma[1], L[1] );

  // K = Sigma/Omega
  for( i=1; i<=3; i++ ){
    for( al=1; al<=3; al++ ){
      K[i][al] = Sigma[i][al]/Omega;
    }
  }

  // G[i][j] = sum_al L[i][al] * L[j][al]
  for( i=1; i<=3; i++ ){
    for( j=1; j<=3; j++ ){
      G[i][j] = Dot_Product( L[i], L[j] );
    }
  }

  // dG[i][j] = sum_al dL[i][al] * L[j][al] +  L[i][al] * dL[j][al]
  for( i=1; i<=3; i++ ){
    for( j=1; j<=3; j++ ){
      dG[i][j] = Dot_Product( dL[i], L[j] ) + Dot_Product( L[i], dL[j] );
    }
  }

  // (1/G)[i][j] = sum_al Sigma[i][al] * Sigma[j][al] /Omega^2
  for( i=1; i<=3; i++ ){
    for( j=1; j<=3; j++ ){
      invG[i][j] = Dot_Product( Sigma[i], Sigma[j] )/(Omega*Omega);
    }
  }

  //  (1/G*dG)[i][j] = sum_k invG[i][k]*dG[k][j] = sum_k invG[i][k]*dG[j][k]
  for( i=1; i<=3; i++ ){
    for( j=1; j<=3; j++ ){
      invGdG[i][j] = Dot_Product( invG[i], dG[j] ); // dG is symmetric
    }
  }

  // E[i][j] = sum_al dL[i][al] * dL[j][al]
  for( i=1; i<=3; i++ ){
    for( j=1; j<=3; j++ ){
      E[i][j] = Dot_Product( dL[i], dL[j] );
    }
  }

  // F[i][j] = sum_al L[i][al] * L[j][al]
  for( i=1; i<=3; i++ ){
    for( j=1; j<=3; j++ ){
      F[i][j] = Dot_Product( Sigma[i], Sigma[j] );
      if( iter==1 ){
	F0[i][j] = F[i][j];
      }
    }
  }

  // invF[i][j] = sum_al L[i][al] * L[j][al] / Omega^2
  for( i=1; i<=3; i++ ){
    for( j=1; j<=3; j++ ){
      invF[i][j] = Dot_Product( L[i], L[j] )/(Omega*Omega);
    }
  }

  // SigmaP
  for( i=1; i<=3; i++ ){
    for( al=1; al<=3; al++ ){
      for( j=1; j<=3; j++ ){
	for( be=1; be<=3; be++ ){
	  SigmaP[i][al][j][be] =
	    (Sigma[i][al]*Sigma[j][be] - Sigma[j][al]*Sigma[i][be])/Omega;
	}
      }
    }
  }

  // dSigma[i][al] = sum_j sum_be SigmaP[i][al][j][be] * dL[j][be]
  for( i=1; i<=3; i++ ){
    for( al=1; al<=3; al++ ){
      dSigma[i][al] = 0.0;
      for( j=1; j<=3; j++ ){
	for( be=1; be<=3; be++ ){
	  dSigma[i][al] += SigmaP[i][al][j][be] * dL[j][be];
	}
      }
    }
  }

  // dF[i][j] = sum_al dSigma[i][al] * Sigma[j][al] + Sigma[i][al] * dSigma[j][al]
  for( i=1; i<=3; i++ ){
    for( j=1; j<=3; j++ ){
      dF[i][j] = Dot_Product( dSigma[i], Sigma[j] )
	+ Dot_Product( dSigma[i], Sigma[j] );
    }
  }

  // FP[i][al][j][k] = sum_be SigmaP[i][al][j][be] * Sigma[k][be] + Sigma[j][be]*SigmaP[i][al][k][be]
  for( i=1; i<=3; i++ ){
    for( al=1; al<=3; al++ ){
      for( j=1; j<=3; j++ ){
	for( k=1; k<=3; k++ ){
	  FP[i][al][j][k] = Dot_Product( SigmaP[i][al][j], Sigma[k] )
	    + Dot_Product( Sigma[j], SigmaP[i][al][k] );
	}
      }
    }
  }

  // TrEFP[i][al] = sum_jk E[j][k] * FP[i][al][j][k]
  for( i=1; i<=3; i++ ){
    for( al=1; al<=3; al++ ){
      TrEFP[i][al] = 0.0;
      for( j=1; j<=3; j++ ){
	for( k=1; k<=3; k++ ){
	  TrEFP[i][al] += E[j][k] * FP[i][al][j][k];
	}
      }
    }
  }

  // dLdF[i][al] = sum_j dF[i][j] * dL[j][al]
  for( i=1; i<=3; i++ ){
    for( al=1; al<=3; al++ ){
      dLdF[i][al] = 0.0;
      for( j=1; j<=3; j++ ){
	dLdF[i][al] += dF[i][j] * dL[j][al];
      }
    }
  }

  //---- NH WV Step 3. setup coords from Gxyz

  thermo->ionic_kinetic=NPT_Kick(md,coords,dcoords,K,invGdG,L,dt,Wscale,dynamics->nose_friction);

  // calc kinetic temperature
  dynamics->temperature = 2.0/(3.0*atomnum*(kB/eV2Hartree)) * thermo->ionic_kinetic;

  // calc given temperature
  for( i=1; i<=TempNum; i++ ){
    int iter_begin    = TempPara[i-1][1];
    int iter_end      = TempPara[i][1];

    if( iter_begin<iter && iter<=iter_end ){
      double x = (double)(iter-iter_begin)/(iter_end-iter_begin);
      dynamics->target_temperature = TempPara[i-1][2] + (TempPara[i][2] - TempPara[i-1][2])*x;
      break;
    }
  }

  //---- NH WV Step 7. evolve Nose-Hoover thermostat
  double NH_dzeta = atomnum*(kB/eV2Hartree)*(dynamics->temperature-dynamics->target_temperature)/(TempQ*Wscale);

  dynamics->nose_next_friction = dynamics->nose_friction + NH_dzeta * dt;
  dynamics->nose_friction = dynamics->nose_next_friction;
  dynamics->nose_coordinate = dynamics->nose_coordinate + dynamics->nose_friction*dt;

  // update GivenPressure
  double GivenPres;
  for( i=1; i<=PreNum; i++ ){
    int iter_begin    = PrePara[i-1][1];
    int iter_end      = PrePara[i  ][1];

    if( iter_begin<iter && iter<=iter_end ){
      double x = (double)(iter-iter_begin)/(iter_end-iter_begin);
      GivenPres = PrePara[i-1][2] + (PrePara[i][2] - PrePara[i-1][2])*x;
      break;
    }
  }

  NPT_Drift(coords,dcoords,dt,Wscale,1.0,Wa);

  // Wb = (sum_abcabc L[abc][xyz]*Wa[abc][abc]*L[abc][xyz]/Omega + Stress[xyz][xyz]
  //    - Pressure[xyz][xyz])/PresW
  for( al=1; al<=3; al++ ){
    for( be=1; be<=3; be++ ){
      Wb[al][be] = 0.0;
      for( i=1; i<=3; i++ ){
	for( j=1; j<=3; j++ ){
	  Wb[al][be] += L[i][al] * Wa[i][j] * L[j][be];
	}
      }
      Wb[al][be] /= Omega;
      Wb[al][be] += getStress(al,be,Omega);
      Wb[al][be] -= GivenPres*(al==be ? 1.0 : 0.0)*(double)MD_applied_pressure_flag[al-1];
      Wb[al][be] /= PresW;
    }
  }

  // restrict lattice vector
  if( LatticeRestriction == 1 ){ // cubic
    Wb[1][2] = Wb[2][1] = 0.0;
    Wb[1][3] = Wb[3][1] = 0.0;
    Wb[2][3] = Wb[3][2] = 0.0;
    Wb[1][1] = Wb[2][2] = Wb[3][3] = (Wb[1][1]+Wb[2][2]+Wb[3][3])/3.0;
  }
  else if( LatticeRestriction == 2 ){ // ortho
    Wb[1][2] = Wb[2][1] = 0.0;
    Wb[1][3] = Wb[3][1] = 0.0;
    Wb[2][3] = Wb[3][2] = 0.0;
  }

  // d2L = Wb*Sigma
  for( i=1; i<=3; i++ ){
    for( al=1; al<=3; al++ ){
      d2L[i][al] = Dot_Product( Wb[al], Sigma[i] );
    }
  }

  // Wa = d2L + 0.5*TrEFP - dLdF
  for( i=1; i<=3; i++ ){
    for( al=1; al<=3; al++ ){
      Wa[i][al] = d2L[i][al] + 0.5*TrEFP[i][al] - dLdF[i][al];
    }
  }

  // d2L[i][al] = sum_k Wa[k][al] * invF[i][k], Wentzcovitch
  for( i=1; i<=3; i++ ){
    for( al=1; al<=3; al++ ){
      d2L[i][al] = 0.0;
      for( k=1; k<=3; k++ ){
	d2L[i][al] += Wa[k][al] * invF[i][k];
      }
    }
  }

  //---- NH WV Step 10. evolve dL and L

  // evolve dL by dt
  for( i=1; i<=3; i++ ){
    for( al=1; al<=3; al++ ){
      dL[i][al] += d2L[i][al] * dt;
      tv_velocity[i][al] = dL[i][al];
    }
  }

  // evolve L by dt
  for( i=1; i<=3; i++ ){
    for( al=1; al<=3; al++ ){
      L[i][al] += dL[i][al] * dt;
    }
  }

  //---- NH WV Step 11. update Gxyz from coords

  // update position, velocity
#pragma omp parallel for schedule(static) if(Matomnum>=128) private(a,i,al)
  for( ai=1; ai<=Matomnum; ai++ ){
    a = M2G[ai];
    for( al=1; al<=3; al++ ){

      /* store the coordinates at the previous step */

      Gxyz[a][20+al] = Gxyz[a][al];

      /* update the coordinates */

      if( !FIXED(atom_Fixed_XYZ[a])[al] ){
	POSITION(Gxyz[a])[al] = Grid_Origin[al];
	for( i=1; i<=3; i++ ){
	  POSITION(Gxyz[a])[al] += coords[ai][i]*L[i][al];
	}
      }
    }
    for( al=1; al<=3; al++ ){

      /* update the velocity */

      if( !FIXED(atom_Fixed_XYZ[a])[al] ){
	VELOCITY(Gxyz[a])[al] = 0.0;
	for( i=1; i<=3; i++ ){
	  VELOCITY(Gxyz[a])[al] += dcoords[ai][i]*L[i][al];
	}
      }
    }
  }

  //---- NH WV Step 12. update tv from L

  // update tv
  for( i=1; i<=3; i++ ){
    for( al=1; al<=3; al++ ){
      tv[i][al] = L[i][al];
    }
  }

  // update rtv, Cell_Volume
  // rtv = (L2*L3,L3*L1,L1*L2)/Omega*2*Pi
  Cross_Product( tv[2], tv[3], rtv[1] );
  Cross_Product( tv[3], tv[1], rtv[2] );
  Cross_Product( tv[1], tv[2], rtv[3] );
  Cell_Volume = fabs(Dot_Product( rtv[1], tv[1] ));
  for( i=1; i<=3; i++ ){
    for( al=1; al<=3; al++ ){
      rtv[i][al] = rtv[i][al]/Cell_Volume*2*M_PI;
    }
  }

  //---- NH WV Step 13. calc Hamiltonian as the conserved quantity
  double TrdLF0dL = 0.0;
  for( i=1; i<=3; i++ ){
    for( j=1; j<=3; j++ ){
      TrdLF0dL += Dot_Product( dL[i], dL[i] ) * F0[i][j];
    }
  }
  double Uthermo = 0.5*dynamics->nose_friction*dynamics->nose_friction*TempQ*Wscale
    + 3.0*atomnum*(kB/eV2Hartree)*dynamics->target_temperature*dynamics->nose_coordinate;
  double Ubaro = 0.5*PresW*TrdLF0dL;
  double Wbaro = GivenPres*Omega;
  double Ham = thermo->total + thermo->ionic_kinetic + Ubaro + Wbaro;

  if ( myid == 0 ){

    static int first = 1;
    static FILE* fptr;
    static char fileE[YOUSO10];

    if ( first ){
      first = 0;
      
      sprintf(fileE,"%s%s.npt.ene",filepath,filename);
      
      fptr = fopen(fileE,"r");

      if (fptr==NULL) fptr = fopen(fileE,"w");
      else            fptr = fopen(fileE,"a");

      fprintf( fptr, "# NPT_NH_WV TimeStep=%e MassBarostat=%e\n", dt, PresW );
      fprintf( fptr, "#  1:iter 2:Volume 3:Temp    4:Stress   5:Utot   6:Ukc    7:Ubaro  8:Wbaro    9:Hamiltonian\n");
    }
    else{
      fptr = fopen(fileE,"a");
    }  

    fprintf( fptr, "%4d %f %f %+f %+f %f %f %f %+f\n", iter,
	     fabs(Omega), dynamics->temperature, (getStress(1,1,Omega)+getStress(2,2,Omega)+getStress(3,3,Omega))*29.42*1000/3.0,
	     thermo->total, thermo->ionic_kinetic, Ubaro, Wbaro, Ham );

    fflush(fptr);
    fclose(fptr);
  }

} // NPT_NH_WV

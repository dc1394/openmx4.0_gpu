#include "openmx_grid_assembly.h"
#include "openmx_env.h"
#include <limits.h>
#include <omp.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static void grid_check(int ok,const char *error)
{
  if (!ok){
    int rank;
    MPI_Comm_rank(mpi_comm_level1,&rank);
    fprintf(stderr,"OpenMX grid assembly (rank %d): %s\n",rank,error);
    MPI_Abort(mpi_comm_level1,1);
    abort();
  }
}

OpenMXGridWorkspace *OpenMX_Grid_Workspace(OpenMXCoreContext *core)
{
  OpenMXGridWorkspace *grid=NULL;
  char error[256];
  grid_check(openmx_core_grid_workspace(core,&grid,error,sizeof(error)),error);
  return grid;
}

OpenMXGridDensity OpenMX_Grid_Density_Storage(OpenMXGridWorkspace *grid)
{
  int a,ranks;
  char error[256];
  OpenMXGridDensity density;
  /* This small topology adapter is consumed synchronously and copied once by
     the owner. Numerical storage is contiguous and survives SCF iterations. */
  size_t points[(size_t)Matomnum+1];
  points[0]=1;
  for (a=1;a<=Matomnum;a++)points[a]=(size_t)GridN_Atom[F_M2G[a]];
  MPI_Comm_size(mpi_comm_level1,&ranks);
  grid_check(openmx_core_grid_density_prepare(grid,SpinP_switch+1,(size_t)Matomnum+1,points,
      (size_t)ranks,Num_Snd_Grid_A2B,Num_Rcv_Grid_A2B,&density,error,sizeof(error)),error);
  return density;
}

static size_t grid_pairs(void)
{
  int a;size_t n=0;
  for (a=1;a<=Matomnum;a++){
    size_t count=(size_t)FNAN[M2G[a]]+1;
    grid_check(n<=SIZE_MAX-count,"atom pair count overflow");n+=count;
  }
  return n;
}

static OpenMXGridPair grid_pair(int a,int h,int rows,int columns,int rank,int contracted)
{
  OpenMXGridPair p={0};
  const int neighbor=natn[M2G[a]][h],m=F_G2M[neighbor];
  p.atom=a;p.rows=rows;p.columns=columns;p.points=NumOLG[a][h];
  p.center_points=GListTAtoms1[a][h];p.neighbor_points=GListTAtoms2[a][h];
  p.center.element_bytes=p.neighbor.element_bytes=sizeof(Type_Orbs_Grid);
  if (contracted){
    p.center.rows=COrbs_Grid[a];p.neighbor.rows=COrbs_Grid[m];
    p.center.transposed=p.neighbor.transposed=1;
  }
  else{
    p.center.rows=Orbs_Grid[a];
    p.neighbor.overlap_rows=G2ID[neighbor]!=rank;
    p.neighbor.rows=p.neighbor.overlap_rows?Orbs_Grid_FNAN[a][h]:Orbs_Grid[m];
  }
  return p;
}

static void grid_trace(int id,const char *label,int mode,int rank)
{
  if (rank==0 && openmx_env_flag_at(id, 0)){
    printf("%s SIMD backend = %s\n",label,openmx_core_grid_simd_name(mode));
    fflush(stdout);
  }
}

void OpenMX_Grid_Density_CPU(OpenMXGridWorkspace *grid,const OpenMXAtomicViews *atomic,
    int Cnt_kind,double *****density_matrix)
{
  int a,h,s,rank,max_orbitals=1;
  size_t k=0;
  char error[256];
  OpenMXGridAtom *atoms;
  OpenMXGridPair *pairs;
  const int mode=openmx_core_grid_simd(openmx_env_at(SpinP_switch==3?OPENMX_ENV_DENSITY_GRID_NC_SIMD:OPENMX_ENV_DENSITY_GRID_COL_SIMD));
  MPI_Comm_rank(mpi_comm_level1,&rank);
  for (a=1;a<=Matomnum;a++){
    int g=M2G[a];
    max_orbitals=max_orbitals>atomic->species.contracted_orbitals[WhatSpecies[g]]?
      max_orbitals:atomic->species.contracted_orbitals[WhatSpecies[g]];
    for (h=0;h<=FNAN[g];h++){
      int n=atomic->species.contracted_orbitals[WhatSpecies[natn[g][h]]];
      if (n>max_orbitals)max_orbitals=n;
    }
  }
  grid_check(openmx_core_grid_plan(grid,OPENMX_GRID_DENSITY,SpinP_switch+1,(size_t)Matomnum,
      grid_pairs(),(size_t)max_orbitals,(size_t)omp_get_max_threads(),mode,&atoms,&pairs,error,sizeof(error)),error);
  for (a=1;a<=Matomnum;a++){
    int g=M2G[a],rows=atomic->species.contracted_orbitals[WhatSpecies[g]];
    atoms[a-1].first=k;atoms[a-1].count=(size_t)FNAN[g]+1;atoms[a-1].global_atom=g;
    for (h=0;h<=FNAN[g];h++,k++){
      int columns=atomic->species.contracted_orbitals[WhatSpecies[natn[g][h]]];
      pairs[k]=grid_pair(a,h,rows,columns,rank,Cnt_kind==0 && Cnt_switch==1);
      for (s=0;s<=SpinP_switch;s++)pairs[k].matrix[s]=density_matrix[s][a][h];
    }
  }
  grid_trace(SpinP_switch==3?OPENMX_ENV_DENSITY_GRID_NC_SIMD_TRACE:OPENMX_ENV_DENSITY_GRID_COL_SIMD_TRACE,
      SpinP_switch==3?"Set_Density_Grid: NC":"Set_Density_Grid: collinear",mode,rank);
  /* Use the caller's OpenMP runtime, including dynamic and nested teams. The
     core receives the actual team and never creates a second runtime team. */
#pragma omp parallel if (omp_get_max_threads()>1)
  {
    char worker_error[256];
    grid_check(openmx_core_grid_density_worker(grid,(size_t)omp_get_thread_num(),(size_t)omp_get_num_threads(),
        time_per_atom,dtime,worker_error,sizeof(worker_error)),worker_error);
  }
}

void OpenMX_Grid_Hamiltonian_CPU(OpenMXGridWorkspace *grid,int Cnt_kind,const OpenMXElectronicViews *electronic)
{
  int a,h,s,rank,max_orbitals=1;
  size_t k=0;
  char error[256];
  OpenMXGridAtom *atoms;
  OpenMXGridPair *pairs;
  double *****matrix;
  const int *orbitals=Cnt_kind==0?Spe_Total_NO:Spe_Total_CNO;
  const int mode=openmx_core_grid_simd(openmx_env_at(OPENMX_ENV_SETHAM_SIMD));
  grid_check(Cnt_kind==0 || Cnt_kind==1,"Hamiltonian Cnt_kind must be 0 or 1");
  grid_check(electronic!=NULL,"missing electronic matrix views");
  matrix=Cnt_kind==0?electronic->H:electronic->CntH;
  MPI_Comm_rank(mpi_comm_level1,&rank);
  for (a=1;a<=Matomnum;a++){
    int g=M2G[a];
    if (orbitals[WhatSpecies[g]]>max_orbitals)max_orbitals=orbitals[WhatSpecies[g]];
    for (h=0;h<=FNAN[g];h++){
      int n=orbitals[WhatSpecies[natn[g][h]]];if (n>max_orbitals)max_orbitals=n;
    }
  }
  grid_check(openmx_core_grid_plan(grid,OPENMX_GRID_HAMILTONIAN,SpinP_switch+1,0,grid_pairs(),
      (size_t)max_orbitals,(size_t)omp_get_max_threads(),mode,&atoms,&pairs,error,sizeof(error)),error);
  for (a=1;a<=Matomnum;a++){
    int g=M2G[a],rows=orbitals[WhatSpecies[g]];
    for (h=0;h<=FNAN[g];h++,k++){
      pairs[k]=grid_pair(a,h,rows,orbitals[WhatSpecies[natn[g][h]]],rank,0);
      pairs[k].volume=GridVol;pairs[k].potential_index=MGridListAtom[a];
      for (s=0;s<=SpinP_switch;s++){
        pairs[k].matrix[s]=matrix[s][a][h];pairs[k].potential[s]=Vpot_Grid[s];
      }
    }
  }
  grid_trace(OPENMX_ENV_SETHAM_SIMD_TRACE,"Set_Hamiltonian",mode,rank);
#pragma omp parallel if (omp_get_max_threads()>1)
  {
    char worker_error[256];
    grid_check(openmx_core_grid_hamiltonian_worker(grid,(size_t)omp_get_thread_num(),(size_t)omp_get_num_threads(),
        worker_error,sizeof(worker_error)),worker_error);
  }
}

OpenMXGridExchange OpenMX_Grid_Exchange(OpenMXGridWorkspace *grid,int kind,
    size_t send_values,size_t receive_values,size_t sends,size_t receives)
{
  char error[256];OpenMXGridExchange exchange;
  grid_check(openmx_core_grid_exchange(grid,kind,send_values,receive_values,sends,receives,
      sizeof(MPI_Request),sizeof(MPI_Status),&exchange,error,sizeof(error)),error);
  return exchange;
}

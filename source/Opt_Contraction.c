/**********************************************************************
  Opt_Contraction.c: native MPI/LAPACK boundary for orbital optimization.
  The numerical algorithm was released by T. Ozaki on 22/Nov/2001 and now
  lives in the shared C++ core with calculation-owned state and scratch.
***********************************************************************/
#include "openmx_common.h"
#include "openmx_orbital_optimization.h"
#include "lapack_prototypes.h"
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct { MPI_Comm communicator; } OrbitalServices;
static int sum_values(void *user,const double *in,double *out,int count)
{
  return MPI_Allreduce(in,out,count,MPI_DOUBLE,MPI_SUM,
      ((OrbitalServices*)user)->communicator)==MPI_SUCCESS;
}
static int maximum_value(void *user,int value,int *result)
{
  return MPI_Allreduce(&value,result,1,MPI_INT,MPI_MAX,
      ((OrbitalServices*)user)->communicator)==MPI_SUCCESS;
}
static int broadcast_values(void *user,double *values,int count,int root)
{
  return MPI_Bcast(values,count,MPI_DOUBLE,root,
      ((OrbitalServices*)user)->communicator)==MPI_SUCCESS;
}
static int exchange_values(void *user,int destination,const double *send,int send_count,
    int source,double *receive,int receive_count)
{
  MPI_Status status;
  return MPI_Sendrecv(send,send_count,MPI_DOUBLE,destination<0 ? MPI_PROC_NULL:destination,999,
      receive,receive_count,MPI_DOUBLE,source<0 ? MPI_PROC_NULL:source,999,
      ((OrbitalServices*)user)->communicator,&status)==MPI_SUCCESS;
}
static int diagonalize(void *user,double **matrix,double *values,int n)
{
  (void)user;
  Eigen_lapack(matrix,values,n,n);
  return 1;
}
static int invert_matrix(void *user,int n,const double *a,size_t stride,double *inverse,
    size_t inverse_stride,double *packed,double *work,int *pivots)
{
  int i,j;
  INTEGER count=n,info,lda=n,lwork=n;
  (void)user;
  _Static_assert(sizeof(INTEGER)==sizeof(int),"orbital optimization requires LP64 LAPACK integers");
  /* Preserve the old row-major packing and transpose cancellation in GETRI. */
  for(i=0;i<n;i++) for(j=0;j<n;j++) packed[(size_t)i*n+j]=a[(size_t)i*stride+j];
  F77_NAME(dgetrf,DGETRF)(&count,&count,packed,&lda,pivots,&info);
  if(info) return 0;
  F77_NAME(dgetri,DGETRI)(&count,packed,&lda,pivots,work,&lwork,&info);
  if(info) return 0;
  for(i=0;i<n;i++) for(j=0;j<n;j++) inverse[(size_t)i*inverse_stride+j]=packed[(size_t)i*n+j];
  return 1;
}
static void orbital_failure(const char *message)
{
  fprintf(stderr,"Orbital optimization: %s\n",message);
  MPI_Abort(mpi_comm_level1,1);
  abort();
}
void Begin_Orbital_Optimization(OpenMXCoreContext *core,const OpenMXAtomicViews *atomic)
{
  int rank,ranks;
  OpenMXOrbOptViews views;
  char error[256];
  MPI_Comm_rank(mpi_comm_level1,&rank);
  MPI_Comm_size(mpi_comm_level1,&ranks);
  const OpenMXOrbOptInput input={
    .atoms=atomnum,.local_atoms=Matomnum,.first_halo_atoms=MatomnumF,
    .max_orbitals=List_YOUSO[7],.max_coefficients=List_YOUSO[24],
    .history=orbitalOpt_History,.spin=SpinP_switch,.species_mode=ACnt_switch,
    .last_radial_only=SCnt_switch,.method=OrbOpt_OptMethod,.start_pulay=orbitalOpt_StartPulay,
    .rank=rank,.ranks=ranks,.host_rank=Host_ID,.verbosity=level_stdout,.descent_step=orbitalOpt_SD_step,
    .species=&atomic->species,.atom_species=WhatSpecies,.local_to_global=M2G,
    .first_to_global=F_M2G,.global_to_first=F_G2M,.neighbors=FNAN,.neighbor_atoms=natn,
    .coefficient_counts=Spe_Specified_Num,.orbital_indices=Spe_Trans_Orbital,
    .send_counts=F_Snd_Num,.receive_counts=F_Rcv_Num,.receive_first=F_TopMAN,
    .send_local_atoms=Snd_MAN,.send_global_atoms=Snd_GAN,.receive_global_atoms=Rcv_GAN
  };
  if(!openmx_core_orbopt_begin(core,&input,&views,error,sizeof(error))) orbital_failure(error);
  if(memoryusage_fileout) {
    const OpenMXOrbOptHistory history[4]={views.atomic_coefficients,views.atomic_gradients,
      views.species_coefficients,views.species_gradients};
    const char *names[4]={"OrbOpt: atomic coefficients","OrbOpt: atomic gradients",
      "OrbOpt: species coefficients","OrbOpt: species gradients"};
    for(int i=0;i<4;i++) {
      size_t n=1;
      for(int d=0;d<4;d++) n*=history[i].extent[d];
      if(n) PrintMemory((char*)names[i],sizeof(double)*n,NULL);
    }
    PrintMemory("OrbOpt: Hessian",sizeof(double)*views.hessian.rows*views.hessian.columns,NULL);
  }
}
double Opt_Contraction(OpenMXCoreContext *core,int iteration,double energy,
    const OpenMXElectronicViews *electronic,const OpenMXLocalBasisViews *local_basis)
{
  OrbitalServices native={mpi_comm_level1};
  const OpenMXOrbOptCallbacks services={&native,sum_values,maximum_value,broadcast_values,
      exchange_values,diagonalize,invert_matrix};
  double norm;
  char error[256];
  if(!openmx_core_orbopt_step(core,iteration,energy,electronic,local_basis,&services,
      &norm,error,sizeof(error))) orbital_failure(error);
  return norm;
}

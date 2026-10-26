#include "openmx_gpu_backend.h"
#include "openmx_gpu_device.h"
/**********************************************************************
  Set_OLP_Kin.c:

     Set_OLP_Kin.c is a subroutine to calculate the overlap matrix
     and the matrix for the kinetic operator in momentum space.

  Log of Set_OLP_Kin.c:

     15/Oct./2002  Released by T.Ozaki
     25/Nov./2014  Memory allocation modified by A.M. Ito (AITUNE)

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include "openmx_common.h"
#include "openmx_env.h"
#include "openmx_gpu_task.h"
#include "openmx_gpu_task_layout.h"
#include "openmx_host_simd.h"
#include "openmx_core_overlap.h"
#include "openmx_core_vna_simd.h"
#include "openmx_two_center.h"
#include "openmx_integral_pairs.h"
#include "mpi.h"
#include <omp.h>

typedef struct {int grid_dim,grid_last,combo_count,h_AN;double h;} OLPKinRadialParameters;
typedef struct {
  int grid_dim,grid_last,combo_count,h_AN,Lmax_Four_Int;
  size_t rf_cache_elems,sph_valid_elems,sum_valid_elems;
  double h;
  const double *norm;
  double *rf0,*rf1,*out[4];
  const double *sph,*sphp;
  int *off0,*off1;
} OLPKinRadialWork;
static void OLPKin_Radial_Kernel(size_t first,size_t count,const void *parameters,void *const v[],int use_gpu)
{
  const OLPKinRadialParameters *p=parameters;
  int grid_dim=p->grid_dim,grid_last=p->grid_last,combo_count=p->combo_count,h_AN=p->h_AN;
  double h=p->h,*Normk_grid=v[0],*RF_BesselCache0=v[1],*RF_BesselCache1=v[2];
  double *SphB_flat=v[3],*SphBp_flat=v[4];int *combo_rf0_offset=v[5],*combo_rf1_offset=v[6];
  double *sumS0_flat=v[7],*sumK0_flat=v[8],*sumSr0_flat=v[9],*sumKr0_flat=v[10];
#if OPENMX_GPU_HIP
#pragma omp target teams distribute parallel for if(use_gpu) is_device_ptr(Normk_grid,RF_BesselCache0,RF_BesselCache1,SphB_flat,SphBp_flat,combo_rf0_offset,combo_rf1_offset,sumS0_flat,sumK0_flat,sumSr0_flat,sumKr0_flat)
#else
#pragma acc parallel loop if(use_gpu) deviceptr(Normk_grid,RF_BesselCache0,RF_BesselCache1,SphB_flat,SphBp_flat,combo_rf0_offset,combo_rf1_offset,sumS0_flat,sumK0_flat,sumSr0_flat,sumKr0_flat)
#endif
                  for (size_t ix=0;ix<count;ix++) {
                    int l=(int)((first+ix)/(size_t)combo_count);
                    int combo=(int)((first+ix)%(size_t)combo_count);
	              int rf0_offset,rf1_offset;
	              double local_sumS0,local_sumK0,local_sumSr0,local_sumKr0;

	              rf0_offset = combo_rf0_offset[combo];
	              rf1_offset = combo_rf1_offset[combo];
	              local_sumS0 = 0.0;
	              local_sumK0 = 0.0;
	              local_sumSr0 = 0.0;
	              local_sumKr0 = 0.0;

#if !OPENMX_GPU_HIP
#pragma acc loop vector reduction(+:local_sumS0,local_sumK0,local_sumSr0,local_sumKr0)
#endif
	              for (int i=0; i<grid_dim; i++){
	                double acc_coe0,acc_Normk,acc_Normk2;
	                double acc_sj,acc_sjp,acc_tmp0,acc_tmp1,acc_tmp2,acc_tmp3,acc_tmp4;

	                if (i==0 || i==grid_last) acc_coe0 = 0.50;
	                else                      acc_coe0 = 1.00;

	                acc_Normk = Normk_grid[i];
	                acc_Normk2 = acc_Normk*acc_Normk;
	                acc_sj  = SphB_flat[l*grid_dim + i];
	                acc_sjp = SphBp_flat[l*grid_dim + i];
	                acc_tmp0 = acc_coe0*h*acc_Normk2*RF_BesselCache0[rf0_offset + i];
	                acc_tmp1 = acc_tmp0*acc_sj;
	                acc_tmp2 = acc_tmp0*acc_Normk*acc_sjp;
	                acc_tmp3 = acc_tmp1*RF_BesselCache1[rf1_offset + i];
	                acc_tmp4 = acc_tmp2*RF_BesselCache1[rf1_offset + i];
	                local_sumS0 += acc_tmp3;
	                local_sumK0 += acc_tmp3*acc_Normk2;
	                local_sumSr0 += acc_tmp4;
	                local_sumKr0 += acc_tmp4*acc_Normk2;
	              }

	              if (h_AN==0){
	                local_sumSr0 = 0.0;
	                local_sumKr0 = 0.0;
	              }

	              sumS0_flat[ix] = local_sumS0;
	              sumK0_flat[ix] = local_sumK0;
	              sumSr0_flat[ix] = local_sumSr0;
	              sumKr0_flat[ix] = local_sumKr0;
                  }
}
static void OLPKin_Radial_Task(OLPKinRadialWork *w,OpenMXGpuTask *t,OLPKinRadialParameters *p)
{
  *p=(OLPKinRadialParameters){w->grid_dim,w->grid_last,w->combo_count,w->h_AN,w->h};
  *t=(OpenMXGpuTask){0};t->count=w->sum_valid_elems;t->parameters=p;t->parameter_bytes=sizeof(*p);
  t->inputs=7;t->outputs=4;t->kernel=OLPKin_Radial_Kernel;
  t->view[0]=(OpenMXGpuTaskView){(void*)w->norm,(size_t)w->grid_dim*sizeof(double),0};
  t->view[1]=(OpenMXGpuTaskView){w->rf0,w->rf_cache_elems*sizeof(double),0};
  t->view[2]=(OpenMXGpuTaskView){w->rf1,w->rf_cache_elems*sizeof(double),0};
  t->view[3]=(OpenMXGpuTaskView){(void*)w->sph,w->sph_valid_elems*sizeof(double),0};
  t->view[4]=(OpenMXGpuTaskView){(void*)w->sphp,w->sph_valid_elems*sizeof(double),0};
  t->view[5]=(OpenMXGpuTaskView){w->off0,(size_t)w->combo_count*sizeof(int),0};
  t->view[6]=(OpenMXGpuTaskView){w->off1,(size_t)w->combo_count*sizeof(int),0};
  for(int c=0;c<4;c++)t->view[7+c]=(OpenMXGpuTaskView){w->out[c],0,sizeof(double)};
}
static void OLPKin_Radial_Distribute(OLPKinRadialWork *w)
{
  OLPKinRadialParameters p;OpenMXGpuTask t;
  OLPKin_Radial_Task(w,&t,&p);
  openmx_gpu_task_execute(mpi_comm_level1,&t,1,"overlap-kinetic-radial");
}
/* One pair's radial integrals on the device. Returns this rank's completion.
   Under the host SIMD policy a failed attempt is not completed by the
   capacity task (whose host body is the device kernel run serially): every
   rank learns whether any rank failed (one Allreduce, all ranks, every
   round), *enabled is cleared on all ranks together, and the caller computes
   a failed pair and every later pair on the CPU path. OPENMX_HOST_SIMD=legacy
   keeps the capacity task. */
static int OLPKin_Radial_Attempt(OLPKinRadialWork *w,int *enabled)
{
  if (OpenMX_HostSimdMode()==OPENMX_HOST_SIMD_LEGACY) {
    OLPKin_Radial_Distribute(w);
    return 1;
  }
  OLPKinRadialParameters p;OpenMXGpuTask t;
  OLPKin_Radial_Task(w,&t,&p);
  const int done=!t.count || (gpu_rank_device_usable() && openmx_gpu_task_local(&t,"overlap-kinetic-radial"));
  int all_done=done;
  MPI_Allreduce(MPI_IN_PLACE,&all_done,1,MPI_INT,MPI_MIN,mpi_comm_level1);
  if (!all_done) *enabled=0;
  return done;
}

/* The host SIMD path of one pair's radial integrals: lanes across the radial
   function pairs (combos) and the legacy per-element order, so the sums are
   bitwise those of the C loops (openmx_core_vna_overlap_radial). The radial
   rows of the species pair are transposed to [point][combo] once per species
   pair, in a table owned by the calling thread. Returns 0 under
   OPENMX_HOST_SIMD=legacy, when the table cannot be allocated, or when the
   Core refuses the request; the caller then runs the C loops. Rank-local. */
typedef struct {
  double *rows;     /* [2][grid_dim][combo_count]: species C rows, then H rows */
  size_t capacity;  /* doubles */
  int valid;        /* rows hold the current species pair */
} OLPKinHostRadial;

static int OLPKin_Radial_Host(OLPKinHostRadial *host,int mode,const OLPKinRadialWork *w)
{
  if (mode==OPENMX_HOST_SIMD_LEGACY || w->combo_count<=0) return 0;
  const size_t points=(size_t)w->grid_dim,combos=(size_t)w->combo_count;
  if (!host->valid){
    const size_t need=2*points*combos;
    if (need>host->capacity){
      double *grown=(double*)realloc(host->rows,need*sizeof(double));
      if (grown==NULL) return 0;
      host->rows=grown;
      host->capacity=need;
    }
    /* point-major, so the writes stream (the source rows stay cached) */
    for (size_t i=0; i<points; i++){
      double *row0=host->rows+i*combos,*row1=host->rows+(points+i)*combos;
      for (size_t c=0; c<combos; c++){
        row0[c]=w->rf0[(size_t)w->off0[c]+i];
        row1[c]=w->rf1[(size_t)w->off1[c]+i];
      }
    }
    host->valid=1;
  }
  OpenMXVnaOverlapRadial q={0};
  q.points=w->grid_dim;q.orders=w->Lmax_Four_Int+1;q.combos=w->combo_count;q.onsite=w->h_AN==0;
  q.h=w->h;q.k=w->norm;q.rf0t=host->rows;q.rf1t=host->rows+points*combos;q.table_stride=combos;
  q.sb=w->sph;q.sbp=w->sphp;q.sb_stride=points;
  for (int c=0; c<4; c++) q.out[c]=w->out[c];
  q.out_stride=combos;
  return openmx_core_vna_overlap_radial(&q,mode);
}

#if OPENMX_GPU_HIP
static int SetOLPKinUseOpenMP(void);
#else
static int SetOLPKinUseOpenACC(void);
#endif


#if OPENMX_GPU_HIP
static int SetOLPKinUseOpenMP(void)
#else
static int SetOLPKinUseOpenACC(void)
#endif
{
  return (scf_eigen_lib_flag == GPUSOLVER);
}

static int SetOLPKinRadialGpuEnabled(void)
{
  /* The per-pair OpenACC offload of the radial integration is dominated by
     data-region setup and transfers (~1.2 s vs ~0.3 s on the CPU for the
     sidia333 benchmark), so it is opt-in via OPENMX_OLPKIN_RADIAL_GPU=1. */
  return openmx_env_flag_at(OPENMX_ENV_OLPKIN_RADIAL_GPU, 0);
}

/* Spherical-Bessel table cache keyed on the exact pair distance r. In
   crystalline systems many pairs share the same r bit pattern, and the
   tables depend only on (r, Lmax, k-grid), so a hit reproduces the computed
   values exactly. Misses just compute as before. Every OpenMP thread
   keeps a cache of its own (no sharing, no locks): the pairs of a thread
   are a contiguous block of the one-dimensionalized loop, so each thread
   meets the frequent shells by itself (RTX 5080, sidia333, 6 ranks x 3
   threads: without the cache the Bessel tables took 8.6 s per rank
   against 0.2 s with it).

   The cache lives for one Set_OLP_Kin call and is freed at its end: all
   real hits come from pairs of the same geometry, while keeping it across
   MD steps held the full table (~130 KB per entry, per rank) for the whole
   run and, because the exact-r keys of a moved geometry never match again,
   an insert-only cache would stay full of dead keys from step one onward.

   The entry limit bounds the transient footprint; the high-frequency
   distances of a crystal (the near shells) are also the first ones seen,
   so a modest limit keeps the hits that matter. Override with
   OPENMX_OLPKIN_BESSEL_CACHE_MAX (entries per thread; 0 disables the
   cache). */
#define OLPKIN_BESSEL_CACHE_DEFAULT 128

static int OLPKin_Bessel_CacheMax(void)
{
  long limit = openmx_env_long_at(OPENMX_ENV_OLPKIN_BESSEL_CACHE_MAX, OLPKIN_BESSEL_CACHE_DEFAULT);

  return (limit < 0) ? 0 : (limit > INT_MAX ? INT_MAX : (int)limit);
}

/* Runtime phase profiling gated by OPENMX_BAND_PROFILE=1 (one OLPKPROF line
   per rank on stderr; zero overhead when the variable is unset). */
typedef struct
{
  double bessel;
  double radial;
  double gaunt;
  double c2r;
} SetOLPKinProfileCounters;

static int SetOLP_ProfileEnabled(void)
{
  return openmx_env_flag_at(OPENMX_ENV_BAND_PROFILE, 0);
}


double Set_OLP_Kin(const OpenMXReciprocalViews *reciprocal, double *****OLP, double *****H0, const OpenMXElectronicViews *electronic)
{
  double *****OLP_L=electronic->OLP_L;

  /****************************************************
          Evaluate overlap and kinetic integrals
                 in the momentum space
  ****************************************************/
  static int firsttime=1;
  size_t orbitals,radial_channels;
  double time0;
  double TStime,TEtime;
  int numprocs,myid;
  int OneD_Nloop;
  const int *OneD2Mc_AN,*OneD2h_AN;
  SetOLPKinProfileCounters profile = {0};
  long bessel_hits = 0, bessel_misses = 0;
  /* Resolve runtime options before entering OpenMP.  A lazy shared static
     initialization from the worker threads races on the first call. */
  const int bessel_cache_max = OLPKin_Bessel_CacheMax();
  const int prof_enabled = SetOLP_ProfileEnabled();

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  dtime(&TStime);

  /****************************************************
   MPI_Barrier
  ****************************************************/

  MPI_Barrier(mpi_comm_level1);

  /* PrintMemory */

  if (firsttime) {

    radial_channels = (size_t)(List_YOUSO[25]+1)*List_YOUSO[24];
    orbitals = radial_channels*(List_YOUSO[25]+1);
    PrintMemory("Set_OLP_Kin: angular workspace",8*sizeof(dcomplex)*orbitals*orbitals,NULL);
    PrintMemory("Set_OLP_Kin: conversion workspace",8*sizeof(dcomplex)*(2*List_YOUSO[25]+1)*(2*List_YOUSO[25]+1),NULL);
    PrintMemory("Set_OLP_Kin: radial workspace",sizeof(double)*(2*radial_channels*(OneD_Grid+1)+
                4*(2*List_YOUSO[25]+1)*radial_channels*radial_channels+OneD_Grid+1),NULL);
    firsttime=0;
  }

  const OpenMXAtomPairInput pair_input={atomnum,Matomnum,SpeciesNum,TCpyCell+1,
      M2G,WhatSpecies,FNAN,natn,ncn,Gxyz,atv};
  OpenMXAtomPairBuffers pairs;
  OpenMXAtomPairPlan *pair_owner=OpenMX_IntegralPairs(&pair_input,&pairs);
  OneD_Nloop=pairs.count;
  OneD2Mc_AN=pairs.local;
  OneD2h_AN=pairs.neighbor;

  /* The optional radial offload is called only by the MPI main thread.
     Empty ranks enter the same number of recovery phases as busy ranks. */
#if OPENMX_GPU_HIP
  int radial_capacity=omp_get_max_threads()==1 && SetOLPKinRadialGpuEnabled() && SetOLPKinUseOpenMP();
#else
  int radial_capacity=omp_get_max_threads()==1 && SetOLPKinRadialGpuEnabled() && SetOLPKinUseOpenACC();
#endif
  MPI_Allreduce(MPI_IN_PLACE,&radial_capacity,1,MPI_INT,MPI_MIN,mpi_comm_level1);
  if (radial_capacity){
    /* Host SIMD vote (every rank, same point): one pair's radial working set
       (Fourier rows of both species, the Bessel tables, the four sums)
       against this rank's free device memory. Splitting it more than
       OPENMX_HOST_SIMD_MAX_PIECES times over, or an unusable device, keeps
       the radial integrals on the CPU path on every rank. */
    const size_t grid=(size_t)OneD_Grid+1,channels=(size_t)(List_YOUSO[25]+1)*List_YOUSO[24];
    const size_t orders=(size_t)(2*List_YOUSO[25]+1);
    const size_t need=sizeof(double)*(grid+2*channels*grid+2*orders*grid+4*orders*channels*channels)
                     +2*sizeof(int)*channels*channels;
    if (OpenMX_HostSimdStageOnHost("Set_OLP_Kin: radial",OneD_Nloop>0 ? need : 0,OpenMX_HostSimdDeviceBudget(0)))
      radial_capacity=0;
  }
  int radial_rounds=OneD_Nloop;
  if(radial_capacity)MPI_Allreduce(MPI_IN_PLACE,&radial_rounds,1,MPI_INT,MPI_MAX,mpi_comm_level1);
  const int host_simd_mode=OpenMX_HostSimdMode();

  /* OpenMP */

#pragma omp parallel
  {

	    int Nloop;
	    int OMPID,Nthrds,Nprocs;
	    int Mc_AN,h_AN,Gc_AN,Cwan;
	    int Gh_AN,Rnh,Hwan;
	    int Ls,L0,Mul0,L1,Mul1,M0,M1;
	    int Lmax_Four_Int;
	    int i,j,k,l,m,p;
	    int num0,num1;
	    int grid_dim;
#if OPENMX_GPU_HIP
	    int use_openmp_radial;
#else
	    int use_openacc_radial;
#endif
	    OpenMXOverlapWorkspace *workspace;
	    OpenMXOverlapBuffers work_buffers;
	    char bessel_error[256];
	    int cached_Cwan,cached_Hwan;
	    int combo_cached_Cwan,combo_cached_Hwan;
	    int combo_count,combo,grid_last;
	    int *combo_rf0_offset,*combo_rf1_offset;

	    double Stime_atom,Etime_atom;
	    double dx,dy,dz;
    double S_coordinate[3];
    double theta,phi,h;
    double Bessel_Pro0,Bessel_Pro1;
    double tmp0,tmp1,tmp2,tmp3,tmp4;
    double siT,coT,siP,coP;
    double kmin,kmax,Sk,Dk,r;
    double sj,sjp,coe0;
    double Normk,Normk2;
	    double gant,SH[2],dSHt[2],dSHp[2];
	    const double *SphB_flat,*SphBp_flat;
	    const double *Normk_grid;
	    double *RF_BesselCache0,*RF_BesselCache1;
	    double *sumS0_flat,*sumK0_flat,*sumSr0_flat,*sumKr0_flat;


    dcomplex Ctmp2,Cpow;
    dcomplex CY,CYt,CYp,CY1,CYt1,CYp1;
    dcomplex *TmpOLP;
    dcomplex *TmpOLPr;
    dcomplex *TmpOLPt;
    dcomplex *TmpOLPp;
    dcomplex *TmpKin;
    dcomplex *TmpKinr;
    dcomplex *TmpKint;
    dcomplex *TmpKinp;

	    size_t rf_cache_elems;
	    OLPKinHostRadial host_radial={0};
	    double prof_t0_local = 0.0, prof_t1_local = 0.0;
	    double prof_bessel = 0.0, prof_radial = 0.0, prof_gaunt = 0.0, prof_c2r = 0.0;

    /****************************************************************
                          allocation of arrays:
    ****************************************************************/

    const OpenMXOverlapShape shape = {List_YOUSO[25],List_YOUSO[24],OneD_Grid+1,
        Radial_kmin,PAO_Nkmax,(size_t)bessel_cache_max};
    workspace = openmx_core_overlap_create(&shape,&work_buffers,bessel_error,sizeof(bessel_error));
    if (workspace == NULL) {
      fprintf(stderr,"Set_OLP_Kin: %s\n",bessel_error);
      abort();
    }
    grid_dim = work_buffers.points;
    grid_last = grid_dim-1;
    rf_cache_elems = work_buffers.radial_elements;
    Normk_grid = work_buffers.wavevectors;
    RF_BesselCache0 = work_buffers.radial[0];
    RF_BesselCache1 = work_buffers.radial[1];
    sumS0_flat = work_buffers.sums[0];
    sumK0_flat = work_buffers.sums[1];
    sumSr0_flat = work_buffers.sums[2];
    sumKr0_flat = work_buffers.sums[3];
    combo_rf0_offset = work_buffers.offsets[0];
    combo_rf1_offset = work_buffers.offsets[1];
    TmpOLP = work_buffers.angular[0];
    TmpOLPr = work_buffers.angular[1];
    TmpOLPt = work_buffers.angular[2];
    TmpOLPp = work_buffers.angular[3];
    TmpKin = work_buffers.angular[4];
    TmpKinr = work_buffers.angular[5];
    TmpKint = work_buffers.angular[6];
    TmpKinp = work_buffers.angular[7];

	    cached_Cwan = -1;
	    cached_Hwan = -1;
	    combo_cached_Cwan = -1;
	    combo_cached_Hwan = -1;

	    /* get info. on OpenMP */

	    OMPID = omp_get_thread_num();
	    Nthrds = omp_get_num_threads();
	    Nprocs = omp_get_num_procs();
#if OPENMX_GPU_HIP
	    use_openmp_radial = radial_capacity;
#else
	    use_openacc_radial = radial_capacity;
#endif
    /* one-dimensionalized loop */

    for (Nloop=OMPID*radial_rounds/Nthrds; Nloop<(OMPID+1)*radial_rounds/Nthrds; Nloop++){
      if(Nloop>=OneD_Nloop){
#if OPENMX_GPU_HIP
        if(use_openmp_radial){OLPKinRadialWork empty={0};(void)OLPKin_Radial_Attempt(&empty,&use_openmp_radial);}
#else
        if(use_openacc_radial){OLPKinRadialWork empty={0};(void)OLPKin_Radial_Attempt(&empty,&use_openacc_radial);}
#endif
        continue;
      }

      dtime(&Stime_atom);

      /* get Mc_AN and h_AN */

      Mc_AN = OneD2Mc_AN[Nloop];
      h_AN  = OneD2h_AN[Nloop];

      /* set data on Mc_AN */

      Gc_AN = M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];

      /* set data on h_AN */

      Gh_AN = natn[Gc_AN][h_AN];
      Rnh = ncn[Gc_AN][h_AN];
      Hwan = WhatSpecies[Gh_AN];

      dx = Gxyz[Gh_AN][1] + atv[Rnh][1] - Gxyz[Gc_AN][1];
      dy = Gxyz[Gh_AN][2] + atv[Rnh][2] - Gxyz[Gc_AN][2];
      dz = Gxyz[Gh_AN][3] + atv[Rnh][3] - Gxyz[Gc_AN][3];

      xyz2spherical(dx,dy,dz,0.0,0.0,0.0,S_coordinate);
      r     = S_coordinate[0];
      theta = S_coordinate[1];
      phi   = S_coordinate[2];

      /* for empty atoms or finite elemens basis */
      if (r<1.0e-10) r = 1.0e-10;

      /* precalculation of sin and cos */

      siT = sin(theta);
      coT = cos(theta);
      siP = sin(phi);
      coP = cos(phi);

      /****************************************************
          Overlap and the derivative
              \int RL(k)*RL'(k)*jl(k*R) k^2 dk^3,
              \int RL(k)*RL'(k)*j'l(k*R) k^3 dk^3

          Kinetic
              \int RL(k)*RL'(k)*jl(k*R) k^4 dk^3,
              \int RL(k)*RL'(k)*j'l(k*R) k^5 dk^3
      ****************************************************/

      kmin = Radial_kmin;
      kmax = PAO_Nkmax;
      Sk = kmax + kmin;
      Dk = kmax - kmin;

      for (L0=0; L0<=Spe_MaxL_Basis[Cwan]; L0++){
	for (Mul0=0; Mul0<Spe_Num_Basis[Cwan][L0]; Mul0++){
	  for (L1=0; L1<=Spe_MaxL_Basis[Hwan]; L1++){
	    for (Mul1=0; Mul1<Spe_Num_Basis[Hwan][L1]; Mul1++){
	      for (M0=-L0; M0<=L0; M0++){
		for (M1=-L1; M1<=L1; M1++){

                  const size_t pair_index = openmx_core_overlap_pair(&work_buffers,L0,Mul0,L0+M0,L1,Mul1,L1+M1);
                  TmpOLP[pair_index]  = Complex(0.0,0.0);
                  TmpOLPr[pair_index] = Complex(0.0,0.0);
                  TmpOLPt[pair_index] = Complex(0.0,0.0);
                  TmpOLPp[pair_index] = Complex(0.0,0.0);

                  TmpKin[pair_index]  = Complex(0.0,0.0);
                  TmpKinr[pair_index] = Complex(0.0,0.0);
                  TmpKint[pair_index] = Complex(0.0,0.0);
                  TmpKinp[pair_index] = Complex(0.0,0.0);

		}
	      }
	    }
	  }
	}
      }

      if (Spe_MaxL_Basis[Cwan]<Spe_MaxL_Basis[Hwan])
        Lmax_Four_Int = 2*Spe_MaxL_Basis[Hwan];
      else
        Lmax_Four_Int = 2*Spe_MaxL_Basis[Cwan];

	      if (prof_enabled) dtime(&prof_t0_local);

      h = (kmax - kmin)/(double)OneD_Grid;

      OpenMXPairBesselTable bessel_table;
      if (!openmx_core_overlap_sample(workspace,r,Lmax_Four_Int,
          &bessel_table,bessel_error,sizeof(bessel_error))) {
        fprintf(stderr,"Set_OLP_Kin: %s\n",bessel_error);
        abort();
      }
      SphB_flat = bessel_table.values;
      SphBp_flat = bessel_table.derivatives;

	      if (prof_enabled) {
	        dtime(&prof_t1_local);
	        prof_bessel += prof_t1_local - prof_t0_local;
	        dtime(&prof_t0_local);
	      }

	      if (cached_Cwan!=Cwan){
	        for (L0=0; L0<=Spe_MaxL_Basis[Cwan]; L0++){
		  for (Mul0=0; Mul0<Spe_Num_Basis[Cwan][L0]; Mul0++){
		    int rf_offset;

		    rf_offset = (L0*List_YOUSO[24] + Mul0)*grid_dim;
		    for (i=0; i<grid_dim; i++){
		      RF_BesselCache0[rf_offset + i] = RF_BesselF(reciprocal, Cwan,L0,Mul0,Normk_grid[i]);
		    }
		  }
		}
		cached_Cwan = Cwan;
	      }

	      if (cached_Hwan!=Hwan){
	        for (L1=0; L1<=Spe_MaxL_Basis[Hwan]; L1++){
		  for (Mul1=0; Mul1<Spe_Num_Basis[Hwan][L1]; Mul1++){
		    int rf_offset;

		    rf_offset = (L1*List_YOUSO[24] + Mul1)*grid_dim;
		    for (i=0; i<grid_dim; i++){
		      RF_BesselCache1[rf_offset + i] = RF_BesselF(reciprocal, Hwan,L1,Mul1,Normk_grid[i]);
		    }
		  }
		}
		cached_Hwan = Hwan;
	      }

	      if (combo_cached_Cwan!=Cwan || combo_cached_Hwan!=Hwan){
	        combo_count = 0;
	        for (L0=0; L0<=Spe_MaxL_Basis[Cwan]; L0++){
		  for (Mul0=0; Mul0<Spe_Num_Basis[Cwan][L0]; Mul0++){
	            for (L1=0; L1<=Spe_MaxL_Basis[Hwan]; L1++){
		      for (Mul1=0; Mul1<Spe_Num_Basis[Hwan][L1]; Mul1++){
		        combo_rf0_offset[combo_count] = (L0*List_YOUSO[24] + Mul0)*grid_dim;
		        combo_rf1_offset[combo_count] = (L1*List_YOUSO[24] + Mul1)*grid_dim;
		        combo_count++;
		      }
		    }
		  }
		}
		combo_cached_Cwan = Cwan;
		combo_cached_Hwan = Hwan;
		host_radial.valid = 0;
	      }

	      int radial_done=0;
	      OLPKinRadialWork radial={0};
	      radial.grid_dim=grid_dim;radial.grid_last=grid_last;radial.combo_count=combo_count;
	      radial.h_AN=h_AN;radial.Lmax_Four_Int=Lmax_Four_Int;radial.rf_cache_elems=rf_cache_elems;
	      radial.sph_valid_elems=(size_t)(Lmax_Four_Int+1)*(size_t)grid_dim;
	      radial.sum_valid_elems=(size_t)(Lmax_Four_Int+1)*(size_t)combo_count;radial.h=h;
	      radial.norm=Normk_grid;radial.rf0=RF_BesselCache0;radial.rf1=RF_BesselCache1;
	      radial.sph=SphB_flat;radial.sphp=SphBp_flat;radial.off0=combo_rf0_offset;radial.off1=combo_rf1_offset;
	      radial.out[0]=sumS0_flat;radial.out[1]=sumK0_flat;radial.out[2]=sumSr0_flat;radial.out[3]=sumKr0_flat;
#if OPENMX_GPU_HIP
	      if (use_openmp_radial)
                radial_done=OLPKin_Radial_Attempt(&radial,&use_openmp_radial);
#else
	      if (use_openacc_radial)
                radial_done=OLPKin_Radial_Attempt(&radial,&use_openacc_radial);
#endif
	      /* the host SIMD path, also for a pair whose device attempt failed */
	      if (!radial_done)
	        radial_done=OLPKin_Radial_Host(&host_radial,host_simd_mode,&radial);
	      /* the C loops: OPENMX_HOST_SIMD=legacy or a refused host request */
	      if (!radial_done){
	        for(l=0; l<=Lmax_Four_Int; l++){
		  for (combo=0; combo<combo_count; combo++){
		    int rf0_offset,rf1_offset;
		    double local_sumS0,local_sumK0,local_sumSr0,local_sumKr0;

		    rf0_offset = combo_rf0_offset[combo];
		    rf1_offset = combo_rf1_offset[combo];
		    local_sumS0 = 0.0;
		    local_sumK0 = 0.0;
		    local_sumSr0 = 0.0;
		    local_sumKr0 = 0.0;

		    for (i=0; i<grid_dim; i++){
		      if (i==0 || i==grid_last) coe0 = 0.50;
		      else                      coe0 = 1.00;

		      Normk = Normk_grid[i];
		      Normk2 = Normk*Normk;
		      sj  = SphB_flat[(size_t)l*grid_dim+i];
		      sjp = SphBp_flat[(size_t)l*grid_dim+i];
		      tmp0 = coe0*h*Normk2*RF_BesselCache0[rf0_offset + i];
		      tmp1 = tmp0*sj;
		      tmp2 = tmp0*Normk*sjp;
		      tmp3 = tmp1*RF_BesselCache1[rf1_offset + i];
		      tmp4 = tmp2*RF_BesselCache1[rf1_offset + i];
		      local_sumS0 += tmp3;
		      local_sumK0 += tmp3*Normk2;
		      local_sumSr0 += tmp4;
		      local_sumKr0 += tmp4*Normk2;
		    }

		    if (h_AN==0){
		      local_sumSr0 = 0.0;
		      local_sumKr0 = 0.0;
		    }

		    sumS0_flat[l*combo_count + combo] = local_sumS0;
		    sumK0_flat[l*combo_count + combo] = local_sumK0;
		    sumSr0_flat[l*combo_count + combo] = local_sumSr0;
		    sumKr0_flat[l*combo_count + combo] = local_sumKr0;
		  }
		}
	      }

	      if (prof_enabled) {
	        dtime(&prof_t1_local);
	        prof_radial += prof_t1_local - prof_t0_local;
	        dtime(&prof_t0_local);
	      }

	      /* l loop */

	      for(l=0; l<=Lmax_Four_Int; l++){


	        /****************************************************
          For overlap and the derivative,
          sum_m 8*(-i)^{-L0+L1+1}*
                C_{L0,-M0,L1,M1,l,m}*Y_{lm}
                \int RL(k)*RL'(k)*jl(k*R) k^2 dk^3,

          For kinetic,
          sum_m 4*(-i)^{-L0+L1+1}*
                C_{L0,-M0,L1,M1,l,m}*
                \int RL(k)*RL'(k)*jl(k*R) k^4 dk^3,
        ****************************************************/

        for(m=-l; m<=l; m++){

          ComplexSH(l,m,theta,phi,SH,dSHt,dSHp);
          SH[1]   = -SH[1];
          dSHt[1] = -dSHt[1];
          dSHp[1] = -dSHp[1];

          CY  = Complex(SH[0],SH[1]);
          CYt = Complex(dSHt[0],dSHt[1]);
          CYp = Complex(dSHp[0],dSHp[1]);

          combo = 0;
          for (L0=0; L0<=Spe_MaxL_Basis[Cwan]; L0++){
            for (Mul0=0; Mul0<Spe_Num_Basis[Cwan][L0]; Mul0++){
              for (L1=0; L1<=Spe_MaxL_Basis[Hwan]; L1++){
                for (Mul1=0; Mul1<Spe_Num_Basis[Hwan][L1]; Mul1++){
                  const size_t radial_index = (size_t)l*combo_count + combo++;

                  Ls = -L0 + L1 + l;

                  if (abs(L1-l)<=L0 && L0<=(L1+l) ){

		    Cpow = Im_pow(-1,Ls);
		    CY1  = Cmul(Cpow,CY);
		    CYt1 = Cmul(Cpow,CYt);
		    CYp1 = Cmul(Cpow,CYp);

		    for (M0=-L0; M0<=L0; M0++){

		      M1 = M0 - m;

                      if (abs(M1)<=L1){
                        const size_t pair_index = openmx_core_overlap_pair(&work_buffers,L0,Mul0,L0+M0,L1,Mul1,L1+M1);

                        gant = Gaunt(L0,M0,L1,M1,l,m);

                        /* S */

                        tmp0 = gant*sumS0_flat[radial_index];
                        Ctmp2 = CRmul(CY1,tmp0);
                        TmpOLP[pair_index] =
			  Cadd(TmpOLP[pair_index],Ctmp2);

                        /* dS/dr */

                        tmp0 = gant*sumSr0_flat[radial_index];
                        Ctmp2 = CRmul(CY1,tmp0);
                        TmpOLPr[pair_index] =
			  Cadd(TmpOLPr[pair_index],Ctmp2);

                        /* dS/dt */

                        tmp0 = gant*sumS0_flat[radial_index];
                        Ctmp2 = CRmul(CYt1,tmp0);
                        TmpOLPt[pair_index] =
			  Cadd(TmpOLPt[pair_index],Ctmp2);

                        /* dS/dp */

                        tmp0 = gant*sumS0_flat[radial_index];
                        Ctmp2 = CRmul(CYp1,tmp0);
                        TmpOLPp[pair_index] =
			  Cadd(TmpOLPp[pair_index],Ctmp2);

                        /* K */

                        tmp0 = gant*sumK0_flat[radial_index];
                        Ctmp2 = CRmul(CY1,tmp0);
                        TmpKin[pair_index] =
			  Cadd(TmpKin[pair_index],Ctmp2);

                        /* dK/dr */

                        tmp0 = gant*sumKr0_flat[radial_index];
                        Ctmp2 = CRmul(CY1,tmp0);
                        TmpKinr[pair_index] =
			  Cadd(TmpKinr[pair_index],Ctmp2);

                        /* dK/dt */

                        tmp0 = gant*sumK0_flat[radial_index];
                        Ctmp2 = CRmul(CYt1,tmp0);
                        TmpKint[pair_index] =
			  Cadd(TmpKint[pair_index],Ctmp2);

                        /* dK/dp */

                        tmp0 = gant*sumK0_flat[radial_index];
                        Ctmp2 = CRmul(CYp1,tmp0);
                        TmpKinp[pair_index] =
			  Cadd(TmpKinp[pair_index],Ctmp2);

		      }
		    }
		  }
		}
	      }

	    }
	  }
        }
      } /* l */

      if (prof_enabled) {
        dtime(&prof_t1_local);
        prof_gaunt += prof_t1_local - prof_t0_local;
        dtime(&prof_t0_local);
      }

      /****************************************************
                         Complex to Real
      ****************************************************/

      const OpenMXTwoCenterGeometry geometry={r,siT,coT,siP,coP,h_AN==0};
      num0 = 0;
      for (L0=0; L0<=Spe_MaxL_Basis[Cwan]; ++L0){
        for (Mul0=0; Mul0<Spe_Num_Basis[Cwan][L0]; ++Mul0){
          num1 = 0;
          for (L1=0; L1<=Spe_MaxL_Basis[Hwan]; ++L1){
            for (Mul1=0; Mul1<Spe_Num_Basis[Hwan][L1]; ++Mul1){
              OpenMX_TwoCenterTransform(&work_buffers.two_center,work_buffers.angular[0],
                  work_buffers.orbital_capacity,work_buffers.orbital_capacity,
                  openmx_core_overlap_orbital(work_buffers.multiplicity,L0,Mul0,0),
                  openmx_core_overlap_orbital(work_buffers.multiplicity,L1,Mul1,0),
                  L0,L1,&Comp2Real[0][0][0], sizeof(Comp2Real)/sizeof(Comp2Real[0]), sizeof(Comp2Real[0][0])/sizeof(Comp2Real[0][0][0]),&geometry);
              for (M0=0; M0<2*L0+1; ++M0){
                for (M1=0; M1<2*L1+1; ++M1){
                  for (int axis=0; axis<3; ++axis)
                    OLP_L[axis][Mc_AN][h_AN][num0+M0][num1+M1] =
                        OpenMX_TwoCenterMomentum(&work_buffers.two_center,axis,M0,M1);
                  /* Keep the stabilization RNG call in the original orbital order. */
                  OLP[0][Mc_AN][h_AN][num0+M0][num1+M1] =
                      OpenMX_TwoCenterValue(&work_buffers.two_center,0,M0,M1)+1.0*rnd(1.0e-13);
                  H0[0][Mc_AN][h_AN][num0+M0][num1+M1] =
                      OpenMX_TwoCenterValue(&work_buffers.two_center,4,M0,M1);
                  for (int channel=1; channel<4; ++channel){
                    OLP[channel][Mc_AN][h_AN][num0+M0][num1+M1] =
                        OpenMX_TwoCenterValue(&work_buffers.two_center,channel,M0,M1);
                    H0[channel][Mc_AN][h_AN][num0+M0][num1+M1] =
                        OpenMX_TwoCenterValue(&work_buffers.two_center,4+channel,M0,M1);
                  }
                }
              }
              num1 += 2*L1+1;
            }
          }
          num0 += 2*L0+1;
        }
      }

      if (prof_enabled) {
        dtime(&prof_t1_local);
        prof_c2r += prof_t1_local - prof_t0_local;
      }

      dtime(&Etime_atom);
#pragma omp atomic
      time_per_atom[Gc_AN] += Etime_atom - Stime_atom;
    } /* end of loop for Nloop */

    if (prof_enabled) {
#pragma omp atomic
      profile.bessel += prof_bessel;
#pragma omp atomic
      profile.radial += prof_radial;
#pragma omp atomic
      profile.gaunt += prof_gaunt;
#pragma omp atomic
      profile.c2r += prof_c2r;
    }


	    if (prof_enabled) {
      long thread_hits, thread_misses;
      openmx_core_overlap_statistics(workspace,&thread_hits,&thread_misses);
#pragma omp atomic
	      bessel_hits += thread_hits;
#pragma omp atomic
	      bessel_misses += thread_misses;
	    }
    free(host_radial.rows);
    openmx_core_overlap_destroy(workspace);

	  } /* #pragma omp parallel */

  /****************************************************
                   freeing of arrays:
  ****************************************************/

  if (prof_enabled) {
    fprintf(stderr, "OLPKPROF id=%d pairs=%d bessel=%.3f radial=%.3f gaunt=%.3f c2r=%.3f bhit=%ld bmiss=%ld\n", myid,
            OneD_Nloop, profile.bessel, profile.radial, profile.gaunt, profile.c2r,
            bessel_hits, bessel_misses);
  }

  openmx_core_atom_pairs_destroy(pair_owner);

  /* for time */
  dtime(&TEtime);
  time0 = TEtime - TStime;

  return time0;
}

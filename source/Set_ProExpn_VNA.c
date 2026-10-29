/**********************************************************************
  Set_ProExpn_VNA.c:

     Set_ProExpn_VNA.c is a subroutine to calculate matrix elements and
     the derivatives of VNA projector expansion in the momentum space.

  Log of Set_ProExpn_VNA.c:

     8/Apr/2004  Released by T.Ozaki

***********************************************************************/

#define  measure_time   0

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <sys/types.h>
#include <sys/times.h>
#include <sys/time.h> 
#include <sys/stat.h>
#include <unistd.h>
#include "openmx_common.h"
#include "openmx_env.h"
#include "openmx_gpu_workers.h"
#include "openmx_gpu_views.h"
#include "openmx_gpu_task.h"
#include "openmx_gpu_task_layout.h"
#include "openmx_host_simd.h"
#include "openmx_mgpu_runtime.h"
#include "openmx_host_memory.h"
#include "openmx_precision.h"
#include "openmx_vna_gpu_context.h"
#include "openmx_vna_preparation.h"
#include "set_cuda_default_device_from_local_rank.h"
#include "mpi.h"
#include "openmx_projector_exchange.h"
#include <omp.h>
#include <openacc.h>

static int SetPro_VNA23_GpuBegin(const OpenMXReciprocalViews *reciprocal, const dcomplex *conversion, size_t conversion_width, SetProGpu3Context *g);
static void SetPro_VNA23_GpuEnd(SetProGpu3Context *g);
/* forward: SetPro_DSVNA_GpuBegin releases its tables through this when the
   device-memory check sends the DS_VNA build down the CPU path */
static void SetPro_DSVNA_GpuEnd(SetProGpu2Context *g);
/* forward: the OPENMX_SETPRO_VERBOSE diagnostics (defined below the driver) */
static void SetPro_note(const char *stage, const char *fmt, ...) __attribute__((noinline));

static inline void Spherical_Bessel2( double x, int lmax, double *sb, double *dsb );


/* ------------------------------------------------------------------ */
/* Batched device evaluation of the HVNA projector traces.

   HVNA[Mc][j][m][n] = dmp * sum_k sum_L ene_L
                           * sum_l DS_VNA[0][Mc][k][m][l]*DS_VNA[0][Mj][kl][n][l]

   is the same energy-weighted projector contraction as the
   Force4B/Force_HNL first-case traces.  The kernel keeps the exact CPU
   accumulation order (k outermost, then the radial groups, then the
   2*L+1 members; products retain legacy FP32 rounding when enabled and
   promote their float operands to double when scf.fp32.enable is off), so the
   final float rounding of DS_VNA itself limits the agreement.  The MPI
   pipeline is untouched: received halo blocks are archived and one
   batched kernel runs after the loops.                                */
/* ------------------------------------------------------------------ */




static void SetPro_gpu_abort(const char *message)
{
  fprintf(stderr,"%s\n",message);
  fflush(stderr);
  if (openmx_gpu_worker_device()>=0) exit(EXIT_FAILURE);
  MPI_Abort(mpi_comm_level1,1);
}

static void *SetPro_checked_malloc(size_t size)
{
  void *ptr;

  if (size==0) size = 1;
  ptr = malloc(size);
  if (ptr==NULL){
    SetPro_gpu_abort("Set_ProExpn_VNA: malloc failed for the GPU batch.");
  }
  return ptr;
}

/* OPENMX_SETPRO_VERBOSE=1: every rank reports each device stage of this
   file (the admission figures, the batch sizes, the elapsed time) together
   with the device memory free at that moment, on stderr */
static int SetPro_verbose(void)
{
  return openmx_env_flag_at(OPENMX_ENV_SETPRO_VERBOSE, 0);

}

static double SetPro_free_mib(void)
{
  size_t free_bytes = 0,total_bytes = 0;

  /* a CPU run must not create a CUDA context for a diagnostic */
  if (scf_eigen_lib_flag!=GPUSOLVER) return -1.0;
  if (OpenMX_GpuMemGetInfo(&free_bytes,&total_bytes)!=cudaSuccess) return -1.0;
  return (double)free_bytes/1048576.0;
}

/* kept out of line: nvc 26.9 miscompiles the inlined copies (an undefined
   value for the formatted buffer in its LLVM IR) */
static void SetPro_note(const char *stage, const char *fmt, ...) __attribute__((noinline));
static void SetPro_note(const char *stage, const char *fmt, ...)
{
  char detail[512];
  va_list ap;
  int myid;

  if (!SetPro_verbose()) return;
  va_start(ap,fmt);
  vsnprintf(detail,sizeof(detail),fmt,ap);
  va_end(ap);
  if (openmx_gpu_worker_device()>=0) myid=0;
  else MPI_Comm_rank(mpi_comm_level1,&myid);
  {
    const double mib = SetPro_free_mib();

    if (mib<0.0) fprintf(stderr,"<Set_ProExpn_VNA> rank %d %s: %s\n",myid,stage,detail);
    else fprintf(stderr,"<Set_ProExpn_VNA> rank %d %s: %s; device free %.0f MiB\n",myid,stage,detail,mib);
  }
  fflush(stderr);
}

/* PCI identities remain comparable when each rank has a different visible
   CUDA ordinal. No GPU admission lock is held across these collectives. */
static int SetPro_GpuSharing(MPI_Comm node_comm)
{
  enum {identity_bytes=64};
  int ranks=1,device,sharing=0;
  char identity[identity_bytes]={0};
  MPI_Comm_size(node_comm,&ranks);
  if(ranks<1)return 1;
  if(gpu_rank_device_usable()){
    if(cudaGetDevice(&device)==cudaSuccess &&
       cudaDeviceGetPCIBusId(identity,sizeof(identity),device)!=cudaSuccess)
      identity[0]='\0';
    (void)cudaGetLastError();
  }
  char *identities=SetPro_checked_malloc((size_t)ranks*identity_bytes);
  MPI_Allgather(identity,identity_bytes,MPI_CHAR,identities,identity_bytes,MPI_CHAR,node_comm);
  if(identity[0])for(int rank=0;rank<ranks;rank++)
    sharing+=strcmp(identity,identities+(size_t)rank*identity_bytes)==0;
  free(identities);
  return sharing>0?sharing:ranks;
}

static size_t SetPro_GpuBatchBudget(int sharing)
{
  int device;
  size_t available=0,total=0;
  const size_t reserve=(size_t)256*1024*1024;
  /* Keep native-FP64 A100/H100 batch sizes. On GeForce, fit simultaneous
     MPS preparations instead of making large batches evict their peers. */
  if(!gpu_rank_device_usable() || cudaGetDevice(&device)!=cudaSuccess ||
     !openmx_gpu_device_weak_fp64(device) ||
     OpenMX_GpuMemGetInfo(&available,&total)!=cudaSuccess)return SIZE_MAX;
  return available>reserve?(available-reserve)/(size_t)(sharing>0?sharing:1):0;
}

/* This rank's (Mc_AN, h_AN) pairs, the list every DS_VNA / HVNA2/3 batch walks. */
static int SetPro_LocalPairs(void)
{
  int pairs=0;
  for (int mc=1; mc<=Matomnum; mc++) pairs+=FNAN[M2G[mc]]+1;
  return pairs;
}

/* One device attempt of a pair batch. Under the host SIMD policy a failure
   is not completed by the capacity task (whose host body is the serial
   device kernel): every rank learns whether any rank failed (one Allreduce,
   all ranks), and the caller hands each rank's unfinished pairs to the host
   SIMD assembly. OPENMX_HOST_SIMD=legacy keeps the capacity task. Returns
   this rank's completion; *all_done is the collective verdict. */
static int SetPro_BatchAttempt(OpenMXGpuTask *task,const char *label,int *all_done)
{
  if (OpenMX_HostSimdMode()==OPENMX_HOST_SIMD_LEGACY) {
    openmx_gpu_task_execute(mpi_comm_level1,task,1,label);
    *all_done=1;
    return 1;
  }
  const int done=!task->count || (gpu_rank_device_usable() && openmx_gpu_task_local(task,label));
  *all_done=done;
  MPI_Allreduce(MPI_IN_PLACE,all_done,1,MPI_INT,MPI_MIN,mpi_comm_level1);
  return done;
}

static int SetPro_HVNA_GpuBegin(const OpenMXReciprocalViews *reciprocal, SetProGpuContext *g, Type_DS_VNA *****DS_VNA,
                                const OpenMXVnaPreparation *preparation, int Num_RVNA)
{
  int Mc_AN,spe,node_ranks = 1,node_rank = 0;
  size_t pos = 0;
  size_t free_bytes = 0,total_bytes = 0;
  MPI_Comm node_comm = MPI_COMM_NULL;
  int requested = scf_eigen_lib_flag==GPUSOLVER;

  memset(g,0,sizeof(*g));
  g->num_proj = (List_YOUSO[35]+1)*(List_YOUSO[35]+1)*List_YOUSO[34];
  g->num_rvna = Num_RVNA;

  MPI_Allreduce(MPI_IN_PLACE,&requested,1,MPI_INT,MPI_MIN,mpi_comm_level1);
  if (!requested || !openmx_env_flag_at(OPENMX_ENV_SETPRO_GPU,1)) return 0;

  MPI_Comm_split_type(mpi_comm_level1,MPI_COMM_TYPE_SHARED,0,MPI_INFO_NULL,&node_comm);
  MPI_Comm_size(node_comm,&node_ranks);
  MPI_Comm_rank(node_comm,&node_rank);
  if (node_ranks<1) node_ranks = 1;
  if (gpu_rank_device_usable()) acc_wait_all();
  if (gpu_rank_device_usable() && cudaDeviceSynchronize()==cudaSuccess){
    acc_clear_freelists();
  }
  MPI_Barrier(node_comm);

  /* sizes of the flat local table and the halo archive */

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    int Gc_AN = M2G[Mc_AN];
    int tno = Spe_Total_NO[WhatSpecies[Gc_AN]];

    pos += (size_t)(FNAN[Gc_AN]+1)*(size_t)tno*(size_t)g->num_proj;
  }

  for (Mc_AN=Matomnum+1; Mc_AN<=Matomnum+MatomnumF; Mc_AN++){
    int Gc_AN = F_M2G[Mc_AN];
    int tno = Spe_Total_NO[WhatSpecies[Gc_AN]];

    g->halo_count += (size_t)(FNAN[Gc_AN]+1)*(size_t)tno*(size_t)g->num_proj;
  }

  g->flat_count = pos==0 ? 1 : pos;
  if (OpenMX_GpuMemGetInfo(&free_bytes,&total_bytes)!=cudaSuccess) free_bytes = 0;
  {
    const size_t reserve = (size_t)256*1024*1024;
    size_t items = 0, kslots = 0, outputs = 0;
    unsigned long long need, max_need, available = free_bytes, min_available;
    int admitted = 0, fit, global_fit;

    /* AMD's rank-turn admission, with the actual OpenACC transient arrays.
       Counting all neighbours also covers large halos that exceeded the
       old fixed 192 MiB allowance. Node-wide grouping is conservative when
       a node has several GPUs, as in the previous admission policy. */
    for (Mc_AN=1; Mc_AN<=Matomnum; ++Mc_AN){
      int gc = M2G[Mc_AN];
      int tno = Spe_Total_NO[WhatSpecies[gc]];
      for (int j=0; j<=FNAN[gc]; ++j){
        ++items;
        outputs += (size_t)tno*Spe_Total_NO[WhatSpecies[natn[gc][j]]];
        for (int kk=0; kk<=FNAN[gc]; ++kk)
          if (0<=RMI1[Mc_AN][j][kk]) ++kslots;
      }
    }
    if (kslots==0) kslots = 1;
    need = (g->flat_count + (g->halo_count==0 ? 1 : g->halo_count))*sizeof(float)
         + items*(4*sizeof(int) + sizeof(size_t)) + outputs*sizeof(double)
         + kslots*(2*sizeof(size_t) + 2*sizeof(int))
         + (size_t)SpeciesNum*Num_RVNA*sizeof(double) + (size_t)Num_RVNA*sizeof(int)
         + (size_t)64*1024*1024;
    g->device_bytes = need;
    MPI_Allreduce(&need,&max_need,1,MPI_UNSIGNED_LONG_LONG,MPI_MAX,node_comm);
    MPI_Allreduce(&available,&min_available,1,MPI_UNSIGNED_LONG_LONG,MPI_MIN,node_comm);
    if (min_available>reserve){
      unsigned long long count = (min_available-reserve)/max_need;
      admitted = count<(unsigned)node_ranks ? (int)count : node_ranks;
    }
    /* Preserve the resident owner path when every local rank fits. All
       other ranks keep host archives and pack only the projector blocks
       needed by each capacity shard; no full archive is re-uploaded. */
    g->host_archive=openmx_gpu_capacity_test_recovery() || admitted<node_ranks || !gpu_rank_device_usable();
    {
      /* Host SIMD vote (every rank, same point): this rank's archives and
         batch arrays against its share of the node's device budget, the
         figures of the admission above. Both device paths keep running when
         the share suffices up to OPENMX_HOST_SIMD_MAX_PIECES times: the
         resident archives (all ranks admitted) and the archived batch, which
         uploads one rank's whole working set at a time. When even one rank's
         working set exceeds the device (need > free - reserve), only the
         per-row capacity recovery would remain, so the stage counts as one
         the device cannot hold (the test-recovery switch keeps exercising
         the device paths). A full device keeps today's admission. */
      const size_t alone=min_available>reserve ? (size_t)(min_available-reserve) : 0;
      const size_t budget=alone/(size_t)node_ranks;
      const size_t stage_need=((size_t)need>alone && !openmx_gpu_capacity_test_recovery()) ? SIZE_MAX : (size_t)need;
      if (OpenMX_HostSimdStageOnHost("Set_ProExpn_VNA: HVNA",stage_need,budget)) {
        SetPro_note("HVNA GPU archives","%.0f MiB per rank, %d of %d node ranks admitted: host SIMD HVNA",
                    (double)need/1048576.0,admitted,node_ranks);
        MPI_Comm_free(&node_comm);
        memset(g,0,sizeof(*g));
        return 0;
      }
    }
    unsigned long long host_available=openmx_host_mem_available(),min_host,total_need;
    unsigned long long host_need=g->host_archive?need:0;
    MPI_Allreduce(&host_available,&min_host,1,MPI_UNSIGNED_LONG_LONG,MPI_MIN,node_comm);
    MPI_Allreduce(&host_need,&total_need,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,node_comm);
    fit=!min_host || total_need<=min_host/2;
    MPI_Allreduce(&fit,&global_fit,1,MPI_INT,MPI_MIN,mpi_comm_level1);
    if (!global_fit) {MPI_Comm_free(&node_comm);memset(g,0,sizeof(*g));return 0;}
    g->turn=0;g->turns=1;
  }
  MPI_Comm_free(&node_comm);

  const OpenMXHvnaInput layout_input = {
    atomnum,Matomnum,MatomnumF,M2G,F_M2G,WhatSpecies,FNAN,reciprocal->vps.neutral_energies
  };
  char layout_error[256];
  if (!SetPro_HVNA_LayoutCreate(g,preparation,&layout_input,layout_error,sizeof(layout_error))) {
    SetPro_gpu_abort(layout_error);
  }

  /* Keep the original device-only archive when all ranks fit. Rank turns
     use host archives with the RAM budget checked above. Rows are staged
     through one bounce slot sized for the largest (atom,neighbour) block. */
  if (g->host_archive){
    g->flat_host = (float*)malloc(sizeof(float)*g->flat_count);
    g->halo_host = (float*)malloc(sizeof(float)*(g->halo_count==0 ? 1 : g->halo_count));
  }
  else{
    g->flat_dev = (float*)acc_malloc(sizeof(float)*g->flat_count);
    g->halo_dev = (float*)acc_malloc(sizeof(float)*(g->halo_count==0 ? 1 : g->halo_count));
  }
  {
    size_t bounce_count = 1;

    for (spe=0; spe<SpeciesNum; spe++){
      size_t cnt = (size_t)Spe_Total_NO[spe]*(size_t)g->num_proj;
      if (bounce_count<cnt) bounce_count = cnt;
    }
    g->bounce_count = bounce_count;
    g->bounce = (float*)SetPro_checked_malloc(sizeof(float)*bounce_count);
  }
  {
    if (!g->host_archive && (!g->flat_dev || !g->halo_dev)) {
      if(g->flat_dev)acc_free(g->flat_dev);if(g->halo_dev)acc_free(g->halo_dev);
      g->flat_dev=g->halo_dev=NULL;g->host_archive=1;
      unsigned long long room=openmx_host_mem_available();
      size_t need=(g->flat_count+(g->halo_count?g->halo_count:1))*sizeof(float);
      if (!room || need<=room/(2ULL*node_ranks)) {
        g->flat_host=malloc(g->flat_count*sizeof(float));g->halo_host=malloc((g->halo_count?g->halo_count:1)*sizeof(float));
      }
    }
    int allocated = g->host_archive ? g->flat_host && g->halo_host : g->flat_dev && g->halo_dev;
    int all_allocated;
    MPI_Allreduce(&allocated,&all_allocated,1,MPI_INT,MPI_MIN,mpi_comm_level1);
    if (!all_allocated){
      if (g->flat_dev!=NULL) acc_free(g->flat_dev);
      if (g->halo_dev!=NULL) acc_free(g->halo_dev);
      acc_clear_freelists();
      free(g->flat_host);
      free(g->halo_host);
      free(g->bounce);
      openmx_core_hvna_destroy(g->layout);
      memset(g,0,sizeof(*g));
      return 0;
    }
  }

  {
    /* flatten the local direction-0 rows straight into the device archive,
       one (atom,neighbour) block at a time through the bounce slot */

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      int Gc_AN = M2G[Mc_AN];
      int tno = Spe_Total_NO[WhatSpecies[Gc_AN]];
      int k2,i;

      for (k2=0; k2<=FNAN[Gc_AN]; k2++){
        size_t base = g->mck_off[g->mck_base[Mc_AN]+k2];

        for (i=0; i<tno; i++){
          memcpy(g->bounce + (size_t)i*(size_t)g->num_proj,
                 DS_VNA[0][Mc_AN][k2][i],
                 sizeof(float)*(size_t)g->num_proj);
        }
        if (g->host_archive)
          memcpy(g->flat_host + base, g->bounce,sizeof(float)*(size_t)tno*(size_t)g->num_proj);
        else
          acc_memcpy_to_device(g->flat_dev + base, g->bounce,
                               sizeof(float)*(size_t)tno*(size_t)g->num_proj);
      }
    }
  }

  SetPro_note("HVNA GPU archives","local archive %.0f MiB, halo archive %.0f MiB on the %s",
              (double)g->flat_count*sizeof(float)/1048576.0,
              (double)g->halo_count*sizeof(float)/1048576.0,
              g->host_archive ? "host (capacity recovery)" : "device");

  g->enabled = 1;
  return 1;
}

static void SetPro_HVNA_GpuEnd(SetProGpuContext *g)
{

  if (!g->enabled) return;

  if (g->flat_dev!=NULL) acc_free(g->flat_dev);
  if (g->halo_dev!=NULL) acc_free(g->halo_dev);
  acc_clear_freelists();

  free(g->flat_host);
  free(g->halo_host);
  free(g->bounce);
  openmx_core_hvna_destroy(g->layout);
  memset(g,0,sizeof(*g));
}

/* archive the direction-0 block of the halo atom just received */
static void SetPro_HVNA_GpuArchiveHalo(SetProGpuContext *g, Type_DS_VNA *****DS_VNA, int Original_Mc_AN,
                                       int Gc_AN)
{
  int tno = Spe_Total_NO[WhatSpecies[Gc_AN]];
  int halo_idx = Original_Mc_AN - Matomnum;
  int k;

  if (halo_idx<1 || MatomnumF<halo_idx){
    SetPro_gpu_abort("Set_ProExpn_VNA GPU: halo index out of range.");
  }

  /* serial: the rows go through the shared bounce slot into the device
     archive, and acc_memcpy_to_device is not called from inside an OpenMP
     region anywhere else in this file either */
  for (k=0; k<=FNAN[Gc_AN]; k++){
    size_t base = g->halo_off[g->halo_base[halo_idx]+k];
    int i;

    for (i=0; i<tno; i++){
      memcpy(g->bounce + (size_t)i*(size_t)g->num_proj,
             DS_VNA[0][Matomnum+1][k][i],
             sizeof(float)*(size_t)g->num_proj);
    }
    if (g->host_archive)
      memcpy(g->halo_host + base, g->bounce,sizeof(float)*(size_t)tno*(size_t)g->num_proj);
    else
      acc_memcpy_to_device(g->halo_dev + base, g->bounce,
                           sizeof(float)*(size_t)tno*(size_t)g->num_proj);
  }
}

/* the unified batch over every (local atom, neighbour) pair */
static int SetPro_HVNA_GpuRun_Part(int first,int last,SetProGpuContext *g,double ****HVNA)
{
  const int fp32 = openmx_fp32_enabled();
  const int num_proj = g->num_proj;
  const int num_rvna = g->num_rvna;
  int Mc_AN,j,k;
  int nitems = 0,kslots = 0;
  size_t out_count = 0;
  int *item_atom,*item_j,*item_koff,*item_kn,*item_tnoC,*item_tnoH;
  size_t *item_out;
  size_t *krowA,*krowB;
  int *krow_halo,*krow_ene;
  double *out;
  int p;
  size_t opos = 0;
  int kpos = 0, ok = 0;
  void *arena = NULL, *device_out = NULL;

  /* count */
  for (Mc_AN=first; Mc_AN<last; Mc_AN++){
    int Gc_AN = M2G[Mc_AN];
    int tnoC = Spe_Total_NO[WhatSpecies[Gc_AN]];

    for (j=0; j<=FNAN[Gc_AN]; j++){
      int tnoH = Spe_Total_NO[WhatSpecies[natn[Gc_AN][j]]];

      nitems++;
      out_count += (size_t)tnoC*(size_t)tnoH;
      for (k=0; k<=FNAN[Gc_AN]; k++){
        if (0<=RMI1[Mc_AN][j][k]) kslots++;
      }
    }
  }

  if (nitems==0) return 1;

  double hvna_t0;
  dtime(&hvna_t0);
  SetPro_note("HVNA GPU batch begin","%d (atom,neighbour) items, %d k slots, %.0f MiB of outputs",
              nitems,kslots,(double)out_count*sizeof(double)/1048576.0);

  item_atom = (int*)SetPro_checked_malloc(sizeof(int)*(size_t)nitems);
  item_j = (int*)SetPro_checked_malloc(sizeof(int)*(size_t)nitems);
  item_koff = (int*)SetPro_checked_malloc(sizeof(int)*(size_t)nitems);
  item_kn = (int*)SetPro_checked_malloc(sizeof(int)*(size_t)nitems);
  item_tnoC = (int*)SetPro_checked_malloc(sizeof(int)*(size_t)nitems);
  item_tnoH = (int*)SetPro_checked_malloc(sizeof(int)*(size_t)nitems);
  item_out = (size_t*)SetPro_checked_malloc(sizeof(size_t)*(size_t)nitems);
  krowA = (size_t*)SetPro_checked_malloc(sizeof(size_t)*(size_t)(kslots==0 ? 1 : kslots));
  krowB = (size_t*)SetPro_checked_malloc(sizeof(size_t)*(size_t)(kslots==0 ? 1 : kslots));
  krow_halo = (int*)SetPro_checked_malloc(sizeof(int)*(size_t)(kslots==0 ? 1 : kslots));
  krow_ene = (int*)SetPro_checked_malloc(sizeof(int)*(size_t)(kslots==0 ? 1 : kslots));
  out = (double*)SetPro_checked_malloc(sizeof(double)*(out_count==0 ? 1 : out_count));

  /* build */
  p = 0;
  for (Mc_AN=first; Mc_AN<last; Mc_AN++){
    int Gc_AN = M2G[Mc_AN];
    int tnoC = Spe_Total_NO[WhatSpecies[Gc_AN]];

    for (j=0; j<=FNAN[Gc_AN]; j++){
      int jg = natn[Gc_AN][j];
      int Mj_AN = F_G2M[jg];
      int tnoH = Spe_Total_NO[WhatSpecies[jg]];

      item_atom[p] = Mc_AN;
      item_j[p] = j;
      item_koff[p] = kpos;
      item_tnoC[p] = tnoC;
      item_tnoH[p] = tnoH;
      item_out[p] = opos;
      opos += (size_t)tnoC*(size_t)tnoH;

      for (k=0; k<=FNAN[Gc_AN]; k++){
        int kl = RMI1[Mc_AN][j][k];

        if (kl<0) continue;
        krowA[kpos] = g->mck_off[g->mck_base[Mc_AN]+k];
        krow_ene[kpos] = WhatSpecies[natn[Gc_AN][k]]*num_rvna;
        if (Mj_AN<=Matomnum){
          krowB[kpos] = g->mck_off[g->mck_base[Mj_AN]+kl];
          krow_halo[kpos] = 0;
        }
        else{
          krowB[kpos] = g->halo_off[g->halo_base[Mj_AN-Matomnum]+kl];
          krow_halo[kpos] = 1;
        }
        kpos++;
      }
      item_kn[p] = kpos - item_koff[p];
      p++;
    }
  }

  if (p!=nitems || kpos!=kslots || opos!=out_count){
    SetPro_gpu_abort("Set_ProExpn_VNA GPU: inconsistent batch.");
  }

  {
    const float *flat = g->flat_dev;
    const float *halo = g->halo_dev;
    double *ene = g->ene;
    int *l2p1 = g->l2p1;
    size_t ene_len = (size_t)SpeciesNum*(size_t)num_rvna;
    const int nitems_c = nitems;
    const size_t kslots_c = (size_t)(kslots==0 ? 1 : kslots);
    const size_t out_c = out_count;

    /* Explicit allocation permits a capacity retry if other ranks consume
       memory after archive admission. No result is published before success. */
    size_t length[]={(nitems_c)*sizeof(int),(nitems_c)*sizeof(int),(nitems_c)*sizeof(int),(nitems_c)*sizeof(int),(nitems_c)*sizeof(size_t),(kslots_c)*sizeof(size_t),(kslots_c)*sizeof(size_t),(kslots_c)*sizeof(int),(kslots_c)*sizeof(int),(ene_len)*sizeof(double),(num_rvna)*sizeof(int),(out_c)*sizeof(double)};
    const void *host[]={item_koff,item_kn,item_tnoC,item_tnoH,item_out,krowA,krowB,krow_halo,krow_ene,ene,l2p1,out};
    size_t off[12],bytes=0;void *dev[12];
    for(int i=0;i<12;i++){off[i]=bytes;bytes+=(length[i]+15)&~(size_t)15;}
    if(cudaMalloc(&arena,bytes)!=cudaSuccess) goto hvna_cleanup;
    for(int i=0;i<12;i++)dev[i]=(char*)arena+off[i];
    device_out=dev[11];
    for(int i=0;i<11;i++)if(cudaMemcpy(dev[i],host[i],length[i],cudaMemcpyHostToDevice)!=cudaSuccess)goto hvna_cleanup;
    {
      int *item_koff=(int*)dev[0];
      int *item_kn=(int*)dev[1];
      int *item_tnoC=(int*)dev[2];
      int *item_tnoH=(int*)dev[3];
      size_t *item_out=(size_t*)dev[4];
      size_t *krowA=(size_t*)dev[5];
      size_t *krowB=(size_t*)dev[6];
      int *krow_halo=(int*)dev[7];
      int *krow_ene=(int*)dev[8];
      double *ene=(double*)dev[9];
      int *l2p1=(int*)dev[10];
      double *out=(double*)dev[11];
#pragma acc parallel loop gang vector_length(128) \
    firstprivate(fp32) deviceptr(flat,halo,item_koff,item_kn,item_tnoC,item_tnoH,item_out,krowA,krowB,krow_halo,krow_ene,ene,l2p1,out)
      for (int pp=0; pp<nitems_c; pp++){
        const int tnoC = item_tnoC[pp];
        const int tnoH = item_tnoH[pp];
        const int k_off = item_koff[pp];
        const int k_n = item_kn[pp];
        const size_t out_off = item_out[pp];
        const int mn = tnoC*tnoH;

#pragma acc loop vector
        for (int idx=0; idx<mn; idx++){
          const int m = idx/tnoH;
          const int n = idx - m*tnoH;
          double nlh = 0.0;

          for (int kk=0; kk<k_n; kk++){
            const float *a = flat + krowA[k_off+kk] + (size_t)m*(size_t)num_proj;
            const float *b = (krow_halo[k_off+kk]
                              ? (halo + krowB[k_off+kk])
                              : (flat + krowB[k_off+kk])) + (size_t)n*(size_t)num_proj;
            const double *el = ene + krow_ene[k_off+kk];
            double sum = 0.0;
            int L = 0;

            for (int L1=0; L1<num_rvna; L1++){
              const int g1 = l2p1[L1];
              double tmp0 = 0.0;

              for (int l3=0; l3<g1; l3++){
                tmp0 += fp32 ? (double)(a[L]*b[L]) : (double)a[L]*(double)b[L];
                L++;
              }
              sum += el[L1]*tmp0;
            }

            nlh += sum;
          }

          out[out_off + (size_t)idx] = nlh;
        }
      }
    }
  }

  if(cudaMemcpy(out,device_out,out_count*sizeof(double),cudaMemcpyDeviceToHost)!=cudaSuccess)goto hvna_cleanup;

  /* damping and store, exactly as the host loops do */

  for (p=0; p<nitems; p++){
    int mc = item_atom[p];
    int j2 = item_j[p];
    int Gc_AN = M2G[mc];
    int jg = natn[Gc_AN][j2];
    int tnoC = item_tnoC[p];
    int tnoH = item_tnoH[p];
    size_t out_off = item_out[p];
    double rcut = Spe_Atom_Cut1[WhatSpecies[Gc_AN]] + Spe_Atom_Cut1[WhatSpecies[jg]];
    double dmp = dampingF(rcut,Dis[Gc_AN][j2]);
    int m,n;

    for (m=0; m<tnoC; m++){
      for (n=0; n<tnoH; n++){
        HVNA[mc][j2][m][n] = dmp*out[out_off + (size_t)m*tnoH + n];
      }
    }
  }

  ok=1;
hvna_cleanup:
  if(arena)cudaFree(arena);
  (void)cudaGetLastError();
  free(out);
  free(krow_ene);
  free(krow_halo);
  free(krowB);
  free(krowA);
  free(item_out);
  free(item_tnoH);
  free(item_tnoC);
  free(item_kn);
  free(item_koff);
  free(item_j);
  free(item_atom);
  {
    double t1;
    dtime(&t1);
    SetPro_note("HVNA GPU batch end","%.2f s",t1-hvna_t0);
  }
  return ok;
}

typedef struct {int num_proj,num_rvna,fp32;} SetProHVNACapacityParameters;
static void SetPro_HVNA_capacity_kernel(size_t first,size_t count,const void *parameters,void *const v[],int gpu)
{
  const SetProHVNACapacityParameters *p=parameters;
  const int np=p->num_proj,nr=p->num_rvna,fp32=p->fp32;
  const size_t *offset=v[0];
  const float *a=openmx_gpu_task_ragged_data(v[0],count);
  const float *b=openmx_gpu_task_ragged_data(v[1],count);
  const size_t *species_offset=v[2];
  const int *species=openmx_gpu_task_ragged_data(v[2],count);
  const double *ene=v[3],*damp=v[5];const int *length=v[4];double *out=v[6];
  (void)first;
/* Each vector lane owns a complete output row. Keeping the projector
     loops sequential also preserves the cumulative l index across radial
     channels; auto-vectorizing the inner loop changes that index's live-out
     value for long projector lists. */
#pragma acc parallel loop gang vector deviceptr(offset,a,b,species_offset,species,ene,length,damp,out) if(gpu)
  for(size_t row=0;row<count;row++){
    double total=0;size_t pos=offset[row];
#pragma acc loop seq
    for(size_t k=species_offset[row];k<species_offset[row+1];k++,pos+=np){
      double sum=0;int l=0;
#pragma acc loop seq
      for(int r=0;r<nr;r++){
        double term=0;
#pragma acc loop seq
        for(int m=0;m<length[r];m++,l++)
          term+=fp32?(double)(a[pos+l]*b[pos+l]):(double)a[pos+l]*(double)b[pos+l];
        sum+=ene[(size_t)species[k]*nr+r]*term;
      }
      total+=sum;
    }
    out[row]=damp[row]*total;
  }
}

static void SetPro_HVNA_archive_row(float *dest,const float *host,const float *device,size_t offset,int n)
{
  if(host)memcpy(dest,host+offset,(size_t)n*sizeof(float));
  else if(cudaMemcpy(dest,device+offset,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost)!=cudaSuccess)
    SetPro_gpu_abort("Set_ProExpn_VNA: cannot recover an existing projector archive.");
}

/* The node may not fit every rank's archives concurrently even though each
   rank fits on its own. In that case, reuse the indexed contraction under
   the device admission lock before trying row-wise capacity recovery.
   Expanding every (m,n,k) product into separate input rows duplicates the
   same projectors many times and can transfer terabytes for a modest cell.
   No collective may run while this lock is held: other ranks may be waiting
   for the same GPU. The caller joins the recovery collectives afterwards. */
static int SetPro_HVNA_GpuRun_Archived(SetProGpuContext *g,double ****HVNA)
{
  int device,ok=0;
  size_t free_bytes=0,total_bytes=0;
  const size_t reserve=(size_t)256*1024*1024;
  void *admission;
  SetProGpuContext staged=*g;
  staged.flat_dev=NULL;
  staged.halo_dev=NULL;

  if (!Matomnum) return 1;
  if (!gpu_rank_device_usable() || openmx_gpu_capacity_test_recovery() ||
      cudaGetDevice(&device)!=cudaSuccess) return 0;
  admission=openmx_mgpu_admit_devices(&device,1);
  if (!admission) return 0;

  acc_wait_all();
  if (cudaDeviceSynchronize()!=cudaSuccess) goto archived_cleanup;
  acc_clear_freelists();
  if (OpenMX_GpuMemGetInfo(&free_bytes,&total_bytes)!=cudaSuccess ||
      free_bytes<=reserve || g->device_bytes>free_bytes-reserve)
    goto archived_cleanup;

  if (cudaMalloc((void**)&staged.flat_dev,g->flat_count*sizeof(float))!=cudaSuccess ||
      cudaMalloc((void**)&staged.halo_dev,(g->halo_count ? g->halo_count : 1)*sizeof(float))!=cudaSuccess)
    goto archived_cleanup;
  if (cudaMemcpy(staged.flat_dev,g->flat_host,g->flat_count*sizeof(float),cudaMemcpyHostToDevice)!=cudaSuccess ||
      (g->halo_count && cudaMemcpy(staged.halo_dev,g->halo_host,g->halo_count*sizeof(float),cudaMemcpyHostToDevice)!=cudaSuccess))
    goto archived_cleanup;

  SetPro_note("HVNA archived GPU batch","uploading each archive once (%.0f MiB)",
              (double)(g->flat_count+g->halo_count)*sizeof(float)/1048576.0);
  ok=SetPro_HVNA_GpuRun_Part(1,Matomnum+1,&staged,HVNA);

archived_cleanup:
  if (staged.halo_dev) cudaFree(staged.halo_dev);
  if (staged.flat_dev) cudaFree(staged.flat_dev);
  (void)cudaGetLastError();
  openmx_mgpu_release_devices(admission);
  return ok;
}

/* Device HVNA of this rank. Under the host SIMD policy the device verdict is
   returned and SetPro_HVNA_DispatchRun completes a failed rank on the host SIMD
   path from the archives; OPENMX_HOST_SIMD=legacy keeps the capacity recovery
   loop below (collective, every rank) and always reports completion. */
static int SetPro_HVNA_GpuRun(SetProGpuContext *g, double ****HVNA)
{
  int done=g->host_archive ? SetPro_HVNA_GpuRun_Archived(g,HVNA)
                          : SetPro_HVNA_GpuRun_Part(1,Matomnum+1,g,HVNA);
  if (OpenMX_HostSimdMode()!=OPENMX_HOST_SIMD_LEGACY) return done;
  int mc=done?Matomnum+1:1,j=0;size_t row=0;
  SetProHVNACapacityParameters p={g->num_proj,g->num_rvna,openmx_fp32_enabled()};
  /* Only recovery duplicates projector rows, and at most one bounded tile.
     Device archives can be read row-wise after a late transient OOM. */
  for(;;){
    int remaining=mc<=Matomnum,any;
    MPI_Allreduce(&remaining,&any,1,MPI_INT,MPI_MAX,mpi_comm_level1);
    if(!any)break;
    size_t n=0,kn=0,*offset=NULL,*species_offset=NULL;
    float *a=NULL,*b=NULL;int *species=NULL;double *damp=NULL,*out=NULL;
    int gc=0,nh=0,mj=0;
    if(remaining){
      gc=M2G[mc];int jg=natn[gc][j];mj=F_G2M[jg];nh=Spe_Total_NO[WhatSpecies[jg]];
      for(int k=0;k<=FNAN[gc];k++)if(RMI1[mc][j][k]>=0)kn++;
      size_t total=(size_t)Spe_Total_NO[WhatSpecies[gc]]*nh;
      size_t per=kn*(2*(size_t)p.num_proj*sizeof(float)+sizeof(int))+2*sizeof(size_t)+2*sizeof(double);
      size_t limit=67108864,available=openmx_host_mem_available();
      int ranks=openmx_gpu_local_size_noncollective();if(ranks<1)ranks=1;
      if(available && available/(4ULL*ranks)<limit)limit=available/(4ULL*ranks);
      n=total-row;if(n>2048)n=2048;if(per && n>limit/per)n=limit/per;if(!n)n=1;
      for(;;){
        offset=malloc((n+1)*sizeof(size_t));species_offset=malloc((n+1)*sizeof(size_t));
        a=malloc((n*kn*p.num_proj?n*kn*p.num_proj:1)*sizeof(float));b=malloc((n*kn*p.num_proj?n*kn*p.num_proj:1)*sizeof(float));
        species=malloc((n*kn?n*kn:1)*sizeof(int));damp=malloc(n*sizeof(double));out=malloc(n*sizeof(double));
        if(offset && species_offset && a && b && species && damp && out)break;
        free(offset);free(species_offset);free(a);free(b);free(species);free(damp);free(out);
        if(n==1)SetPro_gpu_abort("Set_ProExpn_VNA: no host memory for one projector row.");
        n=(n+1)/2;
      }
      double damping=dampingF(Spe_Atom_Cut1[WhatSpecies[gc]]+Spe_Atom_Cut1[WhatSpecies[jg]],Dis[gc][j]);
      for(size_t i=0;i<=n;i++){offset[i]=i*kn*p.num_proj;species_offset[i]=i*kn;}
      for(size_t i=0;i<n;i++){
        int m=(row+i)/nh,q=(row+i)%nh;size_t kpos=0;damp[i]=damping;
        for(int k=0;k<=FNAN[gc];k++){
          int kl=RMI1[mc][j][k];if(kl<0)continue;
          size_t ao=g->mck_off[g->mck_base[mc]+k]+(size_t)m*p.num_proj;
          SetPro_HVNA_archive_row(a+(i*kn+kpos)*p.num_proj,g->flat_host,g->flat_dev,ao,p.num_proj);
          size_t bo=(mj<=Matomnum?g->mck_off[g->mck_base[mj]+kl]:g->halo_off[g->halo_base[mj-Matomnum]+kl])+(size_t)q*p.num_proj;
          SetPro_HVNA_archive_row(b+(i*kn+kpos)*p.num_proj,mj<=Matomnum?g->flat_host:g->halo_host,mj<=Matomnum?g->flat_dev:g->halo_dev,bo,p.num_proj);
          species[i*kn+kpos++]=WhatSpecies[natn[gc][k]];
        }
      }
    }
    OpenMXGpuTask task={0};task.count=n;task.parameters=&p;task.parameter_bytes=sizeof(p);task.inputs=6;task.outputs=1;task.kernel=SetPro_HVNA_capacity_kernel;
    task.view[0]=(OpenMXGpuTaskView){a,0,0,offset,sizeof(float)};
    task.view[1]=(OpenMXGpuTaskView){b,0,0,offset,sizeof(float)};
    task.view[2]=(OpenMXGpuTaskView){species,0,0,species_offset,sizeof(int)};
    task.view[3]=(OpenMXGpuTaskView){g->ene,(size_t)SpeciesNum*p.num_rvna*sizeof(double),0};
    task.view[4]=(OpenMXGpuTaskView){g->l2p1,(size_t)p.num_rvna*sizeof(int),0};
    task.view[5]=(OpenMXGpuTaskView){damp,0,sizeof(double)};task.view[6]=(OpenMXGpuTaskView){out,0,sizeof(double)};
    openmx_gpu_task_execute(mpi_comm_level1,&task,1,"VNA-contraction-capacity");
    for(size_t i=0;i<n;i++)HVNA[mc][j][(row+i)/nh][(row+i)%nh]=out[i];
    free(offset);free(species_offset);free(a);free(b);free(species);free(damp);free(out);
    if(remaining){
      row+=n;
      if(row==(size_t)Spe_Total_NO[WhatSpecies[gc]]*nh){row=0;if(++j>FNAN[gc]){j=0;mc++;}}
    }
  }
  return 1;
}

/* ------------------------------------------------------------------ */
/* Batched device construction of DS_VNA (the <chi0|P> expansion and
   its derivatives).

   Numerically sensitive special functions never run on the device:
   the Gaunt coefficients (factorial-based) are tabulated once on the
   host, the complex spherical harmonics of every pair direction are
   evaluated on the host with the original long-double routine, and the
   radial Bessel tables are the very host arrays.  Only the downward
   Bessel recurrence, the quadrature dot products and the unitary
   transforms run on the device, each keeping the exact CPU loop order.
   Pairs are processed in chunks; the results overwrite DS_VNA rows on
   the host exactly like the original per-pair loop.                   */
/* ------------------------------------------------------------------ */

#define SETPRO_BES_LMAX 30

#pragma acc routine seq
static void SetPro_Spherical_Bessel2_dev(double x, int lmax, double *sb, double *dsb)
{
  /* device copy of Spherical_Bessel2() (xmin==0.0 branch structure) */

  int m,n,nmax;
  double tsb[SETPRO_BES_LMAX+10];
  double vsb0,vsb1,vsb2,vsbi,invx;
  double j0,j1,sf,tmp,si,co,ix,ix2;

  nmax = lmax + 3*x + 20;
  if (nmax<100) nmax = 100;

  if ( 0.0 < x ){

    invx = 1.0/x;

    vsb0 = 0.0;
    vsb1 = 1.0e-14;
    vsbi = 0.0;

    for ( n=nmax-1; (lmax+2)<n; n-- ){

      vsb2 = (2.0*n + 1.0)*invx*vsb1 - vsb0;

      if (1.0e+250<vsb2){
        tmp = 1.0/vsb2;
        vsb2 *= tmp;
        vsb1 *= tmp;
      }

      vsbi = vsb0;
      vsb0 = vsb1;
      vsb1 = vsb2;
    }

    n = lmax + 3;
    tsb[n-1] = vsb1;
    tsb[n  ] = vsb0;
    tsb[n+1] = vsbi;

    tmp = tsb[n-1];
    tsb[n-1] /= tmp;
    tsb[n  ] /= tmp;

    for ( n=lmax+2; 0<n; n-- ){

      tsb[n-1] = (2.0*n + 1.0)*invx*tsb[n] - tsb[n+1];

      if (1.0e+250<tsb[n-1]){
        tmp = tsb[n-1];
        for (m=n-1; m<=lmax+1; m++){
          tsb[m] /= tmp;
        }
      }
    }

    si = sin(x);
    co = cos(x);
    ix = 1.0/x;
    ix2 = ix*ix;
    j0 = si*ix;
    j1 = si*ix*ix - co*ix;

    if (fabs(tsb[1])<fabs(tsb[0])) sf = j0/tsb[0];
    else                           sf = j1/tsb[1];

    for ( n=0; n<=lmax+1; n++ ){
      sb[n] = tsb[n]*sf;
    }

    dsb[0] = co*ix - si*ix*ix;
    for ( n=1; n<=lmax; n++ ){
      dsb[n] = ( (double)n*sb[n-1] - (double)(n+1.0)*sb[n+1] )/(2.0*(double)n + 1.0);
    }
  }

  else {

    /* dsb[lmax] also reads sb[lmax+1], including at x == 0. */
    for ( n=0; n<=lmax+1; n++ ){
      sb[n] = 0.0;
    }
    sb[0] = 1.0;

    dsb[0] = 0.0;
    for ( n=1; n<=lmax; n++ ){
      dsb[n] = ( (double)n*sb[n-1] - (double)(n+1.0)*sb[n+1] )/(2.0*(double)n + 1.0);
    }
  }
}

/* Pair-chunk length of the DS_VNA device pipeline.  Shared by the sizing
   estimate below and by SetPro_DSVNA_GpuRun so the two cannot drift apart --
   an estimate computed against a different CHUNK is worse than no estimate. */
#define SETPRO_DSVNA_CHUNK 256

/* Device bytes the "#pragma acc data" region in SetPro_DSVNA_GpuRun will map.
   Deliberately mirrors that clause term for term, in the same order, so the two
   can be diffed by eye. */
static size_t SetPro_DSVNA_DeviceBytes(const SetProGpu2Context *g)
{
  const size_t CH = (size_t)g->chunk;
  const int nM0 = 2*g->l0max_all + 1;
  const int nM1 = 2*g->l1max + 1;
  const int nLL = g->lfi_max + 1;
  const int nm  = 2*g->lfi_max + 1;
  const int nsh = (g->lfi_max+1)*(g->lfi_max+1);
  const int lc  = (g->l0max_all<g->l1max) ? g->l1max : g->l0max_all;

  const size_t pro_len  = (size_t)SpeciesNum*g->cmb_max*GL_Mesh;
  const size_t vnab_len = (size_t)SpeciesNum*g->num_rvna*GL_Mesh;
  const size_t gnt_len  = (size_t)(g->l0max_all+1)*nM0*(g->l1max+1)*nM1*nLL*nm;
  const size_t c2r_len  = 2*(size_t)g->c2r_off[lc+1];

  const size_t sphb_len = CH*(size_t)nLL*(size_t)GL_Mesh;
  const size_t sum_len  = CH*(size_t)nLL*(size_t)g->cmb_max*(size_t)g->num_rvna;
  const size_t elem_per_pair = (size_t)g->cmb_max*nM0*g->num_rvna*nM1;
  const size_t tmpnl_len = CH*elem_per_pair*8;
  const size_t out_len   = CH*4*(size_t)g->mn_max;

  size_t bytes = 0;

  /* copyin */
  bytes += (2*pro_len + vnab_len + gnt_len + c2r_len)*sizeof(double);
  bytes += (size_t)GL_Mesh*sizeof(double);                       /* GL_NormK   */
  bytes += (size_t)(lc+2)*sizeof(int);                           /* c2r_off    */
  bytes += (size_t)(2*SpeciesNum
                    + 2*(size_t)SpeciesNum*g->cmb_max
                    + 2*(size_t)g->num_rvna)*sizeof(int);

  /* create */
  bytes += (2*sphb_len + 2*sum_len + tmpnl_len)*sizeof(double);
  bytes += out_len*sizeof(float);
  bytes += CH*5*sizeof(double) + CH*4*sizeof(int);               /* pr_*       */
  bytes += CH*(size_t)nsh*6*sizeof(double);                      /* sh_tab     */

  return bytes;
}

static int SetPro_DSVNA_GpuBegin(const OpenMXReciprocalViews *reciprocal, const dcomplex *conversion, size_t conversion_width, SetProGpu2Context *g,
    const OpenMXVnaPreparationBuffers *preparation)
{
  const int *VNA_List = preparation->projector_angular;
  const int *VNA_List2 = preparation->projector_radial;
  const int Num_RVNA = preparation->projectors;
  const int species = preparation->species, points = preparation->points;
  const double *Bessel_Pro00 = preparation->weighted[0];
  const double *Bessel_Pro01 = preparation->weighted[1];

  int spe,L0,Mul0,L,i,k,c;
  size_t free_bytes = 0,total_bytes = 0;
  int node_ranks = 1,node_rank = 0,device_ranks;
  MPI_Comm node_comm = MPI_COMM_NULL;

  memset(g,0,sizeof(*g));

  int requested=scf_eigen_lib_flag==GPUSOLVER;
  MPI_Allreduce(MPI_IN_PLACE,&requested,1,MPI_INT,MPI_MIN,mpi_comm_level1);
  if (!requested) return 0;
  if (!openmx_env_flag_at(OPENMX_ENV_SETPRO_GPU,1)) return 0;

  MPI_Comm_split_type(mpi_comm_level1,MPI_COMM_TYPE_SHARED,0,MPI_INFO_NULL,&node_comm);
  MPI_Comm_size(node_comm,&node_ranks);
  MPI_Comm_rank(node_comm,&node_rank);
  if (node_ranks<1) node_ranks = 1;
  device_ranks=SetPro_GpuSharing(node_comm);
  MPI_Comm_free(&node_comm);

  g->num_proj = preparation->projector_components;
  g->num_rvna = Num_RVNA;
  g->l1max = preparation->projector_lmax;

  g->sp_lfi = (int*)SetPro_checked_malloc(sizeof(int)*species);
  g->sp_ncmb = (int*)SetPro_checked_malloc(sizeof(int)*species);

  g->l0max_all = 0;
  g->cmb_max = 0;
  for (spe=0; spe<species; spe++){
    int Lmax = 0,ncmb = 0;

    for (L=0; L<Num_RVNA; L++){
      if (Lmax<VNA_List[L]) Lmax = VNA_List[L];
    }
    if (preparation->maximum_angular[spe]<Lmax) g->sp_lfi[spe] = 2*Lmax;
    else                          g->sp_lfi[spe] = 2*preparation->maximum_angular[spe];
    if (g->lfi_max<g->sp_lfi[spe]) g->lfi_max = g->sp_lfi[spe];
    if (g->l0max_all<preparation->maximum_angular[spe]) g->l0max_all = preparation->maximum_angular[spe];

    for (L0=0; L0<=preparation->maximum_angular[spe]; L0++){
      ncmb += preparation->basis_counts[(size_t)spe*preparation->angular_channels+L0];
    }
    g->sp_ncmb[spe] = ncmb;
    if (g->cmb_max<ncmb) g->cmb_max = ncmb;
  }
  g->blk_max = g->cmb_max*Num_RVNA;

  g->cmb_L0 = (int*)SetPro_checked_malloc(sizeof(int)*(size_t)species*g->cmb_max);
  g->cmb_num0 = (int*)SetPro_checked_malloc(sizeof(int)*(size_t)species*g->cmb_max);
  g->vna_L1 = (int*)SetPro_checked_malloc(sizeof(int)*Num_RVNA);
  g->vna_num1 = (int*)SetPro_checked_malloc(sizeof(int)*Num_RVNA);
  g->pro00 = (double*)SetPro_checked_malloc(sizeof(double)*(size_t)species*g->cmb_max*points);
  g->pro01 = (double*)SetPro_checked_malloc(sizeof(double)*(size_t)species*g->cmb_max*points);
  g->vnab = (double*)SetPro_checked_malloc(sizeof(double)*(size_t)species*Num_RVNA*points);

  {
    int mn_max = 0;

    for (spe=0; spe<species; spe++){
      int num0 = 0;

      c = 0;
      for (L0=0; L0<=preparation->maximum_angular[spe]; L0++){
        for (Mul0=0; Mul0<preparation->basis_counts[(size_t)spe*preparation->angular_channels+L0]; Mul0++){
          g->cmb_L0[spe*g->cmb_max+c] = L0;
          g->cmb_num0[spe*g->cmb_max+c] = num0;
          num0 += 2*L0 + 1;
          c++;
        }
      }
      for (; c<g->cmb_max; c++){
        g->cmb_L0[spe*g->cmb_max+c] = -1;
        g->cmb_num0[spe*g->cmb_max+c] = 0;
      }
      if (mn_max<num0*g->num_proj) mn_max = num0*g->num_proj;
    }
    g->mn_max = mn_max;
  }

  {
    int num1 = 0;

    for (L=0; L<Num_RVNA; L++){
      g->vna_L1[L] = VNA_List[L];
      g->vna_num1[L] = num1;
      num1 += 2*VNA_List[L] + 1;
    }
  }

  for (spe=0; spe<species; spe++){
    c = 0;
    for (L0=0; L0<=preparation->maximum_angular[spe]; L0++){
      for (Mul0=0; Mul0<preparation->basis_counts[(size_t)spe*preparation->angular_channels+L0]; Mul0++){
        const size_t offset = openmx_core_vna_radial_offset(preparation,spe,L0,Mul0);
        for (i=0; i<points; i++){
          g->pro00[((size_t)spe*g->cmb_max+c)*points+i] = Bessel_Pro00[offset+i];
          g->pro01[((size_t)spe*g->cmb_max+c)*points+i] = Bessel_Pro01[offset+i];
        }
        c++;
      }
    }
    for (L=0; L<Num_RVNA; L++){
      for (i=0; i<points; i++){
        g->vnab[((size_t)spe*g->num_rvna+L)*points+i]
          = openmx_radial_row4(&reciprocal->neutral_transform,spe,VNA_List[L],VNA_List2[L])[i];
      }
    }
  }

  /* Gaunt table: host factorial arithmetic, once */

  {
    const int nL0 = g->l0max_all + 1;
    const int nM0 = 2*g->l0max_all + 1;
    const int nL1 = g->l1max + 1;
    const int nM1 = 2*g->l1max + 1;
    const int nLL = g->lfi_max + 1;
    const int nm = 2*g->lfi_max + 1;
    size_t gsz = (size_t)nL0*nM0*nL1*nM1*nLL*nm;
    int L1,M0,M1,LL,m;

    g->gnt = (double*)SetPro_checked_malloc(sizeof(double)*gsz);
    for (i=0; i<(int)gsz; i++) g->gnt[i] = 0.0;

    for (L0=0; L0<nL0; L0++){
      for (M0=-L0; M0<=L0; M0++){
        for (L1=0; L1<nL1; L1++){
          for (M1=-L1; M1<=L1; M1++){
            for (LL=0; LL<nLL; LL++){
              for (m=-LL; m<=LL; m++){
                if (M1==M0-m && abs(L1-LL)<=L0 && L0<=(L1+LL)){
                  size_t idx = ((((((size_t)L0*nM0 + (M0+g->l0max_all))*nL1 + L1)*nM1
                                  + (M1+g->l1max))*nLL + LL)*nm + (m+g->lfi_max));
                  g->gnt[idx] = Gaunt(L0,M0,L1,M1,LL,m);
                }
              }
            }
          }
        }
      }
    }
  }

  /* packed real-harmonic conversion up to max(l0max, l1max) */

  {
    const int lc = (g->l0max_all<g->l1max) ? g->l1max : g->l0max_all;
    size_t pos = 0;
    int Lc,M;

    g->c2r_off = (int*)SetPro_checked_malloc(sizeof(int)*(lc+2));
    for (Lc=0; Lc<=lc; Lc++){
      g->c2r_off[Lc] = (int)pos;
      pos += (size_t)(2*Lc+1)*(2*Lc+1);
    }
    g->c2r_off[lc+1] = (int)pos;
    g->c2r = (double*)SetPro_checked_malloc(sizeof(double)*2*pos);
    for (Lc=0; Lc<=lc; Lc++){
      for (M=0; M<(2*Lc+1); M++){
        for (k=0; k<(2*Lc+1); k++){
          size_t idx = (size_t)g->c2r_off[Lc] + (size_t)M*(2*Lc+1) + k;
          g->c2r[2*idx  ] = conversion[((size_t)(Lc)*conversion_width+(M))*conversion_width+(k)].r;
          g->c2r[2*idx+1] = conversion[((size_t)(Lc)*conversion_width+(M))*conversion_width+(k)].i;
        }
      }
    }
  }

  /* The common task allocator selects owner, local peers, remote nodes,
     then CPU from the exact batch footprint. Keep host lookup tables. */
  g->chunk = SETPRO_DSVNA_CHUNK;
  const size_t batch_budget=SetPro_GpuBatchBudget(device_ranks);
  {
    /* Host SIMD vote (every rank, same point): the footprint of one full
       batch (at most this rank's own pairs) against the same per-rank budget
       the halving below uses. Halving it more than OPENMX_HOST_SIMD_MAX_PIECES
       times over (e.g. "batches of 1"), or an unusable device, moves DS_VNA to
       the host SIMD assembly on every rank; otherwise the admission below is
       unchanged. */
    const int pairs=SetPro_LocalPairs();
    g->chunk = pairs<SETPRO_DSVNA_CHUNK ? (pairs>0 ? pairs : 1) : SETPRO_DSVNA_CHUNK;
    const size_t need=pairs>0 ? SetPro_DSVNA_DeviceBytes(g) : 0;
    g->chunk = SETPRO_DSVNA_CHUNK;
    if (OpenMX_HostSimdStageOnHost("Set_ProExpn_VNA: DS_VNA",need,batch_budget)) {
      SetPro_note("DS_VNA GPU batch","one batch of %d pairs needs %.0f MiB: host SIMD DS_VNA",
                  pairs<SETPRO_DSVNA_CHUNK ? pairs : SETPRO_DSVNA_CHUNK,(double)need/1048576.0);
      g->enabled = 1;
      SetPro_DSVNA_GpuEnd(g);
      return 0;
    }
  }
  while(g->chunk>1 && SetPro_DSVNA_DeviceBytes(g)>batch_budget)g->chunk/=2;
  /* Every rank must make the same number of collective task calls, including
     mixed devices, nodes with different memory, and ranks without a GPU. */
  MPI_Allreduce(MPI_IN_PLACE,&g->chunk,1,MPI_INT,MPI_MIN,mpi_comm_level1);
  g->enabled = 1;
  return 1;
}

static void SetPro_DSVNA_GpuEnd(SetProGpu2Context *g)
{
  if (!g->enabled) return;

  free(g->c2r);
  free(g->c2r_off);
  free(g->gnt);
  free(g->vnab);
  free(g->pro01);
  free(g->pro00);
  free(g->vna_num1);
  free(g->vna_L1);
  free(g->cmb_num0);
  free(g->cmb_L0);
  free(g->sp_ncmb);
  free(g->sp_lfi);
  memset(g,0,sizeof(*g));
}

/* run the whole (Mc_AN,h_AN) list in chunks and store into DS_VNA */
typedef struct {int mesh,num_proj,num_rvna,lfi_max,cmb_max,blk_max,l0max_all,l1max,mn_max;} SetProDSVNACapacityParameters;
static void SetPro_DSVNA_capacity_kernel(size_t first,size_t count,const void *parameters,void *const v[],int gpu)
{
 (void)first;const SetProDSVNACapacityParameters *p=parameters;const int np=(int)count;
 const int num_proj=p->num_proj;
 const int num_rvna=p->num_rvna;
 const int lfi_max=p->lfi_max;
 const int cmb_max=p->cmb_max;
 const int blk_max=p->blk_max;
 const int l0max_all=p->l0max_all;
 const int l1max=p->l1max;
 const int mn_max=p->mn_max;
 const int mesh=p->mesh;
 const int nM0=2*l0max_all+1,nM1=2*l1max+1,nLL=lfi_max+1,nm=2*lfi_max+1,nsh=(lfi_max+1)*(lfi_max+1);
 const size_t elem_per_pair=(size_t)cmb_max*nM0*num_rvna*nM1;
 /* Task views occupy disjoint arena slices. Preserve that information at
    the numerical boundary instead of losing it through the void* table. */
 const double *restrict pro00=v[0],*restrict pro01=v[1],*restrict vnab=v[2];
 const double *restrict gnt=v[3],*restrict c2r=v[4];
 const int *restrict c2r_off=v[5],*restrict sp_lfi=v[6],*restrict sp_ncmb=v[7];
 const int *restrict cmb_L0=v[8],*restrict cmb_num0=v[9];
 const int *restrict vna_L1=v[10],*restrict vna_num1=v[11];
 const double *restrict GL_NormK=v[12],*restrict pr_r=v[13];
 const double *restrict pr_siT=v[14],*restrict pr_coT=v[15];
 const double *restrict pr_siP=v[16],*restrict pr_coP=v[17];
 const int *restrict pr_spe=v[18],*restrict pr_hwan=v[19];
 const int *restrict pr_h0=v[20],*restrict pr_tno=v[21];
 const double *restrict sh_tab=v[22];
 float *restrict outbuf=v[23];
 double *restrict sphb=v[24],*restrict sphbp=v[25];
 double *restrict sum0=v[26],*restrict sumr0=v[27],*restrict tmpnl=v[28];

#pragma acc parallel loop deviceptr(outbuf) if(gpu)
 for(size_t i=0;i<(size_t)np*4*mn_max;i++)outbuf[i]=0;
      /* K0: spherical Bessel tables of every (pair, GL point) */

#pragma acc parallel loop gang vector vector_length(128) \
    deviceptr(pr_r,pr_spe,sp_lfi,GL_NormK,sphb,sphbp) if(gpu)
      for (int pt=0; pt<np*mesh; pt++){
        const int pp2 = pt/mesh;
        const int i = pt - pp2*mesh;
        const int lfi = sp_lfi[pr_spe[pp2]];
        double tb[SETPRO_BES_LMAX];
        double tbp[SETPRO_BES_LMAX];
        int LL;

        SetPro_Spherical_Bessel2_dev(GL_NormK[i]*pr_r[pp2],lfi,tb,tbp);
        for (LL=0; LL<=lfi; LL++){
          sphb [((size_t)pp2*nLL + LL)*mesh + i] = tb[LL];
          sphbp[((size_t)pp2*nLL + LL)*mesh + i] = tbp[LL];
        }
      }

      /* K1: Gauss-Legendre quadrature sums */

#pragma acc parallel loop gang vector vector_length(128) \
    deviceptr(pr_spe,pr_hwan,pr_h0,sp_lfi,sp_ncmb,pro00,pro01,vnab,sphb,sphbp,sum0,sumr0) if(gpu)
      for (size_t q=0; q<(size_t)np*nLL*cmb_max*num_rvna; q++){
        const int Lq = (int)(q % num_rvna);
        const int cq = (int)((q/num_rvna) % cmb_max);
        const int LLq = (int)((q/((size_t)num_rvna*cmb_max)) % nLL);
        const int pp2 = (int)(q/((size_t)num_rvna*cmb_max*nLL));
        const int spe = pr_spe[pp2];

        if (LLq<=sp_lfi[spe] && cq<sp_ncmb[spe]){
          const double *p00 = pro00 + ((size_t)spe*cmb_max+cq)*mesh;
          const double *p01 = pro01 + ((size_t)spe*cmb_max+cq)*mesh;
          const double *vb = vnab + ((size_t)pr_hwan[pp2]*num_rvna+Lq)*mesh;
          const double *sb = sphb + ((size_t)pp2*nLL + LLq)*mesh;
          const double *sbp = sphbp + ((size_t)pp2*nLL + LLq)*mesh;
          double tmp1 = 0.0,tmp2 = 0.0;
          int i;

          for (i=0; i<mesh; i++){
            tmp1 += (p00[i]*sb[i])*vb[i];
            tmp2 += (p01[i]*sbp[i])*vb[i];
          }

          sum0[q] = tmp1;
          sumr0[q] = pr_h0[pp2] ? 0.0 : tmp2;
        }
        else{
          sum0[q] = 0.0;
          sumr0[q] = 0.0;
        }
      }

      /* K2: assemble TmpNL (the LL,m sums with Gaunt x Y_lm weights) */

#pragma acc parallel loop gang vector vector_length(128) \
    deviceptr(pr_spe,sp_lfi,sp_ncmb,cmb_L0,vna_L1,gnt,sh_tab,sum0,sumr0,tmpnl) if(gpu)
      for (size_t e=0; e<(size_t)np*elem_per_pair; e++){
        const int M1i = (int)(e % nM1);
        const int Lq = (int)((e/nM1) % num_rvna);
        const int M0i = (int)((e/((size_t)nM1*num_rvna)) % nM0);
        const int cq = (int)((e/((size_t)nM1*num_rvna*nM0)) % cmb_max);
        const int pp2 = (int)(e/elem_per_pair);
        const int spe = pr_spe[pp2];
        const int L0 = (cq<sp_ncmb[spe]) ? cmb_L0[spe*cmb_max+cq] : -1;
        const int L1 = vna_L1[Lq];
        const int M0 = M0i - l0max_all;
        const int M1 = M1i - l1max;
        double s_r = 0.0,s_i = 0.0;
        double r_r = 0.0,r_i = 0.0;
        double t_r = 0.0,t_i = 0.0;
        double p_r = 0.0,p_i = 0.0;

        if (L0>=0 && M0>=-L0 && M0<=L0 && M1>=-L1 && M1<=L1){
          const int lfi = sp_lfi[spe];
          int LL,m;

          for (LL=0; LL<=lfi; LL++){

            const int Ls = -L0 + L1 + LL;
            double pw_r,pw_i;

            /* Im_pow(-1,Ls) with the original C modulo semantics */
            if (Ls%2==0){
              if (Ls%4==0){ pw_r = 1.0; pw_i = 0.0; }
              else        { pw_r = -1.0; pw_i = 0.0; }
            }
            else{
              if ((Ls+1)%4==0){ pw_r = 0.0; pw_i = 1.0; }
              else            { pw_r = 0.0; pw_i = -1.0; }
            }

            if ( abs(L1-LL)<=L0 && L0<=(L1+LL) ){

              m = M0 - M1;
              if (-LL<=m && m<=LL){

                const size_t shoff = ((size_t)pp2*nsh + (size_t)LL*LL + (m+LL))*6;
                const double cy_r = sh_tab[shoff+0];
                const double cy_i = sh_tab[shoff+1];
                const double cyt_r = sh_tab[shoff+2];
                const double cyt_i = sh_tab[shoff+3];
                const double cyp_r = sh_tab[shoff+4];
                const double cyp_i = sh_tab[shoff+5];
                const double cy1_r = pw_r*cy_r - pw_i*cy_i;
                const double cy1_i = pw_r*cy_i + pw_i*cy_r;
                const double cyt1_r = pw_r*cyt_r - pw_i*cyt_i;
                const double cyt1_i = pw_r*cyt_i + pw_i*cyt_r;
                const double cyp1_r = pw_r*cyp_r - pw_i*cyp_i;
                const double cyp1_i = pw_r*cyp_i + pw_i*cyp_r;
                const size_t gidx = ((((((size_t)L0*nM0 + (M0+l0max_all))*(l1max+1) + L1)*nM1
                                       + (M1+l1max))*nLL + LL)*nm + (m+lfi_max));
                const double gant = gnt[gidx];
                const size_t sq = (((size_t)pp2*nLL + LL)*cmb_max + cq)*num_rvna + Lq;
                const double tmp2 = gant*sum0[sq];
                const double tmp0 = gant*sumr0[sq];

                s_r += cy1_r*tmp2;
                s_i += cy1_i*tmp2;
                r_r += cy1_r*tmp0;
                r_i += cy1_i*tmp0;
                t_r += cyt1_r*tmp2;
                t_i += cyt1_i*tmp2;
                p_r += cyp1_r*tmp2;
                p_i += cyp1_i*tmp2;
              }
            }
          }
        }

        {
          double *dst = tmpnl + e*8;

          dst[0] = s_r; dst[1] = s_i;
          dst[2] = r_r; dst[3] = r_i;
          dst[4] = t_r; dst[5] = t_i;
          dst[6] = p_r; dst[7] = p_i;
        }
      }

      /* K3: complex-to-real transforms and the DS_VNA elements */

#pragma acc parallel loop gang vector vector_length(128) \
    deviceptr(pr_spe,pr_h0,pr_r,pr_siT,pr_coT,pr_siP,pr_coP,pr_tno,sp_ncmb,cmb_L0,cmb_num0,vna_L1,vna_num1,c2r,c2r_off,tmpnl,outbuf) if(gpu)
      for (size_t e=0; e<(size_t)np*elem_per_pair; e++){
        const int M1i = (int)(e % nM1);
        const int Lq = (int)((e/nM1) % num_rvna);
        const int M0i = (int)((e/((size_t)nM1*num_rvna)) % nM0);
        const int cq = (int)((e/((size_t)nM1*num_rvna*nM0)) % cmb_max);
        const int pp2 = (int)(e/elem_per_pair);
        const int spe = pr_spe[pp2];
        const int L0 = (cq<sp_ncmb[spe]) ? cmb_L0[spe*cmb_max+cq] : -1;
        const int L1 = vna_L1[Lq];
        const int M0 = M0i - l0max_all;
        const int M1 = M1i - l1max;

        if (L0>=0 && M0>=-L0 && M0<=L0 && M1>=-L1 && M1<=L1){

          double s0_r = 0.0,s0_i = 0.0;
          double sr_r = 0.0,sr_i = 0.0;
          double st_r = 0.0,st_i = 0.0;
          double sp_r = 0.0,sp_i = 0.0;
          int k1,k0;

          for (k1=-L1; k1<=L1; k1++){

            /* first transform: Cmat(M0,k1) = sum_k0 conj(C2R0)*TmpNL(k0,k1) */

            double c0_r = 0.0,c0_i = 0.0;
            double cr_r = 0.0,cr_i = 0.0;
            double ct_r = 0.0,ct_i = 0.0;
            double cp_r = 0.0,cp_i = 0.0;

            for (k0=-L0; k0<=L0; k0++){
              const size_t uidx = (size_t)c2r_off[L0] + (size_t)(L0+M0)*(2*L0+1) + (L0+k0);
              const double u_r = c2r[2*uidx];
              const double u_i = -c2r[2*uidx+1];
              const size_t te = ((((size_t)pp2*cmb_max + cq)*nM0 + (k0+l0max_all))
                                 *num_rvna + Lq)*nM1 + (k1+l1max);
              const double *tv = tmpnl + te*8;

              c0_r += u_r*tv[0] - u_i*tv[1];
              c0_i += u_r*tv[1] + u_i*tv[0];
              cr_r += u_r*tv[2] - u_i*tv[3];
              cr_i += u_r*tv[3] + u_i*tv[2];
              ct_r += u_r*tv[4] - u_i*tv[5];
              ct_i += u_r*tv[5] + u_i*tv[4];
              cp_r += u_r*tv[6] - u_i*tv[7];
              cp_i += u_r*tv[7] + u_i*tv[6];
            }

            /* second transform */

            {
              const size_t vidx = (size_t)c2r_off[L1] + (size_t)(L1+M1)*(2*L1+1) + (L1+k1);
              const double v_r = c2r[2*vidx];
              const double v_i = c2r[2*vidx+1];

              s0_r += (c0_r*v_r - c0_i*v_i);
              s0_i += (c0_r*v_i + c0_i*v_r);
              sr_r += (cr_r*v_r - cr_i*v_i);
              sr_i += (cr_r*v_i + cr_i*v_r);
              st_r += (ct_r*v_r - ct_i*v_i);
              st_i += (ct_r*v_i + ct_i*v_r);
              sp_r += (cp_r*v_r - cp_i*v_i);
              sp_i += (cp_r*v_i + cp_i*v_r);
            }
          }

          {
            const int num0 = cmb_num0[spe*cmb_max+cq];
            const int num1 = vna_num1[Lq];
            const int row = num0 + L0 + M0;
            const int col = num1 + L1 + M1;
            const size_t obase = ((size_t)pp2*4)*(size_t)mn_max;
            const size_t oidx = (size_t)row*num_proj + col;
            const double r = pr_r[pp2];
            const double siT = pr_siT[pp2];
            const double coT = pr_coT[pp2];
            const double siP = pr_siP[pp2];
            const double coP = pr_coP[pp2];

            outbuf[obase + oidx] = 8.0*(float)s0_r;

            if (!pr_h0[pp2]){

              if (fabs(siT)<10e-14){
                outbuf[obase + (size_t)mn_max + oidx] =
                  -8.0*(float)(siT*coP*sr_r + coT*coP/r*st_r);
                outbuf[obase + 2*(size_t)mn_max + oidx] =
                  -8.0*(float)(siT*siP*sr_r + coT*siP/r*st_r);
                outbuf[obase + 3*(size_t)mn_max + oidx] =
                  -8.0*(float)(coT*sr_r - siT/r*st_r);
              }
              else{
                outbuf[obase + (size_t)mn_max + oidx] =
                  -8.0*(float)(siT*coP*sr_r + coT*coP/r*st_r
                               - siP/siT/r*sp_r);
                outbuf[obase + 2*(size_t)mn_max + oidx] =
                  -8.0*(float)(siT*siP*sr_r + coT*siP/r*st_r
                               + coP/siT/r*sp_r);
                outbuf[obase + 3*(size_t)mn_max + oidx] =
                  -8.0*(float)(coT*sr_r - siT/r*st_r);
              }
            }
            else{
              outbuf[obase + (size_t)mn_max + oidx] = 0.0f;
              outbuf[obase + 2*(size_t)mn_max + oidx] = 0.0f;
              outbuf[obase + 3*(size_t)mn_max + oidx] = 0.0f;
            }
          }
        }
      }

}

static void SetPro_DSVNA_GpuRun_Part(Type_DS_VNA *****DS_VNA, int OneD_Nloop, const int *OneD2Mc_AN, const int *OneD2h_AN, SetProGpu2Context *g, double *GL_NormK)
{
  const int num_proj = g->num_proj;
  const int num_rvna = g->num_rvna;
  const int lfi_max = g->lfi_max;
  const int cmb_max = g->cmb_max;
  const int blk_max = g->blk_max;
  const int l0max_all = g->l0max_all;
  const int l1max = g->l1max;
  const int nM0 = 2*l0max_all + 1;
  const int nM1 = 2*l1max + 1;
  const int nLL = lfi_max + 1;
  const int nm = 2*lfi_max + 1;
  const int nsh = (lfi_max+1)*(lfi_max+1);
  const int mn_max = g->mn_max;
  /* same constant SetPro_DSVNA_DeviceBytes sized the pre-flight check with */
  const int CHUNK = g->chunk;
  int chunk_start;

  double *pr_r,*pr_siT,*pr_coT,*pr_siP,*pr_coP;
  int *pr_spe,*pr_hwan,*pr_h0,*pr_tno;
  double *sh_tab;
  double *sphb,*sphbp;
  double *sum0,*sumr0;
  double *tmpnl;
  float *outbuf;

  const size_t sphb_len = (size_t)CHUNK*nLL*GL_Mesh;
  const size_t sum_len = (size_t)CHUNK*nLL*cmb_max*num_rvna;
  const size_t elem_per_pair = (size_t)cmb_max*nM0*num_rvna*nM1;
  const size_t tmpnl_len = (size_t)CHUNK*elem_per_pair*8; /* 4 arrays x complex */
  const size_t out_len = (size_t)CHUNK*4*(size_t)g->mn_max;

  if (!g->enabled) return;
  int all_pairs=OneD_Nloop;
  MPI_Allreduce(MPI_IN_PLACE,&all_pairs,1,MPI_INT,MPI_MAX,mpi_comm_level1);

  pr_r = (double*)SetPro_checked_malloc(sizeof(double)*CHUNK);
  pr_siT = (double*)SetPro_checked_malloc(sizeof(double)*CHUNK);
  pr_coT = (double*)SetPro_checked_malloc(sizeof(double)*CHUNK);
  pr_siP = (double*)SetPro_checked_malloc(sizeof(double)*CHUNK);
  pr_coP = (double*)SetPro_checked_malloc(sizeof(double)*CHUNK);
  pr_spe = (int*)SetPro_checked_malloc(sizeof(int)*CHUNK);
  pr_hwan = (int*)SetPro_checked_malloc(sizeof(int)*CHUNK);
  pr_h0 = (int*)SetPro_checked_malloc(sizeof(int)*CHUNK);
  pr_tno = (int*)SetPro_checked_malloc(sizeof(int)*CHUNK);
  sh_tab = (double*)SetPro_checked_malloc(sizeof(double)*(size_t)CHUNK*nsh*6);
  sphb = NULL;
  sphbp = NULL;
  sum0 = NULL;
  sumr0 = NULL;
  tmpnl = NULL;
  outbuf = (float*)SetPro_checked_malloc(sizeof(float)*out_len);

  {
    double *pro00 = g->pro00;
    double *pro01 = g->pro01;
    double *vnab = g->vnab;
    double *gnt = g->gnt;
    double *c2r = g->c2r;
    int *c2r_off = g->c2r_off;
    int *sp_lfi = g->sp_lfi;
    int *sp_ncmb = g->sp_ncmb;
    int *cmb_L0 = g->cmb_L0;
    int *cmb_num0 = g->cmb_num0;
    int *vna_L1 = g->vna_L1;
    int *vna_num1 = g->vna_num1;
    const size_t pro_len = (size_t)SpeciesNum*cmb_max*GL_Mesh;
    const size_t vnab_len = (size_t)SpeciesNum*num_rvna*GL_Mesh;
    const size_t gnt_len = (size_t)(l0max_all+1)*nM0*(l1max+1)*nM1*nLL*nm;
    const size_t c2r_len = 2*(size_t)c2r_off[((l0max_all<l1max)?l1max:l0max_all)+1];
    const int lc_max = (l0max_all<l1max) ? l1max : l0max_all;

    double dsvna_t0;
    dtime(&dsvna_t0);
    SetPro_note("DS_VNA GPU batch begin","%d pairs in batches of %d, estimated %.0f MiB on the device",
                OneD_Nloop,CHUNK,(double)SetPro_DSVNA_DeviceBytes(g)/1048576.0);

    for (chunk_start=0; chunk_start<all_pairs; chunk_start+=CHUNK){

      const int np = chunk_start>=OneD_Nloop?0:((OneD_Nloop-chunk_start)<CHUNK ? OneD_Nloop-chunk_start : CHUNK);
      int pp,LLh,mh;

      if (chunk_start==0) SetPro_note("DS_VNA GPU batch","device arrays mapped");

      memset(sh_tab,0,(size_t)np*nsh*6*sizeof(double));

      /* host: pair geometry and the spherical harmonics of every pair */

#pragma omp parallel for if(!omp_in_parallel()) schedule(dynamic) private(LLh,mh)
      for (pp=0; pp<np; pp++){
        int Mc_AN = OneD2Mc_AN[chunk_start+pp];
        int h_AN = OneD2h_AN[chunk_start+pp];
        int Gc_AN = M2G[Mc_AN];
        int Cwan = WhatSpecies[Gc_AN];
        int Gh_AN = natn[Gc_AN][h_AN];
        int Rnh = ncn[Gc_AN][h_AN];
        double S_co[3];
        double dx,dy,dz,r,theta,phi;
        double SH[2],dSHt[2],dSHp[2];

        dx = Gxyz[Gh_AN][1] + atv[Rnh][1] - Gxyz[Gc_AN][1];
        dy = Gxyz[Gh_AN][2] + atv[Rnh][2] - Gxyz[Gc_AN][2];
        dz = Gxyz[Gh_AN][3] + atv[Rnh][3] - Gxyz[Gc_AN][3];

        xyz2spherical(dx,dy,dz,0.0,0.0,0.0,S_co);
        r = S_co[0];
        theta = S_co[1];
        phi = S_co[2];
        if (r<1.0e-10) r = 1.0e-10;

        pr_r[pp] = r;
        pr_siT[pp] = sin(theta);
        pr_coT[pp] = cos(theta);
        pr_siP[pp] = sin(phi);
        pr_coP[pp] = cos(phi);
        pr_spe[pp] = Cwan;
        pr_hwan[pp] = WhatSpecies[Gh_AN];
        pr_h0[pp] = (h_AN==0);
        pr_tno[pp] = Spe_Total_NO[Cwan];

        for (LLh=0; LLh<=sp_lfi[Cwan]; LLh++){
          for (mh=-LLh; mh<=LLh; mh++){
            size_t off = ((size_t)pp*nsh + (size_t)LLh*LLh + (mh+LLh))*6;

            ComplexSH(LLh,mh,theta,phi,SH,dSHt,dSHp);
            sh_tab[off+0] = SH[0];
            sh_tab[off+1] = -SH[1];
            sh_tab[off+2] = dSHt[0];
            sh_tab[off+3] = -dSHt[1];
            sh_tab[off+4] = dSHp[0];
            sh_tab[off+5] = -dSHp[1];
          }
        }
      }

      SetProDSVNACapacityParameters parameters={GL_Mesh,g->num_proj,g->num_rvna,g->lfi_max,g->cmb_max,g->blk_max,g->l0max_all,g->l1max,g->mn_max};
      OpenMXGpuTask task={0};task.count=np;task.inputs=23;task.outputs=1;task.scratch=5;
      task.parameters=&parameters;task.parameter_bytes=sizeof(parameters);task.kernel=SetPro_DSVNA_capacity_kernel;
      task.view[0]=(OpenMXGpuTaskView){pro00,(pro_len)*sizeof(double),0};
      task.view[1]=(OpenMXGpuTaskView){pro01,(pro_len)*sizeof(double),0};
      task.view[2]=(OpenMXGpuTaskView){vnab,(vnab_len)*sizeof(double),0};
      task.view[3]=(OpenMXGpuTaskView){gnt,(gnt_len)*sizeof(double),0};
      task.view[4]=(OpenMXGpuTaskView){c2r,(c2r_len)*sizeof(double),0};
      task.view[5]=(OpenMXGpuTaskView){c2r_off,(lc_max+2)*sizeof(int),0};
      task.view[6]=(OpenMXGpuTaskView){sp_lfi,(SpeciesNum)*sizeof(int),0};
      task.view[7]=(OpenMXGpuTaskView){sp_ncmb,(SpeciesNum)*sizeof(int),0};
      task.view[8]=(OpenMXGpuTaskView){cmb_L0,((size_t)SpeciesNum*cmb_max)*sizeof(int),0};
      task.view[9]=(OpenMXGpuTaskView){cmb_num0,((size_t)SpeciesNum*cmb_max)*sizeof(int),0};
      task.view[10]=(OpenMXGpuTaskView){vna_L1,(num_rvna)*sizeof(int),0};
      task.view[11]=(OpenMXGpuTaskView){vna_num1,(num_rvna)*sizeof(int),0};
      task.view[12]=(OpenMXGpuTaskView){GL_NormK,(GL_Mesh)*sizeof(double),0};
      task.view[13]=(OpenMXGpuTaskView){pr_r,0,(1)*sizeof(double)};
      task.view[14]=(OpenMXGpuTaskView){pr_siT,0,(1)*sizeof(double)};
      task.view[15]=(OpenMXGpuTaskView){pr_coT,0,(1)*sizeof(double)};
      task.view[16]=(OpenMXGpuTaskView){pr_siP,0,(1)*sizeof(double)};
      task.view[17]=(OpenMXGpuTaskView){pr_coP,0,(1)*sizeof(double)};
      task.view[18]=(OpenMXGpuTaskView){pr_spe,0,(1)*sizeof(int)};
      task.view[19]=(OpenMXGpuTaskView){pr_hwan,0,(1)*sizeof(int)};
      task.view[20]=(OpenMXGpuTaskView){pr_h0,0,(1)*sizeof(int)};
      task.view[21]=(OpenMXGpuTaskView){pr_tno,0,(1)*sizeof(int)};
      task.view[22]=(OpenMXGpuTaskView){sh_tab,0,((size_t)nsh*6)*sizeof(double)};
      task.view[23]=(OpenMXGpuTaskView){outbuf,0,4*(size_t)mn_max*sizeof(float)};
      task.view[24]=(OpenMXGpuTaskView){NULL,0,((size_t)nLL*GL_Mesh)*sizeof(double)};
      task.view[25]=(OpenMXGpuTaskView){NULL,0,((size_t)nLL*GL_Mesh)*sizeof(double)};
      task.view[26]=(OpenMXGpuTaskView){NULL,0,((size_t)nLL*cmb_max*num_rvna)*sizeof(double)};
      task.view[27]=(OpenMXGpuTaskView){NULL,0,((size_t)nLL*cmb_max*num_rvna)*sizeof(double)};
      task.view[28]=(OpenMXGpuTaskView){NULL,0,(elem_per_pair*8)*sizeof(double)};
      int all_done;
      const int done=SetPro_BatchAttempt(&task,"SetPro_DSVNA",&all_done);
      if (!all_done) {
        /* a late device failure on some rank: from here every rank's
           unfinished pairs go to the host SIMD DS_VNA assembly */
        g->host_route = 1;
        g->host_first = done ? (chunk_start+np<OneD_Nloop ? chunk_start+np : OneD_Nloop) : chunk_start;
        SetPro_note("DS_VNA GPU batch","device batch failed on %s; pairs %d..%d of this rank go to the host SIMD assembly",
                    done ? "another rank" : "this rank",g->host_first,OneD_Nloop-1);
      }


      /* host: scatter the chunk results into DS_VNA (none after a failure here) */

      const int np_done = done ? np : 0;
#pragma omp parallel for if(!omp_in_parallel()) schedule(dynamic)
      for (pp=0; pp<np_done; pp++){
        int Mc_AN = OneD2Mc_AN[chunk_start+pp];
        int h_AN = OneD2h_AN[chunk_start+pp];
        int tno = pr_tno[pp];
        int kk,i2;

        for (kk=0; kk<=3; kk++){
          for (i2=0; i2<tno; i2++){
            memcpy(DS_VNA[kk][Mc_AN][h_AN][i2],
                   outbuf + ((size_t)pp*4 + kk)*(size_t)mn_max
                          + (size_t)i2*num_proj,
                   sizeof(float)*(size_t)num_proj);
          }
        }
      }
      if (!all_done) break;

    } /* chunk_start */

    {
      double t1;
      dtime(&t1);
      SetPro_note("DS_VNA GPU batch end","%.2f s",t1-dsvna_t0);
    }
  }

  free(outbuf);
  free(tmpnl);
  free(sumr0);
  free(sum0);
  free(sphbp);
  free(sphb);
  free(sh_tab);
  free(pr_tno);
  free(pr_h0);
  free(pr_hwan);
  free(pr_spe);
  free(pr_coP);
  free(pr_siP);
  free(pr_coT);
  free(pr_siT);
  free(pr_r);
}
static void SetPro_DSVNA_GpuRun(Type_DS_VNA *****matrix,int count,const int *mc,const int *h,double *GL_NormK,SetProGpu2Context *g)
{SetPro_DSVNA_GpuRun_Part(matrix,count,mc,h,g,GL_NormK);}


/* ------------------------------------------------------------------ */
/* Batched device construction of HVNA2 / HVNA3 (the one-centre PAO
   pairs against the crude VNA potential).  Set_VNA2 and Set_VNA3 are
   mirror images: the PAO-product species and the direction sign swap.
   The same host-tabulated Gaunt/Y_lm treatment as the DS_VNA batch is
   used; the upper->lower conjugate copy and both unitary transforms
   keep the exact CPU loop order.                                      */
/* ------------------------------------------------------------------ */



/* full pair-chunk length of the HVNA2/HVNA3 device pipeline; the admitted
   length g->chunk (halved until the footprint fits the rank's budget) is
   shared by the estimate below and by SetPro_VNA23_GpuRun */
#define SETPRO_VNA23_CHUNK 512

/* Device bytes the "#pragma acc data" region of SetPro_VNA23_GpuRun maps,
   term for term in the order of that clause (valid once the per-species
   tables of SetPro_VNA23_GpuBegin are built). */
static size_t SetPro_VNA23_DeviceBytes(const SetProGpu3Context *g)
{
  const size_t CH = (size_t)g->chunk;
  const int nM = 2*g->l0max + 1;
  const int nLL = g->lfi_max + 1;
  const int nsh = (g->lfi_max+1)*(g->lfi_max+1);
  const size_t crude_len = (size_t)SpeciesNum*GL_Mesh;
  const size_t prod_len = (size_t)SpeciesNum*g->cmb_max*g->cmb_max*nLL*GL_Mesh;
  const size_t gnt2_len = (size_t)(g->l0max+1)*nM*(g->l0max+1)*nM*nLL;
  const size_t c2r_len = 2*(size_t)g->c2r_off[g->l0max+1];
  const size_t sphb_len = CH*nLL*GL_Mesh;
  const size_t sum_len = CH*nLL*g->cmb_max*g->cmb_max;
  const size_t elem_per_pair = (size_t)g->cmb_max*nM*g->cmb_max*nM;
  const size_t tmpnl_len = CH*elem_per_pair*8;
  const size_t out_len = CH*4*(size_t)g->mn_max;
  size_t bytes = 0;

  /* copyin */
  bytes += (crude_len + prod_len + gnt2_len + c2r_len)*sizeof(double);
  bytes += (size_t)(g->l0max+2)*sizeof(int);                                 /* c2r_off   */
  bytes += (size_t)(2*SpeciesNum + 2*(size_t)SpeciesNum*g->cmb_max)*sizeof(int);
  bytes += (size_t)2*GL_Mesh*sizeof(double);                                 /* GL_NormK, GL_Weight */

  /* create */
  bytes += (2*sphb_len + 2*sum_len + tmpnl_len + out_len)*sizeof(double);
  bytes += CH*5*sizeof(double) + CH*3*sizeof(int);                           /* pr_*      */
  bytes += CH*(size_t)nsh*6*sizeof(double);                                  /* sh_tab    */

  return bytes;
}

static int SetPro_VNA23_GpuBegin(const OpenMXReciprocalViews *reciprocal, const dcomplex *conversion, size_t conversion_width, SetProGpu3Context *g)
{
  int spe,L0,Mul0,L1,Mul1,LL,i,c0,c1,k;
  size_t free_bytes = 0,total_bytes = 0;
  int node_ranks = 1,node_rank = 0,device_ranks;
  MPI_Comm node_comm = MPI_COMM_NULL;

  memset(g,0,sizeof(*g));
  g->chunk = SETPRO_VNA23_CHUNK;

  int requested=scf_eigen_lib_flag==GPUSOLVER;
  MPI_Allreduce(MPI_IN_PLACE,&requested,1,MPI_INT,MPI_MIN,mpi_comm_level1);
  if (!requested) return 0;
  if (!openmx_env_flag_at(OPENMX_ENV_SETPRO_GPU,1)) return 0;

  MPI_Comm_split_type(mpi_comm_level1,MPI_COMM_TYPE_SHARED,0,MPI_INFO_NULL,&node_comm);
  MPI_Comm_size(node_comm,&node_ranks);
  MPI_Comm_rank(node_comm,&node_rank);
  if (node_ranks<1) node_ranks = 1;
  device_ranks=SetPro_GpuSharing(node_comm);
  MPI_Comm_free(&node_comm);

  g->sp_lfi = (int*)SetPro_checked_malloc(sizeof(int)*SpeciesNum);
  g->sp_ncmb = (int*)SetPro_checked_malloc(sizeof(int)*SpeciesNum);
  g->sp_tno = (int*)SetPro_checked_malloc(sizeof(int)*SpeciesNum);

  g->l0max = 0;
  g->cmb_max = 0;
  g->tno_max = 0;
  for (spe=0; spe<SpeciesNum; spe++){
    int ncmb = 0,tno = 0;

    g->sp_lfi[spe] = 2*Spe_MaxL_Basis[spe];
    if (g->l0max<Spe_MaxL_Basis[spe]) g->l0max = Spe_MaxL_Basis[spe];
    for (L0=0; L0<=Spe_MaxL_Basis[spe]; L0++){
      ncmb += Spe_Num_Basis[spe][L0];
      tno += Spe_Num_Basis[spe][L0]*(2*L0+1);
    }
    g->sp_ncmb[spe] = ncmb;
    g->sp_tno[spe] = tno;
    if (g->cmb_max<ncmb) g->cmb_max = ncmb;
    if (g->tno_max<tno) g->tno_max = tno;
  }
  g->lfi_max = 2*g->l0max;
  g->mn_max = g->tno_max*g->tno_max;

  g->cmb_L0 = (int*)SetPro_checked_malloc(sizeof(int)*(size_t)SpeciesNum*g->cmb_max);
  g->cmb_num0 = (int*)SetPro_checked_malloc(sizeof(int)*(size_t)SpeciesNum*g->cmb_max);

  for (spe=0; spe<SpeciesNum; spe++){
    int num0 = 0,c = 0;

    for (L0=0; L0<=Spe_MaxL_Basis[spe]; L0++){
      for (Mul0=0; Mul0<Spe_Num_Basis[spe][L0]; Mul0++){
        g->cmb_L0[spe*g->cmb_max+c] = L0;
        g->cmb_num0[spe*g->cmb_max+c] = num0;
        num0 += 2*L0 + 1;
        c++;
      }
    }
    for (; c<g->cmb_max; c++){
      g->cmb_L0[spe*g->cmb_max+c] = -1;
      g->cmb_num0[spe*g->cmb_max+c] = 0;
    }
  }

  g->crude = (double*)SetPro_checked_malloc(sizeof(double)*(size_t)SpeciesNum*GL_Mesh);
  for (spe=0; spe<SpeciesNum; spe++){
    for (i=0; i<GL_Mesh; i++){
      g->crude[(size_t)spe*GL_Mesh+i] = openmx_radial_row2(&reciprocal->potential_transform,spe)[i];
    }
  }

  {
    const int nLL = g->lfi_max + 1;
    size_t plen = (size_t)SpeciesNum*g->cmb_max*g->cmb_max*nLL*GL_Mesh;
    size_t q;

    g->prod = (double*)SetPro_checked_malloc(sizeof(double)*plen);
    for (q=0; q<plen; q++) g->prod[q] = 0.0;

    for (spe=0; spe<SpeciesNum; spe++){
      int ca = 0;

      for (L0=0; L0<=Spe_MaxL_Basis[spe]; L0++){
        for (Mul0=0; Mul0<Spe_Num_Basis[spe][L0]; Mul0++){
          int cb = 0;

          for (L1=0; L1<=Spe_MaxL_Basis[spe]; L1++){
            for (Mul1=0; Mul1<Spe_Num_Basis[spe][L1]; Mul1++){

              if (L0<=L1){
                for (LL=0; LL<=2*L1; LL++){
                  for (i=0; i<GL_Mesh; i++){
                    g->prod[((((size_t)spe*g->cmb_max+ca)*g->cmb_max+cb)*nLL+LL)*GL_Mesh+i]
                      = openmx_product_radial_row(&reciprocal->product_transform,spe,L0,Mul0,L1,Mul1,LL)[i];
                  }
                }
              }
              cb++;
            }
          }
          ca++;
        }
      }
    }
  }

  /* (-1)^|m| * Gaunt(L0,M0,L1,M1,LL,-m) with m = M1-M0 */

  {
    const int nM = 2*g->l0max + 1;
    const int nLL = g->lfi_max + 1;
    size_t gsz = (size_t)(g->l0max+1)*nM*(g->l0max+1)*nM*nLL;
    int L0v,M0,L1v,M1,m;
    size_t q;

    g->gnt2 = (double*)SetPro_checked_malloc(sizeof(double)*gsz);
    for (q=0; q<gsz; q++) g->gnt2[q] = 0.0;

    for (L0v=0; L0v<=g->l0max; L0v++){
      for (M0=-L0v; M0<=L0v; M0++){
        for (L1v=0; L1v<=g->l0max; L1v++){
          for (M1=-L1v; M1<=L1v; M1++){
            for (LL=0; LL<nLL; LL++){
              m = M1 - M0;
              if (-LL<=m && m<=LL){
                size_t idx = (((((size_t)L0v*nM + (M0+g->l0max))*(g->l0max+1) + L1v)*nM
                               + (M1+g->l0max))*nLL + LL);
                g->gnt2[idx] = pow(-1.0,(double)abs(m))*Gaunt(L0v,M0,L1v,M1,LL,-m);
              }
            }
          }
        }
      }
    }
  }

  {
    const int lc = g->l0max;
    size_t pos = 0;
    int Lc,M;

    g->c2r_off = (int*)SetPro_checked_malloc(sizeof(int)*(lc+2));
    for (Lc=0; Lc<=lc; Lc++){
      g->c2r_off[Lc] = (int)pos;
      pos += (size_t)(2*Lc+1)*(2*Lc+1);
    }
    g->c2r_off[lc+1] = (int)pos;
    g->c2r = (double*)SetPro_checked_malloc(sizeof(double)*2*pos);
    for (Lc=0; Lc<=lc; Lc++){
      for (M=0; M<(2*Lc+1); M++){
        for (k=0; k<(2*Lc+1); k++){
          size_t idx = (size_t)g->c2r_off[Lc] + (size_t)M*(2*Lc+1) + k;
          g->c2r[2*idx  ] = conversion[((size_t)(Lc)*conversion_width+(M))*conversion_width+(k)].r;
          g->c2r[2*idx+1] = conversion[((size_t)(Lc)*conversion_width+(M))*conversion_width+(k)].i;
        }
      }
    }
  }

  /* The common task allocator selects owner, local peers, remote nodes,
     then CPU from the exact batch footprint. Keep host lookup tables. */
  g->chunk = SETPRO_VNA23_CHUNK;
  const size_t batch_budget=SetPro_GpuBatchBudget(device_ranks);
  {
    /* Host SIMD vote, as for DS_VNA: one full batch against the batch budget. */
    const int pairs=SetPro_LocalPairs();
    g->chunk = pairs<SETPRO_VNA23_CHUNK ? (pairs>0 ? pairs : 1) : SETPRO_VNA23_CHUNK;
    const size_t need=pairs>0 ? SetPro_VNA23_DeviceBytes(g) : 0;
    g->chunk = SETPRO_VNA23_CHUNK;
    if (OpenMX_HostSimdStageOnHost("Set_ProExpn_VNA: HVNA2/3",need,batch_budget)) {
      SetPro_note("HVNA2/3 GPU batch","one batch of %d pairs needs %.0f MiB: host SIMD HVNA2/3",
                  pairs<SETPRO_VNA23_CHUNK ? pairs : SETPRO_VNA23_CHUNK,(double)need/1048576.0);
      g->enabled = 1;
      SetPro_VNA23_GpuEnd(g);
      return 0;
    }
  }
  while(g->chunk>1 && SetPro_VNA23_DeviceBytes(g)>batch_budget)g->chunk/=2;
  MPI_Allreduce(MPI_IN_PLACE,&g->chunk,1,MPI_INT,MPI_MIN,mpi_comm_level1);
  g->enabled = 1;
  return 1;
}

static void SetPro_VNA23_GpuEnd(SetProGpu3Context *g)
{

  if (!g->enabled) return;

  free(g->c2r);
  free(g->c2r_off);
  free(g->gnt2);
  free(g->prod);
  free(g->crude);
  free(g->cmb_num0);
  free(g->cmb_L0);
  free(g->sp_tno);
  free(g->sp_ncmb);
  free(g->sp_lfi);
  memset(g,0,sizeof(*g));
}

/* run VNA2 (mirror==0) or VNA3 (mirror==1) in pair chunks */
typedef struct {int mesh,l0max,lfi_max,cmb_max,mn_max,tno_max;double Dk;} SetProVNA23CapacityParameters;
static void SetPro_VNA23_capacity_kernel(size_t first,size_t count,const void *parameters,void *const v[],int gpu)
{
 (void)first;const SetProVNA23CapacityParameters *p=parameters;const int np=(int)count;
 const int l0max=p->l0max;
 const int lfi_max=p->lfi_max;
 const int cmb_max=p->cmb_max;
 const int mn_max=p->mn_max;
 const int tno_max=p->tno_max;
 const int mesh=p->mesh;
 const int nM=2*l0max+1,nLL=lfi_max+1,nsh=(lfi_max+1)*(lfi_max+1);
 const size_t elem_per_pair=(size_t)cmb_max*nM*cmb_max*nM;
 const double Dk=p->Dk;
 const double *restrict crude=v[0],*restrict prod=v[1],*restrict gnt2=v[2],*restrict c2r=v[3];
 const int *restrict c2r_off=v[4],*restrict sp_lfi=v[5],*restrict sp_ncmb=v[6];
 const int *restrict cmb_L0=v[7],*restrict cmb_num0=v[8];
 const double *restrict GL_NormK=v[9],*restrict GL_Weight=v[10];
 const double *restrict pr_r=v[11],*restrict pr_siT=v[12],*restrict pr_coT=v[13];
 const double *restrict pr_siP=v[14],*restrict pr_coP=v[15];
 const int *restrict pr_spe=v[16],*restrict pr_vspe=v[17],*restrict pr_h0=v[18];
 const double *restrict sh_tab=v[19];
 double *restrict outbuf=v[20],*restrict sphb=v[21],*restrict sphbp=v[22];
 double *restrict sum0=v[23],*restrict sumr0=v[24],*restrict tmpnl=v[25];

#pragma acc parallel loop deviceptr(outbuf) if(gpu)
 for(size_t i=0;i<(size_t)np*4*mn_max;i++)outbuf[i]=0;
      /* K0: spherical Bessel tables */

#pragma acc parallel loop gang vector vector_length(128) \
    deviceptr(pr_r,pr_spe,sp_lfi,GL_NormK,sphb,sphbp) if(gpu)
      for (int pt=0; pt<np*mesh; pt++){
        const int pp2 = pt/mesh;
        const int i = pt - pp2*mesh;
        const int lfi = sp_lfi[pr_spe[pp2]];
        double tb[SETPRO_BES_LMAX];
        double tbp[SETPRO_BES_LMAX];
        int LL;

        SetPro_Spherical_Bessel2_dev(GL_NormK[i]*pr_r[pp2],lfi,tb,tbp);
        for (LL=0; LL<=lfi; LL++){
          sphb [((size_t)pp2*nLL + LL)*mesh + i] = tb[LL];
          sphbp[((size_t)pp2*nLL + LL)*mesh + i] = tbp[LL];
        }
      }

      /* K1: quadrature sums for every (pair, LL, c0<=c1) block */

#pragma acc parallel loop gang vector vector_length(128) \
    deviceptr(pr_spe,pr_vspe,sp_lfi,sp_ncmb,cmb_L0,crude,prod,GL_NormK,GL_Weight,sphb,sphbp,sum0,sumr0) if(gpu)
      for (size_t q=0; q<(size_t)np*nLL*cmb_max*cmb_max; q++){
        const int cb = (int)(q % cmb_max);
        const int ca = (int)((q/cmb_max) % cmb_max);
        const int LLq = (int)((q/((size_t)cmb_max*cmb_max)) % nLL);
        const int pp2 = (int)(q/((size_t)cmb_max*cmb_max*nLL));
        const int spe = pr_spe[pp2];
        const int L0 = (ca<sp_ncmb[spe]) ? cmb_L0[spe*cmb_max+ca] : -1;
        const int L1 = (cb<sp_ncmb[spe]) ? cmb_L0[spe*cmb_max+cb] : -1;
        double s = 0.0,sr = 0.0;

        if (L0>=0 && L1>=0 && L0<=L1 && LLq<=2*L1 &&
            abs(L1-LLq)<=L0 && L0<=(L1+LLq)){

          const double *cru = crude + (size_t)pr_vspe[pp2]*mesh;
          const double *pr2 = prod + ((((size_t)spe*cmb_max+ca)*cmb_max+cb)*nLL+LLq)*mesh;
          const double *sb = sphb + ((size_t)pp2*nLL + LLq)*mesh;
          const double *sbp = sphbp + ((size_t)pp2*nLL + LLq)*mesh;
          int i;

          for (i=0; i<mesh; i++){
            const double Normk = GL_NormK[i];
            const double tmp10 = 0.50*Dk*cru[i]*pr2[i]*GL_Weight[i]*Normk*Normk;

            s += tmp10*sb[i];
            sr += tmp10*sbp[i]*Normk;
          }
        }

        sum0[q] = s;
        sumr0[q] = sr;
      }

      /* K2a: assemble the upper (L0<=L1) TmpHNA elements */

#pragma acc parallel loop gang vector vector_length(128) \
    deviceptr(pr_spe,sp_lfi,sp_ncmb,cmb_L0,gnt2,sh_tab,sum0,sumr0,tmpnl) if(gpu)
      for (size_t e=0; e<(size_t)np*elem_per_pair; e++){
        const int M1i = (int)(e % nM);
        const int cb = (int)((e/nM) % cmb_max);
        const int M0i = (int)((e/((size_t)nM*cmb_max)) % nM);
        const int ca = (int)((e/((size_t)nM*cmb_max*nM)) % cmb_max);
        const int pp2 = (int)(e/elem_per_pair);
        const int spe = pr_spe[pp2];
        const int L0 = (ca<sp_ncmb[spe]) ? cmb_L0[spe*cmb_max+ca] : -1;
        const int L1 = (cb<sp_ncmb[spe]) ? cmb_L0[spe*cmb_max+cb] : -1;
        const int M0 = M0i - l0max;
        const int M1 = M1i - l0max;
        double s_r = 0.0,s_i = 0.0;
        double r_r = 0.0,r_i = 0.0;
        double t_r = 0.0,t_i = 0.0;
        double p_r = 0.0,p_i = 0.0;

        if (L0>=0 && L1>=0 && L0<=L1 &&
            M0>=-L0 && M0<=L0 && M1>=-L1 && M1<=L1){

          const int m = M1 - M0;
          int LL;

          for (LL=0; LL<=2*L1; LL++){

            if (abs(L1-LL)<=L0 && L0<=(L1+LL) && -LL<=m && m<=LL){

              const size_t shoff = ((size_t)pp2*nsh + (size_t)LL*LL + (m+LL))*6;
              const double cy_r = sh_tab[shoff+0];
              const double cy_i = sh_tab[shoff+1];
              const double cyt_r = sh_tab[shoff+2];
              const double cyt_i = sh_tab[shoff+3];
              const double cyp_r = sh_tab[shoff+4];
              const double cyp_i = sh_tab[shoff+5];
              const size_t gidx = (((((size_t)L0*nM + (M0+l0max))*(l0max+1) + L1)*nM
                                    + (M1+l0max))*nLL + LL);
              const double gant = gnt2[gidx];
              const size_t sq = (((size_t)pp2*nLL + LL)*cmb_max + ca)*cmb_max + cb;
              const double sv = sum0[sq];
              const double srv = sumr0[sq];
              const double cs_r = cy_r*gant;
              const double cs_i = cy_i*gant;
              const double cst_r = cyt_r*gant;
              const double cst_i = cyt_i*gant;
              const double csp_r = cyp_r*gant;
              const double csp_i = cyp_i*gant;

              s_r += cs_r*sv;
              s_i += cs_i*sv;
              r_r += cs_r*srv;
              r_i += cs_i*srv;
              t_r += cst_r*sv;
              t_i += cst_i*sv;
              p_r += csp_r*sv;
              p_i += csp_i*sv;
            }
          }
        }

        {
          double *dst = tmpnl + e*8;

          dst[0] = s_r; dst[1] = s_i;
          dst[2] = r_r; dst[3] = r_i;
          dst[4] = t_r; dst[5] = t_i;
          dst[6] = p_r; dst[7] = p_i;
        }
      }

      /* K2b: conjugate copy of the upper part to the lower part */

#pragma acc parallel loop gang vector vector_length(128) \
    deviceptr(pr_spe,sp_ncmb,cmb_L0,tmpnl) if(gpu)
      for (size_t e=0; e<(size_t)np*elem_per_pair; e++){
        const int M1i = (int)(e % nM);
        const int cb = (int)((e/nM) % cmb_max);
        const int M0i = (int)((e/((size_t)nM*cmb_max)) % nM);
        const int ca = (int)((e/((size_t)nM*cmb_max*nM)) % cmb_max);
        const int pp2 = (int)(e/elem_per_pair);
        const int spe = pr_spe[pp2];
        const int L0 = (ca<sp_ncmb[spe]) ? cmb_L0[spe*cmb_max+ca] : -1;
        const int L1 = (cb<sp_ncmb[spe]) ? cmb_L0[spe*cmb_max+cb] : -1;
        const int M0 = M0i - l0max;
        const int M1 = M1i - l0max;

        if (L0>=0 && L1>=0 && L0>L1 &&
            M0>=-L0 && M0<=L0 && M1>=-L1 && M1<=L1){

          const size_t src = ((((size_t)pp2*cmb_max + cb)*nM + M1i)
                              *cmb_max + ca)*nM + M0i;
          const double *sv = tmpnl + src*8;
          double *dst = tmpnl + e*8;

          dst[0] = sv[0]; dst[1] = -sv[1];
          dst[2] = sv[2]; dst[3] = -sv[3];
          dst[4] = sv[4]; dst[5] = -sv[5];
          dst[6] = sv[6]; dst[7] = -sv[7];
        }
        else if (L0>=0 && L1>=0 && L0==L1 && ca==cb && M0==M1 &&
                 M0>=-L0 && M0<=L0){

          /* the host copy loop conjugates the L0==L1 diagonal in place */
          double *dst = tmpnl + e*8;

          dst[1] = -dst[1];
          dst[3] = -dst[3];
          dst[5] = -dst[5];
          dst[7] = -dst[7];
        }
      }

      /* K3: complex-to-real transforms and the HVNA2/3 elements */

#pragma acc parallel loop gang vector vector_length(128) \
    deviceptr(pr_spe,pr_h0,pr_r,pr_siT,pr_coT,pr_siP,pr_coP,sp_ncmb,cmb_L0,cmb_num0,c2r,c2r_off,tmpnl,outbuf) if(gpu)
      for (size_t e=0; e<(size_t)np*elem_per_pair; e++){
        const int M1i = (int)(e % nM);
        const int cb = (int)((e/nM) % cmb_max);
        const int M0i = (int)((e/((size_t)nM*cmb_max)) % nM);
        const int ca = (int)((e/((size_t)nM*cmb_max*nM)) % cmb_max);
        const int pp2 = (int)(e/elem_per_pair);
        const int spe = pr_spe[pp2];
        const int L0 = (ca<sp_ncmb[spe]) ? cmb_L0[spe*cmb_max+ca] : -1;
        const int L1 = (cb<sp_ncmb[spe]) ? cmb_L0[spe*cmb_max+cb] : -1;
        const int M0 = M0i - l0max;
        const int M1 = M1i - l0max;

        if (L0>=0 && L1>=0 && M0>=-L0 && M0<=L0 && M1>=-L1 && M1<=L1){

          double s0_r = 0.0,s0_i = 0.0;
          double sr_r = 0.0,sr_i = 0.0;
          double st_r = 0.0,st_i = 0.0;
          double sp_r = 0.0,sp_i = 0.0;
          int k1,k0;

          for (k1=-L1; k1<=L1; k1++){

            double c0_r = 0.0,c0_i = 0.0;
            double cr_r = 0.0,cr_i = 0.0;
            double ct_r = 0.0,ct_i = 0.0;
            double cp_r = 0.0,cp_i = 0.0;

            for (k0=-L0; k0<=L0; k0++){
              const size_t uidx = (size_t)c2r_off[L0] + (size_t)(L0+M0)*(2*L0+1) + (L0+k0);
              const double u_r = c2r[2*uidx];
              const double u_i = -c2r[2*uidx+1];
              const size_t te = ((((size_t)pp2*cmb_max + ca)*nM + (k0+l0max))
                                 *cmb_max + cb)*nM + (k1+l0max);
              const double *tv = tmpnl + te*8;

              c0_r += u_r*tv[0] - u_i*tv[1];
              c0_i += u_r*tv[1] + u_i*tv[0];
              cr_r += u_r*tv[2] - u_i*tv[3];
              cr_i += u_r*tv[3] + u_i*tv[2];
              ct_r += u_r*tv[4] - u_i*tv[5];
              ct_i += u_r*tv[5] + u_i*tv[4];
              cp_r += u_r*tv[6] - u_i*tv[7];
              cp_i += u_r*tv[7] + u_i*tv[6];
            }

            {
              const size_t vidx = (size_t)c2r_off[L1] + (size_t)(L1+M1)*(2*L1+1) + (L1+k1);
              const double v_r = c2r[2*vidx];
              const double v_i = c2r[2*vidx+1];

              s0_r += (c0_r*v_r - c0_i*v_i);
              s0_i += (c0_r*v_i + c0_i*v_r);
              sr_r += (cr_r*v_r - cr_i*v_i);
              sr_i += (cr_r*v_i + cr_i*v_r);
              st_r += (ct_r*v_r - ct_i*v_i);
              st_i += (ct_r*v_i + ct_i*v_r);
              sp_r += (cp_r*v_r - cp_i*v_i);
              sp_i += (cp_r*v_i + cp_i*v_r);
            }
          }

          {
            const int num0 = cmb_num0[spe*cmb_max+ca];
            const int num1 = cmb_num0[spe*cmb_max+cb];
            const int row = num0 + L0 + M0;
            const int col = num1 + L1 + M1;
            const size_t obase = (size_t)pp2*4*(size_t)mn_max;
            const size_t oidx = (size_t)row*tno_max + col;
            const double r = pr_r[pp2];
            const double siT = pr_siT[pp2];
            const double coT = pr_coT[pp2];
            const double siP = pr_siP[pp2];
            const double coP = pr_coP[pp2];

            outbuf[obase + oidx] = 8.0*s0_r;

            if (!pr_h0[pp2]){

              if (fabs(siT)<10e-14){
                outbuf[obase + (size_t)mn_max + oidx] =
                  -8.0*(siT*coP*sr_r + coT*coP/r*st_r);
                outbuf[obase + 2*(size_t)mn_max + oidx] =
                  -8.0*(siT*siP*sr_r + coT*siP/r*st_r);
                outbuf[obase + 3*(size_t)mn_max + oidx] =
                  -8.0*(coT*sr_r - siT/r*st_r);
              }
              else{
                outbuf[obase + (size_t)mn_max + oidx] =
                  -8.0*(siT*coP*sr_r + coT*coP/r*st_r
                        - siP/siT/r*sp_r);
                outbuf[obase + 2*(size_t)mn_max + oidx] =
                  -8.0*(siT*siP*sr_r + coT*siP/r*st_r
                        + coP/siT/r*sp_r);
                outbuf[obase + 3*(size_t)mn_max + oidx] =
                  -8.0*(coT*sr_r - siT/r*st_r);
              }
            }
            else{
              outbuf[obase + (size_t)mn_max + oidx] = 0.0;
              outbuf[obase + 2*(size_t)mn_max + oidx] = 0.0;
              outbuf[obase + 3*(size_t)mn_max + oidx] = 0.0;
            }
          }
        }
      }

}

static void SetPro_VNA23_GpuRun_Part(double *****HVNA23, int mirror, int OneD_Nloop, const int *OneD2Mc_AN, const int *OneD2h_AN, SetProGpu3Context *g, double *GL_NormK, const double *weights)
{
  const int l0max = g->l0max;
  const int lfi_max = g->lfi_max;
  const int cmb_max = g->cmb_max;
  const int mn_max = g->mn_max;
  const int tno_max = g->tno_max;
  const int nM = 2*l0max + 1;
  const int nLL = lfi_max + 1;
  const int nsh = (lfi_max+1)*(lfi_max+1);
  const int CHUNK = g->chunk;   /* the admitted batch SetPro_VNA23_GpuBegin sized the region with */
  const double Dk = PAO_Nkmax - Radial_kmin;
  int chunk_start;
  double vna23_t0;

  double *pr_r,*pr_siT,*pr_coT,*pr_siP,*pr_coP;
  int *pr_spe,*pr_vspe,*pr_h0;
  double *sh_tab;
  double *sphb,*sphbp;
  double *sum0,*sumr0;
  double *tmpnl;
  double *outbuf;

  const size_t sphb_len = (size_t)CHUNK*nLL*GL_Mesh;
  const size_t sum_len = (size_t)CHUNK*nLL*cmb_max*cmb_max;
  const size_t elem_per_pair = (size_t)cmb_max*nM*cmb_max*nM;
  const size_t tmpnl_len = (size_t)CHUNK*elem_per_pair*8;
  const size_t out_len = (size_t)CHUNK*4*(size_t)mn_max;

  if (!g->enabled) return;
  int all_pairs=OneD_Nloop;
  MPI_Allreduce(MPI_IN_PLACE,&all_pairs,1,MPI_INT,MPI_MAX,mpi_comm_level1);

  dtime(&vna23_t0);
  SetPro_note("HVNA2/3 GPU batch begin","HVNA%d: %d pairs in batches of %d",mirror ? 3 : 2,OneD_Nloop,CHUNK);

  pr_r = (double*)SetPro_checked_malloc(sizeof(double)*CHUNK);
  pr_siT = (double*)SetPro_checked_malloc(sizeof(double)*CHUNK);
  pr_coT = (double*)SetPro_checked_malloc(sizeof(double)*CHUNK);
  pr_siP = (double*)SetPro_checked_malloc(sizeof(double)*CHUNK);
  pr_coP = (double*)SetPro_checked_malloc(sizeof(double)*CHUNK);
  pr_spe = (int*)SetPro_checked_malloc(sizeof(int)*CHUNK);
  pr_vspe = (int*)SetPro_checked_malloc(sizeof(int)*CHUNK);
  pr_h0 = (int*)SetPro_checked_malloc(sizeof(int)*CHUNK);
  sh_tab = (double*)SetPro_checked_malloc(sizeof(double)*(size_t)CHUNK*nsh*6);
  sphb = NULL;
  sphbp = NULL;
  sum0 = NULL;
  sumr0 = NULL;
  tmpnl = NULL;
  outbuf = (double*)SetPro_checked_malloc(sizeof(double)*out_len);

  {
    double *crude = g->crude;
    double *prod = g->prod;
    double *gnt2 = g->gnt2;
    double *c2r = g->c2r;
    int *c2r_off = g->c2r_off;
    int *sp_lfi = g->sp_lfi;
    int *sp_ncmb = g->sp_ncmb;
    int *cmb_L0 = g->cmb_L0;
    int *cmb_num0 = g->cmb_num0;
    const size_t crude_len = (size_t)SpeciesNum*GL_Mesh;
    const size_t prod_len = (size_t)SpeciesNum*cmb_max*cmb_max*nLL*GL_Mesh;
    const size_t gnt2_len = (size_t)(l0max+1)*nM*(l0max+1)*nM*nLL;
    const size_t c2r_len = 2*(size_t)c2r_off[l0max+1];

    for (chunk_start=0; chunk_start<all_pairs; chunk_start+=CHUNK){

      const int np = chunk_start>=OneD_Nloop?0:((OneD_Nloop-chunk_start)<CHUNK ? OneD_Nloop-chunk_start : CHUNK);
      int pp,LLh,mh;
      memset(sh_tab,0,(size_t)np*nsh*6*sizeof(double));

#pragma omp parallel for if(!omp_in_parallel()) schedule(dynamic) private(LLh,mh)
      for (pp=0; pp<np; pp++){
        int Mc_AN = OneD2Mc_AN[chunk_start+pp];
        int h_AN = OneD2h_AN[chunk_start+pp];
        int Gc_AN = M2G[Mc_AN];
        int Gh_AN = natn[Gc_AN][h_AN];
        int Rnh = ncn[Gc_AN][h_AN];
        int spe_pao,spe_vna;
        double S_co[3];
        double dx,dy,dz,r,theta,phi;
        double SH[2],dSHt[2],dSHp[2];

        if (mirror==0){
          spe_pao = WhatSpecies[Gc_AN];
          spe_vna = WhatSpecies[Gh_AN];
          dx = Gxyz[Gh_AN][1] + atv[Rnh][1] - Gxyz[Gc_AN][1];
          dy = Gxyz[Gh_AN][2] + atv[Rnh][2] - Gxyz[Gc_AN][2];
          dz = Gxyz[Gh_AN][3] + atv[Rnh][3] - Gxyz[Gc_AN][3];
        }
        else{
          spe_pao = WhatSpecies[Gh_AN];
          spe_vna = WhatSpecies[Gc_AN];
          dx = Gxyz[Gc_AN][1] - Gxyz[Gh_AN][1] - atv[Rnh][1];
          dy = Gxyz[Gc_AN][2] - Gxyz[Gh_AN][2] - atv[Rnh][2];
          dz = Gxyz[Gc_AN][3] - Gxyz[Gh_AN][3] - atv[Rnh][3];
        }

        xyz2spherical(dx,dy,dz,0.0,0.0,0.0,S_co);
        r = S_co[0];
        theta = S_co[1];
        phi = S_co[2];
        if (r<1.0e-10) r = 1.0e-10;

        pr_r[pp] = r;
        pr_siT[pp] = sin(theta);
        pr_coT[pp] = cos(theta);
        pr_siP[pp] = sin(phi);
        pr_coP[pp] = cos(phi);
        pr_spe[pp] = spe_pao;
        pr_vspe[pp] = spe_vna;
        pr_h0[pp] = (h_AN==0);

        for (LLh=0; LLh<=sp_lfi[spe_pao]; LLh++){
          for (mh=-LLh; mh<=LLh; mh++){
            size_t off = ((size_t)pp*nsh + (size_t)LLh*LLh + (mh+LLh))*6;

            ComplexSH(LLh,mh,theta,phi,SH,dSHt,dSHp);
            sh_tab[off+0] = SH[0];
            sh_tab[off+1] = SH[1];
            sh_tab[off+2] = dSHt[0];
            sh_tab[off+3] = dSHt[1];
            sh_tab[off+4] = dSHp[0];
            sh_tab[off+5] = dSHp[1];
          }
        }
      }

      SetProVNA23CapacityParameters parameters={GL_Mesh,g->l0max,g->lfi_max,g->cmb_max,g->mn_max,g->tno_max,Dk};
      OpenMXGpuTask task={0};task.count=np;task.inputs=20;task.outputs=1;task.scratch=5;
      task.parameters=&parameters;task.parameter_bytes=sizeof(parameters);task.kernel=SetPro_VNA23_capacity_kernel;
      task.view[0]=(OpenMXGpuTaskView){crude,(crude_len)*sizeof(double),0};
      task.view[1]=(OpenMXGpuTaskView){prod,(prod_len)*sizeof(double),0};
      task.view[2]=(OpenMXGpuTaskView){gnt2,(gnt2_len)*sizeof(double),0};
      task.view[3]=(OpenMXGpuTaskView){c2r,(c2r_len)*sizeof(double),0};
      task.view[4]=(OpenMXGpuTaskView){c2r_off,(l0max+2)*sizeof(int),0};
      task.view[5]=(OpenMXGpuTaskView){sp_lfi,(SpeciesNum)*sizeof(int),0};
      task.view[6]=(OpenMXGpuTaskView){sp_ncmb,(SpeciesNum)*sizeof(int),0};
      task.view[7]=(OpenMXGpuTaskView){cmb_L0,((size_t)SpeciesNum*cmb_max)*sizeof(int),0};
      task.view[8]=(OpenMXGpuTaskView){cmb_num0,((size_t)SpeciesNum*cmb_max)*sizeof(int),0};
      task.view[9]=(OpenMXGpuTaskView){GL_NormK,(GL_Mesh)*sizeof(double),0};
      task.view[10]=(OpenMXGpuTaskView){(void*)weights,(GL_Mesh)*sizeof(double),0};
      task.view[11]=(OpenMXGpuTaskView){pr_r,0,(1)*sizeof(double)};
      task.view[12]=(OpenMXGpuTaskView){pr_siT,0,(1)*sizeof(double)};
      task.view[13]=(OpenMXGpuTaskView){pr_coT,0,(1)*sizeof(double)};
      task.view[14]=(OpenMXGpuTaskView){pr_siP,0,(1)*sizeof(double)};
      task.view[15]=(OpenMXGpuTaskView){pr_coP,0,(1)*sizeof(double)};
      task.view[16]=(OpenMXGpuTaskView){pr_spe,0,(1)*sizeof(int)};
      task.view[17]=(OpenMXGpuTaskView){pr_vspe,0,(1)*sizeof(int)};
      task.view[18]=(OpenMXGpuTaskView){pr_h0,0,(1)*sizeof(int)};
      task.view[19]=(OpenMXGpuTaskView){sh_tab,0,((size_t)nsh*6)*sizeof(double)};
      task.view[20]=(OpenMXGpuTaskView){outbuf,0,4*(size_t)mn_max*sizeof(double)};
      task.view[21]=(OpenMXGpuTaskView){NULL,0,((size_t)nLL*GL_Mesh)*sizeof(double)};
      task.view[22]=(OpenMXGpuTaskView){NULL,0,((size_t)nLL*GL_Mesh)*sizeof(double)};
      task.view[23]=(OpenMXGpuTaskView){NULL,0,((size_t)nLL*cmb_max*cmb_max)*sizeof(double)};
      task.view[24]=(OpenMXGpuTaskView){NULL,0,((size_t)nLL*cmb_max*cmb_max)*sizeof(double)};
      task.view[25]=(OpenMXGpuTaskView){NULL,0,(elem_per_pair*8)*sizeof(double)};
      int all_done;
      const int done=SetPro_BatchAttempt(&task,"SetPro_VNA23",&all_done);
      if (!all_done) {
        /* a late device failure on some rank: every rank's unfinished pairs,
           and the rest of HVNA2/3, go to the host SIMD assembly */
        g->host_route = 1;
        g->host_first = done ? (chunk_start+np<OneD_Nloop ? chunk_start+np : OneD_Nloop) : chunk_start;
        SetPro_note("HVNA2/3 GPU batch","HVNA%d device batch failed on %s; pairs %d..%d of this rank go to the host SIMD assembly",
                    mirror ? 3 : 2,done ? "another rank" : "this rank",g->host_first,OneD_Nloop-1);
      }


      /* host: scatter into HVNA2 or HVNA3 (none after a failure here) */

      const int np_done = done ? np : 0;
#pragma omp parallel for if(!omp_in_parallel()) schedule(dynamic)
      for (pp=0; pp<np_done; pp++){
        int Mc_AN = OneD2Mc_AN[chunk_start+pp];
        int h_AN = OneD2h_AN[chunk_start+pp];
        int tno = g->sp_tno[pr_spe[pp]];
        int kk,i2,j2;

        for (kk=0; kk<=3; kk++){
          for (i2=0; i2<tno; i2++){
            for (j2=0; j2<tno; j2++){
              HVNA23[kk][Mc_AN][h_AN][i2][j2] =
                outbuf[((size_t)pp*4 + kk)*(size_t)mn_max
                       + (size_t)i2*g->tno_max + j2];
            }
          }
        }
      }
      if (!all_done) break;

    } /* chunk_start */

    {
      double t1;
      dtime(&t1);
      SetPro_note("HVNA2/3 GPU batch end","HVNA%d: %.2f s",mirror ? 3 : 2,t1-vna23_t0);
    }
  }

  free(outbuf);
  free(tmpnl);
  free(sumr0);
  free(sum0);
  free(sphbp);
  free(sphb);
  free(sh_tab);
  free(pr_h0);
  free(pr_vspe);
  free(pr_spe);
  free(pr_coP);
  free(pr_siP);
  free(pr_coT);
  free(pr_siT);
  free(pr_r);
}
static void SetPro_VNA23_GpuRun(double *****matrix,int mirror,int count,const int *mc,const int *h,double *GL_NormK,const double *weights,SetProGpu3Context *g)
{SetPro_VNA23_GpuRun_Part(matrix,mirror,count,mc,h,g,GL_NormK,weights);}


#include "openmx_dsvna_cpu.h"
#include "openmx_hvna_cpu.h"

/* Late failure of the device HVNA batch on this rank: its HVNA blocks on the
   host SIMD path (openmx_core_vna_hvna_block through the CPU pass's cache),
   from the archives the device path built: local blocks from the flat archive,
   received ones from the halo archive (copied back once when device-resident).
   The CPU pass's orientation, member/group order, first-k initialisation and
   damping; the legacy rows under OPENMX_HOST_SIMD=legacy or a refused block.
   Rank-local: no collectives. */
static void SetPro_HVNA_HostArchives(SetProGpuContext *g,double ****HVNA)
{
  const int fp32=openmx_fp32_enabled(),num_proj=g->num_proj;
  const float *flat=g->flat_host,*halo=g->halo_host;
  float *flat_copy=NULL,*halo_copy=NULL;
  int max_orbitals=1,max_blocks=1;

  if (!Matomnum) return;
  if (!flat) {
    flat_copy=(float*)malloc(g->flat_count*sizeof(float));
    if (!flat_copy) SetPro_gpu_abort("Set_ProExpn_VNA: no host memory to recover the local projector archive.");
    if (cudaMemcpy(flat_copy,g->flat_dev,g->flat_count*sizeof(float),cudaMemcpyDeviceToHost)!=cudaSuccess)
      SetPro_gpu_abort("Set_ProExpn_VNA: cannot recover an existing projector archive.");
    flat=flat_copy;
  }
  if (!halo && g->halo_count) {
    halo_copy=(float*)malloc(g->halo_count*sizeof(float));
    if (!halo_copy) SetPro_gpu_abort("Set_ProExpn_VNA: no host memory to recover the halo projector archive.");
    if (cudaMemcpy(halo_copy,g->halo_dev,g->halo_count*sizeof(float),cudaMemcpyDeviceToHost)!=cudaSuccess)
      SetPro_gpu_abort("Set_ProExpn_VNA: cannot recover an existing projector archive.");
    halo=halo_copy;
  }
  for (int spe=0; spe<SpeciesNum; spe++) if (max_orbitals<Spe_Total_NO[spe]) max_orbitals=Spe_Total_NO[spe];
  for (int mc=1; mc<=Matomnum; mc++) if (max_blocks<FNAN[M2G[mc]]+1) max_blocks=FNAN[M2G[mc]]+1;
  const OpenMXHvnaProjectorBuffers projectors={SpeciesNum,g->num_rvna,num_proj,Spe_Total_NO,g->l2p1,g->ene};
  SetPro_note("HVNA host SIMD recovery","%d local atoms from the %s archives",Matomnum,g->host_archive ? "host" : "device");

#pragma omp parallel
  {
    /* this worker's atom: its transposed left blocks (mc, k) serve all of its pairs */
    SetProHvnaHost host;
    SetPro_HvnaHost_Init(&host,0,max_blocks,num_proj,max_orbitals,fp32);
    double *matrix=(double*)SetPro_checked_malloc(sizeof(double)*(size_t)max_orbitals*(size_t)max_orbitals);
    const OpenMXHvnaWorkspaceBuffers workspace={(size_t)max_orbitals,matrix};
#pragma omp for schedule(dynamic)
    for (int mc=1; mc<=Matomnum; mc++) {
      const int gc=M2G[mc],rows=Spe_Total_NO[WhatSpecies[gc]];
      SetPro_HvnaHost_Reset(&host,mc);
      for (int j=0; j<=FNAN[gc]; j++) {
        const int jg=natn[gc][j],mj=F_G2M[jg],columns=Spe_Total_NO[WhatSpecies[jg]];
        int initialized=0;
        for (int k=0; k<=FNAN[gc]; k++) {
          const int kl=RMI1[mc][j][k];
          if (kl<0) continue;
          const int projector_species=WhatSpecies[natn[gc][k]];
          const float *left=flat+g->mck_off[g->mck_base[mc]+k];
          const float *right=mj<=Matomnum ? flat+g->mck_off[g->mck_base[mj]+kl]
                                          : halo+g->halo_off[g->halo_base[mj-Matomnum]+kl];
          if (!SetPro_HVNA_Host_Block(&host,&projectors,&workspace,rows,columns,projector_species,
                  left,(size_t)num_proj,k,right,(size_t)num_proj,kl,!initialized,0,fp32))
            for (int m=0; m<rows; m++)
              SetPro_HVNA_CPU_Row(&projectors,left,(size_t)num_proj,right,(size_t)num_proj,projector_species,
                  columns,m,matrix,(size_t)max_orbitals,!initialized,fp32);
          initialized=1;
        }
        const double dmp=dampingF(Spe_Atom_Cut1[WhatSpecies[gc]]+Spe_Atom_Cut1[WhatSpecies[jg]],Dis[gc][j]);
        for (int m=0; m<rows; m++)
          for (int n=0; n<columns; n++)
            HVNA[mc][j][m][n]=dmp*(initialized ? matrix[(size_t)m*max_orbitals+n] : 0.0);
      }
    }
    free(matrix);
    SetPro_HvnaHost_Free(&host);
  }
  free(flat_copy);
  free(halo_copy);
}

typedef SetProGpuContext SetProHvnaDispatch;
static int SetPro_HVNA_DispatchBegin(const OpenMXReciprocalViews *reciprocal, SetProHvnaDispatch *g,Type_DS_VNA *****v,
    const OpenMXVnaPreparation *preparation,int n)
{ return SetPro_HVNA_GpuBegin(reciprocal, g,v,preparation,n); }
static void SetPro_HVNA_DispatchArchive(SetProHvnaDispatch *g,Type_DS_VNA *****v,int m,int global)
{ SetPro_HVNA_GpuArchiveHalo(g,v,m,global); }
static int SetPro_HVNA_DispatchRun(SetProHvnaDispatch *g,double ****h)
{
  int done=1;
  for (int turn=0;turn<g->turns;++turn) {
    if (turn==g->turn) done=SetPro_HVNA_GpuRun(g,h);
    if (g->turns>1) MPI_Barrier(mpi_comm_level1);
  }
  if (OpenMX_HostSimdMode()!=OPENMX_HOST_SIMD_LEGACY) {
    /* every rank, once: a failed device batch completes on the host SIMD path
       from the archives (rank-local), never in the capacity recovery */
    int failed=!done,any=0;
    MPI_Allreduce(&failed,&any,1,MPI_INT,MPI_MAX,mpi_comm_level1);
    if (failed) SetPro_HVNA_HostArchives(g,h);
    else if (any) SetPro_note("HVNA GPU batch","another rank recovers its HVNA on the host SIMD path");
  }
  SetPro_HVNA_GpuEnd(g);
  return 1;
}

#include "openmx_vna_assembly.h"


#define xmin  0.0
#define asize_lmax 30

static inline void Spherical_Bessel2( double x, int lmax, double *sb, double *dsb )
{
  int m,n,nmax;
  double tsb[asize_lmax+10];
  double invx,vsb0,vsb1,vsb2,vsbi;
  double j0,j1,j0p,j1p,sf,tmp,si,co,ix,ix2;

  if (x<0.0){
    printf("minus x is invalid for Spherical_Bessel\n");
    exit(0);
  }

  /* find an appropriate nmax */

  nmax = lmax + 3*x + 20;
  if (nmax<100) nmax = 100;

  if (asize_lmax<(lmax+1)){
    printf("asize_lmax should be larger than %d in Spherical_Bessel.c\n",lmax+1);
    exit(0);
  }

  /* if x is larger than xmin */

  if ( xmin < x ){

    invx = 1.0/x;

    /* initial values */

    vsb0 = 0.0;
    vsb1 = 1.0e-14;

    /* downward recurrence from nmax-2 to lmax+2 */

    for ( n=nmax-1; (lmax+2)<n; n-- ){

      vsb2 = (2.0*n + 1.0)*invx*vsb1 - vsb0;

      if (1.0e+250<vsb2){
        tmp = 1.0/vsb2;
        vsb2 *= tmp;
        vsb1 *= tmp;
      }

      vsbi = vsb0;
      vsb0 = vsb1;
      vsb1 = vsb2;
    }

    /* downward recurrence from lmax+1 to 0 */

    n = lmax + 3;
    tsb[n-1] = vsb1;
    tsb[n  ] = vsb0;
    tsb[n+1] = vsbi;

    tmp = tsb[n-1];
    tsb[n-1] /= tmp;
    tsb[n  ] /= tmp;

    for ( n=lmax+2; 0<n; n-- ){

      tsb[n-1] = (2.0*n + 1.0)*invx*tsb[n] - tsb[n+1];

      if (1.0e+250<tsb[n-1]){
        tmp = tsb[n-1];
        for (m=n-1; m<=lmax+1; m++){
          tsb[m] /= tmp;
        }
      }
    }

    /* normalization */

    si = sin(x);
    co = cos(x);
    ix = 1.0/x;
    ix2 = ix*ix;
    j0 = si*ix;
    j1 = si*ix*ix - co*ix;

    if (fabs(tsb[1])<fabs(tsb[0])) sf = j0/tsb[0];
    else                           sf = j1/tsb[1];

    /* tsb to sb */

    for ( n=0; n<=lmax+1; n++ ){
      sb[n] = tsb[n]*sf;
    }

    /* derivative of sb */

    dsb[0] = co*ix - si*ix*ix;
    for ( n=1; n<=lmax; n++ ){
      dsb[n] = ( (double)n*sb[n-1] - (double)(n+1.0)*sb[n+1] )/(2.0*(double)n + 1.0);
    }

  }

  /* if x is smaller than xmin */

  else {

    /* sb */

    for ( n=0; n<=lmax+1; n++ ){
      sb[n] = 0.0;
    }
    sb[0] = 1.0;

    /* derivative of sb */

    dsb[0] = 0.0;
    for ( n=1; n<=lmax; n++ ){
      dsb[n] = ( (double)n*sb[n-1] - (double)(n+1.0)*sb[n+1] )/(2.0*(double)n + 1.0);
    }
  }
}

#include "openmx_target.h"
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
#include <limits.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <sys/types.h>
#include <sys/times.h>
#include <sys/time.h> 
#include <sys/stat.h>
#include <unistd.h>
#include "openmx_common.h"
#include "openmx_gpu_workers.h"
#include "openmx_gpu_views.h"
#include "openmx_gpu_task.h"
#include "openmx_host_memory.h"
#include "openmx_precision.h"
#include "set_hip_default_device_from_local_rank.h"
#include "mpi.h"
#include <omp.h>

extern int openmx_gpu_is_apu(void);

static double Set_ProExpn(double ****HVNA, Type_DS_VNA *****DS_VNA);
static double Set_VNA2(double ****HVNA, double *****HVNA2);
static double Set_VNA3(double *****HVNA3);
static int SetPro_VNA23_GpuBegin(void);
static void SetPro_VNA23_GpuEnd(void);
/* forward: SetPro_DSVNA_GpuBegin releases its tables through this when the
   device-memory check sends the DS_VNA build down the CPU path */
static void SetPro_DSVNA_GpuEnd(void);
/* forward: the OPENMX_SETPRO_VERBOSE diagnostics (defined below the driver) */
static void SetPro_note(const char *stage, const char *fmt, ...);
static int SetPro_gpu_used = 0;

int Set_ProExpn_VNA_Used_GPU(void)
{
  return SetPro_gpu_used;
}

#ifdef kcomp
static void Spherical_Bessel2( double x, int lmax, double *sb, double *dsb );
#else 
static inline void Spherical_Bessel2( double x, int lmax, double *sb, double *dsb );
#endif      


double Set_ProExpn_VNA(double ****HVNA, double *****HVNA2, Type_DS_VNA *****DS_VNA)
{
  double time1,time2,time3;
  int vna23_gpu;

  SetPro_gpu_used = 0;
  SetPro_note("entry","");

  /* separable form */

  time1 = Set_ProExpn(HVNA, DS_VNA);

  /* batched device path for the one-centre pair integrals */

  vna23_gpu = SetPro_VNA23_GpuBegin();

  /* one-center (orbitals) but two-center integrals, <Phi_{LM,L'M'}|VNA> */

  time2 = Set_VNA2(HVNA, HVNA2);

  /* one-center (orbitals) but two-center integrals, <VNA|Phi_{LM,L'M'}> */

  time3 = Set_VNA3(HVNA3);

  if (vna23_gpu) SetPro_VNA23_GpuEnd();

  SetPro_note("exit","Set_ProExpn %.2f s, Set_VNA2 %.2f s, Set_VNA3 %.2f s",time1,time2,time3);

  if (measure_time){
    printf("Time Set_ProExpn=%7.3f Set_VNA2=%7.3f Set_VNA3=%7.3f\n",time1,time2,time3);
  }

  return time1+time2+time3;
}



/* ------------------------------------------------------------------ */
/* Batched device evaluation of the HVNA projector traces.

   HVNA[Mc][j][m][n] = dmp * sum_k sum_L ene_L
                           * sum_l DS_VNA[0][Mc][k][m][l]*DS_VNA[0][Mj][kl][n][l]

   is the same energy-weighted float dot-product contraction as the
   Force4B/Force_HNL first-case traces.  The kernel keeps the exact CPU
   accumulation order (k outermost, then the radial groups, then the
   2*L+1 members with the float product cast to double), so only the
   final float rounding of DS_VNA itself limits the agreement.  The MPI
   pipeline is untouched: received halo blocks are archived and one
   batched kernel runs after the loops.                                */
/* ------------------------------------------------------------------ */

typedef struct {
  int enabled;
  int num_proj;
  int num_rvna;
  int turn, turns;      /* ranks admitted together by the device-memory check */

  /* local direction-0 DS_VNA rows */
  float *flat;
  size_t *mck_off;
  int *mck_base;

  /* halo archive of the received direction-0 blocks */
  float *halo;
  size_t halo_count;
  size_t *halo_off;
  int *halo_base;

  /* per-species group energies and the group lengths */
  double *ene;
  int *l2p1;
} SetProGpuContext;

static SetProGpuContext SetPro_gpu = { 0 };

static void SetPro_gpu_abort(const char *message)
{
  fprintf(stderr,"%s\n",message);
  fflush(stderr);
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

static int SetPro_collective_env_flag(const char *name, int default_value)
{
  int myid,value = default_value;
  const char *env;

  MPI_Comm_rank(mpi_comm_level1,&myid);
  if (myid==0){
    env = getenv(name);
    if (env!=NULL && env[0]!='\0') value = (atoi(env)!=0);
  }
  MPI_Bcast(&value,1,MPI_INT,0,mpi_comm_level1);

  return value;
}

/* The OpenMP-target projector batches are tuned for discrete accelerators.
   On a unified-memory APU, several MPI ranks issuing the small batches to one
   device are slower than the already distributed host implementation.  Keep
   an explicit environment override for experiments and future kernels. */
static int SetPro_gpu_default_enabled(void)
{
  return openmx_gpu_is_apu() ? 0 : 1;
}

/* OPENMX_SETPRO_GPU remains the umbrella for compatibility.  The three
   stages can be selected independently so force derivatives do not have to
   share the policy of the energy-only HVNA contraction. */
static int SetPro_gpu_stage_enabled(const char *stage_name)
{
  int enabled = SetPro_collective_env_flag("OPENMX_SETPRO_GPU",
                                            SetPro_gpu_default_enabled());
  return SetPro_collective_env_flag(stage_name,enabled);
}

/* OPENMX_SETPRO_VERBOSE=1: every rank reports each device stage of this
   file (the admission figures, the batch sizes, the elapsed time) together
   with the device memory free at that moment, on stderr.  Rank-local. */
static int SetPro_verbose(void)
{
  static int verbose = -1;

  if (verbose<0){
    const char *env = getenv("OPENMX_SETPRO_VERBOSE");
    verbose = (env!=NULL && env[0]!='\0' && atoi(env)!=0) ? 1 : 0;
  }
  return verbose;
}

static double SetPro_free_mib(void)
{
  size_t free_bytes = 0,total_bytes = 0;

  /* a CPU run must not create a HIP context for a diagnostic */
  if (scf_eigen_lib_flag!=GPUSOLVER) return -1.0;
  if (OpenMX_GpuMemGetInfo(&free_bytes,&total_bytes)!=hipSuccess) return -1.0;
  return (double)free_bytes/1048576.0;
}

static void SetPro_note(const char *stage, const char *fmt, ...)
{
  char detail[512];
  va_list ap;
  int myid;

  if (!SetPro_verbose()) return;
  va_start(ap,fmt);
  vsnprintf(detail,sizeof(detail),fmt,ap);
  va_end(ap);
  MPI_Comm_rank(mpi_comm_level1,&myid);
  {
    const double mib = SetPro_free_mib();

    if (mib<0.0) fprintf(stderr,"<Set_ProExpn_VNA> rank %d %s: %s\n",myid,stage,detail);
    else fprintf(stderr,"<Set_ProExpn_VNA> rank %d %s: %s; device free %.0f MiB\n",myid,stage,detail,mib);
  }
  fflush(stderr);
}

/* MemAvailable of /proc/meminfo in KiB, ULLONG_MAX when it cannot be read
   (no host cap: the HVNA host archives exist on every path anyway) */
static unsigned long long SetPro_host_available_kib(void)
{
  unsigned long long kib = ULLONG_MAX;
  char line[256];
  FILE *file = fopen("/proc/meminfo","r");

  if (file==NULL) return ULLONG_MAX;
  while (fgets(line,sizeof(line),file)!=NULL){
    unsigned long long value;
    if (sscanf(line,"MemAvailable: %llu kB",&value)==1){
      kib = value;
      break;
    }
  }
  fclose(file);
  return kib;
}

/* 512-byte aligned sub-allocation of one device arena */
static size_t SetPro_arena_off(size_t *pos, size_t bytes)
{
  size_t off = *pos;

  *pos = (off + bytes + 511U) & ~(size_t)511U;
  return off;
}

/* One-shot device arena of the HVNA batch.  NULL lets the caller contract
   on the host: a failed "omp target data" map would abort every rank. */
static void *SetPro_device_try(size_t bytes)
{
  void *ptr = NULL;

  if (bytes==0) bytes = 1;
  if (hipMalloc(&ptr,bytes)==hipSuccess && ptr!=NULL) return ptr;
  (void)hipDeviceSynchronize();
  ptr = NULL;
  if (hipMalloc(&ptr,bytes)==hipSuccess && ptr!=NULL) return ptr;
  /* the caller contracts on the host: do not leave the failure as HIP's
     sticky last error for a later kernel-launch check elsewhere */
  (void)hipGetLastError();
  return NULL;
}

/* One HVNA output element in the exact CPU accumulation order (k outermost,
   then the radial groups, then the 2*L+1 members).  Preserve legacy FP32
   product rounding when enabled; promote operands before multiplying when
   scf.fp32.enable is off. Shared by the device kernel and host fallback. */
#pragma omp declare target
static double SetPro_HVNA_element(size_t oe, int num_proj, int num_rvna,
                                  const float *flat, const float *halo,
                                  const int *out_item, const int *item_koff,
                                  const int *item_kn, const int *item_tnoH,
                                  const size_t *item_out, const size_t *krowA,
                                  const size_t *krowB, const int *krow_halo,
                                  const int *krow_ene, const double *ene,
                                  const int *l2p1, int fp32)
{
  const int pp = out_item[oe];
  const int tnoH = item_tnoH[pp];
  const int k_off = item_koff[pp];
  const int k_n = item_kn[pp];
  const int idx = (int)(oe - item_out[pp]);
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

  return nlh;
}
#pragma omp end declare target

static int SetPro_HVNA_GpuBegin(Type_DS_VNA *****DS_VNA, int *VNA_List,
                                int *VNA_List2, int Num_RVNA)
{
  SetProGpuContext *g = &SetPro_gpu;
  int Mc_AN,k,spe,L1,node_ranks = 1,node_rank = 0;
  size_t pos = 0,slots = 0,halo_slots = 0;
  size_t free_bytes = 0,total_bytes = 0;
  MPI_Comm node_comm = MPI_COMM_NULL;
  int allow_turns;

  memset(g,0,sizeof(*g));
  g->num_proj = (List_YOUSO[35]+1)*(List_YOUSO[35]+1)*List_YOUSO[34];
  g->num_rvna = Num_RVNA;

  if (scf_eigen_lib_flag!=GPUSOLVER) return 0;
  if (!SetPro_gpu_stage_enabled("OPENMX_SETPRO_HVNA_GPU")) return 0;
  /* OPENMX_SETPRO_HVNA_TURNS=0: admit the batch only when every rank of
     the node fits on the device at once (no rank turns) */
  allow_turns = SetPro_collective_env_flag("OPENMX_SETPRO_HVNA_TURNS",1);

  MPI_Comm_split_type(mpi_comm_level1,MPI_COMM_TYPE_SHARED,0,MPI_INFO_NULL,&node_comm);
  MPI_Comm_size(node_comm,&node_ranks);
  MPI_Comm_rank(node_comm,&node_rank);
  if (node_ranks<1) node_ranks = 1;
    if (hipDeviceSynchronize()==hipSuccess){
  }
  MPI_Barrier(node_comm);

  /* sizes of the flat local table and the halo archive */

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    int Gc_AN = M2G[Mc_AN];
    int tno = Spe_Total_NO[WhatSpecies[Gc_AN]];

    slots += (size_t)(FNAN[Gc_AN]+1);
    pos += (size_t)(FNAN[Gc_AN]+1)*(size_t)tno*(size_t)g->num_proj;
  }

  for (Mc_AN=Matomnum+1; Mc_AN<=Matomnum+MatomnumF; Mc_AN++){
    int Gc_AN = F_M2G[Mc_AN];
    int tno = Spe_Total_NO[WhatSpecies[Gc_AN]];

    halo_slots += (size_t)(FNAN[Gc_AN]+1);
    g->halo_count += (size_t)(FNAN[Gc_AN]+1)*(size_t)tno*(size_t)g->num_proj;
  }

  if (OpenMX_GpuMemGetInfo(&free_bytes,&total_bytes)!=hipSuccess) free_bytes = 0;
  {
    const size_t reserve = (size_t)256*1024*1024;
    size_t items = 0, kslots = 0, outputs = 0;
    unsigned long long need, max_need, available = free_bytes, min_available;
    int admitted = 0, fit, global_fit, local_turns;

    /* Count the same arrays mapped by GpuRun; the previous fixed transient
       allowance underestimated large neighbour tables. */
    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++) {
      int gc = M2G[Mc_AN];
      int tno = Spe_Total_NO[WhatSpecies[gc]];
      for (int j=0; j<=FNAN[gc]; ++j) {
        ++items;
        outputs += (size_t)tno * Spe_Total_NO[WhatSpecies[natn[gc][j]]];
        for (int kk=0; kk<=FNAN[gc]; ++kk)
          if (0<=RMI1[Mc_AN][j][kk]) ++kslots;
      }
    }
    if (kslots==0) kslots = 1;
    need = ((pos==0 ? 1 : pos) + (g->halo_count==0 ? 1 : g->halo_count))*sizeof(float)
         + items*(4*sizeof(int) + sizeof(size_t))
         + outputs*(sizeof(int) + sizeof(double))
         + kslots*(2*sizeof(size_t) + 2*sizeof(int))
         + (size_t)SpeciesNum*Num_RVNA*sizeof(double) + (size_t)Num_RVNA*sizeof(int)
         + (size_t)64*1024*1024;
    MPI_Allreduce(&need,&max_need,1,MPI_UNSIGNED_LONG_LONG,MPI_MAX,node_comm);
    MPI_Allreduce(&available,&min_available,1,MPI_UNSIGNED_LONG_LONG,MPI_MIN,node_comm);
    if (min_available>reserve) {
      unsigned long long count = (min_available-reserve)/max_need;
      admitted = count<(unsigned)node_ranks ? (int)count : node_ranks;
    }
    fit = admitted>0 && (allow_turns || admitted==node_ranks);
    local_turns = fit ? (node_ranks+admitted-1)/admitted : 0;
    {
      /* The host archives exist on every path here (each rank maps them in
         its turn), so bound their node-wide storage plus the batch tables
         by half of the available RAM, whatever the turn count.  An
         unreadable /proc/meminfo sets no cap. */
      unsigned long long host_kib = SetPro_host_available_kib(),min_host,total_need;

      MPI_Allreduce(&host_kib,&min_host,1,MPI_UNSIGNED_LONG_LONG,MPI_MIN,node_comm);
      MPI_Allreduce(&need,&total_need,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,node_comm);
      if (min_host<=ULLONG_MAX/512ULL && total_need>min_host*512ULL) fit = 0;
    }
    /* All ranks must enter the same turn barriers, including empty ranks. */
    MPI_Allreduce(&fit,&global_fit,1,MPI_INT,MPI_MIN,mpi_comm_level1);
    SetPro_note("HVNA GPU admission",
                "need %.0f MiB (node max %.0f), node min free %.0f MiB, %d of %d rank(s) per turn, "
                "%d turn(s), fit %d (global %d)",
                (double)need/1048576.0,(double)max_need/1048576.0,(double)min_available/1048576.0,
                admitted,node_ranks,local_turns,fit,global_fit);
    if (!global_fit){
      if (!fit && 3<=level_stdout){
        fprintf(stderr,
                "Set_ProExpn_VNA HVNA GPU: %.2f GiB free for %d rank(s) but %.2f GiB needed per rank; CPU fallback.\n",
                (double)free_bytes/(1024.0*1024.0*1024.0),node_ranks,(double)need/(1024.0*1024.0*1024.0));
        fflush(stderr);
      }
      else if (!fit && node_rank==0 && 2<=level_stdout){
        fprintf(stderr,
                "Set_ProExpn_VNA HVNA GPU: %.3f GiB maximum per rank, %.3f GiB device free; "
                "insufficient device/host budget or rank turns disabled; CPU fallback.\n",
                (double)max_need/1073741824.0,(double)min_available/1073741824.0);
        fflush(stderr);
      }
      MPI_Comm_free(&node_comm);
      memset(g,0,sizeof(*g));
      return 0;
    }
    g->turn = node_rank/admitted;
    MPI_Allreduce(&local_turns,&g->turns,1,MPI_INT,MPI_MAX,mpi_comm_level1);
    if (node_rank==0 && local_turns>1) {
      if (2 <= level_stdout) {
          fprintf(stderr,"Set_ProExpn_VNA HVNA GPU: %d rank(s) per turn, %d turns, %.3f GiB maximum per rank.\n",
                  admitted,local_turns,(double)max_need/1073741824.0);
          fflush(stderr);
      }
    }
  }
  MPI_Comm_free(&node_comm);

  g->mck_base = (int*)SetPro_checked_malloc(sizeof(int)*(size_t)(Matomnum+2));
  g->mck_off = (size_t*)SetPro_checked_malloc(sizeof(size_t)*(slots==0 ? 1 : slots));
  g->flat = (float*)malloc(sizeof(float)*(pos==0 ? 1 : pos));
  g->halo_base = (int*)SetPro_checked_malloc(sizeof(int)*(size_t)(MatomnumF+2));
  g->halo_off = (size_t*)SetPro_checked_malloc(sizeof(size_t)*(halo_slots==0 ? 1 : halo_slots));
  g->halo = (float*)malloc(sizeof(float)*(g->halo_count==0 ? 1 : g->halo_count));
  g->ene = (double*)SetPro_checked_malloc(sizeof(double)*(size_t)SpeciesNum*(size_t)Num_RVNA);
  g->l2p1 = (int*)SetPro_checked_malloc(sizeof(int)*(size_t)Num_RVNA);

  /* The two host archives are the large claims.  Agree on them over level1:
     every rank must enter the same turn barriers, so a rank that cannot
     hold its archives sends all ranks to the CPU path instead of aborting. */
  {
    int allocated = (g->flat!=NULL && g->halo!=NULL),all_allocated;

    MPI_Allreduce(&allocated,&all_allocated,1,MPI_INT,MPI_MIN,mpi_comm_level1);
    if (!all_allocated){
      if (!allocated){
        if (2 <= level_stdout) {
            fprintf(stderr,"Set_ProExpn_VNA HVNA GPU: the %.3f GiB host archives could not be allocated; "
                    "CPU fallback on every rank.\n",
                    (double)(sizeof(float)*((pos==0 ? 1 : pos) + (g->halo_count==0 ? 1 : g->halo_count)))
                    /1073741824.0);
            fflush(stderr);
        }
      }
      free(g->l2p1);
      free(g->ene);
      free(g->halo);
      free(g->halo_off);
      free(g->halo_base);
      free(g->flat);
      free(g->mck_off);
      free(g->mck_base);
      memset(g,0,sizeof(*g));
      return 0;
    }
  }

  for (L1=0; L1<Num_RVNA; L1++){
    g->l2p1[L1] = 2*VNA_List[L1] + 1;
  }
  for (spe=0; spe<SpeciesNum; spe++){
    for (L1=0; L1<Num_RVNA; L1++){
      g->ene[(size_t)spe*Num_RVNA + L1] = VNA_proj_ene[spe][VNA_List[L1]][VNA_List2[L1]];
    }
  }

  {
    size_t flat_len = pos;

    pos = 0;
    slots = 0;
    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      int Gc_AN = M2G[Mc_AN];
      int tno = Spe_Total_NO[WhatSpecies[Gc_AN]];

      g->mck_base[Mc_AN] = (int)slots;
      for (k=0; k<=FNAN[Gc_AN]; k++){
        g->mck_off[slots] = pos;
        pos += (size_t)tno*(size_t)g->num_proj;
        slots++;
      }
    }

    halo_slots = 0;
    pos = 0;
    for (Mc_AN=Matomnum+1; Mc_AN<=Matomnum+MatomnumF; Mc_AN++){
      int Gc_AN = F_M2G[Mc_AN];
      int tno = Spe_Total_NO[WhatSpecies[Gc_AN]];

      g->halo_base[Mc_AN-Matomnum] = (int)halo_slots;
      for (k=0; k<=FNAN[Gc_AN]; k++){
        g->halo_off[halo_slots] = pos;
        pos += (size_t)tno*(size_t)g->num_proj;
        halo_slots++;
      }
    }

    /* flatten the local direction-0 rows */

#pragma omp parallel for schedule(dynamic)
    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      int Gc_AN = M2G[Mc_AN];
      int tno = Spe_Total_NO[WhatSpecies[Gc_AN]];
      int k2,i;

      for (k2=0; k2<=FNAN[Gc_AN]; k2++){
        size_t base = SetPro_gpu.mck_off[SetPro_gpu.mck_base[Mc_AN]+k2];

        for (i=0; i<tno; i++){
          memcpy(SetPro_gpu.flat + base + (size_t)i*(size_t)SetPro_gpu.num_proj,
                 DS_VNA[0][Mc_AN][k2][i],
                 sizeof(float)*(size_t)SetPro_gpu.num_proj);
        }
      }
    }

    {
      float *flat = g->flat;
      size_t len = (flat_len==0 ? 1 : flat_len);
    }

    SetPro_note("HVNA GPU archives","local archive %.0f MiB, halo archive %.0f MiB on the host "
                "(mapped in turn %d of %d)",
                (double)(flat_len==0 ? 1 : flat_len)*sizeof(float)/1048576.0,
                (double)(g->halo_count==0 ? 1 : g->halo_count)*sizeof(float)/1048576.0,
                g->turn+1,g->turns);
  }

  g->enabled = 1;
  return 1;
}

static void SetPro_HVNA_GpuEnd(void)
{
  SetProGpuContext *g = &SetPro_gpu;
  size_t flat_len = 0;
  int Mc_AN;

  if (!g->enabled) return;

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    int Gc_AN = M2G[Mc_AN];
    flat_len += (size_t)(FNAN[Gc_AN]+1)
              *(size_t)Spe_Total_NO[WhatSpecies[Gc_AN]]*(size_t)g->num_proj;
  }

  {
    float *flat = g->flat;
    size_t len = (flat_len==0 ? 1 : flat_len);
  }

  free(g->l2p1);
  free(g->ene);
  free(g->halo);
  free(g->halo_off);
  free(g->halo_base);
  free(g->flat);
  free(g->mck_off);
  free(g->mck_base);
  memset(g,0,sizeof(*g));
}

/* archive the direction-0 block of the halo atom just received */
static void SetPro_HVNA_GpuArchiveHalo(Type_DS_VNA *****DS_VNA, int Original_Mc_AN,
                                       int Gc_AN)
{
  SetProGpuContext *g = &SetPro_gpu;
  int tno = Spe_Total_NO[WhatSpecies[Gc_AN]];
  int halo_idx = Original_Mc_AN - Matomnum;
  int k;

  if (halo_idx<1 || MatomnumF<halo_idx){
    SetPro_gpu_abort("Set_ProExpn_VNA GPU: halo index out of range.");
  }

#pragma omp parallel for schedule(dynamic)
  for (k=0; k<=FNAN[Gc_AN]; k++){
    size_t base = g->halo_off[g->halo_base[halo_idx]+k];
    int i;

    for (i=0; i<tno; i++){
      memcpy(g->halo + base + (size_t)i*(size_t)g->num_proj,
             DS_VNA[0][Matomnum+1][k][i],
             sizeof(float)*(size_t)g->num_proj);
    }
  }
}

/* the unified batch over every (local atom, neighbour) pair */
static void SetPro_HVNA_GpuRun(double ****HVNA)
{
  const int fp32 = openmx_fp32_enabled();
  SetProGpuContext *g = &SetPro_gpu;
  const int num_proj = g->num_proj;
  const int num_rvna = g->num_rvna;
  int Mc_AN,j,k;
  int nitems = 0,kslots = 0;
  size_t out_count = 0;
  int *item_atom,*item_j,*item_koff,*item_kn,*item_tnoC,*item_tnoH,*out_item;
  size_t *item_out;
  size_t *krowA,*krowB;
  int *krow_halo,*krow_ene;
  double *out;
  int p;
  size_t opos = 0;
  int kpos = 0;
  double hvna_t0,hvna_t1;

  /* count */
  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
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

  if (nitems==0) return;

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
  out_item = (int*)SetPro_checked_malloc(sizeof(int)*(out_count==0 ? 1 : out_count));
  krowA = (size_t*)SetPro_checked_malloc(sizeof(size_t)*(size_t)(kslots==0 ? 1 : kslots));
  krowB = (size_t*)SetPro_checked_malloc(sizeof(size_t)*(size_t)(kslots==0 ? 1 : kslots));
  krow_halo = (int*)SetPro_checked_malloc(sizeof(int)*(size_t)(kslots==0 ? 1 : kslots));
  krow_ene = (int*)SetPro_checked_malloc(sizeof(int)*(size_t)(kslots==0 ? 1 : kslots));
  out = (double*)SetPro_checked_malloc(sizeof(double)*(out_count==0 ? 1 : out_count));

  /* build */
  p = 0;
  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
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
      for (int idx=0; idx<tnoC*tnoH; idx++) out_item[opos+(size_t)idx] = p;
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

  /* One explicit device arena instead of "omp target data" maps: a failed
     claim then falls back to the same contraction on the host (the archives
     are host arrays here) instead of aborting every rank, and no collective
     is needed inside the turn. */
  {
    const float *flat = g->flat;
    const float *halo = g->halo;
    const double *ene = g->ene;
    const int *l2p1 = g->l2p1;
    size_t flat_len = 0;
    const size_t halo_len = (g->halo_count==0 ? 1 : g->halo_count);
    const size_t ene_len = (size_t)SpeciesNum*(size_t)num_rvna;
    const size_t items_c = (size_t)nitems;
    const size_t kslots_c = (size_t)(kslots==0 ? 1 : kslots);
    const size_t out_c = out_count;
    const size_t out_n = (out_count==0 ? 1 : out_count);
    size_t asz = 0;
    size_t o_flat,o_halo,o_koff,o_kn,o_tnoH,o_iout,o_oitem,o_krowA,o_krowB,o_khalo,o_kene,o_ene,o_l2p1,o_out;
    char *arena;
    int on_device = 0;

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      int Gc_AN = M2G[Mc_AN];
      flat_len += (size_t)(FNAN[Gc_AN]+1)
                *(size_t)Spe_Total_NO[WhatSpecies[Gc_AN]]*(size_t)num_proj;
    }
    if (flat_len==0) flat_len = 1;

    o_flat = SetPro_arena_off(&asz,sizeof(float)*flat_len);
    o_halo = SetPro_arena_off(&asz,sizeof(float)*halo_len);
    o_koff = SetPro_arena_off(&asz,sizeof(int)*items_c);
    o_kn = SetPro_arena_off(&asz,sizeof(int)*items_c);
    o_tnoH = SetPro_arena_off(&asz,sizeof(int)*items_c);
    o_iout = SetPro_arena_off(&asz,sizeof(size_t)*items_c);
    o_oitem = SetPro_arena_off(&asz,sizeof(int)*out_n);
    o_krowA = SetPro_arena_off(&asz,sizeof(size_t)*kslots_c);
    o_krowB = SetPro_arena_off(&asz,sizeof(size_t)*kslots_c);
    o_khalo = SetPro_arena_off(&asz,sizeof(int)*kslots_c);
    o_kene = SetPro_arena_off(&asz,sizeof(int)*kslots_c);
    o_ene = SetPro_arena_off(&asz,sizeof(double)*ene_len);
    o_l2p1 = SetPro_arena_off(&asz,sizeof(int)*(size_t)num_rvna);
    o_out = SetPro_arena_off(&asz,sizeof(double)*out_n);

    arena = (char*)SetPro_device_try(asz);
    if (arena!=NULL){
      on_device =
           hipMemcpy(arena+o_flat,flat,sizeof(float)*flat_len,hipMemcpyHostToDevice)==hipSuccess
        && hipMemcpy(arena+o_halo,halo,sizeof(float)*halo_len,hipMemcpyHostToDevice)==hipSuccess
        && hipMemcpy(arena+o_koff,item_koff,sizeof(int)*items_c,hipMemcpyHostToDevice)==hipSuccess
        && hipMemcpy(arena+o_kn,item_kn,sizeof(int)*items_c,hipMemcpyHostToDevice)==hipSuccess
        && hipMemcpy(arena+o_tnoH,item_tnoH,sizeof(int)*items_c,hipMemcpyHostToDevice)==hipSuccess
        && hipMemcpy(arena+o_iout,item_out,sizeof(size_t)*items_c,hipMemcpyHostToDevice)==hipSuccess
        && hipMemcpy(arena+o_oitem,out_item,sizeof(int)*out_c,hipMemcpyHostToDevice)==hipSuccess
        && hipMemcpy(arena+o_krowA,krowA,sizeof(size_t)*kslots_c,hipMemcpyHostToDevice)==hipSuccess
        && hipMemcpy(arena+o_krowB,krowB,sizeof(size_t)*kslots_c,hipMemcpyHostToDevice)==hipSuccess
        && hipMemcpy(arena+o_khalo,krow_halo,sizeof(int)*kslots_c,hipMemcpyHostToDevice)==hipSuccess
        && hipMemcpy(arena+o_kene,krow_ene,sizeof(int)*kslots_c,hipMemcpyHostToDevice)==hipSuccess
        && hipMemcpy(arena+o_ene,ene,sizeof(double)*ene_len,hipMemcpyHostToDevice)==hipSuccess
        && hipMemcpy(arena+o_l2p1,l2p1,sizeof(int)*(size_t)num_rvna,hipMemcpyHostToDevice)==hipSuccess;

      if (on_device){
        const float *flat_d = (const float*)(void*)(arena+o_flat);
        const float *halo_d = (const float*)(void*)(arena+o_halo);
        const int *koff_d = (const int*)(void*)(arena+o_koff);
        const int *kn_d = (const int*)(void*)(arena+o_kn);
        const int *tnoH_d = (const int*)(void*)(arena+o_tnoH);
        const size_t *iout_d = (const size_t*)(void*)(arena+o_iout);
        const int *oitem_d = (const int*)(void*)(arena+o_oitem);
        const size_t *krowA_d = (const size_t*)(void*)(arena+o_krowA);
        const size_t *krowB_d = (const size_t*)(void*)(arena+o_krowB);
        const int *khalo_d = (const int*)(void*)(arena+o_khalo);
        const int *kene_d = (const int*)(void*)(arena+o_kene);
        const double *ene_d = (const double*)(void*)(arena+o_ene);
        const int *l2p1_d = (const int*)(void*)(arena+o_l2p1);
        double *out_d = (double*)(void*)(arena+o_out);

#pragma omp target teams distribute parallel for thread_limit(256) \
        firstprivate(fp32) \
        is_device_ptr(flat_d, halo_d, koff_d, kn_d, tnoH_d, iout_d, oitem_d, \
                      krowA_d, krowB_d, khalo_d, kene_d, ene_d, l2p1_d, out_d)
        for (size_t oe=0; oe<out_c; oe++){
          out_d[oe] = SetPro_HVNA_element(oe,num_proj,num_rvna,flat_d,halo_d,oitem_d,
                                          koff_d,kn_d,tnoH_d,iout_d,krowA_d,krowB_d,
                                          khalo_d,kene_d,ene_d,l2p1_d,fp32);
        }

        on_device = hipMemcpy(out,arena+o_out,sizeof(double)*out_c,hipMemcpyDeviceToHost)==hipSuccess;
        if (on_device && out_c > 0) SetPro_gpu_used = 1;
      }
      (void)hipFree(arena);
    }

    if (!on_device){
      (void)hipGetLastError();   /* a failed copy must not linger either */
      if (2 <= level_stdout) {
          fprintf(stderr,"Set_ProExpn_VNA HVNA GPU: the %.1f MiB device batch could not be staged in this "
                  "rank's turn; contracting on the host.\n",(double)asz/1048576.0);
          fflush(stderr);
      }

#pragma omp parallel for schedule(static)
      for (size_t oe=0; oe<out_c; oe++){
        out[oe] = SetPro_HVNA_element(oe,num_proj,num_rvna,flat,halo,out_item,
                                      item_koff,item_kn,item_tnoH,item_out,krowA,krowB,
                                      krow_halo,krow_ene,ene,l2p1,fp32);
      }
    }
  }

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

  free(out);
  free(krow_ene);
  free(krow_halo);
  free(krowB);
  free(krowA);
  free(item_out);
  free(item_tnoH);
  free(item_tnoC);
  free(out_item);
  free(item_kn);
  free(item_koff);
  free(item_j);
  free(item_atom);

  dtime(&hvna_t1);
  SetPro_note("HVNA GPU batch end","%.2f s",hvna_t1-hvna_t0);
}

typedef struct {
  int enabled;
  int turn, turns, host_archive;
  int num_proj;
  int num_rvna;

  /* Local direction-0 DS_VNA rows. Keep them only on the device when all
     local ranks fit; otherwise archive on the host for capacity shards. */
  float *flat_dev;
  float *flat_host;       /* used when the complete archive does not fit */
  size_t flat_count;
  size_t *mck_off;
  int *mck_base;

  /* Halo archive of the received direction-0 blocks, with the same policy. */
  float *halo_dev;
  float *halo_host;
  size_t halo_count;
  size_t *halo_off;
  int *halo_base;

  /* one-slot host staging for uploads into the two archives */
  float *bounce;
  size_t bounce_count;

  /* per-species group energies and the group lengths */
  double *ene;
  int *l2p1;
} SetProCapacityGpuContext;

static SetProCapacityGpuContext SetProCapacity_gpu = { 0 };
static int SetProCapacity_worker_rank = 0;

static void SetProCapacity_gpu_abort(const char *message)
{
  fprintf(stderr,"%s\n",message);
  fflush(stderr);
  if (openmx_gpu_worker_device()>=0) exit(EXIT_FAILURE);
  MPI_Abort(mpi_comm_level1,1);
}

static void *SetProCapacity_checked_malloc(size_t size)
{
  void *ptr;

  if (size==0) size = 1;
  ptr = malloc(size);
  if (ptr==NULL){
    SetProCapacity_gpu_abort("Set_ProExpn_VNA: malloc failed for the GPU batch.");
  }
  return ptr;
}

static int SetProCapacity_collective_env_flag(const char *name, int default_value)
{
  int myid,value = default_value;
  const char *env;

  MPI_Comm_rank(mpi_comm_level1,&myid);
  if (myid==0){
    env = getenv(name);
    if (env!=NULL && env[0]!='\0') value = (atoi(env)!=0);
  }
  MPI_Bcast(&value,1,MPI_INT,0,mpi_comm_level1);

  return value;
}

/* OPENMX_SETPRO_VERBOSE=1: every rank reports each device stage of this
   file (the admission figures, the batch sizes, the elapsed time) together
   with the device memory free at that moment, on stderr */
static int SetProCapacity_verbose(void)
{
  static int verbose = -1;

  if (verbose<0){
    const char *env = getenv("OPENMX_SETPRO_VERBOSE");
    verbose = (env!=NULL && env[0]!='\0' && atoi(env)!=0) ? 1 : 0;
  }
  return verbose;
}

static double SetProCapacity_free_mib(void)
{
  size_t free_bytes = 0,total_bytes = 0;

  /* a CPU run must not create a CUDA context for a diagnostic */
  if (scf_eigen_lib_flag!=GPUSOLVER) return -1.0;
  if (OpenMX_GpuMemGetInfo(&free_bytes,&total_bytes)!=hipSuccess) return -1.0;
  return (double)free_bytes/1048576.0;
}

/* kept out of line: nvc 26.9 miscompiles the inlined copies (an undefined
   value for the formatted buffer in its LLVM IR) */
static void SetProCapacity_note(const char *stage, const char *fmt, ...) __attribute__((noinline));
static void SetProCapacity_note(const char *stage, const char *fmt, ...)
{
  char detail[512];
  va_list ap;
  int myid;

  if (!SetProCapacity_verbose()) return;
  va_start(ap,fmt);
  vsnprintf(detail,sizeof(detail),fmt,ap);
  va_end(ap);
  if (openmx_gpu_worker_device()>=0) myid=SetProCapacity_worker_rank;
  else MPI_Comm_rank(mpi_comm_level1,&myid);
  {
    const double mib = SetProCapacity_free_mib();

    if (mib<0.0) fprintf(stderr,"<Set_ProExpn_VNA> rank %d %s: %s\n",myid,stage,detail);
    else fprintf(stderr,"<Set_ProExpn_VNA> rank %d %s: %s; device free %.0f MiB\n",myid,stage,detail,mib);
  }
  fflush(stderr);
}

static int SetProCapacity_HVNA_GpuBegin(Type_DS_VNA *****DS_VNA, int *VNA_List,
                                int *VNA_List2, int Num_RVNA)
{
  SetProCapacityGpuContext *g = &SetProCapacity_gpu;
  int Mc_AN,k,spe,L1,node_ranks = 1,node_rank = 0;
  size_t pos = 0,slots = 0,halo_slots = 0;
  size_t free_bytes = 0,total_bytes = 0;
  MPI_Comm node_comm = MPI_COMM_NULL;
  int requested = scf_eigen_lib_flag==GPUSOLVER;

  memset(g,0,sizeof(*g));
  g->num_proj = (List_YOUSO[35]+1)*(List_YOUSO[35]+1)*List_YOUSO[34];
  g->num_rvna = Num_RVNA;

  MPI_Allreduce(MPI_IN_PLACE,&requested,1,MPI_INT,MPI_MIN,mpi_comm_level1);
  if (!requested || !SetPro_gpu_stage_enabled("OPENMX_SETPRO_HVNA_GPU")) return 0;

  MPI_Comm_split_type(mpi_comm_level1,MPI_COMM_TYPE_SHARED,0,MPI_INFO_NULL,&node_comm);
  MPI_Comm_size(node_comm,&node_ranks);
  MPI_Comm_rank(node_comm,&node_rank);
  if (node_ranks<1) node_ranks = 1;
  if (gpu_rank_device_usable()) openmx_target_wait();
  if (gpu_rank_device_usable() && hipDeviceSynchronize()==hipSuccess){
    openmx_target_clear_freelists();
  }
  MPI_Barrier(node_comm);

  /* sizes of the flat local table and the halo archive */

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    int Gc_AN = M2G[Mc_AN];
    int tno = Spe_Total_NO[WhatSpecies[Gc_AN]];

    slots += (size_t)(FNAN[Gc_AN]+1);
    pos += (size_t)(FNAN[Gc_AN]+1)*(size_t)tno*(size_t)g->num_proj;
  }

  for (Mc_AN=Matomnum+1; Mc_AN<=Matomnum+MatomnumF; Mc_AN++){
    int Gc_AN = F_M2G[Mc_AN];
    int tno = Spe_Total_NO[WhatSpecies[Gc_AN]];

    halo_slots += (size_t)(FNAN[Gc_AN]+1);
    g->halo_count += (size_t)(FNAN[Gc_AN]+1)*(size_t)tno*(size_t)g->num_proj;
  }

  g->flat_count = pos==0 ? 1 : pos;
  if (OpenMX_GpuMemGetInfo(&free_bytes,&total_bytes)!=hipSuccess) free_bytes = 0;
  {
    const size_t reserve = (size_t)256*1024*1024;
    size_t items = 0, kslots = 0, outputs = 0;
    unsigned long long need, max_need, available = free_bytes, min_available;
    int admitted = 0, fit, global_fit;

    /* AMD's rank-turn admission, with the actual OpenMP transient arrays.
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

  g->mck_base = (int*)SetProCapacity_checked_malloc(sizeof(int)*(size_t)(Matomnum+2));
  g->mck_off = (size_t*)SetProCapacity_checked_malloc(sizeof(size_t)*(slots==0 ? 1 : slots));
  g->halo_base = (int*)SetProCapacity_checked_malloc(sizeof(int)*(size_t)(MatomnumF+2));
  g->halo_off = (size_t*)SetProCapacity_checked_malloc(sizeof(size_t)*(halo_slots==0 ? 1 : halo_slots));
  g->ene = (double*)SetProCapacity_checked_malloc(sizeof(double)*(size_t)SpeciesNum*(size_t)Num_RVNA);
  g->l2p1 = (int*)SetProCapacity_checked_malloc(sizeof(int)*(size_t)Num_RVNA);

  /* Keep the original device-only archive when all ranks fit. Rank turns
     use host archives with the RAM budget checked above. Rows are staged
     through one bounce slot sized for the largest (atom,neighbour) block. */
  if (g->host_archive){
    g->flat_host = (float*)malloc(sizeof(float)*g->flat_count);
    g->halo_host = (float*)malloc(sizeof(float)*(g->halo_count==0 ? 1 : g->halo_count));
  }
  else{
    g->flat_dev = (float*)openmx_target_alloc(sizeof(float)*g->flat_count);
    g->halo_dev = (float*)openmx_target_alloc(sizeof(float)*(g->halo_count==0 ? 1 : g->halo_count));
  }
  {
    size_t bounce_count = 1;

    for (spe=0; spe<SpeciesNum; spe++){
      size_t cnt = (size_t)Spe_Total_NO[spe]*(size_t)g->num_proj;
      if (bounce_count<cnt) bounce_count = cnt;
    }
    g->bounce_count = bounce_count;
    g->bounce = (float*)SetProCapacity_checked_malloc(sizeof(float)*bounce_count);
  }
  {
    if (!g->host_archive && (!g->flat_dev || !g->halo_dev)) {
      if(g->flat_dev)openmx_target_free(g->flat_dev);if(g->halo_dev)openmx_target_free(g->halo_dev);
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
      if (g->flat_dev!=NULL) openmx_target_free(g->flat_dev);
      if (g->halo_dev!=NULL) openmx_target_free(g->halo_dev);
      openmx_target_clear_freelists();
      free(g->flat_host);
      free(g->halo_host);
      free(g->bounce);
      free(g->l2p1);
      free(g->ene);
      free(g->halo_off);
      free(g->halo_base);
      free(g->mck_off);
      free(g->mck_base);
      memset(g,0,sizeof(*g));
      return 0;
    }
  }

  for (L1=0; L1<Num_RVNA; L1++){
    g->l2p1[L1] = 2*VNA_List[L1] + 1;
  }
  for (spe=0; spe<SpeciesNum; spe++){
    for (L1=0; L1<Num_RVNA; L1++){
      g->ene[(size_t)spe*Num_RVNA + L1] = VNA_proj_ene[spe][VNA_List[L1]][VNA_List2[L1]];
    }
  }

  {
    pos = 0;
    slots = 0;
    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      int Gc_AN = M2G[Mc_AN];
      int tno = Spe_Total_NO[WhatSpecies[Gc_AN]];

      g->mck_base[Mc_AN] = (int)slots;
      for (k=0; k<=FNAN[Gc_AN]; k++){
        g->mck_off[slots] = pos;
        pos += (size_t)tno*(size_t)g->num_proj;
        slots++;
      }
    }

    halo_slots = 0;
    pos = 0;
    for (Mc_AN=Matomnum+1; Mc_AN<=Matomnum+MatomnumF; Mc_AN++){
      int Gc_AN = F_M2G[Mc_AN];
      int tno = Spe_Total_NO[WhatSpecies[Gc_AN]];

      g->halo_base[Mc_AN-Matomnum] = (int)halo_slots;
      for (k=0; k<=FNAN[Gc_AN]; k++){
        g->halo_off[halo_slots] = pos;
        pos += (size_t)tno*(size_t)g->num_proj;
        halo_slots++;
      }
    }

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
          openmx_target_upload(g->flat_dev + base, g->bounce,
                               sizeof(float)*(size_t)tno*(size_t)g->num_proj);
      }
    }
  }

  SetProCapacity_note("HVNA GPU archives","local archive %.0f MiB, halo archive %.0f MiB on the %s",
              (double)g->flat_count*sizeof(float)/1048576.0,
              (double)g->halo_count*sizeof(float)/1048576.0,
              g->host_archive ? "host (capacity recovery)" : "device");

  g->enabled = 1;
  return 1;
}

static void SetProCapacity_HVNA_GpuEnd(void)
{
  SetProCapacityGpuContext *g = &SetProCapacity_gpu;

  if (!g->enabled) return;

  if (g->flat_dev!=NULL) openmx_target_free(g->flat_dev);
  if (g->halo_dev!=NULL) openmx_target_free(g->halo_dev);
  openmx_target_clear_freelists();

  free(g->flat_host);
  free(g->halo_host);
  free(g->bounce);
  free(g->l2p1);
  free(g->ene);
  free(g->halo_off);
  free(g->halo_base);
  free(g->mck_off);
  free(g->mck_base);
  memset(g,0,sizeof(*g));
}

/* archive the direction-0 block of the halo atom just received */
static void SetProCapacity_HVNA_GpuArchiveHalo(Type_DS_VNA *****DS_VNA, int Original_Mc_AN,
                                       int Gc_AN)
{
  SetProCapacityGpuContext *g = &SetProCapacity_gpu;
  int tno = Spe_Total_NO[WhatSpecies[Gc_AN]];
  int halo_idx = Original_Mc_AN - Matomnum;
  int k;

  if (halo_idx<1 || MatomnumF<halo_idx){
    SetProCapacity_gpu_abort("Set_ProExpn_VNA GPU: halo index out of range.");
  }

  /* serial: the rows go through the shared bounce slot into the device
     archive, and openmx_target_upload is not called from inside an OpenMP
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
      openmx_target_upload(g->halo_dev + base, g->bounce,
                           sizeof(float)*(size_t)tno*(size_t)g->num_proj);
  }
}

/* the unified batch over every (local atom, neighbour) pair */
static int SetProCapacity_HVNA_GpuRun_Part(int first,int last,SetProCapacityGpuContext *g,double ****HVNA)
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
  SetProCapacity_note("HVNA GPU batch begin","%d (atom,neighbour) items, %d k slots, %.0f MiB of outputs",
              nitems,kslots,(double)out_count*sizeof(double)/1048576.0);

  item_atom = (int*)SetProCapacity_checked_malloc(sizeof(int)*(size_t)nitems);
  item_j = (int*)SetProCapacity_checked_malloc(sizeof(int)*(size_t)nitems);
  item_koff = (int*)SetProCapacity_checked_malloc(sizeof(int)*(size_t)nitems);
  item_kn = (int*)SetProCapacity_checked_malloc(sizeof(int)*(size_t)nitems);
  item_tnoC = (int*)SetProCapacity_checked_malloc(sizeof(int)*(size_t)nitems);
  item_tnoH = (int*)SetProCapacity_checked_malloc(sizeof(int)*(size_t)nitems);
  item_out = (size_t*)SetProCapacity_checked_malloc(sizeof(size_t)*(size_t)nitems);
  krowA = (size_t*)SetProCapacity_checked_malloc(sizeof(size_t)*(size_t)(kslots==0 ? 1 : kslots));
  krowB = (size_t*)SetProCapacity_checked_malloc(sizeof(size_t)*(size_t)(kslots==0 ? 1 : kslots));
  krow_halo = (int*)SetProCapacity_checked_malloc(sizeof(int)*(size_t)(kslots==0 ? 1 : kslots));
  krow_ene = (int*)SetProCapacity_checked_malloc(sizeof(int)*(size_t)(kslots==0 ? 1 : kslots));
  out = (double*)SetProCapacity_checked_malloc(sizeof(double)*(out_count==0 ? 1 : out_count));

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
    SetProCapacity_gpu_abort("Set_ProExpn_VNA GPU: inconsistent batch.");
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
    if(hipMalloc(&arena,bytes)!=hipSuccess) goto hvna_cleanup;
    for(int i=0;i<12;i++)dev[i]=(char*)arena+off[i];
    device_out=dev[11];
    for(int i=0;i<11;i++)if(hipMemcpy(dev[i],host[i],length[i],hipMemcpyHostToDevice)!=hipSuccess)goto hvna_cleanup;
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
#pragma omp target teams distribute parallel for thread_limit(128) firstprivate(fp32) is_device_ptr(flat,halo,item_koff,item_kn,item_tnoC,item_tnoH,item_out,krowA,krowB,krow_halo,krow_ene,ene,l2p1,out)
      for (int pp=0; pp<nitems_c; pp++){
        const int tnoC = item_tnoC[pp];
        const int tnoH = item_tnoH[pp];
        const int k_off = item_koff[pp];
        const int k_n = item_kn[pp];
        const size_t out_off = item_out[pp];
        const int mn = tnoC*tnoH;


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

  if(hipMemcpy(out,device_out,out_count*sizeof(double),hipMemcpyDeviceToHost)!=hipSuccess)goto hvna_cleanup;

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
  if(arena)hipFree(arena);
  (void)hipGetLastError();
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
    SetProCapacity_note("HVNA GPU batch end","%.2f s",t1-hvna_t0);
  }
  return ok;
}

typedef struct {int num_proj,num_rvna,fp32;} SetProCapacityHVNACapacityParameters;
static void SetProCapacity_HVNA_capacity_kernel(size_t first,size_t count,const void *parameters,void *const v[],int gpu)
{
  const SetProCapacityHVNACapacityParameters *p=parameters;
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
#pragma omp target teams distribute parallel for is_device_ptr(offset,a,b,species_offset,species,ene,length,damp,out) if(gpu)
  for(size_t row=0;row<count;row++){
    double total=0;size_t pos=offset[row];

    for(size_t k=species_offset[row];k<species_offset[row+1];k++,pos+=np){
      double sum=0;int l=0;

      for(int r=0;r<nr;r++){
        double term=0;

        for(int m=0;m<length[r];m++,l++)
          term+=fp32?(double)(a[pos+l]*b[pos+l]):(double)a[pos+l]*(double)b[pos+l];
        sum+=ene[(size_t)species[k]*nr+r]*term;
      }
      total+=sum;
    }
    out[row]=damp[row]*total;
  }
}

static void SetProCapacity_HVNA_archive_row(float *dest,const float *host,const float *device,size_t offset,int n)
{
  if(host)memcpy(dest,host+offset,(size_t)n*sizeof(float));
  else if(hipMemcpy(dest,device+offset,(size_t)n*sizeof(float),hipMemcpyDeviceToHost)!=hipSuccess)
    SetProCapacity_gpu_abort("Set_ProExpn_VNA: cannot recover an existing projector archive.");
}

static void SetProCapacity_HVNA_GpuRun(double ****HVNA)
{
  SetProCapacityGpuContext *g=&SetProCapacity_gpu;
  int done=!g->host_archive && SetProCapacity_HVNA_GpuRun_Part(1,Matomnum+1,g,HVNA);
  int mc=done?Matomnum+1:1,j=0;size_t row=0;
  SetProCapacityHVNACapacityParameters p={g->num_proj,g->num_rvna,openmx_fp32_enabled()};
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
        if(n==1)SetProCapacity_gpu_abort("Set_ProExpn_VNA: no host memory for one projector row.");
        n=(n+1)/2;
      }
      double damping=dampingF(Spe_Atom_Cut1[WhatSpecies[gc]]+Spe_Atom_Cut1[WhatSpecies[jg]],Dis[gc][j]);
      for(size_t i=0;i<=n;i++){offset[i]=i*kn*p.num_proj;species_offset[i]=i*kn;}
      for(size_t i=0;i<n;i++){
        int m=(row+i)/nh,q=(row+i)%nh;size_t kpos=0;damp[i]=damping;
        for(int k=0;k<=FNAN[gc];k++){
          int kl=RMI1[mc][j][k];if(kl<0)continue;
          size_t ao=g->mck_off[g->mck_base[mc]+k]+(size_t)m*p.num_proj;
          SetProCapacity_HVNA_archive_row(a+(i*kn+kpos)*p.num_proj,g->flat_host,g->flat_dev,ao,p.num_proj);
          size_t bo=(mj<=Matomnum?g->mck_off[g->mck_base[mj]+kl]:g->halo_off[g->halo_base[mj-Matomnum]+kl])+(size_t)q*p.num_proj;
          SetProCapacity_HVNA_archive_row(b+(i*kn+kpos)*p.num_proj,mj<=Matomnum?g->flat_host:g->halo_host,mj<=Matomnum?g->flat_dev:g->halo_dev,bo,p.num_proj);
          species[i*kn+kpos++]=WhatSpecies[natn[gc][k]];
        }
      }
    }
    OpenMXGpuTask task={0};task.count=n;task.parameters=&p;task.parameter_bytes=sizeof(p);task.inputs=6;task.outputs=1;task.kernel=SetProCapacity_HVNA_capacity_kernel;
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
}


static int SetPro_hvna_native=0;
static int SetPro_HVNA_DispatchBegin(Type_DS_VNA *****v,int *a,int *b,int n)
{
    SetPro_hvna_native=0;
    if(!openmx_gpu_capacity_test_recovery() && !openmx_gpu_workers_requested())SetPro_hvna_native=SetPro_HVNA_GpuBegin(v,a,b,n);
    if(SetPro_hvna_native)return 1;
    return SetProCapacity_HVNA_GpuBegin(v,a,b,n);
}
static void SetPro_HVNA_DispatchArchive(Type_DS_VNA *****v,int m,int g)
{
    if(SetPro_hvna_native)SetPro_HVNA_GpuArchiveHalo(v,m,g);
    else SetProCapacity_HVNA_GpuArchiveHalo(v,m,g);
}
static void SetPro_HVNA_DispatchRun(double ****h)
{
    if(SetPro_hvna_native){
        for(int turn=0;turn<SetPro_gpu.turns;turn++){
            if(turn==SetPro_gpu.turn)SetPro_HVNA_GpuRun(h);
            if(SetPro_gpu.turns>1)MPI_Barrier(mpi_comm_level1);
        }
        SetPro_HVNA_GpuEnd();
    }else{
        SetProCapacity_HVNA_GpuRun(h);
        SetProCapacity_HVNA_GpuEnd();
    }
    SetPro_hvna_native=0;
}

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

/* per-species combo/block tables of the DS_VNA construction batch */
typedef struct {
  int enabled;
  int chunk;           /* device pair batch admitted by the memory guard */
  int num_proj;
  int num_rvna;
  int lfi_max;          /* max Lmax_Four_Int over species             */
  int l0max_all;        /* max Spe_MaxL_Basis                          */
  int l1max;            /* List_YOUSO[35]                              */
  int cmb_max;          /* max #(L0,Mul0) combos per species           */
  int blk_max;          /* cmb_max*num_rvna                            */
  int mn_max;           /* max tno*num_proj                            */

  int *sp_lfi;          /* [spe] Lmax_Four_Int                         */
  int *sp_ncmb;         /* [spe] #(L0,Mul0)                            */
  int *cmb_L0;          /* [spe*cmb_max+c] L0                          */
  int *cmb_num0;        /* [spe*cmb_max+c] row offset of the combo     */
  int *vna_L1;          /* [L] VNA_List                                */
  int *vna_num1;        /* [L] column offset                           */

  double *pro00;        /* [spe][cmb][GL_Mesh]                         */
  double *pro01;
  double *vnab;         /* [spe][num_rvna][GL_Mesh]                    */
  double *gnt;          /* Gaunt table                                 */
  double *c2r;          /* Comp2Real, packed per L                     */
  int *c2r_off;         /* [L] offset into c2r (complex index)         */
} SetProGpu2Context;

static SetProGpu2Context SetPro_gpu2 = { 0 };

/* Maximum pair batch. The memory preflight may shrink it; both the byte
   estimate and the actual target mappings use the admitted g->chunk. */
#define SETPRO_DSVNA_CHUNK 256

/* Device bytes the "#pragma omp target data" region in SetPro_DSVNA_GpuRun
   will map.  Deliberately mirrors that clause term for term, in the same
   order, so the two can be diffed by eye. */
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

  /* map(to:) */
  bytes += (2*pro_len + vnab_len + gnt_len + c2r_len)*sizeof(double);
  bytes += (size_t)GL_Mesh*sizeof(double);                       /* GL_NormK   */
  bytes += (size_t)(lc+2)*sizeof(int);                           /* c2r_off    */
  bytes += (size_t)(2*SpeciesNum
                    + 2*(size_t)SpeciesNum*g->cmb_max
                    + 2*(size_t)g->num_rvna)*sizeof(int);

  /* map(alloc:) */
  bytes += (2*sphb_len + 2*sum_len + tmpnl_len)*sizeof(double);
  bytes += out_len*sizeof(float);
  bytes += CH*5*sizeof(double) + CH*4*sizeof(int);               /* pr_*       */
  bytes += CH*(size_t)nsh*6*sizeof(double);                      /* sh_tab     */

  return bytes;
}

static int SetPro_DSVNA_GpuBegin(int *VNA_List, int *VNA_List2, int Num_RVNA,
                                 double ****Bessel_Pro00, double ****Bessel_Pro01)
{
  SetProGpu2Context *g = &SetPro_gpu2;
  int spe,L0,Mul0,L,i,k,c;
  size_t free_bytes = 0,total_bytes = 0;
  int node_ranks = 1,node_rank = 0;
  MPI_Comm node_comm = MPI_COMM_NULL;

  memset(g,0,sizeof(*g));

  int requested=scf_eigen_lib_flag==GPUSOLVER;
  MPI_Allreduce(MPI_IN_PLACE,&requested,1,MPI_INT,MPI_MIN,mpi_comm_level1);
  if (!requested) return 0;
  if (!SetPro_gpu_stage_enabled("OPENMX_SETPRO_DSVNA_GPU")) return 0;

  MPI_Comm_split_type(mpi_comm_level1,MPI_COMM_TYPE_SHARED,0,MPI_INFO_NULL,&node_comm);
  MPI_Comm_size(node_comm,&node_ranks);
  MPI_Comm_rank(node_comm,&node_rank);
  if (node_ranks<1) node_ranks = 1;
  MPI_Comm_free(&node_comm);

  g->num_proj = (List_YOUSO[35]+1)*(List_YOUSO[35]+1)*List_YOUSO[34];
  g->num_rvna = Num_RVNA;
  g->l1max = List_YOUSO[35];

  g->sp_lfi = (int*)SetPro_checked_malloc(sizeof(int)*SpeciesNum);
  g->sp_ncmb = (int*)SetPro_checked_malloc(sizeof(int)*SpeciesNum);

  g->l0max_all = 0;
  g->cmb_max = 0;
  for (spe=0; spe<SpeciesNum; spe++){
    int Lmax = 0,ncmb = 0;

    for (L=0; L<Num_RVNA; L++){
      if (Lmax<VNA_List[L]) Lmax = VNA_List[L];
    }
    if (Spe_MaxL_Basis[spe]<Lmax) g->sp_lfi[spe] = 2*Lmax;
    else                          g->sp_lfi[spe] = 2*Spe_MaxL_Basis[spe];
    if (g->lfi_max<g->sp_lfi[spe]) g->lfi_max = g->sp_lfi[spe];
    if (g->l0max_all<Spe_MaxL_Basis[spe]) g->l0max_all = Spe_MaxL_Basis[spe];

    for (L0=0; L0<=Spe_MaxL_Basis[spe]; L0++){
      ncmb += Spe_Num_Basis[spe][L0];
    }
    g->sp_ncmb[spe] = ncmb;
    if (g->cmb_max<ncmb) g->cmb_max = ncmb;
  }
  g->blk_max = g->cmb_max*Num_RVNA;

  g->cmb_L0 = (int*)SetPro_checked_malloc(sizeof(int)*(size_t)SpeciesNum*g->cmb_max);
  g->cmb_num0 = (int*)SetPro_checked_malloc(sizeof(int)*(size_t)SpeciesNum*g->cmb_max);
  g->vna_L1 = (int*)SetPro_checked_malloc(sizeof(int)*Num_RVNA);
  g->vna_num1 = (int*)SetPro_checked_malloc(sizeof(int)*Num_RVNA);
  g->pro00 = (double*)SetPro_checked_malloc(sizeof(double)*(size_t)SpeciesNum*g->cmb_max*GL_Mesh);
  g->pro01 = (double*)SetPro_checked_malloc(sizeof(double)*(size_t)SpeciesNum*g->cmb_max*GL_Mesh);
  g->vnab = (double*)SetPro_checked_malloc(sizeof(double)*(size_t)SpeciesNum*Num_RVNA*GL_Mesh);

  {
    int mn_max = 0;

    for (spe=0; spe<SpeciesNum; spe++){
      int num0 = 0;

      c = 0;
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

  for (spe=0; spe<SpeciesNum; spe++){
    c = 0;
    for (L0=0; L0<=Spe_MaxL_Basis[spe]; L0++){
      for (Mul0=0; Mul0<Spe_Num_Basis[spe][L0]; Mul0++){
        for (i=0; i<GL_Mesh; i++){
          g->pro00[((size_t)spe*g->cmb_max+c)*GL_Mesh+i] = Bessel_Pro00[spe][L0][Mul0][i];
          g->pro01[((size_t)spe*g->cmb_max+c)*GL_Mesh+i] = Bessel_Pro01[spe][L0][Mul0][i];
        }
        c++;
      }
    }
    for (L=0; L<Num_RVNA; L++){
      for (i=0; i<GL_Mesh; i++){
        g->vnab[((size_t)spe*g->num_rvna+L)*GL_Mesh+i]
          = Spe_VNA_Bessel[spe][VNA_List[L]][VNA_List2[L]][i];
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
                if (M1==M0-m && __builtin_abs(L1-LL)<=L0 && L0<=(L1+LL)){
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

  /* packed Comp2Real up to max(l0max, l1max) */

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
          g->c2r[2*idx  ] = Comp2Real[Lc][M][k].r;
          g->c2r[2*idx+1] = Comp2Real[Lc][M][k].i;
        }
      }
    }
  }

  /* The common task allocator selects owner, local peers, remote nodes,
     then CPU from the exact batch footprint. Keep host lookup tables. */
  g->chunk = SETPRO_DSVNA_CHUNK;
  g->enabled = 1;
  return 1;
}

static void SetPro_DSVNA_GpuEnd(void)
{
  SetProGpu2Context *g = &SetPro_gpu2;

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
 double *pro00=v[0];
 double *pro01=v[1];
 double *vnab=v[2];
 double *gnt=v[3];
 double *c2r=v[4];
 int *c2r_off=v[5];
 int *sp_lfi=v[6];
 int *sp_ncmb=v[7];
 int *cmb_L0=v[8];
 int *cmb_num0=v[9];
 int *vna_L1=v[10];
 int *vna_num1=v[11];
 double *GL_NormK=v[12];
 double *pr_r=v[13];
 double *pr_siT=v[14];
 double *pr_coT=v[15];
 double *pr_siP=v[16];
 double *pr_coP=v[17];
 int *pr_spe=v[18];
 int *pr_hwan=v[19];
 int *pr_h0=v[20];
 int *pr_tno=v[21];
 double *sh_tab=v[22];
 float *outbuf=v[23];
 double *sphb=v[24];
 double *sphbp=v[25];
 double *sum0=v[26];
 double *sumr0=v[27];
 double *tmpnl=v[28];

#pragma omp target teams distribute parallel for is_device_ptr(outbuf) if(gpu)
 for(size_t i=0;i<(size_t)np*4*mn_max;i++)outbuf[i]=0;
      /* K0: spherical Bessel tables of every (pair, GL point) */

#pragma omp target teams distribute parallel for thread_limit(128) is_device_ptr(pr_r,pr_spe,sp_lfi,GL_NormK,sphb,sphbp) if(gpu)
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

#pragma omp target teams distribute parallel for thread_limit(128) is_device_ptr(pr_spe,pr_hwan,pr_h0,sp_lfi,sp_ncmb,pro00,pro01,vnab,sphb,sphbp,sum0,sumr0) if(gpu)
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

#pragma omp target teams distribute parallel for thread_limit(128) is_device_ptr(pr_spe,sp_lfi,sp_ncmb,cmb_L0,vna_L1,gnt,sh_tab,sum0,sumr0,tmpnl) if(gpu)
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

            if ( __builtin_abs(L1-LL)<=L0 && L0<=(L1+LL) ){

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

#pragma omp target teams distribute parallel for thread_limit(128) is_device_ptr(pr_spe,pr_h0,pr_r,pr_siT,pr_coT,pr_siP,pr_coP,pr_tno,sp_ncmb,cmb_L0,cmb_num0,vna_L1,vna_num1,c2r,c2r_off,tmpnl,outbuf) if(gpu)
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

static void SetPro_DSVNA_GpuRun_Part(Type_DS_VNA *****DS_VNA, int OneD_Nloop, int *OneD2Mc_AN, int *OneD2h_AN, SetProGpu2Context *g)
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
      openmx_gpu_task_execute(mpi_comm_level1,&task,1,"SetPro_DSVNA");


      /* host: scatter the chunk results into DS_VNA */

#pragma omp parallel for if(!omp_in_parallel()) schedule(dynamic)
      for (pp=0; pp<np; pp++){
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
static void SetPro_DSVNA_GpuRun(Type_DS_VNA *****matrix,int count,int *mc,int *h)
{SetPro_DSVNA_GpuRun_Part(matrix,count,mc,h,&SetPro_gpu2);}


/* ------------------------------------------------------------------ */
/* Batched device construction of HVNA2 / HVNA3 (the one-centre PAO
   pairs against the crude VNA potential).  Set_VNA2 and Set_VNA3 are
   mirror images: the PAO-product species and the direction sign swap.
   The same host-tabulated Gaunt/Y_lm treatment as the DS_VNA batch is
   used; the upper->lower conjugate copy and both unitary transforms
   keep the exact CPU loop order.                                      */
/* ------------------------------------------------------------------ */

typedef struct {
  int enabled;
  int chunk;            /* admitted pair batch, shared by sizing and execution */
  int l0max;            /* max Spe_MaxL_Basis                          */
  int lfi_max;          /* 2*l0max                                     */
  int cmb_max;
  int tno_max;
  int mn_max;           /* tno_max^2                                   */

  int *sp_lfi;          /* [spe] 2*Spe_MaxL_Basis                      */
  int *sp_ncmb;
  int *sp_tno;
  int *cmb_L0;          /* [spe*cmb_max+c]                             */
  int *cmb_num0;

  double *crude;        /* [spe][GL_Mesh]                              */
  double *prod;         /* [spe][c0][c1][LL][GL_Mesh] (c0<=c1 pairs)   */
  double *gnt2;         /* (-1)^|m| Gaunt(L0,M0,L1,M1,LL,-m), m=M1-M0  */
  double *c2r;
  int *c2r_off;
} SetProGpu3Context;

static SetProGpu3Context SetPro_gpu3 = { 0 };

/* full pair-chunk length of the HVNA2/HVNA3 device pipeline; the admitted
   length g->chunk (halved until the footprint fits the rank's budget) is
   shared by the estimate below and by SetPro_VNA23_GpuRun */
#define SETPRO_VNA23_CHUNK 512

/* Device bytes the "#pragma omp target data" region of SetPro_VNA23_GpuRun
   maps, term for term in the order of that clause (valid once the
   per-species tables of SetPro_VNA23_GpuBegin are built).  A failed map
   inside that region aborts every rank, so the estimate must be exact. */
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

  /* map(to:) */
  bytes += (crude_len + prod_len + gnt2_len + c2r_len)*sizeof(double);
  bytes += (size_t)(g->l0max+2)*sizeof(int);                                 /* c2r_off   */
  bytes += (size_t)(2*SpeciesNum + 2*(size_t)SpeciesNum*g->cmb_max)*sizeof(int);
  bytes += (size_t)2*GL_Mesh*sizeof(double);                                 /* GL_NormK, GL_Weight */

  /* map(alloc:) */
  bytes += (2*sphb_len + 2*sum_len + tmpnl_len + out_len)*sizeof(double);
  bytes += CH*5*sizeof(double) + CH*3*sizeof(int);                           /* pr_*      */
  bytes += CH*(size_t)nsh*6*sizeof(double);                                  /* sh_tab    */

  return bytes;
}

static int SetPro_VNA23_GpuBegin(void)
{
  SetProGpu3Context *g = &SetPro_gpu3;
  int spe,L0,Mul0,L1,Mul1,LL,i,c0,c1,k;
  size_t free_bytes = 0,total_bytes = 0;
  int node_ranks = 1,node_rank = 0;
  MPI_Comm node_comm = MPI_COMM_NULL;

  memset(g,0,sizeof(*g));
  g->chunk = SETPRO_VNA23_CHUNK;

  int requested=scf_eigen_lib_flag==GPUSOLVER;
  MPI_Allreduce(MPI_IN_PLACE,&requested,1,MPI_INT,MPI_MIN,mpi_comm_level1);
  if (!requested) return 0;
  if (!SetPro_gpu_stage_enabled("OPENMX_SETPRO_VNA23_GPU")) return 0;

  MPI_Comm_split_type(mpi_comm_level1,MPI_COMM_TYPE_SHARED,0,MPI_INFO_NULL,&node_comm);
  MPI_Comm_size(node_comm,&node_ranks);
  MPI_Comm_rank(node_comm,&node_rank);
  if (node_ranks<1) node_ranks = 1;
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
      g->crude[(size_t)spe*GL_Mesh+i] = Spe_CrudeVNA_Bessel[spe][i];
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
                      = Spe_ProductRF_Bessel[spe][L0][Mul0][L1][Mul1][LL][i];
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
                g->gnt2[idx] = pow(-1.0,(double)__builtin_abs(m))*Gaunt(L0v,M0,L1v,M1,LL,-m);
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
          g->c2r[2*idx  ] = Comp2Real[Lc][M][k].r;
          g->c2r[2*idx+1] = Comp2Real[Lc][M][k].i;
        }
      }
    }
  }

  /* The common task allocator selects owner, local peers, remote nodes,
     then CPU from the exact batch footprint. Keep host lookup tables. */
  g->chunk = SETPRO_VNA23_CHUNK;
  g->enabled = 1;
  return 1;
}

static void SetPro_VNA23_GpuEnd(void)
{
  SetProGpu3Context *g = &SetPro_gpu3;

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
 double *crude=v[0];
 double *prod=v[1];
 double *gnt2=v[2];
 double *c2r=v[3];
 int *c2r_off=v[4];
 int *sp_lfi=v[5];
 int *sp_ncmb=v[6];
 int *cmb_L0=v[7];
 int *cmb_num0=v[8];
 double *GL_NormK=v[9];
 double *GL_Weight=v[10];
 double *pr_r=v[11];
 double *pr_siT=v[12];
 double *pr_coT=v[13];
 double *pr_siP=v[14];
 double *pr_coP=v[15];
 int *pr_spe=v[16];
 int *pr_vspe=v[17];
 int *pr_h0=v[18];
 double *sh_tab=v[19];
 double *outbuf=v[20];
 double *sphb=v[21];
 double *sphbp=v[22];
 double *sum0=v[23];
 double *sumr0=v[24];
 double *tmpnl=v[25];

#pragma omp target teams distribute parallel for is_device_ptr(outbuf) if(gpu)
 for(size_t i=0;i<(size_t)np*4*mn_max;i++)outbuf[i]=0;
      /* K0: spherical Bessel tables */

#pragma omp target teams distribute parallel for thread_limit(128) is_device_ptr(pr_r,pr_spe,sp_lfi,GL_NormK,sphb,sphbp) if(gpu)
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

#pragma omp target teams distribute parallel for thread_limit(128) is_device_ptr(pr_spe,pr_vspe,sp_lfi,sp_ncmb,cmb_L0,crude,prod,GL_NormK,GL_Weight,sphb,sphbp,sum0,sumr0) if(gpu)
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
            __builtin_abs(L1-LLq)<=L0 && L0<=(L1+LLq)){

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

#pragma omp target teams distribute parallel for thread_limit(128) is_device_ptr(pr_spe,sp_lfi,sp_ncmb,cmb_L0,gnt2,sh_tab,sum0,sumr0,tmpnl) if(gpu)
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

            if (__builtin_abs(L1-LL)<=L0 && L0<=(L1+LL) && -LL<=m && m<=LL){

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

#pragma omp target teams distribute parallel for thread_limit(128) is_device_ptr(pr_spe,sp_ncmb,cmb_L0,tmpnl) if(gpu)
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

#pragma omp target teams distribute parallel for thread_limit(128) is_device_ptr(pr_spe,pr_h0,pr_r,pr_siT,pr_coT,pr_siP,pr_coP,sp_ncmb,cmb_L0,cmb_num0,c2r,c2r_off,tmpnl,outbuf) if(gpu)
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

static void SetPro_VNA23_GpuRun_Part(double *****HVNA23, int mirror, int OneD_Nloop, int *OneD2Mc_AN, int *OneD2h_AN, SetProGpu3Context *g)
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
  double vna23_t0, vna23_t1;

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
      task.view[10]=(OpenMXGpuTaskView){GL_Weight,(GL_Mesh)*sizeof(double),0};
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
      openmx_gpu_task_execute(mpi_comm_level1,&task,1,"SetPro_VNA23");


      if (np > 0) SetPro_gpu_used = 1;

      /* host: scatter into HVNA2 or HVNA3 */

#pragma omp parallel for if(!omp_in_parallel()) schedule(dynamic)
      for (pp=0; pp<np; pp++){
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

    } /* chunk_start */

    dtime(&vna23_t1);
    SetPro_note("HVNA2/3 GPU batch end","HVNA%d: %.2f s",mirror ? 3 : 2,vna23_t1-vna23_t0);
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
static void SetPro_VNA23_GpuRun(double *****matrix,int mirror,int count,int *mc,int *h)
{SetPro_VNA23_GpuRun_Part(matrix,mirror,count,mc,h,&SetPro_gpu3);}


double Set_ProExpn(double ****HVNA, Type_DS_VNA *****DS_VNA)
{
  const int fp32 = openmx_fp32_enabled();
  /****************************************************
   evaluate matrix elements of neutral atom potential
   in the KB or Blochl separable form in momentum space.
  ****************************************************/
  static int firsttime=1;
  int L2,L3,L,i,j,Mc_AN,Gc_AN,h_AN,Cwan,i1,j1;
  int kk,Ls,n,j0,jg0,Mj_AN0,m,L1,GL,Mul1;
  int tno0,tno1,tno2,p,q,po1,po,Original_Mc_AN;
  int L0,Mul0,spe,Gh_AN,Hwan,fan,jg,k,kg,wakg,kl;
  int size_SumNL0,size_TmpNL;
  int Mc_AN2,Gc_AN2,num,size1,size2;
  int *Snd_DS_VNA_Size,*Rcv_DS_VNA_Size;  
  double kmin,kmax,Sk,Dk,Normk,dmp;
  Type_DS_VNA *tmp_array;
  Type_DS_VNA *tmp_array2;
  double tmp0,tmp3,tmp4,tmp5,tmp10;
  double ****Bessel_Pro00;
  double ****Bessel_Pro01;
  double ene,sum,h,coe0,sj,sy,sjp,syp;
  double rcutA,rcutB,rcut;
  double TStime,TEtime;
  double stime,etime;
  double Stime_atom,Etime_atom;
  int numprocs,myid,tag=999,ID,IDS,IDR;
  int *VNA_List;
  int *VNA_List2;
  int Num_RVNA;
  double **NLH;
  double time0,time1,time2,time3,time4,time5;
  dcomplex sum0,sum1,sum2; 

  MPI_Status stat;
  MPI_Request request;

  int OneD_Nloop,*OneD2Mc_AN,*OneD2h_AN;
  int setpro_gpu = 0;
  int setpro_gpu2 = 0;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  dtime(&TStime);

  /****************************************************
                 allocation of arrays:
  ****************************************************/

  Num_RVNA = List_YOUSO[34]*(List_YOUSO[35] + 1);

  Snd_DS_VNA_Size = (int*)malloc(sizeof(int)*numprocs);
  Rcv_DS_VNA_Size = (int*)malloc(sizeof(int)*numprocs);

  VNA_List  = (int*)malloc(sizeof(int)*(List_YOUSO[34]*(List_YOUSO[35] + 1)+2) ); 
  VNA_List2 = (int*)malloc(sizeof(int)*(List_YOUSO[34]*(List_YOUSO[35] + 1)+2) ); 

  Bessel_Pro00 = (double****)malloc(sizeof(double***)*SpeciesNum); 
  for (spe=0; spe<SpeciesNum; spe++){
    Bessel_Pro00[spe] = (double***)malloc(sizeof(double**)*(Spe_MaxL_Basis[spe]+1)); 
    for (L0=0; L0<=Spe_MaxL_Basis[spe]; L0++){
      Bessel_Pro00[spe][L0] = (double**)malloc(sizeof(double*)*Spe_Num_Basis[spe][L0]); 
      for (Mul0=0; Mul0<Spe_Num_Basis[spe][L0]; Mul0++){
	Bessel_Pro00[spe][L0][Mul0] = (double*)malloc(sizeof(double)*GL_Mesh); 
      }
    }
  }

  Bessel_Pro01 = (double****)malloc(sizeof(double***)*SpeciesNum); 
  for (spe=0; spe<SpeciesNum; spe++){
    Bessel_Pro01[spe] = (double***)malloc(sizeof(double**)*(Spe_MaxL_Basis[spe]+1)); 
    for (L0=0; L0<=Spe_MaxL_Basis[spe]; L0++){
      Bessel_Pro01[spe][L0] = (double**)malloc(sizeof(double*)*Spe_Num_Basis[spe][L0]); 
      for (Mul0=0; Mul0<Spe_Num_Basis[spe][L0]; Mul0++){
        Bessel_Pro01[spe][L0][Mul0] = (double*)malloc(sizeof(double)*GL_Mesh); 
      }
    }
  }

  size_TmpNL = (List_YOUSO[25]+1)*List_YOUSO[24]*
               (2*(List_YOUSO[25]+1)+1)*Num_RVNA*(2*List_YOUSO[30]+1);
  size_SumNL0 = (List_YOUSO[25]+1)*List_YOUSO[24]*(Num_RVNA+1);

  /* PrintMemory */
  if (firsttime) {
    PrintMemory("Set_ProExpn_VNA: SumNL0",sizeof(double)*size_SumNL0,NULL);
    PrintMemory("Set_ProExpn_VNA: SumNLr0",sizeof(double)*size_SumNL0,NULL);
    PrintMemory("Set_ProExpn_VNA: TmpNL", sizeof(dcomplex)*size_TmpNL,NULL);
    PrintMemory("Set_ProExpn_VNA: TmpNLr",sizeof(dcomplex)*size_TmpNL,NULL);
    PrintMemory("Set_ProExpn_VNA: TmpNLt",sizeof(dcomplex)*size_TmpNL,NULL);
    PrintMemory("Set_ProExpn_VNA: TmpNLp",sizeof(dcomplex)*size_TmpNL,NULL);
    firsttime=0;
  }

  if (measure_time){
    time1 = 0.0;
    time2 = 0.0;
    time3 = 0.0;
    time4 = 0.0;
    time5 = 0.0;
  }

  /************************************************************
   Spe_Num_RVPS[Hwan]     ->  Num_RVNA = List_YOUSO[34]*(List_YOUSO[35]+1) 
   Spe_VPS_List[Hwan][L]  ->  VNA_List[L] = 0,0,0,.., 1,1,1,.., 2,2,2,..,
   List_YOUSO[19]         ->  Num_RVNA
   List_YOUSO[30]         ->  List_YOUSO[35]
  *************************************************************/

  L = 0;
  for (i=0; i<=List_YOUSO[35]; i++){    /* max L */
    for (j=0; j<List_YOUSO[34]; j++){   /* # of radial projectors */
      VNA_List[L]  = i;
      VNA_List2[L] = j;
      L++;
    }
  }

  /************************************************************
   pre-calculate the time consuming part in the 'time1' part
  *************************************************************/

  kmin = Radial_kmin;
  kmax = PAO_Nkmax;
  Sk = kmax + kmin;
  Dk = kmax - kmin;

  for (spe=0; spe<SpeciesNum; spe++){

    for (L0=0; L0<=Spe_MaxL_Basis[spe]; L0++){
      for (Mul0=0; Mul0<Spe_Num_Basis[spe][L0]; Mul0++){

#pragma omp parallel shared(Mul0,L0,spe,Bessel_Pro01,Bessel_Pro00,GL_Weight,Dk,GL_NormK) private(i,Normk,tmp10)
	{         

	  int OMPID,Nthrds,Nprocs;

	  /* get info. on OpenMP */ 

	  OMPID = omp_get_thread_num();
	  Nthrds = omp_get_num_threads();
	  Nprocs = omp_get_num_procs();

	  for (i=OMPID*GL_Mesh/Nthrds; i<(OMPID+1)*GL_Mesh/Nthrds; i++){

	    Normk = GL_NormK[i];
	    tmp10 = 0.50*Dk*GL_Weight[i]*Normk*Normk;

	    Bessel_Pro00[spe][L0][Mul0][i] = RF_BesselF(spe,L0,Mul0,Normk)*tmp10;
	    Bessel_Pro01[spe][L0][Mul0][i] = Bessel_Pro00[spe][L0][Mul0][i]*Normk;
	  }

#pragma omp flush(Bessel_Pro00,Bessel_Pro01)

	} /* #pragma omp parallel */

      }
    }
  }

  /************************************************************
    start the main calculation
  *************************************************************/

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

  /* batched device construction of DS_VNA */

  setpro_gpu2 = SetPro_DSVNA_GpuBegin(VNA_List, VNA_List2, Num_RVNA,
                                      Bessel_Pro00, Bessel_Pro01);
  if (setpro_gpu2){
    SetPro_DSVNA_GpuRun(DS_VNA, OneD_Nloop, OneD2Mc_AN, OneD2h_AN);
    SetPro_DSVNA_GpuEnd();
  }

  if (!setpro_gpu2){

#pragma omp parallel shared(time_per_atom,DS_VNA,Comp2Real,Spe_VNA_Bessel,Bessel_Pro01,Bessel_Pro00,GL_NormK,List_YOUSO,VNA_List,Num_RVNA,Spe_Num_Basis,Spe_MaxL_Basis,PAO_Nkmax,atv,Gxyz,ncn,natn,Spe_Atom_Cut1,WhatSpecies,M2G,OneD2h_AN,OneD2Mc_AN,OneD_Nloop,time1,time2,time3) 

  {
    int i,j,k,l,m,num0,num1;
    int Mc_AN,h_AN,Gc_AN,Cwan,Gh_AN; 
    int Rnh,Hwan,L0,Mul0,L,L1,M0,M1,LL,Mul1,Ls;
    int OMPID,Nthrds,Nprocs,Nloop;
    int Lmax,Lmax_Four_Int;
    double Stime_atom,Etime_atom;
    double rcutA,rcutB,rcut;
    double dx,dy,dz;
    double kmin,kmax,Sk,Dk;
    double gant,SH[2],dSHt[2],dSHp[2];
    double S_coordinate[3];
    double r,theta,phi;
    double Normk,tmp0,tmp1,tmp2;
    double siT,coT,siP,coP;
    double ***SumNL0;
    double ***SumNLr0;
    double *Bes00;
    double *Bes01;
    double tmp_SphB[30];
    double tmp_SphBp[30];
    double SphB[30][GL_Mesh];
    double SphBp[30][GL_Mesh];
    double stime,etime;

    dcomplex Ctmp1,Ctmp0,Ctmp2;
    dcomplex CsumNL0,CsumNLr,CsumNLt,CsumNLp;
    dcomplex CY,CYt,CYp,CY1,CYt1,CYp1,Cpow;
    dcomplex *****TmpNL;
    dcomplex *****TmpNLr;
    dcomplex *****TmpNLt;
    dcomplex *****TmpNLp;
    dcomplex **CmatNL0;
    dcomplex **CmatNLr;
    dcomplex **CmatNLt;
    dcomplex **CmatNLp;

    /* allocation of arrays */

    TmpNL = (dcomplex*****)malloc(sizeof(dcomplex****)*(List_YOUSO[25]+1));
    for (i=0; i<(List_YOUSO[25]+1); i++){
      TmpNL[i] = (dcomplex****)malloc(sizeof(dcomplex***)*List_YOUSO[24]);
      for (j=0; j<List_YOUSO[24]; j++){
	TmpNL[i][j] = (dcomplex***)malloc(sizeof(dcomplex**)*(2*(List_YOUSO[25]+1)+1));
	for (k=0; k<(2*(List_YOUSO[25]+1)+1); k++){
	  TmpNL[i][j][k] = (dcomplex**)malloc(sizeof(dcomplex*)*Num_RVNA);
	  for (l=0; l<Num_RVNA; l++){
	    TmpNL[i][j][k][l] = (dcomplex*)malloc(sizeof(dcomplex)*(2*List_YOUSO[35]+1));
	  }
	}
      }
    }

    TmpNLr = (dcomplex*****)malloc(sizeof(dcomplex****)*(List_YOUSO[25]+1));
    for (i=0; i<(List_YOUSO[25]+1); i++){
      TmpNLr[i] = (dcomplex****)malloc(sizeof(dcomplex***)*List_YOUSO[24]);
      for (j=0; j<List_YOUSO[24]; j++){
	TmpNLr[i][j] = (dcomplex***)malloc(sizeof(dcomplex**)*(2*(List_YOUSO[25]+1)+1));
	for (k=0; k<(2*(List_YOUSO[25]+1)+1); k++){
	  TmpNLr[i][j][k] = (dcomplex**)malloc(sizeof(dcomplex*)*Num_RVNA);
	  for (l=0; l<Num_RVNA; l++){
	    TmpNLr[i][j][k][l] = (dcomplex*)malloc(sizeof(dcomplex)*(2*List_YOUSO[35]+1));
	  }
	}
      }
    }

    TmpNLt = (dcomplex*****)malloc(sizeof(dcomplex****)*(List_YOUSO[25]+1));
    for (i=0; i<(List_YOUSO[25]+1); i++){
      TmpNLt[i] = (dcomplex****)malloc(sizeof(dcomplex***)*List_YOUSO[24]);
      for (j=0; j<List_YOUSO[24]; j++){
	TmpNLt[i][j] = (dcomplex***)malloc(sizeof(dcomplex**)*(2*(List_YOUSO[25]+1)+1));
	for (k=0; k<(2*(List_YOUSO[25]+1)+1); k++){
	  TmpNLt[i][j][k] = (dcomplex**)malloc(sizeof(dcomplex*)*Num_RVNA);
	  for (l=0; l<Num_RVNA; l++){
	    TmpNLt[i][j][k][l] = (dcomplex*)malloc(sizeof(dcomplex)*(2*List_YOUSO[35]+1));
	  }
	}
      }
    }

    TmpNLp = (dcomplex*****)malloc(sizeof(dcomplex****)*(List_YOUSO[25]+1));
    for (i=0; i<(List_YOUSO[25]+1); i++){
      TmpNLp[i] = (dcomplex****)malloc(sizeof(dcomplex***)*List_YOUSO[24]);
      for (j=0; j<List_YOUSO[24]; j++){
	TmpNLp[i][j] = (dcomplex***)malloc(sizeof(dcomplex**)*(2*(List_YOUSO[25]+1)+1));
	for (k=0; k<(2*(List_YOUSO[25]+1)+1); k++){
	  TmpNLp[i][j][k] = (dcomplex**)malloc(sizeof(dcomplex*)*Num_RVNA);
	  for (l=0; l<Num_RVNA; l++){
	    TmpNLp[i][j][k][l] = (dcomplex*)malloc(sizeof(dcomplex)*(2*List_YOUSO[35]+1));
	  }
	}
      }
    }

    SumNL0 = (double***)malloc(sizeof(double**)*(List_YOUSO[25]+1));
    for (i=0; i<(List_YOUSO[25]+1); i++){
      SumNL0[i] = (double**)malloc(sizeof(double*)*List_YOUSO[24]);
      for (j=0; j<List_YOUSO[24]; j++){
	SumNL0[i][j] = (double*)malloc(sizeof(double)*Num_RVNA);
      }
    }

    SumNLr0 = (double***)malloc(sizeof(double**)*(List_YOUSO[25]+1));
    for (i=0; i<(List_YOUSO[25]+1); i++){
      SumNLr0[i] = (double**)malloc(sizeof(double*)*List_YOUSO[24]);
      for (j=0; j<List_YOUSO[24]; j++){
	SumNLr0[i][j] = (double*)malloc(sizeof(double)*Num_RVNA);
      }
    }

    CmatNL0 = (dcomplex**)malloc(sizeof(dcomplex*)*(2*(List_YOUSO[25]+1)+1));
    for (i=0; i<(2*(List_YOUSO[25]+1)+1); i++){
      CmatNL0[i] = (dcomplex*)malloc(sizeof(dcomplex)*(2*List_YOUSO[35]+1));
    }

    CmatNLr = (dcomplex**)malloc(sizeof(dcomplex*)*(2*(List_YOUSO[25]+1)+1));
    for (i=0; i<(2*(List_YOUSO[25]+1)+1); i++){
      CmatNLr[i] = (dcomplex*)malloc(sizeof(dcomplex)*(2*List_YOUSO[35]+1));
    }

    CmatNLt = (dcomplex**)malloc(sizeof(dcomplex*)*(2*(List_YOUSO[25]+1)+1));
    for (i=0; i<(2*(List_YOUSO[25]+1)+1); i++){
      CmatNLt[i] = (dcomplex*)malloc(sizeof(dcomplex)*(2*List_YOUSO[35]+1));
    }

    CmatNLp = (dcomplex**)malloc(sizeof(dcomplex*)*(2*(List_YOUSO[25]+1)+1));
    for (i=0; i<(2*(List_YOUSO[25]+1)+1); i++){
      CmatNLp[i] = (dcomplex*)malloc(sizeof(dcomplex)*(2*List_YOUSO[35]+1));
    }

    Bes00 = (double*)malloc(sizeof(double)*GL_Mesh); 
    Bes01 = (double*)malloc(sizeof(double)*GL_Mesh); 

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
      rcutA = Spe_Atom_Cut1[Cwan];

      /* set data on h_AN */

      Gh_AN = natn[Gc_AN][h_AN];        
      Rnh = ncn[Gc_AN][h_AN];
      Hwan = WhatSpecies[Gh_AN];
      rcutB = Spe_Atom_Cut1[Hwan];
      rcut = rcutA + rcutB;

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
         evaluate ovelap integrals <chi0|P> between PAOs
         and progectors of nonlocal VNA potentials. 
      ****************************************************/
      /****************************************************
              \int RL(k)*RL'(k)*jl(k*R) k^2 dk^3 
      ****************************************************/

      kmin = Radial_kmin;
      kmax = PAO_Nkmax;
      Sk = kmax + kmin;
      Dk = kmax - kmin;

      for (L0=0; L0<=Spe_MaxL_Basis[Cwan]; L0++){
        for (Mul0=0; Mul0<Spe_Num_Basis[Cwan][L0]; Mul0++){
          for (L=0; L<Num_RVNA; L++){
            L1 = VNA_List[L];
            for (M0=-L0; M0<=L0; M0++){
	      for (M1=-L1; M1<=L1; M1++){
		TmpNL[L0][Mul0][L0+M0][L][L1+M1] = Complex(0.0,0.0); 
		TmpNLr[L0][Mul0][L0+M0][L][L1+M1] = Complex(0.0,0.0); 
		TmpNLt[L0][Mul0][L0+M0][L][L1+M1] = Complex(0.0,0.0); 
		TmpNLp[L0][Mul0][L0+M0][L][L1+M1] = Complex(0.0,0.0); 
	      }
	    }
	  } 
	} 
      }

      Lmax = -10;
      for (L=0; L<Num_RVNA; L++){
	if (Lmax<VNA_List[L]) Lmax = VNA_List[L];
      }
      if (Spe_MaxL_Basis[Cwan]<Lmax)
	Lmax_Four_Int = 2*Lmax;
      else 
        Lmax_Four_Int = 2*Spe_MaxL_Basis[Cwan];

      /* calculate SphB and SphBp */
#ifdef kcomp
#else 
#pragma forceinline recursive
#endif      
      for (i=0; i<GL_Mesh; i++){
        Normk = GL_NormK[i];
        Spherical_Bessel2(Normk*r,Lmax_Four_Int,tmp_SphB,tmp_SphBp);
        for(LL=0; LL<=Lmax_Four_Int; LL++){ 
          SphB[LL][i]  = tmp_SphB[LL]; 
          SphBp[LL][i] = tmp_SphBp[LL]; 
	}
      }

      /* LL loop */
      
      for(LL=0; LL<=Lmax_Four_Int; LL++){ 

	for (L0=0; L0<=Spe_MaxL_Basis[Cwan]; L0++){
	  for (Mul0=0; Mul0<Spe_Num_Basis[Cwan][L0]; Mul0++){
	    for (L=0; L<Num_RVNA; L++){
	      SumNL0[L0][Mul0][L] = 0.0;
	      SumNLr0[L0][Mul0][L] = 0.0;
	    }
	  }
	}

        /* Gauss-Legendre quadrature */

        if (measure_time) dtime(&stime);

        for (L0=0; L0<=Spe_MaxL_Basis[Cwan]; L0++){
	  for (Mul0=0; Mul0<Spe_Num_Basis[Cwan][L0]; Mul0++){

            for (i=0; i<GL_Mesh; i++){
              Bes00[i] = Bessel_Pro00[Cwan][L0][Mul0][i]*SphB[LL][i]; 
              Bes01[i] = Bessel_Pro01[Cwan][L0][Mul0][i]*SphBp[LL][i]; 
            }

            L = 0;
            for (L1=0; L1<=List_YOUSO[35]; L1++){ 
              for (Mul1=0; Mul1<List_YOUSO[34]; Mul1++){         

                tmp1 = 0.0;
                tmp2 = 0.0;

                for (i=0; i<GL_Mesh; i++){
                  tmp1 += Bes00[i]*Spe_VNA_Bessel[Hwan][L1][Mul1][i];
                  tmp2 += Bes01[i]*Spe_VNA_Bessel[Hwan][L1][Mul1][i];
		}

		SumNL0[L0][Mul0][L]  = tmp1;
		SumNLr0[L0][Mul0][L] = tmp2;

                L++; 
	      }
	    }
	  }
	}
        
        if (measure_time && OMPID==0){
          dtime(&etime);
          time1 += etime - stime;
	}
        
	/* derivatives of "on site" */
        
	if (h_AN==0){ 
	  for (L0=0; L0<=Spe_MaxL_Basis[Cwan]; L0++){
	    for (Mul0=0; Mul0<Spe_Num_Basis[Cwan][L0]; Mul0++){
	      for (L=0; L<Num_RVNA; L++){
		SumNLr0[L0][Mul0][L] = 0.0;
	      }
	    }
	  }
	}

	/****************************************************
         for "overlap",
	    sum_m 8*(-i)^{-L0+L1+l}*
	          C_{L0,-M0,L1,M1,l,m}*
	          \int RL(k)*RL'(k)*jl(k*R) k^2 dk^3,
	****************************************************/

        if (measure_time && OMPID==0) dtime(&stime);

	for(m=-LL; m<=LL; m++){ 

	  ComplexSH(LL,m,theta,phi,SH,dSHt,dSHp);

	  SH[1]   = -SH[1];
	  dSHt[1] = -dSHt[1];
	  dSHp[1] = -dSHp[1];

	  CY  = Complex(SH[0],SH[1]);
	  CYt = Complex(dSHt[0],dSHt[1]);
	  CYp = Complex(dSHp[0],dSHp[1]);

	  for (L0=0; L0<=Spe_MaxL_Basis[Cwan]; L0++){
	    for (Mul0=0; Mul0<Spe_Num_Basis[Cwan][L0]; Mul0++){
	      for (L=0; L<Num_RVNA; L++){

		L1 = VNA_List[L];
		Ls = -L0 + L1 + LL;

            if ( __builtin_abs(L1-LL)<=L0 && L0<=(L1+LL) ){

		  Cpow = Im_pow(-1,Ls);
		  CY1  = Cmul(Cpow,CY);
		  CYt1 = Cmul(Cpow,CYt);
		  CYp1 = Cmul(Cpow,CYp);

		  for (M0=-L0; M0<=L0; M0++){

		    M1 = M0 - m;

                    if (__builtin_abs(M1)<=L1){

		      gant = Gaunt(L0,M0,L1,M1,LL,m);
		      tmp2 = gant*SumNL0[L0][Mul0][L];

		      /* S */ 

		      TmpNL[L0][Mul0][L0+M0][L][L1+M1].r += CY1.r*tmp2;
		      TmpNL[L0][Mul0][L0+M0][L][L1+M1].i += CY1.i*tmp2;

		      /* dS/dr */ 

		      tmp0 = gant*SumNLr0[L0][Mul0][L];

		      TmpNLr[L0][Mul0][L0+M0][L][L1+M1].r += CY1.r*tmp0;
		      TmpNLr[L0][Mul0][L0+M0][L][L1+M1].i += CY1.i*tmp0;

		      /* dS/dt */ 

		      TmpNLt[L0][Mul0][L0+M0][L][L1+M1].r += CYt1.r*tmp2;
		      TmpNLt[L0][Mul0][L0+M0][L][L1+M1].i += CYt1.i*tmp2;

		      /* dS/dp */ 

		      TmpNLp[L0][Mul0][L0+M0][L][L1+M1].r += CYp1.r*tmp2;
		      TmpNLp[L0][Mul0][L0+M0][L][L1+M1].i += CYp1.i*tmp2;

		    }
		  }
		}

	      }
	    }
	  }
	} /* m */

        if (measure_time && OMPID==0){
          dtime(&etime);
          time2 += etime - stime;
	}

      } /* LL */

      /****************************************************
                        Complex to Real
      ****************************************************/

      if (measure_time) dtime(&stime);

      num0 = 0;
      for (L0=0; L0<=Spe_MaxL_Basis[Cwan]; L0++){
	for (Mul0=0; Mul0<Spe_Num_Basis[Cwan][L0]; Mul0++){

	  num1 = 0;
	  for (L=0; L<Num_RVNA; L++){
	    L1 = VNA_List[L];

	    for (M0=-L0; M0<=L0; M0++){
	      for (M1=-L1; M1<=L1; M1++){

		CsumNL0 = Complex(0.0,0.0);
		CsumNLr = Complex(0.0,0.0);
		CsumNLt = Complex(0.0,0.0);
		CsumNLp = Complex(0.0,0.0);

		for (k=-L0; k<=L0; k++){

		  Ctmp1 = Conjg(Comp2Real[L0][L0+M0][L0+k]);

		  /* S */

		  Ctmp0 = TmpNL[L0][Mul0][L0+k][L][L1+M1];
		  Ctmp2 = Cmul(Ctmp1,Ctmp0);
		  CsumNL0 = Cadd(CsumNL0,Ctmp2);

		  /* dS/dr */

		  Ctmp0 = TmpNLr[L0][Mul0][L0+k][L][L1+M1];
		  Ctmp2 = Cmul(Ctmp1,Ctmp0);
		  CsumNLr = Cadd(CsumNLr,Ctmp2);

		  /* dS/dt */

		  Ctmp0 = TmpNLt[L0][Mul0][L0+k][L][L1+M1];
		  Ctmp2 = Cmul(Ctmp1,Ctmp0);
		  CsumNLt = Cadd(CsumNLt,Ctmp2);

		  /* dS/dp */

		  Ctmp0 = TmpNLp[L0][Mul0][L0+k][L][L1+M1];
		  Ctmp2 = Cmul(Ctmp1,Ctmp0);
		  CsumNLp = Cadd(CsumNLp,Ctmp2);

		}

		CmatNL0[L0+M0][L1+M1] = CsumNL0;
		CmatNLr[L0+M0][L1+M1] = CsumNLr;
		CmatNLt[L0+M0][L1+M1] = CsumNLt;
		CmatNLp[L0+M0][L1+M1] = CsumNLp;

	      }
	    }

	    for (M0=-L0; M0<=L0; M0++){
	      for (M1=-L1; M1<=L1; M1++){

		CsumNL0 = Complex(0.0,0.0);
		CsumNLr = Complex(0.0,0.0);
		CsumNLt = Complex(0.0,0.0);
		CsumNLp = Complex(0.0,0.0);

		for (k=-L1; k<=L1; k++){

		  /* S */ 

		  CsumNL0.r += (CmatNL0[L0+M0][L1+k].r*Comp2Real[L1][L1+M1][L1+k].r
				- CmatNL0[L0+M0][L1+k].i*Comp2Real[L1][L1+M1][L1+k].i);

		  CsumNL0.i += (CmatNL0[L0+M0][L1+k].r*Comp2Real[L1][L1+M1][L1+k].i
				+ CmatNL0[L0+M0][L1+k].i*Comp2Real[L1][L1+M1][L1+k].r);

		  /* dS/dr */ 

		  CsumNLr.r += (CmatNLr[L0+M0][L1+k].r*Comp2Real[L1][L1+M1][L1+k].r
				- CmatNLr[L0+M0][L1+k].i*Comp2Real[L1][L1+M1][L1+k].i);

		  CsumNLr.i += (CmatNLr[L0+M0][L1+k].r*Comp2Real[L1][L1+M1][L1+k].i
				+ CmatNLr[L0+M0][L1+k].i*Comp2Real[L1][L1+M1][L1+k].r);

		  /* dS/dt */ 

		  CsumNLt.r += (CmatNLt[L0+M0][L1+k].r*Comp2Real[L1][L1+M1][L1+k].r
				- CmatNLt[L0+M0][L1+k].i*Comp2Real[L1][L1+M1][L1+k].i);

		  CsumNLt.i += (CmatNLt[L0+M0][L1+k].r*Comp2Real[L1][L1+M1][L1+k].i
				+ CmatNLt[L0+M0][L1+k].i*Comp2Real[L1][L1+M1][L1+k].r);

		  /* dS/dp */ 

		  CsumNLp.r += (CmatNLp[L0+M0][L1+k].r*Comp2Real[L1][L1+M1][L1+k].r
				- CmatNLp[L0+M0][L1+k].i*Comp2Real[L1][L1+M1][L1+k].i);

		  CsumNLp.i += (CmatNLp[L0+M0][L1+k].r*Comp2Real[L1][L1+M1][L1+k].i
				+ CmatNLp[L0+M0][L1+k].i*Comp2Real[L1][L1+M1][L1+k].r);

		}

		DS_VNA[0][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] = 8.0*(Type_DS_VNA)CsumNL0.r;

		if (h_AN!=0){

		  if (fabs(siT)<10e-14){

		    DS_VNA[1][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] =
		      -8.0*(Type_DS_VNA)(siT*coP*CsumNLr.r + coT*coP/r*CsumNLt.r);

		    DS_VNA[2][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] =
		      -8.0*(Type_DS_VNA)(siT*siP*CsumNLr.r + coT*siP/r*CsumNLt.r);

		    DS_VNA[3][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] =
		      -8.0*(Type_DS_VNA)(coT*CsumNLr.r - siT/r*CsumNLt.r);
		  }

		  else{

		    DS_VNA[1][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] =
		      -8.0*(Type_DS_VNA)(siT*coP*CsumNLr.r + coT*coP/r*CsumNLt.r
					- siP/siT/r*CsumNLp.r);

		    DS_VNA[2][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] =
		      -8.0*(Type_DS_VNA)(siT*siP*CsumNLr.r + coT*siP/r*CsumNLt.r
					+ coP/siT/r*CsumNLp.r);

		    DS_VNA[3][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] =
		      -8.0*(Type_DS_VNA)(coT*CsumNLr.r - siT/r*CsumNLt.r);
		  }

		}

		else{
		  DS_VNA[1][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] = 0.0;
		  DS_VNA[2][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] = 0.0;
		  DS_VNA[3][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] = 0.0;
		} 

	      }
	    }

	    num1 = num1 + 2*L1 + 1; 
	  }

	  num0 = num0 + 2*L0 + 1; 
	} /* M0 */
      } /* L0 */

      if (measure_time && OMPID==0){
	dtime(&etime);
	time3 += etime - stime;
      }

      dtime(&Etime_atom);
      time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

    } /* Mc_AN */

    /* freeing of arrays */

    for (i=0; i<(2*(List_YOUSO[25]+1)+1); i++){
      free(CmatNL0[i]);
    }
    free(CmatNL0);

    for (i=0; i<(2*(List_YOUSO[25]+1)+1); i++){
      free(CmatNLr[i]);
    }
    free(CmatNLr);

    for (i=0; i<(2*(List_YOUSO[25]+1)+1); i++){
      free(CmatNLt[i]);
    }
    free(CmatNLt);

    for (i=0; i<(2*(List_YOUSO[25]+1)+1); i++){
      free(CmatNLp[i]);
    }
    free(CmatNLp);

    free(Bes00);
    free(Bes01);

    for (i=0; i<(List_YOUSO[25]+1); i++){
      for (j=0; j<List_YOUSO[24]; j++){
	free(SumNLr0[i][j]);
      }
      free(SumNLr0[i]);
    }
    free(SumNLr0);

    for (i=0; i<(List_YOUSO[25]+1); i++){
      for (j=0; j<List_YOUSO[24]; j++){
	free(SumNL0[i][j]);
      }
      free(SumNL0[i]);
    }
    free(SumNL0);

    for (i=0; i<(List_YOUSO[25]+1); i++){
      for (j=0; j<List_YOUSO[24]; j++){
	for (k=0; k<(2*(List_YOUSO[25]+1)+1); k++){
	  for (l=0; l<Num_RVNA; l++){
	    free(TmpNLp[i][j][k][l]);
	  }
          free(TmpNLp[i][j][k]);
	}
        free(TmpNLp[i][j]);
      }
      free(TmpNLp[i]);
    }
    free(TmpNLp);

    for (i=0; i<(List_YOUSO[25]+1); i++){
      for (j=0; j<List_YOUSO[24]; j++){
	for (k=0; k<(2*(List_YOUSO[25]+1)+1); k++){
	  for (l=0; l<Num_RVNA; l++){
	    free(TmpNLt[i][j][k][l]);
	  }
	  free(TmpNLt[i][j][k]);
	}
	free(TmpNLt[i][j]);
      }
      free(TmpNLt[i]);
    }
    free(TmpNLt);

    for (i=0; i<(List_YOUSO[25]+1); i++){
      for (j=0; j<List_YOUSO[24]; j++){
	for (k=0; k<(2*(List_YOUSO[25]+1)+1); k++){
	  for (l=0; l<Num_RVNA; l++){
	    free(TmpNLr[i][j][k][l]);
	  }
	  free(TmpNLr[i][j][k]);
	}
	free(TmpNLr[i][j]);
      }
      free(TmpNLr[i]);
    }
    free(TmpNLr);

    for (i=0; i<(List_YOUSO[25]+1); i++){
      for (j=0; j<List_YOUSO[24]; j++){
	for (k=0; k<(2*(List_YOUSO[25]+1)+1); k++){
	  for (l=0; l<Num_RVNA; l++){
	    free(TmpNL[i][j][k][l]);
	  }
	  free(TmpNL[i][j][k]);
	}
	free(TmpNL[i][j]);
      }
      free(TmpNL[i]);
    }
    free(TmpNL);

#pragma omp flush(DS_VNA)

  } /* #pragma omp parallel */

  } /* if (!setpro_gpu2) */

  /*******************************************************
   *******************************************************
     multiplying overlap integrals WITH COMMUNICATION

     MPI: communicate only for k=0
     DS_VNA
  *******************************************************
  *******************************************************/

  MPI_Barrier(mpi_comm_level1);
  if (measure_time) dtime(&stime);

  /* batched device path for the HVNA traces */

  setpro_gpu = SetPro_HVNA_DispatchBegin(DS_VNA, VNA_List, VNA_List2, Num_RVNA);

  /* allocation of array */

  NLH = (double**)malloc(sizeof(double*)*List_YOUSO[7]); 
  for (i=0; i<List_YOUSO[7]; i++){
    NLH[i] = (double*)malloc(sizeof(double)*List_YOUSO[7]); 
  }

  for (ID=0; ID<numprocs; ID++){
    F_Snd_Num_WK[ID] = 0;
    F_Rcv_Num_WK[ID] = 0;
  }

  do {

    /***********************************
          set the size of data
    ************************************/

    for (ID=0; ID<numprocs; ID++){

      IDS = (myid + ID) % numprocs;
      IDR = (myid - ID + numprocs) % numprocs;

      /* find the data size to send the block data */

      if ( 0<(F_Snd_Num[IDS]-F_Snd_Num_WK[IDS]) ){

	size1 = 0;
	n = F_Snd_Num_WK[IDS];

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

      if ( 0<(F_Rcv_Num[IDR]-F_Rcv_Num_WK[IDR]) ){
	MPI_Recv(&size2, 1, MPI_INT, IDR, tag, mpi_comm_level1, &stat);
	Rcv_DS_VNA_Size[IDR] = size2;
      }
      else{
	Rcv_DS_VNA_Size[IDR] = 0;
      }

      if ( 0<(F_Snd_Num[IDS]-F_Snd_Num_WK[IDS]) )  MPI_Wait(&request,&stat);

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

      if ( 0<(F_Snd_Num[IDS]-F_Snd_Num_WK[IDS]) ){

	size1 = Snd_DS_VNA_Size[IDS];

	/*
	printf("S myid=%2d ID=%2d IDS=%2d IDR=%2d  size1=%2d\n",myid,ID,IDS,IDR,size1);fflush(stdout);
	*/

	/* allocation of the array */

	tmp_array = (Type_DS_VNA*)malloc(sizeof(Type_DS_VNA)*size1);

	/* multidimentional array to the vector array */

	num = 0;
	n = F_Snd_Num_WK[IDS];

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

      if ( 0<(F_Rcv_Num[IDR]-F_Rcv_Num_WK[IDR]) ){
        
	size2 = Rcv_DS_VNA_Size[IDR];

	/*
	printf("R myid=%2d ID=%2d IDS=%2d IDR=%2d  size2=%2d\n",myid,ID,IDS,IDR,size2);fflush(stdout);
	*/

	tmp_array2 = (Type_DS_VNA*)malloc(sizeof(Type_DS_VNA)*size2);
	MPI_Recv(&tmp_array2[0], size2, MPI_Type_DS_VNA, IDR, tag, mpi_comm_level1, &stat);

	/* store */

	num = 0;
	n = F_Rcv_Num_WK[IDR];
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
	free(tmp_array2);

	if (setpro_gpu){
	  SetPro_HVNA_DispatchArchive(DS_VNA, Original_Mc_AN, Gc_AN);
	}

	/*****************************************
              multiplying overlap integrals
	*****************************************/

	if (!setpro_gpu){

	for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

	  dtime(&Stime_atom);
    
	  Gc_AN = M2G[Mc_AN];
	  Cwan = WhatSpecies[Gc_AN];
	  fan = FNAN[Gc_AN];
	  rcutA = Spe_Atom_Cut1[Cwan];

	  n = F_Rcv_Num_WK[IDR];
	  jg = Rcv_GAN[IDR][n];

	  for (j0=0; j0<=fan; j0++){

	    jg0 = natn[Gc_AN][j0];
	    Mj_AN0 = F_G2M[jg0];

  	    po = 0;
	    if (Original_Mc_AN==Mj_AN0){
	      po = 1;
	      j = j0;
	    }
  
	    if (po==1){

	      Hwan = WhatSpecies[jg];
	      rcutB = Spe_Atom_Cut1[Hwan];
	      rcut = rcutA + rcutB;

	      for (k=0; k<=fan; k++){

		kg = natn[Gc_AN][k];
		wakg = WhatSpecies[kg];
		kl = RMI1[Mc_AN][j][k];

		if (0<=kl){
#pragma omp parallel for private(m, n, sum, L ,L1, GL, Mul1, ene, L2 ,tmp0, L3) default(shared) if(Spe_Total_NO[Cwan] > 8)
		  for (m=0; m<Spe_Total_NO[Cwan]; m++){
		    for (n=0; n<Spe_Total_NO[Hwan]; n++){

		      sum = 0.0;
		      L = 0;

		      for (L1=0; L1<Num_RVNA; L1++){

			GL = VNA_List[L1]; 
			Mul1 = VNA_List2[L1];
            
			ene = VNA_proj_ene[wakg][GL][Mul1];
			L2 = 2*VNA_List[L1];

			tmp0 = 0.0;

			for (L3=0; L3<=L2; L3++){

			  tmp0 += fp32 ? (double)(DS_VNA[0][Mc_AN][k][m][L] * DS_VNA[0][Matomnum+1][kl][n][L])
			                   : (double)DS_VNA[0][Mc_AN][k][m][L] * (double)DS_VNA[0][Matomnum+1][kl][n][L];

			  L++;
			}

			sum += ene*tmp0; 

		      }

		      if (k==0)  NLH[m][n]  = sum;  
		      else       NLH[m][n] += sum; 

		    } /* n */
		  } /* m */
		} /* if (0<=kl)*/
	      } /* k */ 
       
	      /****************************************************
                            NLH to HVNA
	      ****************************************************/

	      dmp = dampingF(rcut,Dis[Gc_AN][j]);

	      for (i1=0; i1<Spe_Total_NO[Cwan]; i1++){
		for (j1=0; j1<Spe_Total_NO[Hwan]; j1++){
		  HVNA[Mc_AN][j][i1][j1] = dmp*NLH[i1][j1];
		}
	      }

	    } /* if (po==1) */

	  } /* j */            

	  dtime(&Etime_atom);
	  time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

	} /* Mc_AN */

	} /* if (!setpro_gpu) */

	/********************************************
            increment of F_Rcv_Num_WK[IDR] 
	********************************************/

	F_Rcv_Num_WK[IDR]++;

      }

      if ( 0<(F_Snd_Num[IDS]-F_Snd_Num_WK[IDS]) ) {
	MPI_Wait(&request,&stat);
	free(tmp_array);  /* freeing of array */

	/********************************************
             increment of F_Snd_Num_WK[IDS]
	********************************************/

	F_Snd_Num_WK[IDS]++;
      } 

    } /* ID */

    /*****************************************************
      check whether all the communications have finished
    *****************************************************/

    po = 0;
    for (ID=0; ID<numprocs; ID++){

      IDS = (myid + ID) % numprocs;
      IDR = (myid - ID + numprocs) % numprocs;

      if ( 0<(F_Snd_Num[IDS]-F_Snd_Num_WK[IDS]) ) po += F_Snd_Num[IDS]-F_Snd_Num_WK[IDS];
      if ( 0<(F_Rcv_Num[IDR]-F_Rcv_Num_WK[IDR]) ) po += F_Rcv_Num[IDR]-F_Rcv_Num_WK[IDR];
    }
  
  } while (po!=0);
  
  /* freeing of arrays */
  
  for (i=0; i<List_YOUSO[7]; i++){
    free(NLH[i]);
  }
  free(NLH);
  
  if (measure_time){
    dtime(&etime);
    time4 += etime - stime;
  }
  
  /*******************************************************
   *******************************************************
    multiplying overlap integrals WITHOUT COMMUNICATION
  *******************************************************
  *******************************************************/
  
  if (measure_time) dtime(&stime);

  if (!setpro_gpu){
  
#pragma omp parallel shared(List_YOUSO,time_per_atom,HVNA,Dis,DS_VNA,VNA_proj_ene,VNA_List2,VNA_List,Num_RVNA,Spe_Total_NO,RMI1,F_G2M,natn,Spe_Atom_Cut1,FNAN,WhatSpecies,M2G,Matomnum) 
  {         
  
    int OMPID,Nthrds,Nprocs;
    int Mc_AN,Gc_AN,Cwan,fan,Mj_AN,Hwan;
    int L,L1,GL,Mul1,L2,L3,i1,j1;
    int k,kg,wakg,kl,i,j,jg,m,n;
    double **NLH;
    double rcutA,rcutB,rcut,sum,ene,tmp0,dmp;
    double Stime_atom,Etime_atom;

    /* allocation of array */

    NLH = (double**)malloc(sizeof(double*)*List_YOUSO[7]); 
    for (i=0; i<List_YOUSO[7]; i++){
      NLH[i] = (double*)malloc(sizeof(double)*List_YOUSO[7]); 
    }

    /* get info. on OpenMP */ 

    OMPID = omp_get_thread_num();
    Nthrds = omp_get_num_threads();
    Nprocs = omp_get_num_procs();

    for (Mc_AN=(OMPID*Matomnum/Nthrds+1); Mc_AN<((OMPID+1)*Matomnum/Nthrds+1); Mc_AN++){

      dtime(&Stime_atom);
    
      Gc_AN = M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];
      fan = FNAN[Gc_AN];
      rcutA = Spe_Atom_Cut1[Cwan];
  
      for (j=0; j<=fan; j++){
  
	jg = natn[Gc_AN][j];
	Mj_AN = F_G2M[jg];

	if (Mj_AN<=Matomnum){

	  Hwan = WhatSpecies[jg];
	  rcutB = Spe_Atom_Cut1[Hwan];
	  rcut = rcutA + rcutB;

	  for (k=0; k<=fan; k++){

	    kg = natn[Gc_AN][k];
	    wakg = WhatSpecies[kg];
	    kl = RMI1[Mc_AN][j][k];

	    if (0<=kl){

	      for (m=0; m<Spe_Total_NO[Cwan]; m++){
		for (n=0; n<Spe_Total_NO[Hwan]; n++){

		  sum = 0.0;
		  L = 0;

		  for (L1=0; L1<Num_RVNA; L1++){

		    GL = VNA_List[L1]; 
		    Mul1 = VNA_List2[L1];
            
		    ene = VNA_proj_ene[wakg][GL][Mul1];
		    L2 = 2*VNA_List[L1];

		    tmp0 = 0.0;

		    for (L3=0; L3<=L2; L3++){
		      tmp0 += fp32 ? (double)(DS_VNA[0][Mc_AN][k][m][L] * DS_VNA[0][Mj_AN][kl][n][L])
		                       : (double)DS_VNA[0][Mc_AN][k][m][L] * (double)DS_VNA[0][Mj_AN][kl][n][L];
		      L++;
		    }

		    sum += ene*tmp0; 

		  }

		  if (k==0)  NLH[m][n]  = sum;  
		  else       NLH[m][n] += sum; 

		}
	      }
	    }
	  } /* k */ 
       
	  /****************************************************
                            NLH to HVNA
	  ****************************************************/

	  dmp = dampingF(rcut,Dis[Gc_AN][j]);

	  for (i1=0; i1<Spe_Total_NO[Cwan]; i1++){
	    for (j1=0; j1<Spe_Total_NO[Hwan]; j1++){
	      HVNA[Mc_AN][j][i1][j1] = dmp*NLH[i1][j1];
	    }
	  }

	} /* if (Mj_AN<=Matomnum) */
      } /* j */

      dtime(&Etime_atom);
      time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

    } /* Mc_AN */

    /* freeing of arrays */
 
    for (i=0; i<List_YOUSO[7]; i++){
      free(NLH[i]);
    }
    free(NLH);

#pragma omp flush(HVNA)

  } /* #pragma omp parallel */

  } /* if (!setpro_gpu) */

  if (setpro_gpu){
    SetPro_HVNA_DispatchRun(HVNA);
    setpro_gpu = 0;
  }

  if (measure_time){
    dtime(&etime);
    time5 += etime - stime;
  }

  if (measure_time){
    printf("Set_ProExpn_VNA myid=%2d time1=%7.3f time2=%7.3f time3=%7.3f time4=%7.3f time5=%7.3f\n",
            myid,time1,time2,time3,time4,time5);fflush(stdout); 
  }

  /****************************************************
                   freeing of arrays:
  ****************************************************/

  free(Snd_DS_VNA_Size);
  free(Rcv_DS_VNA_Size);

  free(VNA_List);
  free(VNA_List2);

  for (spe=0; spe<SpeciesNum; spe++){
    for (L0=0; L0<=Spe_MaxL_Basis[spe]; L0++){
      for (Mul0=0; Mul0<Spe_Num_Basis[spe][L0]; Mul0++){
        free(Bessel_Pro00[spe][L0][Mul0]);
      }
      free(Bessel_Pro00[spe][L0]);
    }
    free(Bessel_Pro00[spe]);
  }
  free(Bessel_Pro00);

  for (spe=0; spe<SpeciesNum; spe++){
    for (L0=0; L0<=Spe_MaxL_Basis[spe]; L0++){
      for (Mul0=0; Mul0<Spe_Num_Basis[spe][L0]; Mul0++){
        free(Bessel_Pro01[spe][L0][Mul0]);
      }
      free(Bessel_Pro01[spe][L0]);
    }
    free(Bessel_Pro01[spe]);
  }
  free(Bessel_Pro01);

  free(OneD2Mc_AN);
  free(OneD2h_AN);

  /* for time */
  dtime(&TEtime);
  time0 = TEtime - TStime;

  return time0;
} 




double Set_VNA2(double ****HVNA, double *****HVNA2)
{
  /****************************************************
   Evaluate matrix elements of one-center (orbitals)
   but two-center integrals for neutral atom potentials
   in the momentum space.

   <Phi_{LM,L'M'}|VNA>
  ****************************************************/
  static int firsttime=1;
  int L2,L3,L,GL,i,j;
  int k,kl,h_AN,Gh_AN,Rnh,Ls,n;
  int tno0,tno1,tno2,i1,j1,p;
  int Cwan,Hwan,fan,jg,kg,wakg,Lmax;
  int size_TmpHNA;
  int Mc_AN,Gc_AN,Mj_AN,num,size1,size2;
  double time0;
  double tmp2,tmp3,tmp4,tmp5;
  double TStime,TEtime;
  int Num_RVNA;
  /* for OpenMP */
  int OneD_Nloop,*OneD2Mc_AN,*OneD2h_AN;

  dtime(&TStime);

  /****************************************************
  allocation of arrays:
  ****************************************************/

  size_TmpHNA = (List_YOUSO[25]+1)*List_YOUSO[24]*
                (2*(List_YOUSO[25]+1)+1)*
                (List_YOUSO[25]+1)*List_YOUSO[24]*
                (2*(List_YOUSO[25]+1)+1);

  /* PrintMemory */
  if (firsttime) {
    PrintMemory("Set_ProExpn_VNA: TmpHNA",sizeof(double)*size_TmpHNA,NULL);
    PrintMemory("Set_ProExpn_VNA: TmpHNAr",sizeof(double)*size_TmpHNA,NULL);
    PrintMemory("Set_ProExpn_VNA: TmpHNAt",sizeof(double)*size_TmpHNA,NULL);
    PrintMemory("Set_ProExpn_VNA: TmpHNAp",sizeof(double)*size_TmpHNA,NULL);

    firsttime=0;
  }

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

  if (SetPro_gpu3.enabled){
    SetPro_VNA23_GpuRun(HVNA2, 0, OneD_Nloop, OneD2Mc_AN, OneD2h_AN);
  }

  if (!SetPro_gpu3.enabled){

#pragma omp parallel shared(List_YOUSO,time_per_atom,HVNA2,Comp2Real,GL_Weight,Spe_ProductRF_Bessel,Spe_CrudeVNA_Bessel,GL_NormK,Spe_Num_Basis,Spe_MaxL_Basis,PAO_Nkmax,atv,Gxyz,WhatSpecies,ncn,natn,M2G,OneD2h_AN,OneD2Mc_AN,OneD_Nloop) 
  {         
    int OMPID,Nthrds,Nprocs,Nloop;
    int Mc_AN,h_AN,Gc_AN,Cwan,Gh_AN;
    int Rnh,Hwan,L0,Mul0,L1,Mul1,M0,M1,LL;
    int Lmax_Four_Int,i,j,k,l,m,num0,num1;
    double dx,dy,dz;
    double siT,coT,siP,coP;
    double Normk,kmin,kmax,Sk,Dk;
    double gant,r,theta,phi;
    double SH[2],dSHt[2],dSHp[2];
    double S_coordinate[3];
    double Stime_atom,Etime_atom;
    double sum,sumr,sj,sjp,tmp0,tmp1,tmp10;
    double **SphB,**SphBp;
    double *tmp_SphB,*tmp_SphBp;

    dcomplex Ctmp0,Ctmp1,Ctmp2,Cpow;
    dcomplex Csum,Csumr,Csumt,Csump;
    dcomplex CsumS0,CsumSr,CsumSt,CsumSp;
    dcomplex CY,CYt,CYp,CY1,CYt1,CYp1;
    dcomplex **CmatS0;
    dcomplex **CmatSr;
    dcomplex **CmatSt;
    dcomplex **CmatSp;
    dcomplex ******TmpHNA;
    dcomplex ******TmpHNAr;
    dcomplex ******TmpHNAt;
    dcomplex ******TmpHNAp;

    /* allocation of arrays */

    TmpHNA = (dcomplex******)malloc(sizeof(dcomplex*****)*(List_YOUSO[25]+1));
    for (i=0; i<(List_YOUSO[25]+1); i++){
      TmpHNA[i] = (dcomplex*****)malloc(sizeof(dcomplex****)*List_YOUSO[24]);
      for (j=0; j<List_YOUSO[24]; j++){
	TmpHNA[i][j] = (dcomplex****)malloc(sizeof(dcomplex***)*(2*(List_YOUSO[25]+1)+1));
	for (k=0; k<(2*(List_YOUSO[25]+1)+1); k++){
	  TmpHNA[i][j][k] = (dcomplex***)malloc(sizeof(dcomplex**)*(List_YOUSO[25]+1));
	  for (l=0; l<(List_YOUSO[25]+1); l++){
	    TmpHNA[i][j][k][l] = (dcomplex**)malloc(sizeof(dcomplex*)*List_YOUSO[24]);
	    for (m=0; m<List_YOUSO[24]; m++){
	      TmpHNA[i][j][k][l][m] = (dcomplex*)malloc(sizeof(dcomplex)*(2*(List_YOUSO[25]+1)+1));
	    }
	  }
	}
      }
    }

    TmpHNAr = (dcomplex******)malloc(sizeof(dcomplex*****)*(List_YOUSO[25]+1));
    for (i=0; i<(List_YOUSO[25]+1); i++){
      TmpHNAr[i] = (dcomplex*****)malloc(sizeof(dcomplex****)*List_YOUSO[24]);
      for (j=0; j<List_YOUSO[24]; j++){
	TmpHNAr[i][j] = (dcomplex****)malloc(sizeof(dcomplex***)*(2*(List_YOUSO[25]+1)+1));
	for (k=0; k<(2*(List_YOUSO[25]+1)+1); k++){
	  TmpHNAr[i][j][k] = (dcomplex***)malloc(sizeof(dcomplex**)*(List_YOUSO[25]+1));
	  for (l=0; l<(List_YOUSO[25]+1); l++){
	    TmpHNAr[i][j][k][l] = (dcomplex**)malloc(sizeof(dcomplex*)*List_YOUSO[24]);
	    for (m=0; m<List_YOUSO[24]; m++){
	      TmpHNAr[i][j][k][l][m] = (dcomplex*)malloc(sizeof(dcomplex)*(2*(List_YOUSO[25]+1)+1));
	    }
	  }
	}
      }
    }

    TmpHNAt = (dcomplex******)malloc(sizeof(dcomplex*****)*(List_YOUSO[25]+1));
    for (i=0; i<(List_YOUSO[25]+1); i++){
      TmpHNAt[i] = (dcomplex*****)malloc(sizeof(dcomplex****)*List_YOUSO[24]);
      for (j=0; j<List_YOUSO[24]; j++){
	TmpHNAt[i][j] = (dcomplex****)malloc(sizeof(dcomplex***)*(2*(List_YOUSO[25]+1)+1));
	for (k=0; k<(2*(List_YOUSO[25]+1)+1); k++){
	  TmpHNAt[i][j][k] = (dcomplex***)malloc(sizeof(dcomplex**)*(List_YOUSO[25]+1));
	  for (l=0; l<(List_YOUSO[25]+1); l++){
	    TmpHNAt[i][j][k][l] = (dcomplex**)malloc(sizeof(dcomplex*)*List_YOUSO[24]);
	    for (m=0; m<List_YOUSO[24]; m++){
	      TmpHNAt[i][j][k][l][m] = (dcomplex*)malloc(sizeof(dcomplex)*(2*(List_YOUSO[25]+1)+1));
	    }
	  }
	}
      }
    }

    TmpHNAp = (dcomplex******)malloc(sizeof(dcomplex*****)*(List_YOUSO[25]+1));
    for (i=0; i<(List_YOUSO[25]+1); i++){
      TmpHNAp[i] = (dcomplex*****)malloc(sizeof(dcomplex****)*List_YOUSO[24]);
      for (j=0; j<List_YOUSO[24]; j++){
	TmpHNAp[i][j] = (dcomplex****)malloc(sizeof(dcomplex***)*(2*(List_YOUSO[25]+1)+1));
	for (k=0; k<(2*(List_YOUSO[25]+1)+1); k++){
	  TmpHNAp[i][j][k] = (dcomplex***)malloc(sizeof(dcomplex**)*(List_YOUSO[25]+1));
	  for (l=0; l<(List_YOUSO[25]+1); l++){
	    TmpHNAp[i][j][k][l] = (dcomplex**)malloc(sizeof(dcomplex*)*List_YOUSO[24]);
	    for (m=0; m<List_YOUSO[24]; m++){
	      TmpHNAp[i][j][k][l][m] = (dcomplex*)malloc(sizeof(dcomplex)*(2*(List_YOUSO[25]+1)+1));
	    }
	  }
	}
      }
    }

    CmatS0 = (dcomplex**)malloc(sizeof(dcomplex*)*(2*(List_YOUSO[25]+1)+1));
    for (i=0; i<(2*(List_YOUSO[25]+1)+1); i++){
      CmatS0[i] = (dcomplex*)malloc(sizeof(dcomplex)*(2*(List_YOUSO[25]+1)+1));
    }

    CmatSr = (dcomplex**)malloc(sizeof(dcomplex*)*(2*(List_YOUSO[25]+1)+1));
    for (i=0; i<(2*(List_YOUSO[25]+1)+1); i++){
      CmatSr[i] = (dcomplex*)malloc(sizeof(dcomplex)*(2*(List_YOUSO[25]+1)+1));
    }

    CmatSt = (dcomplex**)malloc(sizeof(dcomplex*)*(2*(List_YOUSO[25]+1)+1));
    for (i=0; i<(2*(List_YOUSO[25]+1)+1); i++){
      CmatSt[i] = (dcomplex*)malloc(sizeof(dcomplex)*(2*(List_YOUSO[25]+1)+1));
    }

    CmatSp = (dcomplex**)malloc(sizeof(dcomplex*)*(2*(List_YOUSO[25]+1)+1));
    for (i=0; i<(2*(List_YOUSO[25]+1)+1); i++){
      CmatSp[i] = (dcomplex*)malloc(sizeof(dcomplex)*(2*(List_YOUSO[25]+1)+1));
    }

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
         evaluate ovelap integrals <Phi_{LM,L'M'}|VNA>
         between VNA and PAO.
      ****************************************************/
      /****************************************************
       \int VNA(k)*Spe_ProductRF_Bessel(k)*jl(k*R) k^2 dk^3 
      ****************************************************/

      kmin = Radial_kmin;
      kmax = PAO_Nkmax;
      Sk = kmax + kmin;
      Dk = kmax - kmin;

      for (L0=0; L0<=Spe_MaxL_Basis[Cwan]; L0++){
	for (Mul0=0; Mul0<Spe_Num_Basis[Cwan][L0]; Mul0++){
	  for (L1=0; L1<=Spe_MaxL_Basis[Cwan]; L1++){
	    for (Mul1=0; Mul1<Spe_Num_Basis[Cwan][L1]; Mul1++){
	      for (M0=-L0; M0<=L0; M0++){
		for (M1=-L1; M1<=L1; M1++){
		  TmpHNA[L0][Mul0][L0+M0][L1][Mul1][L1+M1] = Complex(0.0,0.0); 
		  TmpHNAr[L0][Mul0][L0+M0][L1][Mul1][L1+M1] = Complex(0.0,0.0); 
		  TmpHNAt[L0][Mul0][L0+M0][L1][Mul1][L1+M1] = Complex(0.0,0.0); 
		  TmpHNAp[L0][Mul0][L0+M0][L1][Mul1][L1+M1] = Complex(0.0,0.0); 
		}
	      }
	    }
	  }
	}
      }

      /* allocate SphB and SphBp */

      Lmax_Four_Int = 2*Spe_MaxL_Basis[Cwan];

      SphB = (double**)malloc(sizeof(double*)*(Lmax_Four_Int+3));
      for(LL=0; LL<(Lmax_Four_Int+3); LL++){ 
	SphB[LL] = (double*)malloc(sizeof(double)*GL_Mesh);
      }

      SphBp = (double**)malloc(sizeof(double*)*(Lmax_Four_Int+3));
      for(LL=0; LL<(Lmax_Four_Int+3); LL++){ 
	SphBp[LL] = (double*)malloc(sizeof(double)*GL_Mesh);
      }

      tmp_SphB  = (double*)malloc(sizeof(double)*(Lmax_Four_Int+3));
      tmp_SphBp = (double*)malloc(sizeof(double)*(Lmax_Four_Int+3));

      /* calculate SphB and SphBp */
#ifdef kcomp
#else 
#pragma forceinline recursive
#endif      
      for (i=0; i<GL_Mesh; i++){
	Normk = GL_NormK[i];
	Spherical_Bessel2(Normk*r,Lmax_Four_Int,tmp_SphB,tmp_SphBp);
	for(LL=0; LL<=Lmax_Four_Int; LL++){ 
	  SphB[LL][i]  = tmp_SphB[LL]; 
	  SphBp[LL][i] = tmp_SphBp[LL]; 
	}
      }

      free(tmp_SphB);
      free(tmp_SphBp);

      /* loops for L0, Mul0, L1, and Mul1 */

      for (L0=0; L0<=Spe_MaxL_Basis[Cwan]; L0++){
	for (Mul0=0; Mul0<Spe_Num_Basis[Cwan][L0]; Mul0++){
	  for (L1=0; L1<=Spe_MaxL_Basis[Cwan]; L1++){
	    for (Mul1=0; Mul1<Spe_Num_Basis[Cwan][L1]; Mul1++){

	      if (L0<=L1){
		Lmax_Four_Int = 2*L1;

		/* sum over LL */

		for(LL=0; LL<=Lmax_Four_Int; LL++){ 

		  if (__builtin_abs(L1-LL)<=L0 && L0<=(L1+LL) ){

		    sum  = 0.0;
		    sumr = 0.0;

		    /* Gauss-Legendre quadrature */

		    for (i=0; i<GL_Mesh; i++){

		      Normk = GL_NormK[i];

		      sj  =  SphB[LL][i];
		      sjp = SphBp[LL][i];

		      tmp0 = Spe_CrudeVNA_Bessel[Hwan][i];
		      tmp1 = Spe_ProductRF_Bessel[Cwan][L0][Mul0][L1][Mul1][LL][i]; 
		      tmp10 = 0.50*Dk*tmp0*tmp1*GL_Weight[i]*Normk*Normk;

		      sum  += tmp10*sj;
		      sumr += tmp10*sjp*Normk;
		    }

		    /* sum over m */

		    for (M0=-L0; M0<=L0; M0++){
		      for (M1=-L1; M1<=L1; M1++){

			Csum = Complex(0.0,0.0);
			Csumr = Complex(0.0,0.0);
			Csumt = Complex(0.0,0.0);
			Csump = Complex(0.0,0.0);

			for(m=-LL; m<=LL; m++){ 

			  if ( (M1-m)==M0){

			    ComplexSH(LL,m,theta,phi,SH,dSHt,dSHp);
			    CY  = Complex(SH[0],SH[1]);
			    CYt = Complex(dSHt[0],dSHt[1]);
			    CYp = Complex(dSHp[0],dSHp[1]);

			    gant = pow(-1.0,(double)__builtin_abs(m))*Gaunt(L0,M0,L1,M1,LL,-m);
 
			    /* S */

			    Ctmp2 = CRmul(CY,gant);          
			    Csum = Cadd(Csum,Ctmp2);             

			    /* dS/dt */
                      
			    Ctmp2 = CRmul(CYt,gant);          
			    Csumt = Cadd(Csumt,Ctmp2);             
                      
			    /* dS/dp */
                      
			    Ctmp2 = CRmul(CYp,gant);          
			    Csump = Cadd(Csump,Ctmp2);             
			  }

			}

			/* S */
			Ctmp1 = CRmul(Csum,sum);
			TmpHNA[L0][Mul0][L0+M0][L1][Mul1][L1+M1] 
			  = Cadd(TmpHNA[L0][Mul0][L0+M0][L1][Mul1][L1+M1],Ctmp1);

			/* dS/dr */
			Ctmp1 = CRmul(Csum,sumr);
			TmpHNAr[L0][Mul0][L0+M0][L1][Mul1][L1+M1] 
			  = Cadd(TmpHNAr[L0][Mul0][L0+M0][L1][Mul1][L1+M1],Ctmp1);

			/* dS/dt */
			Ctmp1 = CRmul(Csumt,sum);
			TmpHNAt[L0][Mul0][L0+M0][L1][Mul1][L1+M1] 
			  = Cadd(TmpHNAt[L0][Mul0][L0+M0][L1][Mul1][L1+M1],Ctmp1);

			/* dS/dp */
			Ctmp1 = CRmul(Csump,sum);
			TmpHNAp[L0][Mul0][L0+M0][L1][Mul1][L1+M1] 
			  = Cadd(TmpHNAp[L0][Mul0][L0+M0][L1][Mul1][L1+M1],Ctmp1);
		      }
		    }
		  }
		} /* LL */
	      } /* if (L0<=L1) */
	    }
	  }
	}
      }

      /* free SphB and SphBp */
      
      for(LL=0; LL<(Lmax_Four_Int+3); LL++){ 
	free(SphB[LL]);
      }
      free(SphB);

      for(LL=0; LL<(Lmax_Four_Int+3); LL++){ 
	free(SphBp[LL]);
      }
      free(SphBp);

      /* copy the upper part to the lower part */

      for (L0=0; L0<=Spe_MaxL_Basis[Cwan]; L0++){
	for (Mul0=0; Mul0<Spe_Num_Basis[Cwan][L0]; Mul0++){
	  for (L1=0; L1<=Spe_MaxL_Basis[Cwan]; L1++){
	    for (Mul1=0; Mul1<Spe_Num_Basis[Cwan][L1]; Mul1++){
	      if (L0<=L1){
		for (M0=-L0; M0<=L0; M0++){
		  for (M1=-L1; M1<=L1; M1++){

		    TmpHNA[L1][Mul1][L1+M1][L0][Mul0][L0+M0] 
		      = Conjg(TmpHNA[L0][Mul0][L0+M0][L1][Mul1][L1+M1]); 

		    TmpHNAr[L1][Mul1][L1+M1][L0][Mul0][L0+M0]
		      = Conjg(TmpHNAr[L0][Mul0][L0+M0][L1][Mul1][L1+M1]);
 
		    TmpHNAt[L1][Mul1][L1+M1][L0][Mul0][L0+M0]
		      = Conjg(TmpHNAt[L0][Mul0][L0+M0][L1][Mul1][L1+M1]); 

		    TmpHNAp[L1][Mul1][L1+M1][L0][Mul0][L0+M0]
		      = Conjg(TmpHNAp[L0][Mul0][L0+M0][L1][Mul1][L1+M1]); 

		  }
		}
	      }
	    }
	  }
	}
      }

      /****************************************************
 	                 Complex to Real
      ****************************************************/

      num0 = 0;
      for (L0=0; L0<=Spe_MaxL_Basis[Cwan]; L0++){
	for (Mul0=0; Mul0<Spe_Num_Basis[Cwan][L0]; Mul0++){
       
	  num1 = 0;
	  for (L1=0; L1<=Spe_MaxL_Basis[Cwan]; L1++){
	    for (Mul1=0; Mul1<Spe_Num_Basis[Cwan][L1]; Mul1++){

	      for (M0=-L0; M0<=L0; M0++){
		for (M1=-L1; M1<=L1; M1++){

		  CsumS0 = Complex(0.0,0.0);
		  CsumSr = Complex(0.0,0.0);
		  CsumSt = Complex(0.0,0.0);
		  CsumSp = Complex(0.0,0.0);

		  for (k=-L0; k<=L0; k++){

		    Ctmp1 = Conjg(Comp2Real[L0][L0+M0][L0+k]);

		    /* S */

		    Ctmp0 = TmpHNA[L0][Mul0][L0+k][L1][Mul1][L1+M1];
		    Ctmp2 = Cmul(Ctmp1,Ctmp0);
		    CsumS0 = Cadd(CsumS0,Ctmp2);

		    /* dS/dr */

		    Ctmp0 = TmpHNAr[L0][Mul0][L0+k][L1][Mul1][L1+M1];
		    Ctmp2 = Cmul(Ctmp1,Ctmp0);
		    CsumSr = Cadd(CsumSr,Ctmp2);

		    /* dS/dt */

		    Ctmp0 = TmpHNAt[L0][Mul0][L0+k][L1][Mul1][L1+M1];
		    Ctmp2 = Cmul(Ctmp1,Ctmp0);
		    CsumSt = Cadd(CsumSt,Ctmp2);

		    /* dS/dp */

		    Ctmp0 = TmpHNAp[L0][Mul0][L0+k][L1][Mul1][L1+M1];
		    Ctmp2 = Cmul(Ctmp1,Ctmp0);
		    CsumSp = Cadd(CsumSp,Ctmp2);

		  }

		  CmatS0[L0+M0][L1+M1] = CsumS0;
		  CmatSr[L0+M0][L1+M1] = CsumSr;
		  CmatSt[L0+M0][L1+M1] = CsumSt;
		  CmatSp[L0+M0][L1+M1] = CsumSp;

		}
	      }

	      for (M0=-L0; M0<=L0; M0++){
		for (M1=-L1; M1<=L1; M1++){

		  CsumS0 = Complex(0.0,0.0);
		  CsumSr = Complex(0.0,0.0);
		  CsumSt = Complex(0.0,0.0);
		  CsumSp = Complex(0.0,0.0);

		  for (k=-L1; k<=L1; k++){

		    /* S */ 

		    Ctmp1 = Cmul(CmatS0[L0+M0][L1+k],Comp2Real[L1][L1+M1][L1+k]);
		    CsumS0 = Cadd(CsumS0,Ctmp1);

		    /* dS/dr */ 

		    Ctmp1 = Cmul(CmatSr[L0+M0][L1+k],Comp2Real[L1][L1+M1][L1+k]);
		    CsumSr = Cadd(CsumSr,Ctmp1);

		    /* dS/dt */ 

		    Ctmp1 = Cmul(CmatSt[L0+M0][L1+k],Comp2Real[L1][L1+M1][L1+k]);
		    CsumSt = Cadd(CsumSt,Ctmp1);

		    /* dS/dp */ 

		    Ctmp1 = Cmul(CmatSp[L0+M0][L1+k],Comp2Real[L1][L1+M1][L1+k]);
		    CsumSp = Cadd(CsumSp,Ctmp1);

		  }

		  HVNA2[0][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] = 8.0*CsumS0.r;

		  if (h_AN!=0){

		    if (fabs(siT)<10e-14){

		      HVNA2[1][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] =
			-8.0*(siT*coP*CsumSr.r + coT*coP/r*CsumSt.r);

		      HVNA2[2][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] =
			-8.0*(siT*siP*CsumSr.r + coT*siP/r*CsumSt.r);

		      HVNA2[3][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] =
			-8.0*(coT*CsumSr.r - siT/r*CsumSt.r);
		    }

		    else{

		      HVNA2[1][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] =
			-8.0*(siT*coP*CsumSr.r + coT*coP/r*CsumSt.r
			     - siP/siT/r*CsumSp.r);

		      HVNA2[2][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] =
			-8.0*(siT*siP*CsumSr.r + coT*siP/r*CsumSt.r
			     + coP/siT/r*CsumSp.r);

		      HVNA2[3][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] =
			-8.0*(coT*CsumSr.r - siT/r*CsumSt.r);
		    }
		  }
		  else{
		    HVNA2[1][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] = 0.0;
		    HVNA2[2][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] = 0.0;
		    HVNA2[3][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] = 0.0;
		  }

		}
	      }

	      num1 = num1 + 2*L1 + 1; 
	    }
	  }

	  num0 = num0 + 2*L0 + 1; 
	}
      }

      dtime(&Etime_atom);
      time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

    } /* Mc_AN */

    /* freeing of arrays */

    for (i=0; i<(List_YOUSO[25]+1); i++){
      for (j=0; j<List_YOUSO[24]; j++){
	for (k=0; k<(2*(List_YOUSO[25]+1)+1); k++){
	  for (l=0; l<(List_YOUSO[25]+1); l++){
	    for (m=0; m<List_YOUSO[24]; m++){
	      free(TmpHNA[i][j][k][l][m]);
	    }
	    free(TmpHNA[i][j][k][l]);
	  }
	  free(TmpHNA[i][j][k]);
	}
	free(TmpHNA[i][j]);
      }
      free(TmpHNA[i]);
    }
    free(TmpHNA);

    for (i=0; i<(List_YOUSO[25]+1); i++){
      for (j=0; j<List_YOUSO[24]; j++){
	for (k=0; k<(2*(List_YOUSO[25]+1)+1); k++){
	  for (l=0; l<(List_YOUSO[25]+1); l++){
	    for (m=0; m<List_YOUSO[24]; m++){
	      free(TmpHNAr[i][j][k][l][m]);
	    }
	    free(TmpHNAr[i][j][k][l]);
	  }
	  free(TmpHNAr[i][j][k]);
	}
	free(TmpHNAr[i][j]);
      }
      free(TmpHNAr[i]);
    }
    free(TmpHNAr);

    for (i=0; i<(List_YOUSO[25]+1); i++){
      for (j=0; j<List_YOUSO[24]; j++){
	for (k=0; k<(2*(List_YOUSO[25]+1)+1); k++){
	  for (l=0; l<(List_YOUSO[25]+1); l++){
	    for (m=0; m<List_YOUSO[24]; m++){
	      free(TmpHNAt[i][j][k][l][m]);
	    }
	    free(TmpHNAt[i][j][k][l]);
	  }
	  free(TmpHNAt[i][j][k]);
	}
	free(TmpHNAt[i][j]);
      }
      free(TmpHNAt[i]);
    }
    free(TmpHNAt);

    for (i=0; i<(List_YOUSO[25]+1); i++){
      for (j=0; j<List_YOUSO[24]; j++){
	for (k=0; k<(2*(List_YOUSO[25]+1)+1); k++){
	  for (l=0; l<(List_YOUSO[25]+1); l++){
	    for (m=0; m<List_YOUSO[24]; m++){
	      free(TmpHNAp[i][j][k][l][m]);
	    }
	    free(TmpHNAp[i][j][k][l]);
	  }
	  free(TmpHNAp[i][j][k]);
	}
	free(TmpHNAp[i][j]);
      }
      free(TmpHNAp[i]);
    }
    free(TmpHNAp);

    for (i=0; i<(2*(List_YOUSO[25]+1)+1); i++){
      free(CmatS0[i]);
    }
    free(CmatS0);

    for (i=0; i<(2*(List_YOUSO[25]+1)+1); i++){
      free(CmatSr[i]);
    }
    free(CmatSr);

    for (i=0; i<(2*(List_YOUSO[25]+1)+1); i++){
      free(CmatSt[i]);
    }
    free(CmatSt);

    for (i=0; i<(2*(List_YOUSO[25]+1)+1); i++){
      free(CmatSp[i]);
    }
    free(CmatSp);

#pragma omp flush(HVNA2)

  } /* #pragma omp parallel */

  } /* if (!SetPro_gpu3.enabled) */

  /****************************************************
    HVNA[Mc_AN][0] = sum_{h_AN} HVNA2
  ****************************************************/

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

    Gc_AN = M2G[Mc_AN];
    Cwan = WhatSpecies[Gc_AN];
    for (i=0; i<Spe_Total_NO[Cwan]; i++){
      for (j=0; j<Spe_Total_NO[Cwan]; j++){
	HVNA[Mc_AN][0][i][j] = 0.0;
      }
    }

    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
      for (i=0; i<Spe_Total_NO[Cwan]; i++){
        for (j=0; j<Spe_Total_NO[Cwan]; j++){
          HVNA[Mc_AN][0][i][j] += HVNA2[0][Mc_AN][h_AN][i][j];
	}
      }
    }
  } 

  /****************************************************
  freeing of arrays:
  ****************************************************/

  free(OneD2Mc_AN);
  free(OneD2h_AN);

  /* for time */
  dtime(&TEtime);
  time0 = TEtime - TStime;
  return time0;
} 



double Set_VNA3(double *****HVNA3)
{
  /****************************************************
   Evaluate matrix elements of one-center (orbitals)
   but two-center integrals for neutral atom potentials
   in the momentum space.

   <VNA|Phi_{LM,L'M'}>
  ****************************************************/
  static int firsttime=0; /* due to the same as in Set_VNA2 */
  int L2,L3,L,GL,i,j;
  int k,kl,h_AN,Gh_AN,Rnh,Ls,n;
  int tno0,tno1,tno2,i1,j1,p;
  int Cwan,Hwan,fan,jg,kg,wakg,Lmax;
  int size_TmpHNA;
  int Mc_AN,Gc_AN,Mj_AN,num,size1,size2;
  double time0;
  double tmp2,tmp3,tmp4,tmp5;
  double TStime,TEtime;
  int Num_RVNA;

  /* for OpenMP */
  int OneD_Nloop,*OneD2Mc_AN,*OneD2h_AN;

  dtime(&TStime);

  /****************************************************
  allocation of arrays:
  ****************************************************/

  size_TmpHNA = (List_YOUSO[25]+1)*List_YOUSO[24]*
                (2*(List_YOUSO[25]+1)+1)*
                (List_YOUSO[25]+1)*List_YOUSO[24]*
                (2*(List_YOUSO[25]+1)+1);

  /* PrintMemory */
  if (firsttime) {
    PrintMemory("Set_ProExpn_VNA: TmpHNA",sizeof(double)*size_TmpHNA,NULL);
    PrintMemory("Set_ProExpn_VNA: TmpHNAr",sizeof(double)*size_TmpHNA,NULL);
    PrintMemory("Set_ProExpn_VNA: TmpHNAt",sizeof(double)*size_TmpHNA,NULL);
    PrintMemory("Set_ProExpn_VNA: TmpHNAp",sizeof(double)*size_TmpHNA,NULL);

    firsttime=0;
  }

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

  if (SetPro_gpu3.enabled){
    SetPro_VNA23_GpuRun(HVNA3, 1, OneD_Nloop, OneD2Mc_AN, OneD2h_AN);
  }

  if (!SetPro_gpu3.enabled){

#pragma omp parallel shared(List_YOUSO,time_per_atom,HVNA3,Comp2Real,GL_Weight,Spe_ProductRF_Bessel,Spe_CrudeVNA_Bessel,GL_NormK,Spe_Num_Basis,Spe_MaxL_Basis,PAO_Nkmax,atv,Gxyz,WhatSpecies,ncn,natn,M2G,OneD2h_AN,OneD2Mc_AN,OneD_Nloop) 
  {         
    int OMPID,Nthrds,Nprocs,Nloop;
    int Mc_AN,h_AN,Gc_AN,Cwan,Gh_AN;
    int Rnh,Hwan,L0,Mul0,L1,Mul1,M0,M1,LL;
    int Lmax_Four_Int,i,j,k,l,m,num0,num1;
    double dx,dy,dz;
    double siT,coT,siP,coP;
    double Normk,kmin,kmax,Sk,Dk;
    double gant,r,theta,phi;
    double SH[2],dSHt[2],dSHp[2];
    double S_coordinate[3];
    double Stime_atom,Etime_atom;
    double sum,sumr,sj,sjp,tmp0,tmp1,tmp10;
    double **SphB,**SphBp;
    double *tmp_SphB,*tmp_SphBp;

    dcomplex Ctmp0,Ctmp1,Ctmp2,Cpow;
    dcomplex Csum,Csumr,Csumt,Csump;
    dcomplex CsumS0,CsumSr,CsumSt,CsumSp;
    dcomplex CY,CYt,CYp,CY1,CYt1,CYp1;
    dcomplex **CmatS0;
    dcomplex **CmatSr;
    dcomplex **CmatSt;
    dcomplex **CmatSp;
    dcomplex ******TmpHNA;
    dcomplex ******TmpHNAr;
    dcomplex ******TmpHNAt;
    dcomplex ******TmpHNAp;

    /* allocation of arrays */

    TmpHNA = (dcomplex******)malloc(sizeof(dcomplex*****)*(List_YOUSO[25]+1));
    for (i=0; i<(List_YOUSO[25]+1); i++){
      TmpHNA[i] = (dcomplex*****)malloc(sizeof(dcomplex****)*List_YOUSO[24]);
      for (j=0; j<List_YOUSO[24]; j++){
	TmpHNA[i][j] = (dcomplex****)malloc(sizeof(dcomplex***)*(2*(List_YOUSO[25]+1)+1));
	for (k=0; k<(2*(List_YOUSO[25]+1)+1); k++){
	  TmpHNA[i][j][k] = (dcomplex***)malloc(sizeof(dcomplex**)*(List_YOUSO[25]+1));
	  for (l=0; l<(List_YOUSO[25]+1); l++){
	    TmpHNA[i][j][k][l] = (dcomplex**)malloc(sizeof(dcomplex*)*List_YOUSO[24]);
	    for (m=0; m<List_YOUSO[24]; m++){
	      TmpHNA[i][j][k][l][m] = (dcomplex*)malloc(sizeof(dcomplex)*(2*(List_YOUSO[25]+1)+1));
	    }
	  }
	}
      }
    }

    TmpHNAr = (dcomplex******)malloc(sizeof(dcomplex*****)*(List_YOUSO[25]+1));
    for (i=0; i<(List_YOUSO[25]+1); i++){
      TmpHNAr[i] = (dcomplex*****)malloc(sizeof(dcomplex****)*List_YOUSO[24]);
      for (j=0; j<List_YOUSO[24]; j++){
	TmpHNAr[i][j] = (dcomplex****)malloc(sizeof(dcomplex***)*(2*(List_YOUSO[25]+1)+1));
	for (k=0; k<(2*(List_YOUSO[25]+1)+1); k++){
	  TmpHNAr[i][j][k] = (dcomplex***)malloc(sizeof(dcomplex**)*(List_YOUSO[25]+1));
	  for (l=0; l<(List_YOUSO[25]+1); l++){
	    TmpHNAr[i][j][k][l] = (dcomplex**)malloc(sizeof(dcomplex*)*List_YOUSO[24]);
	    for (m=0; m<List_YOUSO[24]; m++){
	      TmpHNAr[i][j][k][l][m] = (dcomplex*)malloc(sizeof(dcomplex)*(2*(List_YOUSO[25]+1)+1));
	    }
	  }
	}
      }
    }

    TmpHNAt = (dcomplex******)malloc(sizeof(dcomplex*****)*(List_YOUSO[25]+1));
    for (i=0; i<(List_YOUSO[25]+1); i++){
      TmpHNAt[i] = (dcomplex*****)malloc(sizeof(dcomplex****)*List_YOUSO[24]);
      for (j=0; j<List_YOUSO[24]; j++){
	TmpHNAt[i][j] = (dcomplex****)malloc(sizeof(dcomplex***)*(2*(List_YOUSO[25]+1)+1));
	for (k=0; k<(2*(List_YOUSO[25]+1)+1); k++){
	  TmpHNAt[i][j][k] = (dcomplex***)malloc(sizeof(dcomplex**)*(List_YOUSO[25]+1));
	  for (l=0; l<(List_YOUSO[25]+1); l++){
	    TmpHNAt[i][j][k][l] = (dcomplex**)malloc(sizeof(dcomplex*)*List_YOUSO[24]);
	    for (m=0; m<List_YOUSO[24]; m++){
	      TmpHNAt[i][j][k][l][m] = (dcomplex*)malloc(sizeof(dcomplex)*(2*(List_YOUSO[25]+1)+1));
	    }
	  }
	}
      }
    }

    TmpHNAp = (dcomplex******)malloc(sizeof(dcomplex*****)*(List_YOUSO[25]+1));
    for (i=0; i<(List_YOUSO[25]+1); i++){
      TmpHNAp[i] = (dcomplex*****)malloc(sizeof(dcomplex****)*List_YOUSO[24]);
      for (j=0; j<List_YOUSO[24]; j++){
	TmpHNAp[i][j] = (dcomplex****)malloc(sizeof(dcomplex***)*(2*(List_YOUSO[25]+1)+1));
	for (k=0; k<(2*(List_YOUSO[25]+1)+1); k++){
	  TmpHNAp[i][j][k] = (dcomplex***)malloc(sizeof(dcomplex**)*(List_YOUSO[25]+1));
	  for (l=0; l<(List_YOUSO[25]+1); l++){
	    TmpHNAp[i][j][k][l] = (dcomplex**)malloc(sizeof(dcomplex*)*List_YOUSO[24]);
	    for (m=0; m<List_YOUSO[24]; m++){
	      TmpHNAp[i][j][k][l][m] = (dcomplex*)malloc(sizeof(dcomplex)*(2*(List_YOUSO[25]+1)+1));
	    }
	  }
	}
      }
    }

    CmatS0 = (dcomplex**)malloc(sizeof(dcomplex*)*(2*(List_YOUSO[25]+1)+1));
    for (i=0; i<(2*(List_YOUSO[25]+1)+1); i++){
      CmatS0[i] = (dcomplex*)malloc(sizeof(dcomplex)*(2*(List_YOUSO[25]+1)+1));
    }

    CmatSr = (dcomplex**)malloc(sizeof(dcomplex*)*(2*(List_YOUSO[25]+1)+1));
    for (i=0; i<(2*(List_YOUSO[25]+1)+1); i++){
      CmatSr[i] = (dcomplex*)malloc(sizeof(dcomplex)*(2*(List_YOUSO[25]+1)+1));
    }

    CmatSt = (dcomplex**)malloc(sizeof(dcomplex*)*(2*(List_YOUSO[25]+1)+1));
    for (i=0; i<(2*(List_YOUSO[25]+1)+1); i++){
      CmatSt[i] = (dcomplex*)malloc(sizeof(dcomplex)*(2*(List_YOUSO[25]+1)+1));
    }

    CmatSp = (dcomplex**)malloc(sizeof(dcomplex*)*(2*(List_YOUSO[25]+1)+1));
    for (i=0; i<(2*(List_YOUSO[25]+1)+1); i++){
      CmatSp[i] = (dcomplex*)malloc(sizeof(dcomplex)*(2*(List_YOUSO[25]+1)+1));
    }

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
      Rnh = ncn[Gc_AN][h_AN];
      Hwan = WhatSpecies[Gh_AN];

      dx = Gxyz[Gc_AN][1] - Gxyz[Gh_AN][1] - atv[Rnh][1];
      dy = Gxyz[Gc_AN][2] - Gxyz[Gh_AN][2] - atv[Rnh][2];
      dz = Gxyz[Gc_AN][3] - Gxyz[Gh_AN][3] - atv[Rnh][3];

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
         evaluate ovelap integrals <Phi_{LM,L'M'}|VNA>
         between VNA and PAO.
      ****************************************************/
      /****************************************************
       \int VNA(k)*Spe_ProductRF_Bessel(k)*jl(k*R) k^2 dk^3 
      ****************************************************/

      kmin = Radial_kmin;
      kmax = PAO_Nkmax;
      Sk = kmax + kmin;
      Dk = kmax - kmin;

      for (L0=0; L0<=Spe_MaxL_Basis[Hwan]; L0++){
	for (Mul0=0; Mul0<Spe_Num_Basis[Hwan][L0]; Mul0++){
	  for (L1=0; L1<=Spe_MaxL_Basis[Hwan]; L1++){
	    for (Mul1=0; Mul1<Spe_Num_Basis[Hwan][L1]; Mul1++){
	      for (M0=-L0; M0<=L0; M0++){
		for (M1=-L1; M1<=L1; M1++){
		  TmpHNA[L0][Mul0][L0+M0][L1][Mul1][L1+M1] = Complex(0.0,0.0); 
		  TmpHNAr[L0][Mul0][L0+M0][L1][Mul1][L1+M1] = Complex(0.0,0.0); 
		  TmpHNAt[L0][Mul0][L0+M0][L1][Mul1][L1+M1] = Complex(0.0,0.0); 
		  TmpHNAp[L0][Mul0][L0+M0][L1][Mul1][L1+M1] = Complex(0.0,0.0); 
		}
	      }
	    }
	  }
	}
      }

      /* allocate SphB and SphBp */

      Lmax_Four_Int = 2*Spe_MaxL_Basis[Hwan];

      SphB = (double**)malloc(sizeof(double*)*(Lmax_Four_Int+3));
      for(LL=0; LL<(Lmax_Four_Int+3); LL++){ 
	SphB[LL] = (double*)malloc(sizeof(double)*GL_Mesh);
      }

      SphBp = (double**)malloc(sizeof(double*)*(Lmax_Four_Int+3));
      for(LL=0; LL<(Lmax_Four_Int+3); LL++){ 
	SphBp[LL] = (double*)malloc(sizeof(double)*GL_Mesh);
      }

      tmp_SphB  = (double*)malloc(sizeof(double)*(Lmax_Four_Int+3));
      tmp_SphBp = (double*)malloc(sizeof(double)*(Lmax_Four_Int+3));

      /* calculate SphB and SphBp */
#ifdef kcomp
#else 
#pragma forceinline recursive
#endif      
      for (i=0; i<GL_Mesh; i++){
	Normk = GL_NormK[i];
	Spherical_Bessel2(Normk*r,Lmax_Four_Int,tmp_SphB,tmp_SphBp);
	for(LL=0; LL<=Lmax_Four_Int; LL++){ 
	  SphB[LL][i]  = tmp_SphB[LL]; 
	  SphBp[LL][i] = tmp_SphBp[LL]; 
	}
      }

      free(tmp_SphB);
      free(tmp_SphBp);

      /* loops for L0, Mul0, L1, and Mul1 */

      for (L0=0; L0<=Spe_MaxL_Basis[Hwan]; L0++){
	for (Mul0=0; Mul0<Spe_Num_Basis[Hwan][L0]; Mul0++){
	  for (L1=0; L1<=Spe_MaxL_Basis[Hwan]; L1++){
	    for (Mul1=0; Mul1<Spe_Num_Basis[Hwan][L1]; Mul1++){

	      if (L0<=L1){

		Lmax_Four_Int = 2*L1;

		/* sum over LL */

		for(LL=0; LL<=Lmax_Four_Int; LL++){ 

		  if (__builtin_abs(L1-LL)<=L0 && L0<=(L1+LL) ){

		    sum  = 0.0;
		    sumr = 0.0;

		    /* Gauss-Legendre quadrature */

		    for (i=0; i<GL_Mesh; i++){

		      Normk = GL_NormK[i];

		      sj  =  SphB[LL][i];
		      sjp = SphBp[LL][i];

		      tmp0 = Spe_CrudeVNA_Bessel[Cwan][i];
		      tmp1 = Spe_ProductRF_Bessel[Hwan][L0][Mul0][L1][Mul1][LL][i]; 
		      tmp10 = 0.50*Dk*tmp0*tmp1*GL_Weight[i]*Normk*Normk;

		      sum  += tmp10*sj;
		      sumr += tmp10*sjp*Normk;
		    }

		    /* sum over m */

		    for (M0=-L0; M0<=L0; M0++){
		      for (M1=-L1; M1<=L1; M1++){

			Csum = Complex(0.0,0.0);
			Csumr = Complex(0.0,0.0);
			Csumt = Complex(0.0,0.0);
			Csump = Complex(0.0,0.0);

			for(m=-LL; m<=LL; m++){ 

			  if ( (M1-m)==M0){

			    ComplexSH(LL,m,theta,phi,SH,dSHt,dSHp);
			    CY  = Complex(SH[0],SH[1]);
			    CYt = Complex(dSHt[0],dSHt[1]);
			    CYp = Complex(dSHp[0],dSHp[1]);

			    gant = pow(-1.0,(double)__builtin_abs(m))*Gaunt(L0,M0,L1,M1,LL,-m);

			    /* S */

			    Ctmp2 = CRmul(CY,gant);          
			    Csum = Cadd(Csum,Ctmp2);             

			    /* dS/dt */
                      
			    Ctmp2 = CRmul(CYt,gant);          
			    Csumt = Cadd(Csumt,Ctmp2);             
                      
			    /* dS/dp */
                      
			    Ctmp2 = CRmul(CYp,gant);          
			    Csump = Cadd(Csump,Ctmp2);             
			  }

			}

			/* S */
			Ctmp1 = CRmul(Csum,sum);
			TmpHNA[L0][Mul0][L0+M0][L1][Mul1][L1+M1] 
			  = Cadd(TmpHNA[L0][Mul0][L0+M0][L1][Mul1][L1+M1],Ctmp1);

			/* dS/dr */
			Ctmp1 = CRmul(Csum,sumr);
			TmpHNAr[L0][Mul0][L0+M0][L1][Mul1][L1+M1] 
			  = Cadd(TmpHNAr[L0][Mul0][L0+M0][L1][Mul1][L1+M1],Ctmp1);

			/* dS/dt */
			Ctmp1 = CRmul(Csumt,sum);
			TmpHNAt[L0][Mul0][L0+M0][L1][Mul1][L1+M1] 
			  = Cadd(TmpHNAt[L0][Mul0][L0+M0][L1][Mul1][L1+M1],Ctmp1);

			/* dS/dp */
			Ctmp1 = CRmul(Csump,sum);
			TmpHNAp[L0][Mul0][L0+M0][L1][Mul1][L1+M1] 
			  = Cadd(TmpHNAp[L0][Mul0][L0+M0][L1][Mul1][L1+M1],Ctmp1);
		      }
		    }
		  }
		} /* LL */
	      } /* if (L0<=L1) */
	    }
	  }
	}
      }

      /* free SphB and SphBp */
      
      for(LL=0; LL<(Lmax_Four_Int+3); LL++){ 
	free(SphB[LL]);
      }
      free(SphB);

      for(LL=0; LL<(Lmax_Four_Int+3); LL++){ 
	free(SphBp[LL]);
      }
      free(SphBp);

      /* copy the upper part to the lower part */

      for (L0=0; L0<=Spe_MaxL_Basis[Hwan]; L0++){
	for (Mul0=0; Mul0<Spe_Num_Basis[Hwan][L0]; Mul0++){
	  for (L1=0; L1<=Spe_MaxL_Basis[Hwan]; L1++){
	    for (Mul1=0; Mul1<Spe_Num_Basis[Hwan][L1]; Mul1++){
	      if (L0<=L1){
		for (M0=-L0; M0<=L0; M0++){
		  for (M1=-L1; M1<=L1; M1++){

		    TmpHNA[L1][Mul1][L1+M1][L0][Mul0][L0+M0] 
		      = Conjg(TmpHNA[L0][Mul0][L0+M0][L1][Mul1][L1+M1]); 

		    TmpHNAr[L1][Mul1][L1+M1][L0][Mul0][L0+M0]
		      = Conjg(TmpHNAr[L0][Mul0][L0+M0][L1][Mul1][L1+M1]);
 
		    TmpHNAt[L1][Mul1][L1+M1][L0][Mul0][L0+M0]
		      = Conjg(TmpHNAt[L0][Mul0][L0+M0][L1][Mul1][L1+M1]); 

		    TmpHNAp[L1][Mul1][L1+M1][L0][Mul0][L0+M0]
		      = Conjg(TmpHNAp[L0][Mul0][L0+M0][L1][Mul1][L1+M1]); 

		  }
		}
	      }
	    }
	  }
	}
      }

      /****************************************************
 	                 Complex to Real
      ****************************************************/

      num0 = 0;
      for (L0=0; L0<=Spe_MaxL_Basis[Hwan]; L0++){
	for (Mul0=0; Mul0<Spe_Num_Basis[Hwan][L0]; Mul0++){
       
	  num1 = 0;
	  for (L1=0; L1<=Spe_MaxL_Basis[Hwan]; L1++){
	    for (Mul1=0; Mul1<Spe_Num_Basis[Hwan][L1]; Mul1++){

	      for (M0=-L0; M0<=L0; M0++){
		for (M1=-L1; M1<=L1; M1++){

		  CsumS0 = Complex(0.0,0.0);
		  CsumSr = Complex(0.0,0.0);
		  CsumSt = Complex(0.0,0.0);
		  CsumSp = Complex(0.0,0.0);

		  for (k=-L0; k<=L0; k++){

		    Ctmp1 = Conjg(Comp2Real[L0][L0+M0][L0+k]);

		    /* S */

		    Ctmp0 = TmpHNA[L0][Mul0][L0+k][L1][Mul1][L1+M1];
		    Ctmp2 = Cmul(Ctmp1,Ctmp0);
		    CsumS0 = Cadd(CsumS0,Ctmp2);

		    /* dS/dr */

		    Ctmp0 = TmpHNAr[L0][Mul0][L0+k][L1][Mul1][L1+M1];
		    Ctmp2 = Cmul(Ctmp1,Ctmp0);
		    CsumSr = Cadd(CsumSr,Ctmp2);

		    /* dS/dt */

		    Ctmp0 = TmpHNAt[L0][Mul0][L0+k][L1][Mul1][L1+M1];
		    Ctmp2 = Cmul(Ctmp1,Ctmp0);
		    CsumSt = Cadd(CsumSt,Ctmp2);

		    /* dS/dp */

		    Ctmp0 = TmpHNAp[L0][Mul0][L0+k][L1][Mul1][L1+M1];
		    Ctmp2 = Cmul(Ctmp1,Ctmp0);
		    CsumSp = Cadd(CsumSp,Ctmp2);

		  }

		  CmatS0[L0+M0][L1+M1] = CsumS0;
		  CmatSr[L0+M0][L1+M1] = CsumSr;
		  CmatSt[L0+M0][L1+M1] = CsumSt;
		  CmatSp[L0+M0][L1+M1] = CsumSp;

		}
	      }

	      for (M0=-L0; M0<=L0; M0++){
		for (M1=-L1; M1<=L1; M1++){

		  CsumS0 = Complex(0.0,0.0);
		  CsumSr = Complex(0.0,0.0);
		  CsumSt = Complex(0.0,0.0);
		  CsumSp = Complex(0.0,0.0);

		  for (k=-L1; k<=L1; k++){

		    /* S */ 


		    Ctmp1 = Cmul(CmatS0[L0+M0][L1+k],Comp2Real[L1][L1+M1][L1+k]);
		    CsumS0 = Cadd(CsumS0,Ctmp1);

		    /* dS/dr */ 

		    Ctmp1 = Cmul(CmatSr[L0+M0][L1+k],Comp2Real[L1][L1+M1][L1+k]);
		    CsumSr = Cadd(CsumSr,Ctmp1);

		    /* dS/dt */ 

		    Ctmp1 = Cmul(CmatSt[L0+M0][L1+k],Comp2Real[L1][L1+M1][L1+k]);
		    CsumSt = Cadd(CsumSt,Ctmp1);

		    /* dS/dp */ 

		    Ctmp1 = Cmul(CmatSp[L0+M0][L1+k],Comp2Real[L1][L1+M1][L1+k]);
		    CsumSp = Cadd(CsumSp,Ctmp1);

		  }

		  HVNA3[0][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] = 8.0*CsumS0.r;

		  if (h_AN!=0){

		    if (fabs(siT)<10e-14){

		      HVNA3[1][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] =
			-8.0*(siT*coP*CsumSr.r + coT*coP/r*CsumSt.r);

		      HVNA3[2][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] =
			-8.0*(siT*siP*CsumSr.r + coT*siP/r*CsumSt.r);

		      HVNA3[3][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] =
			-8.0*(coT*CsumSr.r - siT/r*CsumSt.r);
		    }

		    else{

		      HVNA3[1][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] =
			-8.0*(siT*coP*CsumSr.r + coT*coP/r*CsumSt.r
			     - siP/siT/r*CsumSp.r);

		      HVNA3[2][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] =
			-8.0*(siT*siP*CsumSr.r + coT*siP/r*CsumSt.r
			     + coP/siT/r*CsumSp.r);

		      HVNA3[3][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] =
			-8.0*(coT*CsumSr.r - siT/r*CsumSt.r);
		    }
		  }
		  else{
		    HVNA3[1][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] = 0.0;
		    HVNA3[2][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] = 0.0;
		    HVNA3[3][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] = 0.0;
		  }

		}
	      }

	      num1 = num1 + 2*L1 + 1; 
	    }
	  }

	  num0 = num0 + 2*L0 + 1; 
	}
      }

      dtime(&Etime_atom);
      time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

    } /* Mc_AN */

    /* freeing of arrays */

    for (i=0; i<(List_YOUSO[25]+1); i++){
      for (j=0; j<List_YOUSO[24]; j++){
	for (k=0; k<(2*(List_YOUSO[25]+1)+1); k++){
	  for (l=0; l<(List_YOUSO[25]+1); l++){
	    for (m=0; m<List_YOUSO[24]; m++){
	      free(TmpHNA[i][j][k][l][m]);
	    }
	    free(TmpHNA[i][j][k][l]);
	  }
	  free(TmpHNA[i][j][k]);
	}
	free(TmpHNA[i][j]);
      }
      free(TmpHNA[i]);
    }
    free(TmpHNA);

    for (i=0; i<(List_YOUSO[25]+1); i++){
      for (j=0; j<List_YOUSO[24]; j++){
	for (k=0; k<(2*(List_YOUSO[25]+1)+1); k++){
	  for (l=0; l<(List_YOUSO[25]+1); l++){
	    for (m=0; m<List_YOUSO[24]; m++){
	      free(TmpHNAr[i][j][k][l][m]);
	    }
	    free(TmpHNAr[i][j][k][l]);
	  }
	  free(TmpHNAr[i][j][k]);
	}
	free(TmpHNAr[i][j]);
      }
      free(TmpHNAr[i]);
    }
    free(TmpHNAr);

    for (i=0; i<(List_YOUSO[25]+1); i++){
      for (j=0; j<List_YOUSO[24]; j++){
	for (k=0; k<(2*(List_YOUSO[25]+1)+1); k++){
	  for (l=0; l<(List_YOUSO[25]+1); l++){
	    for (m=0; m<List_YOUSO[24]; m++){
	      free(TmpHNAt[i][j][k][l][m]);
	    }
	    free(TmpHNAt[i][j][k][l]);
	  }
	  free(TmpHNAt[i][j][k]);
	}
	free(TmpHNAt[i][j]);
      }
      free(TmpHNAt[i]);
    }
    free(TmpHNAt);

    for (i=0; i<(List_YOUSO[25]+1); i++){
      for (j=0; j<List_YOUSO[24]; j++){
	for (k=0; k<(2*(List_YOUSO[25]+1)+1); k++){
	  for (l=0; l<(List_YOUSO[25]+1); l++){
	    for (m=0; m<List_YOUSO[24]; m++){
	      free(TmpHNAp[i][j][k][l][m]);
	    }
	    free(TmpHNAp[i][j][k][l]);
	  }
	  free(TmpHNAp[i][j][k]);
	}
	free(TmpHNAp[i][j]);
      }
      free(TmpHNAp[i]);
    }
    free(TmpHNAp);

    for (i=0; i<(2*(List_YOUSO[25]+1)+1); i++){
      free(CmatS0[i]);
    }
    free(CmatS0);

    for (i=0; i<(2*(List_YOUSO[25]+1)+1); i++){
      free(CmatSr[i]);
    }
    free(CmatSr);

    for (i=0; i<(2*(List_YOUSO[25]+1)+1); i++){
      free(CmatSt[i]);
    }
    free(CmatSt);

    for (i=0; i<(2*(List_YOUSO[25]+1)+1); i++){
      free(CmatSp[i]);
    }
    free(CmatSp);

#pragma omp flush(HVNA3)

  } /* #pragma omp parallel */

  } /* if (!SetPro_gpu3.enabled) */

  /****************************************************
  freeing of arrays:
  ****************************************************/

  free(OneD2Mc_AN);
  free(OneD2h_AN);

  /* for time */
  dtime(&TEtime);
  time0 = TEtime - TStime;
  return time0;
} 


#define xmin  0.0
#define asize_lmax 30

#ifdef kcomp
static void Spherical_Bessel2( double x, int lmax, double *sb, double *dsb )
#else 
static inline void Spherical_Bessel2( double x, int lmax, double *sb, double *dsb )
#endif      
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

    for ( n=0; n<=lmax; n++ ){
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

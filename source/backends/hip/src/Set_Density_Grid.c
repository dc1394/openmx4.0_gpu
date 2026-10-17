/**********************************************************************
  Set_Density_Grid.c:

     Set_Density_Grid.c is a subroutine to calculate a charge density 
     on grid by one-particle wave functions.

  Log of Set_Density_Grid.c:

     22/Nov/2001  Released by T. Ozaki
     19/Apr/2013  Modified by A.M. Ito     

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "openmx_common.h"
#include "openmx_gpu_workers.h"
#include "mpi.h"
#include <omp.h>

#define  measure_time              0
#define  SDG_NC_GRID_BATCH_AVX2    4
#define  SDG_NC_GRID_BATCH_AVX512  8
#define  SDG_NC_GRID_BATCH_MAX     SDG_NC_GRID_BATCH_AVX512

enum {
  SDG_NC_SIMD_SCALAR = 0,
  SDG_NC_SIMD_AVX2 = 1,
  SDG_NC_SIMD_AVX512 = 2
};

#if defined(__x86_64__) && (defined(__clang__) || defined(__GNUC__))
#define SDG_X86_RUNTIME_DISPATCH 1
#define SDG_TARGET_AVX2 \
  __attribute__((target("avx2"),noinline))
#define SDG_TARGET_AVX512 \
  __attribute__((target("avx512f"),noinline))
#else
#define SDG_X86_RUNTIME_DISPATCH 0
#define SDG_TARGET_AVX2
#define SDG_TARGET_AVX512
#endif

static size_t SDG_checked_add(size_t a, size_t b, const char *name, int myid)
{
  if (a>SIZE_MAX-b){
    fprintf(stderr,"Set_Density_Grid: rank %d size overflow for %s\n",myid,name);
    fflush(stderr);
    MPI_Abort(mpi_comm_level1,1);
    abort();
  }
  return a+b;
}

static size_t SDG_checked_mul(size_t a, size_t b, const char *name, int myid)
{
  if (a!=0 && SIZE_MAX/a<b){
    fprintf(stderr,"Set_Density_Grid: rank %d size overflow for %s\n",myid,name);
    fflush(stderr);
    MPI_Abort(mpi_comm_level1,1);
    abort();
  }
  return a*b;
}

static void *SDG_checked_alloc(size_t count, size_t element_size, int clear,
                               const char *name, int myid)
{
  void *p;

  if (count==0) count = 1;
  (void)SDG_checked_mul(count,element_size,name,myid);
  p = clear ? calloc(count,element_size) : malloc(count*element_size);
  if (p==NULL){
    fprintf(stderr,"Set_Density_Grid: rank %d cannot allocate %s (%zu bytes)\n",
            myid,name,count*element_size);
    fflush(stderr);
    MPI_Abort(mpi_comm_level1,1);
    abort();
  }
  return p;
}

static void *SDG_checked_aligned_alloc(size_t alignment, size_t count,
                                       size_t element_size, const char *name,
                                       int myid)
{
  const size_t bytes = SDG_checked_mul(count ? count : 1,element_size,name,myid);
  const size_t padded = SDG_checked_add(bytes,alignment-1,name,myid) & ~(alignment-1);
  void *p = aligned_alloc(alignment,padded);

  if (p==NULL){
    fprintf(stderr,"Set_Density_Grid: rank %d cannot allocate %s (%zu bytes)\n",
            myid,name,padded);
    fflush(stderr);
    MPI_Abort(mpi_comm_level1,1);
    abort();
  }
  return p;
}

static int SDG_env_disabled(const char *value)
{
  if (value==NULL || value[0]=='\0') return 0;
  if (value[0]=='0' || value[0]=='n' || value[0]=='N' ||
      value[0]=='f' || value[0]=='F' ||
      strcmp(value,"scalar")==0 || strcmp(value,"SCALAR")==0) return 1;
  return 0;
}

/* Select only instruction sets which both the processor and the operating
   system can execute.  __builtin_cpu_supports includes the OSXSAVE/XCR0
   checks, so an AVX-512 capable CPU whose vector state is disabled safely
   falls back to AVX2.  OPENMX_DENSITY_GRID_NC_SIMD=avx2 caps the selection
   for validation; 0/scalar selects the original scalar-compatible path. */
static int SDG_nc_simd_mode(void)
{
  const char *value = getenv("OPENMX_DENSITY_GRID_NC_SIMD");
  int allow_avx512 = 1;

  if (SDG_env_disabled(value)) return SDG_NC_SIMD_SCALAR;
  if (value!=NULL &&
      (strcmp(value,"avx2")==0 || strcmp(value,"AVX2")==0)){
    allow_avx512 = 0;
  }

#if SDG_X86_RUNTIME_DISPATCH
  __builtin_cpu_init();
  if (allow_avx512 && __builtin_cpu_supports("avx512f")){
    return SDG_NC_SIMD_AVX512;
  }
  if (__builtin_cpu_supports("avx2")) return SDG_NC_SIMD_AVX2;
#endif

  return SDG_NC_SIMD_SCALAR;
}

/* Keep the collinear policy independent of the NC switch so either path can
   be capped or disabled during A/B validation.  With no override, select the
   widest ISA which is executable by both the processor and the OS. */
static int SDG_col_simd_mode(void)
{
  const char *value = getenv("OPENMX_DENSITY_GRID_COL_SIMD");
  int allow_avx512 = 1;

  if (SDG_env_disabled(value)) return SDG_NC_SIMD_SCALAR;
  if (value!=NULL &&
      (strcmp(value,"avx2")==0 || strcmp(value,"AVX2")==0)){
    allow_avx512 = 0;
  }

#if SDG_X86_RUNTIME_DISPATCH
  __builtin_cpu_init();
  if (allow_avx512 && __builtin_cpu_supports("avx512f")){
    return SDG_NC_SIMD_AVX512;
  }
  if (__builtin_cpu_supports("avx2")) return SDG_NC_SIMD_AVX2;
#endif

  return SDG_NC_SIMD_SCALAR;
}

static const char *SDG_nc_simd_mode_name(int mode)
{
  if (mode==SDG_NC_SIMD_AVX512) return "AVX-512";
  if (mode==SDG_NC_SIMD_AVX2) return "AVX2";
  return "scalar";
}

/* Collinear contractions use the same grid-lane layout as the NC kernels,
   but carry only one or two density components.  The target attributes keep
   AVX-512 instructions out of both the dispatcher and the AVX2 fallback. */
static SDG_TARGET_AVX512 void SDG_accumulate_col8(
                                      int no0, int no1, int spin_count,
                                      const double *restrict dm,
                                      const double *restrict orb0,
                                      const double *restrict orb1,
                                      const int *restrict nc,
                                      double *restrict den0,
                                      double *restrict den1)
{
  double sum0[SDG_NC_GRID_BATCH_AVX512] = {0.0};
  double sum1[SDG_NC_GRID_BATCH_AVX512] = {0.0};
  int i,j,g;

  if (spin_count==1){
    for (i=0; i<no0; i++){
      double tmp0[SDG_NC_GRID_BATCH_AVX512] = {0.0};
      const double *restrict dm_row = dm + (size_t)i*no1;
      const double *restrict orb0_row = orb0 + (size_t)i*SDG_NC_GRID_BATCH_AVX512;

      for (j=0; j<no1; j++){
        const double *restrict orb1_row = orb1 + (size_t)j*SDG_NC_GRID_BATCH_AVX512;
        const double cdm0 = dm_row[j];

#pragma omp simd aligned(orb1_row:64) simdlen(8)
        for (g=0; g<SDG_NC_GRID_BATCH_AVX512; g++){
          tmp0[g] += orb1_row[g]*cdm0;
        }
      }

#pragma omp simd aligned(orb0_row:64) simdlen(8)
      for (g=0; g<SDG_NC_GRID_BATCH_AVX512; g++){
        sum0[g] += orb0_row[g]*tmp0[g];
      }
    }

    for (g=0; g<SDG_NC_GRID_BATCH_AVX512; g++){
      den0[nc[g]] += sum0[g];
    }
  }
  else{
    for (i=0; i<no0; i++){
      double tmp0[SDG_NC_GRID_BATCH_AVX512] = {0.0};
      double tmp1[SDG_NC_GRID_BATCH_AVX512] = {0.0};
      const double *restrict dm_row = dm + (size_t)2*i*no1;
      const double *restrict orb0_row = orb0 + (size_t)i*SDG_NC_GRID_BATCH_AVX512;

      for (j=0; j<no1; j++){
        const double *restrict orb1_row = orb1 + (size_t)j*SDG_NC_GRID_BATCH_AVX512;
        const double cdm0 = dm_row[2*j+0];
        const double cdm1 = dm_row[2*j+1];

#pragma omp simd aligned(orb1_row:64) simdlen(8)
        for (g=0; g<SDG_NC_GRID_BATCH_AVX512; g++){
          const double orb = orb1_row[g];
          tmp0[g] += orb*cdm0;
          tmp1[g] += orb*cdm1;
        }
      }

#pragma omp simd aligned(orb0_row:64) simdlen(8)
      for (g=0; g<SDG_NC_GRID_BATCH_AVX512; g++){
        const double orb = orb0_row[g];
        sum0[g] += orb*tmp0[g];
        sum1[g] += orb*tmp1[g];
      }
    }

    for (g=0; g<SDG_NC_GRID_BATCH_AVX512; g++){
      const int n = nc[g];
      den0[n] += sum0[g];
      den1[n] += sum1[g];
    }
  }
}

static SDG_TARGET_AVX2 void SDG_accumulate_col4(
                                      int no0, int no1, int spin_count,
                                      const double *restrict dm,
                                      const double *restrict orb0,
                                      const double *restrict orb1,
                                      const int *restrict nc,
                                      double *restrict den0,
                                      double *restrict den1)
{
  double sum0[SDG_NC_GRID_BATCH_AVX2] = {0.0};
  double sum1[SDG_NC_GRID_BATCH_AVX2] = {0.0};
  int i,j,g;

  if (spin_count==1){
    for (i=0; i<no0; i++){
      double tmp0[SDG_NC_GRID_BATCH_AVX2] = {0.0};
      const double *restrict dm_row = dm + (size_t)i*no1;
      const double *restrict orb0_row = orb0 + (size_t)i*SDG_NC_GRID_BATCH_AVX2;

      for (j=0; j<no1; j++){
        const double *restrict orb1_row = orb1 + (size_t)j*SDG_NC_GRID_BATCH_AVX2;
        const double cdm0 = dm_row[j];

#pragma omp simd aligned(orb1_row:32) simdlen(4)
        for (g=0; g<SDG_NC_GRID_BATCH_AVX2; g++){
          tmp0[g] += orb1_row[g]*cdm0;
        }
      }

#pragma omp simd aligned(orb0_row:32) simdlen(4)
      for (g=0; g<SDG_NC_GRID_BATCH_AVX2; g++){
        sum0[g] += orb0_row[g]*tmp0[g];
      }
    }

    for (g=0; g<SDG_NC_GRID_BATCH_AVX2; g++){
      den0[nc[g]] += sum0[g];
    }
  }
  else{
    for (i=0; i<no0; i++){
      double tmp0[SDG_NC_GRID_BATCH_AVX2] = {0.0};
      double tmp1[SDG_NC_GRID_BATCH_AVX2] = {0.0};
      const double *restrict dm_row = dm + (size_t)2*i*no1;
      const double *restrict orb0_row = orb0 + (size_t)i*SDG_NC_GRID_BATCH_AVX2;

      for (j=0; j<no1; j++){
        const double *restrict orb1_row = orb1 + (size_t)j*SDG_NC_GRID_BATCH_AVX2;
        const double cdm0 = dm_row[2*j+0];
        const double cdm1 = dm_row[2*j+1];

#pragma omp simd aligned(orb1_row:32) simdlen(4)
        for (g=0; g<SDG_NC_GRID_BATCH_AVX2; g++){
          const double orb = orb1_row[g];
          tmp0[g] += orb*cdm0;
          tmp1[g] += orb*cdm1;
        }
      }

#pragma omp simd aligned(orb0_row:32) simdlen(4)
      for (g=0; g<SDG_NC_GRID_BATCH_AVX2; g++){
        const double orb = orb0_row[g];
        sum0[g] += orb*tmp0[g];
        sum1[g] += orb*tmp1[g];
      }
    }

    for (g=0; g<SDG_NC_GRID_BATCH_AVX2; g++){
      const int n = nc[g];
      den0[n] += sum0[g];
      den1[n] += sum1[g];
    }
  }
}

static inline void SDG_accumulate_col1(int no0, int no1, int spin_count,
                                      const double *restrict dm,
                                      const double *restrict orb0,
                                      const double *restrict orb1,
                                      int nc, double *restrict den0,
                                      double *restrict den1)
{
  double sum0 = 0.0;
  double sum1 = 0.0;
  int i,j;

  if (spin_count==1){
    for (i=0; i<no0; i++){
      const double *restrict dm_row = dm + (size_t)i*no1;
      double tmp0 = 0.0;
      for (j=0; j<no1; j++) tmp0 += orb1[j]*dm_row[j];
      sum0 += orb0[i]*tmp0;
    }
    den0[nc] += sum0;
  }
  else{
    for (i=0; i<no0; i++){
      const double *restrict dm_row = dm + (size_t)2*i*no1;
      double tmp0 = 0.0, tmp1 = 0.0;
      for (j=0; j<no1; j++){
        const double orb = orb1[j];
        tmp0 += orb*dm_row[2*j+0];
        tmp1 += orb*dm_row[2*j+1];
      }
      sum0 += orb0[i]*tmp0;
      sum1 += orb0[i]*tmp1;
    }
    den0[nc] += sum0;
    den1[nc] += sum1;
  }
}

/* Evaluate eight independent non-collinear grid contractions together.  The
   grid point is the SIMD dimension: the j and i accumulation order of every
   individual density value is therefore identical to the scalar formula.
   dm is packed as [i][j][spin], while the orbital batches are [orbital][grid]. */
static SDG_TARGET_AVX512 void SDG_accumulate_nc8(
                                      int no0, int no1,
                                      const double *restrict dm,
                                      const double *restrict orb0,
                                      const double *restrict orb1,
                                      const int *restrict nc,
                                      double *restrict den0,
                                      double *restrict den1,
                                      double *restrict den2,
                                      double *restrict den3)
{
  double sum0[SDG_NC_GRID_BATCH_AVX512] = {0.0};
  double sum1[SDG_NC_GRID_BATCH_AVX512] = {0.0};
  double sum2[SDG_NC_GRID_BATCH_AVX512] = {0.0};
  double sum3[SDG_NC_GRID_BATCH_AVX512] = {0.0};
  int i,j,g;

  for (i=0; i<no0; i++){
    double tmp0[SDG_NC_GRID_BATCH_AVX512] = {0.0};
    double tmp1[SDG_NC_GRID_BATCH_AVX512] = {0.0};
    double tmp2[SDG_NC_GRID_BATCH_AVX512] = {0.0};
    double tmp3[SDG_NC_GRID_BATCH_AVX512] = {0.0};
    const double *restrict dm_row = dm + (size_t)4*i*no1;
    const double *restrict orb0_row = orb0 + (size_t)i*SDG_NC_GRID_BATCH_AVX512;

    for (j=0; j<no1; j++){
      const double *restrict orb1_row = orb1 + (size_t)j*SDG_NC_GRID_BATCH_AVX512;
      const double cdm0 = dm_row[4*j+0];
      const double cdm1 = dm_row[4*j+1];
      const double cdm2 = dm_row[4*j+2];
      const double cdm3 = dm_row[4*j+3];

#pragma omp simd aligned(orb1_row:64) simdlen(8)
      for (g=0; g<SDG_NC_GRID_BATCH_AVX512; g++){
        const double orb = orb1_row[g];
        tmp0[g] += orb*cdm0;
        tmp1[g] += orb*cdm1;
        tmp2[g] += orb*cdm2;
        tmp3[g] += orb*cdm3;
      }
    }

#pragma omp simd aligned(orb0_row:64) simdlen(8)
    for (g=0; g<SDG_NC_GRID_BATCH_AVX512; g++){
      const double orb = orb0_row[g];
      sum0[g] += orb*tmp0[g];
      sum1[g] += orb*tmp1[g];
      sum2[g] += orb*tmp2[g];
      sum3[g] += orb*tmp3[g];
    }
  }

  /* Keep updates in ascending Nog order in case an unusual grid list contains
     the same local grid index more than once. */
  for (g=0; g<SDG_NC_GRID_BATCH_AVX512; g++){
    const int n = nc[g];
    den0[n] += sum0[g];
    den1[n] += sum1[g];
    den2[n] += sum2[g];
    den3[n] += sum3[g];
  }
}

/* Four grid points fill one AVX2 vector.  This function is compiled for AVX2
   independently of the AVX-512 variant and is called only after the runtime
   feature test above succeeds. */
static SDG_TARGET_AVX2 void SDG_accumulate_nc4(
                                      int no0, int no1,
                                      const double *restrict dm,
                                      const double *restrict orb0,
                                      const double *restrict orb1,
                                      const int *restrict nc,
                                      double *restrict den0,
                                      double *restrict den1,
                                      double *restrict den2,
                                      double *restrict den3)
{
  double sum0[SDG_NC_GRID_BATCH_AVX2] = {0.0};
  double sum1[SDG_NC_GRID_BATCH_AVX2] = {0.0};
  double sum2[SDG_NC_GRID_BATCH_AVX2] = {0.0};
  double sum3[SDG_NC_GRID_BATCH_AVX2] = {0.0};
  int i,j,g;

  for (i=0; i<no0; i++){
    double tmp0[SDG_NC_GRID_BATCH_AVX2] = {0.0};
    double tmp1[SDG_NC_GRID_BATCH_AVX2] = {0.0};
    double tmp2[SDG_NC_GRID_BATCH_AVX2] = {0.0};
    double tmp3[SDG_NC_GRID_BATCH_AVX2] = {0.0};
    const double *restrict dm_row = dm + (size_t)4*i*no1;
    const double *restrict orb0_row = orb0 + (size_t)i*SDG_NC_GRID_BATCH_AVX2;

    for (j=0; j<no1; j++){
      const double *restrict orb1_row = orb1 + (size_t)j*SDG_NC_GRID_BATCH_AVX2;
      const double cdm0 = dm_row[4*j+0];
      const double cdm1 = dm_row[4*j+1];
      const double cdm2 = dm_row[4*j+2];
      const double cdm3 = dm_row[4*j+3];

#pragma omp simd aligned(orb1_row:32) simdlen(4)
      for (g=0; g<SDG_NC_GRID_BATCH_AVX2; g++){
        const double orb = orb1_row[g];
        tmp0[g] += orb*cdm0;
        tmp1[g] += orb*cdm1;
        tmp2[g] += orb*cdm2;
        tmp3[g] += orb*cdm3;
      }
    }

#pragma omp simd aligned(orb0_row:32) simdlen(4)
    for (g=0; g<SDG_NC_GRID_BATCH_AVX2; g++){
      const double orb = orb0_row[g];
      sum0[g] += orb*tmp0[g];
      sum1[g] += orb*tmp1[g];
      sum2[g] += orb*tmp2[g];
      sum3[g] += orb*tmp3[g];
    }
  }

  for (g=0; g<SDG_NC_GRID_BATCH_AVX2; g++){
    const int n = nc[g];
    den0[n] += sum0[g];
    den1[n] += sum1[g];
    den2[n] += sum2[g];
    den3[n] += sum3[g];
  }
}

static inline void SDG_accumulate_nc1(int no0, int no1,
                                      const double *restrict dm,
                                      const double *restrict orb0,
                                      const double *restrict orb1,
                                      int nc,
                                      double *restrict den0,
                                      double *restrict den1,
                                      double *restrict den2,
                                      double *restrict den3)
{
  double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
  int i,j;

  for (i=0; i<no0; i++){
    const double *restrict dm_row = dm + (size_t)4*i*no1;
    double tmp0 = 0.0, tmp1 = 0.0, tmp2 = 0.0, tmp3 = 0.0;

    for (j=0; j<no1; j++){
      const double orb = orb1[j];
      tmp0 += orb*dm_row[4*j+0];
      tmp1 += orb*dm_row[4*j+1];
      tmp2 += orb*dm_row[4*j+2];
      tmp3 += orb*dm_row[4*j+3];
    }

    sum0 += orb0[i]*tmp0;
    sum1 += orb0[i]*tmp1;
    sum2 += orb0[i]*tmp2;
    sum3 += orb0[i]*tmp3;
  }

  den0[nc] += sum0;
  den1[nc] += sum1;
  den2[nc] += sum2;
  den3[nc] += sum3;
}



int Set_Density_Grid_GPU_Local_Prepare(int Cnt_kind, int Calc_CntOrbital_ON);
int Set_Density_Grid_GPU_Local_Run(double *****CDM, double ***Tmp_Den_Grid);
int Set_Density_Grid_GPU_Local_OnTheFly(void);
int Set_Density_Grid_GPU_Capacity(int,int,double *****,double ***);
void Set_Density_Grid_GPU_Local_Report(int path, int failed);
int Set_Density_Grid_OTF_Prepare(int Cnt_kind, size_t *need_bytes, size_t *available_bytes);
int openmx_grid_precision_df(void);

double Set_Density_Grid(int Cnt_kind, int Calc_CntOrbital_ON, double *****CDM, double **Density_Grid_B0)
{
  static int firsttime=1;
  int al,L0,Mul0,M0,p,size1,size2;
  int Gc_AN,Mc_AN,Mh_AN,LN,AN,BN,CN;
  int n1,n2,n3,k1,k2,k3,N3[4];
  int Cwan,NO0,NO1,Rn,N,Hwan,i,j,k,n;
  int NN_S,NN_R;
  unsigned long long int N2D,n2D,GN; 
  int Max_Size,My_Max;
  size_t size_Tmp_Den_Grid;
  size_t size_Den_Snd_Grid_A2B;
  size_t size_Den_Rcv_Grid_A2B;
  int h_AN,Gh_AN,Rnh,spin,Nc,GRc,Nh,Nog;
  int Nc_0,Nc_1,Nc_2,Nc_3,Nh_0,Nh_1,Nh_2,Nh_3;

  double threshold;
  double tmp0,tmp1,sk1,sk2,sk3,tot_den,sum;
  double tmp0_0,tmp0_1,tmp0_2,tmp0_3;
  double sum_0,sum_1,sum_2,sum_3;
  double d1,d2,d3,cop,sip,sit,cot;
  double x,y,z,Cxyz[4];
  double TStime,TEtime;
  double ***Tmp_Den_Grid;
  double **Tmp_Den_Grid_storage;
  double **Den_Snd_Grid_A2B;
  double **Den_Rcv_Grid_A2B;
  double *Den_Snd_Grid_A2B_storage;
  double *Den_Rcv_Grid_A2B_storage;
  double *tmp_array;
  double *tmp_array2;
  double *orbs0,*orbs1;
  double *orbs0_0,*orbs0_1,*orbs0_2,*orbs0_3;
  double *orbs1_0,*orbs1_1,*orbs1_2,*orbs1_3;
  double ***tmp_CDM;
  int *Snd_Size,*Rcv_Size;
  int numprocs,myid,tag=999,ID,IDS,IDR;
  double Stime_atom, Etime_atom;
  double time0,time1,time2;
  int use_local_gpu = 0;
  int use_otf_gpu = 0;
  int local_ok = 0, local_any = 0, local_path = 0, local_failed = 0, failed_ranks = 0;
  MPI_Request failed_request = MPI_REQUEST_NULL;
  double sd_t0,sd_t1,sd_pre=0.0,sd_quad=0.0,sd_a2b=0.0,sd_sup=0.0,sd_b2d=0.0;
  static int sd_prof = -1;
  const char *sd_mode = "cpu";
  int use_nc_simd;
  int nc_simd_mode;
  int nc_grid_batch;
  int use_col_simd;
  int col_simd_mode;
  int col_grid_batch;

  MPI_Status stat;
  MPI_Request request;
  MPI_Status *stat_send;
  MPI_Status *stat_recv;
  MPI_Request *request_send;
  MPI_Request *request_recv;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);
  nc_simd_mode = SpinP_switch==3 ? SDG_nc_simd_mode() : SDG_NC_SIMD_SCALAR;
  use_nc_simd = (nc_simd_mode!=SDG_NC_SIMD_SCALAR);
  nc_grid_batch = nc_simd_mode==SDG_NC_SIMD_AVX512 ?
                    SDG_NC_GRID_BATCH_AVX512 : SDG_NC_GRID_BATCH_AVX2;
  col_simd_mode = (SpinP_switch==0 || SpinP_switch==1) ?
                    SDG_col_simd_mode() : SDG_NC_SIMD_SCALAR;
  use_col_simd = (col_simd_mode!=SDG_NC_SIMD_SCALAR);
  col_grid_batch = col_simd_mode==SDG_NC_SIMD_AVX512 ?
                     SDG_NC_GRID_BATCH_AVX512 : SDG_NC_GRID_BATCH_AVX2;
  if (myid==0 && SDG_env_disabled(getenv("OPENMX_DENSITY_GRID_NC_SIMD_TRACE"))==0 &&
      getenv("OPENMX_DENSITY_GRID_NC_SIMD_TRACE")!=NULL){
    static int trace_printed = 0;
    if (!trace_printed){
      printf("Set_Density_Grid: NC SIMD backend = %s\n",
             SDG_nc_simd_mode_name(nc_simd_mode));
      fflush(stdout);
      trace_printed = 1;
    }
  }
  if (myid==0 && (SpinP_switch==0 || SpinP_switch==1) &&
      SDG_env_disabled(getenv("OPENMX_DENSITY_GRID_COL_SIMD_TRACE"))==0 &&
      getenv("OPENMX_DENSITY_GRID_COL_SIMD_TRACE")!=NULL){
    static int trace_printed = 0;
    if (!trace_printed){
      printf("Set_Density_Grid: collinear SIMD backend = %s\n",
             SDG_nc_simd_mode_name(col_simd_mode));
      fflush(stdout);
      trace_printed = 1;
    }
  }
  
  dtime(&TStime);

  /* stage profile: OPENMX_BAND_PROFILE=1 prints one SETDENPROF line per rank
     and call (pre = GPU-mode decision + allocation, quad = the atom-sphere
     quadrature, a2b = the atom-to-B communication, sup = the superposition
     into the B partition, b2d = the B-to-D copy and the rest; mode =
     gpu-local for the CSR kernel, otf for the on-the-fly tiles, cpu for
     the CPU loop).  The single-owner service returns before it. */
  sd_t0 = TStime;
  if (sd_prof < 0){
    const char *e = getenv("OPENMX_BAND_PROFILE");
    sd_prof = (e != NULL && atoi(e) != 0) ? 1 : 0;
  }

  /* Preferred GPU mode: every rank integrates its own atoms on the device
     -- the CSR kernel over its resident Set_Hamiltonian tables, or the
     on-the-fly orbital tiles where it has none -- and the ordinary A-to-B
     communication below distributes the result.  The all-rank on-the-fly
     mode, the single-owner GPU service and the CPU/OpenMP loop remain as
     fallbacks, in that order. */
  {
    int flags[2], reduced[2];
    local_ok=Set_Density_Grid_GPU_Local_Prepare(Cnt_kind,Calc_CntOrbital_ON);
    if ((openmx_gpu_capacity_test_recovery() || openmx_gpu_workers_requested())) local_ok=0;
    flags[0] = local_ok;
    flags[1] = -local_ok;
    MPI_Allreduce(flags,reduced,2,MPI_INT,MPI_MIN,mpi_comm_level1);
    use_local_gpu = reduced[0];
    local_any = -reduced[1];

    /* the first time on: every rank, as use_local_gpu is the agreed value */
    if (use_local_gpu){
      static int announced = 0;
      if (!announced){
        int kinds[2], kinds_sum[2] = {0,0};
        kinds[0] = (0<Matomnum && !Set_Density_Grid_GPU_Local_OnTheFly()) ? 1 : 0;
        kinds[1] = (0<Matomnum && Set_Density_Grid_GPU_Local_OnTheFly()) ? 1 : 0;
        MPI_Reduce(kinds,kinds_sum,2,MPI_INT,MPI_SUM,Host_ID,mpi_comm_level1);
        if (myid==Host_ID && 2 <= level_stdout){
          printf("<Set_Density_Grid> distributed GPU quadrature: every rank integrates its own atoms "
                 "(%d rank(s) on the resident tables, %s kernel; %d with on-the-fly tiles)\n",
                 kinds_sum[0], openmx_grid_precision_df() ? "double-float" : "FP64", kinds_sum[1]);
          fflush(stdout);
        }
        announced = 1;
      }
    }
  }

  /* the orbitals evaluated on the device batch by batch on every rank
     (Set_Hamiltonian.c) rather than the whole density on one owner */
  if (!use_local_gpu){
    size_t need, available;
    int otf_ok=Set_Density_Grid_OTF_Prepare(Cnt_kind,&need,&available);
    MPI_Allreduce(&otf_ok,&use_otf_gpu,1,MPI_INT,MPI_MIN,mpi_comm_level1);
  }

  if (!use_local_gpu && !use_otf_gpu && !(openmx_gpu_capacity_test_recovery() || openmx_gpu_workers_requested())){
    if (Set_Density_Grid_GPU_Service(Cnt_kind,Calc_CntOrbital_ON,CDM,Density_Grid_B0,&time0)){
      if (local_any) Set_Density_Grid_GPU_Local_Report(1,0);
      return time0;
    }
    /* Keep capable ranks on their local GPU path when neither all-rank
       route nor the single-owner service is available. */
    use_local_gpu = local_ok;
    local_path = 2;
  }

  /* allocation of arrays */

  size_Tmp_Den_Grid = 0;
  Tmp_Den_Grid = (double***)SDG_checked_alloc((size_t)(SpinP_switch+1),sizeof(double**),0,
                                               "Tmp_Den_Grid pointers",myid);
  Tmp_Den_Grid_storage = (double**)SDG_checked_alloc((size_t)(SpinP_switch+1),sizeof(double*),0,
                                                      "Tmp_Den_Grid storage pointers",myid);
  for (i=0; i<(SpinP_switch+1); i++){
    size_t atom_grid_count = 1;
    size_t atom_grid_offset = 1;

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = F_M2G[Mc_AN];
      atom_grid_count = SDG_checked_add(atom_grid_count,(size_t)GridN_Atom[Gc_AN],
                                        "Tmp_Den_Grid",myid);
    }
    Tmp_Den_Grid[i] = (double**)SDG_checked_alloc((size_t)(Matomnum+1),sizeof(double*),0,
                                                   "Tmp_Den_Grid atom pointers",myid);
    Tmp_Den_Grid_storage[i] = (double*)SDG_checked_alloc(atom_grid_count,sizeof(double),1,
                                                         "Tmp_Den_Grid",myid);
    Tmp_Den_Grid[i][0] = Tmp_Den_Grid_storage[i];
    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = F_M2G[Mc_AN];
      Tmp_Den_Grid[i][Mc_AN] = Tmp_Den_Grid_storage[i] + atom_grid_offset;
      atom_grid_offset += (size_t)GridN_Atom[Gc_AN];
      size_Tmp_Den_Grid = SDG_checked_add(size_Tmp_Den_Grid,(size_t)GridN_Atom[Gc_AN],
                                          "Tmp_Den_Grid total",myid);
    }
  }

  size_Den_Snd_Grid_A2B = 0;
  for (ID=0; ID<numprocs; ID++){
    size_t count = SDG_checked_mul((size_t)Num_Snd_Grid_A2B[ID],(size_t)(SpinP_switch+1),
                                   "Den_Snd_Grid_A2B rank count",myid);
    size_Den_Snd_Grid_A2B = SDG_checked_add(size_Den_Snd_Grid_A2B,count,
                                            "Den_Snd_Grid_A2B total",myid);
  }
  Den_Snd_Grid_A2B = (double**)SDG_checked_alloc((size_t)numprocs,sizeof(double*),0,
                                                  "Den_Snd_Grid_A2B pointers",myid);
  Den_Snd_Grid_A2B_storage = (double*)SDG_checked_alloc(size_Den_Snd_Grid_A2B,sizeof(double),0,
                                                        "Den_Snd_Grid_A2B",myid);
  {
    size_t offset = 0;
    for (ID=0; ID<numprocs; ID++){
      Den_Snd_Grid_A2B[ID] = Den_Snd_Grid_A2B_storage + offset;
      offset += (size_t)Num_Snd_Grid_A2B[ID]*(size_t)(SpinP_switch+1);
    }
  }

  size_Den_Rcv_Grid_A2B = 0;
  for (ID=0; ID<numprocs; ID++){
    size_t count = SDG_checked_mul((size_t)Num_Rcv_Grid_A2B[ID],(size_t)(SpinP_switch+1),
                                   "Den_Rcv_Grid_A2B rank count",myid);
    size_Den_Rcv_Grid_A2B = SDG_checked_add(size_Den_Rcv_Grid_A2B,count,
                                            "Den_Rcv_Grid_A2B total",myid);
  }
  Den_Rcv_Grid_A2B = (double**)SDG_checked_alloc((size_t)numprocs,sizeof(double*),0,
                                                  "Den_Rcv_Grid_A2B pointers",myid);
  Den_Rcv_Grid_A2B_storage = (double*)SDG_checked_alloc(size_Den_Rcv_Grid_A2B,sizeof(double),0,
                                                        "Den_Rcv_Grid_A2B",myid);
  {
    size_t offset = 0;
    for (ID=0; ID<numprocs; ID++){
      Den_Rcv_Grid_A2B[ID] = Den_Rcv_Grid_A2B_storage + offset;
      offset += (size_t)Num_Rcv_Grid_A2B[ID]*(size_t)(SpinP_switch+1);
    }
  }

  /* PrintMemory */

  if (firsttime==1){
    PrintMemory("Set_Density_Grid: AtomDen_Grid",
                (long int)SDG_checked_mul(sizeof(double),size_Tmp_Den_Grid,
                                          "Tmp_Den_Grid bytes",myid), NULL);
    PrintMemory("Set_Density_Grid: Den_Snd_Grid_A2B",
                (long int)SDG_checked_mul(sizeof(double),size_Den_Snd_Grid_A2B,
                                          "Den_Snd_Grid_A2B bytes",myid), NULL);
    PrintMemory("Set_Density_Grid: Den_Rcv_Grid_A2B",
                (long int)SDG_checked_mul(sizeof(double),size_Den_Rcv_Grid_A2B,
                                          "Den_Rcv_Grid_A2B bytes",myid), NULL);
    firsttime = 0;
  }

  /****************************************************
                when orbital optimization
  ****************************************************/

  if (Calc_CntOrbital_ON==1 && Cnt_kind==0 && Cnt_switch==1){
      
    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
       
      dtime(&Stime_atom);
      
      /* COrbs_Grid */
 
      Gc_AN = M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];
      NO0 = Spe_Total_CNO[Cwan]; 
      for (Nc=0; Nc<GridN_Atom[Gc_AN]; Nc++){

        al = -1;
	for (L0=0; L0<=Spe_MaxL_Basis[Cwan]; L0++){
	  for (Mul0=0; Mul0<Spe_Num_CBasis[Cwan][L0]; Mul0++){
	    for (M0=0; M0<=2*L0; M0++){

	      al++;
	      tmp0 = 0.0;

	      for (p=0; p<Spe_Specified_Num[Cwan][al]; p++){
	        j = Spe_Trans_Orbital[Cwan][al][p];  
	        tmp0 += CntCoes[Mc_AN][al][p]*Orbs_Grid[Mc_AN][Nc][j];/* AITUNE */
	      }

	      COrbs_Grid[Mc_AN][al][Nc] = (Type_Orbs_Grid)tmp0;
	    }
	  }
        }
      }

      dtime(&Etime_atom);
      time_per_atom[Gc_AN] += Etime_atom - Stime_atom;
    }

    /**********************************************
     MPI:

     COrbs_Grid    
    ***********************************************/

    /* allocation of arrays  */
    Snd_Size = (int*)malloc(sizeof(int)*numprocs); 
    Rcv_Size = (int*)malloc(sizeof(int)*numprocs); 

    /* find data size for sending and receiving */

    My_Max = -10000;
    for (ID=0; ID<numprocs; ID++){

      IDS = (myid + ID) % numprocs;
      IDR = (myid - ID + numprocs) % numprocs;

      if (ID!=0){
        /*  sending size */
        if (F_Snd_Num[IDS]!=0){
          /* find data size */ 
          size1 = 0; 
          for (n=0; n<F_Snd_Num[IDS]; n++){
            Gc_AN = Snd_GAN[IDS][n];
            Cwan = WhatSpecies[Gc_AN];
            size1 += GridN_Atom[Gc_AN]*Spe_Total_CNO[Cwan];
          }

          Snd_Size[IDS] = size1;
          MPI_Isend(&size1, 1, MPI_INT, IDS, tag, mpi_comm_level1, &request);
        }
        else{
          Snd_Size[IDS] = 0;
        }

        /* receiving size */
        if (F_Rcv_Num[IDR]!=0){
          MPI_Recv(&size2, 1, MPI_INT, IDR, tag, mpi_comm_level1, &stat);
          Rcv_Size[IDR] = size2;
        }
        else{
          Rcv_Size[IDR] = 0;
        }
        if (F_Snd_Num[IDS]!=0) MPI_Wait(&request,&stat);
      } 
      else{
        Snd_Size[IDS] = 0;
        Rcv_Size[IDR] = 0;
      }

      if (My_Max<Snd_Size[IDS]) My_Max = Snd_Size[IDS];
      if (My_Max<Rcv_Size[IDR]) My_Max = Rcv_Size[IDR];

    }  

    MPI_Allreduce(&My_Max, &Max_Size, 1, MPI_INT, MPI_MAX, mpi_comm_level1);
    /* allocation of arrays */ 
    tmp_array  = (double*)malloc(sizeof(double)*Max_Size);
    tmp_array2 = (double*)malloc(sizeof(double)*Max_Size);

    /* send and receive COrbs_Grid */

    for (ID=0; ID<numprocs; ID++){

      IDS = (myid + ID) % numprocs;
      IDR = (myid - ID + numprocs) % numprocs;

      if (ID!=0){

        /* sending of data */ 

        if (F_Snd_Num[IDS]!=0){

          /* find data size */
          size1 = Snd_Size[IDS];

          /* multidimentional array to vector array */
          k = 0; 
          for (n=0; n<F_Snd_Num[IDS]; n++){
            Mc_AN = Snd_MAN[IDS][n];
            Gc_AN = Snd_GAN[IDS][n];
            Cwan = WhatSpecies[Gc_AN];
            NO0 = Spe_Total_CNO[Cwan]; 
            for (i=0; i<NO0; i++){
              for (Nc=0; Nc<GridN_Atom[Gc_AN]; Nc++){
                tmp_array[k] = COrbs_Grid[Mc_AN][i][Nc];
                k++;
              }          
            }
          } 

          /* MPI_Isend */
          MPI_Isend(&tmp_array[0], size1, MPI_DOUBLE, IDS,
                    tag, mpi_comm_level1, &request);
        }

        /* receiving of block data */

        if (F_Rcv_Num[IDR]!=0){

          /* find data size */
          size2 = Rcv_Size[IDR]; 

          /* MPI_Recv */
          MPI_Recv(&tmp_array2[0], size2, MPI_DOUBLE, IDR, tag, mpi_comm_level1, &stat);

          k = 0;
          Mc_AN = F_TopMAN[IDR] - 1;
          for (n=0; n<F_Rcv_Num[IDR]; n++){
            Mc_AN++;
            Gc_AN = Rcv_GAN[IDR][n];
            Cwan = WhatSpecies[Gc_AN];
            NO0 = Spe_Total_CNO[Cwan]; 

            for (i=0; i<NO0; i++){
              for (Nc=0; Nc<GridN_Atom[Gc_AN]; Nc++){
                COrbs_Grid[Mc_AN][i][Nc] = tmp_array2[k];
                k++;
              }          
            }
          }
        }
        if (F_Snd_Num[IDS]!=0) MPI_Wait(&request,&stat);
      } 
    }  

    /* freeing of arrays  */
    free(tmp_array);
    free(tmp_array2);
    free(Snd_Size);
    free(Rcv_Size);
  }

  /**********************************************
              calculate Tmp_Den_Grid
  ***********************************************/
    
  dtime(&time1);
  sd_pre = time1 - sd_t0; sd_t0 = time1;

  /* the distributed GPU quadrature fills Tmp_Den_Grid directly; a run-time
     failure on any rank simply falls back to the CPU loop on that rank */
  if (use_local_gpu){
    if (!Set_Density_Grid_GPU_Local_Run(CDM,Tmp_Den_Grid)) {
      use_local_gpu=0;
      local_failed=1;
    }
    else sd_mode = Set_Density_Grid_GPU_Local_OnTheFly() ? "otf" : "gpu-local";
  }

  /* the orbitals evaluated on the device batch by batch (Set_Hamiltonian.c),
     rank by rank, before the CPU loop */
  {
    extern int Set_Density_Grid_OTF_Run(int Cnt_kind, double *****CDM, double ***Tmp_Den_Grid);
    if (!use_local_gpu && use_otf_gpu && !(openmx_gpu_capacity_test_recovery() || openmx_gpu_workers_requested())){
      if (Set_Density_Grid_OTF_Run(Cnt_kind,CDM,Tmp_Den_Grid)) {
        use_local_gpu=1;
        local_failed=0;
        sd_mode = "otf";
      }
      else local_failed=1;
    }
  }

  if (Set_Density_Grid_GPU_Capacity(Cnt_kind,use_local_gpu,CDM,Tmp_Den_Grid)) {
    if (!use_local_gpu) sd_mode="gpu-capacity";
    use_local_gpu=1;
  }

  if (!use_local_gpu){
  
  
#pragma omp parallel shared(myid,G2ID,Orbs_Grid_FNAN,List_YOUSO,time_per_atom,Tmp_Den_Grid,Orbs_Grid,COrbs_Grid,Cnt_switch,Cnt_kind,GListTAtoms2,GListTAtoms1,NumOLG,CDM,SpinP_switch,use_nc_simd,nc_simd_mode,nc_grid_batch,use_col_simd,col_simd_mode,col_grid_batch,WhatSpecies,ncn,F_G2M,natn,Spe_Total_CNO,M2G) private(Mc_AN,h_AN,Stime_atom,Etime_atom,Gc_AN,Cwan,NO0,Gh_AN,Mh_AN,Rnh,Hwan,NO1,spin,i,j,tmp_CDM,Nog,Nc_0,Nc_1,Nc_2,Nc_3,Nh_0,Nh_1,Nh_2,Nh_3,orbs0_0,orbs0_1,orbs0_2,orbs0_3,orbs1_0,orbs1_1,orbs1_2,orbs1_3,sum_0,sum_1,sum_2,sum_3,tmp0_0,tmp0_1,tmp0_2,tmp0_3,Nc,Nh,orbs0,orbs1,sum,tmp0)
  {

    orbs0 = (double*)malloc(sizeof(double)*List_YOUSO[7]);
    orbs1 = (double*)malloc(sizeof(double)*List_YOUSO[7]);

    orbs0_0 = NULL;
    orbs0_1 = NULL;
    orbs0_2 = NULL;
    orbs0_3 = NULL;
    orbs1_0 = NULL;
    orbs1_1 = NULL;
    orbs1_2 = NULL;
    orbs1_3 = NULL;
    tmp_CDM = NULL;
    double **tmp_CDM_rows = NULL;
    double *tmp_CDM_storage = NULL;

    if (!use_nc_simd && !use_col_simd){
      const size_t max_orb = (size_t)List_YOUSO[7];
      const size_t nspin = (size_t)(SpinP_switch+1);
      const size_t nrows = SDG_checked_mul(nspin,max_orb,"density matrix rows",myid);
      const size_t nelem = SDG_checked_mul(nrows,max_orb,"density matrix scratch",myid);

      orbs0_0 = (double*)SDG_checked_alloc(max_orb,sizeof(double),0,"orbital scratch",myid);
      orbs0_1 = (double*)SDG_checked_alloc(max_orb,sizeof(double),0,"orbital scratch",myid);
      orbs0_2 = (double*)SDG_checked_alloc(max_orb,sizeof(double),0,"orbital scratch",myid);
      orbs0_3 = (double*)SDG_checked_alloc(max_orb,sizeof(double),0,"orbital scratch",myid);
      orbs1_0 = (double*)SDG_checked_alloc(max_orb,sizeof(double),0,"orbital scratch",myid);
      orbs1_1 = (double*)SDG_checked_alloc(max_orb,sizeof(double),0,"orbital scratch",myid);
      orbs1_2 = (double*)SDG_checked_alloc(max_orb,sizeof(double),0,"orbital scratch",myid);
      orbs1_3 = (double*)SDG_checked_alloc(max_orb,sizeof(double),0,"orbital scratch",myid);

      tmp_CDM = (double***)SDG_checked_alloc(nspin,sizeof(double**),0,
                                              "density matrix pointers",myid);
      tmp_CDM_rows = (double**)SDG_checked_alloc(nrows,sizeof(double*),0,
                                                  "density matrix row pointers",myid);
      tmp_CDM_storage = (double*)SDG_checked_alloc(nelem,sizeof(double),0,
                                                   "density matrix scratch",myid);
      for (i=0; i<(SpinP_switch+1); i++){
        tmp_CDM[i] = tmp_CDM_rows + (size_t)i*max_orb;
        for (j=0; j<List_YOUSO[7]; j++){
          tmp_CDM[i][j] = tmp_CDM_storage + ((size_t)i*max_orb+j)*max_orb;
        }
      }
    }

    /* The non-collinear contraction uses a spin-interleaved density matrix
       and orbital-major batches.  Runtime dispatch selects eight grid points
       for AVX-512 or four grid points for AVX2. */
    double *nc_dm = NULL;
    double *nc_orb0 = NULL;
    double *nc_orb1 = NULL;
    if (use_nc_simd){
      const size_t max_orb = (size_t)List_YOUSO[7];
      const size_t max_orb2 = SDG_checked_mul(max_orb,max_orb,
                                              "NC density matrix scratch",myid);
      nc_dm = (double*)SDG_checked_aligned_alloc(64,
                           SDG_checked_mul((size_t)4,max_orb2,
                                           "NC density matrix scratch",myid),
                           sizeof(double),"NC density matrix scratch",myid);
      nc_orb0 = (double*)SDG_checked_aligned_alloc(64,
                             SDG_checked_mul(max_orb,(size_t)nc_grid_batch,
                                             "NC orbital scratch",myid),
                             sizeof(double),"NC orbital scratch",myid);
      nc_orb1 = (double*)SDG_checked_aligned_alloc(64,
                             SDG_checked_mul(max_orb,(size_t)nc_grid_batch,
                                             "NC orbital scratch",myid),
                             sizeof(double),"NC orbital scratch",myid);
    }

    /* The collinear layout is [i][j][spin] with one or two components.
       Separate buffers leave the established NC packing and kernels intact. */
    double *col_dm = NULL;
    double *col_orb0 = NULL;
    double *col_orb1 = NULL;
    if (use_col_simd){
      const size_t max_orb = (size_t)List_YOUSO[7];
      const size_t max_orb2 = SDG_checked_mul(max_orb,max_orb,
                                              "collinear density matrix scratch",myid);
      const size_t spin_count = (size_t)(SpinP_switch+1);
      col_dm = (double*)SDG_checked_aligned_alloc(64,
                            SDG_checked_mul(spin_count,max_orb2,
                                            "collinear density matrix scratch",myid),
                            sizeof(double),"collinear density matrix scratch",myid);
      col_orb0 = (double*)SDG_checked_aligned_alloc(64,
                              SDG_checked_mul(max_orb,(size_t)col_grid_batch,
                                              "collinear orbital scratch",myid),
                              sizeof(double),"collinear orbital scratch",myid);
      col_orb1 = (double*)SDG_checked_aligned_alloc(64,
                              SDG_checked_mul(max_orb,(size_t)col_grid_batch,
                                              "collinear orbital scratch",myid),
                              sizeof(double),"collinear orbital scratch",myid);
    }

    /* Each center atom owns a disjoint Tmp_Den_Grid slice.  Assign whole
       atoms to threads, avoiding two workshare barriers for every neighbor
       pair and the large per-thread grid reduction used by the old loop. */
#pragma omp for schedule(static)
    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

      dtime(&Stime_atom);

      /* set data on Mc_AN */

      Gc_AN = M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];
      NO0 = Spe_Total_CNO[Cwan]; 
      int spin;

      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	/* set data on h_AN */
    
	Gh_AN = natn[Gc_AN][h_AN];
	Mh_AN = F_G2M[Gh_AN];
	Rnh = ncn[Gc_AN][h_AN];
	Hwan = WhatSpecies[Gh_AN];
	NO1 = Spe_Total_CNO[Hwan];

	/* Pack once per atom pair.  Interleaving the four NC components makes
	   their coefficients one compact load stream in the batched kernel. */
		if (use_nc_simd){
		  for (i=0; i<NO0; i++){
		    for (j=0; j<NO1; j++){
              const size_t ij = (size_t)4*((size_t)i*NO1+j);
              nc_dm[ij+0] = CDM[0][Mc_AN][h_AN][i][j];
              nc_dm[ij+1] = CDM[1][Mc_AN][h_AN][i][j];
              nc_dm[ij+2] = CDM[2][Mc_AN][h_AN][i][j];
              nc_dm[ij+3] = CDM[3][Mc_AN][h_AN][i][j];
		    }
		  }
		}
		else if (use_col_simd){
		  const int spin_count = SpinP_switch+1;
		  for (i=0; i<NO0; i++){
		    for (j=0; j<NO1; j++){
		      const size_t ij = (size_t)spin_count*((size_t)i*NO1+j);
		      for (spin=0; spin<spin_count; spin++){
		        col_dm[ij+(size_t)spin] = CDM[spin][Mc_AN][h_AN][i][j];
		      }
		    }
		  }
		}
		else{
	  for (spin=0; spin<=SpinP_switch; spin++){
	    for (i=0; i<NO0; i++){
	      for (j=0; j<NO1; j++){
	        tmp_CDM[spin][i][j] = CDM[spin][Mc_AN][h_AN][i][j];
	      }
	    }
	  }
	}

	/* summation of non-zero elements */
	/* for (Nog=0; Nog<NumOLG[Mc_AN][h_AN]; Nog++){ */
	if (use_nc_simd){
          int nc_batch[SDG_NC_GRID_BATCH_MAX];
          int nh_batch[SDG_NC_GRID_BATCH_MAX];
          int g;

          for (Nog=0; Nog<=NumOLG[Mc_AN][h_AN]-nc_grid_batch;
               Nog+=nc_grid_batch){
            for (g=0; g<nc_grid_batch; g++){
              nc_batch[g] = GListTAtoms1[Mc_AN][h_AN][Nog+g];
              nh_batch[g] = GListTAtoms2[Mc_AN][h_AN][Nog+g];
            }

            if (Cnt_kind==0 && Cnt_switch==1){
              for (i=0; i<NO0; i++){
#pragma omp simd
                for (g=0; g<nc_grid_batch; g++){
                  nc_orb0[(size_t)i*nc_grid_batch+g] =
                    COrbs_Grid[Mc_AN][i][nc_batch[g]];
                }
              }
              for (j=0; j<NO1; j++){
#pragma omp simd
                for (g=0; g<nc_grid_batch; g++){
                  nc_orb1[(size_t)j*nc_grid_batch+g] =
                    COrbs_Grid[Mh_AN][j][nh_batch[g]];
                }
              }
            }
            else{
              for (i=0; i<NO0; i++){
#pragma omp simd
                for (g=0; g<nc_grid_batch; g++){
                  nc_orb0[(size_t)i*nc_grid_batch+g] =
                    Orbs_Grid[Mc_AN][nc_batch[g]][i];
                }
              }

              if (G2ID[Gh_AN]==myid){
                for (j=0; j<NO1; j++){
#pragma omp simd
                  for (g=0; g<nc_grid_batch; g++){
                    nc_orb1[(size_t)j*nc_grid_batch+g] =
                      Orbs_Grid[Mh_AN][nh_batch[g]][j];
                  }
                }
              }
              else{
                for (j=0; j<NO1; j++){
#pragma omp simd
                  for (g=0; g<nc_grid_batch; g++){
                    nc_orb1[(size_t)j*nc_grid_batch+g] =
                      Orbs_Grid_FNAN[Mc_AN][h_AN][Nog+g][j];
                  }
                }
              }
            }

            if (nc_simd_mode==SDG_NC_SIMD_AVX512){
              SDG_accumulate_nc8(NO0,NO1,nc_dm,nc_orb0,nc_orb1,nc_batch,
                                 Tmp_Den_Grid[0][Mc_AN],Tmp_Den_Grid[1][Mc_AN],
                                 Tmp_Den_Grid[2][Mc_AN],Tmp_Den_Grid[3][Mc_AN]);
            }
            else{
              SDG_accumulate_nc4(NO0,NO1,nc_dm,nc_orb0,nc_orb1,nc_batch,
                                 Tmp_Den_Grid[0][Mc_AN],Tmp_Den_Grid[1][Mc_AN],
                                 Tmp_Den_Grid[2][Mc_AN],Tmp_Den_Grid[3][Mc_AN]);
            }
          }

          /* At most seven (AVX-512) or three (AVX2) points remain.  This tail retains the
             same spin fusion and exact per-component accumulation order. */
          for (; Nog<NumOLG[Mc_AN][h_AN]; Nog++){
            Nc = GListTAtoms1[Mc_AN][h_AN][Nog];
            Nh = GListTAtoms2[Mc_AN][h_AN][Nog];

            if (Cnt_kind==0 && Cnt_switch==1){
              for (i=0; i<NO0; i++) orbs0[i] = COrbs_Grid[Mc_AN][i][Nc];
              for (j=0; j<NO1; j++) orbs1[j] = COrbs_Grid[Mh_AN][j][Nh];
            }
            else{
              for (i=0; i<NO0; i++) orbs0[i] = Orbs_Grid[Mc_AN][Nc][i];
              if (G2ID[Gh_AN]==myid){
                for (j=0; j<NO1; j++) orbs1[j] = Orbs_Grid[Mh_AN][Nh][j];
              }
              else{
                for (j=0; j<NO1; j++)
                  orbs1[j] = Orbs_Grid_FNAN[Mc_AN][h_AN][Nog][j];
              }
            }

            SDG_accumulate_nc1(NO0,NO1,nc_dm,orbs0,orbs1,Nc,
                               Tmp_Den_Grid[0][Mc_AN],Tmp_Den_Grid[1][Mc_AN],
                               Tmp_Den_Grid[2][Mc_AN],Tmp_Den_Grid[3][Mc_AN]);
          }
        }
        else if (use_col_simd){
          int nc_batch[SDG_NC_GRID_BATCH_MAX];
          int nh_batch[SDG_NC_GRID_BATCH_MAX];
          const int spin_count = SpinP_switch+1;
          double *den1 = SpinP_switch==1 ? Tmp_Den_Grid[1][Mc_AN] : NULL;
          int g;

          for (Nog=0; Nog<=NumOLG[Mc_AN][h_AN]-col_grid_batch;
               Nog+=col_grid_batch){
            for (g=0; g<col_grid_batch; g++){
              nc_batch[g] = GListTAtoms1[Mc_AN][h_AN][Nog+g];
              nh_batch[g] = GListTAtoms2[Mc_AN][h_AN][Nog+g];
            }

            if (Cnt_kind==0 && Cnt_switch==1){
              for (i=0; i<NO0; i++){
#pragma omp simd
                for (g=0; g<col_grid_batch; g++){
                  col_orb0[(size_t)i*col_grid_batch+g] =
                    COrbs_Grid[Mc_AN][i][nc_batch[g]];
                }
              }
              for (j=0; j<NO1; j++){
#pragma omp simd
                for (g=0; g<col_grid_batch; g++){
                  col_orb1[(size_t)j*col_grid_batch+g] =
                    COrbs_Grid[Mh_AN][j][nh_batch[g]];
                }
              }
            }
            else{
              for (i=0; i<NO0; i++){
#pragma omp simd
                for (g=0; g<col_grid_batch; g++){
                  col_orb0[(size_t)i*col_grid_batch+g] =
                    Orbs_Grid[Mc_AN][nc_batch[g]][i];
                }
              }

              if (G2ID[Gh_AN]==myid){
                for (j=0; j<NO1; j++){
#pragma omp simd
                  for (g=0; g<col_grid_batch; g++){
                    col_orb1[(size_t)j*col_grid_batch+g] =
                      Orbs_Grid[Mh_AN][nh_batch[g]][j];
                  }
                }
              }
              else{
                for (j=0; j<NO1; j++){
#pragma omp simd
                  for (g=0; g<col_grid_batch; g++){
                    col_orb1[(size_t)j*col_grid_batch+g] =
                      Orbs_Grid_FNAN[Mc_AN][h_AN][Nog+g][j];
                  }
                }
              }
            }

            if (col_simd_mode==SDG_NC_SIMD_AVX512){
              SDG_accumulate_col8(NO0,NO1,spin_count,col_dm,col_orb0,col_orb1,
                                  nc_batch,Tmp_Den_Grid[0][Mc_AN],den1);
            }
            else{
              SDG_accumulate_col4(NO0,NO1,spin_count,col_dm,col_orb0,col_orb1,
                                  nc_batch,Tmp_Den_Grid[0][Mc_AN],den1);
            }
          }

          /* SIMD lanes are independent grid points, so every component keeps
             the original j-then-i sum order.  Update duplicate Nc entries in
             increasing Nog order, just as the scalar path does. */
          for (; Nog<NumOLG[Mc_AN][h_AN]; Nog++){
            Nc = GListTAtoms1[Mc_AN][h_AN][Nog];
            Nh = GListTAtoms2[Mc_AN][h_AN][Nog];

            if (Cnt_kind==0 && Cnt_switch==1){
              for (i=0; i<NO0; i++) orbs0[i] = COrbs_Grid[Mc_AN][i][Nc];
              for (j=0; j<NO1; j++) orbs1[j] = COrbs_Grid[Mh_AN][j][Nh];
            }
            else{
              for (i=0; i<NO0; i++) orbs0[i] = Orbs_Grid[Mc_AN][Nc][i];
              if (G2ID[Gh_AN]==myid){
                for (j=0; j<NO1; j++) orbs1[j] = Orbs_Grid[Mh_AN][Nh][j];
              }
              else{
                for (j=0; j<NO1; j++){
                  orbs1[j] = Orbs_Grid_FNAN[Mc_AN][h_AN][Nog][j];
                }
              }
            }

            SDG_accumulate_col1(NO0,NO1,spin_count,col_dm,orbs0,orbs1,Nc,
                                Tmp_Den_Grid[0][Mc_AN],den1);
          }
        }
        else{
		for (Nog=0; Nog<NumOLG[Mc_AN][h_AN]-3; Nog+=4){

	  Nc_0 = GListTAtoms1[Mc_AN][h_AN][Nog];
	  Nc_1 = GListTAtoms1[Mc_AN][h_AN][Nog+1];
	  Nc_2 = GListTAtoms1[Mc_AN][h_AN][Nog+2];
	  Nc_3 = GListTAtoms1[Mc_AN][h_AN][Nog+3];
	  
	  Nh_0 = GListTAtoms2[Mc_AN][h_AN][Nog];
	  Nh_1 = GListTAtoms2[Mc_AN][h_AN][Nog+1];
	  Nh_2 = GListTAtoms2[Mc_AN][h_AN][Nog+2];
	  Nh_3 = GListTAtoms2[Mc_AN][h_AN][Nog+3];
	  
	  /* Now under the orbital optimization */
	  if (Cnt_kind==0 && Cnt_switch==1){
	    for (i=0; i<NO0; i++){
	      orbs0_0[i] = COrbs_Grid[Mc_AN][i][Nc_0];
	      orbs0_1[i] = COrbs_Grid[Mc_AN][i][Nc_1];
	      orbs0_2[i] = COrbs_Grid[Mc_AN][i][Nc_2];
	      orbs0_3[i] = COrbs_Grid[Mc_AN][i][Nc_3];
	    }
	    for (j=0; j<NO1; j++){
	      orbs1_0[j] = COrbs_Grid[Mh_AN][j][Nh_0];
	      orbs1_1[j] = COrbs_Grid[Mh_AN][j][Nh_1];
	      orbs1_2[j] = COrbs_Grid[Mh_AN][j][Nh_2];
	      orbs1_3[j] = COrbs_Grid[Mh_AN][j][Nh_3];
	    }
	  }
	  /* else if ! "now under the orbital optimization" */
	  else{
	    for (i=0; i<NO0; i++){
	      orbs0_0[i] = Orbs_Grid[Mc_AN][Nc_0][i];
	      orbs0_1[i] = Orbs_Grid[Mc_AN][Nc_1][i];
	      orbs0_2[i] = Orbs_Grid[Mc_AN][Nc_2][i];
	      orbs0_3[i] = Orbs_Grid[Mc_AN][Nc_3][i]; 
	    }

            if (G2ID[Gh_AN]==myid){
	      for (j=0; j<NO1; j++){
		orbs1_0[j] = Orbs_Grid[Mh_AN][Nh_0][j];
		orbs1_1[j] = Orbs_Grid[Mh_AN][Nh_1][j];
		orbs1_2[j] = Orbs_Grid[Mh_AN][Nh_2][j];
		orbs1_3[j] = Orbs_Grid[Mh_AN][Nh_3][j]; 
	      }
	    }
            else{
	      for (j=0; j<NO1; j++){
		orbs1_0[j] = Orbs_Grid_FNAN[Mc_AN][h_AN][Nog  ][j];
		orbs1_1[j] = Orbs_Grid_FNAN[Mc_AN][h_AN][Nog+1][j];
		orbs1_2[j] = Orbs_Grid_FNAN[Mc_AN][h_AN][Nog+2][j];
		orbs1_3[j] = Orbs_Grid_FNAN[Mc_AN][h_AN][Nog+3][j]; 
	      }
	    }
	  }
	  
	  if (SpinP_switch==3){
            /* The first suffix is spin and the second is the grid point.
               Interleave the four independent spin contractions so that the
               orbital values are loaded only once.  The j and i accumulation
               order of every individual density component is unchanged. */
            double sum_00 = 0.0, sum_01 = 0.0, sum_02 = 0.0, sum_03 = 0.0;
            double sum_10 = 0.0, sum_11 = 0.0, sum_12 = 0.0, sum_13 = 0.0;
            double sum_20 = 0.0, sum_21 = 0.0, sum_22 = 0.0, sum_23 = 0.0;
            double sum_30 = 0.0, sum_31 = 0.0, sum_32 = 0.0, sum_33 = 0.0;

            for (i=0; i<NO0; i++){
              const double *dm0 = tmp_CDM[0][i];
              const double *dm1 = tmp_CDM[1][i];
              const double *dm2 = tmp_CDM[2][i];
              const double *dm3 = tmp_CDM[3][i];
              double tmp_00 = 0.0, tmp_01 = 0.0, tmp_02 = 0.0, tmp_03 = 0.0;
              double tmp_10 = 0.0, tmp_11 = 0.0, tmp_12 = 0.0, tmp_13 = 0.0;
              double tmp_20 = 0.0, tmp_21 = 0.0, tmp_22 = 0.0, tmp_23 = 0.0;
              double tmp_30 = 0.0, tmp_31 = 0.0, tmp_32 = 0.0, tmp_33 = 0.0;

              for (j=0; j<NO1; j++){
                const double orb_0 = orbs1_0[j];
                const double orb_1 = orbs1_1[j];
                const double orb_2 = orbs1_2[j];
                const double orb_3 = orbs1_3[j];
                const double cdm_0 = dm0[j];
                const double cdm_1 = dm1[j];
                const double cdm_2 = dm2[j];
                const double cdm_3 = dm3[j];

                tmp_00 += orb_0*cdm_0;
                tmp_01 += orb_1*cdm_0;
                tmp_02 += orb_2*cdm_0;
                tmp_03 += orb_3*cdm_0;
                tmp_10 += orb_0*cdm_1;
                tmp_11 += orb_1*cdm_1;
                tmp_12 += orb_2*cdm_1;
                tmp_13 += orb_3*cdm_1;
                tmp_20 += orb_0*cdm_2;
                tmp_21 += orb_1*cdm_2;
                tmp_22 += orb_2*cdm_2;
                tmp_23 += orb_3*cdm_2;
                tmp_30 += orb_0*cdm_3;
                tmp_31 += orb_1*cdm_3;
                tmp_32 += orb_2*cdm_3;
                tmp_33 += orb_3*cdm_3;
              }

              sum_00 += orbs0_0[i]*tmp_00;
              sum_01 += orbs0_1[i]*tmp_01;
              sum_02 += orbs0_2[i]*tmp_02;
              sum_03 += orbs0_3[i]*tmp_03;
              sum_10 += orbs0_0[i]*tmp_10;
              sum_11 += orbs0_1[i]*tmp_11;
              sum_12 += orbs0_2[i]*tmp_12;
              sum_13 += orbs0_3[i]*tmp_13;
              sum_20 += orbs0_0[i]*tmp_20;
              sum_21 += orbs0_1[i]*tmp_21;
              sum_22 += orbs0_2[i]*tmp_22;
              sum_23 += orbs0_3[i]*tmp_23;
              sum_30 += orbs0_0[i]*tmp_30;
              sum_31 += orbs0_1[i]*tmp_31;
              sum_32 += orbs0_2[i]*tmp_32;
              sum_33 += orbs0_3[i]*tmp_33;
            }

            Tmp_Den_Grid[0][Mc_AN][Nc_0] += sum_00;
            Tmp_Den_Grid[0][Mc_AN][Nc_1] += sum_01;
            Tmp_Den_Grid[0][Mc_AN][Nc_2] += sum_02;
            Tmp_Den_Grid[0][Mc_AN][Nc_3] += sum_03;
            Tmp_Den_Grid[1][Mc_AN][Nc_0] += sum_10;
            Tmp_Den_Grid[1][Mc_AN][Nc_1] += sum_11;
            Tmp_Den_Grid[1][Mc_AN][Nc_2] += sum_12;
            Tmp_Den_Grid[1][Mc_AN][Nc_3] += sum_13;
            Tmp_Den_Grid[2][Mc_AN][Nc_0] += sum_20;
            Tmp_Den_Grid[2][Mc_AN][Nc_1] += sum_21;
            Tmp_Den_Grid[2][Mc_AN][Nc_2] += sum_22;
            Tmp_Den_Grid[2][Mc_AN][Nc_3] += sum_23;
            Tmp_Den_Grid[3][Mc_AN][Nc_0] += sum_30;
            Tmp_Den_Grid[3][Mc_AN][Nc_1] += sum_31;
            Tmp_Den_Grid[3][Mc_AN][Nc_2] += sum_32;
            Tmp_Den_Grid[3][Mc_AN][Nc_3] += sum_33;
          }
          else{
            for (spin=0; spin<=SpinP_switch; spin++){

	      /* Tmp_Den_Grid */

	      sum_0 = 0.0;
	      sum_1 = 0.0;
	      sum_2 = 0.0;
	      sum_3 = 0.0;

	      for (i=0; i<NO0; i++){

	        tmp0_0 = 0.0;
	        tmp0_1 = 0.0;
	        tmp0_2 = 0.0;
	        tmp0_3 = 0.0;

	        for (j=0; j<NO1; j++){
		  tmp0_0 += orbs1_0[j]*tmp_CDM[spin][i][j];
		  tmp0_1 += orbs1_1[j]*tmp_CDM[spin][i][j];
		  tmp0_2 += orbs1_2[j]*tmp_CDM[spin][i][j];
		  tmp0_3 += orbs1_3[j]*tmp_CDM[spin][i][j];
	        }

	        sum_0 += orbs0_0[i]*tmp0_0;
	        sum_1 += orbs0_1[i]*tmp0_1;
	        sum_2 += orbs0_2[i]*tmp0_2;
	        sum_3 += orbs0_3[i]*tmp0_3;
	      }
		
	      Tmp_Den_Grid[spin][Mc_AN][Nc_0] += sum_0;
	      Tmp_Den_Grid[spin][Mc_AN][Nc_1] += sum_1;
	      Tmp_Den_Grid[spin][Mc_AN][Nc_2] += sum_2;
	      Tmp_Den_Grid[spin][Mc_AN][Nc_3] += sum_3;

	    } /* spin */
          }
	} /* Nog */

	for (Nog = NumOLG[Mc_AN][h_AN] - (NumOLG[Mc_AN][h_AN] % 4); Nog<NumOLG[Mc_AN][h_AN]; Nog++){
	  /*for (; Nog<NumOLG[Mc_AN][h_AN]; Nog++){*/
	
	  Nc = GListTAtoms1[Mc_AN][h_AN][Nog];
	  Nh = GListTAtoms2[Mc_AN][h_AN][Nog]; 
 

	  if (Cnt_kind==0 && Cnt_switch==1){
	    for (i=0; i<NO0; i++){
	      orbs0[i] = COrbs_Grid[Mc_AN][i][Nc];
	    }
	    for (j=0; j<NO1; j++){
	      orbs1[j] = COrbs_Grid[Mh_AN][j][Nh];
	    }
	  }
	  else{
	    for (i=0; i<NO0; i++){
	      orbs0[i] = Orbs_Grid[Mc_AN][Nc][i];
	    }

	    if (G2ID[Gh_AN]==myid){
	      for (j=0; j<NO1; j++){
		orbs1[j] = Orbs_Grid[Mh_AN][Nh][j];
	      }
	    }
	    else{
	      for (j=0; j<NO1; j++){
		orbs1[j] = Orbs_Grid_FNAN[Mc_AN][h_AN][Nog][j];
	      }
	    }
	  }

	  if (SpinP_switch==3){
            double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;

            for (i=0; i<NO0; i++){
              const double *dm0 = tmp_CDM[0][i];
              const double *dm1 = tmp_CDM[1][i];
              const double *dm2 = tmp_CDM[2][i];
              const double *dm3 = tmp_CDM[3][i];
              double tmp_spin0 = 0.0, tmp_spin1 = 0.0;
              double tmp_spin2 = 0.0, tmp_spin3 = 0.0;

              for (j=0; j<NO1; j++){
                const double orb = orbs1[j];
                tmp_spin0 += orb*dm0[j];
                tmp_spin1 += orb*dm1[j];
                tmp_spin2 += orb*dm2[j];
                tmp_spin3 += orb*dm3[j];
              }

              sum0 += orbs0[i]*tmp_spin0;
              sum1 += orbs0[i]*tmp_spin1;
              sum2 += orbs0[i]*tmp_spin2;
              sum3 += orbs0[i]*tmp_spin3;
            }

            Tmp_Den_Grid[0][Mc_AN][Nc] += sum0;
            Tmp_Den_Grid[1][Mc_AN][Nc] += sum1;
            Tmp_Den_Grid[2][Mc_AN][Nc] += sum2;
            Tmp_Den_Grid[3][Mc_AN][Nc] += sum3;
          }
          else{
            for (spin=0; spin<=SpinP_switch; spin++){
 
 
	      sum = 0.0;
	      for (i=0; i<NO0; i++){
	        tmp0 = 0.0;
	        for (j=0; j<NO1; j++){
		  tmp0 += orbs1[j]*tmp_CDM[spin][i][j];
	        }
	        sum += orbs0[i]*tmp0;
	      }
 
	      Tmp_Den_Grid[spin][Mc_AN][Nc] += sum;
	    }
          }

	} /* Nog */
	} /* collinear four-grid path */
	
      } /* h_AN */

      dtime(&Etime_atom);
      time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

    } /* Mc_AN */

    /* freeing of arrays */ 

    free(orbs0);
    free(orbs1);

    free(orbs0_0);
    free(orbs0_1);
    free(orbs0_2);
    free(orbs0_3);
    free(orbs1_0);
    free(orbs1_1);
    free(orbs1_2);
    free(orbs1_3);

    free(tmp_CDM_storage);
    free(tmp_CDM_rows);
    free(tmp_CDM);
    free(nc_dm);
    free(nc_orb0);
    free(nc_orb1);
    free(col_dm);
    free(col_orb0);
    free(col_orb1);

#pragma omp flush(Tmp_Den_Grid)

  } /* #pragma omp parallel */
  
  } /* !use_local_gpu */

  dtime(&time2);
  if(myid==0 && measure_time){
    printf("Time for Part1=%18.5f\n",(time2-time1));fflush(stdout);
  }
  sd_quad = time2 - sd_t0; sd_t0 = time2;

  /* Like the NVIDIA path, report runtime fallbacks after communication.
     Participation is independent of level.of.stdout. */
  if (local_any){
    local_failed = Matomnum > 0 && local_failed;
    MPI_Iallreduce(&local_failed,&failed_ranks,1,MPI_INT,MPI_SUM,mpi_comm_level1,&failed_request);
  }

  /******************************************************
      MPI communication from the partitions A to B 
  ******************************************************/
  
  /* copy Tmp_Den_Grid to Den_Snd_Grid_A2B */

  for (ID=0; ID<numprocs; ID++) Num_Snd_Grid_A2B[ID] = 0;
  
  N2D = Ngrid1*Ngrid2;

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

    Gc_AN = M2G[Mc_AN];

    for (AN=0; AN<GridN_Atom[Gc_AN]; AN++){

      GN = GridListAtom[Mc_AN][AN];
      GN2N(GN,N3);
      n2D = N3[1]*Ngrid2 + N3[2];
      ID = (int)(n2D*(unsigned long long int)numprocs/N2D);

      if (SpinP_switch==0){
        Den_Snd_Grid_A2B[ID][Num_Snd_Grid_A2B[ID]] = Tmp_Den_Grid[0][Mc_AN][AN];
      }
      else if (SpinP_switch==1){
        Den_Snd_Grid_A2B[ID][Num_Snd_Grid_A2B[ID]*2+0] = Tmp_Den_Grid[0][Mc_AN][AN];
        Den_Snd_Grid_A2B[ID][Num_Snd_Grid_A2B[ID]*2+1] = Tmp_Den_Grid[1][Mc_AN][AN];
      }
      else if (SpinP_switch==3){
        Den_Snd_Grid_A2B[ID][Num_Snd_Grid_A2B[ID]*4+0] = Tmp_Den_Grid[0][Mc_AN][AN];
        Den_Snd_Grid_A2B[ID][Num_Snd_Grid_A2B[ID]*4+1] = Tmp_Den_Grid[1][Mc_AN][AN];
        Den_Snd_Grid_A2B[ID][Num_Snd_Grid_A2B[ID]*4+2] = Tmp_Den_Grid[2][Mc_AN][AN];
        Den_Snd_Grid_A2B[ID][Num_Snd_Grid_A2B[ID]*4+3] = Tmp_Den_Grid[3][Mc_AN][AN];
      }

      Num_Snd_Grid_A2B[ID]++;
    }
  }    

  /* MPI: A to B */  

  request_send = malloc(sizeof(MPI_Request)*NN_A2B_S);
  request_recv = malloc(sizeof(MPI_Request)*NN_A2B_R);
  stat_send = malloc(sizeof(MPI_Status)*NN_A2B_S);
  stat_recv = malloc(sizeof(MPI_Status)*NN_A2B_R);

  NN_S = 0;
  NN_R = 0;

  tag = 999;
  for (ID=1; ID<numprocs; ID++){

    IDS = (myid + ID) % numprocs;
    IDR = (myid - ID + numprocs) % numprocs;

    if (Num_Snd_Grid_A2B[IDS]!=0){
      MPI_Isend( &Den_Snd_Grid_A2B[IDS][0], Num_Snd_Grid_A2B[IDS]*(SpinP_switch+1), 
	         MPI_DOUBLE, IDS, tag, mpi_comm_level1, &request_send[NN_S]);
      NN_S++;
    }

    if (Num_Rcv_Grid_A2B[IDR]!=0){
      MPI_Irecv( &Den_Rcv_Grid_A2B[IDR][0], Num_Rcv_Grid_A2B[IDR]*(SpinP_switch+1), 
  	         MPI_DOUBLE, IDR, tag, mpi_comm_level1, &request_recv[NN_R]);
      NN_R++;
    }
  }

  if (NN_S!=0) MPI_Waitall(NN_S,request_send,stat_send);
  if (NN_R!=0) MPI_Waitall(NN_R,request_recv,stat_recv);

  free(request_send);
  free(request_recv);
  free(stat_send);
  free(stat_recv);

  /* for myid */
  for (i=0; i<Num_Rcv_Grid_A2B[myid]*(SpinP_switch+1); i++){
    Den_Rcv_Grid_A2B[myid][i] = Den_Snd_Grid_A2B[myid][i];
  }

  dtime(&sd_t1); sd_a2b = sd_t1 - sd_t0; sd_t0 = sd_t1;
  /******************************************************
   superposition of rho_i to calculate charge density 
   in the partition B.
  ******************************************************/

  /* initialize arrays */
  
  for (spin=0; spin<(SpinP_switch+1); spin++){
    for (BN=0; BN<My_NumGridB_AB; BN++){
      Density_Grid_B0[spin][BN] = 0.0;
    }
  }
  
  /* superposition of densities rho_i */

  for (ID=0; ID<numprocs; ID++){

    for (LN=0; LN<Num_Rcv_Grid_A2B[ID]; LN++){

      BN    = Index_Rcv_Grid_A2B[ID][3*LN+0];      
      Gc_AN = Index_Rcv_Grid_A2B[ID][3*LN+1];        
      GRc   = Index_Rcv_Grid_A2B[ID][3*LN+2]; 

      if (Solver!=4 || (Solver==4 && atv_ijk[GRc][1]==0 )){

	/* spin collinear non-polarization */
	if ( SpinP_switch==0 ){
	  Density_Grid_B0[0][BN] += Den_Rcv_Grid_A2B[ID][LN];
	}

	/* spin collinear polarization */
	else if ( SpinP_switch==1 ){
	  Density_Grid_B0[0][BN] += Den_Rcv_Grid_A2B[ID][LN*2  ];
	  Density_Grid_B0[1][BN] += Den_Rcv_Grid_A2B[ID][LN*2+1];
	} 

	/* spin non-collinear */
	else if ( SpinP_switch==3 ){
	  Density_Grid_B0[0][BN] += Den_Rcv_Grid_A2B[ID][LN*4  ];
	  Density_Grid_B0[1][BN] += Den_Rcv_Grid_A2B[ID][LN*4+1];
	  Density_Grid_B0[2][BN] += Den_Rcv_Grid_A2B[ID][LN*4+2];
	  Density_Grid_B0[3][BN] += Den_Rcv_Grid_A2B[ID][LN*4+3];
	} 

      } /* if (Solve!=4.....) */           

    } /* AN */ 
  } /* ID */  

  /****************************************************
   Conjugate complex of Density_Grid[3][MN] due to
   difference in the definition between density matrix
   and charge density
  ****************************************************/

  if (SpinP_switch==3){

    for (BN=0; BN<My_NumGridB_AB; BN++){
      Density_Grid_B0[3][BN] = -Density_Grid_B0[3][BN]; 
    }
  }

  dtime(&sd_t1); sd_sup = sd_t1 - sd_t0; sd_t0 = sd_t1;
  /******************************************************
             MPI: from the partitions B to D
  ******************************************************/

  Density_Grid_Copy_B2D(Density_Grid_B0);

  if (local_any){
    MPI_Wait(&failed_request,MPI_STATUS_IGNORE);
    Set_Density_Grid_GPU_Local_Report(local_path,failed_ranks);
  }

  /* freeing of arrays */

  for (i=0; i<(SpinP_switch+1); i++){
    free(Tmp_Den_Grid_storage[i]);
    free(Tmp_Den_Grid[i]);
  }
  free(Tmp_Den_Grid_storage);
  free(Tmp_Den_Grid);

  free(Den_Snd_Grid_A2B_storage);
  free(Den_Snd_Grid_A2B);

  free(Den_Rcv_Grid_A2B_storage);
  free(Den_Rcv_Grid_A2B);

  /* elapsed time */
  dtime(&TEtime);
  time0 = TEtime - TStime;
  sd_b2d = TEtime - sd_t0;
  if (sd_prof){
    fprintf(stderr,"SETDENPROF id=%d mode=%s pre=%.3f quad=%.3f a2b=%.3f sup=%.3f b2d=%.3f total=%.3f\n",
            myid,sd_mode,sd_pre,sd_quad,sd_a2b,sd_sup,sd_b2d,time0);
  }
  if(myid==0 && measure_time) printf("time0=%18.5f\n",time0);

  return time0;
}



void Data_Grid_Copy_B2C_2(double **data_B, double **data_C)
{
  static int firsttime=1;
  int CN,BN,LN,spin,i,gp,NN_S,NN_R;
  double *Work_Array_Snd_Grid_B2C;
  double *Work_Array_Rcv_Grid_B2C;
  int numprocs,myid,tag=999,ID,IDS,IDR;
  MPI_Status stat;
  MPI_Request request;
  MPI_Status *stat_send;
  MPI_Status *stat_recv;
  MPI_Request *request_send;
  MPI_Request *request_recv;

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /* allocation of arrays */
  /* the send/receive work arrays persist across calls (grow-only, as in
     the NVIDIA version): this copy runs every SCF step, and allocating and
     first touching its tens of MiB were a measurable part of its time.
     The request/status arrays are allocated per call. */
  {
    static double *snd_keep = NULL, *rcv_keep = NULL;
    static size_t snd_cap = 0, rcv_cap = 0;
    const size_t snd_need = (size_t)GP_B2C_S[NN_B2C_S]*(size_t)(SpinP_switch+1);
    const size_t rcv_need = (size_t)GP_B2C_R[NN_B2C_R]*(size_t)(SpinP_switch+1);

    /* SDG_checked_alloc aborts with a message where malloc fails */
    if (snd_keep==NULL || snd_cap<snd_need){
      free(snd_keep);
      snd_keep = (double*)SDG_checked_alloc(snd_need,sizeof(double),0,
                                            "Data_Grid_Copy_B2C_2 send buffer",myid);
      snd_cap = snd_need;
    }
    if (rcv_keep==NULL || rcv_cap<rcv_need){
      free(rcv_keep);
      rcv_keep = (double*)SDG_checked_alloc(rcv_need,sizeof(double),0,
                                            "Data_Grid_Copy_B2C_2 receive buffer",myid);
      rcv_cap = rcv_need;
    }
    Work_Array_Snd_Grid_B2C = snd_keep;
    Work_Array_Rcv_Grid_B2C = rcv_keep;
  }

  if (firsttime==1){
    PrintMemory("Data_Grid_Copy_B2C_2: Work_Array_Snd_Grid_B2C",
		sizeof(double)*GP_B2C_S[NN_B2C_S]*(SpinP_switch+1), NULL);
    PrintMemory("Data_Grid_Copy_B2C_2: Work_Array_Rcv_Grid_B2C",
		sizeof(double)*GP_B2C_R[NN_B2C_R]*(SpinP_switch+1), NULL);
    firsttime = 0;
  }

  /******************************************************
             MPI: from the partitions B to C
  ******************************************************/

  request_send = malloc(sizeof(MPI_Request)*NN_B2C_S);
  request_recv = malloc(sizeof(MPI_Request)*NN_B2C_R);
  stat_send = malloc(sizeof(MPI_Status)*NN_B2C_S);
  stat_recv = malloc(sizeof(MPI_Status)*NN_B2C_R);

  NN_S = 0;
  NN_R = 0;

  /* MPI_Irecv */

  for (ID=0; ID<NN_B2C_R; ID++){

    IDR = ID_NN_B2C_R[ID];
    gp = GP_B2C_R[ID];

    if (IDR!=myid){ 
      MPI_Irecv( &Work_Array_Rcv_Grid_B2C[(SpinP_switch+1)*gp], Num_Rcv_Grid_B2C[IDR]*(SpinP_switch+1),
                 MPI_DOUBLE, IDR, tag, mpi_comm_level1, &request_recv[NN_R]);
      NN_R++;
    }

  }

  /* MPI_Isend */

  for (ID=0; ID<NN_B2C_S; ID++){

    IDS = ID_NN_B2C_S[ID];
    gp = GP_B2C_S[ID];

    /* copy Density_Grid_B to Work_Array_Snd_Grid_B2C */

    for (LN=0; LN<Num_Snd_Grid_B2C[IDS]; LN++){
      BN = Index_Snd_Grid_B2C[IDS][LN];

      if (SpinP_switch==0){
        Work_Array_Snd_Grid_B2C[gp+LN]       = data_B[0][BN];
      }
      else if (SpinP_switch==1){
        Work_Array_Snd_Grid_B2C[2*gp+2*LN+0] = data_B[0][BN];
        Work_Array_Snd_Grid_B2C[2*gp+2*LN+1] = data_B[1][BN];
      }
      else if (SpinP_switch==3){
        Work_Array_Snd_Grid_B2C[4*gp+4*LN+0] = data_B[0][BN];
        Work_Array_Snd_Grid_B2C[4*gp+4*LN+1] = data_B[1][BN];
        Work_Array_Snd_Grid_B2C[4*gp+4*LN+2] = data_B[2][BN];
        Work_Array_Snd_Grid_B2C[4*gp+4*LN+3] = data_B[3][BN];
      }
    } /* LN */        

    if (IDS!=myid){
      MPI_Isend( &Work_Array_Snd_Grid_B2C[(SpinP_switch+1)*gp], Num_Snd_Grid_B2C[IDS]*(SpinP_switch+1), 
		 MPI_DOUBLE, IDS, tag, mpi_comm_level1, &request_send[NN_S]);
      NN_S++;
    }
  }

  /* MPI_Waitall */

  if (NN_S!=0) MPI_Waitall(NN_S,request_send,stat_send);
  if (NN_R!=0) MPI_Waitall(NN_R,request_recv,stat_recv);

  free(request_send);
  free(request_recv);
  free(stat_send);
  free(stat_recv);

  /* copy Work_Array_Rcv_Grid_B2C to data_C */

  for (ID=0; ID<NN_B2C_R; ID++){

    IDR = ID_NN_B2C_R[ID];

    if (IDR==myid){

      gp = GP_B2C_S[ID];

      for (LN=0; LN<Num_Rcv_Grid_B2C[IDR]; LN++){

	CN = Index_Rcv_Grid_B2C[IDR][LN];

	if (SpinP_switch==0){
	  data_C[0][CN] = Work_Array_Snd_Grid_B2C[gp+LN];
	}     
	else if (SpinP_switch==1){
	  data_C[0][CN] = Work_Array_Snd_Grid_B2C[2*gp+2*LN+0];
	  data_C[1][CN] = Work_Array_Snd_Grid_B2C[2*gp+2*LN+1];
	}     
	else if (SpinP_switch==3){
	  data_C[0][CN] = Work_Array_Snd_Grid_B2C[4*gp+4*LN+0];
	  data_C[1][CN] = Work_Array_Snd_Grid_B2C[4*gp+4*LN+1];
	  data_C[2][CN] = Work_Array_Snd_Grid_B2C[4*gp+4*LN+2];
	  data_C[3][CN] = Work_Array_Snd_Grid_B2C[4*gp+4*LN+3];
	}
      } /* LN */   

    }
    else {

      gp = GP_B2C_R[ID];

      for (LN=0; LN<Num_Rcv_Grid_B2C[IDR]; LN++){
	CN = Index_Rcv_Grid_B2C[IDR][LN];

	if (SpinP_switch==0){
	  data_C[0][CN] = Work_Array_Rcv_Grid_B2C[gp+LN];
	}
	else if (SpinP_switch==1){
	  data_C[0][CN] = Work_Array_Rcv_Grid_B2C[2*gp+2*LN+0];
	  data_C[1][CN] = Work_Array_Rcv_Grid_B2C[2*gp+2*LN+1];
	}     
	else if (SpinP_switch==3){
	  data_C[0][CN] = Work_Array_Rcv_Grid_B2C[4*gp+4*LN+0];
	  data_C[1][CN] = Work_Array_Rcv_Grid_B2C[4*gp+4*LN+1];
	  data_C[2][CN] = Work_Array_Rcv_Grid_B2C[4*gp+4*LN+2];
	  data_C[3][CN] = Work_Array_Rcv_Grid_B2C[4*gp+4*LN+3];
	}
      }
    }
  }

  /* if (SpinP_switch==0), 
     copy data_B[0] to data_B[1]
     copy data_C[0] to data_C[1]
  */

  if (SpinP_switch==0){
    for (BN=0; BN<My_NumGridB_AB; BN++){
      data_B[1][BN] = data_B[0][BN]; 
    }

    for (CN=0; CN<My_NumGridC; CN++){
      data_C[1][CN] = data_C[0][CN]; 
    }
  }

  /* freeing of arrays */
  /* Work_Array_Snd_Grid_B2C and Work_Array_Rcv_Grid_B2C persist */
}



void Data_Grid_Copy_B2C_1(double *data_B, double *data_C)
{
  static int firsttime=1;
  int CN,BN,LN,spin,i,gp,NN_S,NN_R;
  double *Work_Array_Snd_Grid_B2C;
  double *Work_Array_Rcv_Grid_B2C;
  int numprocs,myid,tag=999,ID,IDS,IDR;
  MPI_Status stat;
  MPI_Request request;
  MPI_Status *stat_send;
  MPI_Status *stat_recv;
  MPI_Request *request_send;
  MPI_Request *request_recv;

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /* allocation of arrays */
  
  Work_Array_Snd_Grid_B2C = (double*)malloc(sizeof(double)*GP_B2C_S[NN_B2C_S]); 
  Work_Array_Rcv_Grid_B2C = (double*)malloc(sizeof(double)*GP_B2C_R[NN_B2C_R]); 

  if (firsttime==1){
    PrintMemory("Data_Grid_Copy_B2C_1: Work_Array_Snd_Grid_B2C",
		sizeof(double)*GP_B2C_S[NN_B2C_S], NULL);
    PrintMemory("Data_Grid_Copy_B2C_1: Work_Array_Rcv_Grid_B2C",
		sizeof(double)*GP_B2C_R[NN_B2C_R], NULL);
    firsttime = 0;
  }

  /******************************************************
             MPI: from the partitions B to C
  ******************************************************/

  request_send = malloc(sizeof(MPI_Request)*NN_B2C_S);
  request_recv = malloc(sizeof(MPI_Request)*NN_B2C_R);
  stat_send = malloc(sizeof(MPI_Status)*NN_B2C_S);
  stat_recv = malloc(sizeof(MPI_Status)*NN_B2C_R);

  NN_S = 0;
  NN_R = 0;

  /* MPI_Irecv */

  for (ID=0; ID<NN_B2C_R; ID++){

    IDR = ID_NN_B2C_R[ID];
    gp = GP_B2C_R[ID];

    if (IDR!=myid){ 
      MPI_Irecv( &Work_Array_Rcv_Grid_B2C[gp], Num_Rcv_Grid_B2C[IDR],
                 MPI_DOUBLE, IDR, tag, mpi_comm_level1, &request_recv[NN_R]);
      NN_R++;
    }
  }
 
  /* MPI_Isend */

  for (ID=0; ID<NN_B2C_S; ID++){

    IDS = ID_NN_B2C_S[ID];
    gp = GP_B2C_S[ID];

    /* copy Density_Grid_B to Work_Array_Snd_Grid_B2C */

    for (LN=0; LN<Num_Snd_Grid_B2C[IDS]; LN++){
      BN = Index_Snd_Grid_B2C[IDS][LN];
      Work_Array_Snd_Grid_B2C[gp+LN] = data_B[BN];
    } 

    if (IDS!=myid){
      MPI_Isend( &Work_Array_Snd_Grid_B2C[gp], Num_Snd_Grid_B2C[IDS], 
		 MPI_DOUBLE, IDS, tag, mpi_comm_level1, &request_send[NN_S]);
      NN_S++;
    }
  }

  /* MPI_Waitall */

  if (NN_S!=0) MPI_Waitall(NN_S,request_send,stat_send);
  if (NN_R!=0) MPI_Waitall(NN_R,request_recv,stat_recv);

  free(request_send);
  free(request_recv);
  free(stat_send);
  free(stat_recv);

  /* copy Work_Array_Rcv_Grid_B2C to data_C */

  for (ID=0; ID<NN_B2C_R; ID++){

    IDR = ID_NN_B2C_R[ID];

    if (IDR==myid){
      gp = GP_B2C_S[ID];
      for (LN=0; LN<Num_Rcv_Grid_B2C[IDR]; LN++){
	CN = Index_Rcv_Grid_B2C[IDR][LN];
	data_C[CN] = Work_Array_Snd_Grid_B2C[gp+LN];
      } 
    }
    else{

      gp = GP_B2C_R[ID];
      for (LN=0; LN<Num_Rcv_Grid_B2C[IDR]; LN++){
	CN = Index_Rcv_Grid_B2C[IDR][LN];
	data_C[CN] = Work_Array_Rcv_Grid_B2C[gp+LN];
      }
    }
  }

  /* freeing of arrays */
  free(Work_Array_Snd_Grid_B2C);
  free(Work_Array_Rcv_Grid_B2C);
}





void Density_Grid_Copy_B2D(double **Density_Grid_B0)
{
  static int firsttime=1;
  int DN,BN,LN,spin,i,gp,NN_S,NN_R;
  double *Work_Array_Snd_Grid_B2D;
  double *Work_Array_Rcv_Grid_B2D;
  int numprocs,myid,tag=999,ID,IDS,IDR;
  MPI_Status stat;
  MPI_Request request;
  MPI_Status *stat_send;
  MPI_Status *stat_recv;
  MPI_Request *request_send;
  MPI_Request *request_recv;

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /* allocation of arrays */
  
  Work_Array_Snd_Grid_B2D = (double*)malloc(sizeof(double)*GP_B2D_S[NN_B2D_S]*(SpinP_switch+1)); 
  Work_Array_Rcv_Grid_B2D = (double*)malloc(sizeof(double)*GP_B2D_R[NN_B2D_R]*(SpinP_switch+1)); 

  if (firsttime==1){
    PrintMemory("Set_Density_Grid: Work_Array_Snd_Grid_B2D",
		sizeof(double)*GP_B2D_S[NN_B2D_S]*(SpinP_switch+1), NULL);
    PrintMemory("Set_Density_Grid: Work_Array_Rcv_Grid_B2D",
		sizeof(double)*GP_B2D_R[NN_B2D_R]*(SpinP_switch+1), NULL);
    firsttime = 0;
  }

  /******************************************************
             MPI: from the partitions B to D
  ******************************************************/

  request_send = malloc(sizeof(MPI_Request)*NN_B2D_S);
  request_recv = malloc(sizeof(MPI_Request)*NN_B2D_R);
  stat_send = malloc(sizeof(MPI_Status)*NN_B2D_S);
  stat_recv = malloc(sizeof(MPI_Status)*NN_B2D_R);

  NN_S = 0;
  NN_R = 0;

  /* MPI_Irecv */

  for (ID=0; ID<NN_B2D_R; ID++){

    IDR = ID_NN_B2D_R[ID];
    gp = GP_B2D_R[ID];

    if (IDR!=myid){ 
      MPI_Irecv( &Work_Array_Rcv_Grid_B2D[(SpinP_switch+1)*gp], Num_Rcv_Grid_B2D[IDR]*(SpinP_switch+1),
                 MPI_DOUBLE, IDR, tag, mpi_comm_level1, &request_recv[NN_R]);
      NN_R++;
    }
  }

  /* MPI_Isend */

  for (ID=0; ID<NN_B2D_S; ID++){

    IDS = ID_NN_B2D_S[ID];
    gp = GP_B2D_S[ID];

    /* copy Density_Grid_B0 to Work_Array_Snd_Grid_B2D */

    for (LN=0; LN<Num_Snd_Grid_B2D[IDS]; LN++){

      BN = Index_Snd_Grid_B2D[IDS][LN];

      if (SpinP_switch==0){
        Work_Array_Snd_Grid_B2D[gp+LN]       = Density_Grid_B0[0][BN];
      }
      else if (SpinP_switch==1){
        Work_Array_Snd_Grid_B2D[2*gp+2*LN+0] = Density_Grid_B0[0][BN];
        Work_Array_Snd_Grid_B2D[2*gp+2*LN+1] = Density_Grid_B0[1][BN];
      }
      else if (SpinP_switch==3){
        Work_Array_Snd_Grid_B2D[4*gp+4*LN+0] = Density_Grid_B0[0][BN];
        Work_Array_Snd_Grid_B2D[4*gp+4*LN+1] = Density_Grid_B0[1][BN];
        Work_Array_Snd_Grid_B2D[4*gp+4*LN+2] = Density_Grid_B0[2][BN];
        Work_Array_Snd_Grid_B2D[4*gp+4*LN+3] = Density_Grid_B0[3][BN];
      }
    } /* LN */        

    if (IDS!=myid){
      MPI_Isend( &Work_Array_Snd_Grid_B2D[(SpinP_switch+1)*gp], Num_Snd_Grid_B2D[IDS]*(SpinP_switch+1), 
		 MPI_DOUBLE, IDS, tag, mpi_comm_level1, &request_send[NN_S]);
      NN_S++;
    }
  }

  /* MPI_Waitall */

  if (NN_S!=0) MPI_Waitall(NN_S,request_send,stat_send);
  if (NN_R!=0) MPI_Waitall(NN_R,request_recv,stat_recv);

  free(request_send);
  free(request_recv);
  free(stat_send);
  free(stat_recv);

  /* copy Work_Array_Rcv_Grid_B2D to Density_Grid_D */

  for (ID=0; ID<NN_B2D_R; ID++){

    IDR = ID_NN_B2D_R[ID];

    if (IDR==myid){

      gp = GP_B2D_S[ID];

      for (LN=0; LN<Num_Rcv_Grid_B2D[IDR]; LN++){

	DN = Index_Rcv_Grid_B2D[IDR][LN];

	if (SpinP_switch==0){
	  Density_Grid_D[0][DN] = Work_Array_Snd_Grid_B2D[gp+LN];
	}     
	else if (SpinP_switch==1){
	  Density_Grid_D[0][DN] = Work_Array_Snd_Grid_B2D[2*gp+2*LN+0];
	  Density_Grid_D[1][DN] = Work_Array_Snd_Grid_B2D[2*gp+2*LN+1];
	}     
	else if (SpinP_switch==3){
	  Density_Grid_D[0][DN] = Work_Array_Snd_Grid_B2D[4*gp+4*LN+0];
	  Density_Grid_D[1][DN] = Work_Array_Snd_Grid_B2D[4*gp+4*LN+1];
	  Density_Grid_D[2][DN] = Work_Array_Snd_Grid_B2D[4*gp+4*LN+2];
	  Density_Grid_D[3][DN] = Work_Array_Snd_Grid_B2D[4*gp+4*LN+3];
	}
      } /* LN */   

    }

    else{

      gp = GP_B2D_R[ID];

      for (LN=0; LN<Num_Rcv_Grid_B2D[IDR]; LN++){

	DN = Index_Rcv_Grid_B2D[IDR][LN];

	if (SpinP_switch==0){
	  Density_Grid_D[0][DN] = Work_Array_Rcv_Grid_B2D[gp+LN];
	}     
	else if (SpinP_switch==1){
	  Density_Grid_D[0][DN] = Work_Array_Rcv_Grid_B2D[2*gp+2*LN+0];
	  Density_Grid_D[1][DN] = Work_Array_Rcv_Grid_B2D[2*gp+2*LN+1];
	}     
	else if (SpinP_switch==3){
	  Density_Grid_D[0][DN] = Work_Array_Rcv_Grid_B2D[4*gp+4*LN+0];
	  Density_Grid_D[1][DN] = Work_Array_Rcv_Grid_B2D[4*gp+4*LN+1];
	  Density_Grid_D[2][DN] = Work_Array_Rcv_Grid_B2D[4*gp+4*LN+2];
	  Density_Grid_D[3][DN] = Work_Array_Rcv_Grid_B2D[4*gp+4*LN+3];
	}
      }

    }
  }

  /* if (SpinP_switch==0), copy Density_Grid[0] to Density_Grid[1] */

  if (SpinP_switch==0){
    for (BN=0; BN<My_NumGridB_AB; BN++){
      Density_Grid_B0[1][BN] = Density_Grid_B0[0][BN]; 
    }

    for (DN=0; DN<My_NumGridD; DN++){
      Density_Grid_D[1][DN] = Density_Grid_D[0][DN]; 
    }
  }

  /* freeing of arrays */
  free(Work_Array_Snd_Grid_B2D);
  free(Work_Array_Rcv_Grid_B2D);
}


void diagonalize_nc_density(double **Density_Grid_B0)
{
  int BN,DN,Mc_AN,Gc_AN,Nog,GRc;
  double Re11,Re22,Re12,Im12;
  double phi[2],theta[2],sit,cot,sip,cop;
  double d1,d2,d3,x,y,z,Cxyz[4];
  double Nup[2],Ndown[2];
  /* for OpenMP */
  int OMPID,Nthrds;

  /************************************
     Density_Grid in the partition B
  ************************************/

#pragma omp parallel shared(Density_Grid_B0,My_NumGridB_AB) private(OMPID,Nthrds,BN,Re11,Re22,Re12,Im12,Nup,Ndown,theta,phi) default(none)
  {

    /* get info. on OpenMP */ 

    OMPID = omp_get_thread_num();
    Nthrds = omp_get_num_threads();

    for (BN=OMPID; BN<My_NumGridB_AB; BN+=Nthrds){

      Re11 = Density_Grid_B0[0][BN];
      Re22 = Density_Grid_B0[1][BN];
      Re12 = Density_Grid_B0[2][BN];
      Im12 = Density_Grid_B0[3][BN];

      EulerAngle_Spin( 1, Re11, Re22, Re12, Im12, Re12, -Im12, Nup, Ndown, theta, phi );

      Density_Grid_B0[0][BN] = Nup[0];
      Density_Grid_B0[1][BN] = Ndown[0];
      Density_Grid_B0[2][BN] = theta[0];
      Density_Grid_B0[3][BN] = phi[0];
    }

#pragma omp flush(Density_Grid_B)

  } /* #pragma omp parallel */

  /************************************
     Density_Grid in the partition D
  ************************************/

#pragma omp parallel shared(Density_Grid_D,My_NumGridD) private(OMPID,Nthrds,DN,Re11,Re22,Re12,Im12,Nup,Ndown,theta,phi) default(none)
  {

    /* get info. on OpenMP */ 

    OMPID = omp_get_thread_num();
    Nthrds = omp_get_num_threads();

    for (DN=OMPID; DN<My_NumGridD; DN+=Nthrds){

      Re11 = Density_Grid_D[0][DN];
      Re22 = Density_Grid_D[1][DN];
      Re12 = Density_Grid_D[2][DN];
      Im12 = Density_Grid_D[3][DN];

      EulerAngle_Spin( 1, Re11, Re22, Re12, Im12, Re12, -Im12, Nup, Ndown, theta, phi );

      Density_Grid_D[0][DN] = Nup[0];
      Density_Grid_D[1][DN] = Ndown[0];
      Density_Grid_D[2][DN] = theta[0];
      Density_Grid_D[3][DN] = phi[0];
    }

#pragma omp flush(Density_Grid_D)

  } /* #pragma omp parallel */

}

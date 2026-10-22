/**********************************************************************
  Set_Nonlocal.c:

     Set_Nonlocal.c is a subroutine to calculate matrix elements
     and the derivatives of nonlocal potentials in the momentum space.

  Log of Set_Nonlocal.c:

     15/Sep/2002  Released by T.Ozaki

***********************************************************************/

#define  measure_time   0

/* thread-private scratch length of the device spherical Bessel evaluation;
   matches asize_lmax+10 of the host Spherical_Bessel.c */
#define  SETNL_SB_ARR   40

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#include <sys/types.h>
#include <sys/times.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <unistd.h>
#include <omp.h>
#include "openmx_common.h"
#include "openmx_env.h"
#include "openmx_core_radial.h"
#include "openmx_core_pair_bessel.h"
#include "openmx_gpu_workers.h"
#include "openmx_gpu_task.h"
#include "mpi.h"
#include "openmx_projector_exchange.h"
#include "openmx_solver_workspace.h"
#include <omp.h>
#include "hip_runtime_compat.h"
#include "set_hip_default_device_from_local_rank.h"

#ifdef kcomp
static double NLRF_BesselF(const OpenMXReciprocalViews *reciprocal, int Gensi, int L, int so, double R);
static void Nonlocal0(const OpenMXAtomicViews *atomic, const OpenMXReciprocalViews *reciprocal, double *****HNL, double ******DS_NL, const OpenMXElectronicViews *electronic, OpenMXSolverInvocation *scratch);
static void Multiply_DS_NL(const OpenMXAtomicViews *atomic, int Mc_AN, int Mj_AN, int k, int kl,
                           int Cwan, int Hwan, int wakg, const OpenMXSolverComplexTensor *NLH, const OpenMXElectronicViews *electronic);
#else
inline double NLRF_BesselF(const OpenMXReciprocalViews *reciprocal, int Gensi, int L, int so, double R);
inline void Nonlocal0(const OpenMXAtomicViews *atomic, const OpenMXReciprocalViews *reciprocal, double *****HNL, double ******DS_NL, const OpenMXElectronicViews *electronic, OpenMXSolverInvocation *scratch);
inline void Multiply_DS_NL(const OpenMXAtomicViews *atomic, int Mc_AN, int Mj_AN, int k, int kl,
                           int Cwan, int Hwan, int wakg, const OpenMXSolverComplexTensor *NLH, const OpenMXElectronicViews *electronic);
#endif

static void SetNonlocal_AbortWithMessage(const char *message)
{
  fprintf(stderr,"%s\n",message);
  fflush(stderr);
  if (openmx_gpu_worker_device()>=0) exit(EXIT_FAILURE);
  MPI_Abort(mpi_comm_level1,1);
}

static size_t SetNonlocal_CheckedAddCount(size_t a, size_t b, const char *label)
{
  if (b > SIZE_MAX - a){
    char msg[512];
    snprintf(msg,sizeof(msg),"Count overflow in Set_Nonlocal.c: %s",label);
    SetNonlocal_AbortWithMessage(msg);
  }

  return a + b;
}

static size_t SetNonlocal_CheckedMulCount(size_t a, size_t b, const char *label)
{
  if (a != 0 && b > SIZE_MAX/a){
    char msg[512];
    snprintf(msg,sizeof(msg),"Dimension overflow in Set_Nonlocal.c: %s",label);
    SetNonlocal_AbortWithMessage(msg);
  }

  return a * b;
}

static void *SetNonlocal_MallocArray(size_t count, size_t elem_size, const char *label)
{
  size_t bytes;
  void *ptr;

  bytes = SetNonlocal_CheckedMulCount(count,elem_size,label);
  ptr = malloc((bytes == 0) ? 1 : bytes);

  if (ptr == NULL){
    char msg[512];
    snprintf(msg,sizeof(msg),
             "Out of memory in Set_Nonlocal.c: %s (%zu bytes)",label,bytes);
    SetNonlocal_AbortWithMessage(msg);
  }

  return ptr;
}

static int SetNonlocal_SizeTToInt(size_t value, const char *label)
{
  if (value > (size_t)INT_MAX){
    char msg[512];
    snprintf(msg,sizeof(msg),
             "Value exceeds INT_MAX in Set_Nonlocal.c: %s (%zu)",label,value);
    SetNonlocal_AbortWithMessage(msg);
  }

  return (int)value;
}

static long int SetNonlocal_SizeTToLongInt(size_t value, const char *label)
{
  if (value > (size_t)LONG_MAX){
    char msg[512];
    snprintf(msg,sizeof(msg),
             "Value exceeds LONG_MAX in Set_Nonlocal.c: %s (%zu)",label,value);
    SetNonlocal_AbortWithMessage(msg);
  }

  return (long int)value;
}

static size_t SetNonlocal_ArenaOff(size_t *pos, size_t bytes)
{
  size_t off = *pos;

  *pos = (off + bytes + 511U) & ~(size_t)511U;
  return off;
}

/* one shot; the caller shrinks the request or falls back to the host */
static void *SetNonlocal_ArenaTry(size_t bytes)
{
  void *arena = NULL;

  if (hipMalloc(&arena,bytes) != hipSuccess){
    arena = NULL;
    /* Complete pending users before retrying after a transient OOM. */
    if (hipDeviceSynchronize() == hipSuccess &&
        hipMalloc(&arena,bytes) != hipSuccess) arena = NULL;
  }
  return arena;
}

static void SetNonlocal_ValidateGlobalState(const OpenMXAtomicViews *atomic, const OpenMXReciprocalViews *reciprocal)
{
  int spe,L;
  size_t component_count,tmp_count;

  if (SpeciesNum <= 0){
    SetNonlocal_AbortWithMessage("Invalid SpeciesNum in Set_Nonlocal.c.");
  }

  if (OneD_Grid <= 0){
    SetNonlocal_AbortWithMessage("OneD_Grid must be positive in Set_Nonlocal.c.");
  }

  if (Ngrid_NormK <= 0){
    SetNonlocal_AbortWithMessage("Ngrid_NormK must be positive in Set_Nonlocal.c.");
  }

  if (!isfinite(PAO_Nkmax) || PAO_Nkmax <= 0.0 ||
      PAO_Nkmax/(double)Ngrid_NormK == 0.0){
    SetNonlocal_AbortWithMessage("Invalid reciprocal grid spacing in Set_Nonlocal.c.");
  }

  if (reciprocal->nonlocal_transform.values == NULL){
    SetNonlocal_AbortWithMessage("Spe_NLRF_Bessel is NULL in Set_Nonlocal.c.");
  }

  if (List_YOUSO[19] <= 0 || List_YOUSO[24] <= 0 || List_YOUSO[25] < 0 || List_YOUSO[30] < 0){
    SetNonlocal_AbortWithMessage("Invalid List_YOUSO dimensions in Set_Nonlocal.c.");
  }

  for (spe=0; spe<SpeciesNum; spe++){

    if (atomic->species.basis_lmax[spe] < 0 || atomic->species.basis_lmax[spe] > List_YOUSO[25]){
      char msg[512];
      snprintf(msg,sizeof(msg),
               "Spe_MaxL_Basis exceeds List_YOUSO[25] in Set_Nonlocal.c for species %d.",spe);
      SetNonlocal_AbortWithMessage(msg);
    }

    if (atomic->species.orbitals[spe] < 0 || List_YOUSO[7] < atomic->species.orbitals[spe]){
      char msg[512];
      snprintf(msg,sizeof(msg),
               "Spe_Total_NO exceeds List_YOUSO[7] in Set_Nonlocal.c for species %d.",spe);
      SetNonlocal_AbortWithMessage(msg);
    }

    for (L=0; L<=atomic->species.basis_lmax[spe]; L++){
      if (atomic->species.basis[(size_t)(spe)*atomic->species.angular_channels+(L)] < 0 || List_YOUSO[24] < atomic->species.basis[(size_t)(spe)*atomic->species.angular_channels+(L)]){
        char msg[512];
        snprintf(msg,sizeof(msg),
                 "Spe_Num_Basis exceeds List_YOUSO[24] in Set_Nonlocal.c for species %d, L=%d.",
                 spe,L);
        SetNonlocal_AbortWithMessage(msg);
      }
    }

    if (atomic->species.relativistic[spe] < 0 || 1 < atomic->species.relativistic[spe]){
      char msg[512];
      snprintf(msg,sizeof(msg),
               "Unsupported VPS_j_dependency in Set_Nonlocal.c for species %d.",spe);
      SetNonlocal_AbortWithMessage(msg);
    }

    if (atomic->species.radial_projectors[spe] < 0 || List_YOUSO[19] <= atomic->species.radial_projectors[spe]){
      char msg[512];
      snprintf(msg,sizeof(msg),
               "Spe_Num_RVPS exceeds buffer bounds in Set_Nonlocal.c for species %d.",spe);
      SetNonlocal_AbortWithMessage(msg);
    }

    component_count = 0;
    for (L=1; L<=atomic->species.radial_projectors[spe]; L++){
      if (openmx_atomic_projector_angular(atomic,spe)[L] < 0 || List_YOUSO[30] < openmx_atomic_projector_angular(atomic,spe)[L]){
        char msg[512];
        snprintf(msg,sizeof(msg),
                 "Spe_VPS_List exceeds List_YOUSO[30] in Set_Nonlocal.c for species %d, projector %d.",
                 spe,L);
        SetNonlocal_AbortWithMessage(msg);
      }

      if (atomic->species.relativistic[spe] == 1 && 3 < openmx_atomic_projector_angular(atomic,spe)[L]){
        char msg[512];
        snprintf(msg,sizeof(msg),
                 "j-dependent nonlocal projectors above f are unsupported in Set_Nonlocal.c for species %d.",
                 spe);
        SetNonlocal_AbortWithMessage(msg);
      }

      tmp_count = SetNonlocal_CheckedMulCount(2u,(size_t)openmx_atomic_projector_angular(atomic,spe)[L],
                                              "projector magnetic components");
      tmp_count = SetNonlocal_CheckedAddCount(tmp_count,1u,"projector magnetic components");
      component_count = SetNonlocal_CheckedAddCount(component_count,tmp_count,
                                                    "Spe_Total_VPS_Pro");
    }

    if (component_count != (size_t)atomic->species.projector_components[spe]){
      char msg[512];
      snprintf(msg,sizeof(msg),
               "Spe_Total_VPS_Pro mismatch in Set_Nonlocal.c for species %d.",spe);
      SetNonlocal_AbortWithMessage(msg);
    }
  }
}

/* OPENMX_BAND_PROFILE=1: one SETNLPROF line per rank and call on stderr
   (dsnl = the DS_NL integrals, halo = the received-atom products with their
   communication, mult = the local products) */
static int SetNonlocal_ProfileEnabled(void)
{
  const char *value = openmx_env_get("OPENMX_BAND_PROFILE");
  return value != NULL && atoi(value) != 0;
}

static int SetNonlocal_gpu_used = 0;

int Set_Nonlocal_Used_GPU(void)
{
  return SetNonlocal_gpu_used;
}

static int SetNonlocalUseOpenMP(void)
{
  /* opt out with OPENMX_SETNL_OPENMP=0 (falls back to the OpenMP CPU path) */
  const char *value = openmx_env_get("OPENMX_SETNL_OPENMP");

  if (value != NULL && atoi(value) == 0) return 0;
  return scf_eigen_lib_flag == GPUSOLVER;
}

#include "openmx_nonlocal_cpu.h"
#include "openmx_integral_pairs.h"

#pragma omp declare target
static void SetNL_capacity_bessel(double x,int lmax,double *sbv,double *dsbv)
{
    double tsb[SETNL_SB_ARR];int n,m,nmax;
    double invx,vsb0,vsb1,vsb2,vsbi=0,j0,j1,sf,tmp,si,co,ix;
              nmax = lmax + (int)(1.5*x) + 10;
              if (nmax<30) nmax = 30;

              if (0.0 < x){

                invx = 1.0/x;
                vsb0 = 0.0;
                vsb1 = 1.0e-14;

                for ( n=nmax-1; (lmax+2)<n; n-- ){
                  vsb2 = (2.0*(double)n + 1.0)*invx*vsb1 - vsb0;
                  if (0.0<(vsb2-1.0e+250)){
                    tmp = 1.0/vsb2;
                    vsb1 *= tmp;
                    vsb2 = 1.0;
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
                  tsb[n-1] = (2.0*(double)n + 1.0)*invx*tsb[n] - tsb[n+1];
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
                j0 = si*ix;
                j1 = si*ix*ix - co*ix;

                if (fabs(tsb[1])<fabs(tsb[0])) sf = j0/tsb[0];
                else                           sf = j1/tsb[1];

                for ( n=0; n<=lmax+1; n++ ){
                  sbv[n] = tsb[n]*sf;
                }

                dsbv[0] = co*ix - si*ix*ix;
                for ( n=1; n<=lmax; n++ ){
                  dsbv[n] = ( (double)n*sbv[n-1] - ((double)n+1.0)*sbv[n+1] )/(2.0*(double)n + 1.0);
                }
              }

              else{
                for ( n=0; n<=lmax+1; n++ ){
                  sbv[n] = 0.0;
                }
                sbv[0] = 1.0;

                dsbv[0] = 0.0;
                for ( n=1; n<=lmax; n++ ){
                  dsbv[n] = ( (double)n*sbv[n-1] - ((double)n+1.0)*sbv[n+1] )/(2.0*(double)n + 1.0);
                }
              }

}
#pragma omp end declare target
typedef struct {int lmax,l,diag,rf,nlrf;double radius;} SetNLCapacityRow;
typedef struct {int grid;double step;} SetNLCapacityParameters;
static void SetNL_capacity_kernel(size_t first,size_t rows,const void *parameters,void *const v[],int gpu)
{
    (void)first;const SetNLCapacityParameters p=*(const SetNLCapacityParameters*)parameters;
    const int grid=p.grid;const double step=p.step;
    const SetNLCapacityRow *row=v[0];const double *norm=v[1],*rf=v[2],*nlrf=v[3];double *out=v[4],*deriv=v[5];
#pragma omp target teams distribute parallel for is_device_ptr(row,norm,rf,nlrf,out,deriv) if(gpu)
    for(size_t r=0;r<rows;r++) {
        const SetNLCapacityRow q=row[r];double sum=0,gradient=0;

        for(int i=0;i<grid;i++) {
            double sb[SETNL_SB_ARR],dsb[SETNL_SB_ARR];
            SetNL_capacity_bessel(norm[i]*q.radius,q.lmax,sb,dsb);
            double factor=(i==0 || i==grid-1?.5:1)*step*norm[i]*norm[i]*rf[q.rf+i];
            sum+=(factor*sb[q.l])*nlrf[q.nlrf+i];
            gradient+=(factor*norm[i]*dsb[q.l])*nlrf[q.nlrf+i];
        }
        out[r]=sum;deriv[r]=q.diag?0:gradient;
    }
}

/*
   Two-phase evaluation of DS_NL:

     (1) every radial Fourier integral of every (atom pair, so, LL,
         basis x projector) combination is reduced in one batched OpenMP
         kernel (chunked over pairs to bound device memory), and

     (2) the angular assembly (spherical harmonics, Gaunt coefficients and
         the complex-to-real transforms), which has no useful GPU shape,
         runs OpenMP-parallel over the pairs like the CPU path.

   The previous implementation walked the pairs serially on the host and
   launched one small kernel per (pair, so) with a full re-upload of the
   Bessel tables, which left all but one host thread idle.

   All transient device memory is claimed through one hipMalloc arena so
   that an exhausted device (e.g. many MPI ranks sharing one GPU) never
   aborts the run: hipMalloc reports an error, the Bessel chunk is shrunk
   and retried, and if even the smallest footprint does not fit the
   function returns 0 and the caller runs the original host path instead.
   Returns 1 when DS_NL was evaluated here.
*/
static int SetNonlocal_CalcDSNL_Part(const OpenMXAtomicViews *atomic, const OpenMXReciprocalViews *reciprocal, double ******DS_NL,
                                         double *Normk_grid,
                                         int basis_l_dim,
                                         int basis_m_dim,
                                         int rvps_dim,
                                         int rvps_sum_dim,
                                         int proj_m_dim,
                                         int grid_dim,
                                         int max_Lmax_Four_Int,
                                         int max_sph_rows,
                                         const int *OneD2Mc_AN,
                                         const int *OneD2h_AN,
                                         int OneD_Nloop,
                                         double grid_h, int profile_rank)
{
  const int nsp = SpeciesNum;
  int w,cw,hw,p;
  int run_max_nLL;
  int total_rows;
  size_t rf_sp_stride,nlrf_sp_stride,rf_all_elems,nlrf_all_elems;
  size_t max_combo,lookup_stride;
  size_t spair_count,row_scan,sph_stride,chunk_cap_elems;
  int chunk_pairs;
  double *RF_all,*NLRF_all;
  int *rf_base_sp,*nlrf_base_sp;
  int *combo_count_sp,*nLL_sp,*combo_rf_off_sp,*combo_nlrf_off_sp,*combo_lookup_sp;
  int *pair_Cwan,*pair_Hwan,*pair_spair,*pair_diag,*pair_nLL;
  double *pair_r;
  size_t *pair_row_base;
  int *row_pair,*row_sph_off,*row_rf_base,*row_nlrf_base;
  double *sums0_all,*sums1_all;
  unsigned char *arena;
  size_t arena_bytes;
  size_t o_normk,o_rf,o_nlrf,o_row_pair,o_row_sph,o_row_rfb,o_row_nlb;
  size_t o_pair_diag,o_pair_r,o_pair_nLL,o_sums0,o_sums1,o_sphb,o_sphbp;
  int device_ok;
  static _Thread_local int fail_notice_done = 0;
  int maxL0,maxL1;
  size_t gaunt_d1,gaunt_d2,gaunt_d3,gaunt_d4,gaunt_elems;
  double *gaunt_tab;
  double t_mark0,t_mark1,t_setup,t_sphb,t_acc,t_ang;


  t_setup = t_sphb = t_acc = t_ang = 0.0;
  if (measure_time) dtime(&t_mark0);

  /**********************************************************************
     per-species radial tables: the Fourier transforms of the basis
     functions and of the nonlocal projectors on the common k grid
  **********************************************************************/

  rf_sp_stride = SetNonlocal_CheckedMulCount(
                   SetNonlocal_CheckedMulCount((size_t)basis_l_dim,(size_t)List_YOUSO[24],
                                               "OpenMP RF species stride"),
                   (size_t)grid_dim,"OpenMP RF species stride");
  rf_all_elems = SetNonlocal_CheckedMulCount((size_t)nsp,rf_sp_stride,"OpenMP RF_all elements");
  (void)SetNonlocal_SizeTToInt(rf_all_elems,"OpenMP RF_all elements");
  RF_all = (double*)SetNonlocal_MallocArray(rf_all_elems,sizeof(double),"OpenMP RF_all");
  memset(RF_all,0,rf_all_elems*sizeof(double));

  nlrf_sp_stride = SetNonlocal_CheckedMulCount((size_t)rvps_sum_dim,(size_t)grid_dim,
                                               "OpenMP NLRF species stride");
  nlrf_all_elems = SetNonlocal_CheckedMulCount((size_t)(2*nsp),nlrf_sp_stride,
                                               "OpenMP NLRF_all elements");
  (void)SetNonlocal_SizeTToInt(nlrf_all_elems,"OpenMP NLRF_all elements");
  NLRF_all = (double*)SetNonlocal_MallocArray(nlrf_all_elems,sizeof(double),"OpenMP NLRF_all");
  memset(NLRF_all,0,nlrf_all_elems*sizeof(double));

  rf_base_sp = (int*)SetNonlocal_MallocArray((size_t)nsp,sizeof(int),"OpenMP rf_base_sp");
  nlrf_base_sp = (int*)SetNonlocal_MallocArray((size_t)(2*nsp),sizeof(int),"OpenMP nlrf_base_sp");
  for (w=0; w<nsp; w++){
    rf_base_sp[w] = SetNonlocal_SizeTToInt((size_t)w*rf_sp_stride,"OpenMP rf base");
    nlrf_base_sp[w] = SetNonlocal_SizeTToInt((size_t)w*nlrf_sp_stride,"OpenMP nlrf base");
    nlrf_base_sp[nsp+w] = SetNonlocal_SizeTToInt(((size_t)nsp+w)*nlrf_sp_stride,"OpenMP nlrf base");
  }

#pragma omp parallel for if(!omp_in_parallel()) schedule(dynamic)
  for (w=0; w<nsp; w++){
    int L0,Mul0,so,L,i;

    for (L0=0; L0<=atomic->species.basis_lmax[w]; L0++){
      for (Mul0=0; Mul0<atomic->species.basis[(size_t)(w)*atomic->species.angular_channels+(L0)]; Mul0++){
        size_t rf_offset = (size_t)w*rf_sp_stride + (size_t)(L0*List_YOUSO[24] + Mul0)*grid_dim;

        for (i=0; i<grid_dim; i++){
          RF_all[rf_offset + i] = RF_BesselF(reciprocal, w,L0,Mul0,Normk_grid[i]);
        }
      }
    }

    for (so=0; so<=atomic->species.relativistic[w]; so++){
      for (L=1; L<=atomic->species.radial_projectors[w]; L++){
        size_t nlrf_offset = ((size_t)so*nsp + w)*nlrf_sp_stride + (size_t)L*grid_dim;

        for (i=0; i<grid_dim; i++){
          NLRF_all[nlrf_offset + i] = NLRF_BesselF(reciprocal, w,L,so,Normk_grid[i]);
        }
      }
    }
  }

  /**********************************************************************
     per-species-pair (basis, projector) combination tables and the
     angular momentum cutoff of the Fourier integrals
  **********************************************************************/

  spair_count = SetNonlocal_CheckedMulCount((size_t)nsp,(size_t)nsp,"OpenMP species pairs");
  max_combo = SetNonlocal_CheckedMulCount(
                SetNonlocal_CheckedMulCount((size_t)basis_l_dim,(size_t)List_YOUSO[24],
                                            "OpenMP combo bound"),
                (size_t)rvps_dim,"OpenMP combo bound");
  lookup_stride = SetNonlocal_CheckedMulCount(
                    SetNonlocal_CheckedMulCount((size_t)basis_l_dim,(size_t)List_YOUSO[24],
                                                "OpenMP lookup stride"),
                    (size_t)rvps_sum_dim,"OpenMP lookup stride");

  combo_count_sp = (int*)SetNonlocal_MallocArray(spair_count,sizeof(int),"OpenMP combo_count_sp");
  nLL_sp = (int*)SetNonlocal_MallocArray(spair_count,sizeof(int),"OpenMP nLL_sp");
  combo_rf_off_sp = (int*)SetNonlocal_MallocArray(
                       SetNonlocal_CheckedMulCount(spair_count,max_combo,"OpenMP combo offsets"),
                       sizeof(int),"OpenMP combo_rf_off_sp");
  combo_nlrf_off_sp = (int*)SetNonlocal_MallocArray(
                         SetNonlocal_CheckedMulCount(spair_count,max_combo,"OpenMP combo offsets"),
                         sizeof(int),"OpenMP combo_nlrf_off_sp");
  combo_lookup_sp = (int*)SetNonlocal_MallocArray(
                       SetNonlocal_CheckedMulCount(spair_count,lookup_stride,"OpenMP combo lookup"),
                       sizeof(int),"OpenMP combo_lookup_sp");

  for (cw=0; cw<nsp; cw++){
    for (hw=0; hw<nsp; hw++){
      size_t spair = (size_t)cw*nsp + hw;
      int *lookup = combo_lookup_sp + spair*lookup_stride;
      int *rf_off = combo_rf_off_sp + spair*max_combo;
      int *nlrf_off = combo_nlrf_off_sp + spair*max_combo;
      int L0,Mul0,L,Lmax,Lmax_Four_Int;
      int combo = 0;
      size_t li;

      for (li=0; li<lookup_stride; li++){
        lookup[li] = -1;
      }

      for (L0=0; L0<=atomic->species.basis_lmax[cw]; L0++){
        for (Mul0=0; Mul0<atomic->species.basis[(size_t)(cw)*atomic->species.angular_channels+(L0)]; Mul0++){
          for (L=1; L<=atomic->species.radial_projectors[hw]; L++){
            lookup[(size_t)(L0*List_YOUSO[24] + Mul0)*rvps_sum_dim + L] = combo;
            rf_off[combo] = (L0*List_YOUSO[24] + Mul0)*grid_dim;
            nlrf_off[combo] = L*grid_dim;
            combo++;
          }
        }
      }
      combo_count_sp[spair] = combo;

      Lmax = -10;
      for (L=1; L<=atomic->species.radial_projectors[hw]; L++){
        if (Lmax<openmx_atomic_projector_angular(atomic,hw)[L]) Lmax = openmx_atomic_projector_angular(atomic,hw)[L];
      }
      if (atomic->species.basis_lmax[cw]<Lmax)
        Lmax_Four_Int = 2*Lmax;
      else
        Lmax_Four_Int = 2*atomic->species.basis_lmax[cw];

      if (max_Lmax_Four_Int < Lmax_Four_Int){
        SetNonlocal_AbortWithMessage("Lmax_Four_Int exceeds its bound in Set_Nonlocal.c.");
      }
      nLL_sp[spair] = Lmax_Four_Int + 1;
    }
  }

  /**********************************************************************
     enumerate the radial-integral rows: one row per
     (pair, so, LL, basis x projector combination)
  **********************************************************************/

  pair_Cwan = (int*)SetNonlocal_MallocArray((size_t)OneD_Nloop,sizeof(int),"OpenMP pair_Cwan");
  pair_Hwan = (int*)SetNonlocal_MallocArray((size_t)OneD_Nloop,sizeof(int),"OpenMP pair_Hwan");
  pair_spair = (int*)SetNonlocal_MallocArray((size_t)OneD_Nloop,sizeof(int),"OpenMP pair_spair");
  pair_diag = (int*)SetNonlocal_MallocArray((size_t)OneD_Nloop,sizeof(int),"OpenMP pair_diag");
  pair_nLL = (int*)SetNonlocal_MallocArray((size_t)OneD_Nloop,sizeof(int),"OpenMP pair_nLL");
  pair_r = (double*)SetNonlocal_MallocArray((size_t)OneD_Nloop,sizeof(double),"OpenMP pair_r");
  pair_row_base = (size_t*)SetNonlocal_MallocArray((size_t)OneD_Nloop+1u,sizeof(size_t),
                                                   "OpenMP pair_row_base");

  run_max_nLL = 1;
  row_scan = 0;
  for (p=0; p<OneD_Nloop; p++){
    int Mc_AN = OneD2Mc_AN[p];
    int h_AN = OneD2h_AN[p];
    int Gc_AN = M2G[Mc_AN];
    int Cwan = WhatSpecies[Gc_AN];
    int Gh_AN = natn[Gc_AN][h_AN];
    int Rnh = ncn[Gc_AN][h_AN];
    int Hwan = WhatSpecies[Gh_AN];
    size_t spair = (size_t)Cwan*nsp + Hwan;
    size_t socnt = (size_t)atomic->species.relativistic[Hwan] + 1u;
    double dx,dy,dz,r;
    double S_coordinate[3];

    dx = Gxyz[Gh_AN][1] + atv[Rnh][1] - Gxyz[Gc_AN][1];
    dy = Gxyz[Gh_AN][2] + atv[Rnh][2] - Gxyz[Gc_AN][2];
    dz = Gxyz[Gh_AN][3] + atv[Rnh][3] - Gxyz[Gc_AN][3];
    xyz2spherical(dx,dy,dz,0.0,0.0,0.0,S_coordinate);
    r = S_coordinate[0];
    if (r<1.0e-10) r = 1.0e-10;

    pair_Cwan[p] = Cwan;
    pair_Hwan[p] = Hwan;
    pair_spair[p] = SetNonlocal_SizeTToInt(spair,"OpenMP species pair index");
    pair_diag[p] = (h_AN==0);
    pair_nLL[p] = (0<combo_count_sp[spair]) ? nLL_sp[spair] : 0;
    pair_r[p] = r;
    pair_row_base[p] = row_scan;
    row_scan = SetNonlocal_CheckedAddCount(
                 row_scan,
                 SetNonlocal_CheckedMulCount(
                   SetNonlocal_CheckedMulCount(socnt,(size_t)nLL_sp[spair],"OpenMP pair rows"),
                   (size_t)combo_count_sp[spair],"OpenMP pair rows"),
                 "OpenMP row total");
    if (0<combo_count_sp[spair] && run_max_nLL<nLL_sp[spair]){
      run_max_nLL = nLL_sp[spair];
    }
  }
  pair_row_base[OneD_Nloop] = row_scan;
  total_rows = SetNonlocal_SizeTToInt(row_scan,"OpenMP total rows");

  /* same capacity limit as asize_lmax in the host Spherical_Bessel.c */
  if (30 < run_max_nLL){
    SetNonlocal_AbortWithMessage("Lmax exceeds the spherical Bessel capacity in Set_Nonlocal.c.");
  }

  /**********************************************************************
     tabulate the Gaunt coefficients once: the angular loops below would
     otherwise evaluate the factorial-sum Wigner-3j formula tens of
     millions of times for a handful of distinct (L0,M0,L1,M1,LL) tuples
     (the spherical-harmonic index m is fixed to M0-M1 by the selection
     rule, so it needs no axis of its own)
  **********************************************************************/

  maxL0 = 0;
  maxL1 = 0;
  for (w=0; w<nsp; w++){
    int L;

    if (maxL0<atomic->species.basis_lmax[w]) maxL0 = atomic->species.basis_lmax[w];
    for (L=1; L<=atomic->species.radial_projectors[w]; L++){
      if (maxL1<openmx_atomic_projector_angular(atomic,w)[L]) maxL1 = openmx_atomic_projector_angular(atomic,w)[L];
    }
  }

  gaunt_d1 = (size_t)(2*maxL0 + 1);
  gaunt_d2 = (size_t)(maxL1 + 1);
  gaunt_d3 = (size_t)(2*maxL1 + 1);
  gaunt_d4 = (size_t)run_max_nLL;
  gaunt_elems = SetNonlocal_CheckedMulCount(
                  SetNonlocal_CheckedMulCount(
                    SetNonlocal_CheckedMulCount(
                      SetNonlocal_CheckedMulCount((size_t)(maxL0 + 1),gaunt_d1,"OpenMP Gaunt table"),
                      gaunt_d2,"OpenMP Gaunt table"),
                    gaunt_d3,"OpenMP Gaunt table"),
                  gaunt_d4,"OpenMP Gaunt table");
  gaunt_tab = (double*)SetNonlocal_MallocArray(gaunt_elems,sizeof(double),"OpenMP gaunt_tab");

#pragma omp parallel for if(!omp_in_parallel()) schedule(dynamic)
  for (w=0; w<=maxL0; w++){
    int L0 = w;
    int M0,L1,M1,LL;

    for (M0=-L0; M0<=L0; M0++){
      for (L1=0; L1<=maxL1; L1++){
        for (M1=-L1; M1<=L1; M1++){
          for (LL=0; LL<run_max_nLL; LL++){
            size_t gidx = ((((size_t)L0*gaunt_d1 + (size_t)(L0+M0))*gaunt_d2 + (size_t)L1)*gaunt_d3
                           + (size_t)(L1+M1))*gaunt_d4 + (size_t)LL;
            int m = M0 - M1;

            if (abs(m)<=LL && abs(L1-LL)<=L0 && L0<=(L1+LL)){
              gaunt_tab[gidx] = Gaunt(L0,M0,L1,M1,LL,m);
            }
            else{
              gaunt_tab[gidx] = 0.0;
            }
          }
        }
      }
    }
  }

  sums0_all = (double*)SetNonlocal_MallocArray((size_t)total_rows,sizeof(double),"OpenMP sums0");
  sums1_all = (double*)SetNonlocal_MallocArray((size_t)total_rows,sizeof(double),"OpenMP sums1");

  row_pair = (int*)SetNonlocal_MallocArray((size_t)total_rows,sizeof(int),"OpenMP row_pair");
  row_sph_off = (int*)SetNonlocal_MallocArray((size_t)total_rows,sizeof(int),"OpenMP row_sph_off");
  row_rf_base = (int*)SetNonlocal_MallocArray((size_t)total_rows,sizeof(int),"OpenMP row_rf_base");
  row_nlrf_base = (int*)SetNonlocal_MallocArray((size_t)total_rows,sizeof(int),"OpenMP row_nlrf_base");

#pragma omp parallel for if(!omp_in_parallel()) schedule(static)
  for (p=0; p<OneD_Nloop; p++){
    size_t spair = (size_t)pair_spair[p];
    int socnt = atomic->species.relativistic[pair_Hwan[p]] + 1;
    int nLL = nLL_sp[spair];
    int ccount = combo_count_sp[spair];
    const int *rf_off = combo_rf_off_sp + spair*max_combo;
    const int *nlrf_off = combo_nlrf_off_sp + spair*max_combo;
    size_t row = pair_row_base[p];
    int so,LL,c;

    for (so=0; so<socnt; so++){
      for (LL=0; LL<nLL; LL++){
        for (c=0; c<ccount; c++){
          row_pair[row] = p;
          row_sph_off[row] = LL*grid_dim;
          row_rf_base[row] = rf_base_sp[pair_Cwan[p]] + rf_off[c];
          row_nlrf_base[row] = nlrf_base_sp[so*nsp + pair_Hwan[p]] + nlrf_off[c];
          row++;
        }
      }
    }
  }

  /**********************************************************************
     batched radial integrals on the device, chunked over the pairs so
     the spherical Bessel tables stay within a fixed memory budget
  **********************************************************************/

  sph_stride = SetNonlocal_CheckedMulCount((size_t)run_max_nLL,(size_t)grid_dim,
                                           "OpenMP SphB stride");

  /* Claim the whole device footprint in one arena; when it does not fit,
     shrink the Bessel chunk (more launches, same math) before giving up. */
  device_ok = 0;
  arena = NULL;
  arena_bytes = 0;
  chunk_pairs = 1;

  if (0 < total_rows && !openmx_gpu_capacity_test_recovery() && gpu_rank_device_usable()){

    size_t budget_pairs = ((size_t)96*1024*1024)/(sph_stride*sizeof(double));

    if (budget_pairs < 1u) budget_pairs = 1u;
    if ((size_t)OneD_Nloop < budget_pairs) budget_pairs = (size_t)OneD_Nloop;

    for (;;){
      size_t pos = 0;

      chunk_pairs = SetNonlocal_SizeTToInt(budget_pairs,"OpenMP chunk pairs");
      chunk_cap_elems = SetNonlocal_CheckedMulCount((size_t)chunk_pairs,sph_stride,
                                                    "OpenMP SphB chunk");

      o_normk = SetNonlocal_ArenaOff(&pos,sizeof(double)*(size_t)grid_dim);
      o_rf = SetNonlocal_ArenaOff(&pos,sizeof(double)*rf_all_elems);
      o_nlrf = SetNonlocal_ArenaOff(&pos,sizeof(double)*nlrf_all_elems);
      o_row_pair = SetNonlocal_ArenaOff(&pos,sizeof(int)*(size_t)total_rows);
      o_row_sph = SetNonlocal_ArenaOff(&pos,sizeof(int)*(size_t)total_rows);
      o_row_rfb = SetNonlocal_ArenaOff(&pos,sizeof(int)*(size_t)total_rows);
      o_row_nlb = SetNonlocal_ArenaOff(&pos,sizeof(int)*(size_t)total_rows);
      o_pair_diag = SetNonlocal_ArenaOff(&pos,sizeof(int)*(size_t)OneD_Nloop);
      o_pair_r = SetNonlocal_ArenaOff(&pos,sizeof(double)*(size_t)OneD_Nloop);
      o_pair_nLL = SetNonlocal_ArenaOff(&pos,sizeof(int)*(size_t)OneD_Nloop);
      o_sums0 = SetNonlocal_ArenaOff(&pos,sizeof(double)*(size_t)total_rows);
      o_sums1 = SetNonlocal_ArenaOff(&pos,sizeof(double)*(size_t)total_rows);
      o_sphb = SetNonlocal_ArenaOff(&pos,sizeof(double)*chunk_cap_elems);
      o_sphbp = SetNonlocal_ArenaOff(&pos,sizeof(double)*chunk_cap_elems);
      arena_bytes = pos;

      arena = (unsigned char*)SetNonlocal_ArenaTry(arena_bytes);
      if (arena != NULL) break;
      if (chunk_pairs == 1) break;

      budget_pairs = (size_t)chunk_pairs/12u;
      if (budget_pairs < 1u) budget_pairs = 1u;
    }

    if (arena == NULL && !fail_notice_done && 2<=level_stdout){
      fail_notice_done = 1;
      fprintf(stderr,
              "Set_Nonlocal: %.1f MiB of device staging unavailable; trying distributed capacity recovery.\n",
              (double)arena_bytes/(1024.0*1024.0));
      fflush(stderr);
    }
  }
  else{
    /* Empty ranks still enter the collective recovery phase below. */
    device_ok = total_rows==0;
  }

  if (measure_time){
    dtime(&t_mark1);
    t_setup = t_mark1 - t_mark0;
  }

  if (arena != NULL){
    double *d_normk = (double*)(void*)(arena + o_normk);
    double *d_rf_all = (double*)(void*)(arena + o_rf);
    double *d_nlrf_all = (double*)(void*)(arena + o_nlrf);
    int *d_row_pair = (int*)(void*)(arena + o_row_pair);
    int *d_row_sph_off = (int*)(void*)(arena + o_row_sph);
    int *d_row_rf_base = (int*)(void*)(arena + o_row_rfb);
    int *d_row_nlrf_base = (int*)(void*)(arena + o_row_nlb);
    int *d_pair_diag = (int*)(void*)(arena + o_pair_diag);
    double *d_pair_r = (double*)(void*)(arena + o_pair_r);
    int *d_pair_nLL = (int*)(void*)(arena + o_pair_nLL);
    double *d_sums0 = (double*)(void*)(arena + o_sums0);
    double *d_sums1 = (double*)(void*)(arena + o_sums1);
    double *d_SphB = (double*)(void*)(arena + o_sphb);
    double *d_SphBp = (double*)(void*)(arena + o_sphbp);
    int copy_ok;

    copy_ok =
      hipMemcpy(d_normk,Normk_grid,sizeof(double)*(size_t)grid_dim,
                hipMemcpyHostToDevice) == hipSuccess &&
      hipMemcpy(d_rf_all,RF_all,sizeof(double)*rf_all_elems,
                hipMemcpyHostToDevice) == hipSuccess &&
      hipMemcpy(d_nlrf_all,NLRF_all,sizeof(double)*nlrf_all_elems,
                hipMemcpyHostToDevice) == hipSuccess &&
      hipMemcpy(d_row_pair,row_pair,sizeof(int)*(size_t)total_rows,
                hipMemcpyHostToDevice) == hipSuccess &&
      hipMemcpy(d_row_sph_off,row_sph_off,sizeof(int)*(size_t)total_rows,
                hipMemcpyHostToDevice) == hipSuccess &&
      hipMemcpy(d_row_rf_base,row_rf_base,sizeof(int)*(size_t)total_rows,
                hipMemcpyHostToDevice) == hipSuccess &&
      hipMemcpy(d_row_nlrf_base,row_nlrf_base,sizeof(int)*(size_t)total_rows,
                hipMemcpyHostToDevice) == hipSuccess &&
      hipMemcpy(d_pair_diag,pair_diag,sizeof(int)*(size_t)OneD_Nloop,
                hipMemcpyHostToDevice) == hipSuccess &&
      hipMemcpy(d_pair_r,pair_r,sizeof(double)*(size_t)OneD_Nloop,
                hipMemcpyHostToDevice) == hipSuccess &&
      hipMemcpy(d_pair_nLL,pair_nLL,sizeof(int)*(size_t)OneD_Nloop,
                hipMemcpyHostToDevice) == hipSuccess;

    if (copy_ok){
      int p0;

      for (p0=0; p0<OneD_Nloop; p0+=chunk_pairs){
        int p1 = p0 + chunk_pairs;
        int row0,row1,pp,i,row;

        if (OneD_Nloop < p1) p1 = OneD_Nloop;
        row0 = SetNonlocal_SizeTToInt(pair_row_base[p0],"OpenMP chunk row start");
        row1 = SetNonlocal_SizeTToInt(pair_row_base[p1],"OpenMP chunk row end");
        if (row1 <= row0) continue;

        if (measure_time) dtime(&t_mark0);

        /* spherical Bessel tables of this chunk's pairs, evaluated on the
           device with the same downward recurrence, rescalings and sin/cos
           normalization as the host Spherical_Bessel.c */
#pragma omp target teams distribute parallel for collapse(2) \
        is_device_ptr(d_pair_nLL,d_normk,d_pair_r,d_SphB,d_SphBp)
        for (pp=p0; pp<p1; pp++){
          for (i=0; i<grid_dim; i++){
            int lmax = d_pair_nLL[pp] - 1;

            if (0 <= lmax){
              double x = d_normk[i]*d_pair_r[pp];
              size_t sph_base = (size_t)(pp - p0)*sph_stride + (size_t)i;
              double tsb[SETNL_SB_ARR];
              double sbv[SETNL_SB_ARR];
              double dsbv[SETNL_SB_ARR];
              int n,m,nmax;
              double invx,vsb0,vsb1,vsb2,vsbi;
              double j0,j1,sf,tmp,si,co,ix;

              nmax = lmax + (int)(1.5*x) + 10;
              if (nmax<30) nmax = 30;

              if (0.0 < x){

                invx = 1.0/x;
                vsb0 = 0.0;
                vsb1 = 1.0e-14;

                for ( n=nmax-1; (lmax+2)<n; n-- ){
                  vsb2 = (2.0*(double)n + 1.0)*invx*vsb1 - vsb0;
                  if (0.0<(vsb2-1.0e+250)){
                    tmp = 1.0/vsb2;
                    vsb1 *= tmp;
                    vsb2 = 1.0;
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
                  tsb[n-1] = (2.0*(double)n + 1.0)*invx*tsb[n] - tsb[n+1];
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
                j0 = si*ix;
                j1 = si*ix*ix - co*ix;

                if (fabs(tsb[1])<fabs(tsb[0])) sf = j0/tsb[0];
                else                           sf = j1/tsb[1];

                for ( n=0; n<=lmax+1; n++ ){
                  sbv[n] = tsb[n]*sf;
                }

                dsbv[0] = co*ix - si*ix*ix;
                for ( n=1; n<=lmax; n++ ){
                  dsbv[n] = ( (double)n*sbv[n-1] - ((double)n+1.0)*sbv[n+1] )/(2.0*(double)n + 1.0);
                }
              }

              else{
                for ( n=0; n<=lmax+1; n++ ){
                  sbv[n] = 0.0;
                }
                sbv[0] = 1.0;

                dsbv[0] = 0.0;
                for ( n=1; n<=lmax; n++ ){
                  dsbv[n] = ( (double)n*sbv[n-1] - ((double)n+1.0)*sbv[n+1] )/(2.0*(double)n + 1.0);
                }
              }

              for ( n=0; n<=lmax; n++ ){
                d_SphB[sph_base + (size_t)n*grid_dim] = sbv[n];
                d_SphBp[sph_base + (size_t)n*grid_dim] = dsbv[n];
              }
            }
          }
        }

        if (measure_time){
          dtime(&t_mark1);
          t_sphb += t_mark1 - t_mark0;
          t_mark0 = t_mark1;
        }

#pragma omp target teams distribute parallel for \
        is_device_ptr(d_row_pair,d_row_sph_off,d_row_rf_base,d_row_nlrf_base, \
                      d_pair_diag,d_normk,d_rf_all,d_nlrf_all,d_SphB,d_SphBp, \
                      d_sums0,d_sums1)
        for (row=row0; row<row1; row++){
          int pp2 = d_row_pair[row];
          size_t sph_base = (size_t)(pp2 - p0)*sph_stride + (size_t)d_row_sph_off[row];
          size_t rfb = (size_t)d_row_rf_base[row];
          size_t nlb = (size_t)d_row_nlrf_base[row];
          double local_sum0 = 0.0;
          double local_sumr = 0.0;
          int ii;

          for (ii=0; ii<grid_dim; ii++){
            double coe0,Normk,tmp0,tmp1,tmp2,Bessel_Pro1;

            if (ii==0 || ii==(grid_dim-1)) coe0 = 0.50;
            else                           coe0 = 1.00;

            Normk = d_normk[ii];
            tmp0 = coe0*grid_h*Normk*Normk*d_rf_all[rfb + ii];
            tmp1 = tmp0*d_SphB[sph_base + ii];
            tmp2 = tmp0*Normk*d_SphBp[sph_base + ii];
            Bessel_Pro1 = d_nlrf_all[nlb + ii];
            local_sum0 += tmp1*Bessel_Pro1;
            local_sumr += tmp2*Bessel_Pro1;
          }

          d_sums0[row] = local_sum0;
          d_sums1[row] = d_pair_diag[pp2] ? 0.0 : local_sumr;
        }

        if (measure_time){
          dtime(&t_mark1);
          t_acc += t_mark1 - t_mark0;
        }
      }

      copy_ok =
        hipMemcpy(sums0_all,d_sums0,sizeof(double)*(size_t)total_rows,
                  hipMemcpyDeviceToHost) == hipSuccess &&
        hipMemcpy(sums1_all,d_sums1,sizeof(double)*(size_t)total_rows,
                  hipMemcpyDeviceToHost) == hipSuccess;
    }

    hipFree(arena);
    (void)hipDeviceSynchronize();
    if (copy_ok) device_ok = 1;
  }

  if (measure_time) dtime(&t_mark0);

  /* Retry only failed owner staging. A row contains its own geometry and
     Fourier-table offsets; remote helpers never access another rank's maps. */
  SetNLCapacityRow *recovery_rows=NULL;int recovery_ready=1;
  if (!device_ok && total_rows) {
    recovery_rows=malloc((size_t)total_rows*sizeof(*recovery_rows));recovery_ready=recovery_rows!=NULL;
    for(int row=0;recovery_ready && row<total_rows;row++) {
      int pair=row_pair[row];
      recovery_rows[row]=(SetNLCapacityRow){pair_nLL[pair]-1,row_sph_off[row]/grid_dim,pair_diag[pair],row_rf_base[row],row_nlrf_base[row],pair_r[pair]};
    }
  }
  SetNLCapacityParameters parameters={grid_dim,grid_h};OpenMXGpuTask task={0};
  task.count=device_ok || !recovery_ready?0:(size_t)total_rows;task.inputs=4;task.outputs=2;
  task.kernel=SetNL_capacity_kernel;task.parameters=&parameters;task.parameter_bytes=sizeof(parameters);
  task.view[0]=(OpenMXGpuTaskView){recovery_rows,0,sizeof(*recovery_rows)};
  task.view[1]=(OpenMXGpuTaskView){Normk_grid,(size_t)grid_dim*sizeof(double),0};
  task.view[2]=(OpenMXGpuTaskView){RF_all,rf_all_elems*sizeof(double),0};
  task.view[3]=(OpenMXGpuTaskView){NLRF_all,nlrf_all_elems*sizeof(double),0};
  task.view[4]=(OpenMXGpuTaskView){sums0_all,0,sizeof(double)};task.view[5]=(OpenMXGpuTaskView){sums1_all,0,sizeof(double)};
  openmx_gpu_task_execute(mpi_comm_level1,&task,1,"nonlocal-radial-capacity");
  free(recovery_rows);device_ok=device_ok || recovery_ready;
  if (!device_ok) goto cleanup;

  /**********************************************************************
     angular assembly, OpenMP-parallel over the pairs (same structure as
     the CPU path, reading the batched radial integrals)
  **********************************************************************/

#pragma omp parallel if(!omp_in_parallel())
  {
    int OMPID,Nthrds,Nloop;
    int i,j,k,l,m;
    int Mc_AN,h_AN,Gc_AN,Cwan,Gh_AN,Hwan,Rnh,so;
    int LL,L0,L1,L,Mul0,M0,M1,num0,num1;
    int nLL_p,ccount,combo;
    size_t spair;
    const int *lookup;
    double dx,dy,dz,r;
    double S_coordinate[3];
    double siT,coT,siP,coP;
    double gant,theta,phi;
    double SH[2],dSHt[2],dSHp[2];
    double Stime_atom,Etime_atom;
    double tmp0;
    dcomplex Ctmp2,Cpow;
    dcomplex CY,CYt,CYp,CY1,CYt1,CYp1;
    dcomplex *TmpNL;
    dcomplex *TmpNLr;
    dcomplex *TmpNLt;
    dcomplex *TmpNLp;

    OpenMXNonlocalBuffers buffers;
    OpenMXNonlocalWorkspace *workspace = SetNonlocal_Workspace(atomic, 0,NULL,0,&buffers);
    TmpNL = buffers.angular[0];
    TmpNLr = buffers.angular[1];
    TmpNLt = buffers.angular[2];
    TmpNLp = buffers.angular[3];

    OMPID = omp_get_thread_num();
    Nthrds = omp_get_num_threads();

    for (Nloop=OMPID*OneD_Nloop/Nthrds; Nloop<(OMPID+1)*OneD_Nloop/Nthrds; Nloop++){

      dtime(&Stime_atom);

      Mc_AN = OneD2Mc_AN[Nloop];
      h_AN  = OneD2h_AN[Nloop];

      Gc_AN = M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];

      Gh_AN = natn[Gc_AN][h_AN];
      Rnh = ncn[Gc_AN][h_AN];
      Hwan = WhatSpecies[Gh_AN];
      const size_t *projector_offsets = buffers.projector_offsets+buffers.species_offsets[Hwan];
      const int *projector_angular = buffers.projector_angular+buffers.species_offsets[Hwan];
      const int projector_count = buffers.projector_counts[Hwan];

      dx = Gxyz[Gh_AN][1] + atv[Rnh][1] - Gxyz[Gc_AN][1];
      dy = Gxyz[Gh_AN][2] + atv[Rnh][2] - Gxyz[Gc_AN][2];
      dz = Gxyz[Gh_AN][3] + atv[Rnh][3] - Gxyz[Gc_AN][3];

      xyz2spherical(dx,dy,dz,0.0,0.0,0.0,S_coordinate);
      r     = S_coordinate[0];
      theta = S_coordinate[1];
      phi   = S_coordinate[2];

      if (r<1.0e-10) r = 1.0e-10;

      siT = sin(theta);
      coT = cos(theta);
      siP = sin(phi);
      coP = cos(phi);

      spair = (size_t)pair_spair[Nloop];
      nLL_p = nLL_sp[spair];
      ccount = combo_count_sp[spair];
      lookup = combo_lookup_sp + spair*lookup_stride;

      for (so=0; so<=atomic->species.relativistic[Hwan]; so++){

        const double *sum0_so = sums0_all + pair_row_base[Nloop] + (size_t)so*nLL_p*ccount;
        const double *sum1_so = sums1_all + pair_row_base[Nloop] + (size_t)so*nLL_p*ccount;

        for (L0=0; L0<=atomic->species.basis_lmax[Cwan]; L0++){
          for (Mul0=0; Mul0<atomic->species.basis[(size_t)(Cwan)*atomic->species.angular_channels+(L0)]; Mul0++){
            for (L=1; L<=projector_count; L++){
              L1 = projector_angular[L];
              for (M0=-L0; M0<=L0; M0++){
                for (M1=-L1; M1<=L1; M1++){
                  const size_t pair_index = openmx_core_nonlocal_pair(&buffers,projector_offsets,L0,Mul0,L0+M0,L,L1+M1);
                  TmpNL[pair_index]  = Complex(0.0,0.0);
                  TmpNLr[pair_index] = Complex(0.0,0.0);
                  TmpNLt[pair_index] = Complex(0.0,0.0);
                  TmpNLp[pair_index] = Complex(0.0,0.0);
                }
              }
            }
          }
        }

        for (LL=0; LL<nLL_p; LL++){
          for (m=-LL; m<=LL; m++){

            ComplexSH(LL,m,theta,phi,SH,dSHt,dSHp);
            SH[1]   = -SH[1];
            dSHt[1] = -dSHt[1];
            dSHp[1] = -dSHp[1];

            CY  = Complex(SH[0],SH[1]);
            CYt = Complex(dSHt[0],dSHt[1]);
            CYp = Complex(dSHp[0],dSHp[1]);

            for (L0=0; L0<=atomic->species.basis_lmax[Cwan]; L0++){
              for (Mul0=0; Mul0<atomic->species.basis[(size_t)(Cwan)*atomic->species.angular_channels+(L0)]; Mul0++){
                for (L=1; L<=projector_count; L++){

                  int Ls;

                  L1 = projector_angular[L];
                  Ls = -L0 + L1 + LL;
                  combo = lookup[(size_t)(L0*List_YOUSO[24] + Mul0)*rvps_sum_dim + L];

                  if (combo<0){
                    continue;
                  }

                  if (abs(L1-LL)<=L0 && L0<=(L1+LL)){

                    Cpow = Im_pow(-1,Ls);
                    CY1  = Cmul(Cpow,CY);
                    CYt1 = Cmul(Cpow,CYt);
                    CYp1 = Cmul(Cpow,CYp);

                    for (M0=-L0; M0<=L0; M0++){

                      M1 = M0 - m;

                      if (abs(M1)<=L1){
                        const size_t pair_index = openmx_core_nonlocal_pair(&buffers,projector_offsets,L0,Mul0,L0+M0,L,L1+M1);

                        gant = gaunt_tab[((((size_t)L0*gaunt_d1 + (size_t)(L0+M0))*gaunt_d2
                                           + (size_t)L1)*gaunt_d3 + (size_t)(L1+M1))*gaunt_d4
                                         + (size_t)LL];

                        tmp0 = gant*sum0_so[(size_t)LL*ccount + combo];
                        Ctmp2 = CRmul(CY1,tmp0);
                        TmpNL[pair_index] =
                          Cadd(TmpNL[pair_index],Ctmp2);

                        tmp0 = gant*sum1_so[(size_t)LL*ccount + combo];
                        Ctmp2 = CRmul(CY1,tmp0);
                        TmpNLr[pair_index] =
                          Cadd(TmpNLr[pair_index],Ctmp2);

                        tmp0 = gant*sum0_so[(size_t)LL*ccount + combo];
                        Ctmp2 = CRmul(CYt1,tmp0);
                        TmpNLt[pair_index] =
                          Cadd(TmpNLt[pair_index],Ctmp2);

                        tmp0 = gant*sum0_so[(size_t)LL*ccount + combo];
                        Ctmp2 = CRmul(CYp1,tmp0);
                        TmpNLp[pair_index] =
                          Cadd(TmpNLp[pair_index],Ctmp2);
                      }
                    }
                  }
                }
              }
            }
          }
        }

        const OpenMXTwoCenterGeometry geometry={r,siT,coT,siP,coP,h_AN==0};
        SetNonlocal_TransformDSNL(atomic,&buffers,Cwan,Hwan,so,Mc_AN,h_AN,DS_NL,
            &Comp2Real[0][0][0], sizeof(Comp2Real)/sizeof(Comp2Real[0]), sizeof(Comp2Real[0][0])/sizeof(Comp2Real[0][0][0]),&geometry);
      }

      dtime(&Etime_atom);
#pragma omp atomic
      time_per_atom[Gc_AN] += Etime_atom - Stime_atom;
    }

    openmx_core_nonlocal_destroy(workspace);
  } /* #pragma omp parallel */

  if (measure_time){
    int prof_myid;

    dtime(&t_mark1);
    t_ang = t_mark1 - t_mark0;
    prof_myid=profile_rank;
    printf("CalcDSNL: myid=%2d setup=%8.4f sphb=%8.4f acc=%8.4f ang=%8.4f\n",
           prof_myid,t_setup,t_sphb,t_acc,t_ang);
    fflush(stdout);
  }

 cleanup:
  free(gaunt_tab);
  free(sums1_all);
  free(sums0_all);
  free(row_nlrf_base);
  free(row_rf_base);
  free(row_sph_off);
  free(row_pair);
  free(pair_row_base);
  free(pair_r);
  free(pair_nLL);
  free(pair_diag);
  free(pair_spair);
  free(pair_Hwan);
  free(pair_Cwan);
  free(combo_lookup_sp);
  free(combo_nlrf_off_sp);
  free(combo_rf_off_sp);
  free(nLL_sp);
  free(combo_count_sp);
  free(nlrf_base_sp);
  free(rf_base_sp);
  free(NLRF_all);
  free(RF_all);

  if (device_ok && total_rows > 0) SetNonlocal_gpu_used = 1;
  return device_ok;
}


static OpenMXSolverComplexTensor SetNonlocal_HamiltonianScratch(OpenMXSolverInvocation *owner,
    int worker,int orbitals)
{
  const size_t shape[4]={3,(size_t)orbitals,(size_t)orbitals,1};
  OpenMXSolverComplexTensor view;
  char error[256];
  if (!openmx_core_solver_complex(owner,worker,"nonlocal-hamiltonian",shape,&view,error,sizeof(error)))
    OpenMX_Solver_Failure(error);
  return view;
}

typedef struct {
  const OpenMXAtomicViews *atomic;
  const OpenMXReciprocalViews *reciprocal;
  double *****hamiltonian, ******projectors;
  const OpenMXElectronicViews *electronic;
} SetNonlocalArguments;
static int SetNonlocal_Operation(OpenMXSolverInvocation *scratch,void *opaque,double *elapsed)
{
  const SetNonlocalArguments *args=opaque;
  double begin,end;
  dtime(&begin);
  SetNonlocal_gpu_used = 0;
  Nonlocal0(args->atomic,args->reciprocal,args->hamiltonian,args->projectors,args->electronic,scratch);
  dtime(&end);
  *elapsed=end-begin;
  return 1;
}
double Set_Nonlocal(const OpenMXAtomicViews *atomic,const OpenMXReciprocalViews *reciprocal,
    double *****HNL,double ******DS_NL,const OpenMXElectronicViews *electronic)
{
  SetNonlocalArguments args={atomic,reciprocal,HNL,DS_NL,electronic};
  return OpenMX_Solver_Run(SetNonlocal_Operation,&args);
}


void Nonlocal0(const OpenMXAtomicViews *atomic, const OpenMXReciprocalViews *reciprocal, double *****HNL, double ******DS_NL, const OpenMXElectronicViews *electronic, OpenMXSolverInvocation *scratch)
{
  double *****iHNL=electronic->iHNL;
  double *****iHNL0=electronic->iHNL0;

  /****************************************************
   Evaluate matrix elements of nonlocal potentials
   in the KB or Blochl separable form in momentum space.
  ****************************************************/
  static int firsttime=1;
  int i,j,kl,n,m;
  int Mc_AN,Gc_AN,h_AN,k,Cwan,Gh_AN,Hwan,so;
  int tno0,tno1,tno2,i1,j1,p,ct_AN,spin;
  int fan,jg,kg,wakg,jg0,Mj_AN0,j0;
  int basis_l_dim,basis_m_dim,rvps_dim,rvps_sum_dim,proj_m_dim;
  int grid_dim,max_Lmax_Four_Int,max_sph_rows;
  int Mj_AN,num,size1,size2;
  int DSNL_on_device;
  int *Snd_DS_NL_Size,*Rcv_DS_NL_Size;
  int t,nthreads_halo;
  OpenMXSolverComplexTensor *NLH_halo;
  int Original_Mc_AN,po;
  double rcutA,rcutB,rcut,dmp;
  double grid_h;
  double time1,time2,time3;
  double stime,etime;
  double *Normk_grid;
  double *tmp_array;
  double *tmp_array2;
  double TStime,TEtime;
  double Stime_atom,Etime_atom;
  int numprocs,myid,tag=999,ID,IDS,IDR;

  MPI_Status stat;
  MPI_Request request;

  /* for OpenMP */
  int OneD_Nloop;
  const int *OneD2Mc_AN,*OneD2h_AN;
  const char *bessel_cache_option = openmx_env_get("OPENMX_SETNL_BESSEL_CACHE_MAX");
  const int bessel_cache_entries = bessel_cache_option ? atoi(bessel_cache_option) : 128;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);
  OpenMXProjectorTransfer projector_transfer = OpenMX_TransferBegin(mpi_comm_level1,numprocs,sizeof(double));
  int *send_progress=projector_transfer.buffers.sent;
  int *receive_progress=projector_transfer.buffers.received;


  SetNonlocal_ValidateGlobalState(atomic, reciprocal);

  basis_l_dim = SetNonlocal_SizeTToInt(SetNonlocal_CheckedAddCount((size_t)List_YOUSO[25],1u,
                                                                   "basis_l_dim"),
                                       "basis_l_dim");
  basis_m_dim = SetNonlocal_SizeTToInt(
                  SetNonlocal_CheckedAddCount(
                    SetNonlocal_CheckedMulCount(2u,(size_t)basis_l_dim,"basis_m_dim"),
                    1u,"basis_m_dim"),
                  "basis_m_dim");
  rvps_dim = SetNonlocal_SizeTToInt((size_t)List_YOUSO[19],"rvps_dim");
  rvps_sum_dim = SetNonlocal_SizeTToInt(SetNonlocal_CheckedAddCount((size_t)rvps_dim,1u,
                                                                    "rvps_sum_dim"),
                                        "rvps_sum_dim");
  proj_m_dim = SetNonlocal_SizeTToInt(
                 SetNonlocal_CheckedAddCount(
                   SetNonlocal_CheckedMulCount(2u,(size_t)List_YOUSO[30],"proj_m_dim"),
                   1u,"proj_m_dim"),
                 "proj_m_dim");
  grid_dim = SetNonlocal_SizeTToInt(SetNonlocal_CheckedAddCount((size_t)OneD_Grid,1u,"grid_dim"),
                                    "grid_dim");
  max_Lmax_Four_Int = 2*((List_YOUSO[25] < List_YOUSO[30]) ? List_YOUSO[30] : List_YOUSO[25]);
  max_sph_rows = SetNonlocal_SizeTToInt(SetNonlocal_CheckedAddCount((size_t)max_Lmax_Four_Int,3u,
                                                                    "max_sph_rows"),
                                        "max_sph_rows");
  grid_h = (PAO_Nkmax - Radial_kmin)/(double)OneD_Grid;
  Normk_grid = OpenMX_Solver_Real(scratch,"nonlocal-wavevectors",(size_t)grid_dim);
  for (i=0; i<grid_dim; i++){
    Normk_grid[i] = Radial_kmin + (double)i*grid_h;
  }

  /****************************************************
             calculate the size of arrays:
  ****************************************************/

  /*
  Spe_Num_RVPS[Hwan]     ->  YOUSO35*YOUSO34
  Spe_VPS_List[Hwan][L]  ->  0,0,0,.., 1,1,1,.., 2,2,2,..,
  */

  {
    size_t size_SumNL0,size_TmpNL,projector_columns=1,projector_slots=1;
    for (int species=0; species<SpeciesNum; ++species) {
      if (projector_columns < (size_t)atomic->species.projector_components[species])
        projector_columns = (size_t)atomic->species.projector_components[species];
      if (projector_slots < (size_t)atomic->species.radial_projectors[species]+1)
        projector_slots = (size_t)atomic->species.radial_projectors[species]+1;
    }
    size_SumNL0 = SetNonlocal_CheckedMulCount((size_t)basis_l_dim,(size_t)List_YOUSO[24],
                                              "SumNL0 radial channels");
    size_TmpNL = SetNonlocal_CheckedMulCount(size_SumNL0,(size_t)basis_l_dim,
                                             "TmpNL packed orbitals");
    size_TmpNL = SetNonlocal_CheckedMulCount(size_TmpNL,projector_columns,
                                             "TmpNL packed projector components");
    size_SumNL0 = SetNonlocal_CheckedMulCount(size_SumNL0,projector_slots,"SumNL0 elements");

    /* PrintMemory */
    if (firsttime) {
      PrintMemory("Set_Nonlocal: SumNL0",
                  SetNonlocal_SizeTToLongInt(
                    SetNonlocal_CheckedMulCount(size_SumNL0,sizeof(double),
                                                "PrintMemory SumNL0 bytes"),
                    "PrintMemory SumNL0 bytes"),NULL);
      PrintMemory("Set_Nonlocal: SumNLr0",
                  SetNonlocal_SizeTToLongInt(
                    SetNonlocal_CheckedMulCount(size_SumNL0,sizeof(double),
                                                "PrintMemory SumNLr0 bytes"),
                    "PrintMemory SumNLr0 bytes"),NULL);
      PrintMemory("Set_Nonlocal: TmpNL",
                  SetNonlocal_SizeTToLongInt(
                    SetNonlocal_CheckedMulCount(size_TmpNL,sizeof(dcomplex),
                                                "PrintMemory TmpNL bytes"),
                    "PrintMemory TmpNL bytes"),NULL);
      PrintMemory("Set_Nonlocal: TmpNLr",
                  SetNonlocal_SizeTToLongInt(
                    SetNonlocal_CheckedMulCount(size_TmpNL,sizeof(dcomplex),
                                                "PrintMemory TmpNLr bytes"),
                    "PrintMemory TmpNLr bytes"),NULL);
      PrintMemory("Set_Nonlocal: TmpNLt",
                  SetNonlocal_SizeTToLongInt(
                    SetNonlocal_CheckedMulCount(size_TmpNL,sizeof(dcomplex),
                                                "PrintMemory TmpNLt bytes"),
                    "PrintMemory TmpNLt bytes"),NULL);
      PrintMemory("Set_Nonlocal: TmpNLp",
                  SetNonlocal_SizeTToLongInt(
                    SetNonlocal_CheckedMulCount(size_TmpNL,sizeof(dcomplex),
                                                "PrintMemory TmpNLp bytes"),
                    "PrintMemory TmpNLp bytes"),NULL);
      firsttime=0;
    }
  }

  const OpenMXAtomPairInput pair_input={atomnum,Matomnum,SpeciesNum,TCpyCell+1,
      M2G,WhatSpecies,FNAN,natn,ncn,Gxyz,atv};
  OpenMXAtomPairBuffers pairs;
  OpenMXAtomPairPlan *pair_owner=OpenMX_IntegralPairs(&pair_input,&pairs);
  OneD_Nloop=pairs.count;
  OneD2Mc_AN=pairs.local;
  OneD2h_AN=pairs.neighbor;

  /****************************************************
                    calculate DS_NL
  ****************************************************/

  MPI_Barrier(mpi_comm_level1);
  if (measure_time || SetNonlocal_ProfileEnabled()) dtime(&stime);

  DSNL_on_device = 0;
  int radial_requested=SetNonlocalUseOpenMP();
  MPI_Allreduce(MPI_IN_PLACE,&radial_requested,1,MPI_INT,MPI_MIN,mpi_comm_level1);
  if (radial_requested){
    DSNL_on_device = SetNonlocal_CalcDSNL_Part(atomic, reciprocal, DS_NL,Normk_grid,
                                 basis_l_dim,basis_m_dim,rvps_dim,rvps_sum_dim,proj_m_dim,
                                 grid_dim,max_Lmax_Four_Int,max_sph_rows,
                                 OneD2Mc_AN,OneD2h_AN,OneD_Nloop,grid_h,myid);
  }

  if (!DSNL_on_device)
    SetNonlocal_CalcDSNL_CPU(atomic, reciprocal, DS_NL,Normk_grid,grid_dim,grid_h,
        OneD2Mc_AN,OneD2h_AN,OneD_Nloop,bessel_cache_entries);

  if (measure_time || SetNonlocal_ProfileEnabled()){
    dtime(&etime);
    time1 = etime - stime;
  }

  /*******************************************************
   *******************************************************
     multiplying overlap integrals WITH COMMUNICATION

     MPI: communicate only for k=0
     DS_NL
  *******************************************************
  *******************************************************/

  MPI_Barrier(mpi_comm_level1);
  if (measure_time || SetNonlocal_ProfileEnabled()) dtime(&stime);

  /* allocation of arrays */

  /* the products of the received blocks are threaded over the local atoms:
     one NLH per thread */
  nthreads_halo = omp_get_max_threads();
  NLH_halo=OpenMX_Solver_Bytes(scratch,"nonlocal-halo-views",(size_t)nthreads_halo,sizeof(*NLH_halo));
  for (t=0; t<nthreads_halo; ++t)
    NLH_halo[t]=SetNonlocal_HamiltonianScratch(scratch,t,List_YOUSO[7]);

  Snd_DS_NL_Size = projector_transfer.buffers.send_sizes;
  Rcv_DS_NL_Size = projector_transfer.buffers.receive_sizes;

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
        size_t send_count;

        send_count = 0;
        n = send_progress[IDS];

        Mc_AN = Snd_MAN[IDS][n];
        Gc_AN = Snd_GAN[IDS][n];
        Cwan = WhatSpecies[Gc_AN];
        tno1 = atomic->species.orbitals[Cwan];

        for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
          Gh_AN = natn[Gc_AN][h_AN];
          Hwan = WhatSpecies[Gh_AN];
	  tno2 = atomic->species.projector_components[Hwan];
          send_count =
            SetNonlocal_CheckedAddCount(
              send_count,
              SetNonlocal_CheckedMulCount(
                SetNonlocal_CheckedMulCount((size_t)(atomic->species.relativistic[Hwan]+1),(size_t)tno1,
                                            "DS_NL send count"),
                (size_t)tno2,"DS_NL send count"),
              "DS_NL send count");
        }

        size1 = SetNonlocal_SizeTToInt(send_count,"Snd_DS_NL_Size[IDS]");
        Snd_DS_NL_Size[IDS] = size1;
        MPI_Isend(&size1, 1, MPI_INT, IDS, tag, mpi_comm_level1, &request);
      }
      else{
        Snd_DS_NL_Size[IDS] = 0;
      }

      /* receiving of the size of the data */

      if ( 0<(F_Rcv_Num[IDR]-receive_progress[IDR]) ){
        MPI_Recv(&size2, 1, MPI_INT, IDR, tag, mpi_comm_level1, &stat);
        if (size2 < 0){
          SetNonlocal_AbortWithMessage("Received a negative DS_NL block size in Set_Nonlocal.c.");
        }
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

        if (num != size1){
          SetNonlocal_AbortWithMessage("Packed DS_NL block size mismatch in Set_Nonlocal.c.");
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

        if (num != size2){
          SetNonlocal_AbortWithMessage("Unpacked DS_NL block size mismatch in Set_Nonlocal.c.");
        }
	/* free tmp_array2 */


	/*****************************************************************
                           multiplying overlap integrals
	*****************************************************************/

        n = receive_progress[IDR];
        jg = Rcv_GAN[IDR][n];

#pragma omp parallel for schedule(dynamic,4) private(Mc_AN,Gc_AN,Cwan,fan,rcutA,rcutB,rcut,j0,jg0,Mj_AN0,po,j,Hwan,m,n,k,kg,wakg,kl,dmp,p,i1,j1,Stime_atom,Etime_atom)
        for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

          OpenMXSolverComplexTensor NLHt = NLH_halo[omp_get_thread_num()];

	  dtime(&Stime_atom);

          Gc_AN = M2G[Mc_AN];
          Cwan = WhatSpecies[Gc_AN];
          fan = FNAN[Gc_AN];
          rcutA = atomic->species.cutoff[Cwan];

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
              rcutB = atomic->species.cutoff[Hwan];
              rcut = rcutA + rcutB;

	      for (m=0; m<atomic->species.orbitals[Cwan]; m++){
		for (n=0; n<atomic->species.orbitals[Hwan]; n++){
		  openmx_solver_complex_plane_row(&NLHt,0,m)[n].r = 0.0;     /* <up|VNL|up> */
		  openmx_solver_complex_plane_row(&NLHt,1,m)[n].r = 0.0;     /* <dn|VNL|dn> */
		  openmx_solver_complex_plane_row(&NLHt,2,m)[n].r = 0.0;     /* <up|VNL|dn> */
		  openmx_solver_complex_plane_row(&NLHt,0,m)[n].i = 0.0;
		  openmx_solver_complex_plane_row(&NLHt,1,m)[n].i = 0.0;
		  openmx_solver_complex_plane_row(&NLHt,2,m)[n].i = 0.0;
		}
	      }

              for (k=0; k<=fan; k++){

                kg = natn[Gc_AN][k];
                wakg = WhatSpecies[kg];
                kl = RMI1[Mc_AN][j][k];

                if (0<=kl){

                  Multiply_DS_NL(atomic, Mc_AN, Matomnum+1, k, kl, Cwan, Hwan, wakg, &NLHt, electronic);

		} /* if (0<=kl) */

	      } /* k */

              /****************************************************
                      adding NLH to HNL

                 HNL[0] and iHNL[0] for up-up
                 HNL[1] and iHNL[1] for dn-dn
                 HNL[2] and iHNL[2] for up-dn
	      ****************************************************/

              dmp = dampingF(rcut,Dis[Gc_AN][j]);

	      for (p=0; p<List_YOUSO[5]; p++){
		for (i1=0; i1<atomic->species.orbitals[Cwan]; i1++){
		  for (j1=0; j1<atomic->species.orbitals[Hwan]; j1++){

		    HNL[p][Mc_AN][j][i1][j1] = dmp*openmx_solver_complex_plane_row(&NLHt,p,i1)[j1].r*F_NL_flag;

		    if (SO_switch==1){
		      iHNL[ p][Mc_AN][j][i1][j1] = dmp*openmx_solver_complex_plane_row(&NLHt,p,i1)[j1].i*F_NL_flag;
		      iHNL0[p][Mc_AN][j][i1][j1] = dmp*openmx_solver_complex_plane_row(&NLHt,p,i1)[j1].i*F_NL_flag;
		    }
		  }
		}
	      }

	    } /* if (po==1) */
	  } /* j0 */

          dtime(&Etime_atom);
          time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

	} /* Mc_AN */

        /********************************************
            increment of receive_progress[IDR]
	********************************************/

        receive_progress[IDR]++;

      }

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

  /* The same per-worker tensor is reused by the local products below. */

  if (measure_time || SetNonlocal_ProfileEnabled()){
    dtime(&etime);
    time2 = etime - stime;
  }

  /*******************************************************
   *******************************************************
     multiplying overlap integrals WITHOUT COMMUNICATION
  *******************************************************
  *******************************************************/

  MPI_Barrier(mpi_comm_level1);
  if (measure_time || SetNonlocal_ProfileEnabled()) dtime(&stime);

  /* allocation of arrays */

#pragma omp parallel shared(Matomnum,time_per_atom,HNL,iHNL,iHNL0,F_NL_flag,List_YOUSO,Dis,SpinP_switch,DS_NL,RMI1,F_G2M,natn,FNAN,WhatSpecies,M2G,OneD2h_AN,OneD2Mc_AN,OneD_Nloop,SO_switch,atomic)
  {
    int OMPID,Nthrds,Nprocs,Nloop;
    int Mc_AN,j,Gc_AN,Cwan,fan,jg,i1,j1,i;
    int Mj_AN,Hwan,k,kg,wakg,kl;
    int p,m,n,L,L1,L2,L3;
    double rcutA,rcutB,rcut,sum,ene;
    double Stime_atom, Etime_atom;
    double ene_m,ene_p,dmp;
    double tmp0,tmp1,tmp2,tmp3,tmp4,tmp5,tmp6;
    double PFp,PFm;
    OpenMXSolverComplexTensor NLH;
    dcomplex sum0,sum1,sum2;

    /* allocation of arrays */

    NLH=SetNonlocal_HamiltonianScratch(scratch,omp_get_thread_num(),List_YOUSO[7]);

    /* get info. on OpenMP */

    OMPID = omp_get_thread_num();
    Nthrds = omp_get_num_threads();
    Nprocs = omp_get_num_procs();

    /* one-dimensionalized loop */

    for (Nloop=OMPID*OneD_Nloop/Nthrds; Nloop<(OMPID+1)*OneD_Nloop/Nthrds; Nloop++){

      dtime(&Stime_atom);

      /* get Mc_AN and j */

      Mc_AN = OneD2Mc_AN[Nloop];
      j     = OneD2h_AN[Nloop];

      /* set data on Mc_AN */

      Gc_AN = M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];
      fan = FNAN[Gc_AN];
      rcutA = atomic->species.cutoff[Cwan];

      /* set data on j */

      jg = natn[Gc_AN][j];
      Mj_AN = F_G2M[jg];

      if (Mj_AN<=Matomnum){

	Hwan = WhatSpecies[jg];
	rcutB = atomic->species.cutoff[Hwan];
	rcut = rcutA + rcutB;

	for (m=0; m<atomic->species.orbitals[Cwan]; m++){
	  for (n=0; n<atomic->species.orbitals[Hwan]; n++){
	    openmx_solver_complex_plane_row(&NLH,0,m)[n].r = 0.0;     /* <up|VNL|up> */
	    openmx_solver_complex_plane_row(&NLH,1,m)[n].r = 0.0;     /* <dn|VNL|dn> */
	    openmx_solver_complex_plane_row(&NLH,2,m)[n].r = 0.0;     /* <up|VNL|dn> */
	    openmx_solver_complex_plane_row(&NLH,0,m)[n].i = 0.0;
	    openmx_solver_complex_plane_row(&NLH,1,m)[n].i = 0.0;
	    openmx_solver_complex_plane_row(&NLH,2,m)[n].i = 0.0;
	  }
	}

	for (k=0; k<=fan; k++){

	  kg = natn[Gc_AN][k];
	  wakg = WhatSpecies[kg];
	  kl = RMI1[Mc_AN][j][k];

	  if (0<=kl){

	    Multiply_DS_NL(atomic, Mc_AN, Mj_AN, k, kl, Cwan, Hwan, wakg, &NLH, electronic);

	  } /* if (0<=kl) */
	} /* k */

	/****************************************************
                       adding NLH to HNL

                 HNL[0] and iHNL[0] for up-up
                 HNL[1] and iHNL[1] for dn-dn
                 HNL[2] and iHNL[2] for up-dn
	****************************************************/

	dmp = dampingF(rcut,Dis[Gc_AN][j]);

	for (p=0; p<List_YOUSO[5]; p++){
	  for (i1=0; i1<atomic->species.orbitals[Cwan]; i1++){
	    for (j1=0; j1<atomic->species.orbitals[Hwan]; j1++){

	      HNL[p][Mc_AN][j][i1][j1] = dmp*openmx_solver_complex_plane_row(&NLH,p,i1)[j1].r*F_NL_flag;

	      if (SO_switch==1){
		iHNL[p][Mc_AN][j][i1][j1]  = dmp*openmx_solver_complex_plane_row(&NLH,p,i1)[j1].i*F_NL_flag;
		iHNL0[p][Mc_AN][j][i1][j1] = dmp*openmx_solver_complex_plane_row(&NLH,p,i1)[j1].i*F_NL_flag;
	      }
	    }
	  }
	}

      } /* if (Mj_AN<=Matomnum) */

      dtime(&Etime_atom);
#pragma omp atomic
      time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

    } /* Nloop */

    /* freeing of array */

#pragma omp barrier
#pragma omp flush(HNL,iHNL,iHNL0)

  } /* #pragma omp parallel */

  /****************************************************
  freeing of arrays:
  ****************************************************/

  openmx_core_atom_pairs_destroy(pair_owner);


  if (measure_time || SetNonlocal_ProfileEnabled()){
    dtime(&etime);
    time3 = etime - stime;
  }

  if (measure_time){
    printf("Set_Nonlocal: myid=%2d time1=%10.5f time2=%10.5f time3=%10.5f\n",
           myid,time1,time2,time3);
  }
  if (SetNonlocal_ProfileEnabled()){
    fprintf(stderr, "SETNLPROF id=%d dsnl=%.3f halo=%.3f mult=%.3f\n", myid, time1, time2, time3);
    fflush(stderr);
  }

  /****************************************************
   MPI_Barrier
  ****************************************************/

  MPI_Barrier(mpi_comm_level1);
  OpenMX_TransferEnd(&projector_transfer);
}


void Multiply_DS_NL(const OpenMXAtomicViews *atomic, int Mc_AN, int Mj_AN, int k, int kl,
                    int Cwan, int Hwan, int wakg, const OpenMXSolverComplexTensor *NLH, const OpenMXElectronicViews *electronic)
{
  double ******DS_NL=electronic->DS_NL;

  int m,n,L,L1,L2,L3,spin,target_spin;
  int GA_AN,GB_AN;
  double ene_core_hole[2][20];
  double sum,sumMul[2],ene,PFp,PFm,ene_p,ene_m;
  double tmp0,tmp1,tmp2,tmp3,tmp4,tmp5,tmp6;
  dcomplex sum0,sum1,sum2;

  /****************************************************
              l-dependent non-local part
  ****************************************************/

  if (atomic->species.relativistic[wakg]==0){

    for (m=0; m<atomic->species.orbitals[Cwan]; m++){
      for (n=0; n<atomic->species.orbitals[Hwan]; n++){

	sum = 0.0;
	L = 0;

	for (L1=1; L1<=atomic->species.radial_projectors[wakg]; L1++){

	  ene = openmx_radial_row3(&atomic->projector_energies,0,wakg)[L1-1];
	  L2 = 2*openmx_atomic_projector_angular(atomic,wakg)[L1];

	  for (L3=0; L3<=L2; L3++){

	    sum += ene*DS_NL[0][0][Mc_AN][k][m][L]*DS_NL[0][0][Mj_AN][kl][n][L];
	    L++;
	  }

	} /* L1 */

	openmx_solver_complex_plane_row(NLH,0,m)[n].r += sum;    /* <up|VNL|up> */
	openmx_solver_complex_plane_row(NLH,1,m)[n].r += sum;    /* <dn|VNL|dn> */

      }
    }

  } /* if */

  /****************************************************
               j-dependent non-local part
  ****************************************************/

  else if (atomic->species.relativistic[wakg]==1){

    for (m=0; m<atomic->species.orbitals[Cwan]; m++){
      for (n=0; n<atomic->species.orbitals[Hwan]; n++){

	sum0 = Complex(0.0,0.0);
	sum1 = Complex(0.0,0.0);
	sum2 = Complex(0.0,0.0);

	L = 0;
	for (L1=1; L1<=atomic->species.radial_projectors[wakg]; L1++){

	  ene_p = openmx_radial_row3(&atomic->projector_energies,0,wakg)[L1-1];
	  ene_m = openmx_radial_row3(&atomic->projector_energies,1,wakg)[L1-1];

	  if      (openmx_atomic_projector_angular(atomic,wakg)[L1]==0) { L2=0; PFp=1.0;     PFm=0.0;     }
	  else if (openmx_atomic_projector_angular(atomic,wakg)[L1]==1) { L2=2; PFp=2.0/3.0; PFm=1.0/3.0; }
	  else if (openmx_atomic_projector_angular(atomic,wakg)[L1]==2) { L2=4; PFp=3.0/5.0; PFm=2.0/5.0; }
	  else if (openmx_atomic_projector_angular(atomic,wakg)[L1]==3) { L2=6; PFp=4.0/7.0; PFm=3.0/7.0; }
          else{
            char msg[512];
            snprintf(msg,sizeof(msg),
                     "Unsupported j-dependent projector angular momentum in Set_Nonlocal.c: species=%d L=%d.",
                     wakg,openmx_atomic_projector_angular(atomic,wakg)[L1]);
            SetNonlocal_AbortWithMessage(msg);
          }

	  /****************************************************
                  off-diagonal contribution on up-dn
                  for spin non-collinear
	  ****************************************************/

	  if (SpinP_switch==3){

	    /***************
                   p
	    ***************/

	    if (L2==2){

	      /* real contribution of l+1/2 to off-diagonal up-down matrix */
	      sum2.r +=
		 ene_p/3.0*DS_NL[0][0][Mc_AN][k][m][L  ]*DS_NL[0][0][Mj_AN][kl][n][L+2]
		-ene_p/3.0*DS_NL[0][0][Mc_AN][k][m][L+2]*DS_NL[0][0][Mj_AN][kl][n][L  ];

	      /* imaginary contribution of l+1/2 to off-diagonal up-down matrix */
	      sum2.i +=
		-ene_p/3.0*DS_NL[0][0][Mc_AN][k][m][L+1]*DS_NL[0][0][Mj_AN][kl][n][L+2]
		+ene_p/3.0*DS_NL[0][0][Mc_AN][k][m][L+2]*DS_NL[0][0][Mj_AN][kl][n][L+1];

	      /* real contribution of l-1/2 for to off-diagonal up-down matrix */
	      sum2.r -=
		 ene_m/3.0*DS_NL[1][0][Mc_AN][k][m][L  ]*DS_NL[1][0][Mj_AN][kl][n][L+2]
		-ene_m/3.0*DS_NL[1][0][Mc_AN][k][m][L+2]*DS_NL[1][0][Mj_AN][kl][n][L  ];

	      /* imaginary contribution of l-1/2 to off-diagonal up-down matrix */
	      sum2.i -=
		-ene_m/3.0*DS_NL[1][0][Mc_AN][k][m][L+1]*DS_NL[1][0][Mj_AN][kl][n][L+2]
		+ene_m/3.0*DS_NL[1][0][Mc_AN][k][m][L+2]*DS_NL[1][0][Mj_AN][kl][n][L+1];
	    }

	    /***************
                   d
	    ***************/

	    else if (L2==4){

	      /* real contribution of l+1/2 to off diagonal up-down matrix */
	      tmp0 = sqrt(3.0);
	      tmp1 = ene_p/5.0;
	      tmp2 = tmp0*tmp1;

	      sum2.r +=
		-tmp2*DS_NL[0][0][Mc_AN][k][m][L  ]*DS_NL[0][0][Mj_AN][kl][n][L+3]
		+tmp2*DS_NL[0][0][Mc_AN][k][m][L+3]*DS_NL[0][0][Mj_AN][kl][n][L  ]
		+tmp1*DS_NL[0][0][Mc_AN][k][m][L+1]*DS_NL[0][0][Mj_AN][kl][n][L+3]
		-tmp1*DS_NL[0][0][Mc_AN][k][m][L+3]*DS_NL[0][0][Mj_AN][kl][n][L+1]
		+tmp1*DS_NL[0][0][Mc_AN][k][m][L+2]*DS_NL[0][0][Mj_AN][kl][n][L+4]
		-tmp1*DS_NL[0][0][Mc_AN][k][m][L+4]*DS_NL[0][0][Mj_AN][kl][n][L+2];

	      /* imaginary contribution of l+1/2 to off-diagonal up-down matrix */

	      sum2.i +=
		 tmp2*DS_NL[0][0][Mc_AN][k][m][L  ]*DS_NL[0][0][Mj_AN][kl][n][L+4]
		-tmp2*DS_NL[0][0][Mc_AN][k][m][L+4]*DS_NL[0][0][Mj_AN][kl][n][L  ]
		+tmp1*DS_NL[0][0][Mc_AN][k][m][L+1]*DS_NL[0][0][Mj_AN][kl][n][L+4]
		-tmp1*DS_NL[0][0][Mc_AN][k][m][L+4]*DS_NL[0][0][Mj_AN][kl][n][L+1]
		-tmp1*DS_NL[0][0][Mc_AN][k][m][L+2]*DS_NL[0][0][Mj_AN][kl][n][L+3]
		+tmp1*DS_NL[0][0][Mc_AN][k][m][L+3]*DS_NL[0][0][Mj_AN][kl][n][L+2];

	      /* real contribution of l-1/2 for to off-diagonal up-down matrix */

	      tmp1 = ene_m/5.0;
	      tmp2 = tmp0*tmp1;

	      sum2.r -=
		-tmp2*DS_NL[1][0][Mc_AN][k][m][L  ]*DS_NL[1][0][Mj_AN][kl][n][L+3]
		+tmp2*DS_NL[1][0][Mc_AN][k][m][L+3]*DS_NL[1][0][Mj_AN][kl][n][L  ]
		+tmp1*DS_NL[1][0][Mc_AN][k][m][L+1]*DS_NL[1][0][Mj_AN][kl][n][L+3]
		-tmp1*DS_NL[1][0][Mc_AN][k][m][L+3]*DS_NL[1][0][Mj_AN][kl][n][L+1]
		+tmp1*DS_NL[1][0][Mc_AN][k][m][L+2]*DS_NL[1][0][Mj_AN][kl][n][L+4]
		-tmp1*DS_NL[1][0][Mc_AN][k][m][L+4]*DS_NL[1][0][Mj_AN][kl][n][L+2];

	      /* imaginary contribution of l-1/2 to off-diagonal up-down matrix */

	      sum2.i -=
		 tmp2*DS_NL[1][0][Mc_AN][k][m][L  ]*DS_NL[1][0][Mj_AN][kl][n][L+4]
		-tmp2*DS_NL[1][0][Mc_AN][k][m][L+4]*DS_NL[1][0][Mj_AN][kl][n][L  ]
		+tmp1*DS_NL[1][0][Mc_AN][k][m][L+1]*DS_NL[1][0][Mj_AN][kl][n][L+4]
		-tmp1*DS_NL[1][0][Mc_AN][k][m][L+4]*DS_NL[1][0][Mj_AN][kl][n][L+1]
		-tmp1*DS_NL[1][0][Mc_AN][k][m][L+2]*DS_NL[1][0][Mj_AN][kl][n][L+3]
		+tmp1*DS_NL[1][0][Mc_AN][k][m][L+3]*DS_NL[1][0][Mj_AN][kl][n][L+2];

	    }

	    /***************
                   f
	    ***************/

	    else if (L2==6){

	      /* real contribution of l+1/2 to off diagonal up-down matrix */

	      tmp0 = sqrt(6.0);
	      tmp1 = sqrt(3.0/2.0);
	      tmp2 = sqrt(5.0/2.0);

	      tmp3 = ene_p/7.0;
	      tmp4 = tmp1*tmp3; /* sqrt(3.0/2.0) */
	      tmp5 = tmp2*tmp3; /* sqrt(5.0/2.0) */
	      tmp6 = tmp0*tmp3; /* sqrt(6.0)     */

	      sum2.r +=
                -tmp6*DS_NL[0][0][Mc_AN][k][m][L  ]*DS_NL[0][0][Mj_AN][kl][n][L+1]
		+tmp6*DS_NL[0][0][Mc_AN][k][m][L+1]*DS_NL[0][0][Mj_AN][kl][n][L  ]
		-tmp5*DS_NL[0][0][Mc_AN][k][m][L+1]*DS_NL[0][0][Mj_AN][kl][n][L+3]
		+tmp5*DS_NL[0][0][Mc_AN][k][m][L+3]*DS_NL[0][0][Mj_AN][kl][n][L+1]
		-tmp5*DS_NL[0][0][Mc_AN][k][m][L+2]*DS_NL[0][0][Mj_AN][kl][n][L+4]
		+tmp5*DS_NL[0][0][Mc_AN][k][m][L+4]*DS_NL[0][0][Mj_AN][kl][n][L+2]
		-tmp4*DS_NL[0][0][Mc_AN][k][m][L+3]*DS_NL[0][0][Mj_AN][kl][n][L+5]
		+tmp4*DS_NL[0][0][Mc_AN][k][m][L+5]*DS_NL[0][0][Mj_AN][kl][n][L+3]
		-tmp4*DS_NL[0][0][Mc_AN][k][m][L+4]*DS_NL[0][0][Mj_AN][kl][n][L+6]
		+tmp4*DS_NL[0][0][Mc_AN][k][m][L+6]*DS_NL[0][0][Mj_AN][kl][n][L+4];

	      /* imaginary contribution of l+1/2 to off diagonal up-down matrix */

	      sum2.i +=
                 tmp6*DS_NL[0][0][Mc_AN][k][m][L  ]*DS_NL[0][0][Mj_AN][kl][n][L+2]
		-tmp6*DS_NL[0][0][Mc_AN][k][m][L+2]*DS_NL[0][0][Mj_AN][kl][n][L  ]
		+tmp5*DS_NL[0][0][Mc_AN][k][m][L+1]*DS_NL[0][0][Mj_AN][kl][n][L+4]
		-tmp5*DS_NL[0][0][Mc_AN][k][m][L+4]*DS_NL[0][0][Mj_AN][kl][n][L+1]
		-tmp5*DS_NL[0][0][Mc_AN][k][m][L+2]*DS_NL[0][0][Mj_AN][kl][n][L+3]
		+tmp5*DS_NL[0][0][Mc_AN][k][m][L+3]*DS_NL[0][0][Mj_AN][kl][n][L+2]
		+tmp4*DS_NL[0][0][Mc_AN][k][m][L+3]*DS_NL[0][0][Mj_AN][kl][n][L+6]
		-tmp4*DS_NL[0][0][Mc_AN][k][m][L+6]*DS_NL[0][0][Mj_AN][kl][n][L+3]
		-tmp4*DS_NL[0][0][Mc_AN][k][m][L+4]*DS_NL[0][0][Mj_AN][kl][n][L+5]
		+tmp4*DS_NL[0][0][Mc_AN][k][m][L+5]*DS_NL[0][0][Mj_AN][kl][n][L+4];

	      /* real contribution of l-1/2 for to diagonal up-down matrix */

	      tmp3 = ene_m/7.0;
	      tmp4 = tmp1*tmp3; /* sqrt(3.0/2.0) */
	      tmp5 = tmp2*tmp3; /* sqrt(5.0/2.0) */
	      tmp6 = tmp0*tmp3; /* sqrt(6.0)     */

	      sum2.r -=
                -tmp6*DS_NL[1][0][Mc_AN][k][m][L  ]*DS_NL[1][0][Mj_AN][kl][n][L+1]
		+tmp6*DS_NL[1][0][Mc_AN][k][m][L+1]*DS_NL[1][0][Mj_AN][kl][n][L  ]
		-tmp5*DS_NL[1][0][Mc_AN][k][m][L+1]*DS_NL[1][0][Mj_AN][kl][n][L+3]
		+tmp5*DS_NL[1][0][Mc_AN][k][m][L+3]*DS_NL[1][0][Mj_AN][kl][n][L+1]
		-tmp5*DS_NL[1][0][Mc_AN][k][m][L+2]*DS_NL[1][0][Mj_AN][kl][n][L+4]
		+tmp5*DS_NL[1][0][Mc_AN][k][m][L+4]*DS_NL[1][0][Mj_AN][kl][n][L+2]
		-tmp4*DS_NL[1][0][Mc_AN][k][m][L+3]*DS_NL[1][0][Mj_AN][kl][n][L+5]
		+tmp4*DS_NL[1][0][Mc_AN][k][m][L+5]*DS_NL[1][0][Mj_AN][kl][n][L+3]
		-tmp4*DS_NL[1][0][Mc_AN][k][m][L+4]*DS_NL[1][0][Mj_AN][kl][n][L+6]
		+tmp4*DS_NL[1][0][Mc_AN][k][m][L+6]*DS_NL[1][0][Mj_AN][kl][n][L+4];

	      /* imaginary contribution of l-1/2 to off diagonal up-down matrix */

	      sum2.i -=
                 tmp6*DS_NL[1][0][Mc_AN][k][m][L  ]*DS_NL[1][0][Mj_AN][kl][n][L+2]
		-tmp6*DS_NL[1][0][Mc_AN][k][m][L+2]*DS_NL[1][0][Mj_AN][kl][n][L  ]
		+tmp5*DS_NL[1][0][Mc_AN][k][m][L+1]*DS_NL[1][0][Mj_AN][kl][n][L+4]
		-tmp5*DS_NL[1][0][Mc_AN][k][m][L+4]*DS_NL[1][0][Mj_AN][kl][n][L+1]
		-tmp5*DS_NL[1][0][Mc_AN][k][m][L+2]*DS_NL[1][0][Mj_AN][kl][n][L+3]
		+tmp5*DS_NL[1][0][Mc_AN][k][m][L+3]*DS_NL[1][0][Mj_AN][kl][n][L+2]
		+tmp4*DS_NL[1][0][Mc_AN][k][m][L+3]*DS_NL[1][0][Mj_AN][kl][n][L+6]
		-tmp4*DS_NL[1][0][Mc_AN][k][m][L+6]*DS_NL[1][0][Mj_AN][kl][n][L+3]
		-tmp4*DS_NL[1][0][Mc_AN][k][m][L+4]*DS_NL[1][0][Mj_AN][kl][n][L+5]
		+tmp4*DS_NL[1][0][Mc_AN][k][m][L+5]*DS_NL[1][0][Mj_AN][kl][n][L+4];

	    }

	  }

	  /****************************************************
             off-diagonal contribution on up-up and dn-dn
	  ****************************************************/

	  /* p */

	  if (L2==2){

	    tmp0 =
	       ene_p/3.0*DS_NL[0][0][Mc_AN][k][m][L  ]*DS_NL[0][0][Mj_AN][kl][n][L+1]
	      -ene_p/3.0*DS_NL[0][0][Mc_AN][k][m][L+1]*DS_NL[0][0][Mj_AN][kl][n][L  ];

	    /* contribution of l+1/2 for up spin */
	    sum0.i += -tmp0;

	    /* contribution of l+1/2 for down spin */
	    sum1.i += tmp0;

	    tmp0 =
	       ene_m/3.0*DS_NL[1][0][Mc_AN][k][m][L  ]*DS_NL[1][0][Mj_AN][kl][n][L+1]
	      -ene_m/3.0*DS_NL[1][0][Mc_AN][k][m][L+1]*DS_NL[1][0][Mj_AN][kl][n][L  ];

	    /* contribution of l-1/2 for up spin */
	    sum0.i += tmp0;

	    /* contribution of l-1/2 for down spin */
	    sum1.i += -tmp0;
	  }

	  /* d */

	  else if (L2==4){

	    tmp0 =
	       ene_p*2.0/5.0*DS_NL[0][0][Mc_AN][k][m][L+1]*DS_NL[0][0][Mj_AN][kl][n][L+2]
	      -ene_p*2.0/5.0*DS_NL[0][0][Mc_AN][k][m][L+2]*DS_NL[0][0][Mj_AN][kl][n][L+1]
	      +ene_p*1.0/5.0*DS_NL[0][0][Mc_AN][k][m][L+3]*DS_NL[0][0][Mj_AN][kl][n][L+4]
	      -ene_p*1.0/5.0*DS_NL[0][0][Mc_AN][k][m][L+4]*DS_NL[0][0][Mj_AN][kl][n][L+3];

	    /* contribution of l+1/2 for up spin */
	    sum0.i += -tmp0;

	    /* contribution of l+1/2 for down spin */
	    sum1.i += tmp0;

	    tmp0 =
	       ene_m*2.0/5.0*DS_NL[1][0][Mc_AN][k][m][L+1]*DS_NL[1][0][Mj_AN][kl][n][L+2]
	      -ene_m*2.0/5.0*DS_NL[1][0][Mc_AN][k][m][L+2]*DS_NL[1][0][Mj_AN][kl][n][L+1]
	      +ene_m*1.0/5.0*DS_NL[1][0][Mc_AN][k][m][L+3]*DS_NL[1][0][Mj_AN][kl][n][L+4]
	      -ene_m*1.0/5.0*DS_NL[1][0][Mc_AN][k][m][L+4]*DS_NL[1][0][Mj_AN][kl][n][L+3];

	    /* contribution of l-1/2 for up spin */
	    sum0.i += tmp0;

	    /* contribution of l-1/2 for down spin */
	    sum1.i += -tmp0;

	  }

	  /* f */

	  else if (L2==6){

	    tmp0 =
	       ene_p*1.0/7.0*DS_NL[0][0][Mc_AN][k][m][L+1]*DS_NL[0][0][Mj_AN][kl][n][L+2]
	      -ene_p*1.0/7.0*DS_NL[0][0][Mc_AN][k][m][L+2]*DS_NL[0][0][Mj_AN][kl][n][L+1]
	      +ene_p*2.0/7.0*DS_NL[0][0][Mc_AN][k][m][L+3]*DS_NL[0][0][Mj_AN][kl][n][L+4]
	      -ene_p*2.0/7.0*DS_NL[0][0][Mc_AN][k][m][L+4]*DS_NL[0][0][Mj_AN][kl][n][L+3]
	      +ene_p*3.0/7.0*DS_NL[0][0][Mc_AN][k][m][L+5]*DS_NL[0][0][Mj_AN][kl][n][L+6]
	      -ene_p*3.0/7.0*DS_NL[0][0][Mc_AN][k][m][L+6]*DS_NL[0][0][Mj_AN][kl][n][L+5];

	    /* contribution of l+1/2 for up spin */
	    sum0.i += -tmp0;

	    /* contribution of l+1/2 for down spin */
	    sum1.i += tmp0;

	    tmp0 =
	       ene_m*1.0/7.0*DS_NL[1][0][Mc_AN][k][m][L+1]*DS_NL[1][0][Mj_AN][kl][n][L+2]
	      -ene_m*1.0/7.0*DS_NL[1][0][Mc_AN][k][m][L+2]*DS_NL[1][0][Mj_AN][kl][n][L+1]
	      +ene_m*2.0/7.0*DS_NL[1][0][Mc_AN][k][m][L+3]*DS_NL[1][0][Mj_AN][kl][n][L+4]
	      -ene_m*2.0/7.0*DS_NL[1][0][Mc_AN][k][m][L+4]*DS_NL[1][0][Mj_AN][kl][n][L+3]
	      +ene_m*3.0/7.0*DS_NL[1][0][Mc_AN][k][m][L+5]*DS_NL[1][0][Mj_AN][kl][n][L+6]
	      -ene_m*3.0/7.0*DS_NL[1][0][Mc_AN][k][m][L+6]*DS_NL[1][0][Mj_AN][kl][n][L+5];

	    /* contribution of l-1/2 for up spin */
	    sum0.i += tmp0;

	    /* contribution of l-1/2 for down spin */
	    sum1.i += -tmp0;
	  }

	  /****************************************************
                 diagonal contribution on up-up and dn-dn
	  ****************************************************/

	  for (L3=0; L3<=L2; L3++){

	    /* VNL for j=l+1/2 */
	    sum0.r += PFp*ene_p*DS_NL[0][0][Mc_AN][k][m][L]*DS_NL[0][0][Mj_AN][kl][n][L];
	    sum1.r += PFp*ene_p*DS_NL[0][0][Mc_AN][k][m][L]*DS_NL[0][0][Mj_AN][kl][n][L];

	    /* VNL for j=l-1/2 */
	    sum0.r += PFm*ene_m*DS_NL[1][0][Mc_AN][k][m][L]*DS_NL[1][0][Mj_AN][kl][n][L];
	    sum1.r += PFm*ene_m*DS_NL[1][0][Mc_AN][k][m][L]*DS_NL[1][0][Mj_AN][kl][n][L];

	    L++;
	  }

	}

	openmx_solver_complex_plane_row(NLH,0,m)[n].r += sum0.r;    /* <up|VNL|up> */
	openmx_solver_complex_plane_row(NLH,1,m)[n].r += sum1.r;    /* <dn|VNL|dn> */
	openmx_solver_complex_plane_row(NLH,2,m)[n].r += sum2.r;    /* <up|VNL|dn> */

	openmx_solver_complex_plane_row(NLH,0,m)[n].i += sum0.i;    /* <up|VNL|up> */
	openmx_solver_complex_plane_row(NLH,1,m)[n].i += sum1.i;    /* <dn|VNL|dn> */
	openmx_solver_complex_plane_row(NLH,2,m)[n].i += sum2.i;    /* <up|VNL|dn> */

      } /* n */
    } /* m */

  } /* else if */

  else{
    SetNonlocal_AbortWithMessage("Unsupported VPS_j_dependency in Multiply_DS_NL of Set_Nonlocal.c.");
  }

}





double NLRF_BesselF(const OpenMXReciprocalViews *reciprocal, int Gensi, int L, int so, double R)
{
  double result;
  if (!openmx_core_reciprocal_sample(OPENMX_RECIPROCAL_PROJECTOR,Ngrid_NormK,
        PAO_Nkmax/(double)Ngrid_NormK,openmx_radial_row4(&reciprocal->nonlocal_transform,so,Gensi,L),R,&result)) {
    SetNonlocal_AbortWithMessage("Invalid reciprocal interpolation input in NLRF_BesselF.");
    return 0.0;
  }
  return result;
}

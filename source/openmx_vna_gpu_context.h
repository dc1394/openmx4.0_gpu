#ifndef OPENMX_VNA_GPU_CONTEXT_H
#define OPENMX_VNA_GPU_CONTEXT_H
#include "openmx_core_hvna.h"

/* Local to one Set_ProExpn invocation. Layout owns the six metadata arrays;
   the backend owns numeric archives and releases them before the layout. */
typedef struct {
  int enabled, turn, turns, host_archive, num_proj, num_rvna;
  size_t device_bytes, flat_count, halo_count, bounce_count;
  float *flat_dev, *flat_host, *halo_dev, *halo_host, *bounce;
  float *flat, *halo; /* Native HIP host archives, mapped one rank turn at a time. */
  size_t *mck_off, *halo_off;
  int *mck_base, *halo_base, *l2p1;
  double *ene;
  OpenMXHvnaLayout *layout;
} SetProGpuContext;

static inline int SetPro_HVNA_LayoutCreate(SetProGpuContext *g,
    const OpenMXVnaPreparation *preparation, const OpenMXHvnaInput *input,
    char *error, size_t capacity)
{
  OpenMXHvnaBuffers views;
  g->layout = openmx_core_hvna_create(preparation,input,&views,error,capacity);
  if (!g->layout) return 0;
  g->mck_base = views.local_bases; g->mck_off = views.local_offsets;
  g->halo_base = views.halo_bases; g->halo_off = views.halo_offsets;
  g->l2p1 = views.lengths; g->ene = views.energies;
  return 1;
}

/* Invocation-owned VNA batch state, borrowed until synchronous
   completion. Native begin/end routines retain their backend memory policy. */
typedef struct {
  int enabled;
  int chunk;           /* admitted pair batch, shared by sizing and execution */
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

/* VNA2/VNA3 pair batches. */
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

#endif

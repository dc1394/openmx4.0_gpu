/**********************************************************************
  Set_Orbitals_Grid.c:

   Set_Orbitals_Grid.c is a subroutine to calculate the value of basis
   functions on each grid point.

  Log of Set_Orbitals_Grid.c:

     22/Nov/2001  Released by T.Ozaki

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "openmx_common.h"
#include "openmx_gpu_workers.h"
#include "openmx_gpu_task.h"
#include "openmx_host_memory.h"
#include <stdint.h>
#include "mpi.h"
#include <omp.h>
#include "set_hip_default_device_from_local_rank.h"
#include "orbs_otf_hip.h"
#include "openmx_precision.h"


/***********************************************************************
   GPU evaluation of the primitive (Cnt_kind==0) basis orbitals.

   The per-point math is the FP64 evaluation of the on-the-fly tiles
   (orbs_otf_hip.cu, a transcription of the inlined Get_Orbitals of the
   host loops below without trigonometry or bisection), so the tables
   equal the FP64 tiles bit for bit; it covers uncontracted bases with
   L0 <= SOG_L0MAX (the explicit real-harmonic formulas).  Anything else
   (orbital optimization, L0 >= 4 via ComplexSH) keeps the untouched host
   path.  The species' tables are those of the tiles; the per-atom
   staging is one hipMalloc arena, and any device failure falls back to
   the host loops instead of aborting the run.
   Disable with OPENMX_ORBS_GRID_GPU=0.
***********************************************************************/

#define SOG_L0MAX  OTF_L0MAX
#define SOG_MULMAX OTF_MULMAX

int Set_Orbitals_Grid_OTF_Prepare(void);

static int SOG_env_flag(const char *name, int default_value)
{
  const char *value = getenv(name);

  if (value == NULL || value[0] == '\0') return default_value;
  return atoi(value) != 0;
}

static int SOG_gpu_requested(int Cnt_kind)
{
  int w, L0;
  const char *gpu=getenv("OPENMX_GPU");
  if (scf_eigen_lib_flag!=GPUSOLVER || (gpu && atoi(gpu)==0)) return 0;

  if (Cnt_kind != 0) return 0;
  if (!SOG_env_flag("OPENMX_ORBS_GRID_GPU", 1)) return 0;

  /* the staging buffers and the device stores are float */
  if (sizeof(Type_Orbs_Grid) != sizeof(float)) return 0;

  for (w = 0; w < SpeciesNum; w++) {
    if (SOG_L0MAX < Spe_MaxL_Basis[w]) return 0;
    for (L0 = 0; L0 <= Spe_MaxL_Basis[w]; L0++) {
      if (SOG_MULMAX < Spe_Num_Basis[w][L0]) return 0;
    }
    if (Spe_Num_Mesh_PAO[w] < 6) return 0;
  }

  return 1;
}
static int SOG_gpu_eligible(int Cnt_kind)
{
    return SOG_gpu_requested(Cnt_kind) && gpu_rank_device_usable();
}


static size_t SOG_arena_off(size_t *pos, size_t bytes)
{
  size_t off = *pos;

  *pos = (off + bytes + 511U) & ~(size_t)511U;
  return off;
}

/* one shot; the caller has a host fallback */
static void *SOG_arena_try(size_t bytes)
{
  void *arena = NULL;

  if (hipMalloc(&arena, bytes) != hipSuccess) arena = NULL;
  if (arena == NULL) {
    /* once more after the frees still in flight */
    (void)hipGetLastError();
    if (hipDeviceSynchronize() == hipSuccess) (void)0;
    if (hipMalloc(&arena, bytes) != hipSuccess) arena = NULL;
    /* the failure is this caller's: a later launch check must not see it */
    if (arena == NULL) (void)hipGetLastError();
  }
  return arena;
}

/* returns 1 when the whole update ran on the device; 0 keeps the host path
   (which then rewrites every value) */
static int Set_Orbitals_Grid_GPU(void)
{
  static int table_notice_done = 0, fail_notice_done = 0, device_notice_done = 0, active_notice_done = 0;
  int myid;
  int Mc_AN, Gc_AN, Cwan, h_AN, Gh_AN, Hwan, Rnh, NO0, NO1;
  int npair_cap;
  size_t max_pts = 0, max_chi = 0, max_fpts = 0, max_fchi = 0, cnt;
  float *chi_host = NULL;
  int *nog_host = NULL, *pair_hAN = NULL;
  unsigned char *desc_host = NULL, *arena = NULL;
  OTF_Atom *atom_host;
  int *npts_host;
  size_t *ioff_host, *ooff_host, *soff_host;
  size_t pos = 0, arena_bytes, desc_bytes;
  size_t o_gla, o_cla, o_nog, o_dsc, o_atm, o_npt, o_iof, o_oof, o_sof, o_chi;
  double Stime_atom, Etime_atom;

  MPI_Comm_rank(mpi_comm_level1, &myid);

  /* nothing to evaluate on this rank */
  if (Matomnum <= 0) return 1;

  /* the species' tables on the device: those of the on-the-fly tiles,
     which keep them for this geometry */
  if (!Set_Orbitals_Grid_OTF_Prepare()) {
    if (!table_notice_done) {
      table_notice_done = 1;
      fprintf(stderr, "Set_Orbitals_Grid: device orbital tables unavailable; using the host path.\n");
      fflush(stderr);
    }
    return 0;
  }

  /* ---- per-atom staging maxima ---- */

  npair_cap = 0;
  for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
    size_t fpts = 0, fchi = 0;
    int npair = 0;

    Gc_AN = M2G[Mc_AN];
    Cwan = WhatSpecies[Gc_AN];
    NO0 = Spe_Total_NO[Cwan];

    if (max_pts < (size_t)GridN_Atom[Gc_AN]) max_pts = (size_t)GridN_Atom[Gc_AN];
    cnt = (size_t)GridN_Atom[Gc_AN] * (size_t)NO0;
    if (max_chi < cnt) max_chi = cnt;

    for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
      Gh_AN = natn[Gc_AN][h_AN];
      if (G2ID[Gh_AN] != myid && 0 < NumOLG[Mc_AN][h_AN]) {
        NO1 = Spe_Total_NO[WhatSpecies[Gh_AN]];
        fpts += (size_t)NumOLG[Mc_AN][h_AN];
        fchi += (size_t)NumOLG[Mc_AN][h_AN] * (size_t)NO1;
        npair++;
      }
    }
    if (max_fpts < fpts) max_fpts = fpts;
    if (max_fchi < fchi) max_fchi = fchi;
    if (npair_cap < npair) npair_cap = npair;
  }
  if (npair_cap < 1) npair_cap = 1;

  /* ---- one arena claim covers the staging; the item descriptors of
     OTF_Orbitals_Eval_FP64 go up as one copy of desc_host ---- */

  o_gla = SOG_arena_off(&pos, sizeof(int) * (max_pts == 0 ? 1 : max_pts));
  o_cla = SOG_arena_off(&pos, sizeof(int) * (max_pts == 0 ? 1 : max_pts));
  o_nog = SOG_arena_off(&pos, sizeof(int) * (max_fpts == 0 ? 1 : max_fpts));
  o_dsc = pos;
  o_atm = SOG_arena_off(&pos, sizeof(OTF_Atom) * (size_t)npair_cap);
  o_npt = SOG_arena_off(&pos, sizeof(int) * (size_t)npair_cap);
  o_iof = SOG_arena_off(&pos, sizeof(size_t) * (size_t)npair_cap);
  o_oof = SOG_arena_off(&pos, sizeof(size_t) * (size_t)npair_cap);
  o_sof = SOG_arena_off(&pos, sizeof(size_t) * (size_t)npair_cap);
  desc_bytes = pos - o_dsc;
  cnt = (max_chi < max_fchi ? max_fchi : max_chi);
  o_chi = SOG_arena_off(&pos, sizeof(float) * (cnt == 0 ? 1 : cnt));
  arena_bytes = pos;

  chi_host = (float*)malloc(sizeof(float) * (cnt == 0 ? 1 : cnt));
  nog_host = (int*)malloc(sizeof(int) * (max_fpts == 0 ? 1 : max_fpts));
  pair_hAN = (int*)malloc(sizeof(int) * (size_t)npair_cap);
  desc_host = (unsigned char*)calloc(desc_bytes, 1);

  if (chi_host == NULL || nog_host == NULL || pair_hAN == NULL || desc_host == NULL) goto host_fallback;

  atom_host = (OTF_Atom*)(void*)(desc_host + (o_atm - o_dsc));
  npts_host = (int*)(void*)(desc_host + (o_npt - o_dsc));
  ioff_host = (size_t*)(void*)(desc_host + (o_iof - o_dsc));
  ooff_host = (size_t*)(void*)(desc_host + (o_oof - o_dsc));
  soff_host = (size_t*)(void*)(desc_host + (o_sof - o_dsc));

  arena = (unsigned char*)SOG_arena_try(arena_bytes);
  if (arena == NULL) {
    if (!fail_notice_done && 2 <= level_stdout) {
      fail_notice_done = 1;
      fprintf(stderr,
              "Set_Orbitals_Grid: %.1f MiB of device staging unavailable; using the host path.\n",
              (double)arena_bytes / (1024.0 * 1024.0));
      fflush(stderr);
    }
    goto host_fallback;
  }

  {
    const int *d_gla = (const int*)(const void*)(arena + o_gla);
    const int *d_cla = (const int*)(const void*)(arena + o_cla);
    const int *d_nog = (const int*)(const void*)(arena + o_nog);
    const OTF_Atom *d_atm = (const OTF_Atom*)(const void*)(arena + o_atm);
    const int *d_npt = (const int*)(const void*)(arena + o_npt);
    const size_t *d_iof = (const size_t*)(const void*)(arena + o_iof);
    const size_t *d_oof = (const size_t*)(const void*)(arena + o_oof);
    const size_t *d_sof = (const size_t*)(const void*)(arena + o_sof);
    float *d_chi = (float*)(void*)(arena + o_chi);

    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {

      int npts, npair, max_nolg;
      size_t fpts, fchi;

      dtime(&Stime_atom);

      Gc_AN = M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];
      NO0 = Spe_Total_NO[Cwan];
      npts = GridN_Atom[Gc_AN];

      /* ---- part 1: the atom's own orbitals on its grid ---- */

      if (0 < npts) {

        int Nc, i;

        atom_host[0].wan = Cwan;
        atom_host[0].no = NO0;
        atom_host[0].gx = Gxyz[Gc_AN][1];
        atom_host[0].gy = Gxyz[Gc_AN][2];
        atom_host[0].gz = Gxyz[Gc_AN][3];
        atom_host[0].ax = 0.0;
        atom_host[0].ay = 0.0;
        atom_host[0].az = 0.0;
        npts_host[0] = npts;
        ioff_host[0] = 0;
        ooff_host[0] = 0;
        soff_host[0] = 0;

        if (hipMemcpy((void*)d_gla, GridListAtom[Mc_AN], sizeof(int) * (size_t)npts,
                      hipMemcpyHostToDevice) != hipSuccess ||
            hipMemcpy((void*)d_cla, CellListAtom[Mc_AN], sizeof(int) * (size_t)npts,
                      hipMemcpyHostToDevice) != hipSuccess ||
            hipMemcpy(arena + o_dsc, desc_host, desc_bytes, hipMemcpyHostToDevice) != hipSuccess ||
            OTF_Orbitals_Eval_FP64(NULL, 1, npts, d_atm, d_npt, d_iof, d_oof, d_sof, NULL,
                                   d_gla, d_cla, d_chi) != 0 ||
            hipMemcpy(chi_host, d_chi, sizeof(float) * (size_t)npts * (size_t)NO0,
                      hipMemcpyDeviceToHost) != hipSuccess) goto device_failure;

#pragma omp parallel for private(i)
        for (Nc = 0; Nc < npts; Nc++) {
          const float *src = chi_host + (size_t)Nc * (size_t)NO0;
          Type_Orbs_Grid *dst = Orbs_Grid[Mc_AN][Nc];

          for (i = 0; i < NO0; i++) dst[i] = src[i];
        }
      }

      /* ---- part 2: remote neighbours on the shared grid points (indices
         into the atom's own grid lists, still on the device) ---- */

      npair = 0;
      max_nolg = 0;
      fpts = 0;
      fchi = 0;
      for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {

        Gh_AN = natn[Gc_AN][h_AN];

        if (G2ID[Gh_AN] != myid && 0 < NumOLG[Mc_AN][h_AN]) {

          const int nolg = NumOLG[Mc_AN][h_AN];

          Rnh = ncn[Gc_AN][h_AN];
          Hwan = WhatSpecies[Gh_AN];
          NO1 = Spe_Total_NO[Hwan];

          atom_host[npair].wan = Hwan;
          atom_host[npair].no = NO1;
          atom_host[npair].gx = Gxyz[Gh_AN][1];
          atom_host[npair].gy = Gxyz[Gh_AN][2];
          atom_host[npair].gz = Gxyz[Gh_AN][3];
          atom_host[npair].ax = atv[Rnh][1];
          atom_host[npair].ay = atv[Rnh][2];
          atom_host[npair].az = atv[Rnh][3];
          npts_host[npair] = nolg;
          ioff_host[npair] = fpts;
          ooff_host[npair] = fchi;
          soff_host[npair] = 0;
          pair_hAN[npair] = h_AN;
          memcpy(nog_host + fpts, GListTAtoms1[Mc_AN][h_AN], sizeof(int) * (size_t)nolg);
          if (max_nolg < nolg) max_nolg = nolg;

          fpts += (size_t)nolg;
          fchi += (size_t)nolg * (size_t)NO1;
          npair++;
        }
      }

      if (0 < npair) {

        int p;

        if (hipMemcpy((void*)d_nog, nog_host, sizeof(int) * fpts, hipMemcpyHostToDevice) != hipSuccess ||
            hipMemcpy(arena + o_dsc, desc_host, desc_bytes, hipMemcpyHostToDevice) != hipSuccess ||
            OTF_Orbitals_Eval_FP64(NULL, npair, max_nolg, d_atm, d_npt, d_iof, d_oof, d_sof, d_nog,
                                   d_gla, d_cla, d_chi) != 0 ||
            hipMemcpy(chi_host, d_chi, sizeof(float) * fchi, hipMemcpyDeviceToHost) != hipSuccess)
          goto device_failure;

        for (p = 0; p < npair; p++) {

          const int nolg = npts_host[p];
          const float *base = chi_host + ooff_host[p];
          Type_Orbs_Grid **dst_rows = Orbs_Grid_FNAN[Mc_AN][pair_hAN[p]];
          int Nog, j;

          NO1 = atom_host[p].no;

#pragma omp parallel for private(j)
          for (Nog = 0; Nog < nolg; Nog++) {
            const float *src = base + (size_t)Nog * (size_t)NO1;
            Type_Orbs_Grid *dst = dst_rows[Nog];

            for (j = 0; j < NO1; j++) dst[j] = src[j];
          }
        }
      }

      dtime(&Etime_atom);
      time_per_atom[Gc_AN] += Etime_atom - Stime_atom;
    }
  }

  hipFree(arena);
  free(chi_host); free(nog_host); free(pair_hAN); free(desc_host);

  if (!active_notice_done) {
    active_notice_done = 1;
    if (SOG_env_flag("OPENMX_ORBS_GRID_GPU_VERBOSE", 0) && myid == Host_ID) {
      fprintf(stderr, "Set_Orbitals_Grid: GPU path active\n");
      fflush(stderr);
    }
  }
  return 1;

device_failure:
  if (!device_notice_done) {
    device_notice_done = 1;
    fprintf(stderr, "Set_Orbitals_Grid: a device call failed; using the host path.\n");
    fflush(stderr);
  }
  (void)hipGetLastError();
  hipFree(arena);

host_fallback:
  free(chi_host); free(nog_host); free(pair_hAN); free(desc_host);
  return 0;
}




/***********************************************************************
   The on-the-fly orbital tiles (orbs_otf_hip.h): the species' PAO tables,
   the cell translations and the grid frame on the device, for the grid
   integrals that evaluate the orbitals of each batch of atom pairs
   instead of holding the multi-GiB tables.  Built by the table build
   above or on first use, and released with every new set of orbitals (a
   new geometry).  OPENMX_ORBS_EVAL_PRECISION=df|fp64 selects the
   evaluation of the tiles (the tables are always FP64); the default is
   double-float on GPUs whose FP32 rate is at least 8 times their FP64
   rate.
***********************************************************************/

extern int openmx_gpu_fp32_fp64_ratio(void);

/* set when this rank cannot evaluate the orbitals on the device (an
   ineligible basis or device, tables outside the evaluator) and kept
   until the next release; a failed allocation or transfer is not, so that
   a rank short of device memory for a moment tries again at the next call
   instead of staying on the host for the rest of the geometry */
static int SOG_otf_unavailable = 0;
static int SOG_otf_fp32_enabled = -1;

static int SOG_otf_double_float(void)
{
  const char *value;

  if (!openmx_fp32_enabled()) return 0;
  value = getenv("OPENMX_ORBS_EVAL_PRECISION");

  if (value != NULL && value[0] != '\0') {
    if (strcmp(value, "fp64") == 0 || strcmp(value, "FP64") == 0 || strcmp(value, "64") == 0 ||
        strcmp(value, "double") == 0) return 0;
    return 1;
  }
  return 8 <= openmx_gpu_fp32_fp64_ratio();
}

void Set_Orbitals_Grid_OTF_Release(void)
{
  OTF_Orbitals_Release();
  SOG_otf_unavailable = 0;
  SOG_otf_fp32_enabled = -1;
}

/* 1 when the device tables are ready (built now if needed); rank-local */
int Set_Orbitals_Grid_OTF_Prepare(void)
{
  OTF_HostTables t;
  size_t *rv_off_h = NULL, *rwf_base_h = NULL;
  int *sp_mesh_h = NULL, *sp_maxl_h = NULL, *sp_nb_h = NULL;
  double *pao_rv = NULL, *pao_rwf = NULL, *atv_flat = NULL;
  size_t rv_cnt = 0, rwf_cnt = 0, r;
  int w, L0, Mul0, i, rc, ok = 0;

  if (OTF_Orbitals_Ready()) {
    if (SOG_otf_fp32_enabled == openmx_fp32_enabled()) return 1;
    Set_Orbitals_Grid_OTF_Release();
  }
  if (SOG_otf_unavailable) return 0;
  if (!SOG_gpu_eligible(0)) {
    SOG_otf_unavailable = 1;
    return 0;
  }

  /* a failed host allocation is retried by the next call */
  rv_off_h = (size_t*)malloc(sizeof(size_t) * (size_t)(SpeciesNum + 1));
  rwf_base_h = (size_t*)malloc(sizeof(size_t) * (size_t)(SpeciesNum + 1));
  sp_mesh_h = (int*)malloc(sizeof(int) * (size_t)SpeciesNum);
  sp_maxl_h = (int*)malloc(sizeof(int) * (size_t)SpeciesNum);
  sp_nb_h = (int*)malloc(sizeof(int) * (size_t)SpeciesNum * (OTF_L0MAX + 1));
  if (!rv_off_h || !rwf_base_h || !sp_mesh_h || !sp_maxl_h || !sp_nb_h) goto done;
  for (w = 0; w < SpeciesNum; w++) {
    rv_off_h[w] = rv_cnt;
    rwf_base_h[w] = rwf_cnt;
    sp_mesh_h[w] = Spe_Num_Mesh_PAO[w];
    sp_maxl_h[w] = Spe_MaxL_Basis[w];
    for (L0 = 0; L0 <= OTF_L0MAX; L0++) {
      int nb = (L0 <= Spe_MaxL_Basis[w] ? Spe_Num_Basis[w][L0] : 0);
      sp_nb_h[w * (OTF_L0MAX + 1) + L0] = nb;
      rwf_cnt += (size_t)nb * (size_t)Spe_Num_Mesh_PAO[w];
    }
    rv_cnt += (size_t)Spe_Num_Mesh_PAO[w];
  }
  rv_off_h[SpeciesNum] = rv_cnt;
  rwf_base_h[SpeciesNum] = rwf_cnt;
  pao_rv = (double*)malloc(sizeof(double) * (rv_cnt == 0 ? 1 : rv_cnt));
  pao_rwf = (double*)malloc(sizeof(double) * (rwf_cnt == 0 ? 1 : rwf_cnt));
  atv_flat = (double*)malloc(sizeof(double) * ((size_t)TCpyCell + 1) * 3);
  if (!pao_rv || !pao_rwf || !atv_flat) goto done;
  for (w = 0; w < SpeciesNum; w++) {
    size_t rpos = rwf_base_h[w];
    for (i = 0; i < Spe_Num_Mesh_PAO[w]; i++) pao_rv[rv_off_h[w] + (size_t)i] = Spe_PAO_RV[w][i];
    for (L0 = 0; L0 <= Spe_MaxL_Basis[w]; L0++)
      for (Mul0 = 0; Mul0 < Spe_Num_Basis[w][L0]; Mul0++)
        for (i = 0; i < Spe_Num_Mesh_PAO[w]; i++) pao_rwf[rpos++] = Spe_PAO_RWF[w][L0][Mul0][i];
  }
  for (r = 0; r <= (size_t)TCpyCell; r++) {
    atv_flat[3 * r + 0] = atv[r][1];
    atv_flat[3 * r + 1] = atv[r][2];
    atv_flat[3 * r + 2] = atv[r][3];
  }

  memset(&t, 0, sizeof(t));
  t.species = SpeciesNum;
  t.rv_off = rv_off_h;
  t.rwf_base = rwf_base_h;
  t.mesh = sp_mesh_h;
  t.maxl = sp_maxl_h;
  t.nb = sp_nb_h;
  t.rv = pao_rv;
  t.rwf = pao_rwf;
  t.rv_count = rv_cnt;
  t.rwf_count = rwf_cnt;
  t.atv = atv_flat;
  t.atv_rows = (size_t)TCpyCell + 1;
  t.ng23 = Ngrid2 * Ngrid3;
  t.ng3 = Ngrid3;
  for (i = 0; i < 3; i++) {
    t.g[i][0] = gtv[i + 1][1];
    t.g[i][1] = gtv[i + 1][2];
    t.g[i][2] = gtv[i + 1][3];
    t.org[i] = Grid_Origin[i + 1];
  }
  rc = OTF_Orbitals_Upload(&t, SOG_otf_double_float());
  /* tables outside the evaluator stay so until the next geometry; a
     failed device allocation or transfer (rc 0) is retried by the next
     call */
  if (rc < 0) SOG_otf_unavailable = 1;
  ok = (0 < rc);
  if (ok) SOG_otf_fp32_enabled = openmx_fp32_enabled();

done:
  free(rv_off_h); free(rwf_base_h); free(sp_mesh_h); free(sp_maxl_h); free(sp_nb_h);
  free(pao_rv); free(pao_rwf); free(atv_flat);
  return ok;
}


#pragma omp declare target
static void SOG_point_eval(int wan, int no, double gx, double gy, double gz,
                           double ax, double ay, double az, int GNc, int GRc,
                           const double *atvf, const double *rv_all, const double *rwf_all,
                           const size_t *rv_off, const size_t *rwf_base,
                           const int *sp_mesh, const int *sp_maxl, const int *sp_nb,
                           int ng23, int ng3,
                           double g11, double g12, double g13,
                           double g21, double g22, double g23,
                           double g31, double g32, double g33,
                           double org1, double org2, double org3,
                           float *out)
{
    /* Get_Grid_XYZ */
    const int n1 = GNc / ng23;
    const int n2 = (GNc - n1 * ng23) / ng3;
    const int n3 = GNc - n1 * ng23 - n2 * ng3;

    const double x = ((double)n1 * g11 + (double)n2 * g21 + (double)n3 * g31 + org1)
                     + atvf[3 * (size_t)GRc + 0] - gx - ax;
    const double y = ((double)n1 * g12 + (double)n2 * g22 + (double)n3 * g32 + org2)
                     + atvf[3 * (size_t)GRc + 1] - gy - ay;
    const double z = ((double)n1 * g13 + (double)n2 * g23 + (double)n3 * g33 + org3)
                     + atvf[3 * (size_t)GRc + 2] - gz - az;

    const int mesh = sp_mesh[wan];
    const int maxl = sp_maxl[wan];
    const double *rv = rv_all + rv_off[wan];

    double RF[SOG_L0MAX + 1][SOG_MULMAX];
    double AF[SOG_L0MAX + 1][2 * SOG_L0MAX + 1];
    double R, Q, P;
    int po = 0;
    int L0, Mul0, M0, i1;

    /* xyz2spherical, without the angles: the real harmonics below need
       only cos/sin of theta and phi, which are z/r, r1/r, x/r1, y/r1 (the
       host path goes through acos/asin and sin/cos, which agrees with this
       to the last bits of the double value) */
    double siQ, coQ, siP, coP;
    {
      const double Min_r = 10e-15;
      double dum = x * x + y * y;
      double r = sqrt(dum + z * z);
      double r1 = sqrt(dum);

      if (Min_r <= r) {
        if (r < fabs(z)) { coQ = (z < 0.0 ? -1.0 : 1.0); siQ = 0.0; }
        else { coQ = z / r; siQ = r1 / r; }
        if (Min_r <= r1) {
          if (r1 < fabs(y)) { siP = (y < 0.0 ? -1.0 : 1.0); coP = 0.0; }
          else { siP = y / r1; coP = x / r1; }
        }
        else {
          siP = 0.0;
          coP = 1.0;
        }
      }
      else {
        coQ = 0.0;
        siQ = 1.0;
        siP = 0.0;
        coP = 1.0;
      }
      R = r;
      Q = 0.0;
      P = 0.0;
    }

    /* radial spline of every (L0, Mul0) */
    if (rv[mesh - 1] < R) {
      /* outside the PAO mesh: every RF is zero, so every orbital is zero */
      po = 1;
    }
    else {

      const int below = (R < rv[0]);
      double h1, h2, h3, x1, x2, y1, y2, y12, y22;
      double dum, dum1, dum2, dum3, dum4;
      double rm = 0.0;
      size_t rrow = rwf_base[wan];
      int m;

      if (below) {
        m = 4;
        rm = rv[m];

        h1 = (m == 1) ? rv[m - 1] - rv[m + 1] : rv[m - 1] - rv[m - 2];
        h2 = rv[m] - rv[m - 1];
        h3 = (m == mesh - 1) ? rv[m - 2] - rv[m] : rv[m + 1] - rv[m];

        x1 = rm - rv[m - 1];
        x2 = rm - rv[m];
      }
      else {
        /* the first mesh index with rv[m] >= R (what the bisection of the
           host path finds): an estimate from the logarithmic mesh, then
           the exact neighbours */
        const double x0 = log(rv[0]);
        const double dx = (log(rv[mesh - 1]) - x0) / (double)(mesh - 1);
        m = (int)((log(R) - x0) / dx) + 1;
        if (m < 1) m = 1;
        if (mesh - 1 < m) m = mesh - 1;
        while (1 < m && R <= rv[m - 1]) m--;
        while (m < mesh - 1 && rv[m] < R) m++;

        h1 = (m == 1) ? rv[m - 1] - rv[m + 1] : rv[m - 1] - rv[m - 2];
        h2 = rv[m] - rv[m - 1];
        h3 = (m == mesh - 1) ? rv[m - 2] - rv[m] : rv[m + 1] - rv[m];

        x1 = R - rv[m - 1];
        x2 = R - rv[m];
      }

      y1 = x1 / h2;
      y2 = x2 / h2;
      y12 = y1 * y1;
      y22 = y2 * y2;

      dum = h1 + h2;
      dum1 = h1 / h2 / dum;
      dum2 = h2 / h1 / dum;
      dum = h2 + h3;
      dum3 = h2 / h3 / dum;
      dum4 = h3 / h2 / dum;

      for (L0 = 0; L0 <= maxl; L0++) {
        const int nb = sp_nb[wan * (SOG_L0MAX + 1) + L0];

        for (Mul0 = 0; Mul0 < nb; Mul0++) {

          const double *rw = rwf_all + rrow;
          double f1 = rw[m == 1 ? m + 1 : m - 2];
          double f2 = rw[m - 1];
          double f3 = rw[m];
          double f4 = rw[m == mesh - 1 ? m - 2 : m + 1];
          double g1, g2, f;

          rrow += (size_t)mesh;

          dum = f3 - f2;
          g1 = dum * dum1 + (f2 - f1) * dum2;
          g2 = (f4 - f3) * dum3 + dum * dum4;

          f = y22 * (3.0 * f2 + h2 * g1 + (2.0 * f2 + h2 * g1) * y2)
            + y12 * (3.0 * f3 - h2 * g2 - (2.0 * f3 - h2 * g2) * y1);

          if (below) {

            double df = 2.0 * y2 / h2 * (3.0 * f2 + h2 * g1 + (2.0 * f2 + h2 * g1) * y2)
              + y22 * (2.0 * f2 + h2 * g1) / h2
              + 2.0 * y1 / h2 * (3.0 * f3 - h2 * g2 - (2.0 * f3 - h2 * g2) * y1)
              - y12 * (2.0 * f3 - h2 * g2) / h2;
            double a, b, c, d;

            if (L0 == 0) {
              a = 0.0;
              b = 0.5 * df / rm;
              c = 0.0;
              d = f - b * rm * rm;
            }
            else if (L0 == 1) {
              a = (rm * df - f) / (2.0 * rm * rm * rm);
              b = 0.0;
              c = df - 3.0 * a * rm * rm;
              d = 0.0;
            }
            else {
              b = (3.0 * f - rm * df) / (rm * rm);
              a = (f - b * rm * rm) / (rm * rm * rm);
              c = 0.0;
              d = 0.0;
            }

            RF[L0][Mul0] = a * R * R * R + b * R * R + c * R + d;
          }
          else {
            RF[L0][Mul0] = f;
          }
        }
      }
    }

    if (po == 0) {

      /* Angular */
      /* siQ, coQ, siP, coP: from the coordinates above */
      double dum, dum1, dum2;

      for (L0 = 0; L0 <= maxl; L0++) {

        if (L0 == 0) {
          AF[0][0] = 0.282094791773878;
        }
        else if (L0 == 1) {
          dum = 0.48860251190292 * siQ;
          AF[1][0] = dum * coP;
          AF[1][1] = dum * siP;
          AF[1][2] = 0.48860251190292 * coQ;
        }
        else if (L0 == 2) {
          dum1 = siQ * siQ;
          dum2 = 1.09254843059208 * siQ * coQ;
          AF[2][0] = 0.94617469575756 * coQ * coQ - 0.31539156525252;
          AF[2][1] = 0.54627421529604 * dum1 * (1.0 - 2.0 * siP * siP);
          AF[2][2] = 1.09254843059208 * dum1 * siP * coP;
          AF[2][3] = dum2 * coP;
          AF[2][4] = dum2 * siP;
        }
        else if (L0 == 3) {
          AF[3][0] = 0.373176332590116 * (5.0 * coQ * coQ * coQ - 3.0 * coQ);
          AF[3][1] = 0.457045799464466 * coP * siQ * (5.0 * coQ * coQ - 1.0);
          AF[3][2] = 0.457045799464466 * siP * siQ * (5.0 * coQ * coQ - 1.0);
          AF[3][3] = 1.44530572132028 * siQ * siQ * coQ * (coP * coP - siP * siP);
          AF[3][4] = 2.89061144264055 * siQ * siQ * coQ * siP * coP;
          AF[3][5] = 0.590043589926644 * siQ * siQ * siQ * (4.0 * coP * coP * coP - 3.0 * coP);
          AF[3][6] = 0.590043589926644 * siQ * siQ * siQ * (3.0 * siP - 4.0 * siP * siP * siP);
        }
      }

      /* Chi */
      i1 = -1;
      for (L0 = 0; L0 <= maxl; L0++) {
        const int nb = sp_nb[wan * (SOG_L0MAX + 1) + L0];

        for (Mul0 = 0; Mul0 < nb; Mul0++) {
          for (M0 = 0; M0 <= 2 * L0; M0++) {
            i1++;
            out[i1] = (float)(RF[L0][Mul0] * AF[L0][M0]);
          }
        }
      }
    }
    else {
      for (i1 = 0; i1 < no; i1++) out[i1] = 0.0f;
    }
}
#pragma omp end declare target

typedef struct {int wan,no,grid,cell;double xyz[3],shift[3];} SOG_CapacityPoint;
typedef struct {int ng23,ng3,max_no;double frame[9],origin[3];} SOG_CapacityParameters;
static void SOG_capacity_kernel(size_t first,size_t n,const void *parameters,void *const v[],int gpu)
{
  (void)first;const SOG_CapacityParameters *p=parameters;
  const int ng23=p->ng23,ng3=p->ng3,max_no=p->max_no;
  const double g11=p->frame[0],g12=p->frame[1],g13=p->frame[2];
  const double g21=p->frame[3],g22=p->frame[4],g23=p->frame[5];
  const double g31=p->frame[6],g32=p->frame[7],g33=p->frame[8];
  const double org1=p->origin[0],org2=p->origin[1],org3=p->origin[2];
  const SOG_CapacityPoint *points=v[0];
  const double *rv=v[1],*rwf=v[2],*translations=v[3];
  const size_t *rvo=v[4],*rwbo=v[5];const int *mesh=v[6],*maxl=v[7],*nb=v[8];
  float *out=v[9];
#pragma omp target teams distribute parallel for thread_limit(128) is_device_ptr(points,rv,rwf,translations,rvo,rwbo,mesh,maxl,nb,out) if(gpu)
  for (size_t i=0;i<n;i++) {
    SOG_CapacityPoint q=points[i];
    for (int j=q.no;j<max_no;j++) out[i*max_no+j]=0;
    SOG_point_eval(q.wan,q.no,q.xyz[0],q.xyz[1],q.xyz[2],q.shift[0],q.shift[1],q.shift[2],q.grid,q.cell,
      translations,rv,rwf,rvo,rwbo,mesh,maxl,nb,ng23,ng3,
      g11,g12,g13,g21,g22,g23,g31,g32,g33,org1,org2,org3,out+i*max_no);
  }
}

/* Failed owner staging is retried as independent point tiles. The radial
   tables are small, immutable lookups; no complete rank orbital table is
   replicated on a helper GPU or sent to another node. */
static int SOG_capacity_recover(int done,int rank,int requested)
{
  int pending=requested && !done,any=0;MPI_Allreduce(&pending,&any,1,MPI_INT,MPI_MAX,mpi_comm_level1);
  if (!any) return done;
  SOG_CapacityParameters p={0};p.ng23=Ngrid2*Ngrid3;p.ng3=Ngrid3;
  for (int i=0;i<3;i++) {p.origin[i]=Grid_Origin[i+1];for(int j=0;j<3;j++)p.frame[3*i+j]=gtv[i+1][j+1];}
  size_t rows=0,rvn=0,rwfn=0,tile=262144;
  int ready=1;
  for (int w=0;w<SpeciesNum;w++) if (p.max_no<Spe_Total_NO[w]) p.max_no=Spe_Total_NO[w];
  size_t *rvo=NULL,*rwbo=NULL;int *mesh=NULL,*maxl=NULL,*nb=NULL;
  double *rv=NULL,*rwf=NULL,*translations=NULL;
  SOG_CapacityPoint *points=NULL;Type_Orbs_Grid **destination=NULL;float *out=NULL;
  if (pending) {
    rvo=malloc((size_t)SpeciesNum*sizeof(size_t));rwbo=malloc((size_t)SpeciesNum*sizeof(size_t));
    mesh=malloc((size_t)SpeciesNum*sizeof(int));maxl=malloc((size_t)SpeciesNum*sizeof(int));
    nb=calloc((size_t)SpeciesNum*(SOG_L0MAX+1),sizeof(int));
    ready=rvo && rwbo && mesh && maxl && nb;
    for (int w=0;ready && w<SpeciesNum;w++) {
      rvo[w]=rvn;rwbo[w]=rwfn;mesh[w]=Spe_Num_Mesh_PAO[w];maxl[w]=Spe_MaxL_Basis[w];rvn+=mesh[w];
      for(int l=0;l<=maxl[w];l++){nb[w*(SOG_L0MAX+1)+l]=Spe_Num_Basis[w][l];rwfn+=(size_t)mesh[w]*Spe_Num_Basis[w][l];}
    }
    if (ready) {
      rv=malloc(rvn*sizeof(double));rwf=malloc(rwfn*sizeof(double));
      translations=malloc(((size_t)TCpyCell+1)*3*sizeof(double));ready=rv && rwf && translations;
    }
    for(int w=0;ready && w<SpeciesNum;w++) {
      memcpy(rv+rvo[w],Spe_PAO_RV[w],(size_t)mesh[w]*sizeof(double));size_t at=rwbo[w];
      for(int l=0;l<=maxl[w];l++)for(int m=0;m<Spe_Num_Basis[w][l];m++) {
        memcpy(rwf+at,Spe_PAO_RWF[w][l][m],(size_t)mesh[w]*sizeof(double));at+=mesh[w];
      }
    }
    if(ready)for(int r=0;r<=TCpyCell;r++)for(int i=0;i<3;i++)translations[3*(size_t)r+i]=atv[r][i+1];
    for(int mc=1;ready && mc<=Matomnum;mc++) {
      int gc=M2G[mc];rows+=(size_t)GridN_Atom[gc];
      for(int h=0;h<=FNAN[gc];h++)if(G2ID[natn[gc][h]]!=rank)rows+=(size_t)NumOLG[mc][h];
    }
    if (tile>rows) tile=rows;
    size_t per=sizeof(*points)+sizeof(*destination)+(size_t)p.max_no*sizeof(float);
    unsigned long long room=openmx_host_mem_available();int sharing=openmx_gpu_local_size_noncollective();if(sharing<1)sharing=1;
    if(room && tile>room/(4ULL*sharing)/per)tile=(size_t)(room/(4ULL*sharing)/per);
    if(rows && !tile)ready=0;
    if(ready && tile){points=malloc(tile*sizeof(*points));destination=malloc(tile*sizeof(*destination));out=malloc(tile*(size_t)p.max_no*sizeof(float));ready=points && destination && out;}
  }
  if(!ready)rows=0;
  uint64_t steps=tile?(rows/tile+(rows%tile!=0)):0,max_steps=0;
  MPI_Allreduce(&steps,&max_steps,1,MPI_UINT64_T,MPI_MAX,mpi_comm_level1);
  for(uint64_t step=0;step<max_steps;step++) {
    size_t begin=step<steps?(size_t)step*tile:rows,end=begin;
    if(step<steps){end=rows-begin<tile?rows:begin+tile;}
    size_t cursor=0;
    for(int mc=1;begin<end && mc<=Matomnum;mc++) {
      int gc=M2G[mc];
      for(int h=-1;h<=FNAN[gc];h++) {
        int gh=h<0?gc:natn[gc][h];if(h>=0 && G2ID[gh]==rank)continue;
        size_t length=h<0?(size_t)GridN_Atom[gc]:(size_t)NumOLG[mc][h],lo=cursor,hi=cursor+length;
        cursor=hi;if(lo<begin)lo=begin;if(hi>end)hi=end;
        for(size_t row=lo;row<hi;row++) {
          size_t k=row-(cursor-length),i=row-begin;int nc=h<0?(int)k:GListTAtoms1[mc][h][k];
          SOG_CapacityPoint *q=points+i;q->wan=WhatSpecies[gh];q->no=Spe_Total_NO[q->wan];
          q->grid=GridListAtom[mc][nc];q->cell=CellListAtom[mc][nc];
          for(int d=0;d<3;d++){q->xyz[d]=Gxyz[gh][d+1];q->shift[d]=h<0?0:atv[ncn[gc][h]][d+1];}
          destination[i]=h<0?Orbs_Grid[mc][k]:Orbs_Grid_FNAN[mc][h][k];
        }
      }
    }
    OpenMXGpuTask t={0};t.count=end-begin;t.inputs=9;t.outputs=1;t.parameters=&p;t.parameter_bytes=sizeof(p);t.kernel=SOG_capacity_kernel;
    t.view[0]=(OpenMXGpuTaskView){points,0,sizeof(*points)};
    t.view[1]=(OpenMXGpuTaskView){rv,rvn*sizeof(double),0};t.view[2]=(OpenMXGpuTaskView){rwf,rwfn*sizeof(double),0};
    t.view[3]=(OpenMXGpuTaskView){translations,translations?((size_t)TCpyCell+1)*3*sizeof(double):0,0};
    t.view[4]=(OpenMXGpuTaskView){rvo,rvo?(size_t)SpeciesNum*sizeof(size_t):0,0};
    t.view[5]=(OpenMXGpuTaskView){rwbo,rwbo?(size_t)SpeciesNum*sizeof(size_t):0,0};
    t.view[6]=(OpenMXGpuTaskView){mesh,mesh?(size_t)SpeciesNum*sizeof(int):0,0};
    t.view[7]=(OpenMXGpuTaskView){maxl,maxl?(size_t)SpeciesNum*sizeof(int):0,0};
    t.view[8]=(OpenMXGpuTaskView){nb,nb?(size_t)SpeciesNum*(SOG_L0MAX+1)*sizeof(int):0,0};
    t.view[9]=(OpenMXGpuTaskView){out,0,(size_t)p.max_no*sizeof(float)};
    openmx_gpu_task_execute(mpi_comm_level1,&t,1,"orbital-capacity");
    for(size_t i=0;i<t.count;i++)for(int j=0;j<points[i].no;j++)destination[i][j]=out[i*(size_t)p.max_no+j];
  }
  free(rvo);free(rwbo);free(mesh);free(maxl);free(nb);free(rv);free(rwf);free(translations);free(points);free(destination);free(out);
  return done || (pending && ready);
}

double Set_Orbitals_Grid(int Cnt_kind)
{
  int i,j,n,Mc_AN,Gc_AN,Cwan,NO0,GNc,GRc;
  int Gh_AN,Mh_AN,Rnh,Hwan,NO1,Nog,h_AN;
  long int k,Nc;
  double time0;
  double x,y,z,dx,dy,dz;
  double TStime,TEtime;
  double Cxyz[4];
  int numprocs,myid,tag=999,ID,IDS,IDR;
  double Stime_atom,Etime_atom;

  MPI_Status stat;
  MPI_Request request;

  /* for OpenMP */
  int OMPID,Nthrds,Nprocs;

  /* Orbital values and their grid topology back the density GPU service
     cache.  Drop the previous epoch before any values are regenerated. */
  Set_Density_Grid_GPU_Invalidate();
  Set_Hamiltonian_Invalidate_OpenMP_MatrixElements_Cache();
  Set_Orbitals_Grid_OTF_Release();
  {
    extern void Set_Hamiltonian_OTF_Release(void);
    Set_Hamiltonian_OTF_Release();
  }

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  dtime(&TStime);

  /* device fast path; on any ineligibility or allocation failure the
     untouched host loops below produce the same values */
  int orbital_done=!(openmx_gpu_capacity_test_recovery() || openmx_gpu_workers_requested()) && SOG_gpu_eligible(Cnt_kind) && Set_Orbitals_Grid_GPU();
  if (SOG_capacity_recover(orbital_done,myid,SOG_gpu_requested(Cnt_kind))) {
    dtime(&TEtime);
    return TEtime - TStime;
  }

  /*****************************************************
                Calculate orbitals on grids
  *****************************************************/

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

    dtime(&Stime_atom);

    Gc_AN = M2G[Mc_AN];
    Cwan = WhatSpecies[Gc_AN];

    if (Cnt_kind==0)  NO0 = Spe_Total_NO[Cwan];
    else              NO0 = Spe_Total_CNO[Cwan];

#pragma omp parallel shared(Comp2Real,Spe_PAO_RWF,Spe_Num_Basis,Spe_MaxL_Basis,Spe_PAO_RV,Spe_Num_Mesh_PAO,List_YOUSO,Orbs_Grid,Cnt_kind,Gxyz,atv,CellListAtom,GridListAtom,GridN_Atom,Gc_AN,Cwan,Mc_AN,NO0) private(OMPID,Nthrds,Nprocs,Nc,GNc,GRc,Cxyz,x,y,z,dx,dy,dz,i,j)
    {
      double *Chi0;
      double Cxyz0[4];
      double **RF;
      double **AF;
      int i,L0,Mul0,M0,i1;
      double S_coordinate[3];
      double dum,dum1,dum2,dum3,dum4,a,b,c,d;
      double siQ,coQ,siP,coP,Q,P,R;
      double rm,df,sum0,sum1;
      double SH[Supported_MaxL*2+1][2];
      double dSHt[Supported_MaxL*2+1][2];
      double dSHp[Supported_MaxL*2+1][2];

      /* Radial */
      int mp_min,mp_max,m,po,wan;
      double h1,h2,h3,f1,f2,f3,f4;
      double g1,g2,x1,x2,y1,y2,y12,y22,f;
      double r,r1,theta,phi,Min_r;

      /* allocation of array */

      Chi0 = (double*)malloc(sizeof(double)*List_YOUSO[7]);

      RF = (double**)malloc(sizeof(double*)*(List_YOUSO[25]+1));
      for (i=0; i<(List_YOUSO[25]+1); i++){
	RF[i] = (double*)malloc(sizeof(double)*List_YOUSO[24]);
      }

      AF = (double**)malloc(sizeof(double*)*(List_YOUSO[25]+1));
      for (i=0; i<(List_YOUSO[25]+1); i++){
	AF[i] = (double*)malloc(sizeof(double)*(2*(List_YOUSO[25]+1)+1));
      }

      /* get info. on OpenMP */

      OMPID = omp_get_thread_num();
      Nthrds = omp_get_num_threads();
      Nprocs = omp_get_num_procs();

      for (Nc=OMPID*GridN_Atom[Gc_AN]/Nthrds; Nc<(OMPID+1)*GridN_Atom[Gc_AN]/Nthrds; Nc++){

	GNc = GridListAtom[Mc_AN][Nc];
	GRc = CellListAtom[Mc_AN][Nc];

	Get_Grid_XYZ(GNc,Cxyz);
	x = Cxyz[1] + atv[GRc][1] - Gxyz[Gc_AN][1];
	y = Cxyz[2] + atv[GRc][2] - Gxyz[Gc_AN][2];
	z = Cxyz[3] + atv[GRc][3] - Gxyz[Gc_AN][3];

	if (Cnt_kind==0){

          /* Get_Orbitals(Cwan,x,y,z,Chi0); */
          /* start of inlining of Get_Orbitals */

          wan = Cwan;

          /* xyz2spherical(x,y,z,0.0,0.0,0.0,S_coordinate); */
          /* start of inlining of xyz2spherical */

	  Min_r = 10e-15;
	  dum = x*x + y*y;
	  r = sqrt(dum + z*z);
	  r1 = sqrt(dum);

	  if (Min_r<=r){

	    if (r<fabs(z))
	      dum1 = sgn(z)*1.0;
	    else
	      dum1 = z/r;

	    theta = acos(dum1);

	    if (Min_r<=r1){
	      if (0.0<=x){

		if (r1<fabs(y))
		  dum1 = sgn(y)*1.0;
		else
		  dum1 = y/r1;

		phi = asin(dum1);
	      }
	      else{

		if (r1<fabs(y))
		  dum1 = sgn(y)*1.0;
		else
		  dum1 = y/r1;

		phi = PI - asin(dum1);
	      }
	    }
	    else{
	      phi = 0.0;
	    }
	  }
	  else{
	    theta = 0.5*PI;
	    phi = 0.0;
	  }

	  R = r;
	  Q = theta;
	  P = phi;

	  /* end of inlining of xyz2spherical */

	  po = 0;
	  mp_min = 0;
	  mp_max = Spe_Num_Mesh_PAO[wan] - 1;

	  if (Spe_PAO_RV[wan][Spe_Num_Mesh_PAO[wan]-1]<R){

	    for (L0=0; L0<=Spe_MaxL_Basis[wan]; L0++){
	      for (Mul0=0; Mul0<Spe_Num_Basis[wan][L0]; Mul0++){
		RF[L0][Mul0] = 0.0;
	      }
	    }

	    po = 1;
	  }

	  else if (R<Spe_PAO_RV[wan][0]){

	    m = 4;
	    rm = Spe_PAO_RV[wan][m];

	    /* Use reflected endpoint samples before forming the spline weights. */
	    h1 = (m==1) ? Spe_PAO_RV[wan][m-1] - Spe_PAO_RV[wan][m+1]
	      : Spe_PAO_RV[wan][m-1] - Spe_PAO_RV[wan][m-2];
	    h2 = Spe_PAO_RV[wan][m]   - Spe_PAO_RV[wan][m-1];
	    h3 = (m==Spe_Num_Mesh_PAO[wan]-1) ? Spe_PAO_RV[wan][m-2] - Spe_PAO_RV[wan][m]
	      : Spe_PAO_RV[wan][m+1] - Spe_PAO_RV[wan][m];

	    x1 = rm - Spe_PAO_RV[wan][m-1];
	    x2 = rm - Spe_PAO_RV[wan][m];
	    y1 = x1/h2;
	    y2 = x2/h2;
	    y12 = y1*y1;
	    y22 = y2*y2;

	    dum = h1 + h2;
	    dum1 = h1/h2/dum;
	    dum2 = h2/h1/dum;
	    dum = h2 + h3;
	    dum3 = h2/h3/dum;
	    dum4 = h3/h2/dum;

	    for (L0=0; L0<=Spe_MaxL_Basis[wan]; L0++){
	      for (Mul0=0; Mul0<Spe_Num_Basis[wan][L0]; Mul0++){

		f1 = Spe_PAO_RWF[wan][L0][Mul0][(m==1) ? m+1 : m-2];
		f2 = Spe_PAO_RWF[wan][L0][Mul0][m-1];
		f3 = Spe_PAO_RWF[wan][L0][Mul0][m];
		f4 = Spe_PAO_RWF[wan][L0][Mul0][(m==Spe_Num_Mesh_PAO[wan]-1) ? m-2 : m+1];

		dum = f3 - f2;
		g1 = dum*dum1 + (f2-f1)*dum2;
		g2 = (f4-f3)*dum3 + dum*dum4;

		f =  y22*(3.0*f2 + h2*g1 + (2.0*f2 + h2*g1)*y2)
		  + y12*(3.0*f3 - h2*g2 - (2.0*f3 - h2*g2)*y1);

		df = 2.0*y2/h2*(3.0*f2 + h2*g1 + (2.0*f2 + h2*g1)*y2)
		  + y22*(2.0*f2 + h2*g1)/h2
		  + 2.0*y1/h2*(3.0*f3 - h2*g2 - (2.0*f3 - h2*g2)*y1)
		  - y12*(2.0*f3 - h2*g2)/h2;

		if (L0==0){
		  a = 0.0;
		  b = 0.5*df/rm;
		  c = 0.0;
		  d = f - b*rm*rm;
		}

		else if (L0==1){
		  a = (rm*df - f)/(2.0*rm*rm*rm);
		  b = 0.0;
		  c = df - 3.0*a*rm*rm;
		  d = 0.0;
		}

		else{
		  b = (3.0*f - rm*df)/(rm*rm);
		  a = (f - b*rm*rm)/(rm*rm*rm);
		  c = 0.0;
		  d = 0.0;
		}

		RF[L0][Mul0] = a*R*R*R + b*R*R + c*R + d;

	      }
	    }

	  }

	  else{

	    do{
	      m = (mp_min + mp_max)/2;
	      if (Spe_PAO_RV[wan][m]<R)
		mp_min = m;
	      else
		mp_max = m;
	    }
	    while((mp_max-mp_min)!=1);
	    m = mp_max;

	    h1 = (m==1) ? Spe_PAO_RV[wan][m-1] - Spe_PAO_RV[wan][m+1]
	      : Spe_PAO_RV[wan][m-1] - Spe_PAO_RV[wan][m-2];
	    h2 = Spe_PAO_RV[wan][m]   - Spe_PAO_RV[wan][m-1];
	    h3 = (m==Spe_Num_Mesh_PAO[wan]-1) ? Spe_PAO_RV[wan][m-2] - Spe_PAO_RV[wan][m]
	      : Spe_PAO_RV[wan][m+1] - Spe_PAO_RV[wan][m];

	    x1 = R - Spe_PAO_RV[wan][m-1];
	    x2 = R - Spe_PAO_RV[wan][m];
	    y1 = x1/h2;
	    y2 = x2/h2;
	    y12 = y1*y1;
	    y22 = y2*y2;

	    dum = h1 + h2;
	    dum1 = h1/h2/dum;
	    dum2 = h2/h1/dum;
	    dum = h2 + h3;
	    dum3 = h2/h3/dum;
	    dum4 = h3/h2/dum;

	    for (L0=0; L0<=Spe_MaxL_Basis[wan]; L0++){
	      for (Mul0=0; Mul0<Spe_Num_Basis[wan][L0]; Mul0++){

		f1 = Spe_PAO_RWF[wan][L0][Mul0][(m==1) ? m+1 : m-2];
		f2 = Spe_PAO_RWF[wan][L0][Mul0][m-1];
		f3 = Spe_PAO_RWF[wan][L0][Mul0][m];
		f4 = Spe_PAO_RWF[wan][L0][Mul0][(m==Spe_Num_Mesh_PAO[wan]-1) ? m-2 : m+1];

		dum = f3 - f2;
		g1 = dum*dum1 + (f2-f1)*dum2;
		g2 = (f4-f3)*dum3 + dum*dum4;

		f =  y22*(3.0*f2 + h2*g1 + (2.0*f2 + h2*g1)*y2)
		  + y12*(3.0*f3 - h2*g2 - (2.0*f3 - h2*g2)*y1);

		RF[L0][Mul0] = f;

	      }
	    }

	  }

	  if (po==0){

	    /* Angular */
	    siQ = sin(Q);
	    coQ = cos(Q);
	    siP = sin(P);
	    coP = cos(P);

	    for (L0=0; L0<=Spe_MaxL_Basis[wan]; L0++){

	      if (L0==0){
		AF[0][0] = 0.282094791773878;
	      }
	      else if (L0==1){
		dum = 0.48860251190292*siQ;
		AF[1][0] = dum*coP;
		AF[1][1] = dum*siP;
		AF[1][2] = 0.48860251190292*coQ;
	      }
	      else if (L0==2){
		dum1 = siQ*siQ;
		dum2 = 1.09254843059208*siQ*coQ;
		AF[2][0] = 0.94617469575756*coQ*coQ - 0.31539156525252;
		AF[2][1] = 0.54627421529604*dum1*(1.0 - 2.0*siP*siP);
		AF[2][2] = 1.09254843059208*dum1*siP*coP;
		AF[2][3] = dum2*coP;
		AF[2][4] = dum2*siP;
	      }

	      else if (L0==3){
		AF[3][0] = 0.373176332590116*(5.0*coQ*coQ*coQ - 3.0*coQ);
		AF[3][1] = 0.457045799464466*coP*siQ*(5.0*coQ*coQ - 1.0);
		AF[3][2] = 0.457045799464466*siP*siQ*(5.0*coQ*coQ - 1.0);
		AF[3][3] = 1.44530572132028*siQ*siQ*coQ*(coP*coP-siP*siP);
		AF[3][4] = 2.89061144264055*siQ*siQ*coQ*siP*coP;
		AF[3][5] = 0.590043589926644*siQ*siQ*siQ*(4.0*coP*coP*coP - 3.0*coP);
		AF[3][6] = 0.590043589926644*siQ*siQ*siQ*(3.0*siP - 4.0*siP*siP*siP);
	      }

	      else if (4<=L0){

		/* calculation of complex spherical harmonics functions */
		for(m=-L0; m<=L0; m++){
		  ComplexSH(L0,m,Q,P,SH[L0+m],dSHt[L0+m],dSHp[L0+m]);
		}

		/* transformation of complex to real */
		for (i=0; i<(L0*2+1); i++){

		  sum0 = 0.0;
		  sum1 = 0.0;
		  for (j=0; j<(L0*2+1); j++){
		    sum0 += Comp2Real[L0][i][j].r*SH[j][0] - Comp2Real[L0][i][j].i*SH[j][1];
		    sum1 += Comp2Real[L0][i][j].r*SH[j][1] + Comp2Real[L0][i][j].i*SH[j][0];
		  }
		  AF[L0][i] = sum0 + sum1;
		}

	      }
	    }
	  }

	  /* Chi0 */
	  i1 = -1;
	  for (L0=0; L0<=Spe_MaxL_Basis[wan]; L0++){
	    for (Mul0=0; Mul0<Spe_Num_Basis[wan][L0]; Mul0++){
	      for (M0=0; M0<=2*L0; M0++){
		i1++;
		Chi0[i1] = (po==0) ? RF[L0][Mul0]*AF[L0][M0] : 0.0;
	      }
	    }
	  }
	  /* end of inlining of Get_Orbitals */

	}
	else{
          Get_Cnt_Orbitals(Mc_AN,x,y,z,Chi0);
	}

	for (i=0; i<NO0; i++){
	  Orbs_Grid[Mc_AN][Nc][i] = (Type_Orbs_Grid)Chi0[i];/* AITUNE */
	}

      } /* Nc */

      /* freeing of array */

      free(Chi0);

      for (i=0; i<(List_YOUSO[25]+1); i++){
	free(RF[i]);
      }
      free(RF);

      for (i=0; i<(List_YOUSO[25]+1); i++){
	free(AF[i]);
      }
      free(AF);

    } /* #pragma omp parallel */

    dtime(&Etime_atom);
    time_per_atom[Gc_AN] += Etime_atom - Stime_atom;
  }

  /****************************************************
     Calculate Orbs_Grid_FNAN
  ****************************************************/

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

    Gc_AN = M2G[Mc_AN];

    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

      Gh_AN = natn[Gc_AN][h_AN];

      if (G2ID[Gh_AN]!=myid){

        Mh_AN = F_G2M[Gh_AN];
        Rnh = ncn[Gc_AN][h_AN];
        Hwan = WhatSpecies[Gh_AN];

        if (Cnt_kind==0)  NO1 = Spe_Total_NO[Hwan];
        else              NO1 = Spe_Total_CNO[Hwan];

#pragma omp parallel shared(List_YOUSO,Orbs_Grid_FNAN,NO1,Mh_AN,Hwan,Cnt_kind,Rnh,Gh_AN,Gxyz,atv,NumOLG,Mc_AN,h_AN,GListTAtoms1,GridListAtom,CellListAtom) private(OMPID,Nthrds,Nprocs,Nog,Nc,GNc,GRc,x,y,z,j)
        {

          double *Chi0;
	  double Cxyz0[4];
          double **RF;
          double **AF;
	  int i,L0,Mul0,M0,i1;
	  double S_coordinate[3];
	  double dum,dum1,dum2,dum3,dum4,a,b,c,d;
	  double siQ,coQ,siP,coP,Q,P,R;
	  double rm,df,sum0,sum1;
	  double SH[Supported_MaxL*2+1][2];
	  double dSHt[Supported_MaxL*2+1][2];
	  double dSHp[Supported_MaxL*2+1][2];

	  /* Radial */
	  int mp_min,mp_max,m,po,wan;
	  double h1,h2,h3,f1,f2,f3,f4;
	  double g1,g2,x1,x2,y1,y2,y12,y22,f;
          double r,r1,theta,phi,Min_r;

          /* allocation of arrays */

	  Chi0 = (double*)malloc(sizeof(double)*List_YOUSO[7]);

	  RF = (double**)malloc(sizeof(double*)*(List_YOUSO[25]+1));
	  for (i=0; i<(List_YOUSO[25]+1); i++){
	    RF[i] = (double*)malloc(sizeof(double)*List_YOUSO[24]);
	  }

	  AF = (double**)malloc(sizeof(double*)*(List_YOUSO[25]+1));
	  for (i=0; i<(List_YOUSO[25]+1); i++){
	    AF[i] = (double*)malloc(sizeof(double)*(2*(List_YOUSO[25]+1)+1));
	  }

	  /* get info. on OpenMP */

	  OMPID = omp_get_thread_num();
	  Nthrds = omp_get_num_threads();
	  Nprocs = omp_get_num_procs();

	  for (Nog=OMPID*NumOLG[Mc_AN][h_AN]/Nthrds; Nog<(OMPID+1)*NumOLG[Mc_AN][h_AN]/Nthrds; Nog++){

	    Nc = GListTAtoms1[Mc_AN][h_AN][Nog];
	    GNc = GridListAtom[Mc_AN][Nc];
	    GRc = CellListAtom[Mc_AN][Nc];

	    Get_Grid_XYZ(GNc,Cxyz0);

	    x = Cxyz0[1] + atv[GRc][1] - Gxyz[Gh_AN][1] - atv[Rnh][1];
	    y = Cxyz0[2] + atv[GRc][2] - Gxyz[Gh_AN][2] - atv[Rnh][2];
	    z = Cxyz0[3] + atv[GRc][3] - Gxyz[Gh_AN][3] - atv[Rnh][3];

	    if (Cnt_kind==0){

              /* Get_Orbitals(Hwan,x,y,z,Chi0); */
              /* start of inlining of Get_Orbitals */

              wan = Hwan;

	      /* xyz2spherical(x,y,z,0.0,0.0,0.0,S_coordinate); */
              /* start of inlining of xyz2spherical */

	      Min_r = 10e-15;
	      dum = x*x + y*y;
	      r = sqrt(dum + z*z);
	      r1 = sqrt(dum);

	      if (Min_r<=r){

		if (r<fabs(z))
		  dum1 = sgn(z)*1.0;
		else
		  dum1 = z/r;

		theta = acos(dum1);

		if (Min_r<=r1){
		  if (0.0<=x){

		    if (r1<fabs(y))
		      dum1 = sgn(y)*1.0;
		    else
		      dum1 = y/r1;

		    phi = asin(dum1);
		  }
		  else{

		    if (r1<fabs(y))
		      dum1 = sgn(y)*1.0;
		    else
		      dum1 = y/r1;

		    phi = PI - asin(dum1);
		  }
		}
		else{
		  phi = 0.0;
		}
	      }
	      else{
		theta = 0.5*PI;
		phi = 0.0;
	      }

	      R = r;
	      Q = theta;
	      P = phi;

              /* end of inlining of xyz2spherical */

	      po = 0;
	      mp_min = 0;
	      mp_max = Spe_Num_Mesh_PAO[wan] - 1;

	      if (Spe_PAO_RV[wan][Spe_Num_Mesh_PAO[wan]-1]<R){

		for (L0=0; L0<=Spe_MaxL_Basis[wan]; L0++){
		  for (Mul0=0; Mul0<Spe_Num_Basis[wan][L0]; Mul0++){
		    RF[L0][Mul0] = 0.0;
		  }
		}

		po = 1;
	      }

	      else if (R<Spe_PAO_RV[wan][0]){

		m = 4;
		rm = Spe_PAO_RV[wan][m];

		h1 = (m==1) ? Spe_PAO_RV[wan][m-1] - Spe_PAO_RV[wan][m+1]
		  : Spe_PAO_RV[wan][m-1] - Spe_PAO_RV[wan][m-2];
		h2 = Spe_PAO_RV[wan][m]   - Spe_PAO_RV[wan][m-1];
		h3 = (m==Spe_Num_Mesh_PAO[wan]-1) ? Spe_PAO_RV[wan][m-2] - Spe_PAO_RV[wan][m]
		  : Spe_PAO_RV[wan][m+1] - Spe_PAO_RV[wan][m];

		x1 = rm - Spe_PAO_RV[wan][m-1];
		x2 = rm - Spe_PAO_RV[wan][m];
		y1 = x1/h2;
		y2 = x2/h2;
		y12 = y1*y1;
		y22 = y2*y2;

		dum = h1 + h2;
		dum1 = h1/h2/dum;
		dum2 = h2/h1/dum;
		dum = h2 + h3;
		dum3 = h2/h3/dum;
		dum4 = h3/h2/dum;

		for (L0=0; L0<=Spe_MaxL_Basis[wan]; L0++){
		  for (Mul0=0; Mul0<Spe_Num_Basis[wan][L0]; Mul0++){

		    f1 = Spe_PAO_RWF[wan][L0][Mul0][(m==1) ? m+1 : m-2];
		    f2 = Spe_PAO_RWF[wan][L0][Mul0][m-1];
		    f3 = Spe_PAO_RWF[wan][L0][Mul0][m];
		    f4 = Spe_PAO_RWF[wan][L0][Mul0][(m==Spe_Num_Mesh_PAO[wan]-1) ? m-2 : m+1];

		    dum = f3 - f2;
		    g1 = dum*dum1 + (f2-f1)*dum2;
		    g2 = (f4-f3)*dum3 + dum*dum4;

		    f =  y22*(3.0*f2 + h2*g1 + (2.0*f2 + h2*g1)*y2)
		      + y12*(3.0*f3 - h2*g2 - (2.0*f3 - h2*g2)*y1);

		    df = 2.0*y2/h2*(3.0*f2 + h2*g1 + (2.0*f2 + h2*g1)*y2)
		      + y22*(2.0*f2 + h2*g1)/h2
		      + 2.0*y1/h2*(3.0*f3 - h2*g2 - (2.0*f3 - h2*g2)*y1)
		      - y12*(2.0*f3 - h2*g2)/h2;

		    if (L0==0){
		      a = 0.0;
		      b = 0.5*df/rm;
		      c = 0.0;
		      d = f - b*rm*rm;
		    }

		    else if (L0==1){
		      a = (rm*df - f)/(2.0*rm*rm*rm);
		      b = 0.0;
		      c = df - 3.0*a*rm*rm;
		      d = 0.0;
		    }

		    else{
		      b = (3.0*f - rm*df)/(rm*rm);
		      a = (f - b*rm*rm)/(rm*rm*rm);
		      c = 0.0;
		      d = 0.0;
		    }

		    RF[L0][Mul0] = a*R*R*R + b*R*R + c*R + d;

		  }
		}

	      }

	      else{

		do{
		  m = (mp_min + mp_max)/2;
		  if (Spe_PAO_RV[wan][m]<R)
		    mp_min = m;
		  else
		    mp_max = m;
		}
		while((mp_max-mp_min)!=1);
		m = mp_max;

		h1 = (m==1) ? Spe_PAO_RV[wan][m-1] - Spe_PAO_RV[wan][m+1]
		  : Spe_PAO_RV[wan][m-1] - Spe_PAO_RV[wan][m-2];
		h2 = Spe_PAO_RV[wan][m]   - Spe_PAO_RV[wan][m-1];
		h3 = (m==Spe_Num_Mesh_PAO[wan]-1) ? Spe_PAO_RV[wan][m-2] - Spe_PAO_RV[wan][m]
		  : Spe_PAO_RV[wan][m+1] - Spe_PAO_RV[wan][m];

		x1 = R - Spe_PAO_RV[wan][m-1];
		x2 = R - Spe_PAO_RV[wan][m];
		y1 = x1/h2;
		y2 = x2/h2;
		y12 = y1*y1;
		y22 = y2*y2;

		dum = h1 + h2;
		dum1 = h1/h2/dum;
		dum2 = h2/h1/dum;
		dum = h2 + h3;
		dum3 = h2/h3/dum;
		dum4 = h3/h2/dum;

		for (L0=0; L0<=Spe_MaxL_Basis[wan]; L0++){
		  for (Mul0=0; Mul0<Spe_Num_Basis[wan][L0]; Mul0++){

		    f1 = Spe_PAO_RWF[wan][L0][Mul0][(m==1) ? m+1 : m-2];
		    f2 = Spe_PAO_RWF[wan][L0][Mul0][m-1];
		    f3 = Spe_PAO_RWF[wan][L0][Mul0][m];
		    f4 = Spe_PAO_RWF[wan][L0][Mul0][(m==Spe_Num_Mesh_PAO[wan]-1) ? m-2 : m+1];

		    dum = f3 - f2;
		    g1 = dum*dum1 + (f2-f1)*dum2;
		    g2 = (f4-f3)*dum3 + dum*dum4;

		    f =  y22*(3.0*f2 + h2*g1 + (2.0*f2 + h2*g1)*y2)
		      + y12*(3.0*f3 - h2*g2 - (2.0*f3 - h2*g2)*y1);

		    RF[L0][Mul0] = f;

		  }
		}

	      }

	      if (po==0){

		/* Angular */
		siQ = sin(Q);
		coQ = cos(Q);
		siP = sin(P);
		coP = cos(P);

		for (L0=0; L0<=Spe_MaxL_Basis[wan]; L0++){

		  if (L0==0){
		    AF[0][0] = 0.282094791773878;
		  }
		  else if (L0==1){
		    dum = 0.48860251190292*siQ;
		    AF[1][0] = dum*coP;
		    AF[1][1] = dum*siP;
		    AF[1][2] = 0.48860251190292*coQ;
		  }
		  else if (L0==2){
		    dum1 = siQ*siQ;
		    dum2 = 1.09254843059208*siQ*coQ;
		    AF[2][0] = 0.94617469575756*coQ*coQ - 0.31539156525252;
		    AF[2][1] = 0.54627421529604*dum1*(1.0 - 2.0*siP*siP);
		    AF[2][2] = 1.09254843059208*dum1*siP*coP;
		    AF[2][3] = dum2*coP;
		    AF[2][4] = dum2*siP;
		  }

		  else if (L0==3){
		    AF[3][0] = 0.373176332590116*(5.0*coQ*coQ*coQ - 3.0*coQ);
		    AF[3][1] = 0.457045799464466*coP*siQ*(5.0*coQ*coQ - 1.0);
		    AF[3][2] = 0.457045799464466*siP*siQ*(5.0*coQ*coQ - 1.0);
		    AF[3][3] = 1.44530572132028*siQ*siQ*coQ*(coP*coP-siP*siP);
		    AF[3][4] = 2.89061144264055*siQ*siQ*coQ*siP*coP;
		    AF[3][5] = 0.590043589926644*siQ*siQ*siQ*(4.0*coP*coP*coP - 3.0*coP);
		    AF[3][6] = 0.590043589926644*siQ*siQ*siQ*(3.0*siP - 4.0*siP*siP*siP);
		  }

		  else if (4<=L0){

		    /* calculation of complex spherical harmonics functions */
		    for(m=-L0; m<=L0; m++){
		      ComplexSH(L0,m,Q,P,SH[L0+m],dSHt[L0+m],dSHp[L0+m]);
		    }

		    /* transformation of complex to real */
		    for (i=0; i<(L0*2+1); i++){

		      sum0 = 0.0;
		      sum1 = 0.0;
		      for (j=0; j<(L0*2+1); j++){
			sum0 += Comp2Real[L0][i][j].r*SH[j][0] - Comp2Real[L0][i][j].i*SH[j][1];
			sum1 += Comp2Real[L0][i][j].r*SH[j][1] + Comp2Real[L0][i][j].i*SH[j][0];
		      }
		      AF[L0][i] = sum0 + sum1;
		    }

		  }
		}
	      }

	      /* Chi0 */
	      i1 = -1;
	      for (L0=0; L0<=Spe_MaxL_Basis[wan]; L0++){
		for (Mul0=0; Mul0<Spe_Num_Basis[wan][L0]; Mul0++){
		  for (M0=0; M0<=2*L0; M0++){
		    i1++;
		    Chi0[i1] = (po==0) ? RF[L0][Mul0]*AF[L0][M0] : 0.0;
		  }
		}
	      }

              /* end of inlining of Get_Orbitals */

	    } /* if (Cnt_kind==0) */

	    else{
              Get_Cnt_Orbitals(Mh_AN,x,y,z,Chi0);
	    }

	    for (j=0; j<NO1; j++){
	      Orbs_Grid_FNAN[Mc_AN][h_AN][Nog][j] = (Type_Orbs_Grid)Chi0[j];/* AITUNE */
	    }

	  } /* Nog */

          /* freeing of arrays */

	  free(Chi0);

	  for (i=0; i<(List_YOUSO[25]+1); i++){
	    free(RF[i]);
	  }
	  free(RF);

	  for (i=0; i<(List_YOUSO[25]+1); i++){
	    free(AF[i]);
	  }
	  free(AF);

        }
      }
    }
  }

  /* time */
  dtime(&TEtime);
  time0 = TEtime - TStime;

  return time0;
}

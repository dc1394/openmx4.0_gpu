#ifndef OPENMX_NC_DM_CPU_H
#define OPENMX_NC_DM_CPU_H

/* The non-collinear density matrix on the host in one pass over the bands
   (Core kernel openmx_core_energy_nc_block, OPENMX_HOST_SIMD policy),
   shared by the cluster and band solvers of both GPU backends.

   The legacy cluster routine Calc_DM_Cluster_non_collinear_ScaLAPACK runs
   once per component (six, ten with the energy density matrix) and reads
   the eigenvectors again each time; the band routine copies every
   neighbour's rows again for each of its ~FNAN partners.  Here the
   neighbour side of one rank's bands is transposed once into four planes
   (real/imaginary, spinor up/down) of a global column width, every (atom,
   neighbour) block reads its columns from them, and the kernel accumulates
   every component of a block together, each element in the legacy band
   order (bitwise the legacy sums).  All of it is rank-local; the cluster
   routine ends with one Allreduce of every component. */

#include "openmx_common.h"
#include "openmx_env.h"
#include "openmx_host_simd.h"
#include "openmx_core_energy_simd.h"
#include <stdlib.h>
#include <string.h>

/* OPENMX_NC_DM_HOST_ONEPASS=0 keeps the legacy per-component loops with the
   other host SIMD paths unchanged (an A/B of this pass alone). */
static int NcDM_OnePassEnabled(int simd_mode)
{
  return simd_mode != OPENMX_HOST_SIMD_LEGACY && openmx_env_flag_at(OPENMX_ENV_NC_DM_HOST_ONEPASS, 1);
}

/* Transposed eigenvector planes [band][column]: re1/im1 are the spinor-up
   components of columns 0..n-1, re2/im2 the spinor-down ones, with a row
   stride that is a multiple of 8 and 8 columns of zero padding beyond n,
   so a block may read whole vectors from any column offset. */
typedef struct {
  double *re1, *im1, *re2, *im2;
  const double **rows;      /* 4*max_rows row pointers of the current atom */
  size_t stride;
  int bands, max_rows;
} OpenMXNcDMPanels;

static void NcDM_PanelsFree(OpenMXNcDMPanels *P)
{
  free(P->re1); free(P->im1); free(P->re2); free(P->im2); free(P->rows);
  memset(P, 0, sizeof(*P));
}

/* 0 (nothing kept) without memory or under OPENMX_HOST_SIMD=legacy. */
static int NcDM_PanelsAllocate(OpenMXNcDMPanels *P, int simd_mode, int bands, int n, int max_rows)
{
  memset(P, 0, sizeof(*P));
  if (simd_mode == OPENMX_HOST_SIMD_LEGACY || bands < 0 || n < 0 || max_rows < 0) return 0;
  P->stride = ((size_t)n + 7) / 8 * 8 + 8;
  P->bands = bands;
  P->max_rows = max_rows;
  {
    const size_t plane = (size_t)(bands > 0 ? bands : 1) * P->stride;
    P->re1 = (double *)calloc(plane, sizeof(double));
    P->im1 = (double *)calloc(plane, sizeof(double));
    P->re2 = (double *)calloc(plane, sizeof(double));
    P->im2 = (double *)calloc(plane, sizeof(double));
    P->rows = (const double **)malloc(sizeof(double *) * 4 * (size_t)(max_rows > 0 ? max_rows : 1));
  }
  if (P->re1 == NULL || P->im1 == NULL || P->re2 == NULL || P->im2 == NULL || P->rows == NULL) {
    NcDM_PanelsFree(P);
    return 0;
  }
  return 1;
}

/* The cluster density matrix of this SCF step: the Fermi-weighted sums of
   Calc_DM_Cluster_non_collinear_ScaLAPACK for the components 1..6 (and
   7..10 with energy), stored to CDM/iDM0/EDM like the legacy routine.
   EVec1 holds this rank's bands is2[myid]..ie2[myid] as consecutive
   2n-vectors.  Returns 0, with nothing changed, under the legacy policy,
   without memory, or when the kernel declines: the caller then runs the
   legacy per-component routine.  *seconds receives the elapsed time. */
static int NcDM_Cluster_OnePass(const OpenMXChemicalPotentials *chemical, int myid, int size_H1,
                                const int *is2, const int *ie2, const int *MP, int n, int n2,
                                double *****CDM, double *****iDM0, double *****EDM, const double *ko,
                                const dcomplex *EVec1, int energy, int simd_mode, double *seconds)
{
  const double max_x = 60.0, FermiEps = 1.0e-13;
  const int comps = energy ? 10 : 6;
  const int kmin = is2[myid], kmax = ie2[myid], nk = kmax - kmin + 1 > 0 ? kmax - kmin + 1 : 0;
  OpenMXNcDMPanels P;
  double *weight = NULL, *eig = NULL, *dm = NULL, *rows = NULL;
  double stime, etime;
  int max_tno = 1, na = 0, k, a, i, j, c, GA_AN, LB_AN, p, ok = 1;

  if (!NcDM_OnePassEnabled(simd_mode)) return 0;
  memset(&P, 0, sizeof(P));   /* released below even when no band got planes */
  dtime(&stime);
  for (GA_AN = 1; GA_AN <= atomnum; GA_AN++) {
    const int tno = Spe_Total_CNO[WhatSpecies[GA_AN]];
    if (max_tno < tno) max_tno = tno;
  }
  weight = (double *)malloc(sizeof(double) * (size_t)(nk > 0 ? nk : 1));
  eig    = (double *)malloc(sizeof(double) * (size_t)(nk > 0 ? nk : 1));
  dm     = (double *)calloc((size_t)comps * (size_t)(size_H1 > 0 ? size_H1 : 1), sizeof(double));
  if (weight == NULL || eig == NULL || dm == NULL) ok = 0;

  /* the active bands of this rank in band order, with their Fermi weights */
  for (k = kmin; ok && k <= kmax; k++) {
    double x = (ko[k] - chemical->chemical_potential) * Beta, FermiF;
    if (x <= -max_x) x = -max_x;
    if (max_x <= x) x = max_x;
    FermiF = FermiFunc_NC(x, k);
    if (FermiEps < FermiF) {
      weight[na] = FermiF;
      eig[na] = ko[k];
      /* the planes of this band: the 2n-vector (k - kmin) of EVec1 */
      if (na == 0 && !NcDM_PanelsAllocate(&P, simd_mode, nk, n, max_tno)) { ok = 0; break; }
      {
        const dcomplex *v = EVec1 + (size_t)(k - kmin) * (size_t)n2;
        double *re1 = P.re1 + (size_t)na * P.stride, *im1 = P.im1 + (size_t)na * P.stride;
        double *re2 = P.re2 + (size_t)na * P.stride, *im2 = P.im2 + (size_t)na * P.stride;
        for (c = 0; c < n; c++) {
          re1[c] = v[c].r;     im1[c] = v[c].i;
          re2[c] = v[n + c].r; im2[c] = v[n + c].i;
        }
      }
      na++;
    }
  }
  if (ok && na == 0 && !NcDM_PanelsAllocate(&P, simd_mode, 0, n, max_tno)) ok = 0;
  if (ok) {
    rows = (double *)malloc(sizeof(double) * 4 * (size_t)max_tno * (size_t)(na > 0 ? na : 1));
    if (rows == NULL) ok = 0;
  }

  /* every (atom, neighbour) block, in the legacy p order */
  p = 0;
  for (GA_AN = 1; ok && GA_AN <= atomnum; GA_AN++) {
    const int tnoA = Spe_Total_CNO[WhatSpecies[GA_AN]], Anum = MP[GA_AN];
    /* the centre's rows [i][band], the four components apart */
    for (i = 0; i < tnoA; i++) {
      for (c = 0; c < 4; c++) P.rows[c * max_tno + i] = rows + ((size_t)c * max_tno + i) * (size_t)(na > 0 ? na : 1);
      for (a = 0; a < na; a++) {
        const size_t at = (size_t)a * P.stride + (size_t)(Anum - 1 + i);
        rows[((size_t)0 * max_tno + i) * na + a] = P.re1[at];
        rows[((size_t)1 * max_tno + i) * na + a] = P.im1[at];
        rows[((size_t)2 * max_tno + i) * na + a] = P.re2[at];
        rows[((size_t)3 * max_tno + i) * na + a] = P.im2[at];
      }
    }
    for (LB_AN = 0; ok && LB_AN <= FNAN[GA_AN]; LB_AN++) {
      const int GB_AN = natn[GA_AN][LB_AN];
      const int tnoB = Spe_Total_CNO[WhatSpecies[GB_AN]], Bnum = MP[GB_AN];
      OpenMXEnergyNcBlock block;
      int q;
      block.rows = tnoA; block.columns = tnoB; block.bands = na; block.stride = P.stride;
      block.re0_1 = P.rows; block.im0_1 = P.rows + max_tno;
      block.re0_2 = P.rows + 2 * max_tno; block.im0_2 = P.rows + 3 * max_tno;
      block.re1T_1 = P.re1 + (Bnum - 1); block.im1T_1 = P.im1 + (Bnum - 1);
      block.re1T_2 = P.re2 + (Bnum - 1); block.im1T_2 = P.im2 + (Bnum - 1);
      block.weight = weight; block.eigen = energy ? eig : NULL;
      block.phase = 0; block.co = 1.0; block.si = 0.0;
      for (q = 0; q < 10; q++) block.out[q] = q < comps ? dm + (size_t)q * size_H1 + p : NULL;
      block.out_stride = (size_t)tnoB;
      if (!openmx_core_energy_nc_block(&block, simd_mode)) ok = 0;
      p += tnoA * tnoB;
    }
  }

  if (ok) {
    /* one reduction for every component, then the legacy store */
    MPI_Allreduce(MPI_IN_PLACE, dm, comps * size_H1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
    p = 0;
    for (GA_AN = 1; GA_AN <= atomnum; GA_AN++) {
      const int MA_AN = F_G2M[GA_AN], tnoA = Spe_Total_CNO[WhatSpecies[GA_AN]], ID = G2ID[GA_AN];
      for (LB_AN = 0; LB_AN <= FNAN[GA_AN]; LB_AN++) {
        const int tnoB = Spe_Total_CNO[WhatSpecies[natn[GA_AN][LB_AN]]];
        if (myid == ID) {
          for (i = 0; i < tnoA; i++) {
            for (j = 0; j < tnoB; j++, p++) {
              CDM[0][MA_AN][LB_AN][i][j]  = dm[(size_t)0 * size_H1 + p];   /* Re11 */
              CDM[1][MA_AN][LB_AN][i][j]  = dm[(size_t)1 * size_H1 + p];   /* Re22 */
              CDM[2][MA_AN][LB_AN][i][j]  = dm[(size_t)2 * size_H1 + p];   /* Re12 */
              CDM[3][MA_AN][LB_AN][i][j]  = dm[(size_t)3 * size_H1 + p];   /* Im12 */
              iDM0[0][MA_AN][LB_AN][i][j] = dm[(size_t)4 * size_H1 + p];   /* Im11 */
              iDM0[1][MA_AN][LB_AN][i][j] = dm[(size_t)5 * size_H1 + p];   /* Im22 */
              if (energy) {
                EDM[0][MA_AN][LB_AN][i][j] = dm[(size_t)6 * size_H1 + p];  /* ReEDM11 */
                EDM[1][MA_AN][LB_AN][i][j] = dm[(size_t)7 * size_H1 + p];  /* ReEDM22 */
                EDM[2][MA_AN][LB_AN][i][j] = dm[(size_t)8 * size_H1 + p];  /* ReEDM12 */
                EDM[3][MA_AN][LB_AN][i][j] = dm[(size_t)9 * size_H1 + p];  /* ImEDM12 */
              }
            }
          }
        }
        else p += tnoA * tnoB;
      }
    }
  }
  NcDM_PanelsFree(&P);
  free(weight); free(eig); free(dm); free(rows);
  dtime(&etime);
  *seconds = etime - stime;
  return ok;
}

/* The neighbour planes of one k point of the band routine: EVec1 holds the
   orbital-major rows of this rank's bands (row c at c*nk, band k at offset
   k - is2), already weighted by sqrt(FermiF); the planes take the bands
   kmin..kmax.  0 (nothing kept) under the legacy policy or without memory:
   the legacy loops run. */
static int NcDM_Band_Panels(OpenMXNcDMPanels *P, int simd_mode, const dcomplex *EVec1, int nk, int is2_rank,
                            int kmin, int kmax, int n, int max_rows)
{
  const int bands = kmax - kmin + 1 > 0 ? kmax - kmin + 1 : 0;
  int c, b;
  if (!NcDM_OnePassEnabled(simd_mode)) { memset(P, 0, sizeof(*P)); return 0; }
  if (!NcDM_PanelsAllocate(P, simd_mode, bands, n, max_rows)) return 0;
  for (c = 0; c < n; c++) {
    const dcomplex *up = EVec1 + (size_t)c * (size_t)nk + (size_t)(kmin - is2_rank);
    const dcomplex *dn = EVec1 + (size_t)(n + c) * (size_t)nk + (size_t)(kmin - is2_rank);
    for (b = 0; b < bands; b++) {
      const size_t at = (size_t)b * P->stride + (size_t)c;
      P->re1[at] = up[b].r; P->im1[at] = up[b].i;
      P->re2[at] = dn[b].r; P->im2[at] = dn[b].i;
    }
  }
  return 1;
}

#endif

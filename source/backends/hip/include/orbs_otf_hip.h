/**********************************************************************
  orbs_otf_hip.h

  Device evaluation of the basis orbitals for the on-the-fly grid
  integrals (HIP port of the NVIDIA version's SOG_Device_* in
  Set_Orbitals_Grid.c / orbs_grid_gpu.h): the species' PAO radial
  tables, the cell translations and the grid frame are uploaded once per
  geometry, and the orbital values of any atom image at any list of grid
  points (an atom's sphere, a pair's overlap points) are evaluated on the
  device instead of read from the multi-GiB tables Orbs_Grid /
  Orbs_Grid_FNAN.  Those tables are built by the FP64 evaluation here
  (OTF_Orbitals_Eval_FP64), so the FP64 values equal them bit for bit
  at the same grid point and image; the double-float evaluation differs
  by one float ulp in about one value in 10^6.  Uncontracted bases with
  L <= 3 and at most 8 radial functions per L only.
***********************************************************************/
#ifndef ORBS_OTF_HIP_H
#define ORBS_OTF_HIP_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define OTF_L0MAX  3
#define OTF_MULMAX 8

/* one evaluation: the orbitals of species wan (no of them) about the atom
   at (gx, gy, gz) shifted by (ax, ay, az) */
typedef struct {
  int wan;
  int no;
  int pad0, pad1;
  double gx, gy, gz;
  double ax, ay, az;
} OTF_Atom;

/* the species tables on the host, flattened (built by Set_Orbitals_Grid.c) */
typedef struct {
  int species;
  const size_t *rv_off;    /* species + 1: first mesh value of each species in rv */
  const size_t *rwf_base;  /* species + 1: first radial value of each species in rwf */
  const int *mesh;         /* species */
  const int *maxl;         /* species */
  const int *nb;           /* species * (OTF_L0MAX + 1): radial functions per L */
  const double *rv;        /* every species' radial mesh */
  const double *rwf;       /* every species' radial functions, L-major then Mul */
  size_t rv_count, rwf_count;
  const double *atv;       /* atv_rows x 3: the cell translations */
  size_t atv_rows;
  int ng23, ng3;           /* Ngrid2 * Ngrid3, Ngrid3 */
  double g[3][3];          /* gtv[1..3][1..3] */
  double org[3];           /* Grid_Origin[1..3] */
} OTF_HostTables;

/* uploads the tables (double_float: also the (hi, lo) tables of the
   double-float evaluation); 1 on success, 0 when a host or device
   allocation or a transfer failed (worth a retry), -1 when the tables are
   outside the evaluator (no species, L > OTF_L0MAX, more than OTF_MULMAX
   radial functions of one L, fewer than 6 mesh points) */
int    OTF_Orbitals_Upload(const OTF_HostTables *t, int double_float);
void   OTF_Orbitals_Release(void);
int    OTF_Orbitals_Ready(void);
int    OTF_Orbitals_DoubleFloat(void);
size_t OTF_Orbitals_Bytes(void);

/* Device arrays.  Item p evaluates atoms[p] at npts[p] points; its k-th
   point is the sphere index Nc = idx[idx_off[p] + k] (k itself when idx is
   NULL) of a sphere whose grid indices and cells are gla[sph_off[p] + Nc],
   cla[sph_off[p] + Nc]; the no values of point k go to
   out[out_off[p] + k * no].  max_npts bounds every npts[p] (it sizes the
   launch).  Asynchronous on stream; 0 on success. */
int OTF_Orbitals_Eval(void *stream, int count, int max_npts, const OTF_Atom *atoms, const int *npts,
                      const size_t *idx_off, const size_t *out_off, const size_t *sph_off, const int *idx,
                      const int *gla, const int *cla, float *out);

/* the same in FP64 whatever the uploaded precision (the tables
   Orbs_Grid / Orbs_Grid_FNAN of Set_Orbitals_Grid.c) */
int OTF_Orbitals_Eval_FP64(void *stream, int count, int max_npts, const OTF_Atom *atoms, const int *npts,
                           const size_t *idx_off, const size_t *out_off, const size_t *sph_off, const int *idx,
                           const int *gla, const int *cla, float *out);

#ifdef __cplusplus
}
#endif

#endif

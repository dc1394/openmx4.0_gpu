/**********************************************************************
  orbs_grid_gpu.h

  The device evaluation of the basis orbitals (Set_Orbitals_Grid.c): the
  species' PAO radial tables, the cell translations and the grid frame
  held on the device (SOG_Device_Prepare, released by SOG_Device_Release
  and whenever the tables are rebuilt), and the orbital tiles the grid
  integrals evaluate on the fly from them.  The values equal the ones of
  the tables Orbs_Grid / Orbs_Grid_FNAN (the same device code fills
  those when the GPU table path is active).
***********************************************************************/
#ifndef ORBS_GRID_GPU_H
#define ORBS_GRID_GPU_H

#include <stddef.h>

typedef struct {
  int wan;              /* species of the evaluated atom */
  int no;               /* orbitals per grid point */
  int pt0;              /* first flat point index of this pair */
  int pad;
  size_t out;           /* first output slot in the chi buffer */
  double gx, gy, gz;    /* Gxyz of the evaluated atom */
  double ax, ay, az;    /* atv[Rnh]; zero for the part-1 self evaluation */
} SOG_GpuPair;

typedef struct {
  int ready;
  int unavailable;
  unsigned char *arena; /* acc_malloc */
  size_t bytes;
  const double *rv, *rwf, *atv;
  const size_t *rvo, *rwb;
  const int *spm, *spx, *spb;
  int ng23, ng3;
  double g[3][3];
  double org[3];
} SOG_DeviceTables;

int SOG_Device_Prepare(void);
void SOG_Device_Release(void);
const SOG_DeviceTables *SOG_Device_Tables(void);
size_t SOG_Device_Bytes(void);
void SOG_Device_EvalPairTiles(int pair_count, const SOG_GpuPair *pairs, const int *pair_NOLG,
                              const size_t *nolg_off, const size_t *out_off, const size_t *gla_off,
                              const int *nolg_Nc, const int *gla, const int *cla, float *out);
void SOG_Device_EvalAtomTile(const SOG_GpuPair *pair, int npts, const int *gla, const int *cla, float *out);

#endif

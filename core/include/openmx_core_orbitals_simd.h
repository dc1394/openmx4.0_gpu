#ifndef OPENMX_CORE_ORBITALS_SIMD_H
#define OPENMX_CORE_ORBITALS_SIMD_H
#include "openmx_core_grid.h"
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

/* One grid point of Set_Orbitals_Grid's capacity path: the species and the
   orbital count of the atom, the global grid index and the cell of the
   point, the atom's position and the cell shift of the pair (zero for the
   atom's own grid). The layout is that of the device task record. */
typedef struct { int wan, no, grid, cell; double xyz[3], shift[3]; } OpenMXOrbitalsPoint;

/* The flat PAO tables of every species, as SOG_capacity_recover stages them
   for the device: radii rv[rv_off[w] + m] for m < mesh[w], radial functions
   rwf[rwf_base[w] + k*mesh[w] + m] with k running over (L0, Mul0) in that
   order, cell translations translations[3*cell + a] for cell < cells, basis
   counts nb[w*(l0max+1) + L0] for L0 <= maxl[w]. Borrowed for the call. */
typedef struct {
  const double *rv, *rwf, *translations;
  const size_t *rv_off, *rwf_base;
  const int *mesh, *maxl, *nb;
  int species, l0max, mulmax, cells;
} OpenMXOrbitalsTables;

/* The grid frame: Ngrid2*Ngrid3, Ngrid3, the row stride of the output,
   gtv[i+1][j+1] as frame[3*i+j] and Grid_Origin[1..3]. */
typedef struct { int ng23, ng3, max_no; double frame[9], origin[3]; } OpenMXOrbitalsFrame;

/* out[i*max_no + j] for j < max_no: the primitive orbital values of point i
   (zero beyond the species' orbitals and outside the PAO mesh), bitwise the
   device routine SOG_point_eval of Set_Orbitals_Grid.c: direction cosines
   from the coordinates, the logarithmic-mesh estimate corrected to the exact
   interval, the cubic spline with its inner extrapolation, the explicit real
   harmonics up to L0 = 3. Returns 0 without writing when the mode is
   unsupported, a buffer is missing, a species exceeds L0 = 3 or eight radial
   functions per L0 or has fewer than six mesh points, or a point's species,
   cell, grid index or orbital count is inconsistent; else 1. Pure,
   re-entrant and allocation-free: callers split the points across threads. */
int openmx_core_orbitals_points(const OpenMXOrbitalsTables*,const OpenMXOrbitalsFrame*,
    const OpenMXOrbitalsPoint*,size_t count,float *out,int mode);

#ifdef __cplusplus
}
#endif
#endif

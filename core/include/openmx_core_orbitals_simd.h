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

/* A grid point of a force quadrature given by its displacement from the atom
   (x, y, z = the dx, dy, dz that Force3 forms from the grid coordinate, the
   cell translation and the atom position) with the atom's species and
   orbital count. No grid frame is needed, so the caller controls bitwise how
   the displacement is formed. */
typedef struct { int wan, no; double x, y, z; } OpenMXOrbitalsDisplacement;

/* out[(i*no + j)*3 + c] for point i, orbital j, c = 0,1,2 (rows follow the
   points: with mixed species the row of point i starts at 3 times the sum of
   the orbital counts of the points before it): the x, y, z
   derivatives dChi[1..3][j] of Get_dOrbitals_work (Get_dOrbitals.c; inlined
   in Force3), bitwise its arithmetic with this build's libm for acos, asin,
   sin and cos: the Rmin = 1e-13 shift of a point on the atom, the spherical
   angles, the radial spline and its derivative in both mesh branches, the
   real harmonics with their theta/phi derivatives for L0 <= 3, and the chain
   rule -dR*dChiR - dQ*dChiQ - dP*dChiP. Zero outside the PAO mesh. The
   orbital value dChi[0] is not written. Returns 0 without writing on an
   unsupported mode, a missing buffer, a species beyond L0 = 3 / eight radial
   functions / fewer than six mesh points, or a point whose species or
   orbital count is inconsistent; else 1. Pure, re-entrant, allocation-free. */
int openmx_core_orbitals_dpoints(const OpenMXOrbitalsTables*,const OpenMXOrbitalsDisplacement*,
    size_t count,double *out,int mode);

/* The same derivatives with sin/cos of theta and phi taken as the direction
   cosines r1/R, z/R, y/r1, x/r1 of the same branches instead of libm's
   sin(acos(.)) and sin(asin(.)): no transcendental calls, last-ulp
   differences from Get_dOrbitals_work (of the order of the libm differences
   between builds), bitwise across ISAs. Same contract otherwise. */
int openmx_core_orbitals_dpoints_cosines(const OpenMXOrbitalsTables*,const OpenMXOrbitalsDisplacement*,
    size_t count,double *out,int mode);

#ifdef __cplusplus
}
#endif
#endif

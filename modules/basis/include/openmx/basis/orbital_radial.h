#ifndef OPENMX_BASIS_ORBITAL_RADIAL_H
#define OPENMX_BASIS_ORBITAL_RADIAL_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  double value, radial_derivative;
} OpenMXOrbitalRadialSample;

/* Real-space orbital interpolation used by Get_Orbitals/Get_dOrbitals.
   This differs from the PAO/PhiF radial-table policy. Radius and grid are in
   bohr; radial_derivative is dR_l/dr, without the center-gradient minus sign.
   At least five finite, strictly increasing nonnegative mesh points and one
   finite row are required. angular is 0..29. Below the first mesh point the
   legacy cubic continuation is matched at index 4; all l>=2 use r^2/r^3.
   The first interval retains the reflected stencil; outside the final point
   both outputs are zero. No cutoff other than the last mesh point is applied.
   Validates the entire row/grid, allocates nothing on success, retains no
   pointers and initializes no runtime. Failure preserves out. */
int openmx_orbital_radial_sample(int angular,const double *radii,const double *values,
    size_t points,double radius,OpenMXOrbitalRadialSample *out,
    char *error,size_t capacity);
#ifdef __cplusplus
}
#endif
#endif

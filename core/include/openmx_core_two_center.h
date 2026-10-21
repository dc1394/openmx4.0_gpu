#ifndef OPENMX_CORE_TWO_CENTER_H
#define OPENMX_CORE_TWO_CENTER_H
#include "openmx_complex.h"
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

/* Spherical derivatives are evaluated by the caller's original recurrence.
   Supplying its trigonometric values also preserves the polar-axis branch. */
typedef struct {
  double distance, sin_theta, cos_theta, sin_phi, cos_phi;
  int onsite;
} OpenMXTwoCenterGeometry;
typedef struct {
  const dcomplex *angular; /* [channels, source_rows, source_columns] */
  size_t source_rows, source_columns, row, column;
  int left_angular, right_angular;
  const dcomplex *conversion; /* [conversion_orders, width, width] */
  size_t conversion_orders, conversion_width;
  OpenMXTwoCenterGeometry geometry;
} OpenMXTwoCenterInput;

/* Borrowed views owned by the enclosing overlap/nonlocal/VNA workspace.
   Channels: value, dx, dy, dz; overlap additionally has kinetic channels.
   momenta is null for four-channel workspaces, otherwise [3, rows, columns].
   Results are reused by the next transform; independent workers never share.
   Sources, scratch and outputs must be disjoint. */
typedef struct {
  size_t channels, rows, columns;
  dcomplex *left;
  double *values, *momenta;
} OpenMXTwoCenterBuffers;

/* No allocation, MPI, GPU initialization or mutable global state. Invalid
   dimensions/geometry leave all outputs intact. No exception crosses C. */
int openmx_core_two_center_transform(const OpenMXTwoCenterInput *input,
    const OpenMXTwoCenterBuffers *buffers, char *error, size_t capacity);
#ifdef __cplusplus
}
#endif
#endif

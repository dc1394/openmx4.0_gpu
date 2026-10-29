#ifndef OPENMX_VNA_TABLES_H
#define OPENMX_VNA_TABLES_H
#include "openmx_complex.h"
#include "openmx_core_radial_views.h"
#include <stddef.h>

/* Fourier/output tables remain borrowed for this synchronous invocation.
   Their ownership is independent of the immutable basis and geometry plan. */
typedef struct {
  OpenMXRadialView4 projectors;
  OpenMXProductRadialView products;
  OpenMXRadialView2 crude;
  const dcomplex *conversion;
  size_t conversion_width;
  const double *weights;
  double minimum_wavevector, maximum_wavevector;
  double *atom_seconds;
} SetProVnaTables;

static inline dcomplex SetPro_Conversion(const SetProVnaTables *tables,int angular,int row,int column)
{
  const size_t width=tables->conversion_width;
  return tables->conversion[((size_t)angular*width+row)*width+column];
}
#endif

#ifndef OPENMX_VNA_PREPARATION_H
#define OPENMX_VNA_PREPARATION_H
#include "openmx_core_vna_preparation.h"
#include <omp.h>

/* Internal CPU adapter. RF_BesselF is the legacy scalar interpolator;
   all loop bounds and table addresses come from the owned basis snapshot. */
static void SetPro_PrepareRadials(const OpenMXReciprocalViews *reciprocal, OpenMXVnaPreparationBuffers *preparation,
    const double *wavevectors, const double *weights, double Dk)
{
  const int points = preparation->points;
  for (int species=0; species<preparation->species; ++species) {
    for (int angular=0; angular<=preparation->maximum_angular[species]; ++angular) {
      const int count = preparation->basis_counts[(size_t)species*preparation->angular_channels+angular];
      for (int radial=0; radial<count; ++radial) {
        const size_t offset = openmx_core_vna_radial_offset(preparation,species,angular,radial);
        double *pro0 = preparation->weighted[0]+offset;
        double *pro1 = preparation->weighted[1]+offset;
#pragma omp parallel
        {
          const int thread = omp_get_thread_num(), threads = omp_get_num_threads();
          for (int i=thread*points/threads; i<(thread+1)*points/threads; ++i) {
            const double Normk = wavevectors[i];
            const double factor = 0.50*Dk*weights[i]*Normk*Normk;
            pro0[i] = RF_BesselF(reciprocal, species,angular,radial,Normk)*factor;
            pro1[i] = pro0[i]*Normk;
          }
        }
      }
    }
  }
}
#endif

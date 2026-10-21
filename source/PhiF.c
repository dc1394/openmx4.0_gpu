/**********************************************************************
  PhiF.c:

    PhiF.c is a subroutine to calculate the value of a radial function  
    at R.

  Log of PhiF.c:

     7/Apr/2004  Released by T.Ozaki

***********************************************************************/

#include <math.h>
#include "openmx_core_radial.h"

/* Legacy scalar ABI; invalid shape/coordinate returns NaN instead of reading
   outside the table. The caller provides finite, strictly increasing radii. */
double PhiF(double R, double *phi0, double *MRV, int Grid_Num)
{
  double value;
  if (!openmx_core_radial_sample(OPENMX_RADIAL_NEUTRAL_PROJECTOR,0,Grid_Num,
                                MRV,phi0,0.0,R,&value)) return NAN;
  return value;
}

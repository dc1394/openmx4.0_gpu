/**********************************************************************
  Spherical_Bessel.c:

     Spherical_Bessel.c is a subroutine to calculate the spherical
     Bessel functions and its derivative from 0 to lmax

  Log of Spherical_Bessel.c:

     08/Nov/2005  Released by T.Ozaki

***********************************************************************/

/* The implementation belongs to the C++ radial module. This C adapter keeps
   existing callers while removing their dependency on openmx_common.h. */
#include <stdio.h>
#include <stdlib.h>
#include "openmx_core.h"

void Spherical_Bessel(double x, int lmax, double *sb, double *dsb)
{
  if (!openmx_core_spherical_bessel(x,lmax,sb,dsb)) {
    fprintf(stderr,"Invalid argument to Spherical_Bessel: x=%g lmax=%d\n",x,lmax);
    exit(EXIT_FAILURE);
  }
}

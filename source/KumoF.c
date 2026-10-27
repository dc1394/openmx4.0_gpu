/**********************************************************************
  KumoF.c:

     KumoF.c is a subroutine to calculate an interpolated value of yv
     at x using the "Kumogata" interpolation.

  Log of KumoF.c:

     26/Feb./2012  Released by T. Ozaki

***********************************************************************/

#include "openmx_basis_density.h"

double KumoF(int N, double x, double *xv, double *rv, double *yv)
{
  return openmx_basis_density_value(N,x,xv,rv,yv);
}

/**********************************************************************
  Dr_KumoF.c:

   Dr_KumoF.c is a subroutine to calculate the radial derivative of 
   an interpolated value of yv at x using the "Kumogata" interpolation.

  Log of Dr_KumoF.c:

     26/Feb./2012  Released by T.Ozaki

***********************************************************************/

#include "openmx_basis_density.h"

double Dr_KumoF(int N, double x, double r, double *xv, double *rv, double *yv)
{
  return openmx_basis_density_derivative(N,x,r,xv,rv,yv);
}

/**********************************************************************
  Gauss_Legendre.c:

     Gauss_Legendre.c is a subrutine to generate radial grids for
     the Gauss_Legendre quadrature. 

  Log of Gauss_Legendre.c:

     22/Nov/2001  Released by T.Ozaki

***********************************************************************/

#include "openmx_basis_quadrature.h"
#include <stdio.h>

void Gauss_Legendre(int n, double x[], double w[], int *ncof, int *flag)
{
  char error[256];
  (void)ncof; /* The active legacy implementation left both control values alone. */
  if (!openmx_basis_gauss_legendre(n,x,w,error,sizeof(error))) {
    if (flag) *flag=0;
    fprintf(stderr,"Gauss_Legendre: %s\n",error);
    return;
  }
  /* Preserve the legacy N+1 scratch element; callers supply N+2 entries. */
  x[n]=x[n-1];
  w[n]=w[n-1];
}

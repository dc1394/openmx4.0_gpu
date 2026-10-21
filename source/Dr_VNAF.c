/**********************************************************************
  Dr_VNAF.c:

     Dr_VNAF.c is a subroutine to calculate the derivative, with
     respect to R, of neutral atom potential of one atom specified
     by "Gensi".

  Log of Dr_VNAF.c:

     22/Nov/2001  Released by T.Ozaki

***********************************************************************/

#include <stdio.h>
#include <math.h>
#include "openmx_common.h"

double Dr_VNAF(const OpenMXAtomicViews *atomic, int Gensi, double R)
{
  int mp_min,mp_max,m,po;
  double h1,h2,h3,f1,f2,f3,f4;
  double g1,g2,x1,x2,y1,y2,f,df;
  double rm,a,b,y12,y22,result;

  mp_min = 0;
  mp_max = atomic->species.vps_mesh[Gensi] - 1;

  if (openmx_radial_row2(&atomic->vps_radii,Gensi)[atomic->species.vps_mesh[Gensi]-1]<R){
    result = 0.0;
  }
  else if (R<openmx_radial_row2(&atomic->vps_radii,Gensi)[0]){
    po = 1;
    m = 4;
    rm = openmx_radial_row2(&atomic->vps_radii,Gensi)[m];

    h1 = openmx_radial_row2(&atomic->vps_radii,Gensi)[m-1] - openmx_radial_row2(&atomic->vps_radii,Gensi)[m-2];
    h2 = openmx_radial_row2(&atomic->vps_radii,Gensi)[m]   - openmx_radial_row2(&atomic->vps_radii,Gensi)[m-1];
    h3 = openmx_radial_row2(&atomic->vps_radii,Gensi)[m+1] - openmx_radial_row2(&atomic->vps_radii,Gensi)[m];

    f1 = openmx_radial_row2(&atomic->neutral_potential,Gensi)[m-2];
    f2 = openmx_radial_row2(&atomic->neutral_potential,Gensi)[m-1];
    f3 = openmx_radial_row2(&atomic->neutral_potential,Gensi)[m];
    f4 = openmx_radial_row2(&atomic->neutral_potential,Gensi)[m+1];

    g1 = ((f3-f2)*h1/h2 + (f2-f1)*h2/h1)/(h1+h2);
    g2 = ((f4-f3)*h2/h3 + (f3-f2)*h3/h2)/(h2+h3);

    x1 = rm - openmx_radial_row2(&atomic->vps_radii,Gensi)[m-1];
    x2 = rm - openmx_radial_row2(&atomic->vps_radii,Gensi)[m];
    y1 = x1/h2;
    y2 = x2/h2;
    y12 = y1*y1;
    y22 = y2*y2;

    f =  y22*(3.0*f2 + h2*g1 + (2.0*f2 + h2*g1)*y2)
       + y12*(3.0*f3 - h2*g2 - (2.0*f3 - h2*g2)*y1);

    df = 2.0*y2/h2*(3.0*f2 + h2*g1 + (2.0*f2 + h2*g1)*y2)
       + y22*(2.0*f2 + h2*g1)/h2
       + 2.0*y1/h2*(3.0*f3 - h2*g2 - (2.0*f3 - h2*g2)*y1)
       - y12*(2.0*f3 - h2*g2)/h2;

    a = 0.5*df/rm;
    b = f - a*rm*rm;      
    result = 2.0*a*R;
  }
  else{

    do{
      m = (mp_min + mp_max)/2;
      if (openmx_radial_row2(&atomic->vps_radii,Gensi)[m]<R)
        mp_min = m;
      else 
        mp_max = m;
    }
    while((mp_max-mp_min)!=1);
    m = mp_max;

    if (m<2)
      m = 2;
    else if (atomic->species.vps_mesh[Gensi]<=m)
      m = atomic->species.vps_mesh[Gensi] - 2;

    /****************************************************
                   Spline like interpolation
    ****************************************************/

    h1 = openmx_radial_row2(&atomic->vps_radii,Gensi)[m-1] - openmx_radial_row2(&atomic->vps_radii,Gensi)[m-2];
    h2 = openmx_radial_row2(&atomic->vps_radii,Gensi)[m]   - openmx_radial_row2(&atomic->vps_radii,Gensi)[m-1];
    h3 = openmx_radial_row2(&atomic->vps_radii,Gensi)[m+1] - openmx_radial_row2(&atomic->vps_radii,Gensi)[m];

    f1 = openmx_radial_row2(&atomic->neutral_potential,Gensi)[m-2];
    f2 = openmx_radial_row2(&atomic->neutral_potential,Gensi)[m-1];
    f3 = openmx_radial_row2(&atomic->neutral_potential,Gensi)[m];
    f4 = openmx_radial_row2(&atomic->neutral_potential,Gensi)[m+1];

    /****************************************************
                   Treatment of edge points
    ****************************************************/

    if (m==1){
      h1 = -(h2+h3);
      f1 = f4;
    }
    if (m==(atomic->species.vps_mesh[Gensi]-1)){
      h3 = -(h1+h2);
      f4 = f1;
    }

    /****************************************************
                Calculate the value at R
    ****************************************************/

    g1 = ((f3-f2)*h1/h2 + (f2-f1)*h2/h1)/(h1+h2);
    g2 = ((f4-f3)*h2/h3 + (f3-f2)*h3/h2)/(h2+h3);

    x1 = R - openmx_radial_row2(&atomic->vps_radii,Gensi)[m-1];
    x2 = R - openmx_radial_row2(&atomic->vps_radii,Gensi)[m];
    y1 = x1/h2;
    y2 = x2/h2;

    f =  y2*y2*(3.0*f2 + h2*g1 + (2.0*f2 + h2*g1)*y2)
       + y1*y1*(3.0*f3 - h2*g2 - (2.0*f3 - h2*g2)*y1);

    df = 2.0*y2/h2*(3.0*f2 + h2*g1 + (2.0*f2 + h2*g1)*y2)
       + y2*y2*(2.0*f2 + h2*g1)/h2
       + 2.0*y1/h2*(3.0*f3 - h2*g2 - (2.0*f3 - h2*g2)*y1)
       - y1*y1*(2.0*f3 - h2*g2)/h2;

    result = df;

  }
  return result;
}







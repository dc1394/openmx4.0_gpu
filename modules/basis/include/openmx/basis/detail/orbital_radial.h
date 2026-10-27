#ifndef OPENMX_BASIS_DETAIL_ORBITAL_RADIAL_H
#define OPENMX_BASIS_DETAIL_ORBITAL_RADIAL_H
#include "../orbital_radial.h"

/* Numerical formulas from OpenMX Get_Orbitals.c (T. Ozaki, 24 April 2002)
   and Get_dOrbitals.c (T. Ozaki, 22 November 2001). Keep their association,
   reflected endpoint samples and origin continuation. See the package LICENSE. These unchecked helpers are shared with the
   native CPU adapters. The caller validates the grid, row and radius. */
typedef struct {
  int outside, below;
  size_t im1, im, i1, i4;
  double radius, rm, h2, y1, y2, y12, y22, dum1, dum2, dum3, dum4;
} OpenMXBasisOrbitalRadialPlan;

static inline OpenMXBasisOrbitalRadialPlan openmx_basis_detail_orbital_radial_plan(
    const double *radii,size_t points,double radius)
{
  #ifdef __cplusplus
  OpenMXBasisOrbitalRadialPlan p{};
  #else
  OpenMXBasisOrbitalRadialPlan p={0};
  #endif
  size_t m;
  double h1,h3,x1,x2,dum;
  p.radius=radius;p.outside=radii[points-1]<radius;
  if(p.outside)return p;
  p.below=radius<radii[0];
  if(p.below)m=4;
  else {
    size_t lo=0,hi=points-1;
    do {
      m=(lo+hi)/2;
      if(radii[m]<radius)lo=m;else hi=m;
    }while(hi-lo!=1);
    m=hi;
  }
  p.im1=m-1;p.im=m;p.i1=m==1?m+1:m-2;p.i4=m==points-1?m-2:m+1;
  p.rm=radii[m];
  h1=radii[m-1]-radii[p.i1];
  p.h2=radii[m]-radii[m-1];
  h3=radii[p.i4]-radii[m];
  x1=(p.below?p.rm:radius)-radii[m-1];
  x2=(p.below?p.rm:radius)-radii[m];
  p.y1=x1/p.h2;p.y2=x2/p.h2;p.y12=p.y1*p.y1;p.y22=p.y2*p.y2;
  dum=h1+p.h2;p.dum1=h1/p.h2/dum;p.dum2=p.h2/h1/dum;
  dum=p.h2+h3;p.dum3=p.h2/h3/dum;p.dum4=h3/p.h2/dum;
  return p;
}

static inline OpenMXOrbitalRadialSample openmx_basis_detail_orbital_radial_apply(
    const OpenMXBasisOrbitalRadialPlan *p,const double *row,int angular,int derivatives)
{
  OpenMXOrbitalRadialSample result={0.0,0.0};
  double f1,f2,f3,f4,dum,g1,g2,f,df=0.0;
  if(p->outside)return result;
  f1=row[p->i1];f2=row[p->im1];f3=row[p->im];f4=row[p->i4];
  dum=f3-f2;
  g1=dum*p->dum1+(f2-f1)*p->dum2;
  g2=(f4-f3)*p->dum3+dum*p->dum4;
  f=p->y22*(3.0*f2+p->h2*g1+(2.0*f2+p->h2*g1)*p->y2)
    +p->y12*(3.0*f3-p->h2*g2-(2.0*f3-p->h2*g2)*p->y1);
  if(derivatives || p->below)
    df=2.0*p->y2/p->h2*(3.0*f2+p->h2*g1+(2.0*f2+p->h2*g1)*p->y2)
      +p->y22*(2.0*f2+p->h2*g1)/p->h2
      +2.0*p->y1/p->h2*(3.0*f3-p->h2*g2-(2.0*f3-p->h2*g2)*p->y1)
      -p->y12*(2.0*f3-p->h2*g2)/p->h2;
  if(p->below) {
    double a,b,c,d;
    const double rm=p->rm,R=p->radius;
    if(angular==0){a=0.0;b=0.5*df/rm;c=0.0;d=f-b*rm*rm;}
    else if(angular==1){a=(rm*df-f)/(2.0*rm*rm*rm);b=0.0;c=df-3.0*a*rm*rm;d=0.0;}
    else {b=(3.0*f-rm*df)/(rm*rm);a=(f-b*rm*rm)/(rm*rm*rm);c=0.0;d=0.0;}
    result.value=a*R*R*R+b*R*R+c*R+d;
    if(derivatives)result.radial_derivative=3.0*a*R*R+2.0*b*R+c;
  }else {
    result.value=f;
    if(derivatives)result.radial_derivative=df;
  }
  return result;
}
#endif

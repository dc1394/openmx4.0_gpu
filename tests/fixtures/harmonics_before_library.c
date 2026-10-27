/* Frozen before independent harmonics extraction at 17f46679.
   legacy_Associated_Legendre and legacy_ComplexSH: openmx_common.c, T. Ozaki, 22/Nov/2001.
   Orbital harmonic: corrected g-orbital implementation at the same revision.
   Test oracle only; do not update this fixture to follow production changes. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#define PI 3.1415926535897932384626
#define sgn(x) ((x)<0 ? -1. : 1.)

void legacy_Associated_Legendre(int l, int m, double x, double ALeg[2])
{
  /*****************************************************
   associated Legendre polynomial Plm(x) with integers
   m (0<=m<=l) and l. The range of x is -1<=x<=1.
   Its derivative is given by
   dP_l^m(x)/dtheta =
   1/sqrt{1-x*x}*(l*x*Plm(x)-(l+m)*P{l-1}m(x))
   where x=cos(theta)
  ******************************************************/

  double cut0=1.0e-24,cut1=1.0e-12;
  double Pm,Pm1,f,p0,p1,dP,tmp0;
  int i,ll;

  if (m<0 || m>l || fabs(x)>1.0){
    printf("Invalid arguments in routine legacy_Associated_Legendre\n");
    exit(0);
  }
  else if ((1.0-cut0)<fabs(x)){
    x = sgn(x)*(1.0-cut0);
  }

  /* calculate Pm */

  Pm = 1.0;

  if (m>0){

    f = 1.0;
    tmp0 = sqrt((1.0 - x)*(1.0 + x));
    for (i=1; i<=m; i++){
      Pm = -Pm*f*tmp0;
      f += 2.0;
    }
  }

  if (l==m){
    p0 = Pm;
    p1 = 0.0;

    tmp0 = sqrt(1.0-x*x);

    if (cut1<tmp0)  dP = ((double)l*x*p0 - (double)(l+m)*p1)/tmp0;
    else            dP = 0.0;

    ALeg[0] = p0;
    ALeg[1] = dP;
  }

  else{

    /* calculate Pm1 */

    Pm1 = x*(2.0*(double)m + 1.0)*Pm;

    if (l==(m+1)){
      p0 = Pm1;
      p1 = Pm;

      tmp0 = sqrt(1.0-x*x);

      if (cut1<tmp0) dP = ((double)l*x*p0 - (double)(l+m)*p1)/tmp0;
      else           dP = 0.0;

      ALeg[0] = p0;
      ALeg[1] = dP;
    }

    /* calculate Plm, l>m+1 */

    else{

      for (ll=m+2; ll<=l; ll++){
        tmp0 = (x*(2.0*(double)ll-1.0)*Pm1 - ((double)ll+(double)m-1.0)*Pm)/(double)(ll-m);
        Pm  = Pm1;
        Pm1 = tmp0;
      }
      p0 = Pm1;
      p1 = Pm;

      tmp0 = sqrt(1.0-x*x);

      if (cut1<tmp0)  dP = ((double)l*x*p0 - (double)(l+m)*p1)/tmp0;
      else            dP = 0.0;

      ALeg[0] = p0;
      ALeg[1] = dP;
    }
  }
}

void legacy_ComplexSH(int l, int m, double theta, double phi,
               double SH[2], double dSHt[2], double dSHp[2])
{
  int i;
  long double fact0,fact1;
  double co,si,tmp0,ALeg[2];

  /* Compute (l-|m|)! */

  fact0 = 1.0;
  for (i=1; i<=(l-abs(m)); i++){
    fact0 *= i;
  }

  /* Compute (l+|m|)! */
  fact1 = 1.0;
  for (i=1; i<=(l+abs(m)); i++){
    fact1 *= i;
  }

  /* sqrt((2*l+1)/(4*PI)*(l-|m|)!/(l+|m|)!) */

  tmp0 = sqrt((2.0*(double)l+1.0)/(4.0*PI)*fact0/fact1);

  /* P_l^|m| */

  legacy_Associated_Legendre(l,abs(m),cos(theta),ALeg);

  /* Ylm */

  co = cos((double)m*phi);
  si = sin((double)m*phi);

  if (0<=m){
    SH[0]   = tmp0*ALeg[0]*co;
    SH[1]   = tmp0*ALeg[0]*si;
    dSHt[0] = tmp0*ALeg[1]*co;
    dSHt[1] = tmp0*ALeg[1]*si;
    dSHp[0] = -(double)m*tmp0*ALeg[0]*si;
    dSHp[1] =  (double)m*tmp0*ALeg[0]*co;
  }
  else{
    if (abs(m)%2==0){

      SH[0]   = tmp0*ALeg[0]*co;
      SH[1]   = tmp0*ALeg[0]*si;
      dSHt[0] = tmp0*ALeg[1]*co;
      dSHt[1] = tmp0*ALeg[1]*si;
      dSHp[0] = -(double)m*tmp0*ALeg[0]*si;
      dSHp[1] =  (double)m*tmp0*ALeg[0]*co;

    }
    else{

      SH[0]   = -tmp0*ALeg[0]*co;
      SH[1]   = -tmp0*ALeg[0]*si;
      dSHt[0] = -tmp0*ALeg[1]*co;
      dSHt[1] = -tmp0*ALeg[1]*si;
      dSHp[0] =  (double)m*tmp0*ALeg[0]*si;
      dSHp[1] = -(double)m*tmp0*ALeg[0]*co;

    }
  }

}

void legacy_openmx_orbital_complex_sh(int l, int m, double st, double ct,
    double phi, double value[2], double dtheta[2], double dphi_over_sin[2])
{
  const int a = m < 0 ? -m : m;
  double t = 1.0, dt = 0.0, previous = 0.0, dprevious = 0.0;
  double power = 1.0;
  long double numerator = 1.0L, denominator = 1.0L;
  for (int k=1; k<=a; ++k) t *= -(2*k-1);
  for (int n=a+1; n<=l; ++n) {
    const double next = ((2*n-1)*ct*t - (n+a-1)*previous)/(n-a);
    const double dnext = ((2*n-1)*(t+ct*dt) - (n+a-1)*dprevious)/(n-a);
    previous = t;
    dprevious = dt;
    t = next;
    dt = dnext;
  }
  for (int k=1; k<a; ++k) power *= st;
  for (int k=1; k<=l-a; ++k) numerator *= k;
  for (int k=1; k<=l+a; ++k) denominator *= k;
  double normalization = sqrt((2.0*l+1.0)/(4.0*3.1415926535897932384626)
                              *numerator/denominator);
  if (m < 0 && a%2) normalization = -normalization;
  const double radial = a == 0 ? t : power*st*t;
  const double polar = a == 0 ? -st*dt : power*(a*ct*t-st*st*dt);
  const double azimuthal = a == 0 ? 0.0 : m*power*t;
  const double c = cos(m*phi), s = sin(m*phi);
  value[0] = normalization*radial*c;
  value[1] = normalization*radial*s;
  dtheta[0] = normalization*polar*c;
  dtheta[1] = normalization*polar*s;
  dphi_over_sin[0] = -normalization*azimuthal*s;
  dphi_over_sin[1] = normalization*azimuthal*c;
}

/**********************************************************************
  openmx_common.c:
  
     openmx_common.c is a collective routine of subroutines
     which are often used.

  Log of openmx_common.c:

     22/Nov/2001  Released by T.Ozaki

***********************************************************************/
  
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include "openmx_common.h"
#include "mpi.h"
#include "lapack_prototypes.h"

void Generation_ATV(int N)
{
  int Rn,i,j,k;
  double di,dj,dk;

  Rn = 1;
  di = -(N+1);
  for (i=-N; i<=N; i++){
    di = di + 1.0;
    dj = -(N+1);
    for (j=-N; j<=N; j++){
      dj = dj + 1.0;
      dk = -(N+1);
      for (k=-N; k<=N; k++){

	dk = dk + 1.0;
	if (i==0 && j==0 && k==0){
	  atv[0][1] = 0.0;
	  atv[0][2] = 0.0;
	  atv[0][3] = 0.0;
	  atv_ijk[0][1] = 0;
	  atv_ijk[0][2] = 0;
	  atv_ijk[0][3] = 0;
	  ratv[i+N][j+N][k+N] = 0;
	}
	else{
	  atv[Rn][1] = di*tv[1][1] + dj*tv[2][1] + dk*tv[3][1];
	  atv[Rn][2] = di*tv[1][2] + dj*tv[2][2] + dk*tv[3][2];
	  atv[Rn][3] = di*tv[1][3] + dj*tv[2][3] + dk*tv[3][3];
	  atv_ijk[Rn][1] = i;
	  atv_ijk[Rn][2] = j;
	  atv_ijk[Rn][3] = k;
	  ratv[i+N][j+N][k+N] = Rn;
	  Rn = Rn + 1;
	}
      }
    }
  }

}

void Cross_Product(double a[4], double b[4], double c[4])
{
  c[1] = a[2]*b[3] - a[3]*b[2]; 
  c[2] = a[3]*b[1] - a[1]*b[3]; 
  c[3] = a[1]*b[2] - a[2]*b[1];
}

double Dot_Product(double a[4], double b[4])
{
  double sum;
  sum = a[1]*b[1] + a[2]*b[2] + a[3]*b[3]; 
  return sum;
}


int R_atv(int N, int i, int j, int k)
{
  int Rn;
  Rn = ratv[i+N][j+N][k+N];
  return Rn;
}

dcomplex Complex(double re, double im)
{
  dcomplex c;
  c.r = re;
  c.i = im;
  return c;
}

dcomplex Cadd(dcomplex a, dcomplex b)
{
  dcomplex c;
  c.r = a.r + b.r;
  c.i = a.i + b.i;
  return c;
}

dcomplex Csub(dcomplex a, dcomplex b)
{
  dcomplex c;
  c.r = a.r - b.r;
  c.i = a.i - b.i;
  return c;
}

dcomplex Cmul(dcomplex a, dcomplex b)
{
  dcomplex c;
  c.r = a.r*b.r - a.i*b.i;
  c.i = a.i*b.r + a.r*b.i;
  return c;
}


dcomplex Conjg(dcomplex z)
{
  dcomplex c;
  c.r =  z.r;
  c.i = -z.i;
  return c;
}


dcomplex Cdiv(dcomplex a, dcomplex b)
{
  dcomplex c;
  double r,den;
  if (fabs(b.r) >= fabs(b.i)){
    r = b.i/b.r;
    den = b.r + r*b.i;
    c.r = (a.r + r*a.i)/den;
    c.i = (a.i - r*a.r)/den;
  }
  else{
    r = b.r/b.i;
    den = b.i + r*b.r;
    c.r = (a.r*r + a.i)/den;
    c.i = (a.i*r - a.r)/den;
  }
  return c;
}


double Cabs(dcomplex z)
{
  double x,y,ans,temp;
  x = fabs(z.r);
  y = fabs(z.i);
  if (x<1.0e-30)
    ans = y;
  else if (y<1.0e-30)
    ans = x;
  else if (x>y){
    temp = y/x;
    ans = x*sqrt(1.0+temp*temp);
  } else{
    temp = x/y;
    ans = y*sqrt(1.0+temp*temp);
  }
  return ans;
}


dcomplex Csqrt(dcomplex z)
{
  dcomplex c;
  double x,y,w,r;
  if ( fabs(z.r)<1.0e-30 && z.i<1.0e-30 ){
    c.r = 0.0;
    c.i = 0.0;
    return c;
  }
  else{
    x = fabs(z.r);
    y = fabs(z.i);
    if (x>=y){
      r = y/x;
      w = sqrt(x)*sqrt(0.5*(1.0+sqrt(1.0+r*r)));
    } else {
      r = x/y;
      w = sqrt(y)*sqrt(0.5*(r+sqrt(1.0+r*r)));
    }
    if (z.r>=0.0){
      c.r = w;
      c.i = z.i/(2.0*w);
    } else {
      c.i = (z.i>=0) ? w : -w;
      c.r = z.i/(2.0*c.i);
    }
    return c;
  }
}



dcomplex Csin(dcomplex a)
{ 
  double ar;

  if(fabs(a.r) > 2. * PI)
    ar = (int)(a.r / 2. / PI) * 2. * PI;
  else
    ar = a.r; 

  return Complex(sin(ar) * cosh(a.i), cos(ar) * sinh(a.i));
}


dcomplex Ccos(dcomplex a)
{
  double ar;

  if(fabs(a.r) > 2. * PI)
    ar = (int)(a.r / 2. / PI) * 2. * PI;
  else	
    ar = a.r;

  return Complex(cos(ar) * cosh(a.i), - sin(ar) * sinh(a.i));
}


dcomplex Cexp(dcomplex a)
{
  double x;

  x = exp(a.r);
  return Complex(x * cos(a.i), x * sin(a.i));
}



dcomplex RCadd(double x, dcomplex a)
{
  dcomplex c;
  c.r = x + a.r;
  c.i = a.i;
  return c;
}

dcomplex RCsub(double x, dcomplex a)
{
  dcomplex c;
  c.r = x - a.r;
  c.i = -a.i;
  return c;
}

dcomplex RCmul(double x, dcomplex a)
{
  dcomplex c;
  c.r = x*a.r;
  c.i = x*a.i;
  return c;
}

dcomplex CRmul(dcomplex a, double x)
{
  dcomplex c;
  c.r = x*a.r;
  c.i = x*a.i;
  return c;
}

dcomplex RCdiv(double x, dcomplex a)
{
  dcomplex c;
  double xx,yy,w;
  xx = a.r;
  yy = a.i;
  w = xx*xx+yy*yy;
  c.r = x*a.r/w;
  c.i = -x*a.i/w;
  return c;
}

dcomplex CRC(dcomplex a, double x, dcomplex b)
{
  dcomplex c;
  c.r = a.r - x - b.r;
  c.i = a.i - b.i;
  return c;
}


void Cswap(dcomplex *a, dcomplex *b)
{
  dcomplex temp;
  temp.r = a->r;
  temp.i = a->i;
  a->r = b->r;
  a->i = b->i;
  b->r = temp.r;
  b->i = temp.i;
}



double rnd(double width)
{
  /****************************************************
       This rnd() function generates random number
                -width/2 to width/2
  ****************************************************/

  double result;

  result = rand();
  result = result*width/(double)RAND_MAX - 0.5*width;
  return result;
}

double rnd0to1()
{
  /****************************************************
   This rnd() function generates random number 0 to 1
  ****************************************************/

  double result;

  result = rand();
  result /= (double)RAND_MAX;
  return result;
}


double sgn(double nu)
{
  double result;
  if (nu<0.0)
    result = -1.0;
  else
    result = 1.0;
  return result;
}

double isgn(int nu)
{
  double result;
  if (nu<0)
    result = -1.0;
  else
    result = 1.0;
  return result;
}

void fnjoint(char name1[YOUSO10],char name2[YOUSO10],char name3[YOUSO10])
{
  char name4[YOUSO10];
  char *f1 = name1,
       *f2 = name2,
       *f3 = name3,
       *f4 = name4;

  while(*f1)
    {
      *f4 = *f1;
      f1++;
      f4++;
    }
  while(*f2)
    {
      *f4 = *f2;
      f2++;
      f4++;
    }
  while(*f3)
    {
      *f4 = *f3;
      f3++;
      f4++;
    }
  *f4 = *f3;
  chcp(name3,name4);
}

void fnjoint2(char name1[YOUSO10], char name2[YOUSO10],
              char name3[YOUSO10], char name4[YOUSO10])
{
  char *f1 = name1,
       *f2 = name2,
       *f3 = name3,
       *f4 = name4;

  while(*f1)
    {
      *f4 = *f1;
      f1++;
      f4++;
    }
  while(*f2)
    {
      *f4 = *f2;
      f2++;
      f4++;
    }
  while(*f3)
    {
      *f4 = *f3;
      f3++;
      f4++;
    }
  *f4 = *f3;
}


void chcp(char name1[YOUSO10],char name2[YOUSO10])
{

  /****************************************************
                    name2 -> name1
  ****************************************************/

  char *f1 = name1,
    *f2 = name2;
  while(*f2){
    *f1 = *f2;
    f1++;
    f2++;
  }
  *f1 = *f2;
}

int SEQ(char str1[YOUSO10], char str2[YOUSO10])
{
  
  int i,result,l1,l2;

  l1 = strlen(str1);
  l2 = strlen(str2);

  result = 1; 
  if (l1 == l2){
    for (i=0; i<=l1-1; i++){
      if (str1[i]!=str2[i])  result = 0;   
    }
  }
  else
    result = 0; 

  return result;
}


void spline3(double r, double r1, double rcut,
             double g, double dg, double value[2])
{

  /****************************************************
    r    ->  a given distance 
    r1   ->  a shortest distatnce in a spline function
    rcut ->  a cut-off distance in a spline function
    g    ->  a function value at r1
    dg   ->  a derivative at r1

    a function value at r -> value[0]
    a derivative at r     -> value[1]
  ****************************************************/

  double a0,a1,a2,a3;
  double rcut2,rcut3,r12,r13,dr; 

  rcut2 = rcut*rcut;
  rcut3 = rcut2*rcut;
  r12 = r1*r1;
  r13 = r12*r1;
  dr = r1 - rcut;
  a3 = (2.0*g-dg*dr)/(rcut3-r13+3.0*r12*rcut-3.0*r1*rcut2);
  a2 = 0.5*dg/dr - 1.5*(r1+rcut)*a3;
  a1 = -rcut*dg/dr + 3.0*r1*rcut*a3;
  a0 = -a1*rcut-a2*rcut2-a3*rcut3;
  value[0] = a0+a1*r+a2*r*r+a3*r*r*r;
  value[1] = a1+2.0*a2*r+3.0*a3*r*r;
}

double largest(double a, double b)
{
  double result;

  if (b<=a) result = a;
  else      result = b;
  return result;
}

double smallest(double a, double b)
{
  double result;

  if (b<=a) result = b;
  else      result = a;
  return result;
}



double weight_CWF2( double e, 
                    double ChemP, 
                    double in0, 
                    double in1, 
                    double out0, 
                    double out1) 
{
  double c0,c1,c2,c3,c4,w,x,y;

  c0 = 1.0;
  c1 = 0.0;
  c2 = -6.0;
  c3 = 8.0;
  c4 = -3.0;

  x = e - ChemP;

  /* out0 in0 in1 out1 */

  if (x<=out0){
    w = 0.0;
  }
  else if (out0<x && x<=in0){
    y = (in0-x)/(in0-out0);
    w = c0 + c1*y + c2*y*y + c3*y*y*y + c4*y*y*y*y;   
  }
  else if (in0<x && x<=in1){
    w = 1.0;
  }
  else if (in1<x && x<=out1){
    y = (x-in1)/(out1-in1);
    w = c0 + c1*y + c2*y*y + c3*y*y*y + c4*y*y*y*y;   
  }
  else if (out1<x){
    w = 0.0;
  }

  return w;
}












void asbessel(int n, double x, double sbe[2])
{

  /* This rourine suffers from numerical instabilities for a small x */

  double x2,x3,x4,x5,x6,x7,x8;

  if (6<n){
    printf("n=%2d is not supported in asbessel.",n);
    exit(0);
  }

  switch(n){

    case 0:
      x2 = x*x;
      sbe[0] = sin(x)/x;
      sbe[1] = (cos(x) - sin(x)/x)/x;
    break;

    case 1:
      x2 = x*x;
      x3 = x2*x;
      sbe[0] = -(cos(x)/x) + sin(x)/(x*x);
      sbe[1] = (2.0*cos(x))/x2 - (2.0*sin(x))/x3 + sin(x)/x;
    break;

    case 2:  
      x2 = x*x;
      x3 = x2*x;
      x4 = x2*x2;
      sbe[0] = (-3.0*cos(x))/x2 + (3.0*sin(x))/x3 - sin(x)/x;
      sbe[1] = (9.0*cos(x))/x3 - cos(x)/x - (9.0*sin(x))/x4
              + (4.0*sin(x))/x2;
    break;

    case 3: 
      x2 = x*x;
      x3 = x2*x;
      x4 = x2*x2;
      x5 = x4*x;
      sbe[0] = (-15.0*cos(x))/x3 + cos(x)/x + (15.0*sin(x))/x4
              - (6.0*sin(x))/x2;
      sbe[1] = (60.0*cos(x))/x4 - (7.0*cos(x))/x2
              - (60.0*sin(x))/x5 + (27.0*sin(x))/x3 - sin(x)/x;
    break;

    case 4:
      x2 = x*x;
      x3 = x2*x;
      x4 = x2*x2;
      x5 = x4*x;
      x6 = x5*x;
      sbe[0] = (-105.0*cos(x))/x4 + (10.0*cos(x))/x2 + (105.0*sin(x))/x5
              - (45.0*sin(x))/x3 + sin(x)/x;
      sbe[1] = (525.0*cos(x))/x5 - (65.0*cos(x))/x3 + cos(x)/x
              - (525.0*sin(x))/x6 + (240.0*sin(x))/x4 - (11.0*sin(x))/x2;
    break;

    case 5:  
      x2 = x*x;
      x3 = x2*x;
      x4 = x2*x2;
      x5 = x4*x;
      x6 = x5*x;
      x7 = x3*x4;
      sbe[0] = (-945.0*cos(x))/x5 + (105.0*cos(x))/x3 - cos(x)/x
              + (945.0*sin(x))/x6 - (420.0*sin(x))/x4 + (15.0*sin(x))/x2;
      sbe[1] = (5670.0*cos(x))/x6 - (735.0*cos(x))/x4 + (16.0*cos(x))/x2
              - (5670.0*sin(x))/x7 + (2625.0*sin(x))/x5 - (135.0*sin(x))/x3
              + sin(x)/x;
    break;

    case 6:  
      x2 = x*x;
      x3 = x2*x;
      x4 = x2*x2;
      x5 = x4*x;
      x6 = x5*x;
      x7 = x3*x4;
      x8 = x4*x4;
      sbe[0] = (-10395.0*cos(x))/x6 + (1260.0*cos(x))/x4
              - (21.0*cos(x))/x2 + (10395.0*sin(x))/x7
              - (4725.0*sin(x))/x5 + (210.0*sin(x))/x3 - sin(x)/x;
              
      sbe[1] = (72765.0*cos(x))/x7 - (9765.0*cos(x))/x5
              + (252.0*cos(x))/x3 - cos(x)/x - (72765.0*sin(x))/x8
              + (34020.0*sin(x))/x6 - (1890.0*sin(x))/x4
              + (22.0*sin(x))/x2;
    break;

  } 
}



void ComplexSH(int l, int m, double theta, double phi,
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

  Associated_Legendre(l,abs(m),cos(theta),ALeg);

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




void Associated_Legendre(int l, int m, double x, double ALeg[2])
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
    printf("Invalid arguments in routine Associated_Legendre\n");
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



dcomplex Im_pow(int fu, int Ls)
{

  dcomplex Cres;

  if (fu!=1 && fu!=-1){
    printf("Invalid arguments in Im_pow\n");
    exit(0);
  } 
  else{
    
    if (fu==1){
      if (Ls%2==0){
        if (Ls%4==0)     Cres = Complex( 1.0, 0.0);
        else             Cres = Complex(-1.0, 0.0);
      }
      else{
        if ((Ls+1)%4==0) Cres = Complex( 0.0,-1.0);
        else             Cres = Complex( 0.0, 1.0);
      }
    }

    else{
      if (Ls%2==0){
        if (Ls%4==0)     Cres = Complex( 1.0, 0.0);
        else             Cres = Complex(-1.0, 0.0);
      }
      else{
        if ((Ls+1)%4==0) Cres = Complex( 0.0, 1.0);
        else             Cres = Complex( 0.0,-1.0);
      }
    } 

  }

  return Cres;
}

void GN2N(int GN, int N3[4])
{
  int n1,n2,n3;

  n1 = GN/(Ngrid2*Ngrid3);
  n2 = (GN - n1*(Ngrid2*Ngrid3))/Ngrid3;
  n3 = GN - n1*(Ngrid2*Ngrid3) - n2*Ngrid3;
  N3[1] = n1;
  N3[2] = n2;
  N3[3] = n3;
}


void GN2N_EGAC(int GN, int N3[4])
{
  int n1,n2,n3;
  int m1,m2,m3;

  m1 = atomnum;
  m2 = SpinP_switch+1;
  m3 = EGAC_Npoles;

  n1 = GN/(m2*m3);
  n2 = (GN - n1*m2*m3)/m3;
  n3 = GN - n1*m2*m3 - n2*m3;
  N3[1] = n1 + 1; /* index of atom starts from 1. */
  N3[2] = n2;
  N3[3] = n3;
}


int AproxFactN(int N0)
{
  int N1,N18,po,N[5],i;
  int result;

  printf("AproxFactN: N0=%d\n",N0);

  if (N0<=4){
    result = 4;
  } 
  else{ 
 
    po = 0;
    N1 = 1;
    do{
      N1 = 2*N1;
      if (N0<N1){
        N18 = N1/16;
        po = 1;
      } 
    } while (po==0);

    printf("AproxFactN: N18=%d\n",N18);
      
    N[0] = N18*4;
    N[1] = N18*5;
    N[2] = N18*6; 
    N[3] = N18*7;
    N[4] = N18*8;

    po = 0;
    i = -1; 
    do{
      i++;
      printf("AproxFactN: i,N[i],N0=%d %d %d\n",i,N[i],N0);
      if (0<=(N[i]-N0)) po = 1;
    } while(po==0);

    result = N[i];    
  }

  return result;
}



void Get_Grid_XYZ(int GN, double xyz[4])
{
  int n1,n2,n3;

  n1 = GN/(Ngrid2*Ngrid3);
  n2 = (GN - n1*(Ngrid2*Ngrid3))/Ngrid3;
  n3 = GN - n1*(Ngrid2*Ngrid3) - n2*Ngrid3;

  xyz[1] = (double)n1*gtv[1][1] + (double)n2*gtv[2][1]
         + (double)n3*gtv[3][1] + Grid_Origin[1];
  xyz[2] = (double)n1*gtv[1][2] + (double)n2*gtv[2][2]
         + (double)n3*gtv[3][2] + Grid_Origin[2];
  xyz[3] = (double)n1*gtv[1][3] + (double)n2*gtv[2][3]
         + (double)n3*gtv[3][3] + Grid_Origin[3];
}


void k_inversion(int i,  int j,  int k, 
                 int mi, int mj, int mk, 
                 int *ii, int *ij, int *ik )
{
       *ii= mi-i-1;
       *ij= mj-j-1;
       *ik= mk-k-1;
}


char *string_tolower(char *buf, char *buf1)
{
  char *c=buf;
  char *c1=buf1;

  while (*c){
    *c1=tolower(*c);
    c++;
    c1++;
  }
 return buf;
}


double FermiFunc(double x, int spin, int orb, int *index, double *popn)
{
  int q;
  double FermiF;
  
  FermiF = 1.0/(1.0 + exp(x));
  
  if (empty_occupation_flag==1){
    for (q=0; q<empty_occupation_num; q++){
      if (spin==empty_occupation_spin[q] && orb==empty_occupation_orbital[q]){
	FermiF = 0.0; 
      }
    }
  }

  if (empty_states_flag==1){
 
    double p0,dp;

    p0 = 0.50*(popn[empty_states_orbitals_num] + popn[empty_states_orbitals_num-1]);

    for (q=0; q<empty_states_orbitals_num*3; q++){ 

      if (index[q]==orb){
        dp = (popn[q] - 0.5)*40.0;
        FermiF = FermiF*(1.0/(1.0 + exp(dp)));

      }
    }        
  }
	
  return FermiF;
}


double FermiFunc_NC(double x, int orb)
{
  int q;
  double FermiF;
  
  FermiF = 1.0/(1.0 + exp(x));
  
  if (empty_occupation_flag==1){
    for (q=0; q<empty_occupation_num; q++){
      if (orb==empty_occupation_orbital[q]){
	FermiF = 0.0; 
      }
    }
  }

  return FermiF;
}



void inverse_LU(int n, double *A)
{
  static char *thisprogram="inverse_LU";
  int *ipiv;
  double *work;
  int lwork;
  int info,i,j;

  /* L*U factorization */

  ipiv = (int*)malloc(sizeof(int)*(n+2));

  F77_NAME(dgetrf,DGETRF)(&n,&n,A,&n,ipiv,&info);
  
  if ( info !=0 ) {
    printf("dgetrf failed, info=%i, %s\n",info,thisprogram);
  }

  /* inverse L*U factorization */

  lwork = 4*n;
  work = (double*)malloc(sizeof(double)*lwork);

  F77_NAME(dgetri,DGETRI)(&n, A, &n, ipiv, work, &lwork, &info);

  if ( info !=0 ) {
    printf("dgetrf failed, info=%i, %s\n",info,thisprogram);
  }

  free(work); free(ipiv);
}





void Set_BLACS_variables ( MPI_Comm MPI_Curret_Comm_WD, int numprocs, int myid,
                           int nrows, int ncols, int *np_rows, int *np_cols, 
                           int *nblk, int *my_prow, int *my_pcol, int *na_rows, int *na_cols, 
                           int *bhandle, int *ictxt, int *na_rows_max, int *na_cols_max,
                           int desc[9])
{
  int nblk_m;
  int ZERO=0, ONE=1,info;

  (*np_cols) = (int)(sqrt((float)numprocs));
  do{
    if((numprocs%(*np_cols))==0) break;
    (*np_cols)--;
  } while ((*np_cols)>=2);
  (*np_rows) = numprocs/(*np_cols);

  nblk_m = NBLK;
  while((nblk_m*(*np_rows)>nrows || nblk_m*(*np_cols)>ncols) && (nblk_m > 1)){
    nblk_m /= 2;
  }
  if(nblk_m<1) nblk_m = 1;

  MPI_Allreduce(&nblk_m,nblk,1,MPI_INT,MPI_MIN,mpi_comm_level1);

  (*my_prow) = myid/(*np_cols);
  (*my_pcol) = myid%(*np_cols);

  (*na_rows) = numroc_(&nrows, nblk, my_prow, &ZERO, np_rows);
  (*na_cols) = numroc_(&ncols, nblk, my_pcol, &ZERO, np_cols );

  (*bhandle) = Csys2blacs_handle(MPI_Curret_Comm_WD);
  (*ictxt) = (*bhandle);
  Cblacs_gridinit(ictxt, "Row", *np_rows, *np_cols);
	
  MPI_Allreduce(na_rows,na_rows_max,1,MPI_INT,MPI_MAX,MPI_Curret_Comm_WD);
  MPI_Allreduce(na_cols,na_cols_max,1,MPI_INT,MPI_MAX,MPI_Curret_Comm_WD);

  descinit_(desc, &nrows, &ncols, nblk, nblk, &ZERO, &ZERO, ictxt, na_rows, &info);
}















void Set_Lebedev_Grid(int Np, double **Leb_Grid_XYZW)
{
  /* 590 */

  if (Np==590){

    Leb_Grid_XYZW[   0][0]= 1.000000000000000;
    Leb_Grid_XYZW[   0][1]= 0.000000000000000;
    Leb_Grid_XYZW[   0][2]= 0.000000000000000;
    Leb_Grid_XYZW[   0][3]= 0.000309512129531;

    Leb_Grid_XYZW[   1][0]=-1.000000000000000;
    Leb_Grid_XYZW[   1][1]= 0.000000000000000;
    Leb_Grid_XYZW[   1][2]= 0.000000000000000;
    Leb_Grid_XYZW[   1][3]= 0.000309512129531;

    Leb_Grid_XYZW[   2][0]= 0.000000000000000;
    Leb_Grid_XYZW[   2][1]= 1.000000000000000;
    Leb_Grid_XYZW[   2][2]= 0.000000000000000;
    Leb_Grid_XYZW[   2][3]= 0.000309512129531;

    Leb_Grid_XYZW[   3][0]= 0.000000000000000;
    Leb_Grid_XYZW[   3][1]=-1.000000000000000;
    Leb_Grid_XYZW[   3][2]= 0.000000000000000;
    Leb_Grid_XYZW[   3][3]= 0.000309512129531;

    Leb_Grid_XYZW[   4][0]= 0.000000000000000;
    Leb_Grid_XYZW[   4][1]= 0.000000000000000;
    Leb_Grid_XYZW[   4][2]= 1.000000000000000;
    Leb_Grid_XYZW[   4][3]= 0.000309512129531;

    Leb_Grid_XYZW[   5][0]= 0.000000000000000;
    Leb_Grid_XYZW[   5][1]= 0.000000000000000;
    Leb_Grid_XYZW[   5][2]=-1.000000000000000;
    Leb_Grid_XYZW[   5][3]= 0.000309512129531;

    Leb_Grid_XYZW[   6][0]= 0.577350269189626;
    Leb_Grid_XYZW[   6][1]= 0.577350269189626;
    Leb_Grid_XYZW[   6][2]= 0.577350269189626;
    Leb_Grid_XYZW[   6][3]= 0.001852379698597;

    Leb_Grid_XYZW[   7][0]=-0.577350269189626;
    Leb_Grid_XYZW[   7][1]= 0.577350269189626;
    Leb_Grid_XYZW[   7][2]= 0.577350269189626;
    Leb_Grid_XYZW[   7][3]= 0.001852379698597;

    Leb_Grid_XYZW[   8][0]= 0.577350269189626;
    Leb_Grid_XYZW[   8][1]=-0.577350269189626;
    Leb_Grid_XYZW[   8][2]= 0.577350269189626;
    Leb_Grid_XYZW[   8][3]= 0.001852379698597;

    Leb_Grid_XYZW[   9][0]=-0.577350269189626;
    Leb_Grid_XYZW[   9][1]=-0.577350269189626;
    Leb_Grid_XYZW[   9][2]= 0.577350269189626;
    Leb_Grid_XYZW[   9][3]= 0.001852379698597;

    Leb_Grid_XYZW[  10][0]= 0.577350269189626;
    Leb_Grid_XYZW[  10][1]= 0.577350269189626;
    Leb_Grid_XYZW[  10][2]=-0.577350269189626;
    Leb_Grid_XYZW[  10][3]= 0.001852379698597;

    Leb_Grid_XYZW[  11][0]=-0.577350269189626;
    Leb_Grid_XYZW[  11][1]= 0.577350269189626;
    Leb_Grid_XYZW[  11][2]=-0.577350269189626;
    Leb_Grid_XYZW[  11][3]= 0.001852379698597;

    Leb_Grid_XYZW[  12][0]= 0.577350269189626;
    Leb_Grid_XYZW[  12][1]=-0.577350269189626;
    Leb_Grid_XYZW[  12][2]=-0.577350269189626;
    Leb_Grid_XYZW[  12][3]= 0.001852379698597;

    Leb_Grid_XYZW[  13][0]=-0.577350269189626;
    Leb_Grid_XYZW[  13][1]=-0.577350269189626;
    Leb_Grid_XYZW[  13][2]=-0.577350269189626;
    Leb_Grid_XYZW[  13][3]= 0.001852379698597;

    Leb_Grid_XYZW[  14][0]= 0.704095493822747;
    Leb_Grid_XYZW[  14][1]= 0.704095493822747;
    Leb_Grid_XYZW[  14][2]= 0.092190407076898;
    Leb_Grid_XYZW[  14][3]= 0.001871790639278;

    Leb_Grid_XYZW[  15][0]=-0.704095493822747;
    Leb_Grid_XYZW[  15][1]= 0.704095493822747;
    Leb_Grid_XYZW[  15][2]= 0.092190407076898;
    Leb_Grid_XYZW[  15][3]= 0.001871790639278;

    Leb_Grid_XYZW[  16][0]= 0.704095493822747;
    Leb_Grid_XYZW[  16][1]=-0.704095493822747;
    Leb_Grid_XYZW[  16][2]= 0.092190407076898;
    Leb_Grid_XYZW[  16][3]= 0.001871790639278;

    Leb_Grid_XYZW[  17][0]=-0.704095493822747;
    Leb_Grid_XYZW[  17][1]=-0.704095493822747;
    Leb_Grid_XYZW[  17][2]= 0.092190407076898;
    Leb_Grid_XYZW[  17][3]= 0.001871790639278;

    Leb_Grid_XYZW[  18][0]= 0.704095493822747;
    Leb_Grid_XYZW[  18][1]= 0.704095493822747;
    Leb_Grid_XYZW[  18][2]=-0.092190407076898;
    Leb_Grid_XYZW[  18][3]= 0.001871790639278;

    Leb_Grid_XYZW[  19][0]=-0.704095493822747;
    Leb_Grid_XYZW[  19][1]= 0.704095493822747;
    Leb_Grid_XYZW[  19][2]=-0.092190407076898;
    Leb_Grid_XYZW[  19][3]= 0.001871790639278;

    Leb_Grid_XYZW[  20][0]= 0.704095493822747;
    Leb_Grid_XYZW[  20][1]=-0.704095493822747;
    Leb_Grid_XYZW[  20][2]=-0.092190407076898;
    Leb_Grid_XYZW[  20][3]= 0.001871790639278;

    Leb_Grid_XYZW[  21][0]=-0.704095493822747;
    Leb_Grid_XYZW[  21][1]=-0.704095493822747;
    Leb_Grid_XYZW[  21][2]=-0.092190407076898;
    Leb_Grid_XYZW[  21][3]= 0.001871790639278;

    Leb_Grid_XYZW[  22][0]= 0.704095493822747;
    Leb_Grid_XYZW[  22][1]= 0.092190407076898;
    Leb_Grid_XYZW[  22][2]= 0.704095493822747;
    Leb_Grid_XYZW[  22][3]= 0.001871790639278;

    Leb_Grid_XYZW[  23][0]=-0.704095493822747;
    Leb_Grid_XYZW[  23][1]= 0.092190407076898;
    Leb_Grid_XYZW[  23][2]= 0.704095493822747;
    Leb_Grid_XYZW[  23][3]= 0.001871790639278;

    Leb_Grid_XYZW[  24][0]= 0.704095493822747;
    Leb_Grid_XYZW[  24][1]=-0.092190407076898;
    Leb_Grid_XYZW[  24][2]= 0.704095493822747;
    Leb_Grid_XYZW[  24][3]= 0.001871790639278;

    Leb_Grid_XYZW[  25][0]=-0.704095493822747;
    Leb_Grid_XYZW[  25][1]=-0.092190407076898;
    Leb_Grid_XYZW[  25][2]= 0.704095493822747;
    Leb_Grid_XYZW[  25][3]= 0.001871790639278;

    Leb_Grid_XYZW[  26][0]= 0.704095493822747;
    Leb_Grid_XYZW[  26][1]= 0.092190407076898;
    Leb_Grid_XYZW[  26][2]=-0.704095493822747;
    Leb_Grid_XYZW[  26][3]= 0.001871790639278;

    Leb_Grid_XYZW[  27][0]=-0.704095493822747;
    Leb_Grid_XYZW[  27][1]= 0.092190407076898;
    Leb_Grid_XYZW[  27][2]=-0.704095493822747;
    Leb_Grid_XYZW[  27][3]= 0.001871790639278;

    Leb_Grid_XYZW[  28][0]= 0.704095493822747;
    Leb_Grid_XYZW[  28][1]=-0.092190407076898;
    Leb_Grid_XYZW[  28][2]=-0.704095493822747;
    Leb_Grid_XYZW[  28][3]= 0.001871790639278;

    Leb_Grid_XYZW[  29][0]=-0.704095493822747;
    Leb_Grid_XYZW[  29][1]=-0.092190407076898;
    Leb_Grid_XYZW[  29][2]=-0.704095493822747;
    Leb_Grid_XYZW[  29][3]= 0.001871790639278;

    Leb_Grid_XYZW[  30][0]= 0.092190407076898;
    Leb_Grid_XYZW[  30][1]= 0.704095493822747;
    Leb_Grid_XYZW[  30][2]= 0.704095493822747;
    Leb_Grid_XYZW[  30][3]= 0.001871790639278;

    Leb_Grid_XYZW[  31][0]=-0.092190407076898;
    Leb_Grid_XYZW[  31][1]= 0.704095493822747;
    Leb_Grid_XYZW[  31][2]= 0.704095493822747;
    Leb_Grid_XYZW[  31][3]= 0.001871790639278;

    Leb_Grid_XYZW[  32][0]= 0.092190407076898;
    Leb_Grid_XYZW[  32][1]=-0.704095493822747;
    Leb_Grid_XYZW[  32][2]= 0.704095493822747;
    Leb_Grid_XYZW[  32][3]= 0.001871790639278;

    Leb_Grid_XYZW[  33][0]=-0.092190407076898;
    Leb_Grid_XYZW[  33][1]=-0.704095493822747;
    Leb_Grid_XYZW[  33][2]= 0.704095493822747;
    Leb_Grid_XYZW[  33][3]= 0.001871790639278;

    Leb_Grid_XYZW[  34][0]= 0.092190407076898;
    Leb_Grid_XYZW[  34][1]= 0.704095493822747;
    Leb_Grid_XYZW[  34][2]=-0.704095493822747;
    Leb_Grid_XYZW[  34][3]= 0.001871790639278;

    Leb_Grid_XYZW[  35][0]=-0.092190407076898;
    Leb_Grid_XYZW[  35][1]= 0.704095493822747;
    Leb_Grid_XYZW[  35][2]=-0.704095493822747;
    Leb_Grid_XYZW[  35][3]= 0.001871790639278;

    Leb_Grid_XYZW[  36][0]= 0.092190407076898;
    Leb_Grid_XYZW[  36][1]=-0.704095493822747;
    Leb_Grid_XYZW[  36][2]=-0.704095493822747;
    Leb_Grid_XYZW[  36][3]= 0.001871790639278;

    Leb_Grid_XYZW[  37][0]=-0.092190407076898;
    Leb_Grid_XYZW[  37][1]=-0.704095493822747;
    Leb_Grid_XYZW[  37][2]=-0.704095493822747;
    Leb_Grid_XYZW[  37][3]= 0.001871790639278;

    Leb_Grid_XYZW[  38][0]= 0.680774406645524;
    Leb_Grid_XYZW[  38][1]= 0.680774406645524;
    Leb_Grid_XYZW[  38][2]= 0.270356088359165;
    Leb_Grid_XYZW[  38][3]= 0.001858812585438;

    Leb_Grid_XYZW[  39][0]=-0.680774406645524;
    Leb_Grid_XYZW[  39][1]= 0.680774406645524;
    Leb_Grid_XYZW[  39][2]= 0.270356088359165;
    Leb_Grid_XYZW[  39][3]= 0.001858812585438;

    Leb_Grid_XYZW[  40][0]= 0.680774406645524;
    Leb_Grid_XYZW[  40][1]=-0.680774406645524;
    Leb_Grid_XYZW[  40][2]= 0.270356088359165;
    Leb_Grid_XYZW[  40][3]= 0.001858812585438;

    Leb_Grid_XYZW[  41][0]=-0.680774406645524;
    Leb_Grid_XYZW[  41][1]=-0.680774406645524;
    Leb_Grid_XYZW[  41][2]= 0.270356088359165;
    Leb_Grid_XYZW[  41][3]= 0.001858812585438;

    Leb_Grid_XYZW[  42][0]= 0.680774406645524;
    Leb_Grid_XYZW[  42][1]= 0.680774406645524;
    Leb_Grid_XYZW[  42][2]=-0.270356088359165;
    Leb_Grid_XYZW[  42][3]= 0.001858812585438;

    Leb_Grid_XYZW[  43][0]=-0.680774406645524;
    Leb_Grid_XYZW[  43][1]= 0.680774406645524;
    Leb_Grid_XYZW[  43][2]=-0.270356088359165;
    Leb_Grid_XYZW[  43][3]= 0.001858812585438;

    Leb_Grid_XYZW[  44][0]= 0.680774406645524;
    Leb_Grid_XYZW[  44][1]=-0.680774406645524;
    Leb_Grid_XYZW[  44][2]=-0.270356088359165;
    Leb_Grid_XYZW[  44][3]= 0.001858812585438;

    Leb_Grid_XYZW[  45][0]=-0.680774406645524;
    Leb_Grid_XYZW[  45][1]=-0.680774406645524;
    Leb_Grid_XYZW[  45][2]=-0.270356088359165;
    Leb_Grid_XYZW[  45][3]= 0.001858812585438;

    Leb_Grid_XYZW[  46][0]= 0.680774406645524;
    Leb_Grid_XYZW[  46][1]= 0.270356088359165;
    Leb_Grid_XYZW[  46][2]= 0.680774406645524;
    Leb_Grid_XYZW[  46][3]= 0.001858812585438;

    Leb_Grid_XYZW[  47][0]=-0.680774406645524;
    Leb_Grid_XYZW[  47][1]= 0.270356088359165;
    Leb_Grid_XYZW[  47][2]= 0.680774406645524;
    Leb_Grid_XYZW[  47][3]= 0.001858812585438;

    Leb_Grid_XYZW[  48][0]= 0.680774406645524;
    Leb_Grid_XYZW[  48][1]=-0.270356088359165;
    Leb_Grid_XYZW[  48][2]= 0.680774406645524;
    Leb_Grid_XYZW[  48][3]= 0.001858812585438;

    Leb_Grid_XYZW[  49][0]=-0.680774406645524;
    Leb_Grid_XYZW[  49][1]=-0.270356088359165;
    Leb_Grid_XYZW[  49][2]= 0.680774406645524;
    Leb_Grid_XYZW[  49][3]= 0.001858812585438;

    Leb_Grid_XYZW[  50][0]= 0.680774406645524;
    Leb_Grid_XYZW[  50][1]= 0.270356088359165;
    Leb_Grid_XYZW[  50][2]=-0.680774406645524;
    Leb_Grid_XYZW[  50][3]= 0.001858812585438;

    Leb_Grid_XYZW[  51][0]=-0.680774406645524;
    Leb_Grid_XYZW[  51][1]= 0.270356088359165;
    Leb_Grid_XYZW[  51][2]=-0.680774406645524;
    Leb_Grid_XYZW[  51][3]= 0.001858812585438;

    Leb_Grid_XYZW[  52][0]= 0.680774406645524;
    Leb_Grid_XYZW[  52][1]=-0.270356088359165;
    Leb_Grid_XYZW[  52][2]=-0.680774406645524;
    Leb_Grid_XYZW[  52][3]= 0.001858812585438;

    Leb_Grid_XYZW[  53][0]=-0.680774406645524;
    Leb_Grid_XYZW[  53][1]=-0.270356088359165;
    Leb_Grid_XYZW[  53][2]=-0.680774406645524;
    Leb_Grid_XYZW[  53][3]= 0.001858812585438;

    Leb_Grid_XYZW[  54][0]= 0.270356088359165;
    Leb_Grid_XYZW[  54][1]= 0.680774406645524;
    Leb_Grid_XYZW[  54][2]= 0.680774406645524;
    Leb_Grid_XYZW[  54][3]= 0.001858812585438;

    Leb_Grid_XYZW[  55][0]=-0.270356088359165;
    Leb_Grid_XYZW[  55][1]= 0.680774406645524;
    Leb_Grid_XYZW[  55][2]= 0.680774406645524;
    Leb_Grid_XYZW[  55][3]= 0.001858812585438;

    Leb_Grid_XYZW[  56][0]= 0.270356088359165;
    Leb_Grid_XYZW[  56][1]=-0.680774406645524;
    Leb_Grid_XYZW[  56][2]= 0.680774406645524;
    Leb_Grid_XYZW[  56][3]= 0.001858812585438;

    Leb_Grid_XYZW[  57][0]=-0.270356088359165;
    Leb_Grid_XYZW[  57][1]=-0.680774406645524;
    Leb_Grid_XYZW[  57][2]= 0.680774406645524;
    Leb_Grid_XYZW[  57][3]= 0.001858812585438;

    Leb_Grid_XYZW[  58][0]= 0.270356088359165;
    Leb_Grid_XYZW[  58][1]= 0.680774406645524;
    Leb_Grid_XYZW[  58][2]=-0.680774406645524;
    Leb_Grid_XYZW[  58][3]= 0.001858812585438;

    Leb_Grid_XYZW[  59][0]=-0.270356088359165;
    Leb_Grid_XYZW[  59][1]= 0.680774406645524;
    Leb_Grid_XYZW[  59][2]=-0.680774406645524;
    Leb_Grid_XYZW[  59][3]= 0.001858812585438;

    Leb_Grid_XYZW[  60][0]= 0.270356088359165;
    Leb_Grid_XYZW[  60][1]=-0.680774406645524;
    Leb_Grid_XYZW[  60][2]=-0.680774406645524;
    Leb_Grid_XYZW[  60][3]= 0.001858812585438;

    Leb_Grid_XYZW[  61][0]=-0.270356088359165;
    Leb_Grid_XYZW[  61][1]=-0.680774406645524;
    Leb_Grid_XYZW[  61][2]=-0.680774406645524;
    Leb_Grid_XYZW[  61][3]= 0.001858812585438;

    Leb_Grid_XYZW[  62][0]= 0.637254693925875;
    Leb_Grid_XYZW[  62][1]= 0.637254693925875;
    Leb_Grid_XYZW[  62][2]= 0.433373868777154;
    Leb_Grid_XYZW[  62][3]= 0.001852028828296;

    Leb_Grid_XYZW[  63][0]=-0.637254693925875;
    Leb_Grid_XYZW[  63][1]= 0.637254693925875;
    Leb_Grid_XYZW[  63][2]= 0.433373868777154;
    Leb_Grid_XYZW[  63][3]= 0.001852028828296;

    Leb_Grid_XYZW[  64][0]= 0.637254693925875;
    Leb_Grid_XYZW[  64][1]=-0.637254693925875;
    Leb_Grid_XYZW[  64][2]= 0.433373868777154;
    Leb_Grid_XYZW[  64][3]= 0.001852028828296;

    Leb_Grid_XYZW[  65][0]=-0.637254693925875;
    Leb_Grid_XYZW[  65][1]=-0.637254693925875;
    Leb_Grid_XYZW[  65][2]= 0.433373868777154;
    Leb_Grid_XYZW[  65][3]= 0.001852028828296;

    Leb_Grid_XYZW[  66][0]= 0.637254693925875;
    Leb_Grid_XYZW[  66][1]= 0.637254693925875;
    Leb_Grid_XYZW[  66][2]=-0.433373868777154;
    Leb_Grid_XYZW[  66][3]= 0.001852028828296;

    Leb_Grid_XYZW[  67][0]=-0.637254693925875;
    Leb_Grid_XYZW[  67][1]= 0.637254693925875;
    Leb_Grid_XYZW[  67][2]=-0.433373868777154;
    Leb_Grid_XYZW[  67][3]= 0.001852028828296;

    Leb_Grid_XYZW[  68][0]= 0.637254693925875;
    Leb_Grid_XYZW[  68][1]=-0.637254693925875;
    Leb_Grid_XYZW[  68][2]=-0.433373868777154;
    Leb_Grid_XYZW[  68][3]= 0.001852028828296;

    Leb_Grid_XYZW[  69][0]=-0.637254693925875;
    Leb_Grid_XYZW[  69][1]=-0.637254693925875;
    Leb_Grid_XYZW[  69][2]=-0.433373868777154;
    Leb_Grid_XYZW[  69][3]= 0.001852028828296;

    Leb_Grid_XYZW[  70][0]= 0.637254693925875;
    Leb_Grid_XYZW[  70][1]= 0.433373868777154;
    Leb_Grid_XYZW[  70][2]= 0.637254693925875;
    Leb_Grid_XYZW[  70][3]= 0.001852028828296;

    Leb_Grid_XYZW[  71][0]=-0.637254693925875;
    Leb_Grid_XYZW[  71][1]= 0.433373868777154;
    Leb_Grid_XYZW[  71][2]= 0.637254693925875;
    Leb_Grid_XYZW[  71][3]= 0.001852028828296;

    Leb_Grid_XYZW[  72][0]= 0.637254693925875;
    Leb_Grid_XYZW[  72][1]=-0.433373868777154;
    Leb_Grid_XYZW[  72][2]= 0.637254693925875;
    Leb_Grid_XYZW[  72][3]= 0.001852028828296;

    Leb_Grid_XYZW[  73][0]=-0.637254693925875;
    Leb_Grid_XYZW[  73][1]=-0.433373868777154;
    Leb_Grid_XYZW[  73][2]= 0.637254693925875;
    Leb_Grid_XYZW[  73][3]= 0.001852028828296;

    Leb_Grid_XYZW[  74][0]= 0.637254693925875;
    Leb_Grid_XYZW[  74][1]= 0.433373868777154;
    Leb_Grid_XYZW[  74][2]=-0.637254693925875;
    Leb_Grid_XYZW[  74][3]= 0.001852028828296;

    Leb_Grid_XYZW[  75][0]=-0.637254693925875;
    Leb_Grid_XYZW[  75][1]= 0.433373868777154;
    Leb_Grid_XYZW[  75][2]=-0.637254693925875;
    Leb_Grid_XYZW[  75][3]= 0.001852028828296;

    Leb_Grid_XYZW[  76][0]= 0.637254693925875;
    Leb_Grid_XYZW[  76][1]=-0.433373868777154;
    Leb_Grid_XYZW[  76][2]=-0.637254693925875;
    Leb_Grid_XYZW[  76][3]= 0.001852028828296;

    Leb_Grid_XYZW[  77][0]=-0.637254693925875;
    Leb_Grid_XYZW[  77][1]=-0.433373868777154;
    Leb_Grid_XYZW[  77][2]=-0.637254693925875;
    Leb_Grid_XYZW[  77][3]= 0.001852028828296;

    Leb_Grid_XYZW[  78][0]= 0.433373868777154;
    Leb_Grid_XYZW[  78][1]= 0.637254693925875;
    Leb_Grid_XYZW[  78][2]= 0.637254693925875;
    Leb_Grid_XYZW[  78][3]= 0.001852028828296;

    Leb_Grid_XYZW[  79][0]=-0.433373868777154;
    Leb_Grid_XYZW[  79][1]= 0.637254693925875;
    Leb_Grid_XYZW[  79][2]= 0.637254693925875;
    Leb_Grid_XYZW[  79][3]= 0.001852028828296;

    Leb_Grid_XYZW[  80][0]= 0.433373868777154;
    Leb_Grid_XYZW[  80][1]=-0.637254693925875;
    Leb_Grid_XYZW[  80][2]= 0.637254693925875;
    Leb_Grid_XYZW[  80][3]= 0.001852028828296;

    Leb_Grid_XYZW[  81][0]=-0.433373868777154;
    Leb_Grid_XYZW[  81][1]=-0.637254693925875;
    Leb_Grid_XYZW[  81][2]= 0.637254693925875;
    Leb_Grid_XYZW[  81][3]= 0.001852028828296;

    Leb_Grid_XYZW[  82][0]= 0.433373868777154;
    Leb_Grid_XYZW[  82][1]= 0.637254693925875;
    Leb_Grid_XYZW[  82][2]=-0.637254693925875;
    Leb_Grid_XYZW[  82][3]= 0.001852028828296;

    Leb_Grid_XYZW[  83][0]=-0.433373868777154;
    Leb_Grid_XYZW[  83][1]= 0.637254693925875;
    Leb_Grid_XYZW[  83][2]=-0.637254693925875;
    Leb_Grid_XYZW[  83][3]= 0.001852028828296;

    Leb_Grid_XYZW[  84][0]= 0.433373868777154;
    Leb_Grid_XYZW[  84][1]=-0.637254693925875;
    Leb_Grid_XYZW[  84][2]=-0.637254693925875;
    Leb_Grid_XYZW[  84][3]= 0.001852028828296;

    Leb_Grid_XYZW[  85][0]=-0.433373868777154;
    Leb_Grid_XYZW[  85][1]=-0.637254693925875;
    Leb_Grid_XYZW[  85][2]=-0.637254693925875;
    Leb_Grid_XYZW[  85][3]= 0.001852028828296;

    Leb_Grid_XYZW[  86][0]= 0.504441970780036;
    Leb_Grid_XYZW[  86][1]= 0.504441970780036;
    Leb_Grid_XYZW[  86][2]= 0.700768575373573;
    Leb_Grid_XYZW[  86][3]= 0.001846715956151;

    Leb_Grid_XYZW[  87][0]=-0.504441970780036;
    Leb_Grid_XYZW[  87][1]= 0.504441970780036;
    Leb_Grid_XYZW[  87][2]= 0.700768575373573;
    Leb_Grid_XYZW[  87][3]= 0.001846715956151;

    Leb_Grid_XYZW[  88][0]= 0.504441970780036;
    Leb_Grid_XYZW[  88][1]=-0.504441970780036;
    Leb_Grid_XYZW[  88][2]= 0.700768575373573;
    Leb_Grid_XYZW[  88][3]= 0.001846715956151;

    Leb_Grid_XYZW[  89][0]=-0.504441970780036;
    Leb_Grid_XYZW[  89][1]=-0.504441970780036;
    Leb_Grid_XYZW[  89][2]= 0.700768575373573;
    Leb_Grid_XYZW[  89][3]= 0.001846715956151;

    Leb_Grid_XYZW[  90][0]= 0.504441970780036;
    Leb_Grid_XYZW[  90][1]= 0.504441970780036;
    Leb_Grid_XYZW[  90][2]=-0.700768575373573;
    Leb_Grid_XYZW[  90][3]= 0.001846715956151;

    Leb_Grid_XYZW[  91][0]=-0.504441970780036;
    Leb_Grid_XYZW[  91][1]= 0.504441970780036;
    Leb_Grid_XYZW[  91][2]=-0.700768575373573;
    Leb_Grid_XYZW[  91][3]= 0.001846715956151;

    Leb_Grid_XYZW[  92][0]= 0.504441970780036;
    Leb_Grid_XYZW[  92][1]=-0.504441970780036;
    Leb_Grid_XYZW[  92][2]=-0.700768575373573;
    Leb_Grid_XYZW[  92][3]= 0.001846715956151;

    Leb_Grid_XYZW[  93][0]=-0.504441970780036;
    Leb_Grid_XYZW[  93][1]=-0.504441970780036;
    Leb_Grid_XYZW[  93][2]=-0.700768575373573;
    Leb_Grid_XYZW[  93][3]= 0.001846715956151;

    Leb_Grid_XYZW[  94][0]= 0.504441970780036;
    Leb_Grid_XYZW[  94][1]= 0.700768575373573;
    Leb_Grid_XYZW[  94][2]= 0.504441970780036;
    Leb_Grid_XYZW[  94][3]= 0.001846715956151;

    Leb_Grid_XYZW[  95][0]=-0.504441970780036;
    Leb_Grid_XYZW[  95][1]= 0.700768575373573;
    Leb_Grid_XYZW[  95][2]= 0.504441970780036;
    Leb_Grid_XYZW[  95][3]= 0.001846715956151;

    Leb_Grid_XYZW[  96][0]= 0.504441970780036;
    Leb_Grid_XYZW[  96][1]=-0.700768575373573;
    Leb_Grid_XYZW[  96][2]= 0.504441970780036;
    Leb_Grid_XYZW[  96][3]= 0.001846715956151;

    Leb_Grid_XYZW[  97][0]=-0.504441970780036;
    Leb_Grid_XYZW[  97][1]=-0.700768575373573;
    Leb_Grid_XYZW[  97][2]= 0.504441970780036;
    Leb_Grid_XYZW[  97][3]= 0.001846715956151;

    Leb_Grid_XYZW[  98][0]= 0.504441970780036;
    Leb_Grid_XYZW[  98][1]= 0.700768575373573;
    Leb_Grid_XYZW[  98][2]=-0.504441970780036;
    Leb_Grid_XYZW[  98][3]= 0.001846715956151;

    Leb_Grid_XYZW[  99][0]=-0.504441970780036;
    Leb_Grid_XYZW[  99][1]= 0.700768575373573;
    Leb_Grid_XYZW[  99][2]=-0.504441970780036;
    Leb_Grid_XYZW[  99][3]= 0.001846715956151;

    Leb_Grid_XYZW[ 100][0]= 0.504441970780036;
    Leb_Grid_XYZW[ 100][1]=-0.700768575373573;
    Leb_Grid_XYZW[ 100][2]=-0.504441970780036;
    Leb_Grid_XYZW[ 100][3]= 0.001846715956151;

    Leb_Grid_XYZW[ 101][0]=-0.504441970780036;
    Leb_Grid_XYZW[ 101][1]=-0.700768575373573;
    Leb_Grid_XYZW[ 101][2]=-0.504441970780036;
    Leb_Grid_XYZW[ 101][3]= 0.001846715956151;

    Leb_Grid_XYZW[ 102][0]= 0.700768575373573;
    Leb_Grid_XYZW[ 102][1]= 0.504441970780036;
    Leb_Grid_XYZW[ 102][2]= 0.504441970780036;
    Leb_Grid_XYZW[ 102][3]= 0.001846715956151;

    Leb_Grid_XYZW[ 103][0]=-0.700768575373573;
    Leb_Grid_XYZW[ 103][1]= 0.504441970780036;
    Leb_Grid_XYZW[ 103][2]= 0.504441970780036;
    Leb_Grid_XYZW[ 103][3]= 0.001846715956151;

    Leb_Grid_XYZW[ 104][0]= 0.700768575373573;
    Leb_Grid_XYZW[ 104][1]=-0.504441970780036;
    Leb_Grid_XYZW[ 104][2]= 0.504441970780036;
    Leb_Grid_XYZW[ 104][3]= 0.001846715956151;

    Leb_Grid_XYZW[ 105][0]=-0.700768575373573;
    Leb_Grid_XYZW[ 105][1]=-0.504441970780036;
    Leb_Grid_XYZW[ 105][2]= 0.504441970780036;
    Leb_Grid_XYZW[ 105][3]= 0.001846715956151;

    Leb_Grid_XYZW[ 106][0]= 0.700768575373573;
    Leb_Grid_XYZW[ 106][1]= 0.504441970780036;
    Leb_Grid_XYZW[ 106][2]=-0.504441970780036;
    Leb_Grid_XYZW[ 106][3]= 0.001846715956151;

    Leb_Grid_XYZW[ 107][0]=-0.700768575373573;
    Leb_Grid_XYZW[ 107][1]= 0.504441970780036;
    Leb_Grid_XYZW[ 107][2]=-0.504441970780036;
    Leb_Grid_XYZW[ 107][3]= 0.001846715956151;

    Leb_Grid_XYZW[ 108][0]= 0.700768575373573;
    Leb_Grid_XYZW[ 108][1]=-0.504441970780036;
    Leb_Grid_XYZW[ 108][2]=-0.504441970780036;
    Leb_Grid_XYZW[ 108][3]= 0.001846715956151;

    Leb_Grid_XYZW[ 109][0]=-0.700768575373573;
    Leb_Grid_XYZW[ 109][1]=-0.504441970780036;
    Leb_Grid_XYZW[ 109][2]=-0.504441970780036;
    Leb_Grid_XYZW[ 109][3]= 0.001846715956151;

    Leb_Grid_XYZW[ 110][0]= 0.421576178401097;
    Leb_Grid_XYZW[ 110][1]= 0.421576178401097;
    Leb_Grid_XYZW[ 110][2]= 0.802836877335274;
    Leb_Grid_XYZW[ 110][3]= 0.001818471778163;

    Leb_Grid_XYZW[ 111][0]=-0.421576178401097;
    Leb_Grid_XYZW[ 111][1]= 0.421576178401097;
    Leb_Grid_XYZW[ 111][2]= 0.802836877335274;
    Leb_Grid_XYZW[ 111][3]= 0.001818471778163;

    Leb_Grid_XYZW[ 112][0]= 0.421576178401097;
    Leb_Grid_XYZW[ 112][1]=-0.421576178401097;
    Leb_Grid_XYZW[ 112][2]= 0.802836877335274;
    Leb_Grid_XYZW[ 112][3]= 0.001818471778163;

    Leb_Grid_XYZW[ 113][0]=-0.421576178401097;
    Leb_Grid_XYZW[ 113][1]=-0.421576178401097;
    Leb_Grid_XYZW[ 113][2]= 0.802836877335274;
    Leb_Grid_XYZW[ 113][3]= 0.001818471778163;

    Leb_Grid_XYZW[ 114][0]= 0.421576178401097;
    Leb_Grid_XYZW[ 114][1]= 0.421576178401097;
    Leb_Grid_XYZW[ 114][2]=-0.802836877335274;
    Leb_Grid_XYZW[ 114][3]= 0.001818471778163;

    Leb_Grid_XYZW[ 115][0]=-0.421576178401097;
    Leb_Grid_XYZW[ 115][1]= 0.421576178401097;
    Leb_Grid_XYZW[ 115][2]=-0.802836877335274;
    Leb_Grid_XYZW[ 115][3]= 0.001818471778163;

    Leb_Grid_XYZW[ 116][0]= 0.421576178401097;
    Leb_Grid_XYZW[ 116][1]=-0.421576178401097;
    Leb_Grid_XYZW[ 116][2]=-0.802836877335274;
    Leb_Grid_XYZW[ 116][3]= 0.001818471778163;

    Leb_Grid_XYZW[ 117][0]=-0.421576178401097;
    Leb_Grid_XYZW[ 117][1]=-0.421576178401097;
    Leb_Grid_XYZW[ 117][2]=-0.802836877335274;
    Leb_Grid_XYZW[ 117][3]= 0.001818471778163;

    Leb_Grid_XYZW[ 118][0]= 0.421576178401097;
    Leb_Grid_XYZW[ 118][1]= 0.802836877335274;
    Leb_Grid_XYZW[ 118][2]= 0.421576178401097;
    Leb_Grid_XYZW[ 118][3]= 0.001818471778163;

    Leb_Grid_XYZW[ 119][0]=-0.421576178401097;
    Leb_Grid_XYZW[ 119][1]= 0.802836877335274;
    Leb_Grid_XYZW[ 119][2]= 0.421576178401097;
    Leb_Grid_XYZW[ 119][3]= 0.001818471778163;

    Leb_Grid_XYZW[ 120][0]= 0.421576178401097;
    Leb_Grid_XYZW[ 120][1]=-0.802836877335274;
    Leb_Grid_XYZW[ 120][2]= 0.421576178401097;
    Leb_Grid_XYZW[ 120][3]= 0.001818471778163;

    Leb_Grid_XYZW[ 121][0]=-0.421576178401097;
    Leb_Grid_XYZW[ 121][1]=-0.802836877335274;
    Leb_Grid_XYZW[ 121][2]= 0.421576178401097;
    Leb_Grid_XYZW[ 121][3]= 0.001818471778163;

    Leb_Grid_XYZW[ 122][0]= 0.421576178401097;
    Leb_Grid_XYZW[ 122][1]= 0.802836877335274;
    Leb_Grid_XYZW[ 122][2]=-0.421576178401097;
    Leb_Grid_XYZW[ 122][3]= 0.001818471778163;

    Leb_Grid_XYZW[ 123][0]=-0.421576178401097;
    Leb_Grid_XYZW[ 123][1]= 0.802836877335274;
    Leb_Grid_XYZW[ 123][2]=-0.421576178401097;
    Leb_Grid_XYZW[ 123][3]= 0.001818471778163;

    Leb_Grid_XYZW[ 124][0]= 0.421576178401097;
    Leb_Grid_XYZW[ 124][1]=-0.802836877335274;
    Leb_Grid_XYZW[ 124][2]=-0.421576178401097;
    Leb_Grid_XYZW[ 124][3]= 0.001818471778163;

    Leb_Grid_XYZW[ 125][0]=-0.421576178401097;
    Leb_Grid_XYZW[ 125][1]=-0.802836877335274;
    Leb_Grid_XYZW[ 125][2]=-0.421576178401097;
    Leb_Grid_XYZW[ 125][3]= 0.001818471778163;

    Leb_Grid_XYZW[ 126][0]= 0.802836877335274;
    Leb_Grid_XYZW[ 126][1]= 0.421576178401097;
    Leb_Grid_XYZW[ 126][2]= 0.421576178401097;
    Leb_Grid_XYZW[ 126][3]= 0.001818471778163;

    Leb_Grid_XYZW[ 127][0]=-0.802836877335274;
    Leb_Grid_XYZW[ 127][1]= 0.421576178401097;
    Leb_Grid_XYZW[ 127][2]= 0.421576178401097;
    Leb_Grid_XYZW[ 127][3]= 0.001818471778163;

    Leb_Grid_XYZW[ 128][0]= 0.802836877335274;
    Leb_Grid_XYZW[ 128][1]=-0.421576178401097;
    Leb_Grid_XYZW[ 128][2]= 0.421576178401097;
    Leb_Grid_XYZW[ 128][3]= 0.001818471778163;

    Leb_Grid_XYZW[ 129][0]=-0.802836877335274;
    Leb_Grid_XYZW[ 129][1]=-0.421576178401097;
    Leb_Grid_XYZW[ 129][2]= 0.421576178401097;
    Leb_Grid_XYZW[ 129][3]= 0.001818471778163;

    Leb_Grid_XYZW[ 130][0]= 0.802836877335274;
    Leb_Grid_XYZW[ 130][1]= 0.421576178401097;
    Leb_Grid_XYZW[ 130][2]=-0.421576178401097;
    Leb_Grid_XYZW[ 130][3]= 0.001818471778163;

    Leb_Grid_XYZW[ 131][0]=-0.802836877335274;
    Leb_Grid_XYZW[ 131][1]= 0.421576178401097;
    Leb_Grid_XYZW[ 131][2]=-0.421576178401097;
    Leb_Grid_XYZW[ 131][3]= 0.001818471778163;

    Leb_Grid_XYZW[ 132][0]= 0.802836877335274;
    Leb_Grid_XYZW[ 132][1]=-0.421576178401097;
    Leb_Grid_XYZW[ 132][2]=-0.421576178401097;
    Leb_Grid_XYZW[ 132][3]= 0.001818471778163;

    Leb_Grid_XYZW[ 133][0]=-0.802836877335274;
    Leb_Grid_XYZW[ 133][1]=-0.421576178401097;
    Leb_Grid_XYZW[ 133][2]=-0.421576178401097;
    Leb_Grid_XYZW[ 133][3]= 0.001818471778163;

    Leb_Grid_XYZW[ 134][0]= 0.331792073647212;
    Leb_Grid_XYZW[ 134][1]= 0.331792073647212;
    Leb_Grid_XYZW[ 134][2]= 0.883078727934133;
    Leb_Grid_XYZW[ 134][3]= 0.001749564657281;

    Leb_Grid_XYZW[ 135][0]=-0.331792073647212;
    Leb_Grid_XYZW[ 135][1]= 0.331792073647212;
    Leb_Grid_XYZW[ 135][2]= 0.883078727934133;
    Leb_Grid_XYZW[ 135][3]= 0.001749564657281;

    Leb_Grid_XYZW[ 136][0]= 0.331792073647212;
    Leb_Grid_XYZW[ 136][1]=-0.331792073647212;
    Leb_Grid_XYZW[ 136][2]= 0.883078727934133;
    Leb_Grid_XYZW[ 136][3]= 0.001749564657281;

    Leb_Grid_XYZW[ 137][0]=-0.331792073647212;
    Leb_Grid_XYZW[ 137][1]=-0.331792073647212;
    Leb_Grid_XYZW[ 137][2]= 0.883078727934133;
    Leb_Grid_XYZW[ 137][3]= 0.001749564657281;

    Leb_Grid_XYZW[ 138][0]= 0.331792073647212;
    Leb_Grid_XYZW[ 138][1]= 0.331792073647212;
    Leb_Grid_XYZW[ 138][2]=-0.883078727934133;
    Leb_Grid_XYZW[ 138][3]= 0.001749564657281;

    Leb_Grid_XYZW[ 139][0]=-0.331792073647212;
    Leb_Grid_XYZW[ 139][1]= 0.331792073647212;
    Leb_Grid_XYZW[ 139][2]=-0.883078727934133;
    Leb_Grid_XYZW[ 139][3]= 0.001749564657281;

    Leb_Grid_XYZW[ 140][0]= 0.331792073647212;
    Leb_Grid_XYZW[ 140][1]=-0.331792073647212;
    Leb_Grid_XYZW[ 140][2]=-0.883078727934133;
    Leb_Grid_XYZW[ 140][3]= 0.001749564657281;

    Leb_Grid_XYZW[ 141][0]=-0.331792073647212;
    Leb_Grid_XYZW[ 141][1]=-0.331792073647212;
    Leb_Grid_XYZW[ 141][2]=-0.883078727934133;
    Leb_Grid_XYZW[ 141][3]= 0.001749564657281;

    Leb_Grid_XYZW[ 142][0]= 0.331792073647212;
    Leb_Grid_XYZW[ 142][1]= 0.883078727934133;
    Leb_Grid_XYZW[ 142][2]= 0.331792073647212;
    Leb_Grid_XYZW[ 142][3]= 0.001749564657281;

    Leb_Grid_XYZW[ 143][0]=-0.331792073647212;
    Leb_Grid_XYZW[ 143][1]= 0.883078727934133;
    Leb_Grid_XYZW[ 143][2]= 0.331792073647212;
    Leb_Grid_XYZW[ 143][3]= 0.001749564657281;

    Leb_Grid_XYZW[ 144][0]= 0.331792073647212;
    Leb_Grid_XYZW[ 144][1]=-0.883078727934133;
    Leb_Grid_XYZW[ 144][2]= 0.331792073647212;
    Leb_Grid_XYZW[ 144][3]= 0.001749564657281;

    Leb_Grid_XYZW[ 145][0]=-0.331792073647212;
    Leb_Grid_XYZW[ 145][1]=-0.883078727934133;
    Leb_Grid_XYZW[ 145][2]= 0.331792073647212;
    Leb_Grid_XYZW[ 145][3]= 0.001749564657281;

    Leb_Grid_XYZW[ 146][0]= 0.331792073647212;
    Leb_Grid_XYZW[ 146][1]= 0.883078727934133;
    Leb_Grid_XYZW[ 146][2]=-0.331792073647212;
    Leb_Grid_XYZW[ 146][3]= 0.001749564657281;

    Leb_Grid_XYZW[ 147][0]=-0.331792073647212;
    Leb_Grid_XYZW[ 147][1]= 0.883078727934133;
    Leb_Grid_XYZW[ 147][2]=-0.331792073647212;
    Leb_Grid_XYZW[ 147][3]= 0.001749564657281;

    Leb_Grid_XYZW[ 148][0]= 0.331792073647212;
    Leb_Grid_XYZW[ 148][1]=-0.883078727934133;
    Leb_Grid_XYZW[ 148][2]=-0.331792073647212;
    Leb_Grid_XYZW[ 148][3]= 0.001749564657281;

    Leb_Grid_XYZW[ 149][0]=-0.331792073647212;
    Leb_Grid_XYZW[ 149][1]=-0.883078727934133;
    Leb_Grid_XYZW[ 149][2]=-0.331792073647212;
    Leb_Grid_XYZW[ 149][3]= 0.001749564657281;

    Leb_Grid_XYZW[ 150][0]= 0.883078727934133;
    Leb_Grid_XYZW[ 150][1]= 0.331792073647212;
    Leb_Grid_XYZW[ 150][2]= 0.331792073647212;
    Leb_Grid_XYZW[ 150][3]= 0.001749564657281;

    Leb_Grid_XYZW[ 151][0]=-0.883078727934133;
    Leb_Grid_XYZW[ 151][1]= 0.331792073647212;
    Leb_Grid_XYZW[ 151][2]= 0.331792073647212;
    Leb_Grid_XYZW[ 151][3]= 0.001749564657281;

    Leb_Grid_XYZW[ 152][0]= 0.883078727934133;
    Leb_Grid_XYZW[ 152][1]=-0.331792073647212;
    Leb_Grid_XYZW[ 152][2]= 0.331792073647212;
    Leb_Grid_XYZW[ 152][3]= 0.001749564657281;

    Leb_Grid_XYZW[ 153][0]=-0.883078727934133;
    Leb_Grid_XYZW[ 153][1]=-0.331792073647212;
    Leb_Grid_XYZW[ 153][2]= 0.331792073647212;
    Leb_Grid_XYZW[ 153][3]= 0.001749564657281;

    Leb_Grid_XYZW[ 154][0]= 0.883078727934133;
    Leb_Grid_XYZW[ 154][1]= 0.331792073647212;
    Leb_Grid_XYZW[ 154][2]=-0.331792073647212;
    Leb_Grid_XYZW[ 154][3]= 0.001749564657281;

    Leb_Grid_XYZW[ 155][0]=-0.883078727934133;
    Leb_Grid_XYZW[ 155][1]= 0.331792073647212;
    Leb_Grid_XYZW[ 155][2]=-0.331792073647212;
    Leb_Grid_XYZW[ 155][3]= 0.001749564657281;

    Leb_Grid_XYZW[ 156][0]= 0.883078727934133;
    Leb_Grid_XYZW[ 156][1]=-0.331792073647212;
    Leb_Grid_XYZW[ 156][2]=-0.331792073647212;
    Leb_Grid_XYZW[ 156][3]= 0.001749564657281;

    Leb_Grid_XYZW[ 157][0]=-0.883078727934133;
    Leb_Grid_XYZW[ 157][1]=-0.331792073647212;
    Leb_Grid_XYZW[ 157][2]=-0.331792073647212;
    Leb_Grid_XYZW[ 157][3]= 0.001749564657281;

    Leb_Grid_XYZW[ 158][0]= 0.238473670142189;
    Leb_Grid_XYZW[ 158][1]= 0.238473670142189;
    Leb_Grid_XYZW[ 158][2]= 0.941414158220403;
    Leb_Grid_XYZW[ 158][3]= 0.001617210647254;

    Leb_Grid_XYZW[ 159][0]=-0.238473670142189;
    Leb_Grid_XYZW[ 159][1]= 0.238473670142189;
    Leb_Grid_XYZW[ 159][2]= 0.941414158220403;
    Leb_Grid_XYZW[ 159][3]= 0.001617210647254;

    Leb_Grid_XYZW[ 160][0]= 0.238473670142189;
    Leb_Grid_XYZW[ 160][1]=-0.238473670142189;
    Leb_Grid_XYZW[ 160][2]= 0.941414158220403;
    Leb_Grid_XYZW[ 160][3]= 0.001617210647254;

    Leb_Grid_XYZW[ 161][0]=-0.238473670142189;
    Leb_Grid_XYZW[ 161][1]=-0.238473670142189;
    Leb_Grid_XYZW[ 161][2]= 0.941414158220403;
    Leb_Grid_XYZW[ 161][3]= 0.001617210647254;

    Leb_Grid_XYZW[ 162][0]= 0.238473670142189;
    Leb_Grid_XYZW[ 162][1]= 0.238473670142189;
    Leb_Grid_XYZW[ 162][2]=-0.941414158220403;
    Leb_Grid_XYZW[ 162][3]= 0.001617210647254;

    Leb_Grid_XYZW[ 163][0]=-0.238473670142189;
    Leb_Grid_XYZW[ 163][1]= 0.238473670142189;
    Leb_Grid_XYZW[ 163][2]=-0.941414158220403;
    Leb_Grid_XYZW[ 163][3]= 0.001617210647254;

    Leb_Grid_XYZW[ 164][0]= 0.238473670142189;
    Leb_Grid_XYZW[ 164][1]=-0.238473670142189;
    Leb_Grid_XYZW[ 164][2]=-0.941414158220403;
    Leb_Grid_XYZW[ 164][3]= 0.001617210647254;

    Leb_Grid_XYZW[ 165][0]=-0.238473670142189;
    Leb_Grid_XYZW[ 165][1]=-0.238473670142189;
    Leb_Grid_XYZW[ 165][2]=-0.941414158220403;
    Leb_Grid_XYZW[ 165][3]= 0.001617210647254;

    Leb_Grid_XYZW[ 166][0]= 0.238473670142189;
    Leb_Grid_XYZW[ 166][1]= 0.941414158220403;
    Leb_Grid_XYZW[ 166][2]= 0.238473670142189;
    Leb_Grid_XYZW[ 166][3]= 0.001617210647254;

    Leb_Grid_XYZW[ 167][0]=-0.238473670142189;
    Leb_Grid_XYZW[ 167][1]= 0.941414158220403;
    Leb_Grid_XYZW[ 167][2]= 0.238473670142189;
    Leb_Grid_XYZW[ 167][3]= 0.001617210647254;

    Leb_Grid_XYZW[ 168][0]= 0.238473670142189;
    Leb_Grid_XYZW[ 168][1]=-0.941414158220403;
    Leb_Grid_XYZW[ 168][2]= 0.238473670142189;
    Leb_Grid_XYZW[ 168][3]= 0.001617210647254;

    Leb_Grid_XYZW[ 169][0]=-0.238473670142189;
    Leb_Grid_XYZW[ 169][1]=-0.941414158220403;
    Leb_Grid_XYZW[ 169][2]= 0.238473670142189;
    Leb_Grid_XYZW[ 169][3]= 0.001617210647254;

    Leb_Grid_XYZW[ 170][0]= 0.238473670142189;
    Leb_Grid_XYZW[ 170][1]= 0.941414158220403;
    Leb_Grid_XYZW[ 170][2]=-0.238473670142189;
    Leb_Grid_XYZW[ 170][3]= 0.001617210647254;

    Leb_Grid_XYZW[ 171][0]=-0.238473670142189;
    Leb_Grid_XYZW[ 171][1]= 0.941414158220403;
    Leb_Grid_XYZW[ 171][2]=-0.238473670142189;
    Leb_Grid_XYZW[ 171][3]= 0.001617210647254;

    Leb_Grid_XYZW[ 172][0]= 0.238473670142189;
    Leb_Grid_XYZW[ 172][1]=-0.941414158220403;
    Leb_Grid_XYZW[ 172][2]=-0.238473670142189;
    Leb_Grid_XYZW[ 172][3]= 0.001617210647254;

    Leb_Grid_XYZW[ 173][0]=-0.238473670142189;
    Leb_Grid_XYZW[ 173][1]=-0.941414158220403;
    Leb_Grid_XYZW[ 173][2]=-0.238473670142189;
    Leb_Grid_XYZW[ 173][3]= 0.001617210647254;

    Leb_Grid_XYZW[ 174][0]= 0.941414158220403;
    Leb_Grid_XYZW[ 174][1]= 0.238473670142189;
    Leb_Grid_XYZW[ 174][2]= 0.238473670142189;
    Leb_Grid_XYZW[ 174][3]= 0.001617210647254;

    Leb_Grid_XYZW[ 175][0]=-0.941414158220403;
    Leb_Grid_XYZW[ 175][1]= 0.238473670142189;
    Leb_Grid_XYZW[ 175][2]= 0.238473670142189;
    Leb_Grid_XYZW[ 175][3]= 0.001617210647254;

    Leb_Grid_XYZW[ 176][0]= 0.941414158220403;
    Leb_Grid_XYZW[ 176][1]=-0.238473670142189;
    Leb_Grid_XYZW[ 176][2]= 0.238473670142189;
    Leb_Grid_XYZW[ 176][3]= 0.001617210647254;

    Leb_Grid_XYZW[ 177][0]=-0.941414158220403;
    Leb_Grid_XYZW[ 177][1]=-0.238473670142189;
    Leb_Grid_XYZW[ 177][2]= 0.238473670142189;
    Leb_Grid_XYZW[ 177][3]= 0.001617210647254;

    Leb_Grid_XYZW[ 178][0]= 0.941414158220403;
    Leb_Grid_XYZW[ 178][1]= 0.238473670142189;
    Leb_Grid_XYZW[ 178][2]=-0.238473670142189;
    Leb_Grid_XYZW[ 178][3]= 0.001617210647254;

    Leb_Grid_XYZW[ 179][0]=-0.941414158220403;
    Leb_Grid_XYZW[ 179][1]= 0.238473670142189;
    Leb_Grid_XYZW[ 179][2]=-0.238473670142189;
    Leb_Grid_XYZW[ 179][3]= 0.001617210647254;

    Leb_Grid_XYZW[ 180][0]= 0.941414158220403;
    Leb_Grid_XYZW[ 180][1]=-0.238473670142189;
    Leb_Grid_XYZW[ 180][2]=-0.238473670142189;
    Leb_Grid_XYZW[ 180][3]= 0.001617210647254;

    Leb_Grid_XYZW[ 181][0]=-0.941414158220403;
    Leb_Grid_XYZW[ 181][1]=-0.238473670142189;
    Leb_Grid_XYZW[ 181][2]=-0.238473670142189;
    Leb_Grid_XYZW[ 181][3]= 0.001617210647254;

    Leb_Grid_XYZW[ 182][0]= 0.145903644915776;
    Leb_Grid_XYZW[ 182][1]= 0.145903644915776;
    Leb_Grid_XYZW[ 182][2]= 0.978480583762694;
    Leb_Grid_XYZW[ 182][3]= 0.001384737234852;

    Leb_Grid_XYZW[ 183][0]=-0.145903644915776;
    Leb_Grid_XYZW[ 183][1]= 0.145903644915776;
    Leb_Grid_XYZW[ 183][2]= 0.978480583762694;
    Leb_Grid_XYZW[ 183][3]= 0.001384737234852;

    Leb_Grid_XYZW[ 184][0]= 0.145903644915776;
    Leb_Grid_XYZW[ 184][1]=-0.145903644915776;
    Leb_Grid_XYZW[ 184][2]= 0.978480583762694;
    Leb_Grid_XYZW[ 184][3]= 0.001384737234852;

    Leb_Grid_XYZW[ 185][0]=-0.145903644915776;
    Leb_Grid_XYZW[ 185][1]=-0.145903644915776;
    Leb_Grid_XYZW[ 185][2]= 0.978480583762694;
    Leb_Grid_XYZW[ 185][3]= 0.001384737234852;

    Leb_Grid_XYZW[ 186][0]= 0.145903644915776;
    Leb_Grid_XYZW[ 186][1]= 0.145903644915776;
    Leb_Grid_XYZW[ 186][2]=-0.978480583762694;
    Leb_Grid_XYZW[ 186][3]= 0.001384737234852;

    Leb_Grid_XYZW[ 187][0]=-0.145903644915776;
    Leb_Grid_XYZW[ 187][1]= 0.145903644915776;
    Leb_Grid_XYZW[ 187][2]=-0.978480583762694;
    Leb_Grid_XYZW[ 187][3]= 0.001384737234852;

    Leb_Grid_XYZW[ 188][0]= 0.145903644915776;
    Leb_Grid_XYZW[ 188][1]=-0.145903644915776;
    Leb_Grid_XYZW[ 188][2]=-0.978480583762694;
    Leb_Grid_XYZW[ 188][3]= 0.001384737234852;

    Leb_Grid_XYZW[ 189][0]=-0.145903644915776;
    Leb_Grid_XYZW[ 189][1]=-0.145903644915776;
    Leb_Grid_XYZW[ 189][2]=-0.978480583762694;
    Leb_Grid_XYZW[ 189][3]= 0.001384737234852;

    Leb_Grid_XYZW[ 190][0]= 0.145903644915776;
    Leb_Grid_XYZW[ 190][1]= 0.978480583762694;
    Leb_Grid_XYZW[ 190][2]= 0.145903644915776;
    Leb_Grid_XYZW[ 190][3]= 0.001384737234852;

    Leb_Grid_XYZW[ 191][0]=-0.145903644915776;
    Leb_Grid_XYZW[ 191][1]= 0.978480583762694;
    Leb_Grid_XYZW[ 191][2]= 0.145903644915776;
    Leb_Grid_XYZW[ 191][3]= 0.001384737234852;

    Leb_Grid_XYZW[ 192][0]= 0.145903644915776;
    Leb_Grid_XYZW[ 192][1]=-0.978480583762694;
    Leb_Grid_XYZW[ 192][2]= 0.145903644915776;
    Leb_Grid_XYZW[ 192][3]= 0.001384737234852;

    Leb_Grid_XYZW[ 193][0]=-0.145903644915776;
    Leb_Grid_XYZW[ 193][1]=-0.978480583762694;
    Leb_Grid_XYZW[ 193][2]= 0.145903644915776;
    Leb_Grid_XYZW[ 193][3]= 0.001384737234852;

    Leb_Grid_XYZW[ 194][0]= 0.145903644915776;
    Leb_Grid_XYZW[ 194][1]= 0.978480583762694;
    Leb_Grid_XYZW[ 194][2]=-0.145903644915776;
    Leb_Grid_XYZW[ 194][3]= 0.001384737234852;

    Leb_Grid_XYZW[ 195][0]=-0.145903644915776;
    Leb_Grid_XYZW[ 195][1]= 0.978480583762694;
    Leb_Grid_XYZW[ 195][2]=-0.145903644915776;
    Leb_Grid_XYZW[ 195][3]= 0.001384737234852;

    Leb_Grid_XYZW[ 196][0]= 0.145903644915776;
    Leb_Grid_XYZW[ 196][1]=-0.978480583762694;
    Leb_Grid_XYZW[ 196][2]=-0.145903644915776;
    Leb_Grid_XYZW[ 196][3]= 0.001384737234852;

    Leb_Grid_XYZW[ 197][0]=-0.145903644915776;
    Leb_Grid_XYZW[ 197][1]=-0.978480583762694;
    Leb_Grid_XYZW[ 197][2]=-0.145903644915776;
    Leb_Grid_XYZW[ 197][3]= 0.001384737234852;

    Leb_Grid_XYZW[ 198][0]= 0.978480583762694;
    Leb_Grid_XYZW[ 198][1]= 0.145903644915776;
    Leb_Grid_XYZW[ 198][2]= 0.145903644915776;
    Leb_Grid_XYZW[ 198][3]= 0.001384737234852;

    Leb_Grid_XYZW[ 199][0]=-0.978480583762694;
    Leb_Grid_XYZW[ 199][1]= 0.145903644915776;
    Leb_Grid_XYZW[ 199][2]= 0.145903644915776;
    Leb_Grid_XYZW[ 199][3]= 0.001384737234852;

    Leb_Grid_XYZW[ 200][0]= 0.978480583762694;
    Leb_Grid_XYZW[ 200][1]=-0.145903644915776;
    Leb_Grid_XYZW[ 200][2]= 0.145903644915776;
    Leb_Grid_XYZW[ 200][3]= 0.001384737234852;

    Leb_Grid_XYZW[ 201][0]=-0.978480583762694;
    Leb_Grid_XYZW[ 201][1]=-0.145903644915776;
    Leb_Grid_XYZW[ 201][2]= 0.145903644915776;
    Leb_Grid_XYZW[ 201][3]= 0.001384737234852;

    Leb_Grid_XYZW[ 202][0]= 0.978480583762694;
    Leb_Grid_XYZW[ 202][1]= 0.145903644915776;
    Leb_Grid_XYZW[ 202][2]=-0.145903644915776;
    Leb_Grid_XYZW[ 202][3]= 0.001384737234852;

    Leb_Grid_XYZW[ 203][0]=-0.978480583762694;
    Leb_Grid_XYZW[ 203][1]= 0.145903644915776;
    Leb_Grid_XYZW[ 203][2]=-0.145903644915776;
    Leb_Grid_XYZW[ 203][3]= 0.001384737234852;

    Leb_Grid_XYZW[ 204][0]= 0.978480583762694;
    Leb_Grid_XYZW[ 204][1]=-0.145903644915776;
    Leb_Grid_XYZW[ 204][2]=-0.145903644915776;
    Leb_Grid_XYZW[ 204][3]= 0.001384737234852;

    Leb_Grid_XYZW[ 205][0]=-0.978480583762694;
    Leb_Grid_XYZW[ 205][1]=-0.145903644915776;
    Leb_Grid_XYZW[ 205][2]=-0.145903644915776;
    Leb_Grid_XYZW[ 205][3]= 0.001384737234852;

    Leb_Grid_XYZW[ 206][0]= 0.060950341155072;
    Leb_Grid_XYZW[ 206][1]= 0.060950341155072;
    Leb_Grid_XYZW[ 206][2]= 0.996278129754016;
    Leb_Grid_XYZW[ 206][3]= 0.000976433116505;

    Leb_Grid_XYZW[ 207][0]=-0.060950341155072;
    Leb_Grid_XYZW[ 207][1]= 0.060950341155072;
    Leb_Grid_XYZW[ 207][2]= 0.996278129754016;
    Leb_Grid_XYZW[ 207][3]= 0.000976433116505;

    Leb_Grid_XYZW[ 208][0]= 0.060950341155072;
    Leb_Grid_XYZW[ 208][1]=-0.060950341155072;
    Leb_Grid_XYZW[ 208][2]= 0.996278129754016;
    Leb_Grid_XYZW[ 208][3]= 0.000976433116505;

    Leb_Grid_XYZW[ 209][0]=-0.060950341155072;
    Leb_Grid_XYZW[ 209][1]=-0.060950341155072;
    Leb_Grid_XYZW[ 209][2]= 0.996278129754016;
    Leb_Grid_XYZW[ 209][3]= 0.000976433116505;

    Leb_Grid_XYZW[ 210][0]= 0.060950341155072;
    Leb_Grid_XYZW[ 210][1]= 0.060950341155072;
    Leb_Grid_XYZW[ 210][2]=-0.996278129754016;
    Leb_Grid_XYZW[ 210][3]= 0.000976433116505;

    Leb_Grid_XYZW[ 211][0]=-0.060950341155072;
    Leb_Grid_XYZW[ 211][1]= 0.060950341155072;
    Leb_Grid_XYZW[ 211][2]=-0.996278129754016;
    Leb_Grid_XYZW[ 211][3]= 0.000976433116505;

    Leb_Grid_XYZW[ 212][0]= 0.060950341155072;
    Leb_Grid_XYZW[ 212][1]=-0.060950341155072;
    Leb_Grid_XYZW[ 212][2]=-0.996278129754016;
    Leb_Grid_XYZW[ 212][3]= 0.000976433116505;

    Leb_Grid_XYZW[ 213][0]=-0.060950341155072;
    Leb_Grid_XYZW[ 213][1]=-0.060950341155072;
    Leb_Grid_XYZW[ 213][2]=-0.996278129754016;
    Leb_Grid_XYZW[ 213][3]= 0.000976433116505;

    Leb_Grid_XYZW[ 214][0]= 0.060950341155072;
    Leb_Grid_XYZW[ 214][1]= 0.996278129754016;
    Leb_Grid_XYZW[ 214][2]= 0.060950341155072;
    Leb_Grid_XYZW[ 214][3]= 0.000976433116505;

    Leb_Grid_XYZW[ 215][0]=-0.060950341155072;
    Leb_Grid_XYZW[ 215][1]= 0.996278129754016;
    Leb_Grid_XYZW[ 215][2]= 0.060950341155072;
    Leb_Grid_XYZW[ 215][3]= 0.000976433116505;

    Leb_Grid_XYZW[ 216][0]= 0.060950341155072;
    Leb_Grid_XYZW[ 216][1]=-0.996278129754016;
    Leb_Grid_XYZW[ 216][2]= 0.060950341155072;
    Leb_Grid_XYZW[ 216][3]= 0.000976433116505;

    Leb_Grid_XYZW[ 217][0]=-0.060950341155072;
    Leb_Grid_XYZW[ 217][1]=-0.996278129754016;
    Leb_Grid_XYZW[ 217][2]= 0.060950341155072;
    Leb_Grid_XYZW[ 217][3]= 0.000976433116505;

    Leb_Grid_XYZW[ 218][0]= 0.060950341155072;
    Leb_Grid_XYZW[ 218][1]= 0.996278129754016;
    Leb_Grid_XYZW[ 218][2]=-0.060950341155072;
    Leb_Grid_XYZW[ 218][3]= 0.000976433116505;

    Leb_Grid_XYZW[ 219][0]=-0.060950341155072;
    Leb_Grid_XYZW[ 219][1]= 0.996278129754016;
    Leb_Grid_XYZW[ 219][2]=-0.060950341155072;
    Leb_Grid_XYZW[ 219][3]= 0.000976433116505;

    Leb_Grid_XYZW[ 220][0]= 0.060950341155072;
    Leb_Grid_XYZW[ 220][1]=-0.996278129754016;
    Leb_Grid_XYZW[ 220][2]=-0.060950341155072;
    Leb_Grid_XYZW[ 220][3]= 0.000976433116505;

    Leb_Grid_XYZW[ 221][0]=-0.060950341155072;
    Leb_Grid_XYZW[ 221][1]=-0.996278129754016;
    Leb_Grid_XYZW[ 221][2]=-0.060950341155072;
    Leb_Grid_XYZW[ 221][3]= 0.000976433116505;

    Leb_Grid_XYZW[ 222][0]= 0.996278129754016;
    Leb_Grid_XYZW[ 222][1]= 0.060950341155072;
    Leb_Grid_XYZW[ 222][2]= 0.060950341155072;
    Leb_Grid_XYZW[ 222][3]= 0.000976433116505;

    Leb_Grid_XYZW[ 223][0]=-0.996278129754016;
    Leb_Grid_XYZW[ 223][1]= 0.060950341155072;
    Leb_Grid_XYZW[ 223][2]= 0.060950341155072;
    Leb_Grid_XYZW[ 223][3]= 0.000976433116505;

    Leb_Grid_XYZW[ 224][0]= 0.996278129754016;
    Leb_Grid_XYZW[ 224][1]=-0.060950341155072;
    Leb_Grid_XYZW[ 224][2]= 0.060950341155072;
    Leb_Grid_XYZW[ 224][3]= 0.000976433116505;

    Leb_Grid_XYZW[ 225][0]=-0.996278129754016;
    Leb_Grid_XYZW[ 225][1]=-0.060950341155072;
    Leb_Grid_XYZW[ 225][2]= 0.060950341155072;
    Leb_Grid_XYZW[ 225][3]= 0.000976433116505;

    Leb_Grid_XYZW[ 226][0]= 0.996278129754016;
    Leb_Grid_XYZW[ 226][1]= 0.060950341155072;
    Leb_Grid_XYZW[ 226][2]=-0.060950341155072;
    Leb_Grid_XYZW[ 226][3]= 0.000976433116505;

    Leb_Grid_XYZW[ 227][0]=-0.996278129754016;
    Leb_Grid_XYZW[ 227][1]= 0.060950341155072;
    Leb_Grid_XYZW[ 227][2]=-0.060950341155072;
    Leb_Grid_XYZW[ 227][3]= 0.000976433116505;

    Leb_Grid_XYZW[ 228][0]= 0.996278129754016;
    Leb_Grid_XYZW[ 228][1]=-0.060950341155072;
    Leb_Grid_XYZW[ 228][2]=-0.060950341155072;
    Leb_Grid_XYZW[ 228][3]= 0.000976433116505;

    Leb_Grid_XYZW[ 229][0]=-0.996278129754016;
    Leb_Grid_XYZW[ 229][1]=-0.060950341155072;
    Leb_Grid_XYZW[ 229][2]=-0.060950341155072;
    Leb_Grid_XYZW[ 229][3]= 0.000976433116505;

    Leb_Grid_XYZW[ 230][0]= 0.611684344200988;
    Leb_Grid_XYZW[ 230][1]= 0.791101929626902;
    Leb_Grid_XYZW[ 230][2]= 0.000000000000000;
    Leb_Grid_XYZW[ 230][3]= 0.001857161196774;

    Leb_Grid_XYZW[ 231][0]=-0.611684344200988;
    Leb_Grid_XYZW[ 231][1]= 0.791101929626902;
    Leb_Grid_XYZW[ 231][2]= 0.000000000000000;
    Leb_Grid_XYZW[ 231][3]= 0.001857161196774;

    Leb_Grid_XYZW[ 232][0]= 0.611684344200988;
    Leb_Grid_XYZW[ 232][1]=-0.791101929626902;
    Leb_Grid_XYZW[ 232][2]= 0.000000000000000;
    Leb_Grid_XYZW[ 232][3]= 0.001857161196774;

    Leb_Grid_XYZW[ 233][0]=-0.611684344200988;
    Leb_Grid_XYZW[ 233][1]=-0.791101929626902;
    Leb_Grid_XYZW[ 233][2]= 0.000000000000000;
    Leb_Grid_XYZW[ 233][3]= 0.001857161196774;

    Leb_Grid_XYZW[ 234][0]= 0.791101929626902;
    Leb_Grid_XYZW[ 234][1]= 0.611684344200988;
    Leb_Grid_XYZW[ 234][2]= 0.000000000000000;
    Leb_Grid_XYZW[ 234][3]= 0.001857161196774;

    Leb_Grid_XYZW[ 235][0]=-0.791101929626902;
    Leb_Grid_XYZW[ 235][1]= 0.611684344200988;
    Leb_Grid_XYZW[ 235][2]= 0.000000000000000;
    Leb_Grid_XYZW[ 235][3]= 0.001857161196774;

    Leb_Grid_XYZW[ 236][0]= 0.791101929626902;
    Leb_Grid_XYZW[ 236][1]=-0.611684344200988;
    Leb_Grid_XYZW[ 236][2]= 0.000000000000000;
    Leb_Grid_XYZW[ 236][3]= 0.001857161196774;

    Leb_Grid_XYZW[ 237][0]=-0.791101929626902;
    Leb_Grid_XYZW[ 237][1]=-0.611684344200988;
    Leb_Grid_XYZW[ 237][2]= 0.000000000000000;
    Leb_Grid_XYZW[ 237][3]= 0.001857161196774;

    Leb_Grid_XYZW[ 238][0]= 0.611684344200988;
    Leb_Grid_XYZW[ 238][1]= 0.000000000000000;
    Leb_Grid_XYZW[ 238][2]= 0.791101929626902;
    Leb_Grid_XYZW[ 238][3]= 0.001857161196774;

    Leb_Grid_XYZW[ 239][0]=-0.611684344200988;
    Leb_Grid_XYZW[ 239][1]= 0.000000000000000;
    Leb_Grid_XYZW[ 239][2]= 0.791101929626902;
    Leb_Grid_XYZW[ 239][3]= 0.001857161196774;

    Leb_Grid_XYZW[ 240][0]= 0.611684344200988;
    Leb_Grid_XYZW[ 240][1]= 0.000000000000000;
    Leb_Grid_XYZW[ 240][2]=-0.791101929626902;
    Leb_Grid_XYZW[ 240][3]= 0.001857161196774;

    Leb_Grid_XYZW[ 241][0]=-0.611684344200988;
    Leb_Grid_XYZW[ 241][1]= 0.000000000000000;
    Leb_Grid_XYZW[ 241][2]=-0.791101929626902;
    Leb_Grid_XYZW[ 241][3]= 0.001857161196774;

    Leb_Grid_XYZW[ 242][0]= 0.791101929626902;
    Leb_Grid_XYZW[ 242][1]= 0.000000000000000;
    Leb_Grid_XYZW[ 242][2]= 0.611684344200988;
    Leb_Grid_XYZW[ 242][3]= 0.001857161196774;

    Leb_Grid_XYZW[ 243][0]=-0.791101929626902;
    Leb_Grid_XYZW[ 243][1]= 0.000000000000000;
    Leb_Grid_XYZW[ 243][2]= 0.611684344200988;
    Leb_Grid_XYZW[ 243][3]= 0.001857161196774;

    Leb_Grid_XYZW[ 244][0]= 0.791101929626902;
    Leb_Grid_XYZW[ 244][1]= 0.000000000000000;
    Leb_Grid_XYZW[ 244][2]=-0.611684344200988;
    Leb_Grid_XYZW[ 244][3]= 0.001857161196774;

    Leb_Grid_XYZW[ 245][0]=-0.791101929626902;
    Leb_Grid_XYZW[ 245][1]= 0.000000000000000;
    Leb_Grid_XYZW[ 245][2]=-0.611684344200988;
    Leb_Grid_XYZW[ 245][3]= 0.001857161196774;

    Leb_Grid_XYZW[ 246][0]= 0.000000000000000;
    Leb_Grid_XYZW[ 246][1]= 0.611684344200988;
    Leb_Grid_XYZW[ 246][2]= 0.791101929626902;
    Leb_Grid_XYZW[ 246][3]= 0.001857161196774;

    Leb_Grid_XYZW[ 247][0]= 0.000000000000000;
    Leb_Grid_XYZW[ 247][1]=-0.611684344200988;
    Leb_Grid_XYZW[ 247][2]= 0.791101929626902;
    Leb_Grid_XYZW[ 247][3]= 0.001857161196774;

    Leb_Grid_XYZW[ 248][0]= 0.000000000000000;
    Leb_Grid_XYZW[ 248][1]= 0.611684344200988;
    Leb_Grid_XYZW[ 248][2]=-0.791101929626902;
    Leb_Grid_XYZW[ 248][3]= 0.001857161196774;

    Leb_Grid_XYZW[ 249][0]= 0.000000000000000;
    Leb_Grid_XYZW[ 249][1]=-0.611684344200988;
    Leb_Grid_XYZW[ 249][2]=-0.791101929626902;
    Leb_Grid_XYZW[ 249][3]= 0.001857161196774;

    Leb_Grid_XYZW[ 250][0]= 0.000000000000000;
    Leb_Grid_XYZW[ 250][1]= 0.791101929626902;
    Leb_Grid_XYZW[ 250][2]= 0.611684344200988;
    Leb_Grid_XYZW[ 250][3]= 0.001857161196774;

    Leb_Grid_XYZW[ 251][0]= 0.000000000000000;
    Leb_Grid_XYZW[ 251][1]=-0.791101929626902;
    Leb_Grid_XYZW[ 251][2]= 0.611684344200988;
    Leb_Grid_XYZW[ 251][3]= 0.001857161196774;

    Leb_Grid_XYZW[ 252][0]= 0.000000000000000;
    Leb_Grid_XYZW[ 252][1]= 0.791101929626902;
    Leb_Grid_XYZW[ 252][2]=-0.611684344200988;
    Leb_Grid_XYZW[ 252][3]= 0.001857161196774;

    Leb_Grid_XYZW[ 253][0]= 0.000000000000000;
    Leb_Grid_XYZW[ 253][1]=-0.791101929626902;
    Leb_Grid_XYZW[ 253][2]=-0.611684344200988;
    Leb_Grid_XYZW[ 253][3]= 0.001857161196774;

    Leb_Grid_XYZW[ 254][0]= 0.396475534819986;
    Leb_Grid_XYZW[ 254][1]= 0.918045287711454;
    Leb_Grid_XYZW[ 254][2]= 0.000000000000000;
    Leb_Grid_XYZW[ 254][3]= 0.001705153996396;

    Leb_Grid_XYZW[ 255][0]=-0.396475534819986;
    Leb_Grid_XYZW[ 255][1]= 0.918045287711454;
    Leb_Grid_XYZW[ 255][2]= 0.000000000000000;
    Leb_Grid_XYZW[ 255][3]= 0.001705153996396;

    Leb_Grid_XYZW[ 256][0]= 0.396475534819986;
    Leb_Grid_XYZW[ 256][1]=-0.918045287711454;
    Leb_Grid_XYZW[ 256][2]= 0.000000000000000;
    Leb_Grid_XYZW[ 256][3]= 0.001705153996396;

    Leb_Grid_XYZW[ 257][0]=-0.396475534819986;
    Leb_Grid_XYZW[ 257][1]=-0.918045287711454;
    Leb_Grid_XYZW[ 257][2]= 0.000000000000000;
    Leb_Grid_XYZW[ 257][3]= 0.001705153996396;

    Leb_Grid_XYZW[ 258][0]= 0.918045287711454;
    Leb_Grid_XYZW[ 258][1]= 0.396475534819986;
    Leb_Grid_XYZW[ 258][2]= 0.000000000000000;
    Leb_Grid_XYZW[ 258][3]= 0.001705153996396;

    Leb_Grid_XYZW[ 259][0]=-0.918045287711454;
    Leb_Grid_XYZW[ 259][1]= 0.396475534819986;
    Leb_Grid_XYZW[ 259][2]= 0.000000000000000;
    Leb_Grid_XYZW[ 259][3]= 0.001705153996396;

    Leb_Grid_XYZW[ 260][0]= 0.918045287711454;
    Leb_Grid_XYZW[ 260][1]=-0.396475534819986;
    Leb_Grid_XYZW[ 260][2]= 0.000000000000000;
    Leb_Grid_XYZW[ 260][3]= 0.001705153996396;

    Leb_Grid_XYZW[ 261][0]=-0.918045287711454;
    Leb_Grid_XYZW[ 261][1]=-0.396475534819986;
    Leb_Grid_XYZW[ 261][2]= 0.000000000000000;
    Leb_Grid_XYZW[ 261][3]= 0.001705153996396;

    Leb_Grid_XYZW[ 262][0]= 0.396475534819986;
    Leb_Grid_XYZW[ 262][1]= 0.000000000000000;
    Leb_Grid_XYZW[ 262][2]= 0.918045287711454;
    Leb_Grid_XYZW[ 262][3]= 0.001705153996396;

    Leb_Grid_XYZW[ 263][0]=-0.396475534819986;
    Leb_Grid_XYZW[ 263][1]= 0.000000000000000;
    Leb_Grid_XYZW[ 263][2]= 0.918045287711454;
    Leb_Grid_XYZW[ 263][3]= 0.001705153996396;

    Leb_Grid_XYZW[ 264][0]= 0.396475534819986;
    Leb_Grid_XYZW[ 264][1]= 0.000000000000000;
    Leb_Grid_XYZW[ 264][2]=-0.918045287711454;
    Leb_Grid_XYZW[ 264][3]= 0.001705153996396;

    Leb_Grid_XYZW[ 265][0]=-0.396475534819986;
    Leb_Grid_XYZW[ 265][1]= 0.000000000000000;
    Leb_Grid_XYZW[ 265][2]=-0.918045287711454;
    Leb_Grid_XYZW[ 265][3]= 0.001705153996396;

    Leb_Grid_XYZW[ 266][0]= 0.918045287711454;
    Leb_Grid_XYZW[ 266][1]= 0.000000000000000;
    Leb_Grid_XYZW[ 266][2]= 0.396475534819986;
    Leb_Grid_XYZW[ 266][3]= 0.001705153996396;

    Leb_Grid_XYZW[ 267][0]=-0.918045287711454;
    Leb_Grid_XYZW[ 267][1]= 0.000000000000000;
    Leb_Grid_XYZW[ 267][2]= 0.396475534819986;
    Leb_Grid_XYZW[ 267][3]= 0.001705153996396;

    Leb_Grid_XYZW[ 268][0]= 0.918045287711454;
    Leb_Grid_XYZW[ 268][1]= 0.000000000000000;
    Leb_Grid_XYZW[ 268][2]=-0.396475534819986;
    Leb_Grid_XYZW[ 268][3]= 0.001705153996396;

    Leb_Grid_XYZW[ 269][0]=-0.918045287711454;
    Leb_Grid_XYZW[ 269][1]= 0.000000000000000;
    Leb_Grid_XYZW[ 269][2]=-0.396475534819986;
    Leb_Grid_XYZW[ 269][3]= 0.001705153996396;

    Leb_Grid_XYZW[ 270][0]= 0.000000000000000;
    Leb_Grid_XYZW[ 270][1]= 0.396475534819986;
    Leb_Grid_XYZW[ 270][2]= 0.918045287711454;
    Leb_Grid_XYZW[ 270][3]= 0.001705153996396;

    Leb_Grid_XYZW[ 271][0]= 0.000000000000000;
    Leb_Grid_XYZW[ 271][1]=-0.396475534819986;
    Leb_Grid_XYZW[ 271][2]= 0.918045287711454;
    Leb_Grid_XYZW[ 271][3]= 0.001705153996396;

    Leb_Grid_XYZW[ 272][0]= 0.000000000000000;
    Leb_Grid_XYZW[ 272][1]= 0.396475534819986;
    Leb_Grid_XYZW[ 272][2]=-0.918045287711454;
    Leb_Grid_XYZW[ 272][3]= 0.001705153996396;

    Leb_Grid_XYZW[ 273][0]= 0.000000000000000;
    Leb_Grid_XYZW[ 273][1]=-0.396475534819986;
    Leb_Grid_XYZW[ 273][2]=-0.918045287711454;
    Leb_Grid_XYZW[ 273][3]= 0.001705153996396;

    Leb_Grid_XYZW[ 274][0]= 0.000000000000000;
    Leb_Grid_XYZW[ 274][1]= 0.918045287711454;
    Leb_Grid_XYZW[ 274][2]= 0.396475534819986;
    Leb_Grid_XYZW[ 274][3]= 0.001705153996396;

    Leb_Grid_XYZW[ 275][0]= 0.000000000000000;
    Leb_Grid_XYZW[ 275][1]=-0.918045287711454;
    Leb_Grid_XYZW[ 275][2]= 0.396475534819986;
    Leb_Grid_XYZW[ 275][3]= 0.001705153996396;

    Leb_Grid_XYZW[ 276][0]= 0.000000000000000;
    Leb_Grid_XYZW[ 276][1]= 0.918045287711454;
    Leb_Grid_XYZW[ 276][2]=-0.396475534819986;
    Leb_Grid_XYZW[ 276][3]= 0.001705153996396;

    Leb_Grid_XYZW[ 277][0]= 0.000000000000000;
    Leb_Grid_XYZW[ 277][1]=-0.918045287711454;
    Leb_Grid_XYZW[ 277][2]=-0.396475534819986;
    Leb_Grid_XYZW[ 277][3]= 0.001705153996396;

    Leb_Grid_XYZW[ 278][0]= 0.172478200990772;
    Leb_Grid_XYZW[ 278][1]= 0.985013335028002;
    Leb_Grid_XYZW[ 278][2]= 0.000000000000000;
    Leb_Grid_XYZW[ 278][3]= 0.001300321685886;

    Leb_Grid_XYZW[ 279][0]=-0.172478200990772;
    Leb_Grid_XYZW[ 279][1]= 0.985013335028002;
    Leb_Grid_XYZW[ 279][2]= 0.000000000000000;
    Leb_Grid_XYZW[ 279][3]= 0.001300321685886;

    Leb_Grid_XYZW[ 280][0]= 0.172478200990772;
    Leb_Grid_XYZW[ 280][1]=-0.985013335028002;
    Leb_Grid_XYZW[ 280][2]= 0.000000000000000;
    Leb_Grid_XYZW[ 280][3]= 0.001300321685886;

    Leb_Grid_XYZW[ 281][0]=-0.172478200990772;
    Leb_Grid_XYZW[ 281][1]=-0.985013335028002;
    Leb_Grid_XYZW[ 281][2]= 0.000000000000000;
    Leb_Grid_XYZW[ 281][3]= 0.001300321685886;

    Leb_Grid_XYZW[ 282][0]= 0.985013335028002;
    Leb_Grid_XYZW[ 282][1]= 0.172478200990772;
    Leb_Grid_XYZW[ 282][2]= 0.000000000000000;
    Leb_Grid_XYZW[ 282][3]= 0.001300321685886;

    Leb_Grid_XYZW[ 283][0]=-0.985013335028002;
    Leb_Grid_XYZW[ 283][1]= 0.172478200990772;
    Leb_Grid_XYZW[ 283][2]= 0.000000000000000;
    Leb_Grid_XYZW[ 283][3]= 0.001300321685886;

    Leb_Grid_XYZW[ 284][0]= 0.985013335028002;
    Leb_Grid_XYZW[ 284][1]=-0.172478200990772;
    Leb_Grid_XYZW[ 284][2]= 0.000000000000000;
    Leb_Grid_XYZW[ 284][3]= 0.001300321685886;

    Leb_Grid_XYZW[ 285][0]=-0.985013335028002;
    Leb_Grid_XYZW[ 285][1]=-0.172478200990772;
    Leb_Grid_XYZW[ 285][2]= 0.000000000000000;
    Leb_Grid_XYZW[ 285][3]= 0.001300321685886;

    Leb_Grid_XYZW[ 286][0]= 0.172478200990772;
    Leb_Grid_XYZW[ 286][1]= 0.000000000000000;
    Leb_Grid_XYZW[ 286][2]= 0.985013335028002;
    Leb_Grid_XYZW[ 286][3]= 0.001300321685886;

    Leb_Grid_XYZW[ 287][0]=-0.172478200990772;
    Leb_Grid_XYZW[ 287][1]= 0.000000000000000;
    Leb_Grid_XYZW[ 287][2]= 0.985013335028002;
    Leb_Grid_XYZW[ 287][3]= 0.001300321685886;

    Leb_Grid_XYZW[ 288][0]= 0.172478200990772;
    Leb_Grid_XYZW[ 288][1]= 0.000000000000000;
    Leb_Grid_XYZW[ 288][2]=-0.985013335028002;
    Leb_Grid_XYZW[ 288][3]= 0.001300321685886;

    Leb_Grid_XYZW[ 289][0]=-0.172478200990772;
    Leb_Grid_XYZW[ 289][1]= 0.000000000000000;
    Leb_Grid_XYZW[ 289][2]=-0.985013335028002;
    Leb_Grid_XYZW[ 289][3]= 0.001300321685886;

    Leb_Grid_XYZW[ 290][0]= 0.985013335028002;
    Leb_Grid_XYZW[ 290][1]= 0.000000000000000;
    Leb_Grid_XYZW[ 290][2]= 0.172478200990772;
    Leb_Grid_XYZW[ 290][3]= 0.001300321685886;

    Leb_Grid_XYZW[ 291][0]=-0.985013335028002;
    Leb_Grid_XYZW[ 291][1]= 0.000000000000000;
    Leb_Grid_XYZW[ 291][2]= 0.172478200990772;
    Leb_Grid_XYZW[ 291][3]= 0.001300321685886;

    Leb_Grid_XYZW[ 292][0]= 0.985013335028002;
    Leb_Grid_XYZW[ 292][1]= 0.000000000000000;
    Leb_Grid_XYZW[ 292][2]=-0.172478200990772;
    Leb_Grid_XYZW[ 292][3]= 0.001300321685886;

    Leb_Grid_XYZW[ 293][0]=-0.985013335028002;
    Leb_Grid_XYZW[ 293][1]= 0.000000000000000;
    Leb_Grid_XYZW[ 293][2]=-0.172478200990772;
    Leb_Grid_XYZW[ 293][3]= 0.001300321685886;

    Leb_Grid_XYZW[ 294][0]= 0.000000000000000;
    Leb_Grid_XYZW[ 294][1]= 0.172478200990772;
    Leb_Grid_XYZW[ 294][2]= 0.985013335028002;
    Leb_Grid_XYZW[ 294][3]= 0.001300321685886;

    Leb_Grid_XYZW[ 295][0]= 0.000000000000000;
    Leb_Grid_XYZW[ 295][1]=-0.172478200990772;
    Leb_Grid_XYZW[ 295][2]= 0.985013335028002;
    Leb_Grid_XYZW[ 295][3]= 0.001300321685886;

    Leb_Grid_XYZW[ 296][0]= 0.000000000000000;
    Leb_Grid_XYZW[ 296][1]= 0.172478200990772;
    Leb_Grid_XYZW[ 296][2]=-0.985013335028002;
    Leb_Grid_XYZW[ 296][3]= 0.001300321685886;

    Leb_Grid_XYZW[ 297][0]= 0.000000000000000;
    Leb_Grid_XYZW[ 297][1]=-0.172478200990772;
    Leb_Grid_XYZW[ 297][2]=-0.985013335028002;
    Leb_Grid_XYZW[ 297][3]= 0.001300321685886;

    Leb_Grid_XYZW[ 298][0]= 0.000000000000000;
    Leb_Grid_XYZW[ 298][1]= 0.985013335028002;
    Leb_Grid_XYZW[ 298][2]= 0.172478200990772;
    Leb_Grid_XYZW[ 298][3]= 0.001300321685886;

    Leb_Grid_XYZW[ 299][0]= 0.000000000000000;
    Leb_Grid_XYZW[ 299][1]=-0.985013335028002;
    Leb_Grid_XYZW[ 299][2]= 0.172478200990772;
    Leb_Grid_XYZW[ 299][3]= 0.001300321685886;

    Leb_Grid_XYZW[ 300][0]= 0.000000000000000;
    Leb_Grid_XYZW[ 300][1]= 0.985013335028002;
    Leb_Grid_XYZW[ 300][2]=-0.172478200990772;
    Leb_Grid_XYZW[ 300][3]= 0.001300321685886;

    Leb_Grid_XYZW[ 301][0]= 0.000000000000000;
    Leb_Grid_XYZW[ 301][1]=-0.985013335028002;
    Leb_Grid_XYZW[ 301][2]=-0.172478200990772;
    Leb_Grid_XYZW[ 301][3]= 0.001300321685886;

    Leb_Grid_XYZW[ 302][0]= 0.561026380862206;
    Leb_Grid_XYZW[ 302][1]= 0.351828092773352;
    Leb_Grid_XYZW[ 302][2]= 0.749310611904116;
    Leb_Grid_XYZW[ 302][3]= 0.001842866472905;

    Leb_Grid_XYZW[ 303][0]=-0.561026380862206;
    Leb_Grid_XYZW[ 303][1]= 0.351828092773352;
    Leb_Grid_XYZW[ 303][2]= 0.749310611904116;
    Leb_Grid_XYZW[ 303][3]= 0.001842866472905;

    Leb_Grid_XYZW[ 304][0]= 0.561026380862206;
    Leb_Grid_XYZW[ 304][1]=-0.351828092773352;
    Leb_Grid_XYZW[ 304][2]= 0.749310611904116;
    Leb_Grid_XYZW[ 304][3]= 0.001842866472905;

    Leb_Grid_XYZW[ 305][0]=-0.561026380862206;
    Leb_Grid_XYZW[ 305][1]=-0.351828092773352;
    Leb_Grid_XYZW[ 305][2]= 0.749310611904116;
    Leb_Grid_XYZW[ 305][3]= 0.001842866472905;

    Leb_Grid_XYZW[ 306][0]= 0.561026380862206;
    Leb_Grid_XYZW[ 306][1]= 0.351828092773352;
    Leb_Grid_XYZW[ 306][2]=-0.749310611904116;
    Leb_Grid_XYZW[ 306][3]= 0.001842866472905;

    Leb_Grid_XYZW[ 307][0]=-0.561026380862206;
    Leb_Grid_XYZW[ 307][1]= 0.351828092773352;
    Leb_Grid_XYZW[ 307][2]=-0.749310611904116;
    Leb_Grid_XYZW[ 307][3]= 0.001842866472905;

    Leb_Grid_XYZW[ 308][0]= 0.561026380862206;
    Leb_Grid_XYZW[ 308][1]=-0.351828092773352;
    Leb_Grid_XYZW[ 308][2]=-0.749310611904116;
    Leb_Grid_XYZW[ 308][3]= 0.001842866472905;

    Leb_Grid_XYZW[ 309][0]=-0.561026380862206;
    Leb_Grid_XYZW[ 309][1]=-0.351828092773352;
    Leb_Grid_XYZW[ 309][2]=-0.749310611904116;
    Leb_Grid_XYZW[ 309][3]= 0.001842866472905;

    Leb_Grid_XYZW[ 310][0]= 0.561026380862206;
    Leb_Grid_XYZW[ 310][1]= 0.749310611904116;
    Leb_Grid_XYZW[ 310][2]= 0.351828092773352;
    Leb_Grid_XYZW[ 310][3]= 0.001842866472905;

    Leb_Grid_XYZW[ 311][0]=-0.561026380862206;
    Leb_Grid_XYZW[ 311][1]= 0.749310611904116;
    Leb_Grid_XYZW[ 311][2]= 0.351828092773352;
    Leb_Grid_XYZW[ 311][3]= 0.001842866472905;

    Leb_Grid_XYZW[ 312][0]= 0.561026380862206;
    Leb_Grid_XYZW[ 312][1]=-0.749310611904116;
    Leb_Grid_XYZW[ 312][2]= 0.351828092773352;
    Leb_Grid_XYZW[ 312][3]= 0.001842866472905;

    Leb_Grid_XYZW[ 313][0]=-0.561026380862206;
    Leb_Grid_XYZW[ 313][1]=-0.749310611904116;
    Leb_Grid_XYZW[ 313][2]= 0.351828092773352;
    Leb_Grid_XYZW[ 313][3]= 0.001842866472905;

    Leb_Grid_XYZW[ 314][0]= 0.561026380862206;
    Leb_Grid_XYZW[ 314][1]= 0.749310611904116;
    Leb_Grid_XYZW[ 314][2]=-0.351828092773352;
    Leb_Grid_XYZW[ 314][3]= 0.001842866472905;

    Leb_Grid_XYZW[ 315][0]=-0.561026380862206;
    Leb_Grid_XYZW[ 315][1]= 0.749310611904116;
    Leb_Grid_XYZW[ 315][2]=-0.351828092773352;
    Leb_Grid_XYZW[ 315][3]= 0.001842866472905;

    Leb_Grid_XYZW[ 316][0]= 0.561026380862206;
    Leb_Grid_XYZW[ 316][1]=-0.749310611904116;
    Leb_Grid_XYZW[ 316][2]=-0.351828092773352;
    Leb_Grid_XYZW[ 316][3]= 0.001842866472905;

    Leb_Grid_XYZW[ 317][0]=-0.561026380862206;
    Leb_Grid_XYZW[ 317][1]=-0.749310611904116;
    Leb_Grid_XYZW[ 317][2]=-0.351828092773352;
    Leb_Grid_XYZW[ 317][3]= 0.001842866472905;

    Leb_Grid_XYZW[ 318][0]= 0.351828092773352;
    Leb_Grid_XYZW[ 318][1]= 0.561026380862206;
    Leb_Grid_XYZW[ 318][2]= 0.749310611904116;
    Leb_Grid_XYZW[ 318][3]= 0.001842866472905;

    Leb_Grid_XYZW[ 319][0]=-0.351828092773352;
    Leb_Grid_XYZW[ 319][1]= 0.561026380862206;
    Leb_Grid_XYZW[ 319][2]= 0.749310611904116;
    Leb_Grid_XYZW[ 319][3]= 0.001842866472905;

    Leb_Grid_XYZW[ 320][0]= 0.351828092773352;
    Leb_Grid_XYZW[ 320][1]=-0.561026380862206;
    Leb_Grid_XYZW[ 320][2]= 0.749310611904116;
    Leb_Grid_XYZW[ 320][3]= 0.001842866472905;

    Leb_Grid_XYZW[ 321][0]=-0.351828092773352;
    Leb_Grid_XYZW[ 321][1]=-0.561026380862206;
    Leb_Grid_XYZW[ 321][2]= 0.749310611904116;
    Leb_Grid_XYZW[ 321][3]= 0.001842866472905;

    Leb_Grid_XYZW[ 322][0]= 0.351828092773352;
    Leb_Grid_XYZW[ 322][1]= 0.561026380862206;
    Leb_Grid_XYZW[ 322][2]=-0.749310611904116;
    Leb_Grid_XYZW[ 322][3]= 0.001842866472905;

    Leb_Grid_XYZW[ 323][0]=-0.351828092773352;
    Leb_Grid_XYZW[ 323][1]= 0.561026380862206;
    Leb_Grid_XYZW[ 323][2]=-0.749310611904116;
    Leb_Grid_XYZW[ 323][3]= 0.001842866472905;

    Leb_Grid_XYZW[ 324][0]= 0.351828092773352;
    Leb_Grid_XYZW[ 324][1]=-0.561026380862206;
    Leb_Grid_XYZW[ 324][2]=-0.749310611904116;
    Leb_Grid_XYZW[ 324][3]= 0.001842866472905;

    Leb_Grid_XYZW[ 325][0]=-0.351828092773352;
    Leb_Grid_XYZW[ 325][1]=-0.561026380862206;
    Leb_Grid_XYZW[ 325][2]=-0.749310611904116;
    Leb_Grid_XYZW[ 325][3]= 0.001842866472905;

    Leb_Grid_XYZW[ 326][0]= 0.351828092773352;
    Leb_Grid_XYZW[ 326][1]= 0.749310611904116;
    Leb_Grid_XYZW[ 326][2]= 0.561026380862206;
    Leb_Grid_XYZW[ 326][3]= 0.001842866472905;

    Leb_Grid_XYZW[ 327][0]=-0.351828092773352;
    Leb_Grid_XYZW[ 327][1]= 0.749310611904116;
    Leb_Grid_XYZW[ 327][2]= 0.561026380862206;
    Leb_Grid_XYZW[ 327][3]= 0.001842866472905;

    Leb_Grid_XYZW[ 328][0]= 0.351828092773352;
    Leb_Grid_XYZW[ 328][1]=-0.749310611904116;
    Leb_Grid_XYZW[ 328][2]= 0.561026380862206;
    Leb_Grid_XYZW[ 328][3]= 0.001842866472905;

    Leb_Grid_XYZW[ 329][0]=-0.351828092773352;
    Leb_Grid_XYZW[ 329][1]=-0.749310611904116;
    Leb_Grid_XYZW[ 329][2]= 0.561026380862206;
    Leb_Grid_XYZW[ 329][3]= 0.001842866472905;

    Leb_Grid_XYZW[ 330][0]= 0.351828092773352;
    Leb_Grid_XYZW[ 330][1]= 0.749310611904116;
    Leb_Grid_XYZW[ 330][2]=-0.561026380862206;
    Leb_Grid_XYZW[ 330][3]= 0.001842866472905;

    Leb_Grid_XYZW[ 331][0]=-0.351828092773352;
    Leb_Grid_XYZW[ 331][1]= 0.749310611904116;
    Leb_Grid_XYZW[ 331][2]=-0.561026380862206;
    Leb_Grid_XYZW[ 331][3]= 0.001842866472905;

    Leb_Grid_XYZW[ 332][0]= 0.351828092773352;
    Leb_Grid_XYZW[ 332][1]=-0.749310611904116;
    Leb_Grid_XYZW[ 332][2]=-0.561026380862206;
    Leb_Grid_XYZW[ 332][3]= 0.001842866472905;

    Leb_Grid_XYZW[ 333][0]=-0.351828092773352;
    Leb_Grid_XYZW[ 333][1]=-0.749310611904116;
    Leb_Grid_XYZW[ 333][2]=-0.561026380862206;
    Leb_Grid_XYZW[ 333][3]= 0.001842866472905;

    Leb_Grid_XYZW[ 334][0]= 0.749310611904116;
    Leb_Grid_XYZW[ 334][1]= 0.561026380862206;
    Leb_Grid_XYZW[ 334][2]= 0.351828092773352;
    Leb_Grid_XYZW[ 334][3]= 0.001842866472905;

    Leb_Grid_XYZW[ 335][0]=-0.749310611904116;
    Leb_Grid_XYZW[ 335][1]= 0.561026380862206;
    Leb_Grid_XYZW[ 335][2]= 0.351828092773352;
    Leb_Grid_XYZW[ 335][3]= 0.001842866472905;

    Leb_Grid_XYZW[ 336][0]= 0.749310611904116;
    Leb_Grid_XYZW[ 336][1]=-0.561026380862206;
    Leb_Grid_XYZW[ 336][2]= 0.351828092773352;
    Leb_Grid_XYZW[ 336][3]= 0.001842866472905;

    Leb_Grid_XYZW[ 337][0]=-0.749310611904116;
    Leb_Grid_XYZW[ 337][1]=-0.561026380862206;
    Leb_Grid_XYZW[ 337][2]= 0.351828092773352;
    Leb_Grid_XYZW[ 337][3]= 0.001842866472905;

    Leb_Grid_XYZW[ 338][0]= 0.749310611904116;
    Leb_Grid_XYZW[ 338][1]= 0.561026380862206;
    Leb_Grid_XYZW[ 338][2]=-0.351828092773352;
    Leb_Grid_XYZW[ 338][3]= 0.001842866472905;

    Leb_Grid_XYZW[ 339][0]=-0.749310611904116;
    Leb_Grid_XYZW[ 339][1]= 0.561026380862206;
    Leb_Grid_XYZW[ 339][2]=-0.351828092773352;
    Leb_Grid_XYZW[ 339][3]= 0.001842866472905;

    Leb_Grid_XYZW[ 340][0]= 0.749310611904116;
    Leb_Grid_XYZW[ 340][1]=-0.561026380862206;
    Leb_Grid_XYZW[ 340][2]=-0.351828092773352;
    Leb_Grid_XYZW[ 340][3]= 0.001842866472905;

    Leb_Grid_XYZW[ 341][0]=-0.749310611904116;
    Leb_Grid_XYZW[ 341][1]=-0.561026380862206;
    Leb_Grid_XYZW[ 341][2]=-0.351828092773352;
    Leb_Grid_XYZW[ 341][3]= 0.001842866472905;

    Leb_Grid_XYZW[ 342][0]= 0.749310611904116;
    Leb_Grid_XYZW[ 342][1]= 0.351828092773352;
    Leb_Grid_XYZW[ 342][2]= 0.561026380862206;
    Leb_Grid_XYZW[ 342][3]= 0.001842866472905;

    Leb_Grid_XYZW[ 343][0]=-0.749310611904116;
    Leb_Grid_XYZW[ 343][1]= 0.351828092773352;
    Leb_Grid_XYZW[ 343][2]= 0.561026380862206;
    Leb_Grid_XYZW[ 343][3]= 0.001842866472905;

    Leb_Grid_XYZW[ 344][0]= 0.749310611904116;
    Leb_Grid_XYZW[ 344][1]=-0.351828092773352;
    Leb_Grid_XYZW[ 344][2]= 0.561026380862206;
    Leb_Grid_XYZW[ 344][3]= 0.001842866472905;

    Leb_Grid_XYZW[ 345][0]=-0.749310611904116;
    Leb_Grid_XYZW[ 345][1]=-0.351828092773352;
    Leb_Grid_XYZW[ 345][2]= 0.561026380862206;
    Leb_Grid_XYZW[ 345][3]= 0.001842866472905;

    Leb_Grid_XYZW[ 346][0]= 0.749310611904116;
    Leb_Grid_XYZW[ 346][1]= 0.351828092773352;
    Leb_Grid_XYZW[ 346][2]=-0.561026380862206;
    Leb_Grid_XYZW[ 346][3]= 0.001842866472905;

    Leb_Grid_XYZW[ 347][0]=-0.749310611904116;
    Leb_Grid_XYZW[ 347][1]= 0.351828092773352;
    Leb_Grid_XYZW[ 347][2]=-0.561026380862206;
    Leb_Grid_XYZW[ 347][3]= 0.001842866472905;

    Leb_Grid_XYZW[ 348][0]= 0.749310611904116;
    Leb_Grid_XYZW[ 348][1]=-0.351828092773352;
    Leb_Grid_XYZW[ 348][2]=-0.561026380862206;
    Leb_Grid_XYZW[ 348][3]= 0.001842866472905;

    Leb_Grid_XYZW[ 349][0]=-0.749310611904116;
    Leb_Grid_XYZW[ 349][1]=-0.351828092773352;
    Leb_Grid_XYZW[ 349][2]=-0.561026380862206;
    Leb_Grid_XYZW[ 349][3]= 0.001842866472905;

    Leb_Grid_XYZW[ 350][0]= 0.474239284255198;
    Leb_Grid_XYZW[ 350][1]= 0.263471665593795;
    Leb_Grid_XYZW[ 350][2]= 0.840047488359050;
    Leb_Grid_XYZW[ 350][3]= 0.001802658934377;

    Leb_Grid_XYZW[ 351][0]=-0.474239284255198;
    Leb_Grid_XYZW[ 351][1]= 0.263471665593795;
    Leb_Grid_XYZW[ 351][2]= 0.840047488359050;
    Leb_Grid_XYZW[ 351][3]= 0.001802658934377;

    Leb_Grid_XYZW[ 352][0]= 0.474239284255198;
    Leb_Grid_XYZW[ 352][1]=-0.263471665593795;
    Leb_Grid_XYZW[ 352][2]= 0.840047488359050;
    Leb_Grid_XYZW[ 352][3]= 0.001802658934377;

    Leb_Grid_XYZW[ 353][0]=-0.474239284255198;
    Leb_Grid_XYZW[ 353][1]=-0.263471665593795;
    Leb_Grid_XYZW[ 353][2]= 0.840047488359050;
    Leb_Grid_XYZW[ 353][3]= 0.001802658934377;

    Leb_Grid_XYZW[ 354][0]= 0.474239284255198;
    Leb_Grid_XYZW[ 354][1]= 0.263471665593795;
    Leb_Grid_XYZW[ 354][2]=-0.840047488359050;
    Leb_Grid_XYZW[ 354][3]= 0.001802658934377;

    Leb_Grid_XYZW[ 355][0]=-0.474239284255198;
    Leb_Grid_XYZW[ 355][1]= 0.263471665593795;
    Leb_Grid_XYZW[ 355][2]=-0.840047488359050;
    Leb_Grid_XYZW[ 355][3]= 0.001802658934377;

    Leb_Grid_XYZW[ 356][0]= 0.474239284255198;
    Leb_Grid_XYZW[ 356][1]=-0.263471665593795;
    Leb_Grid_XYZW[ 356][2]=-0.840047488359050;
    Leb_Grid_XYZW[ 356][3]= 0.001802658934377;

    Leb_Grid_XYZW[ 357][0]=-0.474239284255198;
    Leb_Grid_XYZW[ 357][1]=-0.263471665593795;
    Leb_Grid_XYZW[ 357][2]=-0.840047488359050;
    Leb_Grid_XYZW[ 357][3]= 0.001802658934377;

    Leb_Grid_XYZW[ 358][0]= 0.474239284255198;
    Leb_Grid_XYZW[ 358][1]= 0.840047488359050;
    Leb_Grid_XYZW[ 358][2]= 0.263471665593795;
    Leb_Grid_XYZW[ 358][3]= 0.001802658934377;

    Leb_Grid_XYZW[ 359][0]=-0.474239284255198;
    Leb_Grid_XYZW[ 359][1]= 0.840047488359050;
    Leb_Grid_XYZW[ 359][2]= 0.263471665593795;
    Leb_Grid_XYZW[ 359][3]= 0.001802658934377;

    Leb_Grid_XYZW[ 360][0]= 0.474239284255198;
    Leb_Grid_XYZW[ 360][1]=-0.840047488359050;
    Leb_Grid_XYZW[ 360][2]= 0.263471665593795;
    Leb_Grid_XYZW[ 360][3]= 0.001802658934377;

    Leb_Grid_XYZW[ 361][0]=-0.474239284255198;
    Leb_Grid_XYZW[ 361][1]=-0.840047488359050;
    Leb_Grid_XYZW[ 361][2]= 0.263471665593795;
    Leb_Grid_XYZW[ 361][3]= 0.001802658934377;

    Leb_Grid_XYZW[ 362][0]= 0.474239284255198;
    Leb_Grid_XYZW[ 362][1]= 0.840047488359050;
    Leb_Grid_XYZW[ 362][2]=-0.263471665593795;
    Leb_Grid_XYZW[ 362][3]= 0.001802658934377;

    Leb_Grid_XYZW[ 363][0]=-0.474239284255198;
    Leb_Grid_XYZW[ 363][1]= 0.840047488359050;
    Leb_Grid_XYZW[ 363][2]=-0.263471665593795;
    Leb_Grid_XYZW[ 363][3]= 0.001802658934377;

    Leb_Grid_XYZW[ 364][0]= 0.474239284255198;
    Leb_Grid_XYZW[ 364][1]=-0.840047488359050;
    Leb_Grid_XYZW[ 364][2]=-0.263471665593795;
    Leb_Grid_XYZW[ 364][3]= 0.001802658934377;

    Leb_Grid_XYZW[ 365][0]=-0.474239284255198;
    Leb_Grid_XYZW[ 365][1]=-0.840047488359050;
    Leb_Grid_XYZW[ 365][2]=-0.263471665593795;
    Leb_Grid_XYZW[ 365][3]= 0.001802658934377;

    Leb_Grid_XYZW[ 366][0]= 0.263471665593795;
    Leb_Grid_XYZW[ 366][1]= 0.474239284255198;
    Leb_Grid_XYZW[ 366][2]= 0.840047488359050;
    Leb_Grid_XYZW[ 366][3]= 0.001802658934377;

    Leb_Grid_XYZW[ 367][0]=-0.263471665593795;
    Leb_Grid_XYZW[ 367][1]= 0.474239284255198;
    Leb_Grid_XYZW[ 367][2]= 0.840047488359050;
    Leb_Grid_XYZW[ 367][3]= 0.001802658934377;

    Leb_Grid_XYZW[ 368][0]= 0.263471665593795;
    Leb_Grid_XYZW[ 368][1]=-0.474239284255198;
    Leb_Grid_XYZW[ 368][2]= 0.840047488359050;
    Leb_Grid_XYZW[ 368][3]= 0.001802658934377;

    Leb_Grid_XYZW[ 369][0]=-0.263471665593795;
    Leb_Grid_XYZW[ 369][1]=-0.474239284255198;
    Leb_Grid_XYZW[ 369][2]= 0.840047488359050;
    Leb_Grid_XYZW[ 369][3]= 0.001802658934377;

    Leb_Grid_XYZW[ 370][0]= 0.263471665593795;
    Leb_Grid_XYZW[ 370][1]= 0.474239284255198;
    Leb_Grid_XYZW[ 370][2]=-0.840047488359050;
    Leb_Grid_XYZW[ 370][3]= 0.001802658934377;

    Leb_Grid_XYZW[ 371][0]=-0.263471665593795;
    Leb_Grid_XYZW[ 371][1]= 0.474239284255198;
    Leb_Grid_XYZW[ 371][2]=-0.840047488359050;
    Leb_Grid_XYZW[ 371][3]= 0.001802658934377;

    Leb_Grid_XYZW[ 372][0]= 0.263471665593795;
    Leb_Grid_XYZW[ 372][1]=-0.474239284255198;
    Leb_Grid_XYZW[ 372][2]=-0.840047488359050;
    Leb_Grid_XYZW[ 372][3]= 0.001802658934377;

    Leb_Grid_XYZW[ 373][0]=-0.263471665593795;
    Leb_Grid_XYZW[ 373][1]=-0.474239284255198;
    Leb_Grid_XYZW[ 373][2]=-0.840047488359050;
    Leb_Grid_XYZW[ 373][3]= 0.001802658934377;

    Leb_Grid_XYZW[ 374][0]= 0.263471665593795;
    Leb_Grid_XYZW[ 374][1]= 0.840047488359050;
    Leb_Grid_XYZW[ 374][2]= 0.474239284255198;
    Leb_Grid_XYZW[ 374][3]= 0.001802658934377;

    Leb_Grid_XYZW[ 375][0]=-0.263471665593795;
    Leb_Grid_XYZW[ 375][1]= 0.840047488359050;
    Leb_Grid_XYZW[ 375][2]= 0.474239284255198;
    Leb_Grid_XYZW[ 375][3]= 0.001802658934377;

    Leb_Grid_XYZW[ 376][0]= 0.263471665593795;
    Leb_Grid_XYZW[ 376][1]=-0.840047488359050;
    Leb_Grid_XYZW[ 376][2]= 0.474239284255198;
    Leb_Grid_XYZW[ 376][3]= 0.001802658934377;

    Leb_Grid_XYZW[ 377][0]=-0.263471665593795;
    Leb_Grid_XYZW[ 377][1]=-0.840047488359050;
    Leb_Grid_XYZW[ 377][2]= 0.474239284255198;
    Leb_Grid_XYZW[ 377][3]= 0.001802658934377;

    Leb_Grid_XYZW[ 378][0]= 0.263471665593795;
    Leb_Grid_XYZW[ 378][1]= 0.840047488359050;
    Leb_Grid_XYZW[ 378][2]=-0.474239284255198;
    Leb_Grid_XYZW[ 378][3]= 0.001802658934377;

    Leb_Grid_XYZW[ 379][0]=-0.263471665593795;
    Leb_Grid_XYZW[ 379][1]= 0.840047488359050;
    Leb_Grid_XYZW[ 379][2]=-0.474239284255198;
    Leb_Grid_XYZW[ 379][3]= 0.001802658934377;

    Leb_Grid_XYZW[ 380][0]= 0.263471665593795;
    Leb_Grid_XYZW[ 380][1]=-0.840047488359050;
    Leb_Grid_XYZW[ 380][2]=-0.474239284255198;
    Leb_Grid_XYZW[ 380][3]= 0.001802658934377;

    Leb_Grid_XYZW[ 381][0]=-0.263471665593795;
    Leb_Grid_XYZW[ 381][1]=-0.840047488359050;
    Leb_Grid_XYZW[ 381][2]=-0.474239284255198;
    Leb_Grid_XYZW[ 381][3]= 0.001802658934377;

    Leb_Grid_XYZW[ 382][0]= 0.840047488359050;
    Leb_Grid_XYZW[ 382][1]= 0.474239284255198;
    Leb_Grid_XYZW[ 382][2]= 0.263471665593795;
    Leb_Grid_XYZW[ 382][3]= 0.001802658934377;

    Leb_Grid_XYZW[ 383][0]=-0.840047488359050;
    Leb_Grid_XYZW[ 383][1]= 0.474239284255198;
    Leb_Grid_XYZW[ 383][2]= 0.263471665593795;
    Leb_Grid_XYZW[ 383][3]= 0.001802658934377;

    Leb_Grid_XYZW[ 384][0]= 0.840047488359050;
    Leb_Grid_XYZW[ 384][1]=-0.474239284255198;
    Leb_Grid_XYZW[ 384][2]= 0.263471665593795;
    Leb_Grid_XYZW[ 384][3]= 0.001802658934377;

    Leb_Grid_XYZW[ 385][0]=-0.840047488359050;
    Leb_Grid_XYZW[ 385][1]=-0.474239284255198;
    Leb_Grid_XYZW[ 385][2]= 0.263471665593795;
    Leb_Grid_XYZW[ 385][3]= 0.001802658934377;

    Leb_Grid_XYZW[ 386][0]= 0.840047488359050;
    Leb_Grid_XYZW[ 386][1]= 0.474239284255198;
    Leb_Grid_XYZW[ 386][2]=-0.263471665593795;
    Leb_Grid_XYZW[ 386][3]= 0.001802658934377;

    Leb_Grid_XYZW[ 387][0]=-0.840047488359050;
    Leb_Grid_XYZW[ 387][1]= 0.474239284255198;
    Leb_Grid_XYZW[ 387][2]=-0.263471665593795;
    Leb_Grid_XYZW[ 387][3]= 0.001802658934377;

    Leb_Grid_XYZW[ 388][0]= 0.840047488359050;
    Leb_Grid_XYZW[ 388][1]=-0.474239284255198;
    Leb_Grid_XYZW[ 388][2]=-0.263471665593795;
    Leb_Grid_XYZW[ 388][3]= 0.001802658934377;

    Leb_Grid_XYZW[ 389][0]=-0.840047488359050;
    Leb_Grid_XYZW[ 389][1]=-0.474239284255198;
    Leb_Grid_XYZW[ 389][2]=-0.263471665593795;
    Leb_Grid_XYZW[ 389][3]= 0.001802658934377;

    Leb_Grid_XYZW[ 390][0]= 0.840047488359050;
    Leb_Grid_XYZW[ 390][1]= 0.263471665593795;
    Leb_Grid_XYZW[ 390][2]= 0.474239284255198;
    Leb_Grid_XYZW[ 390][3]= 0.001802658934377;

    Leb_Grid_XYZW[ 391][0]=-0.840047488359050;
    Leb_Grid_XYZW[ 391][1]= 0.263471665593795;
    Leb_Grid_XYZW[ 391][2]= 0.474239284255198;
    Leb_Grid_XYZW[ 391][3]= 0.001802658934377;

    Leb_Grid_XYZW[ 392][0]= 0.840047488359050;
    Leb_Grid_XYZW[ 392][1]=-0.263471665593795;
    Leb_Grid_XYZW[ 392][2]= 0.474239284255198;
    Leb_Grid_XYZW[ 392][3]= 0.001802658934377;

    Leb_Grid_XYZW[ 393][0]=-0.840047488359050;
    Leb_Grid_XYZW[ 393][1]=-0.263471665593795;
    Leb_Grid_XYZW[ 393][2]= 0.474239284255198;
    Leb_Grid_XYZW[ 393][3]= 0.001802658934377;

    Leb_Grid_XYZW[ 394][0]= 0.840047488359050;
    Leb_Grid_XYZW[ 394][1]= 0.263471665593795;
    Leb_Grid_XYZW[ 394][2]=-0.474239284255198;
    Leb_Grid_XYZW[ 394][3]= 0.001802658934377;

    Leb_Grid_XYZW[ 395][0]=-0.840047488359050;
    Leb_Grid_XYZW[ 395][1]= 0.263471665593795;
    Leb_Grid_XYZW[ 395][2]=-0.474239284255198;
    Leb_Grid_XYZW[ 395][3]= 0.001802658934377;

    Leb_Grid_XYZW[ 396][0]= 0.840047488359050;
    Leb_Grid_XYZW[ 396][1]=-0.263471665593795;
    Leb_Grid_XYZW[ 396][2]=-0.474239284255198;
    Leb_Grid_XYZW[ 396][3]= 0.001802658934377;

    Leb_Grid_XYZW[ 397][0]=-0.840047488359050;
    Leb_Grid_XYZW[ 397][1]=-0.263471665593795;
    Leb_Grid_XYZW[ 397][2]=-0.474239284255198;
    Leb_Grid_XYZW[ 397][3]= 0.001802658934377;

    Leb_Grid_XYZW[ 398][0]= 0.598412649788538;
    Leb_Grid_XYZW[ 398][1]= 0.181664084036021;
    Leb_Grid_XYZW[ 398][2]= 0.780320742479920;
    Leb_Grid_XYZW[ 398][3]= 0.001849830560444;

    Leb_Grid_XYZW[ 399][0]=-0.598412649788538;
    Leb_Grid_XYZW[ 399][1]= 0.181664084036021;
    Leb_Grid_XYZW[ 399][2]= 0.780320742479920;
    Leb_Grid_XYZW[ 399][3]= 0.001849830560444;

    Leb_Grid_XYZW[ 400][0]= 0.598412649788538;
    Leb_Grid_XYZW[ 400][1]=-0.181664084036021;
    Leb_Grid_XYZW[ 400][2]= 0.780320742479920;
    Leb_Grid_XYZW[ 400][3]= 0.001849830560444;

    Leb_Grid_XYZW[ 401][0]=-0.598412649788538;
    Leb_Grid_XYZW[ 401][1]=-0.181664084036021;
    Leb_Grid_XYZW[ 401][2]= 0.780320742479920;
    Leb_Grid_XYZW[ 401][3]= 0.001849830560444;

    Leb_Grid_XYZW[ 402][0]= 0.598412649788538;
    Leb_Grid_XYZW[ 402][1]= 0.181664084036021;
    Leb_Grid_XYZW[ 402][2]=-0.780320742479920;
    Leb_Grid_XYZW[ 402][3]= 0.001849830560444;

    Leb_Grid_XYZW[ 403][0]=-0.598412649788538;
    Leb_Grid_XYZW[ 403][1]= 0.181664084036021;
    Leb_Grid_XYZW[ 403][2]=-0.780320742479920;
    Leb_Grid_XYZW[ 403][3]= 0.001849830560444;

    Leb_Grid_XYZW[ 404][0]= 0.598412649788538;
    Leb_Grid_XYZW[ 404][1]=-0.181664084036021;
    Leb_Grid_XYZW[ 404][2]=-0.780320742479920;
    Leb_Grid_XYZW[ 404][3]= 0.001849830560444;

    Leb_Grid_XYZW[ 405][0]=-0.598412649788538;
    Leb_Grid_XYZW[ 405][1]=-0.181664084036021;
    Leb_Grid_XYZW[ 405][2]=-0.780320742479920;
    Leb_Grid_XYZW[ 405][3]= 0.001849830560444;

    Leb_Grid_XYZW[ 406][0]= 0.598412649788538;
    Leb_Grid_XYZW[ 406][1]= 0.780320742479920;
    Leb_Grid_XYZW[ 406][2]= 0.181664084036021;
    Leb_Grid_XYZW[ 406][3]= 0.001849830560444;

    Leb_Grid_XYZW[ 407][0]=-0.598412649788538;
    Leb_Grid_XYZW[ 407][1]= 0.780320742479920;
    Leb_Grid_XYZW[ 407][2]= 0.181664084036021;
    Leb_Grid_XYZW[ 407][3]= 0.001849830560444;

    Leb_Grid_XYZW[ 408][0]= 0.598412649788538;
    Leb_Grid_XYZW[ 408][1]=-0.780320742479920;
    Leb_Grid_XYZW[ 408][2]= 0.181664084036021;
    Leb_Grid_XYZW[ 408][3]= 0.001849830560444;

    Leb_Grid_XYZW[ 409][0]=-0.598412649788538;
    Leb_Grid_XYZW[ 409][1]=-0.780320742479920;
    Leb_Grid_XYZW[ 409][2]= 0.181664084036021;
    Leb_Grid_XYZW[ 409][3]= 0.001849830560444;

    Leb_Grid_XYZW[ 410][0]= 0.598412649788538;
    Leb_Grid_XYZW[ 410][1]= 0.780320742479920;
    Leb_Grid_XYZW[ 410][2]=-0.181664084036021;
    Leb_Grid_XYZW[ 410][3]= 0.001849830560444;

    Leb_Grid_XYZW[ 411][0]=-0.598412649788538;
    Leb_Grid_XYZW[ 411][1]= 0.780320742479920;
    Leb_Grid_XYZW[ 411][2]=-0.181664084036021;
    Leb_Grid_XYZW[ 411][3]= 0.001849830560444;

    Leb_Grid_XYZW[ 412][0]= 0.598412649788538;
    Leb_Grid_XYZW[ 412][1]=-0.780320742479920;
    Leb_Grid_XYZW[ 412][2]=-0.181664084036021;
    Leb_Grid_XYZW[ 412][3]= 0.001849830560444;

    Leb_Grid_XYZW[ 413][0]=-0.598412649788538;
    Leb_Grid_XYZW[ 413][1]=-0.780320742479920;
    Leb_Grid_XYZW[ 413][2]=-0.181664084036021;
    Leb_Grid_XYZW[ 413][3]= 0.001849830560444;

    Leb_Grid_XYZW[ 414][0]= 0.181664084036021;
    Leb_Grid_XYZW[ 414][1]= 0.598412649788538;
    Leb_Grid_XYZW[ 414][2]= 0.780320742479920;
    Leb_Grid_XYZW[ 414][3]= 0.001849830560444;

    Leb_Grid_XYZW[ 415][0]=-0.181664084036021;
    Leb_Grid_XYZW[ 415][1]= 0.598412649788538;
    Leb_Grid_XYZW[ 415][2]= 0.780320742479920;
    Leb_Grid_XYZW[ 415][3]= 0.001849830560444;

    Leb_Grid_XYZW[ 416][0]= 0.181664084036021;
    Leb_Grid_XYZW[ 416][1]=-0.598412649788538;
    Leb_Grid_XYZW[ 416][2]= 0.780320742479920;
    Leb_Grid_XYZW[ 416][3]= 0.001849830560444;

    Leb_Grid_XYZW[ 417][0]=-0.181664084036021;
    Leb_Grid_XYZW[ 417][1]=-0.598412649788538;
    Leb_Grid_XYZW[ 417][2]= 0.780320742479920;
    Leb_Grid_XYZW[ 417][3]= 0.001849830560444;

    Leb_Grid_XYZW[ 418][0]= 0.181664084036021;
    Leb_Grid_XYZW[ 418][1]= 0.598412649788538;
    Leb_Grid_XYZW[ 418][2]=-0.780320742479920;
    Leb_Grid_XYZW[ 418][3]= 0.001849830560444;

    Leb_Grid_XYZW[ 419][0]=-0.181664084036021;
    Leb_Grid_XYZW[ 419][1]= 0.598412649788538;
    Leb_Grid_XYZW[ 419][2]=-0.780320742479920;
    Leb_Grid_XYZW[ 419][3]= 0.001849830560444;

    Leb_Grid_XYZW[ 420][0]= 0.181664084036021;
    Leb_Grid_XYZW[ 420][1]=-0.598412649788538;
    Leb_Grid_XYZW[ 420][2]=-0.780320742479920;
    Leb_Grid_XYZW[ 420][3]= 0.001849830560444;

    Leb_Grid_XYZW[ 421][0]=-0.181664084036021;
    Leb_Grid_XYZW[ 421][1]=-0.598412649788538;
    Leb_Grid_XYZW[ 421][2]=-0.780320742479920;
    Leb_Grid_XYZW[ 421][3]= 0.001849830560444;

    Leb_Grid_XYZW[ 422][0]= 0.181664084036021;
    Leb_Grid_XYZW[ 422][1]= 0.780320742479920;
    Leb_Grid_XYZW[ 422][2]= 0.598412649788538;
    Leb_Grid_XYZW[ 422][3]= 0.001849830560444;

    Leb_Grid_XYZW[ 423][0]=-0.181664084036021;
    Leb_Grid_XYZW[ 423][1]= 0.780320742479920;
    Leb_Grid_XYZW[ 423][2]= 0.598412649788538;
    Leb_Grid_XYZW[ 423][3]= 0.001849830560444;

    Leb_Grid_XYZW[ 424][0]= 0.181664084036021;
    Leb_Grid_XYZW[ 424][1]=-0.780320742479920;
    Leb_Grid_XYZW[ 424][2]= 0.598412649788538;
    Leb_Grid_XYZW[ 424][3]= 0.001849830560444;

    Leb_Grid_XYZW[ 425][0]=-0.181664084036021;
    Leb_Grid_XYZW[ 425][1]=-0.780320742479920;
    Leb_Grid_XYZW[ 425][2]= 0.598412649788538;
    Leb_Grid_XYZW[ 425][3]= 0.001849830560444;

    Leb_Grid_XYZW[ 426][0]= 0.181664084036021;
    Leb_Grid_XYZW[ 426][1]= 0.780320742479920;
    Leb_Grid_XYZW[ 426][2]=-0.598412649788538;
    Leb_Grid_XYZW[ 426][3]= 0.001849830560444;

    Leb_Grid_XYZW[ 427][0]=-0.181664084036021;
    Leb_Grid_XYZW[ 427][1]= 0.780320742479920;
    Leb_Grid_XYZW[ 427][2]=-0.598412649788538;
    Leb_Grid_XYZW[ 427][3]= 0.001849830560444;

    Leb_Grid_XYZW[ 428][0]= 0.181664084036021;
    Leb_Grid_XYZW[ 428][1]=-0.780320742479920;
    Leb_Grid_XYZW[ 428][2]=-0.598412649788538;
    Leb_Grid_XYZW[ 428][3]= 0.001849830560444;

    Leb_Grid_XYZW[ 429][0]=-0.181664084036021;
    Leb_Grid_XYZW[ 429][1]=-0.780320742479920;
    Leb_Grid_XYZW[ 429][2]=-0.598412649788538;
    Leb_Grid_XYZW[ 429][3]= 0.001849830560444;

    Leb_Grid_XYZW[ 430][0]= 0.780320742479920;
    Leb_Grid_XYZW[ 430][1]= 0.598412649788538;
    Leb_Grid_XYZW[ 430][2]= 0.181664084036021;
    Leb_Grid_XYZW[ 430][3]= 0.001849830560444;

    Leb_Grid_XYZW[ 431][0]=-0.780320742479920;
    Leb_Grid_XYZW[ 431][1]= 0.598412649788538;
    Leb_Grid_XYZW[ 431][2]= 0.181664084036021;
    Leb_Grid_XYZW[ 431][3]= 0.001849830560444;

    Leb_Grid_XYZW[ 432][0]= 0.780320742479920;
    Leb_Grid_XYZW[ 432][1]=-0.598412649788538;
    Leb_Grid_XYZW[ 432][2]= 0.181664084036021;
    Leb_Grid_XYZW[ 432][3]= 0.001849830560444;

    Leb_Grid_XYZW[ 433][0]=-0.780320742479920;
    Leb_Grid_XYZW[ 433][1]=-0.598412649788538;
    Leb_Grid_XYZW[ 433][2]= 0.181664084036021;
    Leb_Grid_XYZW[ 433][3]= 0.001849830560444;

    Leb_Grid_XYZW[ 434][0]= 0.780320742479920;
    Leb_Grid_XYZW[ 434][1]= 0.598412649788538;
    Leb_Grid_XYZW[ 434][2]=-0.181664084036021;
    Leb_Grid_XYZW[ 434][3]= 0.001849830560444;

    Leb_Grid_XYZW[ 435][0]=-0.780320742479920;
    Leb_Grid_XYZW[ 435][1]= 0.598412649788538;
    Leb_Grid_XYZW[ 435][2]=-0.181664084036021;
    Leb_Grid_XYZW[ 435][3]= 0.001849830560444;

    Leb_Grid_XYZW[ 436][0]= 0.780320742479920;
    Leb_Grid_XYZW[ 436][1]=-0.598412649788538;
    Leb_Grid_XYZW[ 436][2]=-0.181664084036021;
    Leb_Grid_XYZW[ 436][3]= 0.001849830560444;

    Leb_Grid_XYZW[ 437][0]=-0.780320742479920;
    Leb_Grid_XYZW[ 437][1]=-0.598412649788538;
    Leb_Grid_XYZW[ 437][2]=-0.181664084036021;
    Leb_Grid_XYZW[ 437][3]= 0.001849830560444;

    Leb_Grid_XYZW[ 438][0]= 0.780320742479920;
    Leb_Grid_XYZW[ 438][1]= 0.181664084036021;
    Leb_Grid_XYZW[ 438][2]= 0.598412649788538;
    Leb_Grid_XYZW[ 438][3]= 0.001849830560444;

    Leb_Grid_XYZW[ 439][0]=-0.780320742479920;
    Leb_Grid_XYZW[ 439][1]= 0.181664084036021;
    Leb_Grid_XYZW[ 439][2]= 0.598412649788538;
    Leb_Grid_XYZW[ 439][3]= 0.001849830560444;

    Leb_Grid_XYZW[ 440][0]= 0.780320742479920;
    Leb_Grid_XYZW[ 440][1]=-0.181664084036021;
    Leb_Grid_XYZW[ 440][2]= 0.598412649788538;
    Leb_Grid_XYZW[ 440][3]= 0.001849830560444;

    Leb_Grid_XYZW[ 441][0]=-0.780320742479920;
    Leb_Grid_XYZW[ 441][1]=-0.181664084036021;
    Leb_Grid_XYZW[ 441][2]= 0.598412649788538;
    Leb_Grid_XYZW[ 441][3]= 0.001849830560444;

    Leb_Grid_XYZW[ 442][0]= 0.780320742479920;
    Leb_Grid_XYZW[ 442][1]= 0.181664084036021;
    Leb_Grid_XYZW[ 442][2]=-0.598412649788538;
    Leb_Grid_XYZW[ 442][3]= 0.001849830560444;

    Leb_Grid_XYZW[ 443][0]=-0.780320742479920;
    Leb_Grid_XYZW[ 443][1]= 0.181664084036021;
    Leb_Grid_XYZW[ 443][2]=-0.598412649788538;
    Leb_Grid_XYZW[ 443][3]= 0.001849830560444;

    Leb_Grid_XYZW[ 444][0]= 0.780320742479920;
    Leb_Grid_XYZW[ 444][1]=-0.181664084036021;
    Leb_Grid_XYZW[ 444][2]=-0.598412649788538;
    Leb_Grid_XYZW[ 444][3]= 0.001849830560444;

    Leb_Grid_XYZW[ 445][0]=-0.780320742479920;
    Leb_Grid_XYZW[ 445][1]=-0.181664084036021;
    Leb_Grid_XYZW[ 445][2]=-0.598412649788538;
    Leb_Grid_XYZW[ 445][3]= 0.001849830560444;

    Leb_Grid_XYZW[ 446][0]= 0.379103540769556;
    Leb_Grid_XYZW[ 446][1]= 0.172079522565688;
    Leb_Grid_XYZW[ 446][2]= 0.909213475092374;
    Leb_Grid_XYZW[ 446][3]= 0.001713904507107;

    Leb_Grid_XYZW[ 447][0]=-0.379103540769556;
    Leb_Grid_XYZW[ 447][1]= 0.172079522565688;
    Leb_Grid_XYZW[ 447][2]= 0.909213475092374;
    Leb_Grid_XYZW[ 447][3]= 0.001713904507107;

    Leb_Grid_XYZW[ 448][0]= 0.379103540769556;
    Leb_Grid_XYZW[ 448][1]=-0.172079522565688;
    Leb_Grid_XYZW[ 448][2]= 0.909213475092374;
    Leb_Grid_XYZW[ 448][3]= 0.001713904507107;

    Leb_Grid_XYZW[ 449][0]=-0.379103540769556;
    Leb_Grid_XYZW[ 449][1]=-0.172079522565688;
    Leb_Grid_XYZW[ 449][2]= 0.909213475092374;
    Leb_Grid_XYZW[ 449][3]= 0.001713904507107;

    Leb_Grid_XYZW[ 450][0]= 0.379103540769556;
    Leb_Grid_XYZW[ 450][1]= 0.172079522565688;
    Leb_Grid_XYZW[ 450][2]=-0.909213475092374;
    Leb_Grid_XYZW[ 450][3]= 0.001713904507107;

    Leb_Grid_XYZW[ 451][0]=-0.379103540769556;
    Leb_Grid_XYZW[ 451][1]= 0.172079522565688;
    Leb_Grid_XYZW[ 451][2]=-0.909213475092374;
    Leb_Grid_XYZW[ 451][3]= 0.001713904507107;

    Leb_Grid_XYZW[ 452][0]= 0.379103540769556;
    Leb_Grid_XYZW[ 452][1]=-0.172079522565688;
    Leb_Grid_XYZW[ 452][2]=-0.909213475092374;
    Leb_Grid_XYZW[ 452][3]= 0.001713904507107;

    Leb_Grid_XYZW[ 453][0]=-0.379103540769556;
    Leb_Grid_XYZW[ 453][1]=-0.172079522565688;
    Leb_Grid_XYZW[ 453][2]=-0.909213475092374;
    Leb_Grid_XYZW[ 453][3]= 0.001713904507107;

    Leb_Grid_XYZW[ 454][0]= 0.379103540769556;
    Leb_Grid_XYZW[ 454][1]= 0.909213475092374;
    Leb_Grid_XYZW[ 454][2]= 0.172079522565688;
    Leb_Grid_XYZW[ 454][3]= 0.001713904507107;

    Leb_Grid_XYZW[ 455][0]=-0.379103540769556;
    Leb_Grid_XYZW[ 455][1]= 0.909213475092374;
    Leb_Grid_XYZW[ 455][2]= 0.172079522565688;
    Leb_Grid_XYZW[ 455][3]= 0.001713904507107;

    Leb_Grid_XYZW[ 456][0]= 0.379103540769556;
    Leb_Grid_XYZW[ 456][1]=-0.909213475092374;
    Leb_Grid_XYZW[ 456][2]= 0.172079522565688;
    Leb_Grid_XYZW[ 456][3]= 0.001713904507107;

    Leb_Grid_XYZW[ 457][0]=-0.379103540769556;
    Leb_Grid_XYZW[ 457][1]=-0.909213475092374;
    Leb_Grid_XYZW[ 457][2]= 0.172079522565688;
    Leb_Grid_XYZW[ 457][3]= 0.001713904507107;

    Leb_Grid_XYZW[ 458][0]= 0.379103540769556;
    Leb_Grid_XYZW[ 458][1]= 0.909213475092374;
    Leb_Grid_XYZW[ 458][2]=-0.172079522565688;
    Leb_Grid_XYZW[ 458][3]= 0.001713904507107;

    Leb_Grid_XYZW[ 459][0]=-0.379103540769556;
    Leb_Grid_XYZW[ 459][1]= 0.909213475092374;
    Leb_Grid_XYZW[ 459][2]=-0.172079522565688;
    Leb_Grid_XYZW[ 459][3]= 0.001713904507107;

    Leb_Grid_XYZW[ 460][0]= 0.379103540769556;
    Leb_Grid_XYZW[ 460][1]=-0.909213475092374;
    Leb_Grid_XYZW[ 460][2]=-0.172079522565688;
    Leb_Grid_XYZW[ 460][3]= 0.001713904507107;

    Leb_Grid_XYZW[ 461][0]=-0.379103540769556;
    Leb_Grid_XYZW[ 461][1]=-0.909213475092374;
    Leb_Grid_XYZW[ 461][2]=-0.172079522565688;
    Leb_Grid_XYZW[ 461][3]= 0.001713904507107;

    Leb_Grid_XYZW[ 462][0]= 0.172079522565688;
    Leb_Grid_XYZW[ 462][1]= 0.379103540769556;
    Leb_Grid_XYZW[ 462][2]= 0.909213475092374;
    Leb_Grid_XYZW[ 462][3]= 0.001713904507107;

    Leb_Grid_XYZW[ 463][0]=-0.172079522565688;
    Leb_Grid_XYZW[ 463][1]= 0.379103540769556;
    Leb_Grid_XYZW[ 463][2]= 0.909213475092374;
    Leb_Grid_XYZW[ 463][3]= 0.001713904507107;

    Leb_Grid_XYZW[ 464][0]= 0.172079522565688;
    Leb_Grid_XYZW[ 464][1]=-0.379103540769556;
    Leb_Grid_XYZW[ 464][2]= 0.909213475092374;
    Leb_Grid_XYZW[ 464][3]= 0.001713904507107;

    Leb_Grid_XYZW[ 465][0]=-0.172079522565688;
    Leb_Grid_XYZW[ 465][1]=-0.379103540769556;
    Leb_Grid_XYZW[ 465][2]= 0.909213475092374;
    Leb_Grid_XYZW[ 465][3]= 0.001713904507107;

    Leb_Grid_XYZW[ 466][0]= 0.172079522565688;
    Leb_Grid_XYZW[ 466][1]= 0.379103540769556;
    Leb_Grid_XYZW[ 466][2]=-0.909213475092374;
    Leb_Grid_XYZW[ 466][3]= 0.001713904507107;

    Leb_Grid_XYZW[ 467][0]=-0.172079522565688;
    Leb_Grid_XYZW[ 467][1]= 0.379103540769556;
    Leb_Grid_XYZW[ 467][2]=-0.909213475092374;
    Leb_Grid_XYZW[ 467][3]= 0.001713904507107;

    Leb_Grid_XYZW[ 468][0]= 0.172079522565688;
    Leb_Grid_XYZW[ 468][1]=-0.379103540769556;
    Leb_Grid_XYZW[ 468][2]=-0.909213475092374;
    Leb_Grid_XYZW[ 468][3]= 0.001713904507107;

    Leb_Grid_XYZW[ 469][0]=-0.172079522565688;
    Leb_Grid_XYZW[ 469][1]=-0.379103540769556;
    Leb_Grid_XYZW[ 469][2]=-0.909213475092374;
    Leb_Grid_XYZW[ 469][3]= 0.001713904507107;

    Leb_Grid_XYZW[ 470][0]= 0.172079522565688;
    Leb_Grid_XYZW[ 470][1]= 0.909213475092374;
    Leb_Grid_XYZW[ 470][2]= 0.379103540769556;
    Leb_Grid_XYZW[ 470][3]= 0.001713904507107;

    Leb_Grid_XYZW[ 471][0]=-0.172079522565688;
    Leb_Grid_XYZW[ 471][1]= 0.909213475092374;
    Leb_Grid_XYZW[ 471][2]= 0.379103540769556;
    Leb_Grid_XYZW[ 471][3]= 0.001713904507107;

    Leb_Grid_XYZW[ 472][0]= 0.172079522565688;
    Leb_Grid_XYZW[ 472][1]=-0.909213475092374;
    Leb_Grid_XYZW[ 472][2]= 0.379103540769556;
    Leb_Grid_XYZW[ 472][3]= 0.001713904507107;

    Leb_Grid_XYZW[ 473][0]=-0.172079522565688;
    Leb_Grid_XYZW[ 473][1]=-0.909213475092374;
    Leb_Grid_XYZW[ 473][2]= 0.379103540769556;
    Leb_Grid_XYZW[ 473][3]= 0.001713904507107;

    Leb_Grid_XYZW[ 474][0]= 0.172079522565688;
    Leb_Grid_XYZW[ 474][1]= 0.909213475092374;
    Leb_Grid_XYZW[ 474][2]=-0.379103540769556;
    Leb_Grid_XYZW[ 474][3]= 0.001713904507107;

    Leb_Grid_XYZW[ 475][0]=-0.172079522565688;
    Leb_Grid_XYZW[ 475][1]= 0.909213475092374;
    Leb_Grid_XYZW[ 475][2]=-0.379103540769556;
    Leb_Grid_XYZW[ 475][3]= 0.001713904507107;

    Leb_Grid_XYZW[ 476][0]= 0.172079522565688;
    Leb_Grid_XYZW[ 476][1]=-0.909213475092374;
    Leb_Grid_XYZW[ 476][2]=-0.379103540769556;
    Leb_Grid_XYZW[ 476][3]= 0.001713904507107;

    Leb_Grid_XYZW[ 477][0]=-0.172079522565688;
    Leb_Grid_XYZW[ 477][1]=-0.909213475092374;
    Leb_Grid_XYZW[ 477][2]=-0.379103540769556;
    Leb_Grid_XYZW[ 477][3]= 0.001713904507107;

    Leb_Grid_XYZW[ 478][0]= 0.909213475092374;
    Leb_Grid_XYZW[ 478][1]= 0.379103540769556;
    Leb_Grid_XYZW[ 478][2]= 0.172079522565688;
    Leb_Grid_XYZW[ 478][3]= 0.001713904507107;

    Leb_Grid_XYZW[ 479][0]=-0.909213475092374;
    Leb_Grid_XYZW[ 479][1]= 0.379103540769556;
    Leb_Grid_XYZW[ 479][2]= 0.172079522565688;
    Leb_Grid_XYZW[ 479][3]= 0.001713904507107;

    Leb_Grid_XYZW[ 480][0]= 0.909213475092374;
    Leb_Grid_XYZW[ 480][1]=-0.379103540769556;
    Leb_Grid_XYZW[ 480][2]= 0.172079522565688;
    Leb_Grid_XYZW[ 480][3]= 0.001713904507107;

    Leb_Grid_XYZW[ 481][0]=-0.909213475092374;
    Leb_Grid_XYZW[ 481][1]=-0.379103540769556;
    Leb_Grid_XYZW[ 481][2]= 0.172079522565688;
    Leb_Grid_XYZW[ 481][3]= 0.001713904507107;

    Leb_Grid_XYZW[ 482][0]= 0.909213475092374;
    Leb_Grid_XYZW[ 482][1]= 0.379103540769556;
    Leb_Grid_XYZW[ 482][2]=-0.172079522565688;
    Leb_Grid_XYZW[ 482][3]= 0.001713904507107;

    Leb_Grid_XYZW[ 483][0]=-0.909213475092374;
    Leb_Grid_XYZW[ 483][1]= 0.379103540769556;
    Leb_Grid_XYZW[ 483][2]=-0.172079522565688;
    Leb_Grid_XYZW[ 483][3]= 0.001713904507107;

    Leb_Grid_XYZW[ 484][0]= 0.909213475092374;
    Leb_Grid_XYZW[ 484][1]=-0.379103540769556;
    Leb_Grid_XYZW[ 484][2]=-0.172079522565688;
    Leb_Grid_XYZW[ 484][3]= 0.001713904507107;

    Leb_Grid_XYZW[ 485][0]=-0.909213475092374;
    Leb_Grid_XYZW[ 485][1]=-0.379103540769556;
    Leb_Grid_XYZW[ 485][2]=-0.172079522565688;
    Leb_Grid_XYZW[ 485][3]= 0.001713904507107;

    Leb_Grid_XYZW[ 486][0]= 0.909213475092374;
    Leb_Grid_XYZW[ 486][1]= 0.172079522565688;
    Leb_Grid_XYZW[ 486][2]= 0.379103540769556;
    Leb_Grid_XYZW[ 486][3]= 0.001713904507107;

    Leb_Grid_XYZW[ 487][0]=-0.909213475092374;
    Leb_Grid_XYZW[ 487][1]= 0.172079522565688;
    Leb_Grid_XYZW[ 487][2]= 0.379103540769556;
    Leb_Grid_XYZW[ 487][3]= 0.001713904507107;

    Leb_Grid_XYZW[ 488][0]= 0.909213475092374;
    Leb_Grid_XYZW[ 488][1]=-0.172079522565688;
    Leb_Grid_XYZW[ 488][2]= 0.379103540769556;
    Leb_Grid_XYZW[ 488][3]= 0.001713904507107;

    Leb_Grid_XYZW[ 489][0]=-0.909213475092374;
    Leb_Grid_XYZW[ 489][1]=-0.172079522565688;
    Leb_Grid_XYZW[ 489][2]= 0.379103540769556;
    Leb_Grid_XYZW[ 489][3]= 0.001713904507107;

    Leb_Grid_XYZW[ 490][0]= 0.909213475092374;
    Leb_Grid_XYZW[ 490][1]= 0.172079522565688;
    Leb_Grid_XYZW[ 490][2]=-0.379103540769556;
    Leb_Grid_XYZW[ 490][3]= 0.001713904507107;

    Leb_Grid_XYZW[ 491][0]=-0.909213475092374;
    Leb_Grid_XYZW[ 491][1]= 0.172079522565688;
    Leb_Grid_XYZW[ 491][2]=-0.379103540769556;
    Leb_Grid_XYZW[ 491][3]= 0.001713904507107;

    Leb_Grid_XYZW[ 492][0]= 0.909213475092374;
    Leb_Grid_XYZW[ 492][1]=-0.172079522565688;
    Leb_Grid_XYZW[ 492][2]=-0.379103540769556;
    Leb_Grid_XYZW[ 492][3]= 0.001713904507107;

    Leb_Grid_XYZW[ 493][0]=-0.909213475092374;
    Leb_Grid_XYZW[ 493][1]=-0.172079522565688;
    Leb_Grid_XYZW[ 493][2]=-0.379103540769556;
    Leb_Grid_XYZW[ 493][3]= 0.001713904507107;

    Leb_Grid_XYZW[ 494][0]= 0.277867319058624;
    Leb_Grid_XYZW[ 494][1]= 0.082130215819325;
    Leb_Grid_XYZW[ 494][2]= 0.957102074310073;
    Leb_Grid_XYZW[ 494][3]= 0.001555213603397;

    Leb_Grid_XYZW[ 495][0]=-0.277867319058624;
    Leb_Grid_XYZW[ 495][1]= 0.082130215819325;
    Leb_Grid_XYZW[ 495][2]= 0.957102074310073;
    Leb_Grid_XYZW[ 495][3]= 0.001555213603397;

    Leb_Grid_XYZW[ 496][0]= 0.277867319058624;
    Leb_Grid_XYZW[ 496][1]=-0.082130215819325;
    Leb_Grid_XYZW[ 496][2]= 0.957102074310073;
    Leb_Grid_XYZW[ 496][3]= 0.001555213603397;

    Leb_Grid_XYZW[ 497][0]=-0.277867319058624;
    Leb_Grid_XYZW[ 497][1]=-0.082130215819325;
    Leb_Grid_XYZW[ 497][2]= 0.957102074310073;
    Leb_Grid_XYZW[ 497][3]= 0.001555213603397;

    Leb_Grid_XYZW[ 498][0]= 0.277867319058624;
    Leb_Grid_XYZW[ 498][1]= 0.082130215819325;
    Leb_Grid_XYZW[ 498][2]=-0.957102074310073;
    Leb_Grid_XYZW[ 498][3]= 0.001555213603397;

    Leb_Grid_XYZW[ 499][0]=-0.277867319058624;
    Leb_Grid_XYZW[ 499][1]= 0.082130215819325;
    Leb_Grid_XYZW[ 499][2]=-0.957102074310073;
    Leb_Grid_XYZW[ 499][3]= 0.001555213603397;

    Leb_Grid_XYZW[ 500][0]= 0.277867319058624;
    Leb_Grid_XYZW[ 500][1]=-0.082130215819325;
    Leb_Grid_XYZW[ 500][2]=-0.957102074310073;
    Leb_Grid_XYZW[ 500][3]= 0.001555213603397;

    Leb_Grid_XYZW[ 501][0]=-0.277867319058624;
    Leb_Grid_XYZW[ 501][1]=-0.082130215819325;
    Leb_Grid_XYZW[ 501][2]=-0.957102074310073;
    Leb_Grid_XYZW[ 501][3]= 0.001555213603397;

    Leb_Grid_XYZW[ 502][0]= 0.277867319058624;
    Leb_Grid_XYZW[ 502][1]= 0.957102074310073;
    Leb_Grid_XYZW[ 502][2]= 0.082130215819325;
    Leb_Grid_XYZW[ 502][3]= 0.001555213603397;

    Leb_Grid_XYZW[ 503][0]=-0.277867319058624;
    Leb_Grid_XYZW[ 503][1]= 0.957102074310073;
    Leb_Grid_XYZW[ 503][2]= 0.082130215819325;
    Leb_Grid_XYZW[ 503][3]= 0.001555213603397;

    Leb_Grid_XYZW[ 504][0]= 0.277867319058624;
    Leb_Grid_XYZW[ 504][1]=-0.957102074310073;
    Leb_Grid_XYZW[ 504][2]= 0.082130215819325;
    Leb_Grid_XYZW[ 504][3]= 0.001555213603397;

    Leb_Grid_XYZW[ 505][0]=-0.277867319058624;
    Leb_Grid_XYZW[ 505][1]=-0.957102074310073;
    Leb_Grid_XYZW[ 505][2]= 0.082130215819325;
    Leb_Grid_XYZW[ 505][3]= 0.001555213603397;

    Leb_Grid_XYZW[ 506][0]= 0.277867319058624;
    Leb_Grid_XYZW[ 506][1]= 0.957102074310073;
    Leb_Grid_XYZW[ 506][2]=-0.082130215819325;
    Leb_Grid_XYZW[ 506][3]= 0.001555213603397;

    Leb_Grid_XYZW[ 507][0]=-0.277867319058624;
    Leb_Grid_XYZW[ 507][1]= 0.957102074310073;
    Leb_Grid_XYZW[ 507][2]=-0.082130215819325;
    Leb_Grid_XYZW[ 507][3]= 0.001555213603397;

    Leb_Grid_XYZW[ 508][0]= 0.277867319058624;
    Leb_Grid_XYZW[ 508][1]=-0.957102074310073;
    Leb_Grid_XYZW[ 508][2]=-0.082130215819325;
    Leb_Grid_XYZW[ 508][3]= 0.001555213603397;

    Leb_Grid_XYZW[ 509][0]=-0.277867319058624;
    Leb_Grid_XYZW[ 509][1]=-0.957102074310073;
    Leb_Grid_XYZW[ 509][2]=-0.082130215819325;
    Leb_Grid_XYZW[ 509][3]= 0.001555213603397;

    Leb_Grid_XYZW[ 510][0]= 0.082130215819325;
    Leb_Grid_XYZW[ 510][1]= 0.277867319058624;
    Leb_Grid_XYZW[ 510][2]= 0.957102074310073;
    Leb_Grid_XYZW[ 510][3]= 0.001555213603397;

    Leb_Grid_XYZW[ 511][0]=-0.082130215819325;
    Leb_Grid_XYZW[ 511][1]= 0.277867319058624;
    Leb_Grid_XYZW[ 511][2]= 0.957102074310073;
    Leb_Grid_XYZW[ 511][3]= 0.001555213603397;

    Leb_Grid_XYZW[ 512][0]= 0.082130215819325;
    Leb_Grid_XYZW[ 512][1]=-0.277867319058624;
    Leb_Grid_XYZW[ 512][2]= 0.957102074310073;
    Leb_Grid_XYZW[ 512][3]= 0.001555213603397;

    Leb_Grid_XYZW[ 513][0]=-0.082130215819325;
    Leb_Grid_XYZW[ 513][1]=-0.277867319058624;
    Leb_Grid_XYZW[ 513][2]= 0.957102074310073;
    Leb_Grid_XYZW[ 513][3]= 0.001555213603397;

    Leb_Grid_XYZW[ 514][0]= 0.082130215819325;
    Leb_Grid_XYZW[ 514][1]= 0.277867319058624;
    Leb_Grid_XYZW[ 514][2]=-0.957102074310073;
    Leb_Grid_XYZW[ 514][3]= 0.001555213603397;

    Leb_Grid_XYZW[ 515][0]=-0.082130215819325;
    Leb_Grid_XYZW[ 515][1]= 0.277867319058624;
    Leb_Grid_XYZW[ 515][2]=-0.957102074310073;
    Leb_Grid_XYZW[ 515][3]= 0.001555213603397;

    Leb_Grid_XYZW[ 516][0]= 0.082130215819325;
    Leb_Grid_XYZW[ 516][1]=-0.277867319058624;
    Leb_Grid_XYZW[ 516][2]=-0.957102074310073;
    Leb_Grid_XYZW[ 516][3]= 0.001555213603397;

    Leb_Grid_XYZW[ 517][0]=-0.082130215819325;
    Leb_Grid_XYZW[ 517][1]=-0.277867319058624;
    Leb_Grid_XYZW[ 517][2]=-0.957102074310073;
    Leb_Grid_XYZW[ 517][3]= 0.001555213603397;

    Leb_Grid_XYZW[ 518][0]= 0.082130215819325;
    Leb_Grid_XYZW[ 518][1]= 0.957102074310073;
    Leb_Grid_XYZW[ 518][2]= 0.277867319058624;
    Leb_Grid_XYZW[ 518][3]= 0.001555213603397;

    Leb_Grid_XYZW[ 519][0]=-0.082130215819325;
    Leb_Grid_XYZW[ 519][1]= 0.957102074310073;
    Leb_Grid_XYZW[ 519][2]= 0.277867319058624;
    Leb_Grid_XYZW[ 519][3]= 0.001555213603397;

    Leb_Grid_XYZW[ 520][0]= 0.082130215819325;
    Leb_Grid_XYZW[ 520][1]=-0.957102074310073;
    Leb_Grid_XYZW[ 520][2]= 0.277867319058624;
    Leb_Grid_XYZW[ 520][3]= 0.001555213603397;

    Leb_Grid_XYZW[ 521][0]=-0.082130215819325;
    Leb_Grid_XYZW[ 521][1]=-0.957102074310073;
    Leb_Grid_XYZW[ 521][2]= 0.277867319058624;
    Leb_Grid_XYZW[ 521][3]= 0.001555213603397;

    Leb_Grid_XYZW[ 522][0]= 0.082130215819325;
    Leb_Grid_XYZW[ 522][1]= 0.957102074310073;
    Leb_Grid_XYZW[ 522][2]=-0.277867319058624;
    Leb_Grid_XYZW[ 522][3]= 0.001555213603397;

    Leb_Grid_XYZW[ 523][0]=-0.082130215819325;
    Leb_Grid_XYZW[ 523][1]= 0.957102074310073;
    Leb_Grid_XYZW[ 523][2]=-0.277867319058624;
    Leb_Grid_XYZW[ 523][3]= 0.001555213603397;

    Leb_Grid_XYZW[ 524][0]= 0.082130215819325;
    Leb_Grid_XYZW[ 524][1]=-0.957102074310073;
    Leb_Grid_XYZW[ 524][2]=-0.277867319058624;
    Leb_Grid_XYZW[ 524][3]= 0.001555213603397;

    Leb_Grid_XYZW[ 525][0]=-0.082130215819325;
    Leb_Grid_XYZW[ 525][1]=-0.957102074310073;
    Leb_Grid_XYZW[ 525][2]=-0.277867319058624;
    Leb_Grid_XYZW[ 525][3]= 0.001555213603397;

    Leb_Grid_XYZW[ 526][0]= 0.957102074310073;
    Leb_Grid_XYZW[ 526][1]= 0.277867319058624;
    Leb_Grid_XYZW[ 526][2]= 0.082130215819325;
    Leb_Grid_XYZW[ 526][3]= 0.001555213603397;

    Leb_Grid_XYZW[ 527][0]=-0.957102074310073;
    Leb_Grid_XYZW[ 527][1]= 0.277867319058624;
    Leb_Grid_XYZW[ 527][2]= 0.082130215819325;
    Leb_Grid_XYZW[ 527][3]= 0.001555213603397;

    Leb_Grid_XYZW[ 528][0]= 0.957102074310073;
    Leb_Grid_XYZW[ 528][1]=-0.277867319058624;
    Leb_Grid_XYZW[ 528][2]= 0.082130215819325;
    Leb_Grid_XYZW[ 528][3]= 0.001555213603397;

    Leb_Grid_XYZW[ 529][0]=-0.957102074310073;
    Leb_Grid_XYZW[ 529][1]=-0.277867319058624;
    Leb_Grid_XYZW[ 529][2]= 0.082130215819325;
    Leb_Grid_XYZW[ 529][3]= 0.001555213603397;

    Leb_Grid_XYZW[ 530][0]= 0.957102074310073;
    Leb_Grid_XYZW[ 530][1]= 0.277867319058624;
    Leb_Grid_XYZW[ 530][2]=-0.082130215819325;
    Leb_Grid_XYZW[ 530][3]= 0.001555213603397;

    Leb_Grid_XYZW[ 531][0]=-0.957102074310073;
    Leb_Grid_XYZW[ 531][1]= 0.277867319058624;
    Leb_Grid_XYZW[ 531][2]=-0.082130215819325;
    Leb_Grid_XYZW[ 531][3]= 0.001555213603397;

    Leb_Grid_XYZW[ 532][0]= 0.957102074310073;
    Leb_Grid_XYZW[ 532][1]=-0.277867319058624;
    Leb_Grid_XYZW[ 532][2]=-0.082130215819325;
    Leb_Grid_XYZW[ 532][3]= 0.001555213603397;

    Leb_Grid_XYZW[ 533][0]=-0.957102074310073;
    Leb_Grid_XYZW[ 533][1]=-0.277867319058624;
    Leb_Grid_XYZW[ 533][2]=-0.082130215819325;
    Leb_Grid_XYZW[ 533][3]= 0.001555213603397;

    Leb_Grid_XYZW[ 534][0]= 0.957102074310073;
    Leb_Grid_XYZW[ 534][1]= 0.082130215819325;
    Leb_Grid_XYZW[ 534][2]= 0.277867319058624;
    Leb_Grid_XYZW[ 534][3]= 0.001555213603397;

    Leb_Grid_XYZW[ 535][0]=-0.957102074310073;
    Leb_Grid_XYZW[ 535][1]= 0.082130215819325;
    Leb_Grid_XYZW[ 535][2]= 0.277867319058624;
    Leb_Grid_XYZW[ 535][3]= 0.001555213603397;

    Leb_Grid_XYZW[ 536][0]= 0.957102074310073;
    Leb_Grid_XYZW[ 536][1]=-0.082130215819325;
    Leb_Grid_XYZW[ 536][2]= 0.277867319058624;
    Leb_Grid_XYZW[ 536][3]= 0.001555213603397;

    Leb_Grid_XYZW[ 537][0]=-0.957102074310073;
    Leb_Grid_XYZW[ 537][1]=-0.082130215819325;
    Leb_Grid_XYZW[ 537][2]= 0.277867319058624;
    Leb_Grid_XYZW[ 537][3]= 0.001555213603397;

    Leb_Grid_XYZW[ 538][0]= 0.957102074310073;
    Leb_Grid_XYZW[ 538][1]= 0.082130215819325;
    Leb_Grid_XYZW[ 538][2]=-0.277867319058624;
    Leb_Grid_XYZW[ 538][3]= 0.001555213603397;

    Leb_Grid_XYZW[ 539][0]=-0.957102074310073;
    Leb_Grid_XYZW[ 539][1]= 0.082130215819325;
    Leb_Grid_XYZW[ 539][2]=-0.277867319058624;
    Leb_Grid_XYZW[ 539][3]= 0.001555213603397;

    Leb_Grid_XYZW[ 540][0]= 0.957102074310073;
    Leb_Grid_XYZW[ 540][1]=-0.082130215819325;
    Leb_Grid_XYZW[ 540][2]=-0.277867319058624;
    Leb_Grid_XYZW[ 540][3]= 0.001555213603397;

    Leb_Grid_XYZW[ 541][0]=-0.957102074310073;
    Leb_Grid_XYZW[ 541][1]=-0.082130215819325;
    Leb_Grid_XYZW[ 541][2]=-0.277867319058624;
    Leb_Grid_XYZW[ 541][3]= 0.001555213603397;

    Leb_Grid_XYZW[ 542][0]= 0.503356427107512;
    Leb_Grid_XYZW[ 542][1]= 0.089992058420749;
    Leb_Grid_XYZW[ 542][2]= 0.859379855890721;
    Leb_Grid_XYZW[ 542][3]= 0.001802239128009;

    Leb_Grid_XYZW[ 543][0]=-0.503356427107512;
    Leb_Grid_XYZW[ 543][1]= 0.089992058420749;
    Leb_Grid_XYZW[ 543][2]= 0.859379855890721;
    Leb_Grid_XYZW[ 543][3]= 0.001802239128009;

    Leb_Grid_XYZW[ 544][0]= 0.503356427107512;
    Leb_Grid_XYZW[ 544][1]=-0.089992058420749;
    Leb_Grid_XYZW[ 544][2]= 0.859379855890721;
    Leb_Grid_XYZW[ 544][3]= 0.001802239128009;

    Leb_Grid_XYZW[ 545][0]=-0.503356427107512;
    Leb_Grid_XYZW[ 545][1]=-0.089992058420749;
    Leb_Grid_XYZW[ 545][2]= 0.859379855890721;
    Leb_Grid_XYZW[ 545][3]= 0.001802239128009;

    Leb_Grid_XYZW[ 546][0]= 0.503356427107512;
    Leb_Grid_XYZW[ 546][1]= 0.089992058420749;
    Leb_Grid_XYZW[ 546][2]=-0.859379855890721;
    Leb_Grid_XYZW[ 546][3]= 0.001802239128009;

    Leb_Grid_XYZW[ 547][0]=-0.503356427107512;
    Leb_Grid_XYZW[ 547][1]= 0.089992058420749;
    Leb_Grid_XYZW[ 547][2]=-0.859379855890721;
    Leb_Grid_XYZW[ 547][3]= 0.001802239128009;

    Leb_Grid_XYZW[ 548][0]= 0.503356427107512;
    Leb_Grid_XYZW[ 548][1]=-0.089992058420749;
    Leb_Grid_XYZW[ 548][2]=-0.859379855890721;
    Leb_Grid_XYZW[ 548][3]= 0.001802239128009;

    Leb_Grid_XYZW[ 549][0]=-0.503356427107512;
    Leb_Grid_XYZW[ 549][1]=-0.089992058420749;
    Leb_Grid_XYZW[ 549][2]=-0.859379855890721;
    Leb_Grid_XYZW[ 549][3]= 0.001802239128009;

    Leb_Grid_XYZW[ 550][0]= 0.503356427107512;
    Leb_Grid_XYZW[ 550][1]= 0.859379855890721;
    Leb_Grid_XYZW[ 550][2]= 0.089992058420749;
    Leb_Grid_XYZW[ 550][3]= 0.001802239128009;

    Leb_Grid_XYZW[ 551][0]=-0.503356427107512;
    Leb_Grid_XYZW[ 551][1]= 0.859379855890721;
    Leb_Grid_XYZW[ 551][2]= 0.089992058420749;
    Leb_Grid_XYZW[ 551][3]= 0.001802239128009;

    Leb_Grid_XYZW[ 552][0]= 0.503356427107512;
    Leb_Grid_XYZW[ 552][1]=-0.859379855890721;
    Leb_Grid_XYZW[ 552][2]= 0.089992058420749;
    Leb_Grid_XYZW[ 552][3]= 0.001802239128009;

    Leb_Grid_XYZW[ 553][0]=-0.503356427107512;
    Leb_Grid_XYZW[ 553][1]=-0.859379855890721;
    Leb_Grid_XYZW[ 553][2]= 0.089992058420749;
    Leb_Grid_XYZW[ 553][3]= 0.001802239128009;

    Leb_Grid_XYZW[ 554][0]= 0.503356427107512;
    Leb_Grid_XYZW[ 554][1]= 0.859379855890721;
    Leb_Grid_XYZW[ 554][2]=-0.089992058420749;
    Leb_Grid_XYZW[ 554][3]= 0.001802239128009;

    Leb_Grid_XYZW[ 555][0]=-0.503356427107512;
    Leb_Grid_XYZW[ 555][1]= 0.859379855890721;
    Leb_Grid_XYZW[ 555][2]=-0.089992058420749;
    Leb_Grid_XYZW[ 555][3]= 0.001802239128009;

    Leb_Grid_XYZW[ 556][0]= 0.503356427107512;
    Leb_Grid_XYZW[ 556][1]=-0.859379855890721;
    Leb_Grid_XYZW[ 556][2]=-0.089992058420749;
    Leb_Grid_XYZW[ 556][3]= 0.001802239128009;

    Leb_Grid_XYZW[ 557][0]=-0.503356427107512;
    Leb_Grid_XYZW[ 557][1]=-0.859379855890721;
    Leb_Grid_XYZW[ 557][2]=-0.089992058420749;
    Leb_Grid_XYZW[ 557][3]= 0.001802239128009;

    Leb_Grid_XYZW[ 558][0]= 0.089992058420749;
    Leb_Grid_XYZW[ 558][1]= 0.503356427107512;
    Leb_Grid_XYZW[ 558][2]= 0.859379855890721;
    Leb_Grid_XYZW[ 558][3]= 0.001802239128009;

    Leb_Grid_XYZW[ 559][0]=-0.089992058420749;
    Leb_Grid_XYZW[ 559][1]= 0.503356427107512;
    Leb_Grid_XYZW[ 559][2]= 0.859379855890721;
    Leb_Grid_XYZW[ 559][3]= 0.001802239128009;

    Leb_Grid_XYZW[ 560][0]= 0.089992058420749;
    Leb_Grid_XYZW[ 560][1]=-0.503356427107512;
    Leb_Grid_XYZW[ 560][2]= 0.859379855890721;
    Leb_Grid_XYZW[ 560][3]= 0.001802239128009;

    Leb_Grid_XYZW[ 561][0]=-0.089992058420749;
    Leb_Grid_XYZW[ 561][1]=-0.503356427107512;
    Leb_Grid_XYZW[ 561][2]= 0.859379855890721;
    Leb_Grid_XYZW[ 561][3]= 0.001802239128009;

    Leb_Grid_XYZW[ 562][0]= 0.089992058420749;
    Leb_Grid_XYZW[ 562][1]= 0.503356427107512;
    Leb_Grid_XYZW[ 562][2]=-0.859379855890721;
    Leb_Grid_XYZW[ 562][3]= 0.001802239128009;

    Leb_Grid_XYZW[ 563][0]=-0.089992058420749;
    Leb_Grid_XYZW[ 563][1]= 0.503356427107512;
    Leb_Grid_XYZW[ 563][2]=-0.859379855890721;
    Leb_Grid_XYZW[ 563][3]= 0.001802239128009;

    Leb_Grid_XYZW[ 564][0]= 0.089992058420749;
    Leb_Grid_XYZW[ 564][1]=-0.503356427107512;
    Leb_Grid_XYZW[ 564][2]=-0.859379855890721;
    Leb_Grid_XYZW[ 564][3]= 0.001802239128009;

    Leb_Grid_XYZW[ 565][0]=-0.089992058420749;
    Leb_Grid_XYZW[ 565][1]=-0.503356427107512;
    Leb_Grid_XYZW[ 565][2]=-0.859379855890721;
    Leb_Grid_XYZW[ 565][3]= 0.001802239128009;

    Leb_Grid_XYZW[ 566][0]= 0.089992058420749;
    Leb_Grid_XYZW[ 566][1]= 0.859379855890721;
    Leb_Grid_XYZW[ 566][2]= 0.503356427107512;
    Leb_Grid_XYZW[ 566][3]= 0.001802239128009;

    Leb_Grid_XYZW[ 567][0]=-0.089992058420749;
    Leb_Grid_XYZW[ 567][1]= 0.859379855890721;
    Leb_Grid_XYZW[ 567][2]= 0.503356427107512;
    Leb_Grid_XYZW[ 567][3]= 0.001802239128009;

    Leb_Grid_XYZW[ 568][0]= 0.089992058420749;
    Leb_Grid_XYZW[ 568][1]=-0.859379855890721;
    Leb_Grid_XYZW[ 568][2]= 0.503356427107512;
    Leb_Grid_XYZW[ 568][3]= 0.001802239128009;

    Leb_Grid_XYZW[ 569][0]=-0.089992058420749;
    Leb_Grid_XYZW[ 569][1]=-0.859379855890721;
    Leb_Grid_XYZW[ 569][2]= 0.503356427107512;
    Leb_Grid_XYZW[ 569][3]= 0.001802239128009;

    Leb_Grid_XYZW[ 570][0]= 0.089992058420749;
    Leb_Grid_XYZW[ 570][1]= 0.859379855890721;
    Leb_Grid_XYZW[ 570][2]=-0.503356427107512;
    Leb_Grid_XYZW[ 570][3]= 0.001802239128009;

    Leb_Grid_XYZW[ 571][0]=-0.089992058420749;
    Leb_Grid_XYZW[ 571][1]= 0.859379855890721;
    Leb_Grid_XYZW[ 571][2]=-0.503356427107512;
    Leb_Grid_XYZW[ 571][3]= 0.001802239128009;

    Leb_Grid_XYZW[ 572][0]= 0.089992058420749;
    Leb_Grid_XYZW[ 572][1]=-0.859379855890721;
    Leb_Grid_XYZW[ 572][2]=-0.503356427107512;
    Leb_Grid_XYZW[ 572][3]= 0.001802239128009;

    Leb_Grid_XYZW[ 573][0]=-0.089992058420749;
    Leb_Grid_XYZW[ 573][1]=-0.859379855890721;
    Leb_Grid_XYZW[ 573][2]=-0.503356427107512;
    Leb_Grid_XYZW[ 573][3]= 0.001802239128009;

    Leb_Grid_XYZW[ 574][0]= 0.859379855890721;
    Leb_Grid_XYZW[ 574][1]= 0.503356427107512;
    Leb_Grid_XYZW[ 574][2]= 0.089992058420749;
    Leb_Grid_XYZW[ 574][3]= 0.001802239128009;

    Leb_Grid_XYZW[ 575][0]=-0.859379855890721;
    Leb_Grid_XYZW[ 575][1]= 0.503356427107512;
    Leb_Grid_XYZW[ 575][2]= 0.089992058420749;
    Leb_Grid_XYZW[ 575][3]= 0.001802239128009;

    Leb_Grid_XYZW[ 576][0]= 0.859379855890721;
    Leb_Grid_XYZW[ 576][1]=-0.503356427107512;
    Leb_Grid_XYZW[ 576][2]= 0.089992058420749;
    Leb_Grid_XYZW[ 576][3]= 0.001802239128009;

    Leb_Grid_XYZW[ 577][0]=-0.859379855890721;
    Leb_Grid_XYZW[ 577][1]=-0.503356427107512;
    Leb_Grid_XYZW[ 577][2]= 0.089992058420749;
    Leb_Grid_XYZW[ 577][3]= 0.001802239128009;

    Leb_Grid_XYZW[ 578][0]= 0.859379855890721;
    Leb_Grid_XYZW[ 578][1]= 0.503356427107512;
    Leb_Grid_XYZW[ 578][2]=-0.089992058420749;
    Leb_Grid_XYZW[ 578][3]= 0.001802239128009;

    Leb_Grid_XYZW[ 579][0]=-0.859379855890721;
    Leb_Grid_XYZW[ 579][1]= 0.503356427107512;
    Leb_Grid_XYZW[ 579][2]=-0.089992058420749;
    Leb_Grid_XYZW[ 579][3]= 0.001802239128009;

    Leb_Grid_XYZW[ 580][0]= 0.859379855890721;
    Leb_Grid_XYZW[ 580][1]=-0.503356427107512;
    Leb_Grid_XYZW[ 580][2]=-0.089992058420749;
    Leb_Grid_XYZW[ 580][3]= 0.001802239128009;

    Leb_Grid_XYZW[ 581][0]=-0.859379855890721;
    Leb_Grid_XYZW[ 581][1]=-0.503356427107512;
    Leb_Grid_XYZW[ 581][2]=-0.089992058420749;
    Leb_Grid_XYZW[ 581][3]= 0.001802239128009;

    Leb_Grid_XYZW[ 582][0]= 0.859379855890721;
    Leb_Grid_XYZW[ 582][1]= 0.089992058420749;
    Leb_Grid_XYZW[ 582][2]= 0.503356427107512;
    Leb_Grid_XYZW[ 582][3]= 0.001802239128009;

    Leb_Grid_XYZW[ 583][0]=-0.859379855890721;
    Leb_Grid_XYZW[ 583][1]= 0.089992058420749;
    Leb_Grid_XYZW[ 583][2]= 0.503356427107512;
    Leb_Grid_XYZW[ 583][3]= 0.001802239128009;

    Leb_Grid_XYZW[ 584][0]= 0.859379855890721;
    Leb_Grid_XYZW[ 584][1]=-0.089992058420749;
    Leb_Grid_XYZW[ 584][2]= 0.503356427107512;
    Leb_Grid_XYZW[ 584][3]= 0.001802239128009;

    Leb_Grid_XYZW[ 585][0]=-0.859379855890721;
    Leb_Grid_XYZW[ 585][1]=-0.089992058420749;
    Leb_Grid_XYZW[ 585][2]= 0.503356427107512;
    Leb_Grid_XYZW[ 585][3]= 0.001802239128009;

    Leb_Grid_XYZW[ 586][0]= 0.859379855890721;
    Leb_Grid_XYZW[ 586][1]= 0.089992058420749;
    Leb_Grid_XYZW[ 586][2]=-0.503356427107512;
    Leb_Grid_XYZW[ 586][3]= 0.001802239128009;

    Leb_Grid_XYZW[ 587][0]=-0.859379855890721;
    Leb_Grid_XYZW[ 587][1]= 0.089992058420749;
    Leb_Grid_XYZW[ 587][2]=-0.503356427107512;
    Leb_Grid_XYZW[ 587][3]= 0.001802239128009;

    Leb_Grid_XYZW[ 588][0]= 0.859379855890721;
    Leb_Grid_XYZW[ 588][1]=-0.089992058420749;
    Leb_Grid_XYZW[ 588][2]=-0.503356427107512;
    Leb_Grid_XYZW[ 588][3]= 0.001802239128009;

    Leb_Grid_XYZW[ 589][0]=-0.859379855890721;
    Leb_Grid_XYZW[ 589][1]=-0.089992058420749;
    Leb_Grid_XYZW[ 589][2]=-0.503356427107512;
    Leb_Grid_XYZW[ 589][3]= 0.001802239128009;

  }

  /* 1202 */

  else if (Np==1202){

    Leb_Grid_XYZW[    0][0] =  1.000000000000000;
    Leb_Grid_XYZW[    0][1] =  0.000000000000000;
    Leb_Grid_XYZW[    0][2] =  0.000000000000000;
    Leb_Grid_XYZW[    0][3] =  0.000110518923327;

    Leb_Grid_XYZW[    1][0] = -1.000000000000000;
    Leb_Grid_XYZW[    1][1] =  0.000000000000000;
    Leb_Grid_XYZW[    1][2] =  0.000000000000000;
    Leb_Grid_XYZW[    1][3] =  0.000110518923327;

    Leb_Grid_XYZW[    2][0] =  0.000000000000000;
    Leb_Grid_XYZW[    2][1] =  1.000000000000000;
    Leb_Grid_XYZW[    2][2] =  0.000000000000000;
    Leb_Grid_XYZW[    2][3] =  0.000110518923327;

    Leb_Grid_XYZW[    3][0] =  0.000000000000000;
    Leb_Grid_XYZW[    3][1] = -1.000000000000000;
    Leb_Grid_XYZW[    3][2] =  0.000000000000000;
    Leb_Grid_XYZW[    3][3] =  0.000110518923327;

    Leb_Grid_XYZW[    4][0] =  0.000000000000000;
    Leb_Grid_XYZW[    4][1] =  0.000000000000000;
    Leb_Grid_XYZW[    4][2] =  1.000000000000000;
    Leb_Grid_XYZW[    4][3] =  0.000110518923327;

    Leb_Grid_XYZW[    5][0] =  0.000000000000000;
    Leb_Grid_XYZW[    5][1] =  0.000000000000000;
    Leb_Grid_XYZW[    5][2] = -1.000000000000000;
    Leb_Grid_XYZW[    5][3] =  0.000110518923327;

    Leb_Grid_XYZW[    6][0] =  0.000000000000000;
    Leb_Grid_XYZW[    6][1] =  0.707106781186548;
    Leb_Grid_XYZW[    6][2] =  0.707106781186547;
    Leb_Grid_XYZW[    6][3] =  0.000920523273809;

    Leb_Grid_XYZW[    7][0] =  0.000000000000000;
    Leb_Grid_XYZW[    7][1] =  0.707106781186548;
    Leb_Grid_XYZW[    7][2] = -0.707106781186547;
    Leb_Grid_XYZW[    7][3] =  0.000920523273809;

    Leb_Grid_XYZW[    8][0] =  0.000000000000000;
    Leb_Grid_XYZW[    8][1] = -0.707106781186548;
    Leb_Grid_XYZW[    8][2] =  0.707106781186547;
    Leb_Grid_XYZW[    8][3] =  0.000920523273809;

    Leb_Grid_XYZW[    9][0] =  0.000000000000000;
    Leb_Grid_XYZW[    9][1] = -0.707106781186548;
    Leb_Grid_XYZW[    9][2] = -0.707106781186547;
    Leb_Grid_XYZW[    9][3] =  0.000920523273809;

    Leb_Grid_XYZW[   10][0] =  0.707106781186548;
    Leb_Grid_XYZW[   10][1] =  0.000000000000000;
    Leb_Grid_XYZW[   10][2] =  0.707106781186547;
    Leb_Grid_XYZW[   10][3] =  0.000920523273809;

    Leb_Grid_XYZW[   11][0] =  0.707106781186548;
    Leb_Grid_XYZW[   11][1] =  0.000000000000000;
    Leb_Grid_XYZW[   11][2] = -0.707106781186547;
    Leb_Grid_XYZW[   11][3] =  0.000920523273809;

    Leb_Grid_XYZW[   12][0] = -0.707106781186548;
    Leb_Grid_XYZW[   12][1] =  0.000000000000000;
    Leb_Grid_XYZW[   12][2] =  0.707106781186547;
    Leb_Grid_XYZW[   12][3] =  0.000920523273809;

    Leb_Grid_XYZW[   13][0] = -0.707106781186548;
    Leb_Grid_XYZW[   13][1] =  0.000000000000000;
    Leb_Grid_XYZW[   13][2] = -0.707106781186547;
    Leb_Grid_XYZW[   13][3] =  0.000920523273809;

    Leb_Grid_XYZW[   14][0] =  0.707106781186548;
    Leb_Grid_XYZW[   14][1] =  0.707106781186547;
    Leb_Grid_XYZW[   14][2] =  0.000000000000000;
    Leb_Grid_XYZW[   14][3] =  0.000920523273809;

    Leb_Grid_XYZW[   15][0] =  0.707106781186548;
    Leb_Grid_XYZW[   15][1] = -0.707106781186547;
    Leb_Grid_XYZW[   15][2] =  0.000000000000000;
    Leb_Grid_XYZW[   15][3] =  0.000920523273809;

    Leb_Grid_XYZW[   16][0] = -0.707106781186547;
    Leb_Grid_XYZW[   16][1] =  0.707106781186548;
    Leb_Grid_XYZW[   16][2] =  0.000000000000000;
    Leb_Grid_XYZW[   16][3] =  0.000920523273809;

    Leb_Grid_XYZW[   17][0] = -0.707106781186547;
    Leb_Grid_XYZW[   17][1] = -0.707106781186548;
    Leb_Grid_XYZW[   17][2] =  0.000000000000000;
    Leb_Grid_XYZW[   17][3] =  0.000920523273809;

    Leb_Grid_XYZW[   18][0] =  0.577350269189626;
    Leb_Grid_XYZW[   18][1] =  0.577350269189626;
    Leb_Grid_XYZW[   18][2] =  0.577350269189626;
    Leb_Grid_XYZW[   18][3] =  0.000913315978644;

    Leb_Grid_XYZW[   19][0] =  0.577350269189626;
    Leb_Grid_XYZW[   19][1] =  0.577350269189626;
    Leb_Grid_XYZW[   19][2] = -0.577350269189626;
    Leb_Grid_XYZW[   19][3] =  0.000913315978644;

    Leb_Grid_XYZW[   20][0] =  0.577350269189626;
    Leb_Grid_XYZW[   20][1] = -0.577350269189626;
    Leb_Grid_XYZW[   20][2] =  0.577350269189626;
    Leb_Grid_XYZW[   20][3] =  0.000913315978644;

    Leb_Grid_XYZW[   21][0] =  0.577350269189626;
    Leb_Grid_XYZW[   21][1] = -0.577350269189626;
    Leb_Grid_XYZW[   21][2] = -0.577350269189626;
    Leb_Grid_XYZW[   21][3] =  0.000913315978644;

    Leb_Grid_XYZW[   22][0] = -0.577350269189626;
    Leb_Grid_XYZW[   22][1] =  0.577350269189626;
    Leb_Grid_XYZW[   22][2] =  0.577350269189626;
    Leb_Grid_XYZW[   22][3] =  0.000913315978644;

    Leb_Grid_XYZW[   23][0] = -0.577350269189626;
    Leb_Grid_XYZW[   23][1] =  0.577350269189626;
    Leb_Grid_XYZW[   23][2] = -0.577350269189626;
    Leb_Grid_XYZW[   23][3] =  0.000913315978644;

    Leb_Grid_XYZW[   24][0] = -0.577350269189626;
    Leb_Grid_XYZW[   24][1] = -0.577350269189626;
    Leb_Grid_XYZW[   24][2] =  0.577350269189626;
    Leb_Grid_XYZW[   24][3] =  0.000913315978644;

    Leb_Grid_XYZW[   25][0] = -0.577350269189626;
    Leb_Grid_XYZW[   25][1] = -0.577350269189626;
    Leb_Grid_XYZW[   25][2] = -0.577350269189626;
    Leb_Grid_XYZW[   25][3] =  0.000913315978644;

    Leb_Grid_XYZW[   26][0] =  0.037126364496571;
    Leb_Grid_XYZW[   26][1] =  0.037126364496571;
    Leb_Grid_XYZW[   26][2] =  0.998620681799919;
    Leb_Grid_XYZW[   26][3] =  0.000369042189802;

    Leb_Grid_XYZW[   27][0] =  0.037126364496570;
    Leb_Grid_XYZW[   27][1] =  0.037126364496570;
    Leb_Grid_XYZW[   27][2] = -0.998620681799919;
    Leb_Grid_XYZW[   27][3] =  0.000369042189802;

    Leb_Grid_XYZW[   28][0] =  0.037126364496571;
    Leb_Grid_XYZW[   28][1] = -0.037126364496571;
    Leb_Grid_XYZW[   28][2] =  0.998620681799919;
    Leb_Grid_XYZW[   28][3] =  0.000369042189802;

    Leb_Grid_XYZW[   29][0] =  0.037126364496570;
    Leb_Grid_XYZW[   29][1] = -0.037126364496570;
    Leb_Grid_XYZW[   29][2] = -0.998620681799919;
    Leb_Grid_XYZW[   29][3] =  0.000369042189802;

    Leb_Grid_XYZW[   30][0] = -0.037126364496571;
    Leb_Grid_XYZW[   30][1] =  0.037126364496571;
    Leb_Grid_XYZW[   30][2] =  0.998620681799919;
    Leb_Grid_XYZW[   30][3] =  0.000369042189802;

    Leb_Grid_XYZW[   31][0] = -0.037126364496570;
    Leb_Grid_XYZW[   31][1] =  0.037126364496570;
    Leb_Grid_XYZW[   31][2] = -0.998620681799919;
    Leb_Grid_XYZW[   31][3] =  0.000369042189802;

    Leb_Grid_XYZW[   32][0] = -0.037126364496571;
    Leb_Grid_XYZW[   32][1] = -0.037126364496571;
    Leb_Grid_XYZW[   32][2] =  0.998620681799919;
    Leb_Grid_XYZW[   32][3] =  0.000369042189802;

    Leb_Grid_XYZW[   33][0] = -0.037126364496570;
    Leb_Grid_XYZW[   33][1] = -0.037126364496570;
    Leb_Grid_XYZW[   33][2] = -0.998620681799919;
    Leb_Grid_XYZW[   33][3] =  0.000369042189802;

    Leb_Grid_XYZW[   34][0] =  0.037126364496571;
    Leb_Grid_XYZW[   34][1] =  0.998620681799919;
    Leb_Grid_XYZW[   34][2] =  0.037126364496571;
    Leb_Grid_XYZW[   34][3] =  0.000369042189802;

    Leb_Grid_XYZW[   35][0] =  0.037126364496571;
    Leb_Grid_XYZW[   35][1] = -0.998620681799919;
    Leb_Grid_XYZW[   35][2] =  0.037126364496571;
    Leb_Grid_XYZW[   35][3] =  0.000369042189802;

    Leb_Grid_XYZW[   36][0] =  0.037126364496571;
    Leb_Grid_XYZW[   36][1] =  0.998620681799919;
    Leb_Grid_XYZW[   36][2] = -0.037126364496571;
    Leb_Grid_XYZW[   36][3] =  0.000369042189802;

    Leb_Grid_XYZW[   37][0] =  0.037126364496571;
    Leb_Grid_XYZW[   37][1] = -0.998620681799919;
    Leb_Grid_XYZW[   37][2] = -0.037126364496571;
    Leb_Grid_XYZW[   37][3] =  0.000369042189802;

    Leb_Grid_XYZW[   38][0] = -0.037126364496571;
    Leb_Grid_XYZW[   38][1] =  0.998620681799919;
    Leb_Grid_XYZW[   38][2] =  0.037126364496571;
    Leb_Grid_XYZW[   38][3] =  0.000369042189802;

    Leb_Grid_XYZW[   39][0] = -0.037126364496571;
    Leb_Grid_XYZW[   39][1] = -0.998620681799919;
    Leb_Grid_XYZW[   39][2] =  0.037126364496571;
    Leb_Grid_XYZW[   39][3] =  0.000369042189802;

    Leb_Grid_XYZW[   40][0] = -0.037126364496571;
    Leb_Grid_XYZW[   40][1] =  0.998620681799919;
    Leb_Grid_XYZW[   40][2] = -0.037126364496571;
    Leb_Grid_XYZW[   40][3] =  0.000369042189802;

    Leb_Grid_XYZW[   41][0] = -0.037126364496571;
    Leb_Grid_XYZW[   41][1] = -0.998620681799919;
    Leb_Grid_XYZW[   41][2] = -0.037126364496571;
    Leb_Grid_XYZW[   41][3] =  0.000369042189802;

    Leb_Grid_XYZW[   42][0] =  0.998620681799919;
    Leb_Grid_XYZW[   42][1] =  0.037126364496570;
    Leb_Grid_XYZW[   42][2] =  0.037126364496571;
    Leb_Grid_XYZW[   42][3] =  0.000369042189802;

    Leb_Grid_XYZW[   43][0] = -0.998620681799919;
    Leb_Grid_XYZW[   43][1] =  0.037126364496570;
    Leb_Grid_XYZW[   43][2] =  0.037126364496571;
    Leb_Grid_XYZW[   43][3] =  0.000369042189802;

    Leb_Grid_XYZW[   44][0] =  0.998620681799919;
    Leb_Grid_XYZW[   44][1] =  0.037126364496570;
    Leb_Grid_XYZW[   44][2] = -0.037126364496571;
    Leb_Grid_XYZW[   44][3] =  0.000369042189802;

    Leb_Grid_XYZW[   45][0] = -0.998620681799919;
    Leb_Grid_XYZW[   45][1] =  0.037126364496570;
    Leb_Grid_XYZW[   45][2] = -0.037126364496571;
    Leb_Grid_XYZW[   45][3] =  0.000369042189802;

    Leb_Grid_XYZW[   46][0] =  0.998620681799919;
    Leb_Grid_XYZW[   46][1] = -0.037126364496570;
    Leb_Grid_XYZW[   46][2] =  0.037126364496571;
    Leb_Grid_XYZW[   46][3] =  0.000369042189802;

    Leb_Grid_XYZW[   47][0] = -0.998620681799919;
    Leb_Grid_XYZW[   47][1] = -0.037126364496570;
    Leb_Grid_XYZW[   47][2] =  0.037126364496571;
    Leb_Grid_XYZW[   47][3] =  0.000369042189802;

    Leb_Grid_XYZW[   48][0] =  0.998620681799919;
    Leb_Grid_XYZW[   48][1] = -0.037126364496570;
    Leb_Grid_XYZW[   48][2] = -0.037126364496571;
    Leb_Grid_XYZW[   48][3] =  0.000369042189802;

    Leb_Grid_XYZW[   49][0] = -0.998620681799919;
    Leb_Grid_XYZW[   49][1] = -0.037126364496570;
    Leb_Grid_XYZW[   49][2] = -0.037126364496571;
    Leb_Grid_XYZW[   49][3] =  0.000369042189802;

    Leb_Grid_XYZW[   50][0] =  0.091400604122622;
    Leb_Grid_XYZW[   50][1] =  0.091400604122622;
    Leb_Grid_XYZW[   50][2] =  0.991610739722014;
    Leb_Grid_XYZW[   50][3] =  0.000560399092868;

    Leb_Grid_XYZW[   51][0] =  0.091400604122622;
    Leb_Grid_XYZW[   51][1] =  0.091400604122622;
    Leb_Grid_XYZW[   51][2] = -0.991610739722014;
    Leb_Grid_XYZW[   51][3] =  0.000560399092868;

    Leb_Grid_XYZW[   52][0] =  0.091400604122622;
    Leb_Grid_XYZW[   52][1] = -0.091400604122622;
    Leb_Grid_XYZW[   52][2] =  0.991610739722014;
    Leb_Grid_XYZW[   52][3] =  0.000560399092868;

    Leb_Grid_XYZW[   53][0] =  0.091400604122622;
    Leb_Grid_XYZW[   53][1] = -0.091400604122622;
    Leb_Grid_XYZW[   53][2] = -0.991610739722014;
    Leb_Grid_XYZW[   53][3] =  0.000560399092868;

    Leb_Grid_XYZW[   54][0] = -0.091400604122622;
    Leb_Grid_XYZW[   54][1] =  0.091400604122622;
    Leb_Grid_XYZW[   54][2] =  0.991610739722014;
    Leb_Grid_XYZW[   54][3] =  0.000560399092868;

    Leb_Grid_XYZW[   55][0] = -0.091400604122622;
    Leb_Grid_XYZW[   55][1] =  0.091400604122622;
    Leb_Grid_XYZW[   55][2] = -0.991610739722014;
    Leb_Grid_XYZW[   55][3] =  0.000560399092868;

    Leb_Grid_XYZW[   56][0] = -0.091400604122622;
    Leb_Grid_XYZW[   56][1] = -0.091400604122622;
    Leb_Grid_XYZW[   56][2] =  0.991610739722014;
    Leb_Grid_XYZW[   56][3] =  0.000560399092868;

    Leb_Grid_XYZW[   57][0] = -0.091400604122622;
    Leb_Grid_XYZW[   57][1] = -0.091400604122622;
    Leb_Grid_XYZW[   57][2] = -0.991610739722014;
    Leb_Grid_XYZW[   57][3] =  0.000560399092868;

    Leb_Grid_XYZW[   58][0] =  0.091400604122623;
    Leb_Grid_XYZW[   58][1] =  0.991610739722014;
    Leb_Grid_XYZW[   58][2] =  0.091400604122622;
    Leb_Grid_XYZW[   58][3] =  0.000560399092868;

    Leb_Grid_XYZW[   59][0] =  0.091400604122623;
    Leb_Grid_XYZW[   59][1] = -0.991610739722014;
    Leb_Grid_XYZW[   59][2] =  0.091400604122622;
    Leb_Grid_XYZW[   59][3] =  0.000560399092868;

    Leb_Grid_XYZW[   60][0] =  0.091400604122623;
    Leb_Grid_XYZW[   60][1] =  0.991610739722014;
    Leb_Grid_XYZW[   60][2] = -0.091400604122622;
    Leb_Grid_XYZW[   60][3] =  0.000560399092868;

    Leb_Grid_XYZW[   61][0] =  0.091400604122623;
    Leb_Grid_XYZW[   61][1] = -0.991610739722014;
    Leb_Grid_XYZW[   61][2] = -0.091400604122622;
    Leb_Grid_XYZW[   61][3] =  0.000560399092868;

    Leb_Grid_XYZW[   62][0] = -0.091400604122622;
    Leb_Grid_XYZW[   62][1] =  0.991610739722014;
    Leb_Grid_XYZW[   62][2] =  0.091400604122622;
    Leb_Grid_XYZW[   62][3] =  0.000560399092868;

    Leb_Grid_XYZW[   63][0] = -0.091400604122622;
    Leb_Grid_XYZW[   63][1] = -0.991610739722014;
    Leb_Grid_XYZW[   63][2] =  0.091400604122622;
    Leb_Grid_XYZW[   63][3] =  0.000560399092868;

    Leb_Grid_XYZW[   64][0] = -0.091400604122622;
    Leb_Grid_XYZW[   64][1] =  0.991610739722014;
    Leb_Grid_XYZW[   64][2] = -0.091400604122622;
    Leb_Grid_XYZW[   64][3] =  0.000560399092868;

    Leb_Grid_XYZW[   65][0] = -0.091400604122622;
    Leb_Grid_XYZW[   65][1] = -0.991610739722014;
    Leb_Grid_XYZW[   65][2] = -0.091400604122622;
    Leb_Grid_XYZW[   65][3] =  0.000560399092868;

    Leb_Grid_XYZW[   66][0] =  0.991610739722014;
    Leb_Grid_XYZW[   66][1] =  0.091400604122622;
    Leb_Grid_XYZW[   66][2] =  0.091400604122622;
    Leb_Grid_XYZW[   66][3] =  0.000560399092868;

    Leb_Grid_XYZW[   67][0] = -0.991610739722014;
    Leb_Grid_XYZW[   67][1] =  0.091400604122621;
    Leb_Grid_XYZW[   67][2] =  0.091400604122622;
    Leb_Grid_XYZW[   67][3] =  0.000560399092868;

    Leb_Grid_XYZW[   68][0] =  0.991610739722014;
    Leb_Grid_XYZW[   68][1] =  0.091400604122622;
    Leb_Grid_XYZW[   68][2] = -0.091400604122622;
    Leb_Grid_XYZW[   68][3] =  0.000560399092868;

    Leb_Grid_XYZW[   69][0] = -0.991610739722014;
    Leb_Grid_XYZW[   69][1] =  0.091400604122621;
    Leb_Grid_XYZW[   69][2] = -0.091400604122622;
    Leb_Grid_XYZW[   69][3] =  0.000560399092868;

    Leb_Grid_XYZW[   70][0] =  0.991610739722014;
    Leb_Grid_XYZW[   70][1] = -0.091400604122622;
    Leb_Grid_XYZW[   70][2] =  0.091400604122622;
    Leb_Grid_XYZW[   70][3] =  0.000560399092868;

    Leb_Grid_XYZW[   71][0] = -0.991610739722014;
    Leb_Grid_XYZW[   71][1] = -0.091400604122621;
    Leb_Grid_XYZW[   71][2] =  0.091400604122622;
    Leb_Grid_XYZW[   71][3] =  0.000560399092868;

    Leb_Grid_XYZW[   72][0] =  0.991610739722014;
    Leb_Grid_XYZW[   72][1] = -0.091400604122622;
    Leb_Grid_XYZW[   72][2] = -0.091400604122622;
    Leb_Grid_XYZW[   72][3] =  0.000560399092868;

    Leb_Grid_XYZW[   73][0] = -0.991610739722014;
    Leb_Grid_XYZW[   73][1] = -0.091400604122621;
    Leb_Grid_XYZW[   73][2] = -0.091400604122622;
    Leb_Grid_XYZW[   73][3] =  0.000560399092868;

    Leb_Grid_XYZW[   74][0] =  0.153107785246991;
    Leb_Grid_XYZW[   74][1] =  0.153107785246991;
    Leb_Grid_XYZW[   74][2] =  0.976276606394685;
    Leb_Grid_XYZW[   74][3] =  0.000686529762928;

    Leb_Grid_XYZW[   75][0] =  0.153107785246990;
    Leb_Grid_XYZW[   75][1] =  0.153107785246990;
    Leb_Grid_XYZW[   75][2] = -0.976276606394685;
    Leb_Grid_XYZW[   75][3] =  0.000686529762928;

    Leb_Grid_XYZW[   76][0] =  0.153107785246991;
    Leb_Grid_XYZW[   76][1] = -0.153107785246991;
    Leb_Grid_XYZW[   76][2] =  0.976276606394685;
    Leb_Grid_XYZW[   76][3] =  0.000686529762928;

    Leb_Grid_XYZW[   77][0] =  0.153107785246990;
    Leb_Grid_XYZW[   77][1] = -0.153107785246990;
    Leb_Grid_XYZW[   77][2] = -0.976276606394685;
    Leb_Grid_XYZW[   77][3] =  0.000686529762928;

    Leb_Grid_XYZW[   78][0] = -0.153107785246991;
    Leb_Grid_XYZW[   78][1] =  0.153107785246991;
    Leb_Grid_XYZW[   78][2] =  0.976276606394685;
    Leb_Grid_XYZW[   78][3] =  0.000686529762928;

    Leb_Grid_XYZW[   79][0] = -0.153107785246990;
    Leb_Grid_XYZW[   79][1] =  0.153107785246990;
    Leb_Grid_XYZW[   79][2] = -0.976276606394685;
    Leb_Grid_XYZW[   79][3] =  0.000686529762928;

    Leb_Grid_XYZW[   80][0] = -0.153107785246991;
    Leb_Grid_XYZW[   80][1] = -0.153107785246991;
    Leb_Grid_XYZW[   80][2] =  0.976276606394685;
    Leb_Grid_XYZW[   80][3] =  0.000686529762928;

    Leb_Grid_XYZW[   81][0] = -0.153107785246990;
    Leb_Grid_XYZW[   81][1] = -0.153107785246990;
    Leb_Grid_XYZW[   81][2] = -0.976276606394685;
    Leb_Grid_XYZW[   81][3] =  0.000686529762928;

    Leb_Grid_XYZW[   82][0] =  0.153107785246991;
    Leb_Grid_XYZW[   82][1] =  0.976276606394685;
    Leb_Grid_XYZW[   82][2] =  0.153107785246991;
    Leb_Grid_XYZW[   82][3] =  0.000686529762928;

    Leb_Grid_XYZW[   83][0] =  0.153107785246991;
    Leb_Grid_XYZW[   83][1] = -0.976276606394685;
    Leb_Grid_XYZW[   83][2] =  0.153107785246991;
    Leb_Grid_XYZW[   83][3] =  0.000686529762928;

    Leb_Grid_XYZW[   84][0] =  0.153107785246991;
    Leb_Grid_XYZW[   84][1] =  0.976276606394685;
    Leb_Grid_XYZW[   84][2] = -0.153107785246990;
    Leb_Grid_XYZW[   84][3] =  0.000686529762928;

    Leb_Grid_XYZW[   85][0] =  0.153107785246991;
    Leb_Grid_XYZW[   85][1] = -0.976276606394685;
    Leb_Grid_XYZW[   85][2] = -0.153107785246990;
    Leb_Grid_XYZW[   85][3] =  0.000686529762928;

    Leb_Grid_XYZW[   86][0] = -0.153107785246991;
    Leb_Grid_XYZW[   86][1] =  0.976276606394685;
    Leb_Grid_XYZW[   86][2] =  0.153107785246991;
    Leb_Grid_XYZW[   86][3] =  0.000686529762928;

    Leb_Grid_XYZW[   87][0] = -0.153107785246991;
    Leb_Grid_XYZW[   87][1] = -0.976276606394685;
    Leb_Grid_XYZW[   87][2] =  0.153107785246991;
    Leb_Grid_XYZW[   87][3] =  0.000686529762928;

    Leb_Grid_XYZW[   88][0] = -0.153107785246991;
    Leb_Grid_XYZW[   88][1] =  0.976276606394685;
    Leb_Grid_XYZW[   88][2] = -0.153107785246990;
    Leb_Grid_XYZW[   88][3] =  0.000686529762928;

    Leb_Grid_XYZW[   89][0] = -0.153107785246991;
    Leb_Grid_XYZW[   89][1] = -0.976276606394685;
    Leb_Grid_XYZW[   89][2] = -0.153107785246990;
    Leb_Grid_XYZW[   89][3] =  0.000686529762928;

    Leb_Grid_XYZW[   90][0] =  0.976276606394685;
    Leb_Grid_XYZW[   90][1] =  0.153107785246991;
    Leb_Grid_XYZW[   90][2] =  0.153107785246991;
    Leb_Grid_XYZW[   90][3] =  0.000686529762928;

    Leb_Grid_XYZW[   91][0] = -0.976276606394685;
    Leb_Grid_XYZW[   91][1] =  0.153107785246991;
    Leb_Grid_XYZW[   91][2] =  0.153107785246991;
    Leb_Grid_XYZW[   91][3] =  0.000686529762928;

    Leb_Grid_XYZW[   92][0] =  0.976276606394685;
    Leb_Grid_XYZW[   92][1] =  0.153107785246991;
    Leb_Grid_XYZW[   92][2] = -0.153107785246990;
    Leb_Grid_XYZW[   92][3] =  0.000686529762928;

    Leb_Grid_XYZW[   93][0] = -0.976276606394685;
    Leb_Grid_XYZW[   93][1] =  0.153107785246991;
    Leb_Grid_XYZW[   93][2] = -0.153107785246990;
    Leb_Grid_XYZW[   93][3] =  0.000686529762928;

    Leb_Grid_XYZW[   94][0] =  0.976276606394685;
    Leb_Grid_XYZW[   94][1] = -0.153107785246991;
    Leb_Grid_XYZW[   94][2] =  0.153107785246991;
    Leb_Grid_XYZW[   94][3] =  0.000686529762928;

    Leb_Grid_XYZW[   95][0] = -0.976276606394685;
    Leb_Grid_XYZW[   95][1] = -0.153107785246991;
    Leb_Grid_XYZW[   95][2] =  0.153107785246991;
    Leb_Grid_XYZW[   95][3] =  0.000686529762928;

    Leb_Grid_XYZW[   96][0] =  0.976276606394685;
    Leb_Grid_XYZW[   96][1] = -0.153107785246991;
    Leb_Grid_XYZW[   96][2] = -0.153107785246990;
    Leb_Grid_XYZW[   96][3] =  0.000686529762928;

    Leb_Grid_XYZW[   97][0] = -0.976276606394685;
    Leb_Grid_XYZW[   97][1] = -0.153107785246991;
    Leb_Grid_XYZW[   97][2] = -0.153107785246990;
    Leb_Grid_XYZW[   97][3] =  0.000686529762928;

    Leb_Grid_XYZW[   98][0] =  0.218092889166061;
    Leb_Grid_XYZW[   98][1] =  0.218092889166061;
    Leb_Grid_XYZW[   98][2] =  0.951247067480579;
    Leb_Grid_XYZW[   98][3] =  0.000772033855115;

    Leb_Grid_XYZW[   99][0] =  0.218092889166062;
    Leb_Grid_XYZW[   99][1] =  0.218092889166062;
    Leb_Grid_XYZW[   99][2] = -0.951247067480578;
    Leb_Grid_XYZW[   99][3] =  0.000772033855115;

    Leb_Grid_XYZW[  100][0] =  0.218092889166061;
    Leb_Grid_XYZW[  100][1] = -0.218092889166061;
    Leb_Grid_XYZW[  100][2] =  0.951247067480579;
    Leb_Grid_XYZW[  100][3] =  0.000772033855115;

    Leb_Grid_XYZW[  101][0] =  0.218092889166062;
    Leb_Grid_XYZW[  101][1] = -0.218092889166062;
    Leb_Grid_XYZW[  101][2] = -0.951247067480578;
    Leb_Grid_XYZW[  101][3] =  0.000772033855115;

    Leb_Grid_XYZW[  102][0] = -0.218092889166061;
    Leb_Grid_XYZW[  102][1] =  0.218092889166061;
    Leb_Grid_XYZW[  102][2] =  0.951247067480579;
    Leb_Grid_XYZW[  102][3] =  0.000772033855115;

    Leb_Grid_XYZW[  103][0] = -0.218092889166062;
    Leb_Grid_XYZW[  103][1] =  0.218092889166062;
    Leb_Grid_XYZW[  103][2] = -0.951247067480578;
    Leb_Grid_XYZW[  103][3] =  0.000772033855115;

    Leb_Grid_XYZW[  104][0] = -0.218092889166061;
    Leb_Grid_XYZW[  104][1] = -0.218092889166061;
    Leb_Grid_XYZW[  104][2] =  0.951247067480579;
    Leb_Grid_XYZW[  104][3] =  0.000772033855115;

    Leb_Grid_XYZW[  105][0] = -0.218092889166062;
    Leb_Grid_XYZW[  105][1] = -0.218092889166062;
    Leb_Grid_XYZW[  105][2] = -0.951247067480578;
    Leb_Grid_XYZW[  105][3] =  0.000772033855115;

    Leb_Grid_XYZW[  106][0] =  0.218092889166061;
    Leb_Grid_XYZW[  106][1] =  0.951247067480578;
    Leb_Grid_XYZW[  106][2] =  0.218092889166061;
    Leb_Grid_XYZW[  106][3] =  0.000772033855115;

    Leb_Grid_XYZW[  107][0] =  0.218092889166061;
    Leb_Grid_XYZW[  107][1] = -0.951247067480578;
    Leb_Grid_XYZW[  107][2] =  0.218092889166061;
    Leb_Grid_XYZW[  107][3] =  0.000772033855115;

    Leb_Grid_XYZW[  108][0] =  0.218092889166061;
    Leb_Grid_XYZW[  108][1] =  0.951247067480578;
    Leb_Grid_XYZW[  108][2] = -0.218092889166061;
    Leb_Grid_XYZW[  108][3] =  0.000772033855115;

    Leb_Grid_XYZW[  109][0] =  0.218092889166061;
    Leb_Grid_XYZW[  109][1] = -0.951247067480578;
    Leb_Grid_XYZW[  109][2] = -0.218092889166061;
    Leb_Grid_XYZW[  109][3] =  0.000772033855115;

    Leb_Grid_XYZW[  110][0] = -0.218092889166061;
    Leb_Grid_XYZW[  110][1] =  0.951247067480578;
    Leb_Grid_XYZW[  110][2] =  0.218092889166061;
    Leb_Grid_XYZW[  110][3] =  0.000772033855115;

    Leb_Grid_XYZW[  111][0] = -0.218092889166061;
    Leb_Grid_XYZW[  111][1] = -0.951247067480578;
    Leb_Grid_XYZW[  111][2] =  0.218092889166061;
    Leb_Grid_XYZW[  111][3] =  0.000772033855115;

    Leb_Grid_XYZW[  112][0] = -0.218092889166061;
    Leb_Grid_XYZW[  112][1] =  0.951247067480578;
    Leb_Grid_XYZW[  112][2] = -0.218092889166061;
    Leb_Grid_XYZW[  112][3] =  0.000772033855115;

    Leb_Grid_XYZW[  113][0] = -0.218092889166061;
    Leb_Grid_XYZW[  113][1] = -0.951247067480578;
    Leb_Grid_XYZW[  113][2] = -0.218092889166061;
    Leb_Grid_XYZW[  113][3] =  0.000772033855115;

    Leb_Grid_XYZW[  114][0] =  0.951247067480579;
    Leb_Grid_XYZW[  114][1] =  0.218092889166061;
    Leb_Grid_XYZW[  114][2] =  0.218092889166061;
    Leb_Grid_XYZW[  114][3] =  0.000772033855115;

    Leb_Grid_XYZW[  115][0] = -0.951247067480579;
    Leb_Grid_XYZW[  115][1] =  0.218092889166061;
    Leb_Grid_XYZW[  115][2] =  0.218092889166061;
    Leb_Grid_XYZW[  115][3] =  0.000772033855115;

    Leb_Grid_XYZW[  116][0] =  0.951247067480579;
    Leb_Grid_XYZW[  116][1] =  0.218092889166061;
    Leb_Grid_XYZW[  116][2] = -0.218092889166061;
    Leb_Grid_XYZW[  116][3] =  0.000772033855115;

    Leb_Grid_XYZW[  117][0] = -0.951247067480579;
    Leb_Grid_XYZW[  117][1] =  0.218092889166061;
    Leb_Grid_XYZW[  117][2] = -0.218092889166061;
    Leb_Grid_XYZW[  117][3] =  0.000772033855115;

    Leb_Grid_XYZW[  118][0] =  0.951247067480579;
    Leb_Grid_XYZW[  118][1] = -0.218092889166061;
    Leb_Grid_XYZW[  118][2] =  0.218092889166061;
    Leb_Grid_XYZW[  118][3] =  0.000772033855115;

    Leb_Grid_XYZW[  119][0] = -0.951247067480579;
    Leb_Grid_XYZW[  119][1] = -0.218092889166061;
    Leb_Grid_XYZW[  119][2] =  0.218092889166061;
    Leb_Grid_XYZW[  119][3] =  0.000772033855115;

    Leb_Grid_XYZW[  120][0] =  0.951247067480579;
    Leb_Grid_XYZW[  120][1] = -0.218092889166061;
    Leb_Grid_XYZW[  120][2] = -0.218092889166061;
    Leb_Grid_XYZW[  120][3] =  0.000772033855115;

    Leb_Grid_XYZW[  121][0] = -0.951247067480579;
    Leb_Grid_XYZW[  121][1] = -0.218092889166061;
    Leb_Grid_XYZW[  121][2] = -0.218092889166061;
    Leb_Grid_XYZW[  121][3] =  0.000772033855115;

    Leb_Grid_XYZW[  122][0] =  0.283987453220017;
    Leb_Grid_XYZW[  122][1] =  0.283987453220017;
    Leb_Grid_XYZW[  122][2] =  0.915806886208668;
    Leb_Grid_XYZW[  122][3] =  0.000830154595889;

    Leb_Grid_XYZW[  123][0] =  0.283987453220018;
    Leb_Grid_XYZW[  123][1] =  0.283987453220018;
    Leb_Grid_XYZW[  123][2] = -0.915806886208668;
    Leb_Grid_XYZW[  123][3] =  0.000830154595889;

    Leb_Grid_XYZW[  124][0] =  0.283987453220017;
    Leb_Grid_XYZW[  124][1] = -0.283987453220017;
    Leb_Grid_XYZW[  124][2] =  0.915806886208668;
    Leb_Grid_XYZW[  124][3] =  0.000830154595889;

    Leb_Grid_XYZW[  125][0] =  0.283987453220018;
    Leb_Grid_XYZW[  125][1] = -0.283987453220018;
    Leb_Grid_XYZW[  125][2] = -0.915806886208668;
    Leb_Grid_XYZW[  125][3] =  0.000830154595889;

    Leb_Grid_XYZW[  126][0] = -0.283987453220017;
    Leb_Grid_XYZW[  126][1] =  0.283987453220017;
    Leb_Grid_XYZW[  126][2] =  0.915806886208668;
    Leb_Grid_XYZW[  126][3] =  0.000830154595889;

    Leb_Grid_XYZW[  127][0] = -0.283987453220018;
    Leb_Grid_XYZW[  127][1] =  0.283987453220018;
    Leb_Grid_XYZW[  127][2] = -0.915806886208668;
    Leb_Grid_XYZW[  127][3] =  0.000830154595889;

    Leb_Grid_XYZW[  128][0] = -0.283987453220017;
    Leb_Grid_XYZW[  128][1] = -0.283987453220017;
    Leb_Grid_XYZW[  128][2] =  0.915806886208668;
    Leb_Grid_XYZW[  128][3] =  0.000830154595889;

    Leb_Grid_XYZW[  129][0] = -0.283987453220018;
    Leb_Grid_XYZW[  129][1] = -0.283987453220018;
    Leb_Grid_XYZW[  129][2] = -0.915806886208668;
    Leb_Grid_XYZW[  129][3] =  0.000830154595889;

    Leb_Grid_XYZW[  130][0] =  0.283987453220017;
    Leb_Grid_XYZW[  130][1] =  0.915806886208668;
    Leb_Grid_XYZW[  130][2] =  0.283987453220018;
    Leb_Grid_XYZW[  130][3] =  0.000830154595889;

    Leb_Grid_XYZW[  131][0] =  0.283987453220017;
    Leb_Grid_XYZW[  131][1] = -0.915806886208668;
    Leb_Grid_XYZW[  131][2] =  0.283987453220018;
    Leb_Grid_XYZW[  131][3] =  0.000830154595889;

    Leb_Grid_XYZW[  132][0] =  0.283987453220017;
    Leb_Grid_XYZW[  132][1] =  0.915806886208668;
    Leb_Grid_XYZW[  132][2] = -0.283987453220017;
    Leb_Grid_XYZW[  132][3] =  0.000830154595889;

    Leb_Grid_XYZW[  133][0] =  0.283987453220017;
    Leb_Grid_XYZW[  133][1] = -0.915806886208668;
    Leb_Grid_XYZW[  133][2] = -0.283987453220017;
    Leb_Grid_XYZW[  133][3] =  0.000830154595889;

    Leb_Grid_XYZW[  134][0] = -0.283987453220017;
    Leb_Grid_XYZW[  134][1] =  0.915806886208668;
    Leb_Grid_XYZW[  134][2] =  0.283987453220018;
    Leb_Grid_XYZW[  134][3] =  0.000830154595889;

    Leb_Grid_XYZW[  135][0] = -0.283987453220017;
    Leb_Grid_XYZW[  135][1] = -0.915806886208668;
    Leb_Grid_XYZW[  135][2] =  0.283987453220018;
    Leb_Grid_XYZW[  135][3] =  0.000830154595889;

    Leb_Grid_XYZW[  136][0] = -0.283987453220017;
    Leb_Grid_XYZW[  136][1] =  0.915806886208668;
    Leb_Grid_XYZW[  136][2] = -0.283987453220017;
    Leb_Grid_XYZW[  136][3] =  0.000830154595889;

    Leb_Grid_XYZW[  137][0] = -0.283987453220017;
    Leb_Grid_XYZW[  137][1] = -0.915806886208668;
    Leb_Grid_XYZW[  137][2] = -0.283987453220017;
    Leb_Grid_XYZW[  137][3] =  0.000830154595889;

    Leb_Grid_XYZW[  138][0] =  0.915806886208668;
    Leb_Grid_XYZW[  138][1] =  0.283987453220018;
    Leb_Grid_XYZW[  138][2] =  0.283987453220018;
    Leb_Grid_XYZW[  138][3] =  0.000830154595889;

    Leb_Grid_XYZW[  139][0] = -0.915806886208668;
    Leb_Grid_XYZW[  139][1] =  0.283987453220018;
    Leb_Grid_XYZW[  139][2] =  0.283987453220018;
    Leb_Grid_XYZW[  139][3] =  0.000830154595889;

    Leb_Grid_XYZW[  140][0] =  0.915806886208668;
    Leb_Grid_XYZW[  140][1] =  0.283987453220018;
    Leb_Grid_XYZW[  140][2] = -0.283987453220017;
    Leb_Grid_XYZW[  140][3] =  0.000830154595889;

    Leb_Grid_XYZW[  141][0] = -0.915806886208668;
    Leb_Grid_XYZW[  141][1] =  0.283987453220018;
    Leb_Grid_XYZW[  141][2] = -0.283987453220017;
    Leb_Grid_XYZW[  141][3] =  0.000830154595889;

    Leb_Grid_XYZW[  142][0] =  0.915806886208668;
    Leb_Grid_XYZW[  142][1] = -0.283987453220018;
    Leb_Grid_XYZW[  142][2] =  0.283987453220018;
    Leb_Grid_XYZW[  142][3] =  0.000830154595889;

    Leb_Grid_XYZW[  143][0] = -0.915806886208668;
    Leb_Grid_XYZW[  143][1] = -0.283987453220018;
    Leb_Grid_XYZW[  143][2] =  0.283987453220018;
    Leb_Grid_XYZW[  143][3] =  0.000830154595889;

    Leb_Grid_XYZW[  144][0] =  0.915806886208668;
    Leb_Grid_XYZW[  144][1] = -0.283987453220018;
    Leb_Grid_XYZW[  144][2] = -0.283987453220017;
    Leb_Grid_XYZW[  144][3] =  0.000830154595889;

    Leb_Grid_XYZW[  145][0] = -0.915806886208668;
    Leb_Grid_XYZW[  145][1] = -0.283987453220018;
    Leb_Grid_XYZW[  145][2] = -0.283987453220017;
    Leb_Grid_XYZW[  145][3] =  0.000830154595889;

    Leb_Grid_XYZW[  146][0] =  0.349117760096376;
    Leb_Grid_XYZW[  146][1] =  0.349117760096376;
    Leb_Grid_XYZW[  146][2] =  0.869616915181954;
    Leb_Grid_XYZW[  146][3] =  0.000868669255018;

    Leb_Grid_XYZW[  147][0] =  0.349117760096376;
    Leb_Grid_XYZW[  147][1] =  0.349117760096376;
    Leb_Grid_XYZW[  147][2] = -0.869616915181954;
    Leb_Grid_XYZW[  147][3] =  0.000868669255018;

    Leb_Grid_XYZW[  148][0] =  0.349117760096376;
    Leb_Grid_XYZW[  148][1] = -0.349117760096376;
    Leb_Grid_XYZW[  148][2] =  0.869616915181954;
    Leb_Grid_XYZW[  148][3] =  0.000868669255018;

    Leb_Grid_XYZW[  149][0] =  0.349117760096376;
    Leb_Grid_XYZW[  149][1] = -0.349117760096376;
    Leb_Grid_XYZW[  149][2] = -0.869616915181954;
    Leb_Grid_XYZW[  149][3] =  0.000868669255018;

    Leb_Grid_XYZW[  150][0] = -0.349117760096376;
    Leb_Grid_XYZW[  150][1] =  0.349117760096376;
    Leb_Grid_XYZW[  150][2] =  0.869616915181954;
    Leb_Grid_XYZW[  150][3] =  0.000868669255018;

    Leb_Grid_XYZW[  151][0] = -0.349117760096376;
    Leb_Grid_XYZW[  151][1] =  0.349117760096376;
    Leb_Grid_XYZW[  151][2] = -0.869616915181954;
    Leb_Grid_XYZW[  151][3] =  0.000868669255018;

    Leb_Grid_XYZW[  152][0] = -0.349117760096376;
    Leb_Grid_XYZW[  152][1] = -0.349117760096376;
    Leb_Grid_XYZW[  152][2] =  0.869616915181954;
    Leb_Grid_XYZW[  152][3] =  0.000868669255018;

    Leb_Grid_XYZW[  153][0] = -0.349117760096376;
    Leb_Grid_XYZW[  153][1] = -0.349117760096376;
    Leb_Grid_XYZW[  153][2] = -0.869616915181954;
    Leb_Grid_XYZW[  153][3] =  0.000868669255018;

    Leb_Grid_XYZW[  154][0] =  0.349117760096376;
    Leb_Grid_XYZW[  154][1] =  0.869616915181954;
    Leb_Grid_XYZW[  154][2] =  0.349117760096376;
    Leb_Grid_XYZW[  154][3] =  0.000868669255018;

    Leb_Grid_XYZW[  155][0] =  0.349117760096376;
    Leb_Grid_XYZW[  155][1] = -0.869616915181954;
    Leb_Grid_XYZW[  155][2] =  0.349117760096376;
    Leb_Grid_XYZW[  155][3] =  0.000868669255018;

    Leb_Grid_XYZW[  156][0] =  0.349117760096376;
    Leb_Grid_XYZW[  156][1] =  0.869616915181954;
    Leb_Grid_XYZW[  156][2] = -0.349117760096376;
    Leb_Grid_XYZW[  156][3] =  0.000868669255018;

    Leb_Grid_XYZW[  157][0] =  0.349117760096376;
    Leb_Grid_XYZW[  157][1] = -0.869616915181954;
    Leb_Grid_XYZW[  157][2] = -0.349117760096376;
    Leb_Grid_XYZW[  157][3] =  0.000868669255018;

    Leb_Grid_XYZW[  158][0] = -0.349117760096376;
    Leb_Grid_XYZW[  158][1] =  0.869616915181954;
    Leb_Grid_XYZW[  158][2] =  0.349117760096376;
    Leb_Grid_XYZW[  158][3] =  0.000868669255018;

    Leb_Grid_XYZW[  159][0] = -0.349117760096376;
    Leb_Grid_XYZW[  159][1] = -0.869616915181954;
    Leb_Grid_XYZW[  159][2] =  0.349117760096376;
    Leb_Grid_XYZW[  159][3] =  0.000868669255018;

    Leb_Grid_XYZW[  160][0] = -0.349117760096376;
    Leb_Grid_XYZW[  160][1] =  0.869616915181954;
    Leb_Grid_XYZW[  160][2] = -0.349117760096376;
    Leb_Grid_XYZW[  160][3] =  0.000868669255018;

    Leb_Grid_XYZW[  161][0] = -0.349117760096376;
    Leb_Grid_XYZW[  161][1] = -0.869616915181954;
    Leb_Grid_XYZW[  161][2] = -0.349117760096376;
    Leb_Grid_XYZW[  161][3] =  0.000868669255018;

    Leb_Grid_XYZW[  162][0] =  0.869616915181954;
    Leb_Grid_XYZW[  162][1] =  0.349117760096376;
    Leb_Grid_XYZW[  162][2] =  0.349117760096376;
    Leb_Grid_XYZW[  162][3] =  0.000868669255018;

    Leb_Grid_XYZW[  163][0] = -0.869616915181954;
    Leb_Grid_XYZW[  163][1] =  0.349117760096376;
    Leb_Grid_XYZW[  163][2] =  0.349117760096376;
    Leb_Grid_XYZW[  163][3] =  0.000868669255018;

    Leb_Grid_XYZW[  164][0] =  0.869616915181954;
    Leb_Grid_XYZW[  164][1] =  0.349117760096376;
    Leb_Grid_XYZW[  164][2] = -0.349117760096376;
    Leb_Grid_XYZW[  164][3] =  0.000868669255018;

    Leb_Grid_XYZW[  165][0] = -0.869616915181954;
    Leb_Grid_XYZW[  165][1] =  0.349117760096376;
    Leb_Grid_XYZW[  165][2] = -0.349117760096376;
    Leb_Grid_XYZW[  165][3] =  0.000868669255018;

    Leb_Grid_XYZW[  166][0] =  0.869616915181954;
    Leb_Grid_XYZW[  166][1] = -0.349117760096376;
    Leb_Grid_XYZW[  166][2] =  0.349117760096376;
    Leb_Grid_XYZW[  166][3] =  0.000868669255018;

    Leb_Grid_XYZW[  167][0] = -0.869616915181954;
    Leb_Grid_XYZW[  167][1] = -0.349117760096376;
    Leb_Grid_XYZW[  167][2] =  0.349117760096376;
    Leb_Grid_XYZW[  167][3] =  0.000868669255018;

    Leb_Grid_XYZW[  168][0] =  0.869616915181954;
    Leb_Grid_XYZW[  168][1] = -0.349117760096376;
    Leb_Grid_XYZW[  168][2] = -0.349117760096376;
    Leb_Grid_XYZW[  168][3] =  0.000868669255018;

    Leb_Grid_XYZW[  169][0] = -0.869616915181954;
    Leb_Grid_XYZW[  169][1] = -0.349117760096376;
    Leb_Grid_XYZW[  169][2] = -0.349117760096376;
    Leb_Grid_XYZW[  169][3] =  0.000868669255018;

    Leb_Grid_XYZW[  170][0] =  0.412143146144431;
    Leb_Grid_XYZW[  170][1] =  0.412143146144431;
    Leb_Grid_XYZW[  170][2] =  0.812573722299916;
    Leb_Grid_XYZW[  170][3] =  0.000892707628585;

    Leb_Grid_XYZW[  171][0] =  0.412143146144431;
    Leb_Grid_XYZW[  171][1] =  0.412143146144431;
    Leb_Grid_XYZW[  171][2] = -0.812573722299916;
    Leb_Grid_XYZW[  171][3] =  0.000892707628585;

    Leb_Grid_XYZW[  172][0] =  0.412143146144431;
    Leb_Grid_XYZW[  172][1] = -0.412143146144431;
    Leb_Grid_XYZW[  172][2] =  0.812573722299916;
    Leb_Grid_XYZW[  172][3] =  0.000892707628585;

    Leb_Grid_XYZW[  173][0] =  0.412143146144431;
    Leb_Grid_XYZW[  173][1] = -0.412143146144431;
    Leb_Grid_XYZW[  173][2] = -0.812573722299916;
    Leb_Grid_XYZW[  173][3] =  0.000892707628585;

    Leb_Grid_XYZW[  174][0] = -0.412143146144431;
    Leb_Grid_XYZW[  174][1] =  0.412143146144431;
    Leb_Grid_XYZW[  174][2] =  0.812573722299916;
    Leb_Grid_XYZW[  174][3] =  0.000892707628585;

    Leb_Grid_XYZW[  175][0] = -0.412143146144431;
    Leb_Grid_XYZW[  175][1] =  0.412143146144431;
    Leb_Grid_XYZW[  175][2] = -0.812573722299916;
    Leb_Grid_XYZW[  175][3] =  0.000892707628585;

    Leb_Grid_XYZW[  176][0] = -0.412143146144431;
    Leb_Grid_XYZW[  176][1] = -0.412143146144431;
    Leb_Grid_XYZW[  176][2] =  0.812573722299916;
    Leb_Grid_XYZW[  176][3] =  0.000892707628585;

    Leb_Grid_XYZW[  177][0] = -0.412143146144431;
    Leb_Grid_XYZW[  177][1] = -0.412143146144431;
    Leb_Grid_XYZW[  177][2] = -0.812573722299916;
    Leb_Grid_XYZW[  177][3] =  0.000892707628585;

    Leb_Grid_XYZW[  178][0] =  0.412143146144431;
    Leb_Grid_XYZW[  178][1] =  0.812573722299916;
    Leb_Grid_XYZW[  178][2] =  0.412143146144431;
    Leb_Grid_XYZW[  178][3] =  0.000892707628585;

    Leb_Grid_XYZW[  179][0] =  0.412143146144431;
    Leb_Grid_XYZW[  179][1] = -0.812573722299916;
    Leb_Grid_XYZW[  179][2] =  0.412143146144431;
    Leb_Grid_XYZW[  179][3] =  0.000892707628585;

    Leb_Grid_XYZW[  180][0] =  0.412143146144431;
    Leb_Grid_XYZW[  180][1] =  0.812573722299916;
    Leb_Grid_XYZW[  180][2] = -0.412143146144431;
    Leb_Grid_XYZW[  180][3] =  0.000892707628585;

    Leb_Grid_XYZW[  181][0] =  0.412143146144431;
    Leb_Grid_XYZW[  181][1] = -0.812573722299916;
    Leb_Grid_XYZW[  181][2] = -0.412143146144431;
    Leb_Grid_XYZW[  181][3] =  0.000892707628585;

    Leb_Grid_XYZW[  182][0] = -0.412143146144431;
    Leb_Grid_XYZW[  182][1] =  0.812573722299916;
    Leb_Grid_XYZW[  182][2] =  0.412143146144431;
    Leb_Grid_XYZW[  182][3] =  0.000892707628585;

    Leb_Grid_XYZW[  183][0] = -0.412143146144431;
    Leb_Grid_XYZW[  183][1] = -0.812573722299916;
    Leb_Grid_XYZW[  183][2] =  0.412143146144431;
    Leb_Grid_XYZW[  183][3] =  0.000892707628585;

    Leb_Grid_XYZW[  184][0] = -0.412143146144431;
    Leb_Grid_XYZW[  184][1] =  0.812573722299916;
    Leb_Grid_XYZW[  184][2] = -0.412143146144431;
    Leb_Grid_XYZW[  184][3] =  0.000892707628585;

    Leb_Grid_XYZW[  185][0] = -0.412143146144431;
    Leb_Grid_XYZW[  185][1] = -0.812573722299916;
    Leb_Grid_XYZW[  185][2] = -0.412143146144431;
    Leb_Grid_XYZW[  185][3] =  0.000892707628585;

    Leb_Grid_XYZW[  186][0] =  0.812573722299916;
    Leb_Grid_XYZW[  186][1] =  0.412143146144431;
    Leb_Grid_XYZW[  186][2] =  0.412143146144431;
    Leb_Grid_XYZW[  186][3] =  0.000892707628585;

    Leb_Grid_XYZW[  187][0] = -0.812573722299915;
    Leb_Grid_XYZW[  187][1] =  0.412143146144431;
    Leb_Grid_XYZW[  187][2] =  0.412143146144431;
    Leb_Grid_XYZW[  187][3] =  0.000892707628585;

    Leb_Grid_XYZW[  188][0] =  0.812573722299916;
    Leb_Grid_XYZW[  188][1] =  0.412143146144431;
    Leb_Grid_XYZW[  188][2] = -0.412143146144431;
    Leb_Grid_XYZW[  188][3] =  0.000892707628585;

    Leb_Grid_XYZW[  189][0] = -0.812573722299915;
    Leb_Grid_XYZW[  189][1] =  0.412143146144431;
    Leb_Grid_XYZW[  189][2] = -0.412143146144431;
    Leb_Grid_XYZW[  189][3] =  0.000892707628585;

    Leb_Grid_XYZW[  190][0] =  0.812573722299916;
    Leb_Grid_XYZW[  190][1] = -0.412143146144431;
    Leb_Grid_XYZW[  190][2] =  0.412143146144431;
    Leb_Grid_XYZW[  190][3] =  0.000892707628585;

    Leb_Grid_XYZW[  191][0] = -0.812573722299915;
    Leb_Grid_XYZW[  191][1] = -0.412143146144431;
    Leb_Grid_XYZW[  191][2] =  0.412143146144431;
    Leb_Grid_XYZW[  191][3] =  0.000892707628585;

    Leb_Grid_XYZW[  192][0] =  0.812573722299916;
    Leb_Grid_XYZW[  192][1] = -0.412143146144431;
    Leb_Grid_XYZW[  192][2] = -0.412143146144431;
    Leb_Grid_XYZW[  192][3] =  0.000892707628585;

    Leb_Grid_XYZW[  193][0] = -0.812573722299915;
    Leb_Grid_XYZW[  193][1] = -0.412143146144431;
    Leb_Grid_XYZW[  193][2] = -0.412143146144431;
    Leb_Grid_XYZW[  193][3] =  0.000892707628585;

    Leb_Grid_XYZW[  194][0] =  0.471899362714913;
    Leb_Grid_XYZW[  194][1] =  0.471899362714913;
    Leb_Grid_XYZW[  194][2] =  0.744729469632107;
    Leb_Grid_XYZW[  194][3] =  0.000906082023857;

    Leb_Grid_XYZW[  195][0] =  0.471899362714913;
    Leb_Grid_XYZW[  195][1] =  0.471899362714913;
    Leb_Grid_XYZW[  195][2] = -0.744729469632107;
    Leb_Grid_XYZW[  195][3] =  0.000906082023857;

    Leb_Grid_XYZW[  196][0] =  0.471899362714913;
    Leb_Grid_XYZW[  196][1] = -0.471899362714913;
    Leb_Grid_XYZW[  196][2] =  0.744729469632107;
    Leb_Grid_XYZW[  196][3] =  0.000906082023857;

    Leb_Grid_XYZW[  197][0] =  0.471899362714913;
    Leb_Grid_XYZW[  197][1] = -0.471899362714913;
    Leb_Grid_XYZW[  197][2] = -0.744729469632107;
    Leb_Grid_XYZW[  197][3] =  0.000906082023857;

    Leb_Grid_XYZW[  198][0] = -0.471899362714913;
    Leb_Grid_XYZW[  198][1] =  0.471899362714913;
    Leb_Grid_XYZW[  198][2] =  0.744729469632107;
    Leb_Grid_XYZW[  198][3] =  0.000906082023857;

    Leb_Grid_XYZW[  199][0] = -0.471899362714913;
    Leb_Grid_XYZW[  199][1] =  0.471899362714913;
    Leb_Grid_XYZW[  199][2] = -0.744729469632107;
    Leb_Grid_XYZW[  199][3] =  0.000906082023857;

    Leb_Grid_XYZW[  200][0] = -0.471899362714913;
    Leb_Grid_XYZW[  200][1] = -0.471899362714913;
    Leb_Grid_XYZW[  200][2] =  0.744729469632107;
    Leb_Grid_XYZW[  200][3] =  0.000906082023857;

    Leb_Grid_XYZW[  201][0] = -0.471899362714913;
    Leb_Grid_XYZW[  201][1] = -0.471899362714913;
    Leb_Grid_XYZW[  201][2] = -0.744729469632107;
    Leb_Grid_XYZW[  201][3] =  0.000906082023857;

    Leb_Grid_XYZW[  202][0] =  0.471899362714913;
    Leb_Grid_XYZW[  202][1] =  0.744729469632107;
    Leb_Grid_XYZW[  202][2] =  0.471899362714913;
    Leb_Grid_XYZW[  202][3] =  0.000906082023857;

    Leb_Grid_XYZW[  203][0] =  0.471899362714913;
    Leb_Grid_XYZW[  203][1] = -0.744729469632107;
    Leb_Grid_XYZW[  203][2] =  0.471899362714913;
    Leb_Grid_XYZW[  203][3] =  0.000906082023857;

    Leb_Grid_XYZW[  204][0] =  0.471899362714913;
    Leb_Grid_XYZW[  204][1] =  0.744729469632106;
    Leb_Grid_XYZW[  204][2] = -0.471899362714913;
    Leb_Grid_XYZW[  204][3] =  0.000906082023857;

    Leb_Grid_XYZW[  205][0] =  0.471899362714913;
    Leb_Grid_XYZW[  205][1] = -0.744729469632106;
    Leb_Grid_XYZW[  205][2] = -0.471899362714913;
    Leb_Grid_XYZW[  205][3] =  0.000906082023857;

    Leb_Grid_XYZW[  206][0] = -0.471899362714912;
    Leb_Grid_XYZW[  206][1] =  0.744729469632107;
    Leb_Grid_XYZW[  206][2] =  0.471899362714913;
    Leb_Grid_XYZW[  206][3] =  0.000906082023857;

    Leb_Grid_XYZW[  207][0] = -0.471899362714912;
    Leb_Grid_XYZW[  207][1] = -0.744729469632107;
    Leb_Grid_XYZW[  207][2] =  0.471899362714913;
    Leb_Grid_XYZW[  207][3] =  0.000906082023857;

    Leb_Grid_XYZW[  208][0] = -0.471899362714912;
    Leb_Grid_XYZW[  208][1] =  0.744729469632107;
    Leb_Grid_XYZW[  208][2] = -0.471899362714913;
    Leb_Grid_XYZW[  208][3] =  0.000906082023857;

    Leb_Grid_XYZW[  209][0] = -0.471899362714912;
    Leb_Grid_XYZW[  209][1] = -0.744729469632107;
    Leb_Grid_XYZW[  209][2] = -0.471899362714913;
    Leb_Grid_XYZW[  209][3] =  0.000906082023857;

    Leb_Grid_XYZW[  210][0] =  0.744729469632106;
    Leb_Grid_XYZW[  210][1] =  0.471899362714913;
    Leb_Grid_XYZW[  210][2] =  0.471899362714913;
    Leb_Grid_XYZW[  210][3] =  0.000906082023857;

    Leb_Grid_XYZW[  211][0] = -0.744729469632106;
    Leb_Grid_XYZW[  211][1] =  0.471899362714913;
    Leb_Grid_XYZW[  211][2] =  0.471899362714913;
    Leb_Grid_XYZW[  211][3] =  0.000906082023857;

    Leb_Grid_XYZW[  212][0] =  0.744729469632106;
    Leb_Grid_XYZW[  212][1] =  0.471899362714913;
    Leb_Grid_XYZW[  212][2] = -0.471899362714913;
    Leb_Grid_XYZW[  212][3] =  0.000906082023857;

    Leb_Grid_XYZW[  213][0] = -0.744729469632106;
    Leb_Grid_XYZW[  213][1] =  0.471899362714913;
    Leb_Grid_XYZW[  213][2] = -0.471899362714913;
    Leb_Grid_XYZW[  213][3] =  0.000906082023857;

    Leb_Grid_XYZW[  214][0] =  0.744729469632106;
    Leb_Grid_XYZW[  214][1] = -0.471899362714913;
    Leb_Grid_XYZW[  214][2] =  0.471899362714913;
    Leb_Grid_XYZW[  214][3] =  0.000906082023857;

    Leb_Grid_XYZW[  215][0] = -0.744729469632106;
    Leb_Grid_XYZW[  215][1] = -0.471899362714913;
    Leb_Grid_XYZW[  215][2] =  0.471899362714913;
    Leb_Grid_XYZW[  215][3] =  0.000906082023857;

    Leb_Grid_XYZW[  216][0] =  0.744729469632106;
    Leb_Grid_XYZW[  216][1] = -0.471899362714913;
    Leb_Grid_XYZW[  216][2] = -0.471899362714913;
    Leb_Grid_XYZW[  216][3] =  0.000906082023857;

    Leb_Grid_XYZW[  217][0] = -0.744729469632106;
    Leb_Grid_XYZW[  217][1] = -0.471899362714913;
    Leb_Grid_XYZW[  217][2] = -0.471899362714913;
    Leb_Grid_XYZW[  217][3] =  0.000906082023857;

    Leb_Grid_XYZW[  218][0] =  0.527314545284234;
    Leb_Grid_XYZW[  218][1] =  0.527314545284234;
    Leb_Grid_XYZW[  218][2] =  0.666242253736104;
    Leb_Grid_XYZW[  218][3] =  0.000911977725494;

    Leb_Grid_XYZW[  219][0] =  0.527314545284234;
    Leb_Grid_XYZW[  219][1] =  0.527314545284234;
    Leb_Grid_XYZW[  219][2] = -0.666242253736104;
    Leb_Grid_XYZW[  219][3] =  0.000911977725494;

    Leb_Grid_XYZW[  220][0] =  0.527314545284234;
    Leb_Grid_XYZW[  220][1] = -0.527314545284234;
    Leb_Grid_XYZW[  220][2] =  0.666242253736104;
    Leb_Grid_XYZW[  220][3] =  0.000911977725494;

    Leb_Grid_XYZW[  221][0] =  0.527314545284234;
    Leb_Grid_XYZW[  221][1] = -0.527314545284234;
    Leb_Grid_XYZW[  221][2] = -0.666242253736104;
    Leb_Grid_XYZW[  221][3] =  0.000911977725494;

    Leb_Grid_XYZW[  222][0] = -0.527314545284234;
    Leb_Grid_XYZW[  222][1] =  0.527314545284234;
    Leb_Grid_XYZW[  222][2] =  0.666242253736104;
    Leb_Grid_XYZW[  222][3] =  0.000911977725494;

    Leb_Grid_XYZW[  223][0] = -0.527314545284234;
    Leb_Grid_XYZW[  223][1] =  0.527314545284234;
    Leb_Grid_XYZW[  223][2] = -0.666242253736104;
    Leb_Grid_XYZW[  223][3] =  0.000911977725494;

    Leb_Grid_XYZW[  224][0] = -0.527314545284234;
    Leb_Grid_XYZW[  224][1] = -0.527314545284234;
    Leb_Grid_XYZW[  224][2] =  0.666242253736104;
    Leb_Grid_XYZW[  224][3] =  0.000911977725494;

    Leb_Grid_XYZW[  225][0] = -0.527314545284234;
    Leb_Grid_XYZW[  225][1] = -0.527314545284234;
    Leb_Grid_XYZW[  225][2] = -0.666242253736104;
    Leb_Grid_XYZW[  225][3] =  0.000911977725494;

    Leb_Grid_XYZW[  226][0] =  0.527314545284234;
    Leb_Grid_XYZW[  226][1] =  0.666242253736104;
    Leb_Grid_XYZW[  226][2] =  0.527314545284234;
    Leb_Grid_XYZW[  226][3] =  0.000911977725494;

    Leb_Grid_XYZW[  227][0] =  0.527314545284234;
    Leb_Grid_XYZW[  227][1] = -0.666242253736104;
    Leb_Grid_XYZW[  227][2] =  0.527314545284234;
    Leb_Grid_XYZW[  227][3] =  0.000911977725494;

    Leb_Grid_XYZW[  228][0] =  0.527314545284234;
    Leb_Grid_XYZW[  228][1] =  0.666242253736104;
    Leb_Grid_XYZW[  228][2] = -0.527314545284234;
    Leb_Grid_XYZW[  228][3] =  0.000911977725494;

    Leb_Grid_XYZW[  229][0] =  0.527314545284234;
    Leb_Grid_XYZW[  229][1] = -0.666242253736104;
    Leb_Grid_XYZW[  229][2] = -0.527314545284234;
    Leb_Grid_XYZW[  229][3] =  0.000911977725494;

    Leb_Grid_XYZW[  230][0] = -0.527314545284234;
    Leb_Grid_XYZW[  230][1] =  0.666242253736104;
    Leb_Grid_XYZW[  230][2] =  0.527314545284234;
    Leb_Grid_XYZW[  230][3] =  0.000911977725494;

    Leb_Grid_XYZW[  231][0] = -0.527314545284234;
    Leb_Grid_XYZW[  231][1] = -0.666242253736104;
    Leb_Grid_XYZW[  231][2] =  0.527314545284234;
    Leb_Grid_XYZW[  231][3] =  0.000911977725494;

    Leb_Grid_XYZW[  232][0] = -0.527314545284234;
    Leb_Grid_XYZW[  232][1] =  0.666242253736104;
    Leb_Grid_XYZW[  232][2] = -0.527314545284234;
    Leb_Grid_XYZW[  232][3] =  0.000911977725494;

    Leb_Grid_XYZW[  233][0] = -0.527314545284234;
    Leb_Grid_XYZW[  233][1] = -0.666242253736104;
    Leb_Grid_XYZW[  233][2] = -0.527314545284234;
    Leb_Grid_XYZW[  233][3] =  0.000911977725494;

    Leb_Grid_XYZW[  234][0] =  0.666242253736104;
    Leb_Grid_XYZW[  234][1] =  0.527314545284234;
    Leb_Grid_XYZW[  234][2] =  0.527314545284234;
    Leb_Grid_XYZW[  234][3] =  0.000911977725494;

    Leb_Grid_XYZW[  235][0] = -0.666242253736104;
    Leb_Grid_XYZW[  235][1] =  0.527314545284234;
    Leb_Grid_XYZW[  235][2] =  0.527314545284234;
    Leb_Grid_XYZW[  235][3] =  0.000911977725494;

    Leb_Grid_XYZW[  236][0] =  0.666242253736104;
    Leb_Grid_XYZW[  236][1] =  0.527314545284234;
    Leb_Grid_XYZW[  236][2] = -0.527314545284234;
    Leb_Grid_XYZW[  236][3] =  0.000911977725494;

    Leb_Grid_XYZW[  237][0] = -0.666242253736104;
    Leb_Grid_XYZW[  237][1] =  0.527314545284234;
    Leb_Grid_XYZW[  237][2] = -0.527314545284234;
    Leb_Grid_XYZW[  237][3] =  0.000911977725494;

    Leb_Grid_XYZW[  238][0] =  0.666242253736104;
    Leb_Grid_XYZW[  238][1] = -0.527314545284234;
    Leb_Grid_XYZW[  238][2] =  0.527314545284234;
    Leb_Grid_XYZW[  238][3] =  0.000911977725494;

    Leb_Grid_XYZW[  239][0] = -0.666242253736104;
    Leb_Grid_XYZW[  239][1] = -0.527314545284234;
    Leb_Grid_XYZW[  239][2] =  0.527314545284234;
    Leb_Grid_XYZW[  239][3] =  0.000911977725494;

    Leb_Grid_XYZW[  240][0] =  0.666242253736104;
    Leb_Grid_XYZW[  240][1] = -0.527314545284234;
    Leb_Grid_XYZW[  240][2] = -0.527314545284234;
    Leb_Grid_XYZW[  240][3] =  0.000911977725494;

    Leb_Grid_XYZW[  241][0] = -0.666242253736104;
    Leb_Grid_XYZW[  241][1] = -0.527314545284234;
    Leb_Grid_XYZW[  241][2] = -0.527314545284234;
    Leb_Grid_XYZW[  241][3] =  0.000911977725494;

    Leb_Grid_XYZW[  242][0] =  0.620947533244402;
    Leb_Grid_XYZW[  242][1] =  0.620947533244402;
    Leb_Grid_XYZW[  242][2] =  0.478380938076952;
    Leb_Grid_XYZW[  242][3] =  0.000912872013860;

    Leb_Grid_XYZW[  243][0] =  0.620947533244402;
    Leb_Grid_XYZW[  243][1] =  0.620947533244402;
    Leb_Grid_XYZW[  243][2] = -0.478380938076952;
    Leb_Grid_XYZW[  243][3] =  0.000912872013860;

    Leb_Grid_XYZW[  244][0] =  0.620947533244402;
    Leb_Grid_XYZW[  244][1] = -0.620947533244402;
    Leb_Grid_XYZW[  244][2] =  0.478380938076952;
    Leb_Grid_XYZW[  244][3] =  0.000912872013860;

    Leb_Grid_XYZW[  245][0] =  0.620947533244402;
    Leb_Grid_XYZW[  245][1] = -0.620947533244402;
    Leb_Grid_XYZW[  245][2] = -0.478380938076952;
    Leb_Grid_XYZW[  245][3] =  0.000912872013860;

    Leb_Grid_XYZW[  246][0] = -0.620947533244402;
    Leb_Grid_XYZW[  246][1] =  0.620947533244402;
    Leb_Grid_XYZW[  246][2] =  0.478380938076952;
    Leb_Grid_XYZW[  246][3] =  0.000912872013860;

    Leb_Grid_XYZW[  247][0] = -0.620947533244402;
    Leb_Grid_XYZW[  247][1] =  0.620947533244402;
    Leb_Grid_XYZW[  247][2] = -0.478380938076952;
    Leb_Grid_XYZW[  247][3] =  0.000912872013860;

    Leb_Grid_XYZW[  248][0] = -0.620947533244402;
    Leb_Grid_XYZW[  248][1] = -0.620947533244402;
    Leb_Grid_XYZW[  248][2] =  0.478380938076952;
    Leb_Grid_XYZW[  248][3] =  0.000912872013860;

    Leb_Grid_XYZW[  249][0] = -0.620947533244402;
    Leb_Grid_XYZW[  249][1] = -0.620947533244402;
    Leb_Grid_XYZW[  249][2] = -0.478380938076952;
    Leb_Grid_XYZW[  249][3] =  0.000912872013860;

    Leb_Grid_XYZW[  250][0] =  0.620947533244402;
    Leb_Grid_XYZW[  250][1] =  0.478380938076952;
    Leb_Grid_XYZW[  250][2] =  0.620947533244402;
    Leb_Grid_XYZW[  250][3] =  0.000912872013860;

    Leb_Grid_XYZW[  251][0] =  0.620947533244402;
    Leb_Grid_XYZW[  251][1] = -0.478380938076952;
    Leb_Grid_XYZW[  251][2] =  0.620947533244402;
    Leb_Grid_XYZW[  251][3] =  0.000912872013860;

    Leb_Grid_XYZW[  252][0] =  0.620947533244402;
    Leb_Grid_XYZW[  252][1] =  0.478380938076952;
    Leb_Grid_XYZW[  252][2] = -0.620947533244402;
    Leb_Grid_XYZW[  252][3] =  0.000912872013860;

    Leb_Grid_XYZW[  253][0] =  0.620947533244402;
    Leb_Grid_XYZW[  253][1] = -0.478380938076952;
    Leb_Grid_XYZW[  253][2] = -0.620947533244402;
    Leb_Grid_XYZW[  253][3] =  0.000912872013860;

    Leb_Grid_XYZW[  254][0] = -0.620947533244402;
    Leb_Grid_XYZW[  254][1] =  0.478380938076952;
    Leb_Grid_XYZW[  254][2] =  0.620947533244402;
    Leb_Grid_XYZW[  254][3] =  0.000912872013860;

    Leb_Grid_XYZW[  255][0] = -0.620947533244402;
    Leb_Grid_XYZW[  255][1] = -0.478380938076952;
    Leb_Grid_XYZW[  255][2] =  0.620947533244402;
    Leb_Grid_XYZW[  255][3] =  0.000912872013860;

    Leb_Grid_XYZW[  256][0] = -0.620947533244402;
    Leb_Grid_XYZW[  256][1] =  0.478380938076952;
    Leb_Grid_XYZW[  256][2] = -0.620947533244402;
    Leb_Grid_XYZW[  256][3] =  0.000912872013860;

    Leb_Grid_XYZW[  257][0] = -0.620947533244402;
    Leb_Grid_XYZW[  257][1] = -0.478380938076952;
    Leb_Grid_XYZW[  257][2] = -0.620947533244402;
    Leb_Grid_XYZW[  257][3] =  0.000912872013860;

    Leb_Grid_XYZW[  258][0] =  0.478380938076952;
    Leb_Grid_XYZW[  258][1] =  0.620947533244402;
    Leb_Grid_XYZW[  258][2] =  0.620947533244402;
    Leb_Grid_XYZW[  258][3] =  0.000912872013860;

    Leb_Grid_XYZW[  259][0] = -0.478380938076952;
    Leb_Grid_XYZW[  259][1] =  0.620947533244402;
    Leb_Grid_XYZW[  259][2] =  0.620947533244402;
    Leb_Grid_XYZW[  259][3] =  0.000912872013860;

    Leb_Grid_XYZW[  260][0] =  0.478380938076952;
    Leb_Grid_XYZW[  260][1] =  0.620947533244402;
    Leb_Grid_XYZW[  260][2] = -0.620947533244402;
    Leb_Grid_XYZW[  260][3] =  0.000912872013860;

    Leb_Grid_XYZW[  261][0] = -0.478380938076952;
    Leb_Grid_XYZW[  261][1] =  0.620947533244402;
    Leb_Grid_XYZW[  261][2] = -0.620947533244402;
    Leb_Grid_XYZW[  261][3] =  0.000912872013860;

    Leb_Grid_XYZW[  262][0] =  0.478380938076952;
    Leb_Grid_XYZW[  262][1] = -0.620947533244402;
    Leb_Grid_XYZW[  262][2] =  0.620947533244402;
    Leb_Grid_XYZW[  262][3] =  0.000912872013860;

    Leb_Grid_XYZW[  263][0] = -0.478380938076952;
    Leb_Grid_XYZW[  263][1] = -0.620947533244402;
    Leb_Grid_XYZW[  263][2] =  0.620947533244402;
    Leb_Grid_XYZW[  263][3] =  0.000912872013860;

    Leb_Grid_XYZW[  264][0] =  0.478380938076952;
    Leb_Grid_XYZW[  264][1] = -0.620947533244402;
    Leb_Grid_XYZW[  264][2] = -0.620947533244402;
    Leb_Grid_XYZW[  264][3] =  0.000912872013860;

    Leb_Grid_XYZW[  265][0] = -0.478380938076952;
    Leb_Grid_XYZW[  265][1] = -0.620947533244402;
    Leb_Grid_XYZW[  265][2] = -0.620947533244402;
    Leb_Grid_XYZW[  265][3] =  0.000912872013860;

    Leb_Grid_XYZW[  266][0] =  0.656972271185729;
    Leb_Grid_XYZW[  266][1] =  0.656972271185729;
    Leb_Grid_XYZW[  266][2] =  0.369830866459426;
    Leb_Grid_XYZW[  266][3] =  0.000913071493569;

    Leb_Grid_XYZW[  267][0] =  0.656972271185729;
    Leb_Grid_XYZW[  267][1] =  0.656972271185729;
    Leb_Grid_XYZW[  267][2] = -0.369830866459426;
    Leb_Grid_XYZW[  267][3] =  0.000913071493569;

    Leb_Grid_XYZW[  268][0] =  0.656972271185729;
    Leb_Grid_XYZW[  268][1] = -0.656972271185729;
    Leb_Grid_XYZW[  268][2] =  0.369830866459426;
    Leb_Grid_XYZW[  268][3] =  0.000913071493569;

    Leb_Grid_XYZW[  269][0] =  0.656972271185729;
    Leb_Grid_XYZW[  269][1] = -0.656972271185729;
    Leb_Grid_XYZW[  269][2] = -0.369830866459426;
    Leb_Grid_XYZW[  269][3] =  0.000913071493569;

    Leb_Grid_XYZW[  270][0] = -0.656972271185729;
    Leb_Grid_XYZW[  270][1] =  0.656972271185729;
    Leb_Grid_XYZW[  270][2] =  0.369830866459426;
    Leb_Grid_XYZW[  270][3] =  0.000913071493569;

    Leb_Grid_XYZW[  271][0] = -0.656972271185729;
    Leb_Grid_XYZW[  271][1] =  0.656972271185729;
    Leb_Grid_XYZW[  271][2] = -0.369830866459426;
    Leb_Grid_XYZW[  271][3] =  0.000913071493569;

    Leb_Grid_XYZW[  272][0] = -0.656972271185729;
    Leb_Grid_XYZW[  272][1] = -0.656972271185729;
    Leb_Grid_XYZW[  272][2] =  0.369830866459426;
    Leb_Grid_XYZW[  272][3] =  0.000913071493569;

    Leb_Grid_XYZW[  273][0] = -0.656972271185729;
    Leb_Grid_XYZW[  273][1] = -0.656972271185729;
    Leb_Grid_XYZW[  273][2] = -0.369830866459426;
    Leb_Grid_XYZW[  273][3] =  0.000913071493569;

    Leb_Grid_XYZW[  274][0] =  0.656972271185729;
    Leb_Grid_XYZW[  274][1] =  0.369830866459426;
    Leb_Grid_XYZW[  274][2] =  0.656972271185729;
    Leb_Grid_XYZW[  274][3] =  0.000913071493569;

    Leb_Grid_XYZW[  275][0] =  0.656972271185729;
    Leb_Grid_XYZW[  275][1] = -0.369830866459426;
    Leb_Grid_XYZW[  275][2] =  0.656972271185729;
    Leb_Grid_XYZW[  275][3] =  0.000913071493569;

    Leb_Grid_XYZW[  276][0] =  0.656972271185729;
    Leb_Grid_XYZW[  276][1] =  0.369830866459426;
    Leb_Grid_XYZW[  276][2] = -0.656972271185729;
    Leb_Grid_XYZW[  276][3] =  0.000913071493569;

    Leb_Grid_XYZW[  277][0] =  0.656972271185729;
    Leb_Grid_XYZW[  277][1] = -0.369830866459426;
    Leb_Grid_XYZW[  277][2] = -0.656972271185729;
    Leb_Grid_XYZW[  277][3] =  0.000913071493569;

    Leb_Grid_XYZW[  278][0] = -0.656972271185729;
    Leb_Grid_XYZW[  278][1] =  0.369830866459426;
    Leb_Grid_XYZW[  278][2] =  0.656972271185729;
    Leb_Grid_XYZW[  278][3] =  0.000913071493569;

    Leb_Grid_XYZW[  279][0] = -0.656972271185729;
    Leb_Grid_XYZW[  279][1] = -0.369830866459426;
    Leb_Grid_XYZW[  279][2] =  0.656972271185729;
    Leb_Grid_XYZW[  279][3] =  0.000913071493569;

    Leb_Grid_XYZW[  280][0] = -0.656972271185729;
    Leb_Grid_XYZW[  280][1] =  0.369830866459426;
    Leb_Grid_XYZW[  280][2] = -0.656972271185729;
    Leb_Grid_XYZW[  280][3] =  0.000913071493569;

    Leb_Grid_XYZW[  281][0] = -0.656972271185729;
    Leb_Grid_XYZW[  281][1] = -0.369830866459426;
    Leb_Grid_XYZW[  281][2] = -0.656972271185729;
    Leb_Grid_XYZW[  281][3] =  0.000913071493569;

    Leb_Grid_XYZW[  282][0] =  0.369830866459426;
    Leb_Grid_XYZW[  282][1] =  0.656972271185729;
    Leb_Grid_XYZW[  282][2] =  0.656972271185729;
    Leb_Grid_XYZW[  282][3] =  0.000913071493569;

    Leb_Grid_XYZW[  283][0] = -0.369830866459426;
    Leb_Grid_XYZW[  283][1] =  0.656972271185729;
    Leb_Grid_XYZW[  283][2] =  0.656972271185729;
    Leb_Grid_XYZW[  283][3] =  0.000913071493569;

    Leb_Grid_XYZW[  284][0] =  0.369830866459426;
    Leb_Grid_XYZW[  284][1] =  0.656972271185729;
    Leb_Grid_XYZW[  284][2] = -0.656972271185729;
    Leb_Grid_XYZW[  284][3] =  0.000913071493569;

    Leb_Grid_XYZW[  285][0] = -0.369830866459426;
    Leb_Grid_XYZW[  285][1] =  0.656972271185729;
    Leb_Grid_XYZW[  285][2] = -0.656972271185729;
    Leb_Grid_XYZW[  285][3] =  0.000913071493569;

    Leb_Grid_XYZW[  286][0] =  0.369830866459426;
    Leb_Grid_XYZW[  286][1] = -0.656972271185729;
    Leb_Grid_XYZW[  286][2] =  0.656972271185729;
    Leb_Grid_XYZW[  286][3] =  0.000913071493569;

    Leb_Grid_XYZW[  287][0] = -0.369830866459426;
    Leb_Grid_XYZW[  287][1] = -0.656972271185729;
    Leb_Grid_XYZW[  287][2] =  0.656972271185729;
    Leb_Grid_XYZW[  287][3] =  0.000913071493569;

    Leb_Grid_XYZW[  288][0] =  0.369830866459426;
    Leb_Grid_XYZW[  288][1] = -0.656972271185729;
    Leb_Grid_XYZW[  288][2] = -0.656972271185729;
    Leb_Grid_XYZW[  288][3] =  0.000913071493569;

    Leb_Grid_XYZW[  289][0] = -0.369830866459426;
    Leb_Grid_XYZW[  289][1] = -0.656972271185729;
    Leb_Grid_XYZW[  289][2] = -0.656972271185729;
    Leb_Grid_XYZW[  289][3] =  0.000913071493569;

    Leb_Grid_XYZW[  290][0] =  0.684178830907014;
    Leb_Grid_XYZW[  290][1] =  0.684178830907014;
    Leb_Grid_XYZW[  290][2] =  0.252583955700718;
    Leb_Grid_XYZW[  290][3] =  0.000915287378455;

    Leb_Grid_XYZW[  291][0] =  0.684178830907014;
    Leb_Grid_XYZW[  291][1] =  0.684178830907014;
    Leb_Grid_XYZW[  291][2] = -0.252583955700718;
    Leb_Grid_XYZW[  291][3] =  0.000915287378455;

    Leb_Grid_XYZW[  292][0] =  0.684178830907014;
    Leb_Grid_XYZW[  292][1] = -0.684178830907014;
    Leb_Grid_XYZW[  292][2] =  0.252583955700718;
    Leb_Grid_XYZW[  292][3] =  0.000915287378455;

    Leb_Grid_XYZW[  293][0] =  0.684178830907014;
    Leb_Grid_XYZW[  293][1] = -0.684178830907014;
    Leb_Grid_XYZW[  293][2] = -0.252583955700718;
    Leb_Grid_XYZW[  293][3] =  0.000915287378455;

    Leb_Grid_XYZW[  294][0] = -0.684178830907014;
    Leb_Grid_XYZW[  294][1] =  0.684178830907014;
    Leb_Grid_XYZW[  294][2] =  0.252583955700718;
    Leb_Grid_XYZW[  294][3] =  0.000915287378455;

    Leb_Grid_XYZW[  295][0] = -0.684178830907014;
    Leb_Grid_XYZW[  295][1] =  0.684178830907014;
    Leb_Grid_XYZW[  295][2] = -0.252583955700718;
    Leb_Grid_XYZW[  295][3] =  0.000915287378455;

    Leb_Grid_XYZW[  296][0] = -0.684178830907014;
    Leb_Grid_XYZW[  296][1] = -0.684178830907014;
    Leb_Grid_XYZW[  296][2] =  0.252583955700718;
    Leb_Grid_XYZW[  296][3] =  0.000915287378455;

    Leb_Grid_XYZW[  297][0] = -0.684178830907014;
    Leb_Grid_XYZW[  297][1] = -0.684178830907014;
    Leb_Grid_XYZW[  297][2] = -0.252583955700718;
    Leb_Grid_XYZW[  297][3] =  0.000915287378455;

    Leb_Grid_XYZW[  298][0] =  0.684178830907014;
    Leb_Grid_XYZW[  298][1] =  0.252583955700718;
    Leb_Grid_XYZW[  298][2] =  0.684178830907014;
    Leb_Grid_XYZW[  298][3] =  0.000915287378455;

    Leb_Grid_XYZW[  299][0] =  0.684178830907014;
    Leb_Grid_XYZW[  299][1] = -0.252583955700718;
    Leb_Grid_XYZW[  299][2] =  0.684178830907014;
    Leb_Grid_XYZW[  299][3] =  0.000915287378455;

    Leb_Grid_XYZW[  300][0] =  0.684178830907014;
    Leb_Grid_XYZW[  300][1] =  0.252583955700718;
    Leb_Grid_XYZW[  300][2] = -0.684178830907014;
    Leb_Grid_XYZW[  300][3] =  0.000915287378455;

    Leb_Grid_XYZW[  301][0] =  0.684178830907014;
    Leb_Grid_XYZW[  301][1] = -0.252583955700718;
    Leb_Grid_XYZW[  301][2] = -0.684178830907014;
    Leb_Grid_XYZW[  301][3] =  0.000915287378455;

    Leb_Grid_XYZW[  302][0] = -0.684178830907014;
    Leb_Grid_XYZW[  302][1] =  0.252583955700718;
    Leb_Grid_XYZW[  302][2] =  0.684178830907014;
    Leb_Grid_XYZW[  302][3] =  0.000915287378455;

    Leb_Grid_XYZW[  303][0] = -0.684178830907014;
    Leb_Grid_XYZW[  303][1] = -0.252583955700718;
    Leb_Grid_XYZW[  303][2] =  0.684178830907014;
    Leb_Grid_XYZW[  303][3] =  0.000915287378455;

    Leb_Grid_XYZW[  304][0] = -0.684178830907014;
    Leb_Grid_XYZW[  304][1] =  0.252583955700718;
    Leb_Grid_XYZW[  304][2] = -0.684178830907014;
    Leb_Grid_XYZW[  304][3] =  0.000915287378455;

    Leb_Grid_XYZW[  305][0] = -0.684178830907014;
    Leb_Grid_XYZW[  305][1] = -0.252583955700718;
    Leb_Grid_XYZW[  305][2] = -0.684178830907014;
    Leb_Grid_XYZW[  305][3] =  0.000915287378455;

    Leb_Grid_XYZW[  306][0] =  0.252583955700718;
    Leb_Grid_XYZW[  306][1] =  0.684178830907014;
    Leb_Grid_XYZW[  306][2] =  0.684178830907014;
    Leb_Grid_XYZW[  306][3] =  0.000915287378455;

    Leb_Grid_XYZW[  307][0] = -0.252583955700718;
    Leb_Grid_XYZW[  307][1] =  0.684178830907014;
    Leb_Grid_XYZW[  307][2] =  0.684178830907014;
    Leb_Grid_XYZW[  307][3] =  0.000915287378455;

    Leb_Grid_XYZW[  308][0] =  0.252583955700718;
    Leb_Grid_XYZW[  308][1] =  0.684178830907014;
    Leb_Grid_XYZW[  308][2] = -0.684178830907014;
    Leb_Grid_XYZW[  308][3] =  0.000915287378455;

    Leb_Grid_XYZW[  309][0] = -0.252583955700718;
    Leb_Grid_XYZW[  309][1] =  0.684178830907014;
    Leb_Grid_XYZW[  309][2] = -0.684178830907014;
    Leb_Grid_XYZW[  309][3] =  0.000915287378455;

    Leb_Grid_XYZW[  310][0] =  0.252583955700718;
    Leb_Grid_XYZW[  310][1] = -0.684178830907014;
    Leb_Grid_XYZW[  310][2] =  0.684178830907014;
    Leb_Grid_XYZW[  310][3] =  0.000915287378455;

    Leb_Grid_XYZW[  311][0] = -0.252583955700718;
    Leb_Grid_XYZW[  311][1] = -0.684178830907014;
    Leb_Grid_XYZW[  311][2] =  0.684178830907014;
    Leb_Grid_XYZW[  311][3] =  0.000915287378455;

    Leb_Grid_XYZW[  312][0] =  0.252583955700718;
    Leb_Grid_XYZW[  312][1] = -0.684178830907014;
    Leb_Grid_XYZW[  312][2] = -0.684178830907014;
    Leb_Grid_XYZW[  312][3] =  0.000915287378455;

    Leb_Grid_XYZW[  313][0] = -0.252583955700718;
    Leb_Grid_XYZW[  313][1] = -0.684178830907014;
    Leb_Grid_XYZW[  313][2] = -0.684178830907014;
    Leb_Grid_XYZW[  313][3] =  0.000915287378455;

    Leb_Grid_XYZW[  314][0] =  0.701260433012363;
    Leb_Grid_XYZW[  314][1] =  0.701260433012363;
    Leb_Grid_XYZW[  314][2] =  0.128326186659723;
    Leb_Grid_XYZW[  314][3] =  0.000918743627432;

    Leb_Grid_XYZW[  315][0] =  0.701260433012363;
    Leb_Grid_XYZW[  315][1] =  0.701260433012363;
    Leb_Grid_XYZW[  315][2] = -0.128326186659723;
    Leb_Grid_XYZW[  315][3] =  0.000918743627432;

    Leb_Grid_XYZW[  316][0] =  0.701260433012363;
    Leb_Grid_XYZW[  316][1] = -0.701260433012363;
    Leb_Grid_XYZW[  316][2] =  0.128326186659723;
    Leb_Grid_XYZW[  316][3] =  0.000918743627432;

    Leb_Grid_XYZW[  317][0] =  0.701260433012363;
    Leb_Grid_XYZW[  317][1] = -0.701260433012363;
    Leb_Grid_XYZW[  317][2] = -0.128326186659723;
    Leb_Grid_XYZW[  317][3] =  0.000918743627432;

    Leb_Grid_XYZW[  318][0] = -0.701260433012363;
    Leb_Grid_XYZW[  318][1] =  0.701260433012363;
    Leb_Grid_XYZW[  318][2] =  0.128326186659723;
    Leb_Grid_XYZW[  318][3] =  0.000918743627432;

    Leb_Grid_XYZW[  319][0] = -0.701260433012363;
    Leb_Grid_XYZW[  319][1] =  0.701260433012363;
    Leb_Grid_XYZW[  319][2] = -0.128326186659723;
    Leb_Grid_XYZW[  319][3] =  0.000918743627432;

    Leb_Grid_XYZW[  320][0] = -0.701260433012363;
    Leb_Grid_XYZW[  320][1] = -0.701260433012363;
    Leb_Grid_XYZW[  320][2] =  0.128326186659723;
    Leb_Grid_XYZW[  320][3] =  0.000918743627432;

    Leb_Grid_XYZW[  321][0] = -0.701260433012363;
    Leb_Grid_XYZW[  321][1] = -0.701260433012363;
    Leb_Grid_XYZW[  321][2] = -0.128326186659723;
    Leb_Grid_XYZW[  321][3] =  0.000918743627432;

    Leb_Grid_XYZW[  322][0] =  0.701260433012363;
    Leb_Grid_XYZW[  322][1] =  0.128326186659723;
    Leb_Grid_XYZW[  322][2] =  0.701260433012363;
    Leb_Grid_XYZW[  322][3] =  0.000918743627432;

    Leb_Grid_XYZW[  323][0] =  0.701260433012363;
    Leb_Grid_XYZW[  323][1] = -0.128326186659723;
    Leb_Grid_XYZW[  323][2] =  0.701260433012363;
    Leb_Grid_XYZW[  323][3] =  0.000918743627432;

    Leb_Grid_XYZW[  324][0] =  0.701260433012363;
    Leb_Grid_XYZW[  324][1] =  0.128326186659723;
    Leb_Grid_XYZW[  324][2] = -0.701260433012363;
    Leb_Grid_XYZW[  324][3] =  0.000918743627432;

    Leb_Grid_XYZW[  325][0] =  0.701260433012363;
    Leb_Grid_XYZW[  325][1] = -0.128326186659723;
    Leb_Grid_XYZW[  325][2] = -0.701260433012363;
    Leb_Grid_XYZW[  325][3] =  0.000918743627432;

    Leb_Grid_XYZW[  326][0] = -0.701260433012363;
    Leb_Grid_XYZW[  326][1] =  0.128326186659723;
    Leb_Grid_XYZW[  326][2] =  0.701260433012363;
    Leb_Grid_XYZW[  326][3] =  0.000918743627432;

    Leb_Grid_XYZW[  327][0] = -0.701260433012363;
    Leb_Grid_XYZW[  327][1] = -0.128326186659723;
    Leb_Grid_XYZW[  327][2] =  0.701260433012363;
    Leb_Grid_XYZW[  327][3] =  0.000918743627432;

    Leb_Grid_XYZW[  328][0] = -0.701260433012363;
    Leb_Grid_XYZW[  328][1] =  0.128326186659723;
    Leb_Grid_XYZW[  328][2] = -0.701260433012363;
    Leb_Grid_XYZW[  328][3] =  0.000918743627432;

    Leb_Grid_XYZW[  329][0] = -0.701260433012363;
    Leb_Grid_XYZW[  329][1] = -0.128326186659723;
    Leb_Grid_XYZW[  329][2] = -0.701260433012363;
    Leb_Grid_XYZW[  329][3] =  0.000918743627432;

    Leb_Grid_XYZW[  330][0] =  0.128326186659723;
    Leb_Grid_XYZW[  330][1] =  0.701260433012363;
    Leb_Grid_XYZW[  330][2] =  0.701260433012363;
    Leb_Grid_XYZW[  330][3] =  0.000918743627432;

    Leb_Grid_XYZW[  331][0] = -0.128326186659723;
    Leb_Grid_XYZW[  331][1] =  0.701260433012363;
    Leb_Grid_XYZW[  331][2] =  0.701260433012363;
    Leb_Grid_XYZW[  331][3] =  0.000918743627432;

    Leb_Grid_XYZW[  332][0] =  0.128326186659723;
    Leb_Grid_XYZW[  332][1] =  0.701260433012363;
    Leb_Grid_XYZW[  332][2] = -0.701260433012363;
    Leb_Grid_XYZW[  332][3] =  0.000918743627432;

    Leb_Grid_XYZW[  333][0] = -0.128326186659723;
    Leb_Grid_XYZW[  333][1] =  0.701260433012363;
    Leb_Grid_XYZW[  333][2] = -0.701260433012363;
    Leb_Grid_XYZW[  333][3] =  0.000918743627432;

    Leb_Grid_XYZW[  334][0] =  0.128326186659723;
    Leb_Grid_XYZW[  334][1] = -0.701260433012363;
    Leb_Grid_XYZW[  334][2] =  0.701260433012363;
    Leb_Grid_XYZW[  334][3] =  0.000918743627432;

    Leb_Grid_XYZW[  335][0] = -0.128326186659723;
    Leb_Grid_XYZW[  335][1] = -0.701260433012363;
    Leb_Grid_XYZW[  335][2] =  0.701260433012363;
    Leb_Grid_XYZW[  335][3] =  0.000918743627432;

    Leb_Grid_XYZW[  336][0] =  0.128326186659723;
    Leb_Grid_XYZW[  336][1] = -0.701260433012363;
    Leb_Grid_XYZW[  336][2] = -0.701260433012363;
    Leb_Grid_XYZW[  336][3] =  0.000918743627432;

    Leb_Grid_XYZW[  337][0] = -0.128326186659723;
    Leb_Grid_XYZW[  337][1] = -0.701260433012363;
    Leb_Grid_XYZW[  337][2] = -0.701260433012363;
    Leb_Grid_XYZW[  337][3] =  0.000918743627432;

    Leb_Grid_XYZW[  338][0] =  0.107238221547817;
    Leb_Grid_XYZW[  338][1] =  0.994233354821322;
    Leb_Grid_XYZW[  338][2] =  0.000000000000000;
    Leb_Grid_XYZW[  338][3] =  0.000517697731297;

    Leb_Grid_XYZW[  339][0] =  0.107238221547817;
    Leb_Grid_XYZW[  339][1] = -0.994233354821322;
    Leb_Grid_XYZW[  339][2] =  0.000000000000000;
    Leb_Grid_XYZW[  339][3] =  0.000517697731297;

    Leb_Grid_XYZW[  340][0] = -0.107238221547817;
    Leb_Grid_XYZW[  340][1] =  0.994233354821322;
    Leb_Grid_XYZW[  340][2] =  0.000000000000000;
    Leb_Grid_XYZW[  340][3] =  0.000517697731297;

    Leb_Grid_XYZW[  341][0] = -0.107238221547817;
    Leb_Grid_XYZW[  341][1] = -0.994233354821322;
    Leb_Grid_XYZW[  341][2] =  0.000000000000000;
    Leb_Grid_XYZW[  341][3] =  0.000517697731297;

    Leb_Grid_XYZW[  342][0] =  0.994233354821322;
    Leb_Grid_XYZW[  342][1] =  0.107238221547816;
    Leb_Grid_XYZW[  342][2] =  0.000000000000000;
    Leb_Grid_XYZW[  342][3] =  0.000517697731297;

    Leb_Grid_XYZW[  343][0] =  0.994233354821322;
    Leb_Grid_XYZW[  343][1] = -0.107238221547816;
    Leb_Grid_XYZW[  343][2] =  0.000000000000000;
    Leb_Grid_XYZW[  343][3] =  0.000517697731297;

    Leb_Grid_XYZW[  344][0] = -0.994233354821322;
    Leb_Grid_XYZW[  344][1] =  0.107238221547816;
    Leb_Grid_XYZW[  344][2] =  0.000000000000000;
    Leb_Grid_XYZW[  344][3] =  0.000517697731297;

    Leb_Grid_XYZW[  345][0] = -0.994233354821322;
    Leb_Grid_XYZW[  345][1] = -0.107238221547816;
    Leb_Grid_XYZW[  345][2] =  0.000000000000000;
    Leb_Grid_XYZW[  345][3] =  0.000517697731297;

    Leb_Grid_XYZW[  346][0] =  0.107238221547816;
    Leb_Grid_XYZW[  346][1] =  0.000000000000000;
    Leb_Grid_XYZW[  346][2] =  0.994233354821322;
    Leb_Grid_XYZW[  346][3] =  0.000517697731297;

    Leb_Grid_XYZW[  347][0] =  0.107238221547816;
    Leb_Grid_XYZW[  347][1] =  0.000000000000000;
    Leb_Grid_XYZW[  347][2] = -0.994233354821322;
    Leb_Grid_XYZW[  347][3] =  0.000517697731297;

    Leb_Grid_XYZW[  348][0] = -0.107238221547816;
    Leb_Grid_XYZW[  348][1] =  0.000000000000000;
    Leb_Grid_XYZW[  348][2] =  0.994233354821322;
    Leb_Grid_XYZW[  348][3] =  0.000517697731297;

    Leb_Grid_XYZW[  349][0] = -0.107238221547816;
    Leb_Grid_XYZW[  349][1] =  0.000000000000000;
    Leb_Grid_XYZW[  349][2] = -0.994233354821322;
    Leb_Grid_XYZW[  349][3] =  0.000517697731297;

    Leb_Grid_XYZW[  350][0] =  0.994233354821322;
    Leb_Grid_XYZW[  350][1] =  0.000000000000000;
    Leb_Grid_XYZW[  350][2] =  0.107238221547817;
    Leb_Grid_XYZW[  350][3] =  0.000517697731297;

    Leb_Grid_XYZW[  351][0] =  0.994233354821322;
    Leb_Grid_XYZW[  351][1] =  0.000000000000000;
    Leb_Grid_XYZW[  351][2] = -0.107238221547817;
    Leb_Grid_XYZW[  351][3] =  0.000517697731297;

    Leb_Grid_XYZW[  352][0] = -0.994233354821322;
    Leb_Grid_XYZW[  352][1] =  0.000000000000000;
    Leb_Grid_XYZW[  352][2] =  0.107238221547817;
    Leb_Grid_XYZW[  352][3] =  0.000517697731297;

    Leb_Grid_XYZW[  353][0] = -0.994233354821322;
    Leb_Grid_XYZW[  353][1] =  0.000000000000000;
    Leb_Grid_XYZW[  353][2] = -0.107238221547817;
    Leb_Grid_XYZW[  353][3] =  0.000517697731297;

    Leb_Grid_XYZW[  354][0] =  0.000000000000000;
    Leb_Grid_XYZW[  354][1] =  0.107238221547816;
    Leb_Grid_XYZW[  354][2] =  0.994233354821322;
    Leb_Grid_XYZW[  354][3] =  0.000517697731297;

    Leb_Grid_XYZW[  355][0] =  0.000000000000000;
    Leb_Grid_XYZW[  355][1] =  0.107238221547816;
    Leb_Grid_XYZW[  355][2] = -0.994233354821322;
    Leb_Grid_XYZW[  355][3] =  0.000517697731297;

    Leb_Grid_XYZW[  356][0] =  0.000000000000000;
    Leb_Grid_XYZW[  356][1] = -0.107238221547816;
    Leb_Grid_XYZW[  356][2] =  0.994233354821322;
    Leb_Grid_XYZW[  356][3] =  0.000517697731297;

    Leb_Grid_XYZW[  357][0] =  0.000000000000000;
    Leb_Grid_XYZW[  357][1] = -0.107238221547816;
    Leb_Grid_XYZW[  357][2] = -0.994233354821322;
    Leb_Grid_XYZW[  357][3] =  0.000517697731297;

    Leb_Grid_XYZW[  358][0] =  0.000000000000000;
    Leb_Grid_XYZW[  358][1] =  0.994233354821322;
    Leb_Grid_XYZW[  358][2] =  0.107238221547817;
    Leb_Grid_XYZW[  358][3] =  0.000517697731297;

    Leb_Grid_XYZW[  359][0] =  0.000000000000000;
    Leb_Grid_XYZW[  359][1] =  0.994233354821322;
    Leb_Grid_XYZW[  359][2] = -0.107238221547817;
    Leb_Grid_XYZW[  359][3] =  0.000517697731297;

    Leb_Grid_XYZW[  360][0] =  0.000000000000000;
    Leb_Grid_XYZW[  360][1] = -0.994233354821322;
    Leb_Grid_XYZW[  360][2] =  0.107238221547817;
    Leb_Grid_XYZW[  360][3] =  0.000517697731297;

    Leb_Grid_XYZW[  361][0] =  0.000000000000000;
    Leb_Grid_XYZW[  361][1] = -0.994233354821322;
    Leb_Grid_XYZW[  361][2] = -0.107238221547817;
    Leb_Grid_XYZW[  361][3] =  0.000517697731297;

    Leb_Grid_XYZW[  362][0] =  0.258206895949697;
    Leb_Grid_XYZW[  362][1] =  0.966089643296119;
    Leb_Grid_XYZW[  362][2] =  0.000000000000000;
    Leb_Grid_XYZW[  362][3] =  0.000733114368210;

    Leb_Grid_XYZW[  363][0] =  0.258206895949697;
    Leb_Grid_XYZW[  363][1] = -0.966089643296119;
    Leb_Grid_XYZW[  363][2] =  0.000000000000000;
    Leb_Grid_XYZW[  363][3] =  0.000733114368210;

    Leb_Grid_XYZW[  364][0] = -0.258206895949697;
    Leb_Grid_XYZW[  364][1] =  0.966089643296119;
    Leb_Grid_XYZW[  364][2] =  0.000000000000000;
    Leb_Grid_XYZW[  364][3] =  0.000733114368210;

    Leb_Grid_XYZW[  365][0] = -0.258206895949697;
    Leb_Grid_XYZW[  365][1] = -0.966089643296119;
    Leb_Grid_XYZW[  365][2] =  0.000000000000000;
    Leb_Grid_XYZW[  365][3] =  0.000733114368210;

    Leb_Grid_XYZW[  366][0] =  0.966089643296119;
    Leb_Grid_XYZW[  366][1] =  0.258206895949697;
    Leb_Grid_XYZW[  366][2] =  0.000000000000000;
    Leb_Grid_XYZW[  366][3] =  0.000733114368210;

    Leb_Grid_XYZW[  367][0] =  0.966089643296119;
    Leb_Grid_XYZW[  367][1] = -0.258206895949697;
    Leb_Grid_XYZW[  367][2] =  0.000000000000000;
    Leb_Grid_XYZW[  367][3] =  0.000733114368210;

    Leb_Grid_XYZW[  368][0] = -0.966089643296119;
    Leb_Grid_XYZW[  368][1] =  0.258206895949697;
    Leb_Grid_XYZW[  368][2] =  0.000000000000000;
    Leb_Grid_XYZW[  368][3] =  0.000733114368210;

    Leb_Grid_XYZW[  369][0] = -0.966089643296119;
    Leb_Grid_XYZW[  369][1] = -0.258206895949697;
    Leb_Grid_XYZW[  369][2] =  0.000000000000000;
    Leb_Grid_XYZW[  369][3] =  0.000733114368210;

    Leb_Grid_XYZW[  370][0] =  0.258206895949697;
    Leb_Grid_XYZW[  370][1] =  0.000000000000000;
    Leb_Grid_XYZW[  370][2] =  0.966089643296119;
    Leb_Grid_XYZW[  370][3] =  0.000733114368210;

    Leb_Grid_XYZW[  371][0] =  0.258206895949697;
    Leb_Grid_XYZW[  371][1] =  0.000000000000000;
    Leb_Grid_XYZW[  371][2] = -0.966089643296119;
    Leb_Grid_XYZW[  371][3] =  0.000733114368210;

    Leb_Grid_XYZW[  372][0] = -0.258206895949697;
    Leb_Grid_XYZW[  372][1] =  0.000000000000000;
    Leb_Grid_XYZW[  372][2] =  0.966089643296119;
    Leb_Grid_XYZW[  372][3] =  0.000733114368210;

    Leb_Grid_XYZW[  373][0] = -0.258206895949697;
    Leb_Grid_XYZW[  373][1] =  0.000000000000000;
    Leb_Grid_XYZW[  373][2] = -0.966089643296119;
    Leb_Grid_XYZW[  373][3] =  0.000733114368210;

    Leb_Grid_XYZW[  374][0] =  0.966089643296119;
    Leb_Grid_XYZW[  374][1] =  0.000000000000000;
    Leb_Grid_XYZW[  374][2] =  0.258206895949697;
    Leb_Grid_XYZW[  374][3] =  0.000733114368210;

    Leb_Grid_XYZW[  375][0] =  0.966089643296119;
    Leb_Grid_XYZW[  375][1] =  0.000000000000000;
    Leb_Grid_XYZW[  375][2] = -0.258206895949697;
    Leb_Grid_XYZW[  375][3] =  0.000733114368210;

    Leb_Grid_XYZW[  376][0] = -0.966089643296119;
    Leb_Grid_XYZW[  376][1] =  0.000000000000000;
    Leb_Grid_XYZW[  376][2] =  0.258206895949697;
    Leb_Grid_XYZW[  376][3] =  0.000733114368210;

    Leb_Grid_XYZW[  377][0] = -0.966089643296119;
    Leb_Grid_XYZW[  377][1] =  0.000000000000000;
    Leb_Grid_XYZW[  377][2] = -0.258206895949697;
    Leb_Grid_XYZW[  377][3] =  0.000733114368210;

    Leb_Grid_XYZW[  378][0] =  0.000000000000000;
    Leb_Grid_XYZW[  378][1] =  0.258206895949697;
    Leb_Grid_XYZW[  378][2] =  0.966089643296119;
    Leb_Grid_XYZW[  378][3] =  0.000733114368210;

    Leb_Grid_XYZW[  379][0] =  0.000000000000000;
    Leb_Grid_XYZW[  379][1] =  0.258206895949697;
    Leb_Grid_XYZW[  379][2] = -0.966089643296119;
    Leb_Grid_XYZW[  379][3] =  0.000733114368210;

    Leb_Grid_XYZW[  380][0] =  0.000000000000000;
    Leb_Grid_XYZW[  380][1] = -0.258206895949697;
    Leb_Grid_XYZW[  380][2] =  0.966089643296119;
    Leb_Grid_XYZW[  380][3] =  0.000733114368210;

    Leb_Grid_XYZW[  381][0] =  0.000000000000000;
    Leb_Grid_XYZW[  381][1] = -0.258206895949697;
    Leb_Grid_XYZW[  381][2] = -0.966089643296119;
    Leb_Grid_XYZW[  381][3] =  0.000733114368210;

    Leb_Grid_XYZW[  382][0] =  0.000000000000000;
    Leb_Grid_XYZW[  382][1] =  0.966089643296119;
    Leb_Grid_XYZW[  382][2] =  0.258206895949697;
    Leb_Grid_XYZW[  382][3] =  0.000733114368210;

    Leb_Grid_XYZW[  383][0] =  0.000000000000000;
    Leb_Grid_XYZW[  383][1] =  0.966089643296119;
    Leb_Grid_XYZW[  383][2] = -0.258206895949697;
    Leb_Grid_XYZW[  383][3] =  0.000733114368210;

    Leb_Grid_XYZW[  384][0] =  0.000000000000000;
    Leb_Grid_XYZW[  384][1] = -0.966089643296119;
    Leb_Grid_XYZW[  384][2] =  0.258206895949697;
    Leb_Grid_XYZW[  384][3] =  0.000733114368210;

    Leb_Grid_XYZW[  385][0] =  0.000000000000000;
    Leb_Grid_XYZW[  385][1] = -0.966089643296119;
    Leb_Grid_XYZW[  385][2] = -0.258206895949697;
    Leb_Grid_XYZW[  385][3] =  0.000733114368210;

    Leb_Grid_XYZW[  386][0] =  0.417275295530672;
    Leb_Grid_XYZW[  386][1] =  0.908780131681911;
    Leb_Grid_XYZW[  386][2] =  0.000000000000000;
    Leb_Grid_XYZW[  386][3] =  0.000846323283638;

    Leb_Grid_XYZW[  387][0] =  0.417275295530672;
    Leb_Grid_XYZW[  387][1] = -0.908780131681911;
    Leb_Grid_XYZW[  387][2] =  0.000000000000000;
    Leb_Grid_XYZW[  387][3] =  0.000846323283638;

    Leb_Grid_XYZW[  388][0] = -0.417275295530672;
    Leb_Grid_XYZW[  388][1] =  0.908780131681910;
    Leb_Grid_XYZW[  388][2] =  0.000000000000000;
    Leb_Grid_XYZW[  388][3] =  0.000846323283638;

    Leb_Grid_XYZW[  389][0] = -0.417275295530672;
    Leb_Grid_XYZW[  389][1] = -0.908780131681910;
    Leb_Grid_XYZW[  389][2] =  0.000000000000000;
    Leb_Grid_XYZW[  389][3] =  0.000846323283638;

    Leb_Grid_XYZW[  390][0] =  0.908780131681911;
    Leb_Grid_XYZW[  390][1] =  0.417275295530672;
    Leb_Grid_XYZW[  390][2] =  0.000000000000000;
    Leb_Grid_XYZW[  390][3] =  0.000846323283638;

    Leb_Grid_XYZW[  391][0] =  0.908780131681911;
    Leb_Grid_XYZW[  391][1] = -0.417275295530672;
    Leb_Grid_XYZW[  391][2] =  0.000000000000000;
    Leb_Grid_XYZW[  391][3] =  0.000846323283638;

    Leb_Grid_XYZW[  392][0] = -0.908780131681911;
    Leb_Grid_XYZW[  392][1] =  0.417275295530671;
    Leb_Grid_XYZW[  392][2] =  0.000000000000000;
    Leb_Grid_XYZW[  392][3] =  0.000846323283638;

    Leb_Grid_XYZW[  393][0] = -0.908780131681911;
    Leb_Grid_XYZW[  393][1] = -0.417275295530671;
    Leb_Grid_XYZW[  393][2] =  0.000000000000000;
    Leb_Grid_XYZW[  393][3] =  0.000846323283638;

    Leb_Grid_XYZW[  394][0] =  0.417275295530672;
    Leb_Grid_XYZW[  394][1] =  0.000000000000000;
    Leb_Grid_XYZW[  394][2] =  0.908780131681911;
    Leb_Grid_XYZW[  394][3] =  0.000846323283638;

    Leb_Grid_XYZW[  395][0] =  0.417275295530671;
    Leb_Grid_XYZW[  395][1] =  0.000000000000000;
    Leb_Grid_XYZW[  395][2] = -0.908780131681911;
    Leb_Grid_XYZW[  395][3] =  0.000846323283638;

    Leb_Grid_XYZW[  396][0] = -0.417275295530672;
    Leb_Grid_XYZW[  396][1] =  0.000000000000000;
    Leb_Grid_XYZW[  396][2] =  0.908780131681911;
    Leb_Grid_XYZW[  396][3] =  0.000846323283638;

    Leb_Grid_XYZW[  397][0] = -0.417275295530671;
    Leb_Grid_XYZW[  397][1] =  0.000000000000000;
    Leb_Grid_XYZW[  397][2] = -0.908780131681911;
    Leb_Grid_XYZW[  397][3] =  0.000846323283638;

    Leb_Grid_XYZW[  398][0] =  0.908780131681911;
    Leb_Grid_XYZW[  398][1] =  0.000000000000000;
    Leb_Grid_XYZW[  398][2] =  0.417275295530672;
    Leb_Grid_XYZW[  398][3] =  0.000846323283638;

    Leb_Grid_XYZW[  399][0] =  0.908780131681910;
    Leb_Grid_XYZW[  399][1] =  0.000000000000000;
    Leb_Grid_XYZW[  399][2] = -0.417275295530672;
    Leb_Grid_XYZW[  399][3] =  0.000846323283638;

    Leb_Grid_XYZW[  400][0] = -0.908780131681911;
    Leb_Grid_XYZW[  400][1] =  0.000000000000000;
    Leb_Grid_XYZW[  400][2] =  0.417275295530672;
    Leb_Grid_XYZW[  400][3] =  0.000846323283638;

    Leb_Grid_XYZW[  401][0] = -0.908780131681910;
    Leb_Grid_XYZW[  401][1] =  0.000000000000000;
    Leb_Grid_XYZW[  401][2] = -0.417275295530672;
    Leb_Grid_XYZW[  401][3] =  0.000846323283638;

    Leb_Grid_XYZW[  402][0] =  0.000000000000000;
    Leb_Grid_XYZW[  402][1] =  0.417275295530672;
    Leb_Grid_XYZW[  402][2] =  0.908780131681911;
    Leb_Grid_XYZW[  402][3] =  0.000846323283638;

    Leb_Grid_XYZW[  403][0] =  0.000000000000000;
    Leb_Grid_XYZW[  403][1] =  0.417275295530671;
    Leb_Grid_XYZW[  403][2] = -0.908780131681911;
    Leb_Grid_XYZW[  403][3] =  0.000846323283638;

    Leb_Grid_XYZW[  404][0] =  0.000000000000000;
    Leb_Grid_XYZW[  404][1] = -0.417275295530672;
    Leb_Grid_XYZW[  404][2] =  0.908780131681911;
    Leb_Grid_XYZW[  404][3] =  0.000846323283638;

    Leb_Grid_XYZW[  405][0] =  0.000000000000000;
    Leb_Grid_XYZW[  405][1] = -0.417275295530671;
    Leb_Grid_XYZW[  405][2] = -0.908780131681911;
    Leb_Grid_XYZW[  405][3] =  0.000846323283638;

    Leb_Grid_XYZW[  406][0] =  0.000000000000000;
    Leb_Grid_XYZW[  406][1] =  0.908780131681911;
    Leb_Grid_XYZW[  406][2] =  0.417275295530672;
    Leb_Grid_XYZW[  406][3] =  0.000846323283638;

    Leb_Grid_XYZW[  407][0] =  0.000000000000000;
    Leb_Grid_XYZW[  407][1] =  0.908780131681910;
    Leb_Grid_XYZW[  407][2] = -0.417275295530672;
    Leb_Grid_XYZW[  407][3] =  0.000846323283638;

    Leb_Grid_XYZW[  408][0] =  0.000000000000000;
    Leb_Grid_XYZW[  408][1] = -0.908780131681911;
    Leb_Grid_XYZW[  408][2] =  0.417275295530672;
    Leb_Grid_XYZW[  408][3] =  0.000846323283638;

    Leb_Grid_XYZW[  409][0] =  0.000000000000000;
    Leb_Grid_XYZW[  409][1] = -0.908780131681910;
    Leb_Grid_XYZW[  409][2] = -0.417275295530672;
    Leb_Grid_XYZW[  409][3] =  0.000846323283638;

    Leb_Grid_XYZW[  410][0] =  0.570036691179250;
    Leb_Grid_XYZW[  410][1] =  0.821619237061433;
    Leb_Grid_XYZW[  410][2] =  0.000000000000000;
    Leb_Grid_XYZW[  410][3] =  0.000903112269425;

    Leb_Grid_XYZW[  411][0] =  0.570036691179250;
    Leb_Grid_XYZW[  411][1] = -0.821619237061433;
    Leb_Grid_XYZW[  411][2] =  0.000000000000000;
    Leb_Grid_XYZW[  411][3] =  0.000903112269425;

    Leb_Grid_XYZW[  412][0] = -0.570036691179250;
    Leb_Grid_XYZW[  412][1] =  0.821619237061433;
    Leb_Grid_XYZW[  412][2] =  0.000000000000000;
    Leb_Grid_XYZW[  412][3] =  0.000903112269425;

    Leb_Grid_XYZW[  413][0] = -0.570036691179250;
    Leb_Grid_XYZW[  413][1] = -0.821619237061433;
    Leb_Grid_XYZW[  413][2] =  0.000000000000000;
    Leb_Grid_XYZW[  413][3] =  0.000903112269425;

    Leb_Grid_XYZW[  414][0] =  0.821619237061433;
    Leb_Grid_XYZW[  414][1] =  0.570036691179250;
    Leb_Grid_XYZW[  414][2] =  0.000000000000000;
    Leb_Grid_XYZW[  414][3] =  0.000903112269425;

    Leb_Grid_XYZW[  415][0] =  0.821619237061433;
    Leb_Grid_XYZW[  415][1] = -0.570036691179250;
    Leb_Grid_XYZW[  415][2] =  0.000000000000000;
    Leb_Grid_XYZW[  415][3] =  0.000903112269425;

    Leb_Grid_XYZW[  416][0] = -0.821619237061433;
    Leb_Grid_XYZW[  416][1] =  0.570036691179250;
    Leb_Grid_XYZW[  416][2] =  0.000000000000000;
    Leb_Grid_XYZW[  416][3] =  0.000903112269425;

    Leb_Grid_XYZW[  417][0] = -0.821619237061433;
    Leb_Grid_XYZW[  417][1] = -0.570036691179250;
    Leb_Grid_XYZW[  417][2] =  0.000000000000000;
    Leb_Grid_XYZW[  417][3] =  0.000903112269425;

    Leb_Grid_XYZW[  418][0] =  0.570036691179250;
    Leb_Grid_XYZW[  418][1] =  0.000000000000000;
    Leb_Grid_XYZW[  418][2] =  0.821619237061433;
    Leb_Grid_XYZW[  418][3] =  0.000903112269425;

    Leb_Grid_XYZW[  419][0] =  0.570036691179250;
    Leb_Grid_XYZW[  419][1] =  0.000000000000000;
    Leb_Grid_XYZW[  419][2] = -0.821619237061433;
    Leb_Grid_XYZW[  419][3] =  0.000903112269425;

    Leb_Grid_XYZW[  420][0] = -0.570036691179250;
    Leb_Grid_XYZW[  420][1] =  0.000000000000000;
    Leb_Grid_XYZW[  420][2] =  0.821619237061433;
    Leb_Grid_XYZW[  420][3] =  0.000903112269425;

    Leb_Grid_XYZW[  421][0] = -0.570036691179250;
    Leb_Grid_XYZW[  421][1] =  0.000000000000000;
    Leb_Grid_XYZW[  421][2] = -0.821619237061433;
    Leb_Grid_XYZW[  421][3] =  0.000903112269425;

    Leb_Grid_XYZW[  422][0] =  0.821619237061433;
    Leb_Grid_XYZW[  422][1] =  0.000000000000000;
    Leb_Grid_XYZW[  422][2] =  0.570036691179250;
    Leb_Grid_XYZW[  422][3] =  0.000903112269425;

    Leb_Grid_XYZW[  423][0] =  0.821619237061433;
    Leb_Grid_XYZW[  423][1] =  0.000000000000000;
    Leb_Grid_XYZW[  423][2] = -0.570036691179250;
    Leb_Grid_XYZW[  423][3] =  0.000903112269425;

    Leb_Grid_XYZW[  424][0] = -0.821619237061433;
    Leb_Grid_XYZW[  424][1] =  0.000000000000000;
    Leb_Grid_XYZW[  424][2] =  0.570036691179250;
    Leb_Grid_XYZW[  424][3] =  0.000903112269425;

    Leb_Grid_XYZW[  425][0] = -0.821619237061433;
    Leb_Grid_XYZW[  425][1] =  0.000000000000000;
    Leb_Grid_XYZW[  425][2] = -0.570036691179250;
    Leb_Grid_XYZW[  425][3] =  0.000903112269425;

    Leb_Grid_XYZW[  426][0] =  0.000000000000000;
    Leb_Grid_XYZW[  426][1] =  0.570036691179250;
    Leb_Grid_XYZW[  426][2] =  0.821619237061433;
    Leb_Grid_XYZW[  426][3] =  0.000903112269425;

    Leb_Grid_XYZW[  427][0] =  0.000000000000000;
    Leb_Grid_XYZW[  427][1] =  0.570036691179250;
    Leb_Grid_XYZW[  427][2] = -0.821619237061433;
    Leb_Grid_XYZW[  427][3] =  0.000903112269425;

    Leb_Grid_XYZW[  428][0] =  0.000000000000000;
    Leb_Grid_XYZW[  428][1] = -0.570036691179250;
    Leb_Grid_XYZW[  428][2] =  0.821619237061433;
    Leb_Grid_XYZW[  428][3] =  0.000903112269425;

    Leb_Grid_XYZW[  429][0] =  0.000000000000000;
    Leb_Grid_XYZW[  429][1] = -0.570036691179250;
    Leb_Grid_XYZW[  429][2] = -0.821619237061433;
    Leb_Grid_XYZW[  429][3] =  0.000903112269425;

    Leb_Grid_XYZW[  430][0] =  0.000000000000000;
    Leb_Grid_XYZW[  430][1] =  0.821619237061433;
    Leb_Grid_XYZW[  430][2] =  0.570036691179250;
    Leb_Grid_XYZW[  430][3] =  0.000903112269425;

    Leb_Grid_XYZW[  431][0] =  0.000000000000000;
    Leb_Grid_XYZW[  431][1] =  0.821619237061433;
    Leb_Grid_XYZW[  431][2] = -0.570036691179250;
    Leb_Grid_XYZW[  431][3] =  0.000903112269425;

    Leb_Grid_XYZW[  432][0] =  0.000000000000000;
    Leb_Grid_XYZW[  432][1] = -0.821619237061433;
    Leb_Grid_XYZW[  432][2] =  0.570036691179250;
    Leb_Grid_XYZW[  432][3] =  0.000903112269425;

    Leb_Grid_XYZW[  433][0] =  0.000000000000000;
    Leb_Grid_XYZW[  433][1] = -0.821619237061433;
    Leb_Grid_XYZW[  433][2] = -0.570036691179250;
    Leb_Grid_XYZW[  433][3] =  0.000903112269425;

    Leb_Grid_XYZW[  434][0] =  0.982798601826395;
    Leb_Grid_XYZW[  434][1] =  0.177177402261533;
    Leb_Grid_XYZW[  434][2] =  0.052106394770113;
    Leb_Grid_XYZW[  434][3] =  0.000648577845316;

    Leb_Grid_XYZW[  435][0] =  0.982798601826395;
    Leb_Grid_XYZW[  435][1] =  0.177177402261533;
    Leb_Grid_XYZW[  435][2] = -0.052106394770113;
    Leb_Grid_XYZW[  435][3] =  0.000648577845316;

    Leb_Grid_XYZW[  436][0] =  0.982798601826395;
    Leb_Grid_XYZW[  436][1] = -0.177177402261533;
    Leb_Grid_XYZW[  436][2] =  0.052106394770113;
    Leb_Grid_XYZW[  436][3] =  0.000648577845316;

    Leb_Grid_XYZW[  437][0] =  0.982798601826395;
    Leb_Grid_XYZW[  437][1] = -0.177177402261533;
    Leb_Grid_XYZW[  437][2] = -0.052106394770113;
    Leb_Grid_XYZW[  437][3] =  0.000648577845316;

    Leb_Grid_XYZW[  438][0] = -0.982798601826395;
    Leb_Grid_XYZW[  438][1] =  0.177177402261533;
    Leb_Grid_XYZW[  438][2] =  0.052106394770113;
    Leb_Grid_XYZW[  438][3] =  0.000648577845316;

    Leb_Grid_XYZW[  439][0] = -0.982798601826395;
    Leb_Grid_XYZW[  439][1] =  0.177177402261533;
    Leb_Grid_XYZW[  439][2] = -0.052106394770113;
    Leb_Grid_XYZW[  439][3] =  0.000648577845316;

    Leb_Grid_XYZW[  440][0] = -0.982798601826395;
    Leb_Grid_XYZW[  440][1] = -0.177177402261533;
    Leb_Grid_XYZW[  440][2] =  0.052106394770113;
    Leb_Grid_XYZW[  440][3] =  0.000648577845316;

    Leb_Grid_XYZW[  441][0] = -0.982798601826395;
    Leb_Grid_XYZW[  441][1] = -0.177177402261533;
    Leb_Grid_XYZW[  441][2] = -0.052106394770113;
    Leb_Grid_XYZW[  441][3] =  0.000648577845316;

    Leb_Grid_XYZW[  442][0] =  0.982798601826395;
    Leb_Grid_XYZW[  442][1] =  0.052106394770113;
    Leb_Grid_XYZW[  442][2] =  0.177177402261533;
    Leb_Grid_XYZW[  442][3] =  0.000648577845316;

    Leb_Grid_XYZW[  443][0] =  0.982798601826395;
    Leb_Grid_XYZW[  443][1] =  0.052106394770113;
    Leb_Grid_XYZW[  443][2] = -0.177177402261532;
    Leb_Grid_XYZW[  443][3] =  0.000648577845316;

    Leb_Grid_XYZW[  444][0] =  0.982798601826395;
    Leb_Grid_XYZW[  444][1] = -0.052106394770113;
    Leb_Grid_XYZW[  444][2] =  0.177177402261533;
    Leb_Grid_XYZW[  444][3] =  0.000648577845316;

    Leb_Grid_XYZW[  445][0] =  0.982798601826395;
    Leb_Grid_XYZW[  445][1] = -0.052106394770113;
    Leb_Grid_XYZW[  445][2] = -0.177177402261532;
    Leb_Grid_XYZW[  445][3] =  0.000648577845316;

    Leb_Grid_XYZW[  446][0] = -0.982798601826395;
    Leb_Grid_XYZW[  446][1] =  0.052106394770113;
    Leb_Grid_XYZW[  446][2] =  0.177177402261533;
    Leb_Grid_XYZW[  446][3] =  0.000648577845316;

    Leb_Grid_XYZW[  447][0] = -0.982798601826395;
    Leb_Grid_XYZW[  447][1] =  0.052106394770113;
    Leb_Grid_XYZW[  447][2] = -0.177177402261532;
    Leb_Grid_XYZW[  447][3] =  0.000648577845316;

    Leb_Grid_XYZW[  448][0] = -0.982798601826395;
    Leb_Grid_XYZW[  448][1] = -0.052106394770113;
    Leb_Grid_XYZW[  448][2] =  0.177177402261533;
    Leb_Grid_XYZW[  448][3] =  0.000648577845316;

    Leb_Grid_XYZW[  449][0] = -0.982798601826395;
    Leb_Grid_XYZW[  449][1] = -0.052106394770113;
    Leb_Grid_XYZW[  449][2] = -0.177177402261532;
    Leb_Grid_XYZW[  449][3] =  0.000648577845316;

    Leb_Grid_XYZW[  450][0] =  0.177177402261532;
    Leb_Grid_XYZW[  450][1] =  0.982798601826395;
    Leb_Grid_XYZW[  450][2] =  0.052106394770113;
    Leb_Grid_XYZW[  450][3] =  0.000648577845316;

    Leb_Grid_XYZW[  451][0] =  0.177177402261532;
    Leb_Grid_XYZW[  451][1] =  0.982798601826395;
    Leb_Grid_XYZW[  451][2] = -0.052106394770113;
    Leb_Grid_XYZW[  451][3] =  0.000648577845316;

    Leb_Grid_XYZW[  452][0] =  0.177177402261532;
    Leb_Grid_XYZW[  452][1] = -0.982798601826395;
    Leb_Grid_XYZW[  452][2] =  0.052106394770113;
    Leb_Grid_XYZW[  452][3] =  0.000648577845316;

    Leb_Grid_XYZW[  453][0] =  0.177177402261532;
    Leb_Grid_XYZW[  453][1] = -0.982798601826395;
    Leb_Grid_XYZW[  453][2] = -0.052106394770113;
    Leb_Grid_XYZW[  453][3] =  0.000648577845316;

    Leb_Grid_XYZW[  454][0] = -0.177177402261533;
    Leb_Grid_XYZW[  454][1] =  0.982798601826395;
    Leb_Grid_XYZW[  454][2] =  0.052106394770113;
    Leb_Grid_XYZW[  454][3] =  0.000648577845316;

    Leb_Grid_XYZW[  455][0] = -0.177177402261533;
    Leb_Grid_XYZW[  455][1] =  0.982798601826395;
    Leb_Grid_XYZW[  455][2] = -0.052106394770113;
    Leb_Grid_XYZW[  455][3] =  0.000648577845316;

    Leb_Grid_XYZW[  456][0] = -0.177177402261533;
    Leb_Grid_XYZW[  456][1] = -0.982798601826395;
    Leb_Grid_XYZW[  456][2] =  0.052106394770113;
    Leb_Grid_XYZW[  456][3] =  0.000648577845316;

    Leb_Grid_XYZW[  457][0] = -0.177177402261533;
    Leb_Grid_XYZW[  457][1] = -0.982798601826395;
    Leb_Grid_XYZW[  457][2] = -0.052106394770113;
    Leb_Grid_XYZW[  457][3] =  0.000648577845316;

    Leb_Grid_XYZW[  458][0] =  0.177177402261533;
    Leb_Grid_XYZW[  458][1] =  0.052106394770113;
    Leb_Grid_XYZW[  458][2] =  0.982798601826395;
    Leb_Grid_XYZW[  458][3] =  0.000648577845316;

    Leb_Grid_XYZW[  459][0] =  0.177177402261532;
    Leb_Grid_XYZW[  459][1] =  0.052106394770113;
    Leb_Grid_XYZW[  459][2] = -0.982798601826395;
    Leb_Grid_XYZW[  459][3] =  0.000648577845316;

    Leb_Grid_XYZW[  460][0] =  0.177177402261533;
    Leb_Grid_XYZW[  460][1] = -0.052106394770113;
    Leb_Grid_XYZW[  460][2] =  0.982798601826395;
    Leb_Grid_XYZW[  460][3] =  0.000648577845316;

    Leb_Grid_XYZW[  461][0] =  0.177177402261532;
    Leb_Grid_XYZW[  461][1] = -0.052106394770113;
    Leb_Grid_XYZW[  461][2] = -0.982798601826395;
    Leb_Grid_XYZW[  461][3] =  0.000648577845316;

    Leb_Grid_XYZW[  462][0] = -0.177177402261533;
    Leb_Grid_XYZW[  462][1] =  0.052106394770113;
    Leb_Grid_XYZW[  462][2] =  0.982798601826395;
    Leb_Grid_XYZW[  462][3] =  0.000648577845316;

    Leb_Grid_XYZW[  463][0] = -0.177177402261532;
    Leb_Grid_XYZW[  463][1] =  0.052106394770113;
    Leb_Grid_XYZW[  463][2] = -0.982798601826395;
    Leb_Grid_XYZW[  463][3] =  0.000648577845316;

    Leb_Grid_XYZW[  464][0] = -0.177177402261533;
    Leb_Grid_XYZW[  464][1] = -0.052106394770113;
    Leb_Grid_XYZW[  464][2] =  0.982798601826395;
    Leb_Grid_XYZW[  464][3] =  0.000648577845316;

    Leb_Grid_XYZW[  465][0] = -0.177177402261532;
    Leb_Grid_XYZW[  465][1] = -0.052106394770113;
    Leb_Grid_XYZW[  465][2] = -0.982798601826395;
    Leb_Grid_XYZW[  465][3] =  0.000648577845316;

    Leb_Grid_XYZW[  466][0] =  0.052106394770113;
    Leb_Grid_XYZW[  466][1] =  0.982798601826395;
    Leb_Grid_XYZW[  466][2] =  0.177177402261533;
    Leb_Grid_XYZW[  466][3] =  0.000648577845316;

    Leb_Grid_XYZW[  467][0] =  0.052106394770113;
    Leb_Grid_XYZW[  467][1] =  0.982798601826395;
    Leb_Grid_XYZW[  467][2] = -0.177177402261532;
    Leb_Grid_XYZW[  467][3] =  0.000648577845316;

    Leb_Grid_XYZW[  468][0] =  0.052106394770113;
    Leb_Grid_XYZW[  468][1] = -0.982798601826395;
    Leb_Grid_XYZW[  468][2] =  0.177177402261533;
    Leb_Grid_XYZW[  468][3] =  0.000648577845316;

    Leb_Grid_XYZW[  469][0] =  0.052106394770113;
    Leb_Grid_XYZW[  469][1] = -0.982798601826395;
    Leb_Grid_XYZW[  469][2] = -0.177177402261532;
    Leb_Grid_XYZW[  469][3] =  0.000648577845316;

    Leb_Grid_XYZW[  470][0] = -0.052106394770113;
    Leb_Grid_XYZW[  470][1] =  0.982798601826395;
    Leb_Grid_XYZW[  470][2] =  0.177177402261533;
    Leb_Grid_XYZW[  470][3] =  0.000648577845316;

    Leb_Grid_XYZW[  471][0] = -0.052106394770113;
    Leb_Grid_XYZW[  471][1] =  0.982798601826395;
    Leb_Grid_XYZW[  471][2] = -0.177177402261532;
    Leb_Grid_XYZW[  471][3] =  0.000648577845316;

    Leb_Grid_XYZW[  472][0] = -0.052106394770113;
    Leb_Grid_XYZW[  472][1] = -0.982798601826395;
    Leb_Grid_XYZW[  472][2] =  0.177177402261533;
    Leb_Grid_XYZW[  472][3] =  0.000648577845316;

    Leb_Grid_XYZW[  473][0] = -0.052106394770113;
    Leb_Grid_XYZW[  473][1] = -0.982798601826395;
    Leb_Grid_XYZW[  473][2] = -0.177177402261532;
    Leb_Grid_XYZW[  473][3] =  0.000648577845316;

    Leb_Grid_XYZW[  474][0] =  0.052106394770113;
    Leb_Grid_XYZW[  474][1] =  0.177177402261533;
    Leb_Grid_XYZW[  474][2] =  0.982798601826395;
    Leb_Grid_XYZW[  474][3] =  0.000648577845316;

    Leb_Grid_XYZW[  475][0] =  0.052106394770113;
    Leb_Grid_XYZW[  475][1] =  0.177177402261532;
    Leb_Grid_XYZW[  475][2] = -0.982798601826395;
    Leb_Grid_XYZW[  475][3] =  0.000648577845316;

    Leb_Grid_XYZW[  476][0] =  0.052106394770113;
    Leb_Grid_XYZW[  476][1] = -0.177177402261533;
    Leb_Grid_XYZW[  476][2] =  0.982798601826395;
    Leb_Grid_XYZW[  476][3] =  0.000648577845316;

    Leb_Grid_XYZW[  477][0] =  0.052106394770113;
    Leb_Grid_XYZW[  477][1] = -0.177177402261532;
    Leb_Grid_XYZW[  477][2] = -0.982798601826395;
    Leb_Grid_XYZW[  477][3] =  0.000648577845316;

    Leb_Grid_XYZW[  478][0] = -0.052106394770113;
    Leb_Grid_XYZW[  478][1] =  0.177177402261532;
    Leb_Grid_XYZW[  478][2] =  0.982798601826395;
    Leb_Grid_XYZW[  478][3] =  0.000648577845316;

    Leb_Grid_XYZW[  479][0] = -0.052106394770113;
    Leb_Grid_XYZW[  479][1] =  0.177177402261532;
    Leb_Grid_XYZW[  479][2] = -0.982798601826395;
    Leb_Grid_XYZW[  479][3] =  0.000648577845316;

    Leb_Grid_XYZW[  480][0] = -0.052106394770113;
    Leb_Grid_XYZW[  480][1] = -0.177177402261532;
    Leb_Grid_XYZW[  480][2] =  0.982798601826395;
    Leb_Grid_XYZW[  480][3] =  0.000648577845316;

    Leb_Grid_XYZW[  481][0] = -0.052106394770113;
    Leb_Grid_XYZW[  481][1] = -0.177177402261532;
    Leb_Grid_XYZW[  481][2] = -0.982798601826395;
    Leb_Grid_XYZW[  481][3] =  0.000648577845316;

    Leb_Grid_XYZW[  482][0] =  0.962424923032623;
    Leb_Grid_XYZW[  482][1] =  0.247571646342629;
    Leb_Grid_XYZW[  482][2] =  0.111564095715648;
    Leb_Grid_XYZW[  482][3] =  0.000743503091098;

    Leb_Grid_XYZW[  483][0] =  0.962424923032623;
    Leb_Grid_XYZW[  483][1] =  0.247571646342629;
    Leb_Grid_XYZW[  483][2] = -0.111564095715648;
    Leb_Grid_XYZW[  483][3] =  0.000743503091098;

    Leb_Grid_XYZW[  484][0] =  0.962424923032623;
    Leb_Grid_XYZW[  484][1] = -0.247571646342629;
    Leb_Grid_XYZW[  484][2] =  0.111564095715648;
    Leb_Grid_XYZW[  484][3] =  0.000743503091098;

    Leb_Grid_XYZW[  485][0] =  0.962424923032623;
    Leb_Grid_XYZW[  485][1] = -0.247571646342629;
    Leb_Grid_XYZW[  485][2] = -0.111564095715648;
    Leb_Grid_XYZW[  485][3] =  0.000743503091098;

    Leb_Grid_XYZW[  486][0] = -0.962424923032623;
    Leb_Grid_XYZW[  486][1] =  0.247571646342629;
    Leb_Grid_XYZW[  486][2] =  0.111564095715648;
    Leb_Grid_XYZW[  486][3] =  0.000743503091098;

    Leb_Grid_XYZW[  487][0] = -0.962424923032623;
    Leb_Grid_XYZW[  487][1] =  0.247571646342629;
    Leb_Grid_XYZW[  487][2] = -0.111564095715648;
    Leb_Grid_XYZW[  487][3] =  0.000743503091098;

    Leb_Grid_XYZW[  488][0] = -0.962424923032623;
    Leb_Grid_XYZW[  488][1] = -0.247571646342629;
    Leb_Grid_XYZW[  488][2] =  0.111564095715648;
    Leb_Grid_XYZW[  488][3] =  0.000743503091098;

    Leb_Grid_XYZW[  489][0] = -0.962424923032623;
    Leb_Grid_XYZW[  489][1] = -0.247571646342629;
    Leb_Grid_XYZW[  489][2] = -0.111564095715648;
    Leb_Grid_XYZW[  489][3] =  0.000743503091098;

    Leb_Grid_XYZW[  490][0] =  0.962424923032623;
    Leb_Grid_XYZW[  490][1] =  0.111564095715647;
    Leb_Grid_XYZW[  490][2] =  0.247571646342629;
    Leb_Grid_XYZW[  490][3] =  0.000743503091098;

    Leb_Grid_XYZW[  491][0] =  0.962424923032623;
    Leb_Grid_XYZW[  491][1] =  0.111564095715647;
    Leb_Grid_XYZW[  491][2] = -0.247571646342629;
    Leb_Grid_XYZW[  491][3] =  0.000743503091098;

    Leb_Grid_XYZW[  492][0] =  0.962424923032623;
    Leb_Grid_XYZW[  492][1] = -0.111564095715647;
    Leb_Grid_XYZW[  492][2] =  0.247571646342629;
    Leb_Grid_XYZW[  492][3] =  0.000743503091098;

    Leb_Grid_XYZW[  493][0] =  0.962424923032623;
    Leb_Grid_XYZW[  493][1] = -0.111564095715647;
    Leb_Grid_XYZW[  493][2] = -0.247571646342629;
    Leb_Grid_XYZW[  493][3] =  0.000743503091098;

    Leb_Grid_XYZW[  494][0] = -0.962424923032623;
    Leb_Grid_XYZW[  494][1] =  0.111564095715647;
    Leb_Grid_XYZW[  494][2] =  0.247571646342629;
    Leb_Grid_XYZW[  494][3] =  0.000743503091098;

    Leb_Grid_XYZW[  495][0] = -0.962424923032623;
    Leb_Grid_XYZW[  495][1] =  0.111564095715647;
    Leb_Grid_XYZW[  495][2] = -0.247571646342629;
    Leb_Grid_XYZW[  495][3] =  0.000743503091098;

    Leb_Grid_XYZW[  496][0] = -0.962424923032623;
    Leb_Grid_XYZW[  496][1] = -0.111564095715647;
    Leb_Grid_XYZW[  496][2] =  0.247571646342629;
    Leb_Grid_XYZW[  496][3] =  0.000743503091098;

    Leb_Grid_XYZW[  497][0] = -0.962424923032623;
    Leb_Grid_XYZW[  497][1] = -0.111564095715647;
    Leb_Grid_XYZW[  497][2] = -0.247571646342629;
    Leb_Grid_XYZW[  497][3] =  0.000743503091098;

    Leb_Grid_XYZW[  498][0] =  0.247571646342629;
    Leb_Grid_XYZW[  498][1] =  0.962424923032623;
    Leb_Grid_XYZW[  498][2] =  0.111564095715648;
    Leb_Grid_XYZW[  498][3] =  0.000743503091098;

    Leb_Grid_XYZW[  499][0] =  0.247571646342629;
    Leb_Grid_XYZW[  499][1] =  0.962424923032623;
    Leb_Grid_XYZW[  499][2] = -0.111564095715648;
    Leb_Grid_XYZW[  499][3] =  0.000743503091098;

    Leb_Grid_XYZW[  500][0] =  0.247571646342629;
    Leb_Grid_XYZW[  500][1] = -0.962424923032623;
    Leb_Grid_XYZW[  500][2] =  0.111564095715648;
    Leb_Grid_XYZW[  500][3] =  0.000743503091098;

    Leb_Grid_XYZW[  501][0] =  0.247571646342629;
    Leb_Grid_XYZW[  501][1] = -0.962424923032623;
    Leb_Grid_XYZW[  501][2] = -0.111564095715648;
    Leb_Grid_XYZW[  501][3] =  0.000743503091098;

    Leb_Grid_XYZW[  502][0] = -0.247571646342629;
    Leb_Grid_XYZW[  502][1] =  0.962424923032623;
    Leb_Grid_XYZW[  502][2] =  0.111564095715648;
    Leb_Grid_XYZW[  502][3] =  0.000743503091098;

    Leb_Grid_XYZW[  503][0] = -0.247571646342629;
    Leb_Grid_XYZW[  503][1] =  0.962424923032623;
    Leb_Grid_XYZW[  503][2] = -0.111564095715648;
    Leb_Grid_XYZW[  503][3] =  0.000743503091098;

    Leb_Grid_XYZW[  504][0] = -0.247571646342629;
    Leb_Grid_XYZW[  504][1] = -0.962424923032623;
    Leb_Grid_XYZW[  504][2] =  0.111564095715648;
    Leb_Grid_XYZW[  504][3] =  0.000743503091098;

    Leb_Grid_XYZW[  505][0] = -0.247571646342629;
    Leb_Grid_XYZW[  505][1] = -0.962424923032623;
    Leb_Grid_XYZW[  505][2] = -0.111564095715648;
    Leb_Grid_XYZW[  505][3] =  0.000743503091098;

    Leb_Grid_XYZW[  506][0] =  0.247571646342629;
    Leb_Grid_XYZW[  506][1] =  0.111564095715648;
    Leb_Grid_XYZW[  506][2] =  0.962424923032623;
    Leb_Grid_XYZW[  506][3] =  0.000743503091098;

    Leb_Grid_XYZW[  507][0] =  0.247571646342629;
    Leb_Grid_XYZW[  507][1] =  0.111564095715648;
    Leb_Grid_XYZW[  507][2] = -0.962424923032623;
    Leb_Grid_XYZW[  507][3] =  0.000743503091098;

    Leb_Grid_XYZW[  508][0] =  0.247571646342629;
    Leb_Grid_XYZW[  508][1] = -0.111564095715648;
    Leb_Grid_XYZW[  508][2] =  0.962424923032623;
    Leb_Grid_XYZW[  508][3] =  0.000743503091098;

    Leb_Grid_XYZW[  509][0] =  0.247571646342629;
    Leb_Grid_XYZW[  509][1] = -0.111564095715648;
    Leb_Grid_XYZW[  509][2] = -0.962424923032623;
    Leb_Grid_XYZW[  509][3] =  0.000743503091098;

    Leb_Grid_XYZW[  510][0] = -0.247571646342629;
    Leb_Grid_XYZW[  510][1] =  0.111564095715648;
    Leb_Grid_XYZW[  510][2] =  0.962424923032623;
    Leb_Grid_XYZW[  510][3] =  0.000743503091098;

    Leb_Grid_XYZW[  511][0] = -0.247571646342629;
    Leb_Grid_XYZW[  511][1] =  0.111564095715648;
    Leb_Grid_XYZW[  511][2] = -0.962424923032623;
    Leb_Grid_XYZW[  511][3] =  0.000743503091098;

    Leb_Grid_XYZW[  512][0] = -0.247571646342629;
    Leb_Grid_XYZW[  512][1] = -0.111564095715648;
    Leb_Grid_XYZW[  512][2] =  0.962424923032623;
    Leb_Grid_XYZW[  512][3] =  0.000743503091098;

    Leb_Grid_XYZW[  513][0] = -0.247571646342629;
    Leb_Grid_XYZW[  513][1] = -0.111564095715648;
    Leb_Grid_XYZW[  513][2] = -0.962424923032623;
    Leb_Grid_XYZW[  513][3] =  0.000743503091098;

    Leb_Grid_XYZW[  514][0] =  0.111564095715648;
    Leb_Grid_XYZW[  514][1] =  0.962424923032623;
    Leb_Grid_XYZW[  514][2] =  0.247571646342629;
    Leb_Grid_XYZW[  514][3] =  0.000743503091098;

    Leb_Grid_XYZW[  515][0] =  0.111564095715648;
    Leb_Grid_XYZW[  515][1] =  0.962424923032623;
    Leb_Grid_XYZW[  515][2] = -0.247571646342629;
    Leb_Grid_XYZW[  515][3] =  0.000743503091098;

    Leb_Grid_XYZW[  516][0] =  0.111564095715648;
    Leb_Grid_XYZW[  516][1] = -0.962424923032623;
    Leb_Grid_XYZW[  516][2] =  0.247571646342629;
    Leb_Grid_XYZW[  516][3] =  0.000743503091098;

    Leb_Grid_XYZW[  517][0] =  0.111564095715648;
    Leb_Grid_XYZW[  517][1] = -0.962424923032623;
    Leb_Grid_XYZW[  517][2] = -0.247571646342629;
    Leb_Grid_XYZW[  517][3] =  0.000743503091098;

    Leb_Grid_XYZW[  518][0] = -0.111564095715648;
    Leb_Grid_XYZW[  518][1] =  0.962424923032623;
    Leb_Grid_XYZW[  518][2] =  0.247571646342629;
    Leb_Grid_XYZW[  518][3] =  0.000743503091098;

    Leb_Grid_XYZW[  519][0] = -0.111564095715648;
    Leb_Grid_XYZW[  519][1] =  0.962424923032623;
    Leb_Grid_XYZW[  519][2] = -0.247571646342629;
    Leb_Grid_XYZW[  519][3] =  0.000743503091098;

    Leb_Grid_XYZW[  520][0] = -0.111564095715648;
    Leb_Grid_XYZW[  520][1] = -0.962424923032623;
    Leb_Grid_XYZW[  520][2] =  0.247571646342629;
    Leb_Grid_XYZW[  520][3] =  0.000743503091098;

    Leb_Grid_XYZW[  521][0] = -0.111564095715648;
    Leb_Grid_XYZW[  521][1] = -0.962424923032623;
    Leb_Grid_XYZW[  521][2] = -0.247571646342629;
    Leb_Grid_XYZW[  521][3] =  0.000743503091098;

    Leb_Grid_XYZW[  522][0] =  0.111564095715648;
    Leb_Grid_XYZW[  522][1] =  0.247571646342629;
    Leb_Grid_XYZW[  522][2] =  0.962424923032623;
    Leb_Grid_XYZW[  522][3] =  0.000743503091098;

    Leb_Grid_XYZW[  523][0] =  0.111564095715648;
    Leb_Grid_XYZW[  523][1] =  0.247571646342629;
    Leb_Grid_XYZW[  523][2] = -0.962424923032623;
    Leb_Grid_XYZW[  523][3] =  0.000743503091098;

    Leb_Grid_XYZW[  524][0] =  0.111564095715648;
    Leb_Grid_XYZW[  524][1] = -0.247571646342629;
    Leb_Grid_XYZW[  524][2] =  0.962424923032623;
    Leb_Grid_XYZW[  524][3] =  0.000743503091098;

    Leb_Grid_XYZW[  525][0] =  0.111564095715648;
    Leb_Grid_XYZW[  525][1] = -0.247571646342629;
    Leb_Grid_XYZW[  525][2] = -0.962424923032623;
    Leb_Grid_XYZW[  525][3] =  0.000743503091098;

    Leb_Grid_XYZW[  526][0] = -0.111564095715648;
    Leb_Grid_XYZW[  526][1] =  0.247571646342629;
    Leb_Grid_XYZW[  526][2] =  0.962424923032623;
    Leb_Grid_XYZW[  526][3] =  0.000743503091098;

    Leb_Grid_XYZW[  527][0] = -0.111564095715649;
    Leb_Grid_XYZW[  527][1] =  0.247571646342629;
    Leb_Grid_XYZW[  527][2] = -0.962424923032623;
    Leb_Grid_XYZW[  527][3] =  0.000743503091098;

    Leb_Grid_XYZW[  528][0] = -0.111564095715648;
    Leb_Grid_XYZW[  528][1] = -0.247571646342629;
    Leb_Grid_XYZW[  528][2] =  0.962424923032623;
    Leb_Grid_XYZW[  528][3] =  0.000743503091098;

    Leb_Grid_XYZW[  529][0] = -0.111564095715649;
    Leb_Grid_XYZW[  529][1] = -0.247571646342629;
    Leb_Grid_XYZW[  529][2] = -0.962424923032623;
    Leb_Grid_XYZW[  529][3] =  0.000743503091098;

    Leb_Grid_XYZW[  530][0] =  0.940200799412881;
    Leb_Grid_XYZW[  530][1] =  0.335461628906649;
    Leb_Grid_XYZW[  530][2] =  0.059058888532354;
    Leb_Grid_XYZW[  530][3] =  0.000799852789184;

    Leb_Grid_XYZW[  531][0] =  0.940200799412881;
    Leb_Grid_XYZW[  531][1] =  0.335461628906649;
    Leb_Grid_XYZW[  531][2] = -0.059058888532354;
    Leb_Grid_XYZW[  531][3] =  0.000799852789184;

    Leb_Grid_XYZW[  532][0] =  0.940200799412881;
    Leb_Grid_XYZW[  532][1] = -0.335461628906649;
    Leb_Grid_XYZW[  532][2] =  0.059058888532354;
    Leb_Grid_XYZW[  532][3] =  0.000799852789184;

    Leb_Grid_XYZW[  533][0] =  0.940200799412881;
    Leb_Grid_XYZW[  533][1] = -0.335461628906649;
    Leb_Grid_XYZW[  533][2] = -0.059058888532354;
    Leb_Grid_XYZW[  533][3] =  0.000799852789184;

    Leb_Grid_XYZW[  534][0] = -0.940200799412881;
    Leb_Grid_XYZW[  534][1] =  0.335461628906649;
    Leb_Grid_XYZW[  534][2] =  0.059058888532354;
    Leb_Grid_XYZW[  534][3] =  0.000799852789184;

    Leb_Grid_XYZW[  535][0] = -0.940200799412881;
    Leb_Grid_XYZW[  535][1] =  0.335461628906649;
    Leb_Grid_XYZW[  535][2] = -0.059058888532354;
    Leb_Grid_XYZW[  535][3] =  0.000799852789184;

    Leb_Grid_XYZW[  536][0] = -0.940200799412881;
    Leb_Grid_XYZW[  536][1] = -0.335461628906649;
    Leb_Grid_XYZW[  536][2] =  0.059058888532354;
    Leb_Grid_XYZW[  536][3] =  0.000799852789184;

    Leb_Grid_XYZW[  537][0] = -0.940200799412881;
    Leb_Grid_XYZW[  537][1] = -0.335461628906649;
    Leb_Grid_XYZW[  537][2] = -0.059058888532354;
    Leb_Grid_XYZW[  537][3] =  0.000799852789184;

    Leb_Grid_XYZW[  538][0] =  0.940200799412881;
    Leb_Grid_XYZW[  538][1] =  0.059058888532354;
    Leb_Grid_XYZW[  538][2] =  0.335461628906649;
    Leb_Grid_XYZW[  538][3] =  0.000799852789184;

    Leb_Grid_XYZW[  539][0] =  0.940200799412881;
    Leb_Grid_XYZW[  539][1] =  0.059058888532354;
    Leb_Grid_XYZW[  539][2] = -0.335461628906649;
    Leb_Grid_XYZW[  539][3] =  0.000799852789184;

    Leb_Grid_XYZW[  540][0] =  0.940200799412881;
    Leb_Grid_XYZW[  540][1] = -0.059058888532354;
    Leb_Grid_XYZW[  540][2] =  0.335461628906649;
    Leb_Grid_XYZW[  540][3] =  0.000799852789184;

    Leb_Grid_XYZW[  541][0] =  0.940200799412881;
    Leb_Grid_XYZW[  541][1] = -0.059058888532354;
    Leb_Grid_XYZW[  541][2] = -0.335461628906649;
    Leb_Grid_XYZW[  541][3] =  0.000799852789184;

    Leb_Grid_XYZW[  542][0] = -0.940200799412881;
    Leb_Grid_XYZW[  542][1] =  0.059058888532354;
    Leb_Grid_XYZW[  542][2] =  0.335461628906649;
    Leb_Grid_XYZW[  542][3] =  0.000799852789184;

    Leb_Grid_XYZW[  543][0] = -0.940200799412881;
    Leb_Grid_XYZW[  543][1] =  0.059058888532354;
    Leb_Grid_XYZW[  543][2] = -0.335461628906649;
    Leb_Grid_XYZW[  543][3] =  0.000799852789184;

    Leb_Grid_XYZW[  544][0] = -0.940200799412881;
    Leb_Grid_XYZW[  544][1] = -0.059058888532354;
    Leb_Grid_XYZW[  544][2] =  0.335461628906649;
    Leb_Grid_XYZW[  544][3] =  0.000799852789184;

    Leb_Grid_XYZW[  545][0] = -0.940200799412881;
    Leb_Grid_XYZW[  545][1] = -0.059058888532354;
    Leb_Grid_XYZW[  545][2] = -0.335461628906649;
    Leb_Grid_XYZW[  545][3] =  0.000799852789184;

    Leb_Grid_XYZW[  546][0] =  0.335461628906649;
    Leb_Grid_XYZW[  546][1] =  0.940200799412881;
    Leb_Grid_XYZW[  546][2] =  0.059058888532354;
    Leb_Grid_XYZW[  546][3] =  0.000799852789184;

    Leb_Grid_XYZW[  547][0] =  0.335461628906649;
    Leb_Grid_XYZW[  547][1] =  0.940200799412881;
    Leb_Grid_XYZW[  547][2] = -0.059058888532354;
    Leb_Grid_XYZW[  547][3] =  0.000799852789184;

    Leb_Grid_XYZW[  548][0] =  0.335461628906649;
    Leb_Grid_XYZW[  548][1] = -0.940200799412881;
    Leb_Grid_XYZW[  548][2] =  0.059058888532354;
    Leb_Grid_XYZW[  548][3] =  0.000799852789184;

    Leb_Grid_XYZW[  549][0] =  0.335461628906649;
    Leb_Grid_XYZW[  549][1] = -0.940200799412881;
    Leb_Grid_XYZW[  549][2] = -0.059058888532354;
    Leb_Grid_XYZW[  549][3] =  0.000799852789184;

    Leb_Grid_XYZW[  550][0] = -0.335461628906649;
    Leb_Grid_XYZW[  550][1] =  0.940200799412881;
    Leb_Grid_XYZW[  550][2] =  0.059058888532354;
    Leb_Grid_XYZW[  550][3] =  0.000799852789184;

    Leb_Grid_XYZW[  551][0] = -0.335461628906649;
    Leb_Grid_XYZW[  551][1] =  0.940200799412881;
    Leb_Grid_XYZW[  551][2] = -0.059058888532354;
    Leb_Grid_XYZW[  551][3] =  0.000799852789184;

    Leb_Grid_XYZW[  552][0] = -0.335461628906649;
    Leb_Grid_XYZW[  552][1] = -0.940200799412881;
    Leb_Grid_XYZW[  552][2] =  0.059058888532354;
    Leb_Grid_XYZW[  552][3] =  0.000799852789184;

    Leb_Grid_XYZW[  553][0] = -0.335461628906649;
    Leb_Grid_XYZW[  553][1] = -0.940200799412881;
    Leb_Grid_XYZW[  553][2] = -0.059058888532354;
    Leb_Grid_XYZW[  553][3] =  0.000799852789184;

    Leb_Grid_XYZW[  554][0] =  0.335461628906649;
    Leb_Grid_XYZW[  554][1] =  0.059058888532354;
    Leb_Grid_XYZW[  554][2] =  0.940200799412881;
    Leb_Grid_XYZW[  554][3] =  0.000799852789184;

    Leb_Grid_XYZW[  555][0] =  0.335461628906649;
    Leb_Grid_XYZW[  555][1] =  0.059058888532354;
    Leb_Grid_XYZW[  555][2] = -0.940200799412881;
    Leb_Grid_XYZW[  555][3] =  0.000799852789184;

    Leb_Grid_XYZW[  556][0] =  0.335461628906649;
    Leb_Grid_XYZW[  556][1] = -0.059058888532354;
    Leb_Grid_XYZW[  556][2] =  0.940200799412881;
    Leb_Grid_XYZW[  556][3] =  0.000799852789184;

    Leb_Grid_XYZW[  557][0] =  0.335461628906649;
    Leb_Grid_XYZW[  557][1] = -0.059058888532354;
    Leb_Grid_XYZW[  557][2] = -0.940200799412881;
    Leb_Grid_XYZW[  557][3] =  0.000799852789184;

    Leb_Grid_XYZW[  558][0] = -0.335461628906649;
    Leb_Grid_XYZW[  558][1] =  0.059058888532354;
    Leb_Grid_XYZW[  558][2] =  0.940200799412881;
    Leb_Grid_XYZW[  558][3] =  0.000799852789184;

    Leb_Grid_XYZW[  559][0] = -0.335461628906649;
    Leb_Grid_XYZW[  559][1] =  0.059058888532354;
    Leb_Grid_XYZW[  559][2] = -0.940200799412881;
    Leb_Grid_XYZW[  559][3] =  0.000799852789184;

    Leb_Grid_XYZW[  560][0] = -0.335461628906649;
    Leb_Grid_XYZW[  560][1] = -0.059058888532354;
    Leb_Grid_XYZW[  560][2] =  0.940200799412881;
    Leb_Grid_XYZW[  560][3] =  0.000799852789184;

    Leb_Grid_XYZW[  561][0] = -0.335461628906649;
    Leb_Grid_XYZW[  561][1] = -0.059058888532354;
    Leb_Grid_XYZW[  561][2] = -0.940200799412881;
    Leb_Grid_XYZW[  561][3] =  0.000799852789184;

    Leb_Grid_XYZW[  562][0] =  0.059058888532354;
    Leb_Grid_XYZW[  562][1] =  0.940200799412881;
    Leb_Grid_XYZW[  562][2] =  0.335461628906649;
    Leb_Grid_XYZW[  562][3] =  0.000799852789184;

    Leb_Grid_XYZW[  563][0] =  0.059058888532354;
    Leb_Grid_XYZW[  563][1] =  0.940200799412881;
    Leb_Grid_XYZW[  563][2] = -0.335461628906649;
    Leb_Grid_XYZW[  563][3] =  0.000799852789184;

    Leb_Grid_XYZW[  564][0] =  0.059058888532354;
    Leb_Grid_XYZW[  564][1] = -0.940200799412881;
    Leb_Grid_XYZW[  564][2] =  0.335461628906649;
    Leb_Grid_XYZW[  564][3] =  0.000799852789184;

    Leb_Grid_XYZW[  565][0] =  0.059058888532354;
    Leb_Grid_XYZW[  565][1] = -0.940200799412881;
    Leb_Grid_XYZW[  565][2] = -0.335461628906649;
    Leb_Grid_XYZW[  565][3] =  0.000799852789184;

    Leb_Grid_XYZW[  566][0] = -0.059058888532354;
    Leb_Grid_XYZW[  566][1] =  0.940200799412881;
    Leb_Grid_XYZW[  566][2] =  0.335461628906649;
    Leb_Grid_XYZW[  566][3] =  0.000799852789184;

    Leb_Grid_XYZW[  567][0] = -0.059058888532354;
    Leb_Grid_XYZW[  567][1] =  0.940200799412881;
    Leb_Grid_XYZW[  567][2] = -0.335461628906649;
    Leb_Grid_XYZW[  567][3] =  0.000799852789184;

    Leb_Grid_XYZW[  568][0] = -0.059058888532354;
    Leb_Grid_XYZW[  568][1] = -0.940200799412881;
    Leb_Grid_XYZW[  568][2] =  0.335461628906649;
    Leb_Grid_XYZW[  568][3] =  0.000799852789184;

    Leb_Grid_XYZW[  569][0] = -0.059058888532354;
    Leb_Grid_XYZW[  569][1] = -0.940200799412881;
    Leb_Grid_XYZW[  569][2] = -0.335461628906649;
    Leb_Grid_XYZW[  569][3] =  0.000799852789184;

    Leb_Grid_XYZW[  570][0] =  0.059058888532354;
    Leb_Grid_XYZW[  570][1] =  0.335461628906649;
    Leb_Grid_XYZW[  570][2] =  0.940200799412881;
    Leb_Grid_XYZW[  570][3] =  0.000799852789184;

    Leb_Grid_XYZW[  571][0] =  0.059058888532354;
    Leb_Grid_XYZW[  571][1] =  0.335461628906649;
    Leb_Grid_XYZW[  571][2] = -0.940200799412881;
    Leb_Grid_XYZW[  571][3] =  0.000799852789184;

    Leb_Grid_XYZW[  572][0] =  0.059058888532354;
    Leb_Grid_XYZW[  572][1] = -0.335461628906649;
    Leb_Grid_XYZW[  572][2] =  0.940200799412881;
    Leb_Grid_XYZW[  572][3] =  0.000799852789184;

    Leb_Grid_XYZW[  573][0] =  0.059058888532354;
    Leb_Grid_XYZW[  573][1] = -0.335461628906649;
    Leb_Grid_XYZW[  573][2] = -0.940200799412881;
    Leb_Grid_XYZW[  573][3] =  0.000799852789184;

    Leb_Grid_XYZW[  574][0] = -0.059058888532354;
    Leb_Grid_XYZW[  574][1] =  0.335461628906649;
    Leb_Grid_XYZW[  574][2] =  0.940200799412881;
    Leb_Grid_XYZW[  574][3] =  0.000799852789184;

    Leb_Grid_XYZW[  575][0] = -0.059058888532354;
    Leb_Grid_XYZW[  575][1] =  0.335461628906649;
    Leb_Grid_XYZW[  575][2] = -0.940200799412881;
    Leb_Grid_XYZW[  575][3] =  0.000799852789184;

    Leb_Grid_XYZW[  576][0] = -0.059058888532354;
    Leb_Grid_XYZW[  576][1] = -0.335461628906649;
    Leb_Grid_XYZW[  576][2] =  0.940200799412881;
    Leb_Grid_XYZW[  576][3] =  0.000799852789184;

    Leb_Grid_XYZW[  577][0] = -0.059058888532354;
    Leb_Grid_XYZW[  577][1] = -0.335461628906649;
    Leb_Grid_XYZW[  577][2] = -0.940200799412881;
    Leb_Grid_XYZW[  577][3] =  0.000799852789184;

    Leb_Grid_XYZW[  578][0] =  0.932082204014320;
    Leb_Grid_XYZW[  578][1] =  0.317361524661198;
    Leb_Grid_XYZW[  578][2] =  0.174655167757863;
    Leb_Grid_XYZW[  578][3] =  0.000810173149747;

    Leb_Grid_XYZW[  579][0] =  0.932082204014320;
    Leb_Grid_XYZW[  579][1] =  0.317361524661198;
    Leb_Grid_XYZW[  579][2] = -0.174655167757863;
    Leb_Grid_XYZW[  579][3] =  0.000810173149747;

    Leb_Grid_XYZW[  580][0] =  0.932082204014320;
    Leb_Grid_XYZW[  580][1] = -0.317361524661198;
    Leb_Grid_XYZW[  580][2] =  0.174655167757863;
    Leb_Grid_XYZW[  580][3] =  0.000810173149747;

    Leb_Grid_XYZW[  581][0] =  0.932082204014320;
    Leb_Grid_XYZW[  581][1] = -0.317361524661198;
    Leb_Grid_XYZW[  581][2] = -0.174655167757863;
    Leb_Grid_XYZW[  581][3] =  0.000810173149747;

    Leb_Grid_XYZW[  582][0] = -0.932082204014320;
    Leb_Grid_XYZW[  582][1] =  0.317361524661197;
    Leb_Grid_XYZW[  582][2] =  0.174655167757863;
    Leb_Grid_XYZW[  582][3] =  0.000810173149747;

    Leb_Grid_XYZW[  583][0] = -0.932082204014320;
    Leb_Grid_XYZW[  583][1] =  0.317361524661197;
    Leb_Grid_XYZW[  583][2] = -0.174655167757863;
    Leb_Grid_XYZW[  583][3] =  0.000810173149747;

    Leb_Grid_XYZW[  584][0] = -0.932082204014320;
    Leb_Grid_XYZW[  584][1] = -0.317361524661197;
    Leb_Grid_XYZW[  584][2] =  0.174655167757863;
    Leb_Grid_XYZW[  584][3] =  0.000810173149747;

    Leb_Grid_XYZW[  585][0] = -0.932082204014320;
    Leb_Grid_XYZW[  585][1] = -0.317361524661197;
    Leb_Grid_XYZW[  585][2] = -0.174655167757863;
    Leb_Grid_XYZW[  585][3] =  0.000810173149747;

    Leb_Grid_XYZW[  586][0] =  0.932082204014320;
    Leb_Grid_XYZW[  586][1] =  0.174655167757862;
    Leb_Grid_XYZW[  586][2] =  0.317361524661198;
    Leb_Grid_XYZW[  586][3] =  0.000810173149747;

    Leb_Grid_XYZW[  587][0] =  0.932082204014320;
    Leb_Grid_XYZW[  587][1] =  0.174655167757862;
    Leb_Grid_XYZW[  587][2] = -0.317361524661198;
    Leb_Grid_XYZW[  587][3] =  0.000810173149747;

    Leb_Grid_XYZW[  588][0] =  0.932082204014320;
    Leb_Grid_XYZW[  588][1] = -0.174655167757862;
    Leb_Grid_XYZW[  588][2] =  0.317361524661198;
    Leb_Grid_XYZW[  588][3] =  0.000810173149747;

    Leb_Grid_XYZW[  589][0] =  0.932082204014320;
    Leb_Grid_XYZW[  589][1] = -0.174655167757862;
    Leb_Grid_XYZW[  589][2] = -0.317361524661198;
    Leb_Grid_XYZW[  589][3] =  0.000810173149747;

    Leb_Grid_XYZW[  590][0] = -0.932082204014320;
    Leb_Grid_XYZW[  590][1] =  0.174655167757863;
    Leb_Grid_XYZW[  590][2] =  0.317361524661198;
    Leb_Grid_XYZW[  590][3] =  0.000810173149747;

    Leb_Grid_XYZW[  591][0] = -0.932082204014320;
    Leb_Grid_XYZW[  591][1] =  0.174655167757863;
    Leb_Grid_XYZW[  591][2] = -0.317361524661198;
    Leb_Grid_XYZW[  591][3] =  0.000810173149747;

    Leb_Grid_XYZW[  592][0] = -0.932082204014320;
    Leb_Grid_XYZW[  592][1] = -0.174655167757863;
    Leb_Grid_XYZW[  592][2] =  0.317361524661198;
    Leb_Grid_XYZW[  592][3] =  0.000810173149747;

    Leb_Grid_XYZW[  593][0] = -0.932082204014320;
    Leb_Grid_XYZW[  593][1] = -0.174655167757863;
    Leb_Grid_XYZW[  593][2] = -0.317361524661198;
    Leb_Grid_XYZW[  593][3] =  0.000810173149747;

    Leb_Grid_XYZW[  594][0] =  0.317361524661198;
    Leb_Grid_XYZW[  594][1] =  0.932082204014320;
    Leb_Grid_XYZW[  594][2] =  0.174655167757863;
    Leb_Grid_XYZW[  594][3] =  0.000810173149747;

    Leb_Grid_XYZW[  595][0] =  0.317361524661198;
    Leb_Grid_XYZW[  595][1] =  0.932082204014320;
    Leb_Grid_XYZW[  595][2] = -0.174655167757863;
    Leb_Grid_XYZW[  595][3] =  0.000810173149747;

    Leb_Grid_XYZW[  596][0] =  0.317361524661198;
    Leb_Grid_XYZW[  596][1] = -0.932082204014320;
    Leb_Grid_XYZW[  596][2] =  0.174655167757863;
    Leb_Grid_XYZW[  596][3] =  0.000810173149747;

    Leb_Grid_XYZW[  597][0] =  0.317361524661198;
    Leb_Grid_XYZW[  597][1] = -0.932082204014320;
    Leb_Grid_XYZW[  597][2] = -0.174655167757863;
    Leb_Grid_XYZW[  597][3] =  0.000810173149747;

    Leb_Grid_XYZW[  598][0] = -0.317361524661198;
    Leb_Grid_XYZW[  598][1] =  0.932082204014320;
    Leb_Grid_XYZW[  598][2] =  0.174655167757863;
    Leb_Grid_XYZW[  598][3] =  0.000810173149747;

    Leb_Grid_XYZW[  599][0] = -0.317361524661198;
    Leb_Grid_XYZW[  599][1] =  0.932082204014320;
    Leb_Grid_XYZW[  599][2] = -0.174655167757863;
    Leb_Grid_XYZW[  599][3] =  0.000810173149747;

    Leb_Grid_XYZW[  600][0] = -0.317361524661198;
    Leb_Grid_XYZW[  600][1] = -0.932082204014320;
    Leb_Grid_XYZW[  600][2] =  0.174655167757863;
    Leb_Grid_XYZW[  600][3] =  0.000810173149747;

    Leb_Grid_XYZW[  601][0] = -0.317361524661198;
    Leb_Grid_XYZW[  601][1] = -0.932082204014320;
    Leb_Grid_XYZW[  601][2] = -0.174655167757863;
    Leb_Grid_XYZW[  601][3] =  0.000810173149747;

    Leb_Grid_XYZW[  602][0] =  0.317361524661198;
    Leb_Grid_XYZW[  602][1] =  0.174655167757863;
    Leb_Grid_XYZW[  602][2] =  0.932082204014320;
    Leb_Grid_XYZW[  602][3] =  0.000810173149747;

    Leb_Grid_XYZW[  603][0] =  0.317361524661198;
    Leb_Grid_XYZW[  603][1] =  0.174655167757863;
    Leb_Grid_XYZW[  603][2] = -0.932082204014320;
    Leb_Grid_XYZW[  603][3] =  0.000810173149747;

    Leb_Grid_XYZW[  604][0] =  0.317361524661198;
    Leb_Grid_XYZW[  604][1] = -0.174655167757863;
    Leb_Grid_XYZW[  604][2] =  0.932082204014320;
    Leb_Grid_XYZW[  604][3] =  0.000810173149747;

    Leb_Grid_XYZW[  605][0] =  0.317361524661198;
    Leb_Grid_XYZW[  605][1] = -0.174655167757863;
    Leb_Grid_XYZW[  605][2] = -0.932082204014320;
    Leb_Grid_XYZW[  605][3] =  0.000810173149747;

    Leb_Grid_XYZW[  606][0] = -0.317361524661198;
    Leb_Grid_XYZW[  606][1] =  0.174655167757863;
    Leb_Grid_XYZW[  606][2] =  0.932082204014320;
    Leb_Grid_XYZW[  606][3] =  0.000810173149747;

    Leb_Grid_XYZW[  607][0] = -0.317361524661198;
    Leb_Grid_XYZW[  607][1] =  0.174655167757863;
    Leb_Grid_XYZW[  607][2] = -0.932082204014320;
    Leb_Grid_XYZW[  607][3] =  0.000810173149747;

    Leb_Grid_XYZW[  608][0] = -0.317361524661198;
    Leb_Grid_XYZW[  608][1] = -0.174655167757863;
    Leb_Grid_XYZW[  608][2] =  0.932082204014320;
    Leb_Grid_XYZW[  608][3] =  0.000810173149747;

    Leb_Grid_XYZW[  609][0] = -0.317361524661198;
    Leb_Grid_XYZW[  609][1] = -0.174655167757863;
    Leb_Grid_XYZW[  609][2] = -0.932082204014320;
    Leb_Grid_XYZW[  609][3] =  0.000810173149747;

    Leb_Grid_XYZW[  610][0] =  0.174655167757863;
    Leb_Grid_XYZW[  610][1] =  0.932082204014320;
    Leb_Grid_XYZW[  610][2] =  0.317361524661198;
    Leb_Grid_XYZW[  610][3] =  0.000810173149747;

    Leb_Grid_XYZW[  611][0] =  0.174655167757863;
    Leb_Grid_XYZW[  611][1] =  0.932082204014320;
    Leb_Grid_XYZW[  611][2] = -0.317361524661198;
    Leb_Grid_XYZW[  611][3] =  0.000810173149747;

    Leb_Grid_XYZW[  612][0] =  0.174655167757863;
    Leb_Grid_XYZW[  612][1] = -0.932082204014320;
    Leb_Grid_XYZW[  612][2] =  0.317361524661198;
    Leb_Grid_XYZW[  612][3] =  0.000810173149747;

    Leb_Grid_XYZW[  613][0] =  0.174655167757863;
    Leb_Grid_XYZW[  613][1] = -0.932082204014320;
    Leb_Grid_XYZW[  613][2] = -0.317361524661198;
    Leb_Grid_XYZW[  613][3] =  0.000810173149747;

    Leb_Grid_XYZW[  614][0] = -0.174655167757863;
    Leb_Grid_XYZW[  614][1] =  0.932082204014320;
    Leb_Grid_XYZW[  614][2] =  0.317361524661198;
    Leb_Grid_XYZW[  614][3] =  0.000810173149747;

    Leb_Grid_XYZW[  615][0] = -0.174655167757863;
    Leb_Grid_XYZW[  615][1] =  0.932082204014320;
    Leb_Grid_XYZW[  615][2] = -0.317361524661198;
    Leb_Grid_XYZW[  615][3] =  0.000810173149747;

    Leb_Grid_XYZW[  616][0] = -0.174655167757863;
    Leb_Grid_XYZW[  616][1] = -0.932082204014320;
    Leb_Grid_XYZW[  616][2] =  0.317361524661198;
    Leb_Grid_XYZW[  616][3] =  0.000810173149747;

    Leb_Grid_XYZW[  617][0] = -0.174655167757863;
    Leb_Grid_XYZW[  617][1] = -0.932082204014320;
    Leb_Grid_XYZW[  617][2] = -0.317361524661198;
    Leb_Grid_XYZW[  617][3] =  0.000810173149747;

    Leb_Grid_XYZW[  618][0] =  0.174655167757863;
    Leb_Grid_XYZW[  618][1] =  0.317361524661198;
    Leb_Grid_XYZW[  618][2] =  0.932082204014320;
    Leb_Grid_XYZW[  618][3] =  0.000810173149747;

    Leb_Grid_XYZW[  619][0] =  0.174655167757863;
    Leb_Grid_XYZW[  619][1] =  0.317361524661198;
    Leb_Grid_XYZW[  619][2] = -0.932082204014320;
    Leb_Grid_XYZW[  619][3] =  0.000810173149747;

    Leb_Grid_XYZW[  620][0] =  0.174655167757863;
    Leb_Grid_XYZW[  620][1] = -0.317361524661198;
    Leb_Grid_XYZW[  620][2] =  0.932082204014320;
    Leb_Grid_XYZW[  620][3] =  0.000810173149747;

    Leb_Grid_XYZW[  621][0] =  0.174655167757863;
    Leb_Grid_XYZW[  621][1] = -0.317361524661198;
    Leb_Grid_XYZW[  621][2] = -0.932082204014320;
    Leb_Grid_XYZW[  621][3] =  0.000810173149747;

    Leb_Grid_XYZW[  622][0] = -0.174655167757863;
    Leb_Grid_XYZW[  622][1] =  0.317361524661198;
    Leb_Grid_XYZW[  622][2] =  0.932082204014320;
    Leb_Grid_XYZW[  622][3] =  0.000810173149747;

    Leb_Grid_XYZW[  623][0] = -0.174655167757863;
    Leb_Grid_XYZW[  623][1] =  0.317361524661198;
    Leb_Grid_XYZW[  623][2] = -0.932082204014320;
    Leb_Grid_XYZW[  623][3] =  0.000810173149747;

    Leb_Grid_XYZW[  624][0] = -0.174655167757863;
    Leb_Grid_XYZW[  624][1] = -0.317361524661198;
    Leb_Grid_XYZW[  624][2] =  0.932082204014320;
    Leb_Grid_XYZW[  624][3] =  0.000810173149747;

    Leb_Grid_XYZW[  625][0] = -0.174655167757863;
    Leb_Grid_XYZW[  625][1] = -0.317361524661198;
    Leb_Grid_XYZW[  625][2] = -0.932082204014320;
    Leb_Grid_XYZW[  625][3] =  0.000810173149747;

    Leb_Grid_XYZW[  626][0] =  0.904367419939330;
    Leb_Grid_XYZW[  626][1] =  0.409026842708536;
    Leb_Grid_XYZW[  626][2] =  0.121723505109599;
    Leb_Grid_XYZW[  626][3] =  0.000848338957459;

    Leb_Grid_XYZW[  627][0] =  0.904367419939330;
    Leb_Grid_XYZW[  627][1] =  0.409026842708536;
    Leb_Grid_XYZW[  627][2] = -0.121723505109599;
    Leb_Grid_XYZW[  627][3] =  0.000848338957459;

    Leb_Grid_XYZW[  628][0] =  0.904367419939330;
    Leb_Grid_XYZW[  628][1] = -0.409026842708536;
    Leb_Grid_XYZW[  628][2] =  0.121723505109599;
    Leb_Grid_XYZW[  628][3] =  0.000848338957459;

    Leb_Grid_XYZW[  629][0] =  0.904367419939330;
    Leb_Grid_XYZW[  629][1] = -0.409026842708536;
    Leb_Grid_XYZW[  629][2] = -0.121723505109599;
    Leb_Grid_XYZW[  629][3] =  0.000848338957459;

    Leb_Grid_XYZW[  630][0] = -0.904367419939330;
    Leb_Grid_XYZW[  630][1] =  0.409026842708536;
    Leb_Grid_XYZW[  630][2] =  0.121723505109599;
    Leb_Grid_XYZW[  630][3] =  0.000848338957459;

    Leb_Grid_XYZW[  631][0] = -0.904367419939330;
    Leb_Grid_XYZW[  631][1] =  0.409026842708536;
    Leb_Grid_XYZW[  631][2] = -0.121723505109599;
    Leb_Grid_XYZW[  631][3] =  0.000848338957459;

    Leb_Grid_XYZW[  632][0] = -0.904367419939330;
    Leb_Grid_XYZW[  632][1] = -0.409026842708536;
    Leb_Grid_XYZW[  632][2] =  0.121723505109599;
    Leb_Grid_XYZW[  632][3] =  0.000848338957459;

    Leb_Grid_XYZW[  633][0] = -0.904367419939330;
    Leb_Grid_XYZW[  633][1] = -0.409026842708536;
    Leb_Grid_XYZW[  633][2] = -0.121723505109599;
    Leb_Grid_XYZW[  633][3] =  0.000848338957459;

    Leb_Grid_XYZW[  634][0] =  0.904367419939330;
    Leb_Grid_XYZW[  634][1] =  0.121723505109599;
    Leb_Grid_XYZW[  634][2] =  0.409026842708536;
    Leb_Grid_XYZW[  634][3] =  0.000848338957459;

    Leb_Grid_XYZW[  635][0] =  0.904367419939330;
    Leb_Grid_XYZW[  635][1] =  0.121723505109599;
    Leb_Grid_XYZW[  635][2] = -0.409026842708536;
    Leb_Grid_XYZW[  635][3] =  0.000848338957459;

    Leb_Grid_XYZW[  636][0] =  0.904367419939330;
    Leb_Grid_XYZW[  636][1] = -0.121723505109599;
    Leb_Grid_XYZW[  636][2] =  0.409026842708536;
    Leb_Grid_XYZW[  636][3] =  0.000848338957459;

    Leb_Grid_XYZW[  637][0] =  0.904367419939330;
    Leb_Grid_XYZW[  637][1] = -0.121723505109599;
    Leb_Grid_XYZW[  637][2] = -0.409026842708536;
    Leb_Grid_XYZW[  637][3] =  0.000848338957459;

    Leb_Grid_XYZW[  638][0] = -0.904367419939330;
    Leb_Grid_XYZW[  638][1] =  0.121723505109599;
    Leb_Grid_XYZW[  638][2] =  0.409026842708536;
    Leb_Grid_XYZW[  638][3] =  0.000848338957459;

    Leb_Grid_XYZW[  639][0] = -0.904367419939330;
    Leb_Grid_XYZW[  639][1] =  0.121723505109599;
    Leb_Grid_XYZW[  639][2] = -0.409026842708536;
    Leb_Grid_XYZW[  639][3] =  0.000848338957459;

    Leb_Grid_XYZW[  640][0] = -0.904367419939330;
    Leb_Grid_XYZW[  640][1] = -0.121723505109599;
    Leb_Grid_XYZW[  640][2] =  0.409026842708536;
    Leb_Grid_XYZW[  640][3] =  0.000848338957459;

    Leb_Grid_XYZW[  641][0] = -0.904367419939330;
    Leb_Grid_XYZW[  641][1] = -0.121723505109599;
    Leb_Grid_XYZW[  641][2] = -0.409026842708536;
    Leb_Grid_XYZW[  641][3] =  0.000848338957459;

    Leb_Grid_XYZW[  642][0] =  0.409026842708536;
    Leb_Grid_XYZW[  642][1] =  0.904367419939330;
    Leb_Grid_XYZW[  642][2] =  0.121723505109599;
    Leb_Grid_XYZW[  642][3] =  0.000848338957459;

    Leb_Grid_XYZW[  643][0] =  0.409026842708536;
    Leb_Grid_XYZW[  643][1] =  0.904367419939330;
    Leb_Grid_XYZW[  643][2] = -0.121723505109599;
    Leb_Grid_XYZW[  643][3] =  0.000848338957459;

    Leb_Grid_XYZW[  644][0] =  0.409026842708536;
    Leb_Grid_XYZW[  644][1] = -0.904367419939330;
    Leb_Grid_XYZW[  644][2] =  0.121723505109599;
    Leb_Grid_XYZW[  644][3] =  0.000848338957459;

    Leb_Grid_XYZW[  645][0] =  0.409026842708536;
    Leb_Grid_XYZW[  645][1] = -0.904367419939330;
    Leb_Grid_XYZW[  645][2] = -0.121723505109599;
    Leb_Grid_XYZW[  645][3] =  0.000848338957459;

    Leb_Grid_XYZW[  646][0] = -0.409026842708536;
    Leb_Grid_XYZW[  646][1] =  0.904367419939330;
    Leb_Grid_XYZW[  646][2] =  0.121723505109599;
    Leb_Grid_XYZW[  646][3] =  0.000848338957459;

    Leb_Grid_XYZW[  647][0] = -0.409026842708536;
    Leb_Grid_XYZW[  647][1] =  0.904367419939330;
    Leb_Grid_XYZW[  647][2] = -0.121723505109599;
    Leb_Grid_XYZW[  647][3] =  0.000848338957459;

    Leb_Grid_XYZW[  648][0] = -0.409026842708536;
    Leb_Grid_XYZW[  648][1] = -0.904367419939330;
    Leb_Grid_XYZW[  648][2] =  0.121723505109599;
    Leb_Grid_XYZW[  648][3] =  0.000848338957459;

    Leb_Grid_XYZW[  649][0] = -0.409026842708536;
    Leb_Grid_XYZW[  649][1] = -0.904367419939330;
    Leb_Grid_XYZW[  649][2] = -0.121723505109599;
    Leb_Grid_XYZW[  649][3] =  0.000848338957459;

    Leb_Grid_XYZW[  650][0] =  0.409026842708536;
    Leb_Grid_XYZW[  650][1] =  0.121723505109599;
    Leb_Grid_XYZW[  650][2] =  0.904367419939330;
    Leb_Grid_XYZW[  650][3] =  0.000848338957459;

    Leb_Grid_XYZW[  651][0] =  0.409026842708536;
    Leb_Grid_XYZW[  651][1] =  0.121723505109599;
    Leb_Grid_XYZW[  651][2] = -0.904367419939330;
    Leb_Grid_XYZW[  651][3] =  0.000848338957459;

    Leb_Grid_XYZW[  652][0] =  0.409026842708536;
    Leb_Grid_XYZW[  652][1] = -0.121723505109599;
    Leb_Grid_XYZW[  652][2] =  0.904367419939330;
    Leb_Grid_XYZW[  652][3] =  0.000848338957459;

    Leb_Grid_XYZW[  653][0] =  0.409026842708536;
    Leb_Grid_XYZW[  653][1] = -0.121723505109599;
    Leb_Grid_XYZW[  653][2] = -0.904367419939330;
    Leb_Grid_XYZW[  653][3] =  0.000848338957459;

    Leb_Grid_XYZW[  654][0] = -0.409026842708536;
    Leb_Grid_XYZW[  654][1] =  0.121723505109599;
    Leb_Grid_XYZW[  654][2] =  0.904367419939330;
    Leb_Grid_XYZW[  654][3] =  0.000848338957459;

    Leb_Grid_XYZW[  655][0] = -0.409026842708536;
    Leb_Grid_XYZW[  655][1] =  0.121723505109599;
    Leb_Grid_XYZW[  655][2] = -0.904367419939330;
    Leb_Grid_XYZW[  655][3] =  0.000848338957459;

    Leb_Grid_XYZW[  656][0] = -0.409026842708536;
    Leb_Grid_XYZW[  656][1] = -0.121723505109599;
    Leb_Grid_XYZW[  656][2] =  0.904367419939330;
    Leb_Grid_XYZW[  656][3] =  0.000848338957459;

    Leb_Grid_XYZW[  657][0] = -0.409026842708536;
    Leb_Grid_XYZW[  657][1] = -0.121723505109599;
    Leb_Grid_XYZW[  657][2] = -0.904367419939330;
    Leb_Grid_XYZW[  657][3] =  0.000848338957459;

    Leb_Grid_XYZW[  658][0] =  0.121723505109599;
    Leb_Grid_XYZW[  658][1] =  0.904367419939330;
    Leb_Grid_XYZW[  658][2] =  0.409026842708536;
    Leb_Grid_XYZW[  658][3] =  0.000848338957459;

    Leb_Grid_XYZW[  659][0] =  0.121723505109599;
    Leb_Grid_XYZW[  659][1] =  0.904367419939330;
    Leb_Grid_XYZW[  659][2] = -0.409026842708536;
    Leb_Grid_XYZW[  659][3] =  0.000848338957459;

    Leb_Grid_XYZW[  660][0] =  0.121723505109599;
    Leb_Grid_XYZW[  660][1] = -0.904367419939330;
    Leb_Grid_XYZW[  660][2] =  0.409026842708536;
    Leb_Grid_XYZW[  660][3] =  0.000848338957459;

    Leb_Grid_XYZW[  661][0] =  0.121723505109599;
    Leb_Grid_XYZW[  661][1] = -0.904367419939330;
    Leb_Grid_XYZW[  661][2] = -0.409026842708536;
    Leb_Grid_XYZW[  661][3] =  0.000848338957459;

    Leb_Grid_XYZW[  662][0] = -0.121723505109599;
    Leb_Grid_XYZW[  662][1] =  0.904367419939330;
    Leb_Grid_XYZW[  662][2] =  0.409026842708536;
    Leb_Grid_XYZW[  662][3] =  0.000848338957459;

    Leb_Grid_XYZW[  663][0] = -0.121723505109599;
    Leb_Grid_XYZW[  663][1] =  0.904367419939330;
    Leb_Grid_XYZW[  663][2] = -0.409026842708536;
    Leb_Grid_XYZW[  663][3] =  0.000848338957459;

    Leb_Grid_XYZW[  664][0] = -0.121723505109599;
    Leb_Grid_XYZW[  664][1] = -0.904367419939330;
    Leb_Grid_XYZW[  664][2] =  0.409026842708536;
    Leb_Grid_XYZW[  664][3] =  0.000848338957459;

    Leb_Grid_XYZW[  665][0] = -0.121723505109599;
    Leb_Grid_XYZW[  665][1] = -0.904367419939330;
    Leb_Grid_XYZW[  665][2] = -0.409026842708536;
    Leb_Grid_XYZW[  665][3] =  0.000848338957459;

    Leb_Grid_XYZW[  666][0] =  0.121723505109599;
    Leb_Grid_XYZW[  666][1] =  0.409026842708536;
    Leb_Grid_XYZW[  666][2] =  0.904367419939330;
    Leb_Grid_XYZW[  666][3] =  0.000848338957459;

    Leb_Grid_XYZW[  667][0] =  0.121723505109599;
    Leb_Grid_XYZW[  667][1] =  0.409026842708536;
    Leb_Grid_XYZW[  667][2] = -0.904367419939330;
    Leb_Grid_XYZW[  667][3] =  0.000848338957459;

    Leb_Grid_XYZW[  668][0] =  0.121723505109599;
    Leb_Grid_XYZW[  668][1] = -0.409026842708536;
    Leb_Grid_XYZW[  668][2] =  0.904367419939330;
    Leb_Grid_XYZW[  668][3] =  0.000848338957459;

    Leb_Grid_XYZW[  669][0] =  0.121723505109599;
    Leb_Grid_XYZW[  669][1] = -0.409026842708536;
    Leb_Grid_XYZW[  669][2] = -0.904367419939330;
    Leb_Grid_XYZW[  669][3] =  0.000848338957459;

    Leb_Grid_XYZW[  670][0] = -0.121723505109599;
    Leb_Grid_XYZW[  670][1] =  0.409026842708536;
    Leb_Grid_XYZW[  670][2] =  0.904367419939330;
    Leb_Grid_XYZW[  670][3] =  0.000848338957459;

    Leb_Grid_XYZW[  671][0] = -0.121723505109599;
    Leb_Grid_XYZW[  671][1] =  0.409026842708536;
    Leb_Grid_XYZW[  671][2] = -0.904367419939330;
    Leb_Grid_XYZW[  671][3] =  0.000848338957459;

    Leb_Grid_XYZW[  672][0] = -0.121723505109599;
    Leb_Grid_XYZW[  672][1] = -0.409026842708536;
    Leb_Grid_XYZW[  672][2] =  0.904367419939330;
    Leb_Grid_XYZW[  672][3] =  0.000848338957459;

    Leb_Grid_XYZW[  673][0] = -0.121723505109599;
    Leb_Grid_XYZW[  673][1] = -0.409026842708536;
    Leb_Grid_XYZW[  673][2] = -0.904367419939330;
    Leb_Grid_XYZW[  673][3] =  0.000848338957459;

    Leb_Grid_XYZW[  674][0] =  0.891240756007475;
    Leb_Grid_XYZW[  674][1] =  0.385429115066922;
    Leb_Grid_XYZW[  674][2] =  0.239027847938172;
    Leb_Grid_XYZW[  674][3] =  0.000855629925731;

    Leb_Grid_XYZW[  675][0] =  0.891240756007475;
    Leb_Grid_XYZW[  675][1] =  0.385429115066922;
    Leb_Grid_XYZW[  675][2] = -0.239027847938172;
    Leb_Grid_XYZW[  675][3] =  0.000855629925731;

    Leb_Grid_XYZW[  676][0] =  0.891240756007475;
    Leb_Grid_XYZW[  676][1] = -0.385429115066922;
    Leb_Grid_XYZW[  676][2] =  0.239027847938172;
    Leb_Grid_XYZW[  676][3] =  0.000855629925731;

    Leb_Grid_XYZW[  677][0] =  0.891240756007475;
    Leb_Grid_XYZW[  677][1] = -0.385429115066922;
    Leb_Grid_XYZW[  677][2] = -0.239027847938172;
    Leb_Grid_XYZW[  677][3] =  0.000855629925731;

    Leb_Grid_XYZW[  678][0] = -0.891240756007475;
    Leb_Grid_XYZW[  678][1] =  0.385429115066922;
    Leb_Grid_XYZW[  678][2] =  0.239027847938172;
    Leb_Grid_XYZW[  678][3] =  0.000855629925731;

    Leb_Grid_XYZW[  679][0] = -0.891240756007475;
    Leb_Grid_XYZW[  679][1] =  0.385429115066922;
    Leb_Grid_XYZW[  679][2] = -0.239027847938172;
    Leb_Grid_XYZW[  679][3] =  0.000855629925731;

    Leb_Grid_XYZW[  680][0] = -0.891240756007475;
    Leb_Grid_XYZW[  680][1] = -0.385429115066922;
    Leb_Grid_XYZW[  680][2] =  0.239027847938172;
    Leb_Grid_XYZW[  680][3] =  0.000855629925731;

    Leb_Grid_XYZW[  681][0] = -0.891240756007475;
    Leb_Grid_XYZW[  681][1] = -0.385429115066922;
    Leb_Grid_XYZW[  681][2] = -0.239027847938172;
    Leb_Grid_XYZW[  681][3] =  0.000855629925731;

    Leb_Grid_XYZW[  682][0] =  0.891240756007475;
    Leb_Grid_XYZW[  682][1] =  0.239027847938172;
    Leb_Grid_XYZW[  682][2] =  0.385429115066922;
    Leb_Grid_XYZW[  682][3] =  0.000855629925731;

    Leb_Grid_XYZW[  683][0] =  0.891240756007475;
    Leb_Grid_XYZW[  683][1] =  0.239027847938172;
    Leb_Grid_XYZW[  683][2] = -0.385429115066923;
    Leb_Grid_XYZW[  683][3] =  0.000855629925731;

    Leb_Grid_XYZW[  684][0] =  0.891240756007475;
    Leb_Grid_XYZW[  684][1] = -0.239027847938172;
    Leb_Grid_XYZW[  684][2] =  0.385429115066922;
    Leb_Grid_XYZW[  684][3] =  0.000855629925731;

    Leb_Grid_XYZW[  685][0] =  0.891240756007475;
    Leb_Grid_XYZW[  685][1] = -0.239027847938172;
    Leb_Grid_XYZW[  685][2] = -0.385429115066923;
    Leb_Grid_XYZW[  685][3] =  0.000855629925731;

    Leb_Grid_XYZW[  686][0] = -0.891240756007475;
    Leb_Grid_XYZW[  686][1] =  0.239027847938172;
    Leb_Grid_XYZW[  686][2] =  0.385429115066922;
    Leb_Grid_XYZW[  686][3] =  0.000855629925731;

    Leb_Grid_XYZW[  687][0] = -0.891240756007475;
    Leb_Grid_XYZW[  687][1] =  0.239027847938172;
    Leb_Grid_XYZW[  687][2] = -0.385429115066923;
    Leb_Grid_XYZW[  687][3] =  0.000855629925731;

    Leb_Grid_XYZW[  688][0] = -0.891240756007475;
    Leb_Grid_XYZW[  688][1] = -0.239027847938172;
    Leb_Grid_XYZW[  688][2] =  0.385429115066922;
    Leb_Grid_XYZW[  688][3] =  0.000855629925731;

    Leb_Grid_XYZW[  689][0] = -0.891240756007475;
    Leb_Grid_XYZW[  689][1] = -0.239027847938172;
    Leb_Grid_XYZW[  689][2] = -0.385429115066923;
    Leb_Grid_XYZW[  689][3] =  0.000855629925731;

    Leb_Grid_XYZW[  690][0] =  0.385429115066922;
    Leb_Grid_XYZW[  690][1] =  0.891240756007475;
    Leb_Grid_XYZW[  690][2] =  0.239027847938172;
    Leb_Grid_XYZW[  690][3] =  0.000855629925731;

    Leb_Grid_XYZW[  691][0] =  0.385429115066922;
    Leb_Grid_XYZW[  691][1] =  0.891240756007475;
    Leb_Grid_XYZW[  691][2] = -0.239027847938172;
    Leb_Grid_XYZW[  691][3] =  0.000855629925731;

    Leb_Grid_XYZW[  692][0] =  0.385429115066922;
    Leb_Grid_XYZW[  692][1] = -0.891240756007475;
    Leb_Grid_XYZW[  692][2] =  0.239027847938172;
    Leb_Grid_XYZW[  692][3] =  0.000855629925731;

    Leb_Grid_XYZW[  693][0] =  0.385429115066922;
    Leb_Grid_XYZW[  693][1] = -0.891240756007475;
    Leb_Grid_XYZW[  693][2] = -0.239027847938172;
    Leb_Grid_XYZW[  693][3] =  0.000855629925731;

    Leb_Grid_XYZW[  694][0] = -0.385429115066922;
    Leb_Grid_XYZW[  694][1] =  0.891240756007475;
    Leb_Grid_XYZW[  694][2] =  0.239027847938172;
    Leb_Grid_XYZW[  694][3] =  0.000855629925731;

    Leb_Grid_XYZW[  695][0] = -0.385429115066922;
    Leb_Grid_XYZW[  695][1] =  0.891240756007475;
    Leb_Grid_XYZW[  695][2] = -0.239027847938172;
    Leb_Grid_XYZW[  695][3] =  0.000855629925731;

    Leb_Grid_XYZW[  696][0] = -0.385429115066922;
    Leb_Grid_XYZW[  696][1] = -0.891240756007475;
    Leb_Grid_XYZW[  696][2] =  0.239027847938172;
    Leb_Grid_XYZW[  696][3] =  0.000855629925731;

    Leb_Grid_XYZW[  697][0] = -0.385429115066922;
    Leb_Grid_XYZW[  697][1] = -0.891240756007475;
    Leb_Grid_XYZW[  697][2] = -0.239027847938172;
    Leb_Grid_XYZW[  697][3] =  0.000855629925731;

    Leb_Grid_XYZW[  698][0] =  0.385429115066922;
    Leb_Grid_XYZW[  698][1] =  0.239027847938173;
    Leb_Grid_XYZW[  698][2] =  0.891240756007475;
    Leb_Grid_XYZW[  698][3] =  0.000855629925731;

    Leb_Grid_XYZW[  699][0] =  0.385429115066922;
    Leb_Grid_XYZW[  699][1] =  0.239027847938173;
    Leb_Grid_XYZW[  699][2] = -0.891240756007475;
    Leb_Grid_XYZW[  699][3] =  0.000855629925731;

    Leb_Grid_XYZW[  700][0] =  0.385429115066922;
    Leb_Grid_XYZW[  700][1] = -0.239027847938173;
    Leb_Grid_XYZW[  700][2] =  0.891240756007475;
    Leb_Grid_XYZW[  700][3] =  0.000855629925731;

    Leb_Grid_XYZW[  701][0] =  0.385429115066922;
    Leb_Grid_XYZW[  701][1] = -0.239027847938173;
    Leb_Grid_XYZW[  701][2] = -0.891240756007475;
    Leb_Grid_XYZW[  701][3] =  0.000855629925731;

    Leb_Grid_XYZW[  702][0] = -0.385429115066922;
    Leb_Grid_XYZW[  702][1] =  0.239027847938173;
    Leb_Grid_XYZW[  702][2] =  0.891240756007475;
    Leb_Grid_XYZW[  702][3] =  0.000855629925731;

    Leb_Grid_XYZW[  703][0] = -0.385429115066922;
    Leb_Grid_XYZW[  703][1] =  0.239027847938173;
    Leb_Grid_XYZW[  703][2] = -0.891240756007475;
    Leb_Grid_XYZW[  703][3] =  0.000855629925731;

    Leb_Grid_XYZW[  704][0] = -0.385429115066922;
    Leb_Grid_XYZW[  704][1] = -0.239027847938173;
    Leb_Grid_XYZW[  704][2] =  0.891240756007475;
    Leb_Grid_XYZW[  704][3] =  0.000855629925731;

    Leb_Grid_XYZW[  705][0] = -0.385429115066922;
    Leb_Grid_XYZW[  705][1] = -0.239027847938173;
    Leb_Grid_XYZW[  705][2] = -0.891240756007475;
    Leb_Grid_XYZW[  705][3] =  0.000855629925731;

    Leb_Grid_XYZW[  706][0] =  0.239027847938172;
    Leb_Grid_XYZW[  706][1] =  0.891240756007475;
    Leb_Grid_XYZW[  706][2] =  0.385429115066922;
    Leb_Grid_XYZW[  706][3] =  0.000855629925731;

    Leb_Grid_XYZW[  707][0] =  0.239027847938172;
    Leb_Grid_XYZW[  707][1] =  0.891240756007475;
    Leb_Grid_XYZW[  707][2] = -0.385429115066923;
    Leb_Grid_XYZW[  707][3] =  0.000855629925731;

    Leb_Grid_XYZW[  708][0] =  0.239027847938172;
    Leb_Grid_XYZW[  708][1] = -0.891240756007475;
    Leb_Grid_XYZW[  708][2] =  0.385429115066922;
    Leb_Grid_XYZW[  708][3] =  0.000855629925731;

    Leb_Grid_XYZW[  709][0] =  0.239027847938172;
    Leb_Grid_XYZW[  709][1] = -0.891240756007475;
    Leb_Grid_XYZW[  709][2] = -0.385429115066923;
    Leb_Grid_XYZW[  709][3] =  0.000855629925731;

    Leb_Grid_XYZW[  710][0] = -0.239027847938172;
    Leb_Grid_XYZW[  710][1] =  0.891240756007475;
    Leb_Grid_XYZW[  710][2] =  0.385429115066922;
    Leb_Grid_XYZW[  710][3] =  0.000855629925731;

    Leb_Grid_XYZW[  711][0] = -0.239027847938172;
    Leb_Grid_XYZW[  711][1] =  0.891240756007475;
    Leb_Grid_XYZW[  711][2] = -0.385429115066923;
    Leb_Grid_XYZW[  711][3] =  0.000855629925731;

    Leb_Grid_XYZW[  712][0] = -0.239027847938172;
    Leb_Grid_XYZW[  712][1] = -0.891240756007475;
    Leb_Grid_XYZW[  712][2] =  0.385429115066922;
    Leb_Grid_XYZW[  712][3] =  0.000855629925731;

    Leb_Grid_XYZW[  713][0] = -0.239027847938172;
    Leb_Grid_XYZW[  713][1] = -0.891240756007475;
    Leb_Grid_XYZW[  713][2] = -0.385429115066923;
    Leb_Grid_XYZW[  713][3] =  0.000855629925731;

    Leb_Grid_XYZW[  714][0] =  0.239027847938172;
    Leb_Grid_XYZW[  714][1] =  0.385429115066922;
    Leb_Grid_XYZW[  714][2] =  0.891240756007475;
    Leb_Grid_XYZW[  714][3] =  0.000855629925731;

    Leb_Grid_XYZW[  715][0] =  0.239027847938172;
    Leb_Grid_XYZW[  715][1] =  0.385429115066922;
    Leb_Grid_XYZW[  715][2] = -0.891240756007475;
    Leb_Grid_XYZW[  715][3] =  0.000855629925731;

    Leb_Grid_XYZW[  716][0] =  0.239027847938172;
    Leb_Grid_XYZW[  716][1] = -0.385429115066922;
    Leb_Grid_XYZW[  716][2] =  0.891240756007475;
    Leb_Grid_XYZW[  716][3] =  0.000855629925731;

    Leb_Grid_XYZW[  717][0] =  0.239027847938172;
    Leb_Grid_XYZW[  717][1] = -0.385429115066922;
    Leb_Grid_XYZW[  717][2] = -0.891240756007475;
    Leb_Grid_XYZW[  717][3] =  0.000855629925731;

    Leb_Grid_XYZW[  718][0] = -0.239027847938172;
    Leb_Grid_XYZW[  718][1] =  0.385429115066922;
    Leb_Grid_XYZW[  718][2] =  0.891240756007475;
    Leb_Grid_XYZW[  718][3] =  0.000855629925731;

    Leb_Grid_XYZW[  719][0] = -0.239027847938172;
    Leb_Grid_XYZW[  719][1] =  0.385429115066922;
    Leb_Grid_XYZW[  719][2] = -0.891240756007475;
    Leb_Grid_XYZW[  719][3] =  0.000855629925731;

    Leb_Grid_XYZW[  720][0] = -0.239027847938172;
    Leb_Grid_XYZW[  720][1] = -0.385429115066922;
    Leb_Grid_XYZW[  720][2] =  0.891240756007475;
    Leb_Grid_XYZW[  720][3] =  0.000855629925731;

    Leb_Grid_XYZW[  721][0] = -0.239027847938172;
    Leb_Grid_XYZW[  721][1] = -0.385429115066922;
    Leb_Grid_XYZW[  721][2] = -0.891240756007475;
    Leb_Grid_XYZW[  721][3] =  0.000855629925731;

    Leb_Grid_XYZW[  722][0] =  0.867643562846271;
    Leb_Grid_XYZW[  722][1] =  0.493222118485128;
    Leb_Grid_XYZW[  722][2] =  0.062662506241542;
    Leb_Grid_XYZW[  722][3] =  0.000880320867974;

    Leb_Grid_XYZW[  723][0] =  0.867643562846271;
    Leb_Grid_XYZW[  723][1] =  0.493222118485128;
    Leb_Grid_XYZW[  723][2] = -0.062662506241542;
    Leb_Grid_XYZW[  723][3] =  0.000880320867974;

    Leb_Grid_XYZW[  724][0] =  0.867643562846271;
    Leb_Grid_XYZW[  724][1] = -0.493222118485128;
    Leb_Grid_XYZW[  724][2] =  0.062662506241542;
    Leb_Grid_XYZW[  724][3] =  0.000880320867974;

    Leb_Grid_XYZW[  725][0] =  0.867643562846271;
    Leb_Grid_XYZW[  725][1] = -0.493222118485128;
    Leb_Grid_XYZW[  725][2] = -0.062662506241542;
    Leb_Grid_XYZW[  725][3] =  0.000880320867974;

    Leb_Grid_XYZW[  726][0] = -0.867643562846271;
    Leb_Grid_XYZW[  726][1] =  0.493222118485128;
    Leb_Grid_XYZW[  726][2] =  0.062662506241542;
    Leb_Grid_XYZW[  726][3] =  0.000880320867974;

    Leb_Grid_XYZW[  727][0] = -0.867643562846271;
    Leb_Grid_XYZW[  727][1] =  0.493222118485128;
    Leb_Grid_XYZW[  727][2] = -0.062662506241542;
    Leb_Grid_XYZW[  727][3] =  0.000880320867974;

    Leb_Grid_XYZW[  728][0] = -0.867643562846271;
    Leb_Grid_XYZW[  728][1] = -0.493222118485128;
    Leb_Grid_XYZW[  728][2] =  0.062662506241542;
    Leb_Grid_XYZW[  728][3] =  0.000880320867974;

    Leb_Grid_XYZW[  729][0] = -0.867643562846271;
    Leb_Grid_XYZW[  729][1] = -0.493222118485128;
    Leb_Grid_XYZW[  729][2] = -0.062662506241542;
    Leb_Grid_XYZW[  729][3] =  0.000880320867974;

    Leb_Grid_XYZW[  730][0] =  0.867643562846271;
    Leb_Grid_XYZW[  730][1] =  0.062662506241542;
    Leb_Grid_XYZW[  730][2] =  0.493222118485129;
    Leb_Grid_XYZW[  730][3] =  0.000880320867974;

    Leb_Grid_XYZW[  731][0] =  0.867643562846271;
    Leb_Grid_XYZW[  731][1] =  0.062662506241542;
    Leb_Grid_XYZW[  731][2] = -0.493222118485129;
    Leb_Grid_XYZW[  731][3] =  0.000880320867974;

    Leb_Grid_XYZW[  732][0] =  0.867643562846271;
    Leb_Grid_XYZW[  732][1] = -0.062662506241542;
    Leb_Grid_XYZW[  732][2] =  0.493222118485129;
    Leb_Grid_XYZW[  732][3] =  0.000880320867974;

    Leb_Grid_XYZW[  733][0] =  0.867643562846271;
    Leb_Grid_XYZW[  733][1] = -0.062662506241542;
    Leb_Grid_XYZW[  733][2] = -0.493222118485129;
    Leb_Grid_XYZW[  733][3] =  0.000880320867974;

    Leb_Grid_XYZW[  734][0] = -0.867643562846271;
    Leb_Grid_XYZW[  734][1] =  0.062662506241542;
    Leb_Grid_XYZW[  734][2] =  0.493222118485129;
    Leb_Grid_XYZW[  734][3] =  0.000880320867974;

    Leb_Grid_XYZW[  735][0] = -0.867643562846271;
    Leb_Grid_XYZW[  735][1] =  0.062662506241542;
    Leb_Grid_XYZW[  735][2] = -0.493222118485129;
    Leb_Grid_XYZW[  735][3] =  0.000880320867974;

    Leb_Grid_XYZW[  736][0] = -0.867643562846271;
    Leb_Grid_XYZW[  736][1] = -0.062662506241542;
    Leb_Grid_XYZW[  736][2] =  0.493222118485129;
    Leb_Grid_XYZW[  736][3] =  0.000880320867974;

    Leb_Grid_XYZW[  737][0] = -0.867643562846271;
    Leb_Grid_XYZW[  737][1] = -0.062662506241542;
    Leb_Grid_XYZW[  737][2] = -0.493222118485129;
    Leb_Grid_XYZW[  737][3] =  0.000880320867974;

    Leb_Grid_XYZW[  738][0] =  0.493222118485129;
    Leb_Grid_XYZW[  738][1] =  0.867643562846271;
    Leb_Grid_XYZW[  738][2] =  0.062662506241542;
    Leb_Grid_XYZW[  738][3] =  0.000880320867974;

    Leb_Grid_XYZW[  739][0] =  0.493222118485129;
    Leb_Grid_XYZW[  739][1] =  0.867643562846271;
    Leb_Grid_XYZW[  739][2] = -0.062662506241542;
    Leb_Grid_XYZW[  739][3] =  0.000880320867974;

    Leb_Grid_XYZW[  740][0] =  0.493222118485129;
    Leb_Grid_XYZW[  740][1] = -0.867643562846271;
    Leb_Grid_XYZW[  740][2] =  0.062662506241542;
    Leb_Grid_XYZW[  740][3] =  0.000880320867974;

    Leb_Grid_XYZW[  741][0] =  0.493222118485129;
    Leb_Grid_XYZW[  741][1] = -0.867643562846271;
    Leb_Grid_XYZW[  741][2] = -0.062662506241542;
    Leb_Grid_XYZW[  741][3] =  0.000880320867974;

    Leb_Grid_XYZW[  742][0] = -0.493222118485129;
    Leb_Grid_XYZW[  742][1] =  0.867643562846271;
    Leb_Grid_XYZW[  742][2] =  0.062662506241542;
    Leb_Grid_XYZW[  742][3] =  0.000880320867974;

    Leb_Grid_XYZW[  743][0] = -0.493222118485129;
    Leb_Grid_XYZW[  743][1] =  0.867643562846271;
    Leb_Grid_XYZW[  743][2] = -0.062662506241542;
    Leb_Grid_XYZW[  743][3] =  0.000880320867974;

    Leb_Grid_XYZW[  744][0] = -0.493222118485129;
    Leb_Grid_XYZW[  744][1] = -0.867643562846271;
    Leb_Grid_XYZW[  744][2] =  0.062662506241542;
    Leb_Grid_XYZW[  744][3] =  0.000880320867974;

    Leb_Grid_XYZW[  745][0] = -0.493222118485129;
    Leb_Grid_XYZW[  745][1] = -0.867643562846271;
    Leb_Grid_XYZW[  745][2] = -0.062662506241542;
    Leb_Grid_XYZW[  745][3] =  0.000880320867974;

    Leb_Grid_XYZW[  746][0] =  0.493222118485128;
    Leb_Grid_XYZW[  746][1] =  0.062662506241542;
    Leb_Grid_XYZW[  746][2] =  0.867643562846271;
    Leb_Grid_XYZW[  746][3] =  0.000880320867974;

    Leb_Grid_XYZW[  747][0] =  0.493222118485129;
    Leb_Grid_XYZW[  747][1] =  0.062662506241542;
    Leb_Grid_XYZW[  747][2] = -0.867643562846271;
    Leb_Grid_XYZW[  747][3] =  0.000880320867974;

    Leb_Grid_XYZW[  748][0] =  0.493222118485128;
    Leb_Grid_XYZW[  748][1] = -0.062662506241542;
    Leb_Grid_XYZW[  748][2] =  0.867643562846271;
    Leb_Grid_XYZW[  748][3] =  0.000880320867974;

    Leb_Grid_XYZW[  749][0] =  0.493222118485129;
    Leb_Grid_XYZW[  749][1] = -0.062662506241542;
    Leb_Grid_XYZW[  749][2] = -0.867643562846271;
    Leb_Grid_XYZW[  749][3] =  0.000880320867974;

    Leb_Grid_XYZW[  750][0] = -0.493222118485128;
    Leb_Grid_XYZW[  750][1] =  0.062662506241542;
    Leb_Grid_XYZW[  750][2] =  0.867643562846271;
    Leb_Grid_XYZW[  750][3] =  0.000880320867974;

    Leb_Grid_XYZW[  751][0] = -0.493222118485129;
    Leb_Grid_XYZW[  751][1] =  0.062662506241542;
    Leb_Grid_XYZW[  751][2] = -0.867643562846271;
    Leb_Grid_XYZW[  751][3] =  0.000880320867974;

    Leb_Grid_XYZW[  752][0] = -0.493222118485128;
    Leb_Grid_XYZW[  752][1] = -0.062662506241542;
    Leb_Grid_XYZW[  752][2] =  0.867643562846271;
    Leb_Grid_XYZW[  752][3] =  0.000880320867974;

    Leb_Grid_XYZW[  753][0] = -0.493222118485129;
    Leb_Grid_XYZW[  753][1] = -0.062662506241542;
    Leb_Grid_XYZW[  753][2] = -0.867643562846271;
    Leb_Grid_XYZW[  753][3] =  0.000880320867974;

    Leb_Grid_XYZW[  754][0] =  0.062662506241542;
    Leb_Grid_XYZW[  754][1] =  0.867643562846271;
    Leb_Grid_XYZW[  754][2] =  0.493222118485129;
    Leb_Grid_XYZW[  754][3] =  0.000880320867974;

    Leb_Grid_XYZW[  755][0] =  0.062662506241542;
    Leb_Grid_XYZW[  755][1] =  0.867643562846271;
    Leb_Grid_XYZW[  755][2] = -0.493222118485129;
    Leb_Grid_XYZW[  755][3] =  0.000880320867974;

    Leb_Grid_XYZW[  756][0] =  0.062662506241542;
    Leb_Grid_XYZW[  756][1] = -0.867643562846271;
    Leb_Grid_XYZW[  756][2] =  0.493222118485129;
    Leb_Grid_XYZW[  756][3] =  0.000880320867974;

    Leb_Grid_XYZW[  757][0] =  0.062662506241542;
    Leb_Grid_XYZW[  757][1] = -0.867643562846271;
    Leb_Grid_XYZW[  757][2] = -0.493222118485129;
    Leb_Grid_XYZW[  757][3] =  0.000880320867974;

    Leb_Grid_XYZW[  758][0] = -0.062662506241542;
    Leb_Grid_XYZW[  758][1] =  0.867643562846271;
    Leb_Grid_XYZW[  758][2] =  0.493222118485129;
    Leb_Grid_XYZW[  758][3] =  0.000880320867974;

    Leb_Grid_XYZW[  759][0] = -0.062662506241542;
    Leb_Grid_XYZW[  759][1] =  0.867643562846271;
    Leb_Grid_XYZW[  759][2] = -0.493222118485129;
    Leb_Grid_XYZW[  759][3] =  0.000880320867974;

    Leb_Grid_XYZW[  760][0] = -0.062662506241542;
    Leb_Grid_XYZW[  760][1] = -0.867643562846271;
    Leb_Grid_XYZW[  760][2] =  0.493222118485129;
    Leb_Grid_XYZW[  760][3] =  0.000880320867974;

    Leb_Grid_XYZW[  761][0] = -0.062662506241542;
    Leb_Grid_XYZW[  761][1] = -0.867643562846271;
    Leb_Grid_XYZW[  761][2] = -0.493222118485129;
    Leb_Grid_XYZW[  761][3] =  0.000880320867974;

    Leb_Grid_XYZW[  762][0] =  0.062662506241542;
    Leb_Grid_XYZW[  762][1] =  0.493222118485128;
    Leb_Grid_XYZW[  762][2] =  0.867643562846271;
    Leb_Grid_XYZW[  762][3] =  0.000880320867974;

    Leb_Grid_XYZW[  763][0] =  0.062662506241542;
    Leb_Grid_XYZW[  763][1] =  0.493222118485129;
    Leb_Grid_XYZW[  763][2] = -0.867643562846271;
    Leb_Grid_XYZW[  763][3] =  0.000880320867974;

    Leb_Grid_XYZW[  764][0] =  0.062662506241542;
    Leb_Grid_XYZW[  764][1] = -0.493222118485128;
    Leb_Grid_XYZW[  764][2] =  0.867643562846271;
    Leb_Grid_XYZW[  764][3] =  0.000880320867974;

    Leb_Grid_XYZW[  765][0] =  0.062662506241542;
    Leb_Grid_XYZW[  765][1] = -0.493222118485129;
    Leb_Grid_XYZW[  765][2] = -0.867643562846271;
    Leb_Grid_XYZW[  765][3] =  0.000880320867974;

    Leb_Grid_XYZW[  766][0] = -0.062662506241542;
    Leb_Grid_XYZW[  766][1] =  0.493222118485128;
    Leb_Grid_XYZW[  766][2] =  0.867643562846271;
    Leb_Grid_XYZW[  766][3] =  0.000880320867974;

    Leb_Grid_XYZW[  767][0] = -0.062662506241542;
    Leb_Grid_XYZW[  767][1] =  0.493222118485129;
    Leb_Grid_XYZW[  767][2] = -0.867643562846271;
    Leb_Grid_XYZW[  767][3] =  0.000880320867974;

    Leb_Grid_XYZW[  768][0] = -0.062662506241542;
    Leb_Grid_XYZW[  768][1] = -0.493222118485128;
    Leb_Grid_XYZW[  768][2] =  0.867643562846271;
    Leb_Grid_XYZW[  768][3] =  0.000880320867974;

    Leb_Grid_XYZW[  769][0] = -0.062662506241542;
    Leb_Grid_XYZW[  769][1] = -0.493222118485129;
    Leb_Grid_XYZW[  769][2] = -0.867643562846271;
    Leb_Grid_XYZW[  769][3] =  0.000880320867974;

    Leb_Grid_XYZW[  770][0] =  0.858197998604162;
    Leb_Grid_XYZW[  770][1] =  0.478532067592244;
    Leb_Grid_XYZW[  770][2] =  0.185750519454734;
    Leb_Grid_XYZW[  770][3] =  0.000881104818243;

    Leb_Grid_XYZW[  771][0] =  0.858197998604162;
    Leb_Grid_XYZW[  771][1] =  0.478532067592244;
    Leb_Grid_XYZW[  771][2] = -0.185750519454734;
    Leb_Grid_XYZW[  771][3] =  0.000881104818243;

    Leb_Grid_XYZW[  772][0] =  0.858197998604162;
    Leb_Grid_XYZW[  772][1] = -0.478532067592244;
    Leb_Grid_XYZW[  772][2] =  0.185750519454734;
    Leb_Grid_XYZW[  772][3] =  0.000881104818243;

    Leb_Grid_XYZW[  773][0] =  0.858197998604162;
    Leb_Grid_XYZW[  773][1] = -0.478532067592244;
    Leb_Grid_XYZW[  773][2] = -0.185750519454734;
    Leb_Grid_XYZW[  773][3] =  0.000881104818243;

    Leb_Grid_XYZW[  774][0] = -0.858197998604162;
    Leb_Grid_XYZW[  774][1] =  0.478532067592244;
    Leb_Grid_XYZW[  774][2] =  0.185750519454734;
    Leb_Grid_XYZW[  774][3] =  0.000881104818243;

    Leb_Grid_XYZW[  775][0] = -0.858197998604162;
    Leb_Grid_XYZW[  775][1] =  0.478532067592244;
    Leb_Grid_XYZW[  775][2] = -0.185750519454734;
    Leb_Grid_XYZW[  775][3] =  0.000881104818243;

    Leb_Grid_XYZW[  776][0] = -0.858197998604162;
    Leb_Grid_XYZW[  776][1] = -0.478532067592244;
    Leb_Grid_XYZW[  776][2] =  0.185750519454734;
    Leb_Grid_XYZW[  776][3] =  0.000881104818243;

    Leb_Grid_XYZW[  777][0] = -0.858197998604162;
    Leb_Grid_XYZW[  777][1] = -0.478532067592244;
    Leb_Grid_XYZW[  777][2] = -0.185750519454734;
    Leb_Grid_XYZW[  777][3] =  0.000881104818243;

    Leb_Grid_XYZW[  778][0] =  0.858197998604162;
    Leb_Grid_XYZW[  778][1] =  0.185750519454734;
    Leb_Grid_XYZW[  778][2] =  0.478532067592243;
    Leb_Grid_XYZW[  778][3] =  0.000881104818243;

    Leb_Grid_XYZW[  779][0] =  0.858197998604162;
    Leb_Grid_XYZW[  779][1] =  0.185750519454734;
    Leb_Grid_XYZW[  779][2] = -0.478532067592244;
    Leb_Grid_XYZW[  779][3] =  0.000881104818243;

    Leb_Grid_XYZW[  780][0] =  0.858197998604162;
    Leb_Grid_XYZW[  780][1] = -0.185750519454734;
    Leb_Grid_XYZW[  780][2] =  0.478532067592243;
    Leb_Grid_XYZW[  780][3] =  0.000881104818243;

    Leb_Grid_XYZW[  781][0] =  0.858197998604162;
    Leb_Grid_XYZW[  781][1] = -0.185750519454734;
    Leb_Grid_XYZW[  781][2] = -0.478532067592244;
    Leb_Grid_XYZW[  781][3] =  0.000881104818243;

    Leb_Grid_XYZW[  782][0] = -0.858197998604162;
    Leb_Grid_XYZW[  782][1] =  0.185750519454733;
    Leb_Grid_XYZW[  782][2] =  0.478532067592243;
    Leb_Grid_XYZW[  782][3] =  0.000881104818243;

    Leb_Grid_XYZW[  783][0] = -0.858197998604162;
    Leb_Grid_XYZW[  783][1] =  0.185750519454733;
    Leb_Grid_XYZW[  783][2] = -0.478532067592244;
    Leb_Grid_XYZW[  783][3] =  0.000881104818243;

    Leb_Grid_XYZW[  784][0] = -0.858197998604162;
    Leb_Grid_XYZW[  784][1] = -0.185750519454733;
    Leb_Grid_XYZW[  784][2] =  0.478532067592243;
    Leb_Grid_XYZW[  784][3] =  0.000881104818243;

    Leb_Grid_XYZW[  785][0] = -0.858197998604162;
    Leb_Grid_XYZW[  785][1] = -0.185750519454733;
    Leb_Grid_XYZW[  785][2] = -0.478532067592244;
    Leb_Grid_XYZW[  785][3] =  0.000881104818243;

    Leb_Grid_XYZW[  786][0] =  0.478532067592243;
    Leb_Grid_XYZW[  786][1] =  0.858197998604162;
    Leb_Grid_XYZW[  786][2] =  0.185750519454734;
    Leb_Grid_XYZW[  786][3] =  0.000881104818243;

    Leb_Grid_XYZW[  787][0] =  0.478532067592243;
    Leb_Grid_XYZW[  787][1] =  0.858197998604162;
    Leb_Grid_XYZW[  787][2] = -0.185750519454734;
    Leb_Grid_XYZW[  787][3] =  0.000881104818243;

    Leb_Grid_XYZW[  788][0] =  0.478532067592243;
    Leb_Grid_XYZW[  788][1] = -0.858197998604162;
    Leb_Grid_XYZW[  788][2] =  0.185750519454734;
    Leb_Grid_XYZW[  788][3] =  0.000881104818243;

    Leb_Grid_XYZW[  789][0] =  0.478532067592243;
    Leb_Grid_XYZW[  789][1] = -0.858197998604162;
    Leb_Grid_XYZW[  789][2] = -0.185750519454734;
    Leb_Grid_XYZW[  789][3] =  0.000881104818243;

    Leb_Grid_XYZW[  790][0] = -0.478532067592243;
    Leb_Grid_XYZW[  790][1] =  0.858197998604162;
    Leb_Grid_XYZW[  790][2] =  0.185750519454734;
    Leb_Grid_XYZW[  790][3] =  0.000881104818243;

    Leb_Grid_XYZW[  791][0] = -0.478532067592243;
    Leb_Grid_XYZW[  791][1] =  0.858197998604162;
    Leb_Grid_XYZW[  791][2] = -0.185750519454734;
    Leb_Grid_XYZW[  791][3] =  0.000881104818243;

    Leb_Grid_XYZW[  792][0] = -0.478532067592243;
    Leb_Grid_XYZW[  792][1] = -0.858197998604162;
    Leb_Grid_XYZW[  792][2] =  0.185750519454734;
    Leb_Grid_XYZW[  792][3] =  0.000881104818243;

    Leb_Grid_XYZW[  793][0] = -0.478532067592243;
    Leb_Grid_XYZW[  793][1] = -0.858197998604162;
    Leb_Grid_XYZW[  793][2] = -0.185750519454734;
    Leb_Grid_XYZW[  793][3] =  0.000881104818243;

    Leb_Grid_XYZW[  794][0] =  0.478532067592243;
    Leb_Grid_XYZW[  794][1] =  0.185750519454734;
    Leb_Grid_XYZW[  794][2] =  0.858197998604162;
    Leb_Grid_XYZW[  794][3] =  0.000881104818243;

    Leb_Grid_XYZW[  795][0] =  0.478532067592243;
    Leb_Grid_XYZW[  795][1] =  0.185750519454734;
    Leb_Grid_XYZW[  795][2] = -0.858197998604162;
    Leb_Grid_XYZW[  795][3] =  0.000881104818243;

    Leb_Grid_XYZW[  796][0] =  0.478532067592243;
    Leb_Grid_XYZW[  796][1] = -0.185750519454734;
    Leb_Grid_XYZW[  796][2] =  0.858197998604162;
    Leb_Grid_XYZW[  796][3] =  0.000881104818243;

    Leb_Grid_XYZW[  797][0] =  0.478532067592243;
    Leb_Grid_XYZW[  797][1] = -0.185750519454734;
    Leb_Grid_XYZW[  797][2] = -0.858197998604162;
    Leb_Grid_XYZW[  797][3] =  0.000881104818243;

    Leb_Grid_XYZW[  798][0] = -0.478532067592244;
    Leb_Grid_XYZW[  798][1] =  0.185750519454734;
    Leb_Grid_XYZW[  798][2] =  0.858197998604162;
    Leb_Grid_XYZW[  798][3] =  0.000881104818243;

    Leb_Grid_XYZW[  799][0] = -0.478532067592243;
    Leb_Grid_XYZW[  799][1] =  0.185750519454734;
    Leb_Grid_XYZW[  799][2] = -0.858197998604162;
    Leb_Grid_XYZW[  799][3] =  0.000881104818243;

    Leb_Grid_XYZW[  800][0] = -0.478532067592244;
    Leb_Grid_XYZW[  800][1] = -0.185750519454734;
    Leb_Grid_XYZW[  800][2] =  0.858197998604162;
    Leb_Grid_XYZW[  800][3] =  0.000881104818243;

    Leb_Grid_XYZW[  801][0] = -0.478532067592243;
    Leb_Grid_XYZW[  801][1] = -0.185750519454734;
    Leb_Grid_XYZW[  801][2] = -0.858197998604162;
    Leb_Grid_XYZW[  801][3] =  0.000881104818243;

    Leb_Grid_XYZW[  802][0] =  0.185750519454734;
    Leb_Grid_XYZW[  802][1] =  0.858197998604162;
    Leb_Grid_XYZW[  802][2] =  0.478532067592243;
    Leb_Grid_XYZW[  802][3] =  0.000881104818243;

    Leb_Grid_XYZW[  803][0] =  0.185750519454734;
    Leb_Grid_XYZW[  803][1] =  0.858197998604162;
    Leb_Grid_XYZW[  803][2] = -0.478532067592244;
    Leb_Grid_XYZW[  803][3] =  0.000881104818243;

    Leb_Grid_XYZW[  804][0] =  0.185750519454734;
    Leb_Grid_XYZW[  804][1] = -0.858197998604162;
    Leb_Grid_XYZW[  804][2] =  0.478532067592243;
    Leb_Grid_XYZW[  804][3] =  0.000881104818243;

    Leb_Grid_XYZW[  805][0] =  0.185750519454734;
    Leb_Grid_XYZW[  805][1] = -0.858197998604162;
    Leb_Grid_XYZW[  805][2] = -0.478532067592244;
    Leb_Grid_XYZW[  805][3] =  0.000881104818243;

    Leb_Grid_XYZW[  806][0] = -0.185750519454734;
    Leb_Grid_XYZW[  806][1] =  0.858197998604162;
    Leb_Grid_XYZW[  806][2] =  0.478532067592243;
    Leb_Grid_XYZW[  806][3] =  0.000881104818243;

    Leb_Grid_XYZW[  807][0] = -0.185750519454734;
    Leb_Grid_XYZW[  807][1] =  0.858197998604162;
    Leb_Grid_XYZW[  807][2] = -0.478532067592244;
    Leb_Grid_XYZW[  807][3] =  0.000881104818243;

    Leb_Grid_XYZW[  808][0] = -0.185750519454734;
    Leb_Grid_XYZW[  808][1] = -0.858197998604162;
    Leb_Grid_XYZW[  808][2] =  0.478532067592243;
    Leb_Grid_XYZW[  808][3] =  0.000881104818243;

    Leb_Grid_XYZW[  809][0] = -0.185750519454734;
    Leb_Grid_XYZW[  809][1] = -0.858197998604162;
    Leb_Grid_XYZW[  809][2] = -0.478532067592244;
    Leb_Grid_XYZW[  809][3] =  0.000881104818243;

    Leb_Grid_XYZW[  810][0] =  0.185750519454734;
    Leb_Grid_XYZW[  810][1] =  0.478532067592244;
    Leb_Grid_XYZW[  810][2] =  0.858197998604162;
    Leb_Grid_XYZW[  810][3] =  0.000881104818243;

    Leb_Grid_XYZW[  811][0] =  0.185750519454734;
    Leb_Grid_XYZW[  811][1] =  0.478532067592243;
    Leb_Grid_XYZW[  811][2] = -0.858197998604162;
    Leb_Grid_XYZW[  811][3] =  0.000881104818243;

    Leb_Grid_XYZW[  812][0] =  0.185750519454734;
    Leb_Grid_XYZW[  812][1] = -0.478532067592244;
    Leb_Grid_XYZW[  812][2] =  0.858197998604162;
    Leb_Grid_XYZW[  812][3] =  0.000881104818243;

    Leb_Grid_XYZW[  813][0] =  0.185750519454734;
    Leb_Grid_XYZW[  813][1] = -0.478532067592243;
    Leb_Grid_XYZW[  813][2] = -0.858197998604162;
    Leb_Grid_XYZW[  813][3] =  0.000881104818243;

    Leb_Grid_XYZW[  814][0] = -0.185750519454734;
    Leb_Grid_XYZW[  814][1] =  0.478532067592244;
    Leb_Grid_XYZW[  814][2] =  0.858197998604162;
    Leb_Grid_XYZW[  814][3] =  0.000881104818243;

    Leb_Grid_XYZW[  815][0] = -0.185750519454734;
    Leb_Grid_XYZW[  815][1] =  0.478532067592243;
    Leb_Grid_XYZW[  815][2] = -0.858197998604162;
    Leb_Grid_XYZW[  815][3] =  0.000881104818243;

    Leb_Grid_XYZW[  816][0] = -0.185750519454734;
    Leb_Grid_XYZW[  816][1] = -0.478532067592244;
    Leb_Grid_XYZW[  816][2] =  0.858197998604162;
    Leb_Grid_XYZW[  816][3] =  0.000881104818243;

    Leb_Grid_XYZW[  817][0] = -0.185750519454734;
    Leb_Grid_XYZW[  817][1] = -0.478532067592243;
    Leb_Grid_XYZW[  817][2] = -0.858197998604162;
    Leb_Grid_XYZW[  817][3] =  0.000881104818243;

    Leb_Grid_XYZW[  818][0] =  0.839675362404986;
    Leb_Grid_XYZW[  818][1] =  0.450742259315706;
    Leb_Grid_XYZW[  818][2] =  0.302946697352898;
    Leb_Grid_XYZW[  818][3] =  0.000885028234127;

    Leb_Grid_XYZW[  819][0] =  0.839675362404986;
    Leb_Grid_XYZW[  819][1] =  0.450742259315706;
    Leb_Grid_XYZW[  819][2] = -0.302946697352898;
    Leb_Grid_XYZW[  819][3] =  0.000885028234127;

    Leb_Grid_XYZW[  820][0] =  0.839675362404986;
    Leb_Grid_XYZW[  820][1] = -0.450742259315706;
    Leb_Grid_XYZW[  820][2] =  0.302946697352898;
    Leb_Grid_XYZW[  820][3] =  0.000885028234127;

    Leb_Grid_XYZW[  821][0] =  0.839675362404986;
    Leb_Grid_XYZW[  821][1] = -0.450742259315706;
    Leb_Grid_XYZW[  821][2] = -0.302946697352898;
    Leb_Grid_XYZW[  821][3] =  0.000885028234127;

    Leb_Grid_XYZW[  822][0] = -0.839675362404986;
    Leb_Grid_XYZW[  822][1] =  0.450742259315706;
    Leb_Grid_XYZW[  822][2] =  0.302946697352898;
    Leb_Grid_XYZW[  822][3] =  0.000885028234127;

    Leb_Grid_XYZW[  823][0] = -0.839675362404986;
    Leb_Grid_XYZW[  823][1] =  0.450742259315706;
    Leb_Grid_XYZW[  823][2] = -0.302946697352898;
    Leb_Grid_XYZW[  823][3] =  0.000885028234127;

    Leb_Grid_XYZW[  824][0] = -0.839675362404986;
    Leb_Grid_XYZW[  824][1] = -0.450742259315706;
    Leb_Grid_XYZW[  824][2] =  0.302946697352898;
    Leb_Grid_XYZW[  824][3] =  0.000885028234127;

    Leb_Grid_XYZW[  825][0] = -0.839675362404986;
    Leb_Grid_XYZW[  825][1] = -0.450742259315706;
    Leb_Grid_XYZW[  825][2] = -0.302946697352898;
    Leb_Grid_XYZW[  825][3] =  0.000885028234127;

    Leb_Grid_XYZW[  826][0] =  0.839675362404986;
    Leb_Grid_XYZW[  826][1] =  0.302946697352898;
    Leb_Grid_XYZW[  826][2] =  0.450742259315706;
    Leb_Grid_XYZW[  826][3] =  0.000885028234127;

    Leb_Grid_XYZW[  827][0] =  0.839675362404986;
    Leb_Grid_XYZW[  827][1] =  0.302946697352898;
    Leb_Grid_XYZW[  827][2] = -0.450742259315706;
    Leb_Grid_XYZW[  827][3] =  0.000885028234127;

    Leb_Grid_XYZW[  828][0] =  0.839675362404986;
    Leb_Grid_XYZW[  828][1] = -0.302946697352898;
    Leb_Grid_XYZW[  828][2] =  0.450742259315706;
    Leb_Grid_XYZW[  828][3] =  0.000885028234127;

    Leb_Grid_XYZW[  829][0] =  0.839675362404986;
    Leb_Grid_XYZW[  829][1] = -0.302946697352898;
    Leb_Grid_XYZW[  829][2] = -0.450742259315706;
    Leb_Grid_XYZW[  829][3] =  0.000885028234127;

    Leb_Grid_XYZW[  830][0] = -0.839675362404986;
    Leb_Grid_XYZW[  830][1] =  0.302946697352898;
    Leb_Grid_XYZW[  830][2] =  0.450742259315706;
    Leb_Grid_XYZW[  830][3] =  0.000885028234127;

    Leb_Grid_XYZW[  831][0] = -0.839675362404986;
    Leb_Grid_XYZW[  831][1] =  0.302946697352898;
    Leb_Grid_XYZW[  831][2] = -0.450742259315706;
    Leb_Grid_XYZW[  831][3] =  0.000885028234127;

    Leb_Grid_XYZW[  832][0] = -0.839675362404986;
    Leb_Grid_XYZW[  832][1] = -0.302946697352898;
    Leb_Grid_XYZW[  832][2] =  0.450742259315706;
    Leb_Grid_XYZW[  832][3] =  0.000885028234127;

    Leb_Grid_XYZW[  833][0] = -0.839675362404986;
    Leb_Grid_XYZW[  833][1] = -0.302946697352898;
    Leb_Grid_XYZW[  833][2] = -0.450742259315706;
    Leb_Grid_XYZW[  833][3] =  0.000885028234127;

    Leb_Grid_XYZW[  834][0] =  0.450742259315707;
    Leb_Grid_XYZW[  834][1] =  0.839675362404986;
    Leb_Grid_XYZW[  834][2] =  0.302946697352898;
    Leb_Grid_XYZW[  834][3] =  0.000885028234127;

    Leb_Grid_XYZW[  835][0] =  0.450742259315707;
    Leb_Grid_XYZW[  835][1] =  0.839675362404986;
    Leb_Grid_XYZW[  835][2] = -0.302946697352898;
    Leb_Grid_XYZW[  835][3] =  0.000885028234127;

    Leb_Grid_XYZW[  836][0] =  0.450742259315707;
    Leb_Grid_XYZW[  836][1] = -0.839675362404986;
    Leb_Grid_XYZW[  836][2] =  0.302946697352898;
    Leb_Grid_XYZW[  836][3] =  0.000885028234127;

    Leb_Grid_XYZW[  837][0] =  0.450742259315707;
    Leb_Grid_XYZW[  837][1] = -0.839675362404986;
    Leb_Grid_XYZW[  837][2] = -0.302946697352898;
    Leb_Grid_XYZW[  837][3] =  0.000885028234127;

    Leb_Grid_XYZW[  838][0] = -0.450742259315706;
    Leb_Grid_XYZW[  838][1] =  0.839675362404986;
    Leb_Grid_XYZW[  838][2] =  0.302946697352898;
    Leb_Grid_XYZW[  838][3] =  0.000885028234127;

    Leb_Grid_XYZW[  839][0] = -0.450742259315706;
    Leb_Grid_XYZW[  839][1] =  0.839675362404986;
    Leb_Grid_XYZW[  839][2] = -0.302946697352898;
    Leb_Grid_XYZW[  839][3] =  0.000885028234127;

    Leb_Grid_XYZW[  840][0] = -0.450742259315706;
    Leb_Grid_XYZW[  840][1] = -0.839675362404986;
    Leb_Grid_XYZW[  840][2] =  0.302946697352898;
    Leb_Grid_XYZW[  840][3] =  0.000885028234127;

    Leb_Grid_XYZW[  841][0] = -0.450742259315706;
    Leb_Grid_XYZW[  841][1] = -0.839675362404986;
    Leb_Grid_XYZW[  841][2] = -0.302946697352898;
    Leb_Grid_XYZW[  841][3] =  0.000885028234127;

    Leb_Grid_XYZW[  842][0] =  0.450742259315706;
    Leb_Grid_XYZW[  842][1] =  0.302946697352898;
    Leb_Grid_XYZW[  842][2] =  0.839675362404986;
    Leb_Grid_XYZW[  842][3] =  0.000885028234127;

    Leb_Grid_XYZW[  843][0] =  0.450742259315706;
    Leb_Grid_XYZW[  843][1] =  0.302946697352898;
    Leb_Grid_XYZW[  843][2] = -0.839675362404986;
    Leb_Grid_XYZW[  843][3] =  0.000885028234127;

    Leb_Grid_XYZW[  844][0] =  0.450742259315706;
    Leb_Grid_XYZW[  844][1] = -0.302946697352898;
    Leb_Grid_XYZW[  844][2] =  0.839675362404986;
    Leb_Grid_XYZW[  844][3] =  0.000885028234127;

    Leb_Grid_XYZW[  845][0] =  0.450742259315706;
    Leb_Grid_XYZW[  845][1] = -0.302946697352898;
    Leb_Grid_XYZW[  845][2] = -0.839675362404986;
    Leb_Grid_XYZW[  845][3] =  0.000885028234127;

    Leb_Grid_XYZW[  846][0] = -0.450742259315706;
    Leb_Grid_XYZW[  846][1] =  0.302946697352898;
    Leb_Grid_XYZW[  846][2] =  0.839675362404986;
    Leb_Grid_XYZW[  846][3] =  0.000885028234127;

    Leb_Grid_XYZW[  847][0] = -0.450742259315706;
    Leb_Grid_XYZW[  847][1] =  0.302946697352898;
    Leb_Grid_XYZW[  847][2] = -0.839675362404986;
    Leb_Grid_XYZW[  847][3] =  0.000885028234127;

    Leb_Grid_XYZW[  848][0] = -0.450742259315706;
    Leb_Grid_XYZW[  848][1] = -0.302946697352898;
    Leb_Grid_XYZW[  848][2] =  0.839675362404986;
    Leb_Grid_XYZW[  848][3] =  0.000885028234127;

    Leb_Grid_XYZW[  849][0] = -0.450742259315706;
    Leb_Grid_XYZW[  849][1] = -0.302946697352898;
    Leb_Grid_XYZW[  849][2] = -0.839675362404986;
    Leb_Grid_XYZW[  849][3] =  0.000885028234127;

    Leb_Grid_XYZW[  850][0] =  0.302946697352898;
    Leb_Grid_XYZW[  850][1] =  0.839675362404986;
    Leb_Grid_XYZW[  850][2] =  0.450742259315706;
    Leb_Grid_XYZW[  850][3] =  0.000885028234127;

    Leb_Grid_XYZW[  851][0] =  0.302946697352898;
    Leb_Grid_XYZW[  851][1] =  0.839675362404986;
    Leb_Grid_XYZW[  851][2] = -0.450742259315706;
    Leb_Grid_XYZW[  851][3] =  0.000885028234127;

    Leb_Grid_XYZW[  852][0] =  0.302946697352898;
    Leb_Grid_XYZW[  852][1] = -0.839675362404986;
    Leb_Grid_XYZW[  852][2] =  0.450742259315706;
    Leb_Grid_XYZW[  852][3] =  0.000885028234127;

    Leb_Grid_XYZW[  853][0] =  0.302946697352898;
    Leb_Grid_XYZW[  853][1] = -0.839675362404986;
    Leb_Grid_XYZW[  853][2] = -0.450742259315706;
    Leb_Grid_XYZW[  853][3] =  0.000885028234127;

    Leb_Grid_XYZW[  854][0] = -0.302946697352898;
    Leb_Grid_XYZW[  854][1] =  0.839675362404986;
    Leb_Grid_XYZW[  854][2] =  0.450742259315706;
    Leb_Grid_XYZW[  854][3] =  0.000885028234127;

    Leb_Grid_XYZW[  855][0] = -0.302946697352898;
    Leb_Grid_XYZW[  855][1] =  0.839675362404986;
    Leb_Grid_XYZW[  855][2] = -0.450742259315706;
    Leb_Grid_XYZW[  855][3] =  0.000885028234127;

    Leb_Grid_XYZW[  856][0] = -0.302946697352898;
    Leb_Grid_XYZW[  856][1] = -0.839675362404986;
    Leb_Grid_XYZW[  856][2] =  0.450742259315706;
    Leb_Grid_XYZW[  856][3] =  0.000885028234127;

    Leb_Grid_XYZW[  857][0] = -0.302946697352898;
    Leb_Grid_XYZW[  857][1] = -0.839675362404986;
    Leb_Grid_XYZW[  857][2] = -0.450742259315706;
    Leb_Grid_XYZW[  857][3] =  0.000885028234127;

    Leb_Grid_XYZW[  858][0] =  0.302946697352898;
    Leb_Grid_XYZW[  858][1] =  0.450742259315706;
    Leb_Grid_XYZW[  858][2] =  0.839675362404986;
    Leb_Grid_XYZW[  858][3] =  0.000885028234127;

    Leb_Grid_XYZW[  859][0] =  0.302946697352898;
    Leb_Grid_XYZW[  859][1] =  0.450742259315706;
    Leb_Grid_XYZW[  859][2] = -0.839675362404986;
    Leb_Grid_XYZW[  859][3] =  0.000885028234127;

    Leb_Grid_XYZW[  860][0] =  0.302946697352898;
    Leb_Grid_XYZW[  860][1] = -0.450742259315706;
    Leb_Grid_XYZW[  860][2] =  0.839675362404986;
    Leb_Grid_XYZW[  860][3] =  0.000885028234127;

    Leb_Grid_XYZW[  861][0] =  0.302946697352898;
    Leb_Grid_XYZW[  861][1] = -0.450742259315706;
    Leb_Grid_XYZW[  861][2] = -0.839675362404986;
    Leb_Grid_XYZW[  861][3] =  0.000885028234127;

    Leb_Grid_XYZW[  862][0] = -0.302946697352898;
    Leb_Grid_XYZW[  862][1] =  0.450742259315706;
    Leb_Grid_XYZW[  862][2] =  0.839675362404986;
    Leb_Grid_XYZW[  862][3] =  0.000885028234127;

    Leb_Grid_XYZW[  863][0] = -0.302946697352898;
    Leb_Grid_XYZW[  863][1] =  0.450742259315706;
    Leb_Grid_XYZW[  863][2] = -0.839675362404986;
    Leb_Grid_XYZW[  863][3] =  0.000885028234127;

    Leb_Grid_XYZW[  864][0] = -0.302946697352898;
    Leb_Grid_XYZW[  864][1] = -0.450742259315706;
    Leb_Grid_XYZW[  864][2] =  0.839675362404986;
    Leb_Grid_XYZW[  864][3] =  0.000885028234127;

    Leb_Grid_XYZW[  865][0] = -0.302946697352898;
    Leb_Grid_XYZW[  865][1] = -0.450742259315706;
    Leb_Grid_XYZW[  865][2] = -0.839675362404986;
    Leb_Grid_XYZW[  865][3] =  0.000885028234127;

    Leb_Grid_XYZW[  866][0] =  0.816528856402219;
    Leb_Grid_XYZW[  866][1] =  0.563212302076210;
    Leb_Grid_XYZW[  866][2] =  0.126777480068428;
    Leb_Grid_XYZW[  866][3] =  0.000902134229904;

    Leb_Grid_XYZW[  867][0] =  0.816528856402219;
    Leb_Grid_XYZW[  867][1] =  0.563212302076210;
    Leb_Grid_XYZW[  867][2] = -0.126777480068428;
    Leb_Grid_XYZW[  867][3] =  0.000902134229904;

    Leb_Grid_XYZW[  868][0] =  0.816528856402219;
    Leb_Grid_XYZW[  868][1] = -0.563212302076210;
    Leb_Grid_XYZW[  868][2] =  0.126777480068428;
    Leb_Grid_XYZW[  868][3] =  0.000902134229904;

    Leb_Grid_XYZW[  869][0] =  0.816528856402219;
    Leb_Grid_XYZW[  869][1] = -0.563212302076210;
    Leb_Grid_XYZW[  869][2] = -0.126777480068428;
    Leb_Grid_XYZW[  869][3] =  0.000902134229904;

    Leb_Grid_XYZW[  870][0] = -0.816528856402219;
    Leb_Grid_XYZW[  870][1] =  0.563212302076210;
    Leb_Grid_XYZW[  870][2] =  0.126777480068428;
    Leb_Grid_XYZW[  870][3] =  0.000902134229904;

    Leb_Grid_XYZW[  871][0] = -0.816528856402219;
    Leb_Grid_XYZW[  871][1] =  0.563212302076210;
    Leb_Grid_XYZW[  871][2] = -0.126777480068428;
    Leb_Grid_XYZW[  871][3] =  0.000902134229904;

    Leb_Grid_XYZW[  872][0] = -0.816528856402219;
    Leb_Grid_XYZW[  872][1] = -0.563212302076210;
    Leb_Grid_XYZW[  872][2] =  0.126777480068428;
    Leb_Grid_XYZW[  872][3] =  0.000902134229904;

    Leb_Grid_XYZW[  873][0] = -0.816528856402219;
    Leb_Grid_XYZW[  873][1] = -0.563212302076210;
    Leb_Grid_XYZW[  873][2] = -0.126777480068428;
    Leb_Grid_XYZW[  873][3] =  0.000902134229904;

    Leb_Grid_XYZW[  874][0] =  0.816528856402219;
    Leb_Grid_XYZW[  874][1] =  0.126777480068429;
    Leb_Grid_XYZW[  874][2] =  0.563212302076210;
    Leb_Grid_XYZW[  874][3] =  0.000902134229904;

    Leb_Grid_XYZW[  875][0] =  0.816528856402219;
    Leb_Grid_XYZW[  875][1] =  0.126777480068429;
    Leb_Grid_XYZW[  875][2] = -0.563212302076210;
    Leb_Grid_XYZW[  875][3] =  0.000902134229904;

    Leb_Grid_XYZW[  876][0] =  0.816528856402219;
    Leb_Grid_XYZW[  876][1] = -0.126777480068429;
    Leb_Grid_XYZW[  876][2] =  0.563212302076210;
    Leb_Grid_XYZW[  876][3] =  0.000902134229904;

    Leb_Grid_XYZW[  877][0] =  0.816528856402219;
    Leb_Grid_XYZW[  877][1] = -0.126777480068429;
    Leb_Grid_XYZW[  877][2] = -0.563212302076210;
    Leb_Grid_XYZW[  877][3] =  0.000902134229904;

    Leb_Grid_XYZW[  878][0] = -0.816528856402219;
    Leb_Grid_XYZW[  878][1] =  0.126777480068428;
    Leb_Grid_XYZW[  878][2] =  0.563212302076210;
    Leb_Grid_XYZW[  878][3] =  0.000902134229904;

    Leb_Grid_XYZW[  879][0] = -0.816528856402219;
    Leb_Grid_XYZW[  879][1] =  0.126777480068428;
    Leb_Grid_XYZW[  879][2] = -0.563212302076210;
    Leb_Grid_XYZW[  879][3] =  0.000902134229904;

    Leb_Grid_XYZW[  880][0] = -0.816528856402219;
    Leb_Grid_XYZW[  880][1] = -0.126777480068428;
    Leb_Grid_XYZW[  880][2] =  0.563212302076210;
    Leb_Grid_XYZW[  880][3] =  0.000902134229904;

    Leb_Grid_XYZW[  881][0] = -0.816528856402219;
    Leb_Grid_XYZW[  881][1] = -0.126777480068428;
    Leb_Grid_XYZW[  881][2] = -0.563212302076210;
    Leb_Grid_XYZW[  881][3] =  0.000902134229904;

    Leb_Grid_XYZW[  882][0] =  0.563212302076210;
    Leb_Grid_XYZW[  882][1] =  0.816528856402219;
    Leb_Grid_XYZW[  882][2] =  0.126777480068428;
    Leb_Grid_XYZW[  882][3] =  0.000902134229904;

    Leb_Grid_XYZW[  883][0] =  0.563212302076210;
    Leb_Grid_XYZW[  883][1] =  0.816528856402219;
    Leb_Grid_XYZW[  883][2] = -0.126777480068428;
    Leb_Grid_XYZW[  883][3] =  0.000902134229904;

    Leb_Grid_XYZW[  884][0] =  0.563212302076210;
    Leb_Grid_XYZW[  884][1] = -0.816528856402219;
    Leb_Grid_XYZW[  884][2] =  0.126777480068428;
    Leb_Grid_XYZW[  884][3] =  0.000902134229904;

    Leb_Grid_XYZW[  885][0] =  0.563212302076210;
    Leb_Grid_XYZW[  885][1] = -0.816528856402219;
    Leb_Grid_XYZW[  885][2] = -0.126777480068428;
    Leb_Grid_XYZW[  885][3] =  0.000902134229904;

    Leb_Grid_XYZW[  886][0] = -0.563212302076210;
    Leb_Grid_XYZW[  886][1] =  0.816528856402219;
    Leb_Grid_XYZW[  886][2] =  0.126777480068428;
    Leb_Grid_XYZW[  886][3] =  0.000902134229904;

    Leb_Grid_XYZW[  887][0] = -0.563212302076210;
    Leb_Grid_XYZW[  887][1] =  0.816528856402219;
    Leb_Grid_XYZW[  887][2] = -0.126777480068428;
    Leb_Grid_XYZW[  887][3] =  0.000902134229904;

    Leb_Grid_XYZW[  888][0] = -0.563212302076210;
    Leb_Grid_XYZW[  888][1] = -0.816528856402219;
    Leb_Grid_XYZW[  888][2] =  0.126777480068428;
    Leb_Grid_XYZW[  888][3] =  0.000902134229904;

    Leb_Grid_XYZW[  889][0] = -0.563212302076210;
    Leb_Grid_XYZW[  889][1] = -0.816528856402219;
    Leb_Grid_XYZW[  889][2] = -0.126777480068428;
    Leb_Grid_XYZW[  889][3] =  0.000902134229904;

    Leb_Grid_XYZW[  890][0] =  0.563212302076210;
    Leb_Grid_XYZW[  890][1] =  0.126777480068428;
    Leb_Grid_XYZW[  890][2] =  0.816528856402219;
    Leb_Grid_XYZW[  890][3] =  0.000902134229904;

    Leb_Grid_XYZW[  891][0] =  0.563212302076210;
    Leb_Grid_XYZW[  891][1] =  0.126777480068428;
    Leb_Grid_XYZW[  891][2] = -0.816528856402219;
    Leb_Grid_XYZW[  891][3] =  0.000902134229904;

    Leb_Grid_XYZW[  892][0] =  0.563212302076210;
    Leb_Grid_XYZW[  892][1] = -0.126777480068428;
    Leb_Grid_XYZW[  892][2] =  0.816528856402219;
    Leb_Grid_XYZW[  892][3] =  0.000902134229904;

    Leb_Grid_XYZW[  893][0] =  0.563212302076210;
    Leb_Grid_XYZW[  893][1] = -0.126777480068428;
    Leb_Grid_XYZW[  893][2] = -0.816528856402219;
    Leb_Grid_XYZW[  893][3] =  0.000902134229904;

    Leb_Grid_XYZW[  894][0] = -0.563212302076210;
    Leb_Grid_XYZW[  894][1] =  0.126777480068428;
    Leb_Grid_XYZW[  894][2] =  0.816528856402219;
    Leb_Grid_XYZW[  894][3] =  0.000902134229904;

    Leb_Grid_XYZW[  895][0] = -0.563212302076210;
    Leb_Grid_XYZW[  895][1] =  0.126777480068428;
    Leb_Grid_XYZW[  895][2] = -0.816528856402219;
    Leb_Grid_XYZW[  895][3] =  0.000902134229904;

    Leb_Grid_XYZW[  896][0] = -0.563212302076210;
    Leb_Grid_XYZW[  896][1] = -0.126777480068428;
    Leb_Grid_XYZW[  896][2] =  0.816528856402219;
    Leb_Grid_XYZW[  896][3] =  0.000902134229904;

    Leb_Grid_XYZW[  897][0] = -0.563212302076210;
    Leb_Grid_XYZW[  897][1] = -0.126777480068428;
    Leb_Grid_XYZW[  897][2] = -0.816528856402219;
    Leb_Grid_XYZW[  897][3] =  0.000902134229904;

    Leb_Grid_XYZW[  898][0] =  0.126777480068428;
    Leb_Grid_XYZW[  898][1] =  0.816528856402219;
    Leb_Grid_XYZW[  898][2] =  0.563212302076210;
    Leb_Grid_XYZW[  898][3] =  0.000902134229904;

    Leb_Grid_XYZW[  899][0] =  0.126777480068428;
    Leb_Grid_XYZW[  899][1] =  0.816528856402219;
    Leb_Grid_XYZW[  899][2] = -0.563212302076210;
    Leb_Grid_XYZW[  899][3] =  0.000902134229904;

    Leb_Grid_XYZW[  900][0] =  0.126777480068428;
    Leb_Grid_XYZW[  900][1] = -0.816528856402219;
    Leb_Grid_XYZW[  900][2] =  0.563212302076210;
    Leb_Grid_XYZW[  900][3] =  0.000902134229904;

    Leb_Grid_XYZW[  901][0] =  0.126777480068428;
    Leb_Grid_XYZW[  901][1] = -0.816528856402219;
    Leb_Grid_XYZW[  901][2] = -0.563212302076210;
    Leb_Grid_XYZW[  901][3] =  0.000902134229904;

    Leb_Grid_XYZW[  902][0] = -0.126777480068428;
    Leb_Grid_XYZW[  902][1] =  0.816528856402219;
    Leb_Grid_XYZW[  902][2] =  0.563212302076210;
    Leb_Grid_XYZW[  902][3] =  0.000902134229904;

    Leb_Grid_XYZW[  903][0] = -0.126777480068428;
    Leb_Grid_XYZW[  903][1] =  0.816528856402219;
    Leb_Grid_XYZW[  903][2] = -0.563212302076210;
    Leb_Grid_XYZW[  903][3] =  0.000902134229904;

    Leb_Grid_XYZW[  904][0] = -0.126777480068428;
    Leb_Grid_XYZW[  904][1] = -0.816528856402219;
    Leb_Grid_XYZW[  904][2] =  0.563212302076210;
    Leb_Grid_XYZW[  904][3] =  0.000902134229904;

    Leb_Grid_XYZW[  905][0] = -0.126777480068428;
    Leb_Grid_XYZW[  905][1] = -0.816528856402219;
    Leb_Grid_XYZW[  905][2] = -0.563212302076210;
    Leb_Grid_XYZW[  905][3] =  0.000902134229904;

    Leb_Grid_XYZW[  906][0] =  0.126777480068428;
    Leb_Grid_XYZW[  906][1] =  0.563212302076210;
    Leb_Grid_XYZW[  906][2] =  0.816528856402219;
    Leb_Grid_XYZW[  906][3] =  0.000902134229904;

    Leb_Grid_XYZW[  907][0] =  0.126777480068428;
    Leb_Grid_XYZW[  907][1] =  0.563212302076210;
    Leb_Grid_XYZW[  907][2] = -0.816528856402219;
    Leb_Grid_XYZW[  907][3] =  0.000902134229904;

    Leb_Grid_XYZW[  908][0] =  0.126777480068428;
    Leb_Grid_XYZW[  908][1] = -0.563212302076210;
    Leb_Grid_XYZW[  908][2] =  0.816528856402219;
    Leb_Grid_XYZW[  908][3] =  0.000902134229904;

    Leb_Grid_XYZW[  909][0] =  0.126777480068428;
    Leb_Grid_XYZW[  909][1] = -0.563212302076210;
    Leb_Grid_XYZW[  909][2] = -0.816528856402219;
    Leb_Grid_XYZW[  909][3] =  0.000902134229904;

    Leb_Grid_XYZW[  910][0] = -0.126777480068428;
    Leb_Grid_XYZW[  910][1] =  0.563212302076210;
    Leb_Grid_XYZW[  910][2] =  0.816528856402219;
    Leb_Grid_XYZW[  910][3] =  0.000902134229904;

    Leb_Grid_XYZW[  911][0] = -0.126777480068428;
    Leb_Grid_XYZW[  911][1] =  0.563212302076210;
    Leb_Grid_XYZW[  911][2] = -0.816528856402219;
    Leb_Grid_XYZW[  911][3] =  0.000902134229904;

    Leb_Grid_XYZW[  912][0] = -0.126777480068428;
    Leb_Grid_XYZW[  912][1] = -0.563212302076210;
    Leb_Grid_XYZW[  912][2] =  0.816528856402219;
    Leb_Grid_XYZW[  912][3] =  0.000902134229904;

    Leb_Grid_XYZW[  913][0] = -0.126777480068428;
    Leb_Grid_XYZW[  913][1] = -0.563212302076210;
    Leb_Grid_XYZW[  913][2] = -0.816528856402219;
    Leb_Grid_XYZW[  913][3] =  0.000902134229904;

    Leb_Grid_XYZW[  914][0] =  0.801546937078353;
    Leb_Grid_XYZW[  914][1] =  0.543430356969390;
    Leb_Grid_XYZW[  914][2] =  0.249411216236224;
    Leb_Grid_XYZW[  914][3] =  0.000901009167711;

    Leb_Grid_XYZW[  915][0] =  0.801546937078353;
    Leb_Grid_XYZW[  915][1] =  0.543430356969390;
    Leb_Grid_XYZW[  915][2] = -0.249411216236224;
    Leb_Grid_XYZW[  915][3] =  0.000901009167711;

    Leb_Grid_XYZW[  916][0] =  0.801546937078353;
    Leb_Grid_XYZW[  916][1] = -0.543430356969390;
    Leb_Grid_XYZW[  916][2] =  0.249411216236224;
    Leb_Grid_XYZW[  916][3] =  0.000901009167711;

    Leb_Grid_XYZW[  917][0] =  0.801546937078353;
    Leb_Grid_XYZW[  917][1] = -0.543430356969390;
    Leb_Grid_XYZW[  917][2] = -0.249411216236224;
    Leb_Grid_XYZW[  917][3] =  0.000901009167711;

    Leb_Grid_XYZW[  918][0] = -0.801546937078353;
    Leb_Grid_XYZW[  918][1] =  0.543430356969390;
    Leb_Grid_XYZW[  918][2] =  0.249411216236224;
    Leb_Grid_XYZW[  918][3] =  0.000901009167711;

    Leb_Grid_XYZW[  919][0] = -0.801546937078353;
    Leb_Grid_XYZW[  919][1] =  0.543430356969390;
    Leb_Grid_XYZW[  919][2] = -0.249411216236224;
    Leb_Grid_XYZW[  919][3] =  0.000901009167711;

    Leb_Grid_XYZW[  920][0] = -0.801546937078353;
    Leb_Grid_XYZW[  920][1] = -0.543430356969390;
    Leb_Grid_XYZW[  920][2] =  0.249411216236224;
    Leb_Grid_XYZW[  920][3] =  0.000901009167711;

    Leb_Grid_XYZW[  921][0] = -0.801546937078353;
    Leb_Grid_XYZW[  921][1] = -0.543430356969390;
    Leb_Grid_XYZW[  921][2] = -0.249411216236224;
    Leb_Grid_XYZW[  921][3] =  0.000901009167711;

    Leb_Grid_XYZW[  922][0] =  0.801546937078353;
    Leb_Grid_XYZW[  922][1] =  0.249411216236224;
    Leb_Grid_XYZW[  922][2] =  0.543430356969390;
    Leb_Grid_XYZW[  922][3] =  0.000901009167711;

    Leb_Grid_XYZW[  923][0] =  0.801546937078353;
    Leb_Grid_XYZW[  923][1] =  0.249411216236224;
    Leb_Grid_XYZW[  923][2] = -0.543430356969390;
    Leb_Grid_XYZW[  923][3] =  0.000901009167711;

    Leb_Grid_XYZW[  924][0] =  0.801546937078353;
    Leb_Grid_XYZW[  924][1] = -0.249411216236224;
    Leb_Grid_XYZW[  924][2] =  0.543430356969390;
    Leb_Grid_XYZW[  924][3] =  0.000901009167711;

    Leb_Grid_XYZW[  925][0] =  0.801546937078353;
    Leb_Grid_XYZW[  925][1] = -0.249411216236224;
    Leb_Grid_XYZW[  925][2] = -0.543430356969390;
    Leb_Grid_XYZW[  925][3] =  0.000901009167711;

    Leb_Grid_XYZW[  926][0] = -0.801546937078353;
    Leb_Grid_XYZW[  926][1] =  0.249411216236224;
    Leb_Grid_XYZW[  926][2] =  0.543430356969390;
    Leb_Grid_XYZW[  926][3] =  0.000901009167711;

    Leb_Grid_XYZW[  927][0] = -0.801546937078353;
    Leb_Grid_XYZW[  927][1] =  0.249411216236224;
    Leb_Grid_XYZW[  927][2] = -0.543430356969390;
    Leb_Grid_XYZW[  927][3] =  0.000901009167711;

    Leb_Grid_XYZW[  928][0] = -0.801546937078353;
    Leb_Grid_XYZW[  928][1] = -0.249411216236224;
    Leb_Grid_XYZW[  928][2] =  0.543430356969390;
    Leb_Grid_XYZW[  928][3] =  0.000901009167711;

    Leb_Grid_XYZW[  929][0] = -0.801546937078353;
    Leb_Grid_XYZW[  929][1] = -0.249411216236224;
    Leb_Grid_XYZW[  929][2] = -0.543430356969390;
    Leb_Grid_XYZW[  929][3] =  0.000901009167711;

    Leb_Grid_XYZW[  930][0] =  0.543430356969390;
    Leb_Grid_XYZW[  930][1] =  0.801546937078353;
    Leb_Grid_XYZW[  930][2] =  0.249411216236224;
    Leb_Grid_XYZW[  930][3] =  0.000901009167711;

    Leb_Grid_XYZW[  931][0] =  0.543430356969390;
    Leb_Grid_XYZW[  931][1] =  0.801546937078353;
    Leb_Grid_XYZW[  931][2] = -0.249411216236224;
    Leb_Grid_XYZW[  931][3] =  0.000901009167711;

    Leb_Grid_XYZW[  932][0] =  0.543430356969390;
    Leb_Grid_XYZW[  932][1] = -0.801546937078353;
    Leb_Grid_XYZW[  932][2] =  0.249411216236224;
    Leb_Grid_XYZW[  932][3] =  0.000901009167711;

    Leb_Grid_XYZW[  933][0] =  0.543430356969390;
    Leb_Grid_XYZW[  933][1] = -0.801546937078353;
    Leb_Grid_XYZW[  933][2] = -0.249411216236224;
    Leb_Grid_XYZW[  933][3] =  0.000901009167711;

    Leb_Grid_XYZW[  934][0] = -0.543430356969390;
    Leb_Grid_XYZW[  934][1] =  0.801546937078353;
    Leb_Grid_XYZW[  934][2] =  0.249411216236224;
    Leb_Grid_XYZW[  934][3] =  0.000901009167711;

    Leb_Grid_XYZW[  935][0] = -0.543430356969390;
    Leb_Grid_XYZW[  935][1] =  0.801546937078353;
    Leb_Grid_XYZW[  935][2] = -0.249411216236224;
    Leb_Grid_XYZW[  935][3] =  0.000901009167711;

    Leb_Grid_XYZW[  936][0] = -0.543430356969390;
    Leb_Grid_XYZW[  936][1] = -0.801546937078353;
    Leb_Grid_XYZW[  936][2] =  0.249411216236224;
    Leb_Grid_XYZW[  936][3] =  0.000901009167711;

    Leb_Grid_XYZW[  937][0] = -0.543430356969390;
    Leb_Grid_XYZW[  937][1] = -0.801546937078353;
    Leb_Grid_XYZW[  937][2] = -0.249411216236224;
    Leb_Grid_XYZW[  937][3] =  0.000901009167711;

    Leb_Grid_XYZW[  938][0] =  0.543430356969390;
    Leb_Grid_XYZW[  938][1] =  0.249411216236224;
    Leb_Grid_XYZW[  938][2] =  0.801546937078353;
    Leb_Grid_XYZW[  938][3] =  0.000901009167711;

    Leb_Grid_XYZW[  939][0] =  0.543430356969390;
    Leb_Grid_XYZW[  939][1] =  0.249411216236224;
    Leb_Grid_XYZW[  939][2] = -0.801546937078353;
    Leb_Grid_XYZW[  939][3] =  0.000901009167711;

    Leb_Grid_XYZW[  940][0] =  0.543430356969390;
    Leb_Grid_XYZW[  940][1] = -0.249411216236224;
    Leb_Grid_XYZW[  940][2] =  0.801546937078353;
    Leb_Grid_XYZW[  940][3] =  0.000901009167711;

    Leb_Grid_XYZW[  941][0] =  0.543430356969390;
    Leb_Grid_XYZW[  941][1] = -0.249411216236224;
    Leb_Grid_XYZW[  941][2] = -0.801546937078353;
    Leb_Grid_XYZW[  941][3] =  0.000901009167711;

    Leb_Grid_XYZW[  942][0] = -0.543430356969390;
    Leb_Grid_XYZW[  942][1] =  0.249411216236224;
    Leb_Grid_XYZW[  942][2] =  0.801546937078353;
    Leb_Grid_XYZW[  942][3] =  0.000901009167711;

    Leb_Grid_XYZW[  943][0] = -0.543430356969390;
    Leb_Grid_XYZW[  943][1] =  0.249411216236224;
    Leb_Grid_XYZW[  943][2] = -0.801546937078353;
    Leb_Grid_XYZW[  943][3] =  0.000901009167711;

    Leb_Grid_XYZW[  944][0] = -0.543430356969390;
    Leb_Grid_XYZW[  944][1] = -0.249411216236224;
    Leb_Grid_XYZW[  944][2] =  0.801546937078353;
    Leb_Grid_XYZW[  944][3] =  0.000901009167711;

    Leb_Grid_XYZW[  945][0] = -0.543430356969390;
    Leb_Grid_XYZW[  945][1] = -0.249411216236224;
    Leb_Grid_XYZW[  945][2] = -0.801546937078353;
    Leb_Grid_XYZW[  945][3] =  0.000901009167711;

    Leb_Grid_XYZW[  946][0] =  0.249411216236224;
    Leb_Grid_XYZW[  946][1] =  0.801546937078353;
    Leb_Grid_XYZW[  946][2] =  0.543430356969390;
    Leb_Grid_XYZW[  946][3] =  0.000901009167711;

    Leb_Grid_XYZW[  947][0] =  0.249411216236224;
    Leb_Grid_XYZW[  947][1] =  0.801546937078353;
    Leb_Grid_XYZW[  947][2] = -0.543430356969390;
    Leb_Grid_XYZW[  947][3] =  0.000901009167711;

    Leb_Grid_XYZW[  948][0] =  0.249411216236224;
    Leb_Grid_XYZW[  948][1] = -0.801546937078353;
    Leb_Grid_XYZW[  948][2] =  0.543430356969390;
    Leb_Grid_XYZW[  948][3] =  0.000901009167711;

    Leb_Grid_XYZW[  949][0] =  0.249411216236224;
    Leb_Grid_XYZW[  949][1] = -0.801546937078353;
    Leb_Grid_XYZW[  949][2] = -0.543430356969390;
    Leb_Grid_XYZW[  949][3] =  0.000901009167711;

    Leb_Grid_XYZW[  950][0] = -0.249411216236224;
    Leb_Grid_XYZW[  950][1] =  0.801546937078353;
    Leb_Grid_XYZW[  950][2] =  0.543430356969390;
    Leb_Grid_XYZW[  950][3] =  0.000901009167711;

    Leb_Grid_XYZW[  951][0] = -0.249411216236224;
    Leb_Grid_XYZW[  951][1] =  0.801546937078353;
    Leb_Grid_XYZW[  951][2] = -0.543430356969390;
    Leb_Grid_XYZW[  951][3] =  0.000901009167711;

    Leb_Grid_XYZW[  952][0] = -0.249411216236224;
    Leb_Grid_XYZW[  952][1] = -0.801546937078353;
    Leb_Grid_XYZW[  952][2] =  0.543430356969390;
    Leb_Grid_XYZW[  952][3] =  0.000901009167711;

    Leb_Grid_XYZW[  953][0] = -0.249411216236224;
    Leb_Grid_XYZW[  953][1] = -0.801546937078353;
    Leb_Grid_XYZW[  953][2] = -0.543430356969390;
    Leb_Grid_XYZW[  953][3] =  0.000901009167711;

    Leb_Grid_XYZW[  954][0] =  0.249411216236224;
    Leb_Grid_XYZW[  954][1] =  0.543430356969390;
    Leb_Grid_XYZW[  954][2] =  0.801546937078353;
    Leb_Grid_XYZW[  954][3] =  0.000901009167711;

    Leb_Grid_XYZW[  955][0] =  0.249411216236224;
    Leb_Grid_XYZW[  955][1] =  0.543430356969390;
    Leb_Grid_XYZW[  955][2] = -0.801546937078353;
    Leb_Grid_XYZW[  955][3] =  0.000901009167711;

    Leb_Grid_XYZW[  956][0] =  0.249411216236224;
    Leb_Grid_XYZW[  956][1] = -0.543430356969390;
    Leb_Grid_XYZW[  956][2] =  0.801546937078353;
    Leb_Grid_XYZW[  956][3] =  0.000901009167711;

    Leb_Grid_XYZW[  957][0] =  0.249411216236224;
    Leb_Grid_XYZW[  957][1] = -0.543430356969390;
    Leb_Grid_XYZW[  957][2] = -0.801546937078353;
    Leb_Grid_XYZW[  957][3] =  0.000901009167711;

    Leb_Grid_XYZW[  958][0] = -0.249411216236224;
    Leb_Grid_XYZW[  958][1] =  0.543430356969390;
    Leb_Grid_XYZW[  958][2] =  0.801546937078353;
    Leb_Grid_XYZW[  958][3] =  0.000901009167711;

    Leb_Grid_XYZW[  959][0] = -0.249411216236224;
    Leb_Grid_XYZW[  959][1] =  0.543430356969390;
    Leb_Grid_XYZW[  959][2] = -0.801546937078353;
    Leb_Grid_XYZW[  959][3] =  0.000901009167711;

    Leb_Grid_XYZW[  960][0] = -0.249411216236224;
    Leb_Grid_XYZW[  960][1] = -0.543430356969390;
    Leb_Grid_XYZW[  960][2] =  0.801546937078353;
    Leb_Grid_XYZW[  960][3] =  0.000901009167711;

    Leb_Grid_XYZW[  961][0] = -0.249411216236224;
    Leb_Grid_XYZW[  961][1] = -0.543430356969390;
    Leb_Grid_XYZW[  961][2] = -0.801546937078353;
    Leb_Grid_XYZW[  961][3] =  0.000901009167711;

    Leb_Grid_XYZW[  962][0] =  0.777356306907035;
    Leb_Grid_XYZW[  962][1] =  0.512351848641987;
    Leb_Grid_XYZW[  962][2] =  0.364983226059765;
    Leb_Grid_XYZW[  962][3] =  0.000902269293843;

    Leb_Grid_XYZW[  963][0] =  0.777356306907035;
    Leb_Grid_XYZW[  963][1] =  0.512351848641987;
    Leb_Grid_XYZW[  963][2] = -0.364983226059766;
    Leb_Grid_XYZW[  963][3] =  0.000902269293843;

    Leb_Grid_XYZW[  964][0] =  0.777356306907035;
    Leb_Grid_XYZW[  964][1] = -0.512351848641987;
    Leb_Grid_XYZW[  964][2] =  0.364983226059765;
    Leb_Grid_XYZW[  964][3] =  0.000902269293843;

    Leb_Grid_XYZW[  965][0] =  0.777356306907035;
    Leb_Grid_XYZW[  965][1] = -0.512351848641987;
    Leb_Grid_XYZW[  965][2] = -0.364983226059766;
    Leb_Grid_XYZW[  965][3] =  0.000902269293843;

    Leb_Grid_XYZW[  966][0] = -0.777356306907035;
    Leb_Grid_XYZW[  966][1] =  0.512351848641987;
    Leb_Grid_XYZW[  966][2] =  0.364983226059765;
    Leb_Grid_XYZW[  966][3] =  0.000902269293843;

    Leb_Grid_XYZW[  967][0] = -0.777356306907035;
    Leb_Grid_XYZW[  967][1] =  0.512351848641987;
    Leb_Grid_XYZW[  967][2] = -0.364983226059766;
    Leb_Grid_XYZW[  967][3] =  0.000902269293843;

    Leb_Grid_XYZW[  968][0] = -0.777356306907035;
    Leb_Grid_XYZW[  968][1] = -0.512351848641987;
    Leb_Grid_XYZW[  968][2] =  0.364983226059765;
    Leb_Grid_XYZW[  968][3] =  0.000902269293843;

    Leb_Grid_XYZW[  969][0] = -0.777356306907035;
    Leb_Grid_XYZW[  969][1] = -0.512351848641987;
    Leb_Grid_XYZW[  969][2] = -0.364983226059766;
    Leb_Grid_XYZW[  969][3] =  0.000902269293843;

    Leb_Grid_XYZW[  970][0] =  0.777356306907035;
    Leb_Grid_XYZW[  970][1] =  0.364983226059765;
    Leb_Grid_XYZW[  970][2] =  0.512351848641987;
    Leb_Grid_XYZW[  970][3] =  0.000902269293843;

    Leb_Grid_XYZW[  971][0] =  0.777356306907035;
    Leb_Grid_XYZW[  971][1] =  0.364983226059765;
    Leb_Grid_XYZW[  971][2] = -0.512351848641987;
    Leb_Grid_XYZW[  971][3] =  0.000902269293843;

    Leb_Grid_XYZW[  972][0] =  0.777356306907035;
    Leb_Grid_XYZW[  972][1] = -0.364983226059765;
    Leb_Grid_XYZW[  972][2] =  0.512351848641987;
    Leb_Grid_XYZW[  972][3] =  0.000902269293843;

    Leb_Grid_XYZW[  973][0] =  0.777356306907035;
    Leb_Grid_XYZW[  973][1] = -0.364983226059765;
    Leb_Grid_XYZW[  973][2] = -0.512351848641987;
    Leb_Grid_XYZW[  973][3] =  0.000902269293843;

    Leb_Grid_XYZW[  974][0] = -0.777356306907035;
    Leb_Grid_XYZW[  974][1] =  0.364983226059765;
    Leb_Grid_XYZW[  974][2] =  0.512351848641987;
    Leb_Grid_XYZW[  974][3] =  0.000902269293843;

    Leb_Grid_XYZW[  975][0] = -0.777356306907035;
    Leb_Grid_XYZW[  975][1] =  0.364983226059765;
    Leb_Grid_XYZW[  975][2] = -0.512351848641987;
    Leb_Grid_XYZW[  975][3] =  0.000902269293843;

    Leb_Grid_XYZW[  976][0] = -0.777356306907035;
    Leb_Grid_XYZW[  976][1] = -0.364983226059765;
    Leb_Grid_XYZW[  976][2] =  0.512351848641987;
    Leb_Grid_XYZW[  976][3] =  0.000902269293843;

    Leb_Grid_XYZW[  977][0] = -0.777356306907035;
    Leb_Grid_XYZW[  977][1] = -0.364983226059765;
    Leb_Grid_XYZW[  977][2] = -0.512351848641987;
    Leb_Grid_XYZW[  977][3] =  0.000902269293843;

    Leb_Grid_XYZW[  978][0] =  0.512351848641987;
    Leb_Grid_XYZW[  978][1] =  0.777356306907035;
    Leb_Grid_XYZW[  978][2] =  0.364983226059765;
    Leb_Grid_XYZW[  978][3] =  0.000902269293843;

    Leb_Grid_XYZW[  979][0] =  0.512351848641987;
    Leb_Grid_XYZW[  979][1] =  0.777356306907035;
    Leb_Grid_XYZW[  979][2] = -0.364983226059766;
    Leb_Grid_XYZW[  979][3] =  0.000902269293843;

    Leb_Grid_XYZW[  980][0] =  0.512351848641987;
    Leb_Grid_XYZW[  980][1] = -0.777356306907035;
    Leb_Grid_XYZW[  980][2] =  0.364983226059765;
    Leb_Grid_XYZW[  980][3] =  0.000902269293843;

    Leb_Grid_XYZW[  981][0] =  0.512351848641987;
    Leb_Grid_XYZW[  981][1] = -0.777356306907035;
    Leb_Grid_XYZW[  981][2] = -0.364983226059766;
    Leb_Grid_XYZW[  981][3] =  0.000902269293843;

    Leb_Grid_XYZW[  982][0] = -0.512351848641987;
    Leb_Grid_XYZW[  982][1] =  0.777356306907035;
    Leb_Grid_XYZW[  982][2] =  0.364983226059765;
    Leb_Grid_XYZW[  982][3] =  0.000902269293843;

    Leb_Grid_XYZW[  983][0] = -0.512351848641987;
    Leb_Grid_XYZW[  983][1] =  0.777356306907035;
    Leb_Grid_XYZW[  983][2] = -0.364983226059766;
    Leb_Grid_XYZW[  983][3] =  0.000902269293843;

    Leb_Grid_XYZW[  984][0] = -0.512351848641987;
    Leb_Grid_XYZW[  984][1] = -0.777356306907035;
    Leb_Grid_XYZW[  984][2] =  0.364983226059765;
    Leb_Grid_XYZW[  984][3] =  0.000902269293843;

    Leb_Grid_XYZW[  985][0] = -0.512351848641987;
    Leb_Grid_XYZW[  985][1] = -0.777356306907035;
    Leb_Grid_XYZW[  985][2] = -0.364983226059766;
    Leb_Grid_XYZW[  985][3] =  0.000902269293843;

    Leb_Grid_XYZW[  986][0] =  0.512351848641987;
    Leb_Grid_XYZW[  986][1] =  0.364983226059765;
    Leb_Grid_XYZW[  986][2] =  0.777356306907035;
    Leb_Grid_XYZW[  986][3] =  0.000902269293843;

    Leb_Grid_XYZW[  987][0] =  0.512351848641987;
    Leb_Grid_XYZW[  987][1] =  0.364983226059765;
    Leb_Grid_XYZW[  987][2] = -0.777356306907035;
    Leb_Grid_XYZW[  987][3] =  0.000902269293843;

    Leb_Grid_XYZW[  988][0] =  0.512351848641987;
    Leb_Grid_XYZW[  988][1] = -0.364983226059765;
    Leb_Grid_XYZW[  988][2] =  0.777356306907035;
    Leb_Grid_XYZW[  988][3] =  0.000902269293843;

    Leb_Grid_XYZW[  989][0] =  0.512351848641987;
    Leb_Grid_XYZW[  989][1] = -0.364983226059765;
    Leb_Grid_XYZW[  989][2] = -0.777356306907035;
    Leb_Grid_XYZW[  989][3] =  0.000902269293843;

    Leb_Grid_XYZW[  990][0] = -0.512351848641987;
    Leb_Grid_XYZW[  990][1] =  0.364983226059766;
    Leb_Grid_XYZW[  990][2] =  0.777356306907035;
    Leb_Grid_XYZW[  990][3] =  0.000902269293843;

    Leb_Grid_XYZW[  991][0] = -0.512351848641987;
    Leb_Grid_XYZW[  991][1] =  0.364983226059765;
    Leb_Grid_XYZW[  991][2] = -0.777356306907035;
    Leb_Grid_XYZW[  991][3] =  0.000902269293843;

    Leb_Grid_XYZW[  992][0] = -0.512351848641987;
    Leb_Grid_XYZW[  992][1] = -0.364983226059766;
    Leb_Grid_XYZW[  992][2] =  0.777356306907035;
    Leb_Grid_XYZW[  992][3] =  0.000902269293843;

    Leb_Grid_XYZW[  993][0] = -0.512351848641987;
    Leb_Grid_XYZW[  993][1] = -0.364983226059765;
    Leb_Grid_XYZW[  993][2] = -0.777356306907035;
    Leb_Grid_XYZW[  993][3] =  0.000902269293843;

    Leb_Grid_XYZW[  994][0] =  0.364983226059765;
    Leb_Grid_XYZW[  994][1] =  0.777356306907035;
    Leb_Grid_XYZW[  994][2] =  0.512351848641987;
    Leb_Grid_XYZW[  994][3] =  0.000902269293843;

    Leb_Grid_XYZW[  995][0] =  0.364983226059765;
    Leb_Grid_XYZW[  995][1] =  0.777356306907035;
    Leb_Grid_XYZW[  995][2] = -0.512351848641987;
    Leb_Grid_XYZW[  995][3] =  0.000902269293843;

    Leb_Grid_XYZW[  996][0] =  0.364983226059765;
    Leb_Grid_XYZW[  996][1] = -0.777356306907035;
    Leb_Grid_XYZW[  996][2] =  0.512351848641987;
    Leb_Grid_XYZW[  996][3] =  0.000902269293843;

    Leb_Grid_XYZW[  997][0] =  0.364983226059765;
    Leb_Grid_XYZW[  997][1] = -0.777356306907035;
    Leb_Grid_XYZW[  997][2] = -0.512351848641987;
    Leb_Grid_XYZW[  997][3] =  0.000902269293843;

    Leb_Grid_XYZW[  998][0] = -0.364983226059765;
    Leb_Grid_XYZW[  998][1] =  0.777356306907035;
    Leb_Grid_XYZW[  998][2] =  0.512351848641987;
    Leb_Grid_XYZW[  998][3] =  0.000902269293843;

    Leb_Grid_XYZW[  999][0] = -0.364983226059765;
    Leb_Grid_XYZW[  999][1] =  0.777356306907035;
    Leb_Grid_XYZW[  999][2] = -0.512351848641987;
    Leb_Grid_XYZW[  999][3] =  0.000902269293843;

    Leb_Grid_XYZW[ 1000][0] = -0.364983226059765;
    Leb_Grid_XYZW[ 1000][1] = -0.777356306907035;
    Leb_Grid_XYZW[ 1000][2] =  0.512351848641987;
    Leb_Grid_XYZW[ 1000][3] =  0.000902269293843;

    Leb_Grid_XYZW[ 1001][0] = -0.364983226059765;
    Leb_Grid_XYZW[ 1001][1] = -0.777356306907035;
    Leb_Grid_XYZW[ 1001][2] = -0.512351848641987;
    Leb_Grid_XYZW[ 1001][3] =  0.000902269293843;

    Leb_Grid_XYZW[ 1002][0] =  0.364983226059765;
    Leb_Grid_XYZW[ 1002][1] =  0.512351848641987;
    Leb_Grid_XYZW[ 1002][2] =  0.777356306907035;
    Leb_Grid_XYZW[ 1002][3] =  0.000902269293843;

    Leb_Grid_XYZW[ 1003][0] =  0.364983226059765;
    Leb_Grid_XYZW[ 1003][1] =  0.512351848641987;
    Leb_Grid_XYZW[ 1003][2] = -0.777356306907035;
    Leb_Grid_XYZW[ 1003][3] =  0.000902269293843;

    Leb_Grid_XYZW[ 1004][0] =  0.364983226059765;
    Leb_Grid_XYZW[ 1004][1] = -0.512351848641987;
    Leb_Grid_XYZW[ 1004][2] =  0.777356306907035;
    Leb_Grid_XYZW[ 1004][3] =  0.000902269293843;

    Leb_Grid_XYZW[ 1005][0] =  0.364983226059765;
    Leb_Grid_XYZW[ 1005][1] = -0.512351848641987;
    Leb_Grid_XYZW[ 1005][2] = -0.777356306907035;
    Leb_Grid_XYZW[ 1005][3] =  0.000902269293843;

    Leb_Grid_XYZW[ 1006][0] = -0.364983226059765;
    Leb_Grid_XYZW[ 1006][1] =  0.512351848641987;
    Leb_Grid_XYZW[ 1006][2] =  0.777356306907035;
    Leb_Grid_XYZW[ 1006][3] =  0.000902269293843;

    Leb_Grid_XYZW[ 1007][0] = -0.364983226059765;
    Leb_Grid_XYZW[ 1007][1] =  0.512351848641987;
    Leb_Grid_XYZW[ 1007][2] = -0.777356306907035;
    Leb_Grid_XYZW[ 1007][3] =  0.000902269293843;

    Leb_Grid_XYZW[ 1008][0] = -0.364983226059765;
    Leb_Grid_XYZW[ 1008][1] = -0.512351848641987;
    Leb_Grid_XYZW[ 1008][2] =  0.777356306907035;
    Leb_Grid_XYZW[ 1008][3] =  0.000902269293843;

    Leb_Grid_XYZW[ 1009][0] = -0.364983226059765;
    Leb_Grid_XYZW[ 1009][1] = -0.512351848641987;
    Leb_Grid_XYZW[ 1009][2] = -0.777356306907035;
    Leb_Grid_XYZW[ 1009][3] =  0.000902269293843;

    Leb_Grid_XYZW[ 1010][0] =  0.766162121390039;
    Leb_Grid_XYZW[ 1010][1] =  0.639427963474910;
    Leb_Grid_XYZW[ 1010][2] =  0.064245492242208;
    Leb_Grid_XYZW[ 1010][3] =  0.000915801617469;

    Leb_Grid_XYZW[ 1011][0] =  0.766162121390039;
    Leb_Grid_XYZW[ 1011][1] =  0.639427963474910;
    Leb_Grid_XYZW[ 1011][2] = -0.064245492242207;
    Leb_Grid_XYZW[ 1011][3] =  0.000915801617469;

    Leb_Grid_XYZW[ 1012][0] =  0.766162121390039;
    Leb_Grid_XYZW[ 1012][1] = -0.639427963474910;
    Leb_Grid_XYZW[ 1012][2] =  0.064245492242208;
    Leb_Grid_XYZW[ 1012][3] =  0.000915801617469;

    Leb_Grid_XYZW[ 1013][0] =  0.766162121390039;
    Leb_Grid_XYZW[ 1013][1] = -0.639427963474910;
    Leb_Grid_XYZW[ 1013][2] = -0.064245492242207;
    Leb_Grid_XYZW[ 1013][3] =  0.000915801617469;

    Leb_Grid_XYZW[ 1014][0] = -0.766162121390039;
    Leb_Grid_XYZW[ 1014][1] =  0.639427963474910;
    Leb_Grid_XYZW[ 1014][2] =  0.064245492242208;
    Leb_Grid_XYZW[ 1014][3] =  0.000915801617469;

    Leb_Grid_XYZW[ 1015][0] = -0.766162121390039;
    Leb_Grid_XYZW[ 1015][1] =  0.639427963474910;
    Leb_Grid_XYZW[ 1015][2] = -0.064245492242207;
    Leb_Grid_XYZW[ 1015][3] =  0.000915801617469;

    Leb_Grid_XYZW[ 1016][0] = -0.766162121390039;
    Leb_Grid_XYZW[ 1016][1] = -0.639427963474910;
    Leb_Grid_XYZW[ 1016][2] =  0.064245492242208;
    Leb_Grid_XYZW[ 1016][3] =  0.000915801617469;

    Leb_Grid_XYZW[ 1017][0] = -0.766162121390039;
    Leb_Grid_XYZW[ 1017][1] = -0.639427963474910;
    Leb_Grid_XYZW[ 1017][2] = -0.064245492242207;
    Leb_Grid_XYZW[ 1017][3] =  0.000915801617469;

    Leb_Grid_XYZW[ 1018][0] =  0.766162121390039;
    Leb_Grid_XYZW[ 1018][1] =  0.064245492242208;
    Leb_Grid_XYZW[ 1018][2] =  0.639427963474910;
    Leb_Grid_XYZW[ 1018][3] =  0.000915801617469;

    Leb_Grid_XYZW[ 1019][0] =  0.766162121390039;
    Leb_Grid_XYZW[ 1019][1] =  0.064245492242208;
    Leb_Grid_XYZW[ 1019][2] = -0.639427963474910;
    Leb_Grid_XYZW[ 1019][3] =  0.000915801617469;

    Leb_Grid_XYZW[ 1020][0] =  0.766162121390039;
    Leb_Grid_XYZW[ 1020][1] = -0.064245492242208;
    Leb_Grid_XYZW[ 1020][2] =  0.639427963474910;
    Leb_Grid_XYZW[ 1020][3] =  0.000915801617469;

    Leb_Grid_XYZW[ 1021][0] =  0.766162121390039;
    Leb_Grid_XYZW[ 1021][1] = -0.064245492242208;
    Leb_Grid_XYZW[ 1021][2] = -0.639427963474910;
    Leb_Grid_XYZW[ 1021][3] =  0.000915801617469;

    Leb_Grid_XYZW[ 1022][0] = -0.766162121390039;
    Leb_Grid_XYZW[ 1022][1] =  0.064245492242208;
    Leb_Grid_XYZW[ 1022][2] =  0.639427963474910;
    Leb_Grid_XYZW[ 1022][3] =  0.000915801617469;

    Leb_Grid_XYZW[ 1023][0] = -0.766162121390039;
    Leb_Grid_XYZW[ 1023][1] =  0.064245492242208;
    Leb_Grid_XYZW[ 1023][2] = -0.639427963474910;
    Leb_Grid_XYZW[ 1023][3] =  0.000915801617469;

    Leb_Grid_XYZW[ 1024][0] = -0.766162121390039;
    Leb_Grid_XYZW[ 1024][1] = -0.064245492242208;
    Leb_Grid_XYZW[ 1024][2] =  0.639427963474910;
    Leb_Grid_XYZW[ 1024][3] =  0.000915801617469;

    Leb_Grid_XYZW[ 1025][0] = -0.766162121390039;
    Leb_Grid_XYZW[ 1025][1] = -0.064245492242208;
    Leb_Grid_XYZW[ 1025][2] = -0.639427963474910;
    Leb_Grid_XYZW[ 1025][3] =  0.000915801617469;

    Leb_Grid_XYZW[ 1026][0] =  0.639427963474910;
    Leb_Grid_XYZW[ 1026][1] =  0.766162121390039;
    Leb_Grid_XYZW[ 1026][2] =  0.064245492242208;
    Leb_Grid_XYZW[ 1026][3] =  0.000915801617469;

    Leb_Grid_XYZW[ 1027][0] =  0.639427963474910;
    Leb_Grid_XYZW[ 1027][1] =  0.766162121390039;
    Leb_Grid_XYZW[ 1027][2] = -0.064245492242207;
    Leb_Grid_XYZW[ 1027][3] =  0.000915801617469;

    Leb_Grid_XYZW[ 1028][0] =  0.639427963474910;
    Leb_Grid_XYZW[ 1028][1] = -0.766162121390039;
    Leb_Grid_XYZW[ 1028][2] =  0.064245492242208;
    Leb_Grid_XYZW[ 1028][3] =  0.000915801617469;

    Leb_Grid_XYZW[ 1029][0] =  0.639427963474910;
    Leb_Grid_XYZW[ 1029][1] = -0.766162121390039;
    Leb_Grid_XYZW[ 1029][2] = -0.064245492242207;
    Leb_Grid_XYZW[ 1029][3] =  0.000915801617469;

    Leb_Grid_XYZW[ 1030][0] = -0.639427963474910;
    Leb_Grid_XYZW[ 1030][1] =  0.766162121390040;
    Leb_Grid_XYZW[ 1030][2] =  0.064245492242208;
    Leb_Grid_XYZW[ 1030][3] =  0.000915801617469;

    Leb_Grid_XYZW[ 1031][0] = -0.639427963474910;
    Leb_Grid_XYZW[ 1031][1] =  0.766162121390040;
    Leb_Grid_XYZW[ 1031][2] = -0.064245492242207;
    Leb_Grid_XYZW[ 1031][3] =  0.000915801617469;

    Leb_Grid_XYZW[ 1032][0] = -0.639427963474910;
    Leb_Grid_XYZW[ 1032][1] = -0.766162121390040;
    Leb_Grid_XYZW[ 1032][2] =  0.064245492242208;
    Leb_Grid_XYZW[ 1032][3] =  0.000915801617469;

    Leb_Grid_XYZW[ 1033][0] = -0.639427963474910;
    Leb_Grid_XYZW[ 1033][1] = -0.766162121390040;
    Leb_Grid_XYZW[ 1033][2] = -0.064245492242207;
    Leb_Grid_XYZW[ 1033][3] =  0.000915801617469;

    Leb_Grid_XYZW[ 1034][0] =  0.639427963474910;
    Leb_Grid_XYZW[ 1034][1] =  0.064245492242207;
    Leb_Grid_XYZW[ 1034][2] =  0.766162121390039;
    Leb_Grid_XYZW[ 1034][3] =  0.000915801617469;

    Leb_Grid_XYZW[ 1035][0] =  0.639427963474910;
    Leb_Grid_XYZW[ 1035][1] =  0.064245492242207;
    Leb_Grid_XYZW[ 1035][2] = -0.766162121390039;
    Leb_Grid_XYZW[ 1035][3] =  0.000915801617469;

    Leb_Grid_XYZW[ 1036][0] =  0.639427963474910;
    Leb_Grid_XYZW[ 1036][1] = -0.064245492242207;
    Leb_Grid_XYZW[ 1036][2] =  0.766162121390039;
    Leb_Grid_XYZW[ 1036][3] =  0.000915801617469;

    Leb_Grid_XYZW[ 1037][0] =  0.639427963474910;
    Leb_Grid_XYZW[ 1037][1] = -0.064245492242207;
    Leb_Grid_XYZW[ 1037][2] = -0.766162121390039;
    Leb_Grid_XYZW[ 1037][3] =  0.000915801617469;

    Leb_Grid_XYZW[ 1038][0] = -0.639427963474910;
    Leb_Grid_XYZW[ 1038][1] =  0.064245492242207;
    Leb_Grid_XYZW[ 1038][2] =  0.766162121390039;
    Leb_Grid_XYZW[ 1038][3] =  0.000915801617469;

    Leb_Grid_XYZW[ 1039][0] = -0.639427963474910;
    Leb_Grid_XYZW[ 1039][1] =  0.064245492242207;
    Leb_Grid_XYZW[ 1039][2] = -0.766162121390039;
    Leb_Grid_XYZW[ 1039][3] =  0.000915801617469;

    Leb_Grid_XYZW[ 1040][0] = -0.639427963474910;
    Leb_Grid_XYZW[ 1040][1] = -0.064245492242207;
    Leb_Grid_XYZW[ 1040][2] =  0.766162121390039;
    Leb_Grid_XYZW[ 1040][3] =  0.000915801617469;

    Leb_Grid_XYZW[ 1041][0] = -0.639427963474910;
    Leb_Grid_XYZW[ 1041][1] = -0.064245492242207;
    Leb_Grid_XYZW[ 1041][2] = -0.766162121390039;
    Leb_Grid_XYZW[ 1041][3] =  0.000915801617469;

    Leb_Grid_XYZW[ 1042][0] =  0.064245492242207;
    Leb_Grid_XYZW[ 1042][1] =  0.766162121390039;
    Leb_Grid_XYZW[ 1042][2] =  0.639427963474910;
    Leb_Grid_XYZW[ 1042][3] =  0.000915801617469;

    Leb_Grid_XYZW[ 1043][0] =  0.064245492242207;
    Leb_Grid_XYZW[ 1043][1] =  0.766162121390040;
    Leb_Grid_XYZW[ 1043][2] = -0.639427963474910;
    Leb_Grid_XYZW[ 1043][3] =  0.000915801617469;

    Leb_Grid_XYZW[ 1044][0] =  0.064245492242207;
    Leb_Grid_XYZW[ 1044][1] = -0.766162121390039;
    Leb_Grid_XYZW[ 1044][2] =  0.639427963474910;
    Leb_Grid_XYZW[ 1044][3] =  0.000915801617469;

    Leb_Grid_XYZW[ 1045][0] =  0.064245492242207;
    Leb_Grid_XYZW[ 1045][1] = -0.766162121390040;
    Leb_Grid_XYZW[ 1045][2] = -0.639427963474910;
    Leb_Grid_XYZW[ 1045][3] =  0.000915801617469;

    Leb_Grid_XYZW[ 1046][0] = -0.064245492242208;
    Leb_Grid_XYZW[ 1046][1] =  0.766162121390039;
    Leb_Grid_XYZW[ 1046][2] =  0.639427963474910;
    Leb_Grid_XYZW[ 1046][3] =  0.000915801617469;

    Leb_Grid_XYZW[ 1047][0] = -0.064245492242208;
    Leb_Grid_XYZW[ 1047][1] =  0.766162121390040;
    Leb_Grid_XYZW[ 1047][2] = -0.639427963474910;
    Leb_Grid_XYZW[ 1047][3] =  0.000915801617469;

    Leb_Grid_XYZW[ 1048][0] = -0.064245492242208;
    Leb_Grid_XYZW[ 1048][1] = -0.766162121390039;
    Leb_Grid_XYZW[ 1048][2] =  0.639427963474910;
    Leb_Grid_XYZW[ 1048][3] =  0.000915801617469;

    Leb_Grid_XYZW[ 1049][0] = -0.064245492242208;
    Leb_Grid_XYZW[ 1049][1] = -0.766162121390040;
    Leb_Grid_XYZW[ 1049][2] = -0.639427963474910;
    Leb_Grid_XYZW[ 1049][3] =  0.000915801617469;

    Leb_Grid_XYZW[ 1050][0] =  0.064245492242208;
    Leb_Grid_XYZW[ 1050][1] =  0.639427963474910;
    Leb_Grid_XYZW[ 1050][2] =  0.766162121390039;
    Leb_Grid_XYZW[ 1050][3] =  0.000915801617469;

    Leb_Grid_XYZW[ 1051][0] =  0.064245492242208;
    Leb_Grid_XYZW[ 1051][1] =  0.639427963474910;
    Leb_Grid_XYZW[ 1051][2] = -0.766162121390039;
    Leb_Grid_XYZW[ 1051][3] =  0.000915801617469;

    Leb_Grid_XYZW[ 1052][0] =  0.064245492242208;
    Leb_Grid_XYZW[ 1052][1] = -0.639427963474910;
    Leb_Grid_XYZW[ 1052][2] =  0.766162121390039;
    Leb_Grid_XYZW[ 1052][3] =  0.000915801617469;

    Leb_Grid_XYZW[ 1053][0] =  0.064245492242208;
    Leb_Grid_XYZW[ 1053][1] = -0.639427963474910;
    Leb_Grid_XYZW[ 1053][2] = -0.766162121390039;
    Leb_Grid_XYZW[ 1053][3] =  0.000915801617469;

    Leb_Grid_XYZW[ 1054][0] = -0.064245492242208;
    Leb_Grid_XYZW[ 1054][1] =  0.639427963474910;
    Leb_Grid_XYZW[ 1054][2] =  0.766162121390039;
    Leb_Grid_XYZW[ 1054][3] =  0.000915801617469;

    Leb_Grid_XYZW[ 1055][0] = -0.064245492242208;
    Leb_Grid_XYZW[ 1055][1] =  0.639427963474910;
    Leb_Grid_XYZW[ 1055][2] = -0.766162121390039;
    Leb_Grid_XYZW[ 1055][3] =  0.000915801617469;

    Leb_Grid_XYZW[ 1056][0] = -0.064245492242208;
    Leb_Grid_XYZW[ 1056][1] = -0.639427963474910;
    Leb_Grid_XYZW[ 1056][2] =  0.766162121390039;
    Leb_Grid_XYZW[ 1056][3] =  0.000915801617469;

    Leb_Grid_XYZW[ 1057][0] = -0.064245492242208;
    Leb_Grid_XYZW[ 1057][1] = -0.639427963474910;
    Leb_Grid_XYZW[ 1057][2] = -0.766162121390039;
    Leb_Grid_XYZW[ 1057][3] =  0.000915801617469;

    Leb_Grid_XYZW[ 1058][0] =  0.755358414353351;
    Leb_Grid_XYZW[ 1058][1] =  0.626980550902439;
    Leb_Grid_XYZW[ 1058][2] =  0.190601822277923;
    Leb_Grid_XYZW[ 1058][3] =  0.000913157800319;

    Leb_Grid_XYZW[ 1059][0] =  0.755358414353351;
    Leb_Grid_XYZW[ 1059][1] =  0.626980550902439;
    Leb_Grid_XYZW[ 1059][2] = -0.190601822277924;
    Leb_Grid_XYZW[ 1059][3] =  0.000913157800319;

    Leb_Grid_XYZW[ 1060][0] =  0.755358414353351;
    Leb_Grid_XYZW[ 1060][1] = -0.626980550902439;
    Leb_Grid_XYZW[ 1060][2] =  0.190601822277923;
    Leb_Grid_XYZW[ 1060][3] =  0.000913157800319;

    Leb_Grid_XYZW[ 1061][0] =  0.755358414353351;
    Leb_Grid_XYZW[ 1061][1] = -0.626980550902439;
    Leb_Grid_XYZW[ 1061][2] = -0.190601822277924;
    Leb_Grid_XYZW[ 1061][3] =  0.000913157800319;

    Leb_Grid_XYZW[ 1062][0] = -0.755358414353351;
    Leb_Grid_XYZW[ 1062][1] =  0.626980550902439;
    Leb_Grid_XYZW[ 1062][2] =  0.190601822277923;
    Leb_Grid_XYZW[ 1062][3] =  0.000913157800319;

    Leb_Grid_XYZW[ 1063][0] = -0.755358414353351;
    Leb_Grid_XYZW[ 1063][1] =  0.626980550902439;
    Leb_Grid_XYZW[ 1063][2] = -0.190601822277924;
    Leb_Grid_XYZW[ 1063][3] =  0.000913157800319;

    Leb_Grid_XYZW[ 1064][0] = -0.755358414353351;
    Leb_Grid_XYZW[ 1064][1] = -0.626980550902439;
    Leb_Grid_XYZW[ 1064][2] =  0.190601822277923;
    Leb_Grid_XYZW[ 1064][3] =  0.000913157800319;

    Leb_Grid_XYZW[ 1065][0] = -0.755358414353351;
    Leb_Grid_XYZW[ 1065][1] = -0.626980550902439;
    Leb_Grid_XYZW[ 1065][2] = -0.190601822277924;
    Leb_Grid_XYZW[ 1065][3] =  0.000913157800319;

    Leb_Grid_XYZW[ 1066][0] =  0.755358414353351;
    Leb_Grid_XYZW[ 1066][1] =  0.190601822277924;
    Leb_Grid_XYZW[ 1066][2] =  0.626980550902439;
    Leb_Grid_XYZW[ 1066][3] =  0.000913157800319;

    Leb_Grid_XYZW[ 1067][0] =  0.755358414353351;
    Leb_Grid_XYZW[ 1067][1] =  0.190601822277924;
    Leb_Grid_XYZW[ 1067][2] = -0.626980550902439;
    Leb_Grid_XYZW[ 1067][3] =  0.000913157800319;

    Leb_Grid_XYZW[ 1068][0] =  0.755358414353351;
    Leb_Grid_XYZW[ 1068][1] = -0.190601822277924;
    Leb_Grid_XYZW[ 1068][2] =  0.626980550902439;
    Leb_Grid_XYZW[ 1068][3] =  0.000913157800319;

    Leb_Grid_XYZW[ 1069][0] =  0.755358414353351;
    Leb_Grid_XYZW[ 1069][1] = -0.190601822277924;
    Leb_Grid_XYZW[ 1069][2] = -0.626980550902439;
    Leb_Grid_XYZW[ 1069][3] =  0.000913157800319;

    Leb_Grid_XYZW[ 1070][0] = -0.755358414353351;
    Leb_Grid_XYZW[ 1070][1] =  0.190601822277923;
    Leb_Grid_XYZW[ 1070][2] =  0.626980550902439;
    Leb_Grid_XYZW[ 1070][3] =  0.000913157800319;

    Leb_Grid_XYZW[ 1071][0] = -0.755358414353351;
    Leb_Grid_XYZW[ 1071][1] =  0.190601822277923;
    Leb_Grid_XYZW[ 1071][2] = -0.626980550902439;
    Leb_Grid_XYZW[ 1071][3] =  0.000913157800319;

    Leb_Grid_XYZW[ 1072][0] = -0.755358414353351;
    Leb_Grid_XYZW[ 1072][1] = -0.190601822277923;
    Leb_Grid_XYZW[ 1072][2] =  0.626980550902439;
    Leb_Grid_XYZW[ 1072][3] =  0.000913157800319;

    Leb_Grid_XYZW[ 1073][0] = -0.755358414353351;
    Leb_Grid_XYZW[ 1073][1] = -0.190601822277923;
    Leb_Grid_XYZW[ 1073][2] = -0.626980550902439;
    Leb_Grid_XYZW[ 1073][3] =  0.000913157800319;

    Leb_Grid_XYZW[ 1074][0] =  0.626980550902439;
    Leb_Grid_XYZW[ 1074][1] =  0.755358414353351;
    Leb_Grid_XYZW[ 1074][2] =  0.190601822277923;
    Leb_Grid_XYZW[ 1074][3] =  0.000913157800319;

    Leb_Grid_XYZW[ 1075][0] =  0.626980550902439;
    Leb_Grid_XYZW[ 1075][1] =  0.755358414353351;
    Leb_Grid_XYZW[ 1075][2] = -0.190601822277924;
    Leb_Grid_XYZW[ 1075][3] =  0.000913157800319;

    Leb_Grid_XYZW[ 1076][0] =  0.626980550902439;
    Leb_Grid_XYZW[ 1076][1] = -0.755358414353351;
    Leb_Grid_XYZW[ 1076][2] =  0.190601822277923;
    Leb_Grid_XYZW[ 1076][3] =  0.000913157800319;

    Leb_Grid_XYZW[ 1077][0] =  0.626980550902439;
    Leb_Grid_XYZW[ 1077][1] = -0.755358414353351;
    Leb_Grid_XYZW[ 1077][2] = -0.190601822277924;
    Leb_Grid_XYZW[ 1077][3] =  0.000913157800319;

    Leb_Grid_XYZW[ 1078][0] = -0.626980550902439;
    Leb_Grid_XYZW[ 1078][1] =  0.755358414353351;
    Leb_Grid_XYZW[ 1078][2] =  0.190601822277923;
    Leb_Grid_XYZW[ 1078][3] =  0.000913157800319;

    Leb_Grid_XYZW[ 1079][0] = -0.626980550902439;
    Leb_Grid_XYZW[ 1079][1] =  0.755358414353351;
    Leb_Grid_XYZW[ 1079][2] = -0.190601822277924;
    Leb_Grid_XYZW[ 1079][3] =  0.000913157800319;

    Leb_Grid_XYZW[ 1080][0] = -0.626980550902439;
    Leb_Grid_XYZW[ 1080][1] = -0.755358414353351;
    Leb_Grid_XYZW[ 1080][2] =  0.190601822277923;
    Leb_Grid_XYZW[ 1080][3] =  0.000913157800319;

    Leb_Grid_XYZW[ 1081][0] = -0.626980550902439;
    Leb_Grid_XYZW[ 1081][1] = -0.755358414353351;
    Leb_Grid_XYZW[ 1081][2] = -0.190601822277924;
    Leb_Grid_XYZW[ 1081][3] =  0.000913157800319;

    Leb_Grid_XYZW[ 1082][0] =  0.626980550902439;
    Leb_Grid_XYZW[ 1082][1] =  0.190601822277923;
    Leb_Grid_XYZW[ 1082][2] =  0.755358414353351;
    Leb_Grid_XYZW[ 1082][3] =  0.000913157800319;

    Leb_Grid_XYZW[ 1083][0] =  0.626980550902439;
    Leb_Grid_XYZW[ 1083][1] =  0.190601822277923;
    Leb_Grid_XYZW[ 1083][2] = -0.755358414353351;
    Leb_Grid_XYZW[ 1083][3] =  0.000913157800319;

    Leb_Grid_XYZW[ 1084][0] =  0.626980550902439;
    Leb_Grid_XYZW[ 1084][1] = -0.190601822277923;
    Leb_Grid_XYZW[ 1084][2] =  0.755358414353351;
    Leb_Grid_XYZW[ 1084][3] =  0.000913157800319;

    Leb_Grid_XYZW[ 1085][0] =  0.626980550902439;
    Leb_Grid_XYZW[ 1085][1] = -0.190601822277923;
    Leb_Grid_XYZW[ 1085][2] = -0.755358414353351;
    Leb_Grid_XYZW[ 1085][3] =  0.000913157800319;

    Leb_Grid_XYZW[ 1086][0] = -0.626980550902439;
    Leb_Grid_XYZW[ 1086][1] =  0.190601822277923;
    Leb_Grid_XYZW[ 1086][2] =  0.755358414353351;
    Leb_Grid_XYZW[ 1086][3] =  0.000913157800319;

    Leb_Grid_XYZW[ 1087][0] = -0.626980550902439;
    Leb_Grid_XYZW[ 1087][1] =  0.190601822277923;
    Leb_Grid_XYZW[ 1087][2] = -0.755358414353351;
    Leb_Grid_XYZW[ 1087][3] =  0.000913157800319;

    Leb_Grid_XYZW[ 1088][0] = -0.626980550902439;
    Leb_Grid_XYZW[ 1088][1] = -0.190601822277923;
    Leb_Grid_XYZW[ 1088][2] =  0.755358414353351;
    Leb_Grid_XYZW[ 1088][3] =  0.000913157800319;

    Leb_Grid_XYZW[ 1089][0] = -0.626980550902439;
    Leb_Grid_XYZW[ 1089][1] = -0.190601822277923;
    Leb_Grid_XYZW[ 1089][2] = -0.755358414353351;
    Leb_Grid_XYZW[ 1089][3] =  0.000913157800319;

    Leb_Grid_XYZW[ 1090][0] =  0.190601822277923;
    Leb_Grid_XYZW[ 1090][1] =  0.755358414353351;
    Leb_Grid_XYZW[ 1090][2] =  0.626980550902439;
    Leb_Grid_XYZW[ 1090][3] =  0.000913157800319;

    Leb_Grid_XYZW[ 1091][0] =  0.190601822277923;
    Leb_Grid_XYZW[ 1091][1] =  0.755358414353351;
    Leb_Grid_XYZW[ 1091][2] = -0.626980550902439;
    Leb_Grid_XYZW[ 1091][3] =  0.000913157800319;

    Leb_Grid_XYZW[ 1092][0] =  0.190601822277923;
    Leb_Grid_XYZW[ 1092][1] = -0.755358414353351;
    Leb_Grid_XYZW[ 1092][2] =  0.626980550902439;
    Leb_Grid_XYZW[ 1092][3] =  0.000913157800319;

    Leb_Grid_XYZW[ 1093][0] =  0.190601822277923;
    Leb_Grid_XYZW[ 1093][1] = -0.755358414353351;
    Leb_Grid_XYZW[ 1093][2] = -0.626980550902439;
    Leb_Grid_XYZW[ 1093][3] =  0.000913157800319;

    Leb_Grid_XYZW[ 1094][0] = -0.190601822277923;
    Leb_Grid_XYZW[ 1094][1] =  0.755358414353351;
    Leb_Grid_XYZW[ 1094][2] =  0.626980550902439;
    Leb_Grid_XYZW[ 1094][3] =  0.000913157800319;

    Leb_Grid_XYZW[ 1095][0] = -0.190601822277923;
    Leb_Grid_XYZW[ 1095][1] =  0.755358414353351;
    Leb_Grid_XYZW[ 1095][2] = -0.626980550902439;
    Leb_Grid_XYZW[ 1095][3] =  0.000913157800319;

    Leb_Grid_XYZW[ 1096][0] = -0.190601822277923;
    Leb_Grid_XYZW[ 1096][1] = -0.755358414353351;
    Leb_Grid_XYZW[ 1096][2] =  0.626980550902439;
    Leb_Grid_XYZW[ 1096][3] =  0.000913157800319;

    Leb_Grid_XYZW[ 1097][0] = -0.190601822277923;
    Leb_Grid_XYZW[ 1097][1] = -0.755358414353351;
    Leb_Grid_XYZW[ 1097][2] = -0.626980550902439;
    Leb_Grid_XYZW[ 1097][3] =  0.000913157800319;

    Leb_Grid_XYZW[ 1098][0] =  0.190601822277923;
    Leb_Grid_XYZW[ 1098][1] =  0.626980550902439;
    Leb_Grid_XYZW[ 1098][2] =  0.755358414353351;
    Leb_Grid_XYZW[ 1098][3] =  0.000913157800319;

    Leb_Grid_XYZW[ 1099][0] =  0.190601822277923;
    Leb_Grid_XYZW[ 1099][1] =  0.626980550902439;
    Leb_Grid_XYZW[ 1099][2] = -0.755358414353351;
    Leb_Grid_XYZW[ 1099][3] =  0.000913157800319;

    Leb_Grid_XYZW[ 1100][0] =  0.190601822277923;
    Leb_Grid_XYZW[ 1100][1] = -0.626980550902439;
    Leb_Grid_XYZW[ 1100][2] =  0.755358414353351;
    Leb_Grid_XYZW[ 1100][3] =  0.000913157800319;

    Leb_Grid_XYZW[ 1101][0] =  0.190601822277923;
    Leb_Grid_XYZW[ 1101][1] = -0.626980550902439;
    Leb_Grid_XYZW[ 1101][2] = -0.755358414353351;
    Leb_Grid_XYZW[ 1101][3] =  0.000913157800319;

    Leb_Grid_XYZW[ 1102][0] = -0.190601822277923;
    Leb_Grid_XYZW[ 1102][1] =  0.626980550902439;
    Leb_Grid_XYZW[ 1102][2] =  0.755358414353351;
    Leb_Grid_XYZW[ 1102][3] =  0.000913157800319;

    Leb_Grid_XYZW[ 1103][0] = -0.190601822277923;
    Leb_Grid_XYZW[ 1103][1] =  0.626980550902439;
    Leb_Grid_XYZW[ 1103][2] = -0.755358414353351;
    Leb_Grid_XYZW[ 1103][3] =  0.000913157800319;

    Leb_Grid_XYZW[ 1104][0] = -0.190601822277923;
    Leb_Grid_XYZW[ 1104][1] = -0.626980550902439;
    Leb_Grid_XYZW[ 1104][2] =  0.755358414353351;
    Leb_Grid_XYZW[ 1104][3] =  0.000913157800319;

    Leb_Grid_XYZW[ 1105][0] = -0.190601822277923;
    Leb_Grid_XYZW[ 1105][1] = -0.626980550902439;
    Leb_Grid_XYZW[ 1105][2] = -0.755358414353351;
    Leb_Grid_XYZW[ 1105][3] =  0.000913157800319;

    Leb_Grid_XYZW[ 1106][0] =  0.734430575755950;
    Leb_Grid_XYZW[ 1106][1] =  0.603116169309631;
    Leb_Grid_XYZW[ 1106][2] =  0.311227594714961;
    Leb_Grid_XYZW[ 1106][3] =  0.000910781357948;

    Leb_Grid_XYZW[ 1107][0] =  0.734430575755950;
    Leb_Grid_XYZW[ 1107][1] =  0.603116169309631;
    Leb_Grid_XYZW[ 1107][2] = -0.311227594714961;
    Leb_Grid_XYZW[ 1107][3] =  0.000910781357948;

    Leb_Grid_XYZW[ 1108][0] =  0.734430575755950;
    Leb_Grid_XYZW[ 1108][1] = -0.603116169309631;
    Leb_Grid_XYZW[ 1108][2] =  0.311227594714961;
    Leb_Grid_XYZW[ 1108][3] =  0.000910781357948;

    Leb_Grid_XYZW[ 1109][0] =  0.734430575755950;
    Leb_Grid_XYZW[ 1109][1] = -0.603116169309631;
    Leb_Grid_XYZW[ 1109][2] = -0.311227594714961;
    Leb_Grid_XYZW[ 1109][3] =  0.000910781357948;

    Leb_Grid_XYZW[ 1110][0] = -0.734430575755950;
    Leb_Grid_XYZW[ 1110][1] =  0.603116169309631;
    Leb_Grid_XYZW[ 1110][2] =  0.311227594714961;
    Leb_Grid_XYZW[ 1110][3] =  0.000910781357948;

    Leb_Grid_XYZW[ 1111][0] = -0.734430575755950;
    Leb_Grid_XYZW[ 1111][1] =  0.603116169309631;
    Leb_Grid_XYZW[ 1111][2] = -0.311227594714961;
    Leb_Grid_XYZW[ 1111][3] =  0.000910781357948;

    Leb_Grid_XYZW[ 1112][0] = -0.734430575755950;
    Leb_Grid_XYZW[ 1112][1] = -0.603116169309631;
    Leb_Grid_XYZW[ 1112][2] =  0.311227594714961;
    Leb_Grid_XYZW[ 1112][3] =  0.000910781357948;

    Leb_Grid_XYZW[ 1113][0] = -0.734430575755950;
    Leb_Grid_XYZW[ 1113][1] = -0.603116169309631;
    Leb_Grid_XYZW[ 1113][2] = -0.311227594714961;
    Leb_Grid_XYZW[ 1113][3] =  0.000910781357948;

    Leb_Grid_XYZW[ 1114][0] =  0.734430575755950;
    Leb_Grid_XYZW[ 1114][1] =  0.311227594714961;
    Leb_Grid_XYZW[ 1114][2] =  0.603116169309631;
    Leb_Grid_XYZW[ 1114][3] =  0.000910781357948;

    Leb_Grid_XYZW[ 1115][0] =  0.734430575755950;
    Leb_Grid_XYZW[ 1115][1] =  0.311227594714961;
    Leb_Grid_XYZW[ 1115][2] = -0.603116169309631;
    Leb_Grid_XYZW[ 1115][3] =  0.000910781357948;

    Leb_Grid_XYZW[ 1116][0] =  0.734430575755950;
    Leb_Grid_XYZW[ 1116][1] = -0.311227594714961;
    Leb_Grid_XYZW[ 1116][2] =  0.603116169309631;
    Leb_Grid_XYZW[ 1116][3] =  0.000910781357948;

    Leb_Grid_XYZW[ 1117][0] =  0.734430575755950;
    Leb_Grid_XYZW[ 1117][1] = -0.311227594714961;
    Leb_Grid_XYZW[ 1117][2] = -0.603116169309631;
    Leb_Grid_XYZW[ 1117][3] =  0.000910781357948;

    Leb_Grid_XYZW[ 1118][0] = -0.734430575755950;
    Leb_Grid_XYZW[ 1118][1] =  0.311227594714961;
    Leb_Grid_XYZW[ 1118][2] =  0.603116169309631;
    Leb_Grid_XYZW[ 1118][3] =  0.000910781357948;

    Leb_Grid_XYZW[ 1119][0] = -0.734430575755950;
    Leb_Grid_XYZW[ 1119][1] =  0.311227594714961;
    Leb_Grid_XYZW[ 1119][2] = -0.603116169309631;
    Leb_Grid_XYZW[ 1119][3] =  0.000910781357948;

    Leb_Grid_XYZW[ 1120][0] = -0.734430575755950;
    Leb_Grid_XYZW[ 1120][1] = -0.311227594714961;
    Leb_Grid_XYZW[ 1120][2] =  0.603116169309631;
    Leb_Grid_XYZW[ 1120][3] =  0.000910781357948;

    Leb_Grid_XYZW[ 1121][0] = -0.734430575755950;
    Leb_Grid_XYZW[ 1121][1] = -0.311227594714961;
    Leb_Grid_XYZW[ 1121][2] = -0.603116169309631;
    Leb_Grid_XYZW[ 1121][3] =  0.000910781357948;

    Leb_Grid_XYZW[ 1122][0] =  0.603116169309631;
    Leb_Grid_XYZW[ 1122][1] =  0.734430575755950;
    Leb_Grid_XYZW[ 1122][2] =  0.311227594714961;
    Leb_Grid_XYZW[ 1122][3] =  0.000910781357948;

    Leb_Grid_XYZW[ 1123][0] =  0.603116169309631;
    Leb_Grid_XYZW[ 1123][1] =  0.734430575755950;
    Leb_Grid_XYZW[ 1123][2] = -0.311227594714961;
    Leb_Grid_XYZW[ 1123][3] =  0.000910781357948;

    Leb_Grid_XYZW[ 1124][0] =  0.603116169309631;
    Leb_Grid_XYZW[ 1124][1] = -0.734430575755950;
    Leb_Grid_XYZW[ 1124][2] =  0.311227594714961;
    Leb_Grid_XYZW[ 1124][3] =  0.000910781357948;

    Leb_Grid_XYZW[ 1125][0] =  0.603116169309631;
    Leb_Grid_XYZW[ 1125][1] = -0.734430575755950;
    Leb_Grid_XYZW[ 1125][2] = -0.311227594714961;
    Leb_Grid_XYZW[ 1125][3] =  0.000910781357948;

    Leb_Grid_XYZW[ 1126][0] = -0.603116169309631;
    Leb_Grid_XYZW[ 1126][1] =  0.734430575755951;
    Leb_Grid_XYZW[ 1126][2] =  0.311227594714961;
    Leb_Grid_XYZW[ 1126][3] =  0.000910781357948;

    Leb_Grid_XYZW[ 1127][0] = -0.603116169309631;
    Leb_Grid_XYZW[ 1127][1] =  0.734430575755951;
    Leb_Grid_XYZW[ 1127][2] = -0.311227594714961;
    Leb_Grid_XYZW[ 1127][3] =  0.000910781357948;

    Leb_Grid_XYZW[ 1128][0] = -0.603116169309631;
    Leb_Grid_XYZW[ 1128][1] = -0.734430575755951;
    Leb_Grid_XYZW[ 1128][2] =  0.311227594714961;
    Leb_Grid_XYZW[ 1128][3] =  0.000910781357948;

    Leb_Grid_XYZW[ 1129][0] = -0.603116169309631;
    Leb_Grid_XYZW[ 1129][1] = -0.734430575755951;
    Leb_Grid_XYZW[ 1129][2] = -0.311227594714961;
    Leb_Grid_XYZW[ 1129][3] =  0.000910781357948;

    Leb_Grid_XYZW[ 1130][0] =  0.603116169309631;
    Leb_Grid_XYZW[ 1130][1] =  0.311227594714961;
    Leb_Grid_XYZW[ 1130][2] =  0.734430575755950;
    Leb_Grid_XYZW[ 1130][3] =  0.000910781357948;

    Leb_Grid_XYZW[ 1131][0] =  0.603116169309631;
    Leb_Grid_XYZW[ 1131][1] =  0.311227594714961;
    Leb_Grid_XYZW[ 1131][2] = -0.734430575755950;
    Leb_Grid_XYZW[ 1131][3] =  0.000910781357948;

    Leb_Grid_XYZW[ 1132][0] =  0.603116169309631;
    Leb_Grid_XYZW[ 1132][1] = -0.311227594714961;
    Leb_Grid_XYZW[ 1132][2] =  0.734430575755950;
    Leb_Grid_XYZW[ 1132][3] =  0.000910781357948;

    Leb_Grid_XYZW[ 1133][0] =  0.603116169309631;
    Leb_Grid_XYZW[ 1133][1] = -0.311227594714961;
    Leb_Grid_XYZW[ 1133][2] = -0.734430575755950;
    Leb_Grid_XYZW[ 1133][3] =  0.000910781357948;

    Leb_Grid_XYZW[ 1134][0] = -0.603116169309631;
    Leb_Grid_XYZW[ 1134][1] =  0.311227594714961;
    Leb_Grid_XYZW[ 1134][2] =  0.734430575755950;
    Leb_Grid_XYZW[ 1134][3] =  0.000910781357948;

    Leb_Grid_XYZW[ 1135][0] = -0.603116169309631;
    Leb_Grid_XYZW[ 1135][1] =  0.311227594714961;
    Leb_Grid_XYZW[ 1135][2] = -0.734430575755950;
    Leb_Grid_XYZW[ 1135][3] =  0.000910781357948;

    Leb_Grid_XYZW[ 1136][0] = -0.603116169309631;
    Leb_Grid_XYZW[ 1136][1] = -0.311227594714961;
    Leb_Grid_XYZW[ 1136][2] =  0.734430575755950;
    Leb_Grid_XYZW[ 1136][3] =  0.000910781357948;

    Leb_Grid_XYZW[ 1137][0] = -0.603116169309631;
    Leb_Grid_XYZW[ 1137][1] = -0.311227594714961;
    Leb_Grid_XYZW[ 1137][2] = -0.734430575755950;
    Leb_Grid_XYZW[ 1137][3] =  0.000910781357948;

    Leb_Grid_XYZW[ 1138][0] =  0.311227594714961;
    Leb_Grid_XYZW[ 1138][1] =  0.734430575755950;
    Leb_Grid_XYZW[ 1138][2] =  0.603116169309631;
    Leb_Grid_XYZW[ 1138][3] =  0.000910781357948;

    Leb_Grid_XYZW[ 1139][0] =  0.311227594714961;
    Leb_Grid_XYZW[ 1139][1] =  0.734430575755950;
    Leb_Grid_XYZW[ 1139][2] = -0.603116169309631;
    Leb_Grid_XYZW[ 1139][3] =  0.000910781357948;

    Leb_Grid_XYZW[ 1140][0] =  0.311227594714961;
    Leb_Grid_XYZW[ 1140][1] = -0.734430575755950;
    Leb_Grid_XYZW[ 1140][2] =  0.603116169309631;
    Leb_Grid_XYZW[ 1140][3] =  0.000910781357948;

    Leb_Grid_XYZW[ 1141][0] =  0.311227594714961;
    Leb_Grid_XYZW[ 1141][1] = -0.734430575755950;
    Leb_Grid_XYZW[ 1141][2] = -0.603116169309631;
    Leb_Grid_XYZW[ 1141][3] =  0.000910781357948;

    Leb_Grid_XYZW[ 1142][0] = -0.311227594714961;
    Leb_Grid_XYZW[ 1142][1] =  0.734430575755950;
    Leb_Grid_XYZW[ 1142][2] =  0.603116169309631;
    Leb_Grid_XYZW[ 1142][3] =  0.000910781357948;

    Leb_Grid_XYZW[ 1143][0] = -0.311227594714961;
    Leb_Grid_XYZW[ 1143][1] =  0.734430575755950;
    Leb_Grid_XYZW[ 1143][2] = -0.603116169309631;
    Leb_Grid_XYZW[ 1143][3] =  0.000910781357948;

    Leb_Grid_XYZW[ 1144][0] = -0.311227594714961;
    Leb_Grid_XYZW[ 1144][1] = -0.734430575755950;
    Leb_Grid_XYZW[ 1144][2] =  0.603116169309631;
    Leb_Grid_XYZW[ 1144][3] =  0.000910781357948;

    Leb_Grid_XYZW[ 1145][0] = -0.311227594714961;
    Leb_Grid_XYZW[ 1145][1] = -0.734430575755950;
    Leb_Grid_XYZW[ 1145][2] = -0.603116169309631;
    Leb_Grid_XYZW[ 1145][3] =  0.000910781357948;

    Leb_Grid_XYZW[ 1146][0] =  0.311227594714961;
    Leb_Grid_XYZW[ 1146][1] =  0.603116169309631;
    Leb_Grid_XYZW[ 1146][2] =  0.734430575755950;
    Leb_Grid_XYZW[ 1146][3] =  0.000910781357948;

    Leb_Grid_XYZW[ 1147][0] =  0.311227594714961;
    Leb_Grid_XYZW[ 1147][1] =  0.603116169309631;
    Leb_Grid_XYZW[ 1147][2] = -0.734430575755950;
    Leb_Grid_XYZW[ 1147][3] =  0.000910781357948;

    Leb_Grid_XYZW[ 1148][0] =  0.311227594714961;
    Leb_Grid_XYZW[ 1148][1] = -0.603116169309631;
    Leb_Grid_XYZW[ 1148][2] =  0.734430575755950;
    Leb_Grid_XYZW[ 1148][3] =  0.000910781357948;

    Leb_Grid_XYZW[ 1149][0] =  0.311227594714961;
    Leb_Grid_XYZW[ 1149][1] = -0.603116169309631;
    Leb_Grid_XYZW[ 1149][2] = -0.734430575755950;
    Leb_Grid_XYZW[ 1149][3] =  0.000910781357948;

    Leb_Grid_XYZW[ 1150][0] = -0.311227594714961;
    Leb_Grid_XYZW[ 1150][1] =  0.603116169309631;
    Leb_Grid_XYZW[ 1150][2] =  0.734430575755950;
    Leb_Grid_XYZW[ 1150][3] =  0.000910781357948;

    Leb_Grid_XYZW[ 1151][0] = -0.311227594714961;
    Leb_Grid_XYZW[ 1151][1] =  0.603116169309631;
    Leb_Grid_XYZW[ 1151][2] = -0.734430575755950;
    Leb_Grid_XYZW[ 1151][3] =  0.000910781357948;

    Leb_Grid_XYZW[ 1152][0] = -0.311227594714961;
    Leb_Grid_XYZW[ 1152][1] = -0.603116169309631;
    Leb_Grid_XYZW[ 1152][2] =  0.734430575755950;
    Leb_Grid_XYZW[ 1152][3] =  0.000910781357948;

    Leb_Grid_XYZW[ 1153][0] = -0.311227594714961;
    Leb_Grid_XYZW[ 1153][1] = -0.603116169309631;
    Leb_Grid_XYZW[ 1153][2] = -0.734430575755950;
    Leb_Grid_XYZW[ 1153][3] =  0.000910781357948;

    Leb_Grid_XYZW[ 1154][0] =  0.704383718402177;
    Leb_Grid_XYZW[ 1154][1] =  0.569370249846844;
    Leb_Grid_XYZW[ 1154][2] =  0.423864478152234;
    Leb_Grid_XYZW[ 1154][3] =  0.000910576025897;

    Leb_Grid_XYZW[ 1155][0] =  0.704383718402177;
    Leb_Grid_XYZW[ 1155][1] =  0.569370249846844;
    Leb_Grid_XYZW[ 1155][2] = -0.423864478152234;
    Leb_Grid_XYZW[ 1155][3] =  0.000910576025897;

    Leb_Grid_XYZW[ 1156][0] =  0.704383718402177;
    Leb_Grid_XYZW[ 1156][1] = -0.569370249846844;
    Leb_Grid_XYZW[ 1156][2] =  0.423864478152234;
    Leb_Grid_XYZW[ 1156][3] =  0.000910576025897;

    Leb_Grid_XYZW[ 1157][0] =  0.704383718402177;
    Leb_Grid_XYZW[ 1157][1] = -0.569370249846844;
    Leb_Grid_XYZW[ 1157][2] = -0.423864478152234;
    Leb_Grid_XYZW[ 1157][3] =  0.000910576025897;

    Leb_Grid_XYZW[ 1158][0] = -0.704383718402176;
    Leb_Grid_XYZW[ 1158][1] =  0.569370249846844;
    Leb_Grid_XYZW[ 1158][2] =  0.423864478152234;
    Leb_Grid_XYZW[ 1158][3] =  0.000910576025897;

    Leb_Grid_XYZW[ 1159][0] = -0.704383718402177;
    Leb_Grid_XYZW[ 1159][1] =  0.569370249846844;
    Leb_Grid_XYZW[ 1159][2] = -0.423864478152234;
    Leb_Grid_XYZW[ 1159][3] =  0.000910576025897;

    Leb_Grid_XYZW[ 1160][0] = -0.704383718402176;
    Leb_Grid_XYZW[ 1160][1] = -0.569370249846844;
    Leb_Grid_XYZW[ 1160][2] =  0.423864478152234;
    Leb_Grid_XYZW[ 1160][3] =  0.000910576025897;

    Leb_Grid_XYZW[ 1161][0] = -0.704383718402177;
    Leb_Grid_XYZW[ 1161][1] = -0.569370249846844;
    Leb_Grid_XYZW[ 1161][2] = -0.423864478152234;
    Leb_Grid_XYZW[ 1161][3] =  0.000910576025897;

    Leb_Grid_XYZW[ 1162][0] =  0.704383718402176;
    Leb_Grid_XYZW[ 1162][1] =  0.423864478152234;
    Leb_Grid_XYZW[ 1162][2] =  0.569370249846844;
    Leb_Grid_XYZW[ 1162][3] =  0.000910576025897;

    Leb_Grid_XYZW[ 1163][0] =  0.704383718402176;
    Leb_Grid_XYZW[ 1163][1] =  0.423864478152234;
    Leb_Grid_XYZW[ 1163][2] = -0.569370249846844;
    Leb_Grid_XYZW[ 1163][3] =  0.000910576025897;

    Leb_Grid_XYZW[ 1164][0] =  0.704383718402176;
    Leb_Grid_XYZW[ 1164][1] = -0.423864478152234;
    Leb_Grid_XYZW[ 1164][2] =  0.569370249846844;
    Leb_Grid_XYZW[ 1164][3] =  0.000910576025897;

    Leb_Grid_XYZW[ 1165][0] =  0.704383718402176;
    Leb_Grid_XYZW[ 1165][1] = -0.423864478152234;
    Leb_Grid_XYZW[ 1165][2] = -0.569370249846844;
    Leb_Grid_XYZW[ 1165][3] =  0.000910576025897;

    Leb_Grid_XYZW[ 1166][0] = -0.704383718402177;
    Leb_Grid_XYZW[ 1166][1] =  0.423864478152234;
    Leb_Grid_XYZW[ 1166][2] =  0.569370249846844;
    Leb_Grid_XYZW[ 1166][3] =  0.000910576025897;

    Leb_Grid_XYZW[ 1167][0] = -0.704383718402177;
    Leb_Grid_XYZW[ 1167][1] =  0.423864478152234;
    Leb_Grid_XYZW[ 1167][2] = -0.569370249846844;
    Leb_Grid_XYZW[ 1167][3] =  0.000910576025897;

    Leb_Grid_XYZW[ 1168][0] = -0.704383718402177;
    Leb_Grid_XYZW[ 1168][1] = -0.423864478152234;
    Leb_Grid_XYZW[ 1168][2] =  0.569370249846844;
    Leb_Grid_XYZW[ 1168][3] =  0.000910576025897;

    Leb_Grid_XYZW[ 1169][0] = -0.704383718402177;
    Leb_Grid_XYZW[ 1169][1] = -0.423864478152234;
    Leb_Grid_XYZW[ 1169][2] = -0.569370249846844;
    Leb_Grid_XYZW[ 1169][3] =  0.000910576025897;

    Leb_Grid_XYZW[ 1170][0] =  0.569370249846844;
    Leb_Grid_XYZW[ 1170][1] =  0.704383718402177;
    Leb_Grid_XYZW[ 1170][2] =  0.423864478152234;
    Leb_Grid_XYZW[ 1170][3] =  0.000910576025897;

    Leb_Grid_XYZW[ 1171][0] =  0.569370249846844;
    Leb_Grid_XYZW[ 1171][1] =  0.704383718402177;
    Leb_Grid_XYZW[ 1171][2] = -0.423864478152234;
    Leb_Grid_XYZW[ 1171][3] =  0.000910576025897;

    Leb_Grid_XYZW[ 1172][0] =  0.569370249846844;
    Leb_Grid_XYZW[ 1172][1] = -0.704383718402177;
    Leb_Grid_XYZW[ 1172][2] =  0.423864478152234;
    Leb_Grid_XYZW[ 1172][3] =  0.000910576025897;

    Leb_Grid_XYZW[ 1173][0] =  0.569370249846844;
    Leb_Grid_XYZW[ 1173][1] = -0.704383718402177;
    Leb_Grid_XYZW[ 1173][2] = -0.423864478152234;
    Leb_Grid_XYZW[ 1173][3] =  0.000910576025897;

    Leb_Grid_XYZW[ 1174][0] = -0.569370249846844;
    Leb_Grid_XYZW[ 1174][1] =  0.704383718402177;
    Leb_Grid_XYZW[ 1174][2] =  0.423864478152234;
    Leb_Grid_XYZW[ 1174][3] =  0.000910576025897;

    Leb_Grid_XYZW[ 1175][0] = -0.569370249846844;
    Leb_Grid_XYZW[ 1175][1] =  0.704383718402177;
    Leb_Grid_XYZW[ 1175][2] = -0.423864478152234;
    Leb_Grid_XYZW[ 1175][3] =  0.000910576025897;

    Leb_Grid_XYZW[ 1176][0] = -0.569370249846844;
    Leb_Grid_XYZW[ 1176][1] = -0.704383718402177;
    Leb_Grid_XYZW[ 1176][2] =  0.423864478152234;
    Leb_Grid_XYZW[ 1176][3] =  0.000910576025897;

    Leb_Grid_XYZW[ 1177][0] = -0.569370249846844;
    Leb_Grid_XYZW[ 1177][1] = -0.704383718402177;
    Leb_Grid_XYZW[ 1177][2] = -0.423864478152234;
    Leb_Grid_XYZW[ 1177][3] =  0.000910576025897;

    Leb_Grid_XYZW[ 1178][0] =  0.569370249846844;
    Leb_Grid_XYZW[ 1178][1] =  0.423864478152234;
    Leb_Grid_XYZW[ 1178][2] =  0.704383718402176;
    Leb_Grid_XYZW[ 1178][3] =  0.000910576025897;

    Leb_Grid_XYZW[ 1179][0] =  0.569370249846844;
    Leb_Grid_XYZW[ 1179][1] =  0.423864478152234;
    Leb_Grid_XYZW[ 1179][2] = -0.704383718402176;
    Leb_Grid_XYZW[ 1179][3] =  0.000910576025897;

    Leb_Grid_XYZW[ 1180][0] =  0.569370249846844;
    Leb_Grid_XYZW[ 1180][1] = -0.423864478152234;
    Leb_Grid_XYZW[ 1180][2] =  0.704383718402176;
    Leb_Grid_XYZW[ 1180][3] =  0.000910576025897;

    Leb_Grid_XYZW[ 1181][0] =  0.569370249846844;
    Leb_Grid_XYZW[ 1181][1] = -0.423864478152234;
    Leb_Grid_XYZW[ 1181][2] = -0.704383718402176;
    Leb_Grid_XYZW[ 1181][3] =  0.000910576025897;

    Leb_Grid_XYZW[ 1182][0] = -0.569370249846844;
    Leb_Grid_XYZW[ 1182][1] =  0.423864478152234;
    Leb_Grid_XYZW[ 1182][2] =  0.704383718402176;
    Leb_Grid_XYZW[ 1182][3] =  0.000910576025897;

    Leb_Grid_XYZW[ 1183][0] = -0.569370249846844;
    Leb_Grid_XYZW[ 1183][1] =  0.423864478152234;
    Leb_Grid_XYZW[ 1183][2] = -0.704383718402176;
    Leb_Grid_XYZW[ 1183][3] =  0.000910576025897;

    Leb_Grid_XYZW[ 1184][0] = -0.569370249846844;
    Leb_Grid_XYZW[ 1184][1] = -0.423864478152234;
    Leb_Grid_XYZW[ 1184][2] =  0.704383718402176;
    Leb_Grid_XYZW[ 1184][3] =  0.000910576025897;

    Leb_Grid_XYZW[ 1185][0] = -0.569370249846844;
    Leb_Grid_XYZW[ 1185][1] = -0.423864478152234;
    Leb_Grid_XYZW[ 1185][2] = -0.704383718402176;
    Leb_Grid_XYZW[ 1185][3] =  0.000910576025897;

    Leb_Grid_XYZW[ 1186][0] =  0.423864478152234;
    Leb_Grid_XYZW[ 1186][1] =  0.704383718402176;
    Leb_Grid_XYZW[ 1186][2] =  0.569370249846844;
    Leb_Grid_XYZW[ 1186][3] =  0.000910576025897;

    Leb_Grid_XYZW[ 1187][0] =  0.423864478152234;
    Leb_Grid_XYZW[ 1187][1] =  0.704383718402176;
    Leb_Grid_XYZW[ 1187][2] = -0.569370249846844;
    Leb_Grid_XYZW[ 1187][3] =  0.000910576025897;

    Leb_Grid_XYZW[ 1188][0] =  0.423864478152234;
    Leb_Grid_XYZW[ 1188][1] = -0.704383718402176;
    Leb_Grid_XYZW[ 1188][2] =  0.569370249846844;
    Leb_Grid_XYZW[ 1188][3] =  0.000910576025897;

    Leb_Grid_XYZW[ 1189][0] =  0.423864478152234;
    Leb_Grid_XYZW[ 1189][1] = -0.704383718402176;
    Leb_Grid_XYZW[ 1189][2] = -0.569370249846844;
    Leb_Grid_XYZW[ 1189][3] =  0.000910576025897;

    Leb_Grid_XYZW[ 1190][0] = -0.423864478152234;
    Leb_Grid_XYZW[ 1190][1] =  0.704383718402176;
    Leb_Grid_XYZW[ 1190][2] =  0.569370249846844;
    Leb_Grid_XYZW[ 1190][3] =  0.000910576025897;

    Leb_Grid_XYZW[ 1191][0] = -0.423864478152234;
    Leb_Grid_XYZW[ 1191][1] =  0.704383718402176;
    Leb_Grid_XYZW[ 1191][2] = -0.569370249846844;
    Leb_Grid_XYZW[ 1191][3] =  0.000910576025897;

    Leb_Grid_XYZW[ 1192][0] = -0.423864478152234;
    Leb_Grid_XYZW[ 1192][1] = -0.704383718402176;
    Leb_Grid_XYZW[ 1192][2] =  0.569370249846844;
    Leb_Grid_XYZW[ 1192][3] =  0.000910576025897;

    Leb_Grid_XYZW[ 1193][0] = -0.423864478152234;
    Leb_Grid_XYZW[ 1193][1] = -0.704383718402176;
    Leb_Grid_XYZW[ 1193][2] = -0.569370249846844;
    Leb_Grid_XYZW[ 1193][3] =  0.000910576025897;

    Leb_Grid_XYZW[ 1194][0] =  0.423864478152234;
    Leb_Grid_XYZW[ 1194][1] =  0.569370249846844;
    Leb_Grid_XYZW[ 1194][2] =  0.704383718402176;
    Leb_Grid_XYZW[ 1194][3] =  0.000910576025897;

    Leb_Grid_XYZW[ 1195][0] =  0.423864478152234;
    Leb_Grid_XYZW[ 1195][1] =  0.569370249846844;
    Leb_Grid_XYZW[ 1195][2] = -0.704383718402176;
    Leb_Grid_XYZW[ 1195][3] =  0.000910576025897;

    Leb_Grid_XYZW[ 1196][0] =  0.423864478152234;
    Leb_Grid_XYZW[ 1196][1] = -0.569370249846844;
    Leb_Grid_XYZW[ 1196][2] =  0.704383718402176;
    Leb_Grid_XYZW[ 1196][3] =  0.000910576025897;

    Leb_Grid_XYZW[ 1197][0] =  0.423864478152234;
    Leb_Grid_XYZW[ 1197][1] = -0.569370249846844;
    Leb_Grid_XYZW[ 1197][2] = -0.704383718402176;
    Leb_Grid_XYZW[ 1197][3] =  0.000910576025897;

    Leb_Grid_XYZW[ 1198][0] = -0.423864478152234;
    Leb_Grid_XYZW[ 1198][1] =  0.569370249846844;
    Leb_Grid_XYZW[ 1198][2] =  0.704383718402176;
    Leb_Grid_XYZW[ 1198][3] =  0.000910576025897;

    Leb_Grid_XYZW[ 1199][0] = -0.423864478152234;
    Leb_Grid_XYZW[ 1199][1] =  0.569370249846844;
    Leb_Grid_XYZW[ 1199][2] = -0.704383718402176;
    Leb_Grid_XYZW[ 1199][3] =  0.000910576025897;

    Leb_Grid_XYZW[ 1200][0] = -0.423864478152234;
    Leb_Grid_XYZW[ 1200][1] = -0.569370249846844;
    Leb_Grid_XYZW[ 1200][2] =  0.704383718402176;
    Leb_Grid_XYZW[ 1200][3] =  0.000910576025897;

    Leb_Grid_XYZW[ 1201][0] = -0.423864478152234;
    Leb_Grid_XYZW[ 1201][1] = -0.569370249846844;
    Leb_Grid_XYZW[ 1201][2] = -0.704383718402176;
    Leb_Grid_XYZW[ 1201][3] =  0.000910576025897;

  }

  /* 2030 */

  else if (Np==2030){

    Leb_Grid_XYZW[    0][0] =  1.000000000000000;
    Leb_Grid_XYZW[    0][1] =  0.000000000000000;
    Leb_Grid_XYZW[    0][2] =  0.000000000000000;
    Leb_Grid_XYZW[    0][3] =  0.000046560318992;

    Leb_Grid_XYZW[    1][0] = -1.000000000000000;
    Leb_Grid_XYZW[    1][1] =  0.000000000000000;
    Leb_Grid_XYZW[    1][2] =  0.000000000000000;
    Leb_Grid_XYZW[    1][3] =  0.000046560318992;

    Leb_Grid_XYZW[    2][0] =  0.000000000000000;
    Leb_Grid_XYZW[    2][1] =  1.000000000000000;
    Leb_Grid_XYZW[    2][2] =  0.000000000000000;
    Leb_Grid_XYZW[    2][3] =  0.000046560318992;

    Leb_Grid_XYZW[    3][0] =  0.000000000000000;
    Leb_Grid_XYZW[    3][1] = -1.000000000000000;
    Leb_Grid_XYZW[    3][2] =  0.000000000000000;
    Leb_Grid_XYZW[    3][3] =  0.000046560318992;

    Leb_Grid_XYZW[    4][0] =  0.000000000000000;
    Leb_Grid_XYZW[    4][1] =  0.000000000000000;
    Leb_Grid_XYZW[    4][2] =  1.000000000000000;
    Leb_Grid_XYZW[    4][3] =  0.000046560318992;

    Leb_Grid_XYZW[    5][0] =  0.000000000000000;
    Leb_Grid_XYZW[    5][1] =  0.000000000000000;
    Leb_Grid_XYZW[    5][2] = -1.000000000000000;
    Leb_Grid_XYZW[    5][3] =  0.000046560318992;

    Leb_Grid_XYZW[    6][0] =  0.577350269189626;
    Leb_Grid_XYZW[    6][1] =  0.577350269189626;
    Leb_Grid_XYZW[    6][2] =  0.577350269189626;
    Leb_Grid_XYZW[    6][3] =  0.000542154919530;

    Leb_Grid_XYZW[    7][0] =  0.577350269189626;
    Leb_Grid_XYZW[    7][1] =  0.577350269189626;
    Leb_Grid_XYZW[    7][2] = -0.577350269189626;
    Leb_Grid_XYZW[    7][3] =  0.000542154919530;

    Leb_Grid_XYZW[    8][0] =  0.577350269189626;
    Leb_Grid_XYZW[    8][1] = -0.577350269189626;
    Leb_Grid_XYZW[    8][2] =  0.577350269189626;
    Leb_Grid_XYZW[    8][3] =  0.000542154919530;

    Leb_Grid_XYZW[    9][0] =  0.577350269189626;
    Leb_Grid_XYZW[    9][1] = -0.577350269189626;
    Leb_Grid_XYZW[    9][2] = -0.577350269189626;
    Leb_Grid_XYZW[    9][3] =  0.000542154919530;

    Leb_Grid_XYZW[   10][0] = -0.577350269189626;
    Leb_Grid_XYZW[   10][1] =  0.577350269189626;
    Leb_Grid_XYZW[   10][2] =  0.577350269189626;
    Leb_Grid_XYZW[   10][3] =  0.000542154919530;

    Leb_Grid_XYZW[   11][0] = -0.577350269189626;
    Leb_Grid_XYZW[   11][1] =  0.577350269189626;
    Leb_Grid_XYZW[   11][2] = -0.577350269189626;
    Leb_Grid_XYZW[   11][3] =  0.000542154919530;

    Leb_Grid_XYZW[   12][0] = -0.577350269189626;
    Leb_Grid_XYZW[   12][1] = -0.577350269189626;
    Leb_Grid_XYZW[   12][2] =  0.577350269189626;
    Leb_Grid_XYZW[   12][3] =  0.000542154919530;

    Leb_Grid_XYZW[   13][0] = -0.577350269189626;
    Leb_Grid_XYZW[   13][1] = -0.577350269189626;
    Leb_Grid_XYZW[   13][2] = -0.577350269189626;
    Leb_Grid_XYZW[   13][3] =  0.000542154919530;

    Leb_Grid_XYZW[   14][0] =  0.025408353368143;
    Leb_Grid_XYZW[   14][1] =  0.025408353368143;
    Leb_Grid_XYZW[   14][2] =  0.999354207054856;
    Leb_Grid_XYZW[   14][3] =  0.000177852213335;

    Leb_Grid_XYZW[   15][0] =  0.025408353368143;
    Leb_Grid_XYZW[   15][1] =  0.025408353368143;
    Leb_Grid_XYZW[   15][2] = -0.999354207054856;
    Leb_Grid_XYZW[   15][3] =  0.000177852213335;

    Leb_Grid_XYZW[   16][0] =  0.025408353368143;
    Leb_Grid_XYZW[   16][1] = -0.025408353368143;
    Leb_Grid_XYZW[   16][2] =  0.999354207054856;
    Leb_Grid_XYZW[   16][3] =  0.000177852213335;

    Leb_Grid_XYZW[   17][0] =  0.025408353368143;
    Leb_Grid_XYZW[   17][1] = -0.025408353368143;
    Leb_Grid_XYZW[   17][2] = -0.999354207054856;
    Leb_Grid_XYZW[   17][3] =  0.000177852213335;

    Leb_Grid_XYZW[   18][0] = -0.025408353368143;
    Leb_Grid_XYZW[   18][1] =  0.025408353368143;
    Leb_Grid_XYZW[   18][2] =  0.999354207054856;
    Leb_Grid_XYZW[   18][3] =  0.000177852213335;

    Leb_Grid_XYZW[   19][0] = -0.025408353368143;
    Leb_Grid_XYZW[   19][1] =  0.025408353368143;
    Leb_Grid_XYZW[   19][2] = -0.999354207054856;
    Leb_Grid_XYZW[   19][3] =  0.000177852213335;

    Leb_Grid_XYZW[   20][0] = -0.025408353368143;
    Leb_Grid_XYZW[   20][1] = -0.025408353368143;
    Leb_Grid_XYZW[   20][2] =  0.999354207054856;
    Leb_Grid_XYZW[   20][3] =  0.000177852213335;

    Leb_Grid_XYZW[   21][0] = -0.025408353368143;
    Leb_Grid_XYZW[   21][1] = -0.025408353368143;
    Leb_Grid_XYZW[   21][2] = -0.999354207054856;
    Leb_Grid_XYZW[   21][3] =  0.000177852213335;

    Leb_Grid_XYZW[   22][0] =  0.025408353368144;
    Leb_Grid_XYZW[   22][1] =  0.999354207054855;
    Leb_Grid_XYZW[   22][2] =  0.025408353368144;
    Leb_Grid_XYZW[   22][3] =  0.000177852213335;

    Leb_Grid_XYZW[   23][0] =  0.025408353368144;
    Leb_Grid_XYZW[   23][1] = -0.999354207054855;
    Leb_Grid_XYZW[   23][2] =  0.025408353368144;
    Leb_Grid_XYZW[   23][3] =  0.000177852213335;

    Leb_Grid_XYZW[   24][0] =  0.025408353368144;
    Leb_Grid_XYZW[   24][1] =  0.999354207054855;
    Leb_Grid_XYZW[   24][2] = -0.025408353368144;
    Leb_Grid_XYZW[   24][3] =  0.000177852213335;

    Leb_Grid_XYZW[   25][0] =  0.025408353368144;
    Leb_Grid_XYZW[   25][1] = -0.999354207054855;
    Leb_Grid_XYZW[   25][2] = -0.025408353368144;
    Leb_Grid_XYZW[   25][3] =  0.000177852213335;

    Leb_Grid_XYZW[   26][0] = -0.025408353368143;
    Leb_Grid_XYZW[   26][1] =  0.999354207054855;
    Leb_Grid_XYZW[   26][2] =  0.025408353368144;
    Leb_Grid_XYZW[   26][3] =  0.000177852213335;

    Leb_Grid_XYZW[   27][0] = -0.025408353368143;
    Leb_Grid_XYZW[   27][1] = -0.999354207054855;
    Leb_Grid_XYZW[   27][2] =  0.025408353368144;
    Leb_Grid_XYZW[   27][3] =  0.000177852213335;

    Leb_Grid_XYZW[   28][0] = -0.025408353368143;
    Leb_Grid_XYZW[   28][1] =  0.999354207054855;
    Leb_Grid_XYZW[   28][2] = -0.025408353368144;
    Leb_Grid_XYZW[   28][3] =  0.000177852213335;

    Leb_Grid_XYZW[   29][0] = -0.025408353368143;
    Leb_Grid_XYZW[   29][1] = -0.999354207054855;
    Leb_Grid_XYZW[   29][2] = -0.025408353368144;
    Leb_Grid_XYZW[   29][3] =  0.000177852213335;

    Leb_Grid_XYZW[   30][0] =  0.999354207054856;
    Leb_Grid_XYZW[   30][1] =  0.025408353368141;
    Leb_Grid_XYZW[   30][2] =  0.025408353368144;
    Leb_Grid_XYZW[   30][3] =  0.000177852213335;

    Leb_Grid_XYZW[   31][0] = -0.999354207054856;
    Leb_Grid_XYZW[   31][1] =  0.025408353368141;
    Leb_Grid_XYZW[   31][2] =  0.025408353368144;
    Leb_Grid_XYZW[   31][3] =  0.000177852213335;

    Leb_Grid_XYZW[   32][0] =  0.999354207054856;
    Leb_Grid_XYZW[   32][1] =  0.025408353368141;
    Leb_Grid_XYZW[   32][2] = -0.025408353368144;
    Leb_Grid_XYZW[   32][3] =  0.000177852213335;

    Leb_Grid_XYZW[   33][0] = -0.999354207054856;
    Leb_Grid_XYZW[   33][1] =  0.025408353368141;
    Leb_Grid_XYZW[   33][2] = -0.025408353368144;
    Leb_Grid_XYZW[   33][3] =  0.000177852213335;

    Leb_Grid_XYZW[   34][0] =  0.999354207054856;
    Leb_Grid_XYZW[   34][1] = -0.025408353368141;
    Leb_Grid_XYZW[   34][2] =  0.025408353368144;
    Leb_Grid_XYZW[   34][3] =  0.000177852213335;

    Leb_Grid_XYZW[   35][0] = -0.999354207054856;
    Leb_Grid_XYZW[   35][1] = -0.025408353368141;
    Leb_Grid_XYZW[   35][2] =  0.025408353368144;
    Leb_Grid_XYZW[   35][3] =  0.000177852213335;

    Leb_Grid_XYZW[   36][0] =  0.999354207054856;
    Leb_Grid_XYZW[   36][1] = -0.025408353368141;
    Leb_Grid_XYZW[   36][2] = -0.025408353368144;
    Leb_Grid_XYZW[   36][3] =  0.000177852213335;

    Leb_Grid_XYZW[   37][0] = -0.999354207054856;
    Leb_Grid_XYZW[   37][1] = -0.025408353368141;
    Leb_Grid_XYZW[   37][2] = -0.025408353368144;
    Leb_Grid_XYZW[   37][3] =  0.000177852213335;

    Leb_Grid_XYZW[   38][0] =  0.063993228005049;
    Leb_Grid_XYZW[   38][1] =  0.063993228005049;
    Leb_Grid_XYZW[   38][2] =  0.995896447196689;
    Leb_Grid_XYZW[   38][3] =  0.000281132540568;

    Leb_Grid_XYZW[   39][0] =  0.063993228005049;
    Leb_Grid_XYZW[   39][1] =  0.063993228005049;
    Leb_Grid_XYZW[   39][2] = -0.995896447196689;
    Leb_Grid_XYZW[   39][3] =  0.000281132540568;

    Leb_Grid_XYZW[   40][0] =  0.063993228005049;
    Leb_Grid_XYZW[   40][1] = -0.063993228005049;
    Leb_Grid_XYZW[   40][2] =  0.995896447196689;
    Leb_Grid_XYZW[   40][3] =  0.000281132540568;

    Leb_Grid_XYZW[   41][0] =  0.063993228005049;
    Leb_Grid_XYZW[   41][1] = -0.063993228005049;
    Leb_Grid_XYZW[   41][2] = -0.995896447196689;
    Leb_Grid_XYZW[   41][3] =  0.000281132540568;

    Leb_Grid_XYZW[   42][0] = -0.063993228005049;
    Leb_Grid_XYZW[   42][1] =  0.063993228005049;
    Leb_Grid_XYZW[   42][2] =  0.995896447196689;
    Leb_Grid_XYZW[   42][3] =  0.000281132540568;

    Leb_Grid_XYZW[   43][0] = -0.063993228005049;
    Leb_Grid_XYZW[   43][1] =  0.063993228005049;
    Leb_Grid_XYZW[   43][2] = -0.995896447196689;
    Leb_Grid_XYZW[   43][3] =  0.000281132540568;

    Leb_Grid_XYZW[   44][0] = -0.063993228005049;
    Leb_Grid_XYZW[   44][1] = -0.063993228005049;
    Leb_Grid_XYZW[   44][2] =  0.995896447196689;
    Leb_Grid_XYZW[   44][3] =  0.000281132540568;

    Leb_Grid_XYZW[   45][0] = -0.063993228005049;
    Leb_Grid_XYZW[   45][1] = -0.063993228005049;
    Leb_Grid_XYZW[   45][2] = -0.995896447196689;
    Leb_Grid_XYZW[   45][3] =  0.000281132540568;

    Leb_Grid_XYZW[   46][0] =  0.063993228005049;
    Leb_Grid_XYZW[   46][1] =  0.995896447196689;
    Leb_Grid_XYZW[   46][2] =  0.063993228005049;
    Leb_Grid_XYZW[   46][3] =  0.000281132540568;

    Leb_Grid_XYZW[   47][0] =  0.063993228005049;
    Leb_Grid_XYZW[   47][1] = -0.995896447196689;
    Leb_Grid_XYZW[   47][2] =  0.063993228005049;
    Leb_Grid_XYZW[   47][3] =  0.000281132540568;

    Leb_Grid_XYZW[   48][0] =  0.063993228005049;
    Leb_Grid_XYZW[   48][1] =  0.995896447196689;
    Leb_Grid_XYZW[   48][2] = -0.063993228005049;
    Leb_Grid_XYZW[   48][3] =  0.000281132540568;

    Leb_Grid_XYZW[   49][0] =  0.063993228005049;
    Leb_Grid_XYZW[   49][1] = -0.995896447196689;
    Leb_Grid_XYZW[   49][2] = -0.063993228005049;
    Leb_Grid_XYZW[   49][3] =  0.000281132540568;

    Leb_Grid_XYZW[   50][0] = -0.063993228005049;
    Leb_Grid_XYZW[   50][1] =  0.995896447196689;
    Leb_Grid_XYZW[   50][2] =  0.063993228005049;
    Leb_Grid_XYZW[   50][3] =  0.000281132540568;

    Leb_Grid_XYZW[   51][0] = -0.063993228005049;
    Leb_Grid_XYZW[   51][1] = -0.995896447196689;
    Leb_Grid_XYZW[   51][2] =  0.063993228005049;
    Leb_Grid_XYZW[   51][3] =  0.000281132540568;

    Leb_Grid_XYZW[   52][0] = -0.063993228005049;
    Leb_Grid_XYZW[   52][1] =  0.995896447196689;
    Leb_Grid_XYZW[   52][2] = -0.063993228005049;
    Leb_Grid_XYZW[   52][3] =  0.000281132540568;

    Leb_Grid_XYZW[   53][0] = -0.063993228005049;
    Leb_Grid_XYZW[   53][1] = -0.995896447196689;
    Leb_Grid_XYZW[   53][2] = -0.063993228005049;
    Leb_Grid_XYZW[   53][3] =  0.000281132540568;

    Leb_Grid_XYZW[   54][0] =  0.995896447196689;
    Leb_Grid_XYZW[   54][1] =  0.063993228005049;
    Leb_Grid_XYZW[   54][2] =  0.063993228005049;
    Leb_Grid_XYZW[   54][3] =  0.000281132540568;

    Leb_Grid_XYZW[   55][0] = -0.995896447196689;
    Leb_Grid_XYZW[   55][1] =  0.063993228005049;
    Leb_Grid_XYZW[   55][2] =  0.063993228005049;
    Leb_Grid_XYZW[   55][3] =  0.000281132540568;

    Leb_Grid_XYZW[   56][0] =  0.995896447196689;
    Leb_Grid_XYZW[   56][1] =  0.063993228005049;
    Leb_Grid_XYZW[   56][2] = -0.063993228005049;
    Leb_Grid_XYZW[   56][3] =  0.000281132540568;

    Leb_Grid_XYZW[   57][0] = -0.995896447196689;
    Leb_Grid_XYZW[   57][1] =  0.063993228005049;
    Leb_Grid_XYZW[   57][2] = -0.063993228005049;
    Leb_Grid_XYZW[   57][3] =  0.000281132540568;

    Leb_Grid_XYZW[   58][0] =  0.995896447196689;
    Leb_Grid_XYZW[   58][1] = -0.063993228005049;
    Leb_Grid_XYZW[   58][2] =  0.063993228005049;
    Leb_Grid_XYZW[   58][3] =  0.000281132540568;

    Leb_Grid_XYZW[   59][0] = -0.995896447196689;
    Leb_Grid_XYZW[   59][1] = -0.063993228005049;
    Leb_Grid_XYZW[   59][2] =  0.063993228005049;
    Leb_Grid_XYZW[   59][3] =  0.000281132540568;

    Leb_Grid_XYZW[   60][0] =  0.995896447196689;
    Leb_Grid_XYZW[   60][1] = -0.063993228005049;
    Leb_Grid_XYZW[   60][2] = -0.063993228005049;
    Leb_Grid_XYZW[   60][3] =  0.000281132540568;

    Leb_Grid_XYZW[   61][0] = -0.995896447196689;
    Leb_Grid_XYZW[   61][1] = -0.063993228005049;
    Leb_Grid_XYZW[   61][2] = -0.063993228005049;
    Leb_Grid_XYZW[   61][3] =  0.000281132540568;

    Leb_Grid_XYZW[   62][0] =  0.108826946980412;
    Leb_Grid_XYZW[   62][1] =  0.108826946980412;
    Leb_Grid_XYZW[   62][2] =  0.988085720583920;
    Leb_Grid_XYZW[   62][3] =  0.000354889631263;

    Leb_Grid_XYZW[   63][0] =  0.108826946980412;
    Leb_Grid_XYZW[   63][1] =  0.108826946980412;
    Leb_Grid_XYZW[   63][2] = -0.988085720583920;
    Leb_Grid_XYZW[   63][3] =  0.000354889631263;

    Leb_Grid_XYZW[   64][0] =  0.108826946980412;
    Leb_Grid_XYZW[   64][1] = -0.108826946980412;
    Leb_Grid_XYZW[   64][2] =  0.988085720583920;
    Leb_Grid_XYZW[   64][3] =  0.000354889631263;

    Leb_Grid_XYZW[   65][0] =  0.108826946980412;
    Leb_Grid_XYZW[   65][1] = -0.108826946980412;
    Leb_Grid_XYZW[   65][2] = -0.988085720583920;
    Leb_Grid_XYZW[   65][3] =  0.000354889631263;

    Leb_Grid_XYZW[   66][0] = -0.108826946980412;
    Leb_Grid_XYZW[   66][1] =  0.108826946980412;
    Leb_Grid_XYZW[   66][2] =  0.988085720583920;
    Leb_Grid_XYZW[   66][3] =  0.000354889631263;

    Leb_Grid_XYZW[   67][0] = -0.108826946980412;
    Leb_Grid_XYZW[   67][1] =  0.108826946980412;
    Leb_Grid_XYZW[   67][2] = -0.988085720583920;
    Leb_Grid_XYZW[   67][3] =  0.000354889631263;

    Leb_Grid_XYZW[   68][0] = -0.108826946980412;
    Leb_Grid_XYZW[   68][1] = -0.108826946980412;
    Leb_Grid_XYZW[   68][2] =  0.988085720583920;
    Leb_Grid_XYZW[   68][3] =  0.000354889631263;

    Leb_Grid_XYZW[   69][0] = -0.108826946980412;
    Leb_Grid_XYZW[   69][1] = -0.108826946980412;
    Leb_Grid_XYZW[   69][2] = -0.988085720583920;
    Leb_Grid_XYZW[   69][3] =  0.000354889631263;

    Leb_Grid_XYZW[   70][0] =  0.108826946980413;
    Leb_Grid_XYZW[   70][1] =  0.988085720583920;
    Leb_Grid_XYZW[   70][2] =  0.108826946980412;
    Leb_Grid_XYZW[   70][3] =  0.000354889631263;

    Leb_Grid_XYZW[   71][0] =  0.108826946980413;
    Leb_Grid_XYZW[   71][1] = -0.988085720583920;
    Leb_Grid_XYZW[   71][2] =  0.108826946980412;
    Leb_Grid_XYZW[   71][3] =  0.000354889631263;

    Leb_Grid_XYZW[   72][0] =  0.108826946980413;
    Leb_Grid_XYZW[   72][1] =  0.988085720583920;
    Leb_Grid_XYZW[   72][2] = -0.108826946980412;
    Leb_Grid_XYZW[   72][3] =  0.000354889631263;

    Leb_Grid_XYZW[   73][0] =  0.108826946980413;
    Leb_Grid_XYZW[   73][1] = -0.988085720583920;
    Leb_Grid_XYZW[   73][2] = -0.108826946980412;
    Leb_Grid_XYZW[   73][3] =  0.000354889631263;

    Leb_Grid_XYZW[   74][0] = -0.108826946980413;
    Leb_Grid_XYZW[   74][1] =  0.988085720583920;
    Leb_Grid_XYZW[   74][2] =  0.108826946980412;
    Leb_Grid_XYZW[   74][3] =  0.000354889631263;

    Leb_Grid_XYZW[   75][0] = -0.108826946980413;
    Leb_Grid_XYZW[   75][1] = -0.988085720583920;
    Leb_Grid_XYZW[   75][2] =  0.108826946980412;
    Leb_Grid_XYZW[   75][3] =  0.000354889631263;

    Leb_Grid_XYZW[   76][0] = -0.108826946980413;
    Leb_Grid_XYZW[   76][1] =  0.988085720583920;
    Leb_Grid_XYZW[   76][2] = -0.108826946980412;
    Leb_Grid_XYZW[   76][3] =  0.000354889631263;

    Leb_Grid_XYZW[   77][0] = -0.108826946980413;
    Leb_Grid_XYZW[   77][1] = -0.988085720583920;
    Leb_Grid_XYZW[   77][2] = -0.108826946980412;
    Leb_Grid_XYZW[   77][3] =  0.000354889631263;

    Leb_Grid_XYZW[   78][0] =  0.988085720583920;
    Leb_Grid_XYZW[   78][1] =  0.108826946980413;
    Leb_Grid_XYZW[   78][2] =  0.108826946980412;
    Leb_Grid_XYZW[   78][3] =  0.000354889631263;

    Leb_Grid_XYZW[   79][0] = -0.988085720583920;
    Leb_Grid_XYZW[   79][1] =  0.108826946980413;
    Leb_Grid_XYZW[   79][2] =  0.108826946980412;
    Leb_Grid_XYZW[   79][3] =  0.000354889631263;

    Leb_Grid_XYZW[   80][0] =  0.988085720583920;
    Leb_Grid_XYZW[   80][1] =  0.108826946980413;
    Leb_Grid_XYZW[   80][2] = -0.108826946980412;
    Leb_Grid_XYZW[   80][3] =  0.000354889631263;

    Leb_Grid_XYZW[   81][0] = -0.988085720583920;
    Leb_Grid_XYZW[   81][1] =  0.108826946980413;
    Leb_Grid_XYZW[   81][2] = -0.108826946980412;
    Leb_Grid_XYZW[   81][3] =  0.000354889631263;

    Leb_Grid_XYZW[   82][0] =  0.988085720583920;
    Leb_Grid_XYZW[   82][1] = -0.108826946980413;
    Leb_Grid_XYZW[   82][2] =  0.108826946980412;
    Leb_Grid_XYZW[   82][3] =  0.000354889631263;

    Leb_Grid_XYZW[   83][0] = -0.988085720583920;
    Leb_Grid_XYZW[   83][1] = -0.108826946980413;
    Leb_Grid_XYZW[   83][2] =  0.108826946980412;
    Leb_Grid_XYZW[   83][3] =  0.000354889631263;

    Leb_Grid_XYZW[   84][0] =  0.988085720583920;
    Leb_Grid_XYZW[   84][1] = -0.108826946980413;
    Leb_Grid_XYZW[   84][2] = -0.108826946980412;
    Leb_Grid_XYZW[   84][3] =  0.000354889631263;

    Leb_Grid_XYZW[   85][0] = -0.988085720583920;
    Leb_Grid_XYZW[   85][1] = -0.108826946980413;
    Leb_Grid_XYZW[   85][2] = -0.108826946980412;
    Leb_Grid_XYZW[   85][3] =  0.000354889631263;

    Leb_Grid_XYZW[   86][0] =  0.157067079881829;
    Leb_Grid_XYZW[   86][1] =  0.157067079881829;
    Leb_Grid_XYZW[   86][2] =  0.975017879238525;
    Leb_Grid_XYZW[   86][3] =  0.000409031089717;

    Leb_Grid_XYZW[   87][0] =  0.157067079881829;
    Leb_Grid_XYZW[   87][1] =  0.157067079881829;
    Leb_Grid_XYZW[   87][2] = -0.975017879238525;
    Leb_Grid_XYZW[   87][3] =  0.000409031089717;

    Leb_Grid_XYZW[   88][0] =  0.157067079881829;
    Leb_Grid_XYZW[   88][1] = -0.157067079881829;
    Leb_Grid_XYZW[   88][2] =  0.975017879238525;
    Leb_Grid_XYZW[   88][3] =  0.000409031089717;

    Leb_Grid_XYZW[   89][0] =  0.157067079881829;
    Leb_Grid_XYZW[   89][1] = -0.157067079881829;
    Leb_Grid_XYZW[   89][2] = -0.975017879238525;
    Leb_Grid_XYZW[   89][3] =  0.000409031089717;

    Leb_Grid_XYZW[   90][0] = -0.157067079881829;
    Leb_Grid_XYZW[   90][1] =  0.157067079881829;
    Leb_Grid_XYZW[   90][2] =  0.975017879238525;
    Leb_Grid_XYZW[   90][3] =  0.000409031089717;

    Leb_Grid_XYZW[   91][0] = -0.157067079881829;
    Leb_Grid_XYZW[   91][1] =  0.157067079881829;
    Leb_Grid_XYZW[   91][2] = -0.975017879238525;
    Leb_Grid_XYZW[   91][3] =  0.000409031089717;

    Leb_Grid_XYZW[   92][0] = -0.157067079881829;
    Leb_Grid_XYZW[   92][1] = -0.157067079881829;
    Leb_Grid_XYZW[   92][2] =  0.975017879238525;
    Leb_Grid_XYZW[   92][3] =  0.000409031089717;

    Leb_Grid_XYZW[   93][0] = -0.157067079881829;
    Leb_Grid_XYZW[   93][1] = -0.157067079881829;
    Leb_Grid_XYZW[   93][2] = -0.975017879238525;
    Leb_Grid_XYZW[   93][3] =  0.000409031089717;

    Leb_Grid_XYZW[   94][0] =  0.157067079881829;
    Leb_Grid_XYZW[   94][1] =  0.975017879238525;
    Leb_Grid_XYZW[   94][2] =  0.157067079881829;
    Leb_Grid_XYZW[   94][3] =  0.000409031089717;

    Leb_Grid_XYZW[   95][0] =  0.157067079881829;
    Leb_Grid_XYZW[   95][1] = -0.975017879238525;
    Leb_Grid_XYZW[   95][2] =  0.157067079881829;
    Leb_Grid_XYZW[   95][3] =  0.000409031089717;

    Leb_Grid_XYZW[   96][0] =  0.157067079881829;
    Leb_Grid_XYZW[   96][1] =  0.975017879238525;
    Leb_Grid_XYZW[   96][2] = -0.157067079881829;
    Leb_Grid_XYZW[   96][3] =  0.000409031089717;

    Leb_Grid_XYZW[   97][0] =  0.157067079881829;
    Leb_Grid_XYZW[   97][1] = -0.975017879238525;
    Leb_Grid_XYZW[   97][2] = -0.157067079881829;
    Leb_Grid_XYZW[   97][3] =  0.000409031089717;

    Leb_Grid_XYZW[   98][0] = -0.157067079881829;
    Leb_Grid_XYZW[   98][1] =  0.975017879238525;
    Leb_Grid_XYZW[   98][2] =  0.157067079881829;
    Leb_Grid_XYZW[   98][3] =  0.000409031089717;

    Leb_Grid_XYZW[   99][0] = -0.157067079881829;
    Leb_Grid_XYZW[   99][1] = -0.975017879238525;
    Leb_Grid_XYZW[   99][2] =  0.157067079881829;
    Leb_Grid_XYZW[   99][3] =  0.000409031089717;

    Leb_Grid_XYZW[  100][0] = -0.157067079881829;
    Leb_Grid_XYZW[  100][1] =  0.975017879238525;
    Leb_Grid_XYZW[  100][2] = -0.157067079881829;
    Leb_Grid_XYZW[  100][3] =  0.000409031089717;

    Leb_Grid_XYZW[  101][0] = -0.157067079881829;
    Leb_Grid_XYZW[  101][1] = -0.975017879238525;
    Leb_Grid_XYZW[  101][2] = -0.157067079881829;
    Leb_Grid_XYZW[  101][3] =  0.000409031089717;

    Leb_Grid_XYZW[  102][0] =  0.975017879238525;
    Leb_Grid_XYZW[  102][1] =  0.157067079881828;
    Leb_Grid_XYZW[  102][2] =  0.157067079881829;
    Leb_Grid_XYZW[  102][3] =  0.000409031089717;

    Leb_Grid_XYZW[  103][0] = -0.975017879238525;
    Leb_Grid_XYZW[  103][1] =  0.157067079881828;
    Leb_Grid_XYZW[  103][2] =  0.157067079881829;
    Leb_Grid_XYZW[  103][3] =  0.000409031089717;

    Leb_Grid_XYZW[  104][0] =  0.975017879238525;
    Leb_Grid_XYZW[  104][1] =  0.157067079881828;
    Leb_Grid_XYZW[  104][2] = -0.157067079881829;
    Leb_Grid_XYZW[  104][3] =  0.000409031089717;

    Leb_Grid_XYZW[  105][0] = -0.975017879238525;
    Leb_Grid_XYZW[  105][1] =  0.157067079881828;
    Leb_Grid_XYZW[  105][2] = -0.157067079881829;
    Leb_Grid_XYZW[  105][3] =  0.000409031089717;

    Leb_Grid_XYZW[  106][0] =  0.975017879238525;
    Leb_Grid_XYZW[  106][1] = -0.157067079881828;
    Leb_Grid_XYZW[  106][2] =  0.157067079881829;
    Leb_Grid_XYZW[  106][3] =  0.000409031089717;

    Leb_Grid_XYZW[  107][0] = -0.975017879238525;
    Leb_Grid_XYZW[  107][1] = -0.157067079881828;
    Leb_Grid_XYZW[  107][2] =  0.157067079881829;
    Leb_Grid_XYZW[  107][3] =  0.000409031089717;

    Leb_Grid_XYZW[  108][0] =  0.975017879238525;
    Leb_Grid_XYZW[  108][1] = -0.157067079881828;
    Leb_Grid_XYZW[  108][2] = -0.157067079881829;
    Leb_Grid_XYZW[  108][3] =  0.000409031089717;

    Leb_Grid_XYZW[  109][0] = -0.975017879238525;
    Leb_Grid_XYZW[  109][1] = -0.157067079881828;
    Leb_Grid_XYZW[  109][2] = -0.157067079881829;
    Leb_Grid_XYZW[  109][3] =  0.000409031089717;

    Leb_Grid_XYZW[  110][0] =  0.207116393228251;
    Leb_Grid_XYZW[  110][1] =  0.207116393228251;
    Leb_Grid_XYZW[  110][2] =  0.956140993427350;
    Leb_Grid_XYZW[  110][3] =  0.000449328613417;

    Leb_Grid_XYZW[  111][0] =  0.207116393228252;
    Leb_Grid_XYZW[  111][1] =  0.207116393228252;
    Leb_Grid_XYZW[  111][2] = -0.956140993427350;
    Leb_Grid_XYZW[  111][3] =  0.000449328613417;

    Leb_Grid_XYZW[  112][0] =  0.207116393228251;
    Leb_Grid_XYZW[  112][1] = -0.207116393228251;
    Leb_Grid_XYZW[  112][2] =  0.956140993427350;
    Leb_Grid_XYZW[  112][3] =  0.000449328613417;

    Leb_Grid_XYZW[  113][0] =  0.207116393228252;
    Leb_Grid_XYZW[  113][1] = -0.207116393228252;
    Leb_Grid_XYZW[  113][2] = -0.956140993427350;
    Leb_Grid_XYZW[  113][3] =  0.000449328613417;

    Leb_Grid_XYZW[  114][0] = -0.207116393228251;
    Leb_Grid_XYZW[  114][1] =  0.207116393228251;
    Leb_Grid_XYZW[  114][2] =  0.956140993427350;
    Leb_Grid_XYZW[  114][3] =  0.000449328613417;

    Leb_Grid_XYZW[  115][0] = -0.207116393228252;
    Leb_Grid_XYZW[  115][1] =  0.207116393228252;
    Leb_Grid_XYZW[  115][2] = -0.956140993427350;
    Leb_Grid_XYZW[  115][3] =  0.000449328613417;

    Leb_Grid_XYZW[  116][0] = -0.207116393228251;
    Leb_Grid_XYZW[  116][1] = -0.207116393228251;
    Leb_Grid_XYZW[  116][2] =  0.956140993427350;
    Leb_Grid_XYZW[  116][3] =  0.000449328613417;

    Leb_Grid_XYZW[  117][0] = -0.207116393228252;
    Leb_Grid_XYZW[  117][1] = -0.207116393228252;
    Leb_Grid_XYZW[  117][2] = -0.956140993427350;
    Leb_Grid_XYZW[  117][3] =  0.000449328613417;

    Leb_Grid_XYZW[  118][0] =  0.207116393228251;
    Leb_Grid_XYZW[  118][1] =  0.956140993427350;
    Leb_Grid_XYZW[  118][2] =  0.207116393228251;
    Leb_Grid_XYZW[  118][3] =  0.000449328613417;

    Leb_Grid_XYZW[  119][0] =  0.207116393228251;
    Leb_Grid_XYZW[  119][1] = -0.956140993427350;
    Leb_Grid_XYZW[  119][2] =  0.207116393228251;
    Leb_Grid_XYZW[  119][3] =  0.000449328613417;

    Leb_Grid_XYZW[  120][0] =  0.207116393228251;
    Leb_Grid_XYZW[  120][1] =  0.956140993427350;
    Leb_Grid_XYZW[  120][2] = -0.207116393228251;
    Leb_Grid_XYZW[  120][3] =  0.000449328613417;

    Leb_Grid_XYZW[  121][0] =  0.207116393228251;
    Leb_Grid_XYZW[  121][1] = -0.956140993427350;
    Leb_Grid_XYZW[  121][2] = -0.207116393228251;
    Leb_Grid_XYZW[  121][3] =  0.000449328613417;

    Leb_Grid_XYZW[  122][0] = -0.207116393228251;
    Leb_Grid_XYZW[  122][1] =  0.956140993427350;
    Leb_Grid_XYZW[  122][2] =  0.207116393228251;
    Leb_Grid_XYZW[  122][3] =  0.000449328613417;

    Leb_Grid_XYZW[  123][0] = -0.207116393228251;
    Leb_Grid_XYZW[  123][1] = -0.956140993427350;
    Leb_Grid_XYZW[  123][2] =  0.207116393228251;
    Leb_Grid_XYZW[  123][3] =  0.000449328613417;

    Leb_Grid_XYZW[  124][0] = -0.207116393228251;
    Leb_Grid_XYZW[  124][1] =  0.956140993427350;
    Leb_Grid_XYZW[  124][2] = -0.207116393228251;
    Leb_Grid_XYZW[  124][3] =  0.000449328613417;

    Leb_Grid_XYZW[  125][0] = -0.207116393228251;
    Leb_Grid_XYZW[  125][1] = -0.956140993427350;
    Leb_Grid_XYZW[  125][2] = -0.207116393228251;
    Leb_Grid_XYZW[  125][3] =  0.000449328613417;

    Leb_Grid_XYZW[  126][0] =  0.956140993427350;
    Leb_Grid_XYZW[  126][1] =  0.207116393228251;
    Leb_Grid_XYZW[  126][2] =  0.207116393228251;
    Leb_Grid_XYZW[  126][3] =  0.000449328613417;

    Leb_Grid_XYZW[  127][0] = -0.956140993427350;
    Leb_Grid_XYZW[  127][1] =  0.207116393228251;
    Leb_Grid_XYZW[  127][2] =  0.207116393228251;
    Leb_Grid_XYZW[  127][3] =  0.000449328613417;

    Leb_Grid_XYZW[  128][0] =  0.956140993427350;
    Leb_Grid_XYZW[  128][1] =  0.207116393228251;
    Leb_Grid_XYZW[  128][2] = -0.207116393228251;
    Leb_Grid_XYZW[  128][3] =  0.000449328613417;

    Leb_Grid_XYZW[  129][0] = -0.956140993427350;
    Leb_Grid_XYZW[  129][1] =  0.207116393228251;
    Leb_Grid_XYZW[  129][2] = -0.207116393228251;
    Leb_Grid_XYZW[  129][3] =  0.000449328613417;

    Leb_Grid_XYZW[  130][0] =  0.956140993427350;
    Leb_Grid_XYZW[  130][1] = -0.207116393228251;
    Leb_Grid_XYZW[  130][2] =  0.207116393228251;
    Leb_Grid_XYZW[  130][3] =  0.000449328613417;

    Leb_Grid_XYZW[  131][0] = -0.956140993427350;
    Leb_Grid_XYZW[  131][1] = -0.207116393228251;
    Leb_Grid_XYZW[  131][2] =  0.207116393228251;
    Leb_Grid_XYZW[  131][3] =  0.000449328613417;

    Leb_Grid_XYZW[  132][0] =  0.956140993427350;
    Leb_Grid_XYZW[  132][1] = -0.207116393228251;
    Leb_Grid_XYZW[  132][2] = -0.207116393228251;
    Leb_Grid_XYZW[  132][3] =  0.000449328613417;

    Leb_Grid_XYZW[  133][0] = -0.956140993427350;
    Leb_Grid_XYZW[  133][1] = -0.207116393228251;
    Leb_Grid_XYZW[  133][2] = -0.207116393228251;
    Leb_Grid_XYZW[  133][3] =  0.000449328613417;

    Leb_Grid_XYZW[  134][0] =  0.257891404445085;
    Leb_Grid_XYZW[  134][1] =  0.257891404445085;
    Leb_Grid_XYZW[  134][2] =  0.931119781245509;
    Leb_Grid_XYZW[  134][3] =  0.000479372844796;

    Leb_Grid_XYZW[  135][0] =  0.257891404445085;
    Leb_Grid_XYZW[  135][1] =  0.257891404445085;
    Leb_Grid_XYZW[  135][2] = -0.931119781245508;
    Leb_Grid_XYZW[  135][3] =  0.000479372844796;

    Leb_Grid_XYZW[  136][0] =  0.257891404445085;
    Leb_Grid_XYZW[  136][1] = -0.257891404445085;
    Leb_Grid_XYZW[  136][2] =  0.931119781245509;
    Leb_Grid_XYZW[  136][3] =  0.000479372844796;

    Leb_Grid_XYZW[  137][0] =  0.257891404445085;
    Leb_Grid_XYZW[  137][1] = -0.257891404445085;
    Leb_Grid_XYZW[  137][2] = -0.931119781245508;
    Leb_Grid_XYZW[  137][3] =  0.000479372844796;

    Leb_Grid_XYZW[  138][0] = -0.257891404445085;
    Leb_Grid_XYZW[  138][1] =  0.257891404445085;
    Leb_Grid_XYZW[  138][2] =  0.931119781245509;
    Leb_Grid_XYZW[  138][3] =  0.000479372844796;

    Leb_Grid_XYZW[  139][0] = -0.257891404445085;
    Leb_Grid_XYZW[  139][1] =  0.257891404445085;
    Leb_Grid_XYZW[  139][2] = -0.931119781245508;
    Leb_Grid_XYZW[  139][3] =  0.000479372844796;

    Leb_Grid_XYZW[  140][0] = -0.257891404445085;
    Leb_Grid_XYZW[  140][1] = -0.257891404445085;
    Leb_Grid_XYZW[  140][2] =  0.931119781245509;
    Leb_Grid_XYZW[  140][3] =  0.000479372844796;

    Leb_Grid_XYZW[  141][0] = -0.257891404445085;
    Leb_Grid_XYZW[  141][1] = -0.257891404445085;
    Leb_Grid_XYZW[  141][2] = -0.931119781245508;
    Leb_Grid_XYZW[  141][3] =  0.000479372844796;

    Leb_Grid_XYZW[  142][0] =  0.257891404445084;
    Leb_Grid_XYZW[  142][1] =  0.931119781245509;
    Leb_Grid_XYZW[  142][2] =  0.257891404445084;
    Leb_Grid_XYZW[  142][3] =  0.000479372844796;

    Leb_Grid_XYZW[  143][0] =  0.257891404445084;
    Leb_Grid_XYZW[  143][1] = -0.931119781245509;
    Leb_Grid_XYZW[  143][2] =  0.257891404445084;
    Leb_Grid_XYZW[  143][3] =  0.000479372844796;

    Leb_Grid_XYZW[  144][0] =  0.257891404445084;
    Leb_Grid_XYZW[  144][1] =  0.931119781245509;
    Leb_Grid_XYZW[  144][2] = -0.257891404445085;
    Leb_Grid_XYZW[  144][3] =  0.000479372844796;

    Leb_Grid_XYZW[  145][0] =  0.257891404445084;
    Leb_Grid_XYZW[  145][1] = -0.931119781245509;
    Leb_Grid_XYZW[  145][2] = -0.257891404445085;
    Leb_Grid_XYZW[  145][3] =  0.000479372844796;

    Leb_Grid_XYZW[  146][0] = -0.257891404445084;
    Leb_Grid_XYZW[  146][1] =  0.931119781245509;
    Leb_Grid_XYZW[  146][2] =  0.257891404445084;
    Leb_Grid_XYZW[  146][3] =  0.000479372844796;

    Leb_Grid_XYZW[  147][0] = -0.257891404445084;
    Leb_Grid_XYZW[  147][1] = -0.931119781245509;
    Leb_Grid_XYZW[  147][2] =  0.257891404445084;
    Leb_Grid_XYZW[  147][3] =  0.000479372844796;

    Leb_Grid_XYZW[  148][0] = -0.257891404445084;
    Leb_Grid_XYZW[  148][1] =  0.931119781245509;
    Leb_Grid_XYZW[  148][2] = -0.257891404445085;
    Leb_Grid_XYZW[  148][3] =  0.000479372844796;

    Leb_Grid_XYZW[  149][0] = -0.257891404445084;
    Leb_Grid_XYZW[  149][1] = -0.931119781245509;
    Leb_Grid_XYZW[  149][2] = -0.257891404445085;
    Leb_Grid_XYZW[  149][3] =  0.000479372844796;

    Leb_Grid_XYZW[  150][0] =  0.931119781245509;
    Leb_Grid_XYZW[  150][1] =  0.257891404445084;
    Leb_Grid_XYZW[  150][2] =  0.257891404445084;
    Leb_Grid_XYZW[  150][3] =  0.000479372844796;

    Leb_Grid_XYZW[  151][0] = -0.931119781245509;
    Leb_Grid_XYZW[  151][1] =  0.257891404445084;
    Leb_Grid_XYZW[  151][2] =  0.257891404445084;
    Leb_Grid_XYZW[  151][3] =  0.000479372844796;

    Leb_Grid_XYZW[  152][0] =  0.931119781245509;
    Leb_Grid_XYZW[  152][1] =  0.257891404445084;
    Leb_Grid_XYZW[  152][2] = -0.257891404445085;
    Leb_Grid_XYZW[  152][3] =  0.000479372844796;

    Leb_Grid_XYZW[  153][0] = -0.931119781245509;
    Leb_Grid_XYZW[  153][1] =  0.257891404445084;
    Leb_Grid_XYZW[  153][2] = -0.257891404445085;
    Leb_Grid_XYZW[  153][3] =  0.000479372844796;

    Leb_Grid_XYZW[  154][0] =  0.931119781245509;
    Leb_Grid_XYZW[  154][1] = -0.257891404445084;
    Leb_Grid_XYZW[  154][2] =  0.257891404445084;
    Leb_Grid_XYZW[  154][3] =  0.000479372844796;

    Leb_Grid_XYZW[  155][0] = -0.931119781245509;
    Leb_Grid_XYZW[  155][1] = -0.257891404445084;
    Leb_Grid_XYZW[  155][2] =  0.257891404445084;
    Leb_Grid_XYZW[  155][3] =  0.000479372844796;

    Leb_Grid_XYZW[  156][0] =  0.931119781245509;
    Leb_Grid_XYZW[  156][1] = -0.257891404445084;
    Leb_Grid_XYZW[  156][2] = -0.257891404445085;
    Leb_Grid_XYZW[  156][3] =  0.000479372844796;

    Leb_Grid_XYZW[  157][0] = -0.931119781245509;
    Leb_Grid_XYZW[  157][1] = -0.257891404445084;
    Leb_Grid_XYZW[  157][2] = -0.257891404445085;
    Leb_Grid_XYZW[  157][3] =  0.000479372844796;

    Leb_Grid_XYZW[  158][0] =  0.308568755816962;
    Leb_Grid_XYZW[  158][1] =  0.308568755816962;
    Leb_Grid_XYZW[  158][2] =  0.899761438308591;
    Leb_Grid_XYZW[  158][3] =  0.000501541531916;

    Leb_Grid_XYZW[  159][0] =  0.308568755816962;
    Leb_Grid_XYZW[  159][1] =  0.308568755816962;
    Leb_Grid_XYZW[  159][2] = -0.899761438308591;
    Leb_Grid_XYZW[  159][3] =  0.000501541531916;

    Leb_Grid_XYZW[  160][0] =  0.308568755816962;
    Leb_Grid_XYZW[  160][1] = -0.308568755816962;
    Leb_Grid_XYZW[  160][2] =  0.899761438308591;
    Leb_Grid_XYZW[  160][3] =  0.000501541531916;

    Leb_Grid_XYZW[  161][0] =  0.308568755816962;
    Leb_Grid_XYZW[  161][1] = -0.308568755816962;
    Leb_Grid_XYZW[  161][2] = -0.899761438308591;
    Leb_Grid_XYZW[  161][3] =  0.000501541531916;

    Leb_Grid_XYZW[  162][0] = -0.308568755816962;
    Leb_Grid_XYZW[  162][1] =  0.308568755816962;
    Leb_Grid_XYZW[  162][2] =  0.899761438308591;
    Leb_Grid_XYZW[  162][3] =  0.000501541531916;

    Leb_Grid_XYZW[  163][0] = -0.308568755816962;
    Leb_Grid_XYZW[  163][1] =  0.308568755816962;
    Leb_Grid_XYZW[  163][2] = -0.899761438308591;
    Leb_Grid_XYZW[  163][3] =  0.000501541531916;

    Leb_Grid_XYZW[  164][0] = -0.308568755816962;
    Leb_Grid_XYZW[  164][1] = -0.308568755816962;
    Leb_Grid_XYZW[  164][2] =  0.899761438308591;
    Leb_Grid_XYZW[  164][3] =  0.000501541531916;

    Leb_Grid_XYZW[  165][0] = -0.308568755816962;
    Leb_Grid_XYZW[  165][1] = -0.308568755816962;
    Leb_Grid_XYZW[  165][2] = -0.899761438308591;
    Leb_Grid_XYZW[  165][3] =  0.000501541531916;

    Leb_Grid_XYZW[  166][0] =  0.308568755816962;
    Leb_Grid_XYZW[  166][1] =  0.899761438308591;
    Leb_Grid_XYZW[  166][2] =  0.308568755816963;
    Leb_Grid_XYZW[  166][3] =  0.000501541531916;

    Leb_Grid_XYZW[  167][0] =  0.308568755816962;
    Leb_Grid_XYZW[  167][1] = -0.899761438308591;
    Leb_Grid_XYZW[  167][2] =  0.308568755816963;
    Leb_Grid_XYZW[  167][3] =  0.000501541531916;

    Leb_Grid_XYZW[  168][0] =  0.308568755816962;
    Leb_Grid_XYZW[  168][1] =  0.899761438308591;
    Leb_Grid_XYZW[  168][2] = -0.308568755816962;
    Leb_Grid_XYZW[  168][3] =  0.000501541531916;

    Leb_Grid_XYZW[  169][0] =  0.308568755816962;
    Leb_Grid_XYZW[  169][1] = -0.899761438308591;
    Leb_Grid_XYZW[  169][2] = -0.308568755816962;
    Leb_Grid_XYZW[  169][3] =  0.000501541531916;

    Leb_Grid_XYZW[  170][0] = -0.308568755816962;
    Leb_Grid_XYZW[  170][1] =  0.899761438308591;
    Leb_Grid_XYZW[  170][2] =  0.308568755816963;
    Leb_Grid_XYZW[  170][3] =  0.000501541531916;

    Leb_Grid_XYZW[  171][0] = -0.308568755816962;
    Leb_Grid_XYZW[  171][1] = -0.899761438308591;
    Leb_Grid_XYZW[  171][2] =  0.308568755816963;
    Leb_Grid_XYZW[  171][3] =  0.000501541531916;

    Leb_Grid_XYZW[  172][0] = -0.308568755816962;
    Leb_Grid_XYZW[  172][1] =  0.899761438308591;
    Leb_Grid_XYZW[  172][2] = -0.308568755816962;
    Leb_Grid_XYZW[  172][3] =  0.000501541531916;

    Leb_Grid_XYZW[  173][0] = -0.308568755816962;
    Leb_Grid_XYZW[  173][1] = -0.899761438308591;
    Leb_Grid_XYZW[  173][2] = -0.308568755816962;
    Leb_Grid_XYZW[  173][3] =  0.000501541531916;

    Leb_Grid_XYZW[  174][0] =  0.899761438308591;
    Leb_Grid_XYZW[  174][1] =  0.308568755816962;
    Leb_Grid_XYZW[  174][2] =  0.308568755816963;
    Leb_Grid_XYZW[  174][3] =  0.000501541531916;

    Leb_Grid_XYZW[  175][0] = -0.899761438308591;
    Leb_Grid_XYZW[  175][1] =  0.308568755816962;
    Leb_Grid_XYZW[  175][2] =  0.308568755816963;
    Leb_Grid_XYZW[  175][3] =  0.000501541531916;

    Leb_Grid_XYZW[  176][0] =  0.899761438308591;
    Leb_Grid_XYZW[  176][1] =  0.308568755816962;
    Leb_Grid_XYZW[  176][2] = -0.308568755816962;
    Leb_Grid_XYZW[  176][3] =  0.000501541531916;

    Leb_Grid_XYZW[  177][0] = -0.899761438308591;
    Leb_Grid_XYZW[  177][1] =  0.308568755816962;
    Leb_Grid_XYZW[  177][2] = -0.308568755816962;
    Leb_Grid_XYZW[  177][3] =  0.000501541531916;

    Leb_Grid_XYZW[  178][0] =  0.899761438308591;
    Leb_Grid_XYZW[  178][1] = -0.308568755816962;
    Leb_Grid_XYZW[  178][2] =  0.308568755816963;
    Leb_Grid_XYZW[  178][3] =  0.000501541531916;

    Leb_Grid_XYZW[  179][0] = -0.899761438308591;
    Leb_Grid_XYZW[  179][1] = -0.308568755816962;
    Leb_Grid_XYZW[  179][2] =  0.308568755816963;
    Leb_Grid_XYZW[  179][3] =  0.000501541531916;

    Leb_Grid_XYZW[  180][0] =  0.899761438308591;
    Leb_Grid_XYZW[  180][1] = -0.308568755816962;
    Leb_Grid_XYZW[  180][2] = -0.308568755816962;
    Leb_Grid_XYZW[  180][3] =  0.000501541531916;

    Leb_Grid_XYZW[  181][0] = -0.899761438308591;
    Leb_Grid_XYZW[  181][1] = -0.308568755816962;
    Leb_Grid_XYZW[  181][2] = -0.308568755816962;
    Leb_Grid_XYZW[  181][3] =  0.000501541531916;

    Leb_Grid_XYZW[  182][0] =  0.358471970626702;
    Leb_Grid_XYZW[  182][1] =  0.358471970626702;
    Leb_Grid_XYZW[  182][2] =  0.861971978981926;
    Leb_Grid_XYZW[  182][3] =  0.000517512737268;

    Leb_Grid_XYZW[  183][0] =  0.358471970626702;
    Leb_Grid_XYZW[  183][1] =  0.358471970626702;
    Leb_Grid_XYZW[  183][2] = -0.861971978981926;
    Leb_Grid_XYZW[  183][3] =  0.000517512737268;

    Leb_Grid_XYZW[  184][0] =  0.358471970626702;
    Leb_Grid_XYZW[  184][1] = -0.358471970626702;
    Leb_Grid_XYZW[  184][2] =  0.861971978981926;
    Leb_Grid_XYZW[  184][3] =  0.000517512737268;

    Leb_Grid_XYZW[  185][0] =  0.358471970626702;
    Leb_Grid_XYZW[  185][1] = -0.358471970626702;
    Leb_Grid_XYZW[  185][2] = -0.861971978981926;
    Leb_Grid_XYZW[  185][3] =  0.000517512737268;

    Leb_Grid_XYZW[  186][0] = -0.358471970626702;
    Leb_Grid_XYZW[  186][1] =  0.358471970626702;
    Leb_Grid_XYZW[  186][2] =  0.861971978981926;
    Leb_Grid_XYZW[  186][3] =  0.000517512737268;

    Leb_Grid_XYZW[  187][0] = -0.358471970626702;
    Leb_Grid_XYZW[  187][1] =  0.358471970626702;
    Leb_Grid_XYZW[  187][2] = -0.861971978981926;
    Leb_Grid_XYZW[  187][3] =  0.000517512737268;

    Leb_Grid_XYZW[  188][0] = -0.358471970626702;
    Leb_Grid_XYZW[  188][1] = -0.358471970626702;
    Leb_Grid_XYZW[  188][2] =  0.861971978981926;
    Leb_Grid_XYZW[  188][3] =  0.000517512737268;

    Leb_Grid_XYZW[  189][0] = -0.358471970626702;
    Leb_Grid_XYZW[  189][1] = -0.358471970626702;
    Leb_Grid_XYZW[  189][2] = -0.861971978981926;
    Leb_Grid_XYZW[  189][3] =  0.000517512737268;

    Leb_Grid_XYZW[  190][0] =  0.358471970626702;
    Leb_Grid_XYZW[  190][1] =  0.861971978981926;
    Leb_Grid_XYZW[  190][2] =  0.358471970626703;
    Leb_Grid_XYZW[  190][3] =  0.000517512737268;

    Leb_Grid_XYZW[  191][0] =  0.358471970626702;
    Leb_Grid_XYZW[  191][1] = -0.861971978981926;
    Leb_Grid_XYZW[  191][2] =  0.358471970626703;
    Leb_Grid_XYZW[  191][3] =  0.000517512737268;

    Leb_Grid_XYZW[  192][0] =  0.358471970626702;
    Leb_Grid_XYZW[  192][1] =  0.861971978981926;
    Leb_Grid_XYZW[  192][2] = -0.358471970626702;
    Leb_Grid_XYZW[  192][3] =  0.000517512737268;

    Leb_Grid_XYZW[  193][0] =  0.358471970626702;
    Leb_Grid_XYZW[  193][1] = -0.861971978981926;
    Leb_Grid_XYZW[  193][2] = -0.358471970626702;
    Leb_Grid_XYZW[  193][3] =  0.000517512737268;

    Leb_Grid_XYZW[  194][0] = -0.358471970626702;
    Leb_Grid_XYZW[  194][1] =  0.861971978981926;
    Leb_Grid_XYZW[  194][2] =  0.358471970626703;
    Leb_Grid_XYZW[  194][3] =  0.000517512737268;

    Leb_Grid_XYZW[  195][0] = -0.358471970626702;
    Leb_Grid_XYZW[  195][1] = -0.861971978981926;
    Leb_Grid_XYZW[  195][2] =  0.358471970626703;
    Leb_Grid_XYZW[  195][3] =  0.000517512737268;

    Leb_Grid_XYZW[  196][0] = -0.358471970626702;
    Leb_Grid_XYZW[  196][1] =  0.861971978981926;
    Leb_Grid_XYZW[  196][2] = -0.358471970626702;
    Leb_Grid_XYZW[  196][3] =  0.000517512737268;

    Leb_Grid_XYZW[  197][0] = -0.358471970626702;
    Leb_Grid_XYZW[  197][1] = -0.861971978981926;
    Leb_Grid_XYZW[  197][2] = -0.358471970626702;
    Leb_Grid_XYZW[  197][3] =  0.000517512737268;

    Leb_Grid_XYZW[  198][0] =  0.861971978981926;
    Leb_Grid_XYZW[  198][1] =  0.358471970626702;
    Leb_Grid_XYZW[  198][2] =  0.358471970626703;
    Leb_Grid_XYZW[  198][3] =  0.000517512737268;

    Leb_Grid_XYZW[  199][0] = -0.861971978981926;
    Leb_Grid_XYZW[  199][1] =  0.358471970626702;
    Leb_Grid_XYZW[  199][2] =  0.358471970626703;
    Leb_Grid_XYZW[  199][3] =  0.000517512737268;

    Leb_Grid_XYZW[  200][0] =  0.861971978981926;
    Leb_Grid_XYZW[  200][1] =  0.358471970626703;
    Leb_Grid_XYZW[  200][2] = -0.358471970626702;
    Leb_Grid_XYZW[  200][3] =  0.000517512737268;

    Leb_Grid_XYZW[  201][0] = -0.861971978981926;
    Leb_Grid_XYZW[  201][1] =  0.358471970626703;
    Leb_Grid_XYZW[  201][2] = -0.358471970626702;
    Leb_Grid_XYZW[  201][3] =  0.000517512737268;

    Leb_Grid_XYZW[  202][0] =  0.861971978981926;
    Leb_Grid_XYZW[  202][1] = -0.358471970626702;
    Leb_Grid_XYZW[  202][2] =  0.358471970626703;
    Leb_Grid_XYZW[  202][3] =  0.000517512737268;

    Leb_Grid_XYZW[  203][0] = -0.861971978981926;
    Leb_Grid_XYZW[  203][1] = -0.358471970626702;
    Leb_Grid_XYZW[  203][2] =  0.358471970626703;
    Leb_Grid_XYZW[  203][3] =  0.000517512737268;

    Leb_Grid_XYZW[  204][0] =  0.861971978981926;
    Leb_Grid_XYZW[  204][1] = -0.358471970626703;
    Leb_Grid_XYZW[  204][2] = -0.358471970626702;
    Leb_Grid_XYZW[  204][3] =  0.000517512737268;

    Leb_Grid_XYZW[  205][0] = -0.861971978981926;
    Leb_Grid_XYZW[  205][1] = -0.358471970626703;
    Leb_Grid_XYZW[  205][2] = -0.358471970626702;
    Leb_Grid_XYZW[  205][3] =  0.000517512737268;

    Leb_Grid_XYZW[  206][0] =  0.407013559442871;
    Leb_Grid_XYZW[  206][1] =  0.407013559442871;
    Leb_Grid_XYZW[  206][2] =  0.817728515376154;
    Leb_Grid_XYZW[  206][3] =  0.000528552226208;

    Leb_Grid_XYZW[  207][0] =  0.407013559442871;
    Leb_Grid_XYZW[  207][1] =  0.407013559442871;
    Leb_Grid_XYZW[  207][2] = -0.817728515376154;
    Leb_Grid_XYZW[  207][3] =  0.000528552226208;

    Leb_Grid_XYZW[  208][0] =  0.407013559442871;
    Leb_Grid_XYZW[  208][1] = -0.407013559442871;
    Leb_Grid_XYZW[  208][2] =  0.817728515376154;
    Leb_Grid_XYZW[  208][3] =  0.000528552226208;

    Leb_Grid_XYZW[  209][0] =  0.407013559442871;
    Leb_Grid_XYZW[  209][1] = -0.407013559442871;
    Leb_Grid_XYZW[  209][2] = -0.817728515376154;
    Leb_Grid_XYZW[  209][3] =  0.000528552226208;

    Leb_Grid_XYZW[  210][0] = -0.407013559442871;
    Leb_Grid_XYZW[  210][1] =  0.407013559442871;
    Leb_Grid_XYZW[  210][2] =  0.817728515376154;
    Leb_Grid_XYZW[  210][3] =  0.000528552226208;

    Leb_Grid_XYZW[  211][0] = -0.407013559442871;
    Leb_Grid_XYZW[  211][1] =  0.407013559442871;
    Leb_Grid_XYZW[  211][2] = -0.817728515376154;
    Leb_Grid_XYZW[  211][3] =  0.000528552226208;

    Leb_Grid_XYZW[  212][0] = -0.407013559442871;
    Leb_Grid_XYZW[  212][1] = -0.407013559442871;
    Leb_Grid_XYZW[  212][2] =  0.817728515376154;
    Leb_Grid_XYZW[  212][3] =  0.000528552226208;

    Leb_Grid_XYZW[  213][0] = -0.407013559442871;
    Leb_Grid_XYZW[  213][1] = -0.407013559442871;
    Leb_Grid_XYZW[  213][2] = -0.817728515376154;
    Leb_Grid_XYZW[  213][3] =  0.000528552226208;

    Leb_Grid_XYZW[  214][0] =  0.407013559442871;
    Leb_Grid_XYZW[  214][1] =  0.817728515376154;
    Leb_Grid_XYZW[  214][2] =  0.407013559442871;
    Leb_Grid_XYZW[  214][3] =  0.000528552226208;

    Leb_Grid_XYZW[  215][0] =  0.407013559442871;
    Leb_Grid_XYZW[  215][1] = -0.817728515376154;
    Leb_Grid_XYZW[  215][2] =  0.407013559442871;
    Leb_Grid_XYZW[  215][3] =  0.000528552226208;

    Leb_Grid_XYZW[  216][0] =  0.407013559442871;
    Leb_Grid_XYZW[  216][1] =  0.817728515376154;
    Leb_Grid_XYZW[  216][2] = -0.407013559442871;
    Leb_Grid_XYZW[  216][3] =  0.000528552226208;

    Leb_Grid_XYZW[  217][0] =  0.407013559442871;
    Leb_Grid_XYZW[  217][1] = -0.817728515376154;
    Leb_Grid_XYZW[  217][2] = -0.407013559442871;
    Leb_Grid_XYZW[  217][3] =  0.000528552226208;

    Leb_Grid_XYZW[  218][0] = -0.407013559442871;
    Leb_Grid_XYZW[  218][1] =  0.817728515376154;
    Leb_Grid_XYZW[  218][2] =  0.407013559442871;
    Leb_Grid_XYZW[  218][3] =  0.000528552226208;

    Leb_Grid_XYZW[  219][0] = -0.407013559442871;
    Leb_Grid_XYZW[  219][1] = -0.817728515376154;
    Leb_Grid_XYZW[  219][2] =  0.407013559442871;
    Leb_Grid_XYZW[  219][3] =  0.000528552226208;

    Leb_Grid_XYZW[  220][0] = -0.407013559442871;
    Leb_Grid_XYZW[  220][1] =  0.817728515376154;
    Leb_Grid_XYZW[  220][2] = -0.407013559442871;
    Leb_Grid_XYZW[  220][3] =  0.000528552226208;

    Leb_Grid_XYZW[  221][0] = -0.407013559442871;
    Leb_Grid_XYZW[  221][1] = -0.817728515376154;
    Leb_Grid_XYZW[  221][2] = -0.407013559442871;
    Leb_Grid_XYZW[  221][3] =  0.000528552226208;

    Leb_Grid_XYZW[  222][0] =  0.817728515376154;
    Leb_Grid_XYZW[  222][1] =  0.407013559442871;
    Leb_Grid_XYZW[  222][2] =  0.407013559442871;
    Leb_Grid_XYZW[  222][3] =  0.000528552226208;

    Leb_Grid_XYZW[  223][0] = -0.817728515376154;
    Leb_Grid_XYZW[  223][1] =  0.407013559442871;
    Leb_Grid_XYZW[  223][2] =  0.407013559442871;
    Leb_Grid_XYZW[  223][3] =  0.000528552226208;

    Leb_Grid_XYZW[  224][0] =  0.817728515376154;
    Leb_Grid_XYZW[  224][1] =  0.407013559442871;
    Leb_Grid_XYZW[  224][2] = -0.407013559442871;
    Leb_Grid_XYZW[  224][3] =  0.000528552226208;

    Leb_Grid_XYZW[  225][0] = -0.817728515376154;
    Leb_Grid_XYZW[  225][1] =  0.407013559442871;
    Leb_Grid_XYZW[  225][2] = -0.407013559442871;
    Leb_Grid_XYZW[  225][3] =  0.000528552226208;

    Leb_Grid_XYZW[  226][0] =  0.817728515376154;
    Leb_Grid_XYZW[  226][1] = -0.407013559442871;
    Leb_Grid_XYZW[  226][2] =  0.407013559442871;
    Leb_Grid_XYZW[  226][3] =  0.000528552226208;

    Leb_Grid_XYZW[  227][0] = -0.817728515376154;
    Leb_Grid_XYZW[  227][1] = -0.407013559442871;
    Leb_Grid_XYZW[  227][2] =  0.407013559442871;
    Leb_Grid_XYZW[  227][3] =  0.000528552226208;

    Leb_Grid_XYZW[  228][0] =  0.817728515376154;
    Leb_Grid_XYZW[  228][1] = -0.407013559442871;
    Leb_Grid_XYZW[  228][2] = -0.407013559442871;
    Leb_Grid_XYZW[  228][3] =  0.000528552226208;

    Leb_Grid_XYZW[  229][0] = -0.817728515376154;
    Leb_Grid_XYZW[  229][1] = -0.407013559442871;
    Leb_Grid_XYZW[  229][2] = -0.407013559442871;
    Leb_Grid_XYZW[  229][3] =  0.000528552226208;

    Leb_Grid_XYZW[  230][0] =  0.453661862622264;
    Leb_Grid_XYZW[  230][1] =  0.453661862622264;
    Leb_Grid_XYZW[  230][2] =  0.767060511826933;
    Leb_Grid_XYZW[  230][3] =  0.000535683270371;

    Leb_Grid_XYZW[  231][0] =  0.453661862622264;
    Leb_Grid_XYZW[  231][1] =  0.453661862622264;
    Leb_Grid_XYZW[  231][2] = -0.767060511826933;
    Leb_Grid_XYZW[  231][3] =  0.000535683270371;

    Leb_Grid_XYZW[  232][0] =  0.453661862622264;
    Leb_Grid_XYZW[  232][1] = -0.453661862622264;
    Leb_Grid_XYZW[  232][2] =  0.767060511826933;
    Leb_Grid_XYZW[  232][3] =  0.000535683270371;

    Leb_Grid_XYZW[  233][0] =  0.453661862622264;
    Leb_Grid_XYZW[  233][1] = -0.453661862622264;
    Leb_Grid_XYZW[  233][2] = -0.767060511826933;
    Leb_Grid_XYZW[  233][3] =  0.000535683270371;

    Leb_Grid_XYZW[  234][0] = -0.453661862622264;
    Leb_Grid_XYZW[  234][1] =  0.453661862622264;
    Leb_Grid_XYZW[  234][2] =  0.767060511826933;
    Leb_Grid_XYZW[  234][3] =  0.000535683270371;

    Leb_Grid_XYZW[  235][0] = -0.453661862622264;
    Leb_Grid_XYZW[  235][1] =  0.453661862622264;
    Leb_Grid_XYZW[  235][2] = -0.767060511826933;
    Leb_Grid_XYZW[  235][3] =  0.000535683270371;

    Leb_Grid_XYZW[  236][0] = -0.453661862622264;
    Leb_Grid_XYZW[  236][1] = -0.453661862622264;
    Leb_Grid_XYZW[  236][2] =  0.767060511826933;
    Leb_Grid_XYZW[  236][3] =  0.000535683270371;

    Leb_Grid_XYZW[  237][0] = -0.453661862622264;
    Leb_Grid_XYZW[  237][1] = -0.453661862622264;
    Leb_Grid_XYZW[  237][2] = -0.767060511826933;
    Leb_Grid_XYZW[  237][3] =  0.000535683270371;

    Leb_Grid_XYZW[  238][0] =  0.453661862622264;
    Leb_Grid_XYZW[  238][1] =  0.767060511826933;
    Leb_Grid_XYZW[  238][2] =  0.453661862622264;
    Leb_Grid_XYZW[  238][3] =  0.000535683270371;

    Leb_Grid_XYZW[  239][0] =  0.453661862622264;
    Leb_Grid_XYZW[  239][1] = -0.767060511826933;
    Leb_Grid_XYZW[  239][2] =  0.453661862622264;
    Leb_Grid_XYZW[  239][3] =  0.000535683270371;

    Leb_Grid_XYZW[  240][0] =  0.453661862622264;
    Leb_Grid_XYZW[  240][1] =  0.767060511826933;
    Leb_Grid_XYZW[  240][2] = -0.453661862622264;
    Leb_Grid_XYZW[  240][3] =  0.000535683270371;

    Leb_Grid_XYZW[  241][0] =  0.453661862622264;
    Leb_Grid_XYZW[  241][1] = -0.767060511826933;
    Leb_Grid_XYZW[  241][2] = -0.453661862622264;
    Leb_Grid_XYZW[  241][3] =  0.000535683270371;

    Leb_Grid_XYZW[  242][0] = -0.453661862622264;
    Leb_Grid_XYZW[  242][1] =  0.767060511826933;
    Leb_Grid_XYZW[  242][2] =  0.453661862622264;
    Leb_Grid_XYZW[  242][3] =  0.000535683270371;

    Leb_Grid_XYZW[  243][0] = -0.453661862622264;
    Leb_Grid_XYZW[  243][1] = -0.767060511826933;
    Leb_Grid_XYZW[  243][2] =  0.453661862622264;
    Leb_Grid_XYZW[  243][3] =  0.000535683270371;

    Leb_Grid_XYZW[  244][0] = -0.453661862622264;
    Leb_Grid_XYZW[  244][1] =  0.767060511826933;
    Leb_Grid_XYZW[  244][2] = -0.453661862622264;
    Leb_Grid_XYZW[  244][3] =  0.000535683270371;

    Leb_Grid_XYZW[  245][0] = -0.453661862622264;
    Leb_Grid_XYZW[  245][1] = -0.767060511826933;
    Leb_Grid_XYZW[  245][2] = -0.453661862622264;
    Leb_Grid_XYZW[  245][3] =  0.000535683270371;

    Leb_Grid_XYZW[  246][0] =  0.767060511826933;
    Leb_Grid_XYZW[  246][1] =  0.453661862622264;
    Leb_Grid_XYZW[  246][2] =  0.453661862622264;
    Leb_Grid_XYZW[  246][3] =  0.000535683270371;

    Leb_Grid_XYZW[  247][0] = -0.767060511826933;
    Leb_Grid_XYZW[  247][1] =  0.453661862622264;
    Leb_Grid_XYZW[  247][2] =  0.453661862622264;
    Leb_Grid_XYZW[  247][3] =  0.000535683270371;

    Leb_Grid_XYZW[  248][0] =  0.767060511826933;
    Leb_Grid_XYZW[  248][1] =  0.453661862622264;
    Leb_Grid_XYZW[  248][2] = -0.453661862622264;
    Leb_Grid_XYZW[  248][3] =  0.000535683270371;

    Leb_Grid_XYZW[  249][0] = -0.767060511826933;
    Leb_Grid_XYZW[  249][1] =  0.453661862622264;
    Leb_Grid_XYZW[  249][2] = -0.453661862622264;
    Leb_Grid_XYZW[  249][3] =  0.000535683270371;

    Leb_Grid_XYZW[  250][0] =  0.767060511826933;
    Leb_Grid_XYZW[  250][1] = -0.453661862622264;
    Leb_Grid_XYZW[  250][2] =  0.453661862622264;
    Leb_Grid_XYZW[  250][3] =  0.000535683270371;

    Leb_Grid_XYZW[  251][0] = -0.767060511826933;
    Leb_Grid_XYZW[  251][1] = -0.453661862622264;
    Leb_Grid_XYZW[  251][2] =  0.453661862622264;
    Leb_Grid_XYZW[  251][3] =  0.000535683270371;

    Leb_Grid_XYZW[  252][0] =  0.767060511826933;
    Leb_Grid_XYZW[  252][1] = -0.453661862622264;
    Leb_Grid_XYZW[  252][2] = -0.453661862622264;
    Leb_Grid_XYZW[  252][3] =  0.000535683270371;

    Leb_Grid_XYZW[  253][0] = -0.767060511826933;
    Leb_Grid_XYZW[  253][1] = -0.453661862622264;
    Leb_Grid_XYZW[  253][2] = -0.453661862622264;
    Leb_Grid_XYZW[  253][3] =  0.000535683270371;

    Leb_Grid_XYZW[  254][0] =  0.497919568646358;
    Leb_Grid_XYZW[  254][1] =  0.497919568646358;
    Leb_Grid_XYZW[  254][2] =  0.710036764060883;
    Leb_Grid_XYZW[  254][3] =  0.000539791473618;

    Leb_Grid_XYZW[  255][0] =  0.497919568646358;
    Leb_Grid_XYZW[  255][1] =  0.497919568646358;
    Leb_Grid_XYZW[  255][2] = -0.710036764060883;
    Leb_Grid_XYZW[  255][3] =  0.000539791473618;

    Leb_Grid_XYZW[  256][0] =  0.497919568646358;
    Leb_Grid_XYZW[  256][1] = -0.497919568646358;
    Leb_Grid_XYZW[  256][2] =  0.710036764060883;
    Leb_Grid_XYZW[  256][3] =  0.000539791473618;

    Leb_Grid_XYZW[  257][0] =  0.497919568646358;
    Leb_Grid_XYZW[  257][1] = -0.497919568646358;
    Leb_Grid_XYZW[  257][2] = -0.710036764060883;
    Leb_Grid_XYZW[  257][3] =  0.000539791473618;

    Leb_Grid_XYZW[  258][0] = -0.497919568646358;
    Leb_Grid_XYZW[  258][1] =  0.497919568646358;
    Leb_Grid_XYZW[  258][2] =  0.710036764060883;
    Leb_Grid_XYZW[  258][3] =  0.000539791473618;

    Leb_Grid_XYZW[  259][0] = -0.497919568646358;
    Leb_Grid_XYZW[  259][1] =  0.497919568646358;
    Leb_Grid_XYZW[  259][2] = -0.710036764060883;
    Leb_Grid_XYZW[  259][3] =  0.000539791473618;

    Leb_Grid_XYZW[  260][0] = -0.497919568646358;
    Leb_Grid_XYZW[  260][1] = -0.497919568646358;
    Leb_Grid_XYZW[  260][2] =  0.710036764060883;
    Leb_Grid_XYZW[  260][3] =  0.000539791473618;

    Leb_Grid_XYZW[  261][0] = -0.497919568646358;
    Leb_Grid_XYZW[  261][1] = -0.497919568646358;
    Leb_Grid_XYZW[  261][2] = -0.710036764060883;
    Leb_Grid_XYZW[  261][3] =  0.000539791473618;

    Leb_Grid_XYZW[  262][0] =  0.497919568646358;
    Leb_Grid_XYZW[  262][1] =  0.710036764060883;
    Leb_Grid_XYZW[  262][2] =  0.497919568646358;
    Leb_Grid_XYZW[  262][3] =  0.000539791473618;

    Leb_Grid_XYZW[  263][0] =  0.497919568646358;
    Leb_Grid_XYZW[  263][1] = -0.710036764060883;
    Leb_Grid_XYZW[  263][2] =  0.497919568646358;
    Leb_Grid_XYZW[  263][3] =  0.000539791473618;

    Leb_Grid_XYZW[  264][0] =  0.497919568646358;
    Leb_Grid_XYZW[  264][1] =  0.710036764060883;
    Leb_Grid_XYZW[  264][2] = -0.497919568646358;
    Leb_Grid_XYZW[  264][3] =  0.000539791473618;

    Leb_Grid_XYZW[  265][0] =  0.497919568646358;
    Leb_Grid_XYZW[  265][1] = -0.710036764060883;
    Leb_Grid_XYZW[  265][2] = -0.497919568646358;
    Leb_Grid_XYZW[  265][3] =  0.000539791473618;

    Leb_Grid_XYZW[  266][0] = -0.497919568646358;
    Leb_Grid_XYZW[  266][1] =  0.710036764060883;
    Leb_Grid_XYZW[  266][2] =  0.497919568646358;
    Leb_Grid_XYZW[  266][3] =  0.000539791473618;

    Leb_Grid_XYZW[  267][0] = -0.497919568646358;
    Leb_Grid_XYZW[  267][1] = -0.710036764060883;
    Leb_Grid_XYZW[  267][2] =  0.497919568646358;
    Leb_Grid_XYZW[  267][3] =  0.000539791473618;

    Leb_Grid_XYZW[  268][0] = -0.497919568646358;
    Leb_Grid_XYZW[  268][1] =  0.710036764060883;
    Leb_Grid_XYZW[  268][2] = -0.497919568646358;
    Leb_Grid_XYZW[  268][3] =  0.000539791473618;

    Leb_Grid_XYZW[  269][0] = -0.497919568646358;
    Leb_Grid_XYZW[  269][1] = -0.710036764060883;
    Leb_Grid_XYZW[  269][2] = -0.497919568646358;
    Leb_Grid_XYZW[  269][3] =  0.000539791473618;

    Leb_Grid_XYZW[  270][0] =  0.710036764060883;
    Leb_Grid_XYZW[  270][1] =  0.497919568646358;
    Leb_Grid_XYZW[  270][2] =  0.497919568646358;
    Leb_Grid_XYZW[  270][3] =  0.000539791473618;

    Leb_Grid_XYZW[  271][0] = -0.710036764060883;
    Leb_Grid_XYZW[  271][1] =  0.497919568646358;
    Leb_Grid_XYZW[  271][2] =  0.497919568646358;
    Leb_Grid_XYZW[  271][3] =  0.000539791473618;

    Leb_Grid_XYZW[  272][0] =  0.710036764060883;
    Leb_Grid_XYZW[  272][1] =  0.497919568646358;
    Leb_Grid_XYZW[  272][2] = -0.497919568646358;
    Leb_Grid_XYZW[  272][3] =  0.000539791473618;

    Leb_Grid_XYZW[  273][0] = -0.710036764060883;
    Leb_Grid_XYZW[  273][1] =  0.497919568646358;
    Leb_Grid_XYZW[  273][2] = -0.497919568646358;
    Leb_Grid_XYZW[  273][3] =  0.000539791473618;

    Leb_Grid_XYZW[  274][0] =  0.710036764060883;
    Leb_Grid_XYZW[  274][1] = -0.497919568646358;
    Leb_Grid_XYZW[  274][2] =  0.497919568646358;
    Leb_Grid_XYZW[  274][3] =  0.000539791473618;

    Leb_Grid_XYZW[  275][0] = -0.710036764060883;
    Leb_Grid_XYZW[  275][1] = -0.497919568646358;
    Leb_Grid_XYZW[  275][2] =  0.497919568646358;
    Leb_Grid_XYZW[  275][3] =  0.000539791473618;

    Leb_Grid_XYZW[  276][0] =  0.710036764060883;
    Leb_Grid_XYZW[  276][1] = -0.497919568646358;
    Leb_Grid_XYZW[  276][2] = -0.497919568646358;
    Leb_Grid_XYZW[  276][3] =  0.000539791473618;

    Leb_Grid_XYZW[  277][0] = -0.710036764060883;
    Leb_Grid_XYZW[  277][1] = -0.497919568646358;
    Leb_Grid_XYZW[  277][2] = -0.497919568646358;
    Leb_Grid_XYZW[  277][3] =  0.000539791473618;

    Leb_Grid_XYZW[  278][0] =  0.539307511112700;
    Leb_Grid_XYZW[  278][1] =  0.539307511112700;
    Leb_Grid_XYZW[  278][2] =  0.646757154513849;
    Leb_Grid_XYZW[  278][3] =  0.000541689944160;

    Leb_Grid_XYZW[  279][0] =  0.539307511112700;
    Leb_Grid_XYZW[  279][1] =  0.539307511112700;
    Leb_Grid_XYZW[  279][2] = -0.646757154513848;
    Leb_Grid_XYZW[  279][3] =  0.000541689944160;

    Leb_Grid_XYZW[  280][0] =  0.539307511112700;
    Leb_Grid_XYZW[  280][1] = -0.539307511112700;
    Leb_Grid_XYZW[  280][2] =  0.646757154513849;
    Leb_Grid_XYZW[  280][3] =  0.000541689944160;

    Leb_Grid_XYZW[  281][0] =  0.539307511112700;
    Leb_Grid_XYZW[  281][1] = -0.539307511112700;
    Leb_Grid_XYZW[  281][2] = -0.646757154513848;
    Leb_Grid_XYZW[  281][3] =  0.000541689944160;

    Leb_Grid_XYZW[  282][0] = -0.539307511112700;
    Leb_Grid_XYZW[  282][1] =  0.539307511112700;
    Leb_Grid_XYZW[  282][2] =  0.646757154513849;
    Leb_Grid_XYZW[  282][3] =  0.000541689944160;

    Leb_Grid_XYZW[  283][0] = -0.539307511112700;
    Leb_Grid_XYZW[  283][1] =  0.539307511112700;
    Leb_Grid_XYZW[  283][2] = -0.646757154513848;
    Leb_Grid_XYZW[  283][3] =  0.000541689944160;

    Leb_Grid_XYZW[  284][0] = -0.539307511112700;
    Leb_Grid_XYZW[  284][1] = -0.539307511112700;
    Leb_Grid_XYZW[  284][2] =  0.646757154513849;
    Leb_Grid_XYZW[  284][3] =  0.000541689944160;

    Leb_Grid_XYZW[  285][0] = -0.539307511112700;
    Leb_Grid_XYZW[  285][1] = -0.539307511112700;
    Leb_Grid_XYZW[  285][2] = -0.646757154513848;
    Leb_Grid_XYZW[  285][3] =  0.000541689944160;

    Leb_Grid_XYZW[  286][0] =  0.539307511112700;
    Leb_Grid_XYZW[  286][1] =  0.646757154513849;
    Leb_Grid_XYZW[  286][2] =  0.539307511112700;
    Leb_Grid_XYZW[  286][3] =  0.000541689944160;

    Leb_Grid_XYZW[  287][0] =  0.539307511112700;
    Leb_Grid_XYZW[  287][1] = -0.646757154513849;
    Leb_Grid_XYZW[  287][2] =  0.539307511112700;
    Leb_Grid_XYZW[  287][3] =  0.000541689944160;

    Leb_Grid_XYZW[  288][0] =  0.539307511112700;
    Leb_Grid_XYZW[  288][1] =  0.646757154513849;
    Leb_Grid_XYZW[  288][2] = -0.539307511112700;
    Leb_Grid_XYZW[  288][3] =  0.000541689944160;

    Leb_Grid_XYZW[  289][0] =  0.539307511112700;
    Leb_Grid_XYZW[  289][1] = -0.646757154513849;
    Leb_Grid_XYZW[  289][2] = -0.539307511112700;
    Leb_Grid_XYZW[  289][3] =  0.000541689944160;

    Leb_Grid_XYZW[  290][0] = -0.539307511112700;
    Leb_Grid_XYZW[  290][1] =  0.646757154513849;
    Leb_Grid_XYZW[  290][2] =  0.539307511112700;
    Leb_Grid_XYZW[  290][3] =  0.000541689944160;

    Leb_Grid_XYZW[  291][0] = -0.539307511112700;
    Leb_Grid_XYZW[  291][1] = -0.646757154513849;
    Leb_Grid_XYZW[  291][2] =  0.539307511112700;
    Leb_Grid_XYZW[  291][3] =  0.000541689944160;

    Leb_Grid_XYZW[  292][0] = -0.539307511112700;
    Leb_Grid_XYZW[  292][1] =  0.646757154513849;
    Leb_Grid_XYZW[  292][2] = -0.539307511112700;
    Leb_Grid_XYZW[  292][3] =  0.000541689944160;

    Leb_Grid_XYZW[  293][0] = -0.539307511112700;
    Leb_Grid_XYZW[  293][1] = -0.646757154513849;
    Leb_Grid_XYZW[  293][2] = -0.539307511112700;
    Leb_Grid_XYZW[  293][3] =  0.000541689944160;

    Leb_Grid_XYZW[  294][0] =  0.646757154513849;
    Leb_Grid_XYZW[  294][1] =  0.539307511112700;
    Leb_Grid_XYZW[  294][2] =  0.539307511112700;
    Leb_Grid_XYZW[  294][3] =  0.000541689944160;

    Leb_Grid_XYZW[  295][0] = -0.646757154513849;
    Leb_Grid_XYZW[  295][1] =  0.539307511112700;
    Leb_Grid_XYZW[  295][2] =  0.539307511112700;
    Leb_Grid_XYZW[  295][3] =  0.000541689944160;

    Leb_Grid_XYZW[  296][0] =  0.646757154513849;
    Leb_Grid_XYZW[  296][1] =  0.539307511112700;
    Leb_Grid_XYZW[  296][2] = -0.539307511112700;
    Leb_Grid_XYZW[  296][3] =  0.000541689944160;

    Leb_Grid_XYZW[  297][0] = -0.646757154513849;
    Leb_Grid_XYZW[  297][1] =  0.539307511112700;
    Leb_Grid_XYZW[  297][2] = -0.539307511112700;
    Leb_Grid_XYZW[  297][3] =  0.000541689944160;

    Leb_Grid_XYZW[  298][0] =  0.646757154513849;
    Leb_Grid_XYZW[  298][1] = -0.539307511112700;
    Leb_Grid_XYZW[  298][2] =  0.539307511112700;
    Leb_Grid_XYZW[  298][3] =  0.000541689944160;

    Leb_Grid_XYZW[  299][0] = -0.646757154513849;
    Leb_Grid_XYZW[  299][1] = -0.539307511112700;
    Leb_Grid_XYZW[  299][2] =  0.539307511112700;
    Leb_Grid_XYZW[  299][3] =  0.000541689944160;

    Leb_Grid_XYZW[  300][0] =  0.646757154513849;
    Leb_Grid_XYZW[  300][1] = -0.539307511112700;
    Leb_Grid_XYZW[  300][2] = -0.539307511112700;
    Leb_Grid_XYZW[  300][3] =  0.000541689944160;

    Leb_Grid_XYZW[  301][0] = -0.646757154513849;
    Leb_Grid_XYZW[  301][1] = -0.539307511112700;
    Leb_Grid_XYZW[  301][2] = -0.539307511112700;
    Leb_Grid_XYZW[  301][3] =  0.000541689944160;

    Leb_Grid_XYZW[  302][0] =  0.611561767684392;
    Leb_Grid_XYZW[  302][1] =  0.611561767684392;
    Leb_Grid_XYZW[  302][2] =  0.501980486287549;
    Leb_Grid_XYZW[  302][3] =  0.000541930847689;

    Leb_Grid_XYZW[  303][0] =  0.611561767684392;
    Leb_Grid_XYZW[  303][1] =  0.611561767684392;
    Leb_Grid_XYZW[  303][2] = -0.501980486287549;
    Leb_Grid_XYZW[  303][3] =  0.000541930847689;

    Leb_Grid_XYZW[  304][0] =  0.611561767684392;
    Leb_Grid_XYZW[  304][1] = -0.611561767684392;
    Leb_Grid_XYZW[  304][2] =  0.501980486287549;
    Leb_Grid_XYZW[  304][3] =  0.000541930847689;

    Leb_Grid_XYZW[  305][0] =  0.611561767684392;
    Leb_Grid_XYZW[  305][1] = -0.611561767684392;
    Leb_Grid_XYZW[  305][2] = -0.501980486287549;
    Leb_Grid_XYZW[  305][3] =  0.000541930847689;

    Leb_Grid_XYZW[  306][0] = -0.611561767684392;
    Leb_Grid_XYZW[  306][1] =  0.611561767684392;
    Leb_Grid_XYZW[  306][2] =  0.501980486287549;
    Leb_Grid_XYZW[  306][3] =  0.000541930847689;

    Leb_Grid_XYZW[  307][0] = -0.611561767684392;
    Leb_Grid_XYZW[  307][1] =  0.611561767684392;
    Leb_Grid_XYZW[  307][2] = -0.501980486287549;
    Leb_Grid_XYZW[  307][3] =  0.000541930847689;

    Leb_Grid_XYZW[  308][0] = -0.611561767684392;
    Leb_Grid_XYZW[  308][1] = -0.611561767684392;
    Leb_Grid_XYZW[  308][2] =  0.501980486287549;
    Leb_Grid_XYZW[  308][3] =  0.000541930847689;

    Leb_Grid_XYZW[  309][0] = -0.611561767684392;
    Leb_Grid_XYZW[  309][1] = -0.611561767684392;
    Leb_Grid_XYZW[  309][2] = -0.501980486287549;
    Leb_Grid_XYZW[  309][3] =  0.000541930847689;

    Leb_Grid_XYZW[  310][0] =  0.611561767684392;
    Leb_Grid_XYZW[  310][1] =  0.501980486287549;
    Leb_Grid_XYZW[  310][2] =  0.611561767684392;
    Leb_Grid_XYZW[  310][3] =  0.000541930847689;

    Leb_Grid_XYZW[  311][0] =  0.611561767684392;
    Leb_Grid_XYZW[  311][1] = -0.501980486287549;
    Leb_Grid_XYZW[  311][2] =  0.611561767684392;
    Leb_Grid_XYZW[  311][3] =  0.000541930847689;

    Leb_Grid_XYZW[  312][0] =  0.611561767684392;
    Leb_Grid_XYZW[  312][1] =  0.501980486287549;
    Leb_Grid_XYZW[  312][2] = -0.611561767684391;
    Leb_Grid_XYZW[  312][3] =  0.000541930847689;

    Leb_Grid_XYZW[  313][0] =  0.611561767684392;
    Leb_Grid_XYZW[  313][1] = -0.501980486287549;
    Leb_Grid_XYZW[  313][2] = -0.611561767684391;
    Leb_Grid_XYZW[  313][3] =  0.000541930847689;

    Leb_Grid_XYZW[  314][0] = -0.611561767684392;
    Leb_Grid_XYZW[  314][1] =  0.501980486287549;
    Leb_Grid_XYZW[  314][2] =  0.611561767684392;
    Leb_Grid_XYZW[  314][3] =  0.000541930847689;

    Leb_Grid_XYZW[  315][0] = -0.611561767684392;
    Leb_Grid_XYZW[  315][1] = -0.501980486287549;
    Leb_Grid_XYZW[  315][2] =  0.611561767684392;
    Leb_Grid_XYZW[  315][3] =  0.000541930847689;

    Leb_Grid_XYZW[  316][0] = -0.611561767684392;
    Leb_Grid_XYZW[  316][1] =  0.501980486287549;
    Leb_Grid_XYZW[  316][2] = -0.611561767684391;
    Leb_Grid_XYZW[  316][3] =  0.000541930847689;

    Leb_Grid_XYZW[  317][0] = -0.611561767684392;
    Leb_Grid_XYZW[  317][1] = -0.501980486287549;
    Leb_Grid_XYZW[  317][2] = -0.611561767684391;
    Leb_Grid_XYZW[  317][3] =  0.000541930847689;

    Leb_Grid_XYZW[  318][0] =  0.501980486287549;
    Leb_Grid_XYZW[  318][1] =  0.611561767684392;
    Leb_Grid_XYZW[  318][2] =  0.611561767684392;
    Leb_Grid_XYZW[  318][3] =  0.000541930847689;

    Leb_Grid_XYZW[  319][0] = -0.501980486287549;
    Leb_Grid_XYZW[  319][1] =  0.611561767684392;
    Leb_Grid_XYZW[  319][2] =  0.611561767684392;
    Leb_Grid_XYZW[  319][3] =  0.000541930847689;

    Leb_Grid_XYZW[  320][0] =  0.501980486287549;
    Leb_Grid_XYZW[  320][1] =  0.611561767684392;
    Leb_Grid_XYZW[  320][2] = -0.611561767684391;
    Leb_Grid_XYZW[  320][3] =  0.000541930847689;

    Leb_Grid_XYZW[  321][0] = -0.501980486287549;
    Leb_Grid_XYZW[  321][1] =  0.611561767684392;
    Leb_Grid_XYZW[  321][2] = -0.611561767684391;
    Leb_Grid_XYZW[  321][3] =  0.000541930847689;

    Leb_Grid_XYZW[  322][0] =  0.501980486287549;
    Leb_Grid_XYZW[  322][1] = -0.611561767684392;
    Leb_Grid_XYZW[  322][2] =  0.611561767684392;
    Leb_Grid_XYZW[  322][3] =  0.000541930847689;

    Leb_Grid_XYZW[  323][0] = -0.501980486287549;
    Leb_Grid_XYZW[  323][1] = -0.611561767684392;
    Leb_Grid_XYZW[  323][2] =  0.611561767684392;
    Leb_Grid_XYZW[  323][3] =  0.000541930847689;

    Leb_Grid_XYZW[  324][0] =  0.501980486287549;
    Leb_Grid_XYZW[  324][1] = -0.611561767684392;
    Leb_Grid_XYZW[  324][2] = -0.611561767684391;
    Leb_Grid_XYZW[  324][3] =  0.000541930847689;

    Leb_Grid_XYZW[  325][0] = -0.501980486287549;
    Leb_Grid_XYZW[  325][1] = -0.611561767684392;
    Leb_Grid_XYZW[  325][2] = -0.611561767684391;
    Leb_Grid_XYZW[  325][3] =  0.000541930847689;

    Leb_Grid_XYZW[  326][0] =  0.641430843516016;
    Leb_Grid_XYZW[  326][1] =  0.641430843516016;
    Leb_Grid_XYZW[  326][2] =  0.420871650236346;
    Leb_Grid_XYZW[  326][3] =  0.000541693690203;

    Leb_Grid_XYZW[  327][0] =  0.641430843516016;
    Leb_Grid_XYZW[  327][1] =  0.641430843516016;
    Leb_Grid_XYZW[  327][2] = -0.420871650236345;
    Leb_Grid_XYZW[  327][3] =  0.000541693690203;

    Leb_Grid_XYZW[  328][0] =  0.641430843516016;
    Leb_Grid_XYZW[  328][1] = -0.641430843516016;
    Leb_Grid_XYZW[  328][2] =  0.420871650236346;
    Leb_Grid_XYZW[  328][3] =  0.000541693690203;

    Leb_Grid_XYZW[  329][0] =  0.641430843516016;
    Leb_Grid_XYZW[  329][1] = -0.641430843516016;
    Leb_Grid_XYZW[  329][2] = -0.420871650236345;
    Leb_Grid_XYZW[  329][3] =  0.000541693690203;

    Leb_Grid_XYZW[  330][0] = -0.641430843516016;
    Leb_Grid_XYZW[  330][1] =  0.641430843516016;
    Leb_Grid_XYZW[  330][2] =  0.420871650236346;
    Leb_Grid_XYZW[  330][3] =  0.000541693690203;

    Leb_Grid_XYZW[  331][0] = -0.641430843516016;
    Leb_Grid_XYZW[  331][1] =  0.641430843516016;
    Leb_Grid_XYZW[  331][2] = -0.420871650236345;
    Leb_Grid_XYZW[  331][3] =  0.000541693690203;

    Leb_Grid_XYZW[  332][0] = -0.641430843516016;
    Leb_Grid_XYZW[  332][1] = -0.641430843516016;
    Leb_Grid_XYZW[  332][2] =  0.420871650236346;
    Leb_Grid_XYZW[  332][3] =  0.000541693690203;

    Leb_Grid_XYZW[  333][0] = -0.641430843516016;
    Leb_Grid_XYZW[  333][1] = -0.641430843516016;
    Leb_Grid_XYZW[  333][2] = -0.420871650236345;
    Leb_Grid_XYZW[  333][3] =  0.000541693690203;

    Leb_Grid_XYZW[  334][0] =  0.641430843516016;
    Leb_Grid_XYZW[  334][1] =  0.420871650236346;
    Leb_Grid_XYZW[  334][2] =  0.641430843516016;
    Leb_Grid_XYZW[  334][3] =  0.000541693690203;

    Leb_Grid_XYZW[  335][0] =  0.641430843516016;
    Leb_Grid_XYZW[  335][1] = -0.420871650236346;
    Leb_Grid_XYZW[  335][2] =  0.641430843516016;
    Leb_Grid_XYZW[  335][3] =  0.000541693690203;

    Leb_Grid_XYZW[  336][0] =  0.641430843516016;
    Leb_Grid_XYZW[  336][1] =  0.420871650236346;
    Leb_Grid_XYZW[  336][2] = -0.641430843516016;
    Leb_Grid_XYZW[  336][3] =  0.000541693690203;

    Leb_Grid_XYZW[  337][0] =  0.641430843516016;
    Leb_Grid_XYZW[  337][1] = -0.420871650236346;
    Leb_Grid_XYZW[  337][2] = -0.641430843516016;
    Leb_Grid_XYZW[  337][3] =  0.000541693690203;

    Leb_Grid_XYZW[  338][0] = -0.641430843516016;
    Leb_Grid_XYZW[  338][1] =  0.420871650236346;
    Leb_Grid_XYZW[  338][2] =  0.641430843516016;
    Leb_Grid_XYZW[  338][3] =  0.000541693690203;

    Leb_Grid_XYZW[  339][0] = -0.641430843516016;
    Leb_Grid_XYZW[  339][1] = -0.420871650236346;
    Leb_Grid_XYZW[  339][2] =  0.641430843516016;
    Leb_Grid_XYZW[  339][3] =  0.000541693690203;

    Leb_Grid_XYZW[  340][0] = -0.641430843516016;
    Leb_Grid_XYZW[  340][1] =  0.420871650236346;
    Leb_Grid_XYZW[  340][2] = -0.641430843516016;
    Leb_Grid_XYZW[  340][3] =  0.000541693690203;

    Leb_Grid_XYZW[  341][0] = -0.641430843516016;
    Leb_Grid_XYZW[  341][1] = -0.420871650236346;
    Leb_Grid_XYZW[  341][2] = -0.641430843516016;
    Leb_Grid_XYZW[  341][3] =  0.000541693690203;

    Leb_Grid_XYZW[  342][0] =  0.420871650236345;
    Leb_Grid_XYZW[  342][1] =  0.641430843516016;
    Leb_Grid_XYZW[  342][2] =  0.641430843516016;
    Leb_Grid_XYZW[  342][3] =  0.000541693690203;

    Leb_Grid_XYZW[  343][0] = -0.420871650236345;
    Leb_Grid_XYZW[  343][1] =  0.641430843516016;
    Leb_Grid_XYZW[  343][2] =  0.641430843516016;
    Leb_Grid_XYZW[  343][3] =  0.000541693690203;

    Leb_Grid_XYZW[  344][0] =  0.420871650236345;
    Leb_Grid_XYZW[  344][1] =  0.641430843516016;
    Leb_Grid_XYZW[  344][2] = -0.641430843516016;
    Leb_Grid_XYZW[  344][3] =  0.000541693690203;

    Leb_Grid_XYZW[  345][0] = -0.420871650236345;
    Leb_Grid_XYZW[  345][1] =  0.641430843516016;
    Leb_Grid_XYZW[  345][2] = -0.641430843516016;
    Leb_Grid_XYZW[  345][3] =  0.000541693690203;

    Leb_Grid_XYZW[  346][0] =  0.420871650236345;
    Leb_Grid_XYZW[  346][1] = -0.641430843516016;
    Leb_Grid_XYZW[  346][2] =  0.641430843516016;
    Leb_Grid_XYZW[  346][3] =  0.000541693690203;

    Leb_Grid_XYZW[  347][0] = -0.420871650236345;
    Leb_Grid_XYZW[  347][1] = -0.641430843516016;
    Leb_Grid_XYZW[  347][2] =  0.641430843516016;
    Leb_Grid_XYZW[  347][3] =  0.000541693690203;

    Leb_Grid_XYZW[  348][0] =  0.420871650236345;
    Leb_Grid_XYZW[  348][1] = -0.641430843516016;
    Leb_Grid_XYZW[  348][2] = -0.641430843516016;
    Leb_Grid_XYZW[  348][3] =  0.000541693690203;

    Leb_Grid_XYZW[  349][0] = -0.420871650236345;
    Leb_Grid_XYZW[  349][1] = -0.641430843516016;
    Leb_Grid_XYZW[  349][2] = -0.641430843516016;
    Leb_Grid_XYZW[  349][3] =  0.000541693690203;

    Leb_Grid_XYZW[  350][0] =  0.666409941272161;
    Leb_Grid_XYZW[  350][1] =  0.666409941272161;
    Leb_Grid_XYZW[  350][2] =  0.334358460857910;
    Leb_Grid_XYZW[  350][3] =  0.000541954433870;

    Leb_Grid_XYZW[  351][0] =  0.666409941272161;
    Leb_Grid_XYZW[  351][1] =  0.666409941272161;
    Leb_Grid_XYZW[  351][2] = -0.334358460857910;
    Leb_Grid_XYZW[  351][3] =  0.000541954433870;

    Leb_Grid_XYZW[  352][0] =  0.666409941272161;
    Leb_Grid_XYZW[  352][1] = -0.666409941272161;
    Leb_Grid_XYZW[  352][2] =  0.334358460857910;
    Leb_Grid_XYZW[  352][3] =  0.000541954433870;

    Leb_Grid_XYZW[  353][0] =  0.666409941272161;
    Leb_Grid_XYZW[  353][1] = -0.666409941272161;
    Leb_Grid_XYZW[  353][2] = -0.334358460857910;
    Leb_Grid_XYZW[  353][3] =  0.000541954433870;

    Leb_Grid_XYZW[  354][0] = -0.666409941272161;
    Leb_Grid_XYZW[  354][1] =  0.666409941272161;
    Leb_Grid_XYZW[  354][2] =  0.334358460857910;
    Leb_Grid_XYZW[  354][3] =  0.000541954433870;

    Leb_Grid_XYZW[  355][0] = -0.666409941272161;
    Leb_Grid_XYZW[  355][1] =  0.666409941272161;
    Leb_Grid_XYZW[  355][2] = -0.334358460857910;
    Leb_Grid_XYZW[  355][3] =  0.000541954433870;

    Leb_Grid_XYZW[  356][0] = -0.666409941272161;
    Leb_Grid_XYZW[  356][1] = -0.666409941272161;
    Leb_Grid_XYZW[  356][2] =  0.334358460857910;
    Leb_Grid_XYZW[  356][3] =  0.000541954433870;

    Leb_Grid_XYZW[  357][0] = -0.666409941272161;
    Leb_Grid_XYZW[  357][1] = -0.666409941272161;
    Leb_Grid_XYZW[  357][2] = -0.334358460857910;
    Leb_Grid_XYZW[  357][3] =  0.000541954433870;

    Leb_Grid_XYZW[  358][0] =  0.666409941272161;
    Leb_Grid_XYZW[  358][1] =  0.334358460857910;
    Leb_Grid_XYZW[  358][2] =  0.666409941272161;
    Leb_Grid_XYZW[  358][3] =  0.000541954433870;

    Leb_Grid_XYZW[  359][0] =  0.666409941272161;
    Leb_Grid_XYZW[  359][1] = -0.334358460857910;
    Leb_Grid_XYZW[  359][2] =  0.666409941272161;
    Leb_Grid_XYZW[  359][3] =  0.000541954433870;

    Leb_Grid_XYZW[  360][0] =  0.666409941272161;
    Leb_Grid_XYZW[  360][1] =  0.334358460857910;
    Leb_Grid_XYZW[  360][2] = -0.666409941272161;
    Leb_Grid_XYZW[  360][3] =  0.000541954433870;

    Leb_Grid_XYZW[  361][0] =  0.666409941272161;
    Leb_Grid_XYZW[  361][1] = -0.334358460857910;
    Leb_Grid_XYZW[  361][2] = -0.666409941272161;
    Leb_Grid_XYZW[  361][3] =  0.000541954433870;

    Leb_Grid_XYZW[  362][0] = -0.666409941272161;
    Leb_Grid_XYZW[  362][1] =  0.334358460857910;
    Leb_Grid_XYZW[  362][2] =  0.666409941272161;
    Leb_Grid_XYZW[  362][3] =  0.000541954433870;

    Leb_Grid_XYZW[  363][0] = -0.666409941272161;
    Leb_Grid_XYZW[  363][1] = -0.334358460857910;
    Leb_Grid_XYZW[  363][2] =  0.666409941272161;
    Leb_Grid_XYZW[  363][3] =  0.000541954433870;

    Leb_Grid_XYZW[  364][0] = -0.666409941272161;
    Leb_Grid_XYZW[  364][1] =  0.334358460857910;
    Leb_Grid_XYZW[  364][2] = -0.666409941272161;
    Leb_Grid_XYZW[  364][3] =  0.000541954433870;

    Leb_Grid_XYZW[  365][0] = -0.666409941272161;
    Leb_Grid_XYZW[  365][1] = -0.334358460857910;
    Leb_Grid_XYZW[  365][2] = -0.666409941272161;
    Leb_Grid_XYZW[  365][3] =  0.000541954433870;

    Leb_Grid_XYZW[  366][0] =  0.334358460857910;
    Leb_Grid_XYZW[  366][1] =  0.666409941272161;
    Leb_Grid_XYZW[  366][2] =  0.666409941272161;
    Leb_Grid_XYZW[  366][3] =  0.000541954433870;

    Leb_Grid_XYZW[  367][0] = -0.334358460857910;
    Leb_Grid_XYZW[  367][1] =  0.666409941272161;
    Leb_Grid_XYZW[  367][2] =  0.666409941272161;
    Leb_Grid_XYZW[  367][3] =  0.000541954433870;

    Leb_Grid_XYZW[  368][0] =  0.334358460857910;
    Leb_Grid_XYZW[  368][1] =  0.666409941272161;
    Leb_Grid_XYZW[  368][2] = -0.666409941272161;
    Leb_Grid_XYZW[  368][3] =  0.000541954433870;

    Leb_Grid_XYZW[  369][0] = -0.334358460857910;
    Leb_Grid_XYZW[  369][1] =  0.666409941272161;
    Leb_Grid_XYZW[  369][2] = -0.666409941272161;
    Leb_Grid_XYZW[  369][3] =  0.000541954433870;

    Leb_Grid_XYZW[  370][0] =  0.334358460857910;
    Leb_Grid_XYZW[  370][1] = -0.666409941272161;
    Leb_Grid_XYZW[  370][2] =  0.666409941272161;
    Leb_Grid_XYZW[  370][3] =  0.000541954433870;

    Leb_Grid_XYZW[  371][0] = -0.334358460857910;
    Leb_Grid_XYZW[  371][1] = -0.666409941272161;
    Leb_Grid_XYZW[  371][2] =  0.666409941272161;
    Leb_Grid_XYZW[  371][3] =  0.000541954433870;

    Leb_Grid_XYZW[  372][0] =  0.334358460857910;
    Leb_Grid_XYZW[  372][1] = -0.666409941272161;
    Leb_Grid_XYZW[  372][2] = -0.666409941272161;
    Leb_Grid_XYZW[  372][3] =  0.000541954433870;

    Leb_Grid_XYZW[  373][0] = -0.334358460857910;
    Leb_Grid_XYZW[  373][1] = -0.666409941272161;
    Leb_Grid_XYZW[  373][2] = -0.666409941272161;
    Leb_Grid_XYZW[  373][3] =  0.000541954433870;

    Leb_Grid_XYZW[  374][0] =  0.685916177121491;
    Leb_Grid_XYZW[  374][1] =  0.685916177121491;
    Leb_Grid_XYZW[  374][2] =  0.242977356817622;
    Leb_Grid_XYZW[  374][3] =  0.000542898365663;

    Leb_Grid_XYZW[  375][0] =  0.685916177121491;
    Leb_Grid_XYZW[  375][1] =  0.685916177121491;
    Leb_Grid_XYZW[  375][2] = -0.242977356817622;
    Leb_Grid_XYZW[  375][3] =  0.000542898365663;

    Leb_Grid_XYZW[  376][0] =  0.685916177121491;
    Leb_Grid_XYZW[  376][1] = -0.685916177121491;
    Leb_Grid_XYZW[  376][2] =  0.242977356817622;
    Leb_Grid_XYZW[  376][3] =  0.000542898365663;

    Leb_Grid_XYZW[  377][0] =  0.685916177121491;
    Leb_Grid_XYZW[  377][1] = -0.685916177121491;
    Leb_Grid_XYZW[  377][2] = -0.242977356817622;
    Leb_Grid_XYZW[  377][3] =  0.000542898365663;

    Leb_Grid_XYZW[  378][0] = -0.685916177121491;
    Leb_Grid_XYZW[  378][1] =  0.685916177121491;
    Leb_Grid_XYZW[  378][2] =  0.242977356817622;
    Leb_Grid_XYZW[  378][3] =  0.000542898365663;

    Leb_Grid_XYZW[  379][0] = -0.685916177121491;
    Leb_Grid_XYZW[  379][1] =  0.685916177121491;
    Leb_Grid_XYZW[  379][2] = -0.242977356817622;
    Leb_Grid_XYZW[  379][3] =  0.000542898365663;

    Leb_Grid_XYZW[  380][0] = -0.685916177121491;
    Leb_Grid_XYZW[  380][1] = -0.685916177121491;
    Leb_Grid_XYZW[  380][2] =  0.242977356817622;
    Leb_Grid_XYZW[  380][3] =  0.000542898365663;

    Leb_Grid_XYZW[  381][0] = -0.685916177121491;
    Leb_Grid_XYZW[  381][1] = -0.685916177121491;
    Leb_Grid_XYZW[  381][2] = -0.242977356817622;
    Leb_Grid_XYZW[  381][3] =  0.000542898365663;

    Leb_Grid_XYZW[  382][0] =  0.685916177121491;
    Leb_Grid_XYZW[  382][1] =  0.242977356817622;
    Leb_Grid_XYZW[  382][2] =  0.685916177121491;
    Leb_Grid_XYZW[  382][3] =  0.000542898365663;

    Leb_Grid_XYZW[  383][0] =  0.685916177121491;
    Leb_Grid_XYZW[  383][1] = -0.242977356817622;
    Leb_Grid_XYZW[  383][2] =  0.685916177121491;
    Leb_Grid_XYZW[  383][3] =  0.000542898365663;

    Leb_Grid_XYZW[  384][0] =  0.685916177121491;
    Leb_Grid_XYZW[  384][1] =  0.242977356817621;
    Leb_Grid_XYZW[  384][2] = -0.685916177121491;
    Leb_Grid_XYZW[  384][3] =  0.000542898365663;

    Leb_Grid_XYZW[  385][0] =  0.685916177121491;
    Leb_Grid_XYZW[  385][1] = -0.242977356817621;
    Leb_Grid_XYZW[  385][2] = -0.685916177121491;
    Leb_Grid_XYZW[  385][3] =  0.000542898365663;

    Leb_Grid_XYZW[  386][0] = -0.685916177121491;
    Leb_Grid_XYZW[  386][1] =  0.242977356817622;
    Leb_Grid_XYZW[  386][2] =  0.685916177121491;
    Leb_Grid_XYZW[  386][3] =  0.000542898365663;

    Leb_Grid_XYZW[  387][0] = -0.685916177121491;
    Leb_Grid_XYZW[  387][1] = -0.242977356817622;
    Leb_Grid_XYZW[  387][2] =  0.685916177121491;
    Leb_Grid_XYZW[  387][3] =  0.000542898365663;

    Leb_Grid_XYZW[  388][0] = -0.685916177121491;
    Leb_Grid_XYZW[  388][1] =  0.242977356817621;
    Leb_Grid_XYZW[  388][2] = -0.685916177121491;
    Leb_Grid_XYZW[  388][3] =  0.000542898365663;

    Leb_Grid_XYZW[  389][0] = -0.685916177121491;
    Leb_Grid_XYZW[  389][1] = -0.242977356817621;
    Leb_Grid_XYZW[  389][2] = -0.685916177121491;
    Leb_Grid_XYZW[  389][3] =  0.000542898365663;

    Leb_Grid_XYZW[  390][0] =  0.242977356817622;
    Leb_Grid_XYZW[  390][1] =  0.685916177121491;
    Leb_Grid_XYZW[  390][2] =  0.685916177121491;
    Leb_Grid_XYZW[  390][3] =  0.000542898365663;

    Leb_Grid_XYZW[  391][0] = -0.242977356817622;
    Leb_Grid_XYZW[  391][1] =  0.685916177121491;
    Leb_Grid_XYZW[  391][2] =  0.685916177121491;
    Leb_Grid_XYZW[  391][3] =  0.000542898365663;

    Leb_Grid_XYZW[  392][0] =  0.242977356817622;
    Leb_Grid_XYZW[  392][1] =  0.685916177121491;
    Leb_Grid_XYZW[  392][2] = -0.685916177121491;
    Leb_Grid_XYZW[  392][3] =  0.000542898365663;

    Leb_Grid_XYZW[  393][0] = -0.242977356817622;
    Leb_Grid_XYZW[  393][1] =  0.685916177121491;
    Leb_Grid_XYZW[  393][2] = -0.685916177121491;
    Leb_Grid_XYZW[  393][3] =  0.000542898365663;

    Leb_Grid_XYZW[  394][0] =  0.242977356817622;
    Leb_Grid_XYZW[  394][1] = -0.685916177121491;
    Leb_Grid_XYZW[  394][2] =  0.685916177121491;
    Leb_Grid_XYZW[  394][3] =  0.000542898365663;

    Leb_Grid_XYZW[  395][0] = -0.242977356817622;
    Leb_Grid_XYZW[  395][1] = -0.685916177121491;
    Leb_Grid_XYZW[  395][2] =  0.685916177121491;
    Leb_Grid_XYZW[  395][3] =  0.000542898365663;

    Leb_Grid_XYZW[  396][0] =  0.242977356817622;
    Leb_Grid_XYZW[  396][1] = -0.685916177121491;
    Leb_Grid_XYZW[  396][2] = -0.685916177121491;
    Leb_Grid_XYZW[  396][3] =  0.000542898365663;

    Leb_Grid_XYZW[  397][0] = -0.242977356817622;
    Leb_Grid_XYZW[  397][1] = -0.685916177121491;
    Leb_Grid_XYZW[  397][2] = -0.685916177121491;
    Leb_Grid_XYZW[  397][3] =  0.000542898365663;

    Leb_Grid_XYZW[  398][0] =  0.699362559350389;
    Leb_Grid_XYZW[  398][1] =  0.699362559350389;
    Leb_Grid_XYZW[  398][2] =  0.147594109495424;
    Leb_Grid_XYZW[  398][3] =  0.000544228650010;

    Leb_Grid_XYZW[  399][0] =  0.699362559350389;
    Leb_Grid_XYZW[  399][1] =  0.699362559350389;
    Leb_Grid_XYZW[  399][2] = -0.147594109495423;
    Leb_Grid_XYZW[  399][3] =  0.000544228650010;

    Leb_Grid_XYZW[  400][0] =  0.699362559350389;
    Leb_Grid_XYZW[  400][1] = -0.699362559350389;
    Leb_Grid_XYZW[  400][2] =  0.147594109495424;
    Leb_Grid_XYZW[  400][3] =  0.000544228650010;

    Leb_Grid_XYZW[  401][0] =  0.699362559350389;
    Leb_Grid_XYZW[  401][1] = -0.699362559350389;
    Leb_Grid_XYZW[  401][2] = -0.147594109495423;
    Leb_Grid_XYZW[  401][3] =  0.000544228650010;

    Leb_Grid_XYZW[  402][0] = -0.699362559350389;
    Leb_Grid_XYZW[  402][1] =  0.699362559350389;
    Leb_Grid_XYZW[  402][2] =  0.147594109495424;
    Leb_Grid_XYZW[  402][3] =  0.000544228650010;

    Leb_Grid_XYZW[  403][0] = -0.699362559350389;
    Leb_Grid_XYZW[  403][1] =  0.699362559350389;
    Leb_Grid_XYZW[  403][2] = -0.147594109495423;
    Leb_Grid_XYZW[  403][3] =  0.000544228650010;

    Leb_Grid_XYZW[  404][0] = -0.699362559350389;
    Leb_Grid_XYZW[  404][1] = -0.699362559350389;
    Leb_Grid_XYZW[  404][2] =  0.147594109495424;
    Leb_Grid_XYZW[  404][3] =  0.000544228650010;

    Leb_Grid_XYZW[  405][0] = -0.699362559350389;
    Leb_Grid_XYZW[  405][1] = -0.699362559350389;
    Leb_Grid_XYZW[  405][2] = -0.147594109495423;
    Leb_Grid_XYZW[  405][3] =  0.000544228650010;

    Leb_Grid_XYZW[  406][0] =  0.699362559350389;
    Leb_Grid_XYZW[  406][1] =  0.147594109495424;
    Leb_Grid_XYZW[  406][2] =  0.699362559350389;
    Leb_Grid_XYZW[  406][3] =  0.000544228650010;

    Leb_Grid_XYZW[  407][0] =  0.699362559350389;
    Leb_Grid_XYZW[  407][1] = -0.147594109495424;
    Leb_Grid_XYZW[  407][2] =  0.699362559350389;
    Leb_Grid_XYZW[  407][3] =  0.000544228650010;

    Leb_Grid_XYZW[  408][0] =  0.699362559350389;
    Leb_Grid_XYZW[  408][1] =  0.147594109495424;
    Leb_Grid_XYZW[  408][2] = -0.699362559350389;
    Leb_Grid_XYZW[  408][3] =  0.000544228650010;

    Leb_Grid_XYZW[  409][0] =  0.699362559350389;
    Leb_Grid_XYZW[  409][1] = -0.147594109495424;
    Leb_Grid_XYZW[  409][2] = -0.699362559350389;
    Leb_Grid_XYZW[  409][3] =  0.000544228650010;

    Leb_Grid_XYZW[  410][0] = -0.699362559350389;
    Leb_Grid_XYZW[  410][1] =  0.147594109495424;
    Leb_Grid_XYZW[  410][2] =  0.699362559350389;
    Leb_Grid_XYZW[  410][3] =  0.000544228650010;

    Leb_Grid_XYZW[  411][0] = -0.699362559350389;
    Leb_Grid_XYZW[  411][1] = -0.147594109495424;
    Leb_Grid_XYZW[  411][2] =  0.699362559350389;
    Leb_Grid_XYZW[  411][3] =  0.000544228650010;

    Leb_Grid_XYZW[  412][0] = -0.699362559350389;
    Leb_Grid_XYZW[  412][1] =  0.147594109495424;
    Leb_Grid_XYZW[  412][2] = -0.699362559350389;
    Leb_Grid_XYZW[  412][3] =  0.000544228650010;

    Leb_Grid_XYZW[  413][0] = -0.699362559350389;
    Leb_Grid_XYZW[  413][1] = -0.147594109495424;
    Leb_Grid_XYZW[  413][2] = -0.699362559350389;
    Leb_Grid_XYZW[  413][3] =  0.000544228650010;

    Leb_Grid_XYZW[  414][0] =  0.147594109495424;
    Leb_Grid_XYZW[  414][1] =  0.699362559350389;
    Leb_Grid_XYZW[  414][2] =  0.699362559350389;
    Leb_Grid_XYZW[  414][3] =  0.000544228650010;

    Leb_Grid_XYZW[  415][0] = -0.147594109495423;
    Leb_Grid_XYZW[  415][1] =  0.699362559350389;
    Leb_Grid_XYZW[  415][2] =  0.699362559350389;
    Leb_Grid_XYZW[  415][3] =  0.000544228650010;

    Leb_Grid_XYZW[  416][0] =  0.147594109495424;
    Leb_Grid_XYZW[  416][1] =  0.699362559350389;
    Leb_Grid_XYZW[  416][2] = -0.699362559350389;
    Leb_Grid_XYZW[  416][3] =  0.000544228650010;

    Leb_Grid_XYZW[  417][0] = -0.147594109495423;
    Leb_Grid_XYZW[  417][1] =  0.699362559350389;
    Leb_Grid_XYZW[  417][2] = -0.699362559350389;
    Leb_Grid_XYZW[  417][3] =  0.000544228650010;

    Leb_Grid_XYZW[  418][0] =  0.147594109495424;
    Leb_Grid_XYZW[  418][1] = -0.699362559350389;
    Leb_Grid_XYZW[  418][2] =  0.699362559350389;
    Leb_Grid_XYZW[  418][3] =  0.000544228650010;

    Leb_Grid_XYZW[  419][0] = -0.147594109495423;
    Leb_Grid_XYZW[  419][1] = -0.699362559350389;
    Leb_Grid_XYZW[  419][2] =  0.699362559350389;
    Leb_Grid_XYZW[  419][3] =  0.000544228650010;

    Leb_Grid_XYZW[  420][0] =  0.147594109495424;
    Leb_Grid_XYZW[  420][1] = -0.699362559350389;
    Leb_Grid_XYZW[  420][2] = -0.699362559350389;
    Leb_Grid_XYZW[  420][3] =  0.000544228650010;

    Leb_Grid_XYZW[  421][0] = -0.147594109495423;
    Leb_Grid_XYZW[  421][1] = -0.699362559350389;
    Leb_Grid_XYZW[  421][2] = -0.699362559350389;
    Leb_Grid_XYZW[  421][3] =  0.000544228650010;

    Leb_Grid_XYZW[  422][0] =  0.706239338771938;
    Leb_Grid_XYZW[  422][1] =  0.706239338771938;
    Leb_Grid_XYZW[  422][2] =  0.049517600325052;
    Leb_Grid_XYZW[  422][3] =  0.000545225034506;

    Leb_Grid_XYZW[  423][0] =  0.706239338771938;
    Leb_Grid_XYZW[  423][1] =  0.706239338771938;
    Leb_Grid_XYZW[  423][2] = -0.049517600325052;
    Leb_Grid_XYZW[  423][3] =  0.000545225034506;

    Leb_Grid_XYZW[  424][0] =  0.706239338771938;
    Leb_Grid_XYZW[  424][1] = -0.706239338771938;
    Leb_Grid_XYZW[  424][2] =  0.049517600325052;
    Leb_Grid_XYZW[  424][3] =  0.000545225034506;

    Leb_Grid_XYZW[  425][0] =  0.706239338771938;
    Leb_Grid_XYZW[  425][1] = -0.706239338771938;
    Leb_Grid_XYZW[  425][2] = -0.049517600325052;
    Leb_Grid_XYZW[  425][3] =  0.000545225034506;

    Leb_Grid_XYZW[  426][0] = -0.706239338771938;
    Leb_Grid_XYZW[  426][1] =  0.706239338771938;
    Leb_Grid_XYZW[  426][2] =  0.049517600325052;
    Leb_Grid_XYZW[  426][3] =  0.000545225034506;

    Leb_Grid_XYZW[  427][0] = -0.706239338771938;
    Leb_Grid_XYZW[  427][1] =  0.706239338771938;
    Leb_Grid_XYZW[  427][2] = -0.049517600325052;
    Leb_Grid_XYZW[  427][3] =  0.000545225034506;

    Leb_Grid_XYZW[  428][0] = -0.706239338771938;
    Leb_Grid_XYZW[  428][1] = -0.706239338771938;
    Leb_Grid_XYZW[  428][2] =  0.049517600325052;
    Leb_Grid_XYZW[  428][3] =  0.000545225034506;

    Leb_Grid_XYZW[  429][0] = -0.706239338771938;
    Leb_Grid_XYZW[  429][1] = -0.706239338771938;
    Leb_Grid_XYZW[  429][2] = -0.049517600325052;
    Leb_Grid_XYZW[  429][3] =  0.000545225034506;

    Leb_Grid_XYZW[  430][0] =  0.706239338771938;
    Leb_Grid_XYZW[  430][1] =  0.049517600325052;
    Leb_Grid_XYZW[  430][2] =  0.706239338771938;
    Leb_Grid_XYZW[  430][3] =  0.000545225034506;

    Leb_Grid_XYZW[  431][0] =  0.706239338771938;
    Leb_Grid_XYZW[  431][1] = -0.049517600325052;
    Leb_Grid_XYZW[  431][2] =  0.706239338771938;
    Leb_Grid_XYZW[  431][3] =  0.000545225034506;

    Leb_Grid_XYZW[  432][0] =  0.706239338771938;
    Leb_Grid_XYZW[  432][1] =  0.049517600325052;
    Leb_Grid_XYZW[  432][2] = -0.706239338771938;
    Leb_Grid_XYZW[  432][3] =  0.000545225034506;

    Leb_Grid_XYZW[  433][0] =  0.706239338771938;
    Leb_Grid_XYZW[  433][1] = -0.049517600325052;
    Leb_Grid_XYZW[  433][2] = -0.706239338771938;
    Leb_Grid_XYZW[  433][3] =  0.000545225034506;

    Leb_Grid_XYZW[  434][0] = -0.706239338771938;
    Leb_Grid_XYZW[  434][1] =  0.049517600325052;
    Leb_Grid_XYZW[  434][2] =  0.706239338771938;
    Leb_Grid_XYZW[  434][3] =  0.000545225034506;

    Leb_Grid_XYZW[  435][0] = -0.706239338771938;
    Leb_Grid_XYZW[  435][1] = -0.049517600325052;
    Leb_Grid_XYZW[  435][2] =  0.706239338771938;
    Leb_Grid_XYZW[  435][3] =  0.000545225034506;

    Leb_Grid_XYZW[  436][0] = -0.706239338771938;
    Leb_Grid_XYZW[  436][1] =  0.049517600325052;
    Leb_Grid_XYZW[  436][2] = -0.706239338771938;
    Leb_Grid_XYZW[  436][3] =  0.000545225034506;

    Leb_Grid_XYZW[  437][0] = -0.706239338771938;
    Leb_Grid_XYZW[  437][1] = -0.049517600325052;
    Leb_Grid_XYZW[  437][2] = -0.706239338771938;
    Leb_Grid_XYZW[  437][3] =  0.000545225034506;

    Leb_Grid_XYZW[  438][0] =  0.049517600325052;
    Leb_Grid_XYZW[  438][1] =  0.706239338771938;
    Leb_Grid_XYZW[  438][2] =  0.706239338771938;
    Leb_Grid_XYZW[  438][3] =  0.000545225034506;

    Leb_Grid_XYZW[  439][0] = -0.049517600325052;
    Leb_Grid_XYZW[  439][1] =  0.706239338771938;
    Leb_Grid_XYZW[  439][2] =  0.706239338771938;
    Leb_Grid_XYZW[  439][3] =  0.000545225034506;

    Leb_Grid_XYZW[  440][0] =  0.049517600325052;
    Leb_Grid_XYZW[  440][1] =  0.706239338771938;
    Leb_Grid_XYZW[  440][2] = -0.706239338771938;
    Leb_Grid_XYZW[  440][3] =  0.000545225034506;

    Leb_Grid_XYZW[  441][0] = -0.049517600325052;
    Leb_Grid_XYZW[  441][1] =  0.706239338771938;
    Leb_Grid_XYZW[  441][2] = -0.706239338771938;
    Leb_Grid_XYZW[  441][3] =  0.000545225034506;

    Leb_Grid_XYZW[  442][0] =  0.049517600325052;
    Leb_Grid_XYZW[  442][1] = -0.706239338771938;
    Leb_Grid_XYZW[  442][2] =  0.706239338771938;
    Leb_Grid_XYZW[  442][3] =  0.000545225034506;

    Leb_Grid_XYZW[  443][0] = -0.049517600325052;
    Leb_Grid_XYZW[  443][1] = -0.706239338771938;
    Leb_Grid_XYZW[  443][2] =  0.706239338771938;
    Leb_Grid_XYZW[  443][3] =  0.000545225034506;

    Leb_Grid_XYZW[  444][0] =  0.049517600325052;
    Leb_Grid_XYZW[  444][1] = -0.706239338771938;
    Leb_Grid_XYZW[  444][2] = -0.706239338771938;
    Leb_Grid_XYZW[  444][3] =  0.000545225034506;

    Leb_Grid_XYZW[  445][0] = -0.049517600325052;
    Leb_Grid_XYZW[  445][1] = -0.706239338771938;
    Leb_Grid_XYZW[  445][2] = -0.706239338771938;
    Leb_Grid_XYZW[  445][3] =  0.000545225034506;

    Leb_Grid_XYZW[  446][0] =  0.074790281683498;
    Leb_Grid_XYZW[  446][1] =  0.997199284880261;
    Leb_Grid_XYZW[  446][2] =  0.000000000000000;
    Leb_Grid_XYZW[  446][3] =  0.000256800249773;

    Leb_Grid_XYZW[  447][0] =  0.074790281683498;
    Leb_Grid_XYZW[  447][1] = -0.997199284880261;
    Leb_Grid_XYZW[  447][2] =  0.000000000000000;
    Leb_Grid_XYZW[  447][3] =  0.000256800249773;

    Leb_Grid_XYZW[  448][0] = -0.074790281683498;
    Leb_Grid_XYZW[  448][1] =  0.997199284880261;
    Leb_Grid_XYZW[  448][2] =  0.000000000000000;
    Leb_Grid_XYZW[  448][3] =  0.000256800249773;

    Leb_Grid_XYZW[  449][0] = -0.074790281683498;
    Leb_Grid_XYZW[  449][1] = -0.997199284880261;
    Leb_Grid_XYZW[  449][2] =  0.000000000000000;
    Leb_Grid_XYZW[  449][3] =  0.000256800249773;

    Leb_Grid_XYZW[  450][0] =  0.997199284880261;
    Leb_Grid_XYZW[  450][1] =  0.074790281683498;
    Leb_Grid_XYZW[  450][2] =  0.000000000000000;
    Leb_Grid_XYZW[  450][3] =  0.000256800249773;

    Leb_Grid_XYZW[  451][0] =  0.997199284880261;
    Leb_Grid_XYZW[  451][1] = -0.074790281683498;
    Leb_Grid_XYZW[  451][2] =  0.000000000000000;
    Leb_Grid_XYZW[  451][3] =  0.000256800249773;

    Leb_Grid_XYZW[  452][0] = -0.997199284880261;
    Leb_Grid_XYZW[  452][1] =  0.074790281683498;
    Leb_Grid_XYZW[  452][2] =  0.000000000000000;
    Leb_Grid_XYZW[  452][3] =  0.000256800249773;

    Leb_Grid_XYZW[  453][0] = -0.997199284880261;
    Leb_Grid_XYZW[  453][1] = -0.074790281683498;
    Leb_Grid_XYZW[  453][2] =  0.000000000000000;
    Leb_Grid_XYZW[  453][3] =  0.000256800249773;

    Leb_Grid_XYZW[  454][0] =  0.074790281683498;
    Leb_Grid_XYZW[  454][1] =  0.000000000000000;
    Leb_Grid_XYZW[  454][2] =  0.997199284880261;
    Leb_Grid_XYZW[  454][3] =  0.000256800249773;

    Leb_Grid_XYZW[  455][0] =  0.074790281683498;
    Leb_Grid_XYZW[  455][1] =  0.000000000000000;
    Leb_Grid_XYZW[  455][2] = -0.997199284880261;
    Leb_Grid_XYZW[  455][3] =  0.000256800249773;

    Leb_Grid_XYZW[  456][0] = -0.074790281683498;
    Leb_Grid_XYZW[  456][1] =  0.000000000000000;
    Leb_Grid_XYZW[  456][2] =  0.997199284880261;
    Leb_Grid_XYZW[  456][3] =  0.000256800249773;

    Leb_Grid_XYZW[  457][0] = -0.074790281683498;
    Leb_Grid_XYZW[  457][1] =  0.000000000000000;
    Leb_Grid_XYZW[  457][2] = -0.997199284880261;
    Leb_Grid_XYZW[  457][3] =  0.000256800249773;

    Leb_Grid_XYZW[  458][0] =  0.997199284880261;
    Leb_Grid_XYZW[  458][1] =  0.000000000000000;
    Leb_Grid_XYZW[  458][2] =  0.074790281683498;
    Leb_Grid_XYZW[  458][3] =  0.000256800249773;

    Leb_Grid_XYZW[  459][0] =  0.997199284880261;
    Leb_Grid_XYZW[  459][1] =  0.000000000000000;
    Leb_Grid_XYZW[  459][2] = -0.074790281683498;
    Leb_Grid_XYZW[  459][3] =  0.000256800249773;

    Leb_Grid_XYZW[  460][0] = -0.997199284880261;
    Leb_Grid_XYZW[  460][1] =  0.000000000000000;
    Leb_Grid_XYZW[  460][2] =  0.074790281683498;
    Leb_Grid_XYZW[  460][3] =  0.000256800249773;

    Leb_Grid_XYZW[  461][0] = -0.997199284880261;
    Leb_Grid_XYZW[  461][1] =  0.000000000000000;
    Leb_Grid_XYZW[  461][2] = -0.074790281683498;
    Leb_Grid_XYZW[  461][3] =  0.000256800249773;

    Leb_Grid_XYZW[  462][0] =  0.000000000000000;
    Leb_Grid_XYZW[  462][1] =  0.074790281683498;
    Leb_Grid_XYZW[  462][2] =  0.997199284880261;
    Leb_Grid_XYZW[  462][3] =  0.000256800249773;

    Leb_Grid_XYZW[  463][0] =  0.000000000000000;
    Leb_Grid_XYZW[  463][1] =  0.074790281683498;
    Leb_Grid_XYZW[  463][2] = -0.997199284880261;
    Leb_Grid_XYZW[  463][3] =  0.000256800249773;

    Leb_Grid_XYZW[  464][0] =  0.000000000000000;
    Leb_Grid_XYZW[  464][1] = -0.074790281683498;
    Leb_Grid_XYZW[  464][2] =  0.997199284880261;
    Leb_Grid_XYZW[  464][3] =  0.000256800249773;

    Leb_Grid_XYZW[  465][0] =  0.000000000000000;
    Leb_Grid_XYZW[  465][1] = -0.074790281683498;
    Leb_Grid_XYZW[  465][2] = -0.997199284880261;
    Leb_Grid_XYZW[  465][3] =  0.000256800249773;

    Leb_Grid_XYZW[  466][0] =  0.000000000000000;
    Leb_Grid_XYZW[  466][1] =  0.997199284880261;
    Leb_Grid_XYZW[  466][2] =  0.074790281683498;
    Leb_Grid_XYZW[  466][3] =  0.000256800249773;

    Leb_Grid_XYZW[  467][0] =  0.000000000000000;
    Leb_Grid_XYZW[  467][1] =  0.997199284880261;
    Leb_Grid_XYZW[  467][2] = -0.074790281683498;
    Leb_Grid_XYZW[  467][3] =  0.000256800249773;

    Leb_Grid_XYZW[  468][0] =  0.000000000000000;
    Leb_Grid_XYZW[  468][1] = -0.997199284880261;
    Leb_Grid_XYZW[  468][2] =  0.074790281683498;
    Leb_Grid_XYZW[  468][3] =  0.000256800249773;

    Leb_Grid_XYZW[  469][0] =  0.000000000000000;
    Leb_Grid_XYZW[  469][1] = -0.997199284880261;
    Leb_Grid_XYZW[  469][2] = -0.074790281683498;
    Leb_Grid_XYZW[  469][3] =  0.000256800249773;

    Leb_Grid_XYZW[  470][0] =  0.184895115396937;
    Leb_Grid_XYZW[  470][1] =  0.982758259340695;
    Leb_Grid_XYZW[  470][2] =  0.000000000000000;
    Leb_Grid_XYZW[  470][3] =  0.000382721170029;

    Leb_Grid_XYZW[  471][0] =  0.184895115396937;
    Leb_Grid_XYZW[  471][1] = -0.982758259340695;
    Leb_Grid_XYZW[  471][2] =  0.000000000000000;
    Leb_Grid_XYZW[  471][3] =  0.000382721170029;

    Leb_Grid_XYZW[  472][0] = -0.184895115396937;
    Leb_Grid_XYZW[  472][1] =  0.982758259340695;
    Leb_Grid_XYZW[  472][2] =  0.000000000000000;
    Leb_Grid_XYZW[  472][3] =  0.000382721170029;

    Leb_Grid_XYZW[  473][0] = -0.184895115396937;
    Leb_Grid_XYZW[  473][1] = -0.982758259340695;
    Leb_Grid_XYZW[  473][2] =  0.000000000000000;
    Leb_Grid_XYZW[  473][3] =  0.000382721170029;

    Leb_Grid_XYZW[  474][0] =  0.982758259340695;
    Leb_Grid_XYZW[  474][1] =  0.184895115396937;
    Leb_Grid_XYZW[  474][2] =  0.000000000000000;
    Leb_Grid_XYZW[  474][3] =  0.000382721170029;

    Leb_Grid_XYZW[  475][0] =  0.982758259340695;
    Leb_Grid_XYZW[  475][1] = -0.184895115396937;
    Leb_Grid_XYZW[  475][2] =  0.000000000000000;
    Leb_Grid_XYZW[  475][3] =  0.000382721170029;

    Leb_Grid_XYZW[  476][0] = -0.982758259340695;
    Leb_Grid_XYZW[  476][1] =  0.184895115396937;
    Leb_Grid_XYZW[  476][2] =  0.000000000000000;
    Leb_Grid_XYZW[  476][3] =  0.000382721170029;

    Leb_Grid_XYZW[  477][0] = -0.982758259340695;
    Leb_Grid_XYZW[  477][1] = -0.184895115396937;
    Leb_Grid_XYZW[  477][2] =  0.000000000000000;
    Leb_Grid_XYZW[  477][3] =  0.000382721170029;

    Leb_Grid_XYZW[  478][0] =  0.184895115396937;
    Leb_Grid_XYZW[  478][1] =  0.000000000000000;
    Leb_Grid_XYZW[  478][2] =  0.982758259340695;
    Leb_Grid_XYZW[  478][3] =  0.000382721170029;

    Leb_Grid_XYZW[  479][0] =  0.184895115396937;
    Leb_Grid_XYZW[  479][1] =  0.000000000000000;
    Leb_Grid_XYZW[  479][2] = -0.982758259340695;
    Leb_Grid_XYZW[  479][3] =  0.000382721170029;

    Leb_Grid_XYZW[  480][0] = -0.184895115396937;
    Leb_Grid_XYZW[  480][1] =  0.000000000000000;
    Leb_Grid_XYZW[  480][2] =  0.982758259340695;
    Leb_Grid_XYZW[  480][3] =  0.000382721170029;

    Leb_Grid_XYZW[  481][0] = -0.184895115396937;
    Leb_Grid_XYZW[  481][1] =  0.000000000000000;
    Leb_Grid_XYZW[  481][2] = -0.982758259340695;
    Leb_Grid_XYZW[  481][3] =  0.000382721170029;

    Leb_Grid_XYZW[  482][0] =  0.982758259340695;
    Leb_Grid_XYZW[  482][1] =  0.000000000000000;
    Leb_Grid_XYZW[  482][2] =  0.184895115396937;
    Leb_Grid_XYZW[  482][3] =  0.000382721170029;

    Leb_Grid_XYZW[  483][0] =  0.982758259340695;
    Leb_Grid_XYZW[  483][1] =  0.000000000000000;
    Leb_Grid_XYZW[  483][2] = -0.184895115396937;
    Leb_Grid_XYZW[  483][3] =  0.000382721170029;

    Leb_Grid_XYZW[  484][0] = -0.982758259340695;
    Leb_Grid_XYZW[  484][1] =  0.000000000000000;
    Leb_Grid_XYZW[  484][2] =  0.184895115396937;
    Leb_Grid_XYZW[  484][3] =  0.000382721170029;

    Leb_Grid_XYZW[  485][0] = -0.982758259340695;
    Leb_Grid_XYZW[  485][1] =  0.000000000000000;
    Leb_Grid_XYZW[  485][2] = -0.184895115396937;
    Leb_Grid_XYZW[  485][3] =  0.000382721170029;

    Leb_Grid_XYZW[  486][0] =  0.000000000000000;
    Leb_Grid_XYZW[  486][1] =  0.184895115396937;
    Leb_Grid_XYZW[  486][2] =  0.982758259340695;
    Leb_Grid_XYZW[  486][3] =  0.000382721170029;

    Leb_Grid_XYZW[  487][0] =  0.000000000000000;
    Leb_Grid_XYZW[  487][1] =  0.184895115396937;
    Leb_Grid_XYZW[  487][2] = -0.982758259340695;
    Leb_Grid_XYZW[  487][3] =  0.000382721170029;

    Leb_Grid_XYZW[  488][0] =  0.000000000000000;
    Leb_Grid_XYZW[  488][1] = -0.184895115396937;
    Leb_Grid_XYZW[  488][2] =  0.982758259340695;
    Leb_Grid_XYZW[  488][3] =  0.000382721170029;

    Leb_Grid_XYZW[  489][0] =  0.000000000000000;
    Leb_Grid_XYZW[  489][1] = -0.184895115396937;
    Leb_Grid_XYZW[  489][2] = -0.982758259340695;
    Leb_Grid_XYZW[  489][3] =  0.000382721170029;

    Leb_Grid_XYZW[  490][0] =  0.000000000000000;
    Leb_Grid_XYZW[  490][1] =  0.982758259340695;
    Leb_Grid_XYZW[  490][2] =  0.184895115396937;
    Leb_Grid_XYZW[  490][3] =  0.000382721170029;

    Leb_Grid_XYZW[  491][0] =  0.000000000000000;
    Leb_Grid_XYZW[  491][1] =  0.982758259340695;
    Leb_Grid_XYZW[  491][2] = -0.184895115396937;
    Leb_Grid_XYZW[  491][3] =  0.000382721170029;

    Leb_Grid_XYZW[  492][0] =  0.000000000000000;
    Leb_Grid_XYZW[  492][1] = -0.982758259340695;
    Leb_Grid_XYZW[  492][2] =  0.184895115396937;
    Leb_Grid_XYZW[  492][3] =  0.000382721170029;

    Leb_Grid_XYZW[  493][0] =  0.000000000000000;
    Leb_Grid_XYZW[  493][1] = -0.982758259340695;
    Leb_Grid_XYZW[  493][2] = -0.184895115396937;
    Leb_Grid_XYZW[  493][3] =  0.000382721170029;

    Leb_Grid_XYZW[  494][0] =  0.305952906658131;
    Leb_Grid_XYZW[  494][1] =  0.952046647442992;
    Leb_Grid_XYZW[  494][2] =  0.000000000000000;
    Leb_Grid_XYZW[  494][3] =  0.000457949156192;

    Leb_Grid_XYZW[  495][0] =  0.305952906658131;
    Leb_Grid_XYZW[  495][1] = -0.952046647442992;
    Leb_Grid_XYZW[  495][2] =  0.000000000000000;
    Leb_Grid_XYZW[  495][3] =  0.000457949156192;

    Leb_Grid_XYZW[  496][0] = -0.305952906658131;
    Leb_Grid_XYZW[  496][1] =  0.952046647442992;
    Leb_Grid_XYZW[  496][2] =  0.000000000000000;
    Leb_Grid_XYZW[  496][3] =  0.000457949156192;

    Leb_Grid_XYZW[  497][0] = -0.305952906658131;
    Leb_Grid_XYZW[  497][1] = -0.952046647442992;
    Leb_Grid_XYZW[  497][2] =  0.000000000000000;
    Leb_Grid_XYZW[  497][3] =  0.000457949156192;

    Leb_Grid_XYZW[  498][0] =  0.952046647442992;
    Leb_Grid_XYZW[  498][1] =  0.305952906658131;
    Leb_Grid_XYZW[  498][2] =  0.000000000000000;
    Leb_Grid_XYZW[  498][3] =  0.000457949156192;

    Leb_Grid_XYZW[  499][0] =  0.952046647442992;
    Leb_Grid_XYZW[  499][1] = -0.305952906658131;
    Leb_Grid_XYZW[  499][2] =  0.000000000000000;
    Leb_Grid_XYZW[  499][3] =  0.000457949156192;

    Leb_Grid_XYZW[  500][0] = -0.952046647442992;
    Leb_Grid_XYZW[  500][1] =  0.305952906658130;
    Leb_Grid_XYZW[  500][2] =  0.000000000000000;
    Leb_Grid_XYZW[  500][3] =  0.000457949156192;

    Leb_Grid_XYZW[  501][0] = -0.952046647442992;
    Leb_Grid_XYZW[  501][1] = -0.305952906658130;
    Leb_Grid_XYZW[  501][2] =  0.000000000000000;
    Leb_Grid_XYZW[  501][3] =  0.000457949156192;

    Leb_Grid_XYZW[  502][0] =  0.305952906658131;
    Leb_Grid_XYZW[  502][1] =  0.000000000000000;
    Leb_Grid_XYZW[  502][2] =  0.952046647442992;
    Leb_Grid_XYZW[  502][3] =  0.000457949156192;

    Leb_Grid_XYZW[  503][0] =  0.305952906658130;
    Leb_Grid_XYZW[  503][1] =  0.000000000000000;
    Leb_Grid_XYZW[  503][2] = -0.952046647442992;
    Leb_Grid_XYZW[  503][3] =  0.000457949156192;

    Leb_Grid_XYZW[  504][0] = -0.305952906658131;
    Leb_Grid_XYZW[  504][1] =  0.000000000000000;
    Leb_Grid_XYZW[  504][2] =  0.952046647442992;
    Leb_Grid_XYZW[  504][3] =  0.000457949156192;

    Leb_Grid_XYZW[  505][0] = -0.305952906658130;
    Leb_Grid_XYZW[  505][1] =  0.000000000000000;
    Leb_Grid_XYZW[  505][2] = -0.952046647442992;
    Leb_Grid_XYZW[  505][3] =  0.000457949156192;

    Leb_Grid_XYZW[  506][0] =  0.952046647442992;
    Leb_Grid_XYZW[  506][1] =  0.000000000000000;
    Leb_Grid_XYZW[  506][2] =  0.305952906658131;
    Leb_Grid_XYZW[  506][3] =  0.000457949156192;

    Leb_Grid_XYZW[  507][0] =  0.952046647442992;
    Leb_Grid_XYZW[  507][1] =  0.000000000000000;
    Leb_Grid_XYZW[  507][2] = -0.305952906658131;
    Leb_Grid_XYZW[  507][3] =  0.000457949156192;

    Leb_Grid_XYZW[  508][0] = -0.952046647442992;
    Leb_Grid_XYZW[  508][1] =  0.000000000000000;
    Leb_Grid_XYZW[  508][2] =  0.305952906658131;
    Leb_Grid_XYZW[  508][3] =  0.000457949156192;

    Leb_Grid_XYZW[  509][0] = -0.952046647442992;
    Leb_Grid_XYZW[  509][1] =  0.000000000000000;
    Leb_Grid_XYZW[  509][2] = -0.305952906658131;
    Leb_Grid_XYZW[  509][3] =  0.000457949156192;

    Leb_Grid_XYZW[  510][0] =  0.000000000000000;
    Leb_Grid_XYZW[  510][1] =  0.305952906658131;
    Leb_Grid_XYZW[  510][2] =  0.952046647442992;
    Leb_Grid_XYZW[  510][3] =  0.000457949156192;

    Leb_Grid_XYZW[  511][0] =  0.000000000000000;
    Leb_Grid_XYZW[  511][1] =  0.305952906658130;
    Leb_Grid_XYZW[  511][2] = -0.952046647442992;
    Leb_Grid_XYZW[  511][3] =  0.000457949156192;

    Leb_Grid_XYZW[  512][0] =  0.000000000000000;
    Leb_Grid_XYZW[  512][1] = -0.305952906658131;
    Leb_Grid_XYZW[  512][2] =  0.952046647442992;
    Leb_Grid_XYZW[  512][3] =  0.000457949156192;

    Leb_Grid_XYZW[  513][0] =  0.000000000000000;
    Leb_Grid_XYZW[  513][1] = -0.305952906658130;
    Leb_Grid_XYZW[  513][2] = -0.952046647442992;
    Leb_Grid_XYZW[  513][3] =  0.000457949156192;

    Leb_Grid_XYZW[  514][0] =  0.000000000000000;
    Leb_Grid_XYZW[  514][1] =  0.952046647442992;
    Leb_Grid_XYZW[  514][2] =  0.305952906658131;
    Leb_Grid_XYZW[  514][3] =  0.000457949156192;

    Leb_Grid_XYZW[  515][0] =  0.000000000000000;
    Leb_Grid_XYZW[  515][1] =  0.952046647442992;
    Leb_Grid_XYZW[  515][2] = -0.305952906658131;
    Leb_Grid_XYZW[  515][3] =  0.000457949156192;

    Leb_Grid_XYZW[  516][0] =  0.000000000000000;
    Leb_Grid_XYZW[  516][1] = -0.952046647442992;
    Leb_Grid_XYZW[  516][2] =  0.305952906658131;
    Leb_Grid_XYZW[  516][3] =  0.000457949156192;

    Leb_Grid_XYZW[  517][0] =  0.000000000000000;
    Leb_Grid_XYZW[  517][1] = -0.952046647442992;
    Leb_Grid_XYZW[  517][2] = -0.305952906658131;
    Leb_Grid_XYZW[  517][3] =  0.000457949156192;

    Leb_Grid_XYZW[  518][0] =  0.428555610102136;
    Leb_Grid_XYZW[  518][1] =  0.903515406094432;
    Leb_Grid_XYZW[  518][2] =  0.000000000000000;
    Leb_Grid_XYZW[  518][3] =  0.000504200396908;

    Leb_Grid_XYZW[  519][0] =  0.428555610102136;
    Leb_Grid_XYZW[  519][1] = -0.903515406094432;
    Leb_Grid_XYZW[  519][2] =  0.000000000000000;
    Leb_Grid_XYZW[  519][3] =  0.000504200396908;

    Leb_Grid_XYZW[  520][0] = -0.428555610102136;
    Leb_Grid_XYZW[  520][1] =  0.903515406094432;
    Leb_Grid_XYZW[  520][2] =  0.000000000000000;
    Leb_Grid_XYZW[  520][3] =  0.000504200396908;

    Leb_Grid_XYZW[  521][0] = -0.428555610102136;
    Leb_Grid_XYZW[  521][1] = -0.903515406094432;
    Leb_Grid_XYZW[  521][2] =  0.000000000000000;
    Leb_Grid_XYZW[  521][3] =  0.000504200396908;

    Leb_Grid_XYZW[  522][0] =  0.903515406094432;
    Leb_Grid_XYZW[  522][1] =  0.428555610102136;
    Leb_Grid_XYZW[  522][2] =  0.000000000000000;
    Leb_Grid_XYZW[  522][3] =  0.000504200396908;

    Leb_Grid_XYZW[  523][0] =  0.903515406094432;
    Leb_Grid_XYZW[  523][1] = -0.428555610102136;
    Leb_Grid_XYZW[  523][2] =  0.000000000000000;
    Leb_Grid_XYZW[  523][3] =  0.000504200396908;

    Leb_Grid_XYZW[  524][0] = -0.903515406094432;
    Leb_Grid_XYZW[  524][1] =  0.428555610102136;
    Leb_Grid_XYZW[  524][2] =  0.000000000000000;
    Leb_Grid_XYZW[  524][3] =  0.000504200396908;

    Leb_Grid_XYZW[  525][0] = -0.903515406094432;
    Leb_Grid_XYZW[  525][1] = -0.428555610102136;
    Leb_Grid_XYZW[  525][2] =  0.000000000000000;
    Leb_Grid_XYZW[  525][3] =  0.000504200396908;

    Leb_Grid_XYZW[  526][0] =  0.428555610102136;
    Leb_Grid_XYZW[  526][1] =  0.000000000000000;
    Leb_Grid_XYZW[  526][2] =  0.903515406094432;
    Leb_Grid_XYZW[  526][3] =  0.000504200396908;

    Leb_Grid_XYZW[  527][0] =  0.428555610102136;
    Leb_Grid_XYZW[  527][1] =  0.000000000000000;
    Leb_Grid_XYZW[  527][2] = -0.903515406094432;
    Leb_Grid_XYZW[  527][3] =  0.000504200396908;

    Leb_Grid_XYZW[  528][0] = -0.428555610102136;
    Leb_Grid_XYZW[  528][1] =  0.000000000000000;
    Leb_Grid_XYZW[  528][2] =  0.903515406094432;
    Leb_Grid_XYZW[  528][3] =  0.000504200396908;

    Leb_Grid_XYZW[  529][0] = -0.428555610102136;
    Leb_Grid_XYZW[  529][1] =  0.000000000000000;
    Leb_Grid_XYZW[  529][2] = -0.903515406094432;
    Leb_Grid_XYZW[  529][3] =  0.000504200396908;

    Leb_Grid_XYZW[  530][0] =  0.903515406094432;
    Leb_Grid_XYZW[  530][1] =  0.000000000000000;
    Leb_Grid_XYZW[  530][2] =  0.428555610102136;
    Leb_Grid_XYZW[  530][3] =  0.000504200396908;

    Leb_Grid_XYZW[  531][0] =  0.903515406094432;
    Leb_Grid_XYZW[  531][1] =  0.000000000000000;
    Leb_Grid_XYZW[  531][2] = -0.428555610102136;
    Leb_Grid_XYZW[  531][3] =  0.000504200396908;

    Leb_Grid_XYZW[  532][0] = -0.903515406094432;
    Leb_Grid_XYZW[  532][1] =  0.000000000000000;
    Leb_Grid_XYZW[  532][2] =  0.428555610102136;
    Leb_Grid_XYZW[  532][3] =  0.000504200396908;

    Leb_Grid_XYZW[  533][0] = -0.903515406094432;
    Leb_Grid_XYZW[  533][1] =  0.000000000000000;
    Leb_Grid_XYZW[  533][2] = -0.428555610102136;
    Leb_Grid_XYZW[  533][3] =  0.000504200396908;

    Leb_Grid_XYZW[  534][0] =  0.000000000000000;
    Leb_Grid_XYZW[  534][1] =  0.428555610102136;
    Leb_Grid_XYZW[  534][2] =  0.903515406094432;
    Leb_Grid_XYZW[  534][3] =  0.000504200396908;

    Leb_Grid_XYZW[  535][0] =  0.000000000000000;
    Leb_Grid_XYZW[  535][1] =  0.428555610102136;
    Leb_Grid_XYZW[  535][2] = -0.903515406094432;
    Leb_Grid_XYZW[  535][3] =  0.000504200396908;

    Leb_Grid_XYZW[  536][0] =  0.000000000000000;
    Leb_Grid_XYZW[  536][1] = -0.428555610102136;
    Leb_Grid_XYZW[  536][2] =  0.903515406094432;
    Leb_Grid_XYZW[  536][3] =  0.000504200396908;

    Leb_Grid_XYZW[  537][0] =  0.000000000000000;
    Leb_Grid_XYZW[  537][1] = -0.428555610102136;
    Leb_Grid_XYZW[  537][2] = -0.903515406094432;
    Leb_Grid_XYZW[  537][3] =  0.000504200396908;

    Leb_Grid_XYZW[  538][0] =  0.000000000000000;
    Leb_Grid_XYZW[  538][1] =  0.903515406094432;
    Leb_Grid_XYZW[  538][2] =  0.428555610102136;
    Leb_Grid_XYZW[  538][3] =  0.000504200396908;

    Leb_Grid_XYZW[  539][0] =  0.000000000000000;
    Leb_Grid_XYZW[  539][1] =  0.903515406094432;
    Leb_Grid_XYZW[  539][2] = -0.428555610102136;
    Leb_Grid_XYZW[  539][3] =  0.000504200396908;

    Leb_Grid_XYZW[  540][0] =  0.000000000000000;
    Leb_Grid_XYZW[  540][1] = -0.903515406094432;
    Leb_Grid_XYZW[  540][2] =  0.428555610102136;
    Leb_Grid_XYZW[  540][3] =  0.000504200396908;

    Leb_Grid_XYZW[  541][0] =  0.000000000000000;
    Leb_Grid_XYZW[  541][1] = -0.903515406094432;
    Leb_Grid_XYZW[  541][2] = -0.428555610102136;
    Leb_Grid_XYZW[  541][3] =  0.000504200396908;

    Leb_Grid_XYZW[  542][0] =  0.546875865349653;
    Leb_Grid_XYZW[  542][1] =  0.837213705034783;
    Leb_Grid_XYZW[  542][2] =  0.000000000000000;
    Leb_Grid_XYZW[  542][3] =  0.000531270888998;

    Leb_Grid_XYZW[  543][0] =  0.546875865349653;
    Leb_Grid_XYZW[  543][1] = -0.837213705034783;
    Leb_Grid_XYZW[  543][2] =  0.000000000000000;
    Leb_Grid_XYZW[  543][3] =  0.000531270888998;

    Leb_Grid_XYZW[  544][0] = -0.546875865349652;
    Leb_Grid_XYZW[  544][1] =  0.837213705034783;
    Leb_Grid_XYZW[  544][2] =  0.000000000000000;
    Leb_Grid_XYZW[  544][3] =  0.000531270888998;

    Leb_Grid_XYZW[  545][0] = -0.546875865349652;
    Leb_Grid_XYZW[  545][1] = -0.837213705034783;
    Leb_Grid_XYZW[  545][2] =  0.000000000000000;
    Leb_Grid_XYZW[  545][3] =  0.000531270888998;

    Leb_Grid_XYZW[  546][0] =  0.837213705034783;
    Leb_Grid_XYZW[  546][1] =  0.546875865349653;
    Leb_Grid_XYZW[  546][2] =  0.000000000000000;
    Leb_Grid_XYZW[  546][3] =  0.000531270888998;

    Leb_Grid_XYZW[  547][0] =  0.837213705034783;
    Leb_Grid_XYZW[  547][1] = -0.546875865349653;
    Leb_Grid_XYZW[  547][2] =  0.000000000000000;
    Leb_Grid_XYZW[  547][3] =  0.000531270888998;

    Leb_Grid_XYZW[  548][0] = -0.837213705034783;
    Leb_Grid_XYZW[  548][1] =  0.546875865349653;
    Leb_Grid_XYZW[  548][2] =  0.000000000000000;
    Leb_Grid_XYZW[  548][3] =  0.000531270888998;

    Leb_Grid_XYZW[  549][0] = -0.837213705034783;
    Leb_Grid_XYZW[  549][1] = -0.546875865349653;
    Leb_Grid_XYZW[  549][2] =  0.000000000000000;
    Leb_Grid_XYZW[  549][3] =  0.000531270888998;

    Leb_Grid_XYZW[  550][0] =  0.546875865349653;
    Leb_Grid_XYZW[  550][1] =  0.000000000000000;
    Leb_Grid_XYZW[  550][2] =  0.837213705034783;
    Leb_Grid_XYZW[  550][3] =  0.000531270888998;

    Leb_Grid_XYZW[  551][0] =  0.546875865349653;
    Leb_Grid_XYZW[  551][1] =  0.000000000000000;
    Leb_Grid_XYZW[  551][2] = -0.837213705034783;
    Leb_Grid_XYZW[  551][3] =  0.000531270888998;

    Leb_Grid_XYZW[  552][0] = -0.546875865349653;
    Leb_Grid_XYZW[  552][1] =  0.000000000000000;
    Leb_Grid_XYZW[  552][2] =  0.837213705034783;
    Leb_Grid_XYZW[  552][3] =  0.000531270888998;

    Leb_Grid_XYZW[  553][0] = -0.546875865349653;
    Leb_Grid_XYZW[  553][1] =  0.000000000000000;
    Leb_Grid_XYZW[  553][2] = -0.837213705034783;
    Leb_Grid_XYZW[  553][3] =  0.000531270888998;

    Leb_Grid_XYZW[  554][0] =  0.837213705034783;
    Leb_Grid_XYZW[  554][1] =  0.000000000000000;
    Leb_Grid_XYZW[  554][2] =  0.546875865349653;
    Leb_Grid_XYZW[  554][3] =  0.000531270888998;

    Leb_Grid_XYZW[  555][0] =  0.837213705034783;
    Leb_Grid_XYZW[  555][1] =  0.000000000000000;
    Leb_Grid_XYZW[  555][2] = -0.546875865349652;
    Leb_Grid_XYZW[  555][3] =  0.000531270888998;

    Leb_Grid_XYZW[  556][0] = -0.837213705034783;
    Leb_Grid_XYZW[  556][1] =  0.000000000000000;
    Leb_Grid_XYZW[  556][2] =  0.546875865349653;
    Leb_Grid_XYZW[  556][3] =  0.000531270888998;

    Leb_Grid_XYZW[  557][0] = -0.837213705034783;
    Leb_Grid_XYZW[  557][1] =  0.000000000000000;
    Leb_Grid_XYZW[  557][2] = -0.546875865349652;
    Leb_Grid_XYZW[  557][3] =  0.000531270888998;

    Leb_Grid_XYZW[  558][0] =  0.000000000000000;
    Leb_Grid_XYZW[  558][1] =  0.546875865349653;
    Leb_Grid_XYZW[  558][2] =  0.837213705034783;
    Leb_Grid_XYZW[  558][3] =  0.000531270888998;

    Leb_Grid_XYZW[  559][0] =  0.000000000000000;
    Leb_Grid_XYZW[  559][1] =  0.546875865349653;
    Leb_Grid_XYZW[  559][2] = -0.837213705034783;
    Leb_Grid_XYZW[  559][3] =  0.000531270888998;

    Leb_Grid_XYZW[  560][0] =  0.000000000000000;
    Leb_Grid_XYZW[  560][1] = -0.546875865349653;
    Leb_Grid_XYZW[  560][2] =  0.837213705034783;
    Leb_Grid_XYZW[  560][3] =  0.000531270888998;

    Leb_Grid_XYZW[  561][0] =  0.000000000000000;
    Leb_Grid_XYZW[  561][1] = -0.546875865349653;
    Leb_Grid_XYZW[  561][2] = -0.837213705034783;
    Leb_Grid_XYZW[  561][3] =  0.000531270888998;

    Leb_Grid_XYZW[  562][0] =  0.000000000000000;
    Leb_Grid_XYZW[  562][1] =  0.837213705034783;
    Leb_Grid_XYZW[  562][2] =  0.546875865349653;
    Leb_Grid_XYZW[  562][3] =  0.000531270888998;

    Leb_Grid_XYZW[  563][0] =  0.000000000000000;
    Leb_Grid_XYZW[  563][1] =  0.837213705034783;
    Leb_Grid_XYZW[  563][2] = -0.546875865349652;
    Leb_Grid_XYZW[  563][3] =  0.000531270888998;

    Leb_Grid_XYZW[  564][0] =  0.000000000000000;
    Leb_Grid_XYZW[  564][1] = -0.837213705034783;
    Leb_Grid_XYZW[  564][2] =  0.546875865349653;
    Leb_Grid_XYZW[  564][3] =  0.000531270888998;

    Leb_Grid_XYZW[  565][0] =  0.000000000000000;
    Leb_Grid_XYZW[  565][1] = -0.837213705034783;
    Leb_Grid_XYZW[  565][2] = -0.546875865349652;
    Leb_Grid_XYZW[  565][3] =  0.000531270888998;

    Leb_Grid_XYZW[  566][0] =  0.656582197834344;
    Leb_Grid_XYZW[  566][1] =  0.754254477936341;
    Leb_Grid_XYZW[  566][2] =  0.000000000000000;
    Leb_Grid_XYZW[  566][3] =  0.000543840179075;

    Leb_Grid_XYZW[  567][0] =  0.656582197834344;
    Leb_Grid_XYZW[  567][1] = -0.754254477936341;
    Leb_Grid_XYZW[  567][2] =  0.000000000000000;
    Leb_Grid_XYZW[  567][3] =  0.000543840179075;

    Leb_Grid_XYZW[  568][0] = -0.656582197834344;
    Leb_Grid_XYZW[  568][1] =  0.754254477936341;
    Leb_Grid_XYZW[  568][2] =  0.000000000000000;
    Leb_Grid_XYZW[  568][3] =  0.000543840179075;

    Leb_Grid_XYZW[  569][0] = -0.656582197834344;
    Leb_Grid_XYZW[  569][1] = -0.754254477936341;
    Leb_Grid_XYZW[  569][2] =  0.000000000000000;
    Leb_Grid_XYZW[  569][3] =  0.000543840179075;

    Leb_Grid_XYZW[  570][0] =  0.754254477936341;
    Leb_Grid_XYZW[  570][1] =  0.656582197834344;
    Leb_Grid_XYZW[  570][2] =  0.000000000000000;
    Leb_Grid_XYZW[  570][3] =  0.000543840179075;

    Leb_Grid_XYZW[  571][0] =  0.754254477936341;
    Leb_Grid_XYZW[  571][1] = -0.656582197834344;
    Leb_Grid_XYZW[  571][2] =  0.000000000000000;
    Leb_Grid_XYZW[  571][3] =  0.000543840179075;

    Leb_Grid_XYZW[  572][0] = -0.754254477936341;
    Leb_Grid_XYZW[  572][1] =  0.656582197834344;
    Leb_Grid_XYZW[  572][2] =  0.000000000000000;
    Leb_Grid_XYZW[  572][3] =  0.000543840179075;

    Leb_Grid_XYZW[  573][0] = -0.754254477936341;
    Leb_Grid_XYZW[  573][1] = -0.656582197834344;
    Leb_Grid_XYZW[  573][2] =  0.000000000000000;
    Leb_Grid_XYZW[  573][3] =  0.000543840179075;

    Leb_Grid_XYZW[  574][0] =  0.656582197834344;
    Leb_Grid_XYZW[  574][1] =  0.000000000000000;
    Leb_Grid_XYZW[  574][2] =  0.754254477936341;
    Leb_Grid_XYZW[  574][3] =  0.000543840179075;

    Leb_Grid_XYZW[  575][0] =  0.656582197834344;
    Leb_Grid_XYZW[  575][1] =  0.000000000000000;
    Leb_Grid_XYZW[  575][2] = -0.754254477936341;
    Leb_Grid_XYZW[  575][3] =  0.000543840179075;

    Leb_Grid_XYZW[  576][0] = -0.656582197834344;
    Leb_Grid_XYZW[  576][1] =  0.000000000000000;
    Leb_Grid_XYZW[  576][2] =  0.754254477936341;
    Leb_Grid_XYZW[  576][3] =  0.000543840179075;

    Leb_Grid_XYZW[  577][0] = -0.656582197834344;
    Leb_Grid_XYZW[  577][1] =  0.000000000000000;
    Leb_Grid_XYZW[  577][2] = -0.754254477936341;
    Leb_Grid_XYZW[  577][3] =  0.000543840179075;

    Leb_Grid_XYZW[  578][0] =  0.754254477936341;
    Leb_Grid_XYZW[  578][1] =  0.000000000000000;
    Leb_Grid_XYZW[  578][2] =  0.656582197834344;
    Leb_Grid_XYZW[  578][3] =  0.000543840179075;

    Leb_Grid_XYZW[  579][0] =  0.754254477936341;
    Leb_Grid_XYZW[  579][1] =  0.000000000000000;
    Leb_Grid_XYZW[  579][2] = -0.656582197834344;
    Leb_Grid_XYZW[  579][3] =  0.000543840179075;

    Leb_Grid_XYZW[  580][0] = -0.754254477936341;
    Leb_Grid_XYZW[  580][1] =  0.000000000000000;
    Leb_Grid_XYZW[  580][2] =  0.656582197834344;
    Leb_Grid_XYZW[  580][3] =  0.000543840179075;

    Leb_Grid_XYZW[  581][0] = -0.754254477936341;
    Leb_Grid_XYZW[  581][1] =  0.000000000000000;
    Leb_Grid_XYZW[  581][2] = -0.656582197834344;
    Leb_Grid_XYZW[  581][3] =  0.000543840179075;

    Leb_Grid_XYZW[  582][0] =  0.000000000000000;
    Leb_Grid_XYZW[  582][1] =  0.656582197834344;
    Leb_Grid_XYZW[  582][2] =  0.754254477936341;
    Leb_Grid_XYZW[  582][3] =  0.000543840179075;

    Leb_Grid_XYZW[  583][0] =  0.000000000000000;
    Leb_Grid_XYZW[  583][1] =  0.656582197834344;
    Leb_Grid_XYZW[  583][2] = -0.754254477936341;
    Leb_Grid_XYZW[  583][3] =  0.000543840179075;

    Leb_Grid_XYZW[  584][0] =  0.000000000000000;
    Leb_Grid_XYZW[  584][1] = -0.656582197834344;
    Leb_Grid_XYZW[  584][2] =  0.754254477936341;
    Leb_Grid_XYZW[  584][3] =  0.000543840179075;

    Leb_Grid_XYZW[  585][0] =  0.000000000000000;
    Leb_Grid_XYZW[  585][1] = -0.656582197834344;
    Leb_Grid_XYZW[  585][2] = -0.754254477936341;
    Leb_Grid_XYZW[  585][3] =  0.000543840179075;

    Leb_Grid_XYZW[  586][0] =  0.000000000000000;
    Leb_Grid_XYZW[  586][1] =  0.754254477936341;
    Leb_Grid_XYZW[  586][2] =  0.656582197834344;
    Leb_Grid_XYZW[  586][3] =  0.000543840179075;

    Leb_Grid_XYZW[  587][0] =  0.000000000000000;
    Leb_Grid_XYZW[  587][1] =  0.754254477936341;
    Leb_Grid_XYZW[  587][2] = -0.656582197834344;
    Leb_Grid_XYZW[  587][3] =  0.000543840179075;

    Leb_Grid_XYZW[  588][0] =  0.000000000000000;
    Leb_Grid_XYZW[  588][1] = -0.754254477936341;
    Leb_Grid_XYZW[  588][2] =  0.656582197834344;
    Leb_Grid_XYZW[  588][3] =  0.000543840179075;

    Leb_Grid_XYZW[  589][0] =  0.000000000000000;
    Leb_Grid_XYZW[  589][1] = -0.754254477936341;
    Leb_Grid_XYZW[  589][2] = -0.656582197834344;
    Leb_Grid_XYZW[  589][3] =  0.000543840179075;

    Leb_Grid_XYZW[  590][0] =  0.125390157236712;
    Leb_Grid_XYZW[  590][1] =  0.036819172264396;
    Leb_Grid_XYZW[  590][2] =  0.991424055095456;
    Leb_Grid_XYZW[  590][3] =  0.000331604187320;

    Leb_Grid_XYZW[  591][0] =  0.125390157236712;
    Leb_Grid_XYZW[  591][1] =  0.036819172264396;
    Leb_Grid_XYZW[  591][2] = -0.991424055095456;
    Leb_Grid_XYZW[  591][3] =  0.000331604187320;

    Leb_Grid_XYZW[  592][0] =  0.125390157236712;
    Leb_Grid_XYZW[  592][1] = -0.036819172264396;
    Leb_Grid_XYZW[  592][2] =  0.991424055095456;
    Leb_Grid_XYZW[  592][3] =  0.000331604187320;

    Leb_Grid_XYZW[  593][0] =  0.125390157236712;
    Leb_Grid_XYZW[  593][1] = -0.036819172264396;
    Leb_Grid_XYZW[  593][2] = -0.991424055095456;
    Leb_Grid_XYZW[  593][3] =  0.000331604187320;

    Leb_Grid_XYZW[  594][0] = -0.125390157236712;
    Leb_Grid_XYZW[  594][1] =  0.036819172264396;
    Leb_Grid_XYZW[  594][2] =  0.991424055095456;
    Leb_Grid_XYZW[  594][3] =  0.000331604187320;

    Leb_Grid_XYZW[  595][0] = -0.125390157236712;
    Leb_Grid_XYZW[  595][1] =  0.036819172264396;
    Leb_Grid_XYZW[  595][2] = -0.991424055095456;
    Leb_Grid_XYZW[  595][3] =  0.000331604187320;

    Leb_Grid_XYZW[  596][0] = -0.125390157236712;
    Leb_Grid_XYZW[  596][1] = -0.036819172264396;
    Leb_Grid_XYZW[  596][2] =  0.991424055095456;
    Leb_Grid_XYZW[  596][3] =  0.000331604187320;

    Leb_Grid_XYZW[  597][0] = -0.125390157236712;
    Leb_Grid_XYZW[  597][1] = -0.036819172264396;
    Leb_Grid_XYZW[  597][2] = -0.991424055095456;
    Leb_Grid_XYZW[  597][3] =  0.000331604187320;

    Leb_Grid_XYZW[  598][0] =  0.125390157236712;
    Leb_Grid_XYZW[  598][1] =  0.991424055095456;
    Leb_Grid_XYZW[  598][2] =  0.036819172264396;
    Leb_Grid_XYZW[  598][3] =  0.000331604187320;

    Leb_Grid_XYZW[  599][0] =  0.125390157236712;
    Leb_Grid_XYZW[  599][1] =  0.991424055095456;
    Leb_Grid_XYZW[  599][2] = -0.036819172264397;
    Leb_Grid_XYZW[  599][3] =  0.000331604187320;

    Leb_Grid_XYZW[  600][0] =  0.125390157236712;
    Leb_Grid_XYZW[  600][1] = -0.991424055095456;
    Leb_Grid_XYZW[  600][2] =  0.036819172264396;
    Leb_Grid_XYZW[  600][3] =  0.000331604187320;

    Leb_Grid_XYZW[  601][0] =  0.125390157236712;
    Leb_Grid_XYZW[  601][1] = -0.991424055095456;
    Leb_Grid_XYZW[  601][2] = -0.036819172264397;
    Leb_Grid_XYZW[  601][3] =  0.000331604187320;

    Leb_Grid_XYZW[  602][0] = -0.125390157236712;
    Leb_Grid_XYZW[  602][1] =  0.991424055095456;
    Leb_Grid_XYZW[  602][2] =  0.036819172264396;
    Leb_Grid_XYZW[  602][3] =  0.000331604187320;

    Leb_Grid_XYZW[  603][0] = -0.125390157236712;
    Leb_Grid_XYZW[  603][1] =  0.991424055095456;
    Leb_Grid_XYZW[  603][2] = -0.036819172264397;
    Leb_Grid_XYZW[  603][3] =  0.000331604187320;

    Leb_Grid_XYZW[  604][0] = -0.125390157236712;
    Leb_Grid_XYZW[  604][1] = -0.991424055095456;
    Leb_Grid_XYZW[  604][2] =  0.036819172264396;
    Leb_Grid_XYZW[  604][3] =  0.000331604187320;

    Leb_Grid_XYZW[  605][0] = -0.125390157236712;
    Leb_Grid_XYZW[  605][1] = -0.991424055095456;
    Leb_Grid_XYZW[  605][2] = -0.036819172264397;
    Leb_Grid_XYZW[  605][3] =  0.000331604187320;

    Leb_Grid_XYZW[  606][0] =  0.036819172264396;
    Leb_Grid_XYZW[  606][1] =  0.125390157236712;
    Leb_Grid_XYZW[  606][2] =  0.991424055095456;
    Leb_Grid_XYZW[  606][3] =  0.000331604187320;

    Leb_Grid_XYZW[  607][0] =  0.036819172264396;
    Leb_Grid_XYZW[  607][1] =  0.125390157236712;
    Leb_Grid_XYZW[  607][2] = -0.991424055095456;
    Leb_Grid_XYZW[  607][3] =  0.000331604187320;

    Leb_Grid_XYZW[  608][0] =  0.036819172264396;
    Leb_Grid_XYZW[  608][1] = -0.125390157236712;
    Leb_Grid_XYZW[  608][2] =  0.991424055095456;
    Leb_Grid_XYZW[  608][3] =  0.000331604187320;

    Leb_Grid_XYZW[  609][0] =  0.036819172264396;
    Leb_Grid_XYZW[  609][1] = -0.125390157236712;
    Leb_Grid_XYZW[  609][2] = -0.991424055095456;
    Leb_Grid_XYZW[  609][3] =  0.000331604187320;

    Leb_Grid_XYZW[  610][0] = -0.036819172264396;
    Leb_Grid_XYZW[  610][1] =  0.125390157236712;
    Leb_Grid_XYZW[  610][2] =  0.991424055095456;
    Leb_Grid_XYZW[  610][3] =  0.000331604187320;

    Leb_Grid_XYZW[  611][0] = -0.036819172264396;
    Leb_Grid_XYZW[  611][1] =  0.125390157236712;
    Leb_Grid_XYZW[  611][2] = -0.991424055095456;
    Leb_Grid_XYZW[  611][3] =  0.000331604187320;

    Leb_Grid_XYZW[  612][0] = -0.036819172264396;
    Leb_Grid_XYZW[  612][1] = -0.125390157236712;
    Leb_Grid_XYZW[  612][2] =  0.991424055095456;
    Leb_Grid_XYZW[  612][3] =  0.000331604187320;

    Leb_Grid_XYZW[  613][0] = -0.036819172264396;
    Leb_Grid_XYZW[  613][1] = -0.125390157236712;
    Leb_Grid_XYZW[  613][2] = -0.991424055095456;
    Leb_Grid_XYZW[  613][3] =  0.000331604187320;

    Leb_Grid_XYZW[  614][0] =  0.036819172264397;
    Leb_Grid_XYZW[  614][1] =  0.991424055095456;
    Leb_Grid_XYZW[  614][2] =  0.125390157236712;
    Leb_Grid_XYZW[  614][3] =  0.000331604187320;

    Leb_Grid_XYZW[  615][0] =  0.036819172264397;
    Leb_Grid_XYZW[  615][1] =  0.991424055095456;
    Leb_Grid_XYZW[  615][2] = -0.125390157236712;
    Leb_Grid_XYZW[  615][3] =  0.000331604187320;

    Leb_Grid_XYZW[  616][0] =  0.036819172264397;
    Leb_Grid_XYZW[  616][1] = -0.991424055095456;
    Leb_Grid_XYZW[  616][2] =  0.125390157236712;
    Leb_Grid_XYZW[  616][3] =  0.000331604187320;

    Leb_Grid_XYZW[  617][0] =  0.036819172264397;
    Leb_Grid_XYZW[  617][1] = -0.991424055095456;
    Leb_Grid_XYZW[  617][2] = -0.125390157236712;
    Leb_Grid_XYZW[  617][3] =  0.000331604187320;

    Leb_Grid_XYZW[  618][0] = -0.036819172264397;
    Leb_Grid_XYZW[  618][1] =  0.991424055095456;
    Leb_Grid_XYZW[  618][2] =  0.125390157236712;
    Leb_Grid_XYZW[  618][3] =  0.000331604187320;

    Leb_Grid_XYZW[  619][0] = -0.036819172264397;
    Leb_Grid_XYZW[  619][1] =  0.991424055095456;
    Leb_Grid_XYZW[  619][2] = -0.125390157236712;
    Leb_Grid_XYZW[  619][3] =  0.000331604187320;

    Leb_Grid_XYZW[  620][0] = -0.036819172264397;
    Leb_Grid_XYZW[  620][1] = -0.991424055095456;
    Leb_Grid_XYZW[  620][2] =  0.125390157236712;
    Leb_Grid_XYZW[  620][3] =  0.000331604187320;

    Leb_Grid_XYZW[  621][0] = -0.036819172264397;
    Leb_Grid_XYZW[  621][1] = -0.991424055095456;
    Leb_Grid_XYZW[  621][2] = -0.125390157236712;
    Leb_Grid_XYZW[  621][3] =  0.000331604187320;

    Leb_Grid_XYZW[  622][0] =  0.991424055095456;
    Leb_Grid_XYZW[  622][1] =  0.125390157236712;
    Leb_Grid_XYZW[  622][2] =  0.036819172264396;
    Leb_Grid_XYZW[  622][3] =  0.000331604187320;

    Leb_Grid_XYZW[  623][0] =  0.991424055095456;
    Leb_Grid_XYZW[  623][1] =  0.125390157236712;
    Leb_Grid_XYZW[  623][2] = -0.036819172264397;
    Leb_Grid_XYZW[  623][3] =  0.000331604187320;

    Leb_Grid_XYZW[  624][0] =  0.991424055095456;
    Leb_Grid_XYZW[  624][1] = -0.125390157236712;
    Leb_Grid_XYZW[  624][2] =  0.036819172264396;
    Leb_Grid_XYZW[  624][3] =  0.000331604187320;

    Leb_Grid_XYZW[  625][0] =  0.991424055095456;
    Leb_Grid_XYZW[  625][1] = -0.125390157236712;
    Leb_Grid_XYZW[  625][2] = -0.036819172264397;
    Leb_Grid_XYZW[  625][3] =  0.000331604187320;

    Leb_Grid_XYZW[  626][0] = -0.991424055095456;
    Leb_Grid_XYZW[  626][1] =  0.125390157236712;
    Leb_Grid_XYZW[  626][2] =  0.036819172264396;
    Leb_Grid_XYZW[  626][3] =  0.000331604187320;

    Leb_Grid_XYZW[  627][0] = -0.991424055095456;
    Leb_Grid_XYZW[  627][1] =  0.125390157236712;
    Leb_Grid_XYZW[  627][2] = -0.036819172264397;
    Leb_Grid_XYZW[  627][3] =  0.000331604187320;

    Leb_Grid_XYZW[  628][0] = -0.991424055095456;
    Leb_Grid_XYZW[  628][1] = -0.125390157236712;
    Leb_Grid_XYZW[  628][2] =  0.036819172264396;
    Leb_Grid_XYZW[  628][3] =  0.000331604187320;

    Leb_Grid_XYZW[  629][0] = -0.991424055095456;
    Leb_Grid_XYZW[  629][1] = -0.125390157236712;
    Leb_Grid_XYZW[  629][2] = -0.036819172264397;
    Leb_Grid_XYZW[  629][3] =  0.000331604187320;

    Leb_Grid_XYZW[  630][0] =  0.991424055095456;
    Leb_Grid_XYZW[  630][1] =  0.036819172264397;
    Leb_Grid_XYZW[  630][2] =  0.125390157236712;
    Leb_Grid_XYZW[  630][3] =  0.000331604187320;

    Leb_Grid_XYZW[  631][0] =  0.991424055095456;
    Leb_Grid_XYZW[  631][1] =  0.036819172264397;
    Leb_Grid_XYZW[  631][2] = -0.125390157236712;
    Leb_Grid_XYZW[  631][3] =  0.000331604187320;

    Leb_Grid_XYZW[  632][0] =  0.991424055095456;
    Leb_Grid_XYZW[  632][1] = -0.036819172264397;
    Leb_Grid_XYZW[  632][2] =  0.125390157236712;
    Leb_Grid_XYZW[  632][3] =  0.000331604187320;

    Leb_Grid_XYZW[  633][0] =  0.991424055095456;
    Leb_Grid_XYZW[  633][1] = -0.036819172264397;
    Leb_Grid_XYZW[  633][2] = -0.125390157236712;
    Leb_Grid_XYZW[  633][3] =  0.000331604187320;

    Leb_Grid_XYZW[  634][0] = -0.991424055095456;
    Leb_Grid_XYZW[  634][1] =  0.036819172264397;
    Leb_Grid_XYZW[  634][2] =  0.125390157236712;
    Leb_Grid_XYZW[  634][3] =  0.000331604187320;

    Leb_Grid_XYZW[  635][0] = -0.991424055095456;
    Leb_Grid_XYZW[  635][1] =  0.036819172264397;
    Leb_Grid_XYZW[  635][2] = -0.125390157236712;
    Leb_Grid_XYZW[  635][3] =  0.000331604187320;

    Leb_Grid_XYZW[  636][0] = -0.991424055095456;
    Leb_Grid_XYZW[  636][1] = -0.036819172264397;
    Leb_Grid_XYZW[  636][2] =  0.125390157236712;
    Leb_Grid_XYZW[  636][3] =  0.000331604187320;

    Leb_Grid_XYZW[  637][0] = -0.991424055095456;
    Leb_Grid_XYZW[  637][1] = -0.036819172264397;
    Leb_Grid_XYZW[  637][2] = -0.125390157236712;
    Leb_Grid_XYZW[  637][3] =  0.000331604187320;

    Leb_Grid_XYZW[  638][0] =  0.177572151038394;
    Leb_Grid_XYZW[  638][1] =  0.079824876072133;
    Leb_Grid_XYZW[  638][2] =  0.980864985783297;
    Leb_Grid_XYZW[  638][3] =  0.000389911356715;

    Leb_Grid_XYZW[  639][0] =  0.177572151038394;
    Leb_Grid_XYZW[  639][1] =  0.079824876072133;
    Leb_Grid_XYZW[  639][2] = -0.980864985783297;
    Leb_Grid_XYZW[  639][3] =  0.000389911356715;

    Leb_Grid_XYZW[  640][0] =  0.177572151038394;
    Leb_Grid_XYZW[  640][1] = -0.079824876072133;
    Leb_Grid_XYZW[  640][2] =  0.980864985783297;
    Leb_Grid_XYZW[  640][3] =  0.000389911356715;

    Leb_Grid_XYZW[  641][0] =  0.177572151038394;
    Leb_Grid_XYZW[  641][1] = -0.079824876072133;
    Leb_Grid_XYZW[  641][2] = -0.980864985783297;
    Leb_Grid_XYZW[  641][3] =  0.000389911356715;

    Leb_Grid_XYZW[  642][0] = -0.177572151038394;
    Leb_Grid_XYZW[  642][1] =  0.079824876072133;
    Leb_Grid_XYZW[  642][2] =  0.980864985783297;
    Leb_Grid_XYZW[  642][3] =  0.000389911356715;

    Leb_Grid_XYZW[  643][0] = -0.177572151038394;
    Leb_Grid_XYZW[  643][1] =  0.079824876072133;
    Leb_Grid_XYZW[  643][2] = -0.980864985783297;
    Leb_Grid_XYZW[  643][3] =  0.000389911356715;

    Leb_Grid_XYZW[  644][0] = -0.177572151038394;
    Leb_Grid_XYZW[  644][1] = -0.079824876072133;
    Leb_Grid_XYZW[  644][2] =  0.980864985783297;
    Leb_Grid_XYZW[  644][3] =  0.000389911356715;

    Leb_Grid_XYZW[  645][0] = -0.177572151038394;
    Leb_Grid_XYZW[  645][1] = -0.079824876072133;
    Leb_Grid_XYZW[  645][2] = -0.980864985783297;
    Leb_Grid_XYZW[  645][3] =  0.000389911356715;

    Leb_Grid_XYZW[  646][0] =  0.177572151038394;
    Leb_Grid_XYZW[  646][1] =  0.980864985783296;
    Leb_Grid_XYZW[  646][2] =  0.079824876072133;
    Leb_Grid_XYZW[  646][3] =  0.000389911356715;

    Leb_Grid_XYZW[  647][0] =  0.177572151038394;
    Leb_Grid_XYZW[  647][1] =  0.980864985783296;
    Leb_Grid_XYZW[  647][2] = -0.079824876072133;
    Leb_Grid_XYZW[  647][3] =  0.000389911356715;

    Leb_Grid_XYZW[  648][0] =  0.177572151038394;
    Leb_Grid_XYZW[  648][1] = -0.980864985783296;
    Leb_Grid_XYZW[  648][2] =  0.079824876072133;
    Leb_Grid_XYZW[  648][3] =  0.000389911356715;

    Leb_Grid_XYZW[  649][0] =  0.177572151038394;
    Leb_Grid_XYZW[  649][1] = -0.980864985783296;
    Leb_Grid_XYZW[  649][2] = -0.079824876072133;
    Leb_Grid_XYZW[  649][3] =  0.000389911356715;

    Leb_Grid_XYZW[  650][0] = -0.177572151038394;
    Leb_Grid_XYZW[  650][1] =  0.980864985783296;
    Leb_Grid_XYZW[  650][2] =  0.079824876072133;
    Leb_Grid_XYZW[  650][3] =  0.000389911356715;

    Leb_Grid_XYZW[  651][0] = -0.177572151038394;
    Leb_Grid_XYZW[  651][1] =  0.980864985783296;
    Leb_Grid_XYZW[  651][2] = -0.079824876072133;
    Leb_Grid_XYZW[  651][3] =  0.000389911356715;

    Leb_Grid_XYZW[  652][0] = -0.177572151038394;
    Leb_Grid_XYZW[  652][1] = -0.980864985783296;
    Leb_Grid_XYZW[  652][2] =  0.079824876072133;
    Leb_Grid_XYZW[  652][3] =  0.000389911356715;

    Leb_Grid_XYZW[  653][0] = -0.177572151038394;
    Leb_Grid_XYZW[  653][1] = -0.980864985783296;
    Leb_Grid_XYZW[  653][2] = -0.079824876072133;
    Leb_Grid_XYZW[  653][3] =  0.000389911356715;

    Leb_Grid_XYZW[  654][0] =  0.079824876072133;
    Leb_Grid_XYZW[  654][1] =  0.177572151038394;
    Leb_Grid_XYZW[  654][2] =  0.980864985783297;
    Leb_Grid_XYZW[  654][3] =  0.000389911356715;

    Leb_Grid_XYZW[  655][0] =  0.079824876072133;
    Leb_Grid_XYZW[  655][1] =  0.177572151038394;
    Leb_Grid_XYZW[  655][2] = -0.980864985783297;
    Leb_Grid_XYZW[  655][3] =  0.000389911356715;

    Leb_Grid_XYZW[  656][0] =  0.079824876072133;
    Leb_Grid_XYZW[  656][1] = -0.177572151038394;
    Leb_Grid_XYZW[  656][2] =  0.980864985783297;
    Leb_Grid_XYZW[  656][3] =  0.000389911356715;

    Leb_Grid_XYZW[  657][0] =  0.079824876072133;
    Leb_Grid_XYZW[  657][1] = -0.177572151038394;
    Leb_Grid_XYZW[  657][2] = -0.980864985783297;
    Leb_Grid_XYZW[  657][3] =  0.000389911356715;

    Leb_Grid_XYZW[  658][0] = -0.079824876072133;
    Leb_Grid_XYZW[  658][1] =  0.177572151038394;
    Leb_Grid_XYZW[  658][2] =  0.980864985783297;
    Leb_Grid_XYZW[  658][3] =  0.000389911356715;

    Leb_Grid_XYZW[  659][0] = -0.079824876072133;
    Leb_Grid_XYZW[  659][1] =  0.177572151038394;
    Leb_Grid_XYZW[  659][2] = -0.980864985783297;
    Leb_Grid_XYZW[  659][3] =  0.000389911356715;

    Leb_Grid_XYZW[  660][0] = -0.079824876072133;
    Leb_Grid_XYZW[  660][1] = -0.177572151038394;
    Leb_Grid_XYZW[  660][2] =  0.980864985783297;
    Leb_Grid_XYZW[  660][3] =  0.000389911356715;

    Leb_Grid_XYZW[  661][0] = -0.079824876072133;
    Leb_Grid_XYZW[  661][1] = -0.177572151038394;
    Leb_Grid_XYZW[  661][2] = -0.980864985783297;
    Leb_Grid_XYZW[  661][3] =  0.000389911356715;

    Leb_Grid_XYZW[  662][0] =  0.079824876072133;
    Leb_Grid_XYZW[  662][1] =  0.980864985783297;
    Leb_Grid_XYZW[  662][2] =  0.177572151038394;
    Leb_Grid_XYZW[  662][3] =  0.000389911356715;

    Leb_Grid_XYZW[  663][0] =  0.079824876072133;
    Leb_Grid_XYZW[  663][1] =  0.980864985783296;
    Leb_Grid_XYZW[  663][2] = -0.177572151038394;
    Leb_Grid_XYZW[  663][3] =  0.000389911356715;

    Leb_Grid_XYZW[  664][0] =  0.079824876072133;
    Leb_Grid_XYZW[  664][1] = -0.980864985783297;
    Leb_Grid_XYZW[  664][2] =  0.177572151038394;
    Leb_Grid_XYZW[  664][3] =  0.000389911356715;

    Leb_Grid_XYZW[  665][0] =  0.079824876072133;
    Leb_Grid_XYZW[  665][1] = -0.980864985783296;
    Leb_Grid_XYZW[  665][2] = -0.177572151038394;
    Leb_Grid_XYZW[  665][3] =  0.000389911356715;

    Leb_Grid_XYZW[  666][0] = -0.079824876072133;
    Leb_Grid_XYZW[  666][1] =  0.980864985783297;
    Leb_Grid_XYZW[  666][2] =  0.177572151038394;
    Leb_Grid_XYZW[  666][3] =  0.000389911356715;

    Leb_Grid_XYZW[  667][0] = -0.079824876072133;
    Leb_Grid_XYZW[  667][1] =  0.980864985783296;
    Leb_Grid_XYZW[  667][2] = -0.177572151038394;
    Leb_Grid_XYZW[  667][3] =  0.000389911356715;

    Leb_Grid_XYZW[  668][0] = -0.079824876072133;
    Leb_Grid_XYZW[  668][1] = -0.980864985783297;
    Leb_Grid_XYZW[  668][2] =  0.177572151038394;
    Leb_Grid_XYZW[  668][3] =  0.000389911356715;

    Leb_Grid_XYZW[  669][0] = -0.079824876072133;
    Leb_Grid_XYZW[  669][1] = -0.980864985783296;
    Leb_Grid_XYZW[  669][2] = -0.177572151038394;
    Leb_Grid_XYZW[  669][3] =  0.000389911356715;

    Leb_Grid_XYZW[  670][0] =  0.980864985783296;
    Leb_Grid_XYZW[  670][1] =  0.177572151038394;
    Leb_Grid_XYZW[  670][2] =  0.079824876072133;
    Leb_Grid_XYZW[  670][3] =  0.000389911356715;

    Leb_Grid_XYZW[  671][0] =  0.980864985783296;
    Leb_Grid_XYZW[  671][1] =  0.177572151038394;
    Leb_Grid_XYZW[  671][2] = -0.079824876072133;
    Leb_Grid_XYZW[  671][3] =  0.000389911356715;

    Leb_Grid_XYZW[  672][0] =  0.980864985783296;
    Leb_Grid_XYZW[  672][1] = -0.177572151038394;
    Leb_Grid_XYZW[  672][2] =  0.079824876072133;
    Leb_Grid_XYZW[  672][3] =  0.000389911356715;

    Leb_Grid_XYZW[  673][0] =  0.980864985783296;
    Leb_Grid_XYZW[  673][1] = -0.177572151038394;
    Leb_Grid_XYZW[  673][2] = -0.079824876072133;
    Leb_Grid_XYZW[  673][3] =  0.000389911356715;

    Leb_Grid_XYZW[  674][0] = -0.980864985783296;
    Leb_Grid_XYZW[  674][1] =  0.177572151038394;
    Leb_Grid_XYZW[  674][2] =  0.079824876072133;
    Leb_Grid_XYZW[  674][3] =  0.000389911356715;

    Leb_Grid_XYZW[  675][0] = -0.980864985783296;
    Leb_Grid_XYZW[  675][1] =  0.177572151038394;
    Leb_Grid_XYZW[  675][2] = -0.079824876072133;
    Leb_Grid_XYZW[  675][3] =  0.000389911356715;

    Leb_Grid_XYZW[  676][0] = -0.980864985783296;
    Leb_Grid_XYZW[  676][1] = -0.177572151038394;
    Leb_Grid_XYZW[  676][2] =  0.079824876072133;
    Leb_Grid_XYZW[  676][3] =  0.000389911356715;

    Leb_Grid_XYZW[  677][0] = -0.980864985783296;
    Leb_Grid_XYZW[  677][1] = -0.177572151038394;
    Leb_Grid_XYZW[  677][2] = -0.079824876072133;
    Leb_Grid_XYZW[  677][3] =  0.000389911356715;

    Leb_Grid_XYZW[  678][0] =  0.980864985783297;
    Leb_Grid_XYZW[  678][1] =  0.079824876072134;
    Leb_Grid_XYZW[  678][2] =  0.177572151038394;
    Leb_Grid_XYZW[  678][3] =  0.000389911356715;

    Leb_Grid_XYZW[  679][0] =  0.980864985783296;
    Leb_Grid_XYZW[  679][1] =  0.079824876072134;
    Leb_Grid_XYZW[  679][2] = -0.177572151038394;
    Leb_Grid_XYZW[  679][3] =  0.000389911356715;

    Leb_Grid_XYZW[  680][0] =  0.980864985783297;
    Leb_Grid_XYZW[  680][1] = -0.079824876072134;
    Leb_Grid_XYZW[  680][2] =  0.177572151038394;
    Leb_Grid_XYZW[  680][3] =  0.000389911356715;

    Leb_Grid_XYZW[  681][0] =  0.980864985783296;
    Leb_Grid_XYZW[  681][1] = -0.079824876072134;
    Leb_Grid_XYZW[  681][2] = -0.177572151038394;
    Leb_Grid_XYZW[  681][3] =  0.000389911356715;

    Leb_Grid_XYZW[  682][0] = -0.980864985783297;
    Leb_Grid_XYZW[  682][1] =  0.079824876072134;
    Leb_Grid_XYZW[  682][2] =  0.177572151038394;
    Leb_Grid_XYZW[  682][3] =  0.000389911356715;

    Leb_Grid_XYZW[  683][0] = -0.980864985783296;
    Leb_Grid_XYZW[  683][1] =  0.079824876072134;
    Leb_Grid_XYZW[  683][2] = -0.177572151038394;
    Leb_Grid_XYZW[  683][3] =  0.000389911356715;

    Leb_Grid_XYZW[  684][0] = -0.980864985783297;
    Leb_Grid_XYZW[  684][1] = -0.079824876072134;
    Leb_Grid_XYZW[  684][2] =  0.177572151038394;
    Leb_Grid_XYZW[  684][3] =  0.000389911356715;

    Leb_Grid_XYZW[  685][0] = -0.980864985783296;
    Leb_Grid_XYZW[  685][1] = -0.079824876072134;
    Leb_Grid_XYZW[  685][2] = -0.177572151038394;
    Leb_Grid_XYZW[  685][3] =  0.000389911356715;

    Leb_Grid_XYZW[  686][0] =  0.230569335821611;
    Leb_Grid_XYZW[  686][1] =  0.126464096659233;
    Leb_Grid_XYZW[  686][2] =  0.964802888488081;
    Leb_Grid_XYZW[  686][3] =  0.000434334332720;

    Leb_Grid_XYZW[  687][0] =  0.230569335821611;
    Leb_Grid_XYZW[  687][1] =  0.126464096659233;
    Leb_Grid_XYZW[  687][2] = -0.964802888488081;
    Leb_Grid_XYZW[  687][3] =  0.000434334332720;

    Leb_Grid_XYZW[  688][0] =  0.230569335821611;
    Leb_Grid_XYZW[  688][1] = -0.126464096659233;
    Leb_Grid_XYZW[  688][2] =  0.964802888488081;
    Leb_Grid_XYZW[  688][3] =  0.000434334332720;

    Leb_Grid_XYZW[  689][0] =  0.230569335821611;
    Leb_Grid_XYZW[  689][1] = -0.126464096659233;
    Leb_Grid_XYZW[  689][2] = -0.964802888488081;
    Leb_Grid_XYZW[  689][3] =  0.000434334332720;

    Leb_Grid_XYZW[  690][0] = -0.230569335821611;
    Leb_Grid_XYZW[  690][1] =  0.126464096659233;
    Leb_Grid_XYZW[  690][2] =  0.964802888488081;
    Leb_Grid_XYZW[  690][3] =  0.000434334332720;

    Leb_Grid_XYZW[  691][0] = -0.230569335821611;
    Leb_Grid_XYZW[  691][1] =  0.126464096659233;
    Leb_Grid_XYZW[  691][2] = -0.964802888488081;
    Leb_Grid_XYZW[  691][3] =  0.000434334332720;

    Leb_Grid_XYZW[  692][0] = -0.230569335821611;
    Leb_Grid_XYZW[  692][1] = -0.126464096659233;
    Leb_Grid_XYZW[  692][2] =  0.964802888488081;
    Leb_Grid_XYZW[  692][3] =  0.000434334332720;

    Leb_Grid_XYZW[  693][0] = -0.230569335821611;
    Leb_Grid_XYZW[  693][1] = -0.126464096659233;
    Leb_Grid_XYZW[  693][2] = -0.964802888488081;
    Leb_Grid_XYZW[  693][3] =  0.000434334332720;

    Leb_Grid_XYZW[  694][0] =  0.230569335821611;
    Leb_Grid_XYZW[  694][1] =  0.964802888488081;
    Leb_Grid_XYZW[  694][2] =  0.126464096659233;
    Leb_Grid_XYZW[  694][3] =  0.000434334332720;

    Leb_Grid_XYZW[  695][0] =  0.230569335821611;
    Leb_Grid_XYZW[  695][1] =  0.964802888488081;
    Leb_Grid_XYZW[  695][2] = -0.126464096659233;
    Leb_Grid_XYZW[  695][3] =  0.000434334332720;

    Leb_Grid_XYZW[  696][0] =  0.230569335821611;
    Leb_Grid_XYZW[  696][1] = -0.964802888488081;
    Leb_Grid_XYZW[  696][2] =  0.126464096659233;
    Leb_Grid_XYZW[  696][3] =  0.000434334332720;

    Leb_Grid_XYZW[  697][0] =  0.230569335821611;
    Leb_Grid_XYZW[  697][1] = -0.964802888488081;
    Leb_Grid_XYZW[  697][2] = -0.126464096659233;
    Leb_Grid_XYZW[  697][3] =  0.000434334332720;

    Leb_Grid_XYZW[  698][0] = -0.230569335821611;
    Leb_Grid_XYZW[  698][1] =  0.964802888488081;
    Leb_Grid_XYZW[  698][2] =  0.126464096659233;
    Leb_Grid_XYZW[  698][3] =  0.000434334332720;

    Leb_Grid_XYZW[  699][0] = -0.230569335821611;
    Leb_Grid_XYZW[  699][1] =  0.964802888488081;
    Leb_Grid_XYZW[  699][2] = -0.126464096659233;
    Leb_Grid_XYZW[  699][3] =  0.000434334332720;

    Leb_Grid_XYZW[  700][0] = -0.230569335821611;
    Leb_Grid_XYZW[  700][1] = -0.964802888488081;
    Leb_Grid_XYZW[  700][2] =  0.126464096659233;
    Leb_Grid_XYZW[  700][3] =  0.000434334332720;

    Leb_Grid_XYZW[  701][0] = -0.230569335821611;
    Leb_Grid_XYZW[  701][1] = -0.964802888488081;
    Leb_Grid_XYZW[  701][2] = -0.126464096659233;
    Leb_Grid_XYZW[  701][3] =  0.000434334332720;

    Leb_Grid_XYZW[  702][0] =  0.126464096659233;
    Leb_Grid_XYZW[  702][1] =  0.230569335821611;
    Leb_Grid_XYZW[  702][2] =  0.964802888488081;
    Leb_Grid_XYZW[  702][3] =  0.000434334332720;

    Leb_Grid_XYZW[  703][0] =  0.126464096659233;
    Leb_Grid_XYZW[  703][1] =  0.230569335821611;
    Leb_Grid_XYZW[  703][2] = -0.964802888488081;
    Leb_Grid_XYZW[  703][3] =  0.000434334332720;

    Leb_Grid_XYZW[  704][0] =  0.126464096659233;
    Leb_Grid_XYZW[  704][1] = -0.230569335821611;
    Leb_Grid_XYZW[  704][2] =  0.964802888488081;
    Leb_Grid_XYZW[  704][3] =  0.000434334332720;

    Leb_Grid_XYZW[  705][0] =  0.126464096659233;
    Leb_Grid_XYZW[  705][1] = -0.230569335821611;
    Leb_Grid_XYZW[  705][2] = -0.964802888488081;
    Leb_Grid_XYZW[  705][3] =  0.000434334332720;

    Leb_Grid_XYZW[  706][0] = -0.126464096659233;
    Leb_Grid_XYZW[  706][1] =  0.230569335821611;
    Leb_Grid_XYZW[  706][2] =  0.964802888488081;
    Leb_Grid_XYZW[  706][3] =  0.000434334332720;

    Leb_Grid_XYZW[  707][0] = -0.126464096659233;
    Leb_Grid_XYZW[  707][1] =  0.230569335821611;
    Leb_Grid_XYZW[  707][2] = -0.964802888488081;
    Leb_Grid_XYZW[  707][3] =  0.000434334332720;

    Leb_Grid_XYZW[  708][0] = -0.126464096659233;
    Leb_Grid_XYZW[  708][1] = -0.230569335821611;
    Leb_Grid_XYZW[  708][2] =  0.964802888488081;
    Leb_Grid_XYZW[  708][3] =  0.000434334332720;

    Leb_Grid_XYZW[  709][0] = -0.126464096659233;
    Leb_Grid_XYZW[  709][1] = -0.230569335821611;
    Leb_Grid_XYZW[  709][2] = -0.964802888488081;
    Leb_Grid_XYZW[  709][3] =  0.000434334332720;

    Leb_Grid_XYZW[  710][0] =  0.126464096659233;
    Leb_Grid_XYZW[  710][1] =  0.964802888488081;
    Leb_Grid_XYZW[  710][2] =  0.230569335821611;
    Leb_Grid_XYZW[  710][3] =  0.000434334332720;

    Leb_Grid_XYZW[  711][0] =  0.126464096659233;
    Leb_Grid_XYZW[  711][1] =  0.964802888488081;
    Leb_Grid_XYZW[  711][2] = -0.230569335821611;
    Leb_Grid_XYZW[  711][3] =  0.000434334332720;

    Leb_Grid_XYZW[  712][0] =  0.126464096659233;
    Leb_Grid_XYZW[  712][1] = -0.964802888488081;
    Leb_Grid_XYZW[  712][2] =  0.230569335821611;
    Leb_Grid_XYZW[  712][3] =  0.000434334332720;

    Leb_Grid_XYZW[  713][0] =  0.126464096659233;
    Leb_Grid_XYZW[  713][1] = -0.964802888488081;
    Leb_Grid_XYZW[  713][2] = -0.230569335821611;
    Leb_Grid_XYZW[  713][3] =  0.000434334332720;

    Leb_Grid_XYZW[  714][0] = -0.126464096659233;
    Leb_Grid_XYZW[  714][1] =  0.964802888488081;
    Leb_Grid_XYZW[  714][2] =  0.230569335821611;
    Leb_Grid_XYZW[  714][3] =  0.000434334332720;

    Leb_Grid_XYZW[  715][0] = -0.126464096659233;
    Leb_Grid_XYZW[  715][1] =  0.964802888488081;
    Leb_Grid_XYZW[  715][2] = -0.230569335821611;
    Leb_Grid_XYZW[  715][3] =  0.000434334332720;

    Leb_Grid_XYZW[  716][0] = -0.126464096659233;
    Leb_Grid_XYZW[  716][1] = -0.964802888488081;
    Leb_Grid_XYZW[  716][2] =  0.230569335821611;
    Leb_Grid_XYZW[  716][3] =  0.000434334332720;

    Leb_Grid_XYZW[  717][0] = -0.126464096659233;
    Leb_Grid_XYZW[  717][1] = -0.964802888488081;
    Leb_Grid_XYZW[  717][2] = -0.230569335821611;
    Leb_Grid_XYZW[  717][3] =  0.000434334332720;

    Leb_Grid_XYZW[  718][0] =  0.964802888488081;
    Leb_Grid_XYZW[  718][1] =  0.230569335821612;
    Leb_Grid_XYZW[  718][2] =  0.126464096659233;
    Leb_Grid_XYZW[  718][3] =  0.000434334332720;

    Leb_Grid_XYZW[  719][0] =  0.964802888488081;
    Leb_Grid_XYZW[  719][1] =  0.230569335821612;
    Leb_Grid_XYZW[  719][2] = -0.126464096659233;
    Leb_Grid_XYZW[  719][3] =  0.000434334332720;

    Leb_Grid_XYZW[  720][0] =  0.964802888488081;
    Leb_Grid_XYZW[  720][1] = -0.230569335821612;
    Leb_Grid_XYZW[  720][2] =  0.126464096659233;
    Leb_Grid_XYZW[  720][3] =  0.000434334332720;

    Leb_Grid_XYZW[  721][0] =  0.964802888488081;
    Leb_Grid_XYZW[  721][1] = -0.230569335821612;
    Leb_Grid_XYZW[  721][2] = -0.126464096659233;
    Leb_Grid_XYZW[  721][3] =  0.000434334332720;

    Leb_Grid_XYZW[  722][0] = -0.964802888488081;
    Leb_Grid_XYZW[  722][1] =  0.230569335821611;
    Leb_Grid_XYZW[  722][2] =  0.126464096659233;
    Leb_Grid_XYZW[  722][3] =  0.000434334332720;

    Leb_Grid_XYZW[  723][0] = -0.964802888488081;
    Leb_Grid_XYZW[  723][1] =  0.230569335821611;
    Leb_Grid_XYZW[  723][2] = -0.126464096659233;
    Leb_Grid_XYZW[  723][3] =  0.000434334332720;

    Leb_Grid_XYZW[  724][0] = -0.964802888488081;
    Leb_Grid_XYZW[  724][1] = -0.230569335821611;
    Leb_Grid_XYZW[  724][2] =  0.126464096659233;
    Leb_Grid_XYZW[  724][3] =  0.000434334332720;

    Leb_Grid_XYZW[  725][0] = -0.964802888488081;
    Leb_Grid_XYZW[  725][1] = -0.230569335821611;
    Leb_Grid_XYZW[  725][2] = -0.126464096659233;
    Leb_Grid_XYZW[  725][3] =  0.000434334332720;

    Leb_Grid_XYZW[  726][0] =  0.964802888488081;
    Leb_Grid_XYZW[  726][1] =  0.126464096659234;
    Leb_Grid_XYZW[  726][2] =  0.230569335821611;
    Leb_Grid_XYZW[  726][3] =  0.000434334332720;

    Leb_Grid_XYZW[  727][0] =  0.964802888488081;
    Leb_Grid_XYZW[  727][1] =  0.126464096659234;
    Leb_Grid_XYZW[  727][2] = -0.230569335821611;
    Leb_Grid_XYZW[  727][3] =  0.000434334332720;

    Leb_Grid_XYZW[  728][0] =  0.964802888488081;
    Leb_Grid_XYZW[  728][1] = -0.126464096659234;
    Leb_Grid_XYZW[  728][2] =  0.230569335821611;
    Leb_Grid_XYZW[  728][3] =  0.000434334332720;

    Leb_Grid_XYZW[  729][0] =  0.964802888488081;
    Leb_Grid_XYZW[  729][1] = -0.126464096659234;
    Leb_Grid_XYZW[  729][2] = -0.230569335821611;
    Leb_Grid_XYZW[  729][3] =  0.000434334332720;

    Leb_Grid_XYZW[  730][0] = -0.964802888488081;
    Leb_Grid_XYZW[  730][1] =  0.126464096659234;
    Leb_Grid_XYZW[  730][2] =  0.230569335821611;
    Leb_Grid_XYZW[  730][3] =  0.000434334332720;

    Leb_Grid_XYZW[  731][0] = -0.964802888488081;
    Leb_Grid_XYZW[  731][1] =  0.126464096659234;
    Leb_Grid_XYZW[  731][2] = -0.230569335821611;
    Leb_Grid_XYZW[  731][3] =  0.000434334332720;

    Leb_Grid_XYZW[  732][0] = -0.964802888488081;
    Leb_Grid_XYZW[  732][1] = -0.126464096659234;
    Leb_Grid_XYZW[  732][2] =  0.230569335821611;
    Leb_Grid_XYZW[  732][3] =  0.000434334332720;

    Leb_Grid_XYZW[  733][0] = -0.964802888488081;
    Leb_Grid_XYZW[  733][1] = -0.126464096659234;
    Leb_Grid_XYZW[  733][2] = -0.230569335821611;
    Leb_Grid_XYZW[  733][3] =  0.000434334332720;

    Leb_Grid_XYZW[  734][0] =  0.283650284599206;
    Leb_Grid_XYZW[  734][1] =  0.175158568341896;
    Leb_Grid_XYZW[  734][2] =  0.942794777235856;
    Leb_Grid_XYZW[  734][3] =  0.000467941526232;

    Leb_Grid_XYZW[  735][0] =  0.283650284599206;
    Leb_Grid_XYZW[  735][1] =  0.175158568341896;
    Leb_Grid_XYZW[  735][2] = -0.942794777235856;
    Leb_Grid_XYZW[  735][3] =  0.000467941526232;

    Leb_Grid_XYZW[  736][0] =  0.283650284599206;
    Leb_Grid_XYZW[  736][1] = -0.175158568341896;
    Leb_Grid_XYZW[  736][2] =  0.942794777235856;
    Leb_Grid_XYZW[  736][3] =  0.000467941526232;

    Leb_Grid_XYZW[  737][0] =  0.283650284599206;
    Leb_Grid_XYZW[  737][1] = -0.175158568341896;
    Leb_Grid_XYZW[  737][2] = -0.942794777235856;
    Leb_Grid_XYZW[  737][3] =  0.000467941526232;

    Leb_Grid_XYZW[  738][0] = -0.283650284599206;
    Leb_Grid_XYZW[  738][1] =  0.175158568341896;
    Leb_Grid_XYZW[  738][2] =  0.942794777235856;
    Leb_Grid_XYZW[  738][3] =  0.000467941526232;

    Leb_Grid_XYZW[  739][0] = -0.283650284599206;
    Leb_Grid_XYZW[  739][1] =  0.175158568341896;
    Leb_Grid_XYZW[  739][2] = -0.942794777235856;
    Leb_Grid_XYZW[  739][3] =  0.000467941526232;

    Leb_Grid_XYZW[  740][0] = -0.283650284599206;
    Leb_Grid_XYZW[  740][1] = -0.175158568341896;
    Leb_Grid_XYZW[  740][2] =  0.942794777235856;
    Leb_Grid_XYZW[  740][3] =  0.000467941526232;

    Leb_Grid_XYZW[  741][0] = -0.283650284599206;
    Leb_Grid_XYZW[  741][1] = -0.175158568341896;
    Leb_Grid_XYZW[  741][2] = -0.942794777235856;
    Leb_Grid_XYZW[  741][3] =  0.000467941526232;

    Leb_Grid_XYZW[  742][0] =  0.283650284599206;
    Leb_Grid_XYZW[  742][1] =  0.942794777235856;
    Leb_Grid_XYZW[  742][2] =  0.175158568341896;
    Leb_Grid_XYZW[  742][3] =  0.000467941526232;

    Leb_Grid_XYZW[  743][0] =  0.283650284599206;
    Leb_Grid_XYZW[  743][1] =  0.942794777235856;
    Leb_Grid_XYZW[  743][2] = -0.175158568341896;
    Leb_Grid_XYZW[  743][3] =  0.000467941526232;

    Leb_Grid_XYZW[  744][0] =  0.283650284599206;
    Leb_Grid_XYZW[  744][1] = -0.942794777235856;
    Leb_Grid_XYZW[  744][2] =  0.175158568341896;
    Leb_Grid_XYZW[  744][3] =  0.000467941526232;

    Leb_Grid_XYZW[  745][0] =  0.283650284599206;
    Leb_Grid_XYZW[  745][1] = -0.942794777235856;
    Leb_Grid_XYZW[  745][2] = -0.175158568341896;
    Leb_Grid_XYZW[  745][3] =  0.000467941526232;

    Leb_Grid_XYZW[  746][0] = -0.283650284599206;
    Leb_Grid_XYZW[  746][1] =  0.942794777235856;
    Leb_Grid_XYZW[  746][2] =  0.175158568341896;
    Leb_Grid_XYZW[  746][3] =  0.000467941526232;

    Leb_Grid_XYZW[  747][0] = -0.283650284599206;
    Leb_Grid_XYZW[  747][1] =  0.942794777235856;
    Leb_Grid_XYZW[  747][2] = -0.175158568341896;
    Leb_Grid_XYZW[  747][3] =  0.000467941526232;

    Leb_Grid_XYZW[  748][0] = -0.283650284599206;
    Leb_Grid_XYZW[  748][1] = -0.942794777235856;
    Leb_Grid_XYZW[  748][2] =  0.175158568341896;
    Leb_Grid_XYZW[  748][3] =  0.000467941526232;

    Leb_Grid_XYZW[  749][0] = -0.283650284599206;
    Leb_Grid_XYZW[  749][1] = -0.942794777235856;
    Leb_Grid_XYZW[  749][2] = -0.175158568341896;
    Leb_Grid_XYZW[  749][3] =  0.000467941526232;

    Leb_Grid_XYZW[  750][0] =  0.175158568341896;
    Leb_Grid_XYZW[  750][1] =  0.283650284599206;
    Leb_Grid_XYZW[  750][2] =  0.942794777235856;
    Leb_Grid_XYZW[  750][3] =  0.000467941526232;

    Leb_Grid_XYZW[  751][0] =  0.175158568341896;
    Leb_Grid_XYZW[  751][1] =  0.283650284599206;
    Leb_Grid_XYZW[  751][2] = -0.942794777235856;
    Leb_Grid_XYZW[  751][3] =  0.000467941526232;

    Leb_Grid_XYZW[  752][0] =  0.175158568341896;
    Leb_Grid_XYZW[  752][1] = -0.283650284599206;
    Leb_Grid_XYZW[  752][2] =  0.942794777235856;
    Leb_Grid_XYZW[  752][3] =  0.000467941526232;

    Leb_Grid_XYZW[  753][0] =  0.175158568341896;
    Leb_Grid_XYZW[  753][1] = -0.283650284599206;
    Leb_Grid_XYZW[  753][2] = -0.942794777235856;
    Leb_Grid_XYZW[  753][3] =  0.000467941526232;

    Leb_Grid_XYZW[  754][0] = -0.175158568341896;
    Leb_Grid_XYZW[  754][1] =  0.283650284599206;
    Leb_Grid_XYZW[  754][2] =  0.942794777235856;
    Leb_Grid_XYZW[  754][3] =  0.000467941526232;

    Leb_Grid_XYZW[  755][0] = -0.175158568341896;
    Leb_Grid_XYZW[  755][1] =  0.283650284599206;
    Leb_Grid_XYZW[  755][2] = -0.942794777235856;
    Leb_Grid_XYZW[  755][3] =  0.000467941526232;

    Leb_Grid_XYZW[  756][0] = -0.175158568341896;
    Leb_Grid_XYZW[  756][1] = -0.283650284599206;
    Leb_Grid_XYZW[  756][2] =  0.942794777235856;
    Leb_Grid_XYZW[  756][3] =  0.000467941526232;

    Leb_Grid_XYZW[  757][0] = -0.175158568341896;
    Leb_Grid_XYZW[  757][1] = -0.283650284599206;
    Leb_Grid_XYZW[  757][2] = -0.942794777235856;
    Leb_Grid_XYZW[  757][3] =  0.000467941526232;

    Leb_Grid_XYZW[  758][0] =  0.175158568341896;
    Leb_Grid_XYZW[  758][1] =  0.942794777235856;
    Leb_Grid_XYZW[  758][2] =  0.283650284599206;
    Leb_Grid_XYZW[  758][3] =  0.000467941526232;

    Leb_Grid_XYZW[  759][0] =  0.175158568341896;
    Leb_Grid_XYZW[  759][1] =  0.942794777235856;
    Leb_Grid_XYZW[  759][2] = -0.283650284599206;
    Leb_Grid_XYZW[  759][3] =  0.000467941526232;

    Leb_Grid_XYZW[  760][0] =  0.175158568341896;
    Leb_Grid_XYZW[  760][1] = -0.942794777235856;
    Leb_Grid_XYZW[  760][2] =  0.283650284599206;
    Leb_Grid_XYZW[  760][3] =  0.000467941526232;

    Leb_Grid_XYZW[  761][0] =  0.175158568341896;
    Leb_Grid_XYZW[  761][1] = -0.942794777235856;
    Leb_Grid_XYZW[  761][2] = -0.283650284599206;
    Leb_Grid_XYZW[  761][3] =  0.000467941526232;

    Leb_Grid_XYZW[  762][0] = -0.175158568341896;
    Leb_Grid_XYZW[  762][1] =  0.942794777235856;
    Leb_Grid_XYZW[  762][2] =  0.283650284599206;
    Leb_Grid_XYZW[  762][3] =  0.000467941526232;

    Leb_Grid_XYZW[  763][0] = -0.175158568341896;
    Leb_Grid_XYZW[  763][1] =  0.942794777235856;
    Leb_Grid_XYZW[  763][2] = -0.283650284599206;
    Leb_Grid_XYZW[  763][3] =  0.000467941526232;

    Leb_Grid_XYZW[  764][0] = -0.175158568341896;
    Leb_Grid_XYZW[  764][1] = -0.942794777235856;
    Leb_Grid_XYZW[  764][2] =  0.283650284599206;
    Leb_Grid_XYZW[  764][3] =  0.000467941526232;

    Leb_Grid_XYZW[  765][0] = -0.175158568341896;
    Leb_Grid_XYZW[  765][1] = -0.942794777235856;
    Leb_Grid_XYZW[  765][2] = -0.283650284599206;
    Leb_Grid_XYZW[  765][3] =  0.000467941526232;

    Leb_Grid_XYZW[  766][0] =  0.942794777235856;
    Leb_Grid_XYZW[  766][1] =  0.283650284599206;
    Leb_Grid_XYZW[  766][2] =  0.175158568341896;
    Leb_Grid_XYZW[  766][3] =  0.000467941526232;

    Leb_Grid_XYZW[  767][0] =  0.942794777235856;
    Leb_Grid_XYZW[  767][1] =  0.283650284599206;
    Leb_Grid_XYZW[  767][2] = -0.175158568341896;
    Leb_Grid_XYZW[  767][3] =  0.000467941526232;

    Leb_Grid_XYZW[  768][0] =  0.942794777235856;
    Leb_Grid_XYZW[  768][1] = -0.283650284599206;
    Leb_Grid_XYZW[  768][2] =  0.175158568341896;
    Leb_Grid_XYZW[  768][3] =  0.000467941526232;

    Leb_Grid_XYZW[  769][0] =  0.942794777235856;
    Leb_Grid_XYZW[  769][1] = -0.283650284599206;
    Leb_Grid_XYZW[  769][2] = -0.175158568341896;
    Leb_Grid_XYZW[  769][3] =  0.000467941526232;

    Leb_Grid_XYZW[  770][0] = -0.942794777235856;
    Leb_Grid_XYZW[  770][1] =  0.283650284599206;
    Leb_Grid_XYZW[  770][2] =  0.175158568341896;
    Leb_Grid_XYZW[  770][3] =  0.000467941526232;

    Leb_Grid_XYZW[  771][0] = -0.942794777235856;
    Leb_Grid_XYZW[  771][1] =  0.283650284599206;
    Leb_Grid_XYZW[  771][2] = -0.175158568341896;
    Leb_Grid_XYZW[  771][3] =  0.000467941526232;

    Leb_Grid_XYZW[  772][0] = -0.942794777235856;
    Leb_Grid_XYZW[  772][1] = -0.283650284599206;
    Leb_Grid_XYZW[  772][2] =  0.175158568341896;
    Leb_Grid_XYZW[  772][3] =  0.000467941526232;

    Leb_Grid_XYZW[  773][0] = -0.942794777235856;
    Leb_Grid_XYZW[  773][1] = -0.283650284599206;
    Leb_Grid_XYZW[  773][2] = -0.175158568341896;
    Leb_Grid_XYZW[  773][3] =  0.000467941526232;

    Leb_Grid_XYZW[  774][0] =  0.942794777235856;
    Leb_Grid_XYZW[  774][1] =  0.175158568341895;
    Leb_Grid_XYZW[  774][2] =  0.283650284599206;
    Leb_Grid_XYZW[  774][3] =  0.000467941526232;

    Leb_Grid_XYZW[  775][0] =  0.942794777235856;
    Leb_Grid_XYZW[  775][1] =  0.175158568341895;
    Leb_Grid_XYZW[  775][2] = -0.283650284599206;
    Leb_Grid_XYZW[  775][3] =  0.000467941526232;

    Leb_Grid_XYZW[  776][0] =  0.942794777235856;
    Leb_Grid_XYZW[  776][1] = -0.175158568341895;
    Leb_Grid_XYZW[  776][2] =  0.283650284599206;
    Leb_Grid_XYZW[  776][3] =  0.000467941526232;

    Leb_Grid_XYZW[  777][0] =  0.942794777235856;
    Leb_Grid_XYZW[  777][1] = -0.175158568341895;
    Leb_Grid_XYZW[  777][2] = -0.283650284599206;
    Leb_Grid_XYZW[  777][3] =  0.000467941526232;

    Leb_Grid_XYZW[  778][0] = -0.942794777235856;
    Leb_Grid_XYZW[  778][1] =  0.175158568341896;
    Leb_Grid_XYZW[  778][2] =  0.283650284599206;
    Leb_Grid_XYZW[  778][3] =  0.000467941526232;

    Leb_Grid_XYZW[  779][0] = -0.942794777235856;
    Leb_Grid_XYZW[  779][1] =  0.175158568341896;
    Leb_Grid_XYZW[  779][2] = -0.283650284599206;
    Leb_Grid_XYZW[  779][3] =  0.000467941526232;

    Leb_Grid_XYZW[  780][0] = -0.942794777235856;
    Leb_Grid_XYZW[  780][1] = -0.175158568341896;
    Leb_Grid_XYZW[  780][2] =  0.283650284599206;
    Leb_Grid_XYZW[  780][3] =  0.000467941526232;

    Leb_Grid_XYZW[  781][0] = -0.942794777235856;
    Leb_Grid_XYZW[  781][1] = -0.175158568341896;
    Leb_Grid_XYZW[  781][2] = -0.283650284599206;
    Leb_Grid_XYZW[  781][3] =  0.000467941526232;

    Leb_Grid_XYZW[  782][0] =  0.336179474623259;
    Leb_Grid_XYZW[  782][1] =  0.224799590763267;
    Leb_Grid_XYZW[  782][2] =  0.914575587272423;
    Leb_Grid_XYZW[  782][3] =  0.000493084798163;

    Leb_Grid_XYZW[  783][0] =  0.336179474623258;
    Leb_Grid_XYZW[  783][1] =  0.224799590763267;
    Leb_Grid_XYZW[  783][2] = -0.914575587272423;
    Leb_Grid_XYZW[  783][3] =  0.000493084798163;

    Leb_Grid_XYZW[  784][0] =  0.336179474623259;
    Leb_Grid_XYZW[  784][1] = -0.224799590763267;
    Leb_Grid_XYZW[  784][2] =  0.914575587272423;
    Leb_Grid_XYZW[  784][3] =  0.000493084798163;

    Leb_Grid_XYZW[  785][0] =  0.336179474623258;
    Leb_Grid_XYZW[  785][1] = -0.224799590763267;
    Leb_Grid_XYZW[  785][2] = -0.914575587272423;
    Leb_Grid_XYZW[  785][3] =  0.000493084798163;

    Leb_Grid_XYZW[  786][0] = -0.336179474623259;
    Leb_Grid_XYZW[  786][1] =  0.224799590763267;
    Leb_Grid_XYZW[  786][2] =  0.914575587272423;
    Leb_Grid_XYZW[  786][3] =  0.000493084798163;

    Leb_Grid_XYZW[  787][0] = -0.336179474623258;
    Leb_Grid_XYZW[  787][1] =  0.224799590763267;
    Leb_Grid_XYZW[  787][2] = -0.914575587272423;
    Leb_Grid_XYZW[  787][3] =  0.000493084798163;

    Leb_Grid_XYZW[  788][0] = -0.336179474623259;
    Leb_Grid_XYZW[  788][1] = -0.224799590763267;
    Leb_Grid_XYZW[  788][2] =  0.914575587272423;
    Leb_Grid_XYZW[  788][3] =  0.000493084798163;

    Leb_Grid_XYZW[  789][0] = -0.336179474623258;
    Leb_Grid_XYZW[  789][1] = -0.224799590763267;
    Leb_Grid_XYZW[  789][2] = -0.914575587272423;
    Leb_Grid_XYZW[  789][3] =  0.000493084798163;

    Leb_Grid_XYZW[  790][0] =  0.336179474623259;
    Leb_Grid_XYZW[  790][1] =  0.914575587272423;
    Leb_Grid_XYZW[  790][2] =  0.224799590763267;
    Leb_Grid_XYZW[  790][3] =  0.000493084798163;

    Leb_Grid_XYZW[  791][0] =  0.336179474623259;
    Leb_Grid_XYZW[  791][1] =  0.914575587272423;
    Leb_Grid_XYZW[  791][2] = -0.224799590763267;
    Leb_Grid_XYZW[  791][3] =  0.000493084798163;

    Leb_Grid_XYZW[  792][0] =  0.336179474623259;
    Leb_Grid_XYZW[  792][1] = -0.914575587272423;
    Leb_Grid_XYZW[  792][2] =  0.224799590763267;
    Leb_Grid_XYZW[  792][3] =  0.000493084798163;

    Leb_Grid_XYZW[  793][0] =  0.336179474623259;
    Leb_Grid_XYZW[  793][1] = -0.914575587272423;
    Leb_Grid_XYZW[  793][2] = -0.224799590763267;
    Leb_Grid_XYZW[  793][3] =  0.000493084798163;

    Leb_Grid_XYZW[  794][0] = -0.336179474623259;
    Leb_Grid_XYZW[  794][1] =  0.914575587272423;
    Leb_Grid_XYZW[  794][2] =  0.224799590763267;
    Leb_Grid_XYZW[  794][3] =  0.000493084798163;

    Leb_Grid_XYZW[  795][0] = -0.336179474623259;
    Leb_Grid_XYZW[  795][1] =  0.914575587272423;
    Leb_Grid_XYZW[  795][2] = -0.224799590763267;
    Leb_Grid_XYZW[  795][3] =  0.000493084798163;

    Leb_Grid_XYZW[  796][0] = -0.336179474623259;
    Leb_Grid_XYZW[  796][1] = -0.914575587272423;
    Leb_Grid_XYZW[  796][2] =  0.224799590763267;
    Leb_Grid_XYZW[  796][3] =  0.000493084798163;

    Leb_Grid_XYZW[  797][0] = -0.336179474623259;
    Leb_Grid_XYZW[  797][1] = -0.914575587272423;
    Leb_Grid_XYZW[  797][2] = -0.224799590763267;
    Leb_Grid_XYZW[  797][3] =  0.000493084798163;

    Leb_Grid_XYZW[  798][0] =  0.224799590763267;
    Leb_Grid_XYZW[  798][1] =  0.336179474623259;
    Leb_Grid_XYZW[  798][2] =  0.914575587272423;
    Leb_Grid_XYZW[  798][3] =  0.000493084798163;

    Leb_Grid_XYZW[  799][0] =  0.224799590763267;
    Leb_Grid_XYZW[  799][1] =  0.336179474623258;
    Leb_Grid_XYZW[  799][2] = -0.914575587272423;
    Leb_Grid_XYZW[  799][3] =  0.000493084798163;

    Leb_Grid_XYZW[  800][0] =  0.224799590763267;
    Leb_Grid_XYZW[  800][1] = -0.336179474623259;
    Leb_Grid_XYZW[  800][2] =  0.914575587272423;
    Leb_Grid_XYZW[  800][3] =  0.000493084798163;

    Leb_Grid_XYZW[  801][0] =  0.224799590763267;
    Leb_Grid_XYZW[  801][1] = -0.336179474623258;
    Leb_Grid_XYZW[  801][2] = -0.914575587272423;
    Leb_Grid_XYZW[  801][3] =  0.000493084798163;

    Leb_Grid_XYZW[  802][0] = -0.224799590763267;
    Leb_Grid_XYZW[  802][1] =  0.336179474623259;
    Leb_Grid_XYZW[  802][2] =  0.914575587272423;
    Leb_Grid_XYZW[  802][3] =  0.000493084798163;

    Leb_Grid_XYZW[  803][0] = -0.224799590763267;
    Leb_Grid_XYZW[  803][1] =  0.336179474623258;
    Leb_Grid_XYZW[  803][2] = -0.914575587272423;
    Leb_Grid_XYZW[  803][3] =  0.000493084798163;

    Leb_Grid_XYZW[  804][0] = -0.224799590763267;
    Leb_Grid_XYZW[  804][1] = -0.336179474623259;
    Leb_Grid_XYZW[  804][2] =  0.914575587272423;
    Leb_Grid_XYZW[  804][3] =  0.000493084798163;

    Leb_Grid_XYZW[  805][0] = -0.224799590763267;
    Leb_Grid_XYZW[  805][1] = -0.336179474623258;
    Leb_Grid_XYZW[  805][2] = -0.914575587272423;
    Leb_Grid_XYZW[  805][3] =  0.000493084798163;

    Leb_Grid_XYZW[  806][0] =  0.224799590763267;
    Leb_Grid_XYZW[  806][1] =  0.914575587272423;
    Leb_Grid_XYZW[  806][2] =  0.336179474623259;
    Leb_Grid_XYZW[  806][3] =  0.000493084798163;

    Leb_Grid_XYZW[  807][0] =  0.224799590763267;
    Leb_Grid_XYZW[  807][1] =  0.914575587272423;
    Leb_Grid_XYZW[  807][2] = -0.336179474623259;
    Leb_Grid_XYZW[  807][3] =  0.000493084798163;

    Leb_Grid_XYZW[  808][0] =  0.224799590763267;
    Leb_Grid_XYZW[  808][1] = -0.914575587272423;
    Leb_Grid_XYZW[  808][2] =  0.336179474623259;
    Leb_Grid_XYZW[  808][3] =  0.000493084798163;

    Leb_Grid_XYZW[  809][0] =  0.224799590763267;
    Leb_Grid_XYZW[  809][1] = -0.914575587272423;
    Leb_Grid_XYZW[  809][2] = -0.336179474623259;
    Leb_Grid_XYZW[  809][3] =  0.000493084798163;

    Leb_Grid_XYZW[  810][0] = -0.224799590763267;
    Leb_Grid_XYZW[  810][1] =  0.914575587272423;
    Leb_Grid_XYZW[  810][2] =  0.336179474623259;
    Leb_Grid_XYZW[  810][3] =  0.000493084798163;

    Leb_Grid_XYZW[  811][0] = -0.224799590763267;
    Leb_Grid_XYZW[  811][1] =  0.914575587272423;
    Leb_Grid_XYZW[  811][2] = -0.336179474623259;
    Leb_Grid_XYZW[  811][3] =  0.000493084798163;

    Leb_Grid_XYZW[  812][0] = -0.224799590763267;
    Leb_Grid_XYZW[  812][1] = -0.914575587272423;
    Leb_Grid_XYZW[  812][2] =  0.336179474623259;
    Leb_Grid_XYZW[  812][3] =  0.000493084798163;

    Leb_Grid_XYZW[  813][0] = -0.224799590763267;
    Leb_Grid_XYZW[  813][1] = -0.914575587272423;
    Leb_Grid_XYZW[  813][2] = -0.336179474623259;
    Leb_Grid_XYZW[  813][3] =  0.000493084798163;

    Leb_Grid_XYZW[  814][0] =  0.914575587272423;
    Leb_Grid_XYZW[  814][1] =  0.336179474623259;
    Leb_Grid_XYZW[  814][2] =  0.224799590763267;
    Leb_Grid_XYZW[  814][3] =  0.000493084798163;

    Leb_Grid_XYZW[  815][0] =  0.914575587272423;
    Leb_Grid_XYZW[  815][1] =  0.336179474623259;
    Leb_Grid_XYZW[  815][2] = -0.224799590763267;
    Leb_Grid_XYZW[  815][3] =  0.000493084798163;

    Leb_Grid_XYZW[  816][0] =  0.914575587272423;
    Leb_Grid_XYZW[  816][1] = -0.336179474623259;
    Leb_Grid_XYZW[  816][2] =  0.224799590763267;
    Leb_Grid_XYZW[  816][3] =  0.000493084798163;

    Leb_Grid_XYZW[  817][0] =  0.914575587272423;
    Leb_Grid_XYZW[  817][1] = -0.336179474623259;
    Leb_Grid_XYZW[  817][2] = -0.224799590763267;
    Leb_Grid_XYZW[  817][3] =  0.000493084798163;

    Leb_Grid_XYZW[  818][0] = -0.914575587272423;
    Leb_Grid_XYZW[  818][1] =  0.336179474623259;
    Leb_Grid_XYZW[  818][2] =  0.224799590763267;
    Leb_Grid_XYZW[  818][3] =  0.000493084798163;

    Leb_Grid_XYZW[  819][0] = -0.914575587272423;
    Leb_Grid_XYZW[  819][1] =  0.336179474623259;
    Leb_Grid_XYZW[  819][2] = -0.224799590763267;
    Leb_Grid_XYZW[  819][3] =  0.000493084798163;

    Leb_Grid_XYZW[  820][0] = -0.914575587272423;
    Leb_Grid_XYZW[  820][1] = -0.336179474623259;
    Leb_Grid_XYZW[  820][2] =  0.224799590763267;
    Leb_Grid_XYZW[  820][3] =  0.000493084798163;

    Leb_Grid_XYZW[  821][0] = -0.914575587272423;
    Leb_Grid_XYZW[  821][1] = -0.336179474623259;
    Leb_Grid_XYZW[  821][2] = -0.224799590763267;
    Leb_Grid_XYZW[  821][3] =  0.000493084798163;

    Leb_Grid_XYZW[  822][0] =  0.914575587272423;
    Leb_Grid_XYZW[  822][1] =  0.224799590763267;
    Leb_Grid_XYZW[  822][2] =  0.336179474623259;
    Leb_Grid_XYZW[  822][3] =  0.000493084798163;

    Leb_Grid_XYZW[  823][0] =  0.914575587272423;
    Leb_Grid_XYZW[  823][1] =  0.224799590763267;
    Leb_Grid_XYZW[  823][2] = -0.336179474623259;
    Leb_Grid_XYZW[  823][3] =  0.000493084798163;

    Leb_Grid_XYZW[  824][0] =  0.914575587272423;
    Leb_Grid_XYZW[  824][1] = -0.224799590763267;
    Leb_Grid_XYZW[  824][2] =  0.336179474623259;
    Leb_Grid_XYZW[  824][3] =  0.000493084798163;

    Leb_Grid_XYZW[  825][0] =  0.914575587272423;
    Leb_Grid_XYZW[  825][1] = -0.224799590763267;
    Leb_Grid_XYZW[  825][2] = -0.336179474623259;
    Leb_Grid_XYZW[  825][3] =  0.000493084798163;

    Leb_Grid_XYZW[  826][0] = -0.914575587272423;
    Leb_Grid_XYZW[  826][1] =  0.224799590763267;
    Leb_Grid_XYZW[  826][2] =  0.336179474623259;
    Leb_Grid_XYZW[  826][3] =  0.000493084798163;

    Leb_Grid_XYZW[  827][0] = -0.914575587272423;
    Leb_Grid_XYZW[  827][1] =  0.224799590763267;
    Leb_Grid_XYZW[  827][2] = -0.336179474623259;
    Leb_Grid_XYZW[  827][3] =  0.000493084798163;

    Leb_Grid_XYZW[  828][0] = -0.914575587272423;
    Leb_Grid_XYZW[  828][1] = -0.224799590763267;
    Leb_Grid_XYZW[  828][2] =  0.336179474623259;
    Leb_Grid_XYZW[  828][3] =  0.000493084798163;

    Leb_Grid_XYZW[  829][0] = -0.914575587272423;
    Leb_Grid_XYZW[  829][1] = -0.224799590763267;
    Leb_Grid_XYZW[  829][2] = -0.336179474623259;
    Leb_Grid_XYZW[  829][3] =  0.000493084798163;

    Leb_Grid_XYZW[  830][0] =  0.387597917226482;
    Leb_Grid_XYZW[  830][1] =  0.274529925742225;
    Leb_Grid_XYZW[  830][2] =  0.880000667291600;
    Leb_Grid_XYZW[  830][3] =  0.000511503186754;

    Leb_Grid_XYZW[  831][0] =  0.387597917226483;
    Leb_Grid_XYZW[  831][1] =  0.274529925742225;
    Leb_Grid_XYZW[  831][2] = -0.880000667291600;
    Leb_Grid_XYZW[  831][3] =  0.000511503186754;

    Leb_Grid_XYZW[  832][0] =  0.387597917226482;
    Leb_Grid_XYZW[  832][1] = -0.274529925742225;
    Leb_Grid_XYZW[  832][2] =  0.880000667291600;
    Leb_Grid_XYZW[  832][3] =  0.000511503186754;

    Leb_Grid_XYZW[  833][0] =  0.387597917226483;
    Leb_Grid_XYZW[  833][1] = -0.274529925742225;
    Leb_Grid_XYZW[  833][2] = -0.880000667291600;
    Leb_Grid_XYZW[  833][3] =  0.000511503186754;

    Leb_Grid_XYZW[  834][0] = -0.387597917226482;
    Leb_Grid_XYZW[  834][1] =  0.274529925742225;
    Leb_Grid_XYZW[  834][2] =  0.880000667291600;
    Leb_Grid_XYZW[  834][3] =  0.000511503186754;

    Leb_Grid_XYZW[  835][0] = -0.387597917226483;
    Leb_Grid_XYZW[  835][1] =  0.274529925742225;
    Leb_Grid_XYZW[  835][2] = -0.880000667291600;
    Leb_Grid_XYZW[  835][3] =  0.000511503186754;

    Leb_Grid_XYZW[  836][0] = -0.387597917226482;
    Leb_Grid_XYZW[  836][1] = -0.274529925742225;
    Leb_Grid_XYZW[  836][2] =  0.880000667291600;
    Leb_Grid_XYZW[  836][3] =  0.000511503186754;

    Leb_Grid_XYZW[  837][0] = -0.387597917226483;
    Leb_Grid_XYZW[  837][1] = -0.274529925742225;
    Leb_Grid_XYZW[  837][2] = -0.880000667291600;
    Leb_Grid_XYZW[  837][3] =  0.000511503186754;

    Leb_Grid_XYZW[  838][0] =  0.387597917226482;
    Leb_Grid_XYZW[  838][1] =  0.880000667291600;
    Leb_Grid_XYZW[  838][2] =  0.274529925742225;
    Leb_Grid_XYZW[  838][3] =  0.000511503186754;

    Leb_Grid_XYZW[  839][0] =  0.387597917226482;
    Leb_Grid_XYZW[  839][1] =  0.880000667291600;
    Leb_Grid_XYZW[  839][2] = -0.274529925742224;
    Leb_Grid_XYZW[  839][3] =  0.000511503186754;

    Leb_Grid_XYZW[  840][0] =  0.387597917226482;
    Leb_Grid_XYZW[  840][1] = -0.880000667291600;
    Leb_Grid_XYZW[  840][2] =  0.274529925742225;
    Leb_Grid_XYZW[  840][3] =  0.000511503186754;

    Leb_Grid_XYZW[  841][0] =  0.387597917226482;
    Leb_Grid_XYZW[  841][1] = -0.880000667291600;
    Leb_Grid_XYZW[  841][2] = -0.274529925742224;
    Leb_Grid_XYZW[  841][3] =  0.000511503186754;

    Leb_Grid_XYZW[  842][0] = -0.387597917226482;
    Leb_Grid_XYZW[  842][1] =  0.880000667291600;
    Leb_Grid_XYZW[  842][2] =  0.274529925742225;
    Leb_Grid_XYZW[  842][3] =  0.000511503186754;

    Leb_Grid_XYZW[  843][0] = -0.387597917226483;
    Leb_Grid_XYZW[  843][1] =  0.880000667291600;
    Leb_Grid_XYZW[  843][2] = -0.274529925742224;
    Leb_Grid_XYZW[  843][3] =  0.000511503186754;

    Leb_Grid_XYZW[  844][0] = -0.387597917226482;
    Leb_Grid_XYZW[  844][1] = -0.880000667291600;
    Leb_Grid_XYZW[  844][2] =  0.274529925742225;
    Leb_Grid_XYZW[  844][3] =  0.000511503186754;

    Leb_Grid_XYZW[  845][0] = -0.387597917226483;
    Leb_Grid_XYZW[  845][1] = -0.880000667291600;
    Leb_Grid_XYZW[  845][2] = -0.274529925742224;
    Leb_Grid_XYZW[  845][3] =  0.000511503186754;

    Leb_Grid_XYZW[  846][0] =  0.274529925742225;
    Leb_Grid_XYZW[  846][1] =  0.387597917226482;
    Leb_Grid_XYZW[  846][2] =  0.880000667291600;
    Leb_Grid_XYZW[  846][3] =  0.000511503186754;

    Leb_Grid_XYZW[  847][0] =  0.274529925742225;
    Leb_Grid_XYZW[  847][1] =  0.387597917226483;
    Leb_Grid_XYZW[  847][2] = -0.880000667291600;
    Leb_Grid_XYZW[  847][3] =  0.000511503186754;

    Leb_Grid_XYZW[  848][0] =  0.274529925742225;
    Leb_Grid_XYZW[  848][1] = -0.387597917226482;
    Leb_Grid_XYZW[  848][2] =  0.880000667291600;
    Leb_Grid_XYZW[  848][3] =  0.000511503186754;

    Leb_Grid_XYZW[  849][0] =  0.274529925742225;
    Leb_Grid_XYZW[  849][1] = -0.387597917226483;
    Leb_Grid_XYZW[  849][2] = -0.880000667291600;
    Leb_Grid_XYZW[  849][3] =  0.000511503186754;

    Leb_Grid_XYZW[  850][0] = -0.274529925742225;
    Leb_Grid_XYZW[  850][1] =  0.387597917226483;
    Leb_Grid_XYZW[  850][2] =  0.880000667291600;
    Leb_Grid_XYZW[  850][3] =  0.000511503186754;

    Leb_Grid_XYZW[  851][0] = -0.274529925742225;
    Leb_Grid_XYZW[  851][1] =  0.387597917226483;
    Leb_Grid_XYZW[  851][2] = -0.880000667291600;
    Leb_Grid_XYZW[  851][3] =  0.000511503186754;

    Leb_Grid_XYZW[  852][0] = -0.274529925742225;
    Leb_Grid_XYZW[  852][1] = -0.387597917226483;
    Leb_Grid_XYZW[  852][2] =  0.880000667291600;
    Leb_Grid_XYZW[  852][3] =  0.000511503186754;

    Leb_Grid_XYZW[  853][0] = -0.274529925742225;
    Leb_Grid_XYZW[  853][1] = -0.387597917226483;
    Leb_Grid_XYZW[  853][2] = -0.880000667291600;
    Leb_Grid_XYZW[  853][3] =  0.000511503186754;

    Leb_Grid_XYZW[  854][0] =  0.274529925742224;
    Leb_Grid_XYZW[  854][1] =  0.880000667291600;
    Leb_Grid_XYZW[  854][2] =  0.387597917226483;
    Leb_Grid_XYZW[  854][3] =  0.000511503186754;

    Leb_Grid_XYZW[  855][0] =  0.274529925742225;
    Leb_Grid_XYZW[  855][1] =  0.880000667291600;
    Leb_Grid_XYZW[  855][2] = -0.387597917226482;
    Leb_Grid_XYZW[  855][3] =  0.000511503186754;

    Leb_Grid_XYZW[  856][0] =  0.274529925742224;
    Leb_Grid_XYZW[  856][1] = -0.880000667291600;
    Leb_Grid_XYZW[  856][2] =  0.387597917226483;
    Leb_Grid_XYZW[  856][3] =  0.000511503186754;

    Leb_Grid_XYZW[  857][0] =  0.274529925742225;
    Leb_Grid_XYZW[  857][1] = -0.880000667291600;
    Leb_Grid_XYZW[  857][2] = -0.387597917226482;
    Leb_Grid_XYZW[  857][3] =  0.000511503186754;

    Leb_Grid_XYZW[  858][0] = -0.274529925742224;
    Leb_Grid_XYZW[  858][1] =  0.880000667291600;
    Leb_Grid_XYZW[  858][2] =  0.387597917226483;
    Leb_Grid_XYZW[  858][3] =  0.000511503186754;

    Leb_Grid_XYZW[  859][0] = -0.274529925742224;
    Leb_Grid_XYZW[  859][1] =  0.880000667291600;
    Leb_Grid_XYZW[  859][2] = -0.387597917226482;
    Leb_Grid_XYZW[  859][3] =  0.000511503186754;

    Leb_Grid_XYZW[  860][0] = -0.274529925742224;
    Leb_Grid_XYZW[  860][1] = -0.880000667291600;
    Leb_Grid_XYZW[  860][2] =  0.387597917226483;
    Leb_Grid_XYZW[  860][3] =  0.000511503186754;

    Leb_Grid_XYZW[  861][0] = -0.274529925742224;
    Leb_Grid_XYZW[  861][1] = -0.880000667291600;
    Leb_Grid_XYZW[  861][2] = -0.387597917226482;
    Leb_Grid_XYZW[  861][3] =  0.000511503186754;

    Leb_Grid_XYZW[  862][0] =  0.880000667291600;
    Leb_Grid_XYZW[  862][1] =  0.387597917226483;
    Leb_Grid_XYZW[  862][2] =  0.274529925742225;
    Leb_Grid_XYZW[  862][3] =  0.000511503186754;

    Leb_Grid_XYZW[  863][0] =  0.880000667291600;
    Leb_Grid_XYZW[  863][1] =  0.387597917226483;
    Leb_Grid_XYZW[  863][2] = -0.274529925742224;
    Leb_Grid_XYZW[  863][3] =  0.000511503186754;

    Leb_Grid_XYZW[  864][0] =  0.880000667291600;
    Leb_Grid_XYZW[  864][1] = -0.387597917226483;
    Leb_Grid_XYZW[  864][2] =  0.274529925742225;
    Leb_Grid_XYZW[  864][3] =  0.000511503186754;

    Leb_Grid_XYZW[  865][0] =  0.880000667291600;
    Leb_Grid_XYZW[  865][1] = -0.387597917226483;
    Leb_Grid_XYZW[  865][2] = -0.274529925742224;
    Leb_Grid_XYZW[  865][3] =  0.000511503186754;

    Leb_Grid_XYZW[  866][0] = -0.880000667291600;
    Leb_Grid_XYZW[  866][1] =  0.387597917226482;
    Leb_Grid_XYZW[  866][2] =  0.274529925742225;
    Leb_Grid_XYZW[  866][3] =  0.000511503186754;

    Leb_Grid_XYZW[  867][0] = -0.880000667291600;
    Leb_Grid_XYZW[  867][1] =  0.387597917226482;
    Leb_Grid_XYZW[  867][2] = -0.274529925742224;
    Leb_Grid_XYZW[  867][3] =  0.000511503186754;

    Leb_Grid_XYZW[  868][0] = -0.880000667291600;
    Leb_Grid_XYZW[  868][1] = -0.387597917226482;
    Leb_Grid_XYZW[  868][2] =  0.274529925742225;
    Leb_Grid_XYZW[  868][3] =  0.000511503186754;

    Leb_Grid_XYZW[  869][0] = -0.880000667291600;
    Leb_Grid_XYZW[  869][1] = -0.387597917226482;
    Leb_Grid_XYZW[  869][2] = -0.274529925742224;
    Leb_Grid_XYZW[  869][3] =  0.000511503186754;

    Leb_Grid_XYZW[  870][0] =  0.880000667291600;
    Leb_Grid_XYZW[  870][1] =  0.274529925742225;
    Leb_Grid_XYZW[  870][2] =  0.387597917226483;
    Leb_Grid_XYZW[  870][3] =  0.000511503186754;

    Leb_Grid_XYZW[  871][0] =  0.880000667291600;
    Leb_Grid_XYZW[  871][1] =  0.274529925742225;
    Leb_Grid_XYZW[  871][2] = -0.387597917226482;
    Leb_Grid_XYZW[  871][3] =  0.000511503186754;

    Leb_Grid_XYZW[  872][0] =  0.880000667291600;
    Leb_Grid_XYZW[  872][1] = -0.274529925742225;
    Leb_Grid_XYZW[  872][2] =  0.387597917226483;
    Leb_Grid_XYZW[  872][3] =  0.000511503186754;

    Leb_Grid_XYZW[  873][0] =  0.880000667291600;
    Leb_Grid_XYZW[  873][1] = -0.274529925742225;
    Leb_Grid_XYZW[  873][2] = -0.387597917226482;
    Leb_Grid_XYZW[  873][3] =  0.000511503186754;

    Leb_Grid_XYZW[  874][0] = -0.880000667291600;
    Leb_Grid_XYZW[  874][1] =  0.274529925742224;
    Leb_Grid_XYZW[  874][2] =  0.387597917226483;
    Leb_Grid_XYZW[  874][3] =  0.000511503186754;

    Leb_Grid_XYZW[  875][0] = -0.880000667291600;
    Leb_Grid_XYZW[  875][1] =  0.274529925742224;
    Leb_Grid_XYZW[  875][2] = -0.387597917226482;
    Leb_Grid_XYZW[  875][3] =  0.000511503186754;

    Leb_Grid_XYZW[  876][0] = -0.880000667291600;
    Leb_Grid_XYZW[  876][1] = -0.274529925742224;
    Leb_Grid_XYZW[  876][2] =  0.387597917226483;
    Leb_Grid_XYZW[  876][3] =  0.000511503186754;

    Leb_Grid_XYZW[  877][0] = -0.880000667291600;
    Leb_Grid_XYZW[  877][1] = -0.274529925742224;
    Leb_Grid_XYZW[  877][2] = -0.387597917226482;
    Leb_Grid_XYZW[  877][3] =  0.000511503186754;

    Leb_Grid_XYZW[  878][0] =  0.437401931699907;
    Leb_Grid_XYZW[  878][1] =  0.323637348244112;
    Leb_Grid_XYZW[  878][2] =  0.839010379534550;
    Leb_Grid_XYZW[  878][3] =  0.000524521714846;

    Leb_Grid_XYZW[  879][0] =  0.437401931699907;
    Leb_Grid_XYZW[  879][1] =  0.323637348244112;
    Leb_Grid_XYZW[  879][2] = -0.839010379534550;
    Leb_Grid_XYZW[  879][3] =  0.000524521714846;

    Leb_Grid_XYZW[  880][0] =  0.437401931699907;
    Leb_Grid_XYZW[  880][1] = -0.323637348244112;
    Leb_Grid_XYZW[  880][2] =  0.839010379534550;
    Leb_Grid_XYZW[  880][3] =  0.000524521714846;

    Leb_Grid_XYZW[  881][0] =  0.437401931699907;
    Leb_Grid_XYZW[  881][1] = -0.323637348244112;
    Leb_Grid_XYZW[  881][2] = -0.839010379534550;
    Leb_Grid_XYZW[  881][3] =  0.000524521714846;

    Leb_Grid_XYZW[  882][0] = -0.437401931699907;
    Leb_Grid_XYZW[  882][1] =  0.323637348244112;
    Leb_Grid_XYZW[  882][2] =  0.839010379534550;
    Leb_Grid_XYZW[  882][3] =  0.000524521714846;

    Leb_Grid_XYZW[  883][0] = -0.437401931699907;
    Leb_Grid_XYZW[  883][1] =  0.323637348244112;
    Leb_Grid_XYZW[  883][2] = -0.839010379534550;
    Leb_Grid_XYZW[  883][3] =  0.000524521714846;

    Leb_Grid_XYZW[  884][0] = -0.437401931699907;
    Leb_Grid_XYZW[  884][1] = -0.323637348244112;
    Leb_Grid_XYZW[  884][2] =  0.839010379534550;
    Leb_Grid_XYZW[  884][3] =  0.000524521714846;

    Leb_Grid_XYZW[  885][0] = -0.437401931699907;
    Leb_Grid_XYZW[  885][1] = -0.323637348244112;
    Leb_Grid_XYZW[  885][2] = -0.839010379534550;
    Leb_Grid_XYZW[  885][3] =  0.000524521714846;

    Leb_Grid_XYZW[  886][0] =  0.437401931699907;
    Leb_Grid_XYZW[  886][1] =  0.839010379534550;
    Leb_Grid_XYZW[  886][2] =  0.323637348244112;
    Leb_Grid_XYZW[  886][3] =  0.000524521714846;

    Leb_Grid_XYZW[  887][0] =  0.437401931699907;
    Leb_Grid_XYZW[  887][1] =  0.839010379534550;
    Leb_Grid_XYZW[  887][2] = -0.323637348244112;
    Leb_Grid_XYZW[  887][3] =  0.000524521714846;

    Leb_Grid_XYZW[  888][0] =  0.437401931699907;
    Leb_Grid_XYZW[  888][1] = -0.839010379534550;
    Leb_Grid_XYZW[  888][2] =  0.323637348244112;
    Leb_Grid_XYZW[  888][3] =  0.000524521714846;

    Leb_Grid_XYZW[  889][0] =  0.437401931699907;
    Leb_Grid_XYZW[  889][1] = -0.839010379534550;
    Leb_Grid_XYZW[  889][2] = -0.323637348244112;
    Leb_Grid_XYZW[  889][3] =  0.000524521714846;

    Leb_Grid_XYZW[  890][0] = -0.437401931699907;
    Leb_Grid_XYZW[  890][1] =  0.839010379534550;
    Leb_Grid_XYZW[  890][2] =  0.323637348244112;
    Leb_Grid_XYZW[  890][3] =  0.000524521714846;

    Leb_Grid_XYZW[  891][0] = -0.437401931699907;
    Leb_Grid_XYZW[  891][1] =  0.839010379534550;
    Leb_Grid_XYZW[  891][2] = -0.323637348244112;
    Leb_Grid_XYZW[  891][3] =  0.000524521714846;

    Leb_Grid_XYZW[  892][0] = -0.437401931699907;
    Leb_Grid_XYZW[  892][1] = -0.839010379534550;
    Leb_Grid_XYZW[  892][2] =  0.323637348244112;
    Leb_Grid_XYZW[  892][3] =  0.000524521714846;

    Leb_Grid_XYZW[  893][0] = -0.437401931699907;
    Leb_Grid_XYZW[  893][1] = -0.839010379534550;
    Leb_Grid_XYZW[  893][2] = -0.323637348244112;
    Leb_Grid_XYZW[  893][3] =  0.000524521714846;

    Leb_Grid_XYZW[  894][0] =  0.323637348244112;
    Leb_Grid_XYZW[  894][1] =  0.437401931699907;
    Leb_Grid_XYZW[  894][2] =  0.839010379534550;
    Leb_Grid_XYZW[  894][3] =  0.000524521714846;

    Leb_Grid_XYZW[  895][0] =  0.323637348244112;
    Leb_Grid_XYZW[  895][1] =  0.437401931699907;
    Leb_Grid_XYZW[  895][2] = -0.839010379534550;
    Leb_Grid_XYZW[  895][3] =  0.000524521714846;

    Leb_Grid_XYZW[  896][0] =  0.323637348244112;
    Leb_Grid_XYZW[  896][1] = -0.437401931699907;
    Leb_Grid_XYZW[  896][2] =  0.839010379534550;
    Leb_Grid_XYZW[  896][3] =  0.000524521714846;

    Leb_Grid_XYZW[  897][0] =  0.323637348244112;
    Leb_Grid_XYZW[  897][1] = -0.437401931699907;
    Leb_Grid_XYZW[  897][2] = -0.839010379534550;
    Leb_Grid_XYZW[  897][3] =  0.000524521714846;

    Leb_Grid_XYZW[  898][0] = -0.323637348244112;
    Leb_Grid_XYZW[  898][1] =  0.437401931699907;
    Leb_Grid_XYZW[  898][2] =  0.839010379534550;
    Leb_Grid_XYZW[  898][3] =  0.000524521714846;

    Leb_Grid_XYZW[  899][0] = -0.323637348244112;
    Leb_Grid_XYZW[  899][1] =  0.437401931699907;
    Leb_Grid_XYZW[  899][2] = -0.839010379534550;
    Leb_Grid_XYZW[  899][3] =  0.000524521714846;

    Leb_Grid_XYZW[  900][0] = -0.323637348244112;
    Leb_Grid_XYZW[  900][1] = -0.437401931699907;
    Leb_Grid_XYZW[  900][2] =  0.839010379534550;
    Leb_Grid_XYZW[  900][3] =  0.000524521714846;

    Leb_Grid_XYZW[  901][0] = -0.323637348244112;
    Leb_Grid_XYZW[  901][1] = -0.437401931699907;
    Leb_Grid_XYZW[  901][2] = -0.839010379534550;
    Leb_Grid_XYZW[  901][3] =  0.000524521714846;

    Leb_Grid_XYZW[  902][0] =  0.323637348244112;
    Leb_Grid_XYZW[  902][1] =  0.839010379534550;
    Leb_Grid_XYZW[  902][2] =  0.437401931699907;
    Leb_Grid_XYZW[  902][3] =  0.000524521714846;

    Leb_Grid_XYZW[  903][0] =  0.323637348244112;
    Leb_Grid_XYZW[  903][1] =  0.839010379534550;
    Leb_Grid_XYZW[  903][2] = -0.437401931699907;
    Leb_Grid_XYZW[  903][3] =  0.000524521714846;

    Leb_Grid_XYZW[  904][0] =  0.323637348244112;
    Leb_Grid_XYZW[  904][1] = -0.839010379534550;
    Leb_Grid_XYZW[  904][2] =  0.437401931699907;
    Leb_Grid_XYZW[  904][3] =  0.000524521714846;

    Leb_Grid_XYZW[  905][0] =  0.323637348244112;
    Leb_Grid_XYZW[  905][1] = -0.839010379534550;
    Leb_Grid_XYZW[  905][2] = -0.437401931699907;
    Leb_Grid_XYZW[  905][3] =  0.000524521714846;

    Leb_Grid_XYZW[  906][0] = -0.323637348244112;
    Leb_Grid_XYZW[  906][1] =  0.839010379534550;
    Leb_Grid_XYZW[  906][2] =  0.437401931699907;
    Leb_Grid_XYZW[  906][3] =  0.000524521714846;

    Leb_Grid_XYZW[  907][0] = -0.323637348244112;
    Leb_Grid_XYZW[  907][1] =  0.839010379534550;
    Leb_Grid_XYZW[  907][2] = -0.437401931699907;
    Leb_Grid_XYZW[  907][3] =  0.000524521714846;

    Leb_Grid_XYZW[  908][0] = -0.323637348244112;
    Leb_Grid_XYZW[  908][1] = -0.839010379534550;
    Leb_Grid_XYZW[  908][2] =  0.437401931699907;
    Leb_Grid_XYZW[  908][3] =  0.000524521714846;

    Leb_Grid_XYZW[  909][0] = -0.323637348244112;
    Leb_Grid_XYZW[  909][1] = -0.839010379534550;
    Leb_Grid_XYZW[  909][2] = -0.437401931699907;
    Leb_Grid_XYZW[  909][3] =  0.000524521714846;

    Leb_Grid_XYZW[  910][0] =  0.839010379534550;
    Leb_Grid_XYZW[  910][1] =  0.437401931699907;
    Leb_Grid_XYZW[  910][2] =  0.323637348244112;
    Leb_Grid_XYZW[  910][3] =  0.000524521714846;

    Leb_Grid_XYZW[  911][0] =  0.839010379534550;
    Leb_Grid_XYZW[  911][1] =  0.437401931699907;
    Leb_Grid_XYZW[  911][2] = -0.323637348244112;
    Leb_Grid_XYZW[  911][3] =  0.000524521714846;

    Leb_Grid_XYZW[  912][0] =  0.839010379534550;
    Leb_Grid_XYZW[  912][1] = -0.437401931699907;
    Leb_Grid_XYZW[  912][2] =  0.323637348244112;
    Leb_Grid_XYZW[  912][3] =  0.000524521714846;

    Leb_Grid_XYZW[  913][0] =  0.839010379534550;
    Leb_Grid_XYZW[  913][1] = -0.437401931699907;
    Leb_Grid_XYZW[  913][2] = -0.323637348244112;
    Leb_Grid_XYZW[  913][3] =  0.000524521714846;

    Leb_Grid_XYZW[  914][0] = -0.839010379534550;
    Leb_Grid_XYZW[  914][1] =  0.437401931699907;
    Leb_Grid_XYZW[  914][2] =  0.323637348244112;
    Leb_Grid_XYZW[  914][3] =  0.000524521714846;

    Leb_Grid_XYZW[  915][0] = -0.839010379534550;
    Leb_Grid_XYZW[  915][1] =  0.437401931699907;
    Leb_Grid_XYZW[  915][2] = -0.323637348244112;
    Leb_Grid_XYZW[  915][3] =  0.000524521714846;

    Leb_Grid_XYZW[  916][0] = -0.839010379534550;
    Leb_Grid_XYZW[  916][1] = -0.437401931699907;
    Leb_Grid_XYZW[  916][2] =  0.323637348244112;
    Leb_Grid_XYZW[  916][3] =  0.000524521714846;

    Leb_Grid_XYZW[  917][0] = -0.839010379534550;
    Leb_Grid_XYZW[  917][1] = -0.437401931699907;
    Leb_Grid_XYZW[  917][2] = -0.323637348244112;
    Leb_Grid_XYZW[  917][3] =  0.000524521714846;

    Leb_Grid_XYZW[  918][0] =  0.839010379534550;
    Leb_Grid_XYZW[  918][1] =  0.323637348244112;
    Leb_Grid_XYZW[  918][2] =  0.437401931699907;
    Leb_Grid_XYZW[  918][3] =  0.000524521714846;

    Leb_Grid_XYZW[  919][0] =  0.839010379534550;
    Leb_Grid_XYZW[  919][1] =  0.323637348244112;
    Leb_Grid_XYZW[  919][2] = -0.437401931699907;
    Leb_Grid_XYZW[  919][3] =  0.000524521714846;

    Leb_Grid_XYZW[  920][0] =  0.839010379534550;
    Leb_Grid_XYZW[  920][1] = -0.323637348244112;
    Leb_Grid_XYZW[  920][2] =  0.437401931699907;
    Leb_Grid_XYZW[  920][3] =  0.000524521714846;

    Leb_Grid_XYZW[  921][0] =  0.839010379534550;
    Leb_Grid_XYZW[  921][1] = -0.323637348244112;
    Leb_Grid_XYZW[  921][2] = -0.437401931699907;
    Leb_Grid_XYZW[  921][3] =  0.000524521714846;

    Leb_Grid_XYZW[  922][0] = -0.839010379534550;
    Leb_Grid_XYZW[  922][1] =  0.323637348244112;
    Leb_Grid_XYZW[  922][2] =  0.437401931699907;
    Leb_Grid_XYZW[  922][3] =  0.000524521714846;

    Leb_Grid_XYZW[  923][0] = -0.839010379534550;
    Leb_Grid_XYZW[  923][1] =  0.323637348244112;
    Leb_Grid_XYZW[  923][2] = -0.437401931699907;
    Leb_Grid_XYZW[  923][3] =  0.000524521714846;

    Leb_Grid_XYZW[  924][0] = -0.839010379534550;
    Leb_Grid_XYZW[  924][1] = -0.323637348244112;
    Leb_Grid_XYZW[  924][2] =  0.437401931699907;
    Leb_Grid_XYZW[  924][3] =  0.000524521714846;

    Leb_Grid_XYZW[  925][0] = -0.839010379534550;
    Leb_Grid_XYZW[  925][1] = -0.323637348244112;
    Leb_Grid_XYZW[  925][2] = -0.437401931699907;
    Leb_Grid_XYZW[  925][3] =  0.000524521714846;

    Leb_Grid_XYZW[  926][0] =  0.485127584334002;
    Leb_Grid_XYZW[  926][1] =  0.371496785943674;
    Leb_Grid_XYZW[  926][2] =  0.791606824725365;
    Leb_Grid_XYZW[  926][3] =  0.000533204149990;

    Leb_Grid_XYZW[  927][0] =  0.485127584334002;
    Leb_Grid_XYZW[  927][1] =  0.371496785943674;
    Leb_Grid_XYZW[  927][2] = -0.791606824725365;
    Leb_Grid_XYZW[  927][3] =  0.000533204149990;

    Leb_Grid_XYZW[  928][0] =  0.485127584334002;
    Leb_Grid_XYZW[  928][1] = -0.371496785943674;
    Leb_Grid_XYZW[  928][2] =  0.791606824725365;
    Leb_Grid_XYZW[  928][3] =  0.000533204149990;

    Leb_Grid_XYZW[  929][0] =  0.485127584334002;
    Leb_Grid_XYZW[  929][1] = -0.371496785943674;
    Leb_Grid_XYZW[  929][2] = -0.791606824725365;
    Leb_Grid_XYZW[  929][3] =  0.000533204149990;

    Leb_Grid_XYZW[  930][0] = -0.485127584334002;
    Leb_Grid_XYZW[  930][1] =  0.371496785943674;
    Leb_Grid_XYZW[  930][2] =  0.791606824725365;
    Leb_Grid_XYZW[  930][3] =  0.000533204149990;

    Leb_Grid_XYZW[  931][0] = -0.485127584334002;
    Leb_Grid_XYZW[  931][1] =  0.371496785943674;
    Leb_Grid_XYZW[  931][2] = -0.791606824725365;
    Leb_Grid_XYZW[  931][3] =  0.000533204149990;

    Leb_Grid_XYZW[  932][0] = -0.485127584334002;
    Leb_Grid_XYZW[  932][1] = -0.371496785943674;
    Leb_Grid_XYZW[  932][2] =  0.791606824725365;
    Leb_Grid_XYZW[  932][3] =  0.000533204149990;

    Leb_Grid_XYZW[  933][0] = -0.485127584334002;
    Leb_Grid_XYZW[  933][1] = -0.371496785943674;
    Leb_Grid_XYZW[  933][2] = -0.791606824725365;
    Leb_Grid_XYZW[  933][3] =  0.000533204149990;

    Leb_Grid_XYZW[  934][0] =  0.485127584334002;
    Leb_Grid_XYZW[  934][1] =  0.791606824725366;
    Leb_Grid_XYZW[  934][2] =  0.371496785943674;
    Leb_Grid_XYZW[  934][3] =  0.000533204149990;

    Leb_Grid_XYZW[  935][0] =  0.485127584334002;
    Leb_Grid_XYZW[  935][1] =  0.791606824725366;
    Leb_Grid_XYZW[  935][2] = -0.371496785943674;
    Leb_Grid_XYZW[  935][3] =  0.000533204149990;

    Leb_Grid_XYZW[  936][0] =  0.485127584334002;
    Leb_Grid_XYZW[  936][1] = -0.791606824725366;
    Leb_Grid_XYZW[  936][2] =  0.371496785943674;
    Leb_Grid_XYZW[  936][3] =  0.000533204149990;

    Leb_Grid_XYZW[  937][0] =  0.485127584334002;
    Leb_Grid_XYZW[  937][1] = -0.791606824725366;
    Leb_Grid_XYZW[  937][2] = -0.371496785943674;
    Leb_Grid_XYZW[  937][3] =  0.000533204149990;

    Leb_Grid_XYZW[  938][0] = -0.485127584334002;
    Leb_Grid_XYZW[  938][1] =  0.791606824725366;
    Leb_Grid_XYZW[  938][2] =  0.371496785943674;
    Leb_Grid_XYZW[  938][3] =  0.000533204149990;

    Leb_Grid_XYZW[  939][0] = -0.485127584334002;
    Leb_Grid_XYZW[  939][1] =  0.791606824725365;
    Leb_Grid_XYZW[  939][2] = -0.371496785943674;
    Leb_Grid_XYZW[  939][3] =  0.000533204149990;

    Leb_Grid_XYZW[  940][0] = -0.485127584334002;
    Leb_Grid_XYZW[  940][1] = -0.791606824725366;
    Leb_Grid_XYZW[  940][2] =  0.371496785943674;
    Leb_Grid_XYZW[  940][3] =  0.000533204149990;

    Leb_Grid_XYZW[  941][0] = -0.485127584334002;
    Leb_Grid_XYZW[  941][1] = -0.791606824725365;
    Leb_Grid_XYZW[  941][2] = -0.371496785943674;
    Leb_Grid_XYZW[  941][3] =  0.000533204149990;

    Leb_Grid_XYZW[  942][0] =  0.371496785943674;
    Leb_Grid_XYZW[  942][1] =  0.485127584334002;
    Leb_Grid_XYZW[  942][2] =  0.791606824725365;
    Leb_Grid_XYZW[  942][3] =  0.000533204149990;

    Leb_Grid_XYZW[  943][0] =  0.371496785943674;
    Leb_Grid_XYZW[  943][1] =  0.485127584334002;
    Leb_Grid_XYZW[  943][2] = -0.791606824725365;
    Leb_Grid_XYZW[  943][3] =  0.000533204149990;

    Leb_Grid_XYZW[  944][0] =  0.371496785943674;
    Leb_Grid_XYZW[  944][1] = -0.485127584334002;
    Leb_Grid_XYZW[  944][2] =  0.791606824725365;
    Leb_Grid_XYZW[  944][3] =  0.000533204149990;

    Leb_Grid_XYZW[  945][0] =  0.371496785943674;
    Leb_Grid_XYZW[  945][1] = -0.485127584334002;
    Leb_Grid_XYZW[  945][2] = -0.791606824725365;
    Leb_Grid_XYZW[  945][3] =  0.000533204149990;

    Leb_Grid_XYZW[  946][0] = -0.371496785943674;
    Leb_Grid_XYZW[  946][1] =  0.485127584334002;
    Leb_Grid_XYZW[  946][2] =  0.791606824725365;
    Leb_Grid_XYZW[  946][3] =  0.000533204149990;

    Leb_Grid_XYZW[  947][0] = -0.371496785943674;
    Leb_Grid_XYZW[  947][1] =  0.485127584334002;
    Leb_Grid_XYZW[  947][2] = -0.791606824725365;
    Leb_Grid_XYZW[  947][3] =  0.000533204149990;

    Leb_Grid_XYZW[  948][0] = -0.371496785943674;
    Leb_Grid_XYZW[  948][1] = -0.485127584334002;
    Leb_Grid_XYZW[  948][2] =  0.791606824725365;
    Leb_Grid_XYZW[  948][3] =  0.000533204149990;

    Leb_Grid_XYZW[  949][0] = -0.371496785943674;
    Leb_Grid_XYZW[  949][1] = -0.485127584334002;
    Leb_Grid_XYZW[  949][2] = -0.791606824725365;
    Leb_Grid_XYZW[  949][3] =  0.000533204149990;

    Leb_Grid_XYZW[  950][0] =  0.371496785943674;
    Leb_Grid_XYZW[  950][1] =  0.791606824725365;
    Leb_Grid_XYZW[  950][2] =  0.485127584334002;
    Leb_Grid_XYZW[  950][3] =  0.000533204149990;

    Leb_Grid_XYZW[  951][0] =  0.371496785943674;
    Leb_Grid_XYZW[  951][1] =  0.791606824725366;
    Leb_Grid_XYZW[  951][2] = -0.485127584334002;
    Leb_Grid_XYZW[  951][3] =  0.000533204149990;

    Leb_Grid_XYZW[  952][0] =  0.371496785943674;
    Leb_Grid_XYZW[  952][1] = -0.791606824725365;
    Leb_Grid_XYZW[  952][2] =  0.485127584334002;
    Leb_Grid_XYZW[  952][3] =  0.000533204149990;

    Leb_Grid_XYZW[  953][0] =  0.371496785943674;
    Leb_Grid_XYZW[  953][1] = -0.791606824725366;
    Leb_Grid_XYZW[  953][2] = -0.485127584334002;
    Leb_Grid_XYZW[  953][3] =  0.000533204149990;

    Leb_Grid_XYZW[  954][0] = -0.371496785943674;
    Leb_Grid_XYZW[  954][1] =  0.791606824725365;
    Leb_Grid_XYZW[  954][2] =  0.485127584334002;
    Leb_Grid_XYZW[  954][3] =  0.000533204149990;

    Leb_Grid_XYZW[  955][0] = -0.371496785943674;
    Leb_Grid_XYZW[  955][1] =  0.791606824725366;
    Leb_Grid_XYZW[  955][2] = -0.485127584334002;
    Leb_Grid_XYZW[  955][3] =  0.000533204149990;

    Leb_Grid_XYZW[  956][0] = -0.371496785943674;
    Leb_Grid_XYZW[  956][1] = -0.791606824725365;
    Leb_Grid_XYZW[  956][2] =  0.485127584334002;
    Leb_Grid_XYZW[  956][3] =  0.000533204149990;

    Leb_Grid_XYZW[  957][0] = -0.371496785943674;
    Leb_Grid_XYZW[  957][1] = -0.791606824725366;
    Leb_Grid_XYZW[  957][2] = -0.485127584334002;
    Leb_Grid_XYZW[  957][3] =  0.000533204149990;

    Leb_Grid_XYZW[  958][0] =  0.791606824725365;
    Leb_Grid_XYZW[  958][1] =  0.485127584334002;
    Leb_Grid_XYZW[  958][2] =  0.371496785943674;
    Leb_Grid_XYZW[  958][3] =  0.000533204149990;

    Leb_Grid_XYZW[  959][0] =  0.791606824725365;
    Leb_Grid_XYZW[  959][1] =  0.485127584334002;
    Leb_Grid_XYZW[  959][2] = -0.371496785943674;
    Leb_Grid_XYZW[  959][3] =  0.000533204149990;

    Leb_Grid_XYZW[  960][0] =  0.791606824725365;
    Leb_Grid_XYZW[  960][1] = -0.485127584334002;
    Leb_Grid_XYZW[  960][2] =  0.371496785943674;
    Leb_Grid_XYZW[  960][3] =  0.000533204149990;

    Leb_Grid_XYZW[  961][0] =  0.791606824725365;
    Leb_Grid_XYZW[  961][1] = -0.485127584334002;
    Leb_Grid_XYZW[  961][2] = -0.371496785943674;
    Leb_Grid_XYZW[  961][3] =  0.000533204149990;

    Leb_Grid_XYZW[  962][0] = -0.791606824725365;
    Leb_Grid_XYZW[  962][1] =  0.485127584334002;
    Leb_Grid_XYZW[  962][2] =  0.371496785943674;
    Leb_Grid_XYZW[  962][3] =  0.000533204149990;

    Leb_Grid_XYZW[  963][0] = -0.791606824725365;
    Leb_Grid_XYZW[  963][1] =  0.485127584334002;
    Leb_Grid_XYZW[  963][2] = -0.371496785943674;
    Leb_Grid_XYZW[  963][3] =  0.000533204149990;

    Leb_Grid_XYZW[  964][0] = -0.791606824725365;
    Leb_Grid_XYZW[  964][1] = -0.485127584334002;
    Leb_Grid_XYZW[  964][2] =  0.371496785943674;
    Leb_Grid_XYZW[  964][3] =  0.000533204149990;

    Leb_Grid_XYZW[  965][0] = -0.791606824725365;
    Leb_Grid_XYZW[  965][1] = -0.485127584334002;
    Leb_Grid_XYZW[  965][2] = -0.371496785943674;
    Leb_Grid_XYZW[  965][3] =  0.000533204149990;

    Leb_Grid_XYZW[  966][0] =  0.791606824725365;
    Leb_Grid_XYZW[  966][1] =  0.371496785943674;
    Leb_Grid_XYZW[  966][2] =  0.485127584334002;
    Leb_Grid_XYZW[  966][3] =  0.000533204149990;

    Leb_Grid_XYZW[  967][0] =  0.791606824725366;
    Leb_Grid_XYZW[  967][1] =  0.371496785943674;
    Leb_Grid_XYZW[  967][2] = -0.485127584334002;
    Leb_Grid_XYZW[  967][3] =  0.000533204149990;

    Leb_Grid_XYZW[  968][0] =  0.791606824725365;
    Leb_Grid_XYZW[  968][1] = -0.371496785943674;
    Leb_Grid_XYZW[  968][2] =  0.485127584334002;
    Leb_Grid_XYZW[  968][3] =  0.000533204149990;

    Leb_Grid_XYZW[  969][0] =  0.791606824725366;
    Leb_Grid_XYZW[  969][1] = -0.371496785943674;
    Leb_Grid_XYZW[  969][2] = -0.485127584334002;
    Leb_Grid_XYZW[  969][3] =  0.000533204149990;

    Leb_Grid_XYZW[  970][0] = -0.791606824725365;
    Leb_Grid_XYZW[  970][1] =  0.371496785943675;
    Leb_Grid_XYZW[  970][2] =  0.485127584334002;
    Leb_Grid_XYZW[  970][3] =  0.000533204149990;

    Leb_Grid_XYZW[  971][0] = -0.791606824725365;
    Leb_Grid_XYZW[  971][1] =  0.371496785943675;
    Leb_Grid_XYZW[  971][2] = -0.485127584334002;
    Leb_Grid_XYZW[  971][3] =  0.000533204149990;

    Leb_Grid_XYZW[  972][0] = -0.791606824725365;
    Leb_Grid_XYZW[  972][1] = -0.371496785943675;
    Leb_Grid_XYZW[  972][2] =  0.485127584334002;
    Leb_Grid_XYZW[  972][3] =  0.000533204149990;

    Leb_Grid_XYZW[  973][0] = -0.791606824725365;
    Leb_Grid_XYZW[  973][1] = -0.371496785943675;
    Leb_Grid_XYZW[  973][2] = -0.485127584334002;
    Leb_Grid_XYZW[  973][3] =  0.000533204149990;

    Leb_Grid_XYZW[  974][0] =  0.530339180380687;
    Leb_Grid_XYZW[  974][1] =  0.417535364632174;
    Leb_Grid_XYZW[  974][2] =  0.737837768777540;
    Leb_Grid_XYZW[  974][3] =  0.000538458312602;

    Leb_Grid_XYZW[  975][0] =  0.530339180380687;
    Leb_Grid_XYZW[  975][1] =  0.417535364632174;
    Leb_Grid_XYZW[  975][2] = -0.737837768777540;
    Leb_Grid_XYZW[  975][3] =  0.000538458312602;

    Leb_Grid_XYZW[  976][0] =  0.530339180380687;
    Leb_Grid_XYZW[  976][1] = -0.417535364632174;
    Leb_Grid_XYZW[  976][2] =  0.737837768777540;
    Leb_Grid_XYZW[  976][3] =  0.000538458312602;

    Leb_Grid_XYZW[  977][0] =  0.530339180380687;
    Leb_Grid_XYZW[  977][1] = -0.417535364632174;
    Leb_Grid_XYZW[  977][2] = -0.737837768777540;
    Leb_Grid_XYZW[  977][3] =  0.000538458312602;

    Leb_Grid_XYZW[  978][0] = -0.530339180380687;
    Leb_Grid_XYZW[  978][1] =  0.417535364632174;
    Leb_Grid_XYZW[  978][2] =  0.737837768777540;
    Leb_Grid_XYZW[  978][3] =  0.000538458312602;

    Leb_Grid_XYZW[  979][0] = -0.530339180380687;
    Leb_Grid_XYZW[  979][1] =  0.417535364632174;
    Leb_Grid_XYZW[  979][2] = -0.737837768777540;
    Leb_Grid_XYZW[  979][3] =  0.000538458312602;

    Leb_Grid_XYZW[  980][0] = -0.530339180380687;
    Leb_Grid_XYZW[  980][1] = -0.417535364632174;
    Leb_Grid_XYZW[  980][2] =  0.737837768777540;
    Leb_Grid_XYZW[  980][3] =  0.000538458312602;

    Leb_Grid_XYZW[  981][0] = -0.530339180380687;
    Leb_Grid_XYZW[  981][1] = -0.417535364632174;
    Leb_Grid_XYZW[  981][2] = -0.737837768777540;
    Leb_Grid_XYZW[  981][3] =  0.000538458312602;

    Leb_Grid_XYZW[  982][0] =  0.530339180380687;
    Leb_Grid_XYZW[  982][1] =  0.737837768777540;
    Leb_Grid_XYZW[  982][2] =  0.417535364632174;
    Leb_Grid_XYZW[  982][3] =  0.000538458312602;

    Leb_Grid_XYZW[  983][0] =  0.530339180380687;
    Leb_Grid_XYZW[  983][1] =  0.737837768777540;
    Leb_Grid_XYZW[  983][2] = -0.417535364632175;
    Leb_Grid_XYZW[  983][3] =  0.000538458312602;

    Leb_Grid_XYZW[  984][0] =  0.530339180380687;
    Leb_Grid_XYZW[  984][1] = -0.737837768777540;
    Leb_Grid_XYZW[  984][2] =  0.417535364632174;
    Leb_Grid_XYZW[  984][3] =  0.000538458312602;

    Leb_Grid_XYZW[  985][0] =  0.530339180380687;
    Leb_Grid_XYZW[  985][1] = -0.737837768777540;
    Leb_Grid_XYZW[  985][2] = -0.417535364632175;
    Leb_Grid_XYZW[  985][3] =  0.000538458312602;

    Leb_Grid_XYZW[  986][0] = -0.530339180380687;
    Leb_Grid_XYZW[  986][1] =  0.737837768777540;
    Leb_Grid_XYZW[  986][2] =  0.417535364632174;
    Leb_Grid_XYZW[  986][3] =  0.000538458312602;

    Leb_Grid_XYZW[  987][0] = -0.530339180380687;
    Leb_Grid_XYZW[  987][1] =  0.737837768777540;
    Leb_Grid_XYZW[  987][2] = -0.417535364632175;
    Leb_Grid_XYZW[  987][3] =  0.000538458312602;

    Leb_Grid_XYZW[  988][0] = -0.530339180380687;
    Leb_Grid_XYZW[  988][1] = -0.737837768777540;
    Leb_Grid_XYZW[  988][2] =  0.417535364632174;
    Leb_Grid_XYZW[  988][3] =  0.000538458312602;

    Leb_Grid_XYZW[  989][0] = -0.530339180380687;
    Leb_Grid_XYZW[  989][1] = -0.737837768777540;
    Leb_Grid_XYZW[  989][2] = -0.417535364632175;
    Leb_Grid_XYZW[  989][3] =  0.000538458312602;

    Leb_Grid_XYZW[  990][0] =  0.417535364632174;
    Leb_Grid_XYZW[  990][1] =  0.530339180380687;
    Leb_Grid_XYZW[  990][2] =  0.737837768777540;
    Leb_Grid_XYZW[  990][3] =  0.000538458312602;

    Leb_Grid_XYZW[  991][0] =  0.417535364632174;
    Leb_Grid_XYZW[  991][1] =  0.530339180380687;
    Leb_Grid_XYZW[  991][2] = -0.737837768777540;
    Leb_Grid_XYZW[  991][3] =  0.000538458312602;

    Leb_Grid_XYZW[  992][0] =  0.417535364632174;
    Leb_Grid_XYZW[  992][1] = -0.530339180380687;
    Leb_Grid_XYZW[  992][2] =  0.737837768777540;
    Leb_Grid_XYZW[  992][3] =  0.000538458312602;

    Leb_Grid_XYZW[  993][0] =  0.417535364632174;
    Leb_Grid_XYZW[  993][1] = -0.530339180380687;
    Leb_Grid_XYZW[  993][2] = -0.737837768777540;
    Leb_Grid_XYZW[  993][3] =  0.000538458312602;

    Leb_Grid_XYZW[  994][0] = -0.417535364632174;
    Leb_Grid_XYZW[  994][1] =  0.530339180380687;
    Leb_Grid_XYZW[  994][2] =  0.737837768777540;
    Leb_Grid_XYZW[  994][3] =  0.000538458312602;

    Leb_Grid_XYZW[  995][0] = -0.417535364632174;
    Leb_Grid_XYZW[  995][1] =  0.530339180380687;
    Leb_Grid_XYZW[  995][2] = -0.737837768777540;
    Leb_Grid_XYZW[  995][3] =  0.000538458312602;

    Leb_Grid_XYZW[  996][0] = -0.417535364632174;
    Leb_Grid_XYZW[  996][1] = -0.530339180380687;
    Leb_Grid_XYZW[  996][2] =  0.737837768777540;
    Leb_Grid_XYZW[  996][3] =  0.000538458312602;

    Leb_Grid_XYZW[  997][0] = -0.417535364632174;
    Leb_Grid_XYZW[  997][1] = -0.530339180380687;
    Leb_Grid_XYZW[  997][2] = -0.737837768777540;
    Leb_Grid_XYZW[  997][3] =  0.000538458312602;

    Leb_Grid_XYZW[  998][0] =  0.417535364632175;
    Leb_Grid_XYZW[  998][1] =  0.737837768777540;
    Leb_Grid_XYZW[  998][2] =  0.530339180380687;
    Leb_Grid_XYZW[  998][3] =  0.000538458312602;

    Leb_Grid_XYZW[  999][0] =  0.417535364632175;
    Leb_Grid_XYZW[  999][1] =  0.737837768777540;
    Leb_Grid_XYZW[  999][2] = -0.530339180380687;
    Leb_Grid_XYZW[  999][3] =  0.000538458312602;

    Leb_Grid_XYZW[ 1000][0] =  0.417535364632175;
    Leb_Grid_XYZW[ 1000][1] = -0.737837768777540;
    Leb_Grid_XYZW[ 1000][2] =  0.530339180380687;
    Leb_Grid_XYZW[ 1000][3] =  0.000538458312602;

    Leb_Grid_XYZW[ 1001][0] =  0.417535364632175;
    Leb_Grid_XYZW[ 1001][1] = -0.737837768777540;
    Leb_Grid_XYZW[ 1001][2] = -0.530339180380687;
    Leb_Grid_XYZW[ 1001][3] =  0.000538458312602;

    Leb_Grid_XYZW[ 1002][0] = -0.417535364632175;
    Leb_Grid_XYZW[ 1002][1] =  0.737837768777540;
    Leb_Grid_XYZW[ 1002][2] =  0.530339180380687;
    Leb_Grid_XYZW[ 1002][3] =  0.000538458312602;

    Leb_Grid_XYZW[ 1003][0] = -0.417535364632175;
    Leb_Grid_XYZW[ 1003][1] =  0.737837768777540;
    Leb_Grid_XYZW[ 1003][2] = -0.530339180380687;
    Leb_Grid_XYZW[ 1003][3] =  0.000538458312602;

    Leb_Grid_XYZW[ 1004][0] = -0.417535364632175;
    Leb_Grid_XYZW[ 1004][1] = -0.737837768777540;
    Leb_Grid_XYZW[ 1004][2] =  0.530339180380687;
    Leb_Grid_XYZW[ 1004][3] =  0.000538458312602;

    Leb_Grid_XYZW[ 1005][0] = -0.417535364632175;
    Leb_Grid_XYZW[ 1005][1] = -0.737837768777540;
    Leb_Grid_XYZW[ 1005][2] = -0.530339180380687;
    Leb_Grid_XYZW[ 1005][3] =  0.000538458312602;

    Leb_Grid_XYZW[ 1006][0] =  0.737837768777540;
    Leb_Grid_XYZW[ 1006][1] =  0.530339180380687;
    Leb_Grid_XYZW[ 1006][2] =  0.417535364632174;
    Leb_Grid_XYZW[ 1006][3] =  0.000538458312602;

    Leb_Grid_XYZW[ 1007][0] =  0.737837768777540;
    Leb_Grid_XYZW[ 1007][1] =  0.530339180380687;
    Leb_Grid_XYZW[ 1007][2] = -0.417535364632175;
    Leb_Grid_XYZW[ 1007][3] =  0.000538458312602;

    Leb_Grid_XYZW[ 1008][0] =  0.737837768777540;
    Leb_Grid_XYZW[ 1008][1] = -0.530339180380687;
    Leb_Grid_XYZW[ 1008][2] =  0.417535364632174;
    Leb_Grid_XYZW[ 1008][3] =  0.000538458312602;

    Leb_Grid_XYZW[ 1009][0] =  0.737837768777540;
    Leb_Grid_XYZW[ 1009][1] = -0.530339180380687;
    Leb_Grid_XYZW[ 1009][2] = -0.417535364632175;
    Leb_Grid_XYZW[ 1009][3] =  0.000538458312602;

    Leb_Grid_XYZW[ 1010][0] = -0.737837768777540;
    Leb_Grid_XYZW[ 1010][1] =  0.530339180380687;
    Leb_Grid_XYZW[ 1010][2] =  0.417535364632174;
    Leb_Grid_XYZW[ 1010][3] =  0.000538458312602;

    Leb_Grid_XYZW[ 1011][0] = -0.737837768777540;
    Leb_Grid_XYZW[ 1011][1] =  0.530339180380687;
    Leb_Grid_XYZW[ 1011][2] = -0.417535364632175;
    Leb_Grid_XYZW[ 1011][3] =  0.000538458312602;

    Leb_Grid_XYZW[ 1012][0] = -0.737837768777540;
    Leb_Grid_XYZW[ 1012][1] = -0.530339180380687;
    Leb_Grid_XYZW[ 1012][2] =  0.417535364632174;
    Leb_Grid_XYZW[ 1012][3] =  0.000538458312602;

    Leb_Grid_XYZW[ 1013][0] = -0.737837768777540;
    Leb_Grid_XYZW[ 1013][1] = -0.530339180380687;
    Leb_Grid_XYZW[ 1013][2] = -0.417535364632175;
    Leb_Grid_XYZW[ 1013][3] =  0.000538458312602;

    Leb_Grid_XYZW[ 1014][0] =  0.737837768777540;
    Leb_Grid_XYZW[ 1014][1] =  0.417535364632174;
    Leb_Grid_XYZW[ 1014][2] =  0.530339180380687;
    Leb_Grid_XYZW[ 1014][3] =  0.000538458312602;

    Leb_Grid_XYZW[ 1015][0] =  0.737837768777540;
    Leb_Grid_XYZW[ 1015][1] =  0.417535364632174;
    Leb_Grid_XYZW[ 1015][2] = -0.530339180380687;
    Leb_Grid_XYZW[ 1015][3] =  0.000538458312602;

    Leb_Grid_XYZW[ 1016][0] =  0.737837768777540;
    Leb_Grid_XYZW[ 1016][1] = -0.417535364632174;
    Leb_Grid_XYZW[ 1016][2] =  0.530339180380687;
    Leb_Grid_XYZW[ 1016][3] =  0.000538458312602;

    Leb_Grid_XYZW[ 1017][0] =  0.737837768777540;
    Leb_Grid_XYZW[ 1017][1] = -0.417535364632174;
    Leb_Grid_XYZW[ 1017][2] = -0.530339180380687;
    Leb_Grid_XYZW[ 1017][3] =  0.000538458312602;

    Leb_Grid_XYZW[ 1018][0] = -0.737837768777540;
    Leb_Grid_XYZW[ 1018][1] =  0.417535364632175;
    Leb_Grid_XYZW[ 1018][2] =  0.530339180380687;
    Leb_Grid_XYZW[ 1018][3] =  0.000538458312602;

    Leb_Grid_XYZW[ 1019][0] = -0.737837768777540;
    Leb_Grid_XYZW[ 1019][1] =  0.417535364632175;
    Leb_Grid_XYZW[ 1019][2] = -0.530339180380687;
    Leb_Grid_XYZW[ 1019][3] =  0.000538458312602;

    Leb_Grid_XYZW[ 1020][0] = -0.737837768777540;
    Leb_Grid_XYZW[ 1020][1] = -0.417535364632175;
    Leb_Grid_XYZW[ 1020][2] =  0.530339180380687;
    Leb_Grid_XYZW[ 1020][3] =  0.000538458312602;

    Leb_Grid_XYZW[ 1021][0] = -0.737837768777540;
    Leb_Grid_XYZW[ 1021][1] = -0.417535364632175;
    Leb_Grid_XYZW[ 1021][2] = -0.530339180380687;
    Leb_Grid_XYZW[ 1021][3] =  0.000538458312602;

    Leb_Grid_XYZW[ 1022][0] =  0.572619738059629;
    Leb_Grid_XYZW[ 1022][1] =  0.461208440635546;
    Leb_Grid_XYZW[ 1022][2] =  0.677785666616704;
    Leb_Grid_XYZW[ 1022][3] =  0.000541106721080;

    Leb_Grid_XYZW[ 1023][0] =  0.572619738059628;
    Leb_Grid_XYZW[ 1023][1] =  0.461208440635546;
    Leb_Grid_XYZW[ 1023][2] = -0.677785666616705;
    Leb_Grid_XYZW[ 1023][3] =  0.000541106721080;

    Leb_Grid_XYZW[ 1024][0] =  0.572619738059629;
    Leb_Grid_XYZW[ 1024][1] = -0.461208440635546;
    Leb_Grid_XYZW[ 1024][2] =  0.677785666616704;
    Leb_Grid_XYZW[ 1024][3] =  0.000541106721080;

    Leb_Grid_XYZW[ 1025][0] =  0.572619738059628;
    Leb_Grid_XYZW[ 1025][1] = -0.461208440635546;
    Leb_Grid_XYZW[ 1025][2] = -0.677785666616705;
    Leb_Grid_XYZW[ 1025][3] =  0.000541106721080;

    Leb_Grid_XYZW[ 1026][0] = -0.572619738059629;
    Leb_Grid_XYZW[ 1026][1] =  0.461208440635546;
    Leb_Grid_XYZW[ 1026][2] =  0.677785666616704;
    Leb_Grid_XYZW[ 1026][3] =  0.000541106721080;

    Leb_Grid_XYZW[ 1027][0] = -0.572619738059628;
    Leb_Grid_XYZW[ 1027][1] =  0.461208440635546;
    Leb_Grid_XYZW[ 1027][2] = -0.677785666616705;
    Leb_Grid_XYZW[ 1027][3] =  0.000541106721080;

    Leb_Grid_XYZW[ 1028][0] = -0.572619738059629;
    Leb_Grid_XYZW[ 1028][1] = -0.461208440635546;
    Leb_Grid_XYZW[ 1028][2] =  0.677785666616704;
    Leb_Grid_XYZW[ 1028][3] =  0.000541106721080;

    Leb_Grid_XYZW[ 1029][0] = -0.572619738059628;
    Leb_Grid_XYZW[ 1029][1] = -0.461208440635546;
    Leb_Grid_XYZW[ 1029][2] = -0.677785666616705;
    Leb_Grid_XYZW[ 1029][3] =  0.000541106721080;

    Leb_Grid_XYZW[ 1030][0] =  0.572619738059629;
    Leb_Grid_XYZW[ 1030][1] =  0.677785666616704;
    Leb_Grid_XYZW[ 1030][2] =  0.461208440635546;
    Leb_Grid_XYZW[ 1030][3] =  0.000541106721080;

    Leb_Grid_XYZW[ 1031][0] =  0.572619738059629;
    Leb_Grid_XYZW[ 1031][1] =  0.677785666616704;
    Leb_Grid_XYZW[ 1031][2] = -0.461208440635546;
    Leb_Grid_XYZW[ 1031][3] =  0.000541106721080;

    Leb_Grid_XYZW[ 1032][0] =  0.572619738059629;
    Leb_Grid_XYZW[ 1032][1] = -0.677785666616704;
    Leb_Grid_XYZW[ 1032][2] =  0.461208440635546;
    Leb_Grid_XYZW[ 1032][3] =  0.000541106721080;

    Leb_Grid_XYZW[ 1033][0] =  0.572619738059629;
    Leb_Grid_XYZW[ 1033][1] = -0.677785666616704;
    Leb_Grid_XYZW[ 1033][2] = -0.461208440635546;
    Leb_Grid_XYZW[ 1033][3] =  0.000541106721080;

    Leb_Grid_XYZW[ 1034][0] = -0.572619738059628;
    Leb_Grid_XYZW[ 1034][1] =  0.677785666616704;
    Leb_Grid_XYZW[ 1034][2] =  0.461208440635546;
    Leb_Grid_XYZW[ 1034][3] =  0.000541106721080;

    Leb_Grid_XYZW[ 1035][0] = -0.572619738059629;
    Leb_Grid_XYZW[ 1035][1] =  0.677785666616704;
    Leb_Grid_XYZW[ 1035][2] = -0.461208440635546;
    Leb_Grid_XYZW[ 1035][3] =  0.000541106721080;

    Leb_Grid_XYZW[ 1036][0] = -0.572619738059628;
    Leb_Grid_XYZW[ 1036][1] = -0.677785666616704;
    Leb_Grid_XYZW[ 1036][2] =  0.461208440635546;
    Leb_Grid_XYZW[ 1036][3] =  0.000541106721080;

    Leb_Grid_XYZW[ 1037][0] = -0.572619738059629;
    Leb_Grid_XYZW[ 1037][1] = -0.677785666616704;
    Leb_Grid_XYZW[ 1037][2] = -0.461208440635546;
    Leb_Grid_XYZW[ 1037][3] =  0.000541106721080;

    Leb_Grid_XYZW[ 1038][0] =  0.461208440635546;
    Leb_Grid_XYZW[ 1038][1] =  0.572619738059629;
    Leb_Grid_XYZW[ 1038][2] =  0.677785666616704;
    Leb_Grid_XYZW[ 1038][3] =  0.000541106721080;

    Leb_Grid_XYZW[ 1039][0] =  0.461208440635546;
    Leb_Grid_XYZW[ 1039][1] =  0.572619738059628;
    Leb_Grid_XYZW[ 1039][2] = -0.677785666616705;
    Leb_Grid_XYZW[ 1039][3] =  0.000541106721080;

    Leb_Grid_XYZW[ 1040][0] =  0.461208440635546;
    Leb_Grid_XYZW[ 1040][1] = -0.572619738059629;
    Leb_Grid_XYZW[ 1040][2] =  0.677785666616704;
    Leb_Grid_XYZW[ 1040][3] =  0.000541106721080;

    Leb_Grid_XYZW[ 1041][0] =  0.461208440635546;
    Leb_Grid_XYZW[ 1041][1] = -0.572619738059628;
    Leb_Grid_XYZW[ 1041][2] = -0.677785666616705;
    Leb_Grid_XYZW[ 1041][3] =  0.000541106721080;

    Leb_Grid_XYZW[ 1042][0] = -0.461208440635546;
    Leb_Grid_XYZW[ 1042][1] =  0.572619738059629;
    Leb_Grid_XYZW[ 1042][2] =  0.677785666616704;
    Leb_Grid_XYZW[ 1042][3] =  0.000541106721080;

    Leb_Grid_XYZW[ 1043][0] = -0.461208440635546;
    Leb_Grid_XYZW[ 1043][1] =  0.572619738059628;
    Leb_Grid_XYZW[ 1043][2] = -0.677785666616705;
    Leb_Grid_XYZW[ 1043][3] =  0.000541106721080;

    Leb_Grid_XYZW[ 1044][0] = -0.461208440635546;
    Leb_Grid_XYZW[ 1044][1] = -0.572619738059629;
    Leb_Grid_XYZW[ 1044][2] =  0.677785666616704;
    Leb_Grid_XYZW[ 1044][3] =  0.000541106721080;

    Leb_Grid_XYZW[ 1045][0] = -0.461208440635546;
    Leb_Grid_XYZW[ 1045][1] = -0.572619738059628;
    Leb_Grid_XYZW[ 1045][2] = -0.677785666616705;
    Leb_Grid_XYZW[ 1045][3] =  0.000541106721080;

    Leb_Grid_XYZW[ 1046][0] =  0.461208440635546;
    Leb_Grid_XYZW[ 1046][1] =  0.677785666616704;
    Leb_Grid_XYZW[ 1046][2] =  0.572619738059629;
    Leb_Grid_XYZW[ 1046][3] =  0.000541106721080;

    Leb_Grid_XYZW[ 1047][0] =  0.461208440635546;
    Leb_Grid_XYZW[ 1047][1] =  0.677785666616704;
    Leb_Grid_XYZW[ 1047][2] = -0.572619738059629;
    Leb_Grid_XYZW[ 1047][3] =  0.000541106721080;

    Leb_Grid_XYZW[ 1048][0] =  0.461208440635546;
    Leb_Grid_XYZW[ 1048][1] = -0.677785666616704;
    Leb_Grid_XYZW[ 1048][2] =  0.572619738059629;
    Leb_Grid_XYZW[ 1048][3] =  0.000541106721080;

    Leb_Grid_XYZW[ 1049][0] =  0.461208440635546;
    Leb_Grid_XYZW[ 1049][1] = -0.677785666616704;
    Leb_Grid_XYZW[ 1049][2] = -0.572619738059629;
    Leb_Grid_XYZW[ 1049][3] =  0.000541106721080;

    Leb_Grid_XYZW[ 1050][0] = -0.461208440635546;
    Leb_Grid_XYZW[ 1050][1] =  0.677785666616704;
    Leb_Grid_XYZW[ 1050][2] =  0.572619738059629;
    Leb_Grid_XYZW[ 1050][3] =  0.000541106721080;

    Leb_Grid_XYZW[ 1051][0] = -0.461208440635546;
    Leb_Grid_XYZW[ 1051][1] =  0.677785666616704;
    Leb_Grid_XYZW[ 1051][2] = -0.572619738059629;
    Leb_Grid_XYZW[ 1051][3] =  0.000541106721080;

    Leb_Grid_XYZW[ 1052][0] = -0.461208440635546;
    Leb_Grid_XYZW[ 1052][1] = -0.677785666616704;
    Leb_Grid_XYZW[ 1052][2] =  0.572619738059629;
    Leb_Grid_XYZW[ 1052][3] =  0.000541106721080;

    Leb_Grid_XYZW[ 1053][0] = -0.461208440635546;
    Leb_Grid_XYZW[ 1053][1] = -0.677785666616704;
    Leb_Grid_XYZW[ 1053][2] = -0.572619738059629;
    Leb_Grid_XYZW[ 1053][3] =  0.000541106721080;

    Leb_Grid_XYZW[ 1054][0] =  0.677785666616704;
    Leb_Grid_XYZW[ 1054][1] =  0.572619738059629;
    Leb_Grid_XYZW[ 1054][2] =  0.461208440635546;
    Leb_Grid_XYZW[ 1054][3] =  0.000541106721080;

    Leb_Grid_XYZW[ 1055][0] =  0.677785666616704;
    Leb_Grid_XYZW[ 1055][1] =  0.572619738059629;
    Leb_Grid_XYZW[ 1055][2] = -0.461208440635546;
    Leb_Grid_XYZW[ 1055][3] =  0.000541106721080;

    Leb_Grid_XYZW[ 1056][0] =  0.677785666616704;
    Leb_Grid_XYZW[ 1056][1] = -0.572619738059629;
    Leb_Grid_XYZW[ 1056][2] =  0.461208440635546;
    Leb_Grid_XYZW[ 1056][3] =  0.000541106721080;

    Leb_Grid_XYZW[ 1057][0] =  0.677785666616704;
    Leb_Grid_XYZW[ 1057][1] = -0.572619738059629;
    Leb_Grid_XYZW[ 1057][2] = -0.461208440635546;
    Leb_Grid_XYZW[ 1057][3] =  0.000541106721080;

    Leb_Grid_XYZW[ 1058][0] = -0.677785666616704;
    Leb_Grid_XYZW[ 1058][1] =  0.572619738059629;
    Leb_Grid_XYZW[ 1058][2] =  0.461208440635546;
    Leb_Grid_XYZW[ 1058][3] =  0.000541106721080;

    Leb_Grid_XYZW[ 1059][0] = -0.677785666616704;
    Leb_Grid_XYZW[ 1059][1] =  0.572619738059629;
    Leb_Grid_XYZW[ 1059][2] = -0.461208440635546;
    Leb_Grid_XYZW[ 1059][3] =  0.000541106721080;

    Leb_Grid_XYZW[ 1060][0] = -0.677785666616704;
    Leb_Grid_XYZW[ 1060][1] = -0.572619738059629;
    Leb_Grid_XYZW[ 1060][2] =  0.461208440635546;
    Leb_Grid_XYZW[ 1060][3] =  0.000541106721080;

    Leb_Grid_XYZW[ 1061][0] = -0.677785666616704;
    Leb_Grid_XYZW[ 1061][1] = -0.572619738059629;
    Leb_Grid_XYZW[ 1061][2] = -0.461208440635546;
    Leb_Grid_XYZW[ 1061][3] =  0.000541106721080;

    Leb_Grid_XYZW[ 1062][0] =  0.677785666616704;
    Leb_Grid_XYZW[ 1062][1] =  0.461208440635546;
    Leb_Grid_XYZW[ 1062][2] =  0.572619738059629;
    Leb_Grid_XYZW[ 1062][3] =  0.000541106721080;

    Leb_Grid_XYZW[ 1063][0] =  0.677785666616704;
    Leb_Grid_XYZW[ 1063][1] =  0.461208440635546;
    Leb_Grid_XYZW[ 1063][2] = -0.572619738059629;
    Leb_Grid_XYZW[ 1063][3] =  0.000541106721080;

    Leb_Grid_XYZW[ 1064][0] =  0.677785666616704;
    Leb_Grid_XYZW[ 1064][1] = -0.461208440635546;
    Leb_Grid_XYZW[ 1064][2] =  0.572619738059629;
    Leb_Grid_XYZW[ 1064][3] =  0.000541106721080;

    Leb_Grid_XYZW[ 1065][0] =  0.677785666616704;
    Leb_Grid_XYZW[ 1065][1] = -0.461208440635546;
    Leb_Grid_XYZW[ 1065][2] = -0.572619738059629;
    Leb_Grid_XYZW[ 1065][3] =  0.000541106721080;

    Leb_Grid_XYZW[ 1066][0] = -0.677785666616704;
    Leb_Grid_XYZW[ 1066][1] =  0.461208440635546;
    Leb_Grid_XYZW[ 1066][2] =  0.572619738059629;
    Leb_Grid_XYZW[ 1066][3] =  0.000541106721080;

    Leb_Grid_XYZW[ 1067][0] = -0.677785666616704;
    Leb_Grid_XYZW[ 1067][1] =  0.461208440635546;
    Leb_Grid_XYZW[ 1067][2] = -0.572619738059629;
    Leb_Grid_XYZW[ 1067][3] =  0.000541106721080;

    Leb_Grid_XYZW[ 1068][0] = -0.677785666616704;
    Leb_Grid_XYZW[ 1068][1] = -0.461208440635546;
    Leb_Grid_XYZW[ 1068][2] =  0.572619738059629;
    Leb_Grid_XYZW[ 1068][3] =  0.000541106721080;

    Leb_Grid_XYZW[ 1069][0] = -0.677785666616704;
    Leb_Grid_XYZW[ 1069][1] = -0.461208440635546;
    Leb_Grid_XYZW[ 1069][2] = -0.572619738059629;
    Leb_Grid_XYZW[ 1069][3] =  0.000541106721080;

    Leb_Grid_XYZW[ 1070][0] =  0.243152073256486;
    Leb_Grid_XYZW[ 1070][1] =  0.042580401330439;
    Leb_Grid_XYZW[ 1070][2] =  0.969053135123978;
    Leb_Grid_XYZW[ 1070][3] =  0.000425979739147;

    Leb_Grid_XYZW[ 1071][0] =  0.243152073256486;
    Leb_Grid_XYZW[ 1071][1] =  0.042580401330439;
    Leb_Grid_XYZW[ 1071][2] = -0.969053135123978;
    Leb_Grid_XYZW[ 1071][3] =  0.000425979739147;

    Leb_Grid_XYZW[ 1072][0] =  0.243152073256486;
    Leb_Grid_XYZW[ 1072][1] = -0.042580401330439;
    Leb_Grid_XYZW[ 1072][2] =  0.969053135123978;
    Leb_Grid_XYZW[ 1072][3] =  0.000425979739147;

    Leb_Grid_XYZW[ 1073][0] =  0.243152073256486;
    Leb_Grid_XYZW[ 1073][1] = -0.042580401330439;
    Leb_Grid_XYZW[ 1073][2] = -0.969053135123978;
    Leb_Grid_XYZW[ 1073][3] =  0.000425979739147;

    Leb_Grid_XYZW[ 1074][0] = -0.243152073256486;
    Leb_Grid_XYZW[ 1074][1] =  0.042580401330440;
    Leb_Grid_XYZW[ 1074][2] =  0.969053135123978;
    Leb_Grid_XYZW[ 1074][3] =  0.000425979739147;

    Leb_Grid_XYZW[ 1075][0] = -0.243152073256486;
    Leb_Grid_XYZW[ 1075][1] =  0.042580401330439;
    Leb_Grid_XYZW[ 1075][2] = -0.969053135123978;
    Leb_Grid_XYZW[ 1075][3] =  0.000425979739147;

    Leb_Grid_XYZW[ 1076][0] = -0.243152073256486;
    Leb_Grid_XYZW[ 1076][1] = -0.042580401330440;
    Leb_Grid_XYZW[ 1076][2] =  0.969053135123978;
    Leb_Grid_XYZW[ 1076][3] =  0.000425979739147;

    Leb_Grid_XYZW[ 1077][0] = -0.243152073256486;
    Leb_Grid_XYZW[ 1077][1] = -0.042580401330439;
    Leb_Grid_XYZW[ 1077][2] = -0.969053135123978;
    Leb_Grid_XYZW[ 1077][3] =  0.000425979739147;

    Leb_Grid_XYZW[ 1078][0] =  0.243152073256487;
    Leb_Grid_XYZW[ 1078][1] =  0.969053135123978;
    Leb_Grid_XYZW[ 1078][2] =  0.042580401330440;
    Leb_Grid_XYZW[ 1078][3] =  0.000425979739147;

    Leb_Grid_XYZW[ 1079][0] =  0.243152073256487;
    Leb_Grid_XYZW[ 1079][1] =  0.969053135123978;
    Leb_Grid_XYZW[ 1079][2] = -0.042580401330440;
    Leb_Grid_XYZW[ 1079][3] =  0.000425979739147;

    Leb_Grid_XYZW[ 1080][0] =  0.243152073256487;
    Leb_Grid_XYZW[ 1080][1] = -0.969053135123978;
    Leb_Grid_XYZW[ 1080][2] =  0.042580401330440;
    Leb_Grid_XYZW[ 1080][3] =  0.000425979739147;

    Leb_Grid_XYZW[ 1081][0] =  0.243152073256487;
    Leb_Grid_XYZW[ 1081][1] = -0.969053135123978;
    Leb_Grid_XYZW[ 1081][2] = -0.042580401330440;
    Leb_Grid_XYZW[ 1081][3] =  0.000425979739147;

    Leb_Grid_XYZW[ 1082][0] = -0.243152073256486;
    Leb_Grid_XYZW[ 1082][1] =  0.969053135123978;
    Leb_Grid_XYZW[ 1082][2] =  0.042580401330440;
    Leb_Grid_XYZW[ 1082][3] =  0.000425979739147;

    Leb_Grid_XYZW[ 1083][0] = -0.243152073256486;
    Leb_Grid_XYZW[ 1083][1] =  0.969053135123978;
    Leb_Grid_XYZW[ 1083][2] = -0.042580401330440;
    Leb_Grid_XYZW[ 1083][3] =  0.000425979739147;

    Leb_Grid_XYZW[ 1084][0] = -0.243152073256486;
    Leb_Grid_XYZW[ 1084][1] = -0.969053135123978;
    Leb_Grid_XYZW[ 1084][2] =  0.042580401330440;
    Leb_Grid_XYZW[ 1084][3] =  0.000425979739147;

    Leb_Grid_XYZW[ 1085][0] = -0.243152073256486;
    Leb_Grid_XYZW[ 1085][1] = -0.969053135123978;
    Leb_Grid_XYZW[ 1085][2] = -0.042580401330440;
    Leb_Grid_XYZW[ 1085][3] =  0.000425979739147;

    Leb_Grid_XYZW[ 1086][0] =  0.042580401330439;
    Leb_Grid_XYZW[ 1086][1] =  0.243152073256486;
    Leb_Grid_XYZW[ 1086][2] =  0.969053135123978;
    Leb_Grid_XYZW[ 1086][3] =  0.000425979739147;

    Leb_Grid_XYZW[ 1087][0] =  0.042580401330439;
    Leb_Grid_XYZW[ 1087][1] =  0.243152073256486;
    Leb_Grid_XYZW[ 1087][2] = -0.969053135123978;
    Leb_Grid_XYZW[ 1087][3] =  0.000425979739147;

    Leb_Grid_XYZW[ 1088][0] =  0.042580401330439;
    Leb_Grid_XYZW[ 1088][1] = -0.243152073256486;
    Leb_Grid_XYZW[ 1088][2] =  0.969053135123978;
    Leb_Grid_XYZW[ 1088][3] =  0.000425979739147;

    Leb_Grid_XYZW[ 1089][0] =  0.042580401330439;
    Leb_Grid_XYZW[ 1089][1] = -0.243152073256486;
    Leb_Grid_XYZW[ 1089][2] = -0.969053135123978;
    Leb_Grid_XYZW[ 1089][3] =  0.000425979739147;

    Leb_Grid_XYZW[ 1090][0] = -0.042580401330439;
    Leb_Grid_XYZW[ 1090][1] =  0.243152073256486;
    Leb_Grid_XYZW[ 1090][2] =  0.969053135123978;
    Leb_Grid_XYZW[ 1090][3] =  0.000425979739147;

    Leb_Grid_XYZW[ 1091][0] = -0.042580401330439;
    Leb_Grid_XYZW[ 1091][1] =  0.243152073256486;
    Leb_Grid_XYZW[ 1091][2] = -0.969053135123978;
    Leb_Grid_XYZW[ 1091][3] =  0.000425979739147;

    Leb_Grid_XYZW[ 1092][0] = -0.042580401330439;
    Leb_Grid_XYZW[ 1092][1] = -0.243152073256486;
    Leb_Grid_XYZW[ 1092][2] =  0.969053135123978;
    Leb_Grid_XYZW[ 1092][3] =  0.000425979739147;

    Leb_Grid_XYZW[ 1093][0] = -0.042580401330439;
    Leb_Grid_XYZW[ 1093][1] = -0.243152073256486;
    Leb_Grid_XYZW[ 1093][2] = -0.969053135123978;
    Leb_Grid_XYZW[ 1093][3] =  0.000425979739147;

    Leb_Grid_XYZW[ 1094][0] =  0.042580401330439;
    Leb_Grid_XYZW[ 1094][1] =  0.969053135123978;
    Leb_Grid_XYZW[ 1094][2] =  0.243152073256486;
    Leb_Grid_XYZW[ 1094][3] =  0.000425979739147;

    Leb_Grid_XYZW[ 1095][0] =  0.042580401330439;
    Leb_Grid_XYZW[ 1095][1] =  0.969053135123978;
    Leb_Grid_XYZW[ 1095][2] = -0.243152073256486;
    Leb_Grid_XYZW[ 1095][3] =  0.000425979739147;

    Leb_Grid_XYZW[ 1096][0] =  0.042580401330439;
    Leb_Grid_XYZW[ 1096][1] = -0.969053135123978;
    Leb_Grid_XYZW[ 1096][2] =  0.243152073256486;
    Leb_Grid_XYZW[ 1096][3] =  0.000425979739147;

    Leb_Grid_XYZW[ 1097][0] =  0.042580401330439;
    Leb_Grid_XYZW[ 1097][1] = -0.969053135123978;
    Leb_Grid_XYZW[ 1097][2] = -0.243152073256486;
    Leb_Grid_XYZW[ 1097][3] =  0.000425979739147;

    Leb_Grid_XYZW[ 1098][0] = -0.042580401330439;
    Leb_Grid_XYZW[ 1098][1] =  0.969053135123978;
    Leb_Grid_XYZW[ 1098][2] =  0.243152073256486;
    Leb_Grid_XYZW[ 1098][3] =  0.000425979739147;

    Leb_Grid_XYZW[ 1099][0] = -0.042580401330439;
    Leb_Grid_XYZW[ 1099][1] =  0.969053135123978;
    Leb_Grid_XYZW[ 1099][2] = -0.243152073256486;
    Leb_Grid_XYZW[ 1099][3] =  0.000425979739147;

    Leb_Grid_XYZW[ 1100][0] = -0.042580401330439;
    Leb_Grid_XYZW[ 1100][1] = -0.969053135123978;
    Leb_Grid_XYZW[ 1100][2] =  0.243152073256486;
    Leb_Grid_XYZW[ 1100][3] =  0.000425979739147;

    Leb_Grid_XYZW[ 1101][0] = -0.042580401330439;
    Leb_Grid_XYZW[ 1101][1] = -0.969053135123978;
    Leb_Grid_XYZW[ 1101][2] = -0.243152073256486;
    Leb_Grid_XYZW[ 1101][3] =  0.000425979739147;

    Leb_Grid_XYZW[ 1102][0] =  0.969053135123978;
    Leb_Grid_XYZW[ 1102][1] =  0.243152073256486;
    Leb_Grid_XYZW[ 1102][2] =  0.042580401330440;
    Leb_Grid_XYZW[ 1102][3] =  0.000425979739147;

    Leb_Grid_XYZW[ 1103][0] =  0.969053135123978;
    Leb_Grid_XYZW[ 1103][1] =  0.243152073256486;
    Leb_Grid_XYZW[ 1103][2] = -0.042580401330440;
    Leb_Grid_XYZW[ 1103][3] =  0.000425979739147;

    Leb_Grid_XYZW[ 1104][0] =  0.969053135123978;
    Leb_Grid_XYZW[ 1104][1] = -0.243152073256486;
    Leb_Grid_XYZW[ 1104][2] =  0.042580401330440;
    Leb_Grid_XYZW[ 1104][3] =  0.000425979739147;

    Leb_Grid_XYZW[ 1105][0] =  0.969053135123978;
    Leb_Grid_XYZW[ 1105][1] = -0.243152073256486;
    Leb_Grid_XYZW[ 1105][2] = -0.042580401330440;
    Leb_Grid_XYZW[ 1105][3] =  0.000425979739147;

    Leb_Grid_XYZW[ 1106][0] = -0.969053135123978;
    Leb_Grid_XYZW[ 1106][1] =  0.243152073256486;
    Leb_Grid_XYZW[ 1106][2] =  0.042580401330440;
    Leb_Grid_XYZW[ 1106][3] =  0.000425979739147;

    Leb_Grid_XYZW[ 1107][0] = -0.969053135123978;
    Leb_Grid_XYZW[ 1107][1] =  0.243152073256486;
    Leb_Grid_XYZW[ 1107][2] = -0.042580401330440;
    Leb_Grid_XYZW[ 1107][3] =  0.000425979739147;

    Leb_Grid_XYZW[ 1108][0] = -0.969053135123978;
    Leb_Grid_XYZW[ 1108][1] = -0.243152073256486;
    Leb_Grid_XYZW[ 1108][2] =  0.042580401330440;
    Leb_Grid_XYZW[ 1108][3] =  0.000425979739147;

    Leb_Grid_XYZW[ 1109][0] = -0.969053135123978;
    Leb_Grid_XYZW[ 1109][1] = -0.243152073256486;
    Leb_Grid_XYZW[ 1109][2] = -0.042580401330440;
    Leb_Grid_XYZW[ 1109][3] =  0.000425979739147;

    Leb_Grid_XYZW[ 1110][0] =  0.969053135123978;
    Leb_Grid_XYZW[ 1110][1] =  0.042580401330439;
    Leb_Grid_XYZW[ 1110][2] =  0.243152073256486;
    Leb_Grid_XYZW[ 1110][3] =  0.000425979739147;

    Leb_Grid_XYZW[ 1111][0] =  0.969053135123978;
    Leb_Grid_XYZW[ 1111][1] =  0.042580401330439;
    Leb_Grid_XYZW[ 1111][2] = -0.243152073256486;
    Leb_Grid_XYZW[ 1111][3] =  0.000425979739147;

    Leb_Grid_XYZW[ 1112][0] =  0.969053135123978;
    Leb_Grid_XYZW[ 1112][1] = -0.042580401330439;
    Leb_Grid_XYZW[ 1112][2] =  0.243152073256486;
    Leb_Grid_XYZW[ 1112][3] =  0.000425979739147;

    Leb_Grid_XYZW[ 1113][0] =  0.969053135123978;
    Leb_Grid_XYZW[ 1113][1] = -0.042580401330439;
    Leb_Grid_XYZW[ 1113][2] = -0.243152073256486;
    Leb_Grid_XYZW[ 1113][3] =  0.000425979739147;

    Leb_Grid_XYZW[ 1114][0] = -0.969053135123978;
    Leb_Grid_XYZW[ 1114][1] =  0.042580401330439;
    Leb_Grid_XYZW[ 1114][2] =  0.243152073256486;
    Leb_Grid_XYZW[ 1114][3] =  0.000425979739147;

    Leb_Grid_XYZW[ 1115][0] = -0.969053135123978;
    Leb_Grid_XYZW[ 1115][1] =  0.042580401330439;
    Leb_Grid_XYZW[ 1115][2] = -0.243152073256486;
    Leb_Grid_XYZW[ 1115][3] =  0.000425979739147;

    Leb_Grid_XYZW[ 1116][0] = -0.969053135123978;
    Leb_Grid_XYZW[ 1116][1] = -0.042580401330439;
    Leb_Grid_XYZW[ 1116][2] =  0.243152073256486;
    Leb_Grid_XYZW[ 1116][3] =  0.000425979739147;

    Leb_Grid_XYZW[ 1117][0] = -0.969053135123978;
    Leb_Grid_XYZW[ 1117][1] = -0.042580401330439;
    Leb_Grid_XYZW[ 1117][2] = -0.243152073256486;
    Leb_Grid_XYZW[ 1117][3] =  0.000425979739147;

    Leb_Grid_XYZW[ 1118][0] =  0.300209680089587;
    Leb_Grid_XYZW[ 1118][1] =  0.088694243067227;
    Leb_Grid_XYZW[ 1118][2] =  0.949740743164807;
    Leb_Grid_XYZW[ 1118][3] =  0.000460493136846;

    Leb_Grid_XYZW[ 1119][0] =  0.300209680089587;
    Leb_Grid_XYZW[ 1119][1] =  0.088694243067227;
    Leb_Grid_XYZW[ 1119][2] = -0.949740743164807;
    Leb_Grid_XYZW[ 1119][3] =  0.000460493136846;

    Leb_Grid_XYZW[ 1120][0] =  0.300209680089587;
    Leb_Grid_XYZW[ 1120][1] = -0.088694243067227;
    Leb_Grid_XYZW[ 1120][2] =  0.949740743164807;
    Leb_Grid_XYZW[ 1120][3] =  0.000460493136846;

    Leb_Grid_XYZW[ 1121][0] =  0.300209680089587;
    Leb_Grid_XYZW[ 1121][1] = -0.088694243067227;
    Leb_Grid_XYZW[ 1121][2] = -0.949740743164807;
    Leb_Grid_XYZW[ 1121][3] =  0.000460493136846;

    Leb_Grid_XYZW[ 1122][0] = -0.300209680089587;
    Leb_Grid_XYZW[ 1122][1] =  0.088694243067227;
    Leb_Grid_XYZW[ 1122][2] =  0.949740743164807;
    Leb_Grid_XYZW[ 1122][3] =  0.000460493136846;

    Leb_Grid_XYZW[ 1123][0] = -0.300209680089587;
    Leb_Grid_XYZW[ 1123][1] =  0.088694243067227;
    Leb_Grid_XYZW[ 1123][2] = -0.949740743164807;
    Leb_Grid_XYZW[ 1123][3] =  0.000460493136846;

    Leb_Grid_XYZW[ 1124][0] = -0.300209680089587;
    Leb_Grid_XYZW[ 1124][1] = -0.088694243067227;
    Leb_Grid_XYZW[ 1124][2] =  0.949740743164807;
    Leb_Grid_XYZW[ 1124][3] =  0.000460493136846;

    Leb_Grid_XYZW[ 1125][0] = -0.300209680089587;
    Leb_Grid_XYZW[ 1125][1] = -0.088694243067227;
    Leb_Grid_XYZW[ 1125][2] = -0.949740743164807;
    Leb_Grid_XYZW[ 1125][3] =  0.000460493136846;

    Leb_Grid_XYZW[ 1126][0] =  0.300209680089587;
    Leb_Grid_XYZW[ 1126][1] =  0.949740743164807;
    Leb_Grid_XYZW[ 1126][2] =  0.088694243067227;
    Leb_Grid_XYZW[ 1126][3] =  0.000460493136846;

    Leb_Grid_XYZW[ 1127][0] =  0.300209680089587;
    Leb_Grid_XYZW[ 1127][1] =  0.949740743164807;
    Leb_Grid_XYZW[ 1127][2] = -0.088694243067227;
    Leb_Grid_XYZW[ 1127][3] =  0.000460493136846;

    Leb_Grid_XYZW[ 1128][0] =  0.300209680089587;
    Leb_Grid_XYZW[ 1128][1] = -0.949740743164807;
    Leb_Grid_XYZW[ 1128][2] =  0.088694243067227;
    Leb_Grid_XYZW[ 1128][3] =  0.000460493136846;

    Leb_Grid_XYZW[ 1129][0] =  0.300209680089587;
    Leb_Grid_XYZW[ 1129][1] = -0.949740743164807;
    Leb_Grid_XYZW[ 1129][2] = -0.088694243067227;
    Leb_Grid_XYZW[ 1129][3] =  0.000460493136846;

    Leb_Grid_XYZW[ 1130][0] = -0.300209680089587;
    Leb_Grid_XYZW[ 1130][1] =  0.949740743164807;
    Leb_Grid_XYZW[ 1130][2] =  0.088694243067227;
    Leb_Grid_XYZW[ 1130][3] =  0.000460493136846;

    Leb_Grid_XYZW[ 1131][0] = -0.300209680089587;
    Leb_Grid_XYZW[ 1131][1] =  0.949740743164807;
    Leb_Grid_XYZW[ 1131][2] = -0.088694243067227;
    Leb_Grid_XYZW[ 1131][3] =  0.000460493136846;

    Leb_Grid_XYZW[ 1132][0] = -0.300209680089587;
    Leb_Grid_XYZW[ 1132][1] = -0.949740743164807;
    Leb_Grid_XYZW[ 1132][2] =  0.088694243067227;
    Leb_Grid_XYZW[ 1132][3] =  0.000460493136846;

    Leb_Grid_XYZW[ 1133][0] = -0.300209680089587;
    Leb_Grid_XYZW[ 1133][1] = -0.949740743164807;
    Leb_Grid_XYZW[ 1133][2] = -0.088694243067227;
    Leb_Grid_XYZW[ 1133][3] =  0.000460493136846;

    Leb_Grid_XYZW[ 1134][0] =  0.088694243067227;
    Leb_Grid_XYZW[ 1134][1] =  0.300209680089587;
    Leb_Grid_XYZW[ 1134][2] =  0.949740743164807;
    Leb_Grid_XYZW[ 1134][3] =  0.000460493136846;

    Leb_Grid_XYZW[ 1135][0] =  0.088694243067227;
    Leb_Grid_XYZW[ 1135][1] =  0.300209680089587;
    Leb_Grid_XYZW[ 1135][2] = -0.949740743164807;
    Leb_Grid_XYZW[ 1135][3] =  0.000460493136846;

    Leb_Grid_XYZW[ 1136][0] =  0.088694243067227;
    Leb_Grid_XYZW[ 1136][1] = -0.300209680089587;
    Leb_Grid_XYZW[ 1136][2] =  0.949740743164807;
    Leb_Grid_XYZW[ 1136][3] =  0.000460493136846;

    Leb_Grid_XYZW[ 1137][0] =  0.088694243067227;
    Leb_Grid_XYZW[ 1137][1] = -0.300209680089587;
    Leb_Grid_XYZW[ 1137][2] = -0.949740743164807;
    Leb_Grid_XYZW[ 1137][3] =  0.000460493136846;

    Leb_Grid_XYZW[ 1138][0] = -0.088694243067227;
    Leb_Grid_XYZW[ 1138][1] =  0.300209680089587;
    Leb_Grid_XYZW[ 1138][2] =  0.949740743164807;
    Leb_Grid_XYZW[ 1138][3] =  0.000460493136846;

    Leb_Grid_XYZW[ 1139][0] = -0.088694243067227;
    Leb_Grid_XYZW[ 1139][1] =  0.300209680089587;
    Leb_Grid_XYZW[ 1139][2] = -0.949740743164807;
    Leb_Grid_XYZW[ 1139][3] =  0.000460493136846;

    Leb_Grid_XYZW[ 1140][0] = -0.088694243067227;
    Leb_Grid_XYZW[ 1140][1] = -0.300209680089587;
    Leb_Grid_XYZW[ 1140][2] =  0.949740743164807;
    Leb_Grid_XYZW[ 1140][3] =  0.000460493136846;

    Leb_Grid_XYZW[ 1141][0] = -0.088694243067227;
    Leb_Grid_XYZW[ 1141][1] = -0.300209680089587;
    Leb_Grid_XYZW[ 1141][2] = -0.949740743164807;
    Leb_Grid_XYZW[ 1141][3] =  0.000460493136846;

    Leb_Grid_XYZW[ 1142][0] =  0.088694243067227;
    Leb_Grid_XYZW[ 1142][1] =  0.949740743164807;
    Leb_Grid_XYZW[ 1142][2] =  0.300209680089587;
    Leb_Grid_XYZW[ 1142][3] =  0.000460493136846;

    Leb_Grid_XYZW[ 1143][0] =  0.088694243067227;
    Leb_Grid_XYZW[ 1143][1] =  0.949740743164807;
    Leb_Grid_XYZW[ 1143][2] = -0.300209680089587;
    Leb_Grid_XYZW[ 1143][3] =  0.000460493136846;

    Leb_Grid_XYZW[ 1144][0] =  0.088694243067227;
    Leb_Grid_XYZW[ 1144][1] = -0.949740743164807;
    Leb_Grid_XYZW[ 1144][2] =  0.300209680089587;
    Leb_Grid_XYZW[ 1144][3] =  0.000460493136846;

    Leb_Grid_XYZW[ 1145][0] =  0.088694243067227;
    Leb_Grid_XYZW[ 1145][1] = -0.949740743164807;
    Leb_Grid_XYZW[ 1145][2] = -0.300209680089587;
    Leb_Grid_XYZW[ 1145][3] =  0.000460493136846;

    Leb_Grid_XYZW[ 1146][0] = -0.088694243067227;
    Leb_Grid_XYZW[ 1146][1] =  0.949740743164807;
    Leb_Grid_XYZW[ 1146][2] =  0.300209680089587;
    Leb_Grid_XYZW[ 1146][3] =  0.000460493136846;

    Leb_Grid_XYZW[ 1147][0] = -0.088694243067227;
    Leb_Grid_XYZW[ 1147][1] =  0.949740743164807;
    Leb_Grid_XYZW[ 1147][2] = -0.300209680089587;
    Leb_Grid_XYZW[ 1147][3] =  0.000460493136846;

    Leb_Grid_XYZW[ 1148][0] = -0.088694243067227;
    Leb_Grid_XYZW[ 1148][1] = -0.949740743164807;
    Leb_Grid_XYZW[ 1148][2] =  0.300209680089587;
    Leb_Grid_XYZW[ 1148][3] =  0.000460493136846;

    Leb_Grid_XYZW[ 1149][0] = -0.088694243067227;
    Leb_Grid_XYZW[ 1149][1] = -0.949740743164807;
    Leb_Grid_XYZW[ 1149][2] = -0.300209680089587;
    Leb_Grid_XYZW[ 1149][3] =  0.000460493136846;

    Leb_Grid_XYZW[ 1150][0] =  0.949740743164807;
    Leb_Grid_XYZW[ 1150][1] =  0.300209680089587;
    Leb_Grid_XYZW[ 1150][2] =  0.088694243067227;
    Leb_Grid_XYZW[ 1150][3] =  0.000460493136846;

    Leb_Grid_XYZW[ 1151][0] =  0.949740743164807;
    Leb_Grid_XYZW[ 1151][1] =  0.300209680089587;
    Leb_Grid_XYZW[ 1151][2] = -0.088694243067227;
    Leb_Grid_XYZW[ 1151][3] =  0.000460493136846;

    Leb_Grid_XYZW[ 1152][0] =  0.949740743164807;
    Leb_Grid_XYZW[ 1152][1] = -0.300209680089587;
    Leb_Grid_XYZW[ 1152][2] =  0.088694243067227;
    Leb_Grid_XYZW[ 1152][3] =  0.000460493136846;

    Leb_Grid_XYZW[ 1153][0] =  0.949740743164807;
    Leb_Grid_XYZW[ 1153][1] = -0.300209680089587;
    Leb_Grid_XYZW[ 1153][2] = -0.088694243067227;
    Leb_Grid_XYZW[ 1153][3] =  0.000460493136846;

    Leb_Grid_XYZW[ 1154][0] = -0.949740743164807;
    Leb_Grid_XYZW[ 1154][1] =  0.300209680089587;
    Leb_Grid_XYZW[ 1154][2] =  0.088694243067227;
    Leb_Grid_XYZW[ 1154][3] =  0.000460493136846;

    Leb_Grid_XYZW[ 1155][0] = -0.949740743164807;
    Leb_Grid_XYZW[ 1155][1] =  0.300209680089587;
    Leb_Grid_XYZW[ 1155][2] = -0.088694243067227;
    Leb_Grid_XYZW[ 1155][3] =  0.000460493136846;

    Leb_Grid_XYZW[ 1156][0] = -0.949740743164807;
    Leb_Grid_XYZW[ 1156][1] = -0.300209680089587;
    Leb_Grid_XYZW[ 1156][2] =  0.088694243067227;
    Leb_Grid_XYZW[ 1156][3] =  0.000460493136846;

    Leb_Grid_XYZW[ 1157][0] = -0.949740743164807;
    Leb_Grid_XYZW[ 1157][1] = -0.300209680089587;
    Leb_Grid_XYZW[ 1157][2] = -0.088694243067227;
    Leb_Grid_XYZW[ 1157][3] =  0.000460493136846;

    Leb_Grid_XYZW[ 1158][0] =  0.949740743164807;
    Leb_Grid_XYZW[ 1158][1] =  0.088694243067227;
    Leb_Grid_XYZW[ 1158][2] =  0.300209680089587;
    Leb_Grid_XYZW[ 1158][3] =  0.000460493136846;

    Leb_Grid_XYZW[ 1159][0] =  0.949740743164807;
    Leb_Grid_XYZW[ 1159][1] =  0.088694243067227;
    Leb_Grid_XYZW[ 1159][2] = -0.300209680089587;
    Leb_Grid_XYZW[ 1159][3] =  0.000460493136846;

    Leb_Grid_XYZW[ 1160][0] =  0.949740743164807;
    Leb_Grid_XYZW[ 1160][1] = -0.088694243067227;
    Leb_Grid_XYZW[ 1160][2] =  0.300209680089587;
    Leb_Grid_XYZW[ 1160][3] =  0.000460493136846;

    Leb_Grid_XYZW[ 1161][0] =  0.949740743164807;
    Leb_Grid_XYZW[ 1161][1] = -0.088694243067227;
    Leb_Grid_XYZW[ 1161][2] = -0.300209680089587;
    Leb_Grid_XYZW[ 1161][3] =  0.000460493136846;

    Leb_Grid_XYZW[ 1162][0] = -0.949740743164807;
    Leb_Grid_XYZW[ 1162][1] =  0.088694243067228;
    Leb_Grid_XYZW[ 1162][2] =  0.300209680089587;
    Leb_Grid_XYZW[ 1162][3] =  0.000460493136846;

    Leb_Grid_XYZW[ 1163][0] = -0.949740743164807;
    Leb_Grid_XYZW[ 1163][1] =  0.088694243067228;
    Leb_Grid_XYZW[ 1163][2] = -0.300209680089587;
    Leb_Grid_XYZW[ 1163][3] =  0.000460493136846;

    Leb_Grid_XYZW[ 1164][0] = -0.949740743164807;
    Leb_Grid_XYZW[ 1164][1] = -0.088694243067228;
    Leb_Grid_XYZW[ 1164][2] =  0.300209680089587;
    Leb_Grid_XYZW[ 1164][3] =  0.000460493136846;

    Leb_Grid_XYZW[ 1165][0] = -0.949740743164807;
    Leb_Grid_XYZW[ 1165][1] = -0.088694243067228;
    Leb_Grid_XYZW[ 1165][2] = -0.300209680089587;
    Leb_Grid_XYZW[ 1165][3] =  0.000460493136846;

    Leb_Grid_XYZW[ 1166][0] =  0.355855445745743;
    Leb_Grid_XYZW[ 1166][1] =  0.136881170651066;
    Leb_Grid_XYZW[ 1166][2] =  0.924462247392662;
    Leb_Grid_XYZW[ 1166][3] =  0.000487181487826;

    Leb_Grid_XYZW[ 1167][0] =  0.355855445745743;
    Leb_Grid_XYZW[ 1167][1] =  0.136881170651066;
    Leb_Grid_XYZW[ 1167][2] = -0.924462247392662;
    Leb_Grid_XYZW[ 1167][3] =  0.000487181487826;

    Leb_Grid_XYZW[ 1168][0] =  0.355855445745743;
    Leb_Grid_XYZW[ 1168][1] = -0.136881170651066;
    Leb_Grid_XYZW[ 1168][2] =  0.924462247392662;
    Leb_Grid_XYZW[ 1168][3] =  0.000487181487826;

    Leb_Grid_XYZW[ 1169][0] =  0.355855445745743;
    Leb_Grid_XYZW[ 1169][1] = -0.136881170651066;
    Leb_Grid_XYZW[ 1169][2] = -0.924462247392662;
    Leb_Grid_XYZW[ 1169][3] =  0.000487181487826;

    Leb_Grid_XYZW[ 1170][0] = -0.355855445745743;
    Leb_Grid_XYZW[ 1170][1] =  0.136881170651066;
    Leb_Grid_XYZW[ 1170][2] =  0.924462247392662;
    Leb_Grid_XYZW[ 1170][3] =  0.000487181487826;

    Leb_Grid_XYZW[ 1171][0] = -0.355855445745743;
    Leb_Grid_XYZW[ 1171][1] =  0.136881170651065;
    Leb_Grid_XYZW[ 1171][2] = -0.924462247392662;
    Leb_Grid_XYZW[ 1171][3] =  0.000487181487826;

    Leb_Grid_XYZW[ 1172][0] = -0.355855445745743;
    Leb_Grid_XYZW[ 1172][1] = -0.136881170651066;
    Leb_Grid_XYZW[ 1172][2] =  0.924462247392662;
    Leb_Grid_XYZW[ 1172][3] =  0.000487181487826;

    Leb_Grid_XYZW[ 1173][0] = -0.355855445745743;
    Leb_Grid_XYZW[ 1173][1] = -0.136881170651065;
    Leb_Grid_XYZW[ 1173][2] = -0.924462247392662;
    Leb_Grid_XYZW[ 1173][3] =  0.000487181487826;

    Leb_Grid_XYZW[ 1174][0] =  0.355855445745743;
    Leb_Grid_XYZW[ 1174][1] =  0.924462247392662;
    Leb_Grid_XYZW[ 1174][2] =  0.136881170651066;
    Leb_Grid_XYZW[ 1174][3] =  0.000487181487826;

    Leb_Grid_XYZW[ 1175][0] =  0.355855445745743;
    Leb_Grid_XYZW[ 1175][1] =  0.924462247392662;
    Leb_Grid_XYZW[ 1175][2] = -0.136881170651065;
    Leb_Grid_XYZW[ 1175][3] =  0.000487181487826;

    Leb_Grid_XYZW[ 1176][0] =  0.355855445745743;
    Leb_Grid_XYZW[ 1176][1] = -0.924462247392662;
    Leb_Grid_XYZW[ 1176][2] =  0.136881170651066;
    Leb_Grid_XYZW[ 1176][3] =  0.000487181487826;

    Leb_Grid_XYZW[ 1177][0] =  0.355855445745743;
    Leb_Grid_XYZW[ 1177][1] = -0.924462247392662;
    Leb_Grid_XYZW[ 1177][2] = -0.136881170651065;
    Leb_Grid_XYZW[ 1177][3] =  0.000487181487826;

    Leb_Grid_XYZW[ 1178][0] = -0.355855445745743;
    Leb_Grid_XYZW[ 1178][1] =  0.924462247392662;
    Leb_Grid_XYZW[ 1178][2] =  0.136881170651066;
    Leb_Grid_XYZW[ 1178][3] =  0.000487181487826;

    Leb_Grid_XYZW[ 1179][0] = -0.355855445745743;
    Leb_Grid_XYZW[ 1179][1] =  0.924462247392662;
    Leb_Grid_XYZW[ 1179][2] = -0.136881170651065;
    Leb_Grid_XYZW[ 1179][3] =  0.000487181487826;

    Leb_Grid_XYZW[ 1180][0] = -0.355855445745743;
    Leb_Grid_XYZW[ 1180][1] = -0.924462247392662;
    Leb_Grid_XYZW[ 1180][2] =  0.136881170651066;
    Leb_Grid_XYZW[ 1180][3] =  0.000487181487826;

    Leb_Grid_XYZW[ 1181][0] = -0.355855445745743;
    Leb_Grid_XYZW[ 1181][1] = -0.924462247392662;
    Leb_Grid_XYZW[ 1181][2] = -0.136881170651065;
    Leb_Grid_XYZW[ 1181][3] =  0.000487181487826;

    Leb_Grid_XYZW[ 1182][0] =  0.136881170651065;
    Leb_Grid_XYZW[ 1182][1] =  0.355855445745743;
    Leb_Grid_XYZW[ 1182][2] =  0.924462247392662;
    Leb_Grid_XYZW[ 1182][3] =  0.000487181487826;

    Leb_Grid_XYZW[ 1183][0] =  0.136881170651065;
    Leb_Grid_XYZW[ 1183][1] =  0.355855445745743;
    Leb_Grid_XYZW[ 1183][2] = -0.924462247392662;
    Leb_Grid_XYZW[ 1183][3] =  0.000487181487826;

    Leb_Grid_XYZW[ 1184][0] =  0.136881170651065;
    Leb_Grid_XYZW[ 1184][1] = -0.355855445745743;
    Leb_Grid_XYZW[ 1184][2] =  0.924462247392662;
    Leb_Grid_XYZW[ 1184][3] =  0.000487181487826;

    Leb_Grid_XYZW[ 1185][0] =  0.136881170651065;
    Leb_Grid_XYZW[ 1185][1] = -0.355855445745743;
    Leb_Grid_XYZW[ 1185][2] = -0.924462247392662;
    Leb_Grid_XYZW[ 1185][3] =  0.000487181487826;

    Leb_Grid_XYZW[ 1186][0] = -0.136881170651066;
    Leb_Grid_XYZW[ 1186][1] =  0.355855445745743;
    Leb_Grid_XYZW[ 1186][2] =  0.924462247392662;
    Leb_Grid_XYZW[ 1186][3] =  0.000487181487826;

    Leb_Grid_XYZW[ 1187][0] = -0.136881170651065;
    Leb_Grid_XYZW[ 1187][1] =  0.355855445745743;
    Leb_Grid_XYZW[ 1187][2] = -0.924462247392662;
    Leb_Grid_XYZW[ 1187][3] =  0.000487181487826;

    Leb_Grid_XYZW[ 1188][0] = -0.136881170651066;
    Leb_Grid_XYZW[ 1188][1] = -0.355855445745743;
    Leb_Grid_XYZW[ 1188][2] =  0.924462247392662;
    Leb_Grid_XYZW[ 1188][3] =  0.000487181487826;

    Leb_Grid_XYZW[ 1189][0] = -0.136881170651065;
    Leb_Grid_XYZW[ 1189][1] = -0.355855445745743;
    Leb_Grid_XYZW[ 1189][2] = -0.924462247392662;
    Leb_Grid_XYZW[ 1189][3] =  0.000487181487826;

    Leb_Grid_XYZW[ 1190][0] =  0.136881170651065;
    Leb_Grid_XYZW[ 1190][1] =  0.924462247392663;
    Leb_Grid_XYZW[ 1190][2] =  0.355855445745743;
    Leb_Grid_XYZW[ 1190][3] =  0.000487181487826;

    Leb_Grid_XYZW[ 1191][0] =  0.136881170651065;
    Leb_Grid_XYZW[ 1191][1] =  0.924462247392663;
    Leb_Grid_XYZW[ 1191][2] = -0.355855445745743;
    Leb_Grid_XYZW[ 1191][3] =  0.000487181487826;

    Leb_Grid_XYZW[ 1192][0] =  0.136881170651065;
    Leb_Grid_XYZW[ 1192][1] = -0.924462247392663;
    Leb_Grid_XYZW[ 1192][2] =  0.355855445745743;
    Leb_Grid_XYZW[ 1192][3] =  0.000487181487826;

    Leb_Grid_XYZW[ 1193][0] =  0.136881170651065;
    Leb_Grid_XYZW[ 1193][1] = -0.924462247392663;
    Leb_Grid_XYZW[ 1193][2] = -0.355855445745743;
    Leb_Grid_XYZW[ 1193][3] =  0.000487181487826;

    Leb_Grid_XYZW[ 1194][0] = -0.136881170651066;
    Leb_Grid_XYZW[ 1194][1] =  0.924462247392663;
    Leb_Grid_XYZW[ 1194][2] =  0.355855445745743;
    Leb_Grid_XYZW[ 1194][3] =  0.000487181487826;

    Leb_Grid_XYZW[ 1195][0] = -0.136881170651066;
    Leb_Grid_XYZW[ 1195][1] =  0.924462247392663;
    Leb_Grid_XYZW[ 1195][2] = -0.355855445745743;
    Leb_Grid_XYZW[ 1195][3] =  0.000487181487826;

    Leb_Grid_XYZW[ 1196][0] = -0.136881170651066;
    Leb_Grid_XYZW[ 1196][1] = -0.924462247392663;
    Leb_Grid_XYZW[ 1196][2] =  0.355855445745743;
    Leb_Grid_XYZW[ 1196][3] =  0.000487181487826;

    Leb_Grid_XYZW[ 1197][0] = -0.136881170651066;
    Leb_Grid_XYZW[ 1197][1] = -0.924462247392663;
    Leb_Grid_XYZW[ 1197][2] = -0.355855445745743;
    Leb_Grid_XYZW[ 1197][3] =  0.000487181487826;

    Leb_Grid_XYZW[ 1198][0] =  0.924462247392662;
    Leb_Grid_XYZW[ 1198][1] =  0.355855445745743;
    Leb_Grid_XYZW[ 1198][2] =  0.136881170651066;
    Leb_Grid_XYZW[ 1198][3] =  0.000487181487826;

    Leb_Grid_XYZW[ 1199][0] =  0.924462247392662;
    Leb_Grid_XYZW[ 1199][1] =  0.355855445745743;
    Leb_Grid_XYZW[ 1199][2] = -0.136881170651065;
    Leb_Grid_XYZW[ 1199][3] =  0.000487181487826;

    Leb_Grid_XYZW[ 1200][0] =  0.924462247392662;
    Leb_Grid_XYZW[ 1200][1] = -0.355855445745743;
    Leb_Grid_XYZW[ 1200][2] =  0.136881170651066;
    Leb_Grid_XYZW[ 1200][3] =  0.000487181487826;

    Leb_Grid_XYZW[ 1201][0] =  0.924462247392662;
    Leb_Grid_XYZW[ 1201][1] = -0.355855445745743;
    Leb_Grid_XYZW[ 1201][2] = -0.136881170651065;
    Leb_Grid_XYZW[ 1201][3] =  0.000487181487826;

    Leb_Grid_XYZW[ 1202][0] = -0.924462247392662;
    Leb_Grid_XYZW[ 1202][1] =  0.355855445745743;
    Leb_Grid_XYZW[ 1202][2] =  0.136881170651066;
    Leb_Grid_XYZW[ 1202][3] =  0.000487181487826;

    Leb_Grid_XYZW[ 1203][0] = -0.924462247392662;
    Leb_Grid_XYZW[ 1203][1] =  0.355855445745743;
    Leb_Grid_XYZW[ 1203][2] = -0.136881170651065;
    Leb_Grid_XYZW[ 1203][3] =  0.000487181487826;

    Leb_Grid_XYZW[ 1204][0] = -0.924462247392662;
    Leb_Grid_XYZW[ 1204][1] = -0.355855445745743;
    Leb_Grid_XYZW[ 1204][2] =  0.136881170651066;
    Leb_Grid_XYZW[ 1204][3] =  0.000487181487826;

    Leb_Grid_XYZW[ 1205][0] = -0.924462247392662;
    Leb_Grid_XYZW[ 1205][1] = -0.355855445745743;
    Leb_Grid_XYZW[ 1205][2] = -0.136881170651065;
    Leb_Grid_XYZW[ 1205][3] =  0.000487181487826;

    Leb_Grid_XYZW[ 1206][0] =  0.924462247392662;
    Leb_Grid_XYZW[ 1206][1] =  0.136881170651066;
    Leb_Grid_XYZW[ 1206][2] =  0.355855445745743;
    Leb_Grid_XYZW[ 1206][3] =  0.000487181487826;

    Leb_Grid_XYZW[ 1207][0] =  0.924462247392662;
    Leb_Grid_XYZW[ 1207][1] =  0.136881170651066;
    Leb_Grid_XYZW[ 1207][2] = -0.355855445745743;
    Leb_Grid_XYZW[ 1207][3] =  0.000487181487826;

    Leb_Grid_XYZW[ 1208][0] =  0.924462247392662;
    Leb_Grid_XYZW[ 1208][1] = -0.136881170651066;
    Leb_Grid_XYZW[ 1208][2] =  0.355855445745743;
    Leb_Grid_XYZW[ 1208][3] =  0.000487181487826;

    Leb_Grid_XYZW[ 1209][0] =  0.924462247392662;
    Leb_Grid_XYZW[ 1209][1] = -0.136881170651066;
    Leb_Grid_XYZW[ 1209][2] = -0.355855445745743;
    Leb_Grid_XYZW[ 1209][3] =  0.000487181487826;

    Leb_Grid_XYZW[ 1210][0] = -0.924462247392662;
    Leb_Grid_XYZW[ 1210][1] =  0.136881170651066;
    Leb_Grid_XYZW[ 1210][2] =  0.355855445745743;
    Leb_Grid_XYZW[ 1210][3] =  0.000487181487826;

    Leb_Grid_XYZW[ 1211][0] = -0.924462247392662;
    Leb_Grid_XYZW[ 1211][1] =  0.136881170651066;
    Leb_Grid_XYZW[ 1211][2] = -0.355855445745743;
    Leb_Grid_XYZW[ 1211][3] =  0.000487181487826;

    Leb_Grid_XYZW[ 1212][0] = -0.924462247392662;
    Leb_Grid_XYZW[ 1212][1] = -0.136881170651066;
    Leb_Grid_XYZW[ 1212][2] =  0.355855445745743;
    Leb_Grid_XYZW[ 1212][3] =  0.000487181487826;

    Leb_Grid_XYZW[ 1213][0] = -0.924462247392662;
    Leb_Grid_XYZW[ 1213][1] = -0.136881170651066;
    Leb_Grid_XYZW[ 1213][2] = -0.355855445745743;
    Leb_Grid_XYZW[ 1213][3] =  0.000487181487826;

    Leb_Grid_XYZW[ 1214][0] =  0.409778253704889;
    Leb_Grid_XYZW[ 1214][1] =  0.186073998501503;
    Leb_Grid_XYZW[ 1214][2] =  0.893005179084777;
    Leb_Grid_XYZW[ 1214][3] =  0.000507224291007;

    Leb_Grid_XYZW[ 1215][0] =  0.409778253704889;
    Leb_Grid_XYZW[ 1215][1] =  0.186073998501503;
    Leb_Grid_XYZW[ 1215][2] = -0.893005179084777;
    Leb_Grid_XYZW[ 1215][3] =  0.000507224291007;

    Leb_Grid_XYZW[ 1216][0] =  0.409778253704889;
    Leb_Grid_XYZW[ 1216][1] = -0.186073998501503;
    Leb_Grid_XYZW[ 1216][2] =  0.893005179084777;
    Leb_Grid_XYZW[ 1216][3] =  0.000507224291007;

    Leb_Grid_XYZW[ 1217][0] =  0.409778253704889;
    Leb_Grid_XYZW[ 1217][1] = -0.186073998501503;
    Leb_Grid_XYZW[ 1217][2] = -0.893005179084777;
    Leb_Grid_XYZW[ 1217][3] =  0.000507224291007;

    Leb_Grid_XYZW[ 1218][0] = -0.409778253704889;
    Leb_Grid_XYZW[ 1218][1] =  0.186073998501503;
    Leb_Grid_XYZW[ 1218][2] =  0.893005179084777;
    Leb_Grid_XYZW[ 1218][3] =  0.000507224291007;

    Leb_Grid_XYZW[ 1219][0] = -0.409778253704889;
    Leb_Grid_XYZW[ 1219][1] =  0.186073998501503;
    Leb_Grid_XYZW[ 1219][2] = -0.893005179084777;
    Leb_Grid_XYZW[ 1219][3] =  0.000507224291007;

    Leb_Grid_XYZW[ 1220][0] = -0.409778253704889;
    Leb_Grid_XYZW[ 1220][1] = -0.186073998501503;
    Leb_Grid_XYZW[ 1220][2] =  0.893005179084777;
    Leb_Grid_XYZW[ 1220][3] =  0.000507224291007;

    Leb_Grid_XYZW[ 1221][0] = -0.409778253704889;
    Leb_Grid_XYZW[ 1221][1] = -0.186073998501503;
    Leb_Grid_XYZW[ 1221][2] = -0.893005179084777;
    Leb_Grid_XYZW[ 1221][3] =  0.000507224291007;

    Leb_Grid_XYZW[ 1222][0] =  0.409778253704889;
    Leb_Grid_XYZW[ 1222][1] =  0.893005179084777;
    Leb_Grid_XYZW[ 1222][2] =  0.186073998501503;
    Leb_Grid_XYZW[ 1222][3] =  0.000507224291007;

    Leb_Grid_XYZW[ 1223][0] =  0.409778253704889;
    Leb_Grid_XYZW[ 1223][1] =  0.893005179084777;
    Leb_Grid_XYZW[ 1223][2] = -0.186073998501503;
    Leb_Grid_XYZW[ 1223][3] =  0.000507224291007;

    Leb_Grid_XYZW[ 1224][0] =  0.409778253704889;
    Leb_Grid_XYZW[ 1224][1] = -0.893005179084777;
    Leb_Grid_XYZW[ 1224][2] =  0.186073998501503;
    Leb_Grid_XYZW[ 1224][3] =  0.000507224291007;

    Leb_Grid_XYZW[ 1225][0] =  0.409778253704889;
    Leb_Grid_XYZW[ 1225][1] = -0.893005179084777;
    Leb_Grid_XYZW[ 1225][2] = -0.186073998501503;
    Leb_Grid_XYZW[ 1225][3] =  0.000507224291007;

    Leb_Grid_XYZW[ 1226][0] = -0.409778253704889;
    Leb_Grid_XYZW[ 1226][1] =  0.893005179084777;
    Leb_Grid_XYZW[ 1226][2] =  0.186073998501503;
    Leb_Grid_XYZW[ 1226][3] =  0.000507224291007;

    Leb_Grid_XYZW[ 1227][0] = -0.409778253704889;
    Leb_Grid_XYZW[ 1227][1] =  0.893005179084777;
    Leb_Grid_XYZW[ 1227][2] = -0.186073998501503;
    Leb_Grid_XYZW[ 1227][3] =  0.000507224291007;

    Leb_Grid_XYZW[ 1228][0] = -0.409778253704889;
    Leb_Grid_XYZW[ 1228][1] = -0.893005179084777;
    Leb_Grid_XYZW[ 1228][2] =  0.186073998501503;
    Leb_Grid_XYZW[ 1228][3] =  0.000507224291007;

    Leb_Grid_XYZW[ 1229][0] = -0.409778253704889;
    Leb_Grid_XYZW[ 1229][1] = -0.893005179084777;
    Leb_Grid_XYZW[ 1229][2] = -0.186073998501503;
    Leb_Grid_XYZW[ 1229][3] =  0.000507224291007;

    Leb_Grid_XYZW[ 1230][0] =  0.186073998501503;
    Leb_Grid_XYZW[ 1230][1] =  0.409778253704889;
    Leb_Grid_XYZW[ 1230][2] =  0.893005179084777;
    Leb_Grid_XYZW[ 1230][3] =  0.000507224291007;

    Leb_Grid_XYZW[ 1231][0] =  0.186073998501503;
    Leb_Grid_XYZW[ 1231][1] =  0.409778253704889;
    Leb_Grid_XYZW[ 1231][2] = -0.893005179084777;
    Leb_Grid_XYZW[ 1231][3] =  0.000507224291007;

    Leb_Grid_XYZW[ 1232][0] =  0.186073998501503;
    Leb_Grid_XYZW[ 1232][1] = -0.409778253704889;
    Leb_Grid_XYZW[ 1232][2] =  0.893005179084777;
    Leb_Grid_XYZW[ 1232][3] =  0.000507224291007;

    Leb_Grid_XYZW[ 1233][0] =  0.186073998501503;
    Leb_Grid_XYZW[ 1233][1] = -0.409778253704889;
    Leb_Grid_XYZW[ 1233][2] = -0.893005179084777;
    Leb_Grid_XYZW[ 1233][3] =  0.000507224291007;

    Leb_Grid_XYZW[ 1234][0] = -0.186073998501503;
    Leb_Grid_XYZW[ 1234][1] =  0.409778253704889;
    Leb_Grid_XYZW[ 1234][2] =  0.893005179084777;
    Leb_Grid_XYZW[ 1234][3] =  0.000507224291007;

    Leb_Grid_XYZW[ 1235][0] = -0.186073998501503;
    Leb_Grid_XYZW[ 1235][1] =  0.409778253704889;
    Leb_Grid_XYZW[ 1235][2] = -0.893005179084777;
    Leb_Grid_XYZW[ 1235][3] =  0.000507224291007;

    Leb_Grid_XYZW[ 1236][0] = -0.186073998501503;
    Leb_Grid_XYZW[ 1236][1] = -0.409778253704889;
    Leb_Grid_XYZW[ 1236][2] =  0.893005179084777;
    Leb_Grid_XYZW[ 1236][3] =  0.000507224291007;

    Leb_Grid_XYZW[ 1237][0] = -0.186073998501503;
    Leb_Grid_XYZW[ 1237][1] = -0.409778253704889;
    Leb_Grid_XYZW[ 1237][2] = -0.893005179084777;
    Leb_Grid_XYZW[ 1237][3] =  0.000507224291007;

    Leb_Grid_XYZW[ 1238][0] =  0.186073998501503;
    Leb_Grid_XYZW[ 1238][1] =  0.893005179084777;
    Leb_Grid_XYZW[ 1238][2] =  0.409778253704889;
    Leb_Grid_XYZW[ 1238][3] =  0.000507224291007;

    Leb_Grid_XYZW[ 1239][0] =  0.186073998501503;
    Leb_Grid_XYZW[ 1239][1] =  0.893005179084777;
    Leb_Grid_XYZW[ 1239][2] = -0.409778253704889;
    Leb_Grid_XYZW[ 1239][3] =  0.000507224291007;

    Leb_Grid_XYZW[ 1240][0] =  0.186073998501503;
    Leb_Grid_XYZW[ 1240][1] = -0.893005179084777;
    Leb_Grid_XYZW[ 1240][2] =  0.409778253704889;
    Leb_Grid_XYZW[ 1240][3] =  0.000507224291007;

    Leb_Grid_XYZW[ 1241][0] =  0.186073998501503;
    Leb_Grid_XYZW[ 1241][1] = -0.893005179084777;
    Leb_Grid_XYZW[ 1241][2] = -0.409778253704889;
    Leb_Grid_XYZW[ 1241][3] =  0.000507224291007;

    Leb_Grid_XYZW[ 1242][0] = -0.186073998501503;
    Leb_Grid_XYZW[ 1242][1] =  0.893005179084777;
    Leb_Grid_XYZW[ 1242][2] =  0.409778253704889;
    Leb_Grid_XYZW[ 1242][3] =  0.000507224291007;

    Leb_Grid_XYZW[ 1243][0] = -0.186073998501503;
    Leb_Grid_XYZW[ 1243][1] =  0.893005179084777;
    Leb_Grid_XYZW[ 1243][2] = -0.409778253704889;
    Leb_Grid_XYZW[ 1243][3] =  0.000507224291007;

    Leb_Grid_XYZW[ 1244][0] = -0.186073998501503;
    Leb_Grid_XYZW[ 1244][1] = -0.893005179084777;
    Leb_Grid_XYZW[ 1244][2] =  0.409778253704889;
    Leb_Grid_XYZW[ 1244][3] =  0.000507224291007;

    Leb_Grid_XYZW[ 1245][0] = -0.186073998501503;
    Leb_Grid_XYZW[ 1245][1] = -0.893005179084777;
    Leb_Grid_XYZW[ 1245][2] = -0.409778253704889;
    Leb_Grid_XYZW[ 1245][3] =  0.000507224291007;

    Leb_Grid_XYZW[ 1246][0] =  0.893005179084777;
    Leb_Grid_XYZW[ 1246][1] =  0.409778253704889;
    Leb_Grid_XYZW[ 1246][2] =  0.186073998501503;
    Leb_Grid_XYZW[ 1246][3] =  0.000507224291007;

    Leb_Grid_XYZW[ 1247][0] =  0.893005179084777;
    Leb_Grid_XYZW[ 1247][1] =  0.409778253704889;
    Leb_Grid_XYZW[ 1247][2] = -0.186073998501503;
    Leb_Grid_XYZW[ 1247][3] =  0.000507224291007;

    Leb_Grid_XYZW[ 1248][0] =  0.893005179084777;
    Leb_Grid_XYZW[ 1248][1] = -0.409778253704889;
    Leb_Grid_XYZW[ 1248][2] =  0.186073998501503;
    Leb_Grid_XYZW[ 1248][3] =  0.000507224291007;

    Leb_Grid_XYZW[ 1249][0] =  0.893005179084777;
    Leb_Grid_XYZW[ 1249][1] = -0.409778253704889;
    Leb_Grid_XYZW[ 1249][2] = -0.186073998501503;
    Leb_Grid_XYZW[ 1249][3] =  0.000507224291007;

    Leb_Grid_XYZW[ 1250][0] = -0.893005179084777;
    Leb_Grid_XYZW[ 1250][1] =  0.409778253704889;
    Leb_Grid_XYZW[ 1250][2] =  0.186073998501503;
    Leb_Grid_XYZW[ 1250][3] =  0.000507224291007;

    Leb_Grid_XYZW[ 1251][0] = -0.893005179084777;
    Leb_Grid_XYZW[ 1251][1] =  0.409778253704889;
    Leb_Grid_XYZW[ 1251][2] = -0.186073998501503;
    Leb_Grid_XYZW[ 1251][3] =  0.000507224291007;

    Leb_Grid_XYZW[ 1252][0] = -0.893005179084777;
    Leb_Grid_XYZW[ 1252][1] = -0.409778253704889;
    Leb_Grid_XYZW[ 1252][2] =  0.186073998501503;
    Leb_Grid_XYZW[ 1252][3] =  0.000507224291007;

    Leb_Grid_XYZW[ 1253][0] = -0.893005179084777;
    Leb_Grid_XYZW[ 1253][1] = -0.409778253704889;
    Leb_Grid_XYZW[ 1253][2] = -0.186073998501503;
    Leb_Grid_XYZW[ 1253][3] =  0.000507224291007;

    Leb_Grid_XYZW[ 1254][0] =  0.893005179084777;
    Leb_Grid_XYZW[ 1254][1] =  0.186073998501503;
    Leb_Grid_XYZW[ 1254][2] =  0.409778253704889;
    Leb_Grid_XYZW[ 1254][3] =  0.000507224291007;

    Leb_Grid_XYZW[ 1255][0] =  0.893005179084777;
    Leb_Grid_XYZW[ 1255][1] =  0.186073998501503;
    Leb_Grid_XYZW[ 1255][2] = -0.409778253704889;
    Leb_Grid_XYZW[ 1255][3] =  0.000507224291007;

    Leb_Grid_XYZW[ 1256][0] =  0.893005179084777;
    Leb_Grid_XYZW[ 1256][1] = -0.186073998501503;
    Leb_Grid_XYZW[ 1256][2] =  0.409778253704889;
    Leb_Grid_XYZW[ 1256][3] =  0.000507224291007;

    Leb_Grid_XYZW[ 1257][0] =  0.893005179084777;
    Leb_Grid_XYZW[ 1257][1] = -0.186073998501503;
    Leb_Grid_XYZW[ 1257][2] = -0.409778253704889;
    Leb_Grid_XYZW[ 1257][3] =  0.000507224291007;

    Leb_Grid_XYZW[ 1258][0] = -0.893005179084777;
    Leb_Grid_XYZW[ 1258][1] =  0.186073998501503;
    Leb_Grid_XYZW[ 1258][2] =  0.409778253704889;
    Leb_Grid_XYZW[ 1258][3] =  0.000507224291007;

    Leb_Grid_XYZW[ 1259][0] = -0.893005179084777;
    Leb_Grid_XYZW[ 1259][1] =  0.186073998501503;
    Leb_Grid_XYZW[ 1259][2] = -0.409778253704889;
    Leb_Grid_XYZW[ 1259][3] =  0.000507224291007;

    Leb_Grid_XYZW[ 1260][0] = -0.893005179084777;
    Leb_Grid_XYZW[ 1260][1] = -0.186073998501503;
    Leb_Grid_XYZW[ 1260][2] =  0.409778253704889;
    Leb_Grid_XYZW[ 1260][3] =  0.000507224291007;

    Leb_Grid_XYZW[ 1261][0] = -0.893005179084777;
    Leb_Grid_XYZW[ 1261][1] = -0.186073998501503;
    Leb_Grid_XYZW[ 1261][2] = -0.409778253704889;
    Leb_Grid_XYZW[ 1261][3] =  0.000507224291007;

    Leb_Grid_XYZW[ 1262][0] =  0.461633766606746;
    Leb_Grid_XYZW[ 1262][1] =  0.235423507739585;
    Leb_Grid_XYZW[ 1262][2] =  0.855260216268743;
    Leb_Grid_XYZW[ 1262][3] =  0.000521706984524;

    Leb_Grid_XYZW[ 1263][0] =  0.461633766606746;
    Leb_Grid_XYZW[ 1263][1] =  0.235423507739585;
    Leb_Grid_XYZW[ 1263][2] = -0.855260216268743;
    Leb_Grid_XYZW[ 1263][3] =  0.000521706984524;

    Leb_Grid_XYZW[ 1264][0] =  0.461633766606746;
    Leb_Grid_XYZW[ 1264][1] = -0.235423507739585;
    Leb_Grid_XYZW[ 1264][2] =  0.855260216268743;
    Leb_Grid_XYZW[ 1264][3] =  0.000521706984524;

    Leb_Grid_XYZW[ 1265][0] =  0.461633766606746;
    Leb_Grid_XYZW[ 1265][1] = -0.235423507739585;
    Leb_Grid_XYZW[ 1265][2] = -0.855260216268743;
    Leb_Grid_XYZW[ 1265][3] =  0.000521706984524;

    Leb_Grid_XYZW[ 1266][0] = -0.461633766606746;
    Leb_Grid_XYZW[ 1266][1] =  0.235423507739585;
    Leb_Grid_XYZW[ 1266][2] =  0.855260216268743;
    Leb_Grid_XYZW[ 1266][3] =  0.000521706984524;

    Leb_Grid_XYZW[ 1267][0] = -0.461633766606746;
    Leb_Grid_XYZW[ 1267][1] =  0.235423507739585;
    Leb_Grid_XYZW[ 1267][2] = -0.855260216268743;
    Leb_Grid_XYZW[ 1267][3] =  0.000521706984524;

    Leb_Grid_XYZW[ 1268][0] = -0.461633766606746;
    Leb_Grid_XYZW[ 1268][1] = -0.235423507739585;
    Leb_Grid_XYZW[ 1268][2] =  0.855260216268743;
    Leb_Grid_XYZW[ 1268][3] =  0.000521706984524;

    Leb_Grid_XYZW[ 1269][0] = -0.461633766606746;
    Leb_Grid_XYZW[ 1269][1] = -0.235423507739585;
    Leb_Grid_XYZW[ 1269][2] = -0.855260216268743;
    Leb_Grid_XYZW[ 1269][3] =  0.000521706984524;

    Leb_Grid_XYZW[ 1270][0] =  0.461633766606746;
    Leb_Grid_XYZW[ 1270][1] =  0.855260216268744;
    Leb_Grid_XYZW[ 1270][2] =  0.235423507739585;
    Leb_Grid_XYZW[ 1270][3] =  0.000521706984524;

    Leb_Grid_XYZW[ 1271][0] =  0.461633766606746;
    Leb_Grid_XYZW[ 1271][1] =  0.855260216268744;
    Leb_Grid_XYZW[ 1271][2] = -0.235423507739585;
    Leb_Grid_XYZW[ 1271][3] =  0.000521706984524;

    Leb_Grid_XYZW[ 1272][0] =  0.461633766606746;
    Leb_Grid_XYZW[ 1272][1] = -0.855260216268744;
    Leb_Grid_XYZW[ 1272][2] =  0.235423507739585;
    Leb_Grid_XYZW[ 1272][3] =  0.000521706984524;

    Leb_Grid_XYZW[ 1273][0] =  0.461633766606746;
    Leb_Grid_XYZW[ 1273][1] = -0.855260216268744;
    Leb_Grid_XYZW[ 1273][2] = -0.235423507739585;
    Leb_Grid_XYZW[ 1273][3] =  0.000521706984524;

    Leb_Grid_XYZW[ 1274][0] = -0.461633766606746;
    Leb_Grid_XYZW[ 1274][1] =  0.855260216268744;
    Leb_Grid_XYZW[ 1274][2] =  0.235423507739585;
    Leb_Grid_XYZW[ 1274][3] =  0.000521706984524;

    Leb_Grid_XYZW[ 1275][0] = -0.461633766606746;
    Leb_Grid_XYZW[ 1275][1] =  0.855260216268744;
    Leb_Grid_XYZW[ 1275][2] = -0.235423507739585;
    Leb_Grid_XYZW[ 1275][3] =  0.000521706984524;

    Leb_Grid_XYZW[ 1276][0] = -0.461633766606746;
    Leb_Grid_XYZW[ 1276][1] = -0.855260216268744;
    Leb_Grid_XYZW[ 1276][2] =  0.235423507739585;
    Leb_Grid_XYZW[ 1276][3] =  0.000521706984524;

    Leb_Grid_XYZW[ 1277][0] = -0.461633766606746;
    Leb_Grid_XYZW[ 1277][1] = -0.855260216268744;
    Leb_Grid_XYZW[ 1277][2] = -0.235423507739585;
    Leb_Grid_XYZW[ 1277][3] =  0.000521706984524;

    Leb_Grid_XYZW[ 1278][0] =  0.235423507739585;
    Leb_Grid_XYZW[ 1278][1] =  0.461633766606746;
    Leb_Grid_XYZW[ 1278][2] =  0.855260216268743;
    Leb_Grid_XYZW[ 1278][3] =  0.000521706984524;

    Leb_Grid_XYZW[ 1279][0] =  0.235423507739585;
    Leb_Grid_XYZW[ 1279][1] =  0.461633766606746;
    Leb_Grid_XYZW[ 1279][2] = -0.855260216268743;
    Leb_Grid_XYZW[ 1279][3] =  0.000521706984524;

    Leb_Grid_XYZW[ 1280][0] =  0.235423507739585;
    Leb_Grid_XYZW[ 1280][1] = -0.461633766606746;
    Leb_Grid_XYZW[ 1280][2] =  0.855260216268743;
    Leb_Grid_XYZW[ 1280][3] =  0.000521706984524;

    Leb_Grid_XYZW[ 1281][0] =  0.235423507739585;
    Leb_Grid_XYZW[ 1281][1] = -0.461633766606746;
    Leb_Grid_XYZW[ 1281][2] = -0.855260216268743;
    Leb_Grid_XYZW[ 1281][3] =  0.000521706984524;

    Leb_Grid_XYZW[ 1282][0] = -0.235423507739585;
    Leb_Grid_XYZW[ 1282][1] =  0.461633766606746;
    Leb_Grid_XYZW[ 1282][2] =  0.855260216268743;
    Leb_Grid_XYZW[ 1282][3] =  0.000521706984524;

    Leb_Grid_XYZW[ 1283][0] = -0.235423507739585;
    Leb_Grid_XYZW[ 1283][1] =  0.461633766606746;
    Leb_Grid_XYZW[ 1283][2] = -0.855260216268743;
    Leb_Grid_XYZW[ 1283][3] =  0.000521706984524;

    Leb_Grid_XYZW[ 1284][0] = -0.235423507739585;
    Leb_Grid_XYZW[ 1284][1] = -0.461633766606746;
    Leb_Grid_XYZW[ 1284][2] =  0.855260216268743;
    Leb_Grid_XYZW[ 1284][3] =  0.000521706984524;

    Leb_Grid_XYZW[ 1285][0] = -0.235423507739585;
    Leb_Grid_XYZW[ 1285][1] = -0.461633766606746;
    Leb_Grid_XYZW[ 1285][2] = -0.855260216268743;
    Leb_Grid_XYZW[ 1285][3] =  0.000521706984524;

    Leb_Grid_XYZW[ 1286][0] =  0.235423507739585;
    Leb_Grid_XYZW[ 1286][1] =  0.855260216268743;
    Leb_Grid_XYZW[ 1286][2] =  0.461633766606746;
    Leb_Grid_XYZW[ 1286][3] =  0.000521706984524;

    Leb_Grid_XYZW[ 1287][0] =  0.235423507739585;
    Leb_Grid_XYZW[ 1287][1] =  0.855260216268743;
    Leb_Grid_XYZW[ 1287][2] = -0.461633766606746;
    Leb_Grid_XYZW[ 1287][3] =  0.000521706984524;

    Leb_Grid_XYZW[ 1288][0] =  0.235423507739585;
    Leb_Grid_XYZW[ 1288][1] = -0.855260216268743;
    Leb_Grid_XYZW[ 1288][2] =  0.461633766606746;
    Leb_Grid_XYZW[ 1288][3] =  0.000521706984524;

    Leb_Grid_XYZW[ 1289][0] =  0.235423507739585;
    Leb_Grid_XYZW[ 1289][1] = -0.855260216268743;
    Leb_Grid_XYZW[ 1289][2] = -0.461633766606746;
    Leb_Grid_XYZW[ 1289][3] =  0.000521706984524;

    Leb_Grid_XYZW[ 1290][0] = -0.235423507739585;
    Leb_Grid_XYZW[ 1290][1] =  0.855260216268744;
    Leb_Grid_XYZW[ 1290][2] =  0.461633766606746;
    Leb_Grid_XYZW[ 1290][3] =  0.000521706984524;

    Leb_Grid_XYZW[ 1291][0] = -0.235423507739585;
    Leb_Grid_XYZW[ 1291][1] =  0.855260216268744;
    Leb_Grid_XYZW[ 1291][2] = -0.461633766606746;
    Leb_Grid_XYZW[ 1291][3] =  0.000521706984524;

    Leb_Grid_XYZW[ 1292][0] = -0.235423507739585;
    Leb_Grid_XYZW[ 1292][1] = -0.855260216268744;
    Leb_Grid_XYZW[ 1292][2] =  0.461633766606746;
    Leb_Grid_XYZW[ 1292][3] =  0.000521706984524;

    Leb_Grid_XYZW[ 1293][0] = -0.235423507739585;
    Leb_Grid_XYZW[ 1293][1] = -0.855260216268744;
    Leb_Grid_XYZW[ 1293][2] = -0.461633766606746;
    Leb_Grid_XYZW[ 1293][3] =  0.000521706984524;

    Leb_Grid_XYZW[ 1294][0] =  0.855260216268743;
    Leb_Grid_XYZW[ 1294][1] =  0.461633766606746;
    Leb_Grid_XYZW[ 1294][2] =  0.235423507739585;
    Leb_Grid_XYZW[ 1294][3] =  0.000521706984524;

    Leb_Grid_XYZW[ 1295][0] =  0.855260216268743;
    Leb_Grid_XYZW[ 1295][1] =  0.461633766606746;
    Leb_Grid_XYZW[ 1295][2] = -0.235423507739585;
    Leb_Grid_XYZW[ 1295][3] =  0.000521706984524;

    Leb_Grid_XYZW[ 1296][0] =  0.855260216268743;
    Leb_Grid_XYZW[ 1296][1] = -0.461633766606746;
    Leb_Grid_XYZW[ 1296][2] =  0.235423507739585;
    Leb_Grid_XYZW[ 1296][3] =  0.000521706984524;

    Leb_Grid_XYZW[ 1297][0] =  0.855260216268743;
    Leb_Grid_XYZW[ 1297][1] = -0.461633766606746;
    Leb_Grid_XYZW[ 1297][2] = -0.235423507739585;
    Leb_Grid_XYZW[ 1297][3] =  0.000521706984524;

    Leb_Grid_XYZW[ 1298][0] = -0.855260216268743;
    Leb_Grid_XYZW[ 1298][1] =  0.461633766606746;
    Leb_Grid_XYZW[ 1298][2] =  0.235423507739585;
    Leb_Grid_XYZW[ 1298][3] =  0.000521706984524;

    Leb_Grid_XYZW[ 1299][0] = -0.855260216268743;
    Leb_Grid_XYZW[ 1299][1] =  0.461633766606746;
    Leb_Grid_XYZW[ 1299][2] = -0.235423507739585;
    Leb_Grid_XYZW[ 1299][3] =  0.000521706984524;

    Leb_Grid_XYZW[ 1300][0] = -0.855260216268743;
    Leb_Grid_XYZW[ 1300][1] = -0.461633766606746;
    Leb_Grid_XYZW[ 1300][2] =  0.235423507739585;
    Leb_Grid_XYZW[ 1300][3] =  0.000521706984524;

    Leb_Grid_XYZW[ 1301][0] = -0.855260216268743;
    Leb_Grid_XYZW[ 1301][1] = -0.461633766606746;
    Leb_Grid_XYZW[ 1301][2] = -0.235423507739585;
    Leb_Grid_XYZW[ 1301][3] =  0.000521706984524;

    Leb_Grid_XYZW[ 1302][0] =  0.855260216268743;
    Leb_Grid_XYZW[ 1302][1] =  0.235423507739585;
    Leb_Grid_XYZW[ 1302][2] =  0.461633766606746;
    Leb_Grid_XYZW[ 1302][3] =  0.000521706984524;

    Leb_Grid_XYZW[ 1303][0] =  0.855260216268743;
    Leb_Grid_XYZW[ 1303][1] =  0.235423507739585;
    Leb_Grid_XYZW[ 1303][2] = -0.461633766606746;
    Leb_Grid_XYZW[ 1303][3] =  0.000521706984524;

    Leb_Grid_XYZW[ 1304][0] =  0.855260216268743;
    Leb_Grid_XYZW[ 1304][1] = -0.235423507739585;
    Leb_Grid_XYZW[ 1304][2] =  0.461633766606746;
    Leb_Grid_XYZW[ 1304][3] =  0.000521706984524;

    Leb_Grid_XYZW[ 1305][0] =  0.855260216268743;
    Leb_Grid_XYZW[ 1305][1] = -0.235423507739585;
    Leb_Grid_XYZW[ 1305][2] = -0.461633766606746;
    Leb_Grid_XYZW[ 1305][3] =  0.000521706984524;

    Leb_Grid_XYZW[ 1306][0] = -0.855260216268743;
    Leb_Grid_XYZW[ 1306][1] =  0.235423507739585;
    Leb_Grid_XYZW[ 1306][2] =  0.461633766606746;
    Leb_Grid_XYZW[ 1306][3] =  0.000521706984524;

    Leb_Grid_XYZW[ 1307][0] = -0.855260216268743;
    Leb_Grid_XYZW[ 1307][1] =  0.235423507739585;
    Leb_Grid_XYZW[ 1307][2] = -0.461633766606746;
    Leb_Grid_XYZW[ 1307][3] =  0.000521706984524;

    Leb_Grid_XYZW[ 1308][0] = -0.855260216268743;
    Leb_Grid_XYZW[ 1308][1] = -0.235423507739585;
    Leb_Grid_XYZW[ 1308][2] =  0.461633766606746;
    Leb_Grid_XYZW[ 1308][3] =  0.000521706984524;

    Leb_Grid_XYZW[ 1309][0] = -0.855260216268743;
    Leb_Grid_XYZW[ 1309][1] = -0.235423507739585;
    Leb_Grid_XYZW[ 1309][2] = -0.461633766606746;
    Leb_Grid_XYZW[ 1309][3] =  0.000521706984524;

    Leb_Grid_XYZW[ 1310][0] =  0.511070700841788;
    Leb_Grid_XYZW[ 1310][1] =  0.284207492134701;
    Leb_Grid_XYZW[ 1310][2] =  0.811192233786535;
    Leb_Grid_XYZW[ 1310][3] =  0.000531578596628;

    Leb_Grid_XYZW[ 1311][0] =  0.511070700841788;
    Leb_Grid_XYZW[ 1311][1] =  0.284207492134701;
    Leb_Grid_XYZW[ 1311][2] = -0.811192233786535;
    Leb_Grid_XYZW[ 1311][3] =  0.000531578596628;

    Leb_Grid_XYZW[ 1312][0] =  0.511070700841788;
    Leb_Grid_XYZW[ 1312][1] = -0.284207492134701;
    Leb_Grid_XYZW[ 1312][2] =  0.811192233786535;
    Leb_Grid_XYZW[ 1312][3] =  0.000531578596628;

    Leb_Grid_XYZW[ 1313][0] =  0.511070700841788;
    Leb_Grid_XYZW[ 1313][1] = -0.284207492134701;
    Leb_Grid_XYZW[ 1313][2] = -0.811192233786535;
    Leb_Grid_XYZW[ 1313][3] =  0.000531578596628;

    Leb_Grid_XYZW[ 1314][0] = -0.511070700841787;
    Leb_Grid_XYZW[ 1314][1] =  0.284207492134701;
    Leb_Grid_XYZW[ 1314][2] =  0.811192233786535;
    Leb_Grid_XYZW[ 1314][3] =  0.000531578596628;

    Leb_Grid_XYZW[ 1315][0] = -0.511070700841787;
    Leb_Grid_XYZW[ 1315][1] =  0.284207492134701;
    Leb_Grid_XYZW[ 1315][2] = -0.811192233786535;
    Leb_Grid_XYZW[ 1315][3] =  0.000531578596628;

    Leb_Grid_XYZW[ 1316][0] = -0.511070700841787;
    Leb_Grid_XYZW[ 1316][1] = -0.284207492134701;
    Leb_Grid_XYZW[ 1316][2] =  0.811192233786535;
    Leb_Grid_XYZW[ 1316][3] =  0.000531578596628;

    Leb_Grid_XYZW[ 1317][0] = -0.511070700841787;
    Leb_Grid_XYZW[ 1317][1] = -0.284207492134701;
    Leb_Grid_XYZW[ 1317][2] = -0.811192233786535;
    Leb_Grid_XYZW[ 1317][3] =  0.000531578596628;

    Leb_Grid_XYZW[ 1318][0] =  0.511070700841787;
    Leb_Grid_XYZW[ 1318][1] =  0.811192233786535;
    Leb_Grid_XYZW[ 1318][2] =  0.284207492134701;
    Leb_Grid_XYZW[ 1318][3] =  0.000531578596628;

    Leb_Grid_XYZW[ 1319][0] =  0.511070700841787;
    Leb_Grid_XYZW[ 1319][1] =  0.811192233786535;
    Leb_Grid_XYZW[ 1319][2] = -0.284207492134701;
    Leb_Grid_XYZW[ 1319][3] =  0.000531578596628;

    Leb_Grid_XYZW[ 1320][0] =  0.511070700841787;
    Leb_Grid_XYZW[ 1320][1] = -0.811192233786535;
    Leb_Grid_XYZW[ 1320][2] =  0.284207492134701;
    Leb_Grid_XYZW[ 1320][3] =  0.000531578596628;

    Leb_Grid_XYZW[ 1321][0] =  0.511070700841787;
    Leb_Grid_XYZW[ 1321][1] = -0.811192233786535;
    Leb_Grid_XYZW[ 1321][2] = -0.284207492134701;
    Leb_Grid_XYZW[ 1321][3] =  0.000531578596628;

    Leb_Grid_XYZW[ 1322][0] = -0.511070700841787;
    Leb_Grid_XYZW[ 1322][1] =  0.811192233786535;
    Leb_Grid_XYZW[ 1322][2] =  0.284207492134701;
    Leb_Grid_XYZW[ 1322][3] =  0.000531578596628;

    Leb_Grid_XYZW[ 1323][0] = -0.511070700841787;
    Leb_Grid_XYZW[ 1323][1] =  0.811192233786535;
    Leb_Grid_XYZW[ 1323][2] = -0.284207492134701;
    Leb_Grid_XYZW[ 1323][3] =  0.000531578596628;

    Leb_Grid_XYZW[ 1324][0] = -0.511070700841787;
    Leb_Grid_XYZW[ 1324][1] = -0.811192233786535;
    Leb_Grid_XYZW[ 1324][2] =  0.284207492134701;
    Leb_Grid_XYZW[ 1324][3] =  0.000531578596628;

    Leb_Grid_XYZW[ 1325][0] = -0.511070700841787;
    Leb_Grid_XYZW[ 1325][1] = -0.811192233786535;
    Leb_Grid_XYZW[ 1325][2] = -0.284207492134701;
    Leb_Grid_XYZW[ 1325][3] =  0.000531578596628;

    Leb_Grid_XYZW[ 1326][0] =  0.284207492134701;
    Leb_Grid_XYZW[ 1326][1] =  0.511070700841788;
    Leb_Grid_XYZW[ 1326][2] =  0.811192233786535;
    Leb_Grid_XYZW[ 1326][3] =  0.000531578596628;

    Leb_Grid_XYZW[ 1327][0] =  0.284207492134701;
    Leb_Grid_XYZW[ 1327][1] =  0.511070700841788;
    Leb_Grid_XYZW[ 1327][2] = -0.811192233786535;
    Leb_Grid_XYZW[ 1327][3] =  0.000531578596628;

    Leb_Grid_XYZW[ 1328][0] =  0.284207492134701;
    Leb_Grid_XYZW[ 1328][1] = -0.511070700841788;
    Leb_Grid_XYZW[ 1328][2] =  0.811192233786535;
    Leb_Grid_XYZW[ 1328][3] =  0.000531578596628;

    Leb_Grid_XYZW[ 1329][0] =  0.284207492134701;
    Leb_Grid_XYZW[ 1329][1] = -0.511070700841788;
    Leb_Grid_XYZW[ 1329][2] = -0.811192233786535;
    Leb_Grid_XYZW[ 1329][3] =  0.000531578596628;

    Leb_Grid_XYZW[ 1330][0] = -0.284207492134701;
    Leb_Grid_XYZW[ 1330][1] =  0.511070700841788;
    Leb_Grid_XYZW[ 1330][2] =  0.811192233786535;
    Leb_Grid_XYZW[ 1330][3] =  0.000531578596628;

    Leb_Grid_XYZW[ 1331][0] = -0.284207492134701;
    Leb_Grid_XYZW[ 1331][1] =  0.511070700841788;
    Leb_Grid_XYZW[ 1331][2] = -0.811192233786535;
    Leb_Grid_XYZW[ 1331][3] =  0.000531578596628;

    Leb_Grid_XYZW[ 1332][0] = -0.284207492134701;
    Leb_Grid_XYZW[ 1332][1] = -0.511070700841788;
    Leb_Grid_XYZW[ 1332][2] =  0.811192233786535;
    Leb_Grid_XYZW[ 1332][3] =  0.000531578596628;

    Leb_Grid_XYZW[ 1333][0] = -0.284207492134701;
    Leb_Grid_XYZW[ 1333][1] = -0.511070700841788;
    Leb_Grid_XYZW[ 1333][2] = -0.811192233786535;
    Leb_Grid_XYZW[ 1333][3] =  0.000531578596628;

    Leb_Grid_XYZW[ 1334][0] =  0.284207492134701;
    Leb_Grid_XYZW[ 1334][1] =  0.811192233786535;
    Leb_Grid_XYZW[ 1334][2] =  0.511070700841787;
    Leb_Grid_XYZW[ 1334][3] =  0.000531578596628;

    Leb_Grid_XYZW[ 1335][0] =  0.284207492134701;
    Leb_Grid_XYZW[ 1335][1] =  0.811192233786535;
    Leb_Grid_XYZW[ 1335][2] = -0.511070700841787;
    Leb_Grid_XYZW[ 1335][3] =  0.000531578596628;

    Leb_Grid_XYZW[ 1336][0] =  0.284207492134701;
    Leb_Grid_XYZW[ 1336][1] = -0.811192233786535;
    Leb_Grid_XYZW[ 1336][2] =  0.511070700841787;
    Leb_Grid_XYZW[ 1336][3] =  0.000531578596628;

    Leb_Grid_XYZW[ 1337][0] =  0.284207492134701;
    Leb_Grid_XYZW[ 1337][1] = -0.811192233786535;
    Leb_Grid_XYZW[ 1337][2] = -0.511070700841787;
    Leb_Grid_XYZW[ 1337][3] =  0.000531578596628;

    Leb_Grid_XYZW[ 1338][0] = -0.284207492134701;
    Leb_Grid_XYZW[ 1338][1] =  0.811192233786535;
    Leb_Grid_XYZW[ 1338][2] =  0.511070700841787;
    Leb_Grid_XYZW[ 1338][3] =  0.000531578596628;

    Leb_Grid_XYZW[ 1339][0] = -0.284207492134701;
    Leb_Grid_XYZW[ 1339][1] =  0.811192233786535;
    Leb_Grid_XYZW[ 1339][2] = -0.511070700841787;
    Leb_Grid_XYZW[ 1339][3] =  0.000531578596628;

    Leb_Grid_XYZW[ 1340][0] = -0.284207492134701;
    Leb_Grid_XYZW[ 1340][1] = -0.811192233786535;
    Leb_Grid_XYZW[ 1340][2] =  0.511070700841787;
    Leb_Grid_XYZW[ 1340][3] =  0.000531578596628;

    Leb_Grid_XYZW[ 1341][0] = -0.284207492134701;
    Leb_Grid_XYZW[ 1341][1] = -0.811192233786535;
    Leb_Grid_XYZW[ 1341][2] = -0.511070700841787;
    Leb_Grid_XYZW[ 1341][3] =  0.000531578596628;

    Leb_Grid_XYZW[ 1342][0] =  0.811192233786535;
    Leb_Grid_XYZW[ 1342][1] =  0.511070700841788;
    Leb_Grid_XYZW[ 1342][2] =  0.284207492134701;
    Leb_Grid_XYZW[ 1342][3] =  0.000531578596628;

    Leb_Grid_XYZW[ 1343][0] =  0.811192233786535;
    Leb_Grid_XYZW[ 1343][1] =  0.511070700841788;
    Leb_Grid_XYZW[ 1343][2] = -0.284207492134701;
    Leb_Grid_XYZW[ 1343][3] =  0.000531578596628;

    Leb_Grid_XYZW[ 1344][0] =  0.811192233786535;
    Leb_Grid_XYZW[ 1344][1] = -0.511070700841788;
    Leb_Grid_XYZW[ 1344][2] =  0.284207492134701;
    Leb_Grid_XYZW[ 1344][3] =  0.000531578596628;

    Leb_Grid_XYZW[ 1345][0] =  0.811192233786535;
    Leb_Grid_XYZW[ 1345][1] = -0.511070700841788;
    Leb_Grid_XYZW[ 1345][2] = -0.284207492134701;
    Leb_Grid_XYZW[ 1345][3] =  0.000531578596628;

    Leb_Grid_XYZW[ 1346][0] = -0.811192233786535;
    Leb_Grid_XYZW[ 1346][1] =  0.511070700841787;
    Leb_Grid_XYZW[ 1346][2] =  0.284207492134701;
    Leb_Grid_XYZW[ 1346][3] =  0.000531578596628;

    Leb_Grid_XYZW[ 1347][0] = -0.811192233786535;
    Leb_Grid_XYZW[ 1347][1] =  0.511070700841787;
    Leb_Grid_XYZW[ 1347][2] = -0.284207492134701;
    Leb_Grid_XYZW[ 1347][3] =  0.000531578596628;

    Leb_Grid_XYZW[ 1348][0] = -0.811192233786535;
    Leb_Grid_XYZW[ 1348][1] = -0.511070700841787;
    Leb_Grid_XYZW[ 1348][2] =  0.284207492134701;
    Leb_Grid_XYZW[ 1348][3] =  0.000531578596628;

    Leb_Grid_XYZW[ 1349][0] = -0.811192233786535;
    Leb_Grid_XYZW[ 1349][1] = -0.511070700841787;
    Leb_Grid_XYZW[ 1349][2] = -0.284207492134701;
    Leb_Grid_XYZW[ 1349][3] =  0.000531578596628;

    Leb_Grid_XYZW[ 1350][0] =  0.811192233786535;
    Leb_Grid_XYZW[ 1350][1] =  0.284207492134701;
    Leb_Grid_XYZW[ 1350][2] =  0.511070700841787;
    Leb_Grid_XYZW[ 1350][3] =  0.000531578596628;

    Leb_Grid_XYZW[ 1351][0] =  0.811192233786535;
    Leb_Grid_XYZW[ 1351][1] =  0.284207492134701;
    Leb_Grid_XYZW[ 1351][2] = -0.511070700841787;
    Leb_Grid_XYZW[ 1351][3] =  0.000531578596628;

    Leb_Grid_XYZW[ 1352][0] =  0.811192233786535;
    Leb_Grid_XYZW[ 1352][1] = -0.284207492134701;
    Leb_Grid_XYZW[ 1352][2] =  0.511070700841787;
    Leb_Grid_XYZW[ 1352][3] =  0.000531578596628;

    Leb_Grid_XYZW[ 1353][0] =  0.811192233786535;
    Leb_Grid_XYZW[ 1353][1] = -0.284207492134701;
    Leb_Grid_XYZW[ 1353][2] = -0.511070700841787;
    Leb_Grid_XYZW[ 1353][3] =  0.000531578596628;

    Leb_Grid_XYZW[ 1354][0] = -0.811192233786535;
    Leb_Grid_XYZW[ 1354][1] =  0.284207492134701;
    Leb_Grid_XYZW[ 1354][2] =  0.511070700841787;
    Leb_Grid_XYZW[ 1354][3] =  0.000531578596628;

    Leb_Grid_XYZW[ 1355][0] = -0.811192233786535;
    Leb_Grid_XYZW[ 1355][1] =  0.284207492134701;
    Leb_Grid_XYZW[ 1355][2] = -0.511070700841787;
    Leb_Grid_XYZW[ 1355][3] =  0.000531578596628;

    Leb_Grid_XYZW[ 1356][0] = -0.811192233786535;
    Leb_Grid_XYZW[ 1356][1] = -0.284207492134701;
    Leb_Grid_XYZW[ 1356][2] =  0.511070700841787;
    Leb_Grid_XYZW[ 1356][3] =  0.000531578596628;

    Leb_Grid_XYZW[ 1357][0] = -0.811192233786535;
    Leb_Grid_XYZW[ 1357][1] = -0.284207492134701;
    Leb_Grid_XYZW[ 1357][2] = -0.511070700841787;
    Leb_Grid_XYZW[ 1357][3] =  0.000531578596628;

    Leb_Grid_XYZW[ 1358][0] =  0.557741528616380;
    Leb_Grid_XYZW[ 1358][1] =  0.331778441498410;
    Leb_Grid_XYZW[ 1358][2] =  0.760820250133729;
    Leb_Grid_XYZW[ 1358][3] =  0.000537683370876;

    Leb_Grid_XYZW[ 1359][0] =  0.557741528616380;
    Leb_Grid_XYZW[ 1359][1] =  0.331778441498410;
    Leb_Grid_XYZW[ 1359][2] = -0.760820250133729;
    Leb_Grid_XYZW[ 1359][3] =  0.000537683370876;

    Leb_Grid_XYZW[ 1360][0] =  0.557741528616380;
    Leb_Grid_XYZW[ 1360][1] = -0.331778441498410;
    Leb_Grid_XYZW[ 1360][2] =  0.760820250133729;
    Leb_Grid_XYZW[ 1360][3] =  0.000537683370876;

    Leb_Grid_XYZW[ 1361][0] =  0.557741528616380;
    Leb_Grid_XYZW[ 1361][1] = -0.331778441498410;
    Leb_Grid_XYZW[ 1361][2] = -0.760820250133729;
    Leb_Grid_XYZW[ 1361][3] =  0.000537683370876;

    Leb_Grid_XYZW[ 1362][0] = -0.557741528616379;
    Leb_Grid_XYZW[ 1362][1] =  0.331778441498411;
    Leb_Grid_XYZW[ 1362][2] =  0.760820250133729;
    Leb_Grid_XYZW[ 1362][3] =  0.000537683370876;

    Leb_Grid_XYZW[ 1363][0] = -0.557741528616379;
    Leb_Grid_XYZW[ 1363][1] =  0.331778441498411;
    Leb_Grid_XYZW[ 1363][2] = -0.760820250133729;
    Leb_Grid_XYZW[ 1363][3] =  0.000537683370876;

    Leb_Grid_XYZW[ 1364][0] = -0.557741528616379;
    Leb_Grid_XYZW[ 1364][1] = -0.331778441498411;
    Leb_Grid_XYZW[ 1364][2] =  0.760820250133729;
    Leb_Grid_XYZW[ 1364][3] =  0.000537683370876;

    Leb_Grid_XYZW[ 1365][0] = -0.557741528616379;
    Leb_Grid_XYZW[ 1365][1] = -0.331778441498411;
    Leb_Grid_XYZW[ 1365][2] = -0.760820250133729;
    Leb_Grid_XYZW[ 1365][3] =  0.000537683370876;

    Leb_Grid_XYZW[ 1366][0] =  0.557741528616380;
    Leb_Grid_XYZW[ 1366][1] =  0.760820250133729;
    Leb_Grid_XYZW[ 1366][2] =  0.331778441498410;
    Leb_Grid_XYZW[ 1366][3] =  0.000537683370876;

    Leb_Grid_XYZW[ 1367][0] =  0.557741528616380;
    Leb_Grid_XYZW[ 1367][1] =  0.760820250133729;
    Leb_Grid_XYZW[ 1367][2] = -0.331778441498410;
    Leb_Grid_XYZW[ 1367][3] =  0.000537683370876;

    Leb_Grid_XYZW[ 1368][0] =  0.557741528616380;
    Leb_Grid_XYZW[ 1368][1] = -0.760820250133729;
    Leb_Grid_XYZW[ 1368][2] =  0.331778441498410;
    Leb_Grid_XYZW[ 1368][3] =  0.000537683370876;

    Leb_Grid_XYZW[ 1369][0] =  0.557741528616380;
    Leb_Grid_XYZW[ 1369][1] = -0.760820250133729;
    Leb_Grid_XYZW[ 1369][2] = -0.331778441498410;
    Leb_Grid_XYZW[ 1369][3] =  0.000537683370876;

    Leb_Grid_XYZW[ 1370][0] = -0.557741528616380;
    Leb_Grid_XYZW[ 1370][1] =  0.760820250133729;
    Leb_Grid_XYZW[ 1370][2] =  0.331778441498410;
    Leb_Grid_XYZW[ 1370][3] =  0.000537683370876;

    Leb_Grid_XYZW[ 1371][0] = -0.557741528616380;
    Leb_Grid_XYZW[ 1371][1] =  0.760820250133729;
    Leb_Grid_XYZW[ 1371][2] = -0.331778441498410;
    Leb_Grid_XYZW[ 1371][3] =  0.000537683370876;

    Leb_Grid_XYZW[ 1372][0] = -0.557741528616380;
    Leb_Grid_XYZW[ 1372][1] = -0.760820250133729;
    Leb_Grid_XYZW[ 1372][2] =  0.331778441498410;
    Leb_Grid_XYZW[ 1372][3] =  0.000537683370876;

    Leb_Grid_XYZW[ 1373][0] = -0.557741528616380;
    Leb_Grid_XYZW[ 1373][1] = -0.760820250133729;
    Leb_Grid_XYZW[ 1373][2] = -0.331778441498410;
    Leb_Grid_XYZW[ 1373][3] =  0.000537683370876;

    Leb_Grid_XYZW[ 1374][0] =  0.331778441498410;
    Leb_Grid_XYZW[ 1374][1] =  0.557741528616380;
    Leb_Grid_XYZW[ 1374][2] =  0.760820250133729;
    Leb_Grid_XYZW[ 1374][3] =  0.000537683370876;

    Leb_Grid_XYZW[ 1375][0] =  0.331778441498410;
    Leb_Grid_XYZW[ 1375][1] =  0.557741528616380;
    Leb_Grid_XYZW[ 1375][2] = -0.760820250133729;
    Leb_Grid_XYZW[ 1375][3] =  0.000537683370876;

    Leb_Grid_XYZW[ 1376][0] =  0.331778441498410;
    Leb_Grid_XYZW[ 1376][1] = -0.557741528616380;
    Leb_Grid_XYZW[ 1376][2] =  0.760820250133729;
    Leb_Grid_XYZW[ 1376][3] =  0.000537683370876;

    Leb_Grid_XYZW[ 1377][0] =  0.331778441498410;
    Leb_Grid_XYZW[ 1377][1] = -0.557741528616380;
    Leb_Grid_XYZW[ 1377][2] = -0.760820250133729;
    Leb_Grid_XYZW[ 1377][3] =  0.000537683370876;

    Leb_Grid_XYZW[ 1378][0] = -0.331778441498410;
    Leb_Grid_XYZW[ 1378][1] =  0.557741528616380;
    Leb_Grid_XYZW[ 1378][2] =  0.760820250133729;
    Leb_Grid_XYZW[ 1378][3] =  0.000537683370876;

    Leb_Grid_XYZW[ 1379][0] = -0.331778441498410;
    Leb_Grid_XYZW[ 1379][1] =  0.557741528616380;
    Leb_Grid_XYZW[ 1379][2] = -0.760820250133729;
    Leb_Grid_XYZW[ 1379][3] =  0.000537683370876;

    Leb_Grid_XYZW[ 1380][0] = -0.331778441498410;
    Leb_Grid_XYZW[ 1380][1] = -0.557741528616380;
    Leb_Grid_XYZW[ 1380][2] =  0.760820250133729;
    Leb_Grid_XYZW[ 1380][3] =  0.000537683370876;

    Leb_Grid_XYZW[ 1381][0] = -0.331778441498410;
    Leb_Grid_XYZW[ 1381][1] = -0.557741528616380;
    Leb_Grid_XYZW[ 1381][2] = -0.760820250133729;
    Leb_Grid_XYZW[ 1381][3] =  0.000537683370876;

    Leb_Grid_XYZW[ 1382][0] =  0.331778441498410;
    Leb_Grid_XYZW[ 1382][1] =  0.760820250133729;
    Leb_Grid_XYZW[ 1382][2] =  0.557741528616379;
    Leb_Grid_XYZW[ 1382][3] =  0.000537683370876;

    Leb_Grid_XYZW[ 1383][0] =  0.331778441498410;
    Leb_Grid_XYZW[ 1383][1] =  0.760820250133729;
    Leb_Grid_XYZW[ 1383][2] = -0.557741528616379;
    Leb_Grid_XYZW[ 1383][3] =  0.000537683370876;

    Leb_Grid_XYZW[ 1384][0] =  0.331778441498410;
    Leb_Grid_XYZW[ 1384][1] = -0.760820250133729;
    Leb_Grid_XYZW[ 1384][2] =  0.557741528616379;
    Leb_Grid_XYZW[ 1384][3] =  0.000537683370876;

    Leb_Grid_XYZW[ 1385][0] =  0.331778441498410;
    Leb_Grid_XYZW[ 1385][1] = -0.760820250133729;
    Leb_Grid_XYZW[ 1385][2] = -0.557741528616379;
    Leb_Grid_XYZW[ 1385][3] =  0.000537683370876;

    Leb_Grid_XYZW[ 1386][0] = -0.331778441498410;
    Leb_Grid_XYZW[ 1386][1] =  0.760820250133729;
    Leb_Grid_XYZW[ 1386][2] =  0.557741528616379;
    Leb_Grid_XYZW[ 1386][3] =  0.000537683370876;

    Leb_Grid_XYZW[ 1387][0] = -0.331778441498410;
    Leb_Grid_XYZW[ 1387][1] =  0.760820250133729;
    Leb_Grid_XYZW[ 1387][2] = -0.557741528616379;
    Leb_Grid_XYZW[ 1387][3] =  0.000537683370876;

    Leb_Grid_XYZW[ 1388][0] = -0.331778441498410;
    Leb_Grid_XYZW[ 1388][1] = -0.760820250133729;
    Leb_Grid_XYZW[ 1388][2] =  0.557741528616379;
    Leb_Grid_XYZW[ 1388][3] =  0.000537683370876;

    Leb_Grid_XYZW[ 1389][0] = -0.331778441498410;
    Leb_Grid_XYZW[ 1389][1] = -0.760820250133729;
    Leb_Grid_XYZW[ 1389][2] = -0.557741528616379;
    Leb_Grid_XYZW[ 1389][3] =  0.000537683370876;

    Leb_Grid_XYZW[ 1390][0] =  0.760820250133729;
    Leb_Grid_XYZW[ 1390][1] =  0.557741528616379;
    Leb_Grid_XYZW[ 1390][2] =  0.331778441498410;
    Leb_Grid_XYZW[ 1390][3] =  0.000537683370876;

    Leb_Grid_XYZW[ 1391][0] =  0.760820250133729;
    Leb_Grid_XYZW[ 1391][1] =  0.557741528616379;
    Leb_Grid_XYZW[ 1391][2] = -0.331778441498410;
    Leb_Grid_XYZW[ 1391][3] =  0.000537683370876;

    Leb_Grid_XYZW[ 1392][0] =  0.760820250133729;
    Leb_Grid_XYZW[ 1392][1] = -0.557741528616379;
    Leb_Grid_XYZW[ 1392][2] =  0.331778441498410;
    Leb_Grid_XYZW[ 1392][3] =  0.000537683370876;

    Leb_Grid_XYZW[ 1393][0] =  0.760820250133729;
    Leb_Grid_XYZW[ 1393][1] = -0.557741528616379;
    Leb_Grid_XYZW[ 1393][2] = -0.331778441498410;
    Leb_Grid_XYZW[ 1393][3] =  0.000537683370876;

    Leb_Grid_XYZW[ 1394][0] = -0.760820250133729;
    Leb_Grid_XYZW[ 1394][1] =  0.557741528616379;
    Leb_Grid_XYZW[ 1394][2] =  0.331778441498410;
    Leb_Grid_XYZW[ 1394][3] =  0.000537683370876;

    Leb_Grid_XYZW[ 1395][0] = -0.760820250133729;
    Leb_Grid_XYZW[ 1395][1] =  0.557741528616379;
    Leb_Grid_XYZW[ 1395][2] = -0.331778441498410;
    Leb_Grid_XYZW[ 1395][3] =  0.000537683370876;

    Leb_Grid_XYZW[ 1396][0] = -0.760820250133729;
    Leb_Grid_XYZW[ 1396][1] = -0.557741528616379;
    Leb_Grid_XYZW[ 1396][2] =  0.331778441498410;
    Leb_Grid_XYZW[ 1396][3] =  0.000537683370876;

    Leb_Grid_XYZW[ 1397][0] = -0.760820250133729;
    Leb_Grid_XYZW[ 1397][1] = -0.557741528616379;
    Leb_Grid_XYZW[ 1397][2] = -0.331778441498410;
    Leb_Grid_XYZW[ 1397][3] =  0.000537683370876;

    Leb_Grid_XYZW[ 1398][0] =  0.760820250133729;
    Leb_Grid_XYZW[ 1398][1] =  0.331778441498410;
    Leb_Grid_XYZW[ 1398][2] =  0.557741528616379;
    Leb_Grid_XYZW[ 1398][3] =  0.000537683370876;

    Leb_Grid_XYZW[ 1399][0] =  0.760820250133729;
    Leb_Grid_XYZW[ 1399][1] =  0.331778441498410;
    Leb_Grid_XYZW[ 1399][2] = -0.557741528616379;
    Leb_Grid_XYZW[ 1399][3] =  0.000537683370876;

    Leb_Grid_XYZW[ 1400][0] =  0.760820250133729;
    Leb_Grid_XYZW[ 1400][1] = -0.331778441498410;
    Leb_Grid_XYZW[ 1400][2] =  0.557741528616379;
    Leb_Grid_XYZW[ 1400][3] =  0.000537683370876;

    Leb_Grid_XYZW[ 1401][0] =  0.760820250133729;
    Leb_Grid_XYZW[ 1401][1] = -0.331778441498410;
    Leb_Grid_XYZW[ 1401][2] = -0.557741528616379;
    Leb_Grid_XYZW[ 1401][3] =  0.000537683370876;

    Leb_Grid_XYZW[ 1402][0] = -0.760820250133729;
    Leb_Grid_XYZW[ 1402][1] =  0.331778441498410;
    Leb_Grid_XYZW[ 1402][2] =  0.557741528616379;
    Leb_Grid_XYZW[ 1402][3] =  0.000537683370876;

    Leb_Grid_XYZW[ 1403][0] = -0.760820250133729;
    Leb_Grid_XYZW[ 1403][1] =  0.331778441498410;
    Leb_Grid_XYZW[ 1403][2] = -0.557741528616379;
    Leb_Grid_XYZW[ 1403][3] =  0.000537683370876;

    Leb_Grid_XYZW[ 1404][0] = -0.760820250133729;
    Leb_Grid_XYZW[ 1404][1] = -0.331778441498410;
    Leb_Grid_XYZW[ 1404][2] =  0.557741528616379;
    Leb_Grid_XYZW[ 1404][3] =  0.000537683370876;

    Leb_Grid_XYZW[ 1405][0] = -0.760820250133729;
    Leb_Grid_XYZW[ 1405][1] = -0.331778441498410;
    Leb_Grid_XYZW[ 1405][2] = -0.557741528616379;
    Leb_Grid_XYZW[ 1405][3] =  0.000537683370876;

    Leb_Grid_XYZW[ 1406][0] =  0.601306043136695;
    Leb_Grid_XYZW[ 1406][1] =  0.377529900204070;
    Leb_Grid_XYZW[ 1406][2] =  0.704203249736321;
    Leb_Grid_XYZW[ 1406][3] =  0.000540803209207;

    Leb_Grid_XYZW[ 1407][0] =  0.601306043136695;
    Leb_Grid_XYZW[ 1407][1] =  0.377529900204070;
    Leb_Grid_XYZW[ 1407][2] = -0.704203249736322;
    Leb_Grid_XYZW[ 1407][3] =  0.000540803209207;

    Leb_Grid_XYZW[ 1408][0] =  0.601306043136695;
    Leb_Grid_XYZW[ 1408][1] = -0.377529900204070;
    Leb_Grid_XYZW[ 1408][2] =  0.704203249736321;
    Leb_Grid_XYZW[ 1408][3] =  0.000540803209207;

    Leb_Grid_XYZW[ 1409][0] =  0.601306043136695;
    Leb_Grid_XYZW[ 1409][1] = -0.377529900204070;
    Leb_Grid_XYZW[ 1409][2] = -0.704203249736322;
    Leb_Grid_XYZW[ 1409][3] =  0.000540803209207;

    Leb_Grid_XYZW[ 1410][0] = -0.601306043136695;
    Leb_Grid_XYZW[ 1410][1] =  0.377529900204070;
    Leb_Grid_XYZW[ 1410][2] =  0.704203249736321;
    Leb_Grid_XYZW[ 1410][3] =  0.000540803209207;

    Leb_Grid_XYZW[ 1411][0] = -0.601306043136695;
    Leb_Grid_XYZW[ 1411][1] =  0.377529900204070;
    Leb_Grid_XYZW[ 1411][2] = -0.704203249736322;
    Leb_Grid_XYZW[ 1411][3] =  0.000540803209207;

    Leb_Grid_XYZW[ 1412][0] = -0.601306043136695;
    Leb_Grid_XYZW[ 1412][1] = -0.377529900204070;
    Leb_Grid_XYZW[ 1412][2] =  0.704203249736321;
    Leb_Grid_XYZW[ 1412][3] =  0.000540803209207;

    Leb_Grid_XYZW[ 1413][0] = -0.601306043136695;
    Leb_Grid_XYZW[ 1413][1] = -0.377529900204070;
    Leb_Grid_XYZW[ 1413][2] = -0.704203249736322;
    Leb_Grid_XYZW[ 1413][3] =  0.000540803209207;

    Leb_Grid_XYZW[ 1414][0] =  0.601306043136695;
    Leb_Grid_XYZW[ 1414][1] =  0.704203249736322;
    Leb_Grid_XYZW[ 1414][2] =  0.377529900204070;
    Leb_Grid_XYZW[ 1414][3] =  0.000540803209207;

    Leb_Grid_XYZW[ 1415][0] =  0.601306043136695;
    Leb_Grid_XYZW[ 1415][1] =  0.704203249736322;
    Leb_Grid_XYZW[ 1415][2] = -0.377529900204070;
    Leb_Grid_XYZW[ 1415][3] =  0.000540803209207;

    Leb_Grid_XYZW[ 1416][0] =  0.601306043136695;
    Leb_Grid_XYZW[ 1416][1] = -0.704203249736322;
    Leb_Grid_XYZW[ 1416][2] =  0.377529900204070;
    Leb_Grid_XYZW[ 1416][3] =  0.000540803209207;

    Leb_Grid_XYZW[ 1417][0] =  0.601306043136695;
    Leb_Grid_XYZW[ 1417][1] = -0.704203249736322;
    Leb_Grid_XYZW[ 1417][2] = -0.377529900204070;
    Leb_Grid_XYZW[ 1417][3] =  0.000540803209207;

    Leb_Grid_XYZW[ 1418][0] = -0.601306043136695;
    Leb_Grid_XYZW[ 1418][1] =  0.704203249736321;
    Leb_Grid_XYZW[ 1418][2] =  0.377529900204070;
    Leb_Grid_XYZW[ 1418][3] =  0.000540803209207;

    Leb_Grid_XYZW[ 1419][0] = -0.601306043136695;
    Leb_Grid_XYZW[ 1419][1] =  0.704203249736321;
    Leb_Grid_XYZW[ 1419][2] = -0.377529900204070;
    Leb_Grid_XYZW[ 1419][3] =  0.000540803209207;

    Leb_Grid_XYZW[ 1420][0] = -0.601306043136695;
    Leb_Grid_XYZW[ 1420][1] = -0.704203249736321;
    Leb_Grid_XYZW[ 1420][2] =  0.377529900204070;
    Leb_Grid_XYZW[ 1420][3] =  0.000540803209207;

    Leb_Grid_XYZW[ 1421][0] = -0.601306043136695;
    Leb_Grid_XYZW[ 1421][1] = -0.704203249736321;
    Leb_Grid_XYZW[ 1421][2] = -0.377529900204070;
    Leb_Grid_XYZW[ 1421][3] =  0.000540803209207;

    Leb_Grid_XYZW[ 1422][0] =  0.377529900204070;
    Leb_Grid_XYZW[ 1422][1] =  0.601306043136695;
    Leb_Grid_XYZW[ 1422][2] =  0.704203249736321;
    Leb_Grid_XYZW[ 1422][3] =  0.000540803209207;

    Leb_Grid_XYZW[ 1423][0] =  0.377529900204070;
    Leb_Grid_XYZW[ 1423][1] =  0.601306043136695;
    Leb_Grid_XYZW[ 1423][2] = -0.704203249736322;
    Leb_Grid_XYZW[ 1423][3] =  0.000540803209207;

    Leb_Grid_XYZW[ 1424][0] =  0.377529900204070;
    Leb_Grid_XYZW[ 1424][1] = -0.601306043136695;
    Leb_Grid_XYZW[ 1424][2] =  0.704203249736321;
    Leb_Grid_XYZW[ 1424][3] =  0.000540803209207;

    Leb_Grid_XYZW[ 1425][0] =  0.377529900204070;
    Leb_Grid_XYZW[ 1425][1] = -0.601306043136695;
    Leb_Grid_XYZW[ 1425][2] = -0.704203249736322;
    Leb_Grid_XYZW[ 1425][3] =  0.000540803209207;

    Leb_Grid_XYZW[ 1426][0] = -0.377529900204070;
    Leb_Grid_XYZW[ 1426][1] =  0.601306043136695;
    Leb_Grid_XYZW[ 1426][2] =  0.704203249736321;
    Leb_Grid_XYZW[ 1426][3] =  0.000540803209207;

    Leb_Grid_XYZW[ 1427][0] = -0.377529900204070;
    Leb_Grid_XYZW[ 1427][1] =  0.601306043136695;
    Leb_Grid_XYZW[ 1427][2] = -0.704203249736322;
    Leb_Grid_XYZW[ 1427][3] =  0.000540803209207;

    Leb_Grid_XYZW[ 1428][0] = -0.377529900204070;
    Leb_Grid_XYZW[ 1428][1] = -0.601306043136695;
    Leb_Grid_XYZW[ 1428][2] =  0.704203249736321;
    Leb_Grid_XYZW[ 1428][3] =  0.000540803209207;

    Leb_Grid_XYZW[ 1429][0] = -0.377529900204070;
    Leb_Grid_XYZW[ 1429][1] = -0.601306043136695;
    Leb_Grid_XYZW[ 1429][2] = -0.704203249736322;
    Leb_Grid_XYZW[ 1429][3] =  0.000540803209207;

    Leb_Grid_XYZW[ 1430][0] =  0.377529900204070;
    Leb_Grid_XYZW[ 1430][1] =  0.704203249736322;
    Leb_Grid_XYZW[ 1430][2] =  0.601306043136695;
    Leb_Grid_XYZW[ 1430][3] =  0.000540803209207;

    Leb_Grid_XYZW[ 1431][0] =  0.377529900204070;
    Leb_Grid_XYZW[ 1431][1] =  0.704203249736322;
    Leb_Grid_XYZW[ 1431][2] = -0.601306043136695;
    Leb_Grid_XYZW[ 1431][3] =  0.000540803209207;

    Leb_Grid_XYZW[ 1432][0] =  0.377529900204070;
    Leb_Grid_XYZW[ 1432][1] = -0.704203249736322;
    Leb_Grid_XYZW[ 1432][2] =  0.601306043136695;
    Leb_Grid_XYZW[ 1432][3] =  0.000540803209207;

    Leb_Grid_XYZW[ 1433][0] =  0.377529900204070;
    Leb_Grid_XYZW[ 1433][1] = -0.704203249736322;
    Leb_Grid_XYZW[ 1433][2] = -0.601306043136695;
    Leb_Grid_XYZW[ 1433][3] =  0.000540803209207;

    Leb_Grid_XYZW[ 1434][0] = -0.377529900204070;
    Leb_Grid_XYZW[ 1434][1] =  0.704203249736322;
    Leb_Grid_XYZW[ 1434][2] =  0.601306043136695;
    Leb_Grid_XYZW[ 1434][3] =  0.000540803209207;

    Leb_Grid_XYZW[ 1435][0] = -0.377529900204070;
    Leb_Grid_XYZW[ 1435][1] =  0.704203249736322;
    Leb_Grid_XYZW[ 1435][2] = -0.601306043136695;
    Leb_Grid_XYZW[ 1435][3] =  0.000540803209207;

    Leb_Grid_XYZW[ 1436][0] = -0.377529900204070;
    Leb_Grid_XYZW[ 1436][1] = -0.704203249736322;
    Leb_Grid_XYZW[ 1436][2] =  0.601306043136695;
    Leb_Grid_XYZW[ 1436][3] =  0.000540803209207;

    Leb_Grid_XYZW[ 1437][0] = -0.377529900204070;
    Leb_Grid_XYZW[ 1437][1] = -0.704203249736322;
    Leb_Grid_XYZW[ 1437][2] = -0.601306043136695;
    Leb_Grid_XYZW[ 1437][3] =  0.000540803209207;

    Leb_Grid_XYZW[ 1438][0] =  0.704203249736322;
    Leb_Grid_XYZW[ 1438][1] =  0.601306043136695;
    Leb_Grid_XYZW[ 1438][2] =  0.377529900204070;
    Leb_Grid_XYZW[ 1438][3] =  0.000540803209207;

    Leb_Grid_XYZW[ 1439][0] =  0.704203249736322;
    Leb_Grid_XYZW[ 1439][1] =  0.601306043136695;
    Leb_Grid_XYZW[ 1439][2] = -0.377529900204070;
    Leb_Grid_XYZW[ 1439][3] =  0.000540803209207;

    Leb_Grid_XYZW[ 1440][0] =  0.704203249736322;
    Leb_Grid_XYZW[ 1440][1] = -0.601306043136695;
    Leb_Grid_XYZW[ 1440][2] =  0.377529900204070;
    Leb_Grid_XYZW[ 1440][3] =  0.000540803209207;

    Leb_Grid_XYZW[ 1441][0] =  0.704203249736322;
    Leb_Grid_XYZW[ 1441][1] = -0.601306043136695;
    Leb_Grid_XYZW[ 1441][2] = -0.377529900204070;
    Leb_Grid_XYZW[ 1441][3] =  0.000540803209207;

    Leb_Grid_XYZW[ 1442][0] = -0.704203249736322;
    Leb_Grid_XYZW[ 1442][1] =  0.601306043136695;
    Leb_Grid_XYZW[ 1442][2] =  0.377529900204070;
    Leb_Grid_XYZW[ 1442][3] =  0.000540803209207;

    Leb_Grid_XYZW[ 1443][0] = -0.704203249736322;
    Leb_Grid_XYZW[ 1443][1] =  0.601306043136695;
    Leb_Grid_XYZW[ 1443][2] = -0.377529900204070;
    Leb_Grid_XYZW[ 1443][3] =  0.000540803209207;

    Leb_Grid_XYZW[ 1444][0] = -0.704203249736322;
    Leb_Grid_XYZW[ 1444][1] = -0.601306043136695;
    Leb_Grid_XYZW[ 1444][2] =  0.377529900204070;
    Leb_Grid_XYZW[ 1444][3] =  0.000540803209207;

    Leb_Grid_XYZW[ 1445][0] = -0.704203249736322;
    Leb_Grid_XYZW[ 1445][1] = -0.601306043136695;
    Leb_Grid_XYZW[ 1445][2] = -0.377529900204070;
    Leb_Grid_XYZW[ 1445][3] =  0.000540803209207;

    Leb_Grid_XYZW[ 1446][0] =  0.704203249736322;
    Leb_Grid_XYZW[ 1446][1] =  0.377529900204070;
    Leb_Grid_XYZW[ 1446][2] =  0.601306043136695;
    Leb_Grid_XYZW[ 1446][3] =  0.000540803209207;

    Leb_Grid_XYZW[ 1447][0] =  0.704203249736322;
    Leb_Grid_XYZW[ 1447][1] =  0.377529900204070;
    Leb_Grid_XYZW[ 1447][2] = -0.601306043136695;
    Leb_Grid_XYZW[ 1447][3] =  0.000540803209207;

    Leb_Grid_XYZW[ 1448][0] =  0.704203249736322;
    Leb_Grid_XYZW[ 1448][1] = -0.377529900204070;
    Leb_Grid_XYZW[ 1448][2] =  0.601306043136695;
    Leb_Grid_XYZW[ 1448][3] =  0.000540803209207;

    Leb_Grid_XYZW[ 1449][0] =  0.704203249736322;
    Leb_Grid_XYZW[ 1449][1] = -0.377529900204070;
    Leb_Grid_XYZW[ 1449][2] = -0.601306043136695;
    Leb_Grid_XYZW[ 1449][3] =  0.000540803209207;

    Leb_Grid_XYZW[ 1450][0] = -0.704203249736322;
    Leb_Grid_XYZW[ 1450][1] =  0.377529900204070;
    Leb_Grid_XYZW[ 1450][2] =  0.601306043136695;
    Leb_Grid_XYZW[ 1450][3] =  0.000540803209207;

    Leb_Grid_XYZW[ 1451][0] = -0.704203249736322;
    Leb_Grid_XYZW[ 1451][1] =  0.377529900204070;
    Leb_Grid_XYZW[ 1451][2] = -0.601306043136695;
    Leb_Grid_XYZW[ 1451][3] =  0.000540803209207;

    Leb_Grid_XYZW[ 1452][0] = -0.704203249736322;
    Leb_Grid_XYZW[ 1452][1] = -0.377529900204070;
    Leb_Grid_XYZW[ 1452][2] =  0.601306043136695;
    Leb_Grid_XYZW[ 1452][3] =  0.000540803209207;

    Leb_Grid_XYZW[ 1453][0] = -0.704203249736322;
    Leb_Grid_XYZW[ 1453][1] = -0.377529900204070;
    Leb_Grid_XYZW[ 1453][2] = -0.601306043136695;
    Leb_Grid_XYZW[ 1453][3] =  0.000540803209207;

    Leb_Grid_XYZW[ 1454][0] =  0.366159676726178;
    Leb_Grid_XYZW[ 1454][1] =  0.045993678871646;
    Leb_Grid_XYZW[ 1454][2] =  0.929414693580660;
    Leb_Grid_XYZW[ 1454][3] =  0.000484274491790;

    Leb_Grid_XYZW[ 1455][0] =  0.366159676726178;
    Leb_Grid_XYZW[ 1455][1] =  0.045993678871646;
    Leb_Grid_XYZW[ 1455][2] = -0.929414693580660;
    Leb_Grid_XYZW[ 1455][3] =  0.000484274491790;

    Leb_Grid_XYZW[ 1456][0] =  0.366159676726178;
    Leb_Grid_XYZW[ 1456][1] = -0.045993678871646;
    Leb_Grid_XYZW[ 1456][2] =  0.929414693580660;
    Leb_Grid_XYZW[ 1456][3] =  0.000484274491790;

    Leb_Grid_XYZW[ 1457][0] =  0.366159676726178;
    Leb_Grid_XYZW[ 1457][1] = -0.045993678871646;
    Leb_Grid_XYZW[ 1457][2] = -0.929414693580660;
    Leb_Grid_XYZW[ 1457][3] =  0.000484274491790;

    Leb_Grid_XYZW[ 1458][0] = -0.366159676726178;
    Leb_Grid_XYZW[ 1458][1] =  0.045993678871646;
    Leb_Grid_XYZW[ 1458][2] =  0.929414693580660;
    Leb_Grid_XYZW[ 1458][3] =  0.000484274491790;

    Leb_Grid_XYZW[ 1459][0] = -0.366159676726178;
    Leb_Grid_XYZW[ 1459][1] =  0.045993678871646;
    Leb_Grid_XYZW[ 1459][2] = -0.929414693580660;
    Leb_Grid_XYZW[ 1459][3] =  0.000484274491790;

    Leb_Grid_XYZW[ 1460][0] = -0.366159676726178;
    Leb_Grid_XYZW[ 1460][1] = -0.045993678871646;
    Leb_Grid_XYZW[ 1460][2] =  0.929414693580660;
    Leb_Grid_XYZW[ 1460][3] =  0.000484274491790;

    Leb_Grid_XYZW[ 1461][0] = -0.366159676726178;
    Leb_Grid_XYZW[ 1461][1] = -0.045993678871646;
    Leb_Grid_XYZW[ 1461][2] = -0.929414693580660;
    Leb_Grid_XYZW[ 1461][3] =  0.000484274491790;

    Leb_Grid_XYZW[ 1462][0] =  0.366159676726178;
    Leb_Grid_XYZW[ 1462][1] =  0.929414693580660;
    Leb_Grid_XYZW[ 1462][2] =  0.045993678871646;
    Leb_Grid_XYZW[ 1462][3] =  0.000484274491790;

    Leb_Grid_XYZW[ 1463][0] =  0.366159676726178;
    Leb_Grid_XYZW[ 1463][1] =  0.929414693580660;
    Leb_Grid_XYZW[ 1463][2] = -0.045993678871646;
    Leb_Grid_XYZW[ 1463][3] =  0.000484274491790;

    Leb_Grid_XYZW[ 1464][0] =  0.366159676726178;
    Leb_Grid_XYZW[ 1464][1] = -0.929414693580660;
    Leb_Grid_XYZW[ 1464][2] =  0.045993678871646;
    Leb_Grid_XYZW[ 1464][3] =  0.000484274491790;

    Leb_Grid_XYZW[ 1465][0] =  0.366159676726178;
    Leb_Grid_XYZW[ 1465][1] = -0.929414693580660;
    Leb_Grid_XYZW[ 1465][2] = -0.045993678871646;
    Leb_Grid_XYZW[ 1465][3] =  0.000484274491790;

    Leb_Grid_XYZW[ 1466][0] = -0.366159676726178;
    Leb_Grid_XYZW[ 1466][1] =  0.929414693580660;
    Leb_Grid_XYZW[ 1466][2] =  0.045993678871646;
    Leb_Grid_XYZW[ 1466][3] =  0.000484274491790;

    Leb_Grid_XYZW[ 1467][0] = -0.366159676726178;
    Leb_Grid_XYZW[ 1467][1] =  0.929414693580660;
    Leb_Grid_XYZW[ 1467][2] = -0.045993678871646;
    Leb_Grid_XYZW[ 1467][3] =  0.000484274491790;

    Leb_Grid_XYZW[ 1468][0] = -0.366159676726178;
    Leb_Grid_XYZW[ 1468][1] = -0.929414693580660;
    Leb_Grid_XYZW[ 1468][2] =  0.045993678871646;
    Leb_Grid_XYZW[ 1468][3] =  0.000484274491790;

    Leb_Grid_XYZW[ 1469][0] = -0.366159676726178;
    Leb_Grid_XYZW[ 1469][1] = -0.929414693580660;
    Leb_Grid_XYZW[ 1469][2] = -0.045993678871646;
    Leb_Grid_XYZW[ 1469][3] =  0.000484274491790;

    Leb_Grid_XYZW[ 1470][0] =  0.045993678871646;
    Leb_Grid_XYZW[ 1470][1] =  0.366159676726178;
    Leb_Grid_XYZW[ 1470][2] =  0.929414693580660;
    Leb_Grid_XYZW[ 1470][3] =  0.000484274491790;

    Leb_Grid_XYZW[ 1471][0] =  0.045993678871646;
    Leb_Grid_XYZW[ 1471][1] =  0.366159676726178;
    Leb_Grid_XYZW[ 1471][2] = -0.929414693580660;
    Leb_Grid_XYZW[ 1471][3] =  0.000484274491790;

    Leb_Grid_XYZW[ 1472][0] =  0.045993678871646;
    Leb_Grid_XYZW[ 1472][1] = -0.366159676726178;
    Leb_Grid_XYZW[ 1472][2] =  0.929414693580660;
    Leb_Grid_XYZW[ 1472][3] =  0.000484274491790;

    Leb_Grid_XYZW[ 1473][0] =  0.045993678871646;
    Leb_Grid_XYZW[ 1473][1] = -0.366159676726178;
    Leb_Grid_XYZW[ 1473][2] = -0.929414693580660;
    Leb_Grid_XYZW[ 1473][3] =  0.000484274491790;

    Leb_Grid_XYZW[ 1474][0] = -0.045993678871646;
    Leb_Grid_XYZW[ 1474][1] =  0.366159676726178;
    Leb_Grid_XYZW[ 1474][2] =  0.929414693580660;
    Leb_Grid_XYZW[ 1474][3] =  0.000484274491790;

    Leb_Grid_XYZW[ 1475][0] = -0.045993678871646;
    Leb_Grid_XYZW[ 1475][1] =  0.366159676726178;
    Leb_Grid_XYZW[ 1475][2] = -0.929414693580660;
    Leb_Grid_XYZW[ 1475][3] =  0.000484274491790;

    Leb_Grid_XYZW[ 1476][0] = -0.045993678871646;
    Leb_Grid_XYZW[ 1476][1] = -0.366159676726178;
    Leb_Grid_XYZW[ 1476][2] =  0.929414693580660;
    Leb_Grid_XYZW[ 1476][3] =  0.000484274491790;

    Leb_Grid_XYZW[ 1477][0] = -0.045993678871646;
    Leb_Grid_XYZW[ 1477][1] = -0.366159676726178;
    Leb_Grid_XYZW[ 1477][2] = -0.929414693580660;
    Leb_Grid_XYZW[ 1477][3] =  0.000484274491790;

    Leb_Grid_XYZW[ 1478][0] =  0.045993678871646;
    Leb_Grid_XYZW[ 1478][1] =  0.929414693580660;
    Leb_Grid_XYZW[ 1478][2] =  0.366159676726178;
    Leb_Grid_XYZW[ 1478][3] =  0.000484274491790;

    Leb_Grid_XYZW[ 1479][0] =  0.045993678871646;
    Leb_Grid_XYZW[ 1479][1] =  0.929414693580660;
    Leb_Grid_XYZW[ 1479][2] = -0.366159676726178;
    Leb_Grid_XYZW[ 1479][3] =  0.000484274491790;

    Leb_Grid_XYZW[ 1480][0] =  0.045993678871646;
    Leb_Grid_XYZW[ 1480][1] = -0.929414693580660;
    Leb_Grid_XYZW[ 1480][2] =  0.366159676726178;
    Leb_Grid_XYZW[ 1480][3] =  0.000484274491790;

    Leb_Grid_XYZW[ 1481][0] =  0.045993678871646;
    Leb_Grid_XYZW[ 1481][1] = -0.929414693580660;
    Leb_Grid_XYZW[ 1481][2] = -0.366159676726178;
    Leb_Grid_XYZW[ 1481][3] =  0.000484274491790;

    Leb_Grid_XYZW[ 1482][0] = -0.045993678871646;
    Leb_Grid_XYZW[ 1482][1] =  0.929414693580660;
    Leb_Grid_XYZW[ 1482][2] =  0.366159676726178;
    Leb_Grid_XYZW[ 1482][3] =  0.000484274491790;

    Leb_Grid_XYZW[ 1483][0] = -0.045993678871646;
    Leb_Grid_XYZW[ 1483][1] =  0.929414693580660;
    Leb_Grid_XYZW[ 1483][2] = -0.366159676726178;
    Leb_Grid_XYZW[ 1483][3] =  0.000484274491790;

    Leb_Grid_XYZW[ 1484][0] = -0.045993678871646;
    Leb_Grid_XYZW[ 1484][1] = -0.929414693580660;
    Leb_Grid_XYZW[ 1484][2] =  0.366159676726178;
    Leb_Grid_XYZW[ 1484][3] =  0.000484274491790;

    Leb_Grid_XYZW[ 1485][0] = -0.045993678871646;
    Leb_Grid_XYZW[ 1485][1] = -0.929414693580660;
    Leb_Grid_XYZW[ 1485][2] = -0.366159676726178;
    Leb_Grid_XYZW[ 1485][3] =  0.000484274491790;

    Leb_Grid_XYZW[ 1486][0] =  0.929414693580660;
    Leb_Grid_XYZW[ 1486][1] =  0.366159676726178;
    Leb_Grid_XYZW[ 1486][2] =  0.045993678871646;
    Leb_Grid_XYZW[ 1486][3] =  0.000484274491790;

    Leb_Grid_XYZW[ 1487][0] =  0.929414693580660;
    Leb_Grid_XYZW[ 1487][1] =  0.366159676726178;
    Leb_Grid_XYZW[ 1487][2] = -0.045993678871646;
    Leb_Grid_XYZW[ 1487][3] =  0.000484274491790;

    Leb_Grid_XYZW[ 1488][0] =  0.929414693580660;
    Leb_Grid_XYZW[ 1488][1] = -0.366159676726178;
    Leb_Grid_XYZW[ 1488][2] =  0.045993678871646;
    Leb_Grid_XYZW[ 1488][3] =  0.000484274491790;

    Leb_Grid_XYZW[ 1489][0] =  0.929414693580660;
    Leb_Grid_XYZW[ 1489][1] = -0.366159676726178;
    Leb_Grid_XYZW[ 1489][2] = -0.045993678871646;
    Leb_Grid_XYZW[ 1489][3] =  0.000484274491790;

    Leb_Grid_XYZW[ 1490][0] = -0.929414693580660;
    Leb_Grid_XYZW[ 1490][1] =  0.366159676726178;
    Leb_Grid_XYZW[ 1490][2] =  0.045993678871646;
    Leb_Grid_XYZW[ 1490][3] =  0.000484274491790;

    Leb_Grid_XYZW[ 1491][0] = -0.929414693580660;
    Leb_Grid_XYZW[ 1491][1] =  0.366159676726178;
    Leb_Grid_XYZW[ 1491][2] = -0.045993678871646;
    Leb_Grid_XYZW[ 1491][3] =  0.000484274491790;

    Leb_Grid_XYZW[ 1492][0] = -0.929414693580660;
    Leb_Grid_XYZW[ 1492][1] = -0.366159676726178;
    Leb_Grid_XYZW[ 1492][2] =  0.045993678871646;
    Leb_Grid_XYZW[ 1492][3] =  0.000484274491790;

    Leb_Grid_XYZW[ 1493][0] = -0.929414693580660;
    Leb_Grid_XYZW[ 1493][1] = -0.366159676726178;
    Leb_Grid_XYZW[ 1493][2] = -0.045993678871646;
    Leb_Grid_XYZW[ 1493][3] =  0.000484274491790;

    Leb_Grid_XYZW[ 1494][0] =  0.929414693580660;
    Leb_Grid_XYZW[ 1494][1] =  0.045993678871648;
    Leb_Grid_XYZW[ 1494][2] =  0.366159676726178;
    Leb_Grid_XYZW[ 1494][3] =  0.000484274491790;

    Leb_Grid_XYZW[ 1495][0] =  0.929414693580660;
    Leb_Grid_XYZW[ 1495][1] =  0.045993678871648;
    Leb_Grid_XYZW[ 1495][2] = -0.366159676726178;
    Leb_Grid_XYZW[ 1495][3] =  0.000484274491790;

    Leb_Grid_XYZW[ 1496][0] =  0.929414693580660;
    Leb_Grid_XYZW[ 1496][1] = -0.045993678871648;
    Leb_Grid_XYZW[ 1496][2] =  0.366159676726178;
    Leb_Grid_XYZW[ 1496][3] =  0.000484274491790;

    Leb_Grid_XYZW[ 1497][0] =  0.929414693580660;
    Leb_Grid_XYZW[ 1497][1] = -0.045993678871648;
    Leb_Grid_XYZW[ 1497][2] = -0.366159676726178;
    Leb_Grid_XYZW[ 1497][3] =  0.000484274491790;

    Leb_Grid_XYZW[ 1498][0] = -0.929414693580660;
    Leb_Grid_XYZW[ 1498][1] =  0.045993678871648;
    Leb_Grid_XYZW[ 1498][2] =  0.366159676726178;
    Leb_Grid_XYZW[ 1498][3] =  0.000484274491790;

    Leb_Grid_XYZW[ 1499][0] = -0.929414693580660;
    Leb_Grid_XYZW[ 1499][1] =  0.045993678871648;
    Leb_Grid_XYZW[ 1499][2] = -0.366159676726178;
    Leb_Grid_XYZW[ 1499][3] =  0.000484274491790;

    Leb_Grid_XYZW[ 1500][0] = -0.929414693580660;
    Leb_Grid_XYZW[ 1500][1] = -0.045993678871648;
    Leb_Grid_XYZW[ 1500][2] =  0.366159676726178;
    Leb_Grid_XYZW[ 1500][3] =  0.000484274491790;

    Leb_Grid_XYZW[ 1501][0] = -0.929414693580660;
    Leb_Grid_XYZW[ 1501][1] = -0.045993678871648;
    Leb_Grid_XYZW[ 1501][2] = -0.366159676726178;
    Leb_Grid_XYZW[ 1501][3] =  0.000484274491790;

    Leb_Grid_XYZW[ 1502][0] =  0.423763315350658;
    Leb_Grid_XYZW[ 1502][1] =  0.094048937736544;
    Leb_Grid_XYZW[ 1502][2] =  0.900877044814467;
    Leb_Grid_XYZW[ 1502][3] =  0.000504892607619;

    Leb_Grid_XYZW[ 1503][0] =  0.423763315350658;
    Leb_Grid_XYZW[ 1503][1] =  0.094048937736544;
    Leb_Grid_XYZW[ 1503][2] = -0.900877044814467;
    Leb_Grid_XYZW[ 1503][3] =  0.000504892607619;

    Leb_Grid_XYZW[ 1504][0] =  0.423763315350658;
    Leb_Grid_XYZW[ 1504][1] = -0.094048937736544;
    Leb_Grid_XYZW[ 1504][2] =  0.900877044814467;
    Leb_Grid_XYZW[ 1504][3] =  0.000504892607619;

    Leb_Grid_XYZW[ 1505][0] =  0.423763315350658;
    Leb_Grid_XYZW[ 1505][1] = -0.094048937736544;
    Leb_Grid_XYZW[ 1505][2] = -0.900877044814467;
    Leb_Grid_XYZW[ 1505][3] =  0.000504892607619;

    Leb_Grid_XYZW[ 1506][0] = -0.423763315350658;
    Leb_Grid_XYZW[ 1506][1] =  0.094048937736544;
    Leb_Grid_XYZW[ 1506][2] =  0.900877044814467;
    Leb_Grid_XYZW[ 1506][3] =  0.000504892607619;

    Leb_Grid_XYZW[ 1507][0] = -0.423763315350658;
    Leb_Grid_XYZW[ 1507][1] =  0.094048937736544;
    Leb_Grid_XYZW[ 1507][2] = -0.900877044814467;
    Leb_Grid_XYZW[ 1507][3] =  0.000504892607619;

    Leb_Grid_XYZW[ 1508][0] = -0.423763315350658;
    Leb_Grid_XYZW[ 1508][1] = -0.094048937736544;
    Leb_Grid_XYZW[ 1508][2] =  0.900877044814467;
    Leb_Grid_XYZW[ 1508][3] =  0.000504892607619;

    Leb_Grid_XYZW[ 1509][0] = -0.423763315350658;
    Leb_Grid_XYZW[ 1509][1] = -0.094048937736544;
    Leb_Grid_XYZW[ 1509][2] = -0.900877044814467;
    Leb_Grid_XYZW[ 1509][3] =  0.000504892607619;

    Leb_Grid_XYZW[ 1510][0] =  0.423763315350658;
    Leb_Grid_XYZW[ 1510][1] =  0.900877044814467;
    Leb_Grid_XYZW[ 1510][2] =  0.094048937736544;
    Leb_Grid_XYZW[ 1510][3] =  0.000504892607619;

    Leb_Grid_XYZW[ 1511][0] =  0.423763315350658;
    Leb_Grid_XYZW[ 1511][1] =  0.900877044814467;
    Leb_Grid_XYZW[ 1511][2] = -0.094048937736544;
    Leb_Grid_XYZW[ 1511][3] =  0.000504892607619;

    Leb_Grid_XYZW[ 1512][0] =  0.423763315350658;
    Leb_Grid_XYZW[ 1512][1] = -0.900877044814467;
    Leb_Grid_XYZW[ 1512][2] =  0.094048937736544;
    Leb_Grid_XYZW[ 1512][3] =  0.000504892607619;

    Leb_Grid_XYZW[ 1513][0] =  0.423763315350658;
    Leb_Grid_XYZW[ 1513][1] = -0.900877044814467;
    Leb_Grid_XYZW[ 1513][2] = -0.094048937736544;
    Leb_Grid_XYZW[ 1513][3] =  0.000504892607619;

    Leb_Grid_XYZW[ 1514][0] = -0.423763315350658;
    Leb_Grid_XYZW[ 1514][1] =  0.900877044814467;
    Leb_Grid_XYZW[ 1514][2] =  0.094048937736544;
    Leb_Grid_XYZW[ 1514][3] =  0.000504892607619;

    Leb_Grid_XYZW[ 1515][0] = -0.423763315350658;
    Leb_Grid_XYZW[ 1515][1] =  0.900877044814467;
    Leb_Grid_XYZW[ 1515][2] = -0.094048937736544;
    Leb_Grid_XYZW[ 1515][3] =  0.000504892607619;

    Leb_Grid_XYZW[ 1516][0] = -0.423763315350658;
    Leb_Grid_XYZW[ 1516][1] = -0.900877044814467;
    Leb_Grid_XYZW[ 1516][2] =  0.094048937736544;
    Leb_Grid_XYZW[ 1516][3] =  0.000504892607619;

    Leb_Grid_XYZW[ 1517][0] = -0.423763315350658;
    Leb_Grid_XYZW[ 1517][1] = -0.900877044814467;
    Leb_Grid_XYZW[ 1517][2] = -0.094048937736544;
    Leb_Grid_XYZW[ 1517][3] =  0.000504892607619;

    Leb_Grid_XYZW[ 1518][0] =  0.094048937736544;
    Leb_Grid_XYZW[ 1518][1] =  0.423763315350658;
    Leb_Grid_XYZW[ 1518][2] =  0.900877044814467;
    Leb_Grid_XYZW[ 1518][3] =  0.000504892607619;

    Leb_Grid_XYZW[ 1519][0] =  0.094048937736544;
    Leb_Grid_XYZW[ 1519][1] =  0.423763315350658;
    Leb_Grid_XYZW[ 1519][2] = -0.900877044814467;
    Leb_Grid_XYZW[ 1519][3] =  0.000504892607619;

    Leb_Grid_XYZW[ 1520][0] =  0.094048937736544;
    Leb_Grid_XYZW[ 1520][1] = -0.423763315350658;
    Leb_Grid_XYZW[ 1520][2] =  0.900877044814467;
    Leb_Grid_XYZW[ 1520][3] =  0.000504892607619;

    Leb_Grid_XYZW[ 1521][0] =  0.094048937736544;
    Leb_Grid_XYZW[ 1521][1] = -0.423763315350658;
    Leb_Grid_XYZW[ 1521][2] = -0.900877044814467;
    Leb_Grid_XYZW[ 1521][3] =  0.000504892607619;

    Leb_Grid_XYZW[ 1522][0] = -0.094048937736544;
    Leb_Grid_XYZW[ 1522][1] =  0.423763315350658;
    Leb_Grid_XYZW[ 1522][2] =  0.900877044814467;
    Leb_Grid_XYZW[ 1522][3] =  0.000504892607619;

    Leb_Grid_XYZW[ 1523][0] = -0.094048937736544;
    Leb_Grid_XYZW[ 1523][1] =  0.423763315350658;
    Leb_Grid_XYZW[ 1523][2] = -0.900877044814467;
    Leb_Grid_XYZW[ 1523][3] =  0.000504892607619;

    Leb_Grid_XYZW[ 1524][0] = -0.094048937736544;
    Leb_Grid_XYZW[ 1524][1] = -0.423763315350658;
    Leb_Grid_XYZW[ 1524][2] =  0.900877044814467;
    Leb_Grid_XYZW[ 1524][3] =  0.000504892607619;

    Leb_Grid_XYZW[ 1525][0] = -0.094048937736544;
    Leb_Grid_XYZW[ 1525][1] = -0.423763315350658;
    Leb_Grid_XYZW[ 1525][2] = -0.900877044814467;
    Leb_Grid_XYZW[ 1525][3] =  0.000504892607619;

    Leb_Grid_XYZW[ 1526][0] =  0.094048937736544;
    Leb_Grid_XYZW[ 1526][1] =  0.900877044814466;
    Leb_Grid_XYZW[ 1526][2] =  0.423763315350658;
    Leb_Grid_XYZW[ 1526][3] =  0.000504892607619;

    Leb_Grid_XYZW[ 1527][0] =  0.094048937736544;
    Leb_Grid_XYZW[ 1527][1] =  0.900877044814467;
    Leb_Grid_XYZW[ 1527][2] = -0.423763315350658;
    Leb_Grid_XYZW[ 1527][3] =  0.000504892607619;

    Leb_Grid_XYZW[ 1528][0] =  0.094048937736544;
    Leb_Grid_XYZW[ 1528][1] = -0.900877044814466;
    Leb_Grid_XYZW[ 1528][2] =  0.423763315350658;
    Leb_Grid_XYZW[ 1528][3] =  0.000504892607619;

    Leb_Grid_XYZW[ 1529][0] =  0.094048937736544;
    Leb_Grid_XYZW[ 1529][1] = -0.900877044814467;
    Leb_Grid_XYZW[ 1529][2] = -0.423763315350658;
    Leb_Grid_XYZW[ 1529][3] =  0.000504892607619;

    Leb_Grid_XYZW[ 1530][0] = -0.094048937736544;
    Leb_Grid_XYZW[ 1530][1] =  0.900877044814466;
    Leb_Grid_XYZW[ 1530][2] =  0.423763315350658;
    Leb_Grid_XYZW[ 1530][3] =  0.000504892607619;

    Leb_Grid_XYZW[ 1531][0] = -0.094048937736544;
    Leb_Grid_XYZW[ 1531][1] =  0.900877044814467;
    Leb_Grid_XYZW[ 1531][2] = -0.423763315350658;
    Leb_Grid_XYZW[ 1531][3] =  0.000504892607619;

    Leb_Grid_XYZW[ 1532][0] = -0.094048937736544;
    Leb_Grid_XYZW[ 1532][1] = -0.900877044814466;
    Leb_Grid_XYZW[ 1532][2] =  0.423763315350658;
    Leb_Grid_XYZW[ 1532][3] =  0.000504892607619;

    Leb_Grid_XYZW[ 1533][0] = -0.094048937736544;
    Leb_Grid_XYZW[ 1533][1] = -0.900877044814467;
    Leb_Grid_XYZW[ 1533][2] = -0.423763315350658;
    Leb_Grid_XYZW[ 1533][3] =  0.000504892607619;

    Leb_Grid_XYZW[ 1534][0] =  0.900877044814467;
    Leb_Grid_XYZW[ 1534][1] =  0.423763315350658;
    Leb_Grid_XYZW[ 1534][2] =  0.094048937736544;
    Leb_Grid_XYZW[ 1534][3] =  0.000504892607619;

    Leb_Grid_XYZW[ 1535][0] =  0.900877044814467;
    Leb_Grid_XYZW[ 1535][1] =  0.423763315350658;
    Leb_Grid_XYZW[ 1535][2] = -0.094048937736544;
    Leb_Grid_XYZW[ 1535][3] =  0.000504892607619;

    Leb_Grid_XYZW[ 1536][0] =  0.900877044814467;
    Leb_Grid_XYZW[ 1536][1] = -0.423763315350658;
    Leb_Grid_XYZW[ 1536][2] =  0.094048937736544;
    Leb_Grid_XYZW[ 1536][3] =  0.000504892607619;

    Leb_Grid_XYZW[ 1537][0] =  0.900877044814467;
    Leb_Grid_XYZW[ 1537][1] = -0.423763315350658;
    Leb_Grid_XYZW[ 1537][2] = -0.094048937736544;
    Leb_Grid_XYZW[ 1537][3] =  0.000504892607619;

    Leb_Grid_XYZW[ 1538][0] = -0.900877044814467;
    Leb_Grid_XYZW[ 1538][1] =  0.423763315350658;
    Leb_Grid_XYZW[ 1538][2] =  0.094048937736544;
    Leb_Grid_XYZW[ 1538][3] =  0.000504892607619;

    Leb_Grid_XYZW[ 1539][0] = -0.900877044814467;
    Leb_Grid_XYZW[ 1539][1] =  0.423763315350658;
    Leb_Grid_XYZW[ 1539][2] = -0.094048937736544;
    Leb_Grid_XYZW[ 1539][3] =  0.000504892607619;

    Leb_Grid_XYZW[ 1540][0] = -0.900877044814467;
    Leb_Grid_XYZW[ 1540][1] = -0.423763315350658;
    Leb_Grid_XYZW[ 1540][2] =  0.094048937736544;
    Leb_Grid_XYZW[ 1540][3] =  0.000504892607619;

    Leb_Grid_XYZW[ 1541][0] = -0.900877044814467;
    Leb_Grid_XYZW[ 1541][1] = -0.423763315350658;
    Leb_Grid_XYZW[ 1541][2] = -0.094048937736544;
    Leb_Grid_XYZW[ 1541][3] =  0.000504892607619;

    Leb_Grid_XYZW[ 1542][0] =  0.900877044814467;
    Leb_Grid_XYZW[ 1542][1] =  0.094048937736544;
    Leb_Grid_XYZW[ 1542][2] =  0.423763315350658;
    Leb_Grid_XYZW[ 1542][3] =  0.000504892607619;

    Leb_Grid_XYZW[ 1543][0] =  0.900877044814467;
    Leb_Grid_XYZW[ 1543][1] =  0.094048937736544;
    Leb_Grid_XYZW[ 1543][2] = -0.423763315350658;
    Leb_Grid_XYZW[ 1543][3] =  0.000504892607619;

    Leb_Grid_XYZW[ 1544][0] =  0.900877044814467;
    Leb_Grid_XYZW[ 1544][1] = -0.094048937736544;
    Leb_Grid_XYZW[ 1544][2] =  0.423763315350658;
    Leb_Grid_XYZW[ 1544][3] =  0.000504892607619;

    Leb_Grid_XYZW[ 1545][0] =  0.900877044814467;
    Leb_Grid_XYZW[ 1545][1] = -0.094048937736544;
    Leb_Grid_XYZW[ 1545][2] = -0.423763315350658;
    Leb_Grid_XYZW[ 1545][3] =  0.000504892607619;

    Leb_Grid_XYZW[ 1546][0] = -0.900877044814467;
    Leb_Grid_XYZW[ 1546][1] =  0.094048937736544;
    Leb_Grid_XYZW[ 1546][2] =  0.423763315350658;
    Leb_Grid_XYZW[ 1546][3] =  0.000504892607619;

    Leb_Grid_XYZW[ 1547][0] = -0.900877044814467;
    Leb_Grid_XYZW[ 1547][1] =  0.094048937736544;
    Leb_Grid_XYZW[ 1547][2] = -0.423763315350658;
    Leb_Grid_XYZW[ 1547][3] =  0.000504892607619;

    Leb_Grid_XYZW[ 1548][0] = -0.900877044814467;
    Leb_Grid_XYZW[ 1548][1] = -0.094048937736544;
    Leb_Grid_XYZW[ 1548][2] =  0.423763315350658;
    Leb_Grid_XYZW[ 1548][3] =  0.000504892607619;

    Leb_Grid_XYZW[ 1549][0] = -0.900877044814467;
    Leb_Grid_XYZW[ 1549][1] = -0.094048937736544;
    Leb_Grid_XYZW[ 1549][2] = -0.423763315350658;
    Leb_Grid_XYZW[ 1549][3] =  0.000504892607619;

    Leb_Grid_XYZW[ 1550][0] =  0.478632845465845;
    Leb_Grid_XYZW[ 1550][1] =  0.143137710909197;
    Leb_Grid_XYZW[ 1550][2] =  0.866269123862177;
    Leb_Grid_XYZW[ 1550][3] =  0.000520260798048;

    Leb_Grid_XYZW[ 1551][0] =  0.478632845465845;
    Leb_Grid_XYZW[ 1551][1] =  0.143137710909197;
    Leb_Grid_XYZW[ 1551][2] = -0.866269123862177;
    Leb_Grid_XYZW[ 1551][3] =  0.000520260798048;

    Leb_Grid_XYZW[ 1552][0] =  0.478632845465845;
    Leb_Grid_XYZW[ 1552][1] = -0.143137710909197;
    Leb_Grid_XYZW[ 1552][2] =  0.866269123862177;
    Leb_Grid_XYZW[ 1552][3] =  0.000520260798048;

    Leb_Grid_XYZW[ 1553][0] =  0.478632845465845;
    Leb_Grid_XYZW[ 1553][1] = -0.143137710909197;
    Leb_Grid_XYZW[ 1553][2] = -0.866269123862177;
    Leb_Grid_XYZW[ 1553][3] =  0.000520260798048;

    Leb_Grid_XYZW[ 1554][0] = -0.478632845465845;
    Leb_Grid_XYZW[ 1554][1] =  0.143137710909197;
    Leb_Grid_XYZW[ 1554][2] =  0.866269123862177;
    Leb_Grid_XYZW[ 1554][3] =  0.000520260798048;

    Leb_Grid_XYZW[ 1555][0] = -0.478632845465845;
    Leb_Grid_XYZW[ 1555][1] =  0.143137710909197;
    Leb_Grid_XYZW[ 1555][2] = -0.866269123862177;
    Leb_Grid_XYZW[ 1555][3] =  0.000520260798048;

    Leb_Grid_XYZW[ 1556][0] = -0.478632845465845;
    Leb_Grid_XYZW[ 1556][1] = -0.143137710909197;
    Leb_Grid_XYZW[ 1556][2] =  0.866269123862177;
    Leb_Grid_XYZW[ 1556][3] =  0.000520260798048;

    Leb_Grid_XYZW[ 1557][0] = -0.478632845465845;
    Leb_Grid_XYZW[ 1557][1] = -0.143137710909197;
    Leb_Grid_XYZW[ 1557][2] = -0.866269123862177;
    Leb_Grid_XYZW[ 1557][3] =  0.000520260798048;

    Leb_Grid_XYZW[ 1558][0] =  0.478632845465845;
    Leb_Grid_XYZW[ 1558][1] =  0.866269123862177;
    Leb_Grid_XYZW[ 1558][2] =  0.143137710909197;
    Leb_Grid_XYZW[ 1558][3] =  0.000520260798048;

    Leb_Grid_XYZW[ 1559][0] =  0.478632845465845;
    Leb_Grid_XYZW[ 1559][1] =  0.866269123862177;
    Leb_Grid_XYZW[ 1559][2] = -0.143137710909197;
    Leb_Grid_XYZW[ 1559][3] =  0.000520260798048;

    Leb_Grid_XYZW[ 1560][0] =  0.478632845465845;
    Leb_Grid_XYZW[ 1560][1] = -0.866269123862177;
    Leb_Grid_XYZW[ 1560][2] =  0.143137710909197;
    Leb_Grid_XYZW[ 1560][3] =  0.000520260798048;

    Leb_Grid_XYZW[ 1561][0] =  0.478632845465845;
    Leb_Grid_XYZW[ 1561][1] = -0.866269123862177;
    Leb_Grid_XYZW[ 1561][2] = -0.143137710909197;
    Leb_Grid_XYZW[ 1561][3] =  0.000520260798048;

    Leb_Grid_XYZW[ 1562][0] = -0.478632845465845;
    Leb_Grid_XYZW[ 1562][1] =  0.866269123862177;
    Leb_Grid_XYZW[ 1562][2] =  0.143137710909197;
    Leb_Grid_XYZW[ 1562][3] =  0.000520260798048;

    Leb_Grid_XYZW[ 1563][0] = -0.478632845465845;
    Leb_Grid_XYZW[ 1563][1] =  0.866269123862177;
    Leb_Grid_XYZW[ 1563][2] = -0.143137710909197;
    Leb_Grid_XYZW[ 1563][3] =  0.000520260798048;

    Leb_Grid_XYZW[ 1564][0] = -0.478632845465845;
    Leb_Grid_XYZW[ 1564][1] = -0.866269123862177;
    Leb_Grid_XYZW[ 1564][2] =  0.143137710909197;
    Leb_Grid_XYZW[ 1564][3] =  0.000520260798048;

    Leb_Grid_XYZW[ 1565][0] = -0.478632845465845;
    Leb_Grid_XYZW[ 1565][1] = -0.866269123862177;
    Leb_Grid_XYZW[ 1565][2] = -0.143137710909197;
    Leb_Grid_XYZW[ 1565][3] =  0.000520260798048;

    Leb_Grid_XYZW[ 1566][0] =  0.143137710909197;
    Leb_Grid_XYZW[ 1566][1] =  0.478632845465845;
    Leb_Grid_XYZW[ 1566][2] =  0.866269123862177;
    Leb_Grid_XYZW[ 1566][3] =  0.000520260798048;

    Leb_Grid_XYZW[ 1567][0] =  0.143137710909197;
    Leb_Grid_XYZW[ 1567][1] =  0.478632845465845;
    Leb_Grid_XYZW[ 1567][2] = -0.866269123862177;
    Leb_Grid_XYZW[ 1567][3] =  0.000520260798048;

    Leb_Grid_XYZW[ 1568][0] =  0.143137710909197;
    Leb_Grid_XYZW[ 1568][1] = -0.478632845465845;
    Leb_Grid_XYZW[ 1568][2] =  0.866269123862177;
    Leb_Grid_XYZW[ 1568][3] =  0.000520260798048;

    Leb_Grid_XYZW[ 1569][0] =  0.143137710909197;
    Leb_Grid_XYZW[ 1569][1] = -0.478632845465845;
    Leb_Grid_XYZW[ 1569][2] = -0.866269123862177;
    Leb_Grid_XYZW[ 1569][3] =  0.000520260798048;

    Leb_Grid_XYZW[ 1570][0] = -0.143137710909197;
    Leb_Grid_XYZW[ 1570][1] =  0.478632845465845;
    Leb_Grid_XYZW[ 1570][2] =  0.866269123862177;
    Leb_Grid_XYZW[ 1570][3] =  0.000520260798048;

    Leb_Grid_XYZW[ 1571][0] = -0.143137710909197;
    Leb_Grid_XYZW[ 1571][1] =  0.478632845465845;
    Leb_Grid_XYZW[ 1571][2] = -0.866269123862177;
    Leb_Grid_XYZW[ 1571][3] =  0.000520260798048;

    Leb_Grid_XYZW[ 1572][0] = -0.143137710909197;
    Leb_Grid_XYZW[ 1572][1] = -0.478632845465845;
    Leb_Grid_XYZW[ 1572][2] =  0.866269123862177;
    Leb_Grid_XYZW[ 1572][3] =  0.000520260798048;

    Leb_Grid_XYZW[ 1573][0] = -0.143137710909197;
    Leb_Grid_XYZW[ 1573][1] = -0.478632845465845;
    Leb_Grid_XYZW[ 1573][2] = -0.866269123862177;
    Leb_Grid_XYZW[ 1573][3] =  0.000520260798048;

    Leb_Grid_XYZW[ 1574][0] =  0.143137710909197;
    Leb_Grid_XYZW[ 1574][1] =  0.866269123862177;
    Leb_Grid_XYZW[ 1574][2] =  0.478632845465845;
    Leb_Grid_XYZW[ 1574][3] =  0.000520260798048;

    Leb_Grid_XYZW[ 1575][0] =  0.143137710909197;
    Leb_Grid_XYZW[ 1575][1] =  0.866269123862177;
    Leb_Grid_XYZW[ 1575][2] = -0.478632845465845;
    Leb_Grid_XYZW[ 1575][3] =  0.000520260798048;

    Leb_Grid_XYZW[ 1576][0] =  0.143137710909197;
    Leb_Grid_XYZW[ 1576][1] = -0.866269123862177;
    Leb_Grid_XYZW[ 1576][2] =  0.478632845465845;
    Leb_Grid_XYZW[ 1576][3] =  0.000520260798048;

    Leb_Grid_XYZW[ 1577][0] =  0.143137710909197;
    Leb_Grid_XYZW[ 1577][1] = -0.866269123862177;
    Leb_Grid_XYZW[ 1577][2] = -0.478632845465845;
    Leb_Grid_XYZW[ 1577][3] =  0.000520260798048;

    Leb_Grid_XYZW[ 1578][0] = -0.143137710909197;
    Leb_Grid_XYZW[ 1578][1] =  0.866269123862177;
    Leb_Grid_XYZW[ 1578][2] =  0.478632845465845;
    Leb_Grid_XYZW[ 1578][3] =  0.000520260798048;

    Leb_Grid_XYZW[ 1579][0] = -0.143137710909197;
    Leb_Grid_XYZW[ 1579][1] =  0.866269123862177;
    Leb_Grid_XYZW[ 1579][2] = -0.478632845465845;
    Leb_Grid_XYZW[ 1579][3] =  0.000520260798048;

    Leb_Grid_XYZW[ 1580][0] = -0.143137710909197;
    Leb_Grid_XYZW[ 1580][1] = -0.866269123862177;
    Leb_Grid_XYZW[ 1580][2] =  0.478632845465845;
    Leb_Grid_XYZW[ 1580][3] =  0.000520260798048;

    Leb_Grid_XYZW[ 1581][0] = -0.143137710909197;
    Leb_Grid_XYZW[ 1581][1] = -0.866269123862177;
    Leb_Grid_XYZW[ 1581][2] = -0.478632845465845;
    Leb_Grid_XYZW[ 1581][3] =  0.000520260798048;

    Leb_Grid_XYZW[ 1582][0] =  0.866269123862177;
    Leb_Grid_XYZW[ 1582][1] =  0.478632845465845;
    Leb_Grid_XYZW[ 1582][2] =  0.143137710909197;
    Leb_Grid_XYZW[ 1582][3] =  0.000520260798048;

    Leb_Grid_XYZW[ 1583][0] =  0.866269123862177;
    Leb_Grid_XYZW[ 1583][1] =  0.478632845465845;
    Leb_Grid_XYZW[ 1583][2] = -0.143137710909197;
    Leb_Grid_XYZW[ 1583][3] =  0.000520260798048;

    Leb_Grid_XYZW[ 1584][0] =  0.866269123862177;
    Leb_Grid_XYZW[ 1584][1] = -0.478632845465845;
    Leb_Grid_XYZW[ 1584][2] =  0.143137710909197;
    Leb_Grid_XYZW[ 1584][3] =  0.000520260798048;

    Leb_Grid_XYZW[ 1585][0] =  0.866269123862177;
    Leb_Grid_XYZW[ 1585][1] = -0.478632845465845;
    Leb_Grid_XYZW[ 1585][2] = -0.143137710909197;
    Leb_Grid_XYZW[ 1585][3] =  0.000520260798048;

    Leb_Grid_XYZW[ 1586][0] = -0.866269123862177;
    Leb_Grid_XYZW[ 1586][1] =  0.478632845465846;
    Leb_Grid_XYZW[ 1586][2] =  0.143137710909197;
    Leb_Grid_XYZW[ 1586][3] =  0.000520260798048;

    Leb_Grid_XYZW[ 1587][0] = -0.866269123862177;
    Leb_Grid_XYZW[ 1587][1] =  0.478632845465846;
    Leb_Grid_XYZW[ 1587][2] = -0.143137710909197;
    Leb_Grid_XYZW[ 1587][3] =  0.000520260798048;

    Leb_Grid_XYZW[ 1588][0] = -0.866269123862177;
    Leb_Grid_XYZW[ 1588][1] = -0.478632845465846;
    Leb_Grid_XYZW[ 1588][2] =  0.143137710909197;
    Leb_Grid_XYZW[ 1588][3] =  0.000520260798048;

    Leb_Grid_XYZW[ 1589][0] = -0.866269123862177;
    Leb_Grid_XYZW[ 1589][1] = -0.478632845465846;
    Leb_Grid_XYZW[ 1589][2] = -0.143137710909197;
    Leb_Grid_XYZW[ 1589][3] =  0.000520260798048;

    Leb_Grid_XYZW[ 1590][0] =  0.866269123862177;
    Leb_Grid_XYZW[ 1590][1] =  0.143137710909197;
    Leb_Grid_XYZW[ 1590][2] =  0.478632845465845;
    Leb_Grid_XYZW[ 1590][3] =  0.000520260798048;

    Leb_Grid_XYZW[ 1591][0] =  0.866269123862177;
    Leb_Grid_XYZW[ 1591][1] =  0.143137710909197;
    Leb_Grid_XYZW[ 1591][2] = -0.478632845465845;
    Leb_Grid_XYZW[ 1591][3] =  0.000520260798048;

    Leb_Grid_XYZW[ 1592][0] =  0.866269123862177;
    Leb_Grid_XYZW[ 1592][1] = -0.143137710909197;
    Leb_Grid_XYZW[ 1592][2] =  0.478632845465845;
    Leb_Grid_XYZW[ 1592][3] =  0.000520260798048;

    Leb_Grid_XYZW[ 1593][0] =  0.866269123862177;
    Leb_Grid_XYZW[ 1593][1] = -0.143137710909197;
    Leb_Grid_XYZW[ 1593][2] = -0.478632845465845;
    Leb_Grid_XYZW[ 1593][3] =  0.000520260798048;

    Leb_Grid_XYZW[ 1594][0] = -0.866269123862177;
    Leb_Grid_XYZW[ 1594][1] =  0.143137710909197;
    Leb_Grid_XYZW[ 1594][2] =  0.478632845465845;
    Leb_Grid_XYZW[ 1594][3] =  0.000520260798048;

    Leb_Grid_XYZW[ 1595][0] = -0.866269123862177;
    Leb_Grid_XYZW[ 1595][1] =  0.143137710909197;
    Leb_Grid_XYZW[ 1595][2] = -0.478632845465845;
    Leb_Grid_XYZW[ 1595][3] =  0.000520260798048;

    Leb_Grid_XYZW[ 1596][0] = -0.866269123862177;
    Leb_Grid_XYZW[ 1596][1] = -0.143137710909197;
    Leb_Grid_XYZW[ 1596][2] =  0.478632845465845;
    Leb_Grid_XYZW[ 1596][3] =  0.000520260798048;

    Leb_Grid_XYZW[ 1597][0] = -0.866269123862177;
    Leb_Grid_XYZW[ 1597][1] = -0.143137710909197;
    Leb_Grid_XYZW[ 1597][2] = -0.478632845465845;
    Leb_Grid_XYZW[ 1597][3] =  0.000520260798048;

    Leb_Grid_XYZW[ 1598][0] =  0.530570207678977;
    Leb_Grid_XYZW[ 1598][1] =  0.192418638884357;
    Leb_Grid_XYZW[ 1598][2] =  0.825512157471577;
    Leb_Grid_XYZW[ 1598][3] =  0.000530993238833;

    Leb_Grid_XYZW[ 1599][0] =  0.530570207678977;
    Leb_Grid_XYZW[ 1599][1] =  0.192418638884357;
    Leb_Grid_XYZW[ 1599][2] = -0.825512157471577;
    Leb_Grid_XYZW[ 1599][3] =  0.000530993238833;

    Leb_Grid_XYZW[ 1600][0] =  0.530570207678977;
    Leb_Grid_XYZW[ 1600][1] = -0.192418638884357;
    Leb_Grid_XYZW[ 1600][2] =  0.825512157471577;
    Leb_Grid_XYZW[ 1600][3] =  0.000530993238833;

    Leb_Grid_XYZW[ 1601][0] =  0.530570207678977;
    Leb_Grid_XYZW[ 1601][1] = -0.192418638884357;
    Leb_Grid_XYZW[ 1601][2] = -0.825512157471577;
    Leb_Grid_XYZW[ 1601][3] =  0.000530993238833;

    Leb_Grid_XYZW[ 1602][0] = -0.530570207678977;
    Leb_Grid_XYZW[ 1602][1] =  0.192418638884357;
    Leb_Grid_XYZW[ 1602][2] =  0.825512157471577;
    Leb_Grid_XYZW[ 1602][3] =  0.000530993238833;

    Leb_Grid_XYZW[ 1603][0] = -0.530570207678977;
    Leb_Grid_XYZW[ 1603][1] =  0.192418638884357;
    Leb_Grid_XYZW[ 1603][2] = -0.825512157471577;
    Leb_Grid_XYZW[ 1603][3] =  0.000530993238833;

    Leb_Grid_XYZW[ 1604][0] = -0.530570207678977;
    Leb_Grid_XYZW[ 1604][1] = -0.192418638884357;
    Leb_Grid_XYZW[ 1604][2] =  0.825512157471577;
    Leb_Grid_XYZW[ 1604][3] =  0.000530993238833;

    Leb_Grid_XYZW[ 1605][0] = -0.530570207678977;
    Leb_Grid_XYZW[ 1605][1] = -0.192418638884357;
    Leb_Grid_XYZW[ 1605][2] = -0.825512157471577;
    Leb_Grid_XYZW[ 1605][3] =  0.000530993238833;

    Leb_Grid_XYZW[ 1606][0] =  0.530570207678977;
    Leb_Grid_XYZW[ 1606][1] =  0.825512157471577;
    Leb_Grid_XYZW[ 1606][2] =  0.192418638884357;
    Leb_Grid_XYZW[ 1606][3] =  0.000530993238833;

    Leb_Grid_XYZW[ 1607][0] =  0.530570207678977;
    Leb_Grid_XYZW[ 1607][1] =  0.825512157471577;
    Leb_Grid_XYZW[ 1607][2] = -0.192418638884357;
    Leb_Grid_XYZW[ 1607][3] =  0.000530993238833;

    Leb_Grid_XYZW[ 1608][0] =  0.530570207678977;
    Leb_Grid_XYZW[ 1608][1] = -0.825512157471577;
    Leb_Grid_XYZW[ 1608][2] =  0.192418638884357;
    Leb_Grid_XYZW[ 1608][3] =  0.000530993238833;

    Leb_Grid_XYZW[ 1609][0] =  0.530570207678977;
    Leb_Grid_XYZW[ 1609][1] = -0.825512157471577;
    Leb_Grid_XYZW[ 1609][2] = -0.192418638884357;
    Leb_Grid_XYZW[ 1609][3] =  0.000530993238833;

    Leb_Grid_XYZW[ 1610][0] = -0.530570207678978;
    Leb_Grid_XYZW[ 1610][1] =  0.825512157471577;
    Leb_Grid_XYZW[ 1610][2] =  0.192418638884357;
    Leb_Grid_XYZW[ 1610][3] =  0.000530993238833;

    Leb_Grid_XYZW[ 1611][0] = -0.530570207678978;
    Leb_Grid_XYZW[ 1611][1] =  0.825512157471577;
    Leb_Grid_XYZW[ 1611][2] = -0.192418638884357;
    Leb_Grid_XYZW[ 1611][3] =  0.000530993238833;

    Leb_Grid_XYZW[ 1612][0] = -0.530570207678978;
    Leb_Grid_XYZW[ 1612][1] = -0.825512157471577;
    Leb_Grid_XYZW[ 1612][2] =  0.192418638884357;
    Leb_Grid_XYZW[ 1612][3] =  0.000530993238833;

    Leb_Grid_XYZW[ 1613][0] = -0.530570207678978;
    Leb_Grid_XYZW[ 1613][1] = -0.825512157471577;
    Leb_Grid_XYZW[ 1613][2] = -0.192418638884357;
    Leb_Grid_XYZW[ 1613][3] =  0.000530993238833;

    Leb_Grid_XYZW[ 1614][0] =  0.192418638884357;
    Leb_Grid_XYZW[ 1614][1] =  0.530570207678977;
    Leb_Grid_XYZW[ 1614][2] =  0.825512157471577;
    Leb_Grid_XYZW[ 1614][3] =  0.000530993238833;

    Leb_Grid_XYZW[ 1615][0] =  0.192418638884357;
    Leb_Grid_XYZW[ 1615][1] =  0.530570207678977;
    Leb_Grid_XYZW[ 1615][2] = -0.825512157471577;
    Leb_Grid_XYZW[ 1615][3] =  0.000530993238833;

    Leb_Grid_XYZW[ 1616][0] =  0.192418638884357;
    Leb_Grid_XYZW[ 1616][1] = -0.530570207678977;
    Leb_Grid_XYZW[ 1616][2] =  0.825512157471577;
    Leb_Grid_XYZW[ 1616][3] =  0.000530993238833;

    Leb_Grid_XYZW[ 1617][0] =  0.192418638884357;
    Leb_Grid_XYZW[ 1617][1] = -0.530570207678977;
    Leb_Grid_XYZW[ 1617][2] = -0.825512157471577;
    Leb_Grid_XYZW[ 1617][3] =  0.000530993238833;

    Leb_Grid_XYZW[ 1618][0] = -0.192418638884357;
    Leb_Grid_XYZW[ 1618][1] =  0.530570207678977;
    Leb_Grid_XYZW[ 1618][2] =  0.825512157471577;
    Leb_Grid_XYZW[ 1618][3] =  0.000530993238833;

    Leb_Grid_XYZW[ 1619][0] = -0.192418638884357;
    Leb_Grid_XYZW[ 1619][1] =  0.530570207678977;
    Leb_Grid_XYZW[ 1619][2] = -0.825512157471577;
    Leb_Grid_XYZW[ 1619][3] =  0.000530993238833;

    Leb_Grid_XYZW[ 1620][0] = -0.192418638884357;
    Leb_Grid_XYZW[ 1620][1] = -0.530570207678977;
    Leb_Grid_XYZW[ 1620][2] =  0.825512157471577;
    Leb_Grid_XYZW[ 1620][3] =  0.000530993238833;

    Leb_Grid_XYZW[ 1621][0] = -0.192418638884357;
    Leb_Grid_XYZW[ 1621][1] = -0.530570207678977;
    Leb_Grid_XYZW[ 1621][2] = -0.825512157471577;
    Leb_Grid_XYZW[ 1621][3] =  0.000530993238833;

    Leb_Grid_XYZW[ 1622][0] =  0.192418638884357;
    Leb_Grid_XYZW[ 1622][1] =  0.825512157471577;
    Leb_Grid_XYZW[ 1622][2] =  0.530570207678977;
    Leb_Grid_XYZW[ 1622][3] =  0.000530993238833;

    Leb_Grid_XYZW[ 1623][0] =  0.192418638884357;
    Leb_Grid_XYZW[ 1623][1] =  0.825512157471577;
    Leb_Grid_XYZW[ 1623][2] = -0.530570207678978;
    Leb_Grid_XYZW[ 1623][3] =  0.000530993238833;

    Leb_Grid_XYZW[ 1624][0] =  0.192418638884357;
    Leb_Grid_XYZW[ 1624][1] = -0.825512157471577;
    Leb_Grid_XYZW[ 1624][2] =  0.530570207678977;
    Leb_Grid_XYZW[ 1624][3] =  0.000530993238833;

    Leb_Grid_XYZW[ 1625][0] =  0.192418638884357;
    Leb_Grid_XYZW[ 1625][1] = -0.825512157471577;
    Leb_Grid_XYZW[ 1625][2] = -0.530570207678978;
    Leb_Grid_XYZW[ 1625][3] =  0.000530993238833;

    Leb_Grid_XYZW[ 1626][0] = -0.192418638884357;
    Leb_Grid_XYZW[ 1626][1] =  0.825512157471577;
    Leb_Grid_XYZW[ 1626][2] =  0.530570207678977;
    Leb_Grid_XYZW[ 1626][3] =  0.000530993238833;

    Leb_Grid_XYZW[ 1627][0] = -0.192418638884357;
    Leb_Grid_XYZW[ 1627][1] =  0.825512157471577;
    Leb_Grid_XYZW[ 1627][2] = -0.530570207678978;
    Leb_Grid_XYZW[ 1627][3] =  0.000530993238833;

    Leb_Grid_XYZW[ 1628][0] = -0.192418638884357;
    Leb_Grid_XYZW[ 1628][1] = -0.825512157471577;
    Leb_Grid_XYZW[ 1628][2] =  0.530570207678977;
    Leb_Grid_XYZW[ 1628][3] =  0.000530993238833;

    Leb_Grid_XYZW[ 1629][0] = -0.192418638884357;
    Leb_Grid_XYZW[ 1629][1] = -0.825512157471577;
    Leb_Grid_XYZW[ 1629][2] = -0.530570207678978;
    Leb_Grid_XYZW[ 1629][3] =  0.000530993238833;

    Leb_Grid_XYZW[ 1630][0] =  0.825512157471577;
    Leb_Grid_XYZW[ 1630][1] =  0.530570207678977;
    Leb_Grid_XYZW[ 1630][2] =  0.192418638884357;
    Leb_Grid_XYZW[ 1630][3] =  0.000530993238833;

    Leb_Grid_XYZW[ 1631][0] =  0.825512157471577;
    Leb_Grid_XYZW[ 1631][1] =  0.530570207678977;
    Leb_Grid_XYZW[ 1631][2] = -0.192418638884357;
    Leb_Grid_XYZW[ 1631][3] =  0.000530993238833;

    Leb_Grid_XYZW[ 1632][0] =  0.825512157471577;
    Leb_Grid_XYZW[ 1632][1] = -0.530570207678977;
    Leb_Grid_XYZW[ 1632][2] =  0.192418638884357;
    Leb_Grid_XYZW[ 1632][3] =  0.000530993238833;

    Leb_Grid_XYZW[ 1633][0] =  0.825512157471577;
    Leb_Grid_XYZW[ 1633][1] = -0.530570207678977;
    Leb_Grid_XYZW[ 1633][2] = -0.192418638884357;
    Leb_Grid_XYZW[ 1633][3] =  0.000530993238833;

    Leb_Grid_XYZW[ 1634][0] = -0.825512157471577;
    Leb_Grid_XYZW[ 1634][1] =  0.530570207678977;
    Leb_Grid_XYZW[ 1634][2] =  0.192418638884357;
    Leb_Grid_XYZW[ 1634][3] =  0.000530993238833;

    Leb_Grid_XYZW[ 1635][0] = -0.825512157471577;
    Leb_Grid_XYZW[ 1635][1] =  0.530570207678977;
    Leb_Grid_XYZW[ 1635][2] = -0.192418638884357;
    Leb_Grid_XYZW[ 1635][3] =  0.000530993238833;

    Leb_Grid_XYZW[ 1636][0] = -0.825512157471577;
    Leb_Grid_XYZW[ 1636][1] = -0.530570207678977;
    Leb_Grid_XYZW[ 1636][2] =  0.192418638884357;
    Leb_Grid_XYZW[ 1636][3] =  0.000530993238833;

    Leb_Grid_XYZW[ 1637][0] = -0.825512157471577;
    Leb_Grid_XYZW[ 1637][1] = -0.530570207678977;
    Leb_Grid_XYZW[ 1637][2] = -0.192418638884357;
    Leb_Grid_XYZW[ 1637][3] =  0.000530993238833;

    Leb_Grid_XYZW[ 1638][0] =  0.825512157471577;
    Leb_Grid_XYZW[ 1638][1] =  0.192418638884357;
    Leb_Grid_XYZW[ 1638][2] =  0.530570207678977;
    Leb_Grid_XYZW[ 1638][3] =  0.000530993238833;

    Leb_Grid_XYZW[ 1639][0] =  0.825512157471577;
    Leb_Grid_XYZW[ 1639][1] =  0.192418638884357;
    Leb_Grid_XYZW[ 1639][2] = -0.530570207678978;
    Leb_Grid_XYZW[ 1639][3] =  0.000530993238833;

    Leb_Grid_XYZW[ 1640][0] =  0.825512157471577;
    Leb_Grid_XYZW[ 1640][1] = -0.192418638884357;
    Leb_Grid_XYZW[ 1640][2] =  0.530570207678977;
    Leb_Grid_XYZW[ 1640][3] =  0.000530993238833;

    Leb_Grid_XYZW[ 1641][0] =  0.825512157471577;
    Leb_Grid_XYZW[ 1641][1] = -0.192418638884357;
    Leb_Grid_XYZW[ 1641][2] = -0.530570207678978;
    Leb_Grid_XYZW[ 1641][3] =  0.000530993238833;

    Leb_Grid_XYZW[ 1642][0] = -0.825512157471577;
    Leb_Grid_XYZW[ 1642][1] =  0.192418638884357;
    Leb_Grid_XYZW[ 1642][2] =  0.530570207678977;
    Leb_Grid_XYZW[ 1642][3] =  0.000530993238833;

    Leb_Grid_XYZW[ 1643][0] = -0.825512157471577;
    Leb_Grid_XYZW[ 1643][1] =  0.192418638884357;
    Leb_Grid_XYZW[ 1643][2] = -0.530570207678978;
    Leb_Grid_XYZW[ 1643][3] =  0.000530993238833;

    Leb_Grid_XYZW[ 1644][0] = -0.825512157471577;
    Leb_Grid_XYZW[ 1644][1] = -0.192418638884357;
    Leb_Grid_XYZW[ 1644][2] =  0.530570207678977;
    Leb_Grid_XYZW[ 1644][3] =  0.000530993238833;

    Leb_Grid_XYZW[ 1645][0] = -0.825512157471577;
    Leb_Grid_XYZW[ 1645][1] = -0.192418638884357;
    Leb_Grid_XYZW[ 1645][2] = -0.530570207678978;
    Leb_Grid_XYZW[ 1645][3] =  0.000530993238833;

    Leb_Grid_XYZW[ 1646][0] =  0.579343622423179;
    Leb_Grid_XYZW[ 1646][1] =  0.241159094477519;
    Leb_Grid_XYZW[ 1646][2] =  0.778590558835883;
    Leb_Grid_XYZW[ 1646][3] =  0.000537741977090;

    Leb_Grid_XYZW[ 1647][0] =  0.579343622423179;
    Leb_Grid_XYZW[ 1647][1] =  0.241159094477519;
    Leb_Grid_XYZW[ 1647][2] = -0.778590558835883;
    Leb_Grid_XYZW[ 1647][3] =  0.000537741977090;

    Leb_Grid_XYZW[ 1648][0] =  0.579343622423179;
    Leb_Grid_XYZW[ 1648][1] = -0.241159094477519;
    Leb_Grid_XYZW[ 1648][2] =  0.778590558835883;
    Leb_Grid_XYZW[ 1648][3] =  0.000537741977090;

    Leb_Grid_XYZW[ 1649][0] =  0.579343622423179;
    Leb_Grid_XYZW[ 1649][1] = -0.241159094477519;
    Leb_Grid_XYZW[ 1649][2] = -0.778590558835883;
    Leb_Grid_XYZW[ 1649][3] =  0.000537741977090;

    Leb_Grid_XYZW[ 1650][0] = -0.579343622423179;
    Leb_Grid_XYZW[ 1650][1] =  0.241159094477519;
    Leb_Grid_XYZW[ 1650][2] =  0.778590558835883;
    Leb_Grid_XYZW[ 1650][3] =  0.000537741977090;

    Leb_Grid_XYZW[ 1651][0] = -0.579343622423179;
    Leb_Grid_XYZW[ 1651][1] =  0.241159094477519;
    Leb_Grid_XYZW[ 1651][2] = -0.778590558835883;
    Leb_Grid_XYZW[ 1651][3] =  0.000537741977090;

    Leb_Grid_XYZW[ 1652][0] = -0.579343622423179;
    Leb_Grid_XYZW[ 1652][1] = -0.241159094477519;
    Leb_Grid_XYZW[ 1652][2] =  0.778590558835883;
    Leb_Grid_XYZW[ 1652][3] =  0.000537741977090;

    Leb_Grid_XYZW[ 1653][0] = -0.579343622423179;
    Leb_Grid_XYZW[ 1653][1] = -0.241159094477519;
    Leb_Grid_XYZW[ 1653][2] = -0.778590558835883;
    Leb_Grid_XYZW[ 1653][3] =  0.000537741977090;

    Leb_Grid_XYZW[ 1654][0] =  0.579343622423179;
    Leb_Grid_XYZW[ 1654][1] =  0.778590558835883;
    Leb_Grid_XYZW[ 1654][2] =  0.241159094477519;
    Leb_Grid_XYZW[ 1654][3] =  0.000537741977090;

    Leb_Grid_XYZW[ 1655][0] =  0.579343622423179;
    Leb_Grid_XYZW[ 1655][1] =  0.778590558835883;
    Leb_Grid_XYZW[ 1655][2] = -0.241159094477519;
    Leb_Grid_XYZW[ 1655][3] =  0.000537741977090;

    Leb_Grid_XYZW[ 1656][0] =  0.579343622423179;
    Leb_Grid_XYZW[ 1656][1] = -0.778590558835883;
    Leb_Grid_XYZW[ 1656][2] =  0.241159094477519;
    Leb_Grid_XYZW[ 1656][3] =  0.000537741977090;

    Leb_Grid_XYZW[ 1657][0] =  0.579343622423179;
    Leb_Grid_XYZW[ 1657][1] = -0.778590558835883;
    Leb_Grid_XYZW[ 1657][2] = -0.241159094477519;
    Leb_Grid_XYZW[ 1657][3] =  0.000537741977090;

    Leb_Grid_XYZW[ 1658][0] = -0.579343622423179;
    Leb_Grid_XYZW[ 1658][1] =  0.778590558835883;
    Leb_Grid_XYZW[ 1658][2] =  0.241159094477519;
    Leb_Grid_XYZW[ 1658][3] =  0.000537741977090;

    Leb_Grid_XYZW[ 1659][0] = -0.579343622423179;
    Leb_Grid_XYZW[ 1659][1] =  0.778590558835883;
    Leb_Grid_XYZW[ 1659][2] = -0.241159094477519;
    Leb_Grid_XYZW[ 1659][3] =  0.000537741977090;

    Leb_Grid_XYZW[ 1660][0] = -0.579343622423179;
    Leb_Grid_XYZW[ 1660][1] = -0.778590558835883;
    Leb_Grid_XYZW[ 1660][2] =  0.241159094477519;
    Leb_Grid_XYZW[ 1660][3] =  0.000537741977090;

    Leb_Grid_XYZW[ 1661][0] = -0.579343622423179;
    Leb_Grid_XYZW[ 1661][1] = -0.778590558835883;
    Leb_Grid_XYZW[ 1661][2] = -0.241159094477519;
    Leb_Grid_XYZW[ 1661][3] =  0.000537741977090;

    Leb_Grid_XYZW[ 1662][0] =  0.241159094477519;
    Leb_Grid_XYZW[ 1662][1] =  0.579343622423179;
    Leb_Grid_XYZW[ 1662][2] =  0.778590558835883;
    Leb_Grid_XYZW[ 1662][3] =  0.000537741977090;

    Leb_Grid_XYZW[ 1663][0] =  0.241159094477519;
    Leb_Grid_XYZW[ 1663][1] =  0.579343622423179;
    Leb_Grid_XYZW[ 1663][2] = -0.778590558835883;
    Leb_Grid_XYZW[ 1663][3] =  0.000537741977090;

    Leb_Grid_XYZW[ 1664][0] =  0.241159094477519;
    Leb_Grid_XYZW[ 1664][1] = -0.579343622423179;
    Leb_Grid_XYZW[ 1664][2] =  0.778590558835883;
    Leb_Grid_XYZW[ 1664][3] =  0.000537741977090;

    Leb_Grid_XYZW[ 1665][0] =  0.241159094477519;
    Leb_Grid_XYZW[ 1665][1] = -0.579343622423179;
    Leb_Grid_XYZW[ 1665][2] = -0.778590558835883;
    Leb_Grid_XYZW[ 1665][3] =  0.000537741977090;

    Leb_Grid_XYZW[ 1666][0] = -0.241159094477519;
    Leb_Grid_XYZW[ 1666][1] =  0.579343622423179;
    Leb_Grid_XYZW[ 1666][2] =  0.778590558835883;
    Leb_Grid_XYZW[ 1666][3] =  0.000537741977090;

    Leb_Grid_XYZW[ 1667][0] = -0.241159094477519;
    Leb_Grid_XYZW[ 1667][1] =  0.579343622423179;
    Leb_Grid_XYZW[ 1667][2] = -0.778590558835883;
    Leb_Grid_XYZW[ 1667][3] =  0.000537741977090;

    Leb_Grid_XYZW[ 1668][0] = -0.241159094477519;
    Leb_Grid_XYZW[ 1668][1] = -0.579343622423179;
    Leb_Grid_XYZW[ 1668][2] =  0.778590558835883;
    Leb_Grid_XYZW[ 1668][3] =  0.000537741977090;

    Leb_Grid_XYZW[ 1669][0] = -0.241159094477519;
    Leb_Grid_XYZW[ 1669][1] = -0.579343622423179;
    Leb_Grid_XYZW[ 1669][2] = -0.778590558835883;
    Leb_Grid_XYZW[ 1669][3] =  0.000537741977090;

    Leb_Grid_XYZW[ 1670][0] =  0.241159094477519;
    Leb_Grid_XYZW[ 1670][1] =  0.778590558835883;
    Leb_Grid_XYZW[ 1670][2] =  0.579343622423179;
    Leb_Grid_XYZW[ 1670][3] =  0.000537741977090;

    Leb_Grid_XYZW[ 1671][0] =  0.241159094477519;
    Leb_Grid_XYZW[ 1671][1] =  0.778590558835883;
    Leb_Grid_XYZW[ 1671][2] = -0.579343622423179;
    Leb_Grid_XYZW[ 1671][3] =  0.000537741977090;

    Leb_Grid_XYZW[ 1672][0] =  0.241159094477519;
    Leb_Grid_XYZW[ 1672][1] = -0.778590558835883;
    Leb_Grid_XYZW[ 1672][2] =  0.579343622423179;
    Leb_Grid_XYZW[ 1672][3] =  0.000537741977090;

    Leb_Grid_XYZW[ 1673][0] =  0.241159094477519;
    Leb_Grid_XYZW[ 1673][1] = -0.778590558835883;
    Leb_Grid_XYZW[ 1673][2] = -0.579343622423179;
    Leb_Grid_XYZW[ 1673][3] =  0.000537741977090;

    Leb_Grid_XYZW[ 1674][0] = -0.241159094477519;
    Leb_Grid_XYZW[ 1674][1] =  0.778590558835883;
    Leb_Grid_XYZW[ 1674][2] =  0.579343622423179;
    Leb_Grid_XYZW[ 1674][3] =  0.000537741977090;

    Leb_Grid_XYZW[ 1675][0] = -0.241159094477519;
    Leb_Grid_XYZW[ 1675][1] =  0.778590558835883;
    Leb_Grid_XYZW[ 1675][2] = -0.579343622423179;
    Leb_Grid_XYZW[ 1675][3] =  0.000537741977090;

    Leb_Grid_XYZW[ 1676][0] = -0.241159094477519;
    Leb_Grid_XYZW[ 1676][1] = -0.778590558835883;
    Leb_Grid_XYZW[ 1676][2] =  0.579343622423179;
    Leb_Grid_XYZW[ 1676][3] =  0.000537741977090;

    Leb_Grid_XYZW[ 1677][0] = -0.241159094477519;
    Leb_Grid_XYZW[ 1677][1] = -0.778590558835883;
    Leb_Grid_XYZW[ 1677][2] = -0.579343622423179;
    Leb_Grid_XYZW[ 1677][3] =  0.000537741977090;

    Leb_Grid_XYZW[ 1678][0] =  0.778590558835883;
    Leb_Grid_XYZW[ 1678][1] =  0.579343622423179;
    Leb_Grid_XYZW[ 1678][2] =  0.241159094477519;
    Leb_Grid_XYZW[ 1678][3] =  0.000537741977090;

    Leb_Grid_XYZW[ 1679][0] =  0.778590558835883;
    Leb_Grid_XYZW[ 1679][1] =  0.579343622423179;
    Leb_Grid_XYZW[ 1679][2] = -0.241159094477519;
    Leb_Grid_XYZW[ 1679][3] =  0.000537741977090;

    Leb_Grid_XYZW[ 1680][0] =  0.778590558835883;
    Leb_Grid_XYZW[ 1680][1] = -0.579343622423179;
    Leb_Grid_XYZW[ 1680][2] =  0.241159094477519;
    Leb_Grid_XYZW[ 1680][3] =  0.000537741977090;

    Leb_Grid_XYZW[ 1681][0] =  0.778590558835883;
    Leb_Grid_XYZW[ 1681][1] = -0.579343622423179;
    Leb_Grid_XYZW[ 1681][2] = -0.241159094477519;
    Leb_Grid_XYZW[ 1681][3] =  0.000537741977090;

    Leb_Grid_XYZW[ 1682][0] = -0.778590558835883;
    Leb_Grid_XYZW[ 1682][1] =  0.579343622423179;
    Leb_Grid_XYZW[ 1682][2] =  0.241159094477519;
    Leb_Grid_XYZW[ 1682][3] =  0.000537741977090;

    Leb_Grid_XYZW[ 1683][0] = -0.778590558835883;
    Leb_Grid_XYZW[ 1683][1] =  0.579343622423179;
    Leb_Grid_XYZW[ 1683][2] = -0.241159094477519;
    Leb_Grid_XYZW[ 1683][3] =  0.000537741977090;

    Leb_Grid_XYZW[ 1684][0] = -0.778590558835883;
    Leb_Grid_XYZW[ 1684][1] = -0.579343622423179;
    Leb_Grid_XYZW[ 1684][2] =  0.241159094477519;
    Leb_Grid_XYZW[ 1684][3] =  0.000537741977090;

    Leb_Grid_XYZW[ 1685][0] = -0.778590558835883;
    Leb_Grid_XYZW[ 1685][1] = -0.579343622423179;
    Leb_Grid_XYZW[ 1685][2] = -0.241159094477519;
    Leb_Grid_XYZW[ 1685][3] =  0.000537741977090;

    Leb_Grid_XYZW[ 1686][0] =  0.778590558835883;
    Leb_Grid_XYZW[ 1686][1] =  0.241159094477519;
    Leb_Grid_XYZW[ 1686][2] =  0.579343622423179;
    Leb_Grid_XYZW[ 1686][3] =  0.000537741977090;

    Leb_Grid_XYZW[ 1687][0] =  0.778590558835883;
    Leb_Grid_XYZW[ 1687][1] =  0.241159094477519;
    Leb_Grid_XYZW[ 1687][2] = -0.579343622423179;
    Leb_Grid_XYZW[ 1687][3] =  0.000537741977090;

    Leb_Grid_XYZW[ 1688][0] =  0.778590558835883;
    Leb_Grid_XYZW[ 1688][1] = -0.241159094477519;
    Leb_Grid_XYZW[ 1688][2] =  0.579343622423179;
    Leb_Grid_XYZW[ 1688][3] =  0.000537741977090;

    Leb_Grid_XYZW[ 1689][0] =  0.778590558835883;
    Leb_Grid_XYZW[ 1689][1] = -0.241159094477519;
    Leb_Grid_XYZW[ 1689][2] = -0.579343622423179;
    Leb_Grid_XYZW[ 1689][3] =  0.000537741977090;

    Leb_Grid_XYZW[ 1690][0] = -0.778590558835883;
    Leb_Grid_XYZW[ 1690][1] =  0.241159094477519;
    Leb_Grid_XYZW[ 1690][2] =  0.579343622423179;
    Leb_Grid_XYZW[ 1690][3] =  0.000537741977090;

    Leb_Grid_XYZW[ 1691][0] = -0.778590558835883;
    Leb_Grid_XYZW[ 1691][1] =  0.241159094477519;
    Leb_Grid_XYZW[ 1691][2] = -0.579343622423179;
    Leb_Grid_XYZW[ 1691][3] =  0.000537741977090;

    Leb_Grid_XYZW[ 1692][0] = -0.778590558835883;
    Leb_Grid_XYZW[ 1692][1] = -0.241159094477519;
    Leb_Grid_XYZW[ 1692][2] =  0.579343622423179;
    Leb_Grid_XYZW[ 1692][3] =  0.000537741977090;

    Leb_Grid_XYZW[ 1693][0] = -0.778590558835883;
    Leb_Grid_XYZW[ 1693][1] = -0.241159094477519;
    Leb_Grid_XYZW[ 1693][2] = -0.579343622423179;
    Leb_Grid_XYZW[ 1693][3] =  0.000537741977090;

    Leb_Grid_XYZW[ 1694][0] =  0.624706901709475;
    Leb_Grid_XYZW[ 1694][1] =  0.288687149158360;
    Leb_Grid_XYZW[ 1694][2] =  0.725534986659752;
    Leb_Grid_XYZW[ 1694][3] =  0.000541169633168;

    Leb_Grid_XYZW[ 1695][0] =  0.624706901709475;
    Leb_Grid_XYZW[ 1695][1] =  0.288687149158361;
    Leb_Grid_XYZW[ 1695][2] = -0.725534986659752;
    Leb_Grid_XYZW[ 1695][3] =  0.000541169633168;

    Leb_Grid_XYZW[ 1696][0] =  0.624706901709475;
    Leb_Grid_XYZW[ 1696][1] = -0.288687149158360;
    Leb_Grid_XYZW[ 1696][2] =  0.725534986659752;
    Leb_Grid_XYZW[ 1696][3] =  0.000541169633168;

    Leb_Grid_XYZW[ 1697][0] =  0.624706901709475;
    Leb_Grid_XYZW[ 1697][1] = -0.288687149158361;
    Leb_Grid_XYZW[ 1697][2] = -0.725534986659752;
    Leb_Grid_XYZW[ 1697][3] =  0.000541169633168;

    Leb_Grid_XYZW[ 1698][0] = -0.624706901709475;
    Leb_Grid_XYZW[ 1698][1] =  0.288687149158361;
    Leb_Grid_XYZW[ 1698][2] =  0.725534986659752;
    Leb_Grid_XYZW[ 1698][3] =  0.000541169633168;

    Leb_Grid_XYZW[ 1699][0] = -0.624706901709475;
    Leb_Grid_XYZW[ 1699][1] =  0.288687149158361;
    Leb_Grid_XYZW[ 1699][2] = -0.725534986659752;
    Leb_Grid_XYZW[ 1699][3] =  0.000541169633168;

    Leb_Grid_XYZW[ 1700][0] = -0.624706901709475;
    Leb_Grid_XYZW[ 1700][1] = -0.288687149158361;
    Leb_Grid_XYZW[ 1700][2] =  0.725534986659752;
    Leb_Grid_XYZW[ 1700][3] =  0.000541169633168;

    Leb_Grid_XYZW[ 1701][0] = -0.624706901709475;
    Leb_Grid_XYZW[ 1701][1] = -0.288687149158361;
    Leb_Grid_XYZW[ 1701][2] = -0.725534986659752;
    Leb_Grid_XYZW[ 1701][3] =  0.000541169633168;

    Leb_Grid_XYZW[ 1702][0] =  0.624706901709475;
    Leb_Grid_XYZW[ 1702][1] =  0.725534986659752;
    Leb_Grid_XYZW[ 1702][2] =  0.288687149158361;
    Leb_Grid_XYZW[ 1702][3] =  0.000541169633168;

    Leb_Grid_XYZW[ 1703][0] =  0.624706901709475;
    Leb_Grid_XYZW[ 1703][1] =  0.725534986659753;
    Leb_Grid_XYZW[ 1703][2] = -0.288687149158360;
    Leb_Grid_XYZW[ 1703][3] =  0.000541169633168;

    Leb_Grid_XYZW[ 1704][0] =  0.624706901709475;
    Leb_Grid_XYZW[ 1704][1] = -0.725534986659752;
    Leb_Grid_XYZW[ 1704][2] =  0.288687149158361;
    Leb_Grid_XYZW[ 1704][3] =  0.000541169633168;

    Leb_Grid_XYZW[ 1705][0] =  0.624706901709475;
    Leb_Grid_XYZW[ 1705][1] = -0.725534986659753;
    Leb_Grid_XYZW[ 1705][2] = -0.288687149158360;
    Leb_Grid_XYZW[ 1705][3] =  0.000541169633168;

    Leb_Grid_XYZW[ 1706][0] = -0.624706901709475;
    Leb_Grid_XYZW[ 1706][1] =  0.725534986659752;
    Leb_Grid_XYZW[ 1706][2] =  0.288687149158361;
    Leb_Grid_XYZW[ 1706][3] =  0.000541169633168;

    Leb_Grid_XYZW[ 1707][0] = -0.624706901709475;
    Leb_Grid_XYZW[ 1707][1] =  0.725534986659753;
    Leb_Grid_XYZW[ 1707][2] = -0.288687149158360;
    Leb_Grid_XYZW[ 1707][3] =  0.000541169633168;

    Leb_Grid_XYZW[ 1708][0] = -0.624706901709475;
    Leb_Grid_XYZW[ 1708][1] = -0.725534986659752;
    Leb_Grid_XYZW[ 1708][2] =  0.288687149158361;
    Leb_Grid_XYZW[ 1708][3] =  0.000541169633168;

    Leb_Grid_XYZW[ 1709][0] = -0.624706901709475;
    Leb_Grid_XYZW[ 1709][1] = -0.725534986659753;
    Leb_Grid_XYZW[ 1709][2] = -0.288687149158360;
    Leb_Grid_XYZW[ 1709][3] =  0.000541169633168;

    Leb_Grid_XYZW[ 1710][0] =  0.288687149158360;
    Leb_Grid_XYZW[ 1710][1] =  0.624706901709475;
    Leb_Grid_XYZW[ 1710][2] =  0.725534986659752;
    Leb_Grid_XYZW[ 1710][3] =  0.000541169633168;

    Leb_Grid_XYZW[ 1711][0] =  0.288687149158361;
    Leb_Grid_XYZW[ 1711][1] =  0.624706901709475;
    Leb_Grid_XYZW[ 1711][2] = -0.725534986659752;
    Leb_Grid_XYZW[ 1711][3] =  0.000541169633168;

    Leb_Grid_XYZW[ 1712][0] =  0.288687149158360;
    Leb_Grid_XYZW[ 1712][1] = -0.624706901709475;
    Leb_Grid_XYZW[ 1712][2] =  0.725534986659752;
    Leb_Grid_XYZW[ 1712][3] =  0.000541169633168;

    Leb_Grid_XYZW[ 1713][0] =  0.288687149158361;
    Leb_Grid_XYZW[ 1713][1] = -0.624706901709475;
    Leb_Grid_XYZW[ 1713][2] = -0.725534986659752;
    Leb_Grid_XYZW[ 1713][3] =  0.000541169633168;

    Leb_Grid_XYZW[ 1714][0] = -0.288687149158360;
    Leb_Grid_XYZW[ 1714][1] =  0.624706901709475;
    Leb_Grid_XYZW[ 1714][2] =  0.725534986659752;
    Leb_Grid_XYZW[ 1714][3] =  0.000541169633168;

    Leb_Grid_XYZW[ 1715][0] = -0.288687149158360;
    Leb_Grid_XYZW[ 1715][1] =  0.624706901709475;
    Leb_Grid_XYZW[ 1715][2] = -0.725534986659752;
    Leb_Grid_XYZW[ 1715][3] =  0.000541169633168;

    Leb_Grid_XYZW[ 1716][0] = -0.288687149158360;
    Leb_Grid_XYZW[ 1716][1] = -0.624706901709475;
    Leb_Grid_XYZW[ 1716][2] =  0.725534986659752;
    Leb_Grid_XYZW[ 1716][3] =  0.000541169633168;

    Leb_Grid_XYZW[ 1717][0] = -0.288687149158360;
    Leb_Grid_XYZW[ 1717][1] = -0.624706901709475;
    Leb_Grid_XYZW[ 1717][2] = -0.725534986659752;
    Leb_Grid_XYZW[ 1717][3] =  0.000541169633168;

    Leb_Grid_XYZW[ 1718][0] =  0.288687149158360;
    Leb_Grid_XYZW[ 1718][1] =  0.725534986659752;
    Leb_Grid_XYZW[ 1718][2] =  0.624706901709475;
    Leb_Grid_XYZW[ 1718][3] =  0.000541169633168;

    Leb_Grid_XYZW[ 1719][0] =  0.288687149158360;
    Leb_Grid_XYZW[ 1719][1] =  0.725534986659753;
    Leb_Grid_XYZW[ 1719][2] = -0.624706901709475;
    Leb_Grid_XYZW[ 1719][3] =  0.000541169633168;

    Leb_Grid_XYZW[ 1720][0] =  0.288687149158360;
    Leb_Grid_XYZW[ 1720][1] = -0.725534986659752;
    Leb_Grid_XYZW[ 1720][2] =  0.624706901709475;
    Leb_Grid_XYZW[ 1720][3] =  0.000541169633168;

    Leb_Grid_XYZW[ 1721][0] =  0.288687149158360;
    Leb_Grid_XYZW[ 1721][1] = -0.725534986659753;
    Leb_Grid_XYZW[ 1721][2] = -0.624706901709475;
    Leb_Grid_XYZW[ 1721][3] =  0.000541169633168;

    Leb_Grid_XYZW[ 1722][0] = -0.288687149158361;
    Leb_Grid_XYZW[ 1722][1] =  0.725534986659752;
    Leb_Grid_XYZW[ 1722][2] =  0.624706901709475;
    Leb_Grid_XYZW[ 1722][3] =  0.000541169633168;

    Leb_Grid_XYZW[ 1723][0] = -0.288687149158361;
    Leb_Grid_XYZW[ 1723][1] =  0.725534986659752;
    Leb_Grid_XYZW[ 1723][2] = -0.624706901709475;
    Leb_Grid_XYZW[ 1723][3] =  0.000541169633168;

    Leb_Grid_XYZW[ 1724][0] = -0.288687149158361;
    Leb_Grid_XYZW[ 1724][1] = -0.725534986659752;
    Leb_Grid_XYZW[ 1724][2] =  0.624706901709475;
    Leb_Grid_XYZW[ 1724][3] =  0.000541169633168;

    Leb_Grid_XYZW[ 1725][0] = -0.288687149158361;
    Leb_Grid_XYZW[ 1725][1] = -0.725534986659752;
    Leb_Grid_XYZW[ 1725][2] = -0.624706901709475;
    Leb_Grid_XYZW[ 1725][3] =  0.000541169633168;

    Leb_Grid_XYZW[ 1726][0] =  0.725534986659752;
    Leb_Grid_XYZW[ 1726][1] =  0.624706901709475;
    Leb_Grid_XYZW[ 1726][2] =  0.288687149158361;
    Leb_Grid_XYZW[ 1726][3] =  0.000541169633168;

    Leb_Grid_XYZW[ 1727][0] =  0.725534986659752;
    Leb_Grid_XYZW[ 1727][1] =  0.624706901709475;
    Leb_Grid_XYZW[ 1727][2] = -0.288687149158360;
    Leb_Grid_XYZW[ 1727][3] =  0.000541169633168;

    Leb_Grid_XYZW[ 1728][0] =  0.725534986659752;
    Leb_Grid_XYZW[ 1728][1] = -0.624706901709475;
    Leb_Grid_XYZW[ 1728][2] =  0.288687149158361;
    Leb_Grid_XYZW[ 1728][3] =  0.000541169633168;

    Leb_Grid_XYZW[ 1729][0] =  0.725534986659752;
    Leb_Grid_XYZW[ 1729][1] = -0.624706901709475;
    Leb_Grid_XYZW[ 1729][2] = -0.288687149158360;
    Leb_Grid_XYZW[ 1729][3] =  0.000541169633168;

    Leb_Grid_XYZW[ 1730][0] = -0.725534986659752;
    Leb_Grid_XYZW[ 1730][1] =  0.624706901709475;
    Leb_Grid_XYZW[ 1730][2] =  0.288687149158361;
    Leb_Grid_XYZW[ 1730][3] =  0.000541169633168;

    Leb_Grid_XYZW[ 1731][0] = -0.725534986659752;
    Leb_Grid_XYZW[ 1731][1] =  0.624706901709475;
    Leb_Grid_XYZW[ 1731][2] = -0.288687149158360;
    Leb_Grid_XYZW[ 1731][3] =  0.000541169633168;

    Leb_Grid_XYZW[ 1732][0] = -0.725534986659752;
    Leb_Grid_XYZW[ 1732][1] = -0.624706901709475;
    Leb_Grid_XYZW[ 1732][2] =  0.288687149158361;
    Leb_Grid_XYZW[ 1732][3] =  0.000541169633168;

    Leb_Grid_XYZW[ 1733][0] = -0.725534986659752;
    Leb_Grid_XYZW[ 1733][1] = -0.624706901709475;
    Leb_Grid_XYZW[ 1733][2] = -0.288687149158360;
    Leb_Grid_XYZW[ 1733][3] =  0.000541169633168;

    Leb_Grid_XYZW[ 1734][0] =  0.725534986659752;
    Leb_Grid_XYZW[ 1734][1] =  0.288687149158361;
    Leb_Grid_XYZW[ 1734][2] =  0.624706901709475;
    Leb_Grid_XYZW[ 1734][3] =  0.000541169633168;

    Leb_Grid_XYZW[ 1735][0] =  0.725534986659752;
    Leb_Grid_XYZW[ 1735][1] =  0.288687149158361;
    Leb_Grid_XYZW[ 1735][2] = -0.624706901709475;
    Leb_Grid_XYZW[ 1735][3] =  0.000541169633168;

    Leb_Grid_XYZW[ 1736][0] =  0.725534986659752;
    Leb_Grid_XYZW[ 1736][1] = -0.288687149158361;
    Leb_Grid_XYZW[ 1736][2] =  0.624706901709475;
    Leb_Grid_XYZW[ 1736][3] =  0.000541169633168;

    Leb_Grid_XYZW[ 1737][0] =  0.725534986659752;
    Leb_Grid_XYZW[ 1737][1] = -0.288687149158361;
    Leb_Grid_XYZW[ 1737][2] = -0.624706901709475;
    Leb_Grid_XYZW[ 1737][3] =  0.000541169633168;

    Leb_Grid_XYZW[ 1738][0] = -0.725534986659752;
    Leb_Grid_XYZW[ 1738][1] =  0.288687149158361;
    Leb_Grid_XYZW[ 1738][2] =  0.624706901709475;
    Leb_Grid_XYZW[ 1738][3] =  0.000541169633168;

    Leb_Grid_XYZW[ 1739][0] = -0.725534986659753;
    Leb_Grid_XYZW[ 1739][1] =  0.288687149158361;
    Leb_Grid_XYZW[ 1739][2] = -0.624706901709475;
    Leb_Grid_XYZW[ 1739][3] =  0.000541169633168;

    Leb_Grid_XYZW[ 1740][0] = -0.725534986659752;
    Leb_Grid_XYZW[ 1740][1] = -0.288687149158361;
    Leb_Grid_XYZW[ 1740][2] =  0.624706901709475;
    Leb_Grid_XYZW[ 1740][3] =  0.000541169633168;

    Leb_Grid_XYZW[ 1741][0] = -0.725534986659753;
    Leb_Grid_XYZW[ 1741][1] = -0.288687149158361;
    Leb_Grid_XYZW[ 1741][2] = -0.624706901709475;
    Leb_Grid_XYZW[ 1741][3] =  0.000541169633168;

    Leb_Grid_XYZW[ 1742][0] =  0.487431555253520;
    Leb_Grid_XYZW[ 1742][1] =  0.048049787749532;
    Leb_Grid_XYZW[ 1742][2] =  0.871838113895211;
    Leb_Grid_XYZW[ 1742][3] =  0.000519799629328;

    Leb_Grid_XYZW[ 1743][0] =  0.487431555253520;
    Leb_Grid_XYZW[ 1743][1] =  0.048049787749532;
    Leb_Grid_XYZW[ 1743][2] = -0.871838113895211;
    Leb_Grid_XYZW[ 1743][3] =  0.000519799629328;

    Leb_Grid_XYZW[ 1744][0] =  0.487431555253520;
    Leb_Grid_XYZW[ 1744][1] = -0.048049787749532;
    Leb_Grid_XYZW[ 1744][2] =  0.871838113895211;
    Leb_Grid_XYZW[ 1744][3] =  0.000519799629328;

    Leb_Grid_XYZW[ 1745][0] =  0.487431555253520;
    Leb_Grid_XYZW[ 1745][1] = -0.048049787749532;
    Leb_Grid_XYZW[ 1745][2] = -0.871838113895211;
    Leb_Grid_XYZW[ 1745][3] =  0.000519799629328;

    Leb_Grid_XYZW[ 1746][0] = -0.487431555253520;
    Leb_Grid_XYZW[ 1746][1] =  0.048049787749532;
    Leb_Grid_XYZW[ 1746][2] =  0.871838113895211;
    Leb_Grid_XYZW[ 1746][3] =  0.000519799629328;

    Leb_Grid_XYZW[ 1747][0] = -0.487431555253520;
    Leb_Grid_XYZW[ 1747][1] =  0.048049787749532;
    Leb_Grid_XYZW[ 1747][2] = -0.871838113895211;
    Leb_Grid_XYZW[ 1747][3] =  0.000519799629328;

    Leb_Grid_XYZW[ 1748][0] = -0.487431555253520;
    Leb_Grid_XYZW[ 1748][1] = -0.048049787749532;
    Leb_Grid_XYZW[ 1748][2] =  0.871838113895211;
    Leb_Grid_XYZW[ 1748][3] =  0.000519799629328;

    Leb_Grid_XYZW[ 1749][0] = -0.487431555253520;
    Leb_Grid_XYZW[ 1749][1] = -0.048049787749532;
    Leb_Grid_XYZW[ 1749][2] = -0.871838113895211;
    Leb_Grid_XYZW[ 1749][3] =  0.000519799629328;

    Leb_Grid_XYZW[ 1750][0] =  0.487431555253520;
    Leb_Grid_XYZW[ 1750][1] =  0.871838113895211;
    Leb_Grid_XYZW[ 1750][2] =  0.048049787749532;
    Leb_Grid_XYZW[ 1750][3] =  0.000519799629328;

    Leb_Grid_XYZW[ 1751][0] =  0.487431555253520;
    Leb_Grid_XYZW[ 1751][1] =  0.871838113895211;
    Leb_Grid_XYZW[ 1751][2] = -0.048049787749532;
    Leb_Grid_XYZW[ 1751][3] =  0.000519799629328;

    Leb_Grid_XYZW[ 1752][0] =  0.487431555253520;
    Leb_Grid_XYZW[ 1752][1] = -0.871838113895211;
    Leb_Grid_XYZW[ 1752][2] =  0.048049787749532;
    Leb_Grid_XYZW[ 1752][3] =  0.000519799629328;

    Leb_Grid_XYZW[ 1753][0] =  0.487431555253520;
    Leb_Grid_XYZW[ 1753][1] = -0.871838113895211;
    Leb_Grid_XYZW[ 1753][2] = -0.048049787749532;
    Leb_Grid_XYZW[ 1753][3] =  0.000519799629328;

    Leb_Grid_XYZW[ 1754][0] = -0.487431555253520;
    Leb_Grid_XYZW[ 1754][1] =  0.871838113895211;
    Leb_Grid_XYZW[ 1754][2] =  0.048049787749532;
    Leb_Grid_XYZW[ 1754][3] =  0.000519799629328;

    Leb_Grid_XYZW[ 1755][0] = -0.487431555253520;
    Leb_Grid_XYZW[ 1755][1] =  0.871838113895211;
    Leb_Grid_XYZW[ 1755][2] = -0.048049787749532;
    Leb_Grid_XYZW[ 1755][3] =  0.000519799629328;

    Leb_Grid_XYZW[ 1756][0] = -0.487431555253520;
    Leb_Grid_XYZW[ 1756][1] = -0.871838113895211;
    Leb_Grid_XYZW[ 1756][2] =  0.048049787749532;
    Leb_Grid_XYZW[ 1756][3] =  0.000519799629328;

    Leb_Grid_XYZW[ 1757][0] = -0.487431555253520;
    Leb_Grid_XYZW[ 1757][1] = -0.871838113895211;
    Leb_Grid_XYZW[ 1757][2] = -0.048049787749532;
    Leb_Grid_XYZW[ 1757][3] =  0.000519799629328;

    Leb_Grid_XYZW[ 1758][0] =  0.048049787749532;
    Leb_Grid_XYZW[ 1758][1] =  0.487431555253520;
    Leb_Grid_XYZW[ 1758][2] =  0.871838113895211;
    Leb_Grid_XYZW[ 1758][3] =  0.000519799629328;

    Leb_Grid_XYZW[ 1759][0] =  0.048049787749532;
    Leb_Grid_XYZW[ 1759][1] =  0.487431555253520;
    Leb_Grid_XYZW[ 1759][2] = -0.871838113895211;
    Leb_Grid_XYZW[ 1759][3] =  0.000519799629328;

    Leb_Grid_XYZW[ 1760][0] =  0.048049787749532;
    Leb_Grid_XYZW[ 1760][1] = -0.487431555253520;
    Leb_Grid_XYZW[ 1760][2] =  0.871838113895211;
    Leb_Grid_XYZW[ 1760][3] =  0.000519799629328;

    Leb_Grid_XYZW[ 1761][0] =  0.048049787749532;
    Leb_Grid_XYZW[ 1761][1] = -0.487431555253520;
    Leb_Grid_XYZW[ 1761][2] = -0.871838113895211;
    Leb_Grid_XYZW[ 1761][3] =  0.000519799629328;

    Leb_Grid_XYZW[ 1762][0] = -0.048049787749532;
    Leb_Grid_XYZW[ 1762][1] =  0.487431555253520;
    Leb_Grid_XYZW[ 1762][2] =  0.871838113895211;
    Leb_Grid_XYZW[ 1762][3] =  0.000519799629328;

    Leb_Grid_XYZW[ 1763][0] = -0.048049787749532;
    Leb_Grid_XYZW[ 1763][1] =  0.487431555253520;
    Leb_Grid_XYZW[ 1763][2] = -0.871838113895211;
    Leb_Grid_XYZW[ 1763][3] =  0.000519799629328;

    Leb_Grid_XYZW[ 1764][0] = -0.048049787749532;
    Leb_Grid_XYZW[ 1764][1] = -0.487431555253520;
    Leb_Grid_XYZW[ 1764][2] =  0.871838113895211;
    Leb_Grid_XYZW[ 1764][3] =  0.000519799629328;

    Leb_Grid_XYZW[ 1765][0] = -0.048049787749532;
    Leb_Grid_XYZW[ 1765][1] = -0.487431555253520;
    Leb_Grid_XYZW[ 1765][2] = -0.871838113895211;
    Leb_Grid_XYZW[ 1765][3] =  0.000519799629328;

    Leb_Grid_XYZW[ 1766][0] =  0.048049787749532;
    Leb_Grid_XYZW[ 1766][1] =  0.871838113895211;
    Leb_Grid_XYZW[ 1766][2] =  0.487431555253520;
    Leb_Grid_XYZW[ 1766][3] =  0.000519799629328;

    Leb_Grid_XYZW[ 1767][0] =  0.048049787749532;
    Leb_Grid_XYZW[ 1767][1] =  0.871838113895211;
    Leb_Grid_XYZW[ 1767][2] = -0.487431555253520;
    Leb_Grid_XYZW[ 1767][3] =  0.000519799629328;

    Leb_Grid_XYZW[ 1768][0] =  0.048049787749532;
    Leb_Grid_XYZW[ 1768][1] = -0.871838113895211;
    Leb_Grid_XYZW[ 1768][2] =  0.487431555253520;
    Leb_Grid_XYZW[ 1768][3] =  0.000519799629328;

    Leb_Grid_XYZW[ 1769][0] =  0.048049787749532;
    Leb_Grid_XYZW[ 1769][1] = -0.871838113895211;
    Leb_Grid_XYZW[ 1769][2] = -0.487431555253520;
    Leb_Grid_XYZW[ 1769][3] =  0.000519799629328;

    Leb_Grid_XYZW[ 1770][0] = -0.048049787749532;
    Leb_Grid_XYZW[ 1770][1] =  0.871838113895211;
    Leb_Grid_XYZW[ 1770][2] =  0.487431555253520;
    Leb_Grid_XYZW[ 1770][3] =  0.000519799629328;

    Leb_Grid_XYZW[ 1771][0] = -0.048049787749532;
    Leb_Grid_XYZW[ 1771][1] =  0.871838113895211;
    Leb_Grid_XYZW[ 1771][2] = -0.487431555253520;
    Leb_Grid_XYZW[ 1771][3] =  0.000519799629328;

    Leb_Grid_XYZW[ 1772][0] = -0.048049787749532;
    Leb_Grid_XYZW[ 1772][1] = -0.871838113895211;
    Leb_Grid_XYZW[ 1772][2] =  0.487431555253520;
    Leb_Grid_XYZW[ 1772][3] =  0.000519799629328;

    Leb_Grid_XYZW[ 1773][0] = -0.048049787749532;
    Leb_Grid_XYZW[ 1773][1] = -0.871838113895211;
    Leb_Grid_XYZW[ 1773][2] = -0.487431555253520;
    Leb_Grid_XYZW[ 1773][3] =  0.000519799629328;

    Leb_Grid_XYZW[ 1774][0] =  0.871838113895211;
    Leb_Grid_XYZW[ 1774][1] =  0.487431555253520;
    Leb_Grid_XYZW[ 1774][2] =  0.048049787749532;
    Leb_Grid_XYZW[ 1774][3] =  0.000519799629328;

    Leb_Grid_XYZW[ 1775][0] =  0.871838113895211;
    Leb_Grid_XYZW[ 1775][1] =  0.487431555253520;
    Leb_Grid_XYZW[ 1775][2] = -0.048049787749532;
    Leb_Grid_XYZW[ 1775][3] =  0.000519799629328;

    Leb_Grid_XYZW[ 1776][0] =  0.871838113895211;
    Leb_Grid_XYZW[ 1776][1] = -0.487431555253520;
    Leb_Grid_XYZW[ 1776][2] =  0.048049787749532;
    Leb_Grid_XYZW[ 1776][3] =  0.000519799629328;

    Leb_Grid_XYZW[ 1777][0] =  0.871838113895211;
    Leb_Grid_XYZW[ 1777][1] = -0.487431555253520;
    Leb_Grid_XYZW[ 1777][2] = -0.048049787749532;
    Leb_Grid_XYZW[ 1777][3] =  0.000519799629328;

    Leb_Grid_XYZW[ 1778][0] = -0.871838113895211;
    Leb_Grid_XYZW[ 1778][1] =  0.487431555253520;
    Leb_Grid_XYZW[ 1778][2] =  0.048049787749532;
    Leb_Grid_XYZW[ 1778][3] =  0.000519799629328;

    Leb_Grid_XYZW[ 1779][0] = -0.871838113895211;
    Leb_Grid_XYZW[ 1779][1] =  0.487431555253520;
    Leb_Grid_XYZW[ 1779][2] = -0.048049787749532;
    Leb_Grid_XYZW[ 1779][3] =  0.000519799629328;

    Leb_Grid_XYZW[ 1780][0] = -0.871838113895211;
    Leb_Grid_XYZW[ 1780][1] = -0.487431555253520;
    Leb_Grid_XYZW[ 1780][2] =  0.048049787749532;
    Leb_Grid_XYZW[ 1780][3] =  0.000519799629328;

    Leb_Grid_XYZW[ 1781][0] = -0.871838113895211;
    Leb_Grid_XYZW[ 1781][1] = -0.487431555253520;
    Leb_Grid_XYZW[ 1781][2] = -0.048049787749532;
    Leb_Grid_XYZW[ 1781][3] =  0.000519799629328;

    Leb_Grid_XYZW[ 1782][0] =  0.871838113895211;
    Leb_Grid_XYZW[ 1782][1] =  0.048049787749531;
    Leb_Grid_XYZW[ 1782][2] =  0.487431555253520;
    Leb_Grid_XYZW[ 1782][3] =  0.000519799629328;

    Leb_Grid_XYZW[ 1783][0] =  0.871838113895211;
    Leb_Grid_XYZW[ 1783][1] =  0.048049787749531;
    Leb_Grid_XYZW[ 1783][2] = -0.487431555253520;
    Leb_Grid_XYZW[ 1783][3] =  0.000519799629328;

    Leb_Grid_XYZW[ 1784][0] =  0.871838113895211;
    Leb_Grid_XYZW[ 1784][1] = -0.048049787749531;
    Leb_Grid_XYZW[ 1784][2] =  0.487431555253520;
    Leb_Grid_XYZW[ 1784][3] =  0.000519799629328;

    Leb_Grid_XYZW[ 1785][0] =  0.871838113895211;
    Leb_Grid_XYZW[ 1785][1] = -0.048049787749531;
    Leb_Grid_XYZW[ 1785][2] = -0.487431555253520;
    Leb_Grid_XYZW[ 1785][3] =  0.000519799629328;

    Leb_Grid_XYZW[ 1786][0] = -0.871838113895211;
    Leb_Grid_XYZW[ 1786][1] =  0.048049787749531;
    Leb_Grid_XYZW[ 1786][2] =  0.487431555253520;
    Leb_Grid_XYZW[ 1786][3] =  0.000519799629328;

    Leb_Grid_XYZW[ 1787][0] = -0.871838113895211;
    Leb_Grid_XYZW[ 1787][1] =  0.048049787749531;
    Leb_Grid_XYZW[ 1787][2] = -0.487431555253520;
    Leb_Grid_XYZW[ 1787][3] =  0.000519799629328;

    Leb_Grid_XYZW[ 1788][0] = -0.871838113895211;
    Leb_Grid_XYZW[ 1788][1] = -0.048049787749531;
    Leb_Grid_XYZW[ 1788][2] =  0.487431555253520;
    Leb_Grid_XYZW[ 1788][3] =  0.000519799629328;

    Leb_Grid_XYZW[ 1789][0] = -0.871838113895211;
    Leb_Grid_XYZW[ 1789][1] = -0.048049787749531;
    Leb_Grid_XYZW[ 1789][2] = -0.487431555253520;
    Leb_Grid_XYZW[ 1789][3] =  0.000519799629328;

    Leb_Grid_XYZW[ 1790][0] =  0.542733732205905;
    Leb_Grid_XYZW[ 1790][1] =  0.097168571993666;
    Leb_Grid_XYZW[ 1790][2] =  0.834265164406713;
    Leb_Grid_XYZW[ 1790][3] =  0.000531112083662;

    Leb_Grid_XYZW[ 1791][0] =  0.542733732205905;
    Leb_Grid_XYZW[ 1791][1] =  0.097168571993666;
    Leb_Grid_XYZW[ 1791][2] = -0.834265164406713;
    Leb_Grid_XYZW[ 1791][3] =  0.000531112083662;

    Leb_Grid_XYZW[ 1792][0] =  0.542733732205905;
    Leb_Grid_XYZW[ 1792][1] = -0.097168571993666;
    Leb_Grid_XYZW[ 1792][2] =  0.834265164406713;
    Leb_Grid_XYZW[ 1792][3] =  0.000531112083662;

    Leb_Grid_XYZW[ 1793][0] =  0.542733732205905;
    Leb_Grid_XYZW[ 1793][1] = -0.097168571993666;
    Leb_Grid_XYZW[ 1793][2] = -0.834265164406713;
    Leb_Grid_XYZW[ 1793][3] =  0.000531112083662;

    Leb_Grid_XYZW[ 1794][0] = -0.542733732205905;
    Leb_Grid_XYZW[ 1794][1] =  0.097168571993666;
    Leb_Grid_XYZW[ 1794][2] =  0.834265164406713;
    Leb_Grid_XYZW[ 1794][3] =  0.000531112083662;

    Leb_Grid_XYZW[ 1795][0] = -0.542733732205905;
    Leb_Grid_XYZW[ 1795][1] =  0.097168571993666;
    Leb_Grid_XYZW[ 1795][2] = -0.834265164406713;
    Leb_Grid_XYZW[ 1795][3] =  0.000531112083662;

    Leb_Grid_XYZW[ 1796][0] = -0.542733732205905;
    Leb_Grid_XYZW[ 1796][1] = -0.097168571993666;
    Leb_Grid_XYZW[ 1796][2] =  0.834265164406713;
    Leb_Grid_XYZW[ 1796][3] =  0.000531112083662;

    Leb_Grid_XYZW[ 1797][0] = -0.542733732205905;
    Leb_Grid_XYZW[ 1797][1] = -0.097168571993666;
    Leb_Grid_XYZW[ 1797][2] = -0.834265164406713;
    Leb_Grid_XYZW[ 1797][3] =  0.000531112083662;

    Leb_Grid_XYZW[ 1798][0] =  0.542733732205905;
    Leb_Grid_XYZW[ 1798][1] =  0.834265164406713;
    Leb_Grid_XYZW[ 1798][2] =  0.097168571993667;
    Leb_Grid_XYZW[ 1798][3] =  0.000531112083662;

    Leb_Grid_XYZW[ 1799][0] =  0.542733732205905;
    Leb_Grid_XYZW[ 1799][1] =  0.834265164406713;
    Leb_Grid_XYZW[ 1799][2] = -0.097168571993667;
    Leb_Grid_XYZW[ 1799][3] =  0.000531112083662;

    Leb_Grid_XYZW[ 1800][0] =  0.542733732205905;
    Leb_Grid_XYZW[ 1800][1] = -0.834265164406713;
    Leb_Grid_XYZW[ 1800][2] =  0.097168571993667;
    Leb_Grid_XYZW[ 1800][3] =  0.000531112083662;

    Leb_Grid_XYZW[ 1801][0] =  0.542733732205905;
    Leb_Grid_XYZW[ 1801][1] = -0.834265164406713;
    Leb_Grid_XYZW[ 1801][2] = -0.097168571993667;
    Leb_Grid_XYZW[ 1801][3] =  0.000531112083662;

    Leb_Grid_XYZW[ 1802][0] = -0.542733732205905;
    Leb_Grid_XYZW[ 1802][1] =  0.834265164406713;
    Leb_Grid_XYZW[ 1802][2] =  0.097168571993667;
    Leb_Grid_XYZW[ 1802][3] =  0.000531112083662;

    Leb_Grid_XYZW[ 1803][0] = -0.542733732205905;
    Leb_Grid_XYZW[ 1803][1] =  0.834265164406713;
    Leb_Grid_XYZW[ 1803][2] = -0.097168571993667;
    Leb_Grid_XYZW[ 1803][3] =  0.000531112083662;

    Leb_Grid_XYZW[ 1804][0] = -0.542733732205905;
    Leb_Grid_XYZW[ 1804][1] = -0.834265164406713;
    Leb_Grid_XYZW[ 1804][2] =  0.097168571993667;
    Leb_Grid_XYZW[ 1804][3] =  0.000531112083662;

    Leb_Grid_XYZW[ 1805][0] = -0.542733732205905;
    Leb_Grid_XYZW[ 1805][1] = -0.834265164406713;
    Leb_Grid_XYZW[ 1805][2] = -0.097168571993667;
    Leb_Grid_XYZW[ 1805][3] =  0.000531112083662;

    Leb_Grid_XYZW[ 1806][0] =  0.097168571993667;
    Leb_Grid_XYZW[ 1806][1] =  0.542733732205905;
    Leb_Grid_XYZW[ 1806][2] =  0.834265164406713;
    Leb_Grid_XYZW[ 1806][3] =  0.000531112083662;

    Leb_Grid_XYZW[ 1807][0] =  0.097168571993667;
    Leb_Grid_XYZW[ 1807][1] =  0.542733732205905;
    Leb_Grid_XYZW[ 1807][2] = -0.834265164406713;
    Leb_Grid_XYZW[ 1807][3] =  0.000531112083662;

    Leb_Grid_XYZW[ 1808][0] =  0.097168571993667;
    Leb_Grid_XYZW[ 1808][1] = -0.542733732205905;
    Leb_Grid_XYZW[ 1808][2] =  0.834265164406713;
    Leb_Grid_XYZW[ 1808][3] =  0.000531112083662;

    Leb_Grid_XYZW[ 1809][0] =  0.097168571993667;
    Leb_Grid_XYZW[ 1809][1] = -0.542733732205905;
    Leb_Grid_XYZW[ 1809][2] = -0.834265164406713;
    Leb_Grid_XYZW[ 1809][3] =  0.000531112083662;

    Leb_Grid_XYZW[ 1810][0] = -0.097168571993667;
    Leb_Grid_XYZW[ 1810][1] =  0.542733732205905;
    Leb_Grid_XYZW[ 1810][2] =  0.834265164406713;
    Leb_Grid_XYZW[ 1810][3] =  0.000531112083662;

    Leb_Grid_XYZW[ 1811][0] = -0.097168571993667;
    Leb_Grid_XYZW[ 1811][1] =  0.542733732205905;
    Leb_Grid_XYZW[ 1811][2] = -0.834265164406713;
    Leb_Grid_XYZW[ 1811][3] =  0.000531112083662;

    Leb_Grid_XYZW[ 1812][0] = -0.097168571993667;
    Leb_Grid_XYZW[ 1812][1] = -0.542733732205905;
    Leb_Grid_XYZW[ 1812][2] =  0.834265164406713;
    Leb_Grid_XYZW[ 1812][3] =  0.000531112083662;

    Leb_Grid_XYZW[ 1813][0] = -0.097168571993667;
    Leb_Grid_XYZW[ 1813][1] = -0.542733732205905;
    Leb_Grid_XYZW[ 1813][2] = -0.834265164406713;
    Leb_Grid_XYZW[ 1813][3] =  0.000531112083662;

    Leb_Grid_XYZW[ 1814][0] =  0.097168571993667;
    Leb_Grid_XYZW[ 1814][1] =  0.834265164406713;
    Leb_Grid_XYZW[ 1814][2] =  0.542733732205905;
    Leb_Grid_XYZW[ 1814][3] =  0.000531112083662;

    Leb_Grid_XYZW[ 1815][0] =  0.097168571993667;
    Leb_Grid_XYZW[ 1815][1] =  0.834265164406713;
    Leb_Grid_XYZW[ 1815][2] = -0.542733732205905;
    Leb_Grid_XYZW[ 1815][3] =  0.000531112083662;

    Leb_Grid_XYZW[ 1816][0] =  0.097168571993667;
    Leb_Grid_XYZW[ 1816][1] = -0.834265164406713;
    Leb_Grid_XYZW[ 1816][2] =  0.542733732205905;
    Leb_Grid_XYZW[ 1816][3] =  0.000531112083662;

    Leb_Grid_XYZW[ 1817][0] =  0.097168571993667;
    Leb_Grid_XYZW[ 1817][1] = -0.834265164406713;
    Leb_Grid_XYZW[ 1817][2] = -0.542733732205905;
    Leb_Grid_XYZW[ 1817][3] =  0.000531112083662;

    Leb_Grid_XYZW[ 1818][0] = -0.097168571993667;
    Leb_Grid_XYZW[ 1818][1] =  0.834265164406713;
    Leb_Grid_XYZW[ 1818][2] =  0.542733732205905;
    Leb_Grid_XYZW[ 1818][3] =  0.000531112083662;

    Leb_Grid_XYZW[ 1819][0] = -0.097168571993667;
    Leb_Grid_XYZW[ 1819][1] =  0.834265164406713;
    Leb_Grid_XYZW[ 1819][2] = -0.542733732205905;
    Leb_Grid_XYZW[ 1819][3] =  0.000531112083662;

    Leb_Grid_XYZW[ 1820][0] = -0.097168571993667;
    Leb_Grid_XYZW[ 1820][1] = -0.834265164406713;
    Leb_Grid_XYZW[ 1820][2] =  0.542733732205905;
    Leb_Grid_XYZW[ 1820][3] =  0.000531112083662;

    Leb_Grid_XYZW[ 1821][0] = -0.097168571993667;
    Leb_Grid_XYZW[ 1821][1] = -0.834265164406713;
    Leb_Grid_XYZW[ 1821][2] = -0.542733732205905;
    Leb_Grid_XYZW[ 1821][3] =  0.000531112083662;

    Leb_Grid_XYZW[ 1822][0] =  0.834265164406713;
    Leb_Grid_XYZW[ 1822][1] =  0.542733732205905;
    Leb_Grid_XYZW[ 1822][2] =  0.097168571993667;
    Leb_Grid_XYZW[ 1822][3] =  0.000531112083662;

    Leb_Grid_XYZW[ 1823][0] =  0.834265164406713;
    Leb_Grid_XYZW[ 1823][1] =  0.542733732205905;
    Leb_Grid_XYZW[ 1823][2] = -0.097168571993667;
    Leb_Grid_XYZW[ 1823][3] =  0.000531112083662;

    Leb_Grid_XYZW[ 1824][0] =  0.834265164406713;
    Leb_Grid_XYZW[ 1824][1] = -0.542733732205905;
    Leb_Grid_XYZW[ 1824][2] =  0.097168571993667;
    Leb_Grid_XYZW[ 1824][3] =  0.000531112083662;

    Leb_Grid_XYZW[ 1825][0] =  0.834265164406713;
    Leb_Grid_XYZW[ 1825][1] = -0.542733732205905;
    Leb_Grid_XYZW[ 1825][2] = -0.097168571993667;
    Leb_Grid_XYZW[ 1825][3] =  0.000531112083662;

    Leb_Grid_XYZW[ 1826][0] = -0.834265164406713;
    Leb_Grid_XYZW[ 1826][1] =  0.542733732205905;
    Leb_Grid_XYZW[ 1826][2] =  0.097168571993667;
    Leb_Grid_XYZW[ 1826][3] =  0.000531112083662;

    Leb_Grid_XYZW[ 1827][0] = -0.834265164406713;
    Leb_Grid_XYZW[ 1827][1] =  0.542733732205905;
    Leb_Grid_XYZW[ 1827][2] = -0.097168571993667;
    Leb_Grid_XYZW[ 1827][3] =  0.000531112083662;

    Leb_Grid_XYZW[ 1828][0] = -0.834265164406713;
    Leb_Grid_XYZW[ 1828][1] = -0.542733732205905;
    Leb_Grid_XYZW[ 1828][2] =  0.097168571993667;
    Leb_Grid_XYZW[ 1828][3] =  0.000531112083662;

    Leb_Grid_XYZW[ 1829][0] = -0.834265164406713;
    Leb_Grid_XYZW[ 1829][1] = -0.542733732205905;
    Leb_Grid_XYZW[ 1829][2] = -0.097168571993667;
    Leb_Grid_XYZW[ 1829][3] =  0.000531112083662;

    Leb_Grid_XYZW[ 1830][0] =  0.834265164406713;
    Leb_Grid_XYZW[ 1830][1] =  0.097168571993666;
    Leb_Grid_XYZW[ 1830][2] =  0.542733732205905;
    Leb_Grid_XYZW[ 1830][3] =  0.000531112083662;

    Leb_Grid_XYZW[ 1831][0] =  0.834265164406713;
    Leb_Grid_XYZW[ 1831][1] =  0.097168571993666;
    Leb_Grid_XYZW[ 1831][2] = -0.542733732205905;
    Leb_Grid_XYZW[ 1831][3] =  0.000531112083662;

    Leb_Grid_XYZW[ 1832][0] =  0.834265164406713;
    Leb_Grid_XYZW[ 1832][1] = -0.097168571993666;
    Leb_Grid_XYZW[ 1832][2] =  0.542733732205905;
    Leb_Grid_XYZW[ 1832][3] =  0.000531112083662;

    Leb_Grid_XYZW[ 1833][0] =  0.834265164406713;
    Leb_Grid_XYZW[ 1833][1] = -0.097168571993666;
    Leb_Grid_XYZW[ 1833][2] = -0.542733732205905;
    Leb_Grid_XYZW[ 1833][3] =  0.000531112083662;

    Leb_Grid_XYZW[ 1834][0] = -0.834265164406713;
    Leb_Grid_XYZW[ 1834][1] =  0.097168571993666;
    Leb_Grid_XYZW[ 1834][2] =  0.542733732205905;
    Leb_Grid_XYZW[ 1834][3] =  0.000531112083662;

    Leb_Grid_XYZW[ 1835][0] = -0.834265164406713;
    Leb_Grid_XYZW[ 1835][1] =  0.097168571993666;
    Leb_Grid_XYZW[ 1835][2] = -0.542733732205905;
    Leb_Grid_XYZW[ 1835][3] =  0.000531112083662;

    Leb_Grid_XYZW[ 1836][0] = -0.834265164406713;
    Leb_Grid_XYZW[ 1836][1] = -0.097168571993666;
    Leb_Grid_XYZW[ 1836][2] =  0.542733732205905;
    Leb_Grid_XYZW[ 1836][3] =  0.000531112083662;

    Leb_Grid_XYZW[ 1837][0] = -0.834265164406713;
    Leb_Grid_XYZW[ 1837][1] = -0.097168571993666;
    Leb_Grid_XYZW[ 1837][2] = -0.542733732205905;
    Leb_Grid_XYZW[ 1837][3] =  0.000531112083662;

    Leb_Grid_XYZW[ 1838][0] =  0.594349374724670;
    Leb_Grid_XYZW[ 1838][1] =  0.146520583979506;
    Leb_Grid_XYZW[ 1838][2] =  0.790746823727227;
    Leb_Grid_XYZW[ 1838][3] =  0.000538430931996;

    Leb_Grid_XYZW[ 1839][0] =  0.594349374724670;
    Leb_Grid_XYZW[ 1839][1] =  0.146520583979506;
    Leb_Grid_XYZW[ 1839][2] = -0.790746823727227;
    Leb_Grid_XYZW[ 1839][3] =  0.000538430931996;

    Leb_Grid_XYZW[ 1840][0] =  0.594349374724670;
    Leb_Grid_XYZW[ 1840][1] = -0.146520583979506;
    Leb_Grid_XYZW[ 1840][2] =  0.790746823727227;
    Leb_Grid_XYZW[ 1840][3] =  0.000538430931996;

    Leb_Grid_XYZW[ 1841][0] =  0.594349374724670;
    Leb_Grid_XYZW[ 1841][1] = -0.146520583979506;
    Leb_Grid_XYZW[ 1841][2] = -0.790746823727227;
    Leb_Grid_XYZW[ 1841][3] =  0.000538430931996;

    Leb_Grid_XYZW[ 1842][0] = -0.594349374724670;
    Leb_Grid_XYZW[ 1842][1] =  0.146520583979506;
    Leb_Grid_XYZW[ 1842][2] =  0.790746823727227;
    Leb_Grid_XYZW[ 1842][3] =  0.000538430931996;

    Leb_Grid_XYZW[ 1843][0] = -0.594349374724670;
    Leb_Grid_XYZW[ 1843][1] =  0.146520583979506;
    Leb_Grid_XYZW[ 1843][2] = -0.790746823727227;
    Leb_Grid_XYZW[ 1843][3] =  0.000538430931996;

    Leb_Grid_XYZW[ 1844][0] = -0.594349374724670;
    Leb_Grid_XYZW[ 1844][1] = -0.146520583979506;
    Leb_Grid_XYZW[ 1844][2] =  0.790746823727227;
    Leb_Grid_XYZW[ 1844][3] =  0.000538430931996;

    Leb_Grid_XYZW[ 1845][0] = -0.594349374724670;
    Leb_Grid_XYZW[ 1845][1] = -0.146520583979506;
    Leb_Grid_XYZW[ 1845][2] = -0.790746823727227;
    Leb_Grid_XYZW[ 1845][3] =  0.000538430931996;

    Leb_Grid_XYZW[ 1846][0] =  0.594349374724670;
    Leb_Grid_XYZW[ 1846][1] =  0.790746823727227;
    Leb_Grid_XYZW[ 1846][2] =  0.146520583979505;
    Leb_Grid_XYZW[ 1846][3] =  0.000538430931996;

    Leb_Grid_XYZW[ 1847][0] =  0.594349374724670;
    Leb_Grid_XYZW[ 1847][1] =  0.790746823727227;
    Leb_Grid_XYZW[ 1847][2] = -0.146520583979505;
    Leb_Grid_XYZW[ 1847][3] =  0.000538430931996;

    Leb_Grid_XYZW[ 1848][0] =  0.594349374724670;
    Leb_Grid_XYZW[ 1848][1] = -0.790746823727227;
    Leb_Grid_XYZW[ 1848][2] =  0.146520583979505;
    Leb_Grid_XYZW[ 1848][3] =  0.000538430931996;

    Leb_Grid_XYZW[ 1849][0] =  0.594349374724670;
    Leb_Grid_XYZW[ 1849][1] = -0.790746823727227;
    Leb_Grid_XYZW[ 1849][2] = -0.146520583979505;
    Leb_Grid_XYZW[ 1849][3] =  0.000538430931996;

    Leb_Grid_XYZW[ 1850][0] = -0.594349374724670;
    Leb_Grid_XYZW[ 1850][1] =  0.790746823727227;
    Leb_Grid_XYZW[ 1850][2] =  0.146520583979505;
    Leb_Grid_XYZW[ 1850][3] =  0.000538430931996;

    Leb_Grid_XYZW[ 1851][0] = -0.594349374724670;
    Leb_Grid_XYZW[ 1851][1] =  0.790746823727227;
    Leb_Grid_XYZW[ 1851][2] = -0.146520583979505;
    Leb_Grid_XYZW[ 1851][3] =  0.000538430931996;

    Leb_Grid_XYZW[ 1852][0] = -0.594349374724670;
    Leb_Grid_XYZW[ 1852][1] = -0.790746823727227;
    Leb_Grid_XYZW[ 1852][2] =  0.146520583979505;
    Leb_Grid_XYZW[ 1852][3] =  0.000538430931996;

    Leb_Grid_XYZW[ 1853][0] = -0.594349374724670;
    Leb_Grid_XYZW[ 1853][1] = -0.790746823727227;
    Leb_Grid_XYZW[ 1853][2] = -0.146520583979505;
    Leb_Grid_XYZW[ 1853][3] =  0.000538430931996;

    Leb_Grid_XYZW[ 1854][0] =  0.146520583979506;
    Leb_Grid_XYZW[ 1854][1] =  0.594349374724670;
    Leb_Grid_XYZW[ 1854][2] =  0.790746823727227;
    Leb_Grid_XYZW[ 1854][3] =  0.000538430931996;

    Leb_Grid_XYZW[ 1855][0] =  0.146520583979506;
    Leb_Grid_XYZW[ 1855][1] =  0.594349374724670;
    Leb_Grid_XYZW[ 1855][2] = -0.790746823727227;
    Leb_Grid_XYZW[ 1855][3] =  0.000538430931996;

    Leb_Grid_XYZW[ 1856][0] =  0.146520583979506;
    Leb_Grid_XYZW[ 1856][1] = -0.594349374724670;
    Leb_Grid_XYZW[ 1856][2] =  0.790746823727227;
    Leb_Grid_XYZW[ 1856][3] =  0.000538430931996;

    Leb_Grid_XYZW[ 1857][0] =  0.146520583979506;
    Leb_Grid_XYZW[ 1857][1] = -0.594349374724670;
    Leb_Grid_XYZW[ 1857][2] = -0.790746823727227;
    Leb_Grid_XYZW[ 1857][3] =  0.000538430931996;

    Leb_Grid_XYZW[ 1858][0] = -0.146520583979506;
    Leb_Grid_XYZW[ 1858][1] =  0.594349374724670;
    Leb_Grid_XYZW[ 1858][2] =  0.790746823727227;
    Leb_Grid_XYZW[ 1858][3] =  0.000538430931996;

    Leb_Grid_XYZW[ 1859][0] = -0.146520583979506;
    Leb_Grid_XYZW[ 1859][1] =  0.594349374724670;
    Leb_Grid_XYZW[ 1859][2] = -0.790746823727227;
    Leb_Grid_XYZW[ 1859][3] =  0.000538430931996;

    Leb_Grid_XYZW[ 1860][0] = -0.146520583979506;
    Leb_Grid_XYZW[ 1860][1] = -0.594349374724670;
    Leb_Grid_XYZW[ 1860][2] =  0.790746823727227;
    Leb_Grid_XYZW[ 1860][3] =  0.000538430931996;

    Leb_Grid_XYZW[ 1861][0] = -0.146520583979506;
    Leb_Grid_XYZW[ 1861][1] = -0.594349374724670;
    Leb_Grid_XYZW[ 1861][2] = -0.790746823727227;
    Leb_Grid_XYZW[ 1861][3] =  0.000538430931996;

    Leb_Grid_XYZW[ 1862][0] =  0.146520583979505;
    Leb_Grid_XYZW[ 1862][1] =  0.790746823727227;
    Leb_Grid_XYZW[ 1862][2] =  0.594349374724670;
    Leb_Grid_XYZW[ 1862][3] =  0.000538430931996;

    Leb_Grid_XYZW[ 1863][0] =  0.146520583979505;
    Leb_Grid_XYZW[ 1863][1] =  0.790746823727227;
    Leb_Grid_XYZW[ 1863][2] = -0.594349374724670;
    Leb_Grid_XYZW[ 1863][3] =  0.000538430931996;

    Leb_Grid_XYZW[ 1864][0] =  0.146520583979505;
    Leb_Grid_XYZW[ 1864][1] = -0.790746823727227;
    Leb_Grid_XYZW[ 1864][2] =  0.594349374724670;
    Leb_Grid_XYZW[ 1864][3] =  0.000538430931996;

    Leb_Grid_XYZW[ 1865][0] =  0.146520583979505;
    Leb_Grid_XYZW[ 1865][1] = -0.790746823727227;
    Leb_Grid_XYZW[ 1865][2] = -0.594349374724670;
    Leb_Grid_XYZW[ 1865][3] =  0.000538430931996;

    Leb_Grid_XYZW[ 1866][0] = -0.146520583979505;
    Leb_Grid_XYZW[ 1866][1] =  0.790746823727227;
    Leb_Grid_XYZW[ 1866][2] =  0.594349374724670;
    Leb_Grid_XYZW[ 1866][3] =  0.000538430931996;

    Leb_Grid_XYZW[ 1867][0] = -0.146520583979505;
    Leb_Grid_XYZW[ 1867][1] =  0.790746823727227;
    Leb_Grid_XYZW[ 1867][2] = -0.594349374724670;
    Leb_Grid_XYZW[ 1867][3] =  0.000538430931996;

    Leb_Grid_XYZW[ 1868][0] = -0.146520583979505;
    Leb_Grid_XYZW[ 1868][1] = -0.790746823727227;
    Leb_Grid_XYZW[ 1868][2] =  0.594349374724670;
    Leb_Grid_XYZW[ 1868][3] =  0.000538430931996;

    Leb_Grid_XYZW[ 1869][0] = -0.146520583979505;
    Leb_Grid_XYZW[ 1869][1] = -0.790746823727227;
    Leb_Grid_XYZW[ 1869][2] = -0.594349374724670;
    Leb_Grid_XYZW[ 1869][3] =  0.000538430931996;

    Leb_Grid_XYZW[ 1870][0] =  0.790746823727227;
    Leb_Grid_XYZW[ 1870][1] =  0.594349374724670;
    Leb_Grid_XYZW[ 1870][2] =  0.146520583979505;
    Leb_Grid_XYZW[ 1870][3] =  0.000538430931996;

    Leb_Grid_XYZW[ 1871][0] =  0.790746823727227;
    Leb_Grid_XYZW[ 1871][1] =  0.594349374724670;
    Leb_Grid_XYZW[ 1871][2] = -0.146520583979505;
    Leb_Grid_XYZW[ 1871][3] =  0.000538430931996;

    Leb_Grid_XYZW[ 1872][0] =  0.790746823727227;
    Leb_Grid_XYZW[ 1872][1] = -0.594349374724670;
    Leb_Grid_XYZW[ 1872][2] =  0.146520583979505;
    Leb_Grid_XYZW[ 1872][3] =  0.000538430931996;

    Leb_Grid_XYZW[ 1873][0] =  0.790746823727227;
    Leb_Grid_XYZW[ 1873][1] = -0.594349374724670;
    Leb_Grid_XYZW[ 1873][2] = -0.146520583979505;
    Leb_Grid_XYZW[ 1873][3] =  0.000538430931996;

    Leb_Grid_XYZW[ 1874][0] = -0.790746823727228;
    Leb_Grid_XYZW[ 1874][1] =  0.594349374724669;
    Leb_Grid_XYZW[ 1874][2] =  0.146520583979505;
    Leb_Grid_XYZW[ 1874][3] =  0.000538430931996;

    Leb_Grid_XYZW[ 1875][0] = -0.790746823727228;
    Leb_Grid_XYZW[ 1875][1] =  0.594349374724669;
    Leb_Grid_XYZW[ 1875][2] = -0.146520583979505;
    Leb_Grid_XYZW[ 1875][3] =  0.000538430931996;

    Leb_Grid_XYZW[ 1876][0] = -0.790746823727228;
    Leb_Grid_XYZW[ 1876][1] = -0.594349374724669;
    Leb_Grid_XYZW[ 1876][2] =  0.146520583979505;
    Leb_Grid_XYZW[ 1876][3] =  0.000538430931996;

    Leb_Grid_XYZW[ 1877][0] = -0.790746823727228;
    Leb_Grid_XYZW[ 1877][1] = -0.594349374724669;
    Leb_Grid_XYZW[ 1877][2] = -0.146520583979505;
    Leb_Grid_XYZW[ 1877][3] =  0.000538430931996;

    Leb_Grid_XYZW[ 1878][0] =  0.790746823727227;
    Leb_Grid_XYZW[ 1878][1] =  0.146520583979506;
    Leb_Grid_XYZW[ 1878][2] =  0.594349374724670;
    Leb_Grid_XYZW[ 1878][3] =  0.000538430931996;

    Leb_Grid_XYZW[ 1879][0] =  0.790746823727227;
    Leb_Grid_XYZW[ 1879][1] =  0.146520583979506;
    Leb_Grid_XYZW[ 1879][2] = -0.594349374724670;
    Leb_Grid_XYZW[ 1879][3] =  0.000538430931996;

    Leb_Grid_XYZW[ 1880][0] =  0.790746823727227;
    Leb_Grid_XYZW[ 1880][1] = -0.146520583979506;
    Leb_Grid_XYZW[ 1880][2] =  0.594349374724670;
    Leb_Grid_XYZW[ 1880][3] =  0.000538430931996;

    Leb_Grid_XYZW[ 1881][0] =  0.790746823727227;
    Leb_Grid_XYZW[ 1881][1] = -0.146520583979506;
    Leb_Grid_XYZW[ 1881][2] = -0.594349374724670;
    Leb_Grid_XYZW[ 1881][3] =  0.000538430931996;

    Leb_Grid_XYZW[ 1882][0] = -0.790746823727227;
    Leb_Grid_XYZW[ 1882][1] =  0.146520583979506;
    Leb_Grid_XYZW[ 1882][2] =  0.594349374724670;
    Leb_Grid_XYZW[ 1882][3] =  0.000538430931996;

    Leb_Grid_XYZW[ 1883][0] = -0.790746823727227;
    Leb_Grid_XYZW[ 1883][1] =  0.146520583979506;
    Leb_Grid_XYZW[ 1883][2] = -0.594349374724670;
    Leb_Grid_XYZW[ 1883][3] =  0.000538430931996;

    Leb_Grid_XYZW[ 1884][0] = -0.790746823727227;
    Leb_Grid_XYZW[ 1884][1] = -0.146520583979506;
    Leb_Grid_XYZW[ 1884][2] =  0.594349374724670;
    Leb_Grid_XYZW[ 1884][3] =  0.000538430931996;

    Leb_Grid_XYZW[ 1885][0] = -0.790746823727227;
    Leb_Grid_XYZW[ 1885][1] = -0.146520583979506;
    Leb_Grid_XYZW[ 1885][2] = -0.594349374724670;
    Leb_Grid_XYZW[ 1885][3] =  0.000538430931996;

    Leb_Grid_XYZW[ 1886][0] =  0.642131403356494;
    Leb_Grid_XYZW[ 1886][1] =  0.195357944980358;
    Leb_Grid_XYZW[ 1886][2] =  0.741284381432977;
    Leb_Grid_XYZW[ 1886][3] =  0.000542185950405;

    Leb_Grid_XYZW[ 1887][0] =  0.642131403356494;
    Leb_Grid_XYZW[ 1887][1] =  0.195357944980358;
    Leb_Grid_XYZW[ 1887][2] = -0.741284381432977;
    Leb_Grid_XYZW[ 1887][3] =  0.000542185950405;

    Leb_Grid_XYZW[ 1888][0] =  0.642131403356494;
    Leb_Grid_XYZW[ 1888][1] = -0.195357944980358;
    Leb_Grid_XYZW[ 1888][2] =  0.741284381432977;
    Leb_Grid_XYZW[ 1888][3] =  0.000542185950405;

    Leb_Grid_XYZW[ 1889][0] =  0.642131403356494;
    Leb_Grid_XYZW[ 1889][1] = -0.195357944980358;
    Leb_Grid_XYZW[ 1889][2] = -0.741284381432977;
    Leb_Grid_XYZW[ 1889][3] =  0.000542185950405;

    Leb_Grid_XYZW[ 1890][0] = -0.642131403356494;
    Leb_Grid_XYZW[ 1890][1] =  0.195357944980358;
    Leb_Grid_XYZW[ 1890][2] =  0.741284381432977;
    Leb_Grid_XYZW[ 1890][3] =  0.000542185950405;

    Leb_Grid_XYZW[ 1891][0] = -0.642131403356494;
    Leb_Grid_XYZW[ 1891][1] =  0.195357944980358;
    Leb_Grid_XYZW[ 1891][2] = -0.741284381432977;
    Leb_Grid_XYZW[ 1891][3] =  0.000542185950405;

    Leb_Grid_XYZW[ 1892][0] = -0.642131403356494;
    Leb_Grid_XYZW[ 1892][1] = -0.195357944980358;
    Leb_Grid_XYZW[ 1892][2] =  0.741284381432977;
    Leb_Grid_XYZW[ 1892][3] =  0.000542185950405;

    Leb_Grid_XYZW[ 1893][0] = -0.642131403356494;
    Leb_Grid_XYZW[ 1893][1] = -0.195357944980358;
    Leb_Grid_XYZW[ 1893][2] = -0.741284381432977;
    Leb_Grid_XYZW[ 1893][3] =  0.000542185950405;

    Leb_Grid_XYZW[ 1894][0] =  0.642131403356494;
    Leb_Grid_XYZW[ 1894][1] =  0.741284381432977;
    Leb_Grid_XYZW[ 1894][2] =  0.195357944980357;
    Leb_Grid_XYZW[ 1894][3] =  0.000542185950405;

    Leb_Grid_XYZW[ 1895][0] =  0.642131403356494;
    Leb_Grid_XYZW[ 1895][1] =  0.741284381432977;
    Leb_Grid_XYZW[ 1895][2] = -0.195357944980357;
    Leb_Grid_XYZW[ 1895][3] =  0.000542185950405;

    Leb_Grid_XYZW[ 1896][0] =  0.642131403356494;
    Leb_Grid_XYZW[ 1896][1] = -0.741284381432977;
    Leb_Grid_XYZW[ 1896][2] =  0.195357944980357;
    Leb_Grid_XYZW[ 1896][3] =  0.000542185950405;

    Leb_Grid_XYZW[ 1897][0] =  0.642131403356494;
    Leb_Grid_XYZW[ 1897][1] = -0.741284381432977;
    Leb_Grid_XYZW[ 1897][2] = -0.195357944980357;
    Leb_Grid_XYZW[ 1897][3] =  0.000542185950405;

    Leb_Grid_XYZW[ 1898][0] = -0.642131403356495;
    Leb_Grid_XYZW[ 1898][1] =  0.741284381432976;
    Leb_Grid_XYZW[ 1898][2] =  0.195357944980357;
    Leb_Grid_XYZW[ 1898][3] =  0.000542185950405;

    Leb_Grid_XYZW[ 1899][0] = -0.642131403356495;
    Leb_Grid_XYZW[ 1899][1] =  0.741284381432976;
    Leb_Grid_XYZW[ 1899][2] = -0.195357944980357;
    Leb_Grid_XYZW[ 1899][3] =  0.000542185950405;

    Leb_Grid_XYZW[ 1900][0] = -0.642131403356495;
    Leb_Grid_XYZW[ 1900][1] = -0.741284381432976;
    Leb_Grid_XYZW[ 1900][2] =  0.195357944980357;
    Leb_Grid_XYZW[ 1900][3] =  0.000542185950405;

    Leb_Grid_XYZW[ 1901][0] = -0.642131403356495;
    Leb_Grid_XYZW[ 1901][1] = -0.741284381432976;
    Leb_Grid_XYZW[ 1901][2] = -0.195357944980357;
    Leb_Grid_XYZW[ 1901][3] =  0.000542185950405;

    Leb_Grid_XYZW[ 1902][0] =  0.195357944980357;
    Leb_Grid_XYZW[ 1902][1] =  0.642131403356494;
    Leb_Grid_XYZW[ 1902][2] =  0.741284381432977;
    Leb_Grid_XYZW[ 1902][3] =  0.000542185950405;

    Leb_Grid_XYZW[ 1903][0] =  0.195357944980357;
    Leb_Grid_XYZW[ 1903][1] =  0.642131403356494;
    Leb_Grid_XYZW[ 1903][2] = -0.741284381432977;
    Leb_Grid_XYZW[ 1903][3] =  0.000542185950405;

    Leb_Grid_XYZW[ 1904][0] =  0.195357944980357;
    Leb_Grid_XYZW[ 1904][1] = -0.642131403356494;
    Leb_Grid_XYZW[ 1904][2] =  0.741284381432977;
    Leb_Grid_XYZW[ 1904][3] =  0.000542185950405;

    Leb_Grid_XYZW[ 1905][0] =  0.195357944980357;
    Leb_Grid_XYZW[ 1905][1] = -0.642131403356494;
    Leb_Grid_XYZW[ 1905][2] = -0.741284381432977;
    Leb_Grid_XYZW[ 1905][3] =  0.000542185950405;

    Leb_Grid_XYZW[ 1906][0] = -0.195357944980357;
    Leb_Grid_XYZW[ 1906][1] =  0.642131403356494;
    Leb_Grid_XYZW[ 1906][2] =  0.741284381432977;
    Leb_Grid_XYZW[ 1906][3] =  0.000542185950405;

    Leb_Grid_XYZW[ 1907][0] = -0.195357944980357;
    Leb_Grid_XYZW[ 1907][1] =  0.642131403356494;
    Leb_Grid_XYZW[ 1907][2] = -0.741284381432977;
    Leb_Grid_XYZW[ 1907][3] =  0.000542185950405;

    Leb_Grid_XYZW[ 1908][0] = -0.195357944980357;
    Leb_Grid_XYZW[ 1908][1] = -0.642131403356494;
    Leb_Grid_XYZW[ 1908][2] =  0.741284381432977;
    Leb_Grid_XYZW[ 1908][3] =  0.000542185950405;

    Leb_Grid_XYZW[ 1909][0] = -0.195357944980357;
    Leb_Grid_XYZW[ 1909][1] = -0.642131403356494;
    Leb_Grid_XYZW[ 1909][2] = -0.741284381432977;
    Leb_Grid_XYZW[ 1909][3] =  0.000542185950405;

    Leb_Grid_XYZW[ 1910][0] =  0.195357944980357;
    Leb_Grid_XYZW[ 1910][1] =  0.741284381432977;
    Leb_Grid_XYZW[ 1910][2] =  0.642131403356494;
    Leb_Grid_XYZW[ 1910][3] =  0.000542185950405;

    Leb_Grid_XYZW[ 1911][0] =  0.195357944980357;
    Leb_Grid_XYZW[ 1911][1] =  0.741284381432977;
    Leb_Grid_XYZW[ 1911][2] = -0.642131403356494;
    Leb_Grid_XYZW[ 1911][3] =  0.000542185950405;

    Leb_Grid_XYZW[ 1912][0] =  0.195357944980357;
    Leb_Grid_XYZW[ 1912][1] = -0.741284381432977;
    Leb_Grid_XYZW[ 1912][2] =  0.642131403356494;
    Leb_Grid_XYZW[ 1912][3] =  0.000542185950405;

    Leb_Grid_XYZW[ 1913][0] =  0.195357944980357;
    Leb_Grid_XYZW[ 1913][1] = -0.741284381432977;
    Leb_Grid_XYZW[ 1913][2] = -0.642131403356494;
    Leb_Grid_XYZW[ 1913][3] =  0.000542185950405;

    Leb_Grid_XYZW[ 1914][0] = -0.195357944980357;
    Leb_Grid_XYZW[ 1914][1] =  0.741284381432977;
    Leb_Grid_XYZW[ 1914][2] =  0.642131403356494;
    Leb_Grid_XYZW[ 1914][3] =  0.000542185950405;

    Leb_Grid_XYZW[ 1915][0] = -0.195357944980357;
    Leb_Grid_XYZW[ 1915][1] =  0.741284381432977;
    Leb_Grid_XYZW[ 1915][2] = -0.642131403356494;
    Leb_Grid_XYZW[ 1915][3] =  0.000542185950405;

    Leb_Grid_XYZW[ 1916][0] = -0.195357944980357;
    Leb_Grid_XYZW[ 1916][1] = -0.741284381432977;
    Leb_Grid_XYZW[ 1916][2] =  0.642131403356494;
    Leb_Grid_XYZW[ 1916][3] =  0.000542185950405;

    Leb_Grid_XYZW[ 1917][0] = -0.195357944980357;
    Leb_Grid_XYZW[ 1917][1] = -0.741284381432977;
    Leb_Grid_XYZW[ 1917][2] = -0.642131403356494;
    Leb_Grid_XYZW[ 1917][3] =  0.000542185950405;

    Leb_Grid_XYZW[ 1918][0] =  0.741284381432977;
    Leb_Grid_XYZW[ 1918][1] =  0.642131403356494;
    Leb_Grid_XYZW[ 1918][2] =  0.195357944980357;
    Leb_Grid_XYZW[ 1918][3] =  0.000542185950405;

    Leb_Grid_XYZW[ 1919][0] =  0.741284381432977;
    Leb_Grid_XYZW[ 1919][1] =  0.642131403356494;
    Leb_Grid_XYZW[ 1919][2] = -0.195357944980357;
    Leb_Grid_XYZW[ 1919][3] =  0.000542185950405;

    Leb_Grid_XYZW[ 1920][0] =  0.741284381432977;
    Leb_Grid_XYZW[ 1920][1] = -0.642131403356494;
    Leb_Grid_XYZW[ 1920][2] =  0.195357944980357;
    Leb_Grid_XYZW[ 1920][3] =  0.000542185950405;

    Leb_Grid_XYZW[ 1921][0] =  0.741284381432977;
    Leb_Grid_XYZW[ 1921][1] = -0.642131403356494;
    Leb_Grid_XYZW[ 1921][2] = -0.195357944980357;
    Leb_Grid_XYZW[ 1921][3] =  0.000542185950405;

    Leb_Grid_XYZW[ 1922][0] = -0.741284381432977;
    Leb_Grid_XYZW[ 1922][1] =  0.642131403356494;
    Leb_Grid_XYZW[ 1922][2] =  0.195357944980357;
    Leb_Grid_XYZW[ 1922][3] =  0.000542185950405;

    Leb_Grid_XYZW[ 1923][0] = -0.741284381432977;
    Leb_Grid_XYZW[ 1923][1] =  0.642131403356494;
    Leb_Grid_XYZW[ 1923][2] = -0.195357944980357;
    Leb_Grid_XYZW[ 1923][3] =  0.000542185950405;

    Leb_Grid_XYZW[ 1924][0] = -0.741284381432977;
    Leb_Grid_XYZW[ 1924][1] = -0.642131403356494;
    Leb_Grid_XYZW[ 1924][2] =  0.195357944980357;
    Leb_Grid_XYZW[ 1924][3] =  0.000542185950405;

    Leb_Grid_XYZW[ 1925][0] = -0.741284381432977;
    Leb_Grid_XYZW[ 1925][1] = -0.642131403356494;
    Leb_Grid_XYZW[ 1925][2] = -0.195357944980357;
    Leb_Grid_XYZW[ 1925][3] =  0.000542185950405;

    Leb_Grid_XYZW[ 1926][0] =  0.741284381432977;
    Leb_Grid_XYZW[ 1926][1] =  0.195357944980357;
    Leb_Grid_XYZW[ 1926][2] =  0.642131403356494;
    Leb_Grid_XYZW[ 1926][3] =  0.000542185950405;

    Leb_Grid_XYZW[ 1927][0] =  0.741284381432977;
    Leb_Grid_XYZW[ 1927][1] =  0.195357944980357;
    Leb_Grid_XYZW[ 1927][2] = -0.642131403356494;
    Leb_Grid_XYZW[ 1927][3] =  0.000542185950405;

    Leb_Grid_XYZW[ 1928][0] =  0.741284381432977;
    Leb_Grid_XYZW[ 1928][1] = -0.195357944980357;
    Leb_Grid_XYZW[ 1928][2] =  0.642131403356494;
    Leb_Grid_XYZW[ 1928][3] =  0.000542185950405;

    Leb_Grid_XYZW[ 1929][0] =  0.741284381432977;
    Leb_Grid_XYZW[ 1929][1] = -0.195357944980357;
    Leb_Grid_XYZW[ 1929][2] = -0.642131403356494;
    Leb_Grid_XYZW[ 1929][3] =  0.000542185950405;

    Leb_Grid_XYZW[ 1930][0] = -0.741284381432977;
    Leb_Grid_XYZW[ 1930][1] =  0.195357944980357;
    Leb_Grid_XYZW[ 1930][2] =  0.642131403356494;
    Leb_Grid_XYZW[ 1930][3] =  0.000542185950405;

    Leb_Grid_XYZW[ 1931][0] = -0.741284381432977;
    Leb_Grid_XYZW[ 1931][1] =  0.195357944980357;
    Leb_Grid_XYZW[ 1931][2] = -0.642131403356494;
    Leb_Grid_XYZW[ 1931][3] =  0.000542185950405;

    Leb_Grid_XYZW[ 1932][0] = -0.741284381432977;
    Leb_Grid_XYZW[ 1932][1] = -0.195357944980357;
    Leb_Grid_XYZW[ 1932][2] =  0.642131403356494;
    Leb_Grid_XYZW[ 1932][3] =  0.000542185950405;

    Leb_Grid_XYZW[ 1933][0] = -0.741284381432977;
    Leb_Grid_XYZW[ 1933][1] = -0.195357944980357;
    Leb_Grid_XYZW[ 1933][2] = -0.642131403356494;
    Leb_Grid_XYZW[ 1933][3] =  0.000542185950405;

    Leb_Grid_XYZW[ 1934][0] =  0.602062837471398;
    Leb_Grid_XYZW[ 1934][1] =  0.049163750157382;
    Leb_Grid_XYZW[ 1934][2] =  0.796933664370098;
    Leb_Grid_XYZW[ 1934][3] =  0.000539094835505;

    Leb_Grid_XYZW[ 1935][0] =  0.602062837471398;
    Leb_Grid_XYZW[ 1935][1] =  0.049163750157382;
    Leb_Grid_XYZW[ 1935][2] = -0.796933664370098;
    Leb_Grid_XYZW[ 1935][3] =  0.000539094835505;

    Leb_Grid_XYZW[ 1936][0] =  0.602062837471398;
    Leb_Grid_XYZW[ 1936][1] = -0.049163750157382;
    Leb_Grid_XYZW[ 1936][2] =  0.796933664370098;
    Leb_Grid_XYZW[ 1936][3] =  0.000539094835505;

    Leb_Grid_XYZW[ 1937][0] =  0.602062837471398;
    Leb_Grid_XYZW[ 1937][1] = -0.049163750157382;
    Leb_Grid_XYZW[ 1937][2] = -0.796933664370098;
    Leb_Grid_XYZW[ 1937][3] =  0.000539094835505;

    Leb_Grid_XYZW[ 1938][0] = -0.602062837471398;
    Leb_Grid_XYZW[ 1938][1] =  0.049163750157382;
    Leb_Grid_XYZW[ 1938][2] =  0.796933664370098;
    Leb_Grid_XYZW[ 1938][3] =  0.000539094835505;

    Leb_Grid_XYZW[ 1939][0] = -0.602062837471398;
    Leb_Grid_XYZW[ 1939][1] =  0.049163750157382;
    Leb_Grid_XYZW[ 1939][2] = -0.796933664370098;
    Leb_Grid_XYZW[ 1939][3] =  0.000539094835505;

    Leb_Grid_XYZW[ 1940][0] = -0.602062837471398;
    Leb_Grid_XYZW[ 1940][1] = -0.049163750157382;
    Leb_Grid_XYZW[ 1940][2] =  0.796933664370098;
    Leb_Grid_XYZW[ 1940][3] =  0.000539094835505;

    Leb_Grid_XYZW[ 1941][0] = -0.602062837471398;
    Leb_Grid_XYZW[ 1941][1] = -0.049163750157382;
    Leb_Grid_XYZW[ 1941][2] = -0.796933664370098;
    Leb_Grid_XYZW[ 1941][3] =  0.000539094835505;

    Leb_Grid_XYZW[ 1942][0] =  0.602062837471398;
    Leb_Grid_XYZW[ 1942][1] =  0.796933664370098;
    Leb_Grid_XYZW[ 1942][2] =  0.049163750157381;
    Leb_Grid_XYZW[ 1942][3] =  0.000539094835505;

    Leb_Grid_XYZW[ 1943][0] =  0.602062837471398;
    Leb_Grid_XYZW[ 1943][1] =  0.796933664370098;
    Leb_Grid_XYZW[ 1943][2] = -0.049163750157381;
    Leb_Grid_XYZW[ 1943][3] =  0.000539094835505;

    Leb_Grid_XYZW[ 1944][0] =  0.602062837471398;
    Leb_Grid_XYZW[ 1944][1] = -0.796933664370098;
    Leb_Grid_XYZW[ 1944][2] =  0.049163750157381;
    Leb_Grid_XYZW[ 1944][3] =  0.000539094835505;

    Leb_Grid_XYZW[ 1945][0] =  0.602062837471398;
    Leb_Grid_XYZW[ 1945][1] = -0.796933664370098;
    Leb_Grid_XYZW[ 1945][2] = -0.049163750157381;
    Leb_Grid_XYZW[ 1945][3] =  0.000539094835505;

    Leb_Grid_XYZW[ 1946][0] = -0.602062837471398;
    Leb_Grid_XYZW[ 1946][1] =  0.796933664370098;
    Leb_Grid_XYZW[ 1946][2] =  0.049163750157381;
    Leb_Grid_XYZW[ 1946][3] =  0.000539094835505;

    Leb_Grid_XYZW[ 1947][0] = -0.602062837471398;
    Leb_Grid_XYZW[ 1947][1] =  0.796933664370098;
    Leb_Grid_XYZW[ 1947][2] = -0.049163750157381;
    Leb_Grid_XYZW[ 1947][3] =  0.000539094835505;

    Leb_Grid_XYZW[ 1948][0] = -0.602062837471398;
    Leb_Grid_XYZW[ 1948][1] = -0.796933664370098;
    Leb_Grid_XYZW[ 1948][2] =  0.049163750157381;
    Leb_Grid_XYZW[ 1948][3] =  0.000539094835505;

    Leb_Grid_XYZW[ 1949][0] = -0.602062837471398;
    Leb_Grid_XYZW[ 1949][1] = -0.796933664370098;
    Leb_Grid_XYZW[ 1949][2] = -0.049163750157381;
    Leb_Grid_XYZW[ 1949][3] =  0.000539094835505;

    Leb_Grid_XYZW[ 1950][0] =  0.049163750157381;
    Leb_Grid_XYZW[ 1950][1] =  0.602062837471398;
    Leb_Grid_XYZW[ 1950][2] =  0.796933664370098;
    Leb_Grid_XYZW[ 1950][3] =  0.000539094835505;

    Leb_Grid_XYZW[ 1951][0] =  0.049163750157381;
    Leb_Grid_XYZW[ 1951][1] =  0.602062837471398;
    Leb_Grid_XYZW[ 1951][2] = -0.796933664370098;
    Leb_Grid_XYZW[ 1951][3] =  0.000539094835505;

    Leb_Grid_XYZW[ 1952][0] =  0.049163750157381;
    Leb_Grid_XYZW[ 1952][1] = -0.602062837471398;
    Leb_Grid_XYZW[ 1952][2] =  0.796933664370098;
    Leb_Grid_XYZW[ 1952][3] =  0.000539094835505;

    Leb_Grid_XYZW[ 1953][0] =  0.049163750157381;
    Leb_Grid_XYZW[ 1953][1] = -0.602062837471398;
    Leb_Grid_XYZW[ 1953][2] = -0.796933664370098;
    Leb_Grid_XYZW[ 1953][3] =  0.000539094835505;

    Leb_Grid_XYZW[ 1954][0] = -0.049163750157381;
    Leb_Grid_XYZW[ 1954][1] =  0.602062837471398;
    Leb_Grid_XYZW[ 1954][2] =  0.796933664370098;
    Leb_Grid_XYZW[ 1954][3] =  0.000539094835505;

    Leb_Grid_XYZW[ 1955][0] = -0.049163750157381;
    Leb_Grid_XYZW[ 1955][1] =  0.602062837471398;
    Leb_Grid_XYZW[ 1955][2] = -0.796933664370098;
    Leb_Grid_XYZW[ 1955][3] =  0.000539094835505;

    Leb_Grid_XYZW[ 1956][0] = -0.049163750157381;
    Leb_Grid_XYZW[ 1956][1] = -0.602062837471398;
    Leb_Grid_XYZW[ 1956][2] =  0.796933664370098;
    Leb_Grid_XYZW[ 1956][3] =  0.000539094835505;

    Leb_Grid_XYZW[ 1957][0] = -0.049163750157381;
    Leb_Grid_XYZW[ 1957][1] = -0.602062837471398;
    Leb_Grid_XYZW[ 1957][2] = -0.796933664370098;
    Leb_Grid_XYZW[ 1957][3] =  0.000539094835505;

    Leb_Grid_XYZW[ 1958][0] =  0.049163750157381;
    Leb_Grid_XYZW[ 1958][1] =  0.796933664370098;
    Leb_Grid_XYZW[ 1958][2] =  0.602062837471398;
    Leb_Grid_XYZW[ 1958][3] =  0.000539094835505;

    Leb_Grid_XYZW[ 1959][0] =  0.049163750157381;
    Leb_Grid_XYZW[ 1959][1] =  0.796933664370098;
    Leb_Grid_XYZW[ 1959][2] = -0.602062837471398;
    Leb_Grid_XYZW[ 1959][3] =  0.000539094835505;

    Leb_Grid_XYZW[ 1960][0] =  0.049163750157381;
    Leb_Grid_XYZW[ 1960][1] = -0.796933664370098;
    Leb_Grid_XYZW[ 1960][2] =  0.602062837471398;
    Leb_Grid_XYZW[ 1960][3] =  0.000539094835505;

    Leb_Grid_XYZW[ 1961][0] =  0.049163750157381;
    Leb_Grid_XYZW[ 1961][1] = -0.796933664370098;
    Leb_Grid_XYZW[ 1961][2] = -0.602062837471398;
    Leb_Grid_XYZW[ 1961][3] =  0.000539094835505;

    Leb_Grid_XYZW[ 1962][0] = -0.049163750157381;
    Leb_Grid_XYZW[ 1962][1] =  0.796933664370098;
    Leb_Grid_XYZW[ 1962][2] =  0.602062837471398;
    Leb_Grid_XYZW[ 1962][3] =  0.000539094835505;

    Leb_Grid_XYZW[ 1963][0] = -0.049163750157381;
    Leb_Grid_XYZW[ 1963][1] =  0.796933664370098;
    Leb_Grid_XYZW[ 1963][2] = -0.602062837471398;
    Leb_Grid_XYZW[ 1963][3] =  0.000539094835505;

    Leb_Grid_XYZW[ 1964][0] = -0.049163750157381;
    Leb_Grid_XYZW[ 1964][1] = -0.796933664370098;
    Leb_Grid_XYZW[ 1964][2] =  0.602062837471398;
    Leb_Grid_XYZW[ 1964][3] =  0.000539094835505;

    Leb_Grid_XYZW[ 1965][0] = -0.049163750157381;
    Leb_Grid_XYZW[ 1965][1] = -0.796933664370098;
    Leb_Grid_XYZW[ 1965][2] = -0.602062837471398;
    Leb_Grid_XYZW[ 1965][3] =  0.000539094835505;

    Leb_Grid_XYZW[ 1966][0] =  0.796933664370098;
    Leb_Grid_XYZW[ 1966][1] =  0.602062837471398;
    Leb_Grid_XYZW[ 1966][2] =  0.049163750157381;
    Leb_Grid_XYZW[ 1966][3] =  0.000539094835505;

    Leb_Grid_XYZW[ 1967][0] =  0.796933664370098;
    Leb_Grid_XYZW[ 1967][1] =  0.602062837471398;
    Leb_Grid_XYZW[ 1967][2] = -0.049163750157381;
    Leb_Grid_XYZW[ 1967][3] =  0.000539094835505;

    Leb_Grid_XYZW[ 1968][0] =  0.796933664370098;
    Leb_Grid_XYZW[ 1968][1] = -0.602062837471398;
    Leb_Grid_XYZW[ 1968][2] =  0.049163750157381;
    Leb_Grid_XYZW[ 1968][3] =  0.000539094835505;

    Leb_Grid_XYZW[ 1969][0] =  0.796933664370098;
    Leb_Grid_XYZW[ 1969][1] = -0.602062837471398;
    Leb_Grid_XYZW[ 1969][2] = -0.049163750157381;
    Leb_Grid_XYZW[ 1969][3] =  0.000539094835505;

    Leb_Grid_XYZW[ 1970][0] = -0.796933664370098;
    Leb_Grid_XYZW[ 1970][1] =  0.602062837471398;
    Leb_Grid_XYZW[ 1970][2] =  0.049163750157381;
    Leb_Grid_XYZW[ 1970][3] =  0.000539094835505;

    Leb_Grid_XYZW[ 1971][0] = -0.796933664370098;
    Leb_Grid_XYZW[ 1971][1] =  0.602062837471398;
    Leb_Grid_XYZW[ 1971][2] = -0.049163750157381;
    Leb_Grid_XYZW[ 1971][3] =  0.000539094835505;

    Leb_Grid_XYZW[ 1972][0] = -0.796933664370098;
    Leb_Grid_XYZW[ 1972][1] = -0.602062837471398;
    Leb_Grid_XYZW[ 1972][2] =  0.049163750157381;
    Leb_Grid_XYZW[ 1972][3] =  0.000539094835505;

    Leb_Grid_XYZW[ 1973][0] = -0.796933664370098;
    Leb_Grid_XYZW[ 1973][1] = -0.602062837471398;
    Leb_Grid_XYZW[ 1973][2] = -0.049163750157381;
    Leb_Grid_XYZW[ 1973][3] =  0.000539094835505;

    Leb_Grid_XYZW[ 1974][0] =  0.796933664370098;
    Leb_Grid_XYZW[ 1974][1] =  0.049163750157382;
    Leb_Grid_XYZW[ 1974][2] =  0.602062837471398;
    Leb_Grid_XYZW[ 1974][3] =  0.000539094835505;

    Leb_Grid_XYZW[ 1975][0] =  0.796933664370098;
    Leb_Grid_XYZW[ 1975][1] =  0.049163750157382;
    Leb_Grid_XYZW[ 1975][2] = -0.602062837471398;
    Leb_Grid_XYZW[ 1975][3] =  0.000539094835505;

    Leb_Grid_XYZW[ 1976][0] =  0.796933664370098;
    Leb_Grid_XYZW[ 1976][1] = -0.049163750157382;
    Leb_Grid_XYZW[ 1976][2] =  0.602062837471398;
    Leb_Grid_XYZW[ 1976][3] =  0.000539094835505;

    Leb_Grid_XYZW[ 1977][0] =  0.796933664370098;
    Leb_Grid_XYZW[ 1977][1] = -0.049163750157382;
    Leb_Grid_XYZW[ 1977][2] = -0.602062837471398;
    Leb_Grid_XYZW[ 1977][3] =  0.000539094835505;

    Leb_Grid_XYZW[ 1978][0] = -0.796933664370098;
    Leb_Grid_XYZW[ 1978][1] =  0.049163750157382;
    Leb_Grid_XYZW[ 1978][2] =  0.602062837471398;
    Leb_Grid_XYZW[ 1978][3] =  0.000539094835505;

    Leb_Grid_XYZW[ 1979][0] = -0.796933664370098;
    Leb_Grid_XYZW[ 1979][1] =  0.049163750157382;
    Leb_Grid_XYZW[ 1979][2] = -0.602062837471398;
    Leb_Grid_XYZW[ 1979][3] =  0.000539094835505;

    Leb_Grid_XYZW[ 1980][0] = -0.796933664370098;
    Leb_Grid_XYZW[ 1980][1] = -0.049163750157382;
    Leb_Grid_XYZW[ 1980][2] =  0.602062837471398;
    Leb_Grid_XYZW[ 1980][3] =  0.000539094835505;

    Leb_Grid_XYZW[ 1981][0] = -0.796933664370098;
    Leb_Grid_XYZW[ 1981][1] = -0.049163750157382;
    Leb_Grid_XYZW[ 1981][2] = -0.602062837471398;
    Leb_Grid_XYZW[ 1981][3] =  0.000539094835505;

    Leb_Grid_XYZW[ 1982][0] =  0.652922252985688;
    Leb_Grid_XYZW[ 1982][1] =  0.098616215401270;
    Leb_Grid_XYZW[ 1982][2] =  0.750977611927295;
    Leb_Grid_XYZW[ 1982][3] =  0.000543331270503;

    Leb_Grid_XYZW[ 1983][0] =  0.652922252985688;
    Leb_Grid_XYZW[ 1983][1] =  0.098616215401270;
    Leb_Grid_XYZW[ 1983][2] = -0.750977611927295;
    Leb_Grid_XYZW[ 1983][3] =  0.000543331270503;

    Leb_Grid_XYZW[ 1984][0] =  0.652922252985688;
    Leb_Grid_XYZW[ 1984][1] = -0.098616215401270;
    Leb_Grid_XYZW[ 1984][2] =  0.750977611927295;
    Leb_Grid_XYZW[ 1984][3] =  0.000543331270503;

    Leb_Grid_XYZW[ 1985][0] =  0.652922252985688;
    Leb_Grid_XYZW[ 1985][1] = -0.098616215401270;
    Leb_Grid_XYZW[ 1985][2] = -0.750977611927295;
    Leb_Grid_XYZW[ 1985][3] =  0.000543331270503;

    Leb_Grid_XYZW[ 1986][0] = -0.652922252985688;
    Leb_Grid_XYZW[ 1986][1] =  0.098616215401270;
    Leb_Grid_XYZW[ 1986][2] =  0.750977611927295;
    Leb_Grid_XYZW[ 1986][3] =  0.000543331270503;

    Leb_Grid_XYZW[ 1987][0] = -0.652922252985688;
    Leb_Grid_XYZW[ 1987][1] =  0.098616215401270;
    Leb_Grid_XYZW[ 1987][2] = -0.750977611927295;
    Leb_Grid_XYZW[ 1987][3] =  0.000543331270503;

    Leb_Grid_XYZW[ 1988][0] = -0.652922252985688;
    Leb_Grid_XYZW[ 1988][1] = -0.098616215401270;
    Leb_Grid_XYZW[ 1988][2] =  0.750977611927295;
    Leb_Grid_XYZW[ 1988][3] =  0.000543331270503;

    Leb_Grid_XYZW[ 1989][0] = -0.652922252985688;
    Leb_Grid_XYZW[ 1989][1] = -0.098616215401270;
    Leb_Grid_XYZW[ 1989][2] = -0.750977611927295;
    Leb_Grid_XYZW[ 1989][3] =  0.000543331270503;

    Leb_Grid_XYZW[ 1990][0] =  0.652922252985688;
    Leb_Grid_XYZW[ 1990][1] =  0.750977611927295;
    Leb_Grid_XYZW[ 1990][2] =  0.098616215401270;
    Leb_Grid_XYZW[ 1990][3] =  0.000543331270503;

    Leb_Grid_XYZW[ 1991][0] =  0.652922252985688;
    Leb_Grid_XYZW[ 1991][1] =  0.750977611927295;
    Leb_Grid_XYZW[ 1991][2] = -0.098616215401270;
    Leb_Grid_XYZW[ 1991][3] =  0.000543331270503;

    Leb_Grid_XYZW[ 1992][0] =  0.652922252985688;
    Leb_Grid_XYZW[ 1992][1] = -0.750977611927295;
    Leb_Grid_XYZW[ 1992][2] =  0.098616215401270;
    Leb_Grid_XYZW[ 1992][3] =  0.000543331270503;

    Leb_Grid_XYZW[ 1993][0] =  0.652922252985688;
    Leb_Grid_XYZW[ 1993][1] = -0.750977611927295;
    Leb_Grid_XYZW[ 1993][2] = -0.098616215401270;
    Leb_Grid_XYZW[ 1993][3] =  0.000543331270503;

    Leb_Grid_XYZW[ 1994][0] = -0.652922252985688;
    Leb_Grid_XYZW[ 1994][1] =  0.750977611927295;
    Leb_Grid_XYZW[ 1994][2] =  0.098616215401270;
    Leb_Grid_XYZW[ 1994][3] =  0.000543331270503;

    Leb_Grid_XYZW[ 1995][0] = -0.652922252985688;
    Leb_Grid_XYZW[ 1995][1] =  0.750977611927295;
    Leb_Grid_XYZW[ 1995][2] = -0.098616215401270;
    Leb_Grid_XYZW[ 1995][3] =  0.000543331270503;

    Leb_Grid_XYZW[ 1996][0] = -0.652922252985688;
    Leb_Grid_XYZW[ 1996][1] = -0.750977611927295;
    Leb_Grid_XYZW[ 1996][2] =  0.098616215401270;
    Leb_Grid_XYZW[ 1996][3] =  0.000543331270503;

    Leb_Grid_XYZW[ 1997][0] = -0.652922252985688;
    Leb_Grid_XYZW[ 1997][1] = -0.750977611927295;
    Leb_Grid_XYZW[ 1997][2] = -0.098616215401270;
    Leb_Grid_XYZW[ 1997][3] =  0.000543331270503;

    Leb_Grid_XYZW[ 1998][0] =  0.098616215401270;
    Leb_Grid_XYZW[ 1998][1] =  0.652922252985688;
    Leb_Grid_XYZW[ 1998][2] =  0.750977611927295;
    Leb_Grid_XYZW[ 1998][3] =  0.000543331270503;

    Leb_Grid_XYZW[ 1999][0] =  0.098616215401270;
    Leb_Grid_XYZW[ 1999][1] =  0.652922252985688;
    Leb_Grid_XYZW[ 1999][2] = -0.750977611927295;
    Leb_Grid_XYZW[ 1999][3] =  0.000543331270503;

    Leb_Grid_XYZW[ 2000][0] =  0.098616215401270;
    Leb_Grid_XYZW[ 2000][1] = -0.652922252985688;
    Leb_Grid_XYZW[ 2000][2] =  0.750977611927295;
    Leb_Grid_XYZW[ 2000][3] =  0.000543331270503;

    Leb_Grid_XYZW[ 2001][0] =  0.098616215401270;
    Leb_Grid_XYZW[ 2001][1] = -0.652922252985688;
    Leb_Grid_XYZW[ 2001][2] = -0.750977611927295;
    Leb_Grid_XYZW[ 2001][3] =  0.000543331270503;

    Leb_Grid_XYZW[ 2002][0] = -0.098616215401270;
    Leb_Grid_XYZW[ 2002][1] =  0.652922252985688;
    Leb_Grid_XYZW[ 2002][2] =  0.750977611927295;
    Leb_Grid_XYZW[ 2002][3] =  0.000543331270503;

    Leb_Grid_XYZW[ 2003][0] = -0.098616215401270;
    Leb_Grid_XYZW[ 2003][1] =  0.652922252985688;
    Leb_Grid_XYZW[ 2003][2] = -0.750977611927295;
    Leb_Grid_XYZW[ 2003][3] =  0.000543331270503;

    Leb_Grid_XYZW[ 2004][0] = -0.098616215401270;
    Leb_Grid_XYZW[ 2004][1] = -0.652922252985688;
    Leb_Grid_XYZW[ 2004][2] =  0.750977611927295;
    Leb_Grid_XYZW[ 2004][3] =  0.000543331270503;

    Leb_Grid_XYZW[ 2005][0] = -0.098616215401270;
    Leb_Grid_XYZW[ 2005][1] = -0.652922252985688;
    Leb_Grid_XYZW[ 2005][2] = -0.750977611927295;
    Leb_Grid_XYZW[ 2005][3] =  0.000543331270503;

    Leb_Grid_XYZW[ 2006][0] =  0.098616215401270;
    Leb_Grid_XYZW[ 2006][1] =  0.750977611927295;
    Leb_Grid_XYZW[ 2006][2] =  0.652922252985688;
    Leb_Grid_XYZW[ 2006][3] =  0.000543331270503;

    Leb_Grid_XYZW[ 2007][0] =  0.098616215401270;
    Leb_Grid_XYZW[ 2007][1] =  0.750977611927295;
    Leb_Grid_XYZW[ 2007][2] = -0.652922252985688;
    Leb_Grid_XYZW[ 2007][3] =  0.000543331270503;

    Leb_Grid_XYZW[ 2008][0] =  0.098616215401270;
    Leb_Grid_XYZW[ 2008][1] = -0.750977611927295;
    Leb_Grid_XYZW[ 2008][2] =  0.652922252985688;
    Leb_Grid_XYZW[ 2008][3] =  0.000543331270503;

    Leb_Grid_XYZW[ 2009][0] =  0.098616215401270;
    Leb_Grid_XYZW[ 2009][1] = -0.750977611927295;
    Leb_Grid_XYZW[ 2009][2] = -0.652922252985688;
    Leb_Grid_XYZW[ 2009][3] =  0.000543331270503;

    Leb_Grid_XYZW[ 2010][0] = -0.098616215401270;
    Leb_Grid_XYZW[ 2010][1] =  0.750977611927295;
    Leb_Grid_XYZW[ 2010][2] =  0.652922252985688;
    Leb_Grid_XYZW[ 2010][3] =  0.000543331270503;

    Leb_Grid_XYZW[ 2011][0] = -0.098616215401270;
    Leb_Grid_XYZW[ 2011][1] =  0.750977611927295;
    Leb_Grid_XYZW[ 2011][2] = -0.652922252985688;
    Leb_Grid_XYZW[ 2011][3] =  0.000543331270503;

    Leb_Grid_XYZW[ 2012][0] = -0.098616215401270;
    Leb_Grid_XYZW[ 2012][1] = -0.750977611927295;
    Leb_Grid_XYZW[ 2012][2] =  0.652922252985688;
    Leb_Grid_XYZW[ 2012][3] =  0.000543331270503;

    Leb_Grid_XYZW[ 2013][0] = -0.098616215401270;
    Leb_Grid_XYZW[ 2013][1] = -0.750977611927295;
    Leb_Grid_XYZW[ 2013][2] = -0.652922252985688;
    Leb_Grid_XYZW[ 2013][3] =  0.000543331270503;

    Leb_Grid_XYZW[ 2014][0] =  0.750977611927295;
    Leb_Grid_XYZW[ 2014][1] =  0.652922252985688;
    Leb_Grid_XYZW[ 2014][2] =  0.098616215401270;
    Leb_Grid_XYZW[ 2014][3] =  0.000543331270503;

    Leb_Grid_XYZW[ 2015][0] =  0.750977611927295;
    Leb_Grid_XYZW[ 2015][1] =  0.652922252985688;
    Leb_Grid_XYZW[ 2015][2] = -0.098616215401270;
    Leb_Grid_XYZW[ 2015][3] =  0.000543331270503;

    Leb_Grid_XYZW[ 2016][0] =  0.750977611927295;
    Leb_Grid_XYZW[ 2016][1] = -0.652922252985688;
    Leb_Grid_XYZW[ 2016][2] =  0.098616215401270;
    Leb_Grid_XYZW[ 2016][3] =  0.000543331270503;

    Leb_Grid_XYZW[ 2017][0] =  0.750977611927295;
    Leb_Grid_XYZW[ 2017][1] = -0.652922252985688;
    Leb_Grid_XYZW[ 2017][2] = -0.098616215401270;
    Leb_Grid_XYZW[ 2017][3] =  0.000543331270503;

    Leb_Grid_XYZW[ 2018][0] = -0.750977611927295;
    Leb_Grid_XYZW[ 2018][1] =  0.652922252985688;
    Leb_Grid_XYZW[ 2018][2] =  0.098616215401270;
    Leb_Grid_XYZW[ 2018][3] =  0.000543331270503;

    Leb_Grid_XYZW[ 2019][0] = -0.750977611927295;
    Leb_Grid_XYZW[ 2019][1] =  0.652922252985688;
    Leb_Grid_XYZW[ 2019][2] = -0.098616215401270;
    Leb_Grid_XYZW[ 2019][3] =  0.000543331270503;

    Leb_Grid_XYZW[ 2020][0] = -0.750977611927295;
    Leb_Grid_XYZW[ 2020][1] = -0.652922252985688;
    Leb_Grid_XYZW[ 2020][2] =  0.098616215401270;
    Leb_Grid_XYZW[ 2020][3] =  0.000543331270503;

    Leb_Grid_XYZW[ 2021][0] = -0.750977611927295;
    Leb_Grid_XYZW[ 2021][1] = -0.652922252985688;
    Leb_Grid_XYZW[ 2021][2] = -0.098616215401270;
    Leb_Grid_XYZW[ 2021][3] =  0.000543331270503;

    Leb_Grid_XYZW[ 2022][0] =  0.750977611927295;
    Leb_Grid_XYZW[ 2022][1] =  0.098616215401270;
    Leb_Grid_XYZW[ 2022][2] =  0.652922252985688;
    Leb_Grid_XYZW[ 2022][3] =  0.000543331270503;

    Leb_Grid_XYZW[ 2023][0] =  0.750977611927295;
    Leb_Grid_XYZW[ 2023][1] =  0.098616215401270;
    Leb_Grid_XYZW[ 2023][2] = -0.652922252985688;
    Leb_Grid_XYZW[ 2023][3] =  0.000543331270503;

    Leb_Grid_XYZW[ 2024][0] =  0.750977611927295;
    Leb_Grid_XYZW[ 2024][1] = -0.098616215401270;
    Leb_Grid_XYZW[ 2024][2] =  0.652922252985688;
    Leb_Grid_XYZW[ 2024][3] =  0.000543331270503;

    Leb_Grid_XYZW[ 2025][0] =  0.750977611927295;
    Leb_Grid_XYZW[ 2025][1] = -0.098616215401270;
    Leb_Grid_XYZW[ 2025][2] = -0.652922252985688;
    Leb_Grid_XYZW[ 2025][3] =  0.000543331270503;

    Leb_Grid_XYZW[ 2026][0] = -0.750977611927295;
    Leb_Grid_XYZW[ 2026][1] =  0.098616215401270;
    Leb_Grid_XYZW[ 2026][2] =  0.652922252985688;
    Leb_Grid_XYZW[ 2026][3] =  0.000543331270503;

    Leb_Grid_XYZW[ 2027][0] = -0.750977611927295;
    Leb_Grid_XYZW[ 2027][1] =  0.098616215401270;
    Leb_Grid_XYZW[ 2027][2] = -0.652922252985688;
    Leb_Grid_XYZW[ 2027][3] =  0.000543331270503;

    Leb_Grid_XYZW[ 2028][0] = -0.750977611927295;
    Leb_Grid_XYZW[ 2028][1] = -0.098616215401270;
    Leb_Grid_XYZW[ 2028][2] =  0.652922252985688;
    Leb_Grid_XYZW[ 2028][3] =  0.000543331270503;

    Leb_Grid_XYZW[ 2029][0] = -0.750977611927295;
    Leb_Grid_XYZW[ 2029][1] = -0.098616215401270;
    Leb_Grid_XYZW[ 2029][2] = -0.652922252985688;
    Leb_Grid_XYZW[ 2029][3] =  0.000543331270503;

  }

}

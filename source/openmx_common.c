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
#include "openmx_basis_harmonics.h"

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
	}
	else{
	  atv[Rn][1] = di*tv[1][1] + dj*tv[2][1] + dk*tv[3][1];
	  atv[Rn][2] = di*tv[1][2] + dj*tv[2][2] + dk*tv[3][2];
	  atv[Rn][3] = di*tv[1][3] + dj*tv[2][3] + dk*tv[3][3];
	  atv_ijk[Rn][1] = i;
	  atv_ijk[Rn][2] = j;
	  atv_ijk[Rn][3] = k;
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
  return openmx_periodic_index(N,i,j,k);
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
  if (!openmx_basis_detail_integral_harmonic(l,m,theta,phi,SH,dSHt,dSHp)) {
    printf("Invalid arguments in routine Associated_Legendre\n");
    exit(0);
  }
}




void Associated_Legendre(int l, int m, double x, double ALeg[2])
{
  if (!openmx_basis_detail_associated_legendre(l,m,x,ALeg)) {
    printf("Invalid arguments in routine Associated_Legendre\n");
    exit(0);
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















/* A tiny registry where GPU-heavy phases publish the device memory (bytes,
   per physical GPU, group total) they need to run at full concurrency.
   Long-lived GPU caches (Set_Hamiltonian's resident tables) consult the
   maximum before claiming memory, so a sticky cache never lowers another
   phase's achievable concurrency. */

#define OPENMX_GPU_PHASE_NEED_SLOTS 8

static char   OpenMX_GpuPhaseNeed_Name[OPENMX_GPU_PHASE_NEED_SLOTS][32];
static size_t OpenMX_GpuPhaseNeed_Bytes[OPENMX_GPU_PHASE_NEED_SLOTS];

void OpenMX_GpuPhaseNeed_Register(const char *phase, size_t group_bytes)
{
  int i;

  if (phase==NULL || phase[0]=='\0') return;

  for (i=0; i<OPENMX_GPU_PHASE_NEED_SLOTS; i++){
    if (OpenMX_GpuPhaseNeed_Name[i][0]=='\0' ||
        strncmp(OpenMX_GpuPhaseNeed_Name[i],phase,sizeof(OpenMX_GpuPhaseNeed_Name[i])-1)==0){
      strncpy(OpenMX_GpuPhaseNeed_Name[i],phase,sizeof(OpenMX_GpuPhaseNeed_Name[i])-1);
      OpenMX_GpuPhaseNeed_Name[i][sizeof(OpenMX_GpuPhaseNeed_Name[i])-1] = '\0';
      OpenMX_GpuPhaseNeed_Bytes[i] = group_bytes;
      return;
    }
  }
}

size_t OpenMX_GpuPhaseNeed_Max(void)
{
  size_t need = 0U;
  int i;

  for (i=0; i<OPENMX_GPU_PHASE_NEED_SLOTS; i++){
    if (OpenMX_GpuPhaseNeed_Name[i][0]!='\0' && need<OpenMX_GpuPhaseNeed_Bytes[i]){
      need = OpenMX_GpuPhaseNeed_Bytes[i];
    }
  }
  return need;
}

size_t OpenMX_GpuPhaseNeed_MaxPrefixed(const char *prefix)
{
  size_t need = 0U;
  size_t plen;
  int i;

  if (prefix==NULL || prefix[0]=='\0') return 0U;
  plen = strlen(prefix);

  for (i=0; i<OPENMX_GPU_PHASE_NEED_SLOTS; i++){
    if (OpenMX_GpuPhaseNeed_Name[i][0]!='\0' &&
        strncmp(OpenMX_GpuPhaseNeed_Name[i],prefix,plen)==0 &&
        need<OpenMX_GpuPhaseNeed_Bytes[i]){
      need = OpenMX_GpuPhaseNeed_Bytes[i];
    }
  }
  return need;
}

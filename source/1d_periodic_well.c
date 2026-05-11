/*************************************************************************
  1d_periodic_well.c:

  1d_periodic_well.c is a program to analyze electronic structures of 
  an 1D periodic well model and its dynamical properties in excitations 
  under light irradiation. 

  Log of 1d_periodic_well.c:

     03/Dec./2023  Released by T. Ozaki 

**************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* define struct, global variables, and functions */

typedef struct { double r,i; } dcomplex;

#define print_data  0
#define PI          3.1415926535897932384626

int atomnum,SpinP_switch,NumCell_1,NumCell_2,NumCell_3,TNum_CWFs;
int *Num_CWFs;
double *****Hop;

void zheevx_(char *JOBZ, char *RANGE, char *UPLO, int *N, dcomplex *A, int *LDA, double *VL, double *VU, int *IL, int *IU,
                        double *ABSTOL, int *M, double *W, dcomplex *Z, int *LDZ, dcomplex *WORK, int *LWORK, double *RWORK,
                        int *IWORK, int *IFAIL, int *INFO);

void lapack_zheevx(int N, dcomplex *A, dcomplex *Z, double *W);


int main(int argc, char *argv[]) 
{
  double *eval;
  double si1,co1,si12,co12,G,Gd1,Gd12;
  int Mmax,N,m,n,i,j;
  double k,k0,d1,d2,V1,V2,V3,a,tmp,Nlayer;
  dcomplex *Hk,*Coes;

  /* set parameters */
  
  Mmax = 1000;  
  d1 = 40.0;
  d2 = 100.0;
  V1 = 0.0;
  V2 = 1.0;
  V3 = 0.7;
  a = d1 + d2;
  N = 2*Mmax + 1;
  Nlayer = 10.0;
    
  /* allocation of arrays */

  Hk = (dcomplex*)malloc(sizeof(dcomplex)*N*N);
  Coes = (dcomplex*)malloc(sizeof(dcomplex)*N*N);
  eval = (double*)malloc(sizeof(double)*N);

  k0 = 0.0;
  
  /* set H(k) */

  for (m=0; m<N*N; m++) { Hk[m].r = 0.0; Hk[m].i = 0.0; }
  
  for (m=-Mmax; m<=Mmax; m++){
    for (n=-Mmax; n<=Mmax; n++){

      if (m==n){

        /* contributions from the constant potentials */

        Hk[(n+Mmax)*N+(m+Mmax)].r = 2.0*PI*PI/(a*a)*(k0-(double)m)*(k0-(double)m);  // kinetic energy
        Hk[(n+Mmax)*N+(m+Mmax)].r += (V1*d1+V2*d2)/a;

      }
      
      else{

	G = (double)(m-n)*2.0*PI/a;        
	Gd1 = G*d1;
	Gd12 = G*(d1+d2);
        co1 = cos(Gd1);	
        si1 = sin(Gd1);
	co12 = cos(Gd12);	
        si12 = sin(Gd12);

        /* contributions from the constant potentials */
	
	Hk[(n+Mmax)*N+(m+Mmax)].i = (V1*(1.0 - co1) + V2*(co1-co12))/((double)(m-n)*2.0*PI);
	Hk[(n+Mmax)*N+(m+Mmax)].r = -(-V1*si1 + V2*(si1-si12))/((double)(m-n)*2.0*PI);

        /* contribution from the cosine potentials for 0<=x<=d1 */

	tmp = V3/(a*(d1*d1*G*G-4.0*Nlayer*Nlayer*PI*PI));
        Hk[(n+Mmax)*N+(m+Mmax)].r += tmp*(d1*d1*G*si1*cos(2.0*Nlayer*PI) - 2.0*d1*Nlayer*PI*sin(2.0*Nlayer*PI)*co1);
	Hk[(n+Mmax)*N+(m+Mmax)].i += tmp*(d1*d1*G-d1*d1*G*co1*cos(2.0*Nlayer*PI)-2.0*d1*Nlayer*PI*sin(2.0*Nlayer*PI)*si1);

      }
    }
  }

  /*
  printf("Real\n");
  for (i=0; i<N; i++){
    for (j=0; j<N; j++){
      printf("%6.3f",Hk[j*N+i].r);
    }
    printf("\n");
  }

  printf("Imag\n");
  for (i=0; i<N; i++){
    for (j=0; j<N; j++){
      printf("%6.3f",Hk[j*N+i].i);
    }
    printf("\n");
  }
  */
  
  /* diagonalizing of H(k) */

  lapack_zheevx(N, Hk, Coes, eval);      

  for (m=0; m<100; m++){
    printf("m=%2d eval=%18.15f\n",m+1,eval[m]);
  }

  m = -1+19;

  int Ndiv;
  double xmin,xmax,dx,x,cr,ci,co,si,kgx,fr,fi;
  
  Ndiv = 15000; 
  xmin = -(d1+d2);
  xmax = d1+d2+d1;
  dx = (xmax-xmin)/(double)Ndiv;
  
  for (i=0; i<Ndiv; i++){

    x = xmin + (double)i*dx;
    
    fr = 0.0;
    fi = 0.0;
    
    for (n=-Mmax; n<=Mmax; n++){

      cr = Coes[m*N+(n+Mmax)].r;
      ci = Coes[m*N+(n+Mmax)].i;
      
      kgx = (k0-n)*2.0*PI/a*x;
      co = cos(kgx);
      si = sin(kgx);
      fr += cr*co - ci*si;
      fi += cr*si + ci*co;
    }

    printf("%15.12f %15.12f %15.12f\n",x,fr,fi);
  }
  
  /* freeing of arrays */

  free(eval);
  free(Coes);
  free(Hk);
  
  exit(0);  
}




void lapack_zheevx(int N, dcomplex *A, dcomplex *Z, double *W)
{
  char *JOBZ="V";
  char *UPLO="L";
  char *RANGE="A";
  int i,LDA,LDZ,IL,IU,LWORK;
  int *IWORK,*IFAIL,M,INFO;
  double ABSTOL=1.0e-14;
  double VL,VU;
  double *RWORK;
  dcomplex *WORK;

  /* allocation of arrays */

  LDA = N;  LDZ = N;
  IL = 1;   IU = N;

  LWORK = 3*N;
  WORK  = (dcomplex*)malloc(sizeof(dcomplex)*LWORK);
  RWORK = (double*)malloc(sizeof(double)*7*N);
  IWORK = (int*)malloc(sizeof(int)*5*N);
  IFAIL = (int*)malloc(sizeof(int)*N);

  /* call zheevx */

  zheevx_(JOBZ, RANGE, UPLO, &N, A, &LDA, &VL, &VU, &IL, &IU,
          &ABSTOL, &M, W, Z, &LDZ, WORK, &LWORK, RWORK,
          IWORK, IFAIL, &INFO );

  /* freeing of arrays */

  free(WORK);
  free(RWORK);
  free(IWORK);
  free(IFAIL);
}

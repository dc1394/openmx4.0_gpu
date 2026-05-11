/*************************************************************************
  abc_col.c:
 
  abc_col.c is a program to calculate analytic non-abelian Berry connections 
  and related quanties based on hamiltonian calculated by OpenMX within 
  collinear density functional theory. The hamiltnonian is read from a scfout file.

  Log of abc_col.c:

     21/May/2025  Released by T. Ozaki 

**************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "read_scfout.h"

/* define struct, global variables, and functions */

typedef struct { double r,i; } dcomplex;

#define print_data  0
#define PI          3.1415926535897932384626
#define kB          0.00008617251324000000   /* eV/K          */          
#define AU2Debye    2.54174776               /* AU to Debey   */
#define AU2Mucm     5721.52891433            /* 100 e/bohr/bohr */

int MatDim,Nk1,Nk2,Nk3,T_knum;
int *MP,No[2],Nuo[2],Nomax,Nuomax;

int *TK2i,*TK2j,*TK2k,*T_k_op;
double tweight;
double *T_KGrids1,*T_KGrids2,*T_KGrids3;
double **PF1,**PF2;


void zhegvx_(int *itype, char *jobz, char *range, char *
	     uplo, int *n, dcomplex *a, int *lda, dcomplex *b,
	     int *ldb, double *vl, double *vu, int *il, int *
	     iu, double *abstol, int *m, double *w, dcomplex *z__,
	     int *ldz, dcomplex *work, int *lwork, double *rwork,
	     int *iwork, int *ifail, int *info);

void zgetrf_(int *m, int *n, dcomplex *a,int *lda,int *ipvt, int *info );
void zgetri_(int *n,dcomplex *a,int *lda, int *ipvt, dcomplex *work, int *lwork, int *info);
void dgetri_(int *N,double *A, int *LDA, int *IPIV, double *WORK, int *LWORK, int *INFO);

void zheevx_(char *JOBZ, char *RANGE, char *UPLO, int *N, dcomplex *A, 
             int *LDA, double *VL, double *VU, int *IL, int *IU,
             double *ABSTOL, int *M, double *W, dcomplex *Z, int *LDZ, 
             dcomplex *WORK, int *LWORK, double *RWORK,
             int *IWORK, int *IFAIL, int *INFO);

int zgesvd_(char *jobu, char *jobvt, int *m, int *n, 
	    dcomplex *a, int *lda,
	    double *s, dcomplex *u, 
	    int *ldu, dcomplex *vt,
	    int *ldvt, dcomplex *work, 
	    int *lwork, double *rwork, int *info);

void dsyevx_(char *JOBZ, char *RANGE, char *UPLO, int *N, double *A, int *LDA, 
             double *VL, double *VU,
             int *IL, int *IU, double *ABSTOL, int *M, double *W, double *Z, 
             int *LDZ, double *WORK,
             int *LWORK, int *IWORK, int *IFAIL, int *INFO);

void allocate_free_arrays(char *mode);
dcomplex Complex(double re, double im){ return (dcomplex){re, im}; }
void Cross_Product(double a[4], double b[4], double c[4]);
double Dot_Product(double a[4], double b[4]);

void lapack_zheevx(int N, dcomplex *A, dcomplex *WF, double *W);
void lapack_zhegvx(int N, dcomplex *A, dcomplex *B, dcomplex *Z, double *W);
void lapack_dsyevx(int N, int IL, int IU, double *A, double *WF, double *W);
void Calc_Hk_Sk(dcomplex *Hk, dcomplex *Sk, int spin, double k1, double k2, double k3);
void Calc_dHk_dSk(dcomplex **dHk, dcomplex **dSk, dcomplex *WF, 
                  int spin, double k1, double k2, double k3);

void Calc_Projection_Functions(double **PF1, double **PF2);

void Calc_Inverse(int n, dcomplex *A);
void Calc_Inverse_SVD(int n, dcomplex *A);

void Calc_ABC_Full( int spin, double k1, double k2, double k3, 
                    double **PF1, double **PF2, 
                    dcomplex ***ABC11, dcomplex ***ABC12, 
                    dcomplex ***ABC21, dcomplex ***ABC22, 
                    double *eval );

void Calc_ABC_A11( int spin, double k1, double k2, double k3, 
                   double **PF1, dcomplex ***ABC11, double *eval );

void Dispersion_Berry_Connections();
void Polarization();
void SHG();







int main(int argc, char *argv[]) 
{
  int spin,n1,n2,n3,l1,l2,l3,i,j;
  double k1,k2,k3,co,si,kRn;
  double *eval;
  dcomplex *Hk;

  /* MPI initialization */

  MPI_Init(&argc, &argv);

  /* set the number of k-points */

  Nk1 = 20; 
  Nk2 = 20;
  Nk3 = 20;

  /* read the .scfout file */

  read_scfout(argv);
  allocate_free_arrays("allocate");

  /* calculate projection functions */

  Calc_Projection_Functions(PF1,PF2);  

  /* polarization */

  Polarization();

  /* Berry connections along the designated k-paths */

  //Dispersion_Berry_Connections();

  /* freeing of arrays */

  allocate_free_arrays("free");

  MPI_Finalize();
  exit(0);
}


void Dispersion_Berry_Connections()
{
  int m,i,nkpath,spin,Nk,p,kloop;
  double dk,k1,k2,k3,d,xs,ys,zs,xe,ye,ze;
  double sum0,sum1,sum2,dd,dx,dy,dz,step;
  double *eval,**kpath_s,**kpath_e;
  dcomplex ***A11;

  /*************************************************************
                    allocation of arrays
  *************************************************************/

  nkpath = 4;

  kpath_s = (double**)malloc(sizeof(double*)*nkpath);
  for (i=0; i<nkpath; i++){
    kpath_s[i] = (double*)malloc(sizeof(double)*3);
  }

  kpath_e = (double**)malloc(sizeof(double*)*nkpath);
  for (i=0; i<nkpath; i++){
    kpath_e[i] = (double*)malloc(sizeof(double)*3);
  }

  A11 = (dcomplex***)malloc(sizeof(dcomplex**)*2);
  for (m=0; m<2; m++){
    A11[m] = (dcomplex**)malloc(sizeof(dcomplex*)*3);
    for (i=0; i<3; i++){
      A11[m][i] = (dcomplex*)malloc(sizeof(dcomplex)*Nomax*Nomax);
    }
  }

  eval = (double*)malloc(sizeof(double)*MatDim);

  /*************************************************************
 
  *************************************************************/

  spin = 0;
  dk = 0.005;

  kpath_s[0][0] = 0.00; kpath_s[0][1] = 0.00; kpath_s[0][2] = 0.00; 
  kpath_s[1][0] = 0.50; kpath_s[1][1] = 0.00; kpath_s[1][2] = 0.00; 
  kpath_s[2][0] = 0.50; kpath_s[2][1] = 0.25; kpath_s[2][2] = 0.00; 
  kpath_s[3][0] = 0.25; kpath_s[3][1] = 0.25; kpath_s[3][2] = 0.25; 

  kpath_e[0][0] = 0.50; kpath_e[0][1] = 0.00; kpath_e[0][2] = 0.00; 
  kpath_e[1][0] = 0.50; kpath_e[1][1] = 0.25; kpath_e[1][2] = 0.00; 
  kpath_e[2][0] = 0.25; kpath_e[2][1] = 0.25; kpath_e[2][2] = 0.25; 
  kpath_e[3][0] = 0.00; kpath_e[3][1] = 0.00; kpath_e[3][2] = 0.00; 

  step = 0.0;

  for (p=0; p<nkpath; p++){

    xs = kpath_s[p][0]*rtv[1][1] + kpath_s[p][1]*rtv[2][1] + kpath_s[p][2]*rtv[3][1];    
    ys = kpath_s[p][0]*rtv[1][2] + kpath_s[p][1]*rtv[2][2] + kpath_s[p][2]*rtv[3][2];    
    zs = kpath_s[p][0]*rtv[1][3] + kpath_s[p][1]*rtv[2][3] + kpath_s[p][2]*rtv[3][3];    

    xe = kpath_e[p][0]*rtv[1][1] + kpath_e[p][1]*rtv[2][1] + kpath_e[p][2]*rtv[3][1];    
    ye = kpath_e[p][0]*rtv[1][2] + kpath_e[p][1]*rtv[2][2] + kpath_e[p][2]*rtv[3][2];    
    ze = kpath_e[p][0]*rtv[1][3] + kpath_e[p][1]*rtv[2][3] + kpath_e[p][2]*rtv[3][3];    

    dx = xe - xs;
    dy = ye - ys;
    dz = ze - zs;

    d = sqrt(dx*dx + dy*dy + dz*dz);
    Nk = (int)(d/dk); 
    dd = d/(double)Nk; 

    //printf("Nk=%2d\n",Nk);

    for (kloop=0; kloop<=Nk; kloop++){ 

      k1 = xs + (double)kloop*dx/(double)Nk;
      k2 = ys + (double)kloop*dy/(double)Nk;
      k3 = zs + (double)kloop*dz/(double)Nk;

      Calc_ABC_A11(spin,k1,k2,k3,PF1,A11,eval);

      sum0 = 0.0; sum1 = 0.0; sum2 = 0.0;
      for (i=0; i<No[spin]; i++){
        sum0 += A11[0][0][i*No[spin]+i].r + A11[1][0][i*No[spin]+i].r;
        sum1 += A11[0][1][i*No[spin]+i].r + A11[1][1][i*No[spin]+i].r;
        sum2 += A11[0][2][i*No[spin]+i].r + A11[1][2][i*No[spin]+i].r;
      } 

      printf("%15.12f %15.12f %15.12f %15.12f\n",step,sum0,sum1,sum2);   

      step += dd; 
    }
  }

  /*************************************************************
                    freeing of arrays
  *************************************************************/

  for (i=0; i<nkpath; i++){
    free(kpath_s[i]);
  }
  free(kpath_s);

  for (i=0; i<nkpath; i++){
    free(kpath_e[i]);
  }
  free(kpath_e);

  for (m=0; m<2; m++){
    for (i=0; i<3; i++){
      free(A11[m][i]);
    }
    free(A11[m]);
  }
  free(A11);

  free(eval);
}





void Polarization()
{
  int i,j,spin,kloop,m,gm,lm;
  double k1,k2,k3,tmp,Pol1[2][3],Pol2[2][3],be[4];
  double **ABC1,**ABC2,*eval,sum00,sum01,sum02,sum10,sum11,sum12;
  dcomplex ***A11;

  /*************************************************************
                    allocation of arrays
  *************************************************************/

  A11 = (dcomplex***)malloc(sizeof(dcomplex**)*2);
  for (m=0; m<2; m++){
    A11[m] = (dcomplex**)malloc(sizeof(dcomplex*)*3);
    for (i=0; i<3; i++){
      A11[m][i] = (dcomplex*)malloc(sizeof(dcomplex)*Nomax*Nomax);
    }
  }

  eval = (double*)malloc(sizeof(double)*MatDim);

  /*************************************************************
                  loops for spin and k-points
  *************************************************************/

  for (spin=0; spin<2; spin++){ 
    for (i=0; i<2; i++){
      Pol1[spin][i] = 0.0;
      Pol2[spin][i] = 0.0;
    }
  }

  for (spin=0; spin<=SpinP_switch; spin++){ 

    /* loop for k-points */

    for (kloop=0; kloop<T_knum; kloop++){

      k1 = T_KGrids1[kloop]; 
      k2 = T_KGrids2[kloop]; 
      k3 = T_KGrids3[kloop];

      Calc_ABC_A11(spin,k1,k2,k3,PF1,A11,eval);

      sum00 = 0.0; sum01 = 0.0; sum02 = 0.0;     
      for (i=0; i<No[spin]; i++){
        sum00 += A11[0][0][i*No[spin]+i].r;
        sum01 += A11[0][1][i*No[spin]+i].r;
        sum02 += A11[0][2][i*No[spin]+i].r;
      } 

      Pol1[spin][0] += sum00*(double)T_k_op[kloop];           
      Pol1[spin][1] += sum01*(double)T_k_op[kloop];           
      Pol1[spin][2] += sum02*(double)T_k_op[kloop];           

      sum10 = 0.0; sum11 = 0.0; sum12 = 0.0;     
      for (i=0; i<No[spin]; i++){
        sum10 += A11[1][0][i*No[spin]+i].r;
        sum11 += A11[1][1][i*No[spin]+i].r;
        sum12 += A11[1][2][i*No[spin]+i].r;
      } 

      Pol2[spin][0] += sum10*(double)T_k_op[kloop];           
      Pol2[spin][1] += sum11*(double)T_k_op[kloop];           
      Pol2[spin][2] += sum12*(double)T_k_op[kloop];           

      printf("%15.12f %15.12f %15.12f %15.12f\n",k1,sum00+sum10,sum01+sum11,sum02+sum12);   

      //printf("%15.12f %15.12f %15.12f %15.12f\n",k1,sum00,sum01,sum02);   

      //printf("%15.12f %15.12f %15.12f %15.12f\n",k1,sum00,sum01,sum02);   

    } /* end of kloop */ 
  } /* end of spin */

  /*************************************************************
               taking account of the modulo 2pi
  *************************************************************/
  
  for (spin=0; spin<=SpinP_switch; spin++){
    Pol1[spin][0] /= (double)(Nk1*Nk2*Nk3);
    Pol1[spin][1] /= (double)(Nk1*Nk2*Nk3);
    Pol1[spin][2] /= (double)(Nk1*Nk2*Nk3);
    Pol2[spin][0] /= (double)(Nk1*Nk2*Nk3);
    Pol2[spin][1] /= (double)(Nk1*Nk2*Nk3);
    Pol2[spin][2] /= (double)(Nk1*Nk2*Nk3);
  }

  for (spin=0; spin<=SpinP_switch; spin++){

    for (i=1; i<=3; i++){

      tmp = (Pol2[spin][0]*rtv[i][1] + Pol2[spin][1]*rtv[i][2] + Pol2[spin][2]*rtv[i][3])/(2.0*PI);

      while(tmp<0.0){
	tmp += 1.0;
      }

      /* tmp is adjusted in between 0.0 and 1.0. */
      j = (int)tmp;
      be[i] = tmp - (double)j;
    }

    Pol2[spin][0] = be[1]*tv[1][1] + be[2]*tv[2][1] + be[3]*tv[3][1];
    Pol2[spin][1] = be[1]*tv[1][2] + be[2]*tv[2][2] + be[3]*tv[3][2];
    Pol2[spin][2] = be[1]*tv[1][3] + be[2]*tv[2][3] + be[3]*tv[3][3];

    /* unit is in Debye */

    Pol1[spin][0] *= -AU2Debye;
    Pol1[spin][1] *= -AU2Debye;
    Pol1[spin][2] *= -AU2Debye;

    Pol2[spin][0] *= -AU2Debye;
    Pol2[spin][1] *= -AU2Debye;
    Pol2[spin][2] *= -AU2Debye;

  } // spin

  /* for non-spin polarized case */

  if (SpinP_switch==0){
    Pol1[1][0] = Pol1[0][0];
    Pol1[1][1] = Pol1[0][1];
    Pol1[1][2] = Pol1[0][2];
    Pol2[1][0] = Pol2[0][0];
    Pol2[1][1] = Pol2[0][1];
    Pol2[1][2] = Pol2[0][2];
  }

  /************************************************
         Pol1+Pol2 -> Edpx, Edpy, and Edpz
  *************************************************/

  double AbsD,Cell_Volume,CellV,vtmp[4]; 
  double Edpx,Edpy,Edpz,Cdpx,Cdpy,Cdpz,Bdpx,Bdpy,Bdpz,Tdpx,Tdpy,Tdpz;
  double Edpx_C,Edpy_C,Edpz_C,Cdpx_C,Cdpy_C,Cdpz_C,Bdpx_C,Bdpy_C,Bdpz_C,Tdpx_C,Tdpy_C,Tdpz_C;

  /* calculate Cell_Volume */

  Cross_Product(tv[2],tv[3],vtmp);
  CellV = Dot_Product(tv[1],vtmp); 
  Cell_Volume = fabs(CellV); 

  Edpx = Pol1[0][0]+Pol2[0][0]+Pol1[1][0]+Pol2[1][0];
  Edpy = Pol1[0][1]+Pol2[0][1]+Pol1[1][1]+Pol2[1][1];
  Edpz = Pol1[0][2]+Pol2[0][2]+Pol1[1][2]+Pol2[1][2];

  /************************************************
      find the core charge contribution of
          the macroscopic polarization
  *************************************************/

  Cdpx = dipole_moment_core[1];
  Cdpy = dipole_moment_core[2];
  Cdpz = dipole_moment_core[3];

  /************************************************
      find the background charge contribution of
          the macroscopic polarization
  *************************************************/

  Bdpx = dipole_moment_background[1];
  Bdpy = dipole_moment_background[2];
  Bdpz = dipole_moment_background[3];

  /************************************************
    calculate the total macroscopic polarization 
            as dipolemoment (for molecule)
  *************************************************/

  Tdpx = Cdpx + Edpx + Bdpx;
  Tdpy = Cdpy + Edpy + Bdpy;
  Tdpz = Cdpz + Edpz + Bdpz;

  AbsD = sqrt(Tdpx*Tdpx + Tdpy*Tdpy + Tdpz*Tdpz);

  /************************************************
    calculate the total macroscopic polarization 
                  in micro C/cm^2
  *************************************************/

  Tdpx_C = AU2Mucm*Tdpx/Cell_Volume/AU2Debye;
  Tdpy_C = AU2Mucm*Tdpy/Cell_Volume/AU2Debye;
  Tdpz_C = AU2Mucm*Tdpz/Cell_Volume/AU2Debye;
  Bdpx_C = AU2Mucm*Bdpx/Cell_Volume/AU2Debye;
  Bdpy_C = AU2Mucm*Bdpy/Cell_Volume/AU2Debye;
  Bdpz_C = AU2Mucm*Bdpz/Cell_Volume/AU2Debye;
  Edpx_C = AU2Mucm*Edpx/Cell_Volume/AU2Debye;
  Edpy_C = AU2Mucm*Edpy/Cell_Volume/AU2Debye;
  Edpz_C = AU2Mucm*Edpz/Cell_Volume/AU2Debye;
  Cdpx_C = AU2Mucm*Cdpx/Cell_Volume/AU2Debye;
  Cdpy_C = AU2Mucm*Cdpy/Cell_Volume/AU2Debye;
  Cdpz_C = AU2Mucm*Cdpz/Cell_Volume/AU2Debye;

  /************************************************
               print results to stdout
  *************************************************/

  printf("\n*******************************************************\n");      fflush(stdout);
  printf("                     Cell information                    \n");  fflush(stdout);
  printf("*******************************************************\n\n");      fflush(stdout);

  printf("\n r-space primitive vector (Bohr)\n");
  printf("  tv1:  %10.6f %10.6f %10.6f\n",   tv[1][1], tv[1][2], tv[1][3]);
  printf("  tv2:  %10.6f %10.6f %10.6f\n",   tv[2][1], tv[2][2], tv[2][3]);
  printf("  tv3:  %10.6f %10.6f %10.6f\n\n", tv[3][1], tv[3][2], tv[3][3]);
  printf(" k-space primitive vector (Bohr^-1)\n");
  printf("  rtv1: %10.6f %10.6f %10.6f\n",   rtv[1][1], rtv[1][2], rtv[1][3]);
  printf("  rtv2: %10.6f %10.6f %10.6f\n",   rtv[2][1], rtv[2][2], rtv[2][3]);
  printf("  rtv3: %10.6f %10.6f %10.6f\n\n", rtv[3][1], rtv[3][2], rtv[3][3]);

  printf("  Cell_Volume: %15.10f (Bohr^3)\n\n",Cell_Volume);

  printf("\n*******************************************************\n");      fflush(stdout);
  printf("              Electric dipole  (Debye) : Berry phase         \n");  fflush(stdout);
  printf("*******************************************************\n\n");      fflush(stdout);

  printf(" Absolute dipole moment %17.8f\n\n",AbsD); 

  printf("               Background        Core             Electron          Total\n\n"); fflush(stdout);

  printf(" Dx     %17.8f %17.8f %17.8f %17.8f\n",Bdpx,Cdpx,Edpx,Tdpx);fflush(stdout);
  printf(" Dy     %17.8f %17.8f %17.8f %17.8f\n",Bdpy,Cdpy,Edpy,Tdpy);fflush(stdout);
  printf(" Dz     %17.8f %17.8f %17.8f %17.8f\n",Bdpz,Cdpz,Edpz,Tdpz);fflush(stdout);
  printf("\n\n");

  printf("\n***************************************************************\n"); fflush(stdout);
  printf("              Electric polarization (muC/cm^2) : Berry phase          \n");  fflush(stdout);
  printf("***************************************************************\n\n"); fflush(stdout);

  printf("               Background        Core             Electron          Total\n\n"); fflush(stdout);

  printf(" Px     %17.8f %17.8f %17.8f %17.8f\n",Bdpx_C,Cdpx_C,Edpx_C,Tdpx_C);fflush(stdout);
  printf(" Py     %17.8f %17.8f %17.8f %17.8f\n",Bdpy_C,Cdpy_C,Edpy_C,Tdpy_C);fflush(stdout);
  printf(" Pz     %17.8f %17.8f %17.8f %17.8f\n",Bdpz_C,Cdpz_C,Edpz_C,Tdpz_C);fflush(stdout);
  printf("\n\n");

  /*************************************************************
                    freeing of arrays
  *************************************************************/

  for (m=0; m<2; m++){
    for (i=0; i<3; i++){
      free(A11[m][i]);
    }
    free(A11[m]);
  }
  free(A11);

  free(eval);
}





void SHG()
{
  int i,j,spin,kloop,m,gm,lm;
  double k1,k2,k3,tmp,Pol1[2][3],Pol2[2][3],be[4];
  double **ABC1,**ABC2,*eval,sum0,sum1,sum2;
  dcomplex ***A11,***A12,***A21,***A22;

  /*************************************************************
                    allocation of arrays
  *************************************************************/

  A11 = (dcomplex***)malloc(sizeof(dcomplex**)*2);
  for (m=0; m<2; m++){
    A11[m] = (dcomplex**)malloc(sizeof(dcomplex*)*3);
    for (i=0; i<3; i++){
      A11[m][i] = (dcomplex*)malloc(sizeof(dcomplex)*Nomax*Nomax);
    }
  }

  A12 = (dcomplex***)malloc(sizeof(dcomplex**)*2);
  for (m=0; m<2; m++){
    A12[m] = (dcomplex**)malloc(sizeof(dcomplex*)*3);
    for (i=0; i<3; i++){
      A12[m][i] = (dcomplex*)malloc(sizeof(dcomplex)*Nomax*Nuomax);
    }
  }

  A21 = (dcomplex***)malloc(sizeof(dcomplex**)*2);
  for (m=0; m<2; m++){
    A21[m] = (dcomplex**)malloc(sizeof(dcomplex*)*3);
    for (i=0; i<3; i++){
      A21[m][i] = (dcomplex*)malloc(sizeof(dcomplex)*Nomax*Nuomax);
    }
  }

  A22 = (dcomplex***)malloc(sizeof(dcomplex**)*2);
  for (m=0; m<2; m++){
    A22[m] = (dcomplex**)malloc(sizeof(dcomplex*)*3);
    for (i=0; i<3; i++){
      A22[m][i] = (dcomplex*)malloc(sizeof(dcomplex)*Nuomax*Nuomax);
    }
  }

  eval = (double*)malloc(sizeof(double)*MatDim);

  /*************************************************************
                  loops for spin and k-points
  *************************************************************/

  for (spin=0; spin<2; spin++){ 
    for (i=0; i<2; i++){
      Pol1[spin][i] = 0.0;
      Pol2[spin][i] = 0.0;
    }
  }

  for (spin=0; spin<=SpinP_switch; spin++){ 

    /* loop for k-points */

    for (kloop=0; kloop<T_knum; kloop++){

      k1 = T_KGrids1[kloop]; 
      k2 = T_KGrids2[kloop]; 
      k3 = T_KGrids3[kloop];

      Calc_ABC_Full(spin,k1,k2,k3,PF1,PF2,A11,A12,A21,A22,eval);



      sum0 = 0.0; sum1 = 0.0; sum2 = 0.0;     
      for (i=0; i<No[spin]; i++){
        sum0 += A11[0][0][i*No[spin]+i].r;
        sum1 += A11[0][1][i*No[spin]+i].r;
        sum2 += A11[0][2][i*No[spin]+i].r;
      } 

      Pol1[spin][0] += sum0*(double)T_k_op[kloop];           
      Pol1[spin][1] += sum1*(double)T_k_op[kloop];           
      Pol1[spin][2] += sum2*(double)T_k_op[kloop];           

      sum0 = 0.0; sum1 = 0.0; sum2 = 0.0;     
      for (i=0; i<No[spin]; i++){
        sum0 += A11[1][0][i*No[spin]+i].r;
        sum1 += A11[1][1][i*No[spin]+i].r;
        sum2 += A11[1][2][i*No[spin]+i].r;
      } 

      Pol2[spin][0] += sum0*(double)T_k_op[kloop];           
      Pol2[spin][1] += sum1*(double)T_k_op[kloop];           
      Pol2[spin][2] += sum2*(double)T_k_op[kloop];           

      printf("%15.12f %15.12f %15.12f %15.12f\n",k1,sum0,sum1,sum2);   

    } /* end of kloop */ 
  } /* end of spin */

  /*************************************************************
               taking account of the modulo 2pi
  *************************************************************/
  
  for (spin=0; spin<=SpinP_switch; spin++){
    Pol1[spin][0] /= (double)(Nk1*Nk2*Nk3);
    Pol1[spin][1] /= (double)(Nk1*Nk2*Nk3);
    Pol1[spin][2] /= (double)(Nk1*Nk2*Nk3);
    Pol2[spin][0] /= (double)(Nk1*Nk2*Nk3);
    Pol2[spin][1] /= (double)(Nk1*Nk2*Nk3);
    Pol2[spin][2] /= (double)(Nk1*Nk2*Nk3);
  }

  for (spin=0; spin<=SpinP_switch; spin++){

    for (i=1; i<=3; i++){

      tmp = (Pol2[spin][0]*rtv[i][1] + Pol2[spin][1]*rtv[i][2] + Pol2[spin][2]*rtv[i][3])/(2.0*PI);

      while(tmp<0.0){
	tmp += 1.0;
      }

      /* tmp is adjusted in between 0.0 and 1.0. */
      j = (int)tmp;
      be[i] = tmp - (double)j;
    }

    Pol2[spin][0] = be[1]*tv[1][1] + be[2]*tv[2][1] + be[3]*tv[3][1];
    Pol2[spin][1] = be[1]*tv[1][2] + be[2]*tv[2][2] + be[3]*tv[3][2];
    Pol2[spin][2] = be[1]*tv[1][3] + be[2]*tv[2][3] + be[3]*tv[3][3];

    /* unit is in Debye */

    Pol1[spin][0] *= -AU2Debye;
    Pol1[spin][1] *= -AU2Debye;
    Pol1[spin][2] *= -AU2Debye;

    Pol2[spin][0] *= -AU2Debye;
    Pol2[spin][1] *= -AU2Debye;
    Pol2[spin][2] *= -AU2Debye;

  } // spin

  /* for non-spin polarized case */

  if (SpinP_switch==0){
    Pol1[1][0] = Pol1[0][0];
    Pol1[1][1] = Pol1[0][1];
    Pol1[1][2] = Pol1[0][2];
    Pol2[1][0] = Pol2[0][0];
    Pol2[1][1] = Pol2[0][1];
    Pol2[1][2] = Pol2[0][2];
  }


  /*************************************************************
                    freeing of arrays
  *************************************************************/

  for (m=0; m<2; m++){
    for (i=0; i<3; i++){
      free(A11[m][i]);
    }
    free(A11[m]);
  }
  free(A11);

  for (m=0; m<2; m++){
    for (i=0; i<3; i++){
      free(A12[m][i]);
    }
    free(A12[m]);
  }
  free(A12);

  for (m=0; m<2; m++){
    for (i=0; i<3; i++){
      free(A21[m][i]);
    }
    free(A21[m]);
  }
  free(A21);

  for (m=0; m<2; m++){
    for (i=0; i<3; i++){
      free(A22[m][i]);
    }
    free(A22[m]);
  }
  free(A22);

  free(eval);
}


void Calc_ABC_Full( int spin, double k1, double k2, double k3, 
                    double **PF1, double **PF2, 
                    dcomplex ***ABC11, dcomplex ***ABC12, 
                    dcomplex ***ABC21, dcomplex ***ABC22, 
                    double *eval )
{
  int i,j,k,m,n,xyz_i,l1,l2,l3,Rn;
  int lm,ln,gm,po,lwork,info,GA_AN,LB_AN,GB_AN,tnoA,tnoB,Anum,Bnum,*ipiv;
  double si,co,kRn,tmp1,d,en,em,x,y,z,s,sx,sy,sz,Rx,Ry,Rz;  
  double del=1.0e-20;
  dcomplex ctmp1,ctmp2,csum,*Hk,*Sk,**dHk,**dSk,*WF;
  dcomplex *B11,*B12,*B21,*B22,*C12,*C21,*A12,*A21;
  dcomplex alpha = {1.0,0.0}; 
  dcomplex beta = {0.0,0.0};

  /*************************************************************
                      allocation of arrays
  *************************************************************/

  Hk = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));
  Sk = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));
  WF = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));

  dHk = (dcomplex**)malloc(sizeof(dcomplex*)*3);
  for (i=0; i<3; i++){
    dHk[i] = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));
  }

  dSk = (dcomplex**)malloc(sizeof(dcomplex*)*3);
  for (i=0; i<3; i++){
    dSk[i] = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));
  }

  A12 = (dcomplex*)malloc(sizeof(dcomplex)*No[spin]*Nuo[spin]);
  A21 = (dcomplex*)malloc(sizeof(dcomplex)*No[spin]*Nuo[spin]);
  B11 = (dcomplex*)malloc(sizeof(dcomplex)*No[spin]*No[spin]);
  B12 = (dcomplex*)malloc(sizeof(dcomplex)*No[spin]*Nuo[spin]);
  B21 = (dcomplex*)malloc(sizeof(dcomplex)*No[spin]*Nuo[spin]);
  B22 = (dcomplex*)malloc(sizeof(dcomplex)*Nuo[spin]*Nuo[spin]);
  C12 = (dcomplex*)malloc(sizeof(dcomplex)*No[spin]*Nuo[spin]);
  C21 = (dcomplex*)malloc(sizeof(dcomplex)*No[spin]*Nuo[spin]);

  /* initialize ABC11, ABC12, ABC21, and ABC22 */

  for (m=0; m<2; m++){
    for (i=0; i<3; i++){
      for (j=0; j<No[spin]*No[spin]; j++){
	ABC11[m][i][j] = Complex(0.0,0.0);
      }
      for (j=0; j<No[spin]*Nuo[spin]; j++){
	ABC12[m][i][j] = Complex(0.0,0.0);
	ABC21[m][i][j] = Complex(0.0,0.0);
      }
      for (j=0; j<Nuo[spin]*Nuo[spin]; j++){
	ABC22[m][i][j] = Complex(0.0,0.0);
      }
    }
  }

  /*************************************************************
         diagonalize the generalized eigenvalue problem
  *************************************************************/

  /* make H(k) and S(k), and solve the generalized eigenvalue problem */ 

  Calc_Hk_Sk(Hk,Sk,spin,k1,k2,k3);
  lapack_zhegvx(MatDim,Hk,Sk,WF,eval);

  /************************************************************/
  /*************************************************************
     The 1st contribution:
     calculation of analytic Berry connection
  *************************************************************/
  /************************************************************/

  for (m=0; m<MatDim; m++){
    for (n=0; n<MatDim; n++){

      for (GA_AN=1; GA_AN<=atomnum; GA_AN++){

	tnoA = Total_NumOrbs[GA_AN];
	Anum = MP[GA_AN];

	for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){

	  GB_AN = natn[GA_AN][LB_AN];
	  Rn = ncn[GA_AN][LB_AN];
	  tnoB = Total_NumOrbs[GB_AN];
	  Bnum = MP[GB_AN];

	  l1 = atv_ijk[Rn][1];
	  l2 = atv_ijk[Rn][2];
	  l3 = atv_ijk[Rn][3];
	  Rx = (double)l1*tv[1][1] + (double)l2*tv[2][1] + (double)l3*tv[3][1];
	  Ry = (double)l1*tv[1][2] + (double)l2*tv[2][2] + (double)l3*tv[3][2];
	  Rz = (double)l1*tv[1][3] + (double)l2*tv[2][3] + (double)l3*tv[3][3];

	  kRn = k1*(double)l1 + k2*(double)l2 + k3*(double)l3;
	  si = sin(2.0*PI*kRn);
	  co = cos(2.0*PI*kRn);

	  for (i=0; i<tnoA; i++){
	    for (j=0; j<tnoB; j++){

	      s = OLP[GA_AN][LB_AN][i][j];
	      sx = OLPpo[0][0][GA_AN][LB_AN][i][j] + Gxyz[GA_AN][1]*s;
	      sy = OLPpo[1][0][GA_AN][LB_AN][i][j] + Gxyz[GA_AN][2]*s;
	      sz = OLPpo[2][0][GA_AN][LB_AN][i][j] + Gxyz[GA_AN][3]*s;

	      ctmp1.r = WF[m*MatDim+(Anum+i)].r*WF[n*MatDim+(Bnum+j)].r 
                       +WF[m*MatDim+(Anum+i)].i*WF[n*MatDim+(Bnum+j)].i;  

	      ctmp1.i = WF[m*MatDim+(Anum+i)].r*WF[n*MatDim+(Bnum+j)].i 
                       -WF[m*MatDim+(Anum+i)].i*WF[n*MatDim+(Bnum+j)].r;

	      ctmp2.r = ctmp1.r*co - ctmp1.i*si;
	      ctmp2.i = ctmp1.r*si + ctmp1.i*co;
    
	      /* store the result to A11, A12, A21, or A22 */

              if (m<No[spin] && n<No[spin]){
                ABC11[0][0][n*No[spin]+m].r += ctmp2.r*(sx - Rx*s); 
                ABC11[0][1][n*No[spin]+m].r += ctmp2.r*(sy - Ry*s); 
                ABC11[0][2][n*No[spin]+m].r += ctmp2.r*(sz - Rz*s); 

                ABC11[0][0][n*No[spin]+m].i += ctmp2.i*(sx - Rx*s); 
                ABC11[0][1][n*No[spin]+m].i += ctmp2.i*(sy - Ry*s); 
                ABC11[0][2][n*No[spin]+m].i += ctmp2.i*(sz - Rz*s); 
	      }

              else if (m<No[spin] && !(n<No[spin])){
                ABC12[0][0][(n-No[spin])*No[spin]+m].r += ctmp2.r*(sx - Rx*s); 
                ABC12[0][1][(n-No[spin])*No[spin]+m].r += ctmp2.r*(sy - Ry*s); 
                ABC12[0][2][(n-No[spin])*No[spin]+m].r += ctmp2.r*(sz - Rz*s); 

                ABC12[0][0][(n-No[spin])*No[spin]+m].i += ctmp2.i*(sx - Rx*s); 
                ABC12[0][1][(n-No[spin])*No[spin]+m].i += ctmp2.i*(sy - Ry*s); 
                ABC12[0][2][(n-No[spin])*No[spin]+m].i += ctmp2.i*(sz - Rz*s); 
	      }

              else if (!(m<No[spin]) && n<No[spin]){
                ABC21[0][0][n*Nuo[spin]+m-No[spin]].r += ctmp2.r*(sx - Rx*s); 
                ABC21[0][1][n*Nuo[spin]+m-No[spin]].r += ctmp2.r*(sy - Ry*s); 
                ABC21[0][2][n*Nuo[spin]+m-No[spin]].r += ctmp2.r*(sz - Rz*s); 

                ABC21[0][0][n*Nuo[spin]+m-No[spin]].i += ctmp2.i*(sx - Rx*s); 
                ABC21[0][1][n*Nuo[spin]+m-No[spin]].i += ctmp2.i*(sy - Ry*s); 
                ABC21[0][2][n*Nuo[spin]+m-No[spin]].i += ctmp2.i*(sz - Rz*s); 
	      }

              else{
                ABC22[0][0][(n-No[spin])*Nuo[spin]+m-No[spin]].r += ctmp2.r*(sx - Rx*s); 
                ABC22[0][1][(n-No[spin])*Nuo[spin]+m-No[spin]].r += ctmp2.r*(sy - Ry*s); 
                ABC22[0][2][(n-No[spin])*Nuo[spin]+m-No[spin]].r += ctmp2.r*(sz - Rz*s); 

                ABC22[0][0][(n-No[spin])*Nuo[spin]+m-No[spin]].i += ctmp2.i*(sx - Rx*s); 
                ABC22[0][1][(n-No[spin])*Nuo[spin]+m-No[spin]].i += ctmp2.i*(sy - Ry*s); 
                ABC22[0][2][(n-No[spin])*Nuo[spin]+m-No[spin]].i += ctmp2.i*(sz - Rz*s); 
	      }

	    } // j
	  } // i
	} // LB_AN
      } // GA_AN
    } // m  
  } // m  

  /************************************************************/
  /*************************************************************
     The 2nd contribution:
     calculation of analytic Berry connection
  *************************************************************/
  /************************************************************/

  /* calculate dHk/dk */    

  Calc_dHk_dSk(dHk,dSk,WF,spin,k1,k2,k3);  

  /* calculate B11 */    

  for (i=0; i<No[spin]; i++){
    for (j=0; j<No[spin]; j++){

      csum = Complex(0.0,0.0);
      for (k=0; k<MatDim; k++){
        csum.r += PF1[spin][i*MatDim+k]*WF[j*MatDim+k].r;
        csum.i += PF1[spin][i*MatDim+k]*WF[j*MatDim+k].i;
      }

      B11[j*No[spin]+i] = csum;
    }    
  }    
  
  /* calculate B12 */    

  for (i=0; i<No[spin]; i++){
    for (j=0; j<Nuo[spin]; j++){

      csum = Complex(0.0,0.0);
      for (k=0; k<MatDim; k++){
        csum.r += PF1[spin][i*MatDim+k]*WF[(j+No[spin])*MatDim+k].r;
        csum.i += PF1[spin][i*MatDim+k]*WF[(j+No[spin])*MatDim+k].i;
      }

      B12[j*No[spin]+i] = csum;
    }    
  }    

  /* calculate B21 */    

  for (i=0; i<Nuo[spin]; i++){
    for (j=0; j<No[spin]; j++){

      csum = Complex(0.0,0.0);
      for (k=0; k<MatDim; k++){
        csum.r += PF2[spin][i*MatDim+k]*WF[j*MatDim+k].r;
        csum.i += PF2[spin][i*MatDim+k]*WF[j*MatDim+k].i;
      }

      B21[j*Nuo[spin]+i] = csum;
    }    
  }    

  /* calculate B22 */    

  for (i=0; i<Nuo[spin]; i++){
    for (j=0; j<Nuo[spin]; j++){

      csum = Complex(0.0,0.0);
      for (k=0; k<MatDim; k++){
        csum.r += PF2[spin][i*MatDim+k]*WF[(j+No[spin])*MatDim+k].r;
        csum.i += PF2[spin][i*MatDim+k]*WF[(j+No[spin])*MatDim+k].i;
      }

      B22[j*Nuo[spin]+i] = csum;
    }    
  }    

  /* calculate the inverse B11 and inverse B22 */    
  
  Calc_Inverse(No[spin], B11); // inverse B11 is stored in B11 
  Calc_Inverse(Nuo[spin],B22); // inverse B22 is stored in B22 

  /* calculate B11^{-1} x B12 -> C12 */    
 
  m = No[spin]; n = Nuo[spin]; k = No[spin];
  zgemm_("N","N", &m, &n, &k, &alpha, B11, &m, B12, &k, &beta, C12, &m);
          
  /* calculate B22^{-1} x B21 -> C21 */    
 
  m = Nuo[spin]; n = No[spin]; k = Nuo[spin];
  zgemm_("N","N", &m, &n, &k, &alpha, B22, &m, B21, &k, &beta, C21, &m);

  /* loop for x, y, and z */

  for (xyz_i=0; xyz_i<3; xyz_i++){   

    /* construct A21_{x, y, or z} */

    for (n=0; n<Nuo[spin]; n++){

      ln = No[spin] + n;
      en = eval[ln];

      for (m=0; m<No[spin]; m++){

	em = eval[m];
	d = em - en;
	tmp1 = d/(d*d + del);

	A21[m*Nuo[spin]+n].r = (dHk[xyz_i][m*MatDim+ln].r-em*dSk[xyz_i][m*MatDim+ln].r)*tmp1;
	A21[m*Nuo[spin]+n].i = (dHk[xyz_i][m*MatDim+ln].i-em*dSk[xyz_i][m*MatDim+ln].i)*tmp1;

      } // n
    } // m

    /* calculate C12 x A21_{x, y, or z} -> B11 */

    m = No[spin]; n = No[spin]; k = Nuo[spin];
    zgemm_("N","N", &m, &n, &k, &alpha, C12, &m, A21, &k, &beta, B11, &m);

    /* construct A12_{x, y, or z} */

    for (m=0; m<Nuo[spin]; m++){ // for 2

      lm = No[spin] + m;
      em = eval[lm];

      for (n=0; n<No[spin]; n++){ // for 1

	en = eval[n];

	d = em - en;
	tmp1 = d/(d*d + del);

	A12[m*No[spin]+n].r = (dHk[xyz_i][lm*MatDim+n].r-em*dSk[xyz_i][lm*MatDim+n].r)*tmp1;
	A12[m*No[spin]+n].i = (dHk[xyz_i][lm*MatDim+n].i-em*dSk[xyz_i][lm*MatDim+n].i)*tmp1;

      } // n
    } // m

    /* calculate C21 x A12_{x, y, or z} -> B22 */

    m = Nuo[spin]; n = Nuo[spin]; k = No[spin];
    zgemm_("N","N", &m, &n, &k, &alpha, C21, &m, A12, &k, &beta, B22, &m);

    /* store the results to ABCxx[1] */

    for (j=0; j<No[spin]; j++){
      for (i=0; i<No[spin]; i++){
        ABC11[1][xyz_i][j*No[spin]+i].r = B11[j*No[spin]+i].i;
        ABC11[1][xyz_i][j*No[spin]+i].i =-B11[j*No[spin]+i].r;
      }
    }

    for (j=0; j<Nuo[spin]; j++){
      for (i=0; i<Nuo[spin]; i++){
        ABC22[1][xyz_i][j*Nuo[spin]+i].r = B22[j*Nuo[spin]+i].i;
        ABC22[1][xyz_i][j*Nuo[spin]+i].i =-B22[j*Nuo[spin]+i].r;
      }
    }            

    for (j=0; j<Nuo[spin]; j++){
      for (i=0; i<No[spin]; i++){
        ABC12[1][xyz_i][j*No[spin]+i].r =-A12[j*No[spin]+i].i;
        ABC12[1][xyz_i][j*No[spin]+i].i = A12[j*No[spin]+i].r;
      }
    }

    for (j=0; j<No[spin]; j++){
      for (i=0; i<Nuo[spin]; i++){
        ABC21[1][xyz_i][j*Nuo[spin]+i].r =-A21[j*Nuo[spin]+i].i;
        ABC21[1][xyz_i][j*Nuo[spin]+i].i = A21[j*Nuo[spin]+i].r;
      }
    }

  } // xyz_i

  /*************************************************************
                       freeing of arrays
  *************************************************************/

  free(Hk);
  free(Sk);
  free(WF);

  for (i=0; i<3; i++){
    free(dHk[i]);
  }
  free(dHk);

  for (i=0; i<3; i++){
    free(dSk[i]);
  }
  free(dSk);

  free(A12);
  free(A21);
  free(B11);
  free(B12);
  free(B21);
  free(B22);
  free(C12);
  free(C21);
}



void Calc_ABC_A11( int spin, double k1, double k2, double k3, 
                   double **PF1, dcomplex ***ABC11, double *eval )
{
  int i,j,k,m,n,xyz_i,l1,l2,l3,Rn;
  int lm,ln,gm,po,lwork,info,GA_AN,LB_AN,GB_AN,tnoA,tnoB,Anum,Bnum,*ipiv;
  double si,co,kRn,tmp1,d,en,em,x,y,z,s,sx,sy,sz,Rx,Ry,Rz;  
  double del=1.0e-20;
  dcomplex ctmp1,ctmp2,csum,*Hk,*Sk,**dHk,**dSk,*WF;
  dcomplex *B11,*B12,*C12,*A21;
  dcomplex alpha = {1.0,0.0}; 
  dcomplex beta = {0.0,0.0};

  /*************************************************************
                      allocation of arrays
  *************************************************************/

  Hk = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));
  Sk = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));
  WF = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));

  dHk = (dcomplex**)malloc(sizeof(dcomplex*)*3);
  for (i=0; i<3; i++){
    dHk[i] = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));
  }

  dSk = (dcomplex**)malloc(sizeof(dcomplex*)*3);
  for (i=0; i<3; i++){
    dSk[i] = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));
  }

  B11 = (dcomplex*)malloc(sizeof(dcomplex)*No[spin]*No[spin]);
  B12 = (dcomplex*)malloc(sizeof(dcomplex)*No[spin]*Nuo[spin]);
  C12 = (dcomplex*)malloc(sizeof(dcomplex)*No[spin]*Nuo[spin]);
  A21 = (dcomplex*)malloc(sizeof(dcomplex)*No[spin]*Nuo[spin]);

  /* initialize ABC11 */

  for (m=0; m<2; m++){
    for (i=0; i<3; i++){
      for (j=0; j<No[spin]*No[spin]; j++){
	ABC11[m][i][j] = Complex(0.0,0.0);
      }
    }
  }

  /*************************************************************
         diagonalize the generalized eigenvalue problem
  *************************************************************/

  /* make H(k) and S(k), and solve the generalized eigenvalue problem */ 

  Calc_Hk_Sk(Hk,Sk,spin,k1,k2,k3);
  lapack_zhegvx(MatDim,Hk,Sk,WF,eval);

  /************************************************************/
  /*************************************************************
     The 1st contribution:
     calculation of analytic Berry connection
  *************************************************************/
  /************************************************************/

  for (m=0; m<No[spin]; m++){
    for (n=0; n<No[spin]; n++){

      for (GA_AN=1; GA_AN<=atomnum; GA_AN++){

	tnoA = Total_NumOrbs[GA_AN];
	Anum = MP[GA_AN];

	for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){

	  GB_AN = natn[GA_AN][LB_AN];
	  Rn = ncn[GA_AN][LB_AN];
	  tnoB = Total_NumOrbs[GB_AN];
	  Bnum = MP[GB_AN];

	  l1 = atv_ijk[Rn][1];
	  l2 = atv_ijk[Rn][2];
	  l3 = atv_ijk[Rn][3];
	  Rx = (double)l1*tv[1][1] + (double)l2*tv[2][1] + (double)l3*tv[3][1];
	  Ry = (double)l1*tv[1][2] + (double)l2*tv[2][2] + (double)l3*tv[3][2];
	  Rz = (double)l1*tv[1][3] + (double)l2*tv[2][3] + (double)l3*tv[3][3];

	  kRn = k1*(double)l1 + k2*(double)l2 + k3*(double)l3;
	  si = sin(2.0*PI*kRn);
	  co = cos(2.0*PI*kRn);

	  for (i=0; i<tnoA; i++){
	    for (j=0; j<tnoB; j++){

	      s = OLP[GA_AN][LB_AN][i][j];
	      sx = OLPpo[0][0][GA_AN][LB_AN][i][j] + Gxyz[GA_AN][1]*s;
	      sy = OLPpo[1][0][GA_AN][LB_AN][i][j] + Gxyz[GA_AN][2]*s;
	      sz = OLPpo[2][0][GA_AN][LB_AN][i][j] + Gxyz[GA_AN][3]*s;

	      ctmp1.r = WF[m*MatDim+(Anum+i)].r*WF[n*MatDim+(Bnum+j)].r 
                       +WF[m*MatDim+(Anum+i)].i*WF[n*MatDim+(Bnum+j)].i;  

	      ctmp1.i = WF[m*MatDim+(Anum+i)].r*WF[n*MatDim+(Bnum+j)].i 
                       -WF[m*MatDim+(Anum+i)].i*WF[n*MatDim+(Bnum+j)].r;

	      ctmp2.r = ctmp1.r*co - ctmp1.i*si;
	      ctmp2.i = ctmp1.r*si + ctmp1.i*co;
    
	      /* store the result to A11 */

	      ABC11[0][0][n*No[spin]+m].r += ctmp2.r*(sx - Rx*s); 
	      ABC11[0][1][n*No[spin]+m].r += ctmp2.r*(sy - Ry*s); 
	      ABC11[0][2][n*No[spin]+m].r += ctmp2.r*(sz - Rz*s); 

	      ABC11[0][0][n*No[spin]+m].i += ctmp2.i*(sx - Rx*s); 
	      ABC11[0][1][n*No[spin]+m].i += ctmp2.i*(sy - Ry*s); 
	      ABC11[0][2][n*No[spin]+m].i += ctmp2.i*(sz - Rz*s); 

	    } // j
	  } // i
	} // LB_AN
      } // GA_AN
    } // m  
  } // m  

  /************************************************************/
  /*************************************************************
     The 2nd contribution:
     calculation of analytic Berry connection
  *************************************************************/
  /************************************************************/

  /* calculate dHk/dk */    

  Calc_dHk_dSk(dHk,dSk,WF,spin,k1,k2,k3);  

  /* calculate B11 */    

  for (i=0; i<No[spin]; i++){
    for (j=0; j<No[spin]; j++){

      csum = Complex(0.0,0.0);
      for (k=0; k<MatDim; k++){
        csum.r += PF1[spin][i*MatDim+k]*WF[j*MatDim+k].r;
        csum.i += PF1[spin][i*MatDim+k]*WF[j*MatDim+k].i;
      }

      B11[j*No[spin]+i] = csum;
    }    
  }    
  
  /* calculate B12 */    

  for (i=0; i<No[spin]; i++){
    for (j=0; j<Nuo[spin]; j++){

      csum = Complex(0.0,0.0);
      for (k=0; k<MatDim; k++){
        csum.r += PF1[spin][i*MatDim+k]*WF[(j+No[spin])*MatDim+k].r;
        csum.i += PF1[spin][i*MatDim+k]*WF[(j+No[spin])*MatDim+k].i;
      }

      B12[j*No[spin]+i] = csum;
    }    
  }    

  /* calculate the inverse B11 */    
  
  //Calc_Inverse(No[spin], B11); // inverse B11 is stored in B11 

  Calc_Inverse_SVD(No[spin], B11); // inverse B11 is stored in B11 

  /* calculate B11^{-1} x B12 -> C12 */    
 
  m = No[spin]; n = Nuo[spin]; k = No[spin];
  zgemm_("N","N", &m, &n, &k, &alpha, B11, &m, B12, &k, &beta, C12, &m);
          
  /* loop for x, y, and z */

  for (xyz_i=0; xyz_i<3; xyz_i++){   

    /* construct A21_{x, y, or z} */

    for (n=0; n<Nuo[spin]; n++){

      ln = No[spin] + n;
      en = eval[ln];

      for (m=0; m<No[spin]; m++){

	em = eval[m];
	d = em - en;
	tmp1 = d/(d*d + del);

	A21[m*Nuo[spin]+n].r = (dHk[xyz_i][m*MatDim+ln].r-em*dSk[xyz_i][m*MatDim+ln].r)*tmp1;
	A21[m*Nuo[spin]+n].i = (dHk[xyz_i][m*MatDim+ln].i-em*dSk[xyz_i][m*MatDim+ln].i)*tmp1;

      } // n
    } // m

    /* calculate C12 x A21_{x, y, or z} -> B11 */

    m = No[spin]; n = No[spin]; k = Nuo[spin];
    zgemm_("N","N", &m, &n, &k, &alpha, C12, &m, A21, &k, &beta, B11, &m);

    /* store the results to ABC11[1] */

    for (j=0; j<No[spin]; j++){
      for (i=0; i<No[spin]; i++){
        ABC11[1][xyz_i][j*No[spin]+i].r = B11[j*No[spin]+i].i;
        ABC11[1][xyz_i][j*No[spin]+i].i =-B11[j*No[spin]+i].r;
      }
    }

  } // xyz_i

  /*************************************************************
                       freeing of arrays
  *************************************************************/

  free(Hk);
  free(Sk);
  free(WF);

  for (i=0; i<3; i++){
    free(dHk[i]);
  }
  free(dHk);

  for (i=0; i<3; i++){
    free(dSk[i]);
  }
  free(dSk);

  free(A21);
  free(B11);
  free(B12);
  free(C12);
}




void Calc_Inverse(int n, dcomplex *A)
{
  int info,lwork;
  int *ipiv;
  dcomplex *work;

  lwork = 8*n;
  ipiv = (int*) malloc(sizeof(int)*n);
  work = (dcomplex*)malloc(sizeof(dcomplex)*lwork);

  zgetrf_(&n,&n,A,&n,ipiv,&info);
  zgetri_(&n,A,&n,ipiv,work,&lwork,&info);

  free(ipiv);
  free(work);
}


void Calc_Inverse_SVD(int n, dcomplex *A)
{
  int i,j,lwork,info,M,N,K;
  double *sv,*rwork;
  dcomplex *U,*VT,*work;
  dcomplex alpha = {1.0,0.0}; 
  dcomplex beta = {0.0,0.0};

  //dcomplex *TmpA;

  /* allocation of arrays */

  lwork = 5*n;
  sv = (double*)malloc(sizeof(double)*n);
  U = (dcomplex*)malloc(sizeof(dcomplex)*n*n);
  VT = (dcomplex*)malloc(sizeof(dcomplex)*n*n);
  work = (dcomplex*)malloc(sizeof(dcomplex)*lwork);
  rwork = (double*)malloc(sizeof(double)*lwork);

  /*
  TmpA = (dcomplex*)malloc(sizeof(dcomplex)*n*n);

  for (i=0; i<n; i++){
    for (j=0; j<n; j++){
      TmpA[i*n+j] = A[i*n+j];
    }
  }
  */

  /* SVD of A */

  zgesvd_("A","A",&n,&n,A,&n,sv,U,&n,VT,&n,work,&lwork,rwork,&info);

  for (i=0; i<n; i++){
    printf("AAA i=%2d sv=%18.15f\n",i,sv[i]);
  }

  /* calculation of the inverse A */

  for (i=0; i<n; i++){
    for (j=0; j<n; j++){
      A[i*n+j].r =  VT[j*n+i].r/sv[i];
      A[i*n+j].i = -VT[j*n+i].i/sv[i];
    }
  }
  
  M = n; N = n; K = n;
  zgemm_("N","C", &M, &N, &K, &alpha, A, &M, U, &K, &beta, VT, &M);

  for (i=0; i<n; i++){
    for (j=0; j<n; j++){
      A[i*n+j] = VT[i*n+j];
    }
  }


  /*
  M = n; N = n; K = n;
  zgemm_("N","N", &M, &N, &K, &alpha, A, &M, TmpA, &K, &beta, VT, &M);

  printf("(IA*A).r\n");
  for (i=0; i<n; i++){
    for (j=0; j<n; j++){
      printf("%5.3f ",VT[j*n+i].r);
    }
    printf("\n");
  }

  printf("(IA*A).i\n");
  for (i=0; i<n; i++){
    for (j=0; j<n; j++){
      printf("%5.3f ",VT[j*n+i].i);
    }
    printf("\n");
  }

  MPI_Finalize();
  exit(0);
  */

  /* freeing of arrays */

  free(sv);
  free(U);
  free(VT);
  free(work);
  free(rwork);

  //free(TmpA);
}


void Calc_Projection_Functions(double **PF1, double **PF2)
{
  int i,j,k,mu,ii,ij,ik,jj,kk,kloop,spin,p1,p2,p3;
  int ***k_op;
  double k1,k2,k3,tmp;
  double *DM1,*DM2,*eval;
  dcomplex *Hk,*Sk,*WF;

  /*************************************************************
                      allocation of arrays
  *************************************************************/

  DM1 = (double*)malloc(sizeof(double)*MatDim*MatDim);
  DM2 = (double*)malloc(sizeof(double)*MatDim*MatDim);
  Hk = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));
  Sk = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));
  WF = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));
  eval = (double*)malloc(sizeof(double)*MatDim);

  k_op = (int***)malloc(sizeof(int**)*Nk1);
  for (i=0; i<Nk1; i++){
    k_op[i] = (int**)malloc(sizeof(int*)*Nk2);
    for (j=0; j<Nk2; j++){
      k_op[i][j] = (int*)malloc(sizeof(int)*Nk3);
    }
  }

  /*************************************************************
     k-points by a Gamma-centered mesh

     k_op[i][j][k]: weight of DOS 
                 =0  no calc.
                 =1  time reversal invariant momentum (TRIM)
                 =2  which has k<->-k point
        Now, only the relation, E(k)=E(-k), is used. 
  *************************************************************/

  for (i=0; i<Nk1; i++) {
    for (j=0; j<Nk2; j++) {
      for (k=0; k<Nk3; k++) {
	k_op[i][j][k] = -999;
      }
    }
  }

  for (i=0; i<Nk1; i++) {
    for (j=0; j<Nk2; j++) {
      for (k=0; k<Nk3; k++) {
	if ( k_op[i][j][k]==-999 ) {

	  if (i==0 || 2*i==Nk1){
	    ii = i;
	  } else {
	    ii = Nk1-i;
	  }

	  if (j==0 || 2*j==Nk2){
	    ij = j;
	  } else {
	    ij = Nk2-j;
	  }

	  if (k==0 || 2*k==Nk3){
	    ik = k;
	  } else {
	    ik = Nk3-k;
	  }

	  if ((i==0 || 2*i==Nk1) && (j==0 || 2*j==Nk2) && (k==0 || 2*k==Nk3)){
	    k_op[i][j][k]    = 1;
	  } 
	  else {
	    k_op[i][j][k]    = 2;
	    k_op[ii][ij][ik] = 0;
	  }
	}

      } /* k */
    } /* j */
  } /* i */

  /* one-dimentionalize k-points */

  T_knum = 0;
  for (i=0; i<Nk1; i++){
    for (j=0; j<Nk2; j++){
      for (k=0; k<Nk3; k++){
	if (0<k_op[i][j][k]){ T_knum++;	}
      }
    }
  }

  /* allocation of arrays, which will be freed in allocate_free_arrays(). */

  T_KGrids1 = (double*)malloc(sizeof(double)*T_knum);
  T_KGrids2 = (double*)malloc(sizeof(double)*T_knum);
  T_KGrids3 = (double*)malloc(sizeof(double)*T_knum);
  T_k_op = (int*)malloc(sizeof(int)*T_knum);
  TK2i = (int*)malloc(sizeof(int)*T_knum);
  TK2j = (int*)malloc(sizeof(int)*T_knum);
  TK2k = (int*)malloc(sizeof(int)*T_knum);

  /* set T_KGrid1,2,3, T_k_op, TK2i,j,k */

  T_knum = 0;
  for (i=0; i<Nk1; i++){

    k1 = ((double)i)/((double)Nk1);

    for (j=0; j<Nk2; j++){

      k2 = ((double)j)/((double)Nk2);

      for (k=0; k<Nk3; k++){

	k3 = ((double)k)/((double)Nk3);

	if (0<k_op[i][j][k]){
	  T_KGrids1[T_knum] = k1;
	  T_KGrids2[T_knum] = k2;
	  T_KGrids3[T_knum] = k3;
	  T_k_op[T_knum]    = k_op[i][j][k];
          TK2i[T_knum]     = i;
          TK2j[T_knum]     = j;
          TK2k[T_knum]     = k;

	  T_knum++;
	}
      }
    }
  }

  /* the sum of weights */

  tweight = 0.0;
  for (kloop=0; kloop<T_knum; kloop++){
    tweight += (double)T_k_op[kloop];
  }

  /******************************************************************/
  /*******************************************************************
      Calculate projection functions by a covariance method
  *******************************************************************/
  /******************************************************************/

  for (spin=0; spin<(SpinP_switch+1); spin++){ 

    /* initialize DM1 and DM2 */

    for (i=0; i<MatDim*MatDim; i++){
      DM1[i] = 0.0;
      DM2[i] = 0.0;
    }

    /* loop for k-points */

    for (kloop=0; kloop<T_knum; kloop++){

      /* get the basic information */ 

      p1 = TK2i[kloop];
      p2 = TK2j[kloop];
      p3 = TK2k[kloop];

      k1 = T_KGrids1[kloop]; 
      k2 = T_KGrids2[kloop]; 
      k3 = T_KGrids3[kloop];

      /* make H(k) and S(k), and solve the generalized eigenvalue problem */ 

      Calc_Hk_Sk(Hk,Sk,spin,k1,k2,k3);
      lapack_zhegvx(MatDim,Hk,Sk,WF,eval);

      /* calculate DM1 and DM2 at the k-point */ 

      for (mu=0; mu<No[spin]; mu++){
	for (i=0; i<MatDim; i++){
	  for (j=0; j<MatDim; j++){
            tmp = WF[mu*MatDim+i].r*WF[mu*MatDim+j].r + WF[mu*MatDim+i].i*WF[mu*MatDim+j].i;
            DM1[i*MatDim+j] -= (double)T_k_op[kloop]*tmp;
	  }
	}      
      }

      for (mu=No[spin]; mu<MatDim; mu++){
	for (i=0; i<MatDim; i++){
	  for (j=0; j<MatDim; j++){
            tmp = WF[mu*MatDim+i].r*WF[mu*MatDim+j].r + WF[mu*MatDim+i].i*WF[mu*MatDim+j].i;
            DM2[i*MatDim+j] -= (double)T_k_op[kloop]*tmp;
	  }
	}      
      }

    } // end of kloop

    /****************************************
      calculations of projection functions
    ****************************************/

    /* normalize DM1 and DM2 */

    for (i=0; i<MatDim*MatDim; i++){ DM1[i] /= tweight; }
    for (i=0; i<MatDim*MatDim; i++){ DM2[i] /= tweight; }

    /* diagonalize DM1 and store eigenvectors */

    lapack_dsyevx(MatDim, 1, No[spin],  DM1, PF1[spin], eval);

    /*
    for (i=0; i<No[spin]; i++){
      printf("DM1 %i %15.12f\n",i,eval[i]);
    }
    */

    lapack_dsyevx(MatDim, 1, Nuo[spin], DM2, PF2[spin], eval);

    /*
    printf("\n"); 
    for (i=0; i<Nuo[spin]; i++){
      printf("DM2 %i %15.12f\n",i,eval[i]);
    }
    */

  } // spin

  /*
  MPI_Finalize();
  exit(0);
  */

  /*************************************************************
                      freeing of arrays
  *************************************************************/

  free(DM1);
  free(DM2);
  free(Hk);
  free(Sk);
  free(WF);
  free(eval);

  for (i=0; i<Nk1; i++){
    for (j=0; j<Nk2; j++){
      free(k_op[i][j]);
    }
    free(k_op[i]);
  }
  free(k_op);
}












void Calc_Hk_Sk(dcomplex *Hk, dcomplex *Sk, int spin, double k1, double k2, double k3)
{
  int i,j,GA_AN,LB_AN,GB_AN,Rn,tnoA,tnoB,Anum,Bnum,l1,l2,l3;
  double co,si,kRn,s,h;

  /* initiaize Hk and Sk */  

  for (i=0; i<(MatDim*MatDim); i++){
    Hk[i] = Complex(0.0,0.0);
    Sk[i] = Complex(0.0,0.0);
  } 

  /* calculate Hk and Sk */  

  for (GA_AN=1; GA_AN<=atomnum; GA_AN++){
    tnoA = Total_NumOrbs[GA_AN];
    Anum = MP[GA_AN];

    for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
      GB_AN = natn[GA_AN][LB_AN];
      Rn = ncn[GA_AN][LB_AN];
      tnoB = Total_NumOrbs[GB_AN];

      l1 = atv_ijk[Rn][1];
      l2 = atv_ijk[Rn][2];
      l3 = atv_ijk[Rn][3];
      kRn = k1*(double)l1 + k2*(double)l2 + k3*(double)l3;

      si = sin(2.0*PI*kRn);
      co = cos(2.0*PI*kRn);
      Bnum = MP[GB_AN];

      for (i=0; i<tnoA; i++){
	for (j=0; j<tnoB; j++){
	  h = Hks[spin][GA_AN][LB_AN][i][j];
	  Hk[(Bnum+j)*MatDim+Anum+i].r += h*co;
	  Hk[(Bnum+j)*MatDim+Anum+i].i += h*si;

	  s = OLP[GA_AN][LB_AN][i][j];
	  Sk[(Bnum+j)*MatDim+Anum+i].r += s*co;
	  Sk[(Bnum+j)*MatDim+Anum+i].i += s*si;
	}
      }

    } // LB_AN
  } // GA_AN
}



void Calc_dHk_dSk(dcomplex **dHk, dcomplex **dSk, dcomplex *WF, 
                  int spin, double k1, double k2, double k3)
{
  int GA_AN,GB_AN,LB_AN,Anum,Bnum,tnoA,tnoB;
  int i,j,l1,l2,l3,n1,n2,n3,Rn;
  double co,si,kRn,Rx,Ry,Rz,s,h;
  dcomplex *work;
  dcomplex alpha = {1.0,0.0}; 
  dcomplex beta = {0.0,0.0};

  /* allocate of array */

  work = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));

  /* initialize dH/dkx, dH/dky, and dH/dkz
                dS/dkx, dS/dky, and dS/dkz represented by PAOs */

  for (i=0; i<(MatDim*MatDim); i++){ 
    dHk[0][i] = Complex(0.0,0.0); 
    dHk[1][i] = Complex(0.0,0.0); 
    dHk[2][i] = Complex(0.0,0.0); 
    dSk[0][i] = Complex(0.0,0.0); 
    dSk[1][i] = Complex(0.0,0.0); 
    dSk[2][i] = Complex(0.0,0.0); 
  }

  /* calculate dH/dkx, dH/dky, and dH/dkz
               dS/dkx, dS/dky, and dS/dkz represented by PAOs */

  for (GA_AN=1; GA_AN<=atomnum; GA_AN++){
    tnoA = Total_NumOrbs[GA_AN];
    Anum = MP[GA_AN];

    for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
      GB_AN = natn[GA_AN][LB_AN];
      Rn = ncn[GA_AN][LB_AN];
      tnoB = Total_NumOrbs[GB_AN];

      l1 = atv_ijk[Rn][1];
      l2 = atv_ijk[Rn][2];
      l3 = atv_ijk[Rn][3];

      Rx = (double)l1*tv[1][1] + (double)l2*tv[2][1] + (double)l3*tv[3][1];
      Ry = (double)l1*tv[1][2] + (double)l2*tv[2][2] + (double)l3*tv[3][2];
      Rz = (double)l1*tv[1][3] + (double)l2*tv[2][3] + (double)l3*tv[3][3];

      kRn = k1*(double)l1 + k2*(double)l2 + k3*(double)l3;
      si = sin(2.0*PI*kRn);
      co = cos(2.0*PI*kRn);
      Bnum = MP[GB_AN];

      for (i=0; i<tnoA; i++){
	for (j=0; j<tnoB; j++){

	  h = Hks[spin][GA_AN][LB_AN][i][j];
	  dHk[0][(Bnum+j)*MatDim+Anum+i].r +=-si*h*Rx;
	  dHk[0][(Bnum+j)*MatDim+Anum+i].i += co*h*Rx;
	  dHk[1][(Bnum+j)*MatDim+Anum+i].r +=-si*h*Ry;
	  dHk[1][(Bnum+j)*MatDim+Anum+i].i += co*h*Ry;
	  dHk[2][(Bnum+j)*MatDim+Anum+i].r +=-si*h*Rz;
	  dHk[2][(Bnum+j)*MatDim+Anum+i].i += co*h*Rz;

	  s = OLP[GA_AN][LB_AN][i][j];
	  dSk[0][(Bnum+j)*MatDim+Anum+i].r +=-si*s*Rx;
	  dSk[0][(Bnum+j)*MatDim+Anum+i].i += co*s*Rx;
	  dSk[1][(Bnum+j)*MatDim+Anum+i].r +=-si*s*Ry;
	  dSk[1][(Bnum+j)*MatDim+Anum+i].i += co*s*Ry;
	  dSk[2][(Bnum+j)*MatDim+Anum+i].r +=-si*s*Rz;
	  dSk[2][(Bnum+j)*MatDim+Anum+i].i += co*s*Rz;

	} // j
      } // i
    } // LB_AN
  } // GA_AN

  /* calculate dH/dkx, dH/dky, and dH/dkz
               dS/dkx, dS/dky, and dS/dkz represented by eigenstates */

  for (i=0; i<3; i++){

    zgemm_("N","N",&MatDim,&MatDim,&MatDim, 
	   &alpha, dHk[i], &MatDim, WF, &MatDim,
	   &beta, work, &MatDim);
    
    zgemm_("C","N",&MatDim,&MatDim,&MatDim, 
	   &alpha, WF, &MatDim, work, &MatDim,
	   &beta, dHk[i], &MatDim);

    zgemm_("N","N",&MatDim,&MatDim,&MatDim, 
	   &alpha, dSk[i], &MatDim, WF, &MatDim,
	   &beta, work, &MatDim);
    
    zgemm_("C","N",&MatDim,&MatDim,&MatDim, 
	   &alpha, WF, &MatDim, work, &MatDim,
	   &beta, dSk[i], &MatDim);

    /*
    printf("dSk.r\n");
    for (l1=0; l1<MatDim; l1++){
      for (l2=0; l2<MatDim; l2++){
        printf("%7.4f ",dSk[i][l2*MatDim+l1].r);
      }
      printf("\n");
    }    

    printf("dSk.i\n");
    for (l1=0; l1<MatDim; l1++){
      for (l2=0; l2<MatDim; l2++){
        printf("%7.4f ",dSk[i][l2*MatDim+l1].i);
      }
      printf("\n");
    }    
    MPI_Finalize();
    exit(0);
    */

  } // i

  /* freeing of array */

  free(work);
}













void lapack_zhegvx(int N, dcomplex *A, dcomplex *B, dcomplex *Z, double *W)
{
  char *JOBZ="V";
  char *UPLO="L";
  char *RANGE="A";
  int ITYPE,i,LDA,LDB,LDZ,IL,IU,LWORK;
  int *IWORK,*IFAIL,M,INFO;
  double ABSTOL=1.0e-14;
  double VL,VU;
  double *RWORK;
  dcomplex *WORK;

  /* allocation of arrays */

  ITYPE = 1;

  LDA = N;  LDB = N; LDZ = N;
  IL = 1;   IU = N;

  LWORK = 3*N;
  WORK  = (dcomplex*)malloc(sizeof(dcomplex)*LWORK);
  RWORK = (double*)malloc(sizeof(double)*7*N);
  IWORK = (int*)malloc(sizeof(int)*5*N);
  IFAIL = (int*)malloc(sizeof(int)*N);

  /* call zhegvx */

  zhegvx_(&ITYPE, JOBZ, RANGE, UPLO, &N, A, &LDA, B, &LDB, &VL, &VU, &IL, &IU,
	  &ABSTOL, &M, W, Z, &LDZ, WORK, &LWORK, RWORK,
	  IWORK, IFAIL, &INFO );

  /* freeing of arrays */

  free(WORK);
  free(RWORK);
  free(IWORK);
  free(IFAIL);
}




void lapack_zheevx(int N, dcomplex *A, dcomplex *WF, double *W)
{
  char *JOBZ="V";
  char *UPLO="L";
  char *RANGE="A";
  int i,LDA,LDZ,IL,IU,LWORK;
  int *IWORK,*IFAIL,M,INFO;
  double ABSTOL=1.0e-14;
  double VL,VU;
  double *RWORK;
  dcomplex *WORK,*Z;

  /* allocation of arrays */

  LDA = N;  LDZ = N;
  IL = 1;   IU = N;

  LWORK = 3*N;
  WORK  = (dcomplex*)malloc(sizeof(dcomplex)*LWORK);
  RWORK = (double*)malloc(sizeof(double)*7*N);
  IWORK = (int*)malloc(sizeof(int)*5*N);
  IFAIL = (int*)malloc(sizeof(int)*N);
  Z     = (dcomplex*)malloc(sizeof(dcomplex)*(N*N));

  /* call zheevx */

  zheevx_(JOBZ, RANGE, UPLO, &N, A, &LDA, &VL, &VU, &IL, &IU,
	  &ABSTOL, &M, W, Z, &LDZ, WORK, &LWORK, RWORK,
	  IWORK, IFAIL, &INFO );

  for (i=0; i<(N*N); i++) WF[i] = Z[i];

  /* freeing of arrays */

  free(WORK);
  free(RWORK);
  free(IWORK);
  free(IFAIL);
  free(Z);
}


void lapack_dsyevx(int N, int IL, int IU, double *A, double *WF, double *W)
{
  char *JOBZ="V";
  char *UPLO="L";
  char *RANGE="I";
  int i,LDA,LDZ,LWORK;
  int *IWORK,*IFAIL,M,INFO;
  double ABSTOL=1.0e-14;
  double VL,VU;
  double *WORK;

  /* allocation of arrays */

  LDA = N;  LDZ = N;

  LWORK = 8*N;
  WORK  = (double*)malloc(sizeof(double)*LWORK);
  IWORK = (int*)malloc(sizeof(int)*5*N);
  IFAIL = (int*)malloc(sizeof(int)*N);

  /* call zheevx */

  dsyevx_(JOBZ, RANGE, UPLO, &N, A, &LDA, &VL, &VU, &IL, &IU,
	  &ABSTOL, &M, W, WF, &LDZ, WORK, &LWORK, 
	  IWORK, IFAIL, &INFO );

  /* freeing of arrays */

  free(WORK);
  free(IWORK);
  free(IFAIL);
}








void allocate_free_arrays(char *mode)
{
  int GA_AN,spin,i,j,k;

  /***********************************************************
                 if (strcasecmp(mode,"read")==0)
  ***********************************************************/

  if (strcasecmp(mode,"allocate")==0){

    MP = (int*)malloc(sizeof(int)*(atomnum+1));
    MatDim = 0; 
    for (GA_AN=1; GA_AN<=atomnum; GA_AN++){
      MP[GA_AN] = MatDim;
      MatDim += Total_NumOrbs[GA_AN];
    } // GA_AN

    if (SpinP_switch==0){ 
      No[0] = Valence_Electrons/2;
      No[1] = Valence_Electrons/2;
      Nuo[0] = MatDim - No[0];
      Nuo[1] = MatDim - No[1];
    }
    else{
      No[0] = (Valence_Electrons+2*(int)(floor(Total_SpinS+1.0e-4)))/2;
      No[1] = (Valence_Electrons-2*(int)(floor(Total_SpinS+1.0e-4)))/2;
      Nuo[0] = MatDim - No[0];
      Nuo[1] = MatDim - No[1];
    }

    if (No[0]<No[1])  { Nomax  = No[1];  } else { Nomax  = No[0];  }
    if (Nuo[0]<Nuo[1]){ Nuomax = Nuo[1]; } else { Nuomax = Nuo[0]; }

    PF1 = (double**)malloc(sizeof(double*)*(SpinP_switch+1));
    for (spin=0; spin<=SpinP_switch; spin++){
      PF1[spin] = (double*)malloc(sizeof(double)*No[spin]*MatDim);
    }

    PF2 = (double**)malloc(sizeof(double*)*(SpinP_switch+1));
    for (spin=0; spin<=SpinP_switch; spin++){
      PF2[spin] = (double*)malloc(sizeof(double)*Nuo[spin]*MatDim);
    }

  }

  /***********************************************************
                 if (strcasecmp(mode,"free")==0)
  ***********************************************************/

  else if (strcasecmp(mode,"free")==0){

    free(MP);

    for (spin=0; spin<=SpinP_switch; spin++){
      free(PF1[spin]);
    }
    free(PF1);

    for (spin=0; spin<=SpinP_switch; spin++){
      free(PF2[spin]);
    }
    free(PF2);

    free(T_KGrids1);
    free(T_KGrids2);
    free(T_KGrids3);
    free(T_k_op);
    free(TK2i);
    free(TK2j);
    free(TK2k);
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
  static double sum;
  sum = a[1]*b[1] + a[2]*b[2] + a[3]*b[3]; 
  return sum;
}

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
#include <dirent.h>    // opendir, readdir, closedir
#include <errno.h>     // errno
#include <sys/stat.h>  // lstat, struct stat, S_ISDIR, S_ISREG, S_ISLNK
#include <unistd.h>  
#include <time.h>
#include <sys/types.h>
#include <sys/times.h>
#include <sys/time.h> 
#include "mpi.h"
#include "read_scfout.h"
#include <omp.h>



/* define struct, global variables, and functions */

typedef struct { double r,i; } dcomplex;

#define print_data    0
#define measure_time  1
#define PI          3.1415926535897932384626
#define kB          0.00008617251324000000   /* eV/K          */          
#define AU2Debye    2.54174776               /* AU to Debey   */
#define AU2Mucm     5721.52891433            /* 100 e/bohr/bohr */
#define eV2Hartree  27.2113845                
#define conductivity_unit 4599847.9907856912 /* S/m = Mho/m = (Ohm*m)^{-1} = 4599847.9907856912 */
#define YOUSO10     500
#define Host_ID      0         /* ID of the host CPU in MPI */
#define fp_bsize   2097152     /* buffer size for setvbuf */
#define MIN_VAL  1.0e-12
#define MIN(x)  ( ( (x) ) < MIN_VAL ? MIN_VAL : (x) )
#define Aref(i,j,k) ((i)*MatDim*MatDim+(k)*MatDim+(j))

int job_id;
int MatDim,Nk1,Nk2,Nk3,T_knum;
int *MP,No[2],Nuo[2],Nomax,Nuomax;
char filename[YOUSO10],filepath[YOUSO10];

int *TK2i,*TK2j,*TK2k,*T_k_op,***k_op;
double tweight;
double *T_KGrids1,*T_KGrids2,*T_KGrids3;
double **PF1,**PF2,*****FR1,*****FR2;

double sgn(double nu);
int fact(int N);
dcomplex Cdiv(dcomplex a, dcomplex b);
dcomplex Cmul(dcomplex a, dcomplex b);
dcomplex Cadd(dcomplex a, dcomplex b);
dcomplex Csub(dcomplex a, dcomplex b);

void Polar_Decompose(int n, dcomplex *A, dcomplex *U);
void Set_Kpoints();

void generate_combinations(int p, int start, int depth,
                           int *current,
                           int ***combo_list, int *combo_count);
 
void init_independent_components(int p, int ***out_list, int *out_count);

int map_to_independent(int p, const int *idx,
                       int **ind_list, int ind_count);

int equal_indices(const int *a, const int *b, int p);


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

void dsygvx_(int *itype, char *jobz, char *range, char *uplo, 
             int *n, double *a, int *lda, double *b, int *ldb, 
             double *vl, double *vu, int *il, int *iu,
	     double *abstol, int *m, double *w, double *z,
	     int *ldz, double *work, int *lwork, int *iwork,
	     int *ifail, int *info);

void allocate_free_arrays(char *mode);
dcomplex Complex(double re, double im){ return (dcomplex){re, im}; }
void Cross_Product(double a[4], double b[4], double c[4]);
double Dot_Product(double a[4], double b[4]);

void lapack_zheevx(int N, dcomplex *A, dcomplex *WF, double *W);
void lapack_zhegvx(int N, dcomplex *A, dcomplex *B, dcomplex *Z, double *W);
void lapack_dsyevx(int N, int IL, int IU, double *A, double *WF, double *W);
void lapack_dsygvx(int N, int IL, int IU, double *A, double *B, double *Z, double *W);

dcomplex rval(int spin, int n, int m, 
              dcomplex ***A11, dcomplex ***A12, dcomplex ***A21, dcomplex ***A22,
              dcomplex r[3]);

dcomplex rval_gg( int spin, int n, int m, double *eval, double **omega_gg, 
                  dcomplex ***A11, dcomplex ***A12, dcomplex ***A21, dcomplex ***A22,
                  dcomplex r_gg[3][3] );

void Calc_Momentum_Matrix( int spin, double k1, double k2, double k3, 
                           dcomplex *WF, double *eval, dcomplex **P );

void Calc_Momentum_Matrix_CCLee( int spin, double k1, double k2, double k3, 
                                 dcomplex *WF, double *eval, dcomplex **P );
void Calc_FR_1(double **PF1);
void Calc_FR_1_2(double **PF1, double **PF2);
void Calc_D(double k1, double k2, double k3, dcomplex *WF, dcomplex *D);
void Calc_Hk_Sk(dcomplex *Hk, dcomplex *Sk, int spin, double k1, double k2, double k3);
void Calc_dHk_dSk(dcomplex **dHk, dcomplex **dSk, dcomplex *WF, 
                  int spin, double k1, double k2, double k3);

void Calc_Guiding_Functions_1(double **PF1);
void Calc_Guiding_Functions_1_2(double **PF1, double **PF2);
void Calc_S0(int spin, double k1, double k2, double k3, double *S0);

void Calc_Inverse(int n, dcomplex *A);
void Calc_Inverse_SVD(int n, dcomplex *A);

void Calc_Afull( int spin, 
                 double k1, double k2, double k3, 
		 dcomplex *WF, double *eval,  
		 double **PF1, double **PF2, 
		 dcomplex *A11 );

void Calc_A11_A12_A21_A22( int spin, double k1, double k2, double k3, 
                           dcomplex *WF, double *eval,  
                           double **PF1, double **PF2, 
                           dcomplex ***A11, dcomplex ***A12, 
                           dcomplex ***A21, dcomplex ***A22 );

void Calc_A11( int spin, double k1, double k2, double k3,
               dcomplex *WF, double *eval,                 
               double **PF1, dcomplex ***A11 );

void Calc_A12_A21( int spin, double k1, double k2, double k3, 
                   dcomplex *WF, double *eval, 
                   dcomplex ***A12, dcomplex ***A21 );

void Check_Anti_Hermiticity(int spin, dcomplex ***A11, dcomplex ***A12, dcomplex ***A21, dcomplex ***A22);
void Dispersion_Berry_Connections();
void Polarization();
void Chern_Number();
void Linear_Opt_Conductivity();
void Linear_Opt_Conductivity_Berry_Connection();
void SHG();
void pth_Harmonic_Generation_sigma(int pth);
void pth_Harmonic_Generation_chi(int pth);
void pth_Harmonic_Generation(int pth);

int remove_files_in_dir(const char *dirpath, int *removed_count);
int find_index(int p, int *list, int xyz, int ***independent_list, int *independent_count);
double fread_rho(char filename[YOUSO10], int p, int I, int spin, int ki, int kj, int kk, dcomplex *rho);
double fread_A(char filename[YOUSO10], int spin, int ki, int kj, int kk, dcomplex *Afull);
double fwrite_rho(char filename[YOUSO10], int p, int I, int spin, int ki, int kj, int kk, dcomplex *rho);
void dtime(double *t);


int main(int argc, char *argv[]) 
{

  /* MPI initialization */

  MPI_Init(&argc, &argv);

  /******************************************************
    set job_id 

    1: Chern number
    2: Dispersion of Berry connection
    3: Dispersion of Berry curvature 
    4: Polarization
    5: Linear optical conductivity
    6: Linear optical conductivity by Berry connection
    7: Second harmonic generation (SHG)
    8: pth-order optical susceptibility
  *******************************************************/

  job_id = 7;

  /* set the number of k-points */
 
  Nk1 = 9; 
  Nk2 = 9;
  Nk3 = 9; 

  /* read the .scfout file and allocate arrays */

  read_scfout(argv);
  allocate_free_arrays("allocate");

  /* set k-point information */

  Set_Kpoints(); 

  /* Chern number */

  if (job_id==1)      Chern_Number();

  /* Berry connections along the designated k-paths */

  else if (job_id==2) Dispersion_Berry_Connections();

  /* Berry curvature along the designated k-paths */

  else if (job_id==3) ; // Dispersion_Berry_Curvature();

  /* Polarization */

  else if (job_id==4) Polarization();

  /* Linear optical conductivity */

  else if (job_id==5) Linear_Opt_Conductivity();

  /* Linear optical conductivity by Berry connection */

  else if (job_id==6) Linear_Opt_Conductivity_Berry_Connection();

  /* Second harmonic generation (SHG) */

  else if (job_id==7) SHG();

  /* pth-order susceptibility */

  else if (job_id==8) pth_Harmonic_Generation(2);

  /* freeing of arrays */

  allocate_free_arrays("free");

  MPI_Finalize();
  exit(0);
}





void SHG()
{
  int i,j,k,l,spin,kloop,m,n,Ng,a,b,c,p;
  double k1,k2,k3,tmp1,tmp2,omega,omega_min,omega_max,domega,eta,scissor;
  double fl,fn,fm,en,em,el,x,max_x=30.0,Beta,wmn,wml,wln,fln,fnm,fml,vtmp[4];
  double *eval,**omega_gg,sum0,sum1,sum2,w1,w2,w3,w4,Dmna,Dmnb,Dmnc,Cell_Volume;
  dcomplex cw1,cw2,cw3,cw4,cw5,cw3a,cw3b,cw;
  dcomplex rnm[3],rml[3],rln[3],rmn[3],ctmp1,ctmp2,ctmp3,ctmp4;
  dcomplex rnm_gg[3][3],rmn_gg[3][3],term1,term2,term3,term4,term5;
  dcomplex ***A11,***A12,***A21,***A22,*Hk,*Sk,*WF,**dHk,**dSk;
  dcomplex chi0[3][3][3],****chi_e,****chi_i;
  char nanchar1[300],nanchar2[300];

  /* set parameters */

  Ng = 100;
  omega_min = 0.0;               
  omega_max = 7.0/eV2Hartree;   
  domega = (omega_max - omega_min)/(double)(Ng-1);
  eta = 0.01/eV2Hartree; 
  Beta = 10000.0;
  scissor = 1.151/eV2Hartree; 
  //scissor = 0.0/eV2Hartree; 
    
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

  Hk = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));
  Sk = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));
  WF = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));
  eval = (double*)malloc(sizeof(double)*MatDim);

  omega_gg = (double**)malloc(sizeof(double*)*3);
  for (i=0; i<3; i++){
    omega_gg[i] = (double*)malloc(sizeof(double)*MatDim);
  }

  dHk = (dcomplex**)malloc(sizeof(dcomplex*)*3);
  for (i=0; i<3; i++){
    dHk[i] = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));
  }

  dSk = (dcomplex**)malloc(sizeof(dcomplex*)*3);
  for (i=0; i<3; i++){
    dSk[i] = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));
  }

  chi_e = (dcomplex****)malloc(sizeof(dcomplex***)*3);
  for (a=0; a<3; a++){
    chi_e[a] = (dcomplex***)malloc(sizeof(dcomplex**)*3);
    for (b=0; b<3; b++){
      chi_e[a][b] = (dcomplex**)malloc(sizeof(dcomplex*)*3);
      for (c=0; c<3; c++){
        chi_e[a][b][c] = (dcomplex*)malloc(sizeof(dcomplex)*Ng);
        for (i=0; i<Ng; i++){
          chi_e[a][b][c][i] = Complex(0.0,0.0);
	}
      }
    }
  }

  chi_i = (dcomplex****)malloc(sizeof(dcomplex***)*3);
  for (a=0; a<3; a++){
    chi_i[a] = (dcomplex***)malloc(sizeof(dcomplex**)*3);
    for (b=0; b<3; b++){
      chi_i[a][b] = (dcomplex**)malloc(sizeof(dcomplex*)*3);
      for (c=0; c<3; c++){
        chi_i[a][b][c] = (dcomplex*)malloc(sizeof(dcomplex)*Ng);
        for (i=0; i<Ng; i++){
          chi_i[a][b][c][i] = Complex(0.0,0.0);
	}
      }
    }
  }

  /* calculte cell volume */

  Cross_Product(tv[2],tv[3],vtmp);
  Cell_Volume = fabs(Dot_Product(tv[1],vtmp));

  /*************************************************************
          calculation of guiding functions and FR
  *************************************************************/

  /* calculate guiding functions, PF1 and PF2 */

  Calc_Guiding_Functions_1_2(PF1,PF2);

  /* calculate FR1 and FR2 */

  Calc_FR_1_2(PF1,PF2);  

  /*************************************************************
                    loops for spin and k-points
  *************************************************************/

  for (spin=0; spin<=SpinP_switch; spin++){ 

    /* loop for k-points */

    for (kloop=0; kloop<T_knum; kloop++){

      k1 = T_KGrids1[kloop]; 
      k2 = T_KGrids2[kloop]; 
      k3 = T_KGrids3[kloop];

      /* make H(k) and S(k), and solve the generalized eigenvalue problem */ 

      Calc_Hk_Sk(Hk,Sk,spin,k1,k2,k3);
      lapack_zhegvx(MatDim,Hk,Sk,WF,eval);

      /* apply the scissor operator */

      for (i=No[spin]; i<MatDim; i++){
        eval[i] += scissor; 
      }

      /* calculate A11, A12, A21, and A22 */

      Calc_A11_A12_A21_A22(spin,k1,k2,k3,WF,eval,PF1,PF2,A11,A12,A21,A22);

      /* set Afull */      

      /*
      for (p=0; p<3; p++){ 
	for (i=0; i<No[spin]; i++){
	  for (j=0; j<No[spin]; j++){
	    Afull[p][j*MatDim+i] = Cadd(A11[0][p][j*No[spin]+i],A11[1][p][j*No[spin]+i]);
	  }
	}
      }

      for (p=0; p<3; p++){ 
	for (i=0; i<No[spin]; i++){
	  for (j=0; j<Nuo[spin]; j++){
	    Afull[p][(j+No[spin])*MatDim+i] = Cadd(A12[0][p][j*No[spin]+i],A12[1][p][j*No[spin]+i]);
	  }
	}
      }

      for (p=0; p<3; p++){ 
	for (i=0; i<Nuo[spin]; i++){
	  for (j=0; j<No[spin]; j++){
	    Afull[p][j*MatDim+i+No[spin]] = Cadd(A21[0][p][j*Nuo[spin]+i],A21[1][p][j*Nuo[spin]+i]);
	  }
	}
      }

      for (p=0; p<3; p++){ 
	for (i=0; i<Nuo[spin]; i++){
	  for (j=0; j<Nuo[spin]; j++){
	    Afull[p][(j+No[spin])*MatDim+i+No[spin]] = Cadd(A22[0][p][j*Nuo[spin]+i],A22[1][p][j*Nuo[spin]+i]);
	  }
	}
      }

      for (p=0; p<3; p++){ 
	for (i=0; i<MatDim; i++){
          Afull[p][i*MatDim+i] = Complex(0.0,0.0);
	}
      }
      */

      /* calculate dHk and dSk */

      Calc_dHk_dSk(dHk,dSk,WF,spin,k1,k2,k3);  
 
      /* calculate omega_gg */

      for (n=0; n<MatDim; n++){
        omega_gg[0][n] = dHk[0][n*MatDim+n].r - eval[n]*dSk[0][n*MatDim+n].r;
        omega_gg[1][n] = dHk[1][n*MatDim+n].r - eval[n]*dSk[1][n*MatDim+n].r;
        omega_gg[2][n] = dHk[2][n*MatDim+n].r - eval[n]*dSk[2][n*MatDim+n].r;
      }      

      /*********************************************/
      /* calculation of SHG, contribution of chi_e */
      /*********************************************/

      for (p=0; p<Ng; p++){

        omega = omega_min + (double)p*domega;

        /* initialize chi0 */

	for (a=0; a<3; a++){
	  for (b=0; b<3; b++){
	    for (c=0; c<3; c++){
              chi0[a][b][c] = Complex(0.0,0.0);          
	    } // c
	  } // b
	} // a
	
        /* summation over n, m, and l */

	for (n=0; n<MatDim; n++){

	  en = eval[n];
	  x = (en - ChemP)*Beta;
	  if (x<=-max_x) x = -max_x;
	  if (max_x<=x)  x = max_x;
	  fn = 1.0/(1.0 + exp(x));

	  for (m=0; m<MatDim; m++){

	    if (n!=m){

	      em = eval[m];
	      x = (em - ChemP)*Beta;
	      if (x<=-max_x) x = -max_x;
	      if (max_x<=x)  x = max_x;
	      fm = 1.0/(1.0 + exp(x));

	      for (l=0; l<MatDim; l++){

		if (m!=l && n!=l){

		  el = eval[l];
		  x = (el - ChemP)*Beta;
		  if (x<=-max_x) x =-max_x;
		  if (max_x<=x)  x = max_x;
		  fl = 1.0/(1.0 + exp(x));

		  wln = el - en;
		  wml = em - el;
		  wmn = em - en;
		  fnm = fn - fm;
		  fln = fl - fn;
		  fml = fm - fl;

		  /*
		  tmp1 = (2.0*fnm/(wmn-2.0*omega) + fln/(wln-omega) + fml/(wml-omega))/(wln - wml);
		  */

                  ctmp1.r = wmn - 2.0*omega;                    
                  ctmp1.i = -2.0*eta;
                  ctmp2.r = 2.0*fnm;
                  ctmp2.i = 0.0;
                  cw1 = Cdiv(ctmp2,ctmp1);
                 
                  ctmp1.r = wln - omega;
                  ctmp1.i = -eta;
                  ctmp2.r = fln;
                  ctmp2.i = 0.0;
                  cw2 = Cdiv(ctmp2,ctmp1);

                  ctmp1.r = wml - omega;
                  ctmp1.i = -eta;
                  ctmp2.r = fml;
                  ctmp2.i = 0.0;
                  cw3 = Cdiv(ctmp2,ctmp1);

		  cw.r = (cw1.r + cw2.r + cw3.r)/(wln-wml);
		  cw.i = (cw1.i + cw2.i + cw3.i)/(wln-wml);

		  rval(spin, n, m, A11, A12, A21, A22, rnm);
		  rval(spin, m, l, A11, A12, A21, A22, rml);
		  rval(spin, l, n, A11, A12, A21, A22, rln);

		  for (a=0; a<3; a++){
		    for (b=0; b<3; b++){
		      for (c=0; c<3; c++){

                        ctmp1 = Cmul(rml[b],rln[c]); 
                        ctmp2 = Cmul(rml[c],rln[b]); 
                        ctmp3 = Cadd(ctmp1,ctmp2);
                        ctmp4 = Cmul(rnm[a],ctmp3);

                        chi0[a][b][c].r += ctmp4.r*cw.r - ctmp4.i*cw.i;
                        chi0[a][b][c].i += ctmp4.i*cw.r + ctmp4.r*cw.i;

		      } // c
		    } // b
		  } // a

		} // end of if (m!=l && n!=l)
	      } // l
	    } // end of if (n!=m)
	  } // m
	} // n

        /* add chi0 to chi_e */ 

	for (a=0; a<3; a++){
	  for (b=0; b<3; b++){
	    for (c=0; c<3; c++){
              /* multiply 0.5 and add it to chi_e */  
	      chi_e[a][b][c][p].r += 0.5*chi0[a][b][c].r*(double)T_k_op[kloop];
	      chi_e[a][b][c][p].i += 0.5*chi0[a][b][c].i*(double)T_k_op[kloop];
	    }
	  }
	}

      } // p

      /*********************************************/
      /* calculation of SHG, contribution of chi_i */
      /*********************************************/

      for (p=0; p<Ng; p++){

        omega = omega_min + (double)p*domega;

        /* initialize chi0 */

	for (a=0; a<3; a++){
	  for (b=0; b<3; b++){
	    for (c=0; c<3; c++){
              chi0[a][b][c] = Complex(0.0,0.0);          
	    } // c
	  } // b
	} // a
	
        /* summation over n and m */

	for (n=0; n<MatDim; n++){

	  en = eval[n];
	  x = (en - ChemP)*Beta;
	  if (x<=-max_x) x = -max_x;
	  if (max_x<=x)  x = max_x;
	  fn = 1.0/(1.0 + exp(x));

	  for (m=0; m<MatDim; m++){

	    if (n!=m){

	      em = eval[m];
	      x = (em - ChemP)*Beta;
	      if (x<=-max_x) x = -max_x;
	      if (max_x<=x)  x = max_x;
	      fm = 1.0/(1.0 + exp(x));

	      wmn = em - en;
	      fnm = fn - fm;

	      rval(spin, n, m, A11, A12, A21, A22, rnm);
	      rval(spin, m, n, A11, A12, A21, A22, rmn);

              rval_gg(spin, n, m, eval, omega_gg, A11, A12, A21, A22, rnm_gg);              
              rval_gg(spin, m, n, eval, omega_gg, A11, A12, A21, A22, rmn_gg);              

	      /*
	      w1 = 2.0/(wmn*(wmn-2.0*omega));
	      w2 = 1.0/(wmn*(wmn-omega));
              w3 = 1.0/(wmn*wmn)*(1.0/(wmn-omega)-4.0/(wmn-2.0*omega));
              w4 = -1.0/(2.0*wmn*(wmn-omega));
	      */

              ctmp1.r = wmn*(wmn - 2.0*omega);
              ctmp1.i = -2.0*wmn*eta;
              ctmp2.r = 2.0;
              ctmp2.i = 0.0;
              cw1 = Cdiv(ctmp2,ctmp1);

              ctmp1.r = wmn*(wmn - omega);
              ctmp1.i = -wmn*eta;
              ctmp2.r = 1.0;
              ctmp2.i = 0.0;
              cw2 = Cdiv(ctmp2,ctmp1);
                            
              ctmp1.r = wmn*wmn*(wmn - omega);
              ctmp1.i = -wmn*wmn*eta;
              ctmp2.r = 1.0;
              ctmp2.i = 0.0;
              cw3a = Cdiv(ctmp2,ctmp1);
              ctmp1.r = wmn*wmn*(wmn - 2.0*omega);
              ctmp1.i = -2.0*wmn*wmn*eta;
              ctmp2.r = -4.0;
              ctmp2.i = 0.0;
              cw3b = Cdiv(ctmp2,ctmp1);
              cw3.r = cw3a.r + cw3b.r;
              cw3.i = cw3a.i + cw3b.i;

              ctmp1.r = 2.0*wmn*(wmn - omega);
              ctmp1.i = -2.0*wmn*eta;
              ctmp2.r = -1.0;
              ctmp2.i = 0.0;
              cw4 = Cdiv(ctmp2,ctmp1);

              ctmp1.r = 4.0*wmn*wmn*(wmn - omega);
              ctmp1.i =-4.0*wmn*wmn*eta;
              ctmp2.r = 1.0;
              ctmp2.i = 0.0;
              cw5 = Cdiv(ctmp2,ctmp1);

	      for (a=0; a<3; a++){
		for (b=0; b<3; b++){
		  for (c=0; c<3; c++){

                    /* 1st term */

                    ctmp1 = Cadd(rnm_gg[b][c],rmn_gg[c][b]); 
                    ctmp2 = Cmul(rnm[a],ctmp1);
                    term1.r = cw1.r*ctmp2.r - cw1.i*ctmp2.i;       
                    term1.i = cw1.r*ctmp2.i + cw1.i*ctmp2.r;

                    /* 2nd term */
      
                    ctmp1 = Cmul(rnm_gg[a][c],rmn[b]); 
                    ctmp2 = Cmul(rnm_gg[a][b],rmn[c]); 
                    ctmp3 = Cadd(ctmp1,ctmp2);
                    term2.r = cw2.r*ctmp3.r - cw2.i*ctmp3.i;
                    term2.i = cw2.r*ctmp3.i + cw2.i*ctmp3.r;

                    /* 3rd term */

                    Dmnc = omega_gg[c][m] - omega_gg[c][n];   
                    Dmnb = omega_gg[b][m] - omega_gg[b][n];   
                    ctmp1.r = rmn[b].r*Dmnc + rmn[c].r*Dmnb; 
                    ctmp1.i = rmn[b].i*Dmnc + rmn[c].i*Dmnb; 
                    ctmp2 = Cmul(rnm[a],ctmp1);
                    term3.r = cw3.r*ctmp2.r - cw3.i*ctmp2.i;       
                    term3.i = cw3.r*ctmp2.i + cw3.i*ctmp2.r;

                    /* 4th term */

                    ctmp1 = Cmul(rnm_gg[b][a],rmn[c]);
                    ctmp2 = Cmul(rnm_gg[c][a],rmn[b]);
                    ctmp3 = Cadd(ctmp1,ctmp2);
                    term4.r = cw4.r*ctmp3.r - cw4.i*ctmp3.i;       
                    term4.i = cw4.r*ctmp3.i + cw4.i*ctmp3.r;       

                    /* 5th term */

                    Dmna = omega_gg[a][m] - omega_gg[a][n];   
                    ctmp1 = Cmul(rnm[b],rmn[c]); 
                    ctmp2 = Cmul(rnm[c],rmn[b]); 
                    ctmp3.r = Dmna*ctmp2.r;
                    ctmp3.i = Dmna*ctmp2.i;
                    term5.r = cw5.r*ctmp3.r - cw5.i*ctmp3.i;       
                    term5.i = cw5.r*ctmp3.i + cw5.i*ctmp3.r;       

                    /* add all the terms and add it to chi0 */

		    chi0[a][b][c].r += fnm*(term1.r + term2.r + term3.r + term4.r + term5.r);
		    chi0[a][b][c].i += fnm*(term1.i + term2.i + term3.i + term4.i + term5.i);

		  } // c
		} // b
	      } // a

	    } // end of if (n!=m)
	  } // m
	} // n

        /* add chi0 to chi_i */ 

	for (a=0; a<3; a++){
	  for (b=0; b<3; b++){
	    for (c=0; c<3; c++){
              /* multiply 0.5i and add it to chi_i */  
	      chi_i[a][b][c][p].r +=-0.5*chi0[a][b][c].i*(double)T_k_op[kloop];
	      chi_i[a][b][c][p].i += 0.5*chi0[a][b][c].r*(double)T_k_op[kloop];
	    } // c
	  } // b
	} // a

      } // p

    } /* end of kloop */ 

    /* taking account of the prefactor */ 

    tmp1 = 1.0/(tweight*Cell_Volume);

    for (a=0; a<3; a++){
      for (b=0; b<3; b++){
	for (c=0; c<3; c++){
          for (p=0; p<Ng; p++){
	    chi_e[a][b][c][p].r *= tmp1;
	    chi_e[a][b][c][p].i *= tmp1;
	    chi_i[a][b][c][p].r *= tmp1;
	    chi_i[a][b][c][p].i *= tmp1;
	  } // p
	} // c
      } // b
    } // a

  } /* end of spin */


  a = 0; b = 1; c = 2; 
  //a = 1; b = 1; c = 1; 
  for (p=0; p<Ng; p++){
    omega = omega_min + (double)p*domega;
    printf("%15.12f %15.12f %15.12f\n",
           omega,chi_e[a][b][c][p].r+chi_i[a][b][c][p].r,chi_e[a][b][c][p].i+chi_i[a][b][c][p].i);
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

  free(Hk);
  free(Sk);
  free(WF);
  free(eval);

  for (i=0; i<3; i++){
    free(omega_gg[i]);
  }
  free(omega_gg);

  for (i=0; i<3; i++){
    free(dHk[i]);
  }
  free(dHk);

  for (i=0; i<3; i++){
    free(dSk[i]);
  }
  free(dSk);

  for (a=0; a<3; a++){
    for (b=0; b<3; b++){
      for (c=0; c<3; c++){
        free(chi_e[a][b][c]);
      }
      free(chi_e[a][b]);
    }
    free(chi_e[a]);
  }
  free(chi_e);

  for (a=0; a<3; a++){
    for (b=0; b<3; b++){
      for (c=0; c<3; c++){
        free(chi_i[a][b][c]);
      }
      free(chi_i[a][b]);
    }
    free(chi_i[a]);
  }
  free(chi_i);

}


dcomplex rval(int spin, int n, int m,
              dcomplex ***A11, dcomplex ***A12, dcomplex ***A21, dcomplex ***A22,
              dcomplex r[3])
{

  if ( n==m ){ // diagonal 
    r[0] = Complex(0.0,0.0);
    r[1] = Complex(0.0,0.0);
    r[2] = Complex(0.0,0.0);
  }
  else if ( (n<No[spin]) && (m<No[spin]) ){ // A11
    r[0] = Cadd(A11[0][0][m*No[spin]+n], A11[1][0][m*No[spin]+n]);
    r[1] = Cadd(A11[0][1][m*No[spin]+n], A11[1][1][m*No[spin]+n]);
    r[2] = Cadd(A11[0][2][m*No[spin]+n], A11[1][2][m*No[spin]+n]);
  }

  else if ( (n<No[spin]) && !(m<No[spin]) ){ // A12
    r[0] = Cadd(A12[0][0][(m-No[spin])*No[spin]+n], A12[1][0][(m-No[spin])*No[spin]+n]);
    r[1] = Cadd(A12[0][1][(m-No[spin])*No[spin]+n], A12[1][1][(m-No[spin])*No[spin]+n]);
    r[2] = Cadd(A12[0][2][(m-No[spin])*No[spin]+n], A12[1][2][(m-No[spin])*No[spin]+n]);
  }

  else if ( !(n<No[spin]) && (m<No[spin]) ){ // A21
    r[0] = Cadd(A21[0][0][m*Nuo[spin]+n-No[spin]], A21[1][0][m*Nuo[spin]+n-No[spin]]);
    r[1] = Cadd(A21[0][1][m*Nuo[spin]+n-No[spin]], A21[1][1][m*Nuo[spin]+n-No[spin]]);
    r[2] = Cadd(A21[0][2][m*Nuo[spin]+n-No[spin]], A21[1][2][m*Nuo[spin]+n-No[spin]]);
  }

  else if ( !(n<No[spin]) && !(m<No[spin]) ){ // A22
    r[0] = Cadd(A22[0][0][(m-No[spin])*Nuo[spin]+n-No[spin]], A22[1][0][(m-No[spin])*Nuo[spin]+n-No[spin]]);
    r[1] = Cadd(A22[0][1][(m-No[spin])*Nuo[spin]+n-No[spin]], A22[1][1][(m-No[spin])*Nuo[spin]+n-No[spin]]);
    r[2] = Cadd(A22[0][2][(m-No[spin])*Nuo[spin]+n-No[spin]], A22[1][2][(m-No[spin])*Nuo[spin]+n-No[spin]]);
  }

  /*
  r[0] = Afull[0][m*MatDim+n];
  r[1] = Afull[1][m*MatDim+n];
  r[2] = Afull[2][m*MatDim+n];
  */
}



dcomplex rval_gg( int spin, int n, int m, double *eval, double **omega_gg, 
                  dcomplex ***A11, dcomplex ***A12, dcomplex ***A21, dcomplex ***A22,
                  dcomplex r_gg[3][3] )
{
  int a,b,l;
  double Dmna,Dmnb,wnm,wlm,wnl;
  dcomplex ctmp1,ctmp2,ctmp3,csum,rnm[3],rnl[3],rlm[3];

  /*
  for (b=0; b<3; b++){
    for (a=0; a<3; a++){

      Dmnb = omega_gg[b][m] - omega_gg[b][n];   
      Dmna = omega_gg[a][m] - omega_gg[a][n];   
      wnm = eval[n] - eval[m];
 
      r_gg[b][a].r = (Afull[a][m*MatDim+n].r*Dmnb + Afull[b][m*MatDim+n].r*Dmna)/wnm;
      r_gg[b][a].i = (Afull[a][m*MatDim+n].i*Dmnb + Afull[b][m*MatDim+n].i*Dmna)/wnm;

      csum = Complex(0.0,0.0); 
      for (l=0; l<MatDim; l++){

        wlm = eval[l] - eval[m];
        wnl = eval[n] - eval[l];

        ctmp1 = Cmul(Afull[a][l*MatDim+n],Afull[b][m*MatDim+l]); 
        ctmp2 = Cmul(Afull[b][l*MatDim+n],Afull[a][m*MatDim+l]); 

        csum.r += wlm*ctmp1.r - wnl*ctmp2.r;
        csum.i += wlm*ctmp1.i - wnl*ctmp2.i;
      }

      r_gg[b][a].r +=-csum.i/wnm;
      r_gg[b][a].i += csum.r/wnm;
    }
  }
  */

  rval(spin, n, m, A11, A12, A21, A22, rnm);

  for (b=0; b<3; b++){
    for (a=0; a<3; a++){

      Dmnb = omega_gg[b][m] - omega_gg[b][n];   
      Dmna = omega_gg[a][m] - omega_gg[a][n];   
      wnm = eval[n] - eval[m];
 
      r_gg[b][a].r = (rnm[a].r*Dmnb + rnm[b].r*Dmna)/wnm;
      r_gg[b][a].i = (rnm[a].i*Dmnb + rnm[b].i*Dmna)/wnm;
    }
  }

  for (l=0; l<MatDim; l++){

    rval(spin, n, l, A11, A12, A21, A22, rnl);
    rval(spin, l, m, A11, A12, A21, A22, rlm);

    wlm = eval[l] - eval[m];
    wnl = eval[n] - eval[l];

    for (b=0; b<3; b++){
      for (a=0; a<3; a++){

        ctmp1 = Cmul(rnl[a],rlm[b]); 
        ctmp2 = Cmul(rnl[b],rlm[a]); 
        ctmp3.r = wlm*ctmp1.r - wnl*ctmp2.r;
        ctmp3.i = wlm*ctmp1.i - wnl*ctmp2.i;
        r_gg[b][a].r +=-ctmp3.i/wnm;   
        r_gg[b][a].i += ctmp3.r/wnm;   
      }
    }
  }
}



void Check_Anti_Hermiticity(int spin, dcomplex ***A11, dcomplex ***A12, dcomplex ***A21, dcomplex ***A22)
{
  int p,i,j,k;
  double sum11,sum12,sum21,sum22;
  dcomplex ctmp1,ctmp2;

  /* A11 */ 

  sum11 = 0.0;

  for (p=0; p<3; p++){
    for (i=0; i<No[spin]; i++){
      for (j=i; j<No[spin]; j++){

	ctmp1.r = A11[0][p][j*No[spin]+i].r + A11[1][p][j*No[spin]+i].r;
	ctmp1.i = A11[0][p][j*No[spin]+i].i + A11[1][p][j*No[spin]+i].i;

	ctmp2.r = A11[0][p][i*No[spin]+j].r + A11[1][p][i*No[spin]+j].r;
	ctmp2.i = A11[0][p][i*No[spin]+j].i + A11[1][p][i*No[spin]+j].i;

	sum11 += fabs(ctmp1.r-ctmp2.r) + fabs(ctmp1.i+ctmp2.i);
      }
    }
  }

  printf("sum11=%15.12f\n",sum11);

  printf("A11.r\n");
  p = 0;
  for (i=0; i<No[spin]; i++){
    for (j=0; j<No[spin]; j++){

      ctmp1.r = A11[0][p][j*No[spin]+i].r + A11[1][p][j*No[spin]+i].r;
      ctmp1.i = A11[0][p][j*No[spin]+i].i + A11[1][p][j*No[spin]+i].i;

      //sum11 += fabs(ctmp1.r+ctmp2.r) + fabs(ctmp1.i-ctmp2.i);  

      printf("%6.3f ",ctmp1.r);
    }
    printf("\n");
  }

  printf("A11.i\n");
  p = 0;
  for (i=0; i<No[spin]; i++){
    for (j=0; j<No[spin]; j++){

      ctmp1.r = A11[0][p][j*No[spin]+i].r + A11[1][p][j*No[spin]+i].r;
      ctmp1.i = A11[0][p][j*No[spin]+i].i + A11[1][p][j*No[spin]+i].i;

      //sum11 += fabs(ctmp1.r+ctmp2.r) + fabs(ctmp1.i-ctmp2.i);  

      printf("%6.3f ",ctmp1.i);
    }
    printf("\n");
  }


  printf("A12.r\n");
  p = 0;
  for (i=0; i<No[spin]; i++){
    for (j=0; j<Nuo[spin]; j++){

      ctmp1.r = A12[0][p][j*No[spin]+i].r + A12[1][p][j*No[spin]+i].r;
      ctmp1.i = A12[0][p][j*No[spin]+i].i + A12[1][p][j*No[spin]+i].i;

      printf("%6.3f ",ctmp1.r);
    }
    printf("\n");
  }

  printf("A12.i\n");
  p = 0;
  for (i=0; i<No[spin]; i++){
    for (j=0; j<Nuo[spin]; j++){

      ctmp1.r = A12[0][p][j*No[spin]+i].r + A12[1][p][j*No[spin]+i].r;
      ctmp1.i = A12[0][p][j*No[spin]+i].i + A12[1][p][j*No[spin]+i].i;

      printf("%6.3f ",ctmp1.i);
    }
    printf("\n");
  }

  printf("A21.r\n");
  p = 0;
  for (i=0; i<Nuo[spin]; i++){
    for (j=0; j<No[spin]; j++){

      ctmp1.r = A21[0][p][j*Nuo[spin]+i].r + A21[1][p][j*Nuo[spin]+i].r;
      ctmp1.i = A21[0][p][j*Nuo[spin]+i].i + A21[1][p][j*Nuo[spin]+i].i;

      printf("%6.3f ",ctmp1.r);
    }
    printf("\n");
  }


  printf("A21.i\n");
  p = 0;
  for (i=0; i<Nuo[spin]; i++){
    for (j=0; j<No[spin]; j++){

      ctmp1.r = A21[0][p][j*Nuo[spin]+i].r + A21[1][p][j*Nuo[spin]+i].r;
      ctmp1.i = A21[0][p][j*Nuo[spin]+i].i + A21[1][p][j*Nuo[spin]+i].i;

      printf("%6.3f ",ctmp1.i);
    }
    printf("\n");
  }




}



void Linear_Opt_Conductivity_Berry_Connection()
{
  int i,j,n,m,k,l,ln,p,spin,kloop,Ng;
  double k1,k2,k3,*eval,tmp1,vtmp[4],Cell_Volume;
  double omega,omega_min,omega_max,domega,em,en,eta;
  dcomplex ***A12,***A21,*A12A21,**sigma;
  dcomplex *Hk,*Sk,*WF,ctmp1,ctmp2,ctmp3,csum;

  /* set parameters */

  Ng = 1000;
  omega_min = 0.0;               
  omega_max = 10.0/eV2Hartree;   
  domega = (omega_max - omega_min)/(double)(Ng-1);
  eta = 0.2/eV2Hartree; 

  /*************************************************************
                    allocation of arrays
  *************************************************************/

  Hk = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));
  Sk = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));
  WF = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));
  eval = (double*)malloc(sizeof(double)*MatDim);

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

  A12A21 = (dcomplex*)malloc(sizeof(dcomplex)*Nomax*Nomax);

  sigma = (dcomplex**)malloc(sizeof(dcomplex*)*9);
  for (i=0; i<9; i++){
    sigma[i] = (dcomplex*)malloc(sizeof(dcomplex)*Ng);
    for (j=0; j<Ng; j++){
      sigma[i][j] = Complex(0.0,0.0);
    }
  }

  /* calculte cell volume */

  Cross_Product(tv[2],tv[3],vtmp);
  Cell_Volume = fabs(Dot_Product(tv[1],vtmp));

  /*************************************************************
                  loops for spin and k-points
  *************************************************************/

  for (spin=0; spin<=SpinP_switch; spin++){ 

    /* loop for k-points */

    for (kloop=0; kloop<T_knum; kloop++){

      k1 = T_KGrids1[kloop]; 
      k2 = T_KGrids2[kloop]; 
      k3 = T_KGrids3[kloop];

      /* make H(k) and S(k), and solve the generalized eigenvalue problem */ 

      Calc_Hk_Sk(Hk,Sk,spin,k1,k2,k3);
      lapack_zhegvx(MatDim,Hk,Sk,WF,eval);

      /* calculate A12 and A21 */

      Calc_A12_A21(spin,k1,k2,k3,WF,eval,A12,A21);

      for (i=0; i<3; i++){
        for (j=0; j<No[spin]*Nuo[spin]; j++){
          A12[0][i][j].r += A12[1][i][j].r;    
          A12[0][i][j].i += A12[1][i][j].i;    
          A21[0][i][j].r += A21[1][i][j].r;    
          A21[0][i][j].i += A21[1][i][j].i;
	}    
      }

      /* calculate linear optical conductivity */

      for (p=0; p<Ng; p++){

        omega = omega_min + (double)p*domega;

        for (i=0; i<3; i++){
	  for (j=0; j<3; j++){

            csum = Complex(0.0,0.0);

	    for (m=0; m<No[spin]; m++){

              em = eval[m];

	      for (n=0; n<Nuo[spin]; n++){

                ln = No[spin] + n;
    	        en = eval[ln];

		ctmp1.r = A12[0][i][n*No[spin]+m].r*A21[0][j][m*Nuo[spin]+n].r
 		         -A12[0][i][n*No[spin]+m].i*A21[0][j][m*Nuo[spin]+n].i;

		ctmp1.i = A12[0][i][n*No[spin]+m].r*A21[0][j][m*Nuo[spin]+n].i
 		         +A12[0][i][n*No[spin]+m].i*A21[0][j][m*Nuo[spin]+n].r;

		ctmp2.r = omega + em - en;
		ctmp2.i = eta;
		ctmp3 = Cdiv(ctmp1, ctmp2); 

		csum.r += (em-en)*ctmp3.r;
		csum.i += (em-en)*ctmp3.i;

	      } // n
	    } // m

            /* add csum to sigma */ 

            sigma[j*3+i][p].r += csum.r*(double)T_k_op[kloop];
            sigma[j*3+i][p].i += csum.i*(double)T_k_op[kloop];

	  } // j
	} // i
      } // p

    } /* end of kloop */ 

    /* taking account of the prefactor */ 

    tmp1 = conductivity_unit/(tweight*Cell_Volume);
    for (i=0; i<3; i++){
      for (j=0; j<3; j++){
	for (p=0; p<Ng; p++){
          ctmp1 = sigma[j*3+i][p];
          sigma[j*3+i][p].r = ctmp1.i*tmp1;
          sigma[j*3+i][p].i =-ctmp1.r*tmp1;
	} // p
      } //j
    } // i

  } /* end of spin */

  /* we do not take account of spin multiplicity yet. */

  for (p=0; p<Ng; p++){
    omega = omega_min + (double)p*domega;
    printf("%15.12f %15.12f %15.12f\n",omega,sigma[0][p].r,sigma[0][p].i); 
  }

  /*************************************************************
                    freeing of arrays
  *************************************************************/

  free(Hk);
  free(Sk);
  free(WF);
  free(eval);

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

  free(A12A21);

  for (i=0; i<9; i++){
    free(sigma[i]);
  }
  free(sigma);
}



void Linear_Opt_Conductivity()
{
  int i,j,n,m,k,l,p,spin,kloop,Ng;
  double k1,k2,k3,*eval,eta,max_x=30.0;
  double omega,omega_min,omega_max,domega,em,en,fm,fn;
  double Beta,x,tmp1,vtmp[4],Cell_Volume;
  dcomplex *Hk,*Sk,*WF,**P,**sigma;
  dcomplex csum,ctmp1,ctmp2,ctmp3; 

  /* set parameters */

  Ng = 1000;
  omega_min = 0.0;               
  omega_max = 10.0/eV2Hartree;   
  domega = (omega_max - omega_min)/(double)(Ng-1);
  eta = 0.2/eV2Hartree; 
  Beta = 10000.0;

  /*************************************************************
                    allocation of arrays
  *************************************************************/

  Hk = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));
  Sk = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));
  WF = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));
  eval = (double*)malloc(sizeof(double)*MatDim);

  P = (dcomplex**)malloc(sizeof(dcomplex*)*3);
  for (i=0; i<3; i++){
    P[i] = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));
  }

  sigma = (dcomplex**)malloc(sizeof(dcomplex*)*9);
  for (i=0; i<9; i++){
    sigma[i] = (dcomplex*)malloc(sizeof(dcomplex)*Ng);
    for (j=0; j<Ng; j++){
      sigma[i][j] = Complex(0.0,0.0);
    }
  }

  /* calculte cell volume */

  Cross_Product(tv[2],tv[3],vtmp);
  Cell_Volume = fabs(Dot_Product(tv[1],vtmp));

  /*************************************************************
                  loops for spin and k-points
  *************************************************************/

  for (spin=0; spin<=SpinP_switch; spin++){ 

    /* loop for k-points */

    for (kloop=0; kloop<T_knum; kloop++){

      k1 = T_KGrids1[kloop]; 
      k2 = T_KGrids2[kloop]; 
      k3 = T_KGrids3[kloop];

      /* make H(k) and S(k), and solve the generalized eigenvalue problem */ 

      Calc_Hk_Sk(Hk,Sk,spin,k1,k2,k3);
      lapack_zhegvx(MatDim,Hk,Sk,WF,eval);

      /* calculate the momentum matrix:
         Conventional direct method: Calc_Momentum_Matrix
         C.C. Lee's method: Calc_Momentum_Matrix_CCLee
      */
      //Calc_Momentum_Matrix_CCLee(spin, k1, k2, k3, WF, eval, P);

      Calc_Momentum_Matrix(spin, k1, k2, k3, WF, eval, P);

      /* calculate linear optical conductivity */
  
      for (p=0; p<Ng; p++){

        omega = omega_min + (double)p*domega;

        for (i=0; i<3; i++){
	  for (j=0; j<3; j++){

            csum = Complex(0.0,0.0);
 
	    for (m=0; m<MatDim; m++){

              em = eval[m];
   	      x = (em - ChemP)*Beta;
	      if (x<=-max_x) x = -max_x;
	      if (max_x<=x)  x = max_x;
	      fm = 1.0/(1.0 + exp(x));

	      for (n=0; n<MatDim; n++){

                if (m!=n){

		  en = eval[n];
		  x = (en - ChemP)*Beta;
		  if (x<=-max_x) x = -max_x;
		  if (max_x<=x)  x = max_x;
		  fn = 1.0/(1.0 + exp(x));

		  ctmp1.r = P[i][n*MatDim+m].r*P[j][m*MatDim+n].r - P[i][n*MatDim+m].i*P[j][m*MatDim+n].i;
		  ctmp1.i = P[i][n*MatDim+m].r*P[j][m*MatDim+n].i + P[i][n*MatDim+m].i*P[j][m*MatDim+n].r;

		  ctmp2.r = omega + em - en;
		  ctmp2.i = eta;
		  ctmp3 = Cdiv(ctmp1, ctmp2); 

		  csum.r += (fm-fn)/(em-en+1.0e-13)*ctmp3.r;
		  csum.i += (fm-fn)/(em-en+1.0e-13)*ctmp3.i;

		} // end of if (m!=n) 
	      } // n
	    } // m

            /* add sum to sigma */ 

            sigma[j*3+i][p].r += csum.r*(double)T_k_op[kloop];
            sigma[j*3+i][p].i += csum.i*(double)T_k_op[kloop];

	  } // j
	} // i
      } // p          

    } /* end of kloop */ 

    /* taking account of the prefactor */ 

    tmp1 = conductivity_unit/(tweight*Cell_Volume);
    for (i=0; i<3; i++){
      for (j=0; j<3; j++){
	for (p=0; p<Ng; p++){
          ctmp1 = sigma[j*3+i][p];
          sigma[j*3+i][p].r = ctmp1.i*tmp1;
          sigma[j*3+i][p].i =-ctmp1.r*tmp1;
	} // p
      } //j
    } // i

  } /* end of spin */

  /* we do not take account of spin multiplicity yet. */

  for (p=0; p<Ng; p++){
    omega = omega_min + (double)p*domega;
    printf("%15.12f %15.12f %15.12f\n",omega,sigma[0][p].r,sigma[0][p].i); 
  }

  /*************************************************************
                    freeing of arrays
  *************************************************************/

  free(Hk);
  free(Sk);
  free(WF);
  free(eval);

  for (i=0; i<3; i++){
    free(P[i]);
  }
  free(P);

  for (i=0; i<9; i++){
    free(sigma[i]);
  }
  free(sigma);
}



void pth_Harmonic_Generation(int pth)
{
  int i,j,n,m,k,l,n1,n2,Ii,p,q,xyz,spin,kloop,N,Ng,omg,Nxyz[3];
  int k_idx1,k_idx2,k_idx3;
  double k1,k2,k3,***EIGEN,eta,max_x=30.0,Bi,alphai,scissor;
  double omega0,pomega0,omega_min,omega_max,domega,em,en,fm,fn;
  double Beta,x,tmp1,vtmp[4],Cell_Volume,da,db,dc;
  dcomplex *Hk,*Sk,*WF,*****chi,*Afull;
  dcomplex *rho0,*rho_a_p,*rho_a_m,*rho_b_p,*rho_b_m,*rho_c_p,*rho_c_m;
  dcomplex *work_array,g,d1,d2,d3;
  dcomplex csum[3],ctmp1,ctmp2,ctmp3,ctmp4,Omega; 
  dcomplex alpha = {1.0,0.0}; 
  dcomplex beta = {0.0,0.0};
  dcomplex kappa,kappa_p,mkappa_p;
  char operate[YOUSO10],filepath[YOUSO10],filename[YOUSO10];
  char fname[YOUSO10];
  char buf[fp_bsize];          /* setvbuf */
  int numprocs,myid;
  FILE *fp;
  double time_r,time_w,Stime,Etime;

  MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD,&myid);

  time_r = 0.0;
  time_w = 0.0;

  /* set parameters */

  Ng = 200;
  omega_min = 0.001/eV2Hartree; 
  omega_max =  5.0/eV2Hartree;   
  domega = (omega_max - omega_min)/(double)(Ng-1);
  //eta = 0.07/eV2Hartree; 

  eta = 0.07/eV2Hartree; 
  Beta = 10000.0;

  //scissor = 0.0/eV2Hartree; 

  scissor = 1.151/eV2Hartree; 

  /*************************************************************
         set up the directory in which files are saved. 
  *************************************************************/

  sprintf(filename,"./test1234");

  if (myid==Host_ID){

    sprintf(operate,"%s",filename);
    mkdir(operate,0775); 

    int removed = 0;

    if (remove_files_in_dir(filename, &removed) != 0) {
      perror("remove_files_in_dir");
    }

    //printf("Removed %d entries\n", removed);    
  }

  /*************************************************************
         setting information for the 0-th to p-th chi
  *************************************************************/

  int ***independent_list = malloc(sizeof(int**)*(pth+1));
  int *independent_count = malloc(sizeof(int)*(pth+1));

  for (p=0; p<=pth; p++){

    /* Build independent list */
    init_independent_components(p,
			        &independent_list[p],
			        &independent_count[p]);


    printf("Independent components for p=%d (total %d):\n", p, independent_count[p]);
    for (i=0; i<independent_count[p]; ++i) {
      int j;
      printf("[%2d] (", i);
      for (j=0; j<p; ++j) {
	printf("%d",independent_list[p][i][j]);
	if (j<(p-1)) printf(",");
      }
      printf(")\n");
    }
  }

  /*
  p = 1; 
  q = 0;
  xyz = 0;
  printf("%2d\n",find_index(p,independent_list[p][q],xyz,independent_list,independent_count));

  MPI_Finalize();
  exit(0);
  */

  /*************************************************************
                    allocation of arrays
  *************************************************************/

  Afull = (dcomplex*)malloc(sizeof(dcomplex)*3*MatDim*MatDim);
  work_array = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));

  rho0    = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));
  rho_a_p = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));
  rho_a_m = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));
  rho_b_p = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));
  rho_b_m = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));
  rho_c_p = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));
  rho_c_m = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));

  chi = (dcomplex*****)malloc(sizeof(dcomplex****)*(SpinP_switch+1));
  for (spin=0; spin<=SpinP_switch; spin++){ 
    chi[spin] = (dcomplex****)malloc(sizeof(dcomplex***)*Ng);
    for (omg=0; omg<Ng; omg++){
      chi[spin][omg] = (dcomplex***)malloc(sizeof(dcomplex**)*3);
      for (xyz=0; xyz<3; xyz++){
	chi[spin][omg][xyz] = (dcomplex**)malloc(sizeof(dcomplex*)*(pth+1));
	for (p=0; p<(pth+1); p++){
	  chi[spin][omg][xyz][p] = (dcomplex*)malloc(sizeof(dcomplex)*independent_count[p]);
	  for (i=0; i<independent_count[p]; i++){
	    chi[spin][omg][xyz][p][i] = Complex(0.0,0.0);
	  } 
	}
      }
    }
  }

  Hk = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));
  Sk = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));
  WF = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));

  EIGEN = (double***)malloc(sizeof(double**)*(SpinP_switch+1));
  for (spin=0; spin<=SpinP_switch; spin++){ 
    EIGEN[spin] = (double**)malloc(sizeof(double*)*T_knum);
    for (i=0; i<T_knum; i++){
      EIGEN[spin][i] = (double*)malloc(sizeof(double)*MatDim);
    }
  }

  /* calculte cell volume */

  Cross_Product(tv[2],tv[3],vtmp);
  Cell_Volume = fabs(Dot_Product(tv[1],vtmp));

  /*************************************************************
            calculation of guiding functions and FR
  *************************************************************/

  /* calculate guiding functions, PF1 and PF2 */

  Calc_Guiding_Functions_1_2(PF1,PF2);

  /* calculate FR1 and FR2, which are stored in PF1 and PF2 */

  Calc_FR_1_2(PF1,PF2);  

  /*************************************************************
   Calculations of eigenstates and the Berry connection matrix    
   For all the spin components and k-points, they are calculated
   and saved in files.  
  *************************************************************/

  for (spin=0; spin<=SpinP_switch; spin++){ 

    /* loop for k-points */

    for (kloop=0; kloop<T_knum; kloop++){

      k1 = T_KGrids1[kloop]; 
      k2 = T_KGrids2[kloop]; 
      k3 = T_KGrids3[kloop];
      k_idx1 = TK2i[kloop]; 
      k_idx2 = TK2j[kloop]; 
      k_idx3 = TK2k[kloop]; 

      /* make H(k) and S(k), and solve the generalized eigenvalue problem */ 

      Calc_Hk_Sk(Hk,Sk,spin,k1,k2,k3);
      lapack_zhegvx(MatDim,Hk,Sk,WF,EIGEN[spin][kloop]);

      for (i=No[spin]; i<MatDim; i++){
        EIGEN[spin][kloop][i] += scissor; 
      }

      /* calculate Afull */

      Calc_Afull( spin,k1,k2,k3,WF,EIGEN[spin][kloop],PF1,PF2,Afull );



      printf("%2d %15.12f\n",kloop,Afull[0].r);



      /* measure timing */ 

      if (measure_time) dtime(&Stime);

      /* save Afull */

      sprintf(fname,"%s/A_%i_%i_%i_%i",filename,spin,k_idx1,k_idx2,k_idx3);
      
      if ((fp = fopen(fname,"wb")) != NULL){

        setvbuf(fp,buf,_IOFBF,fp_bsize);  /* setvbuf */

	fwrite(Afull,sizeof(dcomplex),3*MatDim*MatDim,fp);

        /* fclose(fp) */
	fclose(fp);
      }
      else {
	printf("Failed in saving the Berry connection matrix\n");
	MPI_Finalize();
	exit(0);
      }

      /* save rho00, Afull is used temporary. */

      for (i=0; i<MatDim*MatDim; i++){
        Afull[i] = Complex(0.0,0.0);
      }
      for (i=0; i<No[spin]; i++){
        Afull[i*MatDim+i] = Complex(1.0,0.0);
      }

      time_w += fwrite_rho(filename,0,0,spin,k_idx1,k_idx2,k_idx3,Afull);

      //printf("Calc. of A  kloop=%2d\n",kloop);

    } /* end of kloop */ 
  } /* end of spin */



  MPI_Finalize();
  exit(0);



  /*************************************************************
         calculations of density matrices, 
                         susceptibilities, 
                     and conductivities.
  *************************************************************/

  /* loop for spin */

  for (spin=0; spin<=SpinP_switch; spin++){ 

    /* loop for omg */

    for (omg=0; omg<Ng; omg++){

      /* set omega0 */

      omega0 = omega_min + (double)omg*domega;

      /* loop for p */

      for (p=1; p<=pth; p++){

	/* set Omega */

	Omega.r = (double)p*omega0;
	Omega.i = eta;
   
	/* loop for irreducible rho */

	for (q=0; q<independent_count[p]; q++){

          printf("Calc. of rho  omg=%2d p=%2d q=%2d\n",omg,p,q);

	  /* set Nxyz */

	  Nxyz[0] = 0; Nxyz[1] = 0; Nxyz[2] = 0; 
	  for (i=0; i<p; i++){
	    xyz = independent_list[p][q][i];
	    Nxyz[xyz]++;
	  }

	  /* loop for k-points */

	  for (kloop=0; kloop<T_knum; kloop++){

	    k1 = T_KGrids1[kloop]; 
	    k2 = T_KGrids2[kloop]; 
	    k3 = T_KGrids3[kloop];
	    k_idx1 = TK2i[kloop]; 
	    k_idx2 = TK2j[kloop]; 
	    k_idx3 = TK2k[kloop]; 

            /* fread A */

            time_r += fread_A(filename,spin,k_idx1,k_idx2,k_idx3,Afull);

            /* initialize work_array */

            for (i=0; i<MatDim*MatDim; i++){
	      work_array[i] = Complex(0.0,0.0);
	    }






            if ( p==2 && k1<0.001 && k2<0.001 && k3<0.001 ){

              l = 1;
	      xyz = l;

	      m = 0;
	      n = 0;

              for (i=0; i<Nk1*2; i++){
		time_r += fread_rho(filename,p-1,l,spin,(k_idx1+i)%Nk1,k_idx2,k_idx3,rho0);
                printf("%2d %15.12f\n",i,rho0[0].r);
	      }

	      MPI_Finalize();
	      exit(0);


              printf("ABC_p %2d %2d %2d\n",(k_idx1+1)%Nk1,k_idx2,k_idx3); 
              printf("ABC_m %2d %2d %2d\n",((k_idx1-1)%Nk1+Nk1)%Nk1,k_idx2,k_idx3);

              time_r += fread_rho(filename,p-1,l,spin,k_idx1,k_idx2,k_idx3,rho0);
              time_r += fread_rho(filename,p-1,l,spin,(k_idx1+1)%Nk1,k_idx2,k_idx3,rho_a_p);
	      time_r += fread_rho(filename,p-1,l,spin,((k_idx1-1)%Nk1+Nk1)%Nk1,k_idx2,k_idx3,rho_a_m);
	      time_r += fread_rho(filename,p-1,l,spin,k_idx1,(k_idx2+1)%Nk2,k_idx3,rho_b_p);
	      time_r += fread_rho(filename,p-1,l,spin,k_idx1,((k_idx2-1)%Nk2+Nk2)%Nk2,k_idx3,rho_b_m);
	      time_r += fread_rho(filename,p-1,l,spin,k_idx1,k_idx2,(k_idx3+1)%Nk3,rho_c_p);
	      time_r += fread_rho(filename,p-1,l,spin,k_idx1,k_idx2,((k_idx3-1)%Nk3+Nk3)%Nk3,rho_c_m);


              for (i=0; i<20; i++){
                printf("i=%2d %15.12f %15.12f %15.12f\n",i,rho_c_m[i].r,rho0[i].r,rho_c_p[i].r);
	      } 


	      d1.r = (rho_a_p[n*MatDim+m].r - rho_a_m[n*MatDim+m].r)*da;
	      d1.i = (rho_a_p[n*MatDim+m].i - rho_a_m[n*MatDim+m].i)*da;

	      d2.r = (rho_b_p[n*MatDim+m].r - rho_b_m[n*MatDim+m].r)*db;
	      d2.i = (rho_b_p[n*MatDim+m].i - rho_b_m[n*MatDim+m].i)*db;

	      d3.r = (rho_c_p[n*MatDim+m].r - rho_c_m[n*MatDim+m].r)*dc;
	      d3.i = (rho_c_p[n*MatDim+m].i - rho_c_m[n*MatDim+m].i)*dc;

	      g.r = d1.r*tv[1][xyz] + d2.r*tv[2][xyz] + d3.r*tv[3][xyz];
	      g.i = d1.i*tv[1][xyz] + d2.i*tv[2][xyz] + d3.i*tv[3][xyz];

              printf("xyz=%2d %15.12f %15.12f\n",xyz,g.r,g.i); 

	      MPI_Finalize();
	      exit(0);
	    }









            /* sum over xyz */

            for (xyz=0; xyz<3; xyz++){

	      l = find_index(p,independent_list[p][q],xyz,independent_list,independent_count);

              if (l!=-1){

                /************************************************
                 fread rho, rhos of the neighboring 7 points, 
                 including rho at the current k-point, are read.
                ************************************************/

                time_r += fread_rho(filename,p-1,l,spin,k_idx1,k_idx2,k_idx3,rho0);
                time_r += fread_rho(filename,p-1,l,spin,(k_idx1+1)%Nk1,k_idx2,k_idx3,rho_a_p);
                time_r += fread_rho(filename,p-1,l,spin,((k_idx1-1)%Nk1+Nk1)%Nk1,k_idx2,k_idx3,rho_a_m);
                time_r += fread_rho(filename,p-1,l,spin,k_idx1,(k_idx2+1)%Nk2,k_idx3,rho_b_p);
                time_r += fread_rho(filename,p-1,l,spin,k_idx1,((k_idx2-1)%Nk2+Nk2)%Nk2,k_idx3,rho_b_m);
                time_r += fread_rho(filename,p-1,l,spin,k_idx1,k_idx2,(k_idx3+1)%Nk3,rho_c_p);
                time_r += fread_rho(filename,p-1,l,spin,k_idx1,k_idx2,((k_idx3-1)%Nk3+Nk3)%Nk3,rho_c_m);

                /************************************************
                      calculation of commutation: [A, rho]
                ************************************************/

                /* zgemm of Ai and rho0 */

                alphai = (double)Nxyz[xyz]/(double)p;

                alpha = Complex(alphai, 0.0);
                beta = Complex(1.0,0.0); 
	        m = MatDim; n = MatDim; k = MatDim;

		zgemm_("N","N", &m, &n, &k, &alpha, &Afull[Aref(xyz,0,0)], &m, rho0, 
                        &k, &beta, work_array, &m);

                /* zgemm of rho0 and Ai */

                alpha = Complex(-alphai, 0.0);
                beta = Complex(1.0,0.0); 
		m = MatDim; n = MatDim; k = MatDim;

		zgemm_("N","N", &m, &n, &k, &alpha, rho0, &m, &Afull[Aref(xyz,0,0)], 
                       &k, &beta, work_array, &m);

                /************************************************
                   derivative of rho, which is evaluated by 
                   a centered finite difference.
                ************************************************/

                da = (double)Nk1/(4.0*PI);
                db = (double)Nk2/(4.0*PI);
                dc = (double)Nk3/(4.0*PI);

  	        for (m=0; m<MatDim; m++){
                  for (n=0; n<MatDim; n++){
              
                    d1.r = (rho_a_p[n*MatDim+m].r - rho_a_m[n*MatDim+m].r)*da;
                    d1.i = (rho_a_p[n*MatDim+m].i - rho_a_m[n*MatDim+m].i)*da;

                    d2.r = (rho_b_p[n*MatDim+m].r - rho_b_m[n*MatDim+m].r)*db;
                    d2.i = (rho_b_p[n*MatDim+m].i - rho_b_m[n*MatDim+m].i)*db;

                    d3.r = (rho_c_p[n*MatDim+m].r - rho_c_m[n*MatDim+m].r)*dc;
                    d3.i = (rho_c_p[n*MatDim+m].i - rho_c_m[n*MatDim+m].i)*dc;

                    g.r = d1.r*tv[1][xyz] + d2.r*tv[2][xyz] + d3.r*tv[3][xyz];
                    g.i = d1.i*tv[1][xyz] + d2.i*tv[2][xyz] + d3.i*tv[3][xyz];

                    /* add the derivative to work_array */
                    /* i is multiplied */  

                    //work_array[n*MatDim+m].r -= g.i;
                    //work_array[n*MatDim+m].i += g.r;

		  } // n
		} // m                  

	      } // end of if (l!=-1)
	    } // xyz

            /* multiply work_array with 1/(hbar*Omega-(emu-enu)) */

	    for (m=0; m<MatDim; m++){

              em = EIGEN[spin][kloop][m];

	      for (n=0; n<MatDim; n++){

    	        en = EIGEN[spin][kloop][n];

                ctmp1.r = Omega.r - (em - en);
                ctmp1.i = Omega.i;

                ctmp2 = Complex(1.0,0.0);
                ctmp3 = Cdiv(ctmp2,ctmp1);

                ctmp1 = work_array[n*MatDim+m];
                ctmp2 = Cmul(ctmp1,ctmp3);
                work_array[n*MatDim+m] = ctmp2;





                work_array[n*MatDim+m] = Afull[n*MatDim+m];





  	      } // n
	    } // m

            /* fwrite rho */ 

            time_w += fwrite_rho(filename,p,q,spin,k_idx1,k_idx2,k_idx3,work_array);

            /********************************************************
                    calculate the contribution at k to chi 
            ********************************************************/

            /* trace(A_i*rho_I^(p)) */ 

	    csum[0] = Complex(0.0,0.0); 
	    csum[1] = Complex(0.0,0.0); 
	    csum[2] = Complex(0.0,0.0); 

            for (i=0; i<3; i++){
	      for (m=0; m<MatDim; m++){
		for (n=0; n<MatDim; n++){

                  ctmp1 = Afull[Aref(i,m,n)];
                  ctmp2 = work_array[m*MatDim+n];

                  csum[i].r += (ctmp1.r*ctmp2.r - ctmp1.i*ctmp2.i); 
                  csum[i].i += (ctmp1.r*ctmp2.i + ctmp1.i*ctmp2.r);

		}
	      }
	    }              
                        
            /* add the trace to chi */

            for (i=0; i<3; i++){
              csum[i].r *= (double)T_k_op[kloop];
              csum[i].i *= (double)T_k_op[kloop];
	    }

            chi[spin][omg][0][p][q] = Cadd(chi[spin][omg][0][p][q], csum[0]);
            chi[spin][omg][1][p][q] = Cadd(chi[spin][omg][1][p][q], csum[1]);
            chi[spin][omg][2][p][q] = Cadd(chi[spin][omg][2][p][q], csum[2]);
           
 	  } // kloop

          /* taking account of the prefactor */ 

          tmp1 =-1.0/(tweight*Cell_Volume);

          chi[spin][omg][0][p][q].r *= tmp1;
          chi[spin][omg][0][p][q].i *= tmp1;
          chi[spin][omg][1][p][q].r *= tmp1;
          chi[spin][omg][1][p][q].i *= tmp1;
          chi[spin][omg][2][p][q].r *= tmp1;
          chi[spin][omg][2][p][q].i *= tmp1;

	} // q
      } // p
    } // omg
  } // spin



  printf("output of chi\n");
  for (omg=0; omg<Ng; omg++){
    omega0 = omega_min + (double)omg*domega;

    //printf("%15.12f %15.12f %15.12f\n",omega0*eV2Hartree,chi[0][omg][0][1][0].r,chi[0][omg][0][1][0].i);

    //printf("%15.12f %15.12f %15.12f\n",omega0*eV2Hartree,chi[0][omg][0][2][4].r,chi[0][omg][0][2][4].i);
    //printf("%15.12f %15.12f %15.12f\n",omega0*eV2Hartree,chi[0][omg][2][2][1].r,chi[0][omg][2][2][1].i);
    printf("%15.12f %15.12f %15.12f\n",omega0*eV2Hartree,chi[0][omg][1][2][2].r,chi[0][omg][1][2][2].i);

  }




  printf("\ntime_r=%15.12f time_w=%15.12f\n",time_r,time_w);
  MPI_Finalize();
  exit(0);

  /*************************************************************
                    freeing of arrays
  *************************************************************/

  free(Afull);
  free(work_array);
  free(rho0);
  free(rho_a_p);
  free(rho_a_m);
  free(rho_b_p);
  free(rho_b_m);
  free(rho_c_p);
  free(rho_c_m);

  for (spin=0; spin<=SpinP_switch; spin++){ 
    for (omg=0; omg<Ng; omg++){
      for (xyz=0; xyz<3; xyz++){
	for (p=0; p<(pth+1); p++){
	  free(chi[spin][omg][xyz][p]);
	}
        free(chi[spin][omg][xyz]);
      }
      free(chi[spin][omg]);
    }
    free(chi[spin]);
  }
  free(chi);

  free(Hk);
  free(Sk);
  free(WF);

  for (spin=0; spin<=SpinP_switch; spin++){ 
    for (i=0; i<T_knum; i++){
      free(EIGEN[spin][i]);
    }
    free(EIGEN[spin]);
  }
  free(EIGEN);

  for (p=0; p<pth; p++){
    for (i = 0; i < independent_count[p]; ++i) {
      free(independent_list[p][i]);
    }
    free(independent_list[p]);
  }
  free(independent_list);
  
  free(independent_count);
}




double fread_A(char filename[YOUSO10], int spin, int ki, int kj, int kk, dcomplex *Afull)
{
  double Stime,Etime;
  char fname[YOUSO10];
  FILE *fp;

  dtime(&Stime);

  sprintf(fname,"%s/A_%i_%i_%i_%i",filename,spin,ki,kj,kk);

  if ((fp = fopen(fname,"rb")) != NULL){

    fread(Afull,sizeof(dcomplex),3*MatDim*MatDim,fp);

    /* fclose(fp) */
    fclose(fp);
  }
  else {
    printf("Failed in reading Berry connection matrix\n");
    MPI_Finalize();
    exit(0);
  }

  dtime(&Etime);
  return (Etime-Stime);
}




double fread_rho(char filename[YOUSO10], int p, int I, int spin, int ki, int kj, int kk, dcomplex *rho)
{
  double Stime,Etime;
  char fname[YOUSO10];
  FILE *fp;

  dtime(&Stime);

  sprintf(fname,"%s/R_%i_%i_%i_%i_%i_%i",filename,p,I,spin,ki,kj,kk);

  if ((fp = fopen(fname,"rb")) != NULL){

    fread(rho,sizeof(dcomplex),MatDim*MatDim,fp);

    /* fclose(fp) */
    fclose(fp);
  }
  else {
    printf("Failed in reading density matrix\n");
    MPI_Finalize();
    exit(0);
  }

  dtime(&Etime);
  return (Etime-Stime);
}



double fwrite_rho(char filename[YOUSO10], int p, int I, int spin, int ki, int kj, int kk, dcomplex *rho)
{
  double Stime,Etime;
  char fname[YOUSO10];
  FILE *fp;

  dtime(&Stime);

  sprintf(fname,"%s/R_%i_%i_%i_%i_%i_%i",filename,p,I,spin,ki,kj,kk);

  if ((fp = fopen(fname,"wb")) != NULL){

    fwrite(rho,sizeof(dcomplex),MatDim*MatDim,fp);

    /* fclose(fp) */
    fclose(fp);
  }
  else {
    printf("Failed in writing density matrix\n");
    MPI_Finalize();
    exit(0);
  }

  dtime(&Etime);
  return (Etime-Stime);
}





void pth_Harmonic_Generation_chi(int pth)
{
  int i,j,n,m,k,l,n1,n2,Ii,p,q,xyz,pmax_d,spin,kloop,N,Ng,omg,Nxyz[3];
  double k1,k2,k3,*eval,eta,max_x=30.0,Bi,alphai,scissor;
  double omega0,pomega0,omega_min,omega_max,domega,em,en,fm,fn;
  double Beta,x,tmp1,vtmp[4],Cell_Volume;
  dcomplex *Hk,*Sk,*WF,**P,***rho,****D,****chi;
  dcomplex **Afull,***A11,***A12,***A21,***A22;
  dcomplex *work_array;
  dcomplex csum[3],ctmp1,ctmp2,ctmp3,ctmp4,Omega; 
  dcomplex alpha = {1.0,0.0}; 
  dcomplex beta = {0.0,0.0};
  dcomplex kappa,kappa_p,mkappa_p;

  /* set parameters */

  Ng = 200;
  omega_min = 0.001/eV2Hartree; 
  omega_max =  5.0/eV2Hartree;   
  domega = (omega_max - omega_min)/(double)(Ng-1);
  //eta = 0.05/eV2Hartree; 

  eta = 0.05/eV2Hartree; 

  Beta = 10000.0;
  scissor = 1.151/eV2Hartree; 
  //scissor = 0.0/eV2Hartree; 

  pmax_d = pth;

  /*************************************************************
         setting information for the 0-th to p-th chi
  *************************************************************/

  int ***independent_list = malloc(sizeof(int**)*(pth+1));
  int *independent_count = malloc(sizeof(int)*(pth+1));

  for (p=0; p<=pth; p++){

    /* Build independent list */
    init_independent_components(p,
			        &independent_list[p],
			        &independent_count[p]);


    printf("Independent components for p=%d (total %d):\n", p, independent_count[p]);
    for (i=0; i<independent_count[p]; ++i) {
      int j;
      printf("[%2d] (", i);
      for (j=0; j<p; ++j) {
	printf("%d",independent_list[p][i][j]);
	if (j<(p-1)) printf(",");
      }
      printf(")\n");
    }
  }

  /*
  p = 1; 
  q = 0;
  xyz = 0;
  printf("%2d\n",find_index(p,independent_list[p][q],xyz,independent_list,independent_count));

  MPI_Finalize();
  exit(0);
  */

  /*************************************************************
                    allocation of arrays
  *************************************************************/

  Afull = (dcomplex**)malloc(sizeof(dcomplex*)*3);
  for (i=0; i<3; i++){
    Afull[i] = (dcomplex*)malloc(sizeof(dcomplex)*MatDim*MatDim);
  }

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

  work_array = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));

  rho = (dcomplex***)malloc(sizeof(dcomplex**)*(pth+1));
  for (p=0; p<(pth+1); p++){
    rho[p] = (dcomplex**)malloc(sizeof(dcomplex*)*independent_count[p]);
    for (q=0; q<independent_count[p]; q++){
      rho[p][q] = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));
    }
  }

  /* D[p][I][n][MatDim*MatDim]*/
  D = (dcomplex****)malloc(sizeof(dcomplex***)*(pth+1));
  for (p=0; p<(pth+1); p++){
    D[p] = (dcomplex***)malloc(sizeof(dcomplex**)*independent_count[p]);
    for (q=0; q<independent_count[p]; q++){
      D[p][q] = (dcomplex**)malloc(sizeof(dcomplex*)*(pmax_d+1));
      for (n=0; n<(pmax_d+1); n++){
        D[p][q][n] = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));
      }
    }
  }

  chi = (dcomplex****)malloc(sizeof(dcomplex***)*Ng);
  for (omg=0; omg<Ng; omg++){
    chi[omg] = (dcomplex***)malloc(sizeof(dcomplex**)*3);
    for (xyz=0; xyz<3; xyz++){
      chi[omg][xyz] = (dcomplex**)malloc(sizeof(dcomplex*)*(pth+1));
      for (p=0; p<(pth+1); p++){
	chi[omg][xyz][p] = (dcomplex*)malloc(sizeof(dcomplex)*independent_count[p]);
	for (i=0; i<independent_count[p]; i++){
	  chi[omg][xyz][p][i] = Complex(0.0,0.0);
	} 
      }
    }
  }

  Hk = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));
  Sk = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));
  WF = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));
  eval = (double*)malloc(sizeof(double)*MatDim);

  P = (dcomplex**)malloc(sizeof(dcomplex*)*3);
  for (i=0; i<3; i++){
    P[i] = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));
  }

  /* calculte cell volume */

  Cross_Product(tv[2],tv[3],vtmp);
  Cell_Volume = fabs(Dot_Product(tv[1],vtmp));

  /*************************************************************
          calculation of guiding functions and FR
  *************************************************************/

  /* calculate guiding functions, PF1 and PF2 */

  Calc_Guiding_Functions_1_2(PF1,PF2);

  /* calculate FR1 and FR2, which are stored in PF1 and PF2 */

  Calc_FR_1_2(PF1,PF2);  

  /*************************************************************
                  loops for spin and k-points
  *************************************************************/

  for (spin=0; spin<=SpinP_switch; spin++){ 

    /* loop for k-points */

    for (kloop=0; kloop<T_knum; kloop++){

      k1 = T_KGrids1[kloop]; 
      k2 = T_KGrids2[kloop]; 
      k3 = T_KGrids3[kloop];

      /* make H(k) and S(k), and solve the generalized eigenvalue problem */ 

      Calc_Hk_Sk(Hk,Sk,spin,k1,k2,k3);
      lapack_zhegvx(MatDim,Hk,Sk,WF,eval);

      /* apply the scissor operator */

      for (i=No[spin]; i<MatDim; i++){
        eval[i] += scissor; 
      }

      /* calculate A11, A12, A21, and A22 */

      Calc_A11_A12_A21_A22(spin,k1,k2,k3,WF,eval,PF1,PF2,A11,A12,A21,A22);

      for (p=0; p<3; p++){ 
	for (i=0; i<No[spin]; i++){
	  for (j=0; j<No[spin]; j++){
	    Afull[p][j*MatDim+i] = Cadd(A11[0][p][j*No[spin]+i],A11[1][p][j*No[spin]+i]);
	  }
	}
      }

      for (p=0; p<3; p++){ 
	for (i=0; i<No[spin]; i++){
	  for (j=0; j<Nuo[spin]; j++){
	    Afull[p][(j+No[spin])*MatDim+i] = Cadd(A12[0][p][j*No[spin]+i],A12[1][p][j*No[spin]+i]);
	  }
	}
      }

      for (p=0; p<3; p++){ 
	for (i=0; i<Nuo[spin]; i++){
	  for (j=0; j<No[spin]; j++){
	    Afull[p][j*MatDim+i+No[spin]] = Cadd(A21[0][p][j*Nuo[spin]+i],A21[1][p][j*Nuo[spin]+i]);
	  }
	}
      }

      for (p=0; p<3; p++){ 
	for (i=0; i<Nuo[spin]; i++){
	  for (j=0; j<Nuo[spin]; j++){
	    Afull[p][(j+No[spin])*MatDim+i+No[spin]] = Cadd(A22[0][p][j*Nuo[spin]+i],A22[1][p][j*Nuo[spin]+i]);
	  }
	}
      }

      /* calculate the momentum matrix */

      Calc_Momentum_Matrix(spin, k1, k2, k3, WF, eval, P);

      /*************************************************************
       calculate the coefficient matrices D in the omega0 expansion
       by the recurrence formula
      *************************************************************/

      /* set D in which 1 is set to the diagonal parts of the occupied states. */

      for (p=0; p<=pth; p++){
	for (q=0; q<independent_count[p]; q++){
	  for (N=0; N<=pmax_d; N++){
	    for (m=0; m<MatDim*MatDim; m++){
	      D[p][q][N][m] = Complex(0.0,0.0);
	    }
	  }
	}
      }

      for (m=0; m<No[spin]; m++){
	D[0][0][0][m*MatDim+m] = Complex(1.0,0.0);
      }

      /* loop for p */

      for (p=1; p<=pth; p++){

	/* loop for irreducible D */

	for (q=0; q<independent_count[p]; q++){

	  /* set Nxyz */

	  Nxyz[0] = 0; Nxyz[1] = 0; Nxyz[2] = 0; 
	  for (i=0; i<p; i++){
	    xyz = independent_list[p][q][i];
	    Nxyz[xyz]++;
	  }

	  /* loop n1 */

	  for (n1=0; n1<=pmax_d; n1++){

	    /* loop n2 */

	    for (n2=0; n2<=n1; n2++){

	      /* initialize work_array */

	      for (i=0; i<MatDim*MatDim; i++){
		work_array[i] = Complex(0.0,0.0);
	      }

	      /* sum over xyz */

	      for (xyz=0; xyz<3; xyz++){

		Ii = find_index(p,independent_list[p][q],xyz,independent_list,independent_count);

		if (Ii!=-1){

		  alphai = (double)Nxyz[xyz]/(double)p;

		  /* zgemm of P and D */

		  alpha = Complex(alphai, 0.0);
		  beta = Complex(1.0,0.0); 
		  m = MatDim; n = MatDim; k = MatDim;
		  zgemm_( "N","N", &m, &n, &k, &alpha, P[xyz], &m, 
			  D[p-1][Ii][n1-n2], &k, &beta, work_array, &m );

		  /* zgemm of D and P */

		  alpha = Complex(-alphai, 0.0);
		  beta = Complex(1.0,0.0); 
		  m = MatDim; n = MatDim; k = MatDim;
		  zgemm_( "N","N", &m, &n, &k, &alpha, D[p-1][Ii][n1-n2], 
			  &m, P[xyz], &k, &beta, work_array, &m );

		} // end of if (Ii!=-1)
	      } // xyz

              /* Hadamard product of gamma_{n2}^{p} and work_array */

	      for (m=0; m<MatDim; m++){

		em = eval[m];

		for (n=0; n<MatDim; n++){

		  en = eval[n];

		  ctmp1.r = em - en;
		  ctmp1.i =-eta;
		  ctmp2 = Complex(1.0,0.0);
                    
		  for (i=0; i<=n2; i++){
		    ctmp2 = Cmul(ctmp2,ctmp1);
		  } // i

		  ctmp1 = Complex(pow((double)p,(double)n2),0.0);
		  ctmp1 = Cdiv(ctmp1,ctmp2);
		  ctmp1.r = -ctmp1.r;
		  ctmp1.i = -ctmp1.i;

		  ctmp2.r = ctmp1.r*work_array[n*MatDim+m].r - ctmp1.i*work_array[n*MatDim+m].i;
		  ctmp2.i = ctmp1.r*work_array[n*MatDim+m].i + ctmp1.i*work_array[n*MatDim+m].r;

		  D[p][q][n1][n*MatDim+m].r += ctmp2.r;
		  D[p][q][n1][n*MatDim+m].i += ctmp2.i;

		} // n
	      } // m
	    } // n2

	  } // n1
	} // q
      } // p        

      /*********************************************
            calculate pth-order susceptibility 
      *********************************************/

      /* loop for omega0 */ 

      for (omg=0; omg<Ng; omg++){

        /* set omega0 */

        omega0 = omega_min + (double)omg*domega;

	/********************************************************
                calculate density matrices with underline 
                by the recurrence formula
	********************************************************/

        /* set rho[0] in which 1 is set to the diagonal parts of the occupied states. */
 
        for (i=0; i<MatDim*MatDim; i++){ 
          rho[0][0][i] = Complex(0.0,0.0); 
        }
        for (i=0; i<No[spin]; i++){
          rho[0][0][i*MatDim+i] = Complex(1.0,0.0); 
        }

        /* loop for p */

        for (p=1; p<=pth; p++){

          /* set Omega */

          Omega.r = (double)p*omega0;
          Omega.i = eta;

          /* loop for irreducible rho */

          for (q=0; q<independent_count[p]; q++){

            /* set Nxyz */

            Nxyz[0] = 0; Nxyz[1] = 0; Nxyz[2] = 0; 
            for (i=0; i<p; i++){
              xyz = independent_list[p][q][i];
              Nxyz[xyz]++;
	    }

            /* initialize work_array */

            for (i=0; i<MatDim*MatDim; i++){
	      work_array[i] = Complex(0.0,0.0);
	    }

            /* sum over xyz */

            for (xyz=0; xyz<3; xyz++){

	      l = find_index(p,independent_list[p][q],xyz,independent_list,independent_count);

              if (l!=-1){

                alphai = (double)Nxyz[xyz]/(double)p;
                           
                /* zgemm of P and rho */

                alpha = Complex(alphai, 0.0);
                beta = Complex(1.0,0.0); 
	        m = MatDim; n = MatDim; k = MatDim;
		zgemm_("N","N", &m, &n, &k, &alpha, P[xyz], &m, rho[p-1][l], &k, &beta, work_array, &m);

                /* zgemm of rho and P */

                alpha = Complex(-alphai, 0.0);
                beta = Complex(1.0,0.0); 
		m = MatDim; n = MatDim; k = MatDim;
		zgemm_("N","N", &m, &n, &k, &alpha, rho[p-1][l], &m, P[xyz], &k, &beta, work_array, &m);
                  
	      } // end of if (l!=-1)
	    } // xyz

            /* multiply work_array with 1/(hbar*Omega-(emu-enu)) */

	    for (m=0; m<MatDim; m++){

              em = eval[m];

	      for (n=0; n<MatDim; n++){

    	        en = eval[n];

                /* calculate rho with underline */

                ctmp1.r = Omega.r - (em - en);
                ctmp1.i = Omega.i;
                ctmp2 = Complex(1.0,0.0);
                ctmp3 = Cdiv(ctmp2,ctmp1);
                ctmp1 = work_array[n*MatDim+m];
                rho[p][q][n*MatDim+m] = Cmul(ctmp1,ctmp3);

  	      } // n
	    } // m

	  } // q 
	} // p 

	/********************************************************
                        rho_int = rho - rho_div
	********************************************************/

        for (p=1; p<=pth; p++){
          for (q=0; q<independent_count[p]; q++){

	    /* initialize work_array */

	    for (i=0; i<MatDim*MatDim; i++){
	      work_array[i] = Complex(0.0,0.0);
	    }

            /* sum over n1 */

            for (n1=0; n1<=(p-1); n1++){

   	      pomega0 = pow(omega0,(double)n1);

	      for (m=0; m<MatDim; m++){
		for (n=0; n<MatDim; n++){

                  work_array[n*MatDim+m].r += pomega0*D[p][q][n1][n*MatDim+m].r; 
                  work_array[n*MatDim+m].i += pomega0*D[p][q][n1][n*MatDim+m].i; 

		} // n
	      } // m
	    } // n1

            /* substract the divergent contribution from the original rho */

	    for (m=0; m<MatDim; m++){
	      for (n=0; n<MatDim; n++){

		/*
                rho[p][q][n*MatDim+m].r -= work_array[n*MatDim+m].r;
                rho[p][q][n*MatDim+m].i -= work_array[n*MatDim+m].i;
		*/

		/*
                rho[p][q][n*MatDim+m].r = work_array[n*MatDim+m].r;
                rho[p][q][n*MatDim+m].i = work_array[n*MatDim+m].i;
		*/

	      } // n
	    } // m

	  } // q
	} // p

	/********************************************************
                              calculate chi
	********************************************************/

        /* loop for p */

        for (p=1; p<=pth; p++){

          /* loop for irreducible chi */

          for (q=0; q<independent_count[p]; q++){

            Nxyz[0] = 0; Nxyz[1] = 0; Nxyz[2] = 0; 
            for (i=0; i<p; i++){
              xyz = independent_list[p][q][i];
              Nxyz[xyz]++;
	    }

	    csum[0] = Complex(0.0,0.0); 
	    csum[1] = Complex(0.0,0.0); 
	    csum[2] = Complex(0.0,0.0); 

	    /* trace(A_i*\underline{rho_I^(p)}) */

            for (i=0; i<3; i++){
	      for (m=0; m<MatDim; m++){
		for (n=0; n<MatDim; n++){
 
                  ctmp1 = Afull[i][n*MatDim+m];
                  ctmp2 = rho[p][q][m*MatDim+n];

                  csum[i].r += (ctmp1.r*ctmp2.r - ctmp1.i*ctmp2.i); 
                  csum[i].i += (ctmp1.r*ctmp2.i + ctmp1.i*ctmp2.r);
		}
	      }
	    }             

            /* add the trace to chi */

            for (i=0; i<3; i++){
              csum[i].r *= (double)T_k_op[kloop];
              csum[i].i *= (double)T_k_op[kloop];
	    }

            chi[omg][0][p][q] = Cadd(chi[omg][0][p][q], csum[0]);
            chi[omg][1][p][q] = Cadd(chi[omg][1][p][q], csum[1]);
            chi[omg][2][p][q] = Cadd(chi[omg][2][p][q], csum[2]);

	  } // q
	} // p

      } // omg          
    } /* end of kloop */ 

    /* multiplying chi with kappa^p, 
       and taking account of the prefactor */ 

    double f;

    for (omg=0; omg<Ng; omg++){

      omega0 = omega_min + (double)omg*domega;

      kappa = Complex(0.0,-1.0/omega0);
      kappa_p = Complex(1.0,0.0);

      f = 1.0/(1.0+exp(-2000.0*(omega0-1.0/eV2Hartree)));

      for (p=1; p<=pth; p++){

        kappa_p = Cmul(kappa,kappa_p);
        mkappa_p.r =-kappa_p.r*f;
        mkappa_p.i =-kappa_p.i*f;

        for (q=0; q<independent_count[p]; q++){

          chi[omg][0][p][q] = Cmul(chi[omg][0][p][q], mkappa_p);
          chi[omg][1][p][q] = Cmul(chi[omg][1][p][q], mkappa_p);
          chi[omg][2][p][q] = Cmul(chi[omg][2][p][q], mkappa_p);

	} // q
      } // p
    } // omg

  } /* end of spin */

  /* we do not take account of spin multiplicity yet. */

  for (omg=0; omg<Ng; omg++){
    omega0 = omega_min + (double)omg*domega;

    //printf("%15.12f %15.12f %15.12f\n",omega0*eV2Hartree,chi[omg][0][1][0].r,chi[omg][0][1][0].i);

    printf("%15.12f %15.12f %15.12f\n",omega0*eV2Hartree,chi[omg][0][2][4].r,chi[omg][0][2][4].i);

  }

  /*************************************************************
                    freeing of arrays
  *************************************************************/

  for (i=0; i<3; i++){
    free(Afull[i]);
  }
  free(Afull);

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

  free(work_array);

  for (p=0; p<(pth+1); p++){
    for (q=0; q<independent_count[p]; q++){
      free(rho[p][q]);
    }
    free(rho[p]);
  }
  free(rho);

  for (p=0; p<(pth+1); p++){
    for (q=0; q<independent_count[p]; q++){
      for (n=0; n<(pth+1); n++){
        free(D[p][q][n]);
      }
      free(D[p][q]);
    }
    free(D[p]);
  }
  free(D);

  for (omg=0; omg<Ng; omg++){
    for (xyz=0; xyz<3; xyz++){
      for (p=0; p<(pth+1); p++){
	free(chi[omg][xyz][p]);
      }
      free(chi[omg][xyz]);
    }
    free(chi[omg]);
  }
  free(chi);

  free(Hk);
  free(Sk);
  free(WF);
  free(eval);

  for (i=0; i<3; i++){
    free(P[i]);
  }
  free(P);

  for (p=0; p<pth; p++){
    for (i = 0; i < independent_count[p]; ++i) {
      free(independent_list[p][i]);
    }
    free(independent_list[p]);
  }
  free(independent_list);

  free(independent_count);
}


void pth_Harmonic_Generation_sigma(int pth)
{
  int i,j,n,m,k,l,n1,n2,Ii,p,q,xyz,spin,kloop,N,Ng,omg,Nxyz[3];
  double k1,k2,k3,*eval,eta,max_x=30.0,Bi,alphai,scissor;
  double omega0,pomega0,omega_min,omega_max,domega,em,en,fm,fn;
  double Beta,x,tmp1,vtmp[4],Cell_Volume;
  dcomplex *Hk,*Sk,*WF,**P,***rho,****D,****sigma;
  dcomplex *work_array;
  dcomplex csum[3],ctmp1,ctmp2,ctmp3,ctmp4,Omega; 
  dcomplex alpha = {1.0,0.0}; 
  dcomplex beta = {0.0,0.0};
  dcomplex kappa,kappa_p,mkappa_p;

  /* set parameters */

  Ng = 200;
  omega_min = 0.001/eV2Hartree; 
  omega_max =  5.0/eV2Hartree;   
  domega = (omega_max - omega_min)/(double)(Ng-1);
  //eta = 0.05/eV2Hartree; 

  eta = 0.05/eV2Hartree; 

  Beta = 10000.0;
  scissor = 1.151/eV2Hartree; 
  //scissor = 0.0/eV2Hartree; 

  /*************************************************************
         setting information for the 0-th to p-th sigma
  *************************************************************/

  int ***independent_list = malloc(sizeof(int**)*(pth+1));
  int *independent_count = malloc(sizeof(int)*(pth+1));

  for (p=0; p<=pth; p++){

    /* Build independent list */
    init_independent_components(p,
			        &independent_list[p],
			        &independent_count[p]);


    printf("Independent components for p=%d (total %d):\n", p, independent_count[p]);
    for (i=0; i<independent_count[p]; ++i) {
      int j;
      printf("[%2d] (", i);
      for (j=0; j<p; ++j) {
	printf("%d",independent_list[p][i][j]);
	if (j<(p-1)) printf(",");
      }
      printf(")\n");
    }
  }

  /*
  p = 1; 
  q = 0;
  xyz = 0;
  printf("%2d\n",find_index(p,independent_list[p][q],xyz,independent_list,independent_count));

  MPI_Finalize();
  exit(0);
  */

  /*************************************************************
                    allocation of arrays
  *************************************************************/

  work_array = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));

  rho = (dcomplex***)malloc(sizeof(dcomplex**)*(pth+1));
  for (p=0; p<(pth+1); p++){
    rho[p] = (dcomplex**)malloc(sizeof(dcomplex*)*independent_count[p]);
    for (q=0; q<independent_count[p]; q++){
      rho[p][q] = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));
    }
  }

  /* D[p][I][n][MatDim*MatDim]*/
  D = (dcomplex****)malloc(sizeof(dcomplex***)*(pth+1));
  for (p=0; p<(pth+1); p++){
    D[p] = (dcomplex***)malloc(sizeof(dcomplex**)*independent_count[p]);
    for (q=0; q<independent_count[p]; q++){
      D[p][q] = (dcomplex**)malloc(sizeof(dcomplex*)*(pth+1));
      for (n=0; n<(pth+1); n++){
        D[p][q][n] = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));
      }
    }
  }

  sigma = (dcomplex****)malloc(sizeof(dcomplex***)*Ng);
  for (omg=0; omg<Ng; omg++){
    sigma[omg] = (dcomplex***)malloc(sizeof(dcomplex**)*3);
    for (xyz=0; xyz<3; xyz++){
      sigma[omg][xyz] = (dcomplex**)malloc(sizeof(dcomplex*)*(pth+1));
      for (p=0; p<(pth+1); p++){
	sigma[omg][xyz][p] = (dcomplex*)malloc(sizeof(dcomplex)*independent_count[p]);
	for (i=0; i<independent_count[p]; i++){
	  sigma[omg][xyz][p][i] = Complex(0.0,0.0);
	} 
      }
    }
  }

  Hk = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));
  Sk = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));
  WF = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));
  eval = (double*)malloc(sizeof(double)*MatDim);

  P = (dcomplex**)malloc(sizeof(dcomplex*)*3);
  for (i=0; i<3; i++){
    P[i] = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));
  }

  /* calculte cell volume */

  Cross_Product(tv[2],tv[3],vtmp);
  Cell_Volume = fabs(Dot_Product(tv[1],vtmp));

  /*************************************************************
                  loops for spin and k-points
  *************************************************************/

  for (spin=0; spin<=SpinP_switch; spin++){ 

    /* loop for k-points */

    for (kloop=0; kloop<T_knum; kloop++){

      k1 = T_KGrids1[kloop]; 
      k2 = T_KGrids2[kloop]; 
      k3 = T_KGrids3[kloop];

      /* make H(k) and S(k), and solve the generalized eigenvalue problem */ 

      Calc_Hk_Sk(Hk,Sk,spin,k1,k2,k3);
      lapack_zhegvx(MatDim,Hk,Sk,WF,eval);

      /* apply the scissor operator */

      for (i=No[spin]; i<MatDim; i++){
        eval[i] += scissor; 
      }

      /* calculate the momentum matrix */

      Calc_Momentum_Matrix(spin, k1, k2, k3, WF, eval, P);

      /*************************************************************
       calculate the coefficient matrices D in the omega0 expansion
       by the recurrence formula
      *************************************************************/

      /* set D in which 1 is set to the diagonal parts of the occupied states. */

      for (p=0; p<=pth; p++){
	for (q=0; q<independent_count[p]; q++){
	  for (N=0; N<=pth; N++){
	    for (m=0; m<MatDim*MatDim; m++){
	      D[p][q][N][m] = Complex(0.0,0.0);
	    }
	  }
	}
      }

      for (m=0; m<No[spin]; m++){
	D[0][0][0][m*MatDim+m] = Complex(1.0,0.0);
      }

      /* loop for p */

      for (p=1; p<=pth; p++){

	/* loop for irreducible D */

	for (q=0; q<independent_count[p]; q++){

	  /* set Nxyz */

	  Nxyz[0] = 0; Nxyz[1] = 0; Nxyz[2] = 0; 
	  for (i=0; i<p; i++){
	    xyz = independent_list[p][q][i];
	    Nxyz[xyz]++;
	  }

	  /* loop n1 */

	  for (n1=0; n1<=pth; n1++){

	    /* loop n2 */

	    for (n2=0; n2<=n1; n2++){

	      /* initialize work_array */

	      for (i=0; i<MatDim*MatDim; i++){
		work_array[i] = Complex(0.0,0.0);
	      }

	      /* sum over xyz */

	      for (xyz=0; xyz<3; xyz++){

		Ii = find_index(p,independent_list[p][q],xyz,independent_list,independent_count);

		if (Ii!=-1){

		  alphai = (double)Nxyz[xyz]/(double)p;

		  /* zgemm of P and D */

		  alpha = Complex(alphai, 0.0);
		  beta = Complex(1.0,0.0); 
		  m = MatDim; n = MatDim; k = MatDim;
		  zgemm_( "N","N", &m, &n, &k, &alpha, P[xyz], &m, 
			  D[p-1][Ii][n1-n2], &k, &beta, work_array, &m );

		  /* zgemm of D and P */

		  alpha = Complex(-alphai, 0.0);
		  beta = Complex(1.0,0.0); 
		  m = MatDim; n = MatDim; k = MatDim;
		  zgemm_( "N","N", &m, &n, &k, &alpha, D[p-1][Ii][n1-n2], 
			  &m, P[xyz], &k, &beta, work_array, &m );

		} // end of if (Ii!=-1)
	      } // xyz

              /* Hadamard product of gamma_{n2}^{p} and work_array */

	      for (m=0; m<MatDim; m++){

		em = eval[m];

		for (n=0; n<MatDim; n++){

		  en = eval[n];

		  ctmp1.r = em - en;
		  ctmp1.i =-eta;
		  ctmp2 = Complex(1.0,0.0);
                    
		  for (i=0; i<=n2; i++){
		    ctmp2 = Cmul(ctmp2,ctmp1);
		  } // i

		  ctmp1 = Complex(pow((double)p,(double)n2),0.0);
		  ctmp1 = Cdiv(ctmp1,ctmp2);
		  ctmp1.r = -ctmp1.r;
		  ctmp1.i = -ctmp1.i;

		  ctmp2.r = ctmp1.r*work_array[n*MatDim+m].r - ctmp1.i*work_array[n*MatDim+m].i;
		  ctmp2.i = ctmp1.r*work_array[n*MatDim+m].i + ctmp1.i*work_array[n*MatDim+m].r;

		  D[p][q][n1][n*MatDim+m].r += ctmp2.r;
		  D[p][q][n1][n*MatDim+m].i += ctmp2.i;

		} // n
	      } // m
	    } // n2

	  } // n1
	} // q
      } // p        

      /*********************************************
            calculate pth-order susceptibility 
      *********************************************/

      /* loop for omega0 */ 

      for (omg=0; omg<Ng; omg++){

        /* set omega0 */

        omega0 = omega_min + (double)omg*domega;

	/********************************************************
                calculate density matrices with underline 
                by the recurrence formula
	********************************************************/

        /* set rho[0] in which 1 is set to the diagonal parts of the occupied states. */
 
        for (i=0; i<MatDim*MatDim; i++){ 
          rho[0][0][i] = Complex(0.0,0.0); 
        }
        for (i=0; i<No[spin]; i++){
          rho[0][0][i*MatDim+i] = Complex(1.0,0.0); 
        }

        /* loop for p */

        for (p=1; p<=pth; p++){

          /* set Omega */

          Omega.r = (double)p*omega0;
          Omega.i = eta;

          /* loop for irreducible rho */

          for (q=0; q<independent_count[p]; q++){

            /* set Nxyz */

            Nxyz[0] = 0; Nxyz[1] = 0; Nxyz[2] = 0; 
            for (i=0; i<p; i++){
              xyz = independent_list[p][q][i];
              Nxyz[xyz]++;
	    }

            /* initialize work_array */

            for (i=0; i<MatDim*MatDim; i++){
	      work_array[i] = Complex(0.0,0.0);
	    }

            /* sum over xyz */

            for (xyz=0; xyz<3; xyz++){

	      l = find_index(p,independent_list[p][q],xyz,independent_list,independent_count);

              if (l!=-1){

                alphai = (double)Nxyz[xyz]/(double)p;
                           
                /* zgemm of P and rho */

                alpha = Complex(alphai, 0.0);
                beta = Complex(1.0,0.0); 
	        m = MatDim; n = MatDim; k = MatDim;
		zgemm_("N","N", &m, &n, &k, &alpha, P[xyz], &m, rho[p-1][l], &k, &beta, work_array, &m);

                /* zgemm of rho and P */

                alpha = Complex(-alphai, 0.0);
                beta = Complex(1.0,0.0); 
		m = MatDim; n = MatDim; k = MatDim;
		zgemm_("N","N", &m, &n, &k, &alpha, rho[p-1][l], &m, P[xyz], &k, &beta, work_array, &m);
                  
	      } // end of if (l!=-1)
	    } // xyz

            /* multiply work_array with 1/(hbar*Omega-(emu-enu)) */

	    for (m=0; m<MatDim; m++){

              em = eval[m];

	      for (n=0; n<MatDim; n++){

    	        en = eval[n];

                /* calculate rho with underline */

                ctmp1.r = Omega.r - (em - en);
                ctmp1.i = Omega.i;
                ctmp2 = Complex(1.0,0.0);
                ctmp3 = Cdiv(ctmp2,ctmp1);
                ctmp1 = work_array[n*MatDim+m];
                rho[p][q][n*MatDim+m] = Cmul(ctmp1,ctmp3);

  	      } // n
	    } // m

	  } // q 
	} // p 

	/********************************************************
                        rho_int = rho - rho_div
	********************************************************/

        for (p=1; p<=pth; p++){
          for (q=0; q<independent_count[p]; q++){

	    /* initialize work_array */

	    for (i=0; i<MatDim*MatDim; i++){
	      work_array[i] = Complex(0.0,0.0);
	    }

            /* sum over n1 */

            for (n1=0; n1<=(p-1); n1++){

   	      pomega0 = pow(omega0,(double)n1);

	      for (m=0; m<MatDim; m++){
		for (n=0; n<MatDim; n++){

                  work_array[n*MatDim+m].r += pomega0*D[p][q][n1][n*MatDim+m].r; 
                  work_array[n*MatDim+m].i += pomega0*D[p][q][n1][n*MatDim+m].i; 

		} // n
	      } // m
	    } // n1

            /* substract the divergent contribution from the original rho */

	    for (m=0; m<MatDim; m++){
	      for (n=0; n<MatDim; n++){

                rho[p][q][n*MatDim+m].r -= work_array[n*MatDim+m].r;
                rho[p][q][n*MatDim+m].i -= work_array[n*MatDim+m].i;

                //rho[p][q][n*MatDim+m].r = work_array[n*MatDim+m].r;
                //rho[p][q][n*MatDim+m].i = work_array[n*MatDim+m].i;

	      } // n
	    } // m

	  } // q
	} // p


	/********************************************************
                              calculate sigma
	********************************************************/

        /* loop for p */

        for (p=1; p<=pth; p++){

          /* loop for irreducible sigma */

          for (q=0; q<independent_count[p]; q++){

            Nxyz[0] = 0; Nxyz[1] = 0; Nxyz[2] = 0; 
            for (i=0; i<p; i++){
              xyz = independent_list[p][q][i];
              Nxyz[xyz]++;
	    }

	    csum[0] = Complex(0.0,0.0); 
	    csum[1] = Complex(0.0,0.0); 
	    csum[2] = Complex(0.0,0.0); 

	    /* trace(p_i*\underline{rho_I^(p)}) */

            for (i=0; i<3; i++){
	      for (m=0; m<MatDim; m++){
		for (n=0; n<MatDim; n++){
 
                  ctmp1 = P[i][n*MatDim+m];
                  ctmp2 = rho[p][q][m*MatDim+n];

                  csum[i].r += (ctmp1.r*ctmp2.r - ctmp1.i*ctmp2.i); 
                  csum[i].i += (ctmp1.r*ctmp2.i + ctmp1.i*ctmp2.r);
		}
	      }
	    }             



	    /* +trace(\underline{rho_I(i)^(p)}) */

	    if (0){
            for (i=0; i<3; i++){

              l = find_index(p,independent_list[p][q],i,independent_list,independent_count);

              if (l!=-1 && Nxyz[i]!=0){

		Bi = (double)fact(p-1);
		for (j=0; j<3; j++){
		  if (i==j) Bi /= (double)fact(Nxyz[j]-1);
		  else      Bi /= (double)fact(Nxyz[j]);
		}

		ctmp1 = Complex(0.0,0.0); 
		for (m=0; m<MatDim; m++){
		  ctmp1.r += rho[p-1][l][m*MatDim+m].r;               
		  ctmp1.i += rho[p-1][l][m*MatDim+m].i;
		}

		csum[i].r += Bi*ctmp1.r;            
		csum[i].i += Bi*ctmp1.i;            
        
	      } // end of if (l!=-1)
	    } // i
	    }

	    /*
            if (p==1 && q==0){
              printf("VVV0 %18.15f %18.15f\n",csum[0].r,csum[0].i);
	    } 
	    */ 

            /* add the trace to sigma */

            for (i=0; i<3; i++){
              csum[i].r *= (double)T_k_op[kloop];
              csum[i].i *= (double)T_k_op[kloop];
	    }

            sigma[omg][0][p][q] = Cadd(sigma[omg][0][p][q], csum[0]);
            sigma[omg][1][p][q] = Cadd(sigma[omg][1][p][q], csum[1]);
            sigma[omg][2][p][q] = Cadd(sigma[omg][2][p][q], csum[2]);

	  } // q
	} // p


	if (0){

        dcomplex df,de;

        csum[0] = Complex(0.0,0.0);
	for (m=0; m<MatDim; m++){

          em = eval[m];
          if (m<No[spin]) fm = 1.0; else fm = 0.0;

  	  for (n=0; n<MatDim; n++){

            en = eval[n];
            if (n<No[spin]) fn = 1.0; else fn = 0.0;

            de = Complex(eval[n]-eval[m], eta);
            df = Complex(fm-fn, 0.0); 
            ctmp1 = Cdiv(df,de);
            ctmp2 = Cmul(P[0][n*MatDim+m],P[0][m*MatDim+n]);
            ctmp3 = Cmul(ctmp1,ctmp2);
            csum[0] = Cadd(csum[0],ctmp3);
	  }
	}

        printf("VVV1 %18.15f %18.15f\n",csum[0].r,csum[0].i);
        MPI_Finalize();
        exit(0);

	}


      } // omg          
    } /* end of kloop */ 

    /* multiplying sigma with kappa^p, 
       and taking account of the prefactor */ 

    for (omg=0; omg<Ng; omg++){

      omega0 = omega_min + (double)omg*domega;

      kappa = Complex(0.0,-1.0/omega0);
      kappa_p = Complex(1.0,0.0);

      for (p=1; p<=pth; p++){

        kappa_p = Cmul(kappa,kappa_p);
        mkappa_p.r =-kappa_p.r;
        mkappa_p.i =-kappa_p.i;

        for (q=0; q<independent_count[p]; q++){

          sigma[omg][0][p][q] = Cmul(sigma[omg][0][p][q], mkappa_p);
          sigma[omg][1][p][q] = Cmul(sigma[omg][1][p][q], mkappa_p);
          sigma[omg][2][p][q] = Cmul(sigma[omg][2][p][q], mkappa_p);

	} // q
      } // p
    } // omg

  } /* end of spin */

  /* we do not take account of spin multiplicity yet. */

  for (omg=0; omg<Ng; omg++){
    omega0 = omega_min + (double)omg*domega;

    //printf("%15.12f %15.12f %15.12f\n",omega0*eV2Hartree,sigma[omg][0][1][0].r,sigma[omg][0][1][0].i);

    printf("%15.12f %15.12f %15.12f\n",omega0*eV2Hartree,sigma[omg][0][2][4].r,sigma[omg][0][2][4].i);

  }

  /*************************************************************
                    freeing of arrays
  *************************************************************/

  free(work_array);

  for (p=0; p<(pth+1); p++){
    for (q=0; q<independent_count[p]; q++){
      free(rho[p][q]);
    }
    free(rho[p]);
  }
  free(rho);

  for (p=0; p<(pth+1); p++){
    for (q=0; q<independent_count[p]; q++){
      for (n=0; n<(pth+1); n++){
        free(D[p][q][n]);
      }
      free(D[p][q]);
    }
    free(D[p]);
  }
  free(D);

  for (omg=0; omg<Ng; omg++){
    for (xyz=0; xyz<3; xyz++){
      for (p=0; p<(pth+1); p++){
	free(sigma[omg][xyz][p]);
      }
      free(sigma[omg][xyz]);
    }
    free(sigma[omg]);
  }
  free(sigma);

  free(Hk);
  free(Sk);
  free(WF);
  free(eval);

  for (i=0; i<3; i++){
    free(P[i]);
  }
  free(P);

  for (p=0; p<pth; p++){
    for (i = 0; i < independent_count[p]; ++i) {
      free(independent_list[p][i]);
    }
    free(independent_list[p]);
  }
  free(independent_list);

  free(independent_count);
}


int fact(int N)
{
  if      (N<0)  return 0;
  else if (N==0) return 1;
  else {
    int i,j;
    j = 1;
    for (i=1; i<=N; i++){ j *= i; }
    return j;
  }
}

int find_index(int p, int *list, int xyz, int ***independent_list, int *independent_count)
{
  int i,j,k,po,result; 
  int *idx = malloc(sizeof(int)*(p+1));
  
  if (p<=0){
    result = -1;
  }

  else {

    po = 0;
    for (i=0; i<p; i++){
      if (list[i]==xyz && po==0){ 
        po = 1;
        j = i;
      }
    } 

    if (po==0){
      result = -1;
    }

    else{

      k = 0;
      for (i=0; i<p; i++){
        if (i!=j){
          idx[k] = list[i];
          k++;
        }
      }

      /*
      for (i=0; i<(p-1); i++){
        printf("%d ",idx[i]);
      }        
      printf("\n");
      */

      result = map_to_independent(p-1, idx, independent_list[p-1], independent_count[p-1]);
    }
  }    

  return result;
}



void Chern_Number()
{
  int i,j,n,m,k,l,spin,kloop;
  double k1,k2,k3,sum,*eval;
  double ChernNum0[2][3][3],ChernNum[2][3][3];
  dcomplex *Hk,*Sk,*WF,***A12,***A21,*A12A21;
  dcomplex alpha = {1.0,0.0}; 
  dcomplex beta = {0.0,0.0};

  /*************************************************************
                    allocation of arrays
  *************************************************************/

  Hk = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));
  Sk = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));
  WF = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));
  eval = (double*)malloc(sizeof(double)*MatDim);

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

  A12A21 = (dcomplex*)malloc(sizeof(dcomplex)*Nomax*Nomax);

  /*************************************************************
                  loops for spin and k-points
  *************************************************************/

  for (spin=0; spin<=SpinP_switch; spin++){ 
    for (i=0; i<3; i++){
      for (j=0; j<3; j++){
	ChernNum0[spin][i][j] = 0.0;
	ChernNum[spin][i][j] = 0.0;
      }
    }
  }

  for (spin=0; spin<=SpinP_switch; spin++){ 

    /* loop for k-points */

    for (kloop=0; kloop<T_knum; kloop++){

      k1 = T_KGrids1[kloop]; 
      k2 = T_KGrids2[kloop]; 
      k3 = T_KGrids3[kloop];

      /* make H(k) and S(k), and solve the generalized eigenvalue problem */ 

      Calc_Hk_Sk(Hk,Sk,spin,k1,k2,k3);
      lapack_zhegvx(MatDim,Hk,Sk,WF,eval);

      /* calculate A12 and A21 */

      Calc_A12_A21(spin,k1,k2,k3,WF,eval,A12,A21);
 
      for (i=0; i<3; i++){
        for (j=0; j<No[spin]*Nuo[spin]; j++){
          A12[0][i][j].r += A12[1][i][j].r;    
          A12[0][i][j].i += A12[1][i][j].i;    
          A21[0][i][j].r += A21[1][i][j].r;    
          A21[0][i][j].i += A21[1][i][j].i;
	}    
      }

      /* calculate a contribution to the Chern number */

      for (i=0; i<3; i++){
	for (j=0; j<3; j++){

	  m = No[spin]; n = No[spin]; k = Nuo[spin];
	  zgemm_("N","N", &m, &n, &k, &alpha, A12[0][i], &m, A21[0][j], &k, &beta, A12A21, &m);

	  sum = 0.0;
	  for (l=0; l<No[spin]; l++){
	    sum += A12A21[l*No[spin]+l].r; 
	  }

	  ChernNum0[spin][i][j] += sum*(double)T_k_op[kloop]/(2.0*PI);

          //if (i==1 && j==0) printf("%10.5f %15.12f\n",k1,sum);
          //if (i==1 && j==0) printf("%10.5f %10.5f %15.12f\n",k1,k2,sum); 

	} // j
      } // i

    } /* end of kloop */ 
  } /* end of spin */

  /* calculate Chern number */

  for (i=0; i<3; i++){
    for (j=0; j<3; j++){
      ChernNum[0][i][j] = ChernNum0[0][i][j] - ChernNum0[0][j][i]; 
    }
  }

  for (i=0; i<3; i++){
    for (j=0; j<3; j++){
      printf("Chern Number i=%2d j=%2d CN=%15.12f\n",i,j,ChernNum[0][i][j]);  
    }
  }

  /*************************************************************
                    freeing of arrays
  *************************************************************/

  free(Hk);
  free(Sk);
  free(WF);
  free(eval);

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

  free(A12A21);
}




void Polarization()
{
  int i,j,spin,kloop,m,gm,lm;
  double k1,k2,k3,tmp,Pol1[2][3],Pol2[2][3],be[4];
  double *eval,sum00,sum01,sum02,sum10,sum11,sum12;
  dcomplex ***A11,*Hk,*Sk,*WF;

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

  Hk = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));
  Sk = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));
  WF = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));
  eval = (double*)malloc(sizeof(double)*MatDim);

  /*************************************************************
          calculation of guiding functions and FR1
  *************************************************************/

  /* calculate guiding functions, PF1 */

  Calc_Guiding_Functions_1(PF1);

  /* calculate FR1 */

  Calc_FR_1(PF1);  

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

      /* make H(k) and S(k), and solve the generalized eigenvalue problem */ 

      Calc_Hk_Sk(Hk,Sk,spin,k1,k2,k3);
      lapack_zhegvx(MatDim,Hk,Sk,WF,eval);

      /* calculate A11 */

      Calc_A11(spin,k1,k2,k3,WF,eval,PF1,A11);

      /* calculate Polarization */

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
  printf("                     Cell information                    \n");      fflush(stdout);
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

  free(Hk);
  free(Sk);
  free(WF);
  free(eval);

}



void Dispersion_Berry_Connections()
{
  int m,i,nkpath,spin,Nk,p,kloop;
  double dk,k1,k2,k3,d,xs,ys,zs,xe,ye,ze;
  double sum0,sum1,sum2,dd,dx,dy,dz,step;
  double *eval,**kpath_s,**kpath_e;
  dcomplex ***A11,*Hk,*Sk,*WF;

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

  Hk = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));
  Sk = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));
  WF = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));
  eval = (double*)malloc(sizeof(double)*MatDim);

  /* calculate projection functions, PF1 */

  Calc_Guiding_Functions_1(PF1);

  /* calculate FR1 */

  Calc_FR_1(PF1);  

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

      /* make H(k) and S(k), and solve the generalized eigenvalue problem */ 

      Calc_Hk_Sk(Hk,Sk,spin,k1,k2,k3);
      lapack_zhegvx(MatDim,Hk,Sk,WF,eval);

      /* calculate A11 */

      Calc_A11(spin,k1,k2,k3,WF,eval,PF1,A11);

      /* calculate Berry connection */

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

  free(Hk);
  free(Sk);
  free(WF);
  free(eval);
}













void Calc_FR_1(double **PF1)
{
  int i,j,k,mu,ii,ij,ik,jj,kk,kloop,spin,p1,p2,p3,p;
  int Ra,Rb,Rc,m,n,l1,l2,l3;
  double k1,k2,k3,tmp,sum0,sum1,kRn,si,co;
  double *S0,*eval;
  dcomplex *Hk,*Sk,*WF,*D,*PA1,*U1;
  dcomplex alpha = {1.0,0.0}; 
  dcomplex beta = {0.0,0.0};

  /*************************************************************
                      allocation of arrays
  *************************************************************/

  Ra = (int)Nk1/2;
  Rb = (int)Nk2/2;
  Rc = (int)Nk3/2;

  S0 = (double*)malloc(sizeof(double)*MatDim*MatDim);
  PA1 = (dcomplex*)malloc(sizeof(dcomplex)*MatDim*MatDim);
  D = (dcomplex*)malloc(sizeof(dcomplex)*MatDim*MatDim);
  Hk = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));
  Sk = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));
  WF = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));
  U1 = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));
  eval = (double*)malloc(sizeof(double)*MatDim);

  /******************************************************************/
  /*******************************************************************
                            Calculate FR1
  *******************************************************************/
  /******************************************************************/

  for (spin=0; spin<(SpinP_switch+1); spin++){ 

    /* initialize FR1 */

    for (l1=-Ra; l1<=Ra; l1++){
      for (l2=-Rb; l2<=Rb; l2++){
	for (l3=-Rc; l3<=Rc; l3++){
	  for (j=0; j<No[spin]; j++){
	    for (i=0; i<MatDim; i++){
	      FR1[spin][l1+Ra][l2+Rb][l3+Rc][j*MatDim+i] = 0.0;
	    } // i
	  } // j
	} // l3
      } // l2
    } // l1

    /* loop for k-points */

    for (kloop=0; kloop<T_knum; kloop++){

      /* get the basic information */ 

      k1 = T_KGrids1[kloop]; 
      k2 = T_KGrids2[kloop]; 
      k3 = T_KGrids3[kloop];

      /* make H(k) and S(k), and solve the generalized eigenvalue problem */ 

      Calc_Hk_Sk(Hk,Sk,spin,k1,k2,k3);
      lapack_zhegvx(MatDim,Hk,Sk,WF,eval);

      /* calculate overlap between PAOs and Bloch functions */

      Calc_D(k1,k2,k3,WF,D); 

      /* calculate overlap between the guiding functions and occupied Bloch functions */

      for (mu=0; mu<No[spin]; mu++){
        for (p=0; p<No[spin]; p++){

          sum0 = 0.0; sum1 = 0.0;
          for (i=0; i<MatDim; i++){
            sum0 += PF1[spin][p*MatDim+i]*D[mu*MatDim+i].r;
            sum1 += PF1[spin][p*MatDim+i]*D[mu*MatDim+i].i;
	  }

	  PA1[p*No[spin]+mu].r = sum0;
	  PA1[p*No[spin]+mu].i =-sum1;

	} // p
      } // mu

      /* Polar decomposition of PA1 */ 

      Polar_Decompose(No[spin],  PA1, U1); 

      /* calculate \tilde{WF1} = WF1*U1, which is stored in PA1 */ 
 
      m = MatDim; n = No[spin]; k = No[spin];
      zgemm_("N","N", &m, &n, &k, &alpha, &WF[0], &m, U1, &k, &beta, PA1, &m);

      /* calculate FR1 */ 
 
      for (l1=-Ra; l1<=Ra; l1++){
        for (l2=-Rb; l2<=Rb; l2++){
          for (l3=-Rc; l3<=Rc; l3++){

  	    kRn = k1*(double)l1 + k2*(double)l2 + k3*(double)l3;
  	    si = sin(2.0*PI*kRn);
	    co = cos(2.0*PI*kRn);

            /* FR1 */  
           
            for (j=0; j<No[spin]; j++){
              for (i=0; i<MatDim; i++){

                tmp = PA1[j*MatDim+i].r*co - PA1[j*MatDim+i].i*si;
                FR1[spin][l1+Ra][l2+Rb][l3+Rc][j*MatDim+i] += (double)T_k_op[kloop]*tmp;

  	      } // i
	    } // j

          } // l3
        } // l2
      } // l1 

    } // end of kloop

    /****************************************
             normalize FR1 and FR2
    ****************************************/

    for (l1=-Ra; l1<=Ra; l1++){
      for (l2=-Rb; l2<=Rb; l2++){
	for (l3=-Rc; l3<=Rc; l3++){
	  for (j=0; j<No[spin]; j++){
	    for (i=0; i<MatDim; i++){
	      FR1[spin][l1+Ra][l2+Rb][l3+Rc][j*MatDim+i] /= tweight;
	    } // i
	  } // j
	} // l3
      } // l2
    } // l1

  } // spin

  /*
  for (l1=-Ra; l1<=Ra; l1++){
    printf("%2d %15.12f\n",l1,FR1[0][l1+Ra][0][0][0]);
  }

  MPI_Finalize();
  exit(0);
  */

  /*************************************************************
                      freeing of arrays
  *************************************************************/

  free(S0);
  free(PA1);
  free(D);
  free(Hk);
  free(Sk);
  free(WF);
  free(U1);
  free(eval);

}



void Calc_FR_1_2(double **PF1, double **PF2)
{
  int i,j,k,mu,ii,ij,ik,jj,kk,kloop,spin,p1,p2,p3,p;
  int Ra,Rb,Rc,m,n,l1,l2,l3;
  double k1,k2,k3,tmp,sum0,sum1,kRn,si,co;
  double *S0,*eval;
  dcomplex *Hk,*Sk,*WF,*D,*PA1,*PA2,*U1,*U2;
  dcomplex alpha = {1.0,0.0}; 
  dcomplex beta = {0.0,0.0};

  /*************************************************************
                      allocation of arrays
  *************************************************************/

  Ra = (int)Nk1/2;
  Rb = (int)Nk2/2;
  Rc = (int)Nk3/2;

  S0 = (double*)malloc(sizeof(double)*MatDim*MatDim);
  PA1 = (dcomplex*)malloc(sizeof(dcomplex)*MatDim*MatDim);
  PA2 = (dcomplex*)malloc(sizeof(dcomplex)*MatDim*MatDim);
  D = (dcomplex*)malloc(sizeof(dcomplex)*MatDim*MatDim);
  Hk = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));
  Sk = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));
  WF = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));
  U1 = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));
  U2 = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));
  eval = (double*)malloc(sizeof(double)*MatDim);

  /******************************************************************/
  /*******************************************************************
                        Calculate FR1 and FR2
  *******************************************************************/
  /******************************************************************/

  for (spin=0; spin<(SpinP_switch+1); spin++){ 

    /* initialize FR1 and FR2 */

    for (l1=-Ra; l1<=Ra; l1++){
      for (l2=-Rb; l2<=Rb; l2++){
	for (l3=-Rc; l3<=Rc; l3++){

	  for (j=0; j<No[spin]; j++){
	    for (i=0; i<MatDim; i++){
	      FR1[spin][l1+Ra][l2+Rb][l3+Rc][j*MatDim+i] = 0.0;
	    } // i
	  } // j

	  for (j=0; j<Nuo[spin]; j++){
	    for (i=0; i<MatDim; i++){
	      FR2[spin][l1+Ra][l2+Rb][l3+Rc][j*MatDim+i] = 0.0;
	    } // i
	  } // j

	} // l3
      } // l2
    } // l1

    /* loop for k-points */

    for (kloop=0; kloop<T_knum; kloop++){

      /* get the basic information */ 

      k1 = T_KGrids1[kloop]; 
      k2 = T_KGrids2[kloop]; 
      k3 = T_KGrids3[kloop];

      /* make H(k) and S(k), and solve the generalized eigenvalue problem */ 

      Calc_Hk_Sk(Hk,Sk,spin,k1,k2,k3);
      lapack_zhegvx(MatDim,Hk,Sk,WF,eval);

      /* calculate overlap between PAOs and Bloch functions */

      Calc_D(k1,k2,k3,WF,D); 

      /* calculate overlap between the guiding functions and occupied Bloch functions */

      for (mu=0; mu<No[spin]; mu++){
        for (p=0; p<No[spin]; p++){

          sum0 = 0.0; sum1 = 0.0;
          for (i=0; i<MatDim; i++){
            sum0 += PF1[spin][p*MatDim+i]*D[mu*MatDim+i].r;
            sum1 += PF1[spin][p*MatDim+i]*D[mu*MatDim+i].i;
	  }

	  PA1[p*No[spin]+mu].r = sum0;
	  PA1[p*No[spin]+mu].i =-sum1;

	} // p
      } // mu

      /* calculate overlap between the guiding functions and unoccupied Bloch functions */

      for (mu=0; mu<Nuo[spin]; mu++){
        for (p=0; p<Nuo[spin]; p++){

          sum0 = 0.0; sum1 = 0.0;
          for (i=0; i<MatDim; i++){
            sum0 += PF2[spin][p*MatDim+i]*D[(mu+No[spin])*MatDim+i].r;
            sum1 += PF2[spin][p*MatDim+i]*D[(mu+No[spin])*MatDim+i].i;
	  }

	  PA2[p*Nuo[spin]+mu].r = sum0;
	  PA2[p*Nuo[spin]+mu].i =-sum1;

	} // p
      } // mu

      /* Polar decomposition of PA1 and PA2 */ 

      Polar_Decompose(No[spin],  PA1, U1); 
      Polar_Decompose(Nuo[spin], PA2, U2); 

      /* calculate \tilde{WF1} = WF1*U1, which is stored in PA1 */ 
 
      m = MatDim; n = No[spin]; k = No[spin];
      zgemm_("N","N", &m, &n, &k, &alpha, &WF[0], &m, U1, &k, &beta, PA1, &m);

      /* calculate \tilde{WF2} = WF2*U2, which is stored in PA2 */ 
       
      m = MatDim; n = Nuo[spin]; k = Nuo[spin];
      zgemm_("N","N", &m, &n, &k, &alpha, &WF[No[spin]*MatDim], &m, U2, &k, &beta, PA2, &m);

      /* calculate FR */ 
 
      for (l1=-Ra; l1<=Ra; l1++){
        for (l2=-Rb; l2<=Rb; l2++){
          for (l3=-Rc; l3<=Rc; l3++){

  	    kRn = k1*(double)l1 + k2*(double)l2 + k3*(double)l3;
  	    si = sin(2.0*PI*kRn);
	    co = cos(2.0*PI*kRn);

            /* FR1 */  
           
            for (j=0; j<No[spin]; j++){
              for (i=0; i<MatDim; i++){

                tmp = PA1[j*MatDim+i].r*co - PA1[j*MatDim+i].i*si;
                FR1[spin][l1+Ra][l2+Rb][l3+Rc][j*MatDim+i] += (double)T_k_op[kloop]*tmp;

  	      } // i
	    } // j

            /* FR2 */  
           
            for (j=0; j<Nuo[spin]; j++){
              for (i=0; i<MatDim; i++){

                tmp = PA2[j*MatDim+i].r*co - PA2[j*MatDim+i].i*si;
                FR2[spin][l1+Ra][l2+Rb][l3+Rc][j*MatDim+i] += (double)T_k_op[kloop]*tmp;

  	      } // i
	    } // j

          } // l3
        } // l2
      } // l1 

    } // end of kloop

    /****************************************
             normalize FR1 and FR2
    ****************************************/

    for (l1=-Ra; l1<=Ra; l1++){
      for (l2=-Rb; l2<=Rb; l2++){
	for (l3=-Rc; l3<=Rc; l3++){

	  for (j=0; j<No[spin]; j++){
	    for (i=0; i<MatDim; i++){
	      FR1[spin][l1+Ra][l2+Rb][l3+Rc][j*MatDim+i] /= tweight;
	    } // i
	  } // j

	  for (j=0; j<Nuo[spin]; j++){
	    for (i=0; i<MatDim; i++){
	      FR2[spin][l1+Ra][l2+Rb][l3+Rc][j*MatDim+i] /= tweight;
	    } // i
	  } // j

	} // l3
      } // l2
    } // l1

  } // spin

  /*
  for (l1=-Ra; l1<=Ra; l1++){
    printf("%2d %15.12f\n",l1,FR1[0][l1+Ra][0][0][0]);
  }

  MPI_Finalize();
  exit(0);
  */

  /*************************************************************
                      freeing of arrays
  *************************************************************/

  free(S0);
  free(PA1);
  free(PA2);
  free(D);
  free(Hk);
  free(Sk);
  free(WF);
  free(U1);
  free(U2);
  free(eval);
}




void Calc_Afull( int spin, 
                 double k1, double k2, double k3, 
		 dcomplex *WF, double *eval,  
		 double **PF1, double **PF2, 
		 dcomplex *Afull )
{
  int i,j,k,m,n,xyz_i,l1,l2,l3,Rn,p,Ra,Rb,Rc;
  int lm,ln,gm,po,lwork,info,GA_AN,LB_AN,GB_AN,tnoA,tnoB,Anum,Bnum,*ipiv;
  double si,co,kRn,tmp1,tmp2,d,en,em,x,y,z,s,sx,sy,sz,Rx,Ry,Rz;  
  double del=1.0e-20;
  dcomplex ctmp1,ctmp2,csum,*Hk,*Sk,**dHk,**dSk,*B11,*B22,*Fk,**dFk;
  dcomplex alpha = {1.0,0.0}; 
  dcomplex beta = {0.0,0.0};

  /*************************************************************
                      allocation of arrays
  *************************************************************/

  Hk = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));
  Sk = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));
  Fk = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));

  dFk = (dcomplex**)malloc(sizeof(dcomplex*)*3);
  for(i=0; i<3; i++){
    dFk[i] = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));
  }

  dHk = (dcomplex**)malloc(sizeof(dcomplex*)*3);
  for (i=0; i<3; i++){
    dHk[i] = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));
  }

  dSk = (dcomplex**)malloc(sizeof(dcomplex*)*3);
  for (i=0; i<3; i++){
    dSk[i] = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));
  }

  B11 = (dcomplex*)malloc(sizeof(dcomplex)*No[spin]*No[spin]);
  B22 = (dcomplex*)malloc(sizeof(dcomplex)*Nuo[spin]*Nuo[spin]);

  /* initialize Afull */

  for (i=0; i<MatDim*MatDim*3; i++){
    Afull[i] = Complex(0.0,0.0);  
  }

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

	      /* store the result to Afull */

	      Afull[Aref(0,m,n)].r += ctmp2.r*(sx - Rx*s);
	      Afull[Aref(1,m,n)].r += ctmp2.r*(sy - Ry*s);
	      Afull[Aref(2,m,n)].r += ctmp2.r*(sz - Rz*s);

	      Afull[Aref(0,m,n)].i += ctmp2.i*(sx - Rx*s);
	      Afull[Aref(1,m,n)].i += ctmp2.i*(sy - Ry*s);
	      Afull[Aref(2,m,n)].i += ctmp2.i*(sz - Rz*s);

	    } // j
	  } // i
	} // LB_AN
      } // GA_AN
    } // m  
  } // m  

  /************************************************************/
  /*************************************************************
     The 2nd contribution: A12 and A21
     calculation of analytic Berry connection
  *************************************************************/
  /************************************************************/

  /* calculate dHk/dk */    

  Calc_dHk_dSk(dHk,dSk,WF,spin,k1,k2,k3);  

  /* loop for x, y, and z */

  for (xyz_i=0; xyz_i<3; xyz_i++){   

    /* construct A21[1]_{x, y, or z} */

    for (m=0; m<Nuo[spin]; m++){ // for '2'

      lm = No[spin] + m;
      em = eval[lm];

      for (n=0; n<No[spin]; n++){ // for '1'

	en = eval[n];
	d = en - em;
	tmp1 = d/(d*d + del);

        ctmp1.r = (dHk[xyz_i][n*MatDim+lm].r-en*dSk[xyz_i][n*MatDim+lm].r)*tmp1;  
        ctmp1.i = (dHk[xyz_i][n*MatDim+lm].i-en*dSk[xyz_i][n*MatDim+lm].i)*tmp1;

        /* multiplying the imaginary number, i */
        Afull[Aref(xyz_i,m+No[spin],n)].r += (-ctmp1.i);
        Afull[Aref(xyz_i,m+No[spin],n)].i += ctmp1.r;

      } // n
    } // m

    /* construct A12[1]_{x, y, or z} */

    for (m=0; m<No[spin]; m++){ // for '1'

      em = eval[m];

      for (n=0; n<Nuo[spin]; n++){ // for '2'

        ln = No[spin] + n;
	en = eval[ln];

	d = en - em;
	tmp1 = d/(d*d + del);

        ctmp1.r = (dHk[xyz_i][ln*MatDim+m].r-en*dSk[xyz_i][ln*MatDim+m].r)*tmp1;
        ctmp1.i = (dHk[xyz_i][ln*MatDim+m].i-en*dSk[xyz_i][ln*MatDim+m].i)*tmp1; 

        /* multiplying the imaginary number, i */
        Afull[Aref(xyz_i,m,n+No[spin])].r += (-ctmp1.i);
        Afull[Aref(xyz_i,m,n+No[spin])].i += ctmp1.r;

      } // n
    } // m

  } // xyz_i

  /************************************************************/
  /*************************************************************
     The 2nd contribution: A11
     calculation of analytic Berry connection
  *************************************************************/
  /************************************************************/

  /* initialize Fk and dFk */

  for (i=0; i<MatDim*MatDim; i++){
    Fk[i] = Complex(0.0,0.0);
    dFk[0][i] = Complex(0.0,0.0);
    dFk[1][i] = Complex(0.0,0.0);
    dFk[2][i] = Complex(0.0,0.0);
  }

  /* calculate Fk, dFk/dkx, dFk/dky, and dFk/dkz */

  Ra = (int)Nk1/2;
  Rb = (int)Nk2/2;
  Rc = (int)Nk3/2;

  for (l1=-Ra; l1<=Ra; l1++){
    for (l2=-Rb; l2<=Rb; l2++){
      for (l3=-Rc; l3<=Rc; l3++){

	kRn = k1*(double)l1 + k2*(double)l2 + k3*(double)l3;
	si = sin(-2.0*PI*kRn);
	co = cos(-2.0*PI*kRn);

	Rx = (double)l1*tv[1][1] + (double)l2*tv[2][1] + (double)l3*tv[3][1];
	Ry = (double)l1*tv[1][2] + (double)l2*tv[2][2] + (double)l3*tv[3][2];
	Rz = (double)l1*tv[1][3] + (double)l2*tv[2][3] + (double)l3*tv[3][3];

        for (j=0; j<No[spin]; j++){ 
          for (i=0; i<MatDim; i++){ 

            tmp1 = FR1[spin][l1+Ra][l2+Rb][l3+Rc][j*MatDim+i]*co;
            tmp2 = FR1[spin][l1+Ra][l2+Rb][l3+Rc][j*MatDim+i]*si; 

	    Fk[j*MatDim+i].r += tmp1;
	    Fk[j*MatDim+i].i += tmp2;

	    dFk[0][j*MatDim+i].r += Rx*tmp2;
	    dFk[0][j*MatDim+i].i -= Rx*tmp1;

	    dFk[1][j*MatDim+i].r += Ry*tmp2;
	    dFk[1][j*MatDim+i].i -= Ry*tmp1;

	    dFk[2][j*MatDim+i].r += Rz*tmp2;
	    dFk[2][j*MatDim+i].i -= Rz*tmp1;

	  }
	}
      }
    }
  }    

  /* calculate B11 = Fk^dag S C1 */

  Calc_Hk_Sk(Hk,Sk,spin,k1,k2,k3);

  m = MatDim; n = No[spin]; k = MatDim;
  zgemm_("N","N", &m, &n, &k, &alpha, Sk, &m, &WF[0], &k, &beta, Hk, &m);
  for (i=0; i<MatDim*No[spin]; i++){ Sk[i] = Hk[i]; }
    
  m = No[spin]; n = No[spin]; k = MatDim;
  zgemm_("C","N", &m, &n, &k, &alpha, Fk, &k, Sk, &k, &beta, B11, &m);

  /* calculate the inverse of B11 */

  //Calc_Inverse_SVD(No[spin], B11); // inverse B11 is stored in B11 
  Calc_Inverse(No[spin], B11); // inverse B11 is stored in B11 

  /* calculate B11^-1 * (Fk^dag)^(a) * (S C1) */

  for (p=0; p<3; p++){
  
    m = No[spin]; n = MatDim; k = No[spin];
    zgemm_("N","C", &m, &n, &k, &alpha, B11, &m, dFk[p], &n, &beta, Hk, &m);

    m = No[spin]; n = No[spin]; k = MatDim;
    zgemm_("N","N", &m, &n, &k, &alpha, Hk, &m, Sk, &k, &beta, Fk, &m);

    /* multiplying (-i) */

    for (i=0; i<No[spin]; i++){ 
      for (j=0; j<No[spin]; j++){ 

	ctmp1 = Fk[j*No[spin]+i];

	Afull[Aref(p,i,j)].r += ctmp1.i; 
	Afull[Aref(p,i,j)].i += (-ctmp1.r);
      }
    }
  }

  /************************************************************/
  /*************************************************************
     The 2nd contribution: A22
     calculation of analytic Berry connection
  *************************************************************/
  /************************************************************/

  /* initialize Fk and dFk */

  for (i=0; i<MatDim*MatDim; i++){
    Fk[i] = Complex(0.0,0.0);
    dFk[0][i] = Complex(0.0,0.0);
    dFk[1][i] = Complex(0.0,0.0);
    dFk[2][i] = Complex(0.0,0.0);
  }

  /* calculate Fk, dFk/dkx, dFk/dky, and dFk/dkz */

  Ra = (int)Nk1/2;
  Rb = (int)Nk2/2;
  Rc = (int)Nk3/2;

  for (l1=-Ra; l1<=Ra; l1++){
    for (l2=-Rb; l2<=Rb; l2++){
      for (l3=-Rc; l3<=Rc; l3++){

	kRn = k1*(double)l1 + k2*(double)l2 + k3*(double)l3;
	si = sin(-2.0*PI*kRn);
	co = cos(-2.0*PI*kRn);

	Rx = (double)l1*tv[1][1] + (double)l2*tv[2][1] + (double)l3*tv[3][1];
	Ry = (double)l1*tv[1][2] + (double)l2*tv[2][2] + (double)l3*tv[3][2];
	Rz = (double)l1*tv[1][3] + (double)l2*tv[2][3] + (double)l3*tv[3][3];

        for (j=0; j<Nuo[spin]; j++){ 
          for (i=0; i<MatDim; i++){ 

            tmp1 = FR2[spin][l1+Ra][l2+Rb][l3+Rc][j*MatDim+i]*co;
            tmp2 = FR2[spin][l1+Ra][l2+Rb][l3+Rc][j*MatDim+i]*si; 

	    Fk[j*MatDim+i].r += tmp1;
	    Fk[j*MatDim+i].i += tmp2;

	    dFk[0][j*MatDim+i].r += Rx*tmp2;
	    dFk[0][j*MatDim+i].i -= Rx*tmp1;

	    dFk[1][j*MatDim+i].r += Ry*tmp2;
	    dFk[1][j*MatDim+i].i -= Ry*tmp1;

	    dFk[2][j*MatDim+i].r += Rz*tmp2;
	    dFk[2][j*MatDim+i].i -= Rz*tmp1;

	  }
	}
      }
    }
  }    

  /* calculate B22 = Fk^dag S C2 */

  Calc_Hk_Sk(Hk,Sk,spin,k1,k2,k3);

  m = MatDim; n = Nuo[spin]; k = MatDim;
  zgemm_("N","N", &m, &n, &k, &alpha, Sk, &m, &WF[No[spin]*MatDim], &k, &beta, Hk, &m);
  for (i=0; i<MatDim*Nuo[spin]; i++){ Sk[i] = Hk[i]; }
    
  m = Nuo[spin]; n = Nuo[spin]; k = MatDim;
  zgemm_("C","N", &m, &n, &k, &alpha, Fk, &k, Sk, &k, &beta, B22, &m);

  /* calculate the inverse B22 */

  //Calc_Inverse_SVD(Nuo[spin], B22); // inverse B22 is stored in B22 
  Calc_Inverse(Nuo[spin], B22); // inverse B22 is stored in B22 

  /* calculate B22^-1 * (Fk^dag)^(a) * (S C2) */

  for (p=0; p<3; p++){
  
    m = Nuo[spin]; n = MatDim; k = Nuo[spin];
    zgemm_("N","C", &m, &n, &k, &alpha, B22, &m, dFk[p], &n, &beta, Hk, &m);

    m = Nuo[spin]; n = Nuo[spin]; k = MatDim;
    zgemm_("N","N", &m, &n, &k, &alpha, Hk, &m, Sk, &k, &beta, Fk, &m);

    /* multiplying (-i) */

    for (i=0; i<Nuo[spin]; i++){ 
      for (j=0; j<Nuo[spin]; j++){ 

	ctmp1 = Fk[j*Nuo[spin]+i];

	Afull[Aref(p,i+No[spin],j+No[spin])].r += ctmp1.i; 
	Afull[Aref(p,i+No[spin],j+No[spin])].i += (-ctmp1.r);
      }
    }
  }

  /*************************************************************
                       freeing of arrays
  *************************************************************/

  free(Hk);
  free(Sk);
  free(Fk);

  for (i=0; i<3; i++){
    free(dFk[i]);
  }
  free(dFk);

  for (i=0; i<3; i++){
    free(dHk[i]);
  }
  free(dHk);

  for (i=0; i<3; i++){
    free(dSk[i]);
  }
  free(dSk);

  free(B11);
  free(B22);
}




void Calc_A11_A12_A21_A22( int spin, double k1, double k2, double k3, 
                           dcomplex *WF, double *eval,  
                           double **PF1, double **PF2, 
                           dcomplex ***A11, dcomplex ***A12, 
                           dcomplex ***A21, dcomplex ***A22 ) 
{
  int i,j,k,m,n,xyz_i,l1,l2,l3,Rn,p,Ra,Rb,Rc;
  int lm,ln,gm,po,lwork,info,GA_AN,LB_AN,GB_AN,tnoA,tnoB,Anum,Bnum,*ipiv;
  double si,co,kRn,tmp1,tmp2,d,en,em,x,y,z,s,sx,sy,sz,Rx,Ry,Rz;  
  double del=1.0e-20;
  dcomplex ctmp1,ctmp2,csum,*Hk,*Sk,**dHk,**dSk,*B11,*B22,*Fk,**dFk;
  dcomplex alpha = {1.0,0.0}; 
  dcomplex beta = {0.0,0.0};

  /*************************************************************
                      allocation of arrays
  *************************************************************/

  Hk = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));
  Sk = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));
  Fk = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));

  dFk = (dcomplex**)malloc(sizeof(dcomplex*)*3);
  for(i=0; i<3; i++){
    dFk[i] = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));
  }

  dHk = (dcomplex**)malloc(sizeof(dcomplex*)*3);
  for (i=0; i<3; i++){
    dHk[i] = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));
  }

  dSk = (dcomplex**)malloc(sizeof(dcomplex*)*3);
  for (i=0; i<3; i++){
    dSk[i] = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));
  }

  B11 = (dcomplex*)malloc(sizeof(dcomplex)*No[spin]*No[spin]);
  B22 = (dcomplex*)malloc(sizeof(dcomplex)*Nuo[spin]*Nuo[spin]);

  /* initialize A11, A12, A21, and A22 */

  for (m=0; m<2; m++){
    for (i=0; i<3; i++){
      for (j=0; j<No[spin]*No[spin]; j++){
	A11[m][i][j] = Complex(0.0,0.0);
      }
      for (j=0; j<No[spin]*Nuo[spin]; j++){
	A12[m][i][j] = Complex(0.0,0.0);
	A21[m][i][j] = Complex(0.0,0.0);
      }
      for (j=0; j<Nuo[spin]*Nuo[spin]; j++){
	A22[m][i][j] = Complex(0.0,0.0);
      }
    }
  }

  /************************************************************/
  /*************************************************************
     The 1st contribution: ABCxx[0]
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
                A11[0][0][n*No[spin]+m].r += ctmp2.r*(sx - Rx*s);
                A11[0][1][n*No[spin]+m].r += ctmp2.r*(sy - Ry*s);
                A11[0][2][n*No[spin]+m].r += ctmp2.r*(sz - Rz*s);

                A11[0][0][n*No[spin]+m].i += ctmp2.i*(sx - Rx*s);
                A11[0][1][n*No[spin]+m].i += ctmp2.i*(sy - Ry*s);
                A11[0][2][n*No[spin]+m].i += ctmp2.i*(sz - Rz*s);
	      }

              else if (m<No[spin] && !(n<No[spin])){
                A12[0][0][(n-No[spin])*No[spin]+m].r += ctmp2.r*(sx - Rx*s);
                A12[0][1][(n-No[spin])*No[spin]+m].r += ctmp2.r*(sy - Ry*s);
                A12[0][2][(n-No[spin])*No[spin]+m].r += ctmp2.r*(sz - Rz*s);

                A12[0][0][(n-No[spin])*No[spin]+m].i += ctmp2.i*(sx - Rx*s);
                A12[0][1][(n-No[spin])*No[spin]+m].i += ctmp2.i*(sy - Ry*s);
                A12[0][2][(n-No[spin])*No[spin]+m].i += ctmp2.i*(sz - Rz*s);
	      }

              else if (!(m<No[spin]) && n<No[spin]){
                A21[0][0][n*Nuo[spin]+m-No[spin]].r += ctmp2.r*(sx - Rx*s);
                A21[0][1][n*Nuo[spin]+m-No[spin]].r += ctmp2.r*(sy - Ry*s);
                A21[0][2][n*Nuo[spin]+m-No[spin]].r += ctmp2.r*(sz - Rz*s);

                A21[0][0][n*Nuo[spin]+m-No[spin]].i += ctmp2.i*(sx - Rx*s);
                A21[0][1][n*Nuo[spin]+m-No[spin]].i += ctmp2.i*(sy - Ry*s);
                A21[0][2][n*Nuo[spin]+m-No[spin]].i += ctmp2.i*(sz - Rz*s);
	      }

              else{
                A22[0][0][(n-No[spin])*Nuo[spin]+m-No[spin]].r += ctmp2.r*(sx - Rx*s);
                A22[0][1][(n-No[spin])*Nuo[spin]+m-No[spin]].r += ctmp2.r*(sy - Ry*s);
                A22[0][2][(n-No[spin])*Nuo[spin]+m-No[spin]].r += ctmp2.r*(sz - Rz*s);

                A22[0][0][(n-No[spin])*Nuo[spin]+m-No[spin]].i += ctmp2.i*(sx - Rx*s);
                A22[0][1][(n-No[spin])*Nuo[spin]+m-No[spin]].i += ctmp2.i*(sy - Ry*s);
                A22[0][2][(n-No[spin])*Nuo[spin]+m-No[spin]].i += ctmp2.i*(sz - Rz*s);
	      }

	    } // j
	  } // i
	} // LB_AN
      } // GA_AN
    } // m  
  } // m  

  /************************************************************/
  /*************************************************************
     The 2nd contribution: A12[1] and A21[1]
     calculation of analytic Berry connection
  *************************************************************/
  /************************************************************/

  /* calculate dHk/dk */    

  Calc_dHk_dSk(dHk,dSk,WF,spin,k1,k2,k3);  

  /* loop for x, y, and z */

  for (xyz_i=0; xyz_i<3; xyz_i++){   

    /* construct A21[1]_{x, y, or z} */

    for (m=0; m<Nuo[spin]; m++){ // for '2'

      lm = No[spin] + m;
      em = eval[lm];

      for (n=0; n<No[spin]; n++){ // for '1'

	en = eval[n];
	d = en - em;
	tmp1 = d/(d*d + del);

        ctmp1.r = (dHk[xyz_i][n*MatDim+lm].r-en*dSk[xyz_i][n*MatDim+lm].r)*tmp1;  
        ctmp1.i = (dHk[xyz_i][n*MatDim+lm].i-en*dSk[xyz_i][n*MatDim+lm].i)*tmp1;

        /* multiplying the imaginary number, i */
	A21[1][xyz_i][n*Nuo[spin]+m].r =-ctmp1.i; 
	A21[1][xyz_i][n*Nuo[spin]+m].i = ctmp1.r; 

      } // n
    } // m

    /* construct A12[1]_{x, y, or z} */

    for (m=0; m<No[spin]; m++){ // for '1'

      em = eval[m];

      for (n=0; n<Nuo[spin]; n++){ // for '2'

        ln = No[spin] + n;
	en = eval[ln];

	d = en - em;
	tmp1 = d/(d*d + del);

        ctmp1.r = (dHk[xyz_i][ln*MatDim+m].r-en*dSk[xyz_i][ln*MatDim+m].r)*tmp1;
        ctmp1.i = (dHk[xyz_i][ln*MatDim+m].i-en*dSk[xyz_i][ln*MatDim+m].i)*tmp1; 

        /* multiplying the imaginary number, i */
	A12[1][xyz_i][n*No[spin]+m].r =-ctmp1.i; 
	A12[1][xyz_i][n*No[spin]+m].i = ctmp1.r;

      } // n
    } // m

  } // xyz_i

  /************************************************************/
  /*************************************************************
     The 2nd contribution: A11[1]
     calculation of analytic Berry connection
  *************************************************************/
  /************************************************************/

  /* initialize Fk and dFk */

  for (i=0; i<MatDim*MatDim; i++){
    Fk[i] = Complex(0.0,0.0);
    dFk[0][i] = Complex(0.0,0.0);
    dFk[1][i] = Complex(0.0,0.0);
    dFk[2][i] = Complex(0.0,0.0);
  }

  /* calculate Fk, dFk/dkx, dFk/dky, and dFk/dkz */

  Ra = (int)Nk1/2;
  Rb = (int)Nk2/2;
  Rc = (int)Nk3/2;

  for (l1=-Ra; l1<=Ra; l1++){
    for (l2=-Rb; l2<=Rb; l2++){
      for (l3=-Rc; l3<=Rc; l3++){

	kRn = k1*(double)l1 + k2*(double)l2 + k3*(double)l3;
	si = sin(-2.0*PI*kRn);
	co = cos(-2.0*PI*kRn);

	Rx = (double)l1*tv[1][1] + (double)l2*tv[2][1] + (double)l3*tv[3][1];
	Ry = (double)l1*tv[1][2] + (double)l2*tv[2][2] + (double)l3*tv[3][2];
	Rz = (double)l1*tv[1][3] + (double)l2*tv[2][3] + (double)l3*tv[3][3];

        for (j=0; j<No[spin]; j++){ 
          for (i=0; i<MatDim; i++){ 

            tmp1 = FR1[spin][l1+Ra][l2+Rb][l3+Rc][j*MatDim+i]*co;
            tmp2 = FR1[spin][l1+Ra][l2+Rb][l3+Rc][j*MatDim+i]*si; 

	    Fk[j*MatDim+i].r += tmp1;
	    Fk[j*MatDim+i].i += tmp2;

	    dFk[0][j*MatDim+i].r += Rx*tmp2;
	    dFk[0][j*MatDim+i].i -= Rx*tmp1;

	    dFk[1][j*MatDim+i].r += Ry*tmp2;
	    dFk[1][j*MatDim+i].i -= Ry*tmp1;

	    dFk[2][j*MatDim+i].r += Rz*tmp2;
	    dFk[2][j*MatDim+i].i -= Rz*tmp1;

	  }
	}
      }
    }
  }    

  /* calculate B11 = Fk^dag S C1 */

  Calc_Hk_Sk(Hk,Sk,spin,k1,k2,k3);

  m = MatDim; n = No[spin]; k = MatDim;
  zgemm_("N","N", &m, &n, &k, &alpha, Sk, &m, &WF[0], &k, &beta, Hk, &m);
  for (i=0; i<MatDim*No[spin]; i++){ Sk[i] = Hk[i]; }
    
  m = No[spin]; n = No[spin]; k = MatDim;
  zgemm_("C","N", &m, &n, &k, &alpha, Fk, &k, Sk, &k, &beta, B11, &m);

  /* calculate the inverse of B11 */

  //Calc_Inverse_SVD(No[spin], B11); // inverse B11 is stored in B11 
  Calc_Inverse(No[spin], B11); // inverse B11 is stored in B11 

  /* calculate B11^-1 * (Fk^dag)^(a) * (S C1) */

  for (p=0; p<3; p++){
  
    m = No[spin]; n = MatDim; k = No[spin];
    zgemm_("N","C", &m, &n, &k, &alpha, B11, &m, dFk[p], &n, &beta, Hk, &m);

    m = No[spin]; n = No[spin]; k = MatDim;
    zgemm_("N","N", &m, &n, &k, &alpha, Hk, &m, Sk, &k, &beta, A11[1][p], &m);

    /* multiplying (-i) */
    for (i=0; i<No[spin]*No[spin]; i++){ 
      ctmp1 = A11[1][p][i];
      A11[1][p][i].r = ctmp1.i;
      A11[1][p][i].i =-ctmp1.r;
    }
  }

  /************************************************************/
  /*************************************************************
     The 2nd contribution: A22[1]
     calculation of analytic Berry connection
  *************************************************************/
  /************************************************************/

  /* initialize Fk and dFk */

  for (i=0; i<MatDim*MatDim; i++){
    Fk[i] = Complex(0.0,0.0);
    dFk[0][i] = Complex(0.0,0.0);
    dFk[1][i] = Complex(0.0,0.0);
    dFk[2][i] = Complex(0.0,0.0);
  }

  /* calculate Fk, dFk/dkx, dFk/dky, and dFk/dkz */

  Ra = (int)Nk1/2;
  Rb = (int)Nk2/2;
  Rc = (int)Nk3/2;

  for (l1=-Ra; l1<=Ra; l1++){
    for (l2=-Rb; l2<=Rb; l2++){
      for (l3=-Rc; l3<=Rc; l3++){

	kRn = k1*(double)l1 + k2*(double)l2 + k3*(double)l3;
	si = sin(-2.0*PI*kRn);
	co = cos(-2.0*PI*kRn);

	Rx = (double)l1*tv[1][1] + (double)l2*tv[2][1] + (double)l3*tv[3][1];
	Ry = (double)l1*tv[1][2] + (double)l2*tv[2][2] + (double)l3*tv[3][2];
	Rz = (double)l1*tv[1][3] + (double)l2*tv[2][3] + (double)l3*tv[3][3];

        for (j=0; j<Nuo[spin]; j++){ 
          for (i=0; i<MatDim; i++){ 

            tmp1 = FR2[spin][l1+Ra][l2+Rb][l3+Rc][j*MatDim+i]*co;
            tmp2 = FR2[spin][l1+Ra][l2+Rb][l3+Rc][j*MatDim+i]*si; 

	    Fk[j*MatDim+i].r += tmp1;
	    Fk[j*MatDim+i].i += tmp2;

	    dFk[0][j*MatDim+i].r += Rx*tmp2;
	    dFk[0][j*MatDim+i].i -= Rx*tmp1;

	    dFk[1][j*MatDim+i].r += Ry*tmp2;
	    dFk[1][j*MatDim+i].i -= Ry*tmp1;

	    dFk[2][j*MatDim+i].r += Rz*tmp2;
	    dFk[2][j*MatDim+i].i -= Rz*tmp1;

	  }
	}
      }
    }
  }    

  /* calculate B22 = Fk^dag S C2 */

  Calc_Hk_Sk(Hk,Sk,spin,k1,k2,k3);

  m = MatDim; n = Nuo[spin]; k = MatDim;
  zgemm_("N","N", &m, &n, &k, &alpha, Sk, &m, &WF[No[spin]*MatDim], &k, &beta, Hk, &m);
  for (i=0; i<MatDim*Nuo[spin]; i++){ Sk[i] = Hk[i]; }
    
  m = Nuo[spin]; n = Nuo[spin]; k = MatDim;
  zgemm_("C","N", &m, &n, &k, &alpha, Fk, &k, Sk, &k, &beta, B22, &m);

  /* calculate the inverse B22 */

  //Calc_Inverse_SVD(Nuo[spin], B22); // inverse B22 is stored in B22 
  Calc_Inverse(Nuo[spin], B22); // inverse B22 is stored in B22 

  /* calculate B22^-1 * (Fk^dag)^(a) * (S C2) */

  for (p=0; p<3; p++){
  
    m = Nuo[spin]; n = MatDim; k = Nuo[spin];
    zgemm_("N","C", &m, &n, &k, &alpha, B22, &m, dFk[p], &n, &beta, Hk, &m);

    m = Nuo[spin]; n = Nuo[spin]; k = MatDim;
    zgemm_("N","N", &m, &n, &k, &alpha, Hk, &m, Sk, &k, &beta, A22[1][p], &m);

    /* multiplying (-i) */
    for (i=0; i<Nuo[spin]*Nuo[spin]; i++){ 
      ctmp1 = A22[1][p][i];
      A22[1][p][i].r = ctmp1.i;
      A22[1][p][i].i =-ctmp1.r;
    }
  }

  /*************************************************************
                       freeing of arrays
  *************************************************************/

  free(Hk);
  free(Sk);
  free(Fk);

  for (i=0; i<3; i++){
    free(dFk[i]);
  }
  free(dFk);

  for (i=0; i<3; i++){
    free(dHk[i]);
  }
  free(dHk);

  for (i=0; i<3; i++){
    free(dSk[i]);
  }
  free(dSk);

  free(B11);
  free(B22);
}


void Calc_Momentum_Matrix( int spin, double k1, double k2, double k3, 
                           dcomplex *WF, double *eval, dcomplex **P )
  /******************************************************************
    The routine calculates the momentum matrix elements represented 
    by eigenstates using a direct derivative calculation.
  ******************************************************************/
{
  int i,j,k,m,n,xyz_i,l1,l2,l3,Rn;
  int GA_AN,LB_AN,GB_AN,tnoA,tnoB,Anum,Bnum;
  double si,co,kRn,d,en,em,x,y,z,s,sx,sy,sz;
  dcomplex ctmp1,ctmp2,ctmp3,csum,**dHk,**dSk;
  dcomplex p0,p1,p2;

  /* initialize P */

  for (i=0; i<MatDim*MatDim; i++){
    P[0][i] = Complex(0.0,0.0);       
    P[1][i] = Complex(0.0,0.0);       
    P[2][i] = Complex(0.0,0.0);       
  }

  /************************************************************/
  /*************************************************************
         calculation of <psi_m | px, py, pz | psi_n>
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

	  kRn = k1*(double)l1 + k2*(double)l2 + k3*(double)l3;
	  si = sin(2.0*PI*kRn);
	  co = cos(2.0*PI*kRn);

	  for (i=0; i<tnoA; i++){
	    for (j=0; j<tnoB; j++){

	      sx = OLPmo[0][GA_AN][LB_AN][i][j];
	      sy = OLPmo[1][GA_AN][LB_AN][i][j];
	      sz = OLPmo[2][GA_AN][LB_AN][i][j];

	      ctmp1.r = WF[m*MatDim+(Anum+i)].r*WF[n*MatDim+(Bnum+j)].r 
		       +WF[m*MatDim+(Anum+i)].i*WF[n*MatDim+(Bnum+j)].i;  

	      ctmp1.i = WF[m*MatDim+(Anum+i)].r*WF[n*MatDim+(Bnum+j)].i 
		       -WF[m*MatDim+(Anum+i)].i*WF[n*MatDim+(Bnum+j)].r;

	      ctmp2.r = ctmp1.r*co - ctmp1.i*si;
	      ctmp2.i = ctmp1.r*si + ctmp1.i*co;

	      P[0][n*MatDim+m].r += ctmp2.r*sx;
	      P[1][n*MatDim+m].r += ctmp2.r*sy;
	      P[2][n*MatDim+m].r += ctmp2.r*sz;

	      P[0][n*MatDim+m].i += ctmp2.i*sx;
	      P[1][n*MatDim+m].i += ctmp2.i*sy;
	      P[2][n*MatDim+m].i += ctmp2.i*sz;

	    } // j
	  } // i
	} // LB_AN
      } // GA_AN

      /* multiplying (-i). Then, P becomes <psi_mu | -i nabla | psi_nu> */

      p0 = P[0][n*MatDim+m];
      p1 = P[1][n*MatDim+m];
      p2 = P[2][n*MatDim+m];

      P[0][n*MatDim+m].r = p0.i;
      P[1][n*MatDim+m].r = p1.i;
      P[2][n*MatDim+m].r = p2.i;

      P[0][n*MatDim+m].i =-p0.r;
      P[1][n*MatDim+m].i =-p1.r;
      P[2][n*MatDim+m].i =-p2.r;

    } // n
  } // m

  /* preserving the Hermicity  */

  for (i=0; i<3; i++){
    for (m=0; m<MatDim; m++){
      for (n=m; n<MatDim; n++){

        ctmp1 = P[i][n*MatDim+m];
        ctmp2 = P[i][m*MatDim+n];
        ctmp3.r = 0.5*(ctmp1.r + ctmp2.r);
        ctmp3.i = 0.5*(ctmp1.i - ctmp2.i);

        P[i][n*MatDim+m] = ctmp3;
        P[i][m*MatDim+n].r = ctmp3.r;
        P[i][m*MatDim+n].i =-ctmp3.i;
      }
    }   
  }

}



void Calc_Momentum_Matrix_CCLee( int spin, double k1, double k2, double k3, 
                                 dcomplex *WF, double *eval, dcomplex **P )
  /*************************************************************
    The routine calculates the momentum matrix elements 
    represented by eigenstates using a method: 
    C.C. Lee, Y.T. Lee, M. Fukuda, and T. Ozaki, 
    PRB 98, 115115 (2018).   
  *************************************************************/
{

  int i,j,k,m,n,xyz_i,l1,l2,l3,Rn;
  int GA_AN,LB_AN,GB_AN,tnoA,tnoB,Anum,Bnum;
  double si,co,kRn,d,en,em,x,y,z,s,sx,sy,sz;
  dcomplex ctmp1,ctmp2,csum,**dHk,**dSk;
  dcomplex p0,p1,p2;

  /*************************************************************
                      allocation of arrays
  *************************************************************/

  dHk = (dcomplex**)malloc(sizeof(dcomplex*)*3);
  for (i=0; i<3; i++){
    dHk[i] = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));
  }

  dSk = (dcomplex**)malloc(sizeof(dcomplex*)*3);
  for (i=0; i<3; i++){
    dSk[i] = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));
  }

  /* initialize P */

  for (i=0; i<MatDim*MatDim; i++){
    P[0][i] = Complex(0.0,0.0);       
    P[1][i] = Complex(0.0,0.0);       
    P[2][i] = Complex(0.0,0.0);       
  }

  /************************************************************/
  /*************************************************************
     The 1st contribution:
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

	  kRn = k1*(double)l1 + k2*(double)l2 + k3*(double)l3;
	  si = sin(2.0*PI*kRn);
	  co = cos(2.0*PI*kRn);

	  for (i=0; i<tnoA; i++){
	    for (j=0; j<tnoB; j++){

	      sx = OLPpo[0][0][GA_AN][LB_AN][i][j];
	      sy = OLPpo[1][0][GA_AN][LB_AN][i][j];
	      sz = OLPpo[2][0][GA_AN][LB_AN][i][j];

	      ctmp1.r = WF[m*MatDim+(Anum+i)].r*WF[n*MatDim+(Bnum+j)].r 
		       +WF[m*MatDim+(Anum+i)].i*WF[n*MatDim+(Bnum+j)].i;  

	      ctmp1.i = WF[m*MatDim+(Anum+i)].r*WF[n*MatDim+(Bnum+j)].i 
		       -WF[m*MatDim+(Anum+i)].i*WF[n*MatDim+(Bnum+j)].r;

	      ctmp2.r = ctmp1.r*co - ctmp1.i*si;
	      ctmp2.i = ctmp1.r*si + ctmp1.i*co;

	      P[0][n*MatDim+m].r += ctmp2.r*sx;
	      P[1][n*MatDim+m].r += ctmp2.r*sy;
	      P[2][n*MatDim+m].r += ctmp2.r*sz;

	      P[0][n*MatDim+m].i += ctmp2.i*sx;
	      P[1][n*MatDim+m].i += ctmp2.i*sy;
	      P[2][n*MatDim+m].i += ctmp2.i*sz;

	    } // j
	  } // i
	} // LB_AN
      } // GA_AN

      /* multiplying i(ep_m - e_n) */

      p0 = P[0][n*MatDim+m];
      p1 = P[1][n*MatDim+m];
      p2 = P[1][n*MatDim+m];
      d = eval[m] - eval[n]; 

      P[0][n*MatDim+m].r =-d*p0.i;
      P[1][n*MatDim+m].r =-d*p1.i;
      P[2][n*MatDim+m].r =-d*p2.i;

      P[0][n*MatDim+m].i = d*p0.r;
      P[1][n*MatDim+m].i = d*p1.r;
      P[2][n*MatDim+m].i = d*p2.r;

    } // n
  } // m

  /************************************************************/
  /*************************************************************
     The 2nd contribution:
  *************************************************************/
  /************************************************************/

  /* calculate dHk/dk */    

  Calc_dHk_dSk(dHk,dSk,WF,spin,k1,k2,k3);  

  /* loop for x, y, and z */

  for (xyz_i=0; xyz_i<3; xyz_i++){   
    for (m=0; m<MatDim; m++){ 

      em = eval[m];

      for (n=0; n<MatDim; n++){

        ctmp1.r = dHk[xyz_i][n*MatDim+m].r - em*dSk[xyz_i][n*MatDim+m].r;
        ctmp1.i = dHk[xyz_i][n*MatDim+m].i - em*dSk[xyz_i][n*MatDim+m].i;

	P[xyz_i][n*MatDim+m].r += ctmp1.r;
	P[xyz_i][n*MatDim+m].i += ctmp1.i;

      } // n
    } // m
  } // xyz_i

  /*************************************************************
                       freeing of arrays
  *************************************************************/

  for (i=0; i<3; i++){
    free(dHk[i]);
  }
  free(dHk);

  for (i=0; i<3; i++){
    free(dSk[i]);
  }
  free(dSk);
}



void Calc_A12_A21( int spin, double k1, double k2, double k3, 
                   dcomplex *WF, double *eval, 
                   dcomplex ***A12, dcomplex ***A21 ) 
{
  int i,j,k,m,n,xyz_i,l1,l2,l3,Rn;
  int lm,ln,gm,po,lwork,info,GA_AN,LB_AN,GB_AN,tnoA,tnoB,Anum,Bnum;
  double si,co,kRn,tmp1,d,en,em,x,y,z,s,sx,sy,sz,Rx,Ry,Rz;  
  double del=1.0e-20;
  dcomplex ctmp1,ctmp2,csum,**dHk,**dSk;

  /*************************************************************
                      allocation of arrays
  *************************************************************/

  dHk = (dcomplex**)malloc(sizeof(dcomplex*)*3);
  for (i=0; i<3; i++){
    dHk[i] = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));
  }

  dSk = (dcomplex**)malloc(sizeof(dcomplex*)*3);
  for (i=0; i<3; i++){
    dSk[i] = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));
  }

  /* initialize A12 and A21 */

  for (m=0; m<2; m++){
    for (i=0; i<3; i++){
      for (j=0; j<No[spin]*Nuo[spin]; j++){
	A12[m][i][j] = Complex(0.0,0.0);
	A21[m][i][j] = Complex(0.0,0.0);
      }
    }
  }

  /************************************************************/
  /*************************************************************
     The 1st contribution: A12[0] and A21[0]
     calculation of analytic Berry connection
  *************************************************************/
  /************************************************************/

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

	  /* calculate A12 */

	  for (m=0; m<No[spin]; m++){
	    for (n=No[spin]; n<MatDim; n++){

	      ctmp1.r = WF[m*MatDim+(Anum+i)].r*WF[n*MatDim+(Bnum+j)].r 
		       +WF[m*MatDim+(Anum+i)].i*WF[n*MatDim+(Bnum+j)].i;  

	      ctmp1.i = WF[m*MatDim+(Anum+i)].r*WF[n*MatDim+(Bnum+j)].i 
		       -WF[m*MatDim+(Anum+i)].i*WF[n*MatDim+(Bnum+j)].r;

	      ctmp2.r = ctmp1.r*co - ctmp1.i*si;
	      ctmp2.i = ctmp1.r*si + ctmp1.i*co;

	      A12[0][0][(n-No[spin])*No[spin]+m].r += ctmp2.r*(sx - Rx*s); 
	      A12[0][1][(n-No[spin])*No[spin]+m].r += ctmp2.r*(sy - Ry*s); 
	      A12[0][2][(n-No[spin])*No[spin]+m].r += ctmp2.r*(sz - Rz*s); 

	      A12[0][0][(n-No[spin])*No[spin]+m].i += ctmp2.i*(sx - Rx*s); 
	      A12[0][1][(n-No[spin])*No[spin]+m].i += ctmp2.i*(sy - Ry*s); 
	      A12[0][2][(n-No[spin])*No[spin]+m].i += ctmp2.i*(sz - Rz*s); 
	    }
	  }

	  /* calculate A21 */

	  for (m=No[spin]; m<MatDim; m++){
	    for (n=0; n<No[spin]; n++){

	      ctmp1.r = WF[m*MatDim+(Anum+i)].r*WF[n*MatDim+(Bnum+j)].r 
	  	       +WF[m*MatDim+(Anum+i)].i*WF[n*MatDim+(Bnum+j)].i;  

	      ctmp1.i = WF[m*MatDim+(Anum+i)].r*WF[n*MatDim+(Bnum+j)].i 
		       -WF[m*MatDim+(Anum+i)].i*WF[n*MatDim+(Bnum+j)].r;

	      ctmp2.r = ctmp1.r*co - ctmp1.i*si;
	      ctmp2.i = ctmp1.r*si + ctmp1.i*co;

	      A21[0][0][n*Nuo[spin]+m-No[spin]].r += ctmp2.r*(sx - Rx*s); 
	      A21[0][1][n*Nuo[spin]+m-No[spin]].r += ctmp2.r*(sy - Ry*s); 
	      A21[0][2][n*Nuo[spin]+m-No[spin]].r += ctmp2.r*(sz - Rz*s); 

	      A21[0][0][n*Nuo[spin]+m-No[spin]].i += ctmp2.i*(sx - Rx*s); 
	      A21[0][1][n*Nuo[spin]+m-No[spin]].i += ctmp2.i*(sy - Ry*s); 
	      A21[0][2][n*Nuo[spin]+m-No[spin]].i += ctmp2.i*(sz - Rz*s); 
	    }
	  }

	} // j
      } // i
    } // LB_AN
  } // GA_AN

  /************************************************************/
  /*************************************************************
     The 2nd contribution: A12[1] and A21[1]
     calculation of analytic Berry connection
  *************************************************************/
  /************************************************************/

  /* calculate dHk/dk */    

  Calc_dHk_dSk(dHk,dSk,WF,spin,k1,k2,k3);  

  /* loop for x, y, and z */

  for (xyz_i=0; xyz_i<3; xyz_i++){   

    /* construct A21[1]_{x, y, or z} */

    for (m=0; m<Nuo[spin]; m++){ // for '2'

      lm = No[spin] + m;
      em = eval[lm];

      for (n=0; n<No[spin]; n++){ // for '1'

	en = eval[n];
	d = en - em;
	tmp1 = d/(d*d + del);

        ctmp1.r = (dHk[xyz_i][n*MatDim+lm].r-en*dSk[xyz_i][n*MatDim+lm].r)*tmp1;  
        ctmp1.i = (dHk[xyz_i][n*MatDim+lm].i-en*dSk[xyz_i][n*MatDim+lm].i)*tmp1;

        /* multiplying the imaginary number, i */
	A21[1][xyz_i][n*Nuo[spin]+m].r =-ctmp1.i; 
	A21[1][xyz_i][n*Nuo[spin]+m].i = ctmp1.r; 

      } // n
    } // m

    /* construct A12[1]_{x, y, or z} */

    for (m=0; m<No[spin]; m++){ // for '1'

      em = eval[m];

      for (n=0; n<Nuo[spin]; n++){ // for '2'

        ln = No[spin] + n;
	en = eval[ln];

	d = en - em;
	tmp1 = d/(d*d + del);

        ctmp1.r = (dHk[xyz_i][ln*MatDim+m].r-en*dSk[xyz_i][ln*MatDim+m].r)*tmp1;
        ctmp1.i = (dHk[xyz_i][ln*MatDim+m].i-en*dSk[xyz_i][ln*MatDim+m].i)*tmp1; 

        /* multiplying the imaginary number, i */
	A12[1][xyz_i][n*No[spin]+m].r =-ctmp1.i; 
	A12[1][xyz_i][n*No[spin]+m].i = ctmp1.r;

      } // n
    } // m

  } // xyz_i

  /*************************************************************
                       freeing of arrays
  *************************************************************/

  for (i=0; i<3; i++){
    free(dHk[i]);
  }
  free(dHk);

  for (i=0; i<3; i++){
    free(dSk[i]);
  }
  free(dSk);
}





void Calc_A11( int spin, double k1, double k2, double k3,
               dcomplex *WF, double *eval,                 
               double **PF1, dcomplex ***A11 )
{
  int i,j,k,m,n,xyz_i,l1,l2,l3,Rn,p;
  int lm,ln,gm,po,lwork,info,GA_AN,LB_AN,GB_AN,tnoA,tnoB,Anum,Bnum;
  double si,co,kRn,tmp1,tmp2,d,en,em,x,y,z,s,sx,sy,sz,Rx,Ry,Rz;  
  dcomplex ctmp1,ctmp2,csum,*Hk,*Sk;
  dcomplex *B11,*Fk1,**dFk1;
  dcomplex alpha = {1.0,0.0}; 
  dcomplex beta = {0.0,0.0};

  /*************************************************************
                      allocation of arrays
  *************************************************************/

  Hk = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));
  Sk = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));
  Fk1 = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));

  dFk1 = (dcomplex**)malloc(sizeof(dcomplex*)*3);
  for (i=0; i<3; i++){
    dFk1[i] = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));
  }

  for (i=0; i<MatDim*MatDim; i++){
    Fk1[i] = Complex(0.0,0.0);
    dFk1[0][i] = Complex(0.0,0.0);
    dFk1[1][i] = Complex(0.0,0.0);
    dFk1[2][i] = Complex(0.0,0.0);
  }

  B11 = (dcomplex*)malloc(sizeof(dcomplex)*No[spin]*No[spin]);

  /* initialize A11 */

  for (m=0; m<2; m++){
    for (i=0; i<3; i++){
      for (j=0; j<No[spin]*No[spin]; j++){
	A11[m][i][j] = Complex(0.0,0.0);
      }
    }
  }

  /************************************************************/
  /*************************************************************
     The 1st contribution: A11[0]
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

	      A11[0][0][n*No[spin]+m].r += ctmp2.r*sx;
	      A11[0][1][n*No[spin]+m].r += ctmp2.r*sy;
	      A11[0][2][n*No[spin]+m].r += ctmp2.r*sz;

	      A11[0][0][n*No[spin]+m].i += ctmp2.i*sx;
	      A11[0][1][n*No[spin]+m].i += ctmp2.i*sy;
	      A11[0][2][n*No[spin]+m].i += ctmp2.i*sz;

	    } // j
	  } // i
	} // LB_AN
      } // GA_AN
    } // m  
  } // m  

  /************************************************************/
  /*************************************************************
     The 2nd contribution: A11[1]
     calculation of analytic Berry connection
  *************************************************************/
  /************************************************************/

  /* calculate Fk1, dFk1/dkx, dFk1/dky, and dFk1/dkz */

  int Ra,Rb,Rc;

  Ra = (int)Nk1/2;
  Rb = (int)Nk2/2;
  Rc = (int)Nk3/2;

  for (l1=-Ra; l1<=Ra; l1++){
    for (l2=-Rb; l2<=Rb; l2++){
      for (l3=-Rc; l3<=Rc; l3++){

	kRn = k1*(double)l1 + k2*(double)l2 + k3*(double)l3;
	si = sin(-2.0*PI*kRn);
	co = cos(-2.0*PI*kRn);

	Rx = (double)l1*tv[1][1] + (double)l2*tv[2][1] + (double)l3*tv[3][1];
	Ry = (double)l1*tv[1][2] + (double)l2*tv[2][2] + (double)l3*tv[3][2];
	Rz = (double)l1*tv[1][3] + (double)l2*tv[2][3] + (double)l3*tv[3][3];

        for (j=0; j<No[spin]; j++){ 
          for (i=0; i<MatDim; i++){ 

            tmp1 = FR1[spin][l1+Ra][l2+Rb][l3+Rc][j*MatDim+i]*co;
            tmp2 = FR1[spin][l1+Ra][l2+Rb][l3+Rc][j*MatDim+i]*si; 

	    Fk1[j*MatDim+i].r += tmp1;
	    Fk1[j*MatDim+i].i += tmp2;

	    dFk1[0][j*MatDim+i].r += Rx*tmp2;
	    dFk1[0][j*MatDim+i].i -= Rx*tmp1;

	    dFk1[1][j*MatDim+i].r += Ry*tmp2;
	    dFk1[1][j*MatDim+i].i -= Ry*tmp1;

	    dFk1[2][j*MatDim+i].r += Rz*tmp2;
	    dFk1[2][j*MatDim+i].i -= Rz*tmp1;

	  }
	}
      }
    }
  }    

  /* calculate B11 = Fk1^dag S C */

  Calc_Hk_Sk(Hk,Sk,spin,k1,k2,k3);

  m = MatDim; n = No[spin]; k = MatDim;
  zgemm_("N","N", &m, &n, &k, &alpha, Sk, &m, WF, &k, &beta, Hk, &m);
  for (i=0; i<MatDim*No[spin]; i++){ Sk[i] = Hk[i]; }
    
  m = No[spin]; n = No[spin]; k = MatDim;
  zgemm_("C","N", &m, &n, &k, &alpha, Fk1, &k, Sk, &k, &beta, B11, &m);

  /* calculate the inverse B11 */

  //Calc_Inverse_SVD(No[spin], B11); // inverse B11 is stored in B11 
  Calc_Inverse(No[spin], B11); // inverse B11 is stored in B11 

  /* calculate B11^-1 * (Fk1^dag)^(a) * (S C1) */

  for (p=0; p<3; p++){
  
    m = No[spin]; n = MatDim; k = No[spin];
    zgemm_("N","C", &m, &n, &k, &alpha, B11, &m, dFk1[p], &n, &beta, Hk, &m);

    m = No[spin]; n = No[spin]; k = MatDim;
    zgemm_("N","N", &m, &n, &k, &alpha, Hk, &m, Sk, &k, &beta, A11[1][p], &m);

    /* multiplying (-i) */

    for (i=0; i<No[spin]*No[spin]; i++){ 
      ctmp1 = A11[1][p][i];
      A11[1][p][i].r = ctmp1.i;
      A11[1][p][i].i =-ctmp1.r;
    }
  }

  /*************************************************************
                       freeing of arrays
  *************************************************************/

  free(Hk);
  free(Sk);
  free(Fk1);

  for (i=0; i<3; i++){
    free(dFk1[i]);
  }
  free(dFk1);

  free(B11);
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


void Polar_Decompose(int n, dcomplex *A, dcomplex *U)
{
  int i,j,lwork,info,M,N,K;
  double *rwork,*sv;
  dcomplex *W,*VT,*work;
  dcomplex alpha = {1.0,0.0}; 
  dcomplex beta = {0.0,0.0};

  /* allocation of arrays */

  lwork = 5*n;
  sv = (double*)malloc(sizeof(double)*n);
  work = (dcomplex*)malloc(sizeof(dcomplex)*lwork);
  rwork = (double*)malloc(sizeof(double)*lwork);
  W = (dcomplex*)malloc(sizeof(dcomplex)*n*n);
  VT = (dcomplex*)malloc(sizeof(dcomplex)*n*n);

  /* SVD of A */

  zgesvd_("A","A",&n,&n,A,&n,sv,W,&n,VT,&n,work,&lwork,rwork,&info);

  /*
  for (i=0; i<n; i++){
    printf("AAA i=%2d SV=%18.15f\n",i,SV[i]);
  }
  */

  /* U = W*VT */

  M = n; N = n; K = n;
  zgemm_("N","N", &M, &N, &K, &alpha, W, &M, VT, &K, &beta, U, &M);

  /* freeing of arrays */

  free(sv);
  free(work);
  free(rwork);
  free(W);
  free(VT);
}


void Calc_Inverse_SVD(int n, dcomplex *A)
{
  int i,j,lwork,info,M,N,K;
  double *sv,*rwork;
  dcomplex *U,*VT,*work;
  dcomplex alpha = {1.0,0.0}; 
  dcomplex beta = {0.0,0.0};

  /* allocation of arrays */

  lwork = 5*n;
  sv = (double*)malloc(sizeof(double)*n);
  U = (dcomplex*)malloc(sizeof(dcomplex)*n*n);
  VT = (dcomplex*)malloc(sizeof(dcomplex)*n*n);
  work = (dcomplex*)malloc(sizeof(dcomplex)*lwork);
  rwork = (double*)malloc(sizeof(double)*lwork);

  /* SVD of A */

  zgesvd_("A","A",&n,&n,A,&n,sv,U,&n,VT,&n,work,&lwork,rwork,&info);

  /*
  for (i=0; i<n; i++){
    printf("AAA i=%2d sv=%18.15f\n",i,sv[i]);
  }
  */

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

  /* freeing of arrays */

  free(sv);
  free(U);
  free(VT);
  free(work);
  free(rwork);
}






void Calc_Guiding_Functions_1(double **PF1)
{
  int i,j,k,mu,ii,ij,ik,jj,kk,kloop,spin,p1,p2,p3;
  double k1,k2,k3,tmp;
  double *S0,*J1,*eval;
  dcomplex *Hk,*Sk,*WF,*D;

  /*************************************************************
                      allocation of arrays
  *************************************************************/

  S0 = (double*)malloc(sizeof(double)*MatDim*MatDim);
  J1 = (double*)malloc(sizeof(double)*MatDim*MatDim);
  D = (dcomplex*)malloc(sizeof(dcomplex)*MatDim*MatDim);
  Hk = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));
  Sk = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));
  WF = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));
  eval = (double*)malloc(sizeof(double)*MatDim);

  /******************************************************************/
  /*******************************************************************
       Calculate guiding functions by maximizing the projection.
  *******************************************************************/
  /******************************************************************/

  for (spin=0; spin<(SpinP_switch+1); spin++){ 

    /* initialize J1 */

    for (i=0; i<MatDim*MatDim; i++){
      J1[i] = 0.0;
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

      /* calculate D */ 

      Calc_D(k1,k2,k3,WF,D); 

      /* calculate J1 at the k-point */ 

      for (mu=0; mu<No[spin]; mu++){
	for (i=0; i<MatDim; i++){
	  for (j=0; j<MatDim; j++){
            tmp = D[mu*MatDim+i].r*D[mu*MatDim+j].r + D[mu*MatDim+i].i*D[mu*MatDim+j].i;
            J1[i*MatDim+j] -= (double)T_k_op[kloop]*tmp;
	  }
	}      
      }

    } // end of kloop

    /****************************************
      calculations of guiding functions
    ****************************************/

    /* normalize J1 and J2 */

    for (i=0; i<MatDim*MatDim; i++){ J1[i] /= tweight; }

    /* diagonalize the generalized eigenvalue problem */

    Calc_S0(spin,k1,k2,k3,S0);
    lapack_dsygvx(MatDim, 1, No[spin], J1, S0, PF1[spin], eval);

    for (i=0; i<No[spin]; i++){
      printf("J1 %i %15.12f\n",i,eval[i]);
    }

  } // spin

  /*************************************************************
                      freeing of arrays
  *************************************************************/

  free(S0);
  free(J1);
  free(D);
  free(Hk);
  free(Sk);
  free(WF);
  free(eval);
}


void Calc_Guiding_Functions_1_2(double **PF1, double **PF2)
{
  int i,j,k,mu,ii,ij,ik,jj,kk,kloop,spin,p1,p2,p3;
  double k1,k2,k3,tmp;
  double *S0,*J1,*J2,*eval;
  dcomplex *Hk,*Sk,*WF,*D;

  /*************************************************************
                      allocation of arrays
  *************************************************************/

  S0 = (double*)malloc(sizeof(double)*MatDim*MatDim);
  J1 = (double*)malloc(sizeof(double)*MatDim*MatDim);
  J2 = (double*)malloc(sizeof(double)*MatDim*MatDim);
  D = (dcomplex*)malloc(sizeof(dcomplex)*MatDim*MatDim);
  Hk = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));
  Sk = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));
  WF = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));
  eval = (double*)malloc(sizeof(double)*MatDim);

  /******************************************************************/
  /*******************************************************************
       Calculate guiding functions by maximizing the projection.
  *******************************************************************/
  /******************************************************************/

  for (spin=0; spin<(SpinP_switch+1); spin++){ 

    /* initialize J1 and J2 */

    for (i=0; i<MatDim*MatDim; i++){
      J1[i] = 0.0;
      J2[i] = 0.0;
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

      /* calculate D */ 

      Calc_D(k1,k2,k3,WF,D); 

      /* calculate J1 and J2 at the k-point */ 

      for (mu=0; mu<No[spin]; mu++){
	for (i=0; i<MatDim; i++){
	  for (j=0; j<MatDim; j++){
            tmp = D[mu*MatDim+i].r*D[mu*MatDim+j].r + D[mu*MatDim+i].i*D[mu*MatDim+j].i;
            J1[i*MatDim+j] -= (double)T_k_op[kloop]*tmp;
	  }
	}      
      }

      for (mu=No[spin]; mu<MatDim; mu++){
	for (i=0; i<MatDim; i++){
	  for (j=0; j<MatDim; j++){
            tmp = D[mu*MatDim+i].r*D[mu*MatDim+j].r + D[mu*MatDim+i].i*D[mu*MatDim+j].i;
            J2[i*MatDim+j] -= (double)T_k_op[kloop]*tmp;
	  }
	}      
      }

    } // end of kloop

    /****************************************
      calculations of guiding functions
    ****************************************/

    /* normalize J1 and J2 */

    for (i=0; i<MatDim*MatDim; i++){ J1[i] /= tweight; }
    for (i=0; i<MatDim*MatDim; i++){ J2[i] /= tweight; }

    /* diagonalize the generalized eigenvalue problem */

    Calc_S0(spin,k1,k2,k3,S0);
    lapack_dsygvx(MatDim, 1, No[spin], J1, S0, PF1[spin], eval);


    for (i=0; i<No[spin]; i++){
      printf("J1 %i %15.12f\n",i,eval[i]);
    }


    Calc_S0(spin,k1,k2,k3,S0);

    lapack_dsygvx(MatDim, 1, Nuo[spin], J2, S0, PF2[spin], eval);

    printf("\n"); 
    for (i=0; i<Nuo[spin]; i++){
      printf("J2 %i %15.12f\n",i,eval[i]);
    }

  } // spin


  /*************************************************************
                      freeing of arrays
  *************************************************************/

  free(S0);
  free(J1);
  free(J2);
  free(D);
  free(Hk);
  free(Sk);
  free(WF);
  free(eval);
}









void Calc_D(double k1, double k2, double k3, dcomplex *WF, dcomplex *D)
{
  int i,j,mu,GA_AN,LB_AN,GB_AN,Rn,tnoA,tnoB,Anum,Bnum,l1,l2,l3;
  double co,si,kRn,s,s0,s1;

  /* initiaize D */  

  for (i=0; i<(MatDim*MatDim); i++){
    D[i] = Complex(0.0,0.0);
  } 

  /* calculate D */  

  for (mu=0; mu<MatDim; mu++){
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

	    s = OLP[GA_AN][LB_AN][i][j];
            s0 = s*co;
            s1 = s*si; 

	    D[mu*MatDim+Anum+i].r += WF[mu*MatDim+Bnum+j].r*s0 - WF[mu*MatDim+Bnum+j].i*s1;
	    D[mu*MatDim+Anum+i].i += WF[mu*MatDim+Bnum+j].r*s1 + WF[mu*MatDim+Bnum+j].i*s0;
	  }
	}

      } // LB_AN
    } // GA_AN
  } // mu
}

void Calc_S0(int spin, double k1, double k2, double k3, double *S0)
{
  int i,j,GA_AN,LB_AN,GB_AN,Rn,tnoA,tnoB,Anum,Bnum,l1,l2,l3;
  double co,si,kRn,s,h;

  /* initiaize S0 */  

  for (i=0; i<(MatDim*MatDim); i++){
    S0[i] = 0.0;
  } 

  /* calculate S0 */  

  for (GA_AN=1; GA_AN<=atomnum; GA_AN++){
    tnoA = Total_NumOrbs[GA_AN];
    Anum = MP[GA_AN];

    for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
      GB_AN = natn[GA_AN][LB_AN];
      Rn = ncn[GA_AN][LB_AN];
      tnoB = Total_NumOrbs[GB_AN];
      Bnum = MP[GB_AN];

      if (Rn==0){

	for (i=0; i<tnoA; i++){
	  for (j=0; j<tnoB; j++){
	    s = OLP[GA_AN][LB_AN][i][j];
	    S0[(Bnum+j)*MatDim+Anum+i] = s;
	  }
	}

      } // end of if (Rn==0)
    } // LB_AN
  } // GA_AN
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


void lapack_dsygvx(int N, int IL, int IU, double *A, double *B, double *Z, double *W)
{
  char *JOBZ="V";
  char *UPLO="L";
  char *RANGE="A";
  int ITYPE,i,LDA,LDB,LDZ,LWORK;
  int *IWORK,*IFAIL,M,INFO;
  double ABSTOL=1.0e-14;
  double VL,VU;
  double *WORK;

  /* allocation of arrays */

  ITYPE = 1;
  LDA = N;  LDB = N; LDZ = N;

  LWORK = 8*N;
  WORK  = (double*)malloc(sizeof(double)*LWORK);
  IWORK = (int*)malloc(sizeof(int)*5*N);
  IFAIL = (int*)malloc(sizeof(int)*N);

  /* call zhegvx */

  dsygvx_(&ITYPE, JOBZ, RANGE, UPLO, &N, A, &LDA, B, &LDB, &VL, &VU, &IL, &IU,
	  &ABSTOL, &M, W, Z, &LDZ, WORK, &LWORK, 
	  IWORK, IFAIL, &INFO );

  /* freeing of arrays */

  free(WORK);
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
  int GA_AN,spin,i,j,k,l1,l2,l3;

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
      PF1[spin] = (double*)malloc(sizeof(double)*MatDim*MatDim);
    }

    PF2 = (double**)malloc(sizeof(double*)*(SpinP_switch+1));
    for (spin=0; spin<=SpinP_switch; spin++){
      PF2[spin] = (double*)malloc(sizeof(double)*MatDim*MatDim);
    }

    FR1 = (double*****)malloc(sizeof(double****)*(SpinP_switch+1));
    for (spin=0; spin<=SpinP_switch; spin++){
      FR1[spin] = (double****)malloc(sizeof(double***)*Nk1);
      for (l1=0; l1<Nk1; l1++){
        FR1[spin][l1] = (double***)malloc(sizeof(double**)*Nk2);
        for (l2=0; l2<Nk2; l2++){
          FR1[spin][l1][l2] = (double**)malloc(sizeof(double*)*Nk3);
          for (l3=0; l3<Nk3; l3++){
            FR1[spin][l1][l2][l3] = (double*)malloc(sizeof(double)*MatDim*Nomax);
	  }
        }
      }
    }

    FR2 = (double*****)malloc(sizeof(double****)*(SpinP_switch+1));
    for (spin=0; spin<=SpinP_switch; spin++){
      FR2[spin] = (double****)malloc(sizeof(double***)*Nk1);
      for (l1=0; l1<Nk1; l1++){
        FR2[spin][l1] = (double***)malloc(sizeof(double**)*Nk2);
        for (l2=0; l2<Nk2; l2++){
          FR2[spin][l1][l2] = (double**)malloc(sizeof(double*)*Nk3);
          for (l3=0; l3<Nk3; l3++){
            FR2[spin][l1][l2][l3] = (double*)malloc(sizeof(double)*MatDim*Nuomax);
	  }
        }
      }
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

    for (i=0; i<Nk1; i++){
      for (j=0; j<Nk2; j++){
	free(k_op[i][j]);
      }
      free(k_op[i]);
    }
    free(k_op);

    for (spin=0; spin<=SpinP_switch; spin++){
      for (l1=0; l1<Nk1; l1++){
        for (l2=0; l2<Nk2; l2++){
          for (l3=0; l3<Nk3; l3++){
            free(FR1[spin][l1][l2][l3]);
	  }
          free(FR1[spin][l1][l2]);
        }
        free(FR1[spin][l1]);
      }
      free(FR1[spin]);
    }
    free(FR1);

    for (spin=0; spin<=SpinP_switch; spin++){
      for (l1=0; l1<Nk1; l1++){
        for (l2=0; l2<Nk2; l2++){
          for (l3=0; l3<Nk3; l3++){
            free(FR2[spin][l1][l2][l3]);
	  }
          free(FR2[spin][l1][l2]);
        }
        free(FR2[spin][l1]);
      }
      free(FR2[spin]);
    }
    free(FR2);

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


void Set_Kpoints()
{
  /*************************************************************
     k-points by a Gamma-centered mesh

     k_op[i][j][k]: weight of DOS 
                 =0  no calc.
                 =1  time reversal invariant momentum (TRIM)
                 =2  which has k<->-k point
        Now, only the relation, E(k)=E(-k), is used. 
  *************************************************************/

  int i,j,k,kloop,ii,ij,ik;
  double k1,k2,k3;

  k_op = (int***)malloc(sizeof(int**)*Nk1);
  for (i=0; i<Nk1; i++){
    k_op[i] = (int**)malloc(sizeof(int*)*Nk2);
    for (j=0; j<Nk2; j++){
      k_op[i][j] = (int*)malloc(sizeof(int)*Nk3);
    }
  }

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

            // 1:Chern number, 7:SHG, 8:pth-order optical susceptibility

	    /*
            if (job_id==1 || job_id==7 || job_id==8){
	    */
            if (job_id==1 || job_id==8){

  	      k_op[i][j][k]    = 1;
	      k_op[ii][ij][ik] = 1;
            }
            else {
	      k_op[i][j][k]    = 2;
	      k_op[ii][ij][ik] = 0;
	    }
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

dcomplex Cmul(dcomplex a, dcomplex b)
{
  dcomplex c;
  c.r = a.r*b.r - a.i*b.i;
  c.i = a.i*b.r + a.r*b.i;
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



/**
 * Recursively generate non-decreasing combinations of indices of length p.
 *xo - p: tensor order
 * - start: minimum index to allow
 * - depth: current recursion depth
 * - current: array to store the current combination
 * - combo_list: pointer to array of saved combinations
 * - combo_count: pointer to current number of combinations
 */
void generate_combinations(int p, int start, int depth,
                           int *current,
                           int ***combo_list, int *combo_count) 
{
  int i,DIM=3;
  if (depth == p) {
    *combo_list = realloc(*combo_list,
			  sizeof(int*) * (*combo_count + 1));
    (*combo_list)[*combo_count] = malloc(sizeof(int) * p);
    memcpy((*combo_list)[*combo_count], current,
	   sizeof(int) * p);
    (*combo_count)++;
    return;
  }
  for (i=start; i<DIM; ++i) {
    current[depth] = i;
    generate_combinations(p, i, depth + 1,
			  current, combo_list, combo_count);
  }
}


/**
 * Initialize independent tensor components for order p.
 * - out_list: will point to array of index arrays
 * - out_count: will be set to number of independent combinations
 */
void init_independent_components(int p, int ***out_list, int *out_count)
{
  int *buffer;
  *out_count = 0;
  *out_list = NULL;
  buffer = malloc(sizeof(int) * p);
  generate_combinations(p, 0, 0, buffer, out_list, out_count);
  free(buffer);
}

/**
 * Compare two index arrays of length p for equality.
 * Returns 1 if equal, 0 otherwise.
 */
int equal_indices(const int *a, const int *b, int p) 
{
  int i;
  for (i = 0; i < p; ++i) {
    if (a[i] != b[i]) return 0;
  }
  return 1;
}

/**
 * Map a symmetric index array of length p to its index in the independent list.
 * - idx: array of length p containing indices (e.g., {0,1} for x,y)
 * - ind_list: list of independent combinations
 * - ind_count: number of entries in ind_list
 * Returns the matching index or -1 if not found.
 */
int map_to_independent(int p, const int *idx,
                       int **ind_list, int ind_count)
{
  int *sorted = malloc(sizeof(int) * p);
  int i, j, result = -1;
  // Copy and sort
  for (i = 0; i < p; ++i) sorted[i] = idx[i];
  for (i = 0; i < p - 1; ++i) {
    for (j = i + 1; j < p; ++j) {
      if (sorted[i] > sorted[j]) {
	int tmp = sorted[i];
	sorted[i] = sorted[j];
	sorted[j] = tmp;
      }
    }
  }
  // Find in list
  for (i = 0; i < ind_count; ++i) {
    if (equal_indices(sorted, ind_list[i], p)) {
      result = i;
      break;
    }
  }
  free(sorted);
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




// Delete only regular files and symlinks directly under 'dirpath' (non-recursive).
// Returns 0 on success, -1 if any error occurred (continues as much as possible).

int remove_files_in_dir(const char *dirpath, int *removed_count) {

   if (removed_count) *removed_count = 0;

   DIR *dp = opendir(dirpath);

   if (!dp) {
     // Could not open the directory.
     return -1;
   }

   int rc = 0;             // Becomes -1 if we encounter any error.
   int first_errno = 0;     // Remember the first errno.
   struct dirent *de;
   char path[4096];         // Simple fixed-size buffer for joined path.

   while ((de = readdir(dp)) != NULL) {

     // Skip current and parent directory entries.

     if (strcmp(de->d_name, ".") == 0 || strcmp(de->d_name, "..") == 0) continue;

     // Build "dirpath/entry" into 'path'.

     int n = snprintf(path, sizeof(path), "%s/%s", dirpath, de->d_name);

     if (n < 0 || (size_t)n >= sizeof(path)) {

       if (rc == 0) { rc = -1; first_errno = ENAMETOOLONG; }

       continue;
     }



     // Get file type without following symlinks.

     struct stat st;

     if (lstat(path, &st) != 0) {

       if (rc == 0) { rc = -1; first_errno = errno; }

       continue;
     }



     // We delete regular files and symlinks. Directories are kept (non-recursive).

     if (S_ISREG(st.st_mode) || S_ISLNK(st.st_mode)) {

       if (unlink(path) != 0) {

	 if (rc == 0) { rc = -1; first_errno = errno; }

	 continue;
       }

       if (removed_count) (*removed_count)++;
     }

     // Other types (FIFO, socket, device) are left untouched for simplicity.
   }

   // Preserve errno from above if we had any failure.
   if (closedir(dp) != 0 && rc == 0) {
     rc = -1;
     first_errno = errno;
   }

   if (rc != 0 && first_errno) errno = first_errno;
   return rc;
 }



struct timeval2 {
  long tv_sec;    /* second */
  long tv_usec;   /* microsecond */
};

void dtime(double *t)
{


/* AITUNE
if you don't like, please change to
#ifdef noomp
from 
#ifndef _OPENMP
*/

#ifdef noomp
  /* real time */
  struct timeval timev;
  gettimeofday(&timev, NULL);
  *t = timev.tv_sec + (double)timev.tv_usec*1e-6;
#else
  *t = omp_get_wtime();
#endif

  /* user time + system time */
  /*
  float tarray[2];
  clock_t times(), wall;
  struct tms tbuf;
  wall = times(&tbuf);
  tarray[0] = (float) (tbuf.tms_utime / (float)CLK_TCK);
  tarray[1] = (float) (tbuf.tms_stime / (float)CLK_TCK);
  *t = (double) (tarray[0]+tarray[1]);
  printf("dtime: %lf\n",*t);
  */

}

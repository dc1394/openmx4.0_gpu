/*************************************************************************
  dc.c:
 
  dc.c is a program to calculate dc-current tensor which is the the second 
  order optical response, such as shift current, within  collinear and 
  non-collinear DFT. The hamiltnonian is read from a scfout file.

  Log of dc.c:

     20/Dec./2025  Released by T. Ozaki 

**************************************************************************/
 
#include <stdio.h>
#include <math.h>
#include "mpi.h"
#include "read_scfout.h"
#include "Inputtools.h"

/* define struct, global variables, and functions */

typedef struct { double r,i; } dcomplex;

#define print_data    0
#define measure_time  0
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

int MatDim,MatDim2,Nk1,Nk2,Nk3,T_knum,output_allrange;
int *MP,No[2],Nuo[2],Nomax,Nuomax;
char filename[YOUSO10],filepath[YOUSO10];

int *TK2i,*TK2j,*TK2k,*T_k_op,***k_op;
double tweight;
double *T_KGrids1,*T_KGrids2,*T_KGrids3;
double Egap[2];
int Ng; 
double omega_min,omega_max,eta_val[2],scissor;

double sgn(double nu);
dcomplex Cdiv(dcomplex a, dcomplex b);
dcomplex Cmul(dcomplex a, dcomplex b);
dcomplex Cadd(dcomplex a, dcomplex b);
dcomplex Csub(dcomplex a, dcomplex b);

void Calc_Inverse(int n, double *A);

void save_dc_col
(char *fname, int Ng, dcomplex *******sigma, int **irr_count,
 double *realomega_val);

void save_dc_noncol
(char *fname, int Ng, dcomplex ******sigma, int **irr_count,
 double *realomega_val);

void Input_para(char *file);
void Set_Kpoints();

int remove_one_and_get_pmq(int p, int m, int q, int s,
                           int *p2, int *m2, int *q2);
static int cmp_int(const void *a, const void *b);
static const char* comp_name(int v);
static void print_entry_raw(const int *e, int p);
static void free_list_2d(int **list, int count);
static int equal_indices(const int *a, const int *b, int p);
static int gen_axis_multisets_rec(int len, int depth, int start_axis,
                                  int *buffer, int ***out_list, int *out_count);
static int gen_axis_multisets(int len, int ***out_list, int *out_count);
int init_irreducible_components_pm(int p, int m, int ***out_list, int *out_count);
int map_to_irreducible_pm(int p, int m, const int *idx,
                          int **ind_list, int ind_count);

int test(void);






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

void allocate_free_arrays(char *mode);
dcomplex Complex(double re, double im){ return (dcomplex){re, im}; }
void Cross_Product(double a[4], double b[4], double c[4]);
double Dot_Product(double a[4], double b[4]);

void lapack_zhegvx(int N, dcomplex *A, dcomplex *B, dcomplex *Z, double *W);

void Calc_r_VNL_Matrix( int spin, double k1, double k2, double k3, 
                        dcomplex *WF, dcomplex **SCI, dcomplex **r_VNL );

void Calc_Scissor_Matrix( int spin, double k1, double k2, double k3, 
                          double scissor, dcomplex *WF, dcomplex **SCI );

void Calc_Momentum_Matrix( int spin, double k1, double k2, double k3, 
                           dcomplex *WF, double *eval, dcomplex **P );
void Calc_Momentum_Matrix_NC( double k1, double k2, double k3, 
                              dcomplex *WF, double *eval, dcomplex **P );

void Calc_Hk_Sk(dcomplex *Hk, dcomplex *Sk, int spin, double k1, double k2, double k3);
void Calc_Hk_Sk_NC(dcomplex *Hk, dcomplex *Sk, double k1, double k2, double k3);

void DC_col(int pth, char *fname);
void DC_noncol(int pth, char *fname);

int find_index(int p, int *list, int xyz, int ***independent_list, int *independent_count);
void dtime(double *t);


int main(int argc, char *argv[]) 
{
  int myid;
  /* MPI initialization */

  MPI_Init(&argc, &argv);
  if (sizeof(dcomplex)!=(2*sizeof(double))){
    printf("Error 1: 'dc' does not work properly in your computational environment.\n");
    MPI_Finalize();
    exit(0);
  }

  MPI_Comm_rank(MPI_COMM_WORLD,&myid);

  /* show message */

  if (myid==Host_ID){
    printf("\n******************************************************************\n"); 
    printf("******************************************************************\n"); 
    printf(" dc:\n"); 
    printf(" code for calculating the DC current tensor in the 2nd order response."); 
    printf(" Copyright (C), 2026, Taisuke Ozaki\n");
    printf(" This is free software, and you are welcome to\n");
    printf(" redistribute it under the constitution of the GNU-GPLv3.\n");
    printf("******************************************************************\n"); 
    printf("******************************************************************\n\n"); 
  }

  /* read the input file */

  Input_para(argv[2]);

  /* read the .scfout file and allocate arrays */

  read_scfout(argv);
  allocate_free_arrays("allocate");

  /* set k-point information */

  Set_Kpoints(); 

  /* show a message */

  if (myid==Host_ID){
    printf("\n\nThe calculation starts...\n\n");
  }

  /* DC calculation */

  if (SpinP_switch==0 || SpinP_switch==1){ 
    DC_col(2,filename); 
  }
  else {
    DC_noncol(2,filename);
  }

  /* freeing of arrays */

  allocate_free_arrays("free");

  /* show a massage */

  if (myid==Host_ID){
    printf("\n\nThe calculation finished successfully.\n\n");
  }

  MPI_Finalize();
  exit(0);
}



void DC_col(int pth, char *fname)
{
  int i,j,n,m,mm,ms,me,k,l,n1,n2,Ii,p,q,xyz,spin,kloop,kloop_s,kloop_e;
  int N,omg,Nxyz[6],eta_loop,l0,l1,p02,m02,q02,p12,m12,q12;
  double k1,k2,k3,*eval,eta,Bi,alphai,betai;
  double dEcut=-0.1,be,FermiF,max_x=60.0;
  double omega0,pomega0,domega,ei,ej,fm,fn;
  double x,tmp1,vtmp[4],Cell_Volume,f,f0,f1,x0;
  double *realomega_val;
  dcomplex *Hk,*Sk,*WF,**P,****rho,*******sigma;
  dcomplex *work_array,coe1;
  dcomplex csum[3],ctmp1,ctmp2,ctmp3,ctmp4,Omega; 
  dcomplex alpha = {1.0,0.0}; 
  dcomplex beta = {0.0,0.0};
  dcomplex kappa,kappa_p,mkappa_p;
  double Stime,Etime,time0,time1,time2,time3;
  int ****irr_list;
  int **irr_count;
  int numprocs,myid;

  /* set parameters */

  domega = (omega_max - omega_min)/(double)(Ng-1);
  be = 1.0/kB/(E_Temp/eV2Hartree);

  /* MPI setting */ 

  MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD,&myid);

  kloop_s = (myid*T_knum)/numprocs;
  kloop_e = ((myid+1)*T_knum)/numprocs-1;

  /* initialize time */ 

  time0 = 0.0;
  time1 = 0.0;
  time2 = 0.0;
  time3 = 0.0;
  
  /*************************************************************
        setting information for irreducible combinations
  *************************************************************/

  irr_count = (int**)malloc(sizeof(int*)*(pth+1));
  for (p=0; p<=pth; p++){
    irr_count[p] = (int*)malloc(sizeof(int)*(p+1));
  }

  irr_list = (int****)malloc(sizeof(int***)*(pth+1));
  for (p=0; p<=pth; p++){
    irr_list[p] = (int***)malloc(sizeof(int**)*(p+1));
    for (m=0; m<=p; m++){
      init_irreducible_components_pm(p, m, &irr_list[p][m], &irr_count[p][m]);
    }
  }  

  /*************************************************************
                    allocation of arrays
  *************************************************************/

  work_array = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));

  rho = (dcomplex****)malloc(sizeof(dcomplex***)*(pth+1));
  for (p=0; p<(pth+1); p++){
    rho[p] = (dcomplex***)malloc(sizeof(dcomplex**)*(p+1));
    for (m=0; m<(p+1); m++){
      rho[p][m] = (dcomplex**)malloc(sizeof(dcomplex*)*irr_count[p][m]);
      for (q=0; q<irr_count[p][m]; q++){
	rho[p][m][q] = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));
      }
    }
  }

  sigma = (dcomplex*******)malloc(sizeof(dcomplex******)*(SpinP_switch+1));
  for (spin=0; spin<=SpinP_switch; spin++){
    sigma[spin] = (dcomplex******)malloc(sizeof(dcomplex*****)*2);
    for (eta_loop=0; eta_loop<2; eta_loop++){
      sigma[spin][eta_loop] = (dcomplex*****)malloc(sizeof(dcomplex****)*Ng);
      for (omg=0; omg<Ng; omg++){
	sigma[spin][eta_loop][omg] = (dcomplex****)malloc(sizeof(dcomplex***)*3);
	for (xyz=0; xyz<3; xyz++){
	  sigma[spin][eta_loop][omg][xyz] = (dcomplex***)malloc(sizeof(dcomplex**)*(pth+1));
	  for (p=0; p<(pth+1); p++){
	    sigma[spin][eta_loop][omg][xyz][p] = (dcomplex**)malloc(sizeof(dcomplex*)*(p+1));
            for (m=0; m<(p+1); m++){
  	      sigma[spin][eta_loop][omg][xyz][p][m] = (dcomplex*)malloc(sizeof(dcomplex)*irr_count[p][m]);
	      for (i=0; i<irr_count[p][m]; i++){
	        sigma[spin][eta_loop][omg][xyz][p][m][i] = Complex(0.0,0.0);
	      } 
	    }           
	  }
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

  realomega_val = (double*)malloc(sizeof(double)*Ng);

  /* calculte cell volume */

  Cross_Product(tv[2],tv[3],vtmp);
  Cell_Volume = fabs(Dot_Product(tv[1],vtmp));

  Egap[0] = 1.0e+10;
  Egap[1] = 1.0e+10;

  /*************************************************************
                         setting omega
  *************************************************************/

  domega = (omega_max - omega_min)/(double)(Ng-3);

  realomega_val[0] = omega_min;
  realomega_val[1] = omega_min + 1.0*domega/3.0;
  realomega_val[2] = omega_min + 2.0*domega/3.0;

  for (omg=3; omg<Ng; omg++){
    omega0 = omega_min + (double)(omg-2)*domega;
    realomega_val[omg] = omega0;
  }

  /*************************************************************
                loops for spin and k-points
  *************************************************************/

  for (spin=0; spin<=SpinP_switch; spin++){ 

    /* loop for k-points */

    for (kloop=kloop_s; kloop<=kloop_e; kloop++){

      if (myid==Host_ID){
        printf("For myid=0 and spin=%2d, kloop=%5d / %5d\n",
               spin,kloop,kloop_e);fflush(stdout);
      }

      dtime(&Stime);

      k1 = T_KGrids1[kloop]; 
      k2 = T_KGrids2[kloop]; 
      k3 = T_KGrids3[kloop];

      /* make H(k) and S(k), and solve the generalized eigenvalue problem */ 

      Calc_Hk_Sk(Hk,Sk,spin,k1,k2,k3);
      lapack_zhegvx(MatDim,Hk,Sk,WF,eval);

      /* include the zero-th order contribution of the scissor operator */

      for (i=0; i<MatDim; i++){
        x = (eval[i]-ChemP)*be;
        if (x<=-max_x) x = -max_x;
        if (max_x<=x)  x = max_x;
        FermiF = 1.0/(1.0 + exp(x));
	eval[i] += scissor*(1.0-FermiF); 
      }

      /* find the minimum Egap */

      if ( (eval[No[spin]]-eval[No[spin]-1])<Egap[spin] ){
	Egap[spin] = eval[No[spin]] - eval[No[spin]-1];
      }

      /* calculate the momentum matrix */

      Calc_Momentum_Matrix(spin, k1, k2, k3, WF, eval, P);

      dtime(&Etime);
      time0 += Etime - Stime;

      /*************************************************************
                             loops for eta
      *************************************************************/

      for (eta_loop=0; eta_loop<2; eta_loop++){

	/* set eta */

	eta = eta_val[eta_loop];

	/*********************************************
              calculate pth-order conductivity
	*********************************************/

        dtime(&Stime);

	/* loop for omega0 */ 

	for (omg=0; omg<Ng; omg++){

	  /* set omega0 */

	  omega0 = realomega_val[omg];

	  /********************************************************
                calculate density matrices with underline 
                by the recurrence formula
	  ********************************************************/

	  /* set rho[0] in which 1 is set to the diagonal parts of the occupied states. */

	  for (i=0; i<MatDim*MatDim; i++){ 
	    rho[0][0][0][i] = Complex(0.0,0.0); 
	  }
	  for (i=0; i<MatDim; i++){
            x = (eval[i]-ChemP)*be;
            if (x<=-max_x) x = -max_x;
            if (max_x<=x)  x = max_x;
            FermiF = 1.0/(1.0 + exp(x));
	    rho[0][0][0][i*MatDim+i] = Complex(FermiF,0.0);
	  }

	  /* loop for p */

	  for (p=1; p<=pth; p++){

            /* loop for mm */
           
            if (p==1){
              ms = 0; me = p;
            }
            else if (p==2){
              ms = 1; me = 1;
	    }               

            for (mm=ms; mm<=me; mm++){

	      /* set Omega */

	      Omega.r = (double)(2*mm-p)*omega0;
	      Omega.i = eta;

	      /* loop for irreducible rho */

	      for (q=0; q<irr_count[p][mm]; q++){

		/* set Nxyz */

		Nxyz[0] = 0; Nxyz[1] = 0; Nxyz[2] = 0; 
		Nxyz[3] = 0; Nxyz[4] = 0; Nxyz[5] = 0; 
		for (i=0; i<p; i++){
		  xyz = irr_list[p][mm][q][i];
		  Nxyz[xyz]++;
		}

		/* initialize work_array */

		for (i=0; i<MatDim*MatDim; i++){
		  work_array[i] = Complex(0.0,0.0);
		}

		/* sum over xyz */

		for (xyz=0; xyz<3; xyz++){

		  l0 = remove_one_and_get_pmq(p, mm, q, xyz,   &p02, &m02, &q02);                    
		  l1 = remove_one_and_get_pmq(p, mm, q, 3+xyz, &p12, &m12, &q12);                    
             
                  if (l0==1){

		    alphai = (double)Nxyz[xyz]/(double)p;

		    /* zgemm of P and rho */

		    alpha = Complex(alphai, 0.0);
		    beta = Complex(1.0,0.0); 
		    m = MatDim; n = MatDim; k = MatDim;
		    zgemm_("N","N", &m, &n, &k, &alpha, P[xyz], &m, 
			   rho[p02][m02][q02], &k, &beta, work_array, &m);

		    /* zgemm of rho and P */

		    alpha = Complex(-alphai, 0.0);
		    beta = Complex(1.0,0.0); 
		    m = MatDim; n = MatDim; k = MatDim;
		    zgemm_("N","N", &m, &n, &k, &alpha, rho[p02][m02][q02], &m, 
			   P[xyz], &k, &beta, work_array, &m);

		  } // end of if (l0==1)

                  if (l1==1){

		    betai = (double)Nxyz[3+xyz]/(double)p;

		    /* zgemm of P and rho */

		    alpha = Complex(-betai, 0.0);
		    beta = Complex(1.0,0.0); 
		    m = MatDim; n = MatDim; k = MatDim;
		    zgemm_("N","N", &m, &n, &k, &alpha, P[xyz], &m, 
			   rho[p12][m12][q12], &k, &beta, work_array, &m);

		    /* zgemm of rho and P */

		    alpha = Complex(betai, 0.0);
		    beta = Complex(1.0,0.0); 
		    m = MatDim; n = MatDim; k = MatDim;
		    zgemm_("N","N", &m, &n, &k, &alpha, rho[p12][m12][q12], &m, 
			   P[xyz], &k, &beta, work_array, &m);

		  } // end of if (l1==1)

		} // xyz

		/* multiply work_array with 1/(hbar*Omega-(ei-ej)) */

		for (i=0; i<MatDim; i++){

		  ei = eval[i];

		  for (j=0; j<MatDim; j++){

		    ej = eval[j];

		    /* calculate rho with underline */

		    ctmp1.r = Omega.r - (ei - ej);
		    ctmp1.i = Omega.i;

		    ctmp2 = Complex(1.0,0.0);
		    ctmp3 = Cdiv(ctmp2,ctmp1);
		    ctmp1 = work_array[j*MatDim+i];
		    rho[p][mm][q][j*MatDim+i] = Cmul(ctmp1,ctmp3);

		    if (fabs(ei-ej)<dEcut){
		      rho[p][mm][q][j*MatDim+i] = Complex(0.0,0.0);
		    }

		  } // j
		} // i
	      } // q 
	    } // mm
	  } // p 

	  /********************************************************
                              calculate sigma
	  ********************************************************/

	  /* loop for p */

	  for (p=1; p<=pth; p++){

            if (p==1){
              ms = 0; me = p;
            }
            else if (p==2){
              ms = 1; me = 1;
	    }               

            for (mm=ms; mm<=me; mm++){

	      /* loop for irreducible sigma */

	      for (q=0; q<irr_count[p][mm]; q++){

		csum[0] = Complex(0.0,0.0); 
		csum[1] = Complex(0.0,0.0); 
		csum[2] = Complex(0.0,0.0); 

		/* trace(p_i*\underline{rho_I^(p)}) */

		for (i=0; i<3; i++){
		  for (m=0; m<MatDim; m++){
		    for (n=0; n<MatDim; n++){

		      ctmp1 = P[i][m*MatDim+n];            // [n][m]
		      ctmp2 = rho[p][mm][q][m*MatDim+n];   // [n][m]
		      csum[i].r += (ctmp1.r*ctmp2.r + ctmp1.i*ctmp2.i); 
		      csum[i].i += (ctmp1.r*ctmp2.i - ctmp1.i*ctmp2.r);
		    }
		  }
		}             

		/* add the trace to sigma */

		for (i=0; i<3; i++){
		  csum[i].r *= (double)T_k_op[kloop];
		  csum[i].i *= (double)T_k_op[kloop];
		}

		sigma[spin][eta_loop][omg][0][p][mm][q] = Cadd(sigma[spin][eta_loop][omg][0][p][mm][q], csum[0]);
		sigma[spin][eta_loop][omg][1][p][mm][q] = Cadd(sigma[spin][eta_loop][omg][1][p][mm][q], csum[1]);
		sigma[spin][eta_loop][omg][2][p][mm][q] = Cadd(sigma[spin][eta_loop][omg][2][p][mm][q], csum[2]);

	      } // q
	    } // mm
	  } // p
	} // omg          

        dtime(&Etime);
        time2 += Etime - Stime;

      } /* end of eta_loop */
    } /* end of kloop */ 

    /* multiplying sigma with kappa^p, 
       and taking account of the prefactor */ 

    for (eta_loop=0; eta_loop<2; eta_loop++){
      for (omg=0; omg<Ng; omg++){

	omega0 = realomega_val[omg];
	kappa = Complex(0.0,-1.0/omega0);
	kappa_p = Complex(1.0,0.0);

	f = 1.0/(tweight*Cell_Volume);

	for (p=1; p<=pth; p++){

	  kappa_p = Cmul(kappa,kappa_p);
	  mkappa_p.r =-kappa_p.r*f;
	  mkappa_p.i =-kappa_p.i*f;

	  if (p==1){
	    ms = 0; me = p;
          }
	  else if (p==2){
	    ms = 1; me = 1;
	  }               

          for (mm=ms; mm<=me; mm++){
	    for (q=0; q<irr_count[p][mm]; q++){

	      sigma[spin][eta_loop][omg][0][p][mm][q] = Cmul(sigma[spin][eta_loop][omg][0][p][mm][q], mkappa_p);
	      sigma[spin][eta_loop][omg][1][p][mm][q] = Cmul(sigma[spin][eta_loop][omg][1][p][mm][q], mkappa_p);
	      sigma[spin][eta_loop][omg][2][p][mm][q] = Cmul(sigma[spin][eta_loop][omg][2][p][mm][q], mkappa_p);

	    } // q
	  } // mm
	} // p 
      } // omg
    } // eta_loop

    /***************************************************************
       For p=2, eliminate the divergent contribution near omega=0.
    ***************************************************************/

    p = 2;
    mm = 1;
    eta_loop = 0;
    double *A,w[4];
    dcomplex b[4],c[4];

    A = (double*)malloc(sizeof(double)*16);

    for (i=0; i<4; i++) w[i] = realomega_val[i];

    for (i=0; i<4; i++){
      for (j=0; j<4; j++){
        A[j*4+i] = pow(w[i],(double)j);
      }
    }

    Calc_Inverse(4, A);

    for (xyz=0; xyz<3; xyz++){
      for (q=0; q<irr_count[p][mm]; q++){

        for (i=0; i<4; i++){
  	  b[i].r = sigma[spin][eta_loop][i][xyz][p][mm][q].r*w[i]*w[i];
	  b[i].i = sigma[spin][eta_loop][i][xyz][p][mm][q].i*w[i]*w[i];
	}

        for (i=0; i<4; i++){
          c[i] = Complex(0.0,0.0); 
          for (j=0; j<4; j++){
            c[i].r += A[j*4+i]*b[j].r; 
            c[i].i += A[j*4+i]*b[j].i; 
	  }
	}

	for (omg=0; omg<Ng; omg++){
	  omega0 = realomega_val[omg];
	  sigma[spin][eta_loop][omg][xyz][p][mm][q].r -= (c[0].r/(omega0*omega0)+c[1].r/omega0+c[2].r);  
	  sigma[spin][eta_loop][omg][xyz][p][mm][q].i -= (c[0].i/(omega0*omega0)+c[1].i/omega0+c[2].i);
	}
      }  
    } 

    free(A);

  } /* end of spin */

  /*************************************************************
                        MPI communication
  *************************************************************/

  dtime(&Stime);

  for (spin=0; spin<=SpinP_switch; spin++){ 
    MPI_Allreduce(MPI_IN_PLACE, &Egap[spin], 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
  }
  if (SpinP_switch==0) Egap[1] = Egap[0];

  for (spin=0; spin<=SpinP_switch; spin++){ 
    for (eta_loop=0; eta_loop<2; eta_loop++){
      for (omg=0; omg<Ng; omg++){
	for (p=1; p<=pth; p++){

	  if (p==1){
	    ms = 0; me = p;
          }
	  else if (p==2){
	    ms = 1; me = 1;
	  }               

          for (mm=ms; mm<=me; mm++){

	    MPI_Allreduce( MPI_IN_PLACE, &sigma[spin][eta_loop][omg][0][p][mm][0], 
			   2*irr_count[p][mm], MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );

	    MPI_Allreduce( MPI_IN_PLACE, &sigma[spin][eta_loop][omg][1][p][mm][0], 
			   2*irr_count[p][mm], MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );

	    MPI_Allreduce( MPI_IN_PLACE, &sigma[spin][eta_loop][omg][2][p][mm][0], 
			   2*irr_count[p][mm], MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
	  }
	}
      }
    }
  }

  dtime(&Etime);
  time3 += Etime - Stime;

  /*************************************************************
                  calculate weighted sigma 
  *************************************************************/

  for (spin=0; spin<=SpinP_switch; spin++){ 
    for (p=1; p<=pth; p++){

      if (p==1){
	ms = 0; me = p;
      }
      else if (p==2){
	ms = 1; me = 1;
      }               

      for (mm=ms; mm<=me; mm++){
	for (q=0; q<irr_count[p][mm]; q++){
	  for (omg=0; omg<Ng; omg++){

	    omega0 = realomega_val[omg];

	    x0 = Egap[spin]-(1.0e-5);
	    f1 = 0.5*(erf((1.0e+5)*(omega0-x0))+1.0);
	    f0 = 1.0 - f1; 

	    sigma[spin][0][omg][0][p][mm][q].r = f0*sigma[spin][0][omg][0][p][mm][q].r 
	                                       + f1*sigma[spin][1][omg][0][p][mm][q].r;
	    sigma[spin][0][omg][1][p][mm][q].r = f0*sigma[spin][0][omg][1][p][mm][q].r
	                                       + f1*sigma[spin][1][omg][1][p][mm][q].r;
	    sigma[spin][0][omg][2][p][mm][q].r = f0*sigma[spin][0][omg][2][p][mm][q].r
	                                       + f1*sigma[spin][1][omg][2][p][mm][q].r;

	    sigma[spin][0][omg][0][p][mm][q].i = f0*sigma[spin][0][omg][0][p][mm][q].i 
	                                       + f1*sigma[spin][1][omg][0][p][mm][q].i;
	    sigma[spin][0][omg][1][p][mm][q].i = f0*sigma[spin][0][omg][1][p][mm][q].i 
	                                       + f1*sigma[spin][1][omg][1][p][mm][q].i;
	    sigma[spin][0][omg][2][p][mm][q].i = f0*sigma[spin][0][omg][2][p][mm][q].i 
	                                       + f1*sigma[spin][1][omg][2][p][mm][q].i;
				  
	  } // omg 
	} // q      
      } // mm
    } // p
  } // spin

  /*************************************************************
                  save the computational results
  *************************************************************/

  if (myid==Host_ID){

    /* save 2nd order DC current tensors */

    save_dc_col(fname,Ng,sigma,irr_count,realomega_val);

  } // end of if (myid==Host_ID) 

  /* show elapsed time */

  if (measure_time){
    printf("myid=%2d time: %10.6f %10.6f %10.6f %10.6f\n",myid,time0,time1,time2,time3);
  }

  /*************************************************************
                    freeing of arrays
  *************************************************************/

  free(work_array);

  for (p=0; p<(pth+1); p++){
    for (m=0; m<(p+1); m++){
      for (q=0; q<irr_count[p][m]; q++){
	free(rho[p][m][q]);
      }
      free(rho[p][m]);
    }
    free(rho[p]);
  }
  free(rho);

  for (spin=0; spin<=SpinP_switch; spin++){
    for (eta_loop=0; eta_loop<2; eta_loop++){
      for (omg=0; omg<Ng; omg++){
	for (xyz=0; xyz<3; xyz++){
	  for (p=0; p<(pth+1); p++){
            for (m=0; m<(p+1); m++){
  	      free(sigma[spin][eta_loop][omg][xyz][p][m]);
	    }           
	    free(sigma[spin][eta_loop][omg][xyz][p]);
	  }
          free(sigma[spin][eta_loop][omg][xyz]);
	}
        free(sigma[spin][eta_loop][omg]);
      }
      free(sigma[spin][eta_loop]);
    }
    free(sigma[spin]);
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

  free(realomega_val);

  for (p=0; p<=pth; p++){
    for (m=0; m<=p; m++){
      for (q=0; q<irr_count[p][m]; q++){
        free(irr_list[p][m][q]);
      }
      free(irr_list[p][m]);
    }
    free(irr_list[p]);
  }  
  free(irr_list);

  for (p=0; p<=pth; p++){
    free(irr_count[p]);
  }
  free(irr_count);
}




void DC_noncol(int pth, char *fname)
{
  int i,j,n,m,ms,me,k,l,n1,n2,Ii,p,q,xyz,kloop,kloop_s,kloop_e;
  int N,omg,Nxyz[6],eta_loop,mm,p02,m02,q02,p12,m12,q12,l0,l1;
  double k1,k2,k3,*eval,eta,Bi,alphai,betai;
  double dEcut=-0.1,be,FermiF,max_x=60.0;
  double omega0,pomega0,domega,em,en,fm,fn,f0,f1,x0;
  double x,tmp1,vtmp[4],Cell_Volume,f;
  double *realomega_val;
  dcomplex *Hk,*Sk,*WF,**P,****rho,******sigma;
  dcomplex *work_array,coe1;
  dcomplex csum[3],ctmp1,ctmp2,ctmp3,ctmp4,Omega; 
  dcomplex alpha = {1.0,0.0}; 
  dcomplex beta = {0.0,0.0};
  dcomplex kappa,kappa_p,mkappa_p;
  double Stime,Etime,time0,time1,time2,time3;
  int ****irr_list;
  int **irr_count;
  int numprocs,myid;

  /* set parameters */

  domega = (omega_max - omega_min)/(double)(Ng-1);
  be = 1.0/kB/(E_Temp/eV2Hartree);

  /* MPI setting */ 

  MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD,&myid);

  kloop_s = (myid*T_knum)/numprocs;
  kloop_e = ((myid+1)*T_knum)/numprocs-1;

  /* initialize time */ 

  time0 = 0.0;
  time1 = 0.0;
  time2 = 0.0;
  time3 = 0.0;

  /*************************************************************
        setting information for irreducible combinations
  *************************************************************/

  irr_count = (int**)malloc(sizeof(int*)*(pth+1));
  for (p=0; p<=pth; p++){
    irr_count[p] = (int*)malloc(sizeof(int)*(p+1));
  }

  irr_list = (int****)malloc(sizeof(int***)*(pth+1));
  for (p=0; p<=pth; p++){
    irr_list[p] = (int***)malloc(sizeof(int**)*(p+1));
    for (m=0; m<=p; m++){
      init_irreducible_components_pm(p, m, &irr_list[p][m], &irr_count[p][m]);
    }
  }  

  /*************************************************************
                    allocation of arrays
  *************************************************************/

  work_array = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim2*MatDim2));

  rho = (dcomplex****)malloc(sizeof(dcomplex***)*(pth+1));
  for (p=0; p<(pth+1); p++){
    rho[p] = (dcomplex***)malloc(sizeof(dcomplex**)*(p+1));
    for (m=0; m<(p+1); m++){
      rho[p][m] = (dcomplex**)malloc(sizeof(dcomplex*)*irr_count[p][m]);
      for (q=0; q<irr_count[p][m]; q++){
	rho[p][m][q] = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim2*MatDim2));
      }
    }
  }

  sigma = (dcomplex******)malloc(sizeof(dcomplex*****)*2);
  for (eta_loop=0; eta_loop<2; eta_loop++){
    sigma[eta_loop] = (dcomplex*****)malloc(sizeof(dcomplex****)*Ng);
    for (omg=0; omg<Ng; omg++){
      sigma[eta_loop][omg] = (dcomplex****)malloc(sizeof(dcomplex***)*3);
      for (xyz=0; xyz<3; xyz++){
	sigma[eta_loop][omg][xyz] = (dcomplex***)malloc(sizeof(dcomplex**)*(pth+1));
	for (p=0; p<(pth+1); p++){
	  sigma[eta_loop][omg][xyz][p] = (dcomplex**)malloc(sizeof(dcomplex*)*(p+1));
	  for (m=0; m<(p+1); m++){
	    sigma[eta_loop][omg][xyz][p][m] = (dcomplex*)malloc(sizeof(dcomplex)*irr_count[p][m]);
	    for (i=0; i<irr_count[p][m]; i++){
	      sigma[eta_loop][omg][xyz][p][m][i] = Complex(0.0,0.0);
	    } 
	  }           
	}
      }
    }
  }

  Hk = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim2*MatDim2));
  Sk = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim2*MatDim2));
  WF = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim2*MatDim2));
  eval = (double*)malloc(sizeof(double)*MatDim2);

  P = (dcomplex**)malloc(sizeof(dcomplex*)*3);
  for (i=0; i<3; i++){
    P[i] = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim2*MatDim2));
  }

  realomega_val = (double*)malloc(sizeof(double)*Ng);

  /* calculte cell volume */

  Cross_Product(tv[2],tv[3],vtmp);
  Cell_Volume = fabs(Dot_Product(tv[1],vtmp));

  Egap[0] = 1.0e+10;

  /*************************************************************
                         setting omega
  *************************************************************/

  domega = (omega_max - omega_min)/(double)(Ng-3);

  realomega_val[0] = omega_min;
  realomega_val[1] = omega_min + 1.0*domega/3.0;
  realomega_val[2] = omega_min + 2.0*domega/3.0;

  for (omg=3; omg<Ng; omg++){
    omega0 = omega_min + (double)(omg-2)*domega;
    realomega_val[omg] = omega0;
  }

  /*************************************************************
                       loops for k-points
  *************************************************************/

  /* loop for k-points */

  for (kloop=kloop_s; kloop<=kloop_e; kloop++){

    if (myid==Host_ID){
      printf("For myid=0, kloop=%5d / %5d\n",kloop,kloop_e);fflush(stdout);
    }

    dtime(&Stime);

    k1 = T_KGrids1[kloop]; 
    k2 = T_KGrids2[kloop]; 
    k3 = T_KGrids3[kloop];

    /* make H(k) and S(k), and solve the generalized eigenvalue problem */ 

    Calc_Hk_Sk_NC(Hk,Sk,k1,k2,k3);
    lapack_zhegvx(MatDim2,Hk,Sk,WF,eval);

    /* include the zero-th order contribution of the scissor operator */

    for (i=0; i<MatDim2; i++){
      x = (eval[i]-ChemP)*be;
      if (x<=-max_x) x = -max_x;
      if (max_x<=x)  x = max_x;
      FermiF = 1.0/(1.0 + exp(x));
      eval[i] += scissor*(1.0-FermiF); 
    }

    /* find the minimum Egap */

    if ( (eval[No[0]]-eval[No[0]-1])<Egap[0] ){
      Egap[0] = eval[No[0]] - eval[No[0]-1];
    }

    /* calculate the momentum matrix */

    Calc_Momentum_Matrix_NC(k1, k2, k3, WF, eval, P);

    dtime(&Etime);
    time0 += Etime - Stime;

    /*************************************************************
                             loops for eta
    *************************************************************/

    for (eta_loop=0; eta_loop<2; eta_loop++){

      /* set eta */

      eta = eta_val[eta_loop];

      /*********************************************
              calculate pth-order conductivity
      *********************************************/

      dtime(&Stime);

      /* loop for omega0 */ 

      for (omg=0; omg<Ng; omg++){

	/* set omega0 */

	omega0 = realomega_val[omg];

	/********************************************************
                calculate density matrices with underline 
                by the recurrence formula
	********************************************************/

	/* set rho[0] in which 1 is set to the diagonal parts of the occupied states. */
 
	for (i=0; i<MatDim2*MatDim2; i++){ 
	  rho[0][0][0][i] = Complex(0.0,0.0); 
	}
	for (i=0; i<MatDim2; i++){
          x = (eval[i]-ChemP)*be;
          if (x<=-max_x) x = -max_x;
          if (max_x<=x)  x = max_x;
          FermiF = 1.0/(1.0 + exp(x));
	  rho[0][0][0][i*MatDim2+i] = Complex(FermiF,0.0);
	}

	/* loop for p */

	for (p=1; p<=pth; p++){

          /* loop for mm */

	  if (p==1){
	    ms = 0; me = p;
          }
	  else if (p==2){
	    ms = 1; me = 1;
	  }               
              
          for (mm=ms; mm<=me; mm++){

	    /* set Omega */

	    Omega.r = (double)(2*mm-p)*omega0;
	    Omega.i = eta;

	    /* loop for irreducible rho */

	    for (q=0; q<irr_count[p][mm]; q++){

	      /* set Nxyz */

	      Nxyz[0] = 0; Nxyz[1] = 0; Nxyz[2] = 0; 
	      Nxyz[3] = 0; Nxyz[4] = 0; Nxyz[5] = 0; 
	      for (i=0; i<p; i++){
		xyz = irr_list[p][mm][q][i];
		Nxyz[xyz]++;
	      }

	      /* initialize work_array */

	      for (i=0; i<MatDim2*MatDim2; i++){
		work_array[i] = Complex(0.0,0.0);
	      }

	      /* sum over xyz */

	      for (xyz=0; xyz<3; xyz++){

		l0 = remove_one_and_get_pmq(p, mm, q, xyz,   &p02, &m02, &q02);                    
		l1 = remove_one_and_get_pmq(p, mm, q, 3+xyz, &p12, &m12, &q12);                    

                if (l0==1){

		  alphai = (double)Nxyz[xyz]/(double)p;
                           
		  /* zgemm of P and rho */

		  alpha = Complex(alphai, 0.0);
		  beta = Complex(1.0,0.0); 
		  m = MatDim2; n = MatDim2; k = MatDim2;
		  zgemm_("N","N", &m, &n, &k, &alpha, P[xyz], &m, 
			 rho[p02][m02][q02], &k, &beta, work_array, &m);

		  /* zgemm of rho and P */

		  alpha = Complex(-alphai, 0.0);
		  beta = Complex(1.0,0.0); 
		  m = MatDim2; n = MatDim2; k = MatDim2;
		  zgemm_("N","N", &m, &n, &k, &alpha, rho[p02][m02][q02], &m, 
			 P[xyz], &k, &beta, work_array, &m);
                  
		} // end of if (l0==1)

                if (l1==1){

	          betai = (double)Nxyz[3+xyz]/(double)p;

		  /* zgemm of P and rho */

		  alpha = Complex(-betai, 0.0);
		  beta = Complex(1.0,0.0); 
		  m = MatDim2; n = MatDim2; k = MatDim2;
		  zgemm_("N","N", &m, &n, &k, &alpha, P[xyz], &m, 
			 rho[p12][m12][q12], &k, &beta, work_array, &m);

		  /* zgemm of rho and P */

		  alpha = Complex(betai, 0.0);
		  beta = Complex(1.0,0.0); 
		  m = MatDim2; n = MatDim2; k = MatDim2;
		  zgemm_("N","N", &m, &n, &k, &alpha, rho[p12][m12][q12], &m, 
			 P[xyz], &k, &beta, work_array, &m);

                } // end of if (l1==1)

	      } // xyz

	      /* multiply work_array with 1/(hbar*Omega-(emu-enu)) */

	      for (m=0; m<MatDim2; m++){

		em = eval[m];

		for (n=0; n<MatDim2; n++){

		  en = eval[n];

		  /* calculate rho with underline */

		  ctmp1.r = Omega.r - (em - en);
		  ctmp1.i = Omega.i;

		  ctmp2 = Complex(1.0,0.0);
		  ctmp3 = Cdiv(ctmp2,ctmp1);
		  ctmp1 = work_array[n*MatDim2+m];
		  rho[p][mm][q][n*MatDim2+m] = Cmul(ctmp1,ctmp3);

		  if (fabs(em-en)<dEcut){
		    rho[p][mm][q][n*MatDim2+m] = Complex(0.0,0.0);
		  }

		} // n
	      } // m
	    } // q
	  } // mm 
	} // p 

        /********************************************************
                             calculate sigma
        ********************************************************/

        /* loop for p */

	for (p=1; p<=pth; p++){

	  if (p==1){
	    ms = 0; me = p;
          }
	  else if (p==2){
	    ms = 1; me = 1;
	  }               

	  for (mm=ms; mm<=me; mm++){

	    /* loop for irreducible sigma */

	    for (q=0; q<irr_count[p][mm]; q++){

	      csum[0] = Complex(0.0,0.0); 
	      csum[1] = Complex(0.0,0.0); 
	      csum[2] = Complex(0.0,0.0); 

	      /* trace(p_i*\underline{rho_I^(p)}) */

	      for (i=0; i<3; i++){
		for (m=0; m<MatDim2; m++){
		  for (n=0; n<MatDim2; n++){

		    ctmp1 = P[i][m*MatDim2+n];            // [n][m]
		    ctmp2 = rho[p][mm][q][m*MatDim2+n];   // [n][m]
		    csum[i].r += (ctmp1.r*ctmp2.r + ctmp1.i*ctmp2.i); 
		    csum[i].i += (ctmp1.r*ctmp2.i - ctmp1.i*ctmp2.r);
		  }
		}
	      }             

	      /* add the trace to sigma */

	      for (i=0; i<3; i++){
		csum[i].r *= (double)T_k_op[kloop];
		csum[i].i *= (double)T_k_op[kloop];
	      }

	      sigma[eta_loop][omg][0][p][mm][q] = Cadd(sigma[eta_loop][omg][0][p][mm][q], csum[0]);
	      sigma[eta_loop][omg][1][p][mm][q] = Cadd(sigma[eta_loop][omg][1][p][mm][q], csum[1]);
	      sigma[eta_loop][omg][2][p][mm][q] = Cadd(sigma[eta_loop][omg][2][p][mm][q], csum[2]);

	    } // q
	  } // mm
	} // p
      } // omg          

      dtime(&Etime);
      time2 += Etime - Stime;

    } /* end of eta_loop */
  } /* end of kloop */ 

  /* multiplying sigma with kappa^p, 
     and taking account of the prefactor */ 

  for (eta_loop=0; eta_loop<2; eta_loop++){
    for (omg=0; omg<Ng; omg++){

      omega0 = realomega_val[omg];
      kappa = Complex(0.0,-1.0/omega0);
      kappa_p = Complex(1.0,0.0);

      f = 1.0/(tweight*Cell_Volume);

      for (p=1; p<=pth; p++){

	kappa_p = Cmul(kappa,kappa_p);
	mkappa_p.r =-kappa_p.r*f;
	mkappa_p.i =-kappa_p.i*f;

	if (p==1){
	  ms = 0; me = p;
        }
	else if (p==2){
	  ms = 1; me = 1;
	}               

	for (mm=ms; mm<=me; mm++){
	  for (q=0; q<irr_count[p][mm]; q++){

	    sigma[eta_loop][omg][0][p][mm][q] = Cmul(sigma[eta_loop][omg][0][p][mm][q], mkappa_p);
	    sigma[eta_loop][omg][1][p][mm][q] = Cmul(sigma[eta_loop][omg][1][p][mm][q], mkappa_p);
	    sigma[eta_loop][omg][2][p][mm][q] = Cmul(sigma[eta_loop][omg][2][p][mm][q], mkappa_p);

	  } // q
	} // mm
      } // p
    } // omg
  } // eta_loop

  /*************************************************************
                        MPI communication
  *************************************************************/

  dtime(&Stime);

  MPI_Allreduce(MPI_IN_PLACE, &Egap[0], 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

  for (eta_loop=0; eta_loop<2; eta_loop++){
    for (omg=0; omg<Ng; omg++){
      for (p=1; p<=pth; p++){

	if (p==1){
	  ms = 0; me = p;
	}
	else if (p==2){
	  ms = 1; me = 1;
	}               

	for (mm=ms; mm<=me; mm++){

	  MPI_Allreduce( MPI_IN_PLACE, &sigma[eta_loop][omg][0][p][mm][0], 
			 2*irr_count[p][mm], MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );

	  MPI_Allreduce( MPI_IN_PLACE, &sigma[eta_loop][omg][1][p][mm][0], 
			 2*irr_count[p][mm], MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );

	  MPI_Allreduce( MPI_IN_PLACE, &sigma[eta_loop][omg][2][p][mm][0], 
			 2*irr_count[p][mm], MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );

	}
      }
    }
  }

  dtime(&Etime);
  time3 += Etime - Stime;

  /***************************************************************
    For p=2, eliminate the divergent contribution near omega=0.
  ***************************************************************/

  p = 2;
  mm = 1;
  eta_loop = 0;
  double *A,w[4];
  dcomplex b[4],c[4];

  A = (double*)malloc(sizeof(double)*16);

 for (i=0; i<4; i++) w[i] = realomega_val[i];

 for (i=0; i<4; i++){
   for (j=0; j<4; j++){
     A[j*4+i] = pow(w[i],(double)j);
   }
 }

  Calc_Inverse(4, A);

  for (xyz=0; xyz<3; xyz++){
    for (q=0; q<irr_count[p][mm]; q++){

      for (i=0; i<4; i++){
	b[i].r = sigma[eta_loop][i][xyz][p][mm][q].r*w[i]*w[i];
	b[i].i = sigma[eta_loop][i][xyz][p][mm][q].i*w[i]*w[i];
      }

      for (i=0; i<4; i++){
	c[i] = Complex(0.0,0.0); 
	for (j=0; j<4; j++){
	  c[i].r += A[j*4+i]*b[j].r; 
	  c[i].i += A[j*4+i]*b[j].i; 
	}
      }

      for (omg=0; omg<Ng; omg++){
        omega0 = realomega_val[omg];
	sigma[eta_loop][omg][xyz][p][mm][q].r -= (c[0].r/(omega0*omega0)+c[1].r/omega0+c[2].r);  
	sigma[eta_loop][omg][xyz][p][mm][q].i -= (c[0].i/(omega0*omega0)+c[1].i/omega0+c[2].i);
      }
    }  
  } 

  free(A);

  /*************************************************************
                  calculate weighted sigma 
  *************************************************************/

  for (p=1; p<=pth; p++){

    if (p==1){
      ms = 0; me = p;
    }
    else if (p==2){
      ms = 1; me = 1;
    }               

    for (mm=ms; mm<=me; mm++){
      for (q=0; q<irr_count[p][mm]; q++){
	for (omg=0; omg<Ng; omg++){

	  omega0 = realomega_val[omg];

	  x0 = Egap[0]-(1.0e-5);
	  f1 = 0.5*(erf((1.0e+5)*(omega0-x0))+1.0);
	  f0 = 1.0 - f1; 

	  sigma[0][omg][0][p][mm][q].r = f0*sigma[0][omg][0][p][mm][q].r 
	                               + f1*sigma[1][omg][0][p][mm][q].r;
	  sigma[0][omg][1][p][mm][q].r = f0*sigma[0][omg][1][p][mm][q].r
	                               + f1*sigma[1][omg][1][p][mm][q].r;
	  sigma[0][omg][2][p][mm][q].r = f0*sigma[0][omg][2][p][mm][q].r
	                               + f1*sigma[1][omg][2][p][mm][q].r;

	  sigma[0][omg][0][p][mm][q].i = f0*sigma[0][omg][0][p][mm][q].i 
	                               + f1*sigma[1][omg][0][p][mm][q].i;
	  sigma[0][omg][1][p][mm][q].i = f0*sigma[0][omg][1][p][mm][q].i 
	                               + f1*sigma[1][omg][1][p][mm][q].i;
	  sigma[0][omg][2][p][mm][q].i = f0*sigma[0][omg][2][p][mm][q].i 
	                               + f1*sigma[1][omg][2][p][mm][q].i;
				  
	} // omg 
      } // q      
    } // mm
  } // p

  /*************************************************************
                  save the computational results
  *************************************************************/

  if (myid==Host_ID){

    /* save 2nd order DC current tensors */

    save_dc_noncol(fname,Ng,sigma,irr_count,realomega_val);
  }

  /* show elapsed time */

  if (measure_time){
    printf("myid=%2d time: %10.6f %10.6f %10.6f %10.6f\n",myid,time0,time1,time2,time3);
  }

  /*************************************************************
                    freeing of arrays
  *************************************************************/

  free(work_array);

  for (p=0; p<(pth+1); p++){
    for (m=0; m<(p+1); m++){
      for (q=0; q<irr_count[p][m]; q++){
	free(rho[p][m][q]);
      }
      free(rho[p][m]);
    }
    free(rho[p]);
  }
  free(rho);

  for (eta_loop=0; eta_loop<2; eta_loop++){
    for (omg=0; omg<Ng; omg++){
      for (xyz=0; xyz<3; xyz++){
	for (p=0; p<(pth+1); p++){
	  for (m=0; m<(p+1); m++){
	    free(sigma[eta_loop][omg][xyz][p][m]);
	  }           
          free(sigma[eta_loop][omg][xyz][p]);
	}
        free(sigma[eta_loop][omg][xyz]);
      }
      free(sigma[eta_loop][omg]);
    }
    free(sigma[eta_loop]);
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

  free(realomega_val);

  for (p=0; p<=pth; p++){
    for (m=0; m<=p; m++){
      for (q=0; q<irr_count[p][m]; q++){
        free(irr_list[p][m][q]);
      }
      free(irr_list[p][m]);
    }
    free(irr_list[p]);
  }  
  free(irr_list);

  for (p=0; p<=pth; p++){
    free(irr_count[p]);
  }
  free(irr_count);
}




void Input_para(char *file)
{
  char *s_vec[20];
  int i_vec[60],i_vec2[60];

  /* open the input file */

  if (input_open(file)==0){
    MPI_Finalize(); 
    exit(0);
  }

  /* file name of output files */

  input_string("out.filename",filename,"out");
  
  /* kgrids */

  i_vec2[0] = 11;
  i_vec2[1] = 11;
  i_vec2[2] = 11;
  input_intv("Kgrids",3,i_vec,i_vec2);
  Nk1 = i_vec[0];
  Nk2 = i_vec[1];
  Nk3 = i_vec[2];

  /* range of hbar*omega in eV */

  input_double("omega.min",&omega_min,(double)0.01);
  input_double("omega.max",&omega_max,(double)8.0);
  omega_min = omega_min/eV2Hartree;
  omega_max = omega_max/eV2Hartree;   

  /* enforce_flag for output for small omega region */

  input_logical("output.allrange",&output_allrange,1); // default=on

  /* energy grid */
  
  input_int("omega.grid",&Ng,(int)100);
  
  /* eta in eV */

  input_double("eta0",&eta_val[0],(double)(1.0e-4));
  input_double("eta",&eta_val[1],(double)0.1);
  eta_val[0] = eta_val[0]/eV2Hartree;
  eta_val[1] = eta_val[1]/eV2Hartree; 

  /* scissor correction in eV */

  input_double("scissor",&scissor,(double)0.0);
  scissor = scissor/eV2Hartree;

  /* close the input file */

  input_close();
}








void save_dc_col
(char *fname, int Ng, dcomplex *******sigma, int **irr_count,
 double *realomega_val)
{
  /*************************************************************
   save the direct current tensor in the 2nd order response
   within the collinear DFT.

   the unit of direct current:
   10^{-6}*[C^3 J^{-2} s^{-1}] = [micro_A V^{-2}]
  *************************************************************/

  int i,xyz,q,spin,omg,p,m;
  FILE *fp_dc;
  double omega0;
  char fname_dc[2][YOUSO10],fname_dc_sum[YOUSO10];
  double cd_factor = 8.945280419; // 10^{-6}*[C^3 J^{-2} s^{-1}]
  char buf1[fp_bsize];          /* setvbuf */

  printf("\n\n");

  p = 2; m = 1; 

  /* spin resolved quanties */  

  for (spin=0; spin<=SpinP_switch; spin++){ 

    /* set file names to be saved */

    sprintf(fname_dc[spin],"%s.dc%d",fname,spin);

    /* if the file can be opened, save the data */

    if ( (fp_dc = fopen(fname_dc[spin],"w")) != NULL ){

      setvbuf(fp_dc,buf1,_IOFBF,fp_bsize);  /* setvbuf */

      /* writing the header part */

      fprintf(fp_dc,"#\n");
      if (spin==0){
	fprintf(fp_dc,"# 2nd order DC current tensor for up-spin states\n");
      }
      else{
	fprintf(fp_dc,"# 2nd order DC current tensor for down-spin states\n");
      }
      fprintf(fp_dc,"# the unit: micro_A V^{-2}\n");
      fprintf(fp_dc,"# The calculation was perfomed with the collinear DFT.\n");
      fprintf(fp_dc,"# Egap (eV): %15.12f %15.12f\n",Egap[0]*eV2Hartree,Egap[1]*eV2Hartree);
      fprintf(fp_dc,"#\n");
      fprintf(fp_dc,"#  1:energy in eV\n");
      fprintf(fp_dc,"#  2:Re_xxx*,  3:Im_xxx*,  4:Re_xxy*,  5:Im_xxy*,  6:Re_xxz*,  7:Im_xxz*\n");
      fprintf(fp_dc,"#  8:Re_xyx*,  9:Im_xyx*, 10:Re_xyy*, 11:Im_xyy*, 12:Re_xyz*, 13:Im_xyz*\n");
      fprintf(fp_dc,"# 14:Re_xzx*, 15:Im_xzx*, 16:Re_xzy*, 17:Im_xzy*, 18:Re_xzz*, 19:Im_xzz*\n");
      fprintf(fp_dc,"# 20:Re_yxx*, 21:Im_yxx*, 22:Re_yxy*, 23:Im_yxy*, 24:Re_yxz*, 25:Im_yxz*\n");
      fprintf(fp_dc,"# 26:Re_yyx*, 27:Im_yyx*, 28:Re_yyy*, 29:Im_yyy*, 30:Re_yyz*, 31:Im_yyz*\n");
      fprintf(fp_dc,"# 32:Re_yzx*, 33:Im_yzx*, 34:Re_yzy*, 35:Im_yzy*, 36:Re_yzz*, 37:Im_yzz*\n");
      fprintf(fp_dc,"# 38:Re_zxx*, 39:Im_zxx*, 40:Re_zxy*, 41:Im_zxy*, 42:Re_zxz*, 43:Im_zxz*\n");
      fprintf(fp_dc,"# 44:Re_zyx*, 45:Im_zyx*, 46:Re_zyy*, 47:Im_zyy*, 48:Re_zyz*, 49:Im_zyz*\n");
      fprintf(fp_dc,"# 50:Re_zzx*, 51:Im_zzx*, 52:Re_zzy*, 53:Im_zzy*, 54:Re_zzz*, 55:Im_zzz*\n");
      fprintf(fp_dc,"#\n");

      /* loop for omg */

      for (omg=0; omg<Ng; omg++){

	omega0 = realomega_val[omg];

        if (output_allrange || ( (!output_allrange)&&(0.25*Egap[0]<omega0))){

	  fprintf(fp_dc,"%15.12f ",omega0*eV2Hartree);

	  for (xyz=0; xyz<3; xyz++){
	    for (q=0; q<irr_count[p][m]; q++){

	      fprintf(fp_dc,"%15.12f %15.12f ",
		      cd_factor*sigma[spin][0][omg][xyz][p][m][q].r*2.0,  // 2 is due to xx* and x*x
		      cd_factor*sigma[spin][0][omg][xyz][p][m][q].i*2.0);

	    } // q
	  } // xyz

	  fprintf(fp_dc,"\n");

	} // end of if..
      } // omg     

      fclose(fp_dc);

      /*show message */
      printf("Saved: %s\n",fname_dc[spin]);       

    } // end of if 

    else{
      printf("Failed to save %s\n",fname_dc[spin]);
    } 

  } // spin 

  /**********************
     the sum over spin 
  *********************/

  /* set file names to be saved */

  sprintf(fname_dc_sum,"%s.dc",fname);

  /* if the file can be opened, save the data */

  if ( (fp_dc = fopen(fname_dc_sum,"w")) != NULL ){

    /* writing the header part */

    fprintf(fp_dc,"#\n");
    fprintf(fp_dc,"# 2nd order DC current tensor contributed by the sum of the up and down spin states\n");
    fprintf(fp_dc,"# the unit: micro_A V^{-2}\n");
    fprintf(fp_dc,"# The calculation was perfomed with the collinear DFT.\n");
    fprintf(fp_dc,"# Egap (eV): %15.12f %15.12f\n",Egap[0]*eV2Hartree,Egap[1]*eV2Hartree);
    fprintf(fp_dc,"#\n");
    fprintf(fp_dc,"#  1:energy in eV\n");
    fprintf(fp_dc,"#  2:Re_xxx*,  3:Im_xxx*,  4:Re_xxy*,  5:Im_xxy*,  6:Re_xxz*,  7:Im_xxz*\n");
    fprintf(fp_dc,"#  8:Re_xyx*,  9:Im_xyx*, 10:Re_xyy*, 11:Im_xyy*, 12:Re_xyz*, 13:Im_xyz*\n");
    fprintf(fp_dc,"# 14:Re_xzx*, 15:Im_xzx*, 16:Re_xzy*, 17:Im_xzy*, 18:Re_xzz*, 19:Im_xzz*\n");
    fprintf(fp_dc,"# 20:Re_yxx*, 21:Im_yxx*, 22:Re_yxy*, 23:Im_yxy*, 24:Re_yxz*, 25:Im_yxz*\n");
    fprintf(fp_dc,"# 26:Re_yyx*, 27:Im_yyx*, 28:Re_yyy*, 29:Im_yyy*, 30:Re_yyz*, 31:Im_yyz*\n");
    fprintf(fp_dc,"# 32:Re_yzx*, 33:Im_yzx*, 34:Re_yzy*, 35:Im_yzy*, 36:Re_yzz*, 37:Im_yzz*\n");
    fprintf(fp_dc,"# 38:Re_zxx*, 39:Im_zxx*, 40:Re_zxy*, 41:Im_zxy*, 42:Re_zxz*, 43:Im_zxz*\n");
    fprintf(fp_dc,"# 44:Re_zyx*, 45:Im_zyx*, 46:Re_zyy*, 47:Im_zyy*, 48:Re_zyz*, 49:Im_zyz*\n");
    fprintf(fp_dc,"# 50:Re_zzx*, 51:Im_zzx*, 52:Re_zzy*, 53:Im_zzy*, 54:Re_zzz*, 55:Im_zzz*\n");
    fprintf(fp_dc,"#\n");

    /* loop for omg */

    for (omg=0; omg<Ng; omg++){

      omega0 = realomega_val[omg];

      if (output_allrange || (!output_allrange&&(0.25*Egap[0]<omega0))){

	fprintf(fp_dc,"%15.12f ",omega0*eV2Hartree);

	for (xyz=0; xyz<3; xyz++){
          for (q=0; q<irr_count[p][m]; q++){

	    if (SpinP_switch==0){  
	      fprintf(fp_dc,"%15.12f %15.12f ",
		      cd_factor*sigma[0][0][omg][xyz][p][m][q].r*4.0,
	              cd_factor*sigma[0][0][omg][xyz][p][m][q].i*4.0);  
	    }
	    else{
              // factor of 2 is due to x*x and xx*
	      fprintf(fp_dc,"%15.12f %15.12f ",
		      cd_factor*(sigma[0][0][omg][xyz][p][m][q].r+sigma[1][0][omg][xyz][p][m][q].r)*2.0,
		      cd_factor*(sigma[0][0][omg][xyz][p][m][q].i+sigma[1][0][omg][xyz][p][m][q].i)*2.0);
	    }

	  } // q
	} // xyz

	fprintf(fp_dc,"\n");

      } // end of if.. 
    } // omg     

    fclose(fp_dc);

    /*show message */
    printf("Saved: %s\n",fname_dc_sum);       

  } // end of if 

  else{
    printf("Failed to save %s\n",fname_dc_sum);
  } 
}



void save_dc_noncol
(char *fname, int Ng, dcomplex ******sigma, int **irr_count,
 double *realomega_val)
{
  /*************************************************************
   save the direct current tensor in the 2nd order response
   within the non-collinear DFT.

   the unit of direct current:
   10^{-6}*[C^3 J^{-2} s^{-1}] = [micro_A V^{-2}]
  *************************************************************/

  int xyz,q,omg,p,m;
  FILE *fp_dc;
  char fname_dc[YOUSO10];
  double cd_factor = 8.945280419; // 10^{-6}*[C^3 J^{-2} s^{-1}]
  double omega0;
  char buf1[fp_bsize];          /* setvbuf */

  printf("\n\n");

  p = 2; m = 1;

  /* set file names to be saved */

  sprintf(fname_dc,"%s.dc",fname);

  /* if the file can be opened, save the data */

  if ( (fp_dc = fopen(fname_dc,"w")) != NULL ){

    setvbuf(fp_dc,buf1,_IOFBF,fp_bsize);  /* setvbuf */

    /* writing the header part */

    fprintf(fp_dc,"#\n");
    fprintf(fp_dc,"# 2nd order DC current tensor\n");
    fprintf(fp_dc,"# the unit: micro_A V^{-2}\n");
    fprintf(fp_dc,"# The calculation was perfomed with the non-collinear DFT.\n");
    fprintf(fp_dc,"# Egap (eV): %15.12f\n",Egap[0]*eV2Hartree);
    fprintf(fp_dc,"#\n");
    fprintf(fp_dc,"#  1:energy in eV\n");
    fprintf(fp_dc,"#  2:Re_xxx*,  3:Im_xxx*,  4:Re_xxy*,  5:Im_xxy*,  6:Re_xxz*,  7:Im_xxz*\n");
    fprintf(fp_dc,"#  8:Re_xyx*,  9:Im_xyx*, 10:Re_xyy*, 11:Im_xyy*, 12:Re_xyz*, 13:Im_xyz*\n");
    fprintf(fp_dc,"# 14:Re_xzx*, 15:Im_xzx*, 16:Re_xzy*, 17:Im_xzy*, 18:Re_xzz*, 19:Im_xzz*\n");
    fprintf(fp_dc,"# 20:Re_yxx*, 21:Im_yxx*, 22:Re_yxy*, 23:Im_yxy*, 24:Re_yxz*, 25:Im_yxz*\n");
    fprintf(fp_dc,"# 26:Re_yyx*, 27:Im_yyx*, 28:Re_yyy*, 29:Im_yyy*, 30:Re_yyz*, 31:Im_yyz*\n");
    fprintf(fp_dc,"# 32:Re_yzx*, 33:Im_yzx*, 34:Re_yzy*, 35:Im_yzy*, 36:Re_yzz*, 37:Im_yzz*\n");
    fprintf(fp_dc,"# 38:Re_zxx*, 39:Im_zxx*, 40:Re_zxy*, 41:Im_zxy*, 42:Re_zxz*, 43:Im_zxz*\n");
    fprintf(fp_dc,"# 44:Re_zyx*, 45:Im_zyx*, 46:Re_zyy*, 47:Im_zyy*, 48:Re_zyz*, 49:Im_zyz*\n");
    fprintf(fp_dc,"# 50:Re_zzx*, 51:Im_zzx*, 52:Re_zzy*, 53:Im_zzy*, 54:Re_zzz*, 55:Im_zzz*\n");
    fprintf(fp_dc,"#\n");

    /* loop for omg */

    for (omg=0; omg<Ng; omg++){

      omega0 = realomega_val[omg];

      if (output_allrange || ( (!output_allrange)&&(0.25*Egap[0]<omega0))){

	fprintf(fp_dc,"%15.12f ",omega0*eV2Hartree);

	for (xyz=0; xyz<3; xyz++){
          for (q=0; q<irr_count[p][m]; q++){

	    fprintf(fp_dc,"%15.12f %15.12f ",
		    cd_factor*sigma[0][omg][xyz][p][m][q].r*2.0,  // 2 is due to xx* and x*x
		    cd_factor*sigma[0][omg][xyz][p][m][q].i*2.0);

	  } // q
	} // xyz

	fprintf(fp_dc,"\n");

      } // end of if..
    } // omg     

    fclose(fp_dc);

    /*show message */
    printf("Saved: %s\n",fname_dc);       

  } // end of if 

  else{
    printf("Failed to save %s\n",fname_dc);
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



void Calc_Momentum_Matrix( int spin, double k1, double k2, double k3, 
                           dcomplex *WF, double *eval, dcomplex **P )
  /******************************************************************
    The routine calculates the momentum matrix elements represented 
    by eigenstates using a direct derivative calculation.
  ******************************************************************/
{
  int i,j,k,m,n,xyz_i,l1,l2,l3,Rn,xyz;
  int GA_AN,LB_AN,GB_AN,tnoA,tnoB,Anum,Bnum;
  double si,co,kRn,d,en,em,x,y,z,s,sx,sy,sz;
  dcomplex ctmp1,ctmp2,ctmp3,csum,**dHk,**dSk;
  dcomplex p0,p1,p2;
  dcomplex *work_array;
  dcomplex alpha,beta;

  /* allocation of array */

  work_array = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));

  /******************************************
       calculate P_{x,y,z}(k) w.r.t. PAO 
  ******************************************/

  /* initialize P */

  for (i=0; i<MatDim*MatDim; i++){
    P[0][i] = Complex(0.0,0.0);       
    P[1][i] = Complex(0.0,0.0);       
    P[2][i] = Complex(0.0,0.0);       
  }

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

	  P[0][(Bnum+j)*MatDim+Anum+i].r += sx*co;
	  P[0][(Bnum+j)*MatDim+Anum+i].i += sx*si;
	  P[1][(Bnum+j)*MatDim+Anum+i].r += sy*co;
	  P[1][(Bnum+j)*MatDim+Anum+i].i += sy*si;
	  P[2][(Bnum+j)*MatDim+Anum+i].r += sz*co;
	  P[2][(Bnum+j)*MatDim+Anum+i].i += sz*si;

	} // j
      } // i
    } // LB_AN
  } // GA_AN

  /*************************************************
    calculate P_{x,y,z}(k) w.r.t. Bloch functions
  *************************************************/

  m = MatDim; n = MatDim; k = MatDim;
  beta = Complex(0.0,0.0); 

  for (xyz=0; xyz<3; xyz++){

    alpha = Complex(1.0, 0.0);
    zgemm_("C","N", &m, &n, &k, &alpha, WF, &m, 
  	    P[xyz], &k, &beta, work_array, &m);

    alpha = Complex(0.0, -1.0); /* multiply (-i). P becomes <psi_mu | -i nabla | psi_nu> */
    zgemm_("N","N", &m, &n, &k, &alpha, work_array, &m, 
  	    WF, &k, &beta, P[xyz], &m);
  }  

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



void Calc_Momentum_Matrix_NC( double k1, double k2, double k3, 
                              dcomplex *WF, double *eval, dcomplex **P )
  /******************************************************************
    The routine calculates the momentum matrix elements represented 
    by eigenstates using a direct derivative calculation.
  ******************************************************************/
{
  int i,j,k,m,n,xyz_i,l1,l2,l3,Rn,xyz;
  int GA_AN,LB_AN,GB_AN,tnoA,tnoB,Anum,Bnum;
  double si,co,kRn,d,en,em,x,y,z,s,sx,sy,sz;
  dcomplex ctmp1,ctmp2,ctmp3,csum,**dHk,**dSk;
  dcomplex p0,p1,p2;
  dcomplex *work_array;
  dcomplex alpha,beta;

  /* allocation of array */

  work_array = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim2*MatDim2));

  /******************************************
       calculate P_{x,y,z}(k) w.r.t. PAO 
  ******************************************/

  /* initialize P */

  for (i=0; i<MatDim2*MatDim2; i++){
    P[0][i] = Complex(0.0,0.0);       
    P[1][i] = Complex(0.0,0.0);       
    P[2][i] = Complex(0.0,0.0);       
  }

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

	  P[0][(Bnum+j)*MatDim2+Anum+i].r += sx*co;
	  P[0][(Bnum+j)*MatDim2+Anum+i].i += sx*si;
	  P[1][(Bnum+j)*MatDim2+Anum+i].r += sy*co;
	  P[1][(Bnum+j)*MatDim2+Anum+i].i += sy*si;
	  P[2][(Bnum+j)*MatDim2+Anum+i].r += sz*co;
	  P[2][(Bnum+j)*MatDim2+Anum+i].i += sz*si;

	} // j
      } // i
    } // LB_AN
  } // GA_AN

  /* set P22_{x,y,z}(k) */

  for (i=0; i<MatDim; i++){
    for (j=0; j<MatDim; j++){
      P[0][(MatDim+j)*MatDim2+MatDim+i] = P[0][j*MatDim2+i];
      P[1][(MatDim+j)*MatDim2+MatDim+i] = P[1][j*MatDim2+i];
      P[2][(MatDim+j)*MatDim2+MatDim+i] = P[2][j*MatDim2+i];
    }
  }

  /*************************************************
    calculate P_{x,y,z}(k) w.r.t. Bloch functions
  *************************************************/

  m = MatDim2; n = MatDim2; k = MatDim2;
  beta = Complex(0.0,0.0); 

  for (xyz=0; xyz<3; xyz++){

    alpha = Complex(1.0, 0.0);
    zgemm_("C","N", &m, &n, &k, &alpha, WF, &m, 
  	    P[xyz], &k, &beta, work_array, &m);

    alpha = Complex(0.0, -1.0); /* multiply (-i). P becomes <psi_mu | -i nabla | psi_nu> */
    zgemm_("N","N", &m, &n, &k, &alpha, work_array, &m, 
  	    WF, &k, &beta, P[xyz], &m);
  }  

  /* preserving the Hermicity  */

  for (i=0; i<3; i++){
    for (m=0; m<MatDim2; m++){
      for (n=m; n<MatDim2; n++){

        ctmp1 = P[i][n*MatDim2+m];
        ctmp2 = P[i][m*MatDim2+n];
        ctmp3.r = 0.5*(ctmp1.r + ctmp2.r);
        ctmp3.i = 0.5*(ctmp1.i - ctmp2.i);

        P[i][n*MatDim2+m] = ctmp3;
        P[i][m*MatDim2+n].r = ctmp3.r;
        P[i][m*MatDim2+n].i =-ctmp3.i;
      }
    }   
  }

  /* freeing of array */
  free(work_array);
}




void Calc_r_VNL_Matrix( int spin, double k1, double k2, double k3, 
                        dcomplex *WF, dcomplex **SCI, dcomplex **r_VNL )
{
  int i,j,m,n,k,l1,l2,l3,GA_AN,GB_AN,LB_AN;
  int tnoA,tnoB,Rn,Anum,Bnum;
  double co,si,s,sx,sy,sz,Rx,Ry,Rz,kRn;
  dcomplex **Ri,*work_array;
  dcomplex alpha,beta;

  /* allocation of arrays */  
  
  Ri = (dcomplex**)malloc(sizeof(dcomplex*)*3);
  for (i=0; i<3; i++){
    Ri[i] = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));
    for (j=0; j<(MatDim*MatDim); j++){
      Ri[i][j] = Complex(0.0,0.0);
    }
  }

  work_array = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));

  /************************************************************
                      calculate [VNL, Ri]   
  ************************************************************/

  /* calculate Ri w.r.t chi */  

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

	  Ri[0][(Bnum+j)*MatDim+Anum+i].r += (Rx*s-sx)*co;
	  Ri[0][(Bnum+j)*MatDim+Anum+i].i += (Rx*s-sx)*si;
          
	  Ri[1][(Bnum+j)*MatDim+Anum+i].r += (Ry*s-sy)*co;
	  Ri[1][(Bnum+j)*MatDim+Anum+i].i += (Ry*s-sy)*si;

	  Ri[2][(Bnum+j)*MatDim+Anum+i].r += (Rz*s-sz)*co;
	  Ri[2][(Bnum+j)*MatDim+Anum+i].i += (Rz*s-sz)*si;
	}
      }

    } // LB_AN
  } // GA_AN

  /* calculate Ri w.r.t Bloch functions */  
  /*           C^dag x Ri_chi x C -> Ri */  

  alpha = Complex(1.0, 0.0);
  beta = Complex(0.0,0.0); 
  m = MatDim; n = MatDim; k = MatDim;
  
  for (i=0; i<3; i++){

    zgemm_("C","N", &m, &n, &k, &alpha, WF, &m, Ri[i], 
  	   &k, &beta, work_array, &m);

    zgemm_("N","N", &m, &n, &k, &alpha, work_array, &m, WF, 
  	   &k, &beta, Ri[i], &m);
  }

  /* calculate [VNL, Ri] */

  m = MatDim; n = MatDim; k = MatDim;
  
  for (i=0; i<3; i++){

    alpha = Complex(1.0, 0.0);
    beta = Complex(0.0,0.0); 

    zgemm_("N","N", &m, &n, &k, &alpha, SCI[0], &m, Ri[i], 
  	   &k, &beta, r_VNL[i], &m);

    alpha = Complex(-1.0, 0.0);
    beta = Complex(1.0,0.0); 

    zgemm_("N","N", &m, &n, &k, &alpha, Ri[i], &m, SCI[0], 
  	   &k, &beta, r_VNL[i], &m);
  }

  /************************************************************
               calculate -iC^dag(dki S)C x VNL
  ************************************************************/

  /* initialize Ri */

  for (i=0; i<3; i++){
    for (j=0; j<(MatDim*MatDim); j++){
      Ri[i][j] = Complex(0.0,0.0);
    }
  }

  /* calculate -i * dki S */

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

	  Ri[0][(Bnum+j)*MatDim+Anum+i].r += Rx*s*co;
	  Ri[0][(Bnum+j)*MatDim+Anum+i].i += Rx*s*si;
          
	  Ri[1][(Bnum+j)*MatDim+Anum+i].r += Ry*s*co;
	  Ri[1][(Bnum+j)*MatDim+Anum+i].i += Ry*s*si;

	  Ri[2][(Bnum+j)*MatDim+Anum+i].r += Rz*s*co;
	  Ri[2][(Bnum+j)*MatDim+Anum+i].i += Rz*s*si;
	} // j
      } // i

    } // LB_AN
  } // GA_AN

  /* C^dag * (-i * dki S) * C * VNL */

  alpha = Complex(1.0, 0.0);
  beta = Complex(0.0,0.0); 
  m = MatDim; n = MatDim; k = MatDim;

  for (i=0; i<3; i++){

    alpha = Complex(1.0, 0.0);
    beta = Complex(0.0,0.0); 

    /* C^dag * (-i * dki S) */
    zgemm_("C","N", &m, &n, &k, &alpha, WF, &m, Ri[i], 
  	   &k, &beta, work_array, &m);

    /* C^dag * (-i * dki S) * C */
    zgemm_("N","N", &m, &n, &k, &alpha, work_array, &m, WF, 
  	   &k, &beta, Ri[i], &m);

    alpha = Complex(1.0, 0.0);
    beta = Complex(1.0,0.0); 

    /* C^dag * (-i * dki S) * C * VNL */
    zgemm_("N","N", &m, &n, &k, &alpha, Ri[i], &m, SCI[0], 
  	   &k, &beta, r_VNL[i], &m);

  } // i

  /************************************************************
               add iC^dag(dki VNL^{chi})C
  ************************************************************/

  for (i=0; i<3; i++){
    for (j=0; j<MatDim*MatDim; j++){
      r_VNL[i][j].r -= SCI[i+1][j].i;
      r_VNL[i][j].r += SCI[i+1][j].r;
    }
  }  

  /* freeing of arrays */  
  
  for (i=0; i<3; i++){
    free(Ri[i]);
  }
  free(Ri);

  free(work_array);
}



void Calc_Scissor_Matrix( int spin, double k1, double k2, double k3, 
                          double scissor, dcomplex *WF, dcomplex **SCI )
  /******************************************************************
    The routine calculates the scissor matrix elements represented 
    by the Bloch states
  ******************************************************************/
{
  int i,j,m,n,k,GA_AN,LB_AN,GB_AN,Rn,tnoA,tnoB,Anum,Bnum,l1,l2,l3;
  double co,si,kRn,s,Rx,Ry,Rz;
  dcomplex **SumSk,**OLPBS,ctmp,alpha,beta;

  /* allocation of arrays */  

  OLPBS = (dcomplex**)malloc(sizeof(dcomplex*)*4);
  for (i=0; i<4; i++){
    OLPBS[i] = (dcomplex*)malloc(sizeof(dcomplex)*(MatDim*MatDim));
    for (j=0; j<(MatDim*MatDim); j++){
      OLPBS[i][j] = Complex(0.0,0.0);
    }
  }

  SumSk = (dcomplex**)malloc(sizeof(dcomplex*)*4);
  for (i=0; i<4; i++){
    SumSk[i] = (dcomplex*)malloc(sizeof(dcomplex)*MatDim);
  }

  /* initiaize SCI */  

  for (i=0; i<(MatDim*MatDim); i++){
    SCI[0][i] = Complex(0.0,0.0);
    SCI[1][i] = Complex(0.0,0.0);
    SCI[2][i] = Complex(0.0,0.0);
    SCI[3][i] = Complex(0.0,0.0);
  } 

  /* calculate OLPBS */  

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

	  OLPBS[0][(Bnum+j)*MatDim+Anum+i].r += s*co;
	  OLPBS[0][(Bnum+j)*MatDim+Anum+i].i += s*si;

	  OLPBS[1][(Bnum+j)*MatDim+Anum+i].r -= Rx*s*si;
	  OLPBS[1][(Bnum+j)*MatDim+Anum+i].i += Rx*s*co;
          
	  OLPBS[2][(Bnum+j)*MatDim+Anum+i].r -= Ry*s*si;
	  OLPBS[2][(Bnum+j)*MatDim+Anum+i].i += Ry*s*co;

	  OLPBS[3][(Bnum+j)*MatDim+Anum+i].r -= Rz*s*si;
	  OLPBS[3][(Bnum+j)*MatDim+Anum+i].i += Rz*s*co;
	}
      }

    } // LB_AN
  } // GA_AN

  /*
  printf("SCI V\n");
  for (i=0; i<10; i++){
    for (j=0; j<10; j++){
      printf("%7.4f ",OLPBS[1][j*MatDim+i].r);
    }
    printf("\n");
  }
  */

  /* calculate <Bloch sum|P|Bloch sum> */

  for (m=0; m<No[spin]; m++){

    /* calculate SumSk */  

    for (i=0; i<MatDim; i++){
      SumSk[0][i] = Complex(0.0,0.0);
      SumSk[1][i] = Complex(0.0,0.0);
      SumSk[2][i] = Complex(0.0,0.0);
      SumSk[3][i] = Complex(0.0,0.0);
    }

    for (i=0; i<MatDim; i++){
      for (j=0; j<MatDim; j++){

	SumSk[0][i].r += WF[m*MatDim+j].r*OLPBS[0][j*MatDim+i].r - WF[m*MatDim+j].i*OLPBS[0][j*MatDim+i].i;
	SumSk[0][i].i += WF[m*MatDim+j].r*OLPBS[0][j*MatDim+i].i + WF[m*MatDim+j].i*OLPBS[0][j*MatDim+i].r;

	SumSk[1][i].r += WF[m*MatDim+j].r*OLPBS[1][j*MatDim+i].r - WF[m*MatDim+j].i*OLPBS[1][j*MatDim+i].i;
	SumSk[1][i].i += WF[m*MatDim+j].r*OLPBS[1][j*MatDim+i].i + WF[m*MatDim+j].i*OLPBS[1][j*MatDim+i].r;

	SumSk[2][i].r += WF[m*MatDim+j].r*OLPBS[2][j*MatDim+i].r - WF[m*MatDim+j].i*OLPBS[2][j*MatDim+i].i;
	SumSk[2][i].i += WF[m*MatDim+j].r*OLPBS[2][j*MatDim+i].i + WF[m*MatDim+j].i*OLPBS[2][j*MatDim+i].r;

	SumSk[3][i].r += WF[m*MatDim+j].r*OLPBS[3][j*MatDim+i].r - WF[m*MatDim+j].i*OLPBS[3][j*MatDim+i].i;
	SumSk[3][i].i += WF[m*MatDim+j].r*OLPBS[3][j*MatDim+i].i + WF[m*MatDim+j].i*OLPBS[3][j*MatDim+i].r;

      } // j
    } // i

    /*
  printf("SumSk\n");
  for (i=0; i<10; i++){
    printf("i=%d %7.4f %7.4f  %7.4f %7.4f\n",i,SumSk[0][i].r,SumSk[0][i].i,SumSk[1][i].r,SumSk[1][i].i);
  }
    */

    /* calculate SCI w.r.t Bloch-sum PAO basis */  

    for (i=0; i<MatDim; i++){
      for (j=0; j<MatDim; j++){

        SCI[0][j*MatDim+i].r += SumSk[0][i].r*SumSk[0][j].r + SumSk[0][i].i*SumSk[0][j].i;  
        SCI[0][j*MatDim+i].i += SumSk[0][i].i*SumSk[0][j].r - SumSk[0][i].r*SumSk[0][j].i;

        SCI[1][j*MatDim+i].r += SumSk[1][i].r*SumSk[0][j].r + SumSk[1][i].i*SumSk[0][j].i  
                               +SumSk[0][i].r*SumSk[1][j].r + SumSk[0][i].i*SumSk[1][j].i;  
        SCI[1][j*MatDim+i].i += SumSk[1][i].i*SumSk[0][j].r - SumSk[1][i].r*SumSk[0][j].i
                               +SumSk[0][i].i*SumSk[1][j].r - SumSk[0][i].r*SumSk[1][j].i;

        SCI[2][j*MatDim+i].r += SumSk[2][i].r*SumSk[0][j].r + SumSk[2][i].i*SumSk[0][j].i  
                               +SumSk[0][i].r*SumSk[2][j].r + SumSk[0][i].i*SumSk[2][j].i;  
        SCI[2][j*MatDim+i].i += SumSk[2][i].i*SumSk[0][j].r - SumSk[2][i].r*SumSk[0][j].i
                               +SumSk[0][i].i*SumSk[2][j].r - SumSk[0][i].r*SumSk[2][j].i;

        SCI[3][j*MatDim+i].r += SumSk[3][i].r*SumSk[0][j].r + SumSk[3][i].i*SumSk[0][j].i  
                               +SumSk[0][i].r*SumSk[3][j].r + SumSk[0][i].i*SumSk[3][j].i;  
        SCI[3][j*MatDim+i].i += SumSk[3][i].i*SumSk[0][j].r - SumSk[3][i].r*SumSk[0][j].i
                               +SumSk[0][i].i*SumSk[3][j].r - SumSk[0][i].r*SumSk[3][j].i;

      } // j
    } // i      
  } // m

  /* calculate the full matrix of SCI */

  for (i=0; i<MatDim; i++){
    for (j=0; j<MatDim; j++){

      SCI[0][j*MatDim+i].r = scissor*(OLPBS[0][j*MatDim+i].r - SCI[0][j*MatDim+i].r); 
      SCI[0][j*MatDim+i].i = scissor*(OLPBS[0][j*MatDim+i].i - SCI[0][j*MatDim+i].i); 
  
      SCI[1][j*MatDim+i].r = scissor*(OLPBS[1][j*MatDim+i].r - SCI[1][j*MatDim+i].r); 
      SCI[1][j*MatDim+i].i = scissor*(OLPBS[1][j*MatDim+i].i - SCI[1][j*MatDim+i].i); 

      SCI[2][j*MatDim+i].r = scissor*(OLPBS[2][j*MatDim+i].r - SCI[2][j*MatDim+i].r); 
      SCI[2][j*MatDim+i].i = scissor*(OLPBS[2][j*MatDim+i].i - SCI[2][j*MatDim+i].i); 

      SCI[3][j*MatDim+i].r = scissor*(OLPBS[3][j*MatDim+i].r - SCI[3][j*MatDim+i].r); 
      SCI[3][j*MatDim+i].i = scissor*(OLPBS[3][j*MatDim+i].i - SCI[3][j*MatDim+i].i); 

    }
  }       

  /* representing SCI w.r.t Bloch states */

  alpha = Complex(1.0, 0.0);
  beta = Complex(0.0,0.0); 
  m = MatDim; n = MatDim; k = MatDim;

  for (i=0; i<4; i++){

    /* C^dag * SCI */
    zgemm_("C","N", &m, &n, &k, &alpha, WF, &m, SCI[i], 
  	   &k, &beta, OLPBS[i], &m);

    /* C^dag * SCI * C */
    zgemm_("N","N", &m, &n, &k, &alpha, OLPBS[i], &m, WF, 
  	   &k, &beta, SCI[i], &m);

  } // i

  /* freeing of arrays */  

  for (i=0; i<4; i++){
    free(OLPBS[i]);
  }
  free(OLPBS);

  for (i=0; i<4; i++){
    free(SumSk[i]);
  }
  free(SumSk);

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



void Calc_Hk_Sk_NC(dcomplex *Hk, dcomplex *Sk, double k1, double k2, double k3)
{
  int i,j,GA_AN,LB_AN,GB_AN,Rn,tnoA,tnoB,Anum,Bnum,l1,l2,l3;
  double co,si,kRn,s,h0,h1,h2,h3,ih0,ih1,ih2;

  /* initiaize Hk and Sk */  

  for (i=0; i<(MatDim2*MatDim2); i++){
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

	  h0 = Hks[0][GA_AN][LB_AN][i][j];
	  h1 = Hks[1][GA_AN][LB_AN][i][j];
	  h2 = Hks[2][GA_AN][LB_AN][i][j];
	  h3 = Hks[3][GA_AN][LB_AN][i][j];
          ih0 = iHks[0][GA_AN][LB_AN][i][j];
          ih1 = iHks[1][GA_AN][LB_AN][i][j];
          ih2 = iHks[2][GA_AN][LB_AN][i][j];

	  Hk[(Bnum+j)*MatDim2+Anum+i].r += co*h0 - si*ih0;  // H11r
	  Hk[(Bnum+j)*MatDim2+Anum+i].i += si*h0 + co*ih0;  // H11i

	  Hk[(MatDim+Bnum+j)*MatDim2+MatDim+Anum+i].r += co*h1 - si*ih1;  // H22r
	  Hk[(MatDim+Bnum+j)*MatDim2+MatDim+Anum+i].i += si*h1 + co*ih1;  // H22i

	  Hk[(MatDim+Bnum+j)*MatDim2+Anum+i].r += co*h2 - si*(h3+ih2);  // H12r
	  Hk[(MatDim+Bnum+j)*MatDim2+Anum+i].i += si*h2 + co*(h3+ih2);  // H12i

	  s = OLP[GA_AN][LB_AN][i][j];
	  Sk[(Bnum+j)*MatDim2+Anum+i].r += s*co;
	  Sk[(Bnum+j)*MatDim2+Anum+i].i += s*si;
	}
      }

    } // LB_AN
  } // GA_AN

  /* set H21 and S22 */

  for (i=0; i<MatDim; i++){
    for (j=0; j<MatDim; j++){
      Hk[i*MatDim2+MatDim+j].r = Hk[(MatDim+j)*MatDim2+i].r; 
      Hk[i*MatDim2+MatDim+j].i =-Hk[(MatDim+j)*MatDim2+i].i; 
      Sk[(MatDim+j)*MatDim2+MatDim+i] = Sk[j*MatDim2+i];
    }
  }

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
    MatDim2 = 2*MatDim; 

    if (SpinP_switch==0){ 
      No[0] = Valence_Electrons/2;
      No[1] = Valence_Electrons/2;
      Nuo[0] = MatDim - No[0];
      Nuo[1] = MatDim - No[1];
    }
    else if (SpinP_switch==1){
      No[0] = (Valence_Electrons+2*(int)(floor(Total_SpinS+1.0e-4)))/2;
      No[1] = (Valence_Electrons-2*(int)(floor(Total_SpinS+1.0e-4)))/2;
      Nuo[0] = MatDim - No[0];
      Nuo[1] = MatDim - No[1];
    }
    else if (SpinP_switch==3){
      No[0] = Valence_Electrons;
      No[1] = 0;
      Nuo[0] = MatDim2 - No[0];
      Nuo[1] = 0;
    }

    if (No[0]<No[1])  { Nomax  = No[1];  } else { Nomax  = No[0];  }
    if (Nuo[0]<Nuo[1]){ Nuomax = Nuo[1]; } else { Nuomax = Nuo[0]; }
  }

  /***********************************************************
                 if (strcasecmp(mode,"free")==0)
  ***********************************************************/

  else if (strcasecmp(mode,"free")==0){

    free(MP);

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

	    k_op[i][j][k]    = 1;
	    k_op[ii][ij][ik] = 1;
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


void Calc_Inverse(int n, double *A)
{
  int info,lwork;
  int *ipiv;
  double *work;

  lwork = 8*n;
  ipiv = (int*) malloc(sizeof(int)*n);
  work = (double*)malloc(sizeof(double)*lwork);

  dgetrf_(&n,&n,A,&n,ipiv,&info);
  dgetri_(&n,A,&n,ipiv,work,&lwork,&info);

  free(ipiv);
  free(work);
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


/*
 * remove_one_and_get_pmq()
 *
 * Inputs
 * ------
 *   (p, m, q) : identifies the q-th irreducible combination in Table 1.1 order
 *              for the given (p,m). q is 0-based.
 *
 *   s         : symbol code to remove (0..5)
 *               0,1,2 -> x,y,z
 *               3,4,5 -> x*,y*,z*
 *
 * Outputs
 * -------
 *   (p2, m2, q2) : identifies the reduced combination after removing ONE occurrence
 *                 of symbol s, again in Table 1.1 order (0-based q2).
 *
 * Return
 * ------
 *   1 on success, 0 on failure.
 *
 * Example
 * -------
 *   (p,m,q) = (2,1,1) corresponds to "x y*"
 *   remove s=0 ("x") -> "y*"  => (p2,m2,q2) = (1,0,1)
 */

int remove_one_and_get_pmq(int p, int m, int q, int s,
                           int *p2, int *m2, int *q2)
{
  int **list_pm;
  int count_pm;
  const int *entry_q;
  int new_p,new_m,read_pos;
  int symbol_exists;
  int removed_once;
  int *reduced;
  int write_pos;
  int **list_new;
  int count_new;
  int new_q;

  if (p2 == NULL || m2 == NULL || q2 == NULL) return 0;

  *p2 = -1;
  *m2 = -1;
  *q2 = -1;

  if (p <= 0) return 0;
  if (m < 0 || m > p) return 0;
  if (q < 0) return 0;
  if (s < 0 || s > 5) return 0;

  /* Build Table-ordered irreducible list for (p,m) */

  list_pm = NULL;
  count_pm = 0;

  if (!init_irreducible_components_pm(p, m, &list_pm, &count_pm)) {
    return 0;
  }

  if (q >= count_pm) {
    free_list_2d(list_pm, count_pm);
    return 0;
  }

  entry_q = list_pm[q];

  /* Check that symbol s is present in the selected entry */

  symbol_exists = 0;
  for (read_pos = 0; read_pos < p; ++read_pos) {
    if (entry_q[read_pos] == s) {
      symbol_exists = 1;
      break;
    }
  }

  if (!symbol_exists) {
    free_list_2d(list_pm, count_pm);
    return 0;
  }

  /* Compute reduced (p2,m2) */

  new_p = p - 1;

  if (s < 3) {

    new_m = m - 1;            /* removing a non-star factor */
    if (new_m < 0) {
      free_list_2d(list_pm, count_pm);
      return 0;
    }
  } else {
    new_m = m;                /* removing a star factor */
  }

  /* Build reduced canonical array by deleting ONE occurrence of s */

  reduced = (int*)malloc(sizeof(int) * (size_t)new_p);
  if (reduced == NULL) {
    free_list_2d(list_pm, count_pm);
    return 0;
  }

  write_pos = 0;
  removed_once = 0;

  for (read_pos = 0; read_pos < p; ++read_pos) {

    int v = entry_q[read_pos];
    if (!removed_once && v == s) {
      removed_once = 1;
      continue;               /* skip exactly one s */
    }

    reduced[write_pos] = v;
    write_pos++;
  }

  if (write_pos != new_p) {
    free(reduced);
    free_list_2d(list_pm, count_pm);
    return 0;
  }

  /* Build Table-ordered list for (new_p,new_m) and map reduced[] to q2 */

  list_new = NULL;
  count_new = 0;

  if (!init_irreducible_components_pm(new_p, new_m, &list_new, &count_new)) {
    free(reduced);
    free_list_2d(list_pm, count_pm);
    return 0;
  }

  new_q = map_to_irreducible_pm(new_p, new_m, reduced, list_new, count_new);

  free(reduced);
  free_list_2d(list_pm, count_pm);
  free_list_2d(list_new, count_new);

  if (new_q < 0) return 0;

  *p2 = new_p;
  *m2 = new_m;
  *q2 = new_q;

  return 1;
}




/*
 * Irreducible combinations for nonlinear response at given (p, m),
 * ordered to match Table 1.1 exactly (as a list order, not just a set).
 *
 * Ordering rule (matches Table 1.1 for p<=4 and naturally extends):
 *   1) Enumerate the non-star multiset (length m) in the standard
 *      nondecreasing-sequence order over {x,y,z} (i.e., 0<=...<=2).
 *   2) For each non-star multiset in that order, enumerate the star multiset
 *      (length p-m) in the same standard order over {x*,y*,z*}.
 *   3) Output the concatenation [non-star..., star...] in canonical form.
 *

 * Encoding:
 *   non-star: 0,1,2 -> x,y,z
 *   star:     3,4,5 -> x*,y*,z*  (axis* = axis + 3)
 *
 * Canonical entry format (int entry[p]):
 *   entry[0..m-1]       : non-star axes in nondecreasing order (0..2)
 *   entry[m..p-1]       : star axes in nondecreasing order (3..5, i.e., (0..2)+3)
 *
 * Examples:
 *   p=2,m=1 (DC): xx*, xy*, xz*, yx*, yy*, yz*, zx*, zy*, zz*
 *   p=2,m=2 (SHG): x^2, xy, xz, y^2, yz, z^2  (as compressed notation)
 */


/* ---------- Small utilities ---------- */

static int cmp_int(const void *a, const void *b)

{
  int x = *(const int*)a;
  int y = *(const int*)b;
  return (x > y) - (x < y);
}

static const char* comp_name(int v)
{
  static const char* names[] = {"x","y","z","x*","y*","z*"};
  if (v < 0 || v > 5) return "?";
  return names[v];
}

static void print_entry_raw(const int *e, int p)
{
  int i;
  for (i = 0; i < p; ++i) {
    printf("%s", comp_name(e[i]));
    if (i != p - 1) printf(" ");
  }
}


/* Free list[count], each entry is malloc'ed int[p] */
static void free_list_2d(int **list, int count)
{
  int i;
  if (!list) return;
  for (i = 0; i < count; ++i) free(list[i]);
  free(list);
}

/* Compare two index arrays of length p for equality. */
static int equal_indices(const int *a, const int *b, int p)
{
  int i;
  for (i = 0; i < p; ++i) {
    if (a[i] != b[i]) return 0;
  }
  return 1;
}

/* ---------- Generate multisets: standard Table 1.1 ordering ---------- */
/*
 * Generate all nondecreasing axis sequences of length len from {0,1,2}.
 * This is exactly the "combinations with repetition" canonical ordering:
 *   for a0 in [0..2]
 *     for a1 in [a0..2]
 *       ...
 *
 * Output:
 *   out_list  : array of pointers, each points to an int[len] sorted nondecreasing
 *   out_count : number of sequences
 *
 * Special case len==0:
 *   out_count=1 and out_list[0]=NULL
 */

static int gen_axis_multisets_rec(int len, int depth, int start_axis,
                                  int *buffer, int ***out_list, int *out_count)
{
  int i;

  if (depth == len) {
    int *entry = NULL;
    if (len > 0) {
      entry = (int*)malloc(sizeof(int) * (size_t)len);
      if (!entry) return 0;
      for (i = 0; i < len; ++i) entry[i] = buffer[i];
    }

    int **new_list = (int**)realloc(*out_list, sizeof(int*) * (size_t)(*out_count + 1));
    if (!new_list) {
      free(entry);
      return 0;
    }

    *out_list = new_list;
    (*out_list)[*out_count] = entry;
    (*out_count)++;
    return 1;
  }

  int a;
  for (a = start_axis; a < 3; ++a) {
    buffer[depth] = a;
    if (!gen_axis_multisets_rec(len, depth + 1, a, buffer, out_list, out_count))
      return 0;
  }

  return 1;
}

static int gen_axis_multisets(int len, int ***out_list, int *out_count)
{
  *out_list = NULL;
  *out_count = 0;

  if (len < 0) return 0;

  if (len == 0) {
    int **list = (int**)malloc(sizeof(int*));
    if (!list) return 0;
    list[0] = NULL;
    *out_list = list;
    *out_count = 1;
    return 1;
  }

  int *buffer = (int*)malloc(sizeof(int) * (size_t)len);
  if (!buffer) return 0;

  int ok = gen_axis_multisets_rec(len, 0, 0, buffer, out_list, out_count);
  free(buffer);

  if (!ok) {
    free_list_2d(*out_list, *out_count);
    *out_list = NULL;
    *out_count = 0;
  }

  return ok;
}


/* ---------- Core: build irreducible list for given (p,m) in Table order ---------- */
/*
 * Build irreducible tensor components for given order p and m,
 * in the exact nested ordering described at the top.
 *
 * Returns 1 on success, 0 on failure.
 */

int init_irreducible_components_pm(int p, int m, int ***out_list, int *out_count)
{
  *out_list = NULL;
  *out_count = 0;

  if (p < 0) return 0;
  if (m < 0 || m > p) return 0;

  const int n_ns = m;
  const int n_st = p - m;

  /* 1) Generate non-star and star multisets in canonical (Table) order */
  int **ns_list = NULL, **st_list = NULL;
  int ns_count = 0, st_count = 0;

  if (!gen_axis_multisets(n_ns, &ns_list, &ns_count)) return 0;
  if (!gen_axis_multisets(n_st, &st_list, &st_count)) {
    free_list_2d(ns_list, ns_count);
    return 0;
  }

  /* 2) Nested loops: non-star first, then star (Table ordering) */
  long long total = (long long)ns_count * (long long)st_count;

  if (total <= 0) {
    free_list_2d(ns_list, ns_count);
    free_list_2d(st_list, st_count);
    return 0;
  }

  int **list = (int**)malloc(sizeof(int*) * (size_t)total);

  if (!list) {
    free_list_2d(ns_list, ns_count);
    free_list_2d(st_list, st_count);
    return 0;
  }

  int i,j;
  long long k = 0;
  for (i = 0; i < ns_count; ++i) {
    for (j = 0; j < st_count; ++j) {
      int *entry = (int*)malloc(sizeof(int) * (size_t)p);
      if (!entry) {
        long long t;
        /* Fail hard to keep "order = Table" guaranteed (no holes) */
        for (t = 0; t < k; ++t) free(list[t]);
        free(list);
        free_list_2d(ns_list, ns_count);
        free_list_2d(st_list, st_count);
        return 0;
      }

      int t; 
      /* First m entries: non-star axes (0..2) */
      for (t = 0; t < n_ns; ++t) entry[t] = (n_ns > 0) ? ns_list[i][t] : 0;

      /* Last p-m entries: star axes, encoded as axis+3 (3..5) */
      for (t = 0; t < n_st; ++t) entry[n_ns + t] = ((n_st > 0) ? st_list[j][t] : 0) + 3;

      list[k++] = entry;
    }
  }

  *out_list = list;
  *out_count = (int)k;

  free_list_2d(ns_list, ns_count);
  free_list_2d(st_list, st_count);
  return 1;
}


/* ---------- Optional: map an arbitrary idx[] to the canonical irreducible index ---------- */
/*
 * Map idx[0..p-1] (each in 0..5) to the canonical irreducible entry index
 * under the (p,m) class.
 *
 * Canonicalization:
 *   - split into non-star (<3) and star (>=3)
 *   - counts must be exactly (m, p-m)
 *   - sort within each group
 *   - rebuild canonical [non-star..., star...]
 *   - linear search in ind_list (which is in Table order)
 *
 * Returns:
 *   index (0..ind_count-1) on success, -1 on invalid/not found.
 */

int map_to_irreducible_pm(int p, int m, const int *idx,
                          int **ind_list, int ind_count)
{
  const int n_ns = m;
  const int n_st = p - m;

  int *ns = (int*)malloc(sizeof(int) * (size_t)((n_ns > 0) ? n_ns : 1));
  int *st = (int*)malloc(sizeof(int) * (size_t)((n_st > 0) ? n_st : 1));

  if (!ns || !st) { free(ns); free(st); return -1; }

  int ns_k = 0, st_k = 0;

  int i;
  for (i = 0; i < p; ++i) {
    int v = idx[i];
    if (v < 0 || v > 5) { free(ns); free(st); return -1; }

    if (v >= 3) {
      if (st_k >= n_st) { free(ns); free(st); return -1; }
      st[st_k++] = v - 3; /* store 0..2 */
    } else {
      if (ns_k >= n_ns) { free(ns); free(st); return -1; }
      ns[ns_k++] = v;
    }
  }

  if (ns_k != n_ns || st_k != n_st) { free(ns); free(st); return -1; }

  qsort(ns, (size_t)n_ns, sizeof(int), cmp_int);
  qsort(st, (size_t)n_st, sizeof(int), cmp_int);

  int *canon = (int*)malloc(sizeof(int) * (size_t)p);
  if (!canon) { free(ns); free(st); return -1; }

  for (i = 0; i < n_ns; ++i) canon[i] = ns[i];
  for (i = 0; i < n_st; ++i) canon[n_ns + i] = st[i] + 3;

  int result = -1;
  for (i = 0; i < ind_count; ++i) {
    if (equal_indices(canon, ind_list[i], p)) { result = i; break; }
  }

  free(ns);
  free(st);
  free(canon);
  return result;
}




/* ---------- Example main() ---------- */

int test(void)
{
  /* Example: DC p=2, m=1 -> exactly: xx*, xy*, xz*, yx*, yy*, yz*, zx*, zy*, zz* */
  {
    int p = 2, m = 1;
    int **list = NULL;
    int count = 0;

    if (!init_irreducible_components_pm(p, m, &list, &count)) {
      fprintf(stderr, "Failed to build irreducible list for p=%d,m=%d\n", p, m);
      return 1;
    }

    printf("=== irreducible list (Table order) for p=%d, m=%d ===\n", p, m);
    printf("count = %d\n", count);

    int i;
    for (i = 0; i < count; ++i) {
      printf("%2d: ", i);
      print_entry_raw(list[i], p);
      printf("\n");
    }

    /* Optional: mapping test */

    {

      int idx1[] = {0, 4}; /* x y* */
      int k1 = map_to_irreducible_pm(p, m, idx1, list, count);
      printf("\nmap idx {x, y*} -> %d\n", k1);

      int idx2[] = {4, 0}; /* y* x : same monomial as x y* */
      int k2 = map_to_irreducible_pm(p, m, idx2, list, count);
      printf("map idx {y*, x} -> %d\n", k2);

      int idx3[] = {1, 3}; /* y x* */
      int k3 = map_to_irreducible_pm(p, m, idx3, list, count);
      printf("map idx {y, x*} -> %d\n", k3);
    }

    free_list_2d(list, count);
  }

  /* Another example: p=3, m=2 (E E E*) -> 18 components in Table order */
  {
    int p = 3, m = 2;
    int **list = NULL;
    int count = 0;

    if (!init_irreducible_components_pm(p, m, &list, &count)) {
      fprintf(stderr, "Failed to build irreducible list for p=%d,m=%d\n", p, m);
      return 1;
    }

    printf("\n=== irreducible list (Table order) for p=%d, m=%d ===\n", p, m);
    printf("count = %d\n", count);

    int show = (count < 24) ? count : 24;
    int i;
    for (i = 0; i < show; ++i) {
      printf("%2d: ", i);
      print_entry_raw(list[i], p);
      printf("\n");
    }

    if (count > show) printf("... (%d more)\n", count - show);
    free_list_2d(list, count);
  }

  return 0;
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



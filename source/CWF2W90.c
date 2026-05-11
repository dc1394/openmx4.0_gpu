/**********************************************************************
  CWF2W90.c:

   CWF2W90.c is a subroutine to generate files which interface
   with Wannier90, including mmn, amn, eig, and win files.

  Log of CWF2W90.c:

   01/Feb./2026  Released by T.Ozaki and B.B. Prasad

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#include "openmx_common.h"
#include "lapack_prototypes.h"
#include "mpi.h"



#define debug6  0           /* for wannier center */
#define smallvalue   1.0e-5 /*smallvalue close to zero*/


/* begin: For bvectors */
static void Set_bvector( int knum_i, int knum_j, int knum_k, int MAXSHELL );

static int Cal_Weight_of_Shell(double **klatt, int *M_s, int **bvector, int *num_shell, 
		   	       double *wb, int *Reject_Shell, int *searched_shell);

static void Shell_Structure(double **klatt, int *M_s, int **bvector, 
                            int *shell_num, int MAXSHELL);

static void Ascend_Ordering(double *xyz_value, int *ordering, int tot_atom);

static void Wigner_Seitz_Vectors(double metric[3][3], double rtv[4][4], 
       int knum_i, int knum_j, int knum_k, int *r_num, int **rvect, int *ndegen);
/* end: For bvectors */



static int mirror_index(int a, int N);

static void representative_ijk(int i0,int j0,int k0,
			       int Ni,int Nj,int Nk,
			       int ***k_op,
			       int *i,int *j,int *k);

static void Calc_OLPexp( dcomplex *****OLPexp, dcomplex **OLPexp_1D, 
			 int *MP, int *order_GA, 
			 int *My_NZeros, int *SP_NZeros, int *SP_Atoms);


static void Generate_win(int ***k_op, int ***k2k,
			 double *T_KGrids1, 
			 double *T_KGrids2, 
			 double *T_KGrids3);

static void Generate_eig_Col(int ***k_op, int ***k2k);
static void Generate_eig_NC(int ***k_op, int ***k2k);

static void Generate_amn_Col(int ***k_op, int ***k2k);
static void Generate_amn_NC(int ***k_op, int ***k2k);

static void Generate_mmn_Col(dcomplex **OLPexp_1D, int ***k_op, int ***k2k,
			     double *T_KGrids1, double *T_KGrids2, double *T_KGrids3, 
			     int *order_GA, int *MP);

static void Generate_mmn_NC(dcomplex **OLPexp_1D, int ***k_op, int ***k2k, 
			    double *T_KGrids1, double *T_KGrids2, double *T_KGrids3, 
			    int *order_GA, int *MP);

static int Get_OneD_OLPexp_Col(int set_flag, dcomplex ****RH, dcomplex *H1, int *MP, 
                               int *order_GA, int *My_NZeros, int *is1, int *is2);

int error_counter=0;
int tot_bvector;
double **bvector;     /* Cartesian a.u. */
double **frac_bv;     /* fractional coordinate */
int **frac_bv_int;    /* fractional coordinate normalized by the mesh vector */
double r_latt[3][3]; 




		    

void CWF2W90(int ***k_op, int *T_k_op, int **T_k_ID, 
             double *T_KGrids1, double *T_KGrids2, 
             double *T_KGrids3, int *MP, int *order_GA)
{
  int i,j,k,b,tno0,tno1,Cwan,Hwan,Gc_AN,h_AN,Gh_AN,Mc_AN;
  int *My_NZeros,*SP_NZeros,*SP_Atoms,size_OLPexp,***k2k,T_knum;
  dcomplex *****OLPexp,**OLPexp_1D,ctmp;
  int numprocs,myid;

  /* MPI */ 

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  if (sizeof(dcomplex)!=(2*sizeof(double))){
    printf("Error 1: 'CWF2W90' does not work properly in your computational environment.\n");fflush(stdout);
    MPI_Finalize();
    exit(0);
  }

  /****************************************************
                      show message 
  ****************************************************/

  if (myid==Host_ID){
    printf("\n");fflush(stdout);
    printf("CWF2W90 starts.\n"); fflush(stdout);
  }

  /****************************************************
                      Set bvectors
  ****************************************************/
  
  Set_bvector(Kspace_grid1, Kspace_grid2, Kspace_grid3, 30);

  /****************************************************
                allocation of arrays
  ****************************************************/

  OLPexp=(dcomplex*****)malloc(sizeof(dcomplex****)*tot_bvector);
  for (b=0; b<tot_bvector; b++){

    OLPexp[b]=(dcomplex****)malloc(sizeof(dcomplex***)*(Matomnum+1));
    FNAN[0] = 0;
    for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

      if (Mc_AN==0){
	Gc_AN = 0;
	tno0 = 1;
      }
      else{
	Gc_AN = M2G[Mc_AN];
	Cwan = WhatSpecies[Gc_AN];
	tno0 = Spe_Total_CNO[Cwan];
      }

      OLPexp[b][Mc_AN] = (dcomplex***)malloc(sizeof(dcomplex**)*(FNAN[Gc_AN]+1));
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	if (Mc_AN==0){
	  tno1 = 1;
	}
	else{
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  tno1 = Spe_Total_CNO[Hwan];
	}

	OLPexp[b][Mc_AN][h_AN] = (dcomplex**)malloc(sizeof(dcomplex*)*tno0);
	for (i=0; i<tno0; i++){
	  OLPexp[b][Mc_AN][h_AN][i] = (dcomplex*)malloc(sizeof(dcomplex)*tno1);
	}
      }
    }
  }

  My_NZeros = (int*)malloc(sizeof(int)*numprocs);
  SP_NZeros = (int*)malloc(sizeof(int)*numprocs);
  SP_Atoms = (int*)malloc(sizeof(int)*numprocs);
  
  size_OLPexp = Get_OneD_OLPexp_Col(0, OLPexp[0], &ctmp, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);

  OLPexp_1D = (dcomplex**)malloc(sizeof(dcomplex*)*tot_bvector);
  for (b=0; b<tot_bvector; b++){
    OLPexp_1D[b] = (dcomplex*)malloc(sizeof(dcomplex)*size_OLPexp);
  }

  k2k = (int***)malloc(sizeof(int**)*Kspace_grid1);
  for (i=0; i<Kspace_grid1; i++){
    k2k[i] = (int**)malloc(sizeof(int*)*Kspace_grid2);
    for (j=0; j<Kspace_grid2; j++){
      k2k[i][j] = (int*)malloc(sizeof(int)*Kspace_grid3);
    }
  }

   if (Solver==2){  // Cluster
     k_op = (int***)malloc(sizeof(int**)*Kspace_grid1);
     for (i=0;i<Kspace_grid1; i++) {
       k_op[i] = (int**)malloc(sizeof(int*)*Kspace_grid2);
       for (j=0;j<Kspace_grid2; j++) {
	 k_op[i][j] = (int*)malloc(sizeof(int)*Kspace_grid3);
       }
     }
     k_op[0][0][0] = 1;

     T_KGrids1 = (double*)malloc(sizeof(double)*1);
     T_KGrids2 = (double*)malloc(sizeof(double)*1);
     T_KGrids3 = (double*)malloc(sizeof(double)*1);

     T_KGrids1[0] = 0.0;
     T_KGrids2[0] = 0.0;
     T_KGrids3[0] = 0.0;
   }

  /****************************************************
                     construct k2k
  ****************************************************/

   T_knum = 0;
   for (i=0; i<Kspace_grid1; i++){
     for (j=0; j<Kspace_grid2; j++){
       for (k=0; k<Kspace_grid3; k++){
	 if (0<k_op[i][j][k]){
	   k2k[i][j][k] = T_knum;
	   T_knum++;
	 }
       }
     }
   }

  /****************************************************
                   generate .amn file
  ****************************************************/

  if (myid==Host_ID){
    if (SpinP_switch<=1){
      Generate_amn_Col(k_op,k2k);
    }
    else if (SpinP_switch==3){
      Generate_amn_NC(k_op,k2k);
    }  
  }

  /****************************************************
                   generate .mmn file
  ****************************************************/

  /* calculation of OLPexp */

  Calc_OLPexp(OLPexp, OLPexp_1D, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);  

  if (SpinP_switch<=1){
    Generate_mmn_Col(OLPexp_1D,k_op,k2k,T_KGrids1,T_KGrids2,T_KGrids3,order_GA,MP);
  }  
  else if (SpinP_switch==3){
    Generate_mmn_NC(OLPexp_1D,k_op,k2k,T_KGrids1,T_KGrids2,T_KGrids3,order_GA,MP);
  }  

  /****************************************************
                   generate .eig file
  ****************************************************/

  if (myid==Host_ID){
    if (SpinP_switch<=1){
      Generate_eig_Col(k_op,k2k);
    }
    else if (SpinP_switch==3){
      Generate_eig_NC(k_op,k2k);
    }  
  }  

  /****************************************************
                   generate .win file
  ****************************************************/

  if (myid==Host_ID) Generate_win(k_op,k2k,T_KGrids1,T_KGrids2,T_KGrids3);

  /****************************************************
                      show message 
  ****************************************************/

  MPI_Allreduce( MPI_IN_PLACE, &error_counter, 1, MPI_INT, MPI_SUM, mpi_comm_level1);

  if (0<error_counter && myid==Host_ID){
    printf("\nCWF2W90 encountered error. Please check error messages.\n\n"); fflush(stdout);
  }

  else if (myid==Host_ID){

    printf("\nCWF2W90 was successfully performed.\n\n"); fflush(stdout);

    if (SpinP_switch==0){
      printf(" %s_0.amn\n",filename); fflush(stdout);
      printf(" %s_0.mmn\n",filename); fflush(stdout);
      printf(" %s_0.eig\n",filename); fflush(stdout);
      printf(" %s_0.win\n",filename); fflush(stdout);
    }
    else if (SpinP_switch==1){
      printf(" %s_0.amn\n",filename); fflush(stdout);
      printf(" %s_1.amn\n",filename); fflush(stdout);
      printf(" %s_0.mmn\n",filename); fflush(stdout);
      printf(" %s_1.mmn\n",filename); fflush(stdout);
      printf(" %s_0.eig\n",filename); fflush(stdout);
      printf(" %s_1.eig\n",filename); fflush(stdout);
      printf(" %s_0.win\n",filename); fflush(stdout);
      printf(" %s_1.win\n",filename); fflush(stdout);
    }
    else if (SpinP_switch==3){
      printf(" %s.amn\n",filename); fflush(stdout);
      printf(" %s.mmn\n",filename); fflush(stdout);
      printf(" %s.eig\n",filename); fflush(stdout);
      printf(" %s.win\n",filename); fflush(stdout);
    }

    printf("\nare generated.\n\n"); fflush(stdout);
  }

  /****************************************************
               delete the _evec directory
  ****************************************************/

  MPI_Barrier(mpi_comm_level1);

  if (myid==Host_ID){

    char operate[1000];
    sprintf(operate,"%s%s_evec",filepath,filename);
    DIR *dp = opendir(operate);

    if (dp) {

      struct dirent *de;

      while ((de = readdir(dp)) != NULL) {

	if (strcmp(de->d_name, ".")  == 0) continue;
	if (strcmp(de->d_name, "..") == 0) continue;

	char path[8192];
	sprintf(path, "%s/%s", operate, de->d_name);
	unlink(path);  
      }
      closedir(dp);
    }

    rmdir(operate);  
  }

  /****************************************************
                  freeing of arrays
  ****************************************************/

  for(i=0;i<tot_bvector;i++){
    free(frac_bv[i]);
  }
  free(frac_bv);

  for(i=0; i<tot_bvector; i++){
    free(bvector[i]);
  }
  free(bvector);

  for(i=0; i<tot_bvector; i++){
    free(frac_bv_int[i]);
  }
  free(frac_bv_int);

  free(My_NZeros);
  free(SP_NZeros);
  free(SP_Atoms);

  for (b=0; b<tot_bvector; b++){
    free(OLPexp_1D[b]);
  }
  free(OLPexp_1D);

  for (i=0; i<Kspace_grid1; i++){
    for (j=0; j<Kspace_grid2; j++){
      free(k2k[i][j]);
    }
    free(k2k[i]);
  }
  free(k2k);

  for (b=0; b<tot_bvector; b++){

    FNAN[0] = 0;
    for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

      if (Mc_AN==0){
	Gc_AN = 0;
	tno0 = 1;
      }
      else{
	Gc_AN = M2G[Mc_AN];
	Cwan = WhatSpecies[Gc_AN];
	tno0 = Spe_Total_CNO[Cwan];
      }

      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	if (Mc_AN==0){
	  tno1 = 1;
	}
	else{
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  tno1 = Spe_Total_CNO[Hwan];
	}

	for (i=0; i<tno0; i++){
	  free(OLPexp[b][Mc_AN][h_AN][i]);
	}
	free(OLPexp[b][Mc_AN][h_AN]);
      }
      free(OLPexp[b][Mc_AN]);
    }
    free(OLPexp[b]);
  }
  free(OLPexp);

   if (Solver==2){  // Cluster
     for (i=0;i<Kspace_grid1; i++) {
       for (j=0;j<Kspace_grid2; j++) {
	 free(k_op[i][j]);
       }
       free(k_op[i]);
     }
     free(k_op);

     free(T_KGrids1);
     free(T_KGrids2);
     free(T_KGrids3);
   }
}


void Generate_win(int ***k_op, int ***k2k,
                  double *T_KGrids1, 
                  double *T_KGrids2, 
                  double *T_KGrids3)
{
  int spin,spinsize;
  int i,j,k,ia,ja,ka,ik,ika,itmp,Gc_AN;
  double Cxyz[4],sign,k1,k2,k3;
  FILE *fp;
  char buf[fp_bsize];
  char fname[500];

  if      (SpinP_switch==0) spinsize = 1;
  else if (SpinP_switch==1) spinsize = 2;
  else if (SpinP_switch==3) spinsize = 1;

  for (spin=0; spin<spinsize; spin++){

    if (SpinP_switch<=1){
      sprintf(fname,"%s%s_%d.win",filepath,filename,spin);
    }
    else if (SpinP_switch==3){
      sprintf(fname,"%s%s.win",filepath,filename);
    }

    if ((fp = fopen(fname,"w")) != NULL){

      setvbuf(fp,buf,_IOFBF,fp_bsize);  /* setvbuf */

      fprintf(fp, "num_bands %d\n", Num_KS_states_CWF);
      fprintf(fp, "num_wann %d\n",  TNum_CWFs);
      fprintf(fp, "\n");  // blank line                                                       
      fprintf(fp, "!spinors T\n");
      fprintf(fp, "\n");  // blank line                                                       
      fprintf(fp, "dis_num_iter 0\n");
      fprintf(fp, "num_iter 0\n");
      fprintf(fp, "\n");  // blank line
      fprintf(fp, "\n");  // blank line
      fprintf(fp, "bands_plot T\n");
      fprintf(fp, "bands_num_points 200\n");
      fprintf(fp, "bands_plot_format gnuplot\n");
      fprintf(fp, "\n");  // blank line
      fprintf(fp, "!restart = plot\n");  
      fprintf(fp, "\n");  // blank line
      fprintf(fp, "!kpath = .true.\n");  
      fprintf(fp, "!kpath_task = bands\n");
      fprintf(fp, "!kpath_num_points = 200\n");
      fprintf(fp, "!kpath_bands_colour = none\n");
      fprintf(fp, "\n");  // blank line
      fprintf(fp, "begin kpoint_path\n");
      for (i=1;i<=Band_Nkpath;i++) {
	fprintf(fp,"  %s %lf %lf %lf  %s %lf %lf %lf\n",
		Band_kname[i][1],
		Band_kpath[i][1][1], Band_kpath[i][1][2], Band_kpath[i][1][3],
		Band_kname[i][2],
		Band_kpath[i][2][1], Band_kpath[i][2][2], Band_kpath[i][2][3]);
      }
      fprintf(fp, "end kpoint_path\n");
      fprintf(fp, "\n");  // blank line
      fprintf(fp, "!For POSTW90 (Berry Module)\n");
      fprintf(fp, "berry = T                  # default = F\n");  
      fprintf(fp, "berry_task = sc            # kubo|ahc|morb|shc|sc\n");
      fprintf(fp, "berry_kmesh = 100 100 100\n");   
      fprintf(fp, "!berry_curv_adpt_kmesh = 3          # when berry_task = ahc|shc (default = 1)\n");    
      fprintf(fp, "!berry_curv_adpt_kmesh_thresh =     # when berry_task = ahc|shc (default = 100)\n");
      fprintf(fp, "\n");  // blank line
      fprintf(fp, "fermi_energy = 0.0         # default = 0.0 (or, top of VB energy in the case of a semiconductor)\n");
      fprintf(fp, "!fermi_energy_min  = -3.0\n");
      fprintf(fp, "!fermi_energy_max  =  3.0\n");
      fprintf(fp, "!fermi_energy_step =  0.1  # default = 0.01\n");
      fprintf(fp, "\n");  // blank line
      fprintf(fp, "kubo_freq_min = 0.0      # for computing optical conductivity, JDOS, ac SHC, shift current conductivity\n");
      fprintf(fp, "kubo_freq_max = 10.0\n");
      fprintf(fp, "kubo_freq_step = 0.01    # for getting smooth spectrum (default = 0.01)\n");
      fprintf(fp, "kubo_adpt_smr = T\n");
      fprintf(fp, "!kubo_adpt_smr = F        # default = T\n");
      fprintf(fp, "!kubo_smr_fixed_en_width = 0.04  # default = 0\n");
      fprintf(fp, "!scissors_shift =        # default = 0 i.e., no scissors shift applied.\n");
      fprintf(fp, "!num_valence_bands =     # no default value\n");
      fprintf(fp, "\n");  // blank line
      fprintf(fp, "sc_eta = 0.040           # default = 0.04\n");
      fprintf(fp, "sc_phase_conv = 2\n"); 
      fprintf(fp, "\n");  // blank line
      fprintf(fp, "begin unit_cell_cart\n");
      fprintf(fp, "Ang\n");
      for (i=1; i<=3; i++){
	fprintf(fp,"  %18.12f %18.12f %18.12f\n",
		tv[i][1]*BohrR,tv[i][2]*BohrR,tv[i][3]*BohrR);
      }
      fprintf(fp, "end unit_cell_cart\n");
      fprintf(fp, "\n");  // blank line
      fprintf(fp, "begin atoms_frac\n");
      
      for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){

	/* The zero is taken as the origin of the unit cell. */

	Cxyz[1] = Gxyz[Gc_AN][1];
	Cxyz[2] = Gxyz[Gc_AN][2];
	Cxyz[3] = Gxyz[Gc_AN][3];

	Cell_Gxyz[Gc_AN][1] = Dot_Product(Cxyz,rtv[1])*0.5/PI;
	Cell_Gxyz[Gc_AN][2] = Dot_Product(Cxyz,rtv[2])*0.5/PI;
	Cell_Gxyz[Gc_AN][3] = Dot_Product(Cxyz,rtv[3])*0.5/PI;

	/* The fractional coordinates are kept within 0 to 1. */

	for (i=1; i<=3; i++){

	  itmp = (int)Cell_Gxyz[Gc_AN][i]; 

	  if (1.0<Cell_Gxyz[Gc_AN][i]){
	    Cell_Gxyz[Gc_AN][i] = fabs(Cell_Gxyz[Gc_AN][i] - (double)itmp);
	  }
	  else if (Cell_Gxyz[Gc_AN][i]<-1.0e-13){
	    Cell_Gxyz[Gc_AN][i] = fabs(Cell_Gxyz[Gc_AN][i] + (double)(abs(itmp)+1));
	  }
	}

	k = WhatSpecies[Gc_AN];

	fprintf(fp," %4s %18.14f %18.14f %18.14f\n",
		SpeName[k],
		Cell_Gxyz[Gc_AN][1],
		Cell_Gxyz[Gc_AN][2],
		Cell_Gxyz[Gc_AN][3]);
      }

      fprintf(fp, "end atoms_frac\n");
      fprintf(fp, "\n");  // blank line
      fprintf(fp, "mp_grid %d %d %d\n",Kspace_grid1, Kspace_grid2, Kspace_grid3);
      fprintf(fp, "\n");  // blank line
      fprintf(fp, "begin_kpoints\n");

      if (SpinP_switch<=1){

	ik = 0;
	for (i=0; i<Kspace_grid1; i++){
	  for (j=0; j<Kspace_grid2; j++){
	    for (k=0; k<Kspace_grid3; k++){

	      representative_ijk(i,j,k,Kspace_grid1,Kspace_grid2,Kspace_grid3,k_op,&ia,&ja,&ka);
	      ika = k2k[ia][ja][ka];
       
	      if (k_op[i][j][k]==0){ sign =-1.0; }
	      else {                 sign = 1.0; }

	      k1 = sign*T_KGrids1[ika];
	      k2 = sign*T_KGrids2[ika];
	      k3 = sign*T_KGrids3[ika];

	      if (k1<0.0) k1 += 1.0;
	      if (k2<0.0) k2 += 1.0;
	      if (k3<0.0) k3 += 1.0;

	      fprintf(fp, "%16.12f %16.12f %16.12f\n",k1,k2,k3 );

	      ik++;
	    }
	  }
	}
      }

      else if (SpinP_switch==3){

	ik = 0;
	for (i=0; i<Kspace_grid1; i++){
	  for (j=0; j<Kspace_grid2; j++){
	    for (k=0; k<Kspace_grid3; k++){

	      ika = k2k[i][j][k];
       
	      fprintf(fp, "%18.14f %18.14f %18.14f\n",
		      T_KGrids1[ika],
		      T_KGrids2[ika],
		      T_KGrids3[ika] );

	      ik++;
	    }
	  }
	}
      }

      fprintf(fp, "end_kpoints\n");
   
      /* fclose(fp) */
      fclose(fp);
    }
    else {
      printf("Failure of saving %s\n",fname);fflush(stdout);
      error_counter++;
    }

  } // spin
}


void Generate_eig_Col(int ***k_op, int ***k2k)
{
  int spin,m,ik,i,j,k,ii,jj,kk,ik_file;
  char fname[500],fn_eval[500];
  char buf[fp_bsize];
  FILE *fp,*fp_eval;
  double *eval;

  /* allocation of array */
  eval = (double*)malloc(sizeof(double)*Num_KS_states_CWF);

  /* generate .eig */
  
  for (spin=0; spin<=SpinP_switch; spin++){ 

    sprintf(fname,"%s%s_%d.eig",filepath,filename,spin);

    if ((fp = fopen(fname,"w")) != NULL){

      setvbuf(fp,buf,_IOFBF,fp_bsize);  /* setvbuf */

      ik = 0;
      for (i=0; i<Kspace_grid1; i++){
	for (j=0; j<Kspace_grid2; j++){
	  for (k=0; k<Kspace_grid3; k++){

            representative_ijk(i,j,k,Kspace_grid1,Kspace_grid2,Kspace_grid3,k_op,&ii,&jj,&kk);
            ik_file = k2k[ii][jj][kk];              

	    sprintf(fn_eval,"%s%s_evec/eval_%d_%d",filepath,filename,spin,ik_file);

	    if ((fp_eval = fopen(fn_eval,"rb")) != NULL){

	      /* read a pm file */
	      fread(eval,sizeof(double),Num_KS_states_CWF,fp_eval);

	      /* save the .eig file */

	      for (m=0; m<Num_KS_states_CWF; m++){
		fprintf(fp,"%5d %5d %22.14f\n",m+1,ik+1,(eval[m]-ChemP)*eV2Hartree);
	      } // m

	      /* fclose(fp_eval) */
	      fclose(fp_eval);
	    } // end of if 
	    else{
	      printf("Failure of reading %s\n",fn_eval);fflush(stdout);
              error_counter++;
	    }

            /* increment ik */

            ik++;
 
	  } // k 
	} // j
      } // i

      /* fclose(fp) */
      fclose(fp);
    }
    else {
      printf("Failure of saving %s\n",fname);fflush(stdout);
      error_counter++;
    }
  } // spin

  /* freeing of array */
  free(eval);
}



void Generate_eig_NC(int ***k_op, int ***k2k)
{
  int m,ik,i,j,k,ii,jj,kk,ik_file;
  char fname[500],fn_eval[500];
  char buf[fp_bsize];
  FILE *fp,*fp_eval;
  double *eval;

  /* allocation of array */
  eval = (double*)malloc(sizeof(double)*Num_KS_states_CWF);

  /* generate .eig */
  
  sprintf(fname,"%s%s.eig",filepath,filename);

  if ((fp = fopen(fname,"w")) != NULL){

    setvbuf(fp,buf,_IOFBF,fp_bsize);  /* setvbuf */

    ik = 0;
    for (i=0; i<Kspace_grid1; i++){
      for (j=0; j<Kspace_grid2; j++){
	for (k=0; k<Kspace_grid3; k++){

	  ik_file = k2k[i][j][k];

	  sprintf(fn_eval,"%s%s_evec/eval_%d",filepath,filename,ik_file);

	  if ((fp_eval = fopen(fn_eval,"rb")) != NULL){

	    /* read a pm file */
	    fread(eval,sizeof(double),Num_KS_states_CWF,fp_eval);

	    /* save the .eig file */
	    for (m=0; m<Num_KS_states_CWF; m++){
	      fprintf(fp,"%5d %5d %22.14f\n",m+1,ik+1,(eval[m]-ChemP)*eV2Hartree);
	    } // m

	      /* fclose(fp_eval) */
	    fclose(fp_eval);
	  } // end of if 
	  else{
	    printf("Failure of reading %s\n",fn_eval);fflush(stdout);
            error_counter++;
	  }

	  /* increment ik */

	  ik++;
 
	} // k 
      } // j
    } // i

      /* fclose(fp) */
    fclose(fp);
  }
  else {
    printf("Failure of saving %s\n",fname);fflush(stdout);
    error_counter++;
  }

  /* freeing of array */
  free(eval);
}



void Generate_mmn_Col(dcomplex **OLPexp_1D, int ***k_op, int ***k2k, 
                      double *T_KGrids1, double *T_KGrids2, double *T_KGrids3, 
                      int *order_GA, int *MP)
{
  int spin,m,n,ik,ika,ikb,i,j,k,ii,jj,kk,b,Tknum,p,M,N,K;
  int sign1,sign2,ia,ja,ka,ib,jb,kb;
  int iorb,jorb,idx,ig,jg,wanA,wanB,tnoA,tnoB,Rn,l1,l2,l3;
  int AN,GA_AN,Anum,Bnum,LB_AN,GB_AN,ik_s,ik_e,g1,g2,g3;
  double k1,k2,k3,b1,b2,b3,co,si,kRn;
  dcomplex *Ca,*Cb,*S,*WORK,*Mbin;
  dcomplex alpha = {1.0,0.0}; 
  dcomplex beta = {0.0,0.0};
  char fname_bin[500],fna[500],fnb[500];
  char fname[500];
  char buf[fp_bsize];
  FILE *fp,*fp_bin,*fpa,*fpb;
  int numprocs,myid;

  /* MPI */ 

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /*****************************************************
                  allocation of arrays
  *****************************************************/

  Ca = (dcomplex*)malloc(sizeof(dcomplex)*TNum_PAOs*TNum_PAOs);
  Cb = (dcomplex*)malloc(sizeof(dcomplex)*TNum_PAOs*TNum_PAOs);
  S = (dcomplex*)malloc(sizeof(dcomplex)*TNum_PAOs*TNum_PAOs);
  WORK = (dcomplex*)malloc(sizeof(dcomplex)*TNum_PAOs*TNum_PAOs);

  /* set Tknum */

  Tknum = Kspace_grid1*Kspace_grid2*Kspace_grid3;

  /*****************************************************
   division of Tknum by numprocs for MPI parallelization
  *****************************************************/
 
  ik_s = (int)(((long long)myid*(long long)Tknum)/(long long)numprocs);
  ik_e = (int)(((long long)(myid+1)*(long long)Tknum)/(long long)numprocs);

  /*****************************************************
                    calculation of mmn
  *****************************************************/

  for (spin=0; spin<=SpinP_switch; spin++){ 

    if (myid==Host_ID){ 
      printf("\n"); fflush(stdout);
    }

    for (ik=ik_s; ik<ik_e; ik++){

      if (myid==Host_ID){ 
        printf("  Calculating Mmnk spin=%2d %2d/%d\n",spin,ik,ik_e-ik_s-1);fflush(stdout);
      }

      /* open fp_bin */ 

      sprintf(fname_bin,"%s%s_evec/mmnbin_%d_%d",filepath,filename,spin,ik);

      if ((fp_bin = fopen(fname_bin,"wb")) != NULL){

	/* get i, j, k */ 

	i = ik/(Kspace_grid2*Kspace_grid3);
	j = (ik-i*Kspace_grid2*Kspace_grid3)/Kspace_grid3;
	k = ik-i*Kspace_grid2*Kspace_grid3-j*Kspace_grid3;

	representative_ijk(i,j,k,Kspace_grid1,Kspace_grid2,Kspace_grid3,k_op,&ia,&ja,&ka);

	ika = k2k[ia][ja][ka];

	if (k_op[i][j][k]==0){ sign1 =-1; }
	else {                 sign1 = 1; }

	k1 = (double)sign1*T_KGrids1[ika];
	k2 = (double)sign1*T_KGrids2[ika];
	k3 = (double)sign1*T_KGrids3[ika];

	sprintf(fna,"%s%s_evec/evec_%d_%d",filepath,filename,spin,ika);

	if ((fpa = fopen(fna,"rb")) != NULL){

	  /* read the evec file at k */
	  fread(Ca,sizeof(dcomplex),Num_KS_states_CWF*TNum_PAOs,fpa);

	  if (sign1==1){ // Ca is in size of TNum_PAOs x Num_KS_states_CWF
	    for (p=0; p<Num_KS_states_CWF*TNum_PAOs; p++) Ca[p].i = -Ca[p].i;
	  }

	  for (b=0; b<tot_bvector; b++){

	    b1 = frac_bv[b][0];
	    b2 = frac_bv[b][1];
	    b3 = frac_bv[b][2];

	    ii = ((i+frac_bv_int[b][0])%Kspace_grid1 + Kspace_grid1)%Kspace_grid1;
	    jj = ((j+frac_bv_int[b][1])%Kspace_grid2 + Kspace_grid2)%Kspace_grid2;
	    kk = ((k+frac_bv_int[b][2])%Kspace_grid3 + Kspace_grid3)%Kspace_grid3;

	    representative_ijk(ii,jj,kk,Kspace_grid1,Kspace_grid2,Kspace_grid3,k_op,&ib,&jb,&kb);
	    ikb = k2k[ib][jb][kb];

	    if (k_op[ii][jj][kk]==0){ sign2 =-1; }
	    else {                    sign2 = 1; }

	    sprintf(fnb,"%s%s_evec/evec_%d_%d",filepath,filename,spin,ikb);

	    if ((fpb = fopen(fnb,"rb")) != NULL){

	      /* read the evec file at k+b */
	      fread(Cb,sizeof(dcomplex),Num_KS_states_CWF*TNum_PAOs,fpb);

	      if (sign2==-1){
		for (p=0; p<Num_KS_states_CWF*TNum_PAOs; p++) Cb[p].i = -Cb[p].i;
	      }

	      /********************************************************
                                   calculation of S
	      ********************************************************/

	      for (p=0; p<TNum_PAOs*TNum_PAOs; p++) S[p] = Complex(0.0,0.0); 

	      idx = 0;
	      for (AN=1; AN<=atomnum; AN++){

		GA_AN = order_GA[AN];
		wanA = WhatSpecies[GA_AN];
		tnoA = Spe_Total_CNO[wanA];
		Anum = MP[GA_AN] - 1;

		for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){

		  GB_AN = natn[GA_AN][LB_AN];
		  Rn = ncn[GA_AN][LB_AN];
		  wanB = WhatSpecies[GB_AN];
		  tnoB = Spe_Total_CNO[wanB];
		  Bnum = MP[GB_AN] - 1;

		  l1 = atv_ijk[Rn][1];
		  l2 = atv_ijk[Rn][2];
		  l3 = atv_ijk[Rn][3];

		  kRn = (k1+b1)*(double)l1 + (k2+b2)*(double)l2 + (k3+b3)*(double)l3;
		  si = sin(2.0*PI*kRn);
		  co = cos(2.0*PI*kRn);

		  for (iorb=0; iorb<tnoA; iorb++){

		    ig = Anum + iorb;

		    for (jorb=0; jorb<tnoB; jorb++){

		      jg = Bnum + jorb;

		      S[jg*TNum_PAOs+ig].r += OLPexp_1D[b][idx].r*co - OLPexp_1D[b][idx].i*si;
		      S[jg*TNum_PAOs+ig].i += OLPexp_1D[b][idx].r*si + OLPexp_1D[b][idx].i*co;

		      idx++;

		    } // jorb 
		  } // iorb
		} // LB_AN
	      } // AN

	      /********************************************************
                  multiplication of matrices:  
                  Ca^{dag} S Cb

                  Ca (TNum_PAOs2 x Num_KS_states_CWF)
                  S  (TNum_PAOs2 x TNum_PAOs2)                   
                  Cb (TNum_PAOs2 x Num_KS_states_CWF)
	      ********************************************************/

	      M = Num_KS_states_CWF; N = TNum_PAOs; K = TNum_PAOs;
	      zgemm_( "T","N", &M, &N, &K, &alpha, Ca, &K, 
		      S, &K, &beta, WORK, &M );  // WORK (Num_KS_states_CWF x TNum_PAOs)

	      M = Num_KS_states_CWF; N = Num_KS_states_CWF; K = TNum_PAOs;
	      zgemm_( "N","N", &M, &N, &K, &alpha, WORK, &M, 
		      Cb, &K, &beta, S, &M ); // S (Num_KS_states_CWF x Num_KS_states_CWF)

	      /*
              if (ik==1 && b==1){
                for (p=0; p<Num_KS_states_CWF*Num_KS_states_CWF; p++){ 
                printf("ABC3 p=%2d %22.14f %22.14f\n",
                       p,S[p].r,S[p].i);
		}
	      } 
	      */

	      /********************************************************
                        save data temporary in binary mode
	      ********************************************************/

              fwrite(S,sizeof(dcomplex),(Num_KS_states_CWF*Num_KS_states_CWF),fp_bin);

	      /* fclose(fpb) */
	      fclose(fpb);

	    } // end of if
	    else{
	      printf("Failure of reading %s\n",fnb);fflush(stdout);
              error_counter++;
	    }

	  } // b

	  /* fclose(fpa) */
	  fclose(fpa);

	} // end of if
	else{
	  printf("Failure of reading %s\n",fna);fflush(stdout);
          error_counter++;
	}

        /* fclose(fp_bin) */
        fclose(fp_bin);

      } // end of if ((fp_bin =...
      else{
	printf("Failure of saving %s\n",fname_bin);fflush(stdout);
        error_counter++;
      }

    } // ik
  } // spin

  /*****************************************************
       freeing of arrays and allocating an array
  *****************************************************/

  free(Ca);
  free(Cb);
  free(S);
  free(WORK);

  Mbin = (dcomplex*)malloc(sizeof(dcomplex)*tot_bvector*Num_KS_states_CWF*Num_KS_states_CWF);

  MPI_Barrier(mpi_comm_level1);

  /*****************************************************
           merge all the mmnbin files to .mmn 
  *****************************************************/
 
  if (myid==Host_ID){

    printf("\n");
    printf("  Merging all files to a single .mmn file.\n");fflush(stdout);
    
    for (spin=0; spin<=SpinP_switch; spin++){ 

      sprintf(fname,"%s%s_%d.mmn",filepath,filename,spin);

      if ((fp = fopen(fname,"w")) != NULL){

	setvbuf(fp,buf,_IOFBF,fp_bsize);  /* setvbuf */

        fprintf(fp,"2nd line: BANDNUM, KPTNUM, NUMB, Nexts: KPTNUM x NUMBband elements block\n");
        fprintf(fp,"%d %d %d\n",Num_KS_states_CWF,Tknum,tot_bvector);

	for (ik=0; ik<Tknum; ik++){

	  i = ik/(Kspace_grid2*Kspace_grid3);
	  j = (ik-i*Kspace_grid2*Kspace_grid3)/Kspace_grid3;
	  k = ik-i*Kspace_grid2*Kspace_grid3-j*Kspace_grid3;

	  sprintf(fname_bin,"%s%s_evec/mmnbin_%d_%d",filepath,filename,spin,ik);

	  if ((fp_bin = fopen(fname_bin,"rb")) != NULL){

	    fread(Mbin,sizeof(dcomplex),(tot_bvector*Num_KS_states_CWF*Num_KS_states_CWF),fp_bin);

            p = 0;
            for (b=0; b<tot_bvector; b++){

	      ii = ((i+frac_bv_int[b][0])%Kspace_grid1 + Kspace_grid1)%Kspace_grid1;
	      jj = ((j+frac_bv_int[b][1])%Kspace_grid2 + Kspace_grid2)%Kspace_grid2;
	      kk = ((k+frac_bv_int[b][2])%Kspace_grid3 + Kspace_grid3)%Kspace_grid3;

              ikb = ii*Kspace_grid2*Kspace_grid3 + jj*Kspace_grid3 + kk;

	      g1 = ((i+frac_bv_int[b][0])-ii)/Kspace_grid1; 
	      g2 = ((j+frac_bv_int[b][1])-jj)/Kspace_grid2; 
	      g3 = ((k+frac_bv_int[b][2])-kk)/Kspace_grid3;

	      fprintf(fp, "%d %d  %d %d %d\n", ik+1,ikb+1,g1,g2,g3);

	      for (n=0; n<Num_KS_states_CWF; n++) {
                for (m=0; m<Num_KS_states_CWF; m++) {
		  fprintf(fp, " %22.14f %22.14f\n", Mbin[p].r, Mbin[p].i);
                  p++;
                } // m
	      } // n
	    } // b

	    /* fclose(fp_bin) */
	    fclose(fp_bin);

	  } // end of if ((fp_bin =...
	  else{
	    printf("Failure of reading %s\n",fname_bin);fflush(stdout);
            error_counter++;
	  }

	} // ik 

        /* fclose(fp) */
        fclose(fp);

      } // end of if ((fp = fopen(fname,"w"))
      else {
        printf("Failure of saving %s\n",fname);fflush(stdout);
        error_counter++;
      }

    } // spin 
  } // end of if (myid==Host_ID)

  /* freeing of array */
  free(Mbin);
}



void Generate_mmn_NC(dcomplex **OLPexp_1D, int ***k_op, int ***k2k, 
                     double *T_KGrids1, double *T_KGrids2, double *T_KGrids3, 
                     int *order_GA, int *MP)
{
  int m,n,ik,ika,ikb,i,j,k,ii,jj,kk,b,Tknum,p,M,N,K;
  int ib,jb,kb,iorb,jorb,idx,ig,jg,wanA,wanB,tnoA,tnoB,Rn,l1,l2,l3;
  int AN,GA_AN,Anum,Bnum,LB_AN,GB_AN,ik_s,ik_e,g1,g2,g3;
  double k1,k2,k3,b1,b2,b3,co,si,kRn;
  dcomplex *Ca,*Cb,*S,*WORK,*Mbin;
  dcomplex alpha = {1.0,0.0}; 
  dcomplex beta = {0.0,0.0};
  char fname_bin[500],fna[500],fnb[500];
  char fname[500];
  char buf[fp_bsize];
  FILE *fp,*fp_bin,*fpa,*fpb;
  int numprocs,myid;

  /* MPI */ 

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /*****************************************************
                  allocation of arrays
  *****************************************************/

  Ca = (dcomplex*)malloc(sizeof(dcomplex)*TNum_PAOs2*TNum_PAOs2);
  Cb = (dcomplex*)malloc(sizeof(dcomplex)*TNum_PAOs2*TNum_PAOs2);
  S = (dcomplex*)malloc(sizeof(dcomplex)*TNum_PAOs2*TNum_PAOs2);
  WORK = (dcomplex*)malloc(sizeof(dcomplex)*TNum_PAOs2*TNum_PAOs2);

  /* set Tknum */

  Tknum = Kspace_grid1*Kspace_grid2*Kspace_grid3;

  /*****************************************************
   division of Tknum by numprocs for MPI parallelization
  *****************************************************/
 
  ik_s = (int)(((long long)myid*(long long)Tknum)/(long long)numprocs);
  ik_e = (int)(((long long)(myid+1)*(long long)Tknum)/(long long)numprocs);

  /*****************************************************
                    calculation of mmn
  *****************************************************/

  for (ik=ik_s; ik<ik_e; ik++){

    if (myid==Host_ID){ 
      printf("  Calculating Mmnk %2d/%d\n",ik,ik_e-ik_s-1);fflush(stdout);
    }

    /* open fp_bin */ 

    sprintf(fname_bin,"%s%s_evec/mmnbin_%d",filepath,filename,ik);

    if ((fp_bin = fopen(fname_bin,"wb")) != NULL){

      /* get i, j, k */ 

      i = ik/(Kspace_grid2*Kspace_grid3);
      j = (ik-i*Kspace_grid2*Kspace_grid3)/Kspace_grid3;
      k = ik-i*Kspace_grid2*Kspace_grid3-j*Kspace_grid3;

      ika = k2k[i][j][k];

      k1 = T_KGrids1[ika];
      k2 = T_KGrids2[ika];
      k3 = T_KGrids3[ika];

      sprintf(fna,"%s%s_evec/evec_%d",filepath,filename,ika);

      if ((fpa = fopen(fna,"rb")) != NULL){

	/* read the evec file at k */
	fread(Ca,sizeof(dcomplex),Num_KS_states_CWF*TNum_PAOs2,fpa);

	// Ca^dag, Ca is in size of MaxN x N2
        for (p=0; p<Num_KS_states_CWF*TNum_PAOs2; p++) Ca[p].i = -Ca[p].i;

	for (b=0; b<tot_bvector; b++){

	  b1 = frac_bv[b][0];
	  b2 = frac_bv[b][1];
	  b3 = frac_bv[b][2];

	  ii = ((i+frac_bv_int[b][0])%Kspace_grid1 + Kspace_grid1)%Kspace_grid1;
	  jj = ((j+frac_bv_int[b][1])%Kspace_grid2 + Kspace_grid2)%Kspace_grid2;
	  kk = ((k+frac_bv_int[b][2])%Kspace_grid3 + Kspace_grid3)%Kspace_grid3;

	  ikb = k2k[ii][jj][kk];

	  sprintf(fnb,"%s%s_evec/evec_%d",filepath,filename,ikb);

	  if ((fpb = fopen(fnb,"rb")) != NULL){

	    /* read the evec file at k+b */
	    fread(Cb,sizeof(dcomplex),Num_KS_states_CWF*TNum_PAOs2,fpb);

	    /********************************************************
                                  calculation of S
	    ********************************************************/

	    for (p=0; p<TNum_PAOs2*TNum_PAOs2; p++) S[p] = Complex(0.0,0.0); 

	    idx = 0;
	    for (AN=1; AN<=atomnum; AN++){

	      GA_AN = order_GA[AN];
	      wanA = WhatSpecies[GA_AN];
	      tnoA = Spe_Total_CNO[wanA];
	      Anum = MP[GA_AN] - 1;

	      for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){

		GB_AN = natn[GA_AN][LB_AN];
		Rn = ncn[GA_AN][LB_AN];
		wanB = WhatSpecies[GB_AN];
		tnoB = Spe_Total_CNO[wanB];
		Bnum = MP[GB_AN] - 1;

		l1 = atv_ijk[Rn][1];
		l2 = atv_ijk[Rn][2];
		l3 = atv_ijk[Rn][3];

		kRn = (k1+b1)*(double)l1 + (k2+b2)*(double)l2 + (k3+b3)*(double)l3;
		si = sin(2.0*PI*kRn);
		co = cos(2.0*PI*kRn);

		for (iorb=0; iorb<tnoA; iorb++){

		  ig = Anum + iorb;

		  for (jorb=0; jorb<tnoB; jorb++){

		    jg = Bnum + jorb;

		    S[jg*TNum_PAOs2+ig].r += OLPexp_1D[b][idx].r*co - OLPexp_1D[b][idx].i*si;
		    S[jg*TNum_PAOs2+ig].i += OLPexp_1D[b][idx].r*si + OLPexp_1D[b][idx].i*co;

		    idx++;

		  } // jorb 
		} // iorb
	      } // LB_AN
	    } // AN

            /*********************************** 
               Now we have 
c               S = (S1 0)  
                   (0  0)

               So, copy S1 so that we have 
               S = (S1  0)  
                   (0  S1)
            ***********************************/

	    for (jorb=0; jorb<TNum_PAOs; jorb++){
	      for (iorb=0; iorb<TNum_PAOs; iorb++){
                S[(jorb+TNum_PAOs)*TNum_PAOs2+iorb+TNum_PAOs] = S[jorb*TNum_PAOs2+iorb];
	      }
	    }

	    /********************************************************
                multiplication of matrices:  
                Ca^{dag} S Cb, actual calculation:

                Ca (TNum_PAOs2 x Num_KS_states_CWF)
                S  (TNum_PAOs2 x TNum_PAOs2)                   
                Cb (TNum_PAOs2 x Num_KS_states_CWF)
	    ********************************************************/

	    M = Num_KS_states_CWF; N = TNum_PAOs2; K = TNum_PAOs2;
	    zgemm_( "T","N", &M, &N, &K, &alpha, Ca, &K, 
		    S, &K, &beta, WORK, &M );  // WORK (Num_KS_states_CWF x TNum_PAOs2)

	    M = Num_KS_states_CWF; N = Num_KS_states_CWF; K = TNum_PAOs2;
	    zgemm_( "N","N", &M, &N, &K, &alpha, WORK, &M, 
		    Cb, &K, &beta, S, &M ); // S (Num_KS_states_CWF x Num_KS_states_CWF)

	    /*
	    if (ik==1 && b==1){
	      for (p=0; p<Num_KS_states_CWF*Num_KS_states_CWF; p++){ 
		printf("ABC3 p=%2d %22.14f %22.14f\n",
		       p,S[p].r,S[p].i);
	      }
	    } 
	    */

	    /********************************************************
                        save data temporary in binary mode
	    ********************************************************/

	    fwrite(S,sizeof(dcomplex),(Num_KS_states_CWF*Num_KS_states_CWF),fp_bin);

	    /* fclose(fpb) */
	    fclose(fpb);

	  } // end of if
	  else{
	    printf("Failure of reading %s\n",fnb);fflush(stdout);
            error_counter++;
	  }

	} // b

	  /* fclose(fpa) */
	fclose(fpa);

      } // end of if
      else{
	printf("Failure of reading %s\n",fna);fflush(stdout);
        error_counter++;
      }

      /* fclose(fp_bin) */
      fclose(fp_bin);

    } // end of if ((fp_bin =...
    else{
      printf("Failure of saving %s\n",fname_bin);fflush(stdout);
      error_counter++;
    }

  } // ik

  /*****************************************************
       freeing of arrays and allocating an array
  *****************************************************/

  free(Ca);
  free(Cb);
  free(S);
  free(WORK);

  Mbin = (dcomplex*)malloc(sizeof(dcomplex)*tot_bvector*Num_KS_states_CWF*Num_KS_states_CWF);

  MPI_Barrier(mpi_comm_level1);

  /*****************************************************
           merge all the mmnbin files to .mmn 
  *****************************************************/
 
  if (myid==Host_ID){

    printf("\n");
    printf("  Merging all files to a single .mmn file.\n");fflush(stdout);
    
    sprintf(fname,"%s%s.mmn",filepath,filename);

    if ((fp = fopen(fname,"w")) != NULL){

      setvbuf(fp,buf,_IOFBF,fp_bsize);  /* setvbuf */

      fprintf(fp,"2nd line: BANDNUM, KPTNUM, NUMB, Nexts: KPTNUM x NUMBband elements block\n");
      fprintf(fp,"%d %d %d\n",Num_KS_states_CWF,Tknum,tot_bvector);

      for (ik=0; ik<Tknum; ik++){

	i = ik/(Kspace_grid2*Kspace_grid3);
	j = (ik-i*Kspace_grid2*Kspace_grid3)/Kspace_grid3;
	k = ik-i*Kspace_grid2*Kspace_grid3-j*Kspace_grid3;

	sprintf(fname_bin,"%s%s_evec/mmnbin_%d",filepath,filename,ik);

	if ((fp_bin = fopen(fname_bin,"rb")) != NULL){

	  fread(Mbin,sizeof(dcomplex),(tot_bvector*Num_KS_states_CWF*Num_KS_states_CWF),fp_bin);

	  p = 0;
	  for (b=0; b<tot_bvector; b++){

	    ii = ((i+frac_bv_int[b][0])%Kspace_grid1 + Kspace_grid1)%Kspace_grid1;
	    jj = ((j+frac_bv_int[b][1])%Kspace_grid2 + Kspace_grid2)%Kspace_grid2;
	    kk = ((k+frac_bv_int[b][2])%Kspace_grid3 + Kspace_grid3)%Kspace_grid3;

            ikb = ii*Kspace_grid2*Kspace_grid3 + jj*Kspace_grid3 + kk;

	    g1 = ((i+frac_bv_int[b][0])-ii)/Kspace_grid1; 
	    g2 = ((j+frac_bv_int[b][1])-jj)/Kspace_grid2; 
	    g3 = ((k+frac_bv_int[b][2])-kk)/Kspace_grid3;

	    fprintf(fp, "%d %d  %d %d %d\n", ik+1,ikb+1,g1,g2,g3);

	    for (n=0; n<Num_KS_states_CWF; n++) {
	      for (m=0; m<Num_KS_states_CWF; m++) {
		fprintf(fp, " %22.14f %22.14f\n", Mbin[p].r, Mbin[p].i);
		p++;
	      } // m
	    } // n
	  } // b

	    /* fclose(fp_bin) */
	  fclose(fp_bin);

	} // end of if ((fp_bin =...
	else{
	  printf("Failure of reading %s\n",fname_bin);fflush(stdout);
          error_counter++;
	}

      } // ik 

        /* fclose(fp) */
      fclose(fp);

    } // end of if ((fp = fopen(fname,"w"))
    else {
      printf("Failure of saving %s\n",fname);fflush(stdout);
      error_counter++;
    }

  } // end of if (myid==Host_ID)

  /* freeing of array */
  free(Mbin);
}





void Generate_amn_Col(int ***k_op, int ***k2k)
{
  int spin,m,n,ik,i,j,k,ii,jj,kk,ik_file,Tknum;
  double sign;
  char fname[500],fn_pm[500];
  char buf[fp_bsize];
  dcomplex *Amnk;
  FILE *fp,*fp_pm;
  int numprocs,myid;

  /* MPI */ 

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  if (myid==Host_ID){ 
    printf("  Making the .amn file\n");fflush(stdout);
  }

  /* set Tknum */

  Tknum = Kspace_grid1*Kspace_grid2*Kspace_grid3;

  /* allocation of array */

  Amnk = (dcomplex*)malloc(sizeof(dcomplex)*Num_KS_states_CWF*TNum_CWFs);

  /* generate .amn */
  
  for (spin=0; spin<=SpinP_switch; spin++){ 

    sprintf(fname,"%s%s_%d.amn",filepath,filename,spin);

    if ((fp = fopen(fname,"w")) != NULL){

      setvbuf(fp,buf,_IOFBF,fp_bsize);  /* setvbuf */

      fprintf(fp,"2nd line: BANDNUM, KPTNUM, WANNUM, Nexts: band, wan, k, ReAmn, ImAmn\n");
      fprintf(fp,"%d %d %d\n",Num_KS_states_CWF,Tknum,TNum_CWFs);

      ik = 0;
      for (i=0; i<Kspace_grid1; i++){
	for (j=0; j<Kspace_grid2; j++){
	  for (k=0; k<Kspace_grid3; k++){

            representative_ijk(i,j,k,Kspace_grid1,Kspace_grid2,Kspace_grid3,k_op,&ii,&jj,&kk);
            ik_file = k2k[ii][jj][kk];              

            if (k_op[i][j][k]==0){ sign =-1.0; }
            else {                 sign = 1.0; }

	    sprintf(fn_pm,"%s%s_evec/pm_%d_%d",filepath,filename,spin,ik_file);

	    if ((fp_pm = fopen(fn_pm,"rb")) != NULL){

	      /* read a pm file */
	      fread(Amnk,sizeof(dcomplex),Num_KS_states_CWF*TNum_CWFs,fp_pm);

	      /* save the .amn file */
	      for (n=0; n<TNum_CWFs; n++){
		for (m=0; m<Num_KS_states_CWF; m++){

		  fprintf(fp,"%5d %5d %5d %22.14f %22.14f\n",
			  m+1,n+1,ik+1,
  		          Amnk[m*TNum_CWFs+n].r,
			  sign*Amnk[m*TNum_CWFs+n].i);

		} // m
	      } // n 

	      /* fclose(fp_pm) */
	      fclose(fp_pm);
	    } // end of if 
	    else{
	      printf("Failure of reading %s\n",fn_pm);fflush(stdout);
              error_counter++;
	    }

            /* increment ik */

            ik++;
 
	  } // k 
	} // j
      } // i

      /* fclose(fp) */
      fclose(fp);
    }
    else {
      printf("Failure of saving %s\n",fname);fflush(stdout);
      error_counter++;
    }
  } // spin

  /* freeing of array */
  free(Amnk);
}



void Generate_amn_NC(int ***k_op, int ***k2k)
{
  int m,n,ik,i,j,k,ik_file,Tknum;
  char fname[500],fn_pm[500];
  char buf[fp_bsize];
  dcomplex *Amnk;
  FILE *fp,*fp_pm;

  /* set Tknum */

  Tknum = Kspace_grid1*Kspace_grid2*Kspace_grid3;

  /* allocation of array */

  Amnk = (dcomplex*)malloc(sizeof(dcomplex)*Num_KS_states_CWF*TNum_CWFs);

  /* generate .amn */
  
  sprintf(fname,"%s%s.amn",filepath,filename);

  if ((fp = fopen(fname,"w")) != NULL){

    setvbuf(fp,buf,_IOFBF,fp_bsize);  /* setvbuf */

    fprintf(fp,"2nd line: BANDNUM, KPTNUM, WANNUM, Nexts: band, wan, k, ReAmn, ImAmn\n");
    fprintf(fp,"%d %d %d\n",Num_KS_states_CWF,Tknum,TNum_CWFs);

    ik = 0;
    for (i=0; i<Kspace_grid1; i++){
      for (j=0; j<Kspace_grid2; j++){
	for (k=0; k<Kspace_grid3; k++){

	  ik_file = k2k[i][j][k];              

	  sprintf(fn_pm,"%s%s_evec/pm_%d",filepath,filename,ik_file);

	  if ((fp_pm = fopen(fn_pm,"rb")) != NULL){

	    /* read a pm file */
	    fread(Amnk,sizeof(dcomplex),Num_KS_states_CWF*TNum_CWFs,fp_pm);

	    /* save the .amn file */
	    for (n=0; n<TNum_CWFs; n++){
	      for (m=0; m<Num_KS_states_CWF; m++){

		fprintf(fp,"%5d %5d %5d %22.14f %22.14f\n",
			m+1,n+1,ik+1,
			Amnk[m*TNum_CWFs+n].r,
			Amnk[m*TNum_CWFs+n].i);

	      } // m
	    } // n 

	      /* fclose(fp_pm) */
	    fclose(fp_pm);
	  } // end of if 
	  else{
	    printf("Failure of reading %s\n",fn_pm);fflush(stdout);
            error_counter++;
	  }

	  /* increment ik */

	  ik++;
 
	} // k 
      } // j
    } // i

    /* fclose(fp) */
    fclose(fp);
  }
  else {
    printf("Failure of saving %s\n",fname);fflush(stdout);
    error_counter++;
  }

  /* freeing of array */
  free(Amnk);
}



void representative_ijk(int i0,int j0,int k0,
                        int Ni,int Nj,int Nk,
                        int ***k_op,
                        int *i,int *j,int *k)
{
  if (k_op[i0][j0][k0] == 0) {
    *i = mirror_index(i0, Ni);
    *j = mirror_index(j0, Nj);
    *k = mirror_index(k0, Nk);
  } 
  else {
    *i = i0;
    *j = j0;
    *k = k0;
  }
}


int mirror_index(int a, int N)
{
  return (a==0 || 2*a==N) ? a : (N - a);
}




void Calc_OLPexp( dcomplex *****OLPexp, dcomplex **OLPexp_1D, 
                  int *MP, int *order_GA, 
                  int *My_NZeros, int *SP_NZeros, int *SP_Atoms)
{
  double time0;
  int Mc_AN,Gc_AN,Mh_AN,h_AN,Gh_AN,b,tno0,tno1;
  int i,j,Cwan,Hwan,NO0,NO1,size_OLPexp;
  int TNO1,TNO2,wan1,wan2,num;
  int Rnh,spin,N,NumC[4];
  int Nc,GNc,GRc,Nog,Nh,MN,XC_P_switch;
  double x,y,z,mbdotr,co,si;
  double Cxyz[4],*Tmp_Vec;
  double *tmp_Orbs_Grid;
  dcomplex ctmp;
  dcomplex **ChiV;
  dcomplex *tmp_ChiV;
  dcomplex **tmp_OLPexp;
  int numprocs,myid,ID,tag=999;
  MPI_Status stat;
  MPI_Request request;

  /* MPI */

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /*****************************************************
                   allocation of arrays
  *****************************************************/

  Tmp_Vec = (double*)malloc(sizeof(double)*2*List_YOUSO[8]*List_YOUSO[7]*List_YOUSO[7]);

  ChiV = (dcomplex**)malloc(sizeof(dcomplex*)*List_YOUSO[7]);
  for (i=0; i<List_YOUSO[7]; i++){
    ChiV[i] = (dcomplex*)malloc(sizeof(dcomplex)*List_YOUSO[11]);
  }

  tmp_ChiV = (dcomplex*)malloc(sizeof(dcomplex)*List_YOUSO[7]);
  tmp_Orbs_Grid = (double*)malloc(sizeof(double)*List_YOUSO[7]);

  tmp_OLPexp = (dcomplex**)malloc(sizeof(dcomplex*)*List_YOUSO[7]);
  for (i=0; i<List_YOUSO[7]; i++){
    tmp_OLPexp[i] = (dcomplex*)malloc(sizeof(dcomplex)*List_YOUSO[7]);
  }

  /*****************************************************
               matrix elements for OLPexp
  *****************************************************/

  for (b=0; b<tot_bvector; b++){
    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

      Gc_AN = M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];
      NO0 = Spe_Total_CNO[Cwan];

      for (i=0; i<NO0; i++){
	for (Nc=0; Nc<GridN_Atom[Gc_AN]; Nc++){

	  GNc = GridListAtom[Mc_AN][Nc];
	  GRc = CellListAtom[Mc_AN][Nc];

	  Get_Grid_XYZ(GNc,Cxyz);
	  x = Cxyz[1] + atv[GRc][1];
	  y = Cxyz[2] + atv[GRc][2];
	  z = Cxyz[3] + atv[GRc][3];

	  mbdotr = -(bvector[b][0]*x + bvector[b][1]*y + bvector[b][2]*z);
	  co = cos(mbdotr);
	  si = sin(mbdotr);

	  ChiV[i][Nc].r = co*Orbs_Grid[Mc_AN][Nc][i];
	  ChiV[i][Nc].i = si*Orbs_Grid[Mc_AN][Nc][i];
	}
      }

      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	Gh_AN = natn[Gc_AN][h_AN];
	Mh_AN = F_G2M[Gh_AN];

	Rnh = ncn[Gc_AN][h_AN];
	Hwan = WhatSpecies[Gh_AN];
	NO1 = Spe_Total_CNO[Hwan];

	/* initialize */

	for (i=0; i<NO0; i++){
	  for (j=0; j<NO1; j++){
	    tmp_OLPexp[i][j] = Complex(0.0,0.0);
	  }
	}

	/* summation of non-zero elements */

	for (Nog=0; Nog<NumOLG[Mc_AN][h_AN]; Nog++){

	  Nc = GListTAtoms1[Mc_AN][h_AN][Nog];
	  Nh = GListTAtoms2[Mc_AN][h_AN][Nog];

	  /* store ChiV in tmp_ChiV */

	  for (i=0; i<NO0; i++){
	    tmp_ChiV[i] = ChiV[i][Nc];
	  }

	  /* store Orbs_Grid in tmp_Orbs_Grid */

	  if (G2ID[Gh_AN]==myid){
	    for (j=0; j<NO1; j++){
	      tmp_Orbs_Grid[j] = Orbs_Grid[Mh_AN][Nh][j];
	    }
	  }
	  else{
	    for (j=0; j<NO1; j++){
	      tmp_Orbs_Grid[j] = Orbs_Grid_FNAN[Mc_AN][h_AN][Nog][j];
	    }
	  }

	  /* integration */

	  for (i=0; i<NO0; i++){
	    ctmp = tmp_ChiV[i];
	    for (j=0; j<NO1; j++){
	      tmp_OLPexp[i][j].r += ctmp.r*tmp_Orbs_Grid[j];
	      tmp_OLPexp[i][j].i += ctmp.i*tmp_Orbs_Grid[j];
	    }
	  }

	} // Nog

	/* OLPexp */

	for (i=0; i<NO0; i++){
	  for (j=0; j<NO1; j++){

	    OLPexp[b][Mc_AN][h_AN][i][j].r = tmp_OLPexp[i][j].r*GridVol;
	    OLPexp[b][Mc_AN][h_AN][i][j].i = tmp_OLPexp[i][j].i*GridVol;
	  }
	}

      } // h_AN
    } // Mc_AN
  } // b

  /****************************************************
             store OLPexp in OLPexp_1D
  ****************************************************/

  for (b=0; b<tot_bvector; b++){

    size_OLPexp = Get_OneD_OLPexp_Col( 1, OLPexp[b], OLPexp_1D[b], MP, 
                                       order_GA, My_NZeros, SP_NZeros, SP_Atoms);  

  }

  /****************************************************
                    freeing of arrays:
  ****************************************************/

  free(Tmp_Vec);

  for (i=0; i<List_YOUSO[7]; i++){
    free(ChiV[i]);
  }
  free(ChiV);

  free(tmp_ChiV);
  free(tmp_Orbs_Grid);

  for (i=0; i<List_YOUSO[7]; i++){
    free(tmp_OLPexp[i]);
  }
  free(tmp_OLPexp);

}



int Get_OneD_OLPexp_Col(int set_flag, dcomplex ****RH, dcomplex *H1, int *MP, 
                        int *order_GA, int *My_NZeros, int *is1, int *is2)
{
  int i,j,k;
  int MA_AN,GA_AN,LB_AN,GB_AN,AN;
  int wanA,wanB,tnoA,tnoB,Anum,Bnum,NUM;
  int num,tnum,num_orbitals;
  int ID,myid,numprocs,tag=999;
  int *ie1;
  int *My_Matomnum;
  double sum;
  double Stime,Etime;
  MPI_Status stat;
  MPI_Request request;

  /* MPI */

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);
  MPI_Barrier(mpi_comm_level1);

  /* allocation of arrays */

  My_Matomnum = (int*)malloc(sizeof(int)*numprocs);
  ie1 = (int*)malloc(sizeof(int)*numprocs);

  /* find my total number of non-zero elements in myid */

  My_NZeros[myid] = 0;
  for (MA_AN=1; MA_AN<=Matomnum; MA_AN++){
    GA_AN = M2G[MA_AN];
    wanA = WhatSpecies[GA_AN];
    tnoA = Spe_Total_CNO[wanA];

    num = 0;      
    for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
      GB_AN = natn[GA_AN][LB_AN];
      wanB = WhatSpecies[GB_AN];
      tnoB = Spe_Total_CNO[wanB];
      num += tnoB;
    }

    My_NZeros[myid] += tnoA*num;
  }

  for (ID=0; ID<numprocs; ID++){
    MPI_Bcast(&My_NZeros[ID],1,MPI_INT,ID,mpi_comm_level1);
  }

  tnum = 0;
  for (ID=0; ID<numprocs; ID++){
    tnum += My_NZeros[ID];
  }  

  is1[0] = 0;
  ie1[0] = My_NZeros[0] - 1;
  
  for (ID=1; ID<numprocs; ID++){
    is1[ID] = ie1[ID-1] + 1;
    ie1[ID] = is1[ID] + My_NZeros[ID] - 1;
  }

  /* set is2 and order_GA */

  My_Matomnum[myid] = Matomnum;
  for (ID=0; ID<numprocs; ID++){
    MPI_Bcast(&My_Matomnum[ID],1,MPI_INT,ID,mpi_comm_level1);
  }

  is2[0] = 1;
  for (ID=1; ID<numprocs; ID++){
    is2[ID] = is2[ID-1] + My_Matomnum[ID-1];
  }
  
  for (MA_AN=1; MA_AN<=Matomnum; MA_AN++){
    order_GA[is2[myid]+MA_AN-1] = M2G[MA_AN];
  }

  for (ID=0; ID<numprocs; ID++){
    MPI_Bcast(&order_GA[is2[ID]],My_Matomnum[ID],MPI_INT,ID,mpi_comm_level1);
  }

  /* set MP */

  Anum = 1;
  for (i=1; i<=atomnum; i++){
    MP[i] = Anum;
    wanA = WhatSpecies[i];
    Anum += Spe_Total_CNO[wanA];
  }
  NUM = Anum - 1;

  /* set H1 */

  if (set_flag){

    for (i=0; i<tnum; i++) H1[i] = Complex(0.0,0.0);

    k = is1[myid];
    for (MA_AN=1; MA_AN<=Matomnum; MA_AN++){
      GA_AN = M2G[MA_AN];
      wanA = WhatSpecies[GA_AN];
      tnoA = Spe_Total_CNO[wanA];
      for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
	GB_AN = natn[GA_AN][LB_AN];
	wanB = WhatSpecies[GB_AN];
	tnoB = Spe_Total_CNO[wanB];
	for (i=0; i<tnoA; i++){
	  for (j=0; j<tnoB; j++){
	    H1[k] = RH[MA_AN][LB_AN][i][j]; 
            k++;
	  }
	}
      }
    }

    /* MPI H1 */
  
    dtime(&Stime);
    MPI_Allreduce( MPI_IN_PLACE, &H1[0], 2*tnum, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
    dtime(&Etime);

  } // end of if

  /* freeing of arrays */

  free(ie1);
  free(My_Matomnum);

  /* return the size of H1 */

  return tnum;
}






#pragma optimization_level 1
int Cal_Weight_of_Shell(double **klatt, int *M_s, int **bvector, int *num_shell, 
			double *wb, int *Reject_Shell, int *searched_shell){
  /*
    This subroutine will find the weight of shells for given k space defined by klatt[3][3].
    The weight and b vectors are defined in Eqn. 26 in arXiV:0708.0650.
    INPUT
    int *M_s;        b vectors number in each shell. 
    M_s[0] store the vector b number for first nearest neighbor  shell,
    M_s[1] store the vector b number for second nearest neighbor shell,
    and so on

    int **bvector;   b vectors found in each shell. Shells are ordered with ascending shell radius.
    int shell_num;   total shell number.                 
                     
    Return
    double *wb;      weight obtained for shell_num. wb[shell_num].               
    find_w           0 -> not found; 1->founded
  */

  int find_w; /* whether w is found or not? To control the while loop */    
  
  double *qvector; /* qvector is a vector of length six. Corresponding to the combination 
                      of Cartesian indices of two b vectors.
		      qvector[0] --> xx
		      qvector[1] --> yx
		      qvector[2] --> zx
		      qvector[3] --> yy 
		      qvector[4] --> zy
		      qvector[5] --> zz 
                   */      
  double **Amatrix; /* Amatrix[6][shell_num] is the A matrix. 
                       First dimension correspondes to qvector[6];
                       Second correspondes to number of shells. 
                    */
  double **copy_Amatrix;                  
  double **Umatrix; /* a 6x6 matrix for left singular vector of Amatrix[6][shell_num] */
  double **Dmatrix; /* The same dimention as that of Amatrix, 6 x (Shell number).
                       It first min(6, Shell number) x min(6, Shell number) are diagonal
                       with singular value on the diagonal, all other values are zero
                    */                    
  double *dsing; /* store the singular values, which are the diagonal part of Dmatrix. In our
                    program, we need to check these value to see whether one or more of the
                    singular values be close to zero. If it is, reject the new shell and turn
                    to the next one untill Eqn.26 is satisfied. */
  double **VTmatrix; /* a (Shell number)x(Shell number) matrix for right singular vector */
  char jobu, jobvt;
  
  int info;
  /*The following are parameters temporarily used for calculation. */  
  double *da, *du, *dvt;
   
  double sum;
  int i,j,k, reject_indx,flag;
  int current_shell, bvindx, shellindx, startbv, endbv, realshellindx;
  int shell_size, old_shell_size; /* This array stores the possible linear dependent shells. 0, no rejection; 1-rejected  */
  double bx,by,bz,*work;
  char c;
  int shell_num, ROW, COL, lda, lwork;
  
  shell_num=*num_shell;

  qvector=(double*)malloc(sizeof(double)*6);
  for(i=0; i<6; i++){
    qvector[i]=0.0;
  }
  qvector[0]=1.0;/*xx*/
  qvector[3]=1.0;/*yy*/
  qvector[5]=1.0;/*zz*/
  for(i=0;i<shell_num;i++){
    Reject_Shell[i]=0;
  }
  jobu='A';/* all the right singular vectors should be calculated and stored in Umatrix */
  jobvt='A';/* all the left singular vectors should be calculated and stored in Umatrix */
  
  find_w=0;
  current_shell=1;  
  shell_size=1;
  
  if(debug6==1){
    printf("Shell structure:\n");
    for(shellindx=0;shellindx<shell_num;shellindx++){
      printf("In shell %2d, there are %2d vectors.\n",shellindx+1,M_s[shellindx]);
    }
  }
  while(find_w==0&&current_shell<=shell_num){
    old_shell_size=shell_size;
    if(debug6==1){
      /*printf("Start shell_size=%2d, old_shell_size =%2d\n",shell_size,old_shell_size);*/
    }
    Amatrix=(double**)malloc(sizeof(double*)*6);
    for(i=0;i<6;i++){
      Amatrix[i]=(double*)malloc(sizeof(double)*shell_size);
    }
    
    copy_Amatrix=(double**)malloc(sizeof(double*)*6);
    for(i=0;i<6;i++){
      copy_Amatrix[i]=(double*)malloc(sizeof(double)*shell_size);
    }
    
    Dmatrix=(double**)malloc(sizeof(double*)*6);
    for(i=0;i<6;i++){
      Dmatrix[i]=(double*)malloc(sizeof(double)*shell_size);
    }
    
    Umatrix=(double**)malloc(sizeof(double*)*6);
    for(i=0;i<6;i++){
      Umatrix[i]=(double*)malloc(sizeof(double)*6);
    }
    
    VTmatrix=(double**)malloc(sizeof(double*)*shell_size);
    for(i=0;i<shell_size;i++){
      VTmatrix[i]=(double*)malloc(sizeof(double)*shell_size);
    }
    
    if(shell_size>6){
      dsing=(double*)malloc(sizeof(double)*6);
      for(i=0;i<6;i++){
	dsing[i]=0.0;
      }
    }else{
      dsing=(double*)malloc(sizeof(double)*shell_size);
      for(i=0;i<shell_size;i++){
	dsing[i]=0.0;
      }
    }
    
    for(i=0;i<shell_size;i++){
      wb[i]=0.0;
    }
    realshellindx=-1;
    for(shellindx=0;shellindx<current_shell;shellindx++){/* initialize A matrix*/
      if(Reject_Shell[shellindx]==1){
	printf("Shell %d is rejected.\n",shellindx+1);
	continue; /*Ignore this shell if it is linear dependent with existing shells */
      }
      /*The real index for effective shells in Amatrix. Since some shellindx will be discarded */
      realshellindx++; 
      if(debug6==1){
	/*printf("shellindx=%2d and realshellindx=%2d\n",shellindx,realshellindx);*/
      }
      for(i=0;i<6;i++){
	Amatrix[i][realshellindx]=0.0;
      }
      startbv=0;
      if(shellindx==0){
	startbv=0;
      }else{
	for(i=0;i<shellindx;i++){/* find the start index of b vectors in this shell */
	  startbv=startbv+M_s[i];
	}
      }
      if(debug6==1){
	/*printf("startbv=%2d, and M_s[%2d] is %2d. endbv is %2d.\n",startbv,shellindx,M_s[shellindx],M_s[shellindx]+startbv);*/
	printf("In shell %2d, there are %2d b vectors:\n",shellindx+1,M_s[shellindx]);
      }
      for(bvindx=startbv;bvindx<M_s[shellindx]+startbv;bvindx++){
	/* convert the b vectors into Cartesian coordinates */
	bx=(double)bvector[bvindx][0]*klatt[0][0]+(double)bvector[bvindx][1]*klatt[1][0]+(double)bvector[bvindx][2]*klatt[2][0];
	by=(double)bvector[bvindx][0]*klatt[0][1]+(double)bvector[bvindx][1]*klatt[1][1]+(double)bvector[bvindx][2]*klatt[2][1];
	bz=(double)bvector[bvindx][0]*klatt[0][2]+(double)bvector[bvindx][1]*klatt[1][2]+(double)bvector[bvindx][2]*klatt[2][2];
	Amatrix[0][realshellindx]=Amatrix[0][realshellindx]+bx*bx;/*xx*/
	Amatrix[1][realshellindx]=Amatrix[1][realshellindx]+by*bx;/*yx*/
	Amatrix[2][realshellindx]=Amatrix[2][realshellindx]+bz*bx;/*zx*/
	Amatrix[3][realshellindx]=Amatrix[3][realshellindx]+by*by;/*yy*/
	Amatrix[4][realshellindx]=Amatrix[4][realshellindx]+bz*by;/*zy*/
	Amatrix[5][realshellindx]=Amatrix[5][realshellindx]+bz*bz;/*zz*/
	if(debug6==1){
	  /*printf("b*b=%10.5f\n",bx*bx+by*by+bz*bz);*/
	  printf("bvector%2d (%2d,%2d,%2d) Cart(%10.5f,%10.5f,%10.5f) |b|=%10.5f\n",bvindx,bvector[bvindx][0],bvector[bvindx][1],bvector[bvindx][2],bx,by,bz,sqrt(fabs(bx*bx+by*by+bz*bz)));
	}
      }/*all the b vectors within each shell*/
      copy_Amatrix[0][realshellindx]=Amatrix[0][realshellindx];
      copy_Amatrix[1][realshellindx]=Amatrix[1][realshellindx];
      copy_Amatrix[2][realshellindx]=Amatrix[2][realshellindx];
      copy_Amatrix[3][realshellindx]=Amatrix[3][realshellindx];
      copy_Amatrix[4][realshellindx]=Amatrix[4][realshellindx];
      copy_Amatrix[5][realshellindx]=Amatrix[5][realshellindx];
    }/*Sum within each shell. A matrix is OK now*/
    if(debug6==1){
      printf("A matrix are:\n");
      for(i=0;i<6;i++){
	printf("%2d ",i+1);
	for(j=0;j<realshellindx+1;j++){
	  printf("%10.5f ",Amatrix[i][j]);
	}
	printf("\n");
      }
    }
    /*Amatrix should be factorised using a singular value decomposition to get U, D, VT matrix */
    /* To use dgesvd, we need to do some conversion of arrays */
    da=(double*)malloc(sizeof(double)*6*shell_size);
    du=(double*)malloc(sizeof(double)*6*6);
    dvt=(double*)malloc(sizeof(double)*shell_size*shell_size);
    lwork=shell_size*shell_size;
    if(6>shell_size){
      lwork=6*6;
    }

    work=(double*)malloc(sizeof(double)*lwork);

    /* convert from 2-D arrays to 1-D arrays */

    k=0;
    for(j=0;j<shell_size;j++){
      for(i=0;i<6;i++){
	da[k]=Amatrix[i][j];
	k++;
      }
    }

    k=0;
    for(j=0;j<6;j++){
      for(i=0;i<6;i++){
	du[k]=0.0;
	k++;
      }
    }

    k=0;
    for(j=0;j<shell_size;j++){
      for(i=0;i<shell_size;i++){
	dvt[k]=0.0;
	k++;
      }
    }

    /* Singular Value Decompositon  */
    /*    dgesvd(jobu, jobvt, 6, shell_size, da, 6, dsing, du, 6, dvt, shell_size, &info); */

    ROW=6;
    COL=shell_size;  	
    lda=6;
    F77_NAME(dgesvd,DGESVD)(&jobu, &jobvt, &ROW, &COL, da, &lda, dsing, du, &lda, dvt, &COL, work, &lwork, &info);

    if(info==0){
      /*printf("Factorising OK\n");*/
    }else if(info<0){
      printf("!!!!!!!!!!!!!!!!!!!ERROR!!!!!!!!!!!!!!!!!!!!\n");
      printf("! In subroutine Cal_Weight_of_Shell:       !\n");
      printf("! When calling dgesvd, the %2dth argument  !\n",-info);
      printf("! had an illegal value.                    !\n");
      printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
      error_counter++;
      return 0;
    }else{
      printf("!!!!!!!!!!!!!!!!!!!ERROR!!!!!!!!!!!!!!!!!!!!\n");
      printf("! In subroutine Cal_Weight_of_Shell:       !\n");
      printf("! Calling Dgesvd failed in convergence.    !\n");
      printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
      return 0;
    }
    if(debug6==1){
      printf("Singular value are:\n");
      if(shell_size<6){
	for(i=0;i<shell_size;i++){
	  printf("%10.5f\n",dsing[i]);
	}
      }else{
	for(i=0;i<6;i++){
	  printf("%10.5f\n",dsing[i]);
	}
      }
    }
    /* checking the Singular value. If one or more of them are close or equal to 0, 
       reject current shell and go to next shell.*/
    if(shell_size<6){
      for(i=0;i<shell_size;i++){
	if(fabs(dsing[i])<smallvalue){
	  Reject_Shell[current_shell-1]=1;
          if(debug6==1){
	    printf("current shell %d is to be rejected.\n",current_shell);
          }
	  break;
	}
      }
    }else{
      for(i=0;i<6;i++){
	if(fabs(dsing[i])<smallvalue){
	  Reject_Shell[current_shell-1]=1;
	  if(debug6==1){
	    printf("current shell %d is to be rejected.\n",current_shell);
          }
	  break;
	}
      }
    }
    if(Reject_Shell[current_shell-1]==0){/* if this shell is needed */
      /* asign values to U, VT and D and try to calculate w for these shells*/	
      for(j=0;j<shell_size;j++){
	for(i=0;i<6;i++){
	  Dmatrix[i][j]=0.0;
	}
      }
      if(shell_size<6){
	for(i=0;i<shell_size;i++){
	  Dmatrix[i][i]=1.0/dsing[i];
	}
      }else{
	for(i=0;i<6;i++){
	  Dmatrix[i][i]=1.0/dsing[i];
	}
      }
      k=0;
      for(j=0;j<6;j++){
	for(i=0;i<6;i++){
	  Umatrix[i][j]=du[k];
	  k++;
	}
      }
      if(debug6==1){
	printf("U matrix is:\n");
	for(i=0;i<6;i++){
	  printf("%10.5f %10.5f %10.5f %10.5f %10.5f %10.5f\n",Umatrix[i][0],Umatrix[i][1],Umatrix[i][2],Umatrix[i][3],Umatrix[i][4],Umatrix[i][5]);
	}
      }
      k=0;
      for(j=0;j<shell_size;j++){
	for(i=0;i<shell_size;i++){
	  VTmatrix[i][j]=dvt[k];
	  k++;
	}
      }
      if(debug6==1){
	printf("VT matrix is:\n");
	for(i=0;i<shell_size;i++){
	  for(j=0;j<shell_size;j++){
	    printf("%10.5f ",VTmatrix[i][j]);
	  }
	  printf("\n");
	}
      }
      /*Now we have U, VT and D.*/
      /*Calculate VxD^{-1} (NxN)x(NxM) and store the result into Amatrix*/
      for(i=0;i<shell_size;i++){
	for(j=0;j<6;j++){
	  sum=0.0;
	  for(k=0;k<shell_size;k++){
	    sum=sum+VTmatrix[k][i]*Dmatrix[j][k];
	  }
	  Amatrix[j][i]=sum;
	}
      }
      if(debug6==1){/* out put VxD^{-1} */
	printf("VxD^{-1} (shell_sizexshell_size)x(shell_sizex6) is:\n");
	for(i=0;i<shell_size;i++){
	  for(j=0;j<6;j++){
	    printf("%10.5f ",Amatrix[j][i]);
	  }
	  printf("\n");
	}
      }
      /* Calculate VxD^{-1}*UT (NxM)x(MxM) and store the result into Dmatrix */
      for(i=0;i<shell_size;i++){
	for(j=0;j<6;j++){
	  sum=0.0;
	  for(k=0;k<6;k++){
	    sum=sum+Amatrix[k][i]*Umatrix[j][k];
	  }
	  Dmatrix[j][i]=sum;
	}
      }
      if(debug6==1){/* out put VxD^{-1}*UT */
	printf("VxD^{-1}*UT (shell_sizex6)x(6x6) is:\n");
	for(i=0;i<shell_size;i++){
	  for(j=0;j<6;j++){
	    printf("%10.5f ",Dmatrix[j][i]);
	  }
	  printf("\n");
	}
      }
      /* Calculate VxD^{-1}*UT*q (NxM)x(Mx1) and store the result into wb */
      for(i=0;i<shell_size;i++){
	sum=0.0;
	for(k=0;k<6;k++){
	  sum=sum+Dmatrix[k][i]*qvector[k];
	}
	wb[i]=sum;
	if(debug6==1){
	  printf("wb %2d is %10.5f\n",i+1,wb[i]);
	}
      }

      /* After getting wb, Eqn. 26 should be checked to see whether these wb is ok */
      flag=0;
      for(j=0;j<6;j++){
	sum=0.0;
	for(i=0;i<shell_size;i++){
	  sum=sum+wb[i]*copy_Amatrix[j][i];
	  /*printf("amatrix[%2d][%2d]=%10.5f wb[%2d]=%10.5f",j,i,copy_Amatrix[j][i],i,wb[i]);*/
      	}
      	if(fabs(sum-qvector[j])>smallvalue){
	  flag=1;
	  if(debug6==1){
	    /*printf("sum=%10.5f, qvector=%10.5f ",sum,qvector[j]);*/
	    printf("Sorry, this weight is not acceptable.\n");
	  }
      	}
      }
      if(flag==0){/* wb is founded. */
	find_w=1;/*mark it and stop searching */
	*num_shell=shell_size;/* The real number of shells used */
	*searched_shell=current_shell;
	if(debug6==1){
	  printf("Haha! I find them.\n");
	  for(i=0;i<shell_size;i++){
	    printf("wb[%2d]=%10.5f\n",i+1,wb[i]);
	  }
	}
      }else{/* wb not found */
	shell_size++; /* No wb was found, adding one more shell */
      }
    }/* if current shell is not rejected, calculating w */
    if(debug6==1){
      /*printf("End shell_size=%2d, old_shell_size =%2d\n",shell_size,old_shell_size);fflush(0);*/
    }
    current_shell++; /* Going to the next shell */
    if(debug6==1){
      printf("Current Shell=%2d.\n",current_shell);
    }
    free(work);
    free(dvt);
    free(du);
    free(da);
    free(dsing);
    
    for(i=0;i<old_shell_size;i++){
      free(VTmatrix[i]);
    }
    free(VTmatrix);
    
    for(i=0;i<6;i++){
      free(Umatrix[i]);
    }
    free(Umatrix);
    
    for(i=0;i<6;i++){
      free(Dmatrix[i]);
    }
    free(Dmatrix);
       
    for(i=0;i<6;i++){
      free(copy_Amatrix[i]);
    }
    free(copy_Amatrix);    
    
    for(i=0;i<6;i++){
      free(Amatrix[i]);
    }
    free(Amatrix);    
  }/*While not found*/
  free(qvector);
  return find_w;
}/*Cal_Weight_of_Shell */



#pragma optimization_level 1
void Shell_Structure(double **klatt, int *M_s, int **bvector, int *shell_num, int MAXSHELL){
  /*
    This subroutine will analyse the shell structure of given k space defined by klatt[3][3].
    It will return
    int *M_s;               b vectors number in each shell. 
    M_s[0] store the vector b number for first nearest neighbor  shell,
    M_s[1] store the vector b number for second nearest neighbor shell,
    and so on.
    int **bvector;      Store the b vectors in an ascending shell radius 
    b[vector_indx][xyz]
    The first M_s[0] elements are b vectors from first shell;
    The next M_s[1] elements are those from 2nd nearest-neighbor shell,
    and so on.  
    int *shell_num;     The total number of shells determined in this subroutine.
  */
  int i1,i2,i3;/* combination in 3 k-basis vectors */                 
  int current_shell; /* current_shell number */
  double *distance, dx,dy,dz, *wb;
  int kindx, *ordering,i,j,k, tot_kpt, tot_shell, tot_vectors;
  int **combination, **ordered_com;
  char c;
    
  current_shell=0;
  kindx=0;
    
  combination=(int**)malloc(sizeof(int*)*(8*MAXSHELL*MAXSHELL*MAXSHELL));
  ordered_com=(int**)malloc(sizeof(int*)*(8*MAXSHELL*MAXSHELL*MAXSHELL));
  distance=(double*)malloc(sizeof(double)*(8*MAXSHELL*MAXSHELL*MAXSHELL));
  ordering=(int*)malloc(sizeof(int)*(8*MAXSHELL*MAXSHELL*MAXSHELL));
  for(i1=0;i1<(8*MAXSHELL*MAXSHELL*MAXSHELL);i1++){
    combination[i1]=(int*)malloc(sizeof(int)*3);
    ordered_com[i1]=(int*)malloc(sizeof(int)*3);
    distance[i1]=0.0;
  }
    
  for(i1=-MAXSHELL+1;i1<MAXSHELL;i1++){
    for(i2=-MAXSHELL+1;i2<MAXSHELL;i2++){
      for(i3=-MAXSHELL+1;i3<MAXSHELL;i3++){
	/*generate one k point */
        /* Its distance to Zero point. Here Zero point is also shifted by shift[3].
           And the k lattice basis is klatt, which has been divided by knum_i,knum_j and knum_k.
        */
	dx=(double)i1*klatt[0][0]+(double)i2*klatt[1][0]+(double)i3*klatt[2][0];
	dy=(double)i1*klatt[0][1]+(double)i2*klatt[1][1]+(double)i3*klatt[2][1];
	dz=(double)i1*klatt[0][2]+(double)i2*klatt[1][2]+(double)i3*klatt[2][2];
	distance[kindx]=sqrt(fabs(dx*dx+dy*dy+dz*dz));
	combination[kindx][0]=i1;
	combination[kindx][1]=i2;
	combination[kindx][2]=i3;
	ordering[kindx]=kindx;
	kindx++;
      }/*end i3*/      	
    }/*end i2*/      	
  }/*end i1*/      	
  /* Odering these k points by the ascending order of distance and classify them into shells */  
  tot_kpt=kindx;
  if(debug6==1){
    printf("total kpoint number is %2d\n",tot_kpt);
    printf("before ordering, distance is\n");
    for(kindx=0;kindx<tot_kpt;kindx++){
      i1=combination[kindx][0];
      i2=combination[kindx][1];
      i3=combination[kindx][2];
      dx=(double)i1*klatt[0][0]+(double)i2*klatt[1][0]+(double)i3*klatt[2][0];
      dy=(double)i1*klatt[0][1]+(double)i2*klatt[1][1]+(double)i3*klatt[2][1];
      dz=(double)i1*klatt[0][2]+(double)i2*klatt[1][2]+(double)i3*klatt[2][2];
      printf("(%2d,%2d,%2d)(%10.5f%10.5f%10.5f)%10.5f\n",combination[kindx][0],combination[kindx][1],combination[kindx][2],dx/BohrR,dy/BohrR,dz/BohrR,distance[kindx]/BohrR);
    }
  }

  Ascend_Ordering(distance, ordering, tot_kpt);
	  
  for(kindx=0;kindx<tot_kpt;kindx++){
    ordered_com[kindx][0]=combination[ordering[kindx]][0];
    ordered_com[kindx][1]=combination[ordering[kindx]][1];
    ordered_com[kindx][2]=combination[ordering[kindx]][2];
  }

  if(debug6==1){

    printf("After ordering, distance is\n");

    for(kindx=0;kindx<tot_kpt;kindx++){

      i1=ordered_com[kindx][0];
      i2=ordered_com[kindx][1];
      i3=ordered_com[kindx][2];
      dx=(double)i1*klatt[0][0]+(double)i2*klatt[1][0]+(double)i3*klatt[2][0];
      dy=(double)i1*klatt[0][1]+(double)i2*klatt[1][1]+(double)i3*klatt[2][1];
      dz=(double)i1*klatt[0][2]+(double)i2*klatt[1][2]+(double)i3*klatt[2][2];

      printf("(%2d,%2d,%2d) %10.5f%10.5f%10.5f %10.5f\n",
             ordered_com[kindx][0],ordered_com[kindx][1],ordered_com[kindx][2],
             dx/BohrR,dy/BohrR,dz/BohrR,distance[kindx]/BohrR);
    }
    /*    	c=getchar(); */
  }
    
  /* Now, classifying them into shells. Be careful, the first shell determined here 
     is k point itself, with radius=0.0. */

  i=0;dx=distance[0]; /*mark the first shell */
  current_shell=0;

  for(kindx=0;kindx<tot_kpt;kindx++){/*find the total shell number */
    if(fabs(dx-distance[kindx])>smallvalue){ /* Come to the next shell */
      ordering[current_shell]=i; /*Memorize the number of b vectors in last shell */
      i=1; /* re-count the b vectors in next shell */
      current_shell++; /* next shell index */
      dx=distance[kindx]; /* next shell radius */
    }else{/* still in the same shell */
      i++;/*count the number of b vectors in this shell */
    }
  }
  if(debug6==1){
    printf("totally, there are %2d shells.\n",current_shell-1);
    for(i=1;i<current_shell;i++){
      printf("%2d shell has %3d k points.\n",i,ordering[i]);
    }
  }
  if((current_shell-1)>MAXSHELL){
    tot_shell=MAXSHELL;/*some of the last shell maynot be fully included, discard them */
  }else{
    /*    	printf("Please increase the MAXSHELL parameter. Presently it is %3d\n",MAXSHELL); */
    tot_shell=0;
  }

  /* Now we can asign the values for M_s and bvector */

  tot_vectors=0;

  for(i=1;i<=tot_shell;i++){/* Get the total number of b vectors */
    M_s[i-1]=ordering[i];
    tot_vectors=tot_vectors+M_s[i-1];
  }
   
  for(i=0;i<tot_vectors;i++){
    for(j=0;j<3;j++){
      bvector[i][j]=ordered_com[i+1][j];
    }
  }

  if(debug6==1){
    kindx=0;
    for(i=0;i<tot_shell;i++){
      printf("In shell %2d, there are %2d b vectors.\n",i+1,M_s[i]);
      for(j=0;j<M_s[i];j++){
	printf("(%2d,%2d,%2d) %10.5f\n",bvector[kindx][0],bvector[kindx][1],bvector[kindx][2],distance[kindx+1]);
	kindx++;
      }
    }
  }

  *shell_num=tot_shell;
  for(i1=0;i1<(8*MAXSHELL*MAXSHELL*MAXSHELL);i1++){
    free(ordered_com[i1]);
    free(combination[i1]);
  }

  free(ordering);
  free(distance);
  free(ordered_com);
  free(combination);

}/*End of Shell_Structure subroutine*/



#pragma optimization_level 1
static void Ascend_Ordering(double *xyz_value, int *ordering, int tot_atom){
  int i,j,k, tmp_order,myid;
  double tmp_xyz;
	
  MPI_Comm_rank(mpi_comm_level1,&myid);

  if (1){
    qsort_double_int(tot_atom, xyz_value, ordering);
  }
  else{ 

    for (i=1;i<tot_atom; i++){/* taking one value */
      for (j=i;j>0;j--){ /* compare with all the other lower index value */
	if (xyz_value[j]<xyz_value[j-1]){/* if it is smaller than lower index value, exchange */
	  tmp_xyz=xyz_value[j];
	  xyz_value[j]=xyz_value[j-1];
	  xyz_value[j-1]=tmp_xyz;
	  tmp_order = ordering[j];
	  ordering[j]=ordering[j-1];
	  ordering[j-1]=tmp_order;
	}
      }
    }
  }

  return;
}




#pragma optimization_level 1
void Set_bvector( int knum_i, int knum_j, int knum_k, int MAXSHELL )
{
  int spin;
  int i,j,k,l;
  double b[4],ktp[4];
  double sumr;
  double **kg;
  double Cxyz[4];

  /* for file operation*/

  FILE *fpkpt;
  char fname[300];
  int kpt_num, ki, kj, kk;
  double *wb, *tmp_wb, wbtot;
  double dkx, dky, dkz, **klatt;
  int **kplusb, *M_s, shell_num, *tmp_M_s, **tmp_bvector, *Reject_Shell;

  /* bdirection[nbdir] gives the total number of bvectors without inversion symmetry 
     and gives the vector index in frac_bv matrix */

  int *bdirection, nbdir; 
  int bindx, nindx, startbv,find_w,tmp_shellnum,searched_shell;
   
  int r_num, *ndegen;
  int **rvect;
  double metric[3][3];
  int numprocs,myid;

  /* MPI initialize */

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /******************************************
              the standard output
  ******************************************/

  if (myid==Host_ID){
    printf("\n r-space primitive vector (Bohr)\n");
    printf("  tv1=%10.6f %10.6f %10.6f\n",tv[1][1], tv[1][2], tv[1][3]);
    printf("  tv2=%10.6f %10.6f %10.6f\n",tv[2][1], tv[2][2], tv[2][3]);
    printf("  tv3=%10.6f %10.6f %10.6f\n",tv[3][1], tv[3][2], tv[3][3]);
    printf(" k-space primitive vector (Bohr^-1)\n");
    printf("  rtv1=%10.6f %10.6f %10.6f\n",rtv[1][1], rtv[1][2], rtv[1][3]);
    printf("  rtv2=%10.6f %10.6f %10.6f\n",rtv[2][1], rtv[2][2], rtv[2][3]);
    printf("  rtv3=%10.6f %10.6f %10.6f\n\n",rtv[3][1], rtv[3][2], rtv[3][3]);
    rtv[0][0]=sqrt(rtv[1][1]*rtv[1][1]+rtv[1][2]*rtv[1][2]+rtv[1][3]*rtv[1][3]);
    rtv[0][1]=sqrt(rtv[2][1]*rtv[2][1]+rtv[2][2]*rtv[2][2]+rtv[2][3]*rtv[2][3]);
    rtv[0][2]=sqrt(rtv[3][1]*rtv[3][1]+rtv[3][2]*rtv[3][2]+rtv[3][3]*rtv[3][3]);
  }
  
  for (i=1; i<=3; i++){
    for (j=1; j<=3; j++){
      r_latt[i-1][j-1] = tv[i][j]; 
    }
  }

  /* before calculation, check Wannier.Kgrid
     Sometime some settings of Wannier.Kgrid will cause failure in finding
     proper Wigner-Seitz supercell for making interpolation. Avoid this if
     interpolation is needed. 

     metric is a matrix as following:
     a \dot a      a \dot b       a \dot c
     b \dot a      b \dot b       b \dot c
     c \dot a      c \dot b       c \dot c
  */

  for(j=0;j<3;j++){
    for(i=0;i<=j;i++){
      sumr=0.0;
      for(l=0;l<3;l++){
	sumr=sumr+tv[i+1][l+1]*tv[j+1][l+1];
      }
      metric[i][j]=sumr;
      if(i<j){
	metric[j][i]=metric[i][j];
      }
    }
  }

  r_num=1;
  rvect = (int**)malloc(sizeof(int*)*3);
  for(i=0; i<3; i++){
    rvect[i] = (int*)malloc(sizeof(int)*r_num);
  }

  ndegen = (int*)malloc(sizeof(int)*r_num);

  r_num=-1;
  Wigner_Seitz_Vectors(metric, rtv, knum_i, knum_j, knum_k, &r_num, rvect, ndegen);

  free(ndegen);

  for(i=0;i<3;i++){
    free(rvect[i]);
  }
  free(rvect);

  rvect = (int**)malloc(sizeof(int*)*3);
  for(i=0;i<3;i++){
    rvect[i] = (int*)malloc(sizeof(int)*r_num);
  }

  ndegen = (int*)malloc(sizeof(int)*r_num);

  r_num=0;
  Wigner_Seitz_Vectors(metric, rtv,  knum_i, knum_j, knum_k, &r_num, rvect, ndegen);

  /* Taking regular mesh */

  kpt_num=knum_i*knum_j*knum_k;

  kg = (double**)malloc(sizeof(double*)*kpt_num);
  for (i=0; i<kpt_num; i++){
    kg[i] = (double*)malloc(sizeof(double)*3);
  }
  
  /* MPI_Barrier(comm1); */

  kpt_num=0;

  for(i=knum_i;i>=1;i--){
    for(j=knum_j;j>=1;j--){
      for(k=knum_k;k>=1;k--){
	kg[kpt_num][0]=(double)(knum_i-i)/(double)knum_i+1E-12;
	kg[kpt_num][1]=(double)(knum_j-j)/(double)knum_j+1E-12;
	kg[kpt_num][2]=(double)(knum_k-k)/(double)knum_k+1E-12;
	kpt_num++;
      }
    }
  } 

  /* Find the bvectors */

  tmp_M_s=(int*)malloc(sizeof(int)*MAXSHELL);
  for(i=0;i<MAXSHELL;i++){
    tmp_M_s[i]=0;
  }
    	
  tmp_bvector=(int**)malloc(sizeof(int*)*(8*MAXSHELL*MAXSHELL*MAXSHELL));
  for(i=0;i<(8*MAXSHELL*MAXSHELL*MAXSHELL);i++){
    tmp_bvector[i]=(int*)malloc(sizeof(int)*3);
    for(j=0;j<3;j++){
      tmp_bvector[i][j]=0;
    }
  }

  klatt=(double**)malloc(sizeof(double*)*3);
  for(i=0;i<3;i++){
    klatt[i]=(double*)malloc(sizeof(double)*3);
  }

  klatt[0][0] = rtv[1][1]/(double)knum_i;
  klatt[0][1] = rtv[1][2]/(double)knum_i;
  klatt[0][2] = rtv[1][3]/(double)knum_i;
  klatt[1][0] = rtv[2][1]/(double)knum_j;
  klatt[1][1] = rtv[2][2]/(double)knum_j;
  klatt[1][2] = rtv[2][3]/(double)knum_j;
  klatt[2][0] = rtv[3][1]/(double)knum_k;
  klatt[2][1] = rtv[3][2]/(double)knum_k;
  klatt[2][2] = rtv[3][3]/(double)knum_k;

  /* Find the shell structure of k points */

  Shell_Structure(klatt, tmp_M_s, tmp_bvector, &shell_num, MAXSHELL);

  if(shell_num==0){

    if (myid==Host_ID){
        printf("******************************Error********************************\n");
        printf("*    Can not find proper b vectors, please increase parameter     *\n");
        printf("*    MAXSHELL OR change Wannier.Kgrids.                           *\n");
        printf("******************************Error********************************\n");
        printf("***********************************INFO**************************************\n");
        printf("Reciprocal Lattices lengths are:%10.6f %10.6f %10.6f\n",
               rtv[0][0],rtv[0][1],rtv[0][2]);
        printf("The ratio among them are: b1:b2=%10.6f b1:b3=%10.6f b2:b3=%10.6f\n",
               rtv[0][0]/rtv[0][1],rtv[0][0]/rtv[0][2],rtv[0][1]/rtv[0][2]);
        printf("Message: Please try to set Wannier.Kgrid has the similar ratio as above.\n");
        printf("************************************INFO*************************************\n");
    }

    error_counter++;

    for(i=0;i<3;i++){
      free(klatt[i]);
    }
    free(klatt);

    for(i=0;i<(8*MAXSHELL*MAXSHELL*MAXSHELL);i++){
      free(tmp_bvector[i]);
    }

    free(tmp_bvector);
    free(tmp_M_s);

    MPI_Finalize();
    exit(0);
  }

  tot_bvector=0;
  for(i=0;i<shell_num;i++){
    tot_bvector=tot_bvector+tmp_M_s[i];
  }
    	
  Reject_Shell=(int*)malloc(sizeof(int)*shell_num);
  for(i=0;i<shell_num;i++){
    Reject_Shell[i]=0;
  }
    	
  tmp_wb=(double*)malloc(sizeof(double)*shell_num);
  tmp_shellnum=shell_num;
  searched_shell=0;

  find_w=0;
  find_w = Cal_Weight_of_Shell( klatt, tmp_M_s, tmp_bvector, &shell_num, 
                                tmp_wb, Reject_Shell, &searched_shell );

  if(find_w==0){/* wb is not found within these shells */

    if (myid==Host_ID){
       printf("*************************** Error ****************************\n");
       printf("*    Weights for b vectors (totally %3d) are not found.      *\n",tmp_shellnum);
       printf("*    Please increase MAXSHELL (presently it is %3d) OR       *\n",MAXSHELL);
       printf("*    change Wannier.Kgrids and try again                     *\n");
       printf("*************************** Error ****************************\n");
       printf("***********************************INFO**************************************\n");
       printf("Reciprocal Lattices lengths are:%10.6f %10.6f %10.6f\n",
              rtv[0][0],rtv[0][1],rtv[0][2]);            
       printf("The ratio among them are: b1:b2=%10.6f b1:b3=%10.6f b2:b3=%10.6f\n",
              rtv[0][0]/rtv[0][1],rtv[0][0]/rtv[0][2],rtv[0][1]/rtv[0][2]);
       printf("Message: Please try to set Wannier.Kgrid has the similar ratio as above.\n");
       printf("************************************INFO*************************************\n");
    }

    error_counter++;

    free(tmp_wb);
    free(Reject_Shell);

    for(i=0;i<3;i++){
      free(klatt[i]);
    }
    free(klatt);

    for(i=0;i<(8*MAXSHELL*MAXSHELL*MAXSHELL);i++){
      free(tmp_bvector[i]);
    }

    free(tmp_bvector);
    free(tmp_M_s);
    MPI_Finalize();
    exit(0);
  }

  else{ /*success in finding wb */

    /* Rearrange M_s and bvector arrays to remove other unnecessary shells */

    tot_bvector=0;
    M_s=(int*)malloc(sizeof(int)*shell_num);

    j=0;
    for(i=0;i<searched_shell;i++){
      if(Reject_Shell[i]==0){
	M_s[j]=tmp_M_s[i];
	tot_bvector=tot_bvector+M_s[j];
	j++;
      }
    }

    wb=(double*)malloc(sizeof(double)*tot_bvector);

    k=0;
    for(i=0;i<shell_num;i++){
      for(j=0;j<M_s[i];j++){
	wb[k]=tmp_wb[i];
	k++;
      }
    }

    bvector = (double**)malloc(sizeof(double*)*tot_bvector);
    for(i=0; i<tot_bvector; i++){
      bvector[i]=(double*)malloc(sizeof(double)*3);
      for(j=0; j<3; j++){
	bvector[i][j] = 0.0;
      }
    }

    frac_bv_int = (int**)malloc(sizeof(int*)*tot_bvector);
    for(i=0; i<tot_bvector; i++){
      frac_bv_int[i]=(int*)malloc(sizeof(int)*3);
      for(j=0; j<3; j++){
	frac_bv_int[i][j] = 0;
      }
    }

    k=0; 
    for(i=0; i<searched_shell; i++){

      if(Reject_Shell[i]==0){

	startbv=0;

	if(i==0){
	  startbv = 0;
	}
        else{
	  for(j=0; j<i; j++){
	    startbv = startbv+tmp_M_s[j];
	  }
	}

	for(j=startbv; j<startbv+tmp_M_s[i]; j++){

          /* frac_bv_int is fractional bvector normalized by the mesh vector. */

          frac_bv_int[k][0] = tmp_bvector[j][0]; 
          frac_bv_int[k][1] = tmp_bvector[j][1]; 
          frac_bv_int[k][2] = tmp_bvector[j][2]; 

	  bvector[k][0] = (double)tmp_bvector[j][0]/(double)knum_i;
	  bvector[k][1] = (double)tmp_bvector[j][1]/(double)knum_j;
	  bvector[k][2] = (double)tmp_bvector[j][2]/(double)knum_k;
	  k++;
	}

      }/* not rejected shell */    	
    }
                
    /* Now free tmp_M_s, tmp_bvector, tmp_wb, Reject_shell */ 

    free(tmp_wb);
    free(Reject_Shell);
    for(i=0;i<3;i++){
      free(klatt[i]);
    }
    free(klatt);

    for(i=0;i<(8*MAXSHELL*MAXSHELL*MAXSHELL);i++){
      free(tmp_bvector[i]);
    }
    free(tmp_bvector);
    free(tmp_M_s);

    if (myid==Host_ID){
       printf("There are %2d shells and total number of b vectors is %3d\n",shell_num,tot_bvector);
       fflush(0);
    }

    frac_bv = (double**)malloc(sizeof(double*)*tot_bvector);
    for(i=0;i<tot_bvector;i++){
      frac_bv[i] = (double*)malloc(sizeof(double)*3);
    }
   
    tot_bvector=0;
    wbtot=0.0;

    for(i=0;i<shell_num;i++){

      if (myid==Host_ID){
        printf("Shell %2d has %2d b vectors:\n",i+1,M_s[i]);
        printf("No.|      Fractional Coordinate     || Cartesian Coordinate (Angs^-1)||Weight_b(Angs^2)||\n");
      } 

      for(j=0;j<M_s[i];j++){

        /* frac_bv is bvector in fractional coordinate. */

	frac_bv[tot_bvector][0] = bvector[tot_bvector][0];
	frac_bv[tot_bvector][1] = bvector[tot_bvector][1];
	frac_bv[tot_bvector][2] = bvector[tot_bvector][2];

	dkx = bvector[tot_bvector][0]*rtv[1][1]
            + bvector[tot_bvector][1]*rtv[2][1]
            + bvector[tot_bvector][2]*rtv[3][1];

	dky = bvector[tot_bvector][0]*rtv[1][2]
            + bvector[tot_bvector][1]*rtv[2][2]
            + bvector[tot_bvector][2]*rtv[3][2];

	dkz = bvector[tot_bvector][0]*rtv[1][3]
            + bvector[tot_bvector][1]*rtv[2][3]
            + bvector[tot_bvector][2]*rtv[3][3];

        if (myid==Host_ID){

           printf(" %2d|  (%8.5f,%8.5f,%8.5f)  ||  (%8.5f,%8.5f,%8.5f) || %13.5f  ||\n",
                    j+1,   
                    bvector[tot_bvector][0],
                    bvector[tot_bvector][1],
                    bvector[tot_bvector][2],
                    dkx/BohrR,dky/BohrR,dkz/BohrR,
                    wb[tot_bvector]*BohrR*BohrR); 
        }

        /* bvector is stored in Cartesian a.u.  */

	bvector[tot_bvector][0] = dkx;
	bvector[tot_bvector][1] = dky;
	bvector[tot_bvector][2] = dkz;

	wbtot = wbtot + wb[tot_bvector];

	tot_bvector++;

      }
    }
  }
  /* determine the b vectors list without inversion symmetry */

  bdirection=(int*)malloc(sizeof(int)*tot_bvector);
  for(i=0;i<tot_bvector;i++){
    bdirection[i]=-1;
  }
  nbdir=0;

  for(bindx=0;bindx<tot_bvector;bindx++){
    k=0;
    if(nbdir==0){ /* if the list still empty, put the first b into it */
      bdirection[nbdir]=bindx;
      nbdir++;
    }else{ /* compare the existing b in the list with all b vectors, to find new one */
      for(i=0;i<nbdir;i++){
        j=bdirection[i];
        if((frac_bv[bindx][0]+frac_bv[j][0])*(frac_bv[bindx][0]+frac_bv[j][0])+
           (frac_bv[bindx][1]+frac_bv[j][1])*(frac_bv[bindx][1]+frac_bv[j][1])+
           (frac_bv[bindx][2]+frac_bv[j][2])*(frac_bv[bindx][2]+frac_bv[j][2])<1.0e-8){
	  k=1;
        }
      }
      if(k==0){
        bdirection[nbdir]=bindx;
        nbdir++;
      }
    }
  }    

  for(i=0;i<nbdir;i++){
    k=bdirection[i];
    if (myid==Host_ID && 0==1){
      printf(" %2d|  (%8.5f,%8.5f,%8.5f)  ||  (%8.5f,%8.5f,%8.5f) || %10.5f  ||\n",
             i+1,frac_bv[k][0],frac_bv[k][1],frac_bv[k][2],bvector[k][0],
             bvector[k][1],bvector[k][2],wb[k]);

      fflush(0);
    }
  }

  kplusb=(int**)malloc(sizeof(int*)*kpt_num);
  for(k=0;k<kpt_num;k++){
    kplusb[k]=(int*)malloc(sizeof(int)*tot_bvector);
    for(bindx=0;bindx<tot_bvector;bindx++){

      /* k+b */
      ktp[1]=kg[k][0]+frac_bv[bindx][0];  
      ktp[2]=kg[k][1]+frac_bv[bindx][1];
      ktp[3]=kg[k][2]+frac_bv[bindx][2];

      for(i=1;i<=3;i++){
	b[i]=ktp[i];
	if(ktp[i]>=1.0){
	  b[i]=ktp[i]-1.0;
	}
	if(ktp[i]<0.0){
	  b[i]=ktp[i]+1.0;
	}
      }

      dkx=b[1]; dky=b[2]; dkz=b[3];
      kj=0; kk=-1;
      for(ki=0;ki<kpt_num;ki++){
	if(fabs(dkx-kg[ki][0])<smallvalue && fabs(dky-kg[ki][1])<smallvalue && fabs(dkz-kg[ki][2])<smallvalue){
	  kj=1;
	  kk=ki;
	  break;
	}
      }

      if(kj==0){
        if (myid==Host_ID){
   	  printf("***************************** Error *********************************\n");
	  printf("* Check Wannier.Kgrids, the equivalent k points for k+b not found.  *\n");
	  printf("***************************** Error *********************************\n");
          printf("***********************************INFO**************************************\n");
          printf("Reciprocal Lattices lengths are:%10.6f %10.6f %10.6f\n",rtv[0][0],rtv[0][1],rtv[0][2]);            printf("The ratio among them are: b1:b2=%10.6f b1:b3=%10.6f b2:b3=%10.6f\n",rtv[0][0]/rtv[0][1],rtv[0][0]/rtv[0][2],rtv[0][1]/rtv[0][2]);
          printf("Message: Please try to set Wannier.Kgrid has the similar ratio as above.\n");
          printf("************************************INFO*************************************\n");
        }

        error_counter++;

        MPI_Finalize();
        exit(0);	
      }
      kplusb[k][bindx]=kk;
    }
  }

  /**********************************************
                  freeing of arrays
  **********************************************/

  for(k=0; k<kpt_num; k++){
    free(kplusb[k]);
  }
  free(kplusb);

  free(bdirection);

  free(wb);
  free(M_s);

  for (i=0; i<kpt_num; i++){
    free(kg[i]);
  }
  free(kg);

  free(ndegen);

  for(i=0;i<3;i++){
    free(rvect[i]);
  }
  free(rvect);

}



#pragma optimization_level 1
void Wigner_Seitz_Vectors(double metric[3][3], double rtv[4][4], int knum_i, 
                         int knum_j, int knum_k, int *r_num, 
                         int **rvect, int *ndegen)

{
  int i1,i2,i3, n1, n2, n3, icnt, nd[3][125], i,j,rnum;
  double dist[125], dist_min, rdotk;
  int nosym;
 
  int myid;

  /* get MPI ID */
  MPI_Comm_rank(mpi_comm_level1,&myid);

  nosym=0;

  if(nosym==1 && *r_num==-1){
    *r_num=(knum_i)*(knum_j)*(knum_k);
    return;
  }

  if(nosym==1 && *r_num!=-1){
    rnum=0;
    for(n1=0;n1<knum_i;n1++){
      for(n2=0;n2<knum_j;n2++){
	for(n3=0;n3<knum_k;n3++){
	  rvect[0][rnum]=n1;
	  rvect[1][rnum]=n2;
	  rvect[2][rnum]=n3;
	  ndegen[rnum]=1;
	  rnum++;
	}
      }
    } 
    *r_num=rnum;
    return;
  } 
  rnum=0;
  for(n1=-knum_i;n1<knum_i+1;n1++){
    for(n2=-knum_j;n2<knum_j+1;n2++){
      for(n3=-knum_k;n3<knum_k+1;n3++){
	icnt = 0;
	for(i1=-2;i1<3;i1++){
	  for(i2=-2;i2<3;i2++){
	    for(i3=-2;i3<3;i3++){
	      nd[0][icnt]=n1-i1*knum_i;        
	      nd[1][icnt]=n2-i2*knum_j;        
	      nd[2][icnt]=n3-i3*knum_k;       
	      dist[icnt]=0.0;
	      for(i=0;i<3;i++){
		for(j=0;j<3;j++){
		  dist[icnt]=dist[icnt]+(double)(nd[i][icnt])*metric[i][j]*(double)(nd[j][icnt]);
		}
	      } 
	      icnt++;
	    }/* i3 */       
	  }/* i2 */
	} /* i1 */ 
	dist_min=dist[0];
	for(i=0;i<icnt;i++){
	  if(dist_min>dist[i]){
	    dist_min=dist[i];
	  }
	}
	if(fabs(dist[62]-dist_min)<0.0000001){
	  if(*r_num!=-1){
            ndegen[rnum]=0;
	    for(i=0;i<125;i++){
	      if(fabs(dist[i]-dist_min)<0.0000001){
	        ndegen[rnum]++;
		/*
		  if(rnum<1){
                  rvect[0][rnum]=nd[0][i];
                  rvect[1][rnum]=nd[1][i];
                  rvect[2][rnum]=nd[2][i];
                  ndegen[rnum]=1;
                  rnum++;
		  }else{
                  icnt=0;
                  for(j=0;j<rnum;j++){
		  if(rvect[0][j]==nd[0][i] && rvect[1][j]==nd[1][i] && rvect[2][j]==nd[2][i]){
		  icnt=1;
		  break;
		  }
                  }
                  if(icnt==0){
		  rvect[0][rnum]=nd[0][i];
		  rvect[1][rnum]=nd[1][i];
		  rvect[2][rnum]=nd[2][i];
		  ndegen[rnum]=1;   
		  rnum++;
                  }
		  }
		*/
	      }
	    }
            rvect[0][rnum]=n1;
            rvect[1][rnum]=n2;
            rvect[2][rnum]=n3;
	  }
          rnum++;
	} 
      } /* n3 */
    }/* n2 */
  }/* n1 */

  if(*r_num==-1){
    *r_num=rnum;
    return;
  }

  *r_num=rnum; 

  if (myid==Host_ID){
     printf("There are %i lattice points found in Wigner-Seitz supercell.\n",rnum);
  }
/*
  if(rnum>0){
    for(i=0;i<rnum;i++){
      printf("vector (%i,%i,%i) degeneracy: %i\n",rvect[0][i],rvect[1][i],rvect[2][i],ndegen[i]); 
    }
  }
*/
  /* check sum rule  */
  dist_min=0.0;
  for(i=0;i<rnum;i++){
    dist_min=dist_min+1.0/(double)ndegen[i];
  }
  if(fabs(dist_min-(double)knum_i*knum_j*knum_k)>smallvalue){
    if (myid==Host_ID){
      printf("**************************** Error **********************\n");
      printf("*   In Wigner_Seitz_Vectors subroutine, error happens.  *\n");
      printf("*   Please change setting of Wannier.Kgrid.             *\n");
      printf("**************************** Error **********************\n");
      printf("***********************************INFO**************************************\n");
      printf("Reciprocal Lattices lengths are:%10.6f %10.6f %10.6f\n",rtv[0][0],rtv[0][1],rtv[0][2]);            printf("The ratio among them are: b1:b2=%10.6f b1:b3=%10.6f b2:b3=%10.6f\n",rtv[0][0]/rtv[0][1],rtv[0][0]/rtv[0][2],rtv[0][1]/rtv[0][2]);
      printf("Message: Please try to set Wannier.Kgrid has the similar ratio as above.\n");
      printf("************************************INFO*************************************\n");
    }

    error_counter++;

    MPI_Finalize();
    exit(0);
  }
}/* Wigner_Seitz_Vectors */

